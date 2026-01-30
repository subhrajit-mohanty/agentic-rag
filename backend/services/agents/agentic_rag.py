"""
Agentic RAG Service

Production-grade agentic RAG orchestration using a state-machine approach.
Implements intelligent query handling with guardrails, retrieval, grading,
query rewriting, and answer generation.
"""

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from backend.core.config import get_settings
from backend.services.llm.client import BaseLLMClient, get_llm_client
from backend.services.vector_store.service import VectorStoreService, vector_store
from backend.services.cache.redis_cache import CacheManager, cache_manager

from .models import (
    AgentState,
    AgenticAskResponse,
    GradeDocuments,
    GradingResult,
    GuardrailScoring,
    SourceItem,
)
from .prompts import (
    GENERATE_ANSWER_PROMPT,
    GRADE_DOCUMENTS_PROMPT,
    GUARDRAIL_PROMPT,
    OUT_OF_SCOPE_PROMPT,
    REWRITE_PROMPT,
    get_persona_prompt,
)

logger = logging.getLogger(__name__)


class AgenticRAGService:
    """
    Production Agentic RAG orchestration service.
    
    Implements a reasoning loop that intelligently handles user queries through
    specialized processing nodes:
    
    1. GUARDRAIL: Validates query scope against allowed domains
    2. RETRIEVE: Performs hybrid search to gather context
    3. GRADE: Evaluates relevance of retrieved documents
    4. REWRITE: Optimizes query if results are poor (conditional)
    5. GENERATE: Synthesizes final answer with citations
    6. OUT_OF_SCOPE: Handles off-topic queries gracefully
    """
    
    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        vector_store_service: Optional[VectorStoreService] = None,
        cache_client: Optional[CacheManager] = None
    ):
        settings = get_settings()
        
        self.llm = llm_client or get_llm_client()
        self.vector_store = vector_store_service or vector_store
        self.cache = cache_client or cache_manager
        
        self.max_retrieval_attempts = settings.agent.max_retrieval_attempts
        self.guardrail_threshold = settings.agent.guardrail_threshold
        self.top_k = settings.agent.top_k_results
        self.use_hybrid = settings.agent.use_hybrid_search
        
        logger.info(
            f"AgenticRAGService initialized: "
            f"max_attempts={self.max_retrieval_attempts}, "
            f"guardrail_threshold={self.guardrail_threshold}"
        )
    
    async def _guardrail_node(self, state: AgentState) -> Dict[str, Any]:
        """Validate query scope against allowed enterprise domains."""
        query = state["original_query"]
        step = "Node: guardrail (validating query scope)"
        
        logger.info(f"Guardrail: Evaluating query '{query[:50]}...'")
        
        try:
            result = await self.llm.generate_structured(
                prompt=GUARDRAIL_PROMPT.format(query=query),
                response_model=GuardrailScoring
            )
            logger.info(f"Guardrail score: {result.score} - {result.reason}")
            
        except Exception as e:
            logger.warning(f"Guardrail LLM call failed: {e}, using heuristic")
            keywords = ['leave', 'pto', 'policy', 's3', 'aws', 'project', 'benefits', 
                       'how', 'what', 'where', 'help', 'access', 'employee']
            score = 85 if any(kw in query.lower() for kw in keywords) else 30
            result = GuardrailScoring(
                score=score,
                reason="Evaluated using keyword heuristic (LLM unavailable)",
                domains_matched=[]
            )
        
        routing = "continue" if result.score >= self.guardrail_threshold else "out_of_scope"
        
        return {
            "guardrail_result": result,
            "routing_decision": routing,
            "reasoning_steps": [
                step,
                f"Guardrail: Score {result.score}/100 - {result.reason}"
            ]
        }
    
    async def _retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        """Perform hybrid retrieval from the knowledge base."""
        query = state.get("rewritten_query") or state["original_query"]
        attempts = state.get("retrieval_attempts", 0) + 1
        step = f"Node: retrieve (attempt {attempts})"
        
        logger.info(f"Retrieve: Searching for '{query[:50]}...' (attempt {attempts})")
        
        try:
            results = await self.vector_store.search(
                query=query,
                limit=self.top_k,
                use_hybrid=self.use_hybrid
            )
            
            context_parts = []
            sources = []
            
            for idx, doc in enumerate(results):
                context_parts.append(f"[{doc['document_id']}]: {doc['content']}")
                sources.append(SourceItem(
                    document_id=doc["document_id"],
                    title=doc["title"],
                    url=doc.get("metadata", {}).get("url"),
                    relevance_score=doc.get("score", 0.0),
                    chunk_text=doc["content"][:200],
                    metadata=doc.get("metadata", {})
                ))
            
            context_str = "\n\n".join(context_parts) if context_parts else ""
            logger.info(f"Retrieved {len(results)} documents")
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            context_str = ""
            sources = []
        
        return {
            "retrieval_attempts": attempts,
            "sources": sources,
            "retrieved_context": context_str,
            "metadata": {
                **state.get("metadata", {}),
                "last_retrieved_context": context_str
            },
            "reasoning_steps": [step, f"Retrieved {len(sources)} documents"]
        }
    
    async def _grade_documents_node(self, state: AgentState) -> Dict[str, Any]:
        """Evaluate relevance of retrieved documents using LLM."""
        step = "Node: grade_documents (evaluating relevance)"
        context = state.get("retrieved_context", "")
        question = state["original_query"]
        
        logger.info("Grading document relevance...")
        
        if not context or len(context.strip()) < 50:
            logger.warning("Insufficient context for grading")
            return {
                "routing_decision": "rewrite",
                "grading_results": [],
                "reasoning_steps": [step, "Grading: Insufficient context, will rewrite query"]
            }
        
        try:
            result = await self.llm.generate_structured(
                prompt=GRADE_DOCUMENTS_PROMPT.format(
                    context=context[:4000],
                    question=question
                ),
                response_model=GradeDocuments
            )
            is_relevant = result.binary_score == "yes"
            logger.info(f"Grading result: {result.binary_score} - {result.reasoning}")
            
        except Exception as e:
            logger.warning(f"Grading LLM call failed: {e}, using heuristic")
            is_relevant = len(context.strip()) > 100
            result = GradeDocuments(
                binary_score="yes" if is_relevant else "no",
                reasoning="Evaluated using content length heuristic",
                confidence=0.5
            )
        
        if state["retrieval_attempts"] >= self.max_retrieval_attempts:
            routing = "generate"
            step_msg = "Grading: Max attempts reached, proceeding to generate"
        else:
            routing = "generate" if is_relevant else "rewrite"
            step_msg = f"Grading: {'Relevant' if is_relevant else 'Not relevant'} - {result.reasoning}"
        
        grading_result = GradingResult(
            document_id="context_batch",
            is_relevant=is_relevant,
            score=result.confidence,
            reasoning=result.reasoning
        )
        
        return {
            "routing_decision": routing,
            "grading_results": [grading_result],
            "reasoning_steps": [step, step_msg]
        }
    
    async def _rewrite_query_node(self, state: AgentState) -> Dict[str, Any]:
        """Optimize query for better retrieval results."""
        step = "Node: rewrite_query (optimizing search)"
        query = state["original_query"]
        
        logger.info(f"Rewriting query: '{query[:50]}...'")
        
        try:
            rewritten = await self.llm.generate(
                prompt=REWRITE_PROMPT.format(query=query)
            )
            rewritten = rewritten.strip()
            
            if not rewritten or rewritten.lower() == query.lower():
                rewritten = f"detailed information about {query}"
            
            logger.info(f"Rewritten to: '{rewritten[:50]}...'")
            
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            rewritten = f"enterprise {query} policy documentation"
        
        return {
            "rewritten_query": rewritten,
            "reasoning_steps": [step, f"Rewritten: '{rewritten[:100]}'"]
        }
    
    async def _generate_answer_node(self, state: AgentState) -> Dict[str, Any]:
        """Generate final answer using retrieved context."""
        step = "Node: generate_answer (synthesizing response)"
        context = state.get("retrieved_context", "No relevant documents found.")
        question = state["original_query"]
        
        logger.info("Generating answer...")
        
        try:
            answer = await self.llm.generate(
                prompt=GENERATE_ANSWER_PROMPT.format(
                    context=context[:6000],
                    question=question
                )
            )
            logger.info(f"Generated answer: {len(answer)} characters")
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            answer = (
                "I apologize, but I encountered an error generating the answer. "
                "Please try again or rephrase your question."
            )
        
        return {
            "answer": answer,
            "reasoning_steps": [step]
        }
    
    async def _out_of_scope_node(self, state: AgentState) -> Dict[str, Any]:
        """Handle queries outside allowed domains."""
        step = "Node: out_of_scope (query outside domain)"
        query = state["original_query"]
        
        logger.info(f"Out of scope: '{query[:50]}...'")
        
        try:
            answer = await self.llm.generate(
                prompt=OUT_OF_SCOPE_PROMPT.format(query=query)
            )
        except Exception:
            answer = (
                "I apologize, but this query is outside my supported domain parameters. "
                "I can help with HR policies, technical infrastructure, legal/compliance matters, "
                "and internal project documentation. Please rephrase your question."
            )
        
        return {
            "answer": answer,
            "reasoning_steps": [step]
        }
    
    def _build_response(
        self,
        state: AgentState,
        start_time: float,
        cache_hit: bool = False
    ) -> AgenticAskResponse:
        """Build the final API response from agent state."""
        execution_time = round(time.time() - start_time, 3)
        
        sources = state.get("sources", [])
        if sources:
            sources = sorted(sources, key=lambda x: x.relevance_score, reverse=True)[:5]
        
        guardrail_result = state.get("guardrail_result")
        
        return AgenticAskResponse(
            query=state["original_query"],
            answer=state.get("answer", "No answer generated."),
            sources=sources,
            chunks_used=len(sources),
            search_mode="hybrid" if self.use_hybrid else "bm25",
            reasoning_steps=state.get("reasoning_steps", []),
            retrieval_attempts=state.get("retrieval_attempts", 0),
            execution_time=execution_time,
            cache_hit=cache_hit,
            guardrail_score=guardrail_result.score if guardrail_result else None
        )
    
    async def ask(
        self,
        query: str,
        framework: str = "LangGraph",
        persona_id: Optional[str] = None,
        use_cache: bool = True
    ) -> AgenticAskResponse:
        """
        Execute the full agentic RAG workflow.
        
        Args:
            query: User query
            framework: Agent framework identifier
            persona_id: Optional persona for specialized behavior
            use_cache: Whether to use response caching
            
        Returns:
            Complete response with answer, sources, and reasoning trace
        """
        start_time = time.time()
        query_id = str(uuid.uuid4())[:8]
        
        logger.info(f"[{query_id}] Processing query: '{query[:50]}...'")
        
        # Check cache first
        if use_cache:
            cache_key = self.cache.generate_query_key(
                query=query,
                framework=framework,
                persona_id=persona_id
            )
            cached = await self.cache.get(cache_key)
            if cached:
                logger.info(f"[{query_id}] Cache hit!")
                cached["cache_hit"] = True
                cached["execution_time"] = round(time.time() - start_time, 3)
                return AgenticAskResponse.model_validate(cached)
        
        # Initialize agent state
        state: AgentState = {
            "messages": [],
            "original_query": query,
            "rewritten_query": None,
            "retrieval_attempts": 0,
            "retrieved_context": None,
            "guardrail_result": None,
            "routing_decision": None,
            "sources": [],
            "relevant_sources": [],
            "grading_results": [],
            "metadata": {
                "query_id": query_id,
                "framework": framework,
                "persona_id": persona_id
            },
            "answer": None,
            "reasoning_steps": []
        }
        
        try:
            # Step 1: Guardrail
            state.update(await self._guardrail_node(state))
            
            if state["routing_decision"] == "out_of_scope":
                state.update(await self._out_of_scope_node(state))
                return self._build_response(state, start_time)
            
            # Step 2: Retrieval Loop
            while state["retrieval_attempts"] < self.max_retrieval_attempts:
                state.update(await self._retrieve_node(state))
                state.update(await self._grade_documents_node(state))
                
                if state["routing_decision"] == "generate":
                    break
                else:
                    state.update(await self._rewrite_query_node(state))
            
            # Step 3: Generate Answer
            state.update(await self._generate_answer_node(state))
            
            response = self._build_response(state, start_time)
            
            # Cache successful response
            if use_cache and response.answer:
                await self.cache.set(cache_key, response.model_dump())
            
            logger.info(
                f"[{query_id}] Completed in {response.execution_time}s "
                f"({response.retrieval_attempts} retrieval attempts)"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"[{query_id}] Error: {e}")
            state["answer"] = f"An error occurred while processing your query: {str(e)}"
            state["reasoning_steps"].append(f"Error: {str(e)}")
            return self._build_response(state, start_time)


# Factory function
def make_agentic_rag_service(
    llm_client: Optional[BaseLLMClient] = None,
    vector_store_service: Optional[VectorStoreService] = None,
    cache_client: Optional[CacheManager] = None
) -> AgenticRAGService:
    """Factory function to create AgenticRAGService instance."""
    return AgenticRAGService(
        llm_client=llm_client,
        vector_store_service=vector_store_service,
        cache_client=cache_client
    )


# Global service instance
_agentic_service: Optional[AgenticRAGService] = None


def get_agentic_service() -> AgenticRAGService:
    """Get or create the global AgenticRAGService instance."""
    global _agentic_service
    if _agentic_service is None:
        _agentic_service = make_agentic_rag_service()
    return _agentic_service
