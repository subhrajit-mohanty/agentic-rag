"""
Specialized Agents for Enterprise Agentic RAG

Implements the core agents for the multi-agent system:
- PlannerAgent: Plans execution strategy and coordinates other agents
- ResearcherAgent: Analyzes queries and identifies information needs
- RetrieverAgent: Handles document retrieval with multiple strategies
- VerifierAgent: Validates answers and checks facts
- ResponderAgent: Generates final responses with citations
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .base import (
    BaseAgent, AgentContext, AgentMessage, AgentState,
    MessageType, MessagePriority
)

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models for Structured Outputs
# =============================================================================

class ExecutionPlan(BaseModel):
    """Structured execution plan from planner agent."""
    plan_id: str
    steps: List[Dict[str, Any]]
    required_agents: List[str]
    required_tools: List[str]
    estimated_iterations: int
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class QueryAnalysis(BaseModel):
    """Structured query analysis from researcher agent."""
    query_type: str
    intent: str
    entities: List[str]
    keywords: List[str]
    required_knowledge_domains: List[str]
    complexity: str  # simple, moderate, complex
    suggested_retrieval_strategy: str
    follow_up_questions: List[str] = []


class RetrievalResult(BaseModel):
    """Structured retrieval result from retriever agent."""
    documents: List[Dict[str, Any]]
    retrieval_strategy: str
    total_found: int
    relevance_scores: List[float]
    metadata: Dict[str, Any] = {}


class VerificationResult(BaseModel):
    """Structured verification result from verifier agent."""
    is_verified: bool
    confidence: float = Field(ge=0.0, le=1.0)
    issues: List[str] = []
    suggestions: List[str] = []
    citation_check: Dict[str, bool] = {}
    fact_check_results: List[Dict[str, Any]] = []
    hallucination_score: float = Field(ge=0.0, le=1.0, default=0.0)


class FinalResponse(BaseModel):
    """Structured final response from responder agent."""
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning_trace: List[str]
    metadata: Dict[str, Any] = {}


# =============================================================================
# Planner Agent
# =============================================================================

class PlannerAgent(BaseAgent):
    """
    Plans execution strategy and coordinates other agents.
    
    Responsibilities:
    - Analyze query complexity
    - Determine required agents and tools
    - Create execution plan
    - Adapt plan based on feedback
    - Decide when to terminate
    """
    
    def __init__(self, llm_client: Any, **kwargs):
        super().__init__(
            agent_id="planner",
            name="Planner Agent",
            description="Plans execution strategy and coordinates agent workflow",
            capabilities=[
                "query_analysis",
                "execution_planning",
                "agent_coordination",
                "adaptive_replanning"
            ],
            llm_client=llm_client,
            system_prompt=self._get_system_prompt(),
            **kwargs
        )
    
    def _get_system_prompt(self) -> str:
        return """You are a Planner Agent in an enterprise RAG system.

Your role is to analyze queries and create execution plans that coordinate multiple specialized agents.

Available agents:
- researcher: Analyzes queries, identifies information needs, extracts entities
- retriever: Retrieves relevant documents using various strategies
- verifier: Validates answers, checks facts, detects hallucinations
- responder: Generates final responses with proper citations

Available tools:
- search: Vector/hybrid search on knowledge base
- web_search: Search the web for current information
- calculator: Perform calculations
- code_executor: Execute code safely
- database: Query structured databases

For each query, you must:
1. Assess complexity (simple, moderate, complex)
2. Identify required agents and tools
3. Create a step-by-step execution plan
4. Estimate number of iterations needed

Output your plan as structured JSON."""
    
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: AgentContext
    ) -> Dict[str, Any]:
        """Create execution plan for the query."""
        self.state = AgentState.PROCESSING
        
        query = input_data.get("query", context.original_query)
        previous_results = input_data.get("previous_results", {})
        replan = input_data.get("replan", False)
        
        # Build prompt
        prompt = self._build_planning_prompt(query, previous_results, replan, context)
        
        try:
            # Generate plan using LLM
            if self.llm_client:
                plan_json = await self.llm_client.generate_structured(
                    prompt=prompt,
                    response_model=ExecutionPlan,
                    system_prompt=self.system_prompt
                )
                plan = plan_json.model_dump()
            else:
                # Fallback heuristic planning
                plan = self._heuristic_plan(query, context)
            
            # Store plan in context
            context.agent_outputs["planner"] = plan
            
            logger.info(f"Planner created plan with {len(plan.get('steps', []))} steps")
            
            self.state = AgentState.IDLE
            return {
                "status": "success",
                "plan": plan,
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            logger.error(f"Planner execution failed: {e}")
            self.state = AgentState.ERROR
            return {
                "status": "error",
                "error": str(e),
                "plan": self._heuristic_plan(query, context)
            }
    
    def _build_planning_prompt(
        self,
        query: str,
        previous_results: Dict[str, Any],
        replan: bool,
        context: AgentContext
    ) -> str:
        prompt = f"""Query: {query}

Context:
- Iteration: {context.iteration}
- Max iterations: {context.max_iterations}
- Documents retrieved so far: {len(context.retrieved_documents)}

{"Previous results that need replanning:" + json.dumps(previous_results, indent=2) if replan else ""}

Create an execution plan. Consider:
1. What type of query is this? (factual, procedural, analytical, creative)
2. What information sources are needed?
3. Should we search the knowledge base, web, or both?
4. What verification is needed?

Return a JSON execution plan."""
        
        return prompt
    
    def _heuristic_plan(self, query: str, context: AgentContext) -> Dict[str, Any]:
        """Create a plan using heuristics when LLM is unavailable."""
        query_lower = query.lower()
        
        # Determine complexity
        if len(query.split()) < 10 and "?" in query:
            complexity = "simple"
            iterations = 2
        elif any(word in query_lower for word in ["compare", "analyze", "explain"]):
            complexity = "complex"
            iterations = 4
        else:
            complexity = "moderate"
            iterations = 3
        
        # Determine required agents
        agents = ["researcher", "retriever", "responder"]
        tools = ["search"]
        
        if any(word in query_lower for word in ["calculate", "compute", "math"]):
            tools.append("calculator")
        
        if any(word in query_lower for word in ["current", "latest", "today", "now"]):
            tools.append("web_search")
        
        if complexity in ["moderate", "complex"]:
            agents.insert(-1, "verifier")
        
        return {
            "plan_id": f"plan_{context.query_id[:8]}",
            "steps": [
                {"agent": "researcher", "action": "analyze_query"},
                {"agent": "retriever", "action": "retrieve_documents"},
                {"agent": "verifier", "action": "verify_relevance"} if "verifier" in agents else None,
                {"agent": "responder", "action": "generate_response"}
            ],
            "required_agents": agents,
            "required_tools": tools,
            "estimated_iterations": iterations,
            "confidence": 0.7,
            "reasoning": f"Heuristic plan for {complexity} query"
        }
    
    async def reflect(self, context: AgentContext) -> Dict[str, Any]:
        """Reflect on plan effectiveness."""
        plan = context.agent_outputs.get("planner", {})
        
        # Check if plan is working
        if context.iteration > plan.get("estimated_iterations", 3):
            return {
                "agent_id": self.agent_id,
                "reflection": "Plan taking longer than expected, may need replanning",
                "confidence": 0.6,
                "suggest_replan": True
            }
        
        return {
            "agent_id": self.agent_id,
            "reflection": "Plan executing as expected",
            "confidence": 0.8,
            "suggest_replan": False
        }


# =============================================================================
# Researcher Agent
# =============================================================================

class ResearcherAgent(BaseAgent):
    """
    Analyzes queries and identifies information needs.
    
    Responsibilities:
    - Query classification
    - Intent detection
    - Entity extraction
    - Knowledge domain identification
    - Query decomposition for complex queries
    """
    
    def __init__(self, llm_client: Any, **kwargs):
        super().__init__(
            agent_id="researcher",
            name="Researcher Agent",
            description="Analyzes queries and identifies information needs",
            capabilities=[
                "query_classification",
                "intent_detection",
                "entity_extraction",
                "query_decomposition",
                "knowledge_domain_identification"
            ],
            llm_client=llm_client,
            system_prompt=self._get_system_prompt(),
            **kwargs
        )
    
    def _get_system_prompt(self) -> str:
        return """You are a Researcher Agent in an enterprise RAG system.

Your role is to deeply analyze user queries to understand:
1. Query type (definition, how-to, comparison, factual, opinion, calculation, code, policy, troubleshooting)
2. User intent (what they really want to know)
3. Key entities and concepts
4. Required knowledge domains
5. Complexity level

For complex queries, decompose them into simpler sub-queries.

Provide your analysis as structured JSON."""
    
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: AgentContext
    ) -> Dict[str, Any]:
        """Analyze the query and identify information needs."""
        self.state = AgentState.PROCESSING
        
        query = input_data.get("query", context.original_query)
        
        try:
            if self.llm_client:
                analysis = await self.llm_client.generate_structured(
                    prompt=f"Analyze this query: {query}",
                    response_model=QueryAnalysis,
                    system_prompt=self.system_prompt
                )
                result = analysis.model_dump()
            else:
                result = self._heuristic_analysis(query)
            
            context.agent_outputs["researcher"] = result
            
            logger.info(f"Researcher analyzed query: type={result.get('query_type')}, "
                       f"complexity={result.get('complexity')}")
            
            self.state = AgentState.IDLE
            return {
                "status": "success",
                "analysis": result,
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            logger.error(f"Researcher execution failed: {e}")
            self.state = AgentState.ERROR
            return {
                "status": "error",
                "error": str(e),
                "analysis": self._heuristic_analysis(query)
            }
    
    def _heuristic_analysis(self, query: str) -> Dict[str, Any]:
        """Analyze query using heuristics."""
        query_lower = query.lower()
        words = query.split()
        
        # Detect query type
        if query_lower.startswith(("what is", "define", "meaning of")):
            query_type = "definition"
        elif query_lower.startswith(("how to", "how do", "steps to")):
            query_type = "how_to"
        elif "compare" in query_lower or "vs" in query_lower or "difference" in query_lower:
            query_type = "comparison"
        elif any(word in query_lower for word in ["calculate", "compute", "sum", "total"]):
            query_type = "calculation"
        elif "policy" in query_lower or "rule" in query_lower:
            query_type = "policy"
        else:
            query_type = "factual"
        
        # Detect intent
        intent = "information_seeking"
        if "help" in query_lower or "fix" in query_lower:
            intent = "troubleshooting"
        elif "create" in query_lower or "make" in query_lower:
            intent = "task_completion"
        
        # Extract entities (simple approach)
        entities = [w for w in words if w[0].isupper() and len(w) > 1]
        
        # Keywords
        keywords = [w.lower() for w in words if len(w) > 3 and w.lower() not in 
                   ["what", "how", "the", "this", "that", "with", "from"]]
        
        # Complexity
        if len(words) < 8:
            complexity = "simple"
        elif len(words) < 20:
            complexity = "moderate"
        else:
            complexity = "complex"
        
        return {
            "query_type": query_type,
            "intent": intent,
            "entities": entities[:5],
            "keywords": keywords[:10],
            "required_knowledge_domains": ["general"],
            "complexity": complexity,
            "suggested_retrieval_strategy": "hybrid",
            "follow_up_questions": []
        }


# =============================================================================
# Retriever Agent
# =============================================================================

class RetrieverAgent(BaseAgent):
    """
    Handles document retrieval with multiple strategies.
    
    Responsibilities:
    - Execute search queries
    - Apply retrieval strategies (vector, BM25, hybrid)
    - Handle query expansion
    - Re-rank results
    - Manage multi-source retrieval
    """
    
    def __init__(
        self,
        llm_client: Any,
        vector_store: Any = None,
        reranker: Any = None,
        **kwargs
    ):
        super().__init__(
            agent_id="retriever",
            name="Retriever Agent",
            description="Retrieves relevant documents using multiple strategies",
            capabilities=[
                "vector_search",
                "hybrid_search",
                "query_expansion",
                "reranking",
                "multi_source_retrieval"
            ],
            llm_client=llm_client,
            tools=["search", "web_search"],
            **kwargs
        )
        self.vector_store = vector_store
        self.reranker = reranker
    
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: AgentContext
    ) -> Dict[str, Any]:
        """Execute retrieval based on strategy."""
        self.state = AgentState.PROCESSING
        
        query = input_data.get("query", context.original_query)
        strategy = input_data.get("strategy", "hybrid")
        limit = input_data.get("limit", 10)
        use_rerank = input_data.get("use_rerank", True)
        expand_query = input_data.get("expand_query", False)
        
        try:
            # Query expansion if requested
            if expand_query and self.llm_client:
                query = await self._expand_query(query)
            
            # Execute retrieval
            documents = await self._retrieve(query, strategy, limit)
            
            # Re-rank if enabled and available
            if use_rerank and self.reranker and documents:
                documents = await self._rerank(query, documents)
            
            # Store in context
            context.retrieved_documents.extend(documents)
            
            result = {
                "documents": documents,
                "retrieval_strategy": strategy,
                "total_found": len(documents),
                "relevance_scores": [d.get("score", 0) for d in documents],
                "metadata": {
                    "query_expanded": expand_query,
                    "reranked": use_rerank and self.reranker is not None
                }
            }
            
            context.agent_outputs["retriever"] = result
            
            logger.info(f"Retriever found {len(documents)} documents using {strategy}")
            
            self.state = AgentState.IDLE
            return {
                "status": "success",
                "retrieval": result,
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            logger.error(f"Retriever execution failed: {e}")
            self.state = AgentState.ERROR
            return {
                "status": "error",
                "error": str(e),
                "retrieval": {"documents": [], "total_found": 0}
            }
    
    async def _expand_query(self, query: str) -> str:
        """Expand query with related terms."""
        prompt = f"""Expand this search query with related terms and synonyms.
Keep the expanded query concise (max 50 words).

Original query: {query}

Expanded query:"""
        
        try:
            expanded = await self.llm_client.generate(prompt)
            return expanded.strip()
        except Exception:
            return query
    
    async def _retrieve(
        self,
        query: str,
        strategy: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """Execute retrieval strategy."""
        if self.vector_store is None:
            logger.warning("No vector store available, returning empty results")
            return []
        
        use_hybrid = strategy in ["hybrid", "combined"]
        
        results = await self.vector_store.search(
            query=query,
            limit=limit,
            use_hybrid=use_hybrid
        )
        
        return results
    
    async def _rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Re-rank documents using the reranker."""
        try:
            reranked = await self.reranker.rerank(
                query=query,
                documents=documents
            )
            return reranked
        except Exception as e:
            logger.warning(f"Reranking failed: {e}, using original order")
            return documents


# =============================================================================
# Verifier Agent
# =============================================================================

class VerifierAgent(BaseAgent):
    """
    Validates answers and checks facts.
    
    Responsibilities:
    - Verify answer accuracy
    - Check citations
    - Detect hallucinations
    - Validate consistency
    - Fact-check against sources
    """
    
    def __init__(self, llm_client: Any, **kwargs):
        super().__init__(
            agent_id="verifier",
            name="Verifier Agent",
            description="Validates answers, checks facts, and detects hallucinations",
            capabilities=[
                "answer_verification",
                "citation_checking",
                "hallucination_detection",
                "consistency_validation",
                "fact_checking"
            ],
            llm_client=llm_client,
            system_prompt=self._get_system_prompt(),
            **kwargs
        )
    
    def _get_system_prompt(self) -> str:
        return """You are a Verifier Agent in an enterprise RAG system.

Your role is to validate answers and ensure accuracy:

1. Citation Verification: Check that each citation accurately references the source
2. Fact Checking: Verify facts against provided context
3. Hallucination Detection: Identify any claims not supported by sources
4. Consistency Check: Ensure the answer is internally consistent
5. Completeness: Check if the answer fully addresses the query

Be strict but fair. Flag issues with specific explanations.

Output your verification as structured JSON."""
    
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: AgentContext
    ) -> Dict[str, Any]:
        """Verify the answer and check facts."""
        self.state = AgentState.PROCESSING
        
        answer = input_data.get("answer", "")
        query = input_data.get("query", context.original_query)
        sources = input_data.get("sources", context.retrieved_documents)
        
        try:
            if self.llm_client:
                verification = await self._verify_with_llm(answer, query, sources)
            else:
                verification = self._heuristic_verification(answer, sources)
            
            context.agent_outputs["verifier"] = verification
            
            logger.info(f"Verifier: verified={verification.get('is_verified')}, "
                       f"confidence={verification.get('confidence')}")
            
            self.state = AgentState.IDLE
            return {
                "status": "success",
                "verification": verification,
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            logger.error(f"Verifier execution failed: {e}")
            self.state = AgentState.ERROR
            return {
                "status": "error",
                "error": str(e),
                "verification": {"is_verified": False, "confidence": 0.0}
            }
    
    async def _verify_with_llm(
        self,
        answer: str,
        query: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify answer using LLM."""
        sources_text = "\n\n".join([
            f"[{s.get('document_id', 'unknown')}]: {s.get('content', '')[:500]}"
            for s in sources[:5]
        ])
        
        prompt = f"""Verify this answer against the sources.

Query: {query}

Answer to verify:
{answer}

Available sources:
{sources_text}

Check:
1. Are all facts in the answer supported by sources?
2. Are citations accurate?
3. Is there any hallucination (claims not in sources)?
4. Is the answer consistent and complete?

Provide verification results as JSON."""
        
        result = await self.llm_client.generate_structured(
            prompt=prompt,
            response_model=VerificationResult,
            system_prompt=self.system_prompt
        )
        
        return result.model_dump()
    
    def _heuristic_verification(
        self,
        answer: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verify using heuristics."""
        # Simple heuristic: check if answer contains content from sources
        source_content = " ".join([s.get("content", "") for s in sources]).lower()
        answer_words = set(answer.lower().split())
        source_words = set(source_content.split())
        
        overlap = len(answer_words & source_words) / max(len(answer_words), 1)
        
        # Check for citations
        has_citations = "[" in answer and "]" in answer
        
        return {
            "is_verified": overlap > 0.3,
            "confidence": min(overlap + 0.2, 1.0),
            "issues": [] if overlap > 0.3 else ["Low source overlap"],
            "suggestions": ["Add citations"] if not has_citations else [],
            "citation_check": {},
            "fact_check_results": [],
            "hallucination_score": max(0, 1 - overlap - 0.2)
        }


# =============================================================================
# Responder Agent
# =============================================================================

class ResponderAgent(BaseAgent):
    """
    Generates final responses with citations.
    
    Responsibilities:
    - Synthesize information from multiple sources
    - Generate coherent, well-structured responses
    - Add proper citations
    - Maintain appropriate tone
    - Handle uncertainty gracefully
    """
    
    def __init__(self, llm_client: Any, **kwargs):
        super().__init__(
            agent_id="responder",
            name="Responder Agent",
            description="Generates final responses with proper citations",
            capabilities=[
                "response_generation",
                "citation_formatting",
                "tone_adaptation",
                "uncertainty_handling"
            ],
            llm_client=llm_client,
            system_prompt=self._get_system_prompt(),
            **kwargs
        )
    
    def _get_system_prompt(self) -> str:
        return """You are a Responder Agent in an enterprise RAG system.

Your role is to generate final, polished responses:

1. Synthesize information from provided context
2. Write clear, professional responses
3. Include citations in format [doc_id] for every fact
4. Acknowledge uncertainty when appropriate
5. Be concise but complete

Guidelines:
- Only use information from provided sources
- If sources are insufficient, say so
- Match tone to query type (formal for policy, helpful for how-to)
- Structure complex answers with clear organization

Always cite your sources."""
    
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: AgentContext
    ) -> Dict[str, Any]:
        """Generate the final response."""
        self.state = AgentState.PROCESSING
        
        query = input_data.get("query", context.original_query)
        documents = input_data.get("documents", context.retrieved_documents)
        analysis = input_data.get("analysis", context.agent_outputs.get("researcher", {}))
        verification = input_data.get("verification", context.agent_outputs.get("verifier", {}))
        
        try:
            if self.llm_client:
                response = await self._generate_with_llm(
                    query, documents, analysis, verification, context
                )
            else:
                response = self._generate_heuristic(query, documents)
            
            context.agent_outputs["responder"] = response
            
            logger.info(f"Responder generated answer: {len(response.get('answer', ''))} chars")
            
            self.state = AgentState.IDLE
            return {
                "status": "success",
                "response": response,
                "agent_id": self.agent_id
            }
            
        except Exception as e:
            logger.error(f"Responder execution failed: {e}")
            self.state = AgentState.ERROR
            return {
                "status": "error",
                "error": str(e),
                "response": self._generate_heuristic(query, documents)
            }
    
    async def _generate_with_llm(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        analysis: Dict[str, Any],
        verification: Dict[str, Any],
        context: AgentContext
    ) -> Dict[str, Any]:
        """Generate response using LLM."""
        # Build context from documents
        context_parts = []
        for doc in documents[:5]:
            doc_id = doc.get("document_id", "unknown")
            content = doc.get("content", "")[:1000]
            context_parts.append(f"[{doc_id}]: {content}")
        
        context_text = "\n\n".join(context_parts)
        
        prompt = f"""Generate a response to this query using the provided context.

Query: {query}

Query Analysis:
- Type: {analysis.get('query_type', 'unknown')}
- Complexity: {analysis.get('complexity', 'unknown')}

Context (cite using [doc_id]):
{context_text}

Generate a comprehensive, well-cited response."""
        
        answer = await self.llm_client.generate(
            prompt=prompt,
            system_prompt=self.system_prompt
        )
        
        # Extract citations
        citations = []
        for doc in documents[:5]:
            doc_id = doc.get("document_id", "")
            if doc_id and f"[{doc_id}]" in answer:
                citations.append({
                    "document_id": doc_id,
                    "title": doc.get("title", ""),
                    "relevance_score": doc.get("score", 0.0)
                })
        
        # Build reasoning trace from context
        reasoning_trace = [
            f"Query type: {analysis.get('query_type', 'unknown')}",
            f"Retrieved {len(documents)} documents",
            f"Used {len(citations)} sources in response"
        ]
        
        if verification:
            reasoning_trace.append(
                f"Verification: {'passed' if verification.get('is_verified') else 'flagged issues'}"
            )
        
        return {
            "answer": answer,
            "citations": citations,
            "confidence": 0.85 if citations else 0.5,
            "reasoning_trace": reasoning_trace,
            "metadata": {
                "query_type": analysis.get("query_type"),
                "sources_used": len(citations)
            }
        }
    
    def _generate_heuristic(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate response using heuristics."""
        if not documents:
            return {
                "answer": "I don't have enough information to answer this question. "
                         "Please try rephrasing or provide more context.",
                "citations": [],
                "confidence": 0.0,
                "reasoning_trace": ["No documents found"],
                "metadata": {}
            }
        
        # Combine top document content
        top_doc = documents[0]
        answer = f"Based on the available information: {top_doc.get('content', '')[:500]}"
        
        if top_doc.get("document_id"):
            answer += f" [{top_doc['document_id']}]"
        
        return {
            "answer": answer,
            "citations": [{
                "document_id": top_doc.get("document_id", "unknown"),
                "title": top_doc.get("title", ""),
                "relevance_score": top_doc.get("score", 0.0)
            }],
            "confidence": 0.6,
            "reasoning_trace": ["Heuristic response generation"],
            "metadata": {}
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_planner_agent(llm_client: Any = None) -> PlannerAgent:
    """Create a planner agent."""
    return PlannerAgent(llm_client=llm_client)


def create_researcher_agent(llm_client: Any = None) -> ResearcherAgent:
    """Create a researcher agent."""
    return ResearcherAgent(llm_client=llm_client)


def create_retriever_agent(
    llm_client: Any = None,
    vector_store: Any = None,
    reranker: Any = None
) -> RetrieverAgent:
    """Create a retriever agent."""
    return RetrieverAgent(
        llm_client=llm_client,
        vector_store=vector_store,
        reranker=reranker
    )


def create_verifier_agent(llm_client: Any = None) -> VerifierAgent:
    """Create a verifier agent."""
    return VerifierAgent(llm_client=llm_client)


def create_responder_agent(llm_client: Any = None) -> ResponderAgent:
    """Create a responder agent."""
    return ResponderAgent(llm_client=llm_client)
