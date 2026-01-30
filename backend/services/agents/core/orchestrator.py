"""
Dynamic Multi-Agent Orchestrator

Manages agent collaboration with true message passing, similar to AutoGen/CrewAI.
Supports multiple orchestration modes:
- Sequential: Agents execute one after another
- Parallel: Agents execute simultaneously where possible
- Hierarchical: Tree-like execution with delegation
- Dynamic: LLM decides execution order based on context
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from .base import (
    BaseAgent, BaseOrchestrator, AgentContext, AgentMessage,
    AgentRegistry, MessageRouter, MessageType, MessagePriority,
    OrchestratorMode, AgentState, create_context, get_agent_registry
)
from .specialized_agents import (
    PlannerAgent, ResearcherAgent, RetrieverAgent,
    VerifierAgent, ResponderAgent,
    create_planner_agent, create_researcher_agent,
    create_retriever_agent, create_verifier_agent,
    create_responder_agent
)

logger = logging.getLogger(__name__)


# =============================================================================
# Orchestrator Response Models
# =============================================================================

class OrchestrationResult(BaseModel):
    """Result from orchestration."""
    query_id: str
    query: str
    answer: str
    citations: List[Dict[str, Any]] = []
    confidence: float = Field(ge=0.0, le=1.0)
    
    # Execution details
    agents_used: List[str]
    total_iterations: int
    total_agent_calls: int
    execution_time_ms: float
    
    # Reasoning trace
    reasoning_trace: List[str]
    agent_outputs: Dict[str, Any] = {}
    
    # Verification
    verification_passed: bool = True
    verification_issues: List[str] = []
    
    # Metadata
    metadata: Dict[str, Any] = {}


class AgentStep(BaseModel):
    """A single step in the orchestration plan."""
    agent_id: str
    action: str
    depends_on: List[str] = []
    inputs: Dict[str, Any] = {}
    optional: bool = False


# =============================================================================
# Dynamic Orchestrator
# =============================================================================

class DynamicOrchestrator(BaseOrchestrator):
    """
    Dynamic multi-agent orchestrator with LLM-based planning.
    
    Features:
    - Dynamic agent selection based on query
    - Parallel execution where possible
    - Agent reflection and debate
    - Consensus mechanisms
    - Adaptive replanning
    """
    
    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        llm_client: Any = None,
        vector_store: Any = None,
        reranker: Any = None,
        mode: OrchestratorMode = OrchestratorMode.DYNAMIC,
        max_iterations: int = 10,
        max_agent_calls: int = 20,
        enable_reflection: bool = True,
        enable_debate: bool = False,
        consensus_threshold: float = 0.8
    ):
        # Initialize registry
        self._registry = registry or get_agent_registry()
        
        super().__init__(
            registry=self._registry,
            mode=mode,
            max_iterations=max_iterations,
            max_agent_calls=max_agent_calls
        )
        
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.reranker = reranker
        self.enable_reflection = enable_reflection
        self.enable_debate = enable_debate
        self.consensus_threshold = consensus_threshold
        
        # Initialize agents
        self._initialize_agents()
        
        logger.info(
            f"DynamicOrchestrator initialized: mode={mode}, "
            f"max_iterations={max_iterations}, reflection={enable_reflection}"
        )
    
    def _initialize_agents(self) -> None:
        """Initialize and register all agents."""
        # Clear existing agents
        self._registry.clear()
        
        # Create agents
        agents = [
            create_planner_agent(self.llm_client),
            create_researcher_agent(self.llm_client),
            create_retriever_agent(
                self.llm_client,
                self.vector_store,
                self.reranker
            ),
            create_verifier_agent(self.llm_client),
            create_responder_agent(self.llm_client)
        ]
        
        # Register all agents
        for agent in agents:
            self._registry.register(agent)
        
        logger.info(f"Registered {len(agents)} agents")
    
    async def orchestrate(
        self,
        query: str,
        context: Optional[AgentContext] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> OrchestrationResult:
        """
        Orchestrate multi-agent execution for a query.
        
        Args:
            query: User query
            context: Optional pre-existing context
            session_id: Session identifier
            user_id: User identifier
            
        Returns:
            Orchestration result with answer and metadata
        """
        start_time = datetime.utcnow()
        self.reset_counters()
        
        # Create context if not provided
        if context is None:
            context = create_context(
                query=query,
                session_id=session_id,
                user_id=user_id,
                max_iterations=self.max_iterations
            )
        
        logger.info(f"[{context.query_id}] Starting orchestration for: {query[:50]}...")
        
        try:
            # Execute based on mode
            if self.mode == OrchestratorMode.SEQUENTIAL:
                result = await self._sequential_orchestration(context)
            elif self.mode == OrchestratorMode.PARALLEL:
                result = await self._parallel_orchestration(context)
            elif self.mode == OrchestratorMode.DYNAMIC:
                result = await self._dynamic_orchestration(context)
            else:
                result = await self._sequential_orchestration(context)
            
            # Build final result
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return self._build_result(context, result, execution_time)
            
        except Exception as e:
            logger.error(f"[{context.query_id}] Orchestration failed: {e}")
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return OrchestrationResult(
                query_id=context.query_id,
                query=query,
                answer=f"I apologize, but an error occurred: {str(e)}",
                confidence=0.0,
                agents_used=[],
                total_iterations=self._iteration_count,
                total_agent_calls=self._agent_call_count,
                execution_time_ms=execution_time,
                reasoning_trace=[f"Error: {str(e)}"],
                verification_passed=False
            )
    
    async def _sequential_orchestration(
        self,
        context: AgentContext
    ) -> Dict[str, Any]:
        """Execute agents in a fixed sequential order."""
        reasoning_trace = []
        
        # Step 1: Research
        researcher = self._registry.get("researcher")
        if researcher:
            result = await self._execute_agent(
                researcher,
                {"query": context.original_query},
                context
            )
            reasoning_trace.append(f"Researcher: analyzed query")
        
        # Step 2: Retrieve
        retriever = self._registry.get("retriever")
        if retriever:
            analysis = context.agent_outputs.get("researcher", {})
            result = await self._execute_agent(
                retriever,
                {
                    "query": context.original_query,
                    "strategy": analysis.get("suggested_retrieval_strategy", "hybrid")
                },
                context
            )
            reasoning_trace.append(f"Retriever: found {len(context.retrieved_documents)} documents")
        
        # Step 3: Generate response
        responder = self._registry.get("responder")
        if responder:
            result = await self._execute_agent(
                responder,
                {
                    "query": context.original_query,
                    "documents": context.retrieved_documents
                },
                context
            )
            reasoning_trace.append(f"Responder: generated answer")
        
        # Step 4: Verify (optional)
        verifier = self._registry.get("verifier")
        if verifier:
            response = context.agent_outputs.get("responder", {})
            result = await self._execute_agent(
                verifier,
                {
                    "answer": response.get("answer", ""),
                    "query": context.original_query,
                    "sources": context.retrieved_documents
                },
                context
            )
            reasoning_trace.append(f"Verifier: verification complete")
        
        return {
            "reasoning_trace": reasoning_trace,
            "agents_used": ["researcher", "retriever", "responder", "verifier"]
        }
    
    async def _parallel_orchestration(
        self,
        context: AgentContext
    ) -> Dict[str, Any]:
        """Execute independent agents in parallel."""
        reasoning_trace = []
        
        # Phase 1: Research (must be first)
        researcher = self._registry.get("researcher")
        if researcher:
            await self._execute_agent(
                researcher,
                {"query": context.original_query},
                context
            )
            reasoning_trace.append("Research phase complete")
        
        # Phase 2: Retrieve (depends on research)
        retriever = self._registry.get("retriever")
        if retriever:
            await self._execute_agent(
                retriever,
                {"query": context.original_query},
                context
            )
            reasoning_trace.append("Retrieval phase complete")
        
        # Phase 3: Response and Verification (can be parallelized)
        responder = self._registry.get("responder")
        if responder:
            await self._execute_agent(
                responder,
                {
                    "query": context.original_query,
                    "documents": context.retrieved_documents
                },
                context
            )
        
        # Verify after response
        verifier = self._registry.get("verifier")
        if verifier:
            response = context.agent_outputs.get("responder", {})
            await self._execute_agent(
                verifier,
                {
                    "answer": response.get("answer", ""),
                    "query": context.original_query
                },
                context
            )
        
        reasoning_trace.append("Response and verification complete")
        
        return {
            "reasoning_trace": reasoning_trace,
            "agents_used": ["researcher", "retriever", "responder", "verifier"]
        }
    
    async def _dynamic_orchestration(
        self,
        context: AgentContext
    ) -> Dict[str, Any]:
        """
        Dynamic orchestration with LLM-based planning.
        
        The planner agent decides:
        - Which agents to use
        - In what order
        - When to iterate
        - When to stop
        """
        reasoning_trace = []
        agents_used = set()
        
        # Step 1: Get execution plan from planner
        planner = self._registry.get("planner")
        if planner:
            plan_result = await self._execute_agent(
                planner,
                {"query": context.original_query},
                context
            )
            agents_used.add("planner")
            reasoning_trace.append("Planner: created execution plan")
        
        plan = context.agent_outputs.get("planner", {})
        steps = plan.get("steps", [])
        
        # If no plan, fall back to sequential
        if not steps:
            logger.warning("No plan generated, falling back to sequential")
            return await self._sequential_orchestration(context)
        
        # Step 2: Execute plan
        for step in steps:
            if step is None:
                continue
                
            agent_id = step.get("agent")
            action = step.get("action")
            
            if not agent_id:
                continue
            
            agent = self._registry.get(agent_id)
            if not agent:
                logger.warning(f"Agent not found: {agent_id}")
                continue
            
            # Check termination conditions
            if await self._check_termination(context):
                reasoning_trace.append(f"Terminated: {context.termination_reason}")
                break
            
            # Build input for agent
            agent_input = self._build_agent_input(agent_id, action, context)
            
            # Execute agent
            try:
                result = await self._execute_agent(agent, agent_input, context)
                agents_used.add(agent_id)
                reasoning_trace.append(f"{agent_id}: {action} complete")
                
                # Reflection if enabled
                if self.enable_reflection:
                    reflection = await agent.reflect(context)
                    if reflection.get("suggest_replan"):
                        reasoning_trace.append(f"{agent_id}: suggested replanning")
                        # Could trigger replanning here
                
            except Exception as e:
                logger.error(f"Agent {agent_id} failed: {e}")
                reasoning_trace.append(f"{agent_id}: failed - {str(e)}")
            
            self._iteration_count += 1
        
        # Step 3: Final verification if not already done
        if "verifier" not in agents_used:
            verifier = self._registry.get("verifier")
            if verifier:
                response = context.agent_outputs.get("responder", {})
                if response.get("answer"):
                    await self._execute_agent(
                        verifier,
                        {
                            "answer": response.get("answer"),
                            "query": context.original_query,
                            "sources": context.retrieved_documents
                        },
                        context
                    )
                    agents_used.add("verifier")
                    reasoning_trace.append("Verifier: final verification")
        
        return {
            "reasoning_trace": reasoning_trace,
            "agents_used": list(agents_used)
        }
    
    def _build_agent_input(
        self,
        agent_id: str,
        action: str,
        context: AgentContext
    ) -> Dict[str, Any]:
        """Build appropriate input for an agent based on context."""
        base_input = {"query": context.original_query}
        
        if agent_id == "retriever":
            analysis = context.agent_outputs.get("researcher", {})
            base_input.update({
                "strategy": analysis.get("suggested_retrieval_strategy", "hybrid"),
                "expand_query": analysis.get("complexity") == "complex"
            })
        
        elif agent_id == "responder":
            base_input.update({
                "documents": context.retrieved_documents,
                "analysis": context.agent_outputs.get("researcher", {})
            })
        
        elif agent_id == "verifier":
            response = context.agent_outputs.get("responder", {})
            base_input.update({
                "answer": response.get("answer", ""),
                "sources": context.retrieved_documents
            })
        
        return base_input
    
    def _build_result(
        self,
        context: AgentContext,
        orchestration_result: Dict[str, Any],
        execution_time_ms: float
    ) -> OrchestrationResult:
        """Build the final orchestration result."""
        responder_output = context.agent_outputs.get("responder", {})
        verifier_output = context.agent_outputs.get("verifier", {})
        
        # Combine reasoning traces
        reasoning_trace = orchestration_result.get("reasoning_trace", [])
        if responder_output.get("reasoning_trace"):
            reasoning_trace.extend(responder_output["reasoning_trace"])
        
        # Get verification status
        verification_passed = verifier_output.get("is_verified", True)
        verification_issues = verifier_output.get("issues", [])
        
        return OrchestrationResult(
            query_id=context.query_id,
            query=context.original_query,
            answer=responder_output.get("answer", "Unable to generate response."),
            citations=responder_output.get("citations", []),
            confidence=responder_output.get("confidence", 0.0),
            agents_used=orchestration_result.get("agents_used", []),
            total_iterations=self._iteration_count,
            total_agent_calls=self._agent_call_count,
            execution_time_ms=execution_time_ms,
            reasoning_trace=reasoning_trace,
            agent_outputs=context.agent_outputs,
            verification_passed=verification_passed,
            verification_issues=verification_issues,
            metadata={
                "session_id": context.session_id,
                "user_id": context.user_id,
                "mode": self.mode.value
            }
        )
    
    async def send_agent_message(
        self,
        sender_id: str,
        receiver_id: str,
        message_type: MessageType,
        content: Dict[str, Any],
        context: AgentContext
    ) -> Optional[AgentMessage]:
        """
        Send a message from one agent to another.
        
        This enables true inter-agent communication.
        """
        message = AgentMessage(
            type=message_type,
            sender=sender_id,
            receiver=receiver_id,
            content=content,
            correlation_id=context.query_id
        )
        
        # Route the message
        success = await self.router.route(message)
        
        if success:
            context.add_message(message)
            return message
        
        return None
    
    async def broadcast_message(
        self,
        sender_id: str,
        message_type: MessageType,
        content: Dict[str, Any],
        context: AgentContext
    ) -> AgentMessage:
        """Broadcast a message to all agents."""
        message = AgentMessage(
            type=message_type,
            sender=sender_id,
            receiver="broadcast",
            content=content,
            correlation_id=context.query_id
        )
        
        await self.router.route(message)
        context.add_message(message)
        
        return message
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all registered agents."""
        return {
            agent.agent_id: agent.get_info()
            for agent in self._registry.get_all()
        }


# =============================================================================
# Factory Functions
# =============================================================================

_orchestrator_instance: Optional[DynamicOrchestrator] = None


def get_orchestrator(
    llm_client: Any = None,
    vector_store: Any = None,
    reranker: Any = None,
    **kwargs
) -> DynamicOrchestrator:
    """Get or create the global orchestrator instance."""
    global _orchestrator_instance
    
    if _orchestrator_instance is None:
        _orchestrator_instance = DynamicOrchestrator(
            llm_client=llm_client,
            vector_store=vector_store,
            reranker=reranker,
            **kwargs
        )
    
    return _orchestrator_instance


def create_orchestrator(
    llm_client: Any = None,
    vector_store: Any = None,
    reranker: Any = None,
    mode: OrchestratorMode = OrchestratorMode.DYNAMIC,
    **kwargs
) -> DynamicOrchestrator:
    """Create a new orchestrator instance."""
    return DynamicOrchestrator(
        llm_client=llm_client,
        vector_store=vector_store,
        reranker=reranker,
        mode=mode,
        **kwargs
    )
