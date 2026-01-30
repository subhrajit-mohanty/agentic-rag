"""
Enhanced API Router for Enterprise Agentic RAG

Provides REST API endpoints for:
- Multi-agent RAG queries
- Feedback submission
- Analytics and metrics
- Memory management
- Tool management
- System configuration

This integrates all the enhanced components into a unified API.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================

# Query Models
class AgenticQueryRequest(BaseModel):
    """Request for agentic RAG query."""
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    persona_id: Optional[str] = None
    
    # Options
    use_cache: bool = True
    include_reasoning: bool = True
    include_sources: bool = True
    
    # Agent options
    max_iterations: int = Field(default=10, ge=1, le=50)
    orchestration_mode: str = Field(default="dynamic")  # sequential, parallel, dynamic
    enable_verification: bool = True
    enable_memory: bool = True
    
    # Retrieval options
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranking: bool = True
    retrieval_strategy: str = Field(default="hybrid")  # vector, keyword, hybrid


class AgenticQueryResponse(BaseModel):
    """Response from agentic RAG query."""
    query_id: str
    query: str
    answer: str
    
    # Sources
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Confidence
    confidence: float = Field(ge=0.0, le=1.0)
    
    # Execution details
    agents_used: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    total_iterations: int = 0
    execution_time_ms: float = 0.0
    
    # Reasoning trace
    reasoning_steps: List[str] = Field(default_factory=list)
    
    # Verification
    verification_passed: bool = True
    verification_issues: List[str] = Field(default_factory=list)
    
    # Cache
    cache_hit: bool = False
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Feedback Models
class FeedbackRequest(BaseModel):
    """Request for submitting feedback."""
    query_id: str
    feedback_type: str  # rating, thumbs, correction, retrieval
    
    # For ratings
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    
    # For thumbs
    thumbs_up: Optional[bool] = None
    
    # For corrections
    corrected_answer: Optional[str] = None
    
    # For retrieval feedback
    relevant_docs: List[str] = Field(default_factory=list)
    irrelevant_docs: List[str] = Field(default_factory=list)
    missing_info: Optional[str] = None
    
    # Common
    comment: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Response for feedback submission."""
    feedback_id: str
    status: str = "received"
    message: str = "Thank you for your feedback"


# Memory Models
class MemoryRequest(BaseModel):
    """Request for memory operations."""
    content: str
    memory_type: str  # fact, preference, entity, context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    long_term: bool = False


class MemoryResponse(BaseModel):
    """Response for memory operations."""
    memory_id: str
    status: str = "stored"


class MemoryContextResponse(BaseModel):
    """Response with memory context."""
    short_term_count: int = 0
    long_term_count: int = 0
    entities: Dict[str, Any] = Field(default_factory=dict)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    recent_queries: List[str] = Field(default_factory=list)


# Metrics Models
class MetricsResponse(BaseModel):
    """Response with system metrics."""
    period_start: datetime
    period_end: datetime
    
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    avg_response_time_ms: float = 0.0
    
    total_feedback: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    avg_rating: float = 0.0
    
    cache_hit_rate: float = 0.0
    avg_docs_retrieved: float = 0.0
    
    agent_stats: Dict[str, Any] = Field(default_factory=dict)
    tool_stats: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """System health response."""
    status: str = "healthy"
    version: str = "2.0.0"
    components: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Tool Models
class ToolExecutionRequest(BaseModel):
    """Request for direct tool execution."""
    tool_name: str
    input_data: Dict[str, Any]


class ToolExecutionResponse(BaseModel):
    """Response from tool execution."""
    tool_name: str
    status: str  # success, failure, timeout
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0


# =============================================================================
# Router Factory
# =============================================================================

def create_enhanced_router(
    orchestrator: Any = None,
    feedback_service: Any = None,
    memory_manager: Any = None,
    query_understanding: Any = None,
    tool_registry: Any = None,
    cache_service: Any = None
) -> APIRouter:
    """
    Create the enhanced API router with all dependencies.
    
    Args:
        orchestrator: Multi-agent orchestrator
        feedback_service: Feedback collection service
        memory_manager: Memory management service
        query_understanding: Query understanding service
        tool_registry: Tool registry
        cache_service: Cache service
        
    Returns:
        Configured APIRouter
    """
    router = APIRouter(prefix="/api/v2", tags=["Agentic RAG v2"])
    
    # ==========================================================================
    # Query Endpoints
    # ==========================================================================
    
    @router.post("/ask", response_model=AgenticQueryResponse)
    async def ask_agentic(
        request: AgenticQueryRequest,
        background_tasks: BackgroundTasks
    ):
        """
        Process a query using multi-agent RAG pipeline.
        
        This is the main endpoint for the enterprise RAG system.
        It uses multiple agents with message passing to:
        1. Understand the query
        2. Plan execution
        3. Retrieve relevant documents
        4. Verify and generate answer
        """
        start_time = datetime.utcnow()
        
        try:
            # Check cache if enabled
            if request.use_cache and cache_service:
                cached = await cache_service.get(
                    request.query,
                    user_id=request.user_id
                )
                if cached:
                    return AgenticQueryResponse(
                        **cached,
                        cache_hit=True
                    )
            
            # Query understanding (optional preprocessing)
            query_analysis = None
            if query_understanding:
                query_analysis = await query_understanding.analyze(request.query)
            
            # Get memory context if enabled
            memory_context = None
            if request.enable_memory and memory_manager:
                memory_context = await memory_manager.get_context(
                    query=request.query,
                    user_id=request.user_id,
                    session_id=request.session_id
                )
            
            # Execute multi-agent orchestration
            if orchestrator:
                result = await orchestrator.orchestrate(
                    query=request.query,
                    session_id=request.session_id,
                    user_id=request.user_id
                )
                
                response = AgenticQueryResponse(
                    query_id=result.query_id,
                    query=result.query,
                    answer=result.answer,
                    sources=[{"document_id": c.get("document_id"), "relevance_score": c.get("relevance_score")} for c in result.citations],
                    citations=result.citations,
                    confidence=result.confidence,
                    agents_used=result.agents_used,
                    tools_used=[],  # Extract from agent_outputs if needed
                    total_iterations=result.total_iterations,
                    execution_time_ms=result.execution_time_ms,
                    reasoning_steps=result.reasoning_trace,
                    verification_passed=result.verification_passed,
                    verification_issues=result.verification_issues,
                    cache_hit=False,
                    metadata=result.metadata
                )
            else:
                # Fallback response
                response = AgenticQueryResponse(
                    query_id="fallback",
                    query=request.query,
                    answer="The agentic RAG system is not fully configured. Please check the system setup.",
                    confidence=0.0,
                    execution_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
                )
            
            # Cache result
            if request.use_cache and cache_service and response.confidence > 0.5:
                background_tasks.add_task(
                    cache_service.set,
                    request.query,
                    response.model_dump(),
                    user_id=request.user_id
                )
            
            # Log analytics
            if feedback_service:
                background_tasks.add_task(
                    feedback_service.log_query,
                    query_id=response.query_id,
                    query=request.query,
                    success=response.confidence > 0,
                    duration_ms=response.execution_time_ms,
                    cache_hit=response.cache_hit,
                    docs_retrieved=len(response.sources),
                    agents_used=response.agents_used
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            
            # Log error
            if feedback_service:
                background_tasks.add_task(
                    feedback_service.log_error,
                    error_type="query_execution",
                    error_message=str(e),
                    query_id=None
                )
            
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/analyze", response_model=Dict[str, Any])
    async def analyze_query(query: str = Query(..., min_length=1)):
        """
        Analyze a query without executing the full RAG pipeline.
        
        Returns query understanding results including:
        - Query type and intent
        - Extracted entities
        - Suggested retrieval strategy
        """
        if not query_understanding:
            raise HTTPException(status_code=503, detail="Query understanding not available")
        
        try:
            analysis = await query_understanding.analyze(query)
            return analysis.model_dump()
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ==========================================================================
    # Feedback Endpoints
    # ==========================================================================
    
    @router.post("/feedback", response_model=FeedbackResponse)
    async def submit_feedback(request: FeedbackRequest):
        """
        Submit feedback for a query response.
        
        Supports multiple feedback types:
        - rating: 1-5 star rating
        - thumbs: Thumbs up/down
        - correction: Corrected answer
        - retrieval: Document relevance feedback
        """
        if not feedback_service:
            raise HTTPException(status_code=503, detail="Feedback service not available")
        
        try:
            if request.feedback_type == "rating":
                if request.rating is None:
                    raise HTTPException(status_code=400, detail="Rating required for rating feedback")
                
                feedback_id = await feedback_service.submit_rating(
                    query_id=request.query_id,
                    query="",  # Would need to be stored/retrieved
                    answer="",
                    rating=request.rating,
                    comment=request.comment,
                    user_id=request.user_id,
                    session_id=request.session_id
                )
            
            elif request.feedback_type == "thumbs":
                if request.thumbs_up is None:
                    raise HTTPException(status_code=400, detail="thumbs_up required for thumbs feedback")
                
                feedback_id = await feedback_service.submit_thumbs(
                    query_id=request.query_id,
                    query="",
                    answer="",
                    thumbs_up=request.thumbs_up,
                    comment=request.comment,
                    user_id=request.user_id,
                    session_id=request.session_id
                )
            
            elif request.feedback_type == "correction":
                if not request.corrected_answer:
                    raise HTTPException(status_code=400, detail="corrected_answer required")
                
                feedback_id = await feedback_service.submit_correction(
                    query_id=request.query_id,
                    query="",
                    original_answer="",
                    corrected_answer=request.corrected_answer,
                    user_id=request.user_id,
                    session_id=request.session_id
                )
            
            elif request.feedback_type == "retrieval":
                feedback_id = await feedback_service.submit_retrieval_feedback(
                    query_id=request.query_id,
                    query="",
                    answer="",
                    relevant_docs=request.relevant_docs,
                    irrelevant_docs=request.irrelevant_docs,
                    missing_info=request.missing_info,
                    user_id=request.user_id
                )
            
            else:
                raise HTTPException(status_code=400, detail=f"Unknown feedback type: {request.feedback_type}")
            
            return FeedbackResponse(feedback_id=feedback_id)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Feedback submission failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ==========================================================================
    # Memory Endpoints
    # ==========================================================================
    
    @router.post("/memory", response_model=MemoryResponse)
    async def store_memory(request: MemoryRequest):
        """
        Store a memory entry.
        
        Memory types:
        - fact: Factual information
        - preference: User preferences
        - entity: Named entities
        - context: Conversation context
        """
        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory service not available")
        
        try:
            from backend.services.agents.memory.memory_manager import MemoryType
            
            type_map = {
                "fact": MemoryType.FACT,
                "preference": MemoryType.PREFERENCE,
                "entity": MemoryType.ENTITY,
                "context": MemoryType.CONTEXT,
            }
            
            memory_type = type_map.get(request.memory_type)
            if not memory_type:
                raise HTTPException(status_code=400, detail=f"Unknown memory type: {request.memory_type}")
            
            memory_id = await memory_manager.store(
                content=request.content,
                memory_type=memory_type,
                user_id=request.user_id,
                session_id=request.session_id,
                tags=request.tags,
                long_term=request.long_term
            )
            
            return MemoryResponse(memory_id=memory_id)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Memory storage failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/memory/context", response_model=MemoryContextResponse)
    async def get_memory_context(
        query: str = Query(..., min_length=1),
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Get memory context for a query.
        
        Returns relevant memories from both short-term and long-term storage.
        """
        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory service not available")
        
        try:
            context = await memory_manager.get_context(
                query=query,
                user_id=user_id,
                session_id=session_id
            )
            
            return MemoryContextResponse(
                short_term_count=len(context.short_term),
                long_term_count=len(context.long_term),
                entities=context.entities,
                preferences=context.preferences,
                recent_queries=context.recent_queries
            )
            
        except Exception as e:
            logger.error(f"Memory context retrieval failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.delete("/memory")
    async def clear_memory(
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Clear memory for a user or session."""
        if not memory_manager:
            raise HTTPException(status_code=503, detail="Memory service not available")
        
        try:
            count = await memory_manager.short_term.clear(
                user_id=user_id,
                session_id=session_id
            )
            return {"status": "cleared", "entries_removed": count}
        except Exception as e:
            logger.error(f"Memory clear failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ==========================================================================
    # Tools Endpoints
    # ==========================================================================
    
    @router.get("/tools")
    async def list_tools():
        """List all available tools."""
        if not tool_registry:
            return {"tools": []}
        
        return {
            "tools": tool_registry.get_definitions_json(enabled_only=True)
        }
    
    @router.post("/tools/execute", response_model=ToolExecutionResponse)
    async def execute_tool(request: ToolExecutionRequest):
        """
        Execute a tool directly.
        
        This bypasses the agent orchestration for direct tool access.
        """
        if not tool_registry:
            raise HTTPException(status_code=503, detail="Tool registry not available")
        
        try:
            result = await tool_registry.execute(
                request.tool_name,
                request.input_data
            )
            
            return ToolExecutionResponse(
                tool_name=request.tool_name,
                status=result.status.value,
                result=result.result,
                error=result.error,
                execution_time_ms=result.execution_time_ms
            )
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ==========================================================================
    # Metrics Endpoints
    # ==========================================================================
    
    @router.get("/metrics", response_model=MetricsResponse)
    async def get_metrics(period_hours: int = Query(default=24, ge=1, le=168)):
        """
        Get system performance metrics.
        
        Returns aggregated metrics for the specified period.
        """
        if not feedback_service:
            raise HTTPException(status_code=503, detail="Metrics not available")
        
        try:
            metrics = await feedback_service.get_metrics(period_hours)
            
            return MetricsResponse(
                period_start=metrics.period_start,
                period_end=metrics.period_end,
                total_queries=metrics.total_queries,
                successful_queries=metrics.successful_queries,
                failed_queries=metrics.failed_queries,
                avg_response_time_ms=metrics.avg_response_time_ms,
                total_feedback=metrics.total_feedback,
                positive_feedback=metrics.positive_feedback,
                negative_feedback=metrics.negative_feedback,
                avg_rating=metrics.avg_rating,
                cache_hit_rate=metrics.cache_hit_rate,
                avg_docs_retrieved=metrics.avg_docs_retrieved,
                agent_stats={
                    "calls": metrics.agent_calls,
                    "success_rates": metrics.agent_success_rates
                },
                tool_stats={
                    "usage": metrics.tool_usage,
                    "success_rates": metrics.tool_success_rates
                }
            )
            
        except Exception as e:
            logger.error(f"Metrics retrieval failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/metrics/recommendations")
    async def get_recommendations():
        """Get improvement recommendations based on feedback and metrics."""
        if not feedback_service:
            raise HTTPException(status_code=503, detail="Metrics not available")
        
        try:
            recommendations = await feedback_service.get_recommendations()
            return {"recommendations": recommendations}
        except Exception as e:
            logger.error(f"Recommendations failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ==========================================================================
    # Health Endpoints
    # ==========================================================================
    
    @router.get("/health", response_model=HealthResponse)
    async def health_check():
        """
        Get system health status.
        
        Checks all components and returns overall health.
        """
        components = {}
        
        # Check orchestrator
        if orchestrator:
            try:
                agent_status = orchestrator.get_agent_status()
                components["orchestrator"] = {
                    "status": "healthy",
                    "agents": len(agent_status),
                    "mode": orchestrator.mode.value
                }
            except Exception as e:
                components["orchestrator"] = {"status": "unhealthy", "error": str(e)}
        else:
            components["orchestrator"] = {"status": "not_configured"}
        
        # Check memory
        if memory_manager:
            components["memory"] = {"status": "healthy"}
        else:
            components["memory"] = {"status": "not_configured"}
        
        # Check feedback
        if feedback_service:
            components["feedback"] = {"status": "healthy"}
        else:
            components["feedback"] = {"status": "not_configured"}
        
        # Check tools
        if tool_registry:
            tools = tool_registry.get_enabled()
            components["tools"] = {
                "status": "healthy",
                "count": len(tools)
            }
        else:
            components["tools"] = {"status": "not_configured"}
        
        # Overall status
        unhealthy = [k for k, v in components.items() if v.get("status") == "unhealthy"]
        status = "unhealthy" if unhealthy else "healthy"
        
        return HealthResponse(
            status=status,
            version="2.0.0",
            components=components
        )
    
    @router.get("/agents/status")
    async def get_agent_status():
        """Get status of all registered agents."""
        if not orchestrator:
            raise HTTPException(status_code=503, detail="Orchestrator not available")
        
        try:
            return orchestrator.get_agent_status()
        except Exception as e:
            logger.error(f"Agent status failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return router


# =============================================================================
# Integration Helper
# =============================================================================

async def setup_enhanced_services(app, settings):
    """
    Setup all enhanced services and integrate with the FastAPI app.
    
    This is called during app startup to initialize all components.
    """
    from backend.services.agents.core.orchestrator import create_orchestrator, OrchestratorMode
    from backend.services.agents.feedback.service import create_feedback_service
    from backend.services.agents.memory.memory_manager import create_memory_manager
    from backend.services.query_understanding.service import create_query_understanding_service
    from backend.services.agents.tools.base import get_tool_registry
    from backend.services.agents.tools.specialized import (
        create_calculator_tool,
        create_code_executor_tool,
        create_web_search_tool,
        create_search_tool
    )
    from backend.services.llm.client import get_llm_client
    from backend.services.vector_store.service import get_vector_store_service
    from backend.services.reranker.service import RerankerFactory
    
    # Get existing services
    llm_client = await get_llm_client()
    vector_store = await get_vector_store_service()
    
    # Create reranker
    reranker = RerankerFactory.create(
        provider="auto",
        cohere_api_key=getattr(settings, 'cohere_api_key', None)
    )
    
    # Create tool registry and register tools
    tool_registry = get_tool_registry()
    tool_registry.register(create_calculator_tool())
    tool_registry.register(create_code_executor_tool())
    tool_registry.register(create_search_tool(vector_store))
    
    if getattr(settings, 'tavily_api_key', None):
        tool_registry.register(create_web_search_tool(
            provider="tavily",
            tavily_api_key=settings.tavily_api_key
        ))
    
    # Create orchestrator
    mode = OrchestratorMode.DYNAMIC
    orchestrator = create_orchestrator(
        llm_client=llm_client,
        vector_store=vector_store,
        reranker=reranker,
        mode=mode,
        max_iterations=settings.multi_agent.max_iterations if hasattr(settings, 'multi_agent') else 10,
        max_agent_calls=settings.multi_agent.max_agent_calls if hasattr(settings, 'multi_agent') else 20
    )
    
    # Create other services
    feedback_service = create_feedback_service()
    memory_manager = create_memory_manager(llm_client=llm_client)
    query_understanding = create_query_understanding_service(llm_client=llm_client)
    
    # Create and add router
    router = create_enhanced_router(
        orchestrator=orchestrator,
        feedback_service=feedback_service,
        memory_manager=memory_manager,
        query_understanding=query_understanding,
        tool_registry=tool_registry
    )
    
    app.include_router(router)
    
    logger.info("Enhanced services setup complete")
    
    return {
        "orchestrator": orchestrator,
        "feedback_service": feedback_service,
        "memory_manager": memory_manager,
        "query_understanding": query_understanding,
        "tool_registry": tool_registry
    }
