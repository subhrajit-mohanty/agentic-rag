"""
Agentic RAG Models

Pydantic models for the agentic RAG workflow state, 
structured LLM outputs, and response schemas.
"""

from datetime import datetime
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ============================================
# LLM Structured Output Models
# ============================================

class GuardrailScoring(BaseModel):
    """
    Guardrail evaluation result.
    
    Determines if a query is within the allowed enterprise domains.
    """
    score: int = Field(
        ge=0, le=100,
        description="Relevance score to enterprise domains (0-100)"
    )
    reason: str = Field(
        description="Brief explanation for the score"
    )
    domains_matched: List[str] = Field(
        default_factory=list,
        description="List of matched enterprise domains"
    )


class GradeDocuments(BaseModel):
    """
    Document relevance grading result.
    
    Evaluates if retrieved documents are relevant to the query.
    """
    binary_score: Literal["yes", "no"] = Field(
        description="Is the document relevant to the query?"
    )
    reasoning: str = Field(
        description="Explanation for the relevance decision"
    )
    confidence: float = Field(
        default=0.8,
        ge=0.0, le=1.0,
        description="Confidence in the grading decision"
    )


class QueryRewrite(BaseModel):
    """
    Query rewriting result for improved retrieval.
    """
    original_query: str = Field(description="Original user query")
    rewritten_query: str = Field(description="Optimized query for search")
    strategy: str = Field(
        default="expansion",
        description="Rewriting strategy used"
    )


# ============================================
# Source and Citation Models
# ============================================

class SourceItem(BaseModel):
    """
    Source document reference for citations.
    """
    document_id: str = Field(description="Unique document identifier")
    title: str = Field(description="Document title")
    url: Optional[str] = Field(default=None, description="Document URL if available")
    relevance_score: float = Field(
        ge=0.0, le=1.0,
        description="Relevance score from search"
    )
    chunk_text: Optional[str] = Field(
        default=None,
        description="Relevant text chunk"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class GradingResult(BaseModel):
    """
    Result of grading a single document.
    """
    document_id: str
    is_relevant: bool
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str


# ============================================
# Agent State (TypedDict for LangGraph)
# ============================================

def add_messages(left: list, right: list) -> list:
    """Reducer function to append messages."""
    return left + right


def add_steps(left: list, right: list) -> list:
    """Reducer function to append reasoning steps."""
    return left + right


class AgentState(TypedDict):
    """
    State maintained across the agentic RAG workflow.
    
    This TypedDict is compatible with LangGraph's state management.
    """
    # Message history
    messages: Annotated[list, add_messages]
    
    # Query tracking
    original_query: Optional[str]
    rewritten_query: Optional[str]
    
    # Retrieval state
    retrieval_attempts: int
    retrieved_context: Optional[str]
    
    # Guardrail results
    guardrail_result: Optional[GuardrailScoring]
    
    # Routing decision
    routing_decision: Optional[str]
    
    # Sources and grading
    sources: List[SourceItem]
    relevant_sources: List[SourceItem]
    grading_results: List[GradingResult]
    
    # Metadata
    metadata: Dict[str, Any]
    
    # Final output
    answer: Optional[str]
    
    # Reasoning trace
    reasoning_steps: Annotated[List[str], add_steps]


# ============================================
# API Request/Response Models
# ============================================

class AgenticAskRequest(BaseModel):
    """Request model for the agentic RAG endpoint."""
    query: str = Field(
        min_length=1,
        max_length=2000,
        description="User query"
    )
    top_k: int = Field(
        default=5,
        ge=1, le=20,
        description="Number of documents to retrieve"
    )
    use_hybrid: bool = Field(
        default=True,
        description="Use hybrid search (BM25 + vector)"
    )
    model: Optional[str] = Field(
        default=None,
        description="LLM model to use"
    )
    framework: str = Field(
        default="LangGraph",
        description="Agent framework"
    )
    persona_id: Optional[str] = Field(
        default=None,
        description="Persona ID for specialized behavior"
    )
    stream: bool = Field(
        default=False,
        description="Enable streaming response"
    )


class AgenticAskResponse(BaseModel):
    """Response model for the agentic RAG endpoint."""
    query: str = Field(description="Original query")
    answer: str = Field(description="Generated answer")
    sources: List[SourceItem] = Field(
        default_factory=list,
        description="Source documents used"
    )
    chunks_used: int = Field(description="Number of chunks used")
    search_mode: str = Field(description="Search mode used")
    reasoning_steps: List[str] = Field(
        default_factory=list,
        description="Agent reasoning trace"
    )
    retrieval_attempts: int = Field(
        default=1,
        description="Number of retrieval attempts"
    )
    execution_time: float = Field(
        description="Execution time in seconds"
    )
    cache_hit: bool = Field(
        default=False,
        description="Whether response was from cache"
    )
    guardrail_score: Optional[int] = Field(
        default=None,
        description="Guardrail validation score"
    )


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(description="Overall health status")
    version: str = Field(description="API version")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Individual service health"
    )


class StatsResponse(BaseModel):
    """System statistics response model."""
    total_vectors: str = Field(description="Total vectors in store")
    agent_tasks: str = Field(description="Total agent tasks processed")
    active_connectors: str = Field(description="Number of active connectors")
    avg_latency: str = Field(description="Average response latency")
    health: Dict[str, str] = Field(description="Service health summary")
    cache_stats: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Cache statistics"
    )
