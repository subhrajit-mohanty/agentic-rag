"""
MongoDB Document Models

Beanie ODM document models for the Enterprise RAG Platform.
Defines the schema for documents, personas, queries, and analytics.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from beanie import Document, Indexed
from pydantic import BaseModel, Field


# ============================================
# Embedded Models (for nested documents)
# ============================================

class DocumentMetadata(BaseModel):
    """Metadata for knowledge base documents."""
    source: str = Field(description="Source system (SharePoint, S3, etc.)")
    filename: str = Field(description="Original filename")
    category: str = Field(description="Document category")
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)
    custom_metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingData(BaseModel):
    """Vector embedding data."""
    model: str = Field(description="Embedding model used")
    dimension: int = Field(description="Vector dimension")
    vector: List[float] = Field(description="Embedding vector")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SourceReference(BaseModel):
    """Reference to a source document in query results."""
    document_id: str
    title: str
    url: Optional[str] = None
    relevance_score: float = Field(ge=0.0, le=1.0)
    chunk_text: Optional[str] = None


class ReasoningStep(BaseModel):
    """A step in the agent's reasoning process."""
    node: str = Field(description="Node name in the graph")
    action: str = Field(description="Action taken")
    result: Optional[str] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================
# Main Document Models
# ============================================

class KnowledgeDocument(Document):
    """
    Knowledge base document with embeddings.
    
    Stores enterprise documents with their vector embeddings
    for semantic search.
    """
    
    # Indexed fields for fast queries
    document_id: Indexed(str, unique=True)
    title: Indexed(str)
    
    # Content
    content: str = Field(description="Full document content")
    chunk_index: int = Field(default=0, description="Chunk index if document is split")
    total_chunks: int = Field(default=1, description="Total number of chunks")
    
    # Metadata
    metadata: DocumentMetadata
    
    # Embeddings
    embedding: Optional[EmbeddingData] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    indexed_at: Optional[datetime] = None
    
    # Status
    is_active: bool = Field(default=True)
    
    class Settings:
        name = "knowledge_documents"
        indexes = [
            "metadata.category",
            "metadata.source",
            "created_at",
        ]


class Persona(Document):
    """
    AI Persona configuration.
    
    Defines specialized agent behaviors and capabilities.
    """
    
    persona_id: Indexed(str, unique=True)
    name: Indexed(str)
    
    # Configuration
    system_prompt: str = Field(description="System prompt for this persona")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    
    # Capabilities
    allowed_tools: List[str] = Field(default_factory=lambda: ["Retrieval"])
    allowed_categories: List[str] = Field(default_factory=list)
    
    # Metadata
    description: Optional[str] = None
    icon: Optional[str] = None
    is_active: bool = Field(default=True)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "personas"


class QueryLog(Document):
    """
    Query execution log for analytics and debugging.
    
    Tracks all queries through the RAG system.
    """
    
    query_id: Indexed(str, unique=True)
    
    # Query details
    original_query: str
    rewritten_query: Optional[str] = None
    
    # Execution context
    framework: str = Field(default="LangGraph")
    persona_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # Results
    answer: Optional[str] = None
    sources: List[SourceReference] = Field(default_factory=list)
    reasoning_steps: List[ReasoningStep] = Field(default_factory=list)
    
    # Performance metrics
    retrieval_attempts: int = Field(default=0)
    execution_time_ms: float = Field(default=0.0)
    cache_hit: bool = Field(default=False)
    
    # Guardrail results
    guardrail_score: Optional[int] = None
    guardrail_passed: bool = Field(default=True)
    
    # Feedback
    user_rating: Optional[int] = Field(default=None, ge=1, le=5)
    user_feedback: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "query_logs"
        indexes = [
            "created_at",
            "persona_id",
            "user_id",
            "cache_hit",
        ]


class Connector(Document):
    """
    Data connector configuration.
    
    Defines connections to external data sources.
    """
    
    connector_id: Indexed(str, unique=True)
    name: Indexed(str)
    connector_type: str = Field(description="Type: sharepoint, s3, gdrive, etc.")
    
    # Connection settings (encrypted in production)
    connection_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Sync settings
    sync_enabled: bool = Field(default=True)
    sync_interval_minutes: int = Field(default=60)
    last_sync_at: Optional[datetime] = None
    last_sync_status: Optional[str] = None
    
    # Stats
    documents_indexed: int = Field(default=0)
    
    # Status
    is_active: bool = Field(default=True)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Settings:
        name = "connectors"


class SystemStats(Document):
    """
    System statistics snapshot.
    
    Periodic snapshots of system metrics.
    """
    
    timestamp: Indexed(datetime)
    
    # Document stats
    total_documents: int = Field(default=0)
    total_vectors: int = Field(default=0)
    
    # Query stats
    total_queries_24h: int = Field(default=0)
    avg_latency_ms: float = Field(default=0.0)
    cache_hit_rate: float = Field(default=0.0)
    
    # System health
    services_health: Dict[str, str] = Field(default_factory=dict)
    
    class Settings:
        name = "system_stats"


# ============================================
# Document Model Registry
# ============================================

# All document models for Beanie initialization
ALL_DOCUMENT_MODELS = [
    KnowledgeDocument,
    Persona,
    QueryLog,
    Connector,
    SystemStats,
]
