"""
Application Configuration Module

Centralized configuration management using Pydantic Settings.
Supports environment variables, .env files, and sensible defaults.
"""

import os
from functools import lru_cache
from typing import List, Optional, Literal, Dict, Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MongoDBSettings(BaseSettings):
    """MongoDB connection settings."""
    
    uri: str = Field(
        default="mongodb://localhost:27017/enterprise_rag",
        description="MongoDB connection URI"
    )
    database: str = Field(
        default="enterprise_rag",
        description="Database name"
    )
    max_pool_size: int = Field(default=10, ge=1, le=100)
    min_pool_size: int = Field(default=1, ge=1)
    
    model_config = SettingsConfigDict(
        env_prefix="MONGO_",
        extra="ignore"
    )


class RedisSettings(BaseSettings):
    """Redis cache settings."""
    
    host: str = Field(default="localhost")
    port: int = Field(default=6379, ge=1, le=65535)
    db: int = Field(default=0, ge=0, le=15)
    password: Optional[str] = Field(default=None)
    cache_ttl_seconds: int = Field(default=3600, ge=60)
    
    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        extra="ignore"
    )
    
    @property
    def url(self) -> str:
        """Generate Redis URL."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class LLMSettings(BaseSettings):
    """LLM settings - defaults to OpenAI."""
    
    # LLM Provider selection
    provider: Literal["openai", "ollama", "anthropic"] = Field(
        default="openai",
        description="LLM provider to use"
    )
    
    # OpenAI settings (default)
    openai_api_key: Optional[str] = Field(default=None)
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model name")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model"
    )
    
    # Ollama settings (alternative)
    ollama_host: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2:3b")
    
    # Anthropic settings (alternative)
    anthropic_api_key: Optional[str] = Field(default=None)
    anthropic_model: str = Field(default="claude-3-haiku-20240307")
    
    # Common settings
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=100, le=8192)
    
    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        extra="ignore"
    )


class VectorStoreSettings(BaseSettings):
    """Vector store and embedding settings."""
    
    # Vector store provider selection
    provider: Literal["memory", "milvus"] = Field(
        default="milvus",
        description="Vector store provider to use"
    )
    
    # Milvus settings
    milvus_host: str = Field(default="localhost", description="Milvus host")
    milvus_port: int = Field(default=19530, description="Milvus port")
    milvus_collection: str = Field(
        default="enterprise_rag_docs",
        description="Milvus collection name"
    )
    milvus_index_type: str = Field(
        default="IVF_FLAT",
        description="Milvus index type (IVF_FLAT, IVF_SQ8, HNSW, etc.)"
    )
    milvus_metric_type: str = Field(
        default="COSINE",
        description="Milvus metric type (COSINE, L2, IP)"
    )
    milvus_nlist: int = Field(
        default=128,
        description="Number of cluster units for IVF index"
    )
    milvus_nprobe: int = Field(
        default=16,
        description="Number of units to query for IVF index"
    )
    
    # Embedding provider
    embedding_provider: Literal["openai", "sentence_transformers"] = Field(
        default="openai",
        description="Embedding provider to use"
    )
    
    # OpenAI embedding settings (default)
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model"
    )
    openai_embedding_dimension: int = Field(
        default=1536,
        description="OpenAI embedding dimension"
    )
    
    # Sentence Transformers settings (alternative)
    sentence_transformer_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence Transformer model"
    )
    sentence_transformer_dimension: int = Field(
        default=384,
        description="Sentence Transformer dimension"
    )
    
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    model_config = SettingsConfigDict(
        env_prefix="VECTOR_",
        extra="ignore"
    )
    
    @property
    def vector_dimension(self) -> int:
        """Get vector dimension based on provider."""
        if self.embedding_provider == "openai":
            return self.openai_embedding_dimension
        return self.sentence_transformer_dimension
    
    @property
    def embedding_model(self) -> str:
        """Get embedding model based on provider."""
        if self.embedding_provider == "openai":
            return self.openai_embedding_model
        return self.sentence_transformer_model


class AgentSettings(BaseSettings):
    """Enhanced multi-agent RAG settings."""
    
    # Core retrieval settings
    max_retrieval_attempts: int = Field(default=2, ge=1, le=5)
    guardrail_threshold: int = Field(default=60, ge=0, le=100)
    top_k_results: int = Field(default=5, ge=1, le=20)
    use_hybrid_search: bool = Field(default=True)
    
    # Multi-agent orchestration
    max_iterations: int = Field(default=5, ge=1, le=20, description="Max reasoning iterations")
    enable_multi_agent: bool = Field(default=True)
    agent_timeout_seconds: int = Field(default=60, ge=10, le=300)
    max_concurrent_agents: int = Field(default=5, ge=1, le=20)
    
    # Message bus for agent communication
    message_bus_type: Literal["memory", "redis"] = Field(
        default="memory",
        description="Message bus type for agent communication"
    )
    message_ttl_seconds: int = Field(default=300, ge=60, le=3600)
    
    # Query understanding
    enable_query_classification: bool = Field(default=True)
    enable_intent_routing: bool = Field(default=True)
    enable_language_detection: bool = Field(default=True)
    
    # Verification
    enable_verification: bool = Field(default=True)
    enable_citation_check: bool = Field(default=True)
    enable_fact_validation: bool = Field(default=True)
    verification_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # Self-reflection
    enable_self_reflection: bool = Field(default=True)
    reflection_iterations: int = Field(default=2, ge=1, le=5)
    
    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        extra="ignore"
    )


class RerankerSettings(BaseSettings):
    """Re-ranker configuration for improved retrieval accuracy."""
    
    enabled: bool = Field(default=True, description="Enable re-ranking")
    provider: Literal["cross_encoder", "cohere", "auto"] = Field(
        default="auto",
        description="Re-ranker provider (auto = try cohere, fallback to cross_encoder)"
    )
    
    # Cross-encoder settings (local/open-source)
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="HuggingFace cross-encoder model"
    )
    cross_encoder_device: str = Field(
        default="cpu",
        description="Device for cross-encoder (cpu, cuda, mps)"
    )
    cross_encoder_batch_size: int = Field(default=32, ge=1, le=128)
    
    # Cohere settings (API/third-party)
    cohere_api_key: Optional[str] = Field(default=None)
    cohere_model: str = Field(
        default="rerank-english-v3.0",
        description="Cohere rerank model"
    )
    
    # Common settings
    top_n: int = Field(default=5, ge=1, le=50, description="Number of results after re-ranking")
    score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    
    model_config = SettingsConfigDict(
        env_prefix="RERANKER_",
        extra="ignore"
    )


class ToolSettings(BaseSettings):
    """Tool layer configuration."""
    
    # Web Search settings (configurable providers)
    web_search_provider: Literal["tavily", "serp", "bing", "duckduckgo", "google"] = Field(
        default="tavily",
        description="Web search provider"
    )
    tavily_api_key: Optional[str] = Field(default=None)
    serp_api_key: Optional[str] = Field(default=None)
    bing_api_key: Optional[str] = Field(default=None)
    google_api_key: Optional[str] = Field(default=None)
    google_cse_id: Optional[str] = Field(default=None, description="Google Custom Search Engine ID")
    web_search_max_results: int = Field(default=5, ge=1, le=20)
    
    # Code Executor settings (containerized for K8s)
    code_executor_enabled: bool = Field(default=True)
    code_executor_runtime: Literal["docker", "kubernetes", "local"] = Field(
        default="docker",
        description="Container runtime for code execution"
    )
    code_executor_image: str = Field(
        default="python:3.11-slim",
        description="Docker image for code execution"
    )
    code_executor_timeout_seconds: int = Field(default=30, ge=5, le=300)
    code_executor_memory_limit: str = Field(default="256m")
    code_executor_cpu_limit: str = Field(default="0.5")
    code_executor_network_disabled: bool = Field(default=True)
    
    # Kubernetes-specific settings
    k8s_namespace: str = Field(default="agentic-rag", description="K8s namespace for jobs")
    k8s_service_account: str = Field(default="code-executor", description="K8s service account")
    k8s_image_pull_policy: str = Field(default="IfNotPresent")
    
    # Calculator settings
    calculator_precision: int = Field(default=10, ge=1, le=50)
    
    # Database tool settings
    db_tool_enabled: bool = Field(default=True)
    db_tool_max_rows: int = Field(default=100, ge=1, le=1000)
    db_tool_timeout_seconds: int = Field(default=30, ge=5, le=120)
    
    model_config = SettingsConfigDict(
        env_prefix="TOOL_",
        extra="ignore"
    )


class MemorySettings(BaseSettings):
    """Memory layer configuration (short-term Redis + long-term MongoDB)."""
    
    # Short-term memory (Redis-based, session-scoped)
    short_term_enabled: bool = Field(default=True)
    short_term_max_messages: int = Field(default=50, ge=10, le=200)
    short_term_ttl_seconds: int = Field(default=1800, description="30 minutes")
    
    # Long-term memory (MongoDB-based, persistent)
    long_term_enabled: bool = Field(default=True)
    long_term_collection: str = Field(default="user_memory")
    long_term_max_entries_per_user: int = Field(default=1000, ge=100, le=10000)
    
    # Memory retrieval
    memory_search_limit: int = Field(default=10, ge=1, le=50)
    memory_relevance_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    enable_semantic_memory_search: bool = Field(default=True)
    
    # Entity and preference extraction
    enable_entity_extraction: bool = Field(default=True)
    enable_preference_learning: bool = Field(default=True)
    preference_update_frequency: int = Field(default=10, description="Update after N interactions")
    
    # Memory consolidation
    enable_consolidation: bool = Field(default=True)
    consolidation_interval_hours: int = Field(default=24, ge=1, le=168)
    
    model_config = SettingsConfigDict(
        env_prefix="MEMORY_",
        extra="ignore"
    )


class FeedbackSettings(BaseSettings):
    """Feedback and learning loop configuration."""
    
    enabled: bool = Field(default=True)
    
    # Logging
    log_all_queries: bool = Field(default=True)
    log_retrieval_scores: bool = Field(default=True)
    log_agent_decisions: bool = Field(default=True)
    log_tool_usage: bool = Field(default=True)
    
    # Collections
    feedback_collection: str = Field(default="feedback")
    analytics_collection: str = Field(default="analytics")
    
    # Ratings
    enable_user_ratings: bool = Field(default=True)
    rating_scale: int = Field(default=5, ge=3, le=10)
    
    # Learning signals
    enable_learning_signals: bool = Field(default=True)
    learning_batch_size: int = Field(default=100, ge=10, le=1000)
    learning_update_interval_hours: int = Field(default=24, ge=1, le=168)
    
    # Metrics retention
    metrics_retention_days: int = Field(default=90, ge=7, le=365)
    
    model_config = SettingsConfigDict(
        env_prefix="FEEDBACK_",
        extra="ignore"
    )


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application
    app_name: str = Field(default="Enterprise Agentic RAG Platform")
    app_version: str = Field(default="2.0.0")
    app_env: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    
    # API Keys (top level for easy access)
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    cohere_api_key: Optional[str] = Field(default=None)
    tavily_api_key: Optional[str] = Field(default=None)
    
    # Security
    secret_key: str = Field(default="change-me-in-production")
    
    # CORS - stored as string, parsed in validator
    cors_origins_str: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        alias="CORS_ORIGINS"
    )
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100)
    rate_limit_window_seconds: int = Field(default=60)
    
    # Nested settings - initialized manually to avoid env parsing issues
    mongodb: MongoDBSettings = Field(default_factory=MongoDBSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    tools: ToolSettings = Field(default_factory=ToolSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    feedback: FeedbackSettings = Field(default_factory=FeedbackSettings)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True
    )
    
    @property
    def cors_origins(self) -> List[str]:
        """Parse CORS origins from string."""
        if not self.cors_origins_str:
            return ["http://localhost:3000", "http://localhost:5173"]
        return [origin.strip() for origin in self.cors_origins_str.split(",") if origin.strip()]
    
    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        return self.app_env.lower() == "development"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience function for direct access - but don't call at module load time
def get_settings_instance() -> Settings:
    """Get settings instance (non-cached)."""
    return Settings()