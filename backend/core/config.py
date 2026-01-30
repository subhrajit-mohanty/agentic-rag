"""
Application Configuration Module

Centralized configuration management using Pydantic Settings.
Supports environment variables, .env files, and sensible defaults.
"""

import os
from functools import lru_cache
from typing import List, Optional, Literal

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
    """Agentic RAG settings."""
    
    max_retrieval_attempts: int = Field(default=2, ge=1, le=5)
    guardrail_threshold: int = Field(default=60, ge=0, le=100)
    top_k_results: int = Field(default=5, ge=1, le=20)
    use_hybrid_search: bool = Field(default=True)
    
    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        extra="ignore"
    )


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application
    app_name: str = Field(default="Enterprise Agentic RAG Platform")
    app_version: str = Field(default="1.0.0")
    app_env: str = Field(default="development")
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    
    # API Keys (top level for easy access)
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    
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