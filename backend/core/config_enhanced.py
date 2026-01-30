"""
Enhanced Application Configuration Module

Extended configuration for enterprise-grade Agentic RAG system.
Includes settings for:
- Multi-agent collaboration
- Tool execution (containerized)
- Re-ranking (local + cloud)
- Web search (configurable providers)
- Memory (short-term + long-term)
- Verification and feedback loops
"""

import os
from functools import lru_cache
from typing import List, Optional, Literal, Dict, Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# Re-Ranker Settings
# =============================================================================

class RerankerSettings(BaseSettings):
    """Re-ranker configuration supporting multiple providers."""
    
    # Provider selection
    provider: Literal["cross-encoder", "cohere", "both", "none"] = Field(
        default="cross-encoder",
        description="Re-ranker provider to use"
    )
    
    # Cross-encoder settings (local)
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="HuggingFace cross-encoder model"
    )
    cross_encoder_device: str = Field(
        default="cpu",
        description="Device for cross-encoder (cpu/cuda)"
    )
    cross_encoder_batch_size: int = Field(
        default=32,
        ge=1, le=256,
        description="Batch size for cross-encoder inference"
    )
    
    # Cohere settings (cloud)
    cohere_api_key: Optional[str] = Field(default=None)
    cohere_model: str = Field(
        default="rerank-english-v3.0",
        description="Cohere rerank model"
    )
    
    # Fusion settings (when using both)
    fusion_strategy: Literal["weighted", "rrf", "max"] = Field(
        default="weighted",
        description="Strategy to combine scores from multiple re-rankers"
    )
    cross_encoder_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    cohere_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Common settings
    top_k: int = Field(default=10, ge=1, le=100)
    score_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    
    model_config = SettingsConfigDict(
        env_prefix="RERANKER_",
        extra="ignore"
    )


# =============================================================================
# Tool Execution Settings
# =============================================================================

class ToolExecutionSettings(BaseSettings):
    """Tool execution configuration with container support."""
    
    # Container settings
    container_runtime: Literal["docker", "containerd", "kubernetes"] = Field(
        default="docker",
        description="Container runtime to use"
    )
    container_image: str = Field(
        default="python:3.11-slim",
        description="Base image for code execution"
    )
    container_timeout_seconds: int = Field(
        default=30,
        ge=5, le=300,
        description="Max execution time per tool invocation"
    )
    container_memory_limit: str = Field(
        default="256m",
        description="Memory limit for container"
    )
    container_cpu_limit: str = Field(
        default="0.5",
        description="CPU limit for container"
    )
    
    # Kubernetes settings
    k8s_namespace: str = Field(default="agentic-rag")
    k8s_service_account: str = Field(default="tool-executor")
    k8s_image_pull_policy: str = Field(default="IfNotPresent")
    
    # Security settings
    enable_network: bool = Field(
        default=False,
        description="Allow network access in code execution"
    )
    allowed_imports: List[str] = Field(
        default=["math", "statistics", "datetime", "json", "re", "collections"],
        description="Allowed Python imports in code execution"
    )
    max_output_size: int = Field(
        default=10000,
        description="Max output size in bytes"
    )
    
    # Tool registry
    enabled_tools: List[str] = Field(
        default=["search", "calculator", "database", "web_search", "code_executor"],
        description="List of enabled tools"
    )
    
    model_config = SettingsConfigDict(
        env_prefix="TOOL_",
        extra="ignore"
    )


# =============================================================================
# Web Search Settings
# =============================================================================

class WebSearchSettings(BaseSettings):
    """Configurable web search with multiple providers."""
    
    # Provider selection
    provider: Literal["tavily", "serp", "bing", "google", "duckduckgo"] = Field(
        default="tavily",
        description="Web search provider"
    )
    
    # Tavily settings
    tavily_api_key: Optional[str] = Field(default=None)
    tavily_search_depth: Literal["basic", "advanced"] = Field(default="basic")
    tavily_include_domains: List[str] = Field(default=[])
    tavily_exclude_domains: List[str] = Field(default=[])
    
    # SerpAPI settings
    serp_api_key: Optional[str] = Field(default=None)
    serp_engine: str = Field(default="google")
    
    # Bing settings
    bing_api_key: Optional[str] = Field(default=None)
    bing_endpoint: str = Field(
        default="https://api.bing.microsoft.com/v7.0/search"
    )
    
    # Google settings
    google_api_key: Optional[str] = Field(default=None)
    google_cse_id: Optional[str] = Field(default=None)
    
    # Common settings
    max_results: int = Field(default=5, ge=1, le=20)
    timeout_seconds: int = Field(default=10, ge=1, le=60)
    safe_search: bool = Field(default=True)
    
    model_config = SettingsConfigDict(
        env_prefix="WEB_SEARCH_",
        extra="ignore"
    )


# =============================================================================
# Memory Settings
# =============================================================================

class MemorySettings(BaseSettings):
    """Memory configuration for short-term and long-term storage."""
    
    # Short-term memory (Redis)
    short_term_enabled: bool = Field(default=True)
    short_term_ttl_seconds: int = Field(
        default=3600,
        description="Session memory TTL (1 hour default)"
    )
    short_term_max_messages: int = Field(
        default=50,
        description="Max messages per session"
    )
    
    # Long-term memory (MongoDB)
    long_term_enabled: bool = Field(default=True)
    long_term_collection: str = Field(default="user_memory")
    long_term_max_entries: int = Field(
        default=1000,
        description="Max memory entries per user"
    )
    
    # Memory extraction
    extract_entities: bool = Field(default=True)
    extract_preferences: bool = Field(default=True)
    extract_facts: bool = Field(default=True)
    
    # Memory retrieval
    semantic_search_enabled: bool = Field(default=True)
    retrieval_top_k: int = Field(default=5)
    relevance_threshold: float = Field(default=0.7)
    
    model_config = SettingsConfigDict(
        env_prefix="MEMORY_",
        extra="ignore"
    )


# =============================================================================
# Multi-Agent Settings
# =============================================================================

class MultiAgentSettings(BaseSettings):
    """Configuration for multi-agent collaboration system."""
    
    # Orchestration
    orchestration_mode: Literal["sequential", "parallel", "hierarchical", "dynamic"] = Field(
        default="dynamic",
        description="Agent orchestration mode"
    )
    max_iterations: int = Field(
        default=10,
        ge=1, le=50,
        description="Max reasoning iterations"
    )
    max_agent_calls: int = Field(
        default=20,
        ge=1, le=100,
        description="Max total agent invocations per query"
    )
    
    # Agent configuration
    enabled_agents: List[str] = Field(
        default=[
            "planner",
            "researcher",
            "retriever",
            "verifier",
            "responder"
        ],
        description="List of enabled agents"
    )
    
    # Message passing
    message_queue_type: Literal["memory", "redis", "rabbitmq"] = Field(
        default="memory",
        description="Message queue for agent communication"
    )
    message_timeout_seconds: int = Field(default=30)
    
    # Collaboration settings
    enable_agent_reflection: bool = Field(
        default=True,
        description="Allow agents to reflect on their outputs"
    )
    enable_agent_debate: bool = Field(
        default=False,
        description="Allow agents to debate/challenge each other"
    )
    consensus_threshold: float = Field(
        default=0.8,
        ge=0.5, le=1.0,
        description="Agreement threshold for multi-agent consensus"
    )
    
    # Performance
    parallel_execution: bool = Field(default=True)
    agent_timeout_seconds: int = Field(default=60)
    
    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        extra="ignore"
    )


# =============================================================================
# Verification Settings
# =============================================================================

class VerificationSettings(BaseSettings):
    """Configuration for answer verification and fact-checking."""
    
    # Verification pipeline
    enable_verification: bool = Field(default=True)
    verification_mode: Literal["strict", "moderate", "lenient"] = Field(
        default="moderate"
    )
    
    # Citation verification
    verify_citations: bool = Field(default=True)
    citation_match_threshold: float = Field(default=0.85)
    
    # Fact checking
    enable_fact_checking: bool = Field(default=True)
    fact_check_sources: List[str] = Field(
        default=["knowledge_base", "web_search"],
        description="Sources to use for fact checking"
    )
    
    # Consistency checking
    enable_consistency_check: bool = Field(default=True)
    consistency_threshold: float = Field(default=0.8)
    
    # Hallucination detection
    enable_hallucination_detection: bool = Field(default=True)
    hallucination_threshold: float = Field(default=0.7)
    
    # Confidence scoring
    min_confidence_threshold: float = Field(default=0.6)
    require_citation_for_facts: bool = Field(default=True)
    
    model_config = SettingsConfigDict(
        env_prefix="VERIFICATION_",
        extra="ignore"
    )


# =============================================================================
# Query Understanding Settings
# =============================================================================

class QueryUnderstandingSettings(BaseSettings):
    """Configuration for query understanding layer."""
    
    # Classification
    enable_classification: bool = Field(default=True)
    classification_categories: List[str] = Field(
        default=[
            "definition",
            "how_to",
            "comparison",
            "factual",
            "opinion",
            "calculation",
            "code",
            "policy",
            "troubleshooting"
        ]
    )
    
    # Intent detection
    enable_intent_detection: bool = Field(default=True)
    intent_confidence_threshold: float = Field(default=0.7)
    
    # Query rewriting
    enable_query_rewriting: bool = Field(default=True)
    max_rewrites: int = Field(default=3)
    
    # Query expansion
    enable_query_expansion: bool = Field(default=True)
    expansion_terms: int = Field(default=3)
    
    # Language detection
    enable_language_detection: bool = Field(default=True)
    supported_languages: List[str] = Field(
        default=["en", "es", "fr", "de", "zh", "ja"]
    )
    
    # Entity extraction
    enable_entity_extraction: bool = Field(default=True)
    
    model_config = SettingsConfigDict(
        env_prefix="QUERY_",
        extra="ignore"
    )


# =============================================================================
# Feedback Settings
# =============================================================================

class FeedbackSettings(BaseSettings):
    """Configuration for feedback and learning loop."""
    
    # Feedback collection
    enable_feedback: bool = Field(default=True)
    feedback_collection: str = Field(default="feedback_logs")
    
    # Rating system
    enable_ratings: bool = Field(default=True)
    rating_scale: int = Field(default=5, ge=1, le=10)
    
    # Analytics
    enable_analytics: bool = Field(default=True)
    analytics_collection: str = Field(default="analytics_logs")
    
    # Learning signals
    log_retrieval_accuracy: bool = Field(default=True)
    log_response_quality: bool = Field(default=True)
    log_tool_usage: bool = Field(default=True)
    log_agent_performance: bool = Field(default=True)
    
    # Retention
    feedback_retention_days: int = Field(default=90)
    analytics_retention_days: int = Field(default=365)
    
    model_config = SettingsConfigDict(
        env_prefix="FEEDBACK_",
        extra="ignore"
    )


# =============================================================================
# Prompt Management Settings
# =============================================================================

class PromptSettings(BaseSettings):
    """Configuration for prompt management."""
    
    # Prompt versioning
    enable_versioning: bool = Field(default=True)
    prompt_collection: str = Field(default="prompt_templates")
    
    # Safety
    enable_safety_prompts: bool = Field(default=True)
    enable_injection_protection: bool = Field(default=True)
    
    # Dynamic prompts
    enable_dynamic_prompts: bool = Field(default=True)
    
    # Template settings
    max_prompt_length: int = Field(default=4000)
    max_context_length: int = Field(default=8000)
    
    model_config = SettingsConfigDict(
        env_prefix="PROMPT_",
        extra="ignore"
    )


# =============================================================================
# Enhanced Main Settings
# =============================================================================

class EnhancedSettings(BaseSettings):
    """Enhanced main application settings with all new components."""
    
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
    
    # Security
    secret_key: str = Field(default="change-me-in-production")
    
    # CORS
    cors_origins_str: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        alias="CORS_ORIGINS"
    )
    
    # Rate Limiting
    rate_limit_requests: int = Field(default=100)
    rate_limit_window_seconds: int = Field(default=60)
    
    # Nested settings - all components
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    tool_execution: ToolExecutionSettings = Field(default_factory=ToolExecutionSettings)
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    multi_agent: MultiAgentSettings = Field(default_factory=MultiAgentSettings)
    verification: VerificationSettings = Field(default_factory=VerificationSettings)
    query_understanding: QueryUnderstandingSettings = Field(default_factory=QueryUnderstandingSettings)
    feedback: FeedbackSettings = Field(default_factory=FeedbackSettings)
    prompts: PromptSettings = Field(default_factory=PromptSettings)
    
    # Import original settings for backward compatibility
    from backend.core.config import (
        MongoDBSettings, RedisSettings, LLMSettings, 
        VectorStoreSettings, AgentSettings
    )
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


# Cache for settings instance
_enhanced_settings: Optional[EnhancedSettings] = None


def get_enhanced_settings() -> EnhancedSettings:
    """Get cached enhanced settings instance."""
    global _enhanced_settings
    if _enhanced_settings is None:
        _enhanced_settings = EnhancedSettings()
    return _enhanced_settings


def reload_settings() -> EnhancedSettings:
    """Force reload settings (useful for testing)."""
    global _enhanced_settings
    _enhanced_settings = EnhancedSettings()
    return _enhanced_settings
