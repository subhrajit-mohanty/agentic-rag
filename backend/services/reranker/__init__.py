"""
Re-ranker Service Module

Provides document re-ranking capabilities using:
- CrossEncoder (local, open-source HuggingFace models)
- Cohere Rerank API (third-party)
- Hybrid combination
"""

from .service import (
    BaseReranker,
    CrossEncoderReranker,
    CohereReranker,
    HybridReranker,
    MockReranker,
    RerankerFactory,
    RerankResult,
    get_reranker,
    set_reranker
)

__all__ = [
    "BaseReranker",
    "CrossEncoderReranker", 
    "CohereReranker",
    "HybridReranker",
    "MockReranker",
    "RerankerFactory",
    "RerankResult",
    "get_reranker",
    "set_reranker"
]
