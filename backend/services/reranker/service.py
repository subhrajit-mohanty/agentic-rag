"""
Re-Ranker Service

Provides re-ranking capabilities using multiple providers:
- Cross-Encoder (local, open-source): Uses HuggingFace sentence-transformers
- Cohere Rerank (API, third-party): Uses Cohere's rerank API

Re-ranking significantly improves retrieval accuracy by reordering
documents based on their relevance to the query using more sophisticated
models than the initial retrieval.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Result from re-ranking operation."""
    document_id: str
    content: str
    original_score: float
    rerank_score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseReranker(ABC):
    """Abstract base class for re-rankers."""
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Re-rank documents based on their relevance to the query.
        
        Args:
            query: The search query
            documents: List of documents with 'content' and optionally 'document_id', 'score'
            top_n: Number of top results to return (None = all)
            
        Returns:
            List of RerankResult sorted by relevance (highest first)
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the re-ranker is available and working."""
        pass
    
    def _extract_content(self, doc: Dict[str, Any]) -> str:
        """Extract text content from a document."""
        # Try various fields that might contain the content
        for field in ['content', 'text', 'body', 'passage', 'chunk']:
            if field in doc and doc[field]:
                return str(doc[field])
        return str(doc)
    
    def _extract_id(self, doc: Dict[str, Any], index: int) -> str:
        """Extract document ID or generate one."""
        for field in ['document_id', 'id', 'doc_id', '_id']:
            if field in doc and doc[field]:
                return str(doc[field])
        return f"doc_{index}"


class CrossEncoderReranker(BaseReranker):
    """
    Cross-Encoder Re-ranker using HuggingFace sentence-transformers.
    
    Uses a cross-encoder model that jointly encodes query and document
    to produce a relevance score. More accurate than bi-encoder but slower.
    
    Recommended models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good quality)
    - cross-encoder/ms-marco-MiniLM-L-12-v2 (better quality)
    - cross-encoder/ms-marco-TinyBERT-L-2-v2 (fastest)
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 512
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self._initialized = False
        
        logger.info(f"CrossEncoderReranker configured with model: {model_name}")
    
    async def _initialize(self) -> None:
        """Lazy initialization of the model."""
        if self._initialized:
            return
        
        try:
            # Import here to avoid loading if not used
            from sentence_transformers import CrossEncoder
            
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: CrossEncoder(
                    self.model_name,
                    max_length=self.max_length,
                    device=self.device
                )
            )
            self._initialized = True
            logger.info(f"CrossEncoder model loaded: {self.model_name}")
            
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model: {e}")
            raise
    
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: Optional[int] = None
    ) -> List[RerankResult]:
        """Re-rank documents using cross-encoder."""
        if not documents:
            return []
        
        await self._initialize()
        
        # Prepare query-document pairs
        pairs = []
        doc_info = []
        
        for i, doc in enumerate(documents):
            content = self._extract_content(doc)
            pairs.append([query, content])
            doc_info.append({
                'id': self._extract_id(doc, i),
                'content': content,
                'original_score': doc.get('score', 0.0),
                'metadata': {k: v for k, v in doc.items() if k not in ['content', 'text', 'score']}
            })
        
        # Score in batches
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: self._model.predict(pairs, batch_size=self.batch_size)
        )
        
        # Create results with scores
        results = []
        for i, (info, score) in enumerate(zip(doc_info, scores)):
            results.append(RerankResult(
                document_id=info['id'],
                content=info['content'],
                original_score=info['original_score'],
                rerank_score=float(score),
                rank=0,  # Will be set after sorting
                metadata=info['metadata']
            ))
        
        # Sort by rerank score (descending)
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Assign ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        # Return top_n if specified
        if top_n is not None:
            results = results[:top_n]
        
        logger.debug(f"CrossEncoder reranked {len(documents)} docs, returning {len(results)}")
        return results
    
    async def health_check(self) -> bool:
        """Check if cross-encoder is available."""
        try:
            await self._initialize()
            return self._model is not None
        except Exception:
            return False


class CohereReranker(BaseReranker):
    """
    Cohere Re-ranker using Cohere's Rerank API.
    
    High-quality reranking using Cohere's models.
    Requires API key from https://cohere.com
    
    Models:
    - rerank-english-v3.0 (recommended for English)
    - rerank-multilingual-v3.0 (for multilingual)
    - rerank-english-v2.0 (legacy)
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "rerank-english-v3.0",
        timeout: float = 30.0
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        
        logger.info(f"CohereReranker configured with model: {model}")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            )
        return self._client
    
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: Optional[int] = None
    ) -> List[RerankResult]:
        """Re-rank documents using Cohere API."""
        if not documents:
            return []
        
        # Prepare documents for API
        doc_texts = []
        doc_info = []
        
        for i, doc in enumerate(documents):
            content = self._extract_content(doc)
            doc_texts.append(content)
            doc_info.append({
                'id': self._extract_id(doc, i),
                'content': content,
                'original_score': doc.get('score', 0.0),
                'metadata': {k: v for k, v in doc.items() if k not in ['content', 'text', 'score']}
            })
        
        # Call Cohere API
        client = await self._get_client()
        
        payload = {
            "model": self.model,
            "query": query,
            "documents": doc_texts,
            "return_documents": False  # We already have them
        }
        
        if top_n is not None:
            payload["top_n"] = top_n
        
        try:
            response = await client.post(
                "https://api.cohere.com/v1/rerank",
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Cohere API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Cohere rerank failed: {e}")
            raise
        
        # Process results
        results = []
        for rank, item in enumerate(data.get("results", []), 1):
            idx = item["index"]
            info = doc_info[idx]
            
            results.append(RerankResult(
                document_id=info['id'],
                content=info['content'],
                original_score=info['original_score'],
                rerank_score=item["relevance_score"],
                rank=rank,
                metadata=info['metadata']
            ))
        
        logger.debug(f"Cohere reranked {len(documents)} docs, returning {len(results)}")
        return results
    
    async def health_check(self) -> bool:
        """Check if Cohere API is accessible."""
        try:
            client = await self._get_client()
            # Simple test with minimal payload
            response = await client.post(
                "https://api.cohere.com/v1/rerank",
                json={
                    "model": self.model,
                    "query": "test",
                    "documents": ["test document"]
                }
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Cohere health check failed: {e}")
            return False
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class HybridReranker(BaseReranker):
    """
    Hybrid Re-ranker that combines multiple rerankers.
    
    Can use both CrossEncoder and Cohere, then combine scores
    using various strategies (max, average, weighted).
    """
    
    def __init__(
        self,
        rerankers: List[BaseReranker],
        strategy: str = "max",  # max, average, weighted
        weights: Optional[List[float]] = None
    ):
        self.rerankers = rerankers
        self.strategy = strategy
        self.weights = weights or [1.0] * len(rerankers)
        
        if len(self.weights) != len(self.rerankers):
            raise ValueError("Number of weights must match number of rerankers")
    
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: Optional[int] = None
    ) -> List[RerankResult]:
        """Re-rank using multiple rerankers and combine scores."""
        if not documents:
            return []
        
        # Get results from all rerankers
        all_results = await asyncio.gather(
            *[r.rerank(query, documents, None) for r in self.rerankers],
            return_exceptions=True
        )
        
        # Filter out failures
        valid_results = [r for r in all_results if isinstance(r, list)]
        
        if not valid_results:
            logger.error("All rerankers failed")
            return []
        
        # Build score map: doc_id -> list of scores
        score_map: Dict[str, List[Tuple[float, RerankResult]]] = {}
        
        for reranker_results, weight in zip(valid_results, self.weights):
            for result in reranker_results:
                if result.document_id not in score_map:
                    score_map[result.document_id] = []
                score_map[result.document_id].append((result.rerank_score * weight, result))
        
        # Combine scores
        combined_results = []
        for doc_id, scores_and_results in score_map.items():
            scores = [s[0] for s in scores_and_results]
            base_result = scores_and_results[0][1]
            
            if self.strategy == "max":
                combined_score = max(scores)
            elif self.strategy == "average":
                combined_score = sum(scores) / len(scores)
            elif self.strategy == "weighted":
                combined_score = sum(scores) / sum(self.weights[:len(scores)])
            else:
                combined_score = max(scores)
            
            combined_results.append(RerankResult(
                document_id=base_result.document_id,
                content=base_result.content,
                original_score=base_result.original_score,
                rerank_score=combined_score,
                rank=0,
                metadata=base_result.metadata
            ))
        
        # Sort and assign ranks
        combined_results.sort(key=lambda x: x.rerank_score, reverse=True)
        for i, result in enumerate(combined_results):
            result.rank = i + 1
        
        if top_n is not None:
            combined_results = combined_results[:top_n]
        
        return combined_results
    
    async def health_check(self) -> bool:
        """Check if at least one reranker is available."""
        checks = await asyncio.gather(
            *[r.health_check() for r in self.rerankers],
            return_exceptions=True
        )
        return any(c is True for c in checks)


class MockReranker(BaseReranker):
    """Mock re-ranker for testing without actual models."""
    
    def __init__(self, latency_ms: float = 50.0):
        self.latency_ms = latency_ms
    
    async def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: Optional[int] = None
    ) -> List[RerankResult]:
        """Mock reranking with simulated latency."""
        await asyncio.sleep(self.latency_ms / 1000)
        
        results = []
        for i, doc in enumerate(documents):
            content = self._extract_content(doc)
            # Simple heuristic: score based on query term overlap
            query_terms = set(query.lower().split())
            doc_terms = set(content.lower().split())
            overlap = len(query_terms & doc_terms)
            score = min(overlap / max(len(query_terms), 1), 1.0)
            
            results.append(RerankResult(
                document_id=self._extract_id(doc, i),
                content=content,
                original_score=doc.get('score', 0.0),
                rerank_score=score,
                rank=0,
                metadata={}
            ))
        
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        for i, result in enumerate(results):
            result.rank = i + 1
        
        if top_n is not None:
            results = results[:top_n]
        
        return results
    
    async def health_check(self) -> bool:
        return True


class RerankerFactory:
    """Factory for creating re-ranker instances."""
    
    @staticmethod
    def create(
        provider: str = "auto",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        cross_encoder_device: str = "cpu",
        cohere_api_key: Optional[str] = None,
        cohere_model: str = "rerank-english-v3.0",
        **kwargs
    ) -> BaseReranker:
        """
        Create a re-ranker based on provider setting.
        
        Args:
            provider: "cross_encoder", "cohere", "auto", or "mock"
            cross_encoder_model: Model name for cross-encoder
            cross_encoder_device: Device for cross-encoder (cpu, cuda)
            cohere_api_key: API key for Cohere
            cohere_model: Model name for Cohere
            
        Returns:
            Configured re-ranker instance
        """
        if provider == "mock":
            return MockReranker()
        
        if provider == "cross_encoder":
            return CrossEncoderReranker(
                model_name=cross_encoder_model,
                device=cross_encoder_device
            )
        
        if provider == "cohere":
            if not cohere_api_key:
                raise ValueError("Cohere API key required for cohere provider")
            return CohereReranker(
                api_key=cohere_api_key,
                model=cohere_model
            )
        
        if provider == "auto":
            # Try Cohere first if API key available, else cross-encoder
            if cohere_api_key:
                logger.info("Auto-selecting Cohere reranker (API key provided)")
                return CohereReranker(
                    api_key=cohere_api_key,
                    model=cohere_model
                )
            else:
                logger.info("Auto-selecting CrossEncoder reranker (no Cohere API key)")
                return CrossEncoderReranker(
                    model_name=cross_encoder_model,
                    device=cross_encoder_device
                )
        
        if provider == "hybrid":
            rerankers = []
            if cohere_api_key:
                rerankers.append(CohereReranker(api_key=cohere_api_key, model=cohere_model))
            rerankers.append(CrossEncoderReranker(
                model_name=cross_encoder_model,
                device=cross_encoder_device
            ))
            return HybridReranker(rerankers)
        
        raise ValueError(f"Unknown reranker provider: {provider}")


# Singleton instance
_reranker_instance: Optional[BaseReranker] = None


def get_reranker(
    provider: str = "auto",
    **kwargs
) -> BaseReranker:
    """Get or create the global reranker instance."""
    global _reranker_instance
    
    if _reranker_instance is None:
        _reranker_instance = RerankerFactory.create(provider=provider, **kwargs)
    
    return _reranker_instance


def set_reranker(reranker: BaseReranker) -> None:
    """Set a custom reranker instance."""
    global _reranker_instance
    _reranker_instance = reranker
