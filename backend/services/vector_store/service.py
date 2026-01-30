"""
Vector Store Service

Provides hybrid search (BM25 + Vector) capabilities using MongoDB
with OpenAI embeddings (default) or Sentence Transformers.
"""

import asyncio
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import AsyncOpenAI

from backend.core.config import get_settings
from backend.db.mongodb import db_manager
from backend.models.documents import KnowledgeDocument

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Embedding service supporting OpenAI (default) and Sentence Transformers.
    """
    
    def __init__(self):
        self._openai_client: Optional[AsyncOpenAI] = None
        self._sentence_transformer = None
        self._provider: Optional[str] = None
        self._model_loaded: bool = False
    
    async def initialize(self) -> None:
        """Initialize the embedding service based on configuration."""
        settings = get_settings()
        self._provider = settings.vector_store.embedding_provider
        
        if self._provider == "openai":
            await self._init_openai()
        else:
            await self._init_sentence_transformers()
    
    async def _init_openai(self) -> None:
        """Initialize OpenAI embeddings."""
        settings = get_settings()
        api_key = settings.openai_api_key or settings.llm.openai_api_key
        
        if not api_key:
            logger.warning("OpenAI API key not found, falling back to Sentence Transformers")
            await self._init_sentence_transformers()
            return
        
        self._openai_client = AsyncOpenAI(api_key=api_key)
        self._model_loaded = True
        logger.info(f"OpenAI embeddings initialized: {settings.vector_store.openai_embedding_model}")
    
    async def _init_sentence_transformers(self) -> None:
        """Initialize Sentence Transformers embeddings (fallback)."""
        settings = get_settings()
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading Sentence Transformer: {settings.vector_store.sentence_transformer_model}")
            self._sentence_transformer = SentenceTransformer(
                settings.vector_store.sentence_transformer_model
            )
            self._provider = "sentence_transformers"
            self._model_loaded = True
            logger.info("Sentence Transformer loaded successfully")
            
        except ImportError:
            logger.error("sentence-transformers not installed and OpenAI not available")
            self._model_loaded = False
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer: {e}")
            self._model_loaded = False
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not self._model_loaded:
            settings = get_settings()
            return [0.0] * settings.vector_store.vector_dimension
        
        if self._provider == "openai" and self._openai_client:
            return await self._embed_openai(text)
        elif self._sentence_transformer:
            return await self._embed_sentence_transformer(text)
        else:
            settings = get_settings()
            return [0.0] * settings.vector_store.vector_dimension
    
    async def _embed_openai(self, text: str) -> List[float]:
        """Generate embedding using OpenAI."""
        settings = get_settings()
        
        try:
            response = await self._openai_client.embeddings.create(
                model=settings.vector_store.openai_embedding_model,
                input=text
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            return [0.0] * settings.vector_store.openai_embedding_dimension
    
    async def _embed_sentence_transformer(self, text: str) -> List[float]:
        """Generate embedding using Sentence Transformers."""
        try:
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: self._sentence_transformer.encode(text, convert_to_numpy=True)
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Sentence Transformer embedding error: {e}")
            return []
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not self._model_loaded:
            settings = get_settings()
            return [[0.0] * settings.vector_store.vector_dimension for _ in texts]
        
        if self._provider == "openai" and self._openai_client:
            return await self._embed_batch_openai(texts)
        else:
            # Sentence Transformers - process sequentially
            return [await self.embed_text(text) for text in texts]
    
    async def _embed_batch_openai(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings using OpenAI."""
        settings = get_settings()
        
        try:
            response = await self._openai_client.embeddings.create(
                model=settings.vector_store.openai_embedding_model,
                input=texts
            )
            return [item.embedding for item in response.data]
            
        except Exception as e:
            logger.error(f"OpenAI batch embedding error: {e}")
            return [[0.0] * settings.vector_store.openai_embedding_dimension for _ in texts]
    
    @property
    def is_ready(self) -> bool:
        return self._model_loaded
    
    @property
    def provider(self) -> str:
        return self._provider or "none"


class VectorStoreService:
    """
    Hybrid search service combining BM25 keyword search with vector similarity.
    
    Features:
    - BM25 scoring for keyword relevance
    - Cosine similarity for semantic search (OpenAI embeddings by default)
    - Configurable fusion weights
    - MongoDB-backed document storage
    """
    
    def __init__(self):
        self._embedding_service = EmbeddingService()
        self._document_cache: Dict[str, Dict] = {}
        self._idf_cache: Dict[str, float] = {}
    
    async def initialize(self) -> None:
        """Initialize the vector store and embedding service."""
        try:
            await self._embedding_service.initialize()
            logger.info(f"Embedding service ready: {self._embedding_service.provider}")
            
            # Preload documents into cache
            await self._refresh_document_cache()
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            # Load mock data as fallback
            self._load_mock_data()
    
    async def _refresh_document_cache(self) -> None:
        """Refresh in-memory document cache from MongoDB."""
        try:
            documents = await KnowledgeDocument.find(
                KnowledgeDocument.is_active == True
            ).to_list()
            
            self._document_cache = {
                doc.document_id: {
                    "id": doc.document_id,
                    "content": doc.content,
                    "title": doc.title,
                    "metadata": doc.metadata.model_dump() if doc.metadata else {},
                    "embedding": doc.embedding.vector if doc.embedding else None
                }
                for doc in documents
            }
            
            self._build_idf_cache()
            logger.info(f"Loaded {len(self._document_cache)} documents into cache")
            
        except Exception as e:
            logger.warning(f"Failed to refresh document cache: {e}")
            self._load_mock_data()
    
    def _load_mock_data(self) -> None:
        """Load mock data for development/testing."""
        self._document_cache = {
            "doc_001": {
                "id": "doc_001",
                "content": "Employees are entitled to 20 days of paid time off per year. Annual leave can be carried over up to 5 days.",
                "title": "Employee Handbook - PTO Policy",
                "metadata": {"source": "SharePoint", "category": "HR"},
                "embedding": None
            },
            "doc_002": {
                "id": "doc_002", 
                "content": "Project Ares is the internal code name for the 2025 cloud migration initiative. All legacy systems will be moved to AWS.",
                "title": "Project Ares Specifications",
                "metadata": {"source": "Confluence", "category": "Engineering"},
                "embedding": None
            },
            "doc_003": {
                "id": "doc_003",
                "content": "S3 bucket access requires IAM role 'Engineering-Role'. All requests must be authenticated via SSO. Data encryption is mandatory.",
                "title": "Infrastructure Access Guide",
                "metadata": {"source": "S3", "category": "DevOps"},
                "embedding": None
            },
            "doc_004": {
                "id": "doc_004",
                "content": "Health insurance benefits include medical, dental, and vision coverage. Employees can add dependents during open enrollment.",
                "title": "Benefits Guide 2024",
                "metadata": {"source": "SharePoint", "category": "HR"},
                "embedding": None
            },
            "doc_005": {
                "id": "doc_005",
                "content": "Code reviews are mandatory for all pull requests. At least two approvals required before merging to main branch.",
                "title": "Engineering Standards",
                "metadata": {"source": "GitHub", "category": "Engineering"},
                "embedding": None
            },
        }
        self._build_idf_cache()
        logger.info(f"Loaded {len(self._document_cache)} mock documents")
    
    def _build_idf_cache(self) -> None:
        """Build IDF (Inverse Document Frequency) cache for BM25."""
        if not self._document_cache:
            return
        
        df: Dict[str, int] = {}
        for doc in self._document_cache.values():
            words = set(doc["content"].lower().split())
            for word in words:
                df[word] = df.get(word, 0) + 1
        
        n_docs = len(self._document_cache)
        self._idf_cache = {
            word: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1)
            for word, freq in df.items()
        }
    
    def _bm25_score(
        self,
        query: str,
        content: str,
        k1: float = 1.5,
        b: float = 0.75,
        avg_doc_len: float = 100.0
    ) -> float:
        """Calculate BM25 relevance score."""
        query_terms = query.lower().split()
        doc_terms = content.lower().split()
        doc_len = len(doc_terms)
        
        tf: Dict[str, int] = {}
        for term in doc_terms:
            tf[term] = tf.get(term, 0) + 1
        
        score = 0.0
        for term in query_terms:
            if len(term) < 2:
                continue
            
            term_tf = tf.get(term, 0)
            if term_tf == 0:
                continue
            
            idf = self._idf_cache.get(term, 0.5)
            numerator = term_tf * (k1 + 1)
            denominator = term_tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def _vector_similarity(
        self,
        query_embedding: List[float],
        doc_embedding: Optional[List[float]]
    ) -> float:
        """Calculate cosine similarity between query and document embeddings."""
        if doc_embedding is None or not query_embedding:
            return 0.0
        
        try:
            q = np.array(query_embedding)
            d = np.array(doc_embedding)
            
            dot_product = np.dot(q, d)
            norm_q = np.linalg.norm(q)
            norm_d = np.linalg.norm(d)
            
            if norm_q == 0 or norm_d == 0:
                return 0.0
            
            return float(dot_product / (norm_q * norm_d))
            
        except Exception:
            return 0.0
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using configured provider."""
        return await self._embedding_service.embed_text(text)
    
    async def search(
        self,
        query: str,
        limit: int = 5,
        use_hybrid: bool = True,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        min_score: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining BM25 and vector similarity.
        
        Args:
            query: Search query
            limit: Maximum number of results
            use_hybrid: Whether to use hybrid search (BM25 + vector)
            bm25_weight: Weight for BM25 score
            vector_weight: Weight for vector similarity
            min_score: Minimum score threshold
            
        Returns:
            List of search results with scores
        """
        if not self._document_cache:
            await self._refresh_document_cache()
        
        if not self._document_cache:
            logger.warning("No documents available for search")
            return []
        
        # Generate query embedding for vector search
        query_embedding = None
        if use_hybrid and self._embedding_service.is_ready:
            query_embedding = await self.embed_text(query)
        
        # Score all documents
        scored_results: List[Tuple[float, Dict]] = []
        
        for doc_id, doc in self._document_cache.items():
            bm25 = self._bm25_score(query, doc["content"])
            
            vector_sim = 0.0
            if use_hybrid and query_embedding:
                vector_sim = self._vector_similarity(query_embedding, doc.get("embedding"))
            
            if use_hybrid:
                final_score = (bm25_weight * bm25) + (vector_weight * vector_sim)
            else:
                final_score = bm25
            
            if final_score >= min_score:
                scored_results.append((final_score, {
                    "document_id": doc_id,
                    "title": doc["title"],
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": final_score,
                    "bm25_score": bm25,
                    "vector_score": vector_sim
                }))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [result[1] for result in scored_results[:limit]]
    
    async def search_text(
        self,
        query: str,
        limit: int = 5,
        use_hybrid: bool = True
    ) -> str:
        """Search and return concatenated text results."""
        results = await self.search(query, limit=limit, use_hybrid=use_hybrid)
        
        if not results:
            return "No relevant documents found in the knowledge base."
        
        return " | ".join([r["content"] for r in results])
    
    async def add_document(
        self,
        document_id: str,
        content: str,
        title: str,
        metadata: Dict[str, Any],
        generate_embedding: bool = True
    ) -> bool:
        """Add a document to the vector store."""
        try:
            embedding = None
            if generate_embedding and self._embedding_service.is_ready:
                embedding = await self.embed_text(content)
            
            self._document_cache[document_id] = {
                "id": document_id,
                "content": content,
                "title": title,
                "metadata": metadata,
                "embedding": embedding
            }
            
            self._build_idf_cache()
            
            logger.info(f"Added document {document_id} to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check vector store health."""
        settings = get_settings()
        
        return {
            "status": "healthy" if self._embedding_service.is_ready else "degraded",
            "embedding_provider": self._embedding_service.provider,
            "embedding_model": settings.vector_store.embedding_model,
            "documents_cached": len(self._document_cache),
            "idf_terms": len(self._idf_cache)
        }


# Global vector store instance
vector_store = VectorStoreService()


async def get_vector_store() -> VectorStoreService:
    """Dependency for FastAPI."""
    return vector_store


async def init_vector_store() -> None:
    """Initialize vector store."""
    await vector_store.initialize()
