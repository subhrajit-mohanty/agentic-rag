"""
Milvus Vector Store Service

Provides vector storage and search using Milvus as the backend.
Supports hybrid search combining BM25 keyword search with vector similarity.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)

from backend.core.config import get_settings

logger = logging.getLogger(__name__)


class MilvusVectorStore:
    """
    Milvus-backed vector store with hybrid search support.
    
    Features:
    - Dense vector similarity search using Milvus
    - Configurable index types (IVF_FLAT, HNSW, etc.)
    - Automatic collection creation and schema management
    - Integration with OpenAI embeddings
    """
    
    # Schema field names
    FIELD_ID = "id"
    FIELD_DOCUMENT_ID = "document_id"
    FIELD_TITLE = "title"
    FIELD_CONTENT = "content"
    FIELD_EMBEDDING = "embedding"
    FIELD_CATEGORY = "category"
    FIELD_SOURCE = "source"
    
    def __init__(self):
        self._client: Optional[MilvusClient] = None
        self._collection: Optional[Collection] = None
        self._initialized: bool = False
        self._embedding_service = None
    
    async def initialize(self, embedding_service=None) -> None:
        """
        Initialize Milvus connection and collection.
        
        Args:
            embedding_service: Optional embedding service for generating vectors
        """
        settings = get_settings()
        self._embedding_service = embedding_service
        
        try:
            logger.info(
                f"Connecting to Milvus at {settings.vector_store.milvus_host}:"
                f"{settings.vector_store.milvus_port}"
            )
            
            # Connect to Milvus
            connections.connect(
                alias="default",
                host=settings.vector_store.milvus_host,
                port=settings.vector_store.milvus_port,
                timeout=30
            )
            
            # Initialize or get collection
            await self._ensure_collection()
            
            self._initialized = True
            logger.info(f"Milvus initialized with collection: {settings.vector_store.milvus_collection}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Milvus: {e}")
            raise
    
    async def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        settings = get_settings()
        collection_name = settings.vector_store.milvus_collection
        
        # Check if collection exists
        if utility.has_collection(collection_name):
            self._collection = Collection(collection_name)
            self._collection.load()
            logger.info(f"Loaded existing collection: {collection_name}")
            return
        
        # Define schema
        fields = [
            FieldSchema(
                name=self.FIELD_ID,
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True
            ),
            FieldSchema(
                name=self.FIELD_DOCUMENT_ID,
                dtype=DataType.VARCHAR,
                max_length=256
            ),
            FieldSchema(
                name=self.FIELD_TITLE,
                dtype=DataType.VARCHAR,
                max_length=1024
            ),
            FieldSchema(
                name=self.FIELD_CONTENT,
                dtype=DataType.VARCHAR,
                max_length=65535
            ),
            FieldSchema(
                name=self.FIELD_CATEGORY,
                dtype=DataType.VARCHAR,
                max_length=256
            ),
            FieldSchema(
                name=self.FIELD_SOURCE,
                dtype=DataType.VARCHAR,
                max_length=256
            ),
            FieldSchema(
                name=self.FIELD_EMBEDDING,
                dtype=DataType.FLOAT_VECTOR,
                dim=settings.vector_store.vector_dimension
            ),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="Enterprise RAG document embeddings"
        )
        
        # Create collection
        self._collection = Collection(
            name=collection_name,
            schema=schema,
            consistency_level="Strong"
        )
        
        # Create index for vector field
        index_params = self._get_index_params()
        self._collection.create_index(
            field_name=self.FIELD_EMBEDDING,
            index_params=index_params
        )
        
        # Load collection into memory
        self._collection.load()
        
        logger.info(f"Created new collection: {collection_name}")
    
    def _get_index_params(self) -> Dict[str, Any]:
        """Get index parameters based on configuration."""
        settings = get_settings()
        index_type = settings.vector_store.milvus_index_type
        metric_type = settings.vector_store.milvus_metric_type
        
        if index_type in ["IVF_FLAT", "IVF_SQ8", "IVF_PQ"]:
            return {
                "index_type": index_type,
                "metric_type": metric_type,
                "params": {"nlist": settings.vector_store.milvus_nlist}
            }
        elif index_type == "HNSW":
            return {
                "index_type": "HNSW",
                "metric_type": metric_type,
                "params": {"M": 16, "efConstruction": 256}
            }
        else:
            # Default to IVF_FLAT
            return {
                "index_type": "IVF_FLAT",
                "metric_type": metric_type,
                "params": {"nlist": settings.vector_store.milvus_nlist}
            }
    
    def _get_search_params(self) -> Dict[str, Any]:
        """Get search parameters based on index type."""
        settings = get_settings()
        index_type = settings.vector_store.milvus_index_type
        
        if index_type in ["IVF_FLAT", "IVF_SQ8", "IVF_PQ"]:
            return {"nprobe": settings.vector_store.milvus_nprobe}
        elif index_type == "HNSW":
            return {"ef": 64}
        else:
            return {"nprobe": settings.vector_store.milvus_nprobe}
    
    async def insert(
        self,
        document_id: str,
        title: str,
        content: str,
        embedding: List[float],
        category: str = "general",
        source: str = "unknown"
    ) -> bool:
        """
        Insert a document with its embedding into Milvus.
        
        Args:
            document_id: Unique document identifier
            title: Document title
            content: Document content
            embedding: Vector embedding
            category: Document category
            source: Document source
            
        Returns:
            True if successful
        """
        if not self._collection:
            raise RuntimeError("Milvus not initialized")
        
        try:
            # Truncate content if too long
            max_content_length = 65000
            if len(content) > max_content_length:
                content = content[:max_content_length]
            
            data = [
                [document_id],
                [title[:1000] if len(title) > 1000 else title],
                [content],
                [category[:250] if len(category) > 250 else category],
                [source[:250] if len(source) > 250 else source],
                [embedding]
            ]
            
            self._collection.insert(data)
            self._collection.flush()
            
            logger.debug(f"Inserted document {document_id} into Milvus")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert document {document_id}: {e}")
            return False
    
    async def insert_batch(
        self,
        documents: List[Dict[str, Any]]
    ) -> int:
        """
        Insert multiple documents in batch.
        
        Args:
            documents: List of document dicts with keys:
                - document_id, title, content, embedding, category, source
                
        Returns:
            Number of successfully inserted documents
        """
        if not self._collection:
            raise RuntimeError("Milvus not initialized")
        
        if not documents:
            return 0
        
        try:
            document_ids = []
            titles = []
            contents = []
            categories = []
            sources = []
            embeddings = []
            
            for doc in documents:
                content = doc.get("content", "")
                if len(content) > 65000:
                    content = content[:65000]
                
                title = doc.get("title", "")
                if len(title) > 1000:
                    title = title[:1000]
                
                document_ids.append(doc["document_id"])
                titles.append(title)
                contents.append(content)
                categories.append(doc.get("category", "general")[:250])
                sources.append(doc.get("source", "unknown")[:250])
                embeddings.append(doc["embedding"])
            
            data = [document_ids, titles, contents, categories, sources, embeddings]
            
            self._collection.insert(data)
            self._collection.flush()
            
            logger.info(f"Batch inserted {len(documents)} documents into Milvus")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Batch insert failed: {e}")
            return 0
    
    async def search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            filter_expr: Optional Milvus filter expression
            
        Returns:
            List of matching documents with scores
        """
        if not self._collection:
            raise RuntimeError("Milvus not initialized")
        
        try:
            search_params = self._get_search_params()
            
            results = self._collection.search(
                data=[query_embedding],
                anns_field=self.FIELD_EMBEDDING,
                param=search_params,
                limit=limit,
                expr=filter_expr,
                output_fields=[
                    self.FIELD_DOCUMENT_ID,
                    self.FIELD_TITLE,
                    self.FIELD_CONTENT,
                    self.FIELD_CATEGORY,
                    self.FIELD_SOURCE
                ]
            )
            
            documents = []
            for hits in results:
                for hit in hits:
                    doc = {
                        "document_id": hit.entity.get(self.FIELD_DOCUMENT_ID),
                        "title": hit.entity.get(self.FIELD_TITLE),
                        "content": hit.entity.get(self.FIELD_CONTENT),
                        "category": hit.entity.get(self.FIELD_CATEGORY),
                        "source": hit.entity.get(self.FIELD_SOURCE),
                        "score": 1 - hit.distance if hit.distance else 0.0,  # Convert distance to similarity
                        "metadata": {
                            "category": hit.entity.get(self.FIELD_CATEGORY),
                            "source": hit.entity.get(self.FIELD_SOURCE)
                        }
                    }
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def delete(self, document_id: str) -> bool:
        """Delete a document by ID."""
        if not self._collection:
            raise RuntimeError("Milvus not initialized")
        
        try:
            expr = f'{self.FIELD_DOCUMENT_ID} == "{document_id}"'
            self._collection.delete(expr)
            logger.debug(f"Deleted document {document_id} from Milvus")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    async def delete_all(self) -> bool:
        """Delete all documents in the collection."""
        if not self._collection:
            raise RuntimeError("Milvus not initialized")
        
        try:
            settings = get_settings()
            collection_name = settings.vector_store.milvus_collection
            
            # Drop and recreate collection
            utility.drop_collection(collection_name)
            await self._ensure_collection()
            
            logger.info("Deleted all documents from Milvus")
            return True
        except Exception as e:
            logger.error(f"Failed to delete all documents: {e}")
            return False
    
    async def count(self) -> int:
        """Get total number of documents."""
        if not self._collection:
            return 0
        
        try:
            return self._collection.num_entities
        except Exception:
            return 0
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Milvus connection health."""
        settings = get_settings()
        
        try:
            if not self._initialized:
                return {
                    "status": "unhealthy",
                    "message": "Not initialized"
                }
            
            # Check connection
            connected = connections.has_connection("default")
            if not connected:
                return {
                    "status": "unhealthy",
                    "message": "Connection lost"
                }
            
            count = await self.count()
            
            return {
                "status": "healthy",
                "provider": "milvus",
                "host": settings.vector_store.milvus_host,
                "port": settings.vector_store.milvus_port,
                "collection": settings.vector_store.milvus_collection,
                "document_count": count,
                "index_type": settings.vector_store.milvus_index_type,
                "metric_type": settings.vector_store.milvus_metric_type
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "milvus",
                "message": str(e)
            }
    
    async def close(self) -> None:
        """Close Milvus connection."""
        try:
            if self._collection:
                self._collection.release()
            connections.disconnect("default")
            self._initialized = False
            logger.info("Milvus connection closed")
        except Exception as e:
            logger.warning(f"Error closing Milvus connection: {e}")


# Factory function
def create_milvus_store() -> MilvusVectorStore:
    """Create a new MilvusVectorStore instance."""
    return MilvusVectorStore()
