"""
Memory Layer for Agentic RAG

Implements dual-layer memory system:
- Short-term Memory: Session-based, Redis-backed, fast access
- Long-term Memory: Persistent, MongoDB-backed, semantic search enabled

Features:
- Automatic memory extraction from conversations
- Entity and fact extraction
- User preference tracking
- Contextual memory retrieval
- Memory consolidation
"""

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Memory Models
# =============================================================================

class MemoryType(str, Enum):
    """Types of memory entries."""
    FACT = "fact"               # Factual information
    PREFERENCE = "preference"   # User preferences
    ENTITY = "entity"           # Named entities
    CONTEXT = "context"         # Conversation context
    INTERACTION = "interaction" # Past interactions
    FEEDBACK = "feedback"       # User feedback


class MemoryEntry(BaseModel):
    """A single memory entry."""
    id: str
    type: MemoryType
    content: str
    source: str  # Where this memory came from
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    accessed_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    
    # Associations
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    query_id: Optional[str] = None
    
    # Metadata
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # For semantic search
    embedding: Optional[List[float]] = None


class MemoryQuery(BaseModel):
    """Query for retrieving memories."""
    query: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    memory_types: List[MemoryType] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    min_confidence: float = 0.0
    limit: int = 10
    use_semantic: bool = True


class MemoryContext(BaseModel):
    """Memory context for agent execution."""
    short_term: List[MemoryEntry] = Field(default_factory=list)
    long_term: List[MemoryEntry] = Field(default_factory=list)
    entities: Dict[str, Any] = Field(default_factory=dict)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    recent_queries: List[str] = Field(default_factory=list)


# =============================================================================
# Base Memory Store
# =============================================================================

class BaseMemoryStore(ABC):
    """Abstract base class for memory stores."""
    
    @abstractmethod
    async def store(self, entry: MemoryEntry) -> bool:
        """Store a memory entry."""
        pass
    
    @abstractmethod
    async def retrieve(self, query: MemoryQuery) -> List[MemoryEntry]:
        """Retrieve memory entries matching query."""
        pass
    
    @abstractmethod
    async def update(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory entry."""
        pass
    
    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        pass
    
    @abstractmethod
    async def clear(self, user_id: Optional[str] = None, session_id: Optional[str] = None) -> int:
        """Clear memory entries."""
        pass


# =============================================================================
# Short-Term Memory (Redis)
# =============================================================================

class ShortTermMemory(BaseMemoryStore):
    """
    Redis-backed short-term memory.
    
    Features:
    - Fast access
    - TTL-based expiration
    - Session isolation
    - Recent message history
    """
    
    def __init__(
        self,
        redis_client: Any = None,
        ttl_seconds: int = 3600,
        max_messages: int = 50,
        prefix: str = "stm"
    ):
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds
        self.max_messages = max_messages
        self.prefix = prefix
        
        # In-memory fallback when Redis unavailable
        self._fallback: Dict[str, List[MemoryEntry]] = {}
    
    def _key(self, *parts: str) -> str:
        """Build Redis key."""
        return f"{self.prefix}:{':'.join(parts)}"
    
    async def store(self, entry: MemoryEntry) -> bool:
        """Store memory entry in Redis."""
        try:
            if self.redis is None:
                return self._store_fallback(entry)
            
            # Build key based on session/user
            if entry.session_id:
                key = self._key("session", entry.session_id, entry.id)
            elif entry.user_id:
                key = self._key("user", entry.user_id, entry.id)
            else:
                key = self._key("global", entry.id)
            
            # Serialize and store
            data = entry.model_dump_json()
            
            if entry.expires_at:
                ttl = int((entry.expires_at - datetime.utcnow()).total_seconds())
            else:
                ttl = self.ttl_seconds
            
            await self.redis.setex(key, ttl, data)
            
            # Also add to session list for quick retrieval
            if entry.session_id:
                list_key = self._key("session_list", entry.session_id)
                await self.redis.lpush(list_key, entry.id)
                await self.redis.ltrim(list_key, 0, self.max_messages - 1)
                await self.redis.expire(list_key, self.ttl_seconds)
            
            logger.debug(f"Stored short-term memory: {entry.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store short-term memory: {e}")
            return self._store_fallback(entry)
    
    def _store_fallback(self, entry: MemoryEntry) -> bool:
        """Store in fallback memory."""
        key = entry.session_id or entry.user_id or "global"
        
        if key not in self._fallback:
            self._fallback[key] = []
        
        self._fallback[key].append(entry)
        
        # Limit size
        if len(self._fallback[key]) > self.max_messages:
            self._fallback[key] = self._fallback[key][-self.max_messages:]
        
        return True
    
    async def retrieve(self, query: MemoryQuery) -> List[MemoryEntry]:
        """Retrieve memory entries."""
        try:
            if self.redis is None:
                return self._retrieve_fallback(query)
            
            entries = []
            
            # Get from session
            if query.session_id:
                list_key = self._key("session_list", query.session_id)
                entry_ids = await self.redis.lrange(list_key, 0, query.limit - 1)
                
                for entry_id in entry_ids:
                    key = self._key("session", query.session_id, entry_id.decode())
                    data = await self.redis.get(key)
                    
                    if data:
                        entry = MemoryEntry.model_validate_json(data)
                        
                        # Apply filters
                        if self._matches_query(entry, query):
                            entries.append(entry)
            
            # Get from user if session not specified
            elif query.user_id:
                pattern = self._key("user", query.user_id, "*")
                keys = await self.redis.keys(pattern)
                
                for key in keys[:query.limit]:
                    data = await self.redis.get(key)
                    if data:
                        entry = MemoryEntry.model_validate_json(data)
                        if self._matches_query(entry, query):
                            entries.append(entry)
            
            return entries[:query.limit]
            
        except Exception as e:
            logger.error(f"Failed to retrieve short-term memory: {e}")
            return self._retrieve_fallback(query)
    
    def _retrieve_fallback(self, query: MemoryQuery) -> List[MemoryEntry]:
        """Retrieve from fallback memory."""
        key = query.session_id or query.user_id or "global"
        entries = self._fallback.get(key, [])
        
        # Apply filters
        filtered = [e for e in entries if self._matches_query(e, query)]
        
        return filtered[:query.limit]
    
    def _matches_query(self, entry: MemoryEntry, query: MemoryQuery) -> bool:
        """Check if entry matches query filters."""
        if query.memory_types and entry.type not in query.memory_types:
            return False
        
        if query.min_confidence > 0 and entry.confidence < query.min_confidence:
            return False
        
        if query.tags:
            if not any(tag in entry.tags for tag in query.tags):
                return False
        
        return True
    
    async def update(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory entry."""
        # Short-term memory doesn't support updates - entries are immutable
        return False
    
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        try:
            if self.redis:
                pattern = self._key("*", "*", entry_id)
                keys = await self.redis.keys(pattern)
                if keys:
                    await self.redis.delete(*keys)
                    return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete short-term memory: {e}")
            return False
    
    async def clear(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> int:
        """Clear memory entries."""
        try:
            if self.redis is None:
                key = session_id or user_id or "global"
                count = len(self._fallback.get(key, []))
                self._fallback[key] = []
                return count
            
            count = 0
            
            if session_id:
                pattern = self._key("session", session_id, "*")
            elif user_id:
                pattern = self._key("user", user_id, "*")
            else:
                pattern = self._key("*")
            
            keys = await self.redis.keys(pattern)
            if keys:
                count = await self.redis.delete(*keys)
            
            return count
            
        except Exception as e:
            logger.error(f"Failed to clear short-term memory: {e}")
            return 0
    
    async def get_session_history(
        self,
        session_id: str,
        limit: int = 20
    ) -> List[MemoryEntry]:
        """Get recent conversation history for a session."""
        query = MemoryQuery(
            session_id=session_id,
            memory_types=[MemoryType.CONTEXT, MemoryType.INTERACTION],
            limit=limit
        )
        return await self.retrieve(query)


# =============================================================================
# Long-Term Memory (MongoDB)
# =============================================================================

class LongTermMemory(BaseMemoryStore):
    """
    MongoDB-backed long-term memory.
    
    Features:
    - Persistent storage
    - Semantic search with embeddings
    - User profiles
    - Cross-session context
    """
    
    def __init__(
        self,
        collection: Any = None,
        embedding_service: Any = None,
        max_entries_per_user: int = 1000
    ):
        self.collection = collection
        self.embedding_service = embedding_service
        self.max_entries_per_user = max_entries_per_user
        
        # In-memory fallback
        self._fallback: Dict[str, List[MemoryEntry]] = {}
    
    async def store(self, entry: MemoryEntry) -> bool:
        """Store memory entry in MongoDB."""
        try:
            if self.collection is None:
                return self._store_fallback(entry)
            
            # Generate embedding if service available
            if self.embedding_service and not entry.embedding:
                entry.embedding = await self.embedding_service.embed_text(entry.content)
            
            # Convert to dict for MongoDB
            doc = entry.model_dump()
            doc["_id"] = entry.id
            
            # Upsert
            await self.collection.replace_one(
                {"_id": entry.id},
                doc,
                upsert=True
            )
            
            # Enforce max entries per user
            if entry.user_id:
                await self._enforce_limit(entry.user_id)
            
            logger.debug(f"Stored long-term memory: {entry.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store long-term memory: {e}")
            return self._store_fallback(entry)
    
    def _store_fallback(self, entry: MemoryEntry) -> bool:
        """Store in fallback memory."""
        key = entry.user_id or "global"
        
        if key not in self._fallback:
            self._fallback[key] = []
        
        # Check for existing entry
        existing = next(
            (i for i, e in enumerate(self._fallback[key]) if e.id == entry.id),
            None
        )
        
        if existing is not None:
            self._fallback[key][existing] = entry
        else:
            self._fallback[key].append(entry)
        
        # Enforce limit
        if len(self._fallback[key]) > self.max_entries_per_user:
            self._fallback[key] = self._fallback[key][-self.max_entries_per_user:]
        
        return True
    
    async def _enforce_limit(self, user_id: str) -> None:
        """Enforce max entries per user."""
        try:
            count = await self.collection.count_documents({"user_id": user_id})
            
            if count > self.max_entries_per_user:
                # Delete oldest entries
                excess = count - self.max_entries_per_user
                oldest = await self.collection.find(
                    {"user_id": user_id}
                ).sort("created_at", 1).limit(excess).to_list(excess)
                
                ids_to_delete = [doc["_id"] for doc in oldest]
                await self.collection.delete_many({"_id": {"$in": ids_to_delete}})
                
                logger.info(f"Pruned {excess} old memories for user {user_id}")
                
        except Exception as e:
            logger.error(f"Failed to enforce memory limit: {e}")
    
    async def retrieve(self, query: MemoryQuery) -> List[MemoryEntry]:
        """Retrieve memory entries."""
        try:
            if self.collection is None:
                return self._retrieve_fallback(query)
            
            # Build MongoDB query
            mongo_query: Dict[str, Any] = {}
            
            if query.user_id:
                mongo_query["user_id"] = query.user_id
            
            if query.memory_types:
                mongo_query["type"] = {"$in": [t.value for t in query.memory_types]}
            
            if query.min_confidence > 0:
                mongo_query["confidence"] = {"$gte": query.min_confidence}
            
            if query.tags:
                mongo_query["tags"] = {"$in": query.tags}
            
            # Use semantic search if query text provided and embeddings available
            if query.query and query.use_semantic and self.embedding_service:
                return await self._semantic_search(query, mongo_query)
            
            # Regular query
            cursor = self.collection.find(mongo_query).sort(
                "accessed_at", -1
            ).limit(query.limit)
            
            docs = await cursor.to_list(length=query.limit)
            
            entries = []
            for doc in docs:
                doc.pop("_id", None)
                entries.append(MemoryEntry.model_validate(doc))
            
            return entries
            
        except Exception as e:
            logger.error(f"Failed to retrieve long-term memory: {e}")
            return self._retrieve_fallback(query)
    
    async def _semantic_search(
        self,
        query: MemoryQuery,
        base_filter: Dict[str, Any]
    ) -> List[MemoryEntry]:
        """Perform semantic search on memories."""
        # Generate query embedding
        query_embedding = await self.embedding_service.embed_text(query.query)
        
        # Use MongoDB vector search if available (Atlas Search)
        # For simplicity, we'll do a brute-force search here
        # In production, use $vectorSearch aggregation
        
        cursor = self.collection.find(base_filter)
        docs = await cursor.to_list(length=1000)
        
        # Calculate similarities
        scored_docs = []
        for doc in docs:
            if doc.get("embedding"):
                similarity = self._cosine_similarity(
                    query_embedding,
                    doc["embedding"]
                )
                scored_docs.append((similarity, doc))
        
        # Sort by similarity
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Return top results
        entries = []
        for score, doc in scored_docs[:query.limit]:
            doc.pop("_id", None)
            entry = MemoryEntry.model_validate(doc)
            entry.metadata["similarity_score"] = score
            entries.append(entry)
        
        return entries
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _retrieve_fallback(self, query: MemoryQuery) -> List[MemoryEntry]:
        """Retrieve from fallback memory."""
        key = query.user_id or "global"
        entries = self._fallback.get(key, [])
        
        # Apply filters
        filtered = []
        for entry in entries:
            if query.memory_types and entry.type not in query.memory_types:
                continue
            if query.min_confidence > 0 and entry.confidence < query.min_confidence:
                continue
            if query.tags and not any(tag in entry.tags for tag in query.tags):
                continue
            filtered.append(entry)
        
        return filtered[:query.limit]
    
    async def update(self, entry_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory entry."""
        try:
            if self.collection is None:
                return False
            
            updates["accessed_at"] = datetime.utcnow()
            
            result = await self.collection.update_one(
                {"_id": entry_id},
                {"$set": updates}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Failed to update long-term memory: {e}")
            return False
    
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry."""
        try:
            if self.collection is None:
                return False
            
            result = await self.collection.delete_one({"_id": entry_id})
            return result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"Failed to delete long-term memory: {e}")
            return False
    
    async def clear(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> int:
        """Clear memory entries."""
        try:
            if self.collection is None:
                key = user_id or "global"
                count = len(self._fallback.get(key, []))
                self._fallback[key] = []
                return count
            
            filter_query = {}
            if user_id:
                filter_query["user_id"] = user_id
            if session_id:
                filter_query["session_id"] = session_id
            
            result = await self.collection.delete_many(filter_query)
            return result.deleted_count
            
        except Exception as e:
            logger.error(f"Failed to clear long-term memory: {e}")
            return 0


# =============================================================================
# Memory Manager
# =============================================================================

class MemoryManager:
    """
    Unified memory manager combining short-term and long-term memory.
    
    Features:
    - Automatic memory extraction from conversations
    - Memory consolidation (short-term → long-term)
    - Contextual retrieval
    - User profile management
    """
    
    def __init__(
        self,
        short_term: Optional[ShortTermMemory] = None,
        long_term: Optional[LongTermMemory] = None,
        llm_client: Any = None
    ):
        self.short_term = short_term or ShortTermMemory()
        self.long_term = long_term or LongTermMemory()
        self.llm_client = llm_client
    
    async def store(
        self,
        content: str,
        memory_type: MemoryType,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tags: List[str] = None,
        long_term: bool = False,
        **kwargs
    ) -> str:
        """
        Store a memory entry.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            user_id: User identifier
            session_id: Session identifier
            tags: Tags for categorization
            long_term: Whether to store in long-term memory
            
        Returns:
            Memory entry ID
        """
        entry_id = hashlib.md5(
            f"{content}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        entry = MemoryEntry(
            id=entry_id,
            type=memory_type,
            content=content,
            source="explicit",
            user_id=user_id,
            session_id=session_id,
            tags=tags or [],
            metadata=kwargs
        )
        
        # Always store in short-term
        await self.short_term.store(entry)
        
        # Optionally store in long-term
        if long_term:
            await self.long_term.store(entry)
        
        return entry_id
    
    async def get_context(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        include_long_term: bool = True
    ) -> MemoryContext:
        """
        Get relevant memory context for a query.
        
        Args:
            query: Current query
            user_id: User identifier
            session_id: Session identifier
            include_long_term: Whether to include long-term memories
            
        Returns:
            MemoryContext with relevant memories
        """
        context = MemoryContext()
        
        # Get short-term memories
        stm_query = MemoryQuery(
            session_id=session_id,
            user_id=user_id,
            limit=20
        )
        context.short_term = await self.short_term.retrieve(stm_query)
        
        # Get long-term memories if requested
        if include_long_term and user_id:
            ltm_query = MemoryQuery(
                query=query,
                user_id=user_id,
                use_semantic=True,
                limit=10
            )
            context.long_term = await self.long_term.retrieve(ltm_query)
            
            # Extract entities and preferences
            context.entities = await self._get_entities(user_id)
            context.preferences = await self._get_preferences(user_id)
        
        # Get recent queries
        recent_queries = [
            m.content for m in context.short_term
            if m.type == MemoryType.INTERACTION
        ][:5]
        context.recent_queries = recent_queries
        
        return context
    
    async def _get_entities(self, user_id: str) -> Dict[str, Any]:
        """Get known entities for a user."""
        query = MemoryQuery(
            user_id=user_id,
            memory_types=[MemoryType.ENTITY],
            limit=50
        )
        
        entries = await self.long_term.retrieve(query)
        
        entities = {}
        for entry in entries:
            entity_name = entry.metadata.get("entity_name", entry.content[:50])
            entities[entity_name] = {
                "content": entry.content,
                "tags": entry.tags,
                "confidence": entry.confidence
            }
        
        return entities
    
    async def _get_preferences(self, user_id: str) -> Dict[str, Any]:
        """Get user preferences."""
        query = MemoryQuery(
            user_id=user_id,
            memory_types=[MemoryType.PREFERENCE],
            limit=50
        )
        
        entries = await self.long_term.retrieve(query)
        
        preferences = {}
        for entry in entries:
            pref_key = entry.metadata.get("preference_key", entry.tags[0] if entry.tags else "general")
            preferences[pref_key] = entry.content
        
        return preferences
    
    async def extract_and_store(
        self,
        conversation: List[Dict[str, str]],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[str]:
        """
        Extract memories from conversation and store them.
        
        Uses LLM to extract:
        - Facts mentioned
        - User preferences
        - Named entities
        
        Args:
            conversation: List of {"role": "user/assistant", "content": "..."}
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            List of stored memory IDs
        """
        if not self.llm_client or not conversation:
            return []
        
        stored_ids = []
        
        # Build conversation text
        conv_text = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in conversation[-10:]  # Last 10 messages
        ])
        
        # Extract memories using LLM
        prompt = f"""Analyze this conversation and extract important information to remember.

Conversation:
{conv_text}

Extract:
1. Facts: Specific factual information mentioned by the user
2. Preferences: User preferences, likes, dislikes
3. Entities: Important named entities (people, places, projects, etc.)

Output as JSON:
{{
    "facts": ["fact1", "fact2"],
    "preferences": [{{"key": "topic", "value": "preference"}}],
    "entities": [{{"name": "entity", "type": "person/place/project", "context": "brief context"}}]
}}
"""
        
        try:
            response = await self.llm_client.generate(prompt)
            
            # Parse JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                
                # Store facts
                for fact in data.get("facts", []):
                    entry_id = await self.store(
                        content=fact,
                        memory_type=MemoryType.FACT,
                        user_id=user_id,
                        session_id=session_id,
                        long_term=True
                    )
                    stored_ids.append(entry_id)
                
                # Store preferences
                for pref in data.get("preferences", []):
                    entry_id = await self.store(
                        content=pref.get("value", ""),
                        memory_type=MemoryType.PREFERENCE,
                        user_id=user_id,
                        session_id=session_id,
                        tags=[pref.get("key", "general")],
                        long_term=True,
                        preference_key=pref.get("key")
                    )
                    stored_ids.append(entry_id)
                
                # Store entities
                for entity in data.get("entities", []):
                    entry_id = await self.store(
                        content=entity.get("context", entity.get("name", "")),
                        memory_type=MemoryType.ENTITY,
                        user_id=user_id,
                        session_id=session_id,
                        tags=[entity.get("type", "unknown")],
                        long_term=True,
                        entity_name=entity.get("name")
                    )
                    stored_ids.append(entry_id)
                
                logger.info(f"Extracted and stored {len(stored_ids)} memories")
                
        except Exception as e:
            logger.error(f"Failed to extract memories: {e}")
        
        return stored_ids
    
    async def consolidate(
        self,
        user_id: str,
        max_age_hours: int = 24
    ) -> int:
        """
        Consolidate short-term memories to long-term.
        
        Args:
            user_id: User to consolidate for
            max_age_hours: Only consolidate memories older than this
            
        Returns:
            Number of memories consolidated
        """
        # Get old short-term memories
        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        query = MemoryQuery(
            user_id=user_id,
            limit=100
        )
        
        entries = await self.short_term.retrieve(query)
        
        consolidated = 0
        for entry in entries:
            if entry.created_at < cutoff:
                # Move to long-term
                await self.long_term.store(entry)
                await self.short_term.delete(entry.id)
                consolidated += 1
        
        logger.info(f"Consolidated {consolidated} memories for user {user_id}")
        return consolidated


# =============================================================================
# Factory Functions
# =============================================================================

def create_memory_manager(
    redis_client: Any = None,
    mongodb_collection: Any = None,
    embedding_service: Any = None,
    llm_client: Any = None
) -> MemoryManager:
    """Create a memory manager with configured stores."""
    short_term = ShortTermMemory(redis_client=redis_client)
    long_term = LongTermMemory(
        collection=mongodb_collection,
        embedding_service=embedding_service
    )
    
    return MemoryManager(
        short_term=short_term,
        long_term=long_term,
        llm_client=llm_client
    )
