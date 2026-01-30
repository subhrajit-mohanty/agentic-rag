"""
Feedback and Learning Loop Service for Agentic RAG

Implements continuous improvement mechanisms:
- User feedback collection (ratings, corrections)
- Retrieval accuracy logging
- Response quality tracking
- Agent performance metrics
- Learning signals for system improvement

This enables the RAG system to improve over time based on
user interactions and feedback.
"""

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Feedback Types and Models
# =============================================================================

class FeedbackType(str, Enum):
    """Types of feedback."""
    RATING = "rating"                  # Numeric rating (1-5)
    THUMBS = "thumbs"                  # Thumbs up/down
    CORRECTION = "correction"          # User corrected the answer
    RETRIEVAL = "retrieval"            # Feedback on retrieved docs
    COMMENT = "comment"                # Free-text feedback
    IMPLICIT = "implicit"              # Implicit signals (time on page, etc.)


class FeedbackSentiment(str, Enum):
    """Feedback sentiment."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class FeedbackEntry(BaseModel):
    """A single feedback entry."""
    id: str
    type: FeedbackType
    sentiment: FeedbackSentiment
    
    # Context
    query_id: str
    query: str
    answer: str
    
    # Feedback data
    rating: Optional[int] = Field(default=None, ge=1, le=5)
    thumbs_up: Optional[bool] = None
    correction: Optional[str] = None
    comment: Optional[str] = None
    
    # Retrieval feedback
    relevant_docs: List[str] = Field(default_factory=list)
    irrelevant_docs: List[str] = Field(default_factory=list)
    missing_info: Optional[str] = None
    
    # Metadata
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnalyticsEvent(BaseModel):
    """Analytics event for tracking system behavior."""
    id: str
    event_type: str
    
    # Event data
    query_id: Optional[str] = None
    agent_id: Optional[str] = None
    tool_name: Optional[str] = None
    
    # Metrics
    duration_ms: Optional[float] = None
    success: bool = True
    error: Optional[str] = None
    
    # Context
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PerformanceMetrics(BaseModel):
    """Aggregated performance metrics."""
    period_start: datetime
    period_end: datetime
    
    # Query metrics
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    avg_response_time_ms: float = 0.0
    
    # Feedback metrics
    total_feedback: int = 0
    positive_feedback: int = 0
    negative_feedback: int = 0
    avg_rating: float = 0.0
    
    # Retrieval metrics
    avg_docs_retrieved: float = 0.0
    avg_relevance_score: float = 0.0
    cache_hit_rate: float = 0.0
    
    # Agent metrics
    agent_calls: Dict[str, int] = Field(default_factory=dict)
    agent_success_rates: Dict[str, float] = Field(default_factory=dict)
    
    # Tool metrics
    tool_usage: Dict[str, int] = Field(default_factory=dict)
    tool_success_rates: Dict[str, float] = Field(default_factory=dict)


# =============================================================================
# Feedback Collector
# =============================================================================

class FeedbackCollector:
    """
    Collects and stores user feedback.
    
    Supports multiple feedback types and stores for analysis.
    """
    
    def __init__(
        self,
        mongodb_collection: Any = None,
        max_entries: int = 100000
    ):
        self.collection = mongodb_collection
        self.max_entries = max_entries
        
        # In-memory fallback
        self._fallback: List[FeedbackEntry] = []
    
    async def submit_rating(
        self,
        query_id: str,
        query: str,
        answer: str,
        rating: int,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        comment: Optional[str] = None
    ) -> str:
        """Submit a rating feedback."""
        entry = FeedbackEntry(
            id=self._generate_id(query_id, "rating"),
            type=FeedbackType.RATING,
            sentiment=self._rating_to_sentiment(rating),
            query_id=query_id,
            query=query,
            answer=answer,
            rating=rating,
            comment=comment,
            user_id=user_id,
            session_id=session_id
        )
        
        return await self._store(entry)
    
    async def submit_thumbs(
        self,
        query_id: str,
        query: str,
        answer: str,
        thumbs_up: bool,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        comment: Optional[str] = None
    ) -> str:
        """Submit thumbs up/down feedback."""
        entry = FeedbackEntry(
            id=self._generate_id(query_id, "thumbs"),
            type=FeedbackType.THUMBS,
            sentiment=FeedbackSentiment.POSITIVE if thumbs_up else FeedbackSentiment.NEGATIVE,
            query_id=query_id,
            query=query,
            answer=answer,
            thumbs_up=thumbs_up,
            comment=comment,
            user_id=user_id,
            session_id=session_id
        )
        
        return await self._store(entry)
    
    async def submit_correction(
        self,
        query_id: str,
        query: str,
        original_answer: str,
        corrected_answer: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """Submit an answer correction."""
        entry = FeedbackEntry(
            id=self._generate_id(query_id, "correction"),
            type=FeedbackType.CORRECTION,
            sentiment=FeedbackSentiment.NEGATIVE,  # Correction implies issue
            query_id=query_id,
            query=query,
            answer=original_answer,
            correction=corrected_answer,
            user_id=user_id,
            session_id=session_id
        )
        
        return await self._store(entry)
    
    async def submit_retrieval_feedback(
        self,
        query_id: str,
        query: str,
        answer: str,
        relevant_docs: List[str] = None,
        irrelevant_docs: List[str] = None,
        missing_info: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> str:
        """Submit feedback on retrieved documents."""
        entry = FeedbackEntry(
            id=self._generate_id(query_id, "retrieval"),
            type=FeedbackType.RETRIEVAL,
            sentiment=FeedbackSentiment.NEUTRAL,
            query_id=query_id,
            query=query,
            answer=answer,
            relevant_docs=relevant_docs or [],
            irrelevant_docs=irrelevant_docs or [],
            missing_info=missing_info,
            user_id=user_id
        )
        
        return await self._store(entry)
    
    def _generate_id(self, query_id: str, feedback_type: str) -> str:
        """Generate feedback ID."""
        data = f"{query_id}:{feedback_type}:{datetime.utcnow().isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def _rating_to_sentiment(self, rating: int) -> FeedbackSentiment:
        """Convert rating to sentiment."""
        if rating >= 4:
            return FeedbackSentiment.POSITIVE
        elif rating <= 2:
            return FeedbackSentiment.NEGATIVE
        return FeedbackSentiment.NEUTRAL
    
    async def _store(self, entry: FeedbackEntry) -> str:
        """Store feedback entry."""
        try:
            if self.collection is not None:
                doc = entry.model_dump()
                doc["_id"] = entry.id
                await self.collection.replace_one(
                    {"_id": entry.id},
                    doc,
                    upsert=True
                )
            else:
                self._fallback.append(entry)
                if len(self._fallback) > self.max_entries:
                    self._fallback = self._fallback[-self.max_entries:]
            
            logger.info(f"Stored feedback: {entry.id} ({entry.type})")
            return entry.id
            
        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")
            self._fallback.append(entry)
            return entry.id
    
    async def get_feedback(
        self,
        query_id: Optional[str] = None,
        feedback_type: Optional[FeedbackType] = None,
        sentiment: Optional[FeedbackSentiment] = None,
        limit: int = 100
    ) -> List[FeedbackEntry]:
        """Retrieve feedback entries."""
        try:
            if self.collection is not None:
                query = {}
                if query_id:
                    query["query_id"] = query_id
                if feedback_type:
                    query["type"] = feedback_type.value
                if sentiment:
                    query["sentiment"] = sentiment.value
                
                cursor = self.collection.find(query).limit(limit).sort("timestamp", -1)
                docs = await cursor.to_list(length=limit)
                
                return [FeedbackEntry.model_validate(doc) for doc in docs]
            else:
                # Filter fallback
                entries = self._fallback
                if query_id:
                    entries = [e for e in entries if e.query_id == query_id]
                if feedback_type:
                    entries = [e for e in entries if e.type == feedback_type]
                if sentiment:
                    entries = [e for e in entries if e.sentiment == sentiment]
                
                return entries[-limit:]
                
        except Exception as e:
            logger.error(f"Failed to get feedback: {e}")
            return []


# =============================================================================
# Analytics Logger
# =============================================================================

class AnalyticsLogger:
    """
    Logs analytics events for system monitoring.
    
    Tracks:
    - Query execution
    - Agent performance
    - Tool usage
    - Error rates
    """
    
    def __init__(
        self,
        mongodb_collection: Any = None,
        retention_days: int = 90
    ):
        self.collection = mongodb_collection
        self.retention_days = retention_days
        
        # In-memory buffer
        self._buffer: List[AnalyticsEvent] = []
        self._buffer_size = 100
    
    async def log_query(
        self,
        query_id: str,
        query: str,
        success: bool,
        duration_ms: float,
        cache_hit: bool = False,
        docs_retrieved: int = 0,
        agents_used: List[str] = None,
        **kwargs
    ) -> None:
        """Log a query execution."""
        event = AnalyticsEvent(
            id=self._generate_id("query"),
            event_type="query_execution",
            query_id=query_id,
            success=success,
            duration_ms=duration_ms,
            data={
                "query": query[:500],  # Truncate for storage
                "cache_hit": cache_hit,
                "docs_retrieved": docs_retrieved,
                "agents_used": agents_used or [],
                **kwargs
            }
        )
        
        await self._log(event)
    
    async def log_agent_execution(
        self,
        agent_id: str,
        query_id: str,
        success: bool,
        duration_ms: float,
        error: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log agent execution."""
        event = AnalyticsEvent(
            id=self._generate_id("agent"),
            event_type="agent_execution",
            query_id=query_id,
            agent_id=agent_id,
            success=success,
            duration_ms=duration_ms,
            error=error,
            data=kwargs
        )
        
        await self._log(event)
    
    async def log_tool_usage(
        self,
        tool_name: str,
        query_id: str,
        success: bool,
        duration_ms: float,
        error: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log tool usage."""
        event = AnalyticsEvent(
            id=self._generate_id("tool"),
            event_type="tool_usage",
            query_id=query_id,
            tool_name=tool_name,
            success=success,
            duration_ms=duration_ms,
            error=error,
            data=kwargs
        )
        
        await self._log(event)
    
    async def log_retrieval(
        self,
        query_id: str,
        strategy: str,
        docs_retrieved: int,
        avg_score: float,
        duration_ms: float,
        **kwargs
    ) -> None:
        """Log retrieval operation."""
        event = AnalyticsEvent(
            id=self._generate_id("retrieval"),
            event_type="retrieval",
            query_id=query_id,
            success=docs_retrieved > 0,
            duration_ms=duration_ms,
            data={
                "strategy": strategy,
                "docs_retrieved": docs_retrieved,
                "avg_score": avg_score,
                **kwargs
            }
        )
        
        await self._log(event)
    
    async def log_error(
        self,
        error_type: str,
        error_message: str,
        query_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log an error."""
        event = AnalyticsEvent(
            id=self._generate_id("error"),
            event_type="error",
            query_id=query_id,
            agent_id=agent_id,
            success=False,
            error=error_message,
            data={
                "error_type": error_type,
                **kwargs
            }
        )
        
        await self._log(event)
    
    def _generate_id(self, prefix: str) -> str:
        """Generate event ID."""
        data = f"{prefix}:{datetime.utcnow().isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    async def _log(self, event: AnalyticsEvent) -> None:
        """Log an event."""
        self._buffer.append(event)
        
        # Flush buffer if full
        if len(self._buffer) >= self._buffer_size:
            await self._flush()
    
    async def _flush(self) -> None:
        """Flush buffer to storage."""
        if not self._buffer:
            return
        
        try:
            if self.collection is not None:
                docs = [e.model_dump() for e in self._buffer]
                for doc in docs:
                    doc["_id"] = doc["id"]
                await self.collection.insert_many(docs)
            
            self._buffer.clear()
            
        except Exception as e:
            logger.error(f"Failed to flush analytics: {e}")
    
    async def close(self) -> None:
        """Flush remaining events on close."""
        await self._flush()


# =============================================================================
# Metrics Aggregator
# =============================================================================

class MetricsAggregator:
    """
    Aggregates metrics for reporting and analysis.
    
    Provides:
    - Real-time metrics
    - Historical trends
    - Performance dashboards
    """
    
    def __init__(
        self,
        feedback_collection: Any = None,
        analytics_collection: Any = None
    ):
        self.feedback_collection = feedback_collection
        self.analytics_collection = analytics_collection
        
        # Cached metrics
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 60  # seconds
        self._cache_time: Optional[datetime] = None
    
    async def get_metrics(
        self,
        period_hours: int = 24
    ) -> PerformanceMetrics:
        """Get aggregated metrics for a period."""
        # Check cache
        cache_key = f"metrics_{period_hours}"
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]
        
        period_end = datetime.utcnow()
        period_start = period_end - timedelta(hours=period_hours)
        
        metrics = PerformanceMetrics(
            period_start=period_start,
            period_end=period_end
        )
        
        # Aggregate from analytics
        if self.analytics_collection:
            metrics = await self._aggregate_analytics(metrics, period_start, period_end)
        
        # Aggregate from feedback
        if self.feedback_collection:
            metrics = await self._aggregate_feedback(metrics, period_start, period_end)
        
        # Cache result
        self._cache[cache_key] = metrics
        self._cache_time = datetime.utcnow()
        
        return metrics
    
    async def _aggregate_analytics(
        self,
        metrics: PerformanceMetrics,
        start: datetime,
        end: datetime
    ) -> PerformanceMetrics:
        """Aggregate from analytics collection."""
        try:
            # Query execution metrics
            pipeline = [
                {"$match": {
                    "event_type": "query_execution",
                    "timestamp": {"$gte": start, "$lte": end}
                }},
                {"$group": {
                    "_id": None,
                    "total": {"$sum": 1},
                    "successful": {"$sum": {"$cond": ["$success", 1, 0]}},
                    "avg_duration": {"$avg": "$duration_ms"},
                    "cache_hits": {"$sum": {"$cond": ["$data.cache_hit", 1, 0]}},
                    "total_docs": {"$sum": "$data.docs_retrieved"}
                }}
            ]
            
            cursor = self.analytics_collection.aggregate(pipeline)
            results = await cursor.to_list(length=1)
            
            if results:
                r = results[0]
                metrics.total_queries = r.get("total", 0)
                metrics.successful_queries = r.get("successful", 0)
                metrics.failed_queries = metrics.total_queries - metrics.successful_queries
                metrics.avg_response_time_ms = r.get("avg_duration", 0)
                
                if metrics.total_queries > 0:
                    metrics.cache_hit_rate = r.get("cache_hits", 0) / metrics.total_queries
                    metrics.avg_docs_retrieved = r.get("total_docs", 0) / metrics.total_queries
            
            # Agent metrics
            agent_pipeline = [
                {"$match": {
                    "event_type": "agent_execution",
                    "timestamp": {"$gte": start, "$lte": end}
                }},
                {"$group": {
                    "_id": "$agent_id",
                    "calls": {"$sum": 1},
                    "successful": {"$sum": {"$cond": ["$success", 1, 0]}}
                }}
            ]
            
            cursor = self.analytics_collection.aggregate(agent_pipeline)
            agent_results = await cursor.to_list(length=100)
            
            for r in agent_results:
                agent_id = r.get("_id")
                if agent_id:
                    metrics.agent_calls[agent_id] = r.get("calls", 0)
                    calls = r.get("calls", 1)
                    metrics.agent_success_rates[agent_id] = r.get("successful", 0) / calls
            
            # Tool metrics
            tool_pipeline = [
                {"$match": {
                    "event_type": "tool_usage",
                    "timestamp": {"$gte": start, "$lte": end}
                }},
                {"$group": {
                    "_id": "$tool_name",
                    "calls": {"$sum": 1},
                    "successful": {"$sum": {"$cond": ["$success", 1, 0]}}
                }}
            ]
            
            cursor = self.analytics_collection.aggregate(tool_pipeline)
            tool_results = await cursor.to_list(length=100)
            
            for r in tool_results:
                tool_name = r.get("_id")
                if tool_name:
                    metrics.tool_usage[tool_name] = r.get("calls", 0)
                    calls = r.get("calls", 1)
                    metrics.tool_success_rates[tool_name] = r.get("successful", 0) / calls
            
        except Exception as e:
            logger.error(f"Failed to aggregate analytics: {e}")
        
        return metrics
    
    async def _aggregate_feedback(
        self,
        metrics: PerformanceMetrics,
        start: datetime,
        end: datetime
    ) -> PerformanceMetrics:
        """Aggregate from feedback collection."""
        try:
            pipeline = [
                {"$match": {
                    "timestamp": {"$gte": start, "$lte": end}
                }},
                {"$group": {
                    "_id": None,
                    "total": {"$sum": 1},
                    "positive": {"$sum": {"$cond": [
                        {"$eq": ["$sentiment", "positive"]}, 1, 0
                    ]}},
                    "negative": {"$sum": {"$cond": [
                        {"$eq": ["$sentiment", "negative"]}, 1, 0
                    ]}},
                    "avg_rating": {"$avg": "$rating"}
                }}
            ]
            
            cursor = self.feedback_collection.aggregate(pipeline)
            results = await cursor.to_list(length=1)
            
            if results:
                r = results[0]
                metrics.total_feedback = r.get("total", 0)
                metrics.positive_feedback = r.get("positive", 0)
                metrics.negative_feedback = r.get("negative", 0)
                metrics.avg_rating = r.get("avg_rating", 0) or 0
            
        except Exception as e:
            logger.error(f"Failed to aggregate feedback: {e}")
        
        return metrics
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached value is still valid."""
        if key not in self._cache:
            return False
        if self._cache_time is None:
            return False
        
        age = (datetime.utcnow() - self._cache_time).total_seconds()
        return age < self._cache_ttl


# =============================================================================
# Learning Signal Generator
# =============================================================================

class LearningSignalGenerator:
    """
    Generates learning signals for system improvement.
    
    Identifies:
    - Queries that need improvement
    - Successful patterns to reinforce
    - Prompt adjustments needed
    - Retrieval tuning recommendations
    """
    
    def __init__(
        self,
        feedback_collector: FeedbackCollector,
        analytics_logger: AnalyticsLogger,
        metrics_aggregator: MetricsAggregator
    ):
        self.feedback = feedback_collector
        self.analytics = analytics_logger
        self.metrics = metrics_aggregator
    
    async def get_improvement_recommendations(self) -> List[Dict[str, Any]]:
        """Generate improvement recommendations based on feedback and metrics."""
        recommendations = []
        
        # Get recent metrics
        metrics = await self.metrics.get_metrics(period_hours=24)
        
        # Check overall satisfaction
        if metrics.avg_rating < 3.5 and metrics.total_feedback >= 10:
            recommendations.append({
                "type": "quality",
                "priority": "high",
                "issue": "Low average rating",
                "current_value": metrics.avg_rating,
                "target_value": 4.0,
                "recommendation": "Review recent negative feedback for patterns"
            })
        
        # Check response time
        if metrics.avg_response_time_ms > 5000:
            recommendations.append({
                "type": "performance",
                "priority": "high",
                "issue": "Slow response time",
                "current_value": metrics.avg_response_time_ms,
                "target_value": 3000,
                "recommendation": "Enable caching, optimize retrieval, or reduce agent iterations"
            })
        
        # Check cache hit rate
        if metrics.cache_hit_rate < 0.3 and metrics.total_queries >= 100:
            recommendations.append({
                "type": "performance",
                "priority": "medium",
                "issue": "Low cache hit rate",
                "current_value": metrics.cache_hit_rate,
                "target_value": 0.5,
                "recommendation": "Increase cache TTL or improve query normalization"
            })
        
        # Check agent success rates
        for agent_id, success_rate in metrics.agent_success_rates.items():
            if success_rate < 0.9 and metrics.agent_calls.get(agent_id, 0) >= 50:
                recommendations.append({
                    "type": "agent",
                    "priority": "medium",
                    "issue": f"Low success rate for agent '{agent_id}'",
                    "current_value": success_rate,
                    "target_value": 0.95,
                    "recommendation": f"Review and improve {agent_id} agent prompts or logic"
                })
        
        return recommendations
    
    async def get_queries_for_improvement(
        self,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get queries that received negative feedback."""
        negative_feedback = await self.feedback.get_feedback(
            sentiment=FeedbackSentiment.NEGATIVE,
            limit=limit
        )
        
        return [
            {
                "query": f.query,
                "answer": f.answer,
                "feedback_type": f.type,
                "correction": f.correction,
                "comment": f.comment,
                "timestamp": f.timestamp
            }
            for f in negative_feedback
        ]


# =============================================================================
# Feedback Service (Main Interface)
# =============================================================================

class FeedbackService:
    """
    Main interface for the feedback and learning system.
    
    Combines all components for easy use.
    """
    
    def __init__(
        self,
        feedback_collection: Any = None,
        analytics_collection: Any = None
    ):
        self.collector = FeedbackCollector(feedback_collection)
        self.analytics = AnalyticsLogger(analytics_collection)
        self.metrics = MetricsAggregator(feedback_collection, analytics_collection)
        self.learning = LearningSignalGenerator(
            self.collector,
            self.analytics,
            self.metrics
        )
        
        logger.info("FeedbackService initialized")
    
    # Feedback methods
    async def submit_rating(self, **kwargs) -> str:
        return await self.collector.submit_rating(**kwargs)
    
    async def submit_thumbs(self, **kwargs) -> str:
        return await self.collector.submit_thumbs(**kwargs)
    
    async def submit_correction(self, **kwargs) -> str:
        return await self.collector.submit_correction(**kwargs)
    
    async def submit_retrieval_feedback(self, **kwargs) -> str:
        return await self.collector.submit_retrieval_feedback(**kwargs)
    
    # Analytics methods
    async def log_query(self, **kwargs) -> None:
        await self.analytics.log_query(**kwargs)
    
    async def log_agent_execution(self, **kwargs) -> None:
        await self.analytics.log_agent_execution(**kwargs)
    
    async def log_tool_usage(self, **kwargs) -> None:
        await self.analytics.log_tool_usage(**kwargs)
    
    async def log_retrieval(self, **kwargs) -> None:
        await self.analytics.log_retrieval(**kwargs)
    
    async def log_error(self, **kwargs) -> None:
        await self.analytics.log_error(**kwargs)
    
    # Metrics methods
    async def get_metrics(self, period_hours: int = 24) -> PerformanceMetrics:
        return await self.metrics.get_metrics(period_hours)
    
    async def get_recommendations(self) -> List[Dict[str, Any]]:
        return await self.learning.get_improvement_recommendations()
    
    async def get_queries_for_improvement(self, limit: int = 20) -> List[Dict[str, Any]]:
        return await self.learning.get_queries_for_improvement(limit)
    
    async def close(self) -> None:
        """Clean up resources."""
        await self.analytics.close()


# =============================================================================
# Factory Functions
# =============================================================================

def create_feedback_service(
    feedback_collection: Any = None,
    analytics_collection: Any = None
) -> FeedbackService:
    """Create a feedback service."""
    return FeedbackService(
        feedback_collection=feedback_collection,
        analytics_collection=analytics_collection
    )


_service_instance: Optional[FeedbackService] = None


def get_feedback_service(**kwargs) -> FeedbackService:
    """Get or create global feedback service."""
    global _service_instance
    
    if _service_instance is None:
        _service_instance = create_feedback_service(**kwargs)
    
    return _service_instance
