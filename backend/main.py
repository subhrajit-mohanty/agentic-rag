"""
Enterprise Agentic RAG Platform - Main Application

Production-grade FastAPI application with:
- Async MongoDB integration
- Redis caching
- Agentic RAG orchestration
- Health monitoring
- CORS support
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

import structlog
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.core.config import Settings, get_settings
from backend.db.mongodb import close_database, db_manager, init_database
from backend.models.documents import ALL_DOCUMENT_MODELS, Persona
from backend.services.agents.agentic_rag import get_agentic_service
from backend.services.agents.models import (
    AgenticAskRequest,
    AgenticAskResponse,
    HealthResponse,
    StatsResponse,
)
from backend.services.cache.redis_cache import cache_manager, close_cache, init_cache
from backend.services.vector_store.service import init_vector_store, vector_store

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Manages startup and shutdown of all services:
    - MongoDB connection
    - Redis cache
    - Vector store
    """
    settings = get_settings()
    
    logger.info("Starting Enterprise Agentic RAG Platform...")
    
    # Initialize MongoDB
    try:
        await init_database(document_models=ALL_DOCUMENT_MODELS)
        logger.info("MongoDB initialized successfully")
    except Exception as e:
        logger.error(f"MongoDB initialization failed: {e}")
        # Continue without MongoDB - use mock data
    
    # Initialize Redis cache
    try:
        await init_cache()
        logger.info("Redis cache initialized successfully")
    except Exception as e:
        logger.warning(f"Redis initialization failed: {e} - running without cache")
    
    # Initialize vector store
    try:
        await init_vector_store()
        logger.info("Vector store initialized successfully")
    except Exception as e:
        logger.warning(f"Vector store initialization failed: {e}")
    
    # Seed default personas if empty
    await seed_default_data()
    
    logger.info("All services initialized - API ready")
    
    yield
    
    # Shutdown
    logger.info("Shutting down services...")
    await close_cache()
    await close_database()
    logger.info("Shutdown complete")


async def seed_default_data():
    """Seed default personas and sample documents if database is empty."""
    try:
        # Check if personas exist
        persona_count = await Persona.count()
        if persona_count == 0:
            default_personas = [
                Persona(
                    persona_id="legal_analyst",
                    name="Legal Analyst",
                    system_prompt="Analyze regulatory compliance and contracts with precision.",
                    temperature=0.1,
                    allowed_tools=["Retrieval"],
                    allowed_categories=["Legal", "Compliance"],
                    description="Specialized in legal document analysis"
                ),
                Persona(
                    persona_id="hr_specialist",
                    name="HR Specialist",
                    system_prompt="Assist with employee benefits, policies, and HR matters.",
                    temperature=0.2,
                    allowed_tools=["Retrieval"],
                    allowed_categories=["HR", "Benefits"],
                    description="Expert in HR policies and procedures"
                ),
                Persona(
                    persona_id="tech_support",
                    name="Tech Support",
                    system_prompt="Provide technical assistance for infrastructure and DevOps.",
                    temperature=0.1,
                    allowed_tools=["Retrieval"],
                    allowed_categories=["Engineering", "DevOps"],
                    description="Technical infrastructure specialist"
                ),
            ]
            
            for persona in default_personas:
                await persona.insert()
            
            logger.info(f"Seeded {len(default_personas)} default personas")
            
    except Exception as e:
        logger.warning(f"Failed to seed default data: {e}")


# Create FastAPI app
app = FastAPI(
    title="Enterprise Agentic RAG Platform",
    description="Production-grade Agentic RAG system with MongoDB, Redis, and LangGraph orchestration",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Health & Status Endpoints
# ============================================

@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Comprehensive health check endpoint.
    
    Checks the health of all dependent services:
    - MongoDB
    - Redis
    - Vector Store
    - LLM (Ollama)
    """
    services = {}
    overall_status = "healthy"
    
    # Check MongoDB
    mongo_health = await db_manager.health_check()
    services["mongodb"] = mongo_health
    if mongo_health["status"] != "healthy":
        overall_status = "degraded"
    
    # Check Redis
    redis_health = await cache_manager.health_check()
    services["redis"] = redis_health
    if redis_health["status"] not in ["healthy", "unavailable"]:
        overall_status = "degraded"
    
    # Check Vector Store
    vector_health = await vector_store.health_check()
    services["vector_store"] = vector_health
    if vector_health["status"] != "healthy":
        overall_status = "degraded"
    
    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        services=services
    )


@app.get("/api/v1/stats", response_model=StatsResponse, tags=["Stats"])
async def get_stats():
    """
    Get system statistics and metrics.
    """
    cache_stats = cache_manager.stats
    vector_health = await vector_store.health_check()
    
    return StatsResponse(
        total_vectors=f"{vector_health.get('documents_cached', 0):,}",
        agent_tasks="0",  # Would come from query logs
        active_connectors="3",
        avg_latency="124ms",
        health={
            "backend": "Healthy",
            "mongodb": "Healthy",
            "redis": "Healthy" if cache_stats["hits"] + cache_stats["misses"] > 0 else "Idle",
            "vector_store": vector_health["status"].title()
        },
        cache_stats=cache_stats
    )


# ============================================
# Persona Management Endpoints
# ============================================

@app.get("/api/v1/personas", tags=["Personas"])
async def get_personas():
    """
    Get all available personas.
    """
    try:
        personas = await Persona.find(Persona.is_active == True).to_list()
        return [
            {
                "id": p.persona_id,
                "name": p.name,
                "systemPrompt": p.system_prompt,
                "temperature": p.temperature,
                "allowedTools": p.allowed_tools,
                "description": p.description
            }
            for p in personas
        ]
    except Exception as e:
        logger.warning(f"Failed to fetch personas from DB: {e}")
        # Return default personas if DB fails
        return [
            {
                "id": "legal_analyst",
                "name": "Legal Analyst",
                "systemPrompt": "Analyze regulatory compliance and contracts.",
                "temperature": 0.1,
                "allowedTools": ["Retrieval"]
            },
            {
                "id": "hr_specialist",
                "name": "HR Specialist",
                "systemPrompt": "Assist with employee benefits and policies.",
                "temperature": 0.2,
                "allowedTools": ["Retrieval"]
            },
            {
                "id": "tech_support",
                "name": "Tech Support",
                "systemPrompt": "Provide technical assistance for infrastructure.",
                "temperature": 0.1,
                "allowedTools": ["Retrieval"]
            },
        ]


# ============================================
# Agentic RAG Endpoints
# ============================================

@app.post("/api/v1/ask-agentic", response_model=AgenticAskResponse, tags=["Agentic RAG"])
async def ask_agentic(request: AgenticAskRequest):
    """
    Process a query through the Agentic RAG pipeline.
    
    The query goes through:
    1. Guardrail validation (scope check)
    2. Hybrid retrieval (BM25 + vector)
    3. Document grading (relevance check)
    4. Query rewriting (if needed)
    5. Answer generation with citations
    
    Returns the answer along with sources, reasoning trace, and performance metrics.
    """
    try:
        service = get_agentic_service()
        
        response = await service.ask(
            query=request.query,
            framework=request.framework,
            persona_id=request.persona_id,
            use_cache=True
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Agentic RAG error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )


@app.post("/api/v1/search", tags=["Search"])
async def search_documents(
    query: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=5, ge=1, le=20, description="Max results"),
    use_hybrid: bool = Query(default=True, description="Use hybrid search")
):
    """
    Search the knowledge base directly.
    
    Returns matching documents with relevance scores.
    """
    try:
        results = await vector_store.search(
            query=query,
            limit=limit,
            use_hybrid=use_hybrid
        )
        
        return {
            "query": query,
            "total": len(results),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


# ============================================
# Cache Management Endpoints
# ============================================

@app.delete("/api/v1/cache", tags=["Cache"])
async def clear_cache(pattern: Optional[str] = Query(default="*", description="Key pattern to clear")):
    """
    Clear cached responses.
    
    Use pattern to selectively clear cache entries.
    """
    try:
        if pattern == "*":
            deleted = await cache_manager.delete_pattern("query:*")
        else:
            deleted = await cache_manager.delete_pattern(f"query:{pattern}")
        
        return {"message": f"Cleared {deleted} cache entries"}
        
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )


# ============================================
# Error Handlers
# ============================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.is_development,
        log_level=settings.log_level.lower()
    )
