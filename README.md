# Enterprise Agentic RAG Platform

<div align="center">
  <h3>Production-Grade Agentic RAG System</h3>
  <p>Intelligent document retrieval and answer generation with MongoDB, Redis, and LangGraph orchestration</p>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.109+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/MongoDB-7.0-green.svg" alt="MongoDB">
  <img src="https://img.shields.io/badge/Redis-7-red.svg" alt="Redis">
  <img src="https://img.shields.io/badge/Docker-Compose-blue.svg" alt="Docker">
</p>

## 🎯 Overview

The Enterprise Agentic RAG Platform is a production-ready system that combines:

- **Agentic RAG Architecture**: Intelligent query processing with guardrails, document grading, and query rewriting
- **Hybrid Search**: BM25 keyword search + vector similarity for optimal retrieval
- **MongoDB Storage**: Scalable document store with full-text search capabilities
- **Redis Caching**: Sub-100ms response times for repeated queries (150-400x speedup)
- **Local LLM**: Ollama-powered inference for privacy and cost efficiency

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                              │
│                    Dashboard • Chat • Personas                       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend (:8000)                          │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────────┐ │
│  │  /health    │  │ /ask-agentic │  │      /search               │ │
│  └─────────────┘  └──────┬───────┘  └─────────────────────────────┘ │
└──────────────────────────┼──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Agentic RAG Service                                │
│  ┌──────────┐  ┌──────────┐  ┌───────┐  ┌─────────┐  ┌──────────┐  │
│  │Guardrail │→ │ Retrieve │→ │ Grade │→ │ Rewrite │→ │ Generate │  │
│  └──────────┘  └──────────┘  └───────┘  └─────────┘  └──────────┘  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┬──────────────────┐
        ▼                  ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   MongoDB     │  │    Redis      │  │    Milvus     │  │    Ollama     │
│  (:27017)     │  │   (:6379)     │  │   (:19530)    │  │   (:11434)    │
│               │  │               │  │               │  │               │
│  • Documents  │  │  • Response   │  │  • Vectors    │  │  • llama3.2   │
│  • Personas   │  │    Cache      │  │  • Semantic   │  │  • Generation │
│  • Query Logs │  │  • Sessions   │  │    Search     │  │  • Grading    │
└───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Docker Desktop** with Docker Compose
- **8GB+ RAM** recommended
- **20GB+ disk space** for Ollama models

### 1. Clone and Setup

```bash
git clone <repository-url>
cd enterprise-rag-platform

# Copy environment file
cp .env.example .env
```

### 2. Start Services

```bash
# Start all services
make start

# Or with dev tools (Mongo Express, Redis Commander)
make dev
```

### 3. Pull LLM Model

```bash
make pull-model
```

### 4. Verify Installation

```bash
make health
```

### 5. Access Services

| Service | URL | Description |
|---------|-----|-------------|
| **API Docs** | http://localhost:8000/docs | Interactive API documentation |
| **Backend API** | http://localhost:8000 | REST API endpoints |
| **Milvus** | localhost:19530 | Vector database (gRPC) |
| **MinIO Console** | http://localhost:9001 | Object storage UI (Milvus) |
| **Mongo Express** | http://localhost:8081 | MongoDB admin UI (dev mode) |
| **Redis Commander** | http://localhost:8082 | Redis admin UI (dev mode) |

## 📚 API Endpoints

### Agentic RAG

```bash
# Ask a question through the agentic pipeline
curl -X POST http://localhost:8000/api/v1/ask-agentic \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the PTO policy?",
    "use_hybrid": true,
    "framework": "LangGraph"
  }'
```

**Response:**
```json
{
  "query": "What is the PTO policy?",
  "answer": "Based on the Employee Handbook, employees are entitled to 20 days of paid time off per year. You can carry over up to 5 unused days to the next calendar year. [doc_001]",
  "sources": [
    {
      "document_id": "doc_001",
      "title": "Employee Handbook - PTO Policy",
      "relevance_score": 0.95
    }
  ],
  "reasoning_steps": [
    "Node: guardrail (validating query scope)",
    "Guardrail: Score 90/100 - HR policy query",
    "Node: retrieve (attempt 1)",
    "Retrieved 3 documents",
    "Node: grade_documents (evaluating relevance)",
    "Grading: Relevant - Documents contain PTO information",
    "Node: generate_answer (synthesizing response)"
  ],
  "execution_time": 2.45,
  "cache_hit": false
}
```

### Direct Search

```bash
curl "http://localhost:8000/api/v1/search?query=S3+access&limit=5"
```

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGO_URI` | `mongodb://...` | MongoDB connection string |
| `REDIS_HOST` | `localhost` | Redis host |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `LLM_MODEL` | `llama3.2:3b` | Default LLM model |
| `VECTOR_PROVIDER` | `milvus` | Vector store: `milvus` or `memory` |
| `VECTOR_MILVUS_HOST` | `localhost` | Milvus host |
| `VECTOR_MILVUS_PORT` | `19530` | Milvus port |
| `VECTOR_MILVUS_COLLECTION` | `enterprise_rag_docs` | Milvus collection name |
| `MAX_RETRIEVAL_ATTEMPTS` | `2` | Max query rewrites |
| `GUARDRAIL_THRESHOLD` | `60` | Min score to proceed (0-100) |
| `TOP_K_RESULTS` | `5` | Documents to retrieve |
| `USE_HYBRID_SEARCH` | `true` | Enable hybrid search |

See `.env.example` for all options.

## 📁 Project Structure

```
enterprise-rag-platform/
├── backend/
│   ├── core/
│   │   └── config.py          # Pydantic settings
│   ├── db/
│   │   └── mongodb.py         # Async MongoDB client
│   ├── models/
│   │   └── documents.py       # Beanie ODM models
│   ├── services/
│   │   ├── agents/
│   │   │   ├── agentic_rag.py # Main RAG orchestration
│   │   │   ├── models.py      # Pydantic models
│   │   │   └── prompts.py     # LLM prompts
│   │   ├── cache/
│   │   │   └── redis_cache.py # Redis caching
│   │   ├── llm/
│   │   │   └── client.py      # Ollama client
│   │   └── vector_store/
│   │       ├── service.py     # Hybrid search service
│   │       └── milvus_store.py # Milvus vector DB
│   └── main.py                # FastAPI app
├── frontend/                   # React frontend
├── scripts/
│   └── mongo-init.js          # DB initialization
├── docker-compose.yml
├── Makefile
└── README.md
```

## 🔄 Agentic RAG Workflow

```
START
  │
  ▼
┌─────────────────┐
│   GUARDRAIL     │ ──── Score < 60 ────▶ OUT_OF_SCOPE
│ (Scope Check)   │
└────────┬────────┘
         │ Score ≥ 60
         ▼
┌─────────────────┐
│    RETRIEVE     │
│ (Hybrid Search) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     GRADE       │ ──── Not Relevant ──┐
│ (LLM Grading)   │                     │
└────────┬────────┘                     │
         │ Relevant                     ▼
         │                    ┌─────────────────┐
         │                    │    REWRITE      │
         │                    │ (Query Optimize)│
         │                    └────────┬────────┘
         │                             │
         │◀────────────────────────────┘
         │                    (Max 2 attempts)
         ▼
┌─────────────────┐
│    GENERATE     │
│ (Answer + Cite) │
└────────┬────────┘
         │
         ▼
       END
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run linting
make lint
```

## 📊 Performance

| Metric | Without Cache | With Cache | Improvement |
|--------|--------------|------------|-------------|
| Response Time | 2-5 seconds | 50-100ms | **50-100x faster** |
| LLM Calls | Every request | Only on miss | **Cost reduction** |
| Throughput | ~10 req/s | ~200 req/s | **20x higher** |

## 🔐 Security Considerations

- All credentials in environment variables (not in code)
- MongoDB authentication enabled
- Redis password support
- CORS configuration for frontend origins
- Non-root Docker containers
- Health check endpoints for monitoring

## 🚀 Production Deployment

1. Update `.env` with production values
2. Use strong passwords for MongoDB and Redis
3. Configure proper CORS origins
4. Set `APP_ENV=production` and `DEBUG=false`
5. Consider GPU-enabled Ollama for faster inference
6. Set up monitoring and alerting

## 📝 License

MIT License - see LICENSE file for details.

---

<div align="center">
  <p><strong>Built for Enterprise AI Applications</strong></p>
</div>
