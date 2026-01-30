# Enterprise Agentic RAG Platform v2.0.0 - Architecture

## Overview

This document describes the enhanced architecture of the Enterprise Agentic RAG Platform, upgraded from basic state-machine RAG to a full **multi-agent system with message passing**, inspired by AutoGen and CrewAI patterns.

## Core Principle

**Agentic RAG = RAG + Decision Making + Iteration + Self-Correction**

Unlike traditional RAG (retrieve → generate), this system follows:
**Plan → Retrieve → Verify → Iterate → Answer**

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API Layer (FastAPI)                                │
│                    /api/v2/ask  /api/v2/feedback  /api/v2/tools            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Query Understanding Layer                               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│  │   Query      │ │   Intent     │ │   Entity     │ │   Query      │       │
│  │ Classifier   │ │  Detector    │ │  Extractor   │ │  Expander    │       │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Multi-Agent Orchestrator (Dynamic)                       │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     Message Router                                   │   │
│   │            (Agent-to-Agent Communication)                           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌───────────┐   ┌───────────┐   ┌───────────┐   ┌───────────┐           │
│   │  Planner  │◄─►│ Researcher│◄─►│ Retriever │◄─►│  Verifier │           │
│   │   Agent   │   │   Agent   │   │   Agent   │   │   Agent   │           │
│   └───────────┘   └───────────┘   └───────────┘   └───────────┘           │
│         │               │               │               │                   │
│         └───────────────┴───────────────┴───────────────┘                   │
│                                 │                                            │
│                                 ▼                                            │
│                         ┌───────────┐                                        │
│                         │ Responder │                                        │
│                         │   Agent   │                                        │
│                         └───────────┘                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                    │                           │
                    ▼                           ▼
┌───────────────────────────────┐   ┌──────────────────────────────────┐
│         Tool Layer            │   │        Memory Layer               │
│                               │   │                                   │
│  ┌────────────┐ ┌──────────┐ │   │  ┌─────────────┐ ┌─────────────┐ │
│  │ Calculator │ │   Code   │ │   │  │ Short-Term  │ │  Long-Term  │ │
│  │    Tool    │ │ Executor │ │   │  │   (Redis)   │ │  (MongoDB)  │ │
│  └────────────┘ └──────────┘ │   │  └─────────────┘ └─────────────┘ │
│  ┌────────────┐ ┌──────────┐ │   └──────────────────────────────────┘
│  │ Web Search │ │ Database │ │
│  │   (Multi)  │ │   Query  │ │
│  └────────────┘ └──────────┘ │
│  ┌────────────┐              │
│  │  Internal  │              │
│  │   Search   │              │
│  └────────────┘              │
└───────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Knowledge Stores                                      │
│                                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                │
│  │  Vector Store  │  │    MongoDB     │  │   External     │                │
│  │   (Milvus)     │  │  (Documents)   │  │     APIs       │                │
│  │                │  │                │  │  (Web Search)  │                │
│  │ - Embeddings   │  │ - Metadata     │  │                │                │
│  │ - BM25 Index   │  │ - User Data    │  │ - Tavily       │                │
│  │ - Hybrid       │  │ - Feedback     │  │ - SerpAPI      │                │
│  └────────────────┘  └────────────────┘  │ - Bing/Google  │                │
│                                           └────────────────┘                │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Re-Ranking Layer                                     │
│                                                                              │
│   ┌──────────────────────┐        ┌──────────────────────┐                 │
│   │   Cross-Encoder      │        │    Cohere Rerank     │                 │
│   │   (Local/HuggingFace)│  OR    │    (Cloud API)       │                 │
│   │                      │  BOTH  │                      │                 │
│   │ ms-marco-MiniLM-L-6  │        │ rerank-english-v3.0  │                 │
│   └──────────────────────┘        └──────────────────────┘                 │
│                      │                    │                                  │
│                      └─────────┬──────────┘                                  │
│                                ▼                                             │
│                     ┌──────────────────────┐                                │
│                     │   Fusion Re-Ranker   │                                │
│                     │ (Weighted/RRF/Max)   │                                │
│                     └──────────────────────┘                                │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Verification Layer                                      │
│                                                                              │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐   │
│  │    Answer     │ │   Citation    │ │    Fact       │ │ Hallucination │   │
│  │  Verification │ │   Checker     │ │  Validator    │ │   Detector    │   │
│  └───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Feedback Loop                                         │
│                                                                              │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐   │
│  │   Feedback    │ │   Analytics   │ │    Metrics    │ │   Learning    │   │
│  │  Collector    │ │    Logger     │ │  Aggregator   │ │   Signals     │   │
│  │               │ │               │ │               │ │               │   │
│  │ - Ratings     │ │ - Query logs  │ │ - Dashboard   │ │ - Improve     │   │
│  │ - Thumbs      │ │ - Agent perf  │ │ - Trends      │ │   prompts     │   │
│  │ - Corrections │ │ - Tool usage  │ │ - Alerts      │ │ - Retrain     │   │
│  └───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Multi-Agent System

The system uses **true multi-agent collaboration** with message passing:

| Agent | Role | Capabilities |
|-------|------|--------------|
| **Planner** | Execution strategy | Query analysis, agent coordination, adaptive replanning |
| **Researcher** | Information needs | Query classification, entity extraction, query decomposition |
| **Retriever** | Document retrieval | Vector/hybrid search, query expansion, re-ranking |
| **Verifier** | Quality assurance | Citation checking, fact validation, hallucination detection |
| **Responder** | Answer generation | Synthesis, citation formatting, tone adaptation |

**Message Types:**
- `REQUEST` / `RESPONSE` - Standard request-response
- `BROADCAST` - Message to all agents
- `DELEGATE` - Task delegation
- `TOOL_REQUEST` / `TOOL_RESULT` - Tool interactions
- `REFLECTION` - Self-reflection output
- `CONSENSUS` - Multi-agent agreement

### 2. Re-Ranker (Dual Mode)

Supports both local and cloud re-ranking:

```python
# Configuration
RERANKER_PROVIDER: "both"  # cross-encoder, cohere, or both

# Cross-Encoder (Local)
- Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Device: CPU or GPU
- No API key needed

# Cohere (Cloud)
- Model: rerank-english-v3.0
- API key required
- Better quality, has latency

# Fusion Mode
- Combines both using weighted average or RRF
- Best of both worlds
```

### 3. Tool Layer (Container-Based)

Tools execute in isolated containers for security:

```python
# Supported Tools
- calculator: Safe math evaluation
- code_executor: Containerized Python execution
- web_search: Multi-provider (Tavily, SerpAPI, Bing, Google)
- database: MongoDB/SQL queries
- search: Internal knowledge base

# Container Execution
- Runtime: Docker or Kubernetes
- Resource limits: 256Mi memory, 0.5 CPU
- Network isolation (optional)
- Auto-cleanup after execution
```

### 4. Memory Layer (Dual Storage)

```python
# Short-Term Memory (Redis)
- Session-based
- TTL: 1 hour default
- Fast access
- Recent conversation context

# Long-Term Memory (MongoDB)
- Persistent across sessions
- Semantic search enabled
- User preferences
- Entity memory
- Interaction history
```

### 5. Web Search (Configurable)

```python
WEB_SEARCH_PROVIDER: "tavily"  # Options: tavily, serp, bing, google, duckduckgo

# Each provider configuration
- tavily: Built for RAG, includes answer extraction
- serp: Google results via SerpAPI
- bing: Microsoft Bing Search API
- google: Google Custom Search
- duckduckgo: No API key needed
```

---

## Request Flow

```
1. User Query → API
   │
2. Query Understanding
   ├─ Classify query type (definition, how-to, comparison, etc.)
   ├─ Detect intent
   ├─ Extract entities
   └─ Expand query
   │
3. Planner Agent
   ├─ Analyze complexity
   ├─ Select required agents
   └─ Create execution plan
   │
4. Researcher Agent
   ├─ Deep query analysis
   └─ Identify information needs
   │
5. Retriever Agent
   ├─ Execute search (vector + BM25)
   ├─ Apply re-ranking (cross-encoder/Cohere)
   └─ Return top documents
   │
6. Verifier Agent (optional iteration)
   ├─ Check relevance
   ├─ May trigger re-retrieval
   └─ Validate sources
   │
7. Responder Agent
   ├─ Synthesize answer
   ├─ Add citations
   └─ Format response
   │
8. Final Verification
   ├─ Citation accuracy
   ├─ Fact validation
   └─ Hallucination check
   │
9. Response → User
   │
10. Feedback Collection (background)
```

---

## Configuration

### Environment Variables

```bash
# Core
APP_ENV=production
LOG_LEVEL=INFO

# LLM
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4-turbo-preview

# Re-ranker
RERANKER_PROVIDER=both
COHERE_API_KEY=...

# Web Search
WEB_SEARCH_PROVIDER=tavily
TAVILY_API_KEY=tvly-...

# Multi-Agent
AGENT_ORCHESTRATION_MODE=dynamic
AGENT_MAX_ITERATIONS=10
AGENT_MAX_AGENT_CALLS=20

# Memory
MEMORY_SHORT_TERM_TTL=3600
MEMORY_LONG_TERM_ENABLED=true

# Tool Execution
TOOL_CONTAINER_RUNTIME=kubernetes
TOOL_ENABLE_NETWORK=false

# Verification
VERIFICATION_ENABLE=true
VERIFICATION_MODE=moderate
```

---

## API Endpoints

### Query Endpoints
- `POST /api/v2/ask` - Main RAG query with multi-agent processing
- `POST /api/v2/analyze` - Query understanding only

### Feedback Endpoints
- `POST /api/v2/feedback` - Submit rating/thumbs/correction

### Memory Endpoints
- `POST /api/v2/memory` - Store memory
- `GET /api/v2/memory/context` - Get memory context
- `DELETE /api/v2/memory` - Clear memory

### Tool Endpoints
- `GET /api/v2/tools` - List available tools
- `POST /api/v2/tools/execute` - Execute tool directly

### Metrics Endpoints
- `GET /api/v2/metrics` - Get performance metrics
- `GET /api/v2/metrics/recommendations` - Get improvement suggestions

### System Endpoints
- `GET /api/v2/health` - Health check
- `GET /api/v2/agents/status` - Agent status

---

## Accuracy Impact (from Industry Standards)

| Component | Impact | Status |
|-----------|--------|--------|
| Hybrid Retrieval | Very High | ✅ Implemented |
| Re-ranking | Very High | ✅ Implemented |
| Verification Agent | Very High | ✅ Implemented |
| Query Normalization | High | ✅ Implemented |
| Chunking Strategy | High | ✅ Existing |
| Metadata Filtering | High | ✅ Existing |
| Tool Usage | High | ✅ Implemented |
| Domain Embeddings | Medium-High | 📋 Configurable |
| Memory | Medium | ✅ Implemented |
| Prompt Engineering | Medium | ✅ Implemented |

---

## Deployment

### Docker Compose
```bash
cd agentic-rag-0.2
docker-compose up -d
```

### Kubernetes
```bash
kubectl apply -f k8s/deployment.yaml
```

---

## Minimal Viable Stack

For production deployment, these are essential:

1. ✅ Query Rewriter
2. ✅ Hybrid Retrieval (Vector + BM25)
3. ✅ Re-Ranker (Cross-encoder or Cohere)
4. ✅ Tool-Enabled LLM
5. ✅ Verification Step
6. ✅ Feedback Logging

---

## Files Structure

```
backend/
├── api/
│   └── enhanced_router.py          # New v2 API
├── core/
│   └── config_enhanced.py          # Enhanced configuration
├── services/
│   ├── agents/
│   │   ├── core/
│   │   │   ├── base.py             # Agent framework
│   │   │   ├── specialized_agents.py
│   │   │   └── orchestrator.py     # Dynamic orchestrator
│   │   ├── tools/
│   │   │   ├── base.py             # Tool framework
│   │   │   ├── specialized.py      # Calc, Code, Search, DB
│   │   │   └── k8s_executor.py     # K8s code execution
│   │   ├── memory/
│   │   │   └── memory_manager.py   # Short/Long term memory
│   │   ├── feedback/
│   │   │   └── service.py          # Feedback & analytics
│   │   └── verification/
│   │       └── service.py          # Answer verification
│   ├── query_understanding/
│   │   └── service.py              # Query analysis
│   ├── prompt_manager/
│   │   └── service.py              # Prompt management
│   └── reranker/
│       └── service.py              # Re-ranking (existing)
└── k8s/
    └── deployment.yaml             # Kubernetes configs
```

---

## Next Steps

1. **Test the multi-agent system** with complex queries
2. **Configure API keys** for Cohere and web search
3. **Deploy to Kubernetes** for production
4. **Monitor metrics** and collect feedback
5. **Iterate on prompts** based on feedback

---

*Architecture Version: 2.0.0*
*Last Updated: January 2025*
