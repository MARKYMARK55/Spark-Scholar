# arxiv-rag

Production arXiv RAG stack for DGX Spark — hybrid dense+sparse retrieval,
BGE-M3 reranking, LangGraph orchestration, and an OpenAI-compatible proxy
that plugs directly into Open WebUI.

**GitHub:** https://github.com/MARKYMARK55/arxiv-rag

---

## Documentation

| Document | Contents |
|---|---|
| [Getting Started](docs/getting_started.md) | Prerequisites, env setup, startup, VRAM budget, LiteLLM |
| [Ingestion Pipeline](docs/ingestion.md) | arXiv abstracts, PDF ingest, HTML docs, citation expansion |
| [Search & Retrieval](docs/search_retrieval.md) | Dense/sparse/hybrid search, reranking, LangGraph, caching |
| [UI Interfaces](docs/ui_interfaces.md) | Open WebUI, Qdrant, LiteLLM admin, Langflow, Langfuse, SearXNG |
| [Open WebUI Tools](docs/open_webui_tools.md) | Dynamic RAG, corpus expansion tools, Semantic Scholar |
| [Troubleshooting](docs/troubleshooting.md) | Diagnostic commands, known limitations |
| [Embedding Speed](docs/embedding_speed.md) | Throughput benchmarks, VRAM tuning guide |

---

## Architecture

```
                        ┌─────────────────────────────────────────────────┐
                        │                 DGX Spark                       │
                        │                                                  │
  User / Open WebUI     │  ┌──────────────┐    ┌───────────────────────┐  │
  ──────────────────►   │  │  RAG Proxy   │    │    LiteLLM Proxy      │  │
  POST /v1/chat/        │  │  (port 8002) │    │    (port 4000)        │  │
  completions           │  └──────┬───────┘    └───────────┬───────────┘  │
                        │         │                        │               │
                        │  ┌──────▼───────────────────┐   │               │
                        │  │   LangGraph Pipeline      │   │               │
                        │  │                           │   │               │
                        │  │  check_cache              │   │               │
                        │  │       ↓                   │   │               │
                        │  │  route_query              │   │               │
                        │  │       ↓                   │   │               │
                        │  │  embed_query ─────────────┼──►│ BGE-M3 Dense │
                        │  │       │       ────────────┼──►│ BGE-M3 Sparse│
                        │  │       ↓                   │   └──────────────┘
                        │  │  hybrid_retrieve ─────────┼──────────────────►
                        │  │       ↓             Qdrant│  (port 6333)      │
                        │  │  [web_search] ────────────┼──► SearXNG        │
                        │  │       ↓             (opt) │   (port 8888)     │
                        │  │  merge_results            │                   │
                        │  │       ↓                   │                   │
                        │  │  rerank_results ──────────┼──► BGE Reranker   │
                        │  │       ↓             cross-encoder (port 8020) │
                        │  │  build_context            │                   │
                        │  │       ↓                   │                   │
                        │  │  llm_inference ───────────┼──► LiteLLM ──────►│
                        │  │       ↓             (port 4000)    vLLM       │
                        │  │  cache_result ────────────┼──► Redis          │
                        │  │       ↓             (port 6379)               │
                        │  │  trace_result ────────────┼──► Langfuse       │
                        │  └───────────────────────────┘    (port 3000)    │
                        └─────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────┐
  │             Qdrant Collections — 22 total (14 arXiv + 8 docs)        │
  │                                                                      │
  │  arXiv (catch-all)      arxiv-cs-ml-ai       arxiv-cs-systems-theory │
  │  arxiv-condmat          arxiv-astro           arxiv-hep              │
  │  arxiv-math-applied     arxiv-math-phys       arxiv-math-pure        │
  │  arxiv-misc             arxiv-nucl-nlin-physother                    │
  │  arxiv-qbio-qfin-econ   arxiv-quantph-grqc    arxiv-stat-eess        │
  │                                                                      │
  │  docs-python  docs-rust  docs-javascript  docs-docker                │
  │  docs-anthropic  docs-applescript  docs-devops  docs-web             │
  └──────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites
```bash
# 1. Create the Docker network (once only)
docker network create llm-net

# 2. Copy env template (all local keys are pre-filled)
cp .env.example env/.env
nano env/.env   # set VLLM_MODEL_NAME to match your SparkRun model

# 3. Start your inference model via SparkRun (manages vLLM)
#    https://github.com/scitrera/oss-spark-run
sparkrun start Qwen/Qwen3-30B-A3B
```

### Start the full stack
```bash
./scripts/start_stack.sh
```

That's it. The script starts all services in order with health checks:
`Qdrant → Embedding → Redis + SearXNG + LiteLLM + Open WebUI + Langflow → RAG Proxy`

### Verify
```bash
curl http://localhost:8002/health          # RAG proxy
curl http://localhost:4000/health          # LiteLLM
curl http://localhost:6333/readyz          # Qdrant
open http://localhost:8080                 # Open WebUI
```

### Then ingest your corpus
```bash
# Create all 22 collections
python ingest/02_create_collections.py

# Index arXiv abstracts (optional — takes 18–22 hrs for 2.96M papers)
python ingest/01_download_arxiv.py --output-dir data/
bash scripts/start_indexing_mode.sh
python ingest/03_ingest_dense.py --input data/arxiv_with_abstract.jsonl --batch-size 256 &
python ingest/04_ingest_sparse.py --input data/arxiv_with_abstract.jsonl --batch-size 64

# Ingest your own PDFs
python ingest/05_ingest_pdfs.py --input-dir /path/to/pdfs/

# Ingest web documentation
python ingest/07_ingest_html_docs.py \
    --url https://docs.anthropic.com/en/docs/ \
    --collection docs-anthropic --depth 2

# Expand citations (L2 + L3 from a seed paper)
python ingest/08_expand_citations.py --arxiv 2303.08774 --depth 2
```

---

## Service Port Map

| Service | Port | Purpose |
|---|---|---|
| Open WebUI | 8080 | Primary chat UI |
| LiteLLM proxy | 4000 | Unified model gateway |
| RAG proxy | 8002 | OpenAI-compatible RAG endpoint |
| Qdrant | 6333 | Vector database |
| Langflow | 7860 | Visual pipeline builder |
| SearXNG | 8888 | Private web search |
| Langfuse | 3000 | Observability / tracing (optional) |
| BGE-M3 dense | 8025 | Dense embedder (vLLM) |
| BGE-M3 sparse | 8035 | Sparse embedder (FastAPI) |
| BGE reranker | 8020 | Cross-encoder reranker (vLLM) |
| SparkRun vLLM | 8000 | Primary inference model |
| Phi Mini (opt.) | 8001 | Secondary lightweight model |
| Redis/Valkey | 6379 | Response cache |

---

## Repository Structure

```
arxiv-rag/
├── .env.example                   # Rename to env/.env — all local keys pre-filled
├── README.md                      # This file
├── requirements.txt               # Python dependencies
│
├── core_services/                 # Core service compose files + configs
│   ├── core_services.yml          # LiteLLM + Postgres + Open WebUI + Langflow
│   ├── qdrant.yml                 # Qdrant vector DB
│   ├── redis.yml                  # Valkey (Redis-compatible) cache
│   ├── searxng.yml                # Private web search
│   ├── langfuse.yml               # Observability (optional)
│   ├── litellm_local.yaml         # LiteLLM config — SparkRun + BGE-M3 + Phi Mini
│   └── litellm_cloud.yaml         # LiteLLM config — cloud models (reference only)
│
├── embedding/                     # Embedding service compose files
│   ├── bge_m3_dense.yml           # BGE-M3 dense vLLM, port 8025 (production)
│   ├── bge_m3_dense_indexing.yml  # BGE-M3 dense vLLM, port 8025 (50% GPU, bulk ingest)
│   ├── bge_m3_sparse.yml          # BGE-M3 sparse FastAPI, port 8035
│   └── bge_m3_reranker.yml        # BGE-M3 cross-encoder vLLM, port 8020
│
├── rag_proxy/                     # RAG proxy service
│   ├── rag_proxy.py               # FastAPI OpenAI-compatible server
│   ├── rag_proxy.yml              # Docker Compose
│   ├── Dockerfile                 # Build context = repo root
│   └── requirements.txt
│
├── pipeline/                      # Core RAG logic (imported by rag_proxy + ingest)
│   ├── langgraph_pipeline.py      # LangGraph StateGraph (12 nodes)
│   ├── embeddings.py              # BGE-M3 dense + sparse HTTP clients
│   ├── hybrid_search.py           # Qdrant Prefetch + FusionQuery(RRF) + fan-out
│   ├── reranker.py                # BGE-M3 cross-encoder via vLLM /score
│   ├── router.py                  # Query → collection routing (keyword heuristics)
│   ├── cache.py                   # Redis result cache (SHA-256 keyed)
│   └── tracer.py                  # Langfuse spans (no-op if keys absent)
│
├── query/                         # CLI search tools (HTTP only, no local model)
│   ├── dense_search.py            # Dense-only HNSW search
│   ├── sparse_search.py           # Sparse-only inverted index search
│   └── hybrid_search.py           # Full hybrid + cross-encoder rerank
│
├── ingest/                        # One-time ingestion scripts
│   ├── 01_download_arxiv.py       # Stream arXiv HuggingFace dataset → JSONL
│   ├── 02_create_collections.py   # Create 22 Qdrant collections
│   ├── 03_ingest_dense.py         # BGE-M3 dense embed + upsert (arXiv abstracts)
│   ├── 04_ingest_sparse.py        # BGE-M3 sparse embed + upsert (arXiv abstracts)
│   ├── 05_ingest_pdfs.py          # Full-text PDF + HDBSCAN auto-classify
│   ├── 06_caption_figures.py      # Vision model figure captioning (PDFs)
│   ├── 07_ingest_html_docs.py     # BFS HTML crawler → embed → Qdrant
│   └── 08_expand_citations.py     # Citation graph expansion (L2/L3 via Semantic Scholar)
│
├── images/                        # Custom Docker image source
│   └── sparse-embedder/           # BGE-M3 SPLADE FastAPI service
│       ├── sparse_embed.py
│       ├── Dockerfile
│       └── requirements.txt
│
├── scripts/
│   ├── start_stack.sh             # Full stack startup with health polling
│   ├── start_indexing_mode.sh     # Switch dense embedder to 50% GPU (bulk ingest)
│   └── stop_indexing_mode.sh      # Switch back to 12% GPU production mode
│
├── env/
│   └── README.md                  # Environment variable reference
│
└── docs/
    ├── getting_started.md         # Prerequisites, config, startup, VRAM, LiteLLM
    ├── ingestion.md               # All ingestion scripts + per-language commands
    ├── search_retrieval.md        # Dense/sparse/hybrid CLI tools, reranking, LangGraph
    ├── ui_interfaces.md           # Every UI in the stack with setup guides
    ├── open_webui_tools.md        # Dynamic RAG tools, corpus expansion, Semantic Scholar
    ├── troubleshooting.md         # Diagnostic commands + known limitations
    └── embedding_speed.md         # VRAM budgets, throughput benchmarks, tuning
```
