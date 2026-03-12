# spark-scholar

Self-hosted research knowledge base for DGX Spark with three ingestion paths:
**2.96M arXiv abstracts** indexed by topic, **custom PDF chunking and embedding**
via BGE-M3 dense+sparse hybrid, and **dynamic corpus expansion** using Semantic
Scholar / AI2 to auto-fetch L1→L2→L3 citation graphs — all queryable through
Open WebUI tools with cross-encoder reranking and a LangGraph orchestration
pipeline.

**GitHub:** https://github.com/MARKYMARK55/spark-scholar

---

## Index

### 🚀 Getting Started
| Section | Location |
|---|---|
| Hardware & software prerequisites | [docs/getting_started.md → Prerequisites](docs/getting_started.md#prerequisites) |
| Setting up `env/.env` | [docs/getting_started.md → Configuration](docs/getting_started.md#configuration) |
| Environment variable reference (all vars) | [docs/getting_started.md → Env reference](docs/getting_started.md#environment-variable-reference) |
| Start the full stack (`start_stack.sh`) | [docs/getting_started.md → Startup Order](docs/getting_started.md#startup-order) |
| Manual step-by-step startup | [docs/getting_started.md → Manual startup](docs/getting_started.md#manual-startup-step-by-step) |
| Switch to indexing mode (50% GPU) | [docs/getting_started.md → Indexing mode](docs/getting_started.md#switching-to-indexing-mode) |
| Service port map | [README → Port Map](#service-port-map) |

### 🧠 Models & Inference
| Section | Location |
|---|---|
| SparkRun — managing vLLM inference | [docs/getting_started.md → SparkRun](docs/getting_started.md#sparkrun-required--manages-vllm-inference) |
| BGE-M3 dense embedder (port 8025) | [docs/getting_started.md → Dense Embedder](docs/getting_started.md#dense-embedder-port-8025) |
| BGE-M3 sparse embedder (port 8035) | [docs/getting_started.md → Sparse Embedder](docs/getting_started.md#sparse-embedder-port-8035) |
| BGE-M3 cross-encoder reranker (port 8020) | [docs/getting_started.md → Reranker](docs/getting_started.md#reranker-port-8020) |
| VRAM budget table (all services) | [docs/getting_started.md → VRAM Budget](docs/getting_started.md#vram-budget-dgx-spark--128-gb-unified-memory) |
| LiteLLM proxy — config & key pattern | [docs/getting_started.md → LiteLLM Setup](docs/getting_started.md#litellm-setup) |
| Phi Mini — secondary lightweight model | [docs/getting_started.md → Phi Mini](docs/getting_started.md#phi-mini--secondary-lightweight-model) |
| Adding cloud models (OpenAI, Anthropic…) | [docs/ui_interfaces.md → LiteLLM Admin](docs/ui_interfaces.md#3--litellm-admin-ui--httplocalhost4000ui) |
| Cloud model reference YAML | [core_services/litellm_cloud.yaml](core_services/litellm_cloud.yaml) |

### 📥 Ingestion Pipeline
| Section | Location |
|---|---|
| Pipeline overview (arXiv vs custom corpus) | [docs/ingestion.md → Overview](docs/ingestion.md#overview) |
| Step 0 — Create Qdrant collections (22 total) | [docs/ingestion.md → Step 0](docs/ingestion.md#step-0-create-qdrant-collections) |
| Step 1 — Download arXiv metadata (2.96M papers) | [docs/ingestion.md → Step 1](docs/ingestion.md#step-1-download-arxiv-metadata) |
| Step 2 — Dense embed + index (arXiv abstracts) | [docs/ingestion.md → Step 2](docs/ingestion.md#step-2-dense-embedding--indexing) |
| Step 3 — Sparse embed + index (arXiv abstracts) | [docs/ingestion.md → Step 3](docs/ingestion.md#step-3-sparse-embedding--indexing) |
| Step 4 — PDF ingest + HDBSCAN auto-classify | [docs/ingestion.md → Step 4](docs/ingestion.md#step-4-pdf-ingestion-your-own-documents) |
| Step 5 — Figure captioning (vision model) | [docs/ingestion.md → Step 5](docs/ingestion.md#step-5-figure-captioning) |
| Step 6 — HTML / web documentation crawler | [docs/ingestion.md → Step 6](docs/ingestion.md#step-6-web--html-documentation-ingestion) |
| Step 7 — Citation graph expansion (L2/L3) | [docs/ingestion.md → Step 7](docs/ingestion.md#step-7-citation-graph-expansion-l2--l3) |
| HDBSCAN auto-classification pipeline explained | [docs/ingestion.md → What pipeline does](docs/ingestion.md#what-the-python-pipeline-does) |
| Documentation collections (8 docs-* collections) | [docs/ingestion.md → Collections table](docs/ingestion.md#documentation-collections) |
| Per-language ingest commands (Rust/Python/JS/Docker/Anthropic/AppleScript/DevOps) | [docs/ingestion.md → Step 6](docs/ingestion.md#step-6-web--html-documentation-ingestion) |
| Keeping the index fresh (re-ingest, recreate) | [docs/ingestion.md → Freshness](docs/ingestion.md#keeping-the-index-fresh) |
| Browsing collections in Open WebUI | [docs/ingestion.md → Browsing](docs/ingestion.md#browsing-your-collections-in-open-webui) |
| Ingestion timing table | [docs/ingestion.md → Timing](docs/ingestion.md#timing-table) |
| Throughput benchmarks & VRAM tuning | [docs/embedding_speed.md](docs/embedding_speed.md) |

### 🔍 Search & Retrieval
| Section | Location |
|---|---|
| Dense search CLI (`query/dense_search.py`) | [docs/search_retrieval.md → Dense search](docs/search_retrieval.md#dense-search--semantic-similarity) |
| Sparse search CLI (`query/sparse_search.py`) | [docs/search_retrieval.md → Sparse search](docs/search_retrieval.md#sparse-search--keyword-precision) |
| Hybrid search CLI — full pipeline (recommended) | [docs/search_retrieval.md → Hybrid search](docs/search_retrieval.md#hybrid-search--full-pipeline-recommended) |
| Comparing all three modes side-by-side | [docs/search_retrieval.md → Comparison](docs/search_retrieval.md#comparing-all-three-modes) |
| Why hybrid beats dense or sparse alone | [docs/search_retrieval.md → Why hybrid](docs/search_retrieval.md#why-hybrid) |
| Qdrant native RRF — Prefetch + FusionQuery | [docs/search_retrieval.md → Qdrant RRF](docs/search_retrieval.md#qdrants-native-rrf-implementation) |
| Cross-encoder reranking (BGE-M3, top-50→top-10) | [docs/search_retrieval.md → Reranking](docs/search_retrieval.md#reranking) |
| LangGraph pipeline — node graph + descriptions | [docs/search_retrieval.md → LangGraph](docs/search_retrieval.md#langgraph-pipeline) |
| LangGraph conditional web search node | [docs/search_retrieval.md → Conditional edges](docs/search_retrieval.md#conditional-edges) |
| Redis caching strategy + cache invalidation | [docs/search_retrieval.md → Caching](docs/search_retrieval.md#caching-strategy) |

### 🖥 UI Interfaces
| Section | Location |
|---|---|
| All UIs — quick reference table (ports + purpose) | [docs/ui_interfaces.md → Quick Reference](docs/ui_interfaces.md#quick-reference) |
| Open WebUI — setup, RAG Path A vs B, connections | [docs/ui_interfaces.md → Open WebUI](docs/ui_interfaces.md#1--open-webui--httplocalhost8080) |
| Qdrant Dashboard — browse collections, manual search | [docs/ui_interfaces.md → Qdrant](docs/ui_interfaces.md#2--qdrant-dashboard--httplocalhost6333dashboard) |
| LiteLLM Admin — add models, API keys, spend tracking | [docs/ui_interfaces.md → LiteLLM Admin](docs/ui_interfaces.md#3--litellm-admin-ui--httplocalhost4000ui) |
| Langflow — visual pipeline builder | [docs/ui_interfaces.md → Langflow](docs/ui_interfaces.md#4--langflow--httplocalhost7860) |
| Langfuse — traces, spans, tuning workflow | [docs/ui_interfaces.md → Langfuse](docs/ui_interfaces.md#5--langfuse--httplocalhost3000) |
| SearXNG — private web search, bang shortcuts | [docs/ui_interfaces.md → SearXNG](docs/ui_interfaces.md#6--searxng--httplocalhost8888) |
| Redis/Valkey — CLI commands, RedisInsight | [docs/ui_interfaces.md → Redis](docs/ui_interfaces.md#7--redis--valkey--no-built-in-ui) |
| Which UI for which task — decision table | [docs/ui_interfaces.md → Summary](docs/ui_interfaces.md#summary-which-ui-for-which-task) |

### 🛠 Open WebUI Tools (Dynamic RAG)
| Section | Location |
|---|---|
| How tools work — mechanism & tool vs RAG proxy | [docs/open_webui_tools.md → How tools work](docs/open_webui_tools.md#how-open-webui-tools-work) |
| Installing & enabling tools | [docs/open_webui_tools.md → Installing](docs/open_webui_tools.md#installing-tools) |
| Tool: arXiv paper search | [docs/open_webui_tools.md → arXiv Search](docs/open_webui_tools.md#core-tool-arxiv-paper-search) |
| Tool: Semantic Scholar (citation counts, recommendations) | [docs/open_webui_tools.md → Semantic Scholar](docs/open_webui_tools.md#core-tool-semantic-scholar-search) |
| Tool: Ingest PDF to corpus (live corpus expansion) | [docs/open_webui_tools.md → Ingest PDF](docs/open_webui_tools.md#core-tool-ingest-pdf-to-corpus) |
| Tool: Query RAG corpus (targeted collection search) | [docs/open_webui_tools.md → RAG Search](docs/open_webui_tools.md#core-tool-query-the-rag-corpus) |
| Tool: SearXNG web search | [docs/open_webui_tools.md → Web Search](docs/open_webui_tools.md#core-tool-web-search-via-searxng) |
| Dynamic RAG workflow — model expands corpus on-the-fly | [docs/open_webui_tools.md → Dynamic RAG](docs/open_webui_tools.md#dynamic-rag-workflow) |
| Suggested tool combinations (research / docs workflows) | [docs/open_webui_tools.md → Combinations](docs/open_webui_tools.md#suggested-tool-combinations) |
| Corpus expansion strategies (chat / CLI / upload) | [docs/open_webui_tools.md → Expansion](docs/open_webui_tools.md#expanding-your-corpus--ingestion-strategies) |
| Community tools hub + recommended installs | [docs/open_webui_tools.md → Community](docs/open_webui_tools.md#getting-community-tools) |
| Debugging tools (not firing, verbose logs) | [docs/open_webui_tools.md → Debugging](docs/open_webui_tools.md#debugging-tools) |

### 🔧 Troubleshooting & Reference
| Section | Location |
|---|---|
| Qdrant returns 0 results | [docs/troubleshooting.md → Qdrant 0 results](docs/troubleshooting.md#qdrant-returns-0-results) |
| BGE-M3 dense embedder not responding | [docs/troubleshooting.md → Dense not responding](docs/troubleshooting.md#bge-m3-dense-embedder-not-responding) |
| Sparse embedder OOM | [docs/troubleshooting.md → Sparse OOM](docs/troubleshooting.md#sparse-embedder-oom) |
| Redis connection refused | [docs/troubleshooting.md → Redis](docs/troubleshooting.md#redis-connection-refused) |
| LLM returning very short responses | [docs/troubleshooting.md → Short responses](docs/troubleshooting.md#llm-returning-very-short-responses) |
| RAG proxy embedding failed | [docs/troubleshooting.md → Embedding failed](docs/troubleshooting.md#rag-proxy-returns-embedding-failed) |
| Open WebUI shows no models | [docs/troubleshooting.md → No models](docs/troubleshooting.md#open-webui-shows-no-models) |
| Langfuse traces not appearing | [docs/troubleshooting.md → Langfuse](docs/troubleshooting.md#langfuse-traces-not-appearing) |
| Dense ingestion is slow | [docs/troubleshooting.md → Slow ingestion](docs/troubleshooting.md#dense-ingestion-is-slow) |
| Collection routing errors | [docs/troubleshooting.md → Routing](docs/troubleshooting.md#collection-routing-errors) |
| LiteLLM DB connection error | [docs/troubleshooting.md → LiteLLM DB](docs/troubleshooting.md#litellm-fails-to-start-db-connection-error) |
| Phi Mini not in model list | [docs/troubleshooting.md → Phi Mini](docs/troubleshooting.md#phi-mini-not-appearing-in-model-list) |
| Known limitations | [docs/troubleshooting.md → Known Limitations](docs/troubleshooting.md#known-limitations) |
| Repository structure | [README → Repository Structure](#repository-structure) |

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

# 2. Copy env template — all local keys are pre-filled, just set your model name
cp .env.example env/.env
nano env/.env          # set VLLM_MODEL_NAME to match your SparkRun model

# 3. Start your inference model via SparkRun
#    https://github.com/scitrera/oss-spark-run
sparkrun start Qwen/Qwen3-30B-A3B
```

### Start the full stack
```bash
./scripts/start_stack.sh
```

Starts all services in dependency order with health checks:
`Qdrant → Embedding → Redis + SearXNG + LiteLLM + Open WebUI + Langflow → RAG Proxy`

### Verify
```bash
curl http://localhost:8002/health     # RAG proxy
curl http://localhost:4000/health     # LiteLLM
curl http://localhost:6333/readyz     # Qdrant
open http://localhost:8080            # Open WebUI
```

### Ingest your corpus
```bash
# Create all 22 Qdrant collections (run once)
python ingest/02_create_collections.py

# arXiv abstracts — 2.96M papers (optional, takes 18–22 hrs)
python ingest/01_download_arxiv.py --output-dir data/
bash scripts/start_indexing_mode.sh
python ingest/03_ingest_dense.py  --input data/arxiv_with_abstract.jsonl --batch-size 256 &
python ingest/04_ingest_sparse.py --input data/arxiv_with_abstract.jsonl --batch-size 64

# Your own PDFs (auto-classified into collections via HDBSCAN)
python ingest/05_ingest_pdfs.py --input-dir /path/to/pdfs/

# Web documentation — crawl any docs site
python ingest/07_ingest_html_docs.py \
    --url https://docs.anthropic.com/en/docs/ \
    --collection docs-anthropic --depth 2

# Citation graph expansion — L2 + L3 referenced papers
python ingest/08_expand_citations.py --arxiv 2303.08774 --depth 2
```

---

## Service Port Map

| Service | Port | Purpose |
|---|---|---|
| Open WebUI | 8080 | Primary chat UI |
| LiteLLM proxy | 4000 | Unified model gateway |
| RAG proxy | 8002 | OpenAI-compatible RAG endpoint |
| Qdrant | 6333 | Vector database + dashboard |
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
spark-scholar/
├── .env.example                   # Copy to env/.env — all local keys pre-filled
├── README.md                      # This file — index + architecture + quick start
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
│   ├── bge_m3_dense.yml           # BGE-M3 dense vLLM, port 8025 (production 12% GPU)
│   ├── bge_m3_dense_indexing.yml  # BGE-M3 dense vLLM, port 8025 (50% GPU, bulk ingest)
│   ├── bge_m3_sparse.yml          # BGE-M3 sparse FastAPI, port 8035
│   └── bge_m3_reranker.yml        # BGE-M3 cross-encoder vLLM, port 8020
│
├── rag_proxy/                     # RAG proxy service
│   ├── rag_proxy.py               # FastAPI OpenAI-compatible server
│   ├── rag_proxy.yml              # Docker Compose (build context = repo root)
│   ├── Dockerfile
│   └── requirements.txt
│
├── pipeline/                      # Core RAG logic (used by rag_proxy + ingest)
│   ├── langgraph_pipeline.py      # LangGraph StateGraph (12 nodes)
│   ├── embeddings.py              # BGE-M3 dense + sparse HTTP clients
│   ├── hybrid_search.py           # Qdrant Prefetch + FusionQuery(RRF) + fan-out
│   ├── reranker.py                # BGE-M3 cross-encoder via vLLM /score
│   ├── router.py                  # Query → collection routing (keyword heuristics)
│   ├── cache.py                   # Redis result cache (SHA-256 keyed)
│   └── tracer.py                  # Langfuse spans (no-op if keys absent)
│
├── query/                         # CLI search tools (HTTP only — no local model needed)
│   ├── dense_search.py            # Dense-only HNSW search
│   ├── sparse_search.py           # Sparse-only inverted index search
│   └── hybrid_search.py           # Full hybrid + cross-encoder rerank (recommended)
│
├── ingest/                        # Data ingestion scripts
│   ├── 01_download_arxiv.py       # Stream arXiv HuggingFace dataset → JSONL
│   ├── 02_create_collections.py   # Create 22 Qdrant collections (14 arXiv + 8 docs)
│   ├── 03_ingest_dense.py         # BGE-M3 dense embed + upsert (arXiv abstracts)
│   ├── 04_ingest_sparse.py        # BGE-M3 sparse embed + upsert (arXiv abstracts)
│   ├── 05_ingest_pdfs.py          # Full-text PDF + HDBSCAN auto-classify + route
│   ├── 06_caption_figures.py      # Vision model figure captioning (PDFs)
│   ├── 07_ingest_html_docs.py     # BFS HTML crawler → chunk → embed → Qdrant
│   └── 08_expand_citations.py     # Citation graph L2/L3 via Semantic Scholar API
│
├── images/                        # Custom Docker image source code
│   └── sparse-embedder/           # BGE-M3 SPLADE FastAPI service (port 8035)
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
│   └── README.md                  # Environment variable quick-reference
│
└── docs/
    ├── getting_started.md         # Prerequisites, env config, startup, VRAM, LiteLLM, Phi Mini
    ├── ingestion.md               # All 8 ingest scripts, per-language crawl commands, citation expansion
    ├── search_retrieval.md        # Dense/sparse/hybrid CLI, reranking, LangGraph pipeline, caching
    ├── ui_interfaces.md           # Every UI — Open WebUI, Qdrant, LiteLLM, Langflow, Langfuse, SearXNG, Redis
    ├── open_webui_tools.md        # Dynamic RAG tools, corpus expansion, Semantic Scholar, tool debugging
    ├── troubleshooting.md         # 13 diagnostic scenarios + known limitations
    └── embedding_speed.md         # VRAM budgets, throughput benchmarks, tuning guide
```
