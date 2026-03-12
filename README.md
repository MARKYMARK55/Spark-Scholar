# arxiv-rag

Production Arxiv RAG stack for DGX Spark — hybrid dense+sparse retrieval, BGE-M3 reranking, LangGraph orchestration, and an OpenAI-compatible proxy that plugs directly into Open WebUI.

**GitHub:** https://github.com/MARKYMARK55/arxiv-rag

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
                        │  │       ↓             (optional) (port 8888)    │
                        │  │  merge_results            │                   │
                        │  │       ↓                   │                   │
                        │  │  rerank_results ──────────┼──► BGE Reranker   │
                        │  │       ↓             cross-encoder (port 8020) │
                        │  │  build_context            │                   │
                        │  │       ↓                   │                   │
                        │  │  llm_inference ───────────┼──► LiteLLM ──────►│
                        │  │       ↓              (port│4000)    vLLM      │
                        │  │  cache_result ────────────┼──► Redis          │
                        │  │       ↓             (port 6379)               │
                        │  │  trace_result ────────────┼──► Langfuse       │
                        │  └───────────────────────────┘    (port 3000)    │
                        └─────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────┐
  │                     Qdrant Collections (14 total)                   │
  │                                                                      │
  │  arXiv (catch-all)      arxiv-cs-ml-ai       arxiv-cs-systems-theory│
  │  arxiv-condmat          arxiv-astro           arxiv-hep              │
  │  arxiv-math-applied     arxiv-math-phys       arxiv-math-pure        │
  │  arxiv-misc             arxiv-nucl-nlin-physother                    │
  │  arxiv-qbio-qfin-econ   arxiv-quantph-grqc    arxiv-stat-eess        │
  │                                                                      │
  │  Each collection: dense_embedding (1024-dim HNSW) + sparse_text     │
  └──────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Hardware
- NVIDIA DGX Spark (or any system with NVIDIA GPU, 64GB+ RAM, 2TB+ NVMe)
- NVIDIA drivers 535+ and CUDA 12.1+
- Docker Engine 24+ and NVIDIA Container Toolkit

### Software
```bash
# Install Docker (if not present)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && newgrp docker

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

### Docker network
All services communicate on a shared Docker bridge network:
```bash
docker network create llm-net
```

---

## Configuration

### Setting up your .env file

The entire stack — every Docker service, every Python script — reads from a single
`env/.env` file. This is the first thing to get right.

```bash
# 1. Copy the template (safe placeholder values — never commit the real file)
cp .env.example env/.env

# 2. Edit it — fill in at minimum the CORE LOCAL SERVICES section
nano env/.env

# 3. Generate a strong LITELLM_MASTER_KEY (do not leave it as the placeholder)
openssl rand -base64 32
# Paste the result as your LITELLM_MASTER_KEY value

# 4. Verify Python can load it correctly
python3 -c "from dotenv import load_dotenv; import os; load_dotenv('env/.env'); print(os.getenv('QDRANT_URL'))"
# Expected output: http://localhost:6333
```

**The `simple-api-key` pattern explained:** Services running on the private `llm-net`
Docker network use `simple-api-key` as their API key. This is intentional — the
network is not internet-exposed and the key prevents accidental open access between
misconfigured containers. **Never use `simple-api-key` for cloud services.** Generate
real random values for `LITELLM_MASTER_KEY` and `WEBUI_SECRET_KEY`.

**Minimum required values for a fully local stack:**
```bash
LITELLM_MASTER_KEY=<openssl rand -base64 32>
WEBUI_SECRET_KEY=<openssl rand -base64 32>
HF_TOKEN=hf_...            # HuggingFace token — needed to download Arxiv dataset
VLLM_MODEL_NAME=<model name you are serving on port 8000>
```

All other values in `.env.example` are either already correct (`simple-api-key`
for local services) or optional (cloud API keys for fallback routing).

For full explanation of every variable, how Docker Compose loads the file, and
security best practices — see the header comments in `.env.example`.

See `docs/embedding_speed.md` for VRAM allocation and throughput tuning.

### Environment variable reference

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant REST API URL |
| `QDRANT_API_KEY` | _(empty)_ | Qdrant API key (leave empty if no auth) |
| `BGE_M3_DENSE_URL` | `http://localhost:8025` | vLLM serving BAAI/bge-m3 for dense embeddings |
| `BGE_M3_SPARSE_URL` | `http://localhost:8035` | Custom FastAPI sparse embedder |
| `BGE_RERANKER_URL` | `http://localhost:8020` | vLLM serving bge-reranker-v2-m3 |
| `BGE_M3_API_KEY` | `simple-api-key` | Shared API key for all BGE services |
| `VLLM_URL` | `http://localhost:8000` | vLLM inference server URL |
| `VLLM_API_KEY` | `simple-api-key` | vLLM API key |
| `VLLM_MODEL_NAME` | `local-model` | Model name as registered in vLLM |
| `LITELLM_URL` | `http://localhost:4000` | LiteLLM proxy URL |
| `LITELLM_API_KEY` | `simple-api-key` | API key for user requests |
| `LITELLM_MASTER_KEY` | _(required)_ | Admin key for LiteLLM management |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `LANGFUSE_SECRET_KEY` | _(optional)_ | Langfuse secret (tracing disabled if unset) |
| `LANGFUSE_PUBLIC_KEY` | _(optional)_ | Langfuse public key |
| `LANGFUSE_HOST` | `http://localhost:3000` | Langfuse server URL |
| `OPENAI_API_BASE_URL` | `http://localhost:4000` | Used by Open WebUI |
| `OPENAI_API_KEY` | `simple-api-key` | Used by Open WebUI |
| `HF_TOKEN` | _(required for download)_ | HuggingFace token for dataset access |
| `SEARXNG_URL` | `http://localhost:8888` | SearXNG for web search |
| `RAG_PROXY_PORT` | `8002` | Port for the RAG proxy FastAPI server |
| `RAG_TOP_K` | `10` | Final retrieved results after reranking |
| `RAG_RERANK_TOP_N` | `50` | Candidates passed to cross-encoder |
| `CACHE_TTL_SECONDS` | `86400` | Redis TTL for cached responses (24 h default for single-user) |
| `CHUNK_SIZE` | `1000` | PDF chunk size in tokens |
| `CHUNK_OVERLAP` | `150` | Overlap between consecutive chunks |

---

## Startup Order

Services must start in this order due to dependencies:

### Step 1: Infrastructure
```bash
cd docker

# Start Qdrant
docker compose --profile qdrant up -d
# Wait for Qdrant to be healthy
docker compose ps qdrant  # should show "healthy"

# Start Redis
docker compose --profile core up -d redis
```

### Step 2: Embedding services
These require GPU allocation. Check VRAM budget before starting all three simultaneously.
```bash
# Dense embedder (~6GB VRAM on fp16)
docker compose --profile embedding up -d bge-m3-dense-embedder

# Sparse embedder (~6GB VRAM on fp16)
docker compose --profile embedding up -d bge-m3-sparse-embedder

# Reranker (~4GB VRAM on fp16)
docker compose --profile embedding up -d bge-m3-reranker

# Wait for all embedding services to be healthy (model download takes 2-5min first time)
watch docker compose ps
```

### Step 3: LiteLLM + vLLM
```bash
# Your vLLM inference server is already running at port 8000 (from dgx-spark-rag-stack)
# Start LiteLLM proxy
docker compose --profile core up -d litellm-db litellm
```

### Step 4: Open WebUI and supporting services
```bash
docker compose --profile core up -d openwebui searxng langflow
```

### Step 5: Langfuse (optional)
```bash
docker compose -f core_services/langfuse.yml up -d
```

### Step 6: RAG Proxy
```bash
docker compose --profile rag-proxy up -d

# Verify all services are up
docker compose --profile qdrant --profile embedding --profile core --profile rag-proxy ps
```

---

## Embedding Stack

### Dense Embedder (port 8025)
- **Model:** BAAI/bge-m3 served by vLLM
- **Output:** 1024-dimensional float32 L2-normalised vectors
- **VRAM:** ~6GB on fp16
- **Throughput:** ~1,500 abstracts/second on a single A100 (batch size 256)
- **API:** OpenAI /v1/embeddings compatible
- **Why BGE-M3:** State-of-the-art multilingual model (100+ languages), 8192 token context, outperforms OpenAI ada-002 on MTEB benchmarks

### Sparse Embedder (port 8035)
- **Model:** BAAI/bge-m3 via FlagEmbedding (SPLADE-style lexical weights)
- **Output:** Sparse vector of (token_id, weight) pairs, typically 200-500 non-zero dimensions
- **VRAM:** ~6GB on fp16
- **Throughput:** ~300 abstracts/second (batch size 32, CPU-bound tokenisation)
- **API:** Custom FastAPI /encode endpoint
- **Why sparse:** Captures exact term matches that dense vectors miss. Critical for scientific vocabulary, acronyms, equation references, and author names.

### Reranker (port 8020)
- **Model:** BAAI/bge-reranker-v2-m3 served by vLLM
- **VRAM:** ~4GB on fp16
- **Throughput:** ~50 query-document pairs/second
- **API:** vLLM /score endpoint
- **Why cross-encoder:** Bi-encoder (dense) vectors must compress the full document meaning into a fixed-size vector. Cross-encoders attend jointly over (query, document) pairs and therefore catch subtle semantic relevance that bi-encoders miss. The trade-off is latency: cross-encoders are 10-100x slower, so we only apply them to the top 50 candidates.

### VRAM Budget (DGX Spark — 128 GB unified memory)

| Service | Mode | GPU util | VRAM |
|---|---|---|---|
| vLLM inference (Nemotron/Qwen3) | production | auto | ~80–100 GB |
| BGE-M3 dense embedder | production | 0.12 | ~3 GB |
| BGE-M3 sparse embedder | production | fp16 | ~3 GB |
| BGE-M3 reranker | production | 0.15 | ~2 GB |
| **Total (production)** | | | **~88–108 GB** |

During bulk indexing (inference model still running alongside):

| Service | Mode | GPU util | VRAM |
|---|---|---|---|
| BGE-M3 dense embedder | indexing | 0.50 | ~12 GB |
| BGE-M3 sparse embedder | indexing | fp16 | ~3 GB |
| **Total (indexing only)** | | | **~15 GB** |

The embedding stack in indexing mode uses only ~15 GB — the inference model can
continue serving queries simultaneously.

See `docs/embedding_speed.md` for full throughput benchmarks and tuning guide.

---

## Ingestion Pipeline

> **Scope clarification — two separate pipelines:**
>
> - **Steps 1–3** index the Arxiv **metadata + abstracts** corpus (2.96M papers).
>   This is text-only. No HDBSCAN, no PDF extraction, no figure captioning.
>   The abstracts are clean structured data that embed directly.
>
> - **Steps 4–5** are for **your own PDFs** — papers you have downloaded,
>   your own documents, curated collections. These use full-text extraction,
>   HDBSCAN auto-classification, and optionally vision model figure captioning.
>   They are completely separate from the Arxiv abstract pipeline.

### Step 0: Create Qdrant collections
```bash
python ingest/02_create_collections.py

# Verify all 14 collections exist with correct vector config:
python ingest/02_create_collections.py --verify-only
```

### Step 1: Download Arxiv metadata (abstracts only — not PDFs)
```bash
# Full dataset (~4.8 GB JSONL, 2.96M papers)
python ingest/01_download_arxiv.py --output-dir data/

# Quick test with 10k papers:
python ingest/01_download_arxiv.py --output-dir data/ --max-records 10000
```
**Timing:** ~40 minutes on a good connection. Cached by HuggingFace after first run.

### Step 2: Dense embedding + indexing (Arxiv abstracts)
```bash
# Switch to indexing mode — 50% GPU utilisation for maximum throughput
bash scripts/start_indexing_mode.sh

# Run ingestion (batch 256 for indexing mode throughput)
python ingest/03_ingest_dense.py \
    --input data/arxiv_with_abstract.jsonl \
    --batch-size 256

# When done, switch back to production mode (12% GPU)
bash scripts/stop_indexing_mode.sh
```
**Timing:** ~18–22 hours for 2.96M papers at ~1,100 docs/s (indexing mode).
**Resume:** Progress is tracked in `data/arxiv_with_abstract_dense_progress.txt` — re-running skips already-processed records. Upserts are idempotent (deterministic point IDs from arxiv_id hash).

### Step 3: Sparse embedding + indexing (Arxiv abstracts)
```bash
# Run sparse indexing — can overlap with dense (sparse uses the separate FastAPI service)
python ingest/04_ingest_sparse.py \
    --input data/arxiv_with_abstract.jsonl \
    --batch-size 64
```
**Timing:** ~6–8 hours for 2.96M papers at ~1,400–1,600 docs/s.
**Run in parallel with Step 2:** Dense (~18 hrs) dominates — start both simultaneously.
Both scripts write separate progress files so they don't interfere.

### Step 4: PDF full-text ingestion *(your own PDFs — not Arxiv abstracts)*

This pipeline is for PDFs you have downloaded or curated — **not** the Arxiv
abstract corpus (which is already indexed as clean text in Steps 2–3).

It includes:
- **PyMuPDF** text extraction with **unstructured** fallback for complex layouts
- **tiktoken** chunking (CHUNK_SIZE / CHUNK_OVERLAP from `.env`)
- **HDBSCAN + UMAP** clustering to auto-detect topic groups across your PDF set
- **Qwen3** (local vLLM) to name each cluster — so PDFs get routed to the right collection automatically

```bash
# Ingest a directory of PDFs
python ingest/05_ingest_pdfs.py --input-dir /path/to/your/pdfs/

# Force a specific collection (skip auto-classification):
python ingest/05_ingest_pdfs.py --input-dir ./pdfs/ --collection arxiv-cs-ml-ai

# Dry run to preview classification without writing to Qdrant:
python ingest/05_ingest_pdfs.py --input-dir ./pdfs/ --dry-run
```
**Timing:** ~2–4 min per PDF (10–30 pages). HDBSCAN clusters the whole batch once — faster with more PDFs.

### Step 5: Figure captioning *(PDFs only — not Arxiv abstracts)*

Generates captions for figures, tables, and diagrams in PDFs using a vision model
(Qwen2-VL via vLLM). Captions are stored as `type="figure_caption"` points in
Qdrant alongside the text chunks, making figures searchable.

The Arxiv abstract corpus does **not** need this step — abstracts are text-only
and do not contain embedded figures.

```bash
# Caption figures in all PDFs in a directory
python ingest/06_caption_figures.py \
    --input-dir /path/to/pdfs/ \
    --pages-with-figures-only   # skip pages with only text

# Single file:
python ingest/06_caption_figures.py \
    --input-dir /path/to/pdfs/ \
    --file my-paper.pdf
```
**Timing:** ~30–60 seconds per page with figures. Vision inference is slow but offline —
run it once at ingestion, never at query time.

### Timing Table

Measured on DGX Spark (Grace Blackwell, 128 GB unified memory):

| Stage | Scope | Throughput | Wall time |
|---|---|---|---|
| Download Arxiv metadata | 2.96M papers | ~1,300/s (network) | ~40 min |
| Create 14 Qdrant collections | — | instant | ~30 sec |
| Dense embed + upsert *(indexing mode, 50% GPU)* | 2.96M abstracts | ~1,100 docs/s | ~18–22 hrs |
| Sparse embed + upsert | 2.96M abstracts | ~1,400–1,600 docs/s | ~6–8 hrs |
| Dense + sparse **in parallel** | 2.96M abstracts | limited by dense | **~18–22 hrs total** |
| Single PDF, 30 pages *(production mode)* | ~60 chunks | — | ~2–4 min |
| Single PDF, 200 pages | ~400 chunks | — | ~15–25 min |
| Figure captioning *(vision model, per page)* | PDFs only | ~2 pages/min | ~30–60 s/page |

> **Dense is slower than sparse** despite BGE-M3 being the same model. Dense
> requires computing the full 1024-dim projection; sparse uses a shallower lexical
> weight head that is significantly cheaper to evaluate. The upsert overhead
> (writing to both subject-area collection and master `arXiv` collection) also adds
> to total wall time.

See `docs/embedding_speed.md` for full tuning guide, VRAM budget breakdown, and
how to switch between production and indexing modes.

---

## Searching the Index

Three CLI tools in `query/` cover every search mode. All three call the running
**embedding services over HTTP** — they do not load any model locally and are safe
to run alongside the full production stack.

### Services required

| Script | Services needed |
|---|---|
| `query/dense_search.py` | BGE-M3 dense (port 8025) + Qdrant (port 6333) |
| `query/sparse_search.py` | BGE-M3 sparse (port 8035) + Qdrant (port 6333) |
| `query/hybrid_search.py` | dense + sparse + reranker (port 8020) + Qdrant |

Start them first if not already running:
```bash
bash scripts/start_stack.sh embedding
```

---

### Dense search — semantic similarity

Best for: concept-level queries, synonyms, paraphrase variation, cross-domain discovery.

```bash
# Simple query — searches all 13 subject collections and RRF-merges
python query/dense_search.py --query "variational inference latent variable models"

# Restrict to one collection
python query/dense_search.py --query "RLHF alignment language models" \
    --collection arxiv-cs-ml-ai

# With filters
python query/dense_search.py --query "stochastic differential equations" \
    --year-min 2020 --year-max 2024

# More results + JSON output
python query/dense_search.py --query "high frequency trading market microstructure" \
    --top-k 20 --json

# Filter by author
python query/dense_search.py --query "diffusion models" --author "Ho"
```

**How it works:**
The query is encoded to a 1024-dim vector by the vLLM BGE-M3 service, then an HNSW
approximate nearest-neighbour search is run in each collection's `dense_embedding`
slot. When searching all collections, results are merged with RRF.

---

### Sparse search — exact keyword precision

Best for: specific model names (`GPT-4`, `LLaMA-3.1`), author names, arXiv IDs,
equations, chemical formulae, and any query where the exact token matters more than
the semantic meaning.

```bash
# Simple keyword query
python query/sparse_search.py --query "attention mechanism transformers self-attention"

# Exact author + title keywords
python query/sparse_search.py --query "Vaswani attention all you need" \
    --year-min 2017 --year-max 2018

# Search by arXiv ID substring
python query/sparse_search.py --query "2305.14314"

# Restrict to one collection
python query/sparse_search.py --query "Hawkes process excitation kernel" \
    --collection arxiv-stat-eess

# JSON output for programmatic use
python query/sparse_search.py --query "bge-m3 embedding multilingual" --json
```

**How it works:**
The query is encoded by the BGE-M3 sparse service to a SPLADE-style vector of
(token_id → weight) pairs (typically 100–400 non-zero terms). Qdrant's inverted
index scores the dot product against each document's `sparse_text` slot. Results
from all collections are merged with RRF.

---

### Hybrid search — full pipeline (recommended)

Best for: everything. Combines dense semantic retrieval + sparse keyword matching via
Qdrant's native server-side RRF, then cross-encoder reranks the top candidates.
This is the same pipeline used by the RAG proxy for every chat message.

```bash
# Full pipeline — auto-routes to relevant collections, reranks
python query/hybrid_search.py --query "diffusion models image generation"

# More candidates before reranking (default: top-50 reranked → top-10)
python query/hybrid_search.py --query "RLHF alignment" \
    --rerank-n 100 --top-k 20

# Skip reranking (faster, useful for debugging retrieval quality)
python query/hybrid_search.py --query "Higgs boson discovery" --no-rerank

# With all filters + verbose timing
python query/hybrid_search.py \
    --query "protein folding AlphaFold" \
    --year-min 2020 \
    --author "Jumper" \
    --verbose

# Force a specific collection (bypass routing)
python query/hybrid_search.py --query "quantum error correction" \
    --collection arxiv-quantph-grqc

# JSON output — includes rerank_score, rrf_score, source collection
python query/hybrid_search.py --query "superconductor cuprate" --json
```

**How it works:**
1. **Route** — `pipeline/router.py` maps the query to 1-3 subject collections using keyword heuristics.
2. **Encode** — dense and sparse vectors are computed in parallel (async).
3. **Retrieve** — each collection runs `Prefetch([dense, sparse]) + FusionQuery(RRF)` natively inside Qdrant.
4. **Merge** — per-collection results are merged with a second Python-side RRF pass.
5. **Rerank** — the top 50 candidates are scored jointly by the BGE-M3 cross-encoder.
6. **Return** — top 10 results with `rerank_score`, `rrf_score`, `arxiv_id`, title, authors, abstract.

**Verbose output example:**
```
Routing: 'diffusion models image generation' → ['arxiv-cs-ml-ai']        [routing]
Retrieved 48 candidates in 0.31s                                          [retrieve]
Reranked to 10 in 1.42s                                                   [rerank]
Total: 1.73s                                                              [total]
```

---

### Comparing all three modes

```bash
QUERY="contrastive learning self-supervised representations"

echo "=== DENSE ==="
python query/dense_search.py  --query "$QUERY" --top-k 5

echo "=== SPARSE ==="
python query/sparse_search.py --query "$QUERY" --top-k 5

echo "=== HYBRID ==="
python query/hybrid_search.py --query "$QUERY" --top-k 5
```

The hybrid results typically include high-recall papers from dense (conceptually related
but different terminology) plus high-precision papers from sparse (exact term matches)
and then cross-encoder reranking promotes the most relevant of both.

---

## Hybrid Search

### Why hybrid?

Dense retrieval (HNSW on bi-encoder vectors) excels at semantic similarity — finding papers about the same concept even if they use different terminology. For example, "neural network" and "deep learning" map to nearby points in the embedding space.

Sparse retrieval (SPLADE inverted index) excels at exact term matching — catching specific model names, equation identifiers, author names, and technical acronyms that dense vectors tend to blur. If you search for "bge-m3 performance on BEIR", dense search might return broadly relevant papers; sparse search finds papers that literally mention "BGE-M3".

You need both. The canonical result is that hybrid retrieval consistently beats either modality alone across all BEIR benchmarks.

### Qdrant's native RRF implementation

The pipeline uses Qdrant's first-class Prefetch + FusionQuery(RRF) support, which means the fusion happens inside the Qdrant server:

```python
results = client.query_points(
    collection_name="arxiv-cs-ml-ai",
    prefetch=[
        Prefetch(query=dense_vector, using="dense_embedding", limit=100),
        Prefetch(query=sparse_vector, using="sparse_text",    limit=100),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=50,
    with_payload=True,
)
```

This is more efficient than Python-side fusion because Qdrant can use its SIMD-optimised ranking code, avoid re-serialising results over the network, and apply filters after fusion.

### Cross-collection merging

When multiple collections are searched (e.g. the query spans cs.ML and stat.ML), each collection returns its own RRF-fused list, and we do a second Python-side RRF merge. The collections are chosen by `pipeline/router.py` based on keyword heuristics, so most queries only search 1-2 collections.

---

## Reranking

After hybrid search returns up to 50 candidates, the BGE-M3 cross-encoder scores each (query, document) pair jointly:

```
POST http://localhost:8020/score
{
    "model": "bge-reranker-v2-m3",
    "text_1": "What is the attention mechanism?",
    "text_2": ["Abstract of paper 1...", "Abstract of paper 2...", ...]
}
```

The cross-encoder reads the full query and document together in a single forward pass, allowing it to catch:
- Negation ("not quantum computing" vs. "quantum computing")
- Multi-hop reasoning ("paper that cites X and uses method Y")
- Domain disambiguation ("kernel in SVM" vs. "kernel in operating systems")

The returned top-10 are much more precisely relevant than the initial 50 candidates.

**Latency budget:** The full pipeline (embed + retrieve + rerank + LLM) takes 3-8 seconds depending on the LLM. The reranker itself adds ~1-2 seconds for 50 candidates.

---

## LangGraph Pipeline

The pipeline is implemented as a LangGraph StateGraph with typed state (RAGState TypedDict). LangGraph provides:
- Deterministic execution graph (nodes + edges)
- Async node execution
- Built-in state management
- Easy conditional branching

### Node descriptions

1. **check_cache** — Computes a cache key from the query and checks Redis. If hit, the entire pipeline short-circuits to END. Cache keys are SHA-256 hashes of (query, collections) so the same query always hits the same cache entry.

2. **route_query** — Determines which 1-3 Qdrant collections to search. Uses keyword heuristics mapped to domain knowledge (e.g. "transformer" → arxiv-cs-ml-ai, "qubit" → arxiv-quantph-grqc). Falls back to `["arXiv"]` for ambiguous queries. Also checks the cache with the proper (query + collections) key.

3. **embed_query** — Encodes the query to both dense (1024-dim float32) and sparse (indices + values) vectors in parallel using asyncio. This parallelism saves ~200ms on the embedding step.

4. **hybrid_retrieve** — Runs Qdrant hybrid search across all routed collections concurrently. Each collection uses native Prefetch+RRF fusion. Results from multiple collections are merged with a second RRF pass in Python.

5. **web_search** _(conditional)_ — If the query contains time-sensitive keywords ("latest", "2025", "recent", etc.), run a SearXNG search for fresh results. Web results are appended to the candidates with a low initial score.

6. **merge_results** — Combines Qdrant and web results into a unified candidate list.

7. **rerank_results** — Applies the BGE-M3 cross-encoder to the top 50 candidates. Returns the top 10.

8. **build_context** — Formats the top 10 papers into a structured prompt context with inline citations. Each entry includes: ArXiv ID, title, authors, year, categories, and the most relevant text excerpt.

9. **llm_inference** — Sends the formatted system prompt (with context) and user query to LiteLLM. The system prompt instructs the model to cite papers using `[ArXiv:XXXX.XXXXX]` notation.

10. **cache_result** — Stores the response in Redis with CACHE_TTL_SECONDS TTL. Only successful responses are cached.

11. **trace_result** — Finalises the Langfuse trace with latency and success status.

### Conditional edges

- `check_cache → END` if cached=True (full pipeline bypass)
- `route_query → llm_inference` if the proper-key cache hit (skip embed/retrieve/rerank)
- `hybrid_retrieve → web_search` if query is time-sensitive
- `hybrid_retrieve → merge_results` for all other queries

---

## Caching Strategy

Redis is used to cache complete RAG pipeline outputs (the full LLM response + top reranked sources).

**Cache key:** SHA-256(query.strip().lower() + sorted(collections))

This means:
- "What is quantum entanglement?" and "what is quantum entanglement?" → same key
- Queries routing to different collections → different keys
- Filters (year_min, year_max, author) → different keys (via `extra` param)

**TTL:** 3600 seconds (1 hour) by default. Increase for stable datasets; decrease for fast-moving fields.

**What is cached:** The complete response text + top-5 source metadata. The intermediate vectors, candidates, and reranked list are not cached (Redis memory optimisation).

**Cache miss path:** When Redis is unreachable, ResultCache.get() returns None and the pipeline runs normally. The pipeline never fails due to cache unavailability.

---

## Langfuse Setup

Langfuse provides end-to-end observability: trace each request through retrieval, reranking, and generation to see exactly why each response was produced.

### Self-hosted (recommended for DGX Spark)
```bash
# Start Langfuse via docker-compose profile
docker compose -f core_services/langfuse.yml up -d

# Navigate to http://localhost:3000
# Create an account, then create a project
# Copy the secret and public keys from the project settings
```

### Configure keys in env/.env
```bash
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_HOST=http://localhost:3000
```

### What is traced
Each RAG request produces one Langfuse trace with these spans:
- **qdrant-hybrid-retrieval**: query, collections searched, number of candidates
- **bge-reranking**: input candidate count, top reranker scores, top titles
- **llm-generation**: full prompt (system message with context + user query), response, token counts

Use the Langfuse UI to:
- Debug why a specific query returned poor results
- Track latency per stage over time
- Compare prompt iterations A/B
- Monitor token usage and cost

---

## Open WebUI Integration

There are two ways to use Open WebUI with the Arxiv RAG stack:

### Path A: Built-in RAG (document uploads)

Open WebUI has a built-in RAG feature for document uploads. It uses its own embedding model and vector store, which is simpler but does not use your Qdrant collections.

1. Navigate to http://localhost:8080
2. Admin > Settings > Documents
3. Set **Embedding Engine**: OpenAI
4. Set **OpenAI API Base URL**: `http://localhost:4000/v1`
5. Set **OpenAI API Key**: `simple-api-key`
6. Set **Embedding Model**: `bge-m3-embedder`
7. Save settings
8. In a chat, click the paperclip icon to upload a PDF
9. Open WebUI will chunk, embed, and retrieve from your uploaded document

**Limitations of Path A:**
- Uses Open WebUI's own Chroma vector store — not your Qdrant collections
- No cross-encoder reranking (uses cosine similarity only)
- No hybrid search (dense only)
- Does not access the 2.3M indexed arXiv abstracts

### Path B: RAG Proxy (recommended)

This gives you the full hybrid search + reranking pipeline via an OpenAI-compatible endpoint.

**Option 1: Replace the primary LiteLLM connection:**
1. Admin > Settings > Connections
2. Set **OpenAI API Base URL**: `http://localhost:8002/v1`
3. Set **OpenAI API Key**: `simple-api-key`
4. Save and refresh — models now route through the RAG proxy

**Option 2: Add as an additional connection (keeps LiteLLM direct access too):**
1. Admin > Settings > Connections > OpenAI Connections > Add Connection
2. Name: `arxiv-rag`
3. API Base URL: `http://localhost:8002/v1`
4. API Key: `simple-api-key`
5. Save

With Path B, every chat message automatically retrieves relevant arXiv papers, reranks them, and includes them as context for the LLM. The response includes `[ArXiv:XXXX.XXXXX]` citations.

**Custom fields (Path B):** The RAG proxy accepts extra fields in the request body:
```json
{
    "model": "arxiv-rag",
    "messages": [{"role": "user", "content": "What is RLHF?"}],
    "year_min": 2020,
    "year_max": 2025,
    "author_filter": "Ouyang"
}
```

---

## LiteLLM Setup

LiteLLM provides a unified OpenAI-compatible gateway in front of all models. Every service (Open WebUI, Langflow, RAG proxy, scripts) talks to LiteLLM instead of directly to vLLM, which gives you:

- Centralised API key management
- Request logging and cost tracking
- Automatic retries and fallbacks
- Model aliasing (call "local-model" instead of the full Nemotron model path)

### The simple-api-key pattern

All services are configured with `simple-api-key` as the API key for user requests. The LiteLLM master key (LITELLM_MASTER_KEY) is only used for admin operations (adding/removing models via the LiteLLM admin UI).

This means any service on the `llm-net` Docker network can call LiteLLM with:
```python
headers = {"Authorization": "Bearer simple-api-key"}
```

### How models are registered

Models are defined in `core_services/litellm_config.yaml`. The three BGE models (dense embedder, reranker) are registered there so Langflow and other tools can call them by name. The main LLM is registered as "local-model" pointing at your vLLM instance.

Cloud fallbacks (GPT-4o, Claude) are also defined but only activate if the corresponding API keys are set in env/.env.

---

## Troubleshooting

### Qdrant returns 0 results
```bash
# Check collection status
curl http://localhost:6333/collections | python3 -m json.tool

# Check point counts per collection
for col in arxiv-cs-ml-ai arxiv-condmat arxiv-astro arxiv-hep; do
    echo "$col: $(curl -s http://localhost:6333/collections/$col | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["result"]["points_count"])')"
done
```
If all collections show 0 points, you need to run the ingestion pipeline (steps 1-3 above).

### BGE-M3 dense embedder not responding
```bash
# Check container logs
docker logs bge-m3-dense-embedder --tail 50

# Test directly
curl http://localhost:8025/health
curl -X POST http://localhost:8025/v1/embeddings \
  -H "Authorization: Bearer simple-api-key" \
  -H "Content-Type: application/json" \
  -d '{"model": "bge-m3-embedder", "input": ["test sentence"]}'
```
Common issue: first startup downloads the model (~2GB). Check logs for download progress. Allow 5 minutes.

### Sparse embedder OOM
The sparse embedder is memory-intensive. If it crashes with OOM:
- Reduce `SPARSE_BATCH_SIZE` to 16 in env/.env
- Add `--gpus '"device=1"'` to isolate it on a specific GPU

### Redis connection refused
```bash
docker logs redis
redis-cli -h localhost -p 6379 ping
```
Redis is required for caching but the pipeline works without it (just slower, no caching). Check if port 6379 is in use by another process.

### LLM returning very short responses
Check that:
1. `max_tokens` is set to at least 1024 in the request
2. LiteLLM is forwarding the parameter (check LiteLLM logs with `docker logs litellm`)
3. vLLM is not truncating due to `--max-model-len` being too low

### RAG proxy returns "Embedding failed"
The pipeline requires both the dense and sparse embedders to be running. Check:
```bash
curl http://localhost:8025/health  # dense
curl http://localhost:8035/health  # sparse
```

### Open WebUI shows no models
If Open WebUI can't reach LiteLLM:
1. Check that LiteLLM is healthy: `curl http://localhost:4000/health`
2. Check Open WebUI environment: `docker exec openwebui env | grep OPENAI`
3. Verify the URLs use Docker service names (http://litellm:4000) not localhost when running inside Docker

### Langfuse traces not appearing
```bash
# Test connectivity
curl http://localhost:3000/api/public/health

# Check that keys are set
python3 -c "import os; print(os.environ.get('LANGFUSE_SECRET_KEY', 'NOT SET')[:10])"
```
If LANGFUSE_SECRET_KEY is not set, tracing silently no-ops. That is by design.

### Dense ingestion is slow
Normal throughput on an A100: ~80 abstracts/second with batch_size=256. Factors that slow it down:
- Network latency to the vLLM container (use Docker network aliases, not `localhost`)
- Small batch size (increase to 256 or 512 if VRAM allows)
- JSONL file on slow storage (copy to NVMe first)

### Collection routing errors
If papers end up in the wrong collections, check the category strings in your JSONL. The router expects arXiv category codes like "cs.LG" or "hep-th". Run `route_paper` directly for debugging:
```bash
python3 -c "from pipeline.router import route_paper; print(route_paper('cs.LG cs.AI'))"
```

---

## Known Limitations

1. **Figures and diagrams:** The text-only indexed abstracts don't capture visual content. The `06_caption_figures.py` script addresses this for PDF ingestion, but the ~2.3M arXiv abstracts are text-only.

2. **Multi-column layouts:** PyMuPDF sometimes produces garbled text on two-column academic PDFs due to left-to-right reading order assumptions. The `unstructured` fallback handles most cases but is slower.

3. **No reranking in Open WebUI built-in RAG (Path A):** Open WebUI's native document RAG uses only cosine similarity (bi-encoder). Cross-encoder reranking only applies when using Path B (RAG proxy).

4. **arXiv abstracts only (not full text):** The 2.3M paper index contains titles and abstracts. Full-text indexing requires downloading and parsing PDFs individually (use `05_ingest_pdfs.py`).

5. **Equation rendering:** Mathematical equations in abstracts are stored as plain ASCII/unicode. LaTeX rendering in responses depends on Open WebUI's built-in KaTeX support.

6. **Cache invalidation:** The Redis cache has no mechanism for invalidating entries when new papers are indexed. Either set a short TTL or flush the cache after major ingestion runs:
   ```bash
   curl -X DELETE http://localhost:8002/v1/cache
   ```

7. **Single-node Qdrant:** The docker-compose uses a single Qdrant instance. For true HA, configure Qdrant cluster mode (see https://qdrant.tech/documentation/guides/distributed_deployment/).

---

## Repository Structure

```
arxiv-rag/
├── .env.example                        # Template for env/.env (safe placeholder values)
├── .gitignore                          # env/.env + data/ excluded
├── README.md                           # This file
├── requirements.txt                    # Root-level Python dependencies
│
├── pipeline/                           # Core RAG pipeline (imported by api/ and ingest/)
│   ├── __init__.py
│   ├── embeddings.py                   # Dense + sparse HTTP clients (calls services)
│   ├── router.py                       # Collection routing — query keywords → collections
│   ├── cache.py                        # Redis result cache (SHA-256 keyed)
│   ├── hybrid_search.py                # Qdrant Prefetch + FusionQuery(RRF) + async fan-out
│   ├── reranker.py                     # BGE-M3 cross-encoder via vLLM /score
│   ├── langgraph_pipeline.py           # LangGraph StateGraph (11 nodes)
│   └── tracer.py                       # Langfuse spans (no-op if keys absent)
│
├── query/                              # CLI search tools (all use HTTP services, no local model)
│   ├── __init__.py
│   ├── dense_search.py                 # Dense-only search via BGE-M3 vLLM (port 8025)
│   ├── sparse_search.py                # Sparse-only search via BGE-M3 FastAPI (port 8035)
│   └── hybrid_search.py                # Full hybrid + rerank (recommended)
│
├── api/                                # FastAPI RAG proxy (OpenAI-compatible)
│   ├── rag_proxy.py                    # /v1/chat/completions endpoint with source citations
│   ├── Dockerfile
│   └── requirements.txt
│
├── ingest/                             # Data ingestion scripts (run once)
│   ├── __init__.py
│   ├── 01_download_arxiv.py            # Stream Cornell-University/arxiv from HuggingFace → JSONL
│   ├── 02_create_collections.py        # Create 14 Qdrant collections (dense + sparse config)
│   ├── 03_ingest_dense.py              # BGE-M3 dense embed + upsert (Arxiv abstracts)
│   ├── 04_ingest_sparse.py             # BGE-M3 sparse embed + upsert (Arxiv abstracts)
│   ├── 05_ingest_pdfs.py               # *** PDFs only *** full-text + HDBSCAN clustering
│   └── 06_caption_figures.py           # *** PDFs only *** vision model figure captioning
│
├── sparse_embedder/                    # BGE-M3 SPLADE service (FastAPI, port 8035)
│   ├── sparse_embed.py                 # /encode + /health endpoints
│   ├── Dockerfile
│   └── requirements.txt
│
├── config/
│   ├── qdrant.yaml                     # Qdrant server config (GPU indexing, HNSW settings)
│   └── qdrant.yaml                     # Qdrant server config (GPU indexing, HNSW settings)
│
├── docker/
│   ├── docker-compose.yml              # Full stack with profiles (qdrant/embedding/core/rag-proxy/langfuse)
│   ├── qdrant_standalone.yml           # Qdrant standalone (GPU HNSW, llm-net)
│   ├── bge_m3_dense_embedder.yml       # vLLM dense embedder, port 8025, 12% GPU (production)
│   ├── bge_m3_dense_embedder_indexing.yml # vLLM dense embedder, port 8026, 50% GPU (ingestion)
│   ├── bge_m3_sparse_embedder.yml      # FastAPI sparse embedder, port 8035
│   ├── bge_m3_reranker.yml             # vLLM reranker, port 8020 (required extra flags)
│   ├── core_services.yml               # LiteLLM + Postgres + Open WebUI + Pipelines + Langflow
│   └── open_webui_standalone.yml       # Open WebUI pre-configured for Qdrant + BGE-M3
│
├── docs/
│   ├── embedding_speed.md              # VRAM budgets, throughput benchmarks, tuning guide
│   └── open_webui_rag_setup.md         # Open WebUI manual document upload guide
│
├── scripts/
│   ├── start_stack.sh                  # Start full stack with health polling
│   ├── start_core_services_small.sh    # Start core services (LiteLLM, WebUI, etc.)
│   ├── start_embedding_stack.sh        # Start all three embedding services
│   ├── start_indexing_mode.sh          # Switch dense embedder to 50% GPU for bulk ingestion
│   └── stop_indexing_mode.sh           # Switch back to 12% GPU production mode
│
└── env/
    └── .env                            # Your actual secrets — NEVER commit this file
```
