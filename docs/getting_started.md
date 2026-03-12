# Getting Started

← [Back to README](../README.md)

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Startup Order](#startup-order)
- [Embedding Stack](#embedding-stack)
- [LiteLLM Setup](#litellm-setup)

---

## Prerequisites

### Hardware
- NVIDIA DGX Spark (or any system with NVIDIA GPU, 64 GB+ RAM, 2 TB+ NVMe)
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

### SparkRun (required — manages vLLM inference)
SparkRun is a CLI tool that starts and manages vLLM on DGX Spark.
It is **not** managed by docker-compose — it runs alongside the stack.
```bash
# Install SparkRun: https://github.com/scitrera/oss-spark-run
# Start your inference model before starting the stack:
sparkrun start Qwen/Qwen3-30B-A3B
# or: sparkrun start nvidia/Nemotron-H-56B-Instruct-HF
```
SparkRun exposes vLLM at `http://localhost:8000`. The LiteLLM container reaches it
via `host.docker.internal:8000` (set automatically via `extra_hosts`).

---

## Configuration

### Setting up your .env file

The entire stack reads from a single `env/.env` file.

```bash
# 1. Copy the template — all local service keys are already pre-filled
cp .env.example env/.env

# 2. Set the one required value: your SparkRun model name
nano env/.env
# → Set VLLM_MODEL_NAME to match what vLLM reports:
#   GET http://localhost:8000/v1/models

# 3. Optional: generate stronger security keys for production use
openssl rand -base64 32   # use output as LITELLM_MASTER_KEY / WEBUI_SECRET_KEY
```

**For a single-user DGX Spark on a private network:**
The default `simple-api-key` for all local services is fine. The two keys worth
changing if you ever share access:
- `LITELLM_MASTER_KEY` — admin access to the LiteLLM UI (add/remove models)
- `WEBUI_SECRET_KEY` — signs Open WebUI session cookies

**Minimum required values for a fully local stack:**
```bash
VLLM_MODEL_NAME=<model name you loaded via SparkRun>
```
Everything else in `.env.example` has working defaults.

---

### Environment variable reference

| Variable | Default | Description |
|---|---|---|
| `VLLM_URL` | `http://host.docker.internal:8000` | SparkRun vLLM endpoint (inside containers) |
| `VLLM_MODEL_NAME` | `nvidia/Llama-3.1-Nemotron-70B-Instruct-HF` | **Set this to your loaded model** |
| `VLLM_PHI_URL` | `http://host.docker.internal:8001` | Optional Phi Mini on port 8001 |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant REST API URL |
| `QDRANT_API_KEY` | `simple-api-key` | Qdrant API key |
| `BGE_M3_DENSE_URL` | `http://localhost:8025` | vLLM serving BAAI/bge-m3 dense embeddings |
| `BGE_M3_SPARSE_URL` | `http://localhost:8035` | Custom FastAPI sparse embedder |
| `BGE_RERANKER_URL` | `http://localhost:8020` | vLLM serving bge-reranker-v2-m3 |
| `BGE_M3_API_KEY` | `simple-api-key` | Shared API key for all BGE services |
| `LITELLM_URL` | `http://localhost:4000` | LiteLLM proxy URL |
| `LITELLM_API_KEY` | `simple-api-key` | API key for user requests |
| `LITELLM_MASTER_KEY` | `simple-api-key` | Admin key — change for production |
| `REDIS_URL` | `redis://localhost:6379` | Redis/Valkey connection URL |
| `LANGFUSE_SECRET_KEY` | _(empty)_ | Langfuse — tracing disabled if unset |
| `LANGFUSE_PUBLIC_KEY` | _(empty)_ | Langfuse public key |
| `LANGFUSE_HOST` | `http://localhost:3000` | Langfuse server URL |
| `WEBUI_SECRET_KEY` | `simple-api-key` | Open WebUI session signing key |
| `HF_TOKEN` | _(empty)_ | HuggingFace token — needed to download arXiv dataset |
| `SEARXNG_URL` | `http://localhost:8888` | SearXNG for web search |
| `RAG_PROXY_PORT` | `8002` | Port for the RAG proxy FastAPI server |
| `RAG_TOP_K` | `10` | Final retrieved results after reranking |
| `RAG_RERANK_TOP_N` | `50` | Candidates passed to cross-encoder |
| `CACHE_TTL_SECONDS` | `86400` | Redis TTL for cached responses (24 h) |
| `CHUNK_SIZE` | `1000` | PDF/HTML chunk size in tokens |
| `CHUNK_OVERLAP` | `150` | Overlap between consecutive chunks |

---

## Startup Order

The recommended way is the all-in-one script:

```bash
# Full stack (Qdrant → Embedding → Core → RAG Proxy)
./scripts/start_stack.sh

# Partial starts (useful after a reboot)
./scripts/start_stack.sh embedding    # embedding services only
./scripts/start_stack.sh core         # core services only (requires embedding running)
```

The script polls health endpoints between stages and exits with an error if any
service fails to come up within the timeout. It also creates `llm-net` if missing.

---

### Manual startup (step by step)

If you prefer manual control or need to start only specific services:

#### Step 1: Qdrant
```bash
docker compose -f core_services/qdrant.yml up -d
# Wait for healthy:
curl -sf http://localhost:6333/readyz
```

#### Step 2: Embedding services (GPU required)
```bash
# Dense embedder (~3 GB VRAM production / ~12 GB indexing mode)
docker compose -f embedding/bge_m3_dense.yml up -d

# Sparse embedder (~3 GB VRAM)
docker compose -f embedding/bge_m3_sparse.yml build --pull  # first time only
docker compose -f embedding/bge_m3_sparse.yml up -d

# Cross-encoder reranker (~2 GB VRAM)
docker compose -f embedding/bge_m3_reranker.yml up -d

# Check health (model download takes 2-5 min on first run)
curl -sf http://localhost:8025/health   # dense
curl -sf http://localhost:8035/health   # sparse
curl -sf http://localhost:8020/health   # reranker
```

#### Step 3: Core services
```bash
# Redis/Valkey cache
docker compose -f core_services/redis.yml up -d

# SearXNG private web search
docker compose -f core_services/searxng.yml up -d

# LiteLLM proxy + Open WebUI + Langflow
docker compose -f core_services/core_services.yml up -d

# Wait for LiteLLM
until curl -sf http://localhost:4000/health/readiness; do sleep 3; done
```

#### Step 4: RAG Proxy
```bash
docker compose -f rag_proxy/rag_proxy.yml build   # first time only
docker compose -f rag_proxy/rag_proxy.yml up -d
curl -sf http://localhost:8002/health
```

#### Step 5: Langfuse (optional — observability)
```bash
docker compose -f core_services/langfuse.yml up -d
# Then set LANGFUSE_SECRET_KEY + LANGFUSE_PUBLIC_KEY in env/.env
# and restart rag-proxy:
docker compose -f rag_proxy/rag_proxy.yml restart rag-proxy
```

---

### Switching to indexing mode

During bulk ingestion, switch the dense embedder to use 50% GPU for maximum throughput.
The inference model continues running alongside.

```bash
bash scripts/start_indexing_mode.sh   # 50% GPU, port 8025 (replaces production)
# ... run ingestion ...
bash scripts/stop_indexing_mode.sh    # back to 12% GPU production mode
```

---

## Embedding Stack

### Dense Embedder (port 8025)
- **Model:** BAAI/bge-m3 served by vLLM
- **Output:** 1024-dimensional float32 L2-normalised vectors
- **VRAM:** ~3 GB production (12% GPU util) / ~12 GB indexing (50% GPU util)
- **Throughput:** ~1,500 abstracts/second on DGX Spark (batch size 256, indexing mode)
- **API:** OpenAI `/v1/embeddings` compatible
- **Why BGE-M3:** State-of-the-art multilingual model (100+ languages), 8192 token context, outperforms OpenAI ada-002 on MTEB benchmarks

### Sparse Embedder (port 8035)
- **Model:** BAAI/bge-m3 via FlagEmbedding (SPLADE-style lexical weights)
- **Output:** Sparse vector of (token_id, weight) pairs — typically 200–500 non-zero dimensions
- **VRAM:** ~3 GB fp16
- **Throughput:** ~300 abstracts/second (batch size 32, CPU-bound tokenisation)
- **API:** Custom FastAPI `/encode` endpoint
- **Why sparse:** Captures exact term matches that dense vectors miss — critical for model names, acronyms, equation references, author names

### Reranker (port 8020)
- **Model:** BAAI/bge-reranker-v2-m3 served by vLLM
- **VRAM:** ~2 GB fp16
- **Throughput:** ~50 query-document pairs/second
- **API:** vLLM `/score` endpoint
- **Why cross-encoder:** Attends jointly over (query, document) pairs — catches negation, multi-hop reasoning, and domain disambiguation that bi-encoders miss. Applied only to top-50 candidates to keep latency manageable.

### VRAM Budget (DGX Spark — 128 GB unified memory)

| Service | Mode | GPU util | VRAM |
|---|---|---|---|
| SparkRun vLLM (Qwen3-30B-A3B) | production | auto | ~40–50 GB |
| SparkRun vLLM (Nemotron-56B) | production | auto | ~80–100 GB |
| Phi Mini (optional, port 8001) | production | auto | ~4–8 GB |
| BGE-M3 dense embedder | production | 0.12 | ~3 GB |
| BGE-M3 sparse embedder | production | fp16 | ~3 GB |
| BGE-M3 reranker | production | 0.15 | ~2 GB |
| **Total (Qwen3-30B + all services)** | | | **~52–63 GB** |
| **Total (Nemotron-56B + all services)** | | | **~88–108 GB** |

During bulk indexing (inference model still running alongside):

| Service | Mode | GPU util | VRAM |
|---|---|---|---|
| BGE-M3 dense embedder | indexing | 0.50 | ~12 GB |
| BGE-M3 sparse embedder | indexing | fp16 | ~3 GB |
| **Embedding only (indexing mode)** | | | **~15 GB** |

See `docs/embedding_speed.md` for full throughput benchmarks and tuning guide.

---

## LiteLLM Setup

LiteLLM provides a unified OpenAI-compatible gateway in front of all models. Every
service (Open WebUI, Langflow, RAG proxy, scripts) talks to LiteLLM instead of
directly to vLLM, giving you:

- Centralised API key management
- Request logging and spend tracking
- Automatic retries and fallbacks
- Model aliasing (`local-model` instead of the full model path)

### Configuration files

| File | Purpose |
|---|---|
| `core_services/litellm_local.yaml` | Mounted by default — SparkRun + BGE-M3 models |
| `core_services/litellm_cloud.yaml` | Reference only — cloud models (OpenAI, Anthropic, etc.) |

Cloud models can be added at runtime via the LiteLLM admin UI (`http://localhost:4000/ui`)
without restarting the container — they are persisted in the Postgres DB.

### The `simple-api-key` pattern

All services on `llm-net` use `simple-api-key` for normal requests:
```python
headers = {"Authorization": "Bearer simple-api-key"}
```
The `LITELLM_MASTER_KEY` is only required for admin operations (model management via the UI).

### Phi Mini — secondary lightweight model

For a second small model alongside your primary LLM:
```bash
# Start Phi-4-mini on port 8001 via SparkRun
sparkrun start microsoft/Phi-4-mini-instruct --port 8001
```
It registers automatically as `phi-mini` / `phi` in LiteLLM (see `litellm_local.yaml`).

**When to use Phi Mini instead of the primary model:**
- Query classification / collection routing
- Fast tool calling (Phi 4 is fine-tuned for function calling)
- Simple code completion in Langflow prototypes
- When primary model is at >80% VRAM utilisation under concurrent load
- Any task where a 30B+ model is overkill and speed matters
