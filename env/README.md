# env/ — Environment Configuration

## Setup (do this first)

```bash
cp .env.example env/.env
# then open env/.env and fill in your values
```

The `.env` file is **gitignored** — it never gets committed. Only `.env.example` (safe placeholder values) lives in the repo.

---

## Required variables

| Variable | Description | Default |
|---|---|---|
| `VLLM_URL` | SparkRun inference server endpoint | `http://host.docker.internal:8000` |
| `VLLM_API_KEY` | SparkRun vLLM API key | `simple-api-key` |
| `LITELLM_MASTER_KEY` | LiteLLM admin key — **change this** | — |
| `QDRANT_API_KEY` | Qdrant API key | `simple-api-key` |
| `BGE_M3_API_KEY` | BGE-M3 service key | `simple-api-key` |

## Optional variables

| Variable | Description | Default |
|---|---|---|
| `LANGFUSE_SECRET_KEY` | Langfuse tracing — leave unset to disable | — |
| `LANGFUSE_PUBLIC_KEY` | Langfuse tracing | — |
| `LANGFUSE_HOST` | Langfuse server | `http://localhost:3000` |
| `CACHE_TTL_SECONDS` | RAG response cache TTL | `86400` (24 h) |
| `RAG_TOP_K` | Qdrant results before reranking | `10` |
| `RAG_RERANK_TOP_N` | Candidates passed to reranker | `50` |

---

## How Docker Compose loads it

Every service file contains:
```yaml
env_file:
  - ../env/.env
```
Docker injects all variables at container startup — nothing else needed.

## How Python scripts load it

```python
from dotenv import load_dotenv
load_dotenv("env/.env")   # or load_dotenv() if run from repo root
```

## The `simple-api-key` pattern

All services on the internal `llm-net` Docker network use `simple-api-key`.  
This prevents open access from misconfigured containers — these services are **not** internet-facing.  
`LITELLM_MASTER_KEY` is the exception — generate a strong random value:

```bash
openssl rand -base64 32
```

## Inference model

The LLM (inference model) is **not** started by docker-compose. Use SparkRun:

```
github.com/scitrera/oss-spark-run
```

Set `VLLM_URL=http://host.docker.internal:8000` so containers on `llm-net` can reach the SparkRun process on the host.
