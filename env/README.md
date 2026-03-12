# env/ — Environment Configuration

## Setup (do this first)

```bash
cp .env.example env/.env
nano env/.env   # fill in values — see sections below
```

The `.env` file is **gitignored** — it never gets committed.
`.env.example` (safe placeholders) is the only file in this folder that's in git.

---

## Required variables (stack won't start without these)

| Variable | Value | Notes |
|---|---|---|
| `VLLM_URL` | `http://host.docker.internal:8000` | SparkRun inference server |
| `VLLM_API_KEY` | `simple-api-key` | Must match `--api-key` in SparkRun |
| `VLLM_MODEL_NAME` | `nvidia/Llama-3.1-Nemotron-70B-...` | Model name as loaded by SparkRun |
| `LITELLM_MASTER_KEY` | generate with `openssl rand -base64 32` | Admin key — change this |
| `WEBUI_SECRET_KEY` | generate with `openssl rand -base64 32` | Session signing — change this |
| `HF_TOKEN` | `hf_...` | HuggingFace token for dataset + gated models |

---

## Local service keys (all use `simple-api-key`)

These are on the private `llm-net` Docker network — not internet-facing.

| Variable | Value |
|---|---|
| `QDRANT_API_KEY` | `simple-api-key` |
| `BGE_M3_API_KEY` | `simple-api-key` |
| `LITELLM_API_KEY` | `simple-api-key` |
| `OPENAI_API_KEY` | `simple-api-key` |
| `SEARXNG_API_KEY` | `simple-api-key` |
| `REDIS_URL` | `redis://localhost:6379` |
| `CACHE_TTL_SECONDS` | `86400` (24 h) |

---

## Optional: Langfuse tracing

```bash
# Self-hosted:
docker compose -f core_services/langfuse.yml up -d
# Open http://localhost:3000, create account, project, API keys

LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

Leave as placeholder to disable tracing (pipeline silently no-ops).

---

## Optional: Academic research API keys

Used by Open WebUI workspace tools for literature search.
All optional — tools degrade gracefully without a key.

| Variable | Free tier | Register at |
|---|---|---|
| `SEMANTIC_SCHOLAR_API_KEY` | 100 req/5min | [semanticscholar.org/product/api](https://www.semanticscholar.org/product/api) |
| `OPENALEX_API_KEY` | 10 req/s → 100/s with key | [openalex.org](https://openalex.org/) |
| `NCBI_API_KEY` | 3 req/s → 10/s with key | [ncbi.nlm.nih.gov/account](https://www.ncbi.nlm.nih.gov/account/) |
| `CORE_API_KEY` | — | [core.ac.uk/services/api](https://core.ac.uk/services/api) |
| `CROSSREF_MAILTO` | Email for polite pool | [crossref.org](https://www.crossref.org/documentation/retrieve-metadata/rest-api/) |

---

## Optional: Cloud LLM API keys

Used by `core_services/litellm_cloud.yaml`. See comments in that file.
Leave blank for fully local operation.

| Variable | Provider | Get key at |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI GPT-4o, o3 | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| `ANTHROPIC_API_KEY` | Claude Opus/Sonnet/Haiku | [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) |
| `GEMINI_API_KEY` | Gemini 2.5 Pro/Flash | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) |
| `NVIDIA_API_KEY` | Nemotron cloud (free tier) | [build.nvidia.com](https://build.nvidia.com) |
| `XAI_API_KEY` | Grok 3 | [console.x.ai](https://console.x.ai/) |
| `DEEPSEEK_API_KEY` | DeepSeek V3/R1 | [platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys) |
| `PERPLEXITY_API_KEY` | Sonar, Sonar Deep Research | [perplexity.ai/settings/api](https://www.perplexity.ai/settings/api) |
| `OPENROUTER_API_KEY` | 200+ models via one key | [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys) |

To activate cloud models: mount `litellm_cloud.yaml` in `core_services/core_services.yml`
or add models via the LiteLLM admin UI at `http://localhost:4000/ui`.

---

## SparkRun inference model

The LLM is **not** managed by docker-compose.

```bash
# github.com/scitrera/oss-spark-run
sparkrun start <model-name>    # starts vLLM on port 8000
```

Set `VLLM_MODEL_NAME` to match whatever model you load.
Containers reach it via `host.docker.internal:8000` (added by LiteLLM's `extra_hosts`).
