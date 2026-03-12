# UI Interfaces — Spark Scholar Stack

Every service in the stack that exposes a browser-accessible UI, plus notes on
how each one fits into the overall workflow.

---

## Quick Reference

| Interface       | URL                              | Purpose                                    |
|-----------------|----------------------------------|--------------------------------------------|
| Open WebUI      | http://localhost:8080            | Primary chat + document upload UI          |
| Qdrant Dashboard| http://localhost:6333/dashboard  | Vector DB browser — collections & search   |
| LiteLLM Admin   | http://localhost:4000/ui         | Model management, API keys, spend tracking |
| Langflow        | http://localhost:7860            | Visual pipeline / agent builder            |
| Langfuse        | http://localhost:3000            | Traces, spans, prompt versions, metrics    |
| SearXNG         | http://localhost:8888            | Private web search (also used by OW tools) |
| Redis/Valkey    | *(no built-in UI)*               | Cache — use RedisInsight if needed         |

---

## 1 · Open WebUI — http://localhost:8080

The primary day-to-day interface. Everything you do in a chat session starts here.

### What it does

- Full chat UI with model selector, streaming, system prompts
- Document uploads — paste a PDF or file and chat against it (Path A RAG via Qdrant)
- Web search — SearXNG integration for live search results in chat
- Multi-connection support — one instance can talk to LiteLLM local *and* the RAG proxy simultaneously
- Workspace — custom prompt templates, tool definitions, document collections

### Initial setup (first run)

1. Open http://localhost:8080 and create an admin account
   *(signup is disabled by default after first account — `ENABLE_SIGNUP=false`)*
2. Go to **Admin → Settings → Connections**
3. The primary connection (`http://litellm:4000`) is pre-configured via env vars
4. Add the additional connections manually:

   | Name           | URL                         | API Key        |
   |----------------|-----------------------------|----------------|
   | Spark Scholar  | http://rag-proxy:8002/v1    | simple-api-key |

5. After adding connections, the model list will show both `local-model` (SparkRun)
   and `spark-scholar` (RAG-augmented) in the model selector

### Key settings (Admin → Settings)

- **RAG** — embedding model is `bge-m3-embedder`, chunk size 1500 / overlap 200
- **Web Search** — SearXNG is pre-wired; enable per-chat with the globe icon
- **Documents** — uploaded docs go to Qdrant under the `openwebui_` prefix,
  keeping them separate from the arxiv/docs collections

### Two RAG paths

| Path | How to use | Best for |
|------|-----------|---------|
| **Path A — built-in RAG** | Upload a PDF/URL in chat, or via Workspace | Ad-hoc docs, one-off papers, personal files |
| **Path B — spark-scholar proxy** | Select `spark-scholar` model in model selector | Queries against the indexed arxiv/docs collections |

---

## 2 · Qdrant Dashboard — http://localhost:6333/dashboard

The built-in Qdrant web UI. Useful for inspecting collections, running manual
vector searches, and monitoring index health.

### What it does

- **Collections** tab — list all collections, see point counts, vector dimensions,
  index status, disk usage
- **Search** tab — run manual vector similarity searches; paste a JSON payload to
  test retrieval
- **Console** tab — raw REST API call interface (equivalent to `curl` but in-browser)
- **Graph** tab — experimental 2-D visualisation of vector clusters

### Common tasks

**Check a collection exists and has data:**
1. Collections → click `arxiv-cs-ml-ai` (or any collection name)
2. Verify `points_count` matches expected number of indexed chunks

**Manual search (test retrieval):**
1. Collections → `arxiv-cs-ml-ai` → Search
2. Paste the `query_vector` output from your embedder (1024-dim float array)
3. Adjust `limit`, add payload `filter` conditions, compare results

**Check indexing is complete:**
- Collections list shows a ⚡ icon while indexing is in progress
- When it disappears, HNSW indexing is done and search performance is optimal

### Collection naming convention

| Prefix      | Contents                        | Managed by               |
|-------------|----------------------------------|--------------------------|
| `arxiv-*`   | ArXiv paper chunks (14 topics)  | ingest/04 + ingest/05    |
| `docs-*`    | Documentation HTML/PDF chunks   | ingest/06 + ingest/07    |
| `openwebui_`| Open WebUI document uploads     | Open WebUI built-in RAG  |

See `ingest/02_create_collections.py` for the full collection list.

---

## 3 · LiteLLM Admin UI — http://localhost:4000/ui

LiteLLM's built-in management console. Login with the `LITELLM_MASTER_KEY` from
your `env/.env` (default: `simple-api-key`).

### What it does

- **Models** — view all registered models; add cloud models without restarting
- **API Keys** — generate per-team or per-user keys with budget limits
- **Usage / Spend** — request counts, token usage, cost tracking per model
- **Router** — view routing strategy, retry settings, fallback chains
- **Logs** — recent request/response log (useful for debugging)

### Adding cloud models at runtime

Rather than editing `litellm_local.yaml`, you can add cloud models via the UI:

1. **Models → + Add Model**
2. Provider: `openai` / `anthropic` / `gemini` / etc.
3. Model name: e.g. `claude-sonnet-4-5` for Anthropic, `gpt-4o` for OpenAI
4. API key: paste the key from your env file (or from the provider portal)
5. Save — the model is immediately available in Open WebUI

> Models added via the UI are stored in the LiteLLM Postgres DB (`litellm_db`
> volume) and survive container restarts.

### Reference YAML for cloud models

See `core_services/litellm_cloud.yaml` for a pre-populated reference covering:
OpenAI GPT-4o/o4-mini, Anthropic Claude (Opus/Sonnet/Haiku), Google Gemini 2.5,
NVIDIA NIM, xAI Grok, DeepSeek V3/R1, Perplexity Sonar, OpenRouter.

Copy the entries you want into the UI, or temporarily mount the file:
```yaml
# in core_services/core_services.yml, litellm service:
volumes:
  - ./litellm_cloud.yaml:/app/cloud.yaml:ro
# and add to command:
command: ["--config", "/app/litellm_local.yaml", "--config", "/app/cloud.yaml", ...]
```

---

## 4 · Langflow — http://localhost:7860

A visual, drag-and-drop pipeline builder for LLM workflows. Think of it as a
no-code / low-code way to prototype and iterate on LangChain-style pipelines.

### What it does

- Drag components (LLM, embedder, retriever, tool, prompt, etc.) onto a canvas
- Wire them together with edges to form a flow
- Run flows interactively or export them as JSON for use in production
- Built-in component library includes LiteLLM, OpenAI, Qdrant, Redis, SearXNG, etc.

### In this stack

Langflow points at LiteLLM local (`http://litellm:4000/v1`) by default, which
means every flow you build can use:
- `local-model` — SparkRun primary LLM
- `phi-mini` — fast/light model for classification, routing, summarisation
- `bge-m3-embedder` — BGE-M3 dense embeddings

To use cloud models, add them to LiteLLM via the admin UI first — they'll
automatically appear in Langflow's model dropdowns.

### Typical uses

| Use case | Components |
|----------|-----------|
| Rapid prototype a RAG pipeline | Qdrant retriever → Prompt → LiteLLM |
| Test a new prompt template | Prompt → LiteLLM → Output |
| Build a multi-step research agent | Web search → Summarise → Rerank → Answer |
| Export a tested flow to production code | Export JSON → convert to LangGraph |

Flows are persisted in the `spark_scholar_langflow_data` Docker volume.

---

## 5 · Langfuse — http://localhost:3000

Observability and tracing for the RAG pipeline. Langfuse is **optional** — the
stack runs without it, but it dramatically helps with debugging and tuning.

### Start Langfuse

```bash
docker compose -f core_services/langfuse.yml up -d
```

First run: open http://localhost:3000 → create account → create project → copy API keys → add to `env/.env`:

```env
LANGFUSE_SECRET_KEY=sk-lf-xxxx
LANGFUSE_PUBLIC_KEY=pk-lf-xxxx
LANGFUSE_HOST=http://localhost:3000
```

Restart `rag-proxy` to pick up the keys:
```bash
docker compose -f rag_proxy/rag_proxy.yml restart rag-proxy
```

### What it shows

Every request through the RAG proxy creates a **trace** with nested **spans**:

```
Trace: user query "What is RLHF?"
  ├── Span: encode_query          (BGE-M3 dense + sparse)
  ├── Span: qdrant_hybrid_search  (per-collection Prefetch + RRF)
  ├── Span: rerank                (BGE cross-encoder top-50 → top-10)
  ├── Span: context_injection     (format retrieved chunks)
  └── Span: litellm_completion    (LLM call, token counts, latency)
```

### Key views

- **Traces** — full timeline for every request; click any span to see input/output
- **Generations** — just the LLM calls; compare token usage, cost, latency
- **Prompt** — version-control your system prompts; link each version to its traces
- **Metrics / Dashboard** — p50/p95 latency, error rate, token spend over time
- **Sessions** — group traces by conversation for multi-turn analysis

### Tuning workflow

1. Run a batch of test queries
2. Open Traces, filter by low scores or high latency
3. Inspect the `qdrant_hybrid_search` span — are the right chunks being retrieved?
4. If recall is poor: increase `RAG_RERANK_TOP_N` or adjust chunking parameters
5. If latency is high: check which span dominates; reduce `RAG_TOP_K` if the LLM call is the bottleneck

---

## 6 · SearXNG — http://localhost:8888

A self-hosted, privacy-preserving meta-search engine that aggregates results from
Google, Bing, DuckDuckGo, arXiv, GitHub, and many more — without tracking.

### Dual role in this stack

1. **Open WebUI web search tool** — click the 🌐 icon in a chat to search the web
   and inject results as context before the LLM answers
2. **RAG proxy web search** — the proxy can optionally query SearXNG for
   fresh web content before retrieval (configurable in pipeline settings)

### Using the UI

The SearXNG UI at port 8888 is a fully functional search engine:
- Search normally — results come from multiple engines, aggregated and deduplicated
- Choose categories: General, News, Science, Files, Images, etc.
- Use `!` bangs for direct engine routing: `!gh langchain` for GitHub, `!arxiv RLHF` for arXiv

### Configuration

The SearXNG config lives in `core_services/searxng/settings.yml` (if customised).
By default it runs with the built-in config and is accessible only on the Docker
host — it is not exposed to the internet.

---

## 7 · Redis / Valkey — *(no built-in UI)*

Redis/Valkey provides the response cache for the RAG proxy. There is no built-in
browser UI, but you can inspect it using the CLI or an optional GUI tool.

### CLI access

```bash
# Connect to the running Valkey container
docker exec -it redis redis-cli

# Useful commands
KEYS *                  # list all cached query keys (use carefully on large caches)
DBSIZE                  # total number of cached entries
TTL "rag:query:xxxxx"   # check remaining TTL on a specific key
FLUSHDB                 # clear all cached responses (force re-retrieval)
INFO memory             # memory usage stats
```

### Optional GUI — RedisInsight

RedisInsight is a free desktop/web GUI for Redis/Valkey:
```bash
docker run -d -p 5540:5540 redislabs/redisinsight:latest
```
Open http://localhost:5540 → Add Redis Database → Host: `localhost`, Port: `6379`.

### Cache behaviour

The RAG proxy caches full `(query, model, collection)` responses for `CACHE_TTL_SECONDS`
(default 86400 = 24 hours). This means:

- Repeated identical queries return instantly (no embedding/retrieval/LLM call)
- After re-ingesting new papers: `FLUSHDB` to invalidate stale cached responses
- For fast-changing datasets, lower `CACHE_TTL_SECONDS` to `3600` (1 hour)

---

## Summary: Which UI for which task?

| Task | Go to |
|------|-------|
| Chat with local LLM | Open WebUI → 8080 |
| Chat with RAG over arxiv papers | Open WebUI → select `spark-scholar` model |
| Upload and chat over a PDF | Open WebUI → 8080 → 📎 attach |
| Check how many papers are indexed | Qdrant Dashboard → 6333/dashboard |
| Add a cloud model (OpenAI, Claude, etc.) | LiteLLM Admin → 4000/ui |
| Debug why a query returned bad results | Langfuse → 3000 → Traces |
| Prototype a new pipeline | Langflow → 7860 |
| Search the web privately | SearXNG → 8888 |
| Clear the response cache | Redis CLI → FLUSHDB |
