# Search, Retrieval & Pipeline

← [Back to README](../README.md)

---

## Table of Contents

- [CLI Search Tools](#cli-search-tools)
  - [Dense search](#dense-search--semantic-similarity)
  - [Sparse search](#sparse-search--keyword-precision)
  - [Hybrid search (recommended)](#hybrid-search--full-pipeline-recommended)
  - [Comparing all three modes](#comparing-all-three-modes)
- [Why Hybrid?](#why-hybrid)
- [Qdrant RRF Implementation](#qdrants-native-rrf-implementation)
- [Reranking](#reranking)
- [LangGraph Pipeline](#langgraph-pipeline)
- [Caching Strategy](#caching-strategy)

---

## CLI Search Tools

Three CLI tools in `query/` cover every search mode. All call embedding services
over HTTP — they do not load any model locally and are safe to run alongside the
full production stack.

### Services required

| Script | Services needed |
|---|---|
| `query/dense_search.py` | BGE-M3 dense (port 8025) + Qdrant (port 6333) |
| `query/sparse_search.py` | BGE-M3 sparse (port 8035) + Qdrant (port 6333) |
| `query/hybrid_search.py` | dense + sparse + reranker (port 8020) + Qdrant |

```bash
# Start embedding services if not running
bash scripts/start_stack.sh embedding
```

---

### Dense search — semantic similarity

Best for: concept-level queries, synonyms, paraphrase variation, cross-domain discovery.

```bash
# Simple query — searches all collections and RRF-merges results
python query/dense_search.py --query "variational inference latent variable models"

# Restrict to one collection
python query/dense_search.py --query "RLHF alignment language models" \
    --collection arxiv-cs-ml-ai

# With year filter
python query/dense_search.py --query "stochastic differential equations" \
    --year-min 2020 --year-max 2024

# More results + JSON output
python query/dense_search.py --query "high frequency trading market microstructure" \
    --top-k 20 --json

# Filter by author
python query/dense_search.py --query "diffusion models" --author "Ho"
```

**How it works:** The query is encoded to a 1024-dim vector, then an HNSW approximate
nearest-neighbour search runs in each collection's `dense_embedding` slot. Results
across collections are merged with RRF.

---

### Sparse search — keyword precision

Best for: specific model names (`GPT-4`, `LLaMA-3.1`), author names, arXiv IDs,
equations, chemical formulae — any query where the exact token matters.

```bash
# Keyword query
python query/sparse_search.py --query "attention mechanism transformers self-attention"

# Author + title keywords
python query/sparse_search.py --query "Vaswani attention all you need" \
    --year-min 2017 --year-max 2018

# Search by arXiv ID
python query/sparse_search.py --query "2305.14314"

# Restrict to one collection
python query/sparse_search.py --query "Hawkes process excitation kernel" \
    --collection arxiv-stat-eess

# JSON output for programmatic use
python query/sparse_search.py --query "bge-m3 embedding multilingual" --json
```

**How it works:** The query is encoded to SPLADE-style (token_id → weight) pairs.
Qdrant's inverted index scores the dot product against each document's `sparse_text`
slot. Results across collections are merged with RRF.

---

### Hybrid search — full pipeline (recommended)

Best for: everything. Combines dense + sparse via Qdrant's native server-side RRF,
then cross-encoder reranks the top candidates. This is the same pipeline the RAG
proxy runs for every chat message.

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
1. **Route** — `pipeline/router.py` maps the query to 1–3 subject collections using keyword heuristics
2. **Encode** — dense and sparse vectors computed in parallel (async)
3. **Retrieve** — each collection runs `Prefetch([dense, sparse]) + FusionQuery(RRF)` natively inside Qdrant
4. **Merge** — per-collection results merged with a second Python-side RRF pass
5. **Rerank** — top 50 candidates scored jointly by the BGE-M3 cross-encoder
6. **Return** — top 10 with `rerank_score`, `rrf_score`, `arxiv_id`, title, authors, abstract

**Verbose timing output:**
```
Routing: 'diffusion models image generation' → ['arxiv-cs-ml-ai']    [routing]
Retrieved 48 candidates in 0.31s                                      [retrieve]
Reranked to 10 in 1.42s                                               [rerank]
Total: 1.73s                                                          [total]
```

---

### Comparing all three modes

```bash
QUERY="contrastive learning self-supervised representations"

echo "=== DENSE ===" && python query/dense_search.py  --query "$QUERY" --top-k 5
echo "=== SPARSE ===" && python query/sparse_search.py --query "$QUERY" --top-k 5
echo "=== HYBRID ===" && python query/hybrid_search.py --query "$QUERY" --top-k 5
```

The hybrid results typically include high-recall papers from dense (conceptually
related but different terminology) plus high-precision papers from sparse (exact
term matches), then cross-encoder reranking promotes the most relevant of both.

---

## Why Hybrid?

**Dense retrieval** (HNSW on bi-encoder vectors) excels at semantic similarity —
finding papers about the same concept even if they use different terminology.
"Neural network" and "deep learning" map to nearby points in the embedding space.

**Sparse retrieval** (SPLADE inverted index) excels at exact term matching —
catching specific model names, equation identifiers, author names, and technical
acronyms that dense vectors tend to blur. Searching for "bge-m3 performance on BEIR"
with dense might return broadly relevant papers; sparse finds papers that literally
mention "BGE-M3".

You need both. Hybrid retrieval consistently beats either modality alone across all
BEIR benchmarks — the canonical result in information retrieval research.

---

## Qdrant's Native RRF Implementation

The pipeline uses Qdrant's first-class `Prefetch + FusionQuery(RRF)` support —
fusion happens inside the Qdrant server, not in Python:

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

This is more efficient than Python-side fusion: Qdrant uses SIMD-optimised ranking,
avoids re-serialising results over the network, and applies payload filters after fusion.

### Cross-collection merging

When multiple collections are searched (e.g. the query spans cs.ML and stat.ML),
each collection returns its own RRF-fused list. A second Python-side RRF pass merges
them. Most queries only search 1–2 collections based on `pipeline/router.py` heuristics.

---

## Reranking

After hybrid search returns up to 50 candidates, the BGE-M3 cross-encoder scores
each `(query, document)` pair jointly:

```
POST http://localhost:8020/score
{
    "model": "bge-reranker-v2-m3",
    "text_1": "What is the attention mechanism?",
    "text_2": ["Abstract of paper 1...", "Abstract of paper 2...", ...]
}
```

The cross-encoder reads the full query and document together in a single forward
pass, catching:
- **Negation** — "not quantum computing" vs. "quantum computing"
- **Multi-hop reasoning** — "paper that cites X and uses method Y"
- **Domain disambiguation** — "kernel in SVM" vs. "kernel in operating systems"

The returned top-10 are substantially more relevant than the initial 50 candidates.

**Latency budget:** The full pipeline (embed + retrieve + rerank + LLM) takes 3–8
seconds. The reranker adds ~1–2 seconds for 50 candidates.

---

## LangGraph Pipeline

The pipeline is implemented as a LangGraph `StateGraph` with typed state (`RAGState`
TypedDict). This gives deterministic execution, async node scheduling, built-in state
management, and easy conditional branching.

### Graph structure

```
START → check_cache
check_cache → END                    (cache hit)
check_cache → route_query            (cache miss)
route_query → embed_query
embed_query → hybrid_retrieve
hybrid_retrieve → should_web_search  (conditional)
should_web_search → web_search       (time-sensitive query)
should_web_search → merge_results    (otherwise)
web_search → merge_results
merge_results → rerank_results
rerank_results → build_context
build_context → llm_inference
llm_inference → cache_result
cache_result → trace_result
trace_result → END
```

### Node descriptions

| Node | What it does |
|---|---|
| `check_cache` | SHA-256(query + collections) → Redis lookup |
| `route_query` | Keyword heuristics → 1–3 Qdrant collection names |
| `embed_query` | Async BGE-M3 dense + sparse encoding |
| `hybrid_retrieve` | Qdrant Prefetch + RRF per collection, Python-side merge |
| `should_web_search` | Decides if query needs live web results (time, news keywords) |
| `web_search` | SearXNG query → top-20 results → merged with Qdrant candidates |
| `merge_results` | Python RRF across all sources |
| `rerank_results` | BGE cross-encoder top-50 → top-10 |
| `build_context` | Format retrieved chunks + metadata for system prompt |
| `llm_inference` | LiteLLM streaming chat completion |
| `cache_result` | Store response in Redis with TTL |
| `trace_result` | Emit Langfuse spans (no-op if keys absent) |

### Conditional edges

The `should_web_search` node checks for time-sensitive keywords ("today", "latest",
"current year", news topics). If triggered, the web search result is merged with
Qdrant candidates before reranking — not used as a replacement.

---

## Caching Strategy

Redis caches complete RAG pipeline outputs (full LLM response + top reranked sources).

**Cache key:** `SHA-256(query.strip().lower() + sorted(collections))`

- Same query with different capitalisation → same key
- Same query routing to different collections → different key
- Filters (year_min, year_max, author) → different key (via `extra` param)

**TTL:** `CACHE_TTL_SECONDS` (default `86400` = 24 hours).
Recommended values:
- Single-user, stable dataset: `86400` (24 h)
- Multi-user or fast-changing data: `3600` (1 h)

**What is cached:** Full response text + top-5 source metadata. Intermediate vectors,
candidates, and reranked list are NOT cached (Redis memory optimisation).

**Cache miss path:** If Redis is unreachable, `ResultCache.get()` returns `None` and
the pipeline runs normally. The pipeline never fails due to cache unavailability.

**Invalidating after ingestion:**
```bash
# Clear all cached responses (force re-retrieval on next query)
docker exec -it redis redis-cli FLUSHDB
# or via the RAG proxy:
curl -X DELETE http://localhost:8002/v1/cache
```
