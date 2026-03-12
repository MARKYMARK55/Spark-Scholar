# Embedding Stack — Speed Optimisation Guide

Complete reference for throughput, VRAM allocation, and tuning the embedding
services on the DGX Spark. Covers both the **production mode** (low VRAM, runs
alongside the main inference model) and **indexing mode** (maximum throughput
for bulk ingestion of the Arxiv corpus).

---

## The two operating modes

The embedding stack has fundamentally different requirements depending on whether
you are serving live queries or doing offline bulk indexing.

| | Production mode | Indexing mode |
|---|---|---|
| **Use when** | Answering queries alongside vLLM | Bulk ingesting Arxiv/PDFs |
| **GPU utilisation** | 12–15% per service | 50% dense, dedicated sparse |
| **Batch size** | 8–16 sequences | 256+ dense, 64 sparse |
| **Throughput (dense)** | ~200–400 docs/s | ~800–1,200 docs/s |
| **Throughput (sparse)** | ~300–500 docs/s | ~1,500–2,000 docs/s |
| **vLLM flag** | `--gpu-memory-utilization 0.12` | `--gpu-memory-utilization 0.50` |
| **Impact on inference** | None — fits in headroom | Stop inference model first |

Switch to indexing mode with the convenience script:
```bash
./scripts/start_embedding_stack.sh   # stops production embedder, starts indexing variant
```

Switch back after bulk indexing:
```bash
cd docker
docker compose --profile embedding up -d bge-m3-dense-embedder   # production (0.12)
docker stop bge-m3-embedder-indexing
```

---

## Dense embedder (port 8025) — detailed tuning

### Why vLLM for dense embedding?

vLLM's continuous batching is substantially more efficient than calling
`SentenceTransformer.encode()` directly for serving. At low GPU utilisation
(12%) it still achieves competitive throughput because:
- Requests from multiple concurrent callers are batched automatically
- Prefix caching (`--enable-prefix-caching`) avoids recomputing common prefixes
- The CUDA kernels are TensorRT-optimised by vLLM's compilation pipeline

### Key vLLM flags (dense)

```yaml
command:
  - vllm serve BAAI/bge-m3
  - --dtype auto                        # bf16 on Blackwell, fp16 on Ampere
  - --max-model-len 4096                # 4096 is plenty for abstracts (avg ~200 tokens)
  - --gpu-memory-utilization 0.12       # production: leaves room for inference
  - --max-num-seqs 8                    # concurrent sequences; increase with GPU util
  - --max-num-batched-tokens 4096       # must be >= max-model-len
  - --enable-prefix-caching             # caches common query prefixes
  - --served-model-name bge-m3-embedder
  - --api-key simple-api-key
```

**For indexing mode**, change:
```yaml
  - --gpu-memory-utilization 0.50       # ~3x throughput vs 0.12
  - --max-num-seqs 64                   # more concurrent sequences
  - --max-num-batched-tokens 32768      # larger batches
```

### Dense throughput by batch size and GPU utilisation

Measured on DGX Spark (Grace Blackwell, unified memory):

| GPU util | max-num-seqs | Throughput | VRAM used |
|---|---|---|---|
| 0.12 (production) | 8 | ~350 docs/s | ~3 GB |
| 0.25 | 16 | ~550 docs/s | ~6 GB |
| 0.50 (indexing) | 64 | ~1,100 docs/s | ~12 GB |
| 0.80 (max) | 128 | ~1,400 docs/s | ~20 GB |

### Dense embedding API call (how the ingest script calls it)

```python
import httpx, numpy as np

response = httpx.post(
    "http://localhost:8025/v1/embeddings",
    headers={"Authorization": "Bearer simple-api-key"},
    json={
        "model": "bge-m3-embedder",
        "input": texts,           # list of strings, up to max-num-seqs at a time
    },
    timeout=120.0,
)
vecs = np.array([item["embedding"] for item in response.json()["data"]])
# vecs.shape = (len(texts), 1024), already L2-normalised
```

**Optimal client batch size:** Send batches of 64–256 strings. Smaller batches
leave GPU capacity unused; larger batches increase per-request latency. During
indexing, 256 is a good default.

### Why input text is `"title. abstract"` not just the abstract

BGE-M3's retrieval performance improves when the indexed text matches what would
be queried. Prepending the title gives the embedder explicit signal about the
paper's topic. Without it, some high-quality abstracts that bury the key concept
in the third sentence score worse on semantic search.

---

## Sparse embedder (port 8035) — detailed tuning

### Why a custom FastAPI service (not vLLM)?

vLLM does not support SPLADE-style lexical weight output. BGE-M3 has three output
heads — dense, sparse (lexical weights), and ColBERT multi-vector — and only the
dense head is compatible with vLLM's embedding endpoint. The sparse head requires
calling `FlagEmbedding.BGEM3FlagModel.encode()` with `return_sparse=True`, which
returns a dict `{token_id: weight}` that maps to Qdrant's `SparseVector` format.

### Key tuning parameters (`sparse_embedder/sparse_embed.py`)

```python
model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=True,      # halves VRAM, negligible quality loss for sparse
    device="cuda",
)

output = model.encode(
    batch,
    return_dense=False,
    return_sparse=True,
    return_colbert_vecs=False,
    batch_size=64,      # optimal for sparse; dense needs larger batches
)
```

**`return_dense=False`** is critical — computing the dense head doubles inference
time for no benefit since you have a separate dense service.

### Sparse throughput by batch size

| Batch size | Throughput | Notes |
|---|---|---|
| 16 | ~600 docs/s | Safe for low VRAM (8 GB) |
| 32 | ~900 docs/s | Good production default |
| 64 | ~1,400 docs/s | Optimal for indexing on Spark |
| 128 | ~1,600 docs/s | Diminishing returns, more latency |

Sparse encoding is faster than dense per token because the lexical weights are
computed from a shallower part of the network than the full 1024-dim dense projection.

### Sparse vector characteristics

Each document produces a sparse vector with typically 30–200 non-zero entries
(tokens weighted by their importance to the document). By comparison, BM25 uses
exact term frequency counts; BGE-M3's sparse head produces vocabulary-expanded
weights via the SPLADE mechanism, which gives better recall.

```
"attention mechanism transformer":
  {"attention": 0.82, "mechanism": 0.61, "transformer": 0.79,
   "self": 0.43, "query": 0.38, "key": 0.35, ...}
```

Qdrant stores these efficiently in an inverted index on disk (for the master arXiv
collection) or in memory (for subject-area collections).

---

## Reranker (port 8020) — detailed tuning

### Why low GPU utilisation for the reranker?

Unlike the embedders (which process batches independently), the reranker scores
(query, document) pairs jointly in a single forward pass. The input is always
`[CLS] query [SEP] document [SEP]`, so the model length is `len(query) + len(doc)`.
With 50 candidates of average 200-token abstracts, the reranker processes:
- 50 × (20 + 200) = 11,000 tokens per query

At `gpu-memory-utilization 0.15` with `max-model-len 2048` this fits comfortably
and adds ~1–2 seconds per query. Higher GPU utilisation does not meaningfully
improve throughput because the reranker is not the bottleneck.

### Required vLLM flags for the reranker

```yaml
- --no-enable-prefix-caching          # REQUIRED — prefix caching breaks cross-encoder scoring
- --disable-hybrid-kv-cache-manager   # REQUIRED — without this vLLM throws a runtime error
- --max-model-len 2048
- --gpu-memory-utilization 0.15
```

These flags are not optional — the reranker container fails to start or produces
incorrect scores without them.

---

## Running dense and sparse indexing in parallel

The DGX Spark's unified memory architecture allows both pipelines to run
simultaneously on different GPU slices:

```bash
# Terminal 1: dense indexing (50% GPU, port 8026)
docker compose -f docker/bge_m3_dense_embedder_indexing.yml up -d

# Terminal 2: sparse indexing (dedicated slice, port 8035)
docker compose --profile embedding up -d bge-m3-sparse-embedder

# Terminal 3: run both ingest scripts in parallel
python ingest/03_ingest_dense.py  --input data/arxiv.jsonl --batch-size 256 &
python ingest/04_ingest_sparse.py --input data/arxiv.jsonl --batch-size 64 &
wait
echo "Both complete"
```

Both scripts are idempotent — re-running upserts the same point IDs, so they
can be restarted after interruption without duplication.

**Total time for 2.96M papers (both in parallel):**
- Dense (50% GPU, batch 256): ~18–22 hours
- Sparse (dedicated, batch 64): ~6–8 hours
- Running in parallel: dominated by the dense pipeline → **~18–22 hours total**

---

## Indexing time reference — full Arxiv corpus

| Stage | Records | Throughput | Wall time |
|---|---|---|---|
| Download metadata | 2.96M papers | ~1,300/s (network) | ~40 min |
| Create 14 collections | — | instant | ~30 sec |
| Dense embed + upsert (indexing mode) | 2.96M | ~1,100 docs/s | ~45 min embed + upsert overhead → **~18–22 hrs total** |
| Sparse embed + upsert | 2.96M | ~1,400–1,600 docs/s | ~**6–8 hrs** |
| Single PDF (30 pages, production mode) | ~60 chunks | — | ~2–4 min |
| Single PDF (200 pages) | ~400 chunks | — | ~15–25 min |
| Figure captioning (vision model) | per page with figures | ~2 pages/min | ~30–60s/page |

**Note:** Dense upsert overhead dominates because each point goes into both the
subject-area collection AND the master `arXiv` collection (2× network writes).
The actual embedding computation is fast — the bottleneck is Qdrant write throughput
over Docker networking.

To speed up upserts:
```python
# Use wait=False — Qdrant acknowledges receipt and indexes asynchronously
client.upsert(collection_name=collection, points=batch, wait=False)
```

---

## VRAM allocation table

Full production stack on DGX Spark (128 GB unified memory):

| Service | Container | GPU util | VRAM |
|---|---|---|---|
| vLLM inference (Nemotron/Qwen3) | `vllm_node` | auto | ~80–100 GB |
| BGE-M3 dense embedder (production) | `bge-m3-dense-embedder` | 0.12 | ~3 GB |
| BGE-M3 sparse embedder | `bge-m3-sparse-embedder` | fp16 model | ~3 GB |
| BGE-M3 reranker | `bge-m3-reranker` | 0.15 | ~2 GB |
| Qdrant (GPU HNSW indexing) | `qdrant` | varies | ~1–2 GB |
| **Total** | | | **~89–110 GB** |
| **Headroom** | | | **18–39 GB** |

During indexing mode (with inference model stopped):

| Service | GPU util | VRAM |
|---|---|---|
| BGE-M3 dense embedder (indexing) | 0.50 | ~12 GB |
| BGE-M3 sparse embedder | fp16 | ~3 GB |
| Qdrant | varies | ~2 GB |
| **Total** | | **~17 GB** |
| **Headroom for inference** | | **~111 GB** |

The embedding stack in indexing mode uses only ~17 GB — the inference model can
run simultaneously if needed.

---

## Common performance issues and fixes

**Dense embedder slow on first request after startup**
The first batch after container start triggers JIT compilation of CUDA kernels.
Warm up with a dummy request:
```bash
curl -s http://localhost:8025/v1/embeddings \
  -H "Authorization: Bearer simple-api-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"bge-m3-embedder","input":["warmup"]}' > /dev/null
```

**Sparse embedder OOM crash**
Reduce `max_batch_size` in the `/encode` request body to 16 or 32. The sparse
model's tokenisation buffers are the memory bottleneck, not the model weights.

**Upsert throughput low (< 100 points/s)**
- Use `wait=False` in upsert calls
- Check if Qdrant's optimizer is saturated: `curl http://localhost:6333/collections/arxiv-cs-ml-ai | jq .result.optimizer_status`
- If optimizer shows "optimizing", it is building HNSW index — this is expected after large ingestion batches and completes automatically

**Dense embedder returns 413 errors**
Request payload too large. Reduce batch size to 64 or set `--max-num-batched-tokens` higher on the vLLM server.

**Throughput drops during indexing**
The Qdrant WAL (write-ahead log) flushes every 5 seconds. During heavy write load,
HNSW re-indexing competes with incoming writes. This is normal. Throughput recovers
once the optimizer catches up.
