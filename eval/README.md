# eval/ — Retrieval Evaluation

Measures retrieval quality over a small hand-curated QA dataset of 20 landmark ML/AI papers.
Each entry has a natural language query and the ground-truth arXiv IDs that should appear in the results.

## Metrics

| Metric | What it measures |
|---|---|
| **Recall@k** | Fraction of queries where ≥1 relevant paper appears in the top-k results |
| **MRR** | Mean Reciprocal Rank — rewards finding the relevant paper at a higher rank |
| **nDCG@k** | Normalised Discounted Cumulative Gain — penalises relevant papers buried deep in the list |
| **Avg latency** | Mean query time in milliseconds (wall-clock, includes embedding + Qdrant round-trip) |

## Quick start

```bash
# Prerequisites: Qdrant, BGE-M3 dense, BGE-M3 sparse, (optionally) BGE reranker must be running.
# The arXiv papers in the dataset must have been ingested first (ingest steps 01–04).

# Default: hybrid retrieval, k=1,5,10, all 20 queries
python eval/retrieval_eval.py

# Compare all four modes side-by-side
python eval/retrieval_eval.py --mode all

# Fast smoke-test (first 5 queries only, no per-query output)
python eval/retrieval_eval.py --limit 5 --quiet

# Save full results for tracking over time
python eval/retrieval_eval.py --mode all --output eval/results_$(date +%Y%m%d).json
```

## Retrieval modes

| Mode | Description |
|---|---|
| `dense` | BGE-M3 dense ANN (HNSW) only |
| `sparse` | BGE-M3 sparse inverted index (SPLADE) only |
| `hybrid` | Dense + sparse fused with Qdrant's native RRF — **recommended baseline** |
| `hybrid+rr` | Hybrid + BGE-M3 cross-encoder reranking — highest quality, ~2× slower |

## Expected output

```
Dataset: eval/qa_dataset.jsonl  (20 queries)

────────────────────────────────────────────────────────────
  Mode: hybrid
────────────────────────────────────────────────────────────
  [ 1]   rank 1  attention mechanism transformer self-attention multi-head
  [ 2]   rank 2  BERT pre-training deep bidirectional language model
  ...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Retrieval Evaluation Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Mode          Recall@1    nDCG@1    Recall@5    nDCG@5    Recall@10   nDCG@10   MRR     Avg ms
  ──────────────────────────────────────────────────────────────────────
  hybrid        0.750       0.750     0.900       0.842     0.950       0.869     0.821     312
  hybrid+rr     0.800       0.800     0.950       0.896     0.950       0.896     0.872     890
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

> Actual numbers depend on which arXiv papers you've ingested and your embedding model versions.

## Dataset format

`qa_dataset.jsonl` — one JSON object per line:

```json
{
  "query": "attention mechanism transformer self-attention multi-head",
  "relevant_ids": ["1706.03762"],
  "collection": "arxiv-cs-ml-ai",
  "notes": "Vaswani et al. 2017 — Attention Is All You Need"
}
```

| Field | Description |
|---|---|
| `query` | Natural language query (designed to match the paper without using the exact title) |
| `relevant_ids` | arXiv IDs (without version suffix) that count as correct answers |
| `collection` | Qdrant collection to search |
| `notes` | Human-readable label for the paper |

## Adding your own queries

Append lines to `qa_dataset.jsonl`. arXiv IDs are the bare ID without version:
`1706.03762` not `1706.03762v5`.

For domain-specific work, create a second JSONL file and pass it with `--dataset`:

```bash
python eval/retrieval_eval.py --dataset eval/my_domain_qa.jsonl --mode hybrid+rr
```

## Interpreting results

- **Recall@1 > 0.7** — hybrid search is finding the right paper in pole position for most queries
- **Recall@10 > 0.9** — the relevant paper is in the top 10 for nearly all queries
- **MRR > 0.8** — on average the relevant paper appears within the first 1–2 results
- **hybrid+rr gains < 5% over hybrid** — your base retrieval is already strong; reranking adds latency for marginal gain
- **hybrid+rr gains > 10% over hybrid** — reranking is doing real work; consider always enabling it
