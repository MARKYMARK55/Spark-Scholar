#!/usr/bin/env python3
"""
eval/retrieval_eval.py
======================
Retrieval evaluation for Spark-Scholar.

Measures retrieval quality using a small QA dataset (eval/qa_dataset.jsonl)
where each query has one or more known-relevant arXiv IDs. Reports:

  - Recall@k  (k = 1, 5, 10): fraction of queries where a relevant paper
                               appears in the top-k results
  - MRR       (Mean Reciprocal Rank): average of 1/rank of first hit
  - nDCG@k    (normalised Discounted Cumulative Gain): ranking quality metric

Supports three retrieval modes to compare:
  - dense     : BGE-M3 dense ANN only
  - sparse    : BGE-M3 sparse inverted index only
  - hybrid    : dense + sparse + Qdrant RRF fusion  (default)
  - hybrid+rr : hybrid with cross-encoder reranking

Usage
-----
    # Full eval with default settings (hybrid, k=10, all 20 queries)
    python eval/retrieval_eval.py

    # Compare all four modes
    python eval/retrieval_eval.py --mode all

    # Custom k and quiet output
    python eval/retrieval_eval.py --mode hybrid --k 5 --k 10 --quiet

    # Run only first N queries (quick smoke-test)
    python eval/retrieval_eval.py --limit 5

    # Save results to JSON
    python eval/retrieval_eval.py --output eval/results.json

Requirements
------------
Services must be running:
  - Qdrant         (port 6333)
  - BGE-M3 dense   (port 8025)
  - BGE-M3 sparse  (port 8035)
  - BGE reranker   (port 8020)  — only needed for hybrid+rr mode
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../env/.env"), override=False)

# Add repo root to path so pipeline imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# ── Default paths ──────────────────────────────────────────────────────────────
EVAL_DIR = Path(__file__).parent
QA_DATASET = EVAL_DIR / "qa_dataset.jsonl"

# ── Service endpoints (override via env) ──────────────────────────────────────
QDRANT_URL   = os.environ.get("QDRANT_URL",        "http://localhost:6333")
DENSE_URL    = os.environ.get("BGE_M3_DENSE_URL",  "http://localhost:8025")
SPARSE_URL   = os.environ.get("BGE_M3_SPARSE_URL", "http://localhost:8035")
RERANKER_URL = os.environ.get("BGE_RERANKER_URL",  "http://localhost:8020")


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Recall@k: 1.0 if any relevant ID appears in the first k results, else 0.0.

    Binary relevance — we count a query as "hit" if at least one relevant paper
    is retrieved. For queries with multiple relevant IDs, any match counts.
    """
    top_k = set(retrieved_ids[:k])
    return 1.0 if any(rid in top_k for rid in relevant_ids) else 0.0


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """
    Reciprocal Rank: 1/rank of the first relevant result (1-indexed).
    Returns 0.0 if no relevant result found in the retrieved list.
    """
    relevant_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    nDCG@k with binary relevance (rel=1 if in relevant_ids, else 0).

    DCG@k  = sum over i in [1..k] of rel_i / log2(i + 1)
    IDCG@k = sum over i in [1..min(|relevant|, k)] of 1 / log2(i + 1)
    nDCG@k = DCG@k / IDCG@k
    """
    relevant_set = set(relevant_ids)
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k], start=1):
        if rid in relevant_set:
            dcg += 1.0 / math.log2(i + 1)

    # Ideal DCG: assume all relevant docs are at the top
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def compute_metrics(
    results: list[dict],
    k_values: list[int],
) -> dict:
    """
    Aggregate metrics across all query results.

    Parameters
    ----------
    results : list[dict]
        One dict per query, each with:
          - retrieved_ids  : list of arXiv IDs returned by retrieval (in rank order)
          - relevant_ids   : list of ground-truth relevant IDs
          - latency_ms     : retrieval time in milliseconds
    k_values : list[int]
        Values of k to compute Recall@k and nDCG@k for.

    Returns
    -------
    dict with keys: recall@k (per k), mrr, ndcg@k (per k), avg_latency_ms, n_queries
    """
    n = len(results)
    if n == 0:
        return {}

    metrics: dict = {"n_queries": n}

    for k in k_values:
        recall_scores = [recall_at_k(r["retrieved_ids"], r["relevant_ids"], k) for r in results]
        ndcg_scores   = [ndcg_at_k(r["retrieved_ids"], r["relevant_ids"], k) for r in results]
        metrics[f"recall@{k}"]  = sum(recall_scores) / n
        metrics[f"ndcg@{k}"]    = sum(ndcg_scores)   / n

    mrr_scores = [reciprocal_rank(r["retrieved_ids"], r["relevant_ids"]) for r in results]
    metrics["mrr"] = sum(mrr_scores) / n

    latencies = [r.get("latency_ms", 0) for r in results]
    metrics["avg_latency_ms"] = sum(latencies) / n if latencies else 0.0
    metrics["p95_latency_ms"] = sorted(latencies)[int(0.95 * len(latencies))] if latencies else 0.0

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval wrappers
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_hybrid(query: str, collection: str, top_k: int) -> list[str]:
    """Hybrid dense+sparse retrieval via Qdrant RRF fusion."""
    from pipeline.hybrid_search import hybrid_search
    results = hybrid_search(query, collections=[collection], top_k=top_k, rerank_n=top_k * 5)
    return [r.get("arxiv_id", "") for r in results]


def retrieve_hybrid_rerank(query: str, collection: str, top_k: int) -> list[str]:
    """Hybrid retrieval followed by BGE-M3 cross-encoder reranking."""
    from pipeline.hybrid_search import hybrid_search
    from pipeline.reranker import rerank
    candidates = hybrid_search(query, collections=[collection], top_k=top_k * 5, rerank_n=top_k * 5)
    reranked = rerank(query, candidates, top_n=top_k)
    return [r.get("arxiv_id", "") for r in reranked]


def retrieve_dense(query: str, collection: str, top_k: int) -> list[str]:
    """Dense-only retrieval (HNSW ANN)."""
    import asyncio
    import numpy as np
    from pipeline.embeddings import async_encode_dense
    from qdrant_client import QdrantClient
    from qdrant_client.models import NamedVector

    async def _enc():
        vecs = await async_encode_dense([query])
        return vecs[0].tolist()

    dense_vec = asyncio.run(_enc())
    client = QdrantClient(url=QDRANT_URL, timeout=60)
    results = client.query_points(
        collection_name=collection,
        query=dense_vec,
        using="dense_embedding",
        limit=top_k,
        with_payload=True,
    )
    points = results.points if hasattr(results, "points") else results
    return [p.payload.get("arxiv_id", "") for p in points]


def retrieve_sparse(query: str, collection: str, top_k: int) -> list[str]:
    """Sparse-only retrieval (SPLADE inverted index)."""
    import asyncio
    from pipeline.embeddings import async_encode_sparse
    from qdrant_client import QdrantClient
    from qdrant_client.models import NamedSparseVector, SparseVector

    async def _enc():
        vecs = await async_encode_sparse([query])
        return vecs[0]

    sparse = asyncio.run(_enc())
    qdrant_sparse = SparseVector(indices=sparse.indices, values=sparse.values)
    client = QdrantClient(url=QDRANT_URL, timeout=60)
    results = client.query_points(
        collection_name=collection,
        query=NamedSparseVector(name="sparse_text", vector=qdrant_sparse),
        using="sparse_text",
        limit=top_k,
        with_payload=True,
    )
    points = results.points if hasattr(results, "points") else results
    return [p.payload.get("arxiv_id", "") for p in points]


RETRIEVAL_FNS = {
    "dense":      retrieve_dense,
    "sparse":     retrieve_sparse,
    "hybrid":     retrieve_hybrid,
    "hybrid+rr":  retrieve_hybrid_rerank,
}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: Path, limit: Optional[int] = None) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if limit:
        items = items[:limit]
    return items


def run_eval(
    dataset: list[dict],
    mode: str,
    k_values: list[int],
    verbose: bool = True,
) -> dict:
    """
    Run evaluation for a single retrieval mode.

    Returns a dict with per-query results and aggregate metrics.
    """
    retrieve_fn = RETRIEVAL_FNS[mode]
    max_k = max(k_values)
    per_query = []

    for i, item in enumerate(dataset):
        query       = item["query"]
        relevant    = item["relevant_ids"]
        collection  = item.get("collection", "arxiv-cs-ml-ai")

        t0 = time.time()
        try:
            retrieved = retrieve_fn(query, collection, top_k=max_k)
        except Exception as exc:
            logger.warning("Query %d failed: %s", i + 1, exc)
            retrieved = []
        latency_ms = (time.time() - t0) * 1000

        result = {
            "query":         query,
            "relevant_ids":  relevant,
            "retrieved_ids": retrieved,
            "latency_ms":    latency_ms,
            "hit":           any(rid in retrieved[:max_k] for rid in relevant),
        }
        per_query.append(result)

        if verbose:
            hit_rank = None
            for rank, rid in enumerate(retrieved, 1):
                if rid in set(relevant):
                    hit_rank = rank
                    break
            status = f"rank {hit_rank}" if hit_rank else "MISS"
            print(f"  [{i+1:>2}/{len(dataset)}] {status:>8}  {query[:65]}")

    metrics = compute_metrics(per_query, k_values)
    return {"mode": mode, "metrics": metrics, "per_query": per_query}


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_report(eval_results: list[dict], k_values: list[int]) -> None:
    """Print a formatted comparison table."""
    modes = [r["mode"] for r in eval_results]
    col_w = max(len(m) for m in modes) + 2

    print()
    print("━" * 72)
    print("  Retrieval Evaluation Results")
    print("━" * 72)

    # Header
    header = f"  {'Mode':<{col_w}}"
    for k in k_values:
        header += f"  Recall@{k:<3}"
        header += f"  nDCG@{k:<4}"
    header += f"  {'MRR':>6}  {'Avg ms':>7}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    # Rows
    for r in eval_results:
        m = r["metrics"]
        row = f"  {r['mode']:<{col_w}}"
        for k in k_values:
            row += f"  {m.get(f'recall@{k}', 0):.3f}    "
            row += f"  {m.get(f'ndcg@{k}', 0):.3f}  "
        row += f"  {m.get('mrr', 0):.3f}  {m.get('avg_latency_ms', 0):>7.0f}"
        print(row)

    print("━" * 72)
    print(f"  Queries: {eval_results[0]['metrics']['n_queries']}  |  "
          f"k values: {k_values}")
    print("━" * 72)
    print()


def save_results(eval_results: list[dict], output_path: str) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": eval_results,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate retrieval quality (Recall@k, MRR, nDCG@k) over a small QA set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval/retrieval_eval.py                        # hybrid, k=1,5,10, all 20 queries
  python eval/retrieval_eval.py --mode all             # compare dense/sparse/hybrid/hybrid+rr
  python eval/retrieval_eval.py --mode hybrid --k 5   # single k value
  python eval/retrieval_eval.py --limit 5 --quiet     # fast smoke-test, no per-query output
  python eval/retrieval_eval.py --output eval/results.json
        """,
    )
    p.add_argument(
        "--mode",
        choices=["dense", "sparse", "hybrid", "hybrid+rr", "all"],
        default="hybrid",
        help="Retrieval mode to evaluate (default: hybrid)",
    )
    p.add_argument(
        "--k",
        type=int,
        action="append",
        dest="k_values",
        metavar="K",
        help="Value of k for Recall@k and nDCG@k (can repeat: --k 1 --k 5 --k 10)",
    )
    p.add_argument(
        "--dataset",
        default=str(QA_DATASET),
        help=f"Path to JSONL QA dataset (default: {QA_DATASET})",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N queries (useful for quick tests)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Save full results to this JSON file",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-query output — only show summary table",
    )
    p.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    k_values = sorted(set(args.k_values or [1, 5, 10]))
    modes = list(RETRIEVAL_FNS.keys()) if args.mode == "all" else [args.mode]

    dataset = load_dataset(Path(args.dataset), limit=args.limit)
    print(f"\nDataset: {args.dataset}  ({len(dataset)} queries)")

    all_results = []
    for mode in modes:
        print(f"\n{'─' * 60}")
        print(f"  Mode: {mode}")
        print(f"{'─' * 60}")
        eval_result = run_eval(dataset, mode, k_values, verbose=not args.quiet)
        all_results.append(eval_result)

    print_report(all_results, k_values)

    if args.output:
        save_results(all_results, args.output)


if __name__ == "__main__":
    main()
