#!/usr/bin/env python3
"""
eval/answer_eval.py
===================
Answer quality evaluation for Spark-Scholar using RAGAS.

Measures three LLM-graded metrics over a QA dataset:

  - faithfulness        : are all claims in the answer supported by the
                          retrieved context?  (does not require ground_truth)
  - answer_relevancy    : is the answer actually addressing the question?
                          (does not require ground_truth)
  - context_precision   : are the retrieved chunks ranked by usefulness?
                          (requires ground_truth)

This file complements retrieval_eval.py, which measures whether the *right
documents* are retrieved (Recall@k, MRR, nDCG@k).  answer_eval.py measures
whether the *answer generated from those documents* is correct and grounded.

Usage
-----
    # Basic run — uses eval/qa_dataset.jsonl by default
    python eval/answer_eval.py

    # First 5 queries for a quick smoke-test
    python eval/answer_eval.py --limit 5 --verbose

    # Custom dataset and output file
    python eval/answer_eval.py --dataset eval/my_qa.jsonl --output eval/answer_results.json

    # Change how many context chunks are retrieved per query
    python eval/answer_eval.py --k 5

Requirements
------------
    pip install ragas langchain-openai

Services needed:
  - Qdrant           (port 6333)
  - BGE dense        (port 8025)
  - BGE sparse       (port 8035)
  - vLLM via LiteLLM (port 4000)  — used as judge LLM for RAGAS metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../env/.env"), override=False)

# Add repo root to path so pipeline imports work (same pattern as retrieval_eval.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# ── Default paths ──────────────────────────────────────────────────────────────
EVAL_DIR   = Path(__file__).parent
QA_DATASET = EVAL_DIR / "qa_dataset.jsonl"

# ── Service endpoints (override via env) ──────────────────────────────────────
LITELLM_URL     = os.environ.get("LITELLM_URL",         "http://localhost:4000")
LITELLM_API_KEY = os.environ.get("LITELLM_MASTER_KEY",  "simple-api-key")
VLLM_MODEL      = os.environ.get("VLLM_MODEL_NAME",     "local-model")


# ─────────────────────────────────────────────────────────────────────────────
# LLM / embedder wrappers (RAGAS-compatible)
# ─────────────────────────────────────────────────────────────────────────────

def _build_ragas_llm():
    """
    Build a RAGAS-compatible LLM wrapper pointing at the local LiteLLM proxy.
    The proxy exposes an OpenAI-compatible /v1 endpoint backed by vLLM.
    """
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI

    lc_llm = ChatOpenAI(
        model=VLLM_MODEL,
        openai_api_base=f"{LITELLM_URL.rstrip('/')}/v1",
        openai_api_key=LITELLM_API_KEY,
        temperature=0.0,
    )
    return LangchainLLMWrapper(lc_llm)


def _build_ragas_embedder():
    """
    Build a RAGAS-compatible embeddings wrapper pointing at the LiteLLM proxy
    (which exposes an embeddings endpoint compatible with OpenAI's API).
    """
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import OpenAIEmbeddings

    lc_emb = OpenAIEmbeddings(
        model="text-embedding-ada-002",   # model name forwarded by LiteLLM
        openai_api_base=f"{LITELLM_URL.rstrip('/')}/v1",
        openai_api_key=LITELLM_API_KEY,
    )
    return LangchainEmbeddingsWrapper(lc_emb)


# ─────────────────────────────────────────────────────────────────────────────
# Answer generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    contexts: list[str],
    llm_client,
) -> str:
    """
    Generate a RAG-style answer by calling the LiteLLM proxy via httpx.

    Parameters
    ----------
    query    : The user question.
    contexts : List of context strings retrieved for this query.
    llm_client : An httpx.Client configured with the LiteLLM base URL and key.

    Returns
    -------
    str  The generated answer, or an empty string on failure.
    """
    joined_contexts = "\n\n".join(
        f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts)
    )
    prompt = (
        "Answer using ONLY the provided context. "
        "If the context doesn't contain enough information, say so.\n\n"
        f"Context:\n{joined_contexts}\n\n"
        f"Question: {query}"
    )

    try:
        resp = llm_client.post(
            f"{LITELLM_URL.rstrip('/')}/v1/chat/completions",
            json={
                "model": VLLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.0,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning("Answer generation failed: %s", exc)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: Path, limit: Optional[int] = None) -> list[dict]:
    """Load a JSONL QA dataset, optionally capping at limit entries."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if limit:
        items = items[:limit]
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_answer_eval(
    dataset: list[dict],
    limit: Optional[int] = None,
    verbose: bool = False,
    top_k: int = 10,
) -> dict:
    """
    Run answer-quality evaluation over dataset using RAGAS.

    For each item:
      1. Retrieve top-k contexts via hybrid_search.
      2. Generate an answer from those contexts via the LiteLLM proxy.
      3. Collect (question, answer, contexts, ground_truth) for RAGAS.

    RAGAS is then called with faithfulness, answer_relevancy, and
    context_precision.  Items without a ground_truth field get an empty
    string — faithfulness and answer_relevancy do not require it; only
    context_precision may be affected.

    Returns
    -------
    dict with keys:
      - per_query   : list of per-query dicts (question, answer, contexts, …)
      - metrics     : dict of metric_name → float average
      - n_queries   : int
      - timestamp   : ISO-8601 UTC string
    """
    import httpx
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision
    from datasets import Dataset as HFDataset
    from pipeline.hybrid_search import hybrid_search

    if limit:
        dataset = dataset[:limit]

    ragas_llm = _build_ragas_llm()
    ragas_emb = _build_ragas_embedder()

    # Wire the wrappers into each metric
    for metric in (faithfulness, answer_relevancy, context_precision):
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_emb

    llm_client = httpx.Client(
        headers={"Authorization": f"Bearer {LITELLM_API_KEY}"},
        timeout=60.0,
    )

    rows: list[dict] = []
    per_query: list[dict] = []

    for i, item in enumerate(dataset):
        query      = item["query"]
        collection = item.get("collection", "arxiv-cs-ml-ai")
        ground_truth = item.get("ground_truth", "")

        if verbose:
            print(f"  [{i + 1:>2}/{len(dataset)}] {query[:70]}")

        t0 = time.time()
        try:
            results = hybrid_search(
                query,
                collections=[collection],
                top_k=top_k,
                rerank_n=top_k * 5,
            )
            contexts = [
                r.get("chunk_text") or r.get("abstract") or ""
                for r in results
                if r.get("chunk_text") or r.get("abstract")
            ]
        except Exception as exc:
            logger.warning("Retrieval failed for query %d: %s", i + 1, exc)
            contexts = []

        answer = generate_answer(query, contexts, llm_client)
        latency_ms = (time.time() - t0) * 1000

        row = {
            "question":     query,
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": ground_truth,
        }
        rows.append(row)

        per_query.append({
            "query":        query,
            "answer":       answer,
            "n_contexts":   len(contexts),
            "ground_truth": ground_truth,
            "latency_ms":   latency_ms,
        })

    llm_client.close()

    if not rows:
        logger.error("No rows to evaluate.")
        return {"per_query": [], "metrics": {}, "n_queries": 0, "timestamp": datetime.now(timezone.utc).isoformat()}

    hf_dataset = HFDataset.from_list(rows)

    try:
        ragas_result = evaluate(
            hf_dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
        metrics = {
            "faithfulness":      float(ragas_result["faithfulness"]),
            "answer_relevancy":  float(ragas_result["answer_relevancy"]),
            "context_precision": float(ragas_result["context_precision"]),
        }

        # Attach per-query scores if RAGAS returns a DataFrame
        if hasattr(ragas_result, "to_pandas"):
            df = ragas_result.to_pandas()
            for j, row_scores in df.iterrows():
                if j < len(per_query):
                    per_query[j]["faithfulness"]      = row_scores.get("faithfulness")
                    per_query[j]["answer_relevancy"]  = row_scores.get("answer_relevancy")
                    per_query[j]["context_precision"] = row_scores.get("context_precision")

    except Exception as exc:
        logger.error("RAGAS evaluation failed: %s", exc)
        metrics = {}

    return {
        "per_query":  per_query,
        "metrics":    metrics,
        "n_queries":  len(rows),
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_report(results: dict) -> None:
    """
    Print a clean summary table of answer-quality metrics plus per-query rows.
    """
    metrics   = results.get("metrics", {})
    per_query = results.get("per_query", [])
    n         = results.get("n_queries", 0)

    print()
    print("━" * 72)
    print("  Answer Quality Evaluation Results (RAGAS)")
    print("━" * 72)

    # Summary row
    print()
    print(f"  Queries evaluated : {n}")
    print(f"  {'Metric':<25}  {'Score':>7}")
    print("  " + "─" * 35)
    for metric_name, score in sorted(metrics.items()):
        print(f"  {metric_name:<25}  {score:>7.4f}")

    if per_query:
        print()
        print("  Per-query breakdown:")
        print(f"  {'#':>3}  {'Faithfulness':>13}  {'Relevancy':>10}  {'Precision':>10}  {'ms':>6}  Query")
        print("  " + "─" * 70)
        for i, row in enumerate(per_query, 1):
            faith = row.get("faithfulness")
            relev = row.get("answer_relevancy")
            prec  = row.get("context_precision")
            faith_s = f"{faith:.3f}" if faith is not None else "  n/a"
            relev_s = f"{relev:.3f}" if relev is not None else "  n/a"
            prec_s  = f"{prec:.3f}"  if prec  is not None else "  n/a"
            q = row["query"][:50]
            print(f"  {i:>3}  {faith_s:>13}  {relev_s:>10}  {prec_s:>10}  "
                  f"{row.get('latency_ms', 0):>6.0f}  {q}")

    print()
    print("━" * 72)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate answer quality (faithfulness, relevancy, precision) via RAGAS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval/answer_eval.py                              # all queries, k=10
  python eval/answer_eval.py --limit 5 --verbose         # quick smoke-test
  python eval/answer_eval.py --k 5                       # fewer context chunks
  python eval/answer_eval.py --output eval/answer_results.json
        """,
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
        "--verbose",
        action="store_true",
        help="Print per-query progress and full report",
    )
    p.add_argument(
        "--k",
        type=int,
        default=10,
        dest="top_k",
        help="Number of context chunks to retrieve per query (default: 10)",
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

    dataset = load_dataset(Path(args.dataset), limit=args.limit)
    print(f"\nDataset: {args.dataset}  ({len(dataset)} queries)")

    results = run_answer_eval(
        dataset,
        limit=args.limit,
        verbose=args.verbose,
        top_k=args.top_k,
    )

    print_report(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
