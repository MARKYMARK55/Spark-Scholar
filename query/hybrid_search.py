#!/usr/bin/env python3
"""
query/hybrid_search.py
----------------------
Command-line interface for full hybrid search over Arxiv Qdrant collections.

Runs the complete pipeline:
  1. Encode query → BGE-M3 dense (1024-dim) + sparse (SPLADE token weights)
  2. Qdrant native Prefetch + FusionQuery(RRF) per collection
  3. Cross-collection RRF merge (if multiple collections searched)
  4. BGE-M3 cross-encoder reranking of top-N candidates
  5. Returns top-K results with scores and source metadata

This is the CLI wrapper around pipeline/hybrid_search.py and pipeline/reranker.py.
For programmatic use, import those modules directly.

Usage:
    python query/hybrid_search.py --query "attention mechanism self-attention"
    python query/hybrid_search.py --query "Hawkes process excitation" --collection arxiv-stat-eess
    python query/hybrid_search.py --query "RLHF alignment" --year-min 2022 --top-k 20
    python query/hybrid_search.py --query "Vaswani 2017" --no-rerank --json
    python query/hybrid_search.py --query "diffusion models" --author "Ho"
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv("env/.env")

from pipeline.hybrid_search import hybrid_search
from pipeline.reranker import rerank
from pipeline.router import route_query


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid dense+sparse search over Arxiv Qdrant collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--query",      required=True,          help="Search query")
    parser.add_argument("--collection", default=None,           help="Force specific collection (skip auto-routing)")
    parser.add_argument("--top-k",      type=int, default=10,   help="Final results to return (default: 10)")
    parser.add_argument("--rerank-n",   type=int, default=50,   help="Candidates to rerank (default: 50)")
    parser.add_argument("--year-min",   type=int, default=None, help="Filter: minimum publication year")
    parser.add_argument("--year-max",   type=int, default=None, help="Filter: maximum publication year")
    parser.add_argument("--author",     default=None,           help="Filter: author name substring")
    parser.add_argument("--no-rerank",  action="store_true",    help="Skip cross-encoder reranking")
    parser.add_argument("--json",       action="store_true",    help="Output as JSON")
    parser.add_argument("--verbose",    action="store_true",    help="Show timing and routing info")
    args = parser.parse_args()

    t_start = time.time()

    # 1. Route query to collections
    if args.collection:
        collections = [args.collection]
    else:
        collections = route_query(args.query)

    if args.verbose:
        print(f"Routing: {args.query!r} → {collections}", file=sys.stderr)

    # 2. Hybrid search
    t_retrieve = time.time()
    candidates = hybrid_search(
        query=args.query,
        collections=collections,
        top_k=args.rerank_n,
        year_min=args.year_min,
        year_max=args.year_max,
        author=args.author,
    )
    t_retrieved = time.time()

    if args.verbose:
        print(f"Retrieved {len(candidates)} candidates in {t_retrieved - t_retrieve:.2f}s", file=sys.stderr)

    # 3. Rerank
    if not args.no_rerank and candidates:
        t_rerank = time.time()
        results = rerank(args.query, candidates, top_n=args.top_k)
        t_reranked = time.time()
        if args.verbose:
            print(f"Reranked to {len(results)} in {t_reranked - t_rerank:.2f}s", file=sys.stderr)
    else:
        results = candidates[: args.top_k]

    t_end = time.time()

    if args.verbose:
        print(f"Total: {t_end - t_start:.2f}s", file=sys.stderr)

    # 4. Output
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    print(f"\nTop {len(results)} results for: {args.query!r}")
    print(f"Collections: {', '.join(collections)}")
    if not args.no_rerank:
        print(f"Pipeline: hybrid retrieve ({args.rerank_n} candidates) → cross-encoder rerank → top {args.top_k}")
    print("─" * 80)

    for i, doc in enumerate(results, 1):
        score = doc.get("rerank_score") or doc.get("rrf_score") or doc.get("score", 0.0)
        arxiv_id = doc.get("arxiv_id", "unknown")
        year = doc.get("year", "")
        collection = doc.get("collection", "")
        title = doc.get("title", "(no title)")
        authors = doc.get("authors", "")
        abstract = doc.get("abstract", "")

        print(f"\n{i:2}. [{score:.4f}]  {arxiv_id}  ({year})  [{collection}]")
        print(f"    {title}")
        if authors:
            author_str = authors[:90] + ("..." if len(authors) > 90 else "")
            print(f"    {author_str}")
        if abstract:
            abstract_str = abstract[:280].replace("\n", " ") + "..."
            print(f"    {abstract_str}")

    print()


if __name__ == "__main__":
    main()
