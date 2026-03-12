#!/usr/bin/env python3
"""
query/sparse_search.py
----------------------
Sparse (SPLADE) search over Arxiv Qdrant collections using BGE-M3 lexical weights.

Calls the running BGE-M3 sparse embedder service (port 8035) — does NOT load
the model locally, so it runs safely alongside the production embedding stack.

Sparse search excels at keyword-precision queries: exact author names, arXiv IDs,
model names (GPT-4, LLaMA), chemical formulae, and any query where the specific
token matters more than the semantic meaning.

For semantic / concept-level search, use dense_search.py.
For the best of both worlds (recommended), use hybrid_search.py.

Usage
-----
    python query/sparse_search.py --query "attention mechanism transformers self-attention"
    python query/sparse_search.py --query "Hawkes process excitation kernel" \\
                                  --collection arxiv-stat-eess
    python query/sparse_search.py --query "2305.14314"
    python query/sparse_search.py --query "Vaswani attention" --year-min 2017 --year-max 2018

Environment
-----------
    BGE_M3_SPARSE_URL — sparse embedder base URL (default: http://localhost:8035)
    BGE_M3_API_KEY    — API key for the embedder  (default: simple-api-key)
    QDRANT_URL        — Qdrant base URL            (default: http://localhost:6333)
    QDRANT_API_KEY    — Qdrant API key             (default: none)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Allow running from the repo root or the query/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / "env" / ".env", override=False)

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchText,
    NamedSparseVector,
    Range,
    SparseVector,
)

from pipeline.embeddings import encode_sparse

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None

ALL_SUBJECT_COLLECTIONS = [
    "arxiv-cs-ml-ai",
    "arxiv-cs-systems-theory",
    "arxiv-math-pure",
    "arxiv-math-applied",
    "arxiv-math-phys",
    "arxiv-stat-eess",
    "arxiv-quantph-grqc",
    "arxiv-hep",
    "arxiv-condmat",
    "arxiv-astro",
    "arxiv-nucl-nlin-physother",
    "arxiv-qbio-qfin-econ",
    "arxiv-misc",
]


def build_filter(
    year_min: Optional[int],
    year_max: Optional[int],
    author: Optional[str],
) -> Optional[Filter]:
    conditions = []
    if year_min is not None or year_max is not None:
        rng: dict = {}
        if year_min is not None:
            rng["gte"] = float(year_min)
        if year_max is not None:
            rng["lte"] = float(year_max)
        conditions.append(FieldCondition(key="year", range=Range(**rng)))
    if author:
        conditions.append(FieldCondition(key="authors", match=MatchText(text=author)))
    if not conditions:
        return None
    return Filter(must=conditions)


def search_collection(
    client: QdrantClient,
    collection: str,
    sparse_vec: SparseVector,
    top_k: int,
    query_filter: Optional[Filter],
) -> list[dict]:
    """Run sparse search against a single Qdrant collection."""
    try:
        results = client.search(
            collection_name=collection,
            query_vector=NamedSparseVector(name="sparse_text", vector=sparse_vec),
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )
        return [
            {
                "collection": collection,
                "score":      round(r.score, 6),
                "arxiv_id":   r.payload.get("arxiv_id", ""),
                "title":      r.payload.get("title", ""),
                "authors":    r.payload.get("authors", ""),
                "year":       r.payload.get("year", ""),
                "categories": r.payload.get("categories", ""),
                "abstract":   (r.payload.get("abstract", "") or "")[:300] + "...",
            }
            for r in results
        ]
    except Exception as exc:
        print(f"  Warning: search failed for {collection}: {exc}", file=sys.stderr)
        return []


def rrf_merge(results_by_collection: dict[str, list[dict]], k: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion across multiple collection result lists.
    RRF score = Σ 1 / (k + rank) across all lists a document appears in.
    """
    scores: dict[str, float] = {}
    docs:   dict[str, dict]  = {}

    for _coll, results in results_by_collection.items():
        for rank, doc in enumerate(results, start=1):
            uid = doc["arxiv_id"]
            scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank)
            docs[uid]   = doc

    merged = sorted(docs.values(), key=lambda d: scores[d["arxiv_id"]], reverse=True)
    for doc in merged:
        doc["rrf_score"] = round(scores[doc["arxiv_id"]], 6)
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Sparse (SPLADE) search over Arxiv Qdrant collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--query",      required=True,          help="Search query")
    parser.add_argument("--collection", default=None,           help="Target collection (default: all)")
    parser.add_argument("--top-k",      type=int, default=10,   help="Results to return (default: 10)")
    parser.add_argument("--year-min",   type=int, default=None, help="Filter: minimum year")
    parser.add_argument("--year-max",   type=int, default=None, help="Filter: maximum year")
    parser.add_argument("--author",     default=None,           help="Filter: author name substring")
    parser.add_argument("--json",       action="store_true",    help="Output as JSON")
    args = parser.parse_args()

    client       = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
    query_filter = build_filter(args.year_min, args.year_max, args.author)

    print(f"Encoding query via BGE-M3 sparse embedder: '{args.query}'", file=sys.stderr)
    sparse_list = encode_sparse([args.query])
    sv_local    = sparse_list[0]
    qdrant_sv   = SparseVector(indices=sv_local.indices, values=sv_local.values)
    print(f"Sparse vector: {len(qdrant_sv.indices)} non-zero tokens", file=sys.stderr)

    if args.collection:
        results = search_collection(client, args.collection, qdrant_sv, args.top_k, query_filter)
    else:
        results_by_coll: dict[str, list[dict]] = {}
        for coll in ALL_SUBJECT_COLLECTIONS:
            res = search_collection(client, coll, qdrant_sv, args.top_k, query_filter)
            if res:
                results_by_coll[coll] = res
        results = rrf_merge(results_by_coll)[: args.top_k]

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    print(f"\nTop {len(results)} sparse results for: {args.query!r}")
    print(f"Mode: {'collection=' + args.collection if args.collection else 'all collections (RRF merged)'}")
    print("─" * 80)
    for i, doc in enumerate(results, 1):
        score = doc.get("rrf_score") or doc.get("score", 0)
        print(f"\n{i:2}. [{score:.4f}]  {doc['arxiv_id']}  ({doc['year']})  [{doc['collection']}]")
        print(f"    {doc['title']}")
        if doc["authors"]:
            a = doc["authors"]
            print(f"    {a[:90]}{'...' if len(a) > 90 else ''}")
        if doc["abstract"]:
            print(f"    {doc['abstract']}")
    print()


if __name__ == "__main__":
    main()
