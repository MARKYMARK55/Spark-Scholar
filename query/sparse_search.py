#!/usr/bin/env python3
"""
sparse_search.py
----------------
Query the Arxiv sparse vector index in Qdrant.

Demonstrates:
  - Single-collection sparse search
  - Multi-collection fan-out with result merging
  - Filtering by year, category, or author
  - RRF (Reciprocal Rank Fusion) across collections

This is the sparse-only query layer. In the full production stack this is
combined with dense vector search (BGE-M3 1024-dim) and the results are
merged via Qdrant's built-in hybrid RRF before being passed to the reranker.
That full hybrid pipeline is out of scope for this repo — this script shows
the sparse side in isolation.

Usage:
    python sparse_search.py --query "attention mechanism transformers self-attention"
    python sparse_search.py --query "Hawkes process excitation kernel" --collection arxiv-stat-eess
    python sparse_search.py --query "2305.14314"  # search by Arxiv ID
    python sparse_search.py --query "Vaswani attention" --year-min 2017 --year-max 2018
"""

import argparse
import json
import sys
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, SearchRequest, NamedSparseVector

# Collections to fan-out to if no specific collection is given
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


def load_bge_m3_sparse():
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError:
        print("ERROR: pip install FlagEmbedding", file=sys.stderr)
        sys.exit(1)
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda")
    return model


def encode_query_sparse(model, query: str) -> SparseVector:
    """Encode a query string to a Qdrant SparseVector using BGE-M3 lexical weights."""
    output = model.encode(
        [query],
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    sparse_dict = output["lexical_weights"][0]
    indices = [int(k) for k in sparse_dict.keys()]
    values  = [float(v) for v in sparse_dict.values()]
    return SparseVector(indices=indices, values=values)


def build_filter(year_min: Optional[int], year_max: Optional[int], author: Optional[str]) -> Optional[dict]:
    """Build a Qdrant filter dict from optional constraints."""
    conditions = []
    if year_min:
        conditions.append({"key": "year", "range": {"gte": year_min}})
    if year_max:
        conditions.append({"key": "year", "range": {"lte": year_max}})
    if author:
        conditions.append({"key": "authors", "match": {"text": author}})
    if not conditions:
        return None
    return {"must": conditions} if len(conditions) > 1 else conditions[0]


def search_collection(
    client: QdrantClient,
    collection: str,
    query_vec: SparseVector,
    top_k: int,
    filter_: Optional[dict],
) -> list[dict]:
    """Run sparse search against a single collection."""
    try:
        results = client.search(
            collection_name=collection,
            query_vector=NamedSparseVector(name="sparse_text", vector=query_vec),
            limit=top_k,
            with_payload=True,
            query_filter=filter_,
        )
        return [
            {
                "collection": collection,
                "score":      r.score,
                "arxiv_id":   r.payload.get("arxiv_id", ""),
                "title":      r.payload.get("title", ""),
                "authors":    r.payload.get("authors", ""),
                "year":       r.payload.get("year", ""),
                "categories": r.payload.get("categories", ""),
                "abstract":   r.payload.get("abstract", "")[:300] + "...",
            }
            for r in results
        ]
    except Exception as e:
        print(f"  Warning: search failed for {collection}: {e}", file=sys.stderr)
        return []


def rrf_merge(results_by_collection: dict[str, list[dict]], k: int = 60) -> list[dict]:
    """
    Reciprocal Rank Fusion across multiple collection result lists.
    RRF score = sum(1 / (k + rank)) across all lists a document appears in.
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
    parser = argparse.ArgumentParser(description="Sparse search over Arxiv Qdrant collections")
    parser.add_argument("--query",      required=True, help="Search query")
    parser.add_argument("--collection", default=None,  help="Target collection (default: all)")
    parser.add_argument("--top-k",      type=int, default=10)
    parser.add_argument("--year-min",   type=int, default=None)
    parser.add_argument("--year-max",   type=int, default=None)
    parser.add_argument("--author",     default=None)
    parser.add_argument("--host",       default="localhost")
    parser.add_argument("--port",       type=int, default=6333)
    parser.add_argument("--json",       action="store_true", help="Output as JSON")
    args = parser.parse_args()

    client  = QdrantClient(host=args.host, port=args.port, timeout=30)
    model   = load_bge_m3_sparse()
    filter_ = build_filter(args.year_min, args.year_max, args.author)

    print(f"Encoding query: '{args.query}'")
    query_vec = encode_query_sparse(model, args.query)
    print(f"Sparse vector: {len(query_vec.indices)} non-zero tokens\n")

    if args.collection:
        results = search_collection(client, args.collection, query_vec, args.top_k, filter_)
    else:
        # Fan out across all subject collections and RRF-merge
        results_by_coll: dict[str, list[dict]] = {}
        for coll in ALL_SUBJECT_COLLECTIONS:
            res = search_collection(client, coll, query_vec, args.top_k, filter_)
            if res:
                results_by_coll[coll] = res
        results = rrf_merge(results_by_coll)[: args.top_k]

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    print(f"Top {len(results)} results:\n{'─'*80}")
    for i, doc in enumerate(results, 1):
        score = doc.get("rrf_score") or doc.get("score", 0)
        print(f"{i:2}. [{score:.4f}] {doc['arxiv_id']} ({doc['year']}) | {doc['collection']}")
        print(f"    {doc['title']}")
        print(f"    {doc['authors'][:80]}{'...' if len(doc['authors'])>80 else ''}")
        print(f"    {doc['abstract']}")
        print()


if __name__ == "__main__":
    main()
