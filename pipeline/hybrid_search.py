"""
pipeline/hybrid_search.py
=========================
Full hybrid dense + sparse search using Qdrant's native Prefetch + RRF fusion.

Architecture
------------
For each collection:
  1. Issue a query_points() call with two Prefetch legs:
     - Leg A: dense ANN search via HNSW (using_vector="dense_embedding")
     - Leg B: sparse inverted index search (using_vector="sparse_text")
  2. Qdrant fuses the two result lists using Reciprocal Rank Fusion (RRF).

Results from multiple collections are merged with a second Python-side RRF pass.

Optional filters: year range (via payload year field) and author substring match.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchText,
    MatchValue,
    NamedSparseVector,
    NamedVector,
    Prefetch,
    Range,
    SparseVector,
)

from .embeddings import async_encode_dense, async_encode_sparse, SparseVector as LocalSparseVector

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../env/.env"), override=False)

logger = logging.getLogger(__name__)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None

# How many candidates to fetch per collection before RRF merge
PREFETCH_LIMIT = 100


def _make_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)


def _build_filter(
    year_min: Optional[int],
    year_max: Optional[int],
    author: Optional[str],
) -> Optional[Filter]:
    """Build a Qdrant Filter from optional constraints."""
    conditions = []

    if year_min is not None or year_max is not None:
        range_kwargs: dict[str, Any] = {}
        if year_min is not None:
            range_kwargs["gte"] = float(year_min)
        if year_max is not None:
            range_kwargs["lte"] = float(year_max)
        conditions.append(
            FieldCondition(key="year", range=Range(**range_kwargs))
        )

    if author:
        # Match substring in the authors field (case-insensitive)
        conditions.append(
            FieldCondition(key="authors", match=MatchText(text=author))
        )

    if not conditions:
        return None

    return Filter(must=conditions)


def _rrf_score(rank: int, k: int = 60) -> float:
    """Standard Reciprocal Rank Fusion score: 1 / (k + rank)."""
    return 1.0 / (k + rank + 1)  # +1 for 0-based rank


def _python_rrf_merge(
    all_results: list[list[dict]],
    top_k: int,
) -> list[dict]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    all_results : list of lists
        Each inner list is a ranked list of result dicts (must have 'arxiv_id' key).
    top_k : int
        Number of results to return.

    Returns
    -------
    list[dict]
        Merged and re-ranked results with rrf_score added.
    """
    # Map arxiv_id → accumulated RRF score + result dict
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    for result_list in all_results:
        for rank, doc in enumerate(result_list):
            doc_id = doc.get("arxiv_id") or doc.get("id") or str(rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + _rrf_score(rank)
            if doc_id not in docs:
                docs[doc_id] = doc

    # Sort by accumulated RRF score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    merged = []
    for doc_id, score in ranked:
        doc = docs[doc_id].copy()
        doc["score"] = score
        merged.append(doc)

    return merged


def _point_to_dict(point, collection: str) -> dict:
    """Convert a Qdrant ScoredPoint to a plain dict."""
    payload = point.payload or {}
    return {
        "arxiv_id": payload.get("arxiv_id") or payload.get("id") or str(point.id),
        "title": payload.get("title", ""),
        "abstract": payload.get("abstract", ""),
        "authors": payload.get("authors", ""),
        "year": payload.get("year"),
        "categories": payload.get("categories", ""),
        "score": point.score,
        "collection": collection,
        # Extra fields that may be present for PDF chunks
        "chunk_text": payload.get("chunk_text", ""),
        "page_num": payload.get("page_num"),
        "source_file": payload.get("source_file", ""),
        "topic_id": payload.get("topic_id"),
        "topic_name": payload.get("topic_name", ""),
        "type": payload.get("type", "abstract"),
    }


def _search_collection(
    client: QdrantClient,
    collection: str,
    dense_vec: list[float],
    sparse_vec: LocalSparseVector,
    prefetch_limit: int,
    query_filter: Optional[Filter],
) -> list[dict]:
    """
    Run hybrid search on a single Qdrant collection.

    Uses client.query_points() with two Prefetch legs (dense + sparse) and
    FusionQuery(fusion=Fusion.RRF) for native server-side fusion.
    """
    try:
        qdrant_sparse = SparseVector(
            indices=sparse_vec.indices,
            values=sparse_vec.values,
        )

        prefetch = [
            Prefetch(
                query=NamedVector(name="dense_embedding", vector=dense_vec),
                limit=prefetch_limit,
                filter=query_filter,
                using="dense_embedding",
            ),
            Prefetch(
                query=NamedSparseVector(name="sparse_text", vector=qdrant_sparse),
                limit=prefetch_limit,
                filter=query_filter,
                using="sparse_text",
            ),
        ]

        results = client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=prefetch_limit,
            with_payload=True,
        )

        points = results.points if hasattr(results, "points") else results
        return [_point_to_dict(p, collection) for p in points]

    except Exception as exc:
        logger.warning("hybrid_search: error searching collection %r: %s", collection, exc)
        return []


async def _async_search_collection(
    collection: str,
    dense_vec: list[float],
    sparse_vec: LocalSparseVector,
    prefetch_limit: int,
    query_filter: Optional[Filter],
) -> list[dict]:
    """Async wrapper around _search_collection."""
    loop = asyncio.get_event_loop()
    client = _make_qdrant_client()

    def _run():
        return _search_collection(client, collection, dense_vec, sparse_vec, prefetch_limit, query_filter)

    return await loop.run_in_executor(None, _run)


def hybrid_search(
    query: str,
    collections: list[str],
    top_k: int = 10,
    rerank_n: int = 50,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    author: Optional[str] = None,
) -> list[dict]:
    """
    Full hybrid search across one or more Qdrant collections.

    Steps
    -----
    1. Encode the query to dense (float[1024]) and sparse (indices/values) in parallel.
    2. For each collection, run a Qdrant query_points() with Prefetch + RRF.
    3. Merge results from all collections with a second Python-side RRF pass.
    4. Return up to rerank_n candidates (the reranker will trim to top_k).

    Parameters
    ----------
    query : str
        The search query string.
    collections : list[str]
        Qdrant collection names to search.
    top_k : int
        Final number of results to return (after inter-collection merge).
        Note: the reranker will further trim; pass rerank_n for the pre-rerank pool.
    rerank_n : int
        Number of candidates to fetch per collection (before merge and rerank).
    year_min : int, optional
        Filter: only include papers from this year onward.
    year_max : int, optional
        Filter: only include papers up to this year.
    author : str, optional
        Filter: only include papers where authors field contains this substring.

    Returns
    -------
    list[dict]
        Each dict has: arxiv_id, title, abstract, authors, year, categories,
        score, collection, chunk_text, page_num, source_file, topic_id,
        topic_name, type.
    """
    if not collections:
        logger.warning("hybrid_search: empty collections list, returning empty result")
        return []

    async def _run_all():
        # Encode dense and sparse in parallel first
        dense_arr, sparse_list = await asyncio.gather(
            async_encode_dense([query]),
            async_encode_sparse([query]),
        )
        dense_vec = dense_arr[0].tolist()
        sparse_vec = sparse_list[0]

        query_filter = _build_filter(year_min, year_max, author)
        prefetch_limit = max(rerank_n, PREFETCH_LIMIT)

        # Search all collections concurrently
        tasks = [
            _async_search_collection(coll, dense_vec, sparse_vec, prefetch_limit, query_filter)
            for coll in collections
        ]
        per_collection_results = await asyncio.gather(*tasks)
        return list(per_collection_results)

    # Run the async pipeline in the event loop (handles both sync and async callers)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an async context (e.g. FastAPI) — use nest_asyncio or run directly
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _run_all())
                per_collection_results = future.result(timeout=120)
        else:
            per_collection_results = loop.run_until_complete(_run_all())
    except RuntimeError:
        per_collection_results = asyncio.run(_run_all())

    # Merge across collections using RRF
    if len(per_collection_results) == 1:
        # Single collection — just return the results directly (already RRF-fused by Qdrant)
        merged = per_collection_results[0][:rerank_n]
    else:
        merged = _python_rrf_merge(per_collection_results, top_k=rerank_n)

    logger.info(
        "hybrid_search: query=%r collections=%s → %d candidates (top_k=%d)",
        query[:60],
        collections,
        len(merged),
        top_k,
    )
    return merged


async def async_hybrid_search(
    query: str,
    collections: list[str],
    top_k: int = 10,
    rerank_n: int = 50,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    author: Optional[str] = None,
) -> list[dict]:
    """
    Async version of hybrid_search — preferred when called from async context
    (e.g. FastAPI endpoint or LangGraph async pipeline).
    """
    if not collections:
        return []

    dense_arr, sparse_list = await asyncio.gather(
        async_encode_dense([query]),
        async_encode_sparse([query]),
    )
    dense_vec = dense_arr[0].tolist()
    sparse_vec = sparse_list[0]
    query_filter = _build_filter(year_min, year_max, author)
    prefetch_limit = max(rerank_n, PREFETCH_LIMIT)

    tasks = [
        _async_search_collection(coll, dense_vec, sparse_vec, prefetch_limit, query_filter)
        for coll in collections
    ]
    per_collection_results = await asyncio.gather(*tasks)

    if len(per_collection_results) == 1:
        merged = list(per_collection_results[0])[:rerank_n]
    else:
        merged = _python_rrf_merge(list(per_collection_results), top_k=rerank_n)

    logger.info(
        "async_hybrid_search: query=%r → %d candidates from %d collections",
        query[:60],
        len(merged),
        len(collections),
    )
    return merged


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "transformer attention mechanism for NLP"
    print(f"Query: {query!r}")
    results = hybrid_search(query, collections=["arxiv-cs-ml-ai"], top_k=5, rerank_n=20)
    for i, r in enumerate(results):
        print(f"\n[{i+1}] {r.get('title', 'N/A')} ({r.get('year', '?')})")
        print(f"     ID: {r.get('arxiv_id')}  Score: {r.get('score', 0):.4f}")
        print(f"     Authors: {r.get('authors', '')[:80]}")
