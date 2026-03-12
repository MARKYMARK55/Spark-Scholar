"""
pipeline/reranker.py
====================
BGE-M3 cross-encoder reranker via vLLM's /score endpoint.

vLLM serves bge-reranker-v2-m3 and exposes a /score endpoint that accepts:
    POST /score
    {
        "model": "bge-reranker-v2-m3",
        "text_1": "<query>",
        "text_2": ["doc1_text", "doc2_text", ...]
    }

Response:
    {
        "data": [
            {"index": 0, "score": 0.95},
            {"index": 1, "score": 0.72},
            ...
        ]
    }

The reranker takes the candidate documents produced by hybrid_search,
scores each one against the query, and returns the top_n by descending score.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../env/.env"), override=False)

logger = logging.getLogger(__name__)

RERANKER_URL = os.environ.get("BGE_RERANKER_URL", "http://localhost:8020")
API_KEY = os.environ.get("BGE_M3_API_KEY", "simple-api-key")
RERANKER_MODEL = "bge-reranker-v2-m3"

MAX_RETRIES = 3
BASE_BACKOFF = 0.5
REQUEST_TIMEOUT = 120.0

# vLLM /score has a practical limit on text_2 length; batch if needed
SCORE_BATCH_SIZE = 100


def _make_client() -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )


def _post_with_retry(client: httpx.Client, url: str, payload: dict) -> dict:
    """POST with exponential backoff retry."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPStatusError, httpx.TransportError, httpx.TimeoutException) as exc:
            last_exc = exc
            wait = BASE_BACKOFF * (2 ** attempt)
            logger.warning(
                "Reranker request failed (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1,
                MAX_RETRIES,
                exc,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"All {MAX_RETRIES} reranker attempts failed") from last_exc


def _get_document_text(doc: dict) -> str:
    """
    Extract the most informative text from a document dict for reranking.

    Priority: chunk_text (PDF chunk) > abstract > title + abstract
    """
    chunk = doc.get("chunk_text", "").strip()
    if chunk:
        # Truncate very long chunks to avoid token limit issues
        return chunk[:2048]

    title = doc.get("title", "").strip()
    abstract = doc.get("abstract", "").strip()

    if abstract:
        combined = f"{title}\n\n{abstract}" if title else abstract
        return combined[:2048]

    return title[:512] if title else ""


def _score_batch(
    client: httpx.Client,
    query: str,
    doc_texts: List[str],
    score_url: str,
) -> List[float]:
    """
    Score a batch of document texts against the query.

    Returns a list of float scores in the same order as doc_texts.
    """
    payload = {
        "model": RERANKER_MODEL,
        "text_1": query,
        "text_2": doc_texts,
    }

    try:
        data = _post_with_retry(client, score_url, payload)

        # Parse response: {"data": [{"index": i, "score": f}, ...]}
        raw_scores = data.get("data") or data.get("scores") or []

        if not raw_scores:
            logger.warning("Reranker returned empty scores for batch of %d docs", len(doc_texts))
            return [0.0] * len(doc_texts)

        # Handle both formats:
        # - {"data": [{"index": 0, "score": 0.9}, ...]}  (vLLM /score standard)
        # - {"scores": [0.9, 0.8, ...]}                  (flat list)
        if isinstance(raw_scores[0], dict):
            scores_by_index: dict[int, float] = {}
            for item in raw_scores:
                idx = item.get("index", 0)
                score = item.get("score", 0.0)
                # Handle nested score objects from some vLLM versions
                if isinstance(score, dict):
                    score = score.get("score", 0.0)
                scores_by_index[idx] = float(score)
            return [scores_by_index.get(i, 0.0) for i in range(len(doc_texts))]
        else:
            return [float(s) for s in raw_scores]

    except Exception as exc:
        logger.warning("Reranker batch scoring failed: %s — using zero scores", exc)
        return [0.0] * len(doc_texts)


def rerank(
    query: str,
    documents: List[dict],
    top_n: int = 10,
    score_url: Optional[str] = None,
) -> List[dict]:
    """
    Rerank a list of candidate documents using the BGE-M3 cross-encoder.

    The cross-encoder jointly encodes (query, document) pairs, giving much
    better relevance scores than the bi-encoder cosine similarities from ANN
    retrieval. Use this as the final ranking stage.

    Parameters
    ----------
    query : str
        The search query string.
    documents : list[dict]
        Candidate documents from hybrid_search. Each must be a dict with at
        least one of: chunk_text, abstract, title.
    top_n : int
        Number of top-ranked documents to return.
    score_url : str, optional
        Override the reranker URL (defaults to BGE_RERANKER_URL env var).

    Returns
    -------
    list[dict]
        Top-n documents sorted by cross-encoder score descending.
        Each document gets a "rerank_score" key added.
    """
    if not documents:
        return []

    if not query.strip():
        logger.warning("rerank: empty query, returning first %d docs unranked", top_n)
        return documents[:top_n]

    effective_url = (score_url or RERANKER_URL).rstrip("/")
    endpoint = f"{effective_url}/score"

    logger.info("rerank: scoring %d candidates with cross-encoder (top_n=%d)", len(documents), top_n)

    # Extract text for each document
    doc_texts = [_get_document_text(doc) for doc in documents]

    # Score in batches to avoid overwhelming the reranker
    all_scores: List[float] = []
    with _make_client() as client:
        for batch_start in range(0, len(doc_texts), SCORE_BATCH_SIZE):
            batch_texts = doc_texts[batch_start : batch_start + SCORE_BATCH_SIZE]
            batch_scores = _score_batch(client, query, batch_texts, endpoint)
            all_scores.extend(batch_scores)

    # Attach scores and sort
    scored_docs = []
    for doc, score in zip(documents, all_scores):
        doc_copy = doc.copy()
        doc_copy["rerank_score"] = score
        scored_docs.append(doc_copy)

    scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
    result = scored_docs[:top_n]

    logger.info(
        "rerank: top scores = %s",
        [round(d["rerank_score"], 4) for d in result[:5]],
    )
    return result


async def async_rerank(
    query: str,
    documents: List[dict],
    top_n: int = 10,
) -> List[dict]:
    """Async-compatible reranker (runs sync rerank in thread pool)."""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, rerank, query, documents, top_n)


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    query = "transformer attention mechanism"
    dummy_docs = [
        {
            "arxiv_id": "2301.00001",
            "title": "Attention Is All You Need",
            "abstract": "We propose the Transformer architecture based entirely on attention mechanisms.",
        },
        {
            "arxiv_id": "2301.00002",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "abstract": "We introduce BERT, which is designed to pre-train deep bidirectional representations.",
        },
        {
            "arxiv_id": "2301.00003",
            "title": "Superconductivity in cuprate materials",
            "abstract": "We study high-temperature superconductors and their phase transitions.",
        },
    ]

    print(f"Query: {query!r}")
    print(f"Input docs: {len(dummy_docs)}")

    ranked = rerank(query, dummy_docs, top_n=3)
    for i, doc in enumerate(ranked):
        print(f"\n[{i+1}] {doc['title']}")
        print(f"     rerank_score: {doc.get('rerank_score', 0):.4f}")
