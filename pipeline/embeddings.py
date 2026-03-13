"""
pipeline/embeddings.py
======================
Client for BGE-M3 dense and sparse embedding services.

Dense:  vLLM at BGE_M3_DENSE_URL (OpenAI /v1/embeddings compatible)
Sparse: custom FastAPI at BGE_M3_SPARSE_URL (/encode endpoint)

Both use httpx with exponential-backoff retry logic.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import List

import httpx
import numpy as np
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../env/.env"), override=False)

logger = logging.getLogger(__name__)

DENSE_URL = os.environ.get("BGE_M3_DENSE_URL", "http://localhost:8025")
SPARSE_URL = os.environ.get("BGE_M3_SPARSE_URL", "http://localhost:8035")
API_KEY = os.environ.get("BGE_M3_API_KEY", "simple-api-key")

DENSE_MODEL = "bge-m3-embedder"
MAX_RETRIES = 4
BASE_BACKOFF = 0.5  # seconds
REQUEST_TIMEOUT = 120.0  # seconds


@dataclass
class SparseVector:
    """Sparse representation of a BGE-M3 SPLADE-style embedding."""

    indices: List[int]
    values: List[float]

    def __repr__(self) -> str:
        nnz = len(self.indices)
        return f"SparseVector(nnz={nnz})"


# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------

def _make_client() -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=10.0),
        headers={"Authorization": f"Bearer {API_KEY}"},
    )


def _retry_post(client: httpx.Client, url: str, payload: dict) -> dict:
    """POST with exponential back-off; raises on final failure."""
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
                "Request to %s failed (attempt %d/%d): %s — retrying in %.1fs",
                url,
                attempt + 1,
                MAX_RETRIES,
                exc,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"All {MAX_RETRIES} attempts to {url} failed") from last_exc


# ---------------------------------------------------------------------------
# Dense embedding
# ---------------------------------------------------------------------------

def encode_dense(texts: List[str]) -> np.ndarray:
    """
    Call the BGE-M3 dense embedder (vLLM /v1/embeddings) and return
    a float32 numpy array of shape (len(texts), 1024).

    Handles batching transparently — the vLLM server has its own internal
    batch scheduler, so we send all texts in a single request and let the
    server handle throughput optimisation.
    """
    if not texts:
        return np.empty((0, 1024), dtype=np.float32)

    url = f"{DENSE_URL.rstrip('/')}/v1/embeddings"
    payload = {
        "model": DENSE_MODEL,
        "input": texts,
        "encoding_format": "float",
    }

    with _make_client() as client:
        data = _retry_post(client, url, payload)

    # OpenAI response: {"data": [{"embedding": [...], "index": i}, ...]}
    items = sorted(data["data"], key=lambda x: x["index"])
    embeddings = np.array([item["embedding"] for item in items], dtype=np.float32)

    logger.debug("encode_dense: %d texts → shape %s", len(texts), embeddings.shape)
    return embeddings


# ---------------------------------------------------------------------------
# Sparse embedding
# ---------------------------------------------------------------------------

def encode_sparse(texts: List[str]) -> List[SparseVector]:
    """
    Call the custom BGE-M3 sparse embedder (FastAPI /encode) and return
    a list of SparseVector objects, one per input text.

    The /encode endpoint expects:
        {"texts": ["text1", "text2", ...]}
    and returns:
        {"embeddings": [{"indices": [...], "values": [...]}, ...]}
    """
    if not texts:
        return []

    url = f"{SPARSE_URL.rstrip('/')}/encode"
    payload = {"texts": texts}

    with _make_client() as client:
        data = _retry_post(client, url, payload)

    results: List[SparseVector] = []
    # Handle both response formats:
    #   - {"embeddings": [{"indices": ..., "values": ...}, ...]}
    #   - {"items": [{"sparse": {"indices": ..., "values": ...}}, ...]}
    if "items" in data:
        items = [item["sparse"] for item in data["items"]]
    else:
        items = data["embeddings"]
    for item in items:
        results.append(
            SparseVector(
                indices=[int(i) for i in item["indices"]],
                values=[float(v) for v in item["values"]],
            )
        )

    logger.debug("encode_sparse: %d texts → %d sparse vectors", len(texts), len(results))
    return results


# ---------------------------------------------------------------------------
# Async wrappers (used by the LangGraph pipeline)
# ---------------------------------------------------------------------------

async def async_encode_dense(texts: List[str]) -> np.ndarray:
    """Asyncio-compatible dense encoding (runs sync call in thread pool)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, encode_dense, texts)


async def async_encode_sparse(texts: List[str]) -> List[SparseVector]:
    """Asyncio-compatible sparse encoding (runs sync call in thread pool)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, encode_sparse, texts)


async def async_encode_both(texts: List[str]):
    """
    Encode dense and sparse in parallel.

    Returns
    -------
    (dense_array, sparse_list)
    """
    dense_task = asyncio.create_task(async_encode_dense(texts))
    sparse_task = asyncio.create_task(async_encode_sparse(texts))
    dense, sparse = await asyncio.gather(dense_task, sparse_task)
    return dense, sparse


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG)
    sample = sys.argv[1:] if len(sys.argv) > 1 else ["Hello world", "Quantum entanglement"]
    print(f"\n=== Dense embeddings ===")
    d = encode_dense(sample)
    print(f"Shape: {d.shape}, dtype: {d.dtype}")
    print(f"First 5 values of first vector: {d[0, :5]}")

    print(f"\n=== Sparse embeddings ===")
    s = encode_sparse(sample)
    for i, sv in enumerate(s):
        print(f"  [{i}] {sv}")
