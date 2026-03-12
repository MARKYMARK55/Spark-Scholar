"""
sparse_embedder/sparse_embed.py
================================
FastAPI service for BGE-M3 sparse (SPLADE) embeddings.

BGE-M3 is a unified model that can produce three types of representations:
  1. Dense (CLS token) — handled by vLLM on port 8025
  2. Sparse (SPLADE-style weighted term vectors) — this service, port 8035
  3. ColBERT multi-vector — not used in this stack

This service loads the FlagEmbedding BGEM3FlagModel and serves sparse embeddings
via a simple /encode POST endpoint.

The sparse representation is a dict mapping token_id → weight (float), which
we return as parallel arrays of indices and values suitable for Qdrant.

API
---
POST /encode
    Request:  {"texts": ["text1", "text2", ...], "batch_size": 32}
    Response: {"embeddings": [{"indices": [...], "values": [...]}, ...]}

GET /health
    Response: {"status": "ok", "model": "BAAI/bge-m3", "device": "cuda"}

Usage
-----
    uvicorn sparse_embed:app --host 0.0.0.0 --port 8035 --workers 1

Requirements
------------
    pip install FlagEmbedding fastapi uvicorn torch
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("SPARSE_MODEL", "BAAI/bge-m3")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
BATCH_SIZE = int(os.environ.get("SPARSE_BATCH_SIZE", "32"))
MAX_LENGTH = int(os.environ.get("SPARSE_MAX_LENGTH", "512"))

_model = None
_device = "cpu"


def _load_model():
    """Load BGE-M3 model for sparse embedding."""
    global _model, _device

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading %s on %s for sparse embeddings...", MODEL_NAME, _device)

    try:
        from FlagEmbedding import BGEM3FlagModel

        _model = BGEM3FlagModel(
            MODEL_NAME,
            use_fp16=True if _device == "cuda" else False,
            device=_device,
        )
        logger.info("BGE-M3 sparse model loaded successfully")
    except Exception as exc:
        logger.error("Failed to load sparse model: %s", exc)
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    _load_model()
    yield
    logger.info("Sparse embedder shutting down")


app = FastAPI(
    title="BGE-M3 Sparse Embedder",
    description="FastAPI service for SPLADE-style sparse embeddings from BGE-M3",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class EncodeRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to encode")
    batch_size: int = Field(default=BATCH_SIZE, description="Processing batch size")
    max_length: int = Field(default=MAX_LENGTH, description="Max token length")
    return_dense: bool = Field(default=False, description="Also return dense embeddings")


class SparseEmbedding(BaseModel):
    indices: List[int]
    values: List[float]


class EncodeResponse(BaseModel):
    embeddings: List[SparseEmbedding]
    model: str = MODEL_NAME
    usage: dict = {}


# ---------------------------------------------------------------------------
# Encoding logic
# ---------------------------------------------------------------------------

def _encode_sparse(
    texts: List[str],
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
) -> List[SparseEmbedding]:
    """
    Encode texts to sparse SPLADE-style vectors using BGE-M3.

    Parameters
    ----------
    texts : list[str]
    batch_size : int
    max_length : int

    Returns
    -------
    list[SparseEmbedding]
    """
    global _model

    if _model is None:
        raise RuntimeError("Model not loaded")

    results: List[SparseEmbedding] = []
    total_tokens = 0

    # Process in batches
    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start : batch_start + batch_size]

        try:
            # BGE-M3 encode with return_sparse=True
            output = _model.encode(
                batch,
                batch_size=len(batch),
                max_length=max_length,
                return_dense=False,
                return_sparse=True,
                return_colbert_vecs=False,
            )

            sparse_vecs = output.get("lexical_weights", [])

            for sv in sparse_vecs:
                if isinstance(sv, dict):
                    # sv is {token_id: weight, ...}
                    indices = [int(k) for k in sv.keys()]
                    values = [float(v) for v in sv.values()]
                else:
                    # sv might be a tensor
                    sv_tensor = sv if torch.is_tensor(sv) else torch.tensor(sv)
                    nonzero = sv_tensor.nonzero(as_tuple=True)[0]
                    indices = nonzero.tolist()
                    values = sv_tensor[nonzero].tolist()

                # Sort by index for consistency
                if indices:
                    pairs = sorted(zip(indices, values), key=lambda x: x[0])
                    indices, values = zip(*pairs)
                    indices = list(indices)
                    values = list(values)
                else:
                    indices, values = [], []

                total_tokens += len(indices)
                results.append(SparseEmbedding(indices=indices, values=values))

        except Exception as exc:
            logger.error("Batch encoding failed (batch_start=%d): %s", batch_start, exc)
            # Return zero sparse vectors for failed batch
            for _ in batch:
                results.append(SparseEmbedding(indices=[], values=[]))

    return results


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok" if _model is not None else "loading",
        "model": MODEL_NAME,
        "device": _device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


@app.post("/encode", response_model=EncodeResponse)
async def encode(request: EncodeRequest):
    """
    Encode a list of texts to sparse SPLADE vectors.

    Request body
    ------------
    texts : list[str]
        Texts to encode (max 512 per request recommended).
    batch_size : int
        Internal processing batch size (default: 32).

    Response
    --------
    embeddings : list of {indices: [...], values: [...]}
        One sparse vector per input text.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not request.texts:
        return EncodeResponse(embeddings=[], usage={"total_texts": 0})

    if len(request.texts) > 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Too many texts (got {len(request.texts)}, max 1024). Split into smaller batches.",
        )

    start = time.time()

    embeddings = _encode_sparse(
        request.texts,
        batch_size=request.batch_size,
        max_length=request.max_length,
    )

    elapsed = time.time() - start
    avg_nnz = sum(len(e.indices) for e in embeddings) / max(len(embeddings), 1)

    return EncodeResponse(
        embeddings=embeddings,
        model=MODEL_NAME,
        usage={
            "total_texts": len(request.texts),
            "elapsed_seconds": round(elapsed, 3),
            "avg_nonzero": round(avg_nnz, 1),
            "texts_per_second": round(len(request.texts) / elapsed, 1) if elapsed > 0 else 0,
        },
    )


@app.get("/")
async def root():
    return {
        "service": "BGE-M3 Sparse Embedder",
        "model": MODEL_NAME,
        "endpoints": ["/health", "/encode"],
    }


# ---------------------------------------------------------------------------
# Direct run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    host = os.environ.get("SPARSE_EMBED_HOST", "0.0.0.0")
    port = int(os.environ.get("SPARSE_EMBED_PORT", "8035"))

    uvicorn.run(
        "sparse_embed:app",
        host=host,
        port=port,
        workers=1,
        log_level="info",
        reload=False,
    )
