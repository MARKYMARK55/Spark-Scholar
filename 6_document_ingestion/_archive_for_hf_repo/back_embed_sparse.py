#!/usr/bin/env python3
"""
ingest/back_embed_sparse.py
============================
Add sparse vectors to document collections that only have dense embeddings.

For each collection:
1. Recreate the collection with both dense + sparse vector configs
2. Scroll through all points, keeping dense vectors and payloads
3. Encode the text field through the BGE-M3 sparse embedder
4. Upsert points with both dense and sparse vectors

Usage
-----
    # Process all document collections (auto-detects non-arxiv collections)
    python ingest/back_embed_sparse.py

    # Process specific collections
    python ingest/back_embed_sparse.py --collections Python-Books Docker ML

    # Dry run (show what would be processed)
    python ingest/back_embed_sparse.py --dry-run

    # Adjust batch size for sparse embedding
    python ingest/back_embed_sparse.py --batch-size 64
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from tqdm import tqdm

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../env/.env"), override=False)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.embeddings import encode_sparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None

# Collections to skip (arXiv collections already have sparse, system collections)
SKIP_PREFIXES = ("arxiv-", "arXiv", "open-webui_", "chat_")

DENSE_DIM = 1024
HNSW_M = 16
HNSW_EF_CONSTRUCT = 100
SCROLL_BATCH = 100


def get_document_collections(client: QdrantClient) -> list[str]:
    """Return collections that are not arXiv or system collections."""
    all_cols = client.get_collections().collections
    doc_cols = []
    for c in all_cols:
        name = c.name
        if any(name.startswith(p) for p in SKIP_PREFIXES):
            continue
        # Check if it already has sparse vectors
        info = client.get_collection(name)
        if info.config.params.sparse_vectors:
            logger.info("  Skipping %s (already has sparse vectors)", name)
            continue
        doc_cols.append(name)
    return sorted(doc_cols)


def back_embed_collection(
    client: QdrantClient,
    collection: str,
    batch_size: int = 32,
) -> int:
    """
    Add sparse vectors to a collection. Returns number of points processed.

    Strategy: since Qdrant doesn't support adding new vector types to existing
    collections, we recreate the collection and re-upsert all points.
    """
    info = client.get_collection(collection)
    total_points = info.points_count
    logger.info("Processing %s (%d points)", collection, total_points)

    if total_points == 0:
        logger.info("  Empty collection, skipping")
        return 0

    # --- Detect vector dimension from collection config
    vectors_config = info.config.params.vectors
    if isinstance(vectors_config, dict):
        # Named vectors: find the dense one
        for vname, vparams in vectors_config.items():
            detected_dim = vparams.size
            detected_name = vname
            break
    else:
        # Single unnamed vector
        detected_dim = vectors_config.size
        detected_name = ""

    if detected_dim != DENSE_DIM:
        logger.warning(
            "  SKIPPING %s: vector dim is %d, expected %d (different embedding model)",
            collection, detected_dim, DENSE_DIM,
        )
        return 0

    # --- Step 1: Read all points (with dense vectors and payloads)
    logger.info("  Step 1/3: Reading all points...")
    all_points = []
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        all_points.extend(points)
        if next_offset is None:
            break
        offset = next_offset

    logger.info("  Read %d points", len(all_points))

    # --- Step 2: Recreate collection with dense + sparse config
    logger.info("  Step 2/3: Recreating collection with sparse vector support...")
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config={
            "dense_embedding": VectorParams(
                size=DENSE_DIM,
                distance=Distance.COSINE,
                hnsw_config=HnswConfigDiff(m=HNSW_M, ef_construct=HNSW_EF_CONSTRUCT),
            ),
        },
        sparse_vectors_config={
            "sparse_text": SparseVectorParams(
                index=SparseIndexParams(on_disk=True),
            ),
        },
    )

    # --- Step 3: Encode sparse + re-upsert in batches
    logger.info("  Step 3/3: Encoding sparse vectors and upserting...")
    processed = 0

    for i in tqdm(range(0, len(all_points), batch_size), desc=f"  {collection}"):
        batch = all_points[i : i + batch_size]

        # Get text for sparse encoding
        texts = []
        for pt in batch:
            text = pt.payload.get("text", "")
            if not text:
                # Fallback: try title + abstract for arxiv-style points
                title = pt.payload.get("title", "")
                abstract = pt.payload.get("abstract", "")
                text = f"{title}\n{abstract}".strip()
            texts.append(text if text else "empty")

        # Encode sparse
        try:
            sparse_vecs = encode_sparse(texts)
        except Exception as e:
            logger.warning("  Sparse encoding failed for batch %d: %s", i, e)
            # Still upsert with empty sparse vectors
            sparse_vecs = [type("SV", (), {"indices": [], "values": []})() for _ in texts]

        # Build points with both vectors
        new_points = []
        for pt, sv in zip(batch, sparse_vecs):
            # Get the dense vector (handle both dict and list formats)
            if isinstance(pt.vector, dict):
                dense_vec = pt.vector.get("dense_embedding", pt.vector.get("dense", []))
            else:
                dense_vec = pt.vector

            new_points.append(PointStruct(
                id=pt.id,
                vector={
                    "dense_embedding": dense_vec,
                    "sparse_text": SparseVector(
                        indices=sv.indices,
                        values=sv.values,
                    ),
                },
                payload=pt.payload,
            ))

        # Upsert batch
        client.upsert(collection_name=collection, points=new_points, wait=True)
        processed += len(new_points)

    logger.info("  Done: %d points with sparse vectors", processed)
    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Add sparse vectors to document collections"
    )
    parser.add_argument(
        "--collections", nargs="+", metavar="NAME",
        help="Specific collections to process (default: auto-detect all document collections)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for sparse embedding (default: 32)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without making changes",
    )
    args = parser.parse_args()

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    if args.collections:
        collections = args.collections
    else:
        logger.info("Auto-detecting document collections...")
        collections = get_document_collections(client)

    logger.info("Collections to process: %d", len(collections))
    for name in collections:
        info = client.get_collection(name)
        logger.info("  %-25s %d points", name, info.points_count)

    if args.dry_run:
        logger.info("Dry run — no changes made")
        return

    total = 0
    start = time.time()
    for name in collections:
        n = back_embed_collection(client, name, batch_size=args.batch_size)
        total += n

    elapsed = time.time() - start
    logger.info(
        "Complete: %d points across %d collections in %.1f seconds (%.0f points/s)",
        total, len(collections), elapsed, total / max(elapsed, 0.1),
    )


if __name__ == "__main__":
    main()
