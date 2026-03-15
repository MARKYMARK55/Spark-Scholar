#!/usr/bin/env python3
"""
ingest/09_add_sparse_to_monolith.py
====================================
Add sparse_text vectors to existing points in the arXiv monolith collection.

Scrolls the monolith, reads title + abstract from each point's payload,
generates BGE-M3 sparse embeddings, and updates the points in place using
Qdrant's update_vectors() — preserving the existing dense_embedding.

Features
--------
- Resume support: tracks processed point IDs in a progress file
- Configurable batch size (default 64 — sparse is memory-intensive)
- tqdm progress bar with throughput stats
- Skips points that already have sparse vectors (--skip-existing)

Usage
-----
    python ingest/09_add_sparse_to_monolith.py
    python ingest/09_add_sparse_to_monolith.py --batch-size 32
    python ingest/09_add_sparse_to_monolith.py --collection arXiv --max-records 1000
    python ingest/09_add_sparse_to_monolith.py --skip-existing

Requirements
------------
    pip install qdrant-client httpx numpy tqdm python-dotenv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointVectors,
    SparseVector as QdrantSparseVector,
)
# tqdm removed — using simple periodic log lines for nohup compatibility

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../env/.env"), override=False)

# Pipeline package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.embeddings import encode_sparse

logger = logging.getLogger(__name__)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
DEFAULT_COLLECTION = "arXiv"
DEFAULT_BATCH_SIZE = 64
SPARSE_VECTOR_NAME = "sparse_text"
SCROLL_BATCH_SIZE = 256  # How many points to fetch per scroll call

PROGRESS_DIR = os.path.expanduser("~/RAG/arxiv")


def _progress_file(collection: str) -> str:
    """Return the path to the progress file for a given collection."""
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    return os.path.join(PROGRESS_DIR, f"{collection}_sparse_progress.txt")


def _load_progress(progress_file: str) -> set[str]:
    """Load already-processed point IDs from progress file."""
    if not os.path.exists(progress_file):
        return set()
    with open(progress_file, "r") as f:
        return {line.strip() for line in f if line.strip()}


def _save_progress(progress_file: str, point_ids: list[str]) -> None:
    """Append a batch of processed point IDs to progress file."""
    with open(progress_file, "a") as f:
        for pid in point_ids:
            f.write(str(pid) + "\n")


def _update_sparse_batch(
    client: QdrantClient,
    collection: str,
    point_ids: list,
    sparse_vecs: list,
) -> None:
    """Update existing points with sparse vectors using update_vectors()."""
    points = []
    for pid, sv in zip(point_ids, sparse_vecs):
        points.append(
            PointVectors(
                id=pid,
                vector={
                    SPARSE_VECTOR_NAME: QdrantSparseVector(
                        indices=sv.indices,
                        values=sv.values,
                    )
                },
            )
        )

    client.update_vectors(
        collection_name=collection,
        points=points,
    )


def add_sparse_to_collection(
    collection: str = DEFAULT_COLLECTION,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_records: int | None = None,
    skip_existing: bool = False,
) -> int:
    """
    Add sparse vectors to all points in a collection.

    Scrolls the collection, extracts title + abstract from payload,
    generates sparse embeddings, and updates points in place.

    Parameters
    ----------
    collection : str
        Qdrant collection name (default: "arXiv").
    batch_size : int
        Number of texts to sparse-embed per batch (default: 64).
    max_records : int, optional
        Stop after this many records (for testing).
    skip_existing : bool
        If True, check each point for existing sparse vector and skip.

    Returns
    -------
    int
        Number of points updated.
    """
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)

    # Get collection info
    info = client.get_collection(collection)
    total_points = info.points_count or 0
    logger.info("Collection %s: %d points", collection, total_points)

    if total_points == 0:
        logger.warning("Collection %s is empty — nothing to do", collection)
        return 0

    # Progress tracking
    pfile = _progress_file(collection)
    already_done = _load_progress(pfile)
    logger.info("Resuming: %d points already processed", len(already_done))

    # Buffers for batching
    batch_ids: list = []
    batch_texts: list[str] = []

    updated = 0
    skipped_progress = 0
    skipped_no_text = 0
    skipped_has_sparse = 0
    errors = 0
    start_time = time.time()

    def _flush_batch() -> int:
        """Encode and update the current batch. Returns count updated."""
        nonlocal errors
        if not batch_ids:
            return 0

        try:
            sparse_vecs = encode_sparse(batch_texts)
            _update_sparse_batch(client, collection, batch_ids, sparse_vecs)
            _save_progress(pfile, [str(pid) for pid in batch_ids])
            count = len(batch_ids)
            batch_ids.clear()
            batch_texts.clear()
            return count
        except Exception as exc:
            logger.error("Failed to update batch: %s", exc)
            errors += 1
            batch_ids.clear()
            batch_texts.clear()
            return 0

    # Progress reporting — single line every LOG_INTERVAL seconds
    LOG_INTERVAL = 30  # seconds between progress lines
    last_log_time = start_time
    next_offset = None

    def _log_progress(force: bool = False) -> None:
        """Print a single progress line if enough time has passed."""
        nonlocal last_log_time
        now = time.time()
        if not force and (now - last_log_time) < LOG_INTERVAL:
            return
        last_log_time = now
        elapsed = now - start_time
        throughput = updated / elapsed if elapsed > 0 else 0
        pct = (updated / total_points) * 100 if total_points else 0
        eta_s = (total_points - updated) / throughput if throughput > 0 else 0
        eta_h = eta_s / 3600
        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"{updated:,}/{total_points:,} ({pct:.1f}%) | "
            f"{throughput:.1f} pts/s | "
            f"ETA {eta_h:.1f}h | "
            f"skip={skipped_progress} err={errors}",
            flush=True,
        )

    while True:
        # Scroll a page of points
        # Only fetch vectors if we need to check for existing sparse
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_BATCH_SIZE,
            offset=next_offset,
            with_payload=True,
            with_vectors=skip_existing,  # Only fetch vectors if checking
        )

        if not points:
            break

        for point in points:
            # Check if we've hit the limit
            if max_records and updated >= max_records:
                break

            point_id_str = str(point.id)

            # Skip if already processed (from progress file)
            if point_id_str in already_done:
                skipped_progress += 1
                continue

            # Skip if point already has sparse vector
            if skip_existing and point.vector:
                vectors = point.vector
                if isinstance(vectors, dict) and SPARSE_VECTOR_NAME in vectors:
                    skipped_has_sparse += 1
                    continue

            # Extract text from payload
            payload = point.payload or {}
            title = payload.get("title", "")
            abstract = payload.get("abstract", "")

            if not abstract and not title:
                skipped_no_text += 1
                continue

            embed_text = f"{title}\n{abstract}" if title else abstract

            batch_ids.append(point.id)
            batch_texts.append(embed_text)

            # Flush when batch is full
            if len(batch_ids) >= batch_size:
                count = _flush_batch()
                updated += count
                _log_progress()

        # Check if we've hit the limit
        if max_records and updated >= max_records:
            break

        if next_offset is None:
            break

    # Flush remaining
    if batch_ids:
        count = _flush_batch()
        updated += count

    _log_progress(force=True)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Sparse vector update complete for %s:", collection)
    logger.info("  Updated:            %d points", updated)
    logger.info("  Skipped (progress): %d", skipped_progress)
    logger.info("  Skipped (no text):  %d", skipped_no_text)
    logger.info("  Skipped (has sparse): %d", skipped_has_sparse)
    logger.info("  Errors:             %d", errors)
    logger.info("  Time:               %.1f minutes", elapsed / 60)
    if updated > 0:
        logger.info("  Throughput:         %.1f pts/s", updated / elapsed)
    logger.info("=" * 60)

    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Add sparse vectors to existing Qdrant collection points"
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help=f"Collection to update (default: {DEFAULT_COLLECTION})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Sparse embedding batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Stop after this many records (for testing)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip points that already have a sparse vector",
    )
    parser.add_argument(
        "--reset-progress",
        action="store_true",
        help="Clear progress file and start from scratch",
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    # Silence noisy HTTP loggers so they don't pollute progress output
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    if args.reset_progress:
        pfile = _progress_file(args.collection)
        if os.path.exists(pfile):
            os.remove(pfile)
            print(f"Cleared progress file: {pfile}")

    updated = add_sparse_to_collection(
        collection=args.collection,
        batch_size=args.batch_size,
        max_records=args.max_records,
        skip_existing=args.skip_existing,
    )

    print(f"\nUpdated {updated} points in '{args.collection}' with sparse vectors.")


if __name__ == "__main__":
    main()
