#!/usr/bin/env python3
"""
ingest/10_split_monolith.py
===========================
Split the arXiv monolith into subject collections.

Scrolls the monolith (which has dense_embedding + sparse_text + full payload),
routes each paper by its categories, and upserts into the appropriate split
collection — copying both vectors and all payload fields.

The split collections must already exist (run 02_create_collections.py --arxiv-only first).

Features
--------
- Copies both dense + sparse vectors in a single pass (no re-embedding needed)
- Resume support via progress file
- Batched upserts per collection for efficiency
- Progress logging every 30 seconds (nohup-friendly)

Usage
-----
    # First, create the split collections:
    python ingest/02_create_collections.py --arxiv-only

    # Then split:
    python ingest/10_split_monolith.py
    python ingest/10_split_monolith.py --batch-size 256 --max-records 1000
    python ingest/10_split_monolith.py --reset-progress

Requirements
------------
    pip install qdrant-client python-dotenv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import defaultdict

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    SparseVector as QdrantSparseVector,
)

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../env/.env"), override=False)

# Pipeline package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.router import route_paper

logger = logging.getLogger(__name__)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
MONOLITH = "arXiv"
DEFAULT_BATCH_SIZE = 256
SCROLL_BATCH_SIZE = 256
DENSE_VECTOR_NAME = "dense_embedding"
SPARSE_VECTOR_NAME = "sparse_text"

PROGRESS_DIR = os.path.expanduser("~/RAG/arxiv")


def _progress_file() -> str:
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    return os.path.join(PROGRESS_DIR, "split_monolith_progress.txt")


def _load_progress(pfile: str) -> set[str]:
    if not os.path.exists(pfile):
        return set()
    with open(pfile, "r") as f:
        return {line.strip() for line in f if line.strip()}


def _save_progress(pfile: str, point_ids: list[str]) -> None:
    with open(pfile, "a") as f:
        for pid in point_ids:
            f.write(str(pid) + "\n")


def split_monolith(
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_records: int | None = None,
) -> dict[str, int]:
    """
    Split the monolith into subject collections.

    Returns dict mapping collection name → number of points upserted.
    """
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)

    # Get monolith info
    info = client.get_collection(MONOLITH)
    total_points = info.points_count or 0
    logger.info("Monolith '%s': %d points", MONOLITH, total_points)

    if total_points == 0:
        logger.warning("Monolith is empty — nothing to do")
        return {}

    # Verify split collections exist
    existing = {c.name for c in client.get_collections().collections}
    required = ["arxiv-cs-ml-ai", "arxiv-astro", "arxiv-hep", "arxiv-condmat",
                 "arxiv-cs-systems-theory", "arxiv-math-applied", "arxiv-math-pure",
                 "arxiv-math-phys", "arxiv-misc", "arxiv-nucl-nlin-physother",
                 "arxiv-qbio-qfin-econ", "arxiv-quantph-grqc", "arxiv-stat-eess",
                 "arxiv-cs-cv", "arxiv-cs-nlp-ir"]
    missing = [c for c in required if c not in existing]
    if missing:
        logger.error(
            "Split collections not found: %s. Run 02_create_collections.py --arxiv-only first.",
            missing,
        )
        sys.exit(1)

    # Progress tracking
    pfile = _progress_file()
    already_done = _load_progress(pfile)
    logger.info("Resuming: %d points already processed", len(already_done))

    # Buffers: collection → list of PointStruct
    buffers: dict[str, list[PointStruct]] = defaultdict(list)
    # Track IDs per buffer for progress
    buffer_ids: dict[str, list[str]] = defaultdict(list)

    counts: dict[str, int] = defaultdict(int)
    skipped_progress = 0
    skipped_no_route = 0
    errors = 0
    updated = 0
    start_time = time.time()

    LOG_INTERVAL = 30
    last_log_time = start_time

    def _log_progress(force: bool = False) -> None:
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
        colls = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()) if v > 0)
        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"{updated:,}/{total_points:,} ({pct:.1f}%) | "
            f"{throughput:.1f} pts/s | "
            f"ETA {eta_h:.1f}h | "
            f"skip={skipped_progress} err={errors}",
            flush=True,
        )

    def _flush_collection(coll: str) -> None:
        nonlocal updated, errors
        points = buffers[coll]
        ids = buffer_ids[coll]
        if not points:
            return

        try:
            client.upsert(
                collection_name=coll,
                points=points,
                wait=True,
            )
            _save_progress(pfile, ids)
            counts[coll] += len(points)
            updated += len(points)
        except Exception as exc:
            logger.error("Failed to upsert %d points to %s: %s", len(points), coll, exc)
            errors += 1

        buffers[coll] = []
        buffer_ids[coll] = []

    # Main scroll loop
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=MONOLITH,
            limit=SCROLL_BATCH_SIZE,
            offset=next_offset,
            with_payload=True,
            with_vectors=True,
        )

        if not points:
            break

        for point in points:
            if max_records and updated >= max_records:
                break

            point_id_str = str(point.id)

            if point_id_str in already_done:
                skipped_progress += 1
                continue

            payload = point.payload or {}
            categories = payload.get("categories", [])

            # categories is a list in the monolith — join for route_paper()
            if isinstance(categories, list):
                cat_str = " ".join(categories)
            else:
                cat_str = str(categories)

            collection = route_paper(cat_str)

            # Skip papers that route back to "arXiv" (no matching category)
            # They stay in the monolith only
            if collection == "arXiv":
                skipped_no_route += 1
                _save_progress(pfile, [point_id_str])
                continue

            # Build the vector dict — include both dense and sparse
            vectors = point.vector or {}
            vector_dict = {}

            if isinstance(vectors, dict):
                if DENSE_VECTOR_NAME in vectors:
                    vector_dict[DENSE_VECTOR_NAME] = vectors[DENSE_VECTOR_NAME]
                if SPARSE_VECTOR_NAME in vectors:
                    sv = vectors[SPARSE_VECTOR_NAME]
                    if isinstance(sv, dict):
                        vector_dict[SPARSE_VECTOR_NAME] = QdrantSparseVector(
                            indices=sv["indices"],
                            values=sv["values"],
                        )
                    else:
                        # Already a SparseVector object
                        vector_dict[SPARSE_VECTOR_NAME] = sv

            if not vector_dict.get(DENSE_VECTOR_NAME):
                logger.warning("Point %s has no dense vector — skipping", point_id_str)
                continue

            new_point = PointStruct(
                id=point.id,
                vector=vector_dict,
                payload=payload,
            )

            buffers[collection].append(new_point)
            buffer_ids[collection].append(point_id_str)

            # Flush full buffers
            if len(buffers[collection]) >= batch_size:
                _flush_collection(collection)
                _log_progress()

        if max_records and updated >= max_records:
            break
        if next_offset is None:
            break

    # Flush remaining
    for coll in list(buffers.keys()):
        _flush_collection(coll)

    _log_progress(force=True)

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Split complete:")
    logger.info("  Total upserted:     %d points", updated)
    logger.info("  Skipped (progress): %d", skipped_progress)
    logger.info("  Skipped (no route): %d (remain in monolith only)", skipped_no_route)
    logger.info("  Errors:             %d", errors)
    logger.info("  Time:               %.1f minutes", elapsed / 60)
    if updated > 0:
        logger.info("  Throughput:         %.1f pts/s", updated / elapsed)
    logger.info("Per-collection counts:")
    for coll, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        logger.info("  %-35s %d", coll, cnt)
    logger.info("=" * 60)

    return dict(counts)


def main():
    parser = argparse.ArgumentParser(
        description="Split arXiv monolith into subject collections"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Upsert batch size per collection (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Stop after this many records (for testing)",
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
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    if args.reset_progress:
        pfile = _progress_file()
        if os.path.exists(pfile):
            os.remove(pfile)
            print(f"Cleared progress file: {pfile}")

    counts = split_monolith(
        batch_size=args.batch_size,
        max_records=args.max_records,
    )

    total = sum(counts.values())
    print(f"\nSplit {total:,} points across {len(counts)} collections.")


if __name__ == "__main__":
    main()
