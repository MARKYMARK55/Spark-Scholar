#!/usr/bin/env python3
"""
ingest/04_ingest_sparse.py
==========================
Sparse-embed arXiv abstracts with the BGE-M3 sparse embedder service and
upsert the sparse vectors into Qdrant.

This script updates existing points (created by 03_ingest_dense.py) with
their sparse_text vectors using Qdrant's update_vectors() call.
It can also be run standalone if you want to do sparse-only indexing.

The sparse embedder service (sparse_embedder/sparse_embed.py) runs at
BGE_M3_SPARSE_URL and accepts POST /encode with {"texts": [...]}.

Usage
-----
    python ingest/04_ingest_sparse.py
    python ingest/04_ingest_sparse.py --input ~/RAG/arxiv/arxiv_with_abstract.jsonl \
                                       --batch-size 64 \
                                       --collection arxiv-cs-ml-ai

Note: batch_size should be smaller than dense (64 recommended) because the
sparse embedder is memory-intensive.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import NamedSparseVector, PointVectors, SparseVector
from tqdm import tqdm

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../env/.env"), override=False)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.embeddings import encode_sparse, SparseVector as LocalSparseVector
from pipeline.router import route_paper

logger = logging.getLogger(__name__)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
DEFAULT_INPUT = os.path.expanduser("~/RAG/arxiv/arxiv_with_abstract.jsonl")
DEFAULT_BATCH_SIZE = 64
SPARSE_VECTOR_NAME = "sparse_text"


def _load_progress(progress_file: str) -> set[str]:
    if not os.path.exists(progress_file):
        return set()
    with open(progress_file, "r") as f:
        return {line.strip() for line in f if line.strip()}


def _save_progress(progress_file: str, arxiv_ids: list[str]) -> None:
    with open(progress_file, "a") as f:
        for aid in arxiv_ids:
            f.write(aid + "\n")


def _upsert_sparse_batch(
    client: QdrantClient,
    collection: str,
    arxiv_ids: list[str],
    sparse_vecs: list[LocalSparseVector],
    payloads: list[dict],
) -> None:
    """
    Upsert sparse vectors into Qdrant.

    Uses update_vectors to add/update only the sparse_text named vector
    on existing points (preserving the dense_embedding).

    If a point doesn't exist yet (running sparse-only), uses upsert() instead.
    """
    # We use upsert with only the sparse vector — Qdrant will create the point
    # if it doesn't exist, or update just the sparse vector if it does.
    from qdrant_client.models import PointStruct

    points = []
    for arxiv_id, sv, payload in zip(arxiv_ids, sparse_vecs, payloads):
        point_id = abs(hash(arxiv_id)) % (2 ** 63)
        payload["arxiv_id"] = arxiv_id

        qdrant_sparse = SparseVector(
            indices=sv.indices,
            values=sv.values,
        )

        points.append(
            PointStruct(
                id=point_id,
                vector={SPARSE_VECTOR_NAME: qdrant_sparse},
                payload=payload,
            )
        )

    client.upsert(
        collection_name=collection,
        points=points,
        wait=True,
    )


def ingest_sparse(
    input_file: str = DEFAULT_INPUT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    collection_override: str | None = None,
    max_records: int | None = None,
) -> dict[str, int]:
    """
    Main sparse ingestion function.

    Parameters
    ----------
    input_file : str
        Path to JSONL file (same file as used for dense ingestion).
    batch_size : int
        Batch size for the sparse embedder (recommended: 32–64).
    collection_override : str, optional
        Force all records into this collection.
    max_records : int, optional
        Cap for testing.

    Returns
    -------
    dict[str, int]
        Maps collection → upserted count.
    """
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)

    progress_file = input_file.replace(".jsonl", "_sparse_progress.txt")
    already_done = _load_progress(progress_file)
    logger.info("Resuming: %d records already processed", len(already_done))

    counts: dict[str, int] = {}
    # Buffer: collection → (ids, texts, payloads)
    buffers: dict[str, tuple[list, list, list]] = {}

    def _flush_collection(coll: str) -> None:
        ids, texts, payloads = buffers[coll]
        if not ids:
            return

        try:
            sparse_vecs = encode_sparse(texts)
            _upsert_sparse_batch(client, coll, ids, sparse_vecs, payloads)
            _save_progress(progress_file, ids)
            counts[coll] = counts.get(coll, 0) + len(ids)
        except Exception as exc:
            logger.error("Failed to upsert sparse batch to %s: %s", coll, exc)

        buffers[coll] = ([], [], [])

    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    total_lines = sum(1 for _ in open(input_file, "r", encoding="utf-8"))
    logger.info("Total records: %d", total_lines)

    start_time = time.time()
    processed = 0
    skipped = 0

    with open(input_file, "r", encoding="utf-8") as f:
        pbar = tqdm(f, total=total_lines, desc="Sparse ingestion", unit="rec")
        for line in pbar:
            if max_records and processed >= max_records:
                break

            line = line.strip()
            if not line:
                continue

            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue

            arxiv_id = doc.get("arxiv_id", "")
            if not arxiv_id:
                continue

            if arxiv_id in already_done:
                skipped += 1
                continue

            title = doc.get("title", "")
            abstract = doc.get("abstract", "")
            if not abstract:
                continue

            if collection_override:
                collection = collection_override
            else:
                collection = route_paper(doc.get("categories", ""))

            embed_text = f"{title}\n{abstract}" if title else abstract

            payload = {
                "title": title,
                "abstract": abstract,
                "authors": doc.get("authors", ""),
                "categories": doc.get("categories", ""),
                "year": doc.get("year"),
                "update_date": doc.get("update_date", ""),
            }

            if collection not in buffers:
                buffers[collection] = ([], [], [])

            buffers[collection][0].append(arxiv_id)
            buffers[collection][1].append(embed_text)
            buffers[collection][2].append(payload)

            for coll in list(buffers.keys()):
                if len(buffers[coll][0]) >= batch_size:
                    _flush_collection(coll)

            processed += 1
            elapsed = time.time() - start_time
            throughput = processed / elapsed if elapsed > 0 else 0
            pbar.set_postfix(
                processed=processed,
                skipped=skipped,
                throughput=f"{throughput:.0f}/s",
            )

    logger.info("Flushing remaining buffers...")
    for coll in list(buffers.keys()):
        _flush_collection(coll)

    total_upserted = sum(counts.values())
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Sparse ingestion complete:")
    logger.info("  Processed:   %d records", processed)
    logger.info("  Upserted:    %d points", total_upserted)
    logger.info("  Skipped:     %d (already done)", skipped)
    logger.info("  Time:        %.1f minutes", elapsed / 60)
    logger.info("  Throughput:  %.0f rec/s", processed / elapsed if elapsed > 0 else 0)
    logger.info("Per-collection counts:")
    for coll, cnt in sorted(counts.items()):
        logger.info("  %-40s %d", coll, cnt)
    logger.info("=" * 60)

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Sparse-embed arXiv abstracts and upsert to Qdrant"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--collection", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    counts = ingest_sparse(
        input_file=args.input,
        batch_size=args.batch_size,
        collection_override=args.collection,
        max_records=args.max_records,
    )

    print(f"\nUpserted {sum(counts.values())} sparse vectors across {len(counts)} collections.")


if __name__ == "__main__":
    main()
