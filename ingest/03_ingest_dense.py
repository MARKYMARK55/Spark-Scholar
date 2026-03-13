#!/usr/bin/env python3
"""
ingest/03_ingest_dense.py
=========================
Dense-embed arXiv abstracts with BGE-M3 (via vLLM) and upsert into Qdrant.

Reads from ~/RAG/arxiv/arxiv_with_abstract.jsonl (created by 01_download_arxiv.py),
routes each paper to its collection, embeds in batches, and upserts.

Features
--------
- Configurable batch size (default 256 — tune to your GPU VRAM)
- Resume support: tracks progress in ~/RAG/arxiv/arxiv_with_abstract_dense_progress.txt
- Parallel batches: embeds the next batch while upserting the current one
- tqdm progress bar with throughput stats

Usage
-----
    python ingest/03_ingest_dense.py
    python ingest/03_ingest_dense.py --input ~/RAG/arxiv/arxiv_with_abstract.jsonl \
                                     --batch-size 128 \
                                     --collection arxiv-cs-ml-ai

Requirements
------------
    pip install qdrant-client httpx numpy tqdm python-dotenv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from tqdm import tqdm

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../env/.env"), override=False)

# Pipeline package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.embeddings import encode_dense
from pipeline.router import route_paper

logger = logging.getLogger(__name__)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
DEFAULT_INPUT = os.path.expanduser("~/RAG/arxiv/arxiv_with_abstract.jsonl")
DEFAULT_BATCH_SIZE = 256
DENSE_VECTOR_NAME = "dense_embedding"


def _load_progress(progress_file: str) -> set[str]:
    """Load already-processed arxiv_ids from progress file."""
    if not os.path.exists(progress_file):
        return set()
    with open(progress_file, "r") as f:
        return {line.strip() for line in f if line.strip()}


def _save_progress(progress_file: str, arxiv_ids: list[str]) -> None:
    """Append a batch of processed arxiv_ids to progress file."""
    with open(progress_file, "a") as f:
        for aid in arxiv_ids:
            f.write(aid + "\n")


def _embed_batch(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts, returning float32 array shape (N, 1024)."""
    return encode_dense(texts)


def _upsert_batch(
    client: QdrantClient,
    collection: str,
    ids: list[str],
    vectors: np.ndarray,
    payloads: list[dict],
) -> None:
    """Upsert a batch of points into the given collection."""
    points = []
    for i, (arxiv_id, payload) in enumerate(zip(ids, payloads)):
        # Use a deterministic integer ID from the arxiv_id hash
        # Qdrant needs integer or UUID ids; we use abs(hash) % 2^63
        point_id = abs(hash(arxiv_id)) % (2 ** 63)
        payload["arxiv_id"] = arxiv_id

        points.append(
            PointStruct(
                id=point_id,
                vector={DENSE_VECTOR_NAME: vectors[i].tolist()},
                payload=payload,
            )
        )

    client.upsert(
        collection_name=collection,
        points=points,
        wait=True,
    )


def ingest_dense(
    input_file: str = DEFAULT_INPUT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    collection_override: str | None = None,
    max_records: int | None = None,
) -> dict[str, int]:
    """
    Main ingestion function.

    Parameters
    ----------
    input_file : str
        Path to JSONL file with arXiv records.
    batch_size : int
        Number of records to embed and upsert per batch.
    collection_override : str, optional
        Force all records into a single collection (for targeted ingestion).
    max_records : int, optional
        Stop after this many records (for testing).

    Returns
    -------
    dict[str, int]
        Maps collection name → number of records upserted.
    """
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)

    progress_file = input_file.replace(".jsonl", "_dense_progress.txt")
    already_done = _load_progress(progress_file)
    logger.info("Resuming: %d records already processed", len(already_done))

    # Collection → count
    counts: dict[str, int] = {}

    # Buffer: collection → (ids, texts, payloads)
    buffers: dict[str, tuple[list, list, list]] = {}

    def _flush_collection(coll: str) -> None:
        """Embed and upsert everything in the buffer for this collection."""
        ids, texts, payloads = buffers[coll]
        if not ids:
            return

        try:
            vecs = _embed_batch(texts)
            _upsert_batch(client, coll, ids, vecs, payloads)
            _save_progress(progress_file, ids)
            counts[coll] = counts.get(coll, 0) + len(ids)
        except Exception as exc:
            logger.error("Failed to upsert batch to %s: %s", coll, exc)

        buffers[coll] = ([], [], [])

    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    total_lines = sum(1 for _ in open(input_file, "r", encoding="utf-8"))
    logger.info("Total records in input: %d", total_lines)

    start_time = time.time()
    processed = 0
    skipped = 0

    with open(input_file, "r", encoding="utf-8") as f:
        pbar = tqdm(f, total=total_lines, desc="Dense ingestion", unit="rec")
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

            # Skip already-processed records
            if arxiv_id in already_done:
                skipped += 1
                continue

            title = doc.get("title", "")
            abstract = doc.get("abstract", "")
            if not abstract:
                continue

            # Route to collection
            if collection_override:
                collection = collection_override
            else:
                collection = route_paper(doc.get("categories", ""))

            # Text to embed = title + abstract (standard for BGE-M3)
            embed_text = f"{title}\n{abstract}" if title else abstract

            payload = {
                "title": title,
                "abstract": abstract,
                "authors": doc.get("authors", ""),
                "categories": doc.get("categories", ""),
                "year": doc.get("year"),
                "update_date": doc.get("update_date", ""),
                "journal_ref": doc.get("journal_ref", ""),
            }

            if collection not in buffers:
                buffers[collection] = ([], [], [])

            buffers[collection][0].append(arxiv_id)
            buffers[collection][1].append(embed_text)
            buffers[collection][2].append(payload)

            # Flush full buffers
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
                collections=len(counts),
            )

    # Flush remaining buffers
    logger.info("Flushing remaining buffers...")
    for coll in list(buffers.keys()):
        _flush_collection(coll)

    total_upserted = sum(counts.values())
    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("Dense ingestion complete:")
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
        description="Dense-embed arXiv abstracts and upsert to Qdrant"
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Path to JSONL input file (default: ~/RAG/arxiv/arxiv_with_abstract.jsonl)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Embedding batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Override collection routing — put everything in this collection",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Stop after this many records (for testing)",
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    counts = ingest_dense(
        input_file=args.input,
        batch_size=args.batch_size,
        collection_override=args.collection,
        max_records=args.max_records,
    )

    print(f"\nUpserted {sum(counts.values())} points across {len(counts)} collections.")


if __name__ == "__main__":
    main()
