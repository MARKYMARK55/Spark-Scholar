#!/usr/bin/env python3
"""Export arXiv split collections from Qdrant → Parquet → HuggingFace.

Layout:
  Top-level: arxiv-cs-ml-ai.parquet, arxiv-astro.parquet, etc.

For each collection:
  1. Scroll all points into memory
  2. Write {collection}.parquet with zstd compression
  3. Upload to HF immediately
  4. Free memory, move to next

Usage:
    python 11_export_parquet.py
"""

import logging
import os
import subprocess
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from qdrant_client import QdrantClient
except ImportError:
    print("pip install qdrant-client")
    sys.exit(1)

# ── Config ──────────────────────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "simple-api-key")
HF_REPO = "MARKYMARK55/arxiv-bge-m3-embeddings"
HF_CLI = "/home/mark/venv/bin/huggingface-cli"
OUTPUT_DIR = os.path.expanduser("~/RAG/parquet_export")
SCROLL_BATCH = 256
DENSE_VECTOR_NAME = "dense_embedding"
SPARSE_VECTOR_NAME = "sparse_text"

COLLECTIONS = [
    "arxiv-math-phys",
    "arxiv-qbio-qfin-econ",
    "arxiv-misc",
    "arxiv-cs-nlp-ir",
    "arxiv-stat-eess",
    "arxiv-cs-cv",
    "arxiv-cs-ml-ai",
    "arxiv-quantph-grqc",
    "arxiv-cs-systems-theory",
    "arxiv-nucl-nlin-physother",
    "arxiv-math-pure",
    "arxiv-math-applied",
    "arxiv-hep",
    "arxiv-astro",
    "arxiv-condmat",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

SCHEMA = pa.schema([
    ("arxiv_id", pa.string()),
    ("title", pa.string()),
    ("abstract", pa.string()),
    ("categories", pa.list_(pa.string())),
    ("authors", pa.list_(pa.string())),
    ("first_created", pa.string()),
    ("last_updated", pa.string()),
    ("doi", pa.string()),
    ("pdf_url", pa.string()),
    ("collection", pa.string()),
    ("dense_embedding", pa.list_(pa.float32(), 1024)),
    ("sparse_indices", pa.list_(pa.uint32())),
    ("sparse_values", pa.list_(pa.float32())),
])


def extract_record(point, collection: str) -> dict | None:
    payload = point.payload or {}
    vectors = point.vector or {}

    dense = vectors.get(DENSE_VECTOR_NAME, [])
    if not isinstance(dense, list):
        dense = list(dense) if dense else []
    if len(dense) != 1024:
        return None

    sparse = vectors.get(SPARSE_VECTOR_NAME)
    if sparse and hasattr(sparse, "indices"):
        sparse_indices = list(sparse.indices)
        sparse_values = list(sparse.values)
    elif isinstance(sparse, dict):
        sparse_indices = sparse.get("indices", [])
        sparse_values = sparse.get("values", [])
    else:
        sparse_indices = []
        sparse_values = []

    cats = payload.get("categories", [])
    if isinstance(cats, str):
        cats = cats.split()

    authors = payload.get("authors", [])
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(",")]

    return {
        "arxiv_id": payload.get("original_arxiv_id", ""),
        "title": payload.get("title", ""),
        "abstract": payload.get("abstract", ""),
        "categories": cats,
        "authors": authors,
        "first_created": payload.get("first_created", ""),
        "last_updated": payload.get("last_updated", ""),
        "doi": payload.get("doi") or "",
        "pdf_url": payload.get("pdf_url", ""),
        "collection": collection,
        "dense_embedding": dense,
        "sparse_indices": [int(i) for i in sparse_indices],
        "sparse_values": [float(v) for v in sparse_values],
    }


def upload_to_hf(local_path: str, remote_path: str) -> bool:
    result = subprocess.run(
        [HF_CLI, "upload", HF_REPO, local_path, remote_path,
         "--repo-type", "dataset", "--quiet"],
        capture_output=True, text=True, timeout=3600,
    )
    return result.returncode == 0


def process_collection(client: QdrantClient, collection: str) -> tuple[int, float, bool]:
    """Scroll → Parquet → Upload → Cleanup."""
    info = client.get_collection(collection)
    total = info.points_count

    # ── Scroll all points ──
    records = []
    processed = 0
    skipped = 0
    t0 = time.time()
    next_offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_BATCH,
            offset=next_offset,
            with_payload=True,
            with_vectors=True,
        )

        for point in points:
            record = extract_record(point, collection)
            if record is None:
                skipped += 1
                continue
            records.append(record)

        processed += len(points)

        if processed % (SCROLL_BATCH * 20) == 0 or next_offset is None:
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            pct = processed * 100 / total if total > 0 else 0
            eta_s = (total - processed) / rate if rate > 0 else 0
            log.info(
                f"  scroll: [{pct:5.1f}%] {processed:,}/{total:,} "
                f"| {rate:.0f} pts/s | ETA {eta_s:.0f}s"
            )

        if next_offset is None:
            break

    # ── Write Parquet (flat: collection.parquet) ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = f"{collection}.parquet"
    path = os.path.join(OUTPUT_DIR, filename)

    columns = {f.name: [] for f in SCHEMA}
    for r in records:
        for key in columns:
            columns[key].append(r.get(key))

    arrays = [pa.array(columns[f.name], type=f.type) for f in SCHEMA]
    table = pa.Table.from_arrays(arrays, schema=SCHEMA)
    pq.write_table(table, path, compression="zstd")

    size_mb = os.path.getsize(path) / 1024 / 1024
    rows = len(records)
    log.info(f"  wrote: {filename} ({rows:,} rows, {size_mb:.0f} MB)")

    del records, columns, arrays, table

    # ── Upload to HF (top-level) ──
    log.info(f"  uploading {filename} to HF...")
    t_up = time.time()
    ok = upload_to_hf(path, filename)
    upload_time = time.time() - t_up

    if ok:
        log.info(f"  uploaded: {upload_time:.0f}s")
        os.remove(path)
    else:
        log.info(f"  upload FAILED — file saved at {path}")

    return rows, size_mb, ok


def main():
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)

    log.info("=" * 60)
    log.info("Export & Upload: Qdrant splits → Parquet → HuggingFace")
    log.info(f"  Repo:        {HF_REPO}")
    log.info(f"  Layout:      {'{collection}.parquet'} (flat, top-level)")
    log.info(f"  Collections: {len(COLLECTIONS)}")
    log.info("=" * 60)

    grand_rows = 0
    grand_mb = 0
    t_start = time.time()

    for i, col in enumerate(COLLECTIONS, 1):
        log.info("")
        log.info(f"[{i}/{len(COLLECTIONS)}] {col}")
        rows, size_mb, ok = process_collection(client, col)
        grand_rows += rows
        grand_mb += size_mb
        status = "✓" if ok else "✗ (saved locally)"
        elapsed = time.time() - t_start
        log.info(
            f"  {status} {rows:,} rows | {size_mb:.0f} MB | "
            f"cumulative: {grand_rows:,} rows, {grand_mb/1024:.1f} GB, {elapsed/60:.0f}m"
        )

    log.info("")
    log.info("=" * 60)
    log.info(f"Done: {grand_rows:,} rows | {grand_mb/1024:.1f} GB | {(time.time()-t_start)/60:.0f} minutes")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
