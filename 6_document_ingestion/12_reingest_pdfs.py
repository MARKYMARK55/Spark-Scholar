#!/usr/bin/env python3
"""
ingest/12_reingest_pdfs.py
==========================
Re-ingest PDF collections using the updated Docling pipeline.

For each subdirectory under RAG/pdfs/, this script:
1. Snapshots the existing Qdrant collection (if it exists)
2. Deletes and recreates the collection
3. Re-ingests all PDFs via ingest/05_ingest_pdfs.py

Usage
-----
    # Re-ingest all collections
    python ingest/12_reingest_pdfs.py

    # Re-ingest specific collections
    python ingest/12_reingest_pdfs.py --collections ML Deep-Learning

    # Dry run — show what would be re-ingested
    python ingest/12_reingest_pdfs.py --dry-run

    # Skip snapshot (faster, but no rollback safety net)
    python ingest/12_reingest_pdfs.py --no-snapshot
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../env/.env"), override=False)

logger = logging.getLogger(__name__)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
RAG_PDFS_DIR = Path(__file__).resolve().parent.parent / "RAG" / "pdfs"
SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "RAG" / "snapshots"
INGEST_SCRIPT = Path(__file__).resolve().parent / "05_ingest_pdfs.py"


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)


def collection_exists(client: QdrantClient, name: str) -> bool:
    try:
        client.get_collection(name)
        return True
    except (UnexpectedResponse, Exception):
        return False


def snapshot_collection(client: QdrantClient, name: str) -> str | None:
    """Create a snapshot of the collection. Returns snapshot name or None."""
    try:
        snap = client.create_snapshot(collection_name=name)
        snap_name = snap.name if hasattr(snap, "name") else str(snap)
        logger.info(f"  Snapshot created: {snap_name}")
        return snap_name
    except Exception as e:
        logger.warning(f"  Snapshot failed for {name}: {e}")
        return None


def delete_collection(client: QdrantClient, name: str) -> bool:
    """Delete a collection. Returns True on success."""
    try:
        client.delete_collection(collection_name=name)
        logger.info(f"  Deleted collection: {name}")
        return True
    except Exception as e:
        logger.warning(f"  Delete failed for {name}: {e}")
        return False


def reingest_collection(collection_name: str, input_dir: Path, verbose: bool = False) -> bool:
    """Run 05_ingest_pdfs.py for a single collection. Returns True on success."""
    cmd = [
        sys.executable,
        str(INGEST_SCRIPT),
        "--input-dir", str(input_dir),
        "--collection", collection_name,
    ]
    if verbose:
        cmd.append("--verbose")

    logger.info(f"  Running: {' '.join(cmd)}")
    t0 = time.time()

    result = subprocess.run(cmd, capture_output=True, text=True)

    elapsed = time.time() - t0
    if result.returncode == 0:
        logger.info(f"  Ingestion complete in {elapsed:.1f}s")
        # Print last few lines of output (summary)
        for line in result.stdout.strip().split("\n")[-5:]:
            logger.info(f"    {line}")
        return True
    else:
        logger.error(f"  Ingestion FAILED (exit {result.returncode}) after {elapsed:.1f}s")
        if result.stderr:
            for line in result.stderr.strip().split("\n")[-10:]:
                logger.error(f"    {line}")
        return False


def discover_collections() -> list[tuple[str, Path]]:
    """Find all subdirectories under RAG/pdfs/ that contain PDFs."""
    collections = []
    if not RAG_PDFS_DIR.exists():
        logger.error(f"PDF directory not found: {RAG_PDFS_DIR}")
        return collections

    for subdir in sorted(RAG_PDFS_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        pdfs = list(subdir.glob("*.pdf")) + list(subdir.glob("*.PDF"))
        if pdfs:
            collections.append((subdir.name, subdir))
            logger.debug(f"  Found {len(pdfs)} PDFs in {subdir.name}")
        else:
            logger.debug(f"  Skipping {subdir.name} (no PDFs)")

    return collections


def main():
    parser = argparse.ArgumentParser(
        description="Re-ingest PDF collections with updated Docling pipeline"
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        default=None,
        help="Specific collection names to re-ingest (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be re-ingested without doing it",
    )
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Skip snapshotting before deletion (faster but no rollback)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    all_collections = discover_collections()
    if not all_collections:
        logger.error("No PDF collections found under %s", RAG_PDFS_DIR)
        sys.exit(1)

    # Filter if specific collections requested
    if args.collections:
        requested = set(args.collections)
        available = {name for name, _ in all_collections}
        missing = requested - available
        if missing:
            logger.warning("Collections not found on disk: %s", ", ".join(sorted(missing)))
        all_collections = [(n, p) for n, p in all_collections if n in requested]

    logger.info("Collections to re-ingest (%d):", len(all_collections))
    for name, path in all_collections:
        pdf_count = len(list(path.glob("*.pdf")) + list(path.glob("*.PDF")))
        logger.info(f"  {name}: {pdf_count} PDFs in {path}")

    if args.dry_run:
        logger.info("Dry run — exiting without changes.")
        return

    client = get_client()
    results = {"success": [], "failed": [], "skipped": []}

    for name, path in all_collections:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {name}")
        logger.info(f"{'='*60}")

        # Step 1: Snapshot
        if not args.no_snapshot and collection_exists(client, name):
            snapshot_collection(client, name)

        # Step 2: Delete existing collection
        if collection_exists(client, name):
            if not delete_collection(client, name):
                logger.error(f"  Skipping {name} — could not delete existing collection")
                results["skipped"].append(name)
                continue
            # Brief pause for Qdrant to clean up
            time.sleep(1)

        # Step 3: Re-ingest
        success = reingest_collection(name, path, verbose=args.verbose)
        if success:
            results["success"].append(name)
        else:
            results["failed"].append(name)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("RE-INGESTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Success: {len(results['success'])} — {', '.join(results['success']) or 'none'}")
    logger.info(f"  Failed:  {len(results['failed'])} — {', '.join(results['failed']) or 'none'}")
    logger.info(f"  Skipped: {len(results['skipped'])} — {', '.join(results['skipped']) or 'none'}")

    if results["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
