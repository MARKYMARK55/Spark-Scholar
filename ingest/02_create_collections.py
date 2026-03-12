#!/usr/bin/env python3
"""
ingest/02_create_collections.py
================================
Create all 14 Qdrant collections with the correct dense + sparse vector configs.

Each collection gets:
  - dense_embedding: 1024-dim cosine vectors with HNSW (m=16, ef_construct=100)
  - sparse_text:     SPLADE-style sparse vectors (indices + values)
  - on_disk_payload: True  (payload stored on disk to save GPU VRAM)

Collections created:
  arXiv (catch-all)
  arxiv-cs-ml-ai
  arxiv-condmat
  arxiv-astro
  arxiv-hep
  arxiv-math-applied
  arxiv-math-phys
  arxiv-math-pure
  arxiv-misc
  arxiv-nucl-nlin-physother
  arxiv-qbio-qfin-econ
  arxiv-quantph-grqc
  arxiv-stat-eess
  arxiv-cs-systems-theory

Usage
-----
    python ingest/02_create_collections.py
    python ingest/02_create_collections.py --recreate  # Drop and recreate
    python ingest/02_create_collections.py --host qdrant --port 6333
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    VectorsConfig,
)

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../env/.env"), override=False)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Collection names
# ---------------------------------------------------------------------------

COLLECTIONS = [
    "arXiv",
    "arxiv-cs-ml-ai",
    "arxiv-condmat",
    "arxiv-astro",
    "arxiv-hep",
    "arxiv-math-applied",
    "arxiv-math-phys",
    "arxiv-math-pure",
    "arxiv-misc",
    "arxiv-nucl-nlin-physother",
    "arxiv-qbio-qfin-econ",
    "arxiv-quantph-grqc",
    "arxiv-stat-eess",
    "arxiv-cs-systems-theory",
]

# Vector configuration constants
DENSE_DIM = 1024
HNSW_M = 16
HNSW_EF_CONSTRUCT = 100


def create_collection(
    client: QdrantClient,
    name: str,
    recreate: bool = False,
) -> bool:
    """
    Create a single Qdrant collection with dense + sparse vector configs.

    Parameters
    ----------
    client : QdrantClient
    name : str
        Collection name.
    recreate : bool
        If True, drop existing collection before creating.

    Returns
    -------
    bool
        True if created, False if already existed (and recreate=False).
    """
    # Check if already exists
    existing = [c.name for c in client.get_collections().collections]

    if name in existing:
        if recreate:
            logger.info("Dropping existing collection: %s", name)
            client.delete_collection(name)
        else:
            logger.info("Collection already exists (skipping): %s", name)
            return False

    logger.info("Creating collection: %s", name)

    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense_embedding": VectorParams(
                size=DENSE_DIM,
                distance=Distance.COSINE,
                on_disk=False,  # Keep vectors in RAM for fast ANN
                hnsw_config=HnswConfigDiff(
                    m=HNSW_M,
                    ef_construct=HNSW_EF_CONSTRUCT,
                    full_scan_threshold=10000,
                    max_indexing_threads=0,  # Use all available cores
                    on_disk=False,
                ),
            )
        },
        sparse_vectors_config={
            "sparse_text": SparseVectorParams(
                index=SparseIndexParams(
                    on_disk=False,  # Keep sparse index in RAM
                    full_scan_threshold=5000,
                )
            )
        },
        on_disk_payload=True,  # Store payload (metadata) on disk
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=20000,  # Build HNSW index after 20k vectors
            memmap_threshold=50000,    # Use memmap for large segments
        ),
    )

    logger.info("Created collection: %s (dense_dim=%d, hnsw_m=%d)", name, DENSE_DIM, HNSW_M)
    return True


def create_all_collections(
    qdrant_url: str,
    api_key: str | None = None,
    recreate: bool = False,
) -> dict[str, bool]:
    """
    Create all 14 arXiv collections.

    Returns
    -------
    dict[str, bool]
        Maps collection name → True (created) / False (already existed)
    """
    client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=30)

    results: dict[str, bool] = {}
    for name in COLLECTIONS:
        try:
            created = create_collection(client, name, recreate=recreate)
            results[name] = created
        except Exception as exc:
            logger.error("Failed to create collection %s: %s", name, exc)
            results[name] = False

    return results


def verify_collections(qdrant_url: str, api_key: str | None = None) -> None:
    """Print a summary of all collection info."""
    client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=30)

    existing = client.get_collections().collections
    existing_names = {c.name for c in existing}

    print("\n=== Collection Status ===")
    print(f"{'Collection':<35} {'Exists':<8} {'Points':<12} {'Status'}")
    print("-" * 70)

    for name in COLLECTIONS:
        if name in existing_names:
            info = client.get_collection(name)
            points = info.points_count or 0
            status = info.status.value if hasattr(info.status, "value") else str(info.status)
            print(f"{name:<35} {'YES':<8} {points:<12} {status}")
        else:
            print(f"{name:<35} {'NO':<8} {'—':<12} missing")

    missing = [n for n in COLLECTIONS if n not in existing_names]
    if missing:
        print(f"\nWARNING: {len(missing)} collection(s) missing: {missing}")
    else:
        print(f"\nAll {len(COLLECTIONS)} collections present.")


def main():
    parser = argparse.ArgumentParser(
        description="Create Qdrant collections for arXiv RAG"
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Qdrant host (overrides QDRANT_URL env var)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6333,
        help="Qdrant port (default: 6333)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate existing collections (WARNING: deletes all data)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify collections exist, don't create",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    if args.host:
        qdrant_url = f"http://{args.host}:{args.port}"
    api_key = os.environ.get("QDRANT_API_KEY") or None

    if args.recreate:
        print("WARNING: --recreate will delete all data in existing collections!")
        confirm = input("Type 'yes' to continue: ").strip()
        if confirm != "yes":
            print("Aborted.")
            sys.exit(0)

    if args.verify_only:
        verify_collections(qdrant_url, api_key)
        return

    print(f"Connecting to Qdrant at {qdrant_url}")
    results = create_all_collections(qdrant_url, api_key, recreate=args.recreate)

    created = sum(1 for v in results.values() if v)
    skipped = len(results) - created

    print(f"\nDone: {created} created, {skipped} already existed.")
    verify_collections(qdrant_url, api_key)


if __name__ == "__main__":
    main()
