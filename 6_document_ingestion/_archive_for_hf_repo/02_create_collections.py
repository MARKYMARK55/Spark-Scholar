#!/usr/bin/env python3
"""
ingest/02_create_collections.py
================================
Create all Qdrant collections — arXiv subject collections + tech documentation.

Each collection gets:
  - dense_embedding: 1024-dim cosine vectors with HNSW (m=16, ef_construct=100)
  - sparse_text:     SPLADE-style sparse vectors (indices + values)
  - on_disk_payload: True  (payload stored on disk to save GPU VRAM)

arXiv collections (16) — populated by ingest/03_ingest_dense.py + 04_ingest_sparse.py:
  arXiv (catch-all)           arxiv-cs-ml-ai            arxiv-cs-cv
  arxiv-cs-nlp-ir             arxiv-condmat              arxiv-astro
  arxiv-hep                   arxiv-math-applied         arxiv-math-phys
  arxiv-math-pure             arxiv-misc                 arxiv-nucl-nlin-physother
  arxiv-qbio-qfin-econ        arxiv-quantph-grqc         arxiv-stat-eess
  arxiv-cs-systems-theory

Documentation collections (8) — populated by ingest/05_ingest_pdfs.py + 07_ingest_html_docs.py:
  docs-python        Python stdlib, NumPy, Pandas, FastAPI, LangChain, etc.
  docs-rust          The Rust Book, std lib, Cargo docs, Tokio, etc.
  docs-javascript    MDN Web Docs, Node.js, TypeScript, React, etc.
  docs-docker        Docker Engine, Compose, Kubernetes, container ecosystem
  docs-anthropic     Claude API, Computer Use (CUA), Anthropic platform docs
  docs-applescript   AppleScript Language Guide, macOS scripting additions
  docs-devops        GitHub Actions, Terraform, CI/CD, observability tools
  docs-web           HTML5, CSS3, Web APIs, browser internals (MDN non-JS)

Usage
-----
    python ingest/02_create_collections.py                    # create all
    python ingest/02_create_collections.py --arxiv-only       # arXiv only
    python ingest/02_create_collections.py --docs-only        # docs only
    python ingest/02_create_collections.py --collection docs-rust  # single
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

# ---------------------------------------------------------------------------
# arXiv subject collections — populated by ingest/03 + ingest/04
# ---------------------------------------------------------------------------
ARXIV_COLLECTIONS = [
    "arXiv",                     # catch-all (every paper also indexed here)
    "arxiv-cs-ml-ai",            # cs.LG, cs.AI, cs.NE
    "arxiv-cs-cv",               # cs.CV
    "arxiv-cs-nlp-ir",           # cs.CL, cs.IR
    "arxiv-condmat",             # cond-mat.*
    "arxiv-astro",               # astro-ph.*
    "arxiv-hep",                 # hep-th, hep-ph, hep-ex, hep-lat
    "arxiv-math-applied",        # math.NA, math.OC, math.PR, math.ST
    "arxiv-math-phys",           # math-ph, math.MP
    "arxiv-math-pure",           # math.AG, math.AT, math.CA, math.CO, etc.
    "arxiv-misc",                # cs.HC, cs.SE, cs.NI, cs.DB, cs.PF, eess.SP
    "arxiv-nucl-nlin-physother", # nucl-th, nucl-ex, nlin.*, physics.*
    "arxiv-qbio-qfin-econ",      # q-bio.*, q-fin.*, econ.*
    "arxiv-quantph-grqc",        # quant-ph, gr-qc
    "arxiv-stat-eess",           # stat.*, eess.*
    "arxiv-cs-systems-theory",   # cs.DC, cs.DS, cs.CC, cs.IT, cs.GT
]

# ---------------------------------------------------------------------------
# Documentation collections — populated by ingest/05 (PDFs) + ingest/07 (HTML)
# ---------------------------------------------------------------------------
DOCS_COLLECTIONS = [
    "docs-python",        # Python stdlib, NumPy, Pandas, FastAPI, LangChain, Pydantic, etc.
    "docs-rust",          # The Rust Book, std lib, Cargo, Tokio, Axum, Serde, etc.
    "docs-javascript",    # MDN JS, Node.js, TypeScript, React, Next.js, Deno, Bun, etc.
    "docs-docker",        # Docker Engine, Compose, Buildx, Kubernetes, Helm, etc.
    "docs-anthropic",     # Claude API, Computer Use (CUA), Model Context Protocol (MCP)
    "docs-applescript",   # AppleScript Language Guide, OSAX, macOS automation
    "docs-devops",        # GitHub Actions, Terraform, Ansible, CI/CD, Grafana, Prometheus
    "docs-web",           # HTML5, CSS3, Web APIs, HTTP, browser standards (MDN non-JS)
]

# All collections together
COLLECTIONS = ARXIV_COLLECTIONS + DOCS_COLLECTIONS

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
    collection_set: str = "all",
    single_collection: str | None = None,
) -> dict[str, bool]:
    """
    Create Qdrant collections for arXiv and/or documentation.

    Parameters
    ----------
    collection_set : str
        "all"    — all arXiv + docs collections (default)
        "arxiv"  — arXiv subject collections only
        "docs"   — documentation collections only
    single_collection : str | None
        If set, create only this one collection.

    Returns
    -------
    dict[str, bool]
        Maps collection name → True (created) / False (already existed)
    """
    if single_collection:
        target = [single_collection]
    elif collection_set == "arxiv":
        target = ARXIV_COLLECTIONS
    elif collection_set == "docs":
        target = DOCS_COLLECTIONS
    else:
        target = COLLECTIONS

    client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=30)

    results: dict[str, bool] = {}
    for name in target:
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
        description="Create Qdrant collections for arXiv RAG + documentation"
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
        "--arxiv-only",
        action="store_true",
        help="Create only the 16 arXiv subject collections",
    )
    parser.add_argument(
        "--docs-only",
        action="store_true",
        help="Create only the 8 documentation collections",
    )
    parser.add_argument(
        "--collection",
        default=None,
        metavar="NAME",
        help="Create a single named collection (e.g. --collection docs-rust)",
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

    collection_set = "all"
    if args.arxiv_only:
        collection_set = "arxiv"
    elif args.docs_only:
        collection_set = "docs"

    print(f"Connecting to Qdrant at {qdrant_url}")
    results = create_all_collections(
        qdrant_url,
        api_key,
        recreate=args.recreate,
        collection_set=collection_set,
        single_collection=args.collection,
    )

    created = sum(1 for v in results.values() if v)
    skipped = len(results) - created

    print(f"\nDone: {created} created, {skipped} already existed.")
    verify_collections(qdrant_url, api_key)


if __name__ == "__main__":
    main()
