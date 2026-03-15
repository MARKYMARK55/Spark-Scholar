#!/usr/bin/env python3
"""
ingest/05_ingest_pdfs.py
========================
Full-text PDF ingestion with automatic classification.

Pipeline
--------
1. Extract text from each PDF using a three-tier extractor cascade:
     - Tier 1: docling (IBM, Apache 2.0) — best for multi-column academic layouts,
               tables, and figures
     - Tier 2: PyMuPDF (fitz) — fast and reliable for straightforward PDFs
     - Tier 3: unstructured — OCR-capable fallback for scanned / complex PDFs
2. Split text into overlapping chunks using tiktoken (cl100k_base)
3. Embed all chunks with BGE-M3 dense + sparse
4. Cluster chunks with HDBSCAN + UMAP to auto-detect topics
5. Name clusters using Qwen3 via local vLLM
6. Route each chunk to its Qdrant collection (by detected topic + filename)
7. Upsert all chunks as points with rich payload

Payload per chunk
-----------------
    arxiv_id       : filename stem (for PDFs not on arXiv)
    title          : extracted from first page or metadata
    chunk_text     : the raw text of the chunk
    page_num       : page number (1-indexed)
    chunk_idx      : chunk index within document
    source_file    : original filename
    topic_id       : HDBSCAN cluster label (-1 = noise)
    topic_name     : LLM-generated cluster name
    type           : "chunk"
    year           : extracted from PDF metadata if available
    authors        : extracted from PDF metadata if available

Usage
-----
    python ingest/05_ingest_pdfs.py --input-dir /path/to/pdfs/
    python ingest/05_ingest_pdfs.py --input-dir ./pdfs/ --collection arxiv-cs-ml-ai
    python ingest/05_ingest_pdfs.py --input-dir ./pdfs/ --host qdrant --port 6333

Requirements
------------
    pip install docling pymupdf unstructured tiktoken hdbscan umap-learn scikit-learn tqdm httpx qdrant-client
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tiktoken
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from tqdm import tqdm

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../env/.env"), override=False)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.embeddings import encode_dense, encode_sparse
from pipeline.router import route_query

logger = logging.getLogger(__name__)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "simple-api-key")
VLLM_MODEL = os.environ.get("VLLM_MODEL_NAME", "local-model")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))

# Additional "open-webui files" collection for all ingested PDFs
OWUI_FILES_COLLECTION = "open-webui_files"


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

DOCLING_URL = os.environ.get("DOCLING_URL", "http://docling:5001/v1/convert/file")


def _strip_base64_images(md_text: str) -> str:
    """Remove base64-encoded image data from Markdown.

    Docling embeds figures as inline base64 PNGs which pollute chunks.
    Strips both ``![...](data:image/...)`` syntax and bare base64 blobs.
    """
    import re
    # Remove markdown image tags with data URIs
    md_text = re.sub(r'!\[[^\]]*\]\(data:image/[^)]+\)', '<!-- image -->', md_text)
    # Remove bare base64 blobs (100+ chars of base64 alphabet)
    md_text = re.sub(r'[A-Za-z0-9+/=]{100,}', '', md_text)
    return md_text


def _split_markdown_sections(md_text: str) -> list[dict]:
    """Split Markdown text into sections by headings."""
    md_text = _strip_base64_images(md_text)
    sections = []
    current_heading = ""
    current_text: list[str] = []
    page_num = 1

    for line in md_text.split("\n"):
        if line.startswith("#"):
            if current_text:
                text = "\n".join(current_text).strip()
                if text:
                    sections.append({"heading": current_heading, "text": text, "page_num": page_num})
            current_heading = line.lstrip("#").strip()
            current_text = []
        else:
            current_text.append(line)

    if current_text:
        text = "\n".join(current_text).strip()
        if text:
            sections.append({"heading": current_heading, "text": text, "page_num": page_num})

    return sections


def extract_text_docling(pdf_path: str) -> tuple[list[dict], dict]:
    """
    Extract structured Markdown from PDF via docling-serve HTTP API.

    Uses the official docling-serve /v1alpha/convert/file endpoint.

    Returns
    -------
    (sections, metadata)
        sections: list of {"heading": str, "text": str, "page_num": int}
        metadata: dict with title, authors, year
    """
    import httpx

    with open(pdf_path, "rb") as f:
        resp = httpx.post(
            DOCLING_URL,
            files={"files": (Path(pdf_path).name, f, "application/pdf")},
            timeout=600.0,
        )
    resp.raise_for_status()
    data = resp.json()

    # Handle three response formats:
    # 1. docling-serve:  {document: {md_content: "..."}}
    # 2. custom server:  {sections: [...], metadata: {...}}
    # 3. docling-simple: {markdown: "..."}
    doc = data.get("document", {})
    md_content = doc.get("md_content", "") or data.get("markdown", "")

    if md_content:
        sections = _split_markdown_sections(md_content)
    else:
        sections = data.get("sections", [])

    metadata = {
        "title": doc.get("name", "") or Path(pdf_path).stem,
        "authors": "",
        "year": None,
    }

    # Reject near-empty extractions so callers can fall through
    total_chars = sum(len(s.get("text", "")) for s in sections)
    if total_chars < 200:
        return [], {}

    return sections, metadata


def extract_text_pymupdf(pdf_path: str) -> tuple[list[dict], dict]:
    """
    Extract text from PDF using PyMuPDF.

    Returns
    -------
    (pages, metadata)
        pages: list of {"page_num": int, "text": str}
        metadata: dict with title, author, year from PDF metadata
    """
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        text = text.strip()
        if text:
            pages.append({"page_num": page_num + 1, "text": text, "heading": ""})

    meta = doc.metadata or {}
    year = None
    if meta.get("creationDate"):
        try:
            year = int(meta["creationDate"][2:6])
        except (ValueError, IndexError):
            pass

    metadata = {
        "title": meta.get("title", ""),
        "authors": meta.get("author", ""),
        "year": year,
    }
    doc.close()
    return pages, metadata


def extract_text_unstructured(pdf_path: str) -> tuple[list[dict], dict]:
    """
    Extract text using unstructured as fallback for complex layouts
    (two-column papers, scanned PDFs with OCR, etc.)
    """
    from unstructured.partition.pdf import partition_pdf

    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True,
        include_page_breaks=True,
    )

    pages: dict[int, list[str]] = {}
    current_page = 1
    for elem in elements:
        if hasattr(elem, "category") and elem.category == "PageBreak":
            current_page += 1
            continue
        text = str(elem).strip()
        if text:
            pages.setdefault(current_page, []).append(text)

    page_list = [
        {"page_num": pn, "text": "\n".join(texts), "heading": ""}
        for pn, texts in sorted(pages.items())
    ]
    return page_list, {}


def extract_text(pdf_path: str) -> tuple[list[dict], dict]:
    """
    Extract text using a three-tier cascade:
      1. docling  — best for multi-column academic layouts + tables/figures
      2. PyMuPDF  — fast and reliable for straightforward PDFs
      3. unstructured — OCR-capable last-resort fallback
    """
    # 1. Try docling (best for multi-column academic layouts + tables)
    #    Returns sections with headings for section-aware chunking
    try:
        sections, meta = extract_text_docling(pdf_path)
        if sum(len(s["text"]) for s in sections) >= 200:
            logger.info("docling extracted %d sections via Markdown export", len(sections))
            return sections, meta
        logger.info("docling low yield, falling back to PyMuPDF")
    except Exception as exc:
        logger.info("docling failed (%s), falling back to PyMuPDF", exc)

    # 2. PyMuPDF
    try:
        pages, meta = extract_text_pymupdf(pdf_path)
        if sum(len(p["text"]) for p in pages) >= 500:
            return pages, meta
        logger.info("Low text yield from PyMuPDF (%d chars), trying unstructured...", sum(len(p["text"]) for p in pages))
    except Exception as exc:
        logger.error("PyMuPDF extraction failed for %s: %s", pdf_path, exc)

    # 3. unstructured fallback
    try:
        return extract_text_unstructured(pdf_path)
    except Exception as unst_exc:
        logger.error("All extractors failed for %s: %s", pdf_path, unst_exc)
        return [], {}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    sections: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Section-aware chunking: split within sections, never across them.

    If a section fits within chunk_size tokens, it becomes a single chunk.
    If a section is too large, it gets split with token-based sliding window.
    Each chunk carries its section heading as context prefix.

    Parameters
    ----------
    sections : list of {"heading": str, "text": str, "page_num": int}
        From Docling Markdown extraction, or legacy page-based extraction
        (which uses {"page_num": int, "text": str} — heading will be "").
    chunk_size : int
        Target chunk size in tokens.
    chunk_overlap : int
        Overlap between consecutive chunks in tokens.

    Returns
    -------
    list of {"chunk_idx": int, "page_num": int, "text": str, "token_count": int,
             "heading": str}
    """
    enc = tiktoken.get_encoding("cl100k_base")

    chunks = []
    chunk_idx = 0

    step = chunk_size - chunk_overlap
    if step <= 0:
        step = chunk_size

    for section in sections:
        section_text = section["text"]
        heading = section.get("heading", "")
        page_num = section.get("page_num", 1)

        tokens = enc.encode(section_text)

        if len(tokens) <= chunk_size:
            # Section fits in one chunk
            if section_text.strip():
                chunks.append({
                    "chunk_idx": chunk_idx,
                    "page_num": page_num,
                    "text": section_text.strip(),
                    "token_count": len(tokens),
                    "heading": heading,
                })
                chunk_idx += 1
        else:
            # Section too large — sliding window within this section
            for start_tok in range(0, max(1, len(tokens)), step):
                end_tok = min(start_tok + chunk_size, len(tokens))
                chunk_tokens = tokens[start_tok:end_tok]
                chunk_text_str = enc.decode(chunk_tokens)

                if chunk_text_str.strip():
                    # Prefix with heading for context if this isn't the first sub-chunk
                    text = chunk_text_str.strip()
                    if start_tok > 0 and heading:
                        text = f"{heading}\n\n{text}"

                    chunks.append({
                        "chunk_idx": chunk_idx,
                        "page_num": page_num,
                        "text": text,
                        "token_count": len(chunk_tokens),
                        "heading": heading,
                    })
                    chunk_idx += 1

                if end_tok >= len(tokens):
                    break

    return chunks


# ---------------------------------------------------------------------------
# Topic clustering
# ---------------------------------------------------------------------------

def cluster_chunks(
    chunk_texts: list[str],
    min_cluster_size: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster chunk texts using UMAP + HDBSCAN for topic auto-detection.

    Parameters
    ----------
    chunk_texts : list[str]
    min_cluster_size : int

    Returns
    -------
    (labels, embeddings_2d)
        labels: HDBSCAN cluster labels (-1 = noise)
        embeddings_2d: 2D UMAP projections for visualisation
    """
    import umap
    import hdbscan

    if len(chunk_texts) < 5:
        return np.full(len(chunk_texts), -1), np.zeros((len(chunk_texts), 2))

    # Embed all chunks with dense embedder
    embeddings = encode_dense(chunk_texts)

    # UMAP for dimensionality reduction before clustering
    reducer = umap.UMAP(
        n_components=min(5, len(chunk_texts) - 1),
        n_neighbors=min(15, len(chunk_texts) - 1),
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        n_jobs=1,
    )
    reduced = reducer.fit_transform(embeddings)

    # HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=3,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(reduced)

    # 2D projection for visualisation
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, len(chunk_texts) - 1),
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        n_jobs=1,
    )
    embeddings_2d = reducer_2d.fit_transform(embeddings)

    unique_labels = set(labels) - {-1}
    logger.info(
        "Clustering: %d chunks → %d clusters (%d noise)",
        len(chunk_texts),
        len(unique_labels),
        np.sum(labels == -1),
    )
    return labels, embeddings_2d


def name_clusters_with_llm(
    cluster_labels: np.ndarray,
    chunk_texts: list[str],
    vllm_url: str = VLLM_URL,
    vllm_api_key: str = VLLM_API_KEY,
    model: str = VLLM_MODEL,
) -> dict[int, str]:
    """
    Use Qwen3 via vLLM to name each cluster based on a sample of its chunks.

    Parameters
    ----------
    cluster_labels : np.ndarray
        HDBSCAN cluster labels.
    chunk_texts : list[str]

    Returns
    -------
    dict[int, str]
        Maps cluster label → human-readable name.
    """
    import httpx

    unique_labels = sorted(set(cluster_labels) - {-1})
    cluster_names: dict[int, str] = {-1: "misc"}

    client = httpx.Client(
        timeout=30.0,
        headers={"Authorization": f"Bearer {vllm_api_key}"},
    )
    url = f"{vllm_url.rstrip('/')}/v1/chat/completions"

    for label in unique_labels:
        # Get up to 3 representative chunks from this cluster
        indices = [i for i, l in enumerate(cluster_labels) if l == label]
        sample_indices = indices[:3]
        sample_texts = [chunk_texts[i][:300] for i in sample_indices]
        combined_sample = "\n---\n".join(sample_texts)

        prompt = (
            f"The following text excerpts are from the same topical cluster in a scientific paper:\n\n"
            f"{combined_sample}\n\n"
            f"Give a short descriptive name (3-6 words) for this topic. "
            f"Respond with ONLY the topic name, nothing else."
        )

        try:
            resp = client.post(url, json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 20,
                "temperature": 0.1,
            })
            resp.raise_for_status()
            name = resp.json()["choices"][0]["message"]["content"].strip()
            # Clean up common LLM verbosity
            name = re.sub(r'^(Topic:|Name:|Label:)\s*', '', name, flags=re.IGNORECASE)
            name = name.strip('"\'').strip()[:60]
            cluster_names[label] = name or f"topic_{label}"
        except Exception as exc:
            logger.warning("LLM cluster naming failed for cluster %d: %s", label, exc)
            cluster_names[label] = f"topic_{label}"

    client.close()
    return cluster_names


# ---------------------------------------------------------------------------
# Qdrant collection management
# ---------------------------------------------------------------------------

DENSE_DIM = 1024  # BGE-M3 dense vector dimension


def ensure_collection(client: QdrantClient, name: str) -> None:
    """Create the collection if it does not already exist."""
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense_embedding": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse_text": SparseVectorParams(),
        },
    )
    logger.info("Created collection %s (dense=%d + sparse)", name, DENSE_DIM)


# ---------------------------------------------------------------------------
# Qdrant upsert
# ---------------------------------------------------------------------------

def _make_point_id(source_file: str, chunk_idx: int) -> int:
    """Create a deterministic Qdrant point ID from file + chunk index."""
    key = f"{source_file}::{chunk_idx}"
    return abs(int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)) % (2 ** 63)


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    chunks: list[dict],
    dense_vecs: np.ndarray,
    sparse_vecs: list,
    source_file: str,
    file_meta: dict,
    batch_size: int = 64,
) -> int:
    """
    Upsert all chunks of a PDF into the given collection.

    Returns
    -------
    int
        Number of points upserted.
    """
    total_upserted = 0

    for batch_start in range(0, len(chunks), batch_size):
        batch_chunks = chunks[batch_start : batch_start + batch_size]
        batch_dense = dense_vecs[batch_start : batch_start + batch_size]
        batch_sparse = sparse_vecs[batch_start : batch_start + batch_size]

        points = []
        for i, (chunk, dv, sv) in enumerate(zip(batch_chunks, batch_dense, batch_sparse)):
            global_idx = batch_start + i
            point_id = _make_point_id(source_file, chunk["chunk_idx"])

            payload = {
                "arxiv_id": Path(source_file).stem,
                "title": file_meta.get("title") or Path(source_file).stem,
                "authors": file_meta.get("authors", ""),
                "year": file_meta.get("year"),
                "categories": file_meta.get("categories", ""),
                "chunk_text": chunk["text"],
                "chunk_idx": chunk["chunk_idx"],
                "page_num": chunk["page_num"],
                "token_count": chunk["token_count"],
                "section_heading": chunk.get("heading", ""),
                "source_file": source_file,
                "topic_id": int(chunk.get("topic_id", -1)),
                "topic_name": chunk.get("topic_name", "misc"),
                "type": "chunk",
            }

            qdrant_sparse = SparseVector(
                indices=sv.indices,
                values=sv.values,
            )

            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "dense_embedding": dv.tolist(),
                        "sparse_text": qdrant_sparse,
                    },
                    payload=payload,
                )
            )

        client.upsert(collection_name=collection, points=points, wait=True)
        total_upserted += len(points)

    return total_upserted


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def ingest_pdfs(
    input_dir: str,
    collection_override: Optional[str] = None,
    max_files: Optional[int] = None,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
) -> dict[str, int]:
    """
    Ingest all PDFs in input_dir.

    Returns
    -------
    dict[str, int]
        Maps collection → number of chunks upserted.
    """
    # Prefer QDRANT_URL env var (set by Docker) over CLI host/port
    qdrant_url = QDRANT_URL if QDRANT_URL != "http://localhost:6333" else f"http://{qdrant_host}:{qdrant_port}"
    client = QdrantClient(
        url=qdrant_url,
        api_key=QDRANT_API_KEY,
        timeout=120,
    )
    logger.info("Qdrant URL: %s", qdrant_url)

    pdf_files = sorted(Path(input_dir).rglob("*.pdf"))
    if max_files:
        pdf_files = pdf_files[:max_files]

    logger.info("Found %d PDF files in %s", len(pdf_files), input_dir)

    counts: dict[str, int] = {}

    for pdf_path in tqdm(pdf_files, desc="Ingesting PDFs", unit="pdf"):
        try:
            logger.info("Processing: %s", pdf_path.name)

            # 1. Extract text
            pages, file_meta = extract_text(str(pdf_path))
            if not pages:
                logger.warning("No text extracted from %s, skipping", pdf_path.name)
                continue

            # 2. Chunk
            chunks = chunk_text(pages, CHUNK_SIZE, CHUNK_OVERLAP)
            if not chunks:
                logger.warning("No chunks from %s, skipping", pdf_path.name)
                continue

            logger.info("  %d pages → %d chunks", len(pages), len(chunks))

            chunk_texts = [c["text"] for c in chunks]

            # 3. Cluster for topic detection (optional — skipped if umap/hdbscan unavailable)
            try:
                if len(chunks) >= 5:
                    labels, _ = cluster_chunks(chunk_texts)
                    # LLM naming deferred to post-processing (avoids API errors during batch)
                    cluster_names = {int(l): f"topic_{l}" for l in set(labels)}
                    cluster_names[-1] = "misc"
                    for i, chunk in enumerate(chunks):
                        topic_id = int(labels[i]) if i < len(labels) else -1
                        chunk["topic_id"] = topic_id
                        chunk["topic_name"] = cluster_names.get(topic_id, "misc")
                else:
                    for chunk in chunks:
                        chunk["topic_id"] = -1
                        chunk["topic_name"] = "misc"
            except (ImportError, Exception) as e:
                logger.warning("  Clustering skipped: %s", e)
                for chunk in chunks:
                    chunk["topic_id"] = -1
                    chunk["topic_name"] = "misc"

            # 4. Embed all chunks (dense + sparse)
            logger.info("  Embedding %d chunks...", len(chunks))
            dense_vecs = encode_dense(chunk_texts)
            sparse_vecs = encode_sparse(chunk_texts)

            # 5. Route to collection
            if collection_override:
                collection = collection_override
            else:
                # Use the topic names + filename to route
                all_topic_names = " ".join({c["topic_name"] for c in chunks})
                all_text_sample = " ".join(chunk_texts[:3])
                collection = route_query(all_topic_names + " " + all_text_sample)[0]

            # 6. Upsert into main collection
            ensure_collection(client, collection)
            n_upserted = upsert_chunks(
                client,
                collection,
                chunks,
                dense_vecs,
                sparse_vecs,
                str(pdf_path.name),
                file_meta,
            )
            counts[collection] = counts.get(collection, 0) + n_upserted
            logger.info("  → Upserted %d chunks to %s", n_upserted, collection)

            # 7. Also upsert into open-webui_files if the collection exists
            try:
                existing = {c.name for c in client.get_collections().collections}
                if OWUI_FILES_COLLECTION in existing:
                    upsert_chunks(
                        client,
                        OWUI_FILES_COLLECTION,
                        chunks,
                        dense_vecs,
                        sparse_vecs,
                        str(pdf_path.name),
                        file_meta,
                    )
            except Exception:
                pass  # open-webui_files is optional

        except Exception as exc:
            logger.error("Error processing %s: %s", pdf_path.name, exc, exc_info=True)
            continue

    total = sum(counts.values())
    logger.info("PDF ingestion complete: %d chunks across %d collections", total, len(counts))
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDF files into the Spark Scholar vector database"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing PDF files to ingest",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Override automatic collection routing",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Qdrant host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6333,
        help="Qdrant port (default: 6333)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Process at most this many PDF files (for testing)",
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    counts = ingest_pdfs(
        input_dir=args.input_dir,
        collection_override=args.collection,
        max_files=args.max_files,
        qdrant_host=args.host,
        qdrant_port=args.port,
    )

    print(f"\nIngestion complete:")
    for coll, cnt in sorted(counts.items()):
        print(f"  {coll}: {cnt} chunks")
    print(f"Total: {sum(counts.values())} chunks")


if __name__ == "__main__":
    main()
