#!/usr/bin/env python3
"""
ingest/07_ingest_html_docs.py
==============================
Ingest web documentation (HTML pages) into Qdrant using BGE-M3 dense + sparse embeddings.

Overview
--------
This script crawls one or more documentation URLs, extracts clean text from their HTML,
chunks it with tiktoken, embeds each chunk via the BGE-M3 HTTP embedding services, and
upserts the results into a named Qdrant collection as hybrid (dense + sparse) points.

It is designed for developer documentation sites such as:
  - https://docs.anthropic.com/
  - https://doc.rust-lang.org/
  - https://docs.docker.com/

Pipeline
--------
1.  Resolve seed URLs (--url, --url-file, and/or --sitemap discovery)
2.  Crawl each seed up to --depth hops, staying on the same domain + path prefix
3.  Fetch each page with requests (polite delay between requests)
4.  Parse HTML with BeautifulSoup; extract main content, strip boilerplate
5.  Chunk clean text into overlapping token windows with tiktoken (cl100k_base)
6.  Embed chunks in batches via encode_dense() and encode_sparse() HTTP services
7.  Build Qdrant PointStruct with NamedVector + NamedSparseVector
8.  Upsert points into the target collection
9.  Record completed URLs in a progress file for resume support

Point ID
--------
Deterministic SHA-256 of "{source_url}::{chunk_idx}" → first 16 bytes as UUID string.

Payload per Qdrant point
------------------------
    source_url   : str         exact URL the page came from
    title        : str         <title> content, cleaned
    section      : str         first <h1> on the page (empty string if absent)
    chunk_text   : str         raw text of this chunk
    chunk_idx    : int         0-based index within the page's chunks
    collection   : str         Qdrant collection name (from --collection)
    type         : str         always "doc_chunk"
    source_type  : str         always "html"
    tag          : str | None  from --tag (e.g. "rust", "anthropic-api")
    ingested_at  : str         ISO 8601 UTC timestamp

Usage
-----
    # Single URL, default depth=1 (follows same-domain/prefix links once)
    python ingest/07_ingest_html_docs.py \\
        --url https://docs.anthropic.com/en/docs/ \\
        --collection docs-anthropic \\
        --tag anthropic-api

    # Bulk URL list
    python ingest/07_ingest_html_docs.py \\
        --url-file urls.txt \\
        --collection docs-rust \\
        --tag rust --depth 0

    # Dry-run to see what would be ingested
    python ingest/07_ingest_html_docs.py \\
        --url https://docs.docker.com/ \\
        --collection docs-docker \\
        --dry-run --verbose

    # Use sitemap for URL discovery, then crawl
    python ingest/07_ingest_html_docs.py \\
        --url https://docs.anthropic.com/en/docs/ \\
        --collection docs-anthropic \\
        --sitemap --depth 0

Requirements
------------
    pip install requests beautifulsoup4 tiktoken tqdm qdrant-client python-dotenv
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from urllib.parse import urljoin, urldefrag, urlparse

import numpy as np
import requests
import tiktoken
from bs4 import BeautifulSoup, Tag
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Environment & path setup
# ---------------------------------------------------------------------------

load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(__file__), "../env/.env"),
    override=False,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.embeddings import encode_dense, encode_sparse  # noqa: E402

# ---------------------------------------------------------------------------
# Globals from environment
# ---------------------------------------------------------------------------

QDRANT_URL: str = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY: str | None = os.environ.get("QDRANT_API_KEY") or None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CSS/element selectors tried in order for main content extraction
CONTENT_SELECTORS: list[str] = [
    'article[role="main"]',
    "main",
    'div[role="main"]',
    "div.document",
    "div.body",
    "article",
    "#content",
    "#main",
    ".markdown-body",
    ".rst-content",
]

# Tags stripped out before text extraction (boilerplate)
STRIP_SELECTORS: list[str] = [
    "nav",
    "footer",
    "header",
    "aside",
    ".sidebar",
    "#sidebar",
    ".navigation",
    ".toc",
    ".cookie-banner",
    "script",
    "style",
    '[aria-hidden="true"]',
]

# File extensions / query patterns that are not documentation pages
SKIP_URL_PATTERNS: tuple[str, ...] = (
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".whl",
    ".exe",
    ".dmg",
    ".ico",
    "?download",
    "mailto:",
    "javascript:",
)

# Realistic browser User-Agent to avoid bot-blocking
USER_AGENT: str = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Qdrant vector slot names (must match the collection's vector config)
DENSE_VECTOR_NAME: str = "dense_embedding"
SPARSE_VECTOR_NAME: str = "sparse_text"


# ---------------------------------------------------------------------------
# Progress file helpers
# ---------------------------------------------------------------------------


def load_progress(progress_file: str) -> set[str]:
    """
    Load the set of already-ingested URLs from the progress file.

    Returns an empty set if the file does not exist yet.
    """
    path = Path(progress_file)
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as fh:
        return {line.strip() for line in fh if line.strip()}


def append_progress(progress_file: str, url: str) -> None:
    """Append a single URL to the progress file (one URL per line)."""
    path = Path(progress_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(url + "\n")


# ---------------------------------------------------------------------------
# URL utilities
# ---------------------------------------------------------------------------


def normalise_url(url: str) -> str:
    """
    Strip fragment anchors and trailing slashes from *url* for deduplication.

    Examples
    --------
    >>> normalise_url("https://docs.example.com/foo#bar")
    'https://docs.example.com/foo'
    >>> normalise_url("https://docs.example.com/foo/")
    'https://docs.example.com/foo/'
    """
    defrag, _ = urldefrag(url)
    return defrag


def should_skip_url(url: str) -> bool:
    """
    Return True if the URL looks like a binary file or non-doc resource.
    """
    lower = url.lower()
    return any(lower.endswith(pat) or pat in lower for pat in SKIP_URL_PATTERNS)


def same_domain_and_prefix(url: str, seed_parsed: "ParseResult") -> bool:  # noqa: F821
    """
    Return True when *url* is on the same netloc as *seed_parsed* and its
    path starts with the seed's path prefix.

    Parameters
    ----------
    url : str
        Candidate URL to evaluate.
    seed_parsed : urllib.parse.ParseResult
        Parsed form of the original seed URL.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    if parsed.netloc != seed_parsed.netloc:
        return False

    # Normalise seed path: treat it as a directory prefix
    seed_prefix = seed_parsed.path
    if not seed_prefix.endswith("/"):
        # If seed path has no trailing slash, the prefix is up to the last "/"
        seed_prefix = seed_prefix.rsplit("/", 1)[0] + "/"

    return parsed.path.startswith(seed_prefix)


# ---------------------------------------------------------------------------
# Sitemap parsing
# ---------------------------------------------------------------------------


def fetch_sitemap_urls(
    base_url: str,
    session: requests.Session,
    seed_parsed: "ParseResult",  # noqa: F821
    timeout: float = 15.0,
) -> list[str]:
    """
    Attempt to discover crawlable URLs from ``{base_url}/sitemap.xml``.

    Handles both plain sitemaps and sitemap index files (nested sitemaps).
    Filters to URLs on the same domain and path prefix as *seed_parsed*.

    Parameters
    ----------
    base_url : str
        Seed URL base (e.g. ``https://docs.anthropic.com/en/docs/``).
    session : requests.Session
        Configured HTTP session.
    seed_parsed : urllib.parse.ParseResult
        Parsed seed URL used for domain/prefix filtering.
    timeout : float
        Per-request timeout in seconds.

    Returns
    -------
    list[str]
        Filtered list of discovered URLs (may be empty).
    """
    parsed = urlparse(base_url)
    sitemap_url = f"{parsed.scheme}://{parsed.netloc}/sitemap.xml"

    logger.info("Attempting sitemap discovery at %s", sitemap_url)
    try:
        resp = session.get(sitemap_url, timeout=timeout)
    except Exception as exc:
        logger.warning("Could not fetch sitemap: %s", exc)
        return []

    if resp.status_code != 200:
        logger.warning("Sitemap returned HTTP %d", resp.status_code)
        return []

    soup = BeautifulSoup(resp.text, "xml")
    urls: list[str] = []

    # Sitemap index: contains <sitemap><loc>...</loc></sitemap>
    sitemap_locs = soup.find_all("sitemap")
    if sitemap_locs:
        logger.info("Sitemap index found with %d sub-sitemaps", len(sitemap_locs))
        for sm in sitemap_locs:
            loc_tag = sm.find("loc")
            if not loc_tag:
                continue
            sub_url = loc_tag.get_text(strip=True)
            try:
                sub_resp = session.get(sub_url, timeout=timeout)
                if sub_resp.status_code == 200:
                    sub_soup = BeautifulSoup(sub_resp.text, "xml")
                    for loc in sub_soup.find_all("loc"):
                        u = loc.get_text(strip=True)
                        if same_domain_and_prefix(u, seed_parsed) and not should_skip_url(u):
                            urls.append(normalise_url(u))
            except Exception as exc:
                logger.warning("Failed to fetch sub-sitemap %s: %s", sub_url, exc)
        return list(dict.fromkeys(urls))  # deduplicate, preserve order

    # Plain sitemap: contains <url><loc>...</loc></url>
    for loc in soup.find_all("loc"):
        u = loc.get_text(strip=True)
        if same_domain_and_prefix(u, seed_parsed) and not should_skip_url(u):
            urls.append(normalise_url(u))

    logger.info("Sitemap discovery found %d matching URLs", len(urls))
    return list(dict.fromkeys(urls))


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------


def _clean_title(raw_title: str) -> str:
    """
    Strip common documentation suffixes from a page title.

    Handles patterns like:
      - "Functions — Rust Reference"
      - "Installation | Docker Docs"
      - "Overview - Anthropic API Documentation"
    """
    # Remove trailing doc-site suffixes
    raw_title = re.sub(
        r"\s*[|—–\-]\s*(documentation|docs|reference|guide|api|manual|wiki)\s*$",
        "",
        raw_title,
        flags=re.IGNORECASE,
    ).strip()
    return raw_title or raw_title


def extract_page_content(html: str, url: str) -> dict:
    """
    Extract structured content from an HTML page.

    Tries content selectors in priority order, strips navigation/boilerplate,
    preserves code block text, and returns a dict with:

    Parameters
    ----------
    html : str
        Raw HTML source.
    url : str
        Source URL (used only for logging).

    Returns
    -------
    dict with keys:
        title : str     — cleaned page title
        section : str   — text of the first <h1> found (empty string if none)
        text : str      — clean body text, whitespace-normalised
    """
    soup = BeautifulSoup(html, "html.parser")

    # --- Title ---------------------------------------------------------------
    title_tag = soup.find("title")
    raw_title = title_tag.get_text(strip=True) if title_tag else ""
    title = _clean_title(raw_title)

    # --- Strip boilerplate ---------------------------------------------------
    for selector in STRIP_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    # --- Find main content container -----------------------------------------
    content_root: Tag | None = None
    for selector in CONTENT_SELECTORS:
        content_root = soup.select_one(selector)
        if content_root is not None:
            break
    if content_root is None:
        content_root = soup.find("body")
    if content_root is None:
        content_root = soup  # last resort: whole document

    # --- First h1 as section header ------------------------------------------
    h1_tag = content_root.find("h1")
    section = h1_tag.get_text(strip=True) if h1_tag else ""

    # --- Extract text, preserving code blocks --------------------------------
    # Replace <code>/<pre> with their text content marked by newlines so
    # the chunker sees them as separate paragraphs rather than inline blobs.
    for code_tag in content_root.find_all(["pre", "code"]):
        code_text = code_tag.get_text()
        code_tag.replace_with(f"\n{code_text}\n")

    raw_text = content_root.get_text(separator="\n")

    # --- Whitespace normalisation --------------------------------------------
    # Collapse runs of blank lines to at most two, collapse inline whitespace
    lines = raw_text.splitlines()
    cleaned_lines: list[str] = []
    blank_run = 0
    for line in lines:
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            blank_run = 0
            cleaned_lines.append(line)
        else:
            blank_run += 1
            if blank_run <= 2:
                cleaned_lines.append("")

    text = "\n".join(cleaned_lines).strip()

    return {"title": title, "section": section, "text": text}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict]:
    """
    Split *text* into overlapping token-based chunks using tiktoken cl100k_base.

    Parameters
    ----------
    text : str
        Clean document text to split.
    chunk_size : int
        Maximum number of tokens per chunk.
    chunk_overlap : int
        Number of tokens to repeat at the start of each subsequent chunk.

    Returns
    -------
    list of dict, each with:
        chunk_idx  : int   — 0-based index
        text       : str   — decoded chunk text
        token_count: int   — actual token count
    """
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    if not tokens:
        return []

    step = max(1, chunk_size - chunk_overlap)
    chunks: list[dict] = []
    chunk_idx = 0

    for start in range(0, len(tokens), step):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_str = enc.decode(chunk_tokens).strip()
        if chunk_str:
            chunks.append(
                {
                    "chunk_idx": chunk_idx,
                    "text": chunk_str,
                    "token_count": len(chunk_tokens),
                }
            )
            chunk_idx += 1
        if end >= len(tokens):
            break

    return chunks


# ---------------------------------------------------------------------------
# Point ID
# ---------------------------------------------------------------------------


def make_point_id(source_url: str, chunk_idx: int) -> str:
    """
    Create a deterministic UUID string from *source_url* and *chunk_idx*.

    Uses SHA-256 of ``"{source_url}::{chunk_idx}"`` and takes the first
    16 bytes as a UUID (version 4 format, but deterministic).
    """
    key = f"{source_url}::{chunk_idx}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return str(uuid.UUID(bytes=digest[:16]))


# ---------------------------------------------------------------------------
# Qdrant upsert
# ---------------------------------------------------------------------------


def upsert_page_chunks(
    client: QdrantClient,
    collection: str,
    chunks: list[dict],
    dense_vecs: np.ndarray,
    sparse_vecs: list[SparseVector],
    source_url: str,
    title: str,
    section: str,
    tag: str | None,
    dry_run: bool = False,
    verbose: bool = False,
) -> int:
    """
    Upsert all chunks from a single page into Qdrant.

    Parameters
    ----------
    client : QdrantClient
    collection : str
    chunks : list[dict]
        Output of ``chunk_text()``.
    dense_vecs : np.ndarray
        Shape (N, 1024), one row per chunk.
    sparse_vecs : list[SparseVector]
        One SparseVector per chunk.
    source_url : str
    title : str
    section : str
    tag : str | None
    dry_run : bool
        If True, build points but skip the actual upsert call.
    verbose : bool

    Returns
    -------
    int
        Number of points upserted (0 in dry_run mode).
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    points: list[PointStruct] = []

    for chunk, dv, sv in zip(chunks, dense_vecs, sparse_vecs):
        point_id = make_point_id(source_url, chunk["chunk_idx"])

        payload: dict = {
            "source_url": source_url,
            "title": title,
            "section": section,
            "chunk_text": chunk["text"],
            "chunk_idx": chunk["chunk_idx"],
            "collection": collection,
            "type": "doc_chunk",
            "source_type": "html",
            "tag": tag,
            "ingested_at": now_iso,
        }

        qdrant_sparse = SparseVector(indices=sv.indices, values=sv.values)

        points.append(
            PointStruct(
                id=point_id,
                vector={
                    DENSE_VECTOR_NAME: dv.tolist(),
                    SPARSE_VECTOR_NAME: qdrant_sparse,
                },
                payload=payload,
            )
        )

        if verbose:
            logger.debug(
                "  chunk %d | %d tokens | id=%s",
                chunk["chunk_idx"],
                chunk["token_count"],
                point_id,
            )

    if dry_run:
        logger.info(
            "[dry-run] Would upsert %d points for %s", len(points), source_url
        )
        return 0

    if points:
        client.upsert(collection_name=collection, points=points, wait=True)

    return len(points)


# ---------------------------------------------------------------------------
# Page fetching
# ---------------------------------------------------------------------------


def fetch_page(url: str, session: requests.Session, timeout: float = 15.0) -> str | None:
    """
    Fetch a single URL and return its raw HTML.

    Returns None on non-200 status or network error (logs a warning).
    """
    try:
        resp = session.get(url, timeout=timeout)
    except requests.Timeout:
        logger.warning("Timeout fetching %s — skipping", url)
        return None
    except requests.RequestException as exc:
        logger.warning("Network error fetching %s: %s — skipping", url, exc)
        return None

    if resp.status_code != 200:
        logger.warning("HTTP %d for %s — skipping", resp.status_code, url)
        return None

    content_type = resp.headers.get("content-type", "")
    if "html" not in content_type and "text" not in content_type:
        logger.warning(
            "Unexpected content-type %r for %s — skipping", content_type, url
        )
        return None

    return resp.text


# ---------------------------------------------------------------------------
# Crawling
# ---------------------------------------------------------------------------


def crawl(
    seed_url: str,
    session: requests.Session,
    depth: int = 1,
    delay: float = 0.5,
    extra_urls: list[str] | None = None,
) -> Iterator[str]:
    """
    BFS crawl from *seed_url*, yielding each unique page URL to process.

    Stays on the same domain and path prefix as *seed_url*.
    Respects ``--depth``: depth=0 yields only the seed, depth=1 follows
    links from the seed page, depth=2 follows links of those pages, etc.

    Parameters
    ----------
    seed_url : str
        Starting URL.
    session : requests.Session
        Configured HTTP session (headers, etc.)
    depth : int
        Maximum crawl depth (0 = seed only).
    delay : float
        Seconds to sleep between HTTP requests.
    extra_urls : list[str] | None
        Additional pre-discovered URLs (e.g. from sitemap) to add at depth 0.

    Yields
    ------
    str
        Normalised page URL ready to fetch and ingest.
    """
    seed_parsed = urlparse(seed_url)
    visited: set[str] = set()

    # Queue entries: (url, current_depth)
    queue: deque[tuple[str, int]] = deque()
    seed_norm = normalise_url(seed_url)
    queue.append((seed_norm, 0))
    visited.add(seed_norm)

    # Inject sitemap / pre-discovered URLs at depth 0 so they get yielded
    # but their links are only followed if depth > 0.
    if extra_urls:
        for eu in extra_urls:
            norm = normalise_url(eu)
            if norm not in visited:
                visited.add(norm)
                queue.append((norm, 0))

    while queue:
        url, current_depth = queue.popleft()

        if should_skip_url(url):
            logger.debug("Skipping non-doc URL: %s", url)
            continue

        yield url

        if current_depth >= depth:
            # Don't extract links from this page
            continue

        # Fetch to discover child links
        time.sleep(delay)
        html = fetch_page(url, session)
        if html is None:
            continue

        soup = BeautifulSoup(html, "html.parser")
        for anchor in soup.find_all("a", href=True):
            href: str = anchor["href"].strip()
            if not href:
                continue
            # Resolve relative URLs
            absolute = urljoin(url, href)
            norm = normalise_url(absolute)

            if norm in visited:
                continue
            if not same_domain_and_prefix(norm, seed_parsed):
                continue
            if should_skip_url(norm):
                continue

            visited.add(norm)
            queue.append((norm, current_depth + 1))

        logger.debug(
            "Crawled depth=%d: %s — queue size now %d", current_depth, url, len(queue)
        )


# ---------------------------------------------------------------------------
# Main ingestion logic
# ---------------------------------------------------------------------------


def ingest_url(
    url: str,
    session: requests.Session,
    client: QdrantClient | None,
    collection: str,
    tag: str | None,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    delay: float,
    already_done: set[str],
    progress_file: str,
    dry_run: bool,
    verbose: bool,
) -> dict:
    """
    Fetch, parse, chunk, embed, and upsert a single page URL.

    Parameters
    ----------
    url : str
    session : requests.Session
    client : QdrantClient | None
        None in dry-run mode (no actual upsert).
    collection : str
    tag : str | None
    chunk_size : int
    chunk_overlap : int
    batch_size : int
    delay : float
        Polite delay before fetching (caller may have already slept).
    already_done : set[str]
        URLs already ingested (loaded from progress file on startup).
    progress_file : str
    dry_run : bool
    verbose : bool

    Returns
    -------
    dict with keys:
        status : str      — "skipped", "ok", or "error"
        chunks : int      — number of chunks upserted
        message : str     — human-readable status detail
    """
    norm_url = normalise_url(url)

    if norm_url in already_done:
        logger.debug("Already done, skipping: %s", norm_url)
        return {"status": "skipped", "chunks": 0, "message": "already ingested"}

    html = fetch_page(norm_url, session)
    if html is None:
        return {"status": "error", "chunks": 0, "message": "fetch failed"}

    page = extract_page_content(html, norm_url)
    if not page["text"]:
        logger.warning("No extractable text from %s — skipping", norm_url)
        return {"status": "error", "chunks": 0, "message": "no text extracted"}

    chunks = chunk_text(page["text"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        logger.warning("No chunks produced for %s — skipping", norm_url)
        return {"status": "error", "chunks": 0, "message": "no chunks produced"}

    if verbose:
        logger.info(
            "  %s | title=%r | section=%r | %d chunks",
            norm_url,
            page["title"][:60],
            page["section"][:60],
            len(chunks),
        )

    if dry_run:
        logger.info(
            "[dry-run] %s → %d chunks (would embed + upsert)", norm_url, len(chunks)
        )
        return {"status": "ok", "chunks": len(chunks), "message": "dry-run"}

    # Embed in batches (respects batch_size for the embedding service)
    all_dense: list[np.ndarray] = []
    all_sparse: list[SparseVector] = []
    chunk_texts_list = [c["text"] for c in chunks]

    for batch_start in range(0, len(chunk_texts_list), batch_size):
        batch = chunk_texts_list[batch_start : batch_start + batch_size]
        try:
            d_vecs = encode_dense(batch)
            s_vecs = encode_sparse(batch)
        except Exception as exc:
            logger.error("Embedding failed for %s (batch %d): %s", norm_url, batch_start, exc)
            return {"status": "error", "chunks": 0, "message": f"embedding error: {exc}"}
        all_dense.append(d_vecs)
        all_sparse.extend(s_vecs)

    dense_vecs = np.vstack(all_dense)  # shape (N, 1024)

    try:
        n_upserted = upsert_page_chunks(
            client=client,
            collection=collection,
            chunks=chunks,
            dense_vecs=dense_vecs,
            sparse_vecs=all_sparse,
            source_url=norm_url,
            title=page["title"],
            section=page["section"],
            tag=tag,
            dry_run=False,
            verbose=verbose,
        )
    except Exception as exc:
        logger.error("Upsert failed for %s: %s", norm_url, exc)
        return {"status": "error", "chunks": 0, "message": f"upsert error: {exc}"}

    # Record progress
    append_progress(progress_file, norm_url)
    already_done.add(norm_url)

    return {"status": "ok", "chunks": n_upserted, "message": "ok"}


# ---------------------------------------------------------------------------
# Seed URL collection
# ---------------------------------------------------------------------------


def collect_seed_urls(args: argparse.Namespace) -> list[str]:
    """
    Gather all unique seed URLs from --url, --url-file arguments.

    Blank lines and lines starting with ``#`` in the URL file are ignored.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    list[str]
        Deduplicated list of seed URLs.
    """
    seeds: list[str] = []
    seen: set[str] = set()

    def _add(u: str) -> None:
        u = u.strip()
        if u and u not in seen:
            seen.add(u)
            seeds.append(u)

    if args.url:
        _add(args.url)

    if args.url_file:
        url_file_path = Path(args.url_file)
        if not url_file_path.exists():
            logger.error("URL file not found: %s", args.url_file)
            sys.exit(1)
        with url_file_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    _add(line)

    if not seeds:
        logger.error("No URLs provided. Use --url or --url-file.")
        sys.exit(1)

    return seeds


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:  # noqa: C901  (complexity is inherent to CLI orchestration)
    parser = argparse.ArgumentParser(
        description="Ingest web documentation (HTML pages) into Qdrant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input sources
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--url",
        metavar="URL",
        help="Single seed URL to fetch and ingest",
    )
    input_group.add_argument(
        "--url-file",
        metavar="PATH",
        help="Path to text file with one URL per line (# comments ignored)",
    )

    # Required
    parser.add_argument(
        "--collection",
        required=True,
        metavar="NAME",
        help="Target Qdrant collection name (e.g. docs-rust, docs-anthropic)",
    )

    # Crawl options
    parser.add_argument(
        "--depth",
        type=int,
        default=1,
        metavar="INT",
        help="Crawl depth: 0=single page, 1=follow links from seed, 2=two hops",
    )
    parser.add_argument(
        "--sitemap",
        action="store_true",
        help="Attempt to discover URLs from {seed}/sitemap.xml before crawling",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Seconds to sleep between HTTP requests (polite crawling)",
    )

    # Chunking
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        metavar="INT",
        help="Max tokens per chunk (tiktoken cl100k_base)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=64,
        metavar="INT",
        help="Token overlap between consecutive chunks",
    )

    # Embedding
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="INT",
        help="Number of chunks per embedding batch",
    )

    # Metadata
    parser.add_argument(
        "--tag",
        default=None,
        metavar="STR",
        help="Optional metadata tag stored in payload (e.g. rust, anthropic-api)",
    )

    # Progress
    parser.add_argument(
        "--progress-file",
        default="data/html_ingest_progress.txt",
        metavar="PATH",
        help="File tracking ingested URLs for resume support",
    )

    # Behaviour
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be ingested without embedding or upserting",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Validate chunk parameters
    if args.chunk_overlap >= args.chunk_size:
        parser.error(
            f"--chunk-overlap ({args.chunk_overlap}) must be less than "
            f"--chunk-size ({args.chunk_size})"
        )

    # Load progress
    progress_file = args.progress_file
    already_done: set[str] = load_progress(progress_file)
    logger.info(
        "Progress file: %s — %d URLs already ingested", progress_file, len(already_done)
    )

    # Qdrant client (skip in dry-run)
    client: QdrantClient | None = None
    if not args.dry_run:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
        logger.info("Connected to Qdrant at %s, collection=%s", QDRANT_URL, args.collection)

    # HTTP session
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    # Collect seed URLs
    seed_urls = collect_seed_urls(args)
    logger.info("Seed URLs: %d", len(seed_urls))

    # Stats counters
    total_pages_attempted = 0
    total_pages_ok = 0
    total_pages_skipped = 0
    total_pages_error = 0
    total_chunks = 0

    # Process each seed
    for seed_url in seed_urls:
        logger.info("=" * 70)
        logger.info("Seed: %s (depth=%d)", seed_url, args.depth)

        seed_parsed = urlparse(seed_url)

        # Optional sitemap discovery
        sitemap_urls: list[str] = []
        if args.sitemap:
            sitemap_urls = fetch_sitemap_urls(seed_url, session, seed_parsed)
            logger.info("Sitemap discovered %d URLs", len(sitemap_urls))

        # Build the list of pages to process via BFS crawl
        # When depth=0 and no sitemap, this is just [seed_url] + sitemap_urls
        # When depth>0, crawl() will fetch seed pages to extract child links.
        if args.depth == 0 and not sitemap_urls:
            # No crawling needed — just the seed
            pages_to_process = [normalise_url(seed_url)]
        else:
            logger.info("Discovering pages via crawl (depth=%d)...", args.depth)
            pages_to_process = list(
                crawl(
                    seed_url=seed_url,
                    session=session,
                    depth=args.depth,
                    delay=args.delay,
                    extra_urls=sitemap_urls if sitemap_urls else None,
                )
            )
            logger.info("Crawl complete: %d pages to process", len(pages_to_process))

        # --- Ingest each discovered page ---
        pbar = tqdm(pages_to_process, desc=f"Ingesting {seed_url[:50]}", unit="page")
        for page_url in pbar:
            total_pages_attempted += 1

            result = ingest_url(
                url=page_url,
                session=session,
                client=client,
                collection=args.collection,
                tag=args.tag,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                batch_size=args.batch_size,
                delay=args.delay,
                already_done=already_done,
                progress_file=progress_file,
                dry_run=args.dry_run,
                verbose=args.verbose,
            )

            if result["status"] == "skipped":
                total_pages_skipped += 1
            elif result["status"] == "ok":
                total_pages_ok += 1
                total_chunks += result["chunks"]
            else:
                total_pages_error += 1

            pbar.set_postfix(
                ok=total_pages_ok,
                skip=total_pages_skipped,
                err=total_pages_error,
                chunks=total_chunks,
            )

            # Polite delay between pages (skip after last page)
            if page_url != pages_to_process[-1]:
                time.sleep(args.delay)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print()
    print("=" * 70)
    print("HTML Documentation Ingestion — Summary")
    print("=" * 70)
    print(f"  Collection      : {args.collection}")
    print(f"  Tag             : {args.tag or '(none)'}")
    print(f"  Dry run         : {args.dry_run}")
    print(f"  Progress file   : {progress_file}")
    print()
    print(f"  Pages attempted : {total_pages_attempted}")
    print(f"  Pages OK        : {total_pages_ok}")
    print(f"  Pages skipped   : {total_pages_skipped}  (already ingested)")
    print(f"  Pages errored   : {total_pages_error}")
    print(f"  Chunks upserted : {total_chunks}")
    print("=" * 70)

    if args.dry_run:
        print("\n[dry-run mode — no data was written to Qdrant or progress file]")


if __name__ == "__main__":
    main()
