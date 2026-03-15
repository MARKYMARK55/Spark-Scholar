#!/usr/bin/env python3
"""
ingest/08_expand_citations.py
==============================
Citation graph expansion — automatically discovers, downloads, and indexes the
papers referenced by your existing corpus.

Levels
------
  L1  Your original papers (already ingested)
  L2  Papers cited in L1 papers          (--depth 1)
  L3  Papers cited in L2 papers          (--depth 2)

The script generates a JSON manifest of every discovered reference, attempts to
download open-access PDFs, and ingests them into Qdrant using the same
dense + sparse + HDBSCAN pipeline as ingest/05_ingest_pdfs.py.

Reference resolution strategy
------------------------------
1.  Semantic Scholar API — primary source. Returns structured references with
    arXiv IDs, DOIs, open-access PDF URLs, and citation counts. Works for
    almost all arXiv papers and most major venues (NeurIPS, ICML, ACL, Nature).
2.  arXiv API — used to download PDFs once we have arXiv IDs.
3.  Unpaywall API — finds open-access PDFs for papers not on arXiv
    (journal papers, conference proceedings). Requires an email (--email).
4.  PDF text extraction — last resort: parses the references section of
    the source PDF directly, then searches S2 by title.

Output structure
----------------
    output_dir/
    ├── manifests/
    │   ├── L2_<arxiv_id>.json     # all references of one source paper
    │   └── L3_<arxiv_id>.json     # all references of one L2 paper
    ├── L2/
    │   └── <arxiv_id>/
    │       └── <ref_arxiv_id>.pdf
    └── L3/
        └── <arxiv_id>/
            └── <ref_arxiv_id>.pdf

Manifest JSON schema (per paper)
---------------------------------
    {
      "source": {
        "arxiv_id": "2303.08774",
        "title": "GPT-4 Technical Report",
        "authors": ["OpenAI"],
        "year": 2023,
        "semantic_scholar_id": "..."
      },
      "level": "L2",
      "fetched_at": "2026-03-12T10:00:00Z",
      "references": [
        {
          "title": "...",
          "authors": ["..."],
          "year": 2020,
          "arxiv_id": "2005.14165",
          "doi": "...",
          "semantic_scholar_id": "...",
          "open_access_pdf": "https://...",
          "citation_count": 5000,
          "availability": "arxiv",   # arxiv | unpaywall | none
          "downloaded": true,
          "ingested": true,
          "collection": "arxiv-cs-ml-ai"
        }
      ]
    }

Usage
-----
    # Expand a single arXiv paper to L2 + L3
    python ingest/08_expand_citations.py \\
        --arxiv 2303.08774 \\
        --depth 2 \\
        --output-dir data/citations/

    # Expand multiple papers from a text file (one arXiv ID per line)
    python ingest/08_expand_citations.py \\
        --arxiv-file my_papers.txt \\
        --depth 2 \\
        --output-dir data/citations/

    # Expand from a directory of PDFs (arXiv IDs extracted from filenames)
    python ingest/08_expand_citations.py \\
        --input-dir data/papers/ \\
        --depth 1 \\
        --output-dir data/citations/

    # JSON manifest only — no downloading, no ingestion
    python ingest/08_expand_citations.py \\
        --arxiv 2303.08774 \\
        --json-only \\
        --output-dir data/citations/

    # Quality filter: only follow highly-cited references
    python ingest/08_expand_citations.py \\
        --arxiv 2303.08774 \\
        --depth 2 \\
        --min-citations 50 \\
        --max-per-paper 30 \\
        --output-dir data/citations/

    # Resume an interrupted expansion (already-processed papers are skipped)
    python ingest/08_expand_citations.py \\
        --arxiv 2303.08774 \\
        --depth 2 \\
        --output-dir data/citations/   # same dir as before — auto-resumes

Requirements
------------
    pip install httpx pymupdf tiktoken tqdm qdrant-client python-dotenv
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
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../env/.env"), override=False)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.embeddings import encode_dense, encode_sparse
from pipeline.router import route_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))

S2_BASE = "https://api.semanticscholar.org/graph/v1"
ARXIV_PDF_BASE = "https://arxiv.org/pdf"
UNPAYWALL_BASE = "https://api.unpaywall.org/v2"

# Polite delay between HTTP requests (seconds)
REQUEST_DELAY = 1.0

# Fields to request from Semantic Scholar
S2_PAPER_FIELDS = "title,authors,year,externalIds,openAccessPdf,citationCount,abstract"
S2_REF_FIELDS = (
    "title,authors,year,externalIds,openAccessPdf,citationCount,abstract,"
    "citedPaper.title,citedPaper.authors,citedPaper.year,"
    "citedPaper.externalIds,citedPaper.openAccessPdf,citedPaper.citationCount"
)

# ---------------------------------------------------------------------------
# Semantic Scholar helpers
# ---------------------------------------------------------------------------

def _s2_headers() -> dict:
    h = {"Accept": "application/json"}
    if SEMANTIC_SCHOLAR_API_KEY:
        h["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    return h


def s2_get_paper(arxiv_id: str, client: httpx.Client) -> Optional[dict]:
    """Fetch paper metadata from Semantic Scholar by arXiv ID."""
    url = f"{S2_BASE}/paper/arXiv:{arxiv_id}"
    try:
        resp = client.get(url, params={"fields": S2_PAPER_FIELDS}, headers=_s2_headers(), timeout=20)
        if resp.status_code == 404:
            logger.debug("S2 paper not found: %s", arxiv_id)
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning("S2 paper fetch failed for %s: %s", arxiv_id, e)
        return None


def s2_get_references(arxiv_id: str, client: httpx.Client, limit: int = 100) -> list[dict]:
    """Fetch structured references for a paper from Semantic Scholar."""
    url = f"{S2_BASE}/paper/arXiv:{arxiv_id}/references"
    all_refs: list[dict] = []
    offset = 0
    while True:
        try:
            resp = client.get(
                url,
                params={"fields": S2_REF_FIELDS, "limit": min(limit, 100), "offset": offset},
                headers=_s2_headers(),
                timeout=20,
            )
            if resp.status_code == 404:
                logger.debug("S2 references not found: %s", arxiv_id)
                break
            resp.raise_for_status()
            data = resp.json()
            batch = data.get("data", [])
            all_refs.extend(batch)
            if len(batch) < 100 or len(all_refs) >= limit:
                break
            offset += 100
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            logger.warning("S2 references fetch failed for %s at offset %d: %s", arxiv_id, offset, e)
            break
    return all_refs[:limit]


def s2_search_by_title(title: str, client: httpx.Client) -> Optional[dict]:
    """Search Semantic Scholar by title — used for non-arXiv papers found via PDF parsing."""
    try:
        resp = client.get(
            f"{S2_BASE}/paper/search",
            params={"query": title, "fields": S2_PAPER_FIELDS, "limit": 1},
            headers=_s2_headers(),
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        papers = data.get("data", [])
        return papers[0] if papers else None
    except Exception as e:
        logger.debug("S2 title search failed for '%s': %s", title[:60], e)
        return None


# ---------------------------------------------------------------------------
# Reference parsing from S2 response
# ---------------------------------------------------------------------------

def parse_s2_reference(ref_entry: dict) -> dict:
    """Normalise a Semantic Scholar reference entry into our schema."""
    # S2 wraps the actual paper in 'citedPaper'
    paper = ref_entry.get("citedPaper") or ref_entry
    ext_ids = paper.get("externalIds") or {}
    oa = paper.get("openAccessPdf") or {}

    authors = [a.get("name", "") for a in (paper.get("authors") or [])]

    return {
        "title": paper.get("title", ""),
        "authors": authors[:10],  # cap at 10
        "year": paper.get("year"),
        "arxiv_id": ext_ids.get("ArXiv", ""),
        "doi": ext_ids.get("DOI", ""),
        "semantic_scholar_id": paper.get("paperId", ""),
        "open_access_pdf": oa.get("url", ""),
        "citation_count": paper.get("citationCount", 0) or 0,
        "abstract": (paper.get("abstract") or "")[:500],
        "availability": "",   # filled in later
        "downloaded": False,
        "ingested": False,
        "collection": "",
    }


# ---------------------------------------------------------------------------
# PDF reference extraction (fallback for non-S2 papers)
# ---------------------------------------------------------------------------

_ARXIV_PATTERN = re.compile(r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b")


def extract_arxiv_ids_from_pdf(pdf_path: Path) -> list[str]:
    """
    Extract arXiv IDs mentioned anywhere in a PDF (references section or body).
    Uses PyMuPDF for text extraction.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.debug("PyMuPDF not installed — skipping PDF reference extraction")
        return []

    ids: set[str] = set()
    try:
        doc = fitz.open(str(pdf_path))
        for page in doc:
            text = page.get_text()
            for match in _ARXIV_PATTERN.finditer(text):
                raw = match.group(1)
                # Normalise: strip version suffix
                ids.add(raw.split("v")[0])
        doc.close()
    except Exception as e:
        logger.warning("PDF text extraction failed for %s: %s", pdf_path.name, e)
    return list(ids)


def extract_reference_titles_from_pdf(pdf_path: Path) -> list[str]:
    """
    Extract reference titles from the last N pages of a PDF.
    Very heuristic — used only as a last resort.
    """
    try:
        import fitz
    except ImportError:
        return []

    titles = []
    try:
        doc = fitz.open(str(pdf_path))
        # References are usually in the last 10% of pages
        start_page = max(0, len(doc) - max(3, len(doc) // 10))
        text = ""
        for page_num in range(start_page, len(doc)):
            text += doc[page_num].get_text()
        doc.close()

        # Simple heuristic: numbered or bracketed references
        # Match [1] Author, Title ... or 1. Author, Title ...
        patterns = [
            re.compile(r"\[\d+\]\s+(.+?)(?=\[\d+\]|\Z)", re.DOTALL),
            re.compile(r"^\d+\.\s+(.+?)(?=^\d+\.|\Z)", re.MULTILINE | re.DOTALL),
        ]
        for pattern in patterns:
            for m in pattern.finditer(text):
                entry = m.group(1).strip().replace("\n", " ")
                if len(entry) > 30:
                    titles.append(entry[:200])
            if titles:
                break
    except Exception as e:
        logger.debug("Reference title extraction failed: %s", e)
    return titles[:100]


# ---------------------------------------------------------------------------
# arXiv ID resolution from filenames
# ---------------------------------------------------------------------------

def arxiv_id_from_filename(filename: str) -> str:
    """
    Try to extract an arXiv ID from a PDF filename.
    Common patterns:
      2303.08774.pdf
      2303.08774v2.pdf
      arxiv_2303.08774.pdf
      GPT4_2303.08774.pdf
    """
    stem = Path(filename).stem
    m = _ARXIV_PATTERN.search(stem)
    if m:
        return m.group(1).split("v")[0]
    return ""


# ---------------------------------------------------------------------------
# PDF downloading
# ---------------------------------------------------------------------------

def download_arxiv_pdf(arxiv_id: str, dest: Path, client: httpx.Client) -> bool:
    """Download a PDF from arXiv. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 10_000:
        logger.debug("Already downloaded: %s", dest.name)
        return True
    url = f"{ARXIV_PDF_BASE}/{arxiv_id}"
    try:
        resp = client.get(url, follow_redirects=True, timeout=60)
        resp.raise_for_status()
        if b"%PDF" not in resp.content[:8]:
            logger.debug("Response for %s is not a PDF", arxiv_id)
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        logger.debug("Downloaded arXiv:%s → %s", arxiv_id, dest.name)
        return True
    except Exception as e:
        logger.debug("Download failed for arXiv:%s — %s", arxiv_id, e)
        return False


def download_url_pdf(url: str, dest: Path, client: httpx.Client) -> bool:
    """Download a PDF from an arbitrary open-access URL."""
    if dest.exists() and dest.stat().st_size > 10_000:
        return True
    try:
        resp = client.get(url, follow_redirects=True, timeout=60)
        resp.raise_for_status()
        if b"%PDF" not in resp.content[:8]:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        logger.debug("Downloaded OA PDF → %s", dest.name)
        return True
    except Exception as e:
        logger.debug("OA PDF download failed %s — %s", url[:60], e)
        return False


def unpaywall_lookup(doi: str, email: str, client: httpx.Client) -> str:
    """Look up an open-access PDF URL via Unpaywall for a given DOI."""
    if not doi or not email:
        return ""
    try:
        resp = client.get(
            f"{UNPAYWALL_BASE}/{doi}",
            params={"email": email},
            timeout=15,
        )
        if resp.status_code == 404:
            return ""
        resp.raise_for_status()
        data = resp.json()
        oa_loc = data.get("best_oa_location") or {}
        return oa_loc.get("url_for_pdf", "") or oa_loc.get("url", "")
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Qdrant ingestion
# ---------------------------------------------------------------------------

def ingest_pdf_to_qdrant(
    pdf_path: Path,
    collection: str,
    metadata: dict,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> int:
    """
    Chunk, embed, and upsert a single PDF into Qdrant.
    Returns the number of chunks ingested (0 on failure).
    """
    try:
        import fitz
        import tiktoken
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct, SparseVector as QSparseVector
    except ImportError as e:
        logger.error("Missing dependency for ingestion: %s", e)
        return 0

    # --- Extract text
    try:
        doc = fitz.open(str(pdf_path))
        full_text = "\n".join(page.get_text() for page in doc)
        doc.close()
    except Exception as e:
        logger.warning("PDF text extraction failed for %s: %s", pdf_path.name, e)
        return 0

    if not full_text.strip():
        logger.warning("No text extracted from %s — skipping", pdf_path.name)
        return 0

    # --- Chunk
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(full_text)
    chunks: list[str] = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start : start + chunk_size]
        if len(chunk_tokens) < 20:
            continue
        chunks.append(enc.decode(chunk_tokens))

    if not chunks:
        return 0

    # --- Embed + upsert
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    points: list[PointStruct] = []

    for idx, chunk_text in enumerate(chunks):
        try:
            dense_vec = encode_dense(chunk_text)
            sparse_result = encode_sparse(chunk_text)
        except Exception as e:
            logger.warning("Embedding failed for chunk %d of %s: %s", idx, pdf_path.name, e)
            continue

        # Deterministic point ID
        point_id = str(uuid.UUID(
            bytes=hashlib.sha256(f"{pdf_path.name}::{idx}".encode()).digest()[:16]
        ))

        payload = {
            "chunk_text": chunk_text,
            "chunk_idx": idx,
            "source_file": pdf_path.name,
            "collection": collection,
            "type": "citation_chunk",
            "source_type": "pdf_citation_expansion",
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            **{k: v for k, v in metadata.items() if v},
        }

        points.append(PointStruct(
            id=point_id,
            vector={
                "dense": dense_vec,
                "sparse": QSparseVector(
                    indices=sparse_result.get("indices", []),
                    values=sparse_result.get("values", []),
                ),
            },
            payload=payload,
        ))

    if not points:
        return 0

    try:
        client.upsert(collection_name=collection, points=points, wait=True)
    except Exception as e:
        logger.error("Qdrant upsert failed for %s: %s", pdf_path.name, e)
        return 0

    return len(points)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path) -> dict:
    if manifest_path.exists():
        try:
            return json.loads(manifest_path.read_text())
        except Exception:
            pass
    return {}


def save_manifest(manifest_path: Path, data: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def progress_key(arxiv_id: str, level: int) -> str:
    return f"L{level+1}:{arxiv_id}"


# ---------------------------------------------------------------------------
# Core expansion logic
# ---------------------------------------------------------------------------

def expand_paper(
    arxiv_id: str,
    level: int,                    # 1 = L2, 2 = L3
    output_dir: Path,
    http_client: httpx.Client,
    args: argparse.Namespace,
    progress: dict,                # mutable — updated in place
    source_pdf: Optional[Path] = None,
) -> list[str]:
    """
    Expand a single paper: fetch references → download PDFs → ingest.
    Returns list of arXiv IDs for downloaded references (used for next-level expansion).
    """
    level_label = f"L{level + 1}"
    pkey = progress_key(arxiv_id, level)

    if pkey in progress and progress[pkey].get("complete"):
        logger.info("  ↩ Already processed %s (%s) — skipping", arxiv_id, level_label)
        return progress[pkey].get("downloaded_ids", [])

    logger.info("▶ Expanding %s → %s", arxiv_id, level_label)

    manifest_path = output_dir / "manifests" / f"{level_label}_{arxiv_id.replace('/', '_')}.json"
    manifest = load_manifest(manifest_path)

    # --- Fetch paper metadata from S2
    if not manifest.get("source"):
        paper_meta = s2_get_paper(arxiv_id, http_client)
        time.sleep(REQUEST_DELAY)
        source_info = {
            "arxiv_id": arxiv_id,
            "title": (paper_meta or {}).get("title", ""),
            "authors": [(a.get("name", "")) for a in ((paper_meta or {}).get("authors") or [])][:5],
            "year": (paper_meta or {}).get("year"),
            "semantic_scholar_id": (paper_meta or {}).get("paperId", ""),
        }
        manifest = {
            "source": source_info,
            "level": level_label,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "references": [],
        }

    # --- Fetch references from S2
    if not manifest.get("references"):
        logger.info("  Fetching references from Semantic Scholar for %s...", arxiv_id)
        s2_refs = s2_get_references(arxiv_id, http_client, limit=args.max_per_paper * 2)
        time.sleep(REQUEST_DELAY)

        if s2_refs:
            refs = [parse_s2_reference(r) for r in s2_refs]
            # Filter out self and empty entries
            refs = [r for r in refs if r.get("title") and r.get("arxiv_id") != arxiv_id]
            manifest["references"] = refs
            logger.info("  Found %d references via Semantic Scholar", len(refs))
        else:
            # Fallback: extract arXiv IDs from PDF text
            if source_pdf and source_pdf.exists():
                logger.info("  S2 found nothing — falling back to PDF text extraction")
                raw_ids = extract_arxiv_ids_from_pdf(source_pdf)
                # Exclude the source itself
                raw_ids = [i for i in raw_ids if i != arxiv_id]
                refs = []
                for raw_id in raw_ids[:args.max_per_paper * 2]:
                    # Look up each ID in S2 to get metadata
                    paper = s2_get_paper(raw_id, http_client)
                    time.sleep(0.3)
                    if paper:
                        ext_ids = paper.get("externalIds") or {}
                        oa = paper.get("openAccessPdf") or {}
                        refs.append({
                            "title": paper.get("title", ""),
                            "authors": [(a.get("name", "")) for a in (paper.get("authors") or [])][:5],
                            "year": paper.get("year"),
                            "arxiv_id": ext_ids.get("ArXiv", raw_id),
                            "doi": ext_ids.get("DOI", ""),
                            "semantic_scholar_id": paper.get("paperId", ""),
                            "open_access_pdf": oa.get("url", ""),
                            "citation_count": paper.get("citationCount", 0) or 0,
                            "abstract": (paper.get("abstract") or "")[:500],
                            "availability": "",
                            "downloaded": False,
                            "ingested": False,
                            "collection": "",
                        })
                manifest["references"] = refs
                logger.info("  Found %d references via PDF text extraction", len(refs))
            else:
                logger.warning("  No references found for %s", arxiv_id)
                manifest["references"] = []

        save_manifest(manifest_path, manifest)

    # Apply quality filter and cap
    refs = manifest.get("references", [])
    if args.min_citations > 0:
        refs = [r for r in refs if r.get("citation_count", 0) >= args.min_citations]
    refs = sorted(refs, key=lambda r: r.get("citation_count", 0), reverse=True)
    refs = refs[: args.max_per_paper]

    logger.info("  Processing %d references (after quality filter)", len(refs))

    # --- Download + ingest
    downloaded_ids: list[str] = []
    pdf_dir = output_dir / level_label / arxiv_id.replace("/", "_")
    pdf_dir.mkdir(parents=True, exist_ok=True)

    for ref in tqdm(refs, desc=f"  {level_label} refs for {arxiv_id}", leave=False):
        ref_arxiv = ref.get("arxiv_id", "")
        ref_title = ref.get("title", "unknown")[:60]

        # Determine availability
        if not ref.get("availability"):
            if ref_arxiv:
                ref["availability"] = "arxiv"
            elif ref.get("open_access_pdf"):
                ref["availability"] = "unpaywall"
            elif ref.get("doi") and args.email:
                oa_url = unpaywall_lookup(ref["doi"], args.email, http_client)
                if oa_url:
                    ref["open_access_pdf"] = oa_url
                    ref["availability"] = "unpaywall"
                else:
                    ref["availability"] = "none"
            else:
                ref["availability"] = "none"

        if args.json_only or ref["availability"] == "none":
            continue

        # Download PDF
        if not ref.get("downloaded"):
            pdf_stem = (ref_arxiv or hashlib.md5(ref_title.encode()).hexdigest()[:8]).replace("/", "_")
            pdf_path = pdf_dir / f"{pdf_stem}.pdf"

            if ref["availability"] == "arxiv" and ref_arxiv:
                ok = download_arxiv_pdf(ref_arxiv, pdf_path, http_client)
            else:
                ok = download_url_pdf(ref["open_access_pdf"], pdf_path, http_client)
            time.sleep(REQUEST_DELAY)

            ref["downloaded"] = ok
            ref["local_path"] = str(pdf_path) if ok else ""

        # Ingest into Qdrant
        if ref.get("downloaded") and not ref.get("ingested") and not args.no_ingest:
            pdf_path = Path(ref["local_path"])
            # Route to collection based on title + abstract
            query_text = f"{ref.get('title', '')} {ref.get('abstract', '')}"
            collection = args.collection or route_query(query_text)
            ref["collection"] = collection

            n_chunks = ingest_pdf_to_qdrant(
                pdf_path,
                collection=collection,
                metadata={
                    "arxiv_id": ref_arxiv,
                    "title": ref.get("title", ""),
                    "authors": ref.get("authors", []),
                    "year": ref.get("year"),
                    "doi": ref.get("doi", ""),
                    "source_arxiv_id": arxiv_id,
                    "expansion_level": level_label,
                    "citation_count": ref.get("citation_count", 0),
                },
            )
            ref["ingested"] = n_chunks > 0
            ref["chunks_ingested"] = n_chunks
            if n_chunks:
                logger.debug("  Ingested %s → %s (%d chunks)", ref_title, collection, n_chunks)

        if ref.get("downloaded") and ref_arxiv:
            downloaded_ids.append(ref_arxiv)

    # Save updated manifest
    manifest["references"] = refs
    save_manifest(manifest_path, manifest)

    # Mark as complete in progress
    progress[pkey] = {"complete": True, "downloaded_ids": downloaded_ids}

    n_downloaded = sum(1 for r in refs if r.get("downloaded"))
    n_ingested = sum(1 for r in refs if r.get("ingested"))
    n_no_access = sum(1 for r in refs if r.get("availability") == "none")
    logger.info(
        "  ✓ %s: %d refs | %d downloaded | %d ingested | %d no open access",
        arxiv_id, len(refs), n_downloaded, n_ingested, n_no_access,
    )

    return downloaded_ids


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(output_dir: Path, root_ids: list[str], depth: int) -> None:
    print("\n" + "━" * 68)
    print("  Citation Expansion Summary")
    print("━" * 68)

    for level in range(1, depth + 1):
        level_label = f"L{level + 1}"
        manifests = list((output_dir / "manifests").glob(f"{level_label}_*.json"))
        total_refs = downloaded = ingested = no_access = 0
        for m_path in manifests:
            try:
                data = json.loads(m_path.read_text())
                refs = data.get("references", [])
                total_refs += len(refs)
                downloaded += sum(1 for r in refs if r.get("downloaded"))
                ingested += sum(1 for r in refs if r.get("ingested"))
                no_access += sum(1 for r in refs if r.get("availability") == "none")
            except Exception:
                pass
        print(f"\n  {level_label} ({len(manifests)} source papers processed):")
        print(f"    References found  : {total_refs}")
        print(f"    PDFs downloaded   : {downloaded}")
        print(f"    Chunks ingested   : {ingested}")
        print(f"    No open access    : {no_access}")

    print(f"\n  Manifests written to : {output_dir / 'manifests'}")
    pdf_dirs = [output_dir / f"L{l+1}" for l in range(1, depth + 1)]
    for d in pdf_dirs:
        if d.exists():
            pdf_count = len(list(d.rglob("*.pdf")))
            print(f"  PDFs stored in       : {d}  ({pdf_count} files)")
    print("━" * 68)


# ---------------------------------------------------------------------------
# Argument parsing + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Citation graph expansion — discover, download, and index referenced papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Input sources (at least one required)
    src = ap.add_argument_group("input sources (at least one required)")
    src.add_argument("--arxiv", nargs="+", metavar="ID",
                     help="One or more arXiv IDs (e.g. 2303.08774)")
    src.add_argument("--arxiv-file", metavar="FILE",
                     help="Text file with one arXiv ID per line")
    src.add_argument("--input-dir", metavar="DIR",
                     help="Directory of PDFs — arXiv IDs extracted from filenames")

    # --- Output
    ap.add_argument("--output-dir", default="data/citations", metavar="DIR",
                    help="Directory for manifests and downloaded PDFs (default: data/citations)")

    # --- Expansion settings
    ap.add_argument("--depth", type=int, default=2, choices=[1, 2],
                    help="Expansion depth: 1=L2 only, 2=L2+L3 (default: 2)")
    ap.add_argument("--max-per-paper", type=int, default=20, metavar="N",
                    help="Max references to follow per paper (default: 20)")
    ap.add_argument("--min-citations", type=int, default=0, metavar="N",
                    help="Only follow references with ≥N citations (default: 0 = all)")

    # --- Ingestion
    ap.add_argument("--collection", metavar="NAME",
                    help="Force all papers into this collection (default: auto-route)")
    ap.add_argument("--no-ingest", action="store_true",
                    help="Download PDFs but do not ingest into Qdrant")
    ap.add_argument("--json-only", action="store_true",
                    help="Only generate citation JSON manifests — no downloading, no ingestion")

    # --- API / network
    ap.add_argument("--email", default=os.environ.get("CROSSREF_MAILTO", ""),
                    metavar="EMAIL",
                    help="Email for Unpaywall API (finds OA PDFs for non-arXiv papers)")
    ap.add_argument("--delay", type=float, default=REQUEST_DELAY, metavar="SEC",
                    help="Delay between HTTP requests in seconds (default: 1.0)")

    ap.add_argument("--verbose", action="store_true", help="Enable debug logging")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    global REQUEST_DELAY
    REQUEST_DELAY = args.delay

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Collect root arXiv IDs
    root_ids: list[str] = []

    if args.arxiv:
        for raw in args.arxiv:
            aid = raw.strip().rstrip("/").split("/")[-1].replace("abs/", "").split("v")[0]
            if aid:
                root_ids.append(aid)

    if args.arxiv_file:
        p = Path(args.arxiv_file)
        if not p.exists():
            logger.error("arXiv ID file not found: %s", p)
            sys.exit(1)
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                root_ids.append(line.split()[0])

    if args.input_dir:
        d = Path(args.input_dir)
        if not d.is_dir():
            logger.error("Input directory not found: %s", d)
            sys.exit(1)
        for pdf in sorted(d.glob("*.pdf")):
            aid = arxiv_id_from_filename(pdf.name)
            if aid:
                root_ids.append(aid)
            else:
                logger.warning("Could not extract arXiv ID from filename: %s", pdf.name)

    root_ids = list(dict.fromkeys(root_ids))  # deduplicate, preserve order
    if not root_ids:
        logger.error("No arXiv IDs found. Use --arxiv, --arxiv-file, or --input-dir.")
        sys.exit(1)

    logger.info("Root papers (L1): %d", len(root_ids))
    for rid in root_ids:
        logger.info("  • arXiv:%s", rid)

    # --- Load progress
    progress_file = output_dir / "expansion_progress.json"
    progress: dict = {}
    if progress_file.exists():
        try:
            progress = json.loads(progress_file.read_text())
            logger.info("Loaded progress file — %d already processed", len(progress))
        except Exception:
            pass

    def save_progress():
        progress_file.write_text(json.dumps(progress, indent=2))

    # --- HTTP client (shared across all requests)
    transport = httpx.HTTPTransport(retries=3)
    with httpx.Client(transport=transport, follow_redirects=True) as http_client:

        # ── Level 1 expansion (L1 → L2) ─────────────────────────────────────
        l2_ids: list[str] = []
        print(f"\n{'━'*68}")
        print(f"  Expanding L1 → L2  ({len(root_ids)} source papers)")
        print(f"{'━'*68}")

        for arxiv_id in tqdm(root_ids, desc="L1→L2"):
            # Find source PDF if available (for fallback text extraction)
            source_pdf: Optional[Path] = None
            if args.input_dir:
                candidates = list(Path(args.input_dir).glob(f"*{arxiv_id}*.pdf"))
                if candidates:
                    source_pdf = candidates[0]

            downloaded = expand_paper(
                arxiv_id=arxiv_id,
                level=1,
                output_dir=output_dir,
                http_client=http_client,
                args=args,
                progress=progress,
                source_pdf=source_pdf,
            )
            l2_ids.extend(downloaded)
            save_progress()

        l2_ids = list(dict.fromkeys(l2_ids))  # deduplicate
        logger.info("L2 expansion complete: %d unique papers downloaded", len(l2_ids))

        # ── Level 2 expansion (L2 → L3) ─────────────────────────────────────
        if args.depth >= 2 and l2_ids and not args.json_only:
            print(f"\n{'━'*68}")
            print(f"  Expanding L2 → L3  ({len(l2_ids)} source papers)")
            print(f"{'━'*68}")

            for arxiv_id in tqdm(l2_ids, desc="L2→L3"):
                # L2 PDFs are stored in output_dir/L2/<root_id>/
                l2_pdf: Optional[Path] = None
                for candidate in (output_dir / "L2").rglob(f"*{arxiv_id.replace('/', '_')}*.pdf"):
                    l2_pdf = candidate
                    break

                expand_paper(
                    arxiv_id=arxiv_id,
                    level=2,
                    output_dir=output_dir,
                    http_client=http_client,
                    args=args,
                    progress=progress,
                    source_pdf=l2_pdf,
                )
                save_progress()

            logger.info("L3 expansion complete")

    # --- Final summary
    print_summary(output_dir, root_ids, args.depth)


if __name__ == "__main__":
    main()
