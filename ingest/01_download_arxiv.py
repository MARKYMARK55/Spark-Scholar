#!/usr/bin/env python3
"""
ingest/01_download_arxiv.py
===========================
Download (if needed) and process the arXiv metadata dataset into clean JSONL.

Download source: `Cornell-University/arxiv` on HuggingFace Hub (~4.8 GB)
Local cache:     ~/RAG/hf_datasets/arxiv/arxiv-metadata-latest.jsonl

If the raw file already exists at --raw-source, the HuggingFace download is
skipped entirely and we go straight to the conversion step.  This means
re-running the script is always safe — it will never re-download 4.8 GB you
already have.

Output files (written to --output-dir, default ~/RAG/arxiv/):
  arxiv_raw.jsonl            — all records passing --min-year, normalised fields
  arxiv_with_abstract.jsonl  — subset of above with non-empty abstracts (>50 chars)
                               ← this is the file consumed by 03_ingest_dense.py
                                 and 04_ingest_sparse.py

Fields in output records:
  arxiv_id, title, abstract, authors, categories, year, update_date, journal_ref

Usage
-----
    # Normal run — skips download if raw file already present
    python ingest/01_download_arxiv.py

    # Point at a different raw source (e.g. older snapshot)
    python ingest/01_download_arxiv.py --raw-source ~/RAG/hf_datasets/arxiv/arxiv-metadata-oai-snapshot.json

    # Override output location
    python ingest/01_download_arxiv.py --output-dir ~/RAG/arxiv/

    # Test with a small slice
    python ingest/01_download_arxiv.py --max-records 10000

Requirements
------------
    pip install datasets huggingface_hub tqdm python-dotenv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../env/.env"), override=False)

logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATASET_NAME = "Cornell-University/arxiv"

# Where the raw HuggingFace JSONL lives on this machine.
# The download step writes here; if it already exists, download is skipped.
DEFAULT_RAW_SOURCE = os.path.expanduser("~/RAG/hf_datasets/arxiv/arxiv-metadata-latest.jsonl")

# Where processed output files are written.
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/RAG/arxiv")


def _parse_year(update_date: str) -> int | None:
    """Extract year from update_date string like '2023-01-15'."""
    if not update_date:
        return None
    try:
        return int(update_date[:4])
    except (ValueError, IndexError):
        return None


def _clean_authors(authors_str: str) -> str:
    """Normalise the author string — remove extra whitespace."""
    if not authors_str:
        return ""
    return " ".join(authors_str.split())


def _download_from_hf(raw_path: Path) -> None:
    """
    Stream the arXiv dataset from HuggingFace Hub and write raw JSONL.

    Only called when raw_path does not already exist.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed.  Run: pip install datasets huggingface_hub")
        sys.exit(1)

    raw_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s from HuggingFace (streaming ~4.8 GB)…", DATASET_NAME)
    logger.info("Output → %s", raw_path)

    dataset = load_dataset(
        DATASET_NAME,
        trust_remote_code=True,
        token=HF_TOKEN if HF_TOKEN else None,
        streaming=True,
    )

    written = 0
    with open(raw_path, "w", encoding="utf-8") as f:
        for record in tqdm(dataset["train"], desc="Downloading", unit="rec"):
            f.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
            written += 1

    logger.info("Download complete — %d records written to %s", written, raw_path)


def process_arxiv(
    raw_source: str = DEFAULT_RAW_SOURCE,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    min_year: int = 2000,
    max_records: int | None = None,
) -> tuple[Path, Path]:
    """
    Convert raw arXiv JSONL into normalised output files.

    If raw_source does not exist, it is downloaded from HuggingFace first.

    Parameters
    ----------
    raw_source : str
        Path to the raw HuggingFace JSONL.  Downloaded here if absent.
    output_dir : str
        Directory to write arxiv_raw.jsonl and arxiv_with_abstract.jsonl.
    min_year : int
        Only include papers from this year onward.
    max_records : int, optional
        Cap total records processed (useful for smoke-testing).

    Returns
    -------
    (raw_path, filtered_path)
    """
    raw_source_path = Path(raw_source)
    output_path = Path(output_dir)

    # ── Step 1: ensure raw source exists ────────────────────────────────────
    if raw_source_path.exists():
        size_gb = raw_source_path.stat().st_size / 1e9
        logger.info("Raw source found: %s (%.1f GB) — skipping download", raw_source_path, size_gb)
    else:
        logger.info("Raw source not found at %s — downloading from HuggingFace…", raw_source_path)
        _download_from_hf(raw_source_path)

    # ── Step 2: convert ─────────────────────────────────────────────────────
    output_path.mkdir(parents=True, exist_ok=True)
    raw_out_path = output_path / "arxiv_raw.jsonl"
    filtered_path = output_path / "arxiv_with_abstract.jsonl"

    logger.info("Processing %s → %s", raw_source_path, output_path)

    total_written = 0
    total_filtered = 0
    skipped_year = 0
    skipped_abstract = 0

    with (
        open(raw_source_path, "r", encoding="utf-8") as src,
        open(raw_out_path, "w", encoding="utf-8") as raw_f,
        open(filtered_path, "w", encoding="utf-8") as filt_f,
    ):
        pbar = tqdm(src, desc="Processing arXiv records", unit="rec")
        for line in pbar:
            if max_records and total_written >= max_records:
                break

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # HF raw uses "id"; normalise to "arxiv_id"
            arxiv_id = (record.get("id") or record.get("arxiv_id") or "").strip()
            title = (record.get("title") or "").strip().replace("\n", " ")
            abstract = (record.get("abstract") or "").strip().replace("\n", " ")
            authors = _clean_authors(record.get("authors") or "")
            categories = (record.get("categories") or "").strip()
            update_date = record.get("update_date") or ""
            journal_ref = record.get("journal-ref") or record.get("journal_ref") or ""

            year = _parse_year(update_date)

            if min_year and year and year < min_year:
                skipped_year += 1
                continue

            doc = {
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "categories": categories,
                "year": year,
                "update_date": update_date,
                "journal_ref": journal_ref,
            }

            raw_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            total_written += 1

            if abstract and len(abstract) > 50:
                filt_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                total_filtered += 1
            else:
                skipped_abstract += 1

            if total_written % 100_000 == 0:
                pbar.set_postfix(written=total_written, filtered=total_filtered)

    logger.info("=" * 60)
    logger.info("Processing complete:")
    logger.info("  Total records processed:         %d", total_written)
    logger.info("  Filtered (with abstract):        %d", total_filtered)
    logger.info("  Skipped (before %d):             %d", min_year, skipped_year)
    logger.info("  Skipped (no/short abstract):     %d", skipped_abstract)
    logger.info("  Raw output:      %s", raw_out_path)
    logger.info("  Filtered output: %s", filtered_path)
    logger.info("=" * 60)

    return raw_out_path, filtered_path


def main():
    parser = argparse.ArgumentParser(
        description="Download (if needed) and process arXiv metadata into clean JSONL"
    )
    parser.add_argument(
        "--raw-source",
        default=DEFAULT_RAW_SOURCE,
        help=f"Path to raw HF JSONL (downloaded here if absent). Default: {DEFAULT_RAW_SOURCE}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for processed output files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2000,
        help="Only include papers from this year onward (default: 2000)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Cap total records for smoke-testing (default: no limit)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    raw_path, filtered_path = process_arxiv(
        raw_source=args.raw_source,
        output_dir=args.output_dir,
        min_year=args.min_year,
        max_records=args.max_records,
    )

    print(f"\nFiles written:")
    print(f"  {raw_path}")
    print(f"  {filtered_path}")
    print(f"\nNext step:")
    print(f"  python ingest/02_create_collections.py")
    print(f"  python ingest/03_ingest_dense.py --input {filtered_path}")
    print(f"  python ingest/04_ingest_sparse.py --input {filtered_path}")


if __name__ == "__main__":
    main()
