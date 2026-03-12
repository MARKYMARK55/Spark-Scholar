#!/usr/bin/env python3
"""
ingest/01_download_arxiv.py
===========================
Download the arXiv metadata dataset from HuggingFace and save as JSONL.

Dataset: `Cornell-University/arxiv` on HuggingFace Hub
Fields used: id, title, abstract, authors, categories, update_date, journal-ref

The raw JSONL is written to data/arxiv_raw.jsonl (one JSON object per line).
A second filtered file is written to data/arxiv_with_abstract.jsonl containing
only records that have a non-empty abstract (required for embedding).

Usage
-----
    python ingest/01_download_arxiv.py [--output-dir data/] [--min-year 2010]

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
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../data")


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


def download_arxiv(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    min_year: int = 2000,
    max_records: int | None = None,
) -> tuple[Path, Path]:
    """
    Download arXiv metadata from HuggingFace and save to JSONL files.

    Parameters
    ----------
    output_dir : str
        Directory to write output JSONL files.
    min_year : int
        Only include papers from this year onward.
    max_records : int, optional
        Cap the number of records (for testing).

    Returns
    -------
    (raw_path, filtered_path)
        Paths to the raw and filtered JSONL files.
    """
    from datasets import load_dataset

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_path = output_path / "arxiv_raw.jsonl"
    filtered_path = output_path / "arxiv_with_abstract.jsonl"

    logger.info("Loading dataset %s from HuggingFace...", DATASET_NAME)
    logger.info("This will download ~4GB of metadata — please be patient.")

    dataset = load_dataset(
        DATASET_NAME,
        trust_remote_code=True,
        token=HF_TOKEN if HF_TOKEN else None,
        streaming=True,  # Stream to avoid loading everything into RAM
    )

    split = dataset["train"]

    total_written = 0
    total_filtered = 0
    total_skipped = 0
    skipped_year = 0
    skipped_abstract = 0

    with (
        open(raw_path, "w", encoding="utf-8") as raw_f,
        open(filtered_path, "w", encoding="utf-8") as filt_f,
    ):
        pbar = tqdm(split, desc="Downloading arXiv records", unit="rec")
        for record in pbar:
            if max_records and total_written >= max_records:
                break

            arxiv_id = record.get("id", "").strip()
            title = record.get("title", "").strip().replace("\n", " ")
            abstract = record.get("abstract", "").strip().replace("\n", " ")
            authors = _clean_authors(record.get("authors", ""))
            categories = record.get("categories", "").strip()
            update_date = record.get("update_date", "")
            journal_ref = record.get("journal-ref", "") or ""

            year = _parse_year(update_date)

            # Skip very old papers if min_year set
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

            # Filtered: must have abstract
            if abstract and len(abstract) > 50:
                filt_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                total_filtered += 1
            else:
                skipped_abstract += 1

            if total_written % 100_000 == 0:
                pbar.set_postfix(
                    written=total_written,
                    filtered=total_filtered,
                    skipped_year=skipped_year,
                )

    logger.info("=" * 60)
    logger.info("Download complete:")
    logger.info("  Total records written (raw):     %d", total_written)
    logger.info("  Filtered (with abstract):        %d", total_filtered)
    logger.info("  Skipped (before %d):              %d", min_year, skipped_year)
    logger.info("  Skipped (no abstract):           %d", skipped_abstract)
    logger.info("  Raw file:       %s", raw_path)
    logger.info("  Filtered file:  %s", filtered_path)
    logger.info("=" * 60)

    return raw_path, filtered_path


def main():
    parser = argparse.ArgumentParser(
        description="Download arXiv metadata from HuggingFace Hub"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save JSONL files (default: data/)",
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
        help="Cap the number of records for testing (default: no limit)",
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

    try:
        import datasets  # noqa: F401
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets huggingface_hub")
        sys.exit(1)

    raw_path, filtered_path = download_arxiv(
        output_dir=args.output_dir,
        min_year=args.min_year,
        max_records=args.max_records,
    )

    print(f"\nFiles created:")
    print(f"  {raw_path}")
    print(f"  {filtered_path}")


if __name__ == "__main__":
    main()
