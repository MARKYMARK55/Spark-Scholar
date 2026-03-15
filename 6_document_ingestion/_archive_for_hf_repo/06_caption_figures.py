#!/usr/bin/env python3
"""
ingest/06_caption_figures.py
=============================
Extract figures, tables, and equation images from PDFs and generate
captions using a vision LLM (Qwen2-VL or similar) via vLLM.

Pipeline per PDF
----------------
1. Render each page as a high-resolution image (300 DPI) with PyMuPDF
2. Use a simple heuristic to detect figures: look for image blocks and
   regions with low text density surrounded by whitespace
3. Crop each figure/table region, encode as base64 PNG
4. Send to vision LLM (Qwen2-VL) via vLLM /v1/chat/completions with
   image_url content type
5. Store generated captions as new Qdrant points with type="figure_caption"

The figure chunks are indexed alongside text chunks so they appear in
retrieval results for visual content queries.

Usage
-----
    python ingest/06_caption_figures.py --input-dir /path/to/pdfs/
    python ingest/06_caption_figures.py --input-dir ./pdfs/ \
        --vllm-url http://localhost:8000 \
        --collection arxiv-cs-ml-ai \
        --pages-with-figures-only

Requirements
------------
    pip install pymupdf httpx qdrant-client tqdm python-dotenv
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector
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

# DPI for page rendering
PAGE_DPI = 150
# Minimum area ratio for a "figure" region (fraction of page area)
MIN_FIGURE_AREA_RATIO = 0.02
# Maximum area ratio (full-page images are usually background)
MAX_FIGURE_AREA_RATIO = 0.85
# Request timeout for vision model (images are large)
VISION_TIMEOUT = 60.0


# ---------------------------------------------------------------------------
# Figure detection helpers
# ---------------------------------------------------------------------------

def _page_has_figure_content(page) -> bool:
    """
    Quick heuristic check: does this page likely contain figures?
    Checks for image blocks and large rect annotations.
    """
    blocks = page.get_text("dict").get("blocks", [])
    for block in blocks:
        if block.get("type") == 1:  # Type 1 = image block
            return True
    # Also check for large drawings/paths
    drawings = page.get_drawings()
    if drawings:
        page_area = page.rect.width * page.rect.height
        for drawing in drawings:
            rect = drawing.get("rect")
            if rect:
                drawing_area = (rect[2] - rect[0]) * (rect[3] - rect[1])
                if drawing_area / page_area > MIN_FIGURE_AREA_RATIO:
                    return True
    return False


def extract_figure_regions(page, page_num: int) -> list[dict]:
    """
    Extract figure/table/equation image regions from a PDF page.

    Returns a list of:
        {
            "page_num": int,
            "bbox": (x0, y0, x1, y1),
            "region_type": "figure" | "table" | "equation",
            "image_bytes": bytes (PNG),
        }
    """
    import fitz

    regions = []
    page_rect = page.rect
    page_area = page_rect.width * page_rect.height

    blocks = page.get_text("dict").get("blocks", [])

    # Extract embedded image blocks
    for block in blocks:
        if block.get("type") == 1:  # image block
            bbox = fitz.Rect(block["bbox"])
            block_area = bbox.width * bbox.height
            area_ratio = block_area / page_area

            if area_ratio < MIN_FIGURE_AREA_RATIO or area_ratio > MAX_FIGURE_AREA_RATIO:
                continue

            # Render the page clipped to this region
            mat = fitz.Matrix(PAGE_DPI / 72, PAGE_DPI / 72)
            clip = bbox
            pix = page.get_pixmap(matrix=mat, clip=clip)
            img_bytes = pix.tobytes("png")

            regions.append({
                "page_num": page_num,
                "bbox": tuple(bbox),
                "region_type": _classify_region(block, bbox, page_area),
                "image_bytes": img_bytes,
            })

    # Also render the full page if it has significant drawing content
    drawings = page.get_drawings()
    if len(drawings) > 5:
        # Merge drawing bboxes to find the figure region
        all_rects = []
        for d in drawings:
            r = d.get("rect")
            if r:
                all_rects.append(fitz.Rect(r))
        if all_rects:
            merged = all_rects[0]
            for r in all_rects[1:]:
                merged |= r
            area_ratio = (merged.width * merged.height) / page_area
            if MIN_FIGURE_AREA_RATIO < area_ratio < MAX_FIGURE_AREA_RATIO:
                mat = fitz.Matrix(PAGE_DPI / 72, PAGE_DPI / 72)
                pix = page.get_pixmap(matrix=mat, clip=merged)
                img_bytes = pix.tobytes("png")
                regions.append({
                    "page_num": page_num,
                    "bbox": tuple(merged),
                    "region_type": "figure",
                    "image_bytes": img_bytes,
                })

    return regions


def _classify_region(block: dict, bbox, page_area: float) -> str:
    """Classify a region as figure, table, or equation based on heuristics."""
    area_ratio = (bbox.width * bbox.height) / page_area
    # Very small regions are likely equations
    if area_ratio < 0.05:
        return "equation"
    # Wide/thin regions are often tables
    aspect = bbox.width / max(bbox.height, 1)
    if aspect > 3.0:
        return "table"
    return "figure"


# ---------------------------------------------------------------------------
# Vision model captioning
# ---------------------------------------------------------------------------

def caption_image(
    image_bytes: bytes,
    region_type: str,
    context_text: str,
    vllm_url: str,
    vllm_api_key: str,
    model: str,
    client: httpx.Client,
) -> str:
    """
    Generate a caption for an image using a vision LLM.

    Parameters
    ----------
    image_bytes : bytes
        PNG image data.
    region_type : str
        "figure", "table", or "equation".
    context_text : str
        Surrounding text context to help the model.
    vllm_url, vllm_api_key, model : str
        vLLM connection details.
    client : httpx.Client

    Returns
    -------
    str
        Generated caption.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    type_prompts = {
        "figure": "Describe this scientific figure precisely. Include what type of plot/diagram it is, the axes/labels, key trends or results shown, and any important annotations.",
        "table": "Describe this table precisely. Include the column headers, the type of data shown, key values, and any notable patterns.",
        "equation": "Write out this mathematical equation or formula in LaTeX notation and explain what it represents.",
    }
    type_prompt = type_prompts.get(region_type, type_prompts["figure"])

    context_note = ""
    if context_text:
        context_note = f"\n\nContext from surrounding text: {context_text[:300]}"

    prompt = f"{type_prompt}{context_note}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        "max_tokens": 256,
        "temperature": 0.1,
    }

    try:
        url = f"{vllm_url.rstrip('/')}/v1/chat/completions"
        resp = client.post(url, json=payload, timeout=VISION_TIMEOUT)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning("Vision captioning failed: %s", exc)
        return f"[{region_type} captioning failed]"


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def ingest_figures(
    input_dir: str,
    collection_override: Optional[str] = None,
    vllm_url: str = VLLM_URL,
    max_files: Optional[int] = None,
    pages_with_figures_only: bool = False,
) -> dict[str, int]:
    """
    Process all PDFs in input_dir, extract figures, generate captions,
    and store as vector points in Qdrant.

    Returns
    -------
    dict[str, int]
        Maps collection → number of figure caption points upserted.
    """
    import fitz

    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
    http_client = httpx.Client(
        headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
        timeout=VISION_TIMEOUT,
    )

    pdf_files = sorted(Path(input_dir).rglob("*.pdf"))
    if max_files:
        pdf_files = pdf_files[:max_files]

    logger.info("Found %d PDFs for figure captioning", len(pdf_files))
    counts: dict[str, int] = {}

    for pdf_path in tqdm(pdf_files, desc="Captioning figures", unit="pdf"):
        try:
            doc = fitz.open(str(pdf_path))
            source_file = pdf_path.name

            # Determine collection
            if collection_override:
                collection = collection_override
            else:
                collection = route_query(pdf_path.stem.replace("-", " ").replace("_", " "))[0]

            figure_points = []

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)

                if pages_with_figures_only and not _page_has_figure_content(page):
                    continue

                regions = extract_figure_regions(page, page_num + 1)
                if not regions:
                    continue

                # Get surrounding text context
                context_text = page.get_text("text")[:500]

                for region_idx, region in enumerate(regions):
                    caption = caption_image(
                        image_bytes=region["image_bytes"],
                        region_type=region["region_type"],
                        context_text=context_text,
                        vllm_url=vllm_url,
                        vllm_api_key=VLLM_API_KEY,
                        model=VLLM_MODEL,
                        client=http_client,
                    )

                    if not caption or caption.startswith("["):
                        continue

                    # Create embedding of the caption text
                    caption_key = f"{source_file}::page{page_num+1}::region{region_idx}"
                    point_id = abs(int(hashlib.sha256(caption_key.encode()).hexdigest()[:15], 16)) % (2 ** 63)

                    try:
                        dense_arr = encode_dense([caption])
                        sparse_list = encode_sparse([caption])
                        dense_vec = dense_arr[0].tolist()
                        sv = sparse_list[0]
                        qdrant_sparse = SparseVector(indices=sv.indices, values=sv.values)
                    except Exception as emb_exc:
                        logger.warning("Embedding caption failed: %s", emb_exc)
                        continue

                    payload = {
                        "arxiv_id": pdf_path.stem,
                        "title": pdf_path.stem.replace("_", " ").replace("-", " "),
                        "chunk_text": caption,
                        "page_num": page_num + 1,
                        "source_file": source_file,
                        "region_type": region["region_type"],
                        "bbox": list(region["bbox"]),
                        "type": "figure_caption",
                        "year": None,
                        "authors": "",
                        "categories": "",
                        "topic_id": -1,
                        "topic_name": region["region_type"],
                    }

                    figure_points.append(
                        PointStruct(
                            id=point_id,
                            vector={
                                "dense_embedding": dense_vec,
                                "sparse_text": qdrant_sparse,
                            },
                            payload=payload,
                        )
                    )

                    logger.debug(
                        "  Page %d, %s: %s",
                        page_num + 1,
                        region["region_type"],
                        caption[:80],
                    )

            doc.close()

            if figure_points:
                # Upsert in batches of 32
                for batch_start in range(0, len(figure_points), 32):
                    batch = figure_points[batch_start : batch_start + 32]
                    qdrant_client.upsert(
                        collection_name=collection,
                        points=batch,
                        wait=True,
                    )
                counts[collection] = counts.get(collection, 0) + len(figure_points)
                logger.info("  %s: %d figure captions → %s", source_file, len(figure_points), collection)
            else:
                logger.info("  %s: no figures detected", source_file)

        except Exception as exc:
            logger.error("Error processing %s: %s", pdf_path.name, exc, exc_info=True)

    http_client.close()
    total = sum(counts.values())
    logger.info("Figure captioning complete: %d captions across %d collections", total, len(counts))
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Extract and caption figures from PDFs using a vision LLM"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing PDF files",
    )
    parser.add_argument(
        "--vllm-url",
        default=VLLM_URL,
        help=f"vLLM API URL (default: {VLLM_URL})",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Override automatic collection routing",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Process at most this many PDFs",
    )
    parser.add_argument(
        "--pages-with-figures-only",
        action="store_true",
        help="Only process pages that appear to contain images/drawings",
    )
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    counts = ingest_figures(
        input_dir=args.input_dir,
        collection_override=args.collection,
        vllm_url=args.vllm_url,
        max_files=args.max_files,
        pages_with_figures_only=args.pages_with_figures_only,
    )

    print(f"\nFigure captioning complete:")
    for coll, cnt in sorted(counts.items()):
        print(f"  {coll}: {cnt} figure captions")
    print(f"Total: {sum(counts.values())}")


if __name__ == "__main__":
    main()
