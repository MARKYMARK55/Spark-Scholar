# Ingestion Pipeline

<- [Back to README](../README.md)

---

## Table of Contents

- [Overview](#overview)
- [ArXiv Collections (Pre-Embedded)](#arxiv-collections-pre-embedded)
  - [Download from HuggingFace](#download-from-huggingface)
  - [Import snapshots into Qdrant](#import-snapshots-into-qdrant)
- [Custom Knowledge Base -- PDFs & Web Docs](#custom-knowledge-base--pdf--web-documentation)
  - [PDF ingestion](#pdf-ingestion-your-own-documents)
  - [HTML / web docs](#web--html-documentation-ingestion)
  - [TOML crawl targets](#toml-crawl-targets-config-driven-crawling)
  - [Documentation collections](#documentation-collections)
  - [Re-ingestion](#re-ingestion)
- [Helper Scripts](#helper-scripts)
- [Keeping the index fresh](#keeping-the-index-fresh)
- [Browsing collections in Open WebUI](#browsing-your-collections-in-open-webui)
- [Timing Table](#timing-table)

---

## Overview

Spark-Scholar ships with **2.96 million arXiv paper abstracts** pre-embedded with BGE-M3 (dense + sparse vectors). You download them from HuggingFace and restore directly into Qdrant -- no GPU embedding required.

For your own documents, two ingestion scripts handle PDFs and web documentation:

| Pipeline | Script | Input | Method |
|---|---|---|---|
| **arXiv abstracts** | *(download only)* | Pre-embedded Qdrant snapshots from HF | Snapshot restore |
| **Your own PDFs** | `6_document_ingestion/05_ingest_pdfs.py` | PDF files in a directory | Full-text extract, chunk, embed, upsert |
| **Web / HTML docs** | `6_document_ingestion/07_ingest_html_docs.py` | URLs or TOML config files | BFS crawl, extract, chunk, embed, upsert |

Both pipelines write into Qdrant collections searched simultaneously by every query through the RAG proxy.

---

## ArXiv Collections (Pre-Embedded)

The arXiv index covers 2.96M papers across 16 topic collections. Each collection has both dense (1024-dim BGE-M3) and sparse (SPLADE lexical weights) vectors, enabling hybrid retrieval.

### Download from HuggingFace

```bash
# Requires: HF_TOKEN set in env/.env (free HuggingFace account)
pip install huggingface-hub

# Download all 16 arXiv snapshots (~XX GB total)
huggingface-cli download MARKYMARK55/spark-scholar-arxiv-snapshots \
    --repo-type dataset \
    --local-dir data/snapshots/
```

The download includes these collections:

| Snapshot | Topic coverage |
|---|---|
| `arXiv.snapshot` | General / uncategorized |
| `arxiv-cs-ml-ai.snapshot` | CS: Machine Learning, AI |
| `arxiv-cs-nlp-ir.snapshot` | CS: NLP, Information Retrieval |
| `arxiv-cs-cv.snapshot` | CS: Computer Vision |
| `arxiv-cs-systems-theory.snapshot` | CS: Systems, Theory |
| `arxiv-math-applied.snapshot` | Applied Mathematics |
| `arxiv-math-pure.snapshot` | Pure Mathematics |
| `arxiv-math-phys.snapshot` | Mathematical Physics |
| `arxiv-stat-eess.snapshot` | Statistics, EE & Signal Processing |
| `arxiv-hep.snapshot` | High Energy Physics |
| `arxiv-condmat.snapshot` | Condensed Matter |
| `arxiv-quantph-grqc.snapshot` | Quantum Physics, General Relativity |
| `arxiv-astro.snapshot` | Astrophysics |
| `arxiv-nucl-nlin-physother.snapshot` | Nuclear, Nonlinear, Other Physics |
| `arxiv-qbio-qfin-econ.snapshot` | Quantitative Biology, Finance, Economics |
| `arxiv-misc.snapshot` | Miscellaneous |

### Import snapshots into Qdrant

```bash
# Verify checksums
cd data/snapshots/
sha256sum -c checksums.sha256

# Restore each snapshot into Qdrant
for snapshot in snapshots/*.snapshot; do
    collection=$(basename "$snapshot" .snapshot)
    echo "Restoring $collection..."
    curl -X POST "http://localhost:6333/collections/$collection/snapshots/upload" \
        -H "api-key: simple-api-key" \
        -H "Content-Type: multipart/form-data" \
        -F "snapshot=@$snapshot"
done
```

After restore, verify:
```bash
curl -s -H "api-key: simple-api-key" http://localhost:6333/collections | python3 -m json.tool
```

---

## Custom Knowledge Base -- PDF & Web Documentation

This is as important as the arXiv index. The same hybrid retrieval + reranking
pipeline that searches 2.96M arXiv abstracts also searches your own curated
documentation -- language references, API docs, framework guides, tool manuals.

A single RAG query simultaneously searches arXiv papers AND your tech docs, with
the cross-encoder deciding what is most relevant.

---

### PDF ingestion (your own documents)

Each PDF directory maps to one Qdrant collection. Point the script at a directory and specify the target collection:

```bash
# Ingest PDFs into a specific collection
./scripts/run_ingest.sh 6_document_ingestion/05_ingest_pdfs.py \
    --input-dir RAG/PDF_folders/demo-bayesian-statistics \
    --collection demo-bayesian-statistics

# Or run directly (if dependencies are installed locally)
python 6_document_ingestion/05_ingest_pdfs.py \
    --input-dir /path/to/your/pdfs/ \
    --collection docs-python
```

The script:
1. Sends PDFs to the Docling service (GPU-accelerated layout analysis + Markdown export)
2. Falls back to PyMuPDF for simpler documents
3. Chunks with tiktoken (cl100k_base) at configurable CHUNK_SIZE / CHUNK_OVERLAP
4. Embeds all chunks with BGE-M3 dense + sparse vectors
5. Upserts into the specified Qdrant collection with rich payload: title, authors, year, page_num, chunk_idx

**Batch helper script:**
```bash
./scripts/ingest_doc_pdfs.sh    # ingests all PDF directories under RAG/PDF_folders/
```

**Timing:** ~2-4 min per PDF (10-30 pages).

**Resume:** Re-running skips already-processed files. Point IDs are deterministic
hashes of `filename + chunk_idx` -- upserts are idempotent.

---

### Web / HTML documentation ingestion

Ingest any documentation site directly from its URL.

```bash
# Single URL
./scripts/run_ingest.sh 6_document_ingestion/07_ingest_html_docs.py \
    --url https://docs.python.org/3/library/ \
    --collection docs-python --tag python-stdlib --depth 2

# From a TOML config file (recommended)
./scripts/run_ingest.sh 6_document_ingestion/07_ingest_html_docs.py \
    --config RAG/crawl_targets/python-ecosystem.toml
```

The crawler:
- BFS crawl -- follows same-domain links up to `--depth` levels
- Content extraction using priority-ordered CSS selectors (Sphinx, Docusaurus, GitBook, ReadTheDocs)
- Strips nav/footer/sidebar; preserves code blocks as plain text
- Sitemap.xml discovery for comprehensive coverage
- Politeness delay + progress file for resume support

---

### TOML crawl targets (config-driven crawling)

Instead of passing URLs on the command line, define crawl targets in TOML files under `RAG/crawl_targets/`. Each file specifies a collection, default depth, and a list of URLs:

```toml
# RAG/crawl_targets/python-ecosystem.toml
collection = "docs-python"
tag = "python"
depth = 3

[[targets]]
url = "https://docs.python.org/3/"
description = "Python 3 standard library"

[[targets]]
url = "https://fastapi.tiangolo.com/"
description = "FastAPI web framework"
```

Run all targets for a collection:
```bash
./scripts/run_ingest.sh 6_document_ingestion/07_ingest_html_docs.py \
    --config RAG/crawl_targets/python-ecosystem.toml
```

Run all TOML configs at once:
```bash
./scripts/crawl_all.sh        # crawl all targets (HTML + PDFs)
./scripts/crawl_html_only.sh  # crawl HTML targets only
```

---

### Documentation collections

| Collection | TOML config | Contents |
|---|---|---|
| `docs-python` | `python-ecosystem.toml` | Python stdlib, NumPy, Pandas, FastAPI, LangChain, Pydantic |
| `docs-ml-ai` | `ml-ai-frameworks.toml` | PyTorch, scikit-learn, Hugging Face, JAX |
| `docs-llm-agents` | `llm-agents.toml` | LangGraph, CrewAI, Anthropic SDK, MCP |
| `docs-rapids-gpu` | `rapids-gpu.toml` | RAPIDS cuDF, cuML, Dask-CUDA |
| `docs-web-backend` | `web-backend.toml` | FastAPI, Django, Flask, SQLAlchemy |
| `docs-devtools` | `devtools.toml` | Git, Docker, GitHub Actions, Terraform |
| `docs-html-libs` | `html-only-libs.toml` | HTML-rendered library docs |
| `docs-nvidia-rapids` | `nvidia-rapids.toml` | NVIDIA RAPIDS ecosystem |
| `docs-infra` | `infra-sysadmin.toml` | Linux, systemd, networking, Prometheus |
| `Qdrant_Web_Doc` | `qdrant-docs.toml` | Qdrant vector database documentation |
| `docs-applescript` | `applescript.toml` | AppleScript Language Guide, macOS scripting |

---

### Re-ingestion

Batch re-embed and re-upsert existing collections (useful after embedding model updates or schema changes):

```bash
./scripts/run_ingest.sh 6_document_ingestion/12_reingest_pdfs.py \
    --collections demo-bayesian-statistics docs-python
```

---

## Helper Scripts

| Script | Purpose |
|---|---|
| `scripts/run_ingest.sh` | Run any ingest script inside the pipelines Docker container |
| `scripts/ingest_doc_pdfs.sh` | Batch-ingest all PDF directories under `RAG/PDF_folders/` |
| `scripts/crawl_html_only.sh` | Crawl all HTML TOML targets |
| `scripts/crawl_all.sh` | Crawl all TOML targets (HTML + PDFs) |

---

## Keeping the index fresh

Documentation sites update regularly. Maintain freshness with:

```bash
# Re-ingest a URL -- remove it from the progress file to force re-crawl:
grep -v "doc.rust-lang.org" data/html_ingest_progress.txt > /tmp/prog.txt \
    && mv /tmp/prog.txt data/html_ingest_progress.txt

# Use per-source progress files for cleaner management:
./scripts/run_ingest.sh 6_document_ingestion/07_ingest_html_docs.py \
    --url https://docs.anthropic.com/en/docs/ \
    --collection docs-anthropic \
    --progress-file data/anthropic_progress.txt
```

---

## Browsing your collections in Open WebUI

Every Qdrant collection is visible in the RAG proxy's models endpoint:

```bash
curl http://localhost:8002/v1/models   # lists all indexed collections
```

From Open WebUI (Path B -- RAG proxy connection), pass collection filters:

```json
{
    "model": "spark-scholar",
    "messages": [{"role": "user", "content": "How do I use async/await in Rust?"}],
    "collections": ["docs-rust", "arxiv-cs-ml-ai"]
}
```

Without `collections`, the query router picks automatically from all collections.

---

## Timing Table

Measured on DGX Spark (Grace Blackwell, 128 GB unified memory):

| Stage | Scope | Throughput | Wall time |
|---|---|---|---|
| Download pre-embedded arXiv collections from HF | 16 collections, 2.96M papers | Network speed | ~30-60 min |
| Restore snapshots into Qdrant | 16 collections | Disk I/O | ~5-15 min |
| Single PDF, 30 pages *(production mode)* | ~60 chunks | -- | ~2-4 min |
| Single PDF, 200 pages | ~400 chunks | -- | ~15-25 min |
| HTML crawl (typical docs site, 500 pages) | docs-* | Network limited | ~20-40 min |
| Full HTML crawl (all TOML targets) | All doc collections | Network limited | ~4-8 hrs |

> **Note:** Building the arXiv index from scratch takes ~22 hours of dense embedding + ~8 hours of sparse embedding on DGX Spark. By downloading pre-embedded collections from HuggingFace, you skip this entirely. The build pipeline scripts are archived in `6_document_ingestion/_archive_for_hf_repo/` for maintainers who need to regenerate the dataset.

See `docs/embedding_speed.md` for full tuning guide and VRAM breakdown.
