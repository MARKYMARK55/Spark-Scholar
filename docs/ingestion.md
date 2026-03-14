# Ingestion Pipeline

← [Back to README](../README.md)

---

## Table of Contents

- [Overview — Two Separate Pipelines](#overview)
- [Step 0: Create Qdrant Collections](#step-0-create-qdrant-collections)
- [Step 1: Download arXiv Metadata](#step-1-download-arxiv-metadata)
- [Step 2: Dense Embedding (arXiv abstracts)](#step-2-dense-embedding--indexing)
- [Step 3: Sparse Embedding (arXiv abstracts)](#step-3-sparse-embedding--indexing)
- [Custom Knowledge Base — PDFs & Web Docs](#custom-knowledge-base--pdf--web-documentation)
  - [What the pipeline does](#what-the-python-pipeline-does)
  - [Documentation collections](#documentation-collections)
  - [Step 4: PDF ingestion](#step-4-pdf-ingestion-your-own-documents)
  - [Step 5: Figure captioning](#step-5-figure-captioning)
  - [Step 6: HTML / web docs](#step-6-web--html-documentation-ingestion)
  - [Step 7: Citation graph expansion (L2/L3)](#step-7-citation-graph-expansion-l2--l3)
- [Keeping the index fresh](#keeping-the-index-fresh)
- [Browsing collections in Open WebUI](#browsing-your-collections-in-open-webui)
- [Timing Table](#timing-table)

---

## Overview

Two separate pipelines serve different purposes:

| Pipeline | Steps | Input | Method |
|---|---|---|---|
| **arXiv abstracts** | 0–3 | 2.96M paper metadata (JSONL) | Text-only, direct embed |
| **Your own corpus** | 4–7 | PDFs, URLs, HTML sites | Full-text extract + HDBSCAN auto-classify |

The arXiv pipeline is text-only — no PDF downloads, no figure captioning, no HDBSCAN.
The custom corpus pipeline handles any mix of PDFs and web docs with automatic topic classification.

Both pipelines write into the same Qdrant collections and are searched simultaneously by every query through the RAG proxy.

---

## Step 0: Create Qdrant Collections

Run once before any ingestion:

```bash
python ingest/02_create_collections.py                    # all 24 collections
python ingest/02_create_collections.py --arxiv-only       # 16 arXiv only
python ingest/02_create_collections.py --docs-only        # 8 docs only
python ingest/02_create_collections.py --collection docs-rust   # one at a time
python ingest/02_create_collections.py --verify-only      # check what exists
```

Each collection is created with two vector slots:
- `dense_embedding` — 1024-dim HNSW float32 vectors (BGE-M3 dense)
- `sparse_text` — sparse SPLADE vectors (BGE-M3 lexical weights)

---

## Step 1: Download arXiv Metadata

Downloads the Cornell-University/arxiv HuggingFace dataset (~4.8 GB JSONL, 2.96M papers).
Abstracts only — no PDFs.

```bash
# Full dataset
python ingest/01_download_arxiv.py --output-dir data/

# Quick test with 10k papers
python ingest/01_download_arxiv.py --output-dir data/ --max-records 10000
```

**Timing:** ~40 minutes on a good connection. Cached by HuggingFace after first run.
**Requires:** `HF_TOKEN` set in `env/.env` (free HuggingFace account).

---

## Step 2: Dense Embedding + Indexing

Switch to indexing mode first for maximum throughput, then run:

```bash
bash scripts/start_indexing_mode.sh   # 50% GPU util, ~12 GB VRAM

python ingest/03_ingest_dense.py \
    --input data/arxiv_with_abstract.jsonl \
    --batch-size 256

bash scripts/stop_indexing_mode.sh    # back to 12% GPU production mode
```

**Timing:** ~18–22 hours for 2.96M papers at ~1,100 docs/s.
**Resume:** Progress tracked in `data/arxiv_with_abstract_dense_progress.txt`.
Re-running skips already-processed records. Upserts are idempotent.

---

## Step 3: Sparse Embedding + Indexing

```bash
python ingest/04_ingest_sparse.py \
    --input data/arxiv_with_abstract.jsonl \
    --batch-size 64
```

**Timing:** ~6–8 hours for 2.96M papers at ~1,400–1,600 docs/s.
**Run in parallel with Step 2:** Dense (~18 hrs) dominates. Start both simultaneously —
they write separate progress files and don't interfere.

---

## Custom Knowledge Base — PDF & Web Documentation

This is as important as the arXiv index. The same hybrid retrieval + reranking
pipeline that searches 2.96M arXiv abstracts also searches your own curated
documentation — language references, API docs, framework guides, tool manuals.

A single RAG query simultaneously searches arXiv papers AND your tech docs, with
the cross-encoder deciding what is most relevant.

---

### What the Python pipeline does

| Script | Input | Key libraries |
|---|---|---|
| `05_ingest_pdfs.py` | PDF files | PyMuPDF, unstructured, tiktoken, HDBSCAN, UMAP |
| `06_caption_figures.py` | PDF files | PyMuPDF, vLLM vision model (Qwen2-VL) |
| `07_ingest_html_docs.py` | URLs / HTML | requests, BeautifulSoup4, tiktoken |
| `08_expand_citations.py` | arXiv IDs / PDFs | httpx, Semantic Scholar API, PyMuPDF |

**The HDBSCAN auto-classification pipeline** (scripts 05 + 06):
1. Extract full text from PDFs with PyMuPDF — preserves structure, handles multi-column layouts
2. Fallback to `unstructured` for complex formatting (scanned pages, mixed content)
3. Chunk with tiktoken (cl100k_base) at CHUNK_SIZE / CHUNK_OVERLAP tokens
4. Embed all chunks with BGE-M3 dense vectors (1024-dim)
5. Reduce to 5 dimensions with UMAP — reveals topic clusters
6. Cluster with HDBSCAN — finds number of topics automatically, no `k` to choose
7. Send each cluster's representative texts to Qwen3 via local vLLM to generate cluster names
8. Route each chunk to its Qdrant collection based on cluster name + filename keywords
9. Upsert with rich payload: title, authors, year, page_num, chunk_idx, topic_id, topic_name

**Why this matters:** a directory of 200 mixed PDFs (NeurIPS papers, Rust book printouts,
Docker guides) is automatically sorted into the right collections without any labelling.

**The HTML/URL crawler** (`07_ingest_html_docs.py`):
- BFS crawler — follows same-domain links up to `--depth` levels
- Content extraction using priority-ordered CSS selectors (Sphinx, Docusaurus, GitBook, ReadTheDocs)
- Strips nav/footer/sidebar; preserves code blocks as plain text
- Sitemap.xml discovery for comprehensive coverage
- Politeness delay + progress file for resume support

---

### Documentation collections

| Collection | Contents | Ingest method |
|---|---|---|
| `docs-python` | Python stdlib, NumPy, Pandas, FastAPI, LangChain, Pydantic | HTML crawl or PDF |
| `docs-rust` | The Rust Book, std lib, Cargo, Tokio, Axum, Serde | HTML crawl |
| `docs-javascript` | MDN Web Docs, Node.js, TypeScript, React, Next.js | HTML crawl |
| `docs-docker` | Docker Engine, Compose, Buildx, Kubernetes, Helm | HTML crawl or PDF |
| `docs-anthropic` | Claude API, Computer Use (CUA), Model Context Protocol | HTML crawl |
| `docs-applescript` | AppleScript Language Guide, macOS scripting | HTML crawl or PDF |
| `docs-devops` | GitHub Actions, Terraform, Ansible, Prometheus, Grafana | HTML crawl |
| `docs-web` | HTML5, CSS3, Web APIs, HTTP standards | HTML crawl |

Collections prefixed `openwebui_` are managed by Open WebUI's built-in document
upload feature — they don't interfere with the above.

---

### Step 4: PDF ingestion (your own documents)

```bash
# Ingest a directory of PDFs (auto-classifies into collections via HDBSCAN)
python ingest/05_ingest_pdfs.py --input-dir /path/to/your/pdfs/

# Force a specific collection (skip auto-classification)
python ingest/05_ingest_pdfs.py --input-dir ./pdfs/ --collection docs-docker

# Dry run — preview HDBSCAN classification without writing to Qdrant
python ingest/05_ingest_pdfs.py --input-dir ./pdfs/ --dry-run

# Verbose — show cluster names assigned by Qwen3 + routing decisions
python ingest/05_ingest_pdfs.py --input-dir ./pdfs/ --verbose
```

**Timing:** ~2–4 min per PDF (10–30 pages). HDBSCAN clusters the whole batch in one
pass — processing 100 PDFs is only marginally slower than 10.

**Resume:** Re-running skips already-processed files. Point IDs are deterministic
hashes of `filename + chunk_idx` — upserts are idempotent.

---

### Step 5: Figure captioning

Generates captions for figures, diagrams, and tables using a vision model (Qwen2-VL
via vLLM). Captions are stored as `type="figure_caption"` points alongside text
chunks — figures become searchable.

```bash
# Caption all figures in a directory
python ingest/06_caption_figures.py \
    --input-dir /path/to/pdfs/ \
    --pages-with-figures-only    # skip text-only pages (much faster)

# Single file
python ingest/06_caption_figures.py --input-dir ./pdfs/ --file paper.pdf
```

**Timing:** ~30–60 seconds per page with figures. Run once — vision inference never
happens at query time.

---

### Step 6: Web / HTML documentation ingestion

Ingest any documentation site directly from its URL.

#### Rust
```bash
python ingest/07_ingest_html_docs.py \
    --url https://doc.rust-lang.org/book/ \
    --collection docs-rust --tag rust-book --depth 2

python ingest/07_ingest_html_docs.py \
    --url https://doc.rust-lang.org/std/ \
    --collection docs-rust --tag rust-std --depth 2

python ingest/07_ingest_html_docs.py \
    --url https://docs.rs/tokio/latest/tokio/ \
    --collection docs-rust --tag tokio
```

**Offline:** `rustup docs --book` opens the book locally. Scrape with `--url file:///...`

#### Python
```bash
python ingest/07_ingest_html_docs.py \
    --url https://docs.python.org/3/library/ \
    --collection docs-python --tag python-stdlib --depth 2

python ingest/07_ingest_html_docs.py \
    --url https://fastapi.tiangolo.com/ \
    --collection docs-python --tag fastapi --depth 2

python ingest/07_ingest_html_docs.py \
    --url https://python.langchain.com/docs/introduction/ \
    --collection docs-python --tag langchain --depth 2

python ingest/07_ingest_html_docs.py \
    --url https://docs.pydantic.dev/latest/ \
    --collection docs-python --tag pydantic --depth 2
```

#### JavaScript / TypeScript
```bash
python ingest/07_ingest_html_docs.py \
    --url https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference \
    --collection docs-javascript --tag mdn-js --depth 2

python ingest/07_ingest_html_docs.py \
    --url https://www.typescriptlang.org/docs/ \
    --collection docs-javascript --tag typescript --depth 2

python ingest/07_ingest_html_docs.py \
    --url https://nodejs.org/en/docs \
    --collection docs-javascript --tag nodejs --depth 2
```

#### Docker / Kubernetes
```bash
python ingest/07_ingest_html_docs.py \
    --url https://docs.docker.com/engine/ \
    --collection docs-docker --tag docker-engine --depth 2

python ingest/07_ingest_html_docs.py \
    --url https://docs.docker.com/compose/ \
    --collection docs-docker --tag docker-compose --depth 2

python ingest/07_ingest_html_docs.py \
    --url https://kubernetes.io/docs/ \
    --collection docs-docker --tag kubernetes --depth 2 --sitemap
```

#### Anthropic / Claude API / CUA / MCP
```bash
# Full Claude API + prompting guides
python ingest/07_ingest_html_docs.py \
    --url https://docs.anthropic.com/en/docs/ \
    --collection docs-anthropic --tag anthropic-api --depth 2

# Computer Use (CUA)
python ingest/07_ingest_html_docs.py \
    --url https://docs.anthropic.com/en/docs/agents-and-tools/computer-use \
    --collection docs-anthropic --tag cua --depth 1

# Model Context Protocol (MCP)
python ingest/07_ingest_html_docs.py \
    --url https://modelcontextprotocol.io/docs/ \
    --collection docs-anthropic --tag mcp --depth 2
```

#### AppleScript

Apple Developer docs have access controls. Mirror with wget first:
```bash
wget --mirror --convert-links --adjust-extension --page-requisites --no-parent \
    -P /tmp/applescript-docs \
    "https://developer.apple.com/library/archive/documentation/AppleScript/Conceptual/AppleScriptLangGuide/"

python ingest/07_ingest_html_docs.py \
    --url "file:///tmp/applescript-docs/developer.apple.com/library/archive/documentation/AppleScript/" \
    --collection docs-applescript --tag applescript-lang-guide

# Also: export dictionaries from Script Editor → File → Export, then:
python ingest/05_ingest_pdfs.py --input-dir ./applescript-pdfs/ \
    --collection docs-applescript
```

#### DevOps
```bash
python ingest/07_ingest_html_docs.py \
    --url https://docs.github.com/en/actions \
    --collection docs-devops --tag github-actions --depth 2

python ingest/07_ingest_html_docs.py \
    --url https://developer.hashicorp.com/terraform/docs \
    --collection docs-devops --tag terraform --depth 2

python ingest/07_ingest_html_docs.py \
    --url https://prometheus.io/docs/introduction/overview/ \
    --collection docs-devops --tag prometheus --depth 2

python ingest/07_ingest_html_docs.py \
    --url https://grafana.com/docs/grafana/latest/ \
    --collection docs-devops --tag grafana --depth 1
```

---

### Step 7: Citation graph expansion (L2 / L3)

Automatically discovers, downloads, and indexes papers referenced by your existing corpus.
Uses the Semantic Scholar API for structured reference data.

```
L1 = your original papers (already ingested)
L2 = papers cited in L1  (--depth 1)
L3 = papers cited in L2  (--depth 2)
```

```bash
# Expand a single paper to L2 + L3
python ingest/08_expand_citations.py \
    --arxiv 2303.08774 \
    --depth 2 \
    --output-dir data/citations/

# Expand multiple papers from a file (one arXiv ID per line)
python ingest/08_expand_citations.py \
    --arxiv-file my_papers.txt \
    --depth 2 \
    --output-dir data/citations/

# Expand from a directory of PDFs (arXiv IDs extracted from filenames)
python ingest/08_expand_citations.py \
    --input-dir data/papers/ \
    --depth 1 \
    --output-dir data/citations/

# Quality filter: only follow highly-cited references
python ingest/08_expand_citations.py \
    --arxiv 2303.08774 \
    --depth 2 \
    --min-citations 50 \
    --max-per-paper 20 \
    --output-dir data/citations/

# JSON manifest only — no downloading, no ingestion
python ingest/08_expand_citations.py \
    --arxiv 2303.08774 \
    --json-only \
    --output-dir data/citations/
```

**Output structure:**
```
data/citations/
├── manifests/
│   ├── L2_2303.08774.json    # all references of the source paper
│   └── L3_2005.14165.json    # references of an L2 paper
├── L2/2303.08774/            # PDFs downloaded for L2 expansion
└── L3/2005.14165/            # PDFs downloaded for L3 expansion
```

Each manifest lists every discovered reference with title, authors, year, arXiv ID,
DOI, citation count, open-access PDF URL, and ingestion status.

**Resume support:** Re-running with the same `--output-dir` skips already-processed
papers. Progress is saved in `data/citations/expansion_progress.json`.

**Rate limiting:** The Semantic Scholar API allows 1 req/s unauthenticated (100 req/s
with a free API key). Set `SEMANTIC_SCHOLAR_API_KEY` in `env/.env` for faster expansion.

---

## Keeping the index fresh

Documentation sites update regularly. Maintain freshness with:

```bash
# Re-ingest a URL — remove it from the progress file to force re-crawl:
grep -v "doc.rust-lang.org" data/html_ingest_progress.txt > /tmp/prog.txt \
    && mv /tmp/prog.txt data/html_ingest_progress.txt

# Use per-source progress files for cleaner management:
python ingest/07_ingest_html_docs.py \
    --url https://docs.anthropic.com/en/docs/ \
    --collection docs-anthropic \
    --progress-file data/anthropic_progress.txt

# Wipe and re-ingest a collection entirely:
python ingest/02_create_collections.py --collection docs-anthropic --recreate
python ingest/07_ingest_html_docs.py \
    --url https://docs.anthropic.com/en/docs/ \
    --collection docs-anthropic --depth 2
```

---

## Browsing your collections in Open WebUI

Every Qdrant collection is visible in the RAG proxy's models endpoint:

```bash
curl http://localhost:8002/v1/models   # lists all indexed collections
```

From Open WebUI (Path B — RAG proxy connection), pass collection filters:

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
| Download arXiv metadata | 2.96M papers | ~1,300/s (network) | ~40 min |
| Create 24 Qdrant collections | — | instant | ~30 sec |
| Dense embed + upsert *(indexing mode, 50% GPU)* | 2.96M abstracts | ~1,100 docs/s | ~18–22 hrs |
| Sparse embed + upsert | 2.96M abstracts | ~1,400–1,600 docs/s | ~6–8 hrs |
| Dense + sparse **in parallel** | 2.96M abstracts | limited by dense | **~18–22 hrs total** |
| Single PDF, 30 pages *(production mode)* | ~60 chunks | — | ~2–4 min |
| Single PDF, 200 pages | ~400 chunks | — | ~15–25 min |
| Figure captioning *(vision model, per page)* | PDFs only | ~2 pages/min | ~30–60 s/page |
| HTML crawl (typical docs site, 500 pages) | docs-* | network limited | ~20–40 min |
| Citation expansion L2 (20 refs × 1 paper) | — | ~1 paper/2 min | ~40 min |
| Citation expansion L2+L3 (20 refs, 2 levels) | — | — | ~2–8 hrs |

> **Dense is slower than sparse** despite using the same BGE-M3 model. Dense requires
> computing the full 1024-dim projection; sparse uses a shallower lexical weight head
> that is significantly cheaper to evaluate.

See `docs/embedding_speed.md` for full tuning guide and VRAM breakdown.
