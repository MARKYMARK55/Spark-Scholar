# Spark-Scholar

Self-hosted academic research platform with hybrid vector search across 3.08 million arXiv papers, custom document ingestion, and 20+ live academic API tools. Built on NVIDIA DGX Spark but runs on any machine with Docker.

## Three Independent Modules

Deploy any combination. Each module works on its own or alongside the others.

| arXiv Search | Document Ingestion | Academic Tools |
|---|---|---|
| 3.08M papers pre-indexed with BGE-M3 dense + sparse vectors | Add your own PDFs, web docs, and papers into the same search infrastructure | 20+ Open WebUI tools for live searches across 500M+ scholarly records |
| 15 domain-specific Qdrant collections | 3-tier text extraction, auto-topic classification, citation graph expansion | Semantic Scholar, OpenAlex, PubMed, CORE, Ai2 Asta, and more |
| 9ms hybrid search via RRF fusion | Each folder creates its own searchable collection | Install only the tools you need — no Qdrant required |
| [Pre-built snapshots on HuggingFace](https://huggingface.co/datasets/MARKYMARK55/spark-scholar-arxiv-snapshots) skip 24h of GPU embedding | 8 pre-configured doc collections (Python, Rust, JS, Docker, etc.) | Callable by the model during chat inference |

The pipeline **auto-detects available collections** from Qdrant. Restore three arXiv collections and it searches those three. Ingest PDFs later and they appear alongside arXiv automatically. Academic Tools are independent — install whichever tool JSONs you want into Open WebUI.

## Choose Your Approach

Modules and usage modes are independently combinable. Pick what fits your needs:

### Open WebUI + Dense Search (Simplest)

Use Open WebUI's built-in RAG with the dense embedder (port 8025) and optionally the cross-encoder reranker (port 8020). No sparse vectors, no LangGraph needed. Minimal setup, good enough for many use cases.

### Full Hybrid Pipeline (LangGraph + RAG Proxy)

Dense + sparse + RRF fusion + cross-encoder reranking, orchestrated by a LangGraph state machine. The RAG Proxy (port 8002) exposes an OpenAI-compatible `/v1/chat/completions` endpoint. Open WebUI or any OpenAI client connects to it. Includes automatic web search for time-sensitive queries, Redis caching, and Langfuse tracing.

```
START -> check_cache -> route_query -> embed_query -> hybrid_retrieve
      -> should_web_search -> [web_search] -> merge_results
      -> rerank_results -> build_context -> llm_inference
      -> cache_result -> trace_result -> END
```

### Langflow (Visual Pipeline Builder)

Drag-and-drop alternative to coding pipelines. A custom LiteLLM component is included in `langflow/components/models/`. Build your own RAG flows visually. Use alongside or instead of LangGraph.

### Academic Tools Standalone

The Open WebUI tools work on their own. Install them into Open WebUI — no Qdrant, no embedders needed. They call external APIs directly (Semantic Scholar, PubMed, OpenAlex, etc.).

### Mix and Match

- **Just arXiv search with simple dense?** Deploy Qdrant + dense embedder + Open WebUI
- **Just academic API tools?** Deploy Open WebUI + install tool JSONs
- **Full stack?** Deploy everything
- **Custom flows?** Use Langflow instead of or alongside LangGraph

## Service Map

| Service | Port | Role |
|---|---|---|
| **Qdrant** | 6333 | Vector database for all collections |
| **BGE-M3 Dense** | 8025 | Dense embedder (vLLM, 1024-dim, OpenAI-compatible) |
| **BGE-M3 Sparse** | 8035 | Sparse embedder (SPLADE weights, FastAPI) |
| **BGE-M3 Reranker** | 8020 | Cross-encoder reranker (vLLM) |
| **RAG Proxy** | 8002 | OpenAI-compatible `/v1/chat/completions` with RAG |
| **LiteLLM** | 4000 | Routes to local vLLM or 200+ cloud model providers |
| **Open WebUI** | 8080 | Chat interface with academic tools and document upload |
| **Langflow** | 7860 | Visual pipeline builder |
| **Redis** | 6379 | Result caching |
| **SearXNG** | 8888 | Private web search |
| **Langfuse** | 3000 | Observability and tracing (optional) |

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/MARKYMARK55/Spark-Scholar.git
cd Spark-Scholar
cp .env.example env/.env
# Edit env/.env — set VLLM_MODEL_NAME, API keys, etc.

# 2. Create Docker network
docker network create llm-net

# 3. Start core services
docker compose -f core_services/qdrant.yml up -d
docker compose -f core_services/core_services.yml up -d

# 4. Start embedding services
docker compose -f embedding/bge_m3_dense.yml up -d
docker compose -f embedding/bge_m3_sparse.yml up -d
docker compose -f embedding/bge_m3_reranker.yml up -d

# 5. Restore arXiv snapshots from HuggingFace (or build from scratch)
pip install huggingface_hub
hf download MARKYMARK55/spark-scholar-arxiv-snapshots \
  --repo-type dataset \
  --include "snapshots/*.snapshot" \
  --local-dir ./data

for f in ./data/snapshots/*.snapshot; do
  name=$(basename "$f" .snapshot)
  curl -X POST "http://localhost:6333/collections/${name}/snapshots/upload" \
    -H "Content-Type: multipart/form-data" \
    -H "api-key: simple-api-key" \
    -F "snapshot=@${f}"
done

# 6. Open the UI
# http://localhost:8080
```

See [Getting Started](docs/getting_started.md) for full prerequisites, environment configuration, and startup details.

## Hybrid Search Example

Working Python example using the containerized BGE-M3 services. No local model loading needed.

```python
import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Fusion, FusionQuery, NamedSparseVector, NamedVector,
    Prefetch, SparseVector,
)

DENSE_URL  = "http://localhost:8025"
SPARSE_URL = "http://localhost:8035"
QDRANT_URL = "http://localhost:6333"
API_KEY    = "simple-api-key"

query = "attention mechanisms in transformer architectures"

# 1. Dense embedding (OpenAI-compatible endpoint)
dense_resp = httpx.post(
    f"{DENSE_URL}/v1/embeddings",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"model": "bge-m3-embedder", "input": [query]},
)
dense_vec = dense_resp.json()["data"][0]["embedding"]

# 2. Sparse embedding (SPLADE weights)
sparse_resp = httpx.post(
    f"{SPARSE_URL}/encode",
    headers={"Authorization": f"Bearer {API_KEY}"},
    json={"texts": [query]},
)
sp = sparse_resp.json()["embeddings"][0]

# 3. Hybrid search with Qdrant RRF fusion
client = QdrantClient(url=QDRANT_URL, api_key=API_KEY)

results = client.query_points(
    collection_name="arxiv-cs-ml-ai",
    prefetch=[
        Prefetch(
            query=NamedVector(name="dense_embedding", vector=dense_vec),
            using="dense_embedding",
            limit=100,
        ),
        Prefetch(
            query=NamedSparseVector(
                name="sparse_text",
                vector=SparseVector(indices=sp["indices"], values=sp["values"]),
            ),
            using="sparse_text",
            limit=100,
        ),
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=10,
    with_payload=True,
)

for pt in results.points:
    print(f"[{pt.score:.4f}] {pt.payload['title']}")
    print(f"         arXiv:{pt.payload['original_arxiv_id']}")
```

## arXiv Collections

15 domain-specific collections covering every paper in arXiv. Pre-built snapshots available on [HuggingFace](https://huggingface.co/datasets/MARKYMARK55/spark-scholar-arxiv-snapshots).

| Collection | Papers | Topics |
|---|---:|---|
| `arxiv-condmat` | 344,622 | Condensed Matter Physics |
| `arxiv-astro` | 330,467 | Astrophysics |
| `arxiv-math-applied` | 301,943 | Applied Mathematics |
| `arxiv-hep` | 299,312 | High Energy Physics |
| `arxiv-math-pure` | 287,628 | Pure Mathematics |
| `arxiv-nucl-nlin-physother` | 285,526 | Nuclear, Nonlinear, Other Physics |
| `arxiv-cs-systems-theory` | 215,386 | CS Systems, Theory, Databases, Crypto |
| `arxiv-cs-ml-ai` | 206,821 | Machine Learning, AI |
| `arxiv-quantph-grqc` | 199,317 | Quantum Physics, General Relativity |
| `arxiv-cs-cv` | 163,600 | Computer Vision, Graphics |
| `arxiv-stat-eess` | 139,081 | Statistics, Electrical Engineering |
| `arxiv-cs-nlp-ir` | 105,472 | NLP, Information Retrieval, Speech |
| `arxiv-misc` | 90,592 | Robotics, HCI, Social Networks, Other CS |
| `arxiv-qbio-qfin-econ` | 60,426 | Quantitative Biology, Finance, Economics |
| `arxiv-math-phys` | 49,046 | Mathematical Physics |
| **Total** | **3,079,239** | |

116 arXiv categories are mapped across these 15 collections via the [category router](pipeline/router.py).

Also available as database-agnostic Parquet: [arxiv-bge-m3-embeddings](https://huggingface.co/datasets/MARKYMARK55/arxiv-bge-m3-embeddings).

## Document Ingestion

Add your own PDFs, web documentation, and papers into the same vector search infrastructure.

### PDF and Web Doc Ingestion

Each folder in the ingestion directory creates its own searchable collection. Documents are processed through:

1. **Text extraction** (3-tier cascade): Docling for multi-column academic layouts, PyMuPDF for straightforward PDFs, Unstructured with OCR as fallback
2. **Chunking**: Overlapping token-based splits (tiktoken cl100k_base)
3. **Auto-classification**: UMAP dimensionality reduction, HDBSCAN clustering, LLM-generated topic names (Qwen3 via vLLM)
4. **Embedding**: Dense + sparse vectors via BGE-M3 containers
5. **Indexing**: Upserted into Qdrant with rich payload (title, authors, year, page number, topic, chunk text)

Figure extraction and captioning is available via `ingest/06_caption_figures.py`.

**Pre-configured documentation collections**: Python, Rust, JavaScript, Docker, Anthropic, AppleScript, DevOps, Web.

**Query routing**: The router auto-detects all collections from Qdrant. General academic queries search all collections. Code-specific queries route to relevant `docs-*` collections. Domain-specific science queries route to specific arXiv collections.

See [Ingestion Guide](docs/ingestion.md) for full details.

### Citation Graph Expansion

Start from any paper and automatically discover, download, and index its references.

```bash
# Expand to 2 levels deep, only follow highly-cited references
python ingest/08_expand_citations.py \
    --arxiv 2303.08774 \
    --depth 2 \
    --min-citations 50 \
    --max-per-paper 30 \
    --output-dir data/citations/
```

- **Configurable depth**: L1 (your papers) -> L2 (their references) -> L3 (references of references) -> ... Ln
- **Configurable direction**: Follow citations, follow authors (all papers by same authors), or both
- **Quality filters**: Minimum citation count, max references per paper
- **Reference resolution**: Semantic Scholar API -> arXiv API -> Unpaywall (journal open access) -> PDF text extraction (fallback)
- **Resumable**: Re-run the same command and already-processed papers are skipped
- **JSON manifests**: Track every discovered reference and its download/ingestion status

## Academic Tools

20+ Open WebUI workspace tools that search beyond the local index. Install whichever tools you need into Open WebUI.

### Bibliographic Databases (500M+ records combined)

| Tool | Source | Scale |
|---|---|---|
| Semantic Scholar | Ai2 | 200M+ papers, citation graphs, author profiles |
| OpenAlex | OurResearch | 250M+ works, largest open bibliographic catalog |
| PubMed / NCBI | NIH | 40M+ biomedical and life sciences articles |
| CORE | Open University | 200M+ open access papers |
| Europe PMC | EMBL-EBI | 40M+ records, OA papers and preprints |
| Dimensions / Lens.org | Digital Science | Patents, grants, funding data |

### Discovery and Graph Tools

| Tool | What it does |
|---|---|
| Connected Papers | Citation similarity graphs for any paper |
| arXiv Recent Keyword Alert | Recent preprints by keyword and category |
| Consolidated Metadata Fetcher | Unified interface across Semantic Scholar, OpenAlex, CORE, NCBI |

### AI-Powered Research Tools

| Tool | What it does |
|---|---|
| **Ai2 Asta MCP Search** | Semantic snippet search across Ai2's full corpus (new from the Allen Institute for AI) |
| Perplexity Academic Q&A | Research Q&A with live citations |
| Google Gemini | Multimodal academic reasoning (figures, tables, math) |
| Grok | Math, code, obscure references |
| OpenRouter | Access to Claude, Gemini, Grok, Llama for synthesis |

See [Academic Tools Guide](docs/open_webui_tools.md) for installation and API key configuration.

## Documentation

| Guide | Description |
|---|---|
| [Getting Started](docs/getting_started.md) | Prerequisites, environment configuration, startup order |
| [Ingestion](docs/ingestion.md) | arXiv pipeline and custom document ingestion |
| [Search and Retrieval](docs/search_retrieval.md) | Hybrid search mechanics, LangGraph pipeline, caching |
| [Open WebUI RAG Setup](docs/open_webui_rag_setup.md) | Connecting Open WebUI to RAG (built-in vs. proxy) |
| [Academic Tools](docs/open_webui_tools.md) | Installing and configuring academic workspace tools |
| [Embedding Performance](docs/embedding_speed.md) | Throughput benchmarks and optimization |
| [UI Interfaces](docs/ui_interfaces.md) | All web interfaces (Open WebUI, Qdrant, LiteLLM, Langflow, Langfuse) |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and fixes |

## Project Structure

```
spark-scholar/
  core_services/     Docker Compose for Qdrant, LiteLLM, Open WebUI, Redis, SearXNG
  embedding/         BGE-M3 container configs (dense, sparse, reranker)
  ingest/            Ingestion scripts (01-11): download, embed, ingest, expand
  pipeline/          Core Python: hybrid search, routing, embeddings, reranker, LangGraph
  rag_proxy/         OpenAI-compatible RAG proxy server
  query/             Standalone CLI search tools (dense, sparse, hybrid)
  open_webui_tools/  Academic API tools for Open WebUI
  langflow/          Langflow custom components
  images/            Docker image sources (sparse embedder)
  eval/              Retrieval and answer quality evaluation
  docs/              Detailed documentation
  config/            Qdrant configuration
  env/               Environment configuration (.env)
```

## Hardware

Built for NVIDIA DGX Spark (GB10 GPU, 128 GB unified memory). The embedding services and vLLM inference benefit from GPU acceleration. Qdrant, Redis, and the proxy services are CPU-only. The platform will run on any machine with Docker, though embedding and inference will be slower on CPU.

## Related

- [spark-scholar-arxiv-snapshots](https://huggingface.co/datasets/MARKYMARK55/spark-scholar-arxiv-snapshots) -- Pre-built Qdrant snapshots (skip 24h GPU time)
- [arxiv-bge-m3-embeddings](https://huggingface.co/datasets/MARKYMARK55/arxiv-bge-m3-embeddings) -- Database-agnostic Parquet format
- [arxiv-embedding-forge](https://github.com/MARKYMARK55/arxiv-embedding-forge) -- Reproducible embedding build pipeline
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) -- Embedding model
- [Qdrant](https://qdrant.tech/) -- Vector database

## License

MIT License

Underlying arXiv metadata is subject to [arXiv's Terms of Use](https://info.arxiv.org/help/license/index.html) and the original paper licenses.
