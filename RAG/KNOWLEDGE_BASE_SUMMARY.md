# Spark-Scholar RAG Knowledge Base — Full Inventory & Quality Analysis

## System Overview

**Platform:** NVIDIA DGX Spark (GB10 — ARM64 Grace CPU + Blackwell GPU, 128GB unified memory)

**RAG Stack:**
- **Vector DB:** Qdrant (hybrid search — dense + sparse vectors)
- **Embeddings:** BGE-M3 (1024-dim dense + learned sparse, separate serving containers)
- **PDF Conversion:** Docling GPU (official `docling-serve-cu130` image, GPU-accelerated OCR + table extraction)
- **Chunking:** tiktoken cl100k_base, 512 tokens with 64-token overlap
- **LLM Backend:** Nemotron-Fast via vLLM + LiteLLM proxy
- **Frontend:** Open WebUI with RAG query generation enabled

**Design Principle:** One TOML config per tool/library → one Qdrant collection. Collection name = config filename. User decides the name, system enforces it.

---

## Three Data Sources

### 1. Web Documentation Crawls (95 TOML configs, unlimited depth)

Full-depth BFS crawl of official documentation sites. HTML → BeautifulSoup text extraction → tiktoken chunking → BGE-M3 embedding → Qdrant upsert.

**Payload schema per point:**
```
chunk_text, source_url, title, section, chunk_idx, collection, tag, source_type="html", ingested_at
```

| Category | # Configs | Collections |
|----------|-----------|-------------|
| **Python Ecosystem** | 20 | python-stdlib, python-numpy, python-pandas, python-scipy, python-sklearn, python-matplotlib, python-seaborn, python-sympy, python-statsmodels, python-polars, python-networkx, python-pyarrow, python-dask, python-pydantic, python-click, python-typer, python-rich, python-pytest, python-h5py, python-joblib |
| **ML/DL Frameworks** | 11 | pytorch-docs, pytorch-lightning, xgboost-docs, lightgbm-docs, catboost-docs, optuna-docs, mlflow-docs, wandb-docs, onnxruntime-docs, pymc-docs, numpyro-docs |
| **HuggingFace** | 8 | hf-transformers, hf-datasets, hf-tokenizers, hf-accelerate, hf-peft, hf-trl, hf-hub, hf-safetensors |
| **LLM/RAG/Agents** | 10 | langchain-docs, langgraph-docs, llamaindex-docs, anthropic-api, openai-api, litellm-docs, langsmith-docs, ragas-docs, docling-docs, vllm-docs |
| **DevTools** | 11 | git-docs, github-docs, vscode-docs, conda-docs, poetry-docs, uv-docs, ruff-docs, mypy-docs, pre-commit-docs, jupyterlab-docs, unstructured-docs |
| **NVIDIA/GPU** | 10 | nvidia-cuda, nvidia-rapids, nvidia-triton, nvidia-tensorrt, nvidia-nemo, nvidia-nim, nvidia-cudnn, nvidia-container-toolkit, numba-docs, cupy-docs |
| **Infrastructure** | 5 | docker-docs, bash-docs, cmake-docs, homebrew-docs, applescript-docs |
| **Web Standards** | 4 | html-docs, css-docs, svg-docs, tailwindcss-docs |
| **JavaScript/TS** | 4 | javascript-docs, typescript-docs, nodejs-docs, nextjs-docs |
| **Web Frameworks** | 6 | fastapi-docs, flask-docs, celery-docs, redis-docs, postgresql-docs, sqlalchemy-docs |
| **Vector DBs** | 3 | qdrant-docs, chroma-docs, weaviate-docs |
| **Other Languages** | 2 | rust-docs, swift-docs |
| **React** | 1 | react-docs |

### 2. PDF Book/Paper Collections (28 folders, 177 PDFs)

Full textbook and paper PDFs processed through Docling GPU (OCR, table extraction, layout analysis) → Markdown → base64 image stripping → section-based chunking → BGE-M3 embedding → Qdrant upsert.

**Payload schema per point:**
```
chunk_text, source_file, section_heading, chunk_idx, topic_id, topic_name (via UMAP+HDBSCAN clustering)
```

| Collection | PDFs | Subject |
|-----------|------|---------|
| Bayes | 2 | Bayesian statistics (Think Bayes, etc.) |
| Coding | 8 | Programming methodology, algorithms |
| Cuda | 3 | CUDA programming guides |
| Data-Mining | 1 | Data mining textbook |
| Deep-Learning | 29 | Deep learning textbooks and papers |
| Docker | 4 | Docker / containerization |
| Financial-Timeseries | 1 | Financial time series analysis |
| Hawkes_Processes | 7 | Hawkes process theory and applications |
| HFT | 1 | High-frequency trading |
| Linux | 7 | Linux administration and internals |
| LLM | 1 | Large language model papers |
| ML | 25 | Machine learning textbooks and papers |
| Neutrino | 10 | Neutrino physics papers |
| NLP | 1 | Natural language processing |
| PGM | 1 | Probabilistic graphical models (Koller & Friedman) |
| Python-Books | 8 | Python programming books |
| Quantum-Field-Theory | 7 | QFT textbooks |
| Reinforcement-Learning | — | RL textbooks and papers |
| Statistics | 2 | Statistics textbooks |
| Voyager | 4 | Voyager spacecraft mission data |
| demo-bayesian-statistics | 2 | Demo collection (testing) |
| demo-computer-science | 5 | Demo collection (testing) |
| demo-python-programming | 1 | Demo collection (testing) |
| docs-data-science | 5 | Data science reference books |
| docs-dev-tools | 3 | Developer tools reference |
| docs-ml-frameworks | 4 | ML framework documentation |
| docs-nvidia-gpu | 2 | NVIDIA GPU programming |
| docs-python-core | 29 | Python core reference books |
| docs-web-backend | 4 | Web backend reference |

### 3. arXiv Paper Collections (16 collections, pre-built from HuggingFace)

Pre-embedded academic paper abstracts and metadata organized by arXiv subject area.

| Collection | Subject Area |
|-----------|-------------|
| arxiv-astro | Astrophysics |
| arxiv-condmat | Condensed matter physics |
| arxiv-cs-cv | Computer vision |
| arxiv-cs-ml-ai | Machine learning & AI |
| arxiv-cs-nlp-ir | NLP & information retrieval |
| arxiv-cs-systems-theory | CS systems & theory |
| arxiv-hep | High-energy physics |
| arxiv-math-applied | Applied mathematics |
| arxiv-math-phys | Mathematical physics |
| arxiv-math-pure | Pure mathematics |
| arxiv-misc | Miscellaneous |
| arxiv-nucl-nlin-physother | Nuclear, nonlinear, other physics |
| arxiv-qbio-qfin-econ | Quantitative bio, finance, economics |
| arxiv-quantph-grqc | Quantum physics & general relativity |
| arxiv-stat-eess | Statistics & electrical engineering |

---

## Total Scale

| Source | Collections | Estimated Points (when fully ingested) |
|--------|------------|---------------------------------------|
| Web crawls | 95 | 500K–2M+ (unlimited depth on major doc sites) |
| PDF books/papers | 28 | 50K–100K |
| arXiv papers | 16 | Already populated (HF download) |
| **Total** | **139** | **~600K–2M+** |

---

## Expected Quality Improvements vs. Base LLM

### What changes when you add this RAG layer:

**1. Eliminates hallucination on API specifics**
- Base LLM: "I think the pandas function is `df.groupby().agg()`..." (may hallucinate argument names, default values, or deprecated syntax)
- With RAG: Retrieves exact current API signatures from `python-pandas` collection with source URL for verification

**2. Version-accurate answers**
- Base LLM: Trained on data up to cutoff date, mixes Python 3.8 and 3.12 syntax, confuses PyTorch 1.x and 2.x APIs
- With RAG: Crawls current `/stable/` docs — always reflects the latest released version of every library

**3. Cross-domain synthesis that no single LLM covers**
- "How do I deploy a PyTorch model with TensorRT on the DGX Spark?" requires knowledge across pytorch-docs, nvidia-tensorrt, nvidia-cuda, docker-docs
- RAG retrieves relevant chunks from ALL applicable collections simultaneously
- Base LLM would give generic advice; RAG grounds it in actual DGX Spark constraints

**4. Niche/specialized domain coverage**
- Hawkes processes, neutrino physics, PGM (Koller & Friedman), QFT — these are poorly covered in LLM training data
- Full textbook PDFs with equations, proofs, and worked examples embedded as retrievable chunks
- arXiv papers provide cutting-edge research that post-dates any LLM training cutoff

**5. Code examples with verified provenance**
- Every chunk has `source_url` or `source_file` — the LLM can cite exactly where the information came from
- User can click through to the original documentation to verify
- Dramatically reduces "plausible but wrong" code generation

**6. Quantitative/financial domain depth**
- Financial time series, HFT, Hawkes processes — these are Mark's professional domain
- Base LLMs have shallow coverage of quantitative finance; the PDF collections contain specialized textbooks and papers
- Combined with arXiv qfin/econ collection, provides serious research-grade retrieval

**7. Infrastructure debugging accuracy**
- Docker, CUDA, Linux, CMake — when debugging container builds, GPU driver issues, or system-level problems
- Base LLM gives generic Stack Overflow-level answers
- RAG retrieves exact documentation sections (e.g., CUDA driver API for specific error codes)

### What RAG does NOT improve:
- **Reasoning ability** — RAG provides better inputs, but the LLM's reasoning quality is unchanged
- **Multi-step planning** — Still depends on the LLM's inherent capability
- **Creative/novel solutions** — RAG retrieves known information; it doesn't generate new ideas
- **Speed** — Adds retrieval latency (~100-500ms per query depending on collection size)
- **Very large context synthesis** — If the answer requires understanding an entire 500-page textbook holistically, chunked retrieval may miss the big picture

### Key architectural advantages of this setup:
1. **Hybrid search** (dense + sparse via BGE-M3) — catches both semantic and keyword matches
2. **Per-collection isolation** — can search specific domains or all collections
3. **Deterministic point IDs** — re-ingesting the same content updates rather than duplicates
4. **Resume support** — progress files track completed URLs, so interrupted crawls pick up where they left off
5. **Full depth crawling** — no arbitrary depth limits means complete documentation coverage
6. **GPU-accelerated PDF processing** — Docling on Blackwell GPU handles OCR, table extraction, and layout analysis at ~10s/PDF vs 3+ minutes on CPU

---

## Estimated Crawl Time & Storage

With 95 web configs at unlimited depth, major sites like PyTorch docs, scikit-learn, PostgreSQL, and MDN will produce tens of thousands of pages each. Rough estimates:

| Site Category | Pages (est.) | Chunks (est.) | Crawl Time (est.) |
|--------------|-------------|--------------|-------------------|
| Large doc sites (pytorch, sklearn, postgresql, MDN×4) | 50K–200K pages | 200K–800K chunks | 2–7 days |
| Medium doc sites (pandas, numpy, docker, etc.) | 10K–50K pages | 40K–200K chunks | 1–3 days |
| Small doc sites (click, typer, rich, etc.) | 5K–20K pages | 20K–80K chunks | 0.5–1 day |
| **Total web crawl** | **65K–270K pages** | **260K–1M+ chunks** | **3–10 days** |

PDF ingestion (177 PDFs via Docling GPU): ~30 minutes total at ~10s/PDF.

Qdrant storage: ~2–8 GB for vectors + payload at this scale. Well within the 128GB unified memory of the DGX Spark.

---

## Collection Inventory (sorted)

```
WEB CRAWLS (95 collections):
  anthropic-api          — Anthropic Claude API docs
  applescript-docs       — AppleScript Language Guide
  bash-docs              — Bash manual + Advanced Bash Scripting Guide
  catboost-docs          — CatBoost gradient boosting
  celery-docs            — Celery distributed task queue
  chroma-docs            — Chroma vector database
  cmake-docs             — CMake build system
  conda-docs             — Conda package manager
  css-docs               — MDN CSS reference
  cupy-docs              — CuPy GPU arrays (NumPy on CUDA)
  docker-docs            — Docker Engine + Compose + Build + Dockerfile ref
  docling-docs           — Docling PDF conversion
  fastapi-docs           — FastAPI web framework
  flask-docs             — Flask web framework
  git-docs               — Git version control
  github-docs            — GitHub platform + Actions
  hf-accelerate          — HuggingFace Accelerate
  hf-datasets            — HuggingFace Datasets
  hf-hub                 — HuggingFace Hub
  hf-peft                — HuggingFace PEFT (LoRA, QLoRA)
  hf-safetensors         — Safetensors format
  hf-tokenizers          — HuggingFace Tokenizers
  hf-transformers        — HuggingFace Transformers
  hf-trl                 — HuggingFace TRL (RLHF)
  homebrew-docs          — Homebrew package manager
  html-docs              — MDN HTML reference
  javascript-docs        — MDN JavaScript + Web APIs
  jupyterlab-docs        — JupyterLab
  langchain-docs         — LangChain framework
  langgraph-docs         — LangGraph agent framework
  langsmith-docs         — LangSmith tracing/eval
  lightgbm-docs          — LightGBM gradient boosting
  litellm-docs           — LiteLLM unified proxy
  llamaindex-docs        — LlamaIndex RAG framework
  mlflow-docs            — MLflow experiment tracking
  mypy-docs              — mypy static type checker
  nextjs-docs            — Next.js React framework
  nodejs-docs            — Node.js API reference
  numba-docs             — Numba JIT compiler
  numpyro-docs           — NumPyro probabilistic programming
  nvidia-container-toolkit — NVIDIA Container Toolkit
  nvidia-cuda            — CUDA Programming + Best Practices + Runtime/Driver API
  nvidia-cudnn           — cuDNN deep learning primitives
  nvidia-nemo            — NVIDIA NeMo LLM training
  nvidia-nim             — NVIDIA NIM microservices
  nvidia-rapids          — cuDF + cuML + cuGraph + install guide
  nvidia-tensorrt        — TensorRT optimization
  nvidia-triton          — Triton Inference Server
  onnxruntime-docs       — ONNX Runtime inference
  openai-api             — OpenAI API docs
  optuna-docs            — Optuna hyperparameter optimization
  poetry-docs            — Poetry dependency management
  postgresql-docs        — PostgreSQL documentation
  pre-commit-docs        — pre-commit hook framework
  pymc-docs              — PyMC Bayesian modeling
  python-click           — Click CLI framework
  python-dask            — Dask parallel computing
  python-h5py            — h5py HDF5 interface
  python-joblib          — Joblib parallel computing
  python-matplotlib      — Matplotlib plotting
  python-networkx        — NetworkX graph algorithms
  python-numpy           — NumPy numerical computing
  python-pandas          — Pandas data analysis
  python-polars          — Polars fast DataFrames
  python-pyarrow         — PyArrow columnar data
  python-pydantic        — Pydantic v2 validation
  python-pytest          — pytest testing
  python-rich            — Rich terminal formatting
  python-scipy           — SciPy scientific computing
  python-seaborn         — Seaborn statistical visualization
  python-sklearn         — scikit-learn machine learning
  python-statsmodels     — Statsmodels statistical modeling
  python-stdlib          — Python 3 standard library
  python-sympy           — SymPy symbolic math
  python-typer           — Typer CLI framework
  pytorch-docs           — PyTorch docs + tutorials
  pytorch-lightning      — PyTorch Lightning
  qdrant-docs            — Qdrant vector database
  ragas-docs             — RAGAS RAG evaluation
  react-docs             — React UI framework
  redis-docs             — Redis data store
  ruff-docs              — Ruff Python linter
  rust-docs              — Rust book + stdlib + reference + Cargo
  sqlalchemy-docs        — SQLAlchemy ORM
  svg-docs               — MDN SVG reference
  swift-docs             — Swift language + SwiftUI + UIKit + Foundation
  tailwindcss-docs       — Tailwind CSS
  typescript-docs        — TypeScript docs
  unstructured-docs      — Unstructured document parsing
  uv-docs                — uv fast Python package manager
  vllm-docs              — vLLM inference engine
  vscode-docs            — VS Code documentation
  wandb-docs             — Weights & Biases
  weaviate-docs          — Weaviate vector database
  xgboost-docs           — XGBoost gradient boosting

PDF COLLECTIONS (28 folders, 177 PDFs):
  Bayes (2), Coding (8), Cuda (3), Data-Mining (1), Deep-Learning (29),
  Docker (4), Financial-Timeseries (1), Hawkes_Processes (7), HFT (1),
  Linux (7), LLM (1), ML (25), Neutrino (10), NLP (1), PGM (1),
  Python-Books (8), Quantum-Field-Theory (7), Reinforcement-Learning (?),
  Statistics (2), Voyager (4), demo-bayesian-statistics (2),
  demo-computer-science (5), demo-python-programming (1),
  docs-data-science (5), docs-dev-tools (3), docs-ml-frameworks (4),
  docs-nvidia-gpu (2), docs-python-core (29), docs-web-backend (4)

arXiv COLLECTIONS (16, pre-built from HuggingFace):
  arxiv-astro, arxiv-condmat, arxiv-cs-cv, arxiv-cs-ml-ai,
  arxiv-cs-nlp-ir, arxiv-cs-systems-theory, arxiv-hep,
  arxiv-math-applied, arxiv-math-phys, arxiv-math-pure,
  arxiv-misc, arxiv-nucl-nlin-physother, arxiv-qbio-qfin-econ,
  arxiv-quantph-grqc, arxiv-stat-eess
```
