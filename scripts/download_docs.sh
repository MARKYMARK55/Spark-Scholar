#!/bin/bash
# =============================================================================
# Download official PDF documentation for Python ecosystem libraries
# =============================================================================
# Downloads from ReadTheDocs, official sites, and archives.
# Each PDF goes into the appropriate RAG/pdfs/docs-* directory.
# These will be ingested via Docling → section-aware chunking → BGE-M3 embeddings.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PDFS="$PROJECT_DIR/RAG/pdfs"

download() {
    local dir="$1"
    local url="$2"
    local filename="${3:-$(basename "$url")}"
    local dest="$PDFS/$dir/$filename"

    if [[ -f "$dest" ]]; then
        echo "  SKIP (exists): $filename"
        return 0
    fi

    echo "  Downloading: $filename"
    if curl -sL --fail --max-time 120 -o "$dest" "$url"; then
        local size=$(du -h "$dest" | cut -f1)
        echo "    OK ($size): $dest"
    else
        echo "    FAILED: $url"
        rm -f "$dest"
        return 0  # don't fail the whole script
    fi
}

echo "=== Downloading Python Ecosystem Documentation PDFs ==="
echo ""

# -----------------------------------------------------------------------
# Python Core
# -----------------------------------------------------------------------
echo "--- Python Core ---"
mkdir -p "$PDFS/docs-python-core"
download docs-python-core "https://docs.python.org/3/archives/python-3.13.3-docs-pdf-a4.zip" "python-3.13-docs.zip"

# -----------------------------------------------------------------------
# Data Science & Numerics
# -----------------------------------------------------------------------
echo "--- Data Science & Numerics ---"
mkdir -p "$PDFS/docs-data-science"

# NumPy — ReadTheDocs
download docs-data-science "https://numpy.org/doc/2.2/numpy-ref.pdf" "numpy-reference.pdf"
download docs-data-science "https://numpy.org/doc/2.2/numpy-user.pdf" "numpy-user-guide.pdf"

# Pandas
download docs-data-science "https://pandas.pydata.org/docs/pandas.pdf" "pandas.pdf"

# SciPy
download docs-data-science "https://docs.scipy.org/doc/scipy/scipy-ref.pdf" "scipy-reference.pdf"

# Matplotlib
download docs-data-science "https://matplotlib.org/Matplotlib.pdf" "matplotlib.pdf"

# Seaborn — no official PDF, skip

# SymPy
download docs-data-science "https://docs.sympy.org/latest/_downloads/sympy-docs-pdf-1.13.pdf" "sympy.pdf"

# NetworkX
download docs-data-science "https://networkx.org/documentation/stable/_downloads/networkx_reference.pdf" "networkx.pdf"

# Statsmodels
download docs-data-science "https://www.statsmodels.org/stable/statsmodels.pdf" "statsmodels.pdf"

# Polars
download docs-data-science "https://docs.pola.rs/polars.pdf" "polars.pdf"

# Dask
download docs-data-science "https://dask.readthedocs.io/_/downloads/en/stable/pdf/" "dask.pdf"

# h5py
download docs-data-science "https://h5py.readthedocs.io/_/downloads/en/stable/pdf/" "h5py.pdf"

# Arrow/PyArrow
download docs-data-science "https://arrow.apache.org/docs/python/pyarrow.pdf" "pyarrow.pdf"

# -----------------------------------------------------------------------
# ML & Deep Learning Frameworks
# -----------------------------------------------------------------------
echo "--- ML & Deep Learning ---"
mkdir -p "$PDFS/docs-ml-frameworks"

# PyTorch — huge docs, HTML only, no single PDF

# scikit-learn
download docs-ml-frameworks "https://scikit-learn.org/stable/_downloads/scikit-learn-docs.pdf" "scikit-learn.pdf"

# XGBoost
download docs-ml-frameworks "https://xgboost.readthedocs.io/_/downloads/en/stable/pdf/" "xgboost.pdf"

# LightGBM
download docs-ml-frameworks "https://lightgbm.readthedocs.io/_/downloads/en/stable/pdf/" "lightgbm.pdf"

# CatBoost — no PDF, HTML only

# Optuna
download docs-ml-frameworks "https://optuna.readthedocs.io/_/downloads/en/stable/pdf/" "optuna.pdf"

# MLflow
download docs-ml-frameworks "https://mlflow.org/docs/latest/mlflow.pdf" "mlflow.pdf"

# PyMC
download docs-ml-frameworks "https://www.pymc.io/projects/docs/en/latest/pymc.pdf" "pymc.pdf"

# ONNX Runtime
download docs-ml-frameworks "https://onnxruntime.ai/docs/onnxruntime.pdf" "onnxruntime.pdf"

# HuggingFace Transformers — ReadTheDocs
download docs-ml-frameworks "https://huggingface.co/docs/transformers/main/en/transformers.pdf" "transformers.pdf"

# Lightning
download docs-ml-frameworks "https://lightning.ai/docs/pytorch/stable/pytorch_lightning.pdf" "pytorch-lightning.pdf"

# vLLM
download docs-ml-frameworks "https://docs.vllm.ai/_/downloads/en/latest/pdf/" "vllm.pdf"

# -----------------------------------------------------------------------
# LLM / RAG / Agents
# -----------------------------------------------------------------------
echo "--- LLM / RAG / Agents ---"
mkdir -p "$PDFS/docs-llm-agents"

# LangChain — no single PDF

# LlamaIndex
download docs-llm-agents "https://docs.llamaindex.ai/en/stable/llama-index.pdf" "llamaindex.pdf"

# Pydantic
download docs-llm-agents "https://docs.pydantic.dev/latest/pydantic.pdf" "pydantic.pdf"

# Chroma
download docs-llm-agents "https://docs.trychroma.com/chroma.pdf" "chroma.pdf"

# RAGAS
download docs-llm-agents "https://docs.ragas.io/_/downloads/en/stable/pdf/" "ragas.pdf"

# Docling
download docs-llm-agents "https://ds4sd.github.io/docling/docling.pdf" "docling.pdf"

# Unstructured
download docs-llm-agents "https://unstructured-io.github.io/unstructured/unstructured.pdf" "unstructured.pdf"

# -----------------------------------------------------------------------
# Web / Backend
# -----------------------------------------------------------------------
echo "--- Web / Backend ---"
mkdir -p "$PDFS/docs-web-backend"

# FastAPI
download docs-web-backend "https://fastapi.tiangolo.com/fastapi.pdf" "fastapi.pdf"

# Flask
download docs-web-backend "https://flask.palletsprojects.com/en/stable/flask-docs.pdf" "flask.pdf"

# SQLAlchemy
download docs-web-backend "https://docs.sqlalchemy.org/20/sqlalchemy_20.pdf" "sqlalchemy.pdf"

# Celery
download docs-web-backend "https://docs.celeryq.dev/_/downloads/en/stable/pdf/" "celery.pdf"

# Requests
download docs-web-backend "https://requests.readthedocs.io/_/downloads/en/stable/pdf/" "requests.pdf"

# httpx
download docs-web-backend "https://www.python-httpx.org/httpx.pdf" "httpx.pdf"

# aiohttp
download docs-web-backend "https://docs.aiohttp.org/_/downloads/en/stable/pdf/" "aiohttp.pdf"

# Click
download docs-web-backend "https://click.palletsprojects.com/en/stable/click-docs.pdf" "click.pdf"

# -----------------------------------------------------------------------
# Developer Tools
# -----------------------------------------------------------------------
echo "--- Developer Tools ---"
mkdir -p "$PDFS/docs-dev-tools"

# pytest
download docs-dev-tools "https://docs.pytest.org/_/downloads/en/stable/pdf/" "pytest.pdf"

# mypy
download docs-dev-tools "https://mypy.readthedocs.io/_/downloads/en/stable/pdf/" "mypy.pdf"

# Rich
download docs-dev-tools "https://rich.readthedocs.io/_/downloads/en/stable/pdf/" "rich.pdf"

# Typer
download docs-dev-tools "https://typer.tiangolo.com/typer.pdf" "typer.pdf"

# IPython
download docs-dev-tools "https://ipython.readthedocs.io/_/downloads/en/stable/pdf/" "ipython.pdf"

# Conda
download docs-dev-tools "https://docs.conda.io/_/downloads/en/latest/pdf/" "conda.pdf"

# Poetry
download docs-dev-tools "https://python-poetry.org/docs/poetry.pdf" "poetry.pdf"

# JupyterLab
download docs-dev-tools "https://jupyterlab.readthedocs.io/_/downloads/en/stable/pdf/" "jupyterlab.pdf"

# Joblib
download docs-dev-tools "https://joblib.readthedocs.io/_/downloads/en/stable/pdf/" "joblib.pdf"

# -----------------------------------------------------------------------
# NVIDIA / GPU
# -----------------------------------------------------------------------
echo "--- NVIDIA / GPU ---"
mkdir -p "$PDFS/docs-nvidia-gpu"

# Numba
download docs-nvidia-gpu "https://numba.readthedocs.io/_/downloads/en/stable/pdf/" "numba.pdf"

# CuPy
download docs-nvidia-gpu "https://docs.cupy.dev/_/downloads/en/stable/pdf/" "cupy.pdf"

# Qdrant
echo "--- Qdrant ---"
download docs-nvidia-gpu "https://qdrant.tech/documentation/qdrant.pdf" "qdrant.pdf"

# -----------------------------------------------------------------------
echo ""
echo "=== Download complete ==="
echo ""
for dir in "$PDFS"/docs-*/; do
    count=$(find "$dir" -name '*.pdf' -o -name '*.zip' 2>/dev/null | wc -l)
    size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    echo "  $(basename "$dir"): $count files, $size"
done
