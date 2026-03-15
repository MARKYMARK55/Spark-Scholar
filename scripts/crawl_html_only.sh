#!/bin/bash
# =============================================================================
# Crawl HTML-only documentation sites (libraries without PDF downloads)
# Plus NVIDIA/RAPIDS and Qdrant docs
# =============================================================================
# Run this after downloading available PDFs via browser.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE="pipeline-deps:latest"
NETWORK="llm-net"

run_crawl() {
    local config="$1"
    local name=$(basename "$config")
    echo ""
    echo "============================================================"
    echo "  Crawling: $name"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    local container_name="spark-crawl-${name%.toml}"
    docker rm -f "$container_name" 2>/dev/null || true

    docker run --rm \
        --name "$container_name" \
        --entrypoint python3 \
        --network "$NETWORK" \
        -v "$PROJECT_DIR":/workspace \
        -w /workspace \
        -e QDRANT_URL=http://qdrant:6333 \
        -e QDRANT_API_KEY=simple-api-key \
        -e BGE_M3_DENSE_URL=http://bge-m3-dense-embedder:8000 \
        -e BGE_M3_SPARSE_URL=http://bge-m3-sparse-embedder:8001 \
        -e BGE_M3_API_KEY=simple-api-key \
        -e PYTHONPATH=/workspace \
        "$IMAGE" \
        6_document_ingestion/07_ingest_html_docs.py --config "RAG/crawl_targets/$name" 2>&1

    echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
}

echo "=== HTML-Only Documentation Crawl ==="
echo "    Started: $(date '+%Y-%m-%d %H:%M:%S')"

# Main HTML-only libraries (Pandas, scikit-learn, PyTorch, HuggingFace, etc.)
run_crawl html-only-libs.toml

# NVIDIA/RAPIDS
run_crawl nvidia-rapids.toml

# Qdrant docs (re-populate lost collection)
run_crawl qdrant-docs.toml

echo ""
echo "============================================================"
echo "  ALL CRAWLS COMPLETE: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Send notification
"$SCRIPT_DIR/notify.sh" \
    "HTML Doc Crawl Complete" \
    "All HTML-only documentation crawls finished at $(date '+%Y-%m-%d %H:%M:%S')"
