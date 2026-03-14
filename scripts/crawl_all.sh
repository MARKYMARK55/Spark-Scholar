#!/bin/bash
# =============================================================================
# Crawl all TOML targets sequentially inside the pipelines container
# =============================================================================
# Usage:
#   ./scripts/crawl_all.sh           # crawl all configs
#   ./scripts/crawl_all.sh --notify  # crawl all + send email notification
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TOML_DIR="$PROJECT_DIR/RAG/crawl_targets"

IMAGE="pipeline-deps:latest"
NETWORK="llm-net"

NOTIFY=false
if [[ "${1:-}" == "--notify" ]]; then
    NOTIFY=true
fi

echo "=== Crawling all TOML targets ==="
echo "    Config dir: $TOML_DIR"
echo "    Image: $IMAGE"
echo ""

TOTAL=0
OK=0
FAIL=0

for toml in "$TOML_DIR"/*.toml; do
    name=$(basename "$toml")
    echo ""
    echo "============================================================"
    echo "  Crawling: $name"
    echo "============================================================"

    if docker run --rm \
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
        ingest/07_ingest_html_docs.py --config "RAG/crawl_targets/$name" 2>&1; then
        echo "  OK: $name"
        OK=$((OK + 1))
    else
        echo "  FAILED: $name"
        FAIL=$((FAIL + 1))
    fi
    TOTAL=$((TOTAL + 1))
done

echo ""
echo "============================================================"
echo "  CRAWL SUMMARY: $OK/$TOTAL succeeded, $FAIL failed"
echo "============================================================"

if $NOTIFY; then
    "$SCRIPT_DIR/notify.sh" \
        "Doc Crawl Complete" \
        "Crawled $TOTAL configs: $OK succeeded, $FAIL failed"
fi
