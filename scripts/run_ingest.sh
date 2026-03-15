#!/bin/bash
# =============================================================================
# Run ingestion scripts inside the pipelines container
# =============================================================================
# Usage:
#   ./scripts/run_ingest.sh 6_document_ingestion/07_ingest_html_docs.py --config RAG/crawl_targets/qdrant-docs.toml
#   ./scripts/run_ingest.sh 6_document_ingestion/12_reingest_pdfs.py --collections demo-bayesian-statistics
#   ./scripts/run_ingest.sh 6_document_ingestion/05_ingest_pdfs.py --input-dir RAG/PDF_folders/demo-bayesian-statistics --collection demo-bayesian-statistics
#
# All paths are relative to the spark-scholar project root.
# The script mounts spark-scholar at /workspace and runs on the llm-net Docker network.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Container image (same as pipelines service)
IMAGE="pipeline-deps:latest"

# Docker network where Qdrant + embedding services live
NETWORK="llm-net"

# Derive a container name from the script being run
SCRIPT_NAME="${1:-ingest}"
CONTAINER_NAME="spark-ingest-$(basename "$SCRIPT_NAME" .py)"

# Remove any stale container with same name
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo "=== Running inside container: $CONTAINER_NAME ==="
echo "    Command: $@"
echo "    Image: $IMAGE"
echo ""

exec docker run --rm \
  --name "$CONTAINER_NAME" \
  --entrypoint python3 \
  --network "$NETWORK" \
  --gpus all \
  -v "$PROJECT_DIR":/workspace \
  -w /workspace \
  -e QDRANT_URL=http://qdrant:6333 \
  -e QDRANT_API_KEY=simple-api-key \
  -e BGE_M3_DENSE_URL=http://bge-m3-dense-embedder:8000 \
  -e BGE_M3_SPARSE_URL=http://bge-m3-sparse-embedder:8001 \
  -e RERANKER_URL=http://bge-m3-reranker:8000/rerank \
  -e BGE_M3_API_KEY=simple-api-key \
  -e DOCLING_URL=http://docling:5001/v1/convert/file \
  -e PYTHONPATH=/workspace \
  "$IMAGE" \
  "$@"
