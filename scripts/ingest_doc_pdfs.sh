#!/bin/bash
# =============================================================================
# Ingest all PDF collections via Docling → BGE-M3 → Qdrant
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE="pipeline-deps:latest"
NETWORK="llm-net"
PDFS="$PROJECT_DIR/RAG/PDF_folders"

OK=0
FAIL=0

for dir in "$PDFS"/*/; do
    name=$(basename "$dir")
    count=$(find "$dir" -name '*.pdf' 2>/dev/null | wc -l)
    [ "$count" -eq 0 ] && continue

    echo ""
    echo "============================================================"
    echo "  Ingesting: $name ($count PDFs)"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"

    container_name="spark-ingest-$name"
    docker rm -f "$container_name" 2>/dev/null || true

    if docker run --rm \
        --name "$container_name" \
        --entrypoint python3 \
        --network "$NETWORK" \
        --gpus all \
        -v "$PROJECT_DIR":/workspace \
        -w /workspace \
        -e QDRANT_URL=http://qdrant:6333 \
        -e QDRANT_API_KEY=simple-api-key \
        -e BGE_M3_DENSE_URL=http://bge-m3-dense-embedder:8000 \
        -e BGE_M3_SPARSE_URL=http://bge-m3-sparse-embedder:8001 \
        -e BGE_M3_API_KEY=simple-api-key \
        -e DOCLING_URL=http://docling:5001/convert \
        -e VLLM_URL=http://litellm:4000 \
        -e VLLM_API_KEY=simple-api-key \
        -e VLLM_MODEL_NAME=Nemotron-Fast \
        -e PYTHONPATH=/workspace \
        "$IMAGE" \
        6_document_ingestion/05_ingest_pdfs.py \
            --input-dir "RAG/PDF_folders/$name" \
            --collection "$name" 2>&1; then
        echo "  OK: $name"
        OK=$((OK + 1))
    else
        echo "  FAILED: $name"
        FAIL=$((FAIL + 1))
    fi
done

TOTAL=$((OK + FAIL))
echo ""
echo "============================================================"
echo "  PDF INGESTION COMPLETE: $OK/$TOTAL succeeded, $FAIL failed"
echo "  Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

"$SCRIPT_DIR/notify.sh" \
    "PDF Doc Ingestion Complete" \
    "Ingested $TOTAL doc collections: $OK succeeded, $FAIL failed"
