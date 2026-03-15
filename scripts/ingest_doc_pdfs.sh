#!/bin/bash
# =============================================================================
# Ingest all PDF collections via Docling → BGE-M3 → Qdrant
# Runs ONE container that processes all collections sequentially.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE="pipeline-deps:latest"
NETWORK="llm-net"
PDFS="$PROJECT_DIR/RAG/PDF_folders"
CONTAINER_NAME="spark-ingest-pdfs"

# Build the list of collection dirs with PDFs
DIRS=()
for dir in "$PDFS"/*/; do
    count=$(find "$dir" -name '*.pdf' 2>/dev/null | wc -l)
    [ "$count" -eq 0 ] && continue
    DIRS+=("$(basename "$dir")")
done

echo "============================================================"
echo "  PDF Ingestion: ${#DIRS[@]} collections to process"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# Remove any stale container
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Build the Python command that loops through all collections
PYCMD='
import subprocess, sys, os, time

pdfs_root = "RAG/PDF_folders"
collections = sys.argv[1:]
ok, fail = 0, 0

for name in collections:
    input_dir = f"{pdfs_root}/{name}"
    count = len([f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")])
    print(f"\n{'='*60}")
    print(f"  Ingesting: {name} ({count} PDFs)")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}", flush=True)

    try:
        subprocess.run([
            sys.executable, "6_document_ingestion/05_ingest_pdfs.py",
            "--input-dir", input_dir,
            "--collection", name,
        ], check=True)
        print(f"  OK: {name}")
        ok += 1
    except subprocess.CalledProcessError:
        print(f"  FAILED: {name}")
        fail += 1

total = ok + fail
print(f"\n{'='*60}")
print(f"  PDF INGESTION COMPLETE: {ok}/{total} succeeded, {fail} failed")
print(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")
'

# Run ONE container for all collections
docker run --rm \
    --name "$CONTAINER_NAME" \
    --entrypoint python3 \
    --network "$NETWORK" \
    -v "$PROJECT_DIR":/workspace \
    -w /workspace \
    -e QDRANT_URL=http://qdrant:6333 \
    -e QDRANT_API_KEY=simple-api-key \
    -e BGE_M3_DENSE_URL=http://bge-m3-dense-embedder:8000 \
    -e BGE_M3_SPARSE_URL=http://bge-m3-sparse-embedder:8001 \
    -e BGE_M3_API_KEY=simple-api-key \
    -e DOCLING_URL=http://docling:5001/v1/convert/file \
    -e VLLM_URL=http://litellm:4000 \
    -e VLLM_API_KEY=simple-api-key \
    -e VLLM_MODEL_NAME=Nemotron-Fast \
    -e PYTHONPATH=/workspace \
    "$IMAGE" \
    -c "$PYCMD" "${DIRS[@]}"

"$SCRIPT_DIR/notify.sh" \
    "PDF Doc Ingestion Complete" \
    "Ingested ${#DIRS[@]} doc collections"
