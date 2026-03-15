#!/bin/bash
# =============================================================================
# Crawl all TOML targets sequentially inside ONE container
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
CONTAINER_NAME="spark-crawl-docs"

NOTIFY=false
if [[ "${1:-}" == "--notify" ]]; then
    NOTIFY=true
fi

# Build list of TOML files
TOMLS=()
for toml in "$TOML_DIR"/*.toml; do
    TOMLS+=("$(basename "$toml")")
done

echo "=== Crawling all TOML targets ==="
echo "    Config dir: $TOML_DIR"
echo "    Targets: ${#TOMLS[@]}"
echo "    Image: $IMAGE"
echo ""

# Remove any stale container
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

PYCMD='
import subprocess, sys, time

configs = sys.argv[1:]
ok, fail = 0, 0

for name in configs:
    print(f"\n{'='*60}")
    print(f"  Crawling: {name}")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}", flush=True)

    try:
        subprocess.run([
            sys.executable, "6_document_ingestion/07_ingest_html_docs.py",
            "--config", f"RAG/crawl_targets/{name}",
        ], check=True)
        print(f"  OK: {name}")
        ok += 1
    except subprocess.CalledProcessError:
        print(f"  FAILED: {name}")
        fail += 1

total = ok + fail
print(f"\n{'='*60}")
print(f"  CRAWL SUMMARY: {ok}/{total} succeeded, {fail} failed")
print(f"  Finished: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*60}")
'

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
    -e PYTHONPATH=/workspace \
    "$IMAGE" \
    -c "$PYCMD" "${TOMLS[@]}"

if $NOTIFY; then
    "$SCRIPT_DIR/notify.sh" \
        "Doc Crawl Complete" \
        "Crawled ${#TOMLS[@]} configs"
fi
