#!/usr/bin/env bash
# scripts/start_indexing_mode.sh
# ──────────────────────────────────────────────────────────────────────────────
# Switch the dense embedder to high-throughput indexing mode (50% GPU).
# Use this when running bulk ingestion of the Arxiv corpus.
#
# Indexing mode vs production mode:
#   Production : gpu-memory-utilization=0.12, max-num-seqs=8   → ~350 docs/s
#   Indexing   : gpu-memory-utilization=0.50, max-num-seqs=64  → ~1,100 docs/s
#
# The sparse embedder and reranker are unaffected — they continue running.
# The inference model (vLLM on port 8000) is also unaffected.
#
# After bulk indexing completes, run: ./scripts/stop_indexing_mode.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_DIR="$REPO_ROOT/docker"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Switching to INDEXING MODE (50% GPU, high throughput)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "→ Stopping production dense embedder (port 8025)..."
docker compose -f "$DOCKER_DIR/bge_m3_dense_embedder.yml" down 2>/dev/null || true

echo "→ Starting indexing dense embedder (port 8026, 50% GPU)..."
docker compose -f "$DOCKER_DIR/bge_m3_dense_embedder_indexing.yml" up -d

echo "  Waiting for indexing embedder..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:8026/health > /dev/null 2>&1; then
    echo "  ✓ Indexing embedder ready on port 8026"
    break
  fi
  sleep 5
done

echo ""
echo "  Now run your indexing pipeline:"
echo "    python ingest/03_ingest_dense.py --input data/arxiv-metadata.jsonl \\"
echo "        --batch-size 256 --embedder-url http://localhost:8026"
echo ""
echo "  When done: ./scripts/stop_indexing_mode.sh"
