#!/usr/bin/env bash
# scripts/stop_indexing_mode.sh
# Switch back from indexing mode to production mode for the dense embedder.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "→ Stopping indexing embedder (port 8026)..."
docker compose -f "$REPO_ROOT/embedding/bge_m3_dense_indexing.yml" down 2>/dev/null || true

echo "→ Starting production dense embedder (port 8025, 12% GPU)..."
docker compose -f "$REPO_ROOT/embedding/bge_m3_dense.yml" up -d

for i in $(seq 1 20); do
  if curl -sf http://localhost:8025/health > /dev/null 2>&1; then
    echo "✓ Production embedder ready on port 8025"
    break
  fi
  sleep 3
done
