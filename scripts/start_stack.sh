#!/usr/bin/env bash
# scripts/start_stack.sh
# ──────────────────────────────────────────────────────────────────────────────
# Full stack startup for arxiv-rag on DGX Spark.
# Starts all services in dependency order with health checks between stages.
#
# Usage:
#   ./scripts/start_stack.sh              # full stack
#   ./scripts/start_stack.sh embedding    # embedding services only
#   ./scripts/start_stack.sh core         # core services only (needs embedding running)
#
# Prerequisites:
#   - docker network create llm-net       (once only)
#   - env/.env filled in                  (cp .env.example env/.env)
#   - vLLM inference model running on port 8000 (started separately)
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_DIR="$REPO_ROOT/docker"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "         arxiv-rag Stack Startup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Ensure llm-net exists ──────────────────────────────────────────────────────
if ! docker network ls | grep -q llm-net; then
  echo "→ Creating llm-net network..."
  docker network create llm-net
fi

MODE="${1:-full}"

# ── Stop existing containers cleanly ──────────────────────────────────────────
if [[ "$MODE" == "full" ]]; then
  echo "→ Stopping existing containers..."
  docker compose -f "$DOCKER_DIR/qdrant_standalone.yml"             down --remove-orphans 2>/dev/null || true
  docker compose -f "$DOCKER_DIR/bge_m3_dense_embedder.yml"         down --remove-orphans 2>/dev/null || true
  docker compose -f "$DOCKER_DIR/bge_m3_dense_embedder_indexing.yml" down --remove-orphans 2>/dev/null || true
  docker compose -f "$DOCKER_DIR/bge_m3_sparse_embedder.yml"        down --remove-orphans 2>/dev/null || true
  docker compose -f "$DOCKER_DIR/bge_m3_reranker.yml"               down --remove-orphans 2>/dev/null || true
  docker compose -f "$DOCKER_DIR/core_services.yml"                  down --remove-orphans 2>/dev/null || true
fi

# ── Stage 1: Qdrant ───────────────────────────────────────────────────────────
if [[ "$MODE" == "full" || "$MODE" == "qdrant" ]]; then
  echo ""
  echo "→ Starting Qdrant..."
  docker compose -f "$DOCKER_DIR/qdrant_standalone.yml" up -d
  echo "  Waiting for Qdrant to be ready..."
  for i in $(seq 1 20); do
    if curl -sf http://localhost:6333/readyz > /dev/null 2>&1; then
      echo "  ✓ Qdrant ready"
      break
    fi
    sleep 3
  done
fi

# ── Stage 2: Embedding stack ──────────────────────────────────────────────────
if [[ "$MODE" == "full" || "$MODE" == "embedding" ]]; then
  echo ""
  echo "→ Starting BGE-M3 dense embedder (port 8025, production mode)..."
  docker compose -f "$DOCKER_DIR/bge_m3_dense_embedder.yml" up -d

  echo "→ Building and starting BGE-M3 sparse embedder (port 8035)..."
  docker compose -f "$DOCKER_DIR/bge_m3_sparse_embedder.yml" build --pull bge-m3-sparse-embedder 2>/dev/null || true
  docker compose -f "$DOCKER_DIR/bge_m3_sparse_embedder.yml" up -d

  echo "→ Starting BGE-M3 reranker (port 8020)..."
  docker compose -f "$DOCKER_DIR/bge_m3_reranker.yml" up -d

  echo "  Waiting for embedding services (model download may take 2-5 min first run)..."
  for i in $(seq 1 40); do
    DENSE_OK=false; SPARSE_OK=false; RERANK_OK=false
    curl -sf http://localhost:8025/health > /dev/null 2>&1 && DENSE_OK=true
    curl -sf http://localhost:8035/health > /dev/null 2>&1 && SPARSE_OK=true
    curl -sf http://localhost:8020/health > /dev/null 2>&1 && RERANK_OK=true
    if $DENSE_OK && $SPARSE_OK && $RERANK_OK; then
      echo "  ✓ All embedding services ready"
      break
    fi
    printf "  dense:%-4s  sparse:%-4s  reranker:%-4s  (%ds)\r" \
      "$($DENSE_OK && echo "✓" || echo "…")" \
      "$($SPARSE_OK && echo "✓" || echo "…")" \
      "$($RERANK_OK && echo "✓" || echo "…")" \
      "$((i * 5))"
    sleep 5
  done
  echo ""
fi

# ── Stage 3: Core services ────────────────────────────────────────────────────
if [[ "$MODE" == "full" || "$MODE" == "core" ]]; then
  echo ""
  echo "→ Starting core services (LiteLLM, Open WebUI, Pipelines, Langflow)..."
  docker compose -f "$DOCKER_DIR/core_services.yml" up -d

  echo "  Waiting for LiteLLM to be ready..."
  MAX=30; ATTEMPT=1
  until curl -sf http://localhost:4000/health/readiness > /dev/null 2>&1; do
    if [ $ATTEMPT -ge $MAX ]; then
      echo "  ✗ LiteLLM not ready after ${MAX} attempts"
      echo "  Check logs: docker compose -f docker/core_services.yml logs --tail 50 litellm"
      exit 1
    fi
    printf "  Waiting for LiteLLM... (%d/%d)\r" $ATTEMPT $MAX
    sleep 3
    ATTEMPT=$((ATTEMPT + 1))
  done
  echo "  ✓ LiteLLM ready"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Stack ready. Service URLs:"
echo "  Qdrant dashboard    → http://localhost:6333/dashboard"
echo "  Open WebUI          → http://localhost:8080"
echo "  LiteLLM proxy       → http://localhost:4000"
echo "  RAG proxy           → http://localhost:8002  (start separately)"
echo "  Langflow            → http://localhost:7860"
echo "  Langfuse            → http://localhost:3000  (start with: --profile langfuse)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Show available models via LiteLLM
echo ""
echo "Available models:"
curl -s http://localhost:4000/v1/models \
  -H "Authorization: Bearer simple-api-key" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); [print('  •', m['id']) for m in d.get('data',[])]" \
  2>/dev/null || echo "  (LiteLLM models endpoint not yet available)"
