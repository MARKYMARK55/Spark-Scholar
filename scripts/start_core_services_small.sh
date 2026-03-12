# ───────────────────────────────────────────────────────────────
# 1. Clean up any old/stuck containers
# ───────────────────────────────────────────────────────────────

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "       Restarting (Embedding + Reranking + Core Services)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"


cd ~
cd vllm/model_stack/router_embed_rerank

# ───────────────────────────────────────────────────────────────
# Clean up: stop & remove services — one file at a time
# ───────────────────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────
# Clean up: stop & remove services — one file at a time
# ───────────────────────────────────────────────────────────────

echo "→ Stopping & removing services..."

docker compose -f start_phi_4_mini.yml                   down --remove-orphans || true
docker compose -f start_bge_m3_dense_embedder.yml        down --remove-orphans || true
docker compose -f start_bge_m3_sparse_embedder.yml       down --remove-orphans || true
docker compose -f start_bge_reranker_v2_m3.yml           down --remove-orphans || true
docker compose -f start_bge-m3_embedder_indexing.yml     down --remove-orphans || true

echo "→ Cleanup finished (non-critical warnings ignored)"

# ───────────────────────────────────────────────────────────────
# Start only the embedders + reranker
# ───────────────────────────────────────────────────────────────



docker compose -f start_bge_m3_dense_embedder.yml up -d


# Build only rebuilds changed layers — usually < 2 seconds
# Docker automatically uses cache from previous build
echo "  → Checking for changes and building if needed..."
docker compose -f start_bge_m3_sparse_embedder.yml build --pull bge-m3-sparse-embedder || echo "Build step skipped or failed (non-fatal)"

# Start / restart the container
docker compose -f start_bge_m3_sparse_embedder.yml up -d


docker compose -f start_bge_reranker_v2_m3.yml up -d

echo "→ Embedder + reranker startup commands completed"

# ============== Start Core Services ===================

cd ~
cd vllm/model_stack/core_services

docker compose -f start_core_services.yml down

docker compose -f start_core_services.yml up -d

docker compose -f start_qdrant.yml up -d 


echo "→ Waiting for LiteLLM to become ready..."

MAX_ATTEMPTS=30
INTERVAL=3
ATTEMPT=1

until curl -s -f -o /dev/null \
  --connect-timeout 3 \
  --max-time 5 \
  http://127.0.0.1:4000/health/readiness; do
  if [ $ATTEMPT -ge $MAX_ATTEMPTS ]; then
    echo "❌ ERROR: LiteLLM did not become ready after $MAX_ATTEMPTS attempts (${MAX_ATTEMPTS}×${INTERVAL}s = ~${MAX_ATTEMPTS*INTERVAL}s)"
    echo "   Check logs: docker compose -f start_core_services.yml logs --tail 80"
    exit 1
  fi

  printf "Waiting for LiteLLM... (%d/%d)\n" $ATTEMPT $MAX_ATTEMPTS
  sleep $INTERVAL
  ATTEMPT=$((ATTEMPT + 1))
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "                    LiteLLM is Ready                              "
echo "                    Available Models                                "
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

curl -s http://127.0.0.1:4000/v1/models \
  -H "Authorization: Bearer my-simple-api" \
  --connect-timeout 10 \
  --max-time 30 \
  | jq -r '.data[].id' 2>/dev/null \
  | sort \
  | column -t -N "Model" -O "1" || {
      echo "  (no models or jq not installed)"
      echo "  Raw response:"
      curl -s http://127.0.0.1:4000/v1/models -H "Authorization: Bearer my-simple-api"
  }

echo "----------------------------------------"