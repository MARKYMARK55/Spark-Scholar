#!/usr/bin/env bash
# scripts/smoke_test.sh
# ──────────────────────────────────────────────────────────────────────────────
# Smoke-test the full Spark-Scholar stack. Runs a minimal end-to-end query:
#   1. Check every service health endpoint
#   2. Run a hybrid search via the RAG proxy
#   3. Print a short pass/fail summary
#
# Usage:
#   bash scripts/smoke_test.sh
#   bash scripts/smoke_test.sh --query "transformer attention mechanism"
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

QUERY="${2:-"attention mechanism transformer architecture"}"
PASS=0; FAIL=0

check() {
  local label="$1" url="$2"
  if curl -sf "$url" > /dev/null 2>&1; then
    echo "  ✓  $label"
    PASS=$((PASS + 1))
  else
    echo "  ✗  $label  ($url)"
    FAIL=$((FAIL + 1))
  fi
}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Spark-Scholar Smoke Test"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Service health checks:"
check "Qdrant"             "http://localhost:6333/readyz"
check "BGE-M3 dense"       "http://localhost:8025/health"
check "BGE-M3 sparse"      "http://localhost:8035/health"
check "BGE reranker"       "http://localhost:8020/health"
check "LiteLLM"            "http://localhost:4000/health/readiness"
check "RAG proxy"          "http://localhost:8002/health"
check "Open WebUI"         "http://localhost:8080/health"
check "SearXNG"            "http://localhost:8888/search?q=test&format=json"
check "SparkRun vLLM"      "http://localhost:8000/health"

echo ""
echo "End-to-end RAG query:"
echo "  Query: \"$QUERY\""
RESPONSE=$(curl -s -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer simple-api-key" \
  -d "{
    \"model\": \"local-model\",
    \"messages\": [{\"role\": \"user\", \"content\": \"$QUERY\"}],
    \"max_tokens\": 256
  }" 2>/dev/null)

if echo "$RESPONSE" | python3 -c "
import json, sys
d = json.load(sys.stdin)
choices = d.get('choices', [])
if choices:
    content = choices[0].get('message', {}).get('content', '')
    print('  ✓  Got response (%d chars)' % len(content))
    print()
    print('  Preview:', content[:200].replace('\n', ' '))
    sys.exit(0)
print('  ✗  No choices in response:', json.dumps(d)[:200])
sys.exit(1)
" 2>/dev/null; then
  PASS=$((PASS + 1))
else
  echo "  ✗  RAG query failed or no response"
  FAIL=$((FAIL + 1))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Results: $PASS passed, $FAIL failed"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$FAIL" -gt 0 ]; then
  echo "  Troubleshooting: docs/troubleshooting.md"
  echo "  Stack logs:      docker compose -f core_services/core_services.yml logs --tail 50"
  echo ""
  exit 1
fi
