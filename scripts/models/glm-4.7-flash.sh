#!/usr/bin/env bash
# Launch GLM-4.7-Flash AWQ
# Recipe: glm-4.7-flash-awq (@eugr registry)
# VRAM: ~8 GB  |  Best for: ultra-light, fastest responses, low VRAM, tool routing
# LiteLLM aliases registered via: litellm_register glm
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Launching GLM-4.7-Flash AWQ (ultra-light) via SparkRun"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

sparkrun run glm-4.7-flash-awq

echo "-> Waiting for inference endpoint on port 8000..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "  Model ready"
    break
  fi
  sleep 5
done

echo "-> Registering LiteLLM aliases..."
# shellcheck source=../litellm_register.sh
source "$REPO_ROOT/scripts/litellm_register.sh"
litellm_register glm

echo ""
sparkrun status
