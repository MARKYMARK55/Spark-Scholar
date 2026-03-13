#!/usr/bin/env bash
# Launch Qwen3-Coder-Next INT4 AutoRound
# Recipe: qwen3-coder-next-int4-autoround (@eugr registry)
# VRAM: ~25 GB (INT4 quantised)  |  Best for: code generation, debugging, refactoring
# LiteLLM aliases registered via: litellm_register qwen3-coder
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Launching Qwen3-Coder-Next INT4 AutoRound via SparkRun"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

sparkrun run qwen3-coder-next-int4-autoround

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
litellm_register qwen3-coder

echo ""
sparkrun status
