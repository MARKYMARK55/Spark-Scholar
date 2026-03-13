#!/usr/bin/env bash
# Launch Qwen3-Coder-Next FP8
# Recipe: qwen3-coder-next-fp8 (@eugr registry)
# VRAM: ~40 GB  |  Best for: code generation, debugging, refactoring
# LiteLLM aliases registered via: litellm_register qwen3-coder
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Launching Qwen3-Coder-Next FP8 (coding-optimised) via SparkRun"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

sparkrun run qwen3-coder-next-fp8

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
