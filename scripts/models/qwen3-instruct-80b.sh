#!/usr/bin/env bash
# Launch Qwen3-Instruct-80B FP8
# Recipe: qwen3-instruct-80b (@eugr registry)
# VRAM: ~80 GB  |  Best for: general research, deep reasoning, long context
# LiteLLM aliases registered via: litellm_register qwen3-80b
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Launching Qwen3-Instruct-80B via SparkRun"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

sparkrun run qwen3-instruct-80b

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
litellm_register qwen3-80b

echo ""
sparkrun status
