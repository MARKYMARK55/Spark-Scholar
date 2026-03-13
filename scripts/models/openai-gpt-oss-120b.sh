#!/usr/bin/env bash
# Launch GPT-OSS-120B MXFP4
# Recipe: openai-gpt-oss-120b (@eugr registry)
# VRAM: ~80 GB  |  Requires tp:2 (two DGX Spark nodes or 2-GPU setup)
# LiteLLM aliases registered via: litellm_register gpt-oss
#
# NOTE: This recipe requires tensor parallelism across 2 GPUs (tp:2).
# Ensure you have two DGX Spark nodes or a 2-GPU setup configured before running.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Launching GPT-OSS-120B MXFP4 via SparkRun"
echo "  NOTE: Requires tp:2 (two DGX Spark nodes or 2-GPU setup)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

sparkrun run openai-gpt-oss-120b

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
litellm_register gpt-oss

echo ""
sparkrun status
