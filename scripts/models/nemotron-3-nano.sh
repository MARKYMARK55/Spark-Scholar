#!/usr/bin/env bash
# Launch NVIDIA Nemotron-3-Nano-30B-A3B NVFP4
# Recipe: nemotron-3-nano-nvfp4 (@eugr registry)
# VRAM: ~40 GB  |  Best for: general research, reasoning, fast responses
# LiteLLM aliases: Fast, Expert, Heavy, Max, Code, Creative
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Launching Nemotron-3-Nano-30B-A3B NVFP4 via SparkRun"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

sparkrun run nemotron-3-nano-nvfp4

echo "-> Waiting for inference endpoint on port 8000..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "  Model ready"
    break
  fi
  sleep 5
done

echo "-> Registering LiteLLM aliases (Fast/Expert/Heavy/Max/Code/Creative)..."
# shellcheck source=../litellm_register.sh
source "$REPO_ROOT/scripts/litellm_register.sh"
litellm_register nemotron

echo ""
sparkrun status
