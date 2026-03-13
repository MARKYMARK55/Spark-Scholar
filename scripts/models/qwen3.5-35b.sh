#!/usr/bin/env bash
# Launch Qwen3.5-35B-A3B FP8 (MoE)
# Recipe: qwen3.5-35b-a3b-fp8 (@eugr registry)
# VRAM: ~35 GB (MoE — only active parameters loaded)
# LiteLLM aliases: not yet available via litellm_register — see note below
set -euo pipefail

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Launching Qwen3.5-35B-A3B FP8 (MoE) via SparkRun"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

sparkrun run qwen3.5-35b-a3b-fp8

echo "-> Waiting for inference endpoint on port 8000..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "  Model ready"
    break
  fi
  sleep 5
done

echo ""
echo "NOTE: No litellm_register group is defined yet for qwen3.5-35b."
echo "      To register manually, use the LiteLLM admin UI at http://localhost:4000/ui"
echo "      or add a model entry to core_services/litellm_local.yaml."
echo ""
sparkrun status
