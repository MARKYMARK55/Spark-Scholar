#!/usr/bin/env bash
# Launch MiniMax-M2.5 AWQ
# Recipe: minimax-m2.5-awq (@eugr registry)
# VRAM: ~40 GB  |  Best for: general tasks, MoE efficiency
# LiteLLM aliases: not yet available via litellm_register — see note below
set -euo pipefail

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Launching MiniMax-M2.5 AWQ via SparkRun"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

sparkrun run minimax-m2.5-awq

echo "-> Waiting for inference endpoint on port 8000..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "  Model ready"
    break
  fi
  sleep 5
done

echo ""
echo "NOTE: No litellm_register group is defined yet for minimax-m2.5."
echo "      To register manually, use the LiteLLM admin UI at http://localhost:4000/ui"
echo "      or add a model entry to core_services/litellm_local.yaml."
echo ""
sparkrun status
