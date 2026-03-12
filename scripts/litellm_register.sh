#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# litellm_register.sh  —  lives in ~/vllm/model_stack/core_services/
#
# Registers / deregisters local model presets with LiteLLM API.
# Called by each start_*.sh after vLLM passes health check.
#
# Usage:
#   source ~/vllm/model_stack/core_services/litellm_register.sh
#   litellm_register   <group>    # e.g. nemotron | gpt-oss | qwen3-80b | qwen3-coder | glm
#   litellm_deregister <group>
#   wait_for_vllm      <port> [max_seconds]
#
# max_tokens values below are OUTPUT token limits, not context window size.
# ─────────────────────────────────────────────────────────────────────────────

LITELLM_URL="http://localhost:4000"
LITELLM_KEY="simple-api-key"
ID_FILE="/tmp/litellm_model_ids.json"

# Initialise ID store
[[ ! -f "$ID_FILE" ]] && echo '{}' > "$ID_FILE"

# ── Wait for vLLM health ──────────────────────────────────────────────────────
wait_for_vllm() {
  local port="${1:-8000}"
  local max_wait="${2:-300}"
  local elapsed=0

  echo "⏳ Waiting for vLLM on port $port..."
  while [[ $elapsed -lt $max_wait ]]; do
    if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
      echo "✅ vLLM healthy on port $port"
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    echo "   ...${elapsed}s / ${max_wait}s"
  done
  echo "❌ vLLM did not become healthy after ${max_wait}s"
  return 1
}

# ── POST one model, store returned ID ────────────────────────────────────────
_register_model() {
  local name="$1"
  local payload="$2"

  echo "  → Registering $name ..."
  local response
  response=$(curl -s -X POST "${LITELLM_URL}/model/new" \
    -H "Authorization: Bearer ${LITELLM_KEY}" \
    -H "Content-Type: application/json" \
    -d "$payload")

  local model_id
  model_id=$(echo "$response" | python3 -c \
    "import sys,json; d=json.load(sys.stdin); print(d.get('model_id',''))" 2>/dev/null)

  if [[ -n "$model_id" ]]; then
    python3 -c "
import json
with open('$ID_FILE') as f: d=json.load(f)
d['$name']='$model_id'
with open('$ID_FILE','w') as f: json.dump(d,f)
"
    echo "    ✅ $name (id: $model_id)"
  else
    echo "    ⚠️  $name failed: $response"
  fi
}

# ── DELETE one model by stored ID ────────────────────────────────────────────
_deregister_model() {
  local name="$1"
  local model_id
  model_id=$(python3 -c "
import json
with open('$ID_FILE') as f: d=json.load(f)
print(d.get('$name',''))
" 2>/dev/null)

  [[ -z "$model_id" ]] && { echo "  ⚠️  No stored ID for $name — skipping"; return; }

  echo "  → Deregistering $name ..."
  curl -s -X POST "${LITELLM_URL}/model/delete" \
    -H "Authorization: Bearer ${LITELLM_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"id\": \"$model_id\"}" > /dev/null

  python3 -c "
import json
with open('$ID_FILE') as f: d=json.load(f)
d.pop('$name', None)
with open('$ID_FILE','w') as f: json.dump(d,f)
"
  echo "    ✅ $name deregistered"
}


# ═════════════════════════════════════════════════════════════════════════════
# REGISTER — one function per model group
# ═════════════════════════════════════════════════════════════════════════════

litellm_register() {
  local group="$1"
  echo ""
  echo "📋 Registering presets for: $group"

  case "$group" in

    # ── Nemotron-Nano  port 8000  output max ~32768 ───────────────────────
    nemotron)
      _register_model "Nemotron-Fast" '{
        "model_name": "Nemotron-Fast",
        "litellm_params": {
          "model": "openai/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
          "api_base": "http://host.docker.internal:8000/v1",
          "api_key": "dummy",
          "temperature": 0.55, "top_p": 0.98, "max_tokens": 32768}}'

      _register_model "Nemotron-Expert" '{
        "model_name": "Nemotron-Expert",
        "litellm_params": {
          "model": "openai/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
          "api_base": "http://host.docker.internal:8000/v1",
          "api_key": "dummy",
          "temperature": 0.55, "top_p": 0.98, "max_tokens": 32768, "timeout": 400,
          "extra_body": {"chat_template_kwargs": {"enable_thinking": true, "thinking_budget": 4096}}}}'

      _register_model "Nemotron-Heavy" '{
        "model_name": "Nemotron-Heavy",
        "litellm_params": {
          "model": "openai/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
          "api_base": "http://host.docker.internal:8000/v1",
          "api_key": "dummy",
          "temperature": 0.55, "top_p": 0.98, "max_tokens": 32768, "timeout": 600,
          "extra_body": {"chat_template_kwargs": {"enable_thinking": true, "thinking_budget": 16384}}}}'

      _register_model "Nemotron-Max" '{
        "model_name": "Nemotron-Max",
        "litellm_params": {
          "model": "openai/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
          "api_base": "http://host.docker.internal:8000/v1",
          "api_key": "dummy",
          "temperature": 0.55, "top_p": 0.98, "max_tokens": 32768, "timeout": 600,
          "extra_body": {"chat_template_kwargs": {"enable_thinking": true, "thinking_budget": 32768}}}}'

      _register_model "Nemotron-Code" '{
        "model_name": "Nemotron-Code",
        "litellm_params": {
          "model": "openai/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
          "api_base": "http://host.docker.internal:8000/v1",
          "api_key": "dummy",
          "temperature": 0.10, "top_p": 0.95, "max_tokens": 16384, "timeout": 300,
          "extra_body": {"chat_template_kwargs": {"enable_thinking": true, "thinking_budget": 4096}}}}'

      _register_model "Nemotron-Creative" '{
        "model_name": "Nemotron-Creative",
        "litellm_params": {
          "model": "openai/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
          "api_base": "http://host.docker.internal:8000/v1",
          "api_key": "dummy",
          "temperature": 0.85, "top_p": 0.98, "max_tokens": 16384, "timeout": 400,
          "extra_body": {"chat_template_kwargs": {"enable_thinking": true, "thinking_budget": 2048}}}}'
      ;;

    # ── GPT-OSS 120B  port 8000  output max 8192 (batched token limit) ───
    gpt-oss)
      _register_model "GPT-OSS-Fast" '{
        "model_name": "GPT-OSS-Fast",
        "litellm_params": {
          "model": "openai/gpt-oss-120b",
          "api_base": "http://host.docker.internal:8000/v1",
          "api_key": "simple-api-key",
          "temperature": 0.55, "top_p": 0.95, "max_tokens": 8192, "timeout": 400}}'

      _register_model "GPT-OSS-Expert" '{
        "model_name": "GPT-OSS-Expert",
        "litellm_params": {
          "model": "openai/gpt-oss-120b",
          "api_base": "http://host.docker.internal:8000/v1",
          "api_key": "simple-api-key",
          "temperature": 0.55, "top_p": 0.95, "max_tokens": 8192, "timeout": 400,
          "extra_body": {"chat_template_kwargs": {"enable_thinking": true, "effort": "medium"}}}}'

      _register_model "GPT-OSS-Heavy" '{
        "model_name": "GPT-OSS-Heavy",
        "litellm_params": {
          "model": "openai/gpt-oss-120b",
          "api_base": "http://host.docker.internal:8000/v1",
          "api_key": "simple-api-key",
          "temperature": 0.55, "top_p": 0.95, "max_tokens": 8192, "timeout": 600,
          "extra_body": {"chat_template_kwargs": {"enable_thinking": true, "effort": "high"}}}}'

      _register_model "GPT-OSS-Max" '{
        "model_name": "GPT-OSS-Max",
        "litellm_params": {
          "model": "openai/gpt-oss-120b",
          "api_base": "http://host.docker.internal:8000/v1",
          "api_key": "simple-api-key",
          "temperature": 0.55, "top_p": 0.95, "max_tokens": 8192, "timeout": 600,
          "extra_body": {"chat_template_kwargs": {"enable_thinking": true, "effort": "max"}}}}'

      _register_model "GPT-OSS-Code" '{
        "model_name": "GPT-OSS-Code",
        "litellm_params": {
          "model": "openai/gpt-oss-120b",
          "api_base": "http://host.docker.internal:8000/v1",
          "api_key": "simple-api-key",
          "temperature": 0.10, "top_p": 0.95, "max_tokens": 8192, "timeout": 400,
          "extra_body": {"chat_template_kwargs": {"enable_thinking": true, "effort": "medium"}}}}'

      _register_model "GPT-OSS-Creative" '{
        "model_name": "GPT-OSS-Creative",
        "litellm_params": {
          "model": "openai/gpt-oss-120b",
          "api_base": "http://host.docker.internal:8000/v1",
          "api_key": "simple-api-key",
          "temperature": 0.85, "top_p": 0.98, "max_tokens": 8192, "timeout": 400,
          "extra_body": {"chat_template_kwargs": {"enable_thinking": true, "effort": "low"}}}}'
      ;;

    # ── Qwen3-80B  port 8005  output max 32768 ───────────────────────────
    qwen3-80b)
      _register_model "Qwen3-80B-Fast" '{
        "model_name": "Qwen3-80B-Fast",
        "litellm_params": {
          "model": "openai/Superqwen",
          "api_base": "http://host.docker.internal:8005/v1",
          "api_key": "sk-no-key-required",
          "temperature": 0.55, "top_p": 0.95, "max_tokens": 32768, "timeout": 400}}'

      _register_model "Qwen3-80B-Expert" '{
        "model_name": "Qwen3-80B-Expert",
        "litellm_params": {
          "model": "openai/Superqwen",
          "api_base": "http://host.docker.internal:8005/v1",
          "api_key": "sk-no-key-required",
          "temperature": 0.55, "top_p": 0.95, "max_tokens": 32768, "timeout": 400,
          "extra_body": {"chat_template_kwargs": {"enable_thinking": true, "thinking_budget": 4096}}}}'

      _register_model "Qwen3-80B-Heavy" '{
        "model_name": "Qwen3-80B-Heavy",
        "litellm_params": {
          "model": "openai/Superqwen",
          "api_base": "http://host.docker.internal:8005/v1",
          "api_key": "sk-no-key-required",
          "temperature": 0.55, "top_p": 0.95, "max_tokens": 32768, "timeout": 600,
          "extra_body": {"chat_template_kwargs": {"enable_thinking": true, "thinking_budget": 16384}}}}'

      _register_model "Qwen3-80B-Max" '{
        "model_name": "Qwen3-80B-Max",
        "litellm_params": {
          "model": "openai/Superqwen",
          "api_base": "http://host.docker.internal:8005/v1",
          "api_key": "sk-no-key-required",
          "temperature": 0.55, "top_p": 0.95, "max_tokens": 32768, "timeout": 600,
          "extra_body": {"chat_template_kwargs": {"enable_thinking": true, "thinking_budget": 32768}}}}'

      _register_model "Qwen3-80B-Code" '{
        "model_name": "Qwen3-80B-Code",
        "litellm_params": {
          "model": "openai/Superqwen",
          "api_base": "http://host.docker.internal:8005/v1",
          "api_key": "sk-no-key-required",
          "temperature": 0.10, "top_p": 0.95, "max_tokens": 16384, "timeout": 400,
          "extra_body": {"chat_template_kwargs": {"enable_thinking": true, "thinking_budget": 4096}}}}'

      _register_model "Qwen3-80B-Creative" '{
        "model_name": "Qwen3-80B-Creative",
        "litellm_params": {
          "model": "openai/Superqwen",
          "api_base": "http://host.docker.internal:8005/v1",
          "api_key": "sk-no-key-required",
          "temperature": 0.85, "top_p": 0.98, "max_tokens": 16384, "timeout": 400,
          "extra_body": {"chat_template_kwargs": {"enable_thinking": true, "thinking_budget": 2048}}}}'
      ;;

    # ── Qwen3-Coder  port 8000  output max 32768 ─────────────────────────
    qwen3-coder)
      _register_model "Qwen3-Coder-Expert" '{
        "model_name": "Qwen3-Coder-Expert",
        "litellm_params": {
          "model": "openai/Qwen/Qwen3-Coder-Next-FP8",
          "api_base": "http://host.docker.internal:8000/v1",
          "api_key": "simple-api-key",
          "temperature": 0.20, "top_p": 0.95, "max_tokens": 32768, "timeout": 400}}'

      _register_model "Qwen3-Coder-Heavy" '{
        "model_name": "Qwen3-Coder-Heavy",
        "litellm_params": {
          "model": "openai/Qwen/Qwen3-Coder-Next-FP8",
          "api_base": "http://host.docker.internal:8000/v1",
          "api_key": "simple-api-key",
          "temperature": 0.20, "top_p": 0.95, "max_tokens": 32768, "timeout": 600,
          "extra_body": {"chat_template_kwargs": {"enable_thinking": true, "thinking_budget": 16384}}}}'

      _register_model "Qwen3-Coder-Max" '{
        "model_name": "Qwen3-Coder-Max",
        "litellm_params": {
          "model": "openai/Qwen/Qwen3-Coder-Next-FP8",
          "api_base": "http://host.docker.internal:8000/v1",
          "api_key": "simple-api-key",
          "temperature": 0.20, "top_p": 0.95, "max_tokens": 32768, "timeout": 600,
          "extra_body": {"chat_template_kwargs": {"enable_thinking": true, "thinking_budget": 32768}}}}'

      _register_model "Qwen3-Coder-Code" '{
        "model_name": "Qwen3-Coder-Code",
        "litellm_params": {
          "model": "openai/Qwen/Qwen3-Coder-Next-FP8",
          "api_base": "http://host.docker.internal:8000/v1",
          "api_key": "simple-api-key",
          "temperature": 0.10, "top_p": 0.95, "max_tokens": 16384, "timeout": 300,
          "extra_body": {"chat_template_kwargs": {"enable_thinking": true, "thinking_budget": 4096}}}}'
      ;;

    # ── GLM-4.7-Flash  port 8000  output max 16384 ───────────────────────
    glm)
      _register_model "GLM-Fast" '{
        "model_name": "GLM-Fast",
        "litellm_params": {
          "model": "openai/THUDM/glm-4-9b-chat",
          "api_base": "http://host.docker.internal:8000/v1",
          "api_key": "simple-api-key",
          "temperature": 0.55, "top_p": 0.95, "max_tokens": 16384, "timeout": 120}}'

      _register_model "GLM-Expert" '{
        "model_name": "GLM-Expert",
        "litellm_params": {
          "model": "openai/THUDM/glm-4-9b-chat",
          "api_base": "http://host.docker.internal:8000/v1",
          "api_key": "simple-api-key",
          "temperature": 0.55, "top_p": 0.95, "max_tokens": 16384, "timeout": 120}}'

      _register_model "GLM-Code" '{
        "model_name": "GLM-Code",
        "litellm_params": {
          "model": "openai/THUDM/glm-4-9b-chat",
          "api_base": "http://host.docker.internal:8000/v1",
          "api_key": "simple-api-key",
          "temperature": 0.10, "top_p": 0.95, "max_tokens": 16384, "timeout": 120}}'
      ;;

    *)
      echo "❌ Unknown group: $group"
      echo "   Valid: nemotron | gpt-oss | qwen3-80b | qwen3-coder | glm"
      return 1
      ;;
  esac

  echo "✅ Done: $group"
  echo ""
}


# ═════════════════════════════════════════════════════════════════════════════
# DEREGISTER — removes all presets for a model group
# ═════════════════════════════════════════════════════════════════════════════

litellm_deregister() {
  local group="$1"
  echo ""
  echo "🗑️  Deregistering: $group"

  case "$group" in
    nemotron)
      for n in Nemotron-Fast Nemotron-Expert Nemotron-Heavy Nemotron-Max Nemotron-Code Nemotron-Creative; do
        _deregister_model "$n"
      done ;;
    gpt-oss)
      for n in GPT-OSS-Fast GPT-OSS-Expert GPT-OSS-Heavy GPT-OSS-Max GPT-OSS-Code GPT-OSS-Creative; do
        _deregister_model "$n"
      done ;;
    qwen3-80b)
      for n in Qwen3-80B-Fast Qwen3-80B-Expert Qwen3-80B-Heavy Qwen3-80B-Max Qwen3-80B-Code Qwen3-80B-Creative; do
        _deregister_model "$n"
      done ;;
    qwen3-coder)
      for n in Qwen3-Coder-Expert Qwen3-Coder-Heavy Qwen3-Coder-Max Qwen3-Coder-Code; do
        _deregister_model "$n"
      done ;;
    glm)
      for n in GLM-Fast GLM-Expert GLM-Code; do
        _deregister_model "$n"
      done ;;
    *)
      echo "❌ Unknown group: $group"
      return 1 ;;
  esac

  echo "✅ Done: $group"
  echo ""
}
