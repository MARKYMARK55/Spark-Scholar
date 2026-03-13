# scripts/models/ — Model Launch Scripts

Each script starts a SparkRun recipe, waits for the inference endpoint on port 8000,
then registers LiteLLM aliases so Open WebUI and the RAG proxy pick up the new model
without a restart.

## Usage

```bash
# Launch a model (run from repo root or any path — scripts use absolute paths)
bash scripts/models/nemotron-3-nano.sh
bash scripts/models/qwen3-coder-next.sh

# Check what's running
sparkrun status

# Browse all available recipes
sparkrun list

# Search by keyword
sparkrun search <term>

# Show recipe details and VRAM estimate
sparkrun show <recipe-slug>

# Stop a model
sparkrun stop <recipe-slug>
```

## Available Recipes

### @eugr registry — github.com/eugr/spark-vllm-docker (single DGX Spark)

| Recipe slug | Model | VRAM | Best for | Launch script |
|---|---|---|---|---|
| `nemotron-3-nano-nvfp4` | NVIDIA Nemotron-3-Nano-30B-A3B NVFP4 | ~40 GB | General research, reasoning, fast responses | `nemotron-3-nano.sh` |
| `nemotron-3-super-nvfp4` | NVIDIA Nemotron-3-Super NVFP4 | ~70 GB | Higher-capacity reasoning | `nemotron-3-super.sh` |
| `openai-gpt-oss-120b` | GPT-OSS-120B MXFP4 | ~80 GB (tp:2) | Large model — **requires 2 nodes** | `openai-gpt-oss-120b.sh` |
| `qwen3-coder-next-fp8` | Qwen3-Coder-Next FP8 | ~40 GB | Code generation, debugging, refactoring | `qwen3-coder-next.sh` |
| `qwen3-coder-next-int4-autoround` | Qwen3-Coder-Next INT4 AutoRound | ~25 GB | Code, lower VRAM | `qwen3-coder-next-int4.sh` |
| `qwen3-instruct-80b` | Qwen3-Instruct-80B FP8 | ~80 GB | Deep research, long context | `qwen3-instruct-80b.sh` |
| `qwen3.5-35b-a3b-fp8` | Qwen3.5-35B-A3B FP8 (MoE) | ~35 GB | Efficient MoE, general tasks | `qwen3.5-35b.sh` |
| `qwen3.5-122b-fp8` | Qwen3.5-122B FP8 | ~122 GB | Very large, needs most of 128 GB | — |
| `qwen3.5-122b-int4-autoround` | Qwen3.5-122B INT4 AutoRound | ~65 GB | Large model, lower VRAM | — |
| `glm-4.7-flash-awq` | GLM-4.7-Flash AWQ | ~8 GB | Ultra-light, tool routing, fast responses | `glm-4.7-flash.sh` |
| `minimax-m2-awq` | MiniMax-M2 AWQ | ~40 GB | General tasks | — |
| `minimax-m2.5-awq` | MiniMax-M2.5 AWQ | ~40 GB | General tasks, MoE efficiency | `minimax-m2.5.sh` |

### @eugr 4x-spark-cluster — multi-node only

| Recipe slug | Model | VRAM | Notes | Launch script |
|---|---|---|---|---|
| `qwen3.5-397b-a17B-fp8` | Qwen3.5-397B FP8 | 4-node cluster | **Requires 4 DGX Spark nodes** | — |
| `qwen3.5-397b-int4-autoround` | Qwen3.5-397B INT4 | 4-node cluster | **Requires 4 DGX Spark nodes** | — |
| `minimax-m2.5` | MiniMax-M2.5 | 4-node cluster | **Requires 4 DGX Spark nodes** | — |

### @sparkrun-transitional registry — github.com/dbotwinick/sparkrun-recipe-registry (community)

| Recipe slug | Model | VRAM | Best for | Launch script |
|---|---|---|---|---|
| `qwen3-1.7b-vllm` | Qwen3-1.7B vLLM | minimal | Dev/testing, tiny footprint | — |
| `qwen3-1.7b-sglang` | Qwen3-1.7B SGLang | minimal | Dev/testing | — |
| `qwen3-1.7b-llama-cpp` | Qwen3-1.7B llama.cpp | minimal | Dev/testing | — |
| `qwen3.5-0.8b-bf16-sglang` | Qwen3.5-0.8B BF16 SGLang | minimal | Tiny, edge testing | — |
| `qwen3.5-2b-bf16-sglang` | Qwen3.5-2B BF16 SGLang | ~4 GB | Fast, low resource | — |
| `qwen3.5-4b-bf16-sglang` | Qwen3.5-4B BF16 SGLang | ~8 GB | Lightweight general | — |
| `qwen3.5-9b-bf16-sglang` | Qwen3.5-9B BF16 SGLang | ~18 GB | Mid-size general | — |
| `qwen3.5-27b-fp8-sglang` | Qwen3.5-27B FP8 SGLang | ~27 GB | Mid-size, FP8 efficiency | — |
| `qwen3.5-35b-bf16-sglang` | Qwen3.5-35B BF16 SGLang | ~70 GB | Large general | — |
| `qwen3.5-35b-a3b-fp8-sglang` | Qwen3.5-35B-A3B FP8 SGLang | ~35 GB | MoE, FP8 | — |
| `qwen3.5-122b-a10b-fp8-sglang` | Qwen3.5-122B-A10B FP8 SGLang | ~65 GB | Very large MoE | — |
| `qwen3.5-122b-gguf-q4km-llama-cpp` | Qwen3.5-122B GGUF Q4_K_M llama.cpp | ~70 GB | Large, GGUF quantised | — |
| `qwen3.5-397b-gguf-q3km-llama-cpp` | Qwen3.5-397B GGUF Q3_K_M llama.cpp | large | Multi-node GGUF | — |
| `qwen3.5-397b-gguf-q6k-llama-cpp` | Qwen3.5-397B GGUF Q6_K llama.cpp | large | Multi-node GGUF, higher quality | — |
| `qwen3-coder-next-fp8-sglang` | Qwen3-Coder-Next FP8 SGLang | ~40 GB | Coding, SGLang backend | — |
| `qwen3-coder-next-int4-autoround-vllm` | Qwen3-Coder-Next INT4 vLLM | ~25 GB | Coding, INT4, vLLM backend | — |

## Notes

- `sparkrun list` shows all available recipes from all registered registries
- `sparkrun search <term>` filters recipes by keyword (e.g. `sparkrun search qwen`)
- `sparkrun show <recipe-slug>` shows full recipe details including VRAM estimate and options
- `sparkrun registry list` shows all configured registries
- The `--port` / `-o port=<N>` flag can override the default port in any recipe
- Multi-node recipes require the cluster to be pre-configured — see SparkRun docs

## Phi Mini

There is no confirmed SparkRun recipe for Phi-4-mini. To check for one:

```bash
sparkrun search phi
```

If a recipe exists, run it with a port override:

```bash
sparkrun run <phi-recipe-slug> -o port=8001
```
