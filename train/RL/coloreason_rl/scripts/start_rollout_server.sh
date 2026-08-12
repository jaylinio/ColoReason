#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

ROLLOUT_GPU_INDEX=${ROLLOUT_GPU_INDEX:-1}
"$PROJECT_ROOT/scripts/require_idle_gpu.sh" "$ROLLOUT_GPU_INDEX"

model=${ROLLOUT_MODEL:-${BASE_MODEL:-$PROJECT_ROOT/models/Qwen3-1.7B}}
test -d "$model"

CUDA_VISIBLE_DEVICES=$ROLLOUT_GPU_INDEX \
"$ROLLOUT_SWIFT_CLI" rollout \
  --model "$model" \
  --torch_dtype bfloat16 \
  --host "${VLLM_BIND_HOST:-0.0.0.0}" \
  --port "$VLLM_SERVER_PORT" \
  --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE:-1}" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.80}" \
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-32768}" \
  --vllm_enable_lora true \
  --vllm_max_lora_rank "${LORA_RANK:-8}"
