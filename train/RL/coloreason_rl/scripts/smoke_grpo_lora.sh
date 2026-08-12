#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

"$PROJECT_ROOT/scripts/require_trainer_external_client.sh"
TRAIN_GPU_INDEX=${TRAIN_GPU_INDEX:-0}
"$PROJECT_ROOT/scripts/require_idle_gpu.sh" "$TRAIN_GPU_INDEX"
"$PROJECT_ROOT/scripts/wait_rollout_server.sh"

model=${SMOKE_MODEL:-$PROJECT_ROOT/models/Qwen3-1.7B}
dataset=${SMOKE_DATASET:-$PROJECT_ROOT/data/smoke.jsonl}
output=${SMOKE_OUTPUT:-$PROJECT_ROOT/outputs/smoke-qwen3-1.7b-lora}
test -d "$model"
test -d "$COLOREASON_EMBEDDING_MODEL"
test -f "$dataset"

CUDA_VISIBLE_DEVICES=$TRAIN_GPU_INDEX "$TRAIN_SWIFT_CLI" rlhf \
  --rlhf_type grpo \
  --model "$model" \
  --tuner_type lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --target_modules all-linear \
  --dataset "$dataset" \
  --external_plugins "$PROJECT_ROOT/plugin.py" \
  --reward_funcs coloreason_composite \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --max_steps 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --generation_batch_size 2 \
  --num_generations 2 \
  --num_iterations 1 \
  --learning_rate 1e-6 \
  --epsilon 0.2 \
  --beta 0.02 \
  --max_length 1024 \
  --max_completion_length 256 \
  --temperature 0.6 \
  --top_p 0.95 \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host "$VLLM_SERVER_HOST" \
  --vllm_server_port "$VLLM_SERVER_PORT" \
  --vllm_server_group_port "$VLLM_SERVER_GROUP_PORT" \
  --vllm_server_timeout "$VLLM_SERVER_TIMEOUT" \
  --overlong_filter false \
  --gradient_checkpointing true \
  --logging_steps 1 \
  --save_steps 1 \
  --save_total_limit 1 \
  --log_completions true \
  --report_to none \
  --output_dir "$output"
