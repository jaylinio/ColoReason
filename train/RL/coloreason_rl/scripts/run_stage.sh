#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

"$PROJECT_ROOT/scripts/require_trainer_external_client.sh"
"$PROJECT_ROOT/scripts/wait_rollout_server.sh"

stage=${1:?usage: run_stage.sh STAGE_NUMBER}
case "$stage" in
  1) stage_name=$STAGE_1_NAME; dataset=$STAGE_1_DATASET ;;
  2) stage_name=$STAGE_2_NAME; dataset=$STAGE_2_DATASET ;;
  3) stage_name=$STAGE_3_NAME; dataset=$STAGE_3_DATASET ;;
  4) stage_name=$STAGE_4_NAME; dataset=$STAGE_4_DATASET ;;
  *) echo "stage must be 1, 2, 3, or 4" >&2; exit 64 ;;
esac

: "${BASE_MODEL:?set BASE_MODEL to Qwen3-8B or a compatible backbone}"
: "${POLICY_ADAPTER:?set POLICY_ADAPTER to the incoming SFT/previous-stage adapter}"
: "${SFT_REFERENCE_ADAPTER:?set SFT_REFERENCE_ADAPTER to the frozen SFT adapter}"

dataset=$PROJECT_ROOT/${dataset#./}
output=${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}/$stage_name
test -f "$dataset"

trainer_devices=${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3}
device_count=$(awk -F, '{print NF}' <<<"$trainer_devices")
world_size=${NPROC_PER_NODE:-$device_count}
if (( world_size != device_count )); then
  echo "NPROC_PER_NODE=$world_size does not match TRAIN_CUDA_VISIBLE_DEVICES=$trainer_devices" >&2
  exit 64
fi
if (( OPTIMIZATION_BATCH_SIZE % world_size != 0 )); then
  echo "OPTIMIZATION_BATCH_SIZE must be divisible by NPROC_PER_NODE" >&2
  exit 64
fi
per_device_batch=$((OPTIMIZATION_BATCH_SIZE / world_size))

CUDA_VISIBLE_DEVICES=$trainer_devices \
NPROC_PER_NODE=$world_size \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
"$TRAIN_SWIFT_CLI" rlhf \
  --rlhf_type grpo \
  --model "$BASE_MODEL" \
  --tuner_type lora \
  --adapters "$POLICY_ADAPTER" \
  --ref_adapters "$SFT_REFERENCE_ADAPTER" \
  --dataset "$dataset" \
  --external_plugins "$PROJECT_ROOT/plugin.py" \
  --reward_funcs coloreason_composite \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --per_device_train_batch_size "$per_device_batch" \
  --gradient_accumulation_steps 1 \
  --generation_batch_size "$GENERATION_BATCH_SIZE" \
  --num_generations "$NUM_GENERATIONS" \
  --num_iterations "$NUM_ITERATIONS" \
  --learning_rate "$LEARNING_RATE" \
  --epsilon "$EPSILON" \
  --beta "$KL_BETA" \
  --max_completion_length "$MAX_COMPLETION_LENGTH" \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host "$VLLM_SERVER_HOST" \
  --vllm_server_port "$VLLM_SERVER_PORT" \
  --vllm_server_group_port "$VLLM_SERVER_GROUP_PORT" \
  --vllm_server_timeout "$VLLM_SERVER_TIMEOUT" \
  --overlong_filter false \
  --gradient_checkpointing true \
  --deepspeed zero2 \
  --log_completions true \
  --logging_steps 1 \
  --save_steps "${SAVE_STEPS:-50}" \
  --output_dir "$output"
