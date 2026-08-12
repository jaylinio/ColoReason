#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

if ! "$TRAIN_PYTHON" -c '
import torch, deepspeed
from flash_attn import flash_attn_func
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from swift.rlhf_trainers.vllm_client import VLLMClient
' >/dev/null 2>&1; then
  echo "Trainer external-GRPO dependencies are not ready in $TRAIN_ENV." >&2
  echo "This ms-swift version requires flash-attn plus vLLM's PyNccl client modules in the trainer environment." >&2
  exit 78
fi
