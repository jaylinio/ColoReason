#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

test -x "$ROLLOUT_PYTHON"
test -x "$ROLLOUT_SWIFT_CLI"
test -x "$TRAIN_PYTHON"
test -x "$TRAIN_SWIFT_CLI"
test -d "$VENDOR_ROOT/swift"

echo "[rollout environment]"
"$ROLLOUT_PYTHON" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
for package in ("torch", "vllm", "ms-swift", "transformers"):
    try:
        print(f"{package}={version(package)}")
    except PackageNotFoundError:
        print(f"{package}=NOT_INSTALLED")
PY

echo "[trainer environment]"
"$TRAIN_PYTHON" - <<'PY'
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import os

for package in ("torch", "flash-attn", "vllm", "ms-swift", "transformers", "trl", "deepspeed", "fuzzywuzzy", "jsonschema"):
    try:
        print(f"{package}={version(package)}")
    except PackageNotFoundError:
        print(f"{package}=NOT_INSTALLED")
import torch, deepspeed
from flash_attn import flash_attn_func
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from swift.rlhf_trainers.vllm_client import VLLMClient
print("flash_attn_import=ok")
print("external_vllm_client_import=ok")
print(f"deepspeed_import={deepspeed.__version__}")
print(f"embedding_model_exists={Path(os.environ['COLOREASON_EMBEDDING_MODEL']).is_dir()}")
PY

if ! "$TRAIN_PYTHON" -c 'import vllm, msgspec' >/dev/null 2>&1; then
  echo "trainer_external_vllm_client=NOT_READY" >&2
  echo "ms-swift external GRPO still imports vLLM PyNccl client code in the trainer environment." >&2
  exit 78
fi
echo "trainer_external_vllm_client=ok"
