#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?usage: require_idle_gpu.sh GPU_INDEX}
max_util=${MAX_GPU_UTILIZATION:-10}
min_free_mib=${MIN_FREE_GPU_MIB:-20000}

read -r util free_mib < <(
  nvidia-smi --query-gpu=utilization.gpu,memory.free --format=csv,noheader,nounits -i "$gpu" \
    | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1, $2}'
)
process_count=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader -i "$gpu" 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l)

if (( process_count > 0 || util > max_util || free_mib < min_free_mib )); then
  echo "GPU $gpu is busy: processes=$process_count utilization=${util}% free=${free_mib}MiB" >&2
  exit 75
fi
echo "GPU $gpu is idle enough: utilization=${util}% free=${free_mib}MiB"
