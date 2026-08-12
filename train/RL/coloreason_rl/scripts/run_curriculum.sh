#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

: "${BASE_MODEL:?set BASE_MODEL}"
: "${SFT_REFERENCE_ADAPTER:?set SFT_REFERENCE_ADAPTER}"

policy_adapter=$SFT_REFERENCE_ADAPTER
for stage in 1 2 3 4; do
  POLICY_ADAPTER=$policy_adapter "$PROJECT_ROOT/scripts/run_stage.sh" "$stage"
  stage_name_var=STAGE_${stage}_NAME
  stage_name=${!stage_name_var}
  stage_output=${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}/$stage_name
  policy_adapter=$(find "$stage_output" -maxdepth 1 -type d -name 'checkpoint-*' -print \
    | sort -V | tail -n 1)
  if [[ -z "$policy_adapter" ]]; then
    echo "No checkpoint found after $stage_name in $stage_output" >&2
    exit 1
  fi
done
