#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

MODELSCOPE=${MODELSCOPE:-$ROLLOUT_ENV/bin/modelscope}
"$MODELSCOPE" download Qwen/Qwen3-1.7B \
  --local-dir "$PROJECT_ROOT/models/Qwen3-1.7B" --max-workers 4
"$MODELSCOPE" download Qwen/Qwen3-Embedding-0.6B \
  --local-dir "$PROJECT_ROOT/models/Qwen3-Embedding-0.6B" --max-workers 4
date -Iseconds > "$PROJECT_ROOT/models/.download-status"
