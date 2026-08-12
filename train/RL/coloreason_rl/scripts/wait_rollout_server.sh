#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

deadline=$((SECONDS + VLLM_SERVER_TIMEOUT))
url="http://${VLLM_SERVER_HOST}:${VLLM_SERVER_PORT}/health/"
while (( SECONDS < deadline )); do
  if curl --silent --fail --max-time 2 "$url" >/dev/null; then
    echo "rollout_server_ready=$url"
    exit 0
  fi
  sleep 2
done
echo "Rollout server did not become ready within ${VLLM_SERVER_TIMEOUT}s: $url" >&2
exit 75
