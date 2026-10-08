#!/usr/bin/env bash
# tests/e2e/test_smoke_qwen3.5-9b.sh
#
# End-to-end regression for the 2026-05-08 production bug. MUST be run on a host
# with at least 4 NVIDIA GPUs of >=16GiB each. Does not run in CI.
#
# Usage:
#   HF_TOKEN=hf_xxx VW_BASE=http://localhost:8080 ./tests/e2e/test_smoke_qwen3.5-9b.sh
#
# Exit codes: 0 = pass, non-zero = fail.

set -euo pipefail
trap 'echo "FAILED at line $LINENO" >&2' ERR

: "${HF_TOKEN:?HF_TOKEN must be set}"
: "${VW_BASE:?VW_BASE must be set, e.g. http://localhost:8080}"

ADMIN_USER="admin"
ADMIN_PASS="e2e-password-$$"
COOKIES=$(mktemp)
trap 'rm -f $COOKIES' EXIT

curl_json() {
  curl -s -b "$COOKIES" -c "$COOKIES" -H "Content-Type: application/json" "$@"
}

echo "==> 1. Run setup wizard"
# welcome → gpus
curl_json -X POST "$VW_BASE/api/setup/welcome" -d '{}' >/dev/null
# pick GPUs 0-3
curl_json -X POST "$VW_BASE/api/setup/gpus" \
  -d '{"allowed_gpu_indices":[0,1,2,3]}' >/dev/null
# hf_token
curl_json -X POST "$VW_BASE/api/setup/hf_token" \
  -d "{\"hf_token\":\"$HF_TOKEN\"}" >/dev/null
# admin (creates user + flips step=done)
curl_json -X POST "$VW_BASE/api/setup/admin" \
  -d "{\"username\":\"$ADMIN_USER\",\"password\":\"$ADMIN_PASS\"}" >/dev/null

echo "==> 2. Log in as admin"
curl -s -c "$COOKIES" -b "$COOKIES" \
  -d "username=$ADMIN_USER&password=$ADMIN_PASS" \
  "$VW_BASE/login" >/dev/null

echo "==> 3. Register Qwen3.5-9B with TP=2 on GPUs [1,2]"
MODEL_ID=$(curl_json -X POST "$VW_BASE/api/models" -d '{
  "served_name": "qwen3.5-9b",
  "hf_repo": "Qwen/Qwen2.5-9B",
  "hf_revision": "main",
  "tensor_parallel_size": 2,
  "gpu_indices": [1, 2]
}' | jq -r .id)
echo "    model id: $MODEL_ID"

echo "==> 4. Pull"
curl_json -X POST "$VW_BASE/api/models/$MODEL_ID/pull" >/dev/null
for i in $(seq 1 60); do
  STATUS=$(curl_json "$VW_BASE/api/models/$MODEL_ID" | jq -r .status)
  [[ "$STATUS" == "pulled" ]] && break
  [[ "$STATUS" == "failed" ]] && { echo "pull failed"; exit 1; }
  sleep 30
done

echo "==> 5. Load (this is the bug-fix moment)"
curl_json -X POST "$VW_BASE/api/models/$MODEL_ID/load" >/dev/null
for i in $(seq 1 60); do
  STATUS=$(curl_json "$VW_BASE/api/models/$MODEL_ID" | jq -r .status)
  [[ "$STATUS" == "loaded" ]] && break
  [[ "$STATUS" == "failed" ]] && { echo "load failed — check $VW_BASE/api/models/$MODEL_ID/logs/stream"; exit 1; }
  sleep 5
done

echo "==> 6. Assert subprocess env CUDA_VISIBLE_DEVICES=1,2"
PID=$(curl_json "$VW_BASE/api/models/$MODEL_ID" | jq -r .runtime.pid)
ENV_VAL=$(tr '\0' '\n' < /proc/"$PID"/environ | awk -F= '/^CUDA_VISIBLE_DEVICES=/ {print $2}')
echo "    /proc/$PID/environ CUDA_VISIBLE_DEVICES=$ENV_VAL"
[[ "$ENV_VAL" == "1,2" ]] || { echo "BUG REGRESSED: expected '1,2', got '$ENV_VAL'"; exit 1; }

echo "==> 7. Assert exactly GPUs 1 and 2 are busy"
BUSY=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
       awk -F',' '$2+0 > 1000 {gsub(/ /,"",$1); print $1}' | sort | tr '\n' ',')
echo "    busy GPUs: $BUSY"
[[ "$BUSY" == "1,2," ]] || { echo "expected GPUs 1,2 busy, got: $BUSY"; exit 1; }

echo "==> 8. Inference"
TOKEN=$(curl_json -X POST "$VW_BASE/api/tokens" \
  -d '{"name":"e2e"}' | jq -r .token)
RESP=$(curl -s -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -X POST "$VW_BASE/v1/chat/completions" -d '{
    "model": "qwen3.5-9b",
    "messages": [{"role": "user", "content": "Reply with one word."}],
    "max_tokens": 8
  }')
CONTENT=$(echo "$RESP" | jq -r .choices[0].message.content)
echo "    response: $CONTENT"
[[ -n "$CONTENT" && "$CONTENT" != "null" ]] || { echo "empty response: $RESP"; exit 1; }

echo "==> 9. Unload + assert GPUs released"
curl_json -X POST "$VW_BASE/api/models/$MODEL_ID/unload" >/dev/null
for i in $(seq 1 30); do
  STATUS=$(curl_json "$VW_BASE/api/models/$MODEL_ID" | jq -r .status)
  [[ "$STATUS" == "registered" || "$STATUS" == "pulled" ]] && break
  sleep 1
done
sleep 5
BUSY=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
       awk -F',' '$2+0 > 1000 {print $1}' | tr '\n' ',')
[[ -z "$BUSY" ]] || { echo "GPUs still busy after unload: $BUSY"; exit 1; }

echo "==> ALL CHECKS PASSED"
