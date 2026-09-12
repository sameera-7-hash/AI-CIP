#!/usr/bin/env bash
# Quick end-to-end smoke test matching Section 5.1 (Phase 0) exit criteria:
# "curl to /invoke returns a grounded response with latency logged."
#
# Usage: ./scripts/smoke_test.sh [gateway_base_url]
set -euo pipefail

BASE_URL="${1:-http://localhost:8080}"

echo "== 1) vLLM model list (direct) =="
curl -sf "http://localhost:8000/v1/models" | python3 -m json.tool || {
  echo "vLLM does not appear to be reachable on :8000. Is it running?"
  exit 1
}

echo
echo "== 2) Gateway health =="
curl -sf "${BASE_URL}/health" | python3 -m json.tool

echo
echo "== 3) Gateway agent invoke =="
curl -sf -X POST "${BASE_URL}/v1/agent/invoke" \
  -H "Content-Type: application/json" \
  -d '{"goal": "Summarize current GPU utilization and recommend an action."}' \
  | python3 -m json.tool

echo
echo "Smoke test complete."
