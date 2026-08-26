#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:1616}"
MODEL="${OPENAI_MODEL:-ds}"
API_KEY="${OPENAI_API_KEY:-EMPTY}"
DATA_FILE="$SCRIPT_DIR/baseline/smoke.jsonl"
OUTPUT_DIR="$SCRIPT_DIR/results"
EXTRA_BODY="${EXTRA_BODY:-}"
ARGS=()

usage() {
  cat <<'EOF'
Usage: test/hle/run.sh --base-url URL --model MODEL [options]

Common options:
  --base-url URL       OpenAI-compatible API base URL, without /v1.
  --model MODEL        Model name sent in the request payload.
  --api-key KEY        API key. Defaults to OPENAI_API_KEY or EMPTY.
  --data-file PATH     HLE-shaped JSONL/JSON/CSV. Defaults to synthetic smoke data.
  --extra-body JSON    Extra fields merged into every request body.

For the paired 50-question run, pass --workers 1 --temperature 0,
--max-tokens 0, --request-timeout 0, and --warmup-tokens 64. A zero
max-token value omits the request field and uses the server's default
completion limit. Other evaluator arguments are forwarded.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --api-key) API_KEY="$2"; shift 2 ;;
    --data-file) DATA_FILE="$2"; shift 2 ;;
    --extra-body) EXTRA_BODY="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; ARGS+=("$@"); break ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

cd "$REPO_ROOT"
cmd=(
  "$PYTHON_BIN" "$SCRIPT_DIR/hle_api_eval.py"
  --base-url "$BASE_URL"
  --model "$MODEL"
  --api-key "$API_KEY"
  --data-file "$DATA_FILE"
  --output-dir "$OUTPUT_DIR"
)
if [[ -n "$EXTRA_BODY" ]]; then
  cmd+=(--extra-body "$EXTRA_BODY")
fi
cmd+=("${ARGS[@]}")
exec "${cmd[@]}"
