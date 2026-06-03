#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:1616}"
MODEL="${OPENAI_MODEL:-ds}"
API_KEY="${OPENAI_API_KEY:-EMPTY}"
DATA_FILE="$SCRIPT_DIR/baseline/default_cases.jsonl"
EXTRA_BODY="${EXTRA_BODY:-}"
ARGS=()

usage() {
  cat <<'EOF'
Usage: test/agent/run.sh --base-url URL --model MODEL [options forwarded to agent_tool_eval.py]

Common options:
  --base-url URL       OpenAI-compatible API base URL, without /v1.
  --model MODEL        Model name sent in the request payload.
  --api-key KEY        API key. Defaults to OPENAI_API_KEY or EMPTY.
  --data-file PATH     Case file. Defaults to baseline/default_cases.jsonl.
  --extra-body JSON    Extra model/server parameters merged into the request body.

Examples:
  test/agent/run.sh --base-url http://127.0.0.1:1616 --model ds --limit 8
  test/agent/run.sh --base-url http://127.0.0.1:1616 --model ds \
    --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'

Additional arguments such as --workers, --max-steps, --temperature,
--max-tokens, --output-file, --resume, and --overwrite are forwarded.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --api-key)
      API_KEY="$2"
      shift 2
      ;;
    --data-file)
      DATA_FILE="$2"
      shift 2
      ;;
    --extra-body)
      EXTRA_BODY="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      ARGS+=("$@")
      break
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

cd "$REPO_ROOT"
cmd=(
  "$PYTHON_BIN" "$SCRIPT_DIR/agent_tool_eval.py"
  --base-url "$BASE_URL"
  --model "$MODEL"
  --api-key "$API_KEY"
  --data-file "$DATA_FILE"
)
if [[ -n "$EXTRA_BODY" ]]; then
  cmd+=(--extra-body "$EXTRA_BODY")
fi
cmd+=("${ARGS[@]}")

exec "${cmd[@]}"
