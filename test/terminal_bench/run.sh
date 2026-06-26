#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET="${TB_DATASET:-terminal-bench@2.0}"
AGENT="${TB_AGENT:-terminus-2}"
MODEL="${OPENAI_MODEL:-${TB_MODEL:-openai/ds}}"
API_BASE="${OPENAI_BASE_URL:-${OPENAI_API_BASE:-${TB_API_BASE:-}}}"
API_KEY="${OPENAI_API_KEY:-EMPTY}"
ENV_PROVIDER="${TB_ENV:-docker}"
N_CONCURRENT="${TB_N_CONCURRENT:-1}"
N_ATTEMPTS="${TB_N_ATTEMPTS:-1}"
JOBS_DIR="${TB_JOBS_DIR:-$SCRIPT_DIR/jobs}"
TEMPERATURE="${TB_TEMPERATURE:-1.0}"
LLM_CALL_KWARGS="${TB_LLM_CALL_KWARGS:-}"
TIMEOUT_MULTIPLIER="${TB_TIMEOUT_MULTIPLIER:-}"
ARGS=()

usage() {
  cat <<'EOF'
Usage: test/terminal_bench/run.sh [options] [extra Harbor args]

Common options:
  --model MODEL             Harbor/LiteLLM model name. Default: OPENAI_MODEL or openai/ds.
  --api-base URL            OpenAI-compatible API base URL, usually ending with /v1.
  --api-key KEY             API key. Defaults to OPENAI_API_KEY or EMPTY.
  --dataset DATASET         Harbor dataset. Default: terminal-bench@2.0.
  --agent AGENT             Harbor agent. Default: terminus-2.
  --env ENV                 Harbor environment provider. Default: docker.
  -n, --n-concurrent N      Concurrent trials. Default: 1.
  --n-attempts N            Attempts per task. Qwen report uses 5-run average.
  --jobs-dir DIR            Harbor output directory. Default: test/terminal_bench/jobs.
  --temperature FLOAT       Terminus-2 temperature agent kwarg. Default: 1.0.
  --llm-call-kwargs JSON    Terminus-2 llm_call_kwargs agent kwarg.
  --timeout-multiplier N    Multiplier for Harbor task timeouts.

Any unknown arguments are forwarded to harbor run, e.g.:
  test/terminal_bench/run.sh --model openai/ds --api-base http://127.0.0.1:1616/v1 --dataset terminal-bench-sample@2.0
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --api-base|--base-url)
      API_BASE="$2"
      shift 2
      ;;
    --api-key)
      API_KEY="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --agent)
      AGENT="$2"
      shift 2
      ;;
    --env)
      ENV_PROVIDER="$2"
      shift 2
      ;;
    -n|--n-concurrent)
      N_CONCURRENT="$2"
      shift 2
      ;;
    --n-attempts)
      N_ATTEMPTS="$2"
      shift 2
      ;;
    --jobs-dir)
      JOBS_DIR="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --llm-call-kwargs)
      LLM_CALL_KWARGS="$2"
      shift 2
      ;;
    --timeout-multiplier)
      TIMEOUT_MULTIPLIER="$2"
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

if ! command -v harbor >/dev/null 2>&1; then
  echo "harbor not found. Run: bash test/terminal_bench/setup.sh" >&2
  exit 127
fi

case "$JOBS_DIR" in
  /*) ;;
  *) JOBS_DIR="$REPO_ROOT/$JOBS_DIR" ;;
esac

mkdir -p "$JOBS_DIR"

export OPENAI_API_KEY="$API_KEY"
if [[ -n "$API_BASE" ]]; then
  export OPENAI_BASE_URL="$API_BASE"
  export OPENAI_API_BASE="$API_BASE"
fi

cd "$REPO_ROOT"

cmd=(
  harbor run
  -d "$DATASET"
  -a "$AGENT"
  -m "$MODEL"
  -n "$N_CONCURRENT"
  --n-attempts "$N_ATTEMPTS"
  --jobs-dir "$JOBS_DIR"
)

if [[ -n "$ENV_PROVIDER" ]]; then
  cmd+=(--env "$ENV_PROVIDER")
fi

if [[ -n "$API_BASE" ]]; then
  cmd+=(--ak "api_base=$API_BASE")
fi

if [[ -n "$TEMPERATURE" ]]; then
  cmd+=(--ak "temperature=$TEMPERATURE")
fi

if [[ -n "$LLM_CALL_KWARGS" ]]; then
  cmd+=(--ak "llm_call_kwargs=$LLM_CALL_KWARGS")
fi

if [[ -n "$TIMEOUT_MULTIPLIER" ]]; then
  cmd+=(--timeout-multiplier "$TIMEOUT_MULTIPLIER")
fi

cmd+=("${ARGS[@]}")

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
