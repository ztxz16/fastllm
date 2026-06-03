#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

DATASET_NAME="openai/gsm8k"
DATASET_CONFIG="main"
SPLIT="test"
REVISION=""
OUTPUT=""
LIMIT=""

usage() {
  cat <<'EOF'
Usage: test/gsm8k/download.sh [options]

Options:
  --dataset-name NAME       HuggingFace dataset name. Default: openai/gsm8k.
  --dataset-config NAME     Dataset config. Default: main.
  --dataset-revision REV    Optional dataset revision or commit.
  --split SPLIT             Dataset split. Default: test.
  --output PATH             Output JSONL path. Default: baseline/downloaded/gsm8k_<split>.jsonl.
  --limit N                 Keep only first N rows.

The script stores downloaded data under baseline/downloaded by default. That
directory is gitignored to avoid committing full benchmark data accidentally.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-name)
      DATASET_NAME="$2"
      shift 2
      ;;
    --dataset-config)
      DATASET_CONFIG="$2"
      shift 2
      ;;
    --dataset-revision)
      REVISION="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --output)
      OUTPUT="$2"
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$OUTPUT" ]]; then
  OUTPUT="$SCRIPT_DIR/baseline/downloaded/gsm8k_${SPLIT}.jsonl"
fi
mkdir -p "$(dirname "$OUTPUT")"

"$PYTHON_BIN" - "$DATASET_NAME" "$DATASET_CONFIG" "$SPLIT" "$REVISION" "$OUTPUT" "$LIMIT" <<'PY'
import json
import sys

try:
    from datasets import load_dataset
except ModuleNotFoundError as exc:
    raise SystemExit("Missing dependency: datasets. Run test/gsm8k/setup.sh first.") from exc

dataset_name, dataset_config, split, revision, output, limit = sys.argv[1:7]
load_kwargs = {"split": split}
if revision:
    load_kwargs["revision"] = revision

rows = [dict(row) for row in load_dataset(dataset_name, dataset_config, **load_kwargs)]
if limit:
    rows = rows[: int(limit)]

with open(output, "w", encoding="utf-8") as fout:
    for row in rows:
        fout.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

print(f"Wrote {len(rows)} rows to {output}")
PY
