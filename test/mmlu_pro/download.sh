#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

DATASET_NAME="TIGER-Lab/MMLU-Pro"
SPLIT="test"
REVISION=""
OUTPUT=""
LIMIT=""
SAMPLE_PER_CATEGORY=""

usage() {
  cat <<'EOF'
Usage: test/mmlu_pro/download.sh [options]

Options:
  --dataset-name NAME          HuggingFace dataset name. Default: TIGER-Lab/MMLU-Pro.
  --dataset-revision REV       Optional dataset revision or commit.
  --split SPLIT                Dataset split. Default: test.
  --output PATH                Output JSONL path. Default: baseline/downloaded/mmlu_pro_<split>.jsonl.
  --limit N                    Keep only first N rows after sampling.
  --sample-per-category N      Keep up to N rows per category.

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
    --sample-per-category)
      SAMPLE_PER_CATEGORY="$2"
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
  OUTPUT="$SCRIPT_DIR/baseline/downloaded/mmlu_pro_${SPLIT}.jsonl"
fi
mkdir -p "$(dirname "$OUTPUT")"

"$PYTHON_BIN" - "$DATASET_NAME" "$SPLIT" "$REVISION" "$OUTPUT" "$LIMIT" "$SAMPLE_PER_CATEGORY" <<'PY'
import json
import sys
from collections import defaultdict

try:
    from datasets import load_dataset
except ModuleNotFoundError as exc:
    raise SystemExit("Missing dependency: datasets. Run test/mmlu_pro/setup.sh first.") from exc

dataset_name, split, revision, output, limit, sample_per_category = sys.argv[1:7]
load_kwargs = {"split": split}
if revision:
    load_kwargs["revision"] = revision

rows = [dict(row) for row in load_dataset(dataset_name, **load_kwargs)]
if sample_per_category:
    keep = int(sample_per_category)
    grouped = defaultdict(list)
    for row in rows:
        category = str(row.get("category") or row.get("subject") or "unknown")
        if len(grouped[category]) < keep:
            grouped[category].append(row)
    rows = [row for category in sorted(grouped) for row in grouped[category]]
if limit:
    rows = rows[: int(limit)]

with open(output, "w", encoding="utf-8") as fout:
    for row in rows:
        fout.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

print(f"Wrote {len(rows)} rows to {output}")
PY
