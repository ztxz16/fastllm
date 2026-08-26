#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

SOURCE="modelscope"
DATASET_NAME="cais/hle"
SPLIT="test"
REVISION="5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"
MODELSCOPE_REVISION="1ec1f1f25ed4ad891e3a81d1cbc08f261f5e77c6"
MODELSCOPE_FILE="data/test-00000-of-00001.parquet"
MODELSCOPE_SHA256="6d0ee0602e8aea6b159509577e884f48ecac7b8e3f6822a35f51335a446c726a"
MODELSCOPE_DIR="$SCRIPT_DIR/baseline/downloaded/modelscope"
OUTPUT="$SCRIPT_DIR/baseline/downloaded/hle_test_text_mc_seed42_50.jsonl"
ANSWER_TYPE="multipleChoice"
LIMIT="50"
SEED="42"
TEXT_ONLY="1"
SHUFFLE="1"

usage() {
  cat <<'EOF'
Usage: test/hle/download.sh [options]

By default this creates the reproducible 50-question text-only multiple-choice
subset used by the FastLLM target-only versus DFlash2 comparison.

Options:
  --source SOURCE          modelscope (default) or huggingface.
  --dataset-name NAME       Hugging Face dataset. Default: cais/hle.
  --dataset-revision REV    Dataset revision. Default: pinned official revision.
  --modelscope-revision REV ModelScope revision. Default: pinned mirror revision.
  --modelscope-dir PATH     Local directory for the ModelScope parquet.
  --split SPLIT             Dataset split. Default: test.
  --output PATH             Output JSONL path.
  --answer-type TYPE        Answer type filter. Default: multipleChoice.
  --limit N                 Number of selected questions. Default: 50.
  --seed N                  Shuffle seed. Default: 42.
  --include-images          Do not filter image questions.
  --no-shuffle              Keep upstream order.

The ModelScope mirror is public and is verified against the official parquet
SHA-256. Hugging Face is gated; when using --source huggingface, accept its
terms in the browser and run `hf auth login`. Downloaded data is gitignored.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source) SOURCE="$2"; shift 2 ;;
    --dataset-name) DATASET_NAME="$2"; shift 2 ;;
    --dataset-revision) REVISION="$2"; shift 2 ;;
    --modelscope-revision) MODELSCOPE_REVISION="$2"; shift 2 ;;
    --modelscope-dir) MODELSCOPE_DIR="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --answer-type) ANSWER_TYPE="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --include-images) TEXT_ONLY="0"; shift ;;
    --no-shuffle) SHUFFLE="0"; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

mkdir -p "$(dirname "$OUTPUT")"

"$PYTHON_BIN" - "$SOURCE" "$DATASET_NAME" "$SPLIT" "$REVISION" "$OUTPUT" \
  "$ANSWER_TYPE" "$LIMIT" "$SEED" "$TEXT_ONLY" "$SHUFFLE" \
  "$MODELSCOPE_REVISION" "$MODELSCOPE_FILE" "$MODELSCOPE_SHA256" \
  "$MODELSCOPE_DIR" <<'PY'
import hashlib
import json
import random
import re
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ModuleNotFoundError as exc:
    raise SystemExit("Missing dependency: run test/hle/setup.sh first.") from exc

(
    source,
    dataset_name,
    split,
    revision,
    output,
    answer_type,
    limit,
    seed,
    text_only,
    shuffle,
    modelscope_revision,
    modelscope_file,
    modelscope_sha256,
    modelscope_dir,
) = sys.argv[1:15]

fields = ("id", "question", "image", "answer", "answer_type", "category", "raw_subject")

if source == "modelscope":
    try:
        from modelscope.hub.file_download import dataset_file_download
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: run test/hle/setup.sh first.") from exc
    parquet = Path(
        dataset_file_download(
            dataset_id=dataset_name,
            file_path=modelscope_file,
            revision=modelscope_revision,
            local_dir=modelscope_dir,
        )
    )
    digest = hashlib.sha256()
    with parquet.open("rb") as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != modelscope_sha256:
        raise SystemExit(
            f"ModelScope parquet SHA-256 mismatch: expected {modelscope_sha256}, "
            f"got {actual_sha256}"
        )
    rows = load_dataset(
        "parquet", data_files={split: str(parquet)}, split=split
    ).select_columns(fields)
elif source == "huggingface":
    kwargs = {"split": split}
    if revision:
        kwargs["revision"] = revision
    try:
        rows = load_dataset(dataset_name, **kwargs).select_columns(fields)
    except Exception as exc:
        message = str(exc)
        if (
            "gated" in message.lower()
            or "401" in message
            or "authenticated" in message.lower()
        ):
            raise SystemExit(
                "cais/hle requires Hugging Face authorization. Accept the dataset "
                "terms, then run `hf auth login` and retry."
            ) from exc
        raise
else:
    raise SystemExit("--source must be modelscope or huggingface")

def canonical(value):
    return re.sub(r"[^a-z]", "", str(value or "").lower())

wanted = canonical(answer_type)
rows = [row for row in rows if canonical(row.get("answer_type")) == wanted]
if text_only == "1":
    rows = [row for row in rows if not row.get("image")]
if shuffle == "1":
    random.Random(int(seed)).shuffle(rows)
rows = rows[: int(limit)]
if len(rows) != int(limit):
    raise SystemExit(f"Requested {limit} rows but only selected {len(rows)}")

with open(output, "w", encoding="utf-8") as fout:
    for row in rows:
        compact = {field: row.get(field) for field in fields}
        fout.write(json.dumps(compact, ensure_ascii=False, default=str) + "\n")

case_ids = [f"{split}:{row.get('id')}" for row in rows]
digest = hashlib.sha256("\n".join(case_ids).encode("utf-8")).hexdigest()
print(f"Wrote {len(rows)} rows to {output}")
print(f"Selection SHA-256: {digest}")
PY
