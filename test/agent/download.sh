#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_FILE="$SCRIPT_DIR/baseline/default_cases.jsonl"

if [[ ! -f "$DATA_FILE" ]]; then
  echo "Missing baseline data: $DATA_FILE" >&2
  exit 1
fi

case_count="$(wc -l < "$DATA_FILE" | tr -d ' ')"
echo "Agent tool-use uses local baseline data only."
echo "Data file: $DATA_FILE"
echo "Cases: $case_count"
