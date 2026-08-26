#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

"$PYTHON_BIN" -m pip install -r "$SCRIPT_DIR/requirements.txt"
