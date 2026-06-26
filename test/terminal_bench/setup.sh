#!/usr/bin/env bash
set -euo pipefail

HARBOR_VERSION="${HARBOR_VERSION:-0.1.28}"
HARBOR_PYTHON="${HARBOR_PYTHON:-3.12}"

if command -v harbor >/dev/null 2>&1; then
  harbor --help >/dev/null
  echo "Harbor is already installed: $(command -v harbor)"
  uv tool list 2>/dev/null | grep -E '^harbor ' || true
  exit 0
fi

if command -v uv >/dev/null 2>&1; then
  uv tool install "harbor==$HARBOR_VERSION" --python "$HARBOR_PYTHON" --with socksio
else
  PYTHON_BIN="${PYTHON:-python3.12}"
  "$PYTHON_BIN" -m pip install -U "harbor==$HARBOR_VERSION" socksio
fi

harbor --help >/dev/null
echo "Harbor installed: $(command -v harbor)"
