#!/bin/sh

set -eu

APP_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    LD_LIBRARY_PATH="${APP_ROOT}/lib:${LD_LIBRARY_PATH}"
else
    LD_LIBRARY_PATH="${APP_ROOT}/lib"
fi
export LD_LIBRARY_PATH
export FTLLM_RUNTIME_DIR="${FTLLM_RUNTIME_DIR:-${APP_ROOT}}"
export FTLLM_LAUNCHER_DATA_DIR="${FTLLM_LAUNCHER_DATA_DIR:-${APP_ROOT}/data}"
export ELECTRON_OZONE_PLATFORM_HINT="${ELECTRON_OZONE_PLATFORM_HINT:-auto}"

exec "${APP_ROOT}/FastLLM-Launcher.bin" "$@"
