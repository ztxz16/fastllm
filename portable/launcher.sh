#!/usr/bin/env bash

set -euo pipefail

FTLLM_BUNDLE_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=/dev/null
source "${FTLLM_BUNDLE_ROOT}/libexec/activate.sh"

case "$(basename -- "$0")" in
    ftllm)
        exec "${FTLLM_BUNDLE_ROOT}/runtime/bin/python3" -m ftllm.cli "$@"
        ;;
    ftllm-agent-runtime)
        exec "${FTLLM_BUNDLE_ROOT}/runtime/bin/python3" -m ftllm_agent_runtime.cli "$@"
        ;;
    python)
        exec "${FTLLM_BUNDLE_ROOT}/runtime/bin/python3" "$@"
        ;;
    pip)
        exec "${FTLLM_BUNDLE_ROOT}/runtime/bin/python3" -m pip "$@"
        ;;
    ftllm-check)
        exec "${FTLLM_BUNDLE_ROOT}/runtime/bin/python3" \
            "${FTLLM_BUNDLE_ROOT}/libexec/check.py" "$@"
        ;;
    *)
        printf '未知的 ftllm 绿色包启动器：%s\n' "$0" >&2
        exit 2
        ;;
esac
