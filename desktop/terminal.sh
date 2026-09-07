#!/usr/bin/env bash

set -euo pipefail

FTLLM_BUNDLE_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd -- "${FTLLM_BUNDLE_ROOT}/.."
# shellcheck source=/dev/null
source "${FTLLM_BUNDLE_ROOT}/libexec/activate.sh"
printf 'FastLLM 终端已就绪，可以直接输入 ftllm 命令。\n\n'
printf '  ftllm --help        查看命令帮助\n'
printf '  ftllm launch        打开网页 Launcher\n'
printf '  ftllm-check         检查运行环境\n'
printf '  exit               退出终端\n\n'
# Avoid user startup files replacing the bundled Python/PATH with another env.
exec bash --noprofile --norc -i
