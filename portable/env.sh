# 在 bash 中执行：source ./env.sh

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf '请使用 source 加载环境：source %q\n' "$0" >&2
    exit 2
fi

FTLLM_BUNDLE_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=/dev/null
source "${FTLLM_BUNDLE_ROOT}/libexec/activate.sh"
unset FTLLM_BUNDLE_ROOT
