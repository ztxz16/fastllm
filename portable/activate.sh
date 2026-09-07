# This file is sourced by the public launchers. FTLLM_BUNDLE_ROOT must already
# point at the extracted bundle root.

if [[ -z "${FTLLM_BUNDLE_ROOT:-}" || ! -d "${FTLLM_BUNDLE_ROOT}/runtime" ]]; then
    printf '无效的 FTLLM_BUNDLE_ROOT：%s\n' "${FTLLM_BUNDLE_ROOT:-<empty>}" >&2
    return 1
fi

export FTLLM_HOME="$FTLLM_BUNDLE_ROOT"
export PATH="${FTLLM_BUNDLE_ROOT}:${FTLLM_BUNDLE_ROOT}/runtime/bin:${PATH:-/usr/bin:/bin}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONUTF8=1
unset PYTHONHOME
unset PYTHONPATH

_ftllm_library_path="${FTLLM_BUNDLE_ROOT}/runtime/lib"
for _ftllm_lib_dir in "${FTLLM_BUNDLE_ROOT}"/runtime/lib/python*/site-packages/nvidia/*/lib; do
    [[ -d "$_ftllm_lib_dir" ]] || continue
    _ftllm_library_path="${_ftllm_library_path}:${_ftllm_lib_dir}"
done
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    _ftllm_library_path="${_ftllm_library_path}:${LD_LIBRARY_PATH}"
fi
export LD_LIBRARY_PATH="$_ftllm_library_path"

for _ftllm_cert_file in "${FTLLM_BUNDLE_ROOT}"/runtime/lib/python*/site-packages/certifi/cacert.pem; do
    if [[ -f "$_ftllm_cert_file" ]]; then
        export SSL_CERT_FILE="${SSL_CERT_FILE:-$_ftllm_cert_file}"
        break
    fi
done

unset _ftllm_library_path _ftllm_lib_dir _ftllm_cert_file
