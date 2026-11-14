#!/usr/bin/env bash
# 服务器专用：显式包含第三步原生编译；不要在禁止编译的本地工作区执行。
set -euo pipefail
AR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$AR_ROOT"
export AR_PYTHON="${AR_PYTHON:-$(command -v python3)}"
export AR_RUN="${AR_RUN:-$AR_ROOT/test/distributed/results/$(date -u +%Y%m%dT%H%M%SZ)_allreduce_$$}"
AR_PAIR="${AR_PAIR:-0,1}"
AR_GROUPS="${AR_GROUPS:-0,1;2,3}"
AR_DEVICES="${AR_DEVICES:-0,1,2,3}"
AR_TIMEOUT="${AR_TIMEOUT:-900}"
if [[ -e "$AR_RUN" ]]; then
  printf '报告根目录已存在，请设置新的 AR_RUN：%s\n' "$AR_RUN" >&2
  exit 1
fi
AR_BASE=(test/distributed/benchmark_allreduce.py --python "$AR_PYTHON")
AR_COMMON=(--pair "$AR_PAIR" --devices "$AR_DEVICES" --timeout "$AR_TIMEOUT" \
  --native "${AR_BUILD:-$AR_ROOT/build-allreduce-cu126}/libfastllm_allreduce_adapter.so")

# 后续步骤出错时仍生成已有阶段的汇总，原始退出码不会被报告写入覆盖。
ar_partial_report() {
  local ar_status=$?
  if [[ $ar_status -ne 0 && ! -e "$AR_RUN/07-report" ]]; then
    "$AR_PYTHON" "${AR_BASE[@]}" report --input-root "$AR_RUN" --output "$AR_RUN/07-report" --no-plots || true
  fi
}
trap ar_partial_report EXIT

"$AR_PYTHON" "${AR_BASE[@]}" plan --devices "$AR_DEVICES" --output "$AR_RUN/01-plan"
"$AR_PYTHON" "${AR_BASE[@]}" preflight --devices "$AR_DEVICES" --timeout "$AR_TIMEOUT" --output "$AR_RUN/02-preflight"
bash test/distributed/build_allreduce_adapter.sh
"$AR_PYTHON" "${AR_BASE[@]}" smoke "${AR_COMMON[@]}" --output "$AR_RUN/03-smoke"
"$AR_PYTHON" "${AR_BASE[@]}" correctness "${AR_COMMON[@]}" --output "$AR_RUN/04-correctness"
"$AR_PYTHON" "${AR_BASE[@]}" performance "${AR_COMMON[@]}" --correctness "$AR_RUN/04-correctness" --output "$AR_RUN/05-performance"
"$AR_PYTHON" "${AR_BASE[@]}" concurrent "${AR_COMMON[@]}" --groups "$AR_GROUPS" --correctness "$AR_RUN/04-correctness" --output "$AR_RUN/06-concurrent"
"$AR_PYTHON" "${AR_BASE[@]}" report --input-root "$AR_RUN" --output "$AR_RUN/07-report"
printf 'AllReduce reports: %s\n' "$AR_RUN"
