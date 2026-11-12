#!/usr/bin/env bash
# 双 V100 / CUDA 12.6 服务器验收：本脚本会编译，禁止在未获授权的本地运行。
set -euo pipefail

TP_EXIT_MODEL="$(cd -- "${1:-/home/models/Qwen3-1.7B}" && pwd -P)"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[[ -f "$TP_EXIT_MODEL/config.json" ]] || { echo "缺少模型 config.json：$TP_EXIT_MODEL" >&2; exit 1; }
TP_EXIT_BUILD=build-process-tp-cu126
export CUDA_HOME=/usr/local/cuda-12.6
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0,1
export USE_TORCH=0
hash -r

"$CUDACXX" --version
nvidia-smi
git log -1 --oneline
cmake -S . -B "$TP_EXIT_BUILD" -DUSE_CUDA=ON -DUSE_NUMAS=OFF -DBUILD_PROCESS_TP=ON \
    -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCUDA_ARCH=70
cmake --build "$TP_EXIT_BUILD" --target fastllm_process_tp_package fastllm_tp_check fastllm_tp_decode_check -j 4

TP_EXIT_COMMON=(--worker "$TP_EXIT_BUILD/fastllm_tp_worker" --devices 0,1
                --build-cache "$TP_EXIT_BUILD/CMakeCache.txt")
# 两轮都覆盖 11 个控制面退出用例；GPU 部分分别走 NCCL 和生产自定义规约。
python3 test/distributed/run_process_tp.py "${TP_EXIT_COMMON[@]}" --cuda \
    --check "$TP_EXIT_BUILD/fastllm_tp_check"
python3 test/distributed/run_process_tp.py "${TP_EXIT_COMMON[@]}" --cuda --custom-allreduce \
    --check "$TP_EXIT_BUILD/fastllm_tp_check"

# 合法输出、非法输入错误文本及生命周期回归。
python3 test/distributed/run_qwen3_decode.py "${TP_EXIT_COMMON[@]}" --check "$TP_EXIT_BUILD/fastllm_tp_decode_check"
python3 test/distributed/run_qwen3_decode.py "${TP_EXIT_COMMON[@]}" --check "$TP_EXIT_BUILD/fastllm_tp_decode_check" \
    --cuda-graph --custom-allreduce

# 真实模型从正式入口验证两种执行配置，检查输出及服务/进程清理。
TP_EXIT_ENTRY=(--generate "$TP_EXIT_BUILD/fastllm_tp_generate" --model "$TP_EXIT_MODEL"
               --package-root "$TP_EXIT_BUILD/process-tp-python")
python3 test/distributed/run_qwen3_entrypoint.py "${TP_EXIT_COMMON[@]}" "${TP_EXIT_ENTRY[@]}"
python3 test/distributed/run_qwen3_entrypoint.py "${TP_EXIT_COMMON[@]}" "${TP_EXIT_ENTRY[@]}" \
    --cuda-graph --custom-allreduce --prefix-cache

echo "退出监控验收完成；六组报告路径见各自 PASS 输出，包含 Markdown、CSV、JSON 和 XLSX。"
