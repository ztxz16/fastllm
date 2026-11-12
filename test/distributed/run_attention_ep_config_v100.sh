#!/usr/bin/env bash
# 只做 Python/Driver API 配置验收；不编译、不加载权重、不启动推理 worker。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
export PYTHONNOUSERSITE=1
attention_ep_python="${ATTENTION_EP_PYTHON:-/home/venvs/fastllm-dp/bin/python3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-GPU-a88c9669-989e-a8dd-22a0-94a45b467c49,GPU-0310107f-031e-6fcd-eb65-6dfa45981711,GPU-f0a99453-777f-8b16-89a3-bab78019f23d,GPU-081484a6-9730-453b-59a2-32bd75ab6c2c}"
IFS=',' read -r -a attention_ep_devices <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#attention_ep_devices[@]} -ne 4 ]]; then
    echo '需要四个 CUDA_VISIBLE_DEVICES 选择项；推荐使用完整 GPU UUID。' >&2
    exit 1
fi

for attention_ep_size in 4 2; do
    "$attention_ep_python" test/distributed/run_parallel_config.py \
        --real-cuda --dp 1 --tp 1 --attention-dp "$attention_ep_size" --ep "$attention_ep_size"
done
export CUDA_VISIBLE_DEVICES="${attention_ep_devices[3]},${attention_ep_devices[2]},${attention_ep_devices[1]},${attention_ep_devices[0]}"
"$attention_ep_python" test/distributed/run_parallel_config.py \
    --real-cuda --dp 1 --tp 1 --attention-dp 4 --ep 4
