# DeepSeek-V4 Sparse Attention Benchmark

[中文](deepseek_v4.md) · [Benchmark index](../benchmark_en.md) · [Deployment guide](../deepseek_en.md)

## Test configuration

- Model: DeepSeek-V4 / DeepSeek-V4-Flash.
- Parallelism: TP8.
- Tested CUDA architecture: SM120; the kernel path supports SM120/SM121.
- Input/output: 128 / 256 tokens.
- CUDA Graph and custom AllReduce were enabled.
- The source report did not record the exact GPU SKU, memory, or interconnect, so the results must not be attributed to a specific GPU model.
- The measurement date and source revision were not recorded; these results do not establish current HEAD performance.

## Recommended TP8 command

~~~bash
FASTLLM_CUDA_GRAPH=1 \
FASTLLM_CUDA_CUSTOM_ALLREDUCE=1 \
ftllm server /data/models/deepseek-v4 \
  --model_name deepseek-v4 \
  --tp 8 --triton \
  --max_batch 1
~~~

`--triton` uses Triton when it is available in the active Python environment and otherwise falls back to the built-in CUDA path.

## Results

| Metric | Measured value |
| --- | ---: |
| TTFT | 405.47 ms |
| TPOP | 10.01 ms/token |
| Decode throughput | 99.93 token/s |
| Total throughput | 86.57 token/s |
| Total time | 2.9572 s |

SM120 split + merge sparse attention averaged 5.796 µs per layer in the CUDA Graph trace. These measurements cover short context only; they do not establish performance for long context, other hardware or hybrid expert placement.

Kernel contracts, numerical limits and fallback behavior are documented in the [sparse attention guide](../deepseek_v4_sparse_attention.md).
