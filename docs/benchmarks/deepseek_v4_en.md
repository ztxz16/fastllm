# DeepSeek-V4 Sparse Attention Benchmark

[中文](deepseek_v4.md) · [Benchmark index](../benchmark_en.md) · [Deployment guide](../deepseek_en.md)

## Test configuration

- Model: DeepSeek-V4 / DeepSeek-V4-Flash.
- Parallelism: TP8.
- CUDA architecture: SM120/SM121.
- Input/output: 128 / 256 tokens.
- CUDA Graph and custom AllReduce were enabled.
- The source report did not record the exact GPU SKU, memory, or interconnect, so the results must not be attributed to a specific GPU model.

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

## Full-model results

| Metric | Before optimization | SM120 fast path | Change |
| --- | ---: | ---: | ---: |
| TTFT | 415.27 ms | 405.47 ms | -2.36% |
| TPOP | 13.51 ms/token | 10.01 ms/token | -25.91% |
| Decode throughput | 74.03 token/s | 99.93 token/s | +34.99% |
| Total throughput | 66.32 token/s | 86.57 token/s | +30.53% |
| Total time | 3.8600 s | 2.9572 s | -23.39% |

## Sparse-attention core

| Path | Mean GPU time per layer |
| --- | ---: |
| Original serial FastLLM Triton | 81.859 µs |
| vLLM FlashInfer SM120 main kernel plus merge | 13.660 µs |
| FastLLM SM120 split plus merge | 5.796 µs |

This is a short-context workload and must not be extrapolated to long contexts with top-k 512 or 1,024.

## Other layouts

| Device layout | Suggested launch command | Speed |
| --- | --- | --- |
| TP8 on SM120/SM121 | Use the command above | 99.93 decode token/s |
| Single GPU with NUMA MoE | `ftllm server /data/models/deepseek-v4 --device cuda --moe_device numa` | Not measured |
| GPU with CPU MoE | `ftllm server /data/models/deepseek-v4 --device cuda --moe_device cpu` | Not measured |
| GPU, NUMA, and disk experts | `ftllm server /data/models/deepseek-v4 --device cuda --moe_device "{'cuda':1,'numa':8,'disk':1}"` | Not measured |

Different layouts have different bottlenecks; TP8 results cannot be used to estimate hybrid-MoE or disk-expert performance.
