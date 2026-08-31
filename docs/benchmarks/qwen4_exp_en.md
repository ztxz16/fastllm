# Qwen4-Exp / Qwen3.8-Flash-Next Benchmark

[中文](qwen4_exp.md) · [Benchmark index](../benchmark_en.md) · [Deployment guide](../qwen4_exp.md)

## Test scope

- Model: Qwen4-Exp / Qwen3.8-Flash-Next FP8 text model.
- Hardware: RTX PRO 6000 Blackwell 96 GB and a 72-core dual-NUMA host.
- Workload: 2 input tokens and at most 1 output token.
- Purpose: smoke-test the CPU, CUDA + CPU, and CUDA + NUMA paths.
- Limitation: this is not a sustained prefill or decode benchmark.

## Recommended commands

### CPU with CPU MoE

~~~bash
ftllm benchmark /data/models/qwen4-exp \
  --device cpu --moe_device cpu \
  --atype float32 --threads 64 \
  --input_tokens 2 --output_tokens 1 \
  --batch 1 --warmup 0 --temperature 0
~~~

### CUDA with CPU MoE

~~~bash
ftllm benchmark /data/models/qwen4-exp \
  --device cuda --moe_device cpu \
  --atype float16 --moe_atype float32 \
  --threads 64 \
  --input_tokens 2 --output_tokens 1 \
  --batch 1 --warmup 0 --temperature 0
~~~

### CUDA with NUMA MoE

~~~bash
ftllm benchmark /data/models/qwen4-exp \
  --device cuda --moe_device numa \
  --atype float16 --moe_atype float32 \
  --threads 64 \
  --input_tokens 2 --output_tokens 1 \
  --batch 1 --warmup 0 --temperature 0
~~~

## Measured results

| Execution path | TTFT | Short-input prefill |
| --- | ---: | ---: |
| CPU with CPU MoE | 274.44 ms | 7.29 token/s |
| CUDA with CPU MoE | 78.03 ms | 25.63 token/s |
| CUDA with NUMA MoE | 69.68 ms | 28.70 token/s |

These values compare the relative fixed cost of three paths for the same smoke input. With at most one output token, they do not provide sustained decode throughput.

## Deployment starting point

~~~bash
ftllm server /data/models/qwen4-exp \
  --model_name qwen4-exp \
  --device cuda --moe_device numa \
  --chunked_prefill_size 8192 \
  --gpu_mem_ratio 0.9
~~~

The PLE table stays on the CPU by default. Add `--ngram_device disk` when host memory is insufficient; disk-backed PLE does not yet have a published speed result.
