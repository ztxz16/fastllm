# Qwen3.5 / Qwen3.6 / Qwen3.8 Benchmarks

[中文](qwen3.md) · [Benchmark index](../benchmark_en.md) · [Deployment guide](../qwen3_en.md)

Results are separated by exact checkpoint, precision, and inference path. Measurements from different Qwen3 releases must not be combined as if they were one model.

## Qwen3.6-27B-FP8 TP2

- Hardware: 2 × RTX 5090 with 32,607 MiB each, connected over PCIe without NVLink.
- FastLLM: 0.1.7.1, commit `6e01b27663c0`.
- Workload: context limit 1,024, 256 generated tokens per request, three measured runs after warmup.
- Thinking, prefix cache, and MTP were disabled.

~~~bash
FASTLLM_CUDA_GRAPH=1 \
ftllm server /data/models/Qwen3.6-27B-FP8 \
  --model_name qwen3.6-27b \
  --dtype auto \
  --tp 2 --cuda_embedding \
  --max_batch 64 \
  --max_context_length 1024 \
  --gpu_mem_ratio 0.98 \
  --prefix_cache false \
  --enable_thinking false
~~~

| Batch / concurrency | Decode throughput |
| ---: | ---: |
| 1 | 94.8087 token/s |
| 2 | 183.0459 token/s |
| 4 | 338.1496 token/s |
| 8 | 630.5965 token/s |
| 16 | 1163.9353 token/s |
| 32 | 1854.3000 token/s |
| 64 | 2930.4490 token/s |
| 128 / 256 | Unavailable due to capacity or warmup OOM |

The [full Qwen3.6 analysis](../qwen36_27b_fp8_tp2_benchmark_20260809.md) includes framework comparisons and operator profiling.

## Qwen3.8-27B-FP8 with DFlash2 TP2

- Hardware: 2 × RTX 5090 with 32,607 MiB each, connected over PCIe without NVLink.
- Draft model: a separate Qwen3.8-27B-DFlash2 checkpoint.
- Recommended DFlash block: 8, including the anchor token.

~~~bash
ftllm server /data/models/Qwen3.8-27B-FP8 \
  --model_name qwen3.8-dflash2 \
  --dtype auto \
  --tp 2 --cuda_embedding \
  --max_batch 1 \
  --tokens 8192 \
  --gpu_mem_ratio 0.98 \
  --speculative_algorithm dflash \
  --speculative_draft_model_path /data/models/Qwen3.8-27B-DFlash2 \
  --speculative_num_draft_tokens 8
~~~

| Path | Output length | Median fixed-stream throughput |
| --- | ---: | ---: |
| Target only | 512 | 81.45 token/s |
| MTP5 | 1024 | 273.65 token/s |
| DFlash B6 | 1024 | 287.29 token/s |
| DFlash B8 | 1024 | 381.00 token/s |
| DFlash B8 | 4096 | 377.52 token/s |

The final 128-question GSM8K run with DFlash B8 reached 207.06 token/s and 120/128 correct. This end-to-end result includes sampling and variable output lengths, so it is not directly comparable to the forced fixed-token stream. See the [full DFlash2 analysis](../dflash2_qwen38_27b_tp2_20260819.md).

## Qwen3.5

No standardized Qwen3.5 throughput result is currently publishable from this repository.

| Device layout | Suggested launch command | Speed |
| --- | --- | --- |
| Single CUDA GPU | `ftllm server /data/models/qwen3.5 --device cuda --max_batch 1` | Not measured |
| Two-GPU TP2 | `ftllm server /data/models/qwen3.5 --tp 2 --max_batch 8` | Not measured |
| CPU / NUMA | `ftllm server /data/models/qwen3.5 --device numa -t 64` | Not measured |
