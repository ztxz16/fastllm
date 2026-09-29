# Qwen4-Exp / Qwen3.8-Flash-Next Benchmark

[中文](qwen4_exp.md) · [Benchmark index](../benchmark_en.md) · [Deployment guide](../qwen4_exp.md)

## 2026-09-29: Flash-Next NVFP4 on two RTX 2080 Ti GPUs

All six configurations use inference code from commit `d6ef5c784`. See the [deployment commands](../qwen4.md) and [machine-readable measurements](qwen38_flash_next_2080ti_20260929.json).

Hardware: two **22 GiB** RTX 2080 Ti GPUs (22528 MiB each), an AMD EPYC 7452 with 32 cores / 64 threads, one NUMA node and 125.7 GiB RAM; Ubuntu 24.04.4, driver 595.84. This is not the standard 11 GiB 2080 Ti configuration.

Model: `Qwen3.8-Flash-Next-NVFP4`, text inference, FP16 activations, disk-backed PLE, 28 compute threads, batch 1, 1 GiB KV budget and GPU memory ratio 0.95. CUDA Graph, Triton, prefix/history caches and thinking are disabled. MTP weights remain loaded in both decode modes.

### Throughput

Prefill uses **4096 input tokens, chunk size 4096 and MTP=0**, followed by one output token. Throughput comes from the native `[Prompt]` Forward timer, including computation needed for the first token; it is not isolated kernel timing. TTFT measures request submission to the first token.

Decode uses a separate 512-input / fixed-512-output request for each of MTP=0 and MTP=3. Rate is `511 / (last token arrival − first token arrival)`, excluding loading, warmup and TTFT. Each measurement has one warmup and one measured request (128 output tokens for decode warmup). All configurations use identical tokenized coding prompts padded to the target input length. There is no HTTP/SSE transport in these timings.

| Deployment | Prefill token/s | Prefill TTFT s | Decode token/s | MTP3 token/s |
| --- | ---: | ---: | ---: | ---: |
| Sequential, cache off | 919.43 | 4.456 | 22.36 | 30.08 |
| Sequential, 8 GiB cache/GPU | 902.92 | 4.538 | 26.13 | 39.84 |
| Sequential, 12 resident expert layers | 960.99 | 4.273 | 23.99 | 32.92 |
| TP2, cache off | 1240.71 | 3.303 | 20.60 | 32.63 |
| TP2, 8 GiB cache/GPU | 1254.39 | 3.267 | 27.71 | 46.29 |
| TP2, 12 resident expert layers | 1321.08 | 3.102 | 22.57 | 37.82 |

Generated text and speculative acceptance can differ between deployments. These are single-round deployment comparisons, not forced identical-token microbenchmarks or agent-completion tests.

### Memory

| Deployment | GPU0 peak GiB | GPU1 peak GiB | Decode RSS GiB | Host HWM GiB |
| --- | ---: | ---: | ---: | ---: |
| Sequential, cache off | 7.81 | 10.25 | 72.60 | 72.86 |
| Sequential, 8 GiB cache/GPU | 15.83 | 18.26 | 79.64 | 80.42 |
| Sequential, 12 resident expert layers | 12.46 | 20.30 | 56.56 | 65.05 |
| TP2, cache off | 10.76 | 10.84 | 77.05 | 80.79 |
| TP2, 8 GiB cache/GPU | 18.78 | 18.86 | 84.09 | 88.37 |
| TP2, 12 resident expert layers | 18.33 | 18.41 | 61.05 | 80.62 |

GPU peaks are sampled every two seconds across loading, prefill and decode. Decode RSS is the median during measured decoding; host HWM includes loading. All cases exited normally with zero process swap and no OOM/CUDA errors. Lower steady-state RSS does not prove that loading fits within that amount of RAM.

TP2 resident placement uses `--moe_device_layers 36`, keeping the first 12 expert layers sharded across the GPUs. Sequential resident placement uses `cudapp=1:7` with `{'cuda:0':6,'cuda:1':6,'numa':36}`; its decoder split differs from the other sequential cases (24/24). Resident layers reserve cold experts too, while dynamic caching can retain active experts across layers.

### Reproduce

Run the [measurement script](../../test/benchmark/qwen38_hybrid.py) in an installed/built `ftllm` environment. Use a new output directory:

~~~bash
CUDA_VISIBLE_DEVICES=0,1 FT_NUMAS=1 FASTLLM_CUDA_GRAPH=0 \
python test/benchmark/qwen38_hybrid.py /data/models/Qwen3.8-Flash-Next-NVFP4 \
  --output-dir /tmp/qwen38-tp-cache8 \
  --tp 2 --moe_device numa --moe_cuda_cache 8g \
  --atype float16 --ngram_device disk --threads 28 --mtp 3 \
  --max_batch 1 --kv_cache_limit 1g --gpu_mem_ratio 0.95 \
  --chunked_prefill_size 4096 --prefix_cache false --cache_history false \
  --enable_thinking false
~~~

Replace the placement flags for other configurations; clear `FASTLLM_TP` for sequential placement. The script records raw tokens and arrival times in `results.json`, and native prefill throughput in `[Prompt]` log lines. The built-in `ftllm benchmark` Prefill value uses input tokens divided by TTFT and therefore has a slightly different timing boundary.

## Historical FP8 / Blackwell smoke tests

The following uses a different checkpoint format and machine and is retained separately from the NVFP4 measurements above.

### Test scope

- Model: Qwen4-Exp / Qwen3.8-Flash-Next FP8 text model.
- Hardware: RTX PRO 6000 Blackwell 96 GB and a 72-core dual-NUMA host.
- Workload: 2 input tokens and at most 1 output token.
- Purpose: smoke-test the CPU, CUDA + CPU, and CUDA + NUMA paths.
- Limitation: this is not a sustained prefill or decode benchmark.

### Recommended commands

#### CPU with CPU MoE

~~~bash
ftllm benchmark /data/models/qwen4-exp \
  --device cpu --moe_device cpu \
  --atype float32 --threads 64 \
  --input_tokens 2 --output_tokens 1 \
  --batch 1 --warmup 0 --temperature 0
~~~

#### CUDA with CPU MoE

~~~bash
ftllm benchmark /data/models/qwen4-exp \
  --device cuda --moe_device cpu \
  --atype float16 --moe_atype float32 \
  --threads 64 \
  --input_tokens 2 --output_tokens 1 \
  --batch 1 --warmup 0 --temperature 0
~~~

#### CUDA with NUMA MoE

~~~bash
ftllm benchmark /data/models/qwen4-exp \
  --device cuda --moe_device numa \
  --atype float16 --moe_atype float32 \
  --threads 64 \
  --input_tokens 2 --output_tokens 1 \
  --batch 1 --warmup 0 --temperature 0
~~~

### Measured results

| Execution path | TTFT | Short-input prefill |
| --- | ---: | ---: |
| CPU with CPU MoE | 274.44 ms | 7.29 token/s |
| CUDA with CPU MoE | 78.03 ms | 25.63 token/s |
| CUDA with NUMA MoE | 69.68 ms | 28.70 token/s |

These values compare the relative fixed cost of three paths for the same smoke input. With at most one output token, they do not provide sustained decode throughput.

### Deployment starting point

~~~bash
ftllm server /data/models/qwen4-exp \
  --model_name qwen4-exp \
  --device cuda --moe_device numa \
  --chunked_prefill_size 8192 \
  --gpu_mem_ratio 0.9
~~~

The PLE table stays on the CPU by default. Add `--ngram_device disk` when host memory is insufficient; disk-backed PLE was not measured in these FP8 smoke tests.
