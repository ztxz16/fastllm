# Qwen3.8-Flash-Next User Guide

[中文](README.md) · [Project home](../../README_EN.md)

This guide covers text serving with Qwen4-Exp / Qwen3.8-Flash-Next: hybrid inference, serial and parallel use of two GPUs, expert caching, fixed GPU expert layers, MTP, and API requests. The two-GPU examples use the NVFP4 model.

## Prepare the model

Install a FastLLM version that supports the model and prepare a complete local model directory containing its configuration, tokenizer, and Safetensors weights. Replace `/data/models/Qwen3.8-Flash-Next-NVFP4` with your model path.

Hybrid inference requires enough host memory. Store the model on an SSD when using disk-backed PLE.

## Quick start

Run the backbone on one GPU and the experts on the CPU NUMA backend:

```bash
ftllm server /data/models/Qwen3.8-Flash-Next-NVFP4 \
  --model_name qwen3.8-flash-next \
  --host 0.0.0.0 --port 8080 \
  --device cuda --moe_device numa \
  --atype float16 --ngram_device disk --threads 28 \
  --chunked_prefill_size 4096 --gpu_mem_ratio 0.9 \
  --prefix_cache true --enable_thinking false
```

After the model loads, connect clients to `http://SERVER:8080/v1` with the model name `qwen3.8-flash-next`. Adjust the thread count for your CPU's physical cores.

<a id="deployment"></a>

## Two-GPU hybrid deployment

These examples target two **22 GiB RTX 2080 Ti cards**, an AMD EPYC 7452, and approximately 126 GiB of host memory. Standard 11 GiB cards require smaller expert caches or fewer GPU-resident expert layers.

Set the common options in Bash, then choose one deployment command:

```bash
MODEL=/data/models/Qwen3.8-Flash-Next-NVFP4
unset FASTLLM_TP
export CUDA_VISIBLE_DEVICES=0,1
export FASTLLM_CUDA_GRAPH=0

COMMON=(
  --model_name qwen3.8-flash-next
  --host 0.0.0.0 --port 8080
  --atype float16 --ngram_device disk --threads 28
  --mtp 3 --temperature 0 --top_k 1 --repeat_penalty 1
  --max_batch 1 --kv_cache_limit 1g --gpu_mem_ratio 0.95
  --chunked_prefill_size 4096
  --prefix_cache true --enable_thinking false
)
```

These examples disable CUDA Graph and enable prefix caching for repeated text requests. Change `--mtp 3` to `--mtp 0` if the model has no MTP weights or you want ordinary decoding.

### Serial GPUs: expert cache off or on

Serial deployment places consecutive backbone layers on each GPU. `cudapp=2` splits the backbone evenly between two cards.

```bash
# Expert cache disabled
ftllm server "$MODEL" "${COMMON[@]}" \
  --device cudapp=2 --moe_device numa --moe_cuda_cache 0

# 8 GiB of expert cache per GPU
ftllm server "$MODEL" "${COMMON[@]}" \
  --device cudapp=2 --moe_device numa --moe_cuda_cache 8g
```

### Parallel GPUs: expert cache off or on

`--tp 2` makes both GPUs work on each layer. Use it without `cudapp`.

```bash
# Expert cache disabled
ftllm server "$MODEL" "${COMMON[@]}" \
  --tp 2 --moe_device numa --moe_cuda_cache 0

# 8 GiB of expert cache per GPU
ftllm server "$MODEL" "${COMMON[@]}" \
  --tp 2 --moe_device numa --moe_cuda_cache 8g
```

`--moe_cuda_cache 8g` sets a **per-GPU** budget, approximately 16 GiB across two GPUs. Leave room for backbone weights, KV cache, and temporary computation buffers.

### Fixed GPU expert layers with tensor parallelism

Disable the dynamic expert cache and place all experts from the first 12 layers on the GPUs, with the remaining 36 layers on NUMA:

```bash
ftllm server "$MODEL" "${COMMON[@]}" \
  --tp 2 --moe_device numa --moe_device_layers 36 --moe_cuda_cache 0
```

`--moe_device_layers 36` means the **last 36 layers** use `--moe_device numa`. Experts in the first 12 layers are sharded across both GPUs. Reducing this number places more layers on the GPUs and uses more VRAM.

### Fixed GPU expert layers with serial GPUs

Use the following paired backbone and expert mappings:

```bash
ftllm server "$MODEL" "${COMMON[@]}" \
  --device cudapp=1:7 \
  --moe_device "{'cuda:0':6,'cuda:1':6,'numa':36}" \
  --moe_cuda_cache 0
```

Backbone layers 0–5 run on GPU0 and layers 6–47 on GPU1. Experts in layers 0–5 reside on GPU0, layers 6–11 on GPU1, and the last 36 layers and MTP experts use NUMA. When changing layer counts, adjust both mappings so GPU experts share a device with their corresponding backbone layer.

Fixed placement allocates VRAM for every expert in selected layers. Dynamic caching retains experts based on access patterns. Their speed and memory requirements differ.

<a id="parameters"></a>

## Common options

| Option | Purpose |
| --- | --- |
| `--device cuda` | Single-GPU backbone |
| `--device cudapp=2` | Backbone layers split serially across two GPUs |
| `--tp 2` | Two-GPU tensor parallelism |
| `--moe_device numa` | Run experts on the CPU NUMA backend |
| `--moe_cuda_cache 0` / `8g` | Disable expert caching / set an 8 GiB per-GPU budget |
| `--moe_device_layers 36` | Last 36 expert layers use the specified MoE device |
| `--ngram_device disk` | Read PLE rows from disk to reduce resident host memory |
| `--ngram_device cpu` | Keep the PLE table in host memory |
| `--threads 28` | CPU compute threads |
| `--mtp 0` / `3` | Disable MTP / propose up to 3 draft tokens per step |
| `--chunked_prefill_size 4096` | Process up to 4096 input tokens per chunk |
| `--kv_cache_limit 1g` | KV budget; adjust for context length and concurrency |
| `--gpu_mem_ratio 0.95` | Fraction of available GPU memory to use |
| `--prefix_cache true` | Reuse repeated text prefixes across requests |
| `--max_context_length` | Context limit, subject to model and cache capacity |
| `--enable_thinking true` | Enable thinking mode |

### MTP

MTP requires matching MTP weights in the model directory and a CUDA backbone. Experts may run on CUDA or NUMA. The maximum `--mtp` value is 8; start with 3 and adjust for your workload. Speed depends on draft acceptance. The two-GPU TP examples use greedy sampling: `--temperature 0 --top_k 1 --repeat_penalty 1`.

### Memory and context length

Use `--ngram_device disk` when host memory is limited. If VRAM is insufficient, reduce the expert cache or GPU-resident layer count, then adjust KV capacity, chunk size, and concurrency for your requests.

Larger prefill chunks reduce the number of chunks for long inputs but require more temporary VRAM. Prefix caching reuses text prefixes; expert caching retains expert weights. They are independent settings.

<a id="api"></a>

## Call the service

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-flash-next",
    "messages": [{"role": "user", "content": "Introduce yourself in three sentences."}],
    "max_tokens": 512,
    "temperature": 0,
    "stream": false
  }'
```

For remote requests, replace `127.0.0.1` with the server address. If you set `--api_key` at startup, include an `Authorization: Bearer YOUR_KEY` header.

To use a browser chat interface:

```bash
ftllm webui --api_base http://127.0.0.1:8080/v1
```

WebUI connects to the running API service. For tool calls, add `--tool_call_parser auto` to the server command and send the standard `tools` field in API requests.

<a id="performance"></a>

## Performance reference

These NVFP4 measurements were recorded on 2026-09-29 with the two 22 GiB RTX 2080 Ti cards and EPYC 7452 described above, using 28 threads and disk-backed PLE. Prefill used 4096 input tokens and a chunk size of 4096; decoding used 512 input and 512 output tokens, excluding time to the first token. Prefix reuse, CUDA Graph, and thinking were disabled. Each configuration was measured once with one active request.

| Deployment | Prefill token/s | Ordinary decode token/s | MTP=3 token/s |
| --- | ---: | ---: | ---: |
| Serial, cache off | 919.43 | 22.36 | 30.08 |
| Serial, 8 GiB cache per GPU | 902.92 | 26.13 | 39.84 |
| Serial, 12 fixed expert layers | 960.99 | 23.99 | 32.92 |
| TP2, cache off | 1240.71 | 20.60 | 32.63 |
| TP2, 8 GiB cache per GPU | 1254.39 | 27.71 | 46.29 |
| TP2, 12 fixed expert layers | 1321.08 | 22.57 | 37.82 |

In this measurement, TP2 with an 8 GiB cache per GPU had the highest decode rate, while TP2 with 12 fixed expert layers had the highest prefill rate. Actual speed varies with the workload and hardware.

<a id="gguf-resident-performance"></a>

### GGUF IQ2_XS: experts resident on two GPUs (2026-10-01)

The cleaned-up GGUF implementation was measured on the same two 22 GiB RTX 2080 Ti cards and EPYC 7452, using `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS`. Dense layers and experts use TP2 with FP16 activations and CUDA Graph enabled. MTP is disabled and no external MTP weights are loaded. The expert-cache budget is zero; all expert weights remain in their native GGUF format on the GPUs. Embedding stays on CPU (`--low_gpu_mem`) and PLE uses disk, so “GPU-only” here refers to Dense/MoE computation and expert residency.

Keep both GGUF shards in the same directory and pass the first shard as the model. The main startup parameters are below; change `--chunked_prefill_size` for each group in the table.

```bash
CUDA_VISIBLE_DEVICES=0,1 FASTLLM_CUDA_GRAPH=1 FT_NUMAS=1 \
ftllm server /data/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
  --tp 2 --device cuda:0,1 --moe_device cuda:0,1 \
  --atype float16 --low_gpu_mem --ngram_device disk --threads 28 \
  --moe_cuda_cache 0 --mtp 0 --max_batch 1 --tokens 8192 \
  --kv_cache_limit 1g --gpu_mem_ratio 0.95 --chunked_prefill_size 1024 \
  --prefix_cache false --cache_history false --enable_thinking false \
  --temperature 0 --top_k 1 --repeat_penalty 1
```

Timing uses the native generation API without HTTP overhead or a profiler, with greedy sampling and batch size 1. Each group has one warmup followed by three measured requests; the table reports medians. Decode uses 512 input and 512 output tokens, calculated as `511 / (last token arrival - first token arrival)`. Prefill uses 4096 input tokens and one output token, calculated as `4096 / TTFT`, including first-token latency.

| Test | Chunk | Three runs (token/s) | Median (token/s) | Median TTFT (s) |
| --- | ---: | --- | ---: | ---: |
| Decode | 32 | 71.11 / 70.87 / 70.61 | **70.87** | 1.23 |
| Decode | 512 | 69.99 / 69.77 / 69.56 | **69.77** | 0.49 |
| 4096-token prefill | 1024 | 1206.47 / 1205.13 / 1200.43 | **1205.13** | 3.40 |

Outputs match across the three runs within each chunk group. The two chunk groups diverge at output token 200, so the roughly 1.6% decode-rate difference cannot be attributed solely to chunk size. Model, placement and Graph settings also differ from the NVFP4 hybrid results above; these are not a controlled comparison of quantization formats.

| GPU memory (GiB) | GPU0 | GPU1 |
| --- | ---: | ---: |
| Logical weights | 19.3554 | 19.3554 |
| Used after loading and initialization warmup | 20.2088 | 20.2049 |
| Sampled peak over the run | 21.0586 | 20.8379 |

Logical weights come from the TP preparation manifest, loaded memory from `cudaMemGetInfo`, and peaks from `nvidia-smi` sampled every two seconds, which may miss short transients. CPU embedding occupies 2.368 GiB; process host-memory HWM was 41.905 GiB with zero swap. Each GPU holds 49,152 native GGUF expert gate/up and down tensors. CPU/hybrid MoE path counters, expert-cache hits/misses/payload and MTP verifier calls remain zero for all requests. Arithmetic and JSON checks pass, and every 4096-token prefill at chunk=1024 completes without OOM.

[Full per-run data, configuration and validation](../benchmarks/qwen38_flash_next_iq2xs_2080ti_20261001.json). This measurement used an isolated build with diagnostic counters, native library SHA256 `0e9aa1b66b2d502e150395a40bfca36f2b43c0ea937329f43f89d60861add82b`; the installed package was unchanged.

### GGUF Q2_0: expert MMQ and Dense decode optimization (2026-10-01)

The same machine and TP2/Graph/MTP=0 configuration were used to compare `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` before and after optimization. Both builds used identical input tokens, output lengths, warmup and three-run median timing. Changes include 16-bit Q2 expert MMQ loads with vector unpacking, quantizing each input token once before gathering expert routes, and Q8/DP4A dot products for 1–8-row Q2 Dense operations. Dispatch depends on format and tensor dimensions, with no model-name checks or new environment variables.

| Measurement | Chunk | Before token/s | Three runs after, token/s | Median after, token/s | Change |
| --- | ---: | ---: | --- | ---: | ---: |
| Decode, 512 input / 512 output | 32 | 70.14 | 71.12 / 70.74 / 70.44 | **70.74** | +0.85% |
| Decode, 512 input / 512 output | 512 | 68.91 | 69.76 / 69.58 / 69.49 | **69.58** | +0.97% |
| 4096-token prefill, 1 output | 1024 | 1130.22 | 1357.01 / 1354.38 / 1352.08 | **1354.38** | **+19.83%** |

Across two Nsight Systems prefill captures, GPU0 expert gate/up kernel time falls from 1041.96 to 706.54 ms, down from 638.04 to 496.87 ms, and routing/quantization/reduction from 232.44 to 124.61 ms. Key expert counts also pass the GPU1 audit. Decode has only seven Q2 Dense projections per step; total Dense GEMV time falls from 6.131 to 6.003 ms, with other quantization formats and the output head still accounting for substantial time.

GPU0 cold-cache NCU replay shows the two Q2 Dense projections increasing DRAM bandwidth utilization from 16.01% / 17.57% to 56.92% / 61.73%, with an additional 2.08 μs input quantization per operation. Q2 expert gate/up MMQ falls from 4.431 to 3.005 ms while DRAM utilization rises from 3.32% to 4.90%. Expert input quantization changes from 676.86 μs to 38.24 μs for token quantization plus 116.13 μs for gathering. MMQ still does not saturate DRAM bandwidth. These counters use real expert snapshots, real Dense weights and synthetic nonzero Dense inputs with uncontrolled clocks; they are isolated-operator measurements, not full-request bandwidth.

Expert input quantization reuses the product buffer, preserving weight storage and persistent workspace size. Logical weights remain 18.6634 GiB per GPU. Used memory after loading and initialization is unchanged at 19.8436 / 19.8416 GiB on GPU0/GPU1; sampled peaks in both complete runs are 20.6855 / 20.4648 GiB. CPU embedding, disk PLE, GPU-resident experts and zero expert-cache budget match the preceding section.

All 16 gate/output tensors from real expert snapshots are bitwise identical to the original implementation. Tests cover grouped experts, 320-column and shorter K tails, TP shards, GPU-assisted prefill of CPU experts, the generic cache and CUDA Graph. An independent CPU Q8 reference validates the new Dense path for FP32/FP16/BF16, 1/3/8 rows and partial output blocks. Dense activation quantization changes numerical results and generated tokens: three runs with the same chunk agree, but the before/after outputs first differ at token 17 / 250 for chunk 32 / 512. The approximately 1% decode difference therefore includes changed routing; use isolated replay to assess individual kernel gains. The 4096-token prefill first token, arithmetic answer and JSON result are unchanged. These checks do not replace a full model quality evaluation.

[Per-run comparison, numerical validation and profiler results](../benchmarks/qwen38_flash_next_q2_0_2080ti_20261001.json). The optimized isolated native library has SHA256 `cdacb884eef2ec3ce0b70386a69f4ec6211c6cd4e42de091e15af9eab8ec38ee`.

### GGUF Q2_0: Q3_K Dense decode optimization (2026-10-02)

The Q2_0 model also uses Q3_K and other formats for Dense weights. This change reduces repeated sub-scale loads and unpacking instructions in Q3_K. Large single-token projections process eight output rows per CUDA block with shared Q8 activations. Smaller projections and long K retain four warps per row to avoid regressions found during shape sweeps. Dispatch uses format, dimensions and alignment, without model-specific checks, additional environment variables, weight conversions or persistent workspace. Floating-point accumulation order is preserved.

The preceding optimized build was rerun as the baseline on the same two 22 GiB RTX 2080 Ti GPUs, with TP2, CUDA Graph enabled, MTP disabled and all experts resident on GPU. Each group has a warmup and three formal runs; values below are medians:

| Test | Chunk | Previous token/s | Current token/s | Change |
| --- | ---: | ---: | ---: | ---: |
| Decode, 512 input / 512 output | 32 | 70.69 | **71.61** | +1.30% |
| Decode, 512 input / 512 output | 512 | 69.51 | **70.63** | +1.61% |
| 4096 prefill, 1 output | 1024 | 1355.21 | **1358.02** | +0.21% |

All output tokens matched the baseline across 14 requests, including warmups, formal decode/prefill runs, arithmetic and JSON checks. The small prefill difference is treated as measurement variation.

In Nsight Systems, GPU0's 53 Q3_K Dense calls per decode step decreased from 0.958 to 0.787 ms in total. Cold-cache NCU replay of a 5120×2560 Q3_K projection decreased from 23.136 to 19.744 μs; DRAM utilization increased from 48.73% to 56.99%, ALU activity decreased from 63.53% to 35.36%, and registers per thread fell from 66 to 45. Replay uses actual weights and synthetic nonzero input, with unlocked clocks. End-to-end throughput is measured separately without profiling. Other Dense operations, HC projections and cross-GPU reduction still account for much of decode time, limiting the end-to-end gain to roughly 1%–2%.

FP32/FP16/BF16 CPU-reference checks, bitwise comparisons against the original four-warp kernel, partial output rows, dimension boundaries and CUDA Graph replay passed. All 72 public Linear cases matched bitwise. Q2 regression tests and Compute Sanitizer memcheck/synccheck passed, with zero sanitizer errors. Memory use was unchanged: 18.6634 GiB of logical GPU weights per rank, 19.8436 / 19.8416 GiB used after loading and warmup, and sampled peaks of 20.6855 / 20.4648 GiB on GPU0/GPU1.

[Full measurements and validation](../benchmarks/qwen38_flash_next_q3_decode_2080ti_20261002.json). Isolated native build SHA256: `8c3c12485b0031d127aa2353b4e3cb40d0f99d97a0cd7a05f5900af8a67d74db`; installed libraries were preserved. Performance and sanitizer coverage currently target SM75.

### GGUF Q2/Q3 cleanup and regression checks (2026-10-02)

The cleanup shares Q8 input quantization between Dense and grouped experts, Q2 eight-value unpacking, and the Q3 K-accumulation code used by both launch geometries. It retains the measured shape dispatch and fallbacks, removes a duplicate include and unreachable IQ1 initialization in fused gate/up, and preserves weight layouts, arithmetic precision and persistent workspace sizes.

Admission now checks columns against the weight format's block size. Fused gate/up accepts only its implemented IQ2_XXS / IQ2_XS / IQ2_S formats. Unsupported formats and partial weight blocks return false before allocation or device-buffer access, allowing the caller to select its fallback. Boundary regression checks pass, and all 36 supported IQ2 fused-call outputs match the preceding build bitwise.

An audit of 17 GGUF source files found no remaining runtime experiment switches, benchmark hooks or hardcoded model paths. Cache budgets, MTP, TP and Graph controls remain valid configuration options; no switch was added. The cache test entry also now returns explicitly on success, supporting dynamic-loading tests that rename main.

All 12 operator/cache/TP-shard/Graph/memory/synchronization checks pass. All 16 captured expert intermediates, 72 Dense outputs and generated tokens across 14 model requests match the preceding build. Under the same configuration, three-run medians are **71.61 → 71.59 token/s** for chunk-32 decode, **70.63 → 70.67 token/s** for chunk-512 decode, and **1358.02 → 1359.00 token/s** for 4096 prefill. Clocks were unlocked; small differences are not claimed as new optimization gains. Memory use after loading and warmup is unchanged.

[Cleanup record and full regression results](../benchmarks/qwen38_flash_next_gguf_cleanup_2080ti_20261002.json). Isolated build SHA256: `2c60b9f0661321f32447f8d3bd1de75998e6fd9029eda3a95a79aad210241f91`. Runtime validation targets SM75; installed libraries were preserved.
