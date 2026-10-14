# Qwen4-Exp / Qwen3.8-Flash-Next Benchmark

[中文](qwen4_exp.md) · [Benchmark index](../benchmark_en.md) · [Deployment guide](../qwen4_exp.md)

## 2026-09-29: Flash-Next NVFP4 on two RTX 2080 Ti GPUs

All six configurations use inference code from commit `d6ef5c784`. See the [deployment commands](../qwen4.md).

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


## GPU-resident GGUF experts

Hardware: two 22 GiB RTX 2080 Ti GPUs and an EPYC 7452, one NUMA node. TP2, FP16 activations, CUDA Graph enabled, MTP off, expert cache budget 0; native GGUF experts on GPU, embedding on CPU, PLE on disk, 28 threads, 1 GiB KV budget, GPU memory ratio 0.95. See the [launch configuration](../qwen3.8-flash-next/README_EN.md#gguf-resident-performance).

Native generation timings, batch 1, greedy, no HTTP or profiler; median of three measured runs after warmup. Decode: 512 input and 512 output tokens, `511 / (last arrival - first arrival)`. Prefill: 4096 input tokens and one output, `4096 / TTFT`.

| Weights | Date | Decode, chunk 32 | Decode, chunk 512 | Prefill, chunk 1024 |
| --- | --- | ---: | ---: | ---: |
| GSQ-RCO-IQ2_XS | 2026-10-01 | 70.87 | 69.77 | 1205.13 |
| GSQ-RCO-Q2_0 (Q3_K Dense included) | 2026-10-02 | 71.59 | 70.67 | 1359.00 |

Rates are token/s. These separate measurements are not a controlled comparison of quantization formats. Chunk size may change generated trajectories.

| Weights | Logical weights GiB/GPU | GPU0/1 after load and warmup GiB | GPU0/1 sampled peak GiB |
| --- | ---: | --- | --- |
| IQ2_XS | 19.3554 | 20.2088 / 20.2049 | 21.0586 / 20.8379 |
| Q2_0 | 18.6634 | 19.8436 / 19.8416 | 20.6855 / 20.4648 |

Two-second GPU sampling may miss brief peaks. IQ2_XS host HWM was 41.905 GiB with zero swap; CPU embedding used 2.368 GiB. Arithmetic, JSON and 4096-token prefill checks passed; these are not a full quality evaluation. Q8 activation dot products have quantization error and need not match floating-point activation computation bit for bit.

Measured native-library SHA256: IQ2_XS `0e9aa1b66b2d502e150395a40bfca36f2b43c0ea937329f43f89d60861add82b`; Q2_0 `2c60b9f0661321f32447f8d3bd1de75998e6fd9029eda3a95a79aad210241f91`. Results apply to these builds on SM75.
