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
