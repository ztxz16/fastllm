# DeepSeek-V4.1-Flash deployment

[中文与详细实现说明](deepseek_v41.md) · [Project home](../README_EN.md) · [`--auto` tuning](auto_en.md)

This guide covers DeepSeek-V4.1-Flash (`deepseek_v41`), including supported HF checkpoints and GSQ-RCO GGUF files. DeepSeek-V4 uses a different deployment path; see its [separate guide](deepseek_en.md).

## Automatic configuration (recommended)

On Linux with NVIDIA CUDA, pass a complete HF model directory:

```sh
ftllm server /path/to/DeepSeek-V4.1-Flash-1017 --auto \
  --model_name deepseek-v4.1-flash --host 0.0.0.0 --port 8080
```

For GGUF, pass the model file or first shard. Supported files contain configuration and tokenizer metadata and do not require `--ori`:

```sh
ftllm server /path/to/model-00001-of-00002.gguf --auto \
  --model_name deepseek-v4.1-flash --host 0.0.0.0 --port 8080
```

Replace the example filename with the actual path. Keep the GGUF-referenced Engram file beside the backbone, with the filename and source matching its metadata. Multiple quantizations can link to one matching Engram file. HF models need their own Engram weights and configuration. `--ngram_device disk` controls on-demand table reads independently of the expert backend.

Automatic defaults are one request, up to a 1M-token native window, FP4 KV, and fast prefill. Embedded DSpark weights enable DSpark 5 by default; GGUF files without those weights do not. Check `[auto]` for the selected devices, chunks, and caches. The planner prioritizes avoiding disk experts when GPU and host memory can hold them; Engram may still use disk independently.

## Common adjustments

Append only the options you need:

| Adjustment | Options |
| --- | --- |
| 32K session window, one request | `--max_context_length 32768 --max_batch 1` |
| Per-GPU VRAM and total host-memory budgets | `--auto_gpu_memory_limit 22g --auto_host_memory_limit 120g` |
| GPU memory ratio and final host-memory budget (Launcher Resource limits) | `--gpu_mem_ratio 0.9 --force_ddr_limit 120g` |
| Per-GPU expert cache | `--moe_cuda_cache 7g`; `0` disables it |
| Prefill chunks and CPU threads | `--chunked_prefill_size 4096 --threads 32` |
| Disable speculative decoding | `--speculative_algorithm off` |
| Use 5 embedded DSpark drafts | `--dspark 5`; requires matching draft weights |
| DSpark confidence threshold | `--speculative_dspark_confidence_threshold 0.5`; `0` disables the threshold |
| Enable / disable fast prefill | `--fast_prefill` / `--no_fast_prefill` |

`--force_ddr_limit` replaces the final detected host-memory budget, including automatic reserves. Do not combine it with `--auto_host_memory_limit`; it is a planning setting, not an OS memory limit.

Remove explicit settings to restore automatic selection. Resource limits constrain the plan, not actual process allocations, GPU architecture, or compute performance. See the [`--auto` guide](auto_en.md) for detailed effects and Launcher controls.

## Fast prefill

```sh
# Automatic mode enables it by default; explicitly disable it
ftllm server /path/to/DeepSeek-V4.1-Flash --auto --no_fast_prefill

# Explicitly enable it
ftllm server /path/to/DeepSeek-V4.1-Flash --auto --fast_prefill
```

Manual mode leaves it off unless `--fast_prefill` is supplied. Hyphenated aliases `--fast-prefill` and `--no-fast-prefill` also work. To restore automatic selection, remove those options and any explicit `fast_prefill` JSON field. In Launcher, expand **Model parameters** and enter `--fast_prefill` or `--no_fast_prefill` under **Custom arguments**, then save and restart. Remove the flag to restore the default.

## External vision module

For a compatible DeepSeek V4.1 GGUF main model, select the matching visual module in Launcher's collapsed **Vision module (optional)** section, or pass it on the command line:

```sh
ftllm server /path/to/model-00001-of-00002.gguf --auto \
  --mmproj /path/to/mmproj-BF16.gguf
```

The module must match the main model version. See the [detailed vision requirements](deepseek_v41.md) for checkpoints that need vision routing biases in the module.

## Manual deployment and advanced features

For a fixed single-GPU + NUMA setup:

```sh
ftllm server /path/to/DeepSeek-V4.1-Flash \
  --device cuda --moe_device numa --threads 32 --chunked_prefill_size 4096
```

Thread and chunk values are examples. Manual mode does not inherit the `--auto` defaults. KV type and cache sizes still need to fit the actual checkpoint and hardware.

CPU Engram tables automatically prefetch lookup rows and their BF16 conversion for forwards of at least 128 tokens, including native FP8 and GGUF tables stored in memory or on disk. Each forward owns one history snapshot and at most one pending result, limited to the layers its pipeline stage consumes. Short forwards and GPU-resident tables use the synchronous path. Set `FASTLLM_DSV41_ENGRAM_PREFETCH=0` to disable this optimization.

The [detailed Chinese guide](deepseek_v41.md) documents manual layouts, Engram metadata, TP/PP, prefix reuse, DSpark, image input, and numerical validation. Optional image and draft features require matching weights; do not assume a text GGUF contains them. Performance records are kept in the [V4.1 benchmark page (Chinese)](benchmarks/deepseek_v41.md).
