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

Replace the example filename with the actual path. Both the earlier four-shard backbone plus separate Engram layout and the upstream llama.cpp six-shard layout are detected automatically. Existing models do not need conversion or a format flag.

With `--ori`, startup defaults still follow the actual GGUF weights. Draft layers listed in the original HF configuration do not enable DSpark when those weights are absent from the GGUF.

For the old layout, keep the Engram filename and source consistent with the backbone metadata. For ISTA's six-shard layout, keep shards 1–5 and `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` in the same directory. The shared file is shard 6; its original name or a `...-00006-of-00006.gguf` link is accepted. The old and new Engram headers differ, so retain matching files for each layout. Quantizations using the same layout can share a matching Engram file. HF models need their own Engram weights and configuration. `--ngram_device disk` controls on-demand table reads independently of the expert backend.

GGUF Engram supports packed `IQ4_NL`, `Q5_K`, `Q6_K`, and `MXFP8` rows. Loading checks the registered row decoder and block alignment instead of requiring IQ4_NL. CPU-resident and `--ngram_device disk` tables decode only selected rows and keep the full table compressed, without requiring a particular CUDA SM or CPU instruction set. Keep every Unsloth shard, including the shard containing Engram; it is not always the final shard.

For [Unsloth Dynamic V3.1](https://unsloth.ai/docs/models/deepseek-v4.1-flash), IQ1_S/IQ1_M use IQ4_NL Engram; IQ2_XXS/Q2_K_XL use Q5_K; IQ3_XXS/IQ3_S use Q6_K; Q3_K_XL/Q4_K_XL/Q8_K_XL use MXFP8. The MXFP8 type ID and wire layout follow [llama.cpp PR #30817](https://github.com/ggml-org/llama.cpp/pull/30817) (`b72050d03ae14bb3cb5138ecefab3f3f65157d31`): type 43, with one E8M0 scale and 32 E4M3FN values per 33-byte block. This adds lookup/row decoding support, not an MXFP8 matrix multiplication kernel.

Mixed GGUF/MXFP4 experts support CPU/NUMA hybrid inference, GPU expert caching, and full CUDA graphs. Storage backends receive shares of one GPU cache budget; pure MXFP4 layers keep their native FP4 kernels. Mixed layers only reorder quantized bytes on cache admission, without expanding floating-point weights or requantizing. No additional switch is required.

Automatic defaults are one request, up to a 1M-token native window, FP4 KV, and fast prefill. Embedded DSpark weights enable DSpark 5 by default; GGUF files without those weights do not. Check `[auto]` for the selected devices, chunks, and caches. The planner prioritizes avoiding disk experts when GPU and host memory can hold them; Engram may still use disk independently.

`--auto` enables `--kernel-tune quick` by default. During the first warmup for an operator shape, it validates and selects legal registered candidates, including ordinary/grouped projections and expert computation. This applies to single-request and concurrent inference without forcing Triton at a fixed row count. Unsupported architectures or formats retain their existing implementation. Use `--kernel-tune off` to disable tuning; see [operator registration and warmup selection](kernel-policy.md).

Startup reports `SM120 MXFP4 grouped MoE enabled automatically`. With one visible GPU and sufficient space within the existing expert-cache budget, the cache policy prioritizes full draft residency. Unsupported inputs or compilation/loading failures fall back to built-in CUDA. Set `FASTLLM_CUDA_TRITON_MXFP4_GROUPED=0` to disable this operator, or `FASTLLM_CUDA_TRITON=0` to disable Triton automatic selection. Explicit environment settings take precedence.

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
| Require the complete DSpark expert cache on GPU | `--draft-gpu-resident 1`; `0` disables the full-residency preference; omit to retain existing policy |
| DSpark confidence threshold | `--speculative_dspark_confidence_threshold 0.5`; `0` disables the threshold |
| Enable / disable fast prefill | `--fast_prefill` / `--no_fast_prefill` |

`--force_ddr_limit` replaces the final detected host-memory budget, including automatic reserves. Do not combine it with `--auto_host_memory_limit`; it is a planning setting, not an OS memory limit.

Remove explicit settings to restore automatic selection. Resource limits constrain the plan, not actual process allocations, GPU architecture, or compute performance. See the [`--auto` guide](auto_en.md) for detailed effects and Launcher controls.

## Explicit draft GPU residency

For a GGUF backbone and matching MXFP4 DSpark experts:

```sh
ftllm server /path/to/model-IQ2_XXS-00001-of-00004.gguf --auto \
  --draft /path/to/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf \
  --draft-gpu-resident 1
```

The option supports DeepSeek-V4.1 MXFP4 drafts on a single CUDA device, PP, or TP; it does not require Triton. Auto reserves the entire draft expert cache within the existing GPU budget: the final device for PP, or the first TP device. Other devices retain their target-model cache budget; the complete draft is not replicated on every TP device. The draft's existing TP attention/head path remains available. Auto may choose a smaller prefill chunk or reduce the target model's cache. CPU/NUMA expert storage remains allocated. Already GPU-resident draft weights are not cached twice. Add `--tp 2` to the command above for two-device TP.

TP CUDA ordinals must be ascending: reversed TP has not passed numerical validation and this option rejects it before loading weights. To choose a different physical root, reorder `CUDA_VISIBLE_DEVICES` and use `--tp 2`.

Forced-resident hybrid TP supports custom all-reduce together with target CUDA Graph replay. `FASTLLM_CUDA_CUSTOM_ALLREDUCE=0` selects NCCL. Full draft residency consumes some of the target expert cache budget; performance depends on available VRAM, cache hit rates, and draft acceptance, so it is not guaranteed to improve speed.

CPU/NUMA expert caches and already GPU-resident experts are supported. Disk expert caches (`--moe_device disk`) are not supported by this option; Engram may still use disk independently.

Omitting the option preserves the existing policy. `--draft-gpu-resident 0` uses proportional cache sharing instead of prioritizing full residency; it does not force CPU-only draft execution. The underscore alias `--draft_gpu_resident` also works. Manual configurations need sufficient `--moe_cuda_cache`. Insufficient budget, allocation failure, and unsupported configurations produce an error rather than silently accepting partial residency.

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
