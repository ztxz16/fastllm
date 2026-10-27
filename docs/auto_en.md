# Deploying and tuning with `--auto`

[中文](auto.md) · [Documentation index](README_EN.md)

Start with this command for supported models:

```sh
ftllm server /path/to/model --auto
```

The runtime planner currently targets Linux with NVIDIA CUDA. It reads model metadata, visible GPUs, available VRAM, and host memory to choose device placement, expert caches, context capacity, concurrency, threads, and prefill chunks. It estimates capacity and latency without repeatedly loading and benchmarking the model. See [planner details (Chinese)](hybrid-auto.md) for model defaults, or [`--autotune` (Chinese)](autotune.md) for measured DeepSeek V4.1 tuning.

## Choose the deployment entry point

| Model | Deployment |
| --- | --- |
| Dense Qwen3 | Supports `--auto`; no MoE expert cache |
| Dense HF models with the Qwen3.5 architecture, including Qwen3.8-27B-FP8 | Prefer `--auto`; other quantization formats need separate validation. See the [Qwen guide](qwen3_en.md) |
| Qwen4 / Qwen3.8-Flash-Next | Prefer `--auto`; see the [Flash-Next guide](qwen3.8-flash-next/README_EN.md) |
| GLM-5.3-Flash | Prefer `--auto`; see the [GLM guide](glm5_en.md) |
| DeepSeek-V4.1 | Prefer `--auto`; see the [V4.1 guide](deepseek_v41_en.md) |
| DeepSeek-V4, Kimi-K3, Dots3-Note, Laguna, GLM-5/5.2 | Not integrated with the runtime planner; use their manual deployment guides |

Inference support does not imply `--auto` support. CPU-only, Windows, and ROCm deployments use the corresponding manual configuration.

## Override rules

- **Only set options you need to change.** Explicit CLI options take precedence. Values in a JSON configuration also count as explicit; CLI values override JSON. Avoid copying all defaults from a manual configuration.
- `0`, `false`, and `off` can be explicit disable instructions, rather than omitted values. The tables below identify those options.
- Explicit context, batch, chunk, and cache sizes must fit the budget. If no plan fits, startup fails instead of silently reducing your requested sizes.
- Memory limits are **planning budgets**, not process or OS memory caps. `g` and `m` mean GiB and MiB. Actual allocations still depend on the model, request shape, and allocator.
- Inherited environment variables remain effective. `CUDA_VISIBLE_DEVICES` controls visible GPUs. Native `FT_THREADS` and `FASTLLM_NUMA_THREADS` precedence can override CLI threads, and `FASTLLM_TP` can pin the topology.

## Common options and their effects

| Option and example | Effect |
| --- | --- |
| `--max_context_length 32768` | Per-session input plus output limit. A smaller window can free KV space for expert caches or weights, but limits long requests. A larger window needs memory and model/RoPE support. This is not the output length for each request. |
| `--max_batch 2` | Maximum concurrent requests in a decoding batch. May improve total throughput, but increases KV/workspace needs and can reduce expert cache capacity. Single-request latency may increase. |
| `--tokens 65536` | Shared total token capacity across sessions. Automatic mode otherwise budgets context × concurrency. Reducing this limits the combined session lengths; it does not eliminate their actual KV requirements. |
| `--auto_gpu_memory_limit 20g` | Planning VRAM limit for each visible GPU. Can reduce cache, chunks, or resident weights. Does not simulate GPU architecture, compute, or bandwidth. |
| `--auto_host_memory_limit 120g` | Host-memory limit for the whole plan. May reduce resident experts or disk-backend memory caches and require disk experts. Does not limit OS page cache or other processes. |
| `--force_ddr_limit 120g` | Set the final host-memory planning budget to 120 GiB, replacing the budget after automatic detection and reserves. Mutually exclusive with `--auto_host_memory_limit`; does not impose an OS memory limit. |
| `--gpu_mem_ratio 0.9` | GPU memory budget ratio, used together with the VRAM limit. Lower values leave more headroom; higher values may allow larger caches but leave less room for runtime allocations. DeepSeek V4.1 automatic hybrid mode currently defaults to 0.95; other cases usually use 0.9. |
| `--threads 32` | CPU/NUMA computation threads. Tune for physical cores, memory bandwidth, and NUMA topology. More threads can increase contention. This does not set CPU affinity. |
| `--chunked_prefill_size 4096` | Input tokens per prefill chunk. Larger chunks often improve long-input throughput but need more workspace and may reduce expert cache capacity or increase waiting time. Smaller chunks save space but can reduce prefill throughput. |
| `--moe_cuda_cache 7g` | Per-GPU expert cache for hybrid inference; `0` disables it. Larger caches may reduce transfers and CPU work, depending on hit rate. Ordinary weights, KV, and workspace are separate. An explicit cache size also constrains placement; for example, V4.1 no longer searches the automatic NUMA/disk boundary. |
| `--kv_cache_dtype fp4` | KV storage type, such as FP16/BF16/FP8/FP4, subject to model/backend support. Lower precision can save memory but is not universally faster or lossless. DeepSeek V4.1 has a specific lossless packing path for its native fake-quantized KV; this does not generalize to other models. |
| `--prefix_cache true` | Reuse identical prefixes across requests, mainly speeding up repeated prefill. Different prefixes or invalidated caches do not get the same benefit. `false` disables prefix caching, but does not disable all history reuse. |

Usually omit the KV type to preserve model-specific defaults. Explicit `--kv_cache_dtype auto` overrides the planner's selected storage type; it is not equivalent to leaving the option unset.

## Speculative decoding and fast prefill

| Option and example | Effect and scope |
| --- | --- |
| `--speculative_algorithm off` | Disable all speculative decoding. Useful for a baseline or low-acceptance workloads; works even when a draft path remains configured. |
| `--dspark 5` | Enable embedded DSpark with the specified draft token count; `0` disables it. Requires matching model and draft weights. More drafts add draft/verification work and help only when enough tokens are accepted. |
| `--speculative_dspark_confidence_threshold 0.5` | DSpark draft confidence threshold in `[0,1]`. Higher values reduce low-confidence drafts; lower values may increase verification work. `0` disables the threshold, **not target-model verification**. |
| `--mtp 3` | Draft token count for supported MTP models, currently `0–8`; `0` disables it. Cannot be enabled together with DSpark. |
| `--mtp_min_p 0.5` | Greedy MTP probability threshold for supported Qwen4/GLM-5.3 paths; `0` disables it. Higher values reduce low-confidence drafts but do not guarantee better speed. |
| `--draft /path/to/draft` | Load a supported external draft model. Algorithm, format, and target must match. Adds weights and workspace; arbitrary files cannot supply missing DSpark/MTP support. |
| `--fast_prefill` / `--no_fast_prefill` | Enable / disable DeepSeek V4.1 fast prefill. Its `--auto` mode enables it by default. Remove both options to restore automatic selection; also remove any explicit `fast_prefill` field from JSON configuration. |

DeepSeek V4.1 normally enables DSpark 5 automatically, but GGUF files without embedded draft weights do not. Supported dense HF Qwen3.5-architecture checkpoints with embedded MTP weights can enable MTP automatically. Qwen4 and GLM MTP need an explicit setting or supported draft model. Check the startup log. Explicit MTP/DSpark values, including `0`, override the default speculative policy.

## Devices, disk, and expert placement

These options constrain the planner's search more directly. Try context, concurrency, and memory budgets first.

| Option and example | Effect |
| --- | --- |
| `CUDA_VISIBLE_DEVICES=0,1` | Expose only selected GPUs; process-local IDs start at 0 again. `--auto` still selects a plan using the visible devices. |
| `--device cudapp=0,1 --tp off` | Request two-GPU PP; `--tp 2` requests TP instead. Communication and concurrency behavior differ. V4.1 currently searches single-GPU/PP automatically; TP requires an explicit request. |
| `--moe_device numa` / `cpu` / `disk` | Fix the host expert backend, potentially excluding other feasible plans. Supported CPU/NUMA hybrid paths share computation, caching, and dynamic routing; each format still needs backend support. |
| `--moe_device_layers -1` | Put all expert layers on the specified backend. A positive value selects that many final model layers; it is not the GPU-resident layer count. Constrains the search for a resident GPU prefix. |
| `--moe_cpu_cache 64g` | Total host expert cache budget for the disk backend; `0` disables it. This does not expand resident CPU/NUMA expert capacity. Smaller caches depend more on disk during cold requests and misses. |
| `--auto_disk_cache_ratio 0.5` | Recommended minimum host-cache coverage of remaining disk experts when V4.1 chooses a NUMA/disk boundary. Range `[0,1]`: larger values favor more disk cache, smaller values allow more NUMA layers. `0` removes the target; `1` does not prohibit disk. If the target cannot fit, the planner keeps its disk fallback and explains why. This is not a GPU cache ratio or measured hit rate. |
| `--ngram_device cpu` / `disk` | Keep PLE/Engram tables resident or read them on demand, independently of the expert backend. `moe_device=numa` with `ngram_device=disk` does not mean experts use disk. |
| `--prefix_cache false --cache_history false` | Disable both prefix and history reuse to permit supported Qwen4 plans to consider exact KV streaming. Does not force streaming and is not a general memory-saving switch for other models. |

When GPU and host memory can hold the experts, the planner prioritizes avoiding disk experts. Model loading, PLE/Engram reads, and OS swapping are separate. Advanced `--moe_cache_*` policies are covered in the [expert cache guide (Chinese)](cuda-expert-cache.md); usually keep their defaults.

## Example commands

```sh
# Leave resources for other applications
ftllm server /path/to/model --auto \
  --auto_gpu_memory_limit 20g --auto_host_memory_limit 120g

# 32K per-session window, at most 2 concurrent requests
ftllm server /path/to/model --auto --max_context_length 32768 --max_batch 2

# Single-GPU ordinary decoding
CUDA_VISIBLE_DEVICES=0 ftllm server /path/to/model --auto --speculative_algorithm off

# V4.1 with embedded DSpark weights: 5 drafts, fast prefill disabled
ftllm server /path/to/deepseek --auto --dspark 5 --no_fast_prefill
```

API request fields such as `max_tokens`, `temperature`, and `top_p` control a generation, not the capacity budget. For comparisons, fix the prompt, output length, and concurrency. Distinguish cold and repeated requests, prefill, single-request decode, and total concurrent throughput. Different generated tokens and expert routes can produce small timing changes.

## Launcher automatic mode

Run `ftllm`, add a profile, select the model file or folder at the top, then fill in Basic settings and choose **Automatic configuration**. Resource limits, Vision module, and Model parameters are collapsed by default. Leave overrides blank for fully automatic planning:

- **Resource limits (optional):** GPU memory ratio maps to `gpu_mem_ratio`; host memory limit maps to `force_ddr_limit`, such as `120g`, for planning only.
- **Vision module (optional):** expand to choose a vision module file compatible with the GGUF main model; passed as `--mmproj`. Leave blank to skip the external vision module.
- **Model parameters:** collapsed by default, with Prefill Chunk (`chunked_prefill_size`), Maximum concurrency (`max_batch`), Custom arguments, and Custom environment variables.

Leave Chunk and Maximum concurrency blank for automatic planning. Enter other options under Custom arguments, such as `--dspark 5`; they are appended to the command, and values containing spaces must be quoted. Enter environment variables as `KEY=VALUE`, separated by spaces or newlines. Both custom fields apply in automatic and custom modes.

For example, enter GPU memory ratio `0.9` and host memory limit `120g` under Resource limits, then Chunk `4096` and Maximum concurrency `2` under Model parameters:

```sh
ftllm server /path/to/model --auto --gpu_mem_ratio 0.9 \
  --force_ddr_limit 120g --chunked_prefill_size 4096 --max_batch 2
```

Larger chunks may improve prefill throughput but use more workspace. Higher concurrency may improve total throughput but needs more cache capacity. For other options, enter flags such as `--fast_prefill` or `--no_fast_prefill` under Custom arguments; remove them to restore the default.

Legacy automatic profiles keep absolute GPU/host limits as `--auto_gpu_memory_limit` and `--auto_host_memory_limit` in Custom arguments, preserving their original meaning. They are not converted to a GPU ratio or a forced host budget.

Automatic and custom modes store Chunk, Maximum concurrency, and GPU memory ratio separately so hidden values do not unexpectedly override automatic planning. Settings are saved with the profile; inspect the command preview and restart the model service to apply them.

## Read the final plan

The startup `[auto]` JSON contains more detail than the command preview:

| Fields | Meaning |
| --- | --- |
| `device`, `tp`, `mode` | Selected GPUs and parallel layout |
| `max_context_length`, `total_tokens`, `max_batch` | Session window, shared capacity, and concurrency |
| `chunked_prefill_size`, `threads`, `kv_cache_dtype` | Input chunks, thread count, and KV storage |
| `moe_device`, `resident_layers`, `moe_device_layers` | Expert backend and resident-layer boundary |
| `moe_cuda_cache`, `moe_cuda_cache_per_device`, `moe_cpu_cache` | Cache budgets; interpretation depends on the backend and mixed NUMA/disk placement. See [planner details (Chinese)](hybrid-auto.md) |
| `gpu_budget_bytes` / `gpu_estimated_bytes`, `host_budget_bytes` / `host_estimated_bytes` | Budgets and estimates, not measured runtime peaks |
| `automatic.dspark`, `automatic.mtp`, `automatic.fast_prefill` | Final speculative decoding and fast prefill settings |

`automatic.cuda_graph=true` allows Graph use; runtime checks can still select eager execution for a backend, layout, or request shape. An absent model-specific field does not mean that feature is universally enabled.
