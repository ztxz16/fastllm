# Manual deployment and option reference

[中文](deployment.md) · [Documentation index](README_EN.md)

For everyday deployment, start with `ftllm server /path/to/model --auto` for supported models. See the [`--auto` tuning guide](auto_en.md) for common overrides. This page retains manual device layouts, model-specific options, download/export commands, and source-build instructions.

- [Deployment patterns](#deployment-patterns)
- [Option reference](#common-options)
- [Model formats, download, and export](#model-formats-download-and-export)
- [Build from source](#build-from-source)

## Current model focus

The table summarizes model families and limitations; use the corresponding model guide for deployment settings.

| Family | Current focus |
| --- | --- |
| Qwen | Qwen4-Exp / Qwen3.8-Flash-Next text decoding with QSA, PLE n-grams, and CPU/CUDA/NUMA hybrid execution; MTP for Qwen3.8-Flash-Next; MTP and DFlash2 for Qwen3.5/3.6/3.8 |
| DeepSeek | DeepSeek-V4 / V4-Flash, sparse attention, embedded DSpark, multi-GPU CUDA, and CPU/NUMA hybrid MoE |
| Kimi | Kimi-K3, KDA/MLA, external DSpark, CUDA and NUMA execution, CPU/GPU experts, and disk experts |
| GLM | GLM-5 DSA, GLM-5.3-Flash KDA and paged caching, and quantized KV-B CPU inference for GLM-5.2 |
| Others | Dots3-Note, Laguna, HY-V3, Step3.5/3.7, MiniMax-M2, Gemma4, and more |

Qwen4-Exp / Qwen3.8-Flash-Next currently does not load vision weights. Qwen3.8-Flash-Next can load MTP weights on demand and enable speculative decoding with `--mtp`. See the [legacy model compatibility list](models.md) for earlier architectures and the [changelog](version_en.md) for the latest additions and limitations.

## Deployment patterns

### Multi-GPU tensor parallelism

`--tp 0,1` explicitly selects GPUs 0 and 1, `--tp 2` selects the first two visible GPUs, and `--tp auto` uses detected GPUs automatically.

~~~bash
ftllm server /data/models/my-model \
  --device cuda --tp 0,1 \
  --max_batch 16 --gpu_mem_ratio 0.9
~~~

### GPU + NUMA hybrid MoE

~~~bash
ftllm server /data/models/my-moe-model \
  --device cuda --moe_device numa \
  --chunked_prefill_size 8192
~~~

When host memory is limited, a small fraction of expert layers can be placed on disk, for example `--moe_device "{'cuda':1,'numa':8,'disk':1}"`. Disk execution depends heavily on SSD random-read performance. See the [hybrid inference guide](mixforward.md).

### Long context and prefix caching

`--max_context_length` (alias `--max-context-length`) sets the combined input and output limit per session. Extending the model-declared window also requires a valid `--rope_scaling` (alias `--rope-scaling`), which accepts `yarn` or JSON. You can omit the RoPE option when reducing the window. These options take effect at model loading without editing the checkpoint's `config.json`.

For example, extend Qwen3-0.6B to 65536 tokens and enable prefix caching. Its original YaRN length is 32768 and must be specified explicitly; the declared 40960 is a different value:

~~~bash
ftllm server /data/models/Qwen3-0.6B \
  --device cuda --max_batch 1 --tokens 65536 \
  --max_context_length 65536 \
  --rope_scaling '{"rope_type":"yarn","factor":2,"original_max_position_embeddings":32768}' \
  --chunked_prefill_size 8192 \
  --prefix_cache true
~~~

Qwen3.8-27B-FP8 supports the `yarn` shorthand because its original RoPE length is known. This example requests a 1,000,000-token window with two GPUs and FP4 KV cache, resolving to original=262144 and factor=4:

~~~bash
ftllm server /data/models/Qwen3.8-27B-FP8 \
  --tp 2 --kv_cache_dtype fp4 \
  --max_context_length 1000000 --rope_scaling yarn
~~~

`--tokens` sets the KV pool capacity shared by all sessions; omitting it enables automatic budgeting. Startup fails if an explicit target exceeds the configured RoPE range or the KV capacity calibrated during warmup. `/v1/models` reports the effective window, original model limit, and requested target. The 1M example requires sufficient GPU memory: the tested configuration with two 24GB GPUs cannot fit it, and a full 1M input has not been validated.

Extension supports Qwen2, Qwen3, and Qwen3.5 layouts, including Qwen3.8 checkpoints based on Qwen3.5. Launcher's advanced "RoPE extension" field accepts the same configuration. HF directories and GGUF files accept explicit `--rope_scaling` options, subject to model layout support. Setting only a length in the CLI retains shrink-only behavior for GGUF, FLM, and custom GraphLLM; FLM and custom GraphLLM do not support RoPE extension. See the [context extension guide](context-length-extension.md) for parameters, supported layouts, and validation results.

### Speculative decoding

These features require a matching model architecture and checkpoint:

~~~bash
# Example: embedded MTP on Qwen3.5, with up to 8 draft tokens per step
ftllm server /data/models/qwen3.5 --mtp 4

# Attach a standalone MTP module to a Qwen3.5 GGUF without MTP weights
ftllm server /data/models/qwen3.5.gguf \
  --device cuda --cuda_embedding \
  --draft /data/models/qwen3.5-fp8/mtp.safetensors \
  --draft_tokens 5

# Embedded DeepSeek-V4 DSpark
ftllm server /data/models/deepseek-v4 --dspark 7

# Qwen3.8 with a separate DFlash2 draft checkpoint
ftllm server /data/models/qwen3.8 \
  --tp 2 \
  --draft /data/models/qwen3.8-dflash2 \
  --draft_tokens 7
~~~

`--draft` detects MTP, DFlash2, or DSpark from the draft checkpoint. Standalone MTP currently targets Qwen3.5 GGUF and may be either a directory containing `config.json` plus `mtp.safetensors`, or the `mtp.safetensors` file itself; it defaults to five draft tokens. For DFlash2, `--draft_tokens` counts actual draft tokens and excludes the anchor token.

See the [Qwen3.8 DFlash2 report](benchmarks/qwen38_27b_dflash2.md) for the complete configuration and validation results.

### Disk-backed Qwen4 PLE

The PLE table in Qwen4-Exp / Qwen3.8-Flash-Next is large. It can be read from the checkpoint on demand when host memory is constrained:

~~~bash
ftllm server /data/models/qwen4-exp \
  --device cuda --moe_device numa \
  --ngram_device disk
~~~

Disk mode reduces resident memory at the cost of random I/O; a fast SSD is recommended. See the [Flash-Next user guide](qwen3.8-flash-next/README_EN.md) for details.

## Common options

The CLI evolves continuously, so `ftllm <command> --help` is authoritative for the installed version. The tables below cover the options most often used for deployment.

### Model, device, and precision

| Option | Description |
| --- | --- |
| `model` / `-p, --path` | Hugging Face repository ID, local HF directory, FastLLM model, or configuration file |
| `--device` | Main compute device; common values are `cpu`, `cuda`, and `numa` |
| `--vision_device` | Qwen3.5 vision encoder device: `auto` (default), `cpu`, `cuda`, or `cuda:N`; with multiple CUDA TP devices, vision follows normal TP and `cuda:N` does not override placement; `cpu` keeps the tower in host RAM and saves VRAM at the cost of slower encoding |
| `--tp` | CUDA tensor-parallel devices; accepts `0,1`, `2`, or `auto` |
| `--moe_device` | MoE expert device: `cpu`, `cuda`, `numa`, `disk`, or a weighted combination |
| `--moe_device_layers` | Apply `--moe_device` only to the last N MoE layers; `-1` means all |
| `-t, --threads` | CPU/NUMA thread count; `-1` selects it automatically |
| `--dtype` | Weight type when loading HF weights; defaults to `auto` and should normally not override an already quantized checkpoint |
| `--moe_dtype` | Set the MoE weight type separately |
| `--atype` / `--moe_atype` | Activation types for regular and MoE layers |
| `--kv_cache_dtype` | KV-cache type: `auto`, `float16`, `bfloat16`, `fp8_e4m3`, or `fp4`; requires model and backend support |
| `--dtype_config` | Dynamic quantization configuration; see the [quantization guide](dtype_config.md) |
| `--triton` | Enable available Triton CUDA operators |

### Memory, context, and scheduling

| Option | Default | Description |
| --- | ---: | --- |
| `--gpu_mem_ratio` | `0.9` | Fraction of GPU memory available to weights and caches |
| `--auto_gpu_memory_limit` | Unset | Per-visible-GPU VRAM planning limit with `--auto`, e.g. `20g`; otherwise reads available VRAM |
| `--auto_host_memory_limit` | Unset | Total host-memory planning limit with `--auto`, e.g. `128g`; otherwise reads available host memory |
| `--force_ddr_limit` | Unset | Final host-memory planning budget, e.g. `120g`, replacing automatic detection and reserves. Mutually exclusive with `--auto_host_memory_limit`; does not impose an OS memory limit. See the [`--auto` guide](auto_en.md) |
| `--moe_cuda_cache` / `--moe-cuda-cache` | `0` | GPU expert-weight cache budget, e.g. `3g` (3 GiB); `0` disables it. Use with CUDA compute and CPU/NUMA/disk experts. KV cache, common weights, and workspaces are separate; see the [expert-cache guide](cuda-expert-cache.md) |
| `--moe_cpu_cache` / `--moe-cpu-cache` | `0` | Total host expert-cache budget for `--moe_device disk`, e.g. `32g`; can be combined with the GPU expert cache. See [disk-backed multilevel caching](cuda-expert-cache.md#disk-backed-multilevel-cache) |
| `--kv_cache_limit` | `auto` | Maximum KV-cache memory |
| `--tokens` | Automatic | Total token capacity used to size paged KV cache |
| `--page_size` | Backend-defined | Tokens per KV-cache page; the multi-GPU default is normally 16 |
| `--max_batch` | Automatic | Maximum number of requests processed together |
| `--max_context_length` / `--max-context-length` | Automatic | Combined input and output limit per session; explicit HF targets must pass RoPE and KV capacity checks |
| `--rope_scaling` / `--rope-scaling` | Model config | RoPE extension as `yarn` or JSON; available for adapted HF model layouts |
| `--chunked_prefill_size` | Disabled/model-defined | Prefill chunk size, for example `8192` |
| `--fast_prefill` / `--no_fast_prefill` | Off; V4.1 `--auto` enables it | Enable / disable DeepSeek V4.1 fast prefill; omit explicit settings to follow the default |
| `--prefix_cache` | Model/environment-defined | Enable or disable prefix caching with `true` or `false` |
| `--cuda_slab` | `0` | CUDA weight-slab size in MB; `0` disables it |
| `--ngram_device` | `cpu` | Store the Qwen4 PLE table on `cpu` or `disk` |

These are CLI defaults; `--auto` chooses model-specific values for options you leave unset. Planning limits do not impose runtime memory caps. See the [`--auto` guide](auto_en.md) for override behavior and cache-budget interpretation.

### Decoding, templates, and tool calling

| Option | Description |
| --- | --- |
| `--enable_thinking` | Control the model's thinking template when supported |
| `--mtp` | Draft tokens per step for models with MTP support; `0` disables it and the current maximum is 8 |
| `--mtp_fp8_draft_head` / `--mtp-fp8-draft-head` | FP8 draft output head for Qwen3.5-family multi-GPU MTP: `1` enables it; `0` reuses the original head to save VRAM, potentially reducing generation speed. Does not change the MTP draft count. An explicit value overrides `FASTLLM_MTP_FP8_DRAFT_HEAD`; otherwise the environment is preserved, with the head enabled by default |
| `--dspark` | Enable embedded DSpark and set draft tokens per step |
| `--draft` / `--draft_model_path` | External MTP, DSpark, or DFlash checkpoint; MTP may point directly to `mtp.safetensors` |
| `--draft_tokens` | Maximum draft tokens per step; defaults to the draft configuration |
| `--tool_call_parser` | Tool-call parser; defaults to `auto` |
| `--chat_template` | Custom Jinja chat-template file |
| `--cache_dir` | Local cache directory for online models |
| `--ori` | Original model configuration and tokenizer directory for selected GGUF models |
| `--mmproj` | Matching vision-module GGUF for Qwen3.5-family or DeepSeek V4.1 GGUF models; see [Qwen GGUF multimodal deployment](qwen3_en.md#gguf-multimodal) and [DeepSeek V4.1](deepseek_v41_en.md) |

Qwen4 CUDA inference keeps MTP draft experts on the GPU: on the single device, on the final CUDA stage for PP, or as resident shards on each TP rank. Draft experts require additional VRAM outside the target expert-cache budget set by `--moe_cuda_cache`. CPU inference retains the configured MoE placement.

Qwen3.5 MTP and DFlash drafts support the following settings. Compatible paths use NVFP4 by default once the corresponding draft algorithm is enabled. Environment variables can override these defaults before startup:

| Environment variable | Default | Description |
| --- | --- | --- |
| `FASTLLM_DRAFT_QUANT` | `nvfp4` | `off` disables NVFP4 conversion; `nvfp4_head` converts only the separate draft output head; `nvfp4` converts the draft backbone and head. The target head retains its original weights; unsupported devices, types or shapes keep their original path |
| `FASTLLM_CUDA_NVFP4_SWIGLU_MULTIROW` | `1` | Allows multirow NVFP4 Linear + SwiGLU fusion; `0` disables it. The current multirow kernel supports SM75, FP16 input and M=2–8, subject to shape, layout and scratch checks; other cases fall back |
| `FASTLLM_TP_NVFP4_MLP_SWIGLU` | `1` | Tries NVFP4 Linear + SwiGLU fusion in TP MLPs; `0` disables it. Kernel capability checks select the implementation, with the original Linear + SwiGLU as fallback |
| `FASTLLM_TP_NATIVE_GREEDY` | `1` | Enables native-type logits in Qwen3.5 TP greedy sampling, avoiding FP16→FP32 conversion during eager speculative verification; `0` disables it. Sampling, returned logits, graphs and GPU token handoff retain FP32 |
| `FASTLLM_DFLASH_BATCH_PREFIX_SNAPSHOTS` | `1` | DFlash2 CUDA verification restores each accepted prefix without target replay. Deployments with max_batch≤4 retain per-position state snapshots; larger deployments retain compact activations and recover conv/GDN state in a batch, including when the active batch shrinks. There is no 16-request limit. Recovery buffers and draft-window KV are budgeted automatically, reducing target KV capacity. `0` restores the full-prefix replay fallback; the single-request path is unchanged. No SM restriction; state kernels validated through batch 32 on SM75 |
| `FASTLLM_MTP_BATCH_SAMPLING` | `1` | Batch Qwen3.5 CUDA MTP draft and target rejection sampling across independent per-request proposal caches, with combined result readback. Mixed greedy/sampling batches and incompatible verification lengths retain the existing path. `0` restores per-request sampling. Preserves the sampling distribution, not identical random token sequences |
| `FASTLLM_MTP_FP8_MARLIN` | `1` | Prepare eligible Qwen3.5 TP MTP FP8 draft shards in the existing Marlin layout during initialization so multirow draft calls do not miss lazy conversion. Retains the weight quantization format and existing architecture/backend preferences. `0` restores lazy preparation. Validated on SM75/TP2 |
| `FASTLLM_CUDA_GDN_SEQUENCE_PREPARE` | `1` | Precompute normalized Q/K and gate coefficients for eager batched short-sequence GDN with 128-wide K/V and batch≥4, avoiding repeated work across state tiles. Preserves per-token FP16 state rounding and prefix snapshots. Small batches, other V dimensions and graph scopes retain the existing path. `0` disables; validated on SM75/TP2 |
| `FASTLLM_MTP_DRAFT_TOKEN_IDS` | Unset | Optional single- or multi-GPU MTP draft vocabulary token-ID file, used only when NVFP4 conversion is enabled. Unset or `0` keeps the full vocabulary. Multi-GPU runs select rows within each original vocabulary shard and map results back to global token IDs; alignment padding repeats existing candidates. The shortlist is used only for greedy drafts; sampling uses the full output head and target verification retains the full vocabulary |
| `FASTLLM_DFLASH_DRAFT_TOKEN_IDS` | Unset | Optional whitespace-separated token ID file for greedy DFlash2. Supports original-encoding GGUF or NVFP4 draft heads on one GPU, and NVFP4 on multiple GPUs; each contiguous vocabulary shard needs at least selector top-k candidates. Padding is removed before top-k and selected IDs are mapped back to original tokens. The full target head is retained, so the shortlist adds draft-head memory; limited coverage can reduce acceptance. Validate task throughput and output consistency before enabling. Sampling uses the full vocabulary; invalid or unsupported lists fall back to the full vocabulary. Disabled when `FASTLLM_DRAFT_QUANT=off` |
| `FASTLLM_DFLASH_ATTENTION` | Unset: on for SM75, off otherwise | Unified switch for fused FP16 DFlash sliding-window attention: `0` disables it; `1` enables it on supported devices. Requires head_dim=128, 1–16 queries, queries × GQA group size ≤64, queries ≤ window ≤4096, and available FlashInfer support; unsupported cases retain the original path. Correctness and speed of the Q64 path on SM80 and newer have not been validated on those GPUs. SM70 and older always retain their original path |

DFlash2 dynamic convolution, QKV preparation and Gateup preparation select fused implementations automatically when the device, types and shapes are supported, otherwise falling back to regular operators. TP selector projection is always queued early on rank 0's head worker stream. The former switches `FASTLLM_CUDA_DFLASH_FUSED_CONV`, `FASTLLM_CUDA_DFLASH_FUSED_QKV_PREPARE`, `FASTLLM_CUDA_DFLASH_FUSED_GATEUP_PREPARE` and `FASTLLM_DFLASH_TP_EARLY_SELECTOR` have been removed; setting them no longer changes execution.

Unset variables use the defaults listed above; kernel capability checks still apply. Draft NVFP4 supports single- and multi-GPU runs, converting eligible shards after splitting. MTP requires a dense model and FP16 compute; DFlash adapts quantized Linear inputs to FP16 and restores the original activation type. Quantization can change draft acceptance, and separate head and view copies change memory use, so it does not guarantee a speedup for every model. Target weights and full-vocabulary verification stay unchanged. Set `FASTLLM_DRAFT_QUANT=off` to keep the original draft precision; the existing multi-GPU FP8 draft-head switch remains effective.

NVFP4 small-matrix decode tuning is enabled by default on SM75 devices with 68 SMs, such as the RTX 2080 Ti. It covers M=1–8 for fused SwiGLU with N×K=17408×5120 and Linear with N×K=5120×8704 or 5120×3072, subject to the runtime block-residency check. Other architectures, shapes, and prefill with an original M>8 retain their existing paths. Set `FASTLLM_CUDA_NVFP4_SM75_DECODE_TUNE=0` to disable tuning, `1` to explicitly enable both parts, or `linear` / `swiglu` to select one part. The multirow SwiGLU fusion entry remains controlled separately by `FASTLLM_CUDA_NVFP4_SWIGLU_MULTIROW`.

### API server

| Option | Default | Description |
| --- | ---: | --- |
| `--host` | `0.0.0.0` | Listen address |
| `--port` | `8080` | Server port; the WebUI defaults to 1616 |
| `--model_name` | Automatic | Deployed name validated and returned by the API |
| `--api_key` | Empty | Enable Bearer API-key validation when non-empty |
| `--temperature` / `--top_p` / `--top_k` | Model default | Override server-side sampling defaults |
| `--repeat_penalty` | Model default | Override repetition penalty; `--repetition_penalty` is an alias |
| `--hide_input` | Off | Hide request content from server logs |
| `--startup-progress` | `off` | Set to `ndjson` to emit model-loading and readiness events to stderr |

Inspect the complete command-specific help:

~~~bash
ftllm --help
ftllm server --help
ftllm bench --help
ftllm download --help
~~~

## Model formats, download, and export

### Input formats

- Original Hugging Face Safetensors checkpoints, including model-provided FP16, BF16, or FP8 weights.
- Quantized AWQ checkpoints.
- Fixed-precision or dynamically quantized models exported by FastLLM.
- Selected GGUF formats can read supported embedded configuration and tokenizer metadata, or use `--ori` to select the original model directory. Qwen3.5-family GGUF models with a separate vision module also require `--mmproj` and matching vision configuration; see [GGUF multimodal deployment](qwen3_en.md#gguf-multimodal).

Quantization support depends on the model architecture, device, and available kernel. Keep `--dtype auto` for an initial deployment, and do not request online quantization again for an already quantized checkpoint.

### Download a model

~~~bash
ftllm download <repo-id> --local-dir /data/models/model-name
~~~

To download configuration and tokenizer files without weight shards:

~~~bash
ftllm download <repo-id> \
  --exclude "*safetensors*" \
  --local-dir /data/models/model-config
~~~

### Export a model

Online quantization increases startup time on every load. Export the result once in FastLLM format instead:

~~~bash
ftllm export /data/models/source-model \
  -o /data/models/source-model-int4 \
  --dtype int4 -t 16
~~~

Regular and expert layers of an MoE model can use different precisions:

~~~bash
ftllm export /data/models/source-moe \
  -o /data/models/source-moe-mixed \
  --dtype float16 --moe_dtype int4 -t 16
~~~

See the [dynamic quantization guide](dtype_config.md) for layer-specific configurations.

## Build from source

Building requires a C++17 compiler, Make, and CMake. GCC/G++ 9.4+ and CMake 3.23+ are recommended. Linux NUMA builds normally require `libnuma-dev`. Install a compatible CUDA Toolkit and NCCL before a CUDA build.

~~~bash
# Ubuntu/Debian base dependencies
sudo apt-get install -y build-essential cmake libnuma-dev

# NVIDIA CUDA
bash install.sh -DUSE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)"

# Specify a CUDA architecture; for example, 89 for Ada
bash install.sh -DUSE_CUDA=ON -DCUDA_ARCH=89 \
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)"

# CPU-only
bash install.sh
~~~

Additional platform documentation:

- [ROCm build and wheel packaging](rocm.md)
- [TFACC platform](tfacc.md)
- [Examples, Android, and other platforms](../example/README.md)
- [Build and runtime FAQ](faq.md)
