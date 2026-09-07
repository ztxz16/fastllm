# FastLLM

[中文](README.md) · [Quick start](#quick-start) · [Model deployment guides](#model-deployment-guides) · [Benchmarks](docs/benchmark_en.md) · [Common options](#common-options) · [Changelog](docs/version_en.md)

FastLLM is a high-performance inference engine for running and serving large language models. Its core runtime is implemented in C++ and does not depend on PyTorch. It supports dense and MoE models across CUDA, ROCm, CPU, NUMA, disk-assisted hybrid inference, and multi-GPU tensor parallelism.

The project includes command-line chat, a terminal deployment wizard, a WebUI, benchmarking tools, and APIs compatible with OpenAI Chat Completions, OpenAI Responses, and Anthropic Messages.

## Highlights

- **Fast support for new model architectures:** the current development line covers Qwen4-Exp / Qwen3.8-Flash-Next, Qwen3.5/3.6/3.8, DeepSeek-V4, Kimi-K3, GLM-5.3-Flash, Dots3-Note, and more.
- **Large MoE deployment on heterogeneous hardware:** place regular layers and experts independently on CUDA, CPU, NUMA, or disk, or combine devices with explicit ratios.
- **Multi-GPU and high-throughput serving:** tensor parallelism, odd GPU counts, dynamic batching, streaming, paged KV cache, prefix caching, chunked prefill, and CUDA Graph.
- **Speculative decoding:** MTP, embedded or external DSpark, and DFlash2 paths for compatible models.
- **Multiple precisions and formats:** Hugging Face Safetensors, exported FastLLM models, AWQ, and selected GGUF models, with model- and hardware-dependent FP16, BF16, FP8, NVFP4, MXFP4, INT4, and K-Quant paths.
- **Production-facing APIs:** separated reasoning content, tool calling, streaming responses, cache usage statistics, server-side sampling defaults, and structured startup progress.
- **Extensible backends:** native CPU/CUDA/ROCm operators, optional Triton kernels, custom Python model graphs, and source-level integrations for additional accelerators.

> Operator coverage varies by model, quantization format, and hardware backend. Validate accuracy, memory use, and throughput with the exact checkpoint and hardware intended for production.

## Current model focus

The table highlights the active development line instead of using legacy models as the main project description.

| Family | Current focus |
| --- | --- |
| Qwen | Qwen4-Exp / Qwen3.8-Flash-Next text decoding with QSA, PLE n-grams, and CPU/CUDA/NUMA hybrid execution; MTP for Qwen3.8-Flash-Next; MTP and DFlash2 for Qwen3.5/3.6/3.8 |
| DeepSeek | DeepSeek-V4 / V4-Flash, sparse attention, embedded DSpark, multi-GPU CUDA, and CPU/NUMA hybrid MoE |
| Kimi | Kimi-K3, KDA/MLA, external DSpark, CUDA and NUMA execution, CPU/GPU experts, and disk experts |
| GLM | GLM-5 DSA, GLM-5.3-Flash KDA and paged caching, and quantized KV-B CPU inference for GLM-5.2 |
| Others | Dots3-Note, Laguna, HY-V3, Step3.5/3.7, MiniMax-M2, Gemma4, and more |

Qwen4-Exp / Qwen3.8-Flash-Next currently does not load vision weights. Qwen3.8-Flash-Next can load MTP weights on demand and enable speculative decoding with `--mtp`. See the [legacy model compatibility list](docs/models.md) for earlier architectures and the [changelog](docs/version_en.md) for the latest additions and limitations.

## Quick start

### Installation

A dedicated Python virtual environment is recommended. Prebuilt packages cover these common setups:

| Environment | Command | Notes |
| --- | --- | --- |
| Linux + NVIDIA GPU | `python -m pip install -U ftllm` | Includes the Python interface and common CUDA runtime dependencies; the installed driver must be compatible |
| Windows + NVIDIA GPU | `python -m pip install -U ftllm` | Install the Windows dependency wheel below first if required DLLs are missing |
| Linux + AMD GPU | [ROCm installation and build](docs/rocm.md) | Choose the build and installation options for your GPU architecture |
| CPU-only, special architectures, or other accelerators | [Build from source](#build-from-source) | Select the appropriate CMake backend |

Windows dependency package for first-time installations:

~~~bash
python -m pip install https://www.modelscope.cn/models/huangyuyang/fastllmdepend-windows/resolve/master/ftllmdepend-0.0.0.2-py3-none-win_amd64.whl
python -m pip install -U ftllm
~~~

If Conda introduces shared-library conflicts, try a clean `venv`. See the [FAQ](docs/faq.md) when installation or model loading fails.

### Verify the installation

The following command uses the small Qwen3-0.6B checkpoint as an installation smoke test. It is chosen for download size, not as a statement of the project's current model focus.

~~~bash
ftllm run Qwen/Qwen3-0.6B
~~~

The API server is the primary deployment entry point:

~~~bash
# API server, listening on 0.0.0.0:8080 by default
ftllm server Qwen/Qwen3-0.6B

# Command-line chat
ftllm run Qwen/Qwen3-0.6B

# WebUI, connected to the API server above; listens on 127.0.0.1:1616 by default
ftllm webui --api_base http://127.0.0.1:8080/v1

# Browser deployment launcher; no arguments starts it and opens the local page
ftllm
ftllm launch  # Equivalent form

# Terminal deployment wizard
ftllm tui

# Benchmark
ftllm bench Qwen/Qwen3-0.6B \
  --device cuda --input_tokens 512 --output_tokens 128 --batch 4
~~~

`ftllm` (or `ftllm launch`) listens only on `127.0.0.1:8000` by default and opens the browser after the service is ready; use `ftllm launch --no-browser` to disable automatic opening. The page can download models from ModelScope, save and preview launch profiles, and run either `ftllm server` or the chat-oriented `ftllm webui`. After a local model is selected for a new item, Launcher recommends TP, hybrid MoE inference, and N-gram storage settings from the model structure, weight size, GPUs, system memory, and NUMA topology; the optional inference settings can also be re-analyzed or cleared manually. Its interface supports Simplified Chinese and English, preferring the last selection and otherwise following the browser language; terminal output from `ftllm launch` always uses English. To access it from the LAN, run `ftllm launch --host 0.0.0.0`; the terminal and Launcher page then list loopback, LAN, and directly assigned public-interface addresses when available. Public access also requires the host firewall and cloud security group to allow the port, plus port forwarding when behind NAT; Launcher does not discover NAT public mappings. Non-loopback access uses unencrypted HTTP and should only be enabled on a trusted network. It shares profiles with the terminal wizard and stops its managed download and model processes when the launcher exits. Run `ftllm launch --help` for other options.

Once the API Server is ready, click **Open Studio** to use chat, saved conversations, Markdown, attachments, reasoning, and agents directly in Launcher's content area. Model management navigation stays visible, so you can visit launch profiles, downloads, logs, and hardware, then return to the same conversation. Launcher and standalone `ftllm webui` share the chat component and backend; the component adapts its colors, sizing, and language to Launcher. It connects to the active model with the configured API key, without another WebUI process or port. Conversations use WebUI’s existing local storage and survive page refreshes. Stopping or switching models cancels active WebUI tasks and disposes of the old component.

The WebUI does not load a model in its own process, so start an OpenAI-compatible API server first. Its optional `model` positional argument is only a model-name hint; when omitted, the WebUI discovers the model from `/v1/models`.

Code analysis and web search use the Pi agent runtime by default. On Linux
x86-64, build and install the companion wheel as described in
[`tools/ftllm_agent_runtime/`](tools/ftllm_agent_runtime/). The wheel bundles
Pi, so Node.js, npm, and Bun are not required. Use `--agent-runtime builtin`
to select the original single-call paths when the companion wheel is absent.

Launcher automatically uses an installed Pi runtime. **New Agent** lets you select a project directory; use `ftllm launch --agent-workspace-root /path/to/projects` to set the selectable root (your home directory by default). Directory agents are enabled by default on both local and remote Launcher listeners, for example `ftllm launch --host 0.0.0.0 --agent-workspace-root /path/to/projects`. Use `--disable-workspace-agent` to disable directory browsing, creating directory agents, and executing saved directory-agent tasks; ordinary chat remains available. Directory agents can modify files and execute commands, so restrict access to trusted users. The interface explains when the runtime is unavailable or directory agents have been disabled.

For `run`, `server`, and `export`, the `model` positional argument can be a Hugging Face repository ID, a local Hugging Face model directory, an exported FastLLM model, or a configuration file:

~~~bash
ftllm server /data/models/my-model --device cuda
~~~

### Call the API

Start a local model with a stable API name:

~~~bash
ftllm server /data/models/my-model \
  --model_name local-model \
  --host 0.0.0.0 --port 8080 \
  --api_key local-key
~~~

Call the OpenAI Chat Completions-compatible endpoint:

~~~bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer local-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [{"role": "user", "content": "What can FastLLM do?"}],
    "stream": false
  }'
~~~

The server also implements the OpenAI Responses API and Anthropic Messages API. Compatible models can expose separate reasoning content, tool calls, and multimodal inputs.

## Model deployment guides

Attention architecture, MoE placement, speculative decoding, and quantization differ substantially by model. Start with the model-specific guide, then use the generic deployment patterns below as a reference.

| Model | Deployment guide | Recommended configuration topics | Benchmark |
| --- | --- | --- | --- |
| Qwen4-Exp / Qwen3.8-Flash-Next | [Qwen4-Exp guide](docs/qwen4_exp.md) | PLE, QSA, CUDA/NUMA, and `--ngram_device disk` | [Qwen4 benchmark](docs/benchmarks/qwen4_exp_en.md) |
| Qwen3.5 / Qwen3.6 / Qwen3.8 | [Current Qwen guide](docs/qwen3_en.md) | Single GPU, TP, hybrid MoE, MTP, and DFlash2 | [Qwen3 benchmarks](docs/benchmarks/qwen3_en.md) |
| DeepSeek-V4 / V4-Flash | [DeepSeek-V4 guide](docs/deepseek_en.md) | CUDA + NUMA, disk experts, TP, and embedded DSpark | [DeepSeek-V4 benchmark](docs/benchmarks/deepseek_v4_en.md) |
| Kimi-K3 | [Kimi-K3 guide](docs/kimi_k3_en.md) | KDA/MLA, hybrid and disk experts, external DSpark | [Kimi-K3 benchmark](docs/benchmarks/kimi_k3_en.md) |
| Dots3-Note | [Dots3-Note guide](docs/dots3_note_en.md) | DSA, long context, CUDA + CPU/NUMA | [Dots3-Note benchmark](docs/benchmarks/dots3_note_en.md) |
| GLM-5 / GLM-5.3-Flash | [GLM-5 guide](docs/glm5_en.md) | DSA/KDA, paged cache, NUMA, quantized KV-B on CPU | [GLM-5 benchmark](docs/benchmarks/glm5_en.md) |
| Laguna | [Laguna guide](docs/laguna_en.md) | Multi-GPU TP, hybrid MoE, NVFP4, INT4_GROUP32 | [Laguna benchmark](docs/benchmarks/laguna_en.md) |

The [benchmark index](docs/benchmark_en.md) keeps hardware, complete launch commands, TTFT, prefill, decode, and concurrent throughput separate for each model. Configurations without repository measurements are explicitly marked as unmeasured instead of being extrapolated from another device or model.

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

When host memory is limited, a small fraction of expert layers can be placed on disk, for example `--moe_device "{'cuda':1,'numa':8,'disk':1}"`. Disk execution depends heavily on SSD random-read performance. See the [hybrid inference guide](docs/mixforward.md).

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

Extension currently supports HF Qwen2, Qwen3, and Qwen3.5 layouts, including Qwen3.8 checkpoints based on Qwen3.5. Launcher's advanced "RoPE extension" field accepts the same configuration. GGUF, FLM, and custom GraphLLM loaders retain their existing shrink-only length behavior and do not support the new RoPE extension options. See the [context extension guide](docs/context-length-extension-design.md) for parameters, supported layouts, and validation results.

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

See the [Qwen3.8 DFlash2 report](docs/dflash2_qwen38_27b_tp2_20260819.md) for the complete configuration and validation results.

### Disk-backed Qwen4 PLE

The PLE table in Qwen4-Exp / Qwen3.8-Flash-Next is large. It can be read from the checkpoint on demand when host memory is constrained:

~~~bash
ftllm server /data/models/qwen4-exp \
  --device cuda --moe_device numa \
  --ngram_device disk
~~~

Disk mode reduces resident memory at the cost of random I/O; a fast SSD is recommended. See the [Qwen4-Exp guide](docs/qwen4_exp.md) for details.

## Common options

The CLI evolves continuously, so `ftllm <command> --help` is authoritative for the installed version. The tables below cover the options most often used for deployment.

### Model, device, and precision

| Option | Description |
| --- | --- |
| `model` / `-p, --path` | Hugging Face repository ID, local HF directory, FastLLM model, or configuration file |
| `--device` | Main compute device; common values are `cpu`, `cuda`, and `numa` |
| `--tp` | CUDA tensor-parallel devices; accepts `0,1`, `2`, or `auto` |
| `--moe_device` | MoE expert device: `cpu`, `cuda`, `numa`, `disk`, or a weighted combination |
| `--moe_device_layers` | Apply `--moe_device` only to the last N MoE layers; `-1` means all |
| `-t, --threads` | CPU/NUMA thread count; `-1` selects it automatically |
| `--dtype` | Weight type when loading HF weights; defaults to `auto` and should normally not override an already quantized checkpoint |
| `--moe_dtype` | Set the MoE weight type separately |
| `--atype` / `--moe_atype` | Activation types for regular and MoE layers |
| `--kv_cache_dtype` | KV-cache type: `auto`, `float16`, `bfloat16`, `fp8_e4m3`, or `fp4`; requires model and backend support |
| `--dtype_config` | Dynamic quantization configuration; see the [quantization guide](docs/dtype_config.md) |
| `--triton` | Enable available Triton CUDA operators |

### Memory, context, and scheduling

| Option | Default | Description |
| --- | ---: | --- |
| `--gpu_mem_ratio` | `0.9` | Fraction of GPU memory available to weights and caches |
| `--moe_cuda_cache` / `--moe-cuda-cache` | `0` | GPU expert-weight cache budget, e.g. `3g` (3 GiB); `0` disables it. Use with CUDA compute and CPU/NUMA experts. KV cache, common weights, and workspaces are separate; see the [expert-cache guide](docs/cuda-expert-cache.md) |
| `--kv_cache_limit` | `auto` | Maximum KV-cache memory |
| `--tokens` | Automatic | Total token capacity used to size paged KV cache |
| `--page_size` | Backend-defined | Tokens per KV-cache page; the multi-GPU default is normally 16 |
| `--max_batch` | Automatic | Maximum number of requests processed together |
| `--max_context_length` / `--max-context-length` | Automatic | Combined input and output limit per session; explicit HF targets must pass RoPE and KV capacity checks |
| `--rope_scaling` / `--rope-scaling` | Model config | RoPE extension as `yarn` or JSON; available for adapted HF model layouts |
| `--chunked_prefill_size` | Disabled/model-defined | Prefill chunk size, for example `8192` |
| `--prefix_cache` | Model/environment-defined | Enable or disable prefix caching with `true` or `false` |
| `--cuda_slab` | `0` | CUDA weight-slab size in MB; `0` disables it |
| `--ngram_device` | `cpu` | Store the Qwen4 PLE table on `cpu` or `disk` |

### Decoding, templates, and tool calling

| Option | Description |
| --- | --- |
| `--enable_thinking` | Control the model's thinking template when supported |
| `--mtp` | Draft tokens per step for models with MTP support; `0` disables it and the current maximum is 8 |
| `--dspark` | Enable embedded DSpark and set draft tokens per step |
| `--draft` / `--draft_model_path` | External MTP, DSpark, or DFlash checkpoint; MTP may point directly to `mtp.safetensors` |
| `--draft_tokens` | Maximum draft tokens per step; defaults to the draft configuration |
| `--tool_call_parser` | Tool-call parser; defaults to `auto` |
| `--chat_template` | Custom Jinja chat-template file |
| `--cache_dir` | Local cache directory for online models |
| `--ori` | Original model configuration and tokenizer directory for selected GGUF models |

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
- Selected GGUF formats, with `--ori` pointing to the original model's configuration and tokenizer directory.

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

See the [dynamic quantization guide](docs/dtype_config.md) for layer-specific configurations.

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

- [ROCm build and wheel packaging](docs/rocm.md)
- [TFACC platform](docs/tfacc.md)
- [Examples, Android, and other platforms](example/README.md)
- [Build and runtime FAQ](docs/faq.md)

## Documentation

| Topic | Documents |
| --- | --- |
| Releases | [Stable changelog](docs/version_en.md) · [Nightly usage](docs/nightly.md) · [Nightly changelog](docs/nightly_changelog.md) |
| Model deployment | [Qwen4-Exp](docs/qwen4_exp.md) · [Qwen3.5/3.6/3.8](docs/qwen3_en.md) · [DeepSeek-V4](docs/deepseek_en.md) · [Kimi-K3](docs/kimi_k3_en.md) · [Dots3-Note](docs/dots3_note_en.md) · [GLM-5](docs/glm5_en.md) · [Laguna](docs/laguna_en.md) |
| Hybrid inference | [GPU, NUMA, and disk deployment](docs/mixforward.md) |
| Performance and validation | [Benchmarks by model](docs/benchmark_en.md) |
| Quantization | [Dynamic quantization configuration](docs/dtype_config.md) |
| Extensions | [Custom Python models](docs/english_custom.md) · [Custom operators](docs/custom_op.md) |
| Platforms and troubleshooting | [ROCm](docs/rocm.md) · [TFACC](docs/tfacc.md) · [FAQ](docs/faq.md) |

Some detailed backend and benchmark documents are currently written in Chinese; command examples and option names remain directly usable.

## Community and contributing

Report problems through [GitHub Issues](https://github.com/ztxz16/fastllm/issues) and contribute through [Pull Requests](https://github.com/ztxz16/fastllm/pulls).

Deployment discussion QQ group: `831641348`

User group:

<img src="docs/wechat_group0.jpg" width="220" alt="FastLLM user WeChat group QR code">

Developer community:

<img src="docs/develop-group.png" width="220" alt="FastLLM developer WeChat group QR code">

FastLLM is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgements and references

FastLLM uses or draws implementation ideas from the following projects and articles:

- [PyTorch](https://github.com/pytorch/pytorch) for low-level operator implementation ideas.
- [Transformers](https://github.com/huggingface/transformers) for model architectures and reference implementations.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) and [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) for GGUF quantization formats and kernels.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) for attention, MLA, and related operators.
- [TurboMind / LMDeploy GEMM kernels](https://github.com/InternLM/lmdeploy/tree/main/src/turbomind/kernels/gemm): the SM70 s884 (`SM70_MMA_884` / HMMA 8x8x4) core bundled under `third_party/turbomind` is ported from this source and is used for AWQ INT4, block-scaled FP8, and NVFP4 Linear paths.
- [1Cat-vLLM's SM70 TurboMind integration](https://github.com/1CatAI/1Cat-vLLM/tree/main/csrc/sm70_turbomind) for the AWQ integration and the FP8/NVFP4 type, layout, and small-batch tactic references. FastLLM separately implements its Torch-free raw-pointer bridge, model-weight format conversion, unaligned padding, fallbacks, and dispatch.
- [KTransformers](https://github.com/kvcache-ai/ktransformers/blob/main/csrc/ktransformers_ext/cpu_backend/backend.cpp) for dynamic MoE thread scheduling; see also the [design article](https://zhuanlan.zhihu.com/p/1900318746402329329).
- [Lvllm](https://github.com/guqiong96/Lvllm/blob/main/csrc/lk/moe.cpp) for NUMA-aware MoE scheduling.
- [FreeToken](https://github.com/FlashML-org/FreeToken) for design ideas behind CUDA expert caching and related hybrid inference optimizations, including GPU-side routing and LRU cache management, on-demand refills of expert weights from host memory, and CUDA Graph-compatible execution. See the [CUDA expert cache documentation](docs/cuda-expert-cache.md) for implementation and extension details.
- [vLLM](https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/openai/tool_parsers) for tool-call parsing.
- [json11](https://github.com/dropbox/json11) for JSON construction and parsing.

Thank you to every open-source contributor. If an attribution is missing, please report it through an Issue.
