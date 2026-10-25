# FastLLM

[中文](README.md) · [Quick start](#quick-start) · [`--auto` tuning](docs/auto_en.md) · [Documentation](docs/README_EN.md) · [Benchmarks](docs/benchmark_en.md) · [Changelog](docs/version_en.md)

FastLLM is an inference engine for running and serving language models locally. Its C++ runtime does not depend on PyTorch. It supports dense and MoE models, GPU/CPU/NUMA/disk hybrid inference, and multi-GPU deployment.

- **Automatic deployment:** `--auto` plans devices, memory, expert caches, and context capacity, with optional overrides.
- **Browser launcher:** download models, save profiles, inspect logs, and use chat or agents in Studio.
- **Compatible APIs:** OpenAI Chat Completions, Responses, and Anthropic Messages.
- **Multiple weight formats:** Safetensors, FastLLM, AWQ, and supported GGUF models. See the [model guides](#model-deployment-guides) for coverage.

## Quick start

### Installation

Linux / Windows with NVIDIA GPUs:

```sh
python -m pip install -U ftllm
```

<details>
<summary>First-time Windows dependencies</summary>

```sh
python -m pip install https://www.modelscope.cn/models/huangyuyang/fastllmdepend-windows/resolve/master/ftllmdepend-0.0.0.2-py3-none-win_amd64.whl
```

</details>

For AMD GPUs, see [ROCm installation (Chinese)](docs/rocm.md). For CPU-only and other platforms, see [building from source](docs/deployment_en.md#build-from-source). Installation help is in the [FAQ (Chinese)](docs/faq.md).

### Browser launch

```sh
ftllm
```

Select a model → choose Automatic configuration → save and start. Expand Resource limits or Model parameters only when needed. Once ready, open Studio to chat. See the [launcher guide](docs/launcher_en.md).

### Command-line launch

Replace the path with a local model directory or supported GGUF file. Hugging Face repository IDs also work:

```sh
ftllm server /path/to/model --auto
```

`--auto` currently supports selected models on Linux with NVIDIA CUDA. Other platforms and models use [manual deployment](docs/deployment_en.md). The default local service URL is `http://127.0.0.1:8080`.

Append only the options you want to change. For example, a 32K session window and at most 2 concurrent requests:

```sh
ftllm server /path/to/model --auto --max_context_length 32768 --max_batch 2
```

See the [`--auto` tuning guide](docs/auto_en.md) for memory budgets, expert caches, chunks, speculative decoding, and fast prefill. Use `ftllm run` for CLI chat, `ftllm tui` for the terminal wizard, and `ftllm bench` for benchmarks.

### Call the API

You can set a service name and API key at startup:

```sh
ftllm server /path/to/model --auto --model_name local-model --api_key local-key
```

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer local-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"local-model","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

For standalone chat, run `ftllm webui --api_base http://127.0.0.1:8080/v1 --api_key local-key`. Reasoning, tool calling, and multimodal features depend on the model.

## Model deployment guides

| Model | Guide |
| --- | --- |
| Qwen4-Exp / Qwen3.8-Flash-Next | [Flash-Next user guide](docs/qwen3.8-flash-next/README_EN.md) |
| Qwen3.5 / Qwen3.6 / Qwen3.8 | [Qwen guide](docs/qwen3_en.md) |
| DeepSeek-V4 / V4-Flash | [DeepSeek-V4 guide](docs/deepseek_en.md) |
| DeepSeek-V4.1-Flash 1017 | [DeepSeek-V4.1 guide](docs/deepseek_v41_en.md) |
| Kimi-K3 | [Kimi-K3 guide](docs/kimi_k3_en.md) |
| GLM-5 / GLM-5.3-Flash | [GLM-5 guide](docs/glm5_en.md) |
| Dots3-Note / Laguna | [Dots3-Note](docs/dots3_note_en.md) · [Laguna](docs/laguna_en.md) |

Other models are listed in the [support list (Chinese)](docs/models.md). For agent-assisted deployment, provide the [model and option reference (Chinese)](docs/model-deployment-agent.md). Measured speed depends on hardware, workload, and quantization; see the [benchmarks](docs/benchmark_en.md).

## Common options

Start with [`--auto` options and effects](docs/auto_en.md). For fixed devices or model-specific settings, use the [manual option reference](docs/deployment_en.md#common-options). `ftllm server --help` is authoritative for the installed version.

[Device settings](docs/deployment_en.md#model-device-and-precision) · [Decoding, templates, and tools](docs/deployment_en.md#decoding-templates-and-tool-calling) · [Download and export](docs/deployment_en.md#model-formats-download-and-export)

## Build from source

See the [source-build instructions](docs/deployment_en.md#build-from-source) for C++ / CUDA / CPU builds, and the [FAQ (Chinese)](docs/faq.md) for platform troubleshooting.

## Documentation

- [All documentation](docs/README_EN.md) · [Launcher and Studio](docs/launcher_en.md) · [Agent integrations (Chinese)](docs/launcher-agents.md)
- [Hybrid inference (Chinese)](docs/mixforward.md) · [Expert caches (Chinese)](docs/cuda-expert-cache.md) · [Context extension (Chinese)](docs/context-length-extension.md)
- [Development and validation (Chinese)](docs/development-validation.md) · [Custom models (Chinese)](docs/custom.md) · [Custom operators (Chinese)](docs/custom_op.md)
- [Stable changelog](docs/version_en.md) · [Nightly (Chinese)](docs/nightly.md)

## Community and contributing

Report problems through [Issues](https://github.com/ztxz16/fastllm/issues) and contribute through [Pull Requests](https://github.com/ztxz16/fastllm/pulls). Deployment discussion QQ group: `831641348`.

<details>
<summary>User and developer WeChat groups</summary>

User discussion (usage and deployment):

<img src="docs/wechat_group0.jpg" width="220" alt="FastLLM user WeChat group QR code">

Developer community (contributions and development):

<img src="docs/develop-group.png" width="220" alt="FastLLM developer WeChat group QR code">

</details>

FastLLM is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgements and references

<details>
<summary>Open-source projects and references</summary>

FastLLM uses or draws implementation ideas from the following projects and articles:

- [PyTorch](https://github.com/pytorch/pytorch) for low-level operator implementation ideas.
- [Transformers](https://github.com/huggingface/transformers) for model architectures and reference implementations.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) and [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) for GGUF quantization formats and kernels.
- [Strata](https://github.com/Niko1221/Strata) for some operators.
- [KVMem / kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp) for design ideas behind tiered KV caching and content-based retrieval, including a bounded GPU working set, host-backed KV storage, and pre-RoPE mean-K retrieval. See the [KVMem documentation](docs/kvmem.md) for the FastLLM integration.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) for attention, MLA, and related operators.
- [TurboMind / LMDeploy GEMM kernels](https://github.com/InternLM/lmdeploy/tree/main/src/turbomind/kernels/gemm): the SM70 s884 (`SM70_MMA_884` / HMMA 8x8x4) core bundled under `third_party/turbomind` is ported from this source and is used for AWQ INT4, block-scaled FP8, and NVFP4 Linear paths.
- [1Cat-vLLM's SM70 TurboMind integration](https://github.com/1CatAI/1Cat-vLLM/tree/main/csrc/sm70_turbomind) for the AWQ integration and the FP8/NVFP4 type, layout, and small-batch tactic references. FastLLM separately implements its Torch-free raw-pointer bridge, model-weight format conversion, unaligned padding, fallbacks, and dispatch.
- [KTransformers](https://github.com/kvcache-ai/ktransformers/blob/main/csrc/ktransformers_ext/cpu_backend/backend.cpp) for dynamic MoE thread scheduling; see also the [design article](https://zhuanlan.zhihu.com/p/1900318746402329329).
- [Lvllm](https://github.com/guqiong96/Lvllm/blob/main/csrc/lk/moe.cpp) for NUMA-aware MoE scheduling.
- [FreeToken](https://github.com/FlashML-org/FreeToken) for design ideas behind CUDA expert caching and related hybrid inference optimizations, including GPU-side routing and LRU cache management, on-demand refills of expert weights from host memory, and CUDA Graph-compatible execution. See the [CUDA expert cache documentation](docs/cuda-expert-cache.md) for implementation and extension details.
- [vLLM](https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/openai/tool_parsers) for tool-call parsing.
- [json11](https://github.com/dropbox/json11) for JSON construction and parsing.

Thank you to every open-source contributor. If an attribution is missing, please report it through an Issue.

</details>
