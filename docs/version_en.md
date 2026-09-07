# Changelog

[中文](version.md)

## V0.1.8.2

- Added a web deployment launcher and an Electron desktop launcher with model downloads, launch profile management, automatic configuration for long contexts and high concurrency, and inference speed and available context displays.
- Reworked Studio with concurrent chat sessions, generation cancellation, attachments, and history. Integrated the Pi directory agent with web search, document Q&A, data analysis, code processing, and PowerPoint generation.
- Improved Qwen3.8-Flash-Next multimodal inference, MTP, and prefix caching. Added FP8/NVFP4 CUDA expert caching, dynamic hybrid scheduling, and NUMA weight sharing to improve hybrid inference speed and memory usage.
- Added GLM-5.3 MTP speculative decoding and MTP support for DeepSeek-V4 hybrid inference.
- Improved self-describing GGUF loading, with embedded and external MTP weights and multimodal projectors for Qwen3.5-architecture GGUF models. Fixed low-bit quantization compatibility issues and reduced temporary GPU memory usage through chunked dequantization.
- Added the `--low_gpu_mem` mode and FP4 KV Cache for the Qwen3.5 architecture. Reduced memory usage during tensor-parallel weight loading, optimized high-resolution vision inference, and added image feature caching.
- Added `--rope_scaling` with static YaRN context extension for HF Qwen2, Qwen3, and Qwen3.5 families, along with context capacity checks at startup.
- Optimized FP8 Linear, NVFP4, GDN, and paged attention across GPU architectures to improve prefill, decode, and small-batch inference performance.
- Fixed hangs with odd GPU counts in tensor parallelism, deadlocks during multi-GPU quantized-weight repacking, GPU memory leaks in MTP verification graphs, and out-of-bounds accesses in two-GPU multimodal inference. Improved streaming tool-call parsing and argument validation.
- Added native SM86 code to wheels and Triton as a default Linux dependency. Improved portable package dependencies, command-line entry points, and usage instructions for deployment without a desktop environment.

## V0.1.8.1

- Improved Qwen3.8-Flash-Next with MTP speculative decoding, cross-request prefix caching, and faster CUDA/NUMA hybrid inference.
- Improved Qwen3.5 GGUF, DFlash2, and two-GPU inference, and fixed related CUDA Graph issues.
- Improved build and packaging for CUDA 12.9, multi-architecture wheels, the Triton server script, and Linux portable packages.

## V0.1.8.0

### New models and inference capabilities

- Added text-generation support for Qwen4-Exp / Qwen3.8-Flash-Next FP8, including four-stream hyper-connections, Gated DeltaNet, QSA sparse attention, PLE ngrams, and FP8 MoE. Vision and MTP weights are not loaded by the current text decoder.
- Improved Qwen4 CPU, CUDA, and NUMA hybrid inference, long-context and prefix caching, fused prefill/decode operators, and CUDA Graph execution. Added `--ngram_device disk` to read the large PLE table from disk on demand and reduce host-memory usage.
- Added Kimi-K3 support with CUDA, NUMA, CPU/GPU expert parallelism, disk-backed experts, chunked KDA/MLA prefill, DSpark, XTML serving, and tool calling.
- Added Dots3-Note support with DSA indexing, sparse attention, long-context caching, chunked prefill, reasoning output, and tool calling.
- Added GLM-5.3-Flash support with KDA, paged history caches, compressed paged attention, NUMA decode pipelining, reasoning output, and tool calling.
- Added GLM-5 DSA support and pure-CPU inference for GLM-5.2 checkpoints with quantized KV-B weights.
- Added Laguna support and long-context caching, with improved CUDA Graph execution, eight-GPU tensor parallelism, hybrid MoE, NVFP4, and INT4_GROUP32 inference.
- Added HY-V3 model and tool-call parsing support, together with the Poolside V1 tool-call protocol.
- Added DFlash2 speculative decoding for Qwen3.8/Qwen3.5 architectures, including separate draft checkpoints, rejection sampling, batched decoding, two-GPU tensor parallelism, long-context execution, prefix snapshots, and compact INT4 draft weights. Equal-weight two-GPU CUDA deployments now automatically enable DFlash2 backbone tensor parallelism.
- Added embedded DSpark for DeepSeek-V4 and external DSpark draft models for Kimi-K3. Unified DSpark/DFlash2 draft options with the new `--dspark`, `--draft` / `--draft_model_path`, and `--draft_tokens` shorthand while retaining the full `--speculative_*` options.
- Further optimized Qwen3.5 MTP, adding MoE MTP support and improving batched state backup, eight-token verification snapshots, concurrent scheduling, and long-KV verification.

### Performance and backends

- Optimized Qwen3.5 routing, FP8 MoE, RMSNorm, GDN, prefill, and decode fusion. Eligible CUDA tensor-parallel deployments now automatically enable CUDA Graph, GPU Token Handoff, and CUDA Embedding fast paths.
- Changed the default Qwen3.5 chunked-prefill size to 2048.
- Expanded DeepSeek-V4 CUDA, CPU, and NUMA inference with full-step multi-GPU graphs, SM120 sparse MLA, routing, MoE and WoA acceleration, compressed-cache improvements, and mixed NUMA MoE pipelining.
- Optimized Dots3 sparse prefill, FP8 indexing, and long-KV movement; Kimi-K3 KDA/MLA caching and chunked prefill; and the GLM-5.3-Flash NUMA decode pipeline.
- Added CUDA Q2_K GGUF, AWQ MoE Marlin, tensor-parallel NVFP4 Marlin, SM75+ FP8 dense Marlin, automatic SM89 FP8 CUTLASS selection, and compact NVFP4 CUDA/NUMA fallbacks.
- Added compressed-tensors asymmetric INT4 `pack-quantized` checkpoint support, converting weights into portable `INT4_GROUP` tensors for CPU, CUDA, and multi-GPU tensor parallelism, with compatible TFACC metadata handling.
- Improved INT4_GROUP, FP8, NVFP4, MXFP4, Q8_0, and Q8_K loading, compute performance, and memory usage across CUDA, CPU, NUMA, and multi-GPU execution.
- Improved automatic custom AllReduce selection, in-graph multi-GPU communication, no-P2P copy fallbacks, CUDA Graph memory reuse, and paged-cache scheduling.
- Optimized paged-attention prefill and SM70/SM75 D256 GQA decode, added disk-streamed Linear and Embedding execution, and updated FlashInfer to `v0.6.16.post1`.

### Serving and APIs

- Added `--max_context_length` / `--max-context-length` to limit combined per-session input and output length. `/v1/models` now reports the model limit, effective context window, KV-cache capacity, and configured limit.
- Added `--startup-progress ndjson` to emit model initialization, weight loading, warmup, and server-ready progress events on standard error.
- OpenAI Chat Completions, Responses API, and Anthropic API now use native runtime token accounting and report cached input, uncached input, and actual output token counts.
- Added or improved reasoning-effort handling for Qwen3.5, Qwen4-Exp, Kimi-K3, Dots3, and GLM-5.3-Flash, including correct streaming and non-streaming `reasoning_content` separation.
- Added or improved tool-call parsing for GLM, HY-V3, Kimi-K3, Dots3, Qwen4-Exp, and Poolside V1, including better `tool_choice=required`, argument constraints, and fragmented streaming output handling.
- Improved Fastllm Studio model deployment, launch configuration, runtime status, model selection, and local chat workflows.

### Stability and compatibility

- Fixed streaming request-handle reuse races, cleanup after early termination, stop constraints, minimum output length, and cached-token usage accounting.
- Fixed prefix-cache restoration, low-KV-budget scheduling, CUDA Graph KV sizing, communication during graph capture, memory-pool reuse, and multi-GPU quantized-weight partitioning.
- Fixed DeepSeek-V4 multi-GPU cache restoration, pure-CPU and non-CUDA builds, NUMA shutdown hangs, and two-GPU serial decode, as well as Qwen3.5 GGUF merging, multi-GPU sampling, and warmup memory estimation.
- Improved SM60/Pascal compatibility for legacy-only CUDA builds, native paged-attention prefill, and AWQ Marlin fallbacks, and adapted sampling operators to multiple CCCL interfaces.
- Fixed quantization accuracy and cross-platform compatibility issues affecting AWQ, GGUF MoE, FP8 Marlin/CUTLASS, NUMA FP8 MoE, Q8_K, and YaRN.
- Added regression coverage for new models, quantized operators, CUDA Graph, MoE, multi-GPU execution, tool calling, and APIs, together with logits-alignment, NCCL, HLE, and performance-analysis tools.

For release notes before V0.1.8.0, see the [Chinese changelog](version.md).
