# Qwen3.8-Flash-Next User Guide

[中文](README.md) · [Project home](../../README_EN.md)

This guide covers text serving with Qwen4-Exp / Qwen3.8-Flash-Next: hybrid inference, serial and parallel use of two GPUs, expert caching, fixed GPU expert layers, MTP, and API requests. The two-GPU examples use the NVFP4 model.

## Prepare the model

Install a FastLLM version that supports the model and prepare a complete local model directory containing its configuration, tokenizer, and Safetensors weights. Replace `/data/models/Qwen3.8-Flash-Next-NVFP4` with your model path.

Hybrid inference requires enough host memory. Store the model on an SSD when using disk-backed PLE.

## Automatic configuration (recommended)

On Linux with NVIDIA CUDA, pass the complete HF/NVFP4 model directory:

```sh
ftllm server /data/models/Qwen3.8-Flash-Next-NVFP4 \
  --auto --model_name qwen3.8-flash-next --host 0.0.0.0 --port 8080
```

For GSQ GGUF, pass the first shard and keep the remaining shards in the same directory. Replace the sample path with the actual filename:

```sh
ftllm server /data/models/Qwen3.8-Flash-Next-GSQ-RCO/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --auto --model_name qwen3.8-flash-next --host 0.0.0.0 --port 8080
```

Only add options you want to override:

| Adjustment | Append to the command |
| --- | --- |
| Limit planning resources | `--auto_gpu_memory_limit 20g --auto_host_memory_limit 128g` |
| Set the session window | `--max_context_length 32768` |
| Set concurrency | `--max_batch 2` |
| Set the per-GPU expert cache | `--moe_cuda_cache 7g`; `0` disables it |
| Set threads and prefill chunk size | `--threads 32 --chunked_prefill_size 4096` |
| Set embedded MTP draft tokens | `--mtp 3`; requires matching weights; `0` disables it |
| Load a matching external MTP model | `--draft /path/to/mtp.gguf` |
| Disable all speculative decoding | `--speculative_algorithm off` |

Resource limits are planning budgets, not runtime memory caps. `--auto` alone does not enable Flash-Next MTP. A supported external MTP GGUF defaults to 3 draft tokens; override with `--draft_tokens`. Automatic mode keeps prefix and history reuse enabled by default. See the [`--auto` tuning guide](../auto_en.md) for option effects and Launcher controls.

The following commands are manual examples for fixed layouts and troubleshooting. Their thread counts and cache sizes are specific examples, not requirements for automatic mode.

## Cross-request decode batching

With `--max_batch 4`, ready requests share attention input/output projections, HyperConnection operations, routing and expert execution during ordinary decode and MTP verification. Each MTP draft step also batches its projections and experts. Accepted output blocks are published consecutively before requests enter the next verifier round. Sampling keeps each request's temperature, proposal distribution and rejection state.

KV, QSA, GDN, PLE histories and accepted-prefix rollback remain request-owned. Different context lengths, acceptance lengths, request completion and batch reordering are supported without fixed concurrency or verifier-row dispatch limits. The shape-sized hybrid expert entry uses the shared GPU cache, dynamic CPU/GPU routing and NUMA batch scheduler beyond the old single-request verifier width.

Single-GPU ordinary batched decode and batched MTP with sparse QSA contexts support complete CUDA graphs: one backbone replay per ordinary decode step or verification round, and one complete draft replay per proposal step. NVFP4 and GGUF share the graph entry; ordinary decode does not retain speculative rollback outputs. Shared hybrid expert graphs retain the GPU expert cache, dynamic CPU/GPU routing, and NUMA batch dispatch. The first draft step reserves space according to the configured draft length; device-side valid-row counts control QSA state commits and final-row selection, avoiding graph variants for every combination of accepted lengths. Input preparation, independent sampling, and accepted-prefix rollback remain outside the model graph.

Single-GPU `--auto` prepares full graphs for NVFP4 and GGUF hybrid inference through backend device/operator capability checks, without an SM75-only gate or additional environment settings.

Full graphs schedule independent GDN projections and QSA index/Q/K/V preparation branches according to their data dependencies. Ordinary decode, verification and drafting share this mechanism. Warmup records concurrent scratch requirements, with reuse after each join; shared-library workspace conflicts or insufficient scratch retain serial dependencies. Precise RoPE inverse frequencies are precomputed by rotary dimension and base, while positions remain dynamic on replay, without a context-length-sized rotary table.

Text requests can use persistent serving slots bounded by `max_batch`, including non-greedy sampling and concurrent requests. Completion clears logical history and KV page occupancy while retaining captured storage addresses. Later clients reuse the slot's full graphs. Stable slot ordering and separate caches for decode, verification and draft stages avoid rebuilding merely because client handles or scheduler order change.

Graph retention uses LRU eviction: each stage retains at most `max_batch` compositions and at most twice the largest observed row workset in aggregate. This bounds retained workspace growth without limiting executable row counts. New shapes, changed storage addresses/capacities and evicted compositions still warm up and capture. Failed or cancelled slots synchronize and invalidate dependent graphs before release. `Qwen cross-request decode/verify/draft full graph ready` reports a newly composed graph; replay does not repeat the message.

Serial graph segments share only temporary allocations released during capture. Persistent outputs retain stable addresses, while parallel native expert scratch remains separate. Eager execution and prefill never borrow pinned graph scratch, so later inputs can execute safely while graphs remain cached.

Short contexts, prefill, TP, multimodal/KVMem, prefix/history restoration and unsupported capture cases retain their established paths. Cross-request reuse was tested on single-GPU IQ3_S in this change; weight-format/operator dispatch is unchanged, and other models and placements need separate validation.

## Manual configuration reference

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

### Concurrency and batched verification

Hybrid NVFP4 experts (16-element blocks with E4M3 scales) launch expert projections for the whole MTP verification batch together. Supported GGUF IQ experts offer grouped candidates that reuse decoded weights across verifier rows. Resident and dynamically staged branches have separate workspaces and preserve existing quantization, reduction and NUMA arithmetic. No extra tuning environment variable is required. Physical tile sizes do not cap the logical verifier row count.

For GGUF, `--kernel-tune quick` benchmarks and validates each format, shape and device before capture; the default `off` retains the existing kernel. Family `moe.gguf.gate_swiglu_down` offers `q8`, `grouped`, `grouped_shared`, `q8_tiles_8` and `q8_tiles_16` (prefix `cuda.moe.gguf.`): the reference Q8 path, decoded-weight reuse, reuse with staged activations, and two output tilings that preserve the reference reduction order. Startup logs report the selection. See [kernel tuning](../kernel-policy.md) for policy, profile, deny and force options. Small batches may retain the reference; gains at larger batches depend on measurements. Warmup uses resident experts from the current layer and fixed inputs without consuming request sampling state.

`--max_batch` controls concurrent admission, `--max_context_length` bounds each request, and `--tokens` sets the aggregate shared KV token budget. For example, `--max_batch 4 --max_context_length 8192 --tokens 32768` plans capacity for four requests, subject to available VRAM. To keep all 48 expert layers on the NUMA hybrid path, set `--moe_device numa --moe_device_layers 48` explicitly.

Higher concurrency does not guarantee proportional throughput; report simultaneous decode throughput, verification time, each draft step and prefill waits separately.

The complete CUDA graph schedules A/B, QKV plus convolution, and Z as three independent GDN branches, joining before recurrence. Nested regions preserve dependencies and retain serial execution on scratch or workspace conflicts. GDN QKV/Z and QSA Q/K/V each share GGUF activation quantization; mixed layouts generate packed and planar inputs in one pass. Cross-request verification and drafting retain independent convolution and recurrent histories. Larger batches retain the existing chunked MMQ scheduler without a new logical row limit.

`--kernel-tune quick` also selects shared FP16 projections, QSA and small TopK sampling. `off` uses each family's default implementation.

| Family | Candidate IDs | Selection |
|---|---|---|
| `linear.multi.fp16` | `reference`, `cublas`, `tile4`, `tile8`, prefixed by `cuda.linear.multi.fp16.` | Shared-input projections, measured for the actual row count, widths and device |
| `linear.hyper_project.fp16` | `reference`, `tile4`, `tile8`, `gemm_epilogue`, prefixed by `cuda.hyper_project.` | HC Down/Injection; the GEMM candidate merges output promotion and activations |
| `attention.indexed.fp16` | `gather`, `split_pv`, `cublas_qk`, prefixed by `cuda.attention.indexed.` | Materialized gather, direct indexed KV, or cuBLAS QK with direct indexed V; direct indexing requires CUDA SM75+ and supported FP16 layouts |
| `qwen4.ple_gate` | `serial`, `block256`, prefixed by `cuda.ple_gate.` | Original serial or parallel FP32 dot reduction for PLE gating; FP32/FP16/BF16 inputs, parallel by default, no SM or logical row restriction |
| `sampling.logits_topk` | `tile1024`, `tile2048`, `tile4096`, prefixed by `cuda.sampling.topk.` | GPU candidate selection, measured for the actual vocabulary, batch and device; scores, token IDs and tie order must match exactly |

Candidates must pass a separate L2 check for every output before timing. Unsupported or numerically rejected candidates retain the reference. Capture never performs trials, and an existing graph keeps its selected implementation. Tile sizes describe physical blocks, not logical row limits. Direct QSA supports head dimensions up to 256 in multiples of 16, 1–32 query heads per KV head and sparse widths 128–3072; other shapes retain the original path. Warm actual verifier and draft shapes before capture, rather than deriving them from `max_batch`. `--kernel-profile FILE` records choices, timings and errors; existing `--kernel-policy` force/deny settings can override the selection. No new tuning environment variable is required.

Parallel PLE reduction changes the floating-point accumulation order. For the original sequential order, use `--kernel-tune quick` with policy `{"force":{"qwen4.ple_gate":"cuda.ple_gate.serial"}}`. CPU and unsupported inputs retain their original paths. QSA output gating overwrites the complete output buffer, so graph replay no longer clears it first.

GGUF expert tuning waits for the required weights before cold trials; actual inference retains compute/upload overlap. IQ3_S has passed `--kernel-tune quick` warmup and non-speculative 1/2/4-request regression with physical GPU availability capped at 32 GiB.

Ordinary sampling with CUDA FP32 logits, no token mask or effective repetition penalty, and effective `top_k ≤ 64` reads back only GPU-selected candidates while retaining the original CPU top-p and RNG sequence. Other configurations retain the original sampling path. Sampling remains outside the model graph. The 32-row workspace tile bounds temporary storage, not the logical batch size.

Record `nsys --version` when profiling. On the tested SM120 system, the same NVFP4 binary reproduced CUDA 719/Xid 31 during a two-to-four-request transition with Nsight Systems 2025.3.1 node tracing, but passed with 2026.5.1 (CUPTI 13.4, hardware tracing). The exact cause in the older profiling configuration remains unidentified. Use the validated 2026.5.1 with `--trace=cuda,nvtx --cuda-graph-trace=node`, and check CUDA API errors and actual verifier/draft graph-node records.

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

See the [benchmark](../benchmarks/qwen4_exp_en.md) for configuration, throughput, memory and validation scope.

<a id="gguf-resident-performance"></a>

### GPU-resident GGUF experts

Place all shards together and select the first shard. This configuration uses TP2 with all experts on GPU, embedding on CPU and PLE on disk. Use hybrid deployment when GPU memory is insufficient.


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


### HC fusion and operator policy

The optional `qwen4.hyper_read` family fuses the pending residual update, grouped
RMSNorm, Down/Injection, and Up/Mix. Its CUDA kernels are adapted from
[Strata, commit 9f7d977d](https://github.com/Niko1221/Strata/tree/9f7d977d220e9fdf2ec53cf2620cdadac1cfb24e),
Copyright (c) 2026 Niko1221 and the Strata contributors, under the MIT license.
See [provenance and modifications](../../third_party/strata/README.md) and
[the retained license](../../third_party/strata/LICENSE). No Strata runtime is required.

The current specialization uses FP32 residuals/norms and FP16 projection weights,
with 4 streams, width 2560, and rank 320. Other shapes/storage use the existing
implementation. Physical row tiles do not cap the logical batch size. Ordinary,
verification, and draft graph paths use the same entry point; CPU remains unchanged.

Use `--kernel-tune quick --kernel-policy hc-policy.json` with:

```json
{"families": ["qwen4.hyper_read"]}
```

The family filter leaves other operators on their existing backend defaults.
Candidates are `cuda.hc.reference`, `cuda.hc.strata.cols4`,
`cuda.hc.strata.cols8`, `cuda.hc.planar.cols4`, and `cuda.hc.planar.cols8`; `ftllm ops` describes their requirements.
Warmup measures each actual shape and checks all outputs separately against the
reference. Logs and `--kernel-profile FILE` show the selected implementation,
timings and relative L2. Standard `force`/`deny` policy options are available;
unsupported or unmeasured graph shapes retain the reference. `--kernel-tune off`
keeps the original implementation. Selection is not based on a fixed SM/row cutoff.

Concurrent single-token GDN decode batches independent recurrent updates and
output gating in one CUDA launch while preserving the original output/state
arithmetic. Bounded argument tiles do not impose a 32-request limit. Unsupported
layouts, multi-token verification and CPU retain their original paths; verifier
and draft full graphs remain supported. The HC `planar` candidates change only
shared-memory layout and use the same warmup benchmark, without C2/C4 or SM
selection rules or new environment variables.


### GDN recurrence selection

The single-token FP16 recurrence, output normalization and gate share the
`gdn.decode.fp16` kernel family. `--kernel-tune quick` validates both the output
and FP32 state with independent relative-L2 checks, then compares
`cuda.gdn.reference` with `cuda.gdn.register.rg1`, `rg2` and `rg4`.
The register variants are adapted from the [Strata GDN CUDA operator](../../third_party/strata/README.md),
with MIT attribution, FastLLM's head-major state layout and FP16 output boundaries.
Benchmark trials write private scratch states; they never advance live requests.
The selected implementation supports CUDA Graph capture and replay.

There are no SM or logical-request performance cutoffs. The 32-request launch
argument tile loops over arbitrary batches, with a separate measured plan per
actual request count and logged selections. Multi-token verification retains
its sequence recurrence and fallback; CPU execution is unchanged. The existing
`--kernel-policy` supports `families`, `force` and `deny`, for example
`{"families":["gdn.decode.fp16"],"force":{"gdn.decode.fp16":"cuda.gdn.register.rg2"}}`.
Unmeasured shapes, disabled tuning and unsupported layouts retain the original
path. No new environment switch or Strata runtime dependency is introduced.


### Causal convolution decode selection

`conv.causal_depthwise.decode` registers the original general implementation
and four-tap vector implementations. The latter load FP32 history and weights
once, shift history and accumulate in registers, and use FP32 SiLU. FP32,
FP16 and BF16 inputs are supported. `--kernel-tune quick` validates output
and history independently and measures 64/128/256-thread blocks without SM
or logical-batch performance cutoffs. Block size does not limit batch size.

Candidates are `cuda.conv.decode.vec4.block64`, `.block128`, `.block256`, with
`cuda.conv.decode.reference` as the fallback. Other tap counts or insufficient
vector alignment retain the original implementation; CPU execution is
unchanged. Existing `--kernel-policy` families/force/deny options apply, with
no additional environment switch. Validation writes private outputs; timing
initializes private history outside the measured graph and then uses in-place
updates, so tuning never advances live request state. Single-token and
concurrent decode remain graph-compatible; multi-token verification retains
its sequence-convolution path.


### GGUF projections and attention input preparation

`gguf.planar.projection` benchmarks plain Q4_K/IQ4_XS projections with FP16 output.
Candidates are `cuda.planar.reference`, `cuda.planar.w2.r1.full`,
`cuda.planar.w4.r1.occupancy`, and `cuda.planar.w8.r2.occupancy`.
They reuse existing GGUF dequantization/dot products, varying warps, output tiles
and grid geometry. Warmup selects by actual format, rows, K/N, output stride and
device, without architecture or concurrency performance gates. The original
kernel remains the fallback; residual/fused gate-up and large-batch MMQ keep
their existing dispatch.

Enable `--kernel-tune quick` and include `gguf.planar.projection` in the policy's
`families` when filtering. Existing `force`/`deny` options control the candidates.
Capture never benchmarks. Preparation on a caller-owned stream waits for input
readiness and validates private outputs. The public planar entry tiles arbitrary
logical rows in physical groups of up to eight, replacing its old 16-row limit;
the surrounding projection scheduler still decides when to use planar or MMQ.

Fused HC can write the attention activation dtype directly (FP16/BF16), retaining
FP32 residuals and injection scales, and removing a separate cast and same-dtype
copy. Unsupported shapes, CPU and unselected fused kernels retain the original
calculation plus conversion. GDN A/B may overlap shared Q8 quantization; QKV and
Z fork only after quantization and join before recurrence. Ordinary, verifier
and draft execution share this interface.

### QSA across request batches

The FP16 graph path reads packed projections directly while retaining each
request's KV cache, positions and valid row count. It fuses Norm/RoPE with KV
writes, TopK with index expansion, and indexed QK/softmax/PV within attention
chunks. Merge writes the gated output directly. Decode, verification and draft
share this entry point, preserving causal visibility, raw-key checkpoints and
rollback. Request and verifier row counts have no fixed limit; CUDA argument
packs and attention chunks are physical scheduling units.

Family `attention.qsa.requests.fp16` offers optimized `cuda.qsa.requests.chunk64`
and `cuda.qsa.requests.chunk128`, plus the basic fused implementation
`cuda.qsa.requests.reference64`. Use `--kernel-tune quick` and include the family
in an existing policy's `families` list. Logs and `--kernel-profile FILE` record
selections and numerical errors. For example,
`{"force":{"attention.qsa.requests.fp16":"cuda.qsa.requests.chunk64"}}`
pins one candidate. Trials replay only stateless attention, never live KV or
compression-tail updates. Tuning off uses 64-token chunks; capture never tunes.

The implementation uses portable CUDA SIMT with 128/256-dimensional attention
heads, FP16 contiguous or paged KV caches and ragged request lengths. Other dtypes,
layouts, KVMem, CPU and unsupported geometries retain the original path.
Projections continue through the common weight-format dispatcher. Attention
chunk/merge is adapted from the [Strata CUDA operators](../../third_party/strata/README.md),
with the source revision and MIT license retained. Neither the Strata nor Maya
inference engine is required.

The optimized candidates reuse queries across KV cells, vectorize KV reads,
combine head reductions and use cell-major probabilities. Unaligned KV addresses
retain scalar loads. Index and QKV projections run in parallel; index preparation
and selection also overlap the separate Q/K preparation branch, joining before
attention. Single-request decode, batched verification and draft share these
optimizations without request-count or SM-specific performance switches.
