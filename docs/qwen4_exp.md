# Qwen4-Exp / Qwen3.8-Flash-Next FP8 / NVFP4 / GGUF

[中文部署指南](qwen4.md) · [Back to README](../README_EN.md) · [Benchmark](benchmarks/qwen4_exp_en.md)

FastLLM supports FP8, NVFP4 and `qwen4exp` GGUF checkpoints of the Qwen4-Exp / Qwen3.8-Flash-Next text decoder implemented
in `src/models/qwen4_exp.cpp`. Vision tensors in composite checkpoints are not
loaded. For `Qwen3_8FlashNextForConditionalGeneration` checkpoints, MTP
tensors are loaded on demand only when `--mtp` is greater than zero.

## Implemented architecture

- four-stream hyper-connections, including grouped delta-weight RMSNorm;
- 3-linear / 1-full-attention layer schedule;
- separate Q/K/V/Z/b/a Gated DeltaNet projections, depthwise causal
  convolution, recurrent state, L2-normalized Q/K, and sigmoid output gate;
- partial RoPE GQA and the long-context QSA block indexer;
- 512 routed experts plus the shared expert;
- PLE hashed 2/3-gram lookup, raw BF16/E4M3 host or disk-backed shards, the
  checkpoint's common scalar, EOS-aware history, signed hash remainder, gated
  injection, and dilated depthwise convolution;
- per-request PLE, convolution, recurrent, KV, and QSA index-key state.

By default, the PLE embedding stays resident on the CPU.  Only the selected
160-element rows are converted to float and scaled, so loading the model does
not expand the very large table to float32.

Set `--ngram_device disk` to keep the PLE table in its original safetensors
files and read only the selected rows.  This is useful when host memory is
limited; the Qwen3.8-Flash-Next checkpoint's 320,001,536-row table occupies
95.37 GiB.  The default is `--ngram_device cpu`.

```bash
ftllm benchmark "$MODEL" \
  --device cuda --moe_device cuda --ngram_device disk \
  --max_batch 1 --input_tokens 4096 --output_tokens 16 \
  --warmup 1 --temperature 0
```

Disk mode lowers process RSS by avoiding the resident table allocation.  The
operating-system page cache may still use otherwise-free memory, and decode
performs small random reads, so an SSD is recommended when this mode is used.

## GGUF checkpoints

The Unsloth `UD-Q2_K_XL` three-shard checkpoint can be loaded directly by passing
`Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf`. Keep all three shards in the
same directory with their original filenames. The metadata-only first shard is
valid; merging shards or supplying an HF configuration is unnecessary.

Use `--device cudapp=2 --moe_device numa --moe_cuda_cache 0 --ngram_device disk
--mtp 0 --atype float16 --threads 28 --chunked_prefill_size 4096` on the dual
22 GiB RTX 2080 Ti machine. See the [complete command](qwen4.md#ggufunsloth-ud-q2_k_xl).

Routed experts retain their mixed IQ2_XS/IQ3_XXS/IQ4_NL storage. Disk PLE decodes
selected IQ4_NL rows into FP32, while dense projections are imported as FP16.
The importer restores GDN head ordering, joins QSA Q/K projections, preserves
GGUF normalization offsets, and uses the checkpoint's exact PLE hash metadata.
NUMA GPU prefill also supports the CPU's IQ2_XS/IQ3_XXS R4 layout.

These three shards contain no MTP weights; deploy them as a text model with
`--mtp 0`. The NVFP4 expert-cache, resident-layer and performance results below
are separate configurations.

On 2026-09-29, one measured run per phase (after warmup) on the dual 22 GiB
2080 Ti host achieved **801.56 token/s prefill** (4096-token chunk) and
**20.50 token/s decode** (512 input / 512 output). Decode TTFT was 1.29 s,
process VmHWM was 55.58 GiB, and peak GPU usage was 8192/8754 MiB, with no swap.
Arithmetic and structured JSON tasks completed correctly with natural stops.
These tests validate this configuration, not the maximum context length.

This run includes the IQ4_NL multi-row CPU down kernel, with unchanged weight
formats and deployment parameters. Decode improved by 7.5% over the initial
GGUF candidate's 19.07 token/s. A separate short profile measured total down
time across 48 layers falling from 12.97 to 9.82 ms/token. The full requests
share their first 96 output tokens but diverge afterward; these single runs
do not establish a fixed speedup for every workload.

## Hybrid deployment on two GPUs

For `Qwen3.8-Flash-Next-NVFP4`, `--device cudapp=2` places consecutive decoder
layers on two GPUs (24/24 layers); one request traverses them sequentially.
`--tp 2` instead uses both GPUs to compute each layer. Do not combine these
two modes, and clear an inherited `FASTLLM_TP` before using `cudapp`.

Both modes support `--moe_device numa --moe_cuda_cache 0` for host experts,
or `--moe_device numa --moe_cuda_cache 8g` for an 8 GiB expert cache **per GPU**.
Expert caching is independent of `--prefix_cache`, which reuses request
prefix state. The expert budget does not include other weights, KV or workspace.

With TP2, `--moe_device numa --moe_device_layers 36 --moe_cuda_cache 0`
keeps all experts of the first 12 layers on the GPUs and the last 36 layers
on NUMA. The GPU experts are tensor-parallel shards, not whole layers split
between the cards. The MTP experts remain on NUMA.

For sequential placement, use the validated aligned layout:
`--device cudapp=1:7 --moe_device "{'cuda:0':6,'cuda:1':6,'numa':36}" --moe_cuda_cache 0`.
Decoder layers 0–5 and their experts use GPU0; decoder layers 6–47 use GPU1,
with only experts of layers 6–11 resident there. Avoid the `moe_device_layers`
shorthand with this sequential mapping, and do not change the decoder split
to 24/24 while keeping the 6/6 expert mapping: that combination previously
hit an invalid memory access.

See the [deployment commands](qwen4.md) and
[two-2080-Ti measurements](benchmarks/qwen4_exp_en.md) for the common settings,
4096-token prefill, decode and memory figures. The test cards have **22 GiB
each**, not the standard 11 GiB. Fixed expert placement reserves every expert
in selected layers; dynamic caching can retain active experts across layers.
Lower steady-state host RSS does not imply an equally low loading peak.

## MTP speculative decoding

```bash
ftllm server /data/models/qwen3.8-flash-next \
  --device cuda --moe_device numa \
  --mtp 4
```

`--mtp` sets the number of draft tokens per step, with a current maximum of
eight. Qwen3.8-Flash-Next MTP currently requires simple greedy decoding, a CUDA
target device map, and CUDA or NUMA MoE placement. Unsupported configurations
automatically fall back to ordinary target decoding.

Qwen3.8-Flash-Next prefix snapshots preserve target and available MTP state.
Enabling MTP after recording a target-only snapshot causes one recomputation
to replace it with an MTP-compatible snapshot; later hits can continue MTP.

## Prefix caching with hybrid TP

`--tp 2 --moe_device numa --prefix_cache true` supports cross-request prefix
reuse. Every rank must hold a complete snapshot of the same token prefix.
Restore recovers each rank's KV, linear-attention and QSA state, plus rank
zero's PLE history. Missing or incompatible shards cause recomputation.
Image and video requests do not use token-only cross-request snapshots.

`FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_MB` limits the whole TP snapshot store
(4096 MiB by default), divided equally among ranks. The default recording
interval is 16 pages, or 2048 tokens with the default 128-token pages, so
short requests may not produce a snapshot. At least one token remains
uncached on a hit; identical prompts can reuse an earlier chunk snapshot.

## Build and smoke tests

```bash
bash install.sh

ftllm benchmark "$MODEL" \
  --device cpu --moe_device cpu --atype float32 --threads 64 \
  --input_tokens 2 --output_tokens 1 --batch 1 --warmup 0 --temperature 0

ftllm benchmark "$MODEL" \
  --device cuda --moe_device cpu --atype float16 --moe_atype float32 \
  --threads 64 --input_tokens 2 --output_tokens 1 --batch 1 \
  --warmup 0 --temperature 0

ftllm benchmark "$MODEL" \
  --device cuda --moe_device numa --atype float16 --moe_atype float32 \
  --threads 64 --input_tokens 2 --output_tokens 1 --batch 1 \
  --warmup 0 --temperature 0
```

On the validation host (72 CPU cores, two NUMA nodes, RTX PRO 6000 Blackwell
96GB), all three commands completed.  The two-token measurements were:

| execution path | TTFT | prefill |
| --- | ---: | ---: |
| CPU + MoE CPU | 274.44 ms | 7.29 token/s |
| CUDA + MoE CPU | 78.03 ms | 25.63 token/s |
| CUDA + MoE NUMA | 69.68 ms | 28.70 token/s |

These very short measurements are smoke-test figures, not sustained-throughput
benchmarks.

The long-context selector was also exercised past its budget boundary:

```bash
ftllm benchmark "$MODEL" \
  --device cuda --moe_device numa --atype float16 --moe_atype float32 \
  --threads 64 --chunked_prefill_size 2048 \
  --input_tokens 2052 --output_tokens 1 --batch 1 --warmup 0 --temperature 0
```

The 2052-token prefill completed in 8.8368 seconds without a QSA/cache error.
The synthetic request selected an EOS token immediately, so the benchmark
reported zero post-prefill output tokens rather than a TTFT value.

## Layerwise reference check

Set `FASTLLM_QWEN4_DUMP_DIR` to export float32 input IDs, positions,
embeddings, every attention output, every decoder output, the PLE injection
point, final hidden state, and logits.  The directory must already exist.

```bash
mkdir -p /tmp/qwen4_dump
FASTLLM_QWEN4_DUMP_DIR=/tmp/qwen4_dump \
  ftllm benchmark "$MODEL" \
    --device cpu --moe_device cpu --atype float32 --threads 64 \
    --input_tokens 2 --output_tokens 1 --batch 1 --warmup 0 --temperature 0

python tools/qwen4_exp_reference_check.py \
  "$MODEL" /tmp/qwen4_dump --device cpu --threads 64 \
  --json /tmp/qwen4_reference.json
```

The checker follows the public Transformers eager equations, reads one layer
at a time, expands FP8 block scales only for routed experts, and slices only
the PLE rows selected by the test tokens.  For validation tokens `[31114,
3950]`, FastLLM and the reference produced the same argmax token (`44`) and the
same top-10 token set (10/10 overlap).  Logit cosine similarity was
`0.999888539`; relative L2 error was `0.0150250`.

## Official Transformers CPU check

`qwen4_exp_transformers_check.py` constructs the upstream
`Qwen4ExpForCausalLM` on a meta device, loads all regular parameters on CPU,
and keeps only the 512-expert and sharded PLE storage lazy.  The upstream
Transformers implementations of hyper-connections, attention, Gated DeltaNet,
PLE, routing, norms, and the language-model head execute unchanged.

```bash
PYTHONPATH=/path/to/transformers/src \
  python tools/qwen4_exp_transformers_check.py \
    "$MODEL" /tmp/qwen4_dump --threads 64 \
    --json /tmp/qwen4_transformers_cpu.json
```

The command exits non-zero unless argmax matches, logit cosine is at least
`0.999`, and relative L2 error is at most `0.02`.  Against Transformers
`5.16.0.dev0` at `36deb0b53ed0863f4b4dfdea23dcaec7f3df3701`, the validation
run passed all three checks.  It compared all 48 decoder outputs and all
248,320 final logits; argmax was `44` in both implementations, top-10 set
overlap was 10/10, logit cosine was `0.999888480`, and relative L2 error was
`0.015025171`.  The official CPU forward took 9.244 seconds after lazy model
storage was ready.

## Thinking and tool calling

Qwen4-Exp is registered with the existing Qwen tagged-reasoning and XML tool
protocols.  The OpenAI server advertises `low`, `medium`, and `xhigh`
reasoning efforts, with `xhigh` as the default.  Per-request thinking can be
disabled with:

```json
{"chat_template_kwargs":{"enable_thinking":false}}
```

The validation run covered both modes and a required tool call through
`/v1/chat/completions`.  Thinking mode returned separate `reasoning_content`
and final `content`; disabled thinking returned only `391`.  A weather request
returned a structured `get_weather` tool call with `{"city":"北京"}` and
`finish_reason: "tool_calls"`.

The architecture audit used these source snapshots:

| project | snapshot | use in the audit |
| --- | --- | --- |
| Transformers | `36deb0b53` | official CPU model graph and eager QSA, GDN, PLE, hyper-connection, MoE, and logits equations |
| SGLang | `73a255206` | exact Qwen4-Exp inference path, pinned-host PLE layout, and weight-loader semantics |
| vLLM | `17da485` | Qwen3-Next/Qwen3.5 FP8 and GDN baseline |

That vLLM snapshot has no native `Qwen4Exp` model registration, so it cannot be
used as an independent Qwen4-Exp logit oracle.  Treating its Qwen3-Next model as
if it were Qwen4-Exp would omit PLE, hyper-connections, the separate GDN
projections, and QSA, and would therefore be a misleading comparison.
