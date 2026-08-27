# Qwen4-Exp / Qwen3.8-Flash-Next FP8

FastLLM supports the Qwen4-Exp text decoder stored by the released
`Qwen4ExpForConditionalGeneration` checkpoint.  The implementation is isolated
in `src/models/qwen4_exp.cpp`; vision and MTP tensors in the composite
checkpoint are deliberately not loaded by the text-generation model.

## Implemented architecture

- four-stream hyper-connections, including grouped delta-weight RMSNorm;
- 3-linear / 1-full-attention layer schedule;
- separate Q/K/V/Z/b/a Gated DeltaNet projections, depthwise causal
  convolution, recurrent state, L2-normalized Q/K, and sigmoid output gate;
- partial RoPE GQA and the long-context QSA block indexer;
- 512 routed FP8 experts plus the shared expert;
- PLE hashed 2/3-gram lookup, raw E4M3 host-resident shards, the checkpoint's
  common scalar, EOS-aware history, signed hash remainder, gated injection, and
  dilated depthwise convolution;
- per-request PLE, convolution, recurrent, KV, and QSA index-key state.

The PLE embedding is intentionally gathered on the CPU.  Only the selected
160-element rows are converted to float and scaled, so loading the model does
not expand the very large table to float32.

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

The architecture audit used these source snapshots:

| project | snapshot | use in the audit |
| --- | --- | --- |
| Transformers | `dabae5f` | eager QSA, GDN, PLE, hyper-connection, MoE, and logits equations |
| SGLang | `73a255206` | exact Qwen4-Exp inference path, pinned-host PLE layout, and weight-loader semantics |
| vLLM | `17da485` | Qwen3-Next/Qwen3.5 FP8 and GDN baseline |

That vLLM snapshot has no native `Qwen4Exp` model registration, so it cannot be
used as an independent Qwen4-Exp logit oracle.  Treating its Qwen3-Next model as
if it were Qwen4-Exp would omit PLE, hyper-connections, the separate GDN
projections, and QSA, and would therefore be a misleading comparison.
