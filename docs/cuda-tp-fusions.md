# CUDA decode fusions and tensor parallel shards

These kernels operate on local tensor dimensions, not on the TP rank count. They use SIMT CUDA operations and retain complete unfused Block paths. Changing the rank count does not require another weight format or another model implementation.

## Supported local tensors

| Path | Activations / batch | Local shape and weight layout |
| --- | --- | --- |
| GDN input projection + convolution + SiLU + cache update | FP16 or BF16; batch 1–8 | Row-scaled FP8; K is a multiple of 256, 256–32768; N <= 65536; convolution width 4; positive QKV and Z widths |
| Row-scaled FP8 GEMV | FP16 or BF16; batch 1 | 512 <= K <= 32768, N >= 4096; one scale per output row, including blockM >= local K |
| FP8 projection + residual | FP16 or BF16; batch 1 | Same row-scale semantics; 512 <= N <= 65536; float bias or no bias |
| NVFP4 gate/up + SwiGLU, or down + residual | FP16; batch 1 (down shape below also supports 2–8) | Existing NVFP4 block-16 Marlin layout; N multiple of 128, 128–65536; K multiple of 128, 256–32768; no bias |

The FP8 row path selects vector loads for K multiples of 512 or 256. Other widths use masked byte loads, aligned activation pairs, and an explicit odd-element tail. Legacy FP16/BF16 fallback GEMV paths also handle uneven row widths safely.

GDN dimensions are runtime parameters, including the separation between QKV and Z. Cache slot IDs must be valid and distinct, as required by the existing Block. The original full-size specialization is retained. NVFP4 has a K=5120 specialization with runtime N as well as a fully dynamic specialization; the existing full-size kernels are retained.

## Admission and fallback

`CanRun` checks the operation's dtype, layout, alignment, batch, scale metadata, buffer separation where required, and available CUDA image. GDN, FP8 residual and NVFP4 retain an SM75 minimum; the row GEMV also checks its loadable CUDA image. Missing capability returns to the existing complete operation path. `CanRun` does not update residuals or convolution state. A cold FP8 scale cache is not admitted during graph capture.

The uneven gate/up shapes N=11520 or 11776, K=5120 deliberately retain Marlin + SwiGLU: end-to-end measurements on RTX 5090 found this faster than the fused SIMT kernel. This is a conservative performance exception, not a correctness restriction. The other new fusions remain active for these TP configurations.

Residual fusion preserves the existing TP reduction protocol: only the designated rank adds the replicated residual before all-reduce. Adding it on every rank would multiply it by the TP size.

`FASTLLM_CUDA_TP_FUSIONS=0` disables the new shape extensions for an A/B comparison while retaining previously supported fusions. Existing per-operation flags still apply:

- `FASTLLM_CUDA_GDN_INPUT_CONV`
- `FASTLLM_CUDA_FP8_ROW_GEMV`
- `FASTLLM_CUDA_FP8_LINEAR_ADD`
- `FASTLLM_CUDA_NVFP4_SWIGLU`
- `FASTLLM_CUDA_NVFP4_ADD`

## Validation scope

The CUDA tests cover the TP=2/3/4 shard dimensions, other local dimensions, FP16/BF16 where supported, CPU reference values, forced and shape-based fallbacks, changing inputs across CUDA Graph replays, and GDN cache slots. CUDA 13.1 compilation was checked for SM75, 80, 86, 89, 90, 100, 120 and 121. Runtime correctness, memory checking and end-to-end performance were measured on SM120 only; compilation success is not a performance claim for other GPUs.

## Local down projection tuning and small batches

For FP16 NVFP4 down projections with local N=5120, K=8704, batch-one
selects a fixed-dimension instance of the existing SIMT GEMV core. The residual
Block uses its additive epilogue; ordinary Linear uses an overwrite epilogue,
so non-root TP ranks also use GEMV without adding a replicated residual.
The warp layout and FP32 reduction order match the generic fused down kernel.
The dispatch no longer requires SM120: it retains the original SM75 minimum
and checks that a compatible CUDA kernel image is available. FP4 decoding
uses the existing native conversion or its half-based fallback on older SMs.
Gate/up retains its previous dispatch.

For this down shape, batches 2–8 use Marlin Tensor Cores with a fused residual
epilogue. Intermediate split-K results stay in the existing FP32 scratch;
only the final reduction owner adds the residual while writing output. The
Linear result is rounded to half before addition, matching Linear + AddTo.
Ordinary Linear on non-root TP ranks keeps the original Marlin path for these
batches. A batched SIMT GEMV was evaluated and rejected because it was slower.

The small-batch CanRun requires the complete reduction scratch to fit in the
already-prepared weight allocation and checks the selected residual kernel's
image and shared-memory capability. Kernel attributes are prepared and cached
outside CUDA Graph capture; a cold attribute cache during capture falls back.
No scratch allocation or weight-layout change is performed by this fusion.
Batch >8, unsupported dimensions, dtype, bias, layout, overlapping buffers,
missing scratch, or unsupported kernels retain the complete existing Block path.

`FASTLLM_CUDA_NVFP4_SHAPE_TUNING=0` disables this shape's tuning, including
the small-batch residual fusion, while preserving preceding TP fusions.
The existing NVFP4 and TP-fusion flags still apply. Set flags before server
startup / CUDA Graph capture.

Runtime validation is on RTX5090 only. Compilation coverage on other SMs and
the absence of SM120-only instructions do not establish performance on GPUs
that have not been measured.
