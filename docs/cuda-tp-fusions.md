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

The uneven gate/up shapes N=11520 or 11776, K=5120 deliberately retain Marlin + SwiGLU: end-to-end measurements on RTX 5090 found this faster than the fused SIMT kernel. This is a conservative performance exception, not a correctness restriction. The other supported fusions remain active for these TP configurations.

Residual fusion preserves the existing TP reduction protocol: only the designated rank adds the replicated residual before all-reduce. Adding it on every rank would multiply it by the TP size.

`FASTLLM_CUDA_TP_FUSIONS=0` disables the shape extensions while retaining the base fusions. Existing per-operation flags still apply:

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
The dispatch requires SM75 or newer
and checks that a compatible CUDA kernel image is available. FP4 decoding
uses the existing native conversion or its half-based fallback on older SMs.
Gate/up retains its previous dispatch.

For this down shape, batches 2–8 use Marlin Tensor Cores with a fused residual
epilogue. Intermediate split-K results stay in the existing FP32 scratch;
only the final reduction owner adds the residual while writing output. The
Linear result is rounded to half before addition, matching Linear + AddTo.
Ordinary Linear on non-root TP ranks keeps the original Marlin path for these
batches.

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

## Naive-N0.5 attention output and SWA on SM103

The following paths are enabled by default when their data and device
requirements are satisfied. They borrow the original BF16 weights. The output
projection fusion also serves DSA when the same contract is satisfied.

| Environment variable | Default | Scope |
| --- | --- | --- |
| `FASTLLM_NAIVE_SWA_QKV_FUSION` | `1` | SM103, one decode row or 1–16 verification rows (optional fused attention for 2–7), BF16, Q8/KV1, 192/128 dimensions, 64 rotary dimensions, 128-slot ring |
| `FASTLLM_NAIVE_SWA_CORE` | `1` | SM103 BF16 contiguous ring decode/verification, Q/KV = 8/1, 16/2 or 32/4 |
| `FASTLLM_NAIVE_SWA_FUSION` | `1` | TP8 SM103 O projection, peer reduction, residual and RMSNorm; BF16 1024→4096 projection |

For a single decode row with all three stages enabled, an additional two-kernel
path combines the deferred residual, input RMSNorm, QKV, RoPE, ring append,
attention, O projection, TP reduction and output residual in one cooperative
kernel, then combines output RMSNorm and the 256-expert FP32 sigmoid router.
It requires TP8 SM103, hidden size 4096, Q8/KV1, QK/V dimensions 192/128,
64 rotary dimensions, a 128-slot BF16 ring, no QKV bias, FP32 norm/router
weights, aligned contiguous tensors and enough cooperative residency for
128 CTAs of 512 threads with 128 KiB of dynamic shared memory each.

QKV producers publish per-tile readiness with GPU release/acquire operations.
Each attention head waits for its own Q tiles and shared K/V tiles. A cooperative
barrier protects the temporary QKV data before O overwrites that scratch.
The following norm/router kernel clears readiness flags on the same stream.
The flags occupy unused high words in the existing epoch buffer; low-word
mailbox generations and the output-only completion counter remain separate.
There are no extra weight copies, persistent GPU allocations or dependencies.

BF16 MMA changes dot-product accumulation order, so the whole-layer path is
not bitwise equivalent to the staged path. It preserves the BF16 residual,
RoPE, probability, local-projection and TP-reduction materialization boundaries;
regression tests bound output relative L2 by 0.003. The established single-row output-only
path retains its bitwise checks. Full-model generated tokens can differ.

The verification-layer kernel supports 1–64 flattened query rows with separate
request-owned KV descriptors; the model currently selects it for at most 16 rows.
For 9–64 rows, two N8 query tiles share each QKV/O weight fragment. Normalization
keeps the original per-thread sum order, QKV/O read existing outputs directly into
registers, and attention uses Tensor Core QK/PV. This schedule needs 32 KiB of
dynamic shared memory per CTA and reuses the existing eight-row mailbox. If any
rank lacks the required kernel or cooperative residency, the group retains the
original N8 staging/scalar-QK schedule. There are no additional persistent GPU
buffers or implementation-selection environment variables.

Unset means enabled; `0` or another value other than `1` disables that stage.
Disabling any stage also disables the whole-layer path. Set these variables
before starting the process: environment changes do not replace kernels in
existing communication groups or captured graphs. Implementation-selection
switches are not needed; each supported shape has a fixed implementation.

The BF16 attention output tail also accepts 1–8 flattened verification rows, including
verification tokens and rows from independent requests. DSA and SWA share this
path. For 1–7 verification rows, one kernel combines O, rank-ordered TP reduction and the
residual; a second combines RMSNorm and the optional router. Original row-major
O weights are shared across rows without packing, retaining the compensated
GEMV sum tree. Eight rows keep the established GEMM, followed by the fused
reduction/residual and norm/router kernels. This avoids introducing a new
projection rounding boundary. The eight-row GEMM uses the ordinary startup
warmup and the weight object's existing auxiliary cache. Cold weight caches
use the ordinary Linear fallback, so the fusion never allocates on borrowed
weight views. During capture, it also requires an eager eight-row call on every rank;
a registered communication tuple alone is insufficient. Unprepared output
buffers also reject multi-row fusion during capture, before any allocation.

The multi-row norm and router preserve both reference column/reduction orders,
BF16 materializations and sigmoid arithmetic. The router must use the existing
single-row reduction policy on every rank; otherwise the group falls back.
Single-row verification explicitly selects the ordered GEMV output tail;
ordinary decode retains its established MMA path. This prevents a changed O
accumulation order from amplifying through later MoE routing.
For 3–8 rows, four experts share each two-row normalization tile in registers.
The original two-row kernel remains the fallback when this variant is not
available, and remains preferred for two rows. The new tile preserves both BF16
materializations, the per-thread dot-product order, both reduction orders, and
sigmoid arithmetic. Capability is checked during state creation; there are no
new weights, persistent buffers, environment switches, or runtime captures.

Other architectures, more than eight rows, unsupported layouts/dtypes and
mixed execution retain the existing dispatch. The output path checks every
rank's architecture, cooperative residency, kernel availability and native
peer atomics before allocating its 3080 KiB per-device workspace (2695 KiB
more than the one-row mailbox). All row counts share a fixed eight-row frame pitch, so alternating
single-row, whole-layer and multi-row graphs cannot overwrite a prior frame.
Generations advance independently for each active row; dormant rows cannot
match an old tag when another row wraps.
The added storage is communication scratch, not a weight copy. There are no
additional allocation sites, dependencies or environment variables.
Failure of either optional whole-layer or multi-row capability check retains
the established output-only backend. A single rank's incompatible tensor rejects
the whole operation, including an already registered pointer tuple, before GPU writes. A first-seen
tuple during Graph capture also falls back before writes. The model restores
the deferred residual/input norm before executing the staged fallback.
Allocation failure unwinds partial initialization. Kernel execution errors are
reported; a partially executed operation is never retried through a fallback.
Serving graphs resolve request-owned KV through a stable device pointer table;
length and cache changes do not require recapture.

To disable all three stages, start with:

```bash
FASTLLM_NAIVE_SWA_QKV_FUSION=0 FASTLLM_NAIVE_SWA_CORE=0 FASTLLM_NAIVE_SWA_FUSION=0 <command>
```

With `UNIT_TEST=ON`, `naive_n05_swa_admission` exercises simulated mixed-SM,
capability, allocation-failure and lifecycle cases; this is policy testing, not
execution on those architectures. `naive_n05_swa_output` tests eight-device
single-row bitwise output, bitwise multi-row output for 2–8 rows, single-rank
collective rejection, cold-capture fallback (including warm tuples with cold
GEMM state, weights or outputs) and Graph replay; nine rows and
its disabled case test the legacy fallback. The optional `--bench` argument
compares fixed-graph replay times with the staged tail. Queued mixed-row
graphs and generation-wrap tests also cover dormant mailbox rows.
`naive_n05_swa_layer` compares the whole layer against the staged GPU
implementation with deterministic synthetic weights, including ring
boundaries, empty lengths, sink variants, single-rank rejection and KV rebinding
through a graph captured once. Its disabled case tests the staged fallback.
These tests need no model files or Python libraries. `naive_n05_qkv` also tests
unsupported shapes and overlapping views. Tests requiring absent hardware
return the usual skip code 77.

## BF16 DSA split attention

The native split kernel uses SM80 BF16 MMA and asynchronous copies, with FP32
QK and softmax statistics and BF16 probabilities for PV. This changes the older
intermediate BF16 rounding boundaries; tests compare against an independent
reference, rather than requiring bitwise equality with the older kernel.

Admission requires 192/128 query/value dimensions, GQA groups divisible by eight,
at most 2048 selected keys, aligned BF16 vectors, a suitable kernel image and
enough opt-in shared memory. Existing large-batch and quantized-KV paths retain
priority. Unsupported shapes or devices use the existing DSA implementation;
no environment switch is required.

Per-workspace split partials and completion counters are activation scratch,
not weight replicas. The last completing split combines the result without a
resident-grid spin. Graph pointer validation includes the completion storage;
zero-length rows, dynamic lengths and device KV-table rebinding are supported.
The caller must keep one workspace on ordered streams, as with the other decode
buffers. Graphs are prepared during startup; no request-time capture is added.

## Naive verification frontend and head reuse

SWA verification uses the same row descriptors as the causal staged path:
`verifyOffset` identifies the private candidate slot `128 + offset`, while each
row's device length limits attention to its causal prefix. The 128-slot ring is
unchanged until the existing accepted-prefix commit. On SM103, the ordered
QKV/RoPE/cache kernel accepts 1–7 BF16 rows with Q8/KV1 and 4096 input features.
Eight rows retain the faster GEMM. The tuned SWA core also resolves candidate
slots, dynamic lengths and rebound KV tables; the existing large-batch and
FP8 paths keep their priority. Requests may be ragged; unsupported shapes use
the existing projection, cache writer and attention kernels.

The SM103 DSA Front accepts 1–7 verification rows when all cache capacities
are at most 2048 and the existing BF16 Q8/KV1 geometry is satisfied. Verification
uses the original compensated GEMV projection tree; the ordinary decode Front
retains its MMA projection. Consumers wait for K/V readiness of all active row
descriptors when caches can be shared. Cooperative occupancy admission keeps
all producer CTAs resident, and causal lengths exclude future candidates.
Eight-row verification retains GEMM; sparse/long-context and larger batches
retain the staged DSA frontend. Scratch and readiness are workspace-owned;
independent requests must not use one workspace concurrently on unordered streams.

The BF16 residual/norm/head kernel now accepts 1–7 rows on SM103 with its existing
shape, alignment, image and resource checks. Other architectures retain the
original 1–2-row admission; in particular, Hopper keeps its existing multi-row
Lt path. Larger row counts keep GEMM.
The selected kernel must support its dynamic shared-memory size, including the
opt-in requirement for wider row tiles. Verification materializes the hidden
residual before exporting draft features, then writes the normalized head result
as FP32, including the existing optional peer destination. This uses the original
BF16 weights without packing. Head output tests bound relative L2 against the
ordinary norm/GEMV/BF16-to-FP32 sequence.

The whole-layer single-row SWA cooperative kernel is intentionally not used for
verification: its in-kernel ring update, deferred residual and projection arithmetic
have different contracts. The smaller frontend/core/output fusions above preserve
verification candidate ownership and hidden-feature materialization. No startup
or request-time graph policy changes are made; missing prepared graphs use eager
fallback, never recapture. No new dependency or environment switch is introduced.
