# SM70 FP8 TP deadlock regression

`fp8TpRepackDeadlockRegression` requires two visible SM70 GPUs (for example,
two V100s), CUDA, and NCCL. It runs without CUDA Graph. Exit code 77 means the
required GPUs or compiled SM70 FP8 kernels are unavailable.

Build from the repository root:

```bash
cmake -S . -B build-sm70-tests -DUSE_CUDA=ON -DCUDA_ARCH=70 -DUNIT_TEST=ON
cmake --build build-sm70-tests --target fp8TpRepackDeadlockRegression -j8
```

Run each case in a fresh process so the per-device GEMM runtime starts cold.
Use an external timeout because the regression deliberately exercises waits
inside CUDA/NCCL that cannot be cancelled by a test thread:

```bash
set -e
for mode in repack runtime warmup; do
    for rank in 0 1; do
        CUDA_VISIBLE_DEVICES=0,1 FASTLLM_CUDA_GRAPH=0 \
            timeout --kill-after=3s 20s \
            ./build-sm70-tests/fp8TpRepackDeadlockRegression "$mode" "$rank"
    done
done
```

| Mode | Scenario |
| --- | --- |
| `repack` | Lazy FP8 repack while one rank has queued an unmatched AllReduce. |
| `runtime` | Cold GEMM runtime initialization after weights are already packed. |
| `warmup` | Explicit weight preparation during synchronous startup, followed by cold GEMM with asynchronous NCCL. |

The second argument selects which rank queues the collective first. NCCL's
lazy transports are warmed before the asymmetric ordering is imposed. Each
case checks exact FP8 output with distinct data on both devices, AllReduce's
result, and eventual completion of any deferred repack. The warmup case also
checks unsupported row counts and repeated preparation of the same weight.

The original cross-device lock deadlocks in the `repack` and `runtime` cases
and is expected to hit the timeout (exit 124). A fixed build prints `PASS` and
exits 0 for all six cases.

# SM75 FP8 prefill dispatch regression

The SM75 packed FP8 block128 path borrows the existing FlashInfer float workspace
for dequantized FP16 weights. The separate 8 MiB cuBLAS workspace is created in
synchronous warmup, before the KV page budget is calibrated. The float arena is
shared under the existing per-device worker stream ordering; it does not provide
concurrent independent stream leases. The int arena retains attention plans.

`M` below is the current Linear input row count, not total context length:

| Condition | Backend |
| --- | --- |
| `M=1` | Existing GEMV |
| `M<1024`, or either matrix dimension `N`/`K` is below 1024 | Existing Marlin |
| Full FP16 weight fits the float workspace, or sliced workspace is at least 64 MiB | cuBLAS for `M>=1024` |
| Weight needs slices and workspace is 32–63 MiB | cuBLAS for `M>=1536` |
| Weight needs slices below 32 MiB, or the maximum N slice capacity is below 1024 rows | Marlin |
| CUDA Graph capture with `M>1` | Marlin |

Runtime SM must be exactly 75; other architectures keep their existing dispatch. Set
`FASTLLM_CUDA_FP8_PREFILL_CUBLAS=0` to retain the previous SM75 path. FP16 cuBLAS
accumulation is not bitwise equivalent to Marlin; tests compare each selected
backend to its own independent reference, including rounded FP8 block scales.
The final N slice may be smaller than 1024 rows. The row threshold was selected
using the complete dequantization-plus-GEMM time on representative SM75 matrices;
it is not a model-name or exact-shape special case.

Build and run from the repository root:

```bash
cmake -S . -B build-sm75-tests -DUSE_CUDA=ON -DCUDA_ARCH=75 -DUNIT_TEST=ON
cmake --build build-sm75-tests --target cudaFp8Sm75PrefillRegression -j8
ctest --test-dir build-sm75-tests -R '^cuda_fp8_sm75_prefill_' --output-on-failure
```

The regression checks actual routing by poisoning the float workspace and
observing whether inference overwrote it, then compares output bits with the
selected Marlin or CPU-dequant-plus-cuBLAS reference. It covers both sides of
the 1024/1536 row boundaries, bias, K tails, N slices, small matrices, the disable
switch, workspace address/capacity stability, and graph capture. The 4/32/64/256
MiB configurations run in separate processes; exit 77 skips non-SM75 GPUs.

# SM75 NVFP4 prefill dispatch regression

For Marlin-repacked NVFP4 block16 weights, SM75 uses dequantization plus cuBLAS
when the current Linear row count `M>=2048`, `N>=1024`, `K>=1024`, and the full
logical `N*K*sizeof(half)` weight fits the existing FlashInfer float workspace.
Otherwise it retains Marlin, including graph capture. The threshold applies to
the current prefill chunk, not total context length. Set
`FASTLLM_CUDA_NVFP4_PREFILL_CUBLAS=0` to disable this route.

The 64-by-256 transpose kernel reconstructs Marlin's normalized FP16 weights
exactly, including its S0E5M3 block scales. GEMM retains **FP32 accumulation**
and applies the existing **FP32 global scale after accumulation**. The tensor
scale is not folded into FP16 weights. Output is not guaranteed bitwise equal
to Marlin because GEMM reduction order differs; the regression checks both an
independent CPU-dequant-plus-FP32-cuBLAS reference and error against Marlin.
Logical N can be unaligned (for example 8240); padded rows are not written.

FP8 and NVFP4 share one per-device cuBLAS handle and 8 MiB workspace, plus one
FP32 zero scalar. These are initialized during synchronized warmup before KV
budget calibration. The float arena retains its original address and capacity;
no full weight cache or per-layer FP16 allocations are retained. Shared-arena
use requires the existing per-device worker stream ordering described above.
Host/device scalar modes are explicitly switched when alternating FP8/NVFP4.

```bash
cmake -S . -B build-sm75-tests -DUSE_CUDA=ON -DCUDA_ARCH=75 -DUNIT_TEST=ON
cmake --build build-sm75-tests --target cudaNvfp4Sm75PrefillRegression cudaFp8Sm75PrefillRegression -j8
ctest --test-dir build-sm75-tests -R '^cuda_(nvfp4|fp8)_sm75_prefill_' --output-on-failure
```

The NVFP4 regression covers eight shapes, M=4/1024/2047/2048/2049/4096, bias,
logical-N padding and K tails, all finite nonnegative FP8 block-scale codes, zero
and small/large signed inputs, alternating FP8/NVFP4, graph capture with the
registered FastLLM memory pool, and stable scratch address/capacity. Fresh
processes test 4/32/256 MiB arenas and the disable switch. Exit 77 skips non-SM75
GPUs. An optional device index selects the second card; `TEST_N` restricts N
for a targeted Compute Sanitizer run.

## Optional NVFP4 FP16 accumulation

Set `FASTLLM_CUDA_NVFP4_PREFILL_FP16_ACCUM=1` **before starting the process** to
use FP16 cuBLAS accumulation in eligible SM75 NVFP4 prefills. The default is
**off (FP32 accumulation)**. This flag retains all existing dispatch guards:
M>=2048, N/K>=1024, full FP16 weight fitting the float arena, and no graph
capture. `FASTLLM_CUDA_NVFP4_PREFILL_CUBLAS=0` still disables the entire route.
It does not affect FP8's accumulation policy.

The optional path uses FP16 GEMM with alpha=1 and beta=0, then applies the
existing FP32 tensor scale in a separate kernel. Bias is fused into this
kernel after rounding the scaled result to FP16, matching the existing bias
contract. This avoids rounding the global scale itself to FP16 and requires
no additional buffer. FP16 accumulation still loses precision and has a
smaller numerical range; it is not numerically equivalent to the FP32 path.

The added `cuda_nvfp4_sm75_prefill_fp16_*` CTest cases exercise 4/32/256 MiB
arenas and the master disable switch. They compare the selected FP16 backend
against an independent CPU-dequant, FP16-cuBLAS, CPU-FP32-scale/bias reference;
Marlin relative L2 is bounded at 1% for the constructed input data. Default
FP32 cases retain their tighter 0.02% bound. N=4161 additionally exercises
pairs crossing rows and an odd final output element with and without bias.
