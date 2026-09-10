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
