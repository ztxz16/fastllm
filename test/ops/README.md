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
