This change adds a Volta-specific FP8 weight-only MMA path for FP16 activations,
FP32 per-output-row scales and 1–8 tokens. It is selected only on SM70 and does
not change stored weights, vocabulary size or persistent model memory.

The general kernel computes 32 output columns and up to 8 token rows per CTA. Eight warps split K.
Coalesced 128-byte row reads are transposed within each warp's shared-memory
tile, then fed into Volta `mma.sync.m8n8k4` with FP32 accumulation. The 33-vector
tile stride reduces shared-memory bank conflicts. Each CTA uses 41,984 bytes
of static shared memory. Scale and optional bias are applied in the epilogue.

For the GDN input shape K=5120, N=16384, the SM70 path uses
`GdnQuadSplitKernel` for all T=1..8. Each CTA produces eight output columns;
the four independent Volta quad-pairs split each 128-byte K block into adjacent
32-byte slices. This coalesces the original weights directly, removing the
shared transpose without a persistent weight copy. Shared storage is 8,192 bytes
of FP32 partial sums. This shape selects the GDN kernel automatically. The
change reorders the partial-sum reduction; it does not promise bitwise equality
with the general kernel.

`test.cu` tests both kernels against the same independent FP64 oracle, including
finite FP8 codes, bias, tails, uneven K partitions and T=1..8 parity.
It also exercises the production GDN shape K=5120, N=16384 through `Try()`,
with the environment override unset, for every T=1..8 with and without bias.
All outputs are checked against an independent FP64 reference. Exact-sized
input allocations and output sentinels check token bounds; both `0` and
`false` fallback settings must reject the call without modifying output.

The companion paged-attention optimization is guarded by
`FASTLLM_PAGED_SM70_SMALL_T` (default on). It covers single-sequence FP16
Q/K/V, head_dim=256, GQA group=6, T=2..8 and cache lengths up to 4096 on SM70.
It reuses paged KV directly in a fused split softmax/PV kernel and merges each
query head separately. Longer eager contexts and other shapes retain their
previous dispatch. Set this flag to 0 for controlled comparisons. See
`test/basic/test_cuda_paged_sm70_small_t.cpp` for independent numeric checks,
fragmented page lists, causal-tail sentinels, local KV head counts 1/2/4, graph
capture and microbenchmarks. These local shapes do not test TP communication.

`FASTLLM_CUDA_FP8_SM70=0` restores the previous FP8 dispatch. The default is on.
`FASTLLM_CUDA_FP8_SMALL_T` still controls the existing generic small-T path.
Other GPUs, block-scaled weights, FP32/BF16 activations, larger token batches,
unaligned input views and overlapping buffers keep their existing paths.
K must be divisible by 128 and at least 1024. Tail output columns are supported.
Dispatch also verifies that the loaded kernel was compiled for SM70; a legacy
PTX-only build must fall back instead of executing an empty architecture stub.

The calculation uses FP16 MMA operands and FP32 accumulation before applying
the row scale. It is not bitwise identical to the previous native multi-row
GEMV, which scales and rounds intermediate FP16 weights/products. The test
compares against an independent FP64 reference and checks exact first-row
parity between T=1 and all supported batched widths. Finite E4M3 weights are
assumed, consistent with the source quantized model.

From the repository root, compile with CUDA 12.x on a V100:

```bash
nvcc -O3 -std=c++17 -arch=sm_70 --default-stream=per-thread \
  -Iinclude tools/cuda_fp8_sm70/test.cu -o /tmp/fastllm-fp8-sm70-test
/tmp/fastllm-fp8-sm70-test
compute-sanitizer --tool memcheck --error-exitcode 4 /tmp/fastllm-fp8-sm70-test
compute-sanitizer --tool synccheck --error-exitcode 4 /tmp/fastllm-fp8-sm70-test
```

Use `--production-only` to isolate the 16 real-shape `Try()` calls for profiling:

```bash
nsys profile --trace=cuda --sample=none --cpuctxsw=none -o /tmp/fp8-gdn-dispatch \
  /tmp/fastllm-fp8-sm70-test --production-only
nsys stats --report cuda_gpu_kern_sum /tmp/fp8-gdn-dispatch.nsys-rep
```

The kernel summary should contain 16 `GdnQuadSplitKernel` launches and no
`RowKernel` launches. This checks the actual launch selected by `Try()`; numeric
agreement alone cannot distinguish two valid kernel implementations.

NVCC 12.8 builds and V100 memcheck/synccheck pass. Racecheck reports
`Device not supported` on the test host, including for an intentional-race
control; it is not counted as a pass. Attention integration tests require CUDA
initialization before loading FastLLM's static constructors under the sanitizer.

Build the library with `-DUSE_CUDA=ON -DCUDA_ARCH=70`. Provenance and the upstream
Apache-2.0 / llama.cpp MIT licenses are retained under
`third_party/ninfer-fp8-sm70/`.

For Qwen3.8-27B on V100 with FP16 activations, the runtime selects these paths
without enabling environment variables. The same local FP8 dispatch can serve
TP shards, but the dedicated GDN shape is single-GPU-only under ordinary TP.
The small-T attention test covers 1/2/4 local KV heads; it does not establish
end-to-end multi-GPU speedups.

To reproduce the eager, CPU-embedding configuration used for validation in a
fresh shell (without inherited tuning overrides):

```bash
FASTLLM_CUDA_GRAPH=0 FASTLLM_GPU_TOKEN_HANDOFF=0 FASTLLM_CUDA_FP8_SMALL_T=0 \
  ftllm server /path/to/Qwen3.8-27B-NVFP4 --max_batch 1 --mtp 5 \
  --kv_cache_dtype float16 --tokens 69632 --chunked_prefill_size 512
```

The generic `FASTLLM_CUDA_FP8_SMALL_T` override is retained for its other
callers, including fused LinearAdd; it is independent of the SM70 kernels.
Draft NVFP4, short attention and eligible SM70 direct KV reads are already on
by default. The short-attention and direct-KV optimizations require FP16 KV;
FP8 KV remains on its existing dispatch.
The new four-KV-head direct-read path is restricted to SM70 even when
`FASTLLM_PAGED_CUBLAS_LINEAR_KV=1`; other architectures retain the existing
two-head eligibility and fragmentation budget.
