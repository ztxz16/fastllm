# GGUF shared-memory decode and verification regression

`cudaGgufIq3SharedRegression.cu` checks nine formats (Q2_K, Q4_K, IQ1_M,
IQ2_S/XS/XXS, IQ3_S/XXS and IQ4_XS) against independent llama.cpp CPU-decoded
weights and a CPU Q8_1 activation reference. The historical IQ3 filename also
covers the shared small-batch implementation in `fastllm-gguf-small-mmvq.cuh`.

## Dispatch and portability

| Path | Input tokens | Conditions |
| --- | --- | --- |
| IQ3 single-token shared kernel | 1 | M>=128, K in [256,18432], K divisible by 256 |
| Shared small-batch MMVQ, all nine formats | 2–8 | M>=128, K>=256 divisible by 256 |
| BF16-weight small projections | 8 | M<=1024 |

M is the number of output rows, K the input width. GGUF shared paths require
ordinary non-indexed matrices, 16-byte-aligned Q8_1 inputs, input stride >=K
and divisible by 256, and output stride >=M. Other shapes retain existing
kernels. No model names, fixed hidden dimensions or compute-capability
whitelist select these paths. They use CUDA warp/DP4A primitives; ROCm does
not use the new shared kernels. Compilation on another CUDA architecture is
not evidence of a speedup on that GPU.

The single-token K limit bounds its full-input shared cache. Multi-token
kernels cache fixed K=1024 tiles and have no corresponding K upper limit;
maximum shared storage is 25,600 bytes per block (IQ1_M, eight tokens).
IQ3 uses two output rows per warp at M>=4096, with eight warps per block.
Other formats use sixteen output warps for M>=4096 and batches >=5 (>=4
for Q4_K), otherwise eight. These thresholds control scheduling.

There are no runtime environment switches for these shared kernels.
`forceGGUFFp32Dequant` remains a weight safety flag for prefill and unsupported
paths; supported 2–8-token MMVQ shapes bypass forced dequantization.

## Numerical behavior

Weights decode once per lane and are reused across input tokens. IQ2 and
IQ3_XXS retain the original integer scaling and truncating divisions. IQ1_M
represents its signed codebook and +/-1/8 offsets exactly as packed integers
scaled by eight. IQ4_XS uses byte permutations for its sixteen-entry codebook.
Q2_K/Q4_K cache exact sums of quantized input values; Q8_1.ds.y contains the
rounded pre-quantization sum and cannot replace those sums.

Floating-point reduction order can differ from legacy MMVQ. The CPU reference
includes the existing IQ2 integer-truncation error bound, less than
`abs(weight_block_d * q8_group_d)` per 32-value group. Fused IQ3 SiLU/multiply
keeps the FP16 rounding sequence. BF16-weight projections retain input/output
BF16 rounding and add bias after rounding in the eight-token GEMM replacement.
No bitwise equivalence to all legacy GEMM/MMVQ paths is implied.

For cancellation, the absolute FP32 tolerance scales with the accumulation
length using `gamma_n = n*u/(1-n*u)`, where `u=FLT_EPSILON/2` and
`n=ceil(K/1024)+8` includes per-lane accumulation, scale products and warp
reduction. Relative tolerances remain unchanged; a fixed absolute multiplier
is inadequate for long cancelling sums even in the legacy implementation.

## Reproduce the checks

Generate fixtures with `prepareGgufDirectGemvFixtures.py`, following
`cudaGgufDirectGemvRegression.md`. Add `--extra-columns 18688 32768 33024` to
cover widths above the old cap, full input tiles and partial final input tiles.
These extra rows repeat valid quantization blocks and are decoded independently
by llama.cpp, rather than using the CUDA decoder as their arithmetic reference.

From the repository root, set BUILD_DIR, LIB_DIR, FIXTURE_DIR, TEST_GPU and
CUDA_ARCH for the build under test (for example CUDA_ARCH=sm_75):

```bash
nvcc -O3 -std=c++20 -arch="$CUDA_ARCH" -DUSE_CUDA -DUSE_NUMAS \
  --default-stream=per-thread \
  --options-file "$BUILD_DIR/CMakeFiles/fastllm_tools.dir/includes_CUDA.rsp" \
  -Isrc/devices/cuda test/ops/cudaGgufIq3SharedRegression.cu \
  -L"$LIB_DIR" -lfastllm_tools -Xlinker -rpath -Xlinker "$LIB_DIR" \
  -o "$FIXTURE_DIR/test-shared"
for tokens in 1 2 3 4 5 6 7 8; do
  CUDA_VISIBLE_DEVICES="$TEST_GPU" "$FIXTURE_DIR/test-shared" \
    "$FIXTURE_DIR/gemv-cases.bin" "$tokens"
done

# Compile kernels directly: the full runtime may initialize CUDA before
# Compute Sanitizer installs its hooks.
nvcc -O3 -std=c++20 -arch="$CUDA_ARCH" -lineinfo --default-stream=per-thread \
  --options-file "$BUILD_DIR/CMakeFiles/fastllm_tools.dir/includes_CUDA.rsp" \
  -Isrc/devices/cuda test/ops/cudaGgufIq3SharedMemcheck.cu \
  -o "$FIXTURE_DIR/test-memcheck"
CUDA_VISIBLE_DEVICES="$TEST_GPU" compute-sanitizer --tool memcheck \
  --error-exitcode 1 "$FIXTURE_DIR/test-memcheck" "$FIXTURE_DIR/gemv-cases.bin"
```

The public-entry test covers FP16/FP32/BF16, random/cancellation/zero inputs,
distinct tokens, the safety flag on/off, 1/7-row fallbacks, incomplete output
blocks, output guards, IQ3 gate/up and codebook signs. Batch 8 on Blackwell
can select MMQ when the safety flag is off, so bitwise flag equality is not
required there. Also run `cudaGgufFastPathRegression.cu` for the other formats
and the nine-token fallback.

The standalone memory test checks all 2–8 token counts, padded strides,
8/16-row scheduling, IQ3 paired rows, partial input/output tiles, output guards
and CUDA Graph replay. It compares batches with separate single-token calls;
this shares some arithmetic and supplements the independent CPU test. Wide
fixtures include M=4097 at K=18688. It also verifies 4096 IQ4 lookup words.

`cudaBf16SmallBatchRegression.cu` uses double-precision CPU dots and the public
entry's rounding rules. Its 900 cases cover 1/6/7/8/9 tokens, K=33/256/5120,
M=1/96/129/1024/1025, both bias states and output guards. Compile using the same
public-entry arguments above, substituting its source and binary name. Run
without arguments for correctness or with `micro` for CUDA-event timing.
