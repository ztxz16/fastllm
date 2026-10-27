# IQ2 shared-memory decode regression

`cudaGgufIq2SharedRegression.cu` validates IQ2_XS, IQ2_XXS, and IQ2_S
single-token CUDA GEMV against independent llama.cpp CPU-decoded weights
and CPU Q8_1 activation quantization. The CPU reference reconstructs signed
codebook values from those independently decoded weights and includes the
existing IQ2 integer divisions (rounding toward zero); a direct floating-point
dot product omits that truncation. Generate the fixtures using
`prepareGgufDirectGemvFixtures.py`, as described in
`cudaGgufDirectGemvRegression.md`.

Coverage includes FP16/FP32/BF16, random/cancellation/zero input, 1/7/127-row
fallbacks, the 128-row dispatch boundary, incomplete final blocks (129/1031
rows), output guards, and both settings of the prefill safety flag. Fused
SiLU gate/up checks use different gate/up weights and compare separate and
merged storage. Ordinary and fused outputs are also compared bit-for-bit
with duplicate-input batch-two MMVQ, which retains the four-warp path. All IQ2 codebooks are exhaustively checked for packed-byte
negation, and every 7-bit sign code is checked against both sign tables.

Set `BUILD_DIR`, `LIB_DIR`, `FIXTURE_DIR`, `TEST_GPU`, and `CUDA_ARCH` for
the intended build, library, fixtures, GPU, and target architecture:

```bash
nvcc -O3 -std=c++20 -arch="$CUDA_ARCH" -DUSE_CUDA -DUSE_NUMAS \
  --default-stream=per-thread \
  --options-file "$BUILD_DIR/CMakeFiles/fastllm_tools.dir/includes_CUDA.rsp" \
  -Isrc/devices/cuda test/ops/cudaGgufIq2SharedRegression.cu \
  -L"$LIB_DIR" -lfastllm_tools -Xlinker -rpath -Xlinker "$LIB_DIR" \
  -o "$FIXTURE_DIR/test-iq2-shared"
CUDA_VISIBLE_DEVICES="$TEST_GPU" "$FIXTURE_DIR/test-iq2-shared" \
  "$FIXTURE_DIR/gemv-cases.bin"

# Test production kernels without linking the full CUDA-initializing runtime.
nvcc -O3 -std=c++20 -arch="$CUDA_ARCH" -lineinfo --default-stream=per-thread \
  --options-file "$BUILD_DIR/CMakeFiles/fastllm_tools.dir/includes_CUDA.rsp" \
  -Isrc/devices/cuda/gguf_mmq test/ops/cudaGgufIq2SharedMemcheck.cu \
  -o "$FIXTURE_DIR/test-iq2-memcheck"
CUDA_VISIBLE_DEVICES="$TEST_GPU" compute-sanitizer --tool memcheck \
  --error-exitcode 1 "$FIXTURE_DIR/test-iq2-memcheck" \
  "$FIXTURE_DIR/gemv-cases.bin"
```

The standalone test additionally exercises 1/7/33/127/128/129/1031 output
rows directly in the new kernel, including sizes normally handled by the
fallback. Also run `cudaGgufFastPathRegression.cu` and the existing IQ3
regression, then compare model generation/logits and profile graph replay.
For K-boundary coverage, include 12288/18432/18688-column fixtures; the
memory checker skips inputs beyond the optimized dispatch cap, while the
public-entry regression exercises their fallback.

The optimization applies only to batch 1, M >= 128, 256 <= K <= 18432,
K divisible by 256, and 16-byte-aligned Q8_1 input. It has no SM whitelist.
The largest IQ2_S codebook plus input occupies at most 28,928 shared bytes
per block. Other shapes/batches retain the existing kernels.

Four per-thread accumulators preserve the established four-warp summation
order, followed by the same warp reduction. IQ2 integer scaling and
truncating division are preserved, as is the existing FP16 rounding
sequence for fused SiLU. Validate equality to the existing path as well
as the independent CPU reference; do not assume bitwise equivalence on
unmeasured compilers or hardware.
