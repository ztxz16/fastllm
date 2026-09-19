# IQ3 shared-memory decode regression

`cudaGgufIq3SharedRegression.cu` validates the CUDA single-token IQ3_S and
IQ3_XXS kernels against independent llama.cpp CPU-dequantized fixture weights
and a CPU Q8_1 activation reference. Generate the input fixtures with
`prepareGgufDirectGemvFixtures.py` as described in
`cudaGgufDirectGemvRegression.md`.

The test expands fixture rows to 129 or 1031, exercising the shared-memory dispatch
and incomplete final CUDA blocks. It checks:

- FP16, FP32 and BF16 outputs, with a tighter error bound for FP32;
- random, cancellation and zero inputs;
- identical results with the prefill safety flag on and off for batch 1;
- 1/7-row fallback calls and full-row shared-memory calls, including output guards;
- fused gate/up with different gate and up weights, and equality between merged
  and separate weight storage;
- all IQ3 codebook entries with all 16 four-byte sign patterns, and all 128
  IQ3_XXS parity/sign encodings.

From the repository root, with `BUILD_DIR`, `LIB_DIR`, `FIXTURE_DIR` and
`TEST_GPU` set to the intended build/runtime/test device, and `CUDA_ARCH` set
to the target CUDA architecture (for example `sm_75`, `sm_80`, or `sm_90`):

```bash
nvcc -O3 -std=c++20 -arch="$CUDA_ARCH" -DUSE_CUDA -DUSE_NUMAS \
  --default-stream=per-thread \
  --options-file "$BUILD_DIR/CMakeFiles/fastllm_tools.dir/includes_CUDA.rsp" \
  -Isrc/devices/cuda test/ops/cudaGgufIq3SharedRegression.cu \
  -L"$LIB_DIR" -lfastllm_tools -Xlinker -rpath -Xlinker "$LIB_DIR" \
  -o "$FIXTURE_DIR/test-iq3-shared"
CUDA_VISIBLE_DEVICES="$TEST_GPU" "$FIXTURE_DIR/test-iq3-shared" \
  "$FIXTURE_DIR/gemv-cases.bin"

# Compile the production kernels directly for memory checking. Linking the
# full runtime can initialize CUDA before Sanitizer installs its hooks.
nvcc -O3 -std=c++20 -arch="$CUDA_ARCH" -lineinfo --default-stream=per-thread \
  --options-file "$BUILD_DIR/CMakeFiles/fastllm_tools.dir/includes_CUDA.rsp" \
  -Isrc/devices/cuda test/ops/cudaGgufIq3SharedMemcheck.cu \
  -o "$FIXTURE_DIR/test-iq3-memcheck"
CUDA_VISIBLE_DEVICES="$TEST_GPU" compute-sanitizer --tool memcheck \
  --error-exitcode 1 "$FIXTURE_DIR/test-iq3-memcheck" \
  "$FIXTURE_DIR/gemv-cases.bin"
```

Also run `cudaGgufFastPathRegression.cu` against the same runtime, check model
generation/logits, and profile CUDA Graph replay. The optimized dispatch is
limited to batch 1, ordinary non-indexed matrices, at least 128 output
rows, and K in [256, 18432] divisible by 256 with aligned Q8_1 input. Other
cases retain the existing path. It does not alter prefill or quantization.
There is no compute-capability whitelist. The K cap bounds input and codebook
shared memory to 22,784 bytes per block. Performance measurements on SM75 do
not establish the speedup on other GPUs.

The new warp reduction changes floating-point summation order; FP32 results
and model logits need not be bitwise identical. Fused SiLU/multiply retains
the existing FP16 rounding sequence.
