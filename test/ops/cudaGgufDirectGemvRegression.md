Direct GGUF GEMV regression
==========================

`cudaGgufDirectGemvRegression.cu` checks all 17 ordinary GGUF formats accepted
by `FastllmGgufDirectGemv`. Its reference is llama.cpp's CPU dequantization,
followed by rounding weights/activations to the tested dtype and a double
precision dot product. It covers FP16, BF16, FP32, narrow/partial blocks,
odd output sizes, zero/random/cancellation inputs, output guards and CUDA
Graph replay. R4 repacked layouts are deliberately excluded from this
ordinary-block dispatcher.

Generate fixtures from a local GGUF model and a built llama.cpp library:

```bash
PYTHONPATH="$LLAMA_DIR/gguf-py" python3 test/ops/prepareGgufDirectGemvFixtures.py \
  "$MODEL" "$LLAMA_BUILD/bin/libggml-base.so" "$FIXTURE_DIR"
```

The generator samples actual model rows. For formats absent from the model,
it constructs finite packed blocks. Fixture binaries stay outside the source
tree. The generator requires NumPy and llama.cpp's `gguf` Python package.

From the FastLLM repository root, compile against the CUDA-enabled build;
adjust the GPU architecture to the test GPU:

```bash
nvcc -O3 -std=c++20 -arch=sm_75 --default-stream=per-thread \
  --options-file "$BUILD_DIR/CMakeFiles/fastllm_tools.dir/includes_CUDA.rsp" \
  -Isrc/devices/cuda test/ops/cudaGgufDirectGemvRegression.cu \
  -L"$LIB_DIR" -lfastllm_tools -Xlinker -rpath -Xlinker "$LIB_DIR" \
  -o "$FIXTURE_DIR/test-gemv"
CUDA_VISIBLE_DEVICES="$TEST_GPU" "$FIXTURE_DIR/test-gemv" "$FIXTURE_DIR/gemv-cases.bin"
```

Also run the existing `cudaGgufChunkedDequantRegression.cu` against the new
library: the standalone matrix-dequantization kernels share block decoders
with GEMV, and prefill/chunked/R4 behavior must continue to pass.
Record failures also reproduced with the baseline library separately from
regressions introduced by the change.

For Compute Sanitizer, build the same test with
`-DFASTLLM_GGUF_GEMV_STANDALONE` and link `-lggml-base` from llama.cpp instead
of `-lfastllm_tools`. This isolates the new GEMV kernels from the full runtime's
CUDA static initialization, which can run before sanitizer initialization.
The ordinary build additionally validates the runtime's standalone dequant
kernels; the sanitizer build uses the same independent fixtures and GEMV
checks. Run with `compute-sanitizer --tool memcheck --error-exitcode 99` and
require `ERROR SUMMARY: 0 errors`.
