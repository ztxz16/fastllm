// Test-only adapter for the production FP8 and FP16 Data dispatch paths.
#include "fp8Sm90TestBridge.cpp"

extern "C" bool FastllmCudaHalfMatMulFloat16WithRouterSpecialization(
    const fastllm::Data &input, fastllm::Data &weight, const fastllm::Data &bias,
    fastllm::Data &output, int n, int m, int k, bool addTo, bool allowRouterSpecialization);

static bool RunFp16(void *x, void *w, void *b, void *y,
                    int rows, int cols, int outCols, bool addTo) {
    using namespace fastllm;
    Data input(DataType::FLOAT16, {rows, cols});
    Data weight(DataType::FLOAT16, {outCols, cols});
    Data bias(DataType::FLOAT32), output(DataType::FLOAT16, {rows, outCols});
    if (b) bias.Resize({outCols});
    for (Data *data : {&input, &weight, &bias, &output}) {
        data->isFake = true;  // PyTorch owns storage, including the cached half bias.
        data->dataDevice = DataDevice::CUDA;
    }
    input.cudaData = x; weight.cudaData = w; output.cudaData = y;
    // The Data entry point consumes the already prepared FP16 bias cache.
    weight.extraCudaData = {nullptr};
    weight.extraCudaHalfData = {b};
    return FastllmCudaHalfMatMulFloat16WithRouterSpecialization(
        input, weight, bias, output, rows, cols, outCols, addTo, false);
}

extern "C" bool HopperTestFp16(void *x, void *w, void *y, int rows, int cols, int outCols) {
    return RunFp16(x, w, nullptr, y, rows, cols, outCols, false);
}

namespace fastllm {
extern "C" int FastllmCudaGetLinearExactBatchThreshold();
extern "C" void FastllmCudaSetLinearExactBatchThreshold(int threshold);
}

extern "C" int HopperTestSetExactThreshold(int threshold) {
    const int previous = fastllm::FastllmCudaGetLinearExactBatchThreshold();
    fastllm::FastllmCudaSetLinearExactBatchThreshold(threshold);
    return previous;
}

extern "C" bool HopperTestFp16Fallback(void *x, void *w, void *b, void *y,
                                      int rows, int cols, int outCols, bool addTo) {
    return RunFp16(x, w, b, y, rows, cols, outCols, addTo);
}

// BF16 activations with the FP16 lm_head weights used by the model loader.
extern "C" bool FastllmCudaBFloat16MatMulFloat16(
    const fastllm::Data &, fastllm::Data &, const fastllm::Data &,
    fastllm::Data &, int, int, int);

extern "C" bool HopperTestBf16Fp16(void *x, void *w, void *b, void *y,
                                   int rows, int cols, int outCols) {
    using namespace fastllm;
    Data input(BFLOAT16, {rows, cols}), weight(FLOAT16, {outCols, cols});
    Data bias(FLOAT32), output(BFLOAT16, {rows, outCols});
    if (b) bias.Resize({outCols});
    for (Data *data : {&input, &weight, &bias, &output}) {
        data->isFake = true;
        data->dataDevice = DataDevice::CUDA;
    }
    input.cudaData = x; weight.cudaData = w; output.cudaData = y;
    weight.extraCudaData = {nullptr, b}; // Already prepared BF16 bias cache.
    return FastllmCudaBFloat16MatMulFloat16(
        input, weight, bias, output, rows, cols, outCols);
}
