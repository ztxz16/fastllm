// Test-only ctypes adapter. Production exposes only the existing-style Data API.
#include "fastllm.h"
#include <cuda_runtime_api.h>

bool FastllmCudaDeepGemmLinearFp8Sm90(const fastllm::Data &input, fastllm::Data &weight,
    const fastllm::Data &bias, fastllm::Data &output, int n, int m, int k);

extern "C" void *Fp8Sm90TestStream() {
    return cudaStreamPerThread;
}

extern "C" bool Fp8Sm90TestLinear(
        void *x, void *w, void *scales, void *b, void *y,
        int rows, int cols, int outCols, bool bf16, int invalid) {
    using namespace fastllm;
    const DataType dtype = bf16 ? DataType::BFLOAT16 : DataType::FLOAT16;
    Data input(dtype, {rows, cols}), weight(DataType::FP8_E4M3, {outCols, cols});
    Data bias(DataType::FLOAT32);
    if (b) bias.Resize({outCols});
    Data output(dtype, {rows, outCols});
    for (Data *data : {&input, &weight, &bias, &output}) {
        data->isFake = true;  // PyTorch owns all GPU storage.
        data->dataDevice = DataDevice::CUDA;
    }
    input.cudaData = x;
    weight.cudaData = w;
    bias.cudaData = b;
    output.cudaData = y;
    weight.blockM = weight.blockK = 128;
    weight.scales.resize((size_t)(cols / 128) * (outCols / 128));
    weight.extraCudaData = {scales, b};
    switch (invalid) {
        case 1: weight.blockM = 64; break;
        case 2: weight.scales.clear(); break;
        case 3: weight.dims[0] += 128; break;
        case 4:  // Mark an already packed Marlin layout without reading it.
            weight.extraCudaData.resize(4);
            weight.extraCudaData[3] = w;
            weight.extraCudaHalfData = {nullptr, scales};
            break;
        case 5: input.dataType = DataType::FLOAT32; break;
        case 6: output.dataType = DataType::FLOAT32; break;
        case 7: bias.dataType = DataType::BFLOAT16; break;
        case 8: input.dataDevice = DataDevice::CPU; break;
        case 9: weight.extraCudaData[0] = nullptr; break;
        case 10: weight.extraCudaData.clear(); break;  // Unwarmed capture.
    }
    return FastllmCudaDeepGemmLinearFp8Sm90(input, weight, bias, output, rows, cols, outCols);
}
