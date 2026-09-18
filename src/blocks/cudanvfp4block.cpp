#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda-nvfp4-fused.h"
#include "devices/cuda/fastllm-cuda-native-prefill.h"
#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
#include "fastllm-cuda.cuh"
namespace fastllm {
bool CudaNvfp4LinearSwigluBlock(Data &input, Data &weight, const Data &bias, Data &middle, Data &output) {
    if (FastllmCudaNativeNvfp4FusedCanRun(input, weight, bias, output, true) &&
        FastllmCudaNativeNvfp4Fused(input, weight, output, true)) return true;
    if (FastllmCudaNvfp4FusedCanRun(input, weight, bias, output, true)) {
        FastllmCudaNvfp4Fused(input, weight, output, true);
        return true;
    }
    DoCudaLinearReshape(input, weight, middle);
    DoCudaLinear(input, weight, bias, middle);
    DoCudaSwigluReshape(middle, output);
    DoCudaSwiglu(middle, output);
    return false;
}
bool CudaNvfp4LinearAddBlock(Data &input, Data &weight, const Data &bias, Data &middle, Data &output) {
    if (FastllmCudaNativeNvfp4FusedCanRun(input, weight, bias, output, false) &&
        FastllmCudaNativeNvfp4Fused(input, weight, output, false)) return true;
    if (FastllmCudaNvfp4FusedCanRun(input, weight, bias, output, false)) {
        FastllmCudaNvfp4Fused(input, weight, output, false);
        return true;
    }
    DoCudaLinearReshape(input, weight, middle);
    DoCudaLinear(input, weight, bias, middle);
    FastllmCudaAddTo(output, middle, 1.f);
    return false;
}
} // namespace fastllm
#endif
