#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda-rmsnorm-small-linear.h"
#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "utils.h"
namespace fastllm {
namespace {
// Alias restrictions are shared by fusion and fallback. CanRun == false must
// never turn an invalid Block call into an in-place overwrite by the fallback.
void ValidateOutputs(const Data &input, const Data &norm, const Data &weight,
                     const Data &bias, const Data &normalized, const Data &output) {
    auto overlap = [](const Data &a, const Data &b) {
        if (&a == &b) return true;
        if (!a.cudaData || !b.cudaData) return false;
        uintptr_t x = reinterpret_cast<uintptr_t>(a.cudaData);
        uintptr_t y = reinterpret_cast<uintptr_t>(b.cudaData);
        return x < y + b.GetBytes() && y < x + a.GetBytes();
    };
    AssertInFastLLM(!overlap(normalized, output), "RMSNormSmallLinear outputs must not overlap.\n");
    for (const Data *out : {&normalized, &output}) {
        for (const Data *in : {&input, &norm, &weight, &bias}) {
            AssertInFastLLM(!overlap(*out, *in), "RMSNormSmallLinear output aliases input.\n");
        }
    }
}
} // namespace

bool CudaRMSNormSmallLinearOp::CanRun(const std::string &, const DataDict &d, const FloatDict &,
                                      const IntDict &) {
    return FastllmCudaRMSNormSmallLinearCanRun(*d.at("input"), *d.at("norm"), *d.at("weight"), *d.at("bias"),
                                               *d.at("normalized"), *d.at("output"));
}
void CudaRMSNormSmallLinearOp::Reshape(const std::string &, const DataDict &d, const FloatDict &,
                                       const IntDict &) {
    ValidateOutputs(*d.at("input"), *d.at("norm"), *d.at("weight"), *d.at("bias"),
                    *d.at("normalized"), *d.at("output"));
    const Data &x = *d.at("input");
    for (const char *name : {"normalized", "output"}) {
        Data &out = *d.at(name);
        out.dataType = x.dataType;
        out.UpdateUnitSize();
        out.dataDevice = DataDevice::CUDA;
        out.dataDeviceIds = x.dataDeviceIds;
        auto shape = x.dims;
        if (std::string(name) == "output")
            shape.back() = d.at("weight")->dims[0];
        out.Resize(shape);
    }
}
void CudaRMSNormSmallLinearOp::Run(const std::string &, const DataDict &d, const FloatDict &f,
                                   const IntDict &) {
    d.at("normalized")->Allocate();
    d.at("output")->Allocate();
    FastllmCudaRMSNormSmallLinear(*d.at("input"), *d.at("norm"), *d.at("weight"), *d.at("bias"),
                                  *d.at("normalized"), *d.at("output"), f.at("eps"));
}
bool CudaRMSNormSmallLinearBlock(Data &x, Data &g, Data &w, const Data &b, Data &y, Data &o, float eps) {
    DataDict d = {{"input", &x},      {"norm", &g},  {"weight", &w}, {"bias", const_cast<Data *>(&b)},
                  {"normalized", &y}, {"output", &o}};
    FloatDict f = {{"eps", eps}};
    CudaRMSNormSmallLinearOp op;
    bool fused = op.CanRun("RMSNormSmallLinear", d, f, {});
    op.Reshape("RMSNormSmallLinear", d, f, {});
    if (fused) {
        op.Run("RMSNormSmallLinear", d, f, {});
        return true;
    }
    y.Allocate();
    FastllmCudaRMSNorm(x, g, y, eps);
    DoCudaLinearReshape(y, w, o);
    DoCudaLinear(y, w, b, o);
    return false;
}
} // namespace fastllm
#endif
