#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda-gdn.h"
#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
#include "utils.h"
namespace fastllm {
namespace {
const Data *Slots(const DataDict &d) {
    auto it = d.find("slotIds");
    return it == d.end() ? nullptr : it->second;
}
void OutputLike(Data &out, const Data &input, const std::vector<int> &shape) {
    out.dataType = input.dataType;
    out.UpdateUnitSize();
    out.dataDevice = DataDevice::CUDA;
    out.dataDeviceIds = input.dataDeviceIds;
    out.Resize(shape);
}
} // namespace
bool CudaGdnInputConvOp::CanRun(const std::string &, const DataDict &d, const FloatDict &, const IntDict &p) {
    return FastllmCudaGdnInputConvCanRun(*d.at("input"), *d.at("weight"), *d.at("bias"), *d.at("convWeight"),
                                         *d.at("convBias"), *d.at("cache"), Slots(d), p.at("batch"));
}
void CudaGdnInputConvOp::Reshape(const std::string &, const DataDict &d, const FloatDict &,
                                 const IntDict &p) {
    int b = p.at("batch"), c = d.at("convWeight")->dims[0], z = d.at("weight")->dims[0] - c;
    OutputLike(*d.at("output"), *d.at("input"), {1, b, c});
    OutputLike(*d.at("z"), *d.at("input"), {1, b, z});
}
void CudaGdnInputConvOp::Run(const std::string &, const DataDict &d, const FloatDict &, const IntDict &p) {
    d.at("output")->Allocate();
    d.at("z")->Allocate();
    FastllmCudaGdnInputConv(*d.at("input"), *d.at("weight"), *d.at("bias"), *d.at("convWeight"),
                            *d.at("convBias"), *d.at("cache"), Slots(d), *d.at("output"), *d.at("z"),
                            p.at("batch"));
}
bool CudaGdnInputConvBlock(Data &input, Data &weight, const Data &bias, const Data &cw, const Data &cb,
                           Data &cache, const Data *slots, Data &out, Data &z, Data &scratch, int batch) {
    // These are the Block's common contract, including the unfused fallback.
    // Unsupported fusion metadata returns false in CanRun before any cache write.
    AssertInFastLLM(FastllmCudaGdnInputConvValidInputs(input, weight, bias, cw, cb, cache, slots, batch),
                    "Invalid CUDA GDN input/conv Block contract.\n");
    DataDict d = {{"input", &input},
                  {"weight", &weight},
                  {"bias", const_cast<Data *>(&bias)},
                  {"convWeight", const_cast<Data *>(&cw)},
                  {"convBias", const_cast<Data *>(&cb)},
                  {"cache", &cache},
                  {"output", &out},
                  {"z", &z}};
    if (slots)
        d["slotIds"] = const_cast<Data *>(slots);
    IntDict p = {{"batch", batch}};
    CudaGdnInputConvOp op;
    bool fused = op.CanRun("GdnInputConv", d, {}, p);
    op.Reshape("GdnInputConv", d, {}, p);
    if (fused) {
        op.Run("GdnInputConv", d, {}, p);
        return true;
    }
    // Fallback never invokes the fused kernel and owns the sole cache update.
    OutputLike(scratch, input, input.dims);
    DoCudaLinearReshape(input, weight, scratch);
    DoCudaLinear(input, weight, bias, scratch);
    out.Allocate();
    z.Allocate();
    FastllmCudaGdnProjectedConv(scratch, cw, cb, cache, slots, out, z, batch);
    return false;
}
} // namespace fastllm
#endif
