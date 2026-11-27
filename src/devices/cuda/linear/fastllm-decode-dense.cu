#include "devices/cuda/fastllm-cuda-decode-dense.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-linear-output.h"
#include <cuda_bf16.h>
#include "devices/cuda/low_latency/dense.h"
#include "../low_latency/data_layout.h"
#include "utils.h"
#include <map>
#include <tuple>

namespace fastllm {
namespace {
__global__ void CombineGateUp(__nv_bfloat16 *gate, const __nv_bfloat16 *up, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        const float x = __bfloat162float(gate[i]);
        const float silu = __bfloat162float(__float2bfloat16_rn(x / (1.f + expf(-x))));
        gate[i] = __float2bfloat16_rn(silu * __bfloat162float(up[i]));
    }
}
} // namespace
bool FastllmCudaDecodeDense(const Data &input, const Data &gate, const Data &up,
        const Data &down, Data &intermediate, Data &output) {
    namespace ll = cuda::low_latency;
    if (gate.dims.size()!=2 || up.dims!=gate.dims ||
        down.dims!=std::vector<int>{gate.dims[1],gate.dims[0]} || input.dims.empty() ||
        input.dims.back()!=gate.dims[1] || gate.dims[1]<=0) return false;
    const uint64_t logicalRows=input.Count(0)/gate.dims[1];
    if (logicalRows<1 || logicalRows>128 || (logicalRows>8 && FastllmCudaRuntimeArch()!=103)) return false;
    // Wide projection temporarily stores Up in the output. Future registered
    // shapes with an expanded intermediate must use their regular workspace.
    if (logicalRows > 16 && gate.dims[0] > gate.dims[1]) return false;
    const int rows=static_cast<int>(logicalRows);
    int device=-1;
    AssertInFastLLM(cudaGetDevice(&device)==cudaSuccess,"Cannot query Dense device.");
    for (const Data *p : {&input,&gate,&up,&down})
        if (p->dataType!=BFLOAT16 || !ll::DenseData(*p,device)) return false;
    if (ll::DataOverlap(intermediate,output)) return false;
    for (const Data *p : {&intermediate,&output}) {
        if (p->isFake || p->cudaDataBorrowed || p->multiDeviceData ||
            (p->cudaData && p->dataDeviceIds!=std::vector<int>{device})) return false;
        for (const Data *in : {&input,&gate,&up,&down}) if (ll::DataOverlap(*p,*in)) return false;
    }
    using Key=std::tuple<int,int,int,int>;
    static thread_local std::map<Key,ll::DensePlan> plans;
    const Key key{device,rows,gate.dims[1],gate.dims[0]};
    auto found=plans.find(key);
    if (found==plans.end()) {
        if (FastllmCudaGraphIsCapturingFast()) return false;
        ll::DensePlan plan;
        auto error=ll::PrepareDense({rows,gate.dims[1],gate.dims[0]},plan,cudaStreamPerThread);
        if (error==cudaErrorStreamCaptureUnsupported) return false; // Transient: never cache.
        if (ll::BackendUnavailable(error)) {
            const auto pending=cudaGetLastError();
            AssertInFastLLM(pending==cudaSuccess || ll::BackendUnavailable(pending),"Pending CUDA failure during Dense admission.");
            plans.emplace(key,plan); // Cache only stable capability misses.
            return false;
        }
        AssertInFastLLM(error==cudaSuccess,"Dense preparation failed.");
        found=plans.emplace(key,plan).first;
    }
    if (!found->second.Ready()) return false;
    for (Data *p : {&intermediate,&output}) {
        auto dims=input.dims;
        dims.back()=p==&intermediate?gate.dims[0]:gate.dims[1];
        p->dataType=BFLOAT16;p->UpdateUnitSize();p->Resize(dims);
        p->ToDevice(input.dataDevice,input.dataDeviceIds,false);p->Allocate(false);
    }
    if (rows > 16) {
        // All admitted shapes have intermediate <= hidden. The output allocation
        // can hold Up until SwiGLU has consumed it, before Down overwrites it.
        const int hidden = gate.dims[1], width = gate.dims[0];
        AssertInFastLLM(FastllmCudaBf16HeuristicGemm(input.cudaData, gate.cudaData, intermediate.cudaData, rows, hidden, width) &&
                        FastllmCudaBf16HeuristicGemm(input.cudaData, up.cudaData, output.cudaData, rows, hidden, width),
                        "Dense gate/up GEMM failed.");
        CombineGateUp<<<(rows * width + 255) / 256, 256, 0, cudaStreamPerThread>>>(
            static_cast<__nv_bfloat16 *>(intermediate.cudaData), static_cast<const __nv_bfloat16 *>(output.cudaData), rows * width);
        AssertInFastLLM(cudaGetLastError() == cudaSuccess &&
                        FastllmCudaBf16HeuristicGemm(intermediate.cudaData, down.cudaData, output.cudaData, rows, width, hidden),
                        "Dense activation or down GEMM failed.");
        return true;
    }
    const ll::DenseBindings b{{input.cudaData,input.GetBytes()},{gate.cudaData,gate.GetBytes()},
        {up.cudaData,up.GetBytes()},{down.cudaData,down.GetBytes()},
        {intermediate.cudaData,intermediate.GetBytes()},{output.cudaData,output.GetBytes()}};
    AssertInFastLLM(ll::RunDense(found->second,b,cudaStreamPerThread)==cudaSuccess,"Dense launch failed.");
    return true;
}
}
