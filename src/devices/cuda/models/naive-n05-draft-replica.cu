#include "naive-n05-draft-fused.h"
#include "devices/cuda/low_latency/execution.h"
#include "naive-n05-draft-attention.cuh"
#include <algorithm>
#include <stdexcept>

using namespace fastllm_draft_fused;
namespace {
void CheckReplica(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

template<class Op, auto Kernel>
int ReplicaBlocks(int device) {
    static thread_local int lastDevice = -1, blocks = 0;
    if (device == lastDevice) return blocks;
    cudaDeviceProp props{};
    CheckReplica(cudaGetDeviceProperties(&props, device));
    int active = 0;
    const int arch = props.major * 10 + props.minor;
    bool supported = arch == 103 || arch == 120;
#ifdef FASTLLM_DRAFT_HAS_SM90
    supported |= arch == 90;
#endif
    if (supported && props.cooperativeLaunch) {
        auto status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, Kernel, Op::Threads, 0);
        if (fastllm::cuda::low_latency::BackendUnavailable(status)) {
            const auto pending = cudaGetLastError();
            if (pending != status) CheckReplica(pending);
            active = 0;
        } else {
            CheckReplica(status);
        }
    }
    lastDevice = device;
    blocks = std::min(Op::Width / 8, active * props.multiProcessorCount);
    return blocks;
}

template<class Op, auto Kernel>
void RunReplica(int blocks, typename Op::Args &args, void *stream) {
    void *params[] = {&args};
    CheckReplica(cudaLaunchCooperativeKernel(reinterpret_cast<const void *>(Kernel), blocks, Op::Threads,
        params, 0, stream ? static_cast<cudaStream_t>(stream) : cudaStreamPerThread));
}
} // namespace

extern "C" int FastllmDraftLocalAttentionPlan(int device, int rows, int heads, int kvHeads, int dim,
        int window, int *partialFloats) {
    using Op = ReplicaAttention;
    if (rows < 1 || rows > 8 || heads != Op::Heads || kvHeads != Op::KVHeads || dim != Op::Dim ||
        window < rows || window > 1024) return 0;
    *partialFloats = ((window - 1 + rows + Op::Chunk - 1) / Op::Chunk) * heads * rows * Op::Stride;
    return ReplicaBlocks<Op, FusedDraftAttention<Op>>(device);
}

extern "C" void FastllmDraftLocalAttentionRun(int blocks, const FastllmDraftLocalAttention &x, void *stream) {
    using BF = ReplicaAttention::BF;
    ReplicaAttention::Args a{static_cast<const BF *>(x.input), static_cast<const BF *>(x.qkvWeight),
        static_cast<const BF *>(x.oWeight), x.qNorm, x.kNorm, x.postNorm, x.positions, x.live, x.binding,
        static_cast<BF *>(x.raw), static_cast<BF *>(x.attention), static_cast<BF *>(x.residual),
        static_cast<BF *>(x.output), x.partials, x.normScratch, x.rows, x.window, x.eps, x.theta};
    RunReplica<ReplicaAttention, FusedDraftAttention<ReplicaAttention>>(blocks, a, stream);
}
extern "C" int FastllmDraftFusedPairMlpBlocks(int device) {
    return ReplicaBlocks<ReplicaMlpPair, FusedDraftMlp<ReplicaMlpPair>>(device);
}
extern "C" int FastllmDraftFusedLocalMlpBlocks(int device) {
    const int blocks = ReplicaBlocks<ReplicaMlp, FusedDraftMlp<ReplicaMlp>>(device);
    return blocks >= ReplicaMlp::Rows ? blocks : 0;
}
extern "C" void FastllmDraftFusedLocalMlpRun(int blocks, int rows, const void *normalized,
        const void *gateUp, const void *down, void *middle, void *normScratch, void *residual,
        const float *gamma, float eps, void *output, void *stream) {
    ReplicaMlp::Args args{static_cast<const ReplicaMlp::BF *>(normalized), static_cast<const ReplicaMlp::BF *>(gateUp),
        static_cast<const ReplicaMlp::BF *>(down), static_cast<ReplicaMlp::BF *>(middle), static_cast<float *>(normScratch),
        static_cast<ReplicaMlp::BF *>(residual), static_cast<ReplicaMlp::BF *>(output), gamma, eps, rows};
    if (rows > ReplicaMlp::Rows) RunReplica<ReplicaMlpPair, FusedDraftMlp<ReplicaMlpPair>>(blocks, args, stream);
    else RunReplica<ReplicaMlp, FusedDraftMlp<ReplicaMlp>>(blocks, args, stream);
}
