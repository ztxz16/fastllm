// SM103 owner-local FFN input and delayed mHC coefficients. Four CTAs exchange
// the original 32 warp sums through distributed shared memory.
namespace fastllm_v41_owner_mix {
using Bf = __nv_bfloat16;
using Pair = __nv_bfloat162;

struct ExpertBoundary {
    const Pair *slots;
    const int *ids;
    Pair *rounded;
    uint8_t *codes, *scales;
    int pitch;
};

// Keep both scalar rounding points in the paired FP32 instructions.
__device__ __forceinline__ float2 MixMultiplyAdd(float c, float2 x, float2 acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1030
    return __fadd2_rn(acc, __fmul2_rn(make_float2(c, c), x));
#else
    return make_float2(__fadd_rn(acc.x, __fmul_rn(c, x.x)), __fadd_rn(acc.y, __fmul_rn(c, x.y)));
#endif
}

template <bool PackedOrder>
__device__ __forceinline__ float2 ReadExpertSum(const ExpertBoundary &ep, int row, int column, int route) {
    constexpr int maxRows = FastllmDsv41MegaMaxRows / 8;
    float2 values[6], total{}, partial{};
    int order[6];
#pragma unroll
    for (int k = 0; k < 6; ++k) {
        order[k] = PackedOrder ? ((static_cast<unsigned>(route) >> (5 * k)) & 31) : __shfl_sync(0xffffffff, route, k, 8);
        const int slot = order[k] & 7;
        const auto value = (order[k] & 8) ? ep.slots[(slot * maxRows + row) * 2560 + column] :
            __floats2bfloat162_rn(0.f, 0.f);
        values[k] = __bfloat1622float2(value);
    }
#pragma unroll
    for (int k = 0; k < 6; ++k) {
        partial.x += values[k].x; partial.y += values[k].y;
        if (order[k] & 16) {
            total.x += partial.x; total.y += partial.y; partial = {};
        }
    }
    const auto shared = __bfloat1622float2(ep.slots[(6 * maxRows + row) * 2560 + column]);
    return __bfloat1622float2(__floats2bfloat162_rn(total.x + shared.x, total.y + shared.y));
}

template <bool PushSums, bool Experts = false, bool Pack = false, bool Quantize = false, bool WriteNorm = true, int Blocks = 4, bool MoeInput = false, bool PackedOrder = false>
__device__ __forceinline__ void PostNormBody(const Pair *attention, const Pair *residual,
        const float *post, const float *comb, const float *pre, const float *weight, float eps,
        Pair *hidden, Pair *normalized, ExpertBoundary boundary = {}, FastllmDsv41InputPacket packet = {}) {
#if __CUDA_ARCH__ >= 900
    constexpr int width = 2560;
    auto cluster = cooperative_groups::this_cluster();
    const int row = blockIdx.y, rank = cluster.block_rank(), lane = threadIdx.x & 31;
    static_assert(Blocks == 4 || (Blocks == 8 && PushSums));
    const int tid = rank * (1024 / Blocks) + threadIdx.x;
    __shared__ float warpSums[PushSums ? 32 : 8];
    __shared__ unsigned long long barrier;
    const auto barAddress = static_cast<unsigned>(__cvta_generic_to_shared(&barrier));
    if constexpr (PushSums) {
        if (threadIdx.x == 0) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(barAddress) : "memory");
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 128;" :: "r"(barAddress) : "memory");
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
        asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    }
    post += row * 4; comb += row * 16; pre += row * 4;
    residual += size_t(row) * 4 * width; hidden += size_t(row) * 4 * width;
    const int route = Experts ? (PackedOrder ? boundary.ids[row] : FastllmV41RouteOrder(boundary.ids + row * 6)) : 0;
    float2 normWeight[3];
    if constexpr (MoeInput || (PackedOrder && Experts)) {
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            const int d = tid + k * 1024;
            if (d < width) {
                if constexpr (MoeInput)
                    asm volatile("ld.global.v2.f32 {%0,%1}, [%2];" : "=f"(normWeight[k].x), "=f"(normWeight[k].y) : "l"(weight + 2 * d));
                else {
                    asm volatile("ld.global.f32 %0, [%1];" : "=f"(normWeight[k].x) : "l"(weight + 2 * d));
                    asm volatile("ld.global.f32 %0, [%1];" : "=f"(normWeight[k].y) : "l"(weight + 2 * d + 1));
                }
            }
        }
    }
    float2 saved[3];
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int d = tid + k * 1024;
        if (d >= width) continue;
        const float2 x = Experts ? ReadExpertSum<PackedOrder>(boundary, row, d, route) : __bfloat1622float2(attention[row * width + d]);
        float2 r[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) r[i] = __bfloat1622float2(residual[i * width + d]);
        float2 v = {0.f, 0.f};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 mixed{};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float c = comb[i * 4 + j];
                mixed = MixMultiplyAdd(c, r[i], mixed);
            }
            mixed = MixMultiplyAdd(post[j], x, mixed);
            const Pair h = __floats2bfloat162_rn(mixed.x, mixed.y);
            hidden[j * width + d] = h;
            const auto rounded = __bfloat1622float2(h);
            v = MixMultiplyAdd(pre[j], rounded, v);
        }
        v = __bfloat1622float2(__floats2bfloat162_rn(v.x, v.y));
        saved[k] = v;
        sum += v.x * v.x + v.y * v.y;
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);
    float total;
    if constexpr (PushSums) {
        // A warp publishes its original FP32 partial to the four CTA-local
        // tables. Each barrier tracks 32 words, one for each logical warp.
        sum = __shfl_sync(0xffffffff, sum, 0);
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
        if (lane < Blocks) {
            const int slot = tid >> 5;
            const auto dest = static_cast<unsigned>(__cvta_generic_to_shared(cluster.map_shared_rank(warpSums + slot, lane)));
            const auto done = static_cast<unsigned>(__cvta_generic_to_shared(cluster.map_shared_rank(&barrier, lane)));
            asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                         :: "r"(dest), "f"(sum), "r"(done) : "memory");
        }
        unsigned ready;
        do {
            asm volatile("{ .reg .pred ok; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ok, [%1], 0;"
                         " selp.u32 %0, 1, 0, ok; }" : "=r"(ready) : "r"(barAddress) : "memory");
        } while (!ready);
        total = warpSums[lane];
    } else {
        if (!lane) warpSums[threadIdx.x >> 5] = sum;
        cluster.sync();
        total = *cluster.map_shared_rank(warpSums + (lane & 7), lane >> 3);
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) total += __shfl_down_sync(0xffffffff, total, offset);
    const float scale = rsqrtf(__shfl_sync(0xffffffff, total, 0) / 5120.f + eps);
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        const int d = tid + k * 1024;
        if (d < width) {
            const Pair value = __floats2bfloat162_rn(saved[k].x * scale * ((MoeInput || PackedOrder) ? normWeight[k].x : weight[2 * d]),
                                                     saved[k].y * scale * ((MoeInput || PackedOrder) ? normWeight[k].y : weight[2 * d + 1]));
            if constexpr (WriteNorm) normalized[row * width + d] = value;
            if constexpr (MoeInput) {
                const auto v = __bfloat1622float2(value);
                float maximum = fmaxf(fabsf(v.x), fabsf(v.y));
#pragma unroll
                for (int offset = 8; offset; offset >>= 1)
                    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset, 16));
                // The E4M3 exponent and minimum scale exactly match MegaMoE prepare.
                const unsigned exponent = max((__float_as_uint(maximum) + 0x1fffffu) >> 23, 113u) - 8u;
                const float inverse = __uint_as_float((254u - exponent) << 23);
                const auto fp8 = __nv_fp8x2_e4m3(make_float2(v.x * inverse, v.y * inverse));
                reinterpret_cast<uint16_t *>(packet.codes)[row * width + d] = fp8.__x;
                if ((d & 15) == 0) {
                    const int group = d / 16;
                    packet.scales[row * 160 + group] = exponent;
                    const int sfRow = row / 16 * 128 + row % 16 * 4;
                    packet.sharedScales[(group / 4 * packet.sharedScaleRows + sfRow) * 4 + group % 4] = exponent;
                }
            }
            if constexpr (Pack) V41PackNormalizedPair(value, boundary.codes, boundary.scales, boundary.pitch, row, width, d);
            if constexpr (Quantize) {
                const auto v = __bfloat1622float2(value);
                float maximum = fmaxf(fabsf(v.x), fabsf(v.y));
                for (int offset = 8; offset; offset >>= 1)
                    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset, 16));
                const auto q = V41QuantScaleDev(maximum, 1);
                boundary.rounded[row * width + d] = __floats2bfloat162_rn(
                    V41QuantValueDev(v.x, q, 1), V41QuantValueDev(v.y, q, 1));
            }
        }
    }
    if constexpr (!PushSums) cluster.sync(); // Wait for peer DSM reads before retirement.
#endif
}

template <bool PushSums, int Blocks = 4>
__global__ __launch_bounds__(256) void PostNormCluster(const Pair *attention, const Pair *residual,
        const float *post, const float *comb, const float *pre, const float *weight, float eps,
        Pair *hidden, Pair *normalized) {
    PostNormBody<PushSums, false, false, false, true, Blocks>(attention, residual, post, comb, pre, weight, eps, hidden, normalized);
}

__global__ __launch_bounds__(128) void PostNormPacket(const Pair *attention, const Pair *residual,
        const float *post, const float *comb, const float *pre, const float *weight, float eps,
        Pair *hidden, Pair *normalized, FastllmDsv41InputPacket packet) {
    PostNormBody<true, false, false, false, true, 8, true>(attention, residual, post, comb, pre,
        weight, eps, hidden, normalized, {}, packet);
}

template <bool Pack, bool Quantize, bool WriteNorm = true, int Blocks = 4, bool PackedOrder = false>
__global__ __launch_bounds__(256) void ExpertResidualNorm(
        const Pair *residual, const float *post, const float *comb, const float *pre,
        const float *weight, float eps, Pair *hidden, Pair *normalized, ExpertBoundary boundary) {
    PostNormBody<true, true, Pack, Quantize, WriteNorm, Blocks, false, PackedOrder>(nullptr, residual, post, comb, pre, weight, eps,
                                            hidden, normalized, boundary);
}

#include "deepseekv41-verify-post.cuh"

// Same square-sum tree and high/low BF16 dot addition as V41HcJoinTensorDots.
__global__ void TensorSinkhorn(const Bf *x, const float *projected, const float *scale,
        const float *base, int iterations, float mixEps, float normEps, float *pre, float *post, float *comb) {
    const int row = blockIdx.x, tid = threadIdx.x;
    x += row * 20480; projected += row * 48;
    float sum = 0.f;
    for (int i = tid; i < 20480; i += kHcThreads) {
        const float v = __bfloat162float(x[i]);
        sum += v * v;
    }
    __shared__ float partial[kHcThreads / 32], mix[25];
    sum = V41WarpSum(sum);
    if ((tid & 31) == 0) partial[tid / 32] = sum;
    if (tid < 24) mix[tid] = projected[tid] + projected[24 + tid];
    __syncthreads();
    if (tid == 0) {
        float total = 0.f;
        for (int i = 0; i < kHcThreads / 32; ++i) total += partial[i];
        mix[24] = total;
    }
    if (tid < 32) {
        __syncwarp();
        V41HcFinishRow(mix, scale, base, 20480, iterations, mixEps, normEps,
                       pre + row * 4, post + row * 4, comb + row * 16, 0);
    }
}

struct StreamPlan {
    int rows = 0, device = -1;
    bool pushSums = false;
    bool packetCluster8 = false;
    int blocks = 4;
    fastllm_v41_hc_graph::OwnerMixGemm gemm;
    cudaEvent_t ready = nullptr, done = nullptr;
    bool verifyPacket = false;
    ~StreamPlan() {
        if (done) cudaEventDestroy(done);
        if (ready) cudaEventDestroy(ready);
    }
};
} // namespace fastllm_v41_owner_mix

void *FastllmCudaV41CreateOwnerStream(int rows) {
    using namespace fastllm_v41_owner_mix;
    if (rows < 1 || (rows > 8 && rows < 16) || rows > 192 || !V41IsSm<10, 3>()) return nullptr;
    if (rows >= 16 && V41EnvOn("FASTLLM_DSV41_DISABLE_HC_TENSOR")) return nullptr;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(4, rows); cfg.blockDim = dim3(256);
    int maxCluster = 0;
    const bool push = FastllmV41HcEnabledAtRows("FASTLLM_DSV41_HC_CLUSTER_PUSH", rows);
    const bool wide = push && V41EnvOn("FASTLLM_DSV41_WIDE_CLUSTER");
    if (wide) { cfg.gridDim.x = 8; cfg.blockDim.x = 128; }
    auto kernel = wide ? PostNormCluster<true,8> : (push ? PostNormCluster<true> : PostNormCluster<false>);
    if (cudaOccupancyMaxPotentialClusterSize(&maxCluster, kernel, &cfg) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    if (maxCluster < 4) return nullptr;
    auto p = std::make_unique<StreamPlan>();
    p->rows = rows; p->pushSums = push; p->blocks = wide && maxCluster >= 8 ? 8 : 4;
    if (push && p->blocks == 4) {
        cfg.gridDim.x = 8;
        cfg.blockDim.x = 128;
        int packetClusterSize = 0;
        if (cudaOccupancyMaxPotentialClusterSize(&packetClusterSize, PostNormPacket, &cfg) == cudaSuccess)
            p->packetCluster8 = packetClusterSize >= 8;
        else
            cudaGetLastError();
    }
    p->verifyPacket = rows == 6 && p->packetCluster8 && VerifyPostCluster((const void *)VerifyPostPacket);
    if (cudaGetDevice(&p->device) != cudaSuccess || !p->gemm.Prepare(rows >= 16) ||
        cudaEventCreateWithFlags(&p->ready, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&p->done, cudaEventDisableTiming) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    return p.release();
}
void FastllmCudaV41DestroyOwnerStream(void *p) {
    delete static_cast<fastllm_v41_owner_mix::StreamPlan *>(p);
}
void *FastllmCudaV41OwnerClusterPacket(void *plan, int rows, float *scratch,
        const Data &attention, const Data &residual, const Data &oldPre, const Data &oldPost, const Data &oldComb,
        Data &fn, const Data &scale, const Data &base, const Data &weight, int iterations, float mixEps,
        float normEps, Data &hidden, Data &pre, Data &post, Data &comb, Data &normalized, FastllmDsv41InputPacket *packet) {
    using namespace fastllm_v41_owner_mix;
    if (!plan || !scratch) return nullptr;
    auto &p = *static_cast<StreamPlan *>(plan);
    if (rows != p.rows || hidden.dataDeviceIds.size() != 1 || hidden.dataDeviceIds.front() != p.device)
        return nullptr;
    // All inputs here are the BF16/FP32, width-5120, owner-local buffers of
    // the validated EP8 graph. A missing packed tensor falls back before launch.
    const auto valid = [&](const Data &v, DataType type, size_t count) {
        return v.cudaData && !v.multiDeviceData && v.dataDevice == fastllm::DataDevice::CUDA && v.dataType == type &&
            v.dataDeviceIds == hidden.dataDeviceIds && v.Count(0) == count;
    };
    const size_t n = rows;
    if (!valid(attention, DataType::BFLOAT16, n * 5120) || !valid(residual, DataType::BFLOAT16, n * 20480) ||
        !valid(hidden, DataType::BFLOAT16, n * 20480) || !valid(normalized, DataType::BFLOAT16, n * 5120) ||
        !valid(oldPre, DataType::FLOAT32, n * 4) || !valid(oldPost, DataType::FLOAT32, n * 4) || !valid(oldComb, DataType::FLOAT32, n * 16) ||
        !valid(pre, DataType::FLOAT32, n * 4) || !valid(post, DataType::FLOAT32, n * 4) || !valid(comb, DataType::FLOAT32, n * 16) ||
        !valid(fn, DataType::FLOAT32, 24 * 20480) || !valid(weight, DataType::FLOAT32, 5120) ||
        !valid(scale, DataType::FLOAT32, 3) || !valid(base, DataType::FLOAT32, 24)) return nullptr;
    const Data *packed = rows >= 16 ? FastllmCudaHcSplitWeight(fn) : nullptr;
    if (rows >= 16 && !packed) return nullptr;
    auto check = [](cudaError_t e) {
        if (e != cudaSuccess) throw std::runtime_error(std::string("owner mHC: ") + cudaGetErrorString(e));
    };
    cudaLaunchAttribute attributes[2]{};
    auto &attr = attributes[0];
    attr.id = cudaLaunchAttributeClusterDimension; attr.val.clusterDim = {static_cast<unsigned>(p.blocks), 1, 1};
    cudaLaunchConfig_t cfg{}; cfg.gridDim = dim3(p.blocks, rows); cfg.blockDim = dim3(1024 / p.blocks);
    cfg.stream = cudaStreamPerThread; cfg.attrs = &attr; cfg.numAttrs = 1;
    auto kernel = p.blocks == 8 ? PostNormCluster<true,8> : (p.pushSums ? PostNormCluster<true> : PostNormCluster<false>);
    if (packet && packet->codes && p.packetCluster8 &&
        (reinterpret_cast<uintptr_t>(weight.cudaData) & (alignof(float2) - 1)) == 0) {
        attr.val.clusterDim.x = 8;
        cfg.gridDim.x = 8;
        cfg.blockDim.x = 128;
        auto packetKernel = PostNormPacket;
        if (p.verifyPacket && VerifyPostAligned({attention.cudaData, residual.cudaData, weight.cudaData,
                                                  hidden.cudaData, normalized.cudaData, packet->codes})) {
            cfg.blockDim.x = 192;
            packetKernel = VerifyPostPacket;
#if CUDART_VERSION >= 12000
            attributes[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attributes[1].val.programmaticStreamSerializationAllowed = 1;
            cfg.numAttrs = 2;
#endif
        }
        check(cudaLaunchKernelEx(&cfg, packetKernel, (const Pair *)attention.cudaData,
            (const Pair *)residual.cudaData, (const float *)oldPost.cudaData, (const float *)oldComb.cudaData,
            (const float *)oldPre.cudaData, (const float *)weight.cudaData, normEps, (Pair *)hidden.cudaData,
            (Pair *)normalized.cudaData, *packet));
    } else {
        if (packet) *packet = {};
        check(cudaLaunchKernelEx(&cfg, kernel, (const Pair *)attention.cudaData,
            (const Pair *)residual.cudaData, (const float *)oldPost.cudaData, (const float *)oldComb.cudaData,
            (const float *)oldPre.cudaData, (const float *)weight.cudaData, normEps, (Pair *)hidden.cudaData,
            (Pair *)normalized.cudaData));
    }
    check(cudaEventRecord(p.ready, cudaStreamPerThread));
    check(cudaStreamWaitEvent(p.gemm.stream, p.ready));
    if (packed) {
        if (V41HcTensorGemm(p.gemm.handle, hidden.cudaData, packed->cudaData, rows, scratch) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("owner mHC tensor projection failed");
        TensorSinkhorn<<<rows, kHcThreads, 0, p.gemm.stream>>>((const Bf *)hidden.cudaData, scratch,
            (const float *)scale.cudaData, (const float *)base.cudaData, iterations, mixEps, normEps,
            (float *)pre.cudaData, (float *)post.cudaData, (float *)comb.cudaData);
    } else {
        cudaKernelNodeParams dots{};
        // Six rows expose all 25 dot products in a single grid. The kernel
        // preserves the per-lane FMA order and sums the eight warps in order.
        if (rows == 6) {
            V41HcDecodeDotsTiled<40><<<dim3(25, rows), kHcThreads, 0, p.gemm.stream>>>(
                (const Bf *)hidden.cudaData, (const float *)fn.cudaData, 20480, scratch);
        } else if (V41SelectStagedDots(rows, dots)) {
            void *args[] = {&hidden.cudaData, &fn.cudaData, &scratch};
            check(cudaLaunchKernel(dots.func, dots.gridDim, dots.blockDim, args, 0, p.gemm.stream));
        } else {
            V41HcWarpDots5120<<<dim3(25, rows, 8), 32, 0, p.gemm.stream>>>(
                (const Bf *)hidden.cudaData, (const float *)fn.cudaData, scratch);
        }
        auto finish = rows == 6 ? fastllm_v41_hc_graph::FinishOwnerMix<1> : fastllm_v41_hc_graph::FinishOwnerMix<8>;
        finish<<<rows, 32, 0, p.gemm.stream>>>(scratch,
            (const float *)scale.cudaData, (const float *)base.cudaData, iterations, mixEps, normEps,
            (float *)pre.cudaData, (float *)post.cudaData, (float *)comb.cudaData);
    }
    check(cudaGetLastError()); check(cudaEventRecord(p.done, p.gemm.stream));
    return p.done;
}
