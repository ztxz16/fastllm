// The six-row verifier retains the scalar kernel's 1024 logical BF16 pairs
// and its three K stripes. Cooperative vector loads feed those same lanes.
struct alignas(8) VerifyPairs { Pair data[2]; };

struct VerifyDraftFeature {
    Pair *packed;
    const int *globalRows;
    int slot, features;
};

template <bool Experts, bool Publish = false>
__device__ __forceinline__ void VerifyPostBody(const Pair *attention, const Pair *residual,
        const float *post, const float *comb, const float *pre, const float *weight, float eps,
        Pair *hidden, Pair *normalized, ExpertBoundary ep, FastllmDsv41InputPacket packet, VerifyDraftFeature feature = {}) {
#if __CUDA_ARCH__ == 1030
    using Vec = VerifyPairs;
    constexpr int V = 2, Blocks = 8;
    constexpr int width = 2560, logical = 1024 / Blocks, workers = logical / V;
    const int rank = blockIdx.x, row = blockIdx.y, stripe = threadIdx.x / workers;
    const int vector = rank * workers + threadIdx.x % workers + stripe * (1024 / V);
    const int lane = threadIdx.x & 31;
    const bool active = vector < width / V;
    __shared__ float terms[2][logical], sums[32];
    __shared__ unsigned long long barrier;
    auto cluster = cooperative_groups::this_cluster();
    const unsigned bar = __cvta_generic_to_shared(&barrier);
    if (!threadIdx.x) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar) : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 128;" :: "r"(bar) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    // Cluster storage is private. All reads of upstream tensors follow this wait.
    if constexpr (!Experts) cudaGridDependencySynchronize();
    float2 x[V]{}, folded[V]{};
    float4 normWeights;
    float squares[V]{};
    if (active) {
        normWeights = reinterpret_cast<const float4 *>(weight)[vector];
        if constexpr (Experts) {
            const unsigned order = ep.ids[row];
            // Independent slot loads can overlap before the ordered FP32 sum.
            Vec contributions[6]{};
#pragma unroll
            for (int k = 0; k < 6; ++k) {
                const unsigned entry = order >> (5 * k);
                if (entry & 8)
                    contributions[k] = reinterpret_cast<const Vec *>(ep.slots)[
                        (((entry & 7) * (FastllmDsv41MegaMaxRows / 8) + row) * width / V) + vector];
            }
            float2 partial[V]{};
#pragma unroll
            for (int k = 0; k < 6; ++k) {
                const unsigned entry = order >> (5 * k);
                const Vec val = contributions[k];
#pragma unroll
                for (int i = 0; i < V; ++i) {
                    partial[i] = __fadd2_rn(partial[i], __bfloat1622float2(val.data[i]));
                    if (entry & 16) {
                        x[i] = __fadd2_rn(x[i], partial[i]);
                        partial[i] = {};
                    }
                }
            }
            auto shared = reinterpret_cast<const Vec *>(ep.slots)[
                ((6 * (FastllmDsv41MegaMaxRows / 8) + row) * width / V) + vector];
#pragma unroll
            for (int i = 0; i < V; ++i) {
                auto v = __fadd2_rn(x[i], __bfloat1622float2(shared.data[i]));
                x[i] = __bfloat1622float2(__floats2bfloat162_rn(v.x, v.y));
            }
        } else {
            auto v = reinterpret_cast<const Vec *>(attention)[row * width / V + vector];
#pragma unroll
            for (int i = 0; i < V; ++i) x[i] = __bfloat1622float2(v.data[i]);
        }
        float2 mean[V]{};
        Vec residuals[4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            residuals[i] = reinterpret_cast<const Vec *>(residual)[(row * 4 + i) * width / V + vector];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            Vec updated;
#pragma unroll
            for (int i = 0; i < V; ++i) {
                float2 mix{};
#pragma unroll
                for (int k = 0; k < 4; ++k)
                    mix = MixMultiplyAdd(comb[row * 16 + 4 * k + j], __bfloat1622float2(residuals[k].data[i]), mix);
                mix = MixMultiplyAdd(post[row * 4 + j], x[i], mix);
                updated.data[i] = __floats2bfloat162_rn(mix.x, mix.y);
                const float2 rounded = __bfloat1622float2(updated.data[i]);
                folded[i] = MixMultiplyAdd(pre[row * 4 + j], rounded, folded[i]);
                if constexpr (Publish)
                    mean[i] = __fadd2_rn(mean[i], __fmul2_rn(make_float2(.25f, .25f), rounded));
            }
            reinterpret_cast<Vec *>(hidden)[(row * 4 + j) * width / V + vector] = updated;
        }
#pragma unroll
        for (int i = 0; i < V; ++i) {
            if constexpr (Publish) {
                const size_t dest = size_t(feature.globalRows[row]) * feature.features + feature.slot;
                feature.packed[dest * width + vector * V + i] = __floats2bfloat162_rn(mean[i].x, mean[i].y);
            }
            auto v = __bfloat1622float2(__floats2bfloat162_rn(folded[i].x, folded[i].y));
            folded[i] = v;
            squares[i] = v.x * v.x + v.y * v.y;
        }
    }
    if (stripe) {
#pragma unroll
        for (int i = 0; i < V; ++i)
            terms[stripe - 1][(threadIdx.x % workers) * V + i] = squares[i];
    }
    __syncthreads();
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    if (!stripe) {
#pragma unroll
        for (int i = 0; i < V; ++i) {
            int t = threadIdx.x * V + i;
            squares[i] += terms[0][t];
            if (rank < Blocks / 2) squares[i] += terms[1][t];
#pragma unroll
            for (int off = 16 / V; off; off /= 2)
                squares[i] += __shfl_down_sync(0xffffffff, squares[i], off, 32 / V);
        }
        // The remaining shuffles in the reference tree are within a vector.
        float sum = squares[0] + squares[1];
        sum = __shfl_sync(0xffffffff, sum, 0, 32 / V);
        const int target = lane % (32 / V);
        static_assert(Blocks <= 32 / V);
        if (target < Blocks) {
            const int slot = (rank * logical + threadIdx.x * V) / 32;
            const unsigned dst = __cvta_generic_to_shared(cluster.map_shared_rank(sums + slot, target));
            const unsigned done = __cvta_generic_to_shared(cluster.map_shared_rank(&barrier, target));
            asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                         :: "r"(dst), "f"(sum), "r"(done) : "memory");
        }
    }
    asm volatile("{ .reg .pred ok; VWAIT_%=: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 ok,[%0],0; @!ok bra VWAIT_%=; }"
                 :: "r"(bar) : "memory");
    float total = sums[lane];
#pragma unroll
    for (int off = 16; off; off >>= 1) total += __shfl_down_sync(0xffffffff, total, off);
    const float scale = rsqrtf(__shfl_sync(0xffffffff, total, 0) / 5120.f + eps);
    if (!active) return;
    Vec out;
    float maximum = 0.f;
#pragma unroll
    for (int i = 0; i < V; ++i) {
        const float4 w = normWeights;
        out.data[i] = __floats2bfloat162_rn(folded[i].x * scale * (i % 2 ? w.z : w.x),
                                           folded[i].y * scale * (i % 2 ? w.w : w.y));
        if constexpr (!Experts) {
            auto v = __bfloat1622float2(out.data[i]);
            maximum = fmaxf(maximum, fmaxf(fabsf(v.x), fabsf(v.y)));
        }
    }
    reinterpret_cast<Vec *>(normalized)[row * width / V + vector] = out;
    if constexpr (!Experts) {
#pragma unroll
        for (int off = 8 / V; off; off >>= 1)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, off, 16 / V));
        unsigned exponent = max((__float_as_uint(maximum) + 0x1fffffu) >> 23, 113u) - 8u;
        float inv = __uint_as_float((254u - exponent) << 23);
        uint32_t codes = 0;
#pragma unroll
        for (int i = 0; i < V; ++i) {
            auto v = __bfloat1622float2(out.data[i]);
            codes |= uint32_t(__nv_fp8x2_e4m3(make_float2(v.x * inv, v.y * inv)).__x) << (16 * i);
        }
        reinterpret_cast<uint32_t *>(packet.codes)[row * width / V + vector] = codes;
        if (!(vector % (16 / V))) {
            const int group = vector / (16 / V);
            packet.scales[row * 160 + group] = exponent;
            const int sfRow = row / 16 * 128 + row % 16 * 4;
            packet.sharedScales[(group / 4 * packet.sharedScaleRows + sfRow) * 4 + group % 4] = exponent;
        }
    }
#endif
}

__global__ __launch_bounds__(192) void VerifyPostPacket(const Pair *attention, const Pair *residual,
        const float *post, const float *comb, const float *pre, const float *weight, float eps,
        Pair *hidden, Pair *normalized, FastllmDsv41InputPacket packet) {
    VerifyPostBody<false>(attention, residual, post, comb, pre, weight, eps, hidden, normalized, {}, packet);
}

__global__ __launch_bounds__(192) void VerifyExpertNorm(const Pair *residual, const float *post,
        const float *comb, const float *pre, const float *weight, float eps, Pair *hidden,
        Pair *normalized, ExpertBoundary boundary) {
    VerifyPostBody<true>(nullptr, residual, post, comb, pre, weight, eps, hidden, normalized, boundary, {});
}

__global__ __launch_bounds__(192) void VerifyExpertFeatureNorm(const Pair *residual, const float *post,
        const float *comb, const float *pre, const float *weight, float eps, Pair *hidden,
        Pair *normalized, ExpertBoundary boundary, VerifyDraftFeature feature) {
    VerifyPostBody<true, true>(nullptr, residual, post, comb, pre, weight, eps, hidden, normalized, boundary, {}, feature);
}

inline bool VerifyPostCluster(const void *kernel) {
    cudaFuncAttributes attrs{};
    if (cudaFuncGetAttributes(&attrs, kernel) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    if (attrs.ptxVersion != 103) return false;
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(8, 6);
    cfg.blockDim = 192;
    int size = 0;
    if (cudaOccupancyMaxPotentialClusterSize(&size, kernel, &cfg) == cudaSuccess) return size >= 8;
    cudaGetLastError();
    return false;
}

inline bool VerifyPostAligned(std::initializer_list<const void *> pointers) {
    for (auto ptr : pointers)
        if (!ptr || (reinterpret_cast<uintptr_t>(ptr) & 15)) return false;
    return true;
}
