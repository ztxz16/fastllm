// The cluster schedule and shared-memory fence follow Basalt v0.5.6 (MIT,
// Copyright (c) 2026 Jesús Gulfo; see third_party/basalt/LICENSE).
// https://github.com/jesdga95/basalt/blob/v0.5.6/src/models/inkling2/kernels/attn.cu
// The paged scan preserves FastLLM's FlashInfer MMA, softmax partition order
// and compensation for FP16 rounding of queries and probabilities.
enum class WindowDecodeKernel { Paged, Split, Cluster };
using WindowInput = RelativeInputs<__nv_bfloat16>;
constexpr int windowWarps = 4, windowClusterSize = 8;
struct WindowSoftmax {
    static constexpr bool use_softmax = true, use_fp32_softmax_sum = true, use_pv_residual = true;
    static constexpr float sm_scale_log2 = RelativeAttention::sm_scale_log2;
};
using WindowMath = flashinfer::KernelTraits<flashinfer::MaskMode::kCausal, 16, 1, 2, 16, 8, 1, 1,
    flashinfer::PosEncodingMode::kNone, half, half, float, float, int, WindowSoftmax, false>;

struct WindowTile {
    alignas(16) half query[3][256];
    float relative[3][16], bias[windowWarps][3][32];
    alignas(16) uint4 kv[windowWarps][32 * 16];
};

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(CUDA_NO_TENSOR_CORE)
__device__ __forceinline__ void prepareWindowHead(WindowTile &tile, const WindowInput &in,
                                                  const DecodeVisit &visit, int kvHead, bool append) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int task = warp; task < (append ? 5 : 3); task += windowWarps) {
        const bool query = task < 3, value = task == 4;
        const int head = query ? kvHead * 3 + task : kvHead;
        const auto src = in.sources.items[query ? 0 : (value ? 6 : 4)];
        float x[4], weights[4], scaled[4];
        const float clip = value ? 0.f : in.qClip;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = head * 128 + lane + j * 32;
            weights[j] = query ? in.qs[c] : 1.f;
            x[j] = query ? src.read(0, c) : updateConvChannel<false, __nv_bfloat16, __nv_bfloat16>(
                src.data, &visit, value, value ? in.vw : in.kw, nullptr, in.kvHeads * 128, in.taps, in.kvHeads, c, 0, src.stride);
            scaled[j] = clip > 0.f ? x[j] * weights[j] : x[j];
        }
        const float scale = prefillScale(scaled, clip, 128, in.eps, false);
        if (query) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float q = x[j] * scale * weights[j];
                const half hi(q);
                tile.query[task][lane + 32*j] = hi;
                tile.query[task][128 + lane + 32*j] = half(q - float(hi));
            }
            const float r = lane < 16 ? in.sources.items[1].read(0, head * 16 + lane) : 0.f;
            const float w = lane < 16 ? in.rs[head * 16 + lane] : 0.f;
            const float components[4] = {in.rClip > 0.f ? r * w : r, 0.f, 0.f, 0.f};
            const float norm = prefillScale(components, in.rClip, 16, in.eps, false);
            if (lane < 16) tile.relative[task][lane] = r * norm * w;
        } else {
            const auto &pages = visit.pages;
            half *dst = static_cast<half *>(value ? pages.value : pages.key) +
                pages.offset(visit.past, value, in.kvHeads * 128) + kvHead * 128 + lane;
#pragma unroll
            for (int j = 0; j < 4; ++j) dst[32*j] = half(x[j] * scale);
        }
    }
}

template<class Emit>
__device__ __forceinline__ void scanWindowTile(WindowTile &tile, const WindowInput &in, const DecodeVisit &visit,
                                               int kvHead, int begin, int end, int firstKey, Emit emit) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int row = lane / 4, pair = lane % 4, head = min(row, 2), stride = in.kvHeads * 128;
    const auto &pages = visit.pages;
    const auto *keys = static_cast<const half *>(pages.key), *values = static_cast<const half *>(pages.value);
    float maximum[1][2], denom[1][2], output[1][8][8];
    WindowSoftmax softmax{};
    flashinfer::init_states<WindowMath>(softmax, output, maximum, denom);
    flashinfer::smem_t<WindowMath::SWIZZLE_MODE_KV> sm(tile.kv[warp]);

    for (int base = begin; base < end; base += 32) {
        float scores[1][2][8]{};
        const half *v[4];
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int token = max(firstKey, min(base + row + 8*t, end - 1));
            const half *k = keys + pages.offset(token, false, stride) + kvHead * 128;
            v[t] = values + pages.offset(token, true, stride) + kvHead * 128;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = j * 4 + pair;
                sm.load_128b_async(sm.template get_permuted_offset<16>(row + 8*t, col), k + 8*col);
                asm volatile("prefetch.global.L2 [%0];" :: "l"(v[t] + 8*col));
            }
        }
        flashinfer::cp_async::commit_group();
        const int token = base + lane, distance = visit.past - token;
        float dots[3]{};
        if (token >= firstKey && token < end && distance < in.extent) {
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float coeff = in.table[j * in.extent + distance];
#pragma unroll
                for (int h = 0; h < 3; ++h) dots[h] += tile.relative[h][j] * coeff;
            }
        }
#pragma unroll
        for (int h = 0; h < 3; ++h) tile.bias[warp][h][lane] = dots[h];
        flashinfer::cp_async::wait_group<0>();
        __syncwarp();
        // Both query halves multiply the same 128 K channels.
#pragma unroll
        for (int d = 0; d < 16; ++d) {
            const unsigned *q = reinterpret_cast<const unsigned *>(&tile.query[head][d*16 + pair*2]);
            unsigned a[4] = {q[0], 0, q[4], 0};
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                unsigned b[4];
                sm.ldmatrix_m8n8x4(sm.template get_permuted_offset<16>(
                    16*t + 8*(lane/16) + lane%8, 2*(d%8) + (lane%16)/8), b);
                flashinfer::mma::mma_sync_m16n16k16_row_col_f16f16f32<half>(scores[0][t], a, b);
            }
        }
        __syncwarp();
        // Time-share K's tile with V while the warp normalizes its scores.
#pragma unroll
        for (int t = 0; t < 4; ++t) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = j * 4 + pair;
                sm.load_128b_async(sm.template get_permuted_offset<16>(row + 8*t, col), v[t] + 8*col);
            }
        }
        flashinfer::cp_async::commit_group();
#pragma unroll
        for (int t = 0; t < 2; ++t) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = t*16 + (j >= 4 ? 8 : 0) + pair*2 + j%2;
                float &s = scores[0][t][j];
                s = base + col >= firstKey && base + col < end
                    ? s * (1.f / sqrtf(128.f)) + tile.bias[warp][head][col] : -flashinfer::math::inf;
            }
        }
        const float oldMaximum = maximum[0][0];
        float unusedOutput[1][8][8]{}, unusedDenom[1][2]{};
        flashinfer::update_mdo_states<WindowMath>(softmax, scores, unusedOutput, maximum, unusedDenom);
        // Only the first MMA query row group contains live heads. Round its
        // rescale before either 16-key PV result is added, as in the paged scan.
        const float rescale = flashinfer::math::ptx_exp2((oldMaximum - maximum[0][0]) * WindowSoftmax::sm_scale_log2);
        denom[0][0] = __fmul_rn(denom[0][0], rescale);
#pragma unroll
        for (int d = 0; d < 8; ++d) {
#pragma unroll
            for (int j = 0; j < 8; ++j) output[0][d][j] = __fmul_rn(output[0][d][j], rescale);
        }
        flashinfer::cp_async::wait_group<0>();
        __syncwarp();
        unsigned offset = sm.template get_permuted_offset<16>(lane % 16, lane / 16);
        flashinfer::compute_sfm_v<WindowMath>(&sm, &offset, nullptr, lane, scores, output, denom, 0);
        __syncwarp();
    }
    flashinfer::finalize_m<WindowMath>(softmax, maximum);
    emit(row, pair, flashinfer::math::ptx_log2(denom[0][0]) + maximum[0][0],
         output[0], denom[0][0]);
}
#endif

struct WindowExchange {
    alignas(16) float values[windowClusterSize * windowWarps][3][16];
    float lse[windowClusterSize * windowWarps][3];
};
struct NoWindowExchange {};

template<bool Cluster>
__global__ void fusedWindowDecode(VerifyPrefillParams p, WindowInput in, __nv_bfloat16 *out) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(CUDA_NO_TENSOR_CORE)
    __shared__ WindowTile tile;
    __shared__ std::conditional_t<Cluster, WindowExchange, NoWindowExchange> exchange;
    const int warp = threadIdx.x / 32;
    [[maybe_unused]] const int lane = threadIdx.x % 32;
    const int kv = blockIdx.x, rank = Cluster ? blockIdx.z : blockIdx.y;
    const DecodeVisit &visit = *p.visits;
    const VerifyPartition part(visit, 0, 1, p.maxSplits, p.window_left, 1);
    const int active = (visit.past - part.origin) / part.chunk + 1;
    const int firstKey = max(0, visit.past - p.window_left);
    const int firstSplit = (firstKey - part.origin) / part.chunk;
    const int split = Cluster ? warp * windowClusterSize + rank : rank * windowWarps + warp;
    if constexpr (!Cluster) {
        if (rank * windowWarps >= active) return;
    }
#if CUDART_VERSION >= 12000 && __CUDA_ARCH__ >= 900
    auto cluster = cooperative_groups::this_cluster();
    if constexpr (Cluster) {
#if __CUDA_ARCH__ == 1200 && CUDART_VERSION >= 12080
        asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
#else
        cluster.sync();
#endif
    }
#endif
    const int owner = Cluster ? (active - 1) % windowClusterSize : (active - 1) / windowWarps;
    prepareWindowHead(tile, in, visit, kv, rank == owner);
#if CUDART_VERSION >= 12080 && __CUDA_ARCH__ == 1200
    if constexpr (Cluster) asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
#endif
#if CUDART_VERSION >= 12080 && __CUDA_ARCH__ >= 900
    // Let the output projection fetch weights during the window scan.
    if constexpr (Cluster) asm volatile("griddepcontrol.launch_dependents;");
#endif
    __syncthreads(); // Only the owner of the newest split reads the appended token.
    if (split >= firstSplit && split < active) {
        const int begin = part.origin + split * part.chunk;
        scanWindowTile(tile, in, visit, kv, begin, min(visit.past + 1, begin + part.chunk), firstKey,
            [&](int row, int pair, float lse, const float (&values)[8][8], float denom) {
#if CUDART_VERSION >= 12000 && __CUDA_ARCH__ >= 900
                if constexpr (Cluster) {
#pragma unroll
                    for (int h = 0; h < 3; ++h) {
                        const float z = __shfl_sync(0xffffffffu, lse, 4*h);
                        if (lane < windowClusterSize) *cluster.map_shared_rank(&exchange.lse[split][h], lane) = z;
                    }
                }
#endif
                if (row >= 3) return;
                const int head = kv * 3 + row;
                const size_t slot = size_t(split) * in.heads + head;
                if constexpr (!Cluster) {
                    if (pair == 0) p.lse[slot] = lse;
                }
#pragma unroll
                for (int d = 0; d < 8; ++d) {
                    auto store = [&](float *dst) {
                        *reinterpret_cast<float2 *>(dst) = make_float2(values[d][0] / denom, values[d][1] / denom);
                        *reinterpret_cast<float2 *>(dst + 8) = make_float2(values[d][4] / denom, values[d][5] / denom);
                    };
                    if constexpr (!Cluster) store(p.o + slot * 128 + d*16 + 2*pair);
#if CUDART_VERSION >= 12000 && __CUDA_ARCH__ >= 900
                    else store(cluster.map_shared_rank(&exchange.values[split][row][2*pair], d));
#endif
                }
            });
    }
#if CUDART_VERSION >= 12000 && __CUDA_ARCH__ >= 900
    if constexpr (Cluster) {
        // Only distributed shared-memory stores cross CTAs. KV and convolution writes
        // are local to the newest split and become visible downstream at kernel completion.
#if __CUDA_ARCH__ == 1200 && CUDART_VERSION >= 12080
        asm volatile("fence.release.sync_restrict::shared::cta.cluster; "
                     "barrier.cluster.arrive.relaxed.aligned; barrier.cluster.wait.aligned; "
                     "fence.acquire.sync_restrict::shared::cluster.cluster;" ::: "memory");
#else
        cluster.sync();
#endif
        if (warp >= 3) return;
        const int head = 3 * kv + warp, channel = head * 128 + rank * 16 + lane % 16;
        const float z = lane >= firstSplit && lane < active ? exchange.lse[lane][warp] : -INFINITY;
        const float sink = in.sources.items[3].read(0, head) * WindowSoftmax::sm_scale_log2;
        float maximum = fmaxf(sink, z);
#pragma unroll
        for (int step = 16; step; step /= 2) maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, step));
        const float weight = exp2f(z - maximum);
        float denom = exp2f(sink - maximum), value = 0.f;
        for (int i = firstSplit; i < active; ++i) {
            const float w = __shfl_sync(0xffffffffu, weight, i);
            denom += w;
            value += w * exchange.values[i][warp][lane % 16];
        }
        if (lane < 16) out[channel] = __nv_bfloat16(value / denom / (1.f + expf(-in.sources.items[2].read(0, channel))));
    }
#endif
#endif
}

#if CUDART_VERSION >= 12000
inline cudaLaunchConfig_t windowClusterConfig(int kvHeads, cudaLaunchAttribute &attribute) {
    attribute.id = cudaLaunchAttributeClusterDimension;
    attribute.val.clusterDim = {1, 1, windowClusterSize};
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(kvHeads, 1, windowClusterSize);
    cfg.blockDim = windowWarps * 32;
    cfg.stream = cudaStreamPerThread;
    cfg.attrs = &attribute;
    cfg.numAttrs = 1;
    return cfg;
}
#endif

WindowDecodeKernel configureWindowDecode(int gpu, int kvHeads, int splits) {
#ifndef CUDA_NO_TENSOR_CORE
    cudaFuncAttributes attr{};
    if (cudaFuncGetAttributes(&attr, fusedWindowDecode<false>) != cudaSuccess) {
        cudaGetLastError();
        return WindowDecodeKernel::Paged;
    }
    if (attr.binaryVersion < 80 || attr.ptxVersion < 80) return WindowDecodeKernel::Paged;
#if CUDART_VERSION >= 12000
    if (attr.binaryVersion >= 90 && attr.ptxVersion >= 90 && splits <= windowWarps * windowClusterSize) {
        int supported = 0;
        if (cudaDeviceGetAttribute(&supported, cudaDevAttrClusterLaunch, gpu) != cudaSuccess) {
            cudaGetLastError();
            return WindowDecodeKernel::Split;
        }
        if (supported) {
            cudaLaunchAttribute attribute{};
            auto cfg = windowClusterConfig(kvHeads, attribute);
            int resident = 0;
            if (cudaOccupancyMaxActiveClusters(&resident, fusedWindowDecode<true>, &cfg) == cudaSuccess) {
                if (resident > 0) return WindowDecodeKernel::Cluster;
            } else cudaGetLastError();
        }
    }
#endif
    return WindowDecodeKernel::Split;
#else
    return WindowDecodeKernel::Paged;
#endif
}

void launchFusedWindow(WindowDecodeKernel kernel, const DecodeVisit *visit, const WindowInput &in,
                       int window, int splits, const Matrix &scratch, __nv_bfloat16 *out) {
    VerifyPrefillParams p{};
    p.visits = visit;
    p.num_qo_heads = in.heads;
    p.chain = 1;
    p.minTilesPerSplit = 1;
    p.maxSplits = splits;
    p.window_left = window - 1;
#if CUDART_VERSION >= 12000
    if (kernel == WindowDecodeKernel::Cluster) {
        cudaLaunchAttribute attribute{};
        auto cfg = windowClusterConfig(in.kvHeads, attribute);
        check(cudaLaunchKernelEx(&cfg, fusedWindowDecode<true>, p, in, out));
        return;
    }
#endif
    p.o = scratch.p;
    p.lse = p.o + size_t(splits) * in.heads * 128;
    fusedWindowDecode<false><<<dim3(in.kvHeads, (splits + windowWarps - 1) / windowWarps), windowWarps * 32,
                               0, cudaStreamPerThread>>>(p, in, out);
    const auto sink = in.sources.items[3], gate = in.sources.items[2];
    auto merge = splits <= 32 ? mergeVerifyAttention<__nv_bfloat16, false, true> : mergeVerifyAttention<__nv_bfloat16, false>;
    merge<<<dim3(1, in.heads), 128, 0, cudaStreamPerThread>>>(p, sink.data, gate.data, out, sink.stride, gate.stride);
}
