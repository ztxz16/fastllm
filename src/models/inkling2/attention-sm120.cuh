// One warp reads 16 paged KV tokens. QK uses K as the 16-row MMA operand;
// transposing the V fragments in registers gives PV the same output layout.
// The first four columns hold Q or P, the last four their FP16 rounding
// residuals. Decode uses three head columns, verification packs up to four.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200 && !defined(CUDA_NO_TENSOR_CORE)
__device__ __forceinline__ unsigned packRelativePair(float a, float b, bool residual) {
    half2 hi = __floats2half2_rn(a, b);
    if (residual) {
        const float2 rounded = __half22float2(hi);
        hi = __floats2half2_rn(a - rounded.x, b - rounded.y);
    }
    return reinterpret_cast<const unsigned &>(hi);
}

// A warp reads four 64-byte stripes from each KV row.
__device__ __forceinline__ uint4 loadRelativeStripe(const half *p) {
    uint4 v;
    asm("ld.global.L2::128B.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ unsigned transposeRelativeFragment(unsigned v) {
    unsigned t;
    asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(t) : "r"(v));
    return t;
}

__device__ __forceinline__ void relativeMma(float (&out)[4], uint4 a, unsigned b0, unsigned b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(out[0]), "+f"(out[1]), "+f"(out[2]), "+f"(out[3])
        : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

__device__ __forceinline__ int relativeSplits(int length, int splits) {
    const int chunk = max(16, (length + splits - 1) / splits);
    return (length + chunk - 1) / chunk;
}

template<bool Packed>
__device__ __forceinline__ void relativeMmaTile(AttentionRow a, float *partial, int chain, int firstColumn,
                                                int tileLength, int tileActive) {
    constexpr int warps = 4, heads = Packed ? 4 : 3, width = 128;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int row = lane / 4, pair = lane % 4, kvHead = blockIdx.y;
    const int split = blockIdx.z, splits = gridDim.z, length = Packed ? tileLength : a.visit->past + 1;
    const int active = Packed ? tileActive : relativeSplits(length, splits);
    if (split >= active)
        return;
    const PageView pages = a.visit->pages;

    const int stride = a.kvHeads * width;
    const size_t pageStride = size_t(stride) * pages.pageLen;
    const int operand = firstColumn + row % 4;
    const int qr = Packed ? min(operand / 3, chain - 1) : 0;
    int lengths[2];
    float taus[2];
    const float *biasRow[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        int col = firstColumn + (pair % 2) * 2 + i;
        int queryRow = Packed ? min(col / 3, chain - 1) : 0;
        lengths[i] = Packed ? a.visit[queryRow].past + 1 : length;
        taus[i] = 1.f + a.alpha * logf(fmaxf(1.f, float(lengths[i]) / a.floor));
        biasRow[i] = a.relative + (size_t(queryRow) * a.heads + kvHead * 3 + col % 3) * a.distances;
    }
    unsigned query[4][4];
#pragma unroll
    for (int band = 0; band < 4; ++band) {
        const float *q = a.q + (size_t(qr) * a.heads + kvHead * 3 + operand % 3) * width + band * 32 + pair * 8;
#pragma unroll
        for (int i = 0; i < 4; ++i)
            query[band][i] = packRelativePair(q[2*i], q[2*i+1], row >= 4);
    }

    float acc[4][2][4]{}, maximum[2] = {-INFINITY, -INFINITY}, denom[2]{};
    // Interleave whole 16-token tiles between the active splits. On aligned
    // pages the warp resolves one physical K/V page pair for the entire tile.
    for (int base = (split + warp * active) * 16; base < length; base += warps * active * 16) {
        const half *key[2], *value[2];
#pragma unroll
        for (int halfTile = 0; halfTile < 2; ++halfTile) {
            const int token = min(base + row + 8 * halfTile, length - 1);
            const int pageToken = pages.pageLen % 16 == 0 ? base : token;
            const int page = pages.shift < 0 ? pageToken / pages.pageLen : pageToken >> pages.shift;
            const int2 ids = pages.ids[page - pages.first];
            const size_t offset = size_t(token - page * pages.pageLen) * stride + kvHead * width + pair * 8;
            key[halfTile] = static_cast<const half *>(pages.key) + ids.x * pageStride + offset;
            value[halfTile] = static_cast<const half *>(pages.value) + ids.y * pageStride + offset;
        }
        float score[4]{};
#pragma unroll
        for (int band = 0; band < 4; ++band) {
            const uint4 k0 = loadRelativeStripe(key[0] + band * 32);
            const uint4 k1 = loadRelativeStripe(key[1] + band * 32);
            relativeMma(score, make_uint4(k0.x, k1.x, k0.y, k1.y), query[band][0], query[band][1]);
            relativeMma(score, make_uint4(k0.z, k1.z, k0.w, k1.w), query[band][2], query[band][3]);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i)
            score[i] += __shfl_xor_sync(0xffffffffu, score[i], 2);
        float prob[4], rescale[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int h = (pair % 2) * 2 + i;
#pragma unroll
            for (int halfTile = 0; halfTile < 2; ++halfTile) {
                const int token = base + row + halfTile * 8, d = lengths[i] - 1 - token;
                const float bias = unsigned(d) < a.distances && h < heads ? biasRow[i][d] : 0.f;
                float &x = score[halfTile * 2 + i];
                x = token < lengths[i] ? (x * rsqrtf(float(width)) + bias) * taus[i] * RelativeAttention::sm_scale_log2
                                       : -INFINITY;
            }
            const bool live = !Packed || base < lengths[i];
            float m = fmaxf(maximum[i], fmaxf(score[i], score[2 + i]));
#pragma unroll
            for (int d = 4; d < 32; d *= 2) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, d));
            rescale[i] = live ? exp2f(maximum[i] - m) : 1.f;
            maximum[i] = live ? m : maximum[i];
            prob[i] = live ? exp2f(score[i] - m) : 0.f;
            prob[2 + i] = live ? exp2f(score[2 + i] - m) : 0.f;
            float sum = prob[i] + prob[2 + i];
#pragma unroll
            for (int d = 4; d < 32; d *= 2) sum += __shfl_xor_sync(0xffffffffu, sum, d);
            denom[i] = denom[i] * rescale[i] + sum;
        }
        const unsigned p0 = transposeRelativeFragment(packRelativePair(prob[0], prob[1], pair >= 2));
        const unsigned p1 = transposeRelativeFragment(packRelativePair(prob[2], prob[3], pair >= 2));
#pragma unroll
        for (int band = 0; band < 4; ++band) {
            const uint4 v0 = loadRelativeStripe(value[0] + band * 32);
            const uint4 v1 = loadRelativeStripe(value[1] + band * 32);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[band][0][i] *= rescale[i % 2];
                acc[band][1][i] *= rescale[i % 2];
            }
            relativeMma(acc[band][0], make_uint4(transposeRelativeFragment(v0.x), transposeRelativeFragment(v0.y),
                        transposeRelativeFragment(v1.x), transposeRelativeFragment(v1.y)), p0, p1);
            relativeMma(acc[band][1], make_uint4(transposeRelativeFragment(v0.z), transposeRelativeFragment(v0.w),
                        transposeRelativeFragment(v1.z), transposeRelativeFragment(v1.w)), p0, p1);
        }
    }

    __shared__ __align__(16) float values[warps][heads][width];
    __shared__ float maxima[warps][heads], sums[warps][heads];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int h = pair * 2 + i;
        if (!row && h < heads) {
            maxima[warp][h] = maximum[i];
            sums[warp][h] = denom[i];
        }
#pragma unroll
        for (int band = 0; band < 4; ++band) {
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int c = band * 32 + row / 2 * 8 + row % 2 + 4 * u;
                const float x = acc[band][u][i], y = acc[band][u][2+i];
                const float xLow = __shfl_xor_sync(0xffffffffu, x, 2);
                const float yLow = __shfl_xor_sync(0xffffffffu, y, 2);
                if (h < heads) {
                    values[warp][h][c] = x + xLow;
                    values[warp][h][c+2] = y + yLow;
                }
            }
        }
    }
    __syncthreads();
    if (warp < heads) {
        const int h = warp;
        float m = maxima[lane % warps][h];
#pragma unroll
        for (int d = warps / 2; d; d /= 2) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, d));
        const float r = exp2f(maxima[lane % warps][h] - m);
        float sum = sums[lane % warps][h] * r;
#pragma unroll
        for (int d = warps / 2; d; d /= 2) sum += __shfl_xor_sync(0xffffffffu, sum, d);
        float4 v{};
#pragma unroll
        for (int w = 0; w < warps; ++w) {
            const float weight = __shfl_sync(0xffffffffu, r, w);
            const float4 t = *reinterpret_cast<const float4 *>(&values[w][h][lane * 4]);
            v.x += t.x * weight; v.y += t.y * weight;
            v.z += t.z * weight; v.w += t.w * weight;
        }
        const int col = firstColumn + h;
        if constexpr (Packed) {
            if (col >= chain * 3) return;
            if (relativeSplits(a.visit[col / 3].past + 1, splits) != active) return;
            partial += size_t(col / 3) * splits * a.heads * (width + 1);
        }
        const size_t dst = size_t(kvHead * 3 + (Packed ? col % 3 : h)) * splits + split;
        const float inv = 1.f / sum;
        *reinterpret_cast<float4 *>(partial + dst * width + lane * 4) =
            make_float4(v.x * inv, v.y * inv, v.z * inv, v.w * inv);
        if (!lane) partial[size_t(splits) * a.heads * width + dst] = m + log2f(sum);
    }
}
#endif
template <bool Packed = false> __global__ void globalRelativeSm120(AttentionRow a, float *partial, int chain = 1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200 && !defined(CUDA_NO_TENSOR_CORE)
    if constexpr (!Packed) {
        // Each block keeps the original three-head decode layout.
        selectSplitQuery<128>(a, partial, blockIdx.x, gridDim.z);
        relativeMmaTile<false>(a, partial, 1, 0, 0, 0);
    } else {
        // Four independent (candidate, head) columns fit beside their FP16 residuals.
        // A tile spans at most two adjacent queries of the same request and KV head.
        const int tiles = (3 * chain + 3) / 4;
        const int firstColumn = blockIdx.x % tiles * 4;
        selectSplitQuery<128>(a, partial, blockIdx.x / tiles * chain, gridDim.z);
        const int lo = a.visit[firstColumn / 3].past + 1;
        const int hi = a.visit[min((firstColumn + 3) / 3, chain - 1)].past + 1;
        const int loActive = relativeSplits(lo, gridDim.z);
        const int hiActive = relativeSplits(hi, gridDim.z);
        relativeMmaTile<true>(a, partial, chain, firstColumn, hi, hiActive);
        if (loActive != hiActive) {
            // At a short prefix boundary the second query can have a different
            // stripe stride. Replay the first query's partition exactly.
            __syncthreads();
            relativeMmaTile<true>(a, partial, chain, firstColumn, lo, loActive);
        }
    }
#endif
}

// Check both the physical GPU and the virtual architecture of the loaded image.
// A wheel containing only older PTX can still use the ordinary paged CUDA path.
bool supportsGlobalRelativeSm120(int gpu) {
#ifndef CUDA_NO_TENSOR_CORE
    int major, minor;
    check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, gpu));
    if (major != 12) return false;
    check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, gpu));
    if (minor != 0) return false;
    cudaFuncAttributes attr{};
    check(cudaFuncGetAttributes(&attr, globalRelativeSm120<false>));
    return attr.ptxVersion == 120;
#else
    return false;
#endif
}
