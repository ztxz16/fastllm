#pragma once
// Full E4M3 K/V decode. Adapted from NaiveSGLang's B300 mixed DSA kernel
// (28cfec6fbb). A CTA owns one request and one group of 16 query heads.
// Producers gather independent K/V pages; two MMA warps reduce disjoint
// halves of each tile. The final normalization includes the attention sink.
#include "naive-n05-attention-bulk.cuh"
#if !defined(USE_ROCM) && CUDART_VERSION >= 12080
namespace blackwell_decode {
constexpr int TILE = 32, NSTAGE = 4, NCW = 2, NP = 4;
constexpr int K_STRIDE = 208, V_STRIDE = 144;
constexpr int OFF_V = TILE * K_STRIDE;
constexpr int OFF_VALID = OFF_V + TILE * V_STRIDE;
constexpr int STAGE_BYTES = OFF_VALID + 16;
constexpr int SMEM_BYTES = NSTAGE * STAGE_BYTES + 2 * NSTAGE * sizeof(uint64_t);
constexpr float RESCALE_TH = 8.f;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
__device__ __forceinline__ unsigned smem_u32(const void *p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t *bar, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t *bar, unsigned tx) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)), "r"(tx)
                 : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t *bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t *bar, unsigned parity) {
    asm volatile("{\n"
                 ".reg .pred P1;\n"
                 "LAB_WAIT:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
                 "@P1 bra DONE;\n"
                 "bra LAB_WAIT;\n"
                 "DONE:\n"
                 "}\n" ::"r"(smem_u32(bar)),
                 "r"(parity)
                 : "memory");
}
__device__ __forceinline__ void bulk_g2s(void *dst, const void *src, unsigned bytes, uint64_t *bar) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::"
                 "bytes [%0], [%1], %2, [%3];\n" ::"r"(smem_u32(dst)),
                 "l"(src), "r"(bytes), "r"(smem_u32(bar))
                 : "memory");
}
__device__ __forceinline__ void consumer_sync() {
    asm volatile("bar.sync 1, %0;\n" ::"n"(NCW * 32) : "memory");
}

// two e4m3 (low byte -> low half) -> bf16x2, exact
__device__ __forceinline__ unsigned e4m3x2_bf16x2(unsigned short x) {
    return FastllmNaiveKVRow::ConvertPair(x);
}
__device__ __forceinline__ unsigned lo16_bf16x2(unsigned x) {
    return e4m3x2_bf16x2(static_cast<unsigned short>(x & 0xffffu));
}
__device__ __forceinline__ unsigned hi16_bf16x2(unsigned x) {
    return e4m3x2_bf16x2(static_cast<unsigned short>(x >> 16));
}
__device__ __forceinline__ unsigned prmt(unsigned a, unsigned b, unsigned sel) {
    unsigned r;
    asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
    return r;
}
__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<unsigned *>(&v);
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

#endif
template <bool Split = false>
__global__ void __launch_bounds__((NCW + NP) * 32)
    Decode(const __nv_bfloat16 *q, const __grid_constant__ naive_attention_bulk::Batch batch, const float *sink,
           __nv_bfloat16 *out, int q_heads, float *parts = nullptr, float2 *stats = nullptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    const int b = blockIdx.x, hg = blockIdx.y;
    const FastllmNaiveDSARow row = FastllmNaiveResolveDSARow(batch.rows[b]);
    const int length = naive_dsa_decode::Length(row);
    if (length == 0) {
        if constexpr (Split) {
            // Merge skips empty partitions, so their PV storage is never read.
            if (threadIdx.x < 16)
                stats[((size_t)b * q_heads + hg * 16 + threadIdx.x) * gridDim.z + blockIdx.z] =
                    make_float2(-INFINITY, 0.f);
        } else {
            for (int i = threadIdx.x; i < 16 * 128; i += blockDim.x)
                out[((size_t)b * q_heads + hg * 16) * 128 + i] = __float2bfloat16(0.f);
        }
        return;
    }
    constexpr float qkScale = 0.07216878364870322f; // 1 / sqrt(192)
    const int tiles = (row.selectedCount + TILE - 1) / TILE;
    const int begin = Split ? tiles * blockIdx.z / gridDim.z : 0;
    const int n_tiles = Split ? tiles * (blockIdx.z + 1) / gridDim.z - begin : tiles;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    extern __shared__ __align__(128) uint8_t smem[];
    uint64_t *full = reinterpret_cast<uint64_t *>(smem + NSTAGE * STAGE_BYTES);
    uint64_t *empty = full + NSTAGE;
    __shared__ float s_ml[16][2];
    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < NSTAGE; ++s) {
            mbar_init(&full[s], NP);
            mbar_init(&empty[s], NCW);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (warp >= NCW) {
        constexpr int TPW = TILE / NP;
        const int pw = warp - NCW, tok = pw * TPW + lane;
        for (int it = 0; it < n_tiles; ++it) {
            const int s = it % NSTAGE;
            if (it >= NSTAGE) mbar_wait(&empty[s], ((it / NSTAGE) - 1) & 1);
            uint8_t *st = smem + s * STAGE_BYTES;
            const int slot = (begin + it) * TILE + tok;
            const int idx = lane < TPW && slot < row.selectedCount ? row.indices[slot] : -1;
            const bool valid = idx >= 0 && idx < length;
            unsigned mask = __ballot_sync(0xffffffffu, valid);
            if (lane == 0) {
                st[OFF_VALID + pw] = static_cast<uint8_t>(mask);
                mbar_arrive_expect_tx(&full[s], TPW * (192 + 128));
            }
            __syncwarp();
            if (lane < TPW) {
                const int k = FastllmNaivePhysicalRow(row, valid ? idx : 0);
                const int v = FastllmNaivePhysicalRow(row, valid ? idx : 0, true);
                bulk_g2s(st + tok * K_STRIDE, (const uint8_t *)row.key + (size_t)k * row.keyStride + hg * 192,
                         192, &full[s]);
                bulk_g2s(st + OFF_V + tok * V_STRIDE,
                         (const uint8_t *)row.value + ((size_t)v * (q_heads / 16) + hg) * 128, 128, &full[s]);
            }
        }
        return;
    }
    const int g = warp, r = lane >> 2, c = lane & 3;
    unsigned query[6][2][4];
    const auto *qh = q + ((size_t)b * q_heads + hg * 16) * 192;
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
#pragma unroll
        for (int ch = 0; ch < 6; ++ch) {
            uint4 v = *reinterpret_cast<const uint4 *>(qh + (r + rr * 8) * 192 + ch * 32 + c * 8);
            query[ch][rr][0] = v.x;
            query[ch][rr][1] = v.y;
            query[ch][rr][2] = v.z;
            query[ch][rr][3] = v.w;
        }
    }
    float o_acc[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i)
        o_acc[i][0] = o_acc[i][1] = o_acc[i][2] = o_acc[i][3] = 0.f;
    float m_ref[2] = {-INFINITY, -INFINITY}, l_row[2] = {0.f, 0.f};

    for (int it = 0; it < n_tiles; ++it) {
        const int s = it % NSTAGE;
        mbar_wait(&full[s], (it / NSTAGE) & 1);
        const uint8_t *st = smem + s * STAGE_BYTES;
        const unsigned valid = *reinterpret_cast<const unsigned *>(st + OFF_VALID);
        float sc[2][4]{};
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const uint8_t *key = st + (g * 16 + nt * 8 + r) * K_STRIDE;
#pragma unroll
            for (int ch = 0; ch < 6; ++ch) {
                uint2 v = *reinterpret_cast<const uint2 *>(key + ch * 32 + c * 8);
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const unsigned a[4] = {query[ch][0][2 * e], query[ch][1][2 * e], query[ch][0][2 * e + 1],
                                           query[ch][1][2 * e + 1]};
                    unsigned packed = e == 0 ? v.x : v.y;
                    mma_bf16(sc[nt], a, lo16_bf16x2(packed), hi16_bf16x2(packed));
                }
            }
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int token = g * 16 + nt * 8 + 2 * c + (i & 1);
                // Preserve the model's two BF16 roundings before the online softmax.
                const float dot = __bfloat162float(__float2bfloat16_rn(sc[nt][i]));
                const float score = __bfloat162float(__float2bfloat16_rn(dot * qkScale));
                sc[nt][i] = (valid >> token) & 1u ? score * 1.4426950408889634f : -INFINITY;
            }
        }
        // Online softmax with lazy rescaling.
        float alpha[2];
        bool any_rescale = false;
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            float mx = fmaxf(fmaxf(sc[0][2 * rr], sc[0][2 * rr + 1]), fmaxf(sc[1][2 * rr], sc[1][2 * rr + 1]));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const bool need = mx > m_ref[rr] + RESCALE_TH; // also the first finite max (m_ref = -inf)
            alpha[rr] = need ? exp2f(m_ref[rr] - mx) : 1.f;
            m_ref[rr] = need ? mx : m_ref[rr];
            any_rescale |= need;
        }
        if (__any_sync(0xffffffffu, any_rescale)) {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                o_acc[i][0] *= alpha[0];
                o_acc[i][1] *= alpha[0];
                o_acc[i][2] *= alpha[1];
                o_acc[i][3] *= alpha[1];
            }
        }
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            const float mu = (m_ref[rr] == -INFINITY) ? 0.f : m_ref[rr];
            float ps = 0.f;
#pragma unroll
            for (int nt = 0; nt < 2; ++nt)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const float p = exp2f(sc[nt][2 * rr + e] - mu);
                    sc[nt][2 * rr + e] = p;
                    ps += p;
                }
            l_row[rr] = l_row[rr] * alpha[rr] + ps;
        }

        unsigned pa[4];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            pa[2 * nt] = pack_bf16(sc[nt][0], sc[nt][1]);
            pa[2 * nt + 1] = pack_bf16(sc[nt][2], sc[nt][3]);
        }
        // Accumulate PV with B fragments unpacked directly from FP8 values.
        {
            const uint8_t *vb = st + OFF_V + 4 * r;
            const int t0 = g * 16 + 2 * c;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned w0 = *reinterpret_cast<const unsigned *>(vb + t0 * V_STRIDE + 32 * j);
                const unsigned w1 = *reinterpret_cast<const unsigned *>(vb + (t0 + 1) * V_STRIDE + 32 * j);
                const unsigned w2 = *reinterpret_cast<const unsigned *>(vb + (t0 + 8) * V_STRIDE + 32 * j);
                const unsigned w3 = *reinterpret_cast<const unsigned *>(vb + (t0 + 9) * V_STRIDE + 32 * j);
                const unsigned x01 = prmt(w0, w1, 0x5140u), y01 = prmt(w0, w1, 0x7362u);
                const unsigned x23 = prmt(w2, w3, 0x5140u), y23 = prmt(w2, w3, 0x7362u);
                const unsigned(&a)[4] = pa;
                mma_bf16(o_acc[4 * j + 0], a, lo16_bf16x2(x01), lo16_bf16x2(x23));
                mma_bf16(o_acc[4 * j + 1], a, hi16_bf16x2(x01), hi16_bf16x2(x23));
                mma_bf16(o_acc[4 * j + 2], a, lo16_bf16x2(y01), lo16_bf16x2(y23));
                mma_bf16(o_acc[4 * j + 3], a, hi16_bf16x2(y01), hi16_bf16x2(y23));
            }
        }
        // generic-proxy reads of the stage must be ordered before the async-proxy
        // (bulk copy) refill
        asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty[s]);
    }

    // Merge the two token halves of each head.
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        l_row[rr] += __shfl_xor_sync(0xffffffffu, l_row[rr], 1);
        l_row[rr] += __shfl_xor_sync(0xffffffffu, l_row[rr], 2);
    }
    consumer_sync();                               // every MMA warp is done with the stage buffers (all copies
                                                   // have landed)
    float *mbuf = reinterpret_cast<float *>(smem); // fragment order: [i][lane]
    if (g == 1) {
#pragma unroll
        for (int i = 0; i < 16; ++i)
#pragma unroll
            for (int k = 0; k < 4; ++k)
                mbuf[(i * 4 + k) * 32 + lane] = o_acc[i][k];
        if (c == 0) {
            s_ml[r][0] = m_ref[0];
            s_ml[r][1] = l_row[0];
            s_ml[r + 8][0] = m_ref[1];
            s_ml[r + 8][1] = l_row[1];
        }
    }
    consumer_sync();
    if (g == 0) {
        float sc_self[2], sc_other[2], lsum[2], mmax[2];
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            const float m2 = s_ml[r + rr * 8][0], l2 = s_ml[r + rr * 8][1];
            const float mm = fmaxf(m_ref[rr], m2);
            const float mu = (mm == -INFINITY) ? 0.f : mm;
            sc_self[rr] = exp2f(m_ref[rr] - mu);
            sc_other[rr] = exp2f(m2 - mu);
            lsum[rr] = l_row[rr] * sc_self[rr] + l2 * sc_other[rr];
            mmax[rr] = mm;
        }
        float inv[2];
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            float lt = lsum[rr];
            if (!Split && sink != nullptr) {
                const float sk = sink[hg * 16 + r + rr * 8] * 1.4426950408889634f;
                const float mm = fmaxf(mmax[rr], sk);
                const float mu = (mm == -INFINITY) ? 0.f : mm;
                const float f = exp2f(mmax[rr] - mu);
                lt = lt * f + exp2f(sk - mu);
                sc_self[rr] *= f;
                sc_other[rr] *= f;
            }
            inv[rr] = Split ? 1.f : (lt > 0.f ? 1.f / lt : 0.f);
            if constexpr (Split) {
                if (c == 0)
                    stats[((size_t)b * q_heads + hg * 16 + r + rr * 8) * gridDim.z + blockIdx.z] =
                        make_float2(mmax[rr], lt);
            }
        }
        // lane holds, per row and 32-dim block j, dims 32j + 8c + {0..7}: q < 4 ->
        // o_acc[4j+q][2rr], else [4j+q-4][2rr+1]
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            const int head = hg * 16 + r + rr * 8;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float v[8];
#pragma unroll
                for (int qq = 0; qq < 8; ++qq) {
                    const int i = 4 * j + (qq & 3), k = 2 * rr + (qq >> 2);
                    v[qq] =
                        (o_acc[i][k] * sc_self[rr] + mbuf[(i * 4 + k) * 32 + lane] * sc_other[rr]) * inv[rr];
                }
                if constexpr (Split) {
                    const size_t offset =
                        (((size_t)b * q_heads + head) * gridDim.z + blockIdx.z) * 128 + 32 * j + 8 * c;
                    *reinterpret_cast<float4 *>(parts + offset) = make_float4(v[0], v[1], v[2], v[3]);
                    *reinterpret_cast<float4 *>(parts + offset + 4) = make_float4(v[4], v[5], v[6], v[7]);
                } else {
                    uint4 o;
                    o.x = pack_bf16(v[0], v[1]);
                    o.y = pack_bf16(v[2], v[3]);
                    o.z = pack_bf16(v[4], v[5]);
                    o.w = pack_bf16(v[6], v[7]);
                    *reinterpret_cast<uint4 *>(out + ((long)b * q_heads + head) * 128 + 32 * j + 8 * c) = o;
                }
            }
        }
    }
#endif
}
// One warp reduces four adjacent output channels per lane.
__global__ void Merge(const float *parts, const float2 *stats, const float *sink, __nv_bfloat16 *out, int heads,
                      int splits) {
    const int lane = threadIdx.x, head = blockIdx.x;
    const size_t row = size_t(blockIdx.y) * heads + head;
    const float2 st = lane < splits ? stats[row * splits + lane] : make_float2(-INFINITY, 0.f);
    float maximum = fmaxf(st.x, sink ? sink[head] * 1.4426950408889634f : -INFINITY);
    for (int d = 16; d; d /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(~0u, maximum, d));
    const float mul = st.y > 0.f ? exp2f(st.x - maximum) : 0.f;
    float denom = st.y * mul;
    for (int d = 16; d; d /= 2)
        denom += __shfl_xor_sync(~0u, denom, d);
    if (sink) denom += exp2f(sink[head] * 1.4426950408889634f - maximum);
    float acc[4] = {};
    for (int i = 0; i < splits; ++i) {
        const float scale = __shfl_sync(~0u, mul, i);
        if (scale > 0.f) {
            float4 v = reinterpret_cast<const float4 *>(parts + (row * splits + i) * 128)[lane];
            acc[0] = fmaf(v.x, scale, acc[0]);
            acc[1] = fmaf(v.y, scale, acc[1]);
            acc[2] = fmaf(v.z, scale, acc[2]);
            acc[3] = fmaf(v.w, scale, acc[3]);
        }
    }
    const float inv = denom > 0.f ? 1.f / denom : 0.f;
    for (int i = 0; i < 4; ++i)
        out[row * 128 + 4 * lane + i] = __float2bfloat16(acc[i] * inv);
}
} // namespace blackwell_decode
#endif
