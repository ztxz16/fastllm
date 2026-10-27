// Interleave two ordinary Q2 dots one activation at a time, avoiding a second
// set of input loads without keeping an entire block of activations live.
__device__ __forceinline__ float KDotQ2Pair(const uint8_t *record, int row, int secondRow,
        size_t pitch, const float *input, int columns, int warp, int warps,
        float &second) {
    const int lane = threadIdx.x % 32;
    const auto *a = reinterpret_cast<const block_q2_K *>(record + size_t(row) * pitch);
    const auto *b = reinterpret_cast<const block_q2_K *>(record + size_t(secondRow) * pitch);
    float first = 0;
    second = 0;
    for (int block = warp; block < columns / QK_K; block += warps) {
        const float da = __low2float(a[block].dm), ma = __high2float(a[block].dm);
        const float db = __low2float(b[block].dm), mb = __high2float(b[block].dm);
        const unsigned a0 = a[block].qs[lane], a1 = a[block].qs[lane + 32];
        const unsigned b0 = b[block].qs[lane], b1 = b[block].qs[lane + 32];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned sa = a[block].scales[2 * j + lane / 16];
            const unsigned sb = b[block].scales[2 * j + lane / 16];
            const unsigned qa = ((j < 4 ? a0 : a1) >> (2 * (j % 4))) & 3;
            const unsigned qb = ((j < 4 ? b0 : b1) >> (2 * (j % 4))) & 3;
            const float x = input[block * QK_K + lane + 32 * j];
            first = fmaf((da * (sa & 15)) * qa - ma * (sa >> 4), x, first);
            second = fmaf((db * (sb & 15)) * qb - mb * (sb >> 4), x, second);
        }
    }
    for (int mask = 16; mask; mask >>= 1) {
        first += __shfl_down_sync(0xffffffff, first, mask);
        second += __shfl_down_sync(0xffffffff, second, mask);
    }
    return first;
}

// Two adjacent R4 rows share the input and contiguous packed parameters.
// The first row must be even and the record eight-byte aligned. Each dot keeps
// the same lane ownership, FMA order and warp reduction as KDotVector.
__device__ __forceinline__ float KDotQ2R4Pair(const uint8_t *record, int row,
        size_t pitch, const float *input, int columns, int warp, int warps,
        float &second) {
    const int lane = threadIdx.x % 32, r = row % 4;
    const int shift = 8 * (lane % 4) + 2 * ((lane % 16) / 4);
    const auto *blocks = reinterpret_cast<const block_q2_k_r4 *>(record + size_t(row / 4) * 4 * pitch);
    float first = 0;
    second = 0;
    for (int b = warp; b < columns / QK_K; b += warps) {
        const auto &q = blocks[b];
        const float2 d = __half22float2(*reinterpret_cast<const half2 *>(q.d + r));
        const float2 dm = __half22float2(*reinterpret_cast<const half2 *>(q.d + r + 4));
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const unsigned scales = *reinterpret_cast<const uint16_t *>(q.scales + 8 * j + 4 * (lane / 16) + r);
            const uint2 codes = *reinterpret_cast<const uint2 *>(q.qs + 32 * j + 4 * r + 16 * (lane / 16));
            const float x = input[b * QK_K + lane + 32 * j];
            first = fmaf((d.x * (scales & 15)) * ((codes.x >> shift) & 3) - dm.x * ((scales >> 4) & 15), x, first);
            second = fmaf((d.y * ((scales >> 8) & 15)) * ((codes.y >> shift) & 3) - dm.y * (scales >> 12), x, second);
        }
    }
    for (int mask = 16; mask; mask >>= 1) {
        first += __shfl_down_sync(0xffffffff, first, mask);
        second += __shfl_down_sync(0xffffffff, second, mask);
    }
    return first;
}

// Vector loads for GLM's scored Q8_K expert projections. Keep the canonical
// decoder's lane ownership, FMA order and reduction order for both layouts.
// Two rows interleave Gate/Up work so the compiler can reuse input loads.
template<ggml_type Type, bool Native, int Rows = 1>
__device__ __forceinline__ float KDotVector(const uint8_t *record, int row,
        size_t pitch, const float *input, int columns, int warp, int warps,
        int secondRow = 0, float *second = nullptr) {
    const int lane = threadIdx.x % 32;
    const int rows[2] = {row, secondRow};
    float sums[Rows] = {};
    for (int b = warp; b < columns / QK_K; b += warps) {
#pragma unroll
        for (int i = 0; i < Rows; ++i) {
            const int r = rows[i] % 4;
            const uint8_t *base = record + (Native ? size_t(rows[i] / 4) * 4 : size_t(rows[i])) * pitch;
            const uint8_t *block = base + size_t(b) * BlockBytes<Type>() * (Native ? 4 : 1);
            float &sum = sums[i];
            if constexpr (Type == GGML_TYPE_Q2_K) {
                float d, dm;
                unsigned q0 = 0, q1 = 0;
                if constexpr (Native) {
                    const auto &q = *reinterpret_cast<const block_q2_k_r4 *>(block);
                    d = __half2float(q.d[r]); dm = __half2float(q.d[r + 4]);
                } else {
                    const auto &q = *reinterpret_cast<const block_q2_K *>(block);
                    d = __low2float(q.dm); dm = __high2float(q.dm);
                    q0 = q.qs[lane]; q1 = q.qs[lane + 32];
                }
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    unsigned sc, code;
                    if constexpr (Native) {
                        const auto &q = *reinterpret_cast<const block_q2_k_r4 *>(block);
                        sc = q.scales[8 * j + 4 * (lane / 16) + r];
                        code = (q.qs[32 * j + 4 * r + lane % 4 + 16 * (lane / 16)] >> (2 * ((lane % 16) / 4))) & 3;
                    } else {
                        const auto &q = *reinterpret_cast<const block_q2_K *>(block);
                        sc = q.scales[2 * j + lane / 16];
                        code = ((j < 4 ? q0 : q1) >> (2 * (j % 4))) & 3;
                    }
                    sum = fmaf((d * (sc & 15)) * code - dm * (sc >> 4), input[b * 256 + lane + 32 * j], sum);
                }
            } else if constexpr (Type == GGML_TYPE_Q3_K) {
#pragma unroll
                for (int pass = 0; pass < 2; ++pass) {
                    const int first = pass * 128 + lane * 4;
                    const float4 values = *reinterpret_cast<const float4 *>(input + b * 256 + first);
                    float scale;
                    unsigned low, high;
                    int shift, highShift;
                    if constexpr (Native) {
                        const auto &q = *reinterpret_cast<const block_q3_k_r4 *>(block);
                        const int s = 4 * (first / 16) + r, p = first % 32;
                        const int sc = ((q.scales_l[s % 32] >> (4 * (s / 32))) & 15) |
                            (((q.scales_h[s % 16] >> (2 * (s / 16))) & 3) << 4);
                        scale = __half2float(q.d[r]) * (sc - 32);
                        low = *reinterpret_cast<const unsigned *>(q.qs + 32 * (first / 32) + 4 * r + 16 * (p / 16));
                        high = *reinterpret_cast<const unsigned *>(q.qh + 16 * (first / 32) + 4 * r);
                        shift = 2 * ((p % 16) / 4); highShift = p / 4;
                    } else {
                        const auto &q = *reinterpret_cast<const block_q3_K *>(block);
                        const int s = first / 16;
                        const int sc = s < 4 ? (q.scales[s] & 15) | (((q.scales[s + 8] >> 0) & 3) << 4) :
                            s < 8 ? (q.scales[s] & 15) | (((q.scales[s + 4] >> 2) & 3) << 4) :
                            s < 12 ? (q.scales[s - 8] >> 4) | (((q.scales[s] >> 4) & 3) << 4) :
                            (q.scales[s - 8] >> 4) | (((q.scales[s - 4] >> 6) & 3) << 4);
                        scale = __half2float(q.d) * (sc - 32);
                        // Q3_K blocks have only two-byte alignment.
                        memcpy(&low, q.qs + 32 * pass + 4 * (lane % 8), sizeof(low));
                        memcpy(&high, q.hmask + 4 * (lane % 8), sizeof(high));
                        shift = 2 * (lane / 8); highShift = 4 * pass + lane / 8;
                    }
                    const float *v = reinterpret_cast<const float *>(&values);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int code = ((low >> (8 * j + shift)) & 3) + 4 * ((high >> (8 * j + highShift)) & 1) - 4;
                        sum = fmaf(scale * code, v[j], sum);
                    }
                }
            } else {
                const int first = 64 * (lane / 8) + 4 * (lane % 8);
                const float4 lo = *reinterpret_cast<const float4 *>(input + b * 256 + first);
                const float4 hi = *reinterpret_cast<const float4 *>(input + b * 256 + first + 32);
                float d0, m0, d1, m1;
                unsigned q0, q1;
                int shift;
                if constexpr (Native) {
                    const auto &q = *reinterpret_cast<const block_q4_k_r4 *>(block);
                    const int g = first / 32, p = first % 32;
                    const unsigned s0 = q.scales_l[4 * g + r], h0 = q.scales_h[4 * (g % 4) + r] >> (4 * (g / 4));
                    const unsigned s1 = q.scales_l[4 * (g + 1) + r], h1 = q.scales_h[4 * ((g + 1) % 4) + r] >> (4 * ((g + 1) / 4));
                    const float d = __half2float(q.d[r]), dm = __half2float(q.d[r + 4]);
                    d0 = d * ((s0 & 15) + 16 * (h0 & 3)); m0 = dm * ((s0 >> 4) + 16 * ((h0 >> 2) & 3));
                    d1 = d * ((s1 & 15) + 16 * (h1 & 3)); m1 = dm * ((s1 >> 4) + 16 * ((h1 >> 2) & 3));
                    const int offset = 64 * g + 4 * r + 32 * ((p % 8) / 4) + 16 * (p / 16);
                    q0 = *reinterpret_cast<const unsigned *>(q.qs + offset);
                    q1 = *reinterpret_cast<const unsigned *>(q.qs + offset + 64);
                    shift = 4 * ((p % 16) / 8);
                } else {
                    const auto &q = *reinterpret_cast<const block_q4_K *>(block);
                    uint8_t s, m;
                    get_scale_min_k4(2 * (lane / 8), q.scales, s, m);
                    d0 = __low2float(q.dm) * s; m0 = __high2float(q.dm) * m;
                    get_scale_min_k4(2 * (lane / 8) + 1, q.scales, s, m);
                    d1 = __low2float(q.dm) * s; m1 = __high2float(q.dm) * m;
                    q0 = *reinterpret_cast<const unsigned *>(q.qs + 32 * (lane / 8) + 4 * (lane % 8));
                    q1 = q0 >> 4; shift = 0;
                }
                const float *v0 = reinterpret_cast<const float *>(&lo), *v1 = reinterpret_cast<const float *>(&hi);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    sum = fmaf(d0 * ((q0 >> (8 * j + shift)) & 15) - m0, v0[j], sum);
                    sum = fmaf(d1 * ((q1 >> (8 * j + shift)) & 15) - m1, v1[j], sum);
                }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < Rows; ++i)
        for (int mask = 16; mask; mask >>= 1)
            sums[i] += __shfl_down_sync(0xffffffff, sums[i], mask);
    if constexpr (Rows == 2) *second = sums[1];
    return sums[0];
}
