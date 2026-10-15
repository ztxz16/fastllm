// Vector loads for GLM's scored Q8_K expert projections. Keep the canonical
// decoder's lane ownership, FMA order and reduction order for both layouts.
template<ggml_type Type, bool Native>
__device__ __forceinline__ float KDotVector(const uint8_t *record, int row,
        size_t pitch, const float *input, int columns, int warp, int warps) {
    const int lane = threadIdx.x % 32;
    const uint8_t *base = record + (Native ? size_t(row / 4) * 4 : size_t(row)) * pitch;
    const int r = row % 4;
    float sum = 0;
    for (int b = warp; b < columns / QK_K; b += warps) {
        const uint8_t *block = base + size_t(b) * BlockBytes<Type>() * (Native ? 4 : 1);
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
    for (int mask = 16; mask; mask >>= 1) sum += __shfl_down_sync(0xffffffff, sum, mask);
    return sum;
}
