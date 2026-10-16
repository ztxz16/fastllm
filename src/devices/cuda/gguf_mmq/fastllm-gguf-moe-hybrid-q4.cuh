#pragma once
// Hybrid Q4_K streamed prefill: retain the FP32 scale and accumulation order.
// Reuses the MMA fragment layout from fastllm-gguf-moe-v41-mmq.cuh.
// Load one K fragment at a time to avoid spilling large register arrays;
// integer activation sums use warp shuffles instead of redundant all-ones MMA.
namespace hybrid_q4_mmq {
template <int Y, int Warps, bool Check>
__device__ void Load(const char *x, int *tile, const int &block, const int &imax, const int &stride) {
#ifdef INT8_MMA_AVAILABLE
    constexpr int Pitch = MMQ_MMA_TILE_X_K_Q2_K;
    auto *scales = reinterpret_cast<float2 *>(tile + 64);
    for (int row0 = 0; row0 < Y; row0 += Warps) {
        const int row = row0 + threadIdx.y, sourceRow = Check ? min(row, imax) : row;
        const auto &w = reinterpret_cast<const block_q4_K *>(x + sourceRow * stride)[block];
        for (int word = threadIdx.x; word < 64; word += 32) {
            const unsigned values =
                (unsigned(get_int_b2(w.qs, (word / 16) * 8 + word % 8)) >> (4 * ((word % 16) / 8))) &
                0x0f0f0f0f;
            tile[row * Pitch + word] = values;
        }
        if (threadIdx.x < 16) {
            const int group = threadIdx.x, g = group / 2;
            const unsigned s =
                g < 4 ? w.scales[g] & 63 : (w.scales[g + 4] & 15) | ((w.scales[g - 4] >> 6) << 4);
            const unsigned m =
                g < 4 ? w.scales[g + 4] & 63 : (w.scales[g + 4] >> 4) | ((w.scales[g] >> 6) << 4);
            const float2 dm = __half22float2(w.dm);
            scales[row * (Pitch / 2) + group] = make_float2(dm.x * s, dm.y * m);
        }
    }
#else
    NO_DEVICE_CODE;
#endif
}
template <int mmq_x>
static __device__ __forceinline__ void Dot(const int *__restrict__ x, const int *__restrict__ y,
                                           float *__restrict__ sum, const int &k00) {
#ifdef INT8_MMA_AVAILABLE

    typedef mma_int_A_I16K4 mma_A;
    typedef mma_int_A_I16K8 mma_A_K8;
    typedef mma_int_B_J8K4 mma_B;
    typedef mma_int_C_I16J8 mma_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp / mma_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (mma_B::J * MMQ_TILE_Y_K);

    const int *x_qs = (const int *)x;
    const float2 *x_dm = reinterpret_cast<const float2 *>(x_qs + WARP_SIZE * 2);
    const int *y_qs = (const int *)y + 4;
    const float *y_df = reinterpret_cast<const float *>(y);

    const int i0 = (threadIdx.y / ntx) * (ntx * mma_A::I);

#pragma unroll 1
    for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_1) {
        mma_A A[ntx][2];
        float2 dm[ntx][2][2];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            ((mma_A_K8 *)A[n])[0].load(x_qs + (i0 + n * mma_A::I) * MMQ_MMA_TILE_X_K_Q2_K + k00 + k01,
                                       MMQ_MMA_TILE_X_K_Q2_K);
#pragma unroll
            for (int l = 0; l < 2; ++l) {
                const int i = i0 + n * mma_A::I + mma_C::get_i(2 * l);
#pragma unroll
                for (int b = 0; b < 2; ++b)
                    dm[n][l][b] = x_dm[i * (MMQ_MMA_TILE_X_K_Q2_K / 2) + (k00 + k01) / 4 + b];
            }
        }
#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += ntx * mma_C::J) {
            mma_B B[2];
            B[0].load(y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
            B[1].load(y_qs + j0 * MMQ_TILE_Y_K + k01 + mma_B::K, MMQ_TILE_Y_K);
            float dB[2];
#pragma unroll
            for (int l = 0; l < 2; ++l)
                dB[l] = y_df[(j0 + mma_C::get_j(l)) * MMQ_TILE_Y_K];
            int sums[2][2];
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                int v = __dp4a(B[b].x[0], 0x01010101, 0);
                v += __shfl_xor_sync(0xffffffff, v, 1);
                v += __shfl_xor_sync(0xffffffff, v, 2);
                sums[b][0] = __shfl_sync(0xffffffff, v, 8 * (threadIdx.x % 4));
                sums[b][1] = __shfl_sync(0xffffffff, v, 8 * (threadIdx.x % 4) + 4);
            }
#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                mma_C Cd[2];
                Cd[0].mma_K4(A[n][0], B[0]);
                Cd[1].mma_K4(A[n][1], B[1]);
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    float tmp = Cd[0].x[l] * dm[n][l / 2][0].x + Cd[1].x[l] * dm[n][l / 2][1].x;
                    tmp -= sums[0][l % 2] * dm[n][l / 2][0].y + sums[1][l % 2] * dm[n][l / 2][1].y;
                    sum[(j0 / mma_C::J + n) * mma_C::ne + l] += tmp * dB[l % 2];
                }
            }
        }
    }

#else
    GGML_UNUSED(x);
    GGML_UNUSED(y);
    GGML_UNUSED(sum);
    NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}
} // namespace hybrid_q4_mmq

template <int X, int Y, int Warps, bool Check, ggml_type Type> struct hybrid_q4_mmq_type_traits {
    static_assert(Type == GGML_TYPE_Q4_K, "Hybrid Q4_K MMQ format");
    static constexpr load_tiles_mmq_t load_tiles = hybrid_q4_mmq::Load<Y, Warps, Check>;
    static constexpr vec_dot_mmq_t vec_dot_mma = hybrid_q4_mmq::Dot<X>;
    // Host admission requires INT8 MMA; retain a well-formed device template.
    static constexpr vec_dot_mmq_t vec_dot_dp4a = hybrid_q4_mmq::Dot<X>;
};
