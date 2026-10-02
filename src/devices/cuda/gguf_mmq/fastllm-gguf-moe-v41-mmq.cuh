#pragma once
// MMA fragment mapping adapted from mmq.cuh (GGML / Iwan Kawrakow, MIT).
// V4.1 Q8_K must keep its FP32 signed scale. The ordinary MMQ half scale/sum
// metadata can amplify cancellation enough to cross the model's FP8 boundary.
// Reuse MMQ's tiled loop, MMA fragments and write-back with FP32 weight scales
// and integer activation sums computed by MMA instead of rounded half sums.
namespace v41_mmq {
template<int Y, int Warps, bool Check, ggml_type Type>
__device__ void Load(const char *x, int *tile, const int &block,
                    const int &imax, const int &stride) {
#ifdef INT8_MMA_AVAILABLE
    constexpr int Pitch = MMQ_MMA_TILE_X_K_Q2_K;
    auto *scales = reinterpret_cast<float2 *>(tile+64);
    for (int row0 = 0; row0 < Y; row0 += Warps) {
        const int row = row0+threadIdx.y, sourceRow = Check ? min(row, imax) : row;
        using Block = typename std::conditional<Type == GGML_TYPE_Q2_K, block_q2_K, block_q4_K>::type;
        const auto &w = reinterpret_cast<const Block *>(x+sourceRow*stride)[block];
        for (int word = threadIdx.x; word < 64; word += 32) {
            unsigned values = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int c = word*4+j;
                unsigned q;
                if constexpr (Type == GGML_TYPE_Q2_K)
                    q = (w.qs[(c/128)*32+c%32] >> (2*((c%128)/32))) & 3;
                else q = (w.qs[(c/64)*32+c%32] >> (4*((c%64)/32))) & 15;
                values |= q << (8*j);
            }
            tile[row*Pitch+word] = values;
        }
        if (threadIdx.x < 16) {
            const int group = threadIdx.x;
            unsigned s, m;
            if constexpr (Type == GGML_TYPE_Q2_K) {
                s = w.scales[group]&15; m = w.scales[group]>>4;
            } else {
                const int g = group/2;
                s = g < 4 ? w.scales[g]&63 : (w.scales[g+4]&15) | ((w.scales[g-4]>>6)<<4);
                m = g < 4 ? w.scales[g+4]&63 : (w.scales[g+4]>>4) | ((w.scales[g]>>6)<<4);
            }
            const float2 dm = __half22float2(w.dm);
            scales[row*(Pitch/2)+group] = make_float2(dm.x*s, dm.y*m);
        }
    }
#else
    NO_DEVICE_CODE;
#endif
}
template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void DotV41(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int & k00) {
#ifdef INT8_MMA_AVAILABLE

    typedef mma_int_A_I16K4 mma_A;
    typedef mma_int_A_I16K8 mma_A_K8;
    typedef mma_int_B_J8K4  mma_B;
    typedef mma_int_C_I16J8 mma_C;

    constexpr int granularity = mmq_get_granularity_device(mmq_x);
    constexpr int rows_per_warp = 2 * granularity;
    constexpr int ntx = rows_per_warp/mma_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (mma_B::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float2 * x_dm = reinterpret_cast<const float2 *>(x_qs + WARP_SIZE*2);
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = reinterpret_cast<const float *>(y);

    const int i0 = (threadIdx.y / ntx) * (ntx*mma_A::I);

    mma_A   A[ntx][8];
    float  dA[ntx][mma_C::ne/2][8];
    float  mA[ntx][mma_C::ne/2][8];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_1) {
            const int k0 = k00 + k01;

            ((mma_A_K8 *) A[n])[k01/QI8_1].load(x_qs + (i0 + n*mma_A::I)*MMQ_MMA_TILE_X_K_Q2_K + k0, MMQ_MMA_TILE_X_K_Q2_K);
        }
    }

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int i = i0 + n*mma_C::I + mma_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_1/2) {
                const int k0 = k00 + k01;

                const float2 dm = x_dm[i*(MMQ_MMA_TILE_X_K_Q2_K/2) + k0/(QI8_1/2)];

                dA[n][l][k01/(QI8_1/2)] = dm.x;
                mA[n][l][k01/(QI8_1/2)] = dm.y;
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += ntx*mma_C::J) {
        float dB[mma_C::ne/2];

#pragma unroll
        for (int l = 0; l < mma_C::ne/2; ++l) {
            const int j = j0 + mma_C::get_j(l);

            dB[l] = y_df[j*MMQ_TILE_Y_K];
        }

#pragma unroll
        for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_1) {
            mma_B B[2];

            B[0].load(y_qs + j0*MMQ_TILE_Y_K + (k01 + 0),        MMQ_TILE_Y_K);
            B[1].load(y_qs + j0*MMQ_TILE_Y_K + (k01 + mma_B::K), MMQ_TILE_Y_K);

            mma_C Cm[2];
            {
                mma_A A1;
                A1.x[0] = 0x01010101;
                A1.x[1] = 0x01010101;
                Cm[0].mma_K4(A1, B[0]);
                Cm[1].mma_K4(A1, B[1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                mma_C Cd[2];

                Cd[0].mma_K4(A[n][k01/4 + 0], B[0]);
                Cd[1].mma_K4(A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < mma_C::ne; ++l) {
                    float tmp = Cd[0].x[l]*dA[n][l/2][k01/4 + 0] + Cd[1].x[l]*dA[n][l/2][k01/4 + 1];
                    {
                        tmp -= Cm[0].x[l]*mA[n][l/2][k01/4 + 0] + Cm[1].x[l]*mA[n][l/2][k01/4 + 1];
                    }
                    sum[(j0/mma_C::J + n)*mma_C::ne + l] += tmp*dB[l%2];
                }
            }
        }

    }
#else
    GGML_UNUSED(x); GGML_UNUSED(y); GGML_UNUSED(sum);
    NO_DEVICE_CODE;
#endif // INT8_MMA_AVAILABLE
}
} // namespace v41_mmq

template<int X, int Y, int Warps, bool Check, ggml_type Type>
struct v41_mmq_type_traits {
    static_assert(Type == GGML_TYPE_Q2_K || Type == GGML_TYPE_Q4_K, "V4.1 MMQ format");
    static constexpr load_tiles_mmq_t load_tiles = v41_mmq::Load<Y, Warps, Check, Type>;
    static constexpr vec_dot_mmq_t vec_dot_mma = v41_mmq::DotV41<X, Y, Warps>;
    // Host admission requires INT8 MMA; retain a well-formed device template.
    static constexpr vec_dot_mmq_t vec_dot_dp4a = v41_mmq::DotV41<X, Y, Warps>;
};
