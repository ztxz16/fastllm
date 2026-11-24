// Tiled narrow BF16 projection with the same 256-leaf compensated sum as ExactRows.
#pragma once
template<int Rows, int Cols, int Width>
__global__ void FastllmBf16Narrow(const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                                   __nv_bfloat16 *out, int n, int channels) {
    int t = threadIdx.x, row = blockIdx.y * Rows, col = blockIdx.x * Cols;
    float2 sums[Rows][Cols/2]{};
    for (int offset = t * 8; offset < Width; offset += 2048) {
        uint4 weights[Cols], inputs[Rows];
        #pragma unroll
        for (int c = 0; c < Cols; ++c)
            weights[c] = *reinterpret_cast<const uint4 *>(w + size_t(col+c)*Width + offset);
        #pragma unroll
        for (int r = 0; r < Rows; ++r)
            inputs[r] = *reinterpret_cast<const uint4 *>(x + size_t(min(row+r,n-1))*Width + offset);
        float2 partial[Rows][Cols/2]{};
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            #pragma unroll
            for (int c = 0; c < Cols/2; ++c) {
                float2 wc{float(reinterpret_cast<const __nv_bfloat16 *>(&weights[2*c])[j]),
                          float(reinterpret_cast<const __nv_bfloat16 *>(&weights[2*c+1])[j])};
                #pragma unroll
                for (int r = 0; r < Rows; ++r) {
                    float a = float(reinterpret_cast<const __nv_bfloat16 *>(&inputs[r])[j]);
#if CUDART_VERSION >= 12080 && __CUDA_ARCH__ >= 1000
                    partial[r][c] = __ffma2_rn({a,a}, wc, partial[r][c]);
#else
                    partial[r][c] = {fmaf(a,wc.x,partial[r][c].x), fmaf(a,wc.y,partial[r][c].y)};
#endif
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < Rows; ++r)
            #pragma unroll
            for (int c = 0; c < Cols/2; ++c) {
                sums[r][c].x += partial[r][c].x;
                sums[r][c].y += partial[r][c].y;
            }
    }
    __shared__ float leaves[Rows*Cols][256];
    #pragma unroll
    for (int r = 0; r < Rows; ++r)
        #pragma unroll
        for (int c = 0; c < Cols/2; ++c) {
            leaves[r*Cols+2*c][t] = sums[r][c].x;
            leaves[r*Cols+2*c+1][t] = sums[r][c].y;
        }
    __syncthreads();
    // One warp per result. Each lane folds the eight leaves 32 lanes apart.
    const int lane = t % 32;
    for (int rc = t / 32; rc < Rows*Cols; rc += 8) {
        const float *v = leaves[rc] + lane;
        float s[4], e[4]{};
        auto merge = [](float &s, float &e, float peer) {
            float corrected = peer - e, next = s + corrected;
            e = (next - s) - corrected;
            s = next;
        };
        #pragma unroll
        for (int i = 0; i < 4; ++i) { s[i] = v[32*i]; merge(s[i], e[i], v[32*i+128]); }
        merge(s[0], e[0], s[2]);
        merge(s[1], e[1], s[3]);
        merge(s[0], e[0], s[1]);
        #pragma unroll
        for (int d = 16; d; d /= 2) merge(s[0], e[0], __shfl_down_sync(0xffffffffu, s[0], d));
        if (!lane && row+rc/Cols < n) out[size_t(row+rc/Cols)*channels+col+rc%Cols] = __float2bfloat16_rn(s[0]);
    }
}
