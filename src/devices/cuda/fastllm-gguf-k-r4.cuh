#pragma once

// Read one value directly from NUMA's four-row K-quant blocks.
// Shared by dense dequantization and cached expert projections.
#include <cuda_fp16.h>
#include "gguf.h"

template<ggml_type Type>
__device__ __forceinline__ float FastllmGgufKRegroupedValue(const uint8_t *block, int row, int c) {
    static_assert(Type == GGML_TYPE_Q2_K || Type == GGML_TYPE_Q3_K || Type == GGML_TYPE_Q4_K,
                  "Unsupported K-quant R4 decoder");
    const int p = c % 32;
    if constexpr (Type == GGML_TYPE_Q2_K) {
        const auto &q = *reinterpret_cast<const block_q2_k_r4 *>(block);
        const unsigned scale = q.scales[4*(c/16)+row];
        const unsigned code = (q.qs[32*(c/32)+4*row+p%4+16*(p/16)] >> (2*((p%16)/4))) & 3;
        return (__half2float(q.d[row])*(scale&15))*code - __half2float(q.d[row+4])*(scale>>4);
    } else if constexpr (Type == GGML_TYPE_Q3_K) {
        const auto &q = *reinterpret_cast<const block_q3_k_r4 *>(block);
        const int s = 4*(c/16)+row;
        const int scale = ((q.scales_l[s%32] >> (4*(s/32))) & 15) |
            (((q.scales_h[s%16] >> (2*(s/16))) & 3) << 4);
        const int low = (q.qs[32*(c/32)+4*row+p%4+16*(p/16)] >> (2*((p%16)/4))) & 3;
        const int high = (q.qh[16*(c/32)+4*row+p%4] >> (p/4)) & 1;
        return (__half2float(q.d[row])*(scale-32))*(low+4*high-4);
    } else {
        const auto &q = *reinterpret_cast<const block_q4_k_r4 *>(block);
        const int g = c/32;
        const unsigned lo = q.scales_l[4*g+row], hi = q.scales_h[4*(g%4)+row] >> (4*(g/4));
        const int scale = (lo&15)+16*(hi&3), minimum = (lo>>4)+16*((hi>>2)&3);
        const int code = (q.qs[64*g+4*row+p%4+32*((p%8)/4)+16*(p/16)] >> (4*((p%16)/8))) & 15;
        return (__half2float(q.d[row])*scale)*code - __half2float(q.d[row+4])*minimum;
    }
}
