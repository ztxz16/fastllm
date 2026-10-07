#pragma once
// Included inside fastllm_gguf_mmq after the MMQ and Q8 block helpers.

namespace grouped_moe {
// Q2_0 stores four consecutive {-1,0,1,2} codes per byte, with one FP16
// scale per 64 values. Expand only the current weight tile into MMA shared
// memory. Guard the K tail: TP shards may have 320 columns, not 256*k.
template<int Y, int Warps, bool Check>
__device__ void LoadQ2(const char *x, int *tile, const int &kb0,
                       const int &imax, const int &stride) {
#ifdef INT8_MMA_AVAILABLE
    constexpr int Pitch = MMQ_MMA_TILE_X_K_Q8_0;
    auto *scales = reinterpret_cast<float *>(tile + 2*WARP_SIZE);
    for (int row0 = 0; row0 < Y; row0 += Warps) {
        const int row = row0 + threadIdx.y;
        const int srcRow = Check ? min(row, imax) : row;
        const auto *weight = reinterpret_cast<const block_q2_0 *>(x + srcRow*stride);
        // One aligned 16-bit load supplies eight values. Decode both byte
        // vectors in registers instead of issuing two dependent byte loads.
        const int block = kb0 + threadIdx.x/8;
        int2 values = make_int2(0, 0);
        if (block < stride/int(sizeof(block_q2_0))) {
            const uint16_t codes = reinterpret_cast<const uint16_t *>(
                weight[block].qs)[threadIdx.x%8];
            values = gguf_cache_q8::UnpackQ2(codes);
        }
        tile[row*Pitch + 2*threadIdx.x] = values.x;
        tile[row*Pitch + 2*threadIdx.x+1] = values.y;
        if (threadIdx.x < 8) {
            const int block = kb0 + threadIdx.x/2;
            scales[row*Pitch + threadIdx.x] = block < stride/int(sizeof(block_q2_0))
                ? __half2float(weight[block].d) : 0.0f;
        }
    }
#else
    NO_DEVICE_CODE;
#endif
}
} // namespace grouped_moe

template<int X, int Y, int Warps, bool Check>
struct mmq_type_traits<X, Y, Warps, Check, GGML_TYPE_Q2_0> {
    static constexpr load_tiles_mmq_t load_tiles = grouped_moe::LoadQ2<Y, Warps, Check>;
    static constexpr vec_dot_mmq_t vec_dot_mma =
        vec_dot_q8_0_q8_1_mma<X, Y, Warps, MMQ_Q8_1_DS_LAYOUT_D4>;
    // Admission requires NVIDIA SM75+, so the DP4A instantiation is unreachable.
    static constexpr vec_dot_mmq_t vec_dot_dp4a = vec_dot_q8_0_q8_1_dp4a<X, Y, Warps>;
};
