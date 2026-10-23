#pragma once
// Native CUDA V4.1 grouped projections. Keep quantization and SwiGLU in the
// shared indexed launcher; this file only packs Q8_1, multiplies, and scatters.
template<bool DS4>
__global__ void GatherNativeV41(const block_q8_1 *input, block_q8_1_mmq *output,
        const int *groupRoutes, const int *activeRows, int columns, int capacity, int topk) {
    const int row = blockIdx.x*(blockDim.x/32)+threadIdx.x/32;
    if (row >= *activeRows) return;
    const int route = groupRoutes[row], firstBlock = blockIdx.y*4;
    auto *destination = reinterpret_cast<uint32_t *>(output+size_t(blockIdx.y)*capacity+row);
    for (int word = threadIdx.x%32; word < sizeof(block_q8_1_mmq)/sizeof(uint32_t); word += 32) {
        const int block = firstBlock+(word < 4 ? word : (word-4)/8);
        uint32_t value = 0;
        if (route >= 0) {
            const auto &q = input[size_t(route/topk)*(columns/32)+block];
            if (word < 4) {
                if constexpr (DS4) value = *reinterpret_cast<const uint32_t *>(&q.ds);
                else value = __float_as_uint(__low2float(q.ds));
            } else value = reinterpret_cast<const uint32_t *>(q.qs)[(word-4)%8];
        }
        destination[word] = value;
    }
}
__global__ void ScatterNativeV41(const float *products, __nv_bfloat16 *output,
        const int *routeGroups, int width) {
    const int c = blockIdx.x*blockDim.x+threadIdx.x;
    if (c >= width) return;
    const int route = blockIdx.y, group = routeGroups[route];
    output[size_t(route)*width+c] = __float2bfloat16_rn(
        group < 0 ? 0.f : products[size_t(group)*width+c]);
}

static bool NativeV41Matrix(int type, const uint8_t *const *weights, int part,
        Workspace &w, int experts, int columns, int width, cudaStream_t stream) {
    if (type == GGML_TYPE_IQ1_S) ensure_iq1s_grid(stream);
    switch (type) {
#define NATIVE_MATRIX(T) case GGML_TYPE_##T: \
        if (w.inputRows >= 1024) LaunchMatrix<GGML_TYPE_##T, 64, false, true>( \
            weights, part, w, experts, columns, width, stream); \
        else LaunchMatrix<GGML_TYPE_##T, 16, false, true>( \
            weights, part, w, experts, columns, width, stream); break;
        NATIVE_MATRIX(IQ1_S) NATIVE_MATRIX(IQ1_M) NATIVE_MATRIX(Q2_0)
        NATIVE_MATRIX(Q3_K) NATIVE_MATRIX(Q4_K)
#undef NATIVE_MATRIX
        default: return false;
    }
    return true;
}
