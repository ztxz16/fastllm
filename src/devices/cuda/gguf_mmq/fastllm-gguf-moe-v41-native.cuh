#pragma once
// Native CUDA V4.1 grouped projections. Keep quantization and SwiGLU in the
// shared indexed launcher; this file only packs Q8_1, multiplies, and scatters.
namespace grouped_moe {
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
} // namespace grouped_moe
