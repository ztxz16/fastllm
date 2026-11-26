// Static buffers and device route tiles shared by the Pascal and Volta graph paths.
#pragma once
namespace fastllm::legacy_experts {
namespace {
// offsets[experts+1], tile count, (expert, first sorted row) per tile and
// sorted route indices share the otherwise unused final output buffer.
template <int tileM>
__global__ void makeDecodePlan(const int *ids, int routes, int experts, int maxTiles, int *plan) {
    extern __shared__ int cursor[];
    int *tileCount = plan + experts + 1;
    int *tiles = tileCount + 1, *order = tiles + 2 * maxTiles;
    for (int e = threadIdx.x; e < experts; e += blockDim.x)
        cursor[e] = 0;
    __syncthreads();
    for (int r = threadIdx.x; r < routes; r += blockDim.x)
        atomicAdd(cursor + ids[r], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int offset = 0, tile = 0;
        for (int e = 0; e < experts; ++e) {
            const int count = cursor[e];
            plan[e] = cursor[e] = offset;
            for (int i = 0; i < count; i += tileM) {
                tiles[2 * tile] = e;
                tiles[2 * tile + 1] = offset + i;
                ++tile;
            }
            offset += count;
        }
        plan[experts] = offset;
        *tileCount = tile;
    }
    __syncthreads();
    for (int r = threadIdx.x; r < routes; r += blockDim.x)
        order[atomicAdd(cursor + ids[r], 1)] = r;
}

// The nonzero bytes of four FP16 E2M1 values, in little-endian order.
__device__ __forceinline__ unsigned fp4HighBytes(unsigned q) {
    return __byte_perm(0x3e3c3800, 0x46444240, q & 0x7777) | __byte_perm(0x00008000, 0, (q >> 3) & 0x1111);
}

template <auto kernel, int minPtx> bool hasDecodeKernel() {
    const int device = FastllmCudaGetDevice();
    static thread_local int previous = -1;
    static thread_local bool available = false;
    if (device != previous) {
        cudaFuncAttributes attr{};
        available = cudaFuncGetAttributes(&attr, kernel) == cudaSuccess && attr.ptxVersion >= minPtx;
        previous = device;
    }
    return available;
}

inline bool validDecodeLayout(const Data &input, const Data &output, const Data &ids, const Data &scores,
                              const Data &pair, const Data &middle, int experts, int inter,
                              const CudaIndexedMoeSlots &slots) {
    if (slots.gateScales || slots.downScales || input.dataType != FLOAT32 || output.dataType != FLOAT32 ||
        pair.dataType != FLOAT32 || middle.dataType != FLOAT32 || ids.dataType != INT32 ||
        scores.dataType != FLOAT32 || ids.dims.size() != 2 || ids.dims != scores.dims ||
        input.dims != output.dims || ids.dims[0] != input.dims[0] || output.cudaData == input.cudaData)
        return false;
    const int rows = input.dims[0], hidden = input.dims[1], top = ids.dims[1];
    if (rows <= 0 || top <= 0 || rows > 65535 / top || hidden <= 0 || inter <= 0 || hidden % 16 || inter % 16 ||
        experts <= 0 || experts > 1024)
        return false;
    const int routes = rows * top;
    return pair.Count(0) >= size_t(routes) * std::max(hidden, 2 * inter) &&
           middle.Count(0) >= size_t(routes) * inter;
}

} // namespace
} // namespace fastllm::legacy_experts
