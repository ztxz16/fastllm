#pragma once

// Included inside the V4.1 kernel namespace. Decode/verify keep their existing
// dispatch; long SM75 prefill reuses unpacked keys and parallelizes the heads.
template<class T>
__global__ void V41IndexerUnpackKeys(const T *keys, const uint8_t *packed,
                                    float *output, int mode, int rowBytes) {
    const int row = blockIdx.x, d = threadIdx.x;
    output[size_t(row) * 128 + d] = packed
        ? V41LoadKvRow(packed + size_t(row) * rowBytes, 128, d, mode)
        : V41Load<T>(keys, size_t(row) * 128 + d);
}

__global__ void V41IndexerPrefillDots(const __nv_bfloat16 *query, const float *keys,
                                     float *dots, int candidates, int first,
                                     int ratio, int startPos) {
    const int token = blockIdx.z, head = blockIdx.y * 8 + threadIdx.x / 32;
    const int visible = min(candidates, (startPos + first + token + 1) / ratio);
    if (blockIdx.x * 32 >= visible) return;
    const int candidate = blockIdx.x * 32 + threadIdx.x % 32;
    __shared__ float keyTile[128][33], queryTile[8][128];
    for (int i = threadIdx.x; i < 32 * 128; i += 256) {
        const int row = i / 128, d = i % 128, source = blockIdx.x * 32 + row;
        keyTile[d][row] = source < visible ? keys[size_t(source) * 128 + d] : 0.f;
    }
    for (int i = threadIdx.x; i < 8 * 128; i += 256)
        queryTile[i / 128][i % 128] = __bfloat162float(
            query[(size_t(token) * 32 + blockIdx.y * 8) * 128 + i]);
    __syncthreads();
    if (candidate >= visible) return;
    float sum = 0.f;
    // Same 128 FMAs, in the same order, as the scalar score kernel.
    #pragma unroll
    for (int d = 0; d < 128; ++d)
        sum += queryTile[threadIdx.x / 32][d] * keyTile[d][threadIdx.x % 32];
    dots[(size_t(token) * 32 + head) * candidates + candidate] = sum;
}

__global__ void V41IndexerPrefillReduce(const float *dots, const float *weights,
                                       float *output, int candidates, int first,
                                       int ratio, int startPos) {
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x, token = blockIdx.y;
    if (candidate >= candidates) return;
    if (candidate >= (startPos + first + token + 1) / ratio) {
        output[size_t(token) * candidates + candidate] = -INFINITY;
        return;
    }
    float sum = 0.f;
    for (int h = 0; h < 32; ++h)
        sum += fmaxf(dots[(size_t(token) * 32 + h) * candidates + candidate], 0.f) * weights[token * 32 + h];
    output[size_t(token) * candidates + candidate] = sum;
}

static bool V41TryIndexerPrefill(const Data &query, const Data &weights,
                                 const Data &keys, int mode, int ratio, int startPos,
                                 Data &output) {
    if (query.dataType != DataType::BFLOAT16 || query.dims[0] != 1 ||
        query.dims[1] < 128 || query.dims[2] != 32 || keys.dims[0] != 1 ||
        weights.Count(0) != size_t(query.dims[1]) * 32 || ratio <= 0 || startPos < 0 ||
        int64_t(startPos) + query.dims[1] > std::numeric_limits<int>::max() ||
        FastllmCudaRuntimeArch() != 75 || FastllmCudaGraphIsCapturing()) return false;
    const int rows = query.dims[1], candidates = keys.dims[1];
    constexpr size_t budget = 32ull * 1024 * 1024;
    const size_t keyBytes = size_t(candidates) * 128 * sizeof(float);
    const size_t rowBytes = size_t(candidates) * 32 * sizeof(float);
    if (!rowBytes || keyBytes >= budget || rowBytes > budget - keyBytes) return false;
    const int batch = std::min(32, int((budget - keyBytes) / rowBytes));
    void *allocation = nullptr;
    if (FastllmCudaTryMalloc(&allocation, keyBytes + batch * rowBytes) != FASTLLM_CUDA_TRY_MALLOC_SUCCESS)
        return false;
    struct Scratch {
        void *ptr;
        ~Scratch() { if (!FastllmCudaFreeAfterCurrentThreadStream(ptr)) FastllmCudaFree(ptr); }
    } scratch{allocation};
    auto *decoded = static_cast<float *>(allocation), *dots = decoded + size_t(candidates) * 128;
    const bool packed = keys.dataType == DataType::INT8;
    if (packed || keys.dataType == DataType::BFLOAT16)
        V41IndexerUnpackKeys<<<candidates, 128, 0, cudaStreamPerThread>>>(
            packed ? nullptr : static_cast<const __nv_bfloat16 *>(keys.cudaData),
            packed ? static_cast<const uint8_t *>(keys.cudaData) : nullptr,
            decoded, mode, keys.dims[2]);
    else
        V41IndexerUnpackKeys<<<candidates, 128, 0, cudaStreamPerThread>>>(
            static_cast<const float *>(keys.cudaData), nullptr, decoded, mode, 0);
    FastllmCudaHybridCheck(cudaGetLastError());
    for (int first = 0; first < rows; first += batch) {
        const int count = std::min(batch, rows - first);
        V41IndexerPrefillDots<<<dim3((candidates + 31) / 32, 4, count), 256, 0, cudaStreamPerThread>>>(
            static_cast<const __nv_bfloat16 *>(query.cudaData) + size_t(first) * 32 * 128,
            decoded, dots, candidates, first, ratio, startPos);
        V41IndexerPrefillReduce<<<dim3((candidates + 255) / 256, count), 256, 0, cudaStreamPerThread>>>(
            dots, static_cast<const float *>(weights.cudaData) + size_t(first) * 32,
            static_cast<float *>(output.cudaData) + size_t(first) * candidates,
            candidates, first, ratio, startPos);
        FastllmCudaHybridCheck(cudaGetLastError());
    }
    return true;
}
