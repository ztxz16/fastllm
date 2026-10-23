#pragma once

// Normalize the window/chunk/compressed layout once per call, rather than
// repeatedly decoding each selected packed key for every query/head group.
__global__ void V41PrefillKeys(const __nv_bfloat16 *chunk, const __nv_bfloat16 *ring,
                              const __nv_bfloat16 *compressed, const uint8_t *ring8,
                              const uint8_t *compressed8, __nv_bfloat16 *output,
                              int rows, int ringRows, int ringMode, int ringBytes,
                              int cmpMode, int cmpBytes, int *unsafe) {
    const int row = blockIdx.x;
    for (int d = threadIdx.x; d < 512; d += blockDim.x) {
        float value;
        if (row < ringRows)
            value = ring8 ? V41LoadKvRow(ring8 + size_t(row) * ringBytes, 512, d, ringMode)
                          : __bfloat162float(ring[size_t(row) * 512 + d]);
        else if (row < ringRows + rows)
            value = __bfloat162float(chunk[size_t(row - ringRows) * 512 + d]);
        else {
            const int at = row - ringRows - rows;
            value = compressed8 ? V41LoadKvRow(compressed8 + size_t(at) * cmpBytes, 512, d, cmpMode)
                                : __bfloat162float(compressed[size_t(at) * 512 + d]);
        }
        const auto bf16 = __float2bfloat16_rn(value);
        output[size_t(row) * 512 + d] = bf16;
        // Extreme packed scales can produce values below the BF16 range.
        // Preserve the scalar path in that case instead of rounding twice.
        if (__bfloat162float(bf16) != value) atomicExch(unsafe, 1);
    }
}

__global__ void V41PrefillCandidates(const int32_t *compressed, int32_t *output,
                                    int rows, int window, int start, int ringRows,
                                    int capacity, int top) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, token = blockIdx.y;
    const int width = window + top;
    if (c >= width) return;
    int id = -1;
    if (c < window) {
        const int position = start + token, first = max(0, position - window + 1);
        if (c <= position - first) {
            const int p = first + c;
            id = p >= start ? ringRows + p - start : p % window;
        }
    } else {
        const int selected = compressed[size_t(token) * top + c - window];
        if (selected >= 0 && selected < capacity) id = ringRows + rows + selected;
    }
    output[size_t(token) * width + c] = id;
}

static bool V41TrySm75PrefillAttention(const Data &q, const Data &chunk,
                                      const Data *ring, const Data *compressed,
                                      const Data *indices, const Data &sink,
                                      int window, int start, float scale,
                                      int ringMode, int cmpMode, Data &output) {
    const int rows = q.dims[1], heads = q.dims[2];
    if (q.dataType != DataType::BFLOAT16 || q.dims[0] != 1 || rows < 128 ||
        (heads != 32 && heads != 64) || window < 0 || window > 4096 || start < 0 ||
        int64_t(start) + rows > std::numeric_limits<int>::max() ||
        chunk.dims != std::vector<int>({1, rows, 512}) || sink.Count(0) < heads ||
        q.multiDeviceData || chunk.multiDeviceData || sink.multiDeviceData ||
        chunk.dataDeviceIds != q.dataDeviceIds || sink.dataDeviceIds != q.dataDeviceIds ||
        !std::isfinite(scale) || scale <= 0 || scale > 1 ||
        FastllmCudaGraphIsCapturing() || !V41IsSm75()) return false;
    if (ring && (ring->multiDeviceData || ring->dims[0] != 1 || ring->dataDeviceIds != q.dataDeviceIds)) return false;
    const bool hasCompressed = compressed && indices;
    if (hasCompressed && (compressed->multiDeviceData || indices->multiDeviceData ||
        compressed->dims[0] != 1 || indices->dims[0] != 1 || indices->dims[1] != rows ||
        compressed->dataDeviceIds != q.dataDeviceIds || indices->dataDeviceIds != q.dataDeviceIds)) return false;
    const int ringRows = ring ? window : 0, cap = hasCompressed ? compressed->dims[1] : 0;
    const int top = hasCompressed ? indices->dims[2] : 0;
    if (top < 0 || top > 4096 - window) return false;
    const int width = window + top;
    const int64_t total = int64_t(ringRows) + rows + cap;
    const size_t keyBytes = size_t(total) * 512 * sizeof(__nv_bfloat16);
    const size_t indexBytes = size_t(rows) * width * sizeof(int32_t);
    const size_t scratchBytes = keyBytes + indexBytes + sizeof(int);
    // Leave ample room for the generic MMA packer's separate bounded scratch.
    if (!width || total > std::numeric_limits<int>::max() ||
        scratchBytes > 64ull * 1024 * 1024) return false;
    void *allocation = nullptr;
    if (FastllmCudaTryMalloc(&allocation, scratchBytes) != FASTLLM_CUDA_TRY_MALLOC_SUCCESS) return false;
    struct Scratch {
        void *ptr;
        ~Scratch() { if (!FastllmCudaFreeAfterCurrentThreadStream(ptr)) FastllmCudaFree(ptr); }
    } scratch{allocation};
    Data latent(DataType::BFLOAT16, {1, int(total), 512}), selected(DataType::INT32, {1, rows, width});
    for (Data *view : {&latent, &selected}) {
        view->dataDevice = DataDevice::CUDA; view->dataDeviceIds = q.dataDeviceIds;
        view->cudaDataBorrowed = true;
    }
    latent.cudaData = allocation;
    selected.cudaData = static_cast<uint8_t *>(allocation) + keyBytes;
    auto *unsafe = reinterpret_cast<int *>(static_cast<uint8_t *>(allocation) + keyBytes + indexBytes);
    FastllmCudaHybridCheck(cudaMemsetAsync(unsafe, 0, sizeof(int), cudaStreamPerThread));
    const bool ringPacked = ring && ring->dataType == DataType::INT8;
    const bool cmpPacked = hasCompressed && compressed->dataType == DataType::INT8;
    V41PrefillKeys<<<int(total), 128, 0, cudaStreamPerThread>>>(
        static_cast<const __nv_bfloat16 *>(chunk.cudaData),
        ring && !ringPacked ? static_cast<const __nv_bfloat16 *>(ring->cudaData) : nullptr,
        hasCompressed && !cmpPacked ? static_cast<const __nv_bfloat16 *>(compressed->cudaData) : nullptr,
        ringPacked ? static_cast<const uint8_t *>(ring->cudaData) : nullptr,
        cmpPacked ? static_cast<const uint8_t *>(compressed->cudaData) : nullptr,
        static_cast<__nv_bfloat16 *>(latent.cudaData), rows, ringRows,
        ringMode, ringPacked ? ring->dims[2] : 0, cmpMode, cmpPacked ? compressed->dims[2] : 0, unsafe);
    V41PrefillCandidates<<<dim3((width + 255) / 256, rows), 256, 0, cudaStreamPerThread>>>(
        hasCompressed ? static_cast<const int32_t *>(indices->cudaData) : nullptr,
        static_cast<int32_t *>(selected.cudaData), rows, window, start, ringRows, cap, top);
    FastllmCudaHybridCheck(cudaGetLastError());
    int invalid = 0;
    FastllmCudaHybridCheck(cudaMemcpyAsync(&invalid, unsafe, sizeof(int), cudaMemcpyDeviceToHost, cudaStreamPerThread));
    FastllmCudaHybridCheck(cudaStreamSynchronize(cudaStreamPerThread));
    if (invalid) return false;
    return FastllmCudaSm75PrefillAttention(q, latent, selected, scale, output, &sink);
}
