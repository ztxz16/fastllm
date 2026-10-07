#pragma once
// Included inside fastllm_gguf_mmq. Each CUDA translation unit owns its IQ1 table.
// Initialize the table in the same translation unit as the kernels that use it.

static __global__ void initialize_iq1s_grid_gpu() {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= NGRID_IQ1S) {
        return;
    }

    // The canonical table stores eight signed {-1, 0, 1} bytes in a uint64.
    // ik_llama's CUDA table places values 0..3 in the low nibble of four
    // bytes and values 4..7 in their high nibble.  vec_dot_iq1_* extracts
    // those two four-value vectors with masks before feeding them to dp4a.
    const uint64_t source = iq1s_grid[index];
    uint32_t packed = 0;
#pragma unroll
    for (int value = 0; value < 8; ++value) {
        const int8_t signed_value =
            static_cast<int8_t>(source >> (8 * value));
        const int shift = value < 4 ? 8 * value : 8 * (value - 4) + 4;
        packed |= static_cast<uint32_t>(signed_value + 1) << shift;
    }
    iq1s_grid_gpu[index] = packed;
}

static void ensure_iq1s_grid(cudaStream_t stream) {
    static std::once_flag initialized[GGML_CUDA_MAX_DEVICES];
    const int device = ggml_cuda_get_device();
    std::call_once(initialized[device], [stream]() {
        constexpr int threads = 256;
        initialize_iq1s_grid_gpu<<<
            (NGRID_IQ1S + threads - 1) / threads, threads, 0, stream>>>();
        CUDA_CHECK(cudaGetLastError());
        // Initialization is outside steady-state execution and must be visible
        // to every per-thread stream that can subsequently launch an IQ1 op.
        CUDA_CHECK(cudaStreamSynchronize(stream));
    });
}
