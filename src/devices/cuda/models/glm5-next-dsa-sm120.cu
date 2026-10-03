// Host launcher for the pinned, unmodified FlashInfer GLM53_NOPE kernel.
#include "glm5-next-dsa.cuh"
#include <flashinfer/attention/sparse_mla_sm120/compute/q_rope.cuh>
#include <flashinfer/attention/sparse_mla_sm120/kernels/fp8_prefill/prefill_swapab.cuh>
#include <unordered_set>
#include <mutex>

cudaError_t FastllmCudaGlm5NextDsaPrefillSm120Raw(
        const void *query, const void *cache, const int32_t *indices,
        void *output, float *lse, int queries, int width,
        float scale, cudaStream_t stream) {
    if (!query || !cache || !indices || !output || !lse || queries <= 0 ||
        width <= 0 || width % 64) return cudaErrorInvalidValue;
    constexpr int heads = 64;
    constexpr auto model = ModelType::GLM53_NOPE;
    constexpr size_t shared = SmemLayoutSwapAB<model>::TOTAL;
    auto kernel = sparse_mla_prefill_swapab_kernel<model, 64>;
    static std::unordered_set<int> ready;
    static std::mutex mutex;
    int device = 0;
    auto status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!ready.count(device)) {
            int major = 0, minor = 0;
            status = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
            if (status != cudaSuccess) return status;
            status = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
            if (status != cudaSuccess) return status;
            if (major != 12 || minor != 0) return cudaErrorNotSupported;
            status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared);
            if (status != cudaSuccess) return status;
            ready.insert(device);
        }
    }
    PrefillColdParams cold{};
    cold.sm_scale = scale;
    cold.num_tokens = queries;
    cold.kv_stride_bytes = 528;
    cold.out_lse_stride_elems = heads;
    cold.topk = width;
    cold.page_block_size = 1;
    const bf16 *q = static_cast<const bf16 *>(query);
    const uint8_t *kv = static_cast<const uint8_t *>(cache);
    bf16 *out = static_cast<bf16 *>(output);
    const float *sink = nullptr;
    cudaLaunchConfig_t config{dim3(queries), dim3(BLOCK_THREADS), shared, stream, nullptr, 0};
    void *args[] = {&q, &kv, &indices, &sink, &out, &lse, &cold};
    return cudaLaunchKernelExC(&config, (const void *)kernel, args);
}
