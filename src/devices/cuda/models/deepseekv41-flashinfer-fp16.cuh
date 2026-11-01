// Included inside deepseekv41-kernels.cu after the dynamic attention descriptor.
// FlashInfer's FP16 prefill primitive also handles a single sparse query after
// gathering its selected KV rows. No host planning or allocation occurs on replay.
#ifndef USE_ROCM
namespace v41_fp16_attention {
using BaseParams = flashinfer::SinglePrefillParams<half, half, half>;
struct Params : BaseParams { const uint8_t *valid = nullptr; };
struct Attention : flashinfer::DefaultAttention<false, false, false, false> {
    using Base = flashinfer::DefaultAttention<false, false, false, false>;
    using Base::Base;
    REGISTER_LOGITS_MASK(params, batch, query, key, head, kvHead, {
        return key < params.kv_len && params.valid[key];
    })
};
// Two KV warps and a 16-query tile fit the SM75 64-KiB shared-memory limit
// for D512 (50,448 bytes with distributed output). QK and PV accumulate
// into FP32 registers.
using BaseTraits = flashinfer::KernelTraits<flashinfer::MaskMode::kCustom,
    16, 1, 1, 32, 32, 1, 2, flashinfer::PosEncodingMode::kNone,
    half, half, half, float, int, Attention>;
// D512 custom-mask attention can distribute PV output across the KV warps.
// This selects FlashInfer's existing distributed output implementation while
// preserving FP32 QK/PV accumulation and the original eight-way KV partition.
struct Traits : BaseTraits {
    static constexpr bool USE_SINGLE_PREFILL_SOFTMAX_VO_SPLIT = true;
    using SharedStorageSingle = BaseTraits::SharedStoragePaged;
};
static_assert(!Traits::IsInvalid() && sizeof(Traits::SharedStorageSingle) <= 65536);
constexpr int splits = 8;
struct Workspace {
    half *q, *kv, *partial;
    float *lse;
    uint8_t *valid;
    int *fallback;
};
size_t Bytes(int rows, int heads, int keys) {
    return ((size_t(rows) * (heads + keys + splits * heads) * 512 * sizeof(half) + 255) & ~size_t(255)) +
        ((size_t(rows) * splits * heads * sizeof(float) + 255) & ~size_t(255)) +
        ((size_t(rows) * keys + 255) & ~size_t(255)) + size_t(rows) * sizeof(int);
}
Workspace Bind(void *data, int rows, int heads, int keys) {
    Workspace w{};
    auto align = [](void *p) { return (reinterpret_cast<uintptr_t>(p) + 255) & ~uintptr_t(255); };
    w.q = static_cast<half *>(data); w.kv = w.q + size_t(rows) * heads * 512;
    w.partial = w.kv + size_t(rows) * keys * 512;
    w.lse = reinterpret_cast<float *>(align(w.partial + size_t(rows) * splits * heads * 512));
    w.valid = reinterpret_cast<uint8_t *>(align(w.lse + size_t(rows) * splits * heads));
    w.fallback = reinterpret_cast<int *>(align(w.valid + size_t(rows) * keys));
    return w;
}
__device__ half Convert(float value, int *fallback) {
    // FP16 has a narrower exponent range; exceptional rows stay on BF16.
    const half result = __float2half_rn(value);
    if (!isfinite(value) || fabsf(value) > 65504.f) {
        atomicExch(fallback, 1); return __float2half_rn(0.f);
    }
    return result;
}
__global__ void Gather(const V41AttentionDescriptor *items, Workspace w, int heads,
                       int keys, int window) {
    const int row = blockIdx.y, n = blockIdx.x * blockDim.x + threadIdx.x;
    const auto d = items[row];
    const float qValue = n < heads * 512 ? __bfloat162float(d.q[n]) : 0.f;
    if (n < heads * 512)
        w.q[size_t(row) * heads * 512 + n] = Convert(qValue, w.fallback + row);
    const int slot = n / 512, column = n % 512;
    const __nv_bfloat16 *src = nullptr;
    const uint8_t *packed = nullptr;
    int mode = 0;
    if (n < keys * 512 && slot < window) {
        const int position = d.startPos - window + 1 + slot;
        const int chunkStart = d.startPos - d.chunkOffset;
        if (position >= chunkStart) src = d.chunkKV + size_t(position - chunkStart) * 512;
        else if (position >= 0) {
            if (d.ringKV8) { packed = d.ringKV8 + size_t(position % window) * d.ringRowBytes; mode = d.ringMode; }
            else src = d.ringKV + size_t(position % window) * 512;
        }
    } else if (n < keys * 512 && d.indices && slot >= window && slot - window < d.topWidth) {
        const int id = d.indices[slot - window];
        if (id >= 0 && id < d.cap) {
            if (d.compressedKV8) { packed = d.compressedKV8 + size_t(id) * d.cmpRowBytes; mode = d.cmpMode; }
            else src = d.compressedKV + size_t(id) * 512;
        }
    }
    const bool valid = src || packed;
    if (n < keys * 512 && !column) w.valid[size_t(row) * keys + slot] = valid;
    const float value = src ? __bfloat162float(src[column]) : packed ?
        V41LoadKvRow(packed, 512, column, mode) : 0.f;
    if (n < keys * 512)
        w.kv[size_t(row) * keys * 512 + n] = Convert(value, w.fallback + row);
    // FP16 subnormal rounding is harmless next to ordinary activations, but
    // can dominate an entirely tiny vector. Check each 256-value block, with
    // no host synchronization, before allowing Tensor Core computation.
    __shared__ float qMax[8], kvMax[8];
    float qm = fabsf(qValue), km = fabsf(value);
    for (int offset = 16; offset; offset >>= 1) {
        qm = fmaxf(qm, __shfl_xor_sync(0xffffffff, qm, offset));
        km = fmaxf(km, __shfl_xor_sync(0xffffffff, km, offset));
    }
    if (!(threadIdx.x & 31)) { qMax[threadIdx.x / 32] = qm; kvMax[threadIdx.x / 32] = km; }
    __syncthreads();
    if (!threadIdx.x) {
        qm = km = 0.f;
        for (int i = 0; i < 8; ++i) { qm = fmaxf(qm, qMax[i]); km = fmaxf(km, kvMax[i]); }
        if ((qm > 0.f && qm < 1.f/512.f) || (km > 0.f && km < 1.f/512.f))
            atomicExch(w.fallback + row, 1);
    }
}
__global__ __launch_bounds__(64) void Compute(Params p, Workspace w, int keys) {
    const int row = blockIdx.z;
    if (w.fallback[row]) return;
    p.q = w.q + size_t(row) * p.num_qo_heads * 512;
    p.k = p.v = w.kv + size_t(row) * keys * 512;
    p.o = w.partial + size_t(row) * splits * p.num_qo_heads * 512;
    p.lse = w.lse + size_t(row) * splits * p.num_qo_heads;
    p.valid = w.valid + size_t(row) * keys;
    extern __shared__ uint8_t smem[];
    flashinfer::SinglePrefillWithKVCacheDevice<Traits>(p,
        *reinterpret_cast<Traits::SharedStorageSingle *>(smem), threadIdx,
        blockIdx.x, blockIdx.y, 0, gridDim.y, 1);
}
__global__ void Merge(const V41AttentionDescriptor *items, Workspace w, const float *sink, int heads) {
    const int row = blockIdx.y, h = blockIdx.x, t = threadIdx.x;
    if (w.fallback[row]) return;
    const float *lse = w.lse + size_t(row) * splits * heads + h;
    const half *partial = w.partial + (size_t(row) * splits * heads + h) * 512;
    // FlashInfer exposes log2(sum(exp(score))). Count the attention sink once.
    float maximum = sink[h] * 1.4426950408889634f;
    for (int s = 0; s < splits; ++s) maximum = fmaxf(maximum, lse[s * heads]);
    float denominator = exp2f(sink[h] * 1.4426950408889634f - maximum), weight[splits];
    for (int s = 0; s < splits; ++s) {
        weight[s] = lse[s * heads] > -1e29f ? exp2f(lse[s * heads] - maximum) : 0.f;
        denominator += weight[s];
    }
    for (int column = t; column < 512; column += blockDim.x) {
        float value = 0.f;
        for (int s = 0; s < splits; ++s)
            if (weight[s]) value = fmaf(weight[s], __half2float(partial[size_t(s) * heads * 512 + column]), value);
        items[row].output[size_t(h) * 512 + column] = __float2bfloat16_rn(value / denominator);
    }
}
bool Ready() {
    static std::mutex mutex;
    static std::map<int, bool> ready;
    const int device = FastllmCudaGetDevice();
    std::lock_guard<std::mutex> lock(mutex);
    if (ready.count(device)) return ready.at(device);
    if (FastllmCudaGraphIsCapturingFast()) return false;
    cudaFuncAttributes a{}; int limit = 0;
    const bool ok = cudaFuncGetAttributes(&a, Compute) == cudaSuccess && a.ptxVersion >= 75 &&
        a.binaryVersion >= 75 && a.maxThreadsPerBlock >= 64 &&
        cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) == cudaSuccess &&
        sizeof(Traits::SharedStorageSingle) + a.sharedSizeBytes <= size_t(limit) &&
        cudaFuncSetAttribute(Compute, cudaFuncAttributeMaxDynamicSharedMemorySize,
            sizeof(Traits::SharedStorageSingle)) == cudaSuccess;
    if (!ok) cudaGetLastError();
    ready[device] = ok; return ok;
}
bool Run(const V41AttentionDescriptor *items, void *data, int rows, int heads,
         int keys, int window, float scale, const float *sink) {
    const auto w = Bind(data, rows, heads, keys);
    if (cudaMemsetAsync(w.fallback, 0, size_t(rows) * sizeof(int), cudaStreamPerThread) != cudaSuccess) return false;
    Gather<<<dim3((std::max(heads, keys) * 512 + 255) / 256, rows), 256, 0, cudaStreamPerThread>>>(items, w, heads, keys, window);
    Params p;
    static_cast<BaseParams &>(p) = BaseParams(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        heads, 1, 1, keys, heads * 512, 512, 512, 512, 512, -1, 0.f, scale, 1.f, 10000.f);
    p.partition_kv = true;
    Compute<<<dim3((heads + 15) / 16, splits, rows), dim3(32, 1, 2),
        sizeof(Traits::SharedStorageSingle), cudaStreamPerThread>>>(p, w, keys);
    Merge<<<dim3(heads, rows), 128, 0, cudaStreamPerThread>>>(items, w, sink, heads);
    return V41CheckLaunch("FlashInferFP16Attention");
}
__global__ void Descriptors(V41AttentionDescriptor d, V41AttentionDescriptor *items,
                            int rows, int tokens, int heads, int window) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const int b = row / tokens, t = row % tokens;
    d.q += size_t(row) * heads * 512; d.output += size_t(row) * heads * 512;
    d.chunkKV += size_t(b) * tokens * 512;
    if (d.ringKV) d.ringKV += size_t(b) * window * 512;
    if (d.ringKV8) d.ringKV8 += size_t(b) * window * d.ringRowBytes;
    if (d.compressedKV) d.compressedKV += size_t(b) * d.cap * 512;
    if (d.compressedKV8) d.compressedKV8 += size_t(b) * d.cap * d.cmpRowBytes;
    if (d.indices) d.indices += size_t(row) * d.topWidth;
    d.startPos += t; d.chunkOffset = t; items[row] = d;
}
bool Chunk(V41AttentionDescriptor d, int batch, int tokens, int heads, int window,
           float scale, const float *sink) {
    const int rows = batch * tokens;
    // Keep the partition shape identical to a full graph at short contexts.
    const int keys = window + (d.indices ? std::max(512, d.topWidth) : 0);
    const size_t scratchBytes = (Bytes(rows, heads, keys) + 255) & ~size_t(255);
    void *allocation = nullptr;
    if (FastllmCudaTryMalloc(&allocation, scratchBytes + size_t(rows) * sizeof(d)) !=
            FASTLLM_CUDA_TRY_MALLOC_SUCCESS) return false;
    struct Scratch {
        void *data;
        ~Scratch() { if (!FastllmCudaFreeAfterCurrentThreadStream(data)) FastllmCudaFree(data); }
    } scratch{allocation};
    auto *items = reinterpret_cast<V41AttentionDescriptor *>(static_cast<char *>(allocation) + scratchBytes);
    Descriptors<<<(rows + 127) / 128, 128, 0, cudaStreamPerThread>>>(d, items, rows, tokens, heads, window);
    if (!Run(items, allocation, rows, heads, keys, window, scale, sink)) return false;
    const auto w = Bind(allocation, rows, heads, keys);
    V41SparseAttentionKernel<__nv_bfloat16, 8, true, 8, false, true><<<
        dim3(rows, (heads + 7) / 8, 8), 64, 0, cudaStreamPerThread>>>(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, sink,
        1, heads, 512, window, 0, 0, 0, scale, nullptr, 0, 0, 0, 0, items, w.fallback);
    return V41CheckLaunch("FlashInferFP16Chunk");
}
} // namespace v41_fp16_attention
#endif
