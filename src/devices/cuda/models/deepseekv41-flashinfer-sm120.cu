// C++ adapter for FlashInfer's DSV4.1 BF16-compute mixed FP8/FP4 attention.
// Third-party kernel provenance and the pointer-indirection patch are recorded
// in third_party/flashinfer_dsv41/README.md. No Python runtime is required.
#include "deepseekv41-flashinfer.cuh"
#include <flashinfer/attention/sparse_mla_sm120/kernels/dsv41_bf16/decode.cuh>
#include <cmath>
#include <memory>
#include <map>
#include <climits>

namespace {
using namespace flashinfer::sparse_mla_sm120;
using namespace flashinfer::sparse_mla_sm120::kernels::dsv41_bf16;
struct Plan {
    int width = 0, splits = 0, ratio = 0, heads = 0, batch = 0, scratchHeads = 0;
    float scale = 1.f;
    bool verifying = false;
    void *storage = nullptr;
    std::shared_ptr<void> owner;
    int device = -1;
    bool gathered = false;
    bf16 *decoded = nullptr;
    int32_t *decodedIndices = nullptr;
    bf16 *mid = nullptr;
    float *midLse = nullptr;
    void BindStorage() {
        const size_t count = size_t(batch) * scratchHeads * splits;
        mid = static_cast<bf16 *>(storage);
        midLse = reinterpret_cast<float *>(mid + count * 512);
        decoded = gathered ? reinterpret_cast<bf16 *>(midLse + count) : nullptr;
        decodedIndices = gathered ? reinterpret_cast<int32_t *>(decoded + size_t(batch) * (128 + width) * 512) : nullptr;
    }
};

__device__ inline void DecodeAttentionKVVector(const uint8_t *src, int d,
        bool compressed, bf16 *output) {
    uint64_t raw = 0;
    __nv_bfloat162 scale = __floats2bfloat162_rn(0.f, 0.f);
    if (src) {
        raw = compressed ? uint64_t(*reinterpret_cast<const uint32_t *>(src + d / 2))
                         : *reinterpret_cast<const uint64_t *>(src + d);
        scale = decode_e4m3_or_ue8m0_scale_bf16(
            src[(compressed ? 256 : 512) + d / (compressed ? 16 : 32)], compressed);
    }
    __nv_bfloat162 values[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        values[j] = __hmul2(
            decode_e2m1_or_e4m3_pair_bf16(uint32_t(raw >> (j * (compressed ? 8 : 16))), compressed), scale);
    *reinterpret_cast<uint4 *>(output) = *reinterpret_cast<uint4 *>(values);
}

// Decode each selected packed row once, then reuse it across query-head groups.
// This performs the same BF16 multiply as the packed attention gather.
template <bool Verify>
__global__ void GatherAttentionKV(const FastllmV41FullDecodeRequest *requests,
        const int32_t *selected, bf16 *decoded, int32_t *indices,
        int width, int rows, int ratio) {
    const int count = 128 + width;
    const int task = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = task / 64, vector = task % 64;
    if (row >= rows * count) return;
    const int token = row / count, logical = row % count, d = vector * 8;
    const auto request = requests[token];
    const bool compressed = logical >= 128;
    int id = -1;
    if (compressed && request.compressedKV) {
        const int k = logical - 128;
        const int visible = ratio ? (request.position + 1) / ratio : 0;
        const int length = request.chunkLength > 0 ? request.chunkLength : rows;
        const int end = Verify ? requests[request.chunkBegin].position + length : request.position + 1;
        if (k < (ratio ? min(end / ratio, width) : 0)) {
            id = selected[size_t(token) * width + k];
            if (id < 0 || id >= visible) id = -1;
        }
    } else if (!compressed) {
        const int p = request.position - 127 + logical;
        const int end = Verify ? requests[request.chunkBegin].position : request.position;
        if (p >= 0 && p < end) id = p % 128;
    }
    if (!vector) indices[row] = id < 0 ? -1 : row;
    const auto *src = id < 0 ? nullptr :
        static_cast<const uint8_t *>(compressed ? request.compressedKV : request.windowKV)
            + size_t(id) * (compressed ? 288 : 528);
    DecodeAttentionKVVector(src, d, compressed, decoded + size_t(row) * 512 + d);
}

// Merge old-KV splits, sink, and the unquantized current token in one pass.
// Only the final result is rounded to BF16; QK and the weighted sum use FP32.
__global__ void MergeCurrent(const bf16 *q, const bf16 *current, const bf16 *partial, const float *lse,
                             const float *sink, bf16 *output, float scale, int heads, int scratchHeads,
                             int splits) {
    const int h = blockIdx.x, t = threadIdx.x, token = blockIdx.y;
    q += (size_t(token) * heads + h) * 512;
    current += size_t(token) * 512;
    partial += (size_t(token) * scratchHeads + h) * splits * 512;
    lse += (size_t(token) * scratchHeads + h) * splits;
    output += (size_t(token) * heads + h) * 512;
    float dot = 0.f;
#pragma unroll
    for (int d = 0; d < 4; ++d)
        dot = fmaf(float(q[t * 4 + d]), float(current[t * 4 + d]), dot);
    for (int s = 16; s; s >>= 1)
        dot += __shfl_down_sync(0xffffffff, dot, s);
    __shared__ float sums[4], weights[34], currentWeight;
    if ((t & 31) == 0)
        sums[t / 32] = dot;
    __syncthreads();
    if (t < 32) {
        const float score = (sums[0] + sums[1] + sums[2] + sums[3]) * scale * 1.4426950408889634f;
        const float sinkScore = sink[h] * 1.4426950408889634f;
        float maximum = fmaxf(score, sinkScore);
        for (int i = t; i < splits; i += 32)
            maximum = fmaxf(maximum, lse[i]);
        for (int s = 16; s; s >>= 1)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, s));
        float denominator = 0.f;
        for (int i = t; i < splits; i += 32) {
            float w = lse[i] > -1e29f ? exp2f(lse[i] - maximum) : 0.f;
            weights[i] = w;
            denominator += w;
        }
        for (int s = 16; s; s >>= 1)
            denominator += __shfl_xor_sync(0xffffffff, denominator, s);
        const float cw = exp2f(score - maximum);
        const float inv = 1.f / (denominator + cw + exp2f(sinkScore - maximum));
        for (int i = t; i < splits; i += 32)
            weights[i] *= inv;
        if (!t)
            currentWeight = cw * inv;
    }
    __syncthreads();
    float acc[4];
#pragma unroll
    for (int d = 0; d < 4; ++d)
        acc[d] = currentWeight * float(current[t * 4 + d]);
    for (int i = 0; i < splits; ++i) {
        if (weights[i] == 0.f)
            continue;
        const uint2 packed = *reinterpret_cast<const uint2 *>(partial + size_t(i) * 512 + t * 4);
        const auto *pairs = reinterpret_cast<const __nv_bfloat162 *>(&packed);
#pragma unroll
        for (int d = 0; d < 2; ++d) {
            const float2 v = __bfloat1622float2(pairs[d]);
            acc[d * 2] += weights[i] * v.x;
            acc[d * 2 + 1] += weights[i] * v.y;
        }
    }
    uint2 result;
    auto *pairs = reinterpret_cast<__nv_bfloat162 *>(&result);
    pairs[0] = __floats2bfloat162_rn(acc[0], acc[1]);
    pairs[1] = __floats2bfloat162_rn(acc[2], acc[3]);
    *reinterpret_cast<uint2 *>(output + t * 4) = result;
}

// Each verification row attends to the BF16 prefix of the same causal chunk.
// The packed-cache kernel excludes those rows, so no ring commit or temporary
// quantization is needed. Keep QK, softmax and the weighted sum in FP32.
__global__ void MergeVerify(const bf16 *q, const bf16 *chunk, const bf16 *partial, const float *lse,
                            const float *sink, bf16 *output, float scale, int heads, int scratchHeads,
                            int splits, int length, const FastllmV41FullDecodeRequest *requests = nullptr) {
    const int h = blockIdx.x, rowIndex = blockIdx.y, t = threadIdx.x;
    const int token = requests ? rowIndex - requests[rowIndex].chunkBegin : rowIndex % length;
    const int lane = t & 31, warp = t / 32;
    chunk += size_t(rowIndex - token) * 512;
    q += (size_t(rowIndex) * heads + h) * 512;
    partial += (size_t(rowIndex) * scratchHeads + h) * splits * 512;
    lse += (size_t(rowIndex) * scratchHeads + h) * splits;
    output += (size_t(rowIndex) * heads + h) * 512;
    // Plan creation bounds both the causal chunk and the split count.
    __shared__ float prefixWeights[128], splitWeights[18];
    for (int row = warp; row <= token; row += 4) {
        float dot = 0.f;
#pragma unroll
        for (int d = lane; d < 512; d += 32)
            dot = fmaf(float(q[d]), float(chunk[size_t(row) * 512 + d]), dot);
        for (int s = 16; s; s >>= 1)
            dot += __shfl_down_sync(0xffffffff, dot, s);
        if (!lane)
            prefixWeights[row] = dot * scale * 1.4426950408889634f;
    }
    __syncthreads();
    if (t < 32) {
        const float sinkScore = sink[h] * 1.4426950408889634f;
        float maximum = sinkScore;
        for (int row = t; row <= token; row += 32)
            maximum = fmaxf(maximum, prefixWeights[row]);
        for (int i = t; i < splits; i += 32)
            maximum = fmaxf(maximum, lse[i]);
        for (int s = 16; s; s >>= 1)
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, s));
        float denominator = 0.f;
        for (int row = t; row <= token; row += 32) {
            const float w = exp2f(prefixWeights[row] - maximum);
            prefixWeights[row] = w;
            denominator += w;
        }
        for (int i = t; i < splits; i += 32) {
            const float w = lse[i] > -1e29f ? exp2f(lse[i] - maximum) : 0.f;
            splitWeights[i] = w;
            denominator += w;
        }
        for (int s = 16; s; s >>= 1)
            denominator += __shfl_xor_sync(0xffffffff, denominator, s);
        const float inv = 1.f / (denominator + exp2f(sinkScore - maximum));
        for (int row = t; row <= token; row += 32)
            prefixWeights[row] *= inv;
        for (int i = t; i < splits; i += 32)
            splitWeights[i] *= inv;
    }
    __syncthreads();
    float acc[4] = {};
    for (int row = 0; row <= token; ++row) {
#pragma unroll
        for (int d = 0; d < 4; ++d)
            acc[d] = fmaf(prefixWeights[row], float(chunk[size_t(row) * 512 + t * 4 + d]), acc[d]);
    }
    for (int i = 0; i < splits; ++i) {
        if (splitWeights[i] == 0.f)
            continue;
#pragma unroll
        for (int d = 0; d < 4; ++d)
            acc[d] = fmaf(splitWeights[i], float(partial[size_t(i) * 512 + t * 4 + d]), acc[d]);
    }
    uint2 result;
    auto *pairs = reinterpret_cast<__nv_bfloat162 *>(&result);
    pairs[0] = __floats2bfloat162_rn(acc[0], acc[1]);
    pairs[1] = __floats2bfloat162_rn(acc[2], acc[3]);
    *reinterpret_cast<uint2 *>(output + t * 4) = result;
}

template <bool Verify, bool Requests = true, bool Decoded = false>
auto AttentionKernel(int heads, bool compressed) {
    return heads == 64 ? (compressed ? sparse_mla_decode_dsv41_bf16_kernel<64, true, Requests, Verify, Decoded>
                                    : sparse_mla_decode_dsv41_bf16_kernel<64, false, Requests, Verify, Decoded>)
                       : (compressed ? sparse_mla_decode_dsv41_bf16_kernel<0, true, Requests, Verify, Decoded>
                                     : sparse_mla_decode_dsv41_bf16_kernel<0, false, Requests, Verify, Decoded>);
}

template <bool Verify, bool Requests = true, bool Decoded = false>
bool KernelReady(int heads, bool compressed) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
        return false;
    // Initialize each device/specialization once per host worker, outside capture.
    static thread_local std::map<std::pair<int, int>, bool> ready;
    const auto key = std::make_pair(device, (heads == 64 ? 2 : 0) + int(compressed));
    auto found = ready.find(key);
    if (found != ready.end())
        return found->second;
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone)
        return false;
    int major = 0, minor = 0;
    bool ok = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess &&
              cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) == cudaSuccess &&
              major == 12 && minor >= 0 && minor <= 1;
    if (ok)
        ok = cudaFuncSetAttribute(AttentionKernel<Verify, Requests, Decoded>(heads, compressed),
            cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Dsv41Bf16Smem)) == cudaSuccess;
    if (!ok)
        cudaGetLastError();
    ready.emplace(key, ok);
    return ok;
}
} // namespace

void *FastllmV41FlashInferCreate(const FastllmV41FullDecodeConfig &c) {
    // Independent decode requests and one contiguous causal verification
    // chunk share the packed-cache kernel. Other cache formats retain native math.
    if (c.batch < 1 || c.batch > 128 || c.heads < 1 || c.heads > 128 || c.window != 128 ||
        c.ringMode != 1 || c.ringRowBytes != 528 || c.cmpMode != 3 || c.cmpRowBytes != 288 || c.ratio < 0 ||
        c.ratio > 2 || c.topK < 1 || c.topK > 1024 || !std::isfinite(c.softmaxScale) || c.softmaxScale <= 0.f)
        return nullptr;
    if (!(c.verifying ? KernelReady<true>(c.heads, c.ratio != 0)
                      : KernelReady<false>(c.heads, c.ratio != 0)))
        return nullptr;
    auto p = std::make_unique<Plan>();
    // Head groups share the same selected KV. Bound temporary storage so
    // long chunks retain the packed path without a large per-layer allocation.
    const size_t decodedRows = size_t(c.batch) * (128 + c.topK);
    p->gathered = c.ratio && c.heads >= 32 && c.topK >= 128 &&
        decodedRows * (1024 + sizeof(int32_t)) <= (32u << 20);
    if (p->gathered && !KernelReady<false, false, true>(c.heads, true)) p->gathered = false;
    p->width = c.ratio ? c.topK : 0;
    p->splits = 2 + (p->width + 63) / 64;
    p->ratio = c.ratio;
    p->scale = c.softmaxScale;
    p->heads = c.heads;
    p->batch = c.batch;
    p->verifying = c.verifying;
    p->scratchHeads = (c.heads + 15) / 16 * 16;
    // cudaMalloc aligns the BF16 split tensor; each split occupies 1024 bytes.
    const size_t midCount = size_t(c.batch) * p->scratchHeads * p->splits;
    const size_t selectedRows = p->gathered ? decodedRows : 0;
    const size_t baseBytes = midCount * (512 * sizeof(bf16) + sizeof(float));
    const size_t bytes = baseBytes + selectedRows * (1024 + sizeof(int32_t));
    if (cudaMalloc(&p->storage, bytes) != cudaSuccess) {
        cudaGetLastError();
        // Optional decoded KV must not make a formerly supported plan fail.
        if (!p->gathered || cudaMalloc(&p->storage, baseBytes) != cudaSuccess) {
            cudaGetLastError();
            return nullptr;
        }
        p->gathered = false;
    }
    cudaGetDevice(&p->device);
    p->owner = std::shared_ptr<void>(p->storage, [device = p->device](void *data) {
        int old = 0; cudaGetDevice(&old); cudaSetDevice(device);
        cudaFree(data); cudaSetDevice(old);
    });
    p->BindStorage();
    return p.release();
}

bool FastllmV41FlashInferShareWorkspace(void *handle, void *source) {
    auto *p = static_cast<Plan *>(handle), *q = static_cast<Plan *>(source);
    if (!p || !q || p->device != q->device || p->batch != q->batch ||
        p->scratchHeads != q->scratchHeads || p->splits != q->splits ||
        p->width != q->width || p->gathered != q->gathered || !q->owner) return false;
    p->owner = q->owner;
    p->storage = p->owner.get();
    p->BindStorage();
    return true;
}

void FastllmV41FlashInferDestroy(void *plan) {
    delete static_cast<Plan *>(plan);
}

cudaError_t FastllmV41FlashInferRun(void *handle, const FastllmV41FullDecodeBuffers &b,
                                    const FastllmV41FullDecodeRequest *request, cudaStream_t stream) {
    auto *p = static_cast<Plan *>(handle);
    if (!p || !request || !b.q || !b.kv || !b.output || !b.sink || (p->width && !b.topk))
        return cudaErrorInvalidValue;
    if (p->gathered) {
        const int n = p->batch * (128 + p->width);
        auto gather = p->verifying ? GatherAttentionKV<true> : GatherAttentionKV<false>;
        gather<<<(n * 64 + 255) / 256, 256, 0, stream>>>(request,
            static_cast<const int32_t *>(b.topk), p->decoded, p->decodedIndices, p->width, p->batch, p->ratio);
        const cudaError_t gathered = cudaGetLastError();
        if (gathered != cudaSuccess) return gathered;
        auto kernel = AttentionKernel<false, false, true>(p->heads, true);
        kernel<<<dim3(p->batch, p->scratchHeads / 16, p->splits), Dsv41Bf16Resources::BLOCK_THREADS, sizeof(Dsv41Bf16Smem), stream>>>(
            static_cast<const bf16 *>(b.q), reinterpret_cast<const uint8_t *>(p->decoded), p->decodedIndices,
            p->mid, p->midLse, nullptr, reinterpret_cast<const uint8_t *>(p->decoded), p->decodedIndices + 128,
            nullptr, p->width, 1, 1024, p->batch, p->heads, 128, p->splits, 1, p->scale, 1024,
            128 + p->width, 128 + p->width, Dsv41Pages{1, false}, nullptr, nullptr, sizeof(void *), nullptr, 0, nullptr, nullptr);
    } else {
        auto kernel = p->verifying ? AttentionKernel<true>(p->heads, p->width != 0)
                                   : AttentionKernel<false>(p->heads, p->width != 0);
        kernel<<<dim3(p->batch, p->scratchHeads / 16, p->splits), Dsv41Bf16Resources::BLOCK_THREADS, sizeof(Dsv41Bf16Smem), stream>>>(
            static_cast<const bf16 *>(b.q), nullptr, nullptr, p->mid, p->midLse, nullptr, nullptr,
            p->width ? static_cast<const int32_t *>(b.topk) : nullptr, nullptr, p->width, 1, 288, p->batch,
            p->heads, 128, p->splits, 1, p->scale, 528, 128, p->width, Dsv41Pages{1, false}, &request->windowKV,
            p->width ? &request->compressedKV : nullptr, sizeof(*request), &request->position, p->ratio,
            p->verifying ? &request->chunkBegin : nullptr, p->verifying ? &request->chunkLength : nullptr);
    }
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
        return status;
    // Keep a single-query graph on the ordinary one-token reduction order.
    if (p->verifying || p->batch == 1) {
        MergeVerify<<<dim3(p->heads, p->batch), 128, 0, stream>>>(
            static_cast<const bf16 *>(b.q), static_cast<const bf16 *>(b.kv), p->mid, p->midLse,
            static_cast<const float *>(b.sink), static_cast<bf16 *>(b.output), p->scale, p->heads,
            p->scratchHeads, p->splits, p->batch, p->verifying ? request : nullptr);
    } else {
        MergeCurrent<<<dim3(p->heads, p->batch), 128, 0, stream>>>(
            static_cast<const bf16 *>(b.q), static_cast<const bf16 *>(b.kv), p->mid, p->midLse,
            static_cast<const float *>(b.sink), static_cast<bf16 *>(b.output), p->scale, p->heads,
            p->scratchHeads, p->splits);
    }
    return cudaGetLastError();
}

namespace {
// Dense caches use batch-major storage. Convert local row IDs to global IDs
// for FlashInfer's ordinary gather; no host upload or temporary KV copy.
__global__ void PrepareChunkIndices(FastllmV41FlashInferChunkConfig c, const int32_t *selected,
                                    int32_t *ring, int32_t *compressed) {
    const int token = blockIdx.x, i = threadIdx.x;
    const int batch = token / c.length, position = c.startPosition + token % c.length;
    if (i < 128) {
        const int p = position - 127 + i;
        ring[size_t(token) * 128 + i] = p >= 0 && p < c.startPosition ? batch * 128 + p % 128 : -1;
    }
    for (int k = i; k < c.topK; k += blockDim.x) {
        const int id = selected[size_t(token) * c.topK + k];
        compressed[size_t(token) * c.topK + k] = id >= 0 && id < c.compressedRows ?
            batch * c.compressedRows + id : -1;
    }
}
// The ordinary entry owns contiguous batch-major caches. Keep its existing
// index-list and causal-prefix semantics while sharing restored rows across heads.
__global__ void GatherChunkAttentionKV(FastllmV41FlashInferChunkConfig c,
        const uint8_t *ring, const uint8_t *compressed, const int32_t *selected,
        bf16 *decoded, int32_t *indices) {
    const int count = 128 + c.topK;
    const int task = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = task / 64, d = (task % 64) * 8;
    if (row >= c.batches * c.length * count) return;
    const int token = row / count, logical = row % count, batch = token / c.length;
    const bool fromCompressed = logical >= 128;
    int id = -1;
    if (fromCompressed && compressed) {
        const int selectedId = selected[size_t(token) * c.topK + logical - 128];
        if (selectedId >= 0 && selectedId < c.compressedRows)
            id = batch * c.compressedRows + selectedId;
    } else if (!fromCompressed && ring) {
        const int p = c.startPosition + token % c.length - 127 + logical;
        if (p >= 0 && p < c.startPosition) id = batch * 128 + p % 128;
    }
    if (!d) indices[row] = id < 0 ? -1 : row;
    const uint8_t *src = id < 0 ? nullptr :
        (fromCompressed ? compressed : ring) + size_t(id) * (fromCompressed ? 288 : 528);
    DecodeAttentionKVVector(src, d, fromCompressed, decoded + size_t(row) * 512 + d);
}
struct ChunkLayout {
    int heads, splits, tokens;
    bool gathered;
    size_t count, decodedOffset, bytes;
    explicit ChunkLayout(const FastllmV41FlashInferChunkConfig &c)
        : heads((c.heads + 15) / 16 * 16), splits(2 + (c.topK + 63) / 64), tokens(c.batches * c.length),
          gathered(c.heads >= 32 && c.topK >= 128 &&
              size_t(tokens) * (128 + c.topK) * (1024 + sizeof(int32_t)) <= (32u << 20)),
          count(size_t(tokens) * heads * splits) {
        bytes = count * (512 * sizeof(bf16) + sizeof(float)) +
            size_t(tokens) * (128 + c.topK) * sizeof(int32_t);
        decodedOffset = (bytes + 15) & ~size_t(15);
        if (gathered) bytes = decodedOffset + size_t(tokens) * (128 + c.topK) * 1024;
    }
};

bool ValidChunkConfig(const FastllmV41FlashInferChunkConfig &c) {
    return c.batches >= 1 && c.length >= 1 && c.length <= 128 && int64_t(c.batches) * c.length <= 128 &&
           c.heads >= 1 && c.heads <= 128 && c.topK >= 0 && c.topK <= 1024 && c.compressedRows >= 0 &&
           (!c.topK || c.compressedRows > 0) && c.compressedRows <= INT_MAX / c.batches &&
           c.startPosition >= 0 && c.startPosition <= INT_MAX - c.length &&
           std::isfinite(c.scale) && c.scale > 0.f;
}
} // namespace

size_t FastllmV41FlashInferChunkWorkspace(const FastllmV41FlashInferChunkConfig &c) {
    if (!ValidChunkConfig(c) || !KernelReady<false, false>(c.heads, c.topK != 0)) return 0;
    const ChunkLayout layout(c);
    if (layout.gathered && !KernelReady<false, false, true>(c.heads, true)) return 0;
    return layout.bytes;
}

cudaError_t FastllmV41FlashInferChunkRun(const FastllmV41FlashInferChunkConfig &c,
    const FastllmV41FullDecodeBuffers &b, const void *ring, const void *compressed,
    void *workspace, cudaStream_t stream) {
    if (!ValidChunkConfig(c) || !workspace || !b.q || !b.kv || !b.output || !b.sink || (c.startPosition && !ring) ||
        (c.topK && (!compressed || !b.topk))) return cudaErrorInvalidValue;
    const ChunkLayout l(c);
    auto *mid = static_cast<bf16 *>(workspace);
    auto *lse = reinterpret_cast<float *>(mid + l.count * 512);
    auto *ringIndices = reinterpret_cast<int32_t *>(lse + l.count);
    cudaError_t status;
    if (l.gathered) {
        auto *decoded = reinterpret_cast<bf16 *>(static_cast<uint8_t *>(workspace) + l.decodedOffset);
        const int n = l.tokens * (128 + c.topK);
        GatherChunkAttentionKV<<<(n * 64 + 255) / 256, 256, 0, stream>>>(c,
            static_cast<const uint8_t *>(ring), static_cast<const uint8_t *>(compressed),
            static_cast<const int32_t *>(b.topk), decoded, ringIndices);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        auto kernel = AttentionKernel<false, false, true>(c.heads, true);
        kernel<<<dim3(l.tokens, l.heads / 16, l.splits), Dsv41Bf16Resources::BLOCK_THREADS, sizeof(Dsv41Bf16Smem), stream>>>(
            static_cast<const bf16 *>(b.q), reinterpret_cast<const uint8_t *>(decoded), ringIndices, mid, lse, nullptr,
            reinterpret_cast<const uint8_t *>(decoded), ringIndices + 128, nullptr,
            c.topK, 1, 1024, l.tokens, c.heads, 128, l.splits, 1, c.scale, 1024,
            128 + c.topK, 128 + c.topK, Dsv41Pages{1, false}, nullptr, nullptr, sizeof(void *), nullptr, 0, nullptr, nullptr);
    } else {
        auto *cmpIndices = ringIndices + size_t(l.tokens) * 128;
        PrepareChunkIndices<<<l.tokens, 128, 0, stream>>>(c, static_cast<const int32_t *>(b.topk), ringIndices, cmpIndices);
        status = cudaGetLastError();
        if (status != cudaSuccess) return status;
        auto kernel = AttentionKernel<false, false>(c.heads, c.topK != 0);
        kernel<<<dim3(l.tokens, l.heads / 16, l.splits), Dsv41Bf16Resources::BLOCK_THREADS, sizeof(Dsv41Bf16Smem), stream>>>(
            static_cast<const bf16 *>(b.q), static_cast<const uint8_t *>(ring), ringIndices, mid, lse, nullptr,
            static_cast<const uint8_t *>(compressed), c.topK ? cmpIndices : nullptr, nullptr,
            c.topK, 1, 288, l.tokens, c.heads, 128, l.splits, 1, c.scale, 528, 128, c.topK, Dsv41Pages{1, false},
            nullptr, nullptr, sizeof(void *), nullptr, 0, nullptr, nullptr);
    }
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    MergeVerify<<<dim3(c.heads, l.tokens), 128, 0, stream>>>(static_cast<const bf16 *>(b.q),
        static_cast<const bf16 *>(b.kv), mid, lse, static_cast<const float *>(b.sink),
        static_cast<bf16 *>(b.output), c.scale, c.heads, l.heads, l.splits, c.length);
    return cudaGetLastError();
}
