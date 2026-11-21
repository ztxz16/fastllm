// Native FastLLM adapter for the optional FlashMLA SM103 fused attention kernel.
#include "deepseekv41-flashmla.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdlib>
#include <cstring>
#include <flashinfer/attention/sparse_mla_sm120/compute/packed_to_bf16.cuh>

namespace {

using namespace flashinfer::sparse_mla_sm120;

// Treat each independent request as one sparse query. Its last window row
// uses the current BF16 operand directly. Historical rows use their own
// request's ring and compressor, regardless of slot order or pointer reuse.
__global__ void Gather(const FastllmV41FullDecodeRequest *requests, const int *selected,
                       const __nv_bfloat16 *current, __nv_bfloat16 *keys, int *indices, int *positions,
                       int *lengths, int width, int pitch, int ratio) {
    int t = blockIdx.y, vector = threadIdx.x % 64, slot = blockIdx.x * 4 + threadIdx.x / 64;
    if (slot >= pitch)
        return;
    const auto req = requests[t];
    const int visible = ratio ? (req.position + 1) / ratio : 0;
    const int idx = t * pitch + slot;
    if (slot == 0 && vector == 0) {
        positions[t] = t;
        lengths[t] = 128 + min(width, visible);
    }
    const bool extra = slot >= 128;
    const int logical = req.position - 127 + slot;
    int id = -1;
    if (!extra && logical >= 0)
        id = logical % 128;
    if (extra && req.compressedKV && slot - 128 < min(visible, width)) {
        id = selected[size_t(t) * width + slot - 128];
        if (id >= visible)
            id = -1;
    }
    if (!vector)
        indices[idx] = id < 0 ? -1 : idx;
    if (id < 0)
        return;
    const int d = vector * 8;
    auto *out = keys + size_t(idx) * 512 + d;
    if (slot == 127) {
        *reinterpret_cast<uint4 *>(out) = *reinterpret_cast<const uint4 *>(current + size_t(t) * 512 + d);
        return;
    }
    const auto *row =
        static_cast<const uint8_t *>(extra ? req.compressedKV : req.windowKV) + size_t(id) * (extra ? 288 : 528);
    const uint64_t bits = extra ? uint64_t(*reinterpret_cast<const uint32_t *>(row + d / 2))
                                : *reinterpret_cast<const uint64_t *>(row + d);
    const auto scale = decode_e4m3_or_ue8m0_scale_bf16(row[(extra ? 256 : 512) + d / (extra ? 16 : 32)], extra);
    __nv_bfloat162 values[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
        values[j] = __hmul2(decode_e2m1_or_e4m3_pair_bf16(uint32_t(bits >> (j * (extra ? 8 : 16))), extra), scale);
    *reinterpret_cast<uint4 *>(out) = *reinterpret_cast<const uint4 *>(values);
}

// The full decode preparer has both the BF16 current row and its exact
// encoder output. All other rows are accepted in FastLLM's physical packed
// format. Keep raw bytes and scales together, as required by interleaved
// FlashMLA. This also supports independently allocated and reordered caches.
template <bool HistoryOnly>
__global__ void GatherPacked(const FastllmV41FullDecodeRequest *requests, const int *selected,
                             const uint8_t *current, uint8_t *ring, uint8_t *extra, int *indices, int *positions,
                             int *lengths, int width, int pitch, int ratio) {
    const int t = blockIdx.y, slot = blockIdx.x * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (slot >= pitch)
        return;
    const auto req = requests[t];
    const int visible = ratio ? (req.position + 1) / ratio : 0;
    if (!slot && !lane) {
        positions[t] = t;
        lengths[t] = min(width, visible);
    }
    const int col = slot - 128;
    const bool compressed = col >= 0;
    const int destination = compressed ? t * (pitch - 128) + col : t * 128 + slot;
    int id = -1;
    if (compressed) {
        if (col < min(width, visible) && req.compressedKV) {
            id = selected[size_t(t) * width + col];
            if (id >= visible)
                id = -1;
        }
    } else if (req.position - 127 + slot >= 0)
        id = (req.position - 127 + slot) % 128;
    if (!lane)
        indices[t * pitch + slot] = id < 0 ? -1 : destination;
    if (id < 0 || (HistoryOnly && slot == 127))
        return;
    const int bytes = compressed ? 288 : 528;
    const uint8_t *src = slot == 127 ? current + size_t(t) * 528
                                     : static_cast<const uint8_t *>(compressed ? req.compressedKV : req.windowKV) +
                                           size_t(id) * bytes;
    auto *dst = (compressed ? extra : ring) + size_t(destination) * bytes;
    for (int v = lane; v < bytes / 16; v += 32)
        reinterpret_cast<uint4 *>(dst)[v] = reinterpret_cast<const uint4 *>(src)[v];
}

// Fallback consumer for the BF16 output-projection graph. FlashMLA emits
// [token, group, dimension/32, head-in-group, 32] with column-major UE8M0.
__global__ void Unpack(const __nv_fp8_e4m3 *codes, const uint32_t *scales, __nv_bfloat16 *out, int scalePitch, bool headMajor) {
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x;
    const int group = h / 8, block = headMajor ? (h % 8) * 16 + d / 32 : (d / 32) * 8 + h % 8;
    const auto word = scales[(group * 32 + block / 4) * scalePitch + t];
    const int exponent = (word >> (8 * (block % 4))) & 255;
    const float scale = __uint_as_float(uint32_t(exponent) << 23);
    const float value = float(codes[(size_t(t * 8 + group) * 128 + block) * 32 + d % 32]);
    out[(size_t(t) * 64 + h) * 512 + d] = __float2bfloat16_rn(value * scale);
}
} // namespace

bool FastllmV41IsMlaHistoryKernel(const void *func) { return func == (const void *)GatherPacked<true>; }
bool FastllmV41IsMlaPageKernel(const void *func) {
    return func == reinterpret_cast<const void *>(GatherPacked<false>) || FastllmV41IsMlaHistoryKernel(func) || func == reinterpret_cast<const void *>(Gather);
}

FastllmV41MlaWorkspace::~FastllmV41MlaWorkspace() {
    if (!storage)
        return;
    int old = 0;
    cudaGetDevice(&old);
    cudaSetDevice(device);
    cudaFree(storage);
    cudaSetDevice(old);
}

std::shared_ptr<FastllmV41MlaWorkspace> FastllmV41CreateMla(const FastllmV41FullDecodeConfig &c, bool outputOnly) {
    if (c.verifying || c.batch < 1 || c.batch > 128 || c.heads != 64 || c.ropeDim != 64 || c.window != 128 ||
        c.ringMode != 1 || c.ringRowBytes != 528 || c.cmpMode != 3 || c.cmpRowBytes != 288 || c.ratio < 0 ||
        c.ratio > 2 || c.topK < 1 || c.topK > 1024)
        return {};
    auto w = std::make_shared<FastllmV41MlaWorkspace>();
    w->outputOnly = outputOnly;
#ifdef FASTLLM_FLASHMLA_BF16_EPILOGUE
    const char *round = std::getenv("FASTLLM_DSV41_MLA_BF16_ROUND");
    w->bf16Rounding = round && !std::strcmp(round, "1");
    const char *headMajor = std::getenv("FASTLLM_DSV41_MLA_HEAD_MAJOR");
    w->headMajor = headMajor && !std::strcmp(headMajor, "1");
#endif
    cudaGetDevice(&w->device);
    const char *paged = std::getenv("FASTLLM_DSV41_MLA_PAGED");
    w->paged = paged && !std::strcmp(paged, "1");
#ifdef FASTLLM_FLASHMLA_BF16_EPILOGUE
    const char *direct = std::getenv("FASTLLM_DSV41_DIRECT_QUERY");
    const char *stage = std::getenv("FASTLLM_DSV41_STAGE_QUERY");
    w->stageNative = stage && !std::strcmp(stage, "1");
    w->nativeQuery = w->paged && !outputOnly && c.batch <= 16 && direct && !std::strcmp(direct, "1");
#endif
    w->batch = c.batch;
    w->width = c.ratio ? c.topK : 0;
    w->pitch = (128 + w->width + 7) / 8 * 8;
    w->scalePitch = (c.batch + 3) / 4 * 4;
    size_t pos = 0;
    auto layout = [&](uint8_t *base) {
        auto take = [&](size_t bytes) -> void * {
            pos = (pos + 255) & ~size_t(255);
            const size_t off = pos;
            pos += bytes;
            return base ? base + off : nullptr;
        };
        const size_t m = c.batch;
        w->codes = take(m * 32768);
        w->scales = static_cast<uint32_t *>(take(size_t(w->scalePitch) * 256 * 4));
        if (outputOnly)
            return;
        w->query = take(m * 32768 * 2);
        w->keys = take(w->paged ? m * (128 * 528 + (w->pitch - 128) * 288) : m * w->pitch * 1024);
        w->indices = static_cast<int *>(take(m * w->pitch * 4));
        w->angles = static_cast<float *>(take(m * 64 * 4));
        w->positions = static_cast<int *>(take(m * 4));
        w->lengths = static_cast<int *>(take(m * 4));
        w->maxima = static_cast<float *>(take(m * 64 * 4));
        w->lse = static_cast<float *>(take(m * 64 * 4));
    };
    layout(nullptr);
    if (cudaMalloc(&w->storage, pos) != cudaSuccess) {
        cudaGetLastError();
        return {};
    }
    pos = 0;
    layout(static_cast<uint8_t *>(w->storage));
    return w;
}

cudaError_t FastllmV41RunMla(const FastllmV41MlaWorkspace &w, const FastllmV41FullDecodeConfig &c,
                             const FastllmV41FullDecodeBuffers &b, const FastllmV41FullDecodeRequest *requests,
                             cudaStream_t stream, bool unpack) {
    return FastllmV41RunMla(w, c, b, requests, stream, unpack, false);
}

cudaError_t FastllmV41RunMla(const FastllmV41MlaWorkspace &w, const FastllmV41FullDecodeConfig &c,
                             const FastllmV41FullDecodeBuffers &b, const FastllmV41FullDecodeRequest *requests,
                             cudaStream_t stream, bool unpack, bool historyOnly) {
    if (w.outputOnly || c.batch != w.batch || !requests || !b.sink || (unpack && !b.output) ||
        (w.width && !b.topk) || (!w.paged && !b.kv))
        return cudaErrorInvalidValue;
    if (w.paged) {
        if (!b.savedKV)
            return cudaErrorInvalidValue;
        auto *ring = static_cast<uint8_t *>(w.keys);
        auto gather = historyOnly ? GatherPacked<true> : GatherPacked<false>;
        gather<<<dim3((w.pitch + 3) / 4, c.batch), 128, 0, stream>>>(
            requests, static_cast<const int *>(b.topk), static_cast<const uint8_t *>(b.savedKV), ring,
            ring + size_t(c.batch) * 128 * 528, w.indices, w.positions, w.lengths, w.width, w.pitch, c.ratio);
    } else
        Gather<<<dim3((w.pitch + 3) / 4, c.batch), 256, 0, stream>>>(
            requests, static_cast<const int *>(b.topk), static_cast<const __nv_bfloat16 *>(b.kv),
            static_cast<__nv_bfloat16 *>(w.keys), w.indices, w.positions, w.lengths, w.width, w.pitch, c.ratio);
    auto status = cudaGetLastError();
    if (status != cudaSuccess)
        return status;
    FastllmV41MlaCore(w, c, b, stream);
    status = cudaGetLastError();
    if (status == cudaSuccess && unpack)
        Unpack<<<dim3(64, c.batch), 512, 0, stream>>>(static_cast<const __nv_fp8_e4m3 *>(w.codes), w.scales,
                                                      static_cast<__nv_bfloat16 *>(b.output), w.scalePitch, w.headMajor);
    return status == cudaSuccess ? cudaGetLastError() : status;
}
