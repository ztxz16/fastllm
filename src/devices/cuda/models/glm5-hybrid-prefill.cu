#include <cuda_fp16.h>
#include <cstdint>
#include "../fastllm-hybrid-prefill.cuh"
#include <algorithm>
#include <cmath>
#include <mutex>
#include <map>

namespace {
// GLM hybrid sparse prefill. BF16 operands are packed losslessly, with FP32
// dot products and a three-component FP16 probability expansion plus residual.
__device__ __forceinline__ void GlmMma1688(float *d, uint32_t a0, uint32_t a1, uint32_t b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(b));
#endif
}
__device__ __forceinline__ uint32_t GlmHalfPair(const half *x) {
    return *reinterpret_cast<const uint32_t *>(x);
}
__device__ __forceinline__ uint32_t GlmHalfPair(half a, half b) {
    return uint32_t(__half_as_ushort(a)) | (uint32_t(__half_as_ushort(b)) << 16);
}
__global__ void GlmPackVectors(const __nv_bfloat16 *x, half *out, int *exponents, int *bad, int vectors) {
    const int v = blockIdx.x, tid = threadIdx.x;
    if (v >= vectors)
        return;
    __shared__ float maxima[128];
    __shared__ int exponent;
    float m = 0;
    bool valid = true;
    for (int d = tid; d < 512; d += 128) {
        float f = __bfloat162float(x[size_t(v) * 512 + d]);
        m = fmaxf(m, fabsf(f));
        valid = valid && isfinite(f);
    }
    maxima[tid] = m;
    __syncthreads();
    for (int s = 64; s; s >>= 1) {
        if (tid < s)
            maxima[tid] = fmaxf(maxima[tid], maxima[tid + s]);
        __syncthreads();
    }
    if (tid == 0) {
        int e = 14 - (maxima[0] > 0 ? ilogbf(maxima[0]) : 0);
        exponent = max(0, min(32, e));
        exponents[v] = exponent;
    }
    __syncthreads();
    for (int d = tid; d < 512; d += 128) {
        float f = ldexpf(__bfloat162float(x[size_t(v) * 512 + d]), exponent);
        half h = __float2half_rn(f);
        out[size_t(v) * 512 + d] = h;
        valid = valid && isfinite(f) && __half2float(h) == f;
    }
    if (!valid)
        atomicExch(bad, 1);
}
constexpr int kGlmPrefillSharedBytes =
    (32 * 520 + 16 * 520 + 3 * 32 * 24) * 2 + (2 * 32 * 16 + 32 * 24 + 32 * 3) * 4 + 16 * 2 * 4;
__global__ void __launch_bounds__(256)
    GlmPrefillAttention(const half *q, const half *kv, const int *qexp, const int *kexp, const int *indices,
                        int count, int cap, int heads, float scale, __nv_bfloat16 *out) {
    extern __shared__ unsigned char bytes[];
    half *queries = reinterpret_cast<half *>(bytes);
    half *keys = queries + 32 * 520;
    half *prob = keys + 16 * 520;
    float *score = reinterpret_cast<float *>(prob + 3 * 32 * 24);
    float *residual = score + 2 * 32 * 16;
    float *state = residual + 32 * 24;
    int *ids = reinterpret_cast<int *>(state + 32 * 3);
    int *exps = ids + 16;
    const int token = blockIdx.y, head = blockIdx.x * 32, tid = threadIdx.x, warp = tid / 32, lane = tid % 32,
              g = lane / 4, t = lane % 4, hg = warp / 4, vg = warp % 4;
    indices += size_t(token) * count;
    q += size_t(token) * heads * 512;
    qexp += size_t(token) * heads;
    for (int i = tid; i < 32 * 64; i += 256)
        *reinterpret_cast<uint4 *>(queries + i / 64 * 520 + (i % 64) * 8) =
            *reinterpret_cast<const uint4 *>(q + size_t(head + i / 64) * 512 + (i % 64) * 8);
    const int queryExponent = qexp[head + tid / 8];
    if (tid < 32) {
        state[tid] = -INFINITY;
        state[32 + tid] = 0.f;
        state[64 + tid] = 0.f;
    }
    float acc[16][4] = {};
    uint4 staged[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        int i = tid + j * 256, c = i / 64, id = c < count ? indices[c] : -1;
        staged[j] = id >= 0 && id < cap
                        ? *reinterpret_cast<const uint4 *>(kv + size_t(id) * 512 + (i % 64) * 8)
                        : make_uint4(0, 0, 0, 0);
    }
    __syncthreads();
    for (int base = 0; base < count; base += 16) {
        if (tid < 16) {
            int id = base + tid < count ? indices[base + tid] : -1;
            ids[tid] = id >= 0 && id < cap ? id : -1;
            exps[tid] = id >= 0 && id < cap ? kexp[id] : 0;
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int i = tid + j * 256;
            *reinterpret_cast<uint4 *>(keys + i / 64 * 520 + (i % 64) * 8) = staged[j];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int i = tid + j * 256, c = base + 16 + i / 64, id = c < count ? indices[c] : -1;
            staged[j] = id >= 0 && id < cap
                            ? *reinterpret_cast<const uint4 *>(kv + size_t(id) * 512 + (i % 64) * 8)
                            : make_uint4(0, 0, 0, 0);
        }

        float dots[4] = {};
        const int split = warp % 2, cgroup = (warp / 2) % 2;
#pragma unroll
        for (int k = 0; k < 32; k++) {
            int d = split * 256 + k * 8 + t * 2;
            uint32_t a0 = GlmHalfPair(queries + (hg * 16 + g) * 520 + d),
                     a1 = GlmHalfPair(queries + (hg * 16 + g + 8) * 520 + d);
            uint32_t b = GlmHalfPair(keys + (cgroup * 8 + g) * 520 + d);
            GlmMma1688(dots, a0, a1, b);
        }
#pragma unroll
        for (int z = 0; z < 4; z++)
            score[split * 32 * 16 + (hg * 16 + g + (z / 2) * 8) * 16 + cgroup * 8 + t * 2 + z % 2] = dots[z];
        __syncthreads();
        // Eight threads per head maintain its online softmax state.
        int h = tid / 8, c0 = tid % 8;
        float m = -INFINITY;
        float values[2];
#pragma unroll
        for (int j = 0; j < 2; j++) {
            int c = c0 + j * 8;
            float v = ids[c] >= 0 ? ldexpf(score[h * 16 + c] + score[32 * 16 + h * 16 + c],
                                           -queryExponent - exps[c]) *
                                        scale
                                  : -INFINITY;
            values[j] = v;
            m = fmaxf(m, v);
        }
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 4));
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
        float maximum = fmaxf(state[h], m), alpha = isfinite(state[h]) ? __expf(state[h] - maximum) : 0.f,
              sum = 0;
#pragma unroll
        for (int j = 0; j < 2; j++) {
            int c = c0 + j * 8;
            float p = isfinite(values[j]) ? __expf(values[j] - maximum) : 0.f;
            sum += p;
            float x = ldexpf(p, 10 - exps[c]);
            half a = __float2half_rn(x);
            float r = (x - __half2float(a)) * 2048.f;
            half b = __float2half_rn(r);
            float rr = (r - __half2float(b)) * 2048.f;
            half tail = __float2half_rn(rr);
            prob[h * 24 + c] = a;
            prob[32 * 24 + h * 24 + c] = b;
            prob[64 * 24 + h * 24 + c] = tail;
            residual[h * 24 + c] = (rr - __half2float(tail)) * (1.f / 4194304.f);
        }
        sum += __shfl_xor_sync(0xffffffff, sum, 4);
        sum += __shfl_xor_sync(0xffffffff, sum, 2);
        sum += __shfl_xor_sync(0xffffffff, sum, 1);
        if (c0 == 0) {
            state[h] = maximum;
            state[32 + h] = state[32 + h] * alpha + sum;
            state[64 + h] = alpha;
        }
        __syncthreads();
#pragma unroll
        for (int n = 0; n < 16; n++)
            for (int z = 0; z < 4; z++)
                acc[n][z] *= state[64 + hg * 16 + g + (z / 2) * 8];
        uint32_t a0[2][3], a1[2][3];
#pragma unroll
        for (int k = 0; k < 2; k++)
            for (int p = 0; p < 3; p++) {
                a0[k][p] = GlmHalfPair(prob + p * 32 * 24 + (hg * 16 + g) * 24 + k * 8 + t * 2);
                a1[k][p] = GlmHalfPair(prob + p * 32 * 24 + (hg * 16 + g + 8) * 24 + k * 8 + t * 2);
            }
#pragma unroll
        for (int n = 0; n < 16; n++) {
            float correction[4] = {}, tail[4] = {};
#pragma unroll
            for (int k = 0; k < 2; k++) {
                int d = vg * 128 + n * 8 + g;
                uint32_t b =
                    GlmHalfPair(keys[(k * 8 + t * 2) * 520 + d], keys[(k * 8 + t * 2 + 1) * 520 + d]);
                GlmMma1688(acc[n], a0[k][0], a1[k][0], b);
                GlmMma1688(correction, a0[k][1], a1[k][1], b);
                GlmMma1688(tail, a0[k][2], a1[k][2], b);
            }
#pragma unroll
            for (int z = 0; z < 4; z++)
                acc[n][z] += (correction[z] + tail[z] * (1.f / 2048.f)) * (1.f / 2048.f);
        }
        // FP32 residuals also retain probabilities below the FP16 expansion range.
        for (int c = 0; c < 16; c++) {
            float r0 = residual[(hg * 16 + g) * 24 + c], r1 = residual[(hg * 16 + g + 8) * 24 + c];
            if (__any_sync(0xffffffff, r0 != 0.f || r1 != 0.f)) {
#pragma unroll
                for (int n = 0; n < 16; n++)
                    for (int z = 0; z < 4; z++)
                        acc[n][z] =
                            fmaf(z < 2 ? r0 : r1,
                                 __half2float(keys[c * 520 + vg * 128 + n * 8 + t * 2 + z % 2]), acc[n][z]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int n = 0; n < 16; n++)
        for (int z = 0; z < 4; z++) {
            int h = hg * 16 + g + (z / 2) * 8, d = vg * 128 + n * 8 + t * 2 + z % 2;
            float denominator = state[32 + h];
            out[(size_t(token) * heads + head + h) * 512 + d] =
                __float2bfloat16_rn(denominator > 0 ? acc[n][z] * (1.f / 1024.f) / denominator : 0.f);
        }
}

} // namespace

// The hybrid attention entry point validates dimensions, strides, devices and
// aliasing before this attempt. A false return recomputes all rows in fallback.
bool FastllmCudaGlm5HybridPrefillAttention(const fastllm::Data &query, const fastllm::Data &latent,
                                           const fastllm::Data &indices, float scale, fastllm::Data &output) {
    if (FastllmCudaRuntimeArch() != 75 || query.dims[1] < 128 || !std::isfinite(scale) || scale <= 0 ||
        scale > 1)
        return false;
    const auto stream = cudaStreamPerThread;
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess || capture != cudaStreamCaptureStatusNone)
        return false;
    // Do not admit a kernel JITed from a pre-Turing PTX fallback body.
    static std::mutex mutex;
    static std::map<int, bool> available;
    {
        std::lock_guard<std::mutex> guard(mutex);
        const int device = FastllmCudaGetDevice();
        auto it = available.find(device);
        if (it == available.end()) {
            cudaFuncAttributes attr{};
            const bool ready =
                cudaFuncGetAttributes(&attr, GlmPrefillAttention) == cudaSuccess && attr.ptxVersion >= 75 &&
                cudaFuncSetAttribute(GlmPrefillAttention, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     kGlmPrefillSharedBytes) == cudaSuccess;
            it = available.emplace(device, ready).first;
        }
        if (!it->second)
            return false;
    }
    const int rows = query.dims[1], heads = query.dims[2];
    const int capacity = latent.dims[1], count = indices.dims[2];
    constexpr int batch = 128;
    const size_t bytes = (size_t(batch) * heads + capacity) * 1028 + sizeof(int);
    if (bytes > 256ull * 1024 * 1024)
        return false;
    void *allocation = nullptr;
    if (FastllmCudaTryMalloc(&allocation, bytes) != FASTLLM_CUDA_TRY_MALLOC_SUCCESS)
        return false;
    struct Scratch {
        void *ptr;
        ~Scratch() {
            if (!FastllmCudaFreeAfterCurrentThreadStream(ptr))
                FastllmCudaFree(ptr);
        }
    } scratch{allocation};
    auto *q = static_cast<half *>(allocation);
    auto *kv = q + size_t(batch) * heads * 512;
    auto *qe = reinterpret_cast<int *>(kv + size_t(capacity) * 512);
    auto *ke = qe + batch * heads, *unsafe = ke + capacity;
    FastllmCudaHybridCheck(cudaMemsetAsync(unsafe, 0, sizeof(int), stream));
    GlmPackVectors<<<capacity, 128, 0, stream>>>(static_cast<const __nv_bfloat16 *>(latent.cudaData), kv, ke,
                                                 unsafe, capacity);
    output.dataType = fastllm::BFLOAT16;
    output.UpdateUnitSize();
    output.Resize(query.dims);
    output.ToDevice(fastllm::DataDevice::CUDA, query.dataDeviceIds, false);
    output.Allocate(false);
    for (int first = 0; first < rows; first += batch) {
        const int n = std::min(batch, rows - first);
        GlmPackVectors<<<n * heads, 128, 0, stream>>>(static_cast<const __nv_bfloat16 *>(query.cudaData) +
                                                          size_t(first) * heads * 512,
                                                      q, qe, unsafe, n * heads);
        int invalid = 0;
        FastllmCudaHybridCheck(
            cudaMemcpyAsync(&invalid, unsafe, sizeof(int), cudaMemcpyDeviceToHost, stream));
        FastllmCudaHybridCheck(cudaStreamSynchronize(stream));
        if (invalid)
            return false;
        GlmPrefillAttention<<<dim3(heads / 32, n), 256, kGlmPrefillSharedBytes, stream>>>(
            q, kv, qe, ke, static_cast<const int *>(indices.cudaData) + size_t(first) * count, count,
            capacity, heads, scale,
            static_cast<__nv_bfloat16 *>(output.cudaData) + size_t(first) * heads * 512);
        FastllmCudaHybridCheck(cudaGetLastError());
    }
    return true;
}
