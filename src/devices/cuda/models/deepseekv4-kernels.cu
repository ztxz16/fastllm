#include "fastllm-cuda.cuh"
#include "fastllm.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <mutex>

namespace {

constexpr int kDeepSeekV4Sparse1MCompressedKeys = 256 * 1024;
constexpr int kDeepSeekV4SparseMaxKeys = kDeepSeekV4Sparse1MCompressedKeys + 64 * 1024;
constexpr int kDeepSeekV4SparseDecodeMaxKeys = kDeepSeekV4SparseMaxKeys;
constexpr int kDeepSeekV4SparsePrefillMaxKeys = kDeepSeekV4SparseMaxKeys;
constexpr size_t kDeepSeekV4SparsePrefillDefaultTempBytes = 256ULL * 1024ULL * 1024ULL;

struct DeepSeekV4RouteTableCacheEntry {
    void *cudaData = nullptr;
    const void *sourceData = nullptr;
    uint64_t count = 0;
    fastllm::DataType dataType = fastllm::DataType::FLOAT32;
    bool verifyContent = false;
    uint64_t fingerprint = 0;
    // A changed non-model table can still have an in-flight kernel referring
    // to the old address. Keep superseded replicas alive until the owning Data
    // is retired instead of making them immediately reusable.
    std::vector<void*> retiredCudaData;
};

using DeepSeekV4RouteTableDeviceCache =
    std::map<int, DeepSeekV4RouteTableCacheEntry>;
using DeepSeekV4RouteTableCache =
    std::map<const fastllm::Data*, DeepSeekV4RouteTableDeviceCache,
             std::less<const fastllm::Data*> >;

std::mutex &DeepSeekV4RouteTableCacheMutex() {
    // Data objects can be destroyed by language bindings during shared-library
    // teardown. Keep the registry primitives alive until process exit so a
    // late Data destructor never touches an already-destroyed static mutex.
    static std::mutex *mutex = new std::mutex();
    return *mutex;
}

DeepSeekV4RouteTableCache &DeepSeekV4RouteTableCaches() {
    // Entries are explicitly retired by Data::~Data. Intentionally retain the
    // empty registry object itself to avoid cross-translation-unit static
    // destruction ordering hazards in Python extension shutdown.
    static DeepSeekV4RouteTableCache *cache = new DeepSeekV4RouteTableCache();
    return *cache;
}

uint64_t DeepSeekV4RouteTableFingerprint(const void *data, size_t bytes) {
    const uint8_t *p = static_cast<const uint8_t*>(data);
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= p[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

const void *DeepSeekV4GetCudaRouteTable(fastllm::Data &routeTable) {
    if (routeTable.dataDevice == fastllm::DataDevice::CUDA &&
        routeTable.cudaData != nullptr) {
        return routeTable.cudaData;
    }
    if (routeTable.dataDevice != fastllm::DataDevice::CPU ||
        routeTable.cpuData == nullptr) {
        return nullptr;
    }

    uint64_t count = routeTable.Count(0);
    if (count == 0 || count > SIZE_MAX / sizeof(int32_t)) {
        return nullptr;
    }
    size_t bytes = (size_t)count * sizeof(int32_t);
    // Loaded model weights are immutable after construction, so hashing a
    // multi-megabyte route table on every decode token would be pure overhead.
    // Non-model tensors remain mutable and are fingerprinted on every lookup.
    bool verifyContent = !routeTable.isModelWeight;
    uint64_t fingerprint = verifyContent ?
        DeepSeekV4RouteTableFingerprint(routeTable.cpuData, bytes) : 0;
    int device = FastllmCudaGetDevice();

    std::lock_guard<std::mutex> guard(DeepSeekV4RouteTableCacheMutex());
    DeepSeekV4RouteTableCacheEntry &entry =
        DeepSeekV4RouteTableCaches()[&routeTable][device];
    bool matches = entry.cudaData != nullptr &&
                   entry.sourceData == routeTable.cpuData &&
                   entry.count == count &&
                   entry.dataType == routeTable.dataType &&
                   entry.verifyContent == verifyContent &&
                   (!verifyContent || entry.fingerprint == fingerprint);
    if (matches) {
        return entry.cudaData;
    }

    void *next = FastllmCudaMalloc(bytes);
    if (next == nullptr) {
        return nullptr;
    }
    FastllmCudaCopyFromHostToDevice(next, routeTable.cpuData, bytes);
    if (entry.cudaData != nullptr) {
        entry.retiredCudaData.push_back(entry.cudaData);
    }
    entry.cudaData = next;
    entry.sourceData = routeTable.cpuData;
    entry.count = count;
    entry.dataType = routeTable.dataType;
    entry.verifyContent = verifyContent;
    entry.fingerprint = fingerprint;
    return entry.cudaData;
}

size_t DeepSeekV4SparsePrefillTempBytesLimit() {
    size_t ret = kDeepSeekV4SparsePrefillDefaultTempBytes;
    if (const char *env = std::getenv("FASTLLM_DSV4_SPARSE_PREFILL_TEMP_MB")) {
        int mb = std::atoi(env);
        if (mb > 0) {
            ret = (size_t)mb * 1024ULL * 1024ULL;
        }
    }
    return ret;
}

inline int DeepSeekV4HcPreDotParts(int flatDim, int threads) {
    return std::min(16, std::max(1, (flatDim + threads * 4 - 1) / (threads * 4)));
}

inline int DeepSeekV4HcPreFinishParts(int dim, int threads) {
    return std::min(16, std::max(1, (dim + threads - 1) / threads));
}

template <typename Kernel>
bool DeepSeekV4EnsureDynamicSharedMemory(Kernel kernel, size_t sharedBytes) {
    constexpr size_t kDefaultDynamicSharedLimit = 48 * 1024;
    if (sharedBytes <= kDefaultDynamicSharedLimit) {
        return true;
    }
    int device = 0;
    cudaError_t state = cudaGetDevice(&device);
    if (state != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    int maxOptinShared = 0;
    state = cudaDeviceGetAttribute(&maxOptinShared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (state != cudaSuccess || sharedBytes > (size_t)maxOptinShared) {
        cudaGetLastError();
        return false;
    }
    state = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sharedBytes);
    if (state != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
}

__device__ __forceinline__ float Dsv4ToFloat(float v) {
    return v;
}

__device__ __forceinline__ float Dsv4ToFloat(half v) {
    return __half2float(v);
}

__device__ __forceinline__ float Dsv4ToFloat(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename XT, typename WT>
__device__ __forceinline__ float Dsv4PairDot(const XT *x, const WT *w, int k) {
    return Dsv4ToFloat(x[k]) * Dsv4ToFloat(w[k]) +
           Dsv4ToFloat(x[k + 1]) * Dsv4ToFloat(w[k + 1]);
}

__device__ __forceinline__ float Dsv4PairDot(const __nv_bfloat16 *x, const __nv_bfloat16 *w, int k) {
    __nv_bfloat162 xv = *reinterpret_cast<const __nv_bfloat162 *>(x + k);
    __nv_bfloat162 wv = *reinterpret_cast<const __nv_bfloat162 *>(w + k);
    float2 xf = __bfloat1622float2(xv);
    float2 wf = __bfloat1622float2(wv);
    return xf.x * wf.x + xf.y * wf.y;
}

__device__ __forceinline__ float Dsv4PairDot(const half *x, const half *w, int k) {
    half2 xv = *reinterpret_cast<const half2 *>(x + k);
    half2 wv = *reinterpret_cast<const half2 *>(w + k);
    float2 xf = __half22float2(xv);
    float2 wf = __half22float2(wv);
    return xf.x * wf.x + xf.y * wf.y;
}

__device__ __forceinline__ float Dsv4PairDot(const float *x, const float *w, int k) {
    float2 xv = *reinterpret_cast<const float2 *>(x + k);
    float2 wv = *reinterpret_cast<const float2 *>(w + k);
    return xv.x * wv.x + xv.y * wv.y;
}

template <typename T>
__device__ __forceinline__ T Dsv4FromFloat(float v);

template <>
__device__ __forceinline__ float Dsv4FromFloat<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ half Dsv4FromFloat<half>(float v) {
    return __float2half_rn(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 Dsv4FromFloat<__nv_bfloat16>(float v) {
    return __float2bfloat16_rn(v);
}

template <typename InT, typename WT>
__global__ void DeepSeekV4WoAKernel(const InT *o, const WT *w, __nv_bfloat16 *output,
                                    int bsz, int seqlen, int heads, int headDim,
                                    int groups, int oRank) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bsz * seqlen * groups * oRank;
    if (idx >= total) {
        return;
    }

    int r = idx % oRank;
    int tmp = idx / oRank;
    int g = tmp % groups;
    tmp /= groups;
    int s = tmp % seqlen;
    int b = tmp / seqlen;

    int headsPerGroup = heads / groups;
    int groupDim = headsPerGroup * headDim;
    const WT *wrow = w + ((uint64_t)g * oRank + r) * groupDim;
    const InT *src = o + (((uint64_t)b * seqlen + s) * heads + g * headsPerGroup) * headDim;

    double v = 0.0;
    for (int d = 0; d < groupDim; d++) {
        v += (double)Dsv4ToFloat(src[d]) * Dsv4ToFloat(wrow[d]);
    }
    output[idx] = __float2bfloat16_rn((float)v);
}

template <typename InT, typename WT>
__global__ void DeepSeekV4WoABlockReduceKernel(const InT *o, const WT *w, __nv_bfloat16 *output,
                                               int bsz, int seqlen, int heads, int headDim,
                                               int groups, int oRank) {
    extern __shared__ float partial[];
    int idx = blockIdx.x;
    int total = bsz * seqlen * groups * oRank;
    if (idx >= total) {
        return;
    }

    int r = idx % oRank;
    int tmp = idx / oRank;
    int g = tmp % groups;
    tmp /= groups;
    int s = tmp % seqlen;
    int b = tmp / seqlen;

    int headsPerGroup = heads / groups;
    int groupDim = headsPerGroup * headDim;
    const WT *wrow = w + ((uint64_t)g * oRank + r) * groupDim;
    const InT *src = o + (((uint64_t)b * seqlen + s) * heads + g * headsPerGroup) * headDim;

    float sum = 0.0f;
    for (int d = threadIdx.x; d < groupDim; d += blockDim.x) {
        sum += Dsv4ToFloat(src[d]) * Dsv4ToFloat(wrow[d]);
    }
    partial[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[idx] = __float2bfloat16_rn(partial[0]);
    }
}

template <typename InT, typename WT>
__global__ void DeepSeekV4WoAPairBlockReduceKernel(const InT *o, const WT *w, __nv_bfloat16 *output,
                                                   int bsz, int seqlen, int heads, int headDim,
                                                   int groups, int oRank) {
    extern __shared__ float partial[];
    float *partial0 = partial;
    float *partial1 = partial + blockDim.x;
    int pairsPerGroup = oRank / 2;
    int idx = blockIdx.x;
    int total = bsz * seqlen * groups * pairsPerGroup;
    if (idx >= total) {
        return;
    }

    int pair = idx % pairsPerGroup;
    int r0 = pair * 2;
    int r1 = r0 + 1;
    int tmp = idx / pairsPerGroup;
    int g = tmp % groups;
    tmp /= groups;
    int s = tmp % seqlen;
    int b = tmp / seqlen;

    int headsPerGroup = heads / groups;
    int groupDim = headsPerGroup * headDim;
    const WT *wrow0 = w + ((uint64_t)g * oRank + r0) * groupDim;
    const WT *wrow1 = wrow0 + groupDim;
    const InT *src = o + (((uint64_t)b * seqlen + s) * heads + g * headsPerGroup) * headDim;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int d = threadIdx.x; d < groupDim; d += blockDim.x) {
        float x = Dsv4ToFloat(src[d]);
        sum0 += x * Dsv4ToFloat(wrow0[d]);
        sum1 += x * Dsv4ToFloat(wrow1[d]);
    }
    partial0[threadIdx.x] = sum0;
    partial1[threadIdx.x] = sum1;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial0[threadIdx.x] += partial0[threadIdx.x + stride];
            partial1[threadIdx.x] += partial1[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        uint64_t outBase = (((uint64_t)b * seqlen + s) * groups + g) * oRank;
        output[outBase + r0] = __float2bfloat16_rn(partial0[0]);
        output[outBase + r1] = __float2bfloat16_rn(partial1[0]);
    }
}

// The checkpoint stores wo_a as block-scaled FP8 E4M3, but the legacy path
// expands it to FP16 while loading.  Decode is bandwidth-bound on this 64 MiB
// matrix.  Read the original 8-bit payload and reproduce the legacy FP16
// dequantization in registers so the dot-product order and rounded weight
// values stay unchanged while global weight traffic is halved.
__device__ __forceinline__ float DeepSeekV4Fp8E4M3ScaledHalfToFloat(uint8_t value,
                                                                    float scale) {
    uint16_t pseudoHalfBits = ((value & 0x80u) << 8) | ((value & 0x7fu) << 7);
    half pseudoHalf = __ushort_as_half(pseudoHalfBits);
    half rounded = __float2half_rn(__half2float(pseudoHalf) * scale * 0x1p8f);
    return __half2float(rounded);
}

template <typename InT>
__global__ void DeepSeekV4WoAFp8PairBlockReduceKernel(
        const InT *o, const uint8_t *w, const float *scales, __nv_bfloat16 *output,
        int bsz, int seqlen, int heads, int headDim, int groups, int oRank,
        int blockK, int blockM, int scaleCols) {
    extern __shared__ float partial[];
    float *partial0 = partial;
    float *partial1 = partial + blockDim.x;
    int pairsPerGroup = oRank / 2;
    int idx = blockIdx.x;
    int total = bsz * seqlen * groups * pairsPerGroup;
    if (idx >= total) {
        return;
    }

    int pair = idx % pairsPerGroup;
    int r0 = pair * 2;
    int r1 = r0 + 1;
    int tmp = idx / pairsPerGroup;
    int g = tmp % groups;
    tmp /= groups;
    int s = tmp % seqlen;
    int b = tmp / seqlen;

    int headsPerGroup = heads / groups;
    int groupDim = headsPerGroup * headDim;
    int weightRow0 = g * oRank + r0;
    const uint8_t *wrow0 = w + (uint64_t)weightRow0 * groupDim;
    const uint8_t *wrow1 = wrow0 + groupDim;
    const InT *src = o + (((uint64_t)b * seqlen + s) * heads + g * headsPerGroup) * headDim;
    const float *rowScales = scales + (weightRow0 / blockK) * scaleCols;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int d = threadIdx.x; d < groupDim; d += blockDim.x) {
        float x = Dsv4ToFloat(src[d]);
        float scale = rowScales[d / blockM];
        sum0 += x * DeepSeekV4Fp8E4M3ScaledHalfToFloat(wrow0[d], scale);
        sum1 += x * DeepSeekV4Fp8E4M3ScaledHalfToFloat(wrow1[d], scale);
    }
    partial0[threadIdx.x] = sum0;
    partial1[threadIdx.x] = sum1;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial0[threadIdx.x] += partial0[threadIdx.x + stride];
            partial1[threadIdx.x] += partial1[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        uint64_t outBase = (((uint64_t)b * seqlen + s) * groups + g) * oRank;
        output[outBase + r0] = __float2bfloat16_rn(partial0[0]);
        output[outBase + r1] = __float2bfloat16_rn(partial1[0]);
    }
}

template <typename InT, typename WT>
__global__ void DeepSeekV4WoAFloatAccKernel(const InT *o, const WT *w, __nv_bfloat16 *output,
                                            int bsz, int seqlen, int heads, int headDim,
                                            int groups, int oRank) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bsz * seqlen * groups * oRank;
    if (idx >= total) {
        return;
    }

    int r = idx % oRank;
    int tmp = idx / oRank;
    int g = tmp % groups;
    tmp /= groups;
    int s = tmp % seqlen;
    int b = tmp / seqlen;

    int headsPerGroup = heads / groups;
    int groupDim = headsPerGroup * headDim;
    const WT *wrow = w + ((uint64_t)g * oRank + r) * groupDim;
    const InT *src = o + (((uint64_t)b * seqlen + s) * heads + g * headsPerGroup) * headDim;

    float v = 0.0f;
    for (int d = 0; d < groupDim; d++) {
        v += Dsv4ToFloat(src[d]) * Dsv4ToFloat(wrow[d]);
    }
    output[idx] = __float2bfloat16_rn(v);
}

template <typename InT, typename WT>
__global__ void DeepSeekV4WoAKahanAccKernel(const InT *o, const WT *w, __nv_bfloat16 *output,
                                            int bsz, int seqlen, int heads, int headDim,
                                            int groups, int oRank) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bsz * seqlen * groups * oRank;
    if (idx >= total) {
        return;
    }

    int r = idx % oRank;
    int tmp = idx / oRank;
    int g = tmp % groups;
    tmp /= groups;
    int s = tmp % seqlen;
    int b = tmp / seqlen;

    int headsPerGroup = heads / groups;
    int groupDim = headsPerGroup * headDim;
    const WT *wrow = w + ((uint64_t)g * oRank + r) * groupDim;
    const InT *src = o + (((uint64_t)b * seqlen + s) * heads + g * headsPerGroup) * headDim;

    float sum = 0.0f;
    float c = 0.0f;
    for (int d = 0; d < groupDim; d++) {
        float prod = Dsv4ToFloat(src[d]) * Dsv4ToFloat(wrow[d]);
        float y = prod - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    output[idx] = __float2bfloat16_rn(sum);
}

template <typename InT, typename WT>
__global__ void DeepSeekV4WoAPairKernel(const InT *o, const WT *w, __nv_bfloat16 *output,
                                        int bsz, int seqlen, int heads, int headDim,
                                        int groups, int oRank) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pairsPerGroup = oRank / 2;
    int total = bsz * seqlen * groups * pairsPerGroup;
    if (idx >= total) {
        return;
    }

    int pair = idx % pairsPerGroup;
    int r0 = pair * 2;
    int r1 = r0 + 1;
    int tmp = idx / pairsPerGroup;
    int g = tmp % groups;
    tmp /= groups;
    int s = tmp % seqlen;
    int b = tmp / seqlen;

    int headsPerGroup = heads / groups;
    int groupDim = headsPerGroup * headDim;
    const WT *wrow0 = w + ((uint64_t)g * oRank + r0) * groupDim;
    const WT *wrow1 = wrow0 + groupDim;

    double v0 = 0.0;
    double v1 = 0.0;
    int d = 0;
    for (int hh = 0; hh < headsPerGroup; hh++) {
        const InT *src = o + (((uint64_t)b * seqlen + s) * heads + g * headsPerGroup + hh) * headDim;
        for (int localD = 0; localD < headDim; localD++, d++) {
            double x = (double)Dsv4ToFloat(src[localD]);
            v0 += x * Dsv4ToFloat(wrow0[d]);
            v1 += x * Dsv4ToFloat(wrow1[d]);
        }
    }
    uint64_t outBase = (((uint64_t)b * seqlen + s) * groups + g) * oRank;
    output[outBase + r0] = __float2bfloat16_rn((float)v0);
    output[outBase + r1] = __float2bfloat16_rn((float)v1);
}

__device__ __forceinline__ float DeepSeekV4InvFreq(int idx, int ropeDim, float base,
                                                   int originalSeqLen, float factor,
                                                   int betaFast, int betaSlow) {
    float inv = 1.0f / powf(base, (float)(idx * 2) / (float)ropeDim);
    if (originalSeqLen > 0) {
        float lowF = ropeDim * logf((float)originalSeqLen / (betaFast * 2.0f * 3.14159265358979323846f)) /
                     (2.0f * logf(base));
        float highF = ropeDim * logf((float)originalSeqLen / (betaSlow * 2.0f * 3.14159265358979323846f)) /
                      (2.0f * logf(base));
        int low = max((int)floorf(lowF), 0);
        int high = min((int)ceilf(highF), ropeDim - 1);
        if (low == high) {
            high++;
        }
        float ramp = fminf(1.0f, fmaxf(0.0f, ((float)idx - low) / (float)(high - low)));
        float smooth = 1.0f - ramp;
        inv = inv / factor * (1.0f - smooth) + inv * smooth;
    }
    return inv;
}

template <typename T>
__global__ void DeepSeekV4ScaleQRotaryKernel(T *q, int rows, int seqlen, int heads, int dim,
                                             int ropeDim, float ropeBase, int startPos,
                                             int originalSeqLen, float ropeFactor,
                                             int betaFast, int betaSlow, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    T *ptr = q + (uint64_t)row * dim;
    double ss = 0.0;
    for (int d = 0; d < dim; d++) {
        double v = (double)Dsv4ToFloat(ptr[d]);
        ss += v * v;
    }
    float scale = 1.0f / sqrtf((float)(ss / dim) + eps);
    for (int d = 0; d < dim; d++) {
        ptr[d] = Dsv4FromFloat<T>(Dsv4ToFloat(ptr[d]) * scale);
    }

    int s = (row / heads) % seqlen;
    int pos = startPos + s;
    int off = dim - ropeDim;
    for (int i = 0; i < ropeDim; i += 2) {
        float inv = DeepSeekV4InvFreq(i / 2, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
        float ang = pos * inv;
        float c = cosf(ang);
        float sn = sinf(ang);
        float a = Dsv4ToFloat(ptr[off + i]);
        float b = Dsv4ToFloat(ptr[off + i + 1]);
        ptr[off + i] = Dsv4FromFloat<T>(a * c - b * sn);
        ptr[off + i + 1] = Dsv4FromFloat<T>(a * sn + b * c);
    }
}

template <typename T>
__global__ void DeepSeekV4ScaleQRotaryBlockKernel(T *q, int rows, int seqlen, int heads, int dim,
                                                  int ropeDim, float ropeBase, int startPos,
                                                  int originalSeqLen, float ropeFactor,
                                                  int betaFast, int betaSlow, float eps,
                                                  const int32_t *decodeMeta) {
    extern __shared__ float partial[];
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    T *ptr = q + (uint64_t)row * dim;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = Dsv4ToFloat(ptr[d]);
        ss += v * v;
    }
    partial[threadIdx.x] = ss;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[0] = rsqrtf(partial[0] / dim + eps);
    }
    __syncthreads();
    float scale = partial[0];
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        ptr[d] = Dsv4FromFloat<T>(Dsv4ToFloat(ptr[d]) * scale);
    }
    __syncthreads();

    int s = (row / heads) % seqlen;
    int dynamicStartPos = decodeMeta == nullptr ? startPos : decodeMeta[0];
    int pos = dynamicStartPos + s;
    int off = dim - ropeDim;
    for (int i = threadIdx.x * 2; i < ropeDim; i += blockDim.x * 2) {
        float inv = DeepSeekV4InvFreq(i / 2, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
        float ang = pos * inv;
        float c = cosf(ang);
        float sn = sinf(ang);
        float a = Dsv4ToFloat(ptr[off + i]);
        float b = Dsv4ToFloat(ptr[off + i + 1]);
        ptr[off + i] = Dsv4FromFloat<T>(a * c - b * sn);
        ptr[off + i + 1] = Dsv4FromFloat<T>(a * sn + b * c);
    }
}

template <typename T>
__global__ void DeepSeekV4RotaryQuantKernel(T *x, int rows, int seqlen, int heads, int dim,
                                            int ropeDim, float ropeBase, int startPos,
                                            int originalSeqLen, float ropeFactor,
                                            int betaFast, int betaSlow, int quantDim,
                                            int blockSize, int posStep) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    T *ptr = x + (uint64_t)row * dim;
    int s = (row / heads) % seqlen;
    int pos = startPos + s * posStep;
    int off = dim - ropeDim;
    for (int i = 0; i < ropeDim; i += 2) {
        float inv = DeepSeekV4InvFreq(i / 2, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
        float ang = pos * inv;
        float c = cosf(ang);
        float sn = sinf(ang);
        float a = Dsv4ToFloat(ptr[off + i]);
        float b = Dsv4ToFloat(ptr[off + i + 1]);
        ptr[off + i] = Dsv4FromFloat<T>(a * c - b * sn);
        ptr[off + i + 1] = Dsv4FromFloat<T>(a * sn + b * c);
    }

    for (int start = 0; start < quantDim; start += blockSize) {
        int end = min(start + blockSize, quantDim);
        float amax = 1e-4f;
        for (int d = start; d < end; d++) {
            amax = fmaxf(amax, fabsf(Dsv4ToFloat(ptr[d])));
        }
        float qScale = powf(2.0f, ceilf(log2f(amax / 448.0f)));
        for (int d = start; d < end; d++) {
            float qv = fminf(448.0f, fmaxf(-448.0f, Dsv4ToFloat(ptr[d]) / qScale));
            float rounded = __bfloat162float(__float2bfloat16_rn(qv)) * qScale;
            ptr[d] = Dsv4FromFloat<T>(rounded);
        }
    }
}

template <typename T>
__global__ void DeepSeekV4RotaryQuantBlockKernel(T *x, int rows, int seqlen, int heads, int dim,
                                                 int ropeDim, float ropeBase, int startPos,
                                                 int originalSeqLen, float ropeFactor,
                                                 int betaFast, int betaSlow, int quantDim,
                                                 int blockSize, int posStep,
                                                 const int32_t *decodeMeta) {
    extern __shared__ float partial[];
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    T *ptr = x + (uint64_t)row * dim;
    int s = (row / heads) % seqlen;
    int dynamicStartPos = decodeMeta == nullptr ? startPos : decodeMeta[0];
    int pos = dynamicStartPos + s * posStep;
    int off = dim - ropeDim;
    for (int i = threadIdx.x * 2; i < ropeDim; i += blockDim.x * 2) {
        float inv = DeepSeekV4InvFreq(i / 2, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
        float ang = pos * inv;
        float c = cosf(ang);
        float sn = sinf(ang);
        float a = Dsv4ToFloat(ptr[off + i]);
        float b = Dsv4ToFloat(ptr[off + i + 1]);
        ptr[off + i] = Dsv4FromFloat<T>(a * c - b * sn);
        ptr[off + i + 1] = Dsv4FromFloat<T>(a * sn + b * c);
    }
    __syncthreads();

    for (int start = 0; start < quantDim; start += blockSize) {
        int end = min(start + blockSize, quantDim);
        float amax = 1e-4f;
        for (int d = start + threadIdx.x; d < end; d += blockDim.x) {
            amax = fmaxf(amax, fabsf(Dsv4ToFloat(ptr[d])));
        }
        partial[threadIdx.x] = amax;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            partial[0] = powf(2.0f, ceilf(log2f(partial[0] / 448.0f)));
        }
        __syncthreads();
        float qScale = partial[0];
        for (int d = start + threadIdx.x; d < end; d += blockDim.x) {
            float qv = fminf(448.0f, fmaxf(-448.0f, Dsv4ToFloat(ptr[d]) / qScale));
            float rounded = __bfloat162float(__float2bfloat16_rn(qv)) * qScale;
            ptr[d] = Dsv4FromFloat<T>(rounded);
        }
        __syncthreads();
    }
}

__device__ __forceinline__ float DeepSeekV4WarpSum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float DeepSeekV4Warp4Max(float value) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 1));
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, 2));
    return value;
}

// Horizontally fuse the single-token Q and KV post-projection work.  The warp
// layout follows vLLM's DeepSeek-V4 kernel, while the arithmetic deliberately
// preserves FastLLM's two observable BF16 boundaries:
//   Q RMSNorm -> BF16 -> RoPE -> BF16
//   KV weighted RMSNorm -> BF16 -> (RoPE / quant round-trip) -> BF16
// One warp owns one 512-wide Q head; the final warp owns KV and also writes the
// FLOAT32 sliding-window cache.  RoPE sin/cos are computed once per CTA rather
// than once per head.
__global__ void DeepSeekV4FusedQKVRopeCache512Kernel(
        __nv_bfloat16 *q, __nv_bfloat16 *kv, const float *kvNormWeight,
        float *windowKV, const int32_t *decodeMeta, int windowSize,
        int qHeads,
        float ropeBase, int originalSeqLen, float ropeFactor,
        int betaFast, int betaSlow, float eps) {
    constexpr int kHeadDim = 512;
    constexpr int kRopeDim = 64;
    constexpr int kNopeDim = kHeadDim - kRopeDim;
    constexpr int kElemsPerLane = kHeadDim / 32;
    constexpr float kQuantMax = 448.0f;

    __shared__ float ropeCos[kRopeDim / 2];
    __shared__ float ropeSin[kRopeDim / 2];
    if (threadIdx.x < kRopeDim / 2) {
        int pair = threadIdx.x;
        float inv = DeepSeekV4InvFreq(pair, kRopeDim, ropeBase,
                                      originalSeqLen, ropeFactor,
                                      betaFast, betaSlow);
        float angle = (float)decodeMeta[0] * inv;
        float c = cosf(angle);
        float s = sinf(angle);
        ropeCos[pair] = c;
        ropeSin[pair] = s;
    }
    __syncthreads();

    int warpsPerBlock = blockDim.x / 32;
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int slot = blockIdx.x * warpsPerBlock + warp;
    if (slot > qHeads) {
        return;
    }
    bool isKV = slot == qHeads;
    int dimBase = lane * kElemsPerLane;
    __nv_bfloat16 *row = isKV ? kv : q + (uint64_t)slot * kHeadDim;

    uint4 raw0 = *reinterpret_cast<const uint4 *>(row + dimBase);
    uint4 raw1 = *reinterpret_cast<const uint4 *>(row + dimBase + 8);
    const __nv_bfloat162 *pair0 = reinterpret_cast<const __nv_bfloat162 *>(&raw0);
    const __nv_bfloat162 *pair1 = reinterpret_cast<const __nv_bfloat162 *>(&raw1);
    float values[kElemsPerLane];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        float2 value = __bfloat1622float2(pair0[i]);
        values[i * 2] = value.x;
        values[i * 2 + 1] = value.y;
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        float2 value = __bfloat1622float2(pair1[i]);
        values[8 + i * 2] = value.x;
        values[8 + i * 2 + 1] = value.y;
    }

    float sumSquares = 0.0f;
#pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
        sumSquares += values[i] * values[i];
    }
    sumSquares = DeepSeekV4WarpSum(sumSquares);
    float normScale = __shfl_sync(0xffffffffu,
                                  rsqrtf(sumSquares / (float)kHeadDim + eps), 0);
#pragma unroll
    for (int i = 0; i < kElemsPerLane; i++) {
        float value = values[i] * normScale;
        if (isKV) {
            value *= __ldg(kvNormWeight + dimBase + i);
        }
        // Match the standalone Q/KV RMSNorm output before applying RoPE.
        values[i] = __bfloat162float(__float2bfloat16_rn(value));
    }

    if (dimBase >= kNopeDim) {
        int ropePairBase = (dimBase - kNopeDim) / 2;
#pragma unroll
        for (int pair = 0; pair < kElemsPerLane / 2; pair++) {
            float even = values[pair * 2];
            float odd = values[pair * 2 + 1];
            float c = ropeCos[ropePairBase + pair];
            float s = ropeSin[ropePairBase + pair];
            values[pair * 2] = __bfloat162float(
                __float2bfloat16_rn(even * c - odd * s));
            values[pair * 2 + 1] = __bfloat162float(
                __float2bfloat16_rn(even * s + odd * c));
        }
    } else if (isKV) {
        // Four adjacent lanes jointly own one 64-element quant block.
        float amax = 1e-4f;
#pragma unroll
        for (int i = 0; i < kElemsPerLane; i++) {
            amax = fmaxf(amax, fabsf(values[i]));
        }
        amax = DeepSeekV4Warp4Max(amax);
        float quantScale = exp2f(ceilf(log2f(amax / kQuantMax)));
#pragma unroll
        for (int i = 0; i < kElemsPerLane; i++) {
            float quantValue = fminf(kQuantMax,
                                     fmaxf(-kQuantMax, values[i] / quantScale));
            float rounded = __bfloat162float(__float2bfloat16_rn(quantValue)) *
                            quantScale;
            values[i] = __bfloat162float(__float2bfloat16_rn(rounded));
        }
    }

    uint4 out0, out1;
    __nv_bfloat162 *outPair0 = reinterpret_cast<__nv_bfloat162 *>(&out0);
    __nv_bfloat162 *outPair1 = reinterpret_cast<__nv_bfloat162 *>(&out1);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        outPair0[i].x = __float2bfloat16_rn(values[i * 2]);
        outPair0[i].y = __float2bfloat16_rn(values[i * 2 + 1]);
        outPair1[i].x = __float2bfloat16_rn(values[8 + i * 2]);
        outPair1[i].y = __float2bfloat16_rn(values[8 + i * 2 + 1]);
    }
    *reinterpret_cast<uint4 *>(row + dimBase) = out0;
    *reinterpret_cast<uint4 *>(row + dimBase + 8) = out1;

    if (isKV) {
        int cacheSlot = decodeMeta[0] % windowSize;
        float *cache = windowKV + (uint64_t)cacheSlot * kHeadDim + dimBase;
#pragma unroll
        for (int i = 0; i < kElemsPerLane; i += 4) {
            *reinterpret_cast<float4 *>(cache + i) =
                make_float4(values[i], values[i + 1], values[i + 2], values[i + 3]);
        }
    }
}

template <typename T>
__global__ void DeepSeekV4UpdateWindowKVCacheKernel(const T *kv, float *windowKV,
                                                    int bsz, int seqlen, int headDim, int startPos,
                                                    int windowSize, const int32_t *decodeMeta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bsz * seqlen * headDim;
    if (idx >= total) {
        return;
    }
    int d = idx % headDim;
    int tmp = idx / headDim;
    int s = tmp % seqlen;
    int b = tmp / seqlen;
    int dynamicStartPos = decodeMeta == nullptr ? startPos : decodeMeta[0];
    windowKV[((uint64_t)b * windowSize + ((dynamicStartPos + s) % windowSize)) * headDim + d] =
        Dsv4ToFloat(kv[((uint64_t)b * seqlen + s) * headDim + d]);
}

template <typename T>
__global__ void DeepSeekV4StoreWindowKVCacheKernel(const T *kv, float *windowKV,
                                                   int bsz, int seqlen, int headDim,
                                                   int startPos, int windowSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bsz * windowSize * headDim;
    if (idx >= total) {
        return;
    }
    int d = idx % headDim;
    int slot = (idx / headDim) % windowSize;
    int b = idx / (windowSize * headDim);
    int srcToken = -1;
    if (startPos == 0) {
        if (seqlen <= windowSize) {
            srcToken = slot < seqlen ? slot : -1;
        } else {
            int cutoff = seqlen % windowSize;
            srcToken = seqlen - windowSize + ((slot - cutoff + windowSize) % windowSize);
        }
    } else {
        srcToken = slot == (startPos % windowSize) ? 0 : -1;
    }
    windowKV[((uint64_t)b * windowSize + slot) * headDim + d] =
        srcToken >= 0 ? Dsv4ToFloat(kv[((uint64_t)b * seqlen + srcToken) * headDim + d]) : 0.0f;
}

__global__ void DeepSeekV4BuildWindowKVPrefixKernel(const float *windowKV, float *output,
                                                    int bsz, int prefixLen, int headDim,
                                                    int startPos, int windowSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = bsz * prefixLen * headDim;
    if (idx >= total) {
        return;
    }
    int d = idx % headDim;
    int tmp = idx / headDim;
    int s = tmp % prefixLen;
    int b = tmp / prefixLen;
    int firstPos = startPos - prefixLen;
    int srcSlot = (firstPos + s) % windowSize;
    output[((uint64_t)b * prefixLen + s) * headDim + d] =
        windowKV[((uint64_t)b * windowSize + srcSlot) * headDim + d];
}

template <typename T>
__global__ void DeepSeekV4BuildCompressedKVKernel(const T *kv, const T *score, const float *ape,
                                                  float *compressed, int bsz, int rawTokenBase,
                                                  int rawLen, int blockStart, int blockCount,
                                                  int compressRatio, int headDim, int wideDim,
                                                  bool overlap) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)bsz * blockCount * headDim;
    if (idx >= total) {
        return;
    }
    int d = (int)(idx % headDim);
    uint64_t tmp = idx / headDim;
    int localBlock = (int)(tmp % blockCount);
    int b = (int)(tmp / blockCount);
    int block = blockStart + localBlock;
    float mx = -1.0e30f;
    if (overlap) {
        if (block > 0) {
            for (int r = 0; r < compressRatio; r++) {
                int tok = (block - 1) * compressRatio + r;
                int localToken = tok - rawTokenBase;
                uint64_t off = ((uint64_t)b * rawLen + localToken) * wideDim + d;
                mx = fmaxf(mx, Dsv4ToFloat(score[off]) + ape[(uint64_t)r * wideDim + d]);
            }
        }
        for (int r = 0; r < compressRatio; r++) {
            int tok = block * compressRatio + r;
            int localToken = tok - rawTokenBase;
            uint64_t off = ((uint64_t)b * rawLen + localToken) * wideDim + headDim + d;
            mx = fmaxf(mx, Dsv4ToFloat(score[off]) + ape[(uint64_t)r * wideDim + headDim + d]);
        }
    } else {
        for (int r = 0; r < compressRatio; r++) {
            int tok = block * compressRatio + r;
            int localToken = tok - rawTokenBase;
            uint64_t off = ((uint64_t)b * rawLen + localToken) * wideDim + d;
            mx = fmaxf(mx, Dsv4ToFloat(score[off]) + ape[(uint64_t)r * wideDim + d]);
        }
    }

    float sum = 0.0f, value = 0.0f;
    if (overlap) {
        if (block > 0) {
            for (int r = 0; r < compressRatio; r++) {
                int tok = (block - 1) * compressRatio + r;
                int localToken = tok - rawTokenBase;
                uint64_t off = ((uint64_t)b * rawLen + localToken) * wideDim + d;
                float e = expf(Dsv4ToFloat(score[off]) + ape[(uint64_t)r * wideDim + d] - mx);
                sum += e;
                value += e * Dsv4ToFloat(kv[off]);
            }
        }
        for (int r = 0; r < compressRatio; r++) {
            int tok = block * compressRatio + r;
            int localToken = tok - rawTokenBase;
            uint64_t off = ((uint64_t)b * rawLen + localToken) * wideDim + headDim + d;
            float e = expf(Dsv4ToFloat(score[off]) + ape[(uint64_t)r * wideDim + headDim + d] - mx);
            sum += e;
            value += e * Dsv4ToFloat(kv[off]);
        }
    } else {
        for (int r = 0; r < compressRatio; r++) {
            int tok = block * compressRatio + r;
            int localToken = tok - rawTokenBase;
            uint64_t off = ((uint64_t)b * rawLen + localToken) * wideDim + d;
            float e = expf(Dsv4ToFloat(score[off]) + ape[(uint64_t)r * wideDim + d] - mx);
            sum += e;
            value += e * Dsv4ToFloat(kv[off]);
        }
    }
    compressed[((uint64_t)b * blockCount + localBlock) * headDim + d] = value / fmaxf(sum, 1e-30f);
}

template <typename T>
__global__ void DeepSeekV4InitGraphRawRingKernel(const T *raw, T *ring,
                                                 int bsz, int rawLen, int wideDim,
                                                 int rawTokenBase, int ringCapacity) {
    uint64_t total = (uint64_t)bsz * rawLen * wideDim;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int d = idx % wideDim;
    uint64_t tmp = idx / wideDim;
    int s = tmp % rawLen;
    int b = tmp / rawLen;
    int slot = (rawTokenBase + s) % ringCapacity;
    ring[((uint64_t)b * ringCapacity + slot) * wideDim + d] = raw[idx];
}

template <typename T>
__global__ void DeepSeekV4StoreGraphCompressorRawKernel(const T *kv, const T *score,
                                                        T *kvRing, T *scoreRing,
                                                        const int32_t *decodeMeta,
                                                        int bsz, int wideDim,
                                                        int ringCapacity) {
    uint64_t total = (uint64_t)bsz * wideDim;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    int d = idx % wideDim;
    int b = idx / wideDim;
    int startPos = decodeMeta[0];
    int slot = startPos % ringCapacity;
    uint64_t dst = ((uint64_t)b * ringCapacity + slot) * wideDim + d;
    kvRing[dst] = kv[(uint64_t)b * wideDim + d];
    scoreRing[dst] = score[(uint64_t)b * wideDim + d];
}

template <typename T, typename WT>
__global__ void DeepSeekV4UpdateCompressedKVGraphKernel(
                                                        const T *kvRing, const T *scoreRing,
                                                        const float *ape, const WT *normWeight,
                                                        const int32_t *decodeMeta,
                                                        __nv_bfloat16 *compressedKV,
                                                        int bsz, int ringCapacity,
                                                        int compressedCapacity,
                                                        int compressRatio, int headDim,
                                                        int wideDim, int ropeDim,
                                                        float ropeBase, int originalSeqLen,
                                                        float ropeFactor, int betaFast,
                                                        int betaSlow) {
    extern __shared__ float shared[];
    float *values = shared;
    float *red = values + headDim;
    int b = blockIdx.x;
    if (b >= bsz) {
        return;
    }
    int startPos = decodeMeta[0];
    int totalLen = startPos + 1;
    if (totalLen <= 0 || totalLen % compressRatio != 0) {
        return;
    }
    int block = totalLen / compressRatio - 1;
    if (block < 0 || block >= compressedCapacity) {
        return;
    }
    bool overlap = compressRatio == 4;

    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        float mx = -INFINITY;
        if (overlap && block > 0) {
            for (int r = 0; r < compressRatio; r++) {
                int token = (block - 1) * compressRatio + r;
                int slot = token % ringCapacity;
                uint64_t off = ((uint64_t)b * ringCapacity + slot) * wideDim + d;
                mx = fmaxf(mx, Dsv4ToFloat(scoreRing[off]) + ape[(uint64_t)r * wideDim + d]);
            }
        }
        int currentBase = block * compressRatio;
        int currentDimOffset = overlap ? headDim : 0;
        for (int r = 0; r < compressRatio; r++) {
            int token = currentBase + r;
            int slot = token % ringCapacity;
            uint64_t off = ((uint64_t)b * ringCapacity + slot) * wideDim + currentDimOffset + d;
            mx = fmaxf(mx, Dsv4ToFloat(scoreRing[off]) +
                            ape[(uint64_t)r * wideDim + currentDimOffset + d]);
        }

        float sum = 0.0f, value = 0.0f;
        if (overlap && block > 0) {
            for (int r = 0; r < compressRatio; r++) {
                int token = (block - 1) * compressRatio + r;
                int slot = token % ringCapacity;
                uint64_t off = ((uint64_t)b * ringCapacity + slot) * wideDim + d;
                float e = expf(Dsv4ToFloat(scoreRing[off]) +
                               ape[(uint64_t)r * wideDim + d] - mx);
                sum += e;
                value += e * Dsv4ToFloat(kvRing[off]);
            }
        }
        for (int r = 0; r < compressRatio; r++) {
            int token = currentBase + r;
            int slot = token % ringCapacity;
            uint64_t off = ((uint64_t)b * ringCapacity + slot) * wideDim + currentDimOffset + d;
            float e = expf(Dsv4ToFloat(scoreRing[off]) +
                           ape[(uint64_t)r * wideDim + currentDimOffset + d] - mx);
            sum += e;
            value += e * Dsv4ToFloat(kvRing[off]);
        }
        // Match the existing pipeline's FP32 -> BF16 conversion before RMSNorm.
        values[d] = __bfloat162float(__float2bfloat16_rn(value / fmaxf(sum, 1e-30f)));
    }
    __syncthreads();

    float ss = 0.0f;
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        ss += values[d] * values[d];
    }
    red[threadIdx.x] = ss;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float normScale = rsqrtf(red[0] / headDim + 1.0e-6f);
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        values[d] = __bfloat162float(__float2bfloat16_rn(
            values[d] * normScale * Dsv4ToFloat(normWeight[d])));
    }
    __syncthreads();

    int ropeOffset = headDim - ropeDim;
    int rotaryPos = block * compressRatio;
    for (int i = threadIdx.x * 2; i < ropeDim; i += blockDim.x * 2) {
        float inv = DeepSeekV4InvFreq(i / 2, ropeDim, ropeBase, originalSeqLen,
                                      ropeFactor, betaFast, betaSlow);
        float angle = rotaryPos * inv;
        float c = cosf(angle), s = sinf(angle);
        float a = values[ropeOffset + i];
        float bb = values[ropeOffset + i + 1];
        values[ropeOffset + i] = __bfloat162float(__float2bfloat16_rn(a * c - bb * s));
        values[ropeOffset + i + 1] = __bfloat162float(__float2bfloat16_rn(a * s + bb * c));
    }
    __syncthreads();

    for (int groupStart = 0; groupStart < ropeOffset; groupStart += 64) {
        int groupEnd = min(groupStart + 64, ropeOffset);
        float amax = 1.0e-4f;
        for (int d = groupStart + threadIdx.x; d < groupEnd; d += blockDim.x) {
            amax = fmaxf(amax, fabsf(values[d]));
        }
        red[threadIdx.x] = amax;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        float qScale = powf(2.0f, ceilf(log2f(red[0] / 448.0f)));
        for (int d = groupStart + threadIdx.x; d < groupEnd; d += blockDim.x) {
            float qv = fminf(448.0f, fmaxf(-448.0f, values[d] / qScale));
            values[d] = __bfloat162float(__float2bfloat16_rn(qv)) * qScale;
        }
        __syncthreads();
    }

    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        compressedKV[((uint64_t)b * compressedCapacity + block) * headDim + d] =
            __float2bfloat16_rn(values[d]);
    }
}

__device__ __forceinline__ float DeepSeekV4Sigmoid(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

__device__ __forceinline__ float DeepSeekV4SinkhornIteration4x4(
        float value, float eps) {
    float rowSum = value;
    rowSum += __shfl_xor_sync(0xffffffffu, rowSum, 1, 4);
    rowSum += __shfl_xor_sync(0xffffffffu, rowSum, 2, 4);
    value /= rowSum + eps;

    float colSum = value;
    colSum += __shfl_xor_sync(0xffffffffu, colSum, 4, 16);
    colSum += __shfl_xor_sync(0xffffffffu, colSum, 8, 16);
    return value / (colSum + eps);
}

__device__ __forceinline__ float DeepSeekV4Softplus(float x) {
    if (x > 20.0f) {
        return x;
    }
    if (x < -20.0f) {
        return expf(x);
    }
    return log1pf(expf(x));
}

__global__ void DeepSeekV4RouteScoreTransformKernel(float *logits, int rows, int experts, int mode) {
    int row = blockIdx.x;
    float *rowData = logits + (uint64_t)row * experts;
    if (mode == 0) {
        float mx = -INFINITY;
        for (int e = threadIdx.x; e < experts; e += blockDim.x) {
            mx = fmaxf(mx, rowData[e]);
        }
        __shared__ float red[256];
        red[threadIdx.x] = mx;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        mx = red[0];
        double partial = 0.0;
        for (int e = threadIdx.x; e < experts; e += blockDim.x) {
            partial += (double)expf(rowData[e] - mx);
        }
        red[threadIdx.x] = (float)partial;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] += red[threadIdx.x + stride];
            }
            __syncthreads();
        }
        float sum = red[0];
        for (int e = threadIdx.x; e < experts; e += blockDim.x) {
            rowData[e] = expf(rowData[e] - mx) / sum;
        }
    } else {
        for (int e = threadIdx.x; e < experts; e += blockDim.x) {
            float raw = rowData[e];
            rowData[e] = mode == 1 ? DeepSeekV4Sigmoid(raw) : sqrtf(DeepSeekV4Softplus(raw));
        }
    }
}

template <typename RouteT>
__global__ void DeepSeekV4HashRouteScoreKernel(float *logits, const RouteT *tid2eid,
                                               const int *inputIds, int singleTokenId,
                                               int32_t *index,
                                               float *score, int tokens, int experts,
                                               int topk, int mode, float routeScale,
                                               int routeRows) {
    int row = blockIdx.x;
    if (row >= tokens) {
        return;
    }
    float *rowData = logits + (uint64_t)row * experts;
    int32_t *outIndex = index + (uint64_t)row * topk;
    float *outScore = score + (uint64_t)row * topk;
    int tokenId = inputIds != nullptr ? inputIds[row] : singleTokenId;
    tokenId = max(0, min(tokenId, routeRows - 1));
    const RouteT *routeRow = tid2eid + (uint64_t)tokenId * topk;

    __shared__ float red[256];
    if (mode == 0) {
        float mx = -INFINITY;
        for (int e = threadIdx.x; e < experts; e += blockDim.x) {
            mx = fmaxf(mx, rowData[e]);
        }
        red[threadIdx.x] = mx;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        mx = red[0];

        float partial = 0.0f;
        for (int e = threadIdx.x; e < experts; e += blockDim.x) {
            partial += expf(rowData[e] - mx);
        }
        red[threadIdx.x] = partial;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] += red[threadIdx.x + stride];
            }
            __syncthreads();
        }
        float denom = fmaxf(red[0], 1e-30f);
        for (int k = threadIdx.x; k < topk; k += blockDim.x) {
            int expert = (int)routeRow[k];
            expert = max(0, min(expert, experts - 1));
            outIndex[k] = expert;
            outScore[k] = expf(rowData[expert] - mx) / denom * routeScale;
        }
        return;
    }

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int k = 0; k < topk; k++) {
            int expert = (int)routeRow[k];
            expert = max(0, min(expert, experts - 1));
            float raw = rowData[expert];
            float v = mode == 1 ? DeepSeekV4Sigmoid(raw) : sqrtf(DeepSeekV4Softplus(raw));
            outIndex[k] = expert;
            outScore[k] = v;
            sum += v;
        }
        float invSum = 1.0f / fmaxf(sum, 1e-30f);
        for (int k = 0; k < topk; k++) {
            outScore[k] = outScore[k] * invSum * routeScale;
        }
    }
}

template <typename XT, typename WT>
__global__ void DeepSeekV4HcPreDotsKernel(const XT *x, const WT *w, float *dots,
                                          int tokens, int flatDim, int mixHc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = tokens * mixHc;
    if (idx >= total) {
        return;
    }
    int m = idx % mixHc;
    int t = idx / mixHc;
    const XT *xrow = x + (uint64_t)t * flatDim;
    const WT *wrow = w + (uint64_t)m * flatDim;
    double v = 0.0;
    for (int k = 0; k < flatDim; k++) {
        v += (double)Dsv4ToFloat(xrow[k]) * Dsv4ToFloat(wrow[k]);
    }
    dots[idx] = (float)v;
}

template <typename XT, typename WT>
__global__ void DeepSeekV4HcPreDotsBlockKernel(const XT *x, const WT *w, float *dots,
                                               int tokens, int flatDim, int mixHc, int dotsStride,
                                               int dotParts) {
    extern __shared__ float red[];
    int idx = blockIdx.x / dotParts;
    int part = blockIdx.x - idx * dotParts;
    if (idx >= tokens * dotsStride) {
        return;
    }
    int m = idx % dotsStride;
    int t = idx / dotsStride;
    int chunkStart = (int)(((uint64_t)flatDim * part) / dotParts);
    int chunkEnd = (int)(((uint64_t)flatDim * (part + 1)) / dotParts);
    const XT *xrow = x + (uint64_t)t * flatDim;
    float partial = 0.0f;
    if (m == mixHc) {
        for (int k = chunkStart + threadIdx.x; k < chunkEnd; k += blockDim.x) {
            float v = Dsv4ToFloat(xrow[k]);
            partial += v * v;
        }
    } else {
        const WT *wrow = w + (uint64_t)m * flatDim;
        int pairStart = chunkStart;
        if (pairStart & 1) {
            if (threadIdx.x == 0) {
                partial += Dsv4ToFloat(xrow[pairStart]) * Dsv4ToFloat(wrow[pairStart]);
            }
            pairStart++;
        }
        for (int k = pairStart + threadIdx.x * 2; k + 1 < chunkEnd; k += blockDim.x * 2) {
            partial += Dsv4PairDot(xrow, wrow, k);
        }
        if (((chunkEnd - pairStart) & 1) && threadIdx.x == 0) {
            int k = chunkEnd - 1;
            partial += Dsv4ToFloat(xrow[k]) * Dsv4ToFloat(wrow[k]);
        }
    }
    red[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        dots[((uint64_t)t * dotsStride + m) * dotParts + part] = red[0];
    }
}

template <typename XT>
__global__ void DeepSeekV4HcPreSqSumKernel(const XT *x, float *dots,
                                           int tokens, int flatDim, int dotsStride) {
    extern __shared__ float red[];
    int t = blockIdx.x;
    if (t >= tokens) {
        return;
    }
    const XT *xrow = x + (uint64_t)t * flatDim;
    float partial = 0.0f;
    for (int k = threadIdx.x; k < flatDim; k += blockDim.x) {
        float v = Dsv4ToFloat(xrow[k]);
        partial += v * v;
    }
    red[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        dots[(uint64_t)t * dotsStride + (dotsStride - 1)] = red[0];
    }
}

template <typename XT>
__global__ void DeepSeekV4HcPreFinishKernel(const XT *x, const float *dots, const float *scale,
                                            const float *base, XT *y, float *post, float *comb,
                                            int tokens, int dim, int hcMult, int sinkhornIters,
                                            float eps, float normEps, int dotsStride, int dotParts) {
    extern __shared__ float shared[];
    float *mixes = shared;
    float *pre = mixes + (2 + hcMult) * hcMult;
    float *combLocal = pre + hcMult;
    float *rowStats = combLocal + hcMult * hcMult;
    float *colStats = rowStats + hcMult;

    int t = blockIdx.x;
    int finishPart = blockIdx.y;
    int finishParts = gridDim.y;
    int flatDim = hcMult * dim;
    int mixHc = (2 + hcMult) * hcMult;
    const XT *xrow = x + (uint64_t)t * flatDim;

    if (threadIdx.x == 0) {
        float sqSum = 0.0f;
        uint64_t sqBase = ((uint64_t)t * dotsStride + mixHc) * dotParts;
        for (int part = 0; part < dotParts; part++) {
            sqSum += dots[sqBase + part];
        }
        rowStats[0] = rsqrtf(sqSum / flatDim + normEps);
    }
    __syncthreads();
    float rsqrt = rowStats[0];
    for (int m = threadIdx.x; m < mixHc; m += blockDim.x) {
        float dotSum = 0.0f;
        uint64_t dotBase = ((uint64_t)t * dotsStride + m) * dotParts;
        for (int part = 0; part < dotParts; part++) {
            dotSum += dots[dotBase + part];
        }
        mixes[m] = dotSum * rsqrt;
    }
    __syncthreads();

    for (int h = threadIdx.x; h < hcMult; h += blockDim.x) {
        pre[h] = DeepSeekV4Sigmoid(mixes[h] * scale[0] + base[h]) + eps;
        if (finishPart == 0) {
            post[(uint64_t)t * hcMult + h] =
                2.0f * DeepSeekV4Sigmoid(mixes[h + hcMult] * scale[1] + base[h + hcMult]);
        }
    }
    __syncthreads();

    const int combThreads = 32;
    int hcSq = hcMult * hcMult;
    if (finishPart == 0 && threadIdx.x < combThreads) {
        int lane = threadIdx.x;
        for (int idx = lane; idx < hcSq; idx += combThreads) {
            int mixIdx = idx + 2 * hcMult;
            combLocal[idx] = mixes[mixIdx] * scale[2] + base[mixIdx];
        }
        __syncwarp();

        for (int r = lane; r < hcMult; r += combThreads) {
            float rowMax = -INFINITY;
            for (int c = 0; c < hcMult; c++) {
                rowMax = fmaxf(rowMax, combLocal[r * hcMult + c]);
            }
            rowStats[r] = rowMax;
        }
        __syncwarp();
        for (int idx = lane; idx < hcSq; idx += combThreads) {
            int r = idx / hcMult;
            combLocal[idx] = expf(combLocal[idx] - rowStats[r]);
        }
        __syncwarp();
        for (int r = lane; r < hcMult; r += combThreads) {
            float rowSum = 0.0f;
            for (int c = 0; c < hcMult; c++) {
                rowSum += combLocal[r * hcMult + c];
            }
            rowStats[r] = rowSum;
        }
        __syncwarp();
        for (int idx = lane; idx < hcSq; idx += combThreads) {
            int r = idx / hcMult;
            combLocal[idx] = combLocal[idx] / rowStats[r] + eps;
        }
        __syncwarp();

        for (int c = lane; c < hcMult; c += combThreads) {
            float colSum = 0.0f;
            for (int r = 0; r < hcMult; r++) {
                colSum += combLocal[r * hcMult + c];
            }
            colStats[c] = colSum;
        }
        __syncwarp();
        for (int idx = lane; idx < hcSq; idx += combThreads) {
            int c = idx % hcMult;
            combLocal[idx] /= (colStats[c] + eps);
        }
        __syncwarp();

        for (int it = 1; it < sinkhornIters; it++) {
            for (int r = lane; r < hcMult; r += combThreads) {
                float rowSum = 0.0f;
                for (int c = 0; c < hcMult; c++) {
                    rowSum += combLocal[r * hcMult + c];
                }
                rowStats[r] = rowSum;
            }
            __syncwarp();
            for (int idx = lane; idx < hcSq; idx += combThreads) {
                int r = idx / hcMult;
                combLocal[idx] /= (rowStats[r] + eps);
            }
            __syncwarp();

            for (int c = lane; c < hcMult; c += combThreads) {
                float colSum = 0.0f;
                for (int r = 0; r < hcMult; r++) {
                    colSum += combLocal[r * hcMult + c];
                }
                colStats[c] = colSum;
            }
            __syncwarp();
            for (int idx = lane; idx < hcSq; idx += combThreads) {
                int c = idx % hcMult;
                combLocal[idx] /= (colStats[c] + eps);
            }
            __syncwarp();
        }
        for (int i = lane; i < hcSq; i += combThreads) {
            comb[(uint64_t)t * hcMult * hcMult + i] = combLocal[i];
        }
    } else {
        int yStart = (int)(((uint64_t)dim * finishPart) / finishParts);
        int yEnd = (int)(((uint64_t)dim * (finishPart + 1)) / finishParts);
        int yThreads = finishPart == 0 ? blockDim.x - combThreads : blockDim.x;
        int yThread = finishPart == 0 ? threadIdx.x - combThreads : threadIdx.x;
        XT *yrow = y + (uint64_t)t * dim;
        for (int d = yStart + yThread; d < yEnd; d += yThreads) {
            float v = 0.0f;
            for (int h = 0; h < hcMult; h++) {
                v += pre[h] * Dsv4ToFloat(xrow[(uint64_t)h * dim + d]);
            }
            yrow[d] = Dsv4FromFloat<XT>(v);
        }
    }
}

// Decode specialization for hc_mult=4 and hidden_size=4096.  The legacy
// HcPre finish grid uses 16 CTAs and consequently repeats the scalar mix and
// 20 Sinkhorn iterations in every CTA.  This kernel computes those values
// once, keeps the BF16 pre-mix result in shared memory, and immediately
// applies RMSNorm.  The explicit BF16 round before the norm matches the
// observable HcPre -> RMSNorm boundary of the unfused path.
__global__ void DeepSeekV4HcPreFinishNorm4x4096Kernel(
        const __nv_bfloat16 *x, const float *dots, const float *scale,
        const float *base, const float *normWeight, __nv_bfloat16 *normOutput,
        float *post, float *comb, int tokens, int sinkhornIters,
        float eps, float normEps, int dotsStride, int dotParts) {
    constexpr int hcMult = 4;
    constexpr int dim = 4096;
    constexpr int flatDim = hcMult * dim;
    constexpr int mixHc = (2 + hcMult) * hcMult;
    constexpr int hcSq = hcMult * hcMult;
    constexpr int pairCount = dim / 2;

    __shared__ float mixes[mixHc];
    __shared__ float pre[hcMult];
    __shared__ float warpSums[32];
    __shared__ float normScale;
    __shared__ __nv_bfloat162 yShared[pairCount];

    int token = blockIdx.x;
    if (token >= tokens) {
        return;
    }
    const __nv_bfloat16 *xrow = x + (uint64_t)token * flatDim;

    if (threadIdx.x == 0) {
        float sqSum = 0.0f;
        uint64_t sqBase = ((uint64_t)token * dotsStride + mixHc) * dotParts;
        for (int part = 0; part < dotParts; part++) {
            sqSum += dots[sqBase + part];
        }
        normScale = rsqrtf(sqSum / flatDim + normEps);
    }
    __syncthreads();

    if (threadIdx.x < mixHc) {
        int m = threadIdx.x;
        float dotSum = 0.0f;
        uint64_t dotBase = ((uint64_t)token * dotsStride + m) * dotParts;
        for (int part = 0; part < dotParts; part++) {
            dotSum += dots[dotBase + part];
        }
        mixes[m] = dotSum * normScale;
    }
    __syncthreads();

    if (threadIdx.x < hcMult) {
        int h = threadIdx.x;
        pre[h] = DeepSeekV4Sigmoid(mixes[h] * scale[0] + base[h]) + eps;
        post[(uint64_t)token * hcMult + h] =
            2.0f * DeepSeekV4Sigmoid(mixes[h + hcMult] * scale[1] + base[h + hcMult]);
    }
    __syncthreads();

    // One warp owns the tiny 4x4 Sinkhorn transform.  Keep one matrix element
    // in each of the first 16 lanes and use width-4/width-16 shuffles for row
    // and column reductions.  This mirrors TileLang's fragment reduction and
    // avoids four shared-memory round trips and syncwarps per iteration.
    if (threadIdx.x < 32) {
        int lane = threadIdx.x;
        float combValue = 0.0f;
        if (lane < hcSq) {
            int mixIdx = lane + 2 * hcMult;
            combValue = mixes[mixIdx] * scale[2] + base[mixIdx];
        }

        float rowMax = combValue;
        rowMax = fmaxf(rowMax,
                       __shfl_xor_sync(0xffffffffu, rowMax, 1, 4));
        rowMax = fmaxf(rowMax,
                       __shfl_xor_sync(0xffffffffu, rowMax, 2, 4));
        combValue = expf(combValue - rowMax);

        float rowSum = combValue;
        rowSum += __shfl_xor_sync(0xffffffffu, rowSum, 1, 4);
        rowSum += __shfl_xor_sync(0xffffffffu, rowSum, 2, 4);
        combValue = combValue / rowSum + eps;

        float colSum = combValue;
        colSum += __shfl_xor_sync(0xffffffffu, colSum, 4, 16);
        colSum += __shfl_xor_sync(0xffffffffu, colSum, 8, 16);
        combValue /= colSum + eps;

        if (sinkhornIters == 20) {
#pragma unroll
            for (int it = 1; it < 20; it++) {
                combValue = DeepSeekV4SinkhornIteration4x4(combValue, eps);
            }
        } else {
            for (int it = 1; it < sinkhornIters; it++) {
                combValue = DeepSeekV4SinkhornIteration4x4(combValue, eps);
            }
        }
        if (lane < hcSq) {
            comb[(uint64_t)token * hcSq + lane] = combValue;
        }
    }

    // Warp 0 is busy with Sinkhorn; distribute BF16 pre-mix pairs across the
    // remaining warps.  Accumulate the rounded values' square sum while they
    // are still in registers, avoiding a full shared-memory read and an extra
    // block synchronization before RMSNorm.
    int yThread = (int)threadIdx.x - 32;
    int yThreads = (int)blockDim.x - 32;
    float sum2 = 0.0f;
    if (yThread >= 0) {
        for (int i = yThread; i < pairCount; i += yThreads) {
            float lo = 0.0f;
            float hi = 0.0f;
#pragma unroll
            for (int h = 0; h < hcMult; h++) {
                float p = pre[h];
                const __nv_bfloat162 *hcRow =
                    reinterpret_cast<const __nv_bfloat162 *>(
                        xrow + (uint64_t)h * dim);
                float2 values = __bfloat1622float2(hcRow[i]);
                lo += p * values.x;
                hi += p * values.y;
            }
            __nv_bfloat162 rounded = __floats2bfloat162_rn(lo, hi);
            yShared[i] = rounded;
            float2 roundedValues = __bfloat1622float2(rounded);
            sum2 += roundedValues.x * roundedValues.x +
                    roundedValues.y * roundedValues.y;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum2 += __shfl_down_sync(0xffffffff, sum2, offset);
    }
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (lane == 0) {
        warpSums[warp] = sum2;
    }
    __syncthreads();
    if (warp == 0) {
        int numWarps = blockDim.x >> 5;
        float value = lane < numWarps ? warpSums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        if (lane == 0) {
            normScale = rsqrtf(value / dim + normEps);
        }
    }
    __syncthreads();

    __nv_bfloat16 *out = normOutput + (uint64_t)token * dim;
    float finalScale = normScale;
    __nv_bfloat162 *outPairs = reinterpret_cast<__nv_bfloat162 *>(out);
    const float2 *weightPairs = reinterpret_cast<const float2 *>(normWeight);
    for (int i = threadIdx.x; i < pairCount; i += blockDim.x) {
        float2 values = __bfloat1622float2(yShared[i]);
        float2 weights = __ldg(weightPairs + i);
        outPairs[i] = __floats2bfloat162_rn(
            values.x * finalScale * weights.x,
            values.y * finalScale * weights.y);
    }
}

template <typename XT>
__global__ void DeepSeekV4HcHeadFinishKernel(const XT *x, const float *dots,
                                             const float *scale, const float *base,
                                             XT *output, int tokens, int dim,
                                             int hcMult, float eps, float normEps) {
    extern __shared__ float shared[];
    float *pre = shared;
    float *rsqrtValue = pre + hcMult;
    int token = blockIdx.x;
    if (token >= tokens) {
        return;
    }
    int flatDim = hcMult * dim;
    int dotsStride = hcMult + 1;
    if (threadIdx.x == 0) {
        float sqSum = dots[(uint64_t)token * dotsStride + hcMult];
        rsqrtValue[0] = rsqrtf(sqSum / flatDim + normEps);
        for (int h = 0; h < hcMult; h++) {
            pre[h] = DeepSeekV4Sigmoid(
                dots[(uint64_t)token * dotsStride + h] * rsqrtValue[0] * scale[0] + base[h]) + eps;
        }
    }
    __syncthreads();
    const XT *xrow = x + (uint64_t)token * flatDim;
    XT *out = output + (uint64_t)token * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float value = 0.0f;
        for (int h = 0; h < hcMult; h++) {
            value += pre[h] * Dsv4ToFloat(xrow[(uint64_t)h * dim + d]);
        }
        out[d] = Dsv4FromFloat<XT>(value);
    }
}

template <typename XT, typename WT>
__global__ void DeepSeekV4HcHeadDotsKernel(const XT *x, const WT *weight,
                                           float *dots, int tokens,
                                           int flatDim, int hcMult,
                                           int dotsStride) {
    extern __shared__ double hcHeadRed[];
    int index = blockIdx.x;
    int token = index / hcMult;
    int h = index - token * hcMult;
    if (token >= tokens) {
        return;
    }
    const XT *xrow = x + (uint64_t)token * flatDim;
    const WT *wrow = weight + (uint64_t)h * flatDim;
    double sum = 0.0;
    for (int k = threadIdx.x; k < flatDim; k += blockDim.x) {
        sum += (double)Dsv4ToFloat(xrow[k]) * Dsv4ToFloat(wrow[k]);
    }
    hcHeadRed[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            hcHeadRed[threadIdx.x] += hcHeadRed[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        dots[(uint64_t)token * dotsStride + h] = (float)hcHeadRed[0];
    }
}

template <typename XT, typename WT>
__global__ void DeepSeekV4HcPreKernel(const XT *x, const WT *fn, const float *scale,
                                      const float *base, XT *y, float *post, float *comb,
                                      int tokens, int dim, int hcMult, int sinkhornIters,
                                      float eps, float normEps) {
    extern __shared__ float shared[];
    int t = blockIdx.x;
    int flatDim = hcMult * dim;
    int mixHc = (2 + hcMult) * hcMult;
    float *red = shared;
    float *mixes = red + blockDim.x;
    float *pre = mixes + mixHc;
    float *combLocal = pre + hcMult;

    const XT *xrow = x + (uint64_t)t * flatDim;
    float partial = 0.0f;
    for (int k = threadIdx.x; k < flatDim; k += blockDim.x) {
        float v = Dsv4ToFloat(xrow[k]);
        partial += v * v;
    }
    red[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float rsqrt = rsqrtf(red[0] / flatDim + normEps);

    for (int m = 0; m < mixHc; m++) {
        const WT *wrow = fn + (uint64_t)m * flatDim;
        partial = 0.0f;
        for (int k = threadIdx.x; k < flatDim; k += blockDim.x) {
            partial += Dsv4ToFloat(xrow[k]) * Dsv4ToFloat(wrow[k]);
        }
        red[threadIdx.x] = partial;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] += red[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            mixes[m] = red[0] * rsqrt;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (int h = 0; h < hcMult; h++) {
            pre[h] = DeepSeekV4Sigmoid(mixes[h] * scale[0] + base[h]) + eps;
            post[(uint64_t)t * hcMult + h] =
                2.0f * DeepSeekV4Sigmoid(mixes[h + hcMult] * scale[1] + base[h + hcMult]);
        }

        for (int r = 0; r < hcMult; r++) {
            float rowMax = -INFINITY;
            for (int c = 0; c < hcMult; c++) {
                int idx = r * hcMult + c;
                int mixIdx = idx + 2 * hcMult;
                combLocal[idx] = mixes[mixIdx] * scale[2] + base[mixIdx];
                rowMax = fmaxf(rowMax, combLocal[idx]);
            }
            float rowSum = 0.0f;
            for (int c = 0; c < hcMult; c++) {
                int idx = r * hcMult + c;
                float v = expf(combLocal[idx] - rowMax);
                combLocal[idx] = v;
                rowSum += v;
            }
            for (int c = 0; c < hcMult; c++) {
                int idx = r * hcMult + c;
                combLocal[idx] = combLocal[idx] / rowSum + eps;
            }
        }
        for (int c = 0; c < hcMult; c++) {
            float colSum = 0.0f;
            for (int r = 0; r < hcMult; r++) {
                colSum += combLocal[r * hcMult + c];
            }
            for (int r = 0; r < hcMult; r++) {
                combLocal[r * hcMult + c] /= (colSum + eps);
            }
        }
        for (int it = 1; it < sinkhornIters; it++) {
            for (int r = 0; r < hcMult; r++) {
                float rowSum = 0.0f;
                for (int c = 0; c < hcMult; c++) {
                    rowSum += combLocal[r * hcMult + c];
                }
                for (int c = 0; c < hcMult; c++) {
                    combLocal[r * hcMult + c] /= (rowSum + eps);
                }
            }
            for (int c = 0; c < hcMult; c++) {
                float colSum = 0.0f;
                for (int r = 0; r < hcMult; r++) {
                    colSum += combLocal[r * hcMult + c];
                }
                for (int r = 0; r < hcMult; r++) {
                    combLocal[r * hcMult + c] /= (colSum + eps);
                }
            }
        }
        for (int i = 0; i < hcMult * hcMult; i++) {
            comb[(uint64_t)t * hcMult * hcMult + i] = combLocal[i];
        }
    }
    __syncthreads();

    XT *yrow = y + (uint64_t)t * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = 0.0f;
        for (int h = 0; h < hcMult; h++) {
            v += pre[h] * Dsv4ToFloat(xrow[(uint64_t)h * dim + d]);
        }
        yrow[d] = Dsv4FromFloat<XT>(v);
    }
}

template <typename QT, typename CT>
__global__ void DeepSeekV4SparseAttentionDecodeCachedKernel(const QT *q, const float *windowKV,
                                                            const CT *compressedKV, const float *sink,
                                                            float *output, int bsz, int heads, int dim,
                                                            int windowSize, int startPos, int compressedCount,
                                                            float softmaxScale) {
    int bh = blockIdx.x;
    int b = bh / heads;
    int h = bh % heads;
    const QT *qrow = q + ((uint64_t)b * heads + h) * dim;
    float *orow = output + ((uint64_t)b * heads + h) * dim;

    int idxCount = 0;
    int pos = startPos % windowSize;
    int idxs[512];
    if (startPos >= windowSize - 1) {
        for (int i = pos + 1; i < windowSize; i++) {
            idxs[idxCount++] = i;
        }
        for (int i = 0; i <= pos; i++) {
            idxs[idxCount++] = i;
        }
    } else {
        for (int i = 0; i <= startPos; i++) {
            idxs[idxCount++] = i;
        }
    }
    for (int i = 0; i < compressedCount; i++) {
        idxs[idxCount++] = windowSize + i;
    }

    double scores[512];
    float mx = -INFINITY;
    for (int k = 0; k < idxCount; k++) {
        const float *windowRow = nullptr;
        const CT *compressedRow = nullptr;
        int idx = idxs[k];
        if (idx < windowSize) {
            windowRow = windowKV + ((uint64_t)b * windowSize + idx) * dim;
        } else {
            compressedRow = compressedKV + ((uint64_t)b * compressedCount + (idx - windowSize)) * dim;
        }
        double dot = 0.0;
        for (int d = 0; d < dim; d++) {
            float kv = windowRow != nullptr ? windowRow[d] : Dsv4ToFloat(compressedRow[d]);
            dot += (double)Dsv4ToFloat(qrow[d]) * kv;
        }
        float score = (float)dot * softmaxScale;
        scores[k] = (double)score;
        mx = fmaxf(mx, score);
    }

    float safeMx = isfinite(mx) ? mx : 0.0f;
    double denom = exp((double)sink[h] - safeMx);
    for (int k = 0; k < idxCount; k++) {
        denom += exp(scores[k] - safeMx);
    }

    for (int d = 0; d < dim; d++) {
        double v = 0.0;
        for (int k = 0; k < idxCount; k++) {
            int idx = idxs[k];
            float kv;
            if (idx < windowSize) {
                kv = windowKV[((uint64_t)b * windowSize + idx) * dim + d];
            } else {
                kv = Dsv4ToFloat(compressedKV[((uint64_t)b * compressedCount + (idx - windowSize)) * dim + d]);
            }
            double w = exp(scores[k] - safeMx) / fmax(denom, 1e-30);
            v += w * kv;
        }
        orow[d] = (float)v;
    }
}

template <typename QT, typename CT>
__global__ void DeepSeekV4SparseAttentionDecodeCachedBlockKernel(const QT *q, const float *windowKV,
                                                                 const CT *compressedKV, const float *sink,
                                                                 float *output, int bsz, int heads, int dim,
                                                                 int windowSize, int startPos, int compressedCount,
                                                                 int keyCapacity, float softmaxScale) {
    extern __shared__ float shared[];
    float *scores = shared;
    float *red = shared + keyCapacity;

    int bh = blockIdx.x;
    int b = bh / heads;
    int h = bh % heads;
    const QT *qrow = q + ((uint64_t)b * heads + h) * dim;
    float *orow = output + ((uint64_t)b * heads + h) * dim;

    int liveWindow = startPos >= windowSize - 1 ? windowSize : (startPos + 1);
    int idxCount = liveWindow + compressedCount;
    if (idxCount <= 0 || idxCount > keyCapacity || idxCount > kDeepSeekV4SparseDecodeMaxKeys) {
        return;
    }
    int pos = startPos % windowSize;

    __shared__ float mxShared;
    __shared__ float denomShared;
    if (threadIdx.x == 0) {
        mxShared = -INFINITY;
    }
    __syncthreads();

    for (int k = 0; k < idxCount; k++) {
        int idx;
        if (k < liveWindow) {
            idx = (startPos >= windowSize - 1) ? ((pos + 1 + k) % windowSize) : k;
        } else {
            idx = windowSize + (k - liveWindow);
        }

        float partial = 0.0f;
        for (int d = threadIdx.x; d < dim; d += blockDim.x) {
            float kv;
            if (idx < windowSize) {
                kv = windowKV[((uint64_t)b * windowSize + idx) * dim + d];
            } else {
                kv = Dsv4ToFloat(compressedKV[((uint64_t)b * compressedCount + (idx - windowSize)) * dim + d]);
            }
            partial += Dsv4ToFloat(qrow[d]) * kv;
        }

        red[threadIdx.x] = partial;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] += red[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            float score = red[0] * softmaxScale;
            scores[k] = score;
            mxShared = fmaxf(mxShared, score);
        }
        __syncthreads();
    }

    float safeMx = isfinite(mxShared) ? mxShared : 0.0f;
    float partialDenom = 0.0f;
    for (int k = threadIdx.x; k < idxCount; k += blockDim.x) {
        partialDenom += expf(scores[k] - safeMx);
    }
    red[threadIdx.x] = partialDenom;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        denomShared = red[0] + expf(sink[h] - safeMx);
    }
    __syncthreads();

    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = 0.0f;
        for (int k = 0; k < idxCount; k++) {
            int idx;
            if (k < liveWindow) {
                idx = (startPos >= windowSize - 1) ? ((pos + 1 + k) % windowSize) : k;
            } else {
                idx = windowSize + (k - liveWindow);
            }
            float kv;
            if (idx < windowSize) {
                kv = windowKV[((uint64_t)b * windowSize + idx) * dim + d];
            } else {
                kv = Dsv4ToFloat(compressedKV[((uint64_t)b * compressedCount + (idx - windowSize)) * dim + d]);
            }
            float w = expf(scores[k] - safeMx) / fmaxf(denomShared, 1e-30f);
            v += w * kv;
        }
        orow[d] = v;
    }
}

template <typename QT, typename CT>
__global__ void DeepSeekV4SparseAttentionDecodeCachedOnlineKernel(const QT *q, const float *windowKV,
                                                                  const CT *compressedKV, const float *sink,
                                                                  float *output, int bsz, int heads, int dim,
                                                                  int windowSize, int startPos, int compressedCount,
                                                                  float softmaxScale,
                                                                  const int32_t *decodeMeta,
                                                                  int compressRatio) {
    constexpr int kMaxDimsPerThread = 4;
    __shared__ float red[256];
    __shared__ float mxShared;
    __shared__ float denomShared;
    __shared__ float alphaShared;
    __shared__ float betaShared;

    int bh = blockIdx.x;
    int b = bh / heads;
    int h = bh % heads;
    const QT *qrow = q + ((uint64_t)b * heads + h) * dim;
    float *orow = output + ((uint64_t)b * heads + h) * dim;

    if (decodeMeta != nullptr) {
        startPos = decodeMeta[0];
        compressedCount = compressRatio > 0 ? (startPos + 1) / compressRatio : 0;
    }
    int liveWindow = startPos >= windowSize - 1 ? windowSize : (startPos + 1);
    int idxCount = liveWindow + compressedCount;
    if (idxCount <= 0 || idxCount > kDeepSeekV4SparseDecodeMaxKeys ||
        dim > blockDim.x * kMaxDimsPerThread) {
        return;
    }
    int pos = startPos % windowSize;

    int localDims = 0;
    int localOffsets[kMaxDimsPerThread];
    float localAcc[kMaxDimsPerThread];
    float localKV[kMaxDimsPerThread];
    for (int d = threadIdx.x; d < dim && localDims < kMaxDimsPerThread; d += blockDim.x) {
        localOffsets[localDims] = d;
        localAcc[localDims] = 0.0f;
        localKV[localDims] = 0.0f;
        localDims++;
    }

    if (threadIdx.x == 0) {
        mxShared = -INFINITY;
        denomShared = 0.0f;
    }
    __syncthreads();

    for (int k = 0; k < idxCount; k++) {
        int idx = k < liveWindow
                      ? ((startPos >= windowSize - 1) ? ((pos + 1 + k) % windowSize) : k)
                      : (windowSize + (k - liveWindow));

        float partial = 0.0f;
        for (int i = 0; i < localDims; i++) {
            int d = localOffsets[i];
            float kv;
            if (idx < windowSize) {
                kv = windowKV[((uint64_t)b * windowSize + idx) * dim + d];
            } else {
                kv = Dsv4ToFloat(compressedKV[((uint64_t)b * compressedCount + (idx - windowSize)) * dim + d]);
            }
            localKV[i] = kv;
            partial += Dsv4ToFloat(qrow[d]) * kv;
        }

        red[threadIdx.x] = partial;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] += red[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            float score = red[0] * softmaxScale;
            float oldMx = mxShared;
            float newMx = fmaxf(oldMx, score);
            float alpha = isfinite(oldMx) ? expf(oldMx - newMx) : 0.0f;
            float beta = expf(score - newMx);
            mxShared = newMx;
            denomShared = denomShared * alpha + beta;
            alphaShared = alpha;
            betaShared = beta;
        }
        __syncthreads();
        for (int i = 0; i < localDims; i++) {
            localAcc[i] = localAcc[i] * alphaShared + betaShared * localKV[i];
        }
    }

    float finalDenom = denomShared + expf(sink[h] - mxShared);
    finalDenom = fmaxf(finalDenom, 1e-30f);
    for (int i = 0; i < localDims; i++) {
        orow[localOffsets[i]] = localAcc[i] / finalDenom;
    }
}

template <typename QT, typename CT>
__global__ void DeepSeekV4SparseAttentionDecodeCachedBatchOnlineKernel(
        const void * const *qPtrs, const float * const *windowPtrs,
        const void * const *compressedPtrs, const float *sink,
        const int *startPositions, const int *compressedCounts,
        float *output, int batch, int heads, int dim, int windowSize,
        float softmaxScale) {
    constexpr int kMaxDimsPerThread = 4;
    __shared__ float red[256];
    __shared__ float mxShared;
    __shared__ float denomShared;
    __shared__ float alphaShared;
    __shared__ float betaShared;

    int bh = blockIdx.x;
    int b = bh / heads;
    int h = bh % heads;
    if (b >= batch) {
        return;
    }

    const QT *qBase = reinterpret_cast<const QT *>(qPtrs[b]);
    const float *windowKV = windowPtrs[b];
    const CT *compressedKV = reinterpret_cast<const CT *>(compressedPtrs[b]);
    int startPos = startPositions[b];
    int compressedCount = compressedCounts[b];
    int liveWindow = startPos >= windowSize - 1 ? windowSize : (startPos + 1);
    int idxCount = liveWindow + compressedCount;
    if (qBase == nullptr || windowKV == nullptr || startPos < 0 ||
        idxCount <= 0 || idxCount > kDeepSeekV4SparseDecodeMaxKeys ||
        dim > blockDim.x * kMaxDimsPerThread ||
        (compressedCount > 0 && compressedKV == nullptr)) {
        return;
    }

    int pos = startPos % windowSize;
    const QT *qrow = qBase + (uint64_t)h * dim;
    float *orow = output + (uint64_t)bh * dim;

    int localDims = 0;
    int localOffsets[kMaxDimsPerThread];
    float localAcc[kMaxDimsPerThread];
    float localKV[kMaxDimsPerThread];
    for (int d = threadIdx.x; d < dim && localDims < kMaxDimsPerThread; d += blockDim.x) {
        localOffsets[localDims] = d;
        localAcc[localDims] = 0.0f;
        localKV[localDims] = 0.0f;
        localDims++;
    }

    if (threadIdx.x == 0) {
        mxShared = -INFINITY;
        denomShared = 0.0f;
    }
    __syncthreads();

    for (int k = 0; k < idxCount; k++) {
        int idx = k < liveWindow
                      ? ((startPos >= windowSize - 1) ? ((pos + 1 + k) % windowSize) : k)
                      : (windowSize + (k - liveWindow));

        float partial = 0.0f;
        for (int i = 0; i < localDims; i++) {
            int d = localOffsets[i];
            float kv;
            if (idx < windowSize) {
                kv = windowKV[(uint64_t)idx * dim + d];
            } else {
                kv = Dsv4ToFloat(compressedKV[((uint64_t)(idx - windowSize)) * dim + d]);
            }
            localKV[i] = kv;
            partial += Dsv4ToFloat(qrow[d]) * kv;
        }

        red[threadIdx.x] = partial;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                red[threadIdx.x] += red[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            float score = red[0] * softmaxScale;
            float oldMx = mxShared;
            float newMx = fmaxf(oldMx, score);
            float alpha = isfinite(oldMx) ? expf(oldMx - newMx) : 0.0f;
            float beta = expf(score - newMx);
            mxShared = newMx;
            denomShared = denomShared * alpha + beta;
            alphaShared = alpha;
            betaShared = beta;
        }
        __syncthreads();
        for (int i = 0; i < localDims; i++) {
            localAcc[i] = localAcc[i] * alphaShared + betaShared * localKV[i];
        }
    }

    float finalDenom = denomShared + expf(sink[h] - mxShared);
    finalDenom = fmaxf(finalDenom, 1e-30f);
    for (int i = 0; i < localDims; i++) {
        orow[localOffsets[i]] = localAcc[i] / finalDenom;
    }
}

template <typename QT>
__global__ void DeepSeekV4SparseDecodeConvertQKernel(const QT *q, float *qFloat, uint64_t total) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        qFloat[idx] = Dsv4ToFloat(q[idx]);
    }
}

template <typename CT>
__global__ void DeepSeekV4SparseDecodeGatherKVKernel(const float *windowKV, const CT *compressedKV,
                                                     float *kvFloat, int bsz, int dim, int windowSize,
                                                     int startPos, int compressedCount, int liveWindow,
                                                     int keyCount) {
    uint64_t total = (uint64_t)bsz * keyCount * dim;
    uint64_t linear = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }
    int d = (int)(linear % dim);
    uint64_t tmp = linear / dim;
    int k = (int)(tmp % keyCount);
    int b = (int)(tmp / keyCount);
    int pos = startPos % windowSize;

    float kv;
    if (k < liveWindow) {
        int idx = (startPos >= windowSize - 1) ? ((pos + 1 + k) % windowSize) : k;
        kv = windowKV[((uint64_t)b * windowSize + idx) * dim + d];
    } else {
        int ck = k - liveWindow;
        kv = Dsv4ToFloat(compressedKV[((uint64_t)b * compressedCount + ck) * dim + d]);
    }
    kvFloat[((uint64_t)b * keyCount + k) * dim + d] = kv;
}

template <typename KT>
__global__ void DeepSeekV4SparsePrefillCastCompressedKVKernel(const KT *kv, float *kvFloat,
                                                              int bsz, int kvLen, int compressedStart,
                                                              int compressedCount, int dim) {
    uint64_t total = (uint64_t)bsz * compressedCount * dim;
    uint64_t linear = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (linear >= total) {
        return;
    }
    int d = (int)(linear % dim);
    uint64_t tmp = linear / dim;
    int k = (int)(tmp % compressedCount);
    int b = (int)(tmp / compressedCount);
    kvFloat[((uint64_t)b * compressedCount + k) * dim + d] =
        Dsv4ToFloat(kv[((uint64_t)b * kvLen + compressedStart + k) * dim + d]);
}

__global__ void DeepSeekV4SparseDecodeSoftmaxSinkKernel(float *scores, const float *sink,
                                                        int bsz, int heads, int keyCount) {
    extern __shared__ float red[];
    int row = blockIdx.x;
    int h = row % heads;
    float *rowScores = scores + (uint64_t)row * keyCount;

    float mx = -INFINITY;
    for (int k = threadIdx.x; k < keyCount; k += blockDim.x) {
        mx = fmaxf(mx, rowScores[k]);
    }
    red[threadIdx.x] = mx;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float safeMx = isfinite(red[0]) ? red[0] : 0.0f;

    float sum = 0.0f;
    for (int k = threadIdx.x; k < keyCount; k += blockDim.x) {
        sum += expf(rowScores[k] - safeMx);
    }
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float denom = red[0] + expf(sink[h] - safeMx);
    denom = fmaxf(denom, 1e-30f);

    for (int k = threadIdx.x; k < keyCount; k += blockDim.x) {
        rowScores[k] = expf(rowScores[k] - safeMx) / denom;
    }
}

template <typename QT, typename KT>
__global__ void DeepSeekV4SparseAttentionPrefillBlockKernel(const QT *q, const KT *kv,
                                                            const float *sink, float *output,
                                                            int bsz, int seqlen, int heads, int dim,
                                                            int kvLen, int windowSize, int compressRatio,
                                                            int startPos, int prefixLen, float softmaxScale,
                                                            int rowOffset) {
    extern __shared__ float scores[];

    int localRow = blockIdx.x;
    int row = rowOffset + localRow;
    int h = row % heads;
    int tmp = row / heads;
    int s = tmp % seqlen;
    int b = tmp / seqlen;
    if (b >= bsz) {
        return;
    }

    int realPrefixLen = max(0, min(prefixLen, kvLen - seqlen));
    int compressedStart = realPrefixLen + seqlen;
    int compressedCount = max(0, kvLen - compressedStart);
    int liveWindow = min(windowSize, realPrefixLen + s + 1);
    int beginPos = startPos + s - liveWindow + 1;
    int prefixStartPos = startPos - realPrefixLen;
    int liveCompressed = 0;
    if (compressRatio > 0 && compressedCount > 0) {
        liveCompressed = min(compressedCount, (startPos + s + 1) / compressRatio);
    }
    int idxCount = liveWindow + liveCompressed;
    if (idxCount <= 0 || idxCount > kDeepSeekV4SparsePrefillMaxKeys) {
        return;
    }

    __shared__ float mxShared;
    __shared__ float denomShared;
    const QT *qrow = q + (((uint64_t)b * seqlen + s) * heads + h) * dim;

    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    for (int base = 0; base < idxCount; base += warps) {
        int k = base + warpId;
        float dot = 0.0f;
        if (k < idxCount) {
            int idx;
            if (k < liveWindow) {
                int pos = beginPos + k;
                idx = (pos < startPos) ? (pos - prefixStartPos) : (realPrefixLen + pos - startPos);
            } else {
                idx = compressedStart + k - liveWindow;
            }
            const KT *kvrow = kv + ((uint64_t)b * kvLen + idx) * dim;
            for (int d = lane; d < dim; d += 32) {
                dot += Dsv4ToFloat(qrow[d]) * Dsv4ToFloat(kvrow[d]);
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
        if (lane == 0 && k < idxCount) {
            scores[k] = dot * softmaxScale;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float mx = -INFINITY;
        for (int k = 0; k < idxCount; k++) {
            mx = fmaxf(mx, scores[k]);
        }
        mxShared = isfinite(mx) ? mx : 0.0f;
        float denom = expf(sink[h] - mxShared);
        for (int k = 0; k < idxCount; k++) {
            denom += expf(scores[k] - mxShared);
        }
        denomShared = fmaxf(denom, 1e-30f);
    }
    __syncthreads();

    for (int k = threadIdx.x; k < idxCount; k += blockDim.x) {
        scores[k] = expf(scores[k] - mxShared) / denomShared;
    }
    __syncthreads();

    float *orow = output + (uint64_t)localRow * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = 0.0f;
        for (int k = 0; k < idxCount; k++) {
            int idx;
            if (k < liveWindow) {
                int pos = beginPos + k;
                idx = (pos < startPos) ? (pos - prefixStartPos) : (realPrefixLen + pos - startPos);
            } else {
                idx = compressedStart + k - liveWindow;
            }
            const KT *kvrow = kv + ((uint64_t)b * kvLen + idx) * dim;
            v += scores[k] * Dsv4ToFloat(kvrow[d]);
        }
        orow[d] = v;
    }
}

template <typename QT, typename KT>
__global__ void DeepSeekV4SparseAttentionPrefillCublasCompressedKernel(
        const QT *q, const KT *kv, const float *sink, float *compressedScores, float *output,
        int bsz, int seqlen, int heads, int dim, int kvLen, int windowSize, int compressRatio,
        int startPos, int prefixLen, int compressedStart, int compressedCount, float softmaxScale,
        int rowOffset) {
    extern __shared__ float localScores[];

    int localRow = blockIdx.x;
    int row = rowOffset + localRow;
    int h = row % heads;
    int tmp = row / heads;
    int s = tmp % seqlen;
    int b = tmp / seqlen;
    if (b >= bsz) {
        return;
    }

    int realPrefixLen = max(0, min(prefixLen, kvLen - seqlen));
    int liveWindow = min(windowSize, realPrefixLen + s + 1);
    int beginPos = startPos + s - liveWindow + 1;
    int prefixStartPos = startPos - realPrefixLen;
    int liveCompressed = 0;
    if (compressRatio > 0 && compressedCount > 0) {
        liveCompressed = min(compressedCount, (startPos + s + 1) / compressRatio);
    }
    if (liveWindow + liveCompressed <= 0) {
        return;
    }

    __shared__ float mxShared;
    __shared__ float denomShared;
    const QT *qrow = q + (((uint64_t)b * seqlen + s) * heads + h) * dim;
    float *crow = compressedScores + (uint64_t)localRow * compressedCount;

    int lane = threadIdx.x & 31;
    int warpId = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;
    for (int base = 0; base < liveWindow; base += warps) {
        int k = base + warpId;
        float dot = 0.0f;
        if (k < liveWindow) {
            int pos = beginPos + k;
            int idx = (pos < startPos) ? (pos - prefixStartPos) : (realPrefixLen + pos - startPos);
            const KT *kvrow = kv + ((uint64_t)b * kvLen + idx) * dim;
            for (int d = lane; d < dim; d += 32) {
                dot += Dsv4ToFloat(qrow[d]) * Dsv4ToFloat(kvrow[d]);
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
        if (lane == 0 && k < liveWindow) {
            localScores[k] = dot * softmaxScale;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float mx = -INFINITY;
        for (int k = 0; k < liveWindow; k++) {
            mx = fmaxf(mx, localScores[k]);
        }
        for (int k = 0; k < liveCompressed; k++) {
            mx = fmaxf(mx, crow[k]);
        }
        mxShared = isfinite(mx) ? mx : 0.0f;
        float denom = expf(sink[h] - mxShared);
        for (int k = 0; k < liveWindow; k++) {
            denom += expf(localScores[k] - mxShared);
        }
        for (int k = 0; k < liveCompressed; k++) {
            denom += expf(crow[k] - mxShared);
        }
        denomShared = fmaxf(denom, 1e-30f);
    }
    __syncthreads();

    for (int k = threadIdx.x; k < liveWindow; k += blockDim.x) {
        localScores[k] = expf(localScores[k] - mxShared) / denomShared;
    }
    for (int k = threadIdx.x; k < liveCompressed; k += blockDim.x) {
        crow[k] = expf(crow[k] - mxShared) / denomShared;
    }
    for (int k = liveCompressed + threadIdx.x; k < compressedCount; k += blockDim.x) {
        crow[k] = 0.0f;
    }
    __syncthreads();

    float *orow = output + (uint64_t)localRow * dim;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float v = 0.0f;
        for (int k = 0; k < liveWindow; k++) {
            int pos = beginPos + k;
            int idx = (pos < startPos) ? (pos - prefixStartPos) : (realPrefixLen + pos - startPos);
            const KT *kvrow = kv + ((uint64_t)b * kvLen + idx) * dim;
            v += localScores[k] * Dsv4ToFloat(kvrow[d]);
        }
        orow[d] = v;
    }
}

__global__ void DeepSeekV4SparsePrefillCublasLocalSoftmaxKernel(
        float *localScores, float *compressedScores,
        __nv_bfloat16 *localScoresBf16, __nv_bfloat16 *compressedScoresBf16,
        const float *sink,
        int bsz, int seqlen, int heads, int rawStart, int rawKeyCount,
        int compressedCount, int windowSize, int compressRatio, int startPos,
        int prefixLen, int kvLen, int rowOffset, int rows) {
    extern __shared__ float red[];
    int localRow = blockIdx.x;
    if (localRow >= rows) {
        return;
    }

    int row = rowOffset + localRow;
    int h = row % heads;
    int tmp = row / heads;
    int s = tmp % seqlen;
    int b = tmp / seqlen;
    if (b >= bsz) {
        return;
    }

    int realPrefixLen = max(0, min(prefixLen, kvLen - seqlen));
    int liveWindow = min(windowSize, realPrefixLen + s + 1);
    int highRaw = realPrefixLen + s;
    int lowRaw = highRaw - liveWindow + 1;
    int liveCompressed = 0;
    if (compressRatio > 0 && compressedCount > 0) {
        liveCompressed = min(compressedCount, (startPos + s + 1) / compressRatio);
    }

    float *lrow = localScores + (uint64_t)localRow * rawKeyCount;
    float *crow = compressedScores == nullptr ? nullptr : compressedScores + (uint64_t)localRow * compressedCount;
    __nv_bfloat16 *lbrow = localScoresBf16 == nullptr ? nullptr :
        localScoresBf16 + (uint64_t)localRow * rawKeyCount;
    __nv_bfloat16 *cbrow = compressedScoresBf16 == nullptr ? nullptr :
        compressedScoresBf16 + (uint64_t)localRow * compressedCount;

    float mx = -INFINITY;
    for (int k = threadIdx.x; k < rawKeyCount; k += blockDim.x) {
        int rawIdx = rawStart + k;
        bool visible = rawIdx >= lowRaw && rawIdx <= highRaw;
        float v = visible ? lrow[k] : -INFINITY;
        lrow[k] = v;
        mx = fmaxf(mx, v);
    }
    if (crow != nullptr) {
        for (int k = threadIdx.x; k < compressedCount; k += blockDim.x) {
            bool visible = k < liveCompressed;
            float v = visible ? crow[k] : -INFINITY;
            crow[k] = v;
            mx = fmaxf(mx, v);
        }
    }
    red[threadIdx.x] = mx;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float safeMx = isfinite(red[0]) ? red[0] : 0.0f;

    float sum = 0.0f;
    for (int k = threadIdx.x; k < rawKeyCount; k += blockDim.x) {
        if (isfinite(lrow[k])) {
            sum += expf(lrow[k] - safeMx);
        }
    }
    if (crow != nullptr) {
        for (int k = threadIdx.x; k < compressedCount; k += blockDim.x) {
            if (isfinite(crow[k])) {
                sum += expf(crow[k] - safeMx);
            }
        }
    }
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            red[threadIdx.x] += red[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float denom = fmaxf(red[0] + expf(sink[h] - safeMx), 1e-30f);

    for (int k = threadIdx.x; k < rawKeyCount; k += blockDim.x) {
        float p = isfinite(lrow[k]) ? expf(lrow[k] - safeMx) / denom : 0.0f;
        lrow[k] = p;
        if (lbrow != nullptr) {
            lbrow[k] = __float2bfloat16_rn(p);
        }
    }
    if (crow != nullptr) {
        for (int k = threadIdx.x; k < compressedCount; k += blockDim.x) {
            float p = isfinite(crow[k]) ? expf(crow[k] - safeMx) / denom : 0.0f;
            crow[k] = p;
            if (cbrow != nullptr) {
                cbrow[k] = __float2bfloat16_rn(p);
            }
        }
    }
}

__global__ void DeepSeekV4SparsePrefillRotaryCastKernel(const float *input, __nv_bfloat16 *output,
                                                        int rows, int rowOffset,
                                                        int seqlen, int heads, int dim,
                                                        int ropeDim, float ropeBase, int startPos,
                                                        int originalSeqLen, float ropeFactor,
                                                        int betaFast, int betaSlow) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    int globalRow = rowOffset + row;
    int s = (globalRow / heads) % seqlen;
    int pos = startPos + s;
    const float *src = input + (uint64_t)row * dim;
    __nv_bfloat16 *dst = output + (uint64_t)globalRow * dim;
    int off = dim - ropeDim;
    for (int d = 0; d < off; d++) {
        dst[d] = __float2bfloat16_rn(src[d]);
    }
    for (int i = 0; i < ropeDim; i += 2) {
        float inv = DeepSeekV4InvFreq(i / 2, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
        float ang = pos * inv;
        float c = cosf(ang);
        float sn = -sinf(ang);
        float a = src[off + i];
        float b = src[off + i + 1];
        dst[off + i] = __float2bfloat16_rn(a * c - b * sn);
        dst[off + i + 1] = __float2bfloat16_rn(a * sn + b * c);
    }
}

__global__ void DeepSeekV4SparseDecodeRotaryCastKernel(const float *input, __nv_bfloat16 *output,
                                                       int rows, int dim, int ropeDim,
                                                       float ropeBase, int startPos,
                                                       int originalSeqLen, float ropeFactor,
                                                       int betaFast, int betaSlow,
                                                       const int32_t *decodeMeta) {
    int off = dim - ropeDim;
    int ropePairs = ropeDim >> 1;
    int workPerRow = off + ropePairs;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)rows * workPerRow;
    if (idx >= total) {
        return;
    }
    int item = idx % workPerRow;
    int row = idx / workPerRow;
    const float *src = input + (uint64_t)row * dim;
    __nv_bfloat16 *dst = output + (uint64_t)row * dim;
    if (item < off) {
        dst[item] = __float2bfloat16_rn(src[item]);
        return;
    }
    int i = (item - off) << 1;
    float inv = DeepSeekV4InvFreq(i / 2, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
    int dynamicStartPos = decodeMeta == nullptr ? startPos : decodeMeta[0];
    float ang = dynamicStartPos * inv;
    float c = cosf(ang);
    float sn = -sinf(ang);
    float a = src[off + i];
    float b = src[off + i + 1];
    dst[off + i] = __float2bfloat16_rn(a * c - b * sn);
    dst[off + i + 1] = __float2bfloat16_rn(a * sn + b * c);
}

__global__ void DeepSeekV4SparseDecodeRotaryCastBatchKernel(const float *input, __nv_bfloat16 *output,
                                                            const int *startPositions,
                                                            int rows, int heads, int dim, int ropeDim,
                                                            float ropeBase, int originalSeqLen,
                                                            float ropeFactor, int betaFast,
                                                            int betaSlow) {
    int off = dim - ropeDim;
    int ropePairs = ropeDim >> 1;
    int workPerRow = off + ropePairs;
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)rows * workPerRow;
    if (idx >= total) {
        return;
    }
    int item = idx % workPerRow;
    int row = idx / workPerRow;
    int b = row / heads;
    int startPos = startPositions[b];
    const float *src = input + (uint64_t)row * dim;
    __nv_bfloat16 *dst = output + (uint64_t)row * dim;
    if (item < off) {
        dst[item] = __float2bfloat16_rn(src[item]);
        return;
    }
    int i = (item - off) << 1;
    float inv = DeepSeekV4InvFreq(i / 2, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
    float ang = startPos * inv;
    float c = cosf(ang);
    float sn = -sinf(ang);
    float a = src[off + i];
    float bval = src[off + i + 1];
    dst[off + i] = __float2bfloat16_rn(a * c - bval * sn);
    dst[off + i + 1] = __float2bfloat16_rn(a * sn + bval * c);
}

template <typename XT, typename RT, typename OT>
__global__ void DeepSeekV4HcPostKernel(const XT *x, const RT *residual, const float *post,
                                       const float *comb, OT *output, int tokens,
                                       int hcMult, int dim) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)tokens * hcMult * dim;
    if (idx >= total) {
        return;
    }

    int d = idx % dim;
    uint64_t tmp = idx / dim;
    int target = tmp % hcMult;
    int t = tmp / hcMult;

    double v = (double)post[(uint64_t)t * hcMult + target] * Dsv4ToFloat(x[(uint64_t)t * dim + d]);
    const float *combRow = comb + (uint64_t)t * hcMult * hcMult;
    const RT *resRow = residual + (uint64_t)t * hcMult * dim;
    for (int src = 0; src < hcMult; src++) {
        v += (double)combRow[src * hcMult + target] * Dsv4ToFloat(resRow[(uint64_t)src * dim + d]);
    }
    output[idx] = Dsv4FromFloat<OT>((float)v);
}

template <typename XT, typename RT, typename OT>
__global__ void DeepSeekV4HcPost4Kernel(const XT *x, const RT *residual, const float *post,
                                        const float *comb, OT *output, int tokens, int dim) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = (uint64_t)tokens * dim;
    if (idx >= total) {
        return;
    }

    int d = idx % dim;
    int t = idx / dim;
    const RT *resRow = residual + (uint64_t)t * 4 * dim;
    double r0 = (double)Dsv4ToFloat(resRow[d]);
    double r1 = (double)Dsv4ToFloat(resRow[(uint64_t)dim + d]);
    double r2 = (double)Dsv4ToFloat(resRow[(uint64_t)2 * dim + d]);
    double r3 = (double)Dsv4ToFloat(resRow[(uint64_t)3 * dim + d]);
    double xv = (double)Dsv4ToFloat(x[(uint64_t)t * dim + d]);
    const float *postRow = post + (uint64_t)t * 4;
    const float *combRow = comb + (uint64_t)t * 16;
    OT *outRow = output + (uint64_t)t * 4 * dim + d;

    double v0 = (double)postRow[0] * xv +
                (double)combRow[0] * r0 + (double)combRow[4] * r1 +
                (double)combRow[8] * r2 + (double)combRow[12] * r3;
    double v1 = (double)postRow[1] * xv +
                (double)combRow[1] * r0 + (double)combRow[5] * r1 +
                (double)combRow[9] * r2 + (double)combRow[13] * r3;
    double v2 = (double)postRow[2] * xv +
                (double)combRow[2] * r0 + (double)combRow[6] * r1 +
                (double)combRow[10] * r2 + (double)combRow[14] * r3;
    double v3 = (double)postRow[3] * xv +
                (double)combRow[3] * r0 + (double)combRow[7] * r1 +
                (double)combRow[11] * r2 + (double)combRow[15] * r3;
    outRow[0] = Dsv4FromFloat<OT>((float)v0);
    outRow[(uint64_t)dim] = Dsv4FromFloat<OT>((float)v1);
    outRow[(uint64_t)2 * dim] = Dsv4FromFloat<OT>((float)v2);
    outRow[(uint64_t)3 * dim] = Dsv4FromFloat<OT>((float)v3);
}

// Decode specialization matching vLLM's small-FMA mHC transition kernel.
// It combines the previous block's post mapping with the next block's 24
// pre-norm projections.  Keeping newResidual in registers avoids writing and
// re-reading 4 * 4096 BF16 values for every projection.  The projection and
// square-sum intentionally consume the unrounded FP32 value, while the value
// used by the following pre-mix is explicitly written through BF16.  This is
// the same numerical boundary used by vLLM's mhc_fused_tilelang kernel.
__global__ void DeepSeekV4HcPostPreDots4x4096Kernel(
        const __nv_bfloat16 *x, const __nv_bfloat16 *residual,
        const float *post, const float *comb, const float *nextFn,
        float *dots, __nv_bfloat16 *residualOutput, int tokens) {
    constexpr int hcMult = 4;
    constexpr int dim = 4096;
    constexpr int flatDim = hcMult * dim;
    constexpr int mixHc = (2 + hcMult) * hcMult;
    constexpr int dotsStride = mixHc + 1;
    constexpr int tileN = 2;
    constexpr int splitK = 8;
    constexpr int hiddenPerSplit = dim / splitK;
    constexpr int numWarps = 8;

    __shared__ float warpSums[numWarps][tileN + 1];

    int tile = blockIdx.x;
    int part = blockIdx.y;
    int token = blockIdx.z;
    if (token >= tokens || tile >= mixHc / tileN || part >= splitK) {
        return;
    }

    const __nv_bfloat16 *xRow = x + (uint64_t)token * dim;
    const __nv_bfloat16 *residualRow =
        residual + (uint64_t)token * flatDim;
    __nv_bfloat16 *outputRow =
        residualOutput + (uint64_t)token * flatDim;
    const float *postRow = post + (uint64_t)token * hcMult;
    const float *combRow = comb + (uint64_t)token * hcMult * hcMult;

    int n0 = tile * tileN;
    const float *weight0 = nextFn + (uint64_t)n0 * flatDim;
    const float *weight1 = weight0 + flatDim;
    int hiddenStart = part * hiddenPerSplit;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float square = 0.0f;

#pragma unroll
    for (int iteration = 0; iteration < hiddenPerSplit / 256; iteration++) {
        int d = hiddenStart + iteration * 256 + threadIdx.x;
        float xv = __bfloat162float(xRow[d]);
        float r0 = __bfloat162float(residualRow[d]);
        float r1 = __bfloat162float(residualRow[dim + d]);
        float r2 = __bfloat162float(residualRow[2 * dim + d]);
        float r3 = __bfloat162float(residualRow[3 * dim + d]);

        float new0 = postRow[0] * xv;
        float new1 = postRow[1] * xv;
        float new2 = postRow[2] * xv;
        float new3 = postRow[3] * xv;
        new0 += combRow[0] * r0 + combRow[4] * r1 +
                combRow[8] * r2 + combRow[12] * r3;
        new1 += combRow[1] * r0 + combRow[5] * r1 +
                combRow[9] * r2 + combRow[13] * r3;
        new2 += combRow[2] * r0 + combRow[6] * r1 +
                combRow[10] * r2 + combRow[14] * r3;
        new3 += combRow[3] * r0 + combRow[7] * r1 +
                combRow[11] * r2 + combRow[15] * r3;

        // Every n-tile computes the transition in registers, but only tile 0
        // owns the residual and square-sum output to avoid write races.
        if (tile == 0) {
            outputRow[d] = __float2bfloat16_rn(new0);
            outputRow[dim + d] = __float2bfloat16_rn(new1);
            outputRow[2 * dim + d] = __float2bfloat16_rn(new2);
            outputRow[3 * dim + d] = __float2bfloat16_rn(new3);
            square += new0 * new0 + new1 * new1 +
                      new2 * new2 + new3 * new3;
        }

        acc0 += weight0[d] * new0 + weight0[dim + d] * new1 +
                weight0[2 * dim + d] * new2 +
                weight0[3 * dim + d] * new3;
        acc1 += weight1[d] * new0 + weight1[dim + d] * new1 +
                weight1[2 * dim + d] * new2 +
                weight1[3 * dim + d] * new3;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xffffffff, acc0, offset);
        acc1 += __shfl_down_sync(0xffffffff, acc1, offset);
        if (tile == 0) {
            square += __shfl_down_sync(0xffffffff, square, offset);
        }
    }
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (lane == 0) {
        warpSums[warp][0] = acc0;
        warpSums[warp][1] = acc1;
        if (tile == 0) {
            warpSums[warp][2] = square;
        }
    }
    __syncthreads();

    if (warp == 0) {
        float value0 = lane < numWarps ? warpSums[lane][0] : 0.0f;
        float value1 = lane < numWarps ? warpSums[lane][1] : 0.0f;
        float squareValue =
            tile == 0 && lane < numWarps ? warpSums[lane][2] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            value0 += __shfl_down_sync(0xffffffff, value0, offset);
            value1 += __shfl_down_sync(0xffffffff, value1, offset);
            if (tile == 0) {
                squareValue +=
                    __shfl_down_sync(0xffffffff, squareValue, offset);
            }
        }
        if (lane == 0) {
            dots[((uint64_t)token * dotsStride + n0) * splitK + part] =
                value0;
            dots[((uint64_t)token * dotsStride + n0 + 1) * splitK + part] =
                value1;
            if (tile == 0) {
                dots[((uint64_t)token * dotsStride + mixHc) * splitK + part] =
                    squareValue;
            }
        }
    }
}

bool DeepSeekV4PrepareCudaOutput(fastllm::Data &output, fastllm::DataType dataType,
                                 const std::vector<int> &dims) {
    output.dataType = dataType;
    output.Resize(dims);
    // MultiCuda delegates this CUDA operator from one worker per device.  The
    // generic ToDevice(CUDA) overload resolves the executor's default CUDA
    // device (normally device 0), which would switch non-root workers back to
    // GPU 0 before launching the kernel.  Keep temporary/output tensors on the
    // worker's current CUDA device instead.
    output.ToDevice(fastllm::DataDevice::CUDA, {FastllmCudaGetDevice()}, false);
    output.Allocate(false);
    return output.cudaData != nullptr;
}

bool DeepSeekV4CublasDataType(fastllm::DataType dataType, cudaDataType_t &cudaType) {
    if (dataType == fastllm::DataType::BFLOAT16) {
        cudaType = CUDA_R_16BF;
        return true;
    }
    if (dataType == fastllm::DataType::FLOAT16) {
        cudaType = CUDA_R_16F;
        return true;
    }
    if (dataType == fastllm::DataType::FLOAT32) {
        cudaType = CUDA_R_32F;
        return true;
    }
    return false;
}

cublasStatus_t DeepSeekV4CublasSgemmStrict(
        cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k, const float *alpha,
        const float *A, int lda, const float *B, int ldb,
        const float *beta, float *C, int ldc) {
#if CUBLAS_VERSION >= 11000
    return cublasGemmEx(
        handle, transa, transb, m, n, k,
        alpha,
        A, CUDA_R_32F, lda,
        B, CUDA_R_32F, ldb,
        beta,
        C, CUDA_R_32F, ldc,
        CUBLAS_COMPUTE_32F_PEDANTIC,
        CUBLAS_GEMM_DEFAULT);
#else
    return cublasSgemm(
        handle, transa, transb, m, n, k,
        alpha, A, lda, B, ldb, beta, C, ldc);
#endif
}

cublasStatus_t DeepSeekV4CublasBf16GemmToFloat(
        cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k, const float *alpha,
        const __nv_bfloat16 *A, int lda, const __nv_bfloat16 *B, int ldb,
        const float *beta, float *C, int ldc) {
#if CUBLAS_VERSION >= 11000
    return cublasGemmEx(
        handle, transa, transb, m, n, k,
        alpha,
        A, CUDA_R_16BF, lda,
        B, CUDA_R_16BF, ldb,
        beta,
        C, CUDA_R_32F, ldc,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
#else
    return CUBLAS_STATUS_NOT_SUPPORTED;
#endif
}

bool DeepSeekV4LaunchWoAGemm(const fastllm::Data &o, const fastllm::Data &woA,
                             fastllm::Data &output, int bsz, int seqlen,
                             int heads, int headDim, int groups, int oRank) {
    int tokens = bsz * seqlen;
    if (tokens < 16) {
        return false;
    }
    cudaDataType_t oType, wType;
    if (!DeepSeekV4CublasDataType(o.dataType, oType) ||
        !DeepSeekV4CublasDataType(woA.dataType, wType) ||
        output.dataType != fastllm::DataType::BFLOAT16 ||
        heads % groups != 0) {
        return false;
    }

    int headsPerGroup = heads / groups;
    int groupDim = headsPerGroup * headDim;
    int fullDim = heads * headDim;
    float alpha = 1.0f, beta = 0.0f;
    auto handle = getFastllmCublasHandle();

    if (o.dataType == fastllm::DataType::BFLOAT16 && woA.dataType == fastllm::DataType::FLOAT16) {
        half *halfO = (half *)FastllmCudaMalloc(o.Count(0) * sizeof(half));
        half *halfOut = (half *)FastllmCudaMalloc(output.Count(0) * sizeof(half));
        if (halfO != nullptr && halfOut != nullptr) {
            FastllmBF16ToHalf(o.cudaData, halfO, (int)o.Count(0));
            half hAlpha = __float2half(1.0f), hBeta = __float2half(0.0f);
            cublasStatus_t halfStatus = cublasHgemmStridedBatched(
                handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                oRank, tokens, groupDim,
                &hAlpha,
                (const half *)woA.cudaData, groupDim, (long long)oRank * groupDim,
                halfO, fullDim, groupDim,
                &hBeta,
                halfOut, groups * oRank, oRank,
                groups);
            if (halfStatus == CUBLAS_STATUS_SUCCESS) {
                FastllmHalfToBF16(halfOut, output.cudaData, (int)output.Count(0));
                FastllmCudaFree(halfO);
                FastllmCudaFree(halfOut);
                return true;
            }
            if (std::getenv("FASTLLM_DSV4_DEBUG_CUDA_WOA_GEMM") != nullptr) {
                printf("DeepSeekV4WoA half cuBLAS fallback: status=%d tokens=%d groupDim=%d groups=%d oRank=%d\n",
                       (int)halfStatus, tokens, groupDim, groups, oRank);
            }
        }
        if (halfO != nullptr) {
            FastllmCudaFree(halfO);
        }
        if (halfOut != nullptr) {
            FastllmCudaFree(halfOut);
        }
        return false;
    }

    cublasStatus_t status = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        oRank, tokens, groupDim,
        &alpha,
        woA.cudaData, wType, groupDim, (long long)oRank * groupDim,
        o.cudaData, oType, fullDim, groupDim,
        &beta,
        output.cudaData, CUDA_R_16BF, groups * oRank, oRank,
        groups,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status == CUBLAS_STATUS_SUCCESS) {
        return true;
    }

    if (status != CUBLAS_STATUS_SUCCESS && std::getenv("FASTLLM_DSV4_DEBUG_CUDA_WOA_GEMM") != nullptr) {
        printf("DeepSeekV4WoA cuBLAS fallback: status=%d oType=%d wType=%d tokens=%d groupDim=%d groups=%d oRank=%d\n",
               (int)status, (int)o.dataType, (int)woA.dataType, tokens, groupDim, groups, oRank);
    }
    return false;
}

template <typename InT>
bool DeepSeekV4LaunchWoAByWeight(const fastllm::Data &o, const fastllm::Data &woA, int groups,
                                 int oRank, fastllm::Data &output, int bsz, int seqlen,
                                 int heads, int headDim) {
    if (std::getenv("FASTLLM_DSV4_DISABLE_CUDA_WOA_GEMM") == nullptr &&
        DeepSeekV4LaunchWoAGemm(o, woA, output, bsz, seqlen, heads, headDim, groups, oRank)) {
        return true;
    }
    bool usePair = std::getenv("FASTLLM_DSV4_ENABLE_CUDA_WOA_PAIR") != nullptr && (oRank % 2 == 0);
    bool useFloatAcc = !usePair && std::getenv("FASTLLM_DSV4_ENABLE_CUDA_WOA_FLOAT_ACC") != nullptr;
    bool useKahanAcc = !usePair && !useFloatAcc && std::getenv("FASTLLM_DSV4_ENABLE_CUDA_WOA_KAHAN_ACC") != nullptr;
    bool useBlockReduce = !usePair && !useFloatAcc && !useKahanAcc &&
                          std::getenv("FASTLLM_DSV4_DISABLE_CUDA_WOA_BLOCK") == nullptr;
    bool usePairBlockReduce = useBlockReduce && seqlen == 1 && (oRank % 2 == 0) &&
                              std::getenv("FASTLLM_DSV4_DISABLE_CUDA_WOA_PAIR_BLOCK") == nullptr;
    int total = bsz * seqlen * groups * oRank;
    int threads = std::min(256, std::max(1, total));
    int pairTotal = bsz * seqlen * groups * (oRank / 2);
    int launchTotal = (usePair || usePairBlockReduce) ? pairTotal : total;
    int blocks = (launchTotal + threads - 1) / threads;
    const InT *oData = (const InT *)o.cudaData;
    __nv_bfloat16 *outData = (__nv_bfloat16 *)output.cudaData;

    if (woA.dataType == fastllm::DataType::FP8_E4M3) {
        int headsPerGroup = heads / groups;
        int groupDim = headsPerGroup * headDim;
        int weightRows = groups * oRank;
        int scaleRows = woA.blockK > 0 ? (weightRows + woA.blockK - 1) / woA.blockK : 0;
        int scaleCols = woA.blockM > 0 ? (groupDim + woA.blockM - 1) / woA.blockM : 0;
        if ((oRank & 1) != 0 || woA.blockK <= 0 || woA.blockM <= 0 ||
            woA.scales.size() != (size_t)scaleRows * scaleCols) {
            return false;
        }
        fastllm::Data &mutableWeight = const_cast<fastllm::Data&>(woA);
        fastllm::Data emptyBias;
        FastllmCudaFP8E4M3EnsureScalesAndBiasOnDevice(mutableWeight, emptyBias, weightRows);
        if (mutableWeight.extraCudaData.empty() || mutableWeight.extraCudaData[0] == nullptr) {
            return false;
        }
        DeepSeekV4WoAFp8PairBlockReduceKernel<<<pairTotal, 256, 512 * sizeof(float)>>>(
            oData, (const uint8_t*)woA.cudaData,
            (const float*)mutableWeight.extraCudaData[0], outData,
            bsz, seqlen, heads, headDim, groups, oRank,
            woA.blockK, woA.blockM, scaleCols);
        return true;
    }

    if (woA.dataType == fastllm::DataType::BFLOAT16) {
        if (usePair) {
            DeepSeekV4WoAPairKernel<<<blocks, threads>>>(oData, (const __nv_bfloat16 *)woA.cudaData, outData,
                                                         bsz, seqlen, heads, headDim, groups, oRank);
        } else if (usePairBlockReduce) {
            DeepSeekV4WoAPairBlockReduceKernel<<<pairTotal, 256, 512 * sizeof(float)>>>(
                oData, (const __nv_bfloat16 *)woA.cudaData, outData,
                bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useBlockReduce) {
            DeepSeekV4WoABlockReduceKernel<<<total, 256, 256 * sizeof(float)>>>(
                oData, (const __nv_bfloat16 *)woA.cudaData, outData,
                bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useFloatAcc) {
            DeepSeekV4WoAFloatAccKernel<<<blocks, threads>>>(oData, (const __nv_bfloat16 *)woA.cudaData, outData,
                                                             bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useKahanAcc) {
            DeepSeekV4WoAKahanAccKernel<<<blocks, threads>>>(oData, (const __nv_bfloat16 *)woA.cudaData, outData,
                                                             bsz, seqlen, heads, headDim, groups, oRank);
        } else {
            DeepSeekV4WoAKernel<<<blocks, threads>>>(oData, (const __nv_bfloat16 *)woA.cudaData, outData,
                                                     bsz, seqlen, heads, headDim, groups, oRank);
        }
    } else if (woA.dataType == fastllm::DataType::FLOAT16) {
        if (usePair) {
            DeepSeekV4WoAPairKernel<<<blocks, threads>>>(oData, (const half *)woA.cudaData, outData,
                                                         bsz, seqlen, heads, headDim, groups, oRank);
        } else if (usePairBlockReduce) {
            DeepSeekV4WoAPairBlockReduceKernel<<<pairTotal, 256, 512 * sizeof(float)>>>(
                oData, (const half *)woA.cudaData, outData,
                bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useBlockReduce) {
            DeepSeekV4WoABlockReduceKernel<<<total, 256, 256 * sizeof(float)>>>(
                oData, (const half *)woA.cudaData, outData,
                bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useFloatAcc) {
            DeepSeekV4WoAFloatAccKernel<<<blocks, threads>>>(oData, (const half *)woA.cudaData, outData,
                                                             bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useKahanAcc) {
            DeepSeekV4WoAKahanAccKernel<<<blocks, threads>>>(oData, (const half *)woA.cudaData, outData,
                                                             bsz, seqlen, heads, headDim, groups, oRank);
        } else {
            DeepSeekV4WoAKernel<<<blocks, threads>>>(oData, (const half *)woA.cudaData, outData,
                                                     bsz, seqlen, heads, headDim, groups, oRank);
        }
    } else if (woA.dataType == fastllm::DataType::FLOAT32) {
        if (usePair) {
            DeepSeekV4WoAPairKernel<<<blocks, threads>>>(oData, (const float *)woA.cudaData, outData,
                                                         bsz, seqlen, heads, headDim, groups, oRank);
        } else if (usePairBlockReduce) {
            DeepSeekV4WoAPairBlockReduceKernel<<<pairTotal, 256, 512 * sizeof(float)>>>(
                oData, (const float *)woA.cudaData, outData,
                bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useBlockReduce) {
            DeepSeekV4WoABlockReduceKernel<<<total, 256, 256 * sizeof(float)>>>(
                oData, (const float *)woA.cudaData, outData,
                bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useFloatAcc) {
            DeepSeekV4WoAFloatAccKernel<<<blocks, threads>>>(oData, (const float *)woA.cudaData, outData,
                                                             bsz, seqlen, heads, headDim, groups, oRank);
        } else if (useKahanAcc) {
            DeepSeekV4WoAKahanAccKernel<<<blocks, threads>>>(oData, (const float *)woA.cudaData, outData,
                                                             bsz, seqlen, heads, headDim, groups, oRank);
        } else {
            DeepSeekV4WoAKernel<<<blocks, threads>>>(oData, (const float *)woA.cudaData, outData,
                                                     bsz, seqlen, heads, headDim, groups, oRank);
        }
    } else {
        return false;
    }
    return true;
}

template <typename XT, typename RT>
bool DeepSeekV4LaunchHcPostByOutput(const fastllm::Data &x, const fastllm::Data &residual,
                                    const float *cudaPost, const float *cudaComb,
                                    fastllm::Data &output, int tokens, int hcMult, int dim) {
    int threads = 256;
    const XT *xData = (const XT *)x.cudaData;
    const RT *resData = (const RT *)residual.cudaData;

    int hcPost4MinTokens = 16384;
    if (const char *env = std::getenv("FASTLLM_DSV4_HCPOST4_MIN_TOKENS")) {
        hcPost4MinTokens = std::max(1, std::atoi(env));
    }
    bool useHcPost4 = hcMult == 4 &&
                      std::getenv("FASTLLM_DSV4_DISABLE_CUDA_HCPOST4") == nullptr &&
                      (std::getenv("FASTLLM_DSV4_ENABLE_CUDA_HCPOST4") != nullptr ||
                       tokens >= hcPost4MinTokens);
    if (useHcPost4) {
        uint64_t total = (uint64_t)tokens * dim;
        int blocks = (int)((total + threads - 1) / threads);
        if (output.dataType == fastllm::DataType::BFLOAT16) {
            DeepSeekV4HcPost4Kernel<<<blocks, threads>>>(xData, resData, cudaPost, cudaComb,
                                                         (__nv_bfloat16 *)output.cudaData, tokens, dim);
            return true;
        } else if (output.dataType == fastllm::DataType::FLOAT16) {
            DeepSeekV4HcPost4Kernel<<<blocks, threads>>>(xData, resData, cudaPost, cudaComb,
                                                         (half *)output.cudaData, tokens, dim);
            return true;
        } else if (output.dataType == fastllm::DataType::FLOAT32) {
            DeepSeekV4HcPost4Kernel<<<blocks, threads>>>(xData, resData, cudaPost, cudaComb,
                                                         (float *)output.cudaData, tokens, dim);
            return true;
        }
    }

    uint64_t total = (uint64_t)tokens * hcMult * dim;
    int blocks = (int)((total + threads - 1) / threads);
    if (output.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4HcPostKernel<<<blocks, threads>>>(xData, resData, cudaPost, cudaComb,
                                                    (__nv_bfloat16 *)output.cudaData, tokens, hcMult, dim);
    } else if (output.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4HcPostKernel<<<blocks, threads>>>(xData, resData, cudaPost, cudaComb,
                                                    (half *)output.cudaData, tokens, hcMult, dim);
    } else if (output.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4HcPostKernel<<<blocks, threads>>>(xData, resData, cudaPost, cudaComb,
                                                    (float *)output.cudaData, tokens, hcMult, dim);
    } else {
        return false;
    }
    return true;
}

template <typename XT>
bool DeepSeekV4LaunchHcPostByResidual(const fastllm::Data &x, const fastllm::Data &residual,
                                      const float *cudaPost, const float *cudaComb,
                                      fastllm::Data &output, int tokens, int hcMult, int dim) {
    if (residual.dataType == fastllm::DataType::BFLOAT16) {
        return DeepSeekV4LaunchHcPostByOutput<XT, __nv_bfloat16>(x, residual, cudaPost, cudaComb,
                                                                output, tokens, hcMult, dim);
    }
    if (residual.dataType == fastllm::DataType::FLOAT16) {
        return DeepSeekV4LaunchHcPostByOutput<XT, half>(x, residual, cudaPost, cudaComb,
                                                       output, tokens, hcMult, dim);
    }
    if (residual.dataType == fastllm::DataType::FLOAT32) {
        return DeepSeekV4LaunchHcPostByOutput<XT, float>(x, residual, cudaPost, cudaComb,
                                                        output, tokens, hcMult, dim);
    }
    return false;
}

template <typename QT>
bool DeepSeekV4LaunchSparseDecodeCublas(const void *cudaQ, const void *cudaCompressed,
                                        fastllm::DataType compressedType, const float *cudaWindow,
                                        const float *cudaSink, float *outData,
                                        int bsz, int heads, int dim, int windowSize,
                                        int startPos, int compressedCount, int liveWindow,
                                        int keyCount, float softmaxScale) {
    if (std::getenv("FASTLLM_DSV4_DISABLE_CUBLAS_SPARSE_DECODE") != nullptr) {
        return false;
    }
    // A TP shard can contain only a handful of heads.  For small decode
    // problems, converting/gathering into FP32 and launching two GEMMs costs
    // more than the fused online kernel.  Keep cuBLAS for sufficiently large
    // head-key products, where reusing the gathered KV starts to pay off.
    constexpr int minCublasWork = 1024;
    if ((uint64_t)bsz * heads * keyCount < (uint64_t)minCublasWork) {
        return false;
    }
    int maxCublasKeys = kDeepSeekV4SparseDecodeMaxKeys;
    if (const char *env = std::getenv("FASTLLM_DSV4_CUBLAS_SPARSE_DECODE_MAX_KEYS")) {
        maxCublasKeys = std::max(1, std::atoi(env));
    }
    if (keyCount <= 0 || keyCount > maxCublasKeys || dim <= 0 || heads <= 0 || bsz <= 0) {
        return false;
    }

    uint64_t qElems = (uint64_t)bsz * heads * dim;
    uint64_t kvElems = (uint64_t)bsz * keyCount * dim;
    uint64_t scoreElems = (uint64_t)bsz * heads * keyCount;
    float *qFloat = (float *)FastllmCudaMalloc(qElems * sizeof(float));
    float *kvFloat = (float *)FastllmCudaMalloc(kvElems * sizeof(float));
    float *scores = (float *)FastllmCudaMalloc(scoreElems * sizeof(float));
    if (qFloat == nullptr || kvFloat == nullptr || scores == nullptr) {
        if (qFloat != nullptr) {
            FastllmCudaFree(qFloat);
        }
        if (kvFloat != nullptr) {
            FastllmCudaFree(kvFloat);
        }
        if (scores != nullptr) {
            FastllmCudaFree(scores);
        }
        return false;
    }

    int threads = 256;
    DeepSeekV4SparseDecodeConvertQKernel<QT><<<
        (int)((qElems + threads - 1) / threads), threads>>>((const QT *)cudaQ, qFloat, qElems);

    bool gathered = true;
    if (compressedCount == 0 || cudaCompressed == nullptr) {
        DeepSeekV4SparseDecodeGatherKVKernel<float><<<
            (int)((kvElems + threads - 1) / threads), threads>>>(
                cudaWindow, (const float *)nullptr, kvFloat, bsz, dim, windowSize,
                startPos, 0, liveWindow, keyCount);
    } else if (compressedType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4SparseDecodeGatherKVKernel<__nv_bfloat16><<<
            (int)((kvElems + threads - 1) / threads), threads>>>(
                cudaWindow, (const __nv_bfloat16 *)cudaCompressed, kvFloat, bsz, dim, windowSize,
                startPos, compressedCount, liveWindow, keyCount);
    } else if (compressedType == fastllm::DataType::FLOAT16) {
        DeepSeekV4SparseDecodeGatherKVKernel<half><<<
            (int)((kvElems + threads - 1) / threads), threads>>>(
                cudaWindow, (const half *)cudaCompressed, kvFloat, bsz, dim, windowSize,
                startPos, compressedCount, liveWindow, keyCount);
    } else if (compressedType == fastllm::DataType::FLOAT32) {
        DeepSeekV4SparseDecodeGatherKVKernel<float><<<
            (int)((kvElems + threads - 1) / threads), threads>>>(
                cudaWindow, (const float *)cudaCompressed, kvFloat, bsz, dim, windowSize,
                startPos, compressedCount, liveWindow, keyCount);
    } else {
        gathered = false;
    }

    bool ok = false;
    if (gathered) {
        float alpha = softmaxScale;
        float beta = 0.0f;
        auto handle = getFastllmCublasHandle();
        cublasStatus_t status = cublasSgemmStridedBatched(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            keyCount, heads, dim,
            &alpha,
            kvFloat, dim, (long long)keyCount * dim,
            qFloat, dim, (long long)heads * dim,
            &beta,
            scores, keyCount, (long long)heads * keyCount,
            bsz);

        if (status == CUBLAS_STATUS_SUCCESS) {
            DeepSeekV4SparseDecodeSoftmaxSinkKernel<<<bsz * heads, 256, 256 * sizeof(float)>>>(
                scores, cudaSink, bsz, heads, keyCount);

            alpha = 1.0f;
            status = cublasSgemmStridedBatched(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                dim, heads, keyCount,
                &alpha,
                kvFloat, dim, (long long)keyCount * dim,
                scores, keyCount, (long long)heads * keyCount,
                &beta,
                outData, dim, (long long)heads * dim,
                bsz);
        }
        ok = status == CUBLAS_STATUS_SUCCESS;
        if (!ok && std::getenv("FASTLLM_DSV4_DEBUG_CUBLAS_SPARSE_DECODE") != nullptr) {
            std::fprintf(stderr,
                         "DeepSeekV4SparseDecode cuBLAS fallback: status=%d bsz=%d heads=%d dim=%d keys=%d compressed=%d\n",
                         (int)status, bsz, heads, dim, keyCount, compressedCount);
        }
    }

    FastllmCudaFree(qFloat);
    FastllmCudaFree(kvFloat);
    FastllmCudaFree(scores);
    return ok;
}

template <typename QT>
bool DeepSeekV4LaunchSparseDecodeByCompressed(const void *cudaQ, const void *cudaCompressed,
                                              fastllm::DataType compressedType, const float *cudaWindow,
                                              const float *cudaSink, float *outData,
                                              int bsz, int heads, int dim, int windowSize,
                                              int startPos, int compressedCount, float softmaxScale,
                                              int *launchMode = nullptr) {
    const QT *qData = (const QT *)cudaQ;
    int blocks = bsz * heads;
    int liveWindow = startPos >= windowSize - 1 ? windowSize : (startPos + 1);
    int maxKeys = liveWindow + compressedCount;
    if (maxKeys <= 0 || maxKeys > kDeepSeekV4SparseDecodeMaxKeys) {
        return false;
    }
    if (launchMode != nullptr) {
        *launchMode = 0;
    }
    if (DeepSeekV4LaunchSparseDecodeCublas<QT>(
            cudaQ, cudaCompressed, compressedType, cudaWindow, cudaSink, outData,
            bsz, heads, dim, windowSize, startPos, compressedCount, liveWindow,
            maxKeys, softmaxScale)) {
        if (launchMode != nullptr) {
            *launchMode = 2;
        }
        return true;
    }
    bool useOnlineDecode = dim <= 256 * 4 &&
                           std::getenv("FASTLLM_DSV4_DISABLE_ONLINE_SPARSE_DECODE") == nullptr;
    if (useOnlineDecode) {
        if (compressedCount == 0 || cudaCompressed == nullptr) {
            DeepSeekV4SparseAttentionDecodeCachedOnlineKernel<QT, float><<<blocks, 256>>>(
                qData, cudaWindow, (const float *)nullptr, cudaSink, outData,
                bsz, heads, dim, windowSize, startPos, 0, softmaxScale, nullptr, 0);
        } else if (compressedType == fastllm::DataType::BFLOAT16) {
            DeepSeekV4SparseAttentionDecodeCachedOnlineKernel<QT, __nv_bfloat16><<<blocks, 256>>>(
                qData, cudaWindow, (const __nv_bfloat16 *)cudaCompressed, cudaSink, outData,
                bsz, heads, dim, windowSize, startPos, compressedCount, softmaxScale, nullptr, 0);
        } else if (compressedType == fastllm::DataType::FLOAT16) {
            DeepSeekV4SparseAttentionDecodeCachedOnlineKernel<QT, half><<<blocks, 256>>>(
                qData, cudaWindow, (const half *)cudaCompressed, cudaSink, outData,
                bsz, heads, dim, windowSize, startPos, compressedCount, softmaxScale, nullptr, 0);
        } else if (compressedType == fastllm::DataType::FLOAT32) {
            DeepSeekV4SparseAttentionDecodeCachedOnlineKernel<QT, float><<<blocks, 256>>>(
                qData, cudaWindow, (const float *)cudaCompressed, cudaSink, outData,
                bsz, heads, dim, windowSize, startPos, compressedCount, softmaxScale, nullptr, 0);
        } else {
            return false;
        }
        if (launchMode != nullptr) {
            *launchMode = 1;
        }
        return true;
    }
    size_t sharedBytes = ((size_t)maxKeys + 256) * sizeof(float);
    if (compressedCount == 0 || cudaCompressed == nullptr) {
        auto kernel = DeepSeekV4SparseAttentionDecodeCachedBlockKernel<QT, float>;
        if (!DeepSeekV4EnsureDynamicSharedMemory(kernel, sharedBytes)) {
            return false;
        }
        kernel<<<blocks, 256, sharedBytes>>>(
            qData, cudaWindow, (const float *)nullptr, cudaSink, outData,
            bsz, heads, dim, windowSize, startPos, 0, maxKeys, softmaxScale);
    } else if (compressedType == fastllm::DataType::BFLOAT16) {
        auto kernel = DeepSeekV4SparseAttentionDecodeCachedBlockKernel<QT, __nv_bfloat16>;
        if (!DeepSeekV4EnsureDynamicSharedMemory(kernel, sharedBytes)) {
            return false;
        }
        kernel<<<blocks, 256, sharedBytes>>>(
            qData, cudaWindow, (const __nv_bfloat16 *)cudaCompressed, cudaSink, outData,
            bsz, heads, dim, windowSize, startPos, compressedCount, maxKeys, softmaxScale);
    } else if (compressedType == fastllm::DataType::FLOAT16) {
        auto kernel = DeepSeekV4SparseAttentionDecodeCachedBlockKernel<QT, half>;
        if (!DeepSeekV4EnsureDynamicSharedMemory(kernel, sharedBytes)) {
            return false;
        }
        kernel<<<blocks, 256, sharedBytes>>>(
            qData, cudaWindow, (const half *)cudaCompressed, cudaSink, outData,
            bsz, heads, dim, windowSize, startPos, compressedCount, maxKeys, softmaxScale);
    } else if (compressedType == fastllm::DataType::FLOAT32) {
        auto kernel = DeepSeekV4SparseAttentionDecodeCachedBlockKernel<QT, float>;
        if (!DeepSeekV4EnsureDynamicSharedMemory(kernel, sharedBytes)) {
            return false;
        }
        kernel<<<blocks, 256, sharedBytes>>>(
            qData, cudaWindow, (const float *)cudaCompressed, cudaSink, outData,
            bsz, heads, dim, windowSize, startPos, compressedCount, maxKeys, softmaxScale);
    } else {
        return false;
    }
    return true;
}

template <typename QT, typename CT>
bool DeepSeekV4LaunchSparseDecodeBatchOnline(const void * const *cudaQPtrs,
                                             const float * const *cudaWindowPtrs,
                                             const void * const *cudaCompressedPtrs,
                                             const float *cudaSink,
                                             const int *cudaStartPositions,
                                             const int *cudaCompressedCounts,
                                             float *outData, int batch, int heads,
                                             int dim, int windowSize,
                                             float softmaxScale) {
    if (dim <= 0 || dim > 256 * 4 || batch <= 0 || heads <= 0 || windowSize <= 0) {
        return false;
    }
    DeepSeekV4SparseAttentionDecodeCachedBatchOnlineKernel<QT, CT><<<batch * heads, 256>>>(
        cudaQPtrs, cudaWindowPtrs, cudaCompressedPtrs, cudaSink,
        cudaStartPositions, cudaCompressedCounts, outData,
        batch, heads, dim, windowSize, softmaxScale);
    return true;
}

template <typename QT>
bool DeepSeekV4LaunchSparseDecodeBatchByCompressed(const void * const *cudaQPtrs,
                                                   const float * const *cudaWindowPtrs,
                                                   const void * const *cudaCompressedPtrs,
                                                   fastllm::DataType compressedType,
                                                   const float *cudaSink,
                                                   const int *cudaStartPositions,
                                                   const int *cudaCompressedCounts,
                                                   float *outData, int batch, int heads,
                                                   int dim, int windowSize,
                                                   float softmaxScale) {
    if (compressedType == fastllm::DataType::BFLOAT16) {
        return DeepSeekV4LaunchSparseDecodeBatchOnline<QT, __nv_bfloat16>(
            cudaQPtrs, cudaWindowPtrs, cudaCompressedPtrs, cudaSink,
            cudaStartPositions, cudaCompressedCounts, outData,
            batch, heads, dim, windowSize, softmaxScale);
    }
    if (compressedType == fastllm::DataType::FLOAT16) {
        return DeepSeekV4LaunchSparseDecodeBatchOnline<QT, half>(
            cudaQPtrs, cudaWindowPtrs, cudaCompressedPtrs, cudaSink,
            cudaStartPositions, cudaCompressedCounts, outData,
            batch, heads, dim, windowSize, softmaxScale);
    }
    if (compressedType == fastllm::DataType::FLOAT32) {
        return DeepSeekV4LaunchSparseDecodeBatchOnline<QT, float>(
            cudaQPtrs, cudaWindowPtrs, cudaCompressedPtrs, cudaSink,
            cudaStartPositions, cudaCompressedCounts, outData,
            batch, heads, dim, windowSize, softmaxScale);
    }
    return false;
}

template <typename QT, typename KT>
bool DeepSeekV4LaunchSparsePrefillCompressedCublasSegment(
        const fastllm::Data &q, const fastllm::Data &kv, const float *cudaSink,
        float *compressedScores, float *compressedKVFloat, float *outData,
        int bsz, int seqlen, int heads, int dim, int kvLen,
        int windowSize, int compressRatio, int startPos, int prefixLen,
        int compressedStart, int compressedCount, float softmaxScale,
        int rowOffset, int rows) {
    if (rows <= 0 || compressedCount <= 0 || q.dataType != kv.dataType) {
        return false;
    }
    cudaDataType_t inputType;
    if (!DeepSeekV4CublasDataType(q.dataType, inputType)) {
        return false;
    }

    int rowsPerBatch = seqlen * heads;
    int b = rowOffset / rowsPerBatch;
    if (b < 0 || b >= bsz || rowOffset + rows > (b + 1) * rowsPerBatch) {
        return false;
    }

    const QT *qPtr = (const QT *)q.cudaData + (uint64_t)rowOffset * dim;
    const KT *kvPtr = (const KT *)kv.cudaData + ((uint64_t)b * kvLen + compressedStart) * dim;
    float alpha = softmaxScale;
    float beta = 0.0f;
    auto handle = getFastllmCublasHandle();
    cublasStatus_t status;
    if (q.dataType == fastllm::DataType::FLOAT32) {
        status = cublasSgemm(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            compressedCount, rows, dim,
            &alpha,
            (const float *)kvPtr, dim,
            (const float *)qPtr, dim,
            &beta,
            compressedScores, compressedCount);
    } else {
        status = cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            compressedCount, rows, dim,
            &alpha,
            kvPtr, inputType, dim,
            qPtr, inputType, dim,
            &beta,
            compressedScores, CUDA_R_32F, compressedCount,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        if (std::getenv("FASTLLM_DSV4_DEBUG_CUBLAS_SPARSE_PREFILL") != nullptr) {
            std::fprintf(stderr,
                         "DeepSeekV4SparsePrefill compressed cuBLAS QK failed: status=%d rows=%d dim=%d compressed=%d type=%d\n",
                         (int)status, rows, dim, compressedCount, (int)q.dataType);
        }
        return false;
    }

    int realPrefixLen = std::max(0, std::min(prefixLen, kvLen - seqlen));
    int maxLocalKeys = std::min(windowSize, realPrefixLen + seqlen);
    if (maxLocalKeys <= 0 || maxLocalKeys > kDeepSeekV4SparsePrefillMaxKeys) {
        return false;
    }
    size_t sharedBytes = (size_t)maxLocalKeys * sizeof(float);
    auto kernel = DeepSeekV4SparseAttentionPrefillCublasCompressedKernel<QT, KT>;
    if (!DeepSeekV4EnsureDynamicSharedMemory(kernel, sharedBytes)) {
        return false;
    }
    kernel<<<rows, 256, sharedBytes>>>(
        (const QT *)q.cudaData, (const KT *)kv.cudaData, cudaSink, compressedScores, outData,
        bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos, realPrefixLen,
        compressedStart, compressedCount, softmaxScale, rowOffset);

    const float *compressedKVForGemm = nullptr;
    if (kv.dataType == fastllm::DataType::FLOAT32) {
        compressedKVForGemm = (const float *)kvPtr;
    } else {
        if (compressedKVFloat == nullptr) {
            return false;
        }
        uint64_t kvElems = (uint64_t)compressedCount * dim;
        int threads = 256;
        DeepSeekV4SparseDecodeGatherKVKernel<KT><<<
            (int)((kvElems + threads - 1) / threads), threads>>>(
                (const float *)nullptr, kvPtr, compressedKVFloat, 1, dim, 1,
                0, compressedCount, 0, compressedCount);
        compressedKVForGemm = compressedKVFloat;
    }

    float avAlpha = 1.0f;
    float avBeta = 1.0f;
    status = cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        dim, rows, compressedCount,
        &avAlpha,
        compressedKVForGemm, dim,
        compressedScores, compressedCount,
        &avBeta,
        outData, dim);
    if (status != CUBLAS_STATUS_SUCCESS) {
        if (std::getenv("FASTLLM_DSV4_DEBUG_CUBLAS_SPARSE_PREFILL") != nullptr) {
            std::fprintf(stderr,
                         "DeepSeekV4SparsePrefill compressed cuBLAS AV failed: status=%d rows=%d dim=%d compressed=%d type=%d\n",
                         (int)status, rows, dim, compressedCount, (int)kv.dataType);
        }
        return false;
    }
    return true;
}

bool DeepSeekV4RunSparsePrefillCompressedCublas(
        const fastllm::Data &q, const fastllm::Data &kv, const float *cudaSink,
        fastllm::Data &output,
        int bsz, int seqlen, int heads, int dim, int kvLen,
        int windowSize, int compressRatio, int startPos, int prefixLen,
        int ropeDim, float ropeBase, int originalSeqLen, float ropeFactor,
        int betaFast, int betaSlow, float softmaxScale) {
    if (std::getenv("FASTLLM_DSV4_ENABLE_CUBLAS_SPARSE_PREFILL_COMPRESSED") == nullptr) {
        return false;
    }
    if (q.dataType != kv.dataType || compressRatio <= 0 || windowSize <= 0 ||
        dim <= 0 || heads <= 0 || seqlen <= 0) {
        return false;
    }
    int realPrefixLen = std::max(0, std::min(prefixLen, kvLen - seqlen));
    int compressedStart = realPrefixLen + seqlen;
    int compressedCount = std::max(0, kvLen - compressedStart);
    if (compressedCount <= 0) {
        return false;
    }
    int maxCompressed = kDeepSeekV4Sparse1MCompressedKeys;
    if (const char *env = std::getenv("FASTLLM_DSV4_CUBLAS_SPARSE_PREFILL_MAX_COMPRESSED")) {
        maxCompressed = std::max(1, std::atoi(env));
    }
    if (compressedCount > maxCompressed) {
        return false;
    }

    int rows = bsz * seqlen * heads;
    size_t rowBytes = (size_t)dim * sizeof(float);
    size_t scoreRowBytes = (size_t)compressedCount * sizeof(float);
    size_t perRowBytes = rowBytes + scoreRowBytes;
    size_t maxTempBytes = std::max(perRowBytes, DeepSeekV4SparsePrefillTempBytesLimit());
    int rowsPerChunk = (int)std::max<size_t>(1, maxTempBytes / perRowBytes);
    rowsPerChunk = std::min(rowsPerChunk, rows);
    if (rowsPerChunk <= 0) {
        return false;
    }
    size_t tempBytes = (size_t)rowsPerChunk * rowBytes;
    size_t scoreBytes = (size_t)rowsPerChunk * scoreRowBytes;
    size_t kvFloatBytes = q.dataType == fastllm::DataType::FLOAT32 ?
        0 : (size_t)compressedCount * dim * sizeof(float);
    uint8_t *scratch = (uint8_t *)FastllmCudaMalloc(tempBytes + scoreBytes + kvFloatBytes);
    if (scratch == nullptr) {
        return false;
    }
    float *cudaTemp = (float *)scratch;
    float *compressedScores = (float *)(scratch + tempBytes);
    float *compressedKVFloat = kvFloatBytes == 0 ? nullptr : (float *)(scratch + tempBytes + scoreBytes);

    bool ok = true;
    int rowsPerBatch = seqlen * heads;
    for (int rowOffset = 0; rowOffset < rows && ok; rowOffset += rowsPerChunk) {
        int chunkRows = std::min(rowsPerChunk, rows - rowOffset);
        for (int localStart = 0; localStart < chunkRows && ok;) {
            int globalRow = rowOffset + localStart;
            int b = globalRow / rowsPerBatch;
            int batchEnd = std::min(rowOffset + chunkRows, (b + 1) * rowsPerBatch);
            int segmentRows = batchEnd - globalRow;
            float *segmentScores = compressedScores + (uint64_t)localStart * compressedCount;
            float *segmentTemp = cudaTemp + (uint64_t)localStart * dim;
            if (q.dataType == fastllm::DataType::BFLOAT16) {
                ok = DeepSeekV4LaunchSparsePrefillCompressedCublasSegment<__nv_bfloat16, __nv_bfloat16>(
                    q, kv, cudaSink, segmentScores, compressedKVFloat, segmentTemp,
                    bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos,
                    realPrefixLen, compressedStart, compressedCount, softmaxScale,
                    globalRow, segmentRows);
            } else if (q.dataType == fastllm::DataType::FLOAT16) {
                ok = DeepSeekV4LaunchSparsePrefillCompressedCublasSegment<half, half>(
                    q, kv, cudaSink, segmentScores, compressedKVFloat, segmentTemp,
                    bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos,
                    realPrefixLen, compressedStart, compressedCount, softmaxScale,
                    globalRow, segmentRows);
            } else if (q.dataType == fastllm::DataType::FLOAT32) {
                ok = DeepSeekV4LaunchSparsePrefillCompressedCublasSegment<float, float>(
                    q, kv, cudaSink, segmentScores, compressedKVFloat, segmentTemp,
                    bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos,
                    realPrefixLen, compressedStart, compressedCount, softmaxScale,
                    globalRow, segmentRows);
            } else {
                ok = false;
            }
            localStart += segmentRows;
        }
        if (!ok) {
            break;
        }
        int threads = 128;
        int blocks = (chunkRows + threads - 1) / threads;
        DeepSeekV4SparsePrefillRotaryCastKernel<<<blocks, threads>>>(
            cudaTemp, (__nv_bfloat16 *)output.cudaData, chunkRows, rowOffset, seqlen, heads, dim,
            ropeDim, ropeBase, startPos, originalSeqLen, ropeFactor, betaFast, betaSlow);
    }
    if (ok) {
        DeviceSync();
    } else if (std::getenv("FASTLLM_DSV4_DEBUG_CUBLAS_SPARSE_PREFILL") != nullptr) {
        std::fprintf(stderr, "DeepSeekV4SparsePrefill compressed cuBLAS path fell back to row-wise kernel.\n");
    }

    FastllmCudaFree(scratch);
    return ok;
}

template <typename QT, typename KT>
bool DeepSeekV4LaunchSparsePrefillLocalCublasSegment(
        const fastllm::Data &q, const fastllm::Data &kv, const float *cudaSink,
        float *localScores, float *compressedScores,
        __nv_bfloat16 *localScoresBf16, __nv_bfloat16 *compressedScoresBf16,
        float *rawKVFloat,
        float *compressedKVFloat, float *outData,
        int bsz, int seqlen, int heads, int dim, int kvLen,
        int windowSize, int compressRatio, int startPos, int prefixLen,
        int rawStart, int rawKeyCount, int compressedStart, int compressedCount,
        int compressedKVStride, float softmaxScale, int rowOffset, int rows,
        bool compressedKVFloatReady) {
    if (rows <= 0 || rawKeyCount <= 0 || q.dataType != kv.dataType) {
        return false;
    }
    if (compressedCount < 0 || compressedCount > compressedKVStride) {
        return false;
    }

    int rowsPerBatch = seqlen * heads;
    int b = rowOffset / rowsPerBatch;
    if (b < 0 || b >= bsz || rowOffset + rows > (b + 1) * rowsPerBatch) {
        return false;
    }

    const QT *qPtr = (const QT *)q.cudaData + (uint64_t)rowOffset * dim;
    const KT *rawKVPtr = (const KT *)kv.cudaData + ((uint64_t)b * kvLen + rawStart) * dim;
    const KT *compressedKVPtr = compressedCount > 0 ?
        (const KT *)kv.cudaData + ((uint64_t)b * kvLen + compressedStart) * dim : nullptr;
    bool useBf16TensorGemm = q.dataType == fastllm::DataType::BFLOAT16 &&
                             localScoresBf16 != nullptr &&
                             (compressedCount == 0 || compressedScoresBf16 != nullptr);

    const float *qForGemm = nullptr;
    const float *rawKVForGemm = nullptr;
    const float *compressedKVForGemm = nullptr;
    if (useBf16TensorGemm) {
        // BF16 Tensor Core path reads q/kv directly and keeps FP32 scores for softmax.
    } else if (q.dataType == fastllm::DataType::FLOAT32) {
        qForGemm = (const float *)qPtr;
        rawKVForGemm = (const float *)rawKVPtr;
        compressedKVForGemm = (const float *)compressedKVPtr;
    } else {
        if (rawKVFloat == nullptr) {
            return false;
        }
        int threads = 256;
        uint64_t qElems = (uint64_t)rows * dim;
        DeepSeekV4SparseDecodeConvertQKernel<QT><<<
            (int)((qElems + threads - 1) / threads), threads>>>(qPtr, outData, qElems);
        qForGemm = outData;

        uint64_t rawElems = (uint64_t)rawKeyCount * dim;
        DeepSeekV4SparseDecodeGatherKVKernel<KT><<<
            (int)((rawElems + threads - 1) / threads), threads>>>(
                (const float *)nullptr, rawKVPtr, rawKVFloat, 1, dim, 1,
                0, rawKeyCount, 0, rawKeyCount);
        rawKVForGemm = rawKVFloat;

        if (compressedCount > 0) {
            if (compressedKVFloat == nullptr) {
                return false;
            }
            if (compressedKVFloatReady) {
                compressedKVForGemm = compressedKVFloat + (uint64_t)b * compressedKVStride * dim;
            } else {
                uint64_t kvElems = (uint64_t)compressedCount * dim;
                DeepSeekV4SparseDecodeGatherKVKernel<KT><<<
                    (int)((kvElems + threads - 1) / threads), threads>>>(
                        (const float *)nullptr, compressedKVPtr, compressedKVFloat, 1, dim, 1,
                        0, compressedCount, 0, compressedCount);
                compressedKVForGemm = compressedKVFloat;
            }
        }
    }

    float alpha = softmaxScale;
    float beta = 0.0f;
    auto handle = getFastllmCublasHandle();
    cublasStatus_t status;
    if (useBf16TensorGemm) {
        status = DeepSeekV4CublasBf16GemmToFloat(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            rawKeyCount, rows, dim,
            &alpha,
            (const __nv_bfloat16 *)rawKVPtr, dim,
            (const __nv_bfloat16 *)qPtr, dim,
            &beta,
            localScores, rawKeyCount);
    } else {
        status = DeepSeekV4CublasSgemmStrict(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            rawKeyCount, rows, dim,
            &alpha,
            rawKVForGemm, dim,
            qForGemm, dim,
            &beta,
            localScores, rawKeyCount);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        if (std::getenv("FASTLLM_DSV4_DEBUG_CUBLAS_SPARSE_PREFILL") != nullptr) {
            std::fprintf(stderr,
                         "DeepSeekV4SparsePrefill local cuBLAS QK failed: status=%d rows=%d dim=%d rawKeys=%d type=%d\n",
                         (int)status, rows, dim, rawKeyCount, (int)q.dataType);
        }
        return false;
    }

    if (compressedCount > 0) {
        if (useBf16TensorGemm) {
            status = DeepSeekV4CublasBf16GemmToFloat(
                handle, CUBLAS_OP_T, CUBLAS_OP_N,
                compressedCount, rows, dim,
                &alpha,
                (const __nv_bfloat16 *)compressedKVPtr, dim,
                (const __nv_bfloat16 *)qPtr, dim,
                &beta,
                compressedScores, compressedCount);
        } else {
            status = DeepSeekV4CublasSgemmStrict(
                handle, CUBLAS_OP_T, CUBLAS_OP_N,
                compressedCount, rows, dim,
                &alpha,
                compressedKVForGemm, dim,
                qForGemm, dim,
                &beta,
                compressedScores, compressedCount);
        }
        if (status != CUBLAS_STATUS_SUCCESS) {
            if (std::getenv("FASTLLM_DSV4_DEBUG_CUBLAS_SPARSE_PREFILL") != nullptr) {
                std::fprintf(stderr,
                             "DeepSeekV4SparsePrefill local cuBLAS compressed QK failed: status=%d rows=%d dim=%d compressed=%d type=%d\n",
                             (int)status, rows, dim, compressedCount, (int)q.dataType);
            }
            return false;
        }
    }

    DeepSeekV4SparsePrefillCublasLocalSoftmaxKernel<<<rows, 256, 256 * sizeof(float)>>>(
        localScores, compressedCount > 0 ? compressedScores : nullptr,
        useBf16TensorGemm ? localScoresBf16 : nullptr,
        useBf16TensorGemm && compressedCount > 0 ? compressedScoresBf16 : nullptr,
        cudaSink,
        bsz, seqlen, heads, rawStart, rawKeyCount, compressedCount, windowSize, compressRatio,
        startPos, prefixLen, kvLen, rowOffset, rows);

    float avAlpha = 1.0f;
    float avBeta = 0.0f;
    if (useBf16TensorGemm) {
        status = DeepSeekV4CublasBf16GemmToFloat(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            dim, rows, rawKeyCount,
            &avAlpha,
            (const __nv_bfloat16 *)rawKVPtr, dim,
            localScoresBf16, rawKeyCount,
            &avBeta,
            outData, dim);
    } else {
        status = DeepSeekV4CublasSgemmStrict(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            dim, rows, rawKeyCount,
            &avAlpha,
            rawKVForGemm, dim,
            localScores, rawKeyCount,
            &avBeta,
            outData, dim);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        if (std::getenv("FASTLLM_DSV4_DEBUG_CUBLAS_SPARSE_PREFILL") != nullptr) {
            std::fprintf(stderr,
                         "DeepSeekV4SparsePrefill local cuBLAS AV failed: status=%d rows=%d dim=%d rawKeys=%d type=%d\n",
                         (int)status, rows, dim, rawKeyCount, (int)kv.dataType);
        }
        return false;
    }

    if (compressedCount > 0) {
        avBeta = 1.0f;
        if (useBf16TensorGemm) {
            status = DeepSeekV4CublasBf16GemmToFloat(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                dim, rows, compressedCount,
                &avAlpha,
                (const __nv_bfloat16 *)compressedKVPtr, dim,
                compressedScoresBf16, compressedCount,
                &avBeta,
                outData, dim);
        } else {
            status = DeepSeekV4CublasSgemmStrict(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                dim, rows, compressedCount,
                &avAlpha,
                compressedKVForGemm, dim,
                compressedScores, compressedCount,
                &avBeta,
                outData, dim);
        }
        if (status != CUBLAS_STATUS_SUCCESS) {
            if (std::getenv("FASTLLM_DSV4_DEBUG_CUBLAS_SPARSE_PREFILL") != nullptr) {
                std::fprintf(stderr,
                             "DeepSeekV4SparsePrefill local cuBLAS compressed AV failed: status=%d rows=%d dim=%d compressed=%d type=%d\n",
                             (int)status, rows, dim, compressedCount, (int)kv.dataType);
            }
            return false;
        }
    }
    return true;
}

bool DeepSeekV4RunSparsePrefillLocalCublas(
        const fastllm::Data &q, const fastllm::Data &kv, const float *cudaSink,
        fastllm::Data &output,
        int bsz, int seqlen, int heads, int dim, int kvLen,
        int windowSize, int compressRatio, int startPos, int prefixLen,
        int ropeDim, float ropeBase, int originalSeqLen, float ropeFactor,
        int betaFast, int betaSlow, float softmaxScale) {
    if (std::getenv("FASTLLM_DSV4_DISABLE_CUBLAS_SPARSE_PREFILL_LOCAL") != nullptr) {
        return false;
    }
    if (q.dataType != kv.dataType || windowSize <= 0 || dim <= 0 || heads <= 0 || seqlen <= 0) {
        return false;
    }

    int realPrefixLen = std::max(0, std::min(prefixLen, kvLen - seqlen));
    int rawTotal = realPrefixLen + seqlen;
    int compressedStart = realPrefixLen + seqlen;
    int compressedCount = std::max(0, kvLen - compressedStart);
    int maxCompressed = kDeepSeekV4Sparse1MCompressedKeys;
    if (const char *env = std::getenv("FASTLLM_DSV4_CUBLAS_SPARSE_PREFILL_MAX_COMPRESSED")) {
        maxCompressed = std::max(1, std::atoi(env));
    }
    if (compressedCount > maxCompressed || rawTotal <= 0) {
        return false;
    }
    bool useBf16TensorGemm = q.dataType == fastllm::DataType::BFLOAT16 &&
                             std::getenv("FASTLLM_DSV4_DISABLE_BF16_SPARSE_PREFILL_GEMM") == nullptr;

    int tokenBlock = 64;
    if (const char *env = std::getenv("FASTLLM_DSV4_CUBLAS_SPARSE_PREFILL_LOCAL_TOKENS")) {
        tokenBlock = std::max(1, std::atoi(env));
    }
    tokenBlock = std::min(tokenBlock, seqlen);
    size_t tempLimit = DeepSeekV4SparsePrefillTempBytesLimit();
    auto calcScratchBytes = [&](int tb) {
        int rowsMax = tb * heads;
        int maxRawKeys = std::min(rawTotal, windowSize + tb - 1);
        uint64_t tempFloats = (uint64_t)rowsMax * dim;
        uint64_t localScoreFloats = (uint64_t)rowsMax * maxRawKeys;
        uint64_t compressedScoreFloats = (uint64_t)rowsMax * compressedCount;
        uint64_t rawKVFloats = (q.dataType == fastllm::DataType::FLOAT32 || useBf16TensorGemm) ?
            0ULL : (uint64_t)maxRawKeys * dim;
        uint64_t compressedKVFloats = (q.dataType == fastllm::DataType::FLOAT32 ||
                                       useBf16TensorGemm || compressedCount <= 0) ?
            0ULL : (uint64_t)bsz * compressedCount * dim;
        uint64_t localScoreBf16Elems = useBf16TensorGemm ? (uint64_t)rowsMax * maxRawKeys : 0ULL;
        uint64_t compressedScoreBf16Elems = (useBf16TensorGemm && compressedCount > 0) ?
            (uint64_t)rowsMax * compressedCount : 0ULL;
        return (tempFloats + localScoreFloats + compressedScoreFloats + rawKVFloats + compressedKVFloats) *
                   sizeof(float) +
               (localScoreBf16Elems + compressedScoreBf16Elems) * sizeof(__nv_bfloat16);
    };
    while (tokenBlock > 1 && calcScratchBytes(tokenBlock) > tempLimit) {
        tokenBlock = std::max(1, tokenBlock / 2);
    }

    int rowsMax = tokenBlock * heads;
    int maxRawKeys = std::min(rawTotal, windowSize + tokenBlock - 1);
    size_t tempBytes = (size_t)rowsMax * dim * sizeof(float);
    size_t localScoreBytes = (size_t)rowsMax * maxRawKeys * sizeof(float);
    size_t compressedScoreBytes = (size_t)rowsMax * compressedCount * sizeof(float);
    size_t rawKVBytes = (q.dataType == fastllm::DataType::FLOAT32 || useBf16TensorGemm) ?
        0 : (size_t)maxRawKeys * dim * sizeof(float);
    size_t compressedKVBytes = (q.dataType == fastllm::DataType::FLOAT32 ||
                                useBf16TensorGemm || compressedCount <= 0) ?
        0 : (size_t)bsz * compressedCount * dim * sizeof(float);
    size_t localScoreBf16Bytes = useBf16TensorGemm ?
        (size_t)rowsMax * maxRawKeys * sizeof(__nv_bfloat16) : 0;
    size_t compressedScoreBf16Bytes = (useBf16TensorGemm && compressedCount > 0) ?
        (size_t)rowsMax * compressedCount * sizeof(__nv_bfloat16) : 0;
    size_t scratchBytes = tempBytes + localScoreBytes + compressedScoreBytes +
                          rawKVBytes + compressedKVBytes +
                          localScoreBf16Bytes + compressedScoreBf16Bytes;
    if (scratchBytes == 0) {
        return false;
    }
    uint8_t *scratch = (uint8_t *)FastllmCudaMalloc(scratchBytes);
    if (scratch == nullptr) {
        return false;
    }
    float *cudaTemp = (float *)scratch;
    float *localScores = (float *)(scratch + tempBytes);
    float *compressedScores = compressedCount > 0 ? (float *)(scratch + tempBytes + localScoreBytes) : nullptr;
    float *rawKVFloat = rawKVBytes == 0 ? nullptr :
        (float *)(scratch + tempBytes + localScoreBytes + compressedScoreBytes);
    float *compressedKVFloat = compressedKVBytes == 0 ? nullptr :
        (float *)(scratch + tempBytes + localScoreBytes + compressedScoreBytes + rawKVBytes);
    __nv_bfloat16 *localScoresBf16 = localScoreBf16Bytes == 0 ? nullptr :
        (__nv_bfloat16 *)(scratch + tempBytes + localScoreBytes + compressedScoreBytes +
                          rawKVBytes + compressedKVBytes);
    __nv_bfloat16 *compressedScoresBf16 = compressedScoreBf16Bytes == 0 ? nullptr :
        (__nv_bfloat16 *)(scratch + tempBytes + localScoreBytes + compressedScoreBytes +
                          rawKVBytes + compressedKVBytes + localScoreBf16Bytes);
    bool compressedKVFloatReady = false;
    if (compressedKVFloat != nullptr && compressedCount > 0) {
        uint64_t compressedElems = (uint64_t)bsz * compressedCount * dim;
        int threads = 256;
        if (kv.dataType == fastllm::DataType::BFLOAT16) {
            DeepSeekV4SparsePrefillCastCompressedKVKernel<<<
                (int)((compressedElems + threads - 1) / threads), threads>>>(
                    (const __nv_bfloat16 *)kv.cudaData, compressedKVFloat,
                    bsz, kvLen, compressedStart, compressedCount, dim);
            compressedKVFloatReady = true;
        } else if (kv.dataType == fastllm::DataType::FLOAT16) {
            DeepSeekV4SparsePrefillCastCompressedKVKernel<<<
                (int)((compressedElems + threads - 1) / threads), threads>>>(
                    (const half *)kv.cudaData, compressedKVFloat,
                    bsz, kvLen, compressedStart, compressedCount, dim);
            compressedKVFloatReady = true;
        }
    }

    bool ok = true;
    for (int b = 0; b < bsz && ok; b++) {
        for (int tokenStart = 0; tokenStart < seqlen && ok; tokenStart += tokenBlock) {
            int tokenCount = std::min(tokenBlock, seqlen - tokenStart);
            int rows = tokenCount * heads;
            int rowOffset = (b * seqlen + tokenStart) * heads;
            int rawStart = std::max(0, realPrefixLen + tokenStart - windowSize + 1);
            int rawEnd = std::min(rawTotal, realPrefixLen + tokenStart + tokenCount);
            int rawKeyCount = rawEnd - rawStart;
            if (rawKeyCount <= 0 || rawKeyCount > maxRawKeys) {
                ok = false;
                break;
            }
            int activeCompressedCount = 0;
            if (compressRatio > 0 && compressedCount > 0) {
                activeCompressedCount = std::min(compressedCount,
                                                 (startPos + tokenStart + tokenCount) / compressRatio);
            }
            if (q.dataType == fastllm::DataType::BFLOAT16) {
                ok = DeepSeekV4LaunchSparsePrefillLocalCublasSegment<__nv_bfloat16, __nv_bfloat16>(
                    q, kv, cudaSink, localScores, compressedScores, localScoresBf16, compressedScoresBf16,
                    rawKVFloat, compressedKVFloat, cudaTemp,
                    bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos, realPrefixLen,
                    rawStart, rawKeyCount, compressedStart, activeCompressedCount, compressedCount,
                    softmaxScale, rowOffset, rows, compressedKVFloatReady);
            } else if (q.dataType == fastllm::DataType::FLOAT16) {
                ok = DeepSeekV4LaunchSparsePrefillLocalCublasSegment<half, half>(
                    q, kv, cudaSink, localScores, compressedScores, nullptr, nullptr,
                    rawKVFloat, compressedKVFloat, cudaTemp,
                    bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos, realPrefixLen,
                    rawStart, rawKeyCount, compressedStart, activeCompressedCount, compressedCount,
                    softmaxScale, rowOffset, rows, compressedKVFloatReady);
            } else if (q.dataType == fastllm::DataType::FLOAT32) {
                ok = DeepSeekV4LaunchSparsePrefillLocalCublasSegment<float, float>(
                    q, kv, cudaSink, localScores, compressedScores, nullptr, nullptr,
                    rawKVFloat, compressedKVFloat, cudaTemp,
                    bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos, realPrefixLen,
                    rawStart, rawKeyCount, compressedStart, activeCompressedCount, compressedCount,
                    softmaxScale, rowOffset, rows, compressedKVFloatReady);
            } else {
                ok = false;
            }
            if (!ok) {
                break;
            }
            int threads = 128;
            int blocks = (rows + threads - 1) / threads;
            DeepSeekV4SparsePrefillRotaryCastKernel<<<blocks, threads>>>(
                cudaTemp, (__nv_bfloat16 *)output.cudaData, rows, rowOffset, seqlen, heads, dim,
                ropeDim, ropeBase, startPos, originalSeqLen, ropeFactor, betaFast, betaSlow);
        }
    }
    if (ok) {
        DeviceSync();
    } else if (std::getenv("FASTLLM_DSV4_DEBUG_CUBLAS_SPARSE_PREFILL") != nullptr) {
        std::fprintf(stderr, "DeepSeekV4SparsePrefill local cuBLAS path fell back to hybrid kernel.\n");
    }

    FastllmCudaFree(scratch);
    return ok;
}

template <typename QT, typename KT>
bool DeepSeekV4LaunchSparsePrefillByKV(const fastllm::Data &q, const fastllm::Data &kv,
                                       const float *cudaSink, float *outData,
                                       int bsz, int seqlen, int heads, int dim, int kvLen,
                                       int windowSize, int compressRatio, int startPos, int prefixLen,
                                       float softmaxScale, int rowOffset, int rows) {
    int realPrefixLen = std::max(0, std::min(prefixLen, kvLen - seqlen));
    int compressedCount = std::max(0, kvLen - realPrefixLen - seqlen);
    int maxKeys = std::min(windowSize, realPrefixLen + seqlen) + (compressRatio > 0 ? compressedCount : 0);
    if (maxKeys <= 0 || maxKeys > kDeepSeekV4SparsePrefillMaxKeys || rows <= 0) {
        return false;
    }
    size_t sharedBytes = (size_t)maxKeys * sizeof(float);
    auto kernel = DeepSeekV4SparseAttentionPrefillBlockKernel<QT, KT>;
    if (!DeepSeekV4EnsureDynamicSharedMemory(kernel, sharedBytes)) {
        return false;
    }
    kernel<<<rows, 256, sharedBytes>>>(
        (const QT *)q.cudaData, (const KT *)kv.cudaData, cudaSink, outData,
        bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos, realPrefixLen, softmaxScale,
        rowOffset);
    return true;
}

template <typename QT>
bool DeepSeekV4LaunchSparsePrefillByQ(const fastllm::Data &q, const fastllm::Data &kv,
                                      const float *cudaSink, float *outData,
                                      int bsz, int seqlen, int heads, int dim, int kvLen,
                                      int windowSize, int compressRatio, int startPos, int prefixLen,
                                      float softmaxScale, int rowOffset, int rows) {
    if (kv.dataType == fastllm::DataType::BFLOAT16) {
        return DeepSeekV4LaunchSparsePrefillByKV<QT, __nv_bfloat16>(
            q, kv, cudaSink, outData, bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio,
            startPos, prefixLen, softmaxScale, rowOffset, rows);
    }
    if (kv.dataType == fastllm::DataType::FLOAT16) {
        return DeepSeekV4LaunchSparsePrefillByKV<QT, half>(
            q, kv, cudaSink, outData, bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio,
            startPos, prefixLen, softmaxScale, rowOffset, rows);
    }
    if (kv.dataType == fastllm::DataType::FLOAT32) {
        return DeepSeekV4LaunchSparsePrefillByKV<QT, float>(
            q, kv, cudaSink, outData, bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio,
            startPos, prefixLen, softmaxScale, rowOffset, rows);
    }
    return false;
}

bool DeepSeekV4LaunchHcPreCublasDots(const fastllm::Data &x, const fastllm::Data &hcFn,
                                     float *dotsData, int tokens, int flatDim,
                                     int mixHc, int dotsStride) {
    if (std::getenv("FASTLLM_DSV4_DISABLE_CUDA_HCPRE_CUBLAS") != nullptr) {
        return false;
    }
    cudaDataType_t xType, wType;
    if (!DeepSeekV4CublasDataType(x.dataType, xType) ||
        !DeepSeekV4CublasDataType(hcFn.dataType, wType)) {
        return false;
    }
    float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t status = cublasGemmEx(
        getFastllmCublasHandle(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        mixHc, tokens, flatDim,
        &alpha,
        hcFn.cudaData, wType, flatDim,
        x.cudaData, xType, flatDim,
        &beta,
        dotsData, CUDA_R_32F, dotsStride,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS) {
        if (std::getenv("FASTLLM_DSV4_DEBUG_CUDA_HCPRE_CUBLAS") != nullptr) {
            std::fprintf(stderr, "DeepSeekV4HcPre cuBLAS dots failed: status=%d tokens=%d flatDim=%d mixHc=%d xType=%d wType=%d\n",
                         (int)status, tokens, flatDim, mixHc, (int)x.dataType, (int)hcFn.dataType);
        }
        return false;
    }
    return true;
}

template <typename XT>
bool DeepSeekV4LaunchHcPreByWeight(const fastllm::Data &x, const fastllm::Data &hcFn,
                                   const fastllm::Data &hcScale, const fastllm::Data &hcBase,
                                   fastllm::Data &y, fastllm::Data &post, fastllm::Data &comb,
                                   float *dotsData,
                                   int tokens, int dim, int hcMult, int sinkhornIters,
                                   float eps, float normEps) {
    int threads = 256;
    int mixHc = (2 + hcMult) * hcMult;
    int flatDim = hcMult * dim;
    int dotsStride = mixHc + 1;
    int dotParts = DeepSeekV4HcPreDotParts(flatDim, threads);
    int dotBlocks = tokens * dotsStride * dotParts;
    int finishParts = DeepSeekV4HcPreFinishParts(dim, threads);
    dim3 finishGrid(tokens, finishParts);
    size_t dotSharedBytes = (size_t)threads * sizeof(float);
    size_t finishSharedBytes = (size_t)(mixHc + hcMult + hcMult * hcMult + hcMult * 2) * sizeof(float);
    const XT *xData = (const XT *)x.cudaData;
    XT *yData = (XT *)y.cudaData;
    float *postData = (float *)post.cudaData;
    float *combData = (float *)comb.cudaData;
    const float *scaleData = (const float *)hcScale.cudaData;
    const float *baseData = (const float *)hcBase.cudaData;
    if (hcFn.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4HcPreDotsBlockKernel<<<dotBlocks, threads, dotSharedBytes>>>(
            xData, (const __nv_bfloat16 *)hcFn.cudaData, dotsData, tokens, flatDim, mixHc, dotsStride, dotParts);
    } else if (hcFn.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4HcPreDotsBlockKernel<<<dotBlocks, threads, dotSharedBytes>>>(
            xData, (const half *)hcFn.cudaData, dotsData, tokens, flatDim, mixHc, dotsStride, dotParts);
    } else if (hcFn.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4HcPreDotsBlockKernel<<<dotBlocks, threads, dotSharedBytes>>>(
            xData, (const float *)hcFn.cudaData, dotsData, tokens, flatDim, mixHc, dotsStride, dotParts);
    } else {
        return false;
    }
    DeepSeekV4HcPreFinishKernel<<<finishGrid, threads, finishSharedBytes>>>(
        xData, dotsData, scaleData, baseData, yData, postData, combData,
        tokens, dim, hcMult, sinkhornIters, eps, normEps, dotsStride, dotParts);
    return true;
}

template <typename XT>
bool DeepSeekV4LaunchHcPreCublasByWeight(const fastllm::Data &x, const fastllm::Data &hcFn,
                                         const fastllm::Data &hcScale, const fastllm::Data &hcBase,
                                         fastllm::Data &y, fastllm::Data &post, fastllm::Data &comb,
                                         float *dotsData,
                                         int tokens, int dim, int hcMult, int sinkhornIters,
                                         float eps, float normEps) {
    int threads = 256;
    int mixHc = (2 + hcMult) * hcMult;
    int flatDim = hcMult * dim;
    int dotsStride = mixHc + 1;
    if (!DeepSeekV4LaunchHcPreCublasDots(x, hcFn, dotsData, tokens, flatDim, mixHc, dotsStride)) {
        return false;
    }

    DeepSeekV4HcPreSqSumKernel<<<tokens, threads, threads * sizeof(float)>>>(
        (const XT *)x.cudaData, dotsData, tokens, flatDim, dotsStride);

    int finishParts = DeepSeekV4HcPreFinishParts(dim, threads);
    dim3 finishGrid(tokens, finishParts);
    size_t finishSharedBytes = (size_t)(mixHc + hcMult + hcMult * hcMult + hcMult * 2) * sizeof(float);
    DeepSeekV4HcPreFinishKernel<<<finishGrid, threads, finishSharedBytes>>>(
        (const XT *)x.cudaData, dotsData,
        (const float *)hcScale.cudaData, (const float *)hcBase.cudaData,
        (XT *)y.cudaData, (float *)post.cudaData, (float *)comb.cudaData,
        tokens, dim, hcMult, sinkhornIters, eps, normEps, dotsStride, 1);
    return true;
}

template <typename XT>
bool DeepSeekV4LaunchHcPreDotsByWeight(const fastllm::Data &x, const fastllm::Data &hcFn,
                                       fastllm::Data &dotsFloat, int tokens, int flatDim, int mixHc) {
    int total = tokens * mixHc;
    int threads = std::min(256, std::max(1, total));
    int blocks = (total + threads - 1) / threads;
    const XT *xData = (const XT *)x.cudaData;
    float *dotsData = (float *)dotsFloat.cudaData;
    if (hcFn.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4HcPreDotsKernel<<<blocks, threads>>>(xData, (const __nv_bfloat16 *)hcFn.cudaData,
                                                       dotsData, tokens, flatDim, mixHc);
    } else if (hcFn.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4HcPreDotsKernel<<<blocks, threads>>>(xData, (const half *)hcFn.cudaData,
                                                       dotsData, tokens, flatDim, mixHc);
    } else if (hcFn.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4HcPreDotsKernel<<<blocks, threads>>>(xData, (const float *)hcFn.cudaData,
                                                       dotsData, tokens, flatDim, mixHc);
    } else {
        return false;
    }
    return true;
}

template <typename XT>
bool DeepSeekV4LaunchHcHeadDotsByWeight(const fastllm::Data &x,
                                        const fastllm::Data &hcFn,
                                        float *dots, int tokens, int flatDim,
                                        int hcMult, int dotsStride) {
    constexpr int threads = 256;
    int blocks = tokens * hcMult;
    size_t sharedBytes = threads * sizeof(double);
    if (hcFn.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4HcHeadDotsKernel<<<blocks, threads, sharedBytes>>>(
            (const XT *)x.cudaData, (const __nv_bfloat16 *)hcFn.cudaData,
            dots, tokens, flatDim, hcMult, dotsStride);
    } else if (hcFn.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4HcHeadDotsKernel<<<blocks, threads, sharedBytes>>>(
            (const XT *)x.cudaData, (const half *)hcFn.cudaData,
            dots, tokens, flatDim, hcMult, dotsStride);
    } else if (hcFn.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4HcHeadDotsKernel<<<blocks, threads, sharedBytes>>>(
            (const XT *)x.cudaData, (const float *)hcFn.cudaData,
            dots, tokens, flatDim, hcMult, dotsStride);
    } else {
        return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

} // namespace

extern "C" bool FastllmCudaDeepSeekV4HcPre(const fastllm::Data &x, fastllm::Data &hcFn,
                                           fastllm::Data &hcScale, fastllm::Data &hcBase,
                                           int hcMult, int sinkhornIters, float eps, float normEps,
                                           fastllm::Data &y, fastllm::Data &post, fastllm::Data &comb) {
    if (x.dataDevice != fastllm::DataDevice::CUDA || hcFn.dataDevice != fastllm::DataDevice::CUDA ||
        hcScale.dataDevice != fastllm::DataDevice::CUDA || hcBase.dataDevice != fastllm::DataDevice::CUDA ||
        x.dims.size() != 4 || hcMult <= 0 || sinkhornIters <= 0 ||
        hcScale.dataType != fastllm::DataType::FLOAT32 ||
        hcBase.dataType != fastllm::DataType::FLOAT32) {
        return false;
    }
    int bsz = x.dims[0], seqlen = x.dims[1], dim = x.dims[3];
    int flatDim = hcMult * dim;
    int mixHc = (2 + hcMult) * hcMult;
    int tokens = bsz * seqlen;
    if (x.dims[2] != hcMult || hcFn.Count(0) != (uint64_t)mixHc * flatDim ||
        hcScale.Count(0) < 3 || hcBase.Count(0) < (uint64_t)mixHc) {
        return false;
    }
    if (!DeepSeekV4PrepareCudaOutput(y, x.dataType, {bsz, seqlen, dim}) ||
        !DeepSeekV4PrepareCudaOutput(post, fastllm::DataType::FLOAT32, {bsz, seqlen, hcMult}) ||
        !DeepSeekV4PrepareCudaOutput(comb, fastllm::DataType::FLOAT32, {bsz, seqlen, hcMult, hcMult})) {
        return false;
    }

    int cublasMinTokens = 128;
    if (const char *env = std::getenv("FASTLLM_DSV4_HCPRE_CUBLAS_MIN_TOKENS")) {
        cublasMinTokens = std::max(1, std::atoi(env));
    }
    if (std::getenv("FASTLLM_DSV4_DISABLE_CUDA_HCPRE_CUBLAS") == nullptr &&
        tokens >= cublasMinTokens) {
        float *cublasDots = (float *)FastllmCudaMalloc((size_t)tokens * (mixHc + 1) * sizeof(float));
        if (cublasDots != nullptr) {
            bool cublasOk = false;
            if (x.dataType == fastllm::DataType::BFLOAT16) {
                cublasOk = DeepSeekV4LaunchHcPreCublasByWeight<__nv_bfloat16>(
                    x, hcFn, hcScale, hcBase, y, post, comb, cublasDots,
                    tokens, dim, hcMult, sinkhornIters, eps, normEps);
            } else if (x.dataType == fastllm::DataType::FLOAT16) {
                cublasOk = DeepSeekV4LaunchHcPreCublasByWeight<half>(
                    x, hcFn, hcScale, hcBase, y, post, comb, cublasDots,
                    tokens, dim, hcMult, sinkhornIters, eps, normEps);
            } else if (x.dataType == fastllm::DataType::FLOAT32) {
                cublasOk = DeepSeekV4LaunchHcPreCublasByWeight<float>(
                    x, hcFn, hcScale, hcBase, y, post, comb, cublasDots,
                    tokens, dim, hcMult, sinkhornIters, eps, normEps);
            }
            DeviceSync();
            FastllmCudaFree(cublasDots);
            if (cublasOk) {
                return true;
            }
        }
    }

    int dotParts = DeepSeekV4HcPreDotParts(flatDim, 256);
    float *dotsData = (float *)FastllmCudaMalloc((size_t)tokens * (mixHc + 1) * dotParts * sizeof(float));
    if (dotsData == nullptr) {
        return false;
    }

    bool ok = false;
    if (x.dataType == fastllm::DataType::BFLOAT16) {
        ok = DeepSeekV4LaunchHcPreByWeight<__nv_bfloat16>(
            x, hcFn, hcScale, hcBase, y, post, comb, dotsData,
            tokens, dim, hcMult, sinkhornIters, eps, normEps);
    } else if (x.dataType == fastllm::DataType::FLOAT16) {
        ok = DeepSeekV4LaunchHcPreByWeight<half>(
            x, hcFn, hcScale, hcBase, y, post, comb, dotsData,
            tokens, dim, hcMult, sinkhornIters, eps, normEps);
    } else if (x.dataType == fastllm::DataType::FLOAT32) {
        ok = DeepSeekV4LaunchHcPreByWeight<float>(
            x, hcFn, hcScale, hcBase, y, post, comb, dotsData,
            tokens, dim, hcMult, sinkhornIters, eps, normEps);
    }
    DeviceSync();
    FastllmCudaFree(dotsData);
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV4HcPreNorm(const fastllm::Data &x,
                                                fastllm::Data &hcFn,
                                                fastllm::Data &hcScale,
                                                fastllm::Data &hcBase,
                                                fastllm::Data &normWeight,
                                                int hcMult, int sinkhornIters,
                                                float eps, float normEps,
                                                fastllm::Data &normOutput,
                                                fastllm::Data &post,
                                                fastllm::Data &comb) {
    // This specialization intentionally covers the DeepSeek-V4 decode shape
    // only.  Returning false preserves the established HcPre + RMSNorm path
    // for prefill, other dtypes, and future model variants.
    if (x.dataDevice != fastllm::DataDevice::CUDA ||
        hcFn.dataDevice != fastllm::DataDevice::CUDA ||
        hcScale.dataDevice != fastllm::DataDevice::CUDA ||
        hcBase.dataDevice != fastllm::DataDevice::CUDA ||
        normWeight.dataDevice != fastllm::DataDevice::CUDA ||
        x.dataType != fastllm::DataType::BFLOAT16 ||
        hcFn.dataType != fastllm::DataType::FLOAT32 ||
        hcScale.dataType != fastllm::DataType::FLOAT32 ||
        hcBase.dataType != fastllm::DataType::FLOAT32 ||
        normWeight.dataType != fastllm::DataType::FLOAT32 ||
        x.dims.size() != 4 || x.dims[2] != 4 || x.dims[3] != 4096 ||
        hcMult != 4 || sinkhornIters <= 0 || x.dims[1] != 1) {
        return false;
    }

    int bsz = x.dims[0];
    int seqlen = x.dims[1];
    int tokens = bsz * seqlen;
    constexpr int dim = 4096;
    constexpr int flatDim = 4 * dim;
    constexpr int mixHc = 24;
    constexpr int dotsStride = mixHc + 1;
    int dotParts = DeepSeekV4HcPreDotParts(flatDim, 256);
    if (tokens <= 0 || hcFn.Count(0) != (uint64_t)mixHc * flatDim ||
        hcScale.Count(0) < 3 || hcBase.Count(0) < mixHc ||
        normWeight.Count(0) < dim) {
        return false;
    }
    if (!DeepSeekV4PrepareCudaOutput(normOutput, fastllm::DataType::BFLOAT16,
                                     {bsz, seqlen, dim}) ||
        !DeepSeekV4PrepareCudaOutput(post, fastllm::DataType::FLOAT32,
                                     {bsz, seqlen, hcMult}) ||
        !DeepSeekV4PrepareCudaOutput(comb, fastllm::DataType::FLOAT32,
                                     {bsz, seqlen, hcMult, hcMult})) {
        return false;
    }

    size_t dotsBytes = (size_t)tokens * dotsStride * dotParts * sizeof(float);
    float *dots = (float *)FastllmCudaMalloc(dotsBytes);
    if (dots == nullptr) {
        return false;
    }
    int dotThreads = 256;
    int dotBlocks = tokens * dotsStride * dotParts;
    DeepSeekV4HcPreDotsBlockKernel<<<dotBlocks, dotThreads,
                                     dotThreads * sizeof(float)>>>(
        (const __nv_bfloat16 *)x.cudaData,
        (const float *)hcFn.cudaData, dots, tokens, flatDim,
        mixHc, dotsStride, dotParts);

    constexpr int finishThreads = 512;
    DeepSeekV4HcPreFinishNorm4x4096Kernel<<<tokens, finishThreads>>>(
        (const __nv_bfloat16 *)x.cudaData, dots,
        (const float *)hcScale.cudaData, (const float *)hcBase.cudaData,
        (const float *)normWeight.cudaData,
        (__nv_bfloat16 *)normOutput.cudaData,
        (float *)post.cudaData, (float *)comb.cudaData,
        tokens, sinkhornIters, eps, normEps, dotsStride, dotParts);
    DeviceSync();
    FastllmCudaFree(dots);
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4HcPostPreNorm(
        const fastllm::Data &x, const fastllm::Data &residual,
        const fastllm::Data &previousPost,
        const fastllm::Data &previousComb, fastllm::Data &nextHcFn,
        fastllm::Data &nextHcScale, fastllm::Data &nextHcBase,
        fastllm::Data &nextNormWeight, int hcMult, int sinkhornIters,
        float eps, float normEps, fastllm::Data &residualOutput,
        fastllm::Data &normOutput, fastllm::Data &nextPost,
        fastllm::Data &nextComb) {
    // Keep this path deliberately narrow.  The generic HcPost + HcPreNorm
    // sequence remains the fallback for prefill and non-DeepSeek-V4 shapes.
    constexpr int dim = 4096;
    constexpr int flatDim = 4 * dim;
    constexpr int mixHc = 24;
    constexpr int dotsStride = mixHc + 1;
    constexpr int dotParts = 8;
    if (x.dataDevice != fastllm::DataDevice::CUDA ||
        residual.dataDevice != fastllm::DataDevice::CUDA ||
        previousPost.dataDevice != fastllm::DataDevice::CUDA ||
        previousComb.dataDevice != fastllm::DataDevice::CUDA ||
        nextHcFn.dataDevice != fastllm::DataDevice::CUDA ||
        nextHcScale.dataDevice != fastllm::DataDevice::CUDA ||
        nextHcBase.dataDevice != fastllm::DataDevice::CUDA ||
        nextNormWeight.dataDevice != fastllm::DataDevice::CUDA ||
        x.dataType != fastllm::DataType::BFLOAT16 ||
        residual.dataType != fastllm::DataType::BFLOAT16 ||
        previousPost.dataType != fastllm::DataType::FLOAT32 ||
        previousComb.dataType != fastllm::DataType::FLOAT32 ||
        nextHcFn.dataType != fastllm::DataType::FLOAT32 ||
        nextHcScale.dataType != fastllm::DataType::FLOAT32 ||
        nextHcBase.dataType != fastllm::DataType::FLOAT32 ||
        nextNormWeight.dataType != fastllm::DataType::FLOAT32 ||
        residual.dims.size() != 4 || residual.dims[2] != 4 ||
        residual.dims[3] != dim || residual.dims[1] != 1 ||
        hcMult != 4 || sinkhornIters <= 0) {
        return false;
    }

    int bsz = residual.dims[0];
    int seqlen = residual.dims[1];
    int tokens = bsz * seqlen;
    if (tokens <= 0 || x.Count(0) != (uint64_t)tokens * dim ||
        residual.Count(0) != (uint64_t)tokens * flatDim ||
        previousPost.Count(0) != (uint64_t)tokens * hcMult ||
        previousComb.Count(0) != (uint64_t)tokens * hcMult * hcMult ||
        nextHcFn.Count(0) != (uint64_t)mixHc * flatDim ||
        nextHcScale.Count(0) < 3 || nextHcBase.Count(0) < mixHc ||
        nextNormWeight.Count(0) < dim) {
        return false;
    }
    if (!DeepSeekV4PrepareCudaOutput(
            residualOutput, fastllm::DataType::BFLOAT16,
            {bsz, seqlen, hcMult, dim}) ||
        !DeepSeekV4PrepareCudaOutput(
            normOutput, fastllm::DataType::BFLOAT16,
            {bsz, seqlen, dim}) ||
        !DeepSeekV4PrepareCudaOutput(
            nextPost, fastllm::DataType::FLOAT32,
            {bsz, seqlen, hcMult}) ||
        !DeepSeekV4PrepareCudaOutput(
            nextComb, fastllm::DataType::FLOAT32,
            {bsz, seqlen, hcMult, hcMult})) {
        return false;
    }

    size_t dotsBytes =
        (size_t)tokens * dotsStride * dotParts * sizeof(float);
    float *dots = (float *)FastllmCudaMalloc(dotsBytes);
    if (dots == nullptr) {
        return false;
    }

    dim3 transitionGrid(mixHc / 2, dotParts, tokens);
    DeepSeekV4HcPostPreDots4x4096Kernel<<<transitionGrid, 256>>>(
        (const __nv_bfloat16 *)x.cudaData,
        (const __nv_bfloat16 *)residual.cudaData,
        (const float *)previousPost.cudaData,
        (const float *)previousComb.cudaData,
        (const float *)nextHcFn.cudaData, dots,
        (__nv_bfloat16 *)residualOutput.cudaData, tokens);

    constexpr int finishThreads = 512;
    DeepSeekV4HcPreFinishNorm4x4096Kernel<<<tokens, finishThreads>>>(
        (const __nv_bfloat16 *)residualOutput.cudaData, dots,
        (const float *)nextHcScale.cudaData,
        (const float *)nextHcBase.cudaData,
        (const float *)nextNormWeight.cudaData,
        (__nv_bfloat16 *)normOutput.cudaData,
        (float *)nextPost.cudaData, (float *)nextComb.cudaData,
        tokens, sinkhornIters, eps, normEps, dotsStride, dotParts);
    bool ok = cudaGetLastError() == cudaSuccess;
    DeviceSync();
    FastllmCudaFree(dots);
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV4HcPreDots(const fastllm::Data &x, const fastllm::Data &hcFn,
                                               int hcMult, fastllm::Data &dotsFloat) {
    if (x.dataDevice != fastllm::DataDevice::CUDA || hcFn.dataDevice != fastllm::DataDevice::CUDA ||
        x.dims.size() != 4 || hcMult <= 0) {
        return false;
    }
    int bsz = x.dims[0], seqlen = x.dims[1], dim = x.dims[3];
    int flatDim = hcMult * dim;
    int mixHc = (2 + hcMult) * hcMult;
    int tokens = bsz * seqlen;
    if (x.dims[2] != hcMult || hcFn.Count(0) != (uint64_t)mixHc * flatDim) {
        return false;
    }
    dotsFloat.dataType = fastllm::DataType::FLOAT32;
    dotsFloat.Resize({tokens, mixHc});
    dotsFloat.Allocate(false);
    float *cudaOutput = (float *)FastllmCudaPrepareOutput(dotsFloat);
    dotsFloat.cudaData = cudaOutput;

    bool ok = false;
    if (x.dataType == fastllm::DataType::BFLOAT16) {
        ok = DeepSeekV4LaunchHcPreDotsByWeight<__nv_bfloat16>(x, hcFn, dotsFloat, tokens, flatDim, mixHc);
    } else if (x.dataType == fastllm::DataType::FLOAT16) {
        ok = DeepSeekV4LaunchHcPreDotsByWeight<half>(x, hcFn, dotsFloat, tokens, flatDim, mixHc);
    } else if (x.dataType == fastllm::DataType::FLOAT32) {
        ok = DeepSeekV4LaunchHcPreDotsByWeight<float>(x, hcFn, dotsFloat, tokens, flatDim, mixHc);
    }

    FastllmCudaFinishOutput(dotsFloat, cudaOutput);
    dotsFloat.cudaData = nullptr;
    dotsFloat.dataDevice = fastllm::DataDevice::CPU;
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV4HcHead(const fastllm::Data &x,
                                             const fastllm::Data &hcFn,
                                             const fastllm::Data &hcScale,
                                             const fastllm::Data &hcBase,
                                             int hcMult, float eps, float normEps,
                                             fastllm::Data &output) {
    if (x.dataDevice != fastllm::DataDevice::CUDA ||
        hcFn.dataDevice != fastllm::DataDevice::CUDA ||
        hcScale.dataDevice != fastllm::DataDevice::CUDA ||
        hcBase.dataDevice != fastllm::DataDevice::CUDA ||
        x.dims.size() != 4 || hcMult <= 0 || x.dims[2] != hcMult ||
        hcScale.dataType != fastllm::DataType::FLOAT32 ||
        hcBase.dataType != fastllm::DataType::FLOAT32) {
        return false;
    }
    int bsz = x.dims[0], seqlen = x.dims[1], dim = x.dims[3];
    int tokens = bsz * seqlen;
    int flatDim = hcMult * dim;
    if (hcFn.Count(0) != (uint64_t)hcMult * flatDim ||
        hcScale.Count(0) < 1 || hcBase.Count(0) < (uint64_t)hcMult ||
        !DeepSeekV4PrepareCudaOutput(output, x.dataType, {bsz, seqlen, dim})) {
        return false;
    }

    int dotsStride = hcMult + 1;
    float *dots = (float *)FastllmCudaMalloc((size_t)tokens * dotsStride * sizeof(float));
    if (dots == nullptr) {
        return false;
    }
    bool dotsOk = false;
    if (x.dataType == hcFn.dataType) {
        dotsOk = DeepSeekV4LaunchHcPreCublasDots(
            x, hcFn, dots, tokens, flatDim, hcMult, dotsStride);
    }
    if (!dotsOk && x.dataType == fastllm::DataType::BFLOAT16) {
        dotsOk = DeepSeekV4LaunchHcHeadDotsByWeight<__nv_bfloat16>(
            x, hcFn, dots, tokens, flatDim, hcMult, dotsStride);
    } else if (!dotsOk && x.dataType == fastllm::DataType::FLOAT16) {
        dotsOk = DeepSeekV4LaunchHcHeadDotsByWeight<half>(
            x, hcFn, dots, tokens, flatDim, hcMult, dotsStride);
    } else if (!dotsOk && x.dataType == fastllm::DataType::FLOAT32) {
        dotsOk = DeepSeekV4LaunchHcHeadDotsByWeight<float>(
            x, hcFn, dots, tokens, flatDim, hcMult, dotsStride);
    }
    if (!dotsOk) {
        FastllmCudaFree(dots);
        return false;
    }
    int threads = 256;
    bool ok = true;
    if (x.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4HcPreSqSumKernel<<<tokens, threads, threads * sizeof(float)>>>(
            (const __nv_bfloat16 *)x.cudaData, dots, tokens, flatDim, dotsStride);
        DeepSeekV4HcHeadFinishKernel<<<tokens, threads, (hcMult + 1) * sizeof(float)>>>(
            (const __nv_bfloat16 *)x.cudaData, dots,
            (const float *)hcScale.cudaData, (const float *)hcBase.cudaData,
            (__nv_bfloat16 *)output.cudaData, tokens, dim, hcMult, eps, normEps);
    } else if (x.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4HcPreSqSumKernel<<<tokens, threads, threads * sizeof(float)>>>(
            (const half *)x.cudaData, dots, tokens, flatDim, dotsStride);
        DeepSeekV4HcHeadFinishKernel<<<tokens, threads, (hcMult + 1) * sizeof(float)>>>(
            (const half *)x.cudaData, dots,
            (const float *)hcScale.cudaData, (const float *)hcBase.cudaData,
            (half *)output.cudaData, tokens, dim, hcMult, eps, normEps);
    } else if (x.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4HcPreSqSumKernel<<<tokens, threads, threads * sizeof(float)>>>(
            (const float *)x.cudaData, dots, tokens, flatDim, dotsStride);
        DeepSeekV4HcHeadFinishKernel<<<tokens, threads, (hcMult + 1) * sizeof(float)>>>(
            (const float *)x.cudaData, dots,
            (const float *)hcScale.cudaData, (const float *)hcBase.cudaData,
            (float *)output.cudaData, tokens, dim, hcMult, eps, normEps);
    } else {
        ok = false;
    }
    DeviceSync();
    FastllmCudaFree(dots);
    return ok;
}

static bool FastllmCudaDeepSeekV4ScaleQRotaryImpl(fastllm::Data &q, int ropeDim, float ropeBase,
                                                  int startPos, const int32_t *decodeMeta,
                                                  int originalSeqLen, float ropeFactor,
                                                  int betaFast, int betaSlow, float eps) {
    if (q.dataDevice != fastllm::DataDevice::CUDA || q.dims.size() != 4 ||
        ropeDim <= 0 || ropeDim > q.dims[3]) {
        return false;
    }
    int bsz = q.dims[0], seqlen = q.dims[1], heads = q.dims[2], dim = q.dims[3];
    int rows = bsz * seqlen * heads;
    int threads = 256;
    int blocks = rows;
    size_t sharedBytes = (size_t)threads * sizeof(float);
    if (q.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4ScaleQRotaryBlockKernel<<<blocks, threads, sharedBytes>>>(
            (__nv_bfloat16 *)q.cudaData, rows, seqlen, heads, dim, ropeDim, ropeBase, startPos,
            originalSeqLen, ropeFactor, betaFast, betaSlow, eps, decodeMeta);
    } else if (q.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4ScaleQRotaryBlockKernel<<<blocks, threads, sharedBytes>>>(
            (half *)q.cudaData, rows, seqlen, heads, dim, ropeDim, ropeBase, startPos,
            originalSeqLen, ropeFactor, betaFast, betaSlow, eps, decodeMeta);
    } else if (q.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4ScaleQRotaryBlockKernel<<<blocks, threads, sharedBytes>>>(
            (float *)q.cudaData, rows, seqlen, heads, dim, ropeDim, ropeBase, startPos,
            originalSeqLen, ropeFactor, betaFast, betaSlow, eps, decodeMeta);
    } else {
        return false;
    }
    DeviceSync();
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4ScaleQRotary(fastllm::Data &q, int ropeDim, float ropeBase,
                                                    int startPos, int originalSeqLen,
                                                    float ropeFactor, int betaFast, int betaSlow,
                                                    float eps) {
    return FastllmCudaDeepSeekV4ScaleQRotaryImpl(q, ropeDim, ropeBase, startPos, nullptr,
                                                 originalSeqLen, ropeFactor, betaFast, betaSlow, eps);
}

extern "C" bool FastllmCudaDeepSeekV4ScaleQRotaryGraph(
                                                    fastllm::Data &q, int ropeDim, float ropeBase,
                                                    const int32_t *decodeMeta, int originalSeqLen,
                                                    float ropeFactor, int betaFast, int betaSlow,
                                                    float eps) {
    if (decodeMeta == nullptr) {
        return false;
    }
    return FastllmCudaDeepSeekV4ScaleQRotaryImpl(q, ropeDim, ropeBase, 0, decodeMeta,
                                                 originalSeqLen, ropeFactor, betaFast, betaSlow, eps);
}

static bool FastllmCudaDeepSeekV4RotaryQuantImpl(fastllm::Data &x, int ropeDim, float ropeBase,
                                                 int startPos, const int32_t *decodeMeta,
                                                 int originalSeqLen, float ropeFactor,
                                                 int betaFast, int betaSlow, int quantDim,
                                                 int blockSize, int posStep) {
    if (x.dataDevice != fastllm::DataDevice::CUDA || x.dims.size() < 3 || x.dims.size() > 4 ||
        ropeDim <= 0 || blockSize <= 0) {
        return false;
    }
    int bsz = x.dims[0], seqlen = x.dims[1];
    int heads = x.dims.size() == 4 ? x.dims[2] : 1;
    int dim = x.dims.size() == 4 ? x.dims[3] : x.dims[2];
    if (ropeDim > dim || quantDim < 0 || quantDim > dim) {
        return false;
    }
    int rows = bsz * seqlen * heads;
    int threads = 256;
    int blocks = rows;
    size_t sharedBytes = (size_t)threads * sizeof(float);
    if (x.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4RotaryQuantBlockKernel<<<blocks, threads, sharedBytes>>>(
            (__nv_bfloat16 *)x.cudaData, rows, seqlen, heads, dim, ropeDim, ropeBase, startPos,
            originalSeqLen, ropeFactor, betaFast, betaSlow, quantDim, blockSize, posStep, decodeMeta);
    } else if (x.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4RotaryQuantBlockKernel<<<blocks, threads, sharedBytes>>>(
            (half *)x.cudaData, rows, seqlen, heads, dim, ropeDim, ropeBase, startPos,
            originalSeqLen, ropeFactor, betaFast, betaSlow, quantDim, blockSize, posStep, decodeMeta);
    } else if (x.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4RotaryQuantBlockKernel<<<blocks, threads, sharedBytes>>>(
            (float *)x.cudaData, rows, seqlen, heads, dim, ropeDim, ropeBase, startPos,
            originalSeqLen, ropeFactor, betaFast, betaSlow, quantDim, blockSize, posStep, decodeMeta);
    } else {
        return false;
    }
    DeviceSync();
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4RotaryQuant(fastllm::Data &x, int ropeDim, float ropeBase,
                                                   int startPos, int originalSeqLen,
                                                   float ropeFactor, int betaFast, int betaSlow,
                                                   int quantDim, int blockSize, int posStep) {
    return FastllmCudaDeepSeekV4RotaryQuantImpl(x, ropeDim, ropeBase, startPos, nullptr,
                                                originalSeqLen, ropeFactor, betaFast, betaSlow,
                                                quantDim, blockSize, posStep);
}

extern "C" bool FastllmCudaDeepSeekV4RotaryQuantGraph(
                                                   fastllm::Data &x, int ropeDim, float ropeBase,
                                                   const int32_t *decodeMeta, int originalSeqLen,
                                                   float ropeFactor, int betaFast, int betaSlow,
                                                   int quantDim, int blockSize, int posStep) {
    if (decodeMeta == nullptr) {
        return false;
    }
    return FastllmCudaDeepSeekV4RotaryQuantImpl(x, ropeDim, ropeBase, 0, decodeMeta,
                                                originalSeqLen, ropeFactor, betaFast, betaSlow,
                                                quantDim, blockSize, posStep);
}

extern "C" bool FastllmCudaDeepSeekV4FusedQKVRopeCacheGraph(
                                                   fastllm::Data &q, fastllm::Data &kv,
                                                   fastllm::Data &kvNormWeight,
                                                   const int32_t *decodeMeta,
                                                   int ropeDim, float ropeBase,
                                                   int originalSeqLen, float ropeFactor,
                                                   int betaFast, int betaSlow, float eps,
                                                   int quantDim, int quantBlockSize,
                                                   int windowSize, fastllm::Data &windowKV) {
    // Keep this specialization intentionally narrow.  The caller retains the
    // established RMSNorm/rotary/cache-update sequence as the fallback.
    if (decodeMeta == nullptr || q.dataDevice != fastllm::DataDevice::CUDA ||
        kv.dataDevice != fastllm::DataDevice::CUDA ||
        kvNormWeight.dataDevice != fastllm::DataDevice::CUDA ||
        windowKV.dataDevice != fastllm::DataDevice::CUDA ||
        q.dataType != fastllm::DataType::BFLOAT16 ||
        kv.dataType != fastllm::DataType::BFLOAT16 ||
        kvNormWeight.dataType != fastllm::DataType::FLOAT32 ||
        windowKV.dataType != fastllm::DataType::FLOAT32 ||
        q.cudaData == nullptr || kv.cudaData == nullptr ||
        kvNormWeight.cudaData == nullptr || windowKV.cudaData == nullptr ||
        q.dims.size() != 4 || q.dims[0] != 1 || q.dims[1] != 1 ||
        q.dims[2] <= 0 || q.dims[2] > 64 || q.dims[3] != 512 ||
        kv.dims != std::vector<int>({1, 1, 1, 512}) ||
        kvNormWeight.Count(0) != 512 ||
        windowKV.dims != std::vector<int>({1, windowSize, 512}) ||
        ropeDim != 64 || quantDim != 448 || quantBlockSize != 64 ||
        windowSize <= 0 || eps <= 0.0f) {
        return false;
    }
    int qDevice = GetPointerDeviceId(q.cudaData);
    int metaDevice = GetPointerDeviceId((void*)decodeMeta);
    if (qDevice < 0 || (metaDevice >= 0 && metaDevice != qDevice)) {
        return false;
    }

    constexpr int threads = 256;
    int qHeads = q.dims[2];
    int warps = threads / 32;
    int blocks = (qHeads + 1 + warps - 1) / warps;
    DeepSeekV4FusedQKVRopeCache512Kernel<<<blocks, threads>>>(
        (__nv_bfloat16 *)q.cudaData, (__nv_bfloat16 *)kv.cudaData,
        (const float *)kvNormWeight.cudaData, (float *)windowKV.cudaData,
        decodeMeta, windowSize, qHeads, ropeBase, originalSeqLen, ropeFactor,
        betaFast, betaSlow, eps);
    DeviceSync();
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4StoreWindowKVCache(const fastllm::Data &kv, int startPos,
                                                        int windowSize, fastllm::Data &windowKV) {
    if (kv.dataDevice != fastllm::DataDevice::CUDA || kv.dims.size() != 3 || kv.dims[1] <= 0 ||
        startPos < 0 || windowSize <= 0) {
        return false;
    }
    int bsz = kv.dims[0], seqlen = kv.dims[1], headDim = kv.dims[2];
    if (!DeepSeekV4PrepareCudaOutput(windowKV, fastllm::DataType::FLOAT32, {bsz, windowSize, headDim})) {
        return false;
    }
    int total = bsz * windowSize * headDim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (kv.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4StoreWindowKVCacheKernel<<<blocks, threads>>>(
            (__nv_bfloat16 *)kv.cudaData, (float *)windowKV.cudaData, bsz, seqlen, headDim, startPos, windowSize);
    } else if (kv.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4StoreWindowKVCacheKernel<<<blocks, threads>>>(
            (half *)kv.cudaData, (float *)windowKV.cudaData, bsz, seqlen, headDim, startPos, windowSize);
    } else if (kv.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4StoreWindowKVCacheKernel<<<blocks, threads>>>(
            (float *)kv.cudaData, (float *)windowKV.cudaData, bsz, seqlen, headDim, startPos, windowSize);
    } else {
        return false;
    }
    DeviceSync();
    return true;
}

static bool FastllmCudaDeepSeekV4UpdateWindowKVCacheImpl(const fastllm::Data &kv, int startPos,
                                                         const int32_t *decodeMeta, int windowSize,
                                                         fastllm::Data &windowKV) {
    if (kv.dataDevice != fastllm::DataDevice::CUDA || kv.dims.size() != 3 || kv.dims[1] <= 0 ||
        startPos < 0 || windowSize <= 0) {
        return false;
    }
    int bsz = kv.dims[0], seqlen = kv.dims[1], headDim = kv.dims[2];
    if (!DeepSeekV4PrepareCudaOutput(windowKV, fastllm::DataType::FLOAT32, {bsz, windowSize, headDim})) {
        return false;
    }
    int total = bsz * seqlen * headDim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (kv.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4UpdateWindowKVCacheKernel<<<blocks, threads>>>(
            (__nv_bfloat16 *)kv.cudaData, (float *)windowKV.cudaData, bsz, seqlen, headDim,
            startPos, windowSize, decodeMeta);
    } else if (kv.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4UpdateWindowKVCacheKernel<<<blocks, threads>>>(
            (half *)kv.cudaData, (float *)windowKV.cudaData, bsz, seqlen, headDim,
            startPos, windowSize, decodeMeta);
    } else if (kv.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4UpdateWindowKVCacheKernel<<<blocks, threads>>>(
            (float *)kv.cudaData, (float *)windowKV.cudaData, bsz, seqlen, headDim,
            startPos, windowSize, decodeMeta);
    } else {
        return false;
    }
    DeviceSync();
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4UpdateWindowKVCache(const fastllm::Data &kv, int startPos,
                                                           int windowSize, fastllm::Data &windowKV) {
    return FastllmCudaDeepSeekV4UpdateWindowKVCacheImpl(kv, startPos, nullptr, windowSize, windowKV);
}

extern "C" bool FastllmCudaDeepSeekV4UpdateWindowKVCacheGraph(
                                                           const fastllm::Data &kv,
                                                           const int32_t *decodeMeta,
                                                           int windowSize, fastllm::Data &windowKV) {
    if (decodeMeta == nullptr) {
        return false;
    }
    return FastllmCudaDeepSeekV4UpdateWindowKVCacheImpl(kv, 0, decodeMeta, windowSize, windowKV);
}

extern "C" bool FastllmCudaDeepSeekV4BuildWindowKVPrefix(const fastllm::Data &windowKV, int startPos,
                                                         int windowSize, int prefixLen,
                                                         fastllm::Data &output) {
    if (windowKV.dataDevice != fastllm::DataDevice::CUDA ||
        windowKV.dataType != fastllm::DataType::FLOAT32 ||
        windowKV.dims.size() != 3 || prefixLen <= 0 || startPos < prefixLen ||
        windowSize <= 0 || windowKV.dims[1] != windowSize) {
        return false;
    }
    int bsz = windowKV.dims[0], headDim = windowKV.dims[2];
    if (!DeepSeekV4PrepareCudaOutput(output, fastllm::DataType::FLOAT32, {bsz, prefixLen, headDim})) {
        return false;
    }
    int total = bsz * prefixLen * headDim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    DeepSeekV4BuildWindowKVPrefixKernel<<<blocks, threads>>>(
        (const float *)windowKV.cudaData, (float *)output.cudaData,
        bsz, prefixLen, headDim, startPos, windowSize);
    DeviceSync();
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4BuildCompressedKV(const fastllm::Data &kv,
                                                       const fastllm::Data &score,
                                                       const fastllm::Data &ape,
                                                       int rawTokenBase, int rawLen,
                                                       int blockStart, int blockCount,
                                                       int compressRatio, int headDim,
                                                       int wideDim, bool overlap,
                                                       fastllm::Data &output) {
    if (kv.dataDevice != fastllm::DataDevice::CUDA ||
        score.dataDevice != fastllm::DataDevice::CUDA ||
        ape.dataDevice != fastllm::DataDevice::CUDA ||
        ape.dataType != fastllm::DataType::FLOAT32 ||
        score.dataType != kv.dataType ||
        kv.dims.size() != 3 || score.dims != kv.dims ||
        rawLen <= 0 || blockCount <= 0 || compressRatio <= 0 ||
        headDim <= 0 || wideDim <= 0 || kv.dims[1] < rawLen ||
        kv.dims[2] != wideDim || rawTokenBase < 0) {
        return false;
    }
    int bsz = kv.dims[0];
    if (ape.Count(0) < (uint64_t)compressRatio * wideDim) {
        return false;
    }
    int firstNeededToken = blockStart * compressRatio;
    if (overlap && blockStart > 0) {
        firstNeededToken = (blockStart - 1) * compressRatio;
    }
    int lastNeededToken = (blockStart + blockCount) * compressRatio;
    if (rawTokenBase > firstNeededToken || rawTokenBase + rawLen < lastNeededToken) {
        return false;
    }
    if (!DeepSeekV4PrepareCudaOutput(output, fastllm::DataType::FLOAT32, {bsz, blockCount, headDim})) {
        return false;
    }
    uint64_t total = (uint64_t)bsz * blockCount * headDim;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (kv.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4BuildCompressedKVKernel<<<blocks, threads>>>(
            (const __nv_bfloat16 *)kv.cudaData, (const __nv_bfloat16 *)score.cudaData,
            (const float *)ape.cudaData, (float *)output.cudaData, bsz, rawTokenBase, rawLen,
            blockStart, blockCount, compressRatio, headDim, wideDim, overlap);
    } else if (kv.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4BuildCompressedKVKernel<<<blocks, threads>>>(
            (const half *)kv.cudaData, (const half *)score.cudaData,
            (const float *)ape.cudaData, (float *)output.cudaData, bsz, rawTokenBase, rawLen,
            blockStart, blockCount, compressRatio, headDim, wideDim, overlap);
    } else if (kv.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4BuildCompressedKVKernel<<<blocks, threads>>>(
            (const float *)kv.cudaData, (const float *)score.cudaData,
            (const float *)ape.cudaData, (float *)output.cudaData, bsz, rawTokenBase, rawLen,
            blockStart, blockCount, compressRatio, headDim, wideDim, overlap);
    } else {
        return false;
    }
    DeviceSync();
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4InitGraphRawRing(const fastllm::Data &raw,
                                                       int rawTokenBase,
                                                       fastllm::Data &ring) {
    if (raw.dataDevice != fastllm::DataDevice::CUDA ||
        ring.dataDevice != fastllm::DataDevice::CUDA ||
        raw.cudaData == nullptr || ring.cudaData == nullptr ||
        raw.dataType != ring.dataType || raw.dims.size() != 3 ||
        ring.dims.size() != 3 || raw.dims[0] != ring.dims[0] ||
        raw.dims[2] != ring.dims[2] || rawTokenBase < 0 || ring.dims[1] <= 0) {
        return false;
    }
    int bsz = raw.dims[0], rawLen = raw.dims[1], wideDim = raw.dims[2];
    int ringCapacity = ring.dims[1];
    uint64_t total = (uint64_t)bsz * rawLen * wideDim;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (raw.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4InitGraphRawRingKernel<<<blocks, threads>>>(
            (const __nv_bfloat16 *)raw.cudaData, (__nv_bfloat16 *)ring.cudaData,
            bsz, rawLen, wideDim, rawTokenBase, ringCapacity);
    } else if (raw.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4InitGraphRawRingKernel<<<blocks, threads>>>(
            (const half *)raw.cudaData, (half *)ring.cudaData,
            bsz, rawLen, wideDim, rawTokenBase, ringCapacity);
    } else if (raw.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4InitGraphRawRingKernel<<<blocks, threads>>>(
            (const float *)raw.cudaData, (float *)ring.cudaData,
            bsz, rawLen, wideDim, rawTokenBase, ringCapacity);
    } else {
        return false;
    }
    DeviceSync();
    return true;
}

template <typename T>
static bool DeepSeekV4LaunchUpdateCompressedKVGraph(
                                            const fastllm::Data &kv,
                                            const fastllm::Data &score,
                                            const fastllm::Data &ape,
                                            const fastllm::Data &normWeight,
                                            const int32_t *decodeMeta,
                                            int compressRatio, int headDim,
                                            int ropeDim, float ropeBase,
                                            int originalSeqLen, float ropeFactor,
                                            int betaFast, int betaSlow,
                                            fastllm::Data &kvRing,
                                            fastllm::Data &scoreRing,
                                            fastllm::Data &compressedKV,
                                            int bsz, int ringCapacity,
                                            int compressedCapacity, int wideDim) {
    uint64_t rawTotal = (uint64_t)bsz * wideDim;
    int threads = 256;
    DeepSeekV4StoreGraphCompressorRawKernel<<<
        (int)((rawTotal + threads - 1) / threads), threads>>>(
            (const T *)kv.cudaData, (const T *)score.cudaData,
            (T *)kvRing.cudaData, (T *)scoreRing.cudaData,
            decodeMeta, bsz, wideDim, ringCapacity);

    size_t sharedBytes = (size_t)(headDim + threads) * sizeof(float);
    if (normWeight.dataType == fastllm::DataType::BFLOAT16) {
        DeepSeekV4UpdateCompressedKVGraphKernel<<<bsz, threads, sharedBytes>>>(
            (const T *)kvRing.cudaData, (const T *)scoreRing.cudaData,
            (const float *)ape.cudaData, (const __nv_bfloat16 *)normWeight.cudaData,
            decodeMeta, (__nv_bfloat16 *)compressedKV.cudaData,
            bsz, ringCapacity, compressedCapacity, compressRatio, headDim,
            wideDim, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
    } else if (normWeight.dataType == fastllm::DataType::FLOAT16) {
        DeepSeekV4UpdateCompressedKVGraphKernel<<<bsz, threads, sharedBytes>>>(
            (const T *)kvRing.cudaData, (const T *)scoreRing.cudaData,
            (const float *)ape.cudaData, (const half *)normWeight.cudaData,
            decodeMeta, (__nv_bfloat16 *)compressedKV.cudaData,
            bsz, ringCapacity, compressedCapacity, compressRatio, headDim,
            wideDim, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
    } else if (normWeight.dataType == fastllm::DataType::FLOAT32) {
        DeepSeekV4UpdateCompressedKVGraphKernel<<<bsz, threads, sharedBytes>>>(
            (const T *)kvRing.cudaData, (const T *)scoreRing.cudaData,
            (const float *)ape.cudaData, (const float *)normWeight.cudaData,
            decodeMeta, (__nv_bfloat16 *)compressedKV.cudaData,
            bsz, ringCapacity, compressedCapacity, compressRatio, headDim,
            wideDim, ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
    } else {
        return false;
    }
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4UpdateCompressedKVGraph(
                                            const fastllm::Data &kv,
                                            const fastllm::Data &score,
                                            const fastllm::Data &ape,
                                            const fastllm::Data &normWeight,
                                            const int32_t *decodeMeta,
                                            int compressRatio, int headDim,
                                            int ropeDim, float ropeBase,
                                            int originalSeqLen, float ropeFactor,
                                            int betaFast, int betaSlow,
                                            fastllm::Data &kvRing,
                                            fastllm::Data &scoreRing,
                                            fastllm::Data &compressedKV) {
    if (decodeMeta == nullptr || compressRatio <= 0 || headDim <= 0 ||
        ropeDim <= 0 || ropeDim > headDim || kv.dataDevice != fastllm::DataDevice::CUDA ||
        score.dataDevice != fastllm::DataDevice::CUDA ||
        ape.dataDevice != fastllm::DataDevice::CUDA ||
        normWeight.dataDevice != fastllm::DataDevice::CUDA ||
        kvRing.dataDevice != fastllm::DataDevice::CUDA ||
        scoreRing.dataDevice != fastllm::DataDevice::CUDA ||
        compressedKV.dataDevice != fastllm::DataDevice::CUDA ||
        kv.cudaData == nullptr || score.cudaData == nullptr || ape.cudaData == nullptr ||
        normWeight.cudaData == nullptr || kvRing.cudaData == nullptr ||
        scoreRing.cudaData == nullptr || compressedKV.cudaData == nullptr ||
        kv.dataType != score.dataType || kv.dataType != kvRing.dataType ||
        kv.dataType != scoreRing.dataType || ape.dataType != fastllm::DataType::FLOAT32 ||
        compressedKV.dataType != fastllm::DataType::BFLOAT16 ||
        kv.dims.size() != 3 || score.dims != kv.dims ||
        kvRing.dims.size() != 3 || scoreRing.dims != kvRing.dims ||
        compressedKV.dims.size() != 3) {
        return false;
    }
    int bsz = kv.dims[0];
    int wideDim = (compressRatio == 4 ? 2 : 1) * headDim;
    int ringCapacity = kvRing.dims[1];
    int compressedCapacity = compressedKV.dims[1];
    if (compressedKV.expansionDims.size() >= 3) {
        compressedCapacity = std::max(compressedCapacity, compressedKV.expansionDims[1]);
    }
    if (kv.dims[1] != 1 || kv.dims[2] != wideDim ||
        kvRing.dims[0] != bsz || kvRing.dims[2] != wideDim ||
        ringCapacity < (compressRatio == 4 ? 2 * compressRatio : compressRatio) ||
        compressedCapacity <= 0 || compressedKV.dims[0] != bsz ||
        compressedKV.dims[2] != headDim ||
        ape.Count(0) < (uint64_t)compressRatio * wideDim ||
        normWeight.Count(0) < (uint64_t)headDim) {
        return false;
    }
    bool ok = false;
    if (kv.dataType == fastllm::DataType::BFLOAT16) {
        ok = DeepSeekV4LaunchUpdateCompressedKVGraph<__nv_bfloat16>(
            kv, score, ape, normWeight, decodeMeta, compressRatio, headDim,
            ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow,
            kvRing, scoreRing, compressedKV, bsz, ringCapacity,
            compressedCapacity, wideDim);
    } else if (kv.dataType == fastllm::DataType::FLOAT16) {
        ok = DeepSeekV4LaunchUpdateCompressedKVGraph<half>(
            kv, score, ape, normWeight, decodeMeta, compressRatio, headDim,
            ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow,
            kvRing, scoreRing, compressedKV, bsz, ringCapacity,
            compressedCapacity, wideDim);
    } else if (kv.dataType == fastllm::DataType::FLOAT32) {
        ok = DeepSeekV4LaunchUpdateCompressedKVGraph<float>(
            kv, score, ape, normWeight, decodeMeta, compressRatio, headDim,
            ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow,
            kvRing, scoreRing, compressedKV, bsz, ringCapacity,
            compressedCapacity, wideDim);
    }
    DeviceSync();
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV4RouteScoreTransform(fastllm::Data &logits, int scoreFuncMode) {
    if (logits.dataDevice != fastllm::DataDevice::CUDA || logits.dataType != fastllm::DataType::FLOAT32 ||
        logits.dims.empty() || scoreFuncMode < 0 || scoreFuncMode > 2) {
        return false;
    }
    int experts = logits.dims.back();
    int rows = (int)(logits.Count(0) / experts);
    if (experts > 256) {
        return false;
    }
    DeepSeekV4RouteScoreTransformKernel<<<rows, 256>>>((float *)logits.cudaData, rows, experts, scoreFuncMode);
    DeviceSync();
    return true;
}

void FastllmCudaReleaseDeepSeekV4RouteTableCache(
        const fastllm::Data *routeTable) {
    if (routeTable == nullptr) {
        return;
    }

    DeepSeekV4RouteTableDeviceCache retired;
    {
        std::lock_guard<std::mutex> guard(DeepSeekV4RouteTableCacheMutex());
        DeepSeekV4RouteTableCache &cache = DeepSeekV4RouteTableCaches();
        auto it = cache.find(routeTable);
        if (it == cache.end()) {
            return;
        }
        retired = std::move(it->second);
        cache.erase(it);
    }

    int originalDevice = FastllmCudaGetDevice();
    for (auto &deviceEntry : retired) {
        FastllmCudaSetDevice(deviceEntry.first);
        DeepSeekV4RouteTableCacheEntry &entry = deviceEntry.second;
        FastllmCudaFree(entry.cudaData);
        for (void *ptr : entry.retiredCudaData) {
            FastllmCudaFree(ptr);
        }
    }
    FastllmCudaSetDevice(originalDevice);
}

extern "C" bool FastllmCudaDeepSeekV4HashRouteScore(const fastllm::Data &logits,
                                                    fastllm::Data &tid2eid,
                                                    const int *inputIds, int tokens,
                                                    int topk, int scoreFuncMode,
                                                    float routeScale,
                                                    fastllm::Data &expertIndex,
                                                    fastllm::Data &expertScore) {
    if (logits.dataDevice != fastllm::DataDevice::CUDA ||
        logits.dataType != fastllm::DataType::FLOAT32 ||
        logits.dims.empty() || inputIds == nullptr ||
        tokens <= 0 || topk <= 0 || scoreFuncMode < 0 || scoreFuncMode > 2 ||
        (tid2eid.dataType != fastllm::DataType::FLOAT32 &&
         tid2eid.dataType != fastllm::DataType::INT32 &&
         tid2eid.dataType != fastllm::DataType::INT32PARAM)) {
        return false;
    }
    int experts = logits.dims.back();
    int rows = (int)(logits.Count(0) / experts);
    if (rows != tokens || experts <= 0 || experts > 256) {
        return false;
    }
    int maxInputId = -1;
    for (int i = 0; i < tokens; i++) {
        if (inputIds[i] < 0) {
            return false;
        }
        maxInputId = std::max(maxInputId, inputIds[i]);
    }
    if (tid2eid.Count(0) < (uint64_t)(maxInputId + 1) * topk) {
        return false;
    }
    const void *cudaRouteTable = DeepSeekV4GetCudaRouteTable(tid2eid);
    bool intRouteTable = tid2eid.dataType == fastllm::DataType::INT32 ||
                         tid2eid.dataType == fastllm::DataType::INT32PARAM;
    if (cudaRouteTable == nullptr ||
        !DeepSeekV4PrepareCudaOutput(expertIndex, fastllm::DataType::INT32, {tokens, topk}) ||
        !DeepSeekV4PrepareCudaOutput(expertScore, fastllm::DataType::FLOAT32, {tokens, topk})) {
        return false;
    }

    int *cudaInputIds = nullptr;
    int singleTokenId = tokens == 1 ? inputIds[0] : -1;
    if (tokens > 1) {
        size_t inputBytes = (size_t)tokens * sizeof(int);
        cudaInputIds = (int *)FastllmCudaMalloc(inputBytes);
        if (cudaInputIds == nullptr) {
            return false;
        }
        FastllmCudaCopyFromHostToDevice(cudaInputIds, (void *)inputIds, inputBytes);
    }
    if (intRouteTable) {
        DeepSeekV4HashRouteScoreKernel<<<tokens, 256>>>(
            (float *)logits.cudaData, (const int32_t *)cudaRouteTable,
            cudaInputIds, singleTokenId,
            (int32_t *)expertIndex.cudaData, (float *)expertScore.cudaData,
            tokens, experts, topk, scoreFuncMode, routeScale,
            (int)(tid2eid.Count(0) / topk));
    } else {
        DeepSeekV4HashRouteScoreKernel<<<tokens, 256>>>(
            (float *)logits.cudaData, (const float *)cudaRouteTable,
            cudaInputIds, singleTokenId,
            (int32_t *)expertIndex.cudaData, (float *)expertScore.cudaData,
            tokens, experts, topk, scoreFuncMode, routeScale,
            (int)(tid2eid.Count(0) / topk));
    }
    DeviceSync();
    if (cudaInputIds != nullptr) {
        FastllmCudaFree(cudaInputIds);
    }
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4HashRouteScoreGraph(
                                                    const fastllm::Data &logits,
                                                    fastllm::Data &tid2eid,
                                                    const int32_t *decodeMeta,
                                                    int tokens, int topk,
                                                    int scoreFuncMode,
                                                    float routeScale,
                                                    fastllm::Data &expertIndex,
                                                    fastllm::Data &expertScore) {
    if (decodeMeta == nullptr || tokens != 1 ||
        logits.dataDevice != fastllm::DataDevice::CUDA ||
        logits.dataType != fastllm::DataType::FLOAT32 || logits.dims.empty() ||
        topk <= 0 || scoreFuncMode < 0 || scoreFuncMode > 2 ||
        tid2eid.Count(0) < (uint64_t)topk ||
        (tid2eid.dataType != fastllm::DataType::FLOAT32 &&
         tid2eid.dataType != fastllm::DataType::INT32 &&
         tid2eid.dataType != fastllm::DataType::INT32PARAM)) {
        return false;
    }
    int experts = logits.dims.back();
    if ((int)(logits.Count(0) / experts) != tokens || experts <= 0 || experts > 256) {
        return false;
    }

    const void *cudaRouteTable = DeepSeekV4GetCudaRouteTable(tid2eid);
    bool intRouteTable = tid2eid.dataType == fastllm::DataType::INT32 ||
                         tid2eid.dataType == fastllm::DataType::INT32PARAM;
    if (cudaRouteTable == nullptr ||
        !DeepSeekV4PrepareCudaOutput(expertIndex, fastllm::DataType::INT32, {tokens, topk}) ||
        !DeepSeekV4PrepareCudaOutput(expertScore, fastllm::DataType::FLOAT32, {tokens, topk})) {
        return false;
    }

    int routeRows = (int)(tid2eid.Count(0) / topk);
    const int *cudaInputIds = (const int *)(decodeMeta + 1);
    if (intRouteTable) {
        DeepSeekV4HashRouteScoreKernel<<<tokens, 256>>>(
            (float *)logits.cudaData, (const int32_t *)cudaRouteTable,
            cudaInputIds, -1, (int32_t *)expertIndex.cudaData,
            (float *)expertScore.cudaData, tokens, experts, topk,
            scoreFuncMode, routeScale, routeRows);
    } else {
        DeepSeekV4HashRouteScoreKernel<<<tokens, 256>>>(
            (float *)logits.cudaData, (const float *)cudaRouteTable,
            cudaInputIds, -1, (int32_t *)expertIndex.cudaData,
            (float *)expertScore.cudaData, tokens, experts, topk,
            scoreFuncMode, routeScale, routeRows);
    }
    DeviceSync();
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4SparseAttentionPrefill(const fastllm::Data &q,
                                                            const fastllm::Data &kv,
                                                            fastllm::Data &attnSink,
                                                            int windowSize, int startPos,
                                                            int compressRatio,
                                                            int ropeDim, float ropeBase,
                                                            int originalSeqLen,
                                                            float ropeFactor, int betaFast,
                                                            int betaSlow, float softmaxScale,
                                                            fastllm::Data &output, int prefixLen) {
    if (q.dataDevice != fastllm::DataDevice::CUDA ||
        kv.dataDevice != fastllm::DataDevice::CUDA ||
        attnSink.dataDevice != fastllm::DataDevice::CUDA ||
        attnSink.dataType != fastllm::DataType::FLOAT32 ||
        q.dims.size() != 4 || kv.dims.size() != 3 ||
        windowSize <= 0 || ropeDim <= 0 || ropeDim > q.dims[3]) {
        return false;
    }
    int bsz = q.dims[0], seqlen = q.dims[1], heads = q.dims[2], dim = q.dims[3];
    int kvLen = kv.dims[1];
    int realPrefixLen = std::max(0, std::min(prefixLen, kvLen - seqlen));
    if (seqlen <= 0 || kv.dims[0] != bsz || kv.dims[2] != dim ||
        kvLen < realPrefixLen + seqlen || attnSink.Count(0) != (uint64_t)heads) {
        return false;
    }
    int compressedCount = std::max(0, kvLen - realPrefixLen - seqlen);
    int maxKeys = std::min(windowSize, realPrefixLen + seqlen) + (compressRatio > 0 ? compressedCount : 0);
    if (maxKeys <= 0 || maxKeys > kDeepSeekV4SparsePrefillMaxKeys) {
        return false;
    }
    if (!DeepSeekV4PrepareCudaOutput(output, fastllm::DataType::BFLOAT16,
                                     {bsz, seqlen, heads, dim})) {
        return false;
    }
    if (DeepSeekV4RunSparsePrefillLocalCublas(
            q, kv, (const float *)attnSink.cudaData, output,
            bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos, realPrefixLen,
            ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow, softmaxScale)) {
        return true;
    }
    if (DeepSeekV4RunSparsePrefillCompressedCublas(
            q, kv, (const float *)attnSink.cudaData, output,
            bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos, realPrefixLen,
            ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow, softmaxScale)) {
        return true;
    }

    int rows = bsz * seqlen * heads;
    size_t rowBytes = (size_t)dim * sizeof(float);
    size_t maxTempBytes = std::max(rowBytes, DeepSeekV4SparsePrefillTempBytesLimit());
    int rowsPerChunk = (int)std::max<size_t>(1, maxTempBytes / rowBytes);
    if (rowsPerChunk >= heads) {
        rowsPerChunk = std::max(heads, (rowsPerChunk / heads) * heads);
    }
    rowsPerChunk = std::min(rowsPerChunk, rows);
    size_t tempBytes = (size_t)rowsPerChunk * rowBytes;
    float *cudaTemp = (float *)FastllmCudaMalloc(tempBytes);
    if (cudaTemp == nullptr) {
        return false;
    }

    bool ok = true;
    for (int rowOffset = 0; rowOffset < rows; rowOffset += rowsPerChunk) {
        int chunkRows = std::min(rowsPerChunk, rows - rowOffset);
        bool launched = false;
        if (q.dataType == fastllm::DataType::BFLOAT16) {
            launched = DeepSeekV4LaunchSparsePrefillByQ<__nv_bfloat16>(
                q, kv, (const float *)attnSink.cudaData, cudaTemp,
                bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos, realPrefixLen,
                softmaxScale, rowOffset, chunkRows);
        } else if (q.dataType == fastllm::DataType::FLOAT16) {
            launched = DeepSeekV4LaunchSparsePrefillByQ<half>(
                q, kv, (const float *)attnSink.cudaData, cudaTemp,
                bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos, realPrefixLen,
                softmaxScale, rowOffset, chunkRows);
        } else if (q.dataType == fastllm::DataType::FLOAT32) {
            launched = DeepSeekV4LaunchSparsePrefillByQ<float>(
                q, kv, (const float *)attnSink.cudaData, cudaTemp,
                bsz, seqlen, heads, dim, kvLen, windowSize, compressRatio, startPos, realPrefixLen,
                softmaxScale, rowOffset, chunkRows);
        }
        if (!launched) {
            ok = false;
            break;
        }
        int threads = 128;
        int blocks = (chunkRows + threads - 1) / threads;
        DeepSeekV4SparsePrefillRotaryCastKernel<<<blocks, threads>>>(
            cudaTemp, (__nv_bfloat16 *)output.cudaData, chunkRows, rowOffset, seqlen, heads, dim,
            ropeDim, ropeBase, startPos, originalSeqLen, ropeFactor, betaFast, betaSlow);
    }
    if (ok) {
        DeviceSync();
    }

    FastllmCudaFree(cudaTemp);
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV4SparseAttentionDecodeCached(const fastllm::Data &q,
                                                                 const fastllm::Data &windowKV,
                                                                 const fastllm::Data &compressedKV,
                                                                 fastllm::Data &attnSink,
                                                                 int windowSize, int startPos,
                                                                 int compressedCount,
                                                                 int ropeDim, float ropeBase,
                                                                 int originalSeqLen,
                                                                 float ropeFactor, int betaFast,
                                                                 int betaSlow, float softmaxScale,
                                                                 fastllm::Data &output) {
    if (q.dims.size() != 4 || q.dims[1] != 1 ||
        windowSize <= 0 || compressedCount < 0 ||
        windowSize + compressedCount > kDeepSeekV4SparseDecodeMaxKeys ||
        ropeDim <= 0 || ropeDim > q.dims[3] ||
        q.dataDevice != fastllm::DataDevice::CUDA ||
        windowKV.dataDevice != fastllm::DataDevice::CUDA ||
        windowKV.dataType != fastllm::DataType::FLOAT32 ||
        attnSink.dataDevice != fastllm::DataDevice::CUDA ||
        attnSink.dataType != fastllm::DataType::FLOAT32 ||
        (compressedCount > 0 && compressedKV.dataDevice != fastllm::DataDevice::CUDA)) {
        return false;
    }
    int bsz = q.dims[0], heads = q.dims[2], dim = q.dims[3];
    if (attnSink.Count(0) != (uint64_t)heads ||
        windowKV.Count(0) != (uint64_t)bsz * windowSize * dim ||
        (compressedCount > 0 && compressedKV.Count(0) != (uint64_t)bsz * compressedCount * dim)) {
        return false;
    }
    int curDevice = FastllmCudaGetDevice();
    int qDevice = q.cudaData == nullptr ? -1 : GetPointerDeviceId(q.cudaData);
    int windowDevice = windowKV.cudaData == nullptr ? -1 : GetPointerDeviceId(windowKV.cudaData);
    int compressedDevice = compressedCount > 0 && compressedKV.cudaData != nullptr ?
                           GetPointerDeviceId(compressedKV.cudaData) : qDevice;
    int sinkDevice = attnSink.cudaData == nullptr ? -1 : GetPointerDeviceId(attnSink.cudaData);
    int kernelDevice = qDevice >= 0 ? qDevice : curDevice;
    if ((qDevice >= 0 && qDevice != curDevice) ||
        (windowDevice >= 0 && windowDevice != kernelDevice) ||
        (compressedCount > 0 && compressedDevice >= 0 && compressedDevice != kernelDevice) ||
        (sinkDevice >= 0 && sinkDevice != kernelDevice)) {
        return false;
    }
    const float *cudaWindow = (const float *)windowKV.cudaData;

    if (!DeepSeekV4PrepareCudaOutput(output, fastllm::DataType::BFLOAT16, {bsz, 1, heads, dim})) {
        return false;
    }

    size_t tempBytes = (size_t)bsz * heads * dim * sizeof(float);
    float *cudaTemp = (float *)FastllmCudaMalloc(tempBytes);

    bool ok = false;
    bool sparseDecodeStats = std::getenv("FASTLLM_DSV4_SPARSE_DECODE_STATS") != nullptr;
    auto sparseDecodeStart = std::chrono::steady_clock::now();
    int sparseDecodeMode = 0;
    if (q.dataType == fastllm::DataType::BFLOAT16) {
        ok = DeepSeekV4LaunchSparseDecodeByCompressed<__nv_bfloat16>(
            q.cudaData, compressedCount > 0 ? compressedKV.cudaData : nullptr, compressedKV.dataType,
            cudaWindow, (const float *)attnSink.cudaData, cudaTemp,
            bsz, heads, dim, windowSize, startPos, compressedCount, softmaxScale, &sparseDecodeMode);
    } else if (q.dataType == fastllm::DataType::FLOAT16) {
        ok = DeepSeekV4LaunchSparseDecodeByCompressed<half>(
            q.cudaData, compressedCount > 0 ? compressedKV.cudaData : nullptr, compressedKV.dataType,
            cudaWindow, (const float *)attnSink.cudaData, cudaTemp,
            bsz, heads, dim, windowSize, startPos, compressedCount, softmaxScale, &sparseDecodeMode);
    } else if (q.dataType == fastllm::DataType::FLOAT32) {
        ok = DeepSeekV4LaunchSparseDecodeByCompressed<float>(
            q.cudaData, compressedCount > 0 ? compressedKV.cudaData : nullptr, compressedKV.dataType,
            cudaWindow, (const float *)attnSink.cudaData, cudaTemp,
            bsz, heads, dim, windowSize, startPos, compressedCount, softmaxScale, &sparseDecodeMode);
    }
    if (ok) {
        int rows = bsz * heads;
        int threads = 128;
        int rotaryWorkPerRow = dim - ropeDim + (ropeDim >> 1);
        int blocks = (rows * rotaryWorkPerRow + threads - 1) / threads;
        DeepSeekV4SparseDecodeRotaryCastKernel<<<blocks, threads>>>(
            cudaTemp, (__nv_bfloat16 *)output.cudaData, rows, dim, ropeDim, ropeBase,
            startPos, originalSeqLen, ropeFactor, betaFast, betaSlow, nullptr);
        DeviceSync();
    }
    if (sparseDecodeStats) {
        auto sparseDecodeEnd = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(sparseDecodeEnd - sparseDecodeStart).count();
        int liveWindow = startPos >= windowSize - 1 ? windowSize : (startPos + 1);
        int kvPasses = sparseDecodeMode == 0 ? 2 : 1;
        double windowBytes = (double)bsz * heads * dim * liveWindow * sizeof(float) * kvPasses;
        double compressedBytes = compressedCount > 0
                                     ? (double)bsz * heads * fastllm::GetDataBytes(compressedKV.dataType,
                                                                                   compressedCount, dim) * kvPasses
                                     : 0.0;
        double kvGB = (windowBytes + compressedBytes) / 1.0e9;
        static std::mutex statMutex;
        static unsigned long long statCalls = 0;
        static double statMs = 0.0;
        static double statKvGB = 0.0;
        std::lock_guard<std::mutex> statLock(statMutex);
        statCalls++;
        statMs += ms;
        statKvGB += kvGB;
        int printEvery = 43;
        if (const char *envEvery = std::getenv("FASTLLM_DSV4_SPARSE_DECODE_STATS_EVERY")) {
            printEvery = std::max(1, std::atoi(envEvery));
        }
        if (statCalls % (unsigned long long)printEvery == 0) {
            double avgMs = statMs / printEvery;
            double avgKvGB = statKvGB / printEvery;
            double avgGBps = avgMs > 0.0 ? avgKvGB / (avgMs / 1000.0) : 0.0;
            std::fprintf(stderr,
                         "DeepSeekV4SparseDecodeCachedStats calls=%llu last_ms=%.4f avg_ms=%.4f "
                         "last_kv_gb=%.6f avg_kv_gb=%.6f avg_kv_gbps=%.2f startPos=%d "
                         "liveWindow=%d compressedCount=%d dim=%d heads=%d kvPasses=%d mode=%s\n",
                         statCalls, ms, avgMs, kvGB, avgKvGB, avgGBps, startPos,
                         liveWindow, compressedCount, dim, heads, kvPasses,
                         sparseDecodeMode == 2 ? "cublas" : (sparseDecodeMode == 1 ? "online" : "two-pass"));
            std::fflush(stderr);
            statMs = 0.0;
            statKvGB = 0.0;
        }
    }

    FastllmCudaFree(cudaTemp);
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV4SparseAttentionDecodeCachedGraph(
                                                                 const fastllm::Data &q,
                                                                 const fastllm::Data &windowKV,
                                                                 const fastllm::Data &compressedKV,
                                                                 fastllm::Data &attnSink,
                                                                 int windowSize, int compressRatio,
                                                                 const int32_t *decodeMeta,
                                                                 int ropeDim, float ropeBase,
                                                                 int originalSeqLen,
                                                                 float ropeFactor, int betaFast,
                                                                 int betaSlow, float softmaxScale,
                                                                 fastllm::Data &output,
                                                                 bool allowTriton) {
    if (decodeMeta == nullptr || q.dims.size() != 4 || q.dims[1] != 1 ||
        q.dataDevice != fastllm::DataDevice::CUDA ||
        windowKV.dataDevice != fastllm::DataDevice::CUDA ||
        windowKV.dataType != fastllm::DataType::FLOAT32 ||
        compressedKV.dataDevice != fastllm::DataDevice::CUDA ||
        compressedKV.cudaData == nullptr ||
        attnSink.dataDevice != fastllm::DataDevice::CUDA ||
        attnSink.dataType != fastllm::DataType::FLOAT32 ||
        windowSize <= 0 || compressRatio < 0 || ropeDim <= 0 ||
        ropeDim > q.dims[3]) {
        return false;
    }
    int bsz = q.dims[0], heads = q.dims[2], dim = q.dims[3];
    int compressedCapacity = compressedKV.dims.size() >= 3 ? compressedKV.dims[1] : 0;
    if (compressedKV.expansionDims.size() >= 3) {
        compressedCapacity = std::max(compressedCapacity, compressedKV.expansionDims[1]);
    }
    if (attnSink.Count(0) != (uint64_t)heads ||
        windowKV.Count(0) != (uint64_t)bsz * windowSize * dim ||
        compressedCapacity <= 0 || compressedKV.dims.size() < 3 ||
        compressedKV.dims[0] != bsz || compressedKV.dims[2] != dim ||
        windowSize + compressedCapacity > kDeepSeekV4SparseDecodeMaxKeys) {
        return false;
    }

    int curDevice = FastllmCudaGetDevice();
    int qDevice = q.cudaData == nullptr ? -1 : GetPointerDeviceId(q.cudaData);
    int windowDevice = windowKV.cudaData == nullptr ? -1 : GetPointerDeviceId(windowKV.cudaData);
    int compressedDevice = GetPointerDeviceId(compressedKV.cudaData);
    int sinkDevice = attnSink.cudaData == nullptr ? -1 : GetPointerDeviceId(attnSink.cudaData);
    int metaDevice = GetPointerDeviceId((void*)decodeMeta);
    int kernelDevice = qDevice >= 0 ? qDevice : curDevice;
    if ((qDevice >= 0 && qDevice != curDevice) ||
        (windowDevice >= 0 && windowDevice != kernelDevice) ||
        (compressedDevice >= 0 && compressedDevice != kernelDevice) ||
        (sinkDevice >= 0 && sinkDevice != kernelDevice) ||
        (metaDevice >= 0 && metaDevice != kernelDevice)) {
        return false;
    }
    if (!DeepSeekV4PrepareCudaOutput(output, fastllm::DataType::BFLOAT16,
                                     {bsz, 1, heads, dim})) {
        return false;
    }

    size_t tempBytes = (size_t)bsz * heads * dim * sizeof(float);
    float *cudaTemp = (float *)FastllmCudaMalloc(tempBytes);
    if (cudaTemp == nullptr) {
        return false;
    }
    bool ok = allowTriton &&
        fastllm::FastllmCudaTryTritonDeepSeekV4SparseAttentionDecodeGraph(
            q, windowKV, compressedKV, attnSink, windowSize, compressRatio,
            decodeMeta, softmaxScale, cudaTemp);
    int blocks = bsz * heads;
    if (!ok) {
        if (q.dataType == fastllm::DataType::BFLOAT16 &&
            compressedKV.dataType == fastllm::DataType::BFLOAT16) {
            DeepSeekV4SparseAttentionDecodeCachedOnlineKernel<__nv_bfloat16, __nv_bfloat16>
                <<<blocks, 256>>>((const __nv_bfloat16 *)q.cudaData,
                                  (const float *)windowKV.cudaData,
                                  (const __nv_bfloat16 *)compressedKV.cudaData,
                                  (const float *)attnSink.cudaData, cudaTemp,
                                  bsz, heads, dim, windowSize, 0, 0, softmaxScale,
                                  decodeMeta, compressRatio);
            ok = true;
        } else if (q.dataType == fastllm::DataType::FLOAT16 &&
                   compressedKV.dataType == fastllm::DataType::BFLOAT16) {
            DeepSeekV4SparseAttentionDecodeCachedOnlineKernel<half, __nv_bfloat16>
                <<<blocks, 256>>>((const half *)q.cudaData,
                                  (const float *)windowKV.cudaData,
                                  (const __nv_bfloat16 *)compressedKV.cudaData,
                                  (const float *)attnSink.cudaData, cudaTemp,
                                  bsz, heads, dim, windowSize, 0, 0, softmaxScale,
                                  decodeMeta, compressRatio);
            ok = true;
        } else if (q.dataType == fastllm::DataType::FLOAT32 &&
                   compressedKV.dataType == fastllm::DataType::BFLOAT16) {
            DeepSeekV4SparseAttentionDecodeCachedOnlineKernel<float, __nv_bfloat16>
                <<<blocks, 256>>>((const float *)q.cudaData,
                                  (const float *)windowKV.cudaData,
                                  (const __nv_bfloat16 *)compressedKV.cudaData,
                                  (const float *)attnSink.cudaData, cudaTemp,
                                  bsz, heads, dim, windowSize, 0, 0, softmaxScale,
                                  decodeMeta, compressRatio);
            ok = true;
        }
    }
    if (ok) {
        int rows = bsz * heads;
        int threads = 128;
        int rotaryWorkPerRow = dim - ropeDim + (ropeDim >> 1);
        int rotaryBlocks = (rows * rotaryWorkPerRow + threads - 1) / threads;
        DeepSeekV4SparseDecodeRotaryCastKernel<<<rotaryBlocks, threads>>>(
            cudaTemp, (__nv_bfloat16 *)output.cudaData, rows, dim, ropeDim, ropeBase,
            0, originalSeqLen, ropeFactor, betaFast, betaSlow, decodeMeta);
        DeviceSync();
    }
    FastllmCudaFree(cudaTemp);
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV4SparseAttentionDecodeCachedBatch(
                                                                 const std::vector<fastllm::Data*> &qParts,
                                                                 const std::vector<fastllm::Data*> &windowKVs,
                                                                 const std::vector<fastllm::Data*> &compressedKVs,
                                                                 fastllm::Data &attnSink,
                                                                 int windowSize,
                                                                 const std::vector<int> &startPositions,
                                                                 const std::vector<int> &compressedCounts,
                                                                 int ropeDim, float ropeBase,
                                                                 int originalSeqLen,
                                                                 float ropeFactor, int betaFast,
                                                                 int betaSlow, float softmaxScale,
                                                                 fastllm::Data &output) {
    if (std::getenv("FASTLLM_DSV4_DISABLE_BATCH_SPARSE_DECODE") != nullptr) {
        return false;
    }
    int batch = (int)qParts.size();
    if (batch <= 0 || (int)windowKVs.size() != batch || (int)compressedKVs.size() != batch ||
        (int)startPositions.size() != batch || (int)compressedCounts.size() != batch ||
        windowSize <= 0 || ropeDim <= 0) {
        return false;
    }
    if (qParts[0] == nullptr || qParts[0]->dims.size() != 4 || qParts[0]->dims[0] != 1 ||
        qParts[0]->dims[1] != 1 || qParts[0]->dataDevice != fastllm::DataDevice::CUDA) {
        return false;
    }
    int heads = qParts[0]->dims[2], dim = qParts[0]->dims[3];
    if (heads <= 0 || dim <= 0 || ropeDim > dim || dim > 256 * 4 ||
        attnSink.dataDevice != fastllm::DataDevice::CUDA ||
        attnSink.dataType != fastllm::DataType::FLOAT32 ||
        attnSink.Count(0) != (uint64_t)heads) {
        return false;
    }

    fastllm::DataType qType = qParts[0]->dataType;
    fastllm::DataType compressedType = fastllm::DataType::FLOAT32;
    bool hasCompressed = false;
    std::vector<const void*> qPtrs(batch, nullptr);
    std::vector<const float*> windowPtrs(batch, nullptr);
    std::vector<const void*> compressedPtrs(batch, nullptr);
    for (int b = 0; b < batch; b++) {
        const fastllm::Data *q = qParts[b];
        const fastllm::Data *windowKV = windowKVs[b];
        const fastllm::Data *compressedKV = compressedKVs[b];
        int startPos = startPositions[b];
        int compressedCount = compressedCounts[b];
        int liveWindow = startPos >= windowSize - 1 ? windowSize : (startPos + 1);
        if (q == nullptr || windowKV == nullptr || startPos < 0 || compressedCount < 0 ||
            liveWindow + compressedCount <= 0 ||
            liveWindow + compressedCount > kDeepSeekV4SparseDecodeMaxKeys ||
            q->dims.size() != 4 || q->dims[0] != 1 || q->dims[1] != 1 ||
            q->dims[2] != heads || q->dims[3] != dim ||
            q->dataType != qType || q->dataDevice != fastllm::DataDevice::CUDA ||
            windowKV->dataDevice != fastllm::DataDevice::CUDA ||
            windowKV->dataType != fastllm::DataType::FLOAT32 ||
            windowKV->Count(0) != (uint64_t)windowSize * dim) {
            return false;
        }
        qPtrs[b] = q->cudaData;
        windowPtrs[b] = (const float*)windowKV->cudaData;
        if (compressedCount > 0) {
            if (compressedKV == nullptr ||
                compressedKV->dataDevice != fastllm::DataDevice::CUDA ||
                compressedKV->Count(0) < (uint64_t)compressedCount * dim) {
                return false;
            }
            if (!hasCompressed) {
                compressedType = compressedKV->dataType;
                hasCompressed = true;
            } else if (compressedKV->dataType != compressedType) {
                return false;
            }
            if (compressedType != fastllm::DataType::BFLOAT16 &&
                compressedType != fastllm::DataType::FLOAT16 &&
                compressedType != fastllm::DataType::FLOAT32) {
                return false;
            }
            compressedPtrs[b] = compressedKV->cudaData;
        }
    }
    if (!hasCompressed) {
        compressedType = fastllm::DataType::FLOAT32;
    }
    if (qType != fastllm::DataType::BFLOAT16 &&
        qType != fastllm::DataType::FLOAT16 &&
        qType != fastllm::DataType::FLOAT32) {
        return false;
    }

    if (!DeepSeekV4PrepareCudaOutput(output, fastllm::DataType::BFLOAT16, {batch, 1, heads, dim})) {
        return false;
    }

    const void **cudaQPtrs = (const void**)FastllmCudaMalloc((size_t)batch * sizeof(void*));
    const float **cudaWindowPtrs = (const float**)FastllmCudaMalloc((size_t)batch * sizeof(float*));
    const void **cudaCompressedPtrs = (const void**)FastllmCudaMalloc((size_t)batch * sizeof(void*));
    int *cudaStartPositions = (int*)FastllmCudaMalloc((size_t)batch * sizeof(int));
    int *cudaCompressedCounts = (int*)FastllmCudaMalloc((size_t)batch * sizeof(int));
    float *cudaTemp = (float*)FastllmCudaMalloc((size_t)batch * heads * dim * sizeof(float));
    if (cudaQPtrs == nullptr || cudaWindowPtrs == nullptr || cudaCompressedPtrs == nullptr ||
        cudaStartPositions == nullptr || cudaCompressedCounts == nullptr || cudaTemp == nullptr) {
        if (cudaQPtrs != nullptr) {
            FastllmCudaFree((void*)cudaQPtrs);
        }
        if (cudaWindowPtrs != nullptr) {
            FastllmCudaFree((void*)cudaWindowPtrs);
        }
        if (cudaCompressedPtrs != nullptr) {
            FastllmCudaFree((void*)cudaCompressedPtrs);
        }
        if (cudaStartPositions != nullptr) {
            FastllmCudaFree(cudaStartPositions);
        }
        if (cudaCompressedCounts != nullptr) {
            FastllmCudaFree(cudaCompressedCounts);
        }
        if (cudaTemp != nullptr) {
            FastllmCudaFree(cudaTemp);
        }
        return false;
    }

    cudaMemcpy((void*)cudaQPtrs, qPtrs.data(), (size_t)batch * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)cudaWindowPtrs, windowPtrs.data(), (size_t)batch * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)cudaCompressedPtrs, compressedPtrs.data(), (size_t)batch * sizeof(void*), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaStartPositions, startPositions.data(), (size_t)batch * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaCompressedCounts, compressedCounts.data(), (size_t)batch * sizeof(int), cudaMemcpyHostToDevice);

    bool ok = false;
    if (qType == fastllm::DataType::BFLOAT16) {
        ok = DeepSeekV4LaunchSparseDecodeBatchByCompressed<__nv_bfloat16>(
            cudaQPtrs, cudaWindowPtrs, cudaCompressedPtrs, compressedType,
            (const float*)attnSink.cudaData, cudaStartPositions, cudaCompressedCounts,
            cudaTemp, batch, heads, dim, windowSize, softmaxScale);
    } else if (qType == fastllm::DataType::FLOAT16) {
        ok = DeepSeekV4LaunchSparseDecodeBatchByCompressed<half>(
            cudaQPtrs, cudaWindowPtrs, cudaCompressedPtrs, compressedType,
            (const float*)attnSink.cudaData, cudaStartPositions, cudaCompressedCounts,
            cudaTemp, batch, heads, dim, windowSize, softmaxScale);
    } else if (qType == fastllm::DataType::FLOAT32) {
        ok = DeepSeekV4LaunchSparseDecodeBatchByCompressed<float>(
            cudaQPtrs, cudaWindowPtrs, cudaCompressedPtrs, compressedType,
            (const float*)attnSink.cudaData, cudaStartPositions, cudaCompressedCounts,
            cudaTemp, batch, heads, dim, windowSize, softmaxScale);
    }

    if (ok) {
        int rows = batch * heads;
        int threads = 128;
        int rotaryWorkPerRow = dim - ropeDim + (ropeDim >> 1);
        int blocks = (rows * rotaryWorkPerRow + threads - 1) / threads;
        DeepSeekV4SparseDecodeRotaryCastBatchKernel<<<blocks, threads>>>(
            cudaTemp, (__nv_bfloat16*)output.cudaData, cudaStartPositions,
            rows, heads, dim, ropeDim, ropeBase, originalSeqLen,
            ropeFactor, betaFast, betaSlow);
        DeviceSync();
    }

    FastllmCudaFree((void*)cudaQPtrs);
    FastllmCudaFree((void*)cudaWindowPtrs);
    FastllmCudaFree((void*)cudaCompressedPtrs);
    FastllmCudaFree(cudaStartPositions);
    FastllmCudaFree(cudaCompressedCounts);
    FastllmCudaFree(cudaTemp);
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV4WoA(const fastllm::Data &o, const fastllm::Data &woA,
                                         int groups, int oRank, fastllm::Data &output,
                                         bool allowTriton) {
    if (o.dataDevice != fastllm::DataDevice::CUDA || woA.dataDevice != fastllm::DataDevice::CUDA ||
        o.dims.size() != 4 || groups <= 0 || oRank <= 0) {
        return false;
    }

    int bsz = o.dims[0], seqlen = o.dims[1], heads = o.dims[2], headDim = o.dims[3];
    if (heads % groups != 0) {
        return false;
    }
    int groupDim = (heads / groups) * headDim;
    if (woA.Count(0) != (uint64_t)groups * oRank * groupDim) {
        return false;
    }

    if (!DeepSeekV4PrepareCudaOutput(output, fastllm::DataType::BFLOAT16,
                                     {bsz, seqlen, groups * oRank})) {
        return false;
    }

    fastllm::Data &mutableWoA = const_cast<fastllm::Data&>(woA);
    if (allowTriton && fastllm::FastllmCudaTryTritonDeepSeekV4WoA(
            o, mutableWoA, groups, oRank, output)) {
        DeviceSync();
        return true;
    }

    bool ok = false;
    if (o.dataType == fastllm::DataType::BFLOAT16) {
        ok = DeepSeekV4LaunchWoAByWeight<__nv_bfloat16>(o, woA, groups, oRank, output, bsz, seqlen, heads, headDim);
    } else if (o.dataType == fastllm::DataType::FLOAT16) {
        ok = DeepSeekV4LaunchWoAByWeight<half>(o, woA, groups, oRank, output, bsz, seqlen, heads, headDim);
    } else if (o.dataType == fastllm::DataType::FLOAT32) {
        ok = DeepSeekV4LaunchWoAByWeight<float>(o, woA, groups, oRank, output, bsz, seqlen, heads, headDim);
    }
    if (!ok) {
        return false;
    }
    DeviceSync();
    return true;
}

extern "C" bool FastllmCudaDeepSeekV4HcPost(const fastllm::Data &x, const fastllm::Data &residual,
                                            const float *post, const float *comb, int bsz,
                                            int seqlen, int hcMult, int dim, fastllm::Data &output) {
    if (x.dataDevice != fastllm::DataDevice::CUDA || residual.dataDevice != fastllm::DataDevice::CUDA ||
        post == nullptr || comb == nullptr || bsz <= 0 || seqlen <= 0 || hcMult <= 0 || dim <= 0 ||
        x.Count(0) != (uint64_t)bsz * seqlen * dim ||
        residual.Count(0) != (uint64_t)bsz * seqlen * hcMult * dim) {
        return false;
    }

    if (!DeepSeekV4PrepareCudaOutput(output, x.dataType, {bsz, seqlen, hcMult, dim})) {
        return false;
    }

    int tokens = bsz * seqlen;
    size_t postBytes = (size_t)tokens * hcMult * sizeof(float);
    size_t combBytes = (size_t)tokens * hcMult * hcMult * sizeof(float);
    float *cudaPost = (float *)FastllmCudaMalloc(postBytes);
    float *cudaComb = (float *)FastllmCudaMalloc(combBytes);
    FastllmCudaCopyFromHostToDevice(cudaPost, (void *)post, postBytes);
    FastllmCudaCopyFromHostToDevice(cudaComb, (void *)comb, combBytes);

    bool ok = false;
    if (x.dataType == fastllm::DataType::BFLOAT16) {
        ok = DeepSeekV4LaunchHcPostByResidual<__nv_bfloat16>(x, residual, cudaPost, cudaComb,
                                                            output, tokens, hcMult, dim);
    } else if (x.dataType == fastllm::DataType::FLOAT16) {
        ok = DeepSeekV4LaunchHcPostByResidual<half>(x, residual, cudaPost, cudaComb,
                                                   output, tokens, hcMult, dim);
    } else if (x.dataType == fastllm::DataType::FLOAT32) {
        ok = DeepSeekV4LaunchHcPostByResidual<float>(x, residual, cudaPost, cudaComb,
                                                    output, tokens, hcMult, dim);
    }

    DeviceSync();
    FastllmCudaFree(cudaPost);
    FastllmCudaFree(cudaComb);
    return ok;
}

extern "C" bool FastllmCudaDeepSeekV4HcPostCudaMix(const fastllm::Data &x, const fastllm::Data &residual,
                                                   const fastllm::Data &post, const fastllm::Data &comb,
                                                   int bsz, int seqlen, int hcMult, int dim,
                                                   fastllm::Data &output) {
    if (x.dataDevice != fastllm::DataDevice::CUDA || residual.dataDevice != fastllm::DataDevice::CUDA ||
        post.dataDevice != fastllm::DataDevice::CUDA || comb.dataDevice != fastllm::DataDevice::CUDA ||
        post.dataType != fastllm::DataType::FLOAT32 || comb.dataType != fastllm::DataType::FLOAT32 ||
        bsz <= 0 || seqlen <= 0 || hcMult <= 0 || dim <= 0 ||
        x.Count(0) != (uint64_t)bsz * seqlen * dim ||
        residual.Count(0) != (uint64_t)bsz * seqlen * hcMult * dim ||
        post.Count(0) != (uint64_t)bsz * seqlen * hcMult ||
        comb.Count(0) != (uint64_t)bsz * seqlen * hcMult * hcMult) {
        return false;
    }

    if (!DeepSeekV4PrepareCudaOutput(output, x.dataType, {bsz, seqlen, hcMult, dim})) {
        return false;
    }

    int tokens = bsz * seqlen;
    bool ok = false;
    const float *cudaPost = (const float *)post.cudaData;
    const float *cudaComb = (const float *)comb.cudaData;
    if (x.dataType == fastllm::DataType::BFLOAT16) {
        ok = DeepSeekV4LaunchHcPostByResidual<__nv_bfloat16>(x, residual, cudaPost, cudaComb,
                                                            output, tokens, hcMult, dim);
    } else if (x.dataType == fastllm::DataType::FLOAT16) {
        ok = DeepSeekV4LaunchHcPostByResidual<half>(x, residual, cudaPost, cudaComb,
                                                   output, tokens, hcMult, dim);
    } else if (x.dataType == fastllm::DataType::FLOAT32) {
        ok = DeepSeekV4LaunchHcPostByResidual<float>(x, residual, cudaPost, cudaComb,
                                                    output, tokens, hcMult, dim);
    }
    DeviceSync();
    return ok;
}
