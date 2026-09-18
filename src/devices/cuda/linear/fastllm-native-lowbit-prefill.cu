#include "fastllm-cuda.cuh"
#include "fastllm-native-lowbit-prefill.cuh"
#ifdef FASTLLM_NATIVE_PREFILL_SM120
#include "fastllm-native-nvfp4-tma.cuh"
#endif
#include <cstring>
#include <limits>
#if CUDART_VERSION >= 12080
#include <cstdio>
#include <cstdlib>
#include <cublasLt.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <vector>

// Optional activation quantization. Decode retains its existing weight-only path.
namespace fastllm_native_prefill {
bool Enabled(const char *name) {
    const char *v = std::getenv(name);
    return v && (!std::strcmp(v, "1") || !std::strcmp(v, "true"));
}

void CheckCuda(cudaError_t error, const char *operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}
struct Plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo{};
    bool valid = false, tuned = false;
    std::vector<cublasLtMatmulAlgo_t> candidates;
    ~Plan() {
        if (op)
            cublasLtMatmulDescDestroy(op);
        if (a)
            cublasLtMatrixLayoutDestroy(a);
        if (b)
            cublasLtMatrixLayoutDestroy(b);
        if (c)
            cublasLtMatrixLayoutDestroy(c);
    }
};
struct State {
    std::mutex mutex;
    cublasLtHandle_t handle = nullptr;
    void *workspace = nullptr;
    uint8_t *scratch = nullptr;
    cudaEvent_t ready = nullptr;
    static constexpr size_t workspaceBytes = 32ull << 20, capacity = 384ull << 20;
    std::map<std::tuple<int, int, int, int>, std::unique_ptr<Plan>> plans;
    bool initialized = false;
};
inline State *GetState() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess)
        return nullptr;
    static std::mutex mutex;
    static std::map<int, std::unique_ptr<State>> states;
    State *s;
    {
        std::lock_guard<std::mutex> lock(mutex);
        auto &v = states[dev];
        if (!v)
            v = std::make_unique<State>();
        s = v.get();
    }
    std::lock_guard<std::mutex> lock(s->mutex);
    if (s->initialized)
        return s;
    // Reserve before KV budget calibration. Serving never grows this allocation.
    if (!FastllmCudaGetNcclForceSync())
        return nullptr;
    if (cudaMalloc(&s->workspace, State::workspaceBytes + State::capacity) != cudaSuccess) {
        cudaGetLastError();
        return nullptr;
    }
    s->scratch = static_cast<uint8_t *>(s->workspace) + State::workspaceBytes;
    if (cublasLtCreate(&s->handle) != CUBLAS_STATUS_SUCCESS ||
        cudaEventCreateWithFlags(&s->ready, cudaEventDisableTiming) != cudaSuccess) {
        if (s->handle)
            cublasLtDestroy(s->handle);
        s->handle = nullptr;
        cudaFree(s->workspace);
        s->workspace = nullptr;
        return nullptr;
    }
    CheckCuda(cudaEventRecord(s->ready, cudaStreamPerThread), "native prefill init event");
    s->initialized = true;
    return s;
}
inline Plan *GetPlan(State &s, int bits, int M, int N, int K) {
    auto key = std::make_tuple(bits, M, N, K);
    auto found = s.plans.find(key);
    if (found != s.plans.end())
        return found->second->valid ? found->second.get() : nullptr;
    auto ptr = std::make_unique<Plan>();
    Plan *p = ptr.get();
    s.plans.emplace(key, std::move(ptr));
    if (cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS)
        return nullptr;
    cublasOperation_t trans = CUBLAS_OP_T;
    if (cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)) !=
        CUBLAS_STATUS_SUCCESS)
        return nullptr;
    if (bits == 4) {
        auto mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        if (cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode)) !=
                CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode)) !=
                CUBLAS_STATUS_SUCCESS)
            return nullptr;
    }
    if (bits == 4) {
        auto align = [](size_t n) { return (n + 255) & ~size_t(255); };
        uint8_t *sw = s.scratch + align(size_t(N) * K / 2);
        uint8_t *sx = sw + align(size_t(N) * K / 16) + align(size_t(M) * K / 2);
        if (cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sw, sizeof(sw)) !=
                CUBLAS_STATUS_SUCCESS ||
            cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sx, sizeof(sx)) !=
                CUBLAS_STATUS_SUCCESS)
            return nullptr;
    }
    auto type = bits == 8 ? CUDA_R_8F_E4M3 : CUDA_R_4F_E2M1;
    if (cublasLtMatrixLayoutCreate(&p->a, type, K, N, K) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&p->b, type, K, M, K) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&p->c, CUDA_R_32F, N, M, N) != CUBLAS_STATUS_SUCCESS)
        return nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS)
        return nullptr;
    size_t bytes = State::workspaceBytes;
    auto status = cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &bytes,
                                                       sizeof(bytes));
    cublasLtMatmulHeuristicResult_t h[8];
    int count = 0;
    if (status == CUBLAS_STATUS_SUCCESS)
        status = cublasLtMatmulAlgoGetHeuristic(s.handle, p->op, p->a, p->b, p->c, p->c, pref, 8, h, &count);
    cublasLtMatmulPreferenceDestroy(pref);
    if (status != CUBLAS_STATUS_SUCCESS || count == 0)
        return nullptr;
    for (int i = 0; i < count; i++)
        if (h[i].state == CUBLAS_STATUS_SUCCESS)
            p->candidates.push_back(h[i].algo);
    if (p->candidates.empty())
        return nullptr;
    p->algo = p->candidates[0];
    p->valid = true;
    return p;
}
inline void Tune(State &s, Plan &p, const void *a, const void *b, float *c) {
    if (p.tuned)
        return;
    p.tuned = true;
    cudaEvent_t begin = nullptr, end = nullptr;
    if (cudaEventCreate(&begin) != cudaSuccess)
        return;
    if (cudaEventCreate(&end) != cudaSuccess) {
        cudaEventDestroy(begin);
        return;
    }
    float best = 1.e30f, alpha = 1, beta = 0;
    for (auto &candidate : p.candidates) {
        auto run = [&]() {
            return cublasLtMatmul(s.handle, p.op, &alpha, a, p.a, b, p.b, &beta, c, p.c, c, p.c, &candidate,
                                  s.workspace, State::workspaceBytes, cudaStreamPerThread);
        };
        if (run() != CUBLAS_STATUS_SUCCESS)
            continue;
        CheckCuda(cudaEventRecord(begin, cudaStreamPerThread), "native prefill benchmark start");
        bool ok = true;
        for (int i = 0; i < 3; i++)
            if (run() != CUBLAS_STATUS_SUCCESS) {
                ok = false;
                break;
            }
        CheckCuda(cudaEventRecord(end, cudaStreamPerThread), "native prefill benchmark end");
        CheckCuda(cudaEventSynchronize(end), "native prefill algorithm benchmark");
        float ms = 0;
        cudaEventElapsedTime(&ms, begin, end);
        if (ok && ms < best) {
            best = ms;
            p.algo = candidate;
        }
    }
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    if (best == 1.e30f)
        p.valid = false;
}
__device__ inline float RowMax(float value) {
    for (int d = 16; d; d >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, d));
    __shared__ float warp[8];
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (!lane)
        warp[w] = value;
    __syncthreads();
    value = threadIdx.x < 8 ? warp[lane] : 0;
    for (int d = 16; d; d >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, d));
    if (!threadIdx.x)
        warp[0] = value;
    __syncthreads();
    return warp[0];
}
static __global__ void QuantFp8(const half *input, uint8_t *output, float *scales, int K) {
    int row = blockIdx.x, t = threadIdx.x;
    float maxv = 0;
    const half2 *source = reinterpret_cast<const half2 *>(input + size_t(row) * K);
    for (int k = t; k < K / 2; k += 256) {
        float2 v = __half22float2(source[k]);
        maxv = fmaxf(maxv, fmaxf(fabsf(v.x), fabsf(v.y)));
    }
    float scale = fmaxf(RowMax(maxv) / 448.f, 1.e-12f);
    if (!t)
        scales[row] = scale;
    for (int k = t; k < K / 2; k += 256) {
        float2 v = __half22float2(source[k]);
        reinterpret_cast<uint16_t *>(output)[size_t(row) * (K / 2) + k] =
            __nv_cvt_float2_to_fp8x2(make_float2(v.x / scale, v.y / scale), __NV_SATFINITE, __NV_E4M3);
    }
}
template <int Mode>
static __global__ void ScaleOutput(const float *input, half *output, const float *xs, const float *ws,
                                   const half *bias, int M, int N) {
    int width = Mode == 1 ? N / 2 : N, row = blockIdx.y, n = (blockIdx.x * 256 + threadIdx.x) * 4;
    if (n >= width)
        return;
    size_t i = size_t(row) * width + n;
    float4 v = *reinterpret_cast<const float4 *>(input + size_t(row) * N + n);
    float4 w = *reinterpret_cast<const float4 *>(ws + n);
    float x = xs[row];
    half2 lo = __floats2half2_rn(v.x * x * w.x, v.y * x * w.y),
          hi = __floats2half2_rn(v.z * x * w.z, v.w * x * w.w);
    if (bias) {
        lo = __hadd2(lo, *reinterpret_cast<const half2 *>(bias + n));
        hi = __hadd2(hi, *reinterpret_cast<const half2 *>(bias + n + 2));
    }
    if constexpr (Mode == 1) {
        float4 u = *reinterpret_cast<const float4 *>(input + size_t(row) * N + n + width);
        float4 uw = *reinterpret_cast<const float4 *>(ws + n + width);
        half2 ul = __floats2half2_rn(u.x * x * uw.x, u.y * x * uw.y),
              uh = __floats2half2_rn(u.z * x * uw.z, u.w * x * uw.w);
        if (bias) {
            ul = __hadd2(ul, *reinterpret_cast<const half2 *>(bias + n + width));
            uh = __hadd2(uh, *reinterpret_cast<const half2 *>(bias + n + width + 2));
        }
        half2 one = __float2half2_rn(1.f);
        lo = __hmul2(__h2div(lo, __hadd2(one, h2exp(__hneg2(lo)))), ul);
        hi = __hmul2(__h2div(hi, __hadd2(one, h2exp(__hneg2(hi)))), uh);
    } else if constexpr (Mode == 2) {
        lo = __hadd2(*reinterpret_cast<const half2 *>(output + i), lo);
        hi = __hadd2(*reinterpret_cast<const half2 *>(output + i + 2), hi);
    }
    *reinterpret_cast<half2 *>(output + i) = lo;
    *reinterpret_cast<half2 *>(output + i + 2) = hi;
}
bool Fp8(const half *input, const uint8_t *weight, const float *scales, const half *bias, half *output, int M,
         int N, int K, int mode) {
    if (!LinearPrefillEnabled(8, M, N, K) || !Supported(8) || !input || !weight || !scales ||
        !output || N <= 0 || K <= 0 || mode < 0 || mode > 2 || uintptr_t(weight) % 16 || uintptr_t(input) % 4)
        return false;
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone)
        return false;
    State *s = GetState();
    if (!s || M < 32 || M > 4096 || N % 16 || K % 16 || uintptr_t(output) % 4 || uintptr_t(scales) % 16 ||
        (bias && uintptr_t(bias) % 4))
        return false;
    const size_t qbytes = (size_t(M) * K + 255) & ~size_t(255), sbytes = (size_t(M) * 4 + 255) & ~size_t(255),
                 outbytes = size_t(M) * N * 4;
    if (qbytes + sbytes + outbytes > State::capacity)
        return false;
    std::lock_guard<std::mutex> lock(s->mutex);
    Plan *p = GetPlan(*s, 8, M, N, K);
    if (!p)
        return false;
    CheckCuda(cudaStreamWaitEvent(cudaStreamPerThread, s->ready, 0), "native prefill scratch dependency");
    uint8_t *q = s->scratch;
    float *xs = reinterpret_cast<float *>(q + qbytes), *tmp = reinterpret_cast<float *>(q + qbytes + sbytes);
    QuantFp8<<<M, 256, 0, cudaStreamPerThread>>>(input, q, xs, K);
    CheckCuda(cudaGetLastError(), "native FP8 quantize");
    Tune(*s, *p, weight, q, tmp);
    if (!p->valid) {
        CheckCuda(cudaEventRecord(s->ready, cudaStreamPerThread), "native FP8 fallback event");
        return false;
    }
    const float alpha = 1, beta = 0;
    auto status = cublasLtMatmul(s->handle, p->op, &alpha, weight, p->a, q, p->b, &beta, tmp, p->c, tmp, p->c,
                                 &p->algo, s->workspace, State::workspaceBytes, cudaStreamPerThread);
    if (status == CUBLAS_STATUS_SUCCESS) {
        if (mode == 1)
            ScaleOutput<1><<<dim3((N / 2 + 1023) / 1024, M), 256, 0, cudaStreamPerThread>>>(
                tmp, output, xs, scales, bias, M, N);
        else if (mode == 2)
            ScaleOutput<2><<<dim3((N + 1023) / 1024, M), 256, 0, cudaStreamPerThread>>>(tmp, output, xs,
                                                                                        scales, bias, M, N);
        else
            ScaleOutput<0><<<dim3((N + 1023) / 1024, M), 256, 0, cudaStreamPerThread>>>(tmp, output, xs,
                                                                                        scales, bias, M, N);
    }
    CheckCuda(cudaGetLastError(), "native prefill output");
    CheckCuda(cudaEventRecord(s->ready, cudaStreamPerThread), "native prefill completion event");
    return status == CUBLAS_STATUS_SUCCESS;
}
__device__ inline size_t ScaleIndex(int row, int group, int groups) {
    return (size_t(row / 128) * (groups / 4) + group / 4) * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 +
           group % 4;
}
__device__ inline int MarlinScaleIndex(int n) {
    int t = ((n & 7) << 3) | ((n & 63) >> 3), lo = t & 3;
    return (n & ~63) + (t & ~3) + ((lo & 1) << 1) + (lo >> 1);
}
static __global__ void RepackFp4(const uint32_t *weight, const uint8_t *scales, uint8_t *out, uint8_t *sout,
                                 int N, int packedN, int K) {
    __shared__ __align__(16) uint8_t tile[64 * 128];
    int word = threadIdx.x & 127;
    int localN = (word & 3) * 16 + (word >> 4), row = blockIdx.x * 64 + localN;
    int startK = blockIdx.y * 256;
    for (int t = 0; t < 8; t++) {
        int localGroup = t * 2 + (threadIdx.x >> 7), group = startK / 16 + localGroup;
        uint32_t q = group < K / 16 ? weight[(size_t(group) * (packedN / 64) + row / 64) * 128 + word] : 0;
        int pair = localGroup * 8 + ((word >> 2) & 3);
        tile[localN * 128 + pair] = (q & 15) | (((q >> 16) & 15) << 4);
        tile[localN * 128 + pair + 4] = ((q >> 4) & 15) | (((q >> 20) & 15) << 4);
        tile[(localN + 8) * 128 + pair] = ((q >> 8) & 15) | (((q >> 24) & 15) << 4);
        tile[(localN + 8) * 128 + pair + 4] = ((q >> 12) & 15) | (((q >> 28) & 15) << 4);
        if (((word >> 2) & 3) == 0 && group < K / 16) {
            if (row < N) {
                float f = __half2float(__ushort_as_half(
                              uint16_t(scales[size_t(group) * packedN + MarlinScaleIndex(row)]) << 7)) /
                          128;
                sout[ScaleIndex(row, group, K / 16)] = __nv_fp8_e4m3(f).__x;
            }
            if (row + 8 < N) {
                float f = __half2float(__ushort_as_half(
                              uint16_t(scales[size_t(group) * packedN + MarlinScaleIndex(row + 8)]) << 7)) /
                          128;
                sout[ScaleIndex(row + 8, group, K / 16)] = __nv_fp8_e4m3(f).__x;
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 512; i += 256) {
        int row = blockIdx.x * 64 + i / 8, k = startK / 2 + (i % 8) * 16;
        if (row < N && k < K / 2)
            reinterpret_cast<uint4 *>(out)[size_t(row) * (K / 32) + k / 16] =
                reinterpret_cast<const uint4 *>(tile)[i];
    }
}
template <bool TmaScale = false>
static __global__ void QuantFp4(const half *input, uint8_t *output, uint8_t *scales, float *global, int K, int inputRows) {
    int row = blockIdx.x;
    if (row >= inputRows) {
        for (int i = threadIdx.x; i < K / 2; i += blockDim.x) output[size_t(row) * K / 2 + i] = 0;
        for (int g = threadIdx.x; g < K / 16; g += blockDim.x) scales[ScaleIndex(row, g, K / 16)] = 0;
        if (!threadIdx.x) global[row] = 1.f;
        return;
    }
    float maxv = 0;
    for (int k = threadIdx.x; k < K; k += 256)
        maxv = fmaxf(maxv, fabsf(__half2float(input[size_t(row) * K + k])));
    float scale = fmaxf(RowMax(maxv) / (448.f * 6), 1.e-12f);
    if (!threadIdx.x)
        global[row] = scale;
    for (int g = threadIdx.x; g < K / 16; g += 256) {
        float v[16], mx = 0;
        for (int j = 0; j < 16; j++) {
            v[j] = __half2float(input[size_t(row) * K + g * 16 + j]);
            mx = fmaxf(mx, fabsf(v[j]));
        }
        __nv_fp8_e4m3 sf(mx / (6 * scale));
        float denom = float(sf) * scale;
        float inv = denom > 0 ? 1.f / denom : 0;
        uint64_t q = 0;
        for (int j = 0; j < 8; j++)
            q |= uint64_t(__nv_cvt_float2_to_fp4x2(make_float2(v[j * 2] * inv, v[j * 2 + 1] * inv), __NV_E2M1,
                                                   cudaRoundNearest))
                 << (8 * j);
        reinterpret_cast<uint64_t *>(output)[size_t(row) * (K / 16) + g] = q;
        size_t scaleOffset;
        if constexpr (TmaScale)
            scaleOffset = (size_t(row / 256) * ((K / 16) / 16) + g / 16) * 4096 + (row % 256) * 16 + g % 16;
        else
            scaleOffset = ScaleIndex(row, g, K / 16);
        scales[scaleOffset] = sf.__x;
    }
}
template <int Mode>
static __global__ void ScaleFp4Output(const float *input, half *output, const float *xs, const float *global,
                                      const half *bias, int M, int N) {
    int width = Mode == 1 ? N / 2 : N, row = blockIdx.y, n = (blockIdx.x * 256 + threadIdx.x) * 4;
    if (n >= width)
        return;
    size_t i = size_t(row) * width + n;
    float factor = xs[row] * (*global / 128);
    float4 v = *reinterpret_cast<const float4 *>(input + size_t(row) * N + n);
    half2 lo = __floats2half2_rn(v.x * factor, v.y * factor),
          hi = __floats2half2_rn(v.z * factor, v.w * factor);
    if (bias) {
        lo = __hadd2(lo, *reinterpret_cast<const half2 *>(bias + n));
        hi = __hadd2(hi, *reinterpret_cast<const half2 *>(bias + n + 2));
    }
    if constexpr (Mode == 1) {
        float4 u = *reinterpret_cast<const float4 *>(input + size_t(row) * N + n + width);
        half2 ul = __floats2half2_rn(u.x * factor, u.y * factor),
              uh = __floats2half2_rn(u.z * factor, u.w * factor);
        if (bias) {
            ul = __hadd2(ul, *reinterpret_cast<const half2 *>(bias + n + width));
            uh = __hadd2(uh, *reinterpret_cast<const half2 *>(bias + n + width + 2));
        }
        half2 one = __float2half2_rn(1.f);
        lo = __hmul2(__h2div(lo, __hadd2(one, h2exp(__hneg2(lo)))), ul);
        hi = __hmul2(__h2div(hi, __hadd2(one, h2exp(__hneg2(hi)))), uh);
    } else if constexpr (Mode == 2) {
        lo = __hadd2(*reinterpret_cast<const half2 *>(output + i), lo);
        hi = __hadd2(*reinterpret_cast<const half2 *>(output + i + 2), hi);
    }
    *reinterpret_cast<half2 *>(output + i) = lo;
    *reinterpret_cast<half2 *>(output + i + 2) = hi;
}
bool Fp4(const half *input, const uint32_t *weight, const uint8_t *scales, const float *global,
         const half *bias, half *output, int M, int N, int packedN, int K, int mode, bool nativeLayout) {
    if (!LinearPrefillEnabled(4, M, N, K) || !Supported(4) || !input || !weight || !scales ||
        !global || !output || N <= 0 || K <= 0 || packedN < N || packedN % 64 || mode < 0 || mode > 2 ||
        uintptr_t(weight) % 16 || uintptr_t(input) % 4)
        return false;
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone)
        return false;
    State *s = GetState();
    if (!s || N % 128 || K % 64 || uintptr_t(output) % 4 || (bias && uintptr_t(bias) % 4))
        return false;
    std::lock_guard<std::mutex> lock(s->mutex);
    CheckCuda(cudaStreamWaitEvent(cudaStreamPerThread, s->ready, 0), "native FP4 scratch dependency");
    const int outputRows = M;
    // Native-layout small/unaligned prefills can pad the quantized activations,
    // without materializing a second FP16 weight matrix. Decode stays W4A16.
    if (nativeLayout && M > 8 && M <= 4096) M = (M + 127) / 128 * 128;
    if (M < 128 || M > 4096 || M % 128)
        return false;
    bool useTma = false;
#ifdef FASTLLM_NATIVE_PREFILL_SM120
    useTma = !bias && Nvfp4TmaCanRun(outputRows, N, K, mode);
#endif
    auto aligned = [](size_t n) { return (n + 255) & ~size_t(255); };
    size_t wb = aligned(size_t(N) * K / 2), wsb = aligned(size_t(N) * K / 16),
           xb = aligned(size_t(M) * K / 2), xsb = aligned(size_t(useTma ? (M + 255) / 256 * 256 : M) * K / 16), xgb = aligned(size_t(M) * 4),
           ob = size_t(M) * N * 4;
    size_t inputBytes = wb + wsb + xb + xsb + xgb;
    if (inputBytes > State::capacity || (!useTma && inputBytes + ob > State::capacity))
        return false;
    uint8_t *qw = s->scratch, *sw = qw + wb, *qx = sw + wsb, *sx = qx + xb;
    float *gx = reinterpret_cast<float *>(sx + xsb), *tmp = reinterpret_cast<float *>(sx + xsb + xgb);
    if (nativeLayout) {
        qw = const_cast<uint8_t *>(reinterpret_cast<const uint8_t *>(weight));
        sw = const_cast<uint8_t *>(scales);
    } else {
    RepackFp4<<<dim3((N + 63) / 64, (K + 255) / 256), 256, 0, cudaStreamPerThread>>>(weight, scales, qw,
                                                                                         sw, N, packedN, K);
    CheckCuda(cudaGetLastError(), "native FP4 weight repack");
    }
#ifdef FASTLLM_NATIVE_PREFILL_SM120
    if (useTma) {
        QuantFp4<true><<<outputRows, 256, 0, cudaStreamPerThread>>>(input, qx, sx, gx, K, outputRows);
        CheckCuda(cudaGetLastError(), "native FP4 TMA quantization");
        if (TryNvfp4Tma(qx, sx, qw, sw, gx, global, output, outputRows, N, K, mode)) {
            CheckCuda(cudaEventRecord(s->ready, cudaStreamPerThread), "native FP4 TMA completion");
            return true;
        }
        // A descriptor/layout miss must regenerate the cuBLAS scale layout;
        // the packed activation values and model weights are unchanged.
    }
#endif
    if (inputBytes + ob > State::capacity) {
        CheckCuda(cudaEventRecord(s->ready, cudaStreamPerThread), "native FP4 capacity fallback");
        return false;
    }
    Plan *p = GetPlan(*s, 4, M, N, K);
    if (!p) {
        CheckCuda(cudaEventRecord(s->ready, cudaStreamPerThread), "native FP4 plan fallback");
        return false;
    }
    if (cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sw, sizeof(sw)) !=
            CUBLAS_STATUS_SUCCESS ||
        cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sx, sizeof(sx)) !=
            CUBLAS_STATUS_SUCCESS) {
        CheckCuda(cudaEventRecord(s->ready, cudaStreamPerThread), "native FP4 descriptor fallback");
        return false;
    }
    QuantFp4<false><<<M, 256, 0, cudaStreamPerThread>>>(input, qx, sx, gx, K, outputRows);
    CheckCuda(cudaGetLastError(), "native FP4 quantization");
    Tune(*s, *p, qw, qx, tmp);
    if (!p->valid) {
        CheckCuda(cudaEventRecord(s->ready, cudaStreamPerThread), "native FP4 fallback event");
        return false;
    }
    const float alpha = 1, beta = 0;
    auto status = cublasLtMatmul(s->handle, p->op, &alpha, qw, p->a, qx, p->b, &beta, tmp, p->c, tmp, p->c,
                                 &p->algo, s->workspace, State::workspaceBytes, cudaStreamPerThread);
    if (status == CUBLAS_STATUS_SUCCESS) {
        if (mode == 1)
            ScaleFp4Output<1><<<dim3((N / 2 + 1023) / 1024, outputRows), 256, 0, cudaStreamPerThread>>>(
                tmp, output, gx, global, bias, outputRows, N);
        else if (mode == 2)
            ScaleFp4Output<2><<<dim3((N + 1023) / 1024, outputRows), 256, 0, cudaStreamPerThread>>>(
                tmp, output, gx, global, bias, outputRows, N);
        else
            ScaleFp4Output<0><<<dim3((N + 1023) / 1024, outputRows), 256, 0, cudaStreamPerThread>>>(
                tmp, output, gx, global, bias, outputRows, N);
    }
    CheckCuda(cudaGetLastError(), "native prefill output");
    CheckCuda(cudaEventRecord(s->ready, cudaStreamPerThread), "native prefill completion event");
    return status == CUBLAS_STATUS_SUCCESS;
}
bool Supported(int bits) {
    int dev = 0, major = 0, minor = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess)
        return false;
    if (bits == 8 ? !(major >= 9 || (major == 8 && minor >= 9))
                  : !(bits == 4 && (major == 10 || major == 12)))
        return false;
    static thread_local std::map<std::pair<int, int>, bool> images;
    auto key = std::make_pair(dev, bits);
    auto found = images.find(key);
    if (found != images.end())
        return found->second;
    cudaFuncAttributes attr{};
    auto error =
        bits == 8 ? cudaFuncGetAttributes(&attr, QuantFp8) : cudaFuncGetAttributes(&attr, QuantFp4<false>);
    if (error != cudaSuccess)
        cudaGetLastError();
    return images.emplace(key, error == cudaSuccess && attr.maxThreadsPerBlock >= 256).first->second;
}

} // namespace fastllm_native_prefill

#else
namespace fastllm_native_prefill {
bool Enabled(const char *) { return false; }
bool Supported(int) { return false; }
bool Fp8(const half *, const uint8_t *, const float *, const half *, half *, int, int, int, int) {
    return false;
}
bool Fp4(const half *, const uint32_t *, const uint8_t *, const float *, const half *, half *, int, int, int,
         int, int, bool) {
    return false;
}
} // namespace fastllm_native_prefill
#endif
