#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "fastllm-cuda.cuh"

#define GGML_COMMON_DECL_CUDA
#define GGML_COMMON_IMPL_CUDA
#include "gguf.h"

#define STRINGIZE_IMPL(...) #__VA_ARGS__
#define STRINGIZE(...) STRINGIZE_IMPL(__VA_ARGS__)

#define WARP_SIZE 32
#define CC_PASCAL 600
#define MIN_CC_DP4A 610
#define CC_VOLTA 700
#define CC_TURING 750
#define CC_AMPERE 800
#define CC_OFFSET_AMD 1000000
#define CC_RDNA1 (CC_OFFSET_AMD + 1010)
#define GGML_CUDA_MAX_DEVICES 16

#if CUDART_VERSION >= 11100
#define GGML_CUDA_ASSUME(x) __builtin_assume(x)
#else
#define GGML_CUDA_ASSUME(x)
#endif

#if __CUDA_ARCH__ >= CC_TURING
#define INT8_MMA_AVAILABLE
#endif

#ifdef __CUDA_ARCH__
#define NO_DEVICE_CODE asm("trap;")
#else
#define NO_DEVICE_CODE
#endif

[[noreturn]] static inline void fastllm_gguf_mmq_abort(const char *message) {
    std::fprintf(stderr, "FastLLM GGUF MMQ: %s\n", message);
    std::abort();
}

#define GGML_ABORT(...) fastllm_gguf_mmq_abort(__VA_ARGS__)

static inline void fastllm_gguf_mmq_cuda_check(
        cudaError_t status, const char *expression, const char *file, int line) {
    if (status == cudaSuccess) {
        return;
    }
    std::fprintf(stderr, "FastLLM GGUF MMQ CUDA error at %s:%d: %s: %s\n",
                 file, line, expression, cudaGetErrorString(status));
    std::abort();
}

#define CUDA_CHECK(expr) \
    fastllm_gguf_mmq_cuda_check((expr), #expr, __FILE__, __LINE__)

static constexpr bool int8_mma_available(const int cc) {
    return cc >= CC_TURING && cc < CC_OFFSET_AMD;
}

static __device__ __forceinline__ int ggml_cuda_dp4a(
        const int a, const int b, int c) {
#if __CUDA_ARCH__ >= MIN_CC_DP4A
    return __dp4a(a, b, c);
#else
    const int8_t *a8 = reinterpret_cast<const int8_t *>(&a);
    const int8_t *b8 = reinterpret_cast<const int8_t *>(&b);
    return c + a8[0] * b8[0] + a8[1] * b8[1] +
           a8[2] * b8[2] + a8[3] * b8[3];
#endif
}

static __device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        value += __shfl_xor_sync(0xffffffff, value, mask, WARP_SIZE);
    }
    return value;
}

static __device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        value = fmaxf(value,
                      __shfl_xor_sync(0xffffffff, value, mask, WARP_SIZE));
    }
    return value;
}

// Lookup tables referenced by the imported ik_llama templates. IQ4_XS uses
// kvalues_iq4nl directly; the remaining small tables keep the uninstantiated
// optional quant formats well-formed. iq1s_grid_gpu is populated from the
// canonical packed iq1s_grid before the first IQ1 kernel launch (see the CUDA
// translation unit); keeping it as a device table preserves ik_llama's hot
// lookup path without duplicating the 2048-entry table in this adapter.
static constexpr __device__ int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

static constexpr __device__ int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

static constexpr __device__ int8_t iq3nl_values[16] = {
    -63, -40, -23, -10, 1, 13, 28, 47,
    -59, -36, -19,  -6, 5, 17, 32, 51,
};

static constexpr __device__ int8_t iq4k_values[32] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
    -123, -100, -79, -61, -45, -31, -18,  -6,
       5,   17,  29,  42,  57,  73,  93, 117,
};

static constexpr __device__ int8_t iq5nl_values[64] = {
    -126, -114, -103, -92, -83, -74, -65, -57,
     -50,  -43,  -36, -30, -24, -18, -12,  -6,
      -1,    5,   11,  17,  23,  29,  36,  43,
      51,   59,   68,  77,  87,  97, 109, 121,
    -124, -112, -101, -90, -81, -72, -63, -55,
     -48,  -41,  -34, -28, -22, -16, -10,  -4,
       1,    7,   13,  19,  25,  31,  38,  45,
      53,   61,   70,  79,  89,  99, 111, 123,
};

static constexpr __device__ int8_t iq6nl_values[128] = {
    -127, -121, -115, -109, -104, -98, -93, -88,
     -84,  -79,  -74,  -70,  -66, -62, -58, -54,
     -51,  -47,  -44,  -40,  -37, -34, -31, -28,
     -25,  -22,  -19,  -16,  -13, -11,  -8,  -5,
      -2,    0,    3,    6,    9,  12,  14,  17,
      20,   23,   27,   30,   33,  36,  40,  44,
      47,   51,   55,   59,   63,  68,  72,  77,
      82,   87,   92,   98,  103, 109, 115, 121,
    -126, -120, -114, -108, -103, -97, -92, -87,
     -83,  -78,  -73,  -69,  -65, -61, -57, -53,
     -50,  -46,  -43,  -39,  -36, -33, -30, -27,
     -24,  -21,  -18,  -15,  -12, -10,  -7,  -4,
      -1,    1,    4,    7,   10,  13,  15,  18,
      21,   24,   28,   31,   34,  37,  41,  45,
      48,   52,   56,   60,   64,  69,  73,  78,
      83,   88,   93,   99,  104, 110, 116, 122,
};

static __device__ uint32_t iq1s_grid_gpu[NGRID_IQ1S] = {};

// These layouts are used only by optional ik_llama MMQ template definitions.
// FastLLM's current GGUF header intentionally omits them, so keep the minimal
// compatible declarations local to this translation unit.
#ifndef QK_MXFP4
#define QK_MXFP4 32
#endif

struct block_mxfp4 {
    uint8_t e;
    uint8_t qs[QK_MXFP4 / 2];
};

struct block_iq1_s_r4 {
    uint8_t qs[16];
    uint16_t qh[4];
};

struct block_iq2_k {
    ggml_half d;
    uint16_t extra;
    uint8_t scales[QK_K / 32];
    uint8_t qs[QK_K / 4];
};

struct block_iq2_ks {
    uint16_t extra;
    uint8_t scales[QK_K / 64];
    uint8_t qs[QK_K / 4];
};

struct block_iq4_ks {
    uint8_t scales[QK_K / 32];
    uint8_t qs[QK_K / 2];
};

struct block_iq4_ks_r4 {
    uint8_t scales[QK_K / 8];
    uint8_t qs[QK_K * 2];
};

// MXFP4 is not a public FastLLM GGUF type yet. A private unused value keeps
// the imported template definitions well-formed without exposing the type.
#define GGML_TYPE_MXFP4 static_cast<ggml_type>(39)

template <ggml_type type>
struct ggml_cuda_type_traits;

#define FASTLLM_GGUF_MMQ_TRAITS(type_name, qk_value, qr_value, qi_value) \
    template <>                                                           \
    struct ggml_cuda_type_traits<type_name> {                              \
        static constexpr int qk = qk_value;                               \
        static constexpr int qr = qr_value;                               \
        static constexpr int qi = qi_value;                               \
    }

FASTLLM_GGUF_MMQ_TRAITS(GGML_TYPE_Q4_K, QK_K, QR4_K, QI4_K);
FASTLLM_GGUF_MMQ_TRAITS(GGML_TYPE_Q5_K, QK_K, QR5_K, QI5_K);
FASTLLM_GGUF_MMQ_TRAITS(GGML_TYPE_IQ4_XS, QK_K, QR4_XS, QI4_XS);
FASTLLM_GGUF_MMQ_TRAITS(GGML_TYPE_Q3_K, QK_K, QR3_K, QI3_K);
FASTLLM_GGUF_MMQ_TRAITS(GGML_TYPE_Q6_K, QK_K, QR6_K, QI6_K);
FASTLLM_GGUF_MMQ_TRAITS(GGML_TYPE_IQ3_S, QK_K, QR3_S, QI3_S);
FASTLLM_GGUF_MMQ_TRAITS(GGML_TYPE_IQ4_NL, QK4_NL, QR4_NL, QI4_NL);
FASTLLM_GGUF_MMQ_TRAITS(GGML_TYPE_Q4_0, QK4_0, QR4_0, QI4_0);
FASTLLM_GGUF_MMQ_TRAITS(GGML_TYPE_Q4_1, QK4_1, QR4_1, QI4_1);
FASTLLM_GGUF_MMQ_TRAITS(GGML_TYPE_Q8_0, QK8_0, QR8_0, QI8_0);
FASTLLM_GGUF_MMQ_TRAITS(GGML_TYPE_IQ2_XXS, QK_K, QR2_XXS, QI2_XXS);
FASTLLM_GGUF_MMQ_TRAITS(GGML_TYPE_IQ2_XS, QK_K, QR2_XS, QI2_XS);
FASTLLM_GGUF_MMQ_TRAITS(GGML_TYPE_IQ2_S, QK_K, QR2_S, QI2_S);
FASTLLM_GGUF_MMQ_TRAITS(GGML_TYPE_IQ1_S, QK_K, QR1_S, QI1_S);
FASTLLM_GGUF_MMQ_TRAITS(GGML_TYPE_IQ1_M, QK_K, QR1_M, QI1_M);

#undef FASTLLM_GGUF_MMQ_TRAITS

namespace fastllm_gguf_mmq {

struct ggml_cuda_device_info {
    struct cuda_device_info {
        int cc = 0;
        int nsm = 0;
        size_t smpbo = 0;
    };

    cuda_device_info devices[GGML_CUDA_MAX_DEVICES] = {};
};

static inline int ggml_cuda_get_device() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

static inline ggml_cuda_device_info &ggml_cuda_info() {
    static ggml_cuda_device_info info;
    const int device = ggml_cuda_get_device();
    if (info.devices[device].cc == 0) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        info.devices[device].cc = prop.major * 100 + prop.minor * 10;
        info.devices[device].nsm = prop.multiProcessorCount;
        info.devices[device].smpbo = prop.sharedMemPerBlockOptin;
    }
    return info;
}

struct ggml_cuda_pool {};

template <typename T>
struct ggml_cuda_pool_alloc {
    T *ptr = nullptr;

    ggml_cuda_pool_alloc(ggml_cuda_pool &, size_t count)
        : ptr(static_cast<T *>(FastllmCudaMalloc(count * sizeof(T)))) {
        if (ptr == nullptr) {
            fastllm_gguf_mmq_abort("temporary allocation failed");
        }
    }

    ~ggml_cuda_pool_alloc() {
        FastllmCudaFree(ptr);
    }

    ggml_cuda_pool_alloc(const ggml_cuda_pool_alloc &) = delete;
    ggml_cuda_pool_alloc &operator=(const ggml_cuda_pool_alloc &) = delete;
};

struct ggml_backend_cuda_context {
    ggml_cuda_pool scratch_pool;

    ggml_cuda_pool &pool(int) {
        return scratch_pool;
    }
};

} // namespace fastllm_gguf_mmq
