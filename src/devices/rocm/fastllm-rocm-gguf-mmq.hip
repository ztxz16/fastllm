// ROCm-only quantized matrix multiplication. See gguf_mmq/README.md and LICENSE.
#ifndef USE_ROCM
#error "This file is only part of the ROCm backend"
#endif
#define GGML_USE_HIP
#include "gguf_mmq/common.cuh"
#include <cstring>

// Match the optional FastLLM allocator ABI without importing the CUDA GGUF types.
enum FastllmCudaTryMallocResult {
    FASTLLM_CUDA_TRY_MALLOC_SUCCESS = 0,
    FASTLLM_CUDA_TRY_MALLOC_CAPACITY_FAILURE = 1,
    FASTLLM_CUDA_TRY_MALLOC_ERROR = 2,
};
extern "C" FastllmCudaTryMallocResult FastllmCudaTryMalloc(void **ret, size_t size);
extern "C" void FastllmCudaFree(void *ret);

namespace fastllm_rocm_mmq {
#include "gguf_mmq/mmq.cuh"
#include "gguf_mmq/policy.cuh"

struct Workspace {
    void *ptr = nullptr;
    bool alloc(size_t bytes) { return FastllmCudaTryMalloc(&ptr, bytes) == FASTLLM_CUDA_TRY_MALLOC_SUCCESS; }
    ~Workspace() { if (ptr) FastllmCudaFree(ptr); }
    Workspace() = default;
    Workspace(const Workspace &) = delete;
    Workspace &operator=(const Workspace &) = delete;
};

// Read each activation once. Layout and subgroup scales match llama.cpp's
// quantize_mmq_q8_1, with direct FP16/BF16 input and defined all-zero handling.
template<typename T, mmq_q8_1_ds_layout layout>
static __global__ void quantize_input(const T *input, block_q8_1_mmq *y, int n, int m) {
    const int i = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (i >= m) return;
    constexpr int group = layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int sum_group = layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;
    const T *p = input + size_t(blockIdx.x) * m + i;
    float v[4] = {float(p[0]), float(p[1]), float(p[2]), float(p[3])};
    float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    float sum = v[0] + v[1] + v[2] + v[3];
#pragma unroll
    for (int d = group/8; d; d >>= 1) amax = fmaxf(amax, __shfl_xor(amax, d, 32));
    if constexpr (layout != MMQ_Q8_1_DS_LAYOUT_D4) {
#pragma unroll
        for (int d = sum_group/8; d; d >>= 1) sum += __shfl_xor(sum, d, 32);
    }
    const float inv = amax > 0 ? 127.0f / amax : 0.0f;
    const float scale = amax > 0 ? 1.0f / inv : 0.0f;
    char4 q = {int8_t(roundf(v[0]*inv)), int8_t(roundf(v[1]*inv)),
               int8_t(roundf(v[2]*inv)), int8_t(roundf(v[3]*inv))};
    auto &b = y[size_t(i/128)*n + blockIdx.x];
    const int j = i % 128;
    reinterpret_cast<char4 *>(b.qs)[j/4] = q;
    if constexpr (layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (j%16 == 0 && j < 96) b.d2s6[2+j/16] = half(sum);
        if (j%64 == 0) b.d2s6[j/64] = half(scale);
    } else if (j%32 == 0) {
        if constexpr (layout == MMQ_Q8_1_DS_LAYOUT_DS4) b.ds4[j/32] = make_half2(scale, sum);
        else b.d4[j/32] = scale;
    }
}

template<typename T>
static __global__ void convert_output(const float *src, T *dst, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = T(src[i]);
}

template<ggml_type type, int J, bool fallback>
static void launch(const void *weight, const void *quantized, float *dst, float *fixup,
                   int n, int m, int k, int cc, int nsm, int warp, hipStream_t stream) {
    const auto config = ggml_cuda_mmq_get_config(type, J, fallback, cc);
    const dim3 threads(warp, config.nthreads/warp, 1);
    const int nty = (int64_t(k) + config.I-1)/config.I, ntx = (int64_t(n)+J-1)/J;
    const bool stream_k = ggml_cuda_mmq_get_stream_k(type, J, fallback, cc);
    const dim3 blocks = stream_k ? dim3(nsm,1,1) : dim3(nty,ntx,1);
    const bool needs_fixup = stream_k && (ntx*nty)%nsm != 0;
    const uint3 kb = init_fastdiv_values(m/ggml_cuda_type_traits<type>::qk);
    const uint3 tx = init_fastdiv_values(ntx), one = init_fastdiv_values(1);
    mul_mat_q<type,J,fallback><<<blocks,threads,mmq_get_nbytes_shared(config,cc),stream>>>(
        static_cast<const char *>(weight),static_cast<const int *>(quantized),nullptr,nullptr,dst,
        needs_fixup ? fixup : nullptr,nullptr,kb,k,n,m/ggml_cuda_type_traits<type>::qk,n,k,
        one,one,0,0,0,one,one,0,0,0,tx);
    if (needs_fixup) {
        CUDA_CHECK(hipGetLastError());
        mul_mat_q_stream_k_fixup<type,J,fallback><<<dim3(nsm,config.I/warp,1),dim3(warp,threads.y/2,1),0,stream>>>(
            nullptr,nullptr,dst,fixup,kb,k,n,k,one,0,one,0,tx);
    }
}

template<ggml_type type>
static void dispatch(const void *weight, const void *quantized, float *dst, float *fixup,
                     int n, int m, int k, int cc, int nsm, int warp, int J, hipStream_t stream) {
    if (k%128) {
        if (J == 64) launch<type,64,true>(weight,quantized,dst,fixup,n,m,k,cc,nsm,warp,stream);
        else launch<type,128,true>(weight,quantized,dst,fixup,n,m,k,cc,nsm,warp,stream);
    } else {
        if (J == 64) launch<type,64,false>(weight,quantized,dst,fixup,n,m,k,cc,nsm,warp,stream);
        else launch<type,128,false>(weight,quantized,dst,fixup,n,m,k,cc,nsm,warp,stream);
    }
}

#define MMQ_TYPES(X) \
 X(Q8_0) X(Q2_K) X(Q3_K) X(Q4_K) X(Q5_K) X(Q6_K) \
 X(IQ2_XS) X(IQ2_S) X(IQ3_XXS) X(IQ3_S) X(IQ4_NL) X(IQ4_XS)

template<typename T>
static bool matmul(const void *input, const void *weight, void *output, int ggmlType,
                   int n, int m, int k, void *raw_stream) {
    if (n <= 8 || m <= 0 || m%256 || k <= 0 || !input || !weight || !output) return false;
    const auto type = ggml_type(ggmlType);
    switch (type) {
#define SUPPORTED(t) case GGML_TYPE_##t: break;
        MMQ_TYPES(SUPPORTED)
#undef SUPPORTED
        default: return false;
    }
    int device = 0;
    CUDA_CHECK(hipGetDevice(&device));
    // Immutable per-device properties; never retain workspaces across devices or streams.
    static const std::vector<hipDeviceProp_t> properties = [] {
        int count = 0; CUDA_CHECK(hipGetDeviceCount(&count));
        std::vector<hipDeviceProp_t> p(count);
        for (int i=0; i<count; ++i) CUDA_CHECK(hipGetDeviceProperties(&p[i],i));
        return p;
    }();
    const auto &p = properties.at(device);
    const int cc = mmq_device_cc(p.gcnArchName, p.warpSize);
    if (cc < 0 || p.multiProcessorCount <= 0) return false;
    const auto config = mmq_select_config(type, n, k, cc, p.warpSize,
                                          p.sharedMemPerBlock, p.maxThreadsPerBlock);
    if (config.type == GGML_TYPE_COUNT) return false;
    const int J = config.J;
    const size_t ntiles = ((size_t(k) + config.I - 1) / config.I) *
                          ((size_t(n) + J - 1) / J);
    const int qk = (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_IQ4_NL) ? 32 : 256;
    // Upstream stream-K uses a signed 32-bit linear tile/K index.
    if (config.stream_k && ntiles * (m / qk) >= (size_t(1) << 30)) return false;
    const bool needs_fixup = config.stream_k && ntiles % p.multiProcessorCount != 0;
    const size_t qbytes = (size_t(n)*m/128+128)*sizeof(block_q8_1_mmq);
    const size_t obytes = size_t(n)*k*sizeof(float);
    const size_t fbytes = needs_fixup
        ? size_t(p.multiProcessorCount)*J*config.I*sizeof(float) : 0;
    // Keep optional acceleration bounded, including giant output projections.
    if (qbytes + obytes + fbytes > size_t(256)*1024*1024 || size_t(n)*k > INT_MAX) return false;
    Workspace q, out, fixup;
    if (!q.alloc(qbytes) || !out.alloc(obytes) || (fbytes && !fixup.alloc(fbytes))) return false;
    auto stream = reinterpret_cast<hipStream_t>(raw_stream);
    const dim3 grid(n,(m+1023)/1024,1), threads(256,1,1);
    switch (mmq_get_q8_1_ds_layout(type)) {
#define QUANT(l) case l: quantize_input<T,l><<<grid,threads,0,stream>>>(static_cast<const T *>(input),static_cast<block_q8_1_mmq *>(q.ptr),n,m); break;
        QUANT(MMQ_Q8_1_DS_LAYOUT_D4)
        QUANT(MMQ_Q8_1_DS_LAYOUT_DS4)
        QUANT(MMQ_Q8_1_DS_LAYOUT_D2S6)
#undef QUANT
    }
    CUDA_CHECK(hipGetLastError());
    switch (type) {
#define RUN(t) case GGML_TYPE_##t: dispatch<GGML_TYPE_##t>(weight,q.ptr,static_cast<float *>(out.ptr),static_cast<float *>(fixup.ptr),n,m,k,cc,p.multiProcessorCount,p.warpSize,J,stream); break;
        MMQ_TYPES(RUN)
#undef RUN
        default: GGML_ABORT("Unexpected MMQ type");
    }
    CUDA_CHECK(hipGetLastError());
    convert_output<T><<<(size_t(n)*k+255)/256,256,0,stream>>>(static_cast<const float *>(out.ptr),static_cast<T *>(output),size_t(n)*k);
    CUDA_CHECK(hipGetLastError());
    return true;
}
} // namespace fastllm_rocm_mmq

extern "C" bool FastllmCudaHalfMatMulGGUFMMQ(const void *input, const void *weight, void *output,
                                int type, int n, int m, int k, void *stream) {
    return fastllm_rocm_mmq::matmul<half>(input,weight,output,type,n,m,k,stream);
}
extern "C" bool FastllmCudaBFloat16MatMulGGUFMMQ(const void *input, const void *weight, void *output,
                                    int type, int n, int m, int k, void *stream) {
    return fastllm_rocm_mmq::matmul<__hip_bfloat16>(input,weight,output,type,n,m,k,stream);
}
