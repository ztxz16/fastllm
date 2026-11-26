#pragma once

// Readiness of a producer CTA's output within the same GPU.
namespace fastllm_swa {
__device__ __forceinline__ void PublishReady(unsigned *flag) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
    __threadfence();
    *static_cast<volatile unsigned *>(flag) = 1;
#else
    asm volatile("st.release.gpu.global.u32 [%0],1;" :: "l"(flag) : "memory");
#endif
}

__device__ __forceinline__ void WaitReady(const unsigned *flag) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
    while (!*static_cast<const volatile unsigned *>(flag)) {}
    __threadfence();
#else
    unsigned ready;
    do {
        asm volatile("ld.acquire.gpu.global.u32 %0,[%1];" : "=r"(ready) : "l"(flag) : "memory");
    } while (!ready);
#endif
}
} // namespace fastllm_swa
