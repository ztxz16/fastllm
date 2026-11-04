// Test-only LD_PRELOAD fault injector; never linked into FastLLM.
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdlib>
#include <cstdio>
static bool Reject(size_t bytes, const char *overrideName = nullptr) {
    const char *limit = overrideName ? std::getenv(overrideName) : nullptr;
    if (!limit) limit = std::getenv("FASTLLM_TEST_PIN_MAX_BYTES");
    return limit && bytes > std::strtoull(limit, nullptr, 10);
}
extern "C" cudaError_t cudaHostRegister(void *ptr, size_t bytes, unsigned int flags) {
    if (Reject(bytes)) return cudaErrorInvalidValue;
    using Fn = cudaError_t (*)(void *, size_t, unsigned int);
    static Fn next = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "cudaHostRegister"));
    return next ? next(ptr, bytes, flags) : cudaErrorUnknown;
}
extern "C" cudaError_t cudaHostAlloc(void **ptr, size_t bytes, unsigned int flags) {
    if (Reject(bytes, "FASTLLM_TEST_HOST_ALLOC_MAX_BYTES")) { *ptr = nullptr; return cudaErrorInvalidValue; }
    using Fn = cudaError_t (*)(void **, size_t, unsigned int);
    static Fn next = reinterpret_cast<Fn>(dlsym(RTLD_NEXT, "cudaHostAlloc"));
    return next ? next(ptr, bytes, flags) : cudaErrorUnknown;
}
