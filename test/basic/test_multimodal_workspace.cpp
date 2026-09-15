#include "devices/cuda/cudaworkspace.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "utils/utils.h"

#include <iostream>
#include <stdexcept>
#include <vector>

using namespace fastllm;

namespace {
void Require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

void CheckArena() {
    constexpr size_t capacity = 1 << 20;
    auto arena = std::make_shared<CudaWorkspace>(0, capacity);
    std::weak_ptr<CudaWorkspace> weak = arena;
    void *retained;
    {
        CudaWorkspaceScope scope(arena);
        void *a = FastllmCudaMalloc(capacity / 4);
        void *b = FastllmCudaMalloc(capacity / 4);
        void *c = FastllmCudaMalloc(capacity / 2);
        Require(a && b && c && arena->LiveBytes() == capacity, "arena capacity accounting");
        void *overflow = nullptr;
        Require(FastllmCudaTryMalloc(&overflow, 256) == FASTLLM_CUDA_TRY_MALLOC_CAPACITY_FAILURE,
                "exhausted workspace must not allocate outside its reservation");
        Require(!overflow && !FastllmCudaGetThreadError(), "optional allocation poisoned CUDA error state");
        FastllmCudaFree(b);
        FastllmCudaForceFree(a);
        void *ab = FastllmCudaMalloc(capacity / 2);
        Require(ab == a, "adjacent free ranges did not coalesce");
        FastllmCudaDirectFree(c);
        FastllmCudaFree(ab);
        {
            auto inner = std::make_shared<CudaWorkspace>(0, 4096);
            CudaWorkspaceScope nested(inner);
            void *p = FastllmCudaMalloc(4096);
            Require(inner->LiveBytes() == 4096 && arena->LiveBytes() == 0, "nested scope selection");
            FastllmCudaFree(p);
        }
        if (FastllmCudaGetDeviceCount() > 1) {
            FastllmCudaSetDevice(1);
            void *otherDevice = FastllmCudaMalloc(4096);
            Require(otherDevice && !IsCudaWorkspacePointer(otherDevice), "workspace crossed CUDA devices");
            FastllmCudaFree(otherDevice);
            FastllmCudaSetDevice(0);
        }
        retained = FastllmCudaMalloc(capacity);
        std::vector<float> source(1024, 3.25f), result(1024);
        FastllmCudaCopyFromHostToDevice(retained, source.data(), source.size() * sizeof(float));
        FastllmCudaCopyFromDeviceToHost(result.data(), retained, result.size() * sizeof(float));
        Require(source == result, "workspace storage data roundtrip");
        arena.reset();
    }
    Require(!weak.expired(), "outstanding tensor did not retain its workspace");
    Require(FastllmCudaFreeAfterCurrentThreadStream(retained), "deferred release missed workspace");
    Require(weak.expired(), "workspace leaked after last allocation was released");
    void *normal = FastllmCudaMalloc(4096);
    Require(normal && !IsCudaWorkspacePointer(normal), "scope leaked into ordinary CUDA allocations");
    FastllmCudaFree(normal);
}


}

int main() {
    if (FastllmCudaGetDeviceCount() < 1) return 77;
    FastllmCudaSetDevice(0);
    FastllmCudaClearThreadError();
    SetThreads(4);
    try {
        CheckArena();
        Require(!FastllmCudaGetThreadError(), "unexpected CUDA thread error");
        std::cout << "CUDA workspace: PASS\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
