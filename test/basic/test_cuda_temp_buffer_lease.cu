#include "fastllm.h"
#include "fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <thread>

static void Check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
__global__ void DelayedWrite(float *p, float value) {
    const unsigned long long start = clock64();
    while (clock64() - start < 40000000ULL) {}
    p[0] = value;
}
__global__ void CopyFirst(const float *p, float *out) { out[0] = p[0]; }

int main() {
    try {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) {
            std::puts("FASTLLM_TEST_SKIP_NO_DEVICE: requires a GPU");
            return 0;
        }
        Check(cudaSetDevice(0));
        int failures = 0;
        const size_t bytes = 1 << 20;

        // A borrow while the shared buffer is held gets private memory.
        bool ownA = true, ownB = false;
        void *a = FastllmBorrowCudaTempBuffer(bytes, nullptr, &ownA);
        void *b = FastllmBorrowCudaTempBuffer(bytes, nullptr, &ownB);
        if (a == nullptr || b == nullptr || a == b || ownA || !ownB) {
            std::printf("FAIL nested borrow: a=%p b=%p ownA=%d ownB=%d\n", a, b, ownA, ownB); ++failures;
        }
        FastllmReleaseCudaTempBuffer(b, ownB);
        FastllmReleaseCudaTempBuffer(a, ownA);
        bool ownC = true;
        void *c = FastllmBorrowCudaTempBuffer(bytes, nullptr, &ownC);
        if (c != a || ownC) {
            std::printf("FAIL reuse after release: a=%p c=%p ownC=%d\n", a, c, ownC); ++failures;
        }
        FastllmReleaseCudaTempBuffer(c, ownC);

        // Another thread's stream must not touch the buffer before the releaser's kernels finish.
        void *shared = nullptr;
        std::thread([&] {
            Check(cudaSetDevice(0));
            bool own = false;
            shared = FastllmBorrowCudaTempBuffer(bytes, nullptr, &own);
            Check(cudaMemsetAsync(shared, 0, sizeof(float), cudaStreamPerThread));
            DelayedWrite<<<1, 1, 0, cudaStreamPerThread>>>(static_cast<float *>(shared), 17.f);
            FastllmReleaseCudaTempBuffer(shared, own);
        }).join();
        float *out = nullptr;
        Check(cudaMalloc(&out, sizeof(float)));
        float seen = -1.f;
        std::thread([&] {
            Check(cudaSetDevice(0));
            bool own = true;
            void *p = FastllmBorrowCudaTempBuffer(bytes, nullptr, &own);
            if (p != shared || own) {
                std::printf("FAIL cross-thread borrow: shared=%p p=%p own=%d\n", shared, p, own); ++failures;
            }
            CopyFirst<<<1, 1, 0, cudaStreamPerThread>>>(static_cast<const float *>(p), out);
            Check(cudaMemcpyAsync(&seen, out, sizeof(float), cudaMemcpyDeviceToHost, cudaStreamPerThread));
            Check(cudaStreamSynchronize(cudaStreamPerThread));
            FastllmReleaseCudaTempBuffer(p, own);
        }).join();
        if (seen != 17.f) {
            std::printf("FAIL cross-stream ordering: read %g before the releaser's write\n", seen); ++failures;
        }
        Check(cudaDeviceSynchronize());
        Check(cudaFree(out));

        if (failures) return 1;
        std::puts("PASS: temp buffer lease and cross-stream release");
        return 0;
    } catch (const std::exception &e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
