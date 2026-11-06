#include "devices/cuda/fastllm-cuda-mapped-host.h"
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
static void Check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
__global__ void CopyMapped(const uint8_t *source, uint8_t *target, size_t bytes) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < bytes;
         i += size_t(gridDim.x) * blockDim.x) target[i] = source[i];
}
int main(int argc, char **argv) {
    try {
        using fastllm::cuda::MappedHostBuffer;
        const bool failed = argc > 1 && std::string(argv[1]) == "--failure";
        const bool split = argc > 1 && std::string(argv[1]) == "--split";
        const size_t bytes = (size_t(32) << 20) + 37;
        MappedHostBuffer owner;
        const auto state = owner.Allocate(bytes);
        if (failed) {
            Check(state != cudaSuccess && !owner.Host() && !owner.Device(), "failed allocation leaked storage");
            Check(cudaGetLastError() == cudaSuccess, "allocation failure poisoned CUDA state");
            std::puts("PASS mapped host permanent failure cleanup");
            return 0;
        }
        Check(state == cudaSuccess, "mapped allocation failed");
        if (split) Check(owner.Registrations() >= 3, "bounded registration fallback not exercised");
        auto *host = static_cast<uint8_t *>(owner.Host());
        for (size_t i = 0; i < bytes; ++i) host[i] = (i * 31 + i / 4093) & 255;
        uint8_t *target = nullptr;
        Check(cudaMalloc(&target, bytes) == cudaSuccess, "device allocation failed");
        CopyMapped<<<80, 256>>>(static_cast<uint8_t *>(owner.Device()), target, bytes);
        std::vector<uint8_t> actual(bytes);
        Check(cudaMemcpy(actual.data(), target, bytes, cudaMemcpyDeviceToHost) == cudaSuccess,
              "cross-registration CUDA reads failed");
        for (size_t i = 0; i < bytes; ++i) Check(host[i] == actual[i], "mapped record bytes differ");
        // Include entire records and unaligned spans crossing every 16 MiB
        // registration boundary. Direct GPU loads alone do not validate DMA.
        const std::vector<std::pair<size_t, size_t>> ranges = {
            {0, bytes}, {(size_t(16) << 20) - 153608, 307216},
            {(size_t(32) << 20) - 17, 54}, {19, 4099}, {bytes, 0}};
        for (const auto &range : ranges) {
            Check(owner.CopyToDeviceAsync(target, host + range.first, range.second, 0) == cudaSuccess,
                  "cross-registration DMA submission failed");
            if (range.second) {
                Check(cudaMemcpy(actual.data(), target, range.second, cudaMemcpyDeviceToHost) == cudaSuccess,
                      "cross-registration DMA completion failed");
                for (size_t i = 0; i < range.second; ++i)
                    Check(actual[i] == host[range.first+i], "cross-registration DMA changed bytes");
            }
        }
        Check(owner.CopyToDeviceAsync(target, host+bytes, 1, 0) == cudaErrorInvalidValue,
              "out-of-bounds DMA range accepted");
        cudaFree(target);
        void *retired = owner.Host();
        owner.Reset();
        unsigned int flags = 0;
        Check(cudaHostGetFlags(&flags, retired) != cudaSuccess, "mapping survived last owner");
        cudaGetLastError();
        Check(owner.Allocate(4096) == cudaSuccess, "allocation after retirement failed");
        std::puts("PASS mapped host contiguous CUDA reads and registration lifetime");
        return 0;
    } catch (const std::exception &e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
