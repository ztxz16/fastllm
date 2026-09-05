#include "fastllm-cuda-record-copy.cuh"
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

static void Check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}

static void Run(size_t bytes, size_t sourcePitch, size_t destinationPitch,
                int sourceOffset, int destinationOffset, bool mappedHost, bool wideIndex = false) {
    constexpr int records = 19, destinations = 31, maxCopies = 17;
    size_t sourceSize = sourceOffset + records * sourcePitch + 32;
    size_t destinationSize = destinationOffset + destinations * destinationPitch + 32;
    std::vector<uint8_t> input(sourceSize), initial(destinationSize, 0xa5);
    std::mt19937 rng(1927);
    for (auto &x : input) x = uint8_t(rng());
    uint8_t *source, *destination, *host = nullptr;
    if (mappedHost) {
        Check(cudaHostAlloc(&host, sourceSize, cudaHostAllocMapped));
        std::memcpy(host, input.data(), sourceSize);
        Check(cudaHostGetDevicePointer(&source, host, 0));
    } else {
        Check(cudaMalloc(&source, sourceSize));
        Check(cudaMemcpy(source, input.data(), sourceSize, cudaMemcpyHostToDevice));
    }
    Check(cudaMalloc(&destination, destinationSize));
    std::vector<int32_t> sourceIds(maxCopies), destinationIds(maxCopies);
    for (int i = 0; i < maxCopies; ++i) {
        sourceIds[i] = (i * 7) % records;
        destinationIds[i] = (i * 13) % destinations;
    }
    int32_t *s, *d, *count;
    Check(cudaMalloc(&s, maxCopies * sizeof(int32_t)));
    Check(cudaMalloc(&d, maxCopies * sizeof(int32_t)));
    Check(cudaMalloc(&count, sizeof(int32_t)));
    Check(cudaMemcpy(s, sourceIds.data(), maxCopies * sizeof(int32_t), cudaMemcpyHostToDevice));
    Check(cudaMemcpy(d, destinationIds.data(), maxCopies * sizeof(int32_t), cudaMemcpyHostToDevice));
    cudaDeviceProp props; Check(cudaGetDeviceProperties(&props, 0));
    auto launch = fastllm::cuda::RecordCopyConfiguration(bytes, maxCopies, props);
    cudaStream_t stream; Check(cudaStreamCreate(&stream));
    cudaGraph_t graph; cudaGraphExec_t exec;
    Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    if (wideIndex) {
        // Exercise 64-bit indexing with a small allocation as well as
        // compiling the automatic dispatch for batches larger than 2^32 units.
        fastllm::cuda::record_copy_detail::CopyKernel<uint8_t, uint64_t>
            <<<launch.blocks, launch.threads, 0, stream>>>(
                {source + sourceOffset, destination + destinationOffset, sourcePitch, destinationPitch, bytes},
                s, d, count, uint64_t(bytes));
    } else if (!fastllm::cuda::CopyRecords(
            {source + sourceOffset, destination + destinationOffset, sourcePitch, destinationPitch, bytes},
            s, d, count, maxCopies, launch, stream)) throw std::runtime_error("launch rejected");
    Check(cudaStreamEndCapture(stream, &graph));
    Check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int copies : {0, 1, 7, 17}) {
        Check(cudaMemcpy(count, &copies, sizeof(int), cudaMemcpyHostToDevice));
        Check(cudaMemcpy(destination, initial.data(), destinationSize, cudaMemcpyHostToDevice));
        std::vector<uint8_t> expected = initial, actual(destinationSize);
        for (int i = 0; i < copies; ++i)
            std::memcpy(expected.data() + destinationOffset + destinationIds[i] * destinationPitch,
                        input.data() + sourceOffset + sourceIds[i] * sourcePitch, bytes);
        Check(cudaGraphLaunch(exec, stream));
        Check(cudaGraphLaunch(exec, stream));
        Check(cudaStreamSynchronize(stream));
        Check(cudaMemcpy(actual.data(), destination, destinationSize, cudaMemcpyDeviceToHost));
        if (actual != expected) throw std::runtime_error("copy or padding/guard mismatch");
    }
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
    if (mappedHost) cudaFreeHost(host); else cudaFree(source);
    cudaFree(destination); cudaFree(s); cudaFree(d); cudaFree(count);
    std::printf("PASS bytes=%zu pitches=%zu/%zu offsets=%d/%d mapped=%d wide=%d graph and guards\n",
                bytes, sourcePitch, destinationPitch, sourceOffset, destinationOffset, mappedHost, wideIndex);
}

int main() {
    try {
        for (bool mapped : {false, true}) {
            Run(0, 16, 32, 0, 0, mapped);
            Run(1, 3, 5, 1, 2, mapped);
            Run(15, 31, 29, 0, 1, mapped);
            Run(16, 32, 48, 0, 0, mapped);
            Run(127, 128, 256, 0, 0, mapped);
            Run(128, 160, 192, 0, 0, mapped);
            Run(1025, 1031, 1041, 7, 3, mapped);
            Run(8192, 8320, 8448, 0, 0, mapped);
            Run(2764928, 2764928, 2765056, 0, 0, mapped);
            Run(1025, 1031, 1041, 7, 3, mapped, true);
        }
        // Invalid layouts must be rejected before any kernel is enqueued.
        if (fastllm::cuda::CopyRecords({nullptr, nullptr, 16, 8, 16}, nullptr,
                                      nullptr, nullptr, 1, {1, 32}, nullptr))
            throw std::runtime_error("invalid layout accepted");
        std::puts("ALL_PASS"); return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1;
    }
}
