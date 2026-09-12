// Exercise the production registration with delayed H2D completion and old
// pool writers. Both use different streams from the registering thread.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

namespace {
void *staging[2] = {};
bool delayMetadata = false;

void DelayCopy(void *) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
}

cudaError_t DelayedMetadataCopy(void *dest, const void *source, size_t bytes,
                                cudaMemcpyKind kind) {
    if (!delayMetadata || bytes != 8 * sizeof(void *) || kind != cudaMemcpyHostToDevice) {
        return cudaMemcpy(dest, source, bytes, kind);
    }
    int device = -1;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess || device < 0 || device >= 2) return cudaErrorInvalidDevice;
    // Start with a known incomplete table; keep the staged source alive until
    // every queued DMA finishes. No CUDA API is called by the host callback.
    status = cudaMemsetAsync(dest, 0, bytes, cudaStreamPerThread);
    if (status != cudaSuccess) return status;
    status = cudaStreamSynchronize(cudaStreamPerThread);
    if (status != cudaSuccess) return status;
    std::memcpy(staging[device], source, bytes);
    status = cudaLaunchHostFunc(cudaStreamPerThread, DelayCopy, nullptr);
    if (status != cudaSuccess) return status;
    return cudaMemcpyAsync(dest, staging[device], bytes, kind, cudaStreamPerThread);
}
}  // namespace

// Compile the real implementation, replacing only the timing of its pageable
// host-to-device copy. The shared library supplies the allocator/TP helpers.
#ifdef cudaMemcpy
#undef cudaMemcpy
#endif
#define cudaMemcpy DelayedMetadataCopy
#ifndef CUSTOM_AR_SOURCE
#define CUSTOM_AR_SOURCE "../../src/devices/multicuda/fastllm-custom-allreduce.cu"
#endif
#include CUSTOM_AR_SOURCE
#undef cudaMemcpy

static void Require(cudaError_t status, const char *where) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(status));
        std::exit(1);
    }
}

int main(int argc, char **argv) {
    const bool reuse = argc > 1 && std::strcmp(argv[1], "reuse") == 0;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) return 77;
    CustomArState state;
    state.devices = {0, 1};
    std::vector<void *> inputs(2);
    cudaStream_t consumer[2] = {};
    CustomArRankData *observed[2] = {};
    for (int device = 0; device < 2; ++device) {
        Require(cudaSetDevice(device), "set device");
        Require(cudaMalloc(&inputs[device], 16), "input allocation");
        Require(cudaHostAlloc(&staging[device], sizeof(CustomArRankData), 0), "staging allocation");
        Require(cudaHostAlloc(&observed[device], sizeof(CustomArRankData), 0), "result allocation");
        Require(cudaStreamCreateWithFlags(&consumer[device], cudaStreamNonBlocking), "consumer stream");
    }
    if (reuse) {
        // A peer worker may return a temporary to the pool before its stream
        // finishes writing it. Registration must drain that device before
        // borrowing this storage on the registering thread's own stream.
        for (int device = 0; device < 2; ++device) {
            Require(cudaSetDevice(device), "old writer device");
            void *temporary = FastllmCudaMalloc(sizeof(CustomArRankData));
            if (temporary == nullptr) return 1;
            Require(cudaLaunchHostFunc(consumer[device], DelayCopy, nullptr), "delay old writer");
            Require(cudaMemsetAsync(temporary, 0x5a, sizeof(CustomArRankData),
                                    consumer[device]), "old temporary write");
            FastllmCudaFree(temporary);
        }
    }
    delayMetadata = !reuse;
    std::vector<CustomArRankData *> tables;
    const bool registered = BuildCustomArRegistration(state, inputs, tables);
    delayMetadata = false;
    if (!registered) return 1;
    bool passed = true;
    if (reuse) {
        for (int device = 0; device < 2; ++device) {
            Require(cudaSetDevice(device), "old writer wait device");
            Require(cudaStreamSynchronize(consumer[device]), "old writer completion");
        }
    }
    // Read the last registered GPU first, before any cleanup can accidentally
    // synchronize the registration stream on that device.
    for (int device = 1; device >= 0; --device) {
        Require(cudaSetDevice(device), "consumer device");
        Require(cudaMemcpyAsync(observed[device], tables[device], sizeof(CustomArRankData),
                                cudaMemcpyDeviceToHost, consumer[device]), "consume metadata");
        Require(cudaStreamSynchronize(consumer[device]), "consumer completion");
        for (int peer = 0; peer < 2; ++peer) {
            if (observed[device]->ptrs[peer] != inputs[peer]) {
                std::fprintf(stderr, "GPU %d peer %d: observed %p, expected %p\n",
                             device, peer, observed[device]->ptrs[peer], inputs[peer]);
                passed = false;
            }
        }
    }
    for (int device = 0; device < 2; ++device) {
        Require(cudaSetDevice(device), "cleanup device");
        Require(cudaStreamSynchronize(cudaStreamPerThread), "pending DMA completion");
        FastllmCudaForceFree(tables[device]);
        Require(cudaFree(inputs[device]), "input cleanup");
        Require(cudaFreeHost(staging[device]), "staging cleanup");
        Require(cudaFreeHost(observed[device]), "result cleanup");
        Require(cudaStreamDestroy(consumer[device]), "stream cleanup");
    }
    std::printf("%s: custom all-reduce registration %s\n",
                passed ? "PASS" : "FAIL", reuse ? "waits for previous pool writers" : "publishes completed pointer tables");
    return passed ? 0 : 1;
}
