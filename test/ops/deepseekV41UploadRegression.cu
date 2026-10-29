// Verifies pinned staging lifetime, stream ordering and ring reuse on every GPU.
// Build with nvcc --default-stream=per-thread and link libfastllm_tools + cudart.
#include "devices/cuda/fastllm-v41-full-decode.h"
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

static void Cuda(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

__global__ void Snapshot(const unsigned char *request, const unsigned char *a,
                         const unsigned char *b, const unsigned char *c,
                         const unsigned char *d, unsigned char *history, int extra) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = sizeof(FastllmV41FullDecodeRequest) + 3 + 127 + 4096 + extra;
    if (index >= size) return;
    const int round = reinterpret_cast<const FastllmV41FullDecodeRequest *>(request)->position;
    int offset = index;
    unsigned char value;
    if (offset < sizeof(FastllmV41FullDecodeRequest)) value = request[offset];
    else if ((offset -= sizeof(FastllmV41FullDecodeRequest)) < 3) value = a[offset];
    else if ((offset -= 3) < 127) value = b[offset];
    else if ((offset -= 127) < 4096) value = c[offset];
    else value = d[offset - 4096];
    history[(size_t)round * size + index] = value;
}

static void Run(int device, int extra, bool contiguous) {
    constexpr int rounds = 80; // Wrap the 16-slot staging ring several times.
    Cuda(cudaSetDevice(device));
    const size_t sizes[] = {sizeof(FastllmV41FullDecodeRequest), 3, 127, 4096, (size_t)extra};
    size_t bytes = 0;
    for (size_t size : sizes) bytes += size;
    unsigned char *buffers[5], *history, *arena = nullptr;
    if (contiguous) {
        Cuda(cudaMalloc(&arena, bytes + 128));
        Cuda(cudaMemset(arena, 0xa5, bytes + 128));
    }
    size_t begin = 64;
    for (int i = 0; i < 5; ++i) {
        if (contiguous) buffers[i] = arena + begin;
        else Cuda(cudaMalloc(&buffers[i], sizes[i]));
        begin += sizes[i];
    }
    Cuda(cudaMalloc(&history, bytes * rounds));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    Snapshot<<<(bytes + 255) / 256, 256, 0, cudaStreamPerThread>>>(
        buffers[0], buffers[1], buffers[2], buffers[3], buffers[4], history, extra);
    Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Cuda(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    std::vector<unsigned char> host(bytes), expected(bytes * rounds), actual(bytes * rounds);
    void *plan = nullptr; // Null plan slots are valid during graph construction.
    FastllmV41FullDecodeRequest request{};
    for (int round = 0; round < rounds; ++round) {
        std::memset(&request, 0, sizeof(request));
        request.position = round;
        for (size_t i = 0; i < bytes; ++i) host[i] = (i * 29 + round * 17 + device) % 251;
        std::memcpy(host.data(), &request, sizeof(request));
        std::memcpy(expected.data() + round * bytes, host.data(), bytes);
        FastllmV41HostCopy copies[4];
        size_t offset = sizeof(request);
        for (int i = 0; i < 4; ++i) {
            copies[i] = {buffers[i + 1], host.data() + offset, sizes[i + 1]};
            offset += sizes[i + 1];
        }
        Check(FastllmCudaV41UploadFullDecodeInputs(buffers[0], &plan, 1, &request, 1,
                                                 copies, 4), "upload rejected");
        // Both sources may be reused immediately after the API returns.
        std::memset(host.data(), 0xee, bytes);
        request.position = -1;
        Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
    }
    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    Cuda(cudaMemcpy(actual.data(), history, actual.size(), cudaMemcpyDeviceToHost));
    Check(actual == expected, "source lifetime or stream order mismatch");

    std::vector<unsigned char> before(sizeof(request)), after(sizeof(request));
    Cuda(cudaMemcpy(before.data(), buffers[0], before.size(), cudaMemcpyDeviceToHost));
    request.position = 1234;
    FastllmV41HostCopy bad{nullptr, host.data(), 1};
    Check(!FastllmCudaV41UploadFullDecodeInputs(buffers[0], &plan, 1, &request, 1, &bad, 1),
          "invalid copy accepted");
    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    Cuda(cudaMemcpy(after.data(), buffers[0], after.size(), cudaMemcpyDeviceToHost));
    Check(before == after, "invalid batch partially submitted");
    Check(!FastllmCudaV41UploadFullDecodeInputs(buffers[0], &plan, 1, &request, 1, nullptr, 1),
          "null copy array accepted");
    Check(FastllmCudaV41UploadFullDecodeBatch(buffers[0], &plan, 1, &request, 1),
          "legacy wrapper rejected");

    // Additional inputs are not limited to a fixed number of Engram layers.
    std::vector<FastllmV41HostCopy> many(40, {buffers[1], host.data(), 1});
    Check(FastllmCudaV41UploadFullDecodeInputs(buffers[0], &plan, 1, &request, 1,
                                             many.data(), many.size()), "large copy list rejected");
    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    Cuda(cudaGraphExecDestroy(exec));
    Cuda(cudaGraphDestroy(graph));
    if (contiguous) {
        std::vector<unsigned char> guards(128);
        Cuda(cudaMemcpy(guards.data(), arena, 64, cudaMemcpyDeviceToHost));
        Cuda(cudaMemcpy(guards.data() + 64, arena + 64 + bytes, 64, cudaMemcpyDeviceToHost));
        for (unsigned char value : guards) Check(value == 0xa5, "metadata arena guard overwritten");
        Cuda(cudaFree(arena));
    } else {
        for (void *buffer : buffers) Cuda(cudaFree(buffer));
    }
    Cuda(cudaFree(history));
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try {
        for (int device = 0; device < devices; ++device)
            for (bool contiguous : {false, true})
                for (int extra : {5, 131073}) Run(device, extra, contiguous); // Async and large synchronous fallback.
        std::cout << "PASS " << devices << " GPUs x 2 layouts x 2 sizes x 80 changing-source graph replays; "
                     "validation, legacy wrapper and 40-copy batch" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
