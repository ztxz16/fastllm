#include "parallel/cuda_runtime.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <vector>

static thread_local int failures = 0;
static thread_local int attempts = 0;
static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static void Check(cudaError_t status) { Require(status == cudaSuccess, cudaGetErrorString(status)); }

// Test-only linker interposition. Populate CUDA's real last-error slot as well
// as returning OOM, without consuming the machine's memory or adding a knob
// to production code.
extern "C" cudaError_t __real_cudaGraphInstantiate(cudaGraphExec_t *, cudaGraph_t,
    cudaGraphNode_t *, char *, size_t);
extern "C" cudaError_t __wrap_cudaGraphInstantiate(cudaGraphExec_t *exec, cudaGraph_t graph,
        cudaGraphNode_t *node, char *log, size_t size) {
    ++attempts;
    if (failures > 0) {
        --failures;
        void *unused = nullptr;
        size_t freeBytes, totalBytes;
        if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess ||
            cudaMalloc(&unused, totalBytes + (size_t(1) << 20)) != cudaErrorMemoryAllocation ||
            cudaPeekAtLastError() != cudaErrorMemoryAllocation) std::abort();
        *exec = nullptr;
        return cudaErrorMemoryAllocation;
    }
    return __real_cudaGraphInstantiate(exec, graph, node, log, size);
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
        puts("SKIP: requires two CUDA devices"); return 77;
    }
    Check(cudaSetDevice(0));
    void *output = FastllmCudaMalloc(4096);
    void *warm = FastllmCudaMalloc(4096);
    FastllmCudaFree(warm);
    void *idle = FastllmCudaMalloc(size_t(64) << 20);
    FastllmCudaFree(idle);
    Require(FastllmCudaGraphPrepareCaptureDevice(), "capture preparation");
    Require(FastllmCudaGraphMemoryPoolBegin(), "begin graph pool");
    Require(FastllmCudaGraphBeginCapture(), "begin capture");
    void *scratch = FastllmCudaMalloc(4096);
    Check(cudaMemsetAsync(scratch, 0x2a, 4096, cudaStreamPerThread));
    Check(cudaMemcpyAsync(output, scratch, 4096, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    FastllmCudaFree(scratch);
    void *graph = nullptr, *exec = nullptr;
    std::vector<void *> pins;
    Require(FastllmCudaGraphEndCapture(&graph), "end capture");
    Require(FastllmCudaGraphMemoryPoolEnd(pins) && !pins.empty(), "pin scratch");
    size_t before, after, total;
    Check(cudaMemGetInfo(&before, &total));
    failures = 1; attempts = 0;
    Require(FastllmCudaGraphInstantiate(graph, &exec) && attempts == 2, "OOM retry");
    Check(cudaPeekAtLastError());
    Check(cudaMemGetInfo(&after, &total));
    Require(after > before + (size_t(32) << 20), "idle pool was not reclaimed");
    Require(FastllmCudaGraphLaunch(exec), "launch after reclaim");
    Check(cudaStreamSynchronize(cudaStreamPerThread));
    std::vector<unsigned char> host(4096);
    Check(cudaMemcpy(host.data(), output, host.size(), cudaMemcpyDeviceToHost));
    Require(std::all_of(host.begin(), host.end(), [](unsigned char x) { return x == 0x2a; }),
        "reclaim released live or graph-pinned storage");
    FastllmCudaGraphExecDestroy(exec); FastllmCudaGraphDestroy(graph);
    FastllmCudaGraphMemoryPoolRelease(pins); FastllmCudaFree(output);

    using namespace fastllm::parallel;
    const int world = std::min(count, 8);
    std::vector<int> devices(world); std::iota(devices.begin(), devices.end(), 0);
    CudaRuntime runtime(Topology(devices, world, world));
    std::vector<void *> values(world), graphs(world), execs(world);
    runtime.Run([&](int rank, int) { Check(cudaMalloc(&values[rank], 4)); });
    runtime.Run([&](int rank, int) {
        // NCCL connection establishment allocates memory and is not capturable.
        Check(cudaMemsetAsync(values[rank], 1, 4, cudaStreamPerThread));
        runtime.AllReduce(runtime.Layout().world, rank, values[rank], 1, ScalarType::Int32);
    });
    for (int repeat = 0; repeat < 3; ++repeat) {
        runtime.Run([&](int rank, int) {
            Require(FastllmCudaGraphBeginCapture(), "parallel begin capture");
            Check(cudaMemsetAsync(values[rank], 1, 4, cudaStreamPerThread));
            runtime.AllReduce(runtime.Layout().world, rank, values[rank], 1, ScalarType::Int32);
            Require(FastllmCudaGraphEndCapture(&graphs[rank]), "parallel end capture");
        });
        runtime.Run([&](int rank, int) {
            // One rank cannot instantiate; all peers must still leave Run and
            // discard their graphs before using the same NCCL communicator.
            failures = rank == 0 ? 2 : 0; attempts = 0;
            bool ok = FastllmCudaGraphInstantiate(graphs[rank], &execs[rank]);
            Require(ok == (rank != 0), "unexpected graph fallback");
            Require(attempts == (rank == 0 ? 2 : 1), "unbounded graph retry");
            Check(cudaPeekAtLastError());
        });
        runtime.Run([&](int rank, int) {
            FastllmCudaGraphExecDestroy(execs[rank]); FastllmCudaGraphDestroy(graphs[rank]);
            execs[rank] = graphs[rank] = nullptr;
        });
        runtime.Run([&](int rank, int) {
            Check(cudaMemsetAsync(values[rank], 1, 4, cudaStreamPerThread));
            runtime.AllReduce(runtime.Layout().world, rank, values[rank], 1, ScalarType::Int32);
        });
        runtime.Run([&](int rank, int) {
            int actual = 0;
            Check(cudaMemcpyAsync(&actual, values[rank], 4, cudaMemcpyDeviceToHost, cudaStreamPerThread));
            Check(cudaStreamSynchronize(cudaStreamPerThread));
            Require(actual == 0x01010101 * world, "eager collective after graph OOM");
        });
    }
    runtime.Run([&](int rank, int) {
        Require(FastllmCudaGraphBeginCapture(), "fatal-path capture");
        runtime.AllReduce(runtime.Layout().world, rank, values[rank], 1, ScalarType::Int32);
        Require(FastllmCudaGraphEndCapture(&graphs[rank]), "fatal-path end capture");
        Require(FastllmCudaGraphInstantiate(graphs[rank], &execs[rank]), "fatal-path instantiate");
    });
    const auto start = std::chrono::steady_clock::now();
    bool rejected = false;
    try {
        runtime.Run([](int rank, int) {
            if (rank == 0) throw std::runtime_error("injected parallel allocation failure");
        });
    } catch (const std::runtime_error &e) {
        rejected = std::string(e.what()) == "injected parallel allocation failure";
    }
    Require(rejected, "parallel failure did not escape while graphs were alive");
    Require(std::chrono::steady_clock::now() - start < std::chrono::seconds(10), "blocking graph communicator abort");
    rejected = false;
    try { runtime.Run([](int, int) { throw std::logic_error("must not run"); }); }
    catch (const std::runtime_error &) { rejected = true; }
    Require(rejected, "failed parallel runtime accepted another step");
    // Error propagation lets the owner release graph user-object references;
    // runtime destruction subsequently joins every communicator's abort.
    for (int rank = 0; rank < world; ++rank) {
        Check(cudaSetDevice(devices[rank]));
        FastllmCudaGraphExecDestroy(execs[rank]); FastllmCudaGraphDestroy(graphs[rank]);
        Check(cudaFree(values[rank]));
    }
    puts("PASS: graph OOM reclaim, eager fallback and failure propagation with live NCCL graphs");
}
