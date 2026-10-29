#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static void Require(bool ok, const char *message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); std::_Exit(1); }
}
static void Check(cudaError_t state) { Require(state == cudaSuccess, cudaGetErrorString(state)); }

static void RunGroup(const std::vector<int> &devices, bool failGather = false, bool failScatter = false) {
    Require(FastllmInitNccl(devices), "NCCL initialization");
    const int ranks = devices.size(), count = 1003, rounds = 16;
    FastllmCudaSetNcclForceSync(true);
    std::atomic<int> ready{0}, captureTurn{0};
    std::vector<std::atomic<int>> prepared(rounds);
    for (auto &value : prepared) value = 0;
    std::vector<std::thread> workers;
    for (int rank = 0; rank < ranks; ++rank) workers.emplace_back([&, rank] {
        const int device = devices[rank];
        Check(cudaSetDevice(device));
        FastllmCudaClearThreadError();
        int *input = nullptr, *gathered = nullptr, *reduced = nullptr;
        for (auto pointer : {&input, &gathered, &reduced})
            Check(cudaMalloc(reinterpret_cast<void **>(pointer), ranks * count * sizeof(int)));
        std::vector<int> host(ranks * count);
        for (int i = 0; i < ranks * count; ++i) host[i] = (rank + 1) * 10000 + i;
        Check(cudaMemcpy(input, host.data(), host.size() * sizeof(int), cudaMemcpyHostToDevice));
        auto gather = [&](bool inPlace) {
            if (inPlace) Check(cudaMemcpyAsync(gathered + rank * count, input,
                count * sizeof(int), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
            return FastllmNcclAllGather(inPlace ? gathered + rank * count : input,
                gathered, count, fastllm::INT32, device);
        };
        auto scatter = [&](bool inPlace) {
            if (inPlace) Check(cudaMemcpyAsync(reduced, input, ranks * count * sizeof(int),
                cudaMemcpyDeviceToDevice, cudaStreamPerThread));
            return FastllmNcclReduceScatter(inPlace ? reduced : input,
                inPlace ? reduced + rank * count : reduced, count, fastllm::INT32, device);
        };
        auto verify = [&](bool inPlace) {
            Check(cudaStreamSynchronize(cudaStreamPerThread));
            Check(cudaMemcpy(host.data(), gathered, host.size() * sizeof(int), cudaMemcpyDeviceToHost));
            for (int r = 0; r < ranks; ++r) for (int i = 0; i < count; ++i)
                Require(host[r * count + i] == (r + 1) * 10000 + i, "AllGather rank order/value");
            Check(cudaMemcpy(host.data(), inPlace ? reduced + rank * count : reduced,
                count * sizeof(int), cudaMemcpyDeviceToHost));
            for (int i = 0; i < count; ++i)
                Require(host[i] == ranks * (ranks + 1) / 2 * 10000 + ranks * (rank * count + i),
                    "ReduceScatter sum/rank offset");
        };
        Require(gather(false) && scatter(false), "warmup gather/scatter");
        verify(false);
        ++ready;
        while (ready != ranks) std::this_thread::yield();
        FastllmCudaSetNcclForceSync(false);
        if (failGather || failScatter) {
            const auto start = std::chrono::steady_clock::now();
            bool ok = failGather
                ? FastllmNcclAllGather(rank == 0 ? nullptr : input, gathered, count, fastllm::INT32, device)
                : FastllmNcclReduceScatter(input, rank == 0 ? nullptr : reduced, count, fastllm::INT32, device);
            Require(!ok && FastllmCudaGetThreadError(), "invalid rank must release peers with an error");
            Require(std::chrono::steady_clock::now() - start < std::chrono::seconds(2), "error wakeup delayed");
        } else {
            for (int round = 0; round < rounds; ++round) {
                if (rank == round % ranks) std::this_thread::sleep_for(std::chrono::milliseconds(2));
                ++prepared[round];
                Require(gather(round % 2) && scatter(round % 2), "eager gather/scatter");
                Require(prepared[round] == ranks, "returned while a peer had not reached this collective");
                verify(round % 2);
            }
            // Capture one rank at a time: any host rendezvous during capture
            // deadlocks this test. Replay then resumes normal eager epochs.
            while (captureTurn != rank) std::this_thread::yield();
            cudaGraph_t graph = nullptr;
            cudaGraphExec_t executable = nullptr;
            Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeRelaxed));
            Require(gather(false) && scatter(false), "capture gather/scatter");
            Check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            Check(cudaGraphInstantiate(&executable, graph, 0));
            ++captureTurn;
            while (captureTurn != ranks) std::this_thread::yield();
            for (int i = 0; i < 8; ++i) Check(cudaGraphLaunch(executable, cudaStreamPerThread));
            verify(false);
            Require(gather(true) && scatter(true), "eager after graph replay");
            verify(true);
            Check(cudaGraphExecDestroy(executable));
            Check(cudaGraphDestroy(graph));
        }
        for (auto pointer : {input, gathered, reduced}) Check(cudaFree(pointer));
    });
    for (auto &worker : workers) worker.join();
    std::printf("PASS: %d ranks, gather/scatter, failure=%d\n", ranks, failGather + 2 * failScatter);
    std::fflush(stdout);
}

int main(int argc, char **argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) {
        std::puts("SKIP: requires at least two CUDA GPUs"); return 77;
    }
    if (argc == 2) {
        const bool gather = std::strcmp(argv[1], "gather-failure") == 0;
        Require(gather || std::strcmp(argv[1], "scatter-failure") == 0, "unknown test mode");
        RunGroup({0, 1}, gather, !gather);
        // A different communicator must not inherit the aborted rendezvous.
        if (count >= 3) RunGroup({1, 2});
        return 0;
    }
    if (count >= 4) RunGroup({0, 1, 2, 3});
    if (count >= 3) RunGroup({2, 0, 1});
    RunGroup({1, 0});
}
