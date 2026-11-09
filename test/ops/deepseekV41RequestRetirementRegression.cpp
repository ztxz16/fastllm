// A consumer may finish a request while the inference thread captures another.
#include "models/deepseekv41.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <thread>
using namespace fastllm;
static void Need(bool value) { if (!value) throw std::runtime_error("request retirement invariant"); }
static void Cuda(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
class Harness : public DeepSeekV41Model {
public:
    void Track(ResponseContext &ctx, const std::shared_ptr<DeepSeekV41RequestState> &state, bool pointerOnly = false) {
        std::lock_guard<std::mutex> guard(v41StateMutex);
        RegisterState(pointerOnly ? nullptr : &ctx.pastKeyValues, ctx.pastKeyValues.empty() ? nullptr : &ctx.pastKeyValues[0].first, state);
    }
    std::shared_ptr<DeepSeekV41RequestState> Find(ResponseContext &ctx) {
        return GetStateByFirstKey(&ctx.pastKeyValues[0].first);
    }
    void Drain() { ReleaseRetiredStates(); }
};
int main() { try {
    FastllmCudaSetDevice(0);
    void *buffer = nullptr; Cuda(cudaMalloc(&buffer, 64)); Cuda(cudaDeviceSynchronize());
    const auto owner = std::this_thread::get_id();
    Harness model;
    for (int iteration = 0; iteration < 128; ++iteration) {
        ResponseContext ctx;
        ctx.pastKeyValues.emplace_back(Data(FLOAT32), Data(FLOAT32));
        std::atomic<int> freed{0}; cudaError_t cleanupStatus = cudaSuccess; bool ownerReleased = false;
        auto state = std::shared_ptr<DeepSeekV41RequestState>(new DeepSeekV41RequestState,
            [&](DeepSeekV41RequestState *p) {
                ownerReleased = std::this_thread::get_id() == owner;
                // Match graph-workspace retirement's synchronization, and verify
                // that cleanup no longer runs while holding the registry mutex.
                cleanupStatus = cudaDeviceSynchronize();
                Need(!model.Find(ctx));
                ++freed; delete p;
            });
        std::weak_ptr<DeepSeekV41RequestState> weak = state;
        model.Track(ctx, state, iteration % 2 != 0); state.reset();
        Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        Cuda(cudaMemsetAsync(buffer, iteration, 64, cudaStreamPerThread));
        std::thread consumer([&] { model.OnResponseContextRemoved(&ctx); model.OnResponseContextRemoved(&ctx); });
        consumer.join();
        Need(freed == 0 && !weak.expired() && !model.Find(ctx));
        cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
        Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
        Cuda(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        Cuda(cudaGraphLaunch(exec, cudaStreamPerThread)); Cuda(cudaStreamSynchronize(cudaStreamPerThread));
        model.Drain();
        Need(freed == 1 && weak.expired() && ownerReleased && cleanupStatus == cudaSuccess);
        model.Drain(); Need(freed == 1);
        Cuda(cudaGraphExecDestroy(exec)); Cuda(cudaGraphDestroy(graph));
    }
    // The scheduler can retire finished draft graphs while retaining target KV
    // metadata for history snapshots and tokens queued for a slow consumer.
    {
        ResponseContext ctx; ctx.pastKeyValues.emplace_back(Data(FLOAT32), Data(FLOAT32));
        auto state = std::make_shared<DeepSeekV41RequestState>(); state->totalLen = 123;
        int draftsFreed = 0; const auto marker = new int;
        state->dspark = std::shared_ptr<DeepSeekV41DsparkState>(
            reinterpret_cast<DeepSeekV41DsparkState *>(marker), [&](DeepSeekV41DsparkState *p) {
                Need(std::this_thread::get_id() == owner && model.Find(ctx) == state);
                ++draftsFreed; delete reinterpret_cast<int *>(p);
            });
        model.Track(ctx, state);
        model.OnResponseContextFinished(&ctx); model.OnResponseContextFinished(&ctx);
        Need(draftsFreed == 1 && model.Find(ctx) == state && state->totalLen == 123 && !state->dspark);
        model.OnResponseContextRemoved(&ctx); model.Drain();
    }
    // The final idle batch is drained by model shutdown before cache owners die.
    int shutdownReleased = 0;
    {
        Harness idle; ResponseContext ctx;
        auto state = std::shared_ptr<DeepSeekV41RequestState>(new DeepSeekV41RequestState,
            [&](DeepSeekV41RequestState *p) { Need(std::this_thread::get_id() == owner); ++shutdownReleased; delete p; });
        idle.Track(ctx, state); state.reset();
        std::thread consumer([&] { idle.OnResponseContextRemoved(&ctx); }); consumer.join();
        Need(shutdownReleased == 0);
    }
    Need(shutdownReleased == 1); Cuda(cudaFree(buffer));
    puts("PASS 128 capture/consumer races, scheduler retirement and idle shutdown"); return 0;
} catch (const std::exception &e) { fprintf(stderr, "FAIL %s\n", e.what()); return 1; } }
