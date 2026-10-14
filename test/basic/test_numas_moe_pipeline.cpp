#include "fastllm.h"
#include "executor.h"
#include "gguf.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/numas/numasdevice.h"
#include "utils/hybrid_parallel.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
using namespace fastllm;
namespace fastllm { void RegisterNumas(Data *, std::string); }
static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
struct Fixture {
    static constexpr int hidden = 256, inter = 256, experts = 4, topk = 2;
    std::vector<std::unique_ptr<Data>> owned;
    std::vector<Data *> weights = std::vector<Data *>(2 * (experts + 1), nullptr);
    std::vector<Data *> biases = weights;
    explicit Fixture(int seed) {
        for (int e = 1; e <= experts; ++e) for (int part = 0; part < 2; ++part) {
            auto w = std::make_unique<Data>(DATA_GGUF_FORMAT, GGML_TYPE_Q8_0,
                std::vector<int>{part ? hidden : 2 * inter, part ? inter : hidden});
            w->Allocate();
            auto *blocks = reinterpret_cast<block_q8_0 *>(w->cpuData);
            for (size_t i = 0; i < w->GetBytes() / sizeof(block_q8_0); ++i) {
                blocks[i].d = float_to_half(part ? .00390625f : .015625f);
                for (int j = 0; j < 32; ++j)
                    blocks[i].qs[j] = int((i * 7 + j * 3 + e * 5 + seed) % 15) - 7;
            }
            RegisterNumas(w.get(), part ? "linearColumn" : "linearSwiglu");
            weights[2 * e + part] = w.get(); owned.push_back(std::move(w));
        }
    }
    std::vector<float> Run(int device, int rows, int step, std::mutex *cpuMutex) {
        FastllmCudaSetDevice(device);
        Executor executor;
        executor.SetFirstDevice("numa");
        void *previous = GetExecutor();
        struct Restore { void *previous; ~Restore() { SetCurrentThreadExecutor(previous); } } restore{previous};
        SetCurrentThreadExecutor(&executor);
        Data input(FLOAT32, {rows, hidden}), ids(INT32, {rows, topk}), scores(FLOAT32, {rows, topk});
        input.Allocate(); ids.Allocate(); scores.Allocate();
        for (int i = 0; i < rows * hidden; ++i)
            reinterpret_cast<float *>(input.cpuData)[i] = std::sin((i + 13 * step + 17 * device) * .137f);
        for (int row = 0; row < rows; ++row) {
            reinterpret_cast<int32_t *>(ids.cpuData)[2 * row] = 0;
            // Frequent experts execute on CUDA; the final rare routes use CPU.
            reinterpret_cast<int32_t *>(ids.cpuData)[2 * row + 1] = row + 12 < rows ? 1 : 2 + row % 2;
            reinterpret_cast<float *>(scores.cpuData)[2 * row] = .375f;
            reinterpret_cast<float *>(scores.cpuData)[2 * row + 1] = .625f;
        }
        input.ToDevice(CUDA, std::vector<int>{device});
        ids.ToDevice(CUDA, std::vector<int>{device});
        scores.ToDevice(CUDA, std::vector<int>{device});
        std::vector<int> devices{device};
        NumasMoeCudaAssistScope assist(&devices);
        HybridMoePipelineGuard pipeline(input, "numa", cpuMutex, device);
        Data results[3], w1, w2, w3, curInput, curOutput;
        const int count = rows >= 65 ? 3 : 1;
        // Keep successive layer outputs on CUDA: staging reuse must wait for
        // the preceding reduction without an intervening CPU readback.
        for (int layer = 0; layer < count; ++layer)
            MergeMOE(layer ? results[layer - 1] : input, ids, scores, weights, biases,
                     w1, w2, w3, curInput, curOutput, 0.f, results[layer], layer);
        Data &output = results[count - 1];
        if (cpuMutex && rows >= 65)
            Require(!pipeline.HoldsCpuPool(), "Pipelined GPU prefill retained the CPU lock");
        if (cpuMutex && rows == 1)
            Require(pipeline.HoldsCpuPool(), "CPU-only fallback released its guard unexpectedly");
        output.ToDevice(CPU);
        std::vector<float> result(rows * hidden);
        std::memcpy(result.data(), output.cpuData, result.size() * sizeof(float));
        return result;
    }
};
int main() {
    if (FastllmCudaGetDeviceCount() < 2) return 77;
    try {
        SetThreads(4);
        SetDeviceMap({{"cuda:0", 1}, {"cuda:1", 1}});
        Fixture a(0), b(19);
        Fixture *fixtures[] = {&a, &b};
        const std::vector<int> rows{65, 257, 129, 1, 32, 33, 2, 65, 257, 1, 129};
        std::vector<std::vector<float>> expected[2];
        for (int rank = 0; rank < 2; ++rank)
            for (int step = 0; step < int(rows.size()); ++step)
                expected[rank].push_back(fixtures[rank]->Run(rank, rows[step], step, nullptr));
        std::mutex cpuMutex;
        std::exception_ptr errors[2];
        std::thread workers[2];
        for (int rank = 0; rank < 2; ++rank) workers[rank] = std::thread([&, rank] {
            try {
                for (int repeat = 0; repeat < 3; ++repeat)
                    for (int step = 0; step < int(rows.size()); ++step) {
                        auto got = fixtures[rank]->Run(rank, rows[step], step, &cpuMutex);
                        const auto &want = expected[rank][step];
                        for (size_t i = 0; i < got.size(); ++i)
                            Require(std::isfinite(got[i]) && std::abs(got[i] - want[i]) <=
                                2e-5f * (1 + std::abs(want[i])), "Concurrent NUMA pipeline changed its serial result");
                    }
            } catch (...) { errors[rank] = std::current_exception(); }
        });
        for (auto &worker : workers) worker.join();
        for (const auto &error : errors) if (error) std::rethrow_exception(error);
        ClearNumasMoeRuntimeCache();
        std::puts("PASS: NUMA pipeline private staging, mixed CPU/GPU experts and CPU fallback");
        return 0;
    } catch (const std::exception &error) {
        ClearNumasMoeRuntimeCache();
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
