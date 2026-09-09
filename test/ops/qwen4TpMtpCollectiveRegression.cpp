#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include "models/qwen4_tp_sampling.h"

#include <cuda_runtime.h>
#include <array>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

using namespace fastllm;

namespace {
constexpr int kRanks = 4;
constexpr int kSlots = 3;
constexpr int kVocabulary = 248320;

class Barrier {
    std::atomic<int> arrived{0};
    std::atomic<int> generation{0};

public:
    void Wait() {
        const int current = generation.load();
        if (arrived.fetch_add(1) == kRanks - 1) {
            arrived.store(0);
            generation.fetch_add(1);
        } else {
            while (generation.load() == current) std::this_thread::yield();
        }
    }
};

void Require(bool ok, const char *message) {
    if (!ok) {
        std::cerr << "FAIL: " << message << '\n';
        // A peer may be waiting inside NCCL. Fail the process without joining
        // blocked ranks; CTest also bounds collective hangs with a timeout.
        std::exit(1);
    }
}

void Check(cudaError_t status) {
    Require(status == cudaSuccess, cudaGetErrorString(status));
}

std::pair<int, float> Candidate(int rank, int turn) {
    const auto range = qwen4_tp::VocabRange(kVocabulary, kRanks, rank);
    const int width = range.second - range.first;
    int id = (turn * 31 + rank * 97) % width;
    float score = float((rank + turn) % kRanks);
    switch (turn % 4) {
        case 0:
            // Qwen4 TopK ties use bit-reversed lane order, not minimum ID.
            id = rank == 0 ? 1 : 128;
            score = 17.0f;
            break;
        case 1:
            // The four vocabulary shards have different aligned widths.
            id = width - 1 - turn % 7;
            break;
        case 2:
            score = rank == 0 ? std::numeric_limits<float>::quiet_NaN()
                              : std::numeric_limits<float>::infinity();
            break;
        case 3:
            score = -std::numeric_limits<float>::infinity();
            break;
    }
    return {id, score};
}

int ExpectedToken(int turn) {
    std::pair<int, float> best;
    for (int rank = 0; rank < kRanks; ++rank) {
        auto candidate = Candidate(rank, turn);
        candidate.first += qwen4_tp::VocabRange(kVocabulary, kRanks, rank).first;
        if (rank == 0 || qwen4_tp::Top1Before(candidate.first, candidate.second,
                                            best.first, best.second)) {
            best = candidate;
        }
    }
    return best.first;
}
} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < kRanks) {
        std::cout << "SKIP: requires at least four CUDA GPUs\n";
        return 77;
    }
    Require(FastllmInitNccl({0, 1, 2, 3}), "NCCL initialization");
    Barrier barrier;
    std::vector<std::thread> workers;
    for (int rank = 0; rank < kRanks; ++rank) {
        workers.emplace_back([&, rank] {
            Check(cudaSetDevice(rank));
            FastllmCudaSetNcclForceSync(false);
            Data input(FLOAT32, {2}), gathered(FLOAT32, {kRanks, 2});
            Data tokens(INT32, {kSlots}), values(FLOAT32, {kSlots});
            for (Data *data : {&input, &gathered, &tokens, &values}) {
                data->ToDevice(DataDevice::CUDA, std::vector<int>{rank}, false);
                data->Allocate(false);
            }
            auto fill = [&](int turn) {
                const auto candidate = Candidate(rank, turn);
                const float pair[2] = {float(candidate.first), candidate.second};
                Check(cudaMemcpy(input.cudaData, pair, sizeof(pair), cudaMemcpyHostToDevice));
            };
            auto run = [&](int slot) {
                Require(FastllmNcclAllGather(input.cudaData, gathered.cudaData,
                        2, FLOAT32, rank), "NCCL AllGather");
                Require(FastllmCudaQwen4MergeTpGreedy(
                        reinterpret_cast<const float *>(gathered.cudaData),
                        reinterpret_cast<int *>(tokens.cudaData) + slot,
                        reinterpret_cast<float *>(values.cudaData) + slot,
                        kVocabulary, kRanks), "device candidate merge");
            };
            std::array<int, kSlots> expected{};
            auto verify = [&] {
                Check(cudaStreamSynchronize(cudaStreamPerThread));
                std::array<int, kSlots> ids;
                std::array<float, kSlots> floats;
                Check(cudaMemcpy(ids.data(), tokens.cudaData, sizeof(ids), cudaMemcpyDeviceToHost));
                Check(cudaMemcpy(floats.data(), values.cudaData, sizeof(floats), cudaMemcpyDeviceToHost));
                Require(ids == expected, "token mismatch or another draft slot overwritten");
                for (int slot = 0; slot < kSlots; ++slot) {
                    Require(floats[slot] == float(expected[slot]), "embedding token mismatch");
                }
            };
            Check(cudaMemsetAsync(tokens.cudaData, 0, tokens.GetBytes(), cudaStreamPerThread));
            Check(cudaMemsetAsync(values.cudaData, 0, values.GetBytes(), cudaStreamPerThread));

            // Warm NCCL transports before capture and verify the eager path.
            fill(0);
            barrier.Wait();
            run(0);
            expected[0] = ExpectedToken(0);
            verify();
            barrier.Wait();

            std::array<cudaGraph_t, kSlots> graphs{};
            std::array<cudaGraphExec_t, kSlots> executables{};
            for (int slot = 0; slot < kSlots; ++slot) {
                barrier.Wait();
                Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                run(slot);
                Check(cudaStreamEndCapture(cudaStreamPerThread, &graphs[slot]));
                barrier.Wait();
                Check(cudaGraphInstantiate(&executables[slot], graphs[slot], nullptr, nullptr, 0));
            }
            for (int turn = 0; turn < 60; ++turn) {
                fill(turn);
                const int slot = turn % kSlots;
                expected[slot] = ExpectedToken(turn);
                Check(cudaGraphLaunch(executables[slot], cudaStreamPerThread));
                verify();
                barrier.Wait();
            }
            for (int slot = 0; slot < kSlots; ++slot) {
                Check(cudaGraphExecDestroy(executables[slot]));
                Check(cudaGraphDestroy(graphs[slot]));
            }
        });
    }
    for (auto &worker : workers) worker.join();
    std::cout << "PASS: TP4 NCCL AllGather and device merge; 60 changing inputs, "
                 "three graph output slots, shard tails, ties, NaNs and infinities\n";
}
