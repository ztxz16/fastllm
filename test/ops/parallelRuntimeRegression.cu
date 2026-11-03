#include "parallel/cuda_runtime.h"
#include "fastllm.h"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <thread>
#include <vector>
using namespace fastllm::parallel;
static void Require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
static void Cuda(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
__global__ void AddValues(float *data, float value) {
    if (threadIdx.x < 5) data[threadIdx.x] += value;
}
static void CheckPeerLogitGather(const std::vector<int> &devices) {
    if (!FastllmCudaPeerAccessInit(devices)) {
        std::cout << "SKIP direct peer gather: P2P unavailable" << std::endl;
        return;
    }
    // Allocate on the loader thread, gather on a separate serving thread.
    // Different pitches and column offsets must preserve every guard byte.
    for (int rows : {1, 8, 64}) for (size_t width : {size_t(37), size_t(32768 * 4)}) {
        const size_t sourcePitch = width + 32, destPitch = width * devices.size() + 64;
        std::vector<unsigned char*> sources(devices.size());
        unsigned char *dest = nullptr;
        std::vector<unsigned char> expected(rows * destPitch, 0xa5);
        for (size_t r = 0; r < devices.size(); ++r) {
            Cuda(cudaSetDevice(devices[r]));
            Cuda(cudaMalloc(&sources[r], rows * sourcePitch));
            std::vector<unsigned char> input(rows * sourcePitch, 0x5a);
            for (int row = 0; row < rows; ++row) for (size_t col = 0; col < width; ++col)
                expected[row * destPitch + 16 + r * width + col] =
                    input[row * sourcePitch + 8 + col] = (r * 19 + row * 7 + col) % 251;
            Cuda(cudaMemcpy(sources[r], input.data(), input.size(), cudaMemcpyHostToDevice));
        }
        Cuda(cudaSetDevice(devices.front()));
        Cuda(cudaMalloc(&dest, expected.size()));
        Cuda(cudaMemset(dest, 0xa5, expected.size()));
        Cuda(cudaDeviceSynchronize());
        std::exception_ptr error;
        std::thread serving([&] {
            try {
                Cuda(cudaSetDevice(devices.front()));
                std::vector<void*> outputs, inputs;
                std::vector<size_t> destPitches(devices.size(), destPitch),
                    sourcePitches(devices.size(), sourcePitch), widths(devices.size(), width),
                    heights(devices.size(), rows);
                for (size_t r = 0; r < devices.size(); ++r) {
                    outputs.push_back(dest + 16 + r * width); inputs.push_back(sources[r] + 8);
                }
                std::vector<unsigned char> actual(expected.size());
                for (int repeat = 0; repeat < 3; ++repeat) {
                    FastllmCudaMemcpy2DDeviceToDeviceBatch(outputs.data(), destPitches.data(),
                        inputs.data(), sourcePitches.data(), widths.data(), heights.data(), devices.size());
                    Cuda(cudaMemcpy(actual.data(), dest, actual.size(), cudaMemcpyDeviceToHost));
                    Require(actual == expected, "peer logits gather corrupted values or guard bytes");
                }
            } catch (...) { error = std::current_exception(); }
        });
        serving.join();
        for (size_t r = 0; r < devices.size(); ++r) {
            Cuda(cudaSetDevice(devices[r])); Cuda(cudaFree(sources[r]));
        }
        Cuda(cudaSetDevice(devices.front())); Cuda(cudaFree(dest));
        if (error) std::rethrow_exception(error);
    }
    std::cout << "PASS cross-thread pitched peer logits gather" << std::endl;
}
int main() {
    int count = 0;
    Cuda(cudaGetDeviceCount(&count));
    if (count < 8) { std::cout << "SKIP: requires eight CUDA devices\n"; return 77; }
    std::vector<int> devices{6, 2, 7, 0, 5, 1, 4, 3};
    CheckPeerLogitGather(devices);
    for (auto type : {fastllm::FLOAT32, fastllm::FLOAT16, fastllm::BFLOAT16, fastllm::FP8_E4M3})
        for (int axis : {0, 1}) {
            fastllm::Data weight(type, {256, 256}), bias;
            weight.name = "weight-repartition";
            weight.blockK = weight.blockM = 32;
            std::vector<float> scales(64);
            for (int i = 0; i < 64; ++i) scales[i] = (i % 7 + 1) / 16.0f;
            weight.scales = scales;
            weight.Allocate();
            std::vector<unsigned char> expected(weight.GetBytes());
            for (size_t i = 0; i < expected.size(); ++i) expected[i] = (i * 11 + 13) % 251;
            std::memcpy(weight.cpuData, expected.data(), expected.size());
            DivisionScheme ranges;
            for (int r = 0; r < 8; ++r) ranges[devices[r]] = {{r*32, (r+1)*32}};
            Require(SplitMultiCudaWeight(weight, bias, devices, ranges, axis, true, true), "weight split failed");
            if (type == fastllm::FP8_E4M3) for (int r = 0; r < 8; ++r) {
                const auto &local = *weight.multiDeviceDatas.at(devices[r]);
                Require(local.scales.size() == 8, "FP8 shard scale shape");
                for (int i = 0; i < 8; ++i)
                    Require(local.scales[i] == scales[axis == 0 ? r * 8 + i : i * 8 + r],
                            "FP8 scale mapped to the wrong weight block");
            }
            RestoreWeightOnHost(weight);
            Require(!weight.multiDeviceData && weight.dataDevice == fastllm::CPU &&
                weight.name == "weight-repartition" && weight.scales == scales &&
                std::memcmp(weight.cpuData, expected.data(), expected.size()) == 0, "weight reconstruction changed bytes");
        }
    int cases = 0;
    std::map<int, std::thread::id> workers;
    std::map<int, fastllm::Data*> emptyInputs;
    std::mutex workersMutex;
    for (int dp : {1, 2, 4, 8}) {
        CudaRuntime runtime(Topology(devices, dp, 8));
        std::vector<float*> sums(8, nullptr);
        runtime.Run([&](int r, int device) {
            {
                std::lock_guard<std::mutex> lock(workersMutex);
                auto it = workers.emplace(device, std::this_thread::get_id());
                Require(it.first->second == std::this_thread::get_id(), "worker changed across runtimes");
                auto empty = emptyInputs.emplace(device, fastllm::GetEmptyData());
                Require(empty.first->second == fastllm::GetEmptyData(), "worker empty input changed");
                for (const auto &other : emptyInputs)
                    Require(other.first == device || other.second != fastllm::GetEmptyData(),
                            "GPU workers share mutable empty input metadata");
            }
            Cuda(cudaMalloc(&sums[r], 5 * sizeof(float)));
        });
        // Alternate caller/worker producers without a host GPU wait. The
        // final drain must observe every dependency, including event reuse.
        for (int r = 0; r < 8; ++r) {
            Cuda(cudaSetDevice(devices[r]));
            Cuda(cudaMemsetAsync(sums[r], 0, 5 * sizeof(float), cudaStreamPerThread));
        }
        for (int repeat = 0; repeat < 32; ++repeat) {
            runtime.Enqueue([&](int r, int) {
                AddValues<<<1, 32, 0, cudaStreamPerThread>>>(sums[r], r + 1);
                Cuda(cudaGetLastError());
            });
            for (int r = 0; r < 8; ++r) {
                Cuda(cudaSetDevice(devices[r]));
                AddValues<<<1, 32, 0, cudaStreamPerThread>>>(sums[r], 2);
                Cuda(cudaGetLastError());
            }
            runtime.Enqueue([&](int r, int) {
                AddValues<<<1, 32, 0, cudaStreamPerThread>>>(sums[r], 3);
                Cuda(cudaGetLastError());
            });
        }
        runtime.Run([&](int r, int) {
            float values[5];
            Cuda(cudaMemcpyAsync(values, sums[r], sizeof(values), cudaMemcpyDeviceToHost, cudaStreamPerThread));
            Cuda(cudaStreamSynchronize(cudaStreamPerThread));
            for (float value : values) Require(value == 32 * (r + 6), "asynchronous stream handoff lost work");
        });
        // Overlapping groups have independent communicators. Members execute
        // their subsequence of the same total stage order, on one stream.
        for (int repeat = 0; repeat < 3; ++repeat) {
            runtime.Run([&](int r, int) {
                std::vector<float> values(5, r + 1);
                Cuda(cudaMemcpyAsync(sums[r], values.data(), 20, cudaMemcpyHostToDevice, cudaStreamPerThread));
                for (int g = 0; g < int(runtime.Layout().groups.size()); ++g) {
                    auto &group = runtime.Layout().groups[g];
                    if (group.Rank(r) < 0) continue;
                    std::fill(values.begin(), values.end(), r + 1);
                    Cuda(cudaMemcpyAsync(sums[r], values.data(), 20, cudaMemcpyHostToDevice, cudaStreamPerThread));
                    runtime.AllReduce(g, r, sums[r], 5, ScalarType::Float32);
                    Cuda(cudaMemcpyAsync(values.data(), sums[r], 20, cudaMemcpyDeviceToHost, cudaStreamPerThread));
                    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
                    float expected = 0;
                    for (int member : group.ranks) expected += member + 1;
                    for (float value : values) Require(value == expected, "subgroup sum mismatch");
                }
            });
        }
        CudaRowTransfer transfer(runtime, RowPartition(dp, {}));
        for (int n : {0, 1, 2, 7, 8, 9, 32, 129, 513, 2, 0, 9}) for (bool skew : {false, true}) {
            std::vector<int> owners(n);
            for (int i = 0; i < n; ++i) owners[i] = skew ? dp - 1 : (i * 7) % dp;
            RowPartition rows(dp, owners);
            transfer.Reset(rows);
            transfer.Reset(rows); // same permutation keeps existing storage
            transfer.Reserve(37);
            std::vector<unsigned char*> global(8), local(8), collected(8);
            std::vector<unsigned char> input(n * 37);
            for (size_t i = 0; i < input.size(); ++i) input[i] = (i * 31 + 17) % 251;
            runtime.Run([&](int r, int) {
                const size_t bytes = std::max<size_t>(1, input.size());
                Cuda(cudaMalloc(&global[r], bytes)); Cuda(cudaMalloc(&local[r], bytes));
                Cuda(cudaMalloc(&collected[r], bytes));
                if (!input.empty()) Cuda(cudaMemcpyAsync(global[r], input.data(), input.size(), cudaMemcpyHostToDevice, cudaStreamPerThread));
            });
            runtime.Enqueue([&](int r, int) {
                transfer.Select(r, global[r], local[r], 37);
                transfer.Collect(r, local[r], collected[r], 37);
            });
            runtime.Run([&](int r, int) {
                std::vector<unsigned char> result(input.size());
                if (!result.empty()) {
                    Cuda(cudaMemcpyAsync(result.data(), collected[r], result.size(), cudaMemcpyDeviceToHost, cudaStreamPerThread));
                    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
                }
                Require(result == input, "single-field row transfer changed bytes");
            });
            auto packed = [&](int r, int) {
                const size_t localRows = rows.localToGlobal[runtime.Layout().AttentionOwner(r)].size();
                CudaRowTransfer::Field fields[4];
                const size_t widths[] = {13, 0, 7, 17};
                size_t offset = 0;
                for (int i = 0; i < 4; ++i) {
                    auto *source = local[r] + localRows * offset;
                    auto *dest = collected[r] + n * offset;
                    transfer.Select(r, global[r] + n * offset, source, widths[i]);
                    fields[i] = {source, dest, widths[i]};
                    offset += widths[i];
                }
                transfer.Collect(r, fields, 4);
            };
            runtime.Enqueue(packed);
            if (n == 9 || n == 32) {
                runtime.Run([](int, int) {});
                for (int device : devices) {
                    Cuda(cudaSetDevice(device));
                    Require(FastllmCudaGraphBeginCapture(), "parallel graph begin failed");
                }
                runtime.Enqueue([&](int r, int device) {
                    Cuda(cudaMemsetAsync(sums[r], 0, 5 * sizeof(float), cudaStreamPerThread));
                    AddValues<<<1, 32, 0, cudaStreamPerThread>>>(sums[r], r + 1);
                    runtime.AllReduce(runtime.Layout().attention[runtime.Layout().AttentionOwner(r)],
                        r, sums[r], 5, ScalarType::Float32);
                    packed(r, device);
                });
                std::vector<void*> graphs(8, nullptr), execs(8, nullptr);
                for (int r = 0; r < 8; ++r) {
                    Cuda(cudaSetDevice(devices[r]));
                    if (!FastllmCudaGraphEndCapture(&graphs[r])) {
                        std::cerr << "DP" << dp << " rows=" << n << " rank=" << r << " "
                            << FastllmCudaGraphLastError() << std::endl;
                        throw std::runtime_error("parallel graph capture failed");
                    }
                }
                for (int r = 0; r < 8; ++r) {
                    Cuda(cudaSetDevice(devices[r]));
                    Require(FastllmCudaGraphInstantiate(graphs[r], &execs[r]), "parallel graph instantiate failed");
                }
                // Same row counts, new request ordering: graph operands stay
                // fixed while select/unpack maps are uploaded independently.
                std::reverse(owners.begin(), owners.end());
                transfer.Reset(RowPartition(dp, owners));
                for (int replay = 0; replay < 3; ++replay) runtime.Run([&](int r, int) {
                    Require(FastllmCudaGraphLaunch(execs[r]), "parallel graph replay failed");
                });
                runtime.Run([&](int r, int) {
                    float values[5];
                    Cuda(cudaMemcpyAsync(values, sums[r], sizeof(values), cudaMemcpyDeviceToHost, cudaStreamPerThread));
                    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
                    float expected = 0;
                    for (int member : runtime.Layout().groups[runtime.Layout().attention[
                            runtime.Layout().AttentionOwner(r)]].ranks) expected += member + 1;
                    for (float value : values) Require(value == expected, "captured subgroup reduction mismatch");
                    FastllmCudaGraphExecDestroy(execs[r]); FastllmCudaGraphDestroy(graphs[r]);
                });
            }
            runtime.Run([&](int r, int) {
                std::vector<unsigned char> result(input.size());
                if (!result.empty()) {
                    Cuda(cudaMemcpyAsync(result.data(), collected[r], result.size(), cudaMemcpyDeviceToHost, cudaStreamPerThread));
                    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
                }
                Require(result == input, "DP/TP row transfer corrupted data or duplicated TP rows");
                Cuda(cudaFree(global[r])); Cuda(cudaFree(local[r])); Cuda(cudaFree(collected[r]));
            });
            ++cases;
        }
        runtime.Run([&](int r, int) { Cuda(cudaFree(sums[r])); });
        std::cout << "PASS DP" << dp << " TP" << 8 / dp << std::endl;
    }
    Require(workers.size() == 8, "incorrect worker count");
    bool rejected = false;
    std::vector<void*> buffers(8, nullptr);
    try {
        CudaRuntime runtime(Topology(devices, 4, 8));
        runtime.Run([&](int r, int) { Cuda(cudaMalloc(&buffers[r], 8)); });
        runtime.Enqueue([&](int r, int) {
            // Valid buffers; the count mismatch must be rejected before NCCL.
            runtime.AllReduce(runtime.Layout().world, r, buffers[r], r == 0 ? 2 : 1, ScalarType::Float32);
        });
    } catch (const std::runtime_error&) { rejected = true; }
    for (int r = 0; r < 8; ++r) { Cuda(cudaSetDevice(devices[r])); Cuda(cudaFree(buffers[r])); }
    Require(rejected, "mismatched collective accepted");
    rejected = false;
    try {
        CudaRuntime runtime(Topology(devices, 4, 8));
        runtime.Enqueue([&](int r, int) {
            runtime.AllReduce(runtime.Layout().world, r, nullptr, 1, ScalarType::Float32);
        });
    } catch (const std::invalid_argument&) { rejected = true; }
    Require(rejected, "null collective buffer accepted");
    std::cout << "PASS " << cases << " GPU row cases, fixed workers, overlapping groups, mismatch rejection" << std::endl;
}
