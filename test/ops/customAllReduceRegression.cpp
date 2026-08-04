#include "fastllm.h"
#include "devices/multicuda/fastllm-multicuda.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr size_t kSmallBytes = 16 * 1024;
constexpr size_t kLargeBytes = 1024 * 1024;
constexpr size_t kDecodeB8Bytes = 80 * 1024;
constexpr int kGraphCollectivesPerReplay = 32;
constexpr int kGraphReplayIterations = 16;

bool CheckCuda(cudaError_t state, const char *where) {
    if (state == cudaSuccess) {
        return true;
    }
    std::cerr << where << " failed: " << cudaGetErrorString(state) << "\n";
    return false;
}

template <typename T>
T HostFromFloat(float value);

template <>
half HostFromFloat<half>(float value) {
    return __float2half(value);
}

template <>
__nv_bfloat16 HostFromFloat<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <>
float HostFromFloat<float>(float value) {
    return value;
}

template <typename T>
float HostToFloat(T value);

template <>
float HostToFloat<half>(half value) {
    return __half2float(value);
}

template <>
float HostToFloat<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <>
float HostToFloat<float>(float value) {
    return value;
}

template <typename T>
T HostAddRounded(T first, T second) {
    return HostFromFloat<T>(HostToFloat(first) + HostToFloat(second));
}

template <typename T>
float CheckTolerance();

template <>
float CheckTolerance<half>() {
    return 1e-3f;
}

template <>
float CheckTolerance<__nv_bfloat16>() {
    return 1e-2f;
}

template <>
float CheckTolerance<float>() {
    return 1e-6f;
}

bool RunOnRanks(const std::vector<int> &devices,
                const std::function<bool(int)> &operation) {
    std::atomic<bool> ok{true};
    std::vector<std::thread> workers;
    workers.reserve(devices.size());
    for (int rank = 0; rank < (int)devices.size(); ++rank) {
        workers.emplace_back([&, rank]() {
            if (!CheckCuda(cudaSetDevice(devices[rank]), "cudaSetDevice") ||
                !operation(rank) ||
                !CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
                ok.store(false, std::memory_order_relaxed);
            }
        });
    }
    for (std::thread &worker : workers) {
        worker.join();
    }
    return ok.load(std::memory_order_relaxed);
}

template <typename T>
bool CopyAndCheck(const std::vector<int> &devices,
                  const std::vector<void *> &deviceData,
                  const std::vector<T> &expected,
                  int count, const std::string &label) {
    std::vector<T> actual(count);
    for (int rank = 0; rank < (int)devices.size(); ++rank) {
        if (!CheckCuda(cudaSetDevice(devices[rank]), "cudaSetDevice") ||
            !CheckCuda(cudaMemcpy(actual.data(), deviceData[rank],
                                  sizeof(T) * count,
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpyDeviceToHost")) {
            return false;
        }
        for (int index = 0; index < count; ++index) {
            const float actualValue = HostToFloat(actual[index]);
            const float expectedValue = HostToFloat(expected[index]);
            if (std::fabs(actualValue - expectedValue) >
                    CheckTolerance<T>()) {
                std::cerr << label << " mismatch on rank " << rank
                          << " at " << index << ": expected "
                          << expectedValue << ", got " << actualValue
                          << "\n";
                return false;
            }
        }
    }
    return true;
}

template <typename T>
bool CopyInputs(const std::vector<int> &devices,
                const std::vector<void *> &deviceData,
                const std::vector<std::vector<T> > &hostData,
                int count) {
    for (int rank = 0; rank < (int)devices.size(); ++rank) {
        if (!CheckCuda(cudaSetDevice(devices[rank]), "cudaSetDevice") ||
            !CheckCuda(cudaMemcpy(deviceData[rank], hostData[rank].data(),
                                  sizeof(T) * count,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpyHostToDevice")) {
            return false;
        }
    }
    return true;
}

void FreeBuffers(const std::vector<int> &devices,
                 std::vector<void *> &inputs,
                 std::vector<void *> &outputs) {
    for (int rank = 0; rank < (int)devices.size(); ++rank) {
        cudaSetDevice(devices[rank]);
        if (inputs[rank] != nullptr) {
            cudaFree(inputs[rank]);
            inputs[rank] = nullptr;
        }
        if (outputs[rank] != nullptr) {
            cudaFree(outputs[rank]);
            outputs[rank] = nullptr;
        }
    }
}

template <typename T>
bool RunKernelRegression(const std::vector<int> &devices, int count,
                         int dataType, const char *typeName) {
    const int ranks = (int)devices.size();
    std::vector<void *> inputs(ranks, nullptr);
    std::vector<void *> outputs(ranks, nullptr);
    std::vector<std::vector<T> > hostInputs(ranks);
    std::vector<T> expected(count);
    std::vector<T> residual(count, HostFromFloat<T>(4.0f));
    std::vector<T> expectedWithResidual(count);
    bool ok = true;

    for (int rank = 0; rank < ranks; ++rank) {
        hostInputs[rank].resize(count);
        for (int index = 0; index < count; ++index) {
            hostInputs[rank][index] =
                HostFromFloat<T>((float)(rank + 1) +
                                 (float)(index % 7) * 0.25f);
        }
    }
    for (int index = 0; index < count; ++index) {
        float sum = 0.0f;
        for (int rank = 0; rank < ranks; ++rank) {
            sum += HostToFloat(hostInputs[rank][index]);
        }
        expected[index] = HostFromFloat<T>(sum);
        T reduced = HostAddRounded(hostInputs[0][index],
                                   hostInputs[1][index]);
        expectedWithResidual[index] =
            HostAddRounded(reduced, residual[index]);
    }

    for (int rank = 0; rank < ranks && ok; ++rank) {
        ok = CheckCuda(cudaSetDevice(devices[rank]), "cudaSetDevice") &&
             CheckCuda(cudaMalloc(&inputs[rank], sizeof(T) * count),
                       "cudaMalloc(input)") &&
             CheckCuda(cudaMalloc(&outputs[rank], sizeof(T) * count),
                       "cudaMalloc(output)");
    }
    if (!ok) {
        FreeBuffers(devices, inputs, outputs);
        return false;
    }
    if (!CopyInputs(devices, inputs, hostInputs, count)) {
        FreeBuffers(devices, inputs, outputs);
        return false;
    }

    ok = RunOnRanks(devices, [&](int rank) {
        return FastllmCudaCustomAllReduce(
            inputs[rank], outputs[rank], count, dataType, devices[rank]);
    });
    ok = ok && CopyAndCheck(
        devices, outputs, expected, count,
        std::string(typeName) + " out-of-place all-reduce");

    if (ok) {
        ok = RunOnRanks(devices, [&](int rank) {
            return FastllmCudaCustomAllReduce(
                inputs[rank], inputs[rank], count, dataType, devices[rank]);
        });
        ok = ok && CopyAndCheck(
            devices, inputs, expected, count,
            std::string(typeName) + " in-place all-reduce");
    }

    if (ok && ranks == 2) {
        ok = CopyInputs(devices, inputs, hostInputs, count);
        for (int rank = 0; rank < ranks && ok; ++rank) {
            ok = CheckCuda(cudaSetDevice(devices[rank]), "cudaSetDevice") &&
                 CheckCuda(cudaMemcpy(outputs[rank], residual.data(),
                                      sizeof(T) * count,
                                      cudaMemcpyHostToDevice),
                           "cudaMemcpyHostToDevice(residual)");
        }
        if (ok) {
            ok = RunOnRanks(devices, [&](int rank) {
                return FastllmCudaCustomAllReduceAdd(
                    inputs[rank], outputs[rank], count,
                    dataType, devices[rank]);
            });
            ok = ok && CopyAndCheck(
                devices, outputs, expectedWithResidual, count,
                std::string(typeName) + " fused all-reduce-add");
        }
    }

    FreeBuffers(devices, inputs, outputs);
    return ok;
}

template <typename T>
bool RunGraphFusedRegression(const std::vector<int> &devices,
                             int dataType, const char *typeName,
                             float &averageUs) {
    if (devices.size() != 2) {
        return true;
    }
    const int count = (int)(kDecodeB8Bytes / sizeof(T));
    if (!FastllmCudaCustomAllReduceCanRun(count, dataType, devices[0])) {
        return true;
    }

    std::vector<void *> inputs(2, nullptr);
    std::vector<void *> outputs(2, nullptr);
    std::vector<std::vector<T> > hostInputs(2);
    std::vector<T> residual(count, HostFromFloat<T>(4.0f));
    std::vector<T> expected = residual;
    std::vector<cudaGraphExec_t> graphExecs(2, nullptr);
    std::vector<float> elapsedMs(2, 0.0f);
    bool ok = true;

    for (int rank = 0; rank < 2; ++rank) {
        hostInputs[rank].resize(count);
        for (int index = 0; index < count; ++index) {
            const float value = index % 31 == 0
                ? 0.0f
                : (float)(rank + 1) + (float)(index % 7) * 0.25f;
            hostInputs[rank][index] = HostFromFloat<T>(value);
        }
        ok = ok && CheckCuda(cudaSetDevice(devices[rank]), "cudaSetDevice") &&
             CheckCuda(cudaMalloc(&inputs[rank], sizeof(T) * count),
                       "cudaMalloc(graph input)") &&
             CheckCuda(cudaMalloc(&outputs[rank], sizeof(T) * count),
                       "cudaMalloc(graph output)");
    }
    if (ok) {
        ok = CopyInputs(devices, inputs, hostInputs, count);
    }
    for (int rank = 0; rank < 2 && ok; ++rank) {
        ok = CheckCuda(cudaSetDevice(devices[rank]), "cudaSetDevice") &&
             CheckCuda(cudaMemcpy(outputs[rank], residual.data(),
                                  sizeof(T) * count,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(graph residual)");
    }

    // Warm up once so the full peer pointer tuple is registered before stream
    // capture. Reset the residual afterward because capture itself does not
    // execute the collective.
    if (ok) {
        ok = RunOnRanks(devices, [&](int rank) {
            return FastllmCudaCustomAllReduceAdd(
                inputs[rank], outputs[rank], count, dataType, devices[rank]);
        });
    }
    for (int rank = 0; rank < 2 && ok; ++rank) {
        ok = CheckCuda(cudaSetDevice(devices[rank]), "cudaSetDevice") &&
             CheckCuda(cudaMemcpy(outputs[rank], residual.data(),
                                  sizeof(T) * count,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy(graph residual reset)");
    }

    if (ok) {
        ok = RunOnRanks(devices, [&](int rank) {
            cudaGraph_t graph = nullptr;
            bool localOk = CheckCuda(cudaStreamBeginCapture(
                cudaStreamPerThread, cudaStreamCaptureModeRelaxed),
                "cudaStreamBeginCapture");
            for (int collective = 0;
                 collective < kGraphCollectivesPerReplay && localOk;
                 ++collective) {
                localOk = FastllmCudaCustomAllReduceAdd(
                    inputs[rank], outputs[rank], count,
                    dataType, devices[rank]);
            }
            localOk = localOk && CheckCuda(cudaStreamEndCapture(
                cudaStreamPerThread, &graph), "cudaStreamEndCapture") &&
                graph != nullptr;
            if (localOk) {
                localOk = CheckCuda(cudaGraphInstantiate(
                    &graphExecs[rank], graph, 0),
                    "cudaGraphInstantiate");
            }
            if (graph != nullptr) {
                cudaGraphDestroy(graph);
            }
            return localOk;
        });
    }

    if (ok) {
        ok = RunOnRanks(devices, [&](int rank) {
            cudaEvent_t begin = nullptr;
            cudaEvent_t end = nullptr;
            bool localOk = CheckCuda(cudaEventCreate(&begin),
                                     "cudaEventCreate(begin)") &&
                CheckCuda(cudaEventCreate(&end), "cudaEventCreate(end)") &&
                CheckCuda(cudaEventRecord(begin, cudaStreamPerThread),
                          "cudaEventRecord(begin)");
            for (int iteration = 0;
                 iteration < kGraphReplayIterations && localOk; ++iteration) {
                localOk = CheckCuda(cudaGraphLaunch(
                    graphExecs[rank], cudaStreamPerThread),
                    "cudaGraphLaunch");
            }
            localOk = localOk && CheckCuda(cudaEventRecord(
                end, cudaStreamPerThread), "cudaEventRecord(end)") &&
                CheckCuda(cudaEventSynchronize(end),
                          "cudaEventSynchronize(end)") &&
                CheckCuda(cudaEventElapsedTime(
                    &elapsedMs[rank], begin, end),
                    "cudaEventElapsedTime");
            if (begin != nullptr) {
                cudaEventDestroy(begin);
            }
            if (end != nullptr) {
                cudaEventDestroy(end);
            }
            return localOk;
        });
    }

    for (int index = 0; index < count; ++index) {
        T reduced = HostAddRounded(hostInputs[0][index],
                                   hostInputs[1][index]);
        for (int iteration = 0;
             iteration < kGraphReplayIterations *
                             kGraphCollectivesPerReplay;
             ++iteration) {
            expected[index] = HostAddRounded(reduced, expected[index]);
        }
    }
    if (ok) {
        ok = CopyAndCheck(
            devices, outputs, expected, count,
            std::string(typeName) +
                " B8 CUDA Graph fused all-reduce-add");
    }
    averageUs = *std::max_element(elapsedMs.begin(), elapsedMs.end()) *
                1000.0f /
                (float)(kGraphReplayIterations *
                        kGraphCollectivesPerReplay);

    for (int rank = 0; rank < 2; ++rank) {
        if (graphExecs[rank] != nullptr) {
            cudaSetDevice(devices[rank]);
            cudaGraphExecDestroy(graphExecs[rank]);
        }
    }
    FreeBuffers(devices, inputs, outputs);
    return ok;
}

bool SupportedRankCount(int ranks) {
    return ranks == 2 || ranks == 4 || ranks == 6 || ranks == 8;
}

template <typename T>
bool RunSelectedPaths(const std::vector<int> &devices, int dataType,
                      const char *typeName, int &selectedPaths,
                      int &testedPaths) {
    const size_t testBytes[] = {kSmallBytes, kLargeBytes};
    for (size_t bytes : testBytes) {
        const int count = (int)(bytes / sizeof(T));
        if (!FastllmCudaCustomAllReduceCanRun(
                count, dataType, devices[0])) {
            continue;
        }
        selectedPaths++;
        if (!RunKernelRegression<T>(
                devices, count, dataType, typeName)) {
            return false;
        }
        testedPaths++;
    }
    return true;
}

bool RunAllSelectedPaths(const std::vector<int> &devices,
                         int &selectedPaths, int &testedPaths) {
    return RunSelectedPaths<half>(
               devices, (int)fastllm::DataType::FLOAT16, "FP16",
               selectedPaths, testedPaths) &&
           RunSelectedPaths<__nv_bfloat16>(
               devices, (int)fastllm::DataType::BFLOAT16, "BF16",
               selectedPaths, testedPaths) &&
           RunSelectedPaths<float>(
               devices, (int)fastllm::DataType::FLOAT32, "FP32",
               selectedPaths, testedPaths);
}

bool HasFullMeshP2P(const std::vector<int> &devices, bool &queryOk) {
    queryOk = true;
    for (int device : devices) {
        for (int peer : devices) {
            if (device == peer) {
                continue;
            }
            int canAccess = 0;
            if (!CheckCuda(cudaDeviceCanAccessPeer(
                    &canAccess, device, peer),
                    "cudaDeviceCanAccessPeer")) {
                queryOk = false;
                return false;
            }
            if (!canAccess) {
                return false;
            }
        }
    }
    return true;
}

}  // namespace

int main() {
    const char *expectedValue =
        std::getenv("FASTLLM_TEST_EXPECT_CUSTOM_ALLREDUCE");
    if (expectedValue == nullptr ||
        (std::string(expectedValue) != "0" &&
         std::string(expectedValue) != "1" &&
         std::string(expectedValue) != "auto")) {
        std::cerr << "FASTLLM_TEST_EXPECT_CUSTOM_ALLREDUCE must be 0, 1, "
                     "or auto.\n";
        return 2;
    }
    const bool checkExpected = std::string(expectedValue) != "auto";
    const bool expectedEnabled = std::string(expectedValue) == "1";

    int ranks = 2;
    const char *ranksValue =
        std::getenv("FASTLLM_TEST_CUSTOM_ALLREDUCE_RANKS");
    if (ranksValue != nullptr && ranksValue[0] != '\0') {
        try {
            ranks = std::stoi(ranksValue);
        } catch (...) {
            ranks = 0;
        }
    }
    if (!SupportedRankCount(ranks)) {
        std::cerr << "FASTLLM_TEST_CUSTOM_ALLREDUCE_RANKS must be 2, 4, 6, "
                     "or 8.\n";
        return 2;
    }

    int deviceCount = 0;
    if (!CheckCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount")) {
        return 1;
    }
    if (deviceCount < ranks) {
        std::cout << "custom all-reduce regression: SKIP (requires "
                  << ranks << " GPUs)\n";
        return 0;
    }

    std::vector<int> devices(ranks);
    for (int rank = 0; rank < ranks; ++rank) {
        devices[rank] = rank;
    }
    bool p2pQueryOk = true;
    if (!HasFullMeshP2P(devices, p2pQueryOk)) {
        if (!p2pQueryOk) {
            return 1;
        }
        std::cout << "custom all-reduce regression: SKIP (requires "
                     "full-mesh P2P)\n";
        return 0;
    }

    if (!FastllmInitNccl(devices)) {
        std::cerr << "FastllmInitNccl failed.\n";
        return 1;
    }
    const bool actualEnabled = FastllmCudaCustomAllReduceEnabled();
    if (checkExpected && actualEnabled != expectedEnabled) {
        std::cerr << "custom all-reduce policy mismatch: expected enabled="
                  << expectedEnabled << ", got " << actualEnabled << "\n";
        return 1;
    }

    int selectedPaths = 0;
    int testedPaths = 0;
    if (!RunAllSelectedPaths(devices, selectedPaths, testedPaths)) {
        return 1;
    }
    float fp16B8GraphFusedUs = 0.0f;
    float bf16B8GraphFusedUs = 0.0f;
    if (!RunGraphFusedRegression<half>(
            devices, (int)fastllm::DataType::FLOAT16, "FP16",
            fp16B8GraphFusedUs) ||
        !RunGraphFusedRegression<__nv_bfloat16>(
            devices, (int)fastllm::DataType::BFLOAT16, "BF16",
            bf16B8GraphFusedUs)) {
        return 1;
    }
    if (actualEnabled && selectedPaths == 0) {
        std::cerr << "custom all-reduce is globally enabled but neither "
                     "tested dtype/message path is available.\n";
        return 1;
    }

    // Reset must immediately unpublish the old policy, drain every registered
    // pointer tuple and allow the still-active NCCL group to initialize a fresh
    // custom state. This is the two-GPU coverage for the production group-switch
    // lifecycle, even on machines without a third device.
    const uint64_t stableGeneration = FastllmGetNcclGeneration();
    FastllmCudaCustomAllReduceReset();
    if (FastllmCudaCustomAllReduceEnabled() ||
        FastllmCudaCustomAllReduceCanRun(
            (int)(kSmallBytes / sizeof(half)),
            (int)fastllm::DataType::FLOAT16, devices[0]) ||
        FastllmGetNcclGeneration() != stableGeneration) {
        std::cerr << "custom all-reduce reset left stale published state.\n";
        return 1;
    }
    if (!FastllmInitNccl(devices)) {
        std::cerr << "FastllmInitNccl reinitialization failed.\n";
        return 1;
    }
    const bool reinitializedEnabled = FastllmCudaCustomAllReduceEnabled();
    if (checkExpected && reinitializedEnabled != expectedEnabled) {
        std::cerr << "custom all-reduce reinitialized policy mismatch: "
                  << "expected enabled=" << expectedEnabled << ", got "
                  << reinitializedEnabled << "\n";
        return 1;
    }
    int reinitializedSelectedPaths = 0;
    int reinitializedTestedPaths = 0;
    if (!RunAllSelectedPaths(devices, reinitializedSelectedPaths,
                             reinitializedTestedPaths) ||
        (reinitializedEnabled && reinitializedSelectedPaths == 0)) {
        return 1;
    }

    bool switchedGroupTested = false;
    if (deviceCount > ranks) {
        std::vector<int> switchedDevices(ranks);
        for (int rank = 0; rank < ranks; ++rank) {
            switchedDevices[rank] = rank + 1;
        }
        bool switchedP2PQueryOk = true;
        if (HasFullMeshP2P(switchedDevices, switchedP2PQueryOk)) {
            const uint64_t oldGeneration = FastllmGetNcclGeneration();
            if (!FastllmInitNccl(switchedDevices) ||
                FastllmGetNcclGeneration() <= oldGeneration) {
                std::cerr << "NCCL/custom all-reduce device-group switch failed.\n";
                return 1;
            }
            const bool switchedEnabled =
                FastllmCudaCustomAllReduceEnabled();
            if (checkExpected && switchedEnabled != expectedEnabled) {
                std::cerr << "switched custom all-reduce policy mismatch: "
                          << "expected enabled=" << expectedEnabled
                          << ", got " << switchedEnabled << "\n";
                return 1;
            }
            int switchedSelectedPaths = 0;
            int switchedTestedPaths = 0;
            if (!RunAllSelectedPaths(switchedDevices,
                                     switchedSelectedPaths,
                                     switchedTestedPaths) ||
                (switchedEnabled && switchedSelectedPaths == 0)) {
                return 1;
            }
            switchedGroupTested = true;
        } else if (!switchedP2PQueryOk) {
            return 1;
        }
    }

    std::cout << "custom all-reduce policy and kernel regression: PASS "
              << "(ranks=" << ranks << ", enabled=" << actualEnabled
              << ", selected_paths=" << selectedPaths
              << ", tested_paths=" << testedPaths
              << ", fp16_b8_graph_fused_us=" << fp16B8GraphFusedUs
              << ", bf16_b8_graph_fused_us=" << bf16B8GraphFusedUs
              << ", reset_reinit=1, switched_group="
              << switchedGroupTested << ")\n";
    return 0;
}
