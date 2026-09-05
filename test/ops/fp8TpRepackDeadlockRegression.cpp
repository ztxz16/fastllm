#include "fastllm.h"
#include "executor.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-fp8.h"
#include "devices/cuda/fastllm-awq-sm70.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"

#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace fastllm;

// One TP rank can reach a later Linear before its peer submits the preceding
// AllReduce. Repacking / cold GEMM initialization must not stop that peer from
// submitting the collective by holding a mutex shared across devices.
int main(int argc, char **argv) {
    const std::string mode = argc > 1 ? argv[1] : "repack";
    const std::string rank = argc > 2 ? argv[2] : "1";
    if (argc > 3 ||
        (mode != "repack" && mode != "runtime" && mode != "warmup") ||
        (rank != "0" && rank != "1")) {
        std::cerr << "Usage: " << argv[0]
                  << " [repack|runtime|warmup] [0|1]\n";
        return 2;
    }
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount < 2) {
        std::cout << "SKIP: this regression requires two visible SM70 GPUs\n";
        return 77;
    }
    for (int device : {0, 1}) {
        cudaDeviceProp properties;
        if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
            std::cerr << "Could not query CUDA device " << device << '\n';
            return 2;
        }
        if (properties.major != 7 || properties.minor != 0) {
            std::cout << "SKIP: this regression requires two visible SM70 GPUs\n";
            return 77;
        }
        FastllmCudaSetDevice(device);
        if (!awq_sm70::Fp8Supported()) {
            std::cout << "SKIP: SM70 FP8 kernels are unavailable in this build\n";
            return 77;
        }
    }
    const bool warmupOnly = mode == "warmup";
    const bool runtimeOnly = mode == "runtime" || warmupOnly;
    const int fastRank = rank == "0" ? 0 : 1;
    std::atomic<int> ready{0};
    std::atomic<bool> collectiveQueued{false};
    std::cerr << "Initializing NCCL\n";
    FastllmInitNccl({0, 1});
    std::cerr << "NCCL ready\n";
    FastllmCudaSetNcclForceSync(warmupOnly);
    auto worker = [&](int device) {
        try {
            FastllmCudaSetDevice(device);
            std::cerr << "rank " << device << " setup\n";
            constexpr int rows = 8, K = 256, N = 128;
            Executor exec;
            exec.SetFirstDevice("cuda:" + std::to_string(device));
            Data weight(DataType::FP8_E4M3, {N, K});
            weight.blockM = weight.blockK = 128;
            weight.scales.assign((K / 128) * (N / 128), 1.0f);
            weight.Allocate();
            // Different exact values on the two devices also catch accidental
            // cross-device data reuse: E4M3 encodes 1 as 0x38 and 2 as 0x40.
            std::memset(weight.cpuData, device == 0 ? 0x38 : 0x40, N * K);
            weight.ToDevice(DataDevice::CUDA, std::vector<int>{device});
            Data input(DataType::FLOAT16, {rows, K}, std::vector<float>(rows * K, 1.0f));
            input.ToDevice(DataDevice::CUDA, std::vector<int>{device});
            Data output;
            auto linear = [&]() {
                exec.Run("Linear", {{"input", &input}, {"weight", &weight},
                    {"bias", GetEmptyData()}, {"output", &output}}, {}, {});
            };
            linear();  // Initialize scales/buffers; serving also marks the shape.
            std::cerr << "rank " << device << " native Linear ready\n";
            if (weight.IsRepacked || weight.extraCudaData.size() != (warmupOnly ? 2 : 3)) {
                throw std::runtime_error("first eligible call should use native layout");
            }
            if (output.dataType != DataType::FLOAT16) {
                throw std::runtime_error("unexpected Linear output dtype");
            }
            std::vector<half> reference(output.GetBytes() / sizeof(half));
            FastllmCudaCopyFromDeviceToHost(reference.data(), output.cudaData, output.GetBytes());
            for (half value : reference) {
                if (__half2float(value) != float(K * (device + 1))) {
                    throw std::runtime_error("native output differs from exact reference");
                }
            }
            if (warmupOnly) {
                if (FastllmCudaWarmupFp8E4M3Sm70(weight, 4) ||
                    FastllmCudaWarmupFp8E4M3Sm70(weight, 32) || weight.IsRepacked) {
                    throw std::runtime_error("warmup changed an ineligible layout");
                }
                auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                while (!FastllmCudaWarmupFp8E4M3Sm70(weight, rows)) {
                    if (std::chrono::steady_clock::now() > deadline) {
                        throw std::runtime_error("FP8 warmup preparation failed");
                    }
                    std::this_thread::yield();
                }
                if (!weight.IsRepacked) {
                    throw std::runtime_error("warmup did not prepare the weight");
                }
                void *packedScales = weight.extraCudaData[0];
                if (!FastllmCudaWarmupFp8E4M3Sm70(weight, rows) ||
                    weight.extraCudaData[0] != packedScales) {
                    throw std::runtime_error("repeated warmup changed the packed weight");
                }
            } else if (runtimeOnly) {
                half *packedScales = nullptr;
                auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                while (!awq_sm70::PrepareFp8InPlace((uint8_t *)weight.cudaData,
                    (float *)weight.extraCudaData[0], &packedScales,
                    K, N, 128, 128, cudaStreamPerThread)) {
                    if (std::chrono::steady_clock::now() > deadline) {
                        throw std::runtime_error("FP8 preparation failed");
                    }
                    std::this_thread::yield();
                }
                cudaFree(weight.extraCudaData[0]);
                weight.extraCudaData[0] = packedScales;
                weight.IsRepacked = true;
            }
            Data sum(DataType::FLOAT32, {1}, std::vector<float>{float(device + 1)});
            sum.ToDevice(DataDevice::CUDA, std::vector<int>{device});
            FastllmCudaSyncDevice(device);
            ready.fetch_add(1);
            while (ready.load() < 2) std::this_thread::yield();
            FastllmCudaSetNcclForceSync(false);
            // Establish NCCL's lazy transports on both ranks before imposing
            // the asymmetric order; otherwise the first enqueue may block in
            // transport setup, before reaching the code under test.
            FastllmNcclAllReduce(sum.cudaData, sum.cudaData, 1, DataType::FLOAT32, device);
            FastllmCudaSyncDevice(device);
            float initial = float(device + 1);
            FastllmCudaCopyFromHostToDevice(sum.cudaData, &initial, sizeof(initial));
            FastllmCudaSyncDevice(device);
            ready.fetch_add(1);
            while (ready.load() < 4) std::this_thread::yield();
            if (device == fastRank) {
                FastllmNcclAllReduce(sum.cudaData, sum.cudaData, 1, DataType::FLOAT32, device);
                collectiveQueued.store(true);
                std::cerr << "rank " << device << " queued collective; entering "
                          << (runtimeOnly ? "cold GEMM" : "repack") << '\n';
                linear();
            } else {
                while (!collectiveQueued.load()) std::this_thread::yield();
                // Force an ordering that is otherwise intermittent in serving.
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                std::cerr << "rank " << device << " entering "
                          << (runtimeOnly ? "cold GEMM" : "repack")
                          << " before matching collective\n";
                linear();
                FastllmNcclAllReduce(sum.cudaData, sum.cudaData, 1, DataType::FLOAT32, device);
            }
            FastllmCudaSyncDevice(device);
            std::vector<half> actual(reference.size());
            FastllmCudaCopyFromDeviceToHost(actual.data(), output.cudaData, output.GetBytes());
            if (std::memcmp(actual.data(), reference.data(), output.GetBytes()) != 0) {
                throw std::runtime_error("native/repacked output mismatch");
            }
            float total = 0;
            FastllmCudaCopyFromDeviceToHost(&total, sum.cudaData, sizeof(total));
            if (total != 3.0f) {
                throw std::runtime_error("AllReduce output mismatch");
            }
            // A busy converter may defer one rank's repack to unblock the
            // collective. Once both ranks finish that phase, it must retry.
            ready.fetch_add(1);
            while (ready.load() < 6) std::this_thread::yield();
            if (!weight.IsRepacked) {
                linear();
            }
            FastllmCudaSyncDevice(device);
            if (!weight.IsRepacked) {
                throw std::runtime_error("deferred repack was not retried");
            }
            FastllmCudaCopyFromDeviceToHost(actual.data(), output.cudaData, output.GetBytes());
            if (std::memcmp(actual.data(), reference.data(), output.GetBytes()) != 0) {
                throw std::runtime_error("deferred repack output mismatch");
            }
            std::cerr << "rank " << device << " PASS: repacked, bit-exact, AllReduce=3\n";
        } catch (...) {
            try { throw; }
            catch (const std::exception &e) { std::cerr << "rank " << device << ": " << e.what() << '\n'; }
            catch (const char *e) { std::cerr << "rank " << device << ": " << e << '\n'; }
            catch (...) { std::cerr << "rank " << device << ": unknown failure\n"; }
            // A peer can be inside a blocking CUDA call; fail the process
            // instead of trying to join it after a rank-local error.
            std::exit(1);
        }
    };
    std::thread a(worker, 0), b(worker, 1);
    a.join(); b.join();
    std::cout << "PASS " << mode
              << " fast_rank=" << fastRank << '\n';
}
