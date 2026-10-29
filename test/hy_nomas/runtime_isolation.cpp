// No checkpoints required. Also writes deterministic outputs for before/after
// library comparisons; --math-only allows comparison with the old allocator.
#include "models/hy_nomas/runtime.h"
#include <cuda_runtime.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace fastllm::nomas;

static void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
static void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 3) {
        std::cerr << "runtime_isolation OUTPUT.f32 [--math-only]\n";
        return 2;
    }
    try {
        const bool mathOnly = argc == 3 && std::string(argv[2]) == "--math-only";
        require(argc == 2 || mathOnly, "unknown option");
        check(cudaSetDevice(0));
        cudaMemPool_t defaultPool, currentPool;
        check(cudaDeviceGetDefaultMemPool(&defaultPool, 0));
        check(cudaDeviceGetMemPool(&currentPool, 0));
        uint64_t threshold = 0;
        check(cudaMemPoolGetAttribute(defaultPool, cudaMemPoolAttrReleaseThreshold, &threshold));
        auto unchanged = [&] {
            if (mathOnly) return;
            cudaMemPool_t current;
            uint64_t after = 0;
            check(cudaDeviceGetMemPool(&current, 0));
            check(cudaMemPoolGetAttribute(defaultPool, cudaMemPoolAttrReleaseThreshold, &after));
            require(current == currentPool, "HY-Nomas changed the device's current pool");
            require(after == threshold, "HY-Nomas changed the default pool's release threshold");
        };
        float *external = nullptr;
        const float sentinel = 123.5f;
        check(cudaMallocAsync(&external, sizeof(float), cudaStreamPerThread));
        check(cudaMemcpyAsync(external, &sentinel, sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread));
        check(cudaStreamSynchronize(cudaStreamPerThread));
        std::ofstream output(argv[1], std::ios::binary);
        require(bool(output), "cannot open output");
        float values[64], weights[256];
        for (int i = 0; i < 64; ++i) values[i] = float(i % 17 - 8) / 16;
        for (int i = 0; i < 256; ++i) weights[i] = float(i % 13 - 6) / 32;
        for (int cycle = 0; cycle < 3; ++cycle) {
            Tensor survivor;
            {
                Runtime runtime(0);
                require(runtime.isFloat32(), "run with FASTLLM_NOMAS_PRECISION=fp32");
                auto x = runtime.upload(values, 4, 16);
                auto weight = runtime.upload(weights, 16, 16);
                auto linear = runtime.linear(x, weight);
                auto norm = runtime.norm(linear, false);
                auto result = runtime.attention(norm, norm, x, 2);
                for (const auto &tensor : {linear, norm, result, runtime.activation(result)}) {
                    auto host = runtime.download(tensor);
                    for (float v : host) require(std::isfinite(v), "non-finite output");
                    output.write(reinterpret_cast<const char *>(host.data()), host.size() * sizeof(float));
                }
                survivor = x;
                runtime.synchronize();
                unchanged();
            }
            // The caller may transfer a completed world to another worker.
            std::exception_ptr error;
            std::thread worker([&] {
                try {
                    check(cudaSetDevice(0));
                    float actual[64];
                    check(cudaMemcpyAsync(actual, survivor.data, sizeof(actual), cudaMemcpyDeviceToHost, cudaStreamPerThread));
                    check(cudaStreamSynchronize(cudaStreamPerThread));
                    for (int i = 0; i < 64; ++i) require(actual[i] == values[i], "tensor did not outlive Runtime");
                    survivor = {};
                    check(cudaStreamSynchronize(cudaStreamPerThread));
                } catch (...) { error = std::current_exception(); }
            });
            worker.join();
            if (error) std::rethrow_exception(error);
            unchanged();
        }
        float actual = 0;
        check(cudaMemcpyAsync(&actual, external, sizeof(float), cudaMemcpyDeviceToHost, cudaStreamPerThread));
        check(cudaFreeAsync(external, cudaStreamPerThread));
        check(cudaStreamSynchronize(cudaStreamPerThread));
        require(actual == sentinel, "unrelated CUDA allocation was corrupted");
        require(bool(output), "cannot write output");
        unchanged();
        std::cout << "PASS: math, tensor lifetime, worker transfer"
                  << (mathOnly ? "\n" : ", default pool isolation\n");
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
