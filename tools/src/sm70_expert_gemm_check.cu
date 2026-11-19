#include "distributed/sm70_expert_gemm.h"
#include "distributed/batch_invariant.h"
#include "json11.hpp"
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

using json11::Json;
using namespace fastllm::distributed;

namespace {
/** 检查 CUDA 调用并保留具体阶段错误。 */
void Check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

/** 以同一分配/stream 生命周期持有输入、输出、事件及可选对照 handle。 */
struct Resources {
    std::vector<void *> allocations;
    cudaEvent_t start = nullptr, stop = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cublasHandle_t blas = nullptr;

    ~Resources() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (blas) cublasDestroy(blas);
        for (auto p : allocations) cudaFree(p);
    }

    /** 分配并登记缓冲区；空张量分配一个字节，避免把空专家误作 CUDA 分配错误。 */
    void *Allocate(size_t bytes) {
        void *p = nullptr;
        Check(cudaMalloc(&p, std::max<size_t>(1, bytes)));
        allocations.push_back(p);
        return p;
    }

    /** 精确读取连续原始张量并上传；文件长度不符直接拒绝。 */
    void *Load(const std::string &path, size_t bytes) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f || f.tellg() != std::streampos(bytes)) throw std::runtime_error("invalid tensor file: " + path);
        f.seekg(0);
        std::vector<char> data(bytes);
        if (bytes && !f.read(data.data(), bytes)) throw std::runtime_error("tensor read failed: " + path);
        void *p = Allocate(bytes);
        if (bytes) {
            Check(cudaMemcpyAsync(p, data.data(), bytes, cudaMemcpyHostToDevice, cudaStreamPerThread));
            Check(cudaStreamSynchronize(cudaStreamPerThread));
        }
        return p;
    }
};
}

/**
 * 独立专家算子微基准，调用正式 Sm70ExpertGemm；不编译、不替代完整推理验收。
 * 同一 stream、输入、输出类型及同步策略比较固定后端与 cuBLAS，计时外落盘与校验。
 * @param argc/argv 形状、原始张量路径、输出类型、设备、后端和计时参数。
 * @return 0 表示提交及读回成功；数值正确性由调用方按保存结果判定，任何错误返回 1。
 */
int main(int argc, char **argv) {
    try {
        const std::set<std::string> allowed{"--m", "--n", "--k", "--input", "--weight", "--bias",
            "--output", "--fp32", "--device", "--backend", "--graph", "--warmup", "--iterations", "--repeats"};
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 == argc || !allowed.count(argv[i]) || !args.emplace(argv[i], argv[i + 1]).second)
                throw std::invalid_argument("invalid GEMM check arguments");
        }
        auto number = [&](const std::string &name, int fallback) {
            if (!args.count(name)) return fallback;
            const auto &s = args.at(name);
            if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
                throw std::invalid_argument("invalid integer: " + name);
            return std::stoi(s);
        };
        const int m = number("--m", -1), n = number("--n", -1), k = number("--k", -1);
        const int fp32 = number("--fp32", 0), graph = number("--graph", 0);
        const int warmup = number("--warmup", 1), iterations = number("--iterations", 1), repeats = number("--repeats", 5);
        if (m < 0 || n <= 0 || k <= 0 || fp32 > 1 || graph > 1 || warmup < 1 || iterations < 1 || repeats < 5 ||
            int64_t(m) * n > (1 << 28) || int64_t(m) * k > (1 << 28) || int64_t(n) * k > (1 << 28))
            throw std::invalid_argument("invalid GEMM shape/mode/timing bounds");
        const auto backend = args.at("--backend");
        if (backend != "sm70" && backend != "cublas") throw std::invalid_argument("unknown backend");
        if (backend == "cublas" && args.count("--bias"))
            throw std::invalid_argument("cuBLAS microbenchmark currently accepts bias-free common cases only");

        Check(cudaSetDevice(number("--device", 0)));
        if (backend == "sm70") ValidateSm70ExpertGemmDevice();
        Resources resources;
        auto input = static_cast<half *>(resources.Load(args.at("--input"), size_t(m) * k * 2));
        auto weight = static_cast<half *>(resources.Load(args.at("--weight"), size_t(n) * k * 2));
        auto bias = args.count("--bias") ? static_cast<half *>(resources.Load(args.at("--bias"), size_t(n) * 2)) : nullptr;
        size_t outputBytes = size_t(m) * n * (fp32 ? 4 : 2);
        void *output = resources.Allocate(outputBytes);
        if (backend == "cublas") {
            if (cublasCreate(&resources.blas) != CUBLAS_STATUS_SUCCESS ||
                cublasSetStream(resources.blas, cudaStreamPerThread) != CUBLAS_STATUS_SUCCESS ||
                cublasSetMathMode(resources.blas, CUBLAS_DEFAULT_MATH) != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("cuBLAS initialization failed");
            if (cublasSetWorkspace(resources.blas, resources.Allocate(4 << 20), 4 << 20) != CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("cuBLAS workspace initialization failed");
        }
        auto compute = [&]() {
            if (backend == "sm70") Sm70ExpertGemm(input, weight, bias, output, m, n, k, fp32, cudaStreamPerThread);
            else if (m) {
                const float alpha = 1, beta = 0;
                if (cublasGemmEx(resources.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha,
                    weight, CUDA_R_16F, k, input, CUDA_R_16F, k, &beta, output,
                    fp32 ? CUDA_R_32F : CUDA_R_16F, n, CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP) != CUBLAS_STATUS_SUCCESS)
                    throw std::runtime_error("cuBLAS GEMM failed");
            }
        };
        compute(); // 捕获前装载实际 kernel；两边均在计时外准备。
        Check(cudaStreamSynchronize(cudaStreamPerThread));
        if (graph) {
            Check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            try { compute(); }
            catch (...) {
                cudaStreamEndCapture(cudaStreamPerThread, &resources.graph);
                throw;
            }
            Check(cudaStreamEndCapture(cudaStreamPerThread, &resources.graph));
            Check(cudaGraphInstantiate(&resources.executable, resources.graph, nullptr, nullptr, 0));
        }
        auto execute = [&]() {
            if (graph) Check(cudaGraphLaunch(resources.executable, cudaStreamPerThread));
            else compute();
        };
        for (int i = 0; i < warmup; ++i) execute();
        Check(cudaStreamSynchronize(cudaStreamPerThread));
        Check(cudaEventCreate(&resources.start));
        Check(cudaEventCreate(&resources.stop));
        Json::array samples;
        std::vector<char> firstOutput(outputBytes), currentOutput(outputBytes);
        bool repeatOutputsEqual = true;
        for (int r = 0; r < repeats; ++r) {
            Check(cudaEventRecord(resources.start, cudaStreamPerThread));
            for (int i = 0; i < iterations; ++i) execute();
            Check(cudaEventRecord(resources.stop, cudaStreamPerThread));
            Check(cudaEventSynchronize(resources.stop));
            float elapsed = 0;
            Check(cudaEventElapsedTime(&elapsed, resources.start, resources.stop));
            samples.push_back(double(elapsed) / iterations);
            if (outputBytes) {
                Check(cudaMemcpyAsync(currentOutput.data(), output, outputBytes, cudaMemcpyDeviceToHost, cudaStreamPerThread));
                Check(cudaStreamSynchronize(cudaStreamPerThread));
            }
            if (!r) firstOutput = currentOutput;
            else repeatOutputsEqual = repeatOutputsEqual && currentOutput == firstOutput;
        }
        std::ofstream file(args.at("--output"), std::ios::binary);
        if (!file || (outputBytes && !file.write(currentOutput.data(), currentOutput.size())))
            throw std::runtime_error("output write failed");
        cudaDeviceProp properties{};
        Check(cudaGetDeviceProperties(&properties, number("--device", 0)));
        int runtime = 0;
        Check(cudaRuntimeGetVersion(&runtime));
        Json result = Json::object{{"status", "EXECUTED"}, {"backend", backend == "sm70" ? kMoeBatchInvariantBackend : "cublasGemmEx"},
            {"m", m}, {"n", n}, {"k", k}, {"fp32", bool(fp32)}, {"bias", bias != nullptr},
            {"gpu", properties.name}, {"cuda_runtime", runtime}, {"cuda_graph", bool(graph)},
            {"warmup", warmup}, {"iterations", iterations}, {"repeats", repeats}, {"samples_ms", samples},
            {"repeat_outputs_equal", repeatOutputsEqual}};
        std::cout << result.dump() << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
