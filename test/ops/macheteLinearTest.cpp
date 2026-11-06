#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>

namespace {
/** 检查测试中的CUDA操作，任何失败均返回非零退出状态。 */
void Check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
/** 固定种子的无状态输入生成，两个独立后端进程得到相同张量。 */
unsigned Value(unsigned i) {
    i ^= 0x9e3779b9u;
    i ^= i >> 16; i *= 0x7feb352du;
    i ^= i >> 15; i *= 0x846ca68bu;
    return i ^ (i >> 16);
}
}

/**
 * 经正式Linear公开入口验证及测量原生/Machete，不绕过调度和后端选择。
 * 参数为backend bits atype M N K group bias repeats iterations graph device。
 * 原始CUDA Event样本写入JSON行；reference在计时外以CPU FP64累加计算。
 */
int main(int argc, char **argv) {
    try {
        if (argc != 13 && argc != 14) throw std::runtime_error(
            "usage: macheteLinearTest backend bits fp16|bf16 M N K group bias repeats iterations graph device [symmetric|asymmetric|min]");
        std::string backend = argv[1], atype = argv[3];
        int bits = std::stoi(argv[2]), m = std::stoi(argv[4]), n = std::stoi(argv[5]);
        int k = std::stoi(argv[6]), groupSize = std::stoi(argv[7]), hasBias = std::stoi(argv[8]);
        int repeats = std::stoi(argv[9]), iterations = std::stoi(argv[10]);
        bool graph = std::stoi(argv[11]) != 0;
        int device = std::stoi(argv[12]);
        std::string quantMode = argc == 14 ? argv[13] : "symmetric";
        if ((backend != "native" && backend != "machete" && backend != "auto") ||
            (atype != "fp16" && atype != "bf16") || (bits != 4 && bits != 8) ||
            m <= 0 || n <= 0 || k <= 0 || groupSize <= 0 || k % groupSize ||
            repeats < 5 || iterations < 1 || (bits == 8 && groupSize != k) ||
            (quantMode != "symmetric" && quantMode != "asymmetric" && quantMode != "min") ||
            (bits == 8 && quantMode == "min")) {
            throw std::runtime_error("invalid test arguments");
        }
        setenv("FASTLLM_LINEAR_BACKEND", backend.c_str(), 1);
        setenv("FASTLLM_LINEAR_BACKEND_TRACE", "1", 1);
        fastllm::SetDeviceMap({{"cuda:" + std::to_string(device), 1}});
        FastllmCudaSetDevice(device);
        using fastllm::Data;
        using fastllm::DataType;
        using fastllm::DataDevice;
        auto dtype = atype == "fp16" ? DataType::FLOAT16 : DataType::BFLOAT16;
        std::vector<float> values(size_t(m) * k), biases(n);
        for (size_t i = 0; i < values.size(); i++) values[i] = (int(Value(i) % 201) - 100) * 0.001f;
        for (int i = 0; i < n; i++) biases[i] = (int(Value(i + 123) % 21) - 10) * 0.001f;
        Data input(dtype, {m, k}, values);
        fastllm::ToDataTypeForceCPU(input, DataType::FLOAT32);
        values.assign(reinterpret_cast<float*>(input.cpuData), reinterpret_cast<float*>(input.cpuData) + values.size());
        fastllm::ToDataTypeForceCPU(input, dtype);
        Data weight(bits == 8 ? DataType::INT8 : DataType::INT4_GROUP, {n, k});
        weight.name = "machete_compare.weight";
        weight.groupCnt = groupSize;
        weight.group = k / groupSize;
        weight.perChannelAxis = 0;
        weight.scales.resize(size_t(n) * weight.group);
        weight.zeros.resize(weight.scales.size());
        weight.mins.resize(weight.scales.size());
        for (size_t i = 0; i < weight.scales.size(); i++) {
            weight.scales[i] = 0.00317f + (Value(i + 42) % 10) * 0.00013f;
            weight.zeros[i] = bits == 8 ? 128 : 8;
            if (quantMode == "asymmetric") weight.zeros[i] += int(Value(i + 51) % 5) - 2;
            weight.mins[i] = -weight.scales[i] * weight.zeros[i];
            if (quantMode == "min") weight.mins[i] += 0.00073f;
        }
        if (quantMode == "min") weight.zeros.clear();
        weight.Allocate(true);
        for (size_t i = 0; i < size_t(n) * k; i++) {
            unsigned q = Value(i + 321) & ((1u << bits) - 1);
            if (bits == 8) weight.cpuData[i] = q;
            else if ((i & 1) == 0) weight.cpuData[i / 2] = q << 4;
            else weight.cpuData[i / 2] |= q;
        }
        std::vector<float> reference(size_t(m) * n);
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < n; col++) {
                double sum = hasBias ? biases[col] : 0.0;
                for (int x = 0; x < k; x++) {
                    unsigned q = Value(size_t(col) * k + x + 321) & ((1u << bits) - 1);
                    size_t g = size_t(col) * weight.group + x / groupSize;
                    float w = quantMode == "min" ? weight.scales[g] * q + weight.mins[g] :
                                                   weight.scales[g] * (float(q) - weight.zeros[g]);
                    sum += double(values[size_t(row) * k + x]) * w;
                }
                reference[size_t(row) * n + col] = sum;
            }
        }
        Data bias(DataType::FLOAT32), output(dtype);
        if (hasBias) {
            Data sourceBias(DataType::FLOAT32, {n}, biases);
            bias.CopyFrom(sourceBias);
            bias.ToDevice(DataDevice::CUDA, {device});
        }
        input.ToDevice(DataDevice::CUDA, {device});
        weight.ToDevice(DataDevice::CUDA, {device});
        fastllm::Linear(input, weight, bias, output);
        Check(cudaDeviceSynchronize());
        if (backend == "machete") {
            // 正式Data生命周期：深拷贝、释放源GPU缓存、迁移回CPU再迁回GPU。
            // 随后的数值检查必须仍通过，不能命中已释放地址对应的旧预打包。
            Data replacement;
            replacement.CopyFrom(weight);
            weight.CopyFrom(replacement);
            weight.ToDevice(DataDevice::CPU);
            weight.ToDevice(DataDevice::CUDA, {device});
            fastllm::Linear(input, weight, bias, output);
            Check(cudaDeviceSynchronize());
        }
        setenv("FASTLLM_LINEAR_BACKEND_TRACE", "0", 1);
        Data actual;
        actual.CopyFrom(output);
        actual.ToDevice(DataDevice::CPU);
        fastllm::ToDataTypeForceCPU(actual, DataType::FLOAT32);
        double maximum = 0, mean = 0;
        size_t failed = 0;
        const float *got = reinterpret_cast<float*>(actual.cpuData);
        for (size_t i = 0; i < reference.size(); i++) {
            double error = std::abs(double(got[i]) - reference[i]);
            if (!std::isfinite(got[i]) || error > 0.02 + 0.05 * std::abs(reference[i])) failed++;
            maximum = std::max(maximum, error);
            mean += error;
        }
        mean /= reference.size();
        const double firstMaximum = maximum, firstMean = mean;
        const size_t firstFailed = failed;
        for (int i = 0; i < 5; i++) fastllm::Linear(input, weight, bias, output);
        Check(cudaDeviceSynchronize());
        void *captured = nullptr;
        void *executable = nullptr;
        if (graph) {
            if (!FastllmCudaGraphBeginCapture()) throw std::runtime_error(FastllmCudaGraphLastError());
            fastllm::Linear(input, weight, bias, output);
            if (!FastllmCudaGraphEndCapture(&captured) ||
                !FastllmCudaGraphInstantiate(captured, &executable) ||
                !FastllmCudaGraphLaunch(executable)) throw std::runtime_error(FastllmCudaGraphLastError());
            Check(cudaDeviceSynchronize());
        }
        cudaEvent_t start, stop;
        Check(cudaEventCreate(&start)); Check(cudaEventCreate(&stop));
        std::vector<float> samples;
        Check(cudaProfilerStart());
        for (int r = 0; r < repeats; r++) {
            Check(cudaEventRecord(start, cudaStreamPerThread));
            for (int i = 0; i < iterations; i++) {
                if (graph) {
                    if (!FastllmCudaGraphLaunch(executable)) throw std::runtime_error(FastllmCudaGraphLastError());
                } else fastllm::Linear(input, weight, bias, output);
            }
            Check(cudaEventRecord(stop, cudaStreamPerThread));
            Check(cudaEventSynchronize(stop));
            float elapsed;
            Check(cudaEventElapsedTime(&elapsed, start, stop));
            samples.push_back(elapsed / iterations);
        }
        Check(cudaProfilerStop());
        Check(cudaEventDestroy(start)); Check(cudaEventDestroy(stop));
        // 再检查计时后的输出，Graph replay也必须产出正确结果。
        actual.CopyFrom(output);
        actual.ToDevice(DataDevice::CPU);
        fastllm::ToDataTypeForceCPU(actual, DataType::FLOAT32);
        got = reinterpret_cast<float*>(actual.cpuData);
        maximum = 0; mean = 0; failed = 0;
        for (size_t i = 0; i < reference.size(); i++) {
            double error = std::abs(double(got[i]) - reference[i]);
            if (!std::isfinite(got[i]) || error > 0.02 + 0.05 * std::abs(reference[i])) failed++;
            maximum = std::max(maximum, error);
            mean += error;
        }
        mean /= reference.size();
        maximum = std::max(maximum, firstMaximum);
        mean = (mean + firstMean) / 2;
        failed += firstFailed;
        if (executable) FastllmCudaGraphExecDestroy(executable);
        if (captured) FastllmCudaGraphDestroy(captured);
        cudaDeviceProp prop;
        Check(cudaGetDeviceProperties(&prop, device));
        int runtimeVersion;
        Check(cudaRuntimeGetVersion(&runtimeVersion));
        std::cout << std::setprecision(12) << "MACHETE_RESULT {\"backend\":\"" << backend
                  << "\",\"gpu\":\"" << prop.name << "\",\"cuda_runtime\":" << runtimeVersion
                  << ",\"cuda_toolkit\":" << CUDART_VERSION
#ifdef FASTLLM_ENABLE_MACHETE
                  << ",\"machete_build_arch\":\"sm_90a\""
#else
                  << ",\"machete_build_arch\":\"not_built\""
#endif
                  << ",\"max_abs_error\":" << maximum << ",\"mean_abs_error\":" << mean
                  << ",\"failed_elements\":" << failed << ",\"checked_outputs\":2,\"atol\":0.02,\"rtol\":0.05,\"samples_ms\":[";
        for (size_t i = 0; i < samples.size(); i++) std::cout << (i ? "," : "") << samples[i];
        std::cout << "],\"status\":\"" << (failed ? "FAIL" : "PASS") << "\"}\n";
        return failed ? 1 : 0;
    } catch (const std::exception &error) {
        std::cerr << "Machete Linear test failed: " << error.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Machete Linear test failed: non-standard exception\n";
        return 1;
    }
}
