#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace fastllm;
namespace {
void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
void Cuda(cudaError_t error) { Check(error == cudaSuccess, cudaGetErrorString(error)); }
}
int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try {
        FastllmCudaSetDevice(0);
        int cases = 0;
        for (auto type : {INT8, FLOAT16, BFLOAT16, FLOAT32})
        for (auto shape : std::vector<std::vector<int>>{
                {1, 2, 16, 124160}, {2, 3, 7, 1024}, {3, 5, 2, 17},
                {1, 7, 3, 1}, {2, 2, 9, 2051}})
        for (int offset : {0, 1}) {
            Data input(type, shape), output(type, {shape[0], shape[2], shape[1], shape[3]});
            const size_t bytes = input.GetBytes(), rowBytes = shape[3] * input.unitSize;
            Data owner(type, {int(input.Count(0) + 2)});
            owner.ToDevice(DataDevice::CUDA, std::vector<int>{0}, false); owner.Allocate(false);
            input.dataDevice = DataDevice::CUDA;
            input.dataDeviceIds = {0};
            input.cudaData = (char *)owner.cudaData + offset * input.unitSize;
            input.cudaDataBorrowed = true;
            output.ToDevice(DataDevice::CUDA, std::vector<int>{0}, false); output.Allocate(false);
            std::vector<unsigned char> source(bytes), expected(bytes), actual(bytes);
            for (size_t i = 0; i < bytes; ++i) source[i] = (i * 17 + i / 97) % 251;
            auto oracle = [&] {
                for (int b = 0; b < shape[0]; ++b)
                for (int n = 0; n < shape[1]; ++n)
                for (int m = 0; m < shape[2]; ++m)
                    std::memcpy(expected.data() + ((b * shape[2] + m) * shape[1] + n) * rowBytes,
                                source.data() + ((b * shape[1] + n) * shape[2] + m) * rowBytes, rowBytes);
            };
            auto launch = [&] {
                Check(FastllmCudaPermuteTo(input, output, {0, 2, 1, 3}), "transpose failed");
            };
            auto verify = [&] {
                Cuda(cudaStreamSynchronize(cudaStreamPerThread));
                Cuda(cudaMemcpy(actual.data(), output.cudaData, bytes, cudaMemcpyDeviceToHost));
                Check(actual == expected, "transpose oracle mismatch");
                Cuda(cudaMemcpy(actual.data(), input.cudaData, bytes, cudaMemcpyDeviceToHost));
                Check(actual == source, "input changed");
            };
            oracle();
            Cuda(cudaMemcpy(input.cudaData, source.data(), bytes, cudaMemcpyHostToDevice));
            launch(); verify();
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            launch();
            Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            Cuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            for (auto &value : source) value ^= 0x55;
            oracle();
            Cuda(cudaMemcpy(input.cudaData, source.data(), bytes, cudaMemcpyHostToDevice));
            Cuda(cudaGraphLaunch(executable, cudaStreamPerThread)); verify();
            Cuda(cudaGraphExecDestroy(executable)); Cuda(cudaGraphDestroy(graph));
            Data inplace(type, shape);
            inplace.ToDevice(DataDevice::CUDA, std::vector<int>{0}, false); inplace.Allocate(false);
            Cuda(cudaMemcpy(inplace.cudaData, source.data(), bytes, cudaMemcpyHostToDevice));
            Check(FastllmCudaPermute(inplace, {0, 2, 1, 3}), "inplace transpose failed");
            Cuda(cudaStreamSynchronize(cudaStreamPerThread));
            Cuda(cudaMemcpy(actual.data(), inplace.cudaData, bytes, cudaMemcpyDeviceToHost));
            Check(actual == expected && inplace.dims == output.dims, "inplace transpose oracle mismatch");
            ++cases;
        }
        std::cout << "PASS " << cases << " block transpose oracle, offset views and refreshed graph checks\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
