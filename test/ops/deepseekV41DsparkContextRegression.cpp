#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace fastllm;
static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
int main() {
    try {
        FastllmCudaSetDevice(0);
        int cases = 0;
        for (int width : {7, 5120}) for (int rows = 1; rows <= 8; ++rows) {
            std::vector<Data> inputs(3);
            std::vector<const Data *> sources;
            std::vector<std::vector<uint16_t>> values(3);
            for (int k = 0; k < 3; ++k) {
                auto &x = inputs[k]; x.dataType = BFLOAT16; x.Resize({1, 8, width});
                x.ToDevice(DataDevice::CUDA, {0}, false); x.Allocate(false);
                values[k].resize(8 * width); sources.push_back(&x);
            }
            Data output;
            auto prepare = [&](int seed) {
                for (int k = 0; k < 3; ++k) {
                    for (size_t i = 0; i < values[k].size(); ++i)
                        values[k][i] = uint16_t(i * 31337 + k * 17041 + seed);
                    Require(cudaMemcpy(inputs[k].cudaData, values[k].data(), values[k].size() * 2,
                                       cudaMemcpyHostToDevice) == cudaSuccess, "upload failed");
                }
            };
            auto run = [&]() {
                Require(FastllmCudaNaiveDraftConcat(sources, rows, output), "concat rejected valid inputs");
            };
            auto verify = [&]() {
                Require(output.dataType == BFLOAT16 && output.dims == std::vector<int>({1, rows, 3 * width}),
                        "concat layout changed");
                std::vector<uint16_t> actual(output.Count(0));
                Require(cudaMemcpy(actual.data(), output.cudaData, actual.size() * 2,
                                   cudaMemcpyDeviceToHost) == cudaSuccess, "read failed");
                for (int r = 0; r < rows; ++r) for (int k = 0; k < 3; ++k) for (int d = 0; d < width; ++d)
                    Require(actual[(r * 3 + k) * width + d] == values[k][r * width + d],
                            "context concat changed bits or selected the wrong row");
            };
            prepare(1); run(); verify();
            cudaGraph_t graph; cudaGraphExec_t exec;
            Require(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal) == cudaSuccess,
                    "capture failed");
            run();
            Require(cudaStreamEndCapture(cudaStreamPerThread, &graph) == cudaSuccess, "end capture failed");
            Require(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess, "instantiate failed");
            for (int seed : {0, 777, 65535}) {
                prepare(seed);
                Require(cudaMemsetAsync(output.cudaData, 0xff, output.GetBytes(), cudaStreamPerThread) == cudaSuccess,
                        "poison failed");
                Require(cudaGraphLaunch(exec, cudaStreamPerThread) == cudaSuccess, "replay failed");
                verify();
            }
            cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
            auto dims = output.dims;
            void *pointer = output.cudaData;
            for (DataType type : {FLOAT16, FLOAT32}) {
                inputs[1].dataType = type;
                Require(!FastllmCudaNaiveDraftConcat(sources, rows, output), "unsupported type accepted");
            }
            inputs[1].dataType = BFLOAT16;
            inputs[1].strides[1] += 1;
            Require(!FastllmCudaNaiveDraftConcat(sources, rows, output), "strided input accepted");
            inputs[1].strides[1] -= 1;
            Require(output.cudaData == pointer && output.dims == dims, "rejection changed output");
            verify();
            std::printf("rows=%d width=%d eager+graph exact BF16 bits and fallback guards PASS\n", rows, width);
            ++cases;
        }
        std::printf("DSpark context %d cases PASS\n", cases);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what()); return 1;
    }
}
