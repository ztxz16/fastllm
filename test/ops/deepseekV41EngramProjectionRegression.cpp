#include "models/deepseekv41.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-v41-decode.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace fastllm;
struct ProjectionProbe : DeepSeekV41Model {
    using DeepSeekV41Model::LinearWithActivationQuant;
    using DeepSeekV41Model::quantizedLinearNames;
};
static void Require(bool valid, const char *message) {
    if (!valid) throw std::runtime_error(message);
}
static std::vector<unsigned char> Bits(const Data &data) {
    std::vector<unsigned char> bytes(data.GetBytes());
    Require(cudaMemcpy(bytes.data(), data.cudaData, bytes.size(), cudaMemcpyDeviceToHost) == cudaSuccess,
            "copy failed");
    return bytes;
}
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) return 77;
    try {
        const int ranks = count >= 8 ? 8 : 2;
        std::vector<int> devices;
        std::string spec = "multicuda:";
        for (int rank = 0; rank < ranks; ++rank) {
            devices.push_back(rank);
            spec += (rank ? "," : "") + std::to_string(rank);
        }
        SetThreads(16);
        ApplyDeviceMap({{spec, 1}}, 0, 1);
        FastllmMultiCudaSetDevice(devices);
        FastllmCudaSetNcclForceSync(false);
        auto synchronize = [&] {
            for (int device : devices) FastllmCudaSyncDevice(device);
            FastllmCudaSetDevice(0);
        };
        int cases = 0;
        for (auto shape : {std::make_pair(256, 256), std::make_pair(25600, 6144)}) {
            const int columns = shape.first, inner = shape.second;
            ProjectionProbe model;
            const std::string candidateName = "test.engram.wkv.weight";
            const std::string referenceName = "reference.weight";
            Data &weight = model.weight.weight[candidateName];
            weight.dataType = DataType::FP8_E4M3;
            weight.UpdateUnitSize();
            weight.Resize({columns, inner});
            weight.blockM = weight.blockK = 32;
            weight.Allocate();
            for (size_t i = 0; i < weight.GetBytes(); ++i)
                weight.cpuData[i] = ((i * 13 + i / inner * 19) % 2 ? 128 : 0) |
                                    ((i * 31 + i / inner * 7) % 105);
            weight.scales.resize(size_t(columns / 32) * (inner / 32));
            for (size_t i = 0; i < weight.scales.size(); ++i)
                weight.scales[i] = std::ldexp(1.0f, int(i * 13 % 19) - 16);
            Data &referenceWeight = model.weight.weight[referenceName];
            referenceWeight.CopyFrom(weight);
            // Data::CopyFrom copies tensor storage, not quantization metadata.
            referenceWeight.blockM = referenceWeight.blockK = 32;
            referenceWeight.scales = weight.scales;
            model.quantizedLinearNames.insert(candidateName);
            model.quantizedLinearNames.insert(referenceName);
            for (int batch : {1, 2, 8}) {
                FastllmCudaSetDevice(0);
                std::vector<float> values(batch * inner);
                for (size_t i = 0; i < values.size(); ++i)
                    values[i] = std::sin(float(i * 37 % 1031)) * 0.3f;
                Data input(DataType::FLOAT32, {1, batch, inner}, values), reference, actual;
                ToDataType(input, DataType::BFLOAT16);
                input.ToDevice(DataDevice::CUDA, std::vector<int>{0});
                synchronize();
                PrepareMultiCudaReplicatedData(input, devices, true);
                model.LinearWithActivationQuant(input, referenceName, reference, true);
                model.LinearWithActivationQuant(input, candidateName, actual, true);
                synchronize();
                Require(weight.IsTensorParallelReplicated(), "weight fallback layout changed");
                Require(actual.IsTensorParallelReplicated(), "result must be replicated");
                for (int device : devices) {
                    FastllmCudaSetDevice(device);
                    Require(Bits(*reference.multiDeviceDatas.at(device)) == Bits(*actual.multiDeviceDatas.at(device)),
                            "Engram projection changed output bits or channel order");
                }
                if (batch == 1) {
                    FastllmCudaSetDevice(0);
                    Data &localInput = *input.multiDeviceDatas.at(0);
                    Data &localWeight = *weight.multiDeviceDatas.at(0);
                    Data &localOutput = *actual.multiDeviceDatas.at(0);
                    const auto before = Bits(localOutput);
                    for (auto slice : {std::make_pair(-1, 32), std::make_pair(1, 32),
                                       std::make_pair(32, INT_MAX), std::make_pair(columns, 32)}) {
                        Require(!FastllmCudaBFloat16FP8ReplicatedRowSlice(localInput, localWeight,
                            localOutput, slice.first, slice.second) && Bits(localOutput) == before,
                            "invalid row slice must leave output untouched");
                    }
                }
                ++cases;
            }
        }
        std::printf("PASS Engram projection: %d cases, TP%d\n", cases, ranks);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
