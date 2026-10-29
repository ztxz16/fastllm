#include "models/deepseekv41.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-v41-decode.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <cstring>
#include <algorithm>
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
static float Bf16Value(uint16_t bits) {
    uint32_t value = uint32_t(bits) << 16; float result; std::memcpy(&result, &value, 4); return result;
}
static float E4m3Value(uint8_t bits) {
    int e=(bits>>3)&15, m=bits&7;
    float v=e?std::ldexp(1.f+m/8.f,e-7):std::ldexp(float(m),-9);
    return bits&128?-v:v;
}
int main(int argc, char **argv) {
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
        const bool shared = argc == 2 && std::string(argv[1]) == "--shared";
        Require(argc == 1 || shared, "expected --shared or no arguments");
        const std::vector<std::pair<int, int>> shapes = shared ?
            std::vector<std::pair<int, int>>{{4608, 5120}, {512, 1024}, {384, 512}} :
            std::vector<std::pair<int, int>>{{256, 256}, {25600, 6144}};
        int cases = 0;
        for (auto shape : shapes) {
            const int columns = shape.first, inner = shape.second;
            ProjectionProbe model;
            const std::string candidateName = shared ?
                "test.ffn.shared_experts.gateup.weight" : "test.engram.wkv.weight";
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
            const std::vector<uint8_t> original(weight.cpuData, weight.cpuData + weight.GetBytes());
            const std::vector<float> scales = weight.scales;
            model.quantizedLinearNames.insert(candidateName);
            model.quantizedLinearNames.insert(referenceName);
            for (int batch : {1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 32, 64, 128, 256, 513}) {
                FastllmCudaSetDevice(0);
                std::vector<float> values(batch * inner);
                for (size_t i = 0; i < values.size(); ++i)
                    values[i] = std::sin(float(i * 37 % 1031)) * 0.3f;
                Data input(DataType::FLOAT32, {1, batch, inner}, values), reference, actual, quantScratch, gatherScratch;
                ToDataType(input, DataType::BFLOAT16);
                input.ToDevice(DataDevice::CUDA, std::vector<int>{0});
                synchronize();
                PrepareMultiCudaReplicatedData(input, devices, true);
                model.LinearWithActivationQuant(input, referenceName, reference, true);
                model.LinearWithActivationQuant(input, candidateName, actual, true, &quantScratch, false, &gatherScratch);
                synchronize();
                Require(weight.IsTensorParallelReplicated(), "weight fallback layout changed");
                Require(actual.IsTensorParallelReplicated(), "result must be replicated");
                for (int device : devices) {
                    FastllmCudaSetDevice(device);
                    const auto ref = Bits(*reference.multiDeviceDatas.at(device));
                    const auto got = Bits(*actual.multiDeviceDatas.at(device));
                    double maxError = 0, squaredError = 0, squaredRef = 0;
                    size_t changed = 0;
                    for (size_t i = 0; i < ref.size(); i += 2) {
                        uint16_t rb, gb; std::memcpy(&rb, ref.data()+i, 2); std::memcpy(&gb, got.data()+i, 2);
                        uint32_t ri=uint32_t(rb)<<16, gi=uint32_t(gb)<<16; float a,b;
                        std::memcpy(&a,&ri,4); std::memcpy(&b,&gi,4);
                        Require(std::isfinite(a) && std::isfinite(b), "nonfinite projection");
                        maxError=std::max(maxError, double(std::abs(a-b))/std::max(1.0f,std::abs(a)));
                        squaredError += double(a-b)*(a-b); squaredRef += double(a)*a; changed += rb!=gb;
                    }
                    Require(maxError <= 0.008 && squaredError <= 1e-6 * std::max(squaredRef, 1.0),
                            "projection error exceeds BF16 tolerance");
                    if (batch == 1) Require(ref==got,"single-row projection bits changed");
                    if (device == 0) std::printf("shape=%dx%d M=%d changed=%zu max_scaled_error=%.9g relative_rms=%.9g\n",
                        columns,inner,batch,changed,maxError,std::sqrt(squaredError/std::max(squaredRef,1.0)));
                    if (device > 0) {
                        FastllmCudaSetDevice(0); const auto root=Bits(*actual.multiDeviceDatas.at(0));
                        FastllmCudaSetDevice(device); Require(got==root,"rank output mismatch");
                    }
                }
                FastllmCudaSetDevice(0);
                const auto qbits=Bits(*quantScratch.multiDeviceDatas.at(0));
                const auto obits=Bits(*actual.multiDeviceDatas.at(0));
                double oracleError=0,oracleNorm=0,maxOracle=0;
                for(int row: {0,batch/2,batch-1}) for(int part=0;part<ranks;++part)
                for(int within: {0,1,columns/ranks-1}) {
                    int col=part*(columns/ranks)+within;double sum=0;
                    for(int k=0;k<inner;++k) {
                        uint16_t in;std::memcpy(&in,qbits.data()+2*(size_t(row)*inner+k),2);
                        sum+=double(Bf16Value(in))*E4m3Value(original[size_t(col)*inner+k])*
                            scales[size_t(col/32)*(inner/32)+k/32];
                    }
                    uint16_t out;std::memcpy(&out,obits.data()+2*(size_t(row)*columns+col),2);
                    double delta=Bf16Value(out)-sum;oracleError+=delta*delta;oracleNorm+=sum*sum;
                    maxOracle=std::max(maxOracle,std::abs(delta)/(std::abs(sum)+.01));
                }
                Require(std::sqrt(oracleError/std::max(oracleNorm,1e-30))<.004 && maxOracle<.02,
                        "independent FP64 projection oracle mismatch");
                std::vector<std::vector<unsigned char>> expected(ranks);
                for (int device : devices) {FastllmCudaSetDevice(device); expected[device]=Bits(*actual.multiDeviceDatas.at(device));}
                std::vector<void *> graphs(ranks), execs(ranks);
                const bool previousAsync = MultiCudaSetPersistentAsyncDispatch(true);
                const bool previousCapture = MultiCudaSetWorkerOrderedCapture(true);
                Require(MultiCudaRunDeviceCallbacks(devices,[&](int rank,int device) {
                    Require(FastllmCudaGraphPrepareCaptureDevice(),"graph prepare failed");
                }),"prepare dispatch failed");
                Require(MultiCudaRunDeviceCallbacks(devices,[&](int rank,int device) {
                    Require(FastllmCudaGraphBeginCapture(),"graph begin failed");
                }),"begin dispatch failed");
                model.LinearWithActivationQuant(input, candidateName, actual, true, &quantScratch, false, &gatherScratch);
                Require(MultiCudaRunDeviceCallbacks(devices,[&](int rank,int device) {
                    Require(FastllmCudaGraphEndCapture(&graphs[rank]),"graph end failed");
                    Require(FastllmCudaGraphInstantiate(graphs[rank],&execs[rank]),"graph instantiate failed");
                }),"end dispatch failed");
                for(int replay=0;replay<4;++replay) {
                    Require(MultiCudaRunDeviceCallbacks(devices,[&](int rank,int device) {
                        Require(cudaMemsetAsync(actual.multiDeviceDatas.at(device)->cudaData,0xFF,
                            actual.multiDeviceDatas.at(device)->GetBytes(),cudaStreamPerThread)==cudaSuccess,"poison failed");
                        Require(FastllmCudaGraphLaunch(execs[rank]),"graph launch failed");
                    }),"replay dispatch failed");
                    synchronize();
                    for(int device:devices) {FastllmCudaSetDevice(device);Require(Bits(*actual.multiDeviceDatas.at(device))==expected[device],"graph replay changed result");}
                }
                MultiCudaRunDeviceCallbacks(devices,[&](int rank,int device) {
                    FastllmCudaGraphExecDestroy(execs[rank]); FastllmCudaGraphDestroy(graphs[rank]);
                });
                MultiCudaSetWorkerOrderedCapture(previousCapture);
                MultiCudaSetPersistentAsyncDispatch(previousAsync);
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
        std::printf("PASS %s projection: %d cases, TP%d\n", shared ? "Shared" : "Engram", cases, ranks);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
