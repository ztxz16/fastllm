#include "fastllm.h"
#include "executor.h"
#include "baseblock.h"
#include "models/deepseekv41.h"
#include <thread>
#include <chrono>
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static Data Make(DataType type, const std::vector<int> &dims, int seed) {
    size_t n = 1;
    for (int d : dims) n *= d;
    std::vector<float> values(n);
    for (size_t i = 0; i < n; ++i) values[i] = ((int)((i * 37 + seed) % 127) - 63) / 64.f;
    return Data(type, dims, values);
}
static std::vector<uint8_t> Bytes(Data &value) {
    Data cpu;
    cpu.CopyFrom(value);
    cpu.ToDevice(DataDevice::CPU);
    return {cpu.cpuData, cpu.cpuData + cpu.GetBytes()};
}
static int SharedSwigluChecks() {
    int checks = 0;
    for (auto type : {DataType::BFLOAT16, DataType::FLOAT16, DataType::FLOAT32}) {
        for (bool sharded : {false, true}) {
            Data output;
            for (int tokens : {1, 5, 33, 1}) {
                constexpr int mid = 512;
                auto source = Make(type, {tokens, 2 * mid}, tokens);
                auto input = Make(type, {tokens, 2 * mid}, tokens);
                ApplyDeviceMap({{"cuda:0", 1}}, 0, 1);
                source.ToDevice(DataDevice::CUDA, std::vector<int>{0});
                Data expected;
                if (!FastllmCudaDeepSeekV41SharedSwiglu(source, .5f, expected))
                    throw std::runtime_error("single CUDA Swiglu failed");
                auto expectedBytes = Bytes(expected);
                ApplyDeviceMap({{"multicuda:0,1", 1}}, 0, 1);
                FastllmMultiCudaSetDevice({0, 1});
                if (sharded) {
                    DivisionScheme scheme{{0, {{0, mid / 2}, {mid, mid + mid / 2}}},
                                          {1, {{mid / 2, mid}, {mid + mid / 2, mid * 2}}}};
                    // Fill each paired gate/up shard independently of the helper.
                    PrepareMultiCudaShardedData(input, {0, 1}, {tokens, mid * 2}, 1, scheme);
                    auto host = Make(type, {tokens, mid * 2}, tokens);
                    for (int d : {0, 1}) {
                        Data local(type, {tokens, mid}); local.Allocate();
                        for (int row = 0; row < tokens; ++row) {
                            int col = 0;
                            for (auto range : scheme.at(d)) {
                                int n = range.second - range.first;
                                memcpy(local.cpuData + (row * mid + col) * local.unitSize,
                                       host.cpuData + (row * mid * 2 + range.first) * host.unitSize,
                                       n * host.unitSize);
                                col += n;
                            }
                        }
                        local.ToDevice(DataDevice::CUDA, std::vector<int>{d});
                        input.multiDeviceDatas.at(d)->CopyFrom(local);
                    }
                } else {
                    PrepareMultiCudaReplicatedData(input, {0, 1}, true);
                }
                if (!MultiCudaDeepSeekV41SharedSwiglu(input, .5f, output))
                    throw std::runtime_error("TP shared Swiglu failed");
                if (output.dataType != DataType::BFLOAT16 || output.dims != expected.dims)
                    throw std::runtime_error("TP shared Swiglu metadata mismatch");
                for (int d : {0, 1}) {
                    auto actual = Bytes(*output.multiDeviceDatas.at(d));
                    const int width = sharded ? mid / 2 : mid;
                    const int start = sharded ? d * width : 0;
                    if (actual.size() != tokens * width * 2)
                        throw std::runtime_error("TP shared Swiglu local size mismatch");
                    for (int row = 0; row < tokens; ++row)
                        if (memcmp(actual.data() + row * width * 2,
                                   expectedBytes.data() + (row * mid + start) * 2, width * 2))
                            throw std::runtime_error("TP shared Swiglu rounding/clamp mismatch");
                    ++checks;
                }
            }
        }
    }
    return checks;
}
static int AddHcPostChecks() {
    Executor &exec = *((Executor *)GetExecutor());
    int checks = 0;
    for (auto type : {DataType::BFLOAT16, DataType::FLOAT16, DataType::FLOAT32}) {
        Data output;
        for (int tokens : {1, 5, 33, 1}) {
            constexpr int dim = 5120;
            ApplyDeviceMap({{"cuda:0", 1}}, 0, 1);
            auto input = Make(type, {1, tokens, dim}, tokens + 3);
            auto shared = Make(type, {1, tokens, dim}, tokens + 29);
            auto residual = Make(type, {1, tokens, 4, dim}, tokens + 7);
            auto post = Make(DataType::FLOAT32, {1, tokens, 4}, 11);
            auto comb = Make(DataType::FLOAT32, {1, tokens, 4, 4}, 13);
            Data expected;
            exec.Run("AddTo", {{"input0", &input}, {"input1", &shared}}, {{"alpha", 1}}, {});
            exec.Run("DeepSeekV41HcPost", {{"input", &input}, {"residual", &residual},
                {"post", &post}, {"comb", &comb}, {"output", &expected}}, {}, {});
            auto reference = Bytes(expected);
            auto tpInput = Make(type, {1, tokens, dim}, tokens + 3);
            auto tpShared = Make(type, {1, tokens, dim}, tokens + 29);
            ApplyDeviceMap({{"multicuda:0,1", 1}}, 0, 1);
            if (!MultiCudaDeepSeekV41AddHcPost(tpInput, tpShared, residual, post, comb, output))
                throw std::runtime_error("combined FFN post dispatch rejected valid tensors");
            for (int dev : {0, 1}) {
                if (Bytes(*output.multiDeviceDatas.at(dev)) != reference)
                    throw std::runtime_error("combined FFN post changed intermediate rounding");
                ++checks;
            }
        }
    }
    return checks;
}
static int HandoffChecks() {
    int checks = 0;
    Data cpu;
    for (auto type : {DataType::BFLOAT16, DataType::FLOAT16, DataType::FLOAT32, DataType::INT32}) {
        for (int width : {5120, 513, 8192, 5120}) {
            auto input = Make(type == DataType::INT32 ? DataType::FLOAT32 : type, {1, width}, width);
            if (type == DataType::INT32) { input.dataType = type; input.UpdateUnitSize(); }
            auto expected = Bytes(input);
            PrepareMultiCudaReplicatedData(input, {0, 1}, true);
            for (auto devices : {std::vector<int>{0, 1}, std::vector<int>{1, 0}}) {
                void *before = input.multiDeviceDatas.at(devices[0])->cudaData;
                if (!MultiCudaCopyReplicaToCpu(cpu, input, devices) || Bytes(cpu) != expected ||
                    cpu.multiDeviceData || cpu.IsTensorParallelReplicated() ||
                    input.multiDeviceDatas.at(devices[0])->cudaData != before)
                    throw std::runtime_error("direct replica D2H changed bytes, layout or source storage");
                uint8_t *storage = cpu.cpuData;
                if (!MultiCudaCopyReplicaToCpu(cpu, input, devices) || cpu.cpuData != storage)
                    throw std::runtime_error("direct replica D2H did not reuse CPU storage");
                ++checks;
            }
            auto &local = *input.multiDeviceDatas.at(0);
            local.isKVCache = true;
            if (MultiCudaCopyReplicaToCpu(cpu, input, {0}))
                throw std::runtime_error("direct D2H accepted a cache tensor");
            local.isKVCache = false;
            local.strides[0] += 1;
            if (MultiCudaCopyReplicaToCpu(cpu, input, {0}))
                throw std::runtime_error("direct D2H accepted expanded strides");
            local.strides[0] -= 1;
            Data view(type, cpu.dims, DataDevice::CPU, cpu.cpuData);
            if (MultiCudaCopyReplicaToCpu(view, input, {0}) || MultiCudaCopyReplicaToCpu(cpu, cpu, {0}))
                throw std::runtime_error("direct D2H accepted an aliased destination");
            checks += 3;
        }
    }
    const bool previous = MultiCudaSetPersistentAsyncDispatch(true);
    Executor &exec = *((Executor *)GetExecutor());
    for (auto type : {DataType::BFLOAT16, DataType::FLOAT16, DataType::FLOAT32}) {
        for (int hc : {2, 4}) {
            Data output, input;
            for (int tokens : {1, 5, 1, 1}) {
                ApplyDeviceMap({{"cuda:0", 1}}, 0, 1);
                auto original = Make(type, {1, tokens, 513}, tokens + checks);
                auto shared = Make(type, original.dims, 29);
                auto residual = Make(type, {1, tokens, hc, 513}, 7);
                auto post = Make(DataType::FLOAT32, {1, tokens, hc}, 11);
                auto comb = Make(DataType::FLOAT32, {1, tokens, hc, hc}, 13);
                Data expected;
                auto originalBytes = Bytes(original);
                exec.Run("AddTo", {{"input0", &original}, {"input1", &shared}}, {{"alpha", 1}}, {});
                exec.Run("DeepSeekV41HcPost", {{"input", &original}, {"residual", &residual},
                    {"post", &post}, {"comb", &comb}, {"output", &expected}}, {}, {});
                auto reference = Bytes(expected);
                input.dataType = type; input.UpdateUnitSize(); input.Resize({tokens, 513});
                input.Allocate(false);
                std::memcpy(input.cpuData, originalBytes.data(), originalBytes.size());
                ApplyDeviceMap({{"multicuda:0,1", 1}}, 0, 1);
                std::map<int, void *> retained;
                for (const auto &item : input.multiDeviceDatas)
                    if (item.second->dims == input.dims) retained[item.first] = item.second->cudaData;
                if (!MultiCudaDeepSeekV41AddHcPost(input, shared, residual, post, comb, output))
                    throw std::runtime_error("CPU upload + FFN post rejected valid tensors");
                // The source must already be staged even if GPU work remains.
                std::memset(input.cpuData, 0xa5, originalBytes.size());
                for (int dev : {0, 1}) {
                    FastllmCudaSetDevice(dev);
                    if (retained.count(dev) && input.multiDeviceDatas.at(dev)->cudaData != retained.at(dev))
                        throw std::runtime_error("CPU upload replaced an unchanged GPU replica");
                    if (Bytes(*output.multiDeviceDatas.at(dev)) != reference)
                        throw std::runtime_error("CPU upload + FFN post changed rounding or source lifetime");
                    ++checks;
                }
            }
        }
    }
    MultiCudaSetPersistentAsyncDispatch(previous);
    return checks;
}
static int DelayedLogitsGatherChecks() {
    constexpr int vocab = 256, localVocab = vocab / 2;
    ApplyDeviceMap({{"multicuda:0,1", 1}}, 0, 1);
    Data logits(DataType::FLOAT32, {1, 1, vocab});
    DivisionScheme scheme{{0, {{0, localVocab}}}, {1, {{localVocab, vocab}}}};
    PrepareMultiCudaShardedData(logits, {0, 1}, {1, 1, vocab}, 2, scheme);
    float *host[2] = {nullptr, nullptr};
    for (int dev : {0, 1}) {
        FastllmCudaSetDevice(dev);
        logits.multiDeviceDatas.at(dev)->Allocate();
        if (cudaMallocHost(&host[dev], localVocab * sizeof(float)) != cudaSuccess)
            throw std::runtime_error("pinned logits allocation failed");
        for (int i = 0; i < localVocab; ++i) host[dev][i] = dev * localVocab + i + 1;
        cudaMemset(logits.multiDeviceDatas.at(dev)->cudaData, 0, localVocab * sizeof(float));
        cudaDeviceSynchronize();
    }
    const bool previous = MultiCudaSetPersistentAsyncDispatch(true);
    if (!MultiCudaRunDeviceCallbacks({0, 1}, [&](int rank, int dev) {
        if (dev == 1) cudaLaunchHostFunc(cudaStreamPerThread, [](void *) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }, nullptr);
        if (cudaMemcpyAsync(logits.multiDeviceDatas.at(dev)->cudaData, host[dev],
            localVocab * sizeof(float), cudaMemcpyHostToDevice, cudaStreamPerThread) != cudaSuccess)
            throw std::runtime_error("delayed logits upload failed");
    })) throw std::runtime_error("delayed logits dispatch rejected");
    ApplyDeviceMap({{"cuda:0", 1}}, 0, 1);
    DeepSeekV41Model model;
    GenerationConfig config; config.output_logits = true;
    std::vector<std::pair<Data *, Data *>> past;
    LastTokensManager last(1, 64);
    std::vector<float> actual;
    std::vector<std::vector<float> *> results{&actual};
    std::vector<int> tokens;
    LLMSamplingBlock(&model, nullptr, nullptr, nullptr, 1e-5f, 1, true, {1}, past,
                    {config}, last, &results, tokens, &logits);
    MultiCudaSetPersistentAsyncDispatch(previous);
    for (int dev : {0, 1}) {
        FastllmCudaSetDevice(dev); cudaDeviceSynchronize(); cudaFreeHost(host[dev]);
    }
    if (actual.size() != vocab || tokens != std::vector<int>{vocab - 1})
        throw std::runtime_error("TP gather sampled before the remote producer completed");
    for (int i = 0; i < vocab; ++i)
        if (actual[i] != i + 1)
            throw std::runtime_error("TP gather read an unfinished remote logits shard");
    std::cout << "PASS delayed remote logits gather: all 256 values and selected token match\n";
    return 1;
}
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) return 77;
    try {
        Executor &exec = *((Executor *)GetExecutor());
        int checks = SharedSwigluChecks() + AddHcPostChecks() + HandoffChecks() + DelayedLogitsGatherChecks();
        for (auto type : {DataType::BFLOAT16, DataType::FLOAT16, DataType::FLOAT32}) {
            Data tpOutput;
            for (int tokens : {1, 5, 33, 1}) {
                const int channels = 5120;
                ApplyDeviceMap({{"cuda:0", 1}}, 0, 1);
                auto input = Make(type, {1, tokens, channels}, 3 + tokens);
                auto residual = Make(type, {1, tokens, 4, channels}, 7 + tokens);
                auto post = Make(DataType::FLOAT32, {1, tokens, 4}, 11);
                auto comb = Make(DataType::FLOAT32, {1, tokens, 4, 4}, 13);
                Data output;
                exec.Run("DeepSeekV41HcPost", {{"input", &input}, {"residual", &residual},
                    {"post", &post}, {"comb", &comb}, {"output", &output}}, {}, {});
                auto expected = Bytes(output);
                auto tpInput = Make(type, {1, tokens, channels}, 3 + tokens);
                auto tpResidual = Make(type, {1, tokens, 4, channels}, 7 + tokens);
                auto tpPost = Make(DataType::FLOAT32, {1, tokens, 4}, 11);
                auto tpComb = Make(DataType::FLOAT32, {1, tokens, 4, 4}, 13);
                ApplyDeviceMap({{"multicuda:0,1", 1}}, 0, 1);
                exec.Run("DeepSeekV41HcPost", {{"input", &tpInput}, {"residual", &tpResidual},
                    {"post", &tpPost}, {"comb", &tpComb}, {"output", &tpOutput}}, {}, {});
                if (!tpOutput.IsTensorParallelReplicated() || tpOutput.dims != output.dims || tpOutput.dataType != type)
                    throw std::runtime_error("TP output metadata differs from CUDA reference");
                ++checks;
                for (int device : {0, 1}) {
                    auto local = tpOutput.multiDeviceDatas.at(device);
                    if (local->dims != output.dims || Bytes(*local) != expected)
                        throw std::runtime_error("TP rank differs from CUDA reference");
                    ++checks;
                }
            }
        }
        std::cout << "PASS HC post and shared expert activation: " << checks << " checks; BF16/FP16/FP32; token lengths 1,5,33,1; both ranks bitwise match CUDA\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
