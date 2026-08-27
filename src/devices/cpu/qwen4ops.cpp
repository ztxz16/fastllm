#include "devices/cpu/cpudevice.h"
#include "devices/cpu/alivethreadpool.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <vector>

namespace fastllm {
    namespace {
        class Qwen4RangeTask final : public MultiThreadBaseOp {
        public:
            Qwen4RangeTask(int start, int end,
                           const std::function<void(int, int)> &function)
                : start(start), end(end), function(function) {}

            void Run() override {
                function(start, end);
            }

        private:
            int start;
            int end;
            std::function<void(int, int)> function;
        };

        void Qwen4ParallelFor(int count,
                              const std::function<void(int, int)> &function) {
            if (count <= 0) {
                return;
            }
            AliveThreadPool *pool = GetAlivePool();
            const int firstThread = pool->curActivateThreadInterval.first;
            const int availableThreads = std::max(
                1, pool->curActivateThreadInterval.second - firstThread);
            const int threadCount = std::min(count, availableThreads);
            if (threadCount == 1) {
                function(0, count);
                return;
            }

            std::vector<Qwen4RangeTask*> tasks;
            tasks.reserve(threadCount);
            for (int thread = 0; thread < threadCount; thread++) {
                const int start = (int)((int64_t)count * thread / threadCount);
                const int end = (int)((int64_t)count * (thread + 1) /
                                      threadCount);
                tasks.push_back(new Qwen4RangeTask(start, end, function));
                pool->PushOp(firstThread + thread, tasks.back());
            }
            for (int thread = 0; thread < threadCount; thread++) {
                pool->Wait(firstThread + thread);
                delete tasks[thread];
            }
        }

        bool Qwen4ActivationType(DataType type) {
            return type == DataType::FLOAT32 ||
                   type == DataType::FLOAT16 ||
                   type == DataType::BFLOAT16;
        }

        float Qwen4LoadCpu(const uint8_t *data, DataType type,
                           uint64_t index) {
            if (type == DataType::FLOAT32) {
                return ((const float*)data)[index];
            }
            if (type == DataType::FLOAT16) {
                return half_to_float(((const uint16_t*)data)[index]);
            }
            return BFloat16BitsToFloat32(((const uint16_t*)data)[index]);
        }

        void Qwen4StoreCpu(uint8_t *data, DataType type,
                           uint64_t index, float value) {
            if (type == DataType::FLOAT32) {
                ((float*)data)[index] = value;
            } else if (type == DataType::FLOAT16) {
                ((uint16_t*)data)[index] = float_to_half(value);
            } else {
                ((uint16_t*)data)[index] = Float32ToBFloat16RNEBits(value);
            }
        }

        float Qwen4RoundCpu(float value, DataType type) {
            if (type == DataType::FLOAT16) {
                return half_to_float(float_to_half(value));
            }
            if (type == DataType::BFLOAT16) {
                return RoundFloat32ToBFloat16RNE(value);
            }
            return value;
        }

        float Qwen4SigmoidCpu(float value) {
            // Keep the same evaluation order as FastLLM's Sigmoid op.  The
            // algebraically equivalent negative-input branch changes the last
            // few bits and those differences accumulate through 36 layers.
            return 1.0f / (1.0f + std::exp(-value));
        }

        int Qwen4Groups(const IntDict &intParams) {
            auto it = intParams.find("groups");
            return it == intParams.end() ? 1 : it->second;
        }

        void Qwen4AssertCpuTensor(const Data &data,
                                  const std::string &name) {
            AssertInFastLLM(data.dataDevice == DataDevice::CPU &&
                            data.cpuData != nullptr &&
                            Qwen4ActivationType(data.dataType),
                            name + " must be a CPU float32/float16/bfloat16 tensor.\n");
        }
    }

    void CpuQwen4GroupedRMSNormOp::Run(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *datas.find("input")->second;
        Data &weight = *datas.find("weight")->second;
        Data &output = *datas.find("output")->second;
        const int groups = Qwen4Groups(intParams);
        const float eps = floatParams.find("eps") == floatParams.end()
            ? 1e-6f : floatParams.find("eps")->second;

        AssertInFastLLM(!input.dims.empty() && groups > 0 &&
                        input.dims.back() % groups == 0,
                        "Qwen4GroupedRMSNorm received an invalid shape or group count.\n");
        Qwen4AssertCpuTensor(input, "Qwen4GroupedRMSNorm input");
        Qwen4AssertCpuTensor(weight, "Qwen4GroupedRMSNorm weight");
        AssertInFastLLM(weight.Count(0) == (uint64_t)input.dims.back(),
                        "Qwen4GroupedRMSNorm weight shape mismatch.\n");
        output.Allocate(false);

        const int channels = input.dims.back();
        const int groupChannels = channels / groups;
        const int rows = (int)(input.Count(0) / channels);
        const DataType type = input.dataType;
        Qwen4ParallelFor(rows * groups, [&](int start, int end) {
            for (int item = start; item < end; item++) {
                const int row = item / groups;
                const int group = item % groups;
                const uint64_t base = (uint64_t)row * channels +
                                      (uint64_t)group * groupChannels;
                float sum = 0.0f;
                for (int channel = 0; channel < groupChannels; channel++) {
                    const float value = Qwen4LoadCpu(
                        input.cpuData, type, base + channel);
                    sum += value * value;
                }
                const float scale = 1.0f /
                    std::sqrt(sum / groupChannels + eps);
                for (int channel = 0; channel < groupChannels; channel++) {
                    const uint64_t index = base + channel;
                    const float value = Qwen4LoadCpu(input.cpuData, type, index);
                    const float normWeight = Qwen4LoadCpu(
                        weight.cpuData, weight.dataType,
                        (uint64_t)group * groupChannels + channel);
                    Qwen4StoreCpu(output.cpuData, type, index,
                                  value * scale * normWeight);
                }
            }
        });
    }

    void CpuQwen4HyperMixOp::Reshape(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *datas.find("input")->second;
        Data &mixLogits = *datas.find("mixLogits")->second;
        Data &output = *datas.find("output")->second;
        const int groups = Qwen4Groups(intParams);
        AssertInFastLLM(!input.dims.empty() && input.dims == mixLogits.dims &&
                        groups > 0 && input.dims.back() % groups == 0,
                        "Qwen4HyperMix received incompatible shapes.\n");
        std::vector<int> dims = input.dims;
        dims.back() /= groups;
        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void CpuQwen4HyperMixOp::Run(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &normalized = *datas.find("input")->second;
        Data &mixLogits = *datas.find("mixLogits")->second;
        Data &output = *datas.find("output")->second;
        const int groups = Qwen4Groups(intParams);
        Qwen4AssertCpuTensor(normalized, "Qwen4HyperMix input");
        Qwen4AssertCpuTensor(mixLogits, "Qwen4HyperMix logits");
        output.Allocate(false);

        const int channels = normalized.dims.back();
        const int outputChannels = channels / groups;
        const int rows = (int)(normalized.Count(0) / channels);
        const DataType type = normalized.dataType;
        const DataType logitsType = mixLogits.dataType;
        Qwen4ParallelFor(rows, [&](int start, int end) {
            for (int row = start; row < end; row++) {
                const uint64_t inputBase = (uint64_t)row * channels;
                const uint64_t outputBase = (uint64_t)row * outputChannels;
                for (int channel = 0; channel < outputChannels; channel++) {
                    float sum = 0.0f;
                    for (int group = 0; group < groups; group++) {
                        const uint64_t index = inputBase +
                            (uint64_t)group * outputChannels + channel;
                        const float nativeGate = Qwen4RoundCpu(
                            Qwen4SigmoidCpu(Qwen4LoadCpu(
                                mixLogits.cpuData, logitsType, index)),
                            logitsType);
                        const float gate = Qwen4RoundCpu(nativeGate, type);
                        const float product = Qwen4RoundCpu(
                            Qwen4LoadCpu(normalized.cpuData, type, index) * gate,
                            type);
                        sum = group == 0 ? product :
                            Qwen4RoundCpu(sum + product, type);
                    }
                    Qwen4StoreCpu(output.cpuData, type, outputBase + channel,
                                  Qwen4RoundCpu(sum / groups, type));
                }
            }
        });
    }

    void CpuQwen4HyperInjectOp::Run(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *datas.find("input")->second;
        Data &output = *datas.find("output")->second;
        const int groups = Qwen4Groups(intParams);
        Qwen4AssertCpuTensor(input, "Qwen4HyperInject input");
        AssertInFastLLM(groups > 0 && !input.dims.empty() &&
                        input.dims.back() == groups,
                        "Qwen4HyperInject received an invalid shape.\n");
        output.Allocate(false);
        const DataType type = input.dataType;
        const int count = (int)input.Count(0);
        Qwen4ParallelFor(count, [&](int start, int end) {
            for (int index = start; index < end; index++) {
                const float scaled = Qwen4RoundCpu(
                    Qwen4LoadCpu(input.cpuData, type, index) / groups, type);
                const float gate = Qwen4RoundCpu(
                    Qwen4SigmoidCpu(scaled), type);
                Qwen4StoreCpu(output.cpuData, type, index,
                              Qwen4RoundCpu(gate * 2.0f, type));
            }
        });
    }

    void CpuQwen4HyperCombineOp::Run(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &hyperInput = *datas.find("input")->second;
        Data &blockOutput = *datas.find("blockOutput")->second;
        Data &injection = *datas.find("injection")->second;
        Data &output = *datas.find("output")->second;
        const int groups = Qwen4Groups(intParams);
        Qwen4AssertCpuTensor(hyperInput, "Qwen4HyperCombine hyper input");
        Qwen4AssertCpuTensor(blockOutput, "Qwen4HyperCombine block output");
        Qwen4AssertCpuTensor(injection, "Qwen4HyperCombine injection");
        AssertInFastLLM(!hyperInput.dims.empty() && groups > 0 &&
                        hyperInput.dims.back() % groups == 0,
                        "Qwen4HyperCombine received incompatible tensors.\n");

        const int channels = hyperInput.dims.back();
        const int blockChannels = channels / groups;
        const int rows = (int)(hyperInput.Count(0) / channels);
        AssertInFastLLM(blockOutput.Count(0) ==
                            (uint64_t)rows * blockChannels &&
                        injection.Count(0) == (uint64_t)rows * groups,
                        "Qwen4HyperCombine shape mismatch.\n");
        output.Allocate(false);
        const DataType type = hyperInput.dataType;
        const DataType blockType = blockOutput.dataType;
        const DataType injectionType = injection.dataType;
        Qwen4ParallelFor(rows, [&](int start, int end) {
            for (int row = start; row < end; row++) {
                const uint64_t hyperBase = (uint64_t)row * channels;
                const uint64_t blockBase = (uint64_t)row * blockChannels;
                const uint64_t injectionBase = (uint64_t)row * groups;
                for (int group = 0; group < groups; group++) {
                    const float scale = Qwen4RoundCpu(Qwen4LoadCpu(
                        injection.cpuData, injectionType,
                        injectionBase + group), type);
                    for (int channel = 0; channel < blockChannels; channel++) {
                        const uint64_t outputIndex = hyperBase +
                            (uint64_t)group * blockChannels + channel;
                        const float injected = Qwen4RoundCpu(
                            Qwen4RoundCpu(Qwen4LoadCpu(
                                blockOutput.cpuData, blockType,
                                blockBase + channel), type) * scale, type);
                        const float combined = Qwen4RoundCpu(
                            Qwen4LoadCpu(hyperInput.cpuData, type,
                                         outputIndex) + injected, type);
                        Qwen4StoreCpu(output.cpuData, type, outputIndex,
                                      combined);
                    }
                }
            }
        });
    }

    void CpuQwen4GatedDeltaRuleDecodeOp::Reshape(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &qkv = *datas.find("input")->second;
        Data &output = *datas.find("output")->second;
        const int valueHeads = intParams.find("valueHeads")->second;
        const int valueDim = intParams.find("valueDim")->second;
        AssertInFastLLM(qkv.dims.size() == 3 && qkv.dims[1] == 1 &&
                        valueHeads > 0 && valueDim > 0,
                        "Qwen4GatedDeltaRuleDecode expects one decode token.\n");
        output.dataType = DataType::FLOAT32;
        output.Resize({qkv.dims[0], 1, valueHeads * valueDim});
    }

    void CpuQwen4GatedDeltaRuleDecodeOp::Run(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &qkv = *datas.find("input")->second;
        Data &alpha = *datas.find("alpha")->second;
        Data &beta = *datas.find("beta")->second;
        Data &aLog = *datas.find("aLog")->second;
        Data &dtBias = *datas.find("dtBias")->second;
        Data &state = *datas.find("state")->second;
        Data &output = *datas.find("output")->second;
        const int keyHeads = intParams.find("keyHeads")->second;
        const int valueHeads = intParams.find("valueHeads")->second;
        const int keyDim = intParams.find("keyDim")->second;
        const int valueDim = intParams.find("valueDim")->second;
        const float recurrentEps =
            floatParams.find("recurrentEps")->second;
        const int batch = qkv.dims[0];
        const int qkvChannels = 2 * keyHeads * keyDim +
                                valueHeads * valueDim;

        Qwen4AssertCpuTensor(qkv, "Qwen4GatedDeltaRuleDecode qkv");
        Qwen4AssertCpuTensor(alpha, "Qwen4GatedDeltaRuleDecode alpha");
        Qwen4AssertCpuTensor(beta, "Qwen4GatedDeltaRuleDecode beta");
        Qwen4AssertCpuTensor(aLog, "Qwen4GatedDeltaRuleDecode A_log");
        Qwen4AssertCpuTensor(dtBias, "Qwen4GatedDeltaRuleDecode dt_bias");
        Qwen4AssertCpuTensor(state, "Qwen4GatedDeltaRuleDecode state");
        AssertInFastLLM(qkv.dataType == DataType::FLOAT32 &&
                        state.dataType == DataType::FLOAT32 &&
                        aLog.dataType == DataType::FLOAT32 &&
                        dtBias.dataType == DataType::FLOAT32 &&
                        output.dataType == DataType::FLOAT32,
                        "Qwen4GatedDeltaRuleDecode keeps qkv, state and parameters in float32.\n");
        AssertInFastLLM(keyHeads > 0 && valueHeads > 0 &&
                        valueHeads % keyHeads == 0 &&
                        keyDim > 0 && valueDim > 0 &&
                        qkv.dims.size() == 3 && qkv.dims[1] == 1 &&
                        qkv.dims[2] == qkvChannels &&
                        alpha.Count(0) == (uint64_t)batch * valueHeads &&
                        beta.Count(0) == (uint64_t)batch * valueHeads &&
                        aLog.Count(0) == (uint64_t)valueHeads &&
                        dtBias.Count(0) == (uint64_t)valueHeads &&
                        state.dims == std::vector<int>({batch, valueHeads,
                                                       keyDim, valueDim}),
                        "Qwen4GatedDeltaRuleDecode shape mismatch.\n");
        output.Allocate(false);

        const float *qkvData = (const float*)qkv.cpuData;
        const float *aLogData = (const float*)aLog.cpuData;
        const float *dtBiasData = (const float*)dtBias.cpuData;
        float *stateData = (float*)state.cpuData;
        float *outputData = (float*)output.cpuData;
        const int repeat = valueHeads / keyHeads;
        const float inverseHead = 1.0f / std::sqrt((float)keyDim);

        Qwen4ParallelFor(batch * valueHeads, [&](int start, int end) {
            std::vector<float> key(keyDim);
            std::vector<float> query(keyDim);
            for (int item = start; item < end; item++) {
                const int batchIndex = item / valueHeads;
                const int valueHead = item % valueHeads;
                const int keyHead = valueHead / repeat;
                const uint64_t qkvBase =
                    (uint64_t)batchIndex * qkvChannels;
                const float *queryRaw = qkvData + qkvBase +
                    (uint64_t)keyHead * keyDim;
                const float *keyRaw = qkvData + qkvBase +
                    (uint64_t)(keyHeads + keyHead) * keyDim;
                const float *value = qkvData + qkvBase +
                    (uint64_t)2 * keyHeads * keyDim +
                    (uint64_t)valueHead * valueDim;

                float querySquares = 0.0f;
                float keySquares = 0.0f;
                for (int channel = 0; channel < keyDim; channel++) {
                    querySquares += queryRaw[channel] * queryRaw[channel];
                    keySquares += keyRaw[channel] * keyRaw[channel];
                }
                const float queryScale = 1.0f / std::sqrt(
                    querySquares / keyDim + recurrentEps);
                const float keyScale = 1.0f / std::sqrt(
                    keySquares / keyDim + recurrentEps);
                for (int channel = 0; channel < keyDim; channel++) {
                    query[channel] = queryRaw[channel] * queryScale *
                                     inverseHead * inverseHead;
                    key[channel] = keyRaw[channel] * keyScale * inverseHead;
                }

                const uint64_t gateIndex =
                    (uint64_t)batchIndex * valueHeads + valueHead;
                const float betaValue = Qwen4SigmoidCpu(Qwen4LoadCpu(
                    beta.cpuData, beta.dataType, gateIndex));
                const float alphaValue = Qwen4LoadCpu(
                    alpha.cpuData, alpha.dataType, gateIndex);
                const float biasedAlpha = alphaValue +
                    dtBiasData[valueHead];
                const float softplus = biasedAlpha > 20.0f
                    ? biasedAlpha : std::log1p(std::exp(biasedAlpha));
                const float logDecay =
                    -std::exp(aLogData[valueHead]) * softplus;
                const float decay = std::exp(logDecay);
                float *headState = stateData +
                    (uint64_t)item * keyDim * valueDim;

                for (int valueChannel = 0;
                     valueChannel < valueDim; valueChannel++) {
                    float memory = 0.0f;
                    for (int keyChannel = 0;
                         keyChannel < keyDim; keyChannel++) {
                        const uint64_t stateIndex =
                            (uint64_t)keyChannel * valueDim + valueChannel;
                        const float scaled = headState[stateIndex] * decay;
                        headState[stateIndex] = scaled;
                        memory += scaled * key[keyChannel];
                    }
                    const float delta =
                        (value[valueChannel] - memory) * betaValue;
                    float core = 0.0f;
                    for (int keyChannel = 0;
                         keyChannel < keyDim; keyChannel++) {
                        const uint64_t stateIndex =
                            (uint64_t)keyChannel * valueDim + valueChannel;
                        const float updated = headState[stateIndex] +
                                              key[keyChannel] * delta;
                        headState[stateIndex] = updated;
                        core += updated * query[keyChannel];
                    }
                    outputData[(uint64_t)item * valueDim + valueChannel] =
                        core;
                }
            }
        });
    }
}
