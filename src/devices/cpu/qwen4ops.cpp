#include "devices/cpu/cpudevice.h"
#include "devices/cpu/alivethreadpool.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <utility>
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

        uint16_t Qwen4FloatToHalfRz(float value) {
            uint16_t result = float_to_half(value);
            if (!std::isfinite(value)) {
                return result;
            }
            const float rounded = half_to_float(result);
            // float_to_half uses nearest rounding. Move one representable
            // value toward zero only when it rounded away from the input.
            // This matches FastllmCudaFloat2HalfKernel's __float2half_rz.
            if (std::fabs(rounded) > std::fabs(value) &&
                (result & 0x7fff) != 0) {
                result--;
            }
            return result;
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

    void CpuQwen4PLEGateOp::Reshape(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &key = *datas.find("key")->second;
        Data &query = *datas.find("query")->second;
        Data &value = *datas.find("value")->second;
        Data &output = *datas.find("output")->second;
        const int groups = Qwen4Groups(intParams);
        AssertInFastLLM(!key.dims.empty() && key.dims == query.dims &&
                        groups > 0 && key.dims.back() % groups == 0,
                        "Qwen4PLEGate received incompatible key/query shapes.\n");
        const int channels = key.dims.back();
        const int rows = (int)(key.Count(0) / channels);
        AssertInFastLLM(value.Count(0) ==
                            (uint64_t)rows * channels / groups,
                        "Qwen4PLEGate received an incompatible value shape.\n");
        output.dataType = DataType::FLOAT32;
        output.Resize(key.dims);
    }

    void CpuQwen4PLEGateOp::Run(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &key = *datas.find("key")->second;
        Data &query = *datas.find("query")->second;
        Data &value = *datas.find("value")->second;
        Data &output = *datas.find("output")->second;
        const int groups = Qwen4Groups(intParams);
        Qwen4AssertCpuTensor(key, "Qwen4PLEGate key");
        Qwen4AssertCpuTensor(query, "Qwen4PLEGate query");
        Qwen4AssertCpuTensor(value, "Qwen4PLEGate value");
        output.Allocate(false);

        const int channels = key.dims.back();
        const int groupChannels = channels / groups;
        const int rows = (int)(key.Count(0) / channels);
        const float inverseSqrtHidden =
            1.0f / std::sqrt((float)groupChannels);
        float *outputValues = reinterpret_cast<float *>(output.cpuData);
        Qwen4ParallelFor(rows * groups, [&](int start, int end) {
            for (int item = start; item < end; item++) {
                const int row = item / groups;
                const int group = item % groups;
                const uint64_t base = (uint64_t)row * channels +
                                      (uint64_t)group * groupChannels;
                float dot = 0.0f;
                for (int channel = 0; channel < groupChannels; channel++) {
                    dot += Qwen4LoadCpu(key.cpuData, key.dataType,
                                       base + channel) *
                           Qwen4LoadCpu(query.cpuData, query.dataType,
                                       base + channel);
                }
                float gate = dot * inverseSqrtHidden;
                if (gate != 0.0f) {
                    gate = std::copysign(
                        std::sqrt(std::max(std::fabs(gate), 1e-6f)), gate);
                }
                const float probability = Qwen4SigmoidCpu(gate);
                const uint64_t valueBase =
                    (uint64_t)row * groupChannels;
                for (int channel = 0; channel < groupChannels; channel++) {
                    outputValues[base + channel] = probability *
                        Qwen4LoadCpu(value.cpuData, value.dataType,
                                     valueBase + channel);
                }
            }
        });
    }

    void CpuQwen4PLECausalConvOp::Reshape(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *datas.find("input")->second;
        Data &gated = *datas.find("gated")->second;
        Data &history = *datas.find("history")->second;
        Data &output = *datas.find("output")->second;
        Data &newHistory = *datas.find("newHistory")->second;
        const int kernel = intParams.find("kernel")->second;
        const int dilation = intParams.find("dilation")->second;
        const int historyLength = (kernel - 1) * dilation;
        AssertInFastLLM(input.dims.size() == 3 && input.dims[0] == 1 &&
                        input.dims == gated.dims && kernel > 1 &&
                        dilation > 0 && history.dims ==
                            std::vector<int>({historyLength,
                                              input.dims.back()}),
                        "Qwen4PLECausalConv received incompatible shapes.\n");
        output.dataType = DataType::FLOAT32;
        output.Resize(input.dims);
        newHistory.dataType = DataType::FLOAT32;
        newHistory.Resize(history.dims);
    }

    void CpuQwen4PLECausalConvOp::Run(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *datas.find("input")->second;
        Data &gated = *datas.find("gated")->second;
        Data &weight = *datas.find("weight")->second;
        Data &history = *datas.find("history")->second;
        Data &output = *datas.find("output")->second;
        Data &newHistory = *datas.find("newHistory")->second;
        const int kernel = intParams.find("kernel")->second;
        const int dilation = intParams.find("dilation")->second;
        const int historyLength = (kernel - 1) * dilation;
        const int sequence = input.dims[1];
        const int channels = input.dims[2];
        Qwen4AssertCpuTensor(input, "Qwen4PLECausalConv input");
        Qwen4AssertCpuTensor(gated, "Qwen4PLECausalConv gated input");
        Qwen4AssertCpuTensor(weight, "Qwen4PLECausalConv weight");
        Qwen4AssertCpuTensor(history, "Qwen4PLECausalConv history");
        AssertInFastLLM(weight.dataType == DataType::FLOAT32 &&
                        history.dataType == DataType::FLOAT32 &&
                        weight.Count(0) == (uint64_t)channels * kernel,
                        "Qwen4PLECausalConv requires float32 weight/history.\n");
        output.Allocate(false);
        newHistory.Allocate(false);
        const float *weightValues =
            reinterpret_cast<const float *>(weight.cpuData);
        const float *historyValues =
            reinterpret_cast<const float *>(history.cpuData);
        float *outputValues = reinterpret_cast<float *>(output.cpuData);
        float *nextHistory =
            reinterpret_cast<float *>(newHistory.cpuData);

        Qwen4ParallelFor(sequence * channels, [&](int start, int end) {
            for (int item = start; item < end; item++) {
                const int token = item / channels;
                const int channel = item % channels;
                float sum = 0.0f;
                for (int tap = 0; tap < kernel; tap++) {
                    const int relative =
                        token - historyLength + tap * dilation;
                    const float sample = relative >= 0
                        ? Qwen4LoadCpu(
                              input.cpuData, input.dataType,
                              (uint64_t)relative * channels + channel)
                        : historyValues[
                              (uint64_t)(historyLength + relative) *
                                  channels + channel];
                    sum += sample *
                           weightValues[(uint64_t)channel * kernel + tap];
                }
                const float conv = sum / (1.0f + std::exp(-sum));
                outputValues[item] = Qwen4LoadCpu(
                    gated.cpuData, gated.dataType, item) + conv;
            }
        });

        Qwen4ParallelFor(historyLength * channels,
                         [&](int start, int end) {
            for (int item = start; item < end; item++) {
                const int historyIndex = item / channels;
                const int channel = item % channels;
                const int sourceToken =
                    sequence - historyLength + historyIndex;
                nextHistory[item] = sourceToken >= 0
                    ? Qwen4LoadCpu(
                          input.cpuData, input.dataType,
                          (uint64_t)sourceToken * channels + channel)
                    : historyValues[
                          (uint64_t)(historyLength + sourceToken) *
                              channels + channel];
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
        auto weightIt = datas.find("weight");
        const bool projected = weightIt != datas.end();
        bool compatible = !input.dims.empty() && groups > 0 &&
                          input.dims.back() % groups == 0;
        if (projected) {
            Data &weight = *weightIt->second;
            compatible = compatible && !mixLogits.dims.empty() &&
                mixLogits.dims.back() > 0 &&
                weight.dims.size() == 2 &&
                mixLogits.Count(0) / mixLogits.dims.back() ==
                    input.Count(0) / input.dims.back() &&
                mixLogits.dims.back() == weight.dims[1] &&
                input.dims.back() == weight.dims[0];
            weight.weightType = WeightType::LINEAR;
        } else {
            compatible = compatible && input.dims == mixLogits.dims;
        }
        AssertInFastLLM(compatible,
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
        Data projectedLogits;
        Data *effectiveLogits = &mixLogits;
        auto weightIt = datas.find("weight");
        if (weightIt != datas.end()) {
            Data &weight = *weightIt->second;
            Qwen4AssertCpuTensor(mixLogits, "Qwen4HyperMix low rank");
            Qwen4AssertCpuTensor(weight, "Qwen4HyperMix up weight");
            DoCpuLinearReshape(mixLogits, weight, projectedLogits);
            Data emptyBias;
            DoCpuLinear(mixLogits, weight, emptyBias, projectedLogits);
            effectiveLogits = &projectedLogits;
        }
        Qwen4AssertCpuTensor(*effectiveLogits, "Qwen4HyperMix logits");
        output.Allocate(false);

        const int channels = normalized.dims.back();
        const int outputChannels = channels / groups;
        const int rows = (int)(normalized.Count(0) / channels);
        const DataType type = normalized.dataType;
        const DataType logitsType = effectiveLogits->dataType;
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
                                effectiveLogits->cpuData,
                                logitsType, index)),
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

    void CpuQwen4HyperProjectOp::Reshape(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *datas.find("input")->second;
        Data &downWeight = *datas.find("downWeight")->second;
        Data &injectionWeight = *datas.find("injectionWeight")->second;
        Data &activated = *datas.find("output")->second;
        Data &injection = *datas.find("injection")->second;
        const int groups = Qwen4Groups(intParams);
        AssertInFastLLM(
            !input.dims.empty() && downWeight.dims.size() == 2 &&
            injectionWeight.dims.size() == 2 && groups > 0 &&
            input.dims.back() == downWeight.dims[1] &&
            input.dims.back() == injectionWeight.dims[1] &&
            injectionWeight.dims[0] == groups,
            "Qwen4HyperProject received incompatible shapes.\n");

        downWeight.weightType = WeightType::LINEAR;
        injectionWeight.weightType = WeightType::LINEAR;
        std::vector<int> activatedDims = input.dims;
        activatedDims.back() = downWeight.dims[0];
        std::vector<int> injectionDims = input.dims;
        injectionDims.back() = injectionWeight.dims[0];
        auto outputTypeIt = intParams.find("outputType");
        const DataType outputType = outputTypeIt == intParams.end()
            ? input.dataType : (DataType)outputTypeIt->second;
        AssertInFastLLM(Qwen4ActivationType(outputType),
                        "Qwen4HyperProject output type is invalid.\n");
        activated.dataType = outputType;
        injection.dataType = outputType;
        activated.UpdateUnitSize();
        injection.UpdateUnitSize();
        activated.Resize(activatedDims);
        injection.Resize(injectionDims);
    }

    void CpuQwen4HyperProjectOp::Run(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *datas.find("input")->second;
        Data &downWeight = *datas.find("downWeight")->second;
        Data &injectionWeight = *datas.find("injectionWeight")->second;
        Data &activated = *datas.find("output")->second;
        Data &injection = *datas.find("injection")->second;
        const int groups = Qwen4Groups(intParams);
        Qwen4AssertCpuTensor(input, "Qwen4HyperProject input");
        Qwen4AssertCpuTensor(downWeight, "Qwen4HyperProject down weight");
        Qwen4AssertCpuTensor(
            injectionWeight, "Qwen4HyperProject injection weight");

        Data temporaryActivated(input.dataType);
        Data temporaryInjection(input.dataType);
        Data *linearActivated = &activated;
        Data *linearInjection = &injection;
        if (input.dataType != activated.dataType) {
            temporaryActivated.Resize(activated.dims);
            temporaryInjection.Resize(injection.dims);
            linearActivated = &temporaryActivated;
            linearInjection = &temporaryInjection;
        }
        Data emptyBias;
        DoCpuLinear(input, downWeight, emptyBias, *linearActivated);
        DoCpuLinear(input, injectionWeight, emptyBias, *linearInjection);
        if (linearActivated != &activated) {
            activated.Allocate(false);
            injection.Allocate(false);
            const int activatedCount = (int)activated.Count(0);
            const int injectionCount = (int)injection.Count(0);
            Qwen4ParallelFor(activatedCount, [&](int start, int end) {
                for (int index = start; index < end; index++) {
                    Qwen4StoreCpu(activated.cpuData, activated.dataType,
                                  index, Qwen4LoadCpu(
                                      linearActivated->cpuData,
                                      linearActivated->dataType, index));
                }
            });
            Qwen4ParallelFor(injectionCount, [&](int start, int end) {
                for (int index = start; index < end; index++) {
                    Qwen4StoreCpu(injection.cpuData, injection.dataType,
                                  index, Qwen4LoadCpu(
                                      linearInjection->cpuData,
                                      linearInjection->dataType, index));
                }
            });
        }

        const DataType type = activated.dataType;
        const int activatedCount = (int)activated.Count(0);
        const float scale = 1.0f / groups;
        Qwen4ParallelFor(activatedCount, [&](int start, int end) {
            for (int index = start; index < end; index++) {
                const float scaled = Qwen4RoundCpu(
                    Qwen4LoadCpu(activated.cpuData, type, index) * scale,
                    type);
                const float value = type == DataType::BFLOAT16
                    ? scaled / (1.0f + std::exp(-scaled))
                    : scaled / (1.0 + expf(-scaled));
                Qwen4StoreCpu(activated.cpuData, type, index, value);
            }
        });

        const int injectionCount = (int)injection.Count(0);
        Qwen4ParallelFor(injectionCount, [&](int start, int end) {
            for (int index = start; index < end; index++) {
                const float scaled = Qwen4RoundCpu(
                    Qwen4LoadCpu(injection.cpuData, type, index) / groups,
                    type);
                const float gate = Qwen4RoundCpu(
                    Qwen4SigmoidCpu(scaled), type);
                Qwen4StoreCpu(injection.cpuData, type, index,
                              Qwen4RoundCpu(gate * 2.0f, type));
            }
        });
    }

    void CpuQwen4HyperPrepareOp::Reshape(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *datas.find("input")->second;
        Data &activated = *datas.find("output")->second;
        const int groups = Qwen4Groups(intParams);
        AssertInFastLLM(!input.dims.empty() && groups > 0,
                        "Qwen4HyperPrepare received an invalid shape.\n");
        activated.dataType = input.dataType;
        activated.Resize(input.dims);
    }

    void CpuQwen4HyperPrepareOp::Run(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *datas.find("input")->second;
        Data &activated = *datas.find("output")->second;
        const int groups = Qwen4Groups(intParams);
        Qwen4AssertCpuTensor(input, "Qwen4HyperPrepare input");
        AssertInFastLLM(!input.dims.empty() && groups > 0,
                        "Qwen4HyperPrepare received an invalid shape.\n");
        activated.Allocate(false);

        const DataType type = input.dataType;
        const int count = (int)input.Count(0);
        const float scale = 1.0f / groups;
        Qwen4ParallelFor(count, [&](int start, int end) {
            for (int index = start; index < end; index++) {
                const float scaled = Qwen4RoundCpu(
                    Qwen4LoadCpu(input.cpuData, type, index) * scale,
                    type);
                const float activatedValue = type == DataType::BFLOAT16
                    ? scaled / (1.0f + std::exp(-scaled))
                    : scaled / (1.0 + expf(-scaled));
                Qwen4StoreCpu(
                    activated.cpuData, type, index, activatedValue);
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

    void CpuQwen4HyperCombineRMSNormOp::Reshape(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *datas.find("input")->second;
        Data &output = *datas.find("output")->second;
        Data &normalized = *datas.find("normalized")->second;
        output.dataType = input.dataType;
        output.UpdateUnitSize();
        output.Resize(input.dims);
        normalized.dataType = input.dataType;
        normalized.UpdateUnitSize();
        normalized.Resize(input.dims);
        auto storageIt = datas.find("normalizedStorage");
        if (storageIt != datas.end()) {
            auto storageTypeIt = intParams.find("normalizedStorageType");
            const DataType storageType = storageTypeIt == intParams.end()
                ? DataType::FLOAT16 : (DataType)storageTypeIt->second;
            AssertInFastLLM(
                Qwen4ActivationType(storageType),
                "Qwen4HyperCombineRMSNorm storage type is invalid.\n");
            Data &storage = *storageIt->second;
            storage.dataType = storageType;
            storage.UpdateUnitSize();
            storage.Resize(input.dims);
        }
    }

    void CpuQwen4HyperCombineRMSNormOp::Run(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &hyperInput = *datas.find("input")->second;
        Data &blockOutput = *datas.find("blockOutput")->second;
        Data &injection = *datas.find("injection")->second;
        Data &weight = *datas.find("weight")->second;
        Data &output = *datas.find("output")->second;
        Data &normalized = *datas.find("normalized")->second;
        auto storageIt = datas.find("normalizedStorage");
        Data *normalizedStorage = storageIt == datas.end()
            ? nullptr : storageIt->second;
        const int groups = Qwen4Groups(intParams);
        const float eps = floatParams.find("eps") == floatParams.end()
            ? 1e-6f : floatParams.find("eps")->second;
        Qwen4AssertCpuTensor(hyperInput,
                             "Qwen4HyperCombineRMSNorm hyper input");
        Qwen4AssertCpuTensor(blockOutput,
                             "Qwen4HyperCombineRMSNorm block output");
        Qwen4AssertCpuTensor(injection,
                             "Qwen4HyperCombineRMSNorm injection");
        Qwen4AssertCpuTensor(weight,
                             "Qwen4HyperCombineRMSNorm weight");
        AssertInFastLLM(!hyperInput.dims.empty() && groups > 0 &&
                        hyperInput.dims.back() % groups == 0,
                        "Qwen4HyperCombineRMSNorm received incompatible tensors.\n");

        const int channels = hyperInput.dims.back();
        const int groupChannels = channels / groups;
        const int rows = (int)(hyperInput.Count(0) / channels);
        AssertInFastLLM(
            blockOutput.Count(0) == (uint64_t)rows * groupChannels &&
            injection.Count(0) == (uint64_t)rows * groups &&
            weight.Count(0) == (uint64_t)channels,
            "Qwen4HyperCombineRMSNorm shape mismatch.\n");
        output.Allocate(false);
        normalized.Allocate(false);
        if (normalizedStorage != nullptr) {
            normalizedStorage->Allocate(false);
        }

        const DataType type = hyperInput.dataType;
        const DataType blockType = blockOutput.dataType;
        const DataType injectionType = injection.dataType;
        Qwen4ParallelFor(rows * groups, [&](int start, int end) {
            for (int item = start; item < end; item++) {
                const int row = item / groups;
                const int group = item % groups;
                const uint64_t hyperBase = (uint64_t)row * channels +
                    (uint64_t)group * groupChannels;
                const uint64_t blockBase =
                    (uint64_t)row * groupChannels;
                const float scale = Qwen4RoundCpu(Qwen4LoadCpu(
                    injection.cpuData, injectionType,
                    (uint64_t)row * groups + group), type);
                float squareSum = 0.0f;
                for (int channel = 0; channel < groupChannels; channel++) {
                    const float blockValue = Qwen4RoundCpu(Qwen4LoadCpu(
                        blockOutput.cpuData, blockType,
                        blockBase + channel), type);
                    const float injected = Qwen4RoundCpu(
                        blockValue * scale, type);
                    const float combined = Qwen4RoundCpu(
                        Qwen4LoadCpu(hyperInput.cpuData, type,
                                     hyperBase + channel) + injected,
                        type);
                    Qwen4StoreCpu(output.cpuData, type,
                                  hyperBase + channel, combined);
                    squareSum += combined * combined;
                }
                const float normScale = 1.0f /
                    std::sqrt(squareSum / groupChannels + eps);
                for (int channel = 0; channel < groupChannels; channel++) {
                    const uint64_t index = hyperBase + channel;
                    const float value = Qwen4LoadCpu(
                        output.cpuData, type, index);
                    const float normWeight = Qwen4LoadCpu(
                        weight.cpuData, weight.dataType,
                        (uint64_t)group * groupChannels + channel);
                    const float normalizedValue =
                        value * normScale * normWeight;
                    Qwen4StoreCpu(normalized.cpuData, type,
                                  index, normalizedValue);
                    if (normalizedStorage != nullptr) {
                        const float rounded = Qwen4LoadCpu(
                            normalized.cpuData, type, index);
                        if (normalizedStorage->dataType ==
                                DataType::FLOAT16) {
                            ((uint16_t*)normalizedStorage->cpuData)[index] =
                                Qwen4FloatToHalfRz(rounded);
                        } else {
                            Qwen4StoreCpu(normalizedStorage->cpuData,
                                          normalizedStorage->dataType,
                                          index, rounded);
                        }
                    }
                }
            }
        });
    }

    bool CpuCausalDepthwiseConv1DDecodeOp::CanRun(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *datas.find("input")->second;
        Data &weight = *datas.find("weight")->second;
        Data &state = *datas.find("state")->second;
        const int kernel = intParams.find("kernel")->second;
        if (!Qwen4ActivationType(input.dataType) ||
            weight.dataType != DataType::FLOAT32 ||
            state.dataType != DataType::FLOAT32 || kernel <= 0 ||
            input.dims.size() != 3 || input.dims[2] != 1) {
            return false;
        }
        const int batch = input.dims[0];
        const int channels = input.dims[1];
        return batch > 0 && channels > 0 &&
               weight.Count(0) == (uint64_t)channels * kernel &&
               (state.dims.empty() ||
                state.dims == std::vector<int>({batch, channels, kernel}));
    }

    bool CpuCausalDepthwiseConv1DPrefillOp::CanRun(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *datas.find("input")->second;
        Data &weight = *datas.find("weight")->second;
        Data &state = *datas.find("state")->second;
        const int kernel = intParams.find("kernel")->second;
        if (!Qwen4ActivationType(input.dataType) ||
            weight.dataType != DataType::FLOAT32 ||
            state.dataType != DataType::FLOAT32 || kernel <= 0 ||
            input.dims.size() != 3 || input.dims[1] <= 0) {
            return false;
        }
        const int batch = input.dims[0];
        const int channels = input.dims[2];
        return batch > 0 && channels > 0 &&
               weight.Count(0) == (uint64_t)channels * kernel &&
               (state.dims.empty() ||
                state.dims == std::vector<int>({batch, channels, kernel}));
    }

    void CpuQwen4QSASelectOp::Reshape(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &query = *datas.find("query")->second;
        Data &output = *datas.find("output")->second;
        const int keyLength = intParams.find("keyLength")->second;
        const int heads = intParams.find("heads")->second;
        const int headDim = intParams.find("headDim")->second;
        const int tokenBudget = intParams.find("tokenBudget")->second;
        const int compressRatio = intParams.find("compressRatio")->second;
        const auto queryStartIt = intParams.find("queryStart");
        const int queryStart = queryStartIt == intParams.end()
            ? -1 : queryStartIt->second;
        AssertInFastLLM(keyLength > 0 && heads > 0 && headDim > 0 &&
                        tokenBudget > 0 &&
                        compressRatio > 0 &&
                        query.Count(0) % ((uint64_t)heads * headDim) == 0,
                        "Qwen4QSASelect received invalid dimensions.\n");
        const int rows = (int)(query.Count(0) /
                               ((uint64_t)heads * headDim));
        AssertInFastLLM(queryStart == -1 ||
                        (queryStart >= 0 && queryStart + rows <= keyLength),
                        "Qwen4QSASelect received an invalid causal range.\n");
        const int outputWidth = queryStart >= 0
            ? tokenBudget + compressRatio - 1
            : tokenBudget + keyLength % compressRatio;
        output.dataType = DataType::INT32;
        output.UpdateUnitSize();
        output.Resize({rows, outputWidth});
    }

    void CpuQwen4QSASelectOp::Run(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &query = *datas.find("query")->second;
        Data &compressedKeys = *datas.find("compressedKeys")->second;
        Data &output = *datas.find("output")->second;
        const int keyLength = intParams.find("keyLength")->second;
        const int heads = intParams.find("heads")->second;
        const int headDim = intParams.find("headDim")->second;
        const int tokenBudget = intParams.find("tokenBudget")->second;
        const int compressRatio = intParams.find("compressRatio")->second;
        const auto queryStartIt = intParams.find("queryStart");
        const int queryStart = queryStartIt == intParams.end()
            ? -1 : queryStartIt->second;
        const int blockTopK = tokenBudget / compressRatio;
        const int completeBlocks = keyLength / compressRatio;
        const int rows = (int)(query.Count(0) /
                               ((uint64_t)heads * headDim));
        const int outputWidth = queryStart >= 0
            ? tokenBudget + compressRatio - 1
            : tokenBudget + keyLength % compressRatio;

        Qwen4AssertCpuTensor(query, "Qwen4QSASelect query");
        Qwen4AssertCpuTensor(compressedKeys,
                             "Qwen4QSASelect compressed keys");
        AssertInFastLLM(compressedKeys.dataType == DataType::FLOAT32 &&
                        compressedKeys.dims.size() == 2 &&
                        compressedKeys.dims[0] >= completeBlocks &&
                        compressedKeys.dims[1] == headDim &&
                        tokenBudget > 0 && compressRatio > 0 &&
                        tokenBudget % compressRatio == 0 &&
                        blockTopK > 0 && rows > 0 &&
                        (queryStart == -1 ||
                         (queryStart >= 0 &&
                          queryStart + rows <= keyLength)) &&
                        output.dims ==
                            std::vector<int>({rows, outputWidth}),
                        "Qwen4QSASelect received invalid cache or parameters.\n");
        output.Allocate(false);
        int32_t *outputData = (int32_t*)output.cpuData;
        std::fill(outputData, outputData + output.Count(0), -1);

        const float *keyData = (const float*)compressedKeys.cpuData;
        const float inverseSqrt = 1.0f / std::sqrt((float)headDim);
        Qwen4ParallelFor(rows, [&](int start, int end) {
            for (int row = start; row < end; row++) {
                const int visibleLength = queryStart >= 0
                    ? queryStart + row + 1 : keyLength;
                const int rowBlocks = visibleLength / compressRatio;
                const int selectedCount = std::min(blockTopK, rowBlocks);
                std::vector<int> selected(selectedCount);
                if (rowBlocks <= blockTopK) {
                    std::iota(selected.begin(), selected.end(), 0);
                } else {
                    std::vector<std::pair<float, int>> ranked(rowBlocks);
                    for (int block = 0; block < rowBlocks; block++) {
                        const float *blockKey = keyData +
                                                (size_t)block * headDim;
                        float score = 0.0f;
                        for (int head = 0; head < heads; head++) {
                            const uint64_t queryBase =
                                ((uint64_t)row * heads + head) * headDim;
                            float dot = 0.0f;
                            for (int column = 0; column < headDim; column++) {
                                dot += Qwen4LoadCpu(
                                           query.cpuData, query.dataType,
                                           queryBase + column) *
                                       blockKey[column];
                            }
                            score += std::max(dot, 0.0f);
                        }
                        const float value = score * inverseSqrt;
                        ranked[block] = {
                            std::isfinite(value)
                                ? value
                                : -std::numeric_limits<float>::infinity(),
                            block};
                    }
                    std::partial_sort(
                        ranked.begin(), ranked.begin() + selectedCount,
                        ranked.end(),
                        [](const std::pair<float, int> &left,
                           const std::pair<float, int> &right) {
                            return left.first != right.first
                                ? left.first > right.first
                                : left.second < right.second;
                        });
                    for (int rank = 0; rank < selectedCount; rank++) {
                        selected[rank] = ranked[rank].second;
                    }
                    std::sort(selected.begin(), selected.end());
                }
                int32_t *rowOutput = outputData + (size_t)row * outputWidth;
                int cursor = 0;
                for (int block : selected) {
                    for (int member = 0; member < compressRatio; member++) {
                        rowOutput[cursor++] = block * compressRatio + member;
                    }
                }
                for (int token = rowBlocks * compressRatio;
                     token < visibleLength; token++) {
                    rowOutput[cursor++] = token;
                }
                AssertInFastLLM(cursor <= outputWidth,
                                "Qwen4QSASelect output width overflow.\n");
            }
        });
    }

    void CpuQwen4QSABuildMaskOp::Reshape(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &indices = *datas.find("indices")->second;
        Data &reference = *datas.find("reference")->second;
        Data &output = *datas.find("output")->second;
        const int keyLength = intParams.find("keyLength")->second;
        AssertInFastLLM(indices.dims.size() == 2 && keyLength > 0 &&
                        Qwen4ActivationType(reference.dataType),
                        "Qwen4QSABuildMask received invalid dimensions.\n");
        output.dataType = reference.dataType;
        output.UpdateUnitSize();
        output.Resize({1, indices.dims[0], keyLength});
    }

    void CpuQwen4QSABuildMaskOp::Run(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &indices = *datas.find("indices")->second;
        Data &reference = *datas.find("reference")->second;
        Data &output = *datas.find("output")->second;
        const int keyLength = intParams.find("keyLength")->second;
        AssertInFastLLM(indices.dataDevice == DataDevice::CPU &&
                        indices.dataType == DataType::INT32 &&
                        indices.cpuData != nullptr &&
                        reference.dataDevice == DataDevice::CPU &&
                        Qwen4ActivationType(reference.dataType) &&
                        indices.dims.size() == 2 && keyLength > 0,
                        "Qwen4QSABuildMask received incompatible tensors.\n");
        output.Allocate(false);
        const int rows = indices.dims[0];
        const int width = indices.dims[1];
        const int32_t *indexData =
            reinterpret_cast<const int32_t *>(indices.cpuData);
        Qwen4ParallelFor(rows, [&](int start, int end) {
            for (int row = start; row < end; row++) {
                const uint64_t maskBase = (uint64_t)row * keyLength;
                for (int token = 0; token < keyLength; token++) {
                    Qwen4StoreCpu(output.cpuData, output.dataType,
                                  maskBase + token, 1.0f);
                }
                const int32_t *rowIndices = indexData +
                                            (size_t)row * width;
                for (int index = 0; index < width; index++) {
                    const int token = rowIndices[index];
                    if (token >= 0 && token < keyLength) {
                        Qwen4StoreCpu(output.cpuData, output.dataType,
                                      maskBase + token, 0.0f);
                    }
                }
            }
        });
    }

    void CpuQwen4SparseAttentionOp::Reshape(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &query = *datas.find("query")->second;
        Data &output = *datas.find("output")->second;
        output.dataType = query.dataType;
        output.UpdateUnitSize();
        output.Resize(query.dims);
    }

    void CpuQwen4SparseAttentionOp::Run(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &query = *datas.find("query")->second;
        Data &key = *datas.find("key")->second;
        Data &value = *datas.find("value")->second;
        Data &indices = *datas.find("indices")->second;
        Data &output = *datas.find("output")->second;
        const int group = intParams.find("group")->second;
        const float scale = floatParams.find("scale")->second;
        Qwen4AssertCpuTensor(query, "Qwen4SparseAttention query");
        Qwen4AssertCpuTensor(key, "Qwen4SparseAttention key");
        Qwen4AssertCpuTensor(value, "Qwen4SparseAttention value");
        AssertInFastLLM(indices.dataDevice == DataDevice::CPU &&
                        indices.dataType == DataType::INT32 &&
                        indices.cpuData != nullptr &&
                        query.dims.size() == 3 && key.dims.size() == 3 &&
                        value.dims == key.dims && indices.dims.size() == 2 &&
                        query.dataType == key.dataType &&
                        query.dataType == value.dataType && group > 0 &&
                        query.dims[0] == key.dims[0] * group &&
                        query.dims[1] == indices.dims[0] &&
                        query.dims[2] == key.dims[2],
                        "Qwen4SparseAttention received incompatible tensors.\n");
        output.Allocate(false);
        const int queryHeads = query.dims[0];
        const int sequence = query.dims[1];
        const int keyLength = key.dims[1];
        const int headDim = query.dims[2];
        const int indexWidth = indices.dims[1];
        const int32_t *indexData = (const int32_t*)indices.cpuData;
        const DataType type = query.dataType;
        const float nativeScale = Qwen4RoundCpu(scale, type);

        Qwen4ParallelFor(queryHeads * sequence, [&](int start, int end) {
            std::vector<float> logits(indexWidth);
            std::vector<float> probabilities(indexWidth);
            for (int row = start; row < end; row++) {
                const int head = row / sequence;
                const int token = row % sequence;
                const int keyHead = head / group;
                const int32_t *rowIndices = indexData +
                    (size_t)token * indexWidth;
                const uint64_t queryBase =
                    ((uint64_t)head * sequence + token) * headDim;
                float maximum = -std::numeric_limits<float>::infinity();
                int valid = 0;
                for (int selected = 0; selected < indexWidth; selected++) {
                    const int keyIndex = rowIndices[selected];
                    if (keyIndex < 0) {
                        break;
                    }
                    AssertInFastLLM(keyIndex < keyLength,
                                    "Qwen4SparseAttention index is out of range.\n");
                    const uint64_t keyBase =
                        ((uint64_t)keyHead * keyLength + keyIndex) * headDim;
                    float dot = 0.0f;
                    for (int column = 0; column < headDim; column++) {
                        dot += Qwen4LoadCpu(query.cpuData, type,
                                            queryBase + column) *
                               Qwen4LoadCpu(key.cpuData, type,
                                            keyBase + column);
                    }
                    logits[selected] = Qwen4RoundCpu(
                        dot * nativeScale, type);
                    maximum = std::max(maximum, logits[selected]);
                    valid++;
                }
                AssertInFastLLM(valid > 0,
                                "Qwen4SparseAttention selected no keys.\n");
                float sum = 0.0f;
                for (int selected = 0; selected < valid; selected++) {
                    probabilities[selected] =
                        std::exp(logits[selected] - maximum);
                    sum += probabilities[selected];
                }
                for (int selected = 0; selected < valid; selected++) {
                    probabilities[selected] = Qwen4RoundCpu(
                        probabilities[selected] / sum, type);
                }
                for (int column = 0; column < headDim; column++) {
                    float result = 0.0f;
                    for (int selected = 0; selected < valid; selected++) {
                        const int keyIndex = rowIndices[selected];
                        const uint64_t valueIndex =
                            ((uint64_t)keyHead * keyLength + keyIndex) *
                                headDim + column;
                        result += probabilities[selected] *
                                  Qwen4LoadCpu(value.cpuData, type,
                                               valueIndex);
                    }
                    Qwen4StoreCpu(output.cpuData, type,
                                  queryBase + column, result);
                }
            }
        });
    }

    void CpuCausalDepthwiseConv1DDecodeOp::Reshape(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *datas.find("input")->second;
        Data &weight = *datas.find("weight")->second;
        Data &state = *datas.find("state")->second;
        Data &output = *datas.find("output")->second;
        const int kernel = intParams.find("kernel")->second;
        AssertInFastLLM(input.dims.size() == 3 && input.dims[2] == 1 &&
                        kernel > 0,
                        "CausalDepthwiseConv1DDecode expects one token.\n");
        const int batch = input.dims[0];
        const int channels = input.dims[1];
        AssertInFastLLM(weight.dataType == DataType::FLOAT32 &&
                        weight.Count(0) == (uint64_t)channels * kernel,
                        "CausalDepthwiseConv1DDecode weight shape mismatch.\n");
        if (state.dims.empty()) {
            state.dataType = DataType::FLOAT32;
            state.UpdateUnitSize();
            state.Resize({batch, channels, kernel});
        } else {
            AssertInFastLLM(
                state.dataType == DataType::FLOAT32 &&
                state.dims == std::vector<int>({batch, channels, kernel}),
                "CausalDepthwiseConv1DDecode state shape mismatch.\n");
        }
        output.dataType = DataType::FLOAT32;
        output.Resize({batch, 1, channels});
    }

    void CpuCausalDepthwiseConv1DPrefillOp::Reshape(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *datas.find("input")->second;
        Data &weight = *datas.find("weight")->second;
        Data &state = *datas.find("state")->second;
        Data &output = *datas.find("output")->second;
        const int kernel = intParams.find("kernel")->second;
        AssertInFastLLM(input.dims.size() == 3 && input.dims[0] > 0 &&
                        input.dims[1] > 0 && input.dims[2] > 0 && kernel > 0,
                        "CausalDepthwiseConv1DPrefill expects [B, S, C].\n");
        const int batch = input.dims[0];
        const int channels = input.dims[2];
        AssertInFastLLM(weight.dataType == DataType::FLOAT32 &&
                        weight.Count(0) == (uint64_t)channels * kernel,
                        "CausalDepthwiseConv1DPrefill weight shape mismatch.\n");
        if (state.dims.empty()) {
            state.dataType = DataType::FLOAT32;
            state.UpdateUnitSize();
            state.Resize({batch, channels, kernel});
        } else {
            AssertInFastLLM(
                state.dataType == DataType::FLOAT32 &&
                state.dims == std::vector<int>({batch, channels, kernel}),
                "CausalDepthwiseConv1DPrefill state shape mismatch.\n");
        }
        auto outputTypeIt = intParams.find("outputType");
        const DataType outputType = outputTypeIt == intParams.end()
            ? DataType::FLOAT32 : (DataType)outputTypeIt->second;
        AssertInFastLLM(Qwen4ActivationType(outputType),
                        "CausalDepthwiseConv1DPrefill output type is invalid.\n");
        output.dataType = outputType;
        output.UpdateUnitSize();
        output.Resize(input.dims);
    }

    void CpuCausalDepthwiseConv1DPrefillOp::Run(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *datas.find("input")->second;
        Data &weight = *datas.find("weight")->second;
        Data &state = *datas.find("state")->second;
        Data &output = *datas.find("output")->second;
        const int kernel = intParams.find("kernel")->second;
        auto siluIt = intParams.find("silu");
        const bool silu = siluIt != intParams.end() && siluIt->second != 0;
        const bool initializeState = state.cpuData == nullptr;
        state.Allocate(false);
        output.Allocate(false);
        Qwen4AssertCpuTensor(input, "CausalDepthwiseConv1DPrefill input");
        AssertInFastLLM(weight.cpuData != nullptr &&
                        weight.dataType == DataType::FLOAT32 &&
                        state.cpuData != nullptr &&
                        state.dataType == DataType::FLOAT32,
                        "CausalDepthwiseConv1DPrefill requires float32 weight/state.\n");
        float *stateData = (float*)state.cpuData;
        if (initializeState) {
            std::fill(stateData, stateData + state.Count(0), 0.0f);
        }

        const int batch = input.dims[0];
        const int sequence = input.dims[1];
        const int channels = input.dims[2];
        const float *weightData = (const float*)weight.cpuData;
        const int total = batch * sequence * channels;
        Qwen4ParallelFor(total, [&](int start, int end) {
            for (int item = start; item < end; item++) {
                const int channel = item % channels;
                const int row = item / channels;
                const int token = row % sequence;
                const int batchIndex = row / sequence;
                const float *weightRow = weightData +
                    (uint64_t)channel * kernel;
                const float *stateRow = stateData +
                    ((uint64_t)batchIndex * channels + channel) * kernel;
                float value = 0.0f;
                for (int tap = 0; tap < kernel; tap++) {
                    const int sourceToken = token - kernel + 1 + tap;
                    const float sample = sourceToken >= 0
                        ? Qwen4LoadCpu(
                              input.cpuData, input.dataType,
                              ((uint64_t)batchIndex * sequence + sourceToken) *
                                  channels + channel)
                        : stateRow[kernel + sourceToken];
                    value += sample * weightRow[tap];
                }
                if (silu) {
                    // Match the established float32 Silu op exactly.  Its
                    // double literal is observable after the recurrent GDN
                    // amplifies sub-ULP convolution differences.
                    value = value / (1.0 + expf(-value));
                }
                if (output.dataType == DataType::FLOAT16) {
                    ((uint16_t*)output.cpuData)[item] =
                        Qwen4FloatToHalfRz(value);
                } else {
                    Qwen4StoreCpu(output.cpuData, output.dataType,
                                  (uint64_t)item, value);
                }
            }
        });

        Qwen4ParallelFor(batch * channels, [&](int start, int end) {
            for (int item = start; item < end; item++) {
                const int batchIndex = item / channels;
                const int channel = item % channels;
                float *stateRow = stateData + (uint64_t)item * kernel;
                if (sequence >= kernel) {
                    for (int slot = 0; slot < kernel; slot++) {
                        stateRow[slot] = Qwen4LoadCpu(
                            input.cpuData, input.dataType,
                            ((uint64_t)batchIndex * sequence +
                             sequence - kernel + slot) * channels + channel);
                    }
                } else {
                    for (int slot = 0; slot < kernel - sequence; slot++) {
                        stateRow[slot] = stateRow[slot + sequence];
                    }
                    for (int token = 0; token < sequence; token++) {
                        stateRow[kernel - sequence + token] = Qwen4LoadCpu(
                            input.cpuData, input.dataType,
                            ((uint64_t)batchIndex * sequence + token) *
                                channels + channel);
                    }
                }
            }
        });
    }

    void CpuCausalDepthwiseConv1DDecodeOp::Run(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *datas.find("input")->second;
        Data &weight = *datas.find("weight")->second;
        Data &state = *datas.find("state")->second;
        Data &output = *datas.find("output")->second;
        const int kernel = intParams.find("kernel")->second;
        auto siluIt = intParams.find("silu");
        const bool silu = siluIt != intParams.end() && siluIt->second != 0;
        const bool initializeState = state.cpuData == nullptr;
        state.Allocate(false);
        output.Allocate(false);
        Qwen4AssertCpuTensor(input, "CausalDepthwiseConv1DDecode input");
        AssertInFastLLM(weight.cpuData != nullptr &&
                        weight.dataType == DataType::FLOAT32 &&
                        state.cpuData != nullptr &&
                        state.dataType == DataType::FLOAT32,
                        "CausalDepthwiseConv1DDecode requires float32 weight/state.\n");
        if (initializeState) {
            std::fill((float*)state.cpuData,
                      (float*)state.cpuData + state.Count(0), 0.0f);
        }

        const int batch = input.dims[0];
        const int channels = input.dims[1];
        const int total = batch * channels;
        const float *weightData = (const float*)weight.cpuData;
        float *stateData = (float*)state.cpuData;
        float *outputData = (float*)output.cpuData;
        Qwen4ParallelFor(total, [&](int start, int end) {
            for (int item = start; item < end; item++) {
                const int channel = item % channels;
                float *stateRow = stateData + (uint64_t)item * kernel;
                for (int tap = 0; tap + 1 < kernel; tap++) {
                    stateRow[tap] = stateRow[tap + 1];
                }
                stateRow[kernel - 1] = Qwen4LoadCpu(
                    input.cpuData, input.dataType, item);
                const float *weightRow = weightData +
                                         (uint64_t)channel * kernel;
                float value = 0.0f;
                for (int tap = 0; tap < kernel; tap++) {
                    value += stateRow[tap] * weightRow[tap];
                }
                if (silu) {
                    value = value / (1.0 + expf(-value));
                }
                outputData[item] = value;
            }
        });
    }

    void CpuQwen4GatedDeltaRuleDecodeOp::Reshape(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &qkv = *datas.find("input")->second;
        Data &state = *datas.find("state")->second;
        Data &output = *datas.find("output")->second;
        const int valueHeads = intParams.find("valueHeads")->second;
        const int valueDim = intParams.find("valueDim")->second;
        AssertInFastLLM(qkv.dims.size() == 3 && qkv.dims[1] > 0 &&
                        valueHeads > 0 && valueDim > 0,
                        "GatedDeltaRuleSequence expects a non-empty sequence.\n");
        output.dataType = DataType::FLOAT32;
        output.Resize({qkv.dims[0], qkv.dims[1],
                       valueHeads * valueDim});
        auto stateOutputIt = datas.find("stateOutput");
        if (stateOutputIt != datas.end() &&
            stateOutputIt->second != nullptr) {
            stateOutputIt->second->dataType = state.dataType;
            stateOutputIt->second->Resize(state.dims);
        }
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
        auto stateOutputIt = datas.find("stateOutput");
        Data *stateOutput = stateOutputIt == datas.end()
            ? nullptr : stateOutputIt->second;
        const int keyHeads = intParams.find("keyHeads")->second;
        const int valueHeads = intParams.find("valueHeads")->second;
        const int keyDim = intParams.find("keyDim")->second;
        const int valueDim = intParams.find("valueDim")->second;
        const float recurrentEps =
            floatParams.find("recurrentEps")->second;
        const int batch = qkv.dims[0];
        const int sequence = qkv.dims[1];
        const int qkvChannels = 2 * keyHeads * keyDim +
                                valueHeads * valueDim;

        Qwen4AssertCpuTensor(qkv, "GatedDeltaRuleSequence qkv");
        Qwen4AssertCpuTensor(alpha, "GatedDeltaRuleSequence alpha");
        Qwen4AssertCpuTensor(beta, "GatedDeltaRuleSequence beta");
        Qwen4AssertCpuTensor(aLog, "GatedDeltaRuleSequence A_log");
        Qwen4AssertCpuTensor(dtBias, "GatedDeltaRuleSequence dt_bias");
        Qwen4AssertCpuTensor(state, "GatedDeltaRuleSequence state");
        AssertInFastLLM(qkv.dataType == DataType::FLOAT32 &&
                        state.dataType == DataType::FLOAT32 &&
                        aLog.dataType == DataType::FLOAT32 &&
                        dtBias.dataType == DataType::FLOAT32 &&
                        output.dataType == DataType::FLOAT32,
                        "GatedDeltaRuleSequence keeps qkv, state and parameters in float32.\n");
        AssertInFastLLM(keyHeads > 0 && valueHeads > 0 &&
                        valueHeads % keyHeads == 0 &&
                        keyDim > 0 && valueDim > 0 &&
                        qkv.dims.size() == 3 && sequence > 0 &&
                        qkv.dims[2] == qkvChannels &&
                        alpha.Count(0) ==
                            (uint64_t)batch * sequence * valueHeads &&
                        beta.Count(0) ==
                            (uint64_t)batch * sequence * valueHeads &&
                        aLog.Count(0) == (uint64_t)valueHeads &&
                        dtBias.Count(0) == (uint64_t)valueHeads &&
                        state.dims == std::vector<int>({batch, valueHeads,
                                                       keyDim, valueDim}),
                        "GatedDeltaRuleSequence shape mismatch.\n");
        AssertInFastLLM(
                        stateOutput == nullptr ||
                        (stateOutput->dataType == DataType::FLOAT32 &&
                         stateOutput->dims == state.dims),
                        "GatedDeltaRuleSequence state output shape mismatch.\n");
        output.Allocate(false);
        if (stateOutput != nullptr) {
            stateOutput->Allocate(false);
            Qwen4AssertCpuTensor(
                *stateOutput, "GatedDeltaRuleSequence state output");
        }

        const float *qkvData = (const float*)qkv.cpuData;
        const float *aLogData = (const float*)aLog.cpuData;
        const float *dtBiasData = (const float*)dtBias.cpuData;
        float *stateData = (float*)state.cpuData;
        float *nextStateData = stateOutput == nullptr
            ? stateData : (float*)stateOutput->cpuData;
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
                float *headState = stateData +
                    (uint64_t)item * keyDim * valueDim;
                float *headNextState = nextStateData +
                    (uint64_t)item * keyDim * valueDim;

                for (int token = 0; token < sequence; token++) {
                    const uint64_t row =
                        (uint64_t)batchIndex * sequence + token;
                    const uint64_t qkvBase = row * qkvChannels;
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
                        querySquares +=
                            queryRaw[channel] * queryRaw[channel];
                        keySquares += keyRaw[channel] * keyRaw[channel];
                    }
                    const float queryScale = 1.0f / std::sqrt(
                        querySquares / keyDim + recurrentEps);
                    const float keyScale = 1.0f / std::sqrt(
                        keySquares / keyDim + recurrentEps);
                    for (int channel = 0; channel < keyDim; channel++) {
                        query[channel] = queryRaw[channel] * queryScale *
                                         inverseHead * inverseHead;
                        key[channel] =
                            keyRaw[channel] * keyScale * inverseHead;
                    }

                    const uint64_t gateIndex =
                        row * valueHeads + valueHead;
                    const float betaValue = Qwen4SigmoidCpu(Qwen4LoadCpu(
                        beta.cpuData, beta.dataType, gateIndex));
                    const float alphaValue = Qwen4LoadCpu(
                        alpha.cpuData, alpha.dataType, gateIndex);
                    const float biasedAlpha =
                        alphaValue + dtBiasData[valueHead];
                    const float softplus = biasedAlpha > 20.0f
                        ? biasedAlpha : std::log1p(std::exp(biasedAlpha));
                    const float logDecay =
                        -std::exp(aLogData[valueHead]) * softplus;
                    const float decay = std::exp(logDecay);
                    const float *headPreviousState =
                        token == 0 ? headState : headNextState;

                    for (int valueChannel = 0;
                         valueChannel < valueDim; valueChannel++) {
                        float memory = 0.0f;
                        for (int keyChannel = 0;
                             keyChannel < keyDim; keyChannel++) {
                            const uint64_t stateIndex =
                                (uint64_t)keyChannel * valueDim +
                                valueChannel;
                            const float previous =
                                headPreviousState[stateIndex];
                            const float scaled = previous * decay;
                            headNextState[stateIndex] = scaled;
                            memory += scaled * key[keyChannel];
                        }
                        const float delta =
                            (value[valueChannel] - memory) * betaValue;
                        float core = 0.0f;
                        for (int keyChannel = 0;
                             keyChannel < keyDim; keyChannel++) {
                            const uint64_t stateIndex =
                                (uint64_t)keyChannel * valueDim +
                                valueChannel;
                            const float updated = headNextState[stateIndex] +
                                key[keyChannel] * delta;
                            headNextState[stateIndex] = updated;
                            core += updated * query[keyChannel];
                        }
                        outputData[
                            (row * valueHeads + valueHead) * valueDim +
                            valueChannel] = core;
                    }
                }
            }
        });
    }
}
