#include "devices/cpu/kimi_k3_ops.h"

#include "devices/cpu/alivethreadpool.h"
#include "devices/cpu/computeutils.h"
#include "utils.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

namespace fastllm {
    namespace {
        class KimiK3RangeTask final : public MultiThreadBaseOp {
        public:
            KimiK3RangeTask(int start, int end,
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

        void KimiK3ParallelFor(int count,
                               const std::function<void(int, int)> &function) {
            if (count <= 0) {
                return;
            }
            AliveThreadPool *pool = GetAlivePool();
            int firstThread = pool->curActivateThreadInterval.first;
            int availableThreads = pool->curActivateThreadInterval.second - firstThread;
            int threadCount = std::min(count, std::max(1, availableThreads));
            if (threadCount == 1) {
                function(0, count);
                return;
            }

            std::vector<KimiK3RangeTask*> tasks;
            tasks.reserve(threadCount);
            for (int thread = 0; thread < threadCount; thread++) {
                int start = (int)((int64_t)count * thread / threadCount);
                int end = (int)((int64_t)count * (thread + 1) / threadCount);
                tasks.push_back(new KimiK3RangeTask(start, end, function));
                pool->PushOp(firstThread + thread, tasks.back());
            }
            for (int thread = 0; thread < threadCount; thread++) {
                pool->Wait(firstThread + thread);
                delete tasks[thread];
            }
        }

        void KimiK3ReshapeLike(const Data &input, Data &output,
                               DataType outputType) {
            output.dataType = outputType;
            output.Resize(input.dims);
        }

        void KimiK3AssertBFloat16(const Data &data,
                                  const std::string &name) {
            AssertInFastLLM(data.dataType == DataType::BFLOAT16,
                            name + " must use bfloat16 activations.");
        }

        void KimiK3AssertFloat32(const Data &data,
                                 const std::string &name) {
            AssertInFastLLM(data.dataType == DataType::FLOAT32,
                            name + " must use float32 parameters.");
        }

        float KimiK3Sigmoid(float value) {
            return 1.0f / (1.0f + std::exp(-value));
        }

        bool KimiK3SupportedExpertWeight(const Data &weight) {
            return weight.dataType == DataType::NVFP4 ||
                   (weight.dataType == DataType::DATA_GGUF_FORMAT &&
                    (weight.ggmlType == GGML_TYPE_Q2_K ||
                     (weight.IsRepacked &&
                      weight.ggmlType == GGML_TYPE_Q2_K_R4)));
        }

        void KimiK3RunExpertLinear(
                const uint16_t *input, Data &weight, float *output,
                int rows, int inputDimension, int outputDimension,
                AliveThreadPool *pool, int firstThread, int threadCount) {
            if (weight.dataType == DataType::NVFP4) {
                RunLinearBFloat16NVFP4(
                    (uint16_t*)input, weight, output, nullptr,
                    rows, inputDimension, outputDimension,
                    pool, firstThread, threadCount);
                return;
            }
            AssertInFastLLM(
                KimiK3SupportedExpertWeight(weight),
                "KimiK3RoutedExperts supports compact MXFP4 and GGML Q2_K weights.");
            std::vector<float> floatInput((uint64_t)rows * inputDimension);
            for (uint64_t i = 0; i < floatInput.size(); i++) {
                floatInput[i] = BFloat16BitsToFloat32(input[i]);
            }
            RunLinearFloat32GGUF(
                floatInput.data(), (uint8_t*)weight.cpuData, output, nullptr,
                &weight, rows, inputDimension, outputDimension,
                pool, firstThread, threadCount);
        }
    }

    bool CpuKimiK3RMSNormOp::CanRun(
            const std::string &, const DataDict &datas,
            const FloatDict &, const IntDict &) {
        return datas.find("input") != datas.end() &&
               datas.find("input")->second->dataType == DataType::BFLOAT16;
    }

    void CpuKimiK3RMSNormOp::Reshape(
            const std::string &, const DataDict &datas,
            const FloatDict &, const IntDict &) {
        KimiK3ReshapeLike(*datas.find("input")->second,
                          *datas.find("output")->second,
                          DataType::BFLOAT16);
    }

    void CpuKimiK3RMSNormOp::Run(
            const std::string &, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &) {
        Data &input = *datas.find("input")->second;
        Data &weight = *datas.find("weight")->second;
        Data &output = *datas.find("output")->second;
        float eps = floatParams.find("eps") == floatParams.end() ?
                    1e-5f : floatParams.find("eps")->second;
        KimiK3AssertBFloat16(input, "KimiK3RMSNorm input");
        KimiK3AssertFloat32(weight, "KimiK3RMSNorm weight");
        int channels = input.dims.back();
        int rows = (int)(input.Count(0) / channels);
        AssertInFastLLM(weight.Count(0) == (uint64_t)channels,
                        "KimiK3RMSNorm weight shape mismatch.");
        output.Allocate(false);

        const uint16_t *source = (const uint16_t*)input.cpuData;
        const float *normWeight = (const float*)weight.cpuData;
        uint16_t *destination = (uint16_t*)output.cpuData;
        KimiK3ParallelFor(rows, [&](int start, int end) {
            for (int row = start; row < end; row++) {
                uint64_t base = (uint64_t)row * channels;
                float squareSum = 0.0f;
                for (int channel = 0; channel < channels; channel++) {
                    float value = BFloat16BitsToFloat32(source[base + channel]);
                    squareSum += value * value;
                }
                float scale = 1.0f / std::sqrt(squareSum / channels + eps);
                for (int channel = 0; channel < channels; channel++) {
                    float value = BFloat16BitsToFloat32(source[base + channel]);
                    float normalized = RoundFloat32ToBFloat16RNE(value * scale);
                    destination[base + channel] = Float32ToBFloat16RNEBits(
                        normalized * normWeight[channel]);
                }
            }
        });
    }

    bool CpuKimiK3CausalConv1DOp::CanRun(
            const std::string &, const DataDict &datas,
            const FloatDict &, const IntDict &) {
        return datas.find("input") != datas.end() &&
               datas.find("input")->second->dataType == DataType::BFLOAT16;
    }

    void CpuKimiK3CausalConv1DOp::Reshape(
            const std::string &, const DataDict &datas,
            const FloatDict &, const IntDict &intParams) {
        KimiK3ReshapeLike(*datas.find("input")->second,
                          *datas.find("output")->second,
                          DataType::BFLOAT16);
        auto cacheIt = datas.find("cache");
        if (cacheIt != datas.end()) {
            Data &input = *datas.find("input")->second;
            Data &cache = *cacheIt->second;
            int kernelSize = intParams.find("kernelSize")->second;
            cache.dataType = DataType::BFLOAT16;
            cache.Resize({input.dims[0], kernelSize - 1, input.dims[2]});
        }
    }

    void CpuKimiK3CausalConv1DOp::Run(
            const std::string &, const DataDict &datas,
            const FloatDict &, const IntDict &intParams) {
        Data &input = *datas.find("input")->second;
        Data &weight = *datas.find("weight")->second;
        Data &output = *datas.find("output")->second;
        auto cacheIt = datas.find("cache");
        Data *cache = cacheIt == datas.end() ? nullptr : cacheIt->second;
        int kernelSize = intParams.find("kernelSize")->second;
        KimiK3AssertBFloat16(input, "KimiK3CausalConv1D input");
        KimiK3AssertFloat32(weight, "KimiK3CausalConv1D weight");
        AssertInFastLLM(input.dims.size() == 3 && weight.dims.size() == 3 &&
                        weight.dims[1] == 1 && weight.dims[2] == kernelSize &&
                        input.dims[2] == weight.dims[0],
                        "KimiK3CausalConv1D shape mismatch.");
        int batch = input.dims[0];
        int sequence = input.dims[1];
        int channels = input.dims[2];
        output.Allocate(false);

        bool hasCachedValues = cache != nullptr && cache->cpuData != nullptr;
        if (cache != nullptr) {
            cache->Allocate(false);
            if (!hasCachedValues) {
                std::fill((uint16_t*)cache->cpuData,
                          (uint16_t*)cache->cpuData + cache->Count(0), 0);
            }
        }

        const uint16_t *source = (const uint16_t*)input.cpuData;
        const float *convWeight = (const float*)weight.cpuData;
        uint16_t *destination = (uint16_t*)output.cpuData;
        KimiK3ParallelFor(batch * channels, [&](int start, int end) {
            for (int item = start; item < end; item++) {
                int batchIndex = item / channels;
                int channel = item % channels;
                for (int token = 0; token < sequence; token++) {
                    float value = 0.0f;
                    for (int kernel = 0; kernel < kernelSize; kernel++) {
                        int sourceToken = token - kernelSize + 1 + kernel;
                        if (sourceToken >= 0) {
                            uint64_t sourceIndex =
                                ((uint64_t)batchIndex * sequence + sourceToken) *
                                channels + channel;
                            value += BFloat16BitsToFloat32(source[sourceIndex]) *
                                     convWeight[(uint64_t)channel * kernelSize + kernel];
                        } else if (cache != nullptr) {
                            int cacheToken = kernelSize - 1 + sourceToken;
                            uint64_t cacheIndex =
                                ((uint64_t)batchIndex * (kernelSize - 1) +
                                 cacheToken) * channels + channel;
                            value += BFloat16BitsToFloat32(
                                ((const uint16_t*)cache->cpuData)[cacheIndex]) *
                                convWeight[(uint64_t)channel * kernelSize + kernel];
                        }
                    }
                    float activated = value * KimiK3Sigmoid(value);
                    uint64_t outputIndex =
                        ((uint64_t)batchIndex * sequence + token) * channels + channel;
                    destination[outputIndex] = Float32ToBFloat16RNEBits(activated);
                }
            }
        });
        if (cache != nullptr) {
            uint16_t *cacheData = (uint16_t*)cache->cpuData;
            int history = kernelSize - 1;
            KimiK3ParallelFor(batch * channels, [&](int start, int end) {
                std::vector<uint16_t> previous(history);
                for (int item = start; item < end; item++) {
                    int batchIndex = item / channels;
                    int channel = item % channels;
                    for (int token = 0; token < history; token++) {
                        previous[token] = cacheData[
                            ((uint64_t)batchIndex * history + token) *
                            channels + channel];
                    }
                    for (int token = 0; token < history; token++) {
                        int combined = sequence + token;
                        uint16_t value;
                        if (combined < history) {
                            value = previous[combined];
                        } else {
                            int inputToken = combined - history;
                            value = source[
                                ((uint64_t)batchIndex * sequence + inputToken) *
                                channels + channel];
                        }
                        cacheData[
                            ((uint64_t)batchIndex * history + token) *
                            channels + channel] = value;
                    }
                }
            });
        }
    }

    bool CpuKimiK3UpdatePackedConvCacheOp::CanRun(
            const std::string &, const DataDict &datas,
            const FloatDict &, const IntDict &) {
        auto q = datas.find("q");
        return q != datas.end() && q->second != nullptr &&
               q->second->dataType == DataType::BFLOAT16;
    }

    void CpuKimiK3UpdatePackedConvCacheOp::Reshape(
            const std::string &, const DataDict &,
            const FloatDict &, const IntDict &) {
    }

    void CpuKimiK3UpdatePackedConvCacheOp::Run(
            const std::string &, const DataDict &datas,
            const FloatDict &, const IntDict &intParams) {
        Data &q = *datas.find("q")->second;
        Data &k = *datas.find("k")->second;
        Data &v = *datas.find("v")->second;
        Data &cache = *datas.find("cache")->second;
        const int history = intParams.find("history")->second;
        const int tokens = intParams.find("tokens")->second;
        KimiK3AssertBFloat16(q, "KimiK3UpdatePackedConvCache q");
        KimiK3AssertBFloat16(k, "KimiK3UpdatePackedConvCache k");
        KimiK3AssertBFloat16(v, "KimiK3UpdatePackedConvCache v");
        KimiK3AssertBFloat16(cache, "KimiK3UpdatePackedConvCache cache");
        AssertInFastLLM(
            q.dims.size() == 3 && q.dims == k.dims && q.dims == v.dims &&
            history > 0 && tokens > 0 && tokens <= q.dims[1],
            "KimiK3UpdatePackedConvCache input shape mismatch.");
        const int batch = q.dims[0];
        const int sequence = q.dims[1];
        const int channels = q.dims[2];
        const bool legacyLayout =
            batch == 1 && cache.dims ==
                std::vector<int>({3, history, channels});
        const bool batchedLayout =
            cache.dims ==
                std::vector<int>({3, batch, history, channels});
        AssertInFastLLM(
            legacyLayout || batchedLayout,
            "KimiK3UpdatePackedConvCache cache shape mismatch.");
        AssertInFastLLM(cache.cpuData != nullptr,
                        "KimiK3UpdatePackedConvCache cache is uninitialized.");

        const uint16_t *sources[3] = {
            (const uint16_t*)q.cpuData,
            (const uint16_t*)k.cpuData,
            (const uint16_t*)v.cpuData,
        };
        uint16_t *cacheData = (uint16_t*)cache.cpuData;
        KimiK3ParallelFor(3 * batch * channels,
                          [&](int start, int end) {
            for (int item = start; item < end; item++) {
                const int stream = item / (batch * channels);
                const int withinStream = item % (batch * channels);
                const int batchIndex = withinStream / channels;
                const int channel = withinStream % channels;
                const uint16_t *source = sources[stream];
                const uint64_t cacheBase =
                    ((uint64_t)stream * batch + batchIndex) *
                    history * channels;
                for (int slot = 0; slot < history; slot++) {
                    const int combined = tokens + slot;
                    uint16_t value;
                    if (combined < history) {
                        value = cacheData[
                            cacheBase + (uint64_t)combined * channels +
                            channel];
                    } else {
                        const int inputToken = combined - history;
                        value = source[
                            ((uint64_t)batchIndex * sequence + inputToken) *
                            channels + channel];
                    }
                    cacheData[
                        cacheBase + (uint64_t)slot * channels + channel] =
                        value;
                }
            }
        });
    }

    bool CpuKimiK3L2NormOp::CanRun(
            const std::string &, const DataDict &datas,
            const FloatDict &, const IntDict &) {
        return datas.find("input") != datas.end() &&
               datas.find("input")->second->dataType == DataType::BFLOAT16;
    }

    void CpuKimiK3L2NormOp::Reshape(
            const std::string &, const DataDict &datas,
            const FloatDict &, const IntDict &) {
        KimiK3ReshapeLike(*datas.find("input")->second,
                          *datas.find("output")->second,
                          DataType::BFLOAT16);
    }

    void CpuKimiK3L2NormOp::Run(
            const std::string &, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &) {
        Data &input = *datas.find("input")->second;
        Data &output = *datas.find("output")->second;
        float eps = floatParams.find("eps") == floatParams.end() ?
                    1e-6f : floatParams.find("eps")->second;
        KimiK3AssertBFloat16(input, "KimiK3L2Norm input");
        int channels = input.dims.back();
        int rows = (int)(input.Count(0) / channels);
        output.Allocate(false);
        const uint16_t *source = (const uint16_t*)input.cpuData;
        uint16_t *destination = (uint16_t*)output.cpuData;
        KimiK3ParallelFor(rows, [&](int start, int end) {
            for (int row = start; row < end; row++) {
                uint64_t base = (uint64_t)row * channels;
                float squareSum = 0.0f;
                for (int channel = 0; channel < channels; channel++) {
                    float value = BFloat16BitsToFloat32(source[base + channel]);
                    squareSum += value * value;
                }
                float scale = 1.0f / std::sqrt(squareSum + eps);
                for (int channel = 0; channel < channels; channel++) {
                    destination[base + channel] = Float32ToBFloat16RNEBits(
                        BFloat16BitsToFloat32(source[base + channel]) * scale);
                }
            }
        });
    }

    bool CpuKimiK3RecurrentKDAOp::CanRun(
            const std::string &, const DataDict &datas,
            const FloatDict &, const IntDict &) {
        return datas.find("q") != datas.end() &&
               datas.find("q")->second->dataType == DataType::BFLOAT16;
    }

    void CpuKimiK3RecurrentKDAOp::Reshape(
            const std::string &, const DataDict &datas,
            const FloatDict &, const IntDict &intParams) {
        Data &q = *datas.find("q")->second;
        AssertInFastLLM(q.dims.size() == 4,
                        "KimiK3RecurrentKDA q must be [batch, sequence, heads, dim].");
        int batch = q.dims[0];
        int sequence = q.dims[1];
        int heads = q.dims[2];
        int dimension = q.dims[3];
        const bool stateOnly =
            intParams.find("stateOnly") != intParams.end() &&
            intParams.find("stateOnly")->second != 0;
        const bool outputAux =
            intParams.find("outputAux") == intParams.end() ||
            intParams.find("outputAux")->second != 0;
        if (!stateOnly) {
            KimiK3ReshapeLike(q, *datas.find("output")->second,
                              DataType::BFLOAT16);
        }
        Data &state = *datas.find("state")->second;
        state.dataType = DataType::FLOAT32;
        state.Resize({batch, heads, dimension, dimension});
        if (!stateOnly && outputAux) {
            Data &decay = *datas.find("decay")->second;
            decay.dataType = DataType::FLOAT32;
            decay.Resize(q.dims);
            Data &beta = *datas.find("beta")->second;
            beta.dataType = DataType::FLOAT32;
            beta.Resize({batch, sequence, heads});
        }
    }

    void CpuKimiK3RecurrentKDAOp::Run(
            const std::string &, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &q = *datas.find("q")->second;
        Data &k = *datas.find("k")->second;
        Data &v = *datas.find("v")->second;
        Data &rawGate = *datas.find("rawGate")->second;
        Data &rawBeta = *datas.find("rawBeta")->second;
        Data &aLog = *datas.find("aLog")->second;
        Data &dtBias = *datas.find("dtBias")->second;
        Data &state = *datas.find("state")->second;
        Data &output = *datas.find("output")->second;
        Data &decay = *datas.find("decay")->second;
        Data &activatedBeta = *datas.find("beta")->second;
        float lowerBound = floatParams.find("lowerBound") == floatParams.end() ?
                           -5.0f : floatParams.find("lowerBound")->second;

        AssertInFastLLM(q.dims == k.dims && q.dims == v.dims &&
                        q.dims == rawGate.dims && q.dims.size() == 4,
                        "KimiK3RecurrentKDA q/k/v/g shape mismatch.");
        KimiK3AssertBFloat16(q, "KimiK3RecurrentKDA q");
        KimiK3AssertBFloat16(k, "KimiK3RecurrentKDA k");
        KimiK3AssertBFloat16(v, "KimiK3RecurrentKDA v");
        KimiK3AssertBFloat16(rawGate, "KimiK3RecurrentKDA raw gate");
        KimiK3AssertFloat32(rawBeta, "KimiK3RecurrentKDA raw beta");
        KimiK3AssertFloat32(aLog, "KimiK3RecurrentKDA A_log");
        KimiK3AssertFloat32(dtBias, "KimiK3RecurrentKDA dt_bias");
        int batch = q.dims[0];
        int sequence = q.dims[1];
        int heads = q.dims[2];
        int dimension = q.dims[3];
        const bool stateOnly =
            intParams.find("stateOnly") != intParams.end() &&
            intParams.find("stateOnly")->second != 0;
        const bool outputAux =
            intParams.find("outputAux") == intParams.end() ||
            intParams.find("outputAux")->second != 0;
        const int requestedTokens =
            intParams.find("tokenLimit") == intParams.end() ? -1 :
            intParams.find("tokenLimit")->second;
        const int tokenCount = requestedTokens < 0 ?
            sequence : requestedTokens;
        AssertInFastLLM(
            tokenCount > 0 && tokenCount <= sequence &&
            (stateOnly || tokenCount == sequence),
            "KimiK3RecurrentKDA token limit is invalid.");
        AssertInFastLLM(rawBeta.dims == std::vector<int>({batch, sequence, heads}),
                        "KimiK3RecurrentKDA beta shape mismatch.");
        bool aLogPerHead = aLog.Count(0) == (uint64_t)heads;
        bool aLogPerChannel = aLog.Count(0) == (uint64_t)dimension;
        AssertInFastLLM(aLogPerHead || aLogPerChannel,
                        "KimiK3RecurrentKDA A_log must have num_heads or head_dim values.");
        AssertInFastLLM(dtBias.Count(0) == (uint64_t)heads * dimension,
                        "KimiK3RecurrentKDA dt_bias shape mismatch.");
        bool initializeState = state.cpuData == nullptr;
        state.Allocate(false);
        if (initializeState) {
            std::fill((float*)state.cpuData,
                      (float*)state.cpuData + state.Count(0), 0.0f);
        }
        if (!stateOnly) {
            output.Allocate(false);
            if (outputAux) {
                decay.Allocate(false);
                activatedBeta.Allocate(false);
            }
        }

        const uint16_t *qData = (const uint16_t*)q.cpuData;
        const uint16_t *kData = (const uint16_t*)k.cpuData;
        const uint16_t *vData = (const uint16_t*)v.cpuData;
        const uint16_t *gateData = (const uint16_t*)rawGate.cpuData;
        const float *betaData = (const float*)rawBeta.cpuData;
        const float *aLogData = (const float*)aLog.cpuData;
        const float *dtBiasData = (const float*)dtBias.cpuData;
        float *stateData = (float*)state.cpuData;
        uint16_t *outputData = stateOnly ? nullptr :
            (uint16_t*)output.cpuData;
        float *decayData = stateOnly || !outputAux ? nullptr :
            (float*)decay.cpuData;
        float *activatedBetaData = stateOnly || !outputAux ? nullptr :
            (float*)activatedBeta.cpuData;
        float outputScale = 1.0f / std::sqrt((float)dimension);

        KimiK3ParallelFor(batch * heads, [&](int start, int end) {
            std::vector<float> key(dimension);
            std::vector<float> query(dimension);
            std::vector<float> delta(dimension);
            for (int item = start; item < end; item++) {
                int batchIndex = item / heads;
                int head = item % heads;
                float *headState = stateData +
                    ((uint64_t)batchIndex * heads + head) * dimension * dimension;
                for (int token = 0; token < tokenCount; token++) {
                    uint64_t vectorBase =
                        (((uint64_t)batchIndex * sequence + token) * heads + head) *
                        dimension;
                    uint64_t betaIndex =
                        ((uint64_t)batchIndex * sequence + token) * heads + head;
                    float beta = KimiK3Sigmoid(betaData[betaIndex]);
                    if (activatedBetaData != nullptr) {
                        activatedBetaData[betaIndex] = beta;
                    }
                    for (int channel = 0; channel < dimension; channel++) {
                        key[channel] = BFloat16BitsToFloat32(kData[vectorBase + channel]);
                        if (!stateOnly) {
                            query[channel] =
                                BFloat16BitsToFloat32(
                                    qData[vectorBase + channel]);
                        }
                        float raw = BFloat16BitsToFloat32(gateData[vectorBase + channel]);
                        float a = aLogPerHead ? aLogData[head] : aLogData[channel];
                        float gate = lowerBound * KimiK3Sigmoid(
                            std::exp(a) *
                            (raw + dtBiasData[(uint64_t)head * dimension + channel]));
                        if (decayData != nullptr) {
                            decayData[vectorBase + channel] = gate;
                        }
                        float retention = std::exp(gate);
                        float *stateRow = headState + (uint64_t)channel * dimension;
                        for (int value = 0; value < dimension; value++) {
                            stateRow[value] *= retention;
                        }
                    }
                    for (int value = 0; value < dimension; value++) {
                        float prediction = 0.0f;
                        for (int channel = 0; channel < dimension; channel++) {
                            prediction += key[channel] *
                                headState[(uint64_t)channel * dimension + value];
                        }
                        delta[value] =
                            (BFloat16BitsToFloat32(vData[vectorBase + value]) -
                             prediction) * beta;
                    }
                    for (int channel = 0; channel < dimension; channel++) {
                        float *stateRow = headState + (uint64_t)channel * dimension;
                        for (int value = 0; value < dimension; value++) {
                            stateRow[value] += key[channel] * delta[value];
                        }
                    }
                    if (!stateOnly) {
                        for (int value = 0; value < dimension; value++) {
                            float result = 0.0f;
                            for (int channel = 0; channel < dimension;
                                 channel++) {
                                result += query[channel] *
                                    headState[(uint64_t)channel * dimension +
                                              value];
                            }
                            outputData[vectorBase + value] =
                                Float32ToBFloat16RNEBits(
                                    result * outputScale);
                        }
                    }
                }
            }
        });
    }

    bool CpuKimiK3RMSNormSigmoidGateOp::CanRun(
            const std::string &, const DataDict &datas,
            const FloatDict &, const IntDict &) {
        return datas.find("input") != datas.end() &&
               datas.find("input")->second->dataType == DataType::BFLOAT16;
    }

    void CpuKimiK3RMSNormSigmoidGateOp::Reshape(
            const std::string &, const DataDict &datas,
            const FloatDict &, const IntDict &) {
        KimiK3ReshapeLike(*datas.find("input")->second,
                          *datas.find("output")->second,
                          DataType::BFLOAT16);
    }

    void CpuKimiK3RMSNormSigmoidGateOp::Run(
            const std::string &, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &) {
        Data &input = *datas.find("input")->second;
        Data &gate = *datas.find("gate")->second;
        Data &weight = *datas.find("weight")->second;
        Data &output = *datas.find("output")->second;
        float eps = floatParams.find("eps") == floatParams.end() ?
                    1e-5f : floatParams.find("eps")->second;
        AssertInFastLLM(input.dims == gate.dims,
                        "KimiK3RMSNormSigmoidGate input/gate shape mismatch.");
        KimiK3AssertBFloat16(input, "KimiK3RMSNormSigmoidGate input");
        KimiK3AssertBFloat16(gate, "KimiK3RMSNormSigmoidGate gate");
        KimiK3AssertFloat32(weight, "KimiK3RMSNormSigmoidGate weight");
        int dimension = input.dims.back();
        int rows = (int)(input.Count(0) / dimension);
        AssertInFastLLM(weight.Count(0) == (uint64_t)dimension,
                        "KimiK3RMSNormSigmoidGate weight shape mismatch.");
        output.Allocate(false);

        const uint16_t *source = (const uint16_t*)input.cpuData;
        const uint16_t *gateData = (const uint16_t*)gate.cpuData;
        const float *normWeight = (const float*)weight.cpuData;
        uint16_t *destination = (uint16_t*)output.cpuData;
        KimiK3ParallelFor(rows, [&](int start, int end) {
            for (int row = start; row < end; row++) {
                uint64_t base = (uint64_t)row * dimension;
                float squareSum = 0.0f;
                for (int channel = 0; channel < dimension; channel++) {
                    float value = BFloat16BitsToFloat32(source[base + channel]);
                    squareSum += value * value;
                }
                float scale = 1.0f / std::sqrt(squareSum / dimension + eps);
                for (int channel = 0; channel < dimension; channel++) {
                    float value = BFloat16BitsToFloat32(source[base + channel]);
                    float gateValue = BFloat16BitsToFloat32(gateData[base + channel]);
                    destination[base + channel] = Float32ToBFloat16RNEBits(
                        value * scale * normWeight[channel] * KimiK3Sigmoid(gateValue));
                }
            }
        });
    }

    bool CpuKimiK3AttnResOp::CanRun(
            const std::string &, const DataDict &datas,
            const FloatDict &, const IntDict &) {
        return datas.find("prefixSum") != datas.end() &&
               datas.find("prefixSum")->second->dataType == DataType::BFLOAT16;
    }

    void CpuKimiK3AttnResOp::Reshape(
            const std::string &, const DataDict &datas,
            const FloatDict &, const IntDict &) {
        KimiK3ReshapeLike(*datas.find("prefixSum")->second,
                          *datas.find("output")->second,
                          DataType::BFLOAT16);
    }

    void CpuKimiK3AttnResOp::Run(
            const std::string &, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &) {
        Data &prefixSum = *datas.find("prefixSum")->second;
        Data &blockResidual = *datas.find("blockResidual")->second;
        Data &projection = *datas.find("projection")->second;
        Data &norm = *datas.find("norm")->second;
        Data &output = *datas.find("output")->second;
        float eps = floatParams.find("eps") == floatParams.end() ?
                    1e-5f : floatParams.find("eps")->second;
        KimiK3AssertBFloat16(prefixSum, "KimiK3AttnRes prefix sum");
        KimiK3AssertBFloat16(blockResidual, "KimiK3AttnRes block residual");
        KimiK3AssertFloat32(projection, "KimiK3AttnRes projection");
        KimiK3AssertFloat32(norm, "KimiK3AttnRes norm");
        int dimension = prefixSum.dims.back();
        int rows = (int)(prefixSum.Count(0) / dimension);
        AssertInFastLLM(blockResidual.dims.size() == 3 &&
                        blockResidual.dims[0] == rows &&
                        blockResidual.dims[2] == dimension,
                        "KimiK3AttnRes block residual must be "
                        "[tokens, blocks, hidden].");
        int blocks = blockResidual.dims[1];
        AssertInFastLLM(projection.Count(0) == (uint64_t)dimension &&
                        norm.Count(0) == (uint64_t)dimension && blocks > 0,
                        "KimiK3AttnRes parameter shape mismatch.");
        output.Allocate(false);

        const uint16_t *prefixData = (const uint16_t*)prefixSum.cpuData;
        const uint16_t *residualData = (const uint16_t*)blockResidual.cpuData;
        const float *projectionData = (const float*)projection.cpuData;
        const float *normData = (const float*)norm.cpuData;
        uint16_t *destination = (uint16_t*)output.cpuData;
        KimiK3ParallelFor(rows, [&](int start, int end) {
            std::vector<float> scores((size_t)blocks + 1);
            std::vector<float> probabilities((size_t)blocks + 1);
            for (int row = start; row < end; row++) {
                uint64_t base = (uint64_t)row * dimension;
                float prefixSquares = 0.0f;
                for (int channel = 0; channel < dimension; channel++) {
                    float prefix = BFloat16BitsToFloat32(prefixData[base + channel]);
                    prefixSquares += prefix * prefix;
                }
                float prefixScale =
                    1.0f / std::sqrt(prefixSquares / dimension + eps);
                float prefixScore = 0.0f;
                for (int channel = 0; channel < dimension; channel++) {
                    float scoreWeight = normData[channel] * projectionData[channel];
                    prefixScore += BFloat16BitsToFloat32(
                        prefixData[base + channel]) * prefixScale * scoreWeight;
                }
                scores[blocks] = prefixScore;
                float maximum = prefixScore;
                for (int block = 0; block < blocks; block++) {
                    uint64_t residualBase =
                        ((uint64_t)row * blocks + block) * dimension;
                    float residualSquares = 0.0f;
                    for (int channel = 0; channel < dimension; channel++) {
                        float residual = BFloat16BitsToFloat32(
                            residualData[residualBase + channel]);
                        residualSquares += residual * residual;
                    }
                    float residualScale =
                        1.0f / std::sqrt(residualSquares / dimension + eps);
                    float residualScore = 0.0f;
                    for (int channel = 0; channel < dimension; channel++) {
                        float scoreWeight = normData[channel] * projectionData[channel];
                        residualScore += BFloat16BitsToFloat32(
                            residualData[residualBase + channel]) *
                            residualScale * scoreWeight;
                    }
                    scores[block] = residualScore;
                    maximum = std::max(maximum, residualScore);
                }
                if (blocks == 1) {
                    // Keep the established two-way reduction exactly stable
                    // for layers 0..11; the first-four-layer reference was
                    // aligned against this operation order.
                    float prefixExp = std::exp(prefixScore - maximum);
                    float residualExp = std::exp(scores[0] - maximum);
                    float residualProbability =
                        residualExp / (prefixExp + residualExp);
                    float prefixProbability = 1.0f - residualProbability;
                    for (int channel = 0; channel < dimension; channel++) {
                        float value = residualProbability *
                            BFloat16BitsToFloat32(
                                residualData[(uint64_t)row * dimension + channel]) +
                            prefixProbability * BFloat16BitsToFloat32(
                                prefixData[base + channel]);
                        destination[base + channel] =
                            Float32ToBFloat16RNEBits(value);
                    }
                    continue;
                }
                float denominator = 0.0f;
                for (int item = 0; item <= blocks; item++) {
                    probabilities[item] = std::exp(scores[item] - maximum);
                    denominator += probabilities[item];
                }
                for (float &probability : probabilities) {
                    probability /= denominator;
                }
                for (int channel = 0; channel < dimension; channel++) {
                    float value = probabilities[blocks] *
                        BFloat16BitsToFloat32(prefixData[base + channel]);
                    for (int block = 0; block < blocks; block++) {
                        uint64_t residualIndex =
                            ((uint64_t)row * blocks + block) * dimension + channel;
                        value += probabilities[block] *
                            BFloat16BitsToFloat32(residualData[residualIndex]);
                    }
                    destination[base + channel] = Float32ToBFloat16RNEBits(value);
                }
            }
        });
    }

    bool CpuKimiK3SiTUAndMulOp::CanRun(
            const std::string &, const DataDict &datas,
            const FloatDict &, const IntDict &) {
        return datas.find("gate") != datas.end() &&
               datas.find("gate")->second->dataType == DataType::BFLOAT16;
    }

    void CpuKimiK3SiTUAndMulOp::Reshape(
            const std::string &, const DataDict &datas,
            const FloatDict &, const IntDict &) {
        KimiK3ReshapeLike(*datas.find("gate")->second,
                          *datas.find("output")->second,
                          DataType::BFLOAT16);
    }

    void CpuKimiK3SiTUAndMulOp::Run(
            const std::string &, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &) {
        Data &gate = *datas.find("gate")->second;
        Data &up = *datas.find("up")->second;
        Data &output = *datas.find("output")->second;
        float beta = floatParams.find("beta") == floatParams.end() ?
                     1.0f : floatParams.find("beta")->second;
        float linearBeta = floatParams.find("linearBeta") == floatParams.end() ?
                           0.0f : floatParams.find("linearBeta")->second;
        AssertInFastLLM(gate.dims == up.dims,
                        "KimiK3SiTUAndMul gate/up shape mismatch.");
        KimiK3AssertBFloat16(gate, "KimiK3SiTUAndMul gate");
        KimiK3AssertBFloat16(up, "KimiK3SiTUAndMul up");
        AssertInFastLLM(beta > 0.0f,
                        "KimiK3SiTUAndMul beta must be positive.");
        output.Allocate(false);
        const uint16_t *gateData = (const uint16_t*)gate.cpuData;
        const uint16_t *upData = (const uint16_t*)up.cpuData;
        uint16_t *destination = (uint16_t*)output.cpuData;
        int count = (int)gate.Count(0);
        KimiK3ParallelFor(count, [&](int start, int end) {
            for (int index = start; index < end; index++) {
                float gateValue = BFloat16BitsToFloat32(gateData[index]);
                float upValue = BFloat16BitsToFloat32(upData[index]);
                float situ = beta * std::tanh(gateValue / beta) *
                             KimiK3Sigmoid(gateValue);
                float boundedUp = linearBeta > 0.0f ?
                    linearBeta * std::tanh(upValue / linearBeta) : upValue;
                destination[index] =
                    Float32ToBFloat16RNEBits(situ * boundedUp);
            }
        });
    }

    bool CpuKimiK3RoutedExpertsOp::CanRun(
            const std::string &, const DataDict &datas,
            const FloatDict &, const IntDict &) {
        return datas.find("input") != datas.end() &&
               datas.find("input")->second->dataType == DataType::BFLOAT16;
    }

    void CpuKimiK3RoutedExpertsOp::Reshape(
            const std::string &, const DataDict &datas,
            const FloatDict &, const IntDict &) {
        KimiK3ReshapeLike(*datas.find("input")->second,
                          *datas.find("output")->second,
                          DataType::BFLOAT16);
    }

    void CpuKimiK3RoutedExpertsOp::Run(
            const std::string &, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *datas.find("input")->second;
        Data &index = *datas.find("index")->second;
        Data &score = *datas.find("score")->second;
        Data &output = *datas.find("output")->second;
        Data **w1s = (Data**)datas.find("w1s")->second;
        Data **w2s = (Data**)datas.find("w2s")->second;
        Data **w3s = (Data**)datas.find("w3s")->second;
        int expertCount = intParams.find("experts___batch")->second;
        float beta = floatParams.find("beta") == floatParams.end() ?
                     1.0f : floatParams.find("beta")->second;
        float linearBeta = floatParams.find("linearBeta") == floatParams.end() ?
                           0.0f : floatParams.find("linearBeta")->second;

        KimiK3AssertBFloat16(input, "KimiK3RoutedExperts input");
        AssertInFastLLM(input.dims.size() == 2 && index.dims.size() == 2 &&
                        score.dims == index.dims &&
                        index.dataType == DataType::INT32 &&
                        score.dataType == DataType::FLOAT32 &&
                        index.dims[0] == input.dims[0] &&
                        expertCount > 0 && beta > 0.0f,
                        "KimiK3RoutedExperts input/index/score shape mismatch.");
        int tokens = input.dims[0];
        int inputDimension = input.dims[1];
        int topk = index.dims[1];
        AssertInFastLLM(w1s[0] != nullptr && w2s[0] != nullptr &&
                        w3s[0] != nullptr &&
                        w1s[0]->dims.size() == 2 &&
                        w2s[0]->dims.size() == 2 &&
                        w3s[0]->dims.size() == 2,
                        "KimiK3RoutedExperts has an invalid weight table.");
        int intermediateDimension = w1s[0]->dims[0];
        int outputDimension = w2s[0]->dims[0];
        AssertInFastLLM(w1s[0]->dims[1] == inputDimension &&
                        w3s[0]->dims == w1s[0]->dims &&
                        w2s[0]->dims[1] == intermediateDimension &&
                        outputDimension == inputDimension,
                        "KimiK3RoutedExperts expert shape mismatch.");

        const int32_t *indexData = (const int32_t*)index.cpuData;
        const float *scoreData = (const float*)score.cpuData;
        const uint16_t *inputData = (const uint16_t*)input.cpuData;
        std::vector<std::vector<std::pair<int, int>>> routes(expertCount);
        for (int token = 0; token < tokens; token++) {
            for (int slot = 0; slot < topk; slot++) {
                int expert = indexData[(uint64_t)token * topk + slot];
                AssertInFastLLM(expert >= 0 && expert < expertCount,
                                "KimiK3RoutedExperts expert index is out of range.");
                routes[expert].emplace_back(token, slot);
            }
        }

        // Preserve the reference reduction order by materializing each routed
        // BF16 expert result in its original top-k slot, then reducing slots
        // from left to right after all expert batches have completed.
        std::vector<uint16_t> routedOutputs(
            (uint64_t)tokens * topk * outputDimension, 0);
        AliveThreadPool *pool = GetAlivePool();
        int firstThread = pool->curActivateThreadInterval.first;
        int threadCount = std::max(
            1, pool->curActivateThreadInterval.second - firstThread);

        for (int expert = 0; expert < expertCount; expert++) {
            int routeCount = (int)routes[expert].size();
            if (routeCount == 0) {
                continue;
            }
            Data &w1 = *w1s[expert];
            Data &w2 = *w2s[expert];
            Data &w3 = *w3s[expert];
            AssertInFastLLM(KimiK3SupportedExpertWeight(w1) &&
                            KimiK3SupportedExpertWeight(w2) &&
                            KimiK3SupportedExpertWeight(w3) &&
                            w1.dataType == w2.dataType &&
                            w1.dataType == w3.dataType &&
                            w1.ggmlType == w2.ggmlType &&
                            w1.ggmlType == w3.ggmlType &&
                            w1.dims == w1s[0]->dims &&
                            w2.dims == w2s[0]->dims &&
                            w3.dims == w3s[0]->dims,
                            "KimiK3RoutedExperts requires uniform compact "
                            "MXFP4 or GGML Q2_K experts.");

            std::vector<uint16_t> gathered(
                (uint64_t)routeCount * inputDimension);
            for (int row = 0; row < routeCount; row++) {
                int token = routes[expert][row].first;
                std::memcpy(gathered.data() + (uint64_t)row * inputDimension,
                            inputData + (uint64_t)token * inputDimension,
                            (uint64_t)inputDimension * sizeof(uint16_t));
            }

            std::vector<float> gate((uint64_t)routeCount * intermediateDimension);
            std::vector<float> up((uint64_t)routeCount * intermediateDimension);
            KimiK3RunExpertLinear(
                gathered.data(), w1, gate.data(),
                routeCount, inputDimension, intermediateDimension,
                pool, firstThread, threadCount);
            KimiK3RunExpertLinear(
                gathered.data(), w3, up.data(),
                routeCount, inputDimension, intermediateDimension,
                pool, firstThread, threadCount);

            std::vector<uint16_t> activated(
                (uint64_t)routeCount * intermediateDimension);
            int activationCount = routeCount * intermediateDimension;
            KimiK3ParallelFor(activationCount, [&](int start, int end) {
                for (int i = start; i < end; i++) {
                    // nn.Linear returns BF16 for the checkpoint's BF16 input;
                    // SiTU therefore observes rounded w1/w3 results.
                    float gateValue = RoundFloat32ToBFloat16RNE(gate[i]);
                    float upValue = RoundFloat32ToBFloat16RNE(up[i]);
                    float situ = beta * std::tanh(gateValue / beta) *
                                 KimiK3Sigmoid(gateValue);
                    float boundedUp = linearBeta > 0.0f ?
                        linearBeta * std::tanh(upValue / linearBeta) : upValue;
                    activated[i] = Float32ToBFloat16RNEBits(
                        situ * boundedUp);
                }
            });

            std::vector<float> down((uint64_t)routeCount * outputDimension);
            KimiK3RunExpertLinear(
                activated.data(), w2, down.data(),
                routeCount, intermediateDimension, outputDimension,
                pool, firstThread, threadCount);
            for (int row = 0; row < routeCount; row++) {
                int token = routes[expert][row].first;
                int slot = routes[expert][row].second;
                uint16_t *destination = routedOutputs.data() +
                    ((uint64_t)token * topk + slot) * outputDimension;
                const float *source = down.data() +
                    (uint64_t)row * outputDimension;
                for (int channel = 0; channel < outputDimension; channel++) {
                    destination[channel] =
                        Float32ToBFloat16RNEBits(source[channel]);
                }
            }
        }

        output.Allocate(false);
        uint16_t *outputData = (uint16_t*)output.cpuData;
        KimiK3ParallelFor(tokens * outputDimension, [&](int start, int end) {
            for (int item = start; item < end; item++) {
                int token = item / outputDimension;
                int channel = item % outputDimension;
                float sum = 0.0f;
                for (int slot = 0; slot < topk; slot++) {
                    uint64_t routeIndex =
                        ((uint64_t)token * topk + slot) * outputDimension + channel;
                    sum += BFloat16BitsToFloat32(routedOutputs[routeIndex]) *
                           scoreData[(uint64_t)token * topk + slot];
                }
                outputData[item] = Float32ToBFloat16RNEBits(sum);
            }
        });
    }

    bool CpuKimiK3CausalAttentionOp::CanRun(
            const std::string &, const DataDict &datas,
            const FloatDict &, const IntDict &) {
        return datas.find("q") != datas.end() &&
               datas.find("q")->second->dataType == DataType::BFLOAT16;
    }

    void CpuKimiK3CausalAttentionOp::Reshape(
            const std::string &, const DataDict &datas,
            const FloatDict &, const IntDict &) {
        Data &q = *datas.find("q")->second;
        Data &v = *datas.find("v")->second;
        AssertInFastLLM(q.dims.size() == 4 &&
                        (v.dims.size() == 3 || v.dims.size() == 4),
                        "KimiK3CausalAttention expects a 4D query and a "
                        "3D FastLLM cache or 4D current value tensor.");
        Data &output = *datas.find("output")->second;
        output.dataType = DataType::BFLOAT16;
        output.Resize(
            {q.dims[0], q.dims[1], q.dims[2], v.dims.back()});
    }

    void CpuKimiK3CausalAttentionOp::Run(
            const std::string &, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &) {
        Data &q = *datas.find("q")->second;
        Data &k = *datas.find("k")->second;
        Data &v = *datas.find("v")->second;
        Data &output = *datas.find("output")->second;
        float scale = floatParams.find("scale") == floatParams.end() ?
                      1.0f : floatParams.find("scale")->second;
        KimiK3AssertBFloat16(q, "KimiK3CausalAttention q");
        KimiK3AssertBFloat16(k, "KimiK3CausalAttention k");
        KimiK3AssertBFloat16(v, "KimiK3CausalAttention v");
        bool standardCache = k.dims.size() == 3 && v.dims.size() == 3;
        bool currentTensors = k.dims.size() == 4 && v.dims.size() == 4;
        AssertInFastLLM(
            q.dims.size() == 4 && (standardCache || currentTensors),
            "KimiK3CausalAttention q/k/v rank mismatch.");
        int batch = q.dims[0];
        int heads = q.dims[1];
        int queryLength = q.dims[2];
        int keyLength = standardCache ? k.dims[1] : k.dims[2];
        int queryDimension = q.dims[3];
        int valueDimension = v.dims.back();
        if (standardCache) {
            AssertInFastLLM(
                batch == 1 && k.dims[0] == heads && v.dims[0] == heads &&
                k.dims[1] == v.dims[1] &&
                k.dims[2] == queryDimension,
                "KimiK3CausalAttention standard cache shape mismatch.");
        } else {
            AssertInFastLLM(
                q.dims[0] == k.dims[0] && q.dims[0] == v.dims[0] &&
                q.dims[1] == k.dims[1] && q.dims[1] == v.dims[1] &&
                k.dims[2] == v.dims[2] &&
                k.dims[3] == queryDimension,
                "KimiK3CausalAttention current tensor shape mismatch.");
        }
        AssertInFastLLM(keyLength >= queryLength,
                        "KimiK3CausalAttention key length is too short.");
        output.Allocate(false);
        const uint16_t *qData = (const uint16_t*)q.cpuData;
        const uint16_t *kData = (const uint16_t*)k.cpuData;
        const uint16_t *vData = (const uint16_t*)v.cpuData;
        uint16_t *outputData = (uint16_t*)output.cpuData;
        int causalOffset = keyLength - queryLength;

        KimiK3ParallelFor(batch * heads * queryLength,
                          [&](int start, int end) {
            std::vector<float> scores(keyLength);
            std::vector<uint16_t> probabilities(keyLength);
            for (int item = start; item < end; item++) {
                int queryIndex = item % queryLength;
                int headItem = item / queryLength;
                int head = headItem % heads;
                int batchIndex = headItem / heads;
                uint64_t qBase =
                    (uint64_t)batchIndex * q.strides[0] +
                    (uint64_t)head * q.strides[1] +
                    (uint64_t)queryIndex * q.strides[2];
                int lastKey = causalOffset + queryIndex;
                float maximum = -std::numeric_limits<float>::infinity();
                for (int keyIndex = 0; keyIndex <= lastKey; keyIndex++) {
                    uint64_t kBase = standardCache ?
                        (uint64_t)head * k.strides[0] +
                            (uint64_t)keyIndex * k.strides[1] :
                        (uint64_t)batchIndex * k.strides[0] +
                            (uint64_t)head * k.strides[1] +
                            (uint64_t)keyIndex * k.strides[2];
                    float dot = 0.0f;
                    for (int channel = 0; channel < queryDimension; channel++) {
                        dot += BFloat16BitsToFloat32(
                                   qData[qBase +
                                         (uint64_t)channel * q.strides[3]]) *
                               BFloat16BitsToFloat32(
                                   kData[kBase + (uint64_t)channel *
                                       k.strides.back()]);
                    }
                    // eager_attention_forward forms the einsum result and the
                    // scaled score in BF16 before its FP32 softmax.
                    float dotBf16 = RoundFloat32ToBFloat16RNE(dot);
                    scores[keyIndex] = RoundFloat32ToBFloat16RNE(
                        dotBf16 * scale);
                    maximum = std::max(maximum, scores[keyIndex]);
                }
                float denominator = 0.0f;
                for (int keyIndex = 0; keyIndex <= lastKey; keyIndex++) {
                    scores[keyIndex] = std::exp(scores[keyIndex] - maximum);
                    denominator += scores[keyIndex];
                }
                for (int keyIndex = 0; keyIndex <= lastKey; keyIndex++) {
                    probabilities[keyIndex] = Float32ToBFloat16RNEBits(
                        scores[keyIndex] / denominator);
                }
                uint64_t outputBase =
                    (uint64_t)batchIndex * output.strides[0] +
                    (uint64_t)head * output.strides[1] +
                    (uint64_t)queryIndex * output.strides[2];
                for (int channel = 0; channel < valueDimension; channel++) {
                    float value = 0.0f;
                    for (int keyIndex = 0; keyIndex <= lastKey; keyIndex++) {
                        uint64_t vIndex = standardCache ?
                            (uint64_t)head * v.strides[0] +
                                (uint64_t)keyIndex * v.strides[1] +
                                (uint64_t)channel * v.strides[2] :
                            (uint64_t)batchIndex * v.strides[0] +
                                (uint64_t)head * v.strides[1] +
                                (uint64_t)keyIndex * v.strides[2] +
                                (uint64_t)channel * v.strides[3];
                        value += BFloat16BitsToFloat32(probabilities[keyIndex]) *
                                 BFloat16BitsToFloat32(vData[vIndex]);
                    }
                    outputData[outputBase +
                               (uint64_t)channel * output.strides[3]] =
                        Float32ToBFloat16RNEBits(value);
                }
            }
        });
    }

}
