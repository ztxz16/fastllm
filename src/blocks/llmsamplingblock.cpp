#include "baseblock.h"
#include "models/basellm.h"

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

namespace fastllm {
    void LLMSamplingBlock (
        basellm *model,
        Data *hiddenStates,
        Data *normWeight,
        Data *lmHeadWeight,
        float rms_norm_eps,
        int batch,
        bool all1,
        const std::vector<int> &seqLens,
        std::vector<std::pair<Data*, Data*>> &pastKeyValues,
        const std::vector<GenerationConfig> &generationConfigs,
        const LastTokensManager &lastTokens,
        std::vector<std::vector<float>*> *retLogits,
        std::vector<int> &lastRet
    ) {
        Data logits;
        std::vector<Data> curLogits;
        curLogits.resize(batch);

        if (!all1) {
            int total = 0;
            std::vector<Data> lastToks;
            std::vector<Data*> lastTokPointers;
            lastToks.resize(seqLens.size());
            for (int b = 0; b < (int)seqLens.size(); b++) {
                Split(*hiddenStates, 1, total + seqLens[b] - 1, total + seqLens[b], lastToks[b]);
                total += seqLens[b];
                lastTokPointers.push_back(&lastToks[b]);
            }
            CatBatch(lastTokPointers, 1, *hiddenStates);
        }

        RMSNorm(*hiddenStates, *normWeight, rms_norm_eps, *hiddenStates);
        Linear(*hiddenStates, *lmHeadWeight, *GetEmptyData(), logits);
        ToDataType(logits, DataType::FLOAT32);

        bool allSimple = true, needLogits = false;
        int maxTopK = 1;
        for (int b = 0; b < batch; b++) {
            if (!generationConfigs[b].IsSimpleGreedy()) {
                allSimple = false;
                break;
            }
        }
        for (int b = 0; b < batch; b++) {
            needLogits |= generationConfigs[b].output_logits;
            maxTopK = std::max(maxTopK, generationConfigs[b].top_k);
        }

        model->ResetLogitsOfEOS(batch, &logits, pastKeyValues, generationConfigs);

        if (allSimple) {
            Data topk;
            TopK(logits, topk, 1);
            topk.ToDevice(DataDevice::CPU);
            float *topkData = (float*)topk.cpuData;
            for (int b = 0; b < batch; b++) {
                lastRet.push_back((int) (topkData[0] + 1e-3));
                topkData += topk.Count(2);
            }
        } else if (!needLogits) {
            int maxTokenSetSize = 0;
            for (int b = 0; b < batch; b++) {
                maxTokenSetSize = std::max(maxTokenSetSize, (int)lastTokens.units[b].tokenSet.size());
            }
            std::vector<float> penaltyData(batch * maxTokenSetSize, -100.0f);
            std::vector<float> penaltyScaleData(batch, 1.0f);
            for (int b = 0; b < batch; b++) {
                int curId = 0;
                for (int i : lastTokens.units[b].tokenSet) {
                    penaltyData[b * maxTokenSetSize + curId] = i;
                    curId++;
                }
                penaltyScaleData[b] = generationConfigs[b].repeat_penalty;
            }
            Data penalty, penaltyScale;
            penalty.CopyFrom(Data(DataType::FLOAT32, {batch, maxTokenSetSize}, penaltyData));
            penaltyScale.CopyFrom(Data(DataType::FLOAT32, {batch}, penaltyScaleData));
            RepeatPenalty(logits, penalty, penaltyScale);
#ifdef USE_CUDA
            if (logits.dataDevice == DataDevice::CUDA) {
                int vocabSize = logits.dims.back();
                std::vector<int> topKArr(batch);
                std::vector<float> topPArr(batch), tempArr(batch);
                std::vector<int> outputIds(batch);
                for (int b = 0; b < batch; b++) {
                    topKArr[b] = generationConfigs[b].top_k;
                    topPArr[b] = generationConfigs[b].top_p;
                    tempArr[b] = generationConfigs[b].temperature;
                }
                FastllmCudaTopKTopPSampling(
                    (float *)logits.cudaData, tempArr.data(),
                    topKArr.data(), topPArr.data(),
                    outputIds.data(),
                    batch, vocabSize);
                for (int b = 0; b < batch; b++) {
                    lastRet.push_back(outputIds[b]);
                }
            } else
#endif
            {
                Data topk;
                TopK(logits, topk, maxTopK);
                topk.ToDevice(DataDevice::CPU);
                for (int b = 0; b < batch; b++) {
                    lastRet.push_back(LLMSamplingOnly(topk, b, generationConfigs[b]));
                }
            }
        } else {
            std::vector<Data*> pointersK(batch);
            for (int b = 0; b < batch; b++) {
                pointersK[b] = &curLogits[b];
            }
            SplitBatch(logits, 1, batch, pointersK);

            for (int b = 0; b < batch; b++) {
                Data &curLogit = curLogits[b];
                if (generationConfigs[b].output_logits && retLogits != nullptr && (*retLogits)[b] != nullptr) {
                    curLogit.ToDevice(DataDevice::CPU);
                    (*retLogits)[b]->resize(curLogit.Count(0));
                    memcpy((float*)(*retLogits)[b]->data(), (float*)curLogit.cpuData, curLogit.GetBytes());
                }
                if (generationConfigs[b].IsSimpleGreedy()) {
                    Data topk;
                    TopK(curLogit, topk, 1);
                    topk.ToDevice(DataDevice::CPU);
                    lastRet.push_back((int) (((float *) topk.cpuData)[0] + 1e-3));
                } else {
                    lastRet.push_back(LLMSampling(curLogit, 0, generationConfigs[b], lastTokens.units[b]));
                }
            }
        }
    }
}
