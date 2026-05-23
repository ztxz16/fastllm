#pragma once

#include "fastllm.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif

namespace fastllm {

#ifdef USE_CUDA

    inline void Qwen3CudaClearMultiDeviceState(Data &data) {
        for (auto &it : data.multiDeviceDatas) {
            delete it.second;
        }
        data.multiDeviceDatas.clear();
        data.multiDeviceData = false;
        data.ClearTensorParallelLayout();
    }

    inline void Qwen3CudaPrepareLocalOutput(Data &data, int device) {
        if (data.isFake) {
            data.isFake = false;
            data.cpuData = nullptr;
            data.cudaData = nullptr;
            data.deviceData = nullptr;
            data.expansionSize = 0;
            data.expansionBytes = 0;
        }

        bool needFree = false;
        if (data.dataDevice != DataDevice::CUDA) {
            needFree = data.cpuData != nullptr || data.cudaData != nullptr ||
                       data.deviceData != nullptr || data.expansionBytes != 0;
        } else if (!data.dataDeviceIds.empty() && data.dataDeviceIds[0] != device) {
            needFree = true;
        } else if (data.cudaData != nullptr) {
            int ptrDevice = GetPointerDeviceId(data.cudaData);
            needFree = ptrDevice >= 0 && ptrDevice != device;
        }
        if (needFree) {
            data.FreeSpace();
        }
        Qwen3CudaClearMultiDeviceState(data);
        data.dataDevice = DataDevice::CUDA;
        data.dataDeviceIds = {device};
        data.lockInCPU = false;
    }

    class Qwen3CudaDirectRunner {
    public:
        explicit Qwen3CudaDirectRunner(int deviceId) : deviceId(deviceId), device((BaseDevice*)&cudaDevice) {
            device->deviceIds = {deviceId};
        }

        int DeviceId() const {
            return deviceId;
        }

        void Run(const std::string &opType,
                 const DataDict &datas,
                 const FloatDict &floatParams = FloatDict(),
                 const IntDict &intParams = IntDict(),
                 const std::vector<std::string> &outputs = std::vector<std::string>(),
                 bool checkCanRun = true) {
            FastllmCudaSetDevice(deviceId);
            for (auto &name : outputs) {
                auto it = datas.find(name);
                if (it != datas.end() && it->second != nullptr) {
                    Qwen3CudaPrepareLocalOutput(*it->second, deviceId);
                }
            }
            if (checkCanRun) {
                AssertInFastLLM(device->CanRun(opType, datas, floatParams, intParams),
                                "Qwen3 CUDA direct runner can't run " + opType + ".\n");
            }
            device->Reshape(opType, datas, floatParams, intParams);
            device->Run(opType, datas, floatParams, intParams);
        }

    private:
        int deviceId;
        CudaDevice cudaDevice;
        BaseDevice *device;
    };

    namespace qwen3cuda {

    inline void Qwen3CudaRMSNorm(Qwen3CudaDirectRunner &runner,
                                 const Data &input, Data &weight,
                                 float eps, Data &output) {
        runner.Run("RMSNorm",
                   DataDict{{"input", (Data*)&input}, {"weight", &weight}, {"output", &output}},
                   FloatDict{{"eps", eps}}, IntDict(), {"output"});
    }

    inline void Qwen3CudaEmbeddingDirect(Qwen3CudaDirectRunner &runner,
                                         const Data &input, Data &weight, Data &output) {
        runner.Run("EmbeddingDirect",
                   DataDict{{"input", (Data*)&input}, {"weight", &weight}, {"output", &output}},
                   FloatDict(), IntDict(), {"output"}, false);
    }

    inline void Qwen3CudaRMSNormPart(Qwen3CudaDirectRunner &runner,
                                     const Data &input, Data &weight,
                                     float eps, int start, int end, Data &output) {
        runner.Run("RMSNormPart",
                   DataDict{{"input", (Data*)&input}, {"weight", &weight}, {"output", &output}},
                   FloatDict{{"eps", eps}}, IntDict{{"start", start}, {"end", end}}, {"output"});
    }

    inline void Qwen3CudaLinear(Qwen3CudaDirectRunner &runner,
                                Data &input, Data &weight,
                                const Data &bias, Data &output,
                                bool keepTpReplicated = false) {
        IntDict intParams;
        if (keepTpReplicated) {
            intParams["keepTpReplicated"] = 1;
        }
        runner.Run("Linear",
                   DataDict{{"input", &input}, {"weight", &weight}, {"bias", (Data*)&bias}, {"output", &output}},
                   FloatDict(), intParams, {"output"});
    }

    inline void Qwen3CudaTopK(Qwen3CudaDirectRunner &runner,
                              Data &input, Data &output, int topk) {
        runner.Run("TopK",
                   DataDict{{"input", &input}, {"output", &output}},
                   FloatDict(), IntDict{{"topk", topk}}, {"output"});
    }

    inline void Qwen3CudaLinearSwiglu(Qwen3CudaDirectRunner &runner,
                                      Data &input, Data &weight,
                                      const Data &bias, Data &middle, Data &output) {
        runner.Run("LinearSwiglu",
                   DataDict{{"input", &input}, {"weight", &weight}, {"bias", (Data*)&bias},
                            {"middle", &middle}, {"output", &output}},
                   FloatDict(), IntDict(), {"middle", "output"});
    }

    inline void Qwen3CudaLinearAddBlock(Qwen3CudaDirectRunner &runner,
                                        Data *input, Data *weight, Data *bias,
                                        Data *middle, Data *output) {
        runner.Run("LinearAdd",
                   DataDict{{"input", input}, {"weight", weight}, {"bias", bias},
                            {"middle", middle}, {"output", output}},
                   FloatDict(), IntDict(), {"middle"});
    }

    inline void Qwen3CudaSplit(Qwen3CudaDirectRunner &runner,
                               Data &input, int axis, int start, int end, Data &output) {
        runner.Run("Split",
                   DataDict{{"input", &input}, {"output", &output}},
                   FloatDict(), IntDict{{"axis", axis}, {"start", start}, {"end", end}}, {"output"});
    }

    inline void Qwen3CudaCat(Qwen3CudaDirectRunner &runner,
                             Data &input0, Data &input1, int axis, Data &output) {
        runner.Run("Cat",
                   DataDict{{"input0", &input0}, {"input1", &input1}, {"output", &output}},
                   FloatDict(), IntDict{{"axis", axis}}, {"output"});
    }

    inline void Qwen3CudaCatBatch(Qwen3CudaDirectRunner &runner,
                                  std::vector<Data*> &inputs, int axis, Data &output) {
        runner.Run("CatBatch",
                   DataDict{{"input", (Data*)inputs.data()}, {"output", &output}},
                   FloatDict(), IntDict{{"axis", axis}, {"input___batch", (int)inputs.size()}},
                   {"output"});
    }

    inline void Qwen3CudaPermuteSelf(Qwen3CudaDirectRunner &runner,
                                     Data &input, const std::vector<int> &axis) {
        Data axisData = Data(DataType::INT32PARAM, {(int)axis.size()});
        axisData.Allocate();
        for (int i = 0; i < axisData.Count(0); i++) {
            ((int32_t*)axisData.cpuData)[i] = axis[i];
        }
        runner.Run("PermuteSelf",
                   DataDict{{"input", &input}, {"axis", &axisData}},
                   FloatDict(), IntDict());
    }

    inline void Qwen3CudaRopeEncoding(Qwen3CudaDirectRunner &runner,
                                      Data &input, const Data &positionIds,
                                      int rotaryDim, float ropeTheta, float ropeScale) {
        runner.Run("RopeEncoding",
                   DataDict{{"input", &input}, {"positionIds", (Data*)&positionIds}},
                   FloatDict{{"ropeTheta", ropeTheta}, {"ropeScale", ropeScale}},
                   IntDict{{"rotaryDim", rotaryDim}});
    }

    inline void Qwen3CudaAddTo(Qwen3CudaDirectRunner &runner,
                               Data &input0, const Data &input1, float alpha = 1.0f) {
        runner.Run("AddTo",
                   DataDict{{"input0", &input0}, {"input1", (Data*)&input1}},
                   FloatDict{{"alpha", alpha}}, IntDict());
    }

    inline void Qwen3CudaSoftmax(Qwen3CudaDirectRunner &runner,
                                 const Data &input, Data &output, int axis) {
        runner.Run("SoftMax",
                   DataDict{{"input", (Data*)&input}, {"output", &output}},
                   FloatDict(), IntDict{{"axis", axis}}, {"output"});
    }

    inline void Qwen3CudaSelectExpert(Qwen3CudaDirectRunner &runner,
                                      const Data &logits, Data &index, Data &score,
                                      int topk, bool needNorm,
                                      float routeScale, const Data *gateBias) {
        DataDict datas = {{"logits", (Data*)&logits}, {"index", &index}, {"score", &score}};
        if (gateBias != nullptr) {
            datas["gateBias"] = (Data*)gateBias;
        }
        runner.Run("SelectExpert", datas,
                   FloatDict{{"routeScale", routeScale}},
                   IntDict{{"topk", topk}, {"needNorm", needNorm ? 1 : 0}},
                   {"index", "score"});
    }

    inline void Qwen3CudaToDataType(Qwen3CudaDirectRunner &runner, Data &input, DataType dataType) {
        if (input.dataType == dataType) {
            return;
        }
        if (dataType == DataType::FLOAT32) {
            runner.Run("ToFloat32", DataDict{{"input", &input}});
        } else if (dataType == DataType::FLOAT16) {
            runner.Run("ToFloat16", DataDict{{"input", &input}});
        } else if (dataType == DataType::BFLOAT16) {
            runner.Run("ToBFloat16", DataDict{{"input", &input}});
        } else {
            ErrorInFastLLM("Qwen3CudaToDataType: unsupported data type.\n");
        }
    }

    inline void Qwen3CudaLinearResidualReduce(
            Qwen3CudaDirectRunner &runner,
            Data &input, Data &weight, Data &bias,
            Data &middle, Data &hiddenStates,
            bool tensorParallel, bool firstTensorParallelRank,
            int gpuId) {
        DataType residualType = hiddenStates.dataType;
        bool canAddDirectly = input.dataType == residualType;

        if (tensorParallel) {
            if (firstTensorParallelRank) {
                if (canAddDirectly) {
                    Qwen3CudaLinearAddBlock(runner, &input, &weight, &bias, &middle, &hiddenStates);
                } else {
                    Qwen3CudaLinear(runner, input, weight, bias, middle);
                    Qwen3CudaToDataType(runner, middle, residualType);
                    Qwen3CudaAddTo(runner, hiddenStates, middle);
                }
            } else {
                Qwen3CudaLinear(runner, input, weight, bias, hiddenStates);
                Qwen3CudaToDataType(runner, hiddenStates, residualType);
            }
            FastllmNcclAllReduce(hiddenStates.cudaData, hiddenStates.cudaData,
                                 hiddenStates.Count(0), hiddenStates.dataType, gpuId);
            return;
        }

        if (canAddDirectly) {
            Qwen3CudaLinearAddBlock(runner, &input, &weight, &bias, &middle, &hiddenStates);
        } else {
            Qwen3CudaLinear(runner, input, weight, bias, middle);
            Qwen3CudaToDataType(runner, middle, residualType);
            Qwen3CudaAddTo(runner, hiddenStates, middle);
        }
    }

    inline void Qwen3CudaConvertToDataType(Qwen3CudaDirectRunner &runner,
                                           const Data &input, Data &output, DataType dataType) {
        if (dataType == DataType::FLOAT32) {
            runner.Run("ConvertToFloat32",
                       DataDict{{"input", (Data*)&input}, {"output", &output}},
                       FloatDict(), IntDict(), {"output"});
        } else if (dataType == DataType::FLOAT16) {
            runner.Run("ConvertToFloat16",
                       DataDict{{"input", (Data*)&input}, {"output", &output}},
                       FloatDict(), IntDict(), {"output"});
        } else if (dataType == DataType::BFLOAT16) {
            runner.Run("ConvertToBFloat16",
                       DataDict{{"input", (Data*)&input}, {"output", &output}},
                       FloatDict(), IntDict(), {"output"});
        } else {
            ErrorInFastLLM("Qwen3CudaConvertToDataType: unsupported data type.\n");
        }
    }

    inline void Qwen3CudaMergeMOE(Qwen3CudaDirectRunner &runner,
                                  const Data &input, const Data &index, const Data &score,
                                  std::vector<Data*> &weights, std::vector<Data*> &biass,
                                  Data &w1, Data &w2, Data &w3,
                                  Data &curInput, Data &curOutput,
                                  float sharedScale, Data &output,
                                  int layer, MoeGateType gateType = MoeGateSwiglu) {
        runner.Run("MergeMOE",
                   DataDict{{"input", (Data*)&input}, {"index", (Data*)&index}, {"score", (Data*)&score},
                            {"weights", (Data*)weights.data()}, {"biass", (Data*)biass.data()},
                            {"w1", &w1}, {"w2", &w2}, {"w3", &w3},
                            {"curInput", &curInput}, {"curOutput", &curOutput},
                            {"output", &output}},
                   FloatDict{{"sharedScale", sharedScale}},
                   IntDict{{"weights___batch", (int)weights.size()},
                           {"biass___batch", (int)biass.size()},
                           {"layer", layer},
                           {"gateType", (int)gateType}},
                   {"output"});
    }

    inline void Qwen3CudaClearStaleReplicatedView(Data *output) {
        if (output == nullptr || !output->multiDeviceData || !output->IsTensorParallelReplicated()) {
            return;
        }

        bool stale = (!output->tpGlobalDims.empty() && output->tpGlobalDims != output->dims);
        if (!stale) {
            for (auto &it : output->multiDeviceDatas) {
                if (it.second != nullptr && it.second->dims != output->dims) {
                    stale = true;
                    break;
                }
            }
        }
        if (!stale) {
            return;
        }

        for (auto &it : output->multiDeviceDatas) {
            delete it.second;
        }
        output->multiDeviceDatas.clear();
        output->multiDeviceData = false;
        output->ClearTensorParallelLayout();
    }

    inline void Qwen3CudaMergeMOEBlock(
            Qwen3CudaDirectRunner &runner,
            Data *input, Data *expertIndex, Data *expertScore,
            std::vector<Data*> *weights, std::vector<Data*> *biass,
            Data *w1, Data *w2, Data *w3, Data *tempInput, Data *tempOutput,
            float sharedScale, Data *output, int layer,
            DataType dataType, DataType moeAtype,
            Data *moeInputTemp, Data *moeOutputTemp,
            MoeGateType gateType = MoeGateSwiglu) {
        if (dataType == moeAtype) {
            Qwen3CudaMergeMOE(runner, *input, *expertIndex, *expertScore,
                              *weights, *biass, *w1, *w2, *w3,
                              *tempInput, *tempOutput,
                              sharedScale, *output, layer, gateType);
        } else {
            Qwen3CudaConvertToDataType(runner, *input, *moeInputTemp, moeAtype);
            Qwen3CudaMergeMOE(runner, *moeInputTemp, *expertIndex, *expertScore,
                              *weights, *biass, *w1, *w2, *w3,
                              *tempInput, *tempOutput,
                              sharedScale, *moeOutputTemp, layer, gateType);
            Qwen3CudaConvertToDataType(runner, *moeOutputTemp, *output, dataType);
        }
        Qwen3CudaClearStaleReplicatedView(output);
    }

    inline void Qwen3CudaAppendPagedCache(Qwen3CudaDirectRunner &runner,
                                          PagedCacheManager &pagedCacheManager,
                                          Data &cache, Data &input) {
        runner.Run("AppendPagedCache",
                   DataDict{{"pagedCacheManager", (Data*)&pagedCacheManager},
                            {"cache", &cache}, {"input", &input}},
                   FloatDict(), IntDict());
    }

    inline void Qwen3CudaGenerateAppendPagedCacheBatchParams(
            Qwen3CudaDirectRunner &runner,
            PagedCacheManager &pagedCacheManager,
            const std::vector<Data*> &pastKeys,
            int batch,
            Data &insertIndexs,
            Data &insertPositions) {
        runner.Run("GenerateAppendPagedCacheBatchParams",
                   DataDict{{"pagedCacheManager", (Data*)&pagedCacheManager},
                            {"pastKeys", (Data*)pastKeys.data()},
                            {"insertIndexs", &insertIndexs},
                            {"insertPositions", &insertPositions}},
                   FloatDict(),
                   IntDict{{"batch", batch}, {"pastKeys___batch", (int)pastKeys.size()}},
                   {"insertIndexs", "insertPositions"});
    }

    inline void Qwen3CudaGeneratePagedBatchParams(
            Qwen3CudaDirectRunner &runner,
            const Data &q,
            const std::vector<Data*> &pastKeys,
            int batch,
            Data &qSizes,
            Data &pageSizes,
            Data &pageIndexs,
            Data &lastPageLens,
            const std::vector<int> &seqLens,
            bool lastPageLensOnDevice = false) {
        IntDict intParams = {
                {"batch", batch},
                {"pastKeys___batch", (int)pastKeys.size()},
                {"lastPageLensOnDevice", (int)lastPageLensOnDevice},
                {"seqLens___size", (int)seqLens.size()}
        };
        for (int i = 0; i < (int)seqLens.size(); i++) {
            intParams["seqLens___" + std::to_string(i)] = seqLens[i];
        }
        runner.Run("GeneratePagedBatchParams",
                   DataDict{{"q", (Data*)&q},
                            {"pastKeys", (Data*)pastKeys.data()},
                            {"qSizes", &qSizes},
                            {"pageSizes", &pageSizes},
                            {"pageIndexs", &pageIndexs},
                            {"lastPageLens", &lastPageLens}},
                   FloatDict(), intParams,
                   {"qSizes", "pageSizes", "pageIndexs", "lastPageLens"});
    }

    inline void Qwen3CudaAttentionPagedBatch(
            Qwen3CudaDirectRunner &runner,
            Data &q,
            Data &kCaches,
            Data &vCaches,
            Data &qSizes,
            Data &pageSizes,
            Data &pageIndexs,
            Data &lastPageLens,
            Data &output,
            int group,
            float scale,
            int attentionType,
            bool inited) {
        runner.Run("AttentionPagedBatch",
                   DataDict{{"q", &q}, {"kCaches", &kCaches}, {"vCaches", &vCaches},
                            {"output", &output}, {"qSizes", &qSizes}, {"pageSizes", &pageSizes},
                            {"pageIndexs", &pageIndexs}, {"lastPageLens", &lastPageLens}},
                   FloatDict{{"scale", scale}},
                   IntDict{{"group", group}, {"attentionType", attentionType}, {"inited", (int)inited}, {"sync", 0}},
                   {"output"});
    }

    inline void Qwen3CudaQKVRMSNormRopeSplitAppendPagedCache(
            Qwen3CudaDirectRunner &runner,
            Data &qkv,
            Data &qNormWeight,
            Data &kNormWeight,
            const Data &positionIds,
            Data &qOutput,
            Data &pagedKCacheData,
            Data &pagedVCacheData,
            Data &insertIndexs,
            Data &insertPositions,
            int qHeads,
            int kHeads,
            int headDim,
            int rotaryDim,
            float eps,
            float ropeTheta,
            float ropeScale,
            int pageLen,
            int batch,
            bool doQKNorm,
            Data *lastPageLens) {
        DataDict datas = {
                {"qkv", &qkv},
                {"qNormWeight", &qNormWeight},
                {"kNormWeight", &kNormWeight},
                {"positionIds", (Data*)&positionIds},
                {"qOutput", &qOutput},
                {"pagedKCacheData", &pagedKCacheData},
                {"pagedVCacheData", &pagedVCacheData},
                {"insertIndexs", &insertIndexs},
                {"insertPositions", &insertPositions}
        };
        std::vector<std::string> outputs = {"qOutput"};
        if (lastPageLens != nullptr) {
            datas["lastPageLens"] = lastPageLens;
            outputs.push_back("lastPageLens");
        }
        runner.Run("QKVRMSNormRopeSplitAppendPagedCache",
                   datas,
                   FloatDict{{"eps", eps}, {"ropeTheta", ropeTheta}, {"ropeScale", ropeScale}},
                   IntDict{{"q_heads", qHeads}, {"k_heads", kHeads}, {"head_dim", headDim},
                           {"rotaryDim", rotaryDim}, {"pageLen", pageLen}, {"batch", batch},
                           {"doQKNorm", (int)doQKNorm}},
                   outputs);
    }

    inline void Qwen3CudaAttentionPagedBlock(
            Qwen3CudaDirectRunner &runner,
            Data *attenInput,
            Data *mergeQkvWeight, Data *mergeQkvBias,
            Data *qWeight, Data *qBias,
            Data *kWeight, Data *kBias,
            Data *vWeight, Data *vBias,
            Data *preQNormWeight, Data *preKNormWeight,
            Data *qNormWeight, Data *kNormWeight,
            Data *oWeight, Data *oBias,
            Data *allPositionIds,
            std::vector<std::pair<Data*, Data*>> *pastKeyValues,
            std::vector<Data*> *batchPastKeys,
            std::vector<Data*> *batchPastValues,
            Data *qkv, Data *q, Data *attenOutput, Data *attenLastOutput,
            Data *qForAttentionHolder,
            Data *insertIndexs, Data *insertPositions,
            Data *qSizes, Data *pageSizes, Data *pageIndexs, Data *lastPageLens,
            bool *generatedAppendParams, bool *generatedDecodeParams,
            int batch, int blockCnt, int layerIdx,
            const std::vector<int> &seqLens,
            int numAttentionHeads, int numKeyValueHeads, int headDim,
            int rotaryDim, float rmsNormEps,
            float ropeBase, float ropeFactor, int maxPositions,
            int ropeType,
            bool kvCacheInCPU,
            bool isPrefill,
            Data *hiddenStates,
            bool doQKNorm,
            bool doPostQKNorm,
            int pagedCacheLayerOffset,
            bool skipOutputProjection,
            bool externalDecodeMeta) {
        bool mergedQkv = (mergeQkvWeight->dims.size() > 0);
        if (mergedQkv) {
            mergeQkvWeight->tpPackType = TP_PACK_QKV;
            mergeQkvWeight->tpQHeads = numAttentionHeads;
            mergeQkvWeight->tpKVHeads = numKeyValueHeads;
            mergeQkvWeight->tpHeadDim = headDim;
            Qwen3CudaLinear(runner, *attenInput, *mergeQkvWeight, *mergeQkvBias, *qkv);
        } else {
            Data qResult, kResult, vResult, qkResult;
            Qwen3CudaLinear(runner, *attenInput, *qWeight, *qBias, qResult);
            Qwen3CudaLinear(runner, *attenInput, *kWeight, *kBias, kResult);
            Qwen3CudaLinear(runner, *attenInput, *vWeight, *vBias, vResult);
            Qwen3CudaCat(runner, qResult, kResult, -1, qkResult);
            Qwen3CudaCat(runner, qkResult, vResult, -1, *qkv);
        }

        if (doPostQKNorm) {
            int per = qkv->dims.back() / (numAttentionHeads / numKeyValueHeads + 2);
            int qdim = per * (numAttentionHeads / numKeyValueHeads);
            Qwen3CudaRMSNormPart(runner, *qkv, *preQNormWeight, rmsNormEps, 0, qdim, *qkv);
            Qwen3CudaRMSNormPart(runner, *qkv, *preKNormWeight, rmsNormEps, qdim, qdim + per, *qkv);
        }

        int targetSeqLength = 0;
        for (int b = 0; b < batch; b++) {
            Data &pastKey = *(*pastKeyValues)[b * blockCnt + layerIdx].first;
            Data &pastValue = *(*pastKeyValues)[b * blockCnt + layerIdx].second;
            if (kvCacheInCPU) {
                pastKey.lockInCPU = true;
                pastValue.lockInCPU = true;
            } else {
                if (pastKey.dataDeviceIds.empty()) {
                    pastKey.dataDeviceIds = {runner.DeviceId()};
                }
                if (pastValue.dataDeviceIds.empty()) {
                    pastValue.dataDeviceIds = {runner.DeviceId()};
                }
                AssertInFastLLM(pastKey.dataDevice == DataDevice::CUDA &&
                                pastValue.dataDevice == DataDevice::CUDA &&
                                pastKey.dataDeviceIds[0] == runner.DeviceId() &&
                                pastValue.dataDeviceIds[0] == runner.DeviceId(),
                                "Qwen3 direct CUDA TP cache is not on the bound CUDA device.\n");
            }
            targetSeqLength = std::max(targetSeqLength,
                (pastKey.dims.size() > 2) ? pastKey.dims[1] + seqLens[b] : seqLens[b]);
        }

        float curRopeTheta = ropeBase;
        if (targetSeqLength >= maxPositions && RoPEType::DYMAMIC_NTK == ropeType) {
            float scale = std::pow((ropeFactor * targetSeqLength / maxPositions) - (ropeFactor - 1),
                                   rotaryDim / (rotaryDim - 2));
            curRopeTheta = ropeBase * scale;
        }
        float ropeScale = (ropeType == RoPEType::LINEAR_SCALE) ? ropeFactor : 1.0f;

        for (int b = 0; b < batch; b++) {
            (*batchPastKeys)[b] = (*pastKeyValues)[b * blockCnt + layerIdx].first;
            (*batchPastValues)[b] = (*pastKeyValues)[b * blockCnt + layerIdx].second;
        }

        bool useFp8KVCache = ((*batchPastKeys)[0]->dataType == DataType::FP8_E4M3 ||
                              (*batchPastValues)[0]->dataType == DataType::FP8_E4M3);
        if (useFp8KVCache) {
            AssertInFastLLM(!kvCacheInCPU, "FP8 KV cache doesn't support kvCacheInCPU.\n");
            AssertInFastLLM(qkv->dataDevice == DataDevice::CUDA, "FP8 KV cache requires CUDA paged attention.\n");
            AssertInFastLLM(headDim != 64, "FP8 KV cache is not supported when head_dim == 64.\n");
        }

        int bsz = attenInput->dims[0], seqlen = attenInput->dims[1];
        auto resolvePagedAttentionQType = [&](DataType cacheType, DataType queryType) -> DataType {
            if (cacheType == DataType::FLOAT16 || cacheType == DataType::BFLOAT16) {
                return cacheType;
            }
            if (queryType == DataType::FLOAT16 || queryType == DataType::BFLOAT16) {
                return queryType;
            }
            if (attenInput->dataType == DataType::BFLOAT16) {
                return DataType::BFLOAT16;
            }
            return DataType::FLOAT16;
        };
        auto preparePagedAttentionQ = [&](Data &src, DataType cacheType, Data &casted) -> Data& {
            DataType targetType = resolvePagedAttentionQType(cacheType, src.dataType);
            if (src.dataType == targetType) {
                return src;
            }
            Data &holder = qForAttentionHolder == nullptr ? casted : *qForAttentionHolder;
            Qwen3CudaConvertToDataType(runner, src, holder, targetType);
            return holder;
        };

        if (!isPrefill && (*batchPastKeys)[0]->pagedKVCacheData == nullptr) {
            isPrefill = true;
        }

        if (isPrefill) {
            Data k, v;

            int per = qkv->dims.back() / (numAttentionHeads / numKeyValueHeads + 2);
            int qdim = per * (numAttentionHeads / numKeyValueHeads);
            Qwen3CudaSplit(runner, *qkv, -1, 0, qdim, *q);
            Qwen3CudaSplit(runner, *qkv, -1, qdim, qdim + per, k);
            Qwen3CudaSplit(runner, *qkv, -1, qdim + per, qdim + per * 2, v);

            std::vector<int> qkvSize = {bsz, seqlen, -1, headDim};
            q->Reshape(qkvSize);
            k.Reshape(qkvSize);
            v.Reshape(qkvSize);

            if (doQKNorm) {
                Qwen3CudaRMSNorm(runner, *q, *qNormWeight, rmsNormEps, *q);
                Qwen3CudaRMSNorm(runner, k, *kNormWeight, rmsNormEps, k);
            }
            Qwen3CudaRopeEncoding(runner, *q, *allPositionIds, rotaryDim, curRopeTheta, ropeScale);
            Qwen3CudaRopeEncoding(runner, k, *allPositionIds, rotaryDim, curRopeTheta, ropeScale);

            Qwen3CudaPermuteSelf(runner, *q, {0, 2, 1, 3});
            Qwen3CudaPermuteSelf(runner, k, {0, 2, 1, 3});
            Qwen3CudaPermuteSelf(runner, v, {0, 2, 1, 3});

            k.Reshape({-1, seqlen, headDim});
            v.Reshape({-1, seqlen, headDim});
            q->Reshape({-1, seqlen, headDim});

            auto makeCacheDesc = [](const Data &src, DataType targetType) {
                Data desc(targetType);
                desc.dims = src.dims;
                desc.strides = src.strides;
                desc.dataDevice = src.dataDevice;
                desc.dataDeviceIds = src.dataDeviceIds;
                desc.multiDeviceData = src.multiDeviceData;
                desc.tpLayout = src.tpLayout;
                desc.tpAxis = src.tpAxis;
                desc.tpGlobalDims = src.tpGlobalDims;
                desc.tpRanges = src.tpRanges;
                desc.UpdateUnitSize();
                return desc;
            };

            if (batch == 1) {
                Data kCacheDesc = makeCacheDesc(k, (*batchPastKeys)[0]->dataType);
                Data vCacheDesc = makeCacheDesc(v, (*batchPastValues)[0]->dataType);
                int cacheLayerIdx = pagedCacheLayerOffset + layerIdx;
                PagedCacheManager *pagedCacheKManager = AllocatePagedCacheManager(
                    cacheLayerIdx * 2, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, kCacheDesc);
                PagedCacheManager *pagedCacheVManager = AllocatePagedCacheManager(
                    cacheLayerIdx * 2 + 1, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, vCacheDesc);
                Qwen3CudaAppendPagedCache(runner, *pagedCacheKManager, *(*batchPastKeys)[0], k);
                Qwen3CudaAppendPagedCache(runner, *pagedCacheVManager, *(*batchPastValues)[0], v);
            } else {
                int total = 0;
                Data curK, curV;

                for (int b = 0; b < batch; b++) {
                    Data &pastKey = *(*batchPastKeys)[b];
                    Data &pastValue = *(*batchPastValues)[b];

                    Qwen3CudaSplit(runner, k, 1, total, total + seqLens[b], curK);
                    Qwen3CudaSplit(runner, v, 1, total, total + seqLens[b], curV);

                    Data kCacheDesc = makeCacheDesc(curK, pastKey.dataType);
                    Data vCacheDesc = makeCacheDesc(curV, pastValue.dataType);
                    int cacheLayerIdx = pagedCacheLayerOffset + layerIdx;
                    PagedCacheManager *pagedCacheKManager = AllocatePagedCacheManager(
                        cacheLayerIdx * 2, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, kCacheDesc);
                    PagedCacheManager *pagedCacheVManager = AllocatePagedCacheManager(
                        cacheLayerIdx * 2 + 1, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, vCacheDesc);
                    Qwen3CudaAppendPagedCache(runner, *pagedCacheKManager, pastKey, curK);
                    Qwen3CudaAppendPagedCache(runner, *pagedCacheVManager, pastValue, curV);

                    total += seqLens[b];
                }
            }

            {
                Data &kCaches = *(*batchPastKeys)[0];
                Data &vCaches = *(*batchPastValues)[0];
                Data qForAttentionHolder;
                Data &qForAttention = preparePagedAttentionQ(*q, kCaches.dataType, qForAttentionHolder);
                Qwen3CudaGeneratePagedBatchParams(runner, qForAttention, *batchPastKeys, batch,
                    *qSizes, *pageSizes, *pageIndexs, *lastPageLens, seqLens);
                Qwen3CudaAttentionPagedBatch(runner, qForAttention,
                    kCaches, vCaches,
                    *qSizes, *pageSizes, *pageIndexs, *lastPageLens,
                    *attenOutput, numAttentionHeads / numKeyValueHeads, 1.0f / std::sqrt((float)headDim), 1, layerIdx > 0);
            }

            attenOutput->Reshape({1, seqlen, -1});
            if (!skipOutputProjection) {
                Qwen3CudaLinearAddBlock(runner, attenOutput, oWeight, oBias, attenLastOutput, hiddenStates);
            }
        } else {
            Data &kCaches = *(*batchPastKeys)[0];
            Data &vCaches = *(*batchPastValues)[0];
            PagedCacheManager *pagedCacheKManager = kCaches.pagedKVCacheData;
            PagedCacheManager *pagedCacheVManager = vCaches.pagedKVCacheData;

            if (!externalDecodeMeta && !(*generatedAppendParams)) {
                Qwen3CudaGenerateAppendPagedCacheBatchParams(runner, *pagedCacheKManager,
                    *batchPastKeys, batch, *insertIndexs, *insertPositions);
                *generatedAppendParams = true;
            }

            q->dataType = qkv->dataType;
            q->Resize({bsz * numAttentionHeads, seqlen, headDim});
            Qwen3CudaPrepareLocalOutput(*q, runner.DeviceId());
            int curPageLen = kCaches.pageLen;
            bool fillLastPageLensOnDevice = qkv->dataDevice == DataDevice::CUDA &&
                                             !qkv->multiDeviceData &&
                                             !externalDecodeMeta &&
                                             !(*generatedDecodeParams);
            Qwen3CudaQKVRMSNormRopeSplitAppendPagedCache(runner, *qkv,
                *qNormWeight, *kNormWeight,
                *allPositionIds,
                *q,
                *(Data*)pagedCacheKManager, *(Data*)pagedCacheVManager,
                *insertIndexs, *insertPositions,
                numAttentionHeads, numKeyValueHeads, headDim,
                rotaryDim, rmsNormEps, curRopeTheta, ropeScale,
                curPageLen, batch, doQKNorm,
                fillLastPageLensOnDevice ? lastPageLens : nullptr);

            if (!externalDecodeMeta) {
                for (int b = 0; b < batch; b++) {
                    auto updatePageMeta = [](Data *cache, PagedCacheManager *mgr) {
                        if (cache->pageIndex.empty() || cache->lastPageLen >= cache->pageLen) {
                            cache->pageIndex.push_back(mgr->GetUnusedPageIndex(true));
                            cache->lastPageLen = 1;
                        } else {
                            cache->lastPageLen++;
                        }
                    };
                    updatePageMeta((*batchPastKeys)[b], pagedCacheKManager);
                    updatePageMeta((*batchPastValues)[b], pagedCacheVManager);
                }
            }

            if (!externalDecodeMeta && !(*generatedDecodeParams)) {
                Data qForAttentionHolder;
                Data &qForAttention = preparePagedAttentionQ(*q, kCaches.dataType, qForAttentionHolder);
                Qwen3CudaGeneratePagedBatchParams(runner, qForAttention, *batchPastKeys, batch,
                    *qSizes, *pageSizes, *pageIndexs, *lastPageLens, std::vector<int>(),
                    fillLastPageLensOnDevice);
                *generatedDecodeParams = true;
            }
            Data qForAttentionHolder;
            Data &qForAttention = preparePagedAttentionQ(*q, kCaches.dataType, qForAttentionHolder);
            Qwen3CudaAttentionPagedBatch(runner, qForAttention,
                kCaches, vCaches,
                *qSizes, *pageSizes, *pageIndexs, *lastPageLens,
                *attenOutput, numAttentionHeads / numKeyValueHeads, 1.0f / std::sqrt((float)headDim), 1, layerIdx > 0);

            attenOutput->Reshape({seqlen, bsz, -1});
            Qwen3CudaPermuteSelf(runner, *attenOutput, {1, 0, 2});

            if (!skipOutputProjection) {
                Qwen3CudaLinearAddBlock(runner, attenOutput, oWeight, oBias, attenLastOutput, hiddenStates);
            }
        }
    }

    }

#endif

}
