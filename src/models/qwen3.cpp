//
// Created by huangyuyang on 4/29/25.
//

#include "utils.h"

#include "qwen3.h"
#include "blocks/baseblock.h"
#include "executor.h"

#include <sstream>

#include <unordered_map>

#include <cstring>
#include <atomic>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <set>
#include <thread>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif

namespace fastllm {
    extern std::vector <float> GetInterLeavePowerOf2(int n);
    extern std::vector <float> GetInterleave(int n);

#ifdef USE_CUDA
    namespace {
        static std::atomic<int> qwen3ThreadTpNextPagedCacheBase(2000000);

        static std::string TrimString(const std::string &s) {
            int l = 0, r = (int)s.size();
            while (l < r && std::isspace((unsigned char)s[l])) {
                l++;
            }
            while (r > l && std::isspace((unsigned char)s[r - 1])) {
                r--;
            }
            return s.substr(l, r - l);
        }

        static bool IsDisabledTpValue(const std::string &value) {
            std::string v = value;
            std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return (char)std::tolower(c); });
            return v.empty() || v == "0" || v == "false" || v == "off" || v == "none" || v == "disable";
        }

        static bool IsDisabledQwen3TpSpec(const std::string &value) {
            std::string v = value;
            std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return (char)std::tolower(c); });
            return v.empty() || v == "false" || v == "off" || v == "none" || v == "disable";
        }

        static bool Qwen3NeedRepeatPenalty(const GenerationConfig &config) {
            float diff = config.repeat_penalty - 1.0f;
            return diff > 1e-6f || diff < -1e-6f;
        }

        static bool AppendQwen3CudaDevicesFromSpec(const std::string &spec,
                                                   const std::string &type,
                                                   int defaultRatio,
                                                   std::vector<int> &devices,
                                                   std::map<int, int> &ratios) {
            std::map<int, int> parsedRatios;
            std::vector<int> parsed = ParseDeviceIds(spec, type, parsedRatios);
            if (parsed.empty() && (spec == "cuda" || spec == "multicuda")) {
                parsed.push_back(0);
            }
            bool added = false;
            for (int device : parsed) {
                if (device < 0 || device == 99999) {
                    continue;
                }
                int ratio = defaultRatio;
                auto ratioIt = parsedRatios.find(device);
                if (ratioIt != parsedRatios.end() && ratioIt->second > 0) {
                    ratio = ratioIt->second;
                }
                if (ratio <= 0) {
                    ratio = 1;
                }
                if (std::find(devices.begin(), devices.end(), device) == devices.end()) {
                    devices.push_back(device);
                }
                ratios[device] += ratio;
                added = true;
            }
            return added;
        }

        static bool ParseQwen3GPUForwardSpec(const std::string &rawSpec,
                                             std::vector<int> &devices,
                                             std::map<int, int> &ratios) {
            std::string spec = TrimString(rawSpec);
            if (IsDisabledQwen3TpSpec(spec)) {
                return false;
            }

            std::string lower = spec;
            std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return (char)std::tolower(c); });
            if (lower == "1" || lower == "true" || lower == "on" || lower == "auto") {
                int count = FastllmCudaGetDeviceCount();
                for (int i = 0; i < count; i++) {
                    devices.push_back(i);
                    ratios[i] = 1;
                }
                return !devices.empty();
            }

            std::string parseSpec = spec;
            std::string type = "cuda";
            if (StartWith(lower, "multicuda")) {
                type = "multicuda";
            } else if (!StartWith(lower, "cuda")) {
                parseSpec = "cuda:" + spec;
            }
            return AppendQwen3CudaDevicesFromSpec(parseSpec, type, 1, devices, ratios);
        }

        static bool GetQwen3GPUForwardDevices(const std::map<std::string, int> &deviceMap,
                                              std::vector<int> &devices,
                                              std::map<int, int> &ratios) {
            devices.clear();
            ratios.clear();
            const char *env = std::getenv("FASTLLM_TP");
            if (env == nullptr || IsDisabledQwen3TpSpec(TrimString(env))) {
                env = std::getenv("FASTLLM_QWEN3_THREAD_TP");
            }
            if (env != nullptr) {
                ParseQwen3GPUForwardSpec(env, devices, ratios);
            }

            if (devices.empty()) {
                for (auto &it : deviceMap) {
                    std::string lower = it.first;
                    std::transform(lower.begin(), lower.end(), lower.begin(),
                                   [](unsigned char c) { return (char)std::tolower(c); });
                    if (StartWith(lower, "multicuda")) {
                        AppendQwen3CudaDevicesFromSpec(it.first, "multicuda", it.second, devices, ratios);
                    }
                }
            }

            if (devices.empty()) {
                for (auto &it : deviceMap) {
                    std::string lower = it.first;
                    std::transform(lower.begin(), lower.end(), lower.begin(),
                                   [](unsigned char c) { return (char)std::tolower(c); });
                    if (StartWith(lower, "cuda")) {
                        AppendQwen3CudaDevicesFromSpec(it.first, "cuda", it.second, devices, ratios);
                    }
                }
            }

            std::vector<int> uniqueDevices;
            std::set<int> seen;
            for (int device : devices) {
                if (device >= 0 && seen.insert(device).second) {
                    uniqueDevices.push_back(device);
                    if (ratios.find(device) == ratios.end() || ratios[device] <= 0) {
                        ratios[device] = 1;
                    }
                }
            }
            devices.swap(uniqueDevices);
            return !devices.empty();
        }

        static bool GetQwen3ThreadTpDevices(std::vector<int> &devices, std::map<int, int> &ratios) {
            if (!GetQwen3GPUForwardDevices(GetDeviceMap(), devices, ratios)) {
                return false;
            }
            return devices.size() > 1;
        }

        static DivisionScheme ExtractQwen3FirstRangeScheme(const DivisionScheme &scheme) {
            DivisionScheme ret;
            for (auto &it : scheme) {
                ret[it.first];
                if (!it.second.empty()) {
                    ret[it.first].push_back(it.second[0]);
                }
            }
            return ret;
        }

        static DivisionScheme ExtractQwen3KVHeadScheme(const DivisionScheme &qkvScheme,
                                                       int qWidth, int headDim) {
            DivisionScheme ret;
            for (auto &it : qkvScheme) {
                ret[it.first];
                if (it.second.size() < 2) {
                    continue;
                }
                int st = (it.second[1].first - qWidth) / headDim;
                int end = (it.second[1].second - qWidth) / headDim;
                ret[it.first].push_back({st, end});
            }
            return ret;
        }

        static DataType ResolveQwen3ThreadTpComputeType(DataType modelType) {
            if (modelType == DataType::FLOAT16 || modelType == DataType::BFLOAT16) {
                return modelType;
            }
            return DataType::FLOAT16;
        }

        static DataType ResolveQwen3ThreadTpCacheType(DataType cacheType, DataType computeType) {
            if (cacheType == DataType::FLOAT16 ||
                cacheType == DataType::BFLOAT16 ||
                cacheType == DataType::FP8_E4M3) {
                return cacheType;
            }
            return computeType;
        }

        static Data *EnsureQwen3ThreadTpLocalCache(Data &root, int device, DataType localDataType) {
            root.multiDeviceData = true;
            root.dataDevice = DataDevice::CUDA;
            auto &local = root.multiDeviceDatas[device];
            if (local == nullptr) {
                local = new Data(localDataType);
                local->SetKVCache();
                local->cacheUid = root.cacheUid;
                local->dataDevice = DataDevice::CUDA;
                local->dataDeviceIds = {device};
            } else if (local->dataType != localDataType && local->dims.empty()) {
                local->dataType = localDataType;
                local->UpdateUnitSize();
            }
            return local;
        }

        static void SyncQwen3ThreadTpRootCacheMeta(Data &root,
                                                   const std::vector<int> &devices,
                                                   const DivisionScheme &kvHeadScheme,
                                                   int globalKVHeads,
                                                   int headDim) {
            if (devices.empty()) {
                return;
            }
            Data *firstLocal = nullptr;
            for (int device : devices) {
                auto it = root.multiDeviceDatas.find(device);
                if (it != root.multiDeviceDatas.end() && it->second != nullptr &&
                    it->second->dims.size() >= 3) {
                    firstLocal = it->second;
                    break;
                }
            }
            if (firstLocal == nullptr) {
                return;
            }

            root.multiDeviceData = true;
            root.dataType = firstLocal->dataType;
            root.UpdateUnitSize();
            root.dataDevice = DataDevice::CUDA;
            root.dataDeviceIds = devices;
            root.tpLayout = TP_LAYOUT_SHARDED;
            root.tpAxis = 0;
            root.tpRanges = kvHeadScheme;
            root.tpGlobalDims = {globalKVHeads, firstLocal->dims[1], headDim};
            root.Resize(root.tpGlobalDims);
            root.cudaData = nullptr;
            root.isKVCache = true;
            root.isPagedKVCache = firstLocal->isPagedKVCache;
            root.pageLen = firstLocal->pageLen;
            root.pageIndex = firstLocal->pageIndex;
            root.lastPageLen = firstLocal->lastPageLen;
            root.pagedKVCacheData = firstLocal->pagedKVCacheData;
        }

        static void Qwen3CudaClearMultiDeviceState(Data &data) {
            for (auto &it : data.multiDeviceDatas) {
                delete it.second;
            }
            data.multiDeviceDatas.clear();
            data.multiDeviceData = false;
            data.ClearTensorParallelLayout();
        }

        static void Qwen3CudaPrepareLocalOutput(Data &data, int device) {
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
                                    "Qwen3 direct CUDA runner can't run " + opType + ".\n");
                }
                device->Reshape(opType, datas, floatParams, intParams);
                device->Run(opType, datas, floatParams, intParams);
            }

        private:
            int deviceId;
            CudaDevice cudaDevice;
            BaseDevice *device;
        };

        struct Qwen3ForwardSingleBuffers {
            Data hiddenStates;
            Data attenInput;
            Data qkv;
            Data q;
            Data attenOutput;
            Data attenLastOutput;
            Data gateupResult;
            Data swigluResult;
            Data mlpPart;
            Data qSizes;
            Data pageSizes;
            Data pageIndexs;
            Data lastPageLens;
            Data insertIndexs;
            Data insertPositions;
            std::vector<Data*> batchPastKeys;
            std::vector<Data*> batchPastValues;

            Qwen3ForwardSingleBuffers() : batchPastKeys(1), batchPastValues(1) {}
        };

        struct Qwen3CudaGraphDecodeState {
            std::mutex mutex;
            std::string signature;
            bool warmed = false;
            bool captured = false;
            bool disabled = false;
            void *graph = nullptr;
            void *exec = nullptr;
            Data inputIds;
            Data positionIds;
            Qwen3ForwardSingleBuffers buffers;
            Data logits;

            ~Qwen3CudaGraphDecodeState() {
                if (exec != nullptr) {
                    FastllmCudaGraphExecDestroy(exec);
                    exec = nullptr;
                }
                if (graph != nullptr) {
                    FastllmCudaGraphDestroy(graph);
                    graph = nullptr;
                }
            }
        };

        static void Qwen3DestroyCudaGraph(Qwen3CudaGraphDecodeState &state) {
            if (state.exec != nullptr) {
                FastllmCudaGraphExecDestroy(state.exec);
                state.exec = nullptr;
            }
            if (state.graph != nullptr) {
                FastllmCudaGraphDestroy(state.graph);
                state.graph = nullptr;
            }
            state.captured = false;
            state.warmed = false;
        }

        static Qwen3CudaGraphDecodeState &GetQwen3CudaGraphDecodeState(const Qwen3Model *model, int gpuId) {
            static std::mutex statesMutex;
            static std::map<std::pair<const Qwen3Model*, int>, std::unique_ptr<Qwen3CudaGraphDecodeState> > states;
            std::lock_guard<std::mutex> guard(statesMutex);
            auto key = std::make_pair(model, gpuId);
            auto &state = states[key];
            if (state == nullptr) {
                state.reset(new Qwen3CudaGraphDecodeState());
            }
            return *state;
        }

        static bool Qwen3CudaGraphEnabled() {
            const char *env = std::getenv("FASTLLM_QWEN3_CUDA_GRAPH");
            return env == nullptr || !IsDisabledTpValue(TrimString(env));
        }

        static void Qwen3PrepareGraphCudaTensor(Data &dst, const Data &src, int device) {
            AssertInFastLLM(src.dataDevice == DataDevice::CUDA && src.cudaData != nullptr,
                            "Qwen3 CUDA graph requires CUDA source tensor.\n");
            FastllmCudaSetDevice(device);

            bool needReset = dst.isFake || dst.dataDevice != DataDevice::CUDA ||
                             dst.dataType != src.dataType || dst.dims != src.dims ||
                             (!dst.dataDeviceIds.empty() && dst.dataDeviceIds[0] != device);
            if (!needReset && dst.cudaData != nullptr) {
                int ptrDevice = GetPointerDeviceId(dst.cudaData);
                needReset = ptrDevice >= 0 && ptrDevice != device;
            }
            if (needReset) {
                if (dst.isFake) {
                    dst.isFake = false;
                    dst.cpuData = nullptr;
                    dst.cudaData = nullptr;
                    dst.deviceData = nullptr;
                    dst.expansionSize = 0;
                    dst.expansionBytes = 0;
                } else {
                    dst.FreeSpace();
                }
                Qwen3CudaClearMultiDeviceState(dst);
                dst.dataType = src.dataType;
                dst.UpdateUnitSize();
                dst.dataDevice = DataDevice::CUDA;
                dst.dataDeviceIds = {device};
                dst.Resize(src.dims);
            }
            dst.Allocate(false);
            FastllmCudaCopyFromDeviceToDevice(dst.cudaData, src.cudaData, src.GetBytes());
        }

        static void Qwen3PrepareGraphIntTensor(Data &dst, int device, const std::vector<int> &host) {
            AssertInFastLLM(!host.empty(), "Qwen3 CUDA graph got empty int metadata.\n");
            FastllmCudaSetDevice(device);
            bool needReset = dst.isFake || dst.dataDevice != DataDevice::CUDA ||
                             dst.dataType != DataType::INT32 ||
                             dst.dims != std::vector<int>{(int)host.size()} ||
                             (!dst.dataDeviceIds.empty() && dst.dataDeviceIds[0] != device);
            if (!needReset && dst.cudaData != nullptr) {
                int ptrDevice = GetPointerDeviceId(dst.cudaData);
                needReset = ptrDevice >= 0 && ptrDevice != device;
            }
            if (needReset) {
                if (dst.isFake) {
                    dst.isFake = false;
                    dst.cpuData = nullptr;
                    dst.cudaData = nullptr;
                    dst.deviceData = nullptr;
                    dst.expansionSize = 0;
                    dst.expansionBytes = 0;
                } else {
                    dst.FreeSpace();
                }
                Qwen3CudaClearMultiDeviceState(dst);
                dst.dataType = DataType::INT32;
                dst.UpdateUnitSize();
                dst.dataDevice = DataDevice::CUDA;
                dst.dataDeviceIds = {device};
                dst.Resize({(int)host.size()});
            }
            dst.Allocate(false);
            FastllmCudaCopyFromHostToDevice(dst.cudaData, (void*)host.data(), host.size() * sizeof(int32_t));
            dst.cpuIntDatas = host;
        }

        static void Qwen3CudaRMSNorm(Qwen3CudaDirectRunner &runner,
                                     const Data &input, Data &weight,
                                     float eps, Data &output) {
            runner.Run("RMSNorm",
                       DataDict{{"input", (Data*)&input}, {"weight", &weight}, {"output", &output}},
                       FloatDict{{"eps", eps}}, IntDict(), {"output"});
        }

        static void Qwen3CudaEmbeddingDirect(Qwen3CudaDirectRunner &runner,
                                             const Data &input, Data &weight, Data &output) {
            runner.Run("EmbeddingDirect",
                       DataDict{{"input", (Data*)&input}, {"weight", &weight}, {"output", &output}},
                       FloatDict(), IntDict(), {"output"}, false);
        }

        static void Qwen3CudaRMSNormPart(Qwen3CudaDirectRunner &runner,
                                         const Data &input, Data &weight,
                                         float eps, int start, int end, Data &output) {
            runner.Run("RMSNormPart",
                       DataDict{{"input", (Data*)&input}, {"weight", &weight}, {"output", &output}},
                       FloatDict{{"eps", eps}}, IntDict{{"start", start}, {"end", end}}, {"output"});
        }

        static void Qwen3CudaLinear(Qwen3CudaDirectRunner &runner,
                                    Data &input, Data &weight,
                                    const Data &bias, Data &output) {
            runner.Run("Linear",
                       DataDict{{"input", &input}, {"weight", &weight}, {"bias", (Data*)&bias}, {"output", &output}},
                       FloatDict(), IntDict(), {"output"});
        }

        static void Qwen3CudaTopK(Qwen3CudaDirectRunner &runner,
                                  Data &input, Data &output, int topk) {
            runner.Run("TopK",
                       DataDict{{"input", &input}, {"output", &output}},
                       FloatDict(), IntDict{{"topk", topk}}, {"output"});
        }

        static void Qwen3CudaLinearSwiglu(Qwen3CudaDirectRunner &runner,
                                          Data &input, Data &weight,
                                          const Data &bias, Data &middle, Data &output) {
            runner.Run("LinearSwiglu",
                       DataDict{{"input", &input}, {"weight", &weight}, {"bias", (Data*)&bias},
                                {"middle", &middle}, {"output", &output}},
                       FloatDict(), IntDict(), {"middle", "output"});
        }

        static void Qwen3CudaLinearAddBlock(Qwen3CudaDirectRunner &runner,
                                            Data *input, Data *weight, Data *bias,
                                            Data *middle, Data *output) {
            runner.Run("LinearAdd",
                       DataDict{{"input", input}, {"weight", weight}, {"bias", bias},
                                {"middle", middle}, {"output", output}},
                       FloatDict(), IntDict(), {"middle"});
        }

        static void Qwen3CudaSplit(Qwen3CudaDirectRunner &runner,
                                   Data &input, int axis, int start, int end, Data &output) {
            runner.Run("Split",
                       DataDict{{"input", &input}, {"output", &output}},
                       FloatDict(), IntDict{{"axis", axis}, {"start", start}, {"end", end}}, {"output"});
        }

        static void Qwen3CudaCat(Qwen3CudaDirectRunner &runner,
                                 Data &input0, Data &input1, int axis, Data &output) {
            runner.Run("Cat",
                       DataDict{{"input0", &input0}, {"input1", &input1}, {"output", &output}},
                       FloatDict(), IntDict{{"axis", axis}}, {"output"});
        }

        static void Qwen3CudaCatBatch(Qwen3CudaDirectRunner &runner,
                                      std::vector<Data*> &inputs, int axis, Data &output) {
            runner.Run("CatBatch",
                       DataDict{{"input", (Data*)inputs.data()}, {"output", &output}},
                       FloatDict(), IntDict{{"axis", axis}, {"input___batch", (int)inputs.size()}},
                       {"output"});
        }

        static void Qwen3CudaPermuteSelf(Qwen3CudaDirectRunner &runner,
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

        static void Qwen3CudaRopeEncoding(Qwen3CudaDirectRunner &runner,
                                          Data &input, const Data &positionIds,
                                          int rotaryDim, float ropeTheta, float ropeScale) {
            runner.Run("RopeEncoding",
                       DataDict{{"input", &input}, {"positionIds", (Data*)&positionIds}},
                       FloatDict{{"ropeTheta", ropeTheta}, {"ropeScale", ropeScale}},
                       IntDict{{"rotaryDim", rotaryDim}});
        }

        static void Qwen3CudaAddTo(Qwen3CudaDirectRunner &runner,
                                   Data &input0, const Data &input1, float alpha = 1.0f) {
            runner.Run("AddTo",
                       DataDict{{"input0", &input0}, {"input1", (Data*)&input1}},
                       FloatDict{{"alpha", alpha}}, IntDict());
        }

        static void Qwen3CudaToDataType(Qwen3CudaDirectRunner &runner, Data &input, DataType dataType) {
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

        static void Qwen3CudaLinearResidualReduce(
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

        static void Qwen3CudaConvertToDataType(Qwen3CudaDirectRunner &runner,
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

        static void Qwen3CudaAppendPagedCache(Qwen3CudaDirectRunner &runner,
                                              PagedCacheManager &pagedCacheManager,
                                              Data &cache, Data &input) {
            runner.Run("AppendPagedCache",
                       DataDict{{"pagedCacheManager", (Data*)&pagedCacheManager},
                                {"cache", &cache}, {"input", &input}},
                       FloatDict(), IntDict());
        }

        static void Qwen3CudaGenerateAppendPagedCacheBatchParams(
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

        static void Qwen3CudaGeneratePagedBatchParams(
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

        static void Qwen3CudaAttentionPagedBatch(
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

        static void Qwen3CudaQKVRMSNormRopeSplitAppendPagedCache(
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

        static void Qwen3CudaAttentionPagedBlock(
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
                float scale = pow((ropeFactor * targetSeqLength / maxPositions) - (ropeFactor - 1),
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
                        *attenOutput, numAttentionHeads / numKeyValueHeads, 1.0f / sqrt(headDim), 1, layerIdx > 0);
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
                    *attenOutput, numAttentionHeads / numKeyValueHeads, 1.0f / sqrt(headDim), 1, layerIdx > 0);

                attenOutput->Reshape({seqlen, bsz, -1});
                Qwen3CudaPermuteSelf(runner, *attenOutput, {1, 0, 2});

                if (!skipOutputProjection) {
                    Qwen3CudaLinearAddBlock(runner, attenOutput, oWeight, oBias, attenLastOutput, hiddenStates);
                }
                }
            }
        }

        static bool Qwen3CanUseCudaFullLogitsSampling(
                const std::vector<GenerationConfig> &generationConfigs,
                std::vector<std::vector<float>*> *retLogits,
                int batch,
                bool &allSimple,
                int &maxTopK) {
            allSimple = true;
            maxTopK = 1;
            for (int b = 0; b < batch; b++) {
                const GenerationConfig &config = generationConfigs[b];
                allSimple &= config.IsSimpleGreedy();
                if (config.output_logits && retLogits != nullptr &&
                    b < (int)retLogits->size() && (*retLogits)[b] != nullptr) {
                    return false;
                }
                if (Qwen3NeedRepeatPenalty(config)) {
                    return false;
                }
                int curTopK = config.IsSimpleGreedy() ? 1 : config.top_k;
                if (curTopK <= 0 || curTopK > 50) {
                    return false;
                }
                maxTopK = std::max(maxTopK, curTopK);
            }
            return allSimple;
        }

        static Data &Qwen3ThreadLocalCudaSamplingFullLogits() {
            static thread_local Data fullLogits(DataType::FLOAT32);
            return fullLogits;
        }

        static Data &Qwen3ThreadLocalCudaSamplingTopK() {
            static thread_local Data topk(DataType::FLOAT32);
            return topk;
        }

        static Data &Qwen3ThreadLocalCpuSamplingTopK() {
            static thread_local Data topk(DataType::FLOAT32);
            return topk;
        }

        static void Qwen3GatherShardLogitsToRootCuda(
                int rootDevice,
                const std::vector<int> &devices,
                const DivisionScheme &lmHeadScheme,
                std::vector<Data> &localLogits,
                int batch,
                int vocabSize,
                Data &fullLogits) {
            FastllmCudaSetDevice(rootDevice);
            Qwen3CudaPrepareLocalOutput(fullLogits, rootDevice);
            fullLogits.dataType = DataType::FLOAT32;
            fullLogits.UpdateUnitSize();
            fullLogits.Resize({batch, vocabSize});
            fullLogits.Allocate();

            for (int r = 0; r < (int)devices.size(); r++) {
                int device = devices[r];
                auto schemeIt = lmHeadScheme.find(device);
                AssertInFastLLM(schemeIt != lmHeadScheme.end(),
                                "Qwen3 CUDA sampling: missing lm_head split range.\n");
                AssertInFastLLM(localLogits[r].dataDevice == DataDevice::CUDA &&
                                localLogits[r].cudaData != nullptr,
                                "Qwen3 CUDA sampling: local logits must stay on CUDA.\n");
                int localVocab = localLogits[r].dims.back();
                int rows = localLogits[r].Count(0) / localVocab;
                AssertInFastLLM(rows == batch,
                                "Qwen3 CUDA sampling: local logits batch mismatch.\n");

                uint8_t *dstBase = (uint8_t*)fullLogits.cudaData;
                uint8_t *srcBase = (uint8_t*)localLogits[r].cudaData;
                int localOffset = 0;
                for (auto &range : schemeIt->second) {
                    int len = range.second - range.first;
                    AssertInFastLLM(range.first >= 0 && range.second <= vocabSize &&
                                    localOffset + len <= localVocab,
                                    "Qwen3 CUDA sampling: invalid lm_head split range.\n");
                    FastllmCudaMemcpy2DDeviceToDeviceAuto(
                        dstBase + (size_t)range.first * sizeof(float),
                        (size_t)vocabSize * sizeof(float),
                        srcBase + (size_t)localOffset * sizeof(float),
                        (size_t)localVocab * sizeof(float),
                        (size_t)len * sizeof(float),
                        (size_t)batch,
                        rootDevice,
                        device);
                    localOffset += len;
                }
            }
        }

        static std::vector<int> Qwen3SampleFromRootCudaLogits(
                int rootDevice,
                Data &fullLogits,
                int batch,
                int maxTopK,
                bool allSimple,
                const std::vector<GenerationConfig> &generationConfigs) {
            FastllmCudaSetDevice(rootDevice);
            std::vector<int> lastRet;
            lastRet.reserve(batch);
            Qwen3CudaDirectRunner cudaRunner(rootDevice);
            Data &topk = Qwen3ThreadLocalCudaSamplingTopK();
            Qwen3CudaTopK(cudaRunner, fullLogits, topk, maxTopK);
            Data &cpuTopK = Qwen3ThreadLocalCpuSamplingTopK();
            cpuTopK.dataType = DataType::FLOAT32;
            cpuTopK.UpdateUnitSize();
            cpuTopK.Resize(topk.dims);
            if (cpuTopK.dataDevice != DataDevice::CPU) {
                cpuTopK.FreeSpace();
                cpuTopK.dataDevice = DataDevice::CPU;
                cpuTopK.dataDeviceIds = {0};
            }
            cpuTopK.Allocate();
            FastllmCudaCopyFromDeviceToHost(cpuTopK.cpuData, topk.cudaData, topk.GetBytes());
            int stride = cpuTopK.Count(0) / batch;
            float *topkData = (float*)cpuTopK.cpuData;
            if (allSimple) {
                for (int b = 0; b < batch; b++) {
                    lastRet.push_back((int)(topkData[(long long)b * stride] + 1e-3f));
                }
            } else {
                for (int b = 0; b < batch; b++) {
                    lastRet.push_back(LLMSamplingOnly(cpuTopK, b, generationConfigs[b]));
                }
            }
            return lastRet;
        }
#endif

    Qwen3Model::Qwen3Model() {
        this->model_struct = "llama";
        this->model_type = "qwen3";
        this->use_new_engine = true;

        // 默认使用 llama3 的提示词和instruction
        this->pre_prompt="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant.<|eot_id|>";
        this->user_role="<|start_header_id|>user<|end_header_id|>\n";
        this->bot_role="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n";
        this->history_sep="<|eot_id|>\n";

        block_cnt = 32;
        rotary_dim = 128;

        weight.embeddingNames.insert("model.embed_tokens.weight");
        weight.linearNames = {
            "lm_head.weight", "model.layers.*.mlp.down_proj.weight", "model.layers.*.mlp.up_proj.weight",
            "model.layers.*.mlp.gate_proj.weight",  "model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.gateup_proj.weight",
            "model.layers.*.self_attn.o_proj.weight", "model.layers.*.self_attn.q_proj.weight", "model.layers.*.self_attn.k_proj.weight",
            "model.layers.*.self_attn.v_proj.weight", "model.layers.*.self_attn.mergeqkv.weight", "model.layers.*.self_attn.W_pack.weight"
        };
    }

    bool Qwen3Model::IsThreadTensorParallelEnabled() const {
#ifdef USE_CUDA
        std::vector<int> devices;
        std::map<int, int> ratios;
        return GetQwen3ThreadTpDevices(devices, ratios);
#else
        return false;
#endif
    }

    Data &Qwen3Model::GetThreadTensorParallelBias(const std::string &name) {
        auto it = this->weight.weight.find(name);
        if (it != this->weight.weight.end()) {
            return it->second;
        }
        return this->threadTpEmptyBiases[name];
    }

    bool Qwen3Model::ForwardSingleGPUDecodeGraph(
            int gpuId,
            std::map <int, int> ratios,
            int batch,
            const Data &inputIds,
            const Data &positionIds,
            const std::vector <int> &seqLens,
            std::vector <std::pair <Data*, Data*> > &pastKeyValues,
            bool all1,
            bool isPrefill,
            bool tensorParallel,
            bool firstTensorParallelRank,
            int pagedCacheLayerOffset,
            Data &logits) {
#ifndef USE_CUDA
        return false;
#else
        if (!Qwen3CudaGraphEnabled() || batch != 1 || !all1 || isPrefill ||
            seqLens.size() != 1 || seqLens[0] != 1 ||
            (int)pastKeyValues.size() < block_cnt) {
            return false;
        }

        auto requireLocal = [&](Data &data, const std::string &name) -> Data* {
            auto it = data.multiDeviceDatas.find(gpuId);
            AssertInFastLLM(it != data.multiDeviceDatas.end() && it->second != nullptr,
                            "Qwen3 ForwardSingleGPU graph missing local tensor: " + name + ".\n");
            return it->second;
        };

        Data *localInputIds = requireLocal((Data&)inputIds, "inputIds");
        Data *localPositionIds = requireLocal((Data&)positionIds, "positionIds");
        if (localInputIds->dims.size() != 2 || localInputIds->dims[0] != 1 || localInputIds->dims[1] != 1 ||
            localPositionIds->dims.empty() || localPositionIds->Count(0) != 1) {
            return false;
        }

        int currentTokens = 0;
        for (int i = 0; i < block_cnt; i++) {
            Data *pastKey = pastKeyValues[i].first;
            Data *pastValue = pastKeyValues[i].second;
            if (pastKey == nullptr || pastValue == nullptr ||
                pastKey->pagedKVCacheData == nullptr || pastValue->pagedKVCacheData == nullptr ||
                pastKey->pageIndex.empty() || pastValue->pageIndex.empty() ||
                pastKey->dataDevice != DataDevice::CUDA || pastValue->dataDevice != DataDevice::CUDA ||
                pastKey->dataType == DataType::FP8_E4M3 || pastValue->dataType == DataType::FP8_E4M3 ||
                pastKey->pageLen <= 0 || pastKey->pageLen != pastValue->pageLen ||
                pastKey->pageIndex.size() != pastValue->pageIndex.size() ||
                pastKey->lastPageLen != pastValue->lastPageLen) {
                return false;
            }
            int layerTokens = ((int)pastKey->pageIndex.size() - 1) * pastKey->pageLen + pastKey->lastPageLen;
            currentTokens = std::max(currentTokens, layerTokens);
        }
        if (rope_type == RoPEType::DYMAMIC_NTK && currentTokens + 1 >= max_positions) {
            return false;
        }

        Qwen3CudaGraphDecodeState &state = GetQwen3CudaGraphDecodeState(this, gpuId);
        std::unique_lock<std::mutex> graphLock(state.mutex);
        if (state.disabled) {
            return false;
        }

        FastllmCudaSetDevice(gpuId);
        Qwen3PrepareGraphCudaTensor(state.inputIds, *localInputIds, gpuId);
        Qwen3PrepareGraphCudaTensor(state.positionIds, *localPositionIds, gpuId);

        Data *firstKey = pastKeyValues[0].first;
        bool needNewPage = firstKey->pageIndex.empty() || firstKey->lastPageLen >= firstKey->pageLen;
        int insertIndex = needNewPage ? -1 : firstKey->pageIndex.back();
        int insertPosition = needNewPage ? 0 : firstKey->lastPageLen;

        for (int i = 0; i < block_cnt; i++) {
            Data *pastKey = pastKeyValues[i].first;
            Data *pastValue = pastKeyValues[i].second;
            bool layerNeedNewPage = pastKey->pageIndex.empty() || pastKey->lastPageLen >= pastKey->pageLen;
            AssertInFastLLM(layerNeedNewPage == needNewPage,
                            "Qwen3 CUDA graph requires aligned paged cache layout across layers.\n");
            if (needNewPage) {
                int keyPage = pastKey->pagedKVCacheData->GetUnusedPageIndex(true);
                int valuePage = pastValue->pagedKVCacheData->GetUnusedPageIndex(true);
                if (insertIndex < 0) {
                    insertIndex = keyPage;
                }
                AssertInFastLLM(keyPage == insertIndex && valuePage == insertIndex,
                                "Qwen3 CUDA graph requires aligned K/V page indices across layers.\n");
                pastKey->pageIndex.push_back(keyPage);
                pastValue->pageIndex.push_back(valuePage);
                pastKey->lastPageLen = 1;
                pastValue->lastPageLen = 1;
            } else {
                AssertInFastLLM(pastKey->pageIndex.back() == insertIndex &&
                                pastValue->pageIndex.back() == insertIndex &&
                                pastKey->lastPageLen == insertPosition &&
                                pastValue->lastPageLen == insertPosition,
                                "Qwen3 CUDA graph requires aligned paged cache positions across layers.\n");
                pastKey->lastPageLen++;
                pastValue->lastPageLen++;
            }
        }

        std::vector<int> pageIndexHost = firstKey->pageIndex;
        int lastPageLen = firstKey->lastPageLen;
        Qwen3PrepareGraphIntTensor(state.buffers.insertIndexs, gpuId, {insertIndex});
        Qwen3PrepareGraphIntTensor(state.buffers.insertPositions, gpuId, {insertPosition});
        Qwen3PrepareGraphIntTensor(state.buffers.qSizes, gpuId, {0, 1});
        Qwen3PrepareGraphIntTensor(state.buffers.pageSizes, gpuId, {0, (int)pageIndexHost.size()});
        Qwen3PrepareGraphIntTensor(state.buffers.pageIndexs, gpuId, pageIndexHost);
        Qwen3PrepareGraphIntTensor(state.buffers.lastPageLens, gpuId, {lastPageLen});

        std::ostringstream signature;
        signature << "gpu=" << gpuId
                  << ";tp=" << (tensorParallel ? 1 : 0)
                  << ";tpRank0=" << (firstTensorParallelRank ? 1 : 0)
                  << ";pages=" << pageIndexHost.size()
                  << ";inputType=" << (int)state.inputIds.dataType
                  << ";posType=" << (int)state.positionIds.dataType
                  << ";kCache=" << pastKeyValues[0].first->pagedKVCacheData->cudaData
                  << ";vCache=" << pastKeyValues[0].second->pagedKVCacheData->cudaData
                  << ";lmLocal=" << requireLocal(weight["lm_head.weight"], "lm_head.weight")->dims[0];
        std::string newSignature = signature.str();
        if (state.signature != newSignature) {
            Qwen3DestroyCudaGraph(state);
            state.signature = newSignature;
        }

        auto runGraphBody = [&]() {
            Qwen3CudaDirectRunner cudaRunner(gpuId);
            Qwen3ForwardSingleBuffers &buf = state.buffers;
            if ((int)buf.batchPastKeys.size() != batch) {
                buf.batchPastKeys.resize(batch);
                buf.batchPastValues.resize(batch);
            }

            Qwen3CudaEmbeddingDirect(cudaRunner,
                                     state.inputIds,
                                     *requireLocal(weight["model.embed_tokens.weight"], "model.embed_tokens.weight"),
                                     buf.hiddenStates);
            const DataType computeType = ResolveQwen3ThreadTpComputeType(this->dataType);
            if (buf.hiddenStates.dataType != computeType) {
                Qwen3CudaToDataType(cudaRunner, buf.hiddenStates, computeType);
            }

            bool generatedAppendParams = false;
            bool generatedDecodeParams = false;
            for (int i = 0; i < block_cnt; i++) {
                std::string inputRmsName = "model.layers." + std::to_string(i) + ".input_layernorm.weight";
                std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
                std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
                std::string qNormName = "model.layers." + std::to_string(i) + ".self_attn.q_norm.weight";
                std::string kNormName = "model.layers." + std::to_string(i) + ".self_attn.k_norm.weight";
                std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";
                std::string oBiasName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.bias";
                std::string postRmsName = "model.layers." + std::to_string(i) + ".post_attention_layernorm.weight";
                std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
                std::string downWeightName = "model.layers." + std::to_string(i) + ".mlp.down_proj.weight";
                std::string downBiasName = "model.layers." + std::to_string(i) + ".mlp.down_proj.bias";

                Qwen3CudaRMSNorm(cudaRunner, buf.hiddenStates,
                                 *requireLocal(weight[inputRmsName], inputRmsName),
                                 rms_norm_eps, buf.attenInput);

                Data *localMergeW = requireLocal(weight[mergeQkvWeightName], mergeQkvWeightName);
                int group = num_attention_heads / num_key_value_heads;
                int localKVHeads = localMergeW->tpKVHeads > 0 ?
                    localMergeW->tpKVHeads : localMergeW->dims[0] / ((group + 2) * head_dim);
                int localQHeads = localMergeW->tpQHeads > 0 ?
                    localMergeW->tpQHeads : localKVHeads * group;
                AssertInFastLLM(localKVHeads > 0 && localQHeads > 0,
                                "Qwen3 ForwardSingleGPU graph got empty local attention shard.\n");

                Qwen3CudaAttentionPagedBlock(
                    cudaRunner,
                    &buf.attenInput,
                    localMergeW, requireLocal(GetThreadTensorParallelBias(mergeQkvBiasName), mergeQkvBiasName),
                    GetEmptyData(), GetEmptyData(),
                    GetEmptyData(), GetEmptyData(),
                    GetEmptyData(), GetEmptyData(),
                    GetEmptyData(), GetEmptyData(),
                    requireLocal(weight[qNormName], qNormName),
                    requireLocal(weight[kNormName], kNormName),
                    requireLocal(weight[oWeightName], oWeightName),
                    requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                    &state.positionIds,
                    &pastKeyValues,
                    &buf.batchPastKeys, &buf.batchPastValues,
                    &buf.qkv, &buf.q, &buf.attenOutput, &buf.attenLastOutput,
                    nullptr,
                    &buf.insertIndexs, &buf.insertPositions,
                    &buf.qSizes, &buf.pageSizes, &buf.pageIndexs, &buf.lastPageLens,
                    &generatedAppendParams, &generatedDecodeParams,
                    batch, block_cnt, i,
                    seqLens,
                    localQHeads, localKVHeads, head_dim,
                    rotary_dim, rms_norm_eps,
                    rope_base, rope_factor, max_positions,
                    rope_type,
                    GetKVCacheInCPU(),
                    false,
                    &buf.hiddenStates,
                    true,
                    false,
                    pagedCacheLayerOffset,
                    true,
                    true
                );

                Qwen3CudaLinearResidualReduce(
                    cudaRunner, buf.attenOutput,
                    *requireLocal(weight[oWeightName], oWeightName),
                    *requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                    buf.attenLastOutput, buf.hiddenStates,
                    tensorParallel, firstTensorParallelRank, gpuId);

                Qwen3CudaRMSNorm(cudaRunner, buf.hiddenStates,
                                 *requireLocal(weight[postRmsName], postRmsName),
                                 rms_norm_eps, buf.attenInput);
                Qwen3CudaLinearSwiglu(cudaRunner, buf.attenInput,
                                      *requireLocal(weight[swigluWeightName], swigluWeightName),
                                      *requireLocal(GetThreadTensorParallelBias(swigluWeightName + ".tp_bias"),
                                                    swigluWeightName + ".tp_bias"),
                                      buf.gateupResult, buf.swigluResult);
                Qwen3CudaLinearResidualReduce(
                    cudaRunner, buf.swigluResult,
                    *requireLocal(weight[downWeightName], downWeightName),
                    *requireLocal(GetThreadTensorParallelBias(downBiasName), downBiasName),
                    buf.mlpPart, buf.hiddenStates,
                    tensorParallel, firstTensorParallelRank, gpuId);
            }

            Qwen3CudaRMSNorm(cudaRunner, buf.hiddenStates,
                             *requireLocal(weight["model.norm.weight"], "model.norm.weight"),
                             rms_norm_eps, buf.hiddenStates);
            Qwen3CudaLinear(cudaRunner, buf.hiddenStates,
                            *requireLocal(weight["lm_head.weight"], "lm_head.weight"),
                            *requireLocal(GetThreadTensorParallelBias("lm_head.weight.tp_bias"),
                                          "lm_head.weight.tp_bias"),
                            state.logits);
            Qwen3CudaToDataType(cudaRunner, state.logits, DataType::FLOAT32);
        };

        auto finishWithLogits = [&]() {
            Qwen3PrepareGraphCudaTensor(logits, state.logits, gpuId);
        };

        auto runWithoutGraph = [&]() {
            runGraphBody();
            finishWithLogits();
        };

        if (state.captured) {
            if (FastllmCudaGraphLaunch(state.exec)) {
                finishWithLogits();
                return true;
            }
            printf("Warning: Qwen3 CUDA graph replay failed on gpu %d: %s. Disable graph for this GPU.\n",
                   gpuId, FastllmCudaGraphLastError());
            Qwen3DestroyCudaGraph(state);
            state.disabled = true;
            runWithoutGraph();
            return true;
        }

        if (!state.warmed) {
            runWithoutGraph();
            state.warmed = true;
            return true;
        }

        void *capturedGraph = nullptr;
        if (!FastllmCudaGraphBeginCapture()) {
            printf("Warning: Qwen3 CUDA graph begin capture failed on gpu %d: %s. Disable graph for this GPU.\n",
                   gpuId, FastllmCudaGraphLastError());
            state.disabled = true;
            runWithoutGraph();
            return true;
        }
        runGraphBody();
        if (!FastllmCudaGraphEndCapture(&capturedGraph) || capturedGraph == nullptr) {
            printf("Warning: Qwen3 CUDA graph end capture failed on gpu %d: %s. Disable graph for this GPU.\n",
                   gpuId, FastllmCudaGraphLastError());
            if (capturedGraph != nullptr) {
                FastllmCudaGraphDestroy(capturedGraph);
            }
            Qwen3DestroyCudaGraph(state);
            state.disabled = true;
            runWithoutGraph();
            return true;
        }

        void *capturedExec = nullptr;
        if (!FastllmCudaGraphInstantiate(capturedGraph, &capturedExec) || capturedExec == nullptr) {
            printf("Warning: Qwen3 CUDA graph instantiate failed on gpu %d: %s. Disable graph for this GPU.\n",
                   gpuId, FastllmCudaGraphLastError());
            FastllmCudaGraphDestroy(capturedGraph);
            Qwen3DestroyCudaGraph(state);
            state.disabled = true;
            runWithoutGraph();
            return true;
        }

        state.graph = capturedGraph;
        state.exec = capturedExec;
        state.captured = true;
        if (!FastllmCudaGraphLaunch(state.exec)) {
            printf("Warning: Qwen3 CUDA graph first launch failed on gpu %d: %s. Disable graph for this GPU.\n",
                   gpuId, FastllmCudaGraphLastError());
            Qwen3DestroyCudaGraph(state);
            state.disabled = true;
            runWithoutGraph();
            return true;
        }
        finishWithLogits();
        return true;
#endif
    }

    void Qwen3Model::ForwardSingleGPU(
            int gpuId,
            std::map <int, int> ratios,
            int batch,
            const Data &inputIds,
            const Data &positionIds,
            const std::vector <int> &seqLens,
            std::vector <std::pair <Data*, Data*> > &pastKeyValues,
            bool all1,
            bool isPrefill,
            bool tensorParallel,
            bool firstTensorParallelRank,
            int pagedCacheLayerOffset,
            Data &logits) {
#ifndef USE_CUDA
        ErrorInFastLLM("Qwen3 ForwardSingleGPU requires CUDA.\n");
#else
        AssertInFastLLM(ratios.find(gpuId) == ratios.end() || ratios[gpuId] > 0,
                        "Qwen3 ForwardSingleGPU got invalid GPU ratio.\n");
        FastllmCudaSetDevice(gpuId);
        if (ForwardSingleGPUDecodeGraph(gpuId, ratios, batch, inputIds, positionIds,
                                        seqLens, pastKeyValues, all1, isPrefill,
                                        tensorParallel, firstTensorParallelRank,
                                        pagedCacheLayerOffset, logits)) {
            return;
        }
        Qwen3CudaDirectRunner cudaRunner(gpuId);

        auto requireLocal = [&](Data &data, const std::string &name) -> Data* {
            auto it = data.multiDeviceDatas.find(gpuId);
            AssertInFastLLM(it != data.multiDeviceDatas.end() && it->second != nullptr,
                            "Qwen3 ForwardSingleGPU missing local tensor: " + name + ".\n");
            return it->second;
        };

        Data hiddenStates;
        Qwen3CudaEmbeddingDirect(cudaRunner,
                                 *requireLocal((Data&)inputIds, "inputIds"),
                                 *requireLocal(weight["model.embed_tokens.weight"], "model.embed_tokens.weight"),
                                 hiddenStates);
        const DataType computeType = ResolveQwen3ThreadTpComputeType(this->dataType);
        if (hiddenStates.dataType != computeType) {
            Qwen3CudaToDataType(cudaRunner, hiddenStates, computeType);
        }

        Data attenInput, qkv, q, attenOutput, attenLastOutput;
        Data gateupResult, swigluResult, mlpPart;
        Data qSizes, pageSizes, pageIndexs, lastPageLens, insertIndexs, insertPositions;
        std::vector<Data*> batchPastKeys(batch), batchPastValues(batch);
        bool generatedAppendParams = false;
        bool generatedDecodeParams = false;

        for (int i = 0; i < block_cnt; i++) {
            std::string inputRmsName = "model.layers." + std::to_string(i) + ".input_layernorm.weight";
            std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
            std::string qNormName = "model.layers." + std::to_string(i) + ".self_attn.q_norm.weight";
            std::string kNormName = "model.layers." + std::to_string(i) + ".self_attn.k_norm.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";
            std::string oBiasName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.bias";
            std::string postRmsName = "model.layers." + std::to_string(i) + ".post_attention_layernorm.weight";
            std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
            std::string downWeightName = "model.layers." + std::to_string(i) + ".mlp.down_proj.weight";
            std::string downBiasName = "model.layers." + std::to_string(i) + ".mlp.down_proj.bias";

            Qwen3CudaRMSNorm(cudaRunner, hiddenStates,
                             *requireLocal(weight[inputRmsName], inputRmsName),
                             rms_norm_eps, attenInput);

            Data *localMergeW = requireLocal(weight[mergeQkvWeightName], mergeQkvWeightName);
            int group = num_attention_heads / num_key_value_heads;
            int localKVHeads = localMergeW->tpKVHeads > 0 ?
                localMergeW->tpKVHeads : localMergeW->dims[0] / ((group + 2) * head_dim);
            int localQHeads = localMergeW->tpQHeads > 0 ?
                localMergeW->tpQHeads : localKVHeads * group;
            AssertInFastLLM(localKVHeads > 0 && localQHeads > 0,
                            "Qwen3 ForwardSingleGPU got empty local attention shard.\n");

            Qwen3CudaAttentionPagedBlock(
                cudaRunner,
                &attenInput,
                localMergeW, requireLocal(GetThreadTensorParallelBias(mergeQkvBiasName), mergeQkvBiasName),
                GetEmptyData(), GetEmptyData(),
                GetEmptyData(), GetEmptyData(),
                GetEmptyData(), GetEmptyData(),
                GetEmptyData(), GetEmptyData(),
                requireLocal(weight[qNormName], qNormName),
                requireLocal(weight[kNormName], kNormName),
                requireLocal(weight[oWeightName], oWeightName),
                requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                requireLocal((Data&)positionIds, "positionIds"),
                &pastKeyValues,
                &batchPastKeys, &batchPastValues,
                &qkv, &q, &attenOutput, &attenLastOutput,
                nullptr,
                &insertIndexs, &insertPositions,
                &qSizes, &pageSizes, &pageIndexs, &lastPageLens,
                &generatedAppendParams, &generatedDecodeParams,
                batch, block_cnt, i,
                seqLens,
                localQHeads, localKVHeads, head_dim,
                rotary_dim, rms_norm_eps,
                rope_base, rope_factor, max_positions,
                rope_type,
                GetKVCacheInCPU(),
                isPrefill,
                &hiddenStates,
                true,
                false,
                pagedCacheLayerOffset,
                true,
                false
            );

            Qwen3CudaLinearResidualReduce(
                cudaRunner, attenOutput,
                *requireLocal(weight[oWeightName], oWeightName),
                *requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                attenLastOutput, hiddenStates,
                tensorParallel, firstTensorParallelRank, gpuId);

            Qwen3CudaRMSNorm(cudaRunner, hiddenStates,
                             *requireLocal(weight[postRmsName], postRmsName),
                             rms_norm_eps, attenInput);
            Qwen3CudaLinearSwiglu(cudaRunner, attenInput,
                                  *requireLocal(weight[swigluWeightName], swigluWeightName),
                                  *requireLocal(GetThreadTensorParallelBias(swigluWeightName + ".tp_bias"),
                                                swigluWeightName + ".tp_bias"),
                                  gateupResult, swigluResult);
            Qwen3CudaLinearResidualReduce(
                cudaRunner, swigluResult,
                *requireLocal(weight[downWeightName], downWeightName),
                *requireLocal(GetThreadTensorParallelBias(downBiasName), downBiasName),
                mlpPart, hiddenStates,
                tensorParallel, firstTensorParallelRank, gpuId);
        }

        Data lastHiddenStates;
        Data *headInput = &hiddenStates;
        if (!all1) {
            int total = 0;
            std::vector<Data> lastToks(seqLens.size());
            std::vector<Data*> lastTokPointers;
            lastTokPointers.reserve(seqLens.size());
            for (int b = 0; b < (int)seqLens.size(); b++) {
                Qwen3CudaSplit(cudaRunner, hiddenStates, 1,
                               total + seqLens[b] - 1, total + seqLens[b],
                               lastToks[b]);
                total += seqLens[b];
                lastTokPointers.push_back(&lastToks[b]);
            }
            Qwen3CudaCatBatch(cudaRunner, lastTokPointers, 1, lastHiddenStates);
            headInput = &lastHiddenStates;
        }

        Qwen3CudaRMSNorm(cudaRunner, *headInput,
                         *requireLocal(weight["model.norm.weight"], "model.norm.weight"),
                         rms_norm_eps, *headInput);
        Qwen3CudaLinear(cudaRunner, *headInput,
                        *requireLocal(weight["lm_head.weight"], "lm_head.weight"),
                        *requireLocal(GetThreadTensorParallelBias("lm_head.weight.tp_bias"),
                                      "lm_head.weight.tp_bias"),
                        logits);
        Qwen3CudaToDataType(cudaRunner, logits, DataType::FLOAT32);
#endif
    }

    void Qwen3Model::InitParams() {
        basellm::InitParams();
        num_key_value_heads = num_attention_heads;
        if (this->weight.dicts.find("num_key_value_heads") != this->weight.dicts.end()) {
            num_key_value_heads = atoi(this->weight.dicts["num_key_value_heads"].c_str());
        }
        if (this->weight.dicts.find("head_dim") != this->weight.dicts.end()) {
            head_dim = atoi(this->weight.dicts["head_dim"].c_str());
        }
        embed_dim = head_dim * num_attention_heads;
        rotary_dim = head_dim;
        if (this->weight.dicts.find("max_position_embeddings") != this->weight.dicts.end()) {
            max_positions = atoi(this->weight.dicts["max_position_embeddings"].c_str());
        }
        if (this->weight.dicts.find("rms_norm_eps") != this->weight.dicts.end()) {
            rms_norm_eps = atof(this->weight.dicts["rms_norm_eps"].c_str());
        }
        if (this->weight.dicts.find("rope_scaling.type") != this->weight.dicts.end()) {
            std::string type = this->weight.dicts["rope_scaling.type"];
            if (type == "linear")
               rope_type = RoPEType::LINEAR_SCALE;
            else if (type == "dynamic")
               rope_type = RoPEType::DYMAMIC_NTK;
        }
        if (this->weight.dicts.find("rope_theta") != this->weight.dicts.end()) {
            rope_base = atof(this->weight.dicts["rope_theta"].c_str());
        }
        if (this->weight.dicts.find("rope_scaling.factor") != this->weight.dicts.end()) {
            rope_factor = atof(this->weight.dicts["rope_scaling.factor"].c_str());
        }
        std::pair<std::vector<float>, std::vector<float>> &&pair = this->UpdateRotaryPosEmb(rope_base, rope_factor, std::max(max_positions, 16384));
        sinData.ToDevice(DataDevice::CPU);
        cosData.ToDevice(DataDevice::CPU);
        sinData.CopyFrom(Data(DataType::FLOAT32, { (int)this->sin.size(), (int)this->sin[0].size() }, pair.first));
        cosData.CopyFrom(Data(DataType::FLOAT32, { (int)this->cos.size(), (int)this->cos[0].size() }, pair.second));
        for (int i = 0; i < block_cnt; i++) {
            std::string w1WeightName = "model.layers." + std::to_string(i) + ".mlp.gate_proj.weight";
            std::string w3WeightName = "model.layers." + std::to_string(i) + ".mlp.up_proj.weight";
            std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
            this->weightMergeRules.push_back(
                WeightMergeRule({WeightMergeRuleSingle({w1WeightName, w3WeightName}, swigluWeightName, std::string("linearSwiglu"))})
            );

            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string qBiasName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.bias";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string kBiasName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.bias";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string vBiasName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.bias";
            std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
            this->weightMergeRules.push_back(
                WeightMergeRule({WeightMergeRuleSingle({qWeightName, kWeightName, vWeightName}, mergeQkvWeightName, std::string("linear")),
                                 WeightMergeRuleSingle({qBiasName, kBiasName, vBiasName}, mergeQkvBiasName, std::string("bias"))})
            );
        }
    }

    std::pair<std::vector<float>, std::vector<float>> Qwen3Model::UpdateRotaryPosEmb(float base, float factor, int seqLen) {
        int positions = std::max(max_positions, seqLen);
        sin.resize(positions);
        cos.resize(positions);
        std::vector <float> invFreq;
        for (int i = 0; i < rotary_dim; i += 2) {
            invFreq.push_back(1.0 / pow(base, (float)i / rotary_dim));
        }
        float scale = rope_type == RoPEType::LINEAR_SCALE ? factor : 1.0;
        for (int i = 0; i < positions; i++) {
            sin[i].resize(rotary_dim);
            cos[i].resize(rotary_dim);
            for (int j = 0; j < invFreq.size(); j++) {
                sin[i][j] = ::sin((float)i / scale * invFreq[j]);
                cos[i][j] = ::cos((float)i / scale * invFreq[j]);
            }
        }
        std::vector <float> fsin, fcos;
        for (int i = 0; i < sin.size(); i++) {
            fsin.insert(fsin.end(), sin[i].begin(), sin[i].end());
            fcos.insert(fcos.end(), cos[i].begin(), cos[i].end());
        }
        return std::make_pair(fsin, fcos);
    }

    int Qwen3Model::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <float> *retLogits) {
        Data attentionMaskCopy(attentionMask), positionIdsCopy(positionIds);
        std::vector<Data*> attentionMasks = {attentionMaskCopy.dims.empty() ? nullptr : &attentionMaskCopy};
        std::vector<Data*> positionIdsVec = {&positionIdsCopy};
        std::vector<int> seqLens = {(int)inputIds.dims[1]};
        std::vector<GenerationConfig> generationConfigs = {generationConfig};
        std::vector<std::pair<Data*, Data*> > pagedPastKeyValues;
        for (int i = 0; i < (int)pastKeyValues.size(); i++) {
            pagedPastKeyValues.push_back(std::make_pair(&pastKeyValues[i].first, &pastKeyValues[i].second));
        }
        std::vector<std::vector<float>*> batchLogits = {retLogits};
#ifdef USE_CUDA
        std::vector<int> devices;
        std::map<int, int> ratios;
        if (GetQwen3GPUForwardDevices(this->deviceMap, devices, ratios)) {
            return ForwardGPU(1, inputIds, attentionMasks, positionIdsVec, seqLens,
                              pagedPastKeyValues, generationConfigs, lastTokens,
                              &batchLogits)[0];
        }
#endif
        return ForwardV2(1, inputIds, attentionMasks, positionIdsVec, seqLens,
                         pagedPastKeyValues, generationConfigs, lastTokens,
                         &batchLogits)[0];
    }
/*
    std::vector <int> Qwen3Model::ForwardBatchV2(int batch, const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <std::vector <float>*> *retLogits) {
        int maxLen = inputIds.dims[1];
        Data hiddenStates;
        Data attenInput;
        Data qkv;
        Data q, k, v, curInput, curOutput;
        Data attenLastOutput;
        Data w1, w2, w3;
        Data* sinDataPtr = &sinData;
        Data* cosDataPtr = &cosData;

        Embedding(inputIds, this->weight["model.embed_tokens.weight"], hiddenStates);
        ToDataType(hiddenStates, this->dataType);
        int seqlen = hiddenStates.dims[1];
        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    rms_norm_eps, attenInput);
            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string qBiasName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.bias";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string kBiasName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.bias";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string vBiasName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.bias";
            std::string qkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.W_pack.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";
            std::string oBiasName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.bias";
            std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";

            // 1.1 Get q, k, v
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];
            if (weight.weight.find(mergeQkvWeightName) != weight.weight.end()) {
                Linear(attenInput, weight[mergeQkvWeightName], weight[mergeQkvBiasName], qkv);
                int per = qkv.dims.back() / (num_attention_heads / num_key_value_heads + 2);
                int qdim = per * (num_attention_heads / num_key_value_heads);

                Split(qkv, -1, 0, qdim, q);
                Split(qkv, -1, qdim, qdim + per, k);
                Split(qkv, -1, qdim + per, qdim + per * 2, v);
            } else {
                Linear(attenInput, weight[qWeightName], weight[qBiasName], q);
                Linear(attenInput, weight[kWeightName], weight[kBiasName], k);
                Linear(attenInput, weight[vWeightName], weight[vBiasName], v);
            }

            {
                std::vector <int> qkvSize = {bsz, seqlen, -1, head_dim};
                q.Reshape(qkvSize);
                k.Reshape(qkvSize);
                v.Reshape(qkvSize);

                Data &pastKey = pastKeyValues[i].first, &pastValue = pastKeyValues[i].second;
                if (GetKVCacheInCPU()) {
                    pastKey.lockInCPU = true;
                    pastValue.lockInCPU = true;
                } else {
                    pastKey.ToDevice(k.dataDevice);
                    pastValue.ToDevice(k.dataDevice);
                }
                int targetSeqLength = (pastKey.dims.size() > 2) ? pastKey.dims[1] + seqlen : seqlen;
                float curRopeTheta = rope_base;
                if (i == 0 && targetSeqLength >= max_positions && RoPEType::DYMAMIC_NTK == rope_type) {
                    float scale = pow((rope_factor * targetSeqLength / max_positions) - (rope_factor - 1), rotary_dim / (rotary_dim - 2));
                    curRopeTheta = rope_base * scale;
                }
                float ropeScale = (rope_type == RoPEType::LINEAR_SCALE) ? rope_factor : 1.0f;
                RMSNorm(q, this->weight["model.layers." + std::to_string(i) + ".self_attn.q_norm.weight"], rms_norm_eps, q);
                fastllm::RopeEncoding(q, positionIds, rotary_dim, curRopeTheta, ropeScale);

                RMSNorm(k, this->weight["model.layers." + std::to_string(i) + ".self_attn.k_norm.weight"], rms_norm_eps, k);
                fastllm::RopeEncoding(k, positionIds, rotary_dim, curRopeTheta, ropeScale);

                PermuteSelf(q, {0, 2, 1, 3});
                PermuteSelf(k, {0, 2, 1, 3});
                PermuteSelf(v, {0, 2, 1, 3});

                qkvSize = {-1, seqlen, head_dim};
                q.Reshape(qkvSize);
                k.Reshape(qkvSize);
                v.Reshape(qkvSize);

                {
                    // Paged Attention
                    PagedCacheManager *pagedCacheKManager = AllocatePagedCacheManager(i * 2, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, k);
                    PagedCacheManager *pagedCacheVManager = AllocatePagedCacheManager(i * 2 + 1, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, v);
                    AppendPagedCache(*pagedCacheKManager, pastKey, k);
                    AppendPagedCache(*pagedCacheVManager, pastValue, v);
                    AttentionPaged(q, pastKey, pastValue, qkv, q.dims[0] / k.dims[0], 1.0 / sqrt(head_dim), 1, i > 0);
                    PermuteSelf(qkv, {1, 0, 2});
                    qkv.Reshape({seqlen, bsz, -1});
                    PermuteSelf(qkv, {1, 0, 2});
                }
            }
            LinearAddBlock(&qkv, &weight[oWeightName], &weight[oBiasName], &attenInput, &hiddenStates);
            // 2. mlp
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], rms_norm_eps, attenInput);

            std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
            std::string downWeightName = "model.layers." + std::to_string(i) + ".mlp.down_proj.weight";
            MLPBlock(&attenInput, &weight[swigluWeightName], &weight[downWeightName], &v, &q, &hiddenStates);
        }

        Data logits, topk;
        Data tempHiddenStates;
        Data *lastHiddenStates;
        if (maxLen > 1) {
            Split(hiddenStates, 1, maxLen - 1, maxLen, tempHiddenStates);
            lastHiddenStates = &tempHiddenStates;
        } else {
            lastHiddenStates = &hiddenStates;
        }

        std::vector <int> lastRet;
        
        {
            auto &hiddenStates = *lastHiddenStates;
            RMSNorm(hiddenStates, weight["model.norm.weight"], rms_norm_eps, hiddenStates);
            Linear(hiddenStates, weight["lm_head.weight"], *GetEmptyData(), logits);
            ToDataType(logits, DataType::FLOAT32);
            TopK(logits, topk, 1);
            topk.ToDevice(DataDevice::CPU);
            for (int b = 0; b < batch; b++) {
                int base = b;
                lastRet.push_back((int) (((float *) topk.cpuData)[base * 2] + 1e-3));
            }
            return lastRet;
        }
    }
*/
    std::vector <int> Qwen3Model::ForwardGPU(
        int batch,
        const Data &inputIds,
        const std::vector <Data*> &attentionMask,
        const std::vector <Data*> &positionIds,
        const std::vector <int> &seqLens,
        std::vector <std::pair <Data*, Data*> > &pastKeyValues,
        const std::vector <GenerationConfig> &generationConfigs,
        const LastTokensManager &lastTokens,
        std::vector <std::vector <float>*> *retLogits) {
// auto startTime = std::chrono::system_clock::now();
#ifndef USE_CUDA
        return ForwardV2(batch, inputIds, attentionMask, positionIds, seqLens,
                         pastKeyValues, generationConfigs, lastTokens, retLogits);
#else
        std::vector<int> devices;
        std::map<int, int> ratios;
        if (!GetQwen3GPUForwardDevices(this->deviceMap, devices, ratios)) {
            return ForwardV2(batch, inputIds, attentionMask, positionIds, seqLens,
                             pastKeyValues, generationConfigs, lastTokens, retLogits);
        }
// printf("step 0 spend %f s.\n", GetSpan(startTime, std::chrono::system_clock::now()));
        AssertInFastLLM((int)pastKeyValues.size() >= batch * block_cnt,
                        "Qwen3 ForwardGPU: pastKeyValues size mismatch.\n");
        AssertInFastLLM((int)generationConfigs.size() >= batch,
                        "Qwen3 ForwardGPU: generation config size mismatch.\n");
        AssertInFastLLM((int)positionIds.size() >= batch && positionIds[0] != nullptr,
                        "Qwen3 ForwardGPU: positionIds size mismatch.\n");
        AssertInFastLLM(!GetKVCacheInCPU(),
                        "Qwen3 ForwardGPU doesn't support CPU KV cache.\n");
        if (devices.size() > 1) {
            AssertInFastLLM(FastllmInitNccl(devices),
                            "Qwen3 ForwardGPU requires NCCL initialization.\n");
        }

        if (threadTpPagedCacheBase < 0) {
            threadTpPagedCacheBase = qwen3ThreadTpNextPagedCacheBase.fetch_add(
                std::max(1, block_cnt * ((int)devices.size() + 1)));
        }
// printf("step 1 spend %f s.\n", GetSpan(startTime, std::chrono::system_clock::now()));
        int seqLen = inputIds.dims[1];
        bool all1 = true;
        for (int i = 0; i < batch; i++) {
            all1 &= (seqLens[i] == 1);
        }
        bool isPrefill = !all1;

        Data allPositionIds;
        if (all1 && positionIds[0]->dataType == DataType::FLOAT32) {
            std::vector<float> vPositionIds;
            vPositionIds.reserve(batch);
            for (int b = 0; b < batch; b++) {
                vPositionIds.push_back(((float*)positionIds[b]->cpuData)[0]);
            }
            allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, vPositionIds));
        } else {
            std::vector<float> vPositionIds;
            for (int b = 0; b < batch; b++) {
                AssertInFastLLM(positionIds[b] != nullptr,
                                "Qwen3 ForwardGPU: null positionIds.\n");
                for (int i = 0; i < seqLens[b]; i++) {
                    vPositionIds.push_back(((float*)positionIds[b]->cpuData)[i]);
                }
            }
            allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, (int)vPositionIds.size()}, vPositionIds));
        }
// printf("step 2 spend %f s.\n", GetSpan(startTime, std::chrono::system_clock::now()));
        Data gpuInputIds;
        gpuInputIds.CopyFrom(inputIds);
        PrepareMultiCudaReplicatedData(gpuInputIds, devices, true);
        PrepareMultiCudaReplicatedData(allPositionIds, devices, true);
// printf("step 3 spend %f s.\n", GetSpan(startTime, std::chrono::system_clock::now()));
        std::vector<DivisionScheme> kvHeadSchemes;
        DivisionScheme lmHeadScheme;
// printf("step 4 spend %f s.\n", GetSpan(startTime, std::chrono::system_clock::now()));
        {
            std::lock_guard<std::mutex> guard(threadTpWeightPrepareLock);
            if (threadTpWeightsPrepared) {
                AssertInFastLLM(threadTpPreparedDevices == devices && threadTpPreparedRatios == ratios,
                                "Qwen3 ForwardGPU thread TP device config changed after weights were prepared.\n");
                AssertInFastLLM((int)threadTpKVHeadSchemes.size() == block_cnt && !threadTpLmHeadScheme.empty(),
                                "Qwen3 ForwardGPU thread TP cached weight schemes are incomplete.\n");
            } else {
                auto prepareReplicated = [&](const std::string &name) {
                    PrepareMultiCudaReplicatedData(this->weight[name], devices, true);
                };
                prepareReplicated("model.embed_tokens.weight");
                prepareReplicated("model.norm.weight");

                threadTpKVHeadSchemes.assign(block_cnt, DivisionScheme());
                for (int i = 0; i < block_cnt; i++) {
                    std::string inputRmsName = "model.layers." + std::to_string(i) + ".input_layernorm.weight";
                    std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
                    std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
                    std::string qNormName = "model.layers." + std::to_string(i) + ".self_attn.q_norm.weight";
                    std::string kNormName = "model.layers." + std::to_string(i) + ".self_attn.k_norm.weight";
                    std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";
                    std::string oBiasName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.bias";
                    std::string postRmsName = "model.layers." + std::to_string(i) + ".post_attention_layernorm.weight";
                    std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
                    std::string downWeightName = "model.layers." + std::to_string(i) + ".mlp.down_proj.weight";
                    std::string downBiasName = "model.layers." + std::to_string(i) + ".mlp.down_proj.bias";

                    AssertInFastLLM(weight.weight.find(mergeQkvWeightName) != weight.weight.end(),
                                    "Qwen3 ForwardGPU requires merged qkv weight.\n");
                    AssertInFastLLM(weight.weight.find(swigluWeightName) != weight.weight.end(),
                                    "Qwen3 ForwardGPU requires merged gateup weight.\n");

                    prepareReplicated(inputRmsName);
                    prepareReplicated(qNormName);
                    prepareReplicated(kNormName);
                    prepareReplicated(postRmsName);

                    Data &mergeW = weight[mergeQkvWeightName];
                    Data &mergeB = GetThreadTensorParallelBias(mergeQkvBiasName);
                    mergeW.tpPackType = TP_PACK_QKV;
                    mergeW.tpQHeads = num_attention_heads;
                    mergeW.tpKVHeads = num_key_value_heads;
                    mergeW.tpHeadDim = head_dim;
                    std::vector<int> devCopy = devices;
                    DivisionScheme qkvScheme = BuildMultiCudaRowSplitScheme(mergeW, devCopy, ratios);
                    AssertInFastLLM(SplitMultiCudaWeight(mergeW, mergeB, devCopy, qkvScheme, 0),
                                    "Qwen3 ForwardGPU failed to split " + mergeQkvWeightName + ".\n");

                    int qWidth = num_attention_heads * head_dim;
                    DivisionScheme qScheme = ExtractQwen3FirstRangeScheme(qkvScheme);
                    threadTpKVHeadSchemes[i] = ExtractQwen3KVHeadScheme(qkvScheme, qWidth, head_dim);
                    Data &oB = GetThreadTensorParallelBias(oBiasName);
                    devCopy = devices;
                    AssertInFastLLM(SplitMultiCudaWeight(weight[oWeightName], oB, devCopy, qScheme, 1),
                                    "Qwen3 ForwardGPU failed to split " + oWeightName + ".\n");

                    Data &gateup = weight[swigluWeightName];
                    Data &gateupBias = GetThreadTensorParallelBias(swigluWeightName + ".tp_bias");
                    gateup.tpPackType = TP_PACK_GATEUP;
                    devCopy = devices;
                    DivisionScheme gateScheme = BuildMultiCudaRowSplitScheme(gateup, devCopy, ratios);
                    AssertInFastLLM(SplitMultiCudaWeight(gateup, gateupBias, devCopy, gateScheme, 0),
                                    "Qwen3 ForwardGPU failed to split " + swigluWeightName + ".\n");

                    Data &downBias = GetThreadTensorParallelBias(downBiasName);
                    DivisionScheme downScheme = ExtractQwen3FirstRangeScheme(gateScheme);
                    devCopy = devices;
                    AssertInFastLLM(SplitMultiCudaWeight(weight[downWeightName], downBias, devCopy, downScheme, 1),
                                    "Qwen3 ForwardGPU failed to split " + downWeightName + ".\n");
                }

                Data &lmHead = weight["lm_head.weight"];
                Data &lmHeadBias = GetThreadTensorParallelBias("lm_head.weight.tp_bias");
                std::vector<int> devCopy = devices;
                threadTpLmHeadScheme = BuildMultiCudaRowSplitScheme(lmHead, devCopy, ratios);
                AssertInFastLLM(SplitMultiCudaWeight(lmHead, lmHeadBias, devCopy, threadTpLmHeadScheme, 0),
                                "Qwen3 ForwardGPU failed to split lm_head.weight.\n");

                threadTpPreparedDevices = devices;
                threadTpPreparedRatios = ratios;
                threadTpWeightsPrepared = true;
            }
            kvHeadSchemes = threadTpKVHeadSchemes;
            lmHeadScheme = threadTpLmHeadScheme;
        }
// printf("step 5 spend %f s.\n", GetSpan(startTime, std::chrono::system_clock::now()));
        Data &lmHead = weight["lm_head.weight"];
// printf("step 6 spend %f s.\n", GetSpan(startTime, std::chrono::system_clock::now()));
        const DataType computeType = ResolveQwen3ThreadTpComputeType(this->dataType);
        std::vector<std::vector<std::pair<Data*, Data*> > > localPastKeyValues(devices.size());
        for (int r = 0; r < (int)devices.size(); r++) {
            int device = devices[r];
            localPastKeyValues[r].resize(pastKeyValues.size());
            for (int i = 0; i < (int)pastKeyValues.size(); i++) {
                DataType keyCacheType = ResolveQwen3ThreadTpCacheType(
                    pastKeyValues[i].first->dataType, computeType);
                DataType valueCacheType = ResolveQwen3ThreadTpCacheType(
                    pastKeyValues[i].second->dataType, computeType);
                localPastKeyValues[r][i].first = EnsureQwen3ThreadTpLocalCache(
                    *pastKeyValues[i].first, device, keyCacheType);
                localPastKeyValues[r][i].second = EnsureQwen3ThreadTpLocalCache(
                    *pastKeyValues[i].second, device, valueCacheType);
            }
        }
// printf("step 7 spend %f s.\n", GetSpan(startTime, std::chrono::system_clock::now()));
        std::vector<std::exception_ptr> errors(devices.size());
        std::vector<Data> localLogits(devices.size());
        bool tensorParallel = devices.size() > 1;
// printf("step 8 spend %f s.\n", GetSpan(startTime, std::chrono::system_clock::now()));
        if (devices.size() == 1) {
            ForwardSingleGPU(devices[0], ratios, batch, gpuInputIds, allPositionIds,
                             seqLens, localPastKeyValues[0], all1, isPrefill,
                             false, true, threadTpPagedCacheBase, localLogits[0]);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(devices.size());
            for (int r = 0; r < (int)devices.size(); r++) {
                threads.emplace_back([&, r]() {
                    try {
                        ForwardSingleGPU(devices[r], ratios, batch, gpuInputIds, allPositionIds,
                                         seqLens, localPastKeyValues[r], all1, isPrefill,
                                         tensorParallel, r == 0, threadTpPagedCacheBase + r * block_cnt,
                                         localLogits[r]);
                    } catch (...) {
                        errors[r] = std::current_exception();
                    }
                });
            }
            for (auto &thread : threads) {
                thread.join();
            }
            for (auto &error : errors) {
                if (error) {
                    std::rethrow_exception(error);
                }
            }
            for (int device : devices) {
                FastllmCudaSetDevice(device);
                ForceDeviceSync();
            }
        }

        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < block_cnt; i++) {
                int idx = b * block_cnt + i;
                SyncQwen3ThreadTpRootCacheMeta(*pastKeyValues[idx].first, devices, kvHeadSchemes[i],
                                               num_key_value_heads, head_dim);
                SyncQwen3ThreadTpRootCacheMeta(*pastKeyValues[idx].second, devices, kvHeadSchemes[i],
                                               num_key_value_heads, head_dim);
            }
        }

        int vocabSize = lmHead.dims[0];
        bool allSimpleCudaSampling = true;
        int cudaSamplingTopK = 1;
        if (Qwen3CanUseCudaFullLogitsSampling(generationConfigs, retLogits, batch,
                                              allSimpleCudaSampling, cudaSamplingTopK)) {
            Data &fullCudaLogits = Qwen3ThreadLocalCudaSamplingFullLogits();
            Qwen3GatherShardLogitsToRootCuda(devices[0], devices, lmHeadScheme,
                                             localLogits, batch, vocabSize,
                                             fullCudaLogits);
            ResetLogitsOfEOS(batch, &fullCudaLogits, pastKeyValues, generationConfigs);
            return Qwen3SampleFromRootCudaLogits(devices[0], fullCudaLogits, batch,
                                                 cudaSamplingTopK, allSimpleCudaSampling,
                                                 generationConfigs);
        }

        Data fullLogits(DataType::FLOAT32);
        fullLogits.Resize({batch, vocabSize});
        fullLogits.Allocate();
        std::fill((float*)fullLogits.cpuData,
                  (float*)fullLogits.cpuData + fullLogits.Count(0), -1.0e30f);

        for (int r = 0; r < (int)devices.size(); r++) {
            int device = devices[r];
            localLogits[r].ToDevice(DataDevice::CPU);
            int localVocab = localLogits[r].dims.back();
            int rows = localLogits[r].Count(0) / localVocab;
            AssertInFastLLM(rows == batch,
                            "Qwen3 ForwardGPU: local logits batch mismatch.\n");
            float *src = (float*)localLogits[r].cpuData;
            float *dst = (float*)fullLogits.cpuData;
            int localOffset = 0;
            for (auto &range : lmHeadScheme[device]) {
                int len = range.second - range.first;
                AssertInFastLLM(range.first >= 0 && range.second <= vocabSize &&
                                localOffset + len <= localVocab,
                                "Qwen3 ForwardGPU: invalid lm_head split range.\n");
                for (int b = 0; b < batch; b++) {
                    memcpy(dst + (long long)b * vocabSize + range.first,
                           src + (long long)b * localVocab + localOffset,
                           (size_t)len * sizeof(float));
                }
                localOffset += len;
            }
        }

        ResetLogitsOfEOS(batch, &fullLogits, pastKeyValues, generationConfigs);

        std::vector<int> lastRet;
        LastTokensUnit emptyLastTokens;
        for (int b = 0; b < batch; b++) {
            if (generationConfigs[b].output_logits && retLogits != nullptr &&
                b < (int)retLogits->size() && (*retLogits)[b] != nullptr) {
                (*retLogits)[b]->resize(vocabSize);
                memcpy((float*)(*retLogits)[b]->data(),
                       (float*)fullLogits.cpuData + (long long)b * vocabSize,
                       (size_t)vocabSize * sizeof(float));
            }
            const LastTokensUnit &unit = b < (int)lastTokens.units.size() ?
                lastTokens.units[b] : emptyLastTokens;
            lastRet.push_back(LLMSampling(fullLogits, b, generationConfigs[b], unit));
        }
        return lastRet;
#endif
    }

    std::vector <int> Qwen3Model::ForwardV2ThreadTensorParallel(
        int batch,
        const Data &inputIds,
        const std::vector <Data*> &attentionMask,
        const std::vector <Data*> &positionIds,
        const std::vector <int> &seqLens,
        std::vector <std::pair <Data*, Data*> > &pastKeyValues,
        const std::vector <GenerationConfig> &generationConfigs,
        const LastTokensManager &lastTokens,
        std::vector <std::vector <float>*> *retLogits) {
        return ForwardGPU(batch, inputIds, attentionMask, positionIds, seqLens,
                          pastKeyValues, generationConfigs, lastTokens, retLogits);
    }

    std::vector <int> Qwen3Model::ForwardV2(
        int batch,
        const Data &inputIds,
        const std::vector <Data*> &attentionMask,
        const std::vector <Data*> &positionIds,
        const std::vector <int> &seqLens,
        std::vector <std::pair <Data*, Data*> > &pastKeyValues,
        const std::vector <GenerationConfig> &generationConfigs,
        const LastTokensManager &lastTokens,
        std::vector <std::vector <float>*> *retLogits) {
        int seqLen = inputIds.dims[1];

        Data qkv;
        // Data &qkv = this->forwardDataManager.GetData("qkv");
        Data q;
        // Data &q = this->forwardDataManager.GetData("q");
        Data k;
        // Data &k = this->forwardDataManager.GetData("k");
        Data v;
        // Data &v = this->forwardDataManager.GetData("v");
        Data embeddingResult;
        // Data &embeddingResult = this->forwardDataManager.GetData("embeddingResult");
        Data hiddenStates;
        // Data &hiddenStates = this->forwardDataManager.GetData("hiddenStates");
        Data attenInput;
        // Data &attenInput = this->forwardDataManager.GetData("attenInput");
        Data attenLastOutput;
        // Data &attenLastOutput = this->forwardDataManager.GetData("attenLastOutput");
        std::vector <Data*> pointersK;
        pointersK.resize(batch);


        std::vector<Data*> batchPastKeys;
        std::vector<Data*> batchPastValues;
        batchPastKeys.resize(batch);
        batchPastValues.resize(batch);

        Data allPositionIds;
        // Data &allPositionIds = this->forwardDataManager.GetData("allPositionIds");
        Data qSizes;
        // Data &qSizes = this->forwardDataManager.GetData("qSizes");
        Data pageSizes;
        // Data &pageSizes = this->forwardDataManager.GetData("pageSizes");
        Data pageIndexs;
        // Data &pageIndexs = this->forwardDataManager.GetData("pageIndexs");
        Data lastPageLens;
        // Data &lastPageLens = this->forwardDataManager.GetData("lastPageLens");
        Data insertIndexs;
        // Data &insertIndexs = this->forwardDataManager.GetData("insertIndexs");
        Data insertPositions;
        // Data &insertPositions = this->forwardDataManager.GetData("insertPositions");
        Data attenOutput;
        // Data &attenOutput = this->forwardDataManager.GetData("attenOutput");
        bool generatedBatchDecodeParams = false;
        bool generatedAppendPagedCacheBatchParams = false;

        bool all1 = true;
        for (int i = 0; i < batch; i++) {
            all1 &= (seqLens[i] == 1);
        }
        bool isPrefill = !all1;
        if (all1 && positionIds[0]->dataType == DataType::FLOAT32) {
            std::vector <float> vPositionIds;            
            for (int b = 0; b < batch; b++) {
                vPositionIds.push_back(((float*)positionIds[b]->cpuData)[0]);
            }
            allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, vPositionIds));
        } else {
            std::vector <float> vPositionIds;            
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < seqLens[b]; i++) {
                    vPositionIds.push_back(((float*)positionIds[b]->cpuData)[i]);
                }
            }
            allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, (int)vPositionIds.size()}, vPositionIds));
        }

        EmbeddingBlock((Data*)&inputIds, 
            &this->weight["model.embed_tokens.weight"], &hiddenStates, this->dataType);
        // Embedding(inputIds, this->weight["model.embed_tokens.weight"], embeddingResult);
        // ToDataType(embeddingResult, hiddenStates, this->dataType);

        int seqlen = hiddenStates.dims[1];
        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            std::string inputRmsName = "model.layers." + std::to_string(i) + ".input_layernorm.weight";
            std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string qBiasName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.bias";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string kBiasName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.bias";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string vBiasName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.bias";
            std::string qNormName = "model.layers." + std::to_string(i) + ".self_attn.q_norm.weight";
            std::string kNormName = "model.layers." + std::to_string(i) + ".self_attn.k_norm.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";
            std::string oBiasName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.bias";
            std::string postRmsName = "model.layers." + std::to_string(i) + ".post_attention_layernorm.weight";
            std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
            std::string downWeightName = "model.layers." + std::to_string(i) + ".mlp.down_proj.weight";

            bool hasMergeQkv = (weight.weight.find(mergeQkvWeightName) != weight.weight.end());
            Data *mergeW = hasMergeQkv ? &weight[mergeQkvWeightName] : GetEmptyData();
            Data *mergeB = hasMergeQkv ? &weight[mergeQkvBiasName] : GetEmptyData();
            Data *qW = hasMergeQkv ? GetEmptyData() : &weight[qWeightName];
            Data *qB = hasMergeQkv ? GetEmptyData() : &weight[qBiasName];
            Data *kW = hasMergeQkv ? GetEmptyData() : &weight[kWeightName];
            Data *kB = hasMergeQkv ? GetEmptyData() : &weight[kBiasName];
            Data *vW = hasMergeQkv ? GetEmptyData() : &weight[vWeightName];
            Data *vB = hasMergeQkv ? GetEmptyData() : &weight[vBiasName];

            RMSNorm(hiddenStates, this->weight[inputRmsName], rms_norm_eps, attenInput);
            AttentionPagedBlock(
                &attenInput,
                mergeW, mergeB,
                qW, qB, kW, kB, vW, vB,
                GetEmptyData(), GetEmptyData(),
                &weight[qNormName], &weight[kNormName],
                &weight[oWeightName], &weight[oBiasName],
                &allPositionIds,
                &pastKeyValues, &batchPastKeys, &batchPastValues,
                &qkv, &q, &attenOutput, &attenLastOutput,
                &insertIndexs, &insertPositions,
                &qSizes, &pageSizes, &pageIndexs, &lastPageLens,
                &generatedAppendPagedCacheBatchParams, &generatedBatchDecodeParams,
                batch, block_cnt, i,
                seqLens,
                num_attention_heads, num_key_value_heads, head_dim,
                rotary_dim, rms_norm_eps,
                rope_base, rope_factor, max_positions,
                rope_type,
                GetKVCacheInCPU(),
                isPrefill,
                &hiddenStates,
                true,
                false
            );

            RMSNorm(hiddenStates, this->weight[postRmsName], rms_norm_eps, attenInput);
            MLPBlock(&attenInput, &weight[swigluWeightName], &weight[downWeightName], &v, &q, &hiddenStates);
        }

        std::vector <int> lastRet;
        LLMSamplingBlock(
            this, &hiddenStates,
            &weight["model.norm.weight"], &weight["lm_head.weight"],
            rms_norm_eps, batch, all1, seqLens,
            pastKeyValues, generationConfigs, lastTokens,
            retLogits, lastRet
        );
        return lastRet;
    }

    std::vector <int> Qwen3Model::ForwardBatch(int batch, const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <std::vector <float>*> *retLogits) {
        int maxLen = inputIds.dims[1];
        Data hiddenStates;
        Data attenInput;
        Data q, k, v, qkv, curInput, curOutput;
        Data attenWeights, attenOutput;
        Data attenLastOutput;
        Data w1, w2, w3;
        Data* sinDataPtr = &sinData;
        Data* cosDataPtr = &cosData;

        Embedding(inputIds, this->weight["model.embed_tokens.weight"], hiddenStates);
        ToDataType(hiddenStates, this->dataType);

        int seqlen = hiddenStates.dims[1];
        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    rms_norm_eps, attenInput);
            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string qBiasName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.bias";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string kBiasName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.bias";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string vBiasName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.bias";
            std::string qkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.W_pack.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";
            std::string oBiasName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.bias";
            std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";

            // 1.1 Get q, k, v
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];

            if (weight.weight.find(mergeQkvWeightName) != weight.weight.end()
                && CanRunMergeAttention()
                && true) {
                std::vector <Data*> keys, values, masks;
                keys.push_back(&pastKeyValues[i].first);
                values.push_back(&pastKeyValues[i].second);
                masks.push_back((Data*)&attentionMask);
                MergeAttention (
                    attenInput, 
                    weight[mergeQkvWeightName], weight[mergeQkvBiasName], 
                    weight[oWeightName], weight[oBiasName],
                    true,
                    this->weight["model.layers." + std::to_string(i) + ".self_attn.q_norm.weight"],
                    this->weight["model.layers." + std::to_string(i) + ".self_attn.k_norm.weight"],
                    rms_norm_eps,
                    qkv, q, k, v,
                    num_attention_heads, num_key_value_heads, head_dim, rotary_dim, 1.0 / sqrt(head_dim),
                    positionIds, *sinDataPtr, *cosDataPtr, 
                    keys, values, masks, attenLastOutput
                );
                AddTo(hiddenStates, attenLastOutput);
            } else {
                if (weight.weight.find(qkvWeightName) != weight.weight.end()) {
                    Linear(attenInput, weight[qkvWeightName], *GetEmptyData(), qkv);
                    int per = qkv.dims.back() / (num_attention_heads / num_key_value_heads + 2);
                    int qdim = per * (num_attention_heads / num_key_value_heads);
                    Split(qkv, -1, 0, qdim, q);
                    Split(qkv, -1, qdim, qdim + per, k);
                    Split(qkv, -1, qdim + per, qdim + per * 2, v);
                } else {
                    if (weight.weight.find(mergeQkvWeightName) != weight.weight.end()) {
                        Linear(attenInput, weight[mergeQkvWeightName], weight[mergeQkvBiasName], qkv);
                        int per = qkv.dims.back() / (num_attention_heads / num_key_value_heads + 2);
                        int qdim = per * (num_attention_heads / num_key_value_heads);

                        Split(qkv, -1, 0, qdim, q);
                        Split(qkv, -1, qdim, qdim + per, k);
                        Split(qkv, -1, qdim + per, qdim + per * 2, v);
                    } else {
                        Linear(attenInput, weight[qWeightName], weight[qBiasName], q);
                        Linear(attenInput, weight[kWeightName], weight[kBiasName], k);
                        Linear(attenInput, weight[vWeightName], weight[vBiasName], v);
                    }
                }

                std::vector <int> qkvSize = {bsz, seqlen, -1, head_dim};
                q.Reshape(qkvSize);
                k.Reshape(qkvSize);
                v.Reshape(qkvSize);

                RMSNorm(q, this->weight["model.layers." + std::to_string(i) + ".self_attn.q_norm.weight"], rms_norm_eps, q);
                RMSNorm(k, this->weight["model.layers." + std::to_string(i) + ".self_attn.k_norm.weight"], rms_norm_eps, k);

                Data &pastKey = pastKeyValues[i].first, &pastValue = pastKeyValues[i].second;
                if (GetKVCacheInCPU()) {
                    pastKey.lockInCPU = true;
                    pastValue.lockInCPU = true;
                } else {
                    pastKey.ToDevice(k.dataDevice);
                    pastValue.ToDevice(k.dataDevice);
                }
                int targetSeqLength = (pastKey.dims.size() > 2) ? pastKey.dims[1] + seqlen : seqlen;
                if (i == 0 && targetSeqLength >= max_positions && RoPEType::DYMAMIC_NTK == rope_type) {
                    float scale = pow((rope_factor * targetSeqLength / max_positions) - (rope_factor - 1), rotary_dim / (rotary_dim - 2));
                    float newbase = rope_base * scale;
                    std::pair<std::vector<float>, std::vector<float>> &&pair = this->UpdateRotaryPosEmb(newbase, rope_factor, targetSeqLength);
                    sinDataPtr = new Data(DataType::FLOAT32, {(int)this->sin.size(), (int)this->sin[0].size()}, pair.first);
                    cosDataPtr = new Data(DataType::FLOAT32, {(int)this->cos.size(), (int)this->cos[0].size()}, pair.second);
                }

                fastllm::LlamaRotatePosition2D(q, positionIds, *sinDataPtr, *cosDataPtr, rotary_dim);
                fastllm::LlamaRotatePosition2D(k, positionIds, *sinDataPtr, *cosDataPtr, rotary_dim);

                PermuteSelf(q, {0, 2, 1, 3});
                PermuteSelf(k, {0, 2, 1, 3});
                PermuteSelf(v, {0, 2, 1, 3});

                qkvSize = {-1, seqlen, head_dim};
                q.Reshape(qkvSize);
                k.Reshape(qkvSize);
                v.Reshape(qkvSize);

                int unitLen = 64;
    #ifdef USE_CUDA
                unitLen = 128;
    #endif
                while ((pastKey.dims.size() == 0 && (pastKey.expansionDims.size() == 0 || k.dims[1] > pastKey.expansionDims[1]))
                    || (pastKey.dims.size() > 0 && pastKey.dims[1] + k.dims[1] > pastKey.expansionDims[1])) {
                    std::vector <int> newDims;
                    if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                        newDims = std::vector <int> {k.dims[0], ((k.dims[1] - 1) / unitLen + 1) * unitLen, k.dims[2]};
                    } else {
                        newDims = pastKey.dims;
                        newDims[1] += ((k.dims[1] - 1) / unitLen + 1) * unitLen;
                    }
                    pastKey.Expansion(newDims);
                }
                while ((pastValue.dims.size() == 0 && (pastValue.expansionDims.size() == 0 || v.dims[1] > pastValue.expansionDims[1]))
                    || (pastValue.dims.size() > 0 && pastValue.dims[1] + v.dims[1] > pastValue.expansionDims[1])) {
                    std::vector <int> newDims;
                    if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                        newDims = std::vector <int> {v.dims[0], ((v.dims[1] - 1) / unitLen + 1) * unitLen, v.dims[2]};
                    } else {
                        newDims = pastValue.dims;
                        newDims[1] += ((v.dims[1] - 1) / unitLen + 1) * unitLen;
                    }
                    pastValue.Expansion(newDims);
                }

                CatDirect(pastKey, k, 1);
                CatDirect(pastValue, v, 1);

                // 1.2 Attention
                Attention(q, pastKey, pastValue, attentionMask, qkv, q.dims[0] / pastKey.dims[0], 1.0 / sqrt(head_dim), 1);

                PermuteSelf(qkv, {1, 0, 2});
                qkv.Reshape({seqlen, bsz, -1});
                PermuteSelf(qkv, {1, 0, 2});

                Linear(qkv, weight[oWeightName], weight[oBiasName], attenInput);
                AddTo(hiddenStates, attenInput);
            }

            // 2. mlp
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], rms_norm_eps, attenInput);

            std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
            if (weight.weight.find(swigluWeightName) != weight.weight.end() && CanRunMLP()) {
                std::string downWeightName = "model.layers." + std::to_string(i) + ".mlp.down_proj.weight";
                MLP(attenInput, weight[swigluWeightName], *GetEmptyData(), weight[downWeightName], *GetEmptyData(), w1, w2, w3, k);
                AddTo(hiddenStates, k);
            } else {
                if (weight.weight.find(swigluWeightName) != weight.weight.end()) {
                    if (CanRunLinearEx(LinearExType::ExSwiglu)) {
                        LinearEx(attenInput, weight[swigluWeightName], *GetEmptyData(), q, LinearExType::ExSwiglu);
                    } else {
                        Linear(attenInput, weight[swigluWeightName], *GetEmptyData(), v);
                        Swiglu(v, q);
                    }
                } else {
                    if (CanRunLinearEx(LinearExType::ExSilu)) {
                        LinearEx(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], *GetEmptyData(), q, LinearExType::ExSilu);
                    } else {
                        Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], *GetEmptyData(), q);
                        Silu(q, q);
                    }
                    Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.up_proj.weight"], *GetEmptyData(), v);
                    MulTo(q, v);
                }
                Linear(q, weight["model.layers." + std::to_string(i) + ".mlp.down_proj.weight"], *GetEmptyData(), k);
                AddTo(hiddenStates, k);
            }
        }

        Data logits, topk;
        Data tempHiddenStates;
        Data *lastHiddenStates;
        if (maxLen > 1) {
            Split(hiddenStates, 1, maxLen - 1, maxLen, tempHiddenStates);
            lastHiddenStates = &tempHiddenStates;
        } else {
            lastHiddenStates = &hiddenStates;
        }

        std::vector <int> lastRet;
        {
            auto &hiddenStates = *lastHiddenStates;
            RMSNorm(hiddenStates, weight["model.norm.weight"], rms_norm_eps, hiddenStates);
            Linear(hiddenStates, weight["lm_head.weight"], *GetEmptyData(), logits);
            ToDataType(logits, DataType::FLOAT32);
            if (generationConfig.output_logits && retLogits != nullptr) {
                int size = logits.dims.back();
                logits.ToDevice(DataDevice::CPU);
                for (int b = 0; b < batch; b++) {
                    (*retLogits)[b]->resize(size);
                    memcpy((float*)(*retLogits)[b]->data(), 
                        ((float*)logits.cpuData) + ((b + 1) * logits.dims[1] - 1) * size, 
                        size * logits.unitSize);
                }
            }

            if (generationConfig.top_k <= 1) {
                // 禁用simple greedy
                ((GenerationConfig*)&generationConfig)->top_k = 5;
                ((GenerationConfig*)&generationConfig)->top_p = 0.95;
                if (fabs(generationConfig.temperature - 1.0f) < 1e-9) {
                    ((GenerationConfig*)&generationConfig)->temperature = 0.6;
                }
            }

            ResetLogitsOfEOS(batch, &logits, pastKeyValues, generationConfig);
            if (generationConfig.IsSimpleGreedy()) {
                TopK(logits, topk, 1);
                topk.ToDevice(DataDevice::CPU);
                for (int b = 0; b < batch; b++) {
                    int base = b;
                    lastRet.push_back((int) (((float *) topk.cpuData)[base * 2] + 1e-3));
                }
            } else if (generationConfig.top_k <= 50 && !generationConfig.output_logits) {
                if ((generationConfig.repeat_penalty - 1.0f) > 1e-9) {
                    int maxTokenSetSize = 0;
                    for (int b = 0; b < batch; b++) {
                        maxTokenSetSize = std::max(maxTokenSetSize, (int)lastTokens.units[b].tokenSet.size());
                    }
                    std::vector <float> penaltyData = std::vector <float> (batch * maxTokenSetSize, -100.0f);
                    std::vector <float> penaltyScaleData = std::vector <float> (batch, 1.0f);
                    for (int b = 0; b < batch; b++) {
                        int curId = 0;
                        for (int i : lastTokens.units[b].tokenSet) {
                            penaltyData[b * maxTokenSetSize + curId] = i;
                            curId++;
                        }
                        penaltyScaleData[b] = generationConfig.repeat_penalty;
                    }
                    Data penalty, penaltyScale;
                    penalty.CopyFrom(Data(DataType::FLOAT32, {batch, maxTokenSetSize}, penaltyData));
                    penaltyScale.CopyFrom(Data(DataType::FLOAT32, {batch}, penaltyScaleData));
                    RepeatPenalty(logits, penalty, penaltyScale);
                }

                Data topk;
                TopK(logits, topk, generationConfig.top_k);
                topk.ToDevice(DataDevice::CPU);
                for (int b = 0; b < batch; b++) {
                    lastRet.push_back(LLMSamplingOnly(topk, b, generationConfig));
                }
            } else {
                for (int b = 0; b < batch; b++) {
                    int base = b * logits.dims[1] + logits.dims[1] - 1;
                    lastRet.push_back(LLMSampling(logits, base, generationConfig, lastTokens.units[b]));
                }
            }
        }
        if (sinDataPtr != &sinData)
            delete sinDataPtr;
        if (cosDataPtr != &cosData)
            delete cosDataPtr;

        return lastRet;
    }

    std::vector <int> Qwen3Model::ForwardBatch(int batch,
                                               const Data &inputIds,
                                               const std::vector <Data*> &attentionMask,
                                               const std::vector <Data*> &positionIds,
                                               const std::vector <int> &seqLens,
                                               std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                                               const std::vector <GenerationConfig> &generationConfigs,
                                               const LastTokensManager &lastTokens,
                                               std::vector <std::vector <float>*> *retLogits) {
        int seqLen = inputIds.dims[1];

        Data hiddenStates;
        Data attenInput;
        Data q, k, v, qkv;
        Data attenWeights, curAttenOutput;
        Data attenLastOutput;
        Data w1, w2, w3;
        Data* sinDataPtr = &sinData;
        Data* cosDataPtr = &cosData;
        std::vector <Data> curContextLayer;
        curContextLayer.resize(batch);
        std::vector <Data> curKs, curVs, curQs;
        curKs.resize(batch);
        curVs.resize(batch);
        curQs.resize(batch);
        std::vector <Data*> pointersK, pointersV, pointersQ;
        pointersK.resize(batch);
        pointersV.resize(batch);
        pointersQ.resize(batch);
        std::vector <Data*> keys, values, qs, attns, masks, contexts;
        keys.resize(batch);
        values.resize(batch);
        qs.resize(batch);
        attns.resize(batch);
        masks.resize(batch);
        contexts.resize(batch);
        Data allPositionIds;

        bool all1 = true;
        for (int i = 0; i < batch; i++) {
            all1 &= (seqLens[i] == 1);
        }
        if (all1 && positionIds[0]->dataType == DataType::FLOAT32) {
            std::vector <float> vPositionIds;            
            for (int b = 0; b < batch; b++) {
                vPositionIds.push_back(((float*)positionIds[b]->cpuData)[0]);
            }
            allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, vPositionIds));
        } else {
            allPositionIds.CopyFrom(*(Data*)positionIds[0]);
            allPositionIds.Expansion({1, seqLen});
            for (int i = 1; i < batch; i++) {
                CatDirect(allPositionIds, *(Data*)positionIds[i], 1);
            }
        }

        Embedding(inputIds, this->weight["model.embed_tokens.weight"], hiddenStates);
        ToDataType(hiddenStates, this->dataType);

        int seqlen = hiddenStates.dims[1];
        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    rms_norm_eps, attenInput);
            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string qBiasName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.bias";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string kBiasName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.bias";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string vBiasName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.bias";
            std::string qkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.W_pack.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";
            std::string oBiasName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.bias";
            std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";

            // 1.1 Get q, k, v
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];
            if (weight.weight.find(mergeQkvWeightName) != weight.weight.end()
                && CanRunMergeAttention()
                && true) {
                std::vector <Data*> keys, values, masks;
                for (int b = 0; b < batch; b++) {
                    keys.push_back(pastKeyValues[b * block_cnt + i].first);
                    values.push_back(pastKeyValues[b * block_cnt + i].second);
                    masks.push_back(attentionMask[b]);
                }
                MergeAttention (
                    attenInput, 
                    weight[mergeQkvWeightName], weight[mergeQkvBiasName], 
                    weight[oWeightName], weight[oBiasName],
                    true,
                    this->weight["model.layers." + std::to_string(i) + ".self_attn.q_norm.weight"],
                    this->weight["model.layers." + std::to_string(i) + ".self_attn.k_norm.weight"],
                    rms_norm_eps,
                    qkv, q, k, v, 
                    num_attention_heads, num_key_value_heads, head_dim, rotary_dim, 1.0 / sqrt(head_dim),
                    allPositionIds, *sinDataPtr, *cosDataPtr, 
                    keys, values, masks, attenLastOutput
                );
                AddTo(hiddenStates, attenLastOutput);
            } else {
                if (weight.weight.find(qkvWeightName) != weight.weight.end()) {
                    Linear(attenInput, weight[qkvWeightName], *GetEmptyData(), qkv);
                    int per = qkv.dims.back() / (num_attention_heads / num_key_value_heads + 2);
                    int qdim = per * (num_attention_heads / num_key_value_heads);
                    Split(qkv, -1, 0, qdim, q);
                    Split(qkv, -1, qdim, qdim + per, k);
                    Split(qkv, -1, qdim + per, qdim + per * 2, v);
                } else {
                    if (weight.weight.find(mergeQkvWeightName) != weight.weight.end()) {
                        Linear(attenInput, weight[mergeQkvWeightName], weight[mergeQkvBiasName], qkv);
                        int per = qkv.dims.back() / (num_attention_heads / num_key_value_heads + 2);
                        int qdim = per * (num_attention_heads / num_key_value_heads);

                        Split(qkv, -1, 0, qdim, q);
                        Split(qkv, -1, qdim, qdim + per, k);
                        Split(qkv, -1, qdim + per, qdim + per * 2, v);
                    } else {
                        Linear(attenInput, weight[qWeightName], weight[qBiasName], q);
                        Linear(attenInput, weight[kWeightName], weight[kBiasName], k);
                        Linear(attenInput, weight[vWeightName], weight[vBiasName], v);
                    }
                }

                q.Reshape({q.dims[0], q.dims[1], -1, head_dim});
                k.Reshape({k.dims[0], k.dims[1], -1, head_dim});
                v.Reshape({v.dims[0], v.dims[1], -1, head_dim});

                RMSNorm(q, this->weight["model.layers." + std::to_string(i) + ".self_attn.q_norm.weight"], rms_norm_eps, q);
                RMSNorm(k, this->weight["model.layers." + std::to_string(i) + ".self_attn.k_norm.weight"], rms_norm_eps, k);

                int cacheOuter = k.dims[2], cacheInner = k.dims[3];
                int targetSeqLength = 0;
                for (int b = 0; b < batch; b++) {
                        Data &pastKey = *pastKeyValues[b * block_cnt + i].first, &pastValue = *pastKeyValues[b * block_cnt + i].second;
                        if (GetKVCacheInCPU()) {
                            pastKey.lockInCPU = true;
                            pastValue.lockInCPU = true;
                        } else {
                            pastKey.ToDevice(k.dataDevice);
                            pastValue.ToDevice(k.dataDevice);
                        }
                        targetSeqLength = std::max(targetSeqLength, (pastKey.dims.size() > 2) ? pastKey.dims[1] + seqLens[b] : seqLens[b]);
                }

                if (targetSeqLength >= max_positions && RoPEType::DYMAMIC_NTK == rope_type) {
                        float scale = pow((rope_factor * targetSeqLength / max_positions) - (rope_factor - 1), rotary_dim / (rotary_dim - 2));
                        float newbase = rope_base * scale;
                        std::pair<std::vector<float>, std::vector<float>> &&pair = this->UpdateRotaryPosEmb(newbase, rope_factor, targetSeqLength);
                        sinDataPtr = new Data(DataType::FLOAT32, {(int)this->sin.size(), (int)this->sin[0].size()}, pair.first);
                        cosDataPtr = new Data(DataType::FLOAT32, {(int)this->cos.size(), (int)this->cos[0].size()}, pair.second);
                }

                for (int b = 0; b < batch; b++) {
                    Data &pastKey = *pastKeyValues[b * block_cnt + i].first, &pastValue = *pastKeyValues[b * block_cnt + i].second;
                    int curLen = seqLens[b];
                    
                    int unitLen = 64;
    #ifdef USE_CUDA
                    unitLen = 128;
    #endif
                    while ((pastKey.dims.size() == 0 &&
                            (pastKey.expansionDims.size() == 0 || curLen > pastKey.expansionDims[1]))
                        || (pastKey.dims.size() > 0 && pastKey.dims[1] + curLen > pastKey.expansionDims[1])) {
                        std::vector<int> newDims;
                        if (pastKey.Count(0) == 0 || pastKey.dims.size() == 0) {
                            newDims = std::vector<int> {cacheOuter, ((curLen - 1) / unitLen + 1) * unitLen, cacheInner};
                        } else {
                            newDims = pastKey.dims;
                            newDims[1] += ((curLen - 1) / unitLen + 1) * unitLen;
                        }
                        pastKey.Expansion(newDims);
                    }
                    while ((pastValue.dims.size() == 0 &&
                            (pastValue.expansionDims.size() == 0 || curLen > pastValue.expansionDims[1]))
                        || (pastValue.dims.size() > 0 && pastValue.dims[1] + curLen > pastValue.expansionDims[1])) {
                        std::vector<int> newDims;
                        if (pastValue.Count(0) == 0 || pastValue.dims.size() == 0) {
                            newDims = std::vector<int>{cacheOuter, ((curLen - 1) / unitLen + 1) * unitLen, cacheInner};
                        } else {
                            newDims = pastValue.dims;
                            newDims[1] += ((curLen - 1) / unitLen + 1) * unitLen;
                        }
                        pastValue.Expansion(newDims);
                    }
                }

                fastllm::LlamaRotatePosition2D(q, allPositionIds, *sinDataPtr, *cosDataPtr, rotary_dim);
                fastllm::LlamaRotatePosition2D(k, allPositionIds, *sinDataPtr, *cosDataPtr, rotary_dim);

                Data attenOutput = Data(this->dataType);
                int total = 0;

                if (false) {
                    
                } else {
                    if (all1 && batch > 1) {
                        q.Reshape({-1, q.dims[2], q.dims[3]});
                        k.Reshape({-1, k.dims[2], k.dims[3]});
                        v.Reshape({-1, v.dims[2], v.dims[3]});

                        std::vector <int> qdims = {q.dims[1], 1, q.dims[2]};
                        std::vector <uint64_t> qstrides = {(uint64_t)q.dims[2], (uint64_t)q.dims[2], 1};
                        std::vector <int> kdims = {k.dims[1], 1, k.dims[2]};
                        std::vector <uint64_t> kstrides = {(uint64_t)k.dims[2], (uint64_t)k.dims[2], 1};
                        std::vector <int> vdims = {v.dims[1], 1, v.dims[2]};
                        std::vector <uint64_t> vstrides = {(uint64_t)v.dims[2], (uint64_t)v.dims[2], 1};
                        for (int b = 0; b < batch; b++) {
                            curQs[b].dims = qdims;
                            curQs[b].strides = qstrides;
                            curQs[b].FakeFrom(q, b * q.strides[0] * q.unitSize);
                            curKs[b].dims = kdims;
                            curKs[b].strides = kstrides;
                            curKs[b].FakeFrom(k, b * k.strides[0] * k.unitSize);
                            curVs[b].dims = vdims;
                            curVs[b].strides = vstrides;
                            curVs[b].FakeFrom(v, b * v.strides[0] * v.unitSize);
                        }

                        total = batch;
                    } else {
                        PermuteSelf(q, {0, 2, 1, 3});
                        PermuteSelf(k, {0, 2, 1, 3});
                        PermuteSelf(v, {0, 2, 1, 3});

                        std::vector<int> qkvSize = {-1, seqlen, head_dim};
                        q.Reshape(qkvSize);
                        k.Reshape(qkvSize);
                        v.Reshape(qkvSize);

                        for (int b = 0; b < batch; b++) {
                            Split(k, 1, total, total + seqLens[b], curKs[b]);
                            Split(v, 1, total, total + seqLens[b], curVs[b]);
                            Split(q, 1, total, total + seqLens[b], curQs[b]);
                            total += seqLens[b];
                        }
                    }

                    for (int b = 0; b < batch; b++) {
                        keys[b] = (pastKeyValues[b * block_cnt + i].first);
                        values[b] = (pastKeyValues[b * block_cnt + i].second);
                        pointersK[b] = (&curKs[b]);
                        pointersV[b] = (&curVs[b]);
                    }
                    CatDirectBatch(keys, pointersK, 1);
                    CatDirectBatch(values, pointersV, 1);
                }

                if (all1 && batch > 1) {
                    attenOutput.ToDevice(q.dataDevice);
                    attenOutput.Resize({1, batch, embed_dim});
                    attenOutput.Allocate();
                    for (int b = 0; b < batch; b++) {
                        qs[b] = (&curQs[b]);
                        keys[b] = (pastKeyValues[b * block_cnt + i].first);
                        values[b] = (pastKeyValues[b * block_cnt + i].second);
                        masks[b] = attentionMask[b];
                        curContextLayer[b].FakeFrom(attenOutput, b * embed_dim * attenOutput.unitSize);
                        contexts[b] = (&curContextLayer[b]);
                    }
                    AttentionBatch(qs, keys, values, masks, contexts, qs[0]->dims[0] / values[0]->dims[0], 1.0 / scale_attn, 1);
                } else {
                    attenOutput.ToDevice(curQs[0].dataDevice);
                    attenOutput.Resize({1, total, embed_dim});
                    attenOutput.Allocate();
                    int curLen = 0;
                    for (int b = 0; b < batch; b++) {
                        auto &q = curQs[b], &k = curKs[b], &v = curVs[b];
                        Data &pastKey = *pastKeyValues[b * block_cnt + i].first, &pastValue = *pastKeyValues[b * block_cnt + i].second;
                        curAttenOutput.FakeFrom(attenOutput, curLen * embed_dim * attenOutput.unitSize);
                        curLen += seqLens[b];

                        // 1.2 Attention
                        if (attentionMask[b] == nullptr) {
                            Attention(q, pastKey, pastValue, *GetEmptyData(), curAttenOutput, q.dims[0] / pastKey.dims[0], 1.0 / sqrt(head_dim), 1);
                        } else {
                            Attention(q, pastKey, pastValue, *attentionMask[b], curAttenOutput, q.dims[0] / pastKey.dims[0], 1.0 / sqrt(head_dim), 1);
                        }
                        PermuteSelf(curAttenOutput, {1, 0, 2});
                    }
                }

                Linear(attenOutput, weight[oWeightName], weight[oBiasName], attenLastOutput);
                AddTo(hiddenStates, attenLastOutput);
            }

            // 2. mlp
            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], rms_norm_eps, attenInput);

            std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
            if (weight.weight.find(swigluWeightName) != weight.weight.end() && CanRunMLP()) {
                std::string downWeightName = "model.layers." + std::to_string(i) + ".mlp.down_proj.weight";
                MLP(attenInput, weight[swigluWeightName], *GetEmptyData(), weight[downWeightName], *GetEmptyData(), w1, w2, w3, k);
                AddTo(hiddenStates, k);
            } else {
                if (weight.weight.find(swigluWeightName) != weight.weight.end()) {
                    if (CanRunLinearEx(LinearExType::ExSwiglu)) {
                        LinearEx(attenInput, weight[swigluWeightName], *GetEmptyData(), w1, LinearExType::ExSwiglu);
                    } else {
                        Linear(attenInput, weight[swigluWeightName], *GetEmptyData(), w3);
                        Swiglu(w3, w1);
                    }
                } else {
                    if (CanRunLinearEx(LinearExType::ExSilu)) {
                        LinearEx(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], *GetEmptyData(), w1, LinearExType::ExSilu);
                    } else {
                        Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"], *GetEmptyData(), w1);
                        Silu(w1, w1);
                    }
                    Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.up_proj.weight"], *GetEmptyData(), w3);
                    MulTo(w1, w3);
                }

                Linear(w1, weight["model.layers." + std::to_string(i) + ".mlp.down_proj.weight"], *GetEmptyData(), w2);
                AddTo(hiddenStates, w2);
            }
        }

        for (int b = 0; b < batch; b++) {
            if (generationConfigs[b].top_k <= 1) {
                ((GenerationConfig*)&generationConfigs[b])->top_k = 5;
                ((GenerationConfig*)&generationConfigs[b])->top_p = 0.95;
                if (fabs(generationConfigs[b].temperature - 1.0f) < 1e-9) {
                    ((GenerationConfig*)&generationConfigs[b])->temperature = 0.6;
                }
            }
        }

        std::vector <int> lastRet;
        LLMSamplingBlock(
            this, &hiddenStates,
            &weight["model.norm.weight"], &weight["lm_head.weight"],
            rms_norm_eps, batch, all1, seqLens,
            pastKeyValues, generationConfigs, lastTokens,
            retLogits, lastRet
        );
        if (sinDataPtr != &sinData)
            delete sinDataPtr;
        if (cosDataPtr != &cosData)
            delete cosDataPtr;
        return lastRet;
    }

    bool Qwen3Model::NeedAttentionMask(int qlen, int klen) {
        if (((qlen == 1) || (qlen >= 1024))) {
            return false;
        }
        return true;
    }

    void Qwen3Model::FillLLMInputsBatch(std::vector<std::vector<float>> &inputTokens,
                                          const std::vector<std::map<std::string, int>> &params,
                                          fastllm::Data &inputIds, fastllm::Data &attentionMask,
                                          fastllm::Data &positionIds) {
        inputIds.ToDevice(DataDevice::CPU);
        attentionMask.ToDevice(DataDevice::CPU);
        positionIds.ToDevice(DataDevice::CPU);

        int batch = inputTokens.size();
        int index = params[0].find("index")->second;
        if (index == 0) {
            std::vector <int> seqLens;
            seqLens.resize(batch);
            int maxLen = 0;
            for (int i = 0; i < batch; i++) {
                maxLen = std::max(maxLen, (int)inputTokens[i].size());
                seqLens[i] = (int)inputTokens[i].size();
            }

            std::vector <float> ids = std::vector <float> (batch * maxLen, 0);
            std::vector <float> vpids = std::vector <float> (batch * maxLen, 0);
            std::vector <float> vmask = std::vector <float> (batch * maxLen * maxLen, 0);
            for (int i = 0; i < batch; i++) {
                auto &tokens = inputTokens[i];
                int len = tokens.size(), base = maxLen - len;
                for (int j = 0; j < len; j++) {
                    ids[i * maxLen + base + j] = tokens[j];
                }
                for (int j = 0; j < len; j++) {
                    vpids[i * maxLen + base + j] = j;
                }

                std::fill(vmask.data() + i * maxLen * maxLen,
                        vmask.data() + i * maxLen * maxLen + (maxLen - len) * maxLen, 1.0);
                for (int j = maxLen - len; j < maxLen; j++) {
                    std::fill(vmask.data() + i * maxLen * maxLen + j * maxLen,
                            vmask.data() + i * maxLen * maxLen + j * maxLen + maxLen - len, 1.0);
                }
                for (int j = 0; j < len; j++) {
                    for (int k = j + 1; k < len; k++) {
                        vmask[i * maxLen * maxLen + (base + j) * maxLen + base + k] = 1;
                    }
                }
            }

            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen}, ids));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen, maxLen}, vmask));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen}, vpids));
        } else {
            std::vector <float> pids = std::vector <float> (batch);
            std::vector <float> fret;
            for (int i = 0; i < batch; i++) {
                fret.push_back(inputTokens[i][0]);
            }
            int maxLen = 0;
            for (int i = 0; i < batch; i++) {
                int promptLen = params[i].find("promptLen")->second;
                maxLen = std::max(promptLen, maxLen);
                pids[i] = promptLen + index - 1;
            }
            maxLen += index;
            std::vector <float> vmasks = std::vector <float> (batch * maxLen, 0.0f);
            for (int i = 0; i < batch; i++) {
                int curLen = params[i].find("promptLen")->second + index;
                for (int j = 0; j < maxLen - curLen; j++) {
                    vmasks[i * maxLen + j] = 1.0f;
                }
            }

            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, fret));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, 1, maxLen}, vmasks));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, pids));
        }
    }

    std::string Qwen3Model::MakeInput(const std::string &history, int round, const std::string &input) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
    }

    std::string Qwen3Model::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role + output + history_sep;
    }

    void Qwen3Model::Prepare() {
        if (this->weight.weight.find("lm_head.weight") == this->weight.weight.end()) {
            this->weight["lm_head.weight"] = Data();
            this->weight["lm_head.weight"].CopyFrom(this->weight["model.embed_tokens.weight"]);
            ToDataType(this->weight["lm_head.weight"], this->dataType);
        }
    }

    void Qwen3Model::WarmUp() {
        printf("Warmup...\n");
        Prepare();

        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {1});
        Data attentionMask = Data(this->dataType, {1, 1}, {0});
        Data positionIds = Data(this->dataType, {1, 1}, {0, 0});

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(this->dataType),
                                                   Data(this->dataType)));
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        elementsInKVCachePerToken = (long long)block_cnt * 
            (pastKeyValues[0].first.dims[0] * pastKeyValues[0].first.dims[2] + 
             pastKeyValues[0].second.dims[0] * pastKeyValues[0].second.dims[2]);
        printf("finish.\n");
    }
}
