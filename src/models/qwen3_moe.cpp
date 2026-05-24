//
// Created by huangyuyang on 4/29/25.
//

#include "utils.h"

#include "qwen3_moe.h"
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
#include <new>
#include <condition_variable>
#include <set>
#include <thread>

#ifdef USE_CUDA
#include "models/qwen3_cuda_common.h"
#endif

namespace fastllm {
    extern std::vector <float> GetInterLeavePowerOf2(int n);
    extern std::vector <float> GetInterleave(int n);

#ifdef USE_CUDA
    namespace {
        using namespace qwen3cuda;

        static std::atomic<int> qwen3MoeThreadTpNextPagedCacheBase(3000000);

        static std::string Qwen3MoeTrimString(const std::string &s) {
            int l = 0, r = (int)s.size();
            while (l < r && std::isspace((unsigned char)s[l])) {
                l++;
            }
            while (r > l && std::isspace((unsigned char)s[r - 1])) {
                r--;
            }
            return s.substr(l, r - l);
        }

        static bool Qwen3MoeIsDisabledTpSpec(const std::string &value) {
            std::string v = value;
            std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return (char)std::tolower(c); });
            return v.empty() || v == "false" || v == "off" || v == "none" || v == "disable";
        }

        static bool AppendQwen3MoeCudaDevicesFromSpec(const std::string &spec,
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

        static bool ParseQwen3MoeGPUForwardSpec(const std::string &rawSpec,
                                                std::vector<int> &devices,
                                                std::map<int, int> &ratios) {
            std::string spec = Qwen3MoeTrimString(rawSpec);
            if (Qwen3MoeIsDisabledTpSpec(spec)) {
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
            return AppendQwen3MoeCudaDevicesFromSpec(parseSpec, type, 1, devices, ratios);
        }

        static bool GetQwen3MoeGPUForwardDevices(const std::map<std::string, int> &deviceMap,
                                                 std::vector<int> &devices,
                                                 std::map<int, int> &ratios) {
            devices.clear();
            ratios.clear();
            const char *env = std::getenv("FASTLLM_TP");
            if (env == nullptr || Qwen3MoeIsDisabledTpSpec(Qwen3MoeTrimString(env))) {
                env = std::getenv("FASTLLM_QWEN3_MOE_TP");
            }
            if (env == nullptr || Qwen3MoeIsDisabledTpSpec(Qwen3MoeTrimString(env))) {
                env = std::getenv("FASTLLM_QWEN3_THREAD_TP");
            }
            if (env != nullptr) {
                ParseQwen3MoeGPUForwardSpec(env, devices, ratios);
            }

            if (devices.empty()) {
                for (auto &it : deviceMap) {
                    std::string lower = it.first;
                    std::transform(lower.begin(), lower.end(), lower.begin(),
                                   [](unsigned char c) { return (char)std::tolower(c); });
                    if (StartWith(lower, "multicuda")) {
                        AppendQwen3MoeCudaDevicesFromSpec(it.first, "multicuda", it.second, devices, ratios);
                    }
                }
            }

            if (devices.empty()) {
                for (auto &it : deviceMap) {
                    std::string lower = it.first;
                    std::transform(lower.begin(), lower.end(), lower.begin(),
                                   [](unsigned char c) { return (char)std::tolower(c); });
                    if (StartWith(lower, "cuda")) {
                        AppendQwen3MoeCudaDevicesFromSpec(it.first, "cuda", it.second, devices, ratios);
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

        static bool GetQwen3MoeThreadTpDevices(const std::map<std::string, int> &deviceMap,
                                               std::vector<int> &devices,
                                               std::map<int, int> &ratios) {
            if (!GetQwen3MoeGPUForwardDevices(deviceMap, devices, ratios)) {
                return false;
            }
            return devices.size() > 1;
        }

        static DivisionScheme ExtractQwen3MoeFirstRangeScheme(const DivisionScheme &scheme) {
            DivisionScheme ret;
            for (auto &it : scheme) {
                ret[it.first];
                if (!it.second.empty()) {
                    ret[it.first].push_back(it.second[0]);
                }
            }
            return ret;
        }

        static DivisionScheme ExtractQwen3MoeKVHeadScheme(const DivisionScheme &qkvScheme,
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

        static DataType ResolveQwen3MoeThreadTpComputeType(DataType modelType) {
            if (modelType == DataType::FLOAT16 || modelType == DataType::BFLOAT16) {
                return modelType;
            }
            return DataType::FLOAT16;
        }

        static DataType ResolveQwen3MoeThreadTpCacheType(DataType cacheType, DataType computeType) {
            if (cacheType == DataType::FLOAT16 ||
                cacheType == DataType::BFLOAT16 ||
                cacheType == DataType::FP8_E4M3) {
                return cacheType;
            }
            return computeType;
        }

        static Data *EnsureQwen3MoeThreadTpLocalCache(Data &root, int device, DataType localDataType) {
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

        static void PrepareQwen3MoeSingleCudaCache(Data &cache, int device, DataType localDataType) {
            cache.isKVCache = true;
            cache.lockInCPU = false;
            if (cache.dataType != localDataType && cache.dims.empty()) {
                cache.dataType = localDataType;
                cache.UpdateUnitSize();
            }
            cache.ToDevice(DataDevice::CUDA, {device}, false);
        }

        static void SyncQwen3MoeThreadTpRootCacheMeta(Data &root,
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

            AssertInFastLLM(firstLocal->pageIndex.size() < 1000000,
                            "Qwen3-MOE ForwardGPU got invalid local paged cache pageIndex metadata.\n");

            std::vector<int> globalDims = {globalKVHeads, firstLocal->dims[1], headDim};
            root.dataType = firstLocal->dataType;
            root.UpdateUnitSize();
            root.dataDevice = DataDevice::CUDA;
            root.multiDeviceData = false;
            root.ClearTensorParallelLayout();
            root.Resize(globalDims);
            root.multiDeviceData = true;
            root.dataDeviceIds = devices;
            root.tpLayout = TP_LAYOUT_SHARDED;
            root.tpAxis = 0;
            root.tpRanges = kvHeadScheme;
            root.tpGlobalDims = globalDims;
            root.cudaData = nullptr;
            root.isKVCache = true;
            root.isPagedKVCache = firstLocal->isPagedKVCache;
            root.pageLen = firstLocal->pageLen;
            root.pageIndex = firstLocal->pageIndex;
            root.lastPageLen = firstLocal->lastPageLen;
            root.pagedKVCacheData = firstLocal->pagedKVCacheData;
        }

        static bool Qwen3MoeCudaGraphEnabled() {
            return GetFastllmEnv().cudaGraph;
        }

        static bool Qwen3MoeNeedRepeatPenalty(const GenerationConfig &config) {
            float diff = config.repeat_penalty - 1.0f;
            return diff > 1e-6f || diff < -1e-6f;
        }

        static void Qwen3MoeCudaClearMultiDeviceState(Data &data) {
            for (auto &it : data.multiDeviceDatas) {
                delete it.second;
            }
            data.multiDeviceDatas.clear();
            data.multiDeviceData = false;
            data.ClearTensorParallelLayout();
        }

        static void Qwen3MoeCudaPrepareLocalOutput(Data &data, int device) {
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
            Qwen3MoeCudaClearMultiDeviceState(data);
            data.dataDevice = DataDevice::CUDA;
            data.dataDeviceIds = {device};
            data.lockInCPU = false;
        }

        static void Qwen3MoeZeroCudaLike(Data &dst, const Data &like, int device) {
            bool needReset = dst.isFake || dst.dataType != like.dataType ||
                             dst.dataDevice != DataDevice::CUDA || dst.dims != like.dims ||
                             (!dst.dataDeviceIds.empty() && dst.dataDeviceIds[0] != device);
            if (!needReset && dst.cudaData != nullptr) {
                int ptrDevice = GetPointerDeviceId(dst.cudaData);
                needReset = ptrDevice >= 0 && ptrDevice != device;
            }
            if (needReset) {
                if (!dst.isFake) {
                    dst.FreeSpace();
                } else {
                    dst.isFake = false;
                    dst.cpuData = nullptr;
                    dst.cudaData = nullptr;
                    dst.deviceData = nullptr;
                    dst.expansionSize = 0;
                    dst.expansionBytes = 0;
                }
                Qwen3MoeCudaClearMultiDeviceState(dst);
                dst.dataType = like.dataType;
                dst.UpdateUnitSize();
                dst.dataDevice = DataDevice::CUDA;
                dst.dataDeviceIds = {device};
                dst.Resize(like.dims);
            }
            dst.Allocate();
            if (dst.cudaData != nullptr) {
                FastllmCudaMemset0(dst.cudaData, dst.GetBytes());
            }
        }

        static bool Qwen3MoeHasLocalMoeShard(const std::vector<Data*> &localWeights) {
            for (int i = 2; i + 1 < (int)localWeights.size(); i += 2) {
                Data *gateup = localWeights[i];
                Data *down = localWeights[i + 1];
                if (gateup != nullptr && down != nullptr &&
                    gateup->dims.size() == 2 && down->dims.size() == 2 &&
                    gateup->dims[0] > 0 && down->dims[1] > 0) {
                    return true;
                }
            }
            return false;
        }

        static bool Qwen3MoeIsNVFP4WeightType(DataType dataType) {
            return dataType == DataType::NVFP4 ||
                   dataType == DataType::NVFP4_BLOCK_16 ||
                   dataType == DataType::NVFP4_BLOCK_16_E8M0;
        }

        struct Qwen3MoeForwardSingleBuffers {
            Data embedOutput;
            Data hiddenStates;
            Data attenInput;
            Data qkv;
            Data q;
            Data qForAttentionHolder;
            Data attenOutput;
            Data attenLastOutput;
            Data routerLogits;
            Data routerLogitsTemp;
            Data expertIndex;
            Data expertScore;
            Data w1;
            Data w2;
            Data w3;
            Data tempInput;
            Data tempOutput;
            Data moeInputTemp;
            Data moeOutputTemp;
            Data moeFinal;
            Data qSizes;
            Data pageSizes;
            Data pageIndexs;
            Data lastPageLens;
            Data insertIndexs;
            Data insertPositions;
            std::vector<Data*> batchPastKeys;
            std::vector<Data*> batchPastValues;

            Qwen3MoeForwardSingleBuffers() : batchPastKeys(1), batchPastValues(1) {}
        };

        static void Qwen3MoeDetachFakeReusableTensor(Data &data) {
            if (!data.isFake) {
                return;
            }
            data.isFake = false;
            data.cpuData = nullptr;
            data.cudaData = nullptr;
            data.deviceData = nullptr;
            data.expansionSize = 0;
            data.expansionBytes = 0;
        }

        static void Qwen3MoeFreeReusableTensor(Data &data) {
            if (data.isFake) {
                Qwen3MoeDetachFakeReusableTensor(data);
            } else {
                data.FreeSpace();
            }
            data.dims.clear();
            data.strides.clear();
            data.expansionDims.clear();
            data.cpuIntDatas.clear();
            data.dataDevice = DataDevice::CPU;
            data.dataDeviceIds.clear();
            data.weightType = WeightType::NONE;
            data.lockInCPU = false;
            Qwen3MoeCudaClearMultiDeviceState(data);
        }

        static void Qwen3MoeFreeForwardSingleBuffers(Qwen3MoeForwardSingleBuffers &buf) {
            Qwen3MoeFreeReusableTensor(buf.embedOutput);
            Qwen3MoeFreeReusableTensor(buf.hiddenStates);
            Qwen3MoeFreeReusableTensor(buf.attenInput);
            Qwen3MoeFreeReusableTensor(buf.qkv);
            Qwen3MoeFreeReusableTensor(buf.q);
            Qwen3MoeFreeReusableTensor(buf.qForAttentionHolder);
            Qwen3MoeFreeReusableTensor(buf.attenOutput);
            Qwen3MoeFreeReusableTensor(buf.attenLastOutput);
            Qwen3MoeFreeReusableTensor(buf.routerLogits);
            Qwen3MoeFreeReusableTensor(buf.routerLogitsTemp);
            Qwen3MoeFreeReusableTensor(buf.expertIndex);
            Qwen3MoeFreeReusableTensor(buf.expertScore);
            Qwen3MoeFreeReusableTensor(buf.w1);
            Qwen3MoeFreeReusableTensor(buf.w2);
            Qwen3MoeFreeReusableTensor(buf.w3);
            Qwen3MoeFreeReusableTensor(buf.tempInput);
            Qwen3MoeFreeReusableTensor(buf.tempOutput);
            Qwen3MoeFreeReusableTensor(buf.moeInputTemp);
            Qwen3MoeFreeReusableTensor(buf.moeOutputTemp);
            Qwen3MoeFreeReusableTensor(buf.moeFinal);
            Qwen3MoeFreeReusableTensor(buf.qSizes);
            Qwen3MoeFreeReusableTensor(buf.pageSizes);
            Qwen3MoeFreeReusableTensor(buf.pageIndexs);
            Qwen3MoeFreeReusableTensor(buf.lastPageLens);
            Qwen3MoeFreeReusableTensor(buf.insertIndexs);
            Qwen3MoeFreeReusableTensor(buf.insertPositions);
        }

        static void Qwen3MoeReinitializeForwardSingleBuffers(Qwen3MoeForwardSingleBuffers &buf) {
            Qwen3MoeFreeForwardSingleBuffers(buf);
            buf.~Qwen3MoeForwardSingleBuffers();
            new (&buf) Qwen3MoeForwardSingleBuffers();
        }

        struct Qwen3MoeCudaGraphDecodeState {
            std::mutex mutex;
            std::string signature;
            bool warmed = false;
            bool captured = false;
            bool disabled = false;
            void *graph = nullptr;
            void *exec = nullptr;
            Data inputIds;
            Data positionIds;
            Qwen3MoeForwardSingleBuffers buffers;
            Qwen3MoeForwardSingleBuffers metaBuffers;
            Data logitsHalf;
            Data logits;

            ~Qwen3MoeCudaGraphDecodeState() {
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

        struct Qwen3MoeCudaGraphSyncState {
            std::mutex mutex;
            std::condition_variable cv;
            int arrived = 0;
            int generation = 0;
            bool phaseOk = true;
            bool lastPhaseOk = true;
            bool disabled = false;
        };

        static Qwen3MoeCudaGraphSyncState &GetQwen3MoeCudaGraphSyncState(const Qwen3MOEModel *model) {
            static std::mutex syncsMutex;
            static std::map<const Qwen3MOEModel*, std::unique_ptr<Qwen3MoeCudaGraphSyncState> > syncs;
            std::lock_guard<std::mutex> guard(syncsMutex);
            auto &sync = syncs[model];
            if (sync == nullptr) {
                sync.reset(new Qwen3MoeCudaGraphSyncState());
            }
            return *sync;
        }

        static bool Qwen3MoeCudaGraphSyncPhase(const Qwen3MOEModel *model, int participants, bool ok = true) {
            if (participants <= 1) {
                return ok;
            }
            Qwen3MoeCudaGraphSyncState &sync = GetQwen3MoeCudaGraphSyncState(model);
            std::unique_lock<std::mutex> lock(sync.mutex);
            int generation = sync.generation;
            if (sync.arrived == 0) {
                sync.phaseOk = true;
            }
            sync.phaseOk = sync.phaseOk && ok && !sync.disabled;
            sync.arrived++;
            if (sync.arrived == participants) {
                sync.lastPhaseOk = sync.phaseOk;
                sync.arrived = 0;
                sync.generation++;
                sync.cv.notify_all();
                return sync.lastPhaseOk;
            } else {
                sync.cv.wait(lock, [&]() {
                    return sync.generation != generation;
                });
                return sync.lastPhaseOk;
            }
        }

        static bool Qwen3MoeCudaGraphIsDisabled(const Qwen3MOEModel *model) {
            Qwen3MoeCudaGraphSyncState &sync = GetQwen3MoeCudaGraphSyncState(model);
            std::lock_guard<std::mutex> guard(sync.mutex);
            return sync.disabled;
        }

        static void Qwen3MoeDisableCudaGraph(const Qwen3MOEModel *model) {
            Qwen3MoeCudaGraphSyncState &sync = GetQwen3MoeCudaGraphSyncState(model);
            {
                std::lock_guard<std::mutex> guard(sync.mutex);
                sync.disabled = true;
            }
            sync.cv.notify_all();
        }

        static void Qwen3MoeDestroyCudaGraph(Qwen3MoeCudaGraphDecodeState &state) {
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

        static void Qwen3MoeAbortCudaGraphCapture() {
            void *capturedGraph = nullptr;
            if (FastllmCudaGraphEndCapture(&capturedGraph) && capturedGraph != nullptr) {
                FastllmCudaGraphDestroy(capturedGraph);
            }
        }

        static void Qwen3MoeDisableCudaGraphState(
                const Qwen3MOEModel *model,
                Qwen3MoeCudaGraphDecodeState &state) {
            Qwen3MoeDestroyCudaGraph(state);
            state.disabled = true;
            Qwen3MoeDisableCudaGraph(model);
        }

        static void Qwen3MoeWarnCudaGraphStage(
                const char *stage,
                int gpuId,
                bool localOk) {
            if (!localOk) {
                printf("Warning: Qwen3-MOE CUDA graph %s failed on gpu %d: %s. Disable graph for this model.\n",
                       stage, gpuId, FastllmCudaGraphLastError());
                fflush(stdout);
            }
        }

        static bool Qwen3MoeSyncCudaGraphStage(
                const Qwen3MOEModel *model,
                Qwen3MoeCudaGraphDecodeState &state,
                int participants,
                const char *stage,
                int gpuId,
                bool localOk) {
            bool allOk = Qwen3MoeCudaGraphSyncPhase(model, participants, localOk);
            if (!allOk) {
                Qwen3MoeWarnCudaGraphStage(stage, gpuId, localOk);
                Qwen3MoeDisableCudaGraphState(model, state);
            }
            return allOk;
        }

        static Qwen3MoeCudaGraphDecodeState &GetQwen3MoeCudaGraphDecodeState(
                const Qwen3MOEModel *model, int gpuId) {
            static std::mutex statesMutex;
            static std::map<std::pair<const Qwen3MOEModel*, int>,
                            std::unique_ptr<Qwen3MoeCudaGraphDecodeState> > states;
            std::lock_guard<std::mutex> guard(statesMutex);
            auto key = std::make_pair(model, gpuId);
            auto &state = states[key];
            if (state == nullptr) {
                state.reset(new Qwen3MoeCudaGraphDecodeState());
            }
            return *state;
        }

        static void Qwen3MoePrepareGraphCudaTensor(Data &dst, const Data &src, int device) {
            AssertInFastLLM(src.dataDevice == DataDevice::CUDA && src.cudaData != nullptr,
                            "Qwen3-MOE CUDA graph requires CUDA source tensor.\n");
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
                Qwen3MoeCudaClearMultiDeviceState(dst);
                dst.dataType = src.dataType;
                dst.UpdateUnitSize();
                dst.dataDevice = DataDevice::CUDA;
                dst.dataDeviceIds = {device};
                dst.Resize(src.dims);
            }
            dst.Allocate(false);
            FastllmCudaCopyFromDeviceToDevice(dst.cudaData, src.cudaData, src.GetBytes());
        }

        static void Qwen3MoePrepareGraphIntTensor(Data &dst, int device, const std::vector<int> &host) {
            AssertInFastLLM(!host.empty(), "Qwen3-MOE CUDA graph got empty int metadata.\n");
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
                Qwen3MoeCudaClearMultiDeviceState(dst);
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

        static bool Qwen3MoeCanUseCudaFullLogitsSampling(
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
                if (Qwen3MoeNeedRepeatPenalty(config)) {
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

        static Data &Qwen3MoeThreadLocalCudaSamplingFullLogits() {
            static thread_local Data fullLogits(DataType::FLOAT32);
            return fullLogits;
        }

        static Data &Qwen3MoeThreadLocalCudaSamplingTopK() {
            static thread_local Data topk(DataType::FLOAT32);
            return topk;
        }

        static Data &Qwen3MoeThreadLocalCpuSamplingTopK() {
            static thread_local Data topk(DataType::FLOAT32);
            return topk;
        }

        static void Qwen3MoeGatherShardLogitsToRootCuda(
                int rootDevice,
                const std::vector<int> &devices,
                const DivisionScheme &lmHeadScheme,
                std::vector<Data> &localLogits,
                int batch,
                int vocabSize,
                Data &fullLogits) {
            FastllmCudaSetDevice(rootDevice);
            Qwen3MoeCudaPrepareLocalOutput(fullLogits, rootDevice);
            fullLogits.dataType = DataType::FLOAT32;
            fullLogits.UpdateUnitSize();
            fullLogits.Resize({batch, vocabSize});
            fullLogits.Allocate();

            for (int r = 0; r < (int)devices.size(); r++) {
                int device = devices[r];
                auto schemeIt = lmHeadScheme.find(device);
                AssertInFastLLM(schemeIt != lmHeadScheme.end(),
                                "Qwen3-MOE CUDA sampling: missing lm_head split range.\n");
                AssertInFastLLM(localLogits[r].dataDevice == DataDevice::CUDA &&
                                localLogits[r].cudaData != nullptr,
                                "Qwen3-MOE CUDA sampling: local logits must stay on CUDA.\n");
                int localVocab = localLogits[r].dims.back();
                int rows = localLogits[r].Count(0) / localVocab;
                AssertInFastLLM(rows == batch,
                                "Qwen3-MOE CUDA sampling: local logits batch mismatch.\n");

                uint8_t *dstBase = (uint8_t*)fullLogits.cudaData;
                uint8_t *srcBase = (uint8_t*)localLogits[r].cudaData;
                int localOffset = 0;
                for (auto &range : schemeIt->second) {
                    int len = range.second - range.first;
                    AssertInFastLLM(range.first >= 0 && range.second <= vocabSize &&
                                    localOffset + len <= localVocab,
                                    "Qwen3-MOE CUDA sampling: invalid lm_head split range.\n");
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

        static std::vector<int> Qwen3MoeSampleFromRootCudaLogits(
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
            Data &topk = Qwen3MoeThreadLocalCudaSamplingTopK();
            Qwen3CudaTopK(cudaRunner, fullLogits, topk, maxTopK);

            Data &cpuTopK = Qwen3MoeThreadLocalCpuSamplingTopK();
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

        static bool Qwen3MoeCanGraphIndexedMoe(
                const std::unordered_map<int, std::vector<std::vector<Data*> > > &deviceWeights,
                int device,
                int blockCnt,
                int hidden,
                DataType moeAtype) {
            if (moeAtype != DataType::FLOAT16 && moeAtype != DataType::BFLOAT16) {
                return false;
            }
            auto deviceIt = deviceWeights.find(device);
            if (deviceIt == deviceWeights.end() || (int)deviceIt->second.size() < blockCnt) {
                return false;
            }
            for (int i = 0; i < blockCnt; i++) {
                const std::vector<Data*> &layerWeights = deviceIt->second[i];
                if ((int)layerWeights.size() < 4 || layerWeights[0] != nullptr) {
                    return false;
                }

                bool hasShard = false;
                for (int j = 2; j + 1 < (int)layerWeights.size(); j += 2) {
                    Data *gateup = layerWeights[j];
                    Data *down = layerWeights[j + 1];
                    if (gateup == nullptr && down == nullptr) {
                        continue;
                    }
                    if (gateup == nullptr || down == nullptr ||
                        gateup->dataDevice != DataDevice::CUDA ||
                        down->dataDevice != DataDevice::CUDA ||
                        gateup->cudaData == nullptr || down->cudaData == nullptr ||
                        gateup->dataType != down->dataType ||
                        gateup->dims.size() != 2 || down->dims.size() != 2 ||
                        gateup->dims[1] != hidden ||
                        down->dims[0] != hidden ||
                        gateup->dims[0] != down->dims[1] * 2) {
                        return false;
                    }
                    bool supportedWeight =
                        gateup->dataType == DataType::FP8_E4M3 ||
                        gateup->dataType == DataType::FP8_E4M3_BLOCK_128 ||
                        Qwen3MoeIsNVFP4WeightType(gateup->dataType);
                    if (!supportedWeight) {
                        return false;
                    }
                    hasShard = true;
                }
                if (!hasShard) {
                    return false;
                }
            }
            return true;
        }
    }
#endif

    Qwen3MOEModel::Qwen3MOEModel() {
        this->model_type = "qwen3_moe";
        this->model_struct = "qwen3_moe";
        this->use_new_engine = true;

        // 默认使用alpaca的提示词和instruction
        this->pre_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n";
        this->user_role = "### Instruction:\n";
        this->bot_role = "\n\n### Response:";
        this->history_sep = "</s>";

        block_cnt = 32;
        rotary_dim = 128;

        weight.embeddingNames.insert("model.embed_tokens.weight");
        weight.linearNames = {
            "lm_head.weight", "model.layers.*.down_proj.weight", "model.layers.*.up_proj.weight",
            "model.layers.*.gate_proj.weight",  "model.layers.*.gate_proj.weight", "model.layers.*.gateup_proj.weight",
            "model.layers.*.self_attn.o_proj.weight", "model.layers.*.self_attn.q_proj.weight", "model.layers.*.self_attn.k_proj.weight",
            "model.layers.*.self_attn.v_proj.weight", "model.layers.*.self_attn.mergeqkv.weight", "model.layers.*.self_attn.W_pack.weight",
            "model.layers.*.mlp.*.weight"
        };
    }

    bool Qwen3MOEModel::IsThreadTensorParallelEnabled() const {
#ifdef USE_CUDA
        std::vector<int> devices;
        std::map<int, int> ratios;
        return GetQwen3MoeThreadTpDevices(this->deviceMap, devices, ratios);
#else
        return false;
#endif
    }

    Data &Qwen3MOEModel::GetThreadTensorParallelBias(const std::string &name) {
        auto it = this->weight.weight.find(name);
        if (it != this->weight.weight.end()) {
            return it->second;
        }
        return this->threadTpEmptyBiases[name];
    }

    void Qwen3MOEModel::InitParams() {
        basellm::InitParams();
        num_experts = atoi(this->weight.dicts["num_experts"].c_str());
        num_experts_per_tok = atoi(this->weight.dicts["num_experts_per_tok"].c_str());
        norm_topk_prob = (this->weight.dicts["norm_topk_prob"] == "true");

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

        for (int i = 0; i < block_cnt; i++) {
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

        for (int i = 0; i < block_cnt; i++) {
            for (int j = 0; j < this->num_experts; j++) {
                std::string w1WeightName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".gate_proj.weight";
                std::string w3WeightName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".up_proj.weight";
                std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".gateup_proj.weight";
                std::string downWeightName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".down_proj.weight";
                this->weightMergeRules.push_back(
                    WeightMergeRule({WeightMergeRuleSingle({w1WeightName, w3WeightName}, swigluWeightName, std::string("linearSwiglu"))})
                );

                this->AddSpecialWeight(swigluWeightName, "linearSwiglu", i);
                this->AddSpecialWeight(downWeightName, "linearColumn", i);

                this->moeLinears.insert(w1WeightName);
                this->moeLinears.insert(w3WeightName);
                this->moeLinears.insert(downWeightName);
            }
        }
    }

    int Qwen3MOEModel::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <float> *retLogits) {
        Data attentionMaskCopy(attentionMask), positionIdsCopy(positionIds);
        std::vector <Data*> attentionMasks = {attentionMaskCopy.dims.empty() ? nullptr : &attentionMaskCopy};
        std::vector <Data*> positionIdsVec = {&positionIdsCopy};
        std::vector <int> seqLens = {(int)inputIds.dims[1]};
        std::vector <GenerationConfig> generationConfigs = {generationConfig};
        std::vector <std::pair <Data*, Data*> > pagedPastKeyValues;
        for (int i = 0; i < pastKeyValues.size(); i++) {
            pagedPastKeyValues.push_back(std::make_pair(&pastKeyValues[i].first, &pastKeyValues[i].second));
        }
        std::vector <std::vector <float>*> batchLogits;
        batchLogits.push_back(retLogits);
#ifdef USE_CUDA
        std::vector<int> devices;
        std::map<int, int> ratios;
        if (GetQwen3MoeGPUForwardDevices(this->deviceMap, devices, ratios)) {
            return ForwardGPU(1, inputIds, attentionMasks, positionIdsVec, seqLens,
                              pagedPastKeyValues, generationConfigs, lastTokens,
                              &batchLogits)[0];
        }
#endif
        return ForwardV2(1, inputIds, attentionMasks, positionIdsVec, seqLens,
                         pagedPastKeyValues, generationConfigs, lastTokens, &batchLogits)[0];
    }

    bool Qwen3MOEModel::ForwardSingleGPUDecodeGraph(
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
        auto rejectGraph = [](const std::string &) -> bool {
            return false;
        };
        if (!Qwen3MoeCudaGraphEnabled()) {
            return rejectGraph("disabled");
        }
        if (batch != 1) {
            return rejectGraph("batch != 1");
        }
        if (!all1 || isPrefill || seqLens.size() != 1 || seqLens[0] != 1) {
            return rejectGraph("not single-token decode");
        }
        if ((int)pastKeyValues.size() < block_cnt) {
            return rejectGraph("pastKeyValues too small");
        }

        int graphParticipants = tensorParallel ? std::max(2, (int)ratios.size()) : 1;
        auto syncGraphPeers = [&](bool ok = true) {
            return Qwen3MoeCudaGraphSyncPhase(this, graphParticipants, ok);
        };

        const DataType computeType = ResolveQwen3MoeThreadTpComputeType(this->dataType);
        const DataType threadTpMoeAtype = (this->moeAtype == DataType::FLOAT32) ? computeType : this->moeAtype;
        auto &moeWeightsByDevice = tensorParallel ? threadTpMoeWeights : singleGpuMoeWeights;
        auto &moeBiassByDevice = tensorParallel ? threadTpMoeBiass : singleGpuMoeBiass;
        int graphHidden = embed_dim;
        auto embedIt = weight.weight.find("model.embed_tokens.weight");
        if (embedIt != weight.weight.end() && embedIt->second.dims.size() >= 2) {
            graphHidden = embedIt->second.dims.back();
        } else {
            auto normIt = weight.weight.find("model.norm.weight");
            if (normIt != weight.weight.end() && !normIt->second.dims.empty()) {
                graphHidden = normIt->second.dims[0];
            }
        }
        bool indexedMoeOk = Qwen3MoeCanGraphIndexedMoe(moeWeightsByDevice, gpuId, block_cnt,
                                                       graphHidden, threadTpMoeAtype);
        if (!syncGraphPeers(indexedMoeOk)) {
            return rejectGraph("unsupported indexed moe layout");
        }

        auto requireLocal = [&](Data &data, const std::string &name) -> Data* {
            auto it = data.multiDeviceDatas.find(gpuId);
            if (it != data.multiDeviceDatas.end() && it->second != nullptr) {
                return it->second;
            }
            if (!tensorParallel) {
                if (!data.dims.empty() &&
                    (data.dataDevice != DataDevice::CUDA || data.cudaData == nullptr ||
                     data.dataDeviceIds.empty() || data.dataDeviceIds[0] != gpuId)) {
                    data.ToDevice(DataDevice::CUDA, {gpuId}, true);
                }
                return &data;
            }
            ErrorInFastLLM("Qwen3-MOE ForwardSingleGPU graph missing local tensor: " + name + ".\n");
            return nullptr;
        };

        Data *localInputIds = requireLocal((Data&)inputIds, "inputIds");
        Data *localPositionIds = requireLocal((Data&)positionIds, "positionIds");
        if (localInputIds->dims.size() != 2 || localInputIds->dims[0] != 1 || localInputIds->dims[1] != 1 ||
            localPositionIds->dims.empty() || localPositionIds->Count(0) != 1) {
            return rejectGraph("input/position dims mismatch");
        }

        int currentTokens = 0;
        for (int i = 0; i < block_cnt; i++) {
            Data *pastKey = pastKeyValues[i].first;
            Data *pastValue = pastKeyValues[i].second;
            if (pastKey == nullptr || pastValue == nullptr) {
                return rejectGraph("null kv cache");
            }
            if (pastKey->pagedKVCacheData == nullptr || pastValue->pagedKVCacheData == nullptr) {
                return rejectGraph("kv cache is not paged");
            }
            if (pastKey->pageIndex.empty() || pastValue->pageIndex.empty()) {
                return rejectGraph("empty kv page index");
            }
            if (pastKey->dataDevice != DataDevice::CUDA || pastValue->dataDevice != DataDevice::CUDA) {
                return rejectGraph("kv cache not on cuda");
            }
            if (pastKey->dataType == DataType::FP8_E4M3 || pastValue->dataType == DataType::FP8_E4M3) {
                return rejectGraph("fp8 kv cache");
            }
            if (pastKey->pageLen <= 0 || pastKey->pageLen != pastValue->pageLen ||
                pastKey->pageIndex.size() != pastValue->pageIndex.size() ||
                pastKey->lastPageLen != pastValue->lastPageLen) {
                return rejectGraph("unaligned kv cache metadata");
            }
            int layerTokens = ((int)pastKey->pageIndex.size() - 1) * pastKey->pageLen + pastKey->lastPageLen;
            currentTokens = std::max(currentTokens, layerTokens);
        }
        if (rope_type == RoPEType::DYMAMIC_NTK && currentTokens + 1 >= max_positions) {
            return rejectGraph("dynamic ntk beyond max position");
        }

        Qwen3MoeCudaGraphDecodeState &state = GetQwen3MoeCudaGraphDecodeState(this, gpuId);
        std::unique_lock<std::mutex> graphLock(state.mutex);
        if (Qwen3MoeCudaGraphIsDisabled(this)) {
            return false;
        }
        if (!syncGraphPeers(!state.disabled)) {
            if (state.disabled) {
                Qwen3MoeDisableCudaGraph(this);
            }
            return false;
        }
        FastllmCudaSetDevice(gpuId);
        Qwen3MoePrepareGraphCudaTensor(state.inputIds, *localInputIds, gpuId);
        Qwen3MoePrepareGraphCudaTensor(state.positionIds, *localPositionIds, gpuId);

        Data *firstKey = pastKeyValues[0].first;
        bool needNewPage = firstKey->pageIndex.empty() || firstKey->lastPageLen >= firstKey->pageLen;
        int insertIndex = needNewPage ? -1 : firstKey->pageIndex.back();
        int insertPosition = needNewPage ? 0 : firstKey->lastPageLen;

        for (int i = 0; i < block_cnt; i++) {
            Data *pastKey = pastKeyValues[i].first;
            Data *pastValue = pastKeyValues[i].second;
            bool layerNeedNewPage = pastKey->pageIndex.empty() || pastKey->lastPageLen >= pastKey->pageLen;
            AssertInFastLLM(layerNeedNewPage == needNewPage,
                            "Qwen3-MOE CUDA graph requires aligned paged cache layout across layers.\n");
            if (needNewPage) {
                int keyPage = pastKey->pagedKVCacheData->GetUnusedPageIndex(true);
                int valuePage = pastValue->pagedKVCacheData->GetUnusedPageIndex(true);
                if (insertIndex < 0) {
                    insertIndex = keyPage;
                }
                AssertInFastLLM(keyPage == insertIndex && valuePage == insertIndex,
                                "Qwen3-MOE CUDA graph requires aligned K/V page indices across layers.\n");
                pastKey->pageIndex.push_back(keyPage);
                pastValue->pageIndex.push_back(valuePage);
                pastKey->lastPageLen = 1;
                pastValue->lastPageLen = 1;
            } else {
                AssertInFastLLM(pastKey->pageIndex.back() == insertIndex &&
                                pastValue->pageIndex.back() == insertIndex &&
                                pastKey->lastPageLen == insertPosition &&
                                pastValue->lastPageLen == insertPosition,
                                "Qwen3-MOE CUDA graph requires aligned paged cache positions across layers.\n");
                pastKey->lastPageLen++;
                pastValue->lastPageLen++;
            }
        }

        std::vector<int> pageIndexHost = firstKey->pageIndex;
        int lastPageLen = firstKey->lastPageLen;
        Qwen3MoePrepareGraphIntTensor(state.metaBuffers.insertIndexs, gpuId, {insertIndex});
        Qwen3MoePrepareGraphIntTensor(state.metaBuffers.insertPositions, gpuId, {insertPosition});
        Qwen3MoePrepareGraphIntTensor(state.metaBuffers.qSizes, gpuId, {0, 1});
        Qwen3MoePrepareGraphIntTensor(state.metaBuffers.pageSizes, gpuId, {0, (int)pageIndexHost.size()});
        Qwen3MoePrepareGraphIntTensor(state.metaBuffers.pageIndexs, gpuId, pageIndexHost);
        Qwen3MoePrepareGraphIntTensor(state.metaBuffers.lastPageLens, gpuId, {lastPageLen});

        std::ostringstream signature;
        signature << "gpu=" << gpuId
                  << ";tp=" << (tensorParallel ? 1 : 0)
                  << ";tpRank0=" << (firstTensorParallelRank ? 1 : 0)
                  << ";pages=" << pageIndexHost.size()
                  << ";inputType=" << (int)state.inputIds.dataType
                  << ";posType=" << (int)state.positionIds.dataType
                  << ";moeAtype=" << (int)threadTpMoeAtype
                  << ";kCache=" << pastKeyValues[0].first->pagedKVCacheData->cudaData
                  << ";vCache=" << pastKeyValues[0].second->pagedKVCacheData->cudaData
                  << ";lmLocal=" << requireLocal(weight["lm_head.weight"], "lm_head.weight")->dims[0];
        std::string newSignature = signature.str();
        if (state.signature != newSignature) {
            Qwen3MoeDestroyCudaGraph(state);
            state.signature = newSignature;
        }

        auto runGraphBodyWithBuffers = [&](Qwen3MoeForwardSingleBuffers &workBuf,
                                           Qwen3MoeForwardSingleBuffers &metaBuf) {
            Qwen3CudaDirectRunner cudaRunner(gpuId);
                Qwen3MoeForwardSingleBuffers &buf = workBuf;
                if ((int)buf.batchPastKeys.size() != batch) {
                    buf.batchPastKeys.resize(batch);
                    buf.batchPastValues.resize(batch);
                }

                Qwen3CudaEmbeddingDirect(cudaRunner, state.inputIds,
                                         *requireLocal(weight["model.embed_tokens.weight"], "model.embed_tokens.weight"),
                                         buf.embedOutput);
                Qwen3CudaConvertToDataType(cudaRunner, buf.embedOutput, buf.hiddenStates, computeType);

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
                    std::string gateWeightName = "model.layers." + std::to_string(i) + ".mlp.gate.weight";
                    std::string gateBiasName = "model.layers." + std::to_string(i) + ".mlp.gate.e_score_correction_bias";

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
                                    "Qwen3-MOE ForwardSingleGPU graph got empty local attention shard.\n");

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
                        &buf.qForAttentionHolder,
                        &metaBuf.insertIndexs, &metaBuf.insertPositions,
                        &metaBuf.qSizes, &metaBuf.pageSizes, &metaBuf.pageIndexs, &metaBuf.lastPageLens,
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

                    if (tensorParallel) {
                        DataType residualType = buf.hiddenStates.dataType;
                        if (firstTensorParallelRank) {
                            if (buf.attenOutput.dataType == residualType) {
                                Qwen3CudaLinearAddBlock(cudaRunner, &buf.attenOutput,
                                                       requireLocal(weight[oWeightName], oWeightName),
                                                       requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                                                       &buf.attenLastOutput, &buf.hiddenStates);
                            } else {
                                Qwen3CudaLinear(cudaRunner, buf.attenOutput,
                                                *requireLocal(weight[oWeightName], oWeightName),
                                                *requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                                                buf.attenLastOutput);
                                if (buf.attenLastOutput.dataType != residualType) {
                                    Qwen3CudaToDataType(cudaRunner, buf.attenLastOutput, residualType);
                                }
                                Qwen3CudaAddTo(cudaRunner, buf.hiddenStates, buf.attenLastOutput);
                            }
                        } else {
                            Qwen3CudaLinear(cudaRunner, buf.attenOutput,
                                            *requireLocal(weight[oWeightName], oWeightName),
                                            *requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                                            buf.hiddenStates);
                            if (buf.hiddenStates.dataType != residualType) {
                                Qwen3CudaToDataType(cudaRunner, buf.hiddenStates, residualType);
                            }
                        }
                        FastllmNcclAllReduce(buf.hiddenStates.cudaData, buf.hiddenStates.cudaData,
                                             buf.hiddenStates.Count(0), buf.hiddenStates.dataType, gpuId);
                    } else {
                        Qwen3CudaLinear(cudaRunner, buf.attenOutput,
                                        *requireLocal(weight[oWeightName], oWeightName),
                                        *requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                                        buf.attenLastOutput);
                        if (buf.attenLastOutput.dataType != buf.hiddenStates.dataType) {
                            Qwen3CudaToDataType(cudaRunner, buf.attenLastOutput, buf.hiddenStates.dataType);
                        }
                        Qwen3CudaAddTo(cudaRunner, buf.hiddenStates, buf.attenLastOutput);
                    }

                    Qwen3CudaRMSNorm(cudaRunner, buf.hiddenStates,
                                     *requireLocal(weight[postRmsName], postRmsName),
                                     rms_norm_eps, buf.attenInput);
                    int localBatch = buf.attenInput.dims[0];
                    int localLen = buf.attenInput.dims[1];
                    buf.attenInput.Reshape({localBatch * localLen, buf.attenInput.dims[2]});
                    Qwen3CudaLinear(cudaRunner, buf.attenInput,
                                    *requireLocal(weight[gateWeightName], gateWeightName),
                                    *GetEmptyData(), buf.routerLogits, true);
                    Qwen3CudaConvertToDataType(cudaRunner, buf.routerLogits,
                                               buf.routerLogitsTemp, DataType::FLOAT32);
                    Qwen3CudaSoftmax(cudaRunner, buf.routerLogitsTemp, buf.routerLogitsTemp, -1);
                    Data *localGateBias = nullptr;
                    if (weight.weight.find(gateBiasName) != weight.weight.end()) {
                        localGateBias = requireLocal(weight[gateBiasName], gateBiasName);
                    }
                    Qwen3CudaSelectExpert(cudaRunner, buf.routerLogitsTemp, buf.expertIndex, buf.expertScore,
                                          this->num_experts_per_tok, true,
                                          this->routed_scaling_factor, localGateBias);

                    auto &localWeights = moeWeightsByDevice.at(gpuId)[i];
                    auto &localBiass = moeBiassByDevice.at(gpuId)[i];
                    if (Qwen3MoeHasLocalMoeShard(localWeights)) {
                        Qwen3CudaMergeMOEBlock(cudaRunner, &buf.attenInput, &buf.expertIndex, &buf.expertScore,
                            &localWeights, &localBiass,
                            &buf.w1, &buf.w2, &buf.w3,
                            &buf.tempInput, &buf.tempOutput,
                            1.0f, &buf.moeFinal, i,
                            computeType, threadTpMoeAtype,
                            &buf.moeInputTemp, &buf.moeOutputTemp);
                    } else {
                        Qwen3MoeZeroCudaLike(buf.moeFinal, buf.hiddenStates, gpuId);
                    }
                    buf.moeFinal.Reshape(buf.hiddenStates.dims);
                    if (buf.moeFinal.dataType != buf.hiddenStates.dataType) {
                        Qwen3CudaToDataType(cudaRunner, buf.moeFinal, buf.hiddenStates.dataType);
                    }
                    if (tensorParallel) {
                        if (firstTensorParallelRank) {
                            Qwen3CudaAddTo(cudaRunner, buf.moeFinal, buf.hiddenStates);
                        }
                        FastllmNcclAllReduce(buf.moeFinal.cudaData, buf.hiddenStates.cudaData,
                                             buf.moeFinal.Count(0), buf.moeFinal.dataType, gpuId);
                    } else {
                        Qwen3CudaAddTo(cudaRunner, buf.hiddenStates, buf.moeFinal);
                    }
                }

                Qwen3CudaRMSNorm(cudaRunner, buf.hiddenStates,
                                 *requireLocal(weight["model.norm.weight"], "model.norm.weight"),
                                 rms_norm_eps, buf.hiddenStates);
                Qwen3CudaLinear(cudaRunner, buf.hiddenStates,
                                *requireLocal(weight["lm_head.weight"], "lm_head.weight"),
                                *requireLocal(GetThreadTensorParallelBias("lm_head.weight.tp_bias"),
                                              "lm_head.weight.tp_bias"),
                                state.logitsHalf);
	                Qwen3CudaConvertToDataType(cudaRunner, state.logitsHalf, state.logits, DataType::FLOAT32);
        };

        auto runGraphBody = [&]() {
            runGraphBodyWithBuffers(state.buffers, state.metaBuffers);
        };

        auto finishWithLogits = [&]() {
            Qwen3MoePrepareGraphCudaTensor(logits, state.logits, gpuId);
        };

        auto runWithoutGraph = [&]() -> bool {
            FastllmCudaMergeMOEClearGraphUnsafeFallbackFlag();
            runGraphBody();
            bool usedUnsafeMoeFallback = FastllmCudaMergeMOEUsedGraphUnsafeFallback();
            finishWithLogits();
            return usedUnsafeMoeFallback;
        };

        if (state.captured) {
            if (!syncGraphPeers()) {
                return false;
            }
            bool launchOk = FastllmCudaGraphLaunch(state.exec);
            if (Qwen3MoeSyncCudaGraphStage(this, state, graphParticipants,
                                           "replay", gpuId, launchOk)) {
                finishWithLogits();
                return true;
            }
            runWithoutGraph();
            return true;
        }

        if (!state.warmed) {
            bool usedUnsafeMoeFallback = runWithoutGraph();
            if (!syncGraphPeers(!usedUnsafeMoeFallback)) {
                if (usedUnsafeMoeFallback) {
                    printf("Warning: Qwen3-MOE CUDA graph disabled on gpu %d because MergeMOE used CPU expert routing fallback during warmup.\n",
                           gpuId);
                    fflush(stdout);
                }
                Qwen3MoeDisableCudaGraphState(this, state);
                return true;
            }
            state.warmed = true;
            return true;
        }

        void *capturedGraph = nullptr;
        if (!syncGraphPeers()) {
            return false;
        }
        bool beginOk = FastllmCudaGraphBeginCapture();
        if (!Qwen3MoeSyncCudaGraphStage(this, state, graphParticipants,
                                        "begin capture", gpuId, beginOk)) {
            if (beginOk) {
                Qwen3MoeAbortCudaGraphCapture();
            }
            runWithoutGraph();
            return true;
        }
        FastllmCudaMergeMOEClearGraphUnsafeFallbackFlag();
        runGraphBody();
        bool usedUnsafeMoeFallback = FastllmCudaMergeMOEUsedGraphUnsafeFallback();
        if (!syncGraphPeers(!usedUnsafeMoeFallback)) {
            if (usedUnsafeMoeFallback) {
                printf("Warning: Qwen3-MOE CUDA graph disabled on gpu %d because MergeMOE used CPU expert routing fallback during capture.\n",
                       gpuId);
                fflush(stdout);
            }
            Qwen3MoeAbortCudaGraphCapture();
            Qwen3MoeDisableCudaGraphState(this, state);
            runWithoutGraph();
            return true;
        }
        syncGraphPeers();
        bool endOk = FastllmCudaGraphEndCapture(&capturedGraph) && capturedGraph != nullptr;
        if (!Qwen3MoeSyncCudaGraphStage(this, state, graphParticipants,
                                        "end capture", gpuId, endOk)) {
            if (capturedGraph != nullptr) {
                FastllmCudaGraphDestroy(capturedGraph);
            }
            runWithoutGraph();
            return true;
        }

        void *capturedExec = nullptr;
        bool instantiateOk = FastllmCudaGraphInstantiate(capturedGraph, &capturedExec) &&
                             capturedExec != nullptr;
        if (!Qwen3MoeSyncCudaGraphStage(this, state, graphParticipants,
                                        "instantiate", gpuId, instantiateOk)) {
            if (capturedExec != nullptr) {
                FastllmCudaGraphExecDestroy(capturedExec);
            }
            FastllmCudaGraphDestroy(capturedGraph);
            runWithoutGraph();
            return true;
        }

        state.graph = capturedGraph;
        state.exec = capturedExec;
        state.captured = true;
        if (!syncGraphPeers()) {
            return false;
        }
        bool firstLaunchOk = FastllmCudaGraphLaunch(state.exec);
        if (!Qwen3MoeSyncCudaGraphStage(this, state, graphParticipants,
                                        "first launch", gpuId, firstLaunchOk)) {
            runWithoutGraph();
            return true;
        }
        finishWithLogits();
        return true;
#endif
    }

    void Qwen3MOEModel::ForwardSingleGPU(
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
        ErrorInFastLLM("Qwen3-MOE ForwardSingleGPU requires CUDA.\n");
#else
        AssertInFastLLM(ratios.find(gpuId) == ratios.end() || ratios[gpuId] > 0,
                        "Qwen3-MOE ForwardSingleGPU got invalid GPU ratio.\n");
        FastllmCudaSetDevice(gpuId);
        if (isPrefill && Qwen3MoeCudaGraphEnabled()) {
            Qwen3MoeCudaGraphDecodeState &graphState = GetQwen3MoeCudaGraphDecodeState(this, gpuId);
            std::lock_guard<std::mutex> graphGuard(graphState.mutex);
            Qwen3MoeDestroyCudaGraph(graphState);
            Qwen3MoeReinitializeForwardSingleBuffers(graphState.buffers);
            Qwen3MoeReinitializeForwardSingleBuffers(graphState.metaBuffers);
        }
        if (ForwardSingleGPUDecodeGraph(gpuId, ratios, batch, inputIds, positionIds,
                                        seqLens, pastKeyValues, all1, isPrefill,
                                        tensorParallel, firstTensorParallelRank,
                                        pagedCacheLayerOffset, logits)) {
            return;
        }

        Qwen3CudaDirectRunner cudaRunner(gpuId);
        auto requireLocal = [&](Data &data, const std::string &name) -> Data* {
            auto it = data.multiDeviceDatas.find(gpuId);
            if (it != data.multiDeviceDatas.end() && it->second != nullptr) {
                return it->second;
            }
            if (!tensorParallel) {
                if (!data.dims.empty() &&
                    (data.dataDevice != DataDevice::CUDA || data.cudaData == nullptr ||
                     data.dataDeviceIds.empty() || data.dataDeviceIds[0] != gpuId)) {
                    data.ToDevice(DataDevice::CUDA, {gpuId}, true);
                }
                return &data;
            }
            ErrorInFastLLM("Qwen3-MOE ForwardSingleGPU missing local tensor: " + name + ".\n");
            return nullptr;
        };

        Data hiddenStates;
        Qwen3CudaEmbeddingDirect(cudaRunner,
                                 *requireLocal((Data&)inputIds, "inputIds"),
                                 *requireLocal(weight["model.embed_tokens.weight"], "model.embed_tokens.weight"),
                                 hiddenStates);
            const DataType computeType = ResolveQwen3MoeThreadTpComputeType(this->dataType);
            const DataType threadTpMoeAtype = (this->moeAtype == DataType::FLOAT32) ? computeType : this->moeAtype;
            if (hiddenStates.dataType != computeType) {
                Qwen3CudaToDataType(cudaRunner, hiddenStates, computeType);
            }

            Data attenInput, qkv, q, attenOutput, attenLastOutput;
            Data routerLogits, routerLogitsTemp, expertIndex, expertScore;
            Data w1, w2, w3, tempInput, tempOutput, moeInputTemp, moeOutputTemp, moeFinal;
            Data qSizes, pageSizes, pageIndexs, lastPageLens, insertIndexs, insertPositions;
            std::vector<Data*> batchPastKeys(batch), batchPastValues(batch);
            bool generatedAppendParams = false;
            bool generatedDecodeParams = false;
            auto &moeWeightsByDevice = tensorParallel ? threadTpMoeWeights : singleGpuMoeWeights;
            auto &moeBiassByDevice = tensorParallel ? threadTpMoeBiass : singleGpuMoeBiass;

            for (int i = 0; i < block_cnt; i++) {
                std::string inputRmsName = "model.layers." + std::to_string(i) + ".input_layernorm.weight";
                std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
                std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
                std::string qNormName = "model.layers." + std::to_string(i) + ".self_attn.q_norm.weight";
                std::string kNormName = "model.layers." + std::to_string(i) + ".self_attn.k_norm.weight";
                std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";
                std::string oBiasName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.bias";
                std::string postRmsName = "model.layers." + std::to_string(i) + ".post_attention_layernorm.weight";
                std::string gateWeightName = "model.layers." + std::to_string(i) + ".mlp.gate.weight";
                std::string gateBiasName = "model.layers." + std::to_string(i) + ".mlp.gate.e_score_correction_bias";

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
                                "Qwen3-MOE ForwardSingleGPU got empty local attention shard.\n");
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
                if (tensorParallel) {
                    DataType residualType = hiddenStates.dataType;
                    if (firstTensorParallelRank) {
                        if (attenOutput.dataType == residualType) {
                            Qwen3CudaLinearAddBlock(cudaRunner, &attenOutput,
                                                   requireLocal(weight[oWeightName], oWeightName),
                                                   requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                                                   &attenLastOutput, &hiddenStates);
                        } else {
                            Qwen3CudaLinear(cudaRunner, attenOutput,
                                            *requireLocal(weight[oWeightName], oWeightName),
                                            *requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                                            attenLastOutput);
                            if (attenLastOutput.dataType != residualType) {
                                Qwen3CudaToDataType(cudaRunner, attenLastOutput, residualType);
                            }
                            Qwen3CudaAddTo(cudaRunner, hiddenStates, attenLastOutput);
                        }
                    } else {
                        Qwen3CudaLinear(cudaRunner, attenOutput,
                                        *requireLocal(weight[oWeightName], oWeightName),
                                        *requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                                        hiddenStates);
                        if (hiddenStates.dataType != residualType) {
                            Qwen3CudaToDataType(cudaRunner, hiddenStates, residualType);
                        }
                    }
                } else {
                    Qwen3CudaLinear(cudaRunner, attenOutput,
                                    *requireLocal(weight[oWeightName], oWeightName),
                                    *requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                                    attenLastOutput);
                }
                if (tensorParallel) {
                    FastllmNcclAllReduce(hiddenStates.cudaData, hiddenStates.cudaData,
                                         hiddenStates.Count(0), hiddenStates.dataType, gpuId);
                } else {
                    if (attenLastOutput.dataType != hiddenStates.dataType) {
                        Qwen3CudaToDataType(cudaRunner, attenLastOutput, hiddenStates.dataType);
                    }
                    Qwen3CudaAddTo(cudaRunner, hiddenStates, attenLastOutput);
                }
                Qwen3CudaRMSNorm(cudaRunner, hiddenStates,
                                 *requireLocal(weight[postRmsName], postRmsName),
                                 rms_norm_eps, attenInput);
                int localBatch = attenInput.dims[0];
                int localLen = attenInput.dims[1];
                attenInput.Reshape({localBatch * localLen, attenInput.dims[2]});
                Qwen3CudaLinear(cudaRunner, attenInput,
                                *requireLocal(weight[gateWeightName], gateWeightName),
                                *GetEmptyData(), routerLogits, true);
                Qwen3CudaConvertToDataType(cudaRunner, routerLogits, routerLogitsTemp, DataType::FLOAT32);
                Qwen3CudaSoftmax(cudaRunner, routerLogitsTemp, routerLogitsTemp, -1);
                Data *localGateBias = nullptr;
                if (weight.weight.find(gateBiasName) != weight.weight.end()) {
                    localGateBias = requireLocal(weight[gateBiasName], gateBiasName);
                }
                Qwen3CudaSelectExpert(cudaRunner, routerLogitsTemp, expertIndex, expertScore,
                                      this->num_experts_per_tok, true,
                                      this->routed_scaling_factor, localGateBias);
                auto &localWeights = moeWeightsByDevice.at(gpuId)[i];
                auto &localBiass = moeBiassByDevice.at(gpuId)[i];
                if (Qwen3MoeHasLocalMoeShard(localWeights)) {
                    Qwen3CudaMergeMOEBlock(cudaRunner, &attenInput, &expertIndex, &expertScore,
                        &localWeights, &localBiass,
                        &w1, &w2, &w3,
                        &tempInput, &tempOutput,
                        1.0f, &moeFinal, i,
                        computeType, threadTpMoeAtype,
                        &moeInputTemp, &moeOutputTemp);
                } else {
                    Qwen3MoeZeroCudaLike(moeFinal, hiddenStates, gpuId);
                }
                moeFinal.Reshape(hiddenStates.dims);
                if (moeFinal.dataType != hiddenStates.dataType) {
                    Qwen3CudaToDataType(cudaRunner, moeFinal, hiddenStates.dataType);
                }
                if (tensorParallel) {
                    if (firstTensorParallelRank) {
                        Qwen3CudaAddTo(cudaRunner, moeFinal, hiddenStates);
                    }
                    FastllmNcclAllReduce(moeFinal.cudaData, hiddenStates.cudaData,
                                         moeFinal.Count(0), moeFinal.dataType, gpuId);
                } else {
                    Qwen3CudaAddTo(cudaRunner, hiddenStates, moeFinal);
                }
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

    std::vector <int> Qwen3MOEModel::ForwardGPU(
        int batch,
        const Data &inputIds,
        const std::vector <Data*> &attentionMask,
        const std::vector <Data*> &positionIds,
        const std::vector <int> &seqLens,
        std::vector <std::pair <Data*, Data*> > &pastKeyValues,
        const std::vector <GenerationConfig> &generationConfigs,
        const LastTokensManager &lastTokens,
        std::vector <std::vector <float>*> *retLogits) {
#ifndef USE_CUDA
        return ForwardV2(batch, inputIds, attentionMask, positionIds, seqLens,
                         pastKeyValues, generationConfigs, lastTokens, retLogits);
#else
        std::vector<int> devices;
        std::map<int, int> ratios;
        if (!GetQwen3MoeGPUForwardDevices(this->deviceMap, devices, ratios)) {
            return ForwardV2(batch, inputIds, attentionMask, positionIds, seqLens,
                             pastKeyValues, generationConfigs, lastTokens, retLogits);
        }
        bool tensorParallel = devices.size() > 1;

        AssertInFastLLM((int)pastKeyValues.size() >= batch * block_cnt,
                        "Qwen3-MOE ForwardGPU: pastKeyValues size mismatch.\n");
        AssertInFastLLM((int)generationConfigs.size() >= batch,
                        "Qwen3-MOE ForwardGPU: generation config size mismatch.\n");
        AssertInFastLLM((int)positionIds.size() >= batch && positionIds[0] != nullptr,
                        "Qwen3-MOE ForwardGPU: positionIds size mismatch.\n");
        AssertInFastLLM(!GetKVCacheInCPU(),
                        "Qwen3-MOE ForwardGPU doesn't support CPU KV cache.\n");
        if (tensorParallel) {
            AssertInFastLLM(FastllmInitNccl(devices),
                            "Qwen3-MOE ForwardGPU requires NCCL initialization.\n");
        }

        if (threadTpPagedCacheBase < 0) {
            threadTpPagedCacheBase = qwen3MoeThreadTpNextPagedCacheBase.fetch_add(
                std::max(1, block_cnt * ((int)devices.size() + 1)));
        }

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
                                "Qwen3-MOE ForwardGPU: null positionIds.\n");
                for (int i = 0; i < seqLens[b]; i++) {
                    vPositionIds.push_back(((float*)positionIds[b]->cpuData)[i]);
                }
            }
            allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, (int)vPositionIds.size()}, vPositionIds));
        }

        Data gpuInputIds;
        gpuInputIds.CopyFrom(inputIds);
        if (tensorParallel) {
            PrepareMultiCudaReplicatedData(gpuInputIds, devices, true);
            PrepareMultiCudaReplicatedData(allPositionIds, devices, true);
        }

        std::vector<DivisionScheme> kvHeadSchemes;
        DivisionScheme lmHeadScheme;
        Data &lmHead = weight["lm_head.weight"];

        auto hasMoeCache = [&](const std::unordered_map<int, std::vector<std::vector<Data*> > > &weightCache,
                               const std::unordered_map<int, std::vector<std::vector<Data*> > > &biasCache) {
            int expectedSize = this->num_experts * 2 + 2;
            for (int device : devices) {
                auto weightIt = weightCache.find(device);
                auto biasIt = biasCache.find(device);
                if (weightIt == weightCache.end() || biasIt == biasCache.end() ||
                    (int)weightIt->second.size() != block_cnt || (int)biasIt->second.size() != block_cnt) {
                    return false;
                }
                for (int i = 0; i < block_cnt; i++) {
                    if ((int)weightIt->second[i].size() != expectedSize ||
                        (int)biasIt->second[i].size() != expectedSize) {
                        return false;
                    }
                    for (int j = 2; j < expectedSize; j++) {
                        if (weightIt->second[i][j] == nullptr) {
                            return false;
                        }
                    }
                }
            }
            return true;
        };

        auto fillMoeCache = [&](std::unordered_map<int, std::vector<std::vector<Data*> > > &weightCache,
                                std::unordered_map<int, std::vector<std::vector<Data*> > > &biasCache,
                                bool useLocalShards) {
            weightCache.clear();
            biasCache.clear();
            for (int device : devices) {
                auto &deviceWeights = weightCache[device];
                auto &deviceBiass = biasCache[device];
                deviceWeights.resize(block_cnt);
                deviceBiass.resize(block_cnt);
                for (int i = 0; i < block_cnt; i++) {
                    auto &layerWeights = deviceWeights[i];
                    auto &layerBiass = deviceBiass[i];
                    layerWeights.reserve(this->num_experts * 2 + 2);
                    layerBiass.reserve(this->num_experts * 2 + 2);
                    layerWeights.push_back(nullptr);
                    layerWeights.push_back(nullptr);
                    layerBiass.push_back(nullptr);
                    layerBiass.push_back(nullptr);
                    for (int j = 0; j < this->num_experts; j++) {
                        std::string gateupWeightName = "model.layers." + std::to_string(i) +
                            ".mlp.experts." + std::to_string(j) + ".gateup_proj.weight";
                        std::string downWeightName = "model.layers." + std::to_string(i) +
                            ".mlp.experts." + std::to_string(j) + ".down_proj.weight";
                        auto getLocalOrRoot = [&](Data &data) -> Data* {
                            if (useLocalShards) {
                                auto it = data.multiDeviceDatas.find(device);
                                if (it != data.multiDeviceDatas.end() && it->second != nullptr) {
                                    return it->second;
                                }
                            }
                            return &data;
                        };
                        layerWeights.push_back(getLocalOrRoot(weight[gateupWeightName]));
                        layerWeights.push_back(getLocalOrRoot(weight[downWeightName]));
                        layerBiass.push_back(nullptr);
                        layerBiass.push_back(nullptr);
                    }
                }
            }
        };

        if (tensorParallel) {
            std::lock_guard<std::mutex> guard(threadTpWeightPrepareLock);
            if (threadTpWeightsPrepared) {
                AssertInFastLLM(threadTpPreparedDevices == devices && threadTpPreparedRatios == ratios,
                                "Qwen3-MOE ForwardGPU thread TP device config changed after weights were prepared.\n");
                AssertInFastLLM((int)threadTpKVHeadSchemes.size() == block_cnt &&
                                !threadTpLmHeadScheme.empty() &&
                                hasMoeCache(threadTpMoeWeights, threadTpMoeBiass),
                                "Qwen3-MOE ForwardGPU thread TP cached weight schemes are incomplete.\n");
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
                    std::string gateWeightName = "model.layers." + std::to_string(i) + ".mlp.gate.weight";
                    std::string gateBiasName = "model.layers." + std::to_string(i) + ".mlp.gate.e_score_correction_bias";

                    AssertInFastLLM(weight.weight.find(mergeQkvWeightName) != weight.weight.end(),
                                    "Qwen3-MOE ForwardGPU requires merged qkv weight.\n");
                    AssertInFastLLM(weight.weight.find(gateWeightName) != weight.weight.end(),
                                    "Qwen3-MOE ForwardGPU requires router gate weight.\n");

                    prepareReplicated(inputRmsName);
                    prepareReplicated(qNormName);
                    prepareReplicated(kNormName);
                    prepareReplicated(postRmsName);
                    prepareReplicated(gateWeightName);
                    if (weight.weight.find(gateBiasName) != weight.weight.end()) {
                        prepareReplicated(gateBiasName);
                    }

                    Data &mergeW = weight[mergeQkvWeightName];
                    Data &mergeB = GetThreadTensorParallelBias(mergeQkvBiasName);
                    mergeW.tpPackType = TP_PACK_QKV;
                    mergeW.tpQHeads = num_attention_heads;
                    mergeW.tpKVHeads = num_key_value_heads;
                    mergeW.tpHeadDim = head_dim;
                    std::vector<int> devCopy = devices;
                    DivisionScheme qkvScheme = BuildMultiCudaRowSplitScheme(mergeW, devCopy, ratios);
                    AssertInFastLLM(SplitMultiCudaWeight(mergeW, mergeB, devCopy, qkvScheme, 0),
                                    "Qwen3-MOE ForwardGPU failed to split " + mergeQkvWeightName + ".\n");

                    int qWidth = num_attention_heads * head_dim;
                    DivisionScheme qScheme = ExtractQwen3MoeFirstRangeScheme(qkvScheme);
                    threadTpKVHeadSchemes[i] = ExtractQwen3MoeKVHeadScheme(qkvScheme, qWidth, head_dim);
                    Data &oB = GetThreadTensorParallelBias(oBiasName);
                    devCopy = devices;
                    AssertInFastLLM(SplitMultiCudaWeight(weight[oWeightName], oB, devCopy, qScheme, 1),
                                    "Qwen3-MOE ForwardGPU failed to split " + oWeightName + ".\n");

                    DivisionScheme gateScheme;
                    for (int j = 0; j < this->num_experts; j++) {
                        std::string gateupWeightName = "model.layers." + std::to_string(i) + ".mlp.experts." +
                                                        std::to_string(j) + ".gateup_proj.weight";
                        std::string downWeightName = "model.layers." + std::to_string(i) + ".mlp.experts." +
                                                      std::to_string(j) + ".down_proj.weight";
                        AssertInFastLLM(weight.weight.find(gateupWeightName) != weight.weight.end(),
                                        "Qwen3-MOE ForwardGPU requires merged expert gateup weight.\n");
                        AssertInFastLLM(weight.weight.find(downWeightName) != weight.weight.end(),
                                        "Qwen3-MOE ForwardGPU requires expert down weight.\n");

                        Data &gateup = weight[gateupWeightName];
                        Data &gateupBias = GetThreadTensorParallelBias(gateupWeightName + ".tp_bias");
                        gateup.tpLinearType = TP_LINEAR_ROW;
                        gateup.tpPackType = TP_PACK_GATEUP;
                        devCopy = devices;
                        gateScheme = BuildMultiCudaRowSplitScheme(gateup, devCopy, ratios);
                        AssertInFastLLM(SplitMultiCudaWeight(gateup, gateupBias, devCopy, gateScheme, 0),
                                        "Qwen3-MOE ForwardGPU failed to split " + gateupWeightName + ".\n");

                        Data &down = weight[downWeightName];
                        Data &downBias = GetThreadTensorParallelBias(downWeightName + ".tp_bias");
                        down.tpLinearType = TP_LINEAR_COLUMN;
                        DivisionScheme downScheme = ExtractQwen3MoeFirstRangeScheme(gateScheme);
                        devCopy = devices;
                        AssertInFastLLM(SplitMultiCudaWeight(down, downBias, devCopy, downScheme, 1),
                                        "Qwen3-MOE ForwardGPU failed to split " + downWeightName + ".\n");
                    }
                }

                fillMoeCache(threadTpMoeWeights, threadTpMoeBiass, true);

                Data &lmHeadBias = GetThreadTensorParallelBias("lm_head.weight.tp_bias");
                std::vector<int> devCopy = devices;
                threadTpLmHeadScheme = BuildMultiCudaRowSplitScheme(lmHead, devCopy, ratios);
                AssertInFastLLM(SplitMultiCudaWeight(lmHead, lmHeadBias, devCopy, threadTpLmHeadScheme, 0),
                                "Qwen3-MOE ForwardGPU failed to split lm_head.weight.\n");

                threadTpPreparedDevices = devices;
                threadTpPreparedRatios = ratios;
                threadTpWeightsPrepared = true;
            }
            kvHeadSchemes = threadTpKVHeadSchemes;
            lmHeadScheme = threadTpLmHeadScheme;
        } else {
            std::lock_guard<std::mutex> guard(threadTpWeightPrepareLock);
            if (!singleGpuWeightsPrepared.load(std::memory_order_relaxed) ||
                !hasMoeCache(singleGpuMoeWeights, singleGpuMoeBiass)) {
                int device = devices[0];
                for (int i = 0; i < block_cnt; i++) {
                    std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
                    std::string gateWeightName = "model.layers." + std::to_string(i) + ".mlp.gate.weight";
                    AssertInFastLLM(weight.weight.find(mergeQkvWeightName) != weight.weight.end(),
                                    "Qwen3-MOE ForwardGPU requires merged qkv weight.\n");
                    AssertInFastLLM(weight.weight.find(gateWeightName) != weight.weight.end(),
                                    "Qwen3-MOE ForwardGPU requires router gate weight.\n");

                    Data &mergeW = weight[mergeQkvWeightName];
                    mergeW.tpPackType = TP_PACK_QKV;
                    mergeW.tpQHeads = num_attention_heads;
                    mergeW.tpKVHeads = num_key_value_heads;
                    mergeW.tpHeadDim = head_dim;

                    for (int j = 0; j < this->num_experts; j++) {
                        std::string gateupWeightName = "model.layers." + std::to_string(i) +
                            ".mlp.experts." + std::to_string(j) + ".gateup_proj.weight";
                        std::string downWeightName = "model.layers." + std::to_string(i) +
                            ".mlp.experts." + std::to_string(j) + ".down_proj.weight";
                        AssertInFastLLM(weight.weight.find(gateupWeightName) != weight.weight.end(),
                                        "Qwen3-MOE ForwardGPU requires merged expert gateup weight.\n");
                        AssertInFastLLM(weight.weight.find(downWeightName) != weight.weight.end(),
                                        "Qwen3-MOE ForwardGPU requires expert down weight.\n");
                        Data &gateup = weight[gateupWeightName];
                        Data &down = weight[downWeightName];
                        gateup.tpLinearType = TP_LINEAR_ROW;
                        gateup.tpPackType = TP_PACK_GATEUP;
                        down.tpLinearType = TP_LINEAR_COLUMN;
                        gateup.ToDevice(DataDevice::CUDA, {device}, true);
                        down.ToDevice(DataDevice::CUDA, {device}, true);
                    }
                }
                fillMoeCache(singleGpuMoeWeights, singleGpuMoeBiass, false);
                singleGpuWeightsPrepared.store(true, std::memory_order_release);
            }
            kvHeadSchemes.assign(block_cnt, DivisionScheme());
            for (int i = 0; i < block_cnt; i++) {
                kvHeadSchemes[i][devices[0]].push_back({0, num_key_value_heads});
            }
            lmHeadScheme[devices[0]].push_back({0, lmHead.dims[0]});
        }

        const DataType computeType = ResolveQwen3MoeThreadTpComputeType(this->dataType);
        std::vector<std::vector<std::pair<Data*, Data*> > > localPastKeyValues;
        if (tensorParallel) {
            localPastKeyValues.resize(devices.size());
            for (int r = 0; r < (int)devices.size(); r++) {
                int device = devices[r];
                localPastKeyValues[r].resize(pastKeyValues.size());
                for (int i = 0; i < (int)pastKeyValues.size(); i++) {
                    DataType keyCacheType = ResolveQwen3MoeThreadTpCacheType(
                        pastKeyValues[i].first->dataType, computeType);
                    DataType valueCacheType = ResolveQwen3MoeThreadTpCacheType(
                        pastKeyValues[i].second->dataType, computeType);
                    localPastKeyValues[r][i].first = EnsureQwen3MoeThreadTpLocalCache(
                        *pastKeyValues[i].first, device, keyCacheType);
                    localPastKeyValues[r][i].second = EnsureQwen3MoeThreadTpLocalCache(
                        *pastKeyValues[i].second, device, valueCacheType);
                }
            }
        } else {
            int device = devices[0];
            for (int i = 0; i < (int)pastKeyValues.size(); i++) {
                DataType keyCacheType = ResolveQwen3MoeThreadTpCacheType(
                    pastKeyValues[i].first->dataType, computeType);
                DataType valueCacheType = ResolveQwen3MoeThreadTpCacheType(
                    pastKeyValues[i].second->dataType, computeType);
                PrepareQwen3MoeSingleCudaCache(*pastKeyValues[i].first, device, keyCacheType);
                PrepareQwen3MoeSingleCudaCache(*pastKeyValues[i].second, device, valueCacheType);
            }
        }

        std::vector<std::exception_ptr> errors(devices.size());
        std::vector<Data> localLogits(devices.size());
        if (devices.size() == 1) {
            ForwardSingleGPU(devices[0], ratios, batch, gpuInputIds, allPositionIds,
                             seqLens, pastKeyValues, all1, isPrefill,
                             false, true, threadTpPagedCacheBase, localLogits[0]);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(devices.size());
            for (int r = 0; r < (int)devices.size(); r++) {
                threads.emplace_back([&, r]() {
                    try {
                        ForwardSingleGPU(devices[r], ratios, batch, gpuInputIds, allPositionIds,
                                         seqLens, localPastKeyValues[r], all1, isPrefill,
                                         tensorParallel, r == 0,
                                         threadTpPagedCacheBase + r * block_cnt,
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

        if (tensorParallel) {
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < block_cnt; i++) {
                    int idx = b * block_cnt + i;
                    SyncQwen3MoeThreadTpRootCacheMeta(*pastKeyValues[idx].first, devices, kvHeadSchemes[i],
                                                      num_key_value_heads, head_dim);
                    SyncQwen3MoeThreadTpRootCacheMeta(*pastKeyValues[idx].second, devices, kvHeadSchemes[i],
                                                      num_key_value_heads, head_dim);
                }
            }
        }

        int vocabSize = lmHead.dims[0];
        bool allSimpleCudaSampling = true;
        int cudaSamplingTopK = 1;
        if (Qwen3MoeCanUseCudaFullLogitsSampling(generationConfigs, retLogits, batch,
                                                 allSimpleCudaSampling, cudaSamplingTopK)) {
            Data &fullCudaLogits = Qwen3MoeThreadLocalCudaSamplingFullLogits();
            Qwen3MoeGatherShardLogitsToRootCuda(devices[0], devices, lmHeadScheme,
                                                localLogits, batch, vocabSize,
                                                fullCudaLogits);
            void *oldExecutor = GetExecutor();
            Executor samplingExecutor;
            samplingExecutor.SetFirstDevice("cuda:" + std::to_string(devices[0]));
            SetCurrentThreadExecutor(&samplingExecutor);
            ResetLogitsOfEOS(batch, &fullCudaLogits, pastKeyValues, generationConfigs);
            SetCurrentThreadExecutor(oldExecutor);
            return Qwen3MoeSampleFromRootCudaLogits(devices[0], fullCudaLogits, batch,
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
                            "Qwen3-MOE ForwardGPU: local logits batch mismatch.\n");
            float *src = (float*)localLogits[r].cpuData;
            float *dst = (float*)fullLogits.cpuData;
            int localOffset = 0;
            for (auto &range : lmHeadScheme[device]) {
                int len = range.second - range.first;
                AssertInFastLLM(range.first >= 0 && range.second <= vocabSize &&
                                localOffset + len <= localVocab,
                                "Qwen3-MOE ForwardGPU: invalid lm_head split range.\n");
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

    std::vector <int> Qwen3MOEModel::ForwardV2ThreadTensorParallel(
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

    std::vector <int> Qwen3MOEModel::ForwardV2(int batch,
        const Data &inputIds,
        const std::vector <Data*> &attentionMask,
        const std::vector <Data*> &positionIds,
        const std::vector <int> &seqLens,
        std::vector <std::pair <Data*, Data*> > &pastKeyValues,
        const std::vector <GenerationConfig> &generationConfigs,
        const LastTokensManager &lastTokens,
        std::vector <std::vector <float>*> *retLogits) {
        if (IsThreadTensorParallelEnabled()) {
            return ForwardGPU(batch, inputIds, attentionMask, positionIds, seqLens,
                              pastKeyValues, generationConfigs, lastTokens, retLogits);
        }

        int seqLen = inputIds.dims[1];

        Data hiddenStates;
        Data attenInput;
        Data q, qkv;
        Data attenLastOutput;
        Data w1, w2, w3, routerLogits, gate, attenPart, moePart, moeFinal, sharedGate;
        Data tempInput, tempOutput;
        Data routerLogitsTemp;
        Data moeInputTemp, moeOutputTemp;
        Data expertIndex, expertScore;

        std::vector<Data*> batchPastKeys;
        std::vector<Data*> batchPastValues;
        batchPastKeys.resize(batch);
        batchPastValues.resize(batch);

        Data allPositionIds;
        Data qSizes, pageSizes, pageIndexs, lastPageLens;
        Data insertIndexs, insertPositions;
        Data attenOutput;
        bool generatedBatchDecodeParams = false;
        bool generatedAppendPagedCacheBatchParams = false;

        if (weights.size() == 0) {
            weights.resize(block_cnt);
            biass.resize(block_cnt);
            for (int i = 0; i < block_cnt; i++) {
                weights[i].push_back(nullptr);
                weights[i].push_back(nullptr);
                biass[i].push_back(nullptr);
                biass[i].push_back(nullptr);
                for (int j = 0; j < this->num_experts; j++) {
                    Data &gateup = weight["model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".gateup_proj.weight"];
                    Data &down   = weight["model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".down_proj.weight"];
                    gateup.tpLinearType = TP_LINEAR_ROW;
                    gateup.tpPackType   = TP_PACK_GATEUP;
                    down.tpLinearType   = TP_LINEAR_COLUMN;
                    weights[i].push_back(&gateup);
                    weights[i].push_back(&down);
                    biass[i].push_back(nullptr);
                    biass[i].push_back(nullptr);
                }
            }
        }

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

        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
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

            bool hasMergeQkv = (weight.weight.find(mergeQkvWeightName) != weight.weight.end());
            Data *mergeW = hasMergeQkv ? &weight[mergeQkvWeightName] : GetEmptyData();
            Data *mergeB = hasMergeQkv ? &weight[mergeQkvBiasName] : GetEmptyData();
            Data *qW = hasMergeQkv ? GetEmptyData() : &weight[qWeightName];
            Data *qB = hasMergeQkv ? GetEmptyData() : &weight[qBiasName];
            Data *kW = hasMergeQkv ? GetEmptyData() : &weight[kWeightName];
            Data *kB = hasMergeQkv ? GetEmptyData() : &weight[kBiasName];
            Data *vW = hasMergeQkv ? GetEmptyData() : &weight[vWeightName];
            Data *vB = hasMergeQkv ? GetEmptyData() : &weight[vBiasName];

            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    rms_norm_eps, attenInput);
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

            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], rms_norm_eps, attenInput);
        
            // 2. moe mlp
            {
                // 这里是moe mlp
                std::string gateWeightName = "model.layers." + std::to_string(i) + ".mlp.gate.weight";
                std::string gateBiasName = "model.layers." + std::to_string(i) + ".mlp.gate.e_score_correction_bias";

                int batch = attenInput.dims[0], len = attenInput.dims[1];
                attenInput.Reshape({batch * len, attenInput.dims[2]});
                Linear(attenInput, weight[gateWeightName], Data(), routerLogits, true);

                ToDataType(routerLogits, routerLogitsTemp, DataType::FLOAT32);
                bool needNorm = true;
                Softmax(routerLogitsTemp, routerLogitsTemp, -1);

                if (weight.weight.find("model.layers." + std::to_string(i) + ".mlp.experts.0.gateup_proj.weight") != weight.weight.end() 
                    && CanRunMergeMOE(attenInput, biass[i])) {
                    SelectExpert(routerLogitsTemp, expertIndex, expertScore, this->num_experts_per_tok, needNorm, 
                                this->routed_scaling_factor, weight.weight.find(gateBiasName) != weight.weight.end() ? &weight[gateBiasName] : nullptr);
                }
                ApplyDeviceMap(this->moeDeviceMap, i + 1, block_cnt);
                MergeMOEBlock(&attenInput, &expertIndex, &expertScore,
                    &weights[i], &biass[i],
                    &w1, &w2, &w3,
                    &tempInput, &tempOutput,
                    1.0f, &moeFinal, i,
                    this->dataType, this->moeAtype,
                    &moeInputTemp, &moeOutputTemp);
                moeFinal.Reshape(hiddenStates.dims);
                ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
                AddTo(hiddenStates, moeFinal);
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
        return lastRet;
    }

    bool Qwen3MOEModel::NeedAttentionMask(int qlen, int klen) {
        return false;
    }

    void Qwen3MOEModel::FillLLMInputsBatch(std::vector<std::vector<float>> &inputTokens,
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

    std::string Qwen3MOEModel::MakeInput(const std::string &history, int round, const std::string &input) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
    }

    std::string Qwen3MOEModel::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role + output + history_sep;
    }

    void Qwen3MOEModel::WarmUp() {
        printf("Warmup...\n");
        int oldTopk = this->num_experts_per_tok;
        if (!IsThreadTensorParallelEnabled()) {
            this->num_experts_per_tok = this->num_experts;
        }

        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {1});
        Data attentionMask = Data(this->dataType, {1, 1}, {0});
        Data positionIds = Data(this->dataType, {1, 1}, {0, 0});

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(this->dataType),
                                                   Data(this->dataType)));
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        this->num_experts_per_tok = oldTopk;
        elementsInKVCachePerToken = (long long)block_cnt * 
            (pastKeyValues[0].first.dims[0] * pastKeyValues[0].first.dims[2] + 
             pastKeyValues[0].second.dims[0] * pastKeyValues[0].second.dims[2]);
        printf("finish.\n");
    }
}
