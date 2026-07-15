#include "baseblock.h"
#include "models/basellm.h"

#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <memory>
#include <string>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif

namespace fastllm {
    namespace {
        static bool NeedRepeatPenalty(const GenerationConfig &config) {
            float diff = config.repeat_penalty - 1.0f;
            return diff > 1e-6f || diff < -1e-6f;
        }

#ifdef USE_CUDA
        struct TensorParallelGreedyDeviceWorkspace {
            Data cudaIds;
            Data cudaScores;
            Data cudaGather;
            int *hostIds = nullptr;
            float *hostScores = nullptr;
            uint8_t *hostGather = nullptr;
            int hostCapacity = 0;
            int gatherCapacity = 0;
            void *gatherGraphExec = nullptr;
            bool gatherGraphCreateFailed = false;
            int gatherGraphRootDevice = -1;
            int gatherGraphBatch = 0;
            void *gatherGraphCudaBuffer = nullptr;
            void *gatherGraphHostBuffer = nullptr;
            std::vector<int> gatherGraphDevices;
            std::vector<const void*> gatherGraphIds;
            std::vector<const void*> gatherGraphScores;

            void ResetGatherGraph() {
                if (gatherGraphExec != nullptr) {
                    if (gatherGraphRootDevice >= 0) {
                        FastllmCudaSetDevice(gatherGraphRootDevice);
                    }
                    FastllmCudaGraphExecDestroy(gatherGraphExec);
                    gatherGraphExec = nullptr;
                }
                gatherGraphRootDevice = -1;
                gatherGraphBatch = 0;
                gatherGraphCudaBuffer = nullptr;
                gatherGraphHostBuffer = nullptr;
                gatherGraphDevices.clear();
                gatherGraphIds.clear();
                gatherGraphScores.clear();
            }

            bool GatherGraphMatches(
                    int rootDevice, int batch, void *cudaBuffer,
                    void *hostBuffer, const std::vector<int> &devices,
                    const std::vector<Data*> &ids,
                    const std::vector<Data*> &scores) const {
                if (gatherGraphExec == nullptr ||
                    gatherGraphRootDevice != rootDevice ||
                    gatherGraphBatch != batch ||
                    gatherGraphCudaBuffer != cudaBuffer ||
                    gatherGraphHostBuffer != hostBuffer ||
                    gatherGraphDevices != devices ||
                    gatherGraphIds.size() != ids.size() ||
                    gatherGraphScores.size() != scores.size()) {
                    return false;
                }
                for (size_t i = 0; i < ids.size(); ++i) {
                    if (ids[i] == nullptr || scores[i] == nullptr ||
                        gatherGraphIds[i] != ids[i]->cudaData ||
                        gatherGraphScores[i] != scores[i]->cudaData) {
                        return false;
                    }
                }
                return true;
            }

            void SetGatherGraphSignature(
                    int rootDevice, int batch, void *cudaBuffer,
                    void *hostBuffer, const std::vector<int> &devices,
                    const std::vector<Data*> &ids,
                    const std::vector<Data*> &scores) {
                gatherGraphRootDevice = rootDevice;
                gatherGraphBatch = batch;
                gatherGraphCudaBuffer = cudaBuffer;
                gatherGraphHostBuffer = hostBuffer;
                gatherGraphDevices = devices;
                gatherGraphIds.resize(ids.size());
                gatherGraphScores.resize(scores.size());
                for (size_t i = 0; i < ids.size(); ++i) {
                    gatherGraphIds[i] = ids[i]->cudaData;
                    gatherGraphScores[i] = scores[i]->cudaData;
                }
            }

            void EnsureHostCapacity(int batch) {
                if (batch <= hostCapacity && hostIds != nullptr && hostScores != nullptr) {
                    return;
                }
                if (hostIds != nullptr) {
                    FastllmCudaHostFree(hostIds);
                }
                if (hostScores != nullptr) {
                    FastllmCudaHostFree(hostScores);
                }
                hostIds = (int*)FastllmCudaHostMalloc((size_t)batch * sizeof(int));
                hostScores = (float*)FastllmCudaHostMalloc((size_t)batch * sizeof(float));
                hostCapacity = batch;
            }

            void EnsureCudaCapacity(int device, int batch) {
                FastllmCudaSetDevice(device);
                cudaIds.dataType = DataType::INT32;
                cudaIds.UpdateUnitSize();
                cudaIds.Resize({batch});
                cudaIds.dataDevice = DataDevice::CUDA;
                cudaIds.dataDeviceIds = {device};
                cudaIds.Allocate(false);
                cudaScores.dataType = DataType::FLOAT32;
                cudaScores.UpdateUnitSize();
                cudaScores.Resize({batch});
                cudaScores.dataDevice = DataDevice::CUDA;
                cudaScores.dataDeviceIds = {device};
                cudaScores.Allocate(false);
            }

            void EnsureGatherCapacity(int device, int candidates) {
                if (candidates <= gatherCapacity && cudaGather.cudaData != nullptr &&
                    hostGather != nullptr) {
                    return;
                }
                ResetGatherGraph();
                if (hostGather != nullptr) {
                    FastllmCudaHostFree(hostGather);
                    hostGather = nullptr;
                }
                FastllmCudaSetDevice(device);
                cudaGather.dataType = DataType::INT32;
                cudaGather.UpdateUnitSize();
                cudaGather.Resize({2 * candidates});
                cudaGather.dataDevice = DataDevice::CUDA;
                cudaGather.dataDeviceIds = {device};
                cudaGather.Allocate(false);
                hostGather = (uint8_t*)FastllmCudaHostMalloc(
                    (size_t)2 * candidates * sizeof(int));
                gatherCapacity = candidates;
            }
        };

        struct TensorParallelGreedyWorkspace {
            std::map<int, std::unique_ptr<TensorParallelGreedyDeviceWorkspace> > devices;

            TensorParallelGreedyDeviceWorkspace *Get(int device) {
                auto &slot = devices[device];
                if (!slot) {
                    slot.reset(new TensorParallelGreedyDeviceWorkspace());
                }
                return slot.get();
            }
        };

        static TensorParallelGreedyWorkspace &GetTensorParallelGreedyWorkspace() {
            // The scheduler thread can outlive CUDA's static teardown.  Keep this
            // tiny workspace process-lifetime, like the MultiCUDA worker pool.
            static thread_local auto *workspace = new TensorParallelGreedyWorkspace();
            return *workspace;
        }

        static void GatherTensorParallelLogitsToRoot(Data &logits) {
            if (!logits.multiDeviceData || !logits.IsTensorParallelSharded() ||
                logits.multiDeviceDatas.empty()) {
                return;
            }

            int axis = (logits.tpAxis % (int)logits.dims.size() + (int)logits.dims.size()) %
                       (int)logits.dims.size();
            AssertInFastLLM(axis == (int)logits.dims.size() - 1,
                            "Tensor-parallel logits must be sharded on the vocabulary axis.\n");
            int vocabSize = logits.dims.back();
            long long rows = logits.Count(0) / vocabSize;
            AssertInFastLLM(vocabSize > 0 && rows > 0,
                            "Tensor-parallel logits have an invalid shape.\n");

            int rootDevice = logits.multiDeviceDatas.begin()->first;
            FastllmCudaSetDevice(rootDevice);
            if (logits.cudaData != nullptr) {
                logits.FreeSpace();
            }
            logits.dataDevice = DataDevice::CUDA;
            logits.dataDeviceIds = {rootDevice};
            logits.expansionSize = 0;
            logits.expansionBytes = 0;
            logits.Allocate();

            size_t elementBytes = logits.unitSize / logits.unitSizeDiv;
            AssertInFastLLM(logits.unitSizeDiv == 1 && elementBytes > 0,
                            "Tensor-parallel logits must use a byte-aligned data type.\n");
            for (auto &deviceData : logits.multiDeviceDatas) {
                int device = deviceData.first;
                Data *local = deviceData.second;
                auto rangesIt = logits.tpRanges.find(device);
                AssertInFastLLM(local != nullptr && local->cudaData != nullptr &&
                                local->dims.size() == logits.dims.size() &&
                                rangesIt != logits.tpRanges.end(),
                                "Tensor-parallel logits are missing a local CUDA shard.\n");

                int localVocab = local->dims.back();
                int localOffset = 0;
                for (auto &range : rangesIt->second) {
                    int len = range.second - range.first;
                    AssertInFastLLM(range.first >= 0 && range.second <= vocabSize && len >= 0 &&
                                    localOffset + len <= localVocab,
                                    "Tensor-parallel logits have an invalid shard range.\n");
                    if (len > 0) {
                        FastllmCudaMemcpy2DDeviceToDeviceAuto(
                            (uint8_t*)logits.cudaData + (size_t)range.first * elementBytes,
                            (size_t)vocabSize * elementBytes,
                            (uint8_t*)local->cudaData + (size_t)localOffset * elementBytes,
                            (size_t)localVocab * elementBytes,
                            (size_t)len * elementBytes,
                            (size_t)rows,
                            rootDevice,
                            device);
                    }
                    localOffset += len;
                }
                AssertInFastLLM(localOffset == localVocab,
                                "Tensor-parallel logits shard size mismatch.\n");
            }

            logits.ResetMultiDeviceState();
            logits.dataDevice = DataDevice::CUDA;
            logits.dataDeviceIds = {rootDevice};
        }

        static bool SampleTensorParallelGreedyLogits(
                Data &logits, int batch, std::vector<int> &tokens,
                Data *precomputedIds, Data *precomputedScores,
                const std::map<int, void*> *precomputedReadyEvents) {
            if (!logits.multiDeviceData || !logits.IsTensorParallelSharded() ||
                logits.dataType != DataType::FLOAT32 || logits.dims.empty() ||
                batch <= 0 || logits.multiDeviceDatas.empty()) {
                return false;
            }
            int axis = (logits.tpAxis % (int)logits.dims.size() +
                        (int)logits.dims.size()) % (int)logits.dims.size();
            if (axis != (int)logits.dims.size() - 1) {
                return false;
            }

            int vocabSize = logits.dims.back();
            tokens.assign(batch, 0);
            std::vector<float> bestScores(batch, -1.0e30f);
            std::vector<int> devices;
            std::vector<Data*> locals;
            std::vector<Data*> cudaIds;
            std::vector<Data*> cudaScores;
            std::vector<int*> localIds;
            std::vector<float*> localScores;
            std::vector<TensorParallelGreedyDeviceWorkspace*> workspaces;
            bool usePrecomputed = precomputedIds != nullptr &&
                precomputedScores != nullptr &&
                precomputedIds->multiDeviceData &&
                precomputedScores->multiDeviceData;
            cudaIds.reserve(logits.multiDeviceDatas.size());
            cudaScores.reserve(logits.multiDeviceDatas.size());
            localIds.reserve(logits.multiDeviceDatas.size());
            localScores.reserve(logits.multiDeviceDatas.size());
            workspaces.reserve(logits.multiDeviceDatas.size());
            int rank = 0;
            for (const auto &deviceData : logits.multiDeviceDatas) {
                int device = deviceData.first;
                Data *local = deviceData.second;
                auto rangesIt = logits.tpRanges.find(device);
                if (local == nullptr || local->cudaData == nullptr ||
                    local->dataType != DataType::FLOAT32 || local->dims.empty() ||
                    rangesIt == logits.tpRanges.end()) {
                    return false;
                }
                int localVocab = local->dims.back();
                if (localVocab <= 0 ||
                    local->Count(0) / localVocab != (unsigned long long)batch) {
                    return false;
                }

                devices.push_back(device);
                locals.push_back(local);
                TensorParallelGreedyDeviceWorkspace *workspace =
                    GetTensorParallelGreedyWorkspace().Get(device);
                workspace->EnsureHostCapacity(batch);
                Data *candidateIds = nullptr;
                Data *candidateScores = nullptr;
                if (usePrecomputed) {
                    auto idIt = precomputedIds->multiDeviceDatas.find(device);
                    auto scoreIt = precomputedScores->multiDeviceDatas.find(device);
                    if (idIt == precomputedIds->multiDeviceDatas.end() ||
                        scoreIt == precomputedScores->multiDeviceDatas.end() ||
                        idIt->second == nullptr || scoreIt->second == nullptr ||
                        idIt->second->dataType != DataType::INT32 ||
                        scoreIt->second->dataType != DataType::FLOAT32 ||
                        idIt->second->Count(0) < (unsigned long long)batch ||
                        scoreIt->second->Count(0) < (unsigned long long)batch) {
                        usePrecomputed = false;
                    } else {
                        candidateIds = idIt->second;
                        candidateScores = scoreIt->second;
                    }
                }
                if (!usePrecomputed) {
                    workspace->EnsureCudaCapacity(device, batch);
                    candidateIds = &workspace->cudaIds;
                    candidateScores = &workspace->cudaScores;
                }
                cudaIds.push_back(candidateIds);
                cudaScores.push_back(candidateScores);
                localIds.push_back(workspace->hostIds);
                localScores.push_back(workspace->hostScores);
                workspaces.push_back(workspace);
                rank++;
            }

            // If a malformed precomputed set was detected after earlier ranks were
            // accepted, rebuild every candidate pointer from the persistent fallback.
            if (!usePrecomputed) {
                for (int r = 0; r < (int)devices.size(); ++r) {
                    workspaces[r]->EnsureCudaCapacity(devices[r], batch);
                    cudaIds[r] = &workspaces[r]->cudaIds;
                    cudaScores[r] = &workspaces[r]->cudaScores;
                }
            }

            std::vector<int> sampled(devices.size(), 0);
            bool rootGathered = usePrecomputed && precomputedReadyEvents != nullptr &&
                precomputedReadyEvents->size() >= devices.size();
            if (rootGathered) {
                for (int device : devices) {
                    auto eventIt = precomputedReadyEvents->find(device);
                    if (eventIt == precomputedReadyEvents->end() ||
                        eventIt->second == nullptr) {
                        rootGathered = false;
                        break;
                    }
                }
            }
            if (rootGathered) {
                const int rootDevice = devices.front();
                const int candidates = (int)devices.size() * batch;
                TensorParallelGreedyDeviceWorkspace *rootWorkspace = workspaces.front();
                rootWorkspace->EnsureGatherCapacity(rootDevice, candidates);
                uint8_t *cudaGather = (uint8_t*)rootWorkspace->cudaGather.cudaData;
                const size_t rankBytes = (size_t)batch * sizeof(int);
                const size_t scoreBase = (size_t)candidates * sizeof(int);
                FastllmCudaSetDevice(rootDevice);
                for (int r = 0; r < (int)devices.size(); ++r) {
                    FastllmCudaCurrentThreadStreamWaitEvent(
                        precomputedReadyEvents->at(devices[r]));
                }

                auto enqueueGatherCopies = [&]() {
                    bool ok = true;
                    for (int r = 0; r < (int)devices.size(); ++r) {
                        ok = FastllmCudaMemcpyPeerAsyncCurrentThread(
                            rootDevice, cudaGather + (size_t)r * rankBytes,
                            devices[r], cudaIds[r]->cudaData, rankBytes) && ok;
                        ok = FastllmCudaMemcpyPeerAsyncCurrentThread(
                            rootDevice,
                            cudaGather + scoreBase + (size_t)r * rankBytes,
                            devices[r], cudaScores[r]->cudaData, rankBytes) && ok;
                    }
                    if (ok) {
                        ok = FastllmCudaCopyFromDeviceToHostAsyncCurrentThread(
                            rootWorkspace->hostGather, cudaGather,
                            (size_t)2 * candidates * sizeof(int));
                    }
                    return ok;
                };

                bool gatherGraphLaunched = false;
                bool useGatherGraph = !rootWorkspace->gatherGraphCreateFailed;
                if (useGatherGraph &&
                    !rootWorkspace->GatherGraphMatches(
                        rootDevice, batch, cudaGather,
                        rootWorkspace->hostGather, devices,
                        cudaIds, cudaScores)) {
                    rootWorkspace->ResetGatherGraph();
                    void *exec = nullptr;
                    std::vector<const void*> idPointers(cudaIds.size());
                    std::vector<const void*> scorePointers(cudaScores.size());
                    for (size_t i = 0; i < cudaIds.size(); ++i) {
                        idPointers[i] = cudaIds[i]->cudaData;
                        scorePointers[i] = cudaScores[i]->cudaData;
                    }
                    // cudaMemcpyPeerAsync is not stream-capturable on every
                    // CUDA/PCIe topology.  Build explicit memcpy graph nodes
                    // instead.  Replay completion waits remain outside the
                    // graph, so this graph is only pointer-specific.
                    bool created =
                        FastllmCudaTensorParallelGreedyGatherGraphCreate(
                            rootDevice, (int)devices.size(),
                            idPointers.data(), scorePointers.data(),
                            cudaGather, rootWorkspace->hostGather,
                            rankBytes, scoreBase,
                            (size_t)2 * candidates * sizeof(int), &exec);
                    if (created && exec != nullptr) {
                        rootWorkspace->gatherGraphExec = exec;
                        rootWorkspace->SetGatherGraphSignature(
                            rootDevice, batch, cudaGather,
                            rootWorkspace->hostGather, devices,
                            cudaIds, cudaScores);
                    } else {
                        FastllmCudaGraphExecDestroy(exec);
                        rootWorkspace->gatherGraphCreateFailed = true;
                        std::fprintf(stderr,
                            "[Fastllm] TP greedy gather CUDA graph unavailable; "
                            "using async-copy fallback (%s).\n",
                            FastllmCudaGraphLastError());
                        std::fflush(stderr);
                    }
                }
                if (useGatherGraph && rootWorkspace->gatherGraphExec != nullptr) {
                    gatherGraphLaunched =
                        FastllmCudaGraphLaunch(rootWorkspace->gatherGraphExec);
                    if (!gatherGraphLaunched) {
                        rootWorkspace->ResetGatherGraph();
                        rootWorkspace->gatherGraphCreateFailed = true;
                    }
                }
                if (!gatherGraphLaunched) {
                    rootGathered = enqueueGatherCopies();
                }
                FastllmCudaSyncCurrentThreadStream();
                rootGathered = gatherGraphLaunched || rootGathered;
                if (rootGathered) {
                    int *gatheredIds = (int*)rootWorkspace->hostGather;
                    float *gatheredScores = (float*)(
                        rootWorkspace->hostGather + scoreBase);
                    for (int r = 0; r < (int)devices.size(); ++r) {
                        localIds[r] = gatheredIds + (size_t)r * batch;
                        localScores[r] = gatheredScores + (size_t)r * batch;
                        sampled[r] = 1;
                    }
                }
            }
            std::function<void(int, int)> sampleLocal = [&](int rank, int device) {
                Data *local = locals[rank];
                int localVocab = local->dims.back();
                sampled[rank] = usePrecomputed || FastllmCudaGreedySamplingWithScores(
                    (float*)local->cudaData, (int*)cudaIds[rank]->cudaData,
                    (float*)cudaScores[rank]->cudaData, batch, localVocab);
                if (!sampled[rank]) {
                    return;
                }
                FastllmCudaCopyFromDeviceToHost(
                    localIds[rank], cudaIds[rank]->cudaData,
                    (size_t)batch * sizeof(int));
                FastllmCudaCopyFromDeviceToHost(
                    localScores[rank], cudaScores[rank]->cudaData,
                    (size_t)batch * sizeof(float));
            };
            if (!rootGathered) {
                if (!MultiCudaRunDeviceCallbacks(devices, sampleLocal)) {
                    for (int r = 0; r < (int)devices.size(); ++r) {
                        FastllmCudaSetDevice(devices[r]);
                        sampleLocal(r, devices[r]);
                    }
                }
            }
            if (!std::all_of(sampled.begin(), sampled.end(),
                             [](int state) { return state != 0; })) {
                return false;
            }

            for (int r = 0; r < (int)devices.size(); ++r) {
                auto rangesIt = logits.tpRanges.find(devices[r]);
                for (int b = 0; b < batch; ++b) {
                    int localId = localIds[r][b];
                    int localOffset = 0;
                    int globalId = -1;
                    for (const auto &range : rangesIt->second) {
                        int len = range.second - range.first;
                        if (localId >= localOffset && localId < localOffset + len) {
                            globalId = range.first + localId - localOffset;
                            break;
                        }
                        localOffset += len;
                    }
                    if (globalId < 0 || globalId >= vocabSize) {
                        return false;
                    }
                    if (localScores[r][b] > bestScores[b] ||
                        (localScores[r][b] == bestScores[b] && globalId < tokens[b])) {
                        bestScores[b] = localScores[r][b];
                        tokens[b] = globalId;
                    }
                }
            }
            return true;
        }
#endif
    }

    struct LogitsDebugOptions {
        bool printLogits = false;
        std::string dumpPath;
    };

    static bool IsLogitsEnvEnabled(const char *v) {
        return v != nullptr && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
    }

    static const LogitsDebugOptions &GetLogitsDebugOptions() {
        static const LogitsDebugOptions options = []() {
            LogitsDebugOptions ret;
            ret.printLogits = GetFastllmEnv().printLogits ||
                              IsLogitsEnvEnabled(std::getenv("FASTLLM_PRINT_LOGITS"));
            const char *path = std::getenv("FASTLLM_DSV4_DUMP_LOGITS");
            if (path != nullptr) {
                ret.dumpPath = path;
            }
            return ret;
        }();
        return options;
    }

    static bool ShouldPrintLogits() {
        return GetLogitsDebugOptions().printLogits;
    }

    static const std::string &LogitsDumpPath() {
        return GetLogitsDebugOptions().dumpPath;
    }

    static void DumpLogitsIfNeeded(Data &logits) {
        bool needPrint = ShouldPrintLogits();
        if (!needPrint) {
            return;
        }
        logits.ToDevice(DataDevice::CPU);
        float *p = (float*)logits.cpuData;
        uint64_t count = logits.Count(0);
    }

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
        std::vector<int> &lastRet,
        Data *precomputedLogits,
        Data *precomputedGreedyIds,
        Data *precomputedGreedyScores,
        const std::map<int, void*> *precomputedReadyEvents
    ) {
        Data ownedLogits;
        Data &logits = precomputedLogits == nullptr ? ownedLogits : *precomputedLogits;
        std::vector<Data> curLogits;
        curLogits.resize(batch);

        if (precomputedLogits == nullptr) {
            if (!all1) {
                hiddenStates->ResetMultiDeviceState();
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
        } else {
            AssertInFastLLM(logits.dataType == DataType::FLOAT32,
                            "Precomputed sampling logits must use FLOAT32.\n");
        }
#ifdef USE_CUDA
        bool canUseShardedGreedy = logits.IsTensorParallelSharded() &&
            !ShouldPrintLogits();
        for (int b = 0; b < batch && canUseShardedGreedy; ++b) {
            canUseShardedGreedy = generationConfigs[b].IsSimpleGreedy() &&
                generationConfigs[b].output_token_least <= 0 &&
                !generationConfigs[b].output_logits;
        }
        if (canUseShardedGreedy &&
            SampleTensorParallelGreedyLogits(
                logits, batch, lastRet,
                precomputedGreedyIds, precomputedGreedyScores,
                precomputedReadyEvents)) {
            return;
        }
        GatherTensorParallelLogitsToRoot(logits);
#endif

        bool allSimple = true, needLogits = false, needRepeatPenalty = false, needToolNameMask = false;
        int maxTopK = 1;
        for (int b = 0; b < batch; b++) {
            if (!generationConfigs[b].IsSimpleGreedy()) {
                allSimple = false;
                break;
            }
        }
        for (int b = 0; b < batch; b++) {
            needLogits |= generationConfigs[b].output_logits;
            needRepeatPenalty |= NeedRepeatPenalty(generationConfigs[b]);
            needToolNameMask |= !generationConfigs[b].tool_call_allowed_token_ids.empty();
            maxTopK = std::max(maxTopK, generationConfigs[b].top_k);
        }

        model->ResetLogitsOfEOS(batch, &logits, pastKeyValues, generationConfigs);
        DumpLogitsIfNeeded(logits);
        if (ShouldPrintLogits()) {
            printf("LLMSamplingBlock logits:\n");
            logits.Print();
        }

        if (allSimple) {
            Data topk;
            TopK(logits, topk, 1);
            topk.ToDevice(DataDevice::CPU);
            float *topkData = (float*)topk.cpuData;
            for (int b = 0; b < batch; b++) {
                lastRet.push_back((int) (topkData[0] + 1e-3));
                topkData += topk.Count(2);
            }
        } else if (!needLogits && !needToolNameMask) {
            if (needRepeatPenalty) {
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
            }
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
                    LastTokensUnit emptyUnit;
                    const LastTokensUnit &unit =
                            b < (int)lastTokens.units.size() ? lastTokens.units[b] : emptyUnit;
                    lastRet.push_back(LLMSampling(curLogit, 0, generationConfigs[b], unit));
                }
            }
        }
    }
}
