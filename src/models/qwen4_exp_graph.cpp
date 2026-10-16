// Shared CUDA graph storage, capture/replay, attention math and state publication.
#include "qwen4_exp_tp.h"
#include "qwen4_exp_qsa.h"
#include "executor.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include "qwen4_exp_graph.h"
#include "qwen4_exp_execution.h"
#include "models/qwen4_paged_cache.h"
#include "utils.h"
#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-graph-workspace.h"
#include "fastllm-cuda-mtp.cuh"
#endif
namespace fastllm {
#ifdef USE_CUDA
    void FastllmCudaMergeMOEClearGraphUnsafeFallbackFlag();
    bool FastllmCudaMergeMOEUsedGraphUnsafeFallback();
#endif
    void Qwen4ExpModel::DecodeCudaGraphState::DestroyGraphs() {
#ifdef USE_CUDA
        const int previousDevice = FastllmCudaGetDevice();
        if (device >= 0) {
            FastllmCudaSetDevice(device);
        }
        for (auto &item : segments) {
            Segment &segment = item.second;
            if (segment.exec != nullptr) {
                FastllmCudaGraphExecDestroy(segment.exec);
                segment.exec = nullptr;
            }
            if (segment.graph != nullptr) {
                FastllmCudaGraphDestroy(segment.graph);
                segment.graph = nullptr;
            }
            if (!segment.reservedPointers.empty()) {
                FastllmCudaGraphMemoryPoolRelease(segment.reservedPointers);
                segment.reservedPointers.clear();
            }
            segment.captured = false;
        }
        if (previousDevice >= 0) {
            FastllmCudaSetDevice(previousDevice);
        }
#endif
        segments.clear();
    }

    void Qwen4ExpModel::DecodeCudaGraphState::Reset(int newDevice) {
        DestroyGraphs();
#ifdef USE_CUDA
        const int previousDevice = FastllmCudaGetDevice();
        if (device >= 0) {
            FastllmCudaSetDevice(device);
        }
#endif
        hiddenStates[0].FreeSpace();
        hiddenStates[1].FreeSpace();
        hybridBoundaries.clear();
        hybridDenseMode = false;
        attentionInput.FreeSpace();
        attentionInjection.FreeSpace();
        attentionOutput.FreeSpace();
        positionIds.FreeSpace();
        decodeMeta.FreeSpace();
        for (auto &item : denseIndices) {
            item.second.FreeSpace();
        }
        denseIndices.clear();
        logits.FreeSpace();
        for (auto &item : qsaScoreWorkspaces) {
            item.second.FreeSpace();
        }
        for (auto &item : qsaSelectedWorkspaces) {
            item.second.FreeSpace();
        }
        for (auto &item : qsaTailKeySnapshots) {
            item.second.FreeSpace();
        }
        for (auto &item : qsaTailPositionSnapshots) {
            item.second.FreeSpace();
        }
#ifdef USE_CUDA
        if (previousDevice >= 0) {
            FastllmCudaSetDevice(previousDevice);
        }
#endif
        linearCachePointers.clear();
        fullCacheSignature.clear();
        qsaScoreWorkspaces.clear();
        qsaSelectedWorkspaces.clear();
        qsaTailKeySnapshots.clear();
        qsaTailPositionSnapshots.clear();
        wholeGraphMode = false;
        graphStartLayer = -1;
        startBeforeAttention = false;
        device = newDevice;
    }

    Qwen4ExpModel::DecodeCudaGraphState::~DecodeCudaGraphState() {
        DestroyGraphs();
#ifdef USE_CUDA
        if (pinnedDecodeMeta != nullptr) {
            FastllmCudaHostFree(pinnedDecodeMeta);
            pinnedDecodeMeta = nullptr;
        }
#endif
    }

    bool Qwen4ExpModel::TryRunDecodeCudaGraphBackbone(
        ExecutionMode graphMode, int graphStartLayer, bool startBeforeAttention,
        const Data &hiddenStates, const Data &attentionOutput, const Data &attentionInjection,
        const Data &attentionMask, const Data &positionIds, bool qsaDeviceCompatibleMask,
        std::vector<std::pair<Data, Data>> &pastKeyValues, RequestState &requestState, Data &logits,
        Data *targetMultiHidden, TargetVerificationCapture *verificationCapture, int graphEndLayer,
        Data *stageOutput) {
#ifndef USE_CUDA
        return false;
#else
        const bool boundedStage = graphEndLayer >= 0;
        if (!boundedStage)
            graphEndLayer = this->block_cnt;
        if (graphStartLayer < 0 || graphStartLayer >= graphEndLayer ||
            graphEndLayer > this->block_cnt ||
            (boundedStage &&
             (!startBeforeAttention || stageOutput == nullptr ||
              (graphStartLayer < this->pleLayer && graphEndLayer > this->pleLayer)))) {
            return false;
        }
        int firstFullAttentionLayer = this->kvCacheId;
        if (boundedStage) {
            firstFullAttentionLayer = graphStartLayer;
            while (firstFullAttentionLayer < graphEndLayer &&
                   this->IsLinearAttentionLayer(firstFullAttentionLayer)) {
                ++firstFullAttentionLayer;
            }
        }
        const bool supportedStart =
            boundedStage || (!startBeforeAttention && graphStartLayer == firstFullAttentionLayer) ||
            (startBeforeAttention && graphStartLayer == this->pleLayer &&
             graphStartLayer < firstFullAttentionLayer &&
             this->IsLinearAttentionLayer(graphStartLayer));
        const bool mtpTargetGraph = verificationCapture != nullptr &&
                                    qwen4MtpDecodeEquivalentTarget &&
                                    hiddenStates.dims.size() == 3 && hiddenStates.dims[1] > 1 &&
                                    hiddenStates.dims[1] <= this->indexerCompressRatio;
        const auto policy = graphMode == ExecutionMode::Gpu
                                ? GpuDecodeGraphPolicy()
                                : HybridDecodeGraphPolicy(hiddenStates, mtpTargetGraph);
        const bool hybridDenseGraph = policy.hybridDense;
        // Host decisions and NUMA execution cannot be captured in the full
        // backbone graph, including multi-row MTP verification.
        if (!GetFastllmEnv().cudaGraph || (boundedStage && !hybridDenseGraph) ||
            (hybridDenseGraph && !startBeforeAttention) || !supportedStart ||
            hiddenStates.dims.size() != 3 || hiddenStates.dims[0] != 1 ||
            (hiddenStates.dims[1] != 1 && !mtpTargetGraph) ||
            hiddenStates.dataDevice != DataDevice::CUDA || hiddenStates.cudaData == nullptr ||
            (!startBeforeAttention && (attentionOutput.dataDevice != DataDevice::CUDA ||
                                       attentionOutput.cudaData == nullptr ||
                                       attentionInjection.dataDevice != DataDevice::CUDA ||
                                       attentionInjection.cudaData == nullptr)) ||
            GetKVCacheInCPU() || (!boundedStage && this->pleLayer >= firstFullAttentionLayer) ||
            std::getenv("FASTLLM_QWEN4_DUMP_DIR") != nullptr ||
            !Qwen4CudaOnlyDeviceMap(this->deviceMap) || !policy.eligible) {
            return false;
        }

        int device = hiddenStates.dataDeviceIds.empty() ? FastllmCudaGetDevice()
                                                        : hiddenStates.dataDeviceIds[0];
        if (device < 0 || (int)pastKeyValues.size() < this->block_cnt) {
            return false;
        }

        const int mutableLayerStart =
            startBeforeAttention ? graphStartLayer : firstFullAttentionLayer + 1;
        const int fullAttentionCacheStartLayer =
            startBeforeAttention ? firstFullAttentionLayer : firstFullAttentionLayer + 1;
        std::vector<void *> linearCachePointers;
        for (int layer = mutableLayerStart; layer < graphEndLayer; layer++) {
            if (!this->IsLinearAttentionLayer(layer)) {
                continue;
            }
            Data &pastConv = pastKeyValues[layer].first;
            Data &pastRecurrent = pastKeyValues[layer].second;
            if (pastConv.dims.empty() || pastRecurrent.dims.empty() ||
                pastConv.dataDevice != DataDevice::CUDA ||
                pastRecurrent.dataDevice != DataDevice::CUDA || pastConv.cudaData == nullptr ||
                pastRecurrent.cudaData == nullptr ||
                (!pastConv.dataDeviceIds.empty() && pastConv.dataDeviceIds[0] != device) ||
                (!pastRecurrent.dataDeviceIds.empty() &&
                 pastRecurrent.dataDeviceIds[0] != device)) {
                return false;
            }
            linearCachePointers.push_back(pastConv.cudaData);
            linearCachePointers.push_back(pastRecurrent.cudaData);
        }

        const Data *requestKey = &pastKeyValues[0].first;
        // Every nested capture in this backbone uses the FastLLM wrappers.
        // Match TP's scope so PP eager MoE boundaries do not repeatedly query
        // the CUDA runtime while no capture is active.
        struct ManagedCaptureScope {
            bool previous = FastllmCudaGraphSetManagedCaptureOnly(true);
            ~ManagedCaptureScope() { FastllmCudaGraphSetManagedCaptureOnly(previous); }
        } managedCaptureScope;
        DecodeCudaGraphState *graphState = nullptr;
        bool preserveVerifierAttention = false;
        {
            std::lock_guard<std::mutex> guard(this->stateMutex);
            std::unique_ptr<DecodeCudaGraphState> &entry = this->decodeCudaGraphStates[requestKey];
            if (entry == nullptr) {
                entry.reset(new DecodeCudaGraphState());
            }
            if (hybridDenseGraph && Qwen4MtpDraftsPerStep() > 0) {
                const Data &key = pastKeyValues[this->kvCacheId].first;
                entry->verifierEagerAttention |=
                    key.dims.size() == 3 && key.dims[1] < this->indexerBudget;
                preserveVerifierAttention = entry->verifierEagerAttention;
            }
            if (boundedStage) {
                auto &stage = entry->stages[{graphStartLayer, graphEndLayer}];
                if (!stage)
                    stage.reset(new DecodeCudaGraphState());
                graphState = stage.get();
            } else {
                graphState = entry.get();
            }
            if (mtpTargetGraph) {
                auto &variant = graphState->verifierLengths[hiddenStates.dims[1]];
                if (!variant)
                    variant.reset(new DecodeCudaGraphState());
                graphState = variant.get();
            }
        }
        std::lock_guard<std::mutex> graphGuard(graphState->mutex);
        TargetVerificationCapture *callerVerification = verificationCapture;
        if (mtpTargetGraph && hybridDenseGraph) {
            // Replaying one length must not depend on buffers overwritten by
            // another length or an eager fallback. Publish these captures to
            // the caller only after this stage has executed.
            verificationCapture = &graphState->verification;
            verificationCapture->linearConvInputs.resize(this->block_cnt);
            verificationCapture->linearAlphas.resize(this->block_cnt);
            verificationCapture->linearBetas.resize(this->block_cnt);
            verificationCapture->runtimeCheckpoint = callerVerification->runtimeCheckpoint;
        }

        // Full-attention layers used to sit between independently captured
        // graph segments because their QSA and K/V cache metadata changed on
        // the host every token.  Prepare fixed-capacity storage here; the
        // graph-safe kernels below consume the logical length from one stable
        // device scalar.  A cache growth changes this signature and causes a
        // single recapture, while ordinary decode tokens keep one graph.
        const int graphSequence = hiddenStates.dims[1];
        bool wholeGraphReady = (graphSequence == 1 || mtpTargetGraph) && qsaDeviceCompatibleMask &&
                               this->indexerCompressRatio > 0 && this->indexerBudget > 0 &&
                               this->indexerBudget % this->indexerCompressRatio == 0 &&
                               positionIds.dataDevice == DataDevice::CUDA &&
                               positionIds.dataType == DataType::FLOAT32 &&
                               positionIds.cudaData != nullptr;
        int decodePreviousLength = -1;
        int graphFullLayerCount = 0;
        std::vector<uint64_t> fullCacheSignature;
        auto appendSignature = [&](const void *pointer, int capacity, uint64_t stride) {
            fullCacheSignature.push_back((uint64_t)(uintptr_t)pointer);
            fullCacheSignature.push_back((uint64_t)capacity);
            fullCacheSignature.push_back(stride);
        };
        auto sameCudaDevice = [&](const Data &data) {
            return data.dataDevice == DataDevice::CUDA && data.cudaData != nullptr &&
                   (data.dataDeviceIds.empty() || data.dataDeviceIds[0] == device);
        };
        auto prepareFixedTail = [&](std::shared_ptr<Data> &cache,
                                    const std::vector<int> &logicalDims,
                                    const std::vector<int> &capacityDims) {
            if (cache == nullptr) {
                cache = std::make_shared<Data>(DataType::FLOAT32, logicalDims);
                cache->ToDevice(DataDevice::CUDA, std::vector<int>({device}));
            }
            const bool cudaMetadataMatches =
                cache->dataDevice == DataDevice::CUDA &&
                (cache->dataDeviceIds.empty() || cache->dataDeviceIds[0] == device);
            if (!cudaMetadataMatches || cache->dataType != DataType::FLOAT32 ||
                cache->dims != logicalDims) {
                return false;
            }
            bool needsGrowth =
                cache->cudaData == nullptr || cache->expansionDims.size() != capacityDims.size();
            if (!needsGrowth) {
                for (int axis = 0; axis < (int)capacityDims.size(); axis++) {
                    if (cache->expansionDims[axis] < capacityDims[axis]) {
                        needsGrowth = true;
                        break;
                    }
                }
            }
            if (needsGrowth) {
                if (logicalDims[0] == 0) {
                    // A restored empty tail may own compact storage. There
                    // are no rows to preserve; avoid the strided copy used
                    // when growing a nonempty cache.
                    cache->FreeSpace();
                }
                cache->Expansion(capacityDims);
            }
            return sameCudaDevice(*cache);
        };

        if (wholeGraphReady) {
            for (int layer = fullAttentionCacheStartLayer; layer < graphEndLayer; layer++) {
                if (this->IsLinearAttentionLayer(layer)) {
                    continue;
                }
                graphFullLayerCount++;
                Data &pastKey = pastKeyValues[layer].first;
                Data &pastValue = pastKeyValues[layer].second;
                if (!sameCudaDevice(pastKey) || !sameCudaDevice(pastValue) ||
                    pastKey.dims.size() != 3 || pastValue.dims != pastKey.dims ||
                    pastKey.dataType != pastValue.dataType || pastKey.dims[0] <= 0 ||
                    pastKey.dims[2] != this->head_dim || pastKey.strides.size() != 3 ||
                    pastValue.strides.size() != 3) {
                    wholeGraphReady = false;
                    break;
                }
                const int previousLength = pastKey.dims[1];
                if (decodePreviousLength < 0) {
                    decodePreviousLength = previousLength;
                } else if (decodePreviousLength != previousLength) {
                    wholeGraphReady = false;
                    break;
                }

                Data appendShape(pastKey.dataType,
                                 {pastKey.dims[0], graphSequence, pastKey.dims[2]});
                const bool geometricGrowth =
                    requestState.geometricCacheGrowthReadyLayers.count(layer) != 0;
                Qwen4EnsureAppendCapacity(pastKey, appendShape, 1, 128, kQwen4DenseCacheMaxGrowth,
                                          geometricGrowth);
                Qwen4EnsureAppendCapacity(pastValue, appendShape, 1, 128, kQwen4DenseCacheMaxGrowth,
                                          geometricGrowth);
                if (!sameCudaDevice(pastKey) || !sameCudaDevice(pastValue)) {
                    wholeGraphReady = false;
                    break;
                }

                const int ratio = this->indexerCompressRatio;
                const int oldTailCount = previousLength % ratio;
                const int oldBlocks = previousLength / ratio;
                std::shared_ptr<Data> &tailKeys = requestState.indexerTailKeyTensors[layer];
                std::shared_ptr<Data> &tailPositions =
                    requestState.indexerTailPositionTensors[layer];
                std::shared_ptr<Data> &blockCache = requestState.indexerBlockKeyTensors[layer];
                if (!prepareFixedTail(tailKeys, {oldTailCount, this->indexerHeadDim},
                                      {ratio, this->indexerHeadDim}) ||
                    !prepareFixedTail(tailPositions, {oldTailCount}, {ratio}) ||
                    blockCache == nullptr || !sameCudaDevice(*blockCache) ||
                    blockCache->dataType != DataType::FLOAT32 ||
                    blockCache->dims != std::vector<int>({oldBlocks, this->indexerHeadDim})) {
                    wholeGraphReady = false;
                    break;
                }

                const int kvCapacity =
                    std::min(Qwen4AxisCapacity(pastKey, 1), Qwen4AxisCapacity(pastValue, 1));
                const int blockCapacity = std::max(oldBlocks, (kvCapacity + ratio - 1) / ratio);
                if (Qwen4AxisCapacity(*blockCache, 0) < blockCapacity) {
                    blockCache->Expansion({blockCapacity, this->indexerHeadDim});
                }

                std::vector<float> &rawKeys = requestState.indexerRawKeys[layer];
                std::vector<float> &positions = requestState.indexerPositions[layer];
                std::shared_ptr<QsaHostMirrorTransfer> &hostTransfer =
                    GetQsaHostMirror(requestState, layer);
                hostTransfer->Prepare(rawKeys, positions, previousLength - oldTailCount,
                                      this->indexerHeadDim);

                if (mtpTargetGraph) {
                    Data &rawKeyCapture = verificationCapture->qsaRawKeys[layer];
                    if (!Qwen4PrepareDecodeGraphWorkspace(rawKeyCapture, DataType::FLOAT32,
                                                          {graphSequence, this->indexerHeadDim},
                                                          device)) {
                        wholeGraphReady = false;
                        break;
                    }
                    // Reservation is not a captured QSA append. Keep the
                    // logical history empty until the whole graph is chosen;
                    // segmented/eager attention appends its own rows.
                    rawKeyCapture.expansionDims = rawKeyCapture.dims;
                    rawKeyCapture.Resize({0, this->indexerHeadDim});
                    appendSignature(rawKeyCapture.cudaData, graphSequence,
                                    rawKeyCapture.strides[0]);
                }

                appendSignature(pastKey.cudaData, Qwen4AxisCapacity(pastKey, 1),
                                pastKey.strides[0]);
                appendSignature(pastValue.cudaData, Qwen4AxisCapacity(pastValue, 1),
                                pastValue.strides[0]);
                if (Qwen4IsPagedKV(pastKey)) {
                    appendSignature(pastKey.pagedKVCacheData->cudaData, pastKey.pageLen, 0);
                    appendSignature(pastValue.pagedKVCacheData->cudaData, pastValue.pageLen, 0);
                }
                appendSignature(tailKeys->cudaData, Qwen4AxisCapacity(*tailKeys, 0),
                                tailKeys->strides[0]);
                appendSignature(tailPositions->cudaData, Qwen4AxisCapacity(*tailPositions, 0),
                                tailPositions->strides[0]);
                appendSignature(blockCache->cudaData, Qwen4AxisCapacity(*blockCache, 0),
                                blockCache->strides[0]);
            }
        }
        const bool linearOnlyStage =
            boundedStage && graphFullLayerCount == 0 && firstFullAttentionLayer == graphEndLayer;
        const bool wholeDenseGraph = (threadTpRank >= 0 || boundedStage) && !linearOnlyStage &&
                                     (graphSequence == 1 || mtpTargetGraph) &&
                                     decodePreviousLength < this->indexerBudget;
        const bool hybridEagerAttention =
            hybridDenseGraph && (wholeDenseGraph || preserveVerifierAttention);
        wholeGraphReady =
            wholeGraphReady &&
            (linearOnlyStage || (graphFullLayerCount > 0 &&
                                 (decodePreviousLength >= this->indexerBudget || wholeDenseGraph)));
        if (mtpTargetGraph && attentionMask.dims.empty() && !wholeGraphReady) {
            // A verifier may defer construction of its dense causal mask
            // when the whole graph can encode causality from decodeMeta.
            // Segmented/eager attention still consumes an explicit mask, so
            // return before mutating caches and let the caller materialize it.
            return false;
        }

        // Keep a request on its original graph topology. Switching a live
        // request from segmented graphs to the whole-backbone graph changes
        // the full-attention execution path at the sparse threshold and can
        // perturb a close boundary logit. Short prompts that cross the
        // threshold during generation therefore retain the established
        // segmented path; requests that start at or above it use the whole
        // graph from their first decode token.
        if (threadTpRank < 0 && !boundedStage && wholeGraphReady && !graphState->segments.empty() &&
            !graphState->wholeGraphMode) {
            wholeGraphReady = false;
        }

        // Hybrid graphs must use dynamic full-attention metadata; the old
        // segmented body includes MoE and cannot capture host experts.
        if (hybridDenseGraph && !ThreadTpAllTrue(wholeGraphReady)) {
            return false;
        }

        const bool workspaceShapeChanged =
            graphState->hiddenStates[0].cudaData != nullptr &&
            (graphState->hiddenStates[0].dataType != hiddenStates.dataType ||
             graphState->hiddenStates[0].dims != hiddenStates.dims);
        if (mtpTargetGraph && verificationCapture->runtimeCheckpoint) {
            for (int layer = mutableLayerStart; layer < graphEndLayer; ++layer) {
                if (this->IsLinearAttentionLayer(layer)) {
                    appendSignature(
                        verificationCapture->runtimeCheckpoint->linearSecond[layer].cudaData, 0, 0);
                }
            }
        }
        const bool resetGraph =
            graphState->device != device || workspaceShapeChanged ||
            graphState->hybridDenseMode != hybridDenseGraph ||
            graphState->graphStartLayer != graphStartLayer ||
            graphState->startBeforeAttention != startBeforeAttention ||
            (!graphState->linearCachePointers.empty() &&
             graphState->linearCachePointers != linearCachePointers) ||
            (!graphState->segments.empty() && graphState->wholeGraphMode != wholeGraphReady) ||
            (wholeGraphReady && !graphState->fullCacheSignature.empty() &&
             graphState->fullCacheSignature != fullCacheSignature);
        // A reused collective graph must be invalidated on all ranks together,
        // even if only one rank's allocator changed an address or capacity.
        if (!ThreadTpAllTrue(!resetGraph)) {
            graphState->Reset(device);
        }
        graphState->device = device;
        graphState->graphStartLayer = graphStartLayer;
        graphState->startBeforeAttention = startBeforeAttention;
        graphState->linearCachePointers = linearCachePointers;
        graphState->wholeGraphMode = wholeGraphReady;
        graphState->hybridDenseMode = hybridDenseGraph;
        graphState->fullCacheSignature = fullCacheSignature;
        if (!Qwen4PrepareDecodeGraphTensor(graphState->hiddenStates[0], hiddenStates, device) ||
            (!startBeforeAttention &&
             (!Qwen4PrepareDecodeGraphTensor(graphState->attentionOutput, attentionOutput,
                                             device) ||
              !Qwen4PrepareDecodeGraphTensor(graphState->attentionInjection, attentionInjection,
                                             device))) ||
            (wholeGraphReady &&
             !Qwen4PrepareDecodeGraphTensor(graphState->positionIds, positionIds, device))) {
            return false;
        }
        if (mtpTargetGraph) {
            // PLE may finish its CPU-offloaded ngram result on a transfer
            // stream.  The first verifier segment consumes the staged
            // residual immediately, so retire that producer and its staging
            // copy before entering/replaying the capture domain.
            FastllmCudaSyncCurrentThreadStream();
        }
        if (wholeGraphReady) {
            std::vector<void *> tailSnapshotDestinations;
            std::vector<const void *> tailSnapshotSources;
            std::vector<size_t> tailSnapshotSizes;
            if (graphState->decodeMeta.cudaData == nullptr) {
                graphState->decodeMeta.dataType = DataType::INT32;
                graphState->decodeMeta.Resize({1});
                graphState->decodeMeta.dataDevice = DataDevice::CUDA;
                graphState->decodeMeta.dataDeviceIds = {device};
                graphState->decodeMeta.Allocate(false);
            }
            if (graphState->pinnedDecodeMeta == nullptr) {
                graphState->pinnedDecodeMeta =
                    reinterpret_cast<int32_t *>(FastllmCudaHostMalloc(sizeof(int32_t)));
            }
            if (graphState->decodeMeta.cudaData == nullptr ||
                graphState->pinnedDecodeMeta == nullptr) {
                return false;
            }
            graphState->pinnedDecodeMeta[0] = decodePreviousLength;
            if (!FastllmCudaCopyFromPinnedHostToDeviceAsyncCurrentThread(
                    graphState->decodeMeta.cudaData, graphState->pinnedDecodeMeta,
                    sizeof(int32_t))) {
                return false;
            }

            const int selectedBlocks =
                wholeDenseGraph ? 1 : this->indexerBudget / this->indexerCompressRatio;
            if (wholeDenseGraph) {
                const int width =
                    requestState.denseGraphWidth > 0
                        ? requestState.denseGraphWidth
                        : Qwen4AxisCapacity(pastKeyValues[fullAttentionCacheStartLayer].first, 1);
                Data &indicesData = graphState->denseIndices[width];
                if (indicesData.cudaData == nullptr) {
                    if (!Qwen4PrepareDecodeGraphWorkspace(indicesData, DataType::INT32,
                                                          {graphSequence, width}, device))
                        return false;
                    std::vector<int32_t> indices((size_t)graphSequence * width);
                    for (size_t i = 0; i < indices.size(); ++i)
                        indices[i] = i % width;
                    FastllmCudaCopyFromHostToDevice(indicesData.cudaData, indices.data(),
                                                    indices.size() * sizeof(int32_t));
                }
            }
            for (int layer = fullAttentionCacheStartLayer; layer < graphEndLayer; layer++) {
                if (this->IsLinearAttentionLayer(layer)) {
                    continue;
                }
                auto blockCacheIt = requestState.indexerBlockKeyTensors.find(layer);
                if (blockCacheIt == requestState.indexerBlockKeyTensors.end() ||
                    blockCacheIt->second == nullptr) {
                    return false;
                }
                const int blockCapacity = Qwen4AxisCapacity(*blockCacheIt->second, 0);
                if (blockCapacity < selectedBlocks ||
                    !Qwen4PrepareDecodeGraphWorkspace(graphState->qsaScoreWorkspaces[layer],
                                                      DataType::FLOAT32,
                                                      {graphSequence, blockCapacity}, device) ||
                    !Qwen4PrepareDecodeGraphWorkspace(graphState->qsaSelectedWorkspaces[layer],
                                                      DataType::INT32,
                                                      {graphSequence, selectedBlocks}, device) ||
                    (mtpTargetGraph &&
                     (!Qwen4PrepareDecodeGraphWorkspace(
                          graphState->qsaTailKeySnapshots[layer], DataType::FLOAT32,
                          {this->indexerCompressRatio, this->indexerHeadDim}, device) ||
                      !Qwen4PrepareDecodeGraphWorkspace(graphState->qsaTailPositionSnapshots[layer],
                                                        DataType::FLOAT32,
                                                        {this->indexerCompressRatio}, device)))) {
                    return false;
                }
                const int oldTailCount = decodePreviousLength % this->indexerCompressRatio;
                if (mtpTargetGraph && oldTailCount > 0) {
                    const Data &tailKeys = *requestState.indexerTailKeyTensors[layer];
                    const Data &tailPositions = *requestState.indexerTailPositionTensors[layer];
                    tailSnapshotDestinations.push_back(
                        graphState->qsaTailKeySnapshots[layer].cudaData);
                    tailSnapshotSources.push_back(tailKeys.cudaData);
                    tailSnapshotSizes.push_back((size_t)oldTailCount * this->indexerHeadDim *
                                                sizeof(float));
                    tailSnapshotDestinations.push_back(
                        graphState->qsaTailPositionSnapshots[layer].cudaData);
                    tailSnapshotSources.push_back(tailPositions.cudaData);
                    tailSnapshotSizes.push_back((size_t)oldTailCount * sizeof(float));
                }
            }
            if (mtpTargetGraph && !tailSnapshotDestinations.empty() &&
                !FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(
                    tailSnapshotDestinations.data(), tailSnapshotSources.data(),
                    tailSnapshotSizes.data(), (int)tailSnapshotDestinations.size())) {
                return false;
            }
        }

        auto destroySegment = [&](DecodeCudaGraphState::Segment &segment) {
            FastllmCudaSetDevice(device);
            if (segment.exec != nullptr) {
                FastllmCudaGraphExecDestroy(segment.exec);
                segment.exec = nullptr;
            }
            if (segment.graph != nullptr) {
                FastllmCudaGraphDestroy(segment.graph);
                segment.graph = nullptr;
            }
            if (!segment.reservedPointers.empty()) {
                FastllmCudaGraphMemoryPoolRelease(segment.reservedPointers);
                segment.reservedPointers.clear();
            }
            segment.captured = false;
        };

        const int wholeGraphRemainder =
            wholeGraphReady ? (decodePreviousLength + graphSequence) % this->indexerCompressRatio
                            : 0;
        const int wholeGraphSparseWidth =
            wholeDenseGraph && wholeGraphReady
                ? (requestState.denseGraphWidth > 0
                       ? requestState.denseGraphWidth
                       : Qwen4AxisCapacity(pastKeyValues[fullAttentionCacheStartLayer].first, 1))
                : this->indexerBudget +
                      (mtpTargetGraph ? this->indexerCompressRatio - 1 : wholeGraphRemainder);

        auto runFullAttentionGraph = [&](int layer, const Data &input, Data &pastKey,
                                         Data &pastValue, Data &output) -> bool {
            const std::string attention =
                languagePrefix + "layers." + std::to_string(layer) + ".self_attn.";
            const std::string indexer = attention + "indexer.";
            constexpr int batch = 1;
            const int sequence = graphSequence;
            const int ratio = this->indexerCompressRatio;
            const int32_t *decodeMeta =
                reinterpret_cast<const int32_t *>(graphState->decodeMeta.cudaData);

            auto tailKeysIt = requestState.indexerTailKeyTensors.find(layer);
            auto tailPositionsIt = requestState.indexerTailPositionTensors.find(layer);
            auto blockCacheIt = requestState.indexerBlockKeyTensors.find(layer);
            auto scoreWorkspaceIt = graphState->qsaScoreWorkspaces.find(layer);
            auto selectedWorkspaceIt = graphState->qsaSelectedWorkspaces.find(layer);
            if (decodeMeta == nullptr || tailKeysIt == requestState.indexerTailKeyTensors.end() ||
                tailPositionsIt == requestState.indexerTailPositionTensors.end() ||
                blockCacheIt == requestState.indexerBlockKeyTensors.end() ||
                scoreWorkspaceIt == graphState->qsaScoreWorkspaces.end() ||
                selectedWorkspaceIt == graphState->qsaSelectedWorkspaces.end() ||
                tailKeysIt->second == nullptr || tailPositionsIt->second == nullptr ||
                blockCacheIt->second == nullptr) {
                return false;
            }
            Data &tailKeys = *tailKeysIt->second;
            Data &tailPositions = *tailPositionsIt->second;
            Data &blockCache = *blockCacheIt->second;
            Data &scoreWorkspace = scoreWorkspaceIt->second;
            Data &selectedWorkspace = selectedWorkspaceIt->second;

            Data typedInput, projected, indexQuery, currentRawKeys;
            ToDataType(input, typedInput, this->dataType);
            const int qsaParallelRegion = (1 << 20) + layer;
            const bool parallelQsa = mtpTargetGraph;
            if (parallelQsa) {
                FastllmCudaGraphMarkParallelFork(qsaParallelRegion);
            }
            Linear(typedInput, this->weight[indexer + "index_qk_proj.weight"], Data(), projected);
            const int queryColumns = this->indexerHeads * this->indexerHeadDim;
            Split(projected, -1, 0, queryColumns, indexQuery);
            Split(projected, -1, queryColumns,
                  queryColumns + this->indexerKvHeads * this->indexerHeadDim, currentRawKeys);
            indexQuery.Reshape({batch, sequence, this->indexerHeads, this->indexerHeadDim});
            currentRawKeys.Reshape({batch, sequence, this->indexerHeadDim});
            RMSNorm(indexQuery, this->weight[indexer + "q_layernorm.weight"], this->rms_norm_eps,
                    indexQuery);
            RopeEncoding(indexQuery, graphState->positionIds, this->rotary_dim, this->rope_base,
                         1.0f, true);

            Data currentKeysFloat;
            ToDataType(currentRawKeys, currentKeysFloat, DataType::FLOAT32);
            currentKeysFloat.Reshape({sequence, this->indexerHeadDim});
            if (verificationCapture != nullptr) {
                Data &rawKeyCapture = verificationCapture->qsaRawKeys[layer];
                if (rawKeyCapture.dataDevice != DataDevice::CUDA ||
                    rawKeyCapture.cudaData == nullptr ||
                    rawKeyCapture.dataType != DataType::FLOAT32 ||
                    rawKeyCapture.dims != currentKeysFloat.dims ||
                    !FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(
                        rawKeyCapture.cudaData, currentKeysFloat.cudaData,
                        currentKeysFloat.GetBytes())) {
                    return false;
                }
            }

            Data tailKeysFixed, tailPositionsFixed;
            Qwen4BorrowCudaStorage(tailKeysFixed, tailKeys);
            Qwen4BorrowCudaStorage(tailPositionsFixed, tailPositions);
            tailKeysFixed.Resize({ratio, this->indexerHeadDim});
            tailPositionsFixed.Resize({ratio});

            // Append each graph row in causal order. The graph topology is
            // fixed; the device-side base length selects the tail slot and
            // conditionally commits completed groups. Keep all per-row views
            // and work tensors alive until the QSA/attention join below.
            const bool fusedCompress = sequence > 0 && sequence <= ratio && ratio == 4 &&
                                       this->indexerHeadDim == 128 && this->rotary_dim == 64;
            if (fusedCompress) {
                const bool compressed = FastllmCudaQwen4QSAAppendCompress4Graph(
                    currentKeysFloat, graphState->positionIds,
                    this->weight[indexer + "k_layernorm.weight"], this->rope_base, decodeMeta,
                    tailKeys, tailPositions, blockCache, this->rms_norm_eps);
                if (!compressed) {
                    return false;
                }
            } else {
                std::vector<Data> tokenKeys(sequence);
                std::vector<Data> tokenPositions(sequence);
                std::vector<Data> pooledKeys(sequence);
                std::vector<Data> averagedKeys(sequence);
                std::vector<Data> normalizedKeys(sequence);
                std::vector<Data> firstPositions(sequence);
                std::vector<Data> memberKeys(sequence * std::max(0, ratio - 1));
                for (int token = 0; token < sequence; token++) {
                    Data &tokenKey = tokenKeys[token];
                    Data &tokenPosition = tokenPositions[token];
                    Split(currentKeysFloat, 0, token, token + 1, tokenKey);
                    Split(graphState->positionIds, 1, token, token + 1, tokenPosition);
                    tokenKey.Reshape({1, this->indexerHeadDim});
                    tokenPosition.Reshape({1});
                    if (!FastllmCudaQwen4QSAAppendGraph(tokenKey, tokenPosition, decodeMeta, token,
                                                        ratio, tailKeys, tailPositions)) {
                        return false;
                    }
                    // Preserve the established float32 pooling/RMSNorm/RoPE
                    // arithmetic.
                    Data &pooled = pooledKeys[token];
                    Split(tailKeysFixed, 0, 0, 1, pooled);
                    pooled.Reshape({1, this->indexerHeadDim});
                    for (int member = 1; member < ratio; member++) {
                        Data &memberKey = memberKeys[token * (ratio - 1) + member - 1];
                        Split(tailKeysFixed, 0, member, member + 1, memberKey);
                        memberKey.Reshape({1, this->indexerHeadDim});
                        AddTo(pooled, memberKey);
                    }
                    Data &averaged = averagedKeys[token];
                    Data &normalized = normalizedKeys[token];
                    Mul(pooled, 1.0f / (float)ratio, averaged);
                    RMSNorm(averaged, this->weight[indexer + "k_layernorm.weight"],
                            this->rms_norm_eps, normalized);
                    Data &firstPosition = firstPositions[token];
                    Split(tailPositionsFixed, 0, 0, 1, firstPosition);
                    firstPosition.Reshape({1, 1});
                    normalized.Reshape({1, 1, 1, this->indexerHeadDim});
                    RopeEncoding(normalized, firstPosition, this->rotary_dim, this->rope_base, 1.0f,
                                 true);
                    normalized.Reshape({1, this->indexerHeadDim});
                    if (!FastllmCudaQwen4QSACommitGraph(normalized, decodeMeta, token, ratio,
                                                        blockCache)) {
                        return false;
                    }
                }
            }

            Data qsaIndices(DataType::INT32, {sequence, wholeGraphSparseWidth});
            if (wholeDenseGraph) {
                // The graph uses fixed-capacity storage and masks positions
                // beyond the device-side logical length. Preserve dense key
                // order; padding must never become a visible zero key.
                Qwen4BorrowCudaStorage(qsaIndices,
                                       graphState->denseIndices.at(wholeGraphSparseWidth));
            } else {
                qsaIndices.ToDevice(DataDevice::CUDA, std::vector<int>({device}));
                qsaIndices.Allocate(false);
            }
            if (!wholeDenseGraph &&
                !FastllmCudaQwen4QSASelectGraph(indexQuery, blockCache, decodeMeta, scoreWorkspace,
                                                selectedWorkspace, qsaIndices, this->indexerHeads,
                                                this->indexerHeadDim, this->indexerBudget, ratio)) {
                return false;
            }
            if (parallelQsa) {
                FastllmCudaGraphMarkParallelFirstDone(qsaParallelRegion);
                FastllmCudaGraphMarkParallelSecondBegin(qsaParallelRegion);
            }

            Data qGate, query, key, value, gate;
            const Data *projectionWeights[] = {&this->weight[attention + "q_proj.weight"],
                                               &this->weight[attention + "k_proj.weight"],
                                               &this->weight[attention + "v_proj.weight"]};
            Data *projectionOutputs[] = {&qGate, &key, &value};
            const bool fusedProjections =
                (sequence == 1 || qwen4MtpDecodeEquivalentTarget) &&
                Qwen4TryExactCudaMultiLinear(typedInput, projectionWeights, projectionOutputs, 3);
            if (!fusedProjections) {
                Linear(typedInput, this->weight[attention + "q_proj.weight"], Data(), qGate);
                Linear(typedInput, this->weight[attention + "k_proj.weight"], Data(), key);
                Linear(typedInput, this->weight[attention + "v_proj.weight"], Data(), value);
            }
            qGate.Reshape({batch, sequence, -1, this->head_dim * 2});
            Split(qGate, -1, 0, this->head_dim, query);
            Split(qGate, -1, this->head_dim, this->head_dim * 2, gate);
            gate.Reshape({batch, sequence, -1});
            key.Reshape({batch, sequence, -1, this->head_dim});
            value.Reshape({batch, sequence, -1, this->head_dim});
            RMSNorm(query, this->weight[attention + "q_norm.weight"], this->rms_norm_eps, query);
            RMSNorm(key, this->weight[attention + "k_norm.weight"], this->rms_norm_eps, key);
            RopeEncoding(query, graphState->positionIds, this->rotary_dim, this->rope_base, 1.0f,
                         true);
            RopeEncoding(key, graphState->positionIds, this->rotary_dim, this->rope_base, 1.0f,
                         true);
            PermuteSelf(query, {0, 2, 1, 3});
            PermuteSelf(key, {0, 2, 1, 3});
            PermuteSelf(value, {0, 2, 1, 3});
            query.Reshape({-1, sequence, this->head_dim});
            key.Reshape({-1, sequence, this->head_dim});
            value.Reshape({-1, sequence, this->head_dim});
            if (!FastllmCudaQwen4KVAppendGraph(key, value, decodeMeta, pastKey, pastValue)) {
                return false;
            }
            if (parallelQsa) {
                FastllmCudaGraphMarkParallelJoin(qsaParallelRegion);
            }

            Data context;
            const int attentionGroup = query.dims[0] / pastKey.dims[0];
            const float attentionScale = 1.0f / std::sqrt((float)this->head_dim);
            const bool sparseDecode =
                sequence == 1 && !wholeDenseGraph &&
                FastllmCudaQwen4SparseDecode(query, pastKey, pastValue, qsaIndices, context,
                                             attentionGroup, attentionScale, decodeMeta);
            if (!sparseDecode && sequence == 1 && !wholeDenseGraph) {
                Data compactKey(key.dataType,
                                {pastKey.dims[0], wholeGraphSparseWidth, this->head_dim});
                Data compactValue(value.dataType,
                                  {pastValue.dims[0], wholeGraphSparseWidth, this->head_dim});
                compactKey.ToDevice(DataDevice::CUDA, std::vector<int>({device}));
                compactValue.ToDevice(DataDevice::CUDA, std::vector<int>({device}));
                compactKey.Allocate(false);
                compactValue.Allocate(false);
                if (!FastllmCudaQwen4GatherKVGraph(pastKey, pastValue, qsaIndices, decodeMeta,
                                                   compactKey, compactValue)) {
                    return false;
                }
                Attention(query, compactKey, compactValue, Data(), context, attentionGroup,
                          attentionScale, 1);
            } else if (!sparseDecode) {
                const int queryHeads = query.dims[0];
                const int keyHeads = pastKey.dims[0];
                Data packedQuery(query.dataType, {sequence * queryHeads, 1, this->head_dim});
                Data compactKey(key.dataType,
                                {sequence * keyHeads, wholeGraphSparseWidth, this->head_dim});
                Data compactValue(value.dataType,
                                  {sequence * keyHeads, wholeGraphSparseWidth, this->head_dim});
                Data paddingMask(query.dataType, {sequence, 1, wholeGraphSparseWidth});
                Data packedOutput(query.dataType, {sequence * queryHeads, 1, this->head_dim});
                context.dataType = query.dataType;
                context.UpdateUnitSize();
                context.Resize({queryHeads, sequence, this->head_dim});
                for (Data *workspace :
                     std::vector<Data *>({&packedQuery, &compactKey, &compactValue, &paddingMask,
                                          &packedOutput, &context})) {
                    workspace->ToDevice(DataDevice::CUDA, std::vector<int>({device}), false);
                    workspace->Allocate(false);
                }
                if (!FastllmCudaQwen4PrepareSparseBatchGraph(
                        query, pastKey, pastValue, qsaIndices, decodeMeta, sequence, packedQuery,
                        compactKey, compactValue, paddingMask, 0, sequence)) {
                    return false;
                }
                Attention(packedQuery, compactKey, compactValue, paddingMask, packedOutput,
                          attentionGroup, attentionScale, 1);
                if (!FastllmCudaQwen4UnpackSparseBatch(packedOutput, context, 0, sequence)) {
                    return false;
                }
            }
            Data gatedContext;
            const bool fusedOutput = FastllmCudaQwen4AttentionOutput(context, gate, gatedContext);
            if (!fusedOutput) {
                PermuteSelf(context, {1, 0, 2});
                context.Reshape({sequence, batch, -1});
                PermuteSelf(context, {1, 0, 2});
                SigmoidMulTo(context, gate);
            }
            Linear(fusedOutput ? gatedContext : context, this->weight[attention + "o_proj.weight"],
                   Data(), output);
            ThreadTpAllReduce(output);
            return true;
        };

        int capturesAttempted = 0;
        const char *captureBudgetEnv = std::getenv("FASTLLM_CUDA_GRAPH_CAPTURE_BUDGET");
        // TP ranks already commit each capture together below. Limiting TP
        // to one segment per forward leaves rare verifier lengths only
        // partially captured when the first reusable request is frozen.
        // Honor the same startup budget on every rank; serving requests
        // still take the frozen/eager branch before attempting any capture.
        const bool prepareReusableTp = threadTpRank >= 0 && requestState.reusableGraphRequest;
        const int captureBudget = threadTpRank >= 0 && !prepareReusableTp ? 1 :
            (captureBudgetEnv ? std::max(1, std::min(128, std::atoi(captureBudgetEnv)))
                              : (prepareReusableTp ? 8 : 1));
        auto runCapturedSegment = [&](int segmentKey, const auto &runBody) {
            DecodeCudaGraphState::Segment &segment = graphState->segments[segmentKey];
            FastllmCudaSetDevice(device);

            if (segment.captured) {
                const bool launched = FastllmCudaGraphLaunch(segment.exec);
                if (threadTpRank >= 0) {
                    // Replay failure already aborts the process; successful
                    // launches need no host vote. Capture/commit decisions
                    // below still rendezvous before choosing eager fallback.
                    AssertInFastLLM(launched, "Qwen4 TP decode CUDA graph replay failed.");
                }
                if (launched)
                    return;
                std::fprintf(stderr,
                             "[Fastllm] Qwen4 decode CUDA graph replay failed "
                             "on GPU %d segment %d: %s; using eager fallback.\n",
                             device, segmentKey, FastllmCudaGraphLastError());
                destroySegment(segment);
                segment.disabled = true;
                FastllmCudaClearThreadError();
                FastllmCudaClearGraphError();
                runBody();
                return;
            }

            // Serving requests only replay the shapes captured by the first
            // request. An unseen or invalidated shape stays eager thereafter.
            if (requestState.decodeGraphsFrozen) {
                runBody();
                return;
            }
            FastllmCudaMergeMOEClearGraphUnsafeFallbackFlag();
            if (!segment.warmed || segment.disabled) {
                segment.allocationRequests.clear();
                FastllmCudaAllocationRecorder record(
                    (threadTpRank < 0 || prepareReusableTp) ? &segment.allocationRequests : nullptr);
                runBody();
                segment.warmed = true;
                if (FastllmCudaMergeMOEUsedGraphUnsafeFallback()) {
                    segment.disabled = true;
                }
                return;
            }
            if (capturesAttempted >= captureBudget) {
                runBody();
                return;
            }
            ++capturesAttempted;

            void *capturedGraph = nullptr;
            void *capturedExec = nullptr;
            bool poolActive = false;
            bool captureActive = false;
            bool recaptureWithoutParallelMarkers = false;
            const bool enableParallelRegions =
                threadTpRank < 0 && !segment.parallelRegionsUnavailable;
            bool captureOk = FastllmCudaGraphPrepareCaptureDevice();
            captureOk = ThreadTpAllTrue(captureOk);
            if (captureOk) {
                // The capture pool may reuse deferred frees without querying
                // their events. Retire every producer on this device before
                // entering capture; graph replay needs no such synchronization.
                ForceDeviceSync();
                captureOk = ThreadTpAllTrue(!FastllmCudaGetThreadError());
            }
            if (captureOk && (threadTpRank < 0 || prepareReusableTp)) {
                // Earlier segments pin their temporary allocations. Reserve
                // each rank's recorded demand before opening the shared pool,
                // then rendezvous so no peer allocates during capture.
                captureOk = ThreadTpAllTrue(
                    FastllmCudaReserveGraphWorkspace(segment.allocationRequests));
            }
            if (captureOk) {
                if (threadTpRank <= 0) {
                    poolActive = FastllmCudaGraphMemoryPoolBegin();
                    captureOk = poolActive;
                }
                captureOk = ThreadTpAllTrue(captureOk);
            }
            if (captureOk) {
                FastllmCudaClearThreadError();
                FastllmCudaClearGraphError();
                captureActive = FastllmCudaGraphBeginCapture();
                captureOk = captureActive;
            }
            captureOk = ThreadTpAllTrue(captureOk);

            bool bodyThrew = false;
            if (captureOk) {
                FastllmCudaMergeMOEClearGraphUnsafeFallbackFlag();
                const bool previousParallelMarkers =
                    FastllmCudaGraphSetParallelMarkersEnabled(enableParallelRegions);
                try {
                    runBody();
                } catch (...) {
                    bodyThrew = true;
                }
                FastllmCudaGraphSetParallelMarkersEnabled(previousParallelMarkers);
                const bool unsafeMoe = FastllmCudaMergeMOEUsedGraphUnsafeFallback();
                const bool bodyFailed = bodyThrew || unsafeMoe || FastllmCudaGetThreadError() ||
                                        FastllmCudaGetGraphError() ||
                                        FastllmCudaGraphCaptureInvalidated();
                bool endOk = FastllmCudaGraphEndCapture(&capturedGraph);
                captureActive = false;
                captureOk = !bodyFailed && endOk && capturedGraph != nullptr;
                if (unsafeMoe) {
                    segment.disabled = true;
                }
            }
            if (captureActive) {
                void *discardedGraph = nullptr;
                FastllmCudaGraphEndCapture(&discardedGraph);
                if (discardedGraph != nullptr) {
                    FastllmCudaGraphDestroy(discardedGraph);
                }
            }

            // The allocator capture pool is process-wide. Rank 0 owns one
            // transaction containing all device captures and their pointers.
            captureOk = ThreadTpAllTrue(captureOk);
            if (captureOk && threadTpRank <= 0) {
                captureOk = FastllmCudaGraphMemoryPoolEnd(segment.reservedPointers);
                poolActive = false;
            }
            captureOk = ThreadTpAllTrue(captureOk);
            if (poolActive) {
                FastllmCudaGraphMemoryPoolAbort();
            }
            ThreadTpAllTrue(true);
            if (captureOk && enableParallelRegions) {
                const int parallelRegions = FastllmCudaGraphOptimizeParallelRegions(capturedGraph);
                if (parallelRegions == FASTLLM_CUDA_GRAPH_RECAPTURE_WITHOUT_PARALLEL_MARKERS) {
                    // Some CUDA graph node types forbid marker removal. Keep
                    // the segment usable: discard this graph and capture the
                    // original serial topology without markers next token.
                    segment.parallelRegionsUnavailable = true;
                    recaptureWithoutParallelMarkers = true;
                    captureOk = false;
                } else if (parallelRegions < 0) {
                    captureOk = false;
                }
            }
            if (captureOk) {
                captureOk = FastllmCudaGraphInstantiate(capturedGraph, &capturedExec) &&
                            capturedExec != nullptr;
            }

            // A collective graph is committed by every rank together. One
            // failed capture must not leave peers replaying while it falls
            // back to the eager host submission rendezvous.
            captureOk = ThreadTpAllTrue(captureOk);
            if (!captureOk) {
                if (capturedExec != nullptr) {
                    FastllmCudaGraphExecDestroy(capturedExec);
                }
                if (capturedGraph != nullptr) {
                    FastllmCudaGraphDestroy(capturedGraph);
                }
                if (!segment.reservedPointers.empty()) {
                    FastllmCudaGraphMemoryPoolRelease(segment.reservedPointers);
                    segment.reservedPointers.clear();
                }
                if (!recaptureWithoutParallelMarkers) {
                    segment.captureFailures++;
                }
                if (!recaptureWithoutParallelMarkers && segment.captureFailures >= 3) {
                    segment.disabled = true;
                    std::fprintf(stderr,
                                 "[Fastllm] Qwen4 decode CUDA graph disabled "
                                 "on GPU %d segment %d after capture failure: %s\n",
                                 device, segmentKey, FastllmCudaGraphLastError());
                }
                FastllmCudaClearThreadError();
                FastllmCudaClearGraphError();
                FastllmCudaMergeMOEClearGraphUnsafeFallbackFlag();
                runBody();
                if (FastllmCudaMergeMOEUsedGraphUnsafeFallback()) {
                    segment.disabled = true;
                }
                return;
            }

            segment.graph = capturedGraph;
            segment.exec = capturedExec;
            segment.captured = true;
            segment.captureFailures = 0;
            const bool launched = FastllmCudaGraphLaunch(segment.exec);
            if (threadTpRank >= 0) {
                AssertInFastLLM(ThreadTpAllTrue(launched),
                                "Qwen4 TP decode CUDA graph first launch failed.");
            }
            if (!launched) {
                std::fprintf(stderr,
                             "[Fastllm] Qwen4 decode CUDA graph first launch "
                             "failed on GPU %d segment %d: %s; using eager fallback.\n",
                             device, segmentKey, FastllmCudaGraphLastError());
                destroySegment(segment);
                segment.disabled = true;
                FastllmCudaClearThreadError();
                FastllmCudaClearGraphError();
                runBody();
                return;
            }
        };

        DecodeGraphExecution execution{*this,
                                       graphState,
                                       graphStartLayer,
                                       graphEndLayer,
                                       graphSequence,
                                       device,
                                       decodePreviousLength,
                                       wholeGraphRemainder,
                                       wholeGraphSparseWidth,
                                       startBeforeAttention,
                                       mtpTargetGraph,
                                       wholeDenseGraph,
                                       hybridDenseGraph,
                                       hybridEagerAttention,
                                       linearOnlyStage,
                                       wholeGraphReady,
                                       qsaDeviceCompatibleMask,
                                       attentionMask,
                                       positionIds,
                                       pastKeyValues,
                                       requestState,
                                       verificationCapture,
                                       runFullAttentionGraph,
                                       runCapturedSegment};
        if (graphMode != ExecutionMode::Gpu)
            execution.PrepareHybridMask();

        if (wholeGraphReady) {
            if (mtpTargetGraph && !hybridEagerAttention) {
                for (int layer = fullAttentionCacheStartLayer; layer < graphEndLayer; layer++) {
                    if (!this->IsLinearAttentionLayer(layer)) {
                        verificationCapture->qsaRawKeys[layer].Resize(
                            {graphSequence, this->indexerHeadDim});
                    }
                }
            }
            if (graphMode == ExecutionMode::Gpu)
                execution.RunGpu();
            else
                execution.RunHybrid();

            const int keyLength = decodePreviousLength + graphSequence;
            const int tailCount = keyLength % this->indexerCompressRatio;
            const int blockCount = keyLength / this->indexerCompressRatio;
            const int oldTailCount = decodePreviousLength % this->indexerCompressRatio;
            const int newlyCompletedRows =
                ((oldTailCount + graphSequence) / this->indexerCompressRatio) *
                this->indexerCompressRatio;
            for (int layer = fullAttentionCacheStartLayer; layer < graphEndLayer; layer++) {
                if (this->IsLinearAttentionLayer(layer) || hybridEagerAttention) {
                    continue;
                }
                Data &pastKey = pastKeyValues[layer].first;
                Data &pastValue = pastKeyValues[layer].second;
                Qwen4ResizeKVCache(pastKey, keyLength);
                Qwen4ResizeKVCache(pastValue, keyLength);
                requestState.geometricCacheGrowthReadyLayers.insert(layer);

                std::shared_ptr<Data> &tailKeys = requestState.indexerTailKeyTensors[layer];
                std::shared_ptr<Data> &tailPositions =
                    requestState.indexerTailPositionTensors[layer];
                std::shared_ptr<Data> &blockCache = requestState.indexerBlockKeyTensors[layer];
                if (mtpTargetGraph && newlyCompletedRows > 0) {
                    AssertInFastLLM(verificationCapture != nullptr,
                                    "Qwen4-Exp target graph lost its QSA capture state.");
                    std::shared_ptr<QsaHostMirrorTransfer> &transfer =
                        requestState.indexerHostMirrorTransfers[layer];
                    int queuedRows = decodePreviousLength - oldTailCount;
                    if (oldTailCount > 0) {
                        Data oldKeys, oldPositions;
                        oldKeys.FakeFrom(graphState->qsaTailKeySnapshots[layer], 0);
                        oldKeys.Resize({oldTailCount, this->indexerHeadDim});
                        oldPositions.FakeFrom(graphState->qsaTailPositionSnapshots[layer], 0);
                        oldPositions.Resize({oldTailCount});
                        transfer->Queue(oldKeys, oldPositions, queuedRows, oldTailCount,
                                        this->indexerHeadDim);
                        queuedRows += oldTailCount;
                    }
                    const int candidateRows = newlyCompletedRows - oldTailCount;
                    if (candidateRows > 0) {
                        Data rawKeys, candidatePositions;
                        rawKeys.FakeFrom(verificationCapture->qsaRawKeys[layer], 0);
                        rawKeys.Resize({candidateRows, this->indexerHeadDim});
                        candidatePositions.FakeFrom(graphState->positionIds, 0);
                        candidatePositions.Resize({candidateRows});
                        transfer->Queue(rawKeys, candidatePositions, queuedRows, candidateRows,
                                        this->indexerHeadDim);
                    }
                } else if (!mtpTargetGraph && tailCount == 0) {
                    const int previousRows = keyLength - this->indexerCompressRatio;
                    requestState.indexerHostMirrorTransfers[layer]->Queue(
                        *tailKeys, *tailPositions, previousRows, this->indexerCompressRatio,
                        this->indexerHeadDim);
                }
                tailKeys->Resize({tailCount, this->indexerHeadDim});
                tailPositions->Resize({tailCount});
                blockCache->Resize({blockCount, this->indexerHeadDim});
            }
        } else {
            execution.RunGpu();
        }

        if (mtpTargetGraph && hybridDenseGraph) {
            // Prefix commit consumes these outputs before the next forward.
            // The length-specific graph owns them throughout that interval.
            auto publish = [](Data &destination, const Data &source) {
                destination.FreeSpace();
                Qwen4BorrowCudaStorage(destination, source);
            };
            for (int layer = graphStartLayer; layer < graphEndLayer; ++layer) {
                if (this->IsLinearAttentionLayer(layer)) {
                    publish(callerVerification->linearConvInputs[layer],
                            verificationCapture->linearConvInputs[layer]);
                    publish(callerVerification->linearAlphas[layer],
                            verificationCapture->linearAlphas[layer]);
                    publish(callerVerification->linearBetas[layer],
                            verificationCapture->linearBetas[layer]);
                } else {
                    publish(callerVerification->qsaRawKeys[layer],
                            verificationCapture->qsaRawKeys[layer]);
                }
            }
        }
        if (boundedStage && graphEndLayer < this->block_cnt) {
            stageOutput->CopyFrom(graphState->hiddenStates[0]);
            return true;
        }
        if (graphState->logits.dataDevice != DataDevice::CUDA ||
            graphState->logits.cudaData == nullptr || graphState->logits.dims.empty()) {
            return false;
        }
        if (targetMultiHidden != nullptr) {
            targetMultiHidden->CopyFrom(graphState->hiddenStates[0]);
        }
        Qwen4BorrowCudaStorage(logits, graphState->logits);
        return true;
#endif
    }
} // namespace fastllm
