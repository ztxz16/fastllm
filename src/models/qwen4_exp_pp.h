#if defined(USE_CUDA) && !defined(USE_ROCM)
struct Qwen4ExpModel::PipelineMoeState {
    std::vector<int> devices;
    std::vector<std::vector<int>> workerCpus;
    std::thread::id boundPeerThread;
    bool peerAccess = false;
    PersistentWorkerGroup workers;
    std::shared_ptr<FastllmCudaMoeExpertParallel> experts;
    Data peerInput[2], peerOutput[2], sum[2];
    void *ready[2] = {nullptr, nullptr}, *done[2] = {nullptr, nullptr};

#if defined(USE_NUMAS) && defined(__linux__)
    struct CallerAffinity {
        cpu_set_t original;
        bool restore = false;
        explicit CallerAffinity(PipelineMoeState *state) {
            // The persistent peer must already exist before narrowing the
            // caller's mask, otherwise it would inherit the caller's cores.
            if (state && state->workerCpus.size() == 2 && state->workers.HasWorkers() &&
                sched_getaffinity(0, sizeof(original), &original) == 0) {
                restore = BindNumasWorkerCpuSet(state->workerCpus[0]);
            }
        }
        ~CallerAffinity() {
            if (restore) sched_setaffinity(0, sizeof(original), &original);
        }
    };
#endif

    explicit PipelineMoeState(const std::vector<CudaPrefillStage> &stages) {
        if (stages.size() != 2) return;
        Qwen4CudaDeviceGuard guard;
        for (const auto &stage : stages) devices.push_back(stage.device);
#ifdef USE_NUMAS
        workerCpus = GetNumasCudaWorkerCpuSets(devices);
#endif
        // Peer copies otherwise silently stage through host memory, even with
        // NVLink present. Failure leaves the existing staged-copy fallback.
        peerAccess = FastllmCudaPeerAccessInit(devices);
        experts = FastllmCudaCreateMoeExpertParallel(2);
        for (int rank = 0; rank < 2; ++rank) {
            FastllmCudaSetDevice(devices[rank]);
            ready[rank] = FastllmCudaEventCreate();
            done[rank] = FastllmCudaEventCreate();
        }
        AssertInFastLLM(!FastllmCudaGetThreadError(), "PP MoE event initialization failed.");
    }
    ~PipelineMoeState() {
        workers.Stop();
        Qwen4CudaDeviceGuard guard;
        experts.reset();
        for (int rank = 0; rank < int(devices.size()); ++rank) {
            FastllmCudaSetDevice(devices[rank]);
            FastllmCudaSyncCurrentThreadStream();
            peerInput[rank].FreeSpace(); peerOutput[rank].FreeSpace(); sum[rank].FreeSpace();
            if (ready[rank]) FastllmCudaEventDestroy(ready[rank]);
            if (done[rank]) FastllmCudaEventDestroy(done[rank]);
        }
    }
};

bool Qwen4ExpModel::TryPipelineMoeReadback(const Data &input, const Data &index, const Data &score) {
#ifdef USE_NUMAS
    if (threadTpRank >= 0) {
        return threadTpRank == 0 && threadTpOwner && threadTpOwner->expertParallel &&
            FastllmCudaMoeExpertParallelReadback(
                *threadTpOwner->expertParallel, threadTpRank, input, index, score);
    }
    if (threadTpState || !pipelineMoe)
        return false;
    const auto &devices = pipelineMoe->devices;
    const auto found = std::find(devices.begin(), devices.end(), FastllmCudaGetDevice());
    return found != devices.end() && FastllmCudaMoeExpertParallelReadback(
        *pipelineMoe->experts, found - devices.begin(), input, index, score);
#else
    return false;
#endif
}

bool Qwen4ExpModel::TryRunPipelineMoe(int layer, const Data &input,
        const Data &index, const Data &score, Data &output,
        std::vector<Data *> &moeWeights, const std::function<void()> &sharedExpert, Data **deferredPeer, bool preparedReadback) {
#ifdef USE_NUMAS
    if (threadTpState || threadTpRank >= 0 ||
        input.dataDevice != CUDA ||
        input.dims.size() != 2 || input.dims[0] < 1 ||
        input.dims[0] > FASTLLM_CUDA_MOE_CACHE_MAX_BATCH ||
        !MoeCudaCacheRequested() || FastllmCudaGraphIsCapturingFast()) return false;
    if (!pipelineMoe) {
        std::vector<CudaPrefillStage> stages;
        if (!BuildCudaPrefillStages(deviceMap, block_cnt, stages)) stages.clear();
        pipelineMoe.reset(new PipelineMoeState(stages));
    }
    auto &state = *pipelineMoe;
    if (state.devices.size() != 2) return false;
    const int device = FastllmCudaGetDevice();
    const auto found = std::find(state.devices.begin(), state.devices.end(), device);
    if (found == state.devices.end()) return false;
    const int owner = found - state.devices.begin(), peer = 1 - owner;
    Qwen4CudaDeviceGuard guard;
    auto prepare = [&](Data &data, int target, DataType type) {
        FastllmCudaSetDevice(target);
        data.dataType = type;
        data.Resize({FASTLLM_CUDA_MOE_CACHE_MAX_BATCH, input.dims[1]});
        data.ToDevice(CUDA, std::vector<int>{target}, false);
        data.Allocate(false);
        data.Resize(input.dims);
    };
    FastllmCudaEventRecordCurrentThread(state.ready[owner]);
    const bool borrowInput = state.peerAccess && input.dataDeviceIds == std::vector<int>{device};
    bool accepted[2] = {false, false};
    std::vector<std::exception_ptr> errors(2);
    // Keep physical rank/expert ownership stable as the PP layer owner moves.
    // Only the narrow normalized activation and the reduced MoE result travel.
    state.workers.RunWithCaller({0, 1}, [&](int worker) {
        const auto thread = std::this_thread::get_id();
        if (worker != 0 && state.boundPeerThread != thread) {
            BindNumasWorkerCpuSet(state.workerCpus[worker]);
            state.boundPeerThread = thread;
        }
        const int rank = worker == 0 ? owner : peer;
        const int target = state.devices[rank];
        void *previous = GetExecutor();
        struct RestoreExecutor {
            void *previous;
            bool previousCaptureMode;
            ~RestoreExecutor() {
                SetCurrentThreadExecutor(previous);
                FastllmCudaGraphSetManagedCaptureOnly(previousCaptureMode);
            }
        } restore{previous, FastllmCudaGraphSetManagedCaptureOnly(true)};
        FastllmCudaSetDevice(target);
        if (worker != 0) {
            // Dense graph variants retain this peer output across changing
            // verifier lengths. Its allocation must not grow on a later row.
            prepare(state.peerOutput[peer], target, FLOAT32);
            // Allocate on the thread that owns the peer stream, avoiding a
            // switch to the peer device and back on every owner submission.
            if (!borrowInput) prepare(state.peerInput[peer], target, input.dataType);
            static thread_local std::unique_ptr<Executor> executor(new Executor());
            executor->SetFirstDevice("cuda:" + std::to_string(target));
            SetCurrentThreadExecutor(executor.get());
            FastllmCudaCurrentThreadStreamWaitEvent(state.ready[owner]);
            if (!borrowInput && !FastllmCudaMemcpyPeerAsyncCurrentThread(target,
                    state.peerInput[peer].cudaData, device, input.cudaData, input.GetBytes())) {
                FastllmCudaEventSynchronize(state.ready[owner]);
                FastllmCudaMemcpyBetweenDevices(target, state.peerInput[peer].cudaData,
                    device, input.cudaData, input.GetBytes());
            }
        }
        accepted[rank] = FastllmCudaMergeMOEExpertParallel(*state.experts, rank,
            worker == 0 || borrowInput ? input : state.peerInput[peer], index, score,
            worker == 0 ? output : state.peerOutput[peer],
            moeWeights.data(), int(moeWeights.size()), layer,
            worker == 0 ? sharedExpert : std::function<void()>(), owner,
            preparedReadback && worker == 0);
        if (worker != 0) FastllmCudaEventRecordCurrentThread(state.done[rank]);
        // The persistent peer keeps its stream alive. The owner's event wait
        // below joins GPU completion; do not also block its CPU worker here.
    }, errors);
    FastllmCudaSetDevice(device);
    for (const auto &error : errors) if (error) std::rethrow_exception(error);
    AssertInFastLLM(accepted[0] == accepted[1], "PP MoE collective rejection mismatch.");
    if (!accepted[owner]) return false;
    FastllmCudaCurrentThreadStreamWaitEvent(state.done[peer]);
    if (state.peerAccess && output.dataType == FLOAT32 &&
        state.peerOutput[peer].dataType == FLOAT32 &&
        output.dims == state.peerOutput[peer].dims) {
        // Both allocations are peer-accessible. The event protects the remote
        // read; the next input-ready event protects it from peer reuse.
        if (deferredPeer) {
            // The per-rank output stays at a stable address. Consume it in
            // the next dense graph, after this stream's peer-completion wait.
            *deferredPeer = &state.peerOutput[peer];
            return true;
        }
        return FastllmCudaAddTo(output, state.peerOutput[peer], 1.0f);
    }
    prepare(state.sum[owner], device, FLOAT32);
    if (!FastllmCudaMemcpyPeerAsyncCurrentThread(device, state.sum[owner].cudaData,
            state.devices[peer], state.peerOutput[peer].cudaData, state.sum[owner].GetBytes())) {
        FastllmCudaEventSynchronize(state.done[peer]);
        FastllmCudaMemcpyBetweenDevices(device, state.sum[owner].cudaData,
            state.devices[peer], state.peerOutput[peer].cudaData, state.sum[owner].GetBytes());
    }
    AddTo(output, state.sum[owner]);
    return true;
#else
    return false;
#endif
}
#endif

bool Qwen4ExpModel::TryForwardChunkedPrefill(
        const Data &inputIds, const Data &attentionMask, const Data &positionIds,
        std::vector<std::pair<Data, Data>> &pastKeyValues,
        const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
        std::vector<float> *logits, int &outputToken) {
#if defined(USE_CUDA) && !defined(USE_ROCM)
    const int chunkSize = GetChunkedPrefillSize();
    if (threadTpState || threadTpRank >= 0 || Qwen4MtpDraftsPerStep() != 0 ||
        !generationConfig.IsSimpleGreedy() || saveHistoryChat ||
        GetKVCacheInCPU() || GetHistoryCacheInCPU() || GetLowMemMode() ||
        GetFastllmEnv().printProfile || FastllmCudaGraphIsCapturingFast() ||
        inputIds.dataDevice != DataDevice::CPU || inputIds.dataType != FLOAT32 ||
        !inputIds.cpuData || inputIds.dims.size() != 2 || inputIds.dims[0] != 1 ||
        pastKeyValues.size() < size_t(block_cnt) ||
        chunkSize <= 0 || inputIds.dims[1] <= chunkSize ||
        !attentionMask.dims.empty() ||
        positionIds.dataDevice != DataDevice::CPU || positionIds.dataType != FLOAT32 ||
        !positionIds.cpuData || positionIds.dims != inputIds.dims) return false;

    std::vector<CudaPrefillStage> stages;
    if (!BuildCudaPrefillStages(deviceMap, block_cnt, stages)) return false;
    for (const auto &kv : pastKeyValues)
        if (!kv.first.dims.empty() || !kv.second.dims.empty() ||
            !kv.first.pageIndex.empty() || !kv.second.pageIndex.empty()) return false;
    PrepareWeights();
    RequestState *request;
    {
        std::lock_guard<std::mutex> lock(stateMutex);
        request = &requestStates[&pastKeyValues[0].first];
        if (request->hasMultimodalInput || request->borrowedPrefixSnapshot ||
            !request->processedTokens.empty() || request->mtpState) return false;
    }

    const int sequence = inputIds.dims[1];
    const int chunks = 1 + (sequence - 1) / chunkSize;
    Qwen4CudaDeviceGuard restoreDevice;
#ifdef USE_NUMAS
    bool hasNumasMoe = false;
#endif
    for (const auto &stage : stages) {
        FastllmCudaSetDevice(stage.device);
        for (int layer = stage.firstLayer; layer < stage.endLayer; ++layer) {
            const std::string moe = SelectMoeDeviceForLayer(layer);
            const std::string dense = SelectDeviceFromMap(deviceMap, layer + 1, block_cnt);
            if (moe.empty() || moe == dense) continue;
            const bool host = moe == "cpu" || moe == "numa" || moe.rfind("numa:", 0) == 0;
            if (!host) return false;
#ifdef USE_NUMAS
            hasNumasMoe |= moe == "numa" || moe.rfind("numa:", 0) == 0;
#endif
        }
    }

    // Once workers start, never retry a partially appended request through
    // serial Forward. All fallback checks must precede this boundary.
    AcquireServingCache(pastKeyValues, *request, sequence);
    std::vector<RequestState> states(stages.size(), *request);
    std::vector<PrefillStageContext> contexts(stages.size());
    std::mutex hostExpertMutex;
    std::vector<int> devices;
    int pleOwner = 0;
    for (int rank = 0; rank < int(stages.size()); ++rank) {
        const auto &stage = stages[rank];
        devices.push_back(stage.device);
        contexts[rank].firstLayer = stage.firstLayer;
        contexts[rank].endLayer = stage.endLayer;
        contexts[rank].state = &states[rank];
        contexts[rank].hostExpertMutex = &hostExpertMutex;
        if (stage.firstLayer <= pleLayer && pleLayer < stage.endLayer) pleOwner = rank;
        FastllmCudaSetDevice(stage.device);
        auto &scale = prefillLinearInvScales[stage.device];
        if (scale.dims.empty()) {
            scale.CopyFrom(linearInvScaleData);
            scale.ToDevice(CUDA, std::vector<int>{stage.device});
        }
    }
    FastllmCudaPeerAccessInit(devices);
#ifdef USE_NUMAS
    const auto workerCpus = hasNumasMoe ? GetNumasCudaWorkerCpuSets(devices)
        : std::vector<std::vector<int>>(devices.size());
#endif
    if (!prefillPipeline) prefillPipeline.reset(new CudaChunkedPrefillPipeline());
    const DataType type = weight[languagePrefix + "embed_tokens.weight"].dataType == FLOAT16
        ? FLOAT16 : FLOAT32;
    const float *tokens = reinterpret_cast<const float *>(inputIds.cpuData);
    const float *positions = reinterpret_cast<const float *>(positionIds.cpuData);
    std::vector<std::vector<float> *> batchLogits{logits};
    if (verbose) {
        std::printf("[Prompt] Qwen4 pipelined prefill: %zu CUDA stages, %d chunks (%d tokens/chunk).\n",
                    stages.size(), chunks, chunkSize);
        std::fflush(stdout);
    }
    struct ClearPrefetch {
        const std::vector<int> &devices;
        ~ClearPrefetch() {
            Qwen4CudaDeviceGuard guard;
            for (int device : devices) {
                FastllmCudaSetDevice(device);
                FastllmCudaClearMoeGGUFHostPrefetch();
            }
        }
    } clearPrefetch{devices};
    prefillPipeline->Run(devices, chunks, type, {1, chunkSize, hcCount * embed_dim},
        [&](int rank, int chunk, Data &hidden) {
#ifdef USE_NUMAS
            // These pipeline workers own CUDA submission; their MoE helper
            // threads inherit the same spare-core set as the existing TP path.
            if (chunk == 0) BindNumasWorkerCpuSet(workerCpus[rank]);
#endif
            const int start = chunk * chunkSize;
            const int end = start + std::min(chunkSize, sequence - start);
            Data ids(FLOAT32, {1, end - start}, std::vector<float>(tokens + start, tokens + end));
            Data pos(FLOAT32, {1, end - start}, std::vector<float>(positions + start, positions + end));
            auto &context = contexts[rank];
            context.hidden = &hidden;
            context.sampleOutput = rank + 1 == int(stages.size()) && chunk + 1 == chunks;
            const auto result = ForwardTarget(1, ids, Data(), pos, pastKeyValues,
                generationConfig, lastTokens, &batchLogits, nullptr, nullptr, nullptr,
                nullptr, false, false, false, nullptr, false, nullptr, &context);
            if (context.sampleOutput) outputToken = result.front();
        });

    // Only the PLE stage advances token/short-convolution history. Attention
    // maps are owned by disjoint layer ranges and can be joined after draining.
    *request = std::move(states[pleOwner]);
    for (int rank = 0; rank < int(states.size()); ++rank) {
        if (rank == pleOwner) continue;
        auto &other = states[rank];
        request->indexerRawKeys.merge(other.indexerRawKeys);
        request->indexerPositions.merge(other.indexerPositions);
        request->indexerBlockKeys.merge(other.indexerBlockKeys);
        request->indexerHostMirrorTransfers.merge(other.indexerHostMirrorTransfers);
        request->indexerTailKeyTensors.merge(other.indexerTailKeyTensors);
        request->indexerTailPositionTensors.merge(other.indexerTailPositionTensors);
        request->indexerBlockKeyTensors.merge(other.indexerBlockKeyTensors);
        request->geometricCacheGrowthReadyLayers.merge(other.geometricCacheGrowthReadyLayers);
    }
    return true;
#else
    return false;
#endif
}
