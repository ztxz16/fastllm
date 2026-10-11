#if defined(USE_CUDA) && !defined(USE_ROCM)
struct Qwen4ExpModel::PipelineMoeState {
    std::vector<int> devices;
    PersistentWorkerGroup workers;
    std::shared_ptr<FastllmCudaMoeExpertParallel> experts;
    Data peerInput[2], peerOutput[2], sum[2];
    void *ready[2] = {nullptr, nullptr}, *done[2] = {nullptr, nullptr};

    explicit PipelineMoeState(const std::vector<CudaPrefillStage> &stages) {
        if (stages.size() != 2) return;
        Qwen4CudaDeviceGuard guard;
        for (const auto &stage : stages) devices.push_back(stage.device);
        // Peer copies otherwise silently stage through host memory, even with
        // NVLink present. Failure leaves the existing staged-copy fallback.
        FastllmCudaPeerAccessInit(devices);
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

bool Qwen4ExpModel::TryRunPipelineMoe(int layer, const Data &input,
        const Data &index, const Data &score, Data &output,
        std::vector<Data *> &moeWeights, const std::function<void()> &sharedExpert) {
#ifdef USE_NUMAS
    if (threadTpState || threadTpRank >= 0 || Qwen4MtpDraftsPerStep() != 0 ||
        input.dataDevice != CUDA ||
        input.dims.size() != 2 || input.dims[0] != 1 ||
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
        data.Resize(input.dims);
        data.ToDevice(CUDA, std::vector<int>{target}, false);
        data.Allocate(false);
    };
    prepare(state.peerInput[peer], state.devices[peer], input.dataType);
    prepare(state.sum[owner], device, FLOAT32);
    FastllmCudaEventRecordCurrentThread(state.ready[owner]);
    bool accepted[2] = {false, false};
    std::vector<std::exception_ptr> errors(2);
    // Keep physical rank/expert ownership stable as the PP layer owner moves.
    // Only the narrow normalized activation and the reduced MoE result travel.
    state.workers.RunWithCaller({0, 1}, [&](int worker) {
        const int rank = worker == 0 ? owner : peer;
        const int target = state.devices[rank];
        void *previous = GetExecutor();
        struct RestoreExecutor {
            void *previous;
            ~RestoreExecutor() { SetCurrentThreadExecutor(previous); }
        } restore{previous};
        FastllmCudaSetDevice(target);
        if (worker != 0) {
            static thread_local std::unique_ptr<Executor> executor(new Executor());
            executor->SetFirstDevice("cuda:" + std::to_string(target));
            SetCurrentThreadExecutor(executor.get());
            FastllmCudaCurrentThreadStreamWaitEvent(state.ready[owner]);
            if (!FastllmCudaMemcpyPeerAsyncCurrentThread(target,
                    state.peerInput[peer].cudaData, device, input.cudaData, input.GetBytes())) {
                FastllmCudaEventSynchronize(state.ready[owner]);
                FastllmCudaMemcpyBetweenDevices(target, state.peerInput[peer].cudaData,
                    device, input.cudaData, input.GetBytes());
            }
        }
        accepted[rank] = FastllmCudaMergeMOEExpertParallel(*state.experts, rank,
            worker == 0 ? input : state.peerInput[peer], index, score,
            worker == 0 ? output : state.peerOutput[peer],
            moeWeights.data(), int(moeWeights.size()), layer,
            worker == 0 ? sharedExpert : std::function<void()>(), owner);
        FastllmCudaEventRecordCurrentThread(state.done[rank]);
        // The persistent peer keeps its stream alive. The owner's event wait
        // below joins GPU completion; do not also block its CPU worker here.
    }, errors);
    FastllmCudaSetDevice(device);
    for (const auto &error : errors) if (error) std::rethrow_exception(error);
    AssertInFastLLM(accepted[0] == accepted[1], "PP MoE collective rejection mismatch.");
    if (!accepted[owner]) return false;
    FastllmCudaCurrentThreadStreamWaitEvent(state.done[peer]);
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
