// MiMo EP8 keeps exactly one full copy of each expert across the eight ranks.
// Attention remains TP8. Large inputs are partitioned into disjoint token
// ranges before EP dispatch, then merged in-place into the existing output.
// Small or uneven decode batches and verification retain replicated dispatch.
#include "devices/cuda/fastllm-mimo-ep8-mega.h"
#include "devices/cuda/fastllm-mimo-ep8-mega-kernel.h"
#if defined(FASTLLM_ENABLE_MIMO_EP8_MEGA) && !defined(_WIN32) && !defined(USE_ROCM)
#include "devices/multicuda/fastllm-multicuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include <chrono>
#include <condition_variable>
#include <mutex>
struct FastllmCudaMimoEp8Mega {
    struct Rank {
        fastllm::Data workspace{fastllm::INT8};
        std::vector<std::unique_ptr<fastllm::Data>> scales;
        std::vector<void *> layers;
        void *output = nullptr; // Borrowed only during the prefill exchange.
        void *prefillScratch = nullptr;
        void *prefillReady = nullptr;
        ~Rank() {
            for (void *layer : layers) if (layer) fastllm_ep8_mega_moe_destroy(layer);
            if (prefillReady) FastllmCudaEventDestroy(prefillReady);
        }
    };
    std::vector<int> devices;
    std::vector<std::unique_ptr<Rank>> ranks;
    // Publish completed outputs, then keep them alive until all peers finish.
    std::mutex outputMutex;
    std::condition_variable outputReady;
    int arrivals = 0;
    uint64_t generation = 0;
    bool exchangeFailed = false;
    void WaitForOutputs() {
        std::unique_lock<std::mutex> lock(outputMutex);
        const uint64_t current = generation;
        if (++arrivals == (int)devices.size()) {
            arrivals = 0;
            ++generation;
            outputReady.notify_all();
        } else if (!outputReady.wait_for(lock, std::chrono::seconds(60),
                [&] { return generation != current || exchangeFailed; })) {
            exchangeFailed = true;
            outputReady.notify_all();
        }
        if (exchangeFailed) throw std::runtime_error("MiMo EP8 output exchange timed out.");
    }
};
bool FastllmCudaMimoEp8MegaSupported(const std::vector<int> &devices) {
    if (devices.size() != 8) return false;
    int previous = FastllmCudaGetDevice();
    bool ok = true;
    for (int device : devices) {
        FastllmCudaSetDevice(device);
        ok = fastllm_ep8_mega_moe_supported() && ok;
    }
    FastllmCudaSetDevice(previous);
    return ok;
}
size_t FastllmCudaMimoEp8PrefillWorkspaceBytes(int rows) {
    return fastllm_ep8_mega_moe_prefill_workspace_bytes(rows);
}
std::shared_ptr<FastllmCudaMimoEp8Mega> FastllmCudaCreateMimoEp8Mega(
        const std::vector<int> &devices,
        const std::vector<std::vector<std::vector<fastllm::Data *>>> &weights) {
    using namespace fastllm;
    AssertInFastLLM(FastllmCudaMimoEp8MegaSupported(devices) && weights.size() == 8,
                   "MiMo EP8 MegaMoE requires eight H20 GPUs.");
    const size_t bytes = fastllm_ep8_mega_moe_workspace_bytes();
    // Dedicated control storage must survive attention/cuBLAS calls. Sharing
    // the old transient scratch would race remote dispatch writes against a
    // later rank's memset. Bound it; retain no duplicate/repacked FP8 weights.
    AssertInFastLLM(bytes > 0 && bytes <= 48 * 1024 * 1024,
                   "MiMo EP8 symmetric workspace exceeds its memory budget.");
    auto state = std::make_shared<FastllmCudaMimoEp8Mega>();
    state->devices = devices;
    std::vector<const void *> peers;
    int previous = FastllmCudaGetDevice();
    for (int r = 0; r < 8; ++r) {
        FastllmCudaSetDevice(devices[r]);
        auto rank = std::make_unique<FastllmCudaMimoEp8Mega::Rank>();
        rank->prefillReady = FastllmCudaEventCreate();
        rank->workspace.Resize({(int)bytes});
        rank->workspace.ToDevice(DataDevice::CUDA, {devices[r]}, false);
        rank->workspace.Allocate(false);
        AssertInFastLLM(fastllm_ep8_mega_moe_init_workspace(rank->workspace.cudaData, bytes, devices.data(), r), fastllm_ep8_mega_moe_error());
        peers.push_back(rank->workspace.cudaData);
        state->ranks.push_back(std::move(rank));
    }
    for (int r = 0; r < 8; ++r) {
        FastllmCudaSetDevice(devices[r]);
        auto &rank = *state->ranks[r];
        rank.layers.resize(weights[r].size(), nullptr);
        for (size_t l = 0; l < weights[r].size(); ++l) {
            if (weights[r][l].empty()) continue;
            AssertInFastLLM(weights[r][l].size() == 514, "Invalid MiMo EP8 expert table.");
            std::vector<const void *> gate, down, gs, ds;
            for (int e = r * 32; e < (r + 1) * 32; ++e) {
                for (int part = 0; part < 2; ++part) {
                    Data *w = weights[r][l][2 + e * 2 + part];
                    const std::vector<int> shape = part ? std::vector<int>{4096, 2048} : std::vector<int>{4096, 4096};
                    AssertInFastLLM(w && w->dataType == FP8_E4M3 && w->dims == shape &&
                        w->blockM == 128 && w->blockK == 128 && w->cudaData &&
                        w->dataDeviceIds == std::vector<int>{devices[r]} &&
                        w->scales.size() == (size_t)shape[0] * shape[1] / (128 * 128),
                        "MiMo EP8 requires complete native FP8 experts with 128x128 scales.");
                    // Native MergeMOE creates scales plus an unused zero-bias
                    // allocation for every TP shard. EP8 needs only the scales.
                    auto scale = std::make_unique<Data>(FLOAT32, std::vector<int>{(int)w->scales.size()}, w->scales);
                    scale->ToDevice(DataDevice::CUDA, {devices[r]}, true);
                    (part ? down : gate).push_back(w->cudaData);
                    (part ? ds : gs).push_back(scale->cudaData);
                    rank.scales.push_back(std::move(scale));
                }
            }
            rank.layers[l] = fastllm_ep8_mega_moe_create(gate.data(), down.data(), gs.data(), ds.data(),
                rank.workspace.cudaData, bytes, peers.data(), r, l);
            AssertInFastLLM(rank.layers[l] != nullptr, fastllm_ep8_mega_moe_error());
        }
        printf("MIMO_EP8_MEGA gpu=%d experts=32 inter=2048 W8A8 workspace=%zu max_rows=%d Graph-ready backend=builtin\n",
               devices[r], bytes, fastllm_ep8_mega_moe_max_rows());
    }
    FastllmCudaSetDevice(previous);
    return state;
}
void FastllmCudaMimoEp8PrefillReduceScatter(FastllmCudaMimoEp8Mega &state, int rank,
                                          fastllm::Data &output) {
    using namespace fastllm;
    const int ranks = (int)state.devices.size();
    const uint64_t elements = output.Count(0);
    AssertInFastLLM(rank >= 0 && rank < ranks && elements > 0 && elements % ranks == 0 &&
        elements / ranks <= std::numeric_limits<int>::max() &&
        output.cudaData && output.dataType == BFLOAT16 &&
        output.dataDevice == DataDevice::CUDA &&
        output.dataDeviceIds.size() == 1 && output.dataDeviceIds[0] == state.devices[rank],
        "Invalid MiMo EP8 prefill ReduceScatter tensor.");
    const int count = (int)(elements / ranks);
    state.WaitForOutputs();
    const bool ok = FastllmNcclReduceScatter(output.cudaData,
        (uint16_t *)output.cudaData + (size_t)rank * count,
        count, output.dataType, state.devices[rank]);
    state.WaitForOutputs();
    AssertInFastLLM(ok, "MiMo EP8 prefill ReduceScatter failed.");
}
void FastllmCudaMimoEp8PrefillAllGather(FastllmCudaMimoEp8Mega &state, int rank,
                                      fastllm::Data &output) {
    // Publish completed shards, then gather directly into the existing output.
    // All sources stay alive until every peer finishes reading them. This
    // eager-only path needs neither zero padding nor another collective arena.
    using namespace fastllm;
    const int ranks = (int)state.devices.size();
    AssertInFastLLM(rank >= 0 && rank < ranks && output.dims.size() == 2 &&
        output.dims[0] > 0 && output.dims[1] > 0 && output.cudaData &&
        output.dataDevice == DataDevice::CUDA &&
        output.dataDeviceIds == std::vector<int>{state.devices[rank]} &&
        (output.dataType == BFLOAT16 || output.dataType == INT32),
        "Invalid MiMo EP8 prefill AllGather tensor.");
    const int total = output.dims[0];
    const size_t rowBytes = (size_t)output.dims[1] * output.unitSize;
    if (total % ranks == 0) {
        // The collective consumes and produces on each worker's compute
        // stream. Rendezvous only for submission, never for GPU completion.
        const int rows = total / ranks;
        state.WaitForOutputs();
        AssertInFastLLM(FastllmNcclAllGather(
            (const uint8_t *)output.cudaData + (size_t)rank * rows * rowBytes,
            output.cudaData, rows * output.dims[1], output.dataType, state.devices[rank]),
            "MiMo EP8 prefill AllGather failed.");
        state.WaitForOutputs();
        return;
    }
    state.ranks[rank]->output = output.cudaData;
    FastllmCudaSyncCurrentThreadStream();
    state.WaitForOutputs();
    for (int step = 1; step < ranks; ++step) {
        const int peer = (rank + step) % ranks;
        const size_t first = (int64_t)total * peer / ranks;
        const size_t rows = (int64_t)total * (peer + 1) / ranks - first;
        AssertInFastLLM(FastllmCudaMemcpyPeerAsyncCurrentThread(state.devices[rank],
            (uint8_t *)output.cudaData + first * rowBytes, state.devices[peer],
            (const uint8_t *)state.ranks[peer]->output + first * rowBytes,
            rows * rowBytes), "MiMo EP8 peer output copy failed.");
    }
    FastllmCudaSyncCurrentThreadStream();
    state.WaitForOutputs();
}
bool FastllmCudaRunMimoEp8Mega(FastllmCudaMimoEp8Mega &state, int rank, int layer,
        const fastllm::Data &input, const fastllm::Data &index, const fastllm::Data &score,
        fastllm::Data &output, bool graphDecode, fastllm::Data *residual,
        const fastllm::Data *normWeight, fastllm::Data *normOutput, float eps,
        fastllm::Data *prefillScratch, const int *liveKeys) {
    using namespace fastllm;
    AssertInFastLLM(input.dataType == BFLOAT16 && input.dims.size() == 2 &&
        input.dims[1] == 4096, "Invalid MiMo EP8 input shape.");
    const int ranks = (int)state.devices.size();
    const int total = input.dims[0];
    const int begin = (int)((int64_t)total * rank / ranks);
    const int shard = (int)((int64_t)total * (rank + 1) / ranks) - begin;
    const bool localRouting = total >= ranks && (!graphDecode || total % ranks == 0) &&
        index.dims == std::vector<int>({shard, 8});
    AssertInFastLLM((localRouting || index.dims == std::vector<int>({total, 8})) &&
        score.dims == index.dims, "Invalid MiMo EP8 routing shape.");
    output.dataType = BFLOAT16;
    output.Resize(input.dims);
    output.ToDevice(DataDevice::CUDA, {state.devices.at(rank)}, false);
    output.Allocate(false);
    void *handle = state.ranks.at(rank)->layers.at(layer);
    AssertInFastLLM(handle != nullptr, "Missing MiMo EP8 layer handle.");
    int capacity = fastllm_ep8_mega_moe_max_rows();
    const void *prefillPeers[8]{};
    // Eligibility uses the common input shape so every rank participates even
    // when one local buffer is unavailable. Router FP32 input is dead here;
    // it is overwritten by the next layer's conversion, after AllGather's
    // final rendezvous has released every remote reader/writer.
    if (!graphDecode && FastllmCudaMimoEp8PrefillWorkspaceBytes(total)) {
        auto &local = *state.ranks[rank];
        local.prefillScratch = nullptr;
        int prefillCapacity = 0;
        if (prefillScratch && !prefillScratch->isFake && prefillScratch->dataType == FLOAT32 &&
            prefillScratch->dataDevice == DataDevice::CUDA &&
            prefillScratch->dataDeviceIds == std::vector<int>{state.devices[rank]} &&
            !prefillScratch->dims.empty() && prefillScratch->dims.back() == 4096 &&
            prefillScratch->Count(0) == (size_t)total * 4096 &&
            prefillScratch->cudaData != input.cudaData) {
            prefillCapacity = fastllm_ep8_mega_moe_prepare_prefill(
                handle, prefillScratch->cudaData, prefillScratch->expansionBytes, total);
            AssertInFastLLM(prefillCapacity >= 0, fastllm_ep8_mega_moe_error());
            if (prefillCapacity) local.prefillScratch = prefillScratch->cudaData;
        }
        FastllmCudaEventRecordCurrentThread(local.prefillReady);
        state.WaitForOutputs();
        bool ready = true;
        for (const auto &peer : state.ranks) ready = ready && peer->prefillScratch;
        if (ready) {
            capacity = prefillCapacity;
            for (int r = 0; r < ranks; ++r) {
                const auto &peer = state.ranks[r];
                FastllmCudaCurrentThreadStreamWaitEvent(peer->prefillReady);
                prefillPeers[r] = peer->prefillScratch;
            }
        }
        // Every wait must capture this generation before a fast worker can
        // re-record an event during the next layer's preparation.
        state.WaitForOutputs();
    }
    auto run = [&](int start, int rows) {
        const int routeStart = localRouting ? start - begin : start;
        const void *x = (const uint16_t *)input.cudaData + (size_t)start * 4096;
        void *y = (uint16_t *)output.cudaData + (size_t)start * 4096;
        const int32_t *idx = (const int32_t *)index.cudaData + routeStart * 8;
        const float *scores = (const float *)score.cudaData + routeStart * 8;
        const int ok = !prefillPeers[0] ?
            fastllm_ep8_mega_moe_run_masked(handle, x, y, idx, scores, rows, liveKeys ? liveKeys + start : nullptr) :
            fastllm_ep8_mega_moe_run_prefill(handle, prefillPeers, x, y, idx, scores, rows);
        if (!ok)
            throw std::runtime_error(std::string("MiMo EP8 launch failed: ") + fastllm_ep8_mega_moe_error());
    };
    if (total <= capacity && !localRouting) {
        run(0, total);
        return false;
    }

    // Uneven decode batches can replay bounded replicated chunks without
    // padding. Equal shards use graph-compatible AllGather below.
    if (graphDecode && total % ranks != 0) {
        for (int first = 0; first < total; first += capacity)
            run(first, std::min(capacity, total - first));
        return false;
    }
    // Dispatch each token from one rank, using the selected bounded arena.
    // All ranks execute the same number of rounds, including empty shards.
    const int rounds = (int)(((int64_t)total + ranks * capacity - 1) / (ranks * capacity));
    for (int round = 0; round < rounds; ++round) {
        const int start = round * capacity;
        run(begin + std::min(start, shard), std::max(0, std::min(capacity, shard - start)));
    }
    if (graphDecode) {
        if (localRouting && residual && normWeight && normOutput) {
            AssertInFastLLM(residual->dataType == BFLOAT16 && residual->Count(0) == total * 4096 &&
                normWeight->dataType == FLOAT32 && normWeight->Count(0) == 4096,
                "Invalid MiMo EP8 residual/RMSNorm shape.");
            normOutput->dataType = BFLOAT16;
            normOutput->Resize(residual->dims);
            normOutput->ToDevice(output.dataDevice, output.dataDeviceIds, false);
            normOutput->Allocate(false);
            if (FastllmCudaCustomAllGatherAddRMSNorm(output.cudaData, residual->cudaData,
                    (const float *)normWeight->cudaData, normOutput->cudaData,
                    total * 4096, 4096, eps, BFLOAT16, state.devices[rank])) return true;
        }
        if (localRouting && FastllmCudaCustomAllGatherRows(output.cudaData,
                total * 4096, 4096, BFLOAT16, state.devices[rank]))
            return false;
        // Each rank has written its slice in-place. NCCL records the exchange
        // on the same stream, retaining the existing 64-row MoE workspace.
        AssertInFastLLM(FastllmNcclAllGather(
            (const uint16_t *)output.cudaData + (size_t)begin * 4096,
            output.cudaData, shard * 4096, BFLOAT16, state.devices[rank]),
            "MiMo EP8 decode AllGather failed.");
        return false;
    }
    if (localRouting && residual && normWeight && normOutput) {
        // Keep residuals local between consecutive MoE layers. Gather the
        // normalized input needed by the next TP projection, not raw MoE output.
        normOutput->dataType = BFLOAT16;
        normOutput->Resize(residual->dims);
        normOutput->ToDevice(output.dataDevice, output.dataDeviceIds, false);
        normOutput->Allocate(false);
        const size_t offset = (size_t)begin * 4096 * sizeof(uint16_t);
        Data localResidual(BFLOAT16, {shard, 4096}), localOutput, localNorm;
        localResidual.FakeFrom(*residual, offset);
        localOutput.FakeFrom(output, offset);
        localNorm.FakeFrom(*normOutput, offset);
        FastllmCudaNaiveAddDecodeRMSNorm(localResidual, localOutput, *normWeight, eps, localNorm);
        Data gathered(BFLOAT16, {total, 4096});
        gathered.FakeFrom(*normOutput, 0);
        FastllmCudaMimoEp8PrefillAllGather(state, rank, gathered);
        return true;
    }
    FastllmCudaMimoEp8PrefillAllGather(state, rank, output);
    return false;
}
#else
struct FastllmCudaMimoEp8Mega {};
void FastllmCudaMimoEp8PrefillReduceScatter(FastllmCudaMimoEp8Mega &, int, fastllm::Data &) {
    throw std::runtime_error("MiMo EP8 MegaMoE is unavailable.");
}
bool FastllmCudaMimoEp8MegaSupported(const std::vector<int> &) { return false; }
size_t FastllmCudaMimoEp8PrefillWorkspaceBytes(int) { return 0; }
std::shared_ptr<FastllmCudaMimoEp8Mega> FastllmCudaCreateMimoEp8Mega(const std::vector<int> &,
    const std::vector<std::vector<std::vector<fastllm::Data *>>> &) { return {}; }
void FastllmCudaMimoEp8PrefillAllGather(FastllmCudaMimoEp8Mega &, int, fastllm::Data &) {
    throw std::runtime_error("MiMo EP8 prefill AllGather requires the SM90 backend.");
}
bool FastllmCudaRunMimoEp8Mega(FastllmCudaMimoEp8Mega &, int, int, const fastllm::Data &,
    const fastllm::Data &, const fastllm::Data &, fastllm::Data &, bool, fastllm::Data *,
    const fastllm::Data *, fastllm::Data *, float, fastllm::Data *, const int *) {
    throw std::runtime_error("MiMo EP8 MegaMoE requires a Linux CUDA build containing the SM90 backend.");
}
#endif
