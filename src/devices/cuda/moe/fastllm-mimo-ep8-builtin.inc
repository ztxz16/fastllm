// MiMo EP4/EP8 keeps one full copy of each expert across the participating ranks.
// Attention uses the same TP group. Large inputs are partitioned into disjoint token
// ranges before EP dispatch, then merged in-place into the existing output.
// Batches smaller than the rank count retain replicated dispatch.
#include "devices/cuda/fastllm-mimo-ep8-mega.h"
#include "devices/cuda/fastllm-mimo-ep8-mega-kernel.h"
#if defined(FASTLLM_ENABLE_MIMO_EP8_MEGA) && !defined(_WIN32) && !defined(USE_ROCM)
#include "devices/cuda/fastllm-mxfp4-ep.h"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/cuda/fastllm-moe-weight-owner.h"
#include <chrono>
#include <condition_variable>
#include <mutex>
struct FastllmCudaMimoEp8BuiltinMega {
    struct Rank {
        bool hopperMxfp4 = false;
        fastllm::Data workspace{fastllm::INT8};
        std::vector<std::unique_ptr<fastllm::Data>> scales;
        std::vector<void *> layers;
        void *output = nullptr; // Borrowed during the EP4 prefill exchange.
        void *prefillScratch = nullptr;
        void *prefillReady = nullptr;
        ~Rank() {
            for (void *layer : layers) if (layer) {
                if (hopperMxfp4) fastllm_mxfp4_ep_destroy(layer);
                else fastllm_ep8_mega_moe_destroy(layer);
            }
            if (prefillReady) FastllmCudaEventDestroy(prefillReady);
        }
    };
    std::vector<int> devices;
    std::vector<std::unique_ptr<Rank>> ranks;
    // Coordinate matching communication/workspace submissions on all ranks.
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
        if (exchangeFailed) throw std::runtime_error("MiMo EP output exchange timed out.");
    }
};
bool FastllmCudaMimoEp8BuiltinMegaWeights(FastllmCudaMimoEp8BuiltinMega &state, int rank, int layer, const void **out) {
    const auto &local = *state.ranks.at(rank);
    return !local.hopperMxfp4 && fastllm_ep8_mega_moe_weights(local.layers.at(layer), out);
}
bool FastllmCudaMimoEp8BuiltinMegaOwnsPackedWeights(int device) {
    const int previous = FastllmCudaGetDevice();
    FastllmCudaSetDevice(device);
    const bool packed = fastllm_ep8_mega_moe_owns_packed_weights();
    FastllmCudaSetDevice(previous);
    return packed;
}
bool FastllmCudaMimoEp8BuiltinMegaSupported(const std::vector<int> &devices) {
    if ((devices.size() != 4 && devices.size() != 8) ||
        !fastllm_ep8_mega_moe_group_supported(devices.data(), devices.size())) return false;
    int previous = FastllmCudaGetDevice();
    bool ok = true;
    std::string backend;
    for (int device : devices) {
        FastllmCudaSetDevice(device);
        const std::string selected = fastllm_ep8_mega_moe_backend_name();
        if (backend.empty()) backend = selected;
        ok = fastllm_ep8_mega_moe_supported() && selected == backend && ok;
    }
    FastllmCudaSetDevice(previous);
    return ok;
}
size_t FastllmCudaMimoEp8BuiltinPrefillWorkspaceBytes(int rows, int ranks) {
    return fastllm_ep8_mega_moe_prefill_workspace_bytes(rows, ranks);
}
std::shared_ptr<FastllmCudaMimoEp8BuiltinMega> FastllmCudaCreateMimoEp8BuiltinMega(
        const std::vector<int> &devices,
        const std::vector<std::vector<std::vector<fastllm::Data *>>> &weights) {
    using namespace fastllm;
    AssertInFastLLM(FastllmCudaMimoEp8BuiltinMegaSupported(devices) && weights.size() == devices.size(),
                   "MiMo EP MegaMoE requires matching supported SM90 or SM100/103 GPUs.");
    int format = -1;
    // Validate the format across all resident experts before allocating arenas.
    // Dense layers have empty tables; experts on other ranks may be null.
    for (const auto &rank : weights) for (const auto &layer : rank) {
        if (layer.empty()) continue;
        AssertInFastLLM(layer.size() == 514, "Invalid MiMo EP expert table.");
        for (size_t i = 2; i < layer.size(); ++i) if (layer[i]) {
            const auto &w = *layer[i];
            const bool fp8 = w.dataType == FP8_E4M3 && w.blockM == 128 && w.blockK == 128;
            const bool mxfp4 = FastllmCudaMimoIsMxfp4Expert(w);
            AssertInFastLLM((fp8 || mxfp4) && (format < 0 || format == (int)mxfp4),
                "MiMo EP requires uniformly FP8 or block-32 UE8M0 MXFP4 experts.");
            format = (int)mxfp4;
        }
    }
    AssertInFastLLM(format >= 0, "MiMo EP has no experts.");
    const bool mxfp4 = format == 1;
    struct RestoreDevice {
        int previous = FastllmCudaGetDevice();
        ~RestoreDevice() { FastllmCudaSetDevice(previous); }
    } restoreDevice;
    FastllmCudaSetDevice(devices.front());
    const bool hopper = FastllmCudaRuntimeArch() == 90;
    const bool hopperMxfp4 = hopper && mxfp4;
    const bool ownsPacked = hopperMxfp4 || fastllm_ep8_mega_moe_owns_packed_weights();
    const size_t bytes = hopperMxfp4 ? fastllm_mxfp4_ep_workspace_bytes(devices.size())
                                     : fastllm_ep8_mega_moe_workspace_bytes(devices.size());
    // Dedicated control storage must survive attention/cuBLAS calls. Sharing
    // the old transient scratch would race remote dispatch writes against a
    // later rank's memset.
    // The Hopper MXFP4 adapter reserves independent decode and prefill arenas before serving.
    const size_t budget = (hopperMxfp4 ? 1280ull : 512ull) * 1024 * 1024;
    AssertInFastLLM(bytes > 0 && bytes <= budget,
                   "MiMo EP symmetric workspace exceeds its memory budget.");
    auto state = std::make_shared<FastllmCudaMimoEp8BuiltinMega>();
    state->devices = devices;
    std::vector<const void *> peers;
    for (int r = 0; r < (int)devices.size(); ++r) {
        FastllmCudaSetDevice(devices[r]);
        auto rank = std::make_unique<FastllmCudaMimoEp8BuiltinMega::Rank>();
        rank->hopperMxfp4 = hopperMxfp4;
        // Only Hopper FP8 prefill borrows scratch from peer ranks.
        if (hopper && !mxfp4) rank->prefillReady = FastllmCudaEventCreate();
        rank->workspace.Resize({(int)bytes});
        rank->workspace.ToDevice(DataDevice::CUDA, {devices[r]}, false);
        rank->workspace.Allocate(false);
        const int initialized = hopperMxfp4
            ? fastllm_mxfp4_ep_init(rank->workspace.cudaData, bytes, devices.data(), r, devices.size())
            : fastllm_ep8_mega_moe_init_workspace(rank->workspace.cudaData, bytes, devices.data(), r, devices.size());
        AssertInFastLLM(initialized, hopperMxfp4 ? fastllm_mxfp4_ep_error() : fastllm_ep8_mega_moe_error());
        peers.push_back(rank->workspace.cudaData);
        state->ranks.push_back(std::move(rank));
    }
    for (int r = 0; r < (int)devices.size(); ++r) {
        FastllmCudaSetDevice(devices[r]);
        auto &rank = *state->ranks[r];
        size_t released = 0;
        rank.layers.resize(weights[r].size(), nullptr);
        for (size_t l = 0; l < weights[r].size(); ++l) {
            if (weights[r][l].empty()) continue;
            std::vector<const void *> gate, down, gs, ds;
            for (int e = r * (256 / devices.size()); e < (r + 1) * (256 / devices.size()); ++e) {
                for (int part = 0; part < 2; ++part) {
                    Data *w = weights[r][l][2 + e * 2 + part];
                    const std::vector<int> shape = part ? std::vector<int>{4096, 2048} : std::vector<int>{4096, 4096};
                    AssertInFastLLM(w && w->dims == shape && w->cudaData &&
                        w->dataDeviceIds == std::vector<int>{devices[r]},
                        "MiMo EP requires complete experts on their assigned device.");
                    (part ? down : gate).push_back(w->cudaData);
                    if (mxfp4) continue;
                    AssertInFastLLM(w->scales.size() == (size_t)shape[0] * shape[1] / (128 * 128),
                        "MiMo EP requires complete 128x128 FP8 scales.");
                    // Native MergeMOE creates scales plus an unused zero-bias
                    // allocation for every TP shard. EP8 needs only the scales.
                    auto scale = std::make_unique<Data>(FLOAT32, std::vector<int>{(int)w->scales.size()}, w->scales);
                    scale->ToDevice(DataDevice::CUDA, {devices[r]}, true);
                    (part ? ds : gs).push_back(scale->cudaData);
                    rank.scales.push_back(std::move(scale));
                }
            }
            if (hopperMxfp4) {
                rank.layers[l] = fastllm_mxfp4_ep_create(gate.data(), down.data(), rank.workspace.cudaData,
                    bytes, peers.data(), r, devices.size());
            } else if (mxfp4) {
                rank.layers[l] = fastllm_ep8_mega_moe_create_mxfp4(gate.data(), down.data(),
                    rank.workspace.cudaData, bytes, peers.data(), r, l, devices.size());
            } else {
                rank.layers[l] = fastllm_ep8_mega_moe_create(gate.data(), down.data(), gs.data(), ds.data(),
                    rank.workspace.cudaData, bytes, peers.data(), r, l, devices.size());
            }
            AssertInFastLLM(rank.layers[l] != nullptr,
                hopperMxfp4 ? fastllm_mxfp4_ep_error() : fastllm_ep8_mega_moe_error());
            // create() completes packing before the source layer is retired.
            if (ownsPacked) {
                for (size_t e = r * (256 / devices.size()); e < (r + 1) * (256 / devices.size()); ++e) {
                    released += ReleasePackedMoeSource(*weights[r][l][2 + e * 2]);
                    released += ReleasePackedMoeSource(*weights[r][l][3 + e * 2]);
                }
                rank.scales.clear();
            }
        }
        fprintf(stderr, "[MegaMoE] gpu=%d original_bytes_released=%zu packed_owner=%d\n",
            devices[r], released, ownsPacked ? 1 : 0);
        printf("MIMO_EP_MEGA ranks=%d gpu=%d experts=%d inter=2048 %s workspace=%zu max_rows=%d Graph-ready backend=builtin\n",
               (int)devices.size(), devices[r], (int)(256 / devices.size()), mxfp4 ? "MXFP4-W4A8" : "W8A8", bytes,
               hopperMxfp4 ? fastllm_mxfp4_ep_capacity() : fastllm_ep8_mega_moe_decode_capacity(4096, (int)devices.size()));
    }
    return state;
}
void FastllmCudaMimoEp8BuiltinPrefillReduceScatter(FastllmCudaMimoEp8BuiltinMega &state, int rank,
                                          fastllm::Data &output) {
    using namespace fastllm;
    const int ranks = (int)state.devices.size();
    const uint64_t elements = output.Count(0);
    AssertInFastLLM(rank >= 0 && rank < ranks && elements > 0 && elements % ranks == 0 &&
        elements / ranks <= std::numeric_limits<int>::max() &&
        output.cudaData && output.dataType == BFLOAT16 &&
        output.dataDevice == DataDevice::CUDA &&
        output.dataDeviceIds.size() == 1 && output.dataDeviceIds[0] == state.devices[rank],
        "Invalid MiMo EP prefill ReduceScatter tensor.");
    const int count = (int)(elements / ranks);
    state.WaitForOutputs();
    const bool ok = FastllmNcclReduceScatter(output.cudaData,
        (uint16_t *)output.cudaData + (size_t)rank * count,
        count, output.dataType, state.devices[rank]);
    state.WaitForOutputs();
    AssertInFastLLM(ok, "MiMo EP prefill ReduceScatter failed.");
}
void FastllmCudaMimoEp8BuiltinPrefillAllGather(FastllmCudaMimoEp8BuiltinMega &state, int rank,
                                      fastllm::Data &output) {
    // Gather in place on each worker's compute stream. This eager-only path
    // needs neither zero padding nor another collective arena.
    using namespace fastllm;
    const int ranks = (int)state.devices.size();
    AssertInFastLLM(rank >= 0 && rank < ranks && output.dims.size() == 2 &&
        output.dims[0] > 0 && output.dims[1] > 0 && output.cudaData &&
        output.dataDevice == DataDevice::CUDA &&
        output.dataDeviceIds == std::vector<int>{state.devices[rank]} &&
        (output.dataType == BFLOAT16 || output.dataType == INT32 || output.dataType == FLOAT32),
        "Invalid MiMo EP prefill AllGather tensor.");
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
            "MiMo EP prefill AllGather failed.");
        state.WaitForOutputs();
        return;
    }
    if (ranks == 8) {
        std::vector<int> counts(ranks);
        for (int peer = 0; peer < ranks; ++peer) {
            const int64_t rows = (int64_t)total * (peer + 1) / ranks - (int64_t)total * peer / ranks;
            AssertInFastLLM(rows * output.dims[1] <= std::numeric_limits<int>::max(),
                "MiMo EP prefill AllGather shard is too large.");
            counts[peer] = rows * output.dims[1];
        }
        AssertInFastLLM(FastllmNcclAllGatherVInplace(output.cudaData, counts,
            output.dataType, state.devices[rank]), "MiMo EP prefill AllGatherV failed.");
        return;
    }
    // Four-rank direct copies retain better large-shard bandwidth on H20.
    state.ranks[rank]->output = output.cudaData;
    FastllmCudaSyncCurrentThreadStream();
    state.WaitForOutputs();
    for (int step = 1; step < ranks; ++step) {
        const int peer = (rank + step) % ranks;
        const size_t first = (int64_t)total * peer / ranks;
        const size_t rows = (int64_t)total * (peer + 1) / ranks - first;
        if (!rows) continue;
        AssertInFastLLM(FastllmCudaMemcpyPeerAsyncCurrentThread(state.devices[rank],
            (uint8_t *)output.cudaData + first * rowBytes, state.devices[peer],
            (const uint8_t *)state.ranks[peer]->output + first * rowBytes,
            rows * rowBytes), "MiMo EP peer output copy failed.");
    }
    FastllmCudaSyncCurrentThreadStream();
    state.WaitForOutputs();
}
static int PrepareMimoEpPrefill(FastllmCudaMimoEp8BuiltinMega &state, int rank, void *handle,
        int total, fastllm::Data *prefillScratch, const fastllm::Data &input,
        const void **prefillPeers) {
    using namespace fastllm;
    int capacity = 0;
    const int ranks = state.devices.size();
    // Eligibility uses the common input shape so every rank participates even
    // when one local buffer is unavailable. Router FP32 input is dead here;
    // its caller releases all peer readers before reusing the arena.
    if (FastllmCudaMimoEp8BuiltinPrefillWorkspaceBytes(total, state.devices.size())) {
        auto &local = *state.ranks[rank];
        local.prefillScratch = nullptr;
        int prefillCapacity = 0;
        if (prefillScratch && !prefillScratch->isFake && prefillScratch->dataType == FLOAT32 &&
            prefillScratch->dataDevice == DataDevice::CUDA &&
            prefillScratch->dataDeviceIds == std::vector<int>{state.devices[rank]} &&
            !prefillScratch->dims.empty() && prefillScratch->dims.back() == 4096 &&
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
    return capacity;
}
void FastllmCudaRunMimoEp8BuiltinMegaLocal(FastllmCudaMimoEp8BuiltinMega &state, int rank, int layer,
        const fastllm::Data &input, const fastllm::Data &index, const fastllm::Data &score,
        fastllm::Data &output, int maxRows, const int *liveKeys, fastllm::Data *prefillScratch) {
    using namespace fastllm;
    AssertInFastLLM(input.dataType == BFLOAT16 && input.dims.size() == 2 &&
        input.dims[1] == 4096 && input.dims[0] >= 0 && input.dims[0] <= maxRows && maxRows > 0 &&
        maxRows <= std::numeric_limits<int>::max() / (int)state.devices.size(),
        "Invalid local MiMo EP input shape.");
    const int rows = input.dims[0];
    AssertInFastLLM(index.dims == std::vector<int>({rows, 8}) && score.dims == index.dims,
        "Invalid local MiMo EP routing shape.");
    output.dataType = BFLOAT16;
    // Empty sources still join the EP barriers. Keep a valid scratch pointer
    // for their zero-row dispatch; no token payload is read or written.
    output.Resize({std::max(1, rows), 4096});
    output.ToDevice(DataDevice::CUDA, {state.devices.at(rank)}, false);
    output.Allocate(false);
    output.Resize(input.dims);
    const auto &local = *state.ranks.at(rank);
    void *handle = local.layers.at(layer);
    AssertInFastLLM(handle != nullptr, "Missing local MiMo EP layer handle.");
    int capacity = local.hopperMxfp4 ? (maxRows > fastllm_mxfp4_ep_capacity()
        ? fastllm_mxfp4_ep_prefill_capacity() : fastllm_mxfp4_ep_capacity())
        : fastllm_ep8_mega_moe_decode_capacity(maxRows * state.devices.size(), state.devices.size());
    const void *prefillPeers[8]{};
    if (prefillScratch && local.prefillReady) {
        // Use a common maximum, including idle sources and unequal DP loads.
        // Decode/Graph calls never enter these eager workspace rendezvous.
        const int prefillCapacity = PrepareMimoEpPrefill(state, rank, handle,
            maxRows * state.devices.size(), prefillScratch, input, prefillPeers);
        if (prefillCapacity) capacity = prefillCapacity;
    }
    for (int first = 0; first < maxRows; first += capacity) {
        const int start = std::min(first, rows), count = std::max(0, std::min(capacity, rows - start));
        const void *x = rows ? (const uint16_t *)input.cudaData + (size_t)start * 4096 : output.cudaData;
        void *y = (uint16_t *)output.cudaData + (size_t)start * 4096;
        const int32_t *idx = rows ? (const int32_t *)index.cudaData + start * 8 : (const int32_t *)output.cudaData;
        const float *scores = rows ? (const float *)score.cudaData + start * 8 : (const float *)output.cudaData;
        const int ok = local.hopperMxfp4
            ? fastllm_mxfp4_ep_run(handle, x, y, idx, scores, count, std::min(capacity, maxRows - first),
                liveKeys ? liveKeys + start : nullptr)
            : prefillPeers[0]
            ? fastllm_ep8_mega_moe_run_prefill(handle, prefillPeers, x, y, idx, scores, count)
            : fastllm_ep8_mega_moe_run_batched(handle, x, y, idx, scores, count, capacity,
                liveKeys ? liveKeys + start : nullptr);
        if (!ok)
            throw std::runtime_error(std::string("Local MiMo EP launch failed: ") +
                (local.hopperMxfp4 ? fastllm_mxfp4_ep_error() : fastllm_ep8_mega_moe_error()));
    }
    if (prefillPeers[0]) {
        // The next layer overwrites borrowed router scratch and resets EP
        // control data. A fast/empty source must not race slower peer readers.
        FastllmCudaEventRecordCurrentThread(state.ranks[rank]->prefillReady);
        state.WaitForOutputs();
        for (const auto &peer : state.ranks)
            FastllmCudaCurrentThreadStreamWaitEvent(peer->prefillReady);
        state.WaitForOutputs();
    }
}
bool FastllmCudaRunMimoEp8BuiltinMega(FastllmCudaMimoEp8BuiltinMega &state, int rank, int layer,
        const fastllm::Data &input, const fastllm::Data &index, const fastllm::Data &score,
        fastllm::Data &output, bool graphDecode, fastllm::Data *residual,
        const fastllm::Data *normWeight, fastllm::Data *normOutput, float eps,
        fastllm::Data *prefillScratch, const int *liveKeys) {
    using namespace fastllm;
    AssertInFastLLM(input.dataType == BFLOAT16 && input.dims.size() == 2 &&
        input.dims[1] == 4096, "Invalid MiMo EP input shape.");
    const int ranks = (int)state.devices.size();
    const int total = input.dims[0];
    const int begin = (int)((int64_t)total * rank / ranks);
    const int shard = (int)((int64_t)total * (rank + 1) / ranks) - begin;
    const bool localRouting = total >= ranks &&
        index.dims == std::vector<int>({shard, 8});
    AssertInFastLLM((localRouting || index.dims == std::vector<int>({total, 8})) &&
        score.dims == index.dims, "Invalid MiMo EP routing shape.");
    output.dataType = BFLOAT16;
    output.Resize(input.dims);
    output.ToDevice(DataDevice::CUDA, {state.devices.at(rank)}, false);
    output.Allocate(false);
    const auto &local = *state.ranks.at(rank);
    void *handle = local.layers.at(layer);
    AssertInFastLLM(handle != nullptr, "Missing MiMo EP layer handle.");
    int capacity = local.hopperMxfp4 ? (graphDecode ? fastllm_mxfp4_ep_capacity() : fastllm_mxfp4_ep_prefill_capacity())
        : fastllm_ep8_mega_moe_decode_capacity(total, ranks);
    const void *prefillPeers[8]{};
    if (!graphDecode && local.prefillReady) {
        const int prefillCapacity = PrepareMimoEpPrefill(state, rank, handle,
            total, prefillScratch, input, prefillPeers);
        if (prefillCapacity) capacity = prefillCapacity;
    }
    auto run = [&](int start, int rows, int commonRows) {
        const int routeStart = localRouting ? start - begin : start;
        const void *x = (const uint16_t *)input.cudaData + (size_t)start * 4096;
        void *y = (uint16_t *)output.cudaData + (size_t)start * 4096;
        const int32_t *idx = (const int32_t *)index.cudaData + routeStart * 8;
        const float *scores = (const float *)score.cudaData + routeStart * 8;
        const int ok = local.hopperMxfp4 ?
            fastllm_mxfp4_ep_run(handle, x, y, idx, scores, rows, commonRows, liveKeys ? liveKeys + start : nullptr) :
            !prefillPeers[0] ?
            fastllm_ep8_mega_moe_run_batched(handle, x, y, idx, scores, rows, capacity, liveKeys ? liveKeys + start : nullptr) :
            fastllm_ep8_mega_moe_run_prefill(handle, prefillPeers, x, y, idx, scores, rows);
        if (!ok)
            throw std::runtime_error(std::string("MiMo EP launch failed: ") +
                (local.hopperMxfp4 ? fastllm_mxfp4_ep_error() : fastllm_ep8_mega_moe_error()));
    };
    const int replicatedCapacity = local.hopperMxfp4 ? fastllm_mxfp4_ep_capacity() : capacity;
    if (total <= replicatedCapacity && !localRouting) {
        run(0, total, total);
        return false;
    }

    // Dispatch each token from one rank, using the selected bounded arena.
    // All ranks execute the same number of rounds, including empty shards.
    const int rounds = (int)(((int64_t)total + ranks * capacity - 1) / (ranks * capacity));
    for (int round = 0; round < rounds; ++round) {
        const int start = round * capacity;
        const int commonRows = std::min(capacity, (total + ranks - 1) / ranks - start);
        run(begin + std::min(start, shard), std::max(0, std::min(capacity, shard - start)), commonRows);
    }
    if (graphDecode) {
        if (localRouting && residual && normWeight && normOutput) {
            AssertInFastLLM(residual->dataType == BFLOAT16 && residual->Count(0) == total * 4096 &&
                normWeight->dataType == FLOAT32 && normWeight->Count(0) == 4096,
                "Invalid MiMo EP residual/RMSNorm shape.");
            normOutput->dataType = BFLOAT16;
            normOutput->Resize(residual->dims);
            normOutput->ToDevice(output.dataDevice, output.dataDeviceIds, false);
            normOutput->Allocate(false);
            if (FastllmCudaCustomAllGatherAddRMSNorm(output.cudaData, residual->cudaData,
                    (const float *)normWeight->cudaData, normOutput->cudaData,
                    total * 4096, 4096, eps, BFLOAT16, state.devices[rank])) return true;
        }
        if (localRouting && total % ranks == 0 && FastllmCudaCustomAllGatherRows(output.cudaData,
                total * 4096, 4096, BFLOAT16, state.devices[rank]))
            return false;
        // Each rank has written its slice in-place. NCCL records the exchange
        // on the same stream, using the persistent bounded MoE workspace.
        if (total % ranks == 0) {
            AssertInFastLLM(FastllmNcclAllGather(
                (const uint16_t *)output.cudaData + (size_t)begin * 4096,
                output.cudaData, shard * 4096, BFLOAT16, state.devices[rank]),
                "MiMo EP decode AllGather failed.");
        } else {
            // Graph-safe fallback for unequal owner slices. All ranks enqueue
            // the same broadcasts; no host rendezvous or temporary tensor.
            for (int peer = 0; peer < ranks; ++peer) {
                const int first = (int64_t)total * peer / ranks;
                const int count = (int64_t)total * (peer + 1) / ranks - first;
                if (count) FastllmNcclBroadcast(
                    (uint16_t *)output.cudaData + (size_t)first * 4096,
                    count * 4096, BFLOAT16, state.devices[peer], state.devices[rank]);
            }
        }
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
        FastllmCudaMimoEp8BuiltinPrefillAllGather(state, rank, gathered);
        return true;
    }
    FastllmCudaMimoEp8BuiltinPrefillAllGather(state, rank, output);
    return false;
}
#else
struct FastllmCudaMimoEp8BuiltinMega {};
bool FastllmCudaMimoEp8BuiltinMegaWeights(FastllmCudaMimoEp8BuiltinMega &, int, int, const void **) { return false; }
bool FastllmCudaMimoEp8BuiltinMegaOwnsPackedWeights(int) { return false; }
void FastllmCudaRunMimoEp8BuiltinMegaLocal(FastllmCudaMimoEp8BuiltinMega &, int, int, const fastllm::Data &,
    const fastllm::Data &, const fastllm::Data &, fastllm::Data &, int, const int *, fastllm::Data *) {
    throw std::runtime_error("MiMo EP MegaMoE local rows require the SM90 backend.");
}
void FastllmCudaMimoEp8BuiltinPrefillReduceScatter(FastllmCudaMimoEp8BuiltinMega &, int, fastllm::Data &) {
    throw std::runtime_error("MiMo EP MegaMoE is unavailable.");
}
bool FastllmCudaMimoEp8BuiltinMegaSupported(const std::vector<int> &) { return false; }
size_t FastllmCudaMimoEp8BuiltinPrefillWorkspaceBytes(int, int) { return 0; }
std::shared_ptr<FastllmCudaMimoEp8BuiltinMega> FastllmCudaCreateMimoEp8BuiltinMega(const std::vector<int> &,
    const std::vector<std::vector<std::vector<fastllm::Data *>>> &) { return {}; }
void FastllmCudaMimoEp8BuiltinPrefillAllGather(FastllmCudaMimoEp8BuiltinMega &, int, fastllm::Data &) {
    throw std::runtime_error("MiMo EP prefill AllGather requires the SM90 backend.");
}
bool FastllmCudaRunMimoEp8BuiltinMega(FastllmCudaMimoEp8BuiltinMega &, int, int, const fastllm::Data &,
    const fastllm::Data &, const fastllm::Data &, fastllm::Data &, bool, fastllm::Data *,
    const fastllm::Data *, fastllm::Data *, float, fastllm::Data *, const int *) {
    throw std::runtime_error("MiMo EP MegaMoE requires a Linux CUDA build containing the SM90 backend.");
}
#endif
