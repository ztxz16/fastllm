// MiMo EP8 keeps exactly one full copy of each expert across the eight ranks.
// Attention remains TP8. Large inputs are partitioned into disjoint token
// ranges before EP dispatch, then merged in-place into the existing output.
// Small or uneven decode batches and verification retain replicated dispatch.
#include "devices/cuda/fastllm-mimo-ep8-mega.h"
#include "devices/cuda/fastllm-mimo-ep8-mega-kernel.h"
#if defined(FASTLLM_ENABLE_MIMO_EP8_MEGA) && !defined(_WIN32) && !defined(USE_ROCM)
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <chrono>
#include <condition_variable>
#include <mutex>
struct FastllmCudaMimoEp8Mega {
    struct Rank {
        fastllm::Data workspace{fastllm::INT8};
        std::vector<std::unique_ptr<fastllm::Data>> scales;
        std::vector<void *> layers;
        void *output = nullptr; // Borrowed only during the prefill exchange.
        ~Rank() { for (void *layer : layers) if (layer) fastllm_ep8_mega_moe_destroy(layer); }
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
bool FastllmCudaRunMimoEp8Mega(FastllmCudaMimoEp8Mega &state, int rank, int layer,
        const fastllm::Data &input, const fastllm::Data &index, const fastllm::Data &score,
        fastllm::Data &output, bool graphDecode, fastllm::Data *residual,
        const fastllm::Data *normWeight, fastllm::Data *normOutput, float eps) {
    using namespace fastllm;
    AssertInFastLLM(input.dataType == BFLOAT16 && input.dims.size() == 2 &&
        input.dims[1] == 4096, "Invalid MiMo EP8 input shape.");
    const int ranks = (int)state.devices.size();
    const int total = input.dims[0];
    const int begin = (int)((int64_t)total * rank / ranks);
    const bool localRouting = total >= ranks && total % ranks == 0 &&
        index.dims == std::vector<int>({total / ranks, 8});
    AssertInFastLLM((localRouting || index.dims == std::vector<int>({total, 8})) &&
        score.dims == index.dims, "Invalid MiMo EP8 routing shape.");
    output.dataType = BFLOAT16;
    output.Resize(input.dims);
    output.ToDevice(DataDevice::CUDA, {state.devices.at(rank)}, false);
    output.Allocate(false);
    void *handle = state.ranks.at(rank)->layers.at(layer);
    AssertInFastLLM(handle != nullptr, "Missing MiMo EP8 layer handle.");
    const int capacity = fastllm_ep8_mega_moe_max_rows();
    auto run = [&](int start, int rows) {
        const int routeStart = localRouting ? start - begin : start;
        if (!fastllm_ep8_mega_moe_run(handle, (const uint16_t *)input.cudaData + (size_t)start * 4096,
            (uint16_t *)output.cudaData + (size_t)start * 4096,
            (const int32_t *)index.cudaData + routeStart * 8,
            (const float *)score.cudaData + routeStart * 8, rows))
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
    // Dispatch each token from one rank, using the existing 64-row arena.
    // All ranks execute the same number of rounds, including empty shards.
    const int shard = (int)((int64_t)total * (rank + 1) / ranks) - begin;
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
    // Publish completed shards, then gather directly into the existing output.
    // All sources stay alive until every peer finishes reading them. This
    // eager-only path needs neither zero padding nor another collective arena.
    state.ranks[rank]->output = output.cudaData;
    FastllmCudaSyncCurrentThreadStream();
    state.WaitForOutputs();
    for (int step = 1; step < ranks; ++step) {
        const int peer = (rank + step) % ranks;
        const size_t first = (int64_t)total * peer / ranks;
        const size_t rows = (int64_t)total * (peer + 1) / ranks - first;
        AssertInFastLLM(FastllmCudaMemcpyPeerAsyncCurrentThread(state.devices[rank],
            (uint16_t *)output.cudaData + first * 4096, state.devices[peer],
            (const uint16_t *)state.ranks[peer]->output + first * 4096,
            rows * 4096 * sizeof(uint16_t)), "MiMo EP8 peer output copy failed.");
    }
    FastllmCudaSyncCurrentThreadStream();
    state.WaitForOutputs();
    return false;
}
#else
struct FastllmCudaMimoEp8Mega {};
bool FastllmCudaMimoEp8MegaSupported(const std::vector<int> &) { return false; }
std::shared_ptr<FastllmCudaMimoEp8Mega> FastllmCudaCreateMimoEp8Mega(const std::vector<int> &,
    const std::vector<std::vector<std::vector<fastllm::Data *>>> &) { return {}; }
bool FastllmCudaRunMimoEp8Mega(FastllmCudaMimoEp8Mega &, int, int, const fastllm::Data &,
    const fastllm::Data &, const fastllm::Data &, fastllm::Data &, bool, fastllm::Data *,
    const fastllm::Data *, fastllm::Data *, float) {
    throw std::runtime_error("MiMo EP8 MegaMoE requires a Linux CUDA build containing the SM90 backend.");
}
#endif
