#include "clover.h"
#include "utils/persistent_worker_group.h"
#include "executor.h"
#include "utils/hybrid_moe.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <set>
#include <sstream>
#include <stdexcept>
#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cuda_runtime.h>
#endif

namespace fastllm {
namespace {
void RequireTP(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(std::string("Clover TP: ") + message);
}
// Slice each packed segment independently, preserving the checkpoint layout.
// This also handles the expert axis of [E, 2I, H] and [E, H, I].
void SliceWeight(const Data &source, Data &dest, int axis,
        const std::vector<std::pair<int, int>> &ranges) {
    RequireTP(source.cpuData && source.unitSizeDiv == 1, "expected unpacked host weights");
    dest.dataType = source.dataType;
    dest.UpdateUnitSize();
    auto shape = source.dims;
    if (axis < 0) {
        dest.Resize(shape); dest.Allocate(false);
        std::memcpy(dest.cpuData, source.cpuData, source.GetBytes());
    } else {
        int columns = 0;
        for (auto range : ranges) {
            RequireTP(range.first >= 0 && range.second > range.first && range.second <= shape[axis], "invalid weight slice");
            columns += range.second - range.first;
        }
        shape[axis] = columns;
        dest.Resize(shape); dest.Allocate(false);
        size_t outer = 1, inner = source.unitSize;
        for (int i = 0; i < axis; ++i) outer *= shape[i];
        for (int i = axis + 1; i < int(shape.size()); ++i) inner *= shape[i];
        auto *out = dest.cpuData;
        for (size_t i = 0; i < outer; ++i) for (auto range : ranges) {
            size_t bytes = size_t(range.second - range.first) * inner;
            std::memcpy(out, source.cpuData + (i * source.dims[axis] + range.first) * inner, bytes);
            out += bytes;
        }
    }
    dest.name = source.name;
    dest.isModelWeight = true;
    dest.weightType = source.weightType;
}
}

struct CloverModel::Parallel {
    std::vector<std::unique_ptr<CloverModel>> ranks;
    std::vector<std::unique_ptr<Executor>> executors;
    PersistentWorkerGroup workers;
    std::atomic<bool> failed{false};
    uint64_t generation = 0;
    ~Parallel() { workers.Stop(); }
};

bool CloverModel::InitParallel() {
#ifdef USE_CUDA
    auto placement = deviceMap.empty() ? GetDeviceMap() : deviceMap;
    std::vector<int> devices;
    for (auto &entry : placement) {
        auto name = entry.first;
        if (name == "cuda") name = "cuda:0";
        auto colon = name.find(':');
        RequireTP(colon != std::string::npos && (name.substr(0, colon) == "cuda" ||
            name.substr(0, colon) == "multicuda"), "requires CUDA placement");
        std::stringstream stream(name.substr(colon + 1));
        std::string token;
        while (std::getline(stream, token, ',')) {
            RequireTP(!token.empty() && std::all_of(token.begin(), token.end(),
                [](char c) { return c >= '0' && c <= '9'; }), "expected CUDA ordinals without ratios");
            devices.push_back(std::stoi(token));
        }
    }
    if (devices.size() <= 1) return false;
    RequireTP(std::set<int>(devices.begin(), devices.end()).size() == devices.size(), "duplicate device");
    const int size = devices.size();
    RequireTP(num_attention_heads % size == 0 && keyHeads % size == 0 && valueHeads % size == 0 &&
        intermediate % size == 0 && vocabSize % size == 0 &&
        (num_key_value_heads % size == 0 || size % num_key_value_heads == 0), "unsupported TP geometry");
    for (int i = 0; i < block_cnt; ++i) {
        const auto moe = SelectMoeDeviceForLayer(i);
        RequireTP(!IsSharedHostMoeDevice(moe), "TP with host experts is not supported");
    }
    // All ranks must use the same activation/reduction precision.
    for (int device : devices) {
        int major = 0;
        RequireTP(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) == cudaSuccess,
            "cannot query device capability");
        if (major < 8) dataType = kvCacheDataType = FLOAT16;
    }
    auto runtime = std::make_shared<Parallel>();
    for (auto &item : weight.weight) item.second.ToDevice(DataDevice::CPU);
    for (int rank = 0; rank < size; ++rank) {
        auto local = std::make_unique<CloverModel>();
        local->weight.dicts = weight.dicts;
        local->InitParams();
        local->max_positions = max_positions;
        local->dataType = local->kvCacheDataType = dataType;
        local->rankLocal = true; local->tpRank = rank; local->tpSize = size;
        local->cudaDevice = devices[rank];
        local->deviceMap = {{"cuda:" + std::to_string(devices[rank]), 1}};
        local->num_attention_heads /= size;
        local->num_key_value_heads = std::max(1, num_key_value_heads / size);
        local->keyHeads /= size; local->valueHeads /= size; local->intermediate /= size;
        for (auto &item : weight.weight) {
            const auto &name = item.first;
            const auto &source = item.second;
            int axis = -1;
            std::vector<std::pair<int, int>> ranges;
            auto split = [&](int dim, std::vector<int> segments, int index, int parts) {
                axis = dim;
                int offset = 0;
                for (int length : segments) {
                    RequireTP(length % parts == 0, "weight dimension is not divisible by TP");
                    ranges.emplace_back(offset + index * (length / parts), offset + (index + 1) * (length / parts));
                    offset += length;
                }
                RequireTP(offset == source.dims.at(dim), "checkpoint weight shape does not match config");
            };
            auto splitAxis = [&](int dim) { split(dim, {source.dims.at(dim)}, rank, size); };
            if (name == "lm_head.weight") splitAxis(0);
            else if (name.find(".linear_attn.") != std::string::npos) {
                auto suffix = name.substr(name.find(".linear_attn.") + 13);
                if (suffix == "in_proj_qkv.weight" || suffix == "conv1d.weight" || suffix == "conv_right_weight")
                    split(0, {keyHeads * 128, keyHeads * 128, valueHeads * 128}, rank, size);
                else if (suffix == "out_proj.weight") splitAxis(1);
                else if (suffix != "norm.weight") splitAxis(0);
            } else if (name.find(".self_attn.") != std::string::npos) {
                if (name.find("q_proj.weight") != std::string::npos) splitAxis(0);
                else if (name.find("k_proj.weight") != std::string::npos || name.find("v_proj.weight") != std::string::npos) {
                    int parts = std::min(size, num_key_value_heads);
                    split(0, {num_key_value_heads * head_dim}, rank / (size / parts), parts);
                } else if (name.find("o_proj.weight") != std::string::npos) splitAxis(1);
            } else if (name.find(".mlp.experts.gate_up_proj") != std::string::npos)
                split(1, {intermediate, intermediate}, rank, size);
            else if (name.find(".mlp.experts.down_proj") != std::string::npos) splitAxis(2);
            else if (name.find(".mlp.shared_expert.") != std::string::npos)
                splitAxis(name.find("down_proj") != std::string::npos ? 1 : 0);
            SliceWeight(source, local->weight[name], axis, ranges);
        }
        // Upload/merge serially: only forward work needs concurrent ranks.
        local->EnsureResident();
        auto executor = std::make_unique<Executor>();
        executor->SetFirstDevice("cuda:" + std::to_string(devices[rank]));
        runtime->executors.push_back(std::move(executor));
        runtime->ranks.push_back(std::move(local));
    }
    RequireTP(FastllmInitNccl(devices), "NCCL initialization failed");
    runtime->generation = FastllmGetNcclGeneration();
    parallelDevices = devices; cudaDevice = devices[0];
    parallel = std::move(runtime); resident = true;
    FastllmCudaSetDevice(cudaDevice);
    return true;
#else
    return false;
#endif
}

void CloverModel::RunParallel(const std::function<void(CloverModel &, int)> &task) {
#ifdef USE_CUDA
    RequireTP(parallel && !parallel->failed, "parallel runtime is unavailable after a rank failure");
    RequireTP(parallel->generation == FastllmGetNcclGeneration(), "NCCL group changed after initialization");
    std::vector<std::exception_ptr> errors(parallelDevices.size());
    parallel->workers.RunWithCaller(parallelDevices, [&](int rank) {
        struct RestoreExecutor {
            void *previous = GetExecutor();
            ~RestoreExecutor() { SetCurrentThreadExecutor(previous); }
        } restore;
        SetCurrentThreadExecutor(parallel->executors[rank].get());
        try {
            auto &local = *parallel->ranks[rank];
            FastllmCudaSetDevice(local.cudaDevice);
            FastllmCudaClearThreadError();
            task(local, rank);
            RequireTP(!FastllmCudaGetThreadError() && cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess,
                "rank execution failed");
        } catch (...) {
            if (!parallel->failed.exchange(true)) FastllmNcclAbort("Clover rank execution failed");
            throw;
        }
    }, errors);
    FastllmCudaSetDevice(cudaDevice);
    for (auto error : errors) if (error) std::rethrow_exception(error);
#endif
}

CloverModel::CacheView CloverModel::ViewCache(std::vector<std::pair<Data, Data>> &cache, int device) {
    RequireTP(cache.empty() || int(cache.size()) == block_cnt, "cache layer count mismatch");
    if (cache.empty()) cache.resize(block_cnt);
    CacheView view;
    for (auto &layer : cache) {
        auto select = [&](Data &data) -> Data * {
            if (device < 0) return &data;
            RequireTP(data.multiDeviceData || data.dims.empty(), "cannot reuse a TP1 cache in TP");
            data.multiDeviceData = true;
            auto &shard = data.multiDeviceDatas[device];
            if (!shard) {
                shard = new Data();
                shard->isKVCache = data.isKVCache;
                shard->isLinearAttention = data.isLinearAttention;
            }
            return shard;
        };
        view.emplace_back(select(layer.first), select(layer.second));
    }
    return view;
}
void CloverModel::PublishCache(std::vector<std::pair<Data, Data>> &cache) {
    if (!parallel) return;
    for (auto &layer : cache) for (Data *data : {&layer.first, &layer.second}) {
        const auto &shard = *data->multiDeviceDatas.at(cudaDevice);
        // The wrapper carries the logical length; each shard owns its storage
        // and its local head dimensions, including replicated GQA KV heads.
        data->dataType = shard.dataType; data->UpdateUnitSize();
        // Full-attention layers have no rollback snapshot tensor. Resize({})
        // is not a valid Data operation (it requires at least one stride).
        if (shard.dims.empty()) { data->dims.clear(); data->strides.clear(); }
        else data->Resize(shard.dims);
    }
}

void CloverModel::Reduce(Data &data) {
#ifdef USE_CUDA
    if (tpSize == 1) return;
    // Reuse the common thread-TP backend. Keep reductions outside MoE graphs.
    if (!FastllmTryTP2P2PAllReduce(data.cudaData, data.cudaData,
            data.Count(0), data.dataType, cudaDevice))
        FastllmNcclAllReduceNoCustom(data.cudaData, data.cudaData, data.Count(0), data.dataType, cudaDevice);
    RequireTP(!FastllmCudaGetThreadError(), "AllReduce failed");
#endif
}
void CloverModel::GatherLogits(Data &data) {
#ifdef USE_CUDA
    if (tpSize == 1) return;
    const int rows = data.Count(0) / data.dims.back(), localVocab = data.dims.back();
    Data gathered(data.dataType, {tpSize, rows, localVocab});
    gathered.ToDevice(DataDevice::CUDA, std::vector<int>{cudaDevice}, false); gathered.Allocate(false);
    RequireTP(FastllmNcclAllGather(data.cudaData, gathered.cudaData, data.Count(0), data.dataType, cudaDevice),
        "logits AllGather failed");
    if (tpRank == 0) {
        // Use the common CUDA block transpose. Generic Permute currently
        // falls back to CPU; request views of its result must stay on CUDA.
        gathered.Reshape({1, tpSize, rows, localVocab});
        data.Resize({1, rows, tpSize, localVocab}); data.Allocate(false);
        RequireTP(FastllmCudaPermuteTo(gathered, data, {0, 2, 1, 3}), "logits transpose failed");
        data.Reshape({rows, tpSize * localVocab});
    }
#endif
}

void CloverModel::ForwardLogits(const std::vector<int> &tokens,
        std::vector<std::pair<Data, Data>> &cache, bool preview, Data &logits,
        bool lastOnly, CloverSnapshots *snapshots, bool needLogits) {
    EnsureResident();
    if (!parallel) {
        auto view = ViewCache(cache);
        SnapshotView snap;
        if (snapshots) { snap.layers = ViewCache(snapshots->layers); snap.compact = snapshots->compact; }
        ForwardLogitsLocal(tokens, view, preview, logits, lastOnly, snapshots ? &snap : nullptr, needLogits);
        if (snapshots) { snapshots->position = snap.position; snapshots->tokens = snap.tokens; }
        return;
    }
    std::vector<CacheView> views;
    std::vector<SnapshotView> snaps(parallelDevices.size());
    for (int r = 0; r < int(parallelDevices.size()); ++r) {
        views.push_back(ViewCache(cache, parallelDevices[r]));
        if (snapshots) {
            snaps[r].layers = ViewCache(snapshots->layers, parallelDevices[r]);
            snaps[r].compact = snapshots->compact;
        }
    }
    RunParallel([&](CloverModel &local, int rank) {
        Data unused;
        local.ForwardLogitsLocal(tokens, views[rank], preview, rank == 0 ? logits : unused,
            lastOnly, snapshots ? &snaps[rank] : nullptr, needLogits);
    });
    PublishCache(cache);
    if (snapshots) {
        snapshots->position = snaps[0].position; snapshots->tokens = snaps[0].tokens;
        PublishCache(snapshots->layers);
    }
}
void CloverModel::ForwardLogitsBatch(const std::vector<std::vector<int>> &tokens,
        const std::vector<std::vector<std::pair<Data, Data>> *> &caches, bool preview,
        std::vector<Data> &logits, const std::vector<CloverSnapshots *> &snapshots) {
    EnsureResident();
    RequireTP(caches.size() == tokens.size() && (snapshots.empty() || snapshots.size() == tokens.size()), "invalid batch");
    for (size_t b = 0; b < caches.size(); ++b) {
        RequireTP(caches[b] && (snapshots.empty() || snapshots[b]), "missing batch cache");
        for (size_t p = 0; p < b; ++p) RequireTP(caches[b] != caches[p] &&
            (snapshots.empty() || snapshots[b] != snapshots[p]), "batch caches must be independent");
    }
    int ranks = parallel ? parallelDevices.size() : 1;
    std::vector<std::vector<CacheView>> views(ranks);
    std::vector<std::vector<SnapshotView>> snaps(ranks);
    for (int r = 0; r < ranks; ++r) {
        snaps[r].resize(caches.size());
        for (size_t b = 0; b < caches.size(); ++b) {
            int device = parallel ? parallelDevices[r] : -1;
            views[r].push_back(ViewCache(*caches[b], device));
            if (!snapshots.empty()) {
                snaps[r][b].layers = ViewCache(snapshots[b]->layers, device);
                snaps[r][b].compact = snapshots[b]->compact;
            }
        }
    }
    auto run = [&](CloverModel &local, int rank) {
        std::vector<CacheView *> cv; std::vector<SnapshotView *> sv;
        for (size_t b = 0; b < caches.size(); ++b) {
            cv.push_back(&views[rank][b]);
            if (!snapshots.empty()) sv.push_back(&snaps[rank][b]);
        }
        std::vector<Data> unused;
        local.ForwardLogitsBatchLocal(tokens, cv, preview, rank == 0 ? logits : unused, sv);
    };
    if (parallel) RunParallel(run); else run(*this, 0);
    for (size_t b = 0; b < caches.size(); ++b) {
        PublishCache(*caches[b]);
        if (!snapshots.empty()) {
            snapshots[b]->position = snaps[0][b].position; snapshots[b]->tokens = snaps[0][b].tokens;
            PublishCache(snapshots[b]->layers);
        }
    }
}
void CloverModel::RestorePrefix(std::vector<std::pair<Data, Data>> &cache,
        const CloverSnapshots &snapshots, int acceptedInputs) {
    RestorePrefixBatch({&cache}, {&snapshots}, {acceptedInputs});
}
void CloverModel::RestorePrefixBatch(const std::vector<std::vector<std::pair<Data, Data>> *> &caches,
        const std::vector<const CloverSnapshots *> &snapshots, const std::vector<int> &acceptedInputs) {
    RequireTP(caches.size() == snapshots.size() && caches.size() == acceptedInputs.size(),
        "invalid prefix batch");
    std::vector<size_t> active;
    for (size_t b = 0; b < caches.size(); ++b) {
        RequireTP(caches[b] && snapshots[b] && acceptedInputs[b] > 0 &&
            acceptedInputs[b] <= snapshots[b]->tokens && int(caches[b]->size()) == block_cnt &&
            caches[b]->size() == snapshots[b]->layers.size(), "invalid accepted prefix");
        for (size_t previous = 0; previous < b; ++previous)
            RequireTP(caches[previous] != caches[b], "request caches must be independent");
        if (acceptedInputs[b] != snapshots[b]->tokens) active.push_back(b);
    }
    if (active.empty()) return;
    int ranks = parallel ? parallelDevices.size() : 1;
    std::vector<std::vector<CacheView>> views(ranks);
    std::vector<std::vector<SnapshotView>> snaps(ranks);
    for (int r = 0; r < ranks; ++r) {
        int device = parallel ? parallelDevices[r] : -1;
        views[r].reserve(active.size()); snaps[r].resize(active.size());
        for (size_t i = 0; i < active.size(); ++i) {
            size_t b = active[i];
            views[r].push_back(ViewCache(*caches[b], device));
            auto &snap = snaps[r][i];
            snap.position = snapshots[b]->position; snap.tokens = snapshots[b]->tokens;
            snap.layers.reserve(block_cnt);
            // Restore reads snapshots only. No map insertion or metadata mutation.
            for (const auto &layer : snapshots[b]->layers) {
                auto select = [&](const Data &d) { return device < 0 ? const_cast<Data *>(&d) : d.multiDeviceDatas.at(device); };
                snap.layers.emplace_back(select(layer.first), select(layer.second));
            }
        }
    }
    // Reuse the existing full/compact state kernels on every architecture.
    // Independent requests share one worker dispatch and one completion wait.
    auto run = [&](CloverModel &local, int rank) {
        for (size_t i = 0; i < active.size(); ++i)
            local.RestorePrefixLocal(views[rank][i], snaps[rank][i], acceptedInputs[active[i]]);
    };
    if (parallel) RunParallel(run); else run(*this, 0);
    for (size_t b : active) PublishCache(*caches[b]);
}
}
