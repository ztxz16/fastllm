#include "devices/cuda/naive-n05-draft-backend.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "fastllm.h"
#ifndef USE_ROCM
#include "devices/multicuda/fastllm-draft-verify.h"
#endif
#include "utils/utils.h"
#include "devices/cuda/fastllm-cuda-mtp.cuh"
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
#include "devices/cuda/naive-n05-draft-fused.h"
#include "devices/cuda/naive-n05-cuda.cuh"
#include <algorithm>
#include <cmath>
#endif

namespace fastllm {

bool PreferCudaDraftReplicaMatMul(const Data &input, const Data &weight) {
    return input.dataType == BFLOAT16 && weight.dataType == BFLOAT16 && input.dims.size() >= 2 &&
        input.dims[1] >= 2 && input.dims[1] <= 7 && FastllmCudaRuntimeArch() >= 80;
}

namespace {
#ifndef USE_ROCM
bool Resident(const Data &x, int device, DataType type, const std::vector<int> &shape, int alignment = 1) {
    if (x.dataType != type || x.dims != shape || x.dataDevice != CUDA || x.multiDeviceData ||
        x.dataDeviceIds.size() != 1 || x.dataDeviceIds.front() != device || !x.cudaData ||
        reinterpret_cast<uintptr_t>(x.cudaData) % alignment || x.strides.size() != shape.size()) return false;
    uint64_t stride = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        if (x.strides[i] != stride) return false;
        stride *= shape[i];
    }
    return true;
}

#endif
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
bool CanWrite(const Data &x, int device, const std::vector<int> &shape, DataType type = BFLOAT16) {
    return (!x.cudaData && (x.dims.empty() || x.dims == shape) && x.expansionDims.empty() &&
            !x.multiDeviceData && !x.isFake) ||
        Resident(x, device, type, shape, 16);
}

void AllocateOutput(Data &x, int device, const std::vector<int> &shape, DataType type = BFLOAT16) {
    if (x.cudaData) return;
    x.dataType = type;
    x.UpdateUnitSize();
    x.Resize(shape);
    x.ToDevice(CUDA, {device}, false);
    x.Allocate(false);
}

constexpr int MaxBodyRequests = 64;
#endif
} // namespace

void ProjectCudaDraftHeadShard(const DraftVocabShard &shard, const Data &local, int hidden,
        const std::vector<const void *> &sources, const std::vector<int> &offsets,
        const std::vector<std::size_t> &bytes) {
    const int batch = sources.size();
    Data view;
    const bool borrow = batch == 1 && local.cudaData == sources.front() &&
        local.dataDevice == CUDA && local.dataDeviceIds.size() == 1 && local.dataDeviceIds.front() == shard.device;
    Data &input = borrow ? view : shard.staging;
    const std::vector<int> shape{1, offsets.back(), hidden};
    if (borrow) {
        input.FakeFrom(local, 0);
        input.Resize(shape);
    } else {
        input.dataType = BFLOAT16; input.UpdateUnitSize(); input.Resize(shape);
        input.ToDevice(CUDA, {shard.device}, false); input.Allocate(false);
        std::vector<void *> destinations(batch);
        for (int i = 0; i < batch; ++i)
            destinations[i] = static_cast<uint16_t *>(input.cudaData) + size_t(offsets[i]) * hidden;
        AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(destinations.data(),
            sources.data(), bytes.data(), batch), "Cannot gather DP rows on the draft vocabulary shard.");
    }
    auto &projection = shard.weight;
    auto *owner = shard.owner;
    const auto range = shard.columns;
    auto *destination = owner ? static_cast<uint16_t *>(owner->cudaData) + range.first : nullptr;
    bool projected = false;
#ifndef USE_ROCM
    if (owner && projection.dims[0] == range.second - range.first)
        projected = FastllmCudaTryBf16ColumnProjection(input, projection, destination, owner->dims.back());
#endif
    if (!projected) {
        if (offsets.back() > 1) MatMulTransB(input, projection, shard.logits);
        else Linear(input, projection, Data(), shard.logits);
        if (owner) {
            const size_t width = size_t(range.second - range.first) * sizeof(uint16_t);
            AssertInFastLLM(FastllmCudaMemcpy2DDeviceToDeviceAsyncCurrentThread(destination,
                size_t(owner->dims.back()) * sizeof(uint16_t), shard.logits.cudaData, width, width,
                offsets.back()), "Cannot publish the draft vocabulary shard on its sampling owner.");
        }
    }
}

bool TryCudaDraftHead(const DraftProjectedHead &h, int steps) {
#ifndef USE_ROCM
    if (h.base.dims.size() != 3 || steps < 1 || h.base.dataDeviceIds.size() != 1) return false;
    const int device = h.base.dataDeviceIds.front(), vocab = h.base.dims.back();
    if (!Resident(h.base, device, BFLOAT16, {1, steps, vocab}) ||
        !Resident(h.embedding, device, BFLOAT16, {vocab, 256}, 16) ||
        !Resident(h.projection, device, BFLOAT16, {vocab, 256}, 16) ||
        !Resident(h.ids, device, FLOAT32, {steps + 1, 1}) ||
        !Resident(h.probabilities, device, FLOAT32, {1, steps, vocab}, 8)) return false;
    return FastllmCudaMtpTryOwnerHead(h.sampler, h.base.cudaData, h.embedding.cudaData, h.projection.cudaData,
        static_cast<float *>(h.ids.cudaData), static_cast<float *>(h.probabilities.cudaData), vocab, steps);
#else
    return false;
#endif
}

bool TryCudaDraftHead(const DraftShardedHead &h, int steps) {
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
    return FastllmDraftFusedHeadRun(h.state, h.rank, h.row, h.normalized, h.anchor, h.seed,
        h.temperature, steps, nullptr);
#else
    return false;
#endif
}

bool TryCudaDraftMlp(const DraftLayerParallel &p, int requests, const Data &normalized,
        const DraftMlpWeights &w, Data &residual, const Data &gamma, float eps,
        Data &middle, Data &scratch, Data &output, const DraftCollectiveLayer *collective) {
#ifndef USE_ROCM
    if (collective && collective->enabled && w.gateUp && collective->storage)
        return FastllmCudaDraftMlp7(normalized, residual, output, *w.gateUp, w.down, gamma,
                                    eps, p.devices.at(p.rank), *collective->storage);
#endif
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
    if (p.tpSize != 1 || requests != 1 || p.rank < 0 || size_t(p.rank) >= p.devices.size() ||
        !w.gateUp || normalized.dims.size() != 3 || normalized.dims[1] < 7 || normalized.dims[1] > 8)
        return false;
    const int device = p.devices[p.rank];
    const auto &shape = normalized.dims;
    if (!Resident(normalized, device, BFLOAT16, {1, shape[1], 4096}, 16) ||
        !Resident(*w.gateUp, device, BFLOAT16, {8192, 4096}, 32) ||
        !Resident(w.down, device, BFLOAT16, {4096, 4096}, 32) ||
        !Resident(residual, device, BFLOAT16, shape, 16) || !Resident(scratch, device, BFLOAT16, shape, 16) ||
        !Resident(gamma, device, FLOAT32, {4096}, 16) || !CanWrite(middle, device, shape) ||
        !CanWrite(output, device, shape)) return false;
    if (FastllmCudaRuntimeArch() != 103) return false;
    const int blocks = FastllmDraftFusedLocalMlpBlocks(device);
    if (!blocks) return false;
    AllocateOutput(middle, device, shape);
    AllocateOutput(output, device, shape);
    FastllmDraftFusedLocalMlpRun(blocks, shape[1], normalized.cudaData, w.gateUp->cudaData, w.down.cudaData,
        middle.cudaData, scratch.cudaData, residual.cudaData, static_cast<const float *>(gamma.cudaData),
        eps, output.cudaData, nullptr);
    return true;
#else
    return false;
#endif
}

bool TryCudaDraftAttention(const DraftLayerParallel &p, const DraftAttentionGeometry &g,
        const std::vector<FastllmNaiveDraftRow> &requests, const Data *bindings, Data &normalized,
        const DraftAttentionWeights &w,
        Data &hidden, Data &raw, Data &attention, Data &branch, Data &scratch, const DraftCollectiveLayer *collective) {
#ifndef USE_ROCM
    if (collective && collective->enabled && w.qkv && collective->storage && collective->live && collective->bindings)
        return FastllmCudaDraftAttention7(normalized, hidden, *w.qkv, w.o, w.qNorm, w.kNorm, w.postNorm,
            *collective->live, *collective->bindings, collective->layer, g.eps, g.theta, p.devices.at(p.rank),
            *collective->storage);
#endif
#ifdef FASTLLM_ENABLE_DRAFT_FUSED
    if (p.tpSize != 1 || requests.size() != 1 || p.rank < 0 || size_t(p.rank) >= p.devices.size() ||
        !w.qkv || normalized.dims.size() != 3 || !std::isfinite(g.eps) || g.eps <= 0 ||
        !std::isfinite(g.theta) || g.theta <= 0 || FastllmCudaRuntimeArch() != 103) return false;
    const int device = p.devices[p.rank], rows = normalized.dims[1];
    int floats = 0;
    const int blocks = FastllmDraftLocalAttentionPlan(device, rows, g.heads, g.kvHeads, g.headDim, g.window, &floats);
    if (!blocks) return false;
    const int width = g.heads * g.headDim, kw = g.kvHeads * g.headDim;
    const std::vector<int> shape{1, rows, width}, packed{1, rows, width + 2*kw};
    const auto &r = requests.front();
    const auto *table = bindings ? static_cast<const void *const *>(bindings->cudaData) : r.cacheTable;
    if (!table || (bindings && !Resident(*bindings, device, INT32, {1,4}, 8)) ||
        !r.positions || !r.liveKeys || !Resident(*r.positions, device, FLOAT32, {1,rows}) ||
        !Resident(*r.liveKeys, device, INT32, {1}) || !Resident(normalized, device, BFLOAT16, shape, 16) ||
        !Resident(hidden, device, BFLOAT16, shape, 16) || !Resident(*w.qkv, device, BFLOAT16, {packed[2],width}, 32) ||
        !Resident(w.o, device, BFLOAT16, {width,width}, 32) || !Resident(w.qNorm, device, FLOAT32, {g.headDim}) ||
        !Resident(w.kNorm, device, FLOAT32, {g.headDim}) || !Resident(w.postNorm, device, FLOAT32, {width}) ||
        !CanWrite(raw, device, packed) || !CanWrite(attention, device, shape) ||
        !CanWrite(branch, device, shape) || !CanWrite(scratch, device, {floats}, FLOAT32)) return false;
    if (!r.key || !r.value || r.key->dims != r.value->dims) return false;
    for (const Data *cache : {r.key, r.value}) {
        if (cache->isPagedKVCache || cache->multiDeviceData || cache->dataType != BFLOAT16 ||
            cache->dataDevice != CUDA || cache->dataDeviceIds != normalized.dataDeviceIds ||
            !cache->cudaData || (reinterpret_cast<uintptr_t>(cache->cudaData) & 15) ||
            cache->dims.size() != 3 || cache->dims[0] != 1 || cache->dims[1] < 0 || cache->dims[2] != kw ||
            cache->expansionDims.size() != 3 || cache->expansionDims[0] != 1 ||
            cache->expansionDims[1] < g.window-1+rows || cache->expansionDims[2] != kw ||
            cache->strides.size() != 3 || cache->strides[1] != kw || cache->strides[2] != 1) return false;
    }
    AllocateOutput(raw, device, packed);
    AllocateOutput(attention, device, shape);
    AllocateOutput(branch, device, shape);
    AllocateOutput(scratch, device, {floats}, FLOAT32);
    FastllmDraftLocalAttentionRun(blocks, {normalized.cudaData, w.qkv->cudaData, w.o.cudaData,
        static_cast<const float *>(w.qNorm.cudaData), static_cast<const float *>(w.kNorm.cudaData),
        static_cast<const float *>(w.postNorm.cudaData), static_cast<const float *>(r.positions->cudaData),
        static_cast<const int *>(r.liveKeys->cudaData), table, raw.cudaData, attention.cudaData, hidden.cudaData,
        normalized.cudaData, static_cast<float *>(scratch.cudaData), static_cast<float *>(branch.cudaData),
        rows, g.window, g.eps, g.theta}, nullptr);
    return true;
#else
    return false;
#endif
}

#ifdef FASTLLM_ENABLE_DRAFT_FUSED
bool CudaDraftHeadSupportsShard(const Data &normalized, int device, int ranks, int requests,
        int steps, int hidden, int vocab) {
    return ranks == 8 && requests > 0 && requests <= MaxBodyRequests && steps == 7 && hidden == 4096 &&
        vocab == 152576 && Resident(normalized, device, BFLOAT16, {1, requests * steps, hidden});
}

bool PrepareCudaDraftShardedHead(void *state, const std::vector<int> &devices,
        const std::vector<std::pair<int,int>> &ranges, const Data &head, Data &w1, Data &w2,
        Data &confidence, Data &bias, double budget) {
    constexpr int Vocab = 152576, Shard = Vocab / 8, Hidden = 4096, Latent = 256;
    if (devices.size() != 8 || ranges.size() != devices.size() ||
        !head.multiDeviceData || head.dataType != BFLOAT16 || head.dims != std::vector<int>{Vocab, Hidden} ||
        w1.dataType != BFLOAT16 || w2.dataType != BFLOAT16 || w1.dims != std::vector<int>{Vocab, Latent} ||
        w2.dims != w1.dims || w1.strides != std::vector<uint64_t>{Latent, 1} || w2.strides != w1.strides ||
        w1.multiDeviceData || w2.multiDeviceData || confidence.multiDeviceData || bias.multiDeviceData ||
        confidence.dataType != FLOAT32 || confidence.Count(0) != Hidden + Latent ||
        bias.dataType != FLOAT32 || bias.Count(0) != 1) return false;
    const void *shards[8]{};
    for (int rank = 0; rank < 8; ++rank) {
        auto it = head.multiDeviceDatas.find(devices[rank]);
        if (it == head.multiDeviceDatas.end() || !it->second ||
            !Resident(*it->second, devices[rank], BFLOAT16, {Shard, Hidden}) ||
            ranges[rank] != std::make_pair(rank * Shard, (rank + 1) * Shard)) return false;
        shards[rank] = it->second->cudaData;
    }
    for (Data *w : {&w1, &w2, &confidence, &bias}) w->ToDevice(CUDA, std::vector<int>{devices.front()});
    return FastllmDraftFusedHeadPrepareWithBudget(state, shards, w1.cudaData, w2.cudaData,
        static_cast<const float *>(confidence.cudaData), static_cast<const float *>(bias.cudaData),
        devices.front(), budget);
}

bool ExportCudaDraftShardedHead(void *state, const Data &hidden, int steps, int vocab,
        const std::vector<float> &temperatures, const std::vector<int> &limits, Data &q,
        int64_t *ids, float *confidence) {
    const int count = limits.size();
    auto *probabilities = static_cast<float *>(q.cudaData);
    if (count >= 8)
        return FastllmDraftFusedHeadSubmitBatch(state, count, hidden.cudaData,
            temperatures.data(), limits.data(), probabilities, ids, confidence, nullptr);
    for (int row = 0; row < count; ++row) {
        const size_t offset = size_t(row) * steps;
        if (!FastllmDraftFusedHeadSubmit(state, row,
                static_cast<const uint16_t *>(hidden.cudaData) + offset * hidden.dims.back(), temperatures[row],
                probabilities + offset * vocab, ids + offset, confidence ? confidence + offset : nullptr,
                limits[row], nullptr)) return false;
    }
    return true;
}

bool CudaDraftBodySupportsShape(int ranks, const DraftBodyShape &s, int requests) {
    return ranks == 8 && s.attentionTP == ranks && s.headsPerRank == 4 && s.kvHeadsPerRank == 1 &&
        s.headDim == 128 && s.rows >= 2 && s.rows <= 8 && s.hidden == 4096 && s.layers >= 1 &&
        s.layers <= 5 && s.window >= 2 && s.window <= 16384 && requests >= 1 && requests <= MaxBodyRequests;
}

bool CudaDraftUseBody(int requests) {
    // Wider batches share projections and collectives across requests.
    return requests > 0 && requests < 8;
}

int CudaDraftBodyCapacity(int requests, int maxBatch) {
    return std::max(requests, std::min(MaxBodyRequests, std::max(1, maxBatch)));
}

void *CreateCudaDraftBody(const std::vector<int> &devices, const DraftBodyShape &s,
        const std::vector<DraftBodyWeights> &weights, float eps, float theta) {
    if (!CudaDraftBodySupportsShape(devices.size(), s, 1) || weights.size() != devices.size() * s.layers)
        return nullptr;
    std::vector<FastllmDraftFusedWeight> packed;
    packed.reserve(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        const auto &w = weights[i];
        const int device = devices[i / s.layers];
        auto ptr = [&](const Data *tensor, DataType type, const std::vector<int> &dims) -> const void * {
            return tensor && Resident(*tensor, device, type, dims) ? tensor->cudaData : nullptr;
        };
        FastllmDraftFusedWeight raw {
            ptr(w.qkv, BFLOAT16, {768,4096}), ptr(w.o, BFLOAT16, {4096,512}), ptr(w.inputNorm, FLOAT32, {4096}),
            ptr(w.qNorm, FLOAT32, {128}), ptr(w.kNorm, FLOAT32, {128}), ptr(w.gateUp, BFLOAT16, {1024,4096}),
            ptr(w.down, BFLOAT16, {4096,512}), ptr(w.postNorm, FLOAT32, {4096})};
        if (!raw.qkv || !raw.o || !raw.inputNorm || !raw.qNorm || !raw.kNorm ||
            !raw.gateUp || !raw.down || !raw.postNorm) return nullptr;
        packed.push_back(raw);
    }
    return FastllmDraftFusedCreate(devices.data(), devices.size(), s.layers, packed.data(),
        s.rows, s.window, eps, theta);
}

void RunPreparedCudaDraftBody(void *state, int rank, int device, int count, int rows, const Data &ids,
        const Data &embedding, const Data &mask, const Data &live, const Data &cacheTable, Data &gamma,
        float eps, Data &hidden, Data &positions, Data &output) {
    gamma.ToDevice(DataDevice::CUDA,std::vector<int>{device});
    AssertInFastLLM(ids.Count(0) == count && live.Count(0) == count, "Draft input length count mismatch.");
    const bool normalizedGamma = gamma.dataType == FLOAT32 && gamma.Count(0) == 4096 &&
        gamma.strides.back() == 1 && !(reinterpret_cast<uintptr_t>(gamma.cudaData) & 15);
    const bool embedded = normalizedGamma && embedding.dataType == BFLOAT16 &&
        embedding.dims.back() == 4096 && mask.dataType == BFLOAT16 && mask.Count(0) == 4096 &&
        embedding.cudaData && mask.cudaData &&
        !((reinterpret_cast<uintptr_t>(embedding.cudaData) | reinterpret_cast<uintptr_t>(mask.cudaData)) & 15);
    if (embedded) {
        hidden.dataType = BFLOAT16; hidden.UpdateUnitSize(); hidden.Resize({1, rows * count, 4096});
        hidden.ToDevice(DataDevice::CUDA,{device},false); hidden.Allocate(false);
    }
    const bool buildInput = embedded && !(reinterpret_cast<uintptr_t>(hidden.cudaData) & 15);
    if (!buildInput) FastllmCudaNaiveDraftInput(ids, embedding, mask, live, rows, hidden, positions);
    auto *lengths = static_cast<const int *>(live.cudaData);
    auto *table = static_cast<const void *const *>(cacheTable.cudaData);
    if (normalizedGamma && !(reinterpret_cast<uintptr_t>(hidden.cudaData) & 15)) {
        output.dataType = BFLOAT16; output.UpdateUnitSize(); output.Resize(hidden.dims);
        output.ToDevice(DataDevice::CUDA,{device},false); output.Allocate(false);
        auto *norm = static_cast<const float *>(gamma.cudaData);
        bool ok = buildInput ? FastllmDraftFusedRunEmbedded(state,rank,count,hidden.cudaData,
            static_cast<const float *>(ids.cudaData),embedding.cudaData,mask.cudaData,lengths,table,norm,output.cudaData,nullptr)
            : FastllmDraftFusedRunNormalized(state,rank,count,hidden.cudaData,
                lengths,table,norm,output.cudaData,nullptr);
        AssertInFastLLM(ok,"Normalized fused draft backbone launch failed.");
    } else {
        AssertInFastLLM(FastllmDraftFusedRun(state,rank,count,hidden.cudaData,
            lengths,table,nullptr),"Fused draft backbone launch failed.");
        KimiK3RMSNorm(hidden,gamma,eps,output);
    }
}

bool CudaDraftCanShardContext(const std::vector<int> &devices, int targetLayers, int layers,
        int hidden, int headDim, int window, const Data &fc, const Data &gamma) {
    // The prepared context/head graph has eight peer slots independently of the MLP partition.
    return devices.size() == 8 && targetLayers == 8 && layers == 5 && hidden == 4096 && headDim == 128 && window == 1024 &&
        gamma.dataType == FLOAT32 && !fc.multiDeviceData && !gamma.multiDeviceData &&
        FastllmDraftFusedSupported(devices.data(), devices.size());
}

bool PrepareCudaDraftContext(void *state, int device, const Data &fc, const Data &gamma) {
    return Resident(fc, device, BFLOAT16, {4096,32768}) && Resident(gamma, device, FLOAT32, {4096}) &&
        FastllmDraftFusedContextPrepare(state, fc.cudaData, gamma.cudaData, device);
}
#endif
} // namespace fastllm
