#include "fastllm-draft-sm120.h"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-mtp.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "utils/utils.h"
#include <algorithm>
#include <cmath>
#include <climits>

using namespace fastllm;

FastllmDraftLayerSupport FastllmCudaDraftLayerSupport(int ranks, int requests, int steps, int heads,
                                                     int kvHeads, int headDim, int window) {
    if (ranks != 8 || requests != 1 || FastllmCudaRuntimeArch() != 120) return {};
    return {steps == 7 && heads == 4 && kvHeads == 1 && headDim == 128 && window == 1024, steps == 7, true};
}

bool FastllmCudaTryDraftReduceAddNorm(int device, float eps, Data &hidden, Data &branch,
                           Data &gamma, Data &output) {
    if (FastllmCudaRuntimeArch() != 120 || branch.dataType != BFLOAT16 || hidden.dataType != BFLOAT16 ||
        branch.dims != hidden.dims || branch.dims.empty() || branch.dims.back() != 4096 ||
        branch.Count(0) != 7 * 4096 || gamma.dataType != FLOAT32 || gamma.Count(0) != 4096) return false;
    gamma.ToDevice(CUDA, std::vector<int>{device});
    output.dataType = BFLOAT16;
    output.UpdateUnitSize();
    output.Resize(hidden.dims);
    output.ToDevice(CUDA, {device}, false);
    output.Allocate(false);
    return FastllmCudaDraftReduceAddNorm7(branch.cudaData, hidden.cudaData,
        static_cast<const float *>(gamma.cudaData), output.cudaData,
        eps, device);
}

bool FastllmCudaProjectDraftFeatures(const std::vector<int> &devices, const std::vector<FastllmDraftProjectionRank> &ranks,
        Data &combined, Data &fc, Data &norm, Data &projected, Data &hidden, float eps,
        const FastllmDraftRankRunner &run, const std::function<void *()> &publish) {
    const int rows = combined.dims[1], width = combined.dims[2], embed_dim = 4096;
    if (devices.size() != 8 || ranks.size() != devices.size() || rows < 2 || rows > 8 || width != 32768 ||
        !fc.IsTensorParallelReplicated() || fc.dataType != BFLOAT16 || fc.dims != std::vector<int>{embed_dim,width} ||
        fc.strides != std::vector<uint64_t>{uint64_t(width),1} || !norm.IsTensorParallelReplicated() ||
        combined.dataType != BFLOAT16 || combined.dataDevice != CUDA || combined.multiDeviceData ||
        FastllmCudaRuntimeArch() != 120) return false;
    constexpr int workers = 4;
    const int root = devices.front(), shard = embed_dim / workers;
    projected.dataType = BFLOAT16;
    projected.UpdateUnitSize();
    projected.Resize({1, rows, embed_dim});
    projected.ToDevice(CUDA, {root}, false);
    projected.Allocate(false);
    void *ready = publish();
    run([&](int rank) {
        if (rank >= workers) return;
        const auto &r = ranks[rank];
        const int device = devices[rank];
        Data *input = &combined;
        if (rank) {
            FastllmCudaCurrentThreadStreamWaitEvent(ready);
            r.input->dataType = BFLOAT16;
            r.input->UpdateUnitSize();
            r.input->Resize(combined.dims);
            r.input->ToDevice(CUDA, {device}, false);
            r.input->Allocate(false);
            AssertInFastLLM(FastllmCudaMemcpyPeerAsyncCurrentThread(device, r.input->cudaData,
                root, combined.cudaData, combined.GetBytes()), "Cannot publish draft context features.");
            input = r.input;
        }
        Data weightSlice;
        weightSlice.FakeFrom(*fc.multiDeviceDatas.at(device), size_t(rank) * shard * width * sizeof(uint16_t));
        weightSlice.Resize({shard, width});
        MatMulTransB(*input, weightSlice, *r.output);
        auto *dst = static_cast<uint16_t *>(projected.cudaData) + rank * shard;
        AssertInFastLLM(FastllmCudaMemcpy2DDeviceToDeviceAsyncCurrentThread(dst, embed_dim * sizeof(uint16_t),
            r.output->cudaData, shard * sizeof(uint16_t), shard * sizeof(uint16_t), rows),
            "Cannot gather draft context projection.");
        FastllmCudaEventRecordCurrentThread(r.done);
    });
    for (int rank = 1; rank < workers; ++rank)
        FastllmCudaCurrentThreadStreamWaitEvent(ranks[rank].done);
    KimiK3RMSNorm(projected, *norm.multiDeviceDatas.at(root), eps, hidden);
    return true;
}

bool FastllmCudaCommitDraftFeatures(const std::vector<int> &devices,
        const std::vector<FastllmDraftContextRank> &ranks, WeightMap &modelWeights, FastllmDraftKernelPlan &plan,
        FastllmDraftContextShape shape, bool preparing, int count, int position, float eps, float theta,
        int anchor, FastllmDraftContinuation continuation, const FastllmDraftRankRunner &run,
        const std::function<void *()> &publish) {
    auto &[state, unavailable] = plan;
    if (shape.layers != 5 || shape.features != 8 || shape.hidden != 4096 || shape.headDim != 128 ||
        shape.steps != 7 || shape.window != 1024 || shape.attentionRanks != 8 || ranks.empty() ||
        !ranks.front().kv || ranks.front().kv->empty() || ranks.front().kv->front().first.dims.size() != 3)
        return false;
    const int old = ranks.front().kv->front().first.dims[1];
    if (devices.size() != 8 || ranks.size() != 8 || count < 1 || count > 8 || position < 0 || position > INT_MAX - count || old < 0 || old > 1023 ||
        !std::isfinite(eps) || eps <= 0 || !std::isfinite(theta) || theta <= 0 ||
        (!state && (!preparing || unavailable))) return false;
    const int root = devices.front();
    FastllmDraftContextInput120 input[8]{};
    for (int r = 0; r < 8; ++r) {
        if (!ranks[r].feature || !ranks[r].kv) return false;
        const auto &x = *ranks[r].feature;
        if (x.dataType != BFLOAT16 || x.dataDevice != CUDA || x.dataDeviceIds != std::vector<int>{root} ||
            !x.cudaData || (reinterpret_cast<uintptr_t>(x.cudaData) & 15) || x.dims.size() != 3 ||
            x.dims[0] != 1 || x.dims[1] < count || x.dims[2] != 4096 || x.strides.size() != 3 ||
            x.strides[1] != 4096 || x.strides[2] != 1 || ranks[r].kv->size() != 5)
            return false;
        input[r].feature = x.cudaData;
        int i = 0;
        for (auto &pair : *ranks[r].kv)
            for (Data *c : {&pair.first, &pair.second}) {
                if (!c->cudaData || (reinterpret_cast<uintptr_t>(c->cudaData) & 15) ||
                    c->dataType != BFLOAT16 || c->dataDevice != CUDA ||
                    c->dataDeviceIds != std::vector<int>{devices[r]} || c->multiDeviceData ||
                    c->isPagedKVCache || !FastllmNaiveCanResizeKVLength(*c, 1031, 128) || c->dims[1] != old ||
                    c->strides.size() != 3 || c->strides[1] != 128 || c->strides[2] != 1)
                    return false;
                input[r].cache[i++] = c->cudaData;
            }
    }
    if (!state) {
        if (FastllmCudaRuntimeArch() != 120) {
            unavailable = true;
            return false;
        }
        FastllmDraftContextWeights120 weights[8]{};
        auto shard = [&](const std::string &name, int gpu) -> const Data * {
            auto it = modelWeights.weight.find(name);
            if (it == modelWeights.weight.end())
                return nullptr;
            auto local = it->second.multiDeviceDatas.find(gpu);
            return local == it->second.multiDeviceDatas.end() ? nullptr : local->second;
        };
        for (int r = 0; r < 8; ++r) {
            int gpu = devices[r];
            auto &w = weights[r];
            w.fc = shard("dspark.fc.weight", gpu);
            w.gamma = shard("dspark.hidden_norm.weight", gpu);
            for (int l = 0; l < 5; ++l) {
                auto p = "dspark.layers." + std::to_string(l) + ".self_attn.";
                w.qkv[l] = shard(p + "mergeqkv.weight", gpu);
                w.kNorm[l] = shard(p + "k_norm.weight", gpu);
            }
        }
        state = FastllmCudaCreateDraftContext120(devices, weights);
        unavailable = !state;
        if (!state)
            return false;
    }
    auto *replay = continuation.replay;
    auto owner = replay ? replay->ForCommit(continuation.issued, anchor, continuation.head,
                                             continuation.probabilities) : nullptr;
    FastllmDraftGraphRank graphRanks[8]{};
    bool continued = bool(owner);
    for (int r = 0; continued && r < 8; ++r) {
        graphRanks[r] = replay->rank(r);
        continued = graphRanks[r].exec && graphRanks[r].bindings;
    }
    if (continued) {
        replay->request.anchor = anchor;
        replay->request.seed = FastllmCudaPeekDraftSamplingSeed();
        input[0].request = replay->request;
        for (int r = 0; r < 8; ++r) {
            input[r].nextBindings = graphRanks[r].bindings;
            input[r].anchor = anchor;
        }
    }
    // Capture owns the feature rows until this commit joins all readers.
    const bool waitForFeatures = !FastllmCudaDraftFeaturesReady120();
    void *ready = waitForFeatures ? publish() : nullptr;
    run([&](int r) {
        if (r && waitForFeatures)
            FastllmCudaCurrentThreadStreamWaitEvent(ready);
        FastllmCudaCommitDraftContext120(state.get(), r, input[r], count, old, position,
                                         eps, theta);
        if (continued)
            AssertInFastLLM(FastllmCudaGraphLaunch(graphRanks[r].exec),
                            "SM120 draft continuation failed.");
        FastllmCudaEventRecordCurrentThread(ranks[r].done);
    });
    for (int r = 1; r < 8; ++r)
        FastllmCudaCurrentThreadStreamWaitEvent(ranks[r].done);
    FastllmCudaSyncCurrentThreadStream();
    for (int r = 0; r < 8; ++r)
        for (auto &pair : *ranks[r].kv)
            for (Data *c : {&pair.first, &pair.second})
                FastllmNaiveResizeKVLength(*c, std::min(old + count, 1023), 128);
    if (continued) continuation.nextTicket = replay->Complete(position + count);
    return true;
}

static bool PrepareDraftProposalHead(const std::vector<int> &devices, const std::vector<std::pair<int, int>> &ranges,
        WeightMap &weights, bool preparing, bool &unavailable, std::shared_ptr<void> &state) {
    if (state) return true;
    if (!preparing || unavailable || FastllmCudaRuntimeArch() != 120) return false;
    constexpr int vocab = 152576;
    auto h = weights.weight.find("lm_head.weight");
    auto a = weights.weight.find("dspark.markov_head.markov_w1.weight");
    auto b = weights.weight.find("dspark.markov_head.markov_w2.weight");
    if (h == weights.weight.end() || a == weights.weight.end() || b == weights.weight.end()) return false;
    const Data &head = h->second;
    Data &w1 = a->second, &w2 = b->second;
    if (devices.size() != 8 || ranges.size() != 8 || head.dims != std::vector<int>{vocab,4096} ||
        head.dataType != BFLOAT16 || w1.dataType != BFLOAT16 || w2.dataType != BFLOAT16 || w1.multiDeviceData ||
        w2.multiDeviceData || w1.dims != std::vector<int>{vocab,256} || w2.dims != w1.dims) return false;
    std::vector<const Data *> heads;
    for (int i = 0; i < 8; ++i) {
        auto found = head.multiDeviceDatas.find(devices[i]);
        if (found == head.multiDeviceDatas.end() || !found->second ||
            ranges[i] != std::make_pair(i * (vocab / 8), (i + 1) * (vocab / 8)))
            return false;
        heads.push_back(found->second);
    }
    w1.ToDevice(CUDA, std::vector<int>{devices.front()});
    w2.ToDevice(CUDA, std::vector<int>{devices.front()});
    state = FastllmCudaCreateDraftHead120(devices, heads, w1, w2);
    unavailable = !state;
    return bool(state);
}

void FastllmDraftReplay::Bind(const std::shared_ptr<void> &graph, void *head,
                              std::function<FastllmDraftGraphRank(int)> lookup) {
    owner = graph;
    graphHead = head;
    rank = std::move(lookup);
}

bool FastllmDraftReplay::Consume(uint64_t ticket, int length, const FastllmDraftHeadCall &call) {
    if (!ticket || ticket != generation || ticket == issued) return false;
    ++generation;
    return !owner.expired() && graphHead == call.state && committed == length &&
           request.anchor == call.request.anchor && request.seed == call.request.seed &&
           request.temperature == call.request.temperature && request.probabilities == call.request.probabilities;
}

uint64_t FastllmDraftReplay::Issue(const FastllmDraftSamplingRequest &used) {
    request = used;
    return issued = ++generation;
}

std::shared_ptr<void> FastllmDraftReplay::ForCommit(uint64_t ticket, int anchor, void *head, const void *probs) const {
    if (!ticket || ticket != generation || ticket != issued || anchor < 0 || anchor >= 152576 ||
        !probs || probs != request.probabilities || head != graphHead) return {};
    return owner.lock();
}

uint64_t FastllmDraftReplay::Complete(int length) {
    committed = length;
    return ++generation;
}

bool FastllmCudaProposeDraft(const std::vector<int> &devices, const std::vector<std::pair<int,int>> &ranges,
        WeightMap &weights, FastllmDraftKernelPlan &plan, int hidden, int steps, int limit, int anchor,
        float temperature, bool preparing, uint64_t &ticket, int length, int &vocab, std::vector<int> &tokens,
        const std::function<float *(int, int, int)> &samplingMemory,
        const std::function<FastllmDraftReplay *()> &replay,
        const std::function<bool(const FastllmDraftHeadCall &)> &run) {
    constexpr int vocabulary = 152576, proposals = 7, latent = 256;
    if (steps != proposals || limit != steps || hidden != 4096 || anchor < 0 || anchor >= vocabulary ||
        !std::isfinite(temperature) || temperature <= 0 || !std::isfinite(1.f / temperature) ||
        !PrepareDraftProposalHead(devices, ranges, weights, preparing, plan.unavailable, plan.handle)) return false;
    auto *probabilities = samplingMemory(proposals, vocabulary, latent);
    FastllmDraftHeadCall call{plan.handle.get(), {preparing ? 0 : FastllmCudaNextDraftSamplingSeed(),
                                                  probabilities, temperature, anchor}};
    auto *prior = preparing ? nullptr : replay();
    if ((!prior || !prior->Consume(ticket, length, call)) && !run(call)) return false;
    int ids[proposals];
    FastllmCudaReadDraftProposal(call.state, ids);
    for (int id : ids) {
        AssertInFastLLM(id >= 0 && id < vocabulary, "Invalid graph draft token.");
        tokens.push_back(id);
    }
    vocab = vocabulary;
    // Running the backbone may have created or replaced its workspace.
    if (!preparing) {
        if (auto *current = replay()) ticket = current->Issue(call.request);
    }
    return true;
}
