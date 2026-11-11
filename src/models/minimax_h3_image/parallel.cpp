#include "models/minimax_h3_image/parallel.h"
#include <algorithm>
#include <stdexcept>

namespace fastllm::minimax_h3_image {
struct ParallelModel::Rank {
    std::string directory;
    Runtime rt, full;
    MemoryPolicy memory;
    minimax_h3::TextEncoder text;
    Transformer transformer;
    std::unique_ptr<VAE> decoder;

    Rank(const std::string &dir, int device, nomas::ParallelContext context, const MemoryPolicy &root,
         double gpuLimit, bool sequenceParallel)
        : directory(dir), rt(device), full(device, true),
          memory(context.rank ? MemoryPolicy(rt, gpuLimit, 2) : root),
          text(rt, dir + "/text_encoder", {context.group, context.rank}),
          transformer(rt, full, dir, context, sequenceParallel) {
        // One host-cache budget shared by all ranks; GPU limits apply per card.
        memory.host = root.host;
        rt.setLinearAutotuning(sequenceParallel);
        memory.layerStreaming = root.layerStreaming;
        text.setLowMemory(memory.layerStreaming);
        text.setMemoryPool(memory.pool);
        text.setHostMemoryBudget(memory.host);
        transformer.setMemoryPolicy(memory);
    }
    std::unique_ptr<nomas::TensorPool::Scope> scope() {
        return memory.pool ? std::make_unique<nomas::TensorPool::Scope>(*memory.pool) : nullptr;
    }
};

ParallelModel::ParallelModel(const std::string &dir, const std::vector<int> &devices,
                             const MemoryPolicy &root, double gpuLimit, nomas::RowReduction reduction,
                             bool sequenceParallel)
    : group_(devices), ranks_(devices.size()) {
    group_.run([&](int rank) {
        ranks_[rank] =
            std::make_unique<Rank>(dir, devices[rank], nomas::ParallelContext{&group_, rank, reduction}, root,
                                   gpuLimit, sequenceParallel);
    });
}
ParallelModel::~ParallelModel() {
    // Dispose caches on the same threads/streams that own their CUDA runtimes.
    try {
        group_.run([&](int rank) { ranks_[rank].reset(); });
    } catch (...) {
    }
}
Tensor ParallelModel::encode(const std::vector<int> &tokens,
                             const std::vector<minimax_h3::VisionInput> &visual, Trace trace) {
    Tensor result;
    group_.run([&](int rank) {
        auto &r = *ranks_[rank];
        auto scope = r.scope();
        auto x = r.text.encodeVisual(tokens, visual, rank == 0 ? trace : Trace{});
        r.text.unload(!r.memory.layerStreaming);
        if (rank == 0)
            result = std::move(x);
    });
    return result;
}
void ParallelModel::prepare(Progress progress) {
    group_.run([&](int rank) {
        auto scope = ranks_[rank]->scope();
        ranks_[rank]->transformer.prepare(rank == 0 ? progress : Progress{});
    });
}
void ParallelModel::prepareText(const Tensor &text) {
    group_.run([&](int rank) {
        auto scope = ranks_[rank]->scope();
        ranks_[rank]->transformer.prepareText(group_.broadcast(text, rank));
    });
}
Tensor ParallelModel::forward(const Tensor &latent, const Tensor &text, int height, int width, int step,
                              Trace trace, const Conditioning *condition) {
    Tensor result;
    group_.run([&](int rank) {
        auto scope = ranks_[rank]->scope();
        auto z = group_.broadcast(latent, rank), t = group_.broadcast(text, rank);
        // A sequence-parallel trace gathers on every rank, then exposes rank 0.
        Trace rankTrace = trace ? Trace([&](const std::string &name, const Tensor &x) {
            if (rank == 0)
                trace(name, x);
        })
                                : Trace{};
        auto x = ranks_[rank]->transformer.forward(z, t, height, width, step, rankTrace, condition);
        if (rank == 0)
            result = std::move(x);
    });
    return result;
}
const std::vector<float> &ParallelModel::sigmas() const { return ranks_[0]->transformer.sigmas(); }
void ParallelModel::releaseStreamedWeights() {
    group_.run([&](int rank) {
        ranks_[rank]->transformer.clearText();
        if (ranks_[rank]->memory.layerStreaming)
            ranks_[rank]->transformer.unload(true);
    });
}
Tensor ParallelModel::decodeTiles(const Tensor &latent, int height, int width, int batch) {
    if (batch < 1 || height < 1 || width < 1 || int64_t(height) * width * batch != latent.rows ||
        latent.cols != 24 || !latent.isFloat32())
        throw std::invalid_argument("invalid parallel image decoder batch");
    std::vector<Tensor> inputs(group_.size());
    group_.run([&](int rank) {
        auto scope = ranks_[rank]->scope();
        inputs[rank] = group_.broadcast(latent, rank);
    });
    const int partitions = std::min(batch, group_.size()), perRank = (batch + partitions - 1) / partitions;
    auto outputs = group_.mapToRoot((batch + perRank - 1) / perRank, [&](int rank, size_t job) {
        auto &r = *ranks_[rank];
        auto scope = r.scope();
        if (!r.decoder) {
            r.decoder = std::make_unique<VAE>(r.rt, r.full, r.directory);
            r.decoder->setMemoryPolicy(r.memory);
        }
        const int first = int(job) * perRank, count = std::min(perRank, batch - first);
        return r.decoder->decodeTiles(inputs[rank].rowSlice(first * height * width, count * height * width),
                                      height, width, count);
    });
    return ranks_[0]->rt.concat(outputs);
}
} // namespace fastllm::minimax_h3_image
