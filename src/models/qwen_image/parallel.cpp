#include "parallel.h"
#include <algorithm>

namespace fastllm::qwen_image {
struct ParallelDiT::Rank {
    Runtime rt;
    nomas::TensorPool pool;
    DiT dit;
    Condition condition;
    KVCache cache;
    Rank(const std::string &directory, int device, bool fp32, nomas::ParallelContext context)
        : rt(device, fp32), pool(device, size_t(1) << 30), dit(rt, directory, context) {}
};
ParallelDiT::ParallelDiT(const std::string &directory, const std::vector<int> &devices, bool fp32)
    : group_(devices), ranks_(devices.size()) {
    group_.run([&](int rank) {
        ranks_[rank] = std::make_unique<Rank>(directory, devices[rank], fp32,
                                             nomas::ParallelContext{&group_, rank});
    });
}
ParallelDiT::~ParallelDiT() {
    try { group_.run([&](int rank) { ranks_[rank].reset(); }); } catch (...) {}
}
bool ParallelDiT::canKeep(size_t weights, size_t workspace) {
    std::vector<int> fits(ranks_.size(), 1);
    group_.run([&](int rank) {
        auto &r = *ranks_[rank];
        // Rank 0 additionally checks text/VAE residency in Pipeline.
        fits[rank] = r.rt.availableMemory() + r.dit.loadedBytes() >= weights + workspace;
    });
    return std::all_of(fits.begin(), fits.end(), [](int value) { return value; });
}
size_t ParallelDiT::rootLoadedBytes() const { return ranks_.front()->dit.loadedBytes(); }
void ParallelDiT::prepare(const Condition &condition, bool fullCache) {
    group_.run([&](int rank) {
        auto &r = *ranks_[rank];
        nomas::TensorPool::Scope scope(r.pool);
        r.cache = {};
        r.cache.full = fullCache;
        r.condition.features = group_.broadcast(condition.features, rank);
        r.condition.latents = condition.latents.rows ? group_.broadcast(condition.latents, rank) : Tensor{};
        r.condition.imageSlots = condition.imageSlots;
        r.condition.shapes = condition.shapes;
    });
}
Tensor ParallelDiT::forward(const Tensor &latent, float timestep, int height, int width) {
    Tensor output;
    group_.run([&](int rank) {
        auto &r = *ranks_[rank];
        nomas::TensorPool::Scope scope(r.pool);
        auto input = group_.broadcast(latent, rank);
        auto result = r.dit.forward(input, r.condition, timestep, height, width, &r.cache);
        if (rank == 0) output = std::move(result);
    });
    return output;
}
void ParallelDiT::release(bool unload) {
    group_.run([&](int rank) {
        auto &r = *ranks_[rank];
        r.cache = {};
        r.condition = {};
        if (unload) r.dit.unload();
        r.rt.synchronize();
        r.pool.trim();
    });
}
} // namespace fastllm::qwen_image
