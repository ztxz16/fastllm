#include "ffn_peer.h"
#include <cuda_runtime.h>
#include <set>
#include <stdexcept>

namespace fastllm::duo {
using minimax_h3_image::Fp8Weights;
namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
struct RestoreDevice {
    int original;
    RestoreDevice() { check(cudaGetDevice(&original)); }
    ~RestoreDevice() { cudaSetDevice(original); }
};
using Event = std::unique_ptr<CUevent_st, decltype(&cudaEventDestroy)>;
Event eventOn(int device) {
    check(cudaSetDevice(device));
    cudaEvent_t event;
    check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return Event(event, cudaEventDestroy);
}
} // namespace

struct FfnPeer::Impl {
    int device;
    Runtime runtime;
    std::string directory;
    std::map<std::string, std::string> shards;
    std::map<std::string, std::unique_ptr<Fp8Weights>> weights;
    Event inputReady, tailDone;
    Impl(int primary, int peer, const ModelSource &source, const std::map<std::string, std::string> &files)
        : device(peer), runtime(peer, DType::BF16), directory(source.weights), shards(files),
          inputReady(eventOn(primary)), tailDone(eventOn(peer)) {}
    Fp8Weights &weight(const std::string &key) {
        const auto &file = shards.at(key);
        auto &entry = weights[file];
        if (!entry) {
            minimax_h3_image::ProjectionOptions options;
            options.modelopt = true;
            entry = std::make_unique<Fp8Weights>(directory + "/" + file, options);
        }
        return *entry;
    }
};
FfnPeer::FfnPeer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
FfnPeer::~FfnPeer() {
    int previous;
    if (cudaGetDevice(&previous) == cudaSuccess) {
        cudaSetDevice(impl_->device);
        impl_.reset();
        cudaSetDevice(previous);
    }
}
std::unique_ptr<FfnPeer> FfnPeer::create(int primary, int peer, const ModelSource &source, const Config &cfg,
                                         const std::map<std::string, std::string> &shards) {
    if (primary == peer || source.format != "duo-release-modelopt-v1")
        return {};
    RestoreDevice restore;
    int forward = 0, reverse = 0;
    check(cudaDeviceCanAccessPeer(&forward, primary, peer));
    check(cudaDeviceCanAccessPeer(&reverse, peer, primary));
    cudaDeviceProp a{}, b{};
    check(cudaGetDeviceProperties(&a, primary));
    check(cudaGetDeviceProperties(&b, peer));
    if (!forward || !reverse || a.major != 12 || a.minor != 0 || b.major != a.major || b.minor != a.minor)
        return {};
    std::set<std::string> files;
    for (int i = 0; i < cfg.layers; ++i)
        for (auto name : {".mlp.fc1", ".mlp.fc2"}) {
            auto it = shards.find("blocks." + std::to_string(i) + name);
            if (it == shards.end())
                return {};
            files.insert(it->second);
        }
    check(cudaSetDevice(peer));
    size_t available, total;
    check(cudaMemGetInfo(&available, &total));
    const size_t weightBytes = size_t(3) * cfg.hidden * cfg.ffn * cfg.layers;
    // Each shard owns a 32 MiB workspace. Leave room for scales, runtimes and activations.
    const size_t workspaceBytes = files.size() * (size_t(32) << 20);
    if (available < weightBytes + workspaceBytes + (size_t(1) << 30))
        return {};
    try {
        Fp8Weights::validateDevice(peer, true);
    } catch (const std::runtime_error &) {
        // Older multi-device cuBLAS runtimes can still serve the single-device path.
        return {};
    }
    return std::unique_ptr<FfnPeer>(new FfnPeer(std::make_unique<Impl>(primary, peer, source, shards)));
}
Tensor FfnPeer::run(Runtime &primary, const Tensor &x, const std::string &up, const std::string &down,
                    Fp8Weights &upWeights, Fp8Weights &downWeights) {
    auto &p = *impl_;
    const int localRows = ((x.rows + 1) / 2 + 15) / 16 * 16;
    const int tailRows = x.rows - localRows;
    Tensor out(x.rows, x.cols);
    auto local = out.rowSlice(0, localRows), tail = out.rowSlice(localRows, tailRows);
    nomas::enableTensorPeerAccess(x, {p.device});
    nomas::enableTensorPeerAccess(out, {p.device});
    auto evaluate = [&](Runtime &runtime, const Tensor &input, Fp8Weights &w1, Fp8Weights &w2, Tensor *target) {
        auto hidden = runtime.siluMultiply(w1.linearPacked(input, {up}), true);
        return w2.linearPacked(hidden, {down}, target);
    };
    // The peer can start as soon as normalization and output allocation finish.
    // Its output is disjoint from the primary GEMM's destination.
    check(cudaEventRecord(p.inputReady.get(), cudaStreamPerThread));
    try {
        p.runtime.activate();
        check(cudaStreamWaitEvent(cudaStreamPerThread, p.inputReady.get()));
        {
            Tensor input(tailRows, x.cols);
            nomas::enableTensorPeerAccess(input, {primary.device()});
            check(cudaMemcpyPeerAsync(input.data, p.device, x.rowSlice(localRows, tailRows).data, primary.device(),
                                      input.bytes(), cudaStreamPerThread));
            auto result = evaluate(p.runtime, input, p.weight(up), p.weight(down), nullptr);
            nomas::enableTensorPeerAccess(result, {primary.device()});
            check(cudaMemcpyPeerAsync(tail.data, primary.device(), result.data, p.device, result.bytes(),
                                      cudaStreamPerThread));
            check(cudaEventRecord(p.tailDone.get(), cudaStreamPerThread));
        }
        primary.activate();
        evaluate(primary, x.rowSlice(0, localRows), upWeights, downWeights, &local);
        check(cudaStreamWaitEvent(cudaStreamPerThread, p.tailDone.get()));
    } catch (...) {
        // Detached peer copies may still reference primary inputs or output on an error.
        cudaSetDevice(p.device);
        cudaStreamSynchronize(cudaStreamPerThread);
        cudaSetDevice(primary.device());
        throw;
    }
    return out;
}
} // namespace fastllm::duo
