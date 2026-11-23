#pragma once
#include "devices/cuda/fastllm-cuda-host-upload.h"
#include "devices/moe_frequency_policy.h"
#include <stdexcept>

namespace fastllm { namespace cuda {

// The layer owns its publication fence so destruction drains outstanding
// uploads before freeing expert records or their immutable host sources.
class HybridMoeAdmissions {
    static void check(cudaError_t error) {
        if (error != cudaSuccess) throw std::runtime_error(std::string("Hybrid MoE admission: ") + cudaGetErrorString(error));
    }
    std::unique_ptr<ImmutableHostUploadRing> ring_;
    cudaEvent_t readers_ = nullptr;
    cudaStream_t copy_ = nullptr;
public:
    struct Publication {
        cudaEvent_t ready = nullptr;
        Publication() = default;
        Publication(const Publication &) = delete;
        ~Publication() {
            if (ready) { cudaEventSynchronize(ready); cudaEventDestroy(ready); }
        }
        void wait(cudaStream_t stream, bool capture) const {
            if (ready) check(cudaStreamWaitEvent(stream, ready, capture ? cudaEventWaitExternal : cudaEventWaitDefault));
        }
    };
    HybridMoeAdmissions() {
        int device;
        check(cudaGetDevice(&device));
        auto ring = std::make_unique<ImmutableHostUploadRing>();
        if (!ring->Init(device)) return;
        check(cudaEventCreateWithFlags(&readers_, cudaEventDisableTiming));
        auto status = cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking);
        if (status != cudaSuccess) { cudaEventDestroy(readers_); check(status); }
        ring_ = std::move(ring);
    }
    HybridMoeAdmissions(const HybridMoeAdmissions &) = delete;
    ~HybridMoeAdmissions() {
        if (copy_) cudaStreamSynchronize(copy_);
        if (readers_) cudaEventDestroy(readers_);
        if (copy_) cudaStreamDestroy(copy_);
    }
    void attach(Publication &layer) {
        if (copy_) check(cudaEventCreateWithFlags(&layer.ready, cudaEventDisableTiming));
    }
    // Upload maps the policy key and slot to the layer's weight format and returns
    // its fence. A synchronous fallback uses the consumer stream and no host ring.
    template<class Partition, class Upload>
    void publish(MoeFrequencyPolicy &policy, cudaStream_t consumer, Partition partition, Upload upload) {
        auto selected = policy.EndStep();
        if (selected.empty()) return;
        try {
            if (!copy_) {
                for (auto entry : selected) upload(entry, consumer, nullptr);
                // Without a publication event, a later request may read these
                // slots on a different stream as soon as publish returns.
                check(cudaStreamSynchronize(consumer));
                return;
            }
            check(cudaEventRecord(readers_, consumer));
            check(cudaStreamWaitEvent(copy_, readers_, 0));
            std::stable_sort(selected.begin(), selected.end(), [&](auto a, auto b) {
                return partition(a.key) < partition(b.key);
            });
            for (size_t i = 0; i < selected.size(); ++i) {
                auto entry = selected[i];
                auto &layer = upload(entry, copy_, ring_.get());
                if (i + 1 == selected.size() || partition(selected[i + 1].key) != partition(entry.key))
                    check(cudaEventRecord(layer.ready, copy_));
            }
        } catch (...) {
            cudaStreamSynchronize(copy_ ? copy_ : consumer);
            for (auto entry : selected) policy.Evict(entry.key);
            throw;
        }
    }
};
}} // namespace fastllm::cuda
