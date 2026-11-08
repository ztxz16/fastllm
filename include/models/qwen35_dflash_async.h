#ifndef FASTLLM_QWEN35_DFLASH_ASYNC_H
#define FASTLLM_QWEN35_DFLASH_ASYNC_H

#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace fastllm {

// Request-local, single-consumer, two-slot producer/consumer ring. Preparation
// runs on the caller before the worker is created; the worker only consumes
// already-prepared weights and request-local capture payloads.
template <class Payload>
class Qwen35DFlashAsyncRing {
public:
    struct Slot {
        Payload payload;
        void *readyEvent = nullptr;
        void *consumedEvent = nullptr;
        enum class State { Empty, Published, Consuming, Done, Failed } state = State::Empty;
        std::exception_ptr error;
    };
    using Consume = std::function<void(Slot &)>;
    using Prepare = std::function<void()>;

    Qwen35DFlashAsyncRing(int producerDevice, int consumerDevice,
                          Consume consume, Prepare prepare = Prepare())
        : producerDevice_(producerDevice), consumerDevice_(consumerDevice),
          consume_(std::move(consume)) {
        if (producerDevice_ < 0 || consumerDevice_ < 0 || !consume_) {
            throw std::invalid_argument("invalid DFlash async ring configuration");
        }
        const int previous = FastllmCudaGetDevice();
        try {
            // Resolve every lazy DFlash allocation on the caller before the
            // persistent worker receives ownership of the request.
            FastllmCudaClearThreadError();
            if (prepare) prepare();
            if (FastllmCudaGetThreadError()) {
                throw std::runtime_error("DFlash async preparation failed");
            }
            FastllmCudaSetDevice(producerDevice_);
            for (Slot &slot : slots_) {
                slot.readyEvent = FastllmCudaEventCreate();
                if (slot.readyEvent == nullptr) {
                    throw std::runtime_error("DFlash ready event creation failed");
                }
            }
            FastllmCudaSetDevice(consumerDevice_);
            for (Slot &slot : slots_) {
                slot.consumedEvent = FastllmCudaEventCreate();
                if (slot.consumedEvent == nullptr) {
                    throw std::runtime_error("DFlash consumed event creation failed");
                }
            }
            FastllmCudaSetDevice(previous);
        } catch (...) {
            DestroyEvents();
            FastllmCudaSetDevice(previous);
            throw;
        }
        worker_ = std::thread([this]() { Worker(); });
    }

    ~Qwen35DFlashAsyncRing() {
        Cancel();
        DestroyEvents();
    }

    Qwen35DFlashAsyncRing(const Qwen35DFlashAsyncRing &) = delete;
    Qwen35DFlashAsyncRing &operator=(const Qwen35DFlashAsyncRing &) = delete;

    // The reference is intentional: if the worker has already failed, the
    // caller keeps the payload and can replay it on the serial path.
    int Publish(Payload &payload, int preferredSlot) {
        const int index = preferredSlot & 1;
        Slot &slot = slots_[index];
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&]() {
                return stop_ || slot.state == Slot::State::Empty ||
                       slot.state == Slot::State::Done;
            });
            if (stop_) RethrowFailureLocked();
        }
        FastllmCudaSetDevice(producerDevice_);
        FastllmCudaEventSynchronize(slot.consumedEvent);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) RethrowFailureLocked();
            submitted_.erase(std::remove(submitted_.begin(), submitted_.end(), index),
                             submitted_.end());
            slot.payload = std::move(payload);
            slot.error = nullptr;
            slot.state = Slot::State::Published;
            FastllmCudaEventRecordCurrentThread(slot.readyEvent);
            if (FastllmCudaGetThreadError()) {
                slot.state = Slot::State::Failed;
                slot.error = std::make_exception_ptr(
                    std::runtime_error("DFlash ready event record failed"));
                failure_ = slot.error;
                stop_ = true;
                cv_.notify_all();
                RethrowFailureLocked();
            }
            ready_.push_back(index);
            submitted_.push_back(index);
        }
        cv_.notify_all();
        return index;
    }

    void Wait(int index) {
        Slot &slot = slots_[index & 1];
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&]() {
            return slot.state == Slot::State::Done ||
                   slot.state == Slot::State::Failed || stop_;
        });
        if (slot.state != Slot::State::Done) {
            RethrowFailureLocked(slot.error);
        }
        lock.unlock();
        FastllmCudaSetDevice(consumerDevice_);
        FastllmCudaEventSynchronize(slot.consumedEvent);
    }

    void Drain() {
        for (int index : submitted_) Wait(index);
    }

    // Stop after a worker error and return every published-but-not-completed
    // payload in publication order. Completed payloads already committed their
    // context and are deliberately not replayed.
    std::vector<Payload> CancelAndTakePending() {
        Cancel();
        std::vector<Payload> pending;
        std::lock_guard<std::mutex> lock(mutex_);
        for (int index : submitted_) {
            Slot &slot = slots_[index];
            if (slot.state != Slot::State::Done &&
                slot.state != Slot::State::Empty) {
                pending.push_back(std::move(slot.payload));
                slot.state = Slot::State::Empty;
            }
        }
        submitted_.clear();
        return pending;
    }

    void Cancel() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

private:
    [[noreturn]] void RethrowFailureLocked() const {
        if (failure_) std::rethrow_exception(failure_);
        throw std::runtime_error("DFlash async ring stopped");
    }

    [[noreturn]] static void RethrowFailureLocked(const std::exception_ptr &error) {
        if (error) std::rethrow_exception(error);
        throw std::runtime_error("DFlash async slot failed");
    }

    void Worker() {
        FastllmCudaSetDevice(consumerDevice_);
        FastllmCudaClearThreadError();
        for (;;) {
            int index = -1;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [&]() { return stop_ || !ready_.empty(); });
                if (ready_.empty()) break;
                index = ready_.front();
                ready_.erase(ready_.begin());
                slots_[index].state = Slot::State::Consuming;
            }
            try {
                FastllmCudaCurrentThreadStreamWaitEvent(slots_[index].readyEvent);
                if (FastllmCudaGetThreadError()) {
                    throw std::runtime_error("DFlash ready wait failed");
                }
                consume_(slots_[index]);
                FastllmCudaEventRecordCurrentThread(slots_[index].consumedEvent);
                if (FastllmCudaGetThreadError()) {
                    throw std::runtime_error("DFlash consumed event record failed");
                }
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    slots_[index].state = Slot::State::Done;
                }
                cv_.notify_all();
            } catch (...) {
                try {
                    FastllmCudaSyncCurrentThreadStream();
                } catch (...) {}
                const std::exception_ptr error = std::current_exception();
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    slots_[index].state = Slot::State::Failed;
                    slots_[index].error = error;
                    failure_ = error;
                    stop_ = true;
                    for (int pending : ready_) {
                        slots_[pending].state = Slot::State::Failed;
                        slots_[pending].error = error;
                    }
                    ready_.clear();
                }
                cv_.notify_all();
                break;
            }
        }
    }

    void DestroyEvents() {
        const int previous = FastllmCudaGetDevice();
        if (producerDevice_ >= 0) {
            FastllmCudaSetDevice(producerDevice_);
            for (Slot &slot : slots_) {
                if (slot.readyEvent != nullptr) {
                    FastllmCudaEventDestroy(slot.readyEvent);
                    slot.readyEvent = nullptr;
                }
            }
        }
        if (consumerDevice_ >= 0) {
            FastllmCudaSetDevice(consumerDevice_);
            for (Slot &slot : slots_) {
                if (slot.consumedEvent != nullptr) {
                    FastllmCudaEventDestroy(slot.consumedEvent);
                    slot.consumedEvent = nullptr;
                }
            }
        }
        FastllmCudaSetDevice(previous);
    }

    int producerDevice_;
    int consumerDevice_;
    Consume consume_;
    std::array<Slot, 2> slots_;
    std::vector<int> ready_;
    std::vector<int> submitted_;
    std::exception_ptr failure_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
    bool stop_ = false;
};

} // namespace fastllm
#endif // USE_CUDA
#endif // FASTLLM_QWEN35_DFLASH_ASYNC_H
