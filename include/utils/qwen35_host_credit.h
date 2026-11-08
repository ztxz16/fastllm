#ifndef FASTLLM_QWEN35_HOST_CREDIT_H
#define FASTLLM_QWEN35_HOST_CREDIT_H
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <exception>
namespace fastllm { namespace qwen35_host_credit {
enum class Result { Granted, Retry, TooLarge, Closed, Failed, OutputOccupied, PermanentCapacity };
struct State {
    std::mutex mutex;
    std::condition_variable changed;
    size_t used = 0;
    bool closed = false, failed = false;
};
// The last payload owner also owns this non-copyable credit. Sharing/transferring
// the same Credit never adds a second charge; release happens only at destruction.
class Credit {
    friend class Ledger;
    std::shared_ptr<State> state;
    size_t bytes;
public:
    Credit(std::shared_ptr<State> owner, size_t value):state(std::move(owner)),bytes(value) {}
    Credit(const Credit&)=delete;
    Credit& operator=(const Credit&)=delete;
    ~Credit() {
        { std::lock_guard<std::mutex> lock(state->mutex);
          if(bytes>state->used) std::terminate();
          state->used-=bytes;
        }
        state->changed.notify_all();
    }
    size_t Bytes() const { std::lock_guard<std::mutex> lock(state->mutex);return bytes; }
};
class Ledger {
    std::shared_ptr<State> state = std::make_shared<State>();
public:
    Result TryReserve(size_t bytes, size_t limit, size_t external,
                      std::shared_ptr<Credit> &result) {
        // Never replace a live output owner: its last reference may need this
        // same mutex to retire. The caller must keep/transfer or clear it first.
        if(result) return Result::OutputOccupied;
        std::lock_guard<std::mutex> lock(state->mutex);
        if(state->failed) return Result::Failed;
        if(state->closed) return Result::Closed;
        if(bytes>limit) return Result::TooLarge;
        // Subtraction order rejects both overflow and capacity exhaustion.
        if(external>limit-bytes || state->used>limit-bytes-external) return Result::Retry;
        auto pending=std::make_shared<Credit>(state,bytes);
        state->used+=bytes;
        result=std::move(pending);
        return Result::Granted;
    }
    Result TryGrow(const std::shared_ptr<Credit> &credit,size_t bytes,size_t limit,size_t external) {
        if(!credit || credit->state!=state) return Result::Failed;
        std::lock_guard<std::mutex> lock(state->mutex);
        if(bytes<=credit->bytes) return Result::Granted; // keep the original safe high-water claim
        if(state->failed) return Result::Failed;
        if(state->closed) return Result::Closed;
        if(bytes>limit) return Result::TooLarge;
        const size_t extra=bytes-credit->bytes;
        if(external>limit-extra || state->used>limit-extra-external) return Result::Retry;
        state->used+=extra;credit->bytes=bytes;
        return Result::Granted;
    }
    size_t Used() const { std::lock_guard<std::mutex> lock(state->mutex);return state->used; }
    void Close() { {std::lock_guard<std::mutex> lock(state->mutex);state->closed=true;}state->changed.notify_all(); }
    void Fail() { {std::lock_guard<std::mutex> lock(state->mutex);state->failed=true;}state->changed.notify_all(); }
};
} }
#endif
