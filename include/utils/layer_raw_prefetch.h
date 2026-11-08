#pragma once

#include "streaming_load_budget.h"
#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

namespace fastllm {

struct RawPrefetchRange {
    std::string name, file;
    uint64_t offset = 0, bytes = 0;
    bool operator==(const RawPrefetchRange &other) const {
        return name == other.name && file == other.file &&
               offset == other.offset && bytes == other.bytes;
    }
};

struct PrefetchedRawTensor {
    RawPrefetchRange range;
    StreamingLoadBudget::Lease lease;
    // Free payload before returning its budget lease.
    std::unique_ptr<uint8_t[]> data;
    std::atomic<uint64_t> *adoptedCalls = nullptr, *adoptedBytes = nullptr;
    explicit operator bool() const { return data != nullptr; }
};

// Only the tensor currently being loaded on this thread may adopt these bytes.
// Scale reads and other tensors cannot consume a payload with a different
// name/file/range. Scope restoration also handles nested loader operations.
class ScopedPrefetchedRawTensor {
public:
    explicit ScopedPrefetchedRawTensor(PrefetchedRawTensor &tensor)
        : previous_(active_) { active_ = &tensor; }
    ~ScopedPrefetchedRawTensor() { active_ = previous_; }
    ScopedPrefetchedRawTensor(const ScopedPrefetchedRawTensor &) = delete;
    ScopedPrefetchedRawTensor &operator=(const ScopedPrefetchedRawTensor &) = delete;
    static uint8_t *Take(const std::string &name, const std::string &file,
                         uint64_t offset, uint64_t bytes) {
        if (active_ != nullptr && active_->data != nullptr &&
            active_->range == RawPrefetchRange{name, file, offset, bytes}) {
            if (active_->adoptedCalls != nullptr) active_->adoptedCalls->fetch_add(1);
            if (active_->adoptedBytes != nullptr) active_->adoptedBytes->fetch_add(bytes);
            return active_->data.release();
        }
        return nullptr;
    }
private:
    inline static thread_local PrefetchedRawTensor *active_ = nullptr;
    PrefetchedRawTensor *previous_;
};

// One exact next-layer window. The background worker touches only immutable
// file descriptors/ranges and its own buffers, never model or tensor maps.
// Admission is nonblocking and uses the loader's existing raw-memory budget;
// inability to prefetch simply leaves the original reader in use.
// Join before consuming; finish all consumers before Start/Clear.
class BoundedLayerRawPrefetch {
public:
    struct Statistics {
        uint64_t windows, readCalls, readBytes, readUs, joinWaitUs;
        uint64_t adoptedCalls, adoptedBytes, readFailures, skippedWindows, maxWindowBytes, readerWorkers;
    };
    BoundedLayerRawPrefetch(StreamingLoadBudget &budget, uint64_t maxWindowBytes,
                            unsigned readerThreads = 1)
        : budget_(budget), maxWindowBytes_(maxWindowBytes),
          readerThreads_(std::max(1U, readerThreads)) {}
    ~BoundedLayerRawPrefetch() { Clear(); }
    BoundedLayerRawPrefetch(const BoundedLayerRawPrefetch &) = delete;
    BoundedLayerRawPrefetch &operator=(const BoundedLayerRawPrefetch &) = delete;

    bool Start(const std::vector<RawPrefetchRange> &ranges) {
        Clear();
        uint64_t total = 0;
        std::set<std::string> names;
        for (const auto &range : ranges) {
            if (range.bytes == 0 || total > maxWindowBytes_ ||
                range.bytes > maxWindowBytes_ - total ||
                !names.insert(range.name).second) {
                skippedWindows_.fetch_add(1);
                return false;
            }
            total += range.bytes;
        }
        if (ranges.empty() || total > budget_.Limit()) return false;
        for (const auto &range : ranges) {
            auto &entry = entries_.try_emplace(range.name).first->second;
            entry.tensor.range = range;
            entry.tensor.adoptedCalls = &adoptedCalls_;
            entry.tensor.adoptedBytes = &adoptedBytes_;
            if (!budget_.TryAcquire(range.bytes, entry.tensor.lease)) {
                entries_.clear();
                skippedWindows_.fetch_add(1);
                return false;
            }
        }
        ranges_ = ranges;
        cancelled_.store(false);
        try {
            worker_ = std::thread([this] {
                const auto start = Clock::now();
                std::atomic<size_t> next{0};
                auto readRanges = [&] {
                    while (!cancelled_.load()) {
                        const size_t index = next.fetch_add(1);
                        if (index >= ranges_.size()) break;
                        const auto &range = ranges_[index];
                        auto &tensor = entries_.at(range.name).tensor;
                        bool ok = false;
                        try {
                            tensor.data.reset(new uint8_t[range.bytes]);
                            ok = ReadFileRange(range.file, range.offset, range.bytes, 0,
                                               tensor.data.get(), range.bytes, range.bytes);
                        } catch (const std::exception &) {
                            ok = false;
                        }
                        if (!ok) {
                            tensor.data.reset();
                            tensor.lease.Release();
                            readFailures_.fetch_add(1);
                        } else {
                            readCalls_.fetch_add(1);
                            readBytes_.fetch_add(range.bytes);
                        }
                    }
                };
                std::vector<std::thread> readers;
                const size_t count = std::min<size_t>(readerThreads_, ranges_.size());
                try {
                    readers.reserve(count - 1);
                    for (size_t i = 1; i < count; ++i) readers.emplace_back(readRanges);
                } catch (const std::exception &) {
                    // Existing readers plus this coordinator still drain every
                    // range when another CPU thread cannot be created.
                }
                readerWorkers_.fetch_add(readers.size() + 1);
                readRanges();
                for (auto &reader : readers) reader.join();
                readUs_.fetch_add(MicrosSince(start));
            });
        } catch (const std::exception &) {
            entries_.clear();
            ranges_.clear();
            skippedWindows_.fetch_add(1);
            return false;
        }
        windows_.fetch_add(1);
        maxWindowObserved_.store(std::max(maxWindowObserved_.load(), total));
        return true;
    }

    void Join() {
        if (worker_.joinable()) {
            const auto start = Clock::now();
            worker_.join();
            joinWaitUs_.fetch_add(MicrosSince(start));
        }
    }
    void Clear() {
        cancelled_.store(true);
        Join();
        entries_.clear();
        ranges_.clear();
    }
    PrefetchedRawTensor Take(const RawPrefetchRange &range) {
        if (worker_.joinable()) return {};
        auto it = entries_.find(range.name);
        if (it == entries_.end()) return {};
        std::lock_guard<std::mutex> guard(it->second.mutex);
        auto &tensor = it->second.tensor;
        if (!tensor || !(tensor.range == range)) return {};
        return std::move(tensor);
    }
    Statistics Stats() const {
        return {windows_.load(), readCalls_.load(), readBytes_.load(), readUs_.load(),
                joinWaitUs_.load(), adoptedCalls_.load(), adoptedBytes_.load(),
                readFailures_.load(), skippedWindows_.load(), maxWindowObserved_.load(), readerWorkers_.load()};
    }

private:
    using Clock = std::chrono::steady_clock;
    static uint64_t MicrosSince(Clock::time_point start) {
        return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
    }
    struct Entry { std::mutex mutex; PrefetchedRawTensor tensor; };
    StreamingLoadBudget &budget_;
    const uint64_t maxWindowBytes_;
    const unsigned readerThreads_;
    std::map<std::string, Entry> entries_;
    std::vector<RawPrefetchRange> ranges_;
    std::thread worker_;
    std::atomic<bool> cancelled_{false};
    std::atomic<uint64_t> windows_{0}, readCalls_{0}, readBytes_{0}, readUs_{0};
    std::atomic<uint64_t> joinWaitUs_{0}, adoptedCalls_{0}, adoptedBytes_{0};
    std::atomic<uint64_t> readFailures_{0}, skippedWindows_{0}, maxWindowObserved_{0};
    std::atomic<uint64_t> readerWorkers_{0};
};

} // namespace fastllm
