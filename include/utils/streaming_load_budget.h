#pragma once

// Host-side safeguards for large Safetensors loads.  This header is
// dependency-free so the loader and CPU regression tests use exactly the same
// overflow, range and budget rules.

#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace fastllm {

constexpr uint64_t kQwen4HostLoadBudgetBytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;

class StreamingLoadBudget {
public:
    class Lease {
    public:
        Lease() = default;
        Lease(StreamingLoadBudget *owner, uint64_t bytes)
            : owner_(owner), bytes_(bytes) {}
        Lease(const Lease &) = delete;
        Lease &operator=(const Lease &) = delete;
        Lease(Lease &&other) noexcept
            : owner_(other.owner_), bytes_(other.bytes_) {
            other.owner_ = nullptr;
            other.bytes_ = 0;
        }
        Lease &operator=(Lease &&other) noexcept {
            if (this != &other) {
                Release();
                owner_ = other.owner_;
                bytes_ = other.bytes_;
                other.owner_ = nullptr;
                other.bytes_ = 0;
            }
            return *this;
        }
        ~Lease() { Release(); }
        void Release() noexcept {
            if (owner_ != nullptr && bytes_ != 0) {
                owner_->Release(bytes_);
                owner_ = nullptr;
                bytes_ = 0;
            }
        }
        uint64_t Bytes() const { return bytes_; }
        explicit operator bool() const { return owner_ != nullptr; }

    private:
        StreamingLoadBudget *owner_ = nullptr;
        uint64_t bytes_ = 0;
    };

    explicit StreamingLoadBudget(uint64_t limit = kQwen4HostLoadBudgetBytes)
        : limit_(limit) {
        if (limit_ == 0) throw std::invalid_argument("streaming load budget must be non-zero");
    }

    Lease Acquire(uint64_t bytes) {
        Validate(bytes);
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return bytes <= limit_ - in_flight_; });
        in_flight_ += bytes;
        peak_ = std::max(peak_, in_flight_);
        return Lease(this, bytes);
    }

    bool TryAcquire(uint64_t bytes, Lease &lease) {
        Validate(bytes);
        std::lock_guard<std::mutex> lock(mutex_);
        if (bytes > limit_ - in_flight_) return false;
        in_flight_ += bytes;
        peak_ = std::max(peak_, in_flight_);
        lease = Lease(this, bytes);
        return true;
    }

    uint64_t Limit() const { return limit_; }
    uint64_t InFlight() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return in_flight_;
    }
    uint64_t Peak() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return peak_;
    }

private:
    void Validate(uint64_t bytes) const {
        if (bytes == 0 || bytes > limit_) {
            throw std::invalid_argument("streaming load request exceeds its hard budget");
        }
    }
    void Release(uint64_t bytes) noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            in_flight_ -= bytes;
        }
        cv_.notify_all();
    }

    const uint64_t limit_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    uint64_t in_flight_ = 0;
    uint64_t peak_ = 0;
};

inline bool ReadFileRange(const std::string &file_name,
                          uint64_t tensor_offset, uint64_t tensor_bytes,
                          uint64_t relative_offset, void *destination,
                          uint64_t byte_count, uint64_t destination_capacity,
                          std::string *error = nullptr) {
    auto fail = [&](const char *message) {
        if (error != nullptr) *error = message;
        return false;
    };
    if (destination == nullptr || byte_count == 0 || byte_count > destination_capacity ||
        relative_offset > tensor_bytes || byte_count > tensor_bytes - relative_offset ||
        tensor_offset > std::numeric_limits<uint64_t>::max() - relative_offset ||
        tensor_offset + relative_offset > std::numeric_limits<uint64_t>::max() - byte_count) {
        return fail("range exceeds tensor or destination bounds");
    }
    const uint64_t absolute_offset = tensor_offset + relative_offset;
#ifndef _WIN32
    const int fd = open(file_name.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return fail("cannot open tensor file");
    uint8_t *out = static_cast<uint8_t *>(destination);
    uint64_t done = 0;
    while (done < byte_count) {
        const size_t requested = static_cast<size_t>(std::min<uint64_t>(
            byte_count - done, static_cast<uint64_t>(std::numeric_limits<ssize_t>::max())));
        const ssize_t got = pread(fd, out + done, requested,
                                  static_cast<off_t>(absolute_offset + done));
        if (got <= 0) {
            const int saved_errno = errno;
            close(fd);
            if (error != nullptr) *error = saved_errno == EINTR ? "interrupted range read" : "short or failed range read";
            return false;
        }
        done += static_cast<uint64_t>(got);
    }
    close(fd);
    return true;
#else
    FILE *file = std::fopen(file_name.c_str(), "rb");
    if (file == nullptr) return fail("cannot open tensor file");
    if (_fseeki64(file, static_cast<__int64>(absolute_offset), SEEK_SET) != 0) {
        std::fclose(file);
        return fail("cannot seek tensor file");
    }
    const size_t got = std::fread(destination, 1, static_cast<size_t>(byte_count), file);
    std::fclose(file);
    return got == byte_count ? true : fail("short or failed range read");
#endif
}

} // namespace fastllm
