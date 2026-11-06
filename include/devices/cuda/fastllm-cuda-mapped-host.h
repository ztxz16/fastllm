#pragma once
#include "fastllm-cuda-host-upload.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace fastllm { namespace cuda {

// Contiguous host records with a verified contiguous CUDA mapping. Large
// cudaHostAlloc calls can fail on some driver/host configurations; bounded registrations
// avoid requiring one large driver page-pointer allocation. No record layout
// or GPU address arithmetic changes when this fallback is used.
// Explicit ordinary storage has no device mapping; its immutable bytes are
// uploaded through bounded staging by CopyToDeviceAsync.
class MappedHostBuffer {
    void *host_ = nullptr, *device_ = nullptr;
    size_t mappedBytes_ = 0, bytes_ = 0;
    bool pageable_ = false;
    std::vector<void *> registrations_;
public:
    MappedHostBuffer() = default;
    MappedHostBuffer(const MappedHostBuffer &) = delete;
    MappedHostBuffer &operator=(const MappedHostBuffer &) = delete;
    ~MappedHostBuffer() { Reset(); }
    void *Host() const { return host_; }
    void *Device() const { return device_; }
    size_t Registrations() const { return registrations_.size(); }
    void Reset() {
        if (!host_) return;
        if (pageable_) std::free(host_);
        else {
#if defined(__linux__)
        if (mappedBytes_) {
            for (void *ptr : registrations_) cudaHostUnregister(ptr);
            munmap(host_, mappedBytes_);
        } else
#endif
        cudaFreeHost(host_);
        }
        host_ = device_ = nullptr;
        mappedBytes_ = 0;
        bytes_ = 0;
        pageable_ = false;
        registrations_.clear();
    }
    cudaError_t CopyToDeviceAsync(void *destination, const void *source, size_t bytes,
                                 cudaStream_t stream) const {
        const auto address = reinterpret_cast<uintptr_t>(source);
        const auto base = reinterpret_cast<uintptr_t>(host_);
        if (!host_ || address < base || address - base > bytes_ ||
            bytes > bytes_ - (address - base)) return cudaErrorInvalidValue;
        if (!bytes) return cudaSuccess;
        if (pageable_) return UploadImmutableHostWeight(destination, source, bytes, false, stream);
        if (registrations_.empty())
            return cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, stream);
        // CUDA permits direct GPU loads across adjacent mapped registrations,
        // but a host-to-device DMA request must stay within one registration.
        auto next = std::upper_bound(registrations_.begin(), registrations_.end(), address,
            [](uintptr_t value, const void *ptr) { return value < reinterpret_cast<uintptr_t>(ptr); });
        auto *dst = static_cast<uint8_t *>(destination);
        auto *src = static_cast<const uint8_t *>(source);
        while (bytes) {
            const uintptr_t end = next == registrations_.end() ? base + mappedBytes_ :
                reinterpret_cast<uintptr_t>(*next);
            const size_t count = std::min(bytes, size_t(end - reinterpret_cast<uintptr_t>(src)));
            const auto state = cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream);
            if (state != cudaSuccess) return state;
            dst += count; src += count; bytes -= count;
            if (next != registrations_.end()) ++next;
        }
        return cudaSuccess;
    }
    cudaError_t Allocate(size_t bytes, bool mapped = true) {
        Reset();
        if (!bytes) return cudaErrorInvalidValue;
        bytes_ = bytes;
        if (!mapped) {
            host_ = std::malloc(bytes);
            pageable_ = host_ != nullptr;
            return host_ ? cudaSuccess : cudaErrorMemoryAllocation;
        }
        cudaError_t state = cudaHostAlloc(&host_, bytes, cudaHostAllocMapped | cudaHostAllocPortable);
        if (state == cudaSuccess) {
            state = cudaHostGetDevicePointer(&device_, host_, 0);
            if (state == cudaSuccess && device_) return cudaSuccess;
            Reset();
            cudaGetLastError();
            return state == cudaSuccess ? cudaErrorInvalidValue : state;
        }
        cudaGetLastError();
        host_ = nullptr;
#if defined(__linux__)
        const long pageSize = sysconf(_SC_PAGESIZE);
        if (pageSize <= 0) return state;
        const size_t page = size_t(pageSize);
        if (bytes > std::numeric_limits<size_t>::max() - (page - 1)) return state;
        const size_t extent = (bytes + page - 1) / page * page;
        void *mapping = mmap(nullptr, extent, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapping == MAP_FAILED) return cudaErrorMemoryAllocation;
        host_ = mapping;
        mappedBytes_ = extent;
        size_t limit = std::max(page, size_t(32) << 20);
        for (size_t offset = 0; offset < extent;) {
            size_t count = std::min(limit, extent - offset);
            void *ptr = static_cast<uint8_t *>(host_) + offset;
            state = cudaHostRegister(ptr, count, cudaHostRegisterMapped | cudaHostRegisterPortable);
            if (state != cudaSuccess) {
                cudaGetLastError();
                if (count > page) {
                    limit = std::max(page, (count / 2 / page) * page);
                    continue;
                }
                Reset();
                return state;
            }
            registrations_.push_back(ptr);
            void *mapped = nullptr;
            state = cudaHostGetDevicePointer(&mapped, ptr, 0);
            if (offset == 0) device_ = mapped;
            if (state != cudaSuccess || !mapped ||
                reinterpret_cast<uintptr_t>(mapped) != reinterpret_cast<uintptr_t>(device_) + offset) {
                cudaGetLastError();
                Reset();
                return state == cudaSuccess ? cudaErrorInvalidValue : state;
            }
            offset += count;
        }
        return cudaSuccess;
#else
        return state;
#endif
    }
};

}} // namespace fastllm::cuda
