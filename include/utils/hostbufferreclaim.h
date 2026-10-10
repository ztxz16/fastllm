#pragma once

#include <cstddef>
#include <cstdint>
#if defined(__linux__) && defined(__GLIBC__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace fastllm {
namespace detail {

// The caller exclusively owns this writable, unregistered host allocation and
// has finished reading it. Discard only full payload pages BEFORE delete[]:
// free-list links written by the allocator must never be discarded afterwards.
// Preserve the first page and both allocation boundaries, including any live
// neighbours and malloc metadata. Failure leaves normal delete/trim as fallback.
inline size_t DiscardUnusedHostBufferPages(void *buffer, size_t bytes) {
#if defined(__linux__) && defined(__GLIBC__)
    static const long systemPageSize = sysconf(_SC_PAGESIZE);
    if (buffer == nullptr || systemPageSize <= 0) return 0;
    const size_t pageSize = static_cast<size_t>(systemPageSize);
    const size_t prefix = pageSize - reinterpret_cast<uintptr_t>(buffer) % pageSize;
    if (bytes <= prefix) return 0;
    const size_t discardBytes = (bytes - prefix) / pageSize * pageSize;
    if (discardBytes != 0 &&
        madvise(static_cast<uint8_t*>(buffer) + prefix, discardBytes, MADV_DONTNEED) == 0) {
        return discardBytes;
    }
#endif
    return 0;
}

} // namespace detail
} // namespace fastllm
