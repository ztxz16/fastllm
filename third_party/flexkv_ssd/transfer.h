/*
 * Extracted from FlexKV commit 738ddc141a198b4e20de6c5d1f0128e387f7fdb2.
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 * See LICENSE and SOURCE.md. Only linkage/namespace/includes are adapted.
 */
#pragma once
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <sys/uio.h>
#include <unistd.h>
#include <vector>
namespace fastllm_flexkv_ssd {
struct PackedSpan {
    void *cpu_ptr;
    uint32_t comp_bytes;
    uint32_t *dst_table_entry;
};
// Retry to completion (short/interrupted I/O).
inline bool transfer_full(int fd, void *buf, size_t count, off_t offset,
                          bool is_read) {
  size_t done = 0;
  char *p = static_cast<char *>(buf);
  while (done < count) {
    ssize_t n = is_read
                    ? pread(fd, p + done, count - done, offset + (off_t)done)
                    : pwrite(fd, p + done, count - done, offset + (off_t)done);
    if (n < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (n == 0) break;  // EOF
    done += (size_t)n;
  }
  return done == count;
}

// Retry to completion; rebuilds residual iovecs on short I/O.
inline bool transfer_iov_full(int fd, const struct iovec *iov, int iovcnt,
                              off_t base_offset, size_t total, bool is_read) {
  size_t done = 0;
  int head_i = 0;
  size_t head_off = 0;
  while (done < total) {
    std::vector<struct iovec> rem;
    if (head_off > 0) {
      rem.push_back({(char *)iov[head_i].iov_base + head_off,
                     iov[head_i].iov_len - head_off});
      for (int j = head_i + 1; j < iovcnt; ++j) rem.push_back(iov[j]);
    } else {
      for (int j = head_i; j < iovcnt; ++j) rem.push_back(iov[j]);
    }
    ssize_t n = is_read
                    ? preadv(fd, rem.data(), (int)rem.size(),
                             base_offset + (off_t)done)
                    : pwritev(fd, rem.data(), (int)rem.size(),
                              base_offset + (off_t)done);
    if (n < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (n == 0) break;  // EOF
    done += (size_t)n;
    size_t adv = (size_t)n;
    while (head_i < iovcnt) {
      size_t avail = iov[head_i].iov_len - head_off;
      if (adv < avail) {
        head_off += adv;
        adv = 0;
        break;
      }
      adv -= avail;
      head_off = 0;
      ++head_i;
    }
  }
  return done == total;
}

inline void do_transfer_packed_blocks(
    int fd,
    int64_t ssd_off,
    int64_t transfer_bytes,
    bool is_read,
    void *staging_buffer,
    const std::vector<PackedSpan> &spans) {
  char *staging = reinterpret_cast<char *>(staging_buffer);

  // H2DISK gather (host memory only, no I/O yet): the compressed chunks are
  // scattered across the CPU tensor (one per chunk-size slot). memcpy each into
  // the contiguous host staging buffer, then zero-pad the tail up to the
  // 512-aligned transfer length, so the whole block can go out in one write.
  if (!is_read) {
    int64_t cursor = 0;
    for (const auto &span : spans) {
      memcpy(staging + cursor, span.cpu_ptr,
             static_cast<size_t>(span.comp_bytes));
      *span.dst_table_entry = span.comp_bytes;
      cursor += span.comp_bytes;
    }
    if (transfer_bytes > cursor) {
      memset(staging + cursor, 0, static_cast<size_t>(transfer_bytes - cursor));
    }
  }

  // The one block I/O: a single full pread/pwrite of the packed range, looping
  // over partial transfers and retrying on EINTR.
  int64_t done = 0;
  while (done < transfer_bytes) {
    ssize_t rc;
    do {
      rc = is_read
               ? pread(fd, staging + done, transfer_bytes - done, ssd_off + done)
               : pwrite(fd, staging + done, transfer_bytes - done, ssd_off + done);
    } while (rc < 0 && errno == EINTR);
    if (rc <= 0) {
      throw std::runtime_error(
          is_read ? "transfer_kv_blocks_ssd_packed: read failed"
                  : "transfer_kv_blocks_ssd_packed: write failed");
    }
    done += rc;
  }

  // DISK2H scatter (host memory only): the pread above filled the host staging
  // buffer with this block's tightly-packed compressed chunks. Walk the spans in
  // the same order they were packed and, for each chunk: memcpy its comp_bytes
  // from the running staging cursor back to its slot in the CPU tensor
  // (span.cpu_ptr), record comp_bytes into the CPU-side size table
  // (span.dst_table_entry), and advance the cursor tightly by comp_bytes. Any
  // 512-aligned tail left in staging is O_DIRECT padding and is ignored.
  if (is_read) {
    int64_t cursor = 0;
    for (const auto &span : spans) {
      memcpy(span.cpu_ptr, staging + cursor,
             static_cast<size_t>(span.comp_bytes));
      *span.dst_table_entry = span.comp_bytes;
      cursor += span.comp_bytes;
    }
  }

}


} // namespace fastllm_flexkv_ssd
