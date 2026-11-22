#pragma once

#include <cstdint>

namespace naivert_replica_execute_top_p {

constexpr int kSamplerTpRanks = 8;
constexpr int kSamplerMailboxBytes = 128;

struct SamplerCandidate {
  float score;
  int index;
};

struct alignas(128) SamplerPeerState {
  unsigned top1_inbox_generation[kSamplerTpRanks];
  std::uint64_t top1_inbox_packed[kSamplerTpRanks];
};

static_assert(
    sizeof(SamplerPeerState) == kSamplerMailboxBytes,
    "top-1 sampler inbox ABI changed");

}  // namespace naivert_replica_execute_top_p
