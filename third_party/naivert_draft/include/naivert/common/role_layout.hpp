#pragma once

// Reverse evidence:
// - SASS uses threadIdx.x threshold gates:
//     tid <= 0x3f  for dense MLA/MoE control init
//     tid <= 0x7f  for 128-thread consumer/helper subgroups
//     tid <= 0xff  for 256-thread active consumer region
// - FlashSparseMLA init lanes match SlotDepth:
//     tid <= 4 for SlotDepth=5.
//
// Inference boundary:
// - Role names are guessed from instruction placement (LDGSTS/HMMA/SYNCS).
//   SASS proves predicates, not the original C++ names.

#include "cuda_compat.hpp"

#ifndef NAIVERT_ROLE_CONSUMER_GROUP_THREADS
#define NAIVERT_ROLE_CONSUMER_GROUP_THREADS 128
#endif

#ifndef NAIVERT_ROLE_CONSUMER_THREADS
#define NAIVERT_ROLE_CONSUMER_THREADS 256
#endif

#ifndef NAIVERT_ROLE_HELPER_BASE_THREAD
#define NAIVERT_ROLE_HELPER_BASE_THREAD 256
#endif

#ifndef NAIVERT_ROLE_HELPER_THREADS
#define NAIVERT_ROLE_HELPER_THREADS 128
#endif

#ifndef NAIVERT_ROLE_CONTROL_INIT_THREADS
#define NAIVERT_ROLE_CONTROL_INIT_THREADS 64
#endif

#ifndef NAIVERT_ROLE_HELPER_WARP
#define NAIVERT_ROLE_HELPER_WARP -1
#endif

namespace naivert {

struct StandardRoleLayout {
  static constexpr int kWarpSize = 32;
  static constexpr int kConsumerGroupThreads =
      NAIVERT_ROLE_CONSUMER_GROUP_THREADS;
  static constexpr int kConsumerThreads = NAIVERT_ROLE_CONSUMER_THREADS;
  static constexpr int kHelperBaseThread = NAIVERT_ROLE_HELPER_BASE_THREAD;
  static constexpr int kHelperThreads = NAIVERT_ROLE_HELPER_THREADS;
  static constexpr int kControlInitThreads = NAIVERT_ROLE_CONTROL_INIT_THREADS;
  static constexpr int kHelperBaseWarp =
      NAIVERT_ROLE_HELPER_WARP >= 0 ? NAIVERT_ROLE_HELPER_WARP
                                   : kHelperBaseThread / kWarpSize;
  static constexpr int kHelperThread = kHelperBaseWarp * kWarpSize;
  static constexpr bool kHelperInConsumerRegion =
      kHelperThread >= 0 && kHelperThread < kConsumerThreads;
  static constexpr int kConsumerActiveThreads =
      kConsumerThreads - (kHelperInConsumerRegion ? kWarpSize : 0);
  static constexpr int kHelperEndThread = kHelperBaseThread + kHelperThreads;
  static constexpr int kThreadCount =
      (kConsumerThreads > kHelperEndThread) ? kConsumerThreads
                                            : kHelperEndThread;

  static_assert(kConsumerGroupThreads > 0 &&
                    (kConsumerGroupThreads % kWarpSize) == 0,
                "ConsumerA group must contain whole warps");
  static_assert(kConsumerThreads >= kConsumerGroupThreads &&
                    (kConsumerThreads % kWarpSize) == 0,
                "Consumer region must contain whole warps");
  static_assert(kHelperBaseThread >= kConsumerThreads &&
                    (kHelperBaseThread % kWarpSize) == 0,
                "Helper base must start on a warp boundary");
  static_assert(kHelperThreads > 0 && (kHelperThreads % kWarpSize) == 0,
                "Helper region must contain whole warps");

  __device__ static int lane_id(int tid) { return tid & 31; }
  __device__ static int warp_id(int tid) { return tid >> 5; }

  __device__ static bool is_control_init_lane(int tid) {
    return tid < kControlInitThreads;
  }

  template <int SlotDepth>
  __device__ static bool is_barrier_init_lane(int tid) {
    return tid < SlotDepth;
  }

  template <int SlotDepth>
  __device__ static int init_slot(int tid) {
    return tid % SlotDepth;
  }

  __device__ static bool is_consumer_a(int tid) {
    return tid < kConsumerGroupThreads && !is_helper(tid);
  }

  __device__ static bool is_consumer_b(int tid) {
    return tid >= kConsumerGroupThreads && tid < kConsumerThreads &&
           !is_helper(tid);
  }

  __device__ static bool is_consumer(int tid) {
    return tid < kConsumerThreads && !is_helper(tid);
  }

  __device__ static bool is_helper(int tid) {
    int warp = warp_id(tid);
    if constexpr (NAIVERT_ROLE_HELPER_WARP >= 0) {
      return warp == kHelperBaseWarp;
    } else {
      return tid >= kHelperBaseThread && tid < kHelperEndThread;
    }
  }
};

}  // namespace naivert
