//
// NUMA Work-Stealing Scheduler
// Modular component for NUMA-aware task scheduling
//

#ifndef FASTLLM_NUMASCHEDULER_H
#define FASTLLM_NUMASCHEDULER_H

#ifdef USE_NUMA

#include <atomic>
#include <functional>
#include <vector>
#include "numatopology.h"

namespace fastllm {

// Thread status for work-stealing scheduler
enum class NumaThreadStatus {
    WORKING,
    WAITING,
    EXIT
};

// Cache-line aligned to avoid false sharing
constexpr size_t NUMA_CACHE_LINE_SIZE = 64;

// Thread state with cache-line alignment
struct alignas(NUMA_CACHE_LINE_SIZE) NumaThreadState {
    std::atomic<NumaThreadStatus> status;
    char padding1[NUMA_CACHE_LINE_SIZE - sizeof(std::atomic<NumaThreadStatus>)];
    std::atomic<int> curr;
    char padding2[NUMA_CACHE_LINE_SIZE - sizeof(std::atomic<int>)];
    int end;
    int node_id;  // NUMA node this thread belongs to
};

/**
 * NUMA Scheduler
 * Manages work-stealing task scheduling across NUMA nodes
 */
class NumaScheduler {
public:
    NumaScheduler(int num_threads, const std::vector<ThreadInfo>& threads_info);
    ~NumaScheduler() = default;
    
    // Schedule work across all threads (global load balancing)
    void ScheduleWork(int num_tasks,
                     std::function<void(int)> init_func,
                     std::function<void(int)> compute_func,
                     std::function<void(int)> finalize_func);
    
    // NUMA-aware work scheduling (per-node distribution)
    // k: replication factor, nth: number of task groups
    void ScheduleNumaWork(int k, int nth,
                         std::function<void(int)> init_func,
                         std::function<void(int)> compute_func,
                         std::function<void(int)> finalize_func);
    
    // Get thread state
    NumaThreadState* GetThreadState(int thread_id) { return thread_states_[thread_id]; }
    
    // Set work functions (called before scheduling)
    void SetWorkFunctions(std::function<void(int)> init,
                         std::function<void(int)> compute,
                         std::function<void(int)> finalize) {
        init_func_ = init;
        compute_func_ = compute;
        finalize_func_ = finalize;
    }
    
    // Get work functions
    std::function<void(int)> GetInitFunc() const { return init_func_; }
    std::function<void(int)> GetComputeFunc() const { return compute_func_; }
    std::function<void(int)> GetFinalizeFunc() const { return finalize_func_; }
    
    // Process tasks for a specific thread (with work stealing)
    void ProcessTasks(int thread_id);
    
    // Wait for all threads to complete
    void WaitForCompletion();

private:
    int num_threads_;
    int num_nodes_;
    
    // Thread information
    const std::vector<ThreadInfo>& threads_info_;
    std::vector<std::vector<int>> node_threads_;  // [node_id] -> [thread_ids]
    
    // Thread states (cache-aligned)
    std::vector<NumaThreadState*> thread_states_;
    
    // Work functions
    std::function<void(int)> init_func_;
    std::function<void(int)> compute_func_;
    std::function<void(int)> finalize_func_;
};

}  // namespace fastllm

#endif  // USE_NUMA

#endif  // FASTLLM_NUMASCHEDULER_H

