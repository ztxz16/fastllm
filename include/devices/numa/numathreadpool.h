//
// NUMA Thread Pool - Refactored with modular architecture
//

#ifndef FASTLLM_NUMATHREADPOOL_H
#define FASTLLM_NUMATHREADPOOL_H

#ifdef USE_NUMA

#include <functional>
#include <thread>
#include <vector>
#include "numatopology.h"

namespace fastllm {

// Forward declarations
class NumaScheduler;

/**
 * NUMA Thread Pool
 * Main interface for NUMA-aware parallel execution
 * Uses modular components: NumaTopology, NumaScheduler, NumaMemory
 */
class NumaThreadPool {
public:
    // Singleton access
    static NumaThreadPool& GetInstance();
    
    // Delete copy and assignment
    NumaThreadPool(const NumaThreadPool&) = delete;
    NumaThreadPool& operator=(const NumaThreadPool&) = delete;
    
    // Initialize with thread count (call from SetThreads when device is numa)
    void Initialize(int num_threads);
    
    // Check if initialized
    bool IsInitialized() const { return initialized_; }
    
    // Get number of threads
    int GetNumThreads() const { return max_threads_; }
    
    // Get thread-local NUMA node ID
    static int GetThreadNumaNode() { return thread_local_numa_node_; }
    
    // Get thread-local thread ID
    static int GetThreadLocalId() { return thread_local_thread_id_; }
    
    // Main work distribution (global load balancing)
    void DoWork(int nth,
               std::function<void(int)> init_func,
               std::function<void(int)> compute_func,
               std::function<void(int)> finalize_func);
    
    // NUMA-aware work distribution
    void DoNumaWork(int k, int nth,
                   std::function<void(int)> init_func,
                   std::function<void(int)> compute_func,
                   std::function<void(int)> finalize_func);
    
    // Calculate NUMA block distribution
    std::vector<NumaBlock> CalculateNumaBlocks(int total_blocks);

private:
    NumaThreadPool();
    ~NumaThreadPool();
    
    // Worker thread main loop
    void WorkerThread(int thread_id);
    
    // Thread-local storage
    static thread_local int thread_local_numa_node_;
    static thread_local int thread_local_thread_id_;
    
    // Configuration
    int max_threads_;
    bool initialized_;
    
    // Thread information
    std::vector<ThreadInfo> threads_info_;
    std::vector<std::thread> workers_;
    
    // Scheduler (manages work distribution)
    NumaScheduler* scheduler_;
};

}  // namespace fastllm

#endif  // USE_NUMA

#endif  // FASTLLM_NUMATHREADPOOL_H
