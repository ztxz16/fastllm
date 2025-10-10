//
// NUMA Thread Pool - Refactored with modular components
// Uses NumaTopology, NumaScheduler, and NumaMemory
//

#ifdef USE_NUMA

#include "devices/numa/numathreadpool.h"
#include "devices/numa/numatopology.h"
#include "devices/numa/numascheduler.h"
#include "devices/numa/numamemory.h"
#include <chrono>
#include <iostream>
#include <algorithm>

namespace fastllm {

// Thread-local storage
thread_local int NumaThreadPool::thread_local_numa_node_ = -1;
thread_local int NumaThreadPool::thread_local_thread_id_ = -1;

// Singleton
NumaThreadPool& NumaThreadPool::GetInstance() {
    static NumaThreadPool instance;
    return instance;
}

NumaThreadPool::NumaThreadPool() 
    : max_threads_(0), scheduler_(nullptr), initialized_(false) {
    // Initialization delayed until needed
}

NumaThreadPool::~NumaThreadPool() {
    if (!initialized_) return;
    
    // Signal all threads to exit
    for (int i = 0; i < max_threads_; i++) {
        scheduler_->GetThreadState(i)->status.store(
            NumaThreadStatus::EXIT, std::memory_order_release);
    }
    
    // Wait for threads
    for (int i = 0; i < max_threads_; i++) {
        if (workers_[i].joinable()) {
            workers_[i].join();
        }
    }
    
    // Cleanup
    if (scheduler_) {
        delete scheduler_;
    }
}

void NumaThreadPool::Initialize(int num_threads) {
    if (initialized_) {
        std::cout << "[NumaThreadPool] Already initialized" << std::endl;
        return;
    }
    
    std::cout << "\n[NumaThreadPool] Initializing NUMA thread pool..." << std::endl;
    
    // Get topology
    NumaTopology& topo = NumaTopology::GetInstance();
    if (!topo.IsAvailable()) {
        std::cerr << "[NumaThreadPool] NUMA not available" << std::endl;
        return;
    }
    
    int numa_nodes = topo.GetNumNodes();
    int num_cpus = topo.GetNumCpus();
    
    // Determine thread count
    if (num_threads <= 0) {
        num_threads = num_cpus / 2;  // Default: half of CPUs
    }
    
    max_threads_ = std::max(numa_nodes, num_threads);
    max_threads_ = std::min(max_threads_, num_cpus - 2);
    max_threads_ = std::max(1, max_threads_);
    
    std::cout << "[NumaThreadPool] Creating " << max_threads_ << " threads" << std::endl;
    
    // Distribute threads across NUMA nodes
    threads_info_.resize(max_threads_);
    int base = max_threads_ / numa_nodes;
    int remain = max_threads_ % numa_nodes;
    int tid = 0;
    
    for (int nid = 0; nid < numa_nodes; ++nid) {
        int n = base + (nid < remain ? 1 : 0);
        int n_found = 0;
        
        for (int cid = 0; cid < num_cpus && n_found < n; ++cid) {
            const auto& cpu_info = topo.GetCpuById(cid);
            if (cpu_info.node_id == nid) {
                threads_info_[tid].thread_id = tid;
                threads_info_[tid].cpu_id = cid;
                threads_info_[tid].core_id = cpu_info.core_id;
                threads_info_[tid].node_id = cpu_info.node_id;
                threads_info_[tid].package_id = cpu_info.package_id;
                threads_info_[tid].logic_idx = cpu_info.logic_idx;
                
                std::cout << "[NumaThreadPool] Thread " << tid 
                         << " -> CPU " << cid 
                         << " (core " << cpu_info.core_id 
                         << "[" << cpu_info.logic_idx << "]"
                         << ", node " << cpu_info.node_id << ")" << std::endl;
                
                tid++;
                n_found++;
            }
        }
    }
    
    // Create scheduler
    scheduler_ = new NumaScheduler(max_threads_, threads_info_);
    
    // Start worker threads
    workers_.resize(max_threads_);
    for (int i = 0; i < max_threads_; i++) {
        workers_[i] = std::thread(&NumaThreadPool::WorkerThread, this, i);
    }
    
    initialized_ = true;
    std::cout << "[NumaThreadPool] Initialization complete" << std::endl;
}

void NumaThreadPool::WorkerThread(int thread_id) {
    // Set thread-local storage
    thread_local_thread_id_ = thread_id;
    thread_local_numa_node_ = threads_info_[thread_id].node_id;
    int cpu_id = threads_info_[thread_id].cpu_id;
    
    // Bind thread to CPU and set NUMA policy
    CpuAffinity::BindToCpu(cpu_id);
    NumaMemory::SetMemPolicy(thread_local_numa_node_);
    
#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
    // Initialize AMX tile configuration
    AmxGemm::InitTileConfig();
#endif
    
    auto last_run_time = std::chrono::steady_clock::now();
    
    while (true) {
        NumaThreadStatus status = scheduler_->GetThreadState(thread_id)->status.load(
            std::memory_order_acquire);
        
        if (status == NumaThreadStatus::WORKING) {
            scheduler_->ProcessTasks(thread_id);
            last_run_time = std::chrono::steady_clock::now();
        } else if (status == NumaThreadStatus::WAITING) {
            // No power saving - just check duration for potential sleep
            auto now = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_run_time).count();
            
            if (duration > 100) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        } else if (status == NumaThreadStatus::EXIT) {
            return;
        }
    }
}

void NumaThreadPool::DoWork(int nth,
                           std::function<void(int)> init_func,
                           std::function<void(int)> compute_func,
                           std::function<void(int)> finalize_func) {
    if (!initialized_) {
        std::cerr << "[NumaThreadPool] Not initialized!" << std::endl;
        return;
    }
    
    scheduler_->ScheduleWork(nth, init_func, compute_func, finalize_func);
    scheduler_->WaitForCompletion();
}

void NumaThreadPool::DoNumaWork(int k, int nth,
                               std::function<void(int)> init_func,
                               std::function<void(int)> compute_func,
                               std::function<void(int)> finalize_func) {
    if (!initialized_) {
        std::cerr << "[NumaThreadPool] Not initialized!" << std::endl;
        return;
    }
    
    scheduler_->ScheduleNumaWork(k, nth, init_func, compute_func, finalize_func);
    scheduler_->WaitForCompletion();
}

std::vector<NumaBlock> NumaThreadPool::CalculateNumaBlocks(int total_blocks) {
    NumaTopology& topo = NumaTopology::GetInstance();
    return topo.CalculateNumaBlocks(total_blocks);
}

}  // namespace fastllm

#endif  // USE_NUMA
