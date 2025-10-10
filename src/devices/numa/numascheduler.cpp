//
// NUMA Work-Stealing Scheduler Implementation
//

#ifdef USE_NUMA

#include "devices/numa/numascheduler.h"
#include "devices/numa/numatopology.h"
#include <algorithm>
#include <iostream>

namespace fastllm {

NumaScheduler::NumaScheduler(int num_threads, const std::vector<ThreadInfo>& threads_info)
    : num_threads_(num_threads), threads_info_(threads_info) {
    
    // Get number of NUMA nodes
    NumaTopology& topo = NumaTopology::GetInstance();
    num_nodes_ = topo.GetNumNodes();
    
    // Organize threads by NUMA node
    node_threads_.resize(num_nodes_);
    for (const auto& tinfo : threads_info_) {
        node_threads_[tinfo.node_id].push_back(tinfo.thread_id);
    }
    
    // Initialize thread states with cache-line alignment
    thread_states_.resize(num_threads_);
    for (int i = 0; i < num_threads_; i++) {
        NumaThreadState* state = new (std::align_val_t{64}) NumaThreadState();
        state->status.store(NumaThreadStatus::WAITING, std::memory_order_relaxed);
        state->curr.store(0, std::memory_order_relaxed);
        state->end = 0;
        state->node_id = threads_info_[i].node_id;
        thread_states_[i] = state;
    }
    
    std::cout << "[NumaScheduler] Initialized with " << num_threads_ << " threads across "
              << num_nodes_ << " NUMA nodes" << std::endl;
}

void NumaScheduler::ScheduleWork(int num_tasks,
                                 std::function<void(int)> init_func,
                                 std::function<void(int)> compute_func,
                                 std::function<void(int)> finalize_func) {
    init_func_ = init_func;
    compute_func_ = compute_func;
    finalize_func_ = finalize_func;
    
    // Distribute tasks evenly across all threads
    int base = num_tasks / num_threads_;
    int remain = num_tasks % num_threads_;
    
    int begin = 0;
    for (int i = 0; i < num_threads_; i++) {
        int end = begin + base + (i < remain ? 1 : 0);
        
        if (begin >= end) {
            thread_states_[i]->status.store(NumaThreadStatus::WAITING, std::memory_order_release);
            thread_states_[i]->end = -1;
            thread_states_[i]->curr.store(0, std::memory_order_relaxed);
        } else {
            thread_states_[i]->curr.store(begin, std::memory_order_relaxed);
            thread_states_[i]->end = end;
            thread_states_[i]->status.store(NumaThreadStatus::WORKING, std::memory_order_release);
        }
        
        begin = end;
    }
}

void NumaScheduler::ScheduleNumaWork(int k, int nth,
                                     std::function<void(int)> init_func,
                                     std::function<void(int)> compute_func,
                                     std::function<void(int)> finalize_func) {
    init_func_ = init_func;
    compute_func_ = compute_func;
    finalize_func_ = finalize_func;
    
    // First level: distribute across NUMA nodes
    int base = nth / num_nodes_;
    int remain = nth % num_nodes_;
    
    int begin = 0;
    for (int nid = 0; nid < num_nodes_; nid++) {
        int n_tasks = (base + (nid < remain ? 1 : 0)) * k;
        int n_threads = node_threads_[nid].size();
        
        if (n_threads == 0 || n_tasks == 0) {
            continue;
        }
        
        // Second level: distribute within node
        int t_base = n_tasks / n_threads;
        int t_remain = n_tasks % n_threads;
        
        for (int j = 0; j < n_threads; j++) {
            int tid = node_threads_[nid][j];
            int end = begin + t_base + (j < t_remain ? 1 : 0);
            
            if (begin >= end) {
                thread_states_[tid]->status.store(NumaThreadStatus::WAITING, std::memory_order_release);
                thread_states_[tid]->end = -1;
                thread_states_[tid]->curr.store(0, std::memory_order_relaxed);
            } else {
                thread_states_[tid]->curr.store(begin, std::memory_order_relaxed);
                thread_states_[tid]->end = end;
                thread_states_[tid]->status.store(NumaThreadStatus::WORKING, std::memory_order_release);
            }
            
            begin = end;
        }
    }
}

void NumaScheduler::ProcessTasks(int thread_id) {
    // Call init function if provided
    if (init_func_ != nullptr) {
        init_func_(thread_id);
    }
    
    // Process own tasks
    while (true) {
        if (thread_states_[thread_id]->status.load(std::memory_order_acquire) != 
            NumaThreadStatus::WORKING) {
            break;
        }
        
        int task_id = thread_states_[thread_id]->curr.fetch_add(1, std::memory_order_acq_rel);
        if (task_id >= thread_states_[thread_id]->end) {
            break;
        }
        compute_func_(task_id);
    }
    
    // Work stealing: help other threads on the same NUMA node
    int my_node = threads_info_[thread_id].node_id;
    int my_cpu = threads_info_[thread_id].cpu_id;
    
    for (int i = 0; i < num_threads_; i++) {
        const auto& other_thread = threads_info_[i];
        
        // Only steal from threads on same NUMA node but different CPU
        if (other_thread.node_id == my_node && other_thread.cpu_id != my_cpu) {
            if (thread_states_[i]->status.load(std::memory_order_acquire) != 
                NumaThreadStatus::WORKING) {
                continue;
            }
            
            while (true) {
                int task_id = thread_states_[i]->curr.fetch_add(1, std::memory_order_acq_rel);
                if (task_id >= thread_states_[i]->end) {
                    break;
                }
                compute_func_(task_id);
            }
        }
    }
    
    // Call finalize function if provided
    if (finalize_func_ != nullptr) {
        finalize_func_(thread_id);
    }
    
    thread_states_[thread_id]->status.store(NumaThreadStatus::WAITING, std::memory_order_release);
}

void NumaScheduler::WaitForCompletion() {
    for (int i = 0; i < num_threads_; i++) {
        while (thread_states_[i]->status.load(std::memory_order_acquire) == 
               NumaThreadStatus::WORKING) {
            // Spin wait
        }
    }
}

}  // namespace fastllm

#endif  // USE_NUMA

