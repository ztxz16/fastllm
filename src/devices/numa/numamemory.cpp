//
// NUMA Memory Management Implementation
//

#ifdef USE_NUMA

#include "devices/numa/numamemory.h"
#include <iostream>
#include <cstring>
#include <cerrno>
#include <sched.h>
#include <unistd.h>

namespace fastllm {

// NumaMemory implementation
void* NumaMemory::AllocOnNode(size_t size, int node_id, size_t alignment) {
    size_t total_size = size + alignment - 1;
    void* raw_ptr = numa_alloc_onnode(total_size, node_id);
    if (!raw_ptr) {
        std::cerr << "[NumaMemory] Failed to allocate " << size 
                 << " bytes on node " << node_id << std::endl;
        return nullptr;
    }
    
    uintptr_t addr = reinterpret_cast<uintptr_t>(raw_ptr);
    uintptr_t aligned_addr = (addr + alignment - 1) & ~(alignment - 1);
    return reinterpret_cast<void*>(aligned_addr);
}

void NumaMemory::Free(void* ptr, size_t size) {
    if (!ptr) return;
    numa_free(ptr, size);
}

void* NumaMemory::AllocAligned(size_t size, size_t alignment) {
    size_t total_size = size + alignment + sizeof(void*);
    void* raw_ptr = malloc(total_size);
    if (!raw_ptr) return nullptr;
    
    uintptr_t aligned_addr = (reinterpret_cast<uintptr_t>(raw_ptr) + 
                             sizeof(void*) + alignment - 1) & ~(alignment - 1);
    void* aligned_ptr = reinterpret_cast<void*>(aligned_addr);
    
    // Store original pointer before aligned pointer
    void** prev_ptr = reinterpret_cast<void**>(aligned_ptr) - 1;
    *prev_ptr = raw_ptr;
    
    return aligned_ptr;
}

void NumaMemory::FreeAligned(void* aligned_ptr) {
    if (!aligned_ptr) return;
    void** prev_ptr = reinterpret_cast<void**>(aligned_ptr) - 1;
    void* raw_ptr = *prev_ptr;
    free(raw_ptr);
}

void NumaMemory::SetMemPolicy(int node_id) {
    struct bitmask* mask = numa_allocate_nodemask();
    numa_bitmask_setbit(mask, node_id);
    
    int policy = MPOL_BIND;
    if (set_mempolicy(policy, mask->maskp, mask->size) == -1) {
        std::cerr << "[NumaMemory] Failed to set memory policy for node " << node_id 
                 << ": " << strerror(errno) << std::endl;
    }
    numa_free_nodemask(mask);
}

void NumaMemory::ResetMemPolicy() {
    set_mempolicy(MPOL_DEFAULT, nullptr, 0);
}

// CpuAffinity implementation
void CpuAffinity::BindToCpu(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) == -1) {
        std::cerr << "[CpuAffinity] Failed to bind to CPU " << cpu_id 
                 << ": " << strerror(errno) << std::endl;
    }
}

void CpuAffinity::BindToNode(int node_id) {
    struct bitmask* node_cpus = numa_allocate_cpumask();
    if (numa_node_to_cpus(node_id, node_cpus) != 0) {
        std::cerr << "[CpuAffinity] Failed to get CPUs for node " << node_id << std::endl;
        numa_free_cpumask(node_cpus);
        return;
    }
    
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    
    for (unsigned int i = 0; i < node_cpus->size; ++i) {
        if (numa_bitmask_isbitset(node_cpus, i)) {
            CPU_SET(i, &cpuset);
        }
    }
    numa_free_cpumask(node_cpus);
    
    if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
        std::cerr << "[CpuAffinity] Failed to bind to node " << node_id 
                 << ": " << strerror(errno) << std::endl;
    }
}

void CpuAffinity::Unbind() {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int i = 0; i < CPU_SETSIZE; i++) {
        CPU_SET(i, &cpuset);
    }
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
}

}  // namespace fastllm

#endif  // USE_NUMA

