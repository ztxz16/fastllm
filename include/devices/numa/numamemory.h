//
// NUMA Memory Management
// Modular component for NUMA-aware memory allocation
//

#ifndef FASTLLM_NUMAMEMORY_H
#define FASTLLM_NUMAMEMORY_H

#ifdef USE_NUMA

#include <cstddef>
#include <numa.h>
#include <numaif.h>

namespace fastllm {

/**
 * NUMA Memory Allocator
 * Provides NUMA-aware memory allocation functions
 */
class NumaMemory {
public:
    // Allocate aligned memory on a specific NUMA node
    static void* AllocOnNode(size_t size, int node_id, size_t alignment = 64);
    
    // Free NUMA-allocated memory
    static void Free(void* ptr, size_t size);
    
    // Allocate general aligned memory (not NUMA-specific)
    static void* AllocAligned(size_t size, size_t alignment = 64);
    
    // Free general aligned memory
    static void FreeAligned(void* ptr);
    
    // Set NUMA memory policy for current thread
    static void SetMemPolicy(int node_id);
    
    // Reset memory policy to default
    static void ResetMemPolicy();
};

/**
 * CPU Affinity Manager
 * Manages thread-to-CPU bindings
 */
class CpuAffinity {
public:
    // Bind current thread to a specific CPU
    static void BindToCpu(int cpu_id);
    
    // Bind current thread to any CPU on a NUMA node
    static void BindToNode(int node_id);
    
    // Unbind current thread (reset affinity)
    static void Unbind();
};

}  // namespace fastllm

#endif  // USE_NUMA

#endif  // FASTLLM_NUMAMEMORY_H

