//
// NUMA Topology Detection and Management
// Modular component for CPU topology information
//

#ifndef FASTLLM_NUMATOPOLOGY_H
#define FASTLLM_NUMATOPOLOGY_H

#ifdef USE_NUMA

#include <vector>
#include <string>
#include <numa.h>

namespace fastllm {

// CPU information structure
struct CpuInfo {
    int cpu_id;          // Logical CPU ID
    int core_id;         // Physical core ID
    int node_id;         // NUMA node ID
    int package_id;      // Physical package/socket ID
    int logic_idx;       // Logical index within the core (for hyper-threading)
};

// Thread information structure
struct ThreadInfo {
    int thread_id;       // Thread pool ID
    int cpu_id;          // Bound to this CPU
    int core_id;         // Physical core
    int node_id;         // NUMA node
    int package_id;      // Physical package
    int logic_idx;       // HT index
};

// NUMA block descriptor for weight distribution
struct NumaBlock {
    int node_id;         // NUMA node ID
    int start_block;     // Starting block index
    int num_blocks;      // Number of blocks
};

/**
 * NUMA Topology Manager
 * Detects and manages CPU/NUMA topology information
 */
class NumaTopology {
public:
    // Singleton access
    static NumaTopology& GetInstance();
    
    // Delete copy and assignment
    NumaTopology(const NumaTopology&) = delete;
    NumaTopology& operator=(const NumaTopology&) = delete;
    
    // Check if NUMA is available
    bool IsAvailable() const { return numa_available_; }
    
    // Get number of NUMA nodes
    int GetNumNodes() const { return numa_nodes_; }
    
    // Get number of CPUs
    int GetNumCpus() const { return num_cpus_; }
    
    // Get number of physical cores
    int GetNumCores() const { return num_cores_; }
    
    // Check if hyper-threading is enabled
    bool HasHyperThreading() const { return hyper_threading_open_; }
    
    // Get CPUs per NUMA node
    int GetCpusPerNode() const { return cpus_per_node_; }
    
    // Get CPU information
    const std::vector<CpuInfo>& GetCpuInfo() const { return cpus_info_; }
    
    // Get CPU info by ID
    const CpuInfo& GetCpuById(int cpu_id) const { return cpus_info_[cpu_id]; }
    
    // Get NUMA node of a CPU
    int GetNumaNodeOfCpu(int cpu_id) const { return cpus_info_[cpu_id].node_id; }
    
    // Calculate NUMA block distribution
    std::vector<NumaBlock> CalculateNumaBlocks(int total_blocks) const;
    
    // Print topology information
    void PrintTopology() const;

private:
    NumaTopology();
    ~NumaTopology() = default;
    
    // Initialize topology detection
    void InitTopology();
    
    // Read topology from sysfs
    int ReadTopologyValue(const std::string& cpu_path, const char* file);
    
    // NUMA state
    bool numa_available_;
    int numa_nodes_;
    int num_cpus_;
    int num_cores_;
    int cpus_per_node_;
    bool hyper_threading_open_;
    
    // CPU topology
    std::vector<CpuInfo> cpus_info_;
};

// Helper functions

// Get next NUMA node in round-robin fashion
int GetNextNumaNode();

// Check if NUMA is available
bool IsNumaAvailable();

}  // namespace fastllm

#endif  // USE_NUMA

#endif  // FASTLLM_NUMATOPOLOGY_H

