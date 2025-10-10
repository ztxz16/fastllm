//
// NUMA Topology Detection and Management Implementation
//

#ifdef USE_NUMA

#include "devices/numa/numatopology.h"
#include <algorithm>
#include <fstream>
#include <map>
#include <iostream>
#include <cstring>
#include <cerrno>

namespace fastllm {

// Global state
static std::atomic<int> global_numa_counter(0);

// Singleton implementation
NumaTopology& NumaTopology::GetInstance() {
    static NumaTopology instance;
    return instance;
}

NumaTopology::NumaTopology() 
    : numa_available_(false), numa_nodes_(0), num_cpus_(0), 
      num_cores_(0), cpus_per_node_(0), hyper_threading_open_(false) {
    InitTopology();
}

void NumaTopology::InitTopology() {
    // Check NUMA availability
    if (numa_available() < 0) {
        std::cerr << "[NumaTopology] NUMA not available on this system" << std::endl;
        return;
    }
    
    numa_available_ = true;
    numa_nodes_ = numa_num_configured_nodes();
    num_cpus_ = numa_num_configured_cpus();
    
    std::cout << "[NumaTopology] NUMA available with " << numa_nodes_ 
              << " nodes, " << num_cpus_ << " CPUs" << std::endl;
    
    cpus_info_.clear();
    cpus_info_.reserve(num_cpus_);
    
    // Map to track unique physical cores
    std::map<std::pair<int, int>, int> unique_cores;
    int next_phys_id = 0;
    std::map<std::pair<int, int>, int> core_counters;
    
    // First pass: identify unique physical cores
    for (int i = 0; i < num_cpus_; ++i) {
        std::string cpu_dir = "/sys/devices/system/cpu/cpu" + std::to_string(i);
        int cid = ReadTopologyValue(cpu_dir, "topology/core_id");
        int pid = ReadTopologyValue(cpu_dir, "topology/physical_package_id");
        
        std::pair<int, int> core_key = std::make_pair(pid, cid);
        if (unique_cores.find(core_key) == unique_cores.end()) {
            unique_cores[core_key] = next_phys_id++;
        }
        core_counters[core_key] = 0;
    }
    
    num_cores_ = next_phys_id;
    
    // Second pass: populate CPU info
    for (int i = 0; i < num_cpus_; ++i) {
        std::string cpu_dir = "/sys/devices/system/cpu/cpu" + std::to_string(i);
        
        int raw_cid = ReadTopologyValue(cpu_dir, "topology/core_id");
        int pid = ReadTopologyValue(cpu_dir, "topology/physical_package_id");
        int nid = numa_node_of_cpu(i);
        
        std::pair<int, int> core_key = std::make_pair(pid, raw_cid);
        
        CpuInfo info;
        info.cpu_id = i;
        info.core_id = unique_cores[core_key];
        info.node_id = nid;
        info.package_id = pid;
        info.logic_idx = core_counters[core_key];
        
        core_counters[core_key]++;
        cpus_info_.push_back(info);
    }
    
    cpus_per_node_ = num_cpus_ / numa_nodes_;
    hyper_threading_open_ = num_cpus_ > num_cores_;
    
    std::cout << "[NumaTopology] Detected " << num_cores_ << " physical cores, "
              << (hyper_threading_open_ ? "HT enabled" : "HT disabled") << std::endl;
}

int NumaTopology::ReadTopologyValue(const std::string& cpu_path, const char* file) {
    std::ifstream ifs(cpu_path + "/" + file);
    int value;
    return ifs >> value ? value : -1;
}

std::vector<NumaBlock> NumaTopology::CalculateNumaBlocks(int total_blocks) const {
    std::vector<NumaBlock> blocks;
    blocks.resize(numa_nodes_);
    
    int base = total_blocks / numa_nodes_;
    int remain = total_blocks % numa_nodes_;
    int current_block = 0;
    
    for (int nid = 0; nid < numa_nodes_; nid++) {
        int n_blocks = base + (nid < remain ? 1 : 0);
        blocks[nid].node_id = nid;
        blocks[nid].start_block = current_block;
        blocks[nid].num_blocks = n_blocks;
        current_block += n_blocks;
    }
    
    return blocks;
}

void NumaTopology::PrintTopology() const {
    std::cout << "\n[NumaTopology] System Topology:" << std::endl;
    std::cout << "  NUMA Nodes: " << numa_nodes_ << std::endl;
    std::cout << "  Total CPUs: " << num_cpus_ << std::endl;
    std::cout << "  Physical Cores: " << num_cores_ << std::endl;
    std::cout << "  Hyper-Threading: " << (hyper_threading_open_ ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  CPUs per Node: " << cpus_per_node_ << std::endl;
}

// Helper functions
int GetNextNumaNode() {
    NumaTopology& topo = NumaTopology::GetInstance();
    if (!topo.IsAvailable()) {
        return 0;
    }
    return (global_numa_counter.fetch_add(1, std::memory_order_relaxed)) % topo.GetNumNodes();
}

bool IsNumaAvailable() {
    return NumaTopology::GetInstance().IsAvailable();
}

}  // namespace fastllm

#endif  // USE_NUMA

