#include "numa.h"
#include "numaif.h"
#include "fastllm.h"

namespace fastllm {
    struct MachineNumaInfo {
        int numaCnt = 1;
        int threads = -1;
        std::vector <std::vector <int> > cpuIds; // { numaId -> {cpuId} }

        MachineNumaInfo ();
    };

    struct NumaConfig {
        int numaCnt = 1;
        int threads = -1;
        std::vector <std::vector <std::pair <int, int> > > numaToCpuDict; // { numaId -> {(threadId, cpuId)} }
        std::vector <int> threadIdToNumaDict; // {threadId -> numaId}

        NumaConfig () {}
        
        NumaConfig (int threads, AliveThreadPool *pool, MachineNumaInfo *machineNumaInfo);
    };

    void bind_to_cpu(int cpu_id);
    void bind_to_numa_node(int node_id);
    void set_numa_mempolicy(int node_id);
    void* allocate_aligned_numa(size_t size, int node);
    void free_aligned_numa(void* aligned_ptr, size_t size);
    void* allocate_aligned(size_t size);
    void free_aligned(void* aligned_ptr, size_t size);
}