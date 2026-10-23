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
    void* allocate_pinned_numa(size_t size, int node);
    void free_pinned_numa(void* ptr, size_t size);
    // 把已分配的主机缓冲就地绑定到指定 NUMA 节点（页迁移，不复制数据）。
    // 单节点机器上恒为 true；多节点上用 mbind(MPOL_MF_MOVE) 迁移页，
    // 失败返回 false，由调用方回退到拷贝路径。
    bool BindExistingBufferToNuma(void* ptr, size_t size, int node);
}