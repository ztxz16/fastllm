#include <linux/mempolicy.h>
#include "numas.h"
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <mutex>
#include <sys/mman.h>
#include <unordered_map>
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#endif

namespace fastllm {
    static bool ReadSysfsInt(const std::string &path, int &value) {
        std::ifstream input(path);
        input >> value;
        return !input.fail();
    }

    static std::vector<int> SpreadCpusAcrossLastLevelCaches(
            const std::vector<int> &cpuIds) {
        struct CoreGroup {
            std::vector<int> cpuIds;
        };
        struct CacheGroup {
            std::vector<CoreGroup> cores;
            std::map<int, int> coreToIndex;
        };

        std::vector<CacheGroup> caches;
        std::map<int, int> cacheToIndex;
        for (int cpuId : cpuIds) {
            const std::string cpuPath =
                "/sys/devices/system/cpu/cpu" + std::to_string(cpuId);
            int cacheId = -1;
            int coreId = -1;
            if (!ReadSysfsInt(cpuPath + "/cache/index3/id", cacheId) ||
                !ReadSysfsInt(cpuPath + "/topology/core_id", coreId)) {
                return cpuIds;
            }

            auto cacheIt = cacheToIndex.find(cacheId);
            if (cacheIt == cacheToIndex.end()) {
                int cacheIndex = (int)caches.size();
                cacheToIndex[cacheId] = cacheIndex;
                caches.emplace_back();
                cacheIt = cacheToIndex.find(cacheId);
            }
            CacheGroup &cache = caches[cacheIt->second];
            auto coreIt = cache.coreToIndex.find(coreId);
            if (coreIt == cache.coreToIndex.end()) {
                int coreIndex = (int)cache.cores.size();
                cache.coreToIndex[coreId] = coreIndex;
                cache.cores.emplace_back();
                coreIt = cache.coreToIndex.find(coreId);
            }
            cache.cores[coreIt->second].cpuIds.push_back(cpuId);
        }

        if (caches.size() <= 1) {
            return cpuIds;
        }

        size_t maxCores = 0;
        size_t maxThreadsPerCore = 0;
        for (const CacheGroup &cache : caches) {
            maxCores = std::max(maxCores, cache.cores.size());
            for (const CoreGroup &core : cache.cores) {
                maxThreadsPerCore =
                    std::max(maxThreadsPerCore, core.cpuIds.size());
            }
        }

        std::vector<int> spreadCpuIds;
        spreadCpuIds.reserve(cpuIds.size());
        // Select one physical core from every LLC before taking the next core
        // from an already-used LLC.  SMT siblings are considered only after
        // every physical core has been selected.
        for (size_t sibling = 0; sibling < maxThreadsPerCore; sibling++) {
            for (size_t core = 0; core < maxCores; core++) {
                for (const CacheGroup &cache : caches) {
                    if (core < cache.cores.size() &&
                        sibling < cache.cores[core].cpuIds.size()) {
                        spreadCpuIds.push_back(
                            cache.cores[core].cpuIds[sibling]);
                    }
                }
            }
        }
        return spreadCpuIds.size() == cpuIds.size() ? spreadCpuIds : cpuIds;
    }

    namespace {
        // Snapshot before the shared pool pins its workers. Logical shard ids
        // must not be confused with physical ids (e.g. numactl --membind=1).
        struct AllowedNumaTopology {
            std::vector<int> nodes;
            std::vector<std::vector<int>> cpus;
            AllowedNumaTopology() {
                cpu_set_t affinity;
                if (sched_getaffinity(0, sizeof(affinity), &affinity) != 0)
                    throw std::runtime_error("Cannot read inherited CPU affinity");
                std::vector<int> allowedCpus;
                for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
                    if (CPU_ISSET(cpu, &affinity)) allowedCpus.push_back(cpu);
                if (allowedCpus.empty())
                    throw std::runtime_error("Empty inherited CPU affinity");
                if (numa_available() < 0) {
                    nodes.push_back(0);
                    cpus.push_back(allowedCpus);
                    return;
                }
                struct bitmask *allowed = numa_get_mems_allowed();
                struct bitmask *policyMask = numa_allocate_nodemask();
                int policy = MPOL_DEFAULT;
                const bool explicitBind = get_mempolicy(&policy, policyMask->maskp,
                    policyMask->size, nullptr, 0) == 0 &&
                    ((policy & ~(MPOL_F_STATIC_NODES | MPOL_F_RELATIVE_NODES)) == MPOL_BIND);
                int firstMemoryNode = -1;
                for (int node = 0; node <= numa_max_node(); ++node) {
                    if (!numa_bitmask_isbitset(numa_all_nodes_ptr, node) ||
                        (numa_bitmask_weight(allowed) && !numa_bitmask_isbitset(allowed, node)) ||
                        (explicitBind && !numa_bitmask_isbitset(policyMask, node))) continue;
                    if (firstMemoryNode < 0) firstMemoryNode = node;
                    std::vector<int> local;
                    for (int cpu : allowedCpus)
                        if (numa_node_of_cpu(cpu) == node) local.push_back(cpu);
                    if (!local.empty()) {
                        nodes.push_back(node);
                        cpus.push_back(std::move(local));
                    }
                }
                numa_free_nodemask(policyMask);
                numa_free_nodemask(allowed);
                // A deliberate remote CPU/memory binding still must not widen
                // the CPU mask or allocate outside the caller's memory policy.
                if (nodes.empty()) {
                    if (firstMemoryNode < 0)
                        throw std::runtime_error("No allowed NUMA memory node");
                    nodes.push_back(firstMemoryNode);
                    cpus.push_back(allowedCpus);
                }
                const char *value = std::getenv("FT_NUMAS");
                const int requested = value ? std::atoi(value) : 0;
                if (requested > 0 && requested < (int)nodes.size()) {
                    nodes.resize(requested);
                    cpus.resize(requested);
                }
            }
        };
        const AllowedNumaTopology &GetAllowedNumaTopology() {
            static const AllowedNumaTopology topology;
            return topology;
        }
    }

    int GetNumasPhysicalNode(int logicalNode) {
        return GetAllowedNumaTopology().nodes.at(logicalNode);
    }

    struct NumaDetector {
        // Do not probe by temporarily binding to node 0, or change allocation
        // provenance after a later worker's affinity/policy failure.
        const bool canUseNuma = numa_available() >= 0;
    };
    static NumaDetector &GetNumaDetector() {
        static NumaDetector detector;
        return detector;
    }
    bool HostMoeIsUnbound();
    struct PageAlignedAllocations {
        std::mutex mutex;
        std::unordered_map<void *, size_t> mappings;
    };
    static PageAlignedAllocations &GetPageAlignedAllocations() {
        static auto *allocations = new PageAlignedAllocations;
        return *allocations;
    }
    
    void bind_to_cpu(int cpu_id) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_id, &cpuset);
        
        if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) == -1) {
            perror("sched_setaffinity failed");
            exit(EXIT_FAILURE);
        }
    }

    void bind_to_numa_node(int node_id) { 
        struct bitmask *node_cpus = numa_allocate_cpumask();
        if (numa_node_to_cpus(node_id, node_cpus) != 0) {
            perror("Failed to get NUMA node CPUs");
            numa_free_cpumask(node_cpus);
            return;
        }

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
    
        cpu_set_t inherited;
        if (sched_getaffinity(0, sizeof(inherited), &inherited) != 0) {
            numa_free_cpumask(node_cpus);
            return;
        }
        for (unsigned int i = 0; i < node_cpus->size && i < CPU_SETSIZE; ++i) {
            if (numa_bitmask_isbitset(node_cpus, i) && CPU_ISSET(i, &inherited)) {
                CPU_SET(i, &cpuset);
            }
        }
        numa_free_cpumask(node_cpus);
    
        if (sched_setaffinity(0, sizeof(cpuset), &cpuset) != 0) {
            perror("sched_setaffinity failed"); 
        }
    }
    
    void set_numa_mempolicy(int node_id) {
        struct bitmask* mask = numa_allocate_nodemask();
        numa_bitmask_setbit(mask, node_id);
    
        int policy = MPOL_BIND;

        if (set_mempolicy(policy, mask->maskp, mask->size) == -1) {
            std::cerr << "set_mempolicy failed for node " << node_id 
                    << ": " << errno << " (" << strerror(errno) << ")\n";
            numa_free_nodemask(mask);
            return;
        }
        numa_free_nodemask(mask);
    }

    void* allocate_aligned(size_t size) {
        const size_t alignment = 64; 
        size_t total_size = size + alignment + sizeof(void*);
        void* raw_ptr = malloc(total_size);
        if (!raw_ptr) return nullptr;
        
        uintptr_t aligned_addr = (reinterpret_cast<uintptr_t>(raw_ptr) + sizeof(void*) + alignment - 1) & ~(alignment - 1);
        void* aligned_ptr = reinterpret_cast<void*>(aligned_addr);
    
        void** prev_ptr = reinterpret_cast<void**>(aligned_ptr) - 1;
        *prev_ptr = raw_ptr;

        return aligned_ptr;
    } 

    void free_aligned(void* aligned_ptr, size_t size) {
        if (!aligned_ptr) return; 
        void** prev_ptr = reinterpret_cast<void**>(aligned_ptr) - 1;
        void* raw_ptr = *prev_ptr;
        free(raw_ptr);
    }

    void* allocate_aligned_numa(size_t size, int node) {
        if (HostMoeIsUnbound()) {
            // Match NUMA's page-isolated allocations without mbind. malloc
            // arenas can share boundary pages and make cudaHostRegister of
            // otherwise-disjoint expert buffers overlap a pinned mapping.
            const size_t bytes = size + 63;
            void *ptr = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (ptr == MAP_FAILED) return nullptr;
            auto &allocations = GetPageAlignedAllocations();
            std::lock_guard<std::mutex> lock(allocations.mutex);
            allocations.mappings.emplace(ptr, bytes);
            return ptr;
        }
        if (!GetNumaDetector().canUseNuma) {
            return allocate_aligned(size);
        }
        
        size_t alignment = 64;
        size_t total_size = size + alignment - 1;
        void* raw_ptr = numa_alloc_onnode(total_size, GetNumasPhysicalNode(node));
        if (!raw_ptr) {
            std::cerr << "Failed to allocate " << size << " bytes on NUMA node " << node << std::endl;

            std::cerr << "Try increate max_map_count: " << std::endl;
            std::cerr << "Temporary: `sudo sysctl -w vm.max_map_count=262144`" << std::endl;
            std::cerr << "Permanent: `sudo echo \"vm.max_map_count=262144\" | sudo tee -a /etc/sysctl.conf && sudo sysctl -p`" << std::endl;
            exit(0);

            return nullptr;
        }
        
        // libnuma returns a page-aligned mmap. Retain its full length: callers
        // historically supplied either the payload length or payload + 63.
        auto &allocations = GetPageAlignedAllocations();
        std::lock_guard<std::mutex> lock(allocations.mutex);
        allocations.mappings.emplace(raw_ptr, total_size);
        return raw_ptr;
    }

    void free_aligned_numa(void* aligned_ptr, size_t size) {
        {
            auto &allocations = GetPageAlignedAllocations();
            std::lock_guard<std::mutex> lock(allocations.mutex);
            auto it = allocations.mappings.find(aligned_ptr);
            if (it != allocations.mappings.end()) {
                munmap(aligned_ptr, it->second);
                allocations.mappings.erase(it);
                return;
            }
        }
        if (!GetNumaDetector().canUseNuma) {
            free_aligned(aligned_ptr, size);
            return;
        }
        uintptr_t addr = reinterpret_cast<uintptr_t>(aligned_ptr);
        void* raw_ptr = reinterpret_cast<void*>(addr & ~(63));
        numa_free(raw_ptr, size);
    }

    void* allocate_pinned_numa(size_t size, int node) {
#ifdef USE_CUDA
        void *ptr = allocate_aligned_numa(size, node);
        if (ptr) {
            FastllmCudaHostRegister(ptr, size);
        }
        return ptr;
#else
        return allocate_aligned_numa(size, node);
#endif
    }

    void free_pinned_numa(void *ptr, size_t size) {
#ifdef USE_CUDA
        FastllmCudaHostUnregister(ptr);
        free_aligned_numa(ptr, size);
#else
        free_aligned_numa(ptr, size);
#endif
    }

    struct BindCPUOp : MultiThreadBaseOp {
        int cpuId, numaId;

        BindCPUOp (int cpuId, int numaId) : cpuId(cpuId), numaId(numaId) {}

        void Run() {
            try {
                // 尝试绑定到NUMA节点
                try {
                    set_numa_mempolicy(numaId);
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Failed to bind to NUMA node " << numaId 
                            << ": " << e.what() << std::endl;
                    std::cerr << "Continuing without NUMA binding (may affect performance)" << std::endl;
                }
                
                // 尝试设置CPU亲和性
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(this->cpuId, &cpuset);
                if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) == -1) {
                    // 检查是否是权限问题
                    if (errno == EPERM || errno == EACCES) {
                        std::cerr << "Warning: Failed to set CPU affinity to CPU " << this->cpuId 
                                << ": " << strerror(errno) << std::endl;
                        std::cerr << "Running without CPU pinning (requires privileged mode or CAP_SYS_NICE)" 
                                << std::endl;
                        std::cerr << "Consider running with --privileged or --cap-add=SYS_NICE" << std::endl;
                    } else {
                        // 其他错误可能更严重，抛出异常
                        throw std::runtime_error(std::string("sched_setaffinity failed: ") + strerror(errno));
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Error in Run(): " << e.what() << std::endl;
                // 根据需求决定是否继续执行或退出
                // 如果想要继续执行：不做任何事
                // 如果想要退出：throw;
            }
        }
    };

    MachineNumaInfo::MachineNumaInfo() {
        const auto &topology = GetAllowedNumaTopology();
        nodeIds = topology.nodes;
        cpuIds = topology.cpus;
        numaCnt = (int)nodeIds.size();
        int defaultThreads = 0;
        for (const auto &cpus : cpuIds)
            defaultThreads += std::max(1, (int)cpus.size() / 2 - 2);
        const char *ftThreads = std::getenv("FT_THREADS");
        const int requested = ftThreads ? std::atoi(ftThreads) : 0;
        threads = requested > 0 ? requested :
            GetFastllmEnv().numaThreads > 0 ? numaCnt * GetFastllmEnv().numaThreads : defaultThreads;
        SetThreads(threads);
    }

    NumaConfig::NumaConfig (int threads, AliveThreadPool *pool, MachineNumaInfo *machineNumaInfo) {
        this->threads = threads;
        this->numaCnt = machineNumaInfo->numaCnt;

        this->numaToCpuDict.resize(this->numaCnt);
        int per = this->threads / this->numaCnt;
        this->threads = per * this->numaCnt;
        this->threadIdToNumaDict.resize(this->threads);
        int threadIdx = 0;

        for (int i = 0; i < this->numaCnt; i++) {
            std::vector<int> workerCpuIds = machineNumaInfo->cpuIds[i];
            if (std::getenv("FASTLLM_NUMAS_DISABLE_LLC_SPREAD") == nullptr) {
                workerCpuIds = SpreadCpusAcrossLastLevelCaches(workerCpuIds);
            }
            for (int j = 0; j < per; j++) {
                this->threadIdToNumaDict[threadIdx] = i;
                this->numaToCpuDict[i].push_back(
                    std::make_pair(threadIdx++, workerCpuIds[j % workerCpuIds.size()]));
                
                printf("threadIdx: %d, use cpu %d, bind to numa %d\n",
                       threadIdx - 1, workerCpuIds[j % workerCpuIds.size()], machineNumaInfo->nodeIds[i]);
            }
        }

        std::vector<fastllm::BindCPUOp*> ops;
        ops.resize(this->threads);
        for (int i = 0; i < this->numaCnt; i++) {
            for (int j = 0; j < this->numaToCpuDict[i].size(); j++) {
                ops[this->numaToCpuDict[i][j].first] = new BindCPUOp(this->numaToCpuDict[i][j].second, machineNumaInfo->nodeIds[i]);
            }
        }
        
        for (int i = 0; i < ops.size(); i++) {
            pool->PushOp(i, ops[i]);
        }

        for (int i = 0; i < ops.size(); i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }
}
