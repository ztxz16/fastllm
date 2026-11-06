#include "devices/numas/numas.h"
#include "devices/numas/numasdevice.h"
#include <sched.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

using namespace fastllm;
namespace fastllm { NumaConfig *GetNumaConfig(); }
static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
struct CheckWorker : MultiThreadBaseOp {
    int cpu, node;
    bool correct = false;
    CheckWorker(int cpu, int node) : cpu(cpu), node(node) {}
    void Run() override {
        cpu_set_t mask;
        struct bitmask *memory = numa_allocate_nodemask();
        int policy = MPOL_DEFAULT;
        correct = sched_getaffinity(0, sizeof(mask), &mask) == 0 &&
            CPU_COUNT(&mask) == 1 && CPU_ISSET(cpu, &mask) &&
            get_mempolicy(&policy, memory->maskp, memory->size, nullptr, 0) == 0 &&
            policy == MPOL_BIND && numa_bitmask_weight(memory) == 1 &&
            numa_bitmask_isbitset(memory, node);
        numa_free_nodemask(memory);
    }
};
int main(int argc, char **argv) {
    try {
        if (numa_available() < 0) return 77;
        cpu_set_t before, after;
        Require(sched_getaffinity(0, sizeof(before), &before) == 0, "read caller affinity");
        MachineNumaInfo topology;
        if (argc > 1) Require(topology.nodeIds.size() == 1 &&
            topology.nodeIds[0] == std::atoi(argv[1]), "wrong physical node for logical shard zero");
        for (const auto &cpus : topology.cpuIds) for (int cpu : cpus)
            Require(CPU_ISSET(cpu, &before), "topology widened inherited CPU mask");
        SetThreads(std::max(topology.numaCnt, 4));
        auto *config = GetNumaConfig();
        auto *pool = GetAlivePool();
        std::vector<std::unique_ptr<CheckWorker>> ops;
        for (int shard = 0; shard < config->numaCnt; ++shard) {
            const int node = GetNumasPhysicalNode(shard);
            for (const auto &worker : config->numaToCpuDict[shard]) {
                Require(CPU_ISSET(worker.second, &before), "worker assigned a forbidden CPU");
                ops.emplace_back(new CheckWorker(worker.second, node));
                pool->PushOp(worker.first, ops.back().get());
            }
            // Touch every page and query its actual node, not just the policy.
            const size_t bytes = 8 << 20;
            void *buffer = allocate_aligned_numa(bytes, shard);
            Require(buffer != nullptr, "allocate shard memory");
            std::memset(buffer, 0x35, bytes);
            for (size_t offset = 0; offset < bytes; offset += 4096) {
                int actual = -1;
                Require(get_mempolicy(&actual, nullptr, 0, (char *)buffer + offset,
                    MPOL_F_NODE | MPOL_F_ADDR) == 0 && actual == node, "page placed on wrong physical node");
            }
            free_aligned_numa(buffer, bytes);
        }
        for (size_t i = 0; i < ops.size(); ++i) {
            pool->Wait(i);
            Require(ops[i]->correct, "worker CPU or memory policy differs from its shard");
        }
        Require(sched_getaffinity(0, sizeof(after), &after) == 0 && CPU_EQUAL(&before, &after),
                "NUMA initialization changed caller affinity");
        std::puts("PASS: inherited NUMA topology, worker policies and actual page placement");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
