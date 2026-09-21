#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/numas/numas.h"
#include "devices/numas/numasdevice.h"
#include <sched.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <stdexcept>
#include <thread>

namespace fastllm { NumaConfig *GetNumaConfig(); }
using namespace fastllm;

static void Require(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

static std::pair<int, int> Core(int cpu) {
    const auto path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
    std::pair<int, int> result;
    std::ifstream package(path + "physical_package_id"), core(path + "core_id");
    Require(bool(package >> result.first) && bool(core >> result.second), "read CPU topology");
    return result;
}

int main() {
    try {
        const int count = std::min(2, FastllmCudaGetDeviceCount());
        if (count == 0) return 77;
        std::vector<int> devices;
        for (int device = 0; device < count; ++device) devices.push_back(device);
        cpu_set_t original;
        Require(sched_getaffinity(0, sizeof(original), &original) == 0, "read affinity");
        const auto masks = GetNumasCudaWorkerCpuSets(devices);
        Require(masks.size() == devices.size(), "one CPU set per CUDA worker");
        std::set<std::pair<int, int>> occupied;
        const auto &workers = GetNumaConfig()->numaToCpuDict;
        for (const auto &node : workers) {
            for (const auto &worker : node) occupied.insert(Core(worker.second));
        }
        for (int rank = 0; rank < count; ++rank) {
            const int node = FastllmCudaGetHostNumaNode(devices[rank]);
            std::set<std::pair<int, int>> used;
            for (int cpu : masks[rank]) {
                Require(CPU_ISSET(cpu, &original), "planner widened caller affinity");
                Require(numa_node_of_cpu(cpu) == node, "worker is remote from CUDA device");
                Require(!occupied.count(Core(cpu)), "worker shares an expert core");
                Require(used.insert(Core(cpu)).second, "duplicate SMT siblings");
            }
            if (masks[rank].empty()) continue;
            std::exception_ptr failure;
            std::thread test([&] {
                try {
                    Require(BindNumasWorkerCpuSet(masks[rank]), "bind planned set");
                    cpu_set_t bound;
                    Require(sched_getaffinity(0, sizeof(bound), &bound) == 0, "read bound set");
                    Require(CPU_COUNT(&bound) == (int)masks[rank].size(), "wrong bound set size");
                    for (int cpu : masks[rank]) Require(CPU_ISSET(cpu, &bound), "missing planned CPU");
                    cpu_set_t narrowed;
                    CPU_ZERO(&narrowed); CPU_SET(masks[rank].front(), &narrowed);
                    Require(sched_setaffinity(0, sizeof(narrowed), &narrowed) == 0, "narrow affinity");
                    Require(BindNumasWorkerCpuSet(masks[rank]), "intersect narrowed affinity");
                    Require(!BindNumasWorkerCpuSet({-1, CPU_SETSIZE}), "invalid set accepted");
                    Require(!BindNumasWorkerCpuSet({}), "empty set accepted");
                    Require(sched_getaffinity(0, sizeof(bound), &bound) == 0 &&
                            CPU_EQUAL(&bound, &narrowed), "binding widened or lost narrowed set");
                } catch (...) { failure = std::current_exception(); }
            });
            test.join();
            if (failure) std::rethrow_exception(failure);
            std::printf("PASS rank=%d node=%d spare_cores=%zu\n", rank, node, masks[rank].size());
        }
        // With only an expert-worker CPU available, leave scheduling unchanged.
        int occupiedCpu = -1;
        for (const auto &node : workers) for (const auto &worker : node) {
            if (CPU_ISSET(worker.second, &original)) occupiedCpu = worker.second;
        }
        if (occupiedCpu >= 0) {
            std::exception_ptr failure;
            std::thread test([&] {
                try {
                    cpu_set_t single; CPU_ZERO(&single); CPU_SET(occupiedCpu, &single);
                    Require(sched_setaffinity(0, sizeof(single), &single) == 0, "restrict to occupied CPU");
                    for (const auto &mask : GetNumasCudaWorkerCpuSets(devices)) {
                        Require(mask.empty(), "planner reused an occupied physical core");
                    }
                } catch (...) { failure = std::current_exception(); }
            });
            test.join();
            if (failure) std::rethrow_exception(failure);
        }
        cpu_set_t after;
        Require(sched_getaffinity(0, sizeof(after), &after) == 0 && CPU_EQUAL(&after, &original),
                "planning changed the submitting thread's affinity");
        std::puts("ALL_PASS");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
