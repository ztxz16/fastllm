#pragma once
#include "fastkv/fastkv.h"
#include <algorithm>
#if defined(__linux__) && defined(FASTKV_CUDA)
#include <cuda_runtime_api.h>
#include <fstream>
#include <linux/mempolicy.h>
#include <sched.h>
#include <sstream>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace fastkv {
// Best-effort, allocation/thread-local NUMA placement. No libnuma dependency,
// process-wide policy changes or CPU affinity outside the caller's allowed set.
struct DeviceNuma {
    int node = -1;
#if defined(__linux__) && defined(FASTKV_CUDA)
    cpu_set_t cpus{};
    explicit DeviceNuma(int device) {
        if (device < 0) return;
        char bus[32]{};
        if (cudaDeviceGetPCIBusId(bus, sizeof(bus), device) != cudaSuccess) return;
        for (char *p = bus; *p; ++p) if (*p >= 'A' && *p <= 'F') *p += 'a' - 'A';
        std::ifstream topology(std::string("/sys/bus/pci/devices/") + bus + "/numa_node");
        topology >> node;
        if (node < 0) return;
        std::ifstream file("/sys/devices/system/node/node" + std::to_string(node) + "/cpulist");
        std::string list, part; file >> list;
        std::istringstream items(list);
        while (std::getline(items, part, ',')) {
            std::istringstream range(part); int begin, end;
            if (!(range >> begin)) continue;
            end = begin;
            if (range.peek() == '-') { range.get(); if (!(range >> end)) continue; }
            for (int cpu = std::max(0, begin); cpu <= end && cpu < CPU_SETSIZE; ++cpu) CPU_SET(cpu, &cpus);
        }
    }
#else
    explicit DeviceNuma(int) {}
#endif
};
inline const DeviceNuma &NumaForDevice(int device) {
    // The copy worker owns this cache; its lanes drain before thread exit.
    static thread_local std::map<int, DeviceNuma> nodes;
    auto found = nodes.find(device);
    if (found == nodes.end()) found = nodes.emplace(device, DeviceNuma(device)).first;
    return found->second;
}
class NumaScope {
#if defined(__linux__) && defined(FASTKV_CUDA)
    cpu_set_t previous{};
    bool changed = false;
#endif
  public:
    explicit NumaScope(const DeviceNuma &node) {
#if defined(__linux__) && defined(FASTKV_CUDA)
        if (node.node < 0 || sched_getaffinity(0, sizeof(previous), &previous)) return;
        cpu_set_t local{};
        CPU_AND(&local, &previous, &node.cpus);
        if (CPU_COUNT(&local) && !CPU_EQUAL(&local, &previous))
            changed = sched_setaffinity(0, sizeof(local), &local) == 0;
#endif
    }
    ~NumaScope() {
#if defined(__linux__) && defined(FASTKV_CUDA)
        if (changed) sched_setaffinity(0, sizeof(previous), &previous);
#endif
    }
    NumaScope(const NumaScope &) = delete;
    NumaScope &operator=(const NumaScope &) = delete;
};
inline Buffer AllocateNumaHost(size_t bytes, const DeviceNuma &node, bool pageAligned = false) {
#if defined(__linux__) && defined(FASTKV_CUDA)
    if (bytes && (node.node >= 0 || pageAligned)) {
        void *address = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (address == MAP_FAILED) throw std::bad_alloc();
        std::shared_ptr<void> owner(address, [bytes](void *p) { munmap(p, bytes); });
        if (node.node >= 0) {
            constexpr size_t bits = sizeof(unsigned long) * 8;
            std::vector<unsigned long> mask(size_t(node.node) / bits + 1);
            mask[node.node / bits] |= 1UL << (node.node % bits);
            // PREFERRED can spill under memory pressure. Unsupported/denied policy
            // falls back to first touch on the copy worker without breaking restore.
            // Linux reads maxnode - 1 bits. Include the highest set bit too;
            // otherwise a single-node mask silently becomes the local policy.
            syscall(SYS_mbind, address, bytes, MPOL_PREFERRED, mask.data(), mask.size() * bits + 1, 0);
        }
        return Buffer::Borrow(address, bytes, MemoryKind::Host, -1, std::move(owner));
    }
#endif
    return Buffer::Allocate(bytes);
}
} // namespace fastkv
