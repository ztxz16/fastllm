#include "../src/host_pool.h"
#include "../src/numa.h"
#include <iostream>
#include <thread>

using namespace fastkv;
static void Check(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
static void Verify(const Buffer &buffer, uint8_t value) {
    Bytes bytes(buffer.Size()); buffer.Read(bytes.data(), 0, bytes.size());
    Check(std::all_of(bytes.begin(), bytes.end(), [value](uint8_t b) { return b == value; }), "live allocation changed");
}
static void NumaPlacement() {
#if defined(__linux__) && defined(FASTKV_CUDA)
    constexpr size_t bits = sizeof(unsigned long) * 8;
    unsigned long allowed = 0;
    if (syscall(SYS_get_mempolicy, nullptr, &allowed, bits + 1, nullptr, MPOL_F_MEMS_ALLOWED)) return;
    for (size_t node = 0; node < bits; ++node) if (allowed & (1UL << node)) {
        // Probe permission separately: restricted containers retain the
        // documented first-touch fallback and cannot assert placement.
        void *probe = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        Check(probe != MAP_FAILED, "NUMA policy probe allocation");
        unsigned long expected = 1UL << node;
        int status = syscall(SYS_mbind, probe, 4096, MPOL_PREFERRED, &expected, bits + 1, 0);
        munmap(probe, 4096);
        if (status) continue;
        DeviceNuma placement(-1); placement.node = int(node);
        auto memory = AllocateNumaHost(4096, placement);
        unsigned long actual = 0; int mode = -1;
        Check(!syscall(SYS_get_mempolicy, &mode, &actual, bits + 1, memory.Data(), MPOL_F_ADDR), "read NUMA policy");
        Check(mode == MPOL_PREFERRED && actual == expected, "NUMA allocation lost its preferred node");
    }
#endif
}
int main() {
    try {
        NumaPlacement();
        Budget budget(1 << 20, "test host slabs");
        {
            HostPool pool;
            std::vector<Buffer> buffers;
            for (int i = 0; i < 8; ++i) {
                auto b = i ? pool.TryAllocate(8192, 0) :
                    pool.Add(Buffer::Allocate(65536), budget.Reserve(65536), 0, 8192);
                Check(b.Size() == 8192, "slab slicing");
                Bytes bytes(b.Size(), uint8_t(i)); b.Write(bytes.data(), 0, bytes.size());
                buffers.push_back(b);
            }
            Check(budget.Used() == 65536, "charge full slab only once");
            Check(!pool.TryAllocate(1, 0).Size(), "full slab accepted allocation");
            Check(!pool.TryAllocate(1, 1).Size(), "NUMA nodes shared a slab");
            auto held = buffers[1].Slice(10, 100).ReadOnly();
            buffers[1] = {}; buffers[2] = {};
            Check(!pool.TryAllocate(16000, 0).Size(), "slice lease released early");
            held = {};
            auto merged = pool.TryAllocate(16384, 0);
            Check(merged.Size() == 16384, "adjacent free ranges did not coalesce");
            for (int i : {0, 3, 4, 5, 6, 7}) Verify(buffers[i], uint8_t(i));
            merged = {}; buffers.clear();
            Check(budget.Used() == 0, "empty slab retained its budget");
        }
        Buffer survivor;
        {
            HostPool pool;
            survivor = pool.Add(Buffer::Allocate(32768), budget.Reserve(32768), -1, 513);
            Bytes data(513, 73); survivor.Write(data.data(), 0, data.size());
        }
        Verify(survivor, 73);
        Check(budget.Used() == 32768, "pool destruction released a live lease");
        survivor = {}; Check(budget.Used() == 0, "outliving lease leaked its slab");
        // Readers keep one allocation pinned while other threads repeatedly
        // allocate/free differently sized holes in both NUMA groups.
        {
            HostPool pool;
            auto a = pool.Add(Buffer::Allocate(1 << 18), budget.Reserve(1 << 18), 0, 1024);
            auto b = pool.Add(Buffer::Allocate(1 << 18), budget.Reserve(1 << 18), 1, 1024);
            std::vector<std::future<void>> tasks;
            for (int rank = 0; rank < 8; ++rank) tasks.push_back(std::async(std::launch::async, [&, rank] {
                for (int i = 0; i < 1000; ++i) {
                    auto piece = pool.TryAllocate(513 + (i * 17 + rank * 13) % 8192, rank % 2);
                    Check(piece.Size() > 0, "concurrent hole allocation failed");
                    Bytes bytes(piece.Size(), uint8_t(rank + i)); piece.Write(bytes.data(), 0, bytes.size());
                    std::this_thread::yield(); Verify(piece, uint8_t(rank + i));
                }
            }));
            for (auto &task : tasks) task.get();
            Check(budget.Used() == 2 * (1 << 18), "concurrent reuse changed physical budget");
        }
        Check(budget.Used() == 0, "concurrent leases leaked slabs");
        std::cout << "Host slabs: coalescing, NUMA isolation, lifetime, concurrency and physical budget PASS\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
