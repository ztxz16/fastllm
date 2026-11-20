#include "../src/internal.h"
#include <cstring>
#include <iostream>
#include <thread>
using namespace fastkv;
static void Check(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
template<class F> static void Fails(F f) {
    bool failed = false;
    try { f(); } catch (const std::exception &) { failed = true; }
    Check(failed, "pool must enforce its byte budget");
}
int main() {
    try {
        BufferPool pool(8192);
        Budget budget(8192);
        auto a = pool.Acquire(1024, -1, budget.Reserve(1024));
        std::memset(a.MutableData(), 0x15, a.Size());
        auto pinned = a.Slice(8, 64).ReadOnly();
        const auto original = a.Id();
        a = {};
        auto b = pool.Acquire(1024, -1);
        Check(b.Id() != original, "slice lease recycled early");
        std::memset(b.MutableData(), 0x27, b.Size());
        Fails([&] { pool.Acquire(1024, -1); });
        Check(budget.Used() == 1024, "slice lost receive credit");
        for (size_t i = 0; i < pinned.Size(); ++i)
            Check(((const uint8_t *)pinned.Data())[i] == 0x15, "leased data overwritten");
        b.Attachment("test-mr", [](void *, size_t) { return std::make_shared<int>(42); });
        const auto reusable = b.Id();
        b = {};
        auto c = pool.Acquire(1000, -1);
        Check(c.Id() == reusable && c.HasAttachment("test-mr"), "registration not reused across token lengths");
        pinned = {};
        Check(budget.Used() == 0, "idle arena retained receive credit");
        pool.Clear();
        Check(((const uint8_t *)c.Data())[0] == 0x27, "pool destruction invalidated live lease");
        c = {};
        Fails([&] { pool.Acquire(1024, -1); });

        BufferPool messages(16384);
        auto heldPart = messages.Acquire(8192, -1);
        Fails([&] { messages.AcquireBatch({{-1, 4096}, {-1, 8192}}); });
        auto remaining = messages.Acquire(8192, -1);
        Check(remaining.Size() == 8192, "failed message retained a partial reservation");
        const auto largeId = remaining.Id();
        remaining.Attachment("retained-mr", [](void *, size_t) { return std::make_shared<int>(1); });
        remaining = {};
        auto smaller = messages.Acquire(4096, -1);
        Check(smaller.Id() == largeId && smaller.HasAttachment("retained-mr"), "larger registered arena was not reused");
        heldPart = {}; smaller = {};
        auto whole = messages.AcquireBatch({{-1, 4096}, {-1, 4096}, {-1, 8192}});
        Check(whole.size() == 3, "oversized reuse prevented feasible batch allocation");
        whole.clear();

        // Physical pressure can occur below the configured pool budget when
        // another component owns memory. Recover idle arenas, preserve live
        // slices/MRs, and roll back an entire partially allocated message.
        size_t physicalBytes = 0, physicalLimit = 12288;
        bool failSecond = false;
        int allocationCalls = 0;
        BufferAllocator allocator = [&](size_t bytes, int device) {
            ++allocationCalls;
            if ((failSecond && allocationCalls == 2) || bytes > physicalLimit - physicalBytes)
                throw CapacityBusy("injected physical allocation pressure");
            auto owner = std::shared_ptr<uint8_t>(new uint8_t[bytes], [&, bytes](uint8_t *p) {
                physicalBytes -= bytes; delete[] p;
            });
            physicalBytes += bytes;
            return Buffer::Borrow(owner.get(), bytes, MemoryKind::Host, device, owner);
        };
        BufferPool physical(32768, {}, allocator);
        auto pinnedPhysical = physical.Acquire(4096, -1);
        std::memset(pinnedPhysical.MutableData(), 0x61, 4096);
        physical.Acquire(4096, -1); // An incompatible idle arena holds physical memory.
        bool pressure = false;
        try { physical.Acquire(8192, -1); } catch (const CapacityBusy &) { pressure = true; }
        Check(pressure && physicalBytes == 4096, "physical pressure did not release idle arenas");
        auto recovered = physical.Acquire(8192, -1);
        Check(physicalBytes == 12288 && ((const uint8_t *)pinnedPhysical.Data())[0] == 0x61,
              "pressure recovery changed a live lease");
        recovered = {}; pinnedPhysical = {}; physical.Clear();
        Check(physicalBytes == 0, "physical pool leaked allocations");
        BufferPool partial(32768, {}, allocator);
        allocationCalls = 0; failSecond = true;
        pressure = false;
        try { partial.AcquireBatch({{-1,4096},{-1,4096}}); }
        catch (const CapacityBusy &) { pressure = true; }
        Check(pressure && !physicalBytes, "failed batch retained partial physical allocations");
        failSecond = false;
        auto retry = partial.AcquireBatch({{-1,4096},{-1,8192}});
        Check(physicalBytes == 12288, "failed batch retained budget charges");
        retry.clear(); partial.Clear();

        BufferPool broken(8192, {}, [](size_t, int) -> Buffer {
            throw std::runtime_error("permanent allocator failure");
        });
        bool permanent = false;
        try { broken.Acquire(4096, -1); }
        catch (const CapacityBusy &) { throw std::runtime_error("permanent failure became retryable"); }
        catch (const std::runtime_error &) { permanent = true; }
        Check(permanent, "permanent allocator failure was hidden");

        // A message shares one registration and allocation, without per-tensor
        // page rounding. Every slice must retain its credit and private bytes.
        BufferPool packedPool(12288);
        Budget packedBudget(12288);
        auto packed = packedPool.AcquirePacked({{-1, 4097}, {-1, 4097}, {-1, 0}},
                                               256, packedBudget.Reserve(8194));
        Check(packed.size() == 3 && !packed[2].Size() && packed[0].Id() == packed[1].Id(),
              "packed message lost ordering or shared allocation identity");
        Check(packed[1].Offset() % 256 == 0 && packed[1].Offset() >= packed[0].Size(),
              "packed message slices overlap or are misaligned");
        std::memset(packed[0].MutableData(), 0x35, packed[0].Size());
        std::memset(packed[1].MutableData(), 0x57, packed[1].Size());
        packed[0].Attachment("packed-mr", [](void *, size_t bytes) {
            Check(bytes == 12288, "registration must cover the complete physical arena");
            return std::make_shared<int>(1);
        });
        const auto packedId = packed[0].Id();
        auto retained = packed[1].Slice(3, 37);
        packed.clear();
        Fails([&] { packedPool.AcquirePacked({{-1, 1}}); });
        Check(packedBudget.Used() == 8194 && retained.HasAttachment("packed-mr"),
              "packed slice lost credit or registration");
        for (size_t i = 0; i < retained.Size(); ++i)
            Check(((const uint8_t *)retained.Data())[i] == 0x57, "packed slice was overwritten");
        retained = {};
        Check(!packedBudget.Used(), "packed arena retained message credit after retirement");
        Fails([&] { packedPool.AcquirePacked({{-1, 8193}, {-1, 8193}}); });
        Fails([&] { packedPool.AcquirePacked({{-1, SIZE_MAX}, {-1, 1}}); });
        Fails([&] { packedPool.AcquirePacked({{-1, 1}}, 3); });
        auto reusedPacked = packedPool.AcquirePacked({{-1, 4096}, {-1, 4096}});
        Check(reusedPacked[0].Id() == packedId && reusedPacked[1].HasAttachment("packed-mr"),
              "packed arena or MR was not reused after a failed reservation");
        packedPool.Clear();
        Check(((const uint8_t *)reusedPacked[0].Data())[0] == 0x35,
              "pool shutdown invalidated a packed lease");
        reusedPacked.clear();

        // Mixed prompt lengths on two owners reuse physical arenas. Hold a
        // slice across releases, then exercise an export above a fair share.
        {
            size_t allocations = 0;
            BufferPool ranges(64 * 4096, {}, [&](size_t bytes, int device) {
                ++allocations;
                auto storage = std::shared_ptr<uint8_t>(new uint8_t[bytes], std::default_delete<uint8_t[]>());
                return Buffer::Borrow(storage.get(), bytes, MemoryKind::Cuda, device, storage);
            });
            ranges.Preallocate({0, 1}, true);
            Check(allocations == 2, "export arenas were not warmed");
            for (int round = 0; round < 16; ++round) {
                const size_t n = (15 + round % 6) * 4096;
                auto first = ranges.Acquire(n, 0);
                auto last = ranges.Acquire(4 * 4096, 0);
                Check(first.Preallocated() && last.Preallocated() && first.Id() == last.Id() && first.Offset() + first.Size() <= last.Offset(),
                      "simultaneous exports overlapped or lost registration reuse");
                auto other = ranges.Acquire(24 * 4096, 1);
                auto pinned = first.Slice(15, 1);
                *static_cast<uint8_t *>(pinned.MutableData()) = 0xa7;
                first = {};
                bool busy = false;
                try { ranges.Acquire(30 * 4096, 0); }
                catch (const CapacityBusy &) { busy = true; }
                Check(busy && *static_cast<const uint8_t *>(pinned.Data()) == 0xa7,
                      "active export range was not protected");
            }
            Check(allocations == 2, "variable exports churned physical memory");
            for (int round = 0; round < 24; ++round) {
                const int device = round % 2;
                const size_t sizes[] = {17 * 4096, 12 * 4096, size_t(9 + round % 6) * 4096};
                BufferPool::PackedLayout layout({{device, sizes[0]}, {device, sizes[1]}, {device, sizes[2]}});
                auto buffers = ranges.AcquirePeerPackedLayout(layout);
                Check(buffers[0].Device() == device && buffers[2].Device() != device &&
                      buffers[0].Id() == buffers[1].Id(), "oversized message did not borrow a peer");
                for (int i = 0; i < 3; ++i) {
                    Check(buffers[i].Preallocated() && buffers[i].Size() == sizes[i], "peer tensor lost its backing");
                    std::memset(buffers[i].MutableData(), round+i, sizes[i]);
                }
                bool busy = false;
                try { ranges.AcquirePeerPackedLayout(layout); } catch (const CapacityBusy &) { busy = true; }
                Check(busy && allocations == 2, "oversize contention evicted a warm or live GPU slab");
                for (int i = 0; i < 3; ++i)
                    Check(static_cast<const uint8_t *>(buffers[i].Data())[sizes[i]-1] == round+i,
                          "overlapping peer leases overwrote a live tensor");
            }
            BufferPool::PackedLayout indivisible({{1, 48 * 4096}});
            auto big = std::move(ranges.AcquirePeerPackedLayout(indivisible).front());
            Check(!big.Preallocated() && big.Size() == 48 * 4096, "fair share incorrectly capped a long prompt");
            big = {};
            for (int round = 0; round < 4; ++round) {
                auto max = ranges.Acquire(64 * 4096, round % 2);
                Check(max.Size() == 64 * 4096, "could not rebalance the entire idle export budget");
            }
            auto restored = ranges.Acquire(4096, 0);
            Check(restored.Preallocated(), "rebuilt standard export arena lost direct registration");
        }

        {
            BufferPool tight(64 * 4096, {}, [](size_t bytes, int) {
                if (bytes > 16 * 4096) throw CapacityBusy("simulated initial memory pressure");
                return Buffer::Allocate(bytes);
            });
            tight.Preallocate({-1});
            auto a = tight.Acquire(4096, -1);
            auto b = tight.Acquire(8192, -1);
            Check(a.Id() == b.Id(), "smaller preallocation did not recover from initial pressure");
        }

        // Retiring an MR may block. An independent cached allocation must
        // remain usable, and physical capacity stays charged until retirement.
        BufferPool retiring(16384);
        auto old = retiring.Acquire(8192, -1);
        auto cached = retiring.Acquire(4096, -1);
        const auto cachedId = cached.Id();
        std::promise<void> entered, release;
        auto released = release.get_future().share();
        old.Attachment("slow-mr", [&](void *, size_t) {
            return std::shared_ptr<int>(new int(0), [&](int *p) {
                entered.set_value(); released.wait(); delete p;
            });
        });
        old = {}; cached = {};
        auto allocation = std::async(std::launch::async, [&] { return retiring.Acquire(12288, -1); });
        const bool retiringStarted = entered.get_future().wait_for(std::chrono::seconds(2)) == std::future_status::ready;
        auto reuse = std::async(std::launch::async, [&] { return retiring.Acquire(4096, -1); });
        const bool independent = reuse.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
        release.set_value();
        Check(retiringStarted && independent, "MR retirement held the allocation mutex");
        Check(reuse.get().Id() == cachedId, "independent cached arena lost its registration identity");
        Check(allocation.get().Size() == 12288, "retirement failed to release physical capacity");

        Options options; options.workers = 1; options.queueDepth = 16;
        options.maxInflightBytes = 8 << 20;
        Service server(options), client(options);
        server.Start("127.0.0.1:0", [](const Message &m) { return m; });
        std::vector<Message> held;
        for (int i = 0; i < 8; ++i) {
            auto buffer = Buffer::Allocate(65536 + i);
            std::memset(buffer.MutableData(), i, buffer.Size());
            held.push_back(client.Call(server.Address(), {Bytes{uint8_t(i)}, {buffer}}));
        }
        Check(client.Stats().connections == 1 && client.Stats().reusedConnections == 7,
              "sequential RPCs rebuilt connections");
        const auto endpoint = server.Address();
        server.Stop();
        Service replacement(options);
        replacement.Start(endpoint, [](const Message &m) { return m; });
        Check(client.Call(endpoint, {Bytes{93}, {}}).metadata == Bytes{93}, "stale cached channel was not discarded");
        Check(client.Stats().connections == 2, "restart did not establish a fresh connection");
        replacement.Stop(); client.Stop();
        for (int i = 0; i < 8; ++i) {
            auto &buffer = held[i].buffers[0];
            for (size_t j = 0; j < buffer.Size(); ++j)
                Check(((const uint8_t *)buffer.Data())[j] == i, "retained reply recycled or invalidated by Stop");
        }
        // Physical device limits include cached size classes, not only live bytes.
        BufferPool limited(16384, {{-1, 4096}});
        auto live = limited.Acquire(100, -1);
        Fails([&] { limited.Acquire(100, -1); });
        Fails([&] { limited.CheckCapacity({{-1, 1}, {-1, 1}}); });
        live = {};
        Check(limited.Acquire(200, -1).Size() == 200, "device budget did not reclaim a lease");

        Options bounded;
        bounded.workers = 2; bounded.maxInflightBytes = 8192;
        bounded.deviceBufferBytes[-1] = 4096; bounded.timeoutMs = 3000;
        std::atomic<bool> externalPressure{true};
        std::atomic<int> allocationAttempts{0};
        bounded.allocateBuffer = [&](size_t bytes, int device) {
            ++allocationAttempts;
            if (externalPressure) throw CapacityBusy("external pressure");
            return Buffer::Allocate(bytes, device);
        };
        Service responder, receiver(bounded);
        responder.Start("127.0.0.1:0", [](const Message &) {
            auto b = Buffer::Allocate(1024);
            std::memset(b.MutableData(), 73, b.Size());
            return Message{{}, {b}};
        });
        auto initial = receiver.Submit(responder.Address(), {});
        Check(initial.wait_for(std::chrono::milliseconds(80)) == std::future_status::timeout && allocationAttempts > 0,
              "physical pressure did not defer receive");
        externalPressure = false;
        Check(initial.wait_for(std::chrono::seconds(2)) == std::future_status::ready,
              "receive did not retry after external pressure cleared");
        auto first = initial.get();
        auto second = receiver.Submit(responder.Address(), {});
        Check(second.wait_for(std::chrono::milliseconds(80)) == std::future_status::timeout,
              "device pressure must queue instead of allocate or fail");
        first = {};
        Check(second.wait_for(std::chrono::seconds(2)) == std::future_status::ready,
              "released device budget did not unblock transfer");
        auto reply = second.get();
        Check(((const uint8_t *)reply.buffers[0].Data())[0] == 73, "queued payload corrupted");
        auto waiting = receiver.Submit(responder.Address(), {});
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        receiver.Stop();
        Fails([&] { (void)waiting.get(); });
        responder.Stop();

        std::cout << "FastKV buffer leases, memory bounds, registrations and persistent RPC PASS\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
