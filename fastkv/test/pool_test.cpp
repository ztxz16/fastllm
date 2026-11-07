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
        Service responder, receiver(bounded);
        responder.Start("127.0.0.1:0", [](const Message &) {
            auto b = Buffer::Allocate(1024);
            std::memset(b.MutableData(), 73, b.Size());
            return Message{{}, {b}};
        });
        auto first = receiver.Call(responder.Address(), {});
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
