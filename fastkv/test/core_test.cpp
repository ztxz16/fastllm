#include "fastkv/fastkv.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <thread>

using namespace fastkv;
static void Check(bool b, const char *s) {
    if (!b)
        throw std::runtime_error(s);
}
template <class F> static void Fails(F f) {
    bool failed = false;
    try {
        f();
    } catch (const std::exception &) {
        failed = true;
    }
    Check(failed, "expected failure");
}
static Buffer Pattern(size_t n, uint8_t seed) {
    auto b = Buffer::Allocate(n);
    auto *p = static_cast<uint8_t *>(b.MutableData());
    for (size_t i = 0; i < n; ++i)
        p[i] = uint8_t(i * 131 + seed);
    return b;
}
int main() {
    try {
        Writer w;
        w.U32(0xfedcba98);
        w.U64(0x123456789abcdef0ULL);
        w.String("model:state");
        w.Ints({0, -1, 123456});
        Reader r(w.bytes);
        Check(r.U32() == 0xfedcba98, "u32");
        Check(r.U64() == 0x123456789abcdef0ULL, "u64");
        Check(r.String() == "model:state", "string");
        Check(r.Ints() == std::vector<int>({0, -1, 123456}), "ints");
        r.End();
        Fails([&] { r.U32(); });
        Bytes malformed{255, 255, 255, 127};
        Fails([&] {
            Reader b(malformed);
            b.Blob();
        });
        auto source = Pattern(8 * 12, 7);
        std::vector<Shard> sources, targets;
        // Reshard rows and columns simultaneously, with one declared replica.
        for (int i = 0; i < 2; ++i) {
            auto b = Buffer::Allocate(4 * 12);
            b.CopyFrom(source, i * 48, 0, 48);
            sources.push_back({"kv",
                               "u8",
                               "p" + std::to_string(i),
                               b,
                               {8, 12},
                               {uint64_t(i * 4), 0},
                               {4, 12},
                               1,
                               uint64_t(i + 1)});
        }
        sources.push_back(sources[0]);
        sources.back().owner = "replica";
        for (int i = 0; i < 3; ++i)
            targets.push_back({"kv",
                               "u8",
                               "d" + std::to_string(i),
                               Buffer::Allocate(8 * 4),
                               {8, 12},
                               {0, uint64_t(i * 4)},
                               {8, 4},
                               1,
                               0});
        auto plan = PlanCopies(sources, targets);
        for (auto c : plan)
            targets[c.target].buffer.CopyFrom(sources[c.source].buffer, c.sourceOffset, c.targetOffset,
                                              c.bytes);
        for (int i = 0; i < 3; ++i)
            for (int row = 0; row < 8; ++row)
                Check(!std::memcmp(static_cast<const uint8_t *>(targets[i].buffer.Data()) + row * 4,
                                   static_cast<const uint8_t *>(source.Data()) + row * 12 + i * 4, 4),
                      "reshard payload");
        auto broken = sources;
        broken.erase(broken.begin() + 1);
        Fails([&] { PlanCopies(broken, targets); });
        broken = sources;
        broken[2].replica = 0;
        Fails([&] { PlanCopies(broken, targets); });
        broken = sources;
        broken[0].representation = "fp8";
        Fails([&] { PlanCopies(broken, targets); });
        Budget budget(10);
        auto lease = budget.Reserve(8);
        Check(budget.Used() == 8, "budget");
        Fails([&] { budget.Reserve(3); });
        lease.reset();
        Check(!budget.Used(), "release");
        Message fragments;
        fragments.buffers = {Pattern(8, 1), Pattern(8, 2), Pattern(17, 3), Pattern(0, 0)};
        auto packed = Snapshot(fragments, -1, 16);
        Check(packed.buffers[0].Id() == packed.buffers[1].Id(), "small tensors share an arena");
        Check(packed.buffers[1].Id() != packed.buffers[2].Id(), "arena size is bounded");
        for (size_t i = 0; i < fragments.buffers.size(); ++i) {
            Check(packed.buffers[i].Size() == fragments.buffers[i].Size(), "packed slice length");
            Check(!std::memcmp(packed.buffers[i].Data(), fragments.buffers[i].Data(),
                               fragments.buffers[i].Size()),
                  "packed slice contents");
        }
        Fails([&] { packed.buffers[0].MutableData(); });
        ObjectStore store(16);
        Message m;
        m.buffers.push_back(Pattern(8, 3));
        store.Put("a", m);
        auto held = store.Get("a");
        static_cast<uint8_t *>(m.buffers[0].MutableData())[0] = 90;
        Check(static_cast<const uint8_t *>(held->buffers[0].Data())[0] == 3, "immutable publication");
        Fails([&] { held->buffers[0].MutableData(); });
        Fails([&] {
            uint8_t v = 1;
            held->buffers[0].Write(&v, 0, 1);
        });
        store.Put("b", m);
        store.Remove("a");
        Check(store.BytesUsed() == 16, "reader pins removed object");
        held.reset();
        Check(store.BytesUsed() == 8, "release removed object");
        PrefixCache prefixes(4096, 3);
        prefixes.Publish("model/tenant-a", {1, 2, 3}, m);
        prefixes.Publish("model/tenant-a", {1, 2, 3, 4, 5}, m);
        auto hit = prefixes.Find("model/tenant-a", {1, 2, 3, 4, 5, 6}, 5);
        Check(hit.tokens == 5 && hit.state, "longest prefix");
        Check(prefixes.Find("model/tenant-a", {1, 2, 3, 9, 8}, 4).tokens == 3, "branch prefix");
        Check(!prefixes.Find("model/tenant-b", {1, 2, 3, 4, 5}, 5).state, "tenant isolation");
        Check(!prefixes.Find("model/tenant-a", {1, 2, 3}, 2).state, "last token boundary");
        for (int i = 0; i < 8; ++i)
            prefixes.Publish("other", {i, i + 1}, m);
        Check(hit.state->buffers[0].Size() == 8, "prefix lease survives eviction");
        Options options;
        options.timeoutMs = 3000;
        options.workers = 4;
        options.maxPayloadBytes = 8 << 20;
        options.maxInflightBytes = 64 << 20;
        Service server(options);
        server.Start("127.0.0.1:0", [](const Message &m) {
            if (m.metadata == Bytes{9})
                throw std::runtime_error("handler error");
            return m;
        });
        Service client(options);
        std::vector<std::future<Message>> requests;
        for (int i = 0; i < 12; ++i) {
            Message a;
            a.metadata = {uint8_t(i + 20)};
            a.buffers = {Pattern((1 << 20) + i, uint8_t(i))};
            requests.push_back(client.Submit(server.Address(), std::move(a)));
        }
        for (int i = 0; i < 12; ++i) {
            auto b = requests[i].get();
            auto expect = Pattern((1 << 20) + i, uint8_t(i));
            Check(b.metadata == Bytes{uint8_t(i + 20)} && b.buffers.size() == 1, "rpc metadata");
            Check(!std::memcmp(b.buffers[0].Data(), expect.Data(), expect.Size()), "rpc bytes");
        }
        Message error;
        error.metadata = {9};
        Fails([&] { client.Call(server.Address(), error); });
        Check(client.Stats().calls == 12 && client.Stats().tcpBytes > 24 * (1 << 20), "metrics");
        auto unfragmented = client.Call(server.Address(), packed);
        Check(unfragmented.buffers[0].Id() == unfragmented.buffers[1].Id(), "receive arenas");
        for (size_t i = 0; i < fragments.buffers.size(); ++i)
            Check(unfragmented.buffers[i].Size() == fragments.buffers[i].Size() &&
                      !std::memcmp(unfragmented.buffers[i].Data(), fragments.buffers[i].Data(),
                                   fragments.buffers[i].Size()),
                  "packed TCP round trip");
        // A failed admission leaves the receiver usable for subsequent calls.
        Options small = options;
        small.maxInflightBytes = 4;
        Service limited(small);
        limited.Start("127.0.0.1:0", [](const Message &m) { return m; });
        Message large;
        large.buffers = {Pattern(8, 1)};
        Fails([&] { client.Call(limited.Address(), large); });
        auto empty = client.Call(limited.Address(), Message{});
        Check(empty.buffers.empty(), "post-failure service");
        limited.Stop();
        server.Stop();
        client.Stop();
        // A network deadline and a stopped client must release active and
        // queued work without waiting for the server's handler to return.
        Options slowOptions = options;
        slowOptions.timeoutMs = 30000;
        Service slow(slowOptions);
        std::promise<void> entered, release;
        auto gate = release.get_future().share();
        slow.Start("127.0.0.1:0", [&](const Message &request) {
            if (request.metadata == Bytes{1}) {
                std::this_thread::sleep_for(std::chrono::milliseconds(150));
            } else if (request.metadata == Bytes{2}) {
                entered.set_value();
                gate.wait();
            }
            return request;
        });
        Options shortOptions = options;
        shortOptions.timeoutMs = 50;
        Service impatient(shortOptions);
        Message deadlineRequest;
        deadlineRequest.metadata = {1};
        Fails([&] { impatient.Call(slow.Address(), deadlineRequest); });
        Check(impatient.Call(slow.Address(), {}).buffers.empty(), "usable after timeout");
        Options oneWorker = slowOptions;
        oneWorker.workers = 1;
        Service cancellable(oneWorker);
        Message blocked;
        blocked.metadata = {2};
        auto active = cancellable.Submit(slow.Address(), blocked);
        entered.get_future().wait();
        auto queued = cancellable.Submit(slow.Address(), {});
        const auto begin = std::chrono::steady_clock::now();
        cancellable.Stop();
        const auto elapsed = std::chrono::steady_clock::now() - begin;
        // Release first so a failed assertion cannot deadlock server teardown.
        release.set_value();
        Check(elapsed < std::chrono::seconds(2), "stop must interrupt transport waits");
        Fails([&] { (void)active.get(); });
        Fails([&] { (void)queued.get(); });
        slow.Stop();
        std::cout << "Fast KV Manager: codec, resharding, leases, immutable store, TCP concurrency and "
                     "failure recovery PASS\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
