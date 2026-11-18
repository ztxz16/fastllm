#include "fastkv/host_cache.h"
#include <iostream>
#include <thread>

using namespace fastkv;
static void Check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
template <class F> static void Fails(F f) {
    bool failed = false; try { f(); } catch (const std::exception &) { failed = true; }
    Check(failed, "expected failure");
}
static CacheSnapshot State(int pages, int device, const std::string &version = "a") {
    CacheSnapshot s;
    s.state.metadata = {1, 2, uint8_t(pages)};
    auto range = Buffer::Allocate(pages * 4096, device), window = Buffer::Allocate(513, device);
    CachePayload ranges; ranges.prefixRange = true;
    for (int i = 0; i < pages; ++i) {
        Bytes bytes(4096); for (size_t j = 0; j < bytes.size(); ++j) bytes[j] = uint8_t(i * 37 + j * 17);
        range.Write(bytes.data(), i * 4096, bytes.size());
        ranges.regions.push_back({size_t(i * 4096), 4096, version + std::to_string(i)});
    }
    Bytes bytes(513, uint8_t(pages)); window.Write(bytes.data(), 0, bytes.size());
    s.state.buffers = {range.ReadOnly(), window.ReadOnly(), Buffer::Allocate(0, device)};
    s.payloads = {ranges, {false, {{0, 513, {}}}, CacheStorage::Window}, {}};
    return s;
}
static Message Restore(HostCache &cache, HostCache::Hit hit, int device) {
    std::future<Message> result;
    Check(cache.RestoreAsync(hit, {device, device, device}, result), "restore queue");
    return result.get();
}
static void Equal(const Message &a, const Message &b) {
    Check(a.metadata == b.metadata && a.buffers.size() == b.buffers.size(), "metadata/boundary");
    for (size_t i = 0; i < a.buffers.size(); ++i) {
        Check(a.buffers[i].Size() == b.buffers[i].Size(), "payload size");
        Bytes x(a.buffers[i].Size()), y(b.buffers[i].Size());
        a.buffers[i].Read(x.data(), 0, x.size()); b.buffers[i].Read(y.data(), 0, y.size());
        Check(x == y, "payload bytes");
    }
}
static void TestSplitBudgets(int device) {
    HostCache::Options options;
    options.SetBudgets(20000, 18000);
    Check(options.bytes == 20000 && options.swaBytes == 2000, "window remainder");
    options.SetBudgets(20000, -1, 3000);
    Check(options.bytes == 20000 && options.swaBytes == 3000, "KV remainder");
    Fails([&] { options.SetBudgets(20000, 18000, 3000); });
    Fails([&] { options.SetBudgets(0, -2, 3000); });
    Check(options.bytes == 20000 && options.swaBytes == 3000, "invalid budget changed configuration");

    for (bool windowPressure : {true, false}) {
        const size_t kvLimit = windowPressure ? 1 << 20 : 16000;
        const size_t windowLimit = windowPressure ? 513 : 1 << 16;
        options.SetBudgets(0, kvLimit, windowLimit);
        HostCache cache(options);
        auto first = State(2, device);
        auto next = State(2, device, "other");
        auto publish = [&](int token, const CacheSnapshot &state) {
            Check(cache.PublishAsync("split", {token}, state), "split publication not queued");
            cache.Drain();
        };
        publish(1, first);
        auto pinned = cache.Find("split", {1});
        Check(bool(pinned), "first split entry missing");
        publish(2, next);
        Check(!cache.Find("split", {2}), "pinned entry let another class fund a partial hit");
        Equal(first.state, Restore(cache, pinned, device));
        pinned = {};
        publish(2, next);
        auto hit = cache.Find("split", {2});
        Check(hit && !cache.Find("split", {1}), "budget pressure did not evict the old checkpoint");
        Equal(next.state, Restore(cache, hit, device));
        const auto stats = cache.Stats();
        Check(stats.kvBytes > 8192 && stats.kvBytes <= kvLimit && stats.swaBytes >= 513 &&
              stats.swaBytes <= windowLimit && stats.bytes == stats.kvBytes + stats.swaBytes,
              "physical per-class limit exceeded");
        hit = {};
        auto tooBig = State(windowPressure ? 2 : 5, device);
        if (windowPressure) {
            tooBig.state.buffers[1] = Buffer::Allocate(514, device);
            tooBig.payloads[1].regions[0].bytes = 514;
        }
        publish(3, tooBig);
        Check(!cache.Find("split", {3}) && cache.Find("split", {2}) &&
              cache.Stats().evictions == stats.evictions, "oversize entry displaced a usable checkpoint");
    }
    options.SetBudgets(0, 1 << 20, 0);
    HostCache noWindows(options);
    noWindows.PublishAsync("mixed", {1}, State(1, device));
    noWindows.Drain();
    Check(!noWindows.Find("mixed", {1}), "zero SWA cap admitted incomplete mixed state");
    auto kvOnly = State(2, device);
    kvOnly.state.buffers.resize(1); kvOnly.payloads.resize(1);
    noWindows.PublishAsync("kv", {2}, kvOnly); noWindows.Drain();
    Check(noWindows.Find("kv", {2}).Bytes() == 8192 && !noWindows.Stats().swaBytes, "KV-only hit");
}

int main(int argc, char **argv) {
    try {
        const int device = argc > 1 ? std::stoi(argv[1]) : -1;
        TestSplitBudgets(device);
        HostCache cache({1 << 20, 1 << 20, 8192, 16, 2});
        auto a = State(2, device);
        Check(cache.PublishAsync("model/layout", {1, 2}, a), "publish"); cache.Drain();
        auto hit = cache.Find("model/layout", {1, 2, 3});
        Check(hit.Tokens() == 2 && hit.Bytes() == 8705, "complete prefix match");
        Equal(a.state, Restore(cache, hit, device));
        std::vector<Buffer> targets;
        for (const auto &source : a.state.buffers)
            targets.push_back(source.Size() ? Buffer::Allocate(source.Size() + 8, device).Slice(4, source.Size()) : Buffer{});
        std::future<Message> direct;
        Fails([&] { cache.RestoreIntoAsync(hit, {targets.front()}, direct); });
        Check(cache.RestoreIntoAsync(hit, targets, direct), "direct restore queue");
        auto restored = direct.get();
        Equal(a.state, restored);
        for (size_t i=0; i<targets.size(); ++i) Check(targets[i].Data()==restored.buffers[i].Data(), "direct target replaced");
        Check(cache.ProbeTokens("other/layout", {1, 2}) == 0, "namespace isolation");
        Check(cache.ProbeTokens("model/layout", {1, 9}) == 0, "token isolation");
        auto b = State(3, device);
        Check(cache.PublishAsync("model/layout", {1, 2, 3}, b), "incremental publish"); cache.Drain();
        Check(cache.Stats().copiedToHost == 2 * 513 + 3 * 4096, "only new page and boundary copied");
        Equal(b.state, Restore(cache, cache.Find("model/layout", {1, 2, 3}), device));
        auto moved = State(4, device, "new-allocation");
        Check(cache.PublishAsync("model/layout", {1, 2, 3, 4}, moved, cache.Find("model/layout", {1, 2, 3})), "parent publish");
        cache.Drain();
        Check(cache.Stats().copiedToHost == 3 * 513 + 4 * 4096, "restored parent provenance");
        Equal(moved.state, Restore(cache, cache.Find("model/layout", {1, 2, 3, 4}), device));
        Equal(a.state, Restore(cache, hit, device)); // old boundary remains immutable
        auto invalid = State(2, device); invalid.payloads[0].regions[1].offset++;
        Fails([&] { cache.PublishAsync("model/layout", {7}, invalid); });
        auto cancel = std::make_shared<Cancellation>(); cancel->Cancel();
        std::future<Message> cancelled;
        Check(cache.RestoreAsync(hit, {device, device, device}, cancelled, cancel), "cancel queue");
        Fails([&] { (void)cancelled.get(); }); cache.Drain();
        // Physical budget includes outstanding readers; eviction cannot invalidate them.
        HostCache small({18000, 1 << 20, 8192, 4, 1});
        small.PublishAsync("a", {1}, a); small.Drain(); auto pinned = small.Find("a", {1});
        Check(bool(pinned), "small cache publication");
        small.PublishAsync("b", {2}, a); small.Drain();
        Check(small.ProbeTokens("b", {2}) == 0, "pinned cache rejects pressure");
        Equal(a.state, Restore(small, pinned, device));
        pinned = {}; small.PublishAsync("b", {2}, a); small.Drain();
        Check(small.ProbeTokens("a", {1}) == 0 && small.ProbeTokens("b", {2}) == 1, "unpin eviction");
        Check(small.Stats().bytes <= 18000, "physical bound");
        // Fill and evict repeatedly while the pool contains partly live slabs.
        // Region versions differ so every new checkpoint writes fresh bytes.
        {
            HostCache churn({256 << 10, 1 << 20, 8192, 4, 2});
            for (int i = 0; i < 100; ++i) {
                auto snapshot = State(4, device, std::to_string(i) + ":");
                Check(churn.PublishAsync("churn", {i}, snapshot), "churn publish"); churn.Drain();
                auto latest = churn.Find("churn", {i});
                Check(bool(latest), "slab fragmentation prevented publication");
                Equal(snapshot.state, Restore(churn, latest, device));
                Check(churn.Stats().bytes <= (256 << 10), "slab physical budget exceeded");
            }
            Check(churn.Stats().failures == 0 && churn.Stats().skipped == 0 && churn.Stats().evictions > 0,
                  "churn did not evict cleanly");
        }
        // Different recomputations may not inherit the earlier numeric state.
        auto wrong = State(3, device, "different");
        auto changed = Buffer::Allocate(wrong.state.buffers[0].Size(), device);
        Bytes changedBytes(changed.Size(), 217); changed.Write(changedBytes.data(), 0, changedBytes.size());
        wrong.state.buffers[0] = changed.ReadOnly();
        cache.PublishAsync("model/layout", {9, 8, 7}, wrong, hit); cache.Drain();
        Equal(wrong.state, Restore(cache, cache.Find("model/layout", {9, 8, 7}), device));
        HostCache::Options limits{1 << 20, 1 << 20, 8192, 4, 1};
        limits.restoreDeviceBytes[device] = 10000;
        HostCache bounded(limits);
        bounded.PublishAsync("a", {1}, a); bounded.Drain();
        auto limitedHit = bounded.Find("a", {1});
        auto held = Restore(bounded, limitedHit, device);
        bounded.Drain();
        std::future<Message> second;
        Check(!bounded.RestoreAsync(limitedHit, {device,device,device}, second) && !second.valid(),
              "completed but unconsumed restore lost destination credit");
        held = {};
        Equal(a.state, Restore(bounded, limitedHit, device));
        // Both staging slots cross region boundaries, including a short final
        // slot. A one-byte budget keeps the same contract without rounding up.
        for (size_t staging : {size_t(1), size_t(4099), size_t(65537)}) {
            HostCache odd({1 << 20, 1 << 20, staging, 4, 1});
            Check(odd.PublishAsync("odd", {1}, a), "odd staging publish"); odd.Drain();
            Equal(a.state, Restore(odd, odd.Find("odd", {1}), device));
        }
        // Larger than every staging window: cancel while DMA is in flight,
        // then reuse the worker. Never release a source/destination early.
        if (device >= 0) {
            auto large = State(1024, device);
            HostCache streamed({32 << 20, 32 << 20, 8192, 4, 1});
            Check(streamed.PublishAsync("stream", {1}, large), "stream publish"); streamed.Drain();
            auto streamHit = streamed.Find("stream", {1});
            auto mid = std::make_shared<Cancellation>();
            std::future<Message> work;
            Check(streamed.RestoreAsync(streamHit, {device,device,device}, work, mid), "stream restore");
            std::this_thread::sleep_for(std::chrono::milliseconds(1)); mid->Cancel();
            // Completion may win a cancellation race; either outcome is valid.
            try { Equal(large.state, work.get()); } catch (const std::runtime_error &e) {
                Check(std::string(e.what()).find("cancelled") != std::string::npos, "unexpected restore error");
            }
            Equal(large.state, Restore(streamed, streamHit, argc > 2 ? std::stoi(argv[2]) : device));
            std::future<Message> dropped;
            Check(streamed.RestoreAsync(streamHit, {device,device,device}, dropped), "dropped restore");
            dropped = {}; streamed.Drain();
            if (argc > 2) {
                const int other = std::stoi(argv[2]);
                auto mixed = State(1024, other);
                mixed.state.buffers.insert(mixed.state.buffers.end(), large.state.buffers.begin(), large.state.buffers.end());
                mixed.payloads.insert(mixed.payloads.end(), large.payloads.begin(), large.payloads.end());
                Check(streamed.PublishAsync("mixed", {2}, mixed), "multi-device publish"); streamed.Drain();
                auto mixedHit = streamed.Find("mixed", {2});
                const std::vector<int> placement{device,device,device,other,other,other};
                std::future<Message> parallel;
                Check(streamed.RestoreAsync(mixedHit, placement, parallel), "multi-device restore");
                Equal(mixed.state, parallel.get());
                auto stop = std::make_shared<Cancellation>();
                Check(streamed.RestoreAsync(mixedHit, placement, parallel, stop), "multi-device cancellation");
                std::this_thread::sleep_for(std::chrono::milliseconds(1)); stop->Cancel();
                try { Equal(mixed.state, parallel.get()); } catch (const std::runtime_error &e) {
                    Check(std::string(e.what()).find("cancelled") != std::string::npos, "multi-device restore error");
                }
                Check(streamed.RestoreAsync(mixedHit, placement, parallel), "multi-device retry");
                Equal(mixed.state, parallel.get());
            }
        }
        std::cout << "Host cache: byte-exact incremental, boundary, eviction, cancellation, namespace PASS device=" << device << '\n';
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
