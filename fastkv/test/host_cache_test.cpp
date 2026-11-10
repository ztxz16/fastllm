#include "fastkv/host_cache.h"
#include <iostream>

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
    s.payloads = {ranges, {false, {{0, 513, {}}}}, {}};
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
int main(int argc, char **argv) {
    try {
        const int device = argc > 1 ? std::stoi(argv[1]) : -1;
        HostCache cache({1 << 20, 1 << 20, 8192, 16, 2});
        auto a = State(2, device);
        Check(cache.PublishAsync("model/layout", {1, 2}, a), "publish"); cache.Drain();
        auto hit = cache.Find("model/layout", {1, 2, 3});
        Check(hit.Tokens() == 2 && hit.Bytes() == 8705, "complete prefix match");
        Equal(a.state, Restore(cache, hit, device));
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
        std::cout << "Host cache: byte-exact incremental, boundary, eviction, cancellation, namespace PASS device=" << device << '\n';
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
