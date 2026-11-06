#include "fastkv/fastkv.h"
#include <algorithm>
#include <iostream>
#include <random>

using namespace fastkv;
static void Check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}
template <class F> static void Fails(F call) {
    bool failed = false;
    try {
        call();
    } catch (const std::exception &) {
        failed = true;
    }
    Check(failed, "expected failure");
}
static Buffer Payload(size_t bytes, uint64_t seed, int device = -1) {
    auto result = Buffer::Allocate(bytes, device);
    Bytes data(bytes);
    // Distinct pages; a repeating 256-byte pattern would hide dedup bugs.
    for (auto &v : data) {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        v = uint8_t(seed);
    }
    result.Write(data.data(), 0, bytes);
    return result;
}
static void Equal(const Message &a, const Message &b) {
    Check(a.metadata == b.metadata && a.buffers.size() == b.buffers.size(), "state layout");
    for (size_t i = 0; i < a.buffers.size(); ++i) {
        Check(a.buffers[i].Size() == b.buffers[i].Size(), "tensor boundary");
        Bytes x(a.buffers[i].Size()), y(b.buffers[i].Size());
        a.buffers[i].Read(x.data(), 0, x.size());
        b.buffers[i].Read(y.data(), 0, y.size());
        Check(x == y, "state bytes");
    }
}
static void BoundariesAndLeases(int device) {
    PrefixCache cache(8 << 20, 0, device);
    Message original{{1, 0, 9}, {Payload(200003, 17, device), Payload(0, 7, device), Payload(19, 3, device)}};
    auto expected = Snapshot(original, -1);
    cache.Publish("model/a", {1, 2, 3}, original);
    uint8_t changed = 99;
    original.buffers[0].Write(&changed, 0, 1);
    auto hit = cache.Find("model/a", {1, 2, 3, 4}, 4);
    Check(hit.tokens == 3, "exact checkpoint boundary");
    Equal(expected, *hit.state);
    Fails([&] { hit.state->buffers[0].Write(&changed, 0, 1); });
    const auto before = cache.BytesUsed();
    cache.Publish("model/a", {1, 2, 3}, original);
    Check(cache.BytesUsed() == before && cache.Stats().entries == 1, "duplicate checkpoint");
    Equal(expected, *cache.Find("model/a", {1, 2, 3}, 3).state);
    Check(!cache.Find("model/b", {1, 2, 3}, 3).state, "namespace isolation");
    Check(!cache.Find("model/a", {1, 2, 8}, 3).state, "token mismatch");
    Check(!cache.Find("model/a", {1, 2, 3}, 2).state, "no synthesized checkpoint");
    cache.Publish("model/a", {1, 2, 3, 4, 5}, original);
    Check(cache.Find("model/a", {1, 2, 3, 4, 8}, 5).tokens == 3, "branch boundary");
    Check(cache.Find("model/a", {1, 2, 3, 4, 5}, 4).tokens == 3, "maxTokens bound");
    cache.Publish("model/a", {1, 2}, expected); // checkpoint inserted before an existing child
    Check(cache.Find("model/a", {1, 2, 8}, 3).tokens == 2, "late ancestor checkpoint");
    PrefixCache small(300000, 1, device);
    small.Publish("a", {1}, expected);
    auto pinned = small.Find("a", {1}, 1);
    Fails([&] { small.Publish("b", {2}, expected); });
    Equal(expected, *pinned.state);
    pinned = {};
    small.Publish("b", {2}, expected);
    Check(!small.Find("a", {1}, 1).state, "unpin allows eviction");
    Check(small.BytesUsed() < 300000, "budget release after unpin");
    auto held = small.Find("b", {2}, 1);
    Message tooLarge{{}, {Payload(400000, 88, device)}};
    Fails([&] { small.Publish("oversized", {3}, tooLarge); });
    Equal(expected, *small.Find("b", {2}, 1).state);
    std::shared_ptr<const Message> survivor;
    {
        PrefixCache temporary(1 << 20, 0, device);
        temporary.Publish("a", {1}, expected);
        survivor = temporary.Find("a", {1}, 1).state;
    }
    Equal(expected, *survivor);
}
static void GrowingPages() {
    PrefixCache cache(12 << 20);
    auto history = Payload(10 << 20, 91);
    std::vector<int> tokens{1};
    std::vector<Message> expected;
    for (int round = 0; round < 32; ++round) {
        tokens.push_back(round);
        const size_t size = (4 << 20) + round * 65536 + 13;
        Message value{{uint8_t(round)}, {history.Slice(0, size), Payload(70001, 99 + round)}};
        cache.Publish("growing", tokens, value);
        expected.push_back(value);
    }
    auto stats = cache.Stats();
    Check(stats.entries == 32, "shared pages retain history within budget");
    Check(stats.logicalBytes > stats.bytes * 15, "incremental storage sharing");
    for (int round = 0; round < 32; ++round) {
        auto hit = cache.Find("growing", tokens, round + 2);
        Check(hit.tokens == size_t(round + 2), "historical prefix retained");
        Equal(expected[round], *hit.state);
    }
    std::cout << "shared_versions=" << stats.entries << " logical_bytes=" << stats.logicalBytes
              << " stored_bytes=" << stats.bytes << '\n';
}
static void BytePressureAndCollisions() {
    PrefixCache cache(64 << 10);
    std::vector<std::vector<int>> tokens{{1}, {2}, {3}, {4}};
    std::vector<Message> states;
    for (int i = 0; i < 4; ++i)
        states.push_back({{uint8_t(i)}, {Payload(8192, 300 + i)}});
    for (int step = 0; step < 400; ++step) {
        int session = step % 4;
        if (step >= 4) {
            auto hit = cache.Find("budget", tokens[session], tokens[session].size());
            Check(bool(hit.state), "byte pressure evicted a live tip before old versions");
            Equal(states[session], *hit.state);
        }
        tokens[session].push_back(step);
        cache.Publish("budget", tokens[session], states[session]);
        Check(cache.BytesUsed() <= (64 << 10), "byte capacity exceeded");
    }
    // Deliberate collision in the two-word page hash. Exact comparison must
    // keep the distinct byte strings, including when their token prefixes match.
    uint64_t words[2] = {123, 456}, other[2] = {789, 0};
    constexpr uint64_t seed = 1469598103934665603ULL ^ 16;
    constexpr uint64_t prime = 1099511628211ULL;
    other[1] = ((seed ^ words[0]) * prime) ^ ((seed ^ other[0]) * prime) ^ words[1];
    Message a, b;
    a.buffers.push_back(Buffer::Allocate(16));
    b.buffers.push_back(Buffer::Allocate(16));
    a.buffers[0].Write(words, 0, 16);
    b.buffers[0].Write(other, 0, 16);
    cache.Publish("collision", {11}, a);
    cache.Publish("collision", {11, 12}, b);
    Equal(a, *cache.Find("collision", {11}, 1).state);
    Equal(b, *cache.Find("collision", {11, 12}, 2).state);
}
static void MultiSession() {
    // Live session tips must survive pressure from superseded checkpoints,
    // including when a caller explicitly sets an entry limit.
    PrefixCache cache(1 << 26, 256);
    std::vector<std::vector<int>> sessions(64);
    for (int i = 0; i < 64; ++i)
        sessions[i] = {10000 + i};
    std::mt19937 random(7);
    int misses = 0;
    for (int step = 0; step < 2064; ++step) {
        auto &tokens = sessions[random() % sessions.size()];
        const bool seen = tokens.size() > 1;
        tokens.push_back(step);
        auto hit = cache.Find("model", tokens, tokens.size());
        if (seen && !hit.state)
            ++misses;
        Message value{{uint8_t(step)}, {}};
        cache.Publish("model", tokens, value);
    }
    Check(misses == 0, "live session tips evicted by ancestors");
    Check(cache.Stats().entries == 256, "explicit entry cap");
    std::cout << "sessions=64 requests=2064 repeat_session_complete_misses=" << misses << '\n';
    PrefixCache many(1 << 22);
    for (int i = 0; i < 512; ++i)
        many.Publish("model", {i}, {});
    Check(many.Stats().entries == 512, "default cache has no arbitrary 256-entry cutoff");
    for (int i = 0; i < 512; ++i)
        Check(bool(many.Find("model", {i}, 1).state), "default cache retention");
}
static void ConcurrentWire() {
    PrefixCache cache(8 << 20);
    Message expected{{8, 9}, {Payload(131099, 87), Payload(17, 23)}};
    cache.Publish("wire", {1, 2, 3}, expected);
    Options options;
    options.workers = 4;
    Service server(options), client(options);
    server.Start("127.0.0.1:0", [&](const Message &) { return *cache.Find("wire", {1, 2, 3, 4}, 4).state; });
    std::vector<std::future<Message>> calls;
    for (int i = 0; i < 16; ++i)
        calls.push_back(client.Submit(server.Address(), {}));
    for (auto &call : calls)
        Equal(expected, call.get());
    server.Stop();
    std::vector<std::future<void>> workers;
    for (int t = 0; t < 4; ++t)
        workers.push_back(std::async(std::launch::async, [&, t] {
            std::vector<int> tokens{100 + t};
            for (int i = 0; i < 64; ++i) {
                tokens.push_back(i);
                cache.Publish("threads", tokens, expected);
                Equal(expected, *cache.Find("threads", tokens, tokens.size()).state);
            }
        }));
    for (auto &worker : workers)
        worker.get();
}
static void BranchOracle() {
    struct Record {
        std::string name;
        std::vector<int> tokens;
        Message value;
    };
    PrefixCache cache(16 << 20);
    std::vector<Record> records;
    std::mt19937 random(41);
    for (int step = 0; step < 600; ++step) {
        std::string name = "branch/" + std::to_string(random() % 3);
        std::vector<int> tokens{1, 2, 3};
        if (!records.empty())
            tokens = records[random() % records.size()].tokens;
        if (random() % 2)
            tokens.resize(1 + random() % tokens.size());
        tokens.push_back(int(random() % 8));
        const size_t limit = random() % (tokens.size() + 1);
        const Record *best = nullptr;
        for (const auto &record : records)
            if (record.name == name && record.tokens.size() <= limit &&
                (!best || record.tokens.size() > best->tokens.size()) &&
                std::equal(record.tokens.begin(), record.tokens.end(), tokens.begin()))
                best = &record;
        auto hit = cache.Find(name, tokens, limit);
        Check(hit.tokens == (best ? best->tokens.size() : 0), "branch reference longest prefix");
        if (best)
            Equal(best->value, *hit.state);
        const bool duplicate = std::any_of(records.begin(), records.end(), [&](const auto &r) {
            return r.name == name && r.tokens == tokens;
        });
        Message value{{uint8_t(step), uint8_t(step >> 8)}, {Payload(73, step + 1)}};
        cache.Publish(name, tokens, value);
        if (!duplicate)
            records.push_back({name, tokens, value});
    }
}
int main(int argc, char **argv) {
    try {
        const bool cuda = argc > 1 && std::string(argv[1]) == "--cuda";
        BoundariesAndLeases(-1);
        if (cuda)
            BoundariesAndLeases(0);
        GrowingPages();
        BytePressureAndCollisions();
        MultiSession();
        ConcurrentWire();
        BranchOracle();
        std::cout << "Prefix cache: immutable pages, exact boundaries, leases, growth, branches, concurrency "
                     "and wire PASS\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
