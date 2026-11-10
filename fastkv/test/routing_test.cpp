#include "fastkv/fastkv.h"
#include <iostream>
#include <thread>
#include <limits>
using namespace fastkv;
using namespace std::chrono_literals;
static void Check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
int main() {
    try {
        PrefillRouter router(2);
        std::vector<PrefillRouter::Estimate> quotes(2);
        std::vector<std::shared_ptr<void>> held;
        for (int i = 0; i < 128; ++i) held.push_back(router.Reserve(1000, quotes).lease);
        Check(router.Pending() == std::vector<size_t>({64000,64000}), "stale telemetry caused a routing herd");
        held.clear();
        Check(router.Pending() == std::vector<size_t>({0,0}), "route cancellation leaked work");
        quotes[0].cachedTokens = 900;
        Check(router.Reserve(1000, quotes).peer == 0, "idle warm cache was ignored");
        quotes[0].pendingTokens = 10000;
        Check(router.Reserve(1000, quotes).peer == 1, "affinity stranded work behind a busy producer");
        quotes[0] = {}; quotes[1] = {};
        quotes[0].hostCachedTokens = 9900; quotes[0].restoreSeconds = .02;
        Check(router.Reserve(10000, quotes).peer == 0, "L2 hit treated as a full recompute");
        quotes[1].cachedTokens = 9900;
        Check(router.Reserve(10000, quotes).peer == 1, "L1 ignored restore cost");
        quotes[1] = {}; quotes[0].pendingRestoreSeconds = 10;
        Check(router.Reserve(10000, quotes).peer == 1, "L2 transfer queue ignored");
        quotes[0] = {}; quotes[1] = {};
        quotes[0].cachedTokens = 9000;
        quotes[0].pendingTokens = 11000;
        Check(router.Reserve(10000, quotes).peer == 0,
              "small queue advantage duplicated a large resident prefix");
        quotes[0].pendingTokens = 110000;
        Check(router.Reserve(10000, quotes).peer == 1,
              "cache protection blocked a useful spill from an overloaded producer");
        quotes[0] = {}; quotes[1].tokensPerSecond = 20000;
        int fast = 0;
        for (int i = 0; i < 120; ++i) {
            auto route = router.Reserve(1000, quotes); fast += route.peer == 1;
            held.push_back(std::move(route.lease));
        }
        Check(fast >= 79 && fast <= 81, "heterogeneous rates did not distribute proportional work");
        held.clear();
        quotes[0].available = quotes[1].available = false;
        bool busy = false;
        try { (void)router.Reserve(1000, quotes); } catch (const CapacityBusy &) { busy = true; }
        Check(busy, "no available producer was silently selected");

        quotes[0] = {}; quotes[1] = {};
        for (int i = 0; i < 10; ++i)
            Check(router.Reserve(10000, quotes, 1).peer == 1, "stable session ownership was ignored");
        quotes[1].pendingTokens = 100000;
        Check(router.Reserve(10000, quotes, 1).peer == 0, "session ownership prevented overload spill");
        quotes[1] = {}; quotes[0].cachedTokens = 9900;
        Check(router.Reserve(10000, quotes, 1).peer == 0, "cold ownership displaced an exact warm prefix");
        quotes[1].available = false;
        Check(router.Reserve(10000, quotes, 1).peer == 0, "unavailable owner blocked routing");

        quotes[0] = {}; quotes[1] = {}; quotes[1].tokensPerSecond = 0;
        Check(router.Reserve(10000, quotes, 1).peer == 0, "zero-rate owner was selected");
        quotes[1].tokensPerSecond = std::numeric_limits<double>::quiet_NaN();
        Check(router.Reserve(10000, quotes, 1).peer == 0, "invalid-rate owner was selected");

        quotes[0] = {}; quotes[1] = {};
        quotes[0].cachedTokens = quotes[1].cachedTokens = 9900;
        quotes[1].pendingTokens = 10000;
        for (int i = 0; i < 32; ++i)
            Check(router.Reserve(10000, quotes, 1, true).peer == 1,
                  "one prefill batch's imbalance moved a committed session");
        quotes[1].pendingTokens = 100000;
        Check(router.Reserve(10000, quotes, 1, true).peer == 1, "transient overload moved a session");
        PrefillRouter longSessions(2);
        auto longQuotes = quotes; longQuotes[1].pendingTokens = 2000000;
        Check(longSessions.Reserve(100000, longQuotes, 1, true).peer == 1,
              "long-prefix migration ignored rebuild time");
        std::this_thread::sleep_for(1050ms);
        Check(longSessions.Reserve(100000, longQuotes, 1, true).peer == 1,
              "one batch wave moved a long session before its rebuild interval");
        Check(router.Reserve(10000, quotes, 1, true).peer == 0, "persistent overload cannot migrate");
        quotes[1].available = false;
        Check(router.Reserve(10000, quotes, 1, true).peer == 0, "failed owner cannot fail over");
        quotes[1] = {};
        Check(router.Reserve(10000, quotes, 1, true).peer == 0,
              "lost owner prefix forced recomputation despite a warm producer");

        SessionDirectory directory(2);
        Check(directory.Update("a", {1,"p0"}), "first placement missing");
        Check(directory.Update("a", {2,"p1"}), "migration missing");
        Check(!directory.Update("a", {1,"p0"}) && directory.Get("a").version == 2,
              "delayed announcement reversed committed ownership");
        Check(!directory.Update("a", {2,"p0"}), "concurrent ownership tie did not converge");
        Check(directory.Get("a").owner == "p1", "placement query changed the owner");
        directory.Update("b", {1,"p0"}); directory.Get("a"); directory.Update("c", {1,"p0"});
        Check(!directory.Get("b").version && directory.Get("a").version == 2,
              "session directory exceeded capacity or evicted the wrong entry");

        PrefixCache cache(4096, 1);
        cache.Publish("model", {1,2,3}, {{9}, {Buffer::Allocate(128)}});
        auto stats = cache.Stats();
        Check(cache.ProbeTokens("model", {1,2,3,4}) == 3, "live prefix probe missed");
        Check(cache.Stats().lookups == stats.lookups && cache.Stats().hits == stats.hits,
              "routing query inflated cache metrics");
        cache.Publish("model", {5,6,7}, {{8}, {}});
        Check(cache.ProbeTokens("model", {1,2,3,4}) == 0 && cache.Stats().evictions == 1,
              "probe pinned or resurrected an evicted prefix");

        Options options; options.maxActiveRequests = 1; options.workers = 1;
        Service server(options), client(options), control(options);
        std::promise<Message> compute;
        std::atomic<bool> entered{false};
        server.StartAsync("127.0.0.1:0", [&](Message, std::shared_ptr<Cancellation>) {
            entered = true; return compute.get_future();
        }, [](const Message &request, Message &reply) {
            if (request.metadata != Bytes{2}) return false;
            reply.metadata = {3}; return true;
        });
        auto result = client.Submit(server.Address(), {{1},{}});
        for (int i = 0; i < 3000 && !entered; ++i) std::this_thread::sleep_for(1ms);
        Check(entered, "producer did not enter");
        auto query = control.Submit(server.Address(), {{2},{}});
        Check(query.wait_for(1s) == std::future_status::ready && query.get().metadata == Bytes{3},
              "metadata query waited behind saturated producer admission");
        Check(server.Stats().activeHandlers == 1, "query consumed or released producer admission");
        compute.set_value({{4},{}});
        Check(result.get().metadata == Bytes{4}, "query changed the original computation");
        std::cout << "FastKV routing, cache residency and control admission PASS\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
