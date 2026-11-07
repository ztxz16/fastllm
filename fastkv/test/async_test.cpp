#include "fastkv/fastkv.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
using namespace fastkv;
using namespace std::chrono_literals;
static void Check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}
template<class F> static void Fails(F &&f) {
    bool failed = false;
    try { f(); } catch (const std::exception &) { failed = true; }
    Check(failed, "expected failure");
}
template<class F> static void Until(F &&f) {
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (!f()) {
        Check(std::chrono::steady_clock::now() < deadline, "asynchronous progress stalled");
        std::this_thread::sleep_for(1ms);
    }
}
static std::future<Message> Ready(Message message) {
    std::promise<Message> result;
    result.set_value(std::move(message));
    return result.get_future();
}
int main() {
    try {
        Options options;
        options.workers = 1;
        options.maxActiveRequests = 4;
        options.queueDepth = 16;
        options.timeoutMs = 3000;
        Service slow(options), fast(options), client(options);
        std::promise<Message> held[3];
        std::atomic<int> entered{0};
        slow.StartAsync("127.0.0.1:0", [&](Message request) {
            if (!request.metadata.empty() && request.metadata[0] < 3) {
                auto result = held[request.metadata[0]].get_future();
                ++entered;
                return result;
            }
            if (request.metadata == Bytes{9}) {
                std::promise<Message> failure;
                failure.set_exception(std::make_exception_ptr(std::runtime_error("async error")));
                return failure.get_future();
            }
            return Ready(std::move(request));
        });
        fast.Start("127.0.0.1:0", [](const Message &m) { return m; });
        std::future<Message> pending[3];
        for (int i = 0; i < 3; ++i)
            pending[i] = client.Submit(slow.Address(), {Bytes{uint8_t(i)}, {}});
        Until([&] { return entered == 3; });
        // More computations than workers, with progress both to the same P
        // and another P. None of these checks release a held computation.
        auto samePeer = client.Submit(slow.Address(), {Bytes{7}, {}});
        auto otherPeer = client.Submit(fast.Address(), {Bytes{8}, {}});
        Check(samePeer.wait_for(1s) == std::future_status::ready, "same peer starved");
        Check(otherPeer.wait_for(1s) == std::future_status::ready, "slow peer blocked another peer");
        Check(samePeer.get().metadata == Bytes{7} && otherPeer.get().metadata == Bytes{8}, "wrong reply");
        Fails([&] { client.Call(slow.Address(), {Bytes{9}, {}}); });
        for (int i = 0; i < 3; ++i) held[i].set_value({Bytes{uint8_t(i)}, {}});
        for (int i = 0; i < 3; ++i) Check(pending[i].get().metadata == Bytes{uint8_t(i)}, "mixed results");
        for (int round = 0; round < 8; ++round) {
            std::vector<std::future<Message>> calls;
            for (int i = 0; i < 8; ++i) {
                auto payload = Buffer::Allocate(65536 + i);
                std::memset(payload.MutableData(), round * 8 + i, payload.Size());
                calls.push_back(client.Submit(slow.Address(), {Bytes{10}, {payload}}));
            }
            for (int i = 0; i < 8; ++i) {
                auto reply = calls[i].get();
                Check(reply.buffers.size() == 1 && reply.buffers[0].Size() == size_t(65536 + i), "payload shape");
                const auto *bytes = static_cast<const uint8_t *>(reply.buffers[0].Data());
                for (size_t n = 0; n < reply.buffers[0].Size(); ++n)
                    Check(bytes[n] == uint8_t(round * 8 + i), "payload cross-contamination");
            }
        }
        // The component can also act as a client and a server at once.
        std::vector<std::future<Message>> bidirectional;
        for (int i = 0; i < 8; ++i) {
            bidirectional.push_back(slow.Submit(fast.Address(), {Bytes{10}, {}}));
            bidirectional.push_back(fast.Submit(slow.Address(), {Bytes{10}, {}}));
        }
        for (auto &reply : bidirectional) Check(reply.get().metadata == Bytes{10}, "duplex reply");
        slow.Stop(); fast.Stop(); client.Stop();

        // A timed out producer still consumes its admission credit until it
        // finishes. Repeated timeouts cannot accumulate unlimited model work.
        Options shortOptions = options;
        shortOptions.maxActiveRequests = 1;
        shortOptions.timeoutMs = 100;
        Service bounded(shortOptions), caller(options);
        std::promise<Message> abandoned;
        std::atomic<int> admitted{0};
        bounded.StartAsync("127.0.0.1:0", [&](Message request) {
            ++admitted;
            if (request.metadata == Bytes{1}) return abandoned.get_future();
            return Ready(std::move(request));
        });
        auto expired = caller.Submit(bounded.Address(), {Bytes{1}, {}});
        Until([&] { return admitted == 1; });
        Fails([&] { (void)expired.get(); });
        Fails([&] { caller.Call(bounded.Address(), {}); });
        Check(admitted == 1, "timed out producer released its admission credit early");
        abandoned.set_value({});
        Check(caller.Call(bounded.Address(), {}).metadata.empty(), "credit not returned after drain");
        Check(admitted == 2, "admission failed after drain");
        bounded.Stop();

        // Stop has no dependency on unresolved producer promises.
        Service stopping(options);
        std::promise<Message> unresolved;
        std::atomic<bool> started{false};
        stopping.StartAsync("127.0.0.1:0", [&](Message) {
            started = true;
            return unresolved.get_future();
        });
        auto unfinished = caller.Submit(stopping.Address(), {});
        Until([&] { return started.load(); });
        const auto begin = std::chrono::steady_clock::now();
        stopping.Stop();
        Check(std::chrono::steady_clock::now() - begin < 1s, "Stop waited for an async result");
        Fails([&] { (void)unfinished.get(); });
        unresolved.set_value({});
        caller.Stop();
        std::cout << "FastKV async isolation, bounded admission, payloads, deadline drain and shutdown PASS\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
