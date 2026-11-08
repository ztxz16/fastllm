#include "fastkv/fastkv.h"
#include <iostream>
#include <thread>
using namespace fastkv;
using namespace std::chrono_literals;
static void Check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
template<class F> static void Until(F f) {
    auto end = std::chrono::steady_clock::now() + 3s;
    while (!f()) {
        Check(std::chrono::steady_clock::now() < end, "cancellation stalled");
        std::this_thread::sleep_for(1ms);
    }
}
static void Cancelled(std::future<Message> &result) {
    Check(result.wait_for(3s) == std::future_status::ready, "cancel did not complete future");
    bool failed = false;
    try { (void)result.get(); } catch (const std::exception &) { failed = true; }
    Check(failed, "cancelled operation returned a payload");
}
int main(int argc, char **argv) {
    try {
        Options o; o.workers = 1; o.queueDepth = 1; o.maxActiveRequests = 1;
        if (argc > 1) {
            Check(std::string(argv[1]) == "--rdma", "usage: fastkv_cancel_test [--rdma [device]]");
            o.transport = Transport::Rdma;
        }
        if (argc > 2) o.receiveDevice = std::stoi(argv[2]);
        Service server(o), client(o);
        std::promise<Message> held;
        std::shared_ptr<Cancellation> producerCancel;
        std::atomic<int> entered{0};
        server.StartAsync("127.0.0.1:0", [&](Message m, std::shared_ptr<Cancellation> cancel) {
            if (m.metadata == Bytes{1}) {
                producerCancel = std::move(cancel);
                ++entered;
                return held.get_future();
            }
            ++entered;
            std::promise<Message> ready; ready.set_value(std::move(m)); return ready.get_future();
        });
        Budget modelSlots(2, "test model slots");
        auto lease = modelSlots.Reserve(1);
        auto cancellation = std::make_shared<Cancellation>();
        Message first{{1}, {}};
        std::future<Message> active;
        Check(client.TrySubmit(server.Address(), first, active, cancellation, lease), "first admission");
        lease.reset();
        Until([&] { return entered == 1; });
        Check(modelSlots.Used() == 1, "dropped caller released in-flight resources");

        auto queuedCancel = std::make_shared<Cancellation>();
        Message second{{2}, {}};
        std::future<Message> queued;
        Check(client.TrySubmit(server.Address(), second, queued, queuedCancel), "queued admission");
        auto payload = Buffer::Allocate(4096, o.receiveDevice);
        std::vector<uint8_t> expected(4096, 37), actual(4096);
        payload.Write(expected.data(), 0, expected.size());
        Message retry{{3}, {payload}};
        std::future<Message> retried;
        Check(!client.TrySubmit(server.Address(), retry, retried), "full client queue must backpressure");
        Check(retry.metadata == Bytes{3} && retry.buffers[0].Id() == payload.Id() && !retried.valid(),
              "busy submission consumed its request or future");
        queuedCancel->Cancel();
        Cancelled(queued);
        cancellation->Cancel();
        Cancelled(active);
        Until([&] { return producerCancel->Cancelled() && client.Stats().clientRequests == 0; });
        Until([&] { return modelSlots.Used() == 0; });
        Check(entered == 1 && server.Stats().activeHandlers == 1,
              "cancelled queued work ran, or producer admission was released before drain");
        held.set_value({});
        Until([&] { return server.Stats().activeHandlers == 0; });
        auto retryCancel = std::make_shared<Cancellation>();
        Check(client.TrySubmit(server.Address(), retry, retried, retryCancel), "retry after cancellation");
        auto reply = retried.get();
        Check(reply.metadata == Bytes{3} && reply.buffers.size() == 1, "retry lost request");
        reply.buffers[0].Read(actual.data(), 0, actual.size());
        Check(actual == expected, "retry payload corrupted");
        if (o.transport == Transport::Rdma) Check(client.Stats().rdmaBytes >= 8192, "RDMA payload path unused");
        // A late cancellation must not cancel a reused connection's next RPC.
        retryCancel->Cancel();
        Check(client.Call(server.Address(), {{4}, {}}).metadata == Bytes{4}, "connection reuse after cancel");
        Check(client.Stats().admissionWaits == 1 && client.Stats().cancelledRequests == 2,
              "admission/cancellation metrics");
        client.Stop(); server.Stop();

        // Cancel while waiting for receive memory, including behind the
        // complete-message allocation lock; retained replies remain valid.
        Options bounded; bounded.workers = 2; bounded.maxInflightBytes = 4096;
        bounded.transport = o.transport; bounded.receiveDevice = o.receiveDevice;
        bounded.deviceBufferBytes[o.receiveDevice] = 4096;
        Options sourceOptions = o; sourceOptions.maxActiveRequests = 3;
        Service source(sourceOptions), receiver(bounded);
        source.Start("127.0.0.1:0", [payload](const Message &) { return Message{{9}, {payload}}; });
        auto pinned = receiver.Call(source.Address(), {});
        std::future<Message> waiting[2];
        std::shared_ptr<Cancellation> signals[2];
        for (int i = 0; i < 2; ++i) {
            signals[i] = std::make_shared<Cancellation>(); Message request;
            Check(receiver.TrySubmit(source.Address(), request, waiting[i], signals[i]), "memory wait admission");
        }
        std::this_thread::sleep_for(30ms);
        for (int i = 0; i < 2; ++i) { signals[i]->Cancel(); Cancelled(waiting[i]); }
        Until([&] { return receiver.Stats().clientRequests == 0; });
        pinned.buffers[0].Read(actual.data(), 0, actual.size());
        Check(actual == expected, "cancel invalidated retained data");
        pinned = {};
        Check(receiver.Call(source.Address(), {}).buffers.size() == 1, "memory credits lost after cancellation");
        receiver.Stop(); source.Stop();
        std::cout << "FastKV cancellation, bounded admission, request preservation and safe drain PASS\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
