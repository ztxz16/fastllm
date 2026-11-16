#include "fastkv/fastkv.h"
#include <chrono>
#include <iostream>
#include <thread>
using namespace fastkv;
using namespace std::chrono_literals;
static void Check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
int main(int argc, char **argv) {
    try {
        Options o;
        if (argc > 1) { Check(std::string(argv[1]) == "--rdma", "usage: receive_target_test [--rdma [GPU]]"); o.transport = Transport::Rdma; }
        const int gpu = argc > 2 ? std::stoi(argv[2]) : -1;
        auto input = Buffer::Allocate(123457, gpu);
        std::vector<unsigned char> bytes(input.Size()), actual(bytes.size());
        for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = (i * 71 + i / 97) % 251;
        input.Write(bytes.data(), 0, bytes.size());
        Service server(o), client(o);
        server.Start("127.0.0.1:0", [input](const Message &request) {
            if (request.metadata == Bytes{0}) return Message{{19}, {input.Slice(0,0), input.Slice(0,0)}};
            return Message{{19}, {input, input.Slice(11, 4099)}};
        });
        auto host = Buffer::Allocate(input.Size() + 16384);
        auto device = Buffer::Allocate(4099, gpu);
        Budget pages(1); auto lease = pages.Reserve(1);
        std::weak_ptr<void> weak = lease;
        std::vector<std::vector<Buffer>> spans{{host.Slice(8192, 7), host.Slice(0, 4093), host.Slice(12288, input.Size()-4100)}, {device}};
        // The host spans intentionally reorder disjoint physical ranges.
        for (auto &group : spans) for (auto &b : group) b = b.WithLease(lease);
        ReceiveTarget target = [spans](const Bytes &metadata, const std::vector<size_t> &sizes) {
            Check(metadata == Bytes{19} && sizes == std::vector<size_t>{123457,4099}, "wire descriptor"); return spans;
        };
        Message request; std::future<Message> future;
        Check(client.TrySubmit(server.Address(), request, future, {}, {}, target), "submit");
        auto reply = future.get();
        Check(reply.buffers.size() == 4, "target result must preserve spans");
        size_t at = 0;
        for (int i = 0; i < 3; ++i) { reply.buffers[i].Read(actual.data()+at,0,reply.buffers[i].Size()); at += reply.buffers[i].Size(); }
        Check(actual == bytes, "scatter receive bytes differ");
        actual.resize(4099); reply.buffers[3].Read(actual.data(),0,4099);
        Check(std::equal(actual.begin(),actual.end(),bytes.begin()+11), "mixed target bytes differ");
        lease.reset(); spans.clear(); target = {};
        // Reject bad target lengths before any write, then reuse the service.
        Message bad; std::future<Message> failed;
        Check(client.TrySubmit(server.Address(),bad,failed,{}, {},[host](const Bytes &, const std::vector<size_t> &) {
            return std::vector<std::vector<Buffer>>{{host.Slice(0,1)}, {host.Slice(2,1)}};
        }), "bad submit");
        bool rejected = false; try { (void)failed.get(); } catch (const std::exception &) { rejected = true; }
        Check(rejected, "invalid target was accepted");
        Check(client.Call(server.Address(),{}).buffers.size() == 2, "default receive regressed");
        // Per-request maps must survive asynchronous submission independently
        // of the service default and other in-flight requests.
        std::atomic<int> placements{0};
        std::vector<std::future<Message>> placed(8);
        for (int i = 0; i < 8; ++i) {
            Message request;
            Check(client.TrySubmit(server.Address(), request, placed[i], {}, {}, {},
                [&, destination = i % 2 ? gpu : -1](const Bytes &metadata, const std::vector<size_t> &sizes) {
                    Check(metadata == Bytes{19} && sizes.size() == 2, "per-call placement metadata");
                    ++placements; return std::vector<int>(sizes.size(), destination);
                }), "per-call placement submit");
        }
        for (int i = 0; i < 8; ++i) {
            auto value = placed[i].get();
            Check(value.buffers[0].Device() == (i % 2 ? gpu : -1), "per-call placement device");
            std::vector<unsigned char> got(bytes.size()); value.buffers[0].Read(got.data(), 0, got.size());
            Check(got == bytes, "per-call placement bytes");
        }
        Check(placements == 8, "per-call placement lost or reused");
        reply = {};
        Message empty{{0}, {}}; std::future<Message> emptyResult;
        Check(client.TrySubmit(server.Address(),empty,emptyResult,{}, {},[](const Bytes &,const std::vector<size_t> &sizes) {
            Check(sizes == std::vector<size_t>{0,0}, "empty wire buffers");
            return std::vector<std::vector<Buffer>>(2);
        }), "empty target submit");
        Check(emptyResult.get().buffers.empty(), "empty targets must contain no spans");
        // Cancellation after target reservation must drain before the caller
        // can reuse the pages, even if it drops its future/receiver capture.
        std::promise<Message> delayed;
        std::atomic<bool> entered{false};
        Service slow(o);
        slow.StartAsync("127.0.0.1:0", [&](Message) { entered = true; return delayed.get_future(); });
        Budget cancelledPages(1); auto reserved = cancelledPages.Reserve(1);
        auto signal = std::make_shared<Cancellation>();
        Message cancelRequest; std::future<Message> cancelled;
        Check(client.TrySubmit(slow.Address(),cancelRequest,cancelled,signal,{},
            [reserved,host](const Bytes &,const std::vector<size_t> &) {
                return std::vector<std::vector<Buffer>>{{host.WithLease(reserved)}};
            }), "cancel submit");
        reserved.reset();
        for (int i = 0; !entered && i < 3000; ++i) std::this_thread::sleep_for(1ms);
        Check(entered && cancelledPages.Used() == 1, "reservation not retained while waiting");
        signal->Cancel();
        Check(cancelled.wait_for(3s) == std::future_status::ready, "target cancellation stalled");
        rejected = false; try { (void)cancelled.get(); } catch (const std::exception &) { rejected = true; }
        Check(rejected, "cancelled target succeeded");
        delayed.set_value({}); slow.Stop();
        for (int i = 0; cancelledPages.Used() && i < 3000; ++i) std::this_thread::sleep_for(1ms);
        Check(cancelledPages.Used() == 0, "cancelled reservation leaked");
        client.Stop(); server.Stop();
        Check(weak.expired() && pages.Used() == 0, "target lease leaked");
        std::cout << "FastKV receive targets: fragmented/mixed placement, validation, leases and fallback PASS\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
