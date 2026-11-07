#include "fastkv/fastkv.h"
#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>
using namespace fastkv;
static volatile std::sig_atomic_t stopped = 0;
static void Stop(int) { stopped = 1; }
static void Verify(const Buffer &b, uint8_t seed) {
    std::vector<uint8_t> host(std::min(size_t(1 << 20), b.Size()));
    for (size_t at = 0; at < b.Size();) {
        size_t n = std::min(host.size(), b.Size() - at);
        b.Read(host.data(), at, n);
        for (size_t i = 0; i < n; ++i)
            if (host[i] != uint8_t((i + at) * 131 + seed))
                throw std::runtime_error("payload mismatch at " + std::to_string(at + i));
        at += n;
    }
}
static Buffer Payload(size_t size, int device, int seed) {
    auto buffer = Buffer::Allocate(size, device);
    std::vector<uint8_t> host(std::min(size, size_t(1 << 20)));
    for (size_t at = 0; at < size;) {
        size_t n = std::min(host.size(), size - at);
        for (size_t i = 0; i < n; ++i) host[i] = uint8_t((at + i) * 131 + seed);
        buffer.Write(host.data(), at, n);
        at += n;
    }
    return buffer;
}
int main(int argc, char **argv) {
    try {
        if (argc < 3)
            throw std::runtime_error("usage: fastkv_probe serve|call|read host:port [--transport "
                                     "tcp|rdma|auto] "
                                     "[--device N] [--bytes N] [--buffers N] [--count N] [--seed N] "
                                     "[--workers N] [--concurrency N] [--warmup N] [--cancel-after-ms N] [--rdma-devices mlx5_0,...]");
        Options o;
        o.timeoutMs = 30000;
        size_t size = 16 << 20;
        int count = 3, buffers = 1, seed = 17, cancelMs = 0, concurrency = 1, warmup = 0;
        for (int i = 3; i < argc; i += 2) {
            if (i + 1 == argc)
                throw std::runtime_error("missing argument");
            std::string k = argv[i], v = argv[i + 1];
            if (k == "--transport") {
                if (v == "tcp")
                    o.transport = Transport::Tcp;
                else if (v == "rdma")
                    o.transport = Transport::Rdma;
                else if (v == "auto")
                    o.transport = Transport::Auto;
                else
                    throw std::runtime_error("transport");
            } else if (k == "--device")
                o.receiveDevice = std::stoi(v);
            else if (k == "--bytes")
                size = std::stoull(v);
            else if (k == "--count")
                count = std::stoi(v);
            else if (k == "--buffers")
                buffers = std::stoi(v);
            else if (k == "--workers")
                o.workers = std::stoi(v);
            else if (k == "--concurrency")
                concurrency = std::stoi(v);
            else if (k == "--warmup")
                warmup = std::stoi(v);
            else if (k == "--seed")
                seed = std::stoi(v);
            else if (k == "--cancel-after-ms")
                cancelMs = std::stoi(v);
            else if (k == "--rdma-devices") {
                size_t start = 0;
                for (;;) {
                    auto end = v.find(',', start);
                    o.rdmaDevices.push_back(v.substr(start, end - start));
                    if (end == std::string::npos)
                        break;
                    start = end + 1;
                }
            } else
                throw std::runtime_error("unknown option " + k);
        }
        if (count < 1 || buffers < 1 || buffers > int(o.maxBuffers) || size > o.maxPayloadBytes / buffers ||
            seed < 0 || seed > 255 || cancelMs < 0 || o.workers < 1 || o.workers > 64 ||
            concurrency < 1 || concurrency > o.queueDepth + o.workers || warmup < 0)
            throw std::runtime_error("invalid count, buffer count or payload size");
        const bool read = std::string(argv[1]) == "read";
        Service service(o);
        if (std::string(argv[1]) == "serve") {
            std::signal(SIGINT, Stop);
            std::signal(SIGTERM, Stop);
            auto payload = std::make_shared<Buffer>(Payload(size, o.receiveDevice, seed));
            service.Start(argv[2], [payload, buffers, seed](const Message &m) {
                if (m.metadata.size() != 3)
                    throw std::runtime_error("invalid diagnostic metadata");
                if (m.metadata[0] == 2) {
                    if (!m.buffers.empty() || m.metadata[2] != seed)
                        throw std::runtime_error("invalid read diagnostic request");
                    // Fresh allocation identity for each reply, as with a new
                    // model snapshot. The immutable backing data is prepared
                    // outside the transfer timing; no source MR can be reused.
                    auto source = Buffer::Borrow(payload->MutableData(), payload->Size(),
                        payload->Kind(), payload->Device(), payload);
                    return Message{m.metadata, std::vector<Buffer>(buffers, source.ReadOnly())};
                }
                // Echo without CPU readback; the caller verifies every byte
                // after the timed interval, covering both transfer directions.
                return m;
            });
            std::cout << "Fast KV Manager listening " << service.Address() << std::endl;
            while (!stopped)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            service.Stop();
        } else if (std::string(argv[1]) == "call" || read) {
            auto buffer = read ? Buffer{} : Payload(size, o.receiveDevice, seed);
            Message request;
            request.metadata = {uint8_t(read ? 2 : 1), 2, uint8_t(seed)};
            if (!read) request.buffers.assign(buffers, buffer);
            if (!read && buffers > 1)
                request = Snapshot(request);
            auto verifyReply = [&](const Message &reply) {
                if (reply.metadata != request.metadata || reply.buffers.size() != size_t(buffers))
                    throw std::runtime_error("reply shape mismatch");
                for (const auto &part : reply.buffers) {
                    if (part.Size() != size)
                        throw std::runtime_error("reply buffer size mismatch");
                    Verify(part, uint8_t(seed));
                }
            };
            auto begin = std::chrono::steady_clock::now();
            if (cancelMs) {
                for (int i = 0; i < warmup; ++i)
                    verifyReply(service.Call(argv[2], request));
                auto pending = service.Submit(argv[2], request);
                std::this_thread::sleep_for(std::chrono::milliseconds(cancelMs));
                const auto stopping = std::chrono::steady_clock::now();
                service.Stop();
                bool cancelled = false;
                try {
                    (void)pending.get();
                } catch (const std::exception &) {
                    cancelled = true;
                }
                const double stopMs =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stopping)
                        .count();
                std::cout << "{\"cancelled\":" << (cancelled ? "true" : "false")
                          << ",\"requested_delay_ms\":" << cancelMs << ",\"stop_ms\":" << stopMs << "}\n";
                return cancelled ? 0 : 2;
            }
            double seconds = 0;
            for (int i = -warmup; i < count;) {
                const int batch = std::min(concurrency, i < 0 ? -i : count - i);
                const auto transferBegin = std::chrono::steady_clock::now();
                std::vector<std::future<Message>> pending;
                for (int j = 0; j < batch; ++j)
                    pending.push_back(service.Submit(argv[2], request));
                std::vector<Message> replies;
                for (auto &future : pending) replies.push_back(future.get());
                if (i >= 0) seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - transferBegin).count();
                i += batch;
                for (const auto &reply : replies) verifyReply(reply);
            }
            const double wallSeconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
            auto stats = service.Stats();
            std::cout << "{\"correct\":true,\"bytes\":" << size * buffers << ",\"buffers\":" << buffers
                      << ",\"mode\":\"" << (read ? "read" : "echo") << "\""
                      << ",\"workers\":" << o.workers << ",\"concurrency\":" << concurrency
                      << ",\"warmup_calls\":" << warmup << ",\"timed_calls\":" << count
                      << ",\"connections\":" << stats.connections << ",\"reused_connections\":" << stats.reusedConnections
                      << ",\"calls\":" << stats.calls << ",\"tcp_bytes\":" << stats.tcpBytes
                      << ",\"rdma_bytes\":" << stats.rdmaBytes << ",\"seconds\":" << seconds
                      << ",\"wall_seconds\":" << wallSeconds
                      << ",\"aggregate_GBps\":" << (read ? 1.0 : 2.0) * size * buffers * count / seconds / 1e9
                      << ",\"amortized_call_ms\":" << seconds * 1000 / count << "}\n";
        } else
            throw std::runtime_error("expected serve, call or read");
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
