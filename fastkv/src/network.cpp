#include "internal.h"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iterator>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <set>
#include <sys/socket.h>
#include <unistd.h>

namespace fastkv {
static constexpr uint32_t Magic = 0x314b5646, Version = 2;
static constexpr uint32_t ReceiveTargetFormat = 0x31544753;
static constexpr size_t Chunk = 1 << 20;
static std::runtime_error SystemError(const char *s) {
    return std::runtime_error(std::string("FastKV ") + s + ": " + std::strerror(errno));
}
Socket::~Socket() {
    if (fd >= 0)
        ::close(fd);
}
Socket &Socket::operator=(Socket &&s) noexcept {
    if (fd >= 0)
        ::close(fd);
    fd = s.fd;
    stopped = s.stopped;
    cancellation = std::move(s.cancellation);
    s.fd = -1;
    return *this;
}
void Socket::CheckRunning(Deadline deadline) const {
    Require(!stopped || !stopped->load(), "FastKV: service stopped");
    Require(!cancellation || !cancellation->Cancelled(), "FastKV: request cancelled");
    Require(std::chrono::steady_clock::now() < deadline, "FastKV: transport deadline exceeded");
}
static void Wait(int fd, short events, Deadline deadline, const std::atomic<bool> *stopped,
                 const Cancellation *cancellation = nullptr) {
    for (;;) {
        Require(!stopped || !stopped->load(), "FastKV: service stopped");
        Require(!cancellation || !cancellation->Cancelled(), "FastKV: request cancelled");
        auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline -
                                                                          std::chrono::steady_clock::now())
                        .count();
        Require(left > 0, "FastKV: transport deadline exceeded");
        pollfd p{fd, events, 0};
        int n = poll(&p, 1, int(std::min<int64_t>(left, cancellation ? 50 : 1000)));
        if (n < 0 && errno == EINTR)
            continue;
        if (n < 0)
            throw SystemError("poll");
        if (!n)
            continue;
        Require(!(p.revents & POLLNVAL), "FastKV: invalid connection");
        if (p.revents & (events | POLLERR | POLLHUP))
            return;
    }
}
void Socket::Send(const void *data, size_t n, Deadline d) const {
    const char *p = static_cast<const char *>(data);
    while (n) {
        CheckRunning(d);
#ifdef MSG_NOSIGNAL
        ssize_t k = ::send(fd, p, n, MSG_NOSIGNAL);
#else
        ssize_t k = ::send(fd, p, n, 0);
#endif
        if (k < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            Wait(fd, POLLOUT, d, stopped, cancellation.get());
            continue;
        }
        if (k < 0 && errno == EINTR)
            continue;
        if (k < 0)
            throw SystemError("send");
        Require(k > 0, "FastKV: connection closed during send");
        p += k;
        n -= k;
        Require(std::chrono::steady_clock::now() < d, "FastKV: send deadline exceeded");
    }
}
void Socket::Receive(void *data, size_t n, Deadline d) const {
    char *p = static_cast<char *>(data);
    while (n) {
        CheckRunning(d);
        ssize_t k = ::recv(fd, p, n, 0);
        if (k < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            Wait(fd, POLLIN, d, stopped, cancellation.get());
            continue;
        }
        if (k < 0 && errno == EINTR)
            continue;
        if (k < 0)
            throw SystemError("receive");
        Require(k > 0, "FastKV: peer closed connection");
        p += k;
        n -= k;
        Require(std::chrono::steady_clock::now() < d, "FastKV: receive deadline exceeded");
    }
}
void Socket::SendFrame(const Bytes &b, Deadline d) const {
    Require(b.size() <= UINT32_MAX, "FastKV: frame overflow");
    Writer w;
    w.U32(uint32_t(b.size()));
    Send(w.bytes.data(), 4, d);
    Send(b.data(), b.size(), d);
}
Bytes Socket::ReceiveFrame(size_t limit, Deadline d) const {
    uint8_t b[4];
    Receive(b, 4, d);
    Reader r(b, 4);
    size_t n = r.U32();
    Require(n <= limit, "FastKV: frame limit exceeded");
    Bytes out(n);
    Receive(out.data(), n, d);
    return out;
}
static void Configure(int fd) {
    if (fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK) < 0)
        throw SystemError("fcntl");
    fcntl(fd, F_SETFD, FD_CLOEXEC);
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#ifdef SO_NOSIGPIPE
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
#endif
}
static std::pair<std::string, std::string> SplitAddress(const std::string &s) {
    auto i = s.rfind(':');
    Require(i != std::string::npos && i + 1 < s.size(), "FastKV: endpoint must be host:port");
    auto host = s.substr(0, i);
    if (host.size() > 1 && host.front() == '[' && host.back() == ']')
        host = host.substr(1, host.size() - 2);
    return {host, s.substr(i + 1)};
}
static Socket Open(const std::string &endpoint, bool listen, Deadline deadline,
                   const std::atomic<bool> *stopped = nullptr,
                   std::shared_ptr<Cancellation> cancellation = {}) {
    auto parts = SplitAddress(endpoint);
    addrinfo hints{}, *raw = nullptr;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = listen ? AI_PASSIVE : 0;
    int result = getaddrinfo(parts.first.empty() ? nullptr : parts.first.c_str(), parts.second.c_str(),
                             &hints, &raw);
    if (result)
        throw std::runtime_error(std::string("FastKV resolve: ") + gai_strerror(result));
    std::unique_ptr<addrinfo, decltype(&freeaddrinfo)> addresses(raw, freeaddrinfo);
    std::string error;
    for (auto *a = raw; a; a = a->ai_next) {
        try {
            Socket socket(::socket(a->ai_family, a->ai_socktype, a->ai_protocol), stopped);
            socket.cancellation = cancellation;
            socket.CheckRunning(deadline);
            if (socket.fd < 0)
                throw SystemError("socket");
            Configure(socket.fd);
            if (listen) {
                int one = 1;
                setsockopt(socket.fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
                if (bind(socket.fd, a->ai_addr, a->ai_addrlen) < 0 || ::listen(socket.fd, 128) < 0)
                    throw SystemError("listen");
            } else if (connect(socket.fd, a->ai_addr, a->ai_addrlen) < 0) {
                if (errno != EINPROGRESS)
                    throw SystemError("connect");
                Wait(socket.fd, POLLOUT, deadline, stopped, cancellation.get());
                int status = 0;
                socklen_t len = sizeof(status);
                if (getsockopt(socket.fd, SOL_SOCKET, SO_ERROR, &status, &len) < 0)
                    throw SystemError("connect status");
                if (status) {
                    errno = status;
                    throw SystemError("connect");
                }
            }
            return socket;
        } catch (const std::exception &e) {
            error = e.what();
        }
    }
    throw std::runtime_error(error.empty() ? "FastKV: no usable endpoint" : error);
}
static std::string BoundAddress(const Socket &s) {
    sockaddr_storage addr{};
    socklen_t len = sizeof(addr);
    if (getsockname(s.fd, reinterpret_cast<sockaddr *>(&addr), &len) < 0)
        throw SystemError("getsockname");
    char host[NI_MAXHOST], port[NI_MAXSERV];
    int r = getnameinfo(reinterpret_cast<sockaddr *>(&addr), len, host, sizeof(host), port, sizeof(port),
                        NI_NUMERICHOST | NI_NUMERICSERV);
    Require(!r, "FastKV: invalid bound endpoint");
    return addr.ss_family == AF_INET6 ? "[" + std::string(host) + "]:" + port
                                      : std::string(host) + ":" + port;
}
struct LegacyHandshake : std::runtime_error { using std::runtime_error::runtime_error; };
static std::unique_ptr<RdmaConnection> Handshake(const Socket &s, const Options &o, bool server,
                                                 Deadline d, bool &reusable, bool legacy = false) {
    Transport chosen = Transport::Tcp;
    if (server) {
        auto b = s.ReceiveFrame(64, d);
        Reader r(b);
        Require(r.U32() == Magic, "FastKV: incompatible protocol");
        const auto version = r.U32();
        Require(version == 1 || version == Version, "FastKV: incompatible protocol");
        reusable = version == Version;
        uint32_t mode = r.U32();
        r.End();
        Require(mode <= uint32_t(Transport::Auto), "FastKV: unknown transport");
        bool available = o.transport != Transport::Tcp && RdmaAvailable();
        chosen = (mode != uint32_t(Transport::Tcp) && available) ? Transport::Rdma : Transport::Tcp;
        Writer reply;
        reply.U32(Magic);
        reply.U32(version);
        bool compatible = (mode != uint32_t(Transport::Rdma) || available) &&
                          (o.transport != Transport::Rdma || mode != uint32_t(Transport::Tcp));
        reply.U32(compatible ? uint32_t(chosen) : UINT32_MAX);
        s.SendFrame(reply.bytes, d);
        Require(compatible, "FastKV: requested transport is unavailable");
    } else {
        Writer hello;
        hello.U32(Magic);
        hello.U32(legacy ? 1 : Version);
        const auto offered =
            o.transport == Transport::Auto && !RdmaAvailable() ? Transport::Tcp : o.transport;
        hello.U32(uint32_t(offered));
        s.SendFrame(hello.bytes, d);
        Bytes b;
        try { b = s.ReceiveFrame(64, d); }
        catch (const std::exception &e) {
            if (!legacy) throw LegacyHandshake(e.what());
            throw;
        }
        Reader r(b);
        Require(r.U32() == Magic, "FastKV: incompatible protocol");
        const auto version = r.U32();
        Require(version == 1 || version == Version, "FastKV: incompatible protocol");
        reusable = version == Version;
        uint32_t mode = r.U32();
        r.End();
        Require(mode <= uint32_t(Transport::Rdma), "FastKV: peer rejected transport");
        chosen = Transport(mode);
        Require(o.transport != Transport::Rdma || chosen == Transport::Rdma,
                "FastKV: unexpected transport downgrade");
    }
    return chosen == Transport::Rdma ? ConnectRdma(s, o, d) : nullptr;
}

struct Service::Impl {
    Options options;
    Budget budget, clientSlots, serverSlots, handlerSlots;
    BufferPool receiveBuffers, sendBuffers;
    std::timed_mutex receiveAllocationMutex;
    // Separate request and reply lanes: a new request handshake must never
    // wait behind a reply whose receiver is occupied sending that request.
    Pool servers, clients, sendReplies, receiveReplies;
    Socket listener;
    CancellableHandler handler;
    ControlHandler control;
    std::thread acceptor, progress;
    std::condition_variable wake;
    enum class Stage { Connect, ReceiveRequest, IdleRequest, Admit, Handler, Reply, Drain };
    struct Channel {
        Impl &owner;
        Socket socket;
        std::unique_ptr<RdmaConnection> rdma;
        std::shared_ptr<void> serverSlot;
        bool ready = false, reusable = false, tracked = false;
        explicit Channel(Impl &owner) : owner(owner) {}
        ~Channel() {
            if (tracked) {
                std::lock_guard<std::mutex> lock(owner.mutex);
                owner.sockets.erase(socket.fd);
            }
            // On failure, QP teardown precedes every buffer/credit release.
            rdma.reset();
        }
    };
    struct Exchange {
        std::shared_ptr<Channel> channel;
        std::shared_ptr<void> slot, handlerSlot;
        std::shared_ptr<void> lifetime;
        std::shared_ptr<Cancellation> cancellation = std::make_shared<Cancellation>();
        Message request;
        ReceiveTarget target;
        std::future<Message> result;
        std::promise<Message> completion;
        std::string peer;
        Deadline deadline;
        Stage stage;
        bool server;
        Exchange(Impl &owner, bool server, std::shared_ptr<void> slot = {})
            : slot(std::move(slot)),
              deadline(std::chrono::steady_clock::now() + std::chrono::milliseconds(owner.options.timeoutMs)),
              stage(server ? Stage::ReceiveRequest : Stage::Connect), server(server) {}
        ~Exchange() { channel.reset(); } // QP teardown before caller admission is released
    };
    std::multimap<std::string, std::shared_ptr<Channel>> idleClients;
    std::set<std::string> legacyPeers;
    std::vector<std::shared_ptr<Exchange>> pending;
    mutable std::mutex mutex;
    std::set<int> sockets;
    std::atomic<bool> stopped{false};
    std::atomic<uint64_t> tcpBytes{0}, rdmaBytes{0}, calls{0}, failures{0};
    std::atomic<uint64_t> connections{0}, reusedConnections{0};
    std::atomic<uint64_t> cancelledRequests{0}, admissionWaits{0};
    explicit Impl(Options o)
        : options(std::move(o)), budget(options.maxInflightBytes),
          clientSlots(size_t(options.queueDepth) + options.workers, "client request slots"),
          serverSlots(size_t(options.queueDepth) + ActiveLimit(options), "server connection slots"),
          handlerSlots(ActiveLimit(options), "handler slots"),
          receiveBuffers(options.maxInflightBytes, options.deviceBufferBytes),
          sendBuffers(std::min(options.maxInflightBytes, size_t(options.workers) * (size_t(512) << 20))),
          servers(options.workers, options.queueDepth), clients(options.workers, options.queueDepth),
          sendReplies(options.workers, options.queueDepth), receiveReplies(options.workers, options.queueDepth) {
        Require(options.timeoutMs > 0 && options.maxPayloadBytes && options.maxMetadataBytes &&
                    options.maxBuffers <= 65536 && options.maxActiveRequests >= 0,
                "FastKV: invalid options");
        Require(options.transport != Transport::Rdma || RdmaAvailable(),
                "FastKV: RDMA requires libibverbs support and an available device");
    }
    static int ActiveLimit(const Options &o) {
        return o.maxActiveRequests > 0 ? o.maxActiveRequests : o.workers;
    }
    void Track(const std::shared_ptr<Channel> &c) {
        std::lock_guard<std::mutex> lock(mutex);
        Require(!stopped, "FastKV: service stopped");
        sockets.insert(c->socket.fd);
        c->tracked = true;
    }
    void Park(const std::shared_ptr<Exchange> &c) {
        std::lock_guard<std::mutex> lock(mutex);
        Require(!stopped, "FastKV: service stopped");
        pending.push_back(c);
        wake.notify_one();
    }
    void Fail(const std::shared_ptr<Exchange> &c, std::exception_ptr error) {
        ++failures;
        if (!c->server && c->cancellation->Cancelled()) ++cancelledRequests;
        c->cancellation->Cancel();
        c->channel.reset();
        if (!c->server) c->completion.set_exception(error);
    }
    static bool PeerClosed(const Socket &socket) {
        pollfd fd{socket.fd, POLLIN, 0};
        if (poll(&fd, 1, 0) <= 0) return false;
        char byte;
        auto n = recv(socket.fd, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
        return n == 0 || (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR);
    }
    // Host snapshots are already resident: copying them is slower than their
    // registration. GPU snapshots benefit from reusing expensive GPU MRs.
    std::vector<Buffer> PrepareSend(RdmaConnection &rdma, const std::vector<Buffer> &buffers) {
        auto result = buffers;
        for (size_t first = 0; first < buffers.size();) {
            if (buffers[first].Kind() == MemoryKind::Host || buffers[first].PoolBacked() || rdma.Registered(buffers[first])) { ++first; continue; }
            size_t end = first + 1, bytes = buffers[first].Size();
            const int device = buffers[first].Device();
            constexpr size_t arenaBytes = 64 << 20;
            while (end < buffers.size() && bytes < arenaBytes && buffers[end].Size() <= arenaBytes - bytes &&
                   buffers[end].Device() == device && !buffers[end].PoolBacked() && !rdma.Registered(buffers[end]))
                bytes = CheckedAdd(bytes, buffers[end++].Size());
            Buffer arena;
            try { arena = sendBuffers.Acquire(bytes, device); }
            catch (const std::runtime_error &) {
                // Pool pressure changes performance, never delivery semantics.
                // Register the caller-owned buffers directly in this case.
                first = end;
                continue;
            }
            size_t offset = 0;
            for (size_t i = first; i < end; ++i) {
                arena.CopyFrom(buffers[i], 0, offset, buffers[i].Size());
                result[i] = arena.Slice(offset, buffers[i].Size());
                offset += buffers[i].Size();
            }
            first = end;
        }
        return result;
    }
    void SendMessage(const Socket &s, RdmaConnection *rdma, const Message &message, Deadline deadline) {
        Require(message.metadata.size() <= options.maxMetadataBytes &&
                    message.buffers.size() <= options.maxBuffers,
                "FastKV: message limits exceeded");
        Writer w;
        w.U32(0);
        w.Blob(message.metadata.data(), message.metadata.size());
        w.U32(uint32_t(message.buffers.size()));
        size_t total = 0;
        for (auto &b : message.buffers) {
            total = CheckedAdd(total, b.Size());
            w.U64(b.Size());
        }
        Require(total <= options.maxPayloadBytes, "FastKV: payload limit exceeded");
        s.SendFrame(w.bytes, deadline);
        auto ack = s.ReceiveFrame(16 << 20, deadline);
        Reader r(ack);
        uint32_t accepted = r.U32();
        if (!accepted)
            throw std::runtime_error("FastKV receiver: " + r.String());
        std::vector<Buffer> spans;
        const bool targeted = r.Remaining() != 0;
        if (targeted) {
            Require(r.U32() == ReceiveTargetFormat, "FastKV: unknown receive target format");
            size_t count = 0;
            for (const auto &buffer : message.buffers) {
                const size_t n = r.U32();
                count = CheckedAdd(count, n);
                Require(count <= (1u << 20), "FastKV: too many receive spans");
                size_t offset = 0;
                for (size_t i = 0; i < n; ++i) {
                    const uint64_t bytes = r.U64();
                    Require(bytes && bytes <= buffer.Size() - offset, "FastKV: invalid receive span");
                    if (rdma) spans.push_back(buffer.Slice(offset, size_t(bytes)));
                    offset += size_t(bytes);
                }
                Require(offset == buffer.Size(), "FastKV: incomplete receive target");
            }
        }
        r.End();
        if (rdma) {
            auto prepared = PrepareSend(*rdma, targeted ? spans : message.buffers);
            rdma->Send(s, prepared, deadline);
            rdmaBytes += total;
        } else {
            std::vector<uint8_t> host;
            for (auto &b : message.buffers)
                for (size_t offset = 0; offset < b.Size();) {
                    size_t n = std::min(Chunk, b.Size() - offset);
                    if (b.Kind() == MemoryKind::Host)
                        s.Send(static_cast<const uint8_t *>(b.Data()) + offset, n, deadline);
                    else {
                        host.resize(n);
                        b.Read(host.data(), offset, n);
                        s.Send(host.data(), n, deadline);
                    }
                    offset += n;
                }
            tcpBytes += total;
        }
        auto committed = s.ReceiveFrame(64, deadline);
        Reader done(committed);
        Require(done.U32() == Magic, "FastKV: payload not acknowledged");
        done.End();
        if (rdma) rdma->Complete();
    }
    Message ReceiveMessage(const Socket &s, RdmaConnection *rdma, Deadline deadline, const ReceiveTarget &target = {}) {
        auto b = s.ReceiveFrame(
            CheckedAdd(options.maxMetadataBytes, CheckedAdd(16, options.maxBuffers * 8)), deadline);
        Reader r(b);
        uint32_t status = r.U32();
        if (status == 1)
            throw std::runtime_error("FastKV remote: " + r.String());
        Require(status == 0, "FastKV: unknown message status");
        Message m;
        m.metadata = r.Blob(options.maxMetadataBytes);
        size_t count = r.U32();
        Require(count <= options.maxBuffers, "FastKV: too many buffers");
        std::vector<size_t> sizes;
        size_t total = 0;
        for (size_t i = 0; i < count; ++i) {
            uint64_t n = r.U64();
            Require(n <= SIZE_MAX, "FastKV: buffer size overflow");
            sizes.push_back(size_t(n));
            total = CheckedAdd(total, size_t(n));
        }
        r.End();
        Writer ready;
        std::shared_ptr<void> targetCredit;
        try {
            Require(total <= options.maxPayloadBytes, "FastKV: payload limit exceeded");
            if (target) {
                // Caller pages are already reserved. Count transfer pressure,
                // without charging these pages to our staging arena pool.
                for (;;) {
                    s.CheckRunning(deadline);
                    try { targetCredit = budget.Reserve(total); break; }
                    catch (const CapacityBusy &) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
                }
                auto destinations = target(m.metadata, sizes);
                Require(destinations.size() == sizes.size(), "FastKV: receive target count mismatch");
                ready.U32(1); ready.U32(ReceiveTargetFormat);
                size_t count = 0;
                for (size_t i = 0; i < sizes.size(); ++i) {
                    count = CheckedAdd(count, destinations[i].size());
                    Require(count <= (1u << 20), "FastKV: too many receive spans");
                    ready.U32(uint32_t(destinations[i].size()));
                    size_t bytes = 0;
                    for (auto &span : destinations[i]) {
                        Require(span.Size() && span.MutableData(), "FastKV: empty receive span");
                        bytes = CheckedAdd(bytes, span.Size());
                        ready.U64(span.Size()); m.buffers.push_back(span.WithLease(targetCredit));
                    }
                    Require(bytes == sizes[i], "FastKV: receive target size mismatch");
                }
            } else {
                auto devices = options.receivePlacement ? options.receivePlacement(m.metadata, sizes)
                                                        : std::vector<int>{};
                if (devices.empty()) devices.assign(sizes.size(), options.receiveDevice);
                Require(devices.size() == sizes.size(), "FastKV: invalid receive placement count");
                for (int device : devices) Require(device >= -1, "FastKV: invalid receive device");
                struct Arena { size_t begin, end, bytes; int device; };
                std::vector<Arena> plan;
                std::vector<std::pair<int, size_t>> allocations;
                for (size_t start = 0; start < sizes.size();) {
                    size_t end = start, bytes = 0;
                    constexpr size_t arenaBytes = 64 << 20;
                    while (end < sizes.size()) {
                        if (end > start && (devices[end] != devices[start] ||
                                            bytes >= arenaBytes || sizes[end] > arenaBytes - bytes)) break;
                        bytes = CheckedAdd(bytes, sizes[end++]);
                    }
                    plan.push_back({start, end, bytes, devices[start]});
                    allocations.emplace_back(devices[start], bytes);
                    start = end;
                }
                receiveBuffers.CheckCapacity(allocations);
                // Reserve a complete message, not competing partial messages. Only
                // transfer workers wait here; the model can consume earlier replies.
                std::unique_lock<std::timed_mutex> allocationLock(receiveAllocationMutex, std::defer_lock);
                while (!allocationLock.try_lock_for(std::chrono::milliseconds(5))) s.CheckRunning(deadline);
                for (;;) {
                    s.CheckRunning(deadline);
                    try {
                        auto credit = budget.Reserve(total);
                        for (const auto &part : plan) {
                            auto arena = receiveBuffers.Acquire(part.bytes, part.device, credit);
                            size_t offset = 0;
                            for (size_t i = part.begin; i < part.end; ++i) {
                                m.buffers.push_back(arena.Slice(offset, sizes[i]));
                                offset += sizes[i];
                            }
                        }
                        break;
                    } catch (const CapacityBusy &) {
                        m.buffers.clear();
                        receiveBuffers.WaitForRelease();
                    }
                }
                ready.U32(1);
            }
        } catch (const std::exception &e) {
            ready.bytes.clear();
            ready.U32(0);
            ready.String(e.what());
            s.SendFrame(ready.bytes, deadline);
            throw;
        }
        s.SendFrame(ready.bytes, deadline);
        if (rdma) {
            rdma->Receive(s, m.buffers, deadline);
            rdmaBytes += total;
        } else {
            std::vector<uint8_t> host;
            for (auto &buffer : m.buffers)
                for (size_t offset = 0; offset < buffer.Size();) {
                    size_t n = std::min(Chunk, buffer.Size() - offset);
                    if (buffer.Kind() == MemoryKind::Host)
                        s.Receive(static_cast<uint8_t *>(buffer.MutableData()) + offset, n, deadline);
                    else {
                        host.resize(n);
                        s.Receive(host.data(), n, deadline);
                        buffer.Write(host.data(), offset, n);
                    }
                    offset += n;
                }
            tcpBytes += total;
        }
        Writer done;
        done.U32(Magic);
        s.SendFrame(done.bytes, deadline);
        if (rdma) rdma->Complete();
        return m;
    }
    std::shared_ptr<Channel> TakeChannel(const std::string &peer) {
        for (;;) {
            std::shared_ptr<Channel> channel;
            {
                std::lock_guard<std::mutex> lock(mutex);
                auto it = idleClients.find(peer);
                if (it == idleClients.end()) return {};
                channel = std::move(it->second);
                idleClients.erase(it);
            }
            pollfd fd{channel->socket.fd, POLLIN, 0};
            // An idle channel must have no unread bytes or disconnect event.
            if (poll(&fd, 1, 0) == 0) return channel;
        }
    }
    void RecycleServer(const std::shared_ptr<Exchange> &c) {
        if (!c->channel->reusable) return;
        auto next = std::make_shared<Exchange>(*this, true);
        next->channel = c->channel;
        next->stage = Stage::IdleRequest;
        Park(next);
    }
    void BeginCall(const std::shared_ptr<Exchange> &c) {
        try {
            Require(!c->cancellation->Cancelled(), "FastKV: request cancelled");
            c->channel = TakeChannel(c->peer);
            if (c->channel) {
                c->channel->socket.cancellation = c->cancellation;
                ++reusedConnections;
            }
            else {
                bool legacy;
                { std::lock_guard<std::mutex> lock(mutex); legacy = legacyPeers.count(c->peer); }
                auto connect = [&](bool oldProtocol) {
                    c->channel = std::make_shared<Channel>(*this);
                    c->channel->socket = Open(c->peer, false, c->deadline, &stopped, c->cancellation);
                    Track(c->channel);
                    c->channel->rdma = Handshake(c->channel->socket, options, false, c->deadline,
                                                c->channel->reusable, oldProtocol);
                };
                try { connect(legacy); }
                catch (const LegacyHandshake &) {
                    Require(!c->cancellation->Cancelled(), "FastKV: request cancelled");
                    // Negotiation failed before any application bytes were
                    // sent. Retry v1 once; never replay an application request.
                    c->channel.reset();
                    connect(true);
                    std::lock_guard<std::mutex> lock(mutex);
                    if (legacyPeers.size() >= size_t(options.queueDepth)) legacyPeers.clear();
                    legacyPeers.insert(c->peer);
                }
                c->channel->ready = true;
                ++connections;
            }
            SendMessage(c->channel->socket, c->channel->rdma.get(), c->request, c->deadline);
            c->request = {};
            c->stage = Stage::Reply;
            Park(c);
        } catch (...) { Fail(c, std::current_exception()); }
    }
    void ReceiveRequest(const std::shared_ptr<Exchange> &c) {
        try {
            c->channel->socket.cancellation = c->cancellation;
            if (!c->channel->ready) {
                c->channel->rdma = Handshake(c->channel->socket, options, true, c->deadline,
                                            c->channel->reusable);
                c->channel->ready = true;
                ++connections;
            } else ++reusedConnections;
            c->request = ReceiveMessage(c->channel->socket, c->channel->rdma.get(), c->deadline);
            c->stage = Stage::Admit;
            if (control && c->request.buffers.empty()) {
                std::promise<Message> ready;
                Message reply;
                bool handled = false;
                try {
                    handled = control(c->request, reply);
                    if (handled) {
                        Require(reply.buffers.empty(), "FastKV: control reply must be metadata-only");
                        ready.set_value(std::move(reply));
                    }
                } catch (...) { handled = true; ready.set_exception(std::current_exception()); }
                if (handled) {
                    c->request = {};
                    c->result = ready.get_future();
                    c->stage = Stage::Handler;
                }
            }
            Park(c);
        } catch (...) { Fail(c, std::current_exception()); }
    }
    void BeginHandler(const std::shared_ptr<Exchange> &c) {
        try {
            c->channel->socket.CheckRunning(c->deadline);
            // Keep the admission lease through sending the reply. A producer
            // cannot fill GPU memory with completed, unsent snapshots.
            std::promise<Message> failed;
            try { c->result = handler(std::move(c->request), c->cancellation); }
            catch (...) {
                failed.set_exception(std::current_exception());
                c->result = failed.get_future();
            }
            Require(c->result.valid(), "FastKV: handler returned an invalid future");
            Require(c->result.wait_for(std::chrono::seconds(0)) != std::future_status::deferred,
                    "FastKV: async handler requires a promise-backed future");
            c->stage = Stage::Handler;
            Park(c);
        } catch (...) { Fail(c, std::current_exception()); }
    }
    void SendReply(const std::shared_ptr<Exchange> &c) {
        try {
            Message reply;
            try { reply = c->result.get(); }
            catch (const std::exception &e) {
                Writer error;
                error.U32(1); error.String(e.what());
                c->channel->socket.SendFrame(error.bytes, c->deadline);
                ++failures;
                return;
            }
            SendMessage(c->channel->socket, c->channel->rdma.get(), reply, c->deadline);
            ++calls;
            c->channel->socket.cancellation.reset();
            RecycleServer(c);
        } catch (...) { Fail(c, std::current_exception()); }
    }
    void ReceiveReply(const std::shared_ptr<Exchange> &c) {
        try {
            auto reply = ReceiveMessage(c->channel->socket, c->channel->rdma.get(), c->deadline, c->target);
            ++calls;
            c->channel->socket.cancellation.reset();
            if (c->channel->reusable) {
                std::lock_guard<std::mutex> lock(mutex);
                if (!stopped && idleClients.size() < size_t(options.queueDepth) + options.workers)
                    idleClients.emplace(c->peer, c->channel);
            }
            c->channel.reset();
            c->completion.set_value(std::move(reply));
        } catch (...) { Fail(c, std::current_exception()); }
    }
    void Progress() {
        for (;;) {
            std::vector<std::shared_ptr<Exchange>> work, waiting;
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [&] { return stopped || !pending.empty(); });
                work.swap(pending);
            }
            for (auto &c : work) {
                try {
                    if (c->stage == Stage::IdleRequest &&
                        (stopped || std::chrono::steady_clock::now() >= c->deadline)) continue;
                    Require(!stopped, "FastKV: service stopped");
                    if (!c->server)
                        Require(!c->cancellation->Cancelled(), "FastKV: request cancelled");
                    const bool expired = std::chrono::steady_clock::now() >= c->deadline;
                    const bool disconnected = c->server &&
                        (c->stage == Stage::Admit || c->stage == Stage::Handler) && PeerClosed(c->channel->socket);
                    if (c->stage != Stage::Drain && (expired || disconnected)) {
                        if (c->stage != Stage::Handler) {
                            Require(!disconnected, "FastKV: peer closed connection");
                            throw std::runtime_error("FastKV: transport deadline exceeded");
                        }
                        // Computation may still be running. Close the network
                        // endpoint, but retain admission until its result dies.
                        c->cancellation->Cancel();
                        ++cancelledRequests;
                        shutdown(c->channel->socket.fd, SHUT_RDWR);
                        c->channel->rdma.reset();
                        c->stage = Stage::Drain;
                        ++failures;
                    }
                    bool scheduled = false;
                    switch (c->stage) {
                    case Stage::Connect:
                        scheduled = clients.TryEnqueue([this, c] { BeginCall(c); });
                        break;
                    case Stage::ReceiveRequest:
                        scheduled = servers.TryEnqueue([this, c] { ReceiveRequest(c); });
                        break;
                    case Stage::IdleRequest: {
                        pollfd fd{c->channel->socket.fd, POLLIN, 0};
                        if (poll(&fd, 1, 0) > 0) {
                            char byte;
                            const auto n = recv(fd.fd, &byte, 1, MSG_PEEK | MSG_DONTWAIT);
                            if (n == 0 || (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK)) {
                                scheduled = true; // normal idle peer disconnect
                                break;
                            }
                            if (n < 0) break;
                            c->deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(options.timeoutMs);
                            scheduled = servers.TryEnqueue([this, c] { ReceiveRequest(c); });
                        }
                        break;
                    }
                    case Stage::Admit:
                        if (!c->handlerSlot) {
                            try { c->handlerSlot = handlerSlots.Reserve(1); }
                            catch (const std::runtime_error &) { break; }
                        }
                        scheduled = servers.TryEnqueue([this, c] { BeginHandler(c); });
                        break;
                    case Stage::Handler:
                        if (c->result.wait_for(std::chrono::seconds(0)) != std::future_status::timeout)
                            scheduled = sendReplies.TryEnqueue([this, c] { SendReply(c); });
                        break;
                    case Stage::Drain:
                        // No get(): discarding an exceptional result is safe.
                        scheduled = c->result.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
                        break;
                    case Stage::Reply: {
                        pollfd fd{c->channel->socket.fd, POLLIN, 0};
                        if (poll(&fd, 1, 0) > 0)
                            scheduled = receiveReplies.TryEnqueue([this, c] { ReceiveReply(c); });
                        break;
                    }
                    }
                    if (!scheduled) waiting.push_back(std::move(c));
                } catch (...) { Fail(c, std::current_exception()); }
            }
            // Destroy completed exchange owners outside mutex (they untrack
            // their descriptors there). New work may arrive during dispatch.
            work.clear();
            std::unique_lock<std::mutex> lock(mutex);
            pending.insert(pending.end(), std::make_move_iterator(waiting.begin()),
                           std::make_move_iterator(waiting.end()));
            if (stopped && pending.empty()) break;
            if (!pending.empty()) wake.wait_for(lock, std::chrono::milliseconds(1));
        }
    }
};
Service::Service(Options o) : impl(new Impl(std::move(o))) {
    impl->progress = std::thread([this] { impl->Progress(); });
}
Service::~Service() { Stop(); }
void Service::Start(const std::string &address, Handler h) {
    Require(bool(h), "FastKV: invalid handler");
    StartAsync(address, [h = std::move(h)](Message request) {
        std::promise<Message> result;
        result.set_value(h(request));
        return result.get_future();
    });
}
void Service::StartAsync(const std::string &address, AsyncHandler h) {
    Require(bool(h), "FastKV: invalid handler");
    StartAsync(address, [h = std::move(h)](Message request, std::shared_ptr<Cancellation>) {
        return h(std::move(request));
    });
}
void Service::StartAsync(const std::string &address, CancellableHandler h, ControlHandler control) {
    Require(!impl->stopped && !impl->acceptor.joinable() && bool(h), "FastKV: invalid service start");
    impl->handler = std::move(h);
    impl->control = std::move(control);
    impl->listener = Open(address, true, std::chrono::steady_clock::now() + std::chrono::seconds(30));
    impl->acceptor = std::thread([this] {
        while (!impl->stopped) {
            pollfd p{impl->listener.fd, POLLIN, 0};
            int n = poll(&p, 1, 100);
            if (n <= 0)
                continue;
            std::shared_ptr<void> slot;
            try { slot = impl->serverSlots.Reserve(1); }
            catch (const CapacityBusy &) {
                // Leave excess connections in the kernel backlog instead of
                // accepting and immediately resetting an otherwise valid RPC.
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            int fd = accept(impl->listener.fd, nullptr, nullptr);
            if (fd < 0)
                continue;
            Socket socket(fd, &impl->stopped);
            try {
                Configure(fd);
                auto c = std::make_shared<Impl::Exchange>(*impl, true);
                c->channel = std::make_shared<Impl::Channel>(*impl);
                c->channel->serverSlot = std::move(slot);
                c->channel->socket = std::move(socket);
                impl->Track(c->channel);
                impl->Park(c);
            } catch (...) {
                ++impl->failures;
            }
        }
    });
}
void Service::Stop() {
    if (impl->stopped.exchange(true))
        return;
    if (impl->listener.fd >= 0)
        shutdown(impl->listener.fd, SHUT_RDWR);
    {
        std::lock_guard<std::mutex> lock(impl->mutex);
        for (int fd : impl->sockets)
            shutdown(fd, SHUT_RDWR);
    }
    if (impl->acceptor.joinable())
        impl->acceptor.join();
    impl->wake.notify_all();
    if (impl->progress.joinable()) impl->progress.join();
    impl->clients.Stop();
    impl->servers.Stop();
    impl->sendReplies.Stop();
    impl->receiveReplies.Stop();
    decltype(impl->idleClients) idle;
    { std::lock_guard<std::mutex> lock(impl->mutex); idle.swap(impl->idleClients); }
    idle.clear();
    impl->receiveBuffers.Clear();
    impl->sendBuffers.Clear();
    impl->listener = Socket();
}
std::string Service::Address() const { return BoundAddress(impl->listener); }
Message Service::Call(const std::string &peer, const Message &m) { return Submit(peer, m).get(); }
std::future<Message> Service::Submit(const std::string &peer, Message m) {
    std::future<Message> result;
    if (!TrySubmit(peer, m, result)) throw CapacityBusy("FastKV: client request slots exhausted");
    return result;
}
bool Service::TrySubmit(const std::string &peer, Message &request, std::future<Message> &result,
                        std::shared_ptr<Cancellation> cancellation, std::shared_ptr<void> lifetime, ReceiveTarget target) {
    Require(!impl->stopped, "FastKV: service stopped");
    Require(!result.valid(), "FastKV: result already holds a submission");
    std::shared_ptr<void> slot;
    try { slot = impl->clientSlots.Reserve(1); }
    catch (const CapacityBusy &) { ++impl->admissionWaits; return false; }
    auto c = std::make_shared<Impl::Exchange>(*impl, false, std::move(slot));
    if (cancellation) c->cancellation = std::move(cancellation);
    c->lifetime = std::move(lifetime);
    c->target = std::move(target);
    c->peer = peer;
    auto future = c->completion.get_future();
    {
        std::lock_guard<std::mutex> lock(impl->mutex);
        Require(!impl->stopped, "FastKV: service stopped");
        c->request = std::move(request);
        impl->pending.push_back(c);
    }
    result = std::move(future);
    impl->wake.notify_one();
    return true;
}
Metrics Service::Stats() const {
    return {impl->tcpBytes.load(), impl->rdmaBytes.load(), impl->calls.load(), impl->failures.load(),
            impl->connections.load(), impl->reusedConnections.load(),
            impl->clientSlots.Used(), impl->handlerSlots.Used(),
            impl->cancelledRequests.load(), impl->admissionWaits.load()};
}
bool Service::HasRdma() { return RdmaAvailable(); }
} // namespace fastkv
