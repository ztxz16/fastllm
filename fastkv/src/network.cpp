#include "internal.h"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <set>
#include <sys/socket.h>
#include <unistd.h>

namespace fastkv {
static constexpr uint32_t Magic = 0x314b5646, Version = 1;
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
    s.fd = -1;
    return *this;
}
void Socket::CheckRunning(Deadline deadline) const {
    Require(!stopped || !stopped->load(), "FastKV: service stopped");
    Require(std::chrono::steady_clock::now() < deadline, "FastKV: transport deadline exceeded");
}
static void Wait(int fd, short events, Deadline deadline, const std::atomic<bool> *stopped) {
    for (;;) {
        Require(!stopped || !stopped->load(), "FastKV: service stopped");
        auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline -
                                                                          std::chrono::steady_clock::now())
                        .count();
        Require(left > 0, "FastKV: transport deadline exceeded");
        pollfd p{fd, events, 0};
        int n = poll(&p, 1, int(std::min<int64_t>(left, 1000)));
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
            Wait(fd, POLLOUT, d, stopped);
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
            Wait(fd, POLLIN, d, stopped);
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
                   const std::atomic<bool> *stopped = nullptr) {
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
                Wait(socket.fd, POLLOUT, deadline, stopped);
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
static std::unique_ptr<RdmaConnection> Handshake(const Socket &s, const Options &o, bool server,
                                                 Deadline d) {
    Transport chosen = Transport::Tcp;
    if (server) {
        auto b = s.ReceiveFrame(64, d);
        Reader r(b);
        Require(r.U32() == Magic && r.U32() == Version, "FastKV: incompatible protocol");
        uint32_t mode = r.U32();
        r.End();
        Require(mode <= uint32_t(Transport::Auto), "FastKV: unknown transport");
        bool available = o.transport != Transport::Tcp && RdmaAvailable();
        chosen = (mode != uint32_t(Transport::Tcp) && available) ? Transport::Rdma : Transport::Tcp;
        Writer reply;
        reply.U32(Magic);
        reply.U32(Version);
        bool compatible = (mode != uint32_t(Transport::Rdma) || available) &&
                          (o.transport != Transport::Rdma || mode != uint32_t(Transport::Tcp));
        reply.U32(compatible ? uint32_t(chosen) : UINT32_MAX);
        s.SendFrame(reply.bytes, d);
        Require(compatible, "FastKV: requested transport is unavailable");
    } else {
        Writer hello;
        hello.U32(Magic);
        hello.U32(Version);
        const auto offered =
            o.transport == Transport::Auto && !RdmaAvailable() ? Transport::Tcp : o.transport;
        hello.U32(uint32_t(offered));
        s.SendFrame(hello.bytes, d);
        auto b = s.ReceiveFrame(64, d);
        Reader r(b);
        Require(r.U32() == Magic && r.U32() == Version, "FastKV: incompatible protocol");
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
    Budget budget;
    Pool servers, clients;
    Socket listener;
    Handler handler;
    std::thread acceptor;
    mutable std::mutex mutex;
    std::set<int> sockets;
    std::atomic<bool> stopped{false};
    std::atomic<uint64_t> tcpBytes{0}, rdmaBytes{0}, calls{0}, failures{0};
    explicit Impl(Options o)
        : options(std::move(o)), budget(options.maxInflightBytes),
          servers(options.workers, options.queueDepth), clients(options.workers, options.queueDepth) {
        Require(options.timeoutMs > 0 && options.maxPayloadBytes && options.maxMetadataBytes &&
                    options.maxBuffers <= 65536,
                "FastKV: invalid options");
        Require(options.transport != Transport::Rdma || RdmaAvailable(),
                "FastKV: RDMA requires libibverbs support and an available device");
    }
    struct Active {
        Impl &impl;
        int fd;
        Active(Impl &i, int f) : impl(i), fd(f) {
            std::lock_guard<std::mutex> lock(i.mutex);
            Require(!i.stopped, "FastKV: service stopped");
            i.sockets.insert(f);
        }
        ~Active() {
            std::lock_guard<std::mutex> lock(impl.mutex);
            impl.sockets.erase(fd);
        }
    };
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
        auto ack = s.ReceiveFrame(65536, deadline);
        Reader r(ack);
        uint32_t accepted = r.U32();
        if (!accepted)
            throw std::runtime_error("FastKV receiver: " + r.String());
        r.End();
        if (rdma) {
            rdma->Send(s, message.buffers, deadline);
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
    }
    Message ReceiveMessage(const Socket &s, RdmaConnection *rdma, Deadline deadline) {
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
        try {
            Require(total <= options.maxPayloadBytes, "FastKV: payload limit exceeded");
            auto credit = budget.Reserve(total);
            for (size_t start = 0; start < sizes.size();) {
                size_t end = start, bytes = 0;
                constexpr size_t arenaBytes = 64 << 20;
                while (end < sizes.size()) {
                    if (end > start && (bytes >= arenaBytes || sizes[end] > arenaBytes - bytes))
                        break;
                    bytes = CheckedAdd(bytes, sizes[end++]);
                }
                auto arena = Buffer::Allocate(bytes, options.receiveDevice);
                arena.Attachment("receive-credit", [credit](void *, size_t) { return credit; });
                size_t offset = 0;
                for (size_t i = start; i < end; ++i) {
                    m.buffers.push_back(arena.Slice(offset, sizes[i]));
                    offset += sizes[i];
                }
                start = end;
            }
            ready.U32(1);
        } catch (const std::exception &e) {
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
        return m;
    }
    Message Call(const std::string &peer, const Message &request) {
        Require(!stopped, "FastKV: service stopped");
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(options.timeoutMs);
        Socket socket = Open(peer, false, deadline, &stopped);
        Active active(*this, socket.fd);
        try {
            auto rdma = Handshake(socket, options, false, deadline);
            SendMessage(socket, rdma.get(), request, deadline);
            auto reply = ReceiveMessage(socket, rdma.get(), deadline);
            ++calls;
            return reply;
        } catch (...) {
            ++failures;
            throw;
        }
    }
    void Serve(const std::shared_ptr<Socket> &socket) {
        Active active(*this, socket->fd);
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(options.timeoutMs);
        try {
            auto rdma = Handshake(*socket, options, true, deadline);
            auto request = ReceiveMessage(*socket, rdma.get(), deadline);
            Message reply;
            try {
                reply = handler(request);
            } catch (const std::exception &e) {
                Writer error;
                error.U32(1);
                error.String(e.what());
                socket->SendFrame(error.bytes, deadline);
                ++failures;
                return;
            }
            SendMessage(*socket, rdma.get(), reply, deadline);
            ++calls;
        } catch (...) {
            ++failures;
        }
    }
};
Service::Service(Options o) : impl(new Impl(std::move(o))) {}
Service::~Service() { Stop(); }
void Service::Start(const std::string &address, Handler h) {
    Require(!impl->stopped && !impl->acceptor.joinable() && bool(h), "FastKV: invalid service start");
    impl->handler = std::move(h);
    impl->listener = Open(address, true, std::chrono::steady_clock::now() + std::chrono::seconds(30));
    impl->acceptor = std::thread([this] {
        while (!impl->stopped) {
            pollfd p{impl->listener.fd, POLLIN, 0};
            int n = poll(&p, 1, 100);
            if (n <= 0)
                continue;
            int fd = accept(impl->listener.fd, nullptr, nullptr);
            if (fd < 0)
                continue;
            auto socket = std::make_shared<Socket>(fd, &impl->stopped);
            try {
                Configure(fd);
                impl->servers.Enqueue([this, socket] { impl->Serve(socket); });
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
    impl->clients.Stop();
    impl->servers.Stop();
    impl->listener = Socket();
}
std::string Service::Address() const { return BoundAddress(impl->listener); }
Message Service::Call(const std::string &peer, const Message &m) { return impl->Call(peer, m); }
std::future<Message> Service::Submit(const std::string &peer, Message m) {
    auto task = std::make_shared<std::packaged_task<Message()>>(
        [this, peer, m = std::move(m)] { return impl->Call(peer, m); });
    auto future = task->get_future();
    impl->clients.Enqueue([task] { (*task)(); });
    return future;
}
Metrics Service::Stats() const {
    return {impl->tcpBytes.load(), impl->rdmaBytes.load(), impl->calls.load(), impl->failures.load()};
}
bool Service::HasRdma() { return RdmaAvailable(); }
} // namespace fastkv
