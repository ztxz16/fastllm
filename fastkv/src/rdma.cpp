#include "internal.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <set>
#include <cstdlib>
#include <deque>
#ifdef FASTKV_CUDA
#include <cuda_runtime_api.h>
#endif
#ifdef FASTKV_RDMA
#include <infiniband/verbs.h>
#endif

namespace fastkv {
#ifndef FASTKV_RDMA
bool RdmaAvailable() {
    return false;
}
std::unique_ptr<RdmaConnection> ConnectRdma(const Socket &, const Options &, Deadline) {
    throw std::runtime_error("FastKV was built without libibverbs transport");
}
#else
namespace {
constexpr uint32_t Done = 0x52444d41;
std::atomic<uint64_t> identities{1};
struct Device {
    ibv_context *context = nullptr;
    ibv_pd *pd = nullptr;
    const uint64_t id = identities.fetch_add(1);
    const std::string registrationKey = "verbs:" + std::to_string(id);
    const std::string sourceKey = registrationKey + ":source";
    ~Device() {
        if (pd)
            ibv_dealloc_pd(pd);
        if (context)
            ibv_close_device(context);
    }
};
std::mutex deviceMutex;
std::map<std::string, std::weak_ptr<Device>> deviceCache;
std::shared_ptr<Device> OpenDevice(ibv_device *device) {
    const std::string name = ibv_get_device_name(device);
    std::lock_guard<std::mutex> lock(deviceMutex);
    auto &cached = deviceCache[name];
    if (auto existing = cached.lock())
        return existing;
    auto result = std::make_shared<Device>();
    result->context = ibv_open_device(device);
    Require(result->context, "FastKV: open RDMA device failed");
    result->pd = ibv_alloc_pd(result->context);
    Require(result->pd, "FastKV: allocate protection domain failed");
    cached = result;
    return result;
}
struct Registration {
    std::shared_ptr<Device> device;
    ibv_mr *mr = nullptr;
    ~Registration() {
        if (mr)
            ibv_dereg_mr(mr);
    }
};
struct Endpoint {
    uint32_t qpn = 0, psn = 0, lid = 0, mtu = 0;
    ibv_gid gid{};
};
std::string PciPath(const std::string &path) {
    char *resolved = realpath(path.c_str(), nullptr);
    if (!resolved) return {};
    std::string result(resolved);
    free(resolved);
    return result;
}
size_t CommonPath(const std::string &a, const std::string &b) {
    size_t last = 0;
    for (size_t i = 0; i < std::min(a.size(), b.size()) && a[i] == b[i]; ++i)
        if (a[i] == '/') last = i;
    return last;
}
struct Rail {
    std::string pciPath;
    std::shared_ptr<Device> device;
    ibv_cq *cq = nullptr;
    ibv_qp *qp = nullptr;
    Endpoint local;
    int port = 1, gidIndex = 0;
    size_t outstanding = 0;
    uint64_t nextId = 1;
    void Close() noexcept {
        if (qp) {
            ibv_qp_attr attr{};
            attr.qp_state = IBV_QPS_ERR;
            ibv_modify_qp(qp, &attr, IBV_QP_STATE);
            ibv_destroy_qp(qp);
            qp = nullptr;
        }
        if (cq) {
            ibv_destroy_cq(cq);
            cq = nullptr;
        }
    }
    ~Rail() { Close(); }
    std::shared_ptr<Registration> Register(const Buffer &buffer, bool optionalSource = false) {
        const auto &key = optionalSource ? device->sourceKey : device->registrationKey;
        auto value = buffer.Attachment(key, [d = device, optionalSource](void *p, size_t n) {
            auto reg = std::make_shared<Registration>();
            reg->device = d;
            // GPU peer memory needs a writable local pin even for a source-only
            // MR. The send side never advertises remote access to this backing.
            const int flags = IBV_ACCESS_LOCAL_WRITE |
                (optionalSource ? 0 : (IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ));
            reg->mr = ibv_reg_mr(d->pd, p, std::max(size_t(1), n), flags);
            // Cache an optional failure as well. All slices on this physical
            // slab can then use send windows without retrying MR creation.
            Require(optionalSource || reg->mr, "FastKV: ibv_reg_mr failed (check memlock and GPU peer-memory support)");
            return reg;
        });
        auto registration = std::static_pointer_cast<Registration>(value);
        return registration->mr ? registration : nullptr;
    }
    void Poll(const Socket &socket, Deadline deadline, bool drain) {
        while (outstanding) {
            socket.CheckRunning(deadline);
            ibv_wc completions[32];
            int count = ibv_poll_cq(cq, 32, completions);
            Require(count >= 0, "FastKV: ibv_poll_cq failed");
            for (int i = 0; i < count; ++i) {
                if (completions[i].status != IBV_WC_SUCCESS)
                    throw std::runtime_error(std::string("FastKV RDMA completion: ") +
                                             ibv_wc_status_str(completions[i].status));
                Require(outstanding > 0, "FastKV: unexpected RDMA completion");
                --outstanding;
            }
            if (count && !drain)
                return;
            Require(std::chrono::steady_clock::now() < deadline, "FastKV: RDMA deadline exceeded");
            if (!count)
                std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
    }
};
std::unique_ptr<Rail> CreateRail(const std::string &name, const Options &o) {
    auto rail = std::make_unique<Rail>();
    rail->port = o.rdmaPort;
    int count = 0;
    ibv_device **list = ibv_get_device_list(&count);
    Require(list, "FastKV: enumerate RDMA devices failed");
    std::unique_ptr<ibv_device *, decltype(&ibv_free_device_list)> guard(list, ibv_free_device_list);
    std::string selected;
    for (int i = 0; i < count; ++i)
        if (name.empty() || name == ibv_get_device_name(list[i])) {
            rail->device = OpenDevice(list[i]);
            selected = ibv_get_device_name(list[i]);
            break;
        }
    Require(bool(rail->device), "FastKV: RDMA device unavailable");
    rail->pciPath = PciPath("/sys/class/infiniband/" + selected + "/device");
    ibv_port_attr port{};
    Require(!ibv_query_port(rail->device->context, o.rdmaPort, &port) && port.state == IBV_PORT_ACTIVE,
            "FastKV: RDMA port is not active");
    rail->gidIndex = o.gidIndex;
    if (rail->gidIndex < 0) {
        rail->gidIndex = 0;
        if (port.link_layer == IBV_LINK_LAYER_ETHERNET) {
            rail->gidIndex = -1;
            for (int i = 0; i < port.gid_tbl_len; ++i) {
                ibv_gid gid{};
                if (ibv_query_gid(rail->device->context, o.rdmaPort, i, &gid))
                    continue;
                std::ifstream f("/sys/class/infiniband/" + selected + "/ports/" + std::to_string(o.rdmaPort) +
                                "/gid_attrs/types/" + std::to_string(i));
                std::string type;
                std::getline(f, type);
                if (type == "RoCE v2" && gid.raw[10] == 0xff && gid.raw[11] == 0xff) {
                    rail->gidIndex = i;
                    break;
                }
            }
            Require(rail->gidIndex >= 0, "FastKV: no IPv4 RoCE-v2 GID; configure gidIndex explicitly");
        }
    }
    Require(!ibv_query_gid(rail->device->context, o.rdmaPort, rail->gidIndex, &rail->local.gid),
            "FastKV: query GID failed");
    rail->cq = ibv_create_cq(rail->device->context, 128, nullptr, nullptr, 0);
    Require(rail->cq, "FastKV: create CQ failed");
    ibv_qp_init_attr init{};
    init.send_cq = rail->cq;
    init.recv_cq = rail->cq;
    init.qp_type = IBV_QPT_RC;
    init.cap.max_send_wr = 64;
    init.cap.max_recv_wr = 1;
    init.cap.max_send_sge = 1;
    init.cap.max_recv_sge = 1;
    rail->qp = ibv_create_qp(rail->device->pd, &init);
    Require(rail->qp, "FastKV: create RC QP failed");
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = o.rdmaPort;
    attr.pkey_index = 0;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    Require(
        !ibv_modify_qp(rail->qp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS),
        "FastKV: QP INIT failed");
    rail->local.qpn = rail->qp->qp_num;
    rail->local.lid = port.lid;
    rail->local.mtu = port.active_mtu;
    rail->local.psn = uint32_t((rail->device->id * 2654435761ULL) ^
                               std::chrono::steady_clock::now().time_since_epoch().count()) &
                      0xffffff;
    return rail;
}
void ConnectRail(Rail &rail, const Endpoint &peer) {
    Require(peer.mtu >= IBV_MTU_256 && peer.mtu <= IBV_MTU_4096 && peer.qpn,
            "FastKV: invalid RDMA peer metadata");
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = ibv_mtu(std::min(rail.local.mtu, peer.mtu));
    attr.dest_qp_num = peer.qpn;
    attr.rq_psn = peer.psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.dlid = peer.lid;
    attr.ah_attr.port_num = rail.port;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid = peer.gid;
    attr.ah_attr.grh.sgid_index = rail.gidIndex;
    attr.ah_attr.grh.hop_limit = 64;
    Require(!ibv_modify_qp(rail.qp, &attr,
                           IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                               IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER),
            "FastKV: QP RTR failed");
    attr = {};
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = rail.local.psn;
    attr.max_rd_atomic = 1;
    Require(!ibv_modify_qp(rail.qp, &attr,
                           IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                               IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC),
            "FastKV: QP RTS failed");
}
class Connection final : public RdmaConnection {
  public:
    std::vector<std::unique_ptr<Rail>> rails;
    std::vector<Buffer> held;
    using Registrations = std::vector<std::shared_ptr<Registration>>;
    struct SendWindow {
        Buffer buffer;
        Registrations regs;
        size_t used = 0;
    };
    // A failed send keeps its local DMA sources alive through QP teardown.
    std::map<int, SendWindow> sendWindows;
    std::map<int, std::vector<size_t>> deviceRails;
    const std::vector<size_t> &SendRails(const Buffer &buffer) {
        const int device = buffer.Kind() == MemoryKind::Cuda ? buffer.Device() : -1;
        auto found = deviceRails.find(device);
        if (found != deviceRails.end()) return found->second;
        std::vector<size_t> selected;
        std::string path;
#ifdef FASTKV_CUDA
        char address[32]{};
        if (device >= 0 && cudaDeviceGetPCIBusId(address, sizeof(address), device) == cudaSuccess) {
            for (char &c : address) if (c >= 'A' && c <= 'F') c += 'a' - 'A';
            path = PciPath(std::string("/sys/bus/pci/devices/") + address);
        }
#endif
        size_t best = 0;
        for (size_t i = 0; i < rails.size(); ++i) {
            size_t locality = path.empty() ? 0 : CommonPath(path, rails[i]->pciPath);
            if (locality > best) { selected.clear(); best = locality; }
            if (locality == best) selected.push_back(i);
        }
        // Host transfers stripe over all rails. Unknown GPU topology does too.
        return deviceRails.emplace(device, std::move(selected)).first->second;
    }
    ~Connection() override {
        // Destroy every QP before releasing any receive allocation. On error,
        // posted writes must not outlive the MR or touch a recycled slot.
        for (auto &r : rails)
            r->Close();
        held.clear();
    }
    void Complete() override {
        for (auto &rail : rails) Require(!rail->outstanding, "FastKV: incomplete RDMA operation");
        held.clear();
    }
    bool Registered(const Buffer &buffer) {
        for (auto index : SendRails(buffer))
            if (!buffer.HasAttachment(rails[index]->device->registrationKey)) return false;
        return true;
    }
    Registrations Register(const Buffer &source, bool optionalSource = false) {
        Registrations regs(rails.size());
        for (size_t index : SendRails(source)) {
            regs[index] = rails[index]->Register(source, optionalSource);
            if (!regs[index]) return {};
        }
        return regs;
    }
    void Drain(const Socket &socket, Deadline deadline) {
        for (auto &rail : rails) rail->Poll(socket, deadline, true);
        for (auto &item : sendWindows) item.second.used = 0;
    }
    SendWindow &Window(int device, BufferPool &pool, const Socket &socket, Deadline deadline) {
        auto found = sendWindows.find(device);
        if (found != sendWindows.end()) return found->second;
        const size_t bytes = std::min(size_t(16) << 20, pool.Limit(device));
        Buffer buffer;
        for (;;) {
            socket.CheckRunning(deadline);
            try { buffer = pool.Acquire(bytes, device); break; }
            catch (const CapacityBusy &) {
                // A small pool can be shared by several devices/connections.
                // Drain and release our own leases before waiting on another.
                if (!sendWindows.empty()) {
                    Drain(socket, deadline);
                    sendWindows.clear();
                } else pool.WaitForRelease();
            }
        }
        auto regs = Register(buffer);
        return sendWindows.emplace(device, SendWindow{std::move(buffer), std::move(regs)}).first->second;
    }
    void Send(const Socket &socket, const std::vector<Buffer> &payloads,
              const std::vector<size_t> &spanSizes, BufferPool &windows, Deadline deadline) override {
        held.insert(held.end(), payloads.begin(), payloads.end());
        auto bytes = socket.ReceiveFrame(32 << 20, deadline);
        Reader reader(bytes);
        Require(reader.U32() == spanSizes.size() && reader.U32() == rails.size(),
                "FastKV: RDMA destination shape mismatch");
        struct Remote {
            uint64_t address, size;
            std::vector<uint32_t> keys;
        };
        std::vector<Remote> remote;
        for (size_t size : spanSizes) {
            Remote target;
            target.address = reader.U64();
            target.size = reader.U64();
            Require(target.size == size, "FastKV: RDMA destination size mismatch");
            for (size_t i = 0; i < rails.size(); ++i)
                target.keys.push_back(reader.U32());
            remote.push_back(std::move(target));
        }
        reader.End();
        uint64_t total = 0, sequence = 0;
        auto post = [&](const Buffer &source, size_t localOffset, size_t b, size_t offset,
                        uint32_t n, const std::vector<size_t> &selected,
                        const Registrations &regs) {
            const size_t index = selected[sequence++ % selected.size()];
            auto &rail = *rails[index];
            if (rail.outstanding >= 32) rail.Poll(socket, deadline, false);
            ibv_sge sge{};
            sge.addr = uint64_t(reinterpret_cast<uintptr_t>(source.Data())) + localOffset;
            sge.length = n;
            sge.lkey = regs[index]->mr->lkey;
            ibv_send_wr wr{}, *bad = nullptr;
            wr.wr_id = rail.nextId++;
            wr.sg_list = &sge;
            wr.num_sge = 1;
            wr.opcode = IBV_WR_RDMA_WRITE;
            wr.send_flags = IBV_SEND_SIGNALED;
            Require(offset <= UINT64_MAX - remote[b].address, "FastKV: remote address overflow");
            wr.wr.rdma.remote_addr = remote[b].address + offset;
            wr.wr.rdma.rkey = remote[b].keys[index];
            Require(!ibv_post_send(rail.qp, &wr, &bad), "FastKV: ibv_post_send failed");
            ++rail.outstanding;
            total += n;
        };
        struct Pending {
            size_t payload, span;
            size_t offset = 0, spanOffset = 0;
            Registrations direct;
        };
        std::deque<Pending> pending;
        size_t span = 0;
        for (size_t i = 0; i < payloads.size(); ++i) {
            const auto &payload = payloads[i];
            if (!payload.Size()) continue;
            while (span < remote.size() && !remote[span].size) ++span;
            Pending work{i, span, 0, 0, {}};
            for (size_t left = payload.Size(); left;) {
                Require(span < remote.size() && remote[span].size && remote[span].size <= left,
                        "FastKV: invalid RDMA destination span");
                left -= remote[span++].size;
            }
            // A zero device budget explicitly disallows transport-owned staging.
            if (payload.Kind() == MemoryKind::Host || !windows.Limit(payload.Device()) || Registered(payload))
                work.direct = Register(payload);
            else if (payload.Preallocated())
                work.direct = Register(payload, true);
            pending.push_back(std::move(work));
        }
        // Give each payload a turn so different GPUs can issue DMA on their
        // local NICs concurrently. Copy an unregistered GPU source once per
        // window, independently of how the peer fragmented its destination.
        while (!pending.empty()) {
            auto work = std::move(pending.front());
            pending.pop_front();
            const auto &payload = payloads[work.payload];
            const auto &selected = SendRails(payload);
            size_t n = payload.Size() - work.offset, local = work.offset;
            const Buffer *source = &payload;
            const Registrations *regs = &work.direct;
            if (work.direct.empty()) {
                auto &window = Window(payload.Device(), windows, socket, deadline);
                if (window.used == window.buffer.Size()) Drain(socket, deadline);
                n = std::min(n, window.buffer.Size() - window.used);
                local = window.used;
                window.buffer.CopyFrom(payload, work.offset, local, n);
                window.used += n;
                source = &window.buffer;
                regs = &window.regs;
            } else {
                n = std::min(n, size_t(1) << 20);
            }
            for (size_t sent = 0; sent < n;) {
                const size_t left = size_t(remote[work.span].size) - work.spanOffset;
                const auto chunk = uint32_t(std::min({size_t(1) << 20, n - sent, left}));
                post(*source, local + sent, work.span, work.spanOffset, chunk, selected, *regs);
                sent += chunk;
                work.spanOffset += chunk;
                if (work.spanOffset == remote[work.span].size) { ++work.span; work.spanOffset = 0; }
            }
            work.offset += n;
            if (work.offset < payload.Size()) pending.push_back(std::move(work));
        }
        Drain(socket, deadline);
        sendWindows.clear();
        Writer done;
        done.U32(Done);
        done.U64(total);
        socket.SendFrame(done.bytes, deadline);
    }
    void Receive(const Socket &socket, const std::vector<Buffer> &buffers, Deadline deadline) override {
        held.insert(held.end(), buffers.begin(), buffers.end());
        Writer w;
        w.U32(uint32_t(buffers.size()));
        w.U32(uint32_t(rails.size()));
        uint64_t total = 0;
        for (auto &b : buffers) {
            w.U64(uint64_t(reinterpret_cast<uintptr_t>(b.MutableData())));
            w.U64(b.Size());
            total += b.Size();
            for (auto &r : rails)
                w.U32(b.Size() ? r->Register(b)->mr->rkey : 0);
        }
        socket.SendFrame(w.bytes, deadline);
        auto bytes = socket.ReceiveFrame(64, deadline);
        Reader done(bytes);
        Require(done.U32() == Done && done.U64() == total, "FastKV: incomplete RDMA transfer");
        done.End();
        std::set<int> visible;
        for (auto &b : buffers)
            if (b.Kind() == MemoryKind::Cuda && visible.insert(b.Device()).second)
                b.MakeRemoteWritesVisible();
    }
};
} // namespace
bool RdmaAvailable() {
    int n = 0;
    auto **devices = ibv_get_device_list(&n);
    if (devices)
        ibv_free_device_list(devices);
    return n > 0;
}
std::unique_ptr<RdmaConnection> ConnectRdma(const Socket &socket, const Options &o, Deadline deadline) {
    auto connection = std::make_unique<Connection>();
    std::vector<std::string> names = o.rdmaDevices;
    if (names.empty())
        names.emplace_back();
    Require(names.size() <= 16, "FastKV: too many RDMA rails");
    for (auto &name : names)
        connection->rails.push_back(CreateRail(name, o));
    Writer w;
    w.U32(uint32_t(connection->rails.size()));
    for (auto &r : connection->rails) {
        w.U32(r->local.qpn);
        w.U32(r->local.psn);
        w.U32(r->local.lid);
        w.U32(r->local.mtu);
        w.Blob(r->local.gid.raw, 16);
    }
    socket.SendFrame(w.bytes, deadline);
    auto bytes = socket.ReceiveFrame(4096, deadline);
    Reader reader(bytes);
    size_t count = reader.U32();
    Require(count > 0 && count <= 16, "FastKV: invalid RDMA rail count");
    std::vector<Endpoint> peers;
    for (size_t i = 0; i < count; ++i) {
        Endpoint p;
        p.qpn = reader.U32();
        p.psn = reader.U32();
        p.lid = reader.U32();
        p.mtu = reader.U32();
        auto gid = reader.Blob(16);
        Require(gid.size() == 16, "FastKV: malformed GID");
        std::memcpy(p.gid.raw, gid.data(), 16);
        peers.push_back(p);
    }
    reader.End();
    size_t used = std::min(count, connection->rails.size());
    connection->rails.resize(used);
    for (size_t i = 0; i < used; ++i)
        ConnectRail(*connection->rails[i], peers[i]);
    Writer ready;
    ready.U32(Done);
    socket.SendFrame(ready.bytes, deadline);
    auto ack = socket.ReceiveFrame(64, deadline);
    Reader r(ack);
    Require(r.U32() == Done, "FastKV: peer QP not ready");
    r.End();
    return connection;
}
#endif
} // namespace fastkv
