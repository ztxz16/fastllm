#include "internal.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <set>
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
    uint64_t id = identities.fetch_add(1);
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
struct Rail {
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
    std::shared_ptr<Registration> Register(const Buffer &buffer) {
        auto value =
            buffer.Attachment("verbs:" + std::to_string(device->id), [d = device](void *p, size_t n) {
                auto reg = std::make_shared<Registration>();
                reg->device = d;
                reg->mr =
                    ibv_reg_mr(d->pd, p, std::max(size_t(1), n),
                               IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
                Require(reg->mr, "FastKV: ibv_reg_mr failed (check memlock and GPU peer-memory support)");
                return reg;
            });
        return std::static_pointer_cast<Registration>(value);
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
    ~Connection() override {
        // Destroy every QP before releasing any receive allocation. On error,
        // posted writes must not outlive the MR or touch a recycled slot.
        for (auto &r : rails)
            r->Close();
        held.clear();
    }
    void Send(const Socket &socket, const std::vector<Buffer> &buffers, Deadline deadline) override {
        held.insert(held.end(), buffers.begin(), buffers.end());
        auto bytes = socket.ReceiveFrame(32 << 20, deadline);
        Reader reader(bytes);
        Require(reader.U32() == buffers.size() && reader.U32() == rails.size(),
                "FastKV: RDMA destination shape mismatch");
        struct Remote {
            uint64_t address, size;
            std::vector<uint32_t> keys;
        };
        std::vector<Remote> remote;
        for (auto &buffer : buffers) {
            Remote target;
            target.address = reader.U64();
            target.size = reader.U64();
            Require(target.size == buffer.Size(), "FastKV: RDMA destination size mismatch");
            for (size_t i = 0; i < rails.size(); ++i)
                target.keys.push_back(reader.U32());
            remote.push_back(std::move(target));
        }
        reader.End();
        uint64_t total = 0, sequence = 0;
        for (size_t b = 0; b < buffers.size(); ++b) {
            if (!buffers[b].Size())
                continue;
            std::vector<std::shared_ptr<Registration>> regs;
            for (auto &r : rails)
                regs.push_back(r->Register(buffers[b]));
            for (size_t offset = 0; offset < buffers[b].Size();) {
                size_t index = sequence++ % rails.size();
                auto &rail = *rails[index];
                if (rail.outstanding >= 32)
                    rail.Poll(socket, deadline, false);
                uint32_t n = uint32_t(std::min(size_t(1 << 20), buffers[b].Size() - offset));
                ibv_sge sge{};
                sge.addr = uint64_t(reinterpret_cast<uintptr_t>(buffers[b].Data())) + offset;
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
                offset += n;
                total += n;
            }
        }
        for (auto &rail : rails)
            rail->Poll(socket, deadline, true);
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
