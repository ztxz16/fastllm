// Linux host-local replica admission and shared request buffers.
#include "fastipc.h"

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <fcntl.h>
#include <limits>
#include <list>
#include <mutex>
#include <new>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
struct Rank { int capacity, pending = 0, assigned = 0; bool online = true; };
struct Session { size_t rank; int users; Clock::time_point seen; std::list<std::string>::iterator order; };
}

struct FastIpcRouter {
    std::mutex mutex;
    std::vector<Rank> ranks;
    std::unordered_map<std::string, Session> sessions;
    std::list<std::string> idle, busy;
    size_t cursor = 0, limit;
    int queue;
    std::chrono::duration<double> ttl;

    void expire(bool makeRoom) {
        auto now = Clock::now();
        while (!idle.empty()) {
            auto it = sessions.find(idle.front());
            if ((!makeRoom || sessions.size() < limit) && now - it->second.seen < ttl) break;
            --ranks[it->second.rank].assigned;
            idle.pop_front();
            sessions.erase(it);
        }
    }
    void releaseSession(const std::string &key) {
        auto it = sessions.find(key);
        if (it != sessions.end() && --it->second.users == 0) {
            it->second.seen = Clock::now();
            idle.splice(idle.end(), busy, it->second.order);
        }
    }
    // Compare load fractions without rounding, retaining cursor order for ties.
    bool prefer(const Rank &a, const Rank &b, bool balanceSessions) const {
        if ((a.pending < a.capacity) != (b.pending < b.capacity))
            return a.pending < a.capacity;
        if (balanceSessions) {
            int64_t left = int64_t(a.assigned) * b.capacity;
            int64_t right = int64_t(b.assigned) * a.capacity;
            if (left != right) return left < right;
        }
        return int64_t(a.pending) * b.capacity < int64_t(b.pending) * a.capacity;
    }

    int acquire(const std::string &key) {
        std::lock_guard<std::mutex> lock(mutex);
        expire(false);
        auto it = sessions.find(key);
        size_t best = ranks.size();
        if (it != sessions.end()) {
            best = it->second.rank;
            if (!ranks[best].online) return -EHOSTDOWN;
        }
        // An active session stays on its owner. Between turns it may move
        // to an available replica instead of waiting behind a full owner.
        if (best == ranks.size() || (!it->second.users && ranks[best].pending >= ranks[best].capacity)) {
            bool anyOnline = false;
            for (size_t n = 0; n < ranks.size(); ++n) {
                size_t i = (n + cursor) % ranks.size();
                const auto &r = ranks[i];
                if (!r.online) continue;
                anyOnline = true;
                if (r.pending >= r.capacity + queue) continue;
                if (it != sessions.end() && r.pending >= r.capacity) continue;
                if (best == ranks.size() || prefer(r, ranks[best], it == sessions.end() && !key.empty()))
                    best = i;
            }
            if (!anyOnline) return -EHOSTDOWN;
        }
        if (best == ranks.size() || ranks[best].pending >= ranks[best].capacity + queue)
            return -EAGAIN;
        auto &r = ranks[best];
        if (!key.empty()) {
            if (it == sessions.end()) {
                expire(true);
                if (sessions.size() >= limit) return -ENOSPC;
                busy.push_back(key);
                try { sessions.emplace(key, Session{best, 1, Clock::now(), std::prev(busy.end())}); }
                catch (...) { busy.pop_back(); throw; }
                ++r.assigned;
            } else if (it->second.users++ == 0) {
                if (best != it->second.rank) {
                    --ranks[it->second.rank].assigned;
                    ++r.assigned;
                    it->second.rank = best;
                }
                busy.splice(busy.end(), idle, it->second.order);
            }
        }
        cursor = (best + 1) % ranks.size();
        ++r.pending;
        return int(best);
    }
};

extern "C" {
FastIpcRouter *fastipc_router_create(const int *caps, int count, int queue, int sessions, double ttl) {
    if (!caps || count < 1 || queue < 0 || sessions < 1 || ttl <= 0) return nullptr;
    for (int i = 0; i < count; ++i)
        if (caps[i] < 1 || caps[i] > std::numeric_limits<int>::max() - queue) return nullptr;
    auto *router = new(std::nothrow) FastIpcRouter;
    if (!router) return nullptr;
    try {
        for (int i = 0; i < count; ++i) router->ranks.push_back({caps[i]});
        router->limit = sessions; router->queue = queue;
        router->ttl = std::chrono::duration<double>(ttl);
        return router;
    } catch (...) { delete router; return nullptr; }
}
void fastipc_router_destroy(FastIpcRouter *r) { delete r; }
int fastipc_reserve(FastIpcRouter *r, const char *key) {
    try { return r->acquire(key); } catch (const std::bad_alloc &) { return -ENOMEM; }
}
void fastipc_release(FastIpcRouter *r, int rank, const char *key) {
    std::lock_guard<std::mutex> lock(r->mutex);
    --r->ranks[rank].pending;
    r->releaseSession(key);
}
int fastipc_forget(FastIpcRouter *r, const char *key) {
    std::lock_guard<std::mutex> lock(r->mutex);
    auto it = r->sessions.find(key);
    if (it == r->sessions.end()) return 0;
    if (it->second.users) return -EBUSY;
    --r->ranks[it->second.rank].assigned;
    r->idle.erase(it->second.order);
    r->sessions.erase(it);
    return 0;
}
void fastipc_rank_offline(FastIpcRouter *r, int rank) {
    std::lock_guard<std::mutex> lock(r->mutex);
    r->ranks[rank].online = false;
}
void fastipc_rank_state(FastIpcRouter *r, int rank, int *values) {
    std::lock_guard<std::mutex> lock(r->mutex);
    r->expire(false);
    const auto &v = r->ranks[rank];
    values[0] = v.pending; values[1] = v.assigned;
}
int fastipc_create_buffer(size_t bytes) {
    // The syscall is also available on Linux distributions predating the glibc wrapper.
    int fd = syscall(SYS_memfd_create, "ftllm-request", 3U /* CLOEXEC | ALLOW_SEALING */);
    if (fd < 0) return -errno;
    if (ftruncate(fd, bytes) < 0 || fcntl(fd, F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL) < 0) {
        int error = errno; close(fd); return -error;
    }
    return fd;
}
}
