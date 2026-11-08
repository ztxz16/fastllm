// T0 probe (2026-10-17): measure v4 load-path read amplification and CPU cost.
//
// User restore (117760 tokens) spent read_ms=18478.005 for a 4.00 GB payload
// while /proc/<pid>/io showed rchar ~31.66 GB and read_bytes ~4.62 GB in the
// bracket. That points at repeated logical reads (per-worker record rereads +
// SHA-256) rather than raw SSD bandwidth. This probe drives the real store
// Load() path on a copied, isolated v4 store and reports:
//   - bytes requested from the kernel (rchar) vs payload returned
//   - CPU time vs wall time (hashing/copying vs I/O wait)
//   - open vs load split
// No model, no GPU, no user SSD, no service operation. Read-only on a copy.
#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "utils/disk_prefix_cache.h"

namespace {

struct IoCounters {
    unsigned long long rchar = 0;
    unsigned long long read_bytes = 0;
};

IoCounters ReadIo() {
    IoCounters counters;
    FILE *file = std::fopen("/proc/self/io", "r");
    if (file == nullptr) return counters;
    char line[256];
    while (std::fgets(line, sizeof(line), file) != nullptr) {
        unsigned long long value = 0;
        if (std::sscanf(line, "rchar: %llu", &value) == 1) {
            counters.rchar = value;
        } else if (std::sscanf(line, "read_bytes: %llu", &value) == 1) {
            counters.read_bytes = value;
        }
    }
    std::fclose(file);
    return counters;
}

double CpuMs(const rusage &usage, bool user) {
    const timeval &tv = user ? usage.ru_utime : usage.ru_stime;
    return double(tv.tv_sec) * 1000.0 + double(tv.tv_usec) / 1000.0;
}

double NowMs() {
    using clock = std::chrono::steady_clock;
    static const clock::time_point start = clock::now();
    return std::chrono::duration<double, std::milli>(clock::now() - start).count();
}

// Evict our own store copy from the page cache so the next run is cold.
// Scoped to the copied file; never touches the user's store.
void EvictFromPageCache(const std::string &path) {
    int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return;
    // Dirty pages from the copy cannot be dropped until written back.
    ::fsync(fd);
    ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    ::close(fd);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: %s <prefix_cache_dir> <key> <expected_identity> [evict=1]\n",
                     argv[0]);
        return 2;
    }
    const std::string root = argv[1];
    const std::string key = argv[2];
    const std::string identity = argv[3];
    const bool evict = argc > 4 ? std::atoi(argv[4]) != 0 : true;

    const IoCounters io0 = ReadIo();
    rusage usage0{};
    ::getrusage(RUSAGE_SELF, &usage0);
    const double t0 = NowMs();

    std::string error;
    std::unique_ptr<fastllm::DiskPrefixV4Store> store =
        fastllm::DiskPrefixV4Store::Open(root, 128ull << 30, error);
    const double t1 = NowMs();
    const IoCounters ioOpen = ReadIo();
    if (!store) {
        std::printf("{\"ok\":false,\"stage\":\"open\",\"error\":\"%s\"}\n", error.c_str());
        return 1;
    }

    std::vector<std::string> objects;
    std::vector<std::string> hashes;
    const bool ok = store->Load(key, identity, objects, hashes, error);
    const double t2 = NowMs();
    const IoCounters ioLoad = ReadIo();
    rusage usage1{};
    ::getrusage(RUSAGE_SELF, &usage1);

    unsigned long long payload = 0;
    for (const std::string &object : objects) payload += object.size();
    // 顺序/内容指纹：把 outHashes 按序拼起来再取摘要，用于基线与候选逐字节比对。
    std::string joinedHashes;
    joinedHashes.reserve(hashes.size() * 64);
    for (const std::string &hash : hashes) joinedHashes += hash;
    const std::string hashOrderDigest = fastllm::DiskPrefixCache::Digest(joinedHashes);

    const char *threads = std::getenv("FASTLLM_PREFIX_CACHE_LOAD_THREADS");
    std::printf(
        "{\"ok\":%s,\"error\":\"%s\",\"objects\":%zu,\"hashes\":%zu,"
        "\"payload_bytes\":%llu,\"open_ms\":%.3f,\"load_ms\":%.3f,"
        "\"rchar_open\":%llu,\"rchar_load\":%llu,"
        "\"read_bytes_open\":%llu,\"read_bytes_load\":%llu,"
        "\"cpu_user_ms\":%.3f,\"cpu_sys_ms\":%.3f,\"maxrss_kb\":%ld,"
        "\"load_threads\":\"%s\",\"store\":\"%s\",\"hash_order_digest\":\"%s\"}\n",
        ok ? "true" : "false", error.c_str(), objects.size(), hashes.size(), payload,
        t1 - t0, t2 - t1,
        ioOpen.rchar - io0.rchar, ioLoad.rchar - ioOpen.rchar,
        ioOpen.read_bytes - io0.read_bytes, ioLoad.read_bytes - ioOpen.read_bytes,
        CpuMs(usage1, true) - CpuMs(usage0, true),
        CpuMs(usage1, false) - CpuMs(usage0, false),
        usage1.ru_maxrss, threads != nullptr ? threads : "(default)", root.c_str(),
        hashOrderDigest.c_str());

    if (evict) EvictFromPageCache(root + "/v4/control.log");
    return ok ? 0 : 1;
}
