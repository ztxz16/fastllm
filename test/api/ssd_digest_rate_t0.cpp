// T0 micro-probe: measure this library's Digest (SHA-256) rate.
#include <chrono>
#include <cstdio>
#include <string>

#include "utils/disk_prefix_cache.h"

int main() {
    const std::string buffer(256ull << 20, 'a');  // 256 MiB
    fastllm::DiskPrefixCache::Digest(buffer);    // warm up
    const int rounds = 4;
    const auto start = std::chrono::steady_clock::now();
    for (int round = 0; round < rounds; ++round) fastllm::DiskPrefixCache::Digest(buffer);
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const double bytes = double(buffer.size()) * rounds;
    std::printf("{\"bytes\":%.0f,\"seconds\":%.3f,\"mib_per_s\":%.1f,\"mb_per_s\":%.1f}\n",
                bytes, seconds, bytes / seconds / (1024.0 * 1024.0), bytes / seconds / 1e6);
    return 0;
}
