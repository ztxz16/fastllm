// T0: aggregate SHA-256 throughput vs threads (2026-10-17, B21).
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "utils/disk_prefix_cache.h"

int main(int argc, char **argv) {
    const int threads = argc > 1 ? std::atoi(argv[1]) : 1;
    const size_t perThreadBytes = (argc > 2 ? std::atoi(argv[2]) : 128) * size_t(1 << 20);
    std::vector<std::string> buffers;
    buffers.reserve(threads);
    for (int i = 0; i < threads; ++i) buffers.emplace_back(perThreadBytes, char('a' + i));
    std::atomic<int> ready{0};
    std::vector<std::thread> workers;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < threads; ++i) {
        workers.emplace_back([&, i]() {
            ready.fetch_add(1);
            fastllm::DiskPrefixCache::Digest(buffers[i]);
        });
    }
    for (auto &worker : workers) worker.join();
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    const double total = double(perThreadBytes) * threads;
    std::printf("threads=%d total_MB=%.0f seconds=%.3f aggregate_MB_s=%.1f per_thread_MB_s=%.1f\n",
                threads, total / (1024.0 * 1024.0), seconds, total / seconds / (1024.0 * 1024.0),
                total / seconds / (1024.0 * 1024.0) / threads);
    return 0;
}
