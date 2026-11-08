// HOSTTRIM: periodic glibc malloc_trim(0) on a background thread (FASTLLM_HOST_MALLOC_TRIM_SECONDS=N, default 10 s, 0 = off).
//
// Why: SSD checkpoint saves allocate and free many large host blocks on many threads. glibc keeps freed memory
// inside its per-thread arenas (the dynamic mmap threshold moves these blocks out of mmap after the first
// frees), so process RSS keeps the high-water mark of all arenas. Measured on 2026-11-03
// (logs/a6-mallocinfo-20261103/REPORT.txt): after the long-context request set, malloc_info reported 10.9 GiB
// free inside the arenas and one malloc_trim(0) lowered RSS+swap from 22158 MiB to 13817 MiB.
// fastllm already trims once after weight load (src/model.cpp); this repeats it while serving.
//
// Each round prints one line: "[HostMem] malloc-trim: n= rss_before_mib= rss_after_mib= released_mib= trim_ms=".
#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#if defined(__linux__) && defined(__GLIBC__)
#include <malloc.h>
#endif

namespace fastllm {

// Strict parse: only a plain positive integer number of seconds enables the trimmer.
inline int HostMallocTrimSeconds() {
    const char *text = std::getenv("FASTLLM_HOST_MALLOC_TRIM_SECONDS");
    if (text == nullptr || *text == '\0') return 10;   // default on since 2026-11-08; "0" turns it off
    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == nullptr || *end != '\0' || value <= 0 || value > 3600) return 0;
    return (int)value;
}

// Resident + swapped bytes of this process, from /proc/self/status (VmRSS + VmSwap). 0 if unreadable.
inline size_t HostProcessRssSwapBytes() {
    FILE *file = std::fopen("/proc/self/status", "r");
    if (file == nullptr) return 0;
    char line[256];
    size_t kib = 0;
    while (std::fgets(line, sizeof line, file)) {
        unsigned long long value = 0;
        if (std::sscanf(line, "VmRSS: %llu kB", &value) == 1 || std::sscanf(line, "VmSwap: %llu kB", &value) == 1)
            kib += (size_t)value;
    }
    std::fclose(file);
    return kib * 1024;
}

// Starts the trimmer thread once per process when the flag is set; otherwise does nothing.
inline void HostMallocTrimStartOnce() {
#if defined(__linux__) && defined(__GLIBC__)
    static std::once_flag once;
    std::call_once(once, [] {
        const int seconds = HostMallocTrimSeconds();
        if (seconds <= 0) return;
        std::printf("[HostMem] malloc-trim enabled: every %d s (FASTLLM_HOST_MALLOC_TRIM_SECONDS).\n", seconds);
        std::fflush(stdout);
        std::thread([seconds] {
            for (unsigned long long round = 1;; ++round) {
                std::this_thread::sleep_for(std::chrono::seconds(seconds));
                const size_t before = HostProcessRssSwapBytes();
                const auto begin = std::chrono::steady_clock::now();
                malloc_trim(0);
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
                const size_t after = HostProcessRssSwapBytes();
                std::printf("[HostMem] malloc-trim: n=%llu rss_before_mib=%zu rss_after_mib=%zu released_mib=%lld trim_ms=%.1f.\n",
                            round, before >> 20, after >> 20, (long long)(before >> 20) - (long long)(after >> 20), ms);
                std::fflush(stdout);
            }
        }).detach();
    });
#endif
}

}  // namespace fastllm
