#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

struct Options {
    std::vector<std::string> paths;
    int threads = 1;
    int seconds = 10;
    size_t blockBytes = 16ULL << 20;
    bool direct = false;
};

static bool EndsWith(const std::string &s, const std::string &suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static void PrintUsage(const char *argv0) {
    std::cout << "Usage: " << argv0 << " [--threads N] [--seconds N] [--block-mb N] [--direct] PATH...\n"
              << "PATH can be a file or a directory. Directories are expanded to *.safetensors.\n";
}

static Options ParseArgs(int argc, char **argv) {
    Options opt;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) {
            opt.threads = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--seconds" && i + 1 < argc) {
            opt.seconds = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--block-mb" && i + 1 < argc) {
            opt.blockBytes = std::max<size_t>(1, std::stoull(argv[++i])) << 20;
        } else if (arg == "--direct") {
            opt.direct = true;
        } else if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        } else {
            opt.paths.push_back(arg);
        }
    }
    if (opt.paths.empty()) {
        PrintUsage(argv[0]);
        std::exit(1);
    }
    opt.blockBytes = ((opt.blockBytes + 4095) / 4096) * 4096;
    return opt;
}

static std::vector<std::string> ExpandPaths(const std::vector<std::string> &inputs) {
    std::vector<std::string> files;
    for (auto &path : inputs) {
        if (fs::is_directory(path)) {
            for (auto &entry : fs::directory_iterator(path)) {
                if (entry.is_regular_file()) {
                    auto s = entry.path().string();
                    if (EndsWith(s, ".safetensors")) {
                        files.push_back(s);
                    }
                }
            }
        } else if (fs::is_regular_file(path)) {
            files.push_back(path);
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

static void *AllocAligned(size_t bytes) {
    void *ptr = nullptr;
    if (posix_memalign(&ptr, 4096, bytes) != 0 || ptr == nullptr) {
        std::cerr << "posix_memalign failed\n";
        std::exit(2);
    }
    return ptr;
}

int main(int argc, char **argv) {
    Options opt = ParseArgs(argc, argv);
    auto files = ExpandPaths(opt.paths);
    if (files.empty()) {
        std::cerr << "No input files.\n";
        return 1;
    }

    std::atomic<uint64_t> bytes{0};
    std::atomic<uint64_t> reads{0};
    std::atomic<uint64_t> checksum{0};
    std::atomic<bool> start{false};
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(opt.seconds);

    std::vector<std::thread> workers;
    for (int tid = 0; tid < opt.threads; tid++) {
        workers.emplace_back([&, tid]() {
            uint8_t *buffer = (uint8_t*)AllocAligned(opt.blockBytes);
            int flags = O_RDONLY;
#ifdef O_DIRECT
            if (opt.direct) {
                flags |= O_DIRECT;
            }
#endif
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            size_t fileIndex = tid % files.size();
            while (std::chrono::steady_clock::now() < deadline) {
                const std::string &file = files[fileIndex];
                int fd = open(file.c_str(), flags);
                if (fd < 0) {
                    std::cerr << "open failed: " << file << " error=" << std::strerror(errno) << "\n";
                    fileIndex = (fileIndex + opt.threads) % files.size();
                    continue;
                }
                off_t offset = 0;
                while (std::chrono::steady_clock::now() < deadline) {
                    ssize_t ret = pread(fd, buffer, opt.blockBytes, offset);
                    if (ret < 0) {
                        std::cerr << "pread failed: " << file << " error=" << std::strerror(errno) << "\n";
                        break;
                    }
                    if (ret == 0) {
                        break;
                    }
                    bytes.fetch_add((uint64_t)ret, std::memory_order_relaxed);
                    reads.fetch_add(1, std::memory_order_relaxed);
                    checksum.fetch_add(buffer[0], std::memory_order_relaxed);
                    offset += ret;
                }
                close(fd);
                fileIndex = (fileIndex + opt.threads) % files.size();
            }
            free(buffer);
        });
    }

    auto begin = std::chrono::steady_clock::now();
    deadline = begin + std::chrono::seconds(opt.seconds);
    start.store(true, std::memory_order_release);
    for (auto &worker : workers) {
        worker.join();
    }
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - begin).count();
    double gb = bytes.load() / 1e9;

    std::cout << "files=" << files.size()
              << " threads=" << opt.threads
              << " block_mb=" << (opt.blockBytes >> 20)
              << " direct=" << (opt.direct ? "on" : "off")
              << " elapsed=" << elapsed << "s"
              << " bytes=" << bytes.load()
              << " reads=" << reads.load()
              << " checksum=" << checksum.load()
              << "\n";
    std::cout << "bandwidth=" << (gb / elapsed) << " GB/s (decimal), "
              << (gb * 1024 / elapsed) << " MiB/s\n";
    return 0;
}
