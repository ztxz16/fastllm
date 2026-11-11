// CPU-only regression for native image host-memory planning.
#include "../../src/models/minimax_h3_image/memory.h"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

namespace fs = std::filesystem;
using fastllm::minimax_h3_image::detail::availableHostMemory;
namespace {
constexpr uint64_t GiB = uint64_t(1) << 30;
void check(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}
void write(const fs::path &path, const std::string &text) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    out << text;
    check(bool(out), "cannot write fixture");
}
std::string escaped(const fs::path &path) {
    std::string result;
    for (char c : path.string())
        result += c == ' ' ? "\\040" : std::string(1, c);
    return result;
}
struct Fixture {
    fs::path root, proc, mount;
    Fixture() {
        std::mt19937_64 random(std::random_device{}());
        for (int i = 0; i < 32; ++i) {
            root = fs::temp_directory_path() / ("fastllm-image-memory-" + std::to_string(random()));
            if (fs::create_directory(root))
                break;
            if (i == 31)
                throw std::runtime_error("cannot create test directory");
        }
        proc = root / "proc";
        mount = root / "memory mount";
        write(proc / "meminfo", "MemTotal: 134217728 kB\nMemAvailable: 67108864 kB\n");
    }
    ~Fixture() {
        std::error_code error;
        fs::remove_all(root, error);
    }
    void layout(bool v2, const std::string &group, const std::string &mountRoot = "/") {
        write(proc / "self/cgroup",
              v2 ? "0::" + group + "\n" : "5:cpu,cpuacct:/other\n6:memory:" + group + "\n");
        write(proc / "self/mountinfo", "31 20 0:25 " + mountRoot + " " + escaped(mount) +
                                           " rw,nosuid shared:7 - " +
                                           (v2 ? "cgroup2 cgroup rw\n" : "cgroup cgroup rw,memory\n"));
    }
    void limit(bool v2, const std::string &group, uint64_t bytes, uint64_t used) {
        write(mount / group / (v2 ? "memory.max" : "memory.limit_in_bytes"), std::to_string(bytes));
        write(mount / group / (v2 ? "memory.current" : "memory.usage_in_bytes"), std::to_string(used));
    }
    size_t available() { return availableHostMemory(proc.string()); }
};
} // namespace
int main() {
    try {
        Fixture f;
        check(f.available() == 64 * GiB, "missing cgroup metadata must preserve host availability");
        f.layout(true, "/user.slice/session.scope");
        f.limit(true, "user.slice/session.scope", 4 * GiB, GiB);
        f.limit(true, "user.slice", 8 * GiB, 6 * GiB);
        write(f.mount / "memory.max", "max");
        write(f.mount / "memory.current", "0");
        check(f.available() == 2 * GiB, "nested v2 parent quota must include sibling usage");
        f.limit(true, "user.slice/session.scope", 4 * GiB, 5 * GiB);
        check(f.available() == 0, "over-limit usage must saturate at zero");
        f.layout(true, "/docker/id/worker", "/docker/id");
        f.limit(true, "worker", 3 * GiB, GiB);
        check(f.available() == 2 * GiB, "bind-mounted cgroup root must be translated");
        f.layout(true, "/");
        f.limit(true, "", 4 * GiB, GiB);
        check(f.available() == 3 * GiB, "private cgroup namespace root");
        f.layout(false, "/docker/id/worker", "/docker/id");
        f.limit(false, "", 6 * GiB, 5 * GiB);
        f.limit(false, "worker", 4 * GiB, GiB);
        check(f.available() == GiB, "v1 memory controller and parent quota");
        write(f.proc / "meminfo", "MemAvailable: 524288 kB\n");
        check(f.available() == GiB / 2, "host availability must also constrain cgroup quota");
        f.layout(true, "/unrelated", "/other");
        check(f.available() == GiB / 2, "ignore a mount that does not expose this process");
        write(f.proc / "meminfo", "MemAvailable: unknown kB\n");
        check(f.available() == 0, "unknown host availability must disable caching safely");
        std::cout << "PASS host memory, nested cgroups v1/v2, mount roots, ancestor quotas and saturation\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
