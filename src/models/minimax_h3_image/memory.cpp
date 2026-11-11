#include "memory.h"
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

namespace fastllm::minimax_h3_image::detail {
namespace {
bool contains(const std::string &list, const std::string &value) {
    return ("," + list + ",").find("," + value + ",") != std::string::npos;
}
std::string unescape(const std::string &path) {
    std::string result;
    for (size_t i = 0; i < path.size(); ++i) {
        if (path[i] == '\\' && i + 3 < path.size() && path[i + 1] >= '0' && path[i + 1] <= '7' &&
            path[i + 2] >= '0' && path[i + 2] <= '7' && path[i + 3] >= '0' && path[i + 3] <= '7') {
            result += char((path[i + 1] - '0') * 64 + (path[i + 2] - '0') * 8 + path[i + 3] - '0');
            i += 3;
        } else {
            result += path[i];
        }
    }
    return result;
}
std::filesystem::path normalized(std::filesystem::path path) {
    path = path.lexically_normal();
    if (path.has_relative_path() && path.filename().empty())
        path = path.parent_path();
    return path;
}
} // namespace

size_t availableHostMemory(const std::string &procRoot) {
    namespace fs = std::filesystem;
    uint64_t available = 0;
    std::ifstream info(procRoot + "/meminfo");
    std::string line;
    while (std::getline(info, line)) {
        if (line.compare(0, 13, "MemAvailable:") == 0) {
            uint64_t kib = 0;
            if (std::istringstream(line.substr(13)) >> kib)
                available = kib * 1024;
            break;
        }
    }
    std::string unified, memory;
    std::ifstream groups(procRoot + "/self/cgroup");
    while (std::getline(groups, line)) {
        auto first = line.find(':'), second = line.find(':', first + 1);
        if (first == std::string::npos || second == std::string::npos)
            continue;
        auto controllers = line.substr(first + 1, second - first - 1);
        if (controllers.empty())
            unified = line.substr(second + 1);
        if (contains(controllers, "memory"))
            memory = line.substr(second + 1);
    }
    std::ifstream mounts(procRoot + "/self/mountinfo");
    while (std::getline(mounts, line)) {
        std::istringstream fields(line);
        std::vector<std::string> tokens;
        for (std::string token; fields >> token;)
            tokens.push_back(token);
        auto separator = std::find(tokens.begin(), tokens.end(), "-");
        if (tokens.size() < 10 || separator < tokens.begin() + 6 || tokens.end() - separator < 4)
            continue;
        bool v2 = separator[1] == "cgroup2";
        if (!v2 && (separator[1] != "cgroup" || !contains(separator[3], "memory")))
            continue;
        fs::path group(v2 ? unified : memory), mount(unescape(tokens[4])), root(unescape(tokens[3]));
        if (!group.is_absolute() || !mount.is_absolute() || !root.is_absolute())
            continue;
        auto relative = normalized(group).lexically_relative(normalized(root));
        // Ignore mounts that do not expose this process's cgroup. In a private
        // cgroup namespace both the membership and mount root may simply be '/'.
        if (relative.empty() || *relative.begin() == "..")
            continue;
        mount = normalized(mount);
        auto directory = normalized(mount / relative);
        for (;;) {
            uint64_t limit = 0, used = 0;
            std::ifstream maximum(directory / (v2 ? "memory.max" : "memory.limit_in_bytes"));
            std::ifstream current(directory / (v2 ? "memory.current" : "memory.usage_in_bytes"));
            if ((maximum >> limit) && (current >> used))
                available = std::min(available, limit > used ? limit - used : 0);
            // Parent quotas include sibling usage and can be tighter than the
            // leaf's limit. Do not walk outside the visible cgroup mount.
            if (directory == mount)
                break;
            directory = directory.parent_path();
        }
    }
    return size_t(available);
}
} // namespace fastllm::minimax_h3_image::detail
