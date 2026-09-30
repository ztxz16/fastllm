#pragma once
#include <algorithm>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace fastllm { namespace mtp_shortlist {
struct Shard {
    std::vector<int> rows;
    std::vector<int> tokenIds;
    int logicalSize = 0;
};

// Compact IDs offset by the original contiguous shard start preserve global
// token ordering across shards. Map only logical rows; padding is never valid.
inline int MapProxyId(int proxy, int shardStart, const std::vector<int> &ids) {
    const long long index = (long long)proxy - shardStart;
    return index >= 0 && (size_t)index < ids.size() ? ids[(size_t)index] : -1;
}

inline bool Read(const char *path, int vocab, std::vector<int> &ids) {
    ids.clear();
    if (!path || !*path || std::string(path) == "0" || vocab <= 0) return false;
    std::ifstream file(path);
    std::string token;
    while (file >> token) {
        size_t used = 0;
        long long id;
        try { id = std::stoll(token, &used); }
        catch (...) { ids.clear(); return false; }
        if (used != token.size() || id < 0 || id >= vocab || ids.size() >= size_t(vocab)) {
            ids.clear(); return false;
        }
        ids.push_back(int(id));
    }
    std::sort(ids.begin(), ids.end());
    if (!file.eof() || ids.empty() || ids.size() >= size_t(vocab) ||
        std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
        ids.clear(); return false;
    }
    return true;
}

// Source rows are concatenated in range order, which need not be ID order.
// Output rows are in ascending global-ID order so local argmax ties remain
// consistent with full-vocabulary greedy sampling. Padding repeats the last
// selected row: it never adds a candidate token or changes its score.
inline bool Project(const std::vector<int> &ids,
                    const std::vector<std::pair<int, int>> &ranges,
                    int sourceRows, int vocab, Shard &out, int alignment = 128) {
    out = Shard{};
    if (sourceRows < 0 || vocab <= 0 || alignment <= 0 || ids.empty() ||
        !std::is_sorted(ids.begin(), ids.end()) || ids.front() < 0 || ids.back() >= vocab ||
        std::adjacent_find(ids.begin(), ids.end()) != ids.end()) return false;
    long long size = 0;
    for (size_t i = 0; i < ranges.size(); ++i) {
        auto r = ranges[i];
        if (r.first < 0 || r.first > r.second || r.second > vocab) return false;
        for (size_t j = 0; j < i; ++j)
            if (std::max(r.first, ranges[j].first) < std::min(r.second, ranges[j].second)) return false;
        size += r.second - r.first;
    }
    if (size != sourceRows) return false;
    for (int id : ids) {
        int offset = 0;
        for (auto r : ranges) {
            if (id >= r.first && id < r.second) {
                out.rows.push_back(offset + id - r.first);
                out.tokenIds.push_back(id);
                break;
            }
            offset += r.second - r.first;
        }
    }
    out.logicalSize = int(out.rows.size());
    if (out.rows.empty()) return true;
    const size_t padded = (out.rows.size() + alignment - 1) / alignment * alignment;
    if (padded > size_t(sourceRows)) { out = Shard{}; return false; }
    out.rows.resize(padded, out.rows.back());
    out.tokenIds.resize(padded, out.tokenIds.back());
    return true;
}
}} // namespace fastllm::mtp_shortlist
