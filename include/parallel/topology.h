#ifndef FASTLLM_PARALLEL_TOPOLOGY_H
#define FASTLLM_PARALLEL_TOPOLOGY_H

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fastllm { namespace parallel {

struct Group {
    std::string name;
    std::vector<int> ranks; // logical ranks, never CUDA ordinals
    int Rank(int worldRank) const {
        auto it = std::find(ranks.begin(), ranks.end(), worldRank);
        return it == ranks.end() ? -1 : int(it - ranks.begin());
    }
};

// Attention and experts independently partition the same device domain.
class Topology {
public:
    const std::vector<int> devices;
    const int attentionDP, attentionTP, expertEP, expertTP;
    std::vector<Group> groups;
    std::vector<int> attention, expertTensor, expertParallel;
    int leaders, world;

    Topology(std::vector<int> devices, int attentionDP, int expertEP)
        : devices(Check(std::move(devices), attentionDP, expertEP)),
          attentionDP(attentionDP), attentionTP(this->devices.size() / attentionDP),
          expertEP(expertEP), expertTP(this->devices.size() / expertEP) {
        for (int d = 0; d < attentionDP; ++d)
            attention.push_back(Add("attention:" + std::to_string(d), Range(d * attentionTP, attentionTP)));
        std::vector<int> roots;
        for (int d = 0; d < attentionDP; ++d) roots.push_back(d * attentionTP);
        leaders = Add("leaders", roots);
        for (int e = 0; e < expertEP; ++e)
            expertTensor.push_back(Add("expert_tp:" + std::to_string(e), Range(e * expertTP, expertTP)));
        for (int t = 0; t < expertTP; ++t) {
            std::vector<int> members;
            for (int e = 0; e < expertEP; ++e) members.push_back(e * expertTP + t);
            expertParallel.push_back(Add("expert_ep:" + std::to_string(t), members));
        }
        world = Add("world", Range(0, this->devices.size()));
    }
    int Size() const { return devices.size(); }
    int AttentionOwner(int rank) const { CheckRank(rank); return rank / attentionTP; }
    int AttentionRank(int rank) const { CheckRank(rank); return rank % attentionTP; }
private:
    void CheckRank(int rank) const {
        if (rank < 0 || rank >= Size()) throw std::out_of_range("parallel world rank");
    }
    static std::vector<int> Check(std::vector<int> devices, int dp, int ep) {
        if (devices.empty() || dp <= 0 || ep <= 0 || devices.size() % dp || devices.size() % ep ||
            std::set<int>(devices.begin(), devices.end()).size() != devices.size() ||
            *std::min_element(devices.begin(), devices.end()) < 0)
            throw std::invalid_argument("invalid parallel devices, attention DP, or expert EP");
        return devices;
    }
    static std::vector<int> Range(int first, int size) {
        std::vector<int> values(size);
        std::iota(values.begin(), values.end(), first);
        return values;
    }
    int Add(std::string name, std::vector<int> ranks) {
        groups.push_back({std::move(name), std::move(ranks)});
        return groups.size() - 1;
    }
};

// A step-local permutation. TP replicas share this map and are never counted
// as additional rows. A request with several tokens repeats its owner per row.
class RowPartition {
public:
    std::vector<std::vector<int>> localToGlobal;
    std::vector<int> globalToPadded;
    int bucket = 0;
    RowPartition(int groups, const std::vector<int> &owners) : localToGlobal(groups > 0 ? groups : 0) {
        if (groups <= 0) throw std::invalid_argument("row partition needs groups");
        for (size_t i = 0; i < owners.size(); ++i) {
            if (owners[i] < 0 || owners[i] >= groups) throw std::invalid_argument("invalid row owner");
            localToGlobal[owners[i]].push_back(i);
        }
        for (const auto &rows : localToGlobal) bucket = std::max(bucket, int(rows.size()));
        globalToPadded.resize(owners.size());
        for (int d = 0; d < groups; ++d)
            for (size_t i = 0; i < localToGlobal[d].size(); ++i)
                globalToPadded[localToGlobal[d][i]] = d * bucket + i;
    }
    int Rows() const { return globalToPadded.size(); }
    int PaddedRows() const { return bucket * localToGlobal.size(); }
};

}} // namespace fastllm::parallel
#endif
