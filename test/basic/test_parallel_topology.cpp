#include "parallel/topology.h"
#undef NDEBUG
#include <cassert>
#include <iostream>
#include <random>
using namespace fastllm::parallel;
int main() {
    int cases = 0;
    std::mt19937 random(731);
    for (int world : {1, 2, 4, 8, 16}) for (int dp = 1; dp <= world; ++dp) {
        if (world % dp) continue;
        for (int ep = 1; ep <= world; ++ep) {
            if (world % ep) continue;
            std::vector<int> devices;
            for (int i = 0; i < world; ++i) devices.push_back(2 * i + 1);
            std::shuffle(devices.begin(), devices.end(), random);
            Topology topology(devices, dp, ep);
            for (int r = 0; r < world; ++r) {
                assert(topology.groups[topology.attention[r / topology.attentionTP]].Rank(r) == r % topology.attentionTP);
                assert(topology.groups[topology.expertTensor[r / topology.expertTP]].Rank(r) == r % topology.expertTP);
                assert(topology.groups[topology.expertParallel[r % topology.expertTP]].Rank(r) == r / topology.expertTP);
            }
            for (int n : {0, 1, 2, 7, 8, 9, 31, 32, 33, 127, 128, 129, 511, 512, 513}) {
                for (int skew = 0; skew < 2; ++skew) {
                    std::vector<int> owners(n), padded;
                    for (int &d : owners) d = skew ? dp - 1 : random() % dp;
                    RowPartition rows(dp, owners);
                    padded.assign(rows.PaddedRows(), -1);
                    for (int d = 0; d < dp; ++d)
                        for (int i = 0; i < int(rows.localToGlobal[d].size()); ++i)
                            padded[d * rows.bucket + i] = rows.localToGlobal[d][i];
                    for (int i = 0; i < n; ++i) assert(padded.at(rows.globalToPadded[i]) == i);
                    ++cases;
                }
            }
        }
    }
    auto rejects = [](auto fn) {
        bool failed = false;
        try { fn(); } catch (const std::invalid_argument&) { failed = true; }
        assert(failed);
    };
    rejects([] { Topology({}, 1, 1); });
    rejects([] { Topology({0, 0}, 1, 1); });
    rejects([] { Topology({0, 1}, 3, 1); });
    rejects([] { Topology({0, -1}, 1, 1); });
    rejects([] { RowPartition(2, {0, 2}); });
    std::cout << "PASS " << cases << " topology/row cases and 5 rejected inputs\n";
}
