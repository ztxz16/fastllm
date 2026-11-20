#include "utils/prefill_batch.h"
#include <iostream>
#include <numeric>

using namespace fastllm;
static void Check(bool value, const char *message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    try {
        // SWE-style prompts all share a short prefix resident on GPU7. Even
        // before KV capacity is exhausted, all eight groups must receive work.
        std::vector<size_t> available(8, 4194304), work(8, 0);
        std::vector<int> requests(8, 0);
        for (int i = 0; i < 64; ++i) {
            int group = ChoosePDPrefillGroup(7, available, 8192, work, 8192, 2816);
            Check(group >= 0, "unexpected admission failure");
            ++requests[group]; available[group] -= 8192;
            work[group] += 8192 - (group == 7 ? 2816 : 0);
        }
        Check(*std::min_element(requests.begin(), requests.end()) > 0, "shared prefix starved a DP group");
        Check(*std::max_element(work.begin(), work.end()) - *std::min_element(work.begin(), work.end()) <= 8192,
              "shared prefix caused an unbounded work imbalance");
        // Keep a valuable long prefix when waiting costs less than recomputing.
        work.assign(8, 0); work[7] = 6000;
        Check(ChoosePDPrefillGroup(7, available, 16384, work, 16384, 16000) == 7,
              "discarded a beneficial long prefix");
        work[7] = 50000;
        Check(ChoosePDPrefillGroup(7, available, 16384, work, 16384, 16000) != 7,
              "long prefix forced waiting beyond its recomputation cost");
        available.assign(8, 0); available[3] = 16384;
        Check(ChoosePDPrefillGroup(7, available, 16384, work, 16384, 16000) == 3,
              "placement ignored actual capacity");
        Check(ChoosePDPrefillGroup(7, available, 16385, work, 16384, 16000) == -1,
              "placement overbooked capacity");
        Check(ChoosePDPrefillGroup(0, {128}, 128, {1000}, 128, 0) == 0,
              "single-group admission changed");
        Check(ChoosePDPrefillGroup(1, {128,128}, 128, {0,0}, 128, 0) == 1,
              "equal-cost placement lost its existing owner");
        std::cout << "PASS PD shared-prefix distribution, cache benefit, capacity and ownership ties\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
