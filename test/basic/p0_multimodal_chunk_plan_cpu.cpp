#include "utils/qwen35_multimodal_chunk_plan.h"
#include <iostream>
#include <stdexcept>
static void Need(bool ok, const char *name) {
    std::cout << "{\"kind\":\"chunk_case\",\"name\":\"" << name
        << "\",\"pass\":" << (ok ? "true" : "false") << "}\n";
    if (!ok) throw std::runtime_error(name);
}
int main() {
    using fastllm::qwen35_packed::PlanPrefillChunks;
    const auto plan = PlanPrefillChunks({16384, 4096}, 2048);
    Need(plan.size() == 8, "16k_4k_has_eight_bounded_rounds");
    int left = 0, right = 0;
    for (const auto &round : plan)
        Need(round.size() == 2 && round[0] >= 2 && round[0] <= 2048 &&
             round[1] >= 2 && round[1] <= 2048, "ragged_round_inside_kernel_cap");
    for (const auto &round : plan) { left += round[0]; right += round[1]; }
    Need(left == 16384 && right == 4096, "original_tokens_consumed_once");
    const auto shortPeer = PlanPrefillChunks({4096, 9}, 2048);
    Need(shortPeer.size() == 2, "short_peer_has_two_rounds");
    int first = 0, second = 0;
    for (const auto &round : shortPeer) {
        Need(round[0] >= 2 && round[0] <= 2048 && round[1] >= 2, "short_peer_stays_prefill");
        first += round[0]; second += round[1];
    }
    Need(first == 4096 && second == 9, "short_peer_offsets_lossless");
    bool rejected = false;
    try { (void)PlanPrefillChunks({16384, 9}, 2048); }
    catch (const std::invalid_argument &) { rejected = true; }
    Need(rejected, "extreme_ragged_rejected_before_target");
    for (const auto &bad : std::vector<std::vector<int>>{{}, {0, 4}, {-1, 4}, {1, 4}}) {
        rejected = false;
        try { (void)PlanPrefillChunks(bad, 2048); }
        catch (const std::invalid_argument &) { rejected = true; }
        Need(rejected, "invalid_rows_rejected_before_target");
    }
    rejected = false;
    try { (void)PlanPrefillChunks({8, 8}, 1); }
    catch (const std::invalid_argument &) { rejected = true; }
    Need(rejected, "invalid_cap_rejected_before_target");
    return 0;
}
