#include "devices/numas/moeexpertpartition.h"
#include <cstdio>
#include <initializer_list>

using fastllm::detail::MoeExpertCost;
using fastllm::detail::SelectMoeExpertLimit;

int main() {
    int failures = 0;
    auto check = [&](const char *name, std::initializer_list<MoeExpertCost> experts,
                     int gpuCount, int expected) {
        const int actual = SelectMoeExpertLimit(experts, 128, gpuCount);
        if (actual != expected) {
            std::fprintf(stderr, "%s: threshold %d, expected %d\n", name, actual, expected);
            ++failures;
        }
    };
    // All CPU takes 2 us; every partition retaining a GPU job takes >=100 us.
    check("all CPU beyond default threshold", {{200, 1, 100}, {400, 1, 100}}, 2, 401);
    // One CPU job and one GPU job finish in 1 us, even with two GPUs available.
    check("fewer GPU jobs than devices", {{64, 1, 50}, {256, 100, 1}}, 2, 65);
    // All GPU takes 10 us. Balancing CPU=11 and GPU=9 increases completion time.
    check("minimize completion time", {{1, 11, 1}, {2, 100, 9}}, 1, 1);
    check("equal route counts stay together", {{200, 1, 100}, {200, 1, 100}}, 2, 201);
    // Two GPUs make all-GPU optimal (8 us); one GPU prefers a split (10 us).
    check("parallel GPU load", {{1, 10, 8}, {2, 100, 8}}, 2, 1);
    check("single GPU load", {{1, 10, 8}, {2, 100, 8}}, 1, 2);
    check("empty experts use fallback", {}, 2, 128);
    check("inactive experts use fallback", {{0, 100, 100}}, 2, 128);
    check("inactive experts do not add cost", {{0, 100, 100}, {8, 1, 10}}, 1, 9);
    if (failures == 0) std::puts("PASS MoE expert partition regressions");
    return failures ? 1 : 0;
}
