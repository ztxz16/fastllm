#include "devices/moe_decode_scheduler.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    using fastllm::MoeDecodePolicy;
    // Four alternating experts fit in a global cache, but cannot borrow
    // unused slots from a different record-size partition.
    MoeDecodePolicy global(8, 6), split({0,0,0,0,1,1,1,1}, {2,4});
    auto sample = [](MoeDecodePolicy &p, int step) {
        int ids[] = {(step % 2) * 2, (step % 2) * 2 + 1};
        p.ObserveRoutes(0, ids, 2);
        int resident[] = {0,1}; p.ObserveRoutes(4, resident, 2);
        p.ObserveStep(1000, 100, 10, 2, 2);
    };
    for (int i = 0; i < 32; ++i) { sample(global,i); sample(split,i); }
    Require(global.UseGpu() && global.MissRate() < .05, "global reference should fit");
    Require(split.MissRate() > .50 && split.MissRate() < .52, "partition capacity was borrowed");

    // Nonadjacent layers of equal record size share a single partition.
    MoeDecodePolicy shared({0,0,1,1,0,0}, {1,2});
    for (int i = 0; i < 32; ++i) {
        int id = 0;
        shared.ObserveRoutes(0,&id,1); shared.ObserveRoutes(4,&id,1);
        shared.ObserveStep(1000,1000,10,2,1);
    }
    Require(shared.MissRate() == 1 && !shared.UseGpu(), "shared partition did not evict across layers");

    // A failed GPU trial clears occupancy, but preserves partition ownership.
    for (int i = 0; i < 180; ++i) split.ObserveStep(2000,100,10,2,2);
    Require(!split.UseGpu(), "slower GPU trial did not return to hybrid");
    for (int i = 0; i < 64; ++i) sample(split,i);
    Require(std::abs(split.MissRate() - .5) < .001, "reset lost partition configuration");
    MoeDecodePolicy zero({0,0}, {0});
    for (int i = 0; i < 32; ++i) {
        int ids[] = {0,-1,9}; zero.ObserveRoutes(0,ids,3);
        zero.ObserveStep(1000,1000,10,1,1);
    }
    Require(zero.MissRate() == 1 && !zero.UseGpu(), "zero capacity incorrectly counted hits");
    std::cout << "ALL_PASS\n";
}
