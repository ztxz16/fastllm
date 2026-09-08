#include "models/qwen3_5_paged_cache.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

static void Check(const std::vector<int> &current,
                  const std::vector<int> &retained) {
    std::vector<int> expected;
    for (int page : current) {
        if (std::find(retained.begin(), retained.end(), page) == retained.end()) {
            expected.push_back(page);
        }
    }
    if (fastllm::Qwen35UnreferencedPages(current, retained) != expected) {
        throw std::runtime_error("rollback released a retained page or leaked an old page");
    }
}

int main() {
    Check({}, {});
    Check({}, {1, 2});
    Check({1, 2}, {});
    Check({7, 2, 9}, {7, 2, 9});
    Check({7, 2, 9, 3}, {7, 2, 9});
    Check({7, 2}, {7, 2, 9});
    Check({7, 2, 9}, {9, 7, 2});
    Check({7, 2, 9}, {7, 2, 8}); // Copy-on-write last page.
    Check({7, 2, 7, 9, 9}, {7, 2}); // Preserve membership and release order.

    std::mt19937 rng(20260908);
    for (int size : {1, 16, 128, 1280, 1536, 2048}) {
        std::vector<int> pages(size);
        std::iota(pages.begin(), pages.end(), 11);
        std::shuffle(pages.begin(), pages.end(), rng);
        for (int trial = 0; trial < 32; ++trial) {
            auto retained = pages;
            retained.resize(rng() % (size + 1));
            Check(pages, retained);
            retained.push_back(size + 100);
            std::shuffle(retained.begin(), retained.end(), rng);
            Check(pages, retained);
        }
    }
    for (int trial = 0; trial < 1000; ++trial) {
        std::vector<int> current(rng() % 64), retained(rng() % 64);
        for (int &page : current) page = rng() % 32;
        for (int &page : retained) page = rng() % 32;
        Check(current, retained);
    }
    std::cout << "Qwen3.5 rollback page ownership: PASS\n";
}
