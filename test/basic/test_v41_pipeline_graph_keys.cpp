#include "devices/cuda/fastllm-v41-full-decode.h"
#include <climits>
#include <iostream>
#include <set>
#include <stdexcept>

int main() {
    std::set<int64_t> keys;
    // Decode and verification can have the same total row count. Rows also
    // cross the old 64-row stage stride and the 32-bit key range.
    for (int layer : {0, 1, 20, 40, 128})
        for (int rows : {1, 2, 6, 12, 24, 63, 64, 65, 128, 129, INT_MAX})
            for (int mode = 0; mode < 3; ++mode) {
                const int64_t key = FastllmV41PipelineGraphKey(layer, rows, mode != 0, mode == 2);
                if (key <= INT_MAX || !keys.insert(key).second ||
                    key == FastllmV41VerificationGraphKey(rows, false) ||
                    key == FastllmV41VerificationGraphKey(rows, true))
                    throw std::runtime_error("PP graph layout key collision");
            }
    std::cout << "PASS: " << keys.size() << " independent PP graph layouts\n";
}
