#pragma once

#include "gguf.h"
#include <cassert>

// Construct legacy R4 weights independently of the loader's preferred layout.
// IQ2_S stays in its original GGUF layout on x86, but R4 consumers remain valid.
static inline void PackIQ2SR4Fixture(int rows, int columns, const char *src,
                                   char *dst, bool) {
    assert(rows % 4 == 0 && columns % QK_K == 0);
    const int blocks = columns / QK_K;
    const auto *raw = reinterpret_cast<const block_iq2_s *>(src);
    auto *packed = reinterpret_cast<block_iq2_s_r4 *>(dst);
    for (int row = 0; row < rows; ++row) {
        const int lane = row % 4;
        for (int block = 0; block < blocks; ++block) {
            const auto &x = raw[row * blocks + block];
            auto &y = packed[(row / 4) * blocks + block];
            y.d[lane] = x.d;
            for (int group = 0; group < QK_K / 32; ++group) {
                y.qh[4 * group + lane] = x.qh[group];
                y.scales[4 * group + lane] = x.scales[group];
                for (int i = 0; i < 4; ++i) {
                    y.qs[16 * group + 4 * lane + i] = x.qs[4 * group + i];
                    y.signs[16 * group + 4 * lane + i] = x.qs[QK_K / 8 + 4 * group + i];
                }
            }
        }
    }
}
