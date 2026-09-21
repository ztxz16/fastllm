#include "fastllm.h"
#include "devices/cpu/computeutils.h"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace fastllm;

static void Check(int rows, int hidden, int topk) {
    constexpr int sources = 23, guard = 17;
    constexpr float sentinel = -12345.f;
    std::vector<float> input(sources * hidden), weights(sources);
    std::vector<int> positions(rows * topk, -1);
    std::vector<float> expected(rows * hidden, 0.f);
    std::vector<float> output(rows * hidden + 2 * guard, sentinel);
    // Binary fractions keep the scalar oracle independent of FMA contraction.
    for (int i = 0; i < sources; ++i) weights[i] = (i % 9 - 4) / 16.f;
    for (int i = 0; i < (int)input.size(); ++i) input[i] = (i % 127 - 63) / 32.f;
    for (int row = 0; row < rows; ++row) {
        for (int k = 0; k < topk; ++k) {
            // Empty rows, holes, duplicate expert positions and signed scores.
            if (row % 5 == 0 || k % 3 == 1) continue;
            int source = (row * 7 + k / 2) % sources;
            positions[row * topk + k] = source;
            for (int h = 0; h < hidden; ++h) {
                expected[row * hidden + h] += weights[source] * input[source * hidden + h];
            }
        }
    }
    // No caller-side zeroing: the reduction must also overwrite empty rows.
    MultiThreadReduceBatch((uint8_t*)input.data(), FLOAT32, weights.data(),
                          output.data() + guard, positions.data(), rows, topk, hidden);
    if (!std::equal(expected.begin(), expected.end(), output.begin() + guard)) {
        throw std::runtime_error("ReduceBatch differs from scalar reference");
    }
    for (int i = 0; i < guard; ++i) {
        if (output[i] != sentinel || output[guard + rows * hidden + i] != sentinel) {
            throw std::runtime_error("ReduceBatch overwrote output guard");
        }
    }
}

static void CheckCopies(int count) {
    constexpr size_t guard = 31;
    std::vector<size_t> lengths(count);
    size_t total = 0;
    for (int i = 0; i < count; ++i) {
        lengths[i] = (i % 7 == 0) ? 0 : (i * 7919 + count * 11) % 32769;
        total += lengths[i];
    }
    std::vector<uint8_t> input(total + guard, 0);
    std::vector<uint8_t> output(total + guard * 2, 0xcd);
    for (size_t i = 0; i < total; ++i) input[i] = (i * 17 + i / 257) % 251;
    std::vector<MultiThreadMemcpyMultiLinesTask> tasks;
    size_t offset = 0;
    for (size_t length : lengths) {
        tasks.emplace_back(output.data() + guard + offset,
                           input.data() + offset, length);
        offset += length;
    }
    RunMultiThreadMemcpyMultiLines(tasks, GetAlivePool());
    if (!std::equal(input.begin(), input.begin() + total, output.begin() + guard)) {
        throw std::runtime_error("routed input copy differs from source");
    }
    for (size_t i = 0; i < guard; ++i) {
        if (output[i] != 0xcd || output[guard + total + i] != 0xcd) {
            throw std::runtime_error("routed input copy overwrote guard");
        }
    }
}

int main() {
    try {
        // Grow only: the pool retains previously created workers on shrink.
        for (int threads : {1, 3, 6, 40}) {
            SetThreads(threads);
            for (int count : {0, 1, 3, 7, 39, 40, 41, 50, 129, 256, 5}) {
                CheckCopies(count);
            }
            for (int rows : {0, 1, 3, 4, 5, 7, 8, 16, 31, 32, 39, 40, 41, 64, 128, 256}) {
                for (int hidden : {0, 1, 7, 31, 40, 65, 257}) {
                    for (int topk : {0, 1, 10}) Check(rows, hidden, topk);
                }
            }
        }
        auto *pool = GetAlivePool();
        auto saved = pool->curActivateThreadInterval;
        pool->curActivateThreadInterval = {3, 17};
        CheckCopies(129);
        Check(5, 2560, 10);
        pool->curActivateThreadInterval = saved;
        std::puts("PASS ReduceBatch: uneven partitions, empty rows, guards, up to 40 workers");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
