#pragma once

#include "devices/cpu/computeutils.h"
#include "gguf.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

// Only invoked by cpu_tuning.py's short-lived subprocess, so calibration's
// synthetic weights and cache state do not affect the inference process.
static std::string BenchmarkCpuLinearThreads(int hidden, int intermediate, int kind,
                                            const int *counts, int count, int budgetMs) {
    using namespace fastllm;
    using Clock = std::chrono::steady_clock;
    const auto started = Clock::now();
    auto elapsed = [&] { return std::chrono::duration<double, std::milli>(Clock::now() - started).count(); };
    hidden = (std::clamp(hidden, 256, 8192) + 255) / 256 * 256;
    intermediate = (std::clamp(intermediate, 256, 4096) + 255) / 256 * 256;
    const auto type = kind == GGML_TYPE_IQ2_XXS ? GGML_TYPE_IQ2_XXS : GGML_TYPE_Q4_K;
    const size_t pairBytes = ggml_row_size(type, hidden) * (2 * intermediate) +
                             ggml_row_size(type, intermediate) * hidden;
    const int banks = std::clamp(int(((32ull << 20) + pairBytes - 1) / pairBytes), 2, 8);
    std::vector<std::unique_ptr<Data>> weights;
    for (int bank = 0; bank < banks; ++bank) {
        for (int part = 0; part < 2; ++part) {
            auto w = std::make_unique<Data>(DATA_GGUF_FORMAT, type,
                std::vector<int>{part ? hidden : 2 * intermediate, part ? intermediate : hidden});
            w->disableGGUFRepack = true; // canonical records, as in the disk expert cache
            w->Allocate();
            const size_t bytes = w->GetBytes(), blockBytes = ggml_type_size(type);
            std::memset(w->cpuData, 0x11 + bank, bytes);
            const uint16_t scale = 0x3c00; // finite, nonzero FP16 scale in each canonical block
            for (size_t off = 0; off < bytes; off += blockBytes)
                std::memcpy(w->cpuData + off, &scale, sizeof(scale));
            weights.push_back(std::move(w));
        }
    }
    std::vector<float> input(hidden), downInput(intermediate), output(std::max(hidden, 2 * intermediate));
    for (size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 31) - 15) / 32;
    for (size_t i = 0; i < downInput.size(); ++i) downInput[i] = float(int(i % 23) - 11) / 32;
    auto pool = std::make_unique<AliveThreadPool>(1);
    const double setupMs = elapsed();
    int nextBank = 0;
    auto pass = [&](int threads) {
        const int bank = nextBank++ % banks;
        RunLinearFloat32GGUF(input.data(), weights[bank * 2]->cpuData, output.data(), nullptr,
            weights[bank * 2].get(), 1, hidden, 2 * intermediate, pool.get(), 0, threads);
        RunLinearFloat32GGUF(downInput.data(), weights[bank * 2 + 1]->cpuData, output.data(), nullptr,
            weights[bank * 2 + 1].get(), 1, intermediate, hidden, pool.get(), 0, threads);
    };
    pass(1); // warm one expert's full shapes; do not time first-use dispatch
    std::ostringstream samples;
    bool first = true;
    int previous = 0;
    for (int i = 0; i < count; ++i) {
        const int threads = counts[i];
        if (threads <= previous || threads > 128 || elapsed() >= budgetMs) break;
        previous = threads;
        pool->ResizeThreads(threads);
        std::vector<double> times;
        for (int repeat = 0; repeat < (threads == 1 ? 1 : 3); ++repeat) {
            const auto begin = Clock::now();
            pass(threads);
            times.push_back(std::chrono::duration<double, std::milli>(Clock::now() - begin).count());
            if (elapsed() >= budgetMs) break;
        }
        if (!std::isfinite(output[0])) throw std::runtime_error("nonfinite linear calibration output");
        std::sort(times.begin(), times.end());
        if (!first) samples << ',';
        first = false;
        samples << '"' << threads << "\":" << times[times.size() / 2];
    }
    std::ostringstream out;
    out << "{\"samples_ms\":{" << samples.str() << "},\"setup_ms\":" << setupMs
        << ",\"native_elapsed_ms\":" << elapsed()
        << ",\"hidden\":" << hidden << ",\"intermediate\":" << intermediate
        << ",\"gguf_type\":\"" << ggml_type_name(type)
        << "\",\"kernel_type\":\"" << ggml_type_name((ggml_type)weights[0]->ggmlType)
        << "\",\"weight_bytes\":" << pairBytes * banks << '}';
    return out.str();
}
