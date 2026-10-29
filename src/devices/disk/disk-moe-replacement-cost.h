#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>

namespace fastllm {
// Disk canonical records only. Heat decays every 4096 expert observations;
// do not interpret a different backend's configured token half-life as this
// clock. Project one heat window, charge every displaced record and retain
// a margin for GPU arithmetic and unmodelled dispatch overhead.
struct DiskMoeReplacementCost {
    static constexpr uint64_t residenceWindow = 4096;
    struct Record { uint64_t bytes; unsigned heat; bool download; };
    static bool Pays(Record candidate, const Record *victims, size_t count,
                     double cpuUsPerByte, double uploadUsPerByte, double downloadUsPerByte) {
        double saved = candidate.heat * double(candidate.bytes) * cpuUsPerByte / 8;
        double cost = candidate.bytes * uploadUsPerByte;
        for (size_t i = 0; i < count; ++i) {
            const auto &v = victims[i];
            saved -= v.heat * double(v.bytes) * cpuUsPerByte / 8;
            if (v.download) cost += v.bytes * downloadUsPerByte;
        }
        return saved > 1.25 * cost;
    }
    static bool Pays(Record candidate, const std::vector<Record> &victims,
                     double cpu, double upload, double download) {
        return Pays(candidate, victims.data(), victims.size(), cpu, upload, download);
    }
};
} // namespace fastllm
