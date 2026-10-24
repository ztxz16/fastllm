#pragma once

#include "device.h"
#include <atomic>
#include <memory>
#include <string>

namespace fastllm {
// Explicitly enabled by a disposable tuning worker, never by --auto or an env var.
extern std::atomic<bool> autotuneRecording;
inline bool AutotuneRecording() { return autotuneRecording.load(std::memory_order_relaxed); }
bool StartAutotuneRecording(const std::string &path);
bool StopAutotuneRecording();
void RecordAutotuneSample(const std::string &kind, const std::vector<int> &key,
                          double units, double milliseconds, int device = -1) noexcept;
class AutotuneSampleScope {
    std::string kind;
    std::vector<int> key;
    double units, start;
public:
    AutotuneSampleScope(std::string kind, std::vector<int> key, double units);
    ~AutotuneSampleScope();
};

// Calibration only: synchronize the selected stream, keeping parallel work
// *inside* an MoE call intact. Normal performance samples run with this disabled.
class AutotuneOpScope {
    struct Impl;
    std::unique_ptr<Impl> impl;
    bool entered = false;
public:
    AutotuneOpScope(const std::string &op, const std::string &backend,
                    const DataDict &datas, const IntDict &params);
    ~AutotuneOpScope();
};
}
