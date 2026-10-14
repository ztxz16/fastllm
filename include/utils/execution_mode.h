#ifndef FASTLLM_EXECUTION_MODE_H
#define FASTLLM_EXECUTION_MODE_H

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

namespace fastllm {
// Process-wide selector, resolved once per model against its effective devices.
// Models opt in explicitly; this header does not change other architectures.
enum class ExecutionMode { Gpu, Hybrid, Cpu };

inline const char *ExecutionModeName(ExecutionMode mode) {
    switch (mode) {
        case ExecutionMode::Gpu: return "gpu";
        case ExecutionMode::Hybrid: return "hybrid";
        case ExecutionMode::Cpu: return "cpu";
    }
    return "unknown";
}

inline std::string NormalizeExecutionMode(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "auto";
    value = value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return value;
}

inline bool ExecutionDeviceIsGpu(const std::string &spec) {
    const auto type = NormalizeExecutionMode(spec.substr(0, spec.find(':')));
    return type == "cuda" || type == "multicuda";
}

// Include the effective main and routed-expert device for every layer.
// Engram/embedding storage and temporary CPU routing do not define the mode.
inline ExecutionMode ResolveExecutionMode(const char *requested,
                                          const std::vector<std::string> &devices) {
    const std::string value = NormalizeExecutionMode(requested ? requested : "auto");
    if (value != "auto" && value != "gpu" && value != "hybrid" && value != "cpu") {
        throw std::invalid_argument("FASTLLM_EXECUTION_MODE must be auto, gpu, hybrid or cpu (got '" + value + "').");
    }
    if (devices.empty()) throw std::invalid_argument("Execution mode needs effective device placements.");
    bool gpu = false, host = false;
    for (const auto &spec : devices) {
        const auto type = NormalizeExecutionMode(spec.substr(0, spec.find(':')));
        if (ExecutionDeviceIsGpu(spec)) gpu = true;
        // The disk expert backend owns host staging and CPU/GPU offload. It
        // must retain the hybrid orchestration when the main model uses CUDA.
        else if (type == "cpu" || type == "numa" || type == "disk") host = true;
        else throw std::invalid_argument("Execution mode: unsupported effective device '" + spec + "'.");
    }
    const auto detected = gpu ? (host ? ExecutionMode::Hybrid : ExecutionMode::Gpu) : ExecutionMode::Cpu;
    if (value != "auto" && value != ExecutionModeName(detected)) {
        throw std::invalid_argument("FASTLLM_EXECUTION_MODE=" + value +
            " conflicts with effective device placement (" + ExecutionModeName(detected) + ").");
    }
    return detected;
}
} // namespace fastllm
#endif
