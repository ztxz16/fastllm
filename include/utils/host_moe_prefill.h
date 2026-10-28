#pragma once
#include "fastllm.h"
namespace fastllm {
struct HostMoePrefillStats {
    int cpuExperts = 0, gpuExperts = 0;
    int cpuRoutes = 0, gpuRoutes = 0;
    double cpuMs = 0, gpuMs = 0;
};
// Unbound CPU-resident GGUF SwiGLU experts. No NUMA allocation/affinity or
// permanent GPU weight cache is required. Unsupported calls leave output alone.
bool RunHostMoePrefill(const Data &input, const Data &index, const Data &score,
                      Data &output, Data **weights, int count, int layer,
                      HostMoePrefillStats *stats = nullptr);
}
