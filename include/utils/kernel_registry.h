#pragma once

#include "json11.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace fastllm::kernels {
// A semantic family may implement one operation or a fused, multi-kernel block.
// Adapters supply actual shapes/layouts, never the server's maximum batch size.
struct Signature {
    std::array<int64_t, 12> fields{};
    bool operator==(const Signature &other) const { return fields == other.fields; }
};
struct Implementation {
    const char *id;
    json11::Json description;
};
struct Family {
    int id;
    const char *name;
    const char *version;
    std::vector<Implementation> implementations; // entry zero is the reference/fallback
};
struct Call {
    const Family *family;
    Signature signature;
    void *context;
    bool capturing = false;
    // Support checks must not launch, allocate, mutate tensors or compile.
    std::string (*unsupported)(void *, int) = nullptr;
    // Backends own stream synchronization and numeric checks; registry is CPU/GPU agnostic.
    double (*measure)(void *, int, int) = nullptr; // milliseconds per invocation
    bool (*validate)(void *, int, double &, std::string &) = nullptr;
    std::string (*hardware)(void *) = nullptr;
};

extern std::atomic<bool> enabled;
extern thread_local int bypassDepth;
inline bool Enabled() { return enabled.load(std::memory_order_relaxed) && bypassDepth == 0; }
// Avoid recursively dispatching a reference implementation or its legacy Triton branch.
struct Bypass { Bypass() { ++bypassDepth; } ~Bypass() { --bypassDepth; } };
void Register(const Family *family);
// Returns -1 for the original path; otherwise a validated implementation index.
int Select(const Call &call);
bool NeedsPreparation(const Signature &signature);
bool CandidateAllowed(const char *id);
bool Configure(const std::string &json, std::string &error);
void Freeze();
std::string Report();
std::string Catalog();
}

extern "C" {
int fastllm_kernel_configure(const char *json);
void fastllm_kernel_freeze();
const char *fastllm_kernel_report();
const char *fastllm_kernel_catalog();
const char *fastllm_kernel_error();
}
