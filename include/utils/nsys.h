#pragma once
#include <cstdlib>
#include <string>
#if defined(_WIN32) && defined(USE_CUDA) && !defined(USE_ROCM)
#include <nvtx3/nvToolsExt.h>
#endif
namespace fastllm {
inline bool NsysEnabled() {
    static const bool enabled = std::getenv("FASTLLM_NVTX") != nullptr;
    return enabled;
}
// Diagnostic ranges do not synchronize CUDA or change operator dispatch.
class NsysRange {
public:
    explicit NsysRange(const std::string &name) : active(NsysEnabled()) { Push(name); }
    ~NsysRange() { Pop(); }
    void Next(const std::string &name) { Pop(); Push(name); }
    NsysRange(const NsysRange &) = delete;
    NsysRange &operator=(const NsysRange &) = delete;
private:
    bool active;
    void Push(const std::string &name) {
#if defined(_WIN32) && defined(USE_CUDA) && !defined(USE_ROCM)
        if (active) nvtxRangePushA(name.c_str());
#endif
    }
    void Pop() {
#if defined(_WIN32) && defined(USE_CUDA) && !defined(USE_ROCM)
        if (active) nvtxRangePop();
#endif
    }
};
}
