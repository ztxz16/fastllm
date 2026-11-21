#pragma once
#include "fastllm.h"

namespace fastllm::hy4 {
using GraphKey = std::vector<uintptr_t>;

// Owns only the CUDA stream segments it captured. Model-specific code composes
// those segments and owns the executable and any external children.
class CaptureSegments {
  public:
    explicit CaptureSegments(bool enabled) : enabled(enabled) {}
    CaptureSegments(const CaptureSegments &) = delete;
    CaptureSegments &operator=(const CaptureSegments &) = delete;
    ~CaptureSegments();
    void Start(int device);
    void *Close();

  protected:
    void Abort();
    std::vector<void *> children;

  private:
    bool enabled;
    int activeDevice = -1;
};
} // namespace fastllm::hy4
