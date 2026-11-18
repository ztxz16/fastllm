#pragma once
#include "hy4_flash_capture.h"

namespace fastllm::hy4 {
struct GpuGraphs {
    std::map<GraphKey, std::shared_ptr<void>> replays;
    size_t launches = 0, warmedRows = 0;
    bool warmedSingle = false;
    bool peerChecked = false, peerReady = false;
};
class GpuGraphCapture : public CaptureSegments {
  public:
    explicit GpuGraphCapture(GpuGraphs *graphs) : CaptureSegments(graphs != nullptr), graphs(graphs) {}
    void Finish(const GraphKey &key, int kind, int rows);

  private:
    GpuGraphs *graphs;
};
bool ReplayGpu(GpuGraphs &, const GraphKey &, int kind, int rows);
bool EnableGraphPeers(const std::vector<int> &devices);
} // namespace fastllm::hy4
