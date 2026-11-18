#pragma once
#include "hy4_flash_capture.h"

namespace fastllm {
class DiskMoeV41Graph;
namespace hy4 {
struct HybridExecutable;
struct HybridGraphs {
    std::map<GraphKey, std::shared_ptr<HybridExecutable>> replays;
    size_t launches = 0, warmedRows = 0;
    bool warmedSingle = false;
};
class HybridGraphCapture : private CaptureSegments {
  public:
    HybridGraphCapture(HybridGraphs *graphs, bool diskWeights);
    ~HybridGraphCapture();
    using CaptureSegments::Start;
    const Data &AddExperts(const Data &input, const Data &index, const Data &score,
                           std::vector<Data *> &weights, int layer);
    void Finish(const GraphKey &key, int kind, int rows);

  private:
    HybridGraphs *graphs;
    std::shared_ptr<DiskMoeV41Graph> disk;
    std::vector<std::shared_ptr<Data>> outputs;
    std::vector<void *> nativeHandles;
    std::vector<const void *> childGraphs;
};
bool ReplayHybrid(HybridGraphs &, const GraphKey &, int kind, int rows);
} // namespace hy4
} // namespace fastllm
