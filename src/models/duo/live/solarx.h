#pragma once
#include "weights.h"
#include <array>

namespace fastllm::duo::live {
class SolarX {
  public:
    SolarX(const std::string &file, int device, std::vector<int> peerDevices = {});
    // Input is the normalized BF16 [24,5,H/16,W/16] packet. Returns [17,H,W,3] uint8.
    at::Tensor decode(const at::Tensor &packet);
    void reset();

  private:
    Weights weights;
    std::string weightFile;
    std::vector<int> peerDevices;
    std::vector<std::unique_ptr<SolarX>> peers;
    int localTiles = 0;

    std::vector<int64_t> xs, ys;
    int height = 0, width = 0, th = 0, tw = 0, index = 0, window = -1;
    bool failed = false;
    std::array<bool, 7> valid{};
    std::map<int, at::Tensor> history;
    std::vector<at::Tensor> keys, values;
    at::Tensor cosine, sine, mean, deviation, rgbMean, rgbStd;
    std::array<at::Tensor, 3> first;
    std::array<at::Tensor, 2> steady;
    void plan(int h, int w, int assignedTiles = 0);
    at::Tensor push(const at::Tensor &z);
    at::Tensor layers(const at::Tensor &tiles, const std::vector<int> &fresh, int slot);
    at::Tensor blend(const at::Tensor &tiles);
};
} // namespace fastllm::duo::live
