#pragma once
#include <cstdint>
#include <filesystem>
#include <vector>

struct x264_t;
namespace fastllm::duo::live {
class AvcEncoder {
  public:
    AvcEncoder(int width, int height);
    ~AvcEncoder();
    AvcEncoder(const AvcEncoder &) = delete;
    AvcEncoder &operator=(const AvcEncoder &) = delete;
    void write(const std::filesystem::path &file, int block, const uint8_t *i420,
               const std::vector<float> &audio);

  private:
    int width, height;
    int64_t frame = 0;
    x264_t *encoder = nullptr;
};
} // namespace fastllm::duo::live
