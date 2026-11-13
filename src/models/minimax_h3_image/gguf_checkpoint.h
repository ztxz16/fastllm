#pragma once
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace fastllm::minimax_h3_image::detail {
struct GgufTensor {
    std::vector<uint64_t> dimensions; // GGML order: contiguous dimension first.
    uint32_t type = 0;
    uint64_t offset = 0, bytes = 0;
};
struct GgufBlock {
    uint32_t values, bytes;
};
GgufBlock ggufBlock(uint32_t type);
// This reader is independent of safetensors. No mapping or full host copy of
// the payload: metadata and each transfer have explicit size bounds.
class GgufCheckpoint {
  public:
    explicit GgufCheckpoint(const std::string &path);
    const GgufTensor &tensor(const std::string &name) const;
    void read(const GgufTensor &, uint64_t offset, void *out, size_t bytes);
    void validateModel() const;
    void validateVideoModel() const;

  private:
    std::ifstream file_;
    uint64_t length_ = 0, position_ = 0, base_ = 0;
    std::map<std::string, GgufTensor> tensors_;
    std::string architecture_, config_;
    uint32_t alignment_ = 32;
    void header(void *, size_t);
    uint64_t integer(unsigned bytes);
    std::string string();
    void skipValue(uint32_t type);
    void validateProjections() const;
};
} // namespace fastllm::minimax_h3_image::detail
