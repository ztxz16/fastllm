#pragma once
#include "align.h"
#include "intent.h"
#include "models/minimax_h3/runtime.h"
#include "json11.hpp"
#include <filesystem>

namespace fastllm::duo::live {
struct InputError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct Tokenizer {
    explicit Tokenizer(const std::string &dir);
    nomas::QwenTokenizer bpe;
    std::vector<std::string> pieces;
    std::string text(const std::vector<int> &ids) const;
};
void command(std::vector<std::string> args, const std::filesystem::path &output = {});
std::string readFile(const std::filesystem::path &);
struct Input {
    std::vector<std::vector<int>> bodies;
    std::vector<Visual> views;
    int release = 0, lastSpeaking = 0;
};
struct AVBlock {
    int width = 0, height = 0;
    std::vector<uint8_t> rgb;
    std::vector<int16_t> pcm;
    bool end = false;
    std::vector<uint8_t> mjpeg;
    void decodeJpeg();
};
struct ResidentSpeech;
class InputWorker {
  public:
    InputWorker(std::string model, int speechDevice, int visionDevice);
    ~InputWorker();
    std::string recognize(const std::vector<float> &pcm);
    Visual vision(const std::vector<uint8_t> &rgb, int width, int height);
    Input prepare(const std::filesystem::path &input, const std::filesystem::path &folder,
                  const Tokenizer &tokenizer, const std::function<void(const std::string &)> &transcribed);

  private:
    std::string root;
    std::unique_ptr<ResidentSpeech> recognizer;
    int speechDevice, visionDevice;
    std::unique_ptr<Aligner> aligner;
    std::unique_ptr<nomas::Runtime> visionRuntime;
    std::unique_ptr<nomas::Weights> visionWeights;
};
} // namespace fastllm::duo::live
