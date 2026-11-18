#pragma once
#include "models/duo/model.h"
#include <filesystem>
#include <fstream>
#include <cstring>
#include <limits>
namespace fastllm::duo {
inline std::vector<float> readFloats(const std::filesystem::path &path, size_t count) {
    if (count > size_t(1) << 28 || std::filesystem::file_size(path) != count * sizeof(float))
        throw std::invalid_argument("tensor byte count differs: " + path.string());
    std::vector<float> out(count);
    std::ifstream f(path, std::ios::binary);
    if (!f.read(reinterpret_cast<char *>(out.data()), out.size() * sizeof(float)))
        throw std::runtime_error("cannot read tensor: " + path.string());
    return out;
}
inline void writeFloats(const std::filesystem::path &path, const std::vector<float> &values) {
    std::ofstream f(path, std::ios::binary);
    if (!f.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float)))
        throw std::runtime_error("cannot write tensor: " + path.string());
}
inline int integer(const json11::Json &j, const std::string &key, int lo, int hi) {
    double v = j[key].number_value();
    if (!j[key].is_number() || v != j[key].int_value() || v < lo || v > hi)
        throw std::invalid_argument("invalid integer: " + key);
    return j[key].int_value();
}
inline json11::Json resultMetadata(const ChunkResult &r, int h, int w) {
    return json11::Json::object{{"protocol", "fastllm-duo-v1"},
                                {"kind", "latent_chunk"},
                                {"chunk", r.chunk},
                                {"first_frame", double(r.firstFrame)},
                                {"frames", 17},
                                {"fps", 24},
                                {"first_sample", double(r.firstSample)},
                                {"end_sample", double(r.endSample)},
                                {"sample_rate", 32000},
                                {"video_shape", json11::Json::array{5 * h * w / 4, 96}},
                                {"audio_shape", json11::Json::array{58, 32}},
                                {"cache_bytes", double(r.cacheBytes)},
                                {"seconds", r.seconds}};
}
} // namespace fastllm::duo
