#include "gguf_checkpoint.h"
#include "json11.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace fastllm::minimax_h3_image::detail {
namespace {
void require(bool ok, const std::string &message) {
    if (!ok)
        throw std::invalid_argument("invalid image GGUF: " + message);
}
constexpr uint64_t maxHeader = 16u << 20;
} // namespace
GgufBlock ggufBlock(uint32_t type) {
    switch (type) {
    case 0:
        return {1, 4}; // F32
    case 1:
    case 30:
        return {1, 2}; // F16 / BF16
    case 2:
        return {32, 18}; // Q4_0
    case 8:
        return {32, 34}; // Q8_0
    case 10:
        return {256, 84}; // Q2_K
    case 11:
        return {256, 110}; // Q3_K
    case 12:
        return {256, 144}; // Q4_K
    case 13:
        return {256, 176}; // Q5_K
    case 14:
        return {256, 210}; // Q6_K
    default:
        throw std::invalid_argument("unsupported image GGUF tensor type: " + std::to_string(type));
    }
}
void GgufCheckpoint::header(void *out, size_t bytes) {
    require(position_ <= std::min(length_, maxHeader) && bytes <= std::min(length_, maxHeader) - position_,
            "truncated or oversized header");
    file_.read(static_cast<char *>(out), bytes);
    require(bool(file_), "truncated header");
    position_ += bytes;
}
uint64_t GgufCheckpoint::integer(unsigned bytes) {
    uint8_t data[8]{};
    require(bytes <= 8, "integer size");
    header(data, bytes);
    uint64_t value = 0;
    for (unsigned i = 0; i < bytes; ++i)
        value |= uint64_t(data[i]) << (8 * i);
    return value;
}
std::string GgufCheckpoint::string() {
    uint64_t size = integer(8);
    require(size <= (1u << 20), "oversized string");
    std::string out(size, '\0');
    header(out.data(), size);
    return out;
}
void GgufCheckpoint::skipValue(uint32_t type) {
    switch (type) {
    case 0:
    case 1:
    case 7:
        integer(1);
        return;
    case 2:
    case 3:
        integer(2);
        return;
    case 4:
    case 5:
    case 6:
        integer(4);
        return;
    case 10:
    case 11:
    case 12:
        integer(8);
        return;
    case 8:
        string();
        return;
    case 9: {
        uint32_t item = integer(4);
        uint64_t count = integer(8);
        require(item != 9 && item <= 12 && count <= (1u << 20), "metadata array");
        for (uint64_t i = 0; i < count; ++i)
            skipValue(item);
        return;
    }
    default:
        require(false, "metadata type");
    }
}
GgufCheckpoint::GgufCheckpoint(const std::string &path) : file_(path, std::ios::binary) {
    require(bool(file_), "cannot open " + path);
    file_.seekg(0, std::ios::end);
    auto end = file_.tellg();
    require(end >= 24, "truncated header");
    length_ = uint64_t(end);
    file_.seekg(0);
    require(integer(4) == 0x46554747, "magic/endian");
    auto version = integer(4);
    require(version == 2 || version == 3, "version");
    uint64_t count = integer(8), metadata = integer(8);
    require(count > 0 && count <= 8192 && metadata <= 8192, "entry count");
    std::set<std::string> keys;
    for (uint64_t i = 0; i < metadata; ++i) {
        auto key = string();
        require(keys.insert(key).second, "duplicate metadata");
        uint32_t type = integer(4);
        if (key == "general.alignment") {
            require(type == 4, "alignment type");
            alignment_ = integer(4);
            require(alignment_ && alignment_ <= 4096 && !(alignment_ & (alignment_ - 1)), "alignment");
        } else if (key == "general.architecture" || key == "config") {
            require(type == 8, "model metadata type");
            (key == "config" ? config_ : architecture_) = string();
        } else
            skipValue(type);
    }
    for (uint64_t i = 0; i < count; ++i) {
        auto name = string();
        require(!name.empty() && name.size() <= 1024, "tensor name");
        uint32_t rank = integer(4);
        require(rank >= 1 && rank <= 4, "tensor rank");
        GgufTensor t;
        uint64_t elements = 1;
        for (uint32_t d = 0; d < rank; ++d) {
            uint64_t size = integer(8);
            require(size > 0 && size <= std::numeric_limits<int>::max() &&
                        elements <= std::numeric_limits<uint64_t>::max() / size,
                    "tensor dimensions");
            t.dimensions.push_back(size);
            elements *= size;
        }
        t.type = integer(4);
        t.offset = integer(8);
        const auto block = ggufBlock(t.type);
        require(t.dimensions[0] % block.values == 0 && elements / block.values <= length_ / block.bytes,
                "tensor block size");
        t.bytes = elements / block.values * block.bytes;
        require(t.offset % alignment_ == 0, "unaligned tensor offset");
        require(tensors_.emplace(name, std::move(t)).second, "duplicate tensor");
    }
    base_ = (position_ + alignment_ - 1) / alignment_ * alignment_;
    require(base_ <= length_, "missing payload");
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    for (const auto &entry : tensors_) {
        const auto &t = entry.second;
        require(t.offset <= length_ - base_ && t.bytes <= length_ - base_ - t.offset, "truncated tensor");
        ranges.emplace_back(t.offset, t.offset + t.bytes);
    }
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i)
        require(ranges[i].first >= ranges[i - 1].second, "overlapping tensors");
}
const GgufTensor &GgufCheckpoint::tensor(const std::string &name) const {
    auto it = tensors_.find(name);
    require(it != tensors_.end(), "missing tensor " + name);
    return it->second;
}
void GgufCheckpoint::read(const GgufTensor &t, uint64_t offset, void *out, size_t bytes) {
    require(offset <= t.bytes && bytes <= t.bytes - offset, "tensor read range");
    file_.clear();
    file_.seekg(base_ + t.offset + offset);
    file_.read(static_cast<char *>(out), bytes);
    require(bool(file_), "truncated tensor read");
}
void GgufCheckpoint::validateModel() const {
    require(architecture_ == "wan", "expected Comfy H3 Image layout");
    // Video H3 shares most block names. Check image-specific heads and the
    // six-step schedule before any inference; never infer these from filenames.
    std::string error;
    auto config = json11::Json::parse(config_, error)["transformer"];
    require(error.empty() && config["pdd_steps"].number_value() == 6, "image PDD config");
    const double sigmas[] = {
        1., .9675236940383911, .8875485062599182, .75, .5327739119529724, .23200887441635132, 0.};
    const auto &s = config["pdd_sigmas"].array_items();
    require(s.size() == 7, "PDD sigma count");
    for (size_t i = 0; i < 7; ++i)
        require(s[i].is_number() && std::abs(s[i].number_value() - sigmas[i]) < 1e-7, "PDD sigmas");
    for (const auto &item :
         std::vector<std::pair<std::string, uint64_t>>{{"video_patch_proj.weight", 5376u * 96},
                                                       {"final_layer.video_out.weight", 576u * 5376},
                                                       {"adaln_t_table", 1025u * 8}}) {
        const auto &t = tensor(item.first);
        require(t.type == 0 && t.bytes == item.second * 4, "image protected tensor " + item.first);
    }
    validateProjections();
}
void GgufCheckpoint::validateVideoModel() const {
    require(architecture_=="wan" || architecture_=="minimax_h3", "expected Comfy H3 video layout");
    for (const auto &item : std::vector<std::pair<std::string,uint64_t>>{
             {"video_patch_proj.weight",5376u*96}, {"audio_patch_proj.weight",5376u*32},
             {"final_layer.video_out.weight",96u*5376}, {"final_layer.audio_out.weight",32u*5376}}) {
        const auto &t=tensor(item.first);
        const auto block=ggufBlock(t.type);
        uint64_t count=1;
        for(auto dim:t.dimensions) count*=dim;
        require(count==item.second && t.bytes==count/block.values*block.bytes,"video head "+item.first);
    }
    validateProjections();
}
void GgufCheckpoint::validateProjections() const {
    for (int i = 0; i < 52; ++i) {
        auto p = i < 50 ? "blocks." + std::to_string(i) : "token_refiner.blocks." + std::to_string(i - 50);
        for (const auto &item : std::vector<std::pair<std::string, std::vector<uint64_t>>>{
                 {".attn.qkv_proj.weight", {5376, 21504}},
                 {".attn.out_proj.weight", {7168, 5376}},
                 {".mlp.fc1.weight", {5376, 28672}},
                 {".mlp.fc2.weight", {14336, 5376}}})
            require(tensor(p + item.first).dimensions == item.second, "projection shape " + p + item.first);
    }
}
} // namespace fastllm::minimax_h3_image::detail
