#include "quantized.h"
#include "models/minimax_h3_image/fp8.h"
#include "../minimax_h3_image/quantized_checkpoint.h"
#include "../minimax_h3_image/quantized_backends.h"
#include "../minimax_h3_image/gguf_checkpoint.h"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>

namespace fastllm::qwen_image {
namespace {
void require(bool ok, const std::string &message) {
    if (!ok)
        throw std::invalid_argument("Qwen-Image quantized: " + message);
}
std::map<std::string, std::pair<int, int>> projectionShapes() {
    std::map<std::string, std::pair<int, int>> shapes{
        {"img_in", {4096, 64}}, {"proj_out", {64, 4096}},
        {"txt_in.in_layer", {4096, 4096}}, {"txt_in.out_layer", {4096, 4096}},
        {"time_text_embed.timestep_embedder.linear_1", {4096, 256}},
        {"time_text_embed.timestep_embedder.linear_2", {4096, 4096}},
        {"modulation.1", {16384, 4096}}, {"norm_out.linear", {4096, 4096}}};
    for (int i = 0; i < 32; ++i) {
        const auto p = "transformer_blocks." + std::to_string(i) + ".";
        for (const auto *name : {"attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"})
            shapes[p + name] = {4096, 4096};
        shapes[p + "img_mlp.gate_layer"] = shapes[p + "img_mlp.proj"] = {12288, 4096};
        shapes[p + "img_mlp.gate_up"] = {24576, 4096};
        shapes[p + "img_mlp.out"] = {4096, 12288};
    }
    return shapes;
}
bool ggufMagic(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    char magic[4]{};
    file.read(magic, 4);
    return std::string(magic, 4) == "GGUF";
}
} // namespace
namespace detail = minimax_h3_image::detail;
struct QuantizedProjections::Impl {
    std::set<std::string> names;
    Runtime *runtime = nullptr;
    std::unique_ptr<minimax_h3_image::QuantizedWeights> weights;
    std::unique_ptr<detail::GgufCheckpoint> gguf;
    std::string prefix;
    std::map<std::string, Tensor> vectors;
    void loadGguf(const std::string &path) {
        gguf = std::make_unique<detail::GgufCheckpoint>(path);
        const auto &tensors = gguf->tensors();
        if (tensors.count("model.diffusion_model.img_in.weight"))
            prefix = "model.diffusion_model.";
        const auto shapes = projectionShapes();
        std::map<std::string, int> norms{{"txt_in.text_norm.weight", 4096}};
        for (int i = 0; i < 32; ++i)
            for (const auto *name : {"attn.norm_q.weight", "attn.norm_k.weight"})
                norms["transformer_blocks." + std::to_string(i) + "." + name] = 128;
        for (const auto &entry : tensors) {
            require(entry.first.compare(0, prefix.size(), prefix) == 0, "mixed GGUF name prefixes");
            const auto name = entry.first.substr(prefix.size());
            const auto &d = entry.second;
            auto norm = norms.find(name);
            if (norm != norms.end()) {
                require(d.dimensions == std::vector<uint64_t>{uint64_t(norm->second)} &&
                            (d.type == 0 || d.type == 1 || d.type == 30), "GGUF norm shape/type: " + name);
                continue;
            }
            require(name.size() > 7 && name.substr(name.size() - 7) == ".weight", "unsupported GGUF tensor: " + name);
            const auto key = name.substr(0, name.size() - 7);
            const auto expected = shapes.find(key);
            require(expected != shapes.end(), "unknown GGUF projection: " + key);
            require(d.dimensions == std::vector<uint64_t>{uint64_t(expected->second.second), uint64_t(expected->second.first)},
                    "GGUF matrix shape: " + key);
            names.insert(key);
        }
        for (const auto &entry : norms)
            require(tensors.count(prefix + entry.first), "missing GGUF norm: " + entry.first);
        for (const auto &entry : shapes) {
            const auto &name = entry.first;
            const auto at = name.find("img_mlp.");
            if (at != std::string::npos && name != name.substr(0, at) + "img_mlp.out") {
                const auto stem = name.substr(0, at) + "img_mlp.";
                const bool fused = names.count(stem + "gate_up");
                require(fused ? !names.count(stem + "gate_layer") && !names.count(stem + "proj") :
                                    names.count(stem + "gate_layer") && names.count(stem + "proj"),
                        "missing or ambiguous GGUF gate/up: " + stem);
            } else
                require(names.count(name), "missing GGUF matrix: " + name);
        }
        weights = detail::loadGguf(path, {});
    }
};
std::unique_ptr<QuantizedProjections> QuantizedProjections::load(Runtime &rt, const std::string &path) {
    // BF16 sharded checkpoints keep the existing loader and allocation policy.
    if (std::filesystem::is_directory(path))
        return {};
    auto impl = std::make_unique<Impl>();
    impl->runtime = &rt;
    rt.activate();
    if (ggufMagic(path)) {
        require(!rt.isFloat32(), "GGUF checkpoints require BF16 activations");
        int device = 0;
        require(cudaGetDevice(&device) == cudaSuccess, "cannot select CUDA device");
        detail::validateGgufDevice(device);
        impl->loadGguf(path);
        return std::unique_ptr<QuantizedProjections>(new QuantizedProjections(std::move(impl)));
    }
    detail::QuantizedCheckpoint checkpoint(path);
    using minimax_h3_image::WeightFormat;
    auto format = WeightFormat::BF16;
    const auto shapes = projectionShapes();
    const std::string suffix = "._weight_qdata";
    for (const auto &entry : checkpoint.header.object_items()) {
        const auto &name = entry.first;
        const auto dtype = entry.second["dtype"].string_value();
        if (dtype == "I8") {
            require(format != WeightFormat::FP8, "mixed FP8 and INT8 checkpoints are unsupported");
            format = WeightFormat::INT8;
            require(name.size() > 7 && name.substr(name.size() - 7) == ".weight", "expected Comfy INT8 weight: " + name);
            const auto prefix = name.substr(0, name.size() - 7);
            const auto expected = shapes.find(prefix);
            require(expected != shapes.end(), "unknown INT8 projection: " + prefix);
            const auto [rows, cols] = expected->second;
            require(cols % 256 == 0, "INT8 ConvRot width: " + prefix);
            const auto d = checkpoint.descriptor(name, "I8");
            require(d["shape"] == json11::Json::array{rows, cols}, "INT8 matrix shape: " + prefix);
            const auto scale = checkpoint.desc(prefix + ".weight_scale", "F32", size_t(rows)*4);
            require(scale["shape"] == json11::Json::array{rows, 1} || scale["shape"] == json11::Json::array{rows},
                    "INT8 scale shape: " + prefix);
            const auto marker = checkpoint.marker(prefix + ".comfy_quant");
            require(marker["format"].string_value() == "int8_tensorwise" && marker["convrot"].bool_value() &&
                        marker["convrot_groupsize"].number_value() == 256 && !marker["full_precision_matrix_mult"].bool_value(),
                    "expected INT8 ConvRot H256 marker: " + prefix);
            require(checkpoint.header[prefix + ".bias"].is_null(), "biased INT8 projection: " + prefix);
            impl->names.insert(prefix);
            continue;
        }
        if (dtype != "F8_E4M3")
            continue;
        require(format != WeightFormat::INT8, "mixed FP8 and INT8 checkpoints are unsupported");
        format = WeightFormat::FP8;
        require(name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0,
                "expected TorchAO ._weight_qdata and ._weight_scale: " + name);
        const auto prefix = name.substr(0, name.size() - suffix.size());
        const auto expected = shapes.find(prefix);
        require(expected != shapes.end(), "unsupported quantized projection: " + prefix);
        const auto [rows, cols] = expected->second;
        const auto d = checkpoint.descriptor(name, "F8_E4M3");
        require(d["shape"] == json11::Json::array{rows, cols}, "matrix shape: " + prefix);
        const auto scale = checkpoint.desc(prefix + "._weight_scale", "F32", size_t(rows) * 4);
        require(scale["shape"] == json11::Json::array{rows, 1} ||
                    scale["shape"] == json11::Json::array{rows}, "scale shape: " + prefix);
        std::string error;
        auto metadata = json11::Json::parse(checkpoint.header["__metadata__"][prefix + ".weight"].string_value(), error);
        const auto &spec = metadata["_data"];
        const auto &activation = spec["act_quant_kwargs"]["_data"];
        require(error.empty() && metadata["_type"].string_value() == "Float8Tensor" &&
                    spec["block_size"] == json11::Json::array{1, cols} &&
                    spec["dtype"]["_data"].string_value() == "bfloat16" &&
                    activation["float8_dtype"]["_data"].string_value() == "float8_e4m3fn" &&
                    activation["granularity"]["_type"].string_value() == "PerRow" &&
                    activation["granularity"]["_data"]["dim"].int_value() == -1 &&
                    activation["kernel_preference"]["_data"].string_value() == "TORCH" &&
                    activation["hp_value_lb"].number_value() == 1e-12 && activation["hp_value_ub"].is_null(),
                "expected TorchAO BF16 per-row FP8 metadata: " + prefix);
        require(checkpoint.header[prefix + ".weight"].is_null() &&
                    checkpoint.header[prefix + ".bias"].is_null(), "ambiguous or biased FP8 projection: " + prefix);
        impl->names.insert(prefix);
    }
    if (impl->names.empty())
        return {};
    require(!rt.isFloat32(), "quantized checkpoints require BF16 activations");
    // Each fused group must use one format; whole BF16 layers can remain plain.
    for (int i = 0; i < 32; ++i) {
        const auto p = "transformer_blocks." + std::to_string(i) + ".";
        if (impl->names.count(p + "img_mlp.gate_up"))
            for (const auto *suffix : {"img_mlp.gate_layer.weight", "img_mlp.proj.weight",
                                      "img_mlp.gate_layer._weight_qdata", "img_mlp.proj._weight_qdata"})
                require(checkpoint.header[p + suffix].is_null(), "ambiguous fused gate/up: " + p);
        for (const auto &group : std::vector<std::vector<std::string>>{
                 {"attn.to_q", "attn.to_k", "attn.to_v"}, {"img_mlp.gate_layer", "img_mlp.proj"}})
            for (const auto &name : group)
                require(impl->names.count(p + name) == impl->names.count(p + group.front()),
                        "inconsistent fused projection formats: " + p);
    }
    int device = 0;
    const auto status = cudaGetDevice(&device);
    require(status == cudaSuccess, cudaGetErrorString(status));
    if (format == WeightFormat::INT8) {
        detail::validateInt8Device(device);
        impl->weights = detail::loadInt8(path, {});
    } else {
        minimax_h3_image::Fp8Weights::validateDevice(device);
        impl->weights = std::make_unique<minimax_h3_image::Fp8Weights>(path);
    }
    return std::unique_ptr<QuantizedProjections>(new QuantizedProjections(std::move(impl)));
}
QuantizedProjections::QuantizedProjections(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
QuantizedProjections::~QuantizedProjections() = default;
bool QuantizedProjections::contains(const std::string &name) const { return impl_->names.count(name); }
void QuantizedProjections::setPrefill(bool prefill) { impl_->weights->setPrefill(prefill); }
Tensor QuantizedProjections::linear(const Tensor &x, const std::vector<std::string> &names, Tensor *out) {
    for (const auto &name : names)
        require(contains(name), "missing packed projection: " + name);
    std::vector<std::string> actual;
    for (const auto &name : names)
        actual.push_back(impl_->prefix + name);
    return impl_->weights->linearPacked(x, actual, out);
}
bool QuantizedProjections::isGguf() const { return bool(impl_->gguf); }
const Tensor &QuantizedProjections::get(const std::string &name) {
    require(isGguf(), "plain safetensors use the ordinary weight reader");
    auto found = impl_->vectors.find(name);
    if (found != impl_->vectors.end()) return found->second;
    const auto &d = impl_->gguf->tensor(impl_->prefix + name);
    require(d.dimensions.size() == 1 && d.dimensions[0] <= 4096 &&
                (d.type == 0 || d.type == 1 || d.type == 30), "expected a plain GGUF norm vector");
    std::vector<float> values(d.dimensions[0]);
    if (d.type == 0)
        impl_->gguf->read(d, 0, values.data(), d.bytes);
    else {
        std::vector<uint16_t> raw(values.size());
        impl_->gguf->read(d, 0, raw.data(), d.bytes);
        for (size_t i = 0; i < raw.size(); ++i) {
            if (d.type == 30) {
                const uint32_t bits = uint32_t(raw[i]) << 16;
                std::memcpy(&values[i], &bits, 4);
            } else {
                const int h = raw[i], exponent = (h >> 10) & 31;
                values[i] = (h & 32768 ? -1.f : 1.f) * (exponent == 31 ?
                    (h & 1023 ? NAN : INFINITY) : exponent ?
                    std::ldexp(1.f + (h & 1023) / 1024.f, exponent - 15) : std::ldexp(float(h & 1023), -24));
            }
        }
    }
    return impl_->vectors.emplace(name, impl_->runtime->upload(values.data(), 1, int(values.size()))).first->second;
}
void QuantizedProjections::clear() { impl_->weights->clear(); impl_->vectors.clear(); }
size_t QuantizedProjections::loadedBytes() const {
    size_t bytes = impl_->weights->loadedBytes();
    for (const auto &entry : impl_->vectors) bytes += entry.second.bytes();
    return bytes;
}
} // namespace fastllm::qwen_image
