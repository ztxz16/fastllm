#include "weights.h"
#include "json11.hpp"
#include <fstream>
#include <limits>

namespace fastllm::duo::live {
void Weights::load(const std::string &file, const std::vector<std::string> &prefixes,
                   at::ScalarType precision,
                   const std::function<void(const std::string &, at::Tensor &)> &transform) {
    std::ifstream in(file, std::ios::binary | std::ios::ate);
    if (!in)
        throw std::runtime_error("cannot open " + file);
    const int64_t length = in.tellg();
    in.seekg(0);
    uint64_t header = 0;
    in.read(reinterpret_cast<char *>(&header), 8);
    if (!in || header > 100000000 || int64_t(8 + header) > length)
        throw std::runtime_error("invalid safetensors header in " + file);
    std::string json(header, ' '), error;
    in.read(json.data(), header);
    auto root = json11::Json::parse(json, error);
    if (!error.empty() || !root.is_object())
        throw std::runtime_error("invalid tensor metadata: " + file);
    for (const auto &[name, entry] : root.object_items()) {
        if (name == "__metadata__" || std::none_of(prefixes.begin(), prefixes.end(), [&](const auto &prefix) {
                return name.compare(0, prefix.size(), prefix) == 0;
            }))
            continue;
        auto type = entry["dtype"].string_value();
        at::ScalarType dtype;
        if (type == "BF16")
            dtype = at::kBFloat16;
        else if (type == "F32")
            dtype = at::kFloat;
        else if (type == "F16")
            dtype = at::kHalf;
        else if (type == "U8")
            dtype = at::kByte;
        else
            throw std::runtime_error("unsupported weight dtype: " + name + " " + type);
        std::vector<int64_t> shape;
        int64_t bytes = c10::elementSize(dtype);
        for (const auto &dim : entry["shape"].array_items()) {
            int64_t n = static_cast<int64_t>(dim.number_value());
            if (n <= 0 || n > length / bytes)
                throw std::runtime_error("invalid tensor shape: " + name);
            shape.push_back(n);
            bytes *= n;
        }
        auto offsets = entry["data_offsets"].array_items();
        if (offsets.size() != 2)
            throw std::runtime_error("invalid tensor offsets: " + name);
        int64_t start = offsets[0].number_value(), end = offsets[1].number_value();
        if (start < 0 || end < start || end > length - 8 - int64_t(header) || bytes != end - start)
            throw std::runtime_error("invalid tensor extent: " + name);
        auto host = at::empty(shape, at::TensorOptions().dtype(dtype));
        in.seekg(8 + header + start);
        in.read(reinterpret_cast<char *>(host.data_ptr()), bytes);
        if (!in)
            throw std::runtime_error("short weight read: " + name);
        auto tensor = host.to(device, precision == at::ScalarType::Undefined ? dtype : precision);
        if (transform)
            transform(name, tensor);
        if (tensor.defined())
            tensors.emplace(name, std::move(tensor));
    }
}
const at::Tensor &Weights::get(const std::string &name) const {
    auto it = tensors.find(name);
    if (it == tensors.end())
        throw std::runtime_error("missing weight " + name);
    return it->second;
}
at::Tensor Weights::linear(const at::Tensor &x, const std::string &name) const {
    auto bias = tensors.find(name + ".bias");
    return at::linear(x, get(name + ".weight"), bias == tensors.end() ? at::Tensor() : bias->second);
}
} // namespace fastllm::duo::live
