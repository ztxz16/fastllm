#include "intent.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <utility>
#include "json11.hpp"

extern "C" void duo_nf4_encode(void *, void *, void *, int, cudaStream_t);
extern "C" void duo_nf4_unpack(void *, void *, void *, int, cudaStream_t);
extern "C" void duo_nf4_gemv(void *, void *, void *, void *, int, int, cudaStream_t);
namespace fastllm::duo::live {
using native::Tensor;
namespace {
const std::string language = "model.language_model.";
Tensor norm(const Tensor &x, const Tensor &w) {
    return native::normalize(x, w, {}, false, 1e-6f, true);
}
Tensor rotary(const Tensor &x, const Tensor &cos, const Tensor &sin) {
    auto half = x.chunk(2, -1);
    return x * cos + native::cat({-half[1], half[0]}, -1) * sin;
}
} // namespace
Intent::Intent(const std::string &model, int device, bool cachePrefill, const std::filesystem::path &dir)
    : base(device), adapter(base.device) {

    native::DeviceGuard guard(base.device);
    native::Options opts(native::kFloat, device);
    std::vector<float> codebook{0, 1};
    for (int tier = 0; tier < 7; ++tier) {
        int count = 1 << tier;
        for (int j = 0; j < count; ++j) {
            float lo = .1f + .9f * j / count, hi = .1f + .9f * (j + 1) / count;
            float value = (lo + hi) * .5f * float(std::pow(10., tier - 6));
            codebook.push_back(value);
            codebook.push_back(-value);
        }
    }
    std::sort(codebook.begin(), codebook.end());
    dynamicMap = native::from_blob(codebook.data(), {256}, native::kFloat).to(device);
    frequency = native::pow(5000000., native::arange(64, opts) / 64.).reciprocal();
    std::vector<std::string> shards;
    for (const auto &file : std::filesystem::directory_iterator(std::filesystem::path(model) / "intent"))
        if (file.path().extension() == ".safetensors")
            shards.push_back(file.path());
    std::sort(shards.begin(), shards.end());
    std::ostringstream signature;
    auto identify = [&](const std::string &name) {
        auto path = std::filesystem::path(name);
        signature << path.filename().string() << ' ' << std::filesystem::file_size(path) << ' '
                  << std::filesystem::last_write_time(path).time_since_epoch().count() << '\n';
    };
    for (auto &s : shards)
        identify(s);
    identify(model + "/intent_adapter/adapter_model.safetensors");
    identity = signature.str();
    if (!dir.empty()) {
        std::ifstream f(dir / "source.v1");
        std::string signature{std::istreambuf_iterator<char>(f), {}};
        if (signature != identity)
            throw std::runtime_error("Intent cache fingerprint differs; rebuild with duo_pack_intent");
        base.load((dir / "intent.safetensors").string(), {""}, native::Type::Undefined,
                  [&](const std::string &key, Tensor &weight) {
                      auto suffix = key.rfind('.');
                      if (suffix == std::string::npos)
                          return;
                      auto kind = key.substr(suffix + 1);
                      if (kind != "nf4" && kind != "scales")
                          return;
                      auto &q = quant[key.substr(0, suffix)];
                      if (kind == "nf4") {
                          q.rows = weight.size(0);
                          q.cols = 2 * weight.size(1);
                          q.packed = std::exchange(weight, Tensor{});
                      } else
                          q.scales = std::exchange(weight, Tensor{});
                  });
    } else {
        for (auto &s : shards)
            base.load(s, {language, "lm_head."}, native::kBFloat16,
                      [&](const std::string &key, Tensor &w) { compress(key, w); });
    }
    if (quant.size() != 64 * 7)
        throw std::runtime_error("incomplete 64-layer Intent checkpoint");
    for (auto &[key, q] : quant)
        if (!q.packed.defined() || !q.scales.defined() || q.scales.numel() != int64_t(q.rows) * q.cols / 64)
            throw std::runtime_error("incomplete NF4 projection " + key);
    adapter.load(model + "/intent_adapter/adapter_model.safetensors", {"base_model.model."}, native::kFloat);
    auto ids = native::rowIndices({151676, 151677, 151679, 151680}, base.device);
    base.get(language + "embed_tokens.weight")
        .index_copy_(
            0, ids,
            adapter.get("base_model.model." + language + "embed_tokens.token_adapter.trainable_tokens_delta")
                .to(native::kBFloat16));
    if (cachePrefill) {
        size_t freeBytes, totalBytes;
        native::check(cudaMemGetInfo(&freeBytes, &totalBytes));
        prefillBudget = std::min(size_t(36) << 30, freeBytes / 3 * 2);
    }
}
void Intent::writeCache(const std::string &path) const {
    namespace fs = std::filesystem;
    fs::path output(path), temp = output.string() + ".tmp." + std::to_string(getpid());
    if (fs::exists(output) || !fs::create_directory(temp))
        throw std::runtime_error("cache destination already exists");
    try {
        auto storage = base.tensors;
        for (auto &[name, q] : quant) {
            storage.emplace(name + ".nf4", q.packed.reshape({q.rows, q.cols / 2}));
            storage.emplace(name + ".scales", q.scales.reshape({q.rows, q.cols / 64}));
        }
        json11::Json::object items;
        int64_t offset = 0;
        for (auto &[name, t] : storage) {
            json11::Json::array dims;
            for (auto n : t.sizes())
                dims.emplace_back(double(n));
            std::string type = t.scalar_type() == native::kByte    ? "U8"
                               : t.scalar_type() == native::kFloat ? "F32"
                                                                   : "BF16";
            auto end = offset + t.nbytes();
            items[name] =
                json11::Json::object{{"dtype", type},
                                     {"shape", dims},
                                     {"data_offsets", json11::Json::array{double(offset), double(end)}}};
            offset = end;
        }
        auto header = json11::Json(items).dump();
        while (header.size() % 8)
            header += ' ';
        uint64_t len = header.size();
        std::ofstream file(temp / "intent.safetensors", std::ios::binary);
        file.exceptions(std::ios::failbit | std::ios::badbit);
        file.write(reinterpret_cast<const char *>(&len), 8);
        file.write(header.data(), len);
        for (auto &[name, t] : storage) {
            auto host = t.cpu().contiguous();
            file.write(static_cast<const char *>(host.data_ptr()), host.nbytes());
        }
        file.close();
        std::ofstream meta(temp / "source.v1");
        meta.exceptions(std::ios::failbit | std::ios::badbit);
        meta << identity;
        meta.close();
        fs::rename(temp, output);
    } catch (...) {
        fs::remove_all(temp);
        throw;
    }
}
void Intent::compress(const std::string &name, Tensor &w) {
    if (name == "lm_head.weight") {
        w = w.narrow(0, 0, 151681);
        return;
    }
    if (w.dim() != 2 || name.find(".layers.") == std::string::npos)
        return;
    if (w.size(0) % 4 || w.size(1) % 64 || w.numel() % 256)
        throw std::invalid_argument("invalid NF4 projection " + name);
    auto packed = native::empty({w.numel() / 2}, w.options().dtype(native::kByte));
    auto scales = native::empty({w.numel() / 64}, w.options().dtype(native::kFloat));
    duo_nf4_encode(w.data_ptr(), packed.data_ptr(), scales.data_ptr(), w.numel(), cudaStream_t{});
    native::check(cudaGetLastError());
    native::quantizeScales(scales, dynamicMap);
    quant.emplace(name, Quant{packed, scales, int(w.size(0)), int(w.size(1))});
    w = Tensor();
}
Tensor Intent::project(const Tensor &x, const std::string &name) {
    auto it = quant.find(name + ".weight");
    Tensor y;
    if (it == quant.end())
        y = base.linear(x, name);
    else {
        auto &w = it->second;
        auto stream = cudaStream_t{};
        if (x.size(0) == 1) {
            y = native::empty({1, w.rows}, x.options());
            duo_nf4_gemv(w.packed.data_ptr(), w.scales.data_ptr(), x.contiguous().data_ptr(), y.data_ptr(),
                         w.rows, w.cols, stream);
        } else {
            Tensor full = w.dense;
            if (!full.defined()) {
                full = native::empty({w.rows, w.cols}, x.options());
                duo_nf4_unpack(w.packed.data_ptr(), w.scales.data_ptr(), full.data_ptr(), full.numel(),
                               stream);
                if (full.nbytes() <= prefillBudget - prefillBytes) {
                    w.dense = full;
                    prefillBytes += full.nbytes();
                }
            }
            y = native::linear(x, full);
        }
        native::check(cudaGetLastError());
    }
    const auto prefix = "base_model.model." + name + ".lora_";
    if (auto a = adapter.tensors.find(prefix + "A.weight"); a != adapter.tensors.end()) {
        auto delta =
            native::linear(native::linear(x.to(native::kFloat), a->second), adapter.get(prefix + "B.weight"));
        y = (y.to(native::kFloat) + delta * 2.).to(x.scalar_type());
    }
    return y;
}
void Intent::reset() {
    past = nextPosition = 0;
    logits = Tensor();
}
void Intent::rewind(int length, int position) {
    if (length < 0 || length > past || position < 0 || position > nextPosition)
        throw std::invalid_argument("invalid Intent rewind");
    past = length;
    nextPosition = position;
    logits = Tensor();
}
int Intent::nextToken() const {
    return int(*logits.argmax().cpu().data_ptr<int64_t>());
}
Tensor Intent::forward(const std::vector<int> &ids, const Visual *visual, bool needLogits,
                       const HiddenReady &hiddenReady) {

    native::DeviceGuard guard(base.device);
    const int n = ids.size();
    if (!n || past + n > 262144)
        throw std::invalid_argument("Intent context length out of range");
    auto tokens = native::rowIndices(ids, base.device);
    auto x = base.get(language + "embed_tokens.weight").index_select(0, tokens);
    std::vector<int64_t> pos(3 * n);
    std::vector<int> starts;
    int cursor = nextPosition;
    for (int i = 0; i < n;) {
        if (ids[i] == 151656) {
            if (!visual || starts.size() >= visual->pairs.size())
                throw std::invalid_argument("unexpected video token");
            starts.push_back(i);
            for (int y = 0; y < visual->rows; ++y)
                for (int col = 0; col < visual->cols; ++col, ++i) {
                    if (i >= n || ids[i] != 151656)
                        throw std::invalid_argument("video grid mismatch");
                    pos[i] = cursor;
                    pos[n + i] = cursor + y;
                    pos[2 * n + i] = cursor + col;
                }
            cursor += std::max(visual->rows, visual->cols);
        } else {
            pos[i] = pos[n + i] = pos[2 * n + i] = cursor++;
            ++i;
        }
    }
    if (visual && starts.size() != visual->pairs.size())
        throw std::invalid_argument("unused video frames");
    auto insert = [&](int deep) {
        for (size_t i = 0; i < starts.size(); ++i) {
            const auto &v = visual->pairs[i][deep];
            auto target = x.narrow(0, starts[i], v.size(0));
            if (deep)
                target.add_(v);
            else
                target.copy_(v);
        }
    };
    insert(0);
    auto positions = native::tensor(pos).to(base.device, native::kFloat).reshape({3, n, 1});
    auto all = positions * frequency.reshape({1, 1, 64});
    auto angles = all[0].clone();
    for (int axis = 1; axis <= 2; ++axis)
        angles.slice(1, axis, 60, 3).copy_(all[axis].slice(1, axis, 60, 3));
    angles = native::cat({angles, angles}, 1).unsqueeze(0).unsqueeze(0);
    auto cos = angles.cos().to(x.scalar_type()), sin = angles.sin().to(x.scalar_type());
    const int required = past + n, allocation = std::max(512, ((required + 511) / 512) * 512);
    if (required > capacity && prefillBytes) {
        size_t freeBytes, totalBytes;
        native::check(cudaMemGetInfo(&freeBytes, &totalBytes));
        const size_t growth = size_t(allocation - capacity) * 64 * 2 * 8 * 128 * sizeof(__nv_bfloat16);
        if (freeBytes < growth + (size_t(5) << 30)) {
            // Optional dense views must yield to conversation KV. NF4 is authoritative.
            for (auto &[name, weight] : quant)
                weight.dense = Tensor();
            prefillBudget = prefillBytes = 0;
        }
    }
    Tensor hidden;
    for (int i = 0; i < 64; ++i) {
        auto p = language + "layers." + std::to_string(i) + ".", a = p + "self_attn.";
        auto y = norm(x, base.get(p + "input_layernorm.weight"));
        auto heads = [&](const std::string &which, int count) {
            return project(y, a + which + "_proj").reshape({1, n, count, 128}).transpose(1, 2);
        };
        auto q = rotary(norm(heads("q", 64), base.get(a + "q_norm.weight")), cos, sin);
        auto k = rotary(norm(heads("k", 8), base.get(a + "k_norm.weight")), cos, sin);
        auto v = heads("v", 8);
        if (required > capacity) {
            auto grow = [&](Tensor &cache) {
                auto dest = native::empty({1, 8, allocation, 128}, x.options());
                if (past)
                    dest.narrow(2, 0, past).copy_(cache.narrow(2, 0, past));
                cache = dest;
            };
            grow(keys[i]);
            grow(values[i]);
        }
        keys[i].narrow(2, past, n).copy_(k);
        values[i].narrow(2, past, n).copy_(v);
        auto attn =
            native::attention(q, keys[i].narrow(2, 0, required), values[i].narrow(2, 0, required), n > 1);
        x = x + project(attn.transpose(1, 2).reshape({n, 8192}), a + "o_proj");
        y = norm(x, base.get(p + "post_attention_layernorm.weight"));
        auto f = native::silu(project(y, p + "mlp.gate_proj")) * project(y, p + "mlp.up_proj");
        x = x + project(f, p + "mlp.down_proj");
        if (i < 3)
            insert(i + 1);
        if (i == 49) {
            hidden = x;
            if (hiddenReady)
                hiddenReady(hidden);
        }
    }
    if (required > capacity)
        capacity = allocation;
    nextPosition = cursor;
    past = required;
    if (needLogits)
        logits = project(norm(x.slice(0, n - 1), base.get(language + "norm.weight")), "lm_head");
    return hidden;
}
} // namespace fastllm::duo::live
