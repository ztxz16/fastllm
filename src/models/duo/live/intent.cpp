#include "intent.h"
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include "json11.hpp"

extern "C" void duo_nf4_encode(void *, void *, void *, int, cudaStream_t);
extern "C" void duo_nf4_unpack(void *, void *, void *, int, cudaStream_t);
extern "C" void duo_nf4_gemv(void *, void *, void *, void *, int, int, cudaStream_t);
namespace fastllm::duo::live {
using at::Tensor;
namespace {
const std::string language = "model.language_model.";
Tensor norm(const Tensor &x, const Tensor &w) {
    auto f = x.to(at::kFloat);
    return w * (f * at::rsqrt(f.square().mean(-1, true) + 1e-6)).to(x.scalar_type());
}
Tensor rotary(const Tensor &x, const Tensor &cos, const Tensor &sin) {
    auto half = x.chunk(2, -1);
    return x * cos + at::cat({-half[1], half[0]}, -1) * sin;
}
} // namespace
Intent::Intent(const std::string &model, int device, bool cachePrefill, const std::filesystem::path &dir)
    : base(at::Device(at::kCUDA, device)), adapter(base.device) {
    c10::InferenceMode inference;
    c10::cuda::CUDAGuard guard(base.device);
    auto opts = at::TensorOptions().device(base.device).dtype(at::kFloat);
    auto cpu = at::TensorOptions().dtype(at::kFloat);
    std::vector<Tensor> map{at::zeros({1}, cpu), at::ones({1}, cpu)};
    for (int i = 0; i < 7; ++i) {
        int n = 1 << i;
        auto b = at::linspace(.1, 1, n + 1, cpu);
        auto m = ((b.slice(0, 0, n) + b.slice(0, 1)) * .5) * std::pow(10., i - 6);
        map.push_back(m);
        map.push_back(-m);
    }
    dynamicMap = std::get<0>(at::cat(map).sort()).to(base.device);
    frequency = at::pow(5000000., at::arange(64, opts) / 64.).reciprocal();
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
        base.load((dir / "intent.safetensors").string(), {""}, at::ScalarType::Undefined,
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
                          q.packed = std::move(weight);
                      } else
                          q.scales = std::move(weight);
                  });
    } else {
        for (auto &s : shards)
            base.load(s, {language, "lm_head."}, at::kBFloat16,
                      [&](const std::string &key, Tensor &w) { compress(key, w); });
    }
    if (quant.size() != 64 * 7)
        throw std::runtime_error("incomplete 64-layer Intent checkpoint");
    for (auto &[key, q] : quant)
        if (!q.packed.defined() || !q.scales.defined() || q.scales.numel() != int64_t(q.rows) * q.cols / 64)
            throw std::runtime_error("incomplete NF4 projection " + key);
    adapter.load(model + "/intent_adapter/adapter_model.safetensors", {"base_model.model."}, at::kFloat);
    auto ids = at::tensor({151676, 151677, 151679, 151680}, at::kLong).to(base.device);
    base.get(language + "embed_tokens.weight")
        .index_copy_(
            0, ids,
            adapter.get("base_model.model." + language + "embed_tokens.token_adapter.trainable_tokens_delta")
                .to(at::kBFloat16));
    c10::cuda::CUDACachingAllocator::emptyCache();
    if (cachePrefill) {
        size_t freeBytes, totalBytes;
        C10_CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
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
            storage.emplace(name + ".nf4", q.packed.view({q.rows, q.cols / 2}));
            storage.emplace(name + ".scales", q.scales.view({q.rows, q.cols / 64}));
        }
        json11::Json::object items;
        int64_t offset = 0;
        for (auto &[name, t] : storage) {
            json11::Json::array dims;
            for (auto n : t.sizes())
                dims.emplace_back(double(n));
            std::string type = t.scalar_type() == at::kByte    ? "U8"
                               : t.scalar_type() == at::kFloat ? "F32"
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
    auto packed = at::empty({w.numel() / 2}, w.options().dtype(at::kByte));
    auto scales = at::empty({w.numel() / 64}, w.options().dtype(at::kFloat));
    duo_nf4_encode(w.data_ptr(), packed.data_ptr(), scales.data_ptr(), w.numel(),
                   c10::cuda::getCurrentCUDAStream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    auto mean = scales.mean(), groups = (scales - mean).view({-1, 256});
    auto absmax = groups.abs().amax(1, true);
    auto normalized = groups / absmax.clamp_min(1e-30);
    auto bounds = (dynamicMap.slice(0, 0, 255) + dynamicMap.slice(0, 1)) * .5;
    auto code = at::bucketize(normalized, bounds);
    scales = (dynamicMap.index_select(0, code.flatten()).view_as(groups) * absmax + mean).flatten();
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
        auto stream = c10::cuda::getCurrentCUDAStream();
        if (x.size(0) == 1) {
            y = at::empty({1, w.rows}, x.options());
            duo_nf4_gemv(w.packed.data_ptr(), w.scales.data_ptr(), x.contiguous().data_ptr(), y.data_ptr(),
                         w.rows, w.cols, stream);
        } else {
            Tensor full = w.dense;
            if (!full.defined()) {
                full = at::empty({w.rows, w.cols}, x.options());
                duo_nf4_unpack(w.packed.data_ptr(), w.scales.data_ptr(), full.data_ptr(), full.numel(),
                               stream);
                if (full.nbytes() <= prefillBudget - prefillBytes) {
                    w.dense = full;
                    prefillBytes += full.nbytes();
                }
            }
            y = at::linear(x, full);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const auto prefix = "base_model.model." + name + ".lora_";
    if (auto a = adapter.tensors.find(prefix + "A.weight"); a != adapter.tensors.end()) {
        auto delta = at::linear(at::linear(x.to(at::kFloat), a->second), adapter.get(prefix + "B.weight"));
        y = (y.to(at::kFloat) + delta * 2.).to(x.scalar_type());
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
    return logits.argmax().item<int>();
}
Tensor Intent::forward(const std::vector<int> &ids, const Visual *visual, bool needLogits,
                       const HiddenReady &hiddenReady) {
    c10::InferenceMode mode;
    c10::cuda::CUDAGuard guard(base.device);
    const int n = ids.size();
    if (!n || past + n > 262144)
        throw std::invalid_argument("Intent context length out of range");
    auto tokens = at::tensor(ids, at::kInt).to(base.device, at::kLong);
    auto x = at::embedding(base.get(language + "embed_tokens.weight"), tokens);
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
    auto positions = at::tensor(pos, at::kLong).to(base.device, at::kFloat).view({3, n, 1});
    auto all = positions * frequency.view({1, 1, 64});
    auto angles = all[0].clone();
    for (int axis = 1; axis <= 2; ++axis)
        angles.slice(1, axis, 60, 3).copy_(all[axis].slice(1, axis, 60, 3));
    angles = angles.repeat({1, 2}).unsqueeze(0).unsqueeze(0);
    auto cos = angles.cos().to(x.scalar_type()), sin = angles.sin().to(x.scalar_type());
    const int required = past + n, allocation = std::max(512, ((required + 511) / 512) * 512);
    if (required > capacity && prefillBytes) {
        size_t freeBytes, totalBytes;
        C10_CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
        const size_t growth = size_t(allocation - capacity) * 64 * 2 * 8 * 128 * sizeof(at::BFloat16);
        if (freeBytes < growth + (size_t(5) << 30)) {
            // Optional dense views must yield to conversation KV. NF4 is authoritative.
            for (auto &[name, weight] : quant)
                weight.dense = Tensor();
            prefillBudget = prefillBytes = 0;
        }
    }
    std::optional<Tensor> mask;
    if (n > 1 && past) {
        auto t = at::arange(required, x.options().dtype(at::kLong));
        mask = t.unsqueeze(0) <= t.slice(0, past).unsqueeze(1);
    }
    Tensor hidden;
    for (int i = 0; i < 64; ++i) {
        auto p = language + "layers." + std::to_string(i) + ".", a = p + "self_attn.";
        auto y = norm(x, base.get(p + "input_layernorm.weight"));
        auto heads = [&](const std::string &which, int count) {
            return project(y, a + which + "_proj").view({1, n, count, 128}).transpose(1, 2);
        };
        auto q = rotary(norm(heads("q", 64), base.get(a + "q_norm.weight")), cos, sin);
        auto k = rotary(norm(heads("k", 8), base.get(a + "k_norm.weight")), cos, sin);
        auto v = heads("v", 8);
        if (required > capacity) {
            auto grow = [&](Tensor &cache) {
                auto dest = at::empty({1, 8, allocation, 128}, x.options());
                if (past)
                    dest.narrow(2, 0, past).copy_(cache.narrow(2, 0, past));
                cache = dest;
            };
            grow(keys[i]);
            grow(values[i]);
        }
        keys[i].narrow(2, past, n).copy_(k);
        values[i].narrow(2, past, n).copy_(v);
        auto attn = at::scaled_dot_product_attention(q, keys[i].narrow(2, 0, required),
                                                     values[i].narrow(2, 0, required), mask, 0.,
                                                     n > 1 && !past, std::nullopt, true);
        x = x + project(attn.transpose(1, 2).reshape({n, 8192}), a + "o_proj");
        y = norm(x, base.get(p + "post_attention_layernorm.weight"));
        auto f = at::silu(project(y, p + "mlp.gate_proj")) * project(y, p + "mlp.up_proj");
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
