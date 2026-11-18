#include "hy4_flash_ops.h"
#include "utils.h"
#include <fstream>
#include <cstring>
#include <set>

namespace fastllm {
Hy4FlashModel::Hy4FlashModel() : source(new Weights) {
    model_type = model_struct = "hy_v4_flash";
    canDoBatchForward = true;
    defaultChunkedPrefillSize = 64;
}
Hy4FlashModel::~Hy4FlashModel() {
#ifdef FASTLLM_ENABLE_HY4_CUDA
    hybridGraphs.reset();
    gpuGraphs.reset();
#endif
    ReleaseMoeCudaCache(hostExperts);
}
void Hy4FlashModel::InitParams() {
    for (const char *key :
         {"hidden_size", "num_hidden_layers", "num_attention_heads", "max_position_embeddings", "vocab_size",
          "eos_token_id", "bos_token_id", "rms_norm_eps"}) {
        auto it = weight.dicts.find(std::string("text_config.") + key);
        if (it != weight.dicts.end())
            weight.dicts[key] = it->second;
    }
    basellm::InitParams();
    devices.clear();
    const auto dense = deviceMap.empty() ? std::map<std::string, int>{{"cuda:0", 1}} : deviceMap;
    for (const auto &entry : dense) {
        const bool multi = entry.first.find("multicuda") == 0;
        AssertInFastLLM(multi || entry.first.find("cuda") == 0, "HY4-Flash dense layers require CUDA.");
        std::map<int, int> ratios;
        auto ids = ParseDeviceIds(entry.first, multi ? "multicuda" : "cuda", ratios);
        if (ids.empty())
            ids.push_back(0);
        for (int id : ids)
            if (std::find(devices.begin(), devices.end(), id) == devices.end())
                devices.push_back(id);
    }
    hybrid =
        !moeDeviceMap.empty() && std::any_of(moeDeviceMap.begin(), moeDeviceMap.end(), [](const auto &p) {
            return p.first == "cpu" || p.first.find("numa") == 0 || p.first == "disk";
        });
    AssertInFastLLM(!hybrid || devices.size() == 1, "HY4-Flash hybrid bring-up requires one dense GPU.");
}
void Hy4FlashModel::SetConfig(const json11::Json &root) {
    const auto &cfg = root["text_config"];
    // Keep only the draft metadata here. Its tensors are loaded lazily if this
    // model instance enables DFly2 before serving requests.
    const auto &providers = root["dfly_target_kv_layers"].array_items();
    if (providers.size() == 5 && root["dfly_num_hidden_layers"].int_value() == 5 &&
        root["dfly_enable_hidden_correction"].bool_value() &&
        root["dfly_mtp_fusion"].string_value() == "slot0" &&
        root["dfly_target_kv_layers"] == root["dfly_target_index_layer"] &&
        root["dfly_mask_token_id"].is_number()) {
        std::vector<int> layers;
        for (const auto &provider : providers)
            layers.push_back(provider.int_value());
        if (std::all_of(layers.begin(), layers.end(),
                        [this](int layer) { return layer >= 0 && layer < block_cnt; }) &&
            std::set<int>(layers.begin(), layers.end()).size() == layers.size()) {
            draftProviders = std::move(layers);
            draftMask = root["dfly_mask_token_id"].int_value();
        }
    }
    source->config = cfg;
    AssertInFastLLM(cfg["hidden_size"].int_value() == 3072 && cfg["num_hidden_layers"].int_value() == 42 &&
                        cfg["hc_mult"].int_value() == 4 && cfg["num_experts_per_tok"].int_value() == 16 &&
                        cfg["routed_expert_hidden_size"].int_value() == 1536 &&
                        cfg["n_routed_experts"].int_value() == 512 &&
                        cfg["moe_intermediate_size"].int_value() == 1536,
                    "Unsupported HY4-Flash geometry.");
}
bool Hy4FlashModel::EnableDFly2(int width) {
    if (width < 2 || width > 16 || draftProviders.size() != 5)
        return false;
    draftWidth = width;
    return true;
}
void Hy4FlashModel::OnResponseContextCreated(ResponseContext *context) {
#ifdef FASTLLM_ENABLE_HY4_CUDA
    if (draftWidth && context && !context->pastKeyValues.empty()) {
        auto &owner = context->pastKeyValues.front().first.kvStorageOwner;
        if (!owner)
            owner = std::make_shared<hy4::DraftRequest>();
    }
#endif
}
bool Hy4FlashModel::PopPendingResponseToken(ResponseContext *context, int previous, int &token) {
#ifdef FASTLLM_ENABLE_HY4_CUDA
    if (!draftWidth || !context || context->pastKeyValues.empty())
        return false;
    auto state =
        std::static_pointer_cast<hy4::DraftRequest>(context->pastKeyValues.front().first.kvStorageOwner);
    if (!state || state->pending.empty())
        return false;
    AssertInFastLLM(state->expected == previous, "DFly publication lost its verified predecessor.");
    state->expected = token = state->pending.front();
    state->pending.pop_front();
    return true;
#else
    return false;
#endif
}
void Hy4FlashModel::AddSource(const std::string &name, const std::string &file, const std::string &dtype,
                              const std::vector<int> &dims, uint64_t offset, uint64_t bytes) {
    if (name.rfind("vit.", 0) == 0)
        return;
    source->meta.emplace(name, Hy4Source{file, dtype, offset, bytes, dims});
}
Hy4FlashModel::Mapping Hy4FlashModel::GetTensorMap(const std::vector<std::string> &names) {
    Mapping result;
    result["model.norm.weight"] = {{"model.norm.weight", BFLOAT16}};
    if (!hybrid)
        return result;
    for (auto &name : names) {
        if (name.find(".mlp.experts.") == std::string::npos || !StringEndWith(name, ".weight"))
            continue;
        const int layer = std::stoi(name.substr(strlen("model.layers.")));
        bool gate = name.find(".gateup_proj.") != std::string::npos;
        std::string stack =
            "model.layers." + std::to_string(layer) + ".mlp.experts." + (gate ? "gate_up_proj" : "down_proj");
        AddSpecialWeight(name, gate ? "linearSwiglu" : "linearColumn", layer);
        weight.linearNames.insert(name);
        moeLinears.insert(name);
        result[name] = {{name, source->meta.at(stack).dtype == "F8_E4M3" ? FP8_E4M3 : BFLOAT16}};
    }
    return result;
}
static std::ifstream OpenRange(const Hy4Source &s) {
    std::ifstream f(s.file, std::ios::binary);
    f.exceptions(std::ios::badbit | std::ios::failbit);
    f.seekg(s.offset);
    return f;
}
const void *Hy4FlashModel::Weights::Host(const std::string &name) {
    auto it = host.find(name);
    if (it != host.end())
        return it->second.data();
    const auto &s = meta.at(name);
    std::vector<uint8_t> bytes(s.bytes);
    OpenRange(s).read(reinterpret_cast<char *>(bytes.data()), s.bytes);
    return host.emplace(name, std::move(bytes)).first->second.data();
}
Hy4FlashModel::Weights::Allocation::~Allocation() {
#ifdef FASTLLM_ENABLE_HY4_CUDA
    if (ptr)
        hy4::Free(device, ptr);
#endif
}
const void *Hy4FlashModel::Weights::Device(const std::string &name, int device) {
#ifdef FASTLLM_ENABLE_HY4_CUDA
    auto key = std::make_pair(device, name);
    auto it = gpu.find(key);
    if (it != gpu.end())
        return it->second->ptr;
    const auto &s = meta.at(name);
    std::unique_ptr<Allocation> a(new Allocation{device, hy4::Allocate(device, s.bytes)});
    auto f = OpenRange(s);
    std::vector<char> chunk(std::min<uint64_t>(s.bytes, 16U << 20));
    for (uint64_t off = 0; off < s.bytes; off += chunk.size()) {
        const size_t bytes = std::min<uint64_t>(chunk.size(), s.bytes - off);
        f.read(chunk.data(), bytes);
        hy4::CopyToGPU(static_cast<char *>(a->ptr) + off, chunk.data(), bytes);
    }
    return gpu.emplace(key, std::move(a)).first->second->ptr;
#else
    throw std::runtime_error("HY4-Flash requires a CUDA build.");
#endif
}
Hy4View Hy4FlashModel::Weights::View(const std::string &name, int device) {
    auto type = [](const std::string &dtype) {
        if (dtype == "BF16")
            return 0;
        if (dtype == "F32")
            return 1;
        if (dtype == "F8_E4M3")
            return 2;
        if (dtype == "I32")
            return 3;
        throw std::runtime_error("Unsupported HY4-Flash storage type " + dtype);
    };
    const auto &s = meta.at(name);
    Hy4View v;
    v.type = type(s.dtype);
    v.data = device < 0 ? Host(name) : Device(name, device);
    v.cols = s.dims.empty() ? 1 : s.dims.back();
    v.rows = s.dims.size() < 2 ? 1 : s.dims[s.dims.size() - 2];
    if (v.type == 2) {
        auto n = name + "_scale_inv";
        v.scaleType = type(meta.at(n).dtype);
        v.scale = device < 0 ? Host(n) : Device(n, device);
    }
    return v;
}
std::string Hy4FlashModel::MakeInput(const std::string &history, int round, const std::string &input) {
    return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
}
std::string Hy4FlashModel::MakeHistory(const std::string &history, int round, const std::string &input,
                                       const std::string &out) {
    return MakeInput(history, round, input) + out;
}
int Hy4FlashModel::Forward(const Data &ids, const Data &mask, const Data &pos,
                           std::vector<std::pair<Data, Data>> &past, const GenerationConfig &cfg,
                           const LastTokensManager &last, std::vector<float> *logits) {
    std::vector<std::pair<Data *, Data *>> views;
    for (auto &pair : past)
        views.emplace_back(&pair.first, &pair.second);
    std::vector<std::vector<float> *> outs{logits};
    return ForwardV2(1, ids, {}, {}, {int(ids.Count(0))}, views, {cfg}, last, &outs).at(0);
}
std::vector<int> Hy4FlashModel::ForwardV2(int batch, const Data &ids, const std::vector<Data *> &,
                                          const std::vector<Data *> &, const std::vector<int> &lengths,
                                          std::vector<std::pair<Data *, Data *>> &past,
                                          const std::vector<GenerationConfig> &cfg,
                                          const LastTokensManager &last,
                                          std::vector<std::vector<float> *> *outs) {
    AssertInFastLLM(batch > 0 && lengths.size() == size_t(batch) && cfg.size() == size_t(batch) &&
                        past.size() == size_t(batch) * block_cnt && ids.dataDevice == CPU &&
                        ids.dataType == FLOAT32,
                    "HY4-Flash invalid request batch.");
    int total = 0;
    for (int length : lengths) {
        AssertInFastLLM(length > 0 && length <= ids.Count(0) - total, "Invalid HY4-Flash segment length.");
        total += length;
    }
    AssertInFastLLM(total == ids.Count(0), "HY4-Flash token count mismatch.");
    const auto *src = reinterpret_cast<const float *>(ids.cpuData);
    std::vector<int> tokens(src, src + total);
    auto values = hybrid ? ForwardHybrid(tokens, lengths, past, cfg) : ForwardGpu(tokens, lengths, past, cfg);
    std::vector<int> sampled;
    for (int i = 0; i < batch; ++i) {
        if (outs && i < outs->size() && (*outs)[i])
            *(*outs)[i] = values[i];
        Data logits(FLOAT32, {1, int(values[i].size())}, CPU, values[i].data());
        sampled.push_back(
            LLMSampling(logits, 0, cfg[i], i < last.units.size() ? last.units[i] : LastTokensUnit()));
    }
    return sampled;
}
} // namespace fastllm
