#include "clover.h"
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <stdexcept>
#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include "devices/cuda/fastllm-clover.h"
#endif

namespace fastllm {
namespace {
void Check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(std::string("Clover: ") + message);
}
#ifdef USE_CUDA
void PrepareCloverData(Data &data, DataType type, const std::vector<int> &dims, int device, bool zero = false) {
    data.dataType = type;
    data.UpdateUnitSize();
    data.Resize(dims);
    data.ToDevice(DataDevice::CUDA, std::vector<int>{device}, false);
    if (zero) data.Allocate(0.f); else data.Allocate(false);
}
#endif
std::string Layer(int i) { return "model.layers." + std::to_string(i) + "."; }
}
// One fixed-width set of stateless MoE graphs belongs to this model. The
// scheduler serializes forwards; graphs are released after ShutdownRuntime
// joins its worker, before the resident weights are destroyed.
struct CloverModel::MoeGraphs {
#ifdef USE_CUDA
    struct Entry {
        Data input, output;
        void *graph = nullptr, *executable = nullptr;
        std::vector<void *> reserved;
        bool disabled = false;
        ~Entry() {
            if (executable) FastllmCudaGraphExecDestroy(executable);
            if (graph) FastllmCudaGraphDestroy(graph);
            FastllmCudaGraphMemoryPoolRelease(reserved);
        }
    };
    int device, rows = 0, batch = 1;
    bool enabled;
    std::map<std::pair<int, bool>, std::unique_ptr<Entry>> entries;
    explicit MoeGraphs(int device) : device(device) {
        const char *flag = std::getenv("FASTLLM_CUDA_GRAPH");
        std::string value = flag ? flag : "";
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
        enabled = value != "0" && value != "false" && value != "off" && value != "no";
    }
    ~MoeGraphs() {
        // Destruction can happen on the API thread, not the scheduler's
        // per-thread stream. Wait on the device before releasing graph pins.
        int previous = device;
        cudaGetDevice(&previous);
        cudaSetDevice(device);
        cudaDeviceSynchronize();
        entries.clear();
        cudaSetDevice(previous);
    }
    void SetRows(int value, int batchValue) {
        if (rows == value && batch == batchValue) return;
        Check(cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess, "MoE graph reset sync failed");
        entries.clear();
        rows = value;
        batch = batchValue;
    }
#endif
};
CloverModel::CloverModel() {
    model_type = model_struct = "clover_moe";
    dataType = kvCacheDataType = BFLOAT16;
    maxBatch = 1;
    defaultChunkedPrefillSize = 4096;
    weight.embeddingNames.insert("model.embed_tokens.weight");
    weight.linearNames = {"lm_head.weight", "model.layers.*.linear_attn.*_proj*.weight",
        "model.layers.*.self_attn.*_proj.weight", "model.layers.*.mlp.gate.weight",
        "model.layers.*.mlp.shared_expert.*_proj.weight", "model.layers.*.mlp.shared_expert_gate.weight"};
}
void CloverModel::SetDataType(DataType type) {
    Check(type == BFLOAT16, "activations must be BF16");
    dataType = type;
}
void CloverModel::SetKVCacheDataType(DataType type) {
    Check(type == BFLOAT16, "attention KV must be BF16; recurrent state remains FP32");
    kvCacheDataType = type;
}
void CloverModel::InitParams() {
    basellm::InitParams();
    auto integer = [&](const char *name, int fallback) {
        auto it = weight.dicts.find(name);
        return it == weight.dicts.end() ? fallback : std::stoi(it->second);
    };
    auto real = [&](const char *name, float fallback) {
        auto it = weight.dicts.find(name);
        return it == weight.dicts.end() ? fallback : std::stof(it->second);
    };
    embed_dim = integer("hidden_size", 2048);
    block_cnt = integer("num_hidden_layers", 40);
    num_attention_heads = integer("num_attention_heads", 16);
    num_key_value_heads = integer("num_key_value_heads", 2);
    head_dim = integer("head_dim", 256);
    rotary_dim = int(head_dim * real("partial_rotary_factor", .25f));
    keyHeads = integer("linear_num_key_heads", 16);
    valueHeads = integer("linear_num_value_heads", 32);
    intermediate = integer("moe_intermediate_size", 512);
    num_experts = integer("num_experts", 256);
    num_experts_per_tok = integer("num_experts_per_tok", 8);
    blockSize = integer("block_size", 32);
    max_positions = integer("max_position_embeddings", 131072);
    if (contextPlan.configured) {
        Check(contextPlan.rope.type == "default" && contextPlan.effectiveLength <= max_positions,
            "only the native RoPE and context window are supported");
        max_positions = contextPlan.effectiveLength;
    }
    maskToken = integer("mask_token_id", 151669);
    vocabSize = integer("vocab_size", 248320);
    defaults.width = blockSize;
    defaults.presence = real("presence_penalty", .9f);
    defaults.frequency = real("frequency_penalty", 0.f);
    defaults.lengthPenalty = real("dllm_length_penalty", 10.f);
    defaults.lengthStart = integer("dllm_length_penalty_start", 8192);
    defaults.lengthWindow = integer("dllm_length_penalty_window", 8192);
    defaults.thinkBias = real("dllm_think_end_bias", 10.f);
    defaults.thinkStart = integer("dllm_think_end_start", 4096);
    defaults.thinkWindow = integer("dllm_think_end_window", 4096);
    defaults.thinkToken = integer("dllm_think_end_token_id", 248069);
    rms_norm_eps = real("rms_norm_eps", 1.e-6f);
    ropeTheta = real("rope_parameters.rope_theta", 1.e7f);
    coresetTopP = real("moe_block_coreset_top_p", .9f);
    coresetEnabled = weight.dicts["moe_block_coreset_enabled"] != "false";
    Check(head_dim == 256 && integer("linear_key_head_dim", 128) == 128 &&
          integer("linear_value_head_dim", 128) == 128 &&
          integer("linear_conv_kernel_dim", 4) == 4, "unsupported attention geometry");
    Check(blockSize == 32 && num_experts <= 256 && num_experts_per_tok <= 16 &&
          keyHeads > 0 && valueHeads % keyHeads == 0 && num_key_value_heads > 0 &&
          num_attention_heads % num_key_value_heads == 0, "unsupported block/router geometry");
    Check(integer("num_shared_experts", 1) == 1, "expected one shared expert");
    // Config flattening retains arrays as JSON. Honor layer_types instead of
    // silently assuming every fourth layer for other Clover checkpoints.
    std::string err;
    auto types = json11::Json::parse(weight.dicts["layer_types"], err);
    Check(err.empty() && (int)types.array_items().size() == block_cnt, "invalid layer_types");
    linearLayers.clear();
    kvCacheId = -1;
    for (int i = 0; i < block_cnt; ++i) {
        std::string type = types[i].string_value();
        Check(type == "linear_attention" || type == "full_attention", "unknown layer type");
        linearLayers.push_back(type == "linear_attention");
        if (!linearLayers.back() && kvCacheId < 0) kvCacheId = i;
    }
    Check(kvCacheId >= 0, "missing full attention layer");
}
std::map<std::string, std::vector<std::pair<std::string, DataType>>>
CloverModel::GetTensorMap(const std::vector<std::string> &names) {
    std::map<std::string, std::vector<std::pair<std::string, DataType>>> result;
    for (const auto &name : names) {
        bool scalar = name.find("norm.weight") != std::string::npos ||
            name.find(".A_log") != std::string::npos || name.find(".dt_bias") != std::string::npos ||
            name.find(".bd_softmax_") != std::string::npos;
        result[name].push_back({name, scalar ? FLOAT32 : BFLOAT16});
    }
    return result;
}
void CloverModel::EnsureResident() {
#ifdef USE_CUDA
    if (resident) { FastllmCudaSetDevice(cudaDevice); return; }
    auto devices = deviceMap.empty() ? GetDeviceMap() : deviceMap;
    Check(devices.size() <= 1, "initial implementation requires one CUDA device");
    if (!devices.empty()) {
        const auto &name = devices.begin()->first;
        Check(name == "cuda" || name.rfind("cuda:", 0) == 0, "requires a CUDA device");
        if (name != "cuda") {
            const std::string ordinal = name.substr(5);
            Check(!ordinal.empty() && std::all_of(ordinal.begin(), ordinal.end(),
                [](char c) { return c >= '0' && c <= '9'; }), "requires exactly one CUDA ordinal");
            cudaDevice = std::stoi(ordinal);
        }
    }
    FastllmCudaSetDevice(cudaDevice);
    for (auto &item : weight.weight) item.second.ToDevice(DataDevice::CUDA, std::vector<int>{cudaDevice});
    std::vector<std::string> mergedSources;
    auto merge = [&](const std::string &destination, const std::vector<std::string> &sources) {
        int rows = 0;
        for (const auto &name : sources) {
            auto &w = weight[name];
            Check(w.dataType == BFLOAT16 && w.dims.size() == 2 && w.dims[1] == embed_dim, "invalid fused projection weight");
            rows += w.dims[0];
        }
        auto &fused = weight[destination];
        PrepareCloverData(fused, BFLOAT16, {rows, embed_dim}, cudaDevice);
        fused.name = destination; fused.isModelWeight = true; fused.weightType = WeightType::LINEAR;
        size_t offset = 0;
        for (const auto &name : sources) {
            auto &w = weight[name];
            FastllmCudaCopyFromDeviceToDevice((char *)fused.cudaData + offset, w.cudaData, w.GetBytes());
            offset += w.GetBytes(); mergedSources.push_back(name);
        }
    };
    for (int i = 0; i < block_cnt; ++i) {
        std::string p = Layer(i);
        if (linearLayers[i]) merge(p + "linear_attn.fused_in_proj.weight", {
            p + "linear_attn.in_proj_qkv.weight", p + "linear_attn.in_proj_z.weight",
            p + "linear_attn.in_proj_a.weight", p + "linear_attn.in_proj_b.weight"});
        merge(p + "mlp.shared_expert.gate_up_proj.weight", {
            p + "mlp.shared_expert.gate_proj.weight", p + "mlp.shared_expert.up_proj.weight"});
    }
    FastllmCudaStreamSynchronize(nullptr);
    // Keep checkpoint tensors available on the host, without duplicate resident weights.
    for (const auto &name : mergedSources) weight[name].ToDevice(DataDevice::CPU);
    resident = true;
#else
    Check(false, "requires a CUDA build");
#endif
}
void CloverModel::Norm(Data &input, const std::string &name, Data &output, Data *gate, bool zeroCentered) {
#ifdef USE_CUDA
    auto &w = weight[name];
    Check(w.dataType == FLOAT32 && w.Count(0) == input.dims.back(), "invalid normalization weight");
    PrepareCloverData(output, BFLOAT16, input.dims, cudaDevice);
    Check(FastllmCudaCloverNorm(input.cudaData, (float *)w.cudaData,
        gate ? gate->cudaData : nullptr, output.cudaData, input.Count(0) / input.dims.back(),
        input.dims.back(), rms_norm_eps, zeroCentered), "normalization failed");
#endif
}
void CloverModel::AddNorm(Data &residual, Data &input, const std::string &name, Data &output) {
#ifdef USE_CUDA
    auto &w = weight[name];
    Check(residual.dims == input.dims && w.dataType == FLOAT32 && w.Count(0) == input.dims.back(),
        "invalid residual normalization shape or weight");
    PrepareCloverData(output, BFLOAT16, input.dims, cudaDevice);
    Check(FastllmCudaCloverAddNorm(residual.cudaData, input.cudaData, (float *)w.cudaData,
        output.cudaData, input.Count(0) / input.dims.back(), input.dims.back(), rms_norm_eps, true),
        "residual normalization failed");
#endif
}
void CloverModel::Gdn(Data &input, int layer, std::pair<Data, Data> &cache, bool preview, Data &output, std::pair<Data, Data> *snapshots) {
#ifdef USE_CUDA
    std::string p = Layer(layer) + "linear_attn.";
    Data projected, normalized;
    Linear(input, weight[p + "fused_in_proj.weight"], *GetEmptyData(), projected);
    GdnCore(projected, layer, cache, preview, normalized, snapshots);
    Linear(normalized, weight[p + "out_proj.weight"], *GetEmptyData(), output);
#endif
}
void CloverModel::GdnCore(Data &projected, int layer, std::pair<Data, Data> &cache,
        bool preview, Data &normalized, std::pair<Data, Data> *snapshots) {
#ifdef USE_CUDA
    std::string p = Layer(layer) + "linear_attn.";
    const int t = projected.dims[0], channels = (2 * keyHeads + valueHeads) * 128;
    auto &conv = cache.first; auto &state = cache.second;
    if (conv.dims.empty()) PrepareCloverData(conv, BFLOAT16, {1, channels, 3}, cudaDevice, true);
    if (state.dims.empty()) PrepareCloverData(state, FLOAT32, {1, valueHeads, 128, 128}, cudaDevice, true);
    if (snapshots) {
        PrepareCloverData(snapshots->first, BFLOAT16, {t, 1, channels, 3}, cudaDevice);
        PrepareCloverData(snapshots->second, FLOAT32, {t, 1, valueHeads, 128, 128}, cudaDevice);
    }
    Data raw, mixed, g, beta, y, z, causalWorkspace;
    PrepareCloverData(raw, BFLOAT16, {t, channels}, cudaDevice);
    PrepareCloverData(z, BFLOAT16, {t, valueHeads, 128}, cudaDevice);
    PrepareCloverData(g, FLOAT32, {t, valueHeads}, cudaDevice);
    if (!preview) PrepareCloverData(beta, BFLOAT16, {t, valueHeads}, cudaDevice);
    Check(FastllmCudaCloverUnpackGdn(projected.cudaData, (float *)weight[p + "A_log"].cudaData,
        (float *)weight[p + "dt_bias"].cudaData, raw.cudaData, z.cudaData, (float *)g.cudaData,
        beta.cudaData, t, channels, valueHeads, nullptr), "GDN projection unpack failed");
    PrepareCloverData(mixed, BFLOAT16, {t, channels}, cudaDevice);
    if (preview)
        Check(FastllmCudaCloverPreviewConv(raw.cudaData, conv.cudaData,
            weight[p + "conv1d.weight"].cudaData, weight[p + "conv_right_weight"].cudaData,
            nullptr, nullptr, mixed.cudaData, 1, t, channels), "preview convolution failed");
    else
        Check(FastllmCudaCloverCausalConv(raw.cudaData, conv.cudaData,
            weight[p + "conv1d.weight"].cudaData, mixed.cudaData, snapshots ? snapshots->first.cudaData : nullptr, 1, t, channels), "causal convolution failed");
    // Read strided views directly from the convolved QKV buffer. This removes
    // three copies while keeping the BF16 values and FP32 recurrence unchanged.
    const auto *q = static_cast<const uint16_t *>(mixed.cudaData);
    const auto *k = q + keyHeads * 128;
    const auto *v = q + keyHeads * 256;
    PrepareCloverData(y, BFLOAT16, {t, valueHeads, 128}, cudaDevice);
    if (preview)
        Check(FastllmCudaCloverPreviewGdn(q, k, v, (float *)g.cudaData,
            (float *)state.cudaData, (float *)weight[p + "bd_softmax_logit_scale"].cudaData,
            (float *)weight[p + "bd_softmax_history_bias"].cudaData,
            (float *)weight[p + "bd_softmax_history_c"].cudaData,
            nullptr, nullptr, y.cudaData, 1, t, keyHeads, valueHeads, nullptr, channels), "preview GDN failed");
    else {
        size_t blockBytes = !snapshots && t >= 2048 && valueHeads >= 16
            ? FastllmCudaCloverPrefillGdnWorkspaceBytes(1, t, keyHeads, valueHeads) : 0;
        if (blockBytes) {
            if (gdnPrefillWorkspace.GetBytes() < blockBytes)
                PrepareCloverData(gdnPrefillWorkspace, INT32,
                    {int((blockBytes + 3) / 4)}, cudaDevice);
            Check(FastllmCudaCloverPrefillGdn(q, k, v, (float *)g.cudaData,
                beta.cudaData, (float *)state.cudaData, y.cudaData, 1, t, keyHeads, valueHeads,
                gdnPrefillWorkspace.cudaData, gdnPrefillWorkspace.GetBytes(), nullptr, channels),
                "block prefill GDN failed");
        } else {
            if (t >= 16) PrepareCloverData(causalWorkspace, FLOAT32,
                {t, 2 * keyHeads * 128 + valueHeads}, cudaDevice);
            Check(FastllmCudaCloverCausalGdn(q, k, v, (float *)g.cudaData,
                beta.cudaData, (float *)state.cudaData, y.cudaData, snapshots ? (float *)snapshots->second.cudaData : nullptr,
                1, t, keyHeads, valueHeads, nullptr, (float *)causalWorkspace.cudaData, channels), "causal GDN failed");
        }
    }
    z.Reshape(y.dims);
    Norm(y, p + "norm.weight", normalized, &z, false);
    normalized.Reshape({t, valueHeads * 128});
#endif
}
void CloverModel::FullAttention(Data &input, int layer, std::pair<Data, Data> &cache,
        int position, bool preview, Data &output) {
#ifdef USE_CUDA
    std::string p = Layer(layer) + "self_attn.";
    Data qgate, k, v, y;
    Linear(input, weight[p + "q_proj.weight"], *GetEmptyData(), qgate);
    Linear(input, weight[p + "k_proj.weight"], *GetEmptyData(), k);
    Linear(input, weight[p + "v_proj.weight"], *GetEmptyData(), v);
    FullAttentionCore(qgate, k, v, layer, cache, position, preview, y);
    Linear(y, weight[p + "o_proj.weight"], *GetEmptyData(), output);
#endif
}
void CloverModel::FullAttentionCore(Data &qgate, Data &k, Data &v, int layer,
        std::pair<Data, Data> &cache, int position, bool preview, Data &y) {
#ifdef USE_CUDA
    const int t = qgate.dims[0];
    std::string p = Layer(layer) + "self_attn.";
    Data q, gate, qn, kn, attentionWorkspace;
    k.Reshape({t, num_key_value_heads, head_dim}); v.Reshape(k.dims);
    if (t <= blockSize) {
        auto &qw = weight[p + "q_norm.weight"], &kw = weight[p + "k_norm.weight"];
        Check(qw.dataType == FLOAT32 && kw.dataType == FLOAT32 &&
            qw.Count(0) == head_dim && kw.Count(0) == head_dim, "invalid Q/K normalization weights");
        PrepareCloverData(qn, BFLOAT16, {t, num_attention_heads, head_dim}, cudaDevice);
        PrepareCloverData(kn, BFLOAT16, k.dims, cudaDevice);
        PrepareCloverData(gate, BFLOAT16, qn.dims, cudaDevice);
        Check(FastllmCudaCloverPrepareAttention(qgate.cudaData, k.cudaData, (float *)qw.cudaData,
            (float *)kw.cudaData, qn.cudaData, kn.cudaData, gate.cudaData, t, num_attention_heads,
            num_key_value_heads, rotary_dim, position, ropeTheta, rms_norm_eps), "attention preparation failed");
    } else {
        qgate.Reshape({t, num_attention_heads, 2 * head_dim});
        Split(qgate, 2, 0, head_dim, q); Split(qgate, 2, head_dim, 2 * head_dim, gate);
        Norm(q, p + "q_norm.weight", qn); Norm(k, p + "k_norm.weight", kn);
        Check(FastllmCudaCloverRotary(qn.cudaData, t, num_attention_heads, head_dim, rotary_dim, position, ropeTheta), "Q rotary failed");
        Check(FastllmCudaCloverRotary(kn.cudaData, t, num_key_value_heads, head_dim, rotary_dim, position, ropeTheta), "K rotary failed");
    }
    auto append = [&](Data &dst, const Data &src) {
        dst.dataType = BFLOAT16; dst.UpdateUnitSize();
        dst.ToDevice(DataDevice::CUDA, std::vector<int>{cudaDevice});
        int required = position + t;
        if (dst.expansionDims.empty() || dst.expansionDims[0] < required)
            dst.Expansion({(required + 255) / 256 * 256, num_key_value_heads, head_dim});
        CatDirect(dst, src, 0);
    };
    append(cache.first, kn); append(cache.second, v);
    PrepareCloverData(y, BFLOAT16, {t, num_attention_heads, head_dim}, cudaDevice);
    // Small query blocks otherwise expose only a few CTAs across the KV heads.
    // The FlashInfer planner splits the history and merges their softmax states.
    if (t <= blockSize && position + t > 256) {
        size_t chunks = (position + t + 255) / 256;
        size_t bytes = chunks * t * num_attention_heads * (head_dim * sizeof(uint16_t) + sizeof(float));
        PrepareCloverData(attentionWorkspace, INT32, {int((bytes + 3) / 4)}, cudaDevice);
    }
    bool ok = FastllmCudaCloverAttention(qn.cudaData, cache.first.cudaData, cache.second.cudaData,
        y.cudaData, t, position, num_attention_heads, num_key_value_heads, preview, nullptr, attentionWorkspace.cudaData);
    // Restore even on unsupported attention dispatch. Spare bytes are never
    // considered committed history and are overwritten by the next append.
    if (preview) {
        cache.first.Resize({position, num_key_value_heads, head_dim});
        cache.second.Resize({position, num_key_value_heads, head_dim});
    }
    Check(ok, "BF16 attention requires a supported CUDA architecture");
    SigmoidMulTo(y, gate);
    y.Reshape({t, num_attention_heads * head_dim});
#endif
}
void CloverModel::Moe(Data &input, int layer, bool preview, Data &output, bool graphAllowed, int batch) {
#ifdef USE_CUDA
    const int t = input.dims[0];
    Check(batch > 0 && t % batch == 0, "invalid MoE request geometry");
    std::string p = Layer(layer) + "mlp.";
    auto eager = [&](Data &input, Data &output, bool graphPath = false) {
        Data sharedGateUp, sharedMiddle, shared, sharedScale;
        Linear(input, weight[p + "shared_expert.gate_up_proj.weight"], *GetEmptyData(), sharedGateUp);
        Swiglu(sharedGateUp, sharedMiddle);
        Linear(sharedMiddle, weight[p + "shared_expert.down_proj.weight"], *GetEmptyData(), shared);
        Linear(input, weight[p + "shared_expert_gate.weight"], *GetEmptyData(), sharedScale);
        Data router, ids, scores, scratchIds, scratchScores, allowed, middle, parts, expertWorkspace,
            packedExperts;
        Linear(input, weight[p + "gate.weight"], *GetEmptyData(), router);
        PrepareCloverData(ids, INT32, {t, num_experts_per_tok}, cudaDevice);
        PrepareCloverData(scores, FLOAT32, ids.dims, cudaDevice);
        if (preview && coresetEnabled) {
            PrepareCloverData(scratchIds, INT32, ids.dims, cudaDevice);
            PrepareCloverData(scratchScores, FLOAT32, ids.dims, cudaDevice);
            // Byte mask uses a padded INT32 allocation; kernels address bytes.
            PrepareCloverData(allowed, INT32, {(batch * num_experts + 3) / 4}, cudaDevice);
        }
        Check(FastllmCudaCloverRoute(router.cudaData, (int *)ids.cudaData, (float *)scores.cudaData,
            (int *)scratchIds.cudaData, (float *)scratchScores.cudaData, (unsigned char *)allowed.cudaData,
            batch, t / batch, num_experts, num_experts_per_tok, preview && coresetEnabled ? coresetTopP : 0.f), "routing failed");
        PrepareCloverData(middle, BFLOAT16, {t, num_experts_per_tok, intermediate}, cudaDevice);
        PrepareCloverData(parts, BFLOAT16, {t, num_experts_per_tok, embed_dim}, cudaDevice);
        PrepareCloverData(output, BFLOAT16, input.dims, cudaDevice);
        if (t >= 4) PrepareCloverData(expertWorkspace, INT32,
            {num_experts * (t + 1) + num_experts + (t * num_experts_per_tok + 15) / 16 + 1}, cudaDevice);
        size_t packedBytes = FastllmCudaCloverExpertsPackedBytes(t, embed_dim, intermediate, num_experts, num_experts_per_tok);
        if (packedBytes && (t > blockSize || graphPath)) PrepareCloverData(packedExperts, INT32, {int((packedBytes + 3) / 4)}, cudaDevice);
        Check(FastllmCudaCloverExperts(input.cudaData, weight[p + "experts.gate_up_proj"].cudaData,
                                       weight[p + "experts.down_proj"].cudaData, (int *)ids.cudaData,
                                       (float *)scores.cudaData, middle.cudaData, parts.cudaData,
                                       output.cudaData, t, embed_dim, intermediate, num_experts,
                                       num_experts_per_tok, nullptr, (int *)expertWorkspace.cudaData,
                                       packedExperts.cudaData, shared.cudaData, sharedScale.cudaData),
              "experts failed");
    };
    const int width = t / batch;
    if (!graphAllowed || batch > 8 || width < 4 || width > blockSize || blockSize % width ||
        FastllmCudaGraphIsCapturingFast()) {
        eager(input, output);
        return;
    }
    if (!moeGraphs) moeGraphs = std::make_shared<MoeGraphs>(cudaDevice);
    if (!moeGraphs->enabled) { eager(input, output); return; }
    moeGraphs->SetRows(t, batch);
    auto &entry = moeGraphs->entries[{layer, preview}];
    if (!entry) {
        auto candidate = std::make_unique<MoeGraphs::Entry>();
        bool poolActive = false, captureActive = false, ready = false;
        try {
            PrepareCloverData(candidate->input, BFLOAT16, input.dims, cudaDevice);
            Check(FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(candidate->input.cudaData,
                input.cudaData, input.GetBytes()), "MoE graph input copy failed");
            // MoE has no committed state. Warmup resolves lazy GEMM resources
            // and tensor maps without advancing any KV or recurrent cache.
            eager(candidate->input, candidate->output, true);
            Check(cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess, "MoE graph warmup failed");
            poolActive = FastllmCudaGraphMemoryPoolBegin();
            if (poolActive && (captureActive = FastllmCudaGraphBeginCapture())) {
                eager(candidate->input, candidate->output, true);
                bool captured = FastllmCudaGraphEndCapture(&candidate->graph);
                captureActive = false;
                bool pinned = FastllmCudaGraphMemoryPoolEnd(candidate->reserved);
                poolActive = false;
                ready = captured && pinned && !FastllmCudaGetThreadError() && candidate->graph &&
                    FastllmCudaGraphInstantiate(candidate->graph, &candidate->executable);
            }
        } catch (const std::exception &) {
            // Discard an invalid/unsupported capture, then execute this
            // stateless operation normally. Real device errors still surface
            // from the eager operation's checked CUDA calls.
        }
        if (captureActive) FastllmCudaGraphEndCapture(&candidate->graph);
        if (poolActive) FastllmCudaGraphMemoryPoolAbort();
        if (!ready) {
            cudaStreamSynchronize(cudaStreamPerThread);
            candidate.reset();
            FastllmCudaClearLastError();
            FastllmCudaClearThreadError();
            FastllmCudaClearGraphError();
            candidate = std::make_unique<MoeGraphs::Entry>();
            candidate->disabled = true;
        }
        entry = std::move(candidate);
    }
    if (entry->disabled) { eager(input, output); return; }
    Check(FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(entry->input.cudaData,
        input.cudaData, input.GetBytes()), "MoE graph input copy failed");
    // Launch failures are execution errors, not a reason to silently accept
    // stale output or replay a potentially partially executed graph.
    Check(FastllmCudaGraphLaunch(entry->executable), "MoE graph replay failed");
    // ForwardLogits consumes this view on the same stream before another
    // replay. The graph owns its storage; width changes synchronize before
    // releasing it, and the local view never escapes the forward call.
    output.Resize(entry->output.dims);
    output.FakeFrom(entry->output, 0);
#endif
}

void CloverModel::ForwardLogits(const std::vector<int> &tokens,
        std::vector<std::pair<Data, Data>> &cache, bool preview, Data &logits, bool lastOnly, CloverSnapshots *snapshots, bool needLogits) {
    Check(!tokens.empty(), "empty input");
    Check(!preview || (tokens.size() <= (size_t)blockSize && blockSize % tokens.size() == 0), "invalid preview block");
    EnsureResident();
    Check(cache.empty() || (int)cache.size() == block_cnt, "cache layer count mismatch");
    if (cache.empty()) cache.resize(block_cnt);
    int position = cache[kvCacheId].first.dims.empty() ? 0 : cache[kvCacheId].first.dims[0];
    Check(position + tokens.size() <= (size_t)max_positions, "context limit exceeded");
    if (snapshots) {
        Check(!preview, "preview cannot write snapshots");
        snapshots->position = position; snapshots->tokens = tokens.size();
        snapshots->layers.resize(block_cnt);
    }
    std::vector<float> values(tokens.begin(), tokens.end());
    Data ids(FLOAT32, {1, (int)tokens.size()}, values), hidden;
    Embedding(ids, weight["model.embed_tokens.weight"], hidden);
    ToDataType(hidden, BFLOAT16);
    hidden.ToDevice(DataDevice::CUDA, std::vector<int>{cudaDevice});
    hidden.Reshape({(int)tokens.size(), embed_dim});
    Data norm;
    for (int i = 0; i < block_cnt; ++i) {
        Data attention, post, moe;
        if (i == 0) Norm(hidden, Layer(i) + "input_layernorm.weight", norm);
        if (linearLayers[i]) Gdn(norm, i, cache[i], preview, attention, snapshots ? &snapshots->layers[i] : nullptr);
        else FullAttention(norm, i, cache[i], position, preview, attention);
#ifdef USE_CUDA
        // Replays can normalize directly into the graph's fixed input buffer.
        // A new width, disabled graph or external capture uses ordinary owned
        // storage. Moe then either captures it or copies it into a valid graph.
        if ((preview || snapshots) && moeGraphs && moeGraphs->enabled &&
            moeGraphs->rows == (int)tokens.size() && moeGraphs->batch == 1 && !FastllmCudaGraphIsCapturingFast()) {
            auto found = moeGraphs->entries.find({i, preview});
            if (found != moeGraphs->entries.end() && found->second && !found->second->disabled &&
                found->second->input.dims == hidden.dims)
                post.FakeFrom(found->second->input, 0);
        }
#endif
        AddNorm(hidden, attention, Layer(i) + "post_attention_layernorm.weight", post);
        Moe(post, i, preview, moe, preview || snapshots != nullptr);
        if (i + 1 < block_cnt) AddNorm(hidden, moe, Layer(i + 1) + "input_layernorm.weight", norm);
        else AddTo(hidden, moe);
    }
    if (!needLogits) return;
    Data final, last;
    if (lastOnly) {
        Split(hidden, 0, tokens.size() - 1, tokens.size(), last);
        Norm(last, "model.norm.weight", final);
    } else Norm(hidden, "model.norm.weight", final);
    Linear(final, weight["lm_head.weight"], *GetEmptyData(), logits);
    ToDataType(logits, FLOAT32);
}

void CloverModel::ForwardLogitsBatch(const std::vector<std::vector<int>> &tokens,
        const std::vector<std::vector<std::pair<Data, Data>> *> &caches,
        bool preview, std::vector<Data> &logits,
        const std::vector<CloverSnapshots *> &snapshots) {
    const int batch = tokens.size();
    Check(batch >= 2 && batch <= 8 && caches.size() == tokens.size(), "invalid request batch size");
    const int width = tokens[0].size(), rows = batch * width;
    Check(width > 0 && width <= blockSize && blockSize % width == 0, "invalid batch block width");
    Check(snapshots.empty() || (!preview && snapshots.size() == tokens.size()), "invalid batch snapshots");
    for (int b = 0; b < batch; ++b) {
        Check(caches[b] && tokens[b].size() == (size_t)width, "invalid request cache or block width");
        Check(snapshots.empty() || snapshots[b], "missing request snapshots");
        for (int prior = 0; prior < b; ++prior) {
            Check(caches[b] != caches[prior], "request caches must be independent");
            Check(snapshots.empty() || snapshots[b] != snapshots[prior], "request snapshots must be independent");
        }
    }
    // Small row counts select a different BF16 linear/expert dispatch when
    // combined. Preserve the established path for short context-tail blocks.
    if (width < 16) {
        logits.resize(batch);
        for (int b = 0; b < batch; ++b)
            ForwardLogits(tokens[b], *caches[b], preview, logits[b], false,
                snapshots.empty() ? nullptr : snapshots[b]);
        return;
    }
#ifdef USE_CUDA
    EnsureResident();
    std::vector<int> positions(batch);
    std::vector<float> values;
    for (int b = 0; b < batch; ++b) {
        auto &cache = *caches[b];
        Check(cache.empty() || cache.size() == (size_t)block_cnt, "cache layer count mismatch");
        if (cache.empty()) cache.resize(block_cnt);
        positions[b] = cache[kvCacheId].first.dims.empty() ? 0 : cache[kvCacheId].first.dims[0];
        Check(positions[b] + width <= max_positions, "context limit exceeded");
        if (!snapshots.empty()) {
            snapshots[b]->position = positions[b]; snapshots[b]->tokens = width;
            snapshots[b]->layers.resize(block_cnt);
        }
        values.insert(values.end(), tokens[b].begin(), tokens[b].end());
    }
    Data ids(FLOAT32, {batch, width}, values), hidden, norm;
    Embedding(ids, weight["model.embed_tokens.weight"], hidden);
    ToDataType(hidden, BFLOAT16);
    hidden.ToDevice(DataDevice::CUDA, std::vector<int>{cudaDevice});
    hidden.Reshape({rows, embed_dim});
    auto project = [&](Data &input, Data &matrix, Data &output) {
        if (rows <= 32) {
            Linear(input, matrix, *GetEmptyData(), output);
            return;
        }
        // Larger M selects a different BF16 reduction. Preserve the validated
        // M=16/32 dispatch while writing into one combined allocation.
        PrepareCloverData(output, BFLOAT16, {rows, matrix.dims[0]}, cudaDevice);
        for (int row = 0; row < rows; row += 32) {
            const int count = std::min(32, rows - row);
            Data x(BFLOAT16, {count, input.dims.back()}), y(BFLOAT16, {count, matrix.dims[0]});
            x.FakeFrom(input, (size_t)row * input.dims.back() * sizeof(uint16_t));
            y.FakeFrom(output, (size_t)row * matrix.dims[0] * sizeof(uint16_t));
            Linear(x, matrix, *GetEmptyData(), y);
        }
    };
    for (int layer = 0; layer < block_cnt; ++layer) {
        Data attention, post, moe;
        if (layer == 0) Norm(hidden, Layer(layer) + "input_layernorm.weight", norm);
        // Project all request rows together. Only the stateful attention core
        // uses per-request views; it writes directly into the combined buffer.
        const std::string p = Layer(layer) + (linearLayers[layer] ? "linear_attn." : "self_attn.");
        const int attentionDim = linearLayers[layer] ? valueHeads * 128 : num_attention_heads * head_dim;
        Data projected, keys, values, attended;
        PrepareCloverData(attended, BFLOAT16, {rows, attentionDim}, cudaDevice);
        if (linearLayers[layer])
            project(norm, weight[p + "fused_in_proj.weight"], projected);
        else {
            project(norm, weight[p + "q_proj.weight"], projected);
            project(norm, weight[p + "k_proj.weight"], keys);
            project(norm, weight[p + "v_proj.weight"], values);
        }
        for (int b = 0; b < batch; ++b) {
            Data input(BFLOAT16, {width, projected.dims.back()}), output(BFLOAT16, {width, attentionDim});
            input.FakeFrom(projected, (size_t)b * width * projected.dims.back() * sizeof(uint16_t));
            output.FakeFrom(attended, (size_t)b * width * attentionDim * sizeof(uint16_t));
            if (linearLayers[layer])
                GdnCore(input, layer, (*caches[b])[layer], preview, output,
                    snapshots.empty() ? nullptr : &snapshots[b]->layers[layer]);
            else {
                Data k(BFLOAT16, {width, keys.dims.back()}), v(BFLOAT16, {width, values.dims.back()});
                k.FakeFrom(keys, (size_t)b * width * keys.dims.back() * sizeof(uint16_t));
                v.FakeFrom(values, (size_t)b * width * values.dims.back() * sizeof(uint16_t));
                FullAttentionCore(input, k, v, layer, (*caches[b])[layer], positions[b], preview, output);
            }
        }
        project(attended, weight[p + (linearLayers[layer] ? "out_proj.weight" : "o_proj.weight")], attention);
        if (moeGraphs && moeGraphs->enabled && moeGraphs->rows == rows && moeGraphs->batch == batch &&
            !FastllmCudaGraphIsCapturingFast()) {
            auto found = moeGraphs->entries.find({layer, preview});
            if (found != moeGraphs->entries.end() && found->second && !found->second->disabled &&
                found->second->input.dims == hidden.dims)
                post.FakeFrom(found->second->input, 0);
        }
        AddNorm(hidden, attention, Layer(layer) + "post_attention_layernorm.weight", post);
        Moe(post, layer, preview, moe, true, batch);
        if (layer + 1 < block_cnt) AddNorm(hidden, moe, Layer(layer + 1) + "input_layernorm.weight", norm);
        else AddTo(hidden, moe);
    }
    Data final, combined;
    Norm(hidden, "model.norm.weight", final);
    Linear(final, weight["lm_head.weight"], *GetEmptyData(), combined);
    logits.resize(batch);
    for (int b = 0; b < batch; ++b) {
        Data view(combined.dataType, {width, combined.dims.back()});
        view.FakeFrom(combined, (size_t)b * combined.GetBytes() / batch);
        // Cast directly into each request's owned logits; avoid splitting a
        // full-vocabulary FP32 temporary after converting the entire batch.
        ToDataType(view, logits[b], FLOAT32);
    }
#else
    throw std::runtime_error("Clover: batch forward requires CUDA");
#endif
}

void CloverModel::RestorePrefix(std::vector<std::pair<Data, Data>> &cache,
        const CloverSnapshots &snapshots, int acceptedInputs) {
    Check(acceptedInputs > 0 && acceptedInputs <= snapshots.tokens &&
          cache.size() == (size_t)block_cnt && snapshots.layers.size() == cache.size(), "invalid accepted prefix");
    if (acceptedInputs == snapshots.tokens) return;
#ifdef USE_CUDA
    for (int i = 0; i < block_cnt; ++i) {
        if (linearLayers[i]) {
            for (bool state : {false, true}) {
                auto &dst = state ? cache[i].second : cache[i].first;
                const auto &src = state ? snapshots.layers[i].second : snapshots.layers[i].first;
                const size_t bytes = dst.GetBytes();
                FastllmCudaCopyFromDeviceToDevice(dst.cudaData,
                    (char *)src.cudaData + (size_t)(acceptedInputs - 1) * bytes, bytes);
            }
        } else {
            const int length = snapshots.position + acceptedInputs;
            cache[i].first.Resize({length, num_key_value_heads, head_dim});
            cache[i].second.Resize({length, num_key_value_heads, head_dim});
        }
    }
#endif
}

int CloverModel::Forward(const Data &inputIds, const Data &attentionMask, const Data &positionIds,
        std::vector<std::pair<Data, Data>> &cache, const GenerationConfig &config,
        const LastTokensManager &lastTokens, std::vector<float> *retLogits) {
    Check(inputIds.dims.size() == 2 && inputIds.dims[0] == 1, "single-request forward expected");
    Data hostIds; hostIds.CopyFrom(inputIds); hostIds.ToDevice(DataDevice::CPU);
    Check(hostIds.dataType == FLOAT32, "expected FLOAT32 token IDs");
    auto p = (const float *)hostIds.cpuData;
    std::vector<int> tokens(p, p + inputIds.Count(0));
    Data logits;
    ForwardLogits(tokens, cache, false, logits, true);
    logits.ToDevice(DataDevice::CPU);
    if (retLogits) retLogits->assign((float *)logits.cpuData, (float *)logits.cpuData + logits.Count(0));
    LastTokensUnit empty;
    return LLMSampling(logits, 0, config, lastTokens.units.empty() ? empty : lastTokens.units[0]);
}
void CloverModel::WarmUp() {
    // Loading validates residency; real request lengths are initialized lazily.
    EnsureResident();
}
std::string CloverModel::MakeInput(const std::string &history, int, const std::string &input) {
    return history + input;
}
std::string CloverModel::MakeHistory(const std::string &history, int, const std::string &input, const std::string &output) {
    return history + input + output;
}
}
