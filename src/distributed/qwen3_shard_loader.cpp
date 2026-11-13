#include "distributed/qwen3_shard_loader.h"
#include "json11.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace fastllm {
namespace distributed {
namespace {
using Json = json11::Json;
namespace fs = std::filesystem;
constexpr uint64_t kMaxHeaderBytes = 64ULL * 1024 * 1024;

/** 校验输入约束；失败时抛出包含上下文的异常。 */
void Require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error("Qwen3 shard loader: " + message);
    }
}

/** 返回经过溢出检查的非负整数乘积。 */
uint64_t Multiply(uint64_t a, uint64_t b) {
    Require(b == 0 || a <= std::numeric_limits<uint64_t>::max() / b, "size overflow");
    return a * b;
}

/** 从 JSON 读取可精确表示的非负整数，拒绝小数、负数和缺失字段。 */
uint64_t Integer(const Json &value, const std::string &name) {
    double n = value.number_value();
    Require(value.is_number() && std::isfinite(n) && n >= 0 &&
            n <= 9007199254740991.0 && n == std::floor(n), "invalid integer " + name);
    return static_cast<uint64_t>(n);
}

/** 返回未量化浮点存储宽度；其他 dtype 必须走独立的量化布局适配。 */
uint64_t ItemSize(const std::string &dtype) {
    Require(dtype == "F16" || dtype == "BF16" || dtype == "F32", "unsupported dtype " + dtype);
    return dtype == "F32" ? 4 : 2;
}

/** 读取指定字节区间，短读/越界失败；不读取该区间以外的张量 payload。 */
void ReadAt(std::ifstream &stream, uint64_t offset, uint8_t *destination, uint64_t bytes) {
    Require(offset <= static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max()) &&
            bytes <= static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()), "I/O size overflow");
    stream.seekg(static_cast<std::streamoff>(offset));
    stream.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(bytes));
    Require(stream.good() && static_cast<uint64_t>(stream.gcount()) == bytes, "short read or seek failure");
}

/**
 * 拒绝同一对象内的重复 JSON 字段，避免 json11 默认覆盖前一个字段。
 *
 * 仅接收已通过 json11 语法检查的文本；遍历时跳过字符串中的括号和转义，
 * 键名先解码再比较，因此 Unicode 转义与其字面形式不能绕过重复检查。
 * @param text 已验证语法的 JSON 文本。
 * @param context 出错时报告的配置、索引或 safetensors 文件路径。
 */
void RejectDuplicateKeys(const std::string &text, const std::string &context) {
    std::vector<std::set<std::string>> objects;
    size_t position = 0;
    while (position < text.size()) {
        char ch = text[position++];
        if (ch == '{') {
            objects.emplace_back();
        } else if (ch == '}') {
            Require(!objects.empty(), context + ": invalid object nesting");
            objects.pop_back();
        } else if (ch == '"') {
            size_t begin = position - 1;
            while (position < text.size() && text[position] != '"') {
                if (text[position] == '\\') {
                    ++position;
                }
                ++position;
            }
            Require(position < text.size(), context + ": unterminated string");
            ++position;
            size_t next = text.find_first_not_of(" \t\r\n", position);
            if (next != std::string::npos && text[next] == ':') {
                std::string error;
                Json key = Json::parse(text.substr(begin, position - begin), error);
                Require(error.empty() && key.is_string() && !objects.empty(), context + ": invalid object key");
                Require(objects.back().insert(key.string_value()).second,
                        context + ": duplicate JSON key: " + key.string_value());
            }
        }
    }
}

/** 解析有大小上限的 JSON 文本，要求顶层为对象且所有对象的键唯一。 */
Json ParseObject(const std::string &text, const std::string &context) {
    std::string error;
    Json json = Json::parse(text, error);
    Require(error.empty() && json.is_object(), context + ": invalid JSON: " + error);
    RejectDuplicateKeys(text, context);
    return json;
}

/** 读取配置/索引文件，不用于读取权重 payload。 */
Json ReadJson(const fs::path &path) {
    uint64_t size = fs::file_size(path);
    Require(size > 0 && size <= kMaxHeaderBytes, "invalid JSON size: " + path.string());
    std::ifstream stream(path, std::ios::binary);
    std::string text(static_cast<size_t>(size), '\0');
    ReadAt(stream, 0, reinterpret_cast<uint8_t*>(&text[0]), size);
    return ParseObject(text, path.string());
}

/** 将 n 个单位均匀分成 world 份，返回非空的半开区间。 */
std::pair<uint64_t, uint64_t> Partition(uint64_t n, int rank, int world) {
    Require(world > 0 && rank >= 0 && rank < world && n >= static_cast<uint64_t>(world),
            "empty shard or invalid rank/world size");
    // 商余分解避免 n * rank 溢出；较前 rank 多分一个单位。
    uint64_t width = n / world, remainder = n % world;
    uint64_t begin = width * rank + std::min<uint64_t>(rank, remainder);
    return {begin, begin + width + (static_cast<uint64_t>(rank) < remainder ? 1 : 0)};
}

struct SourceTensor {
    fs::path file;
    std::string dtype;
    std::vector<uint64_t> shape;
    uint64_t offset = 0, bytes = 0, fileSize = 0;
};
} // namespace

struct Qwen3ShardLoader::Impl {
    std::map<std::string, SourceTensor> sources;
    Qwen3ShardPlan plan;

    /** 校验文件头与全部数据区间，只保存 metadata，不读取张量。 */
    void IndexFile(const fs::path &path) {
        uint64_t fileSize = fs::file_size(path);
        Require(fileSize >= 8, "truncated safetensors: " + path.string());
        std::ifstream stream(path, std::ios::binary);
        uint8_t prefix[8];
        ReadAt(stream, 0, prefix, 8);
        uint64_t headerSize = 0;
        for (int i = 0; i < 8; ++i) {
            headerSize |= static_cast<uint64_t>(prefix[i]) << (8 * i);
        }
        Require(headerSize > 0 && headerSize <= kMaxHeaderBytes && headerSize <= fileSize - 8,
                "invalid safetensors header size: " + path.string());
        std::string text(static_cast<size_t>(headerSize), '\0');
        ReadAt(stream, 8, reinterpret_cast<uint8_t*>(&text[0]), headerSize);
        Json header = ParseObject(text, path.string());
        uint64_t base = 8 + headerSize;
        std::vector<std::pair<uint64_t, uint64_t>> intervals;

        for (const auto &entry : header.object_items()) {
            if (entry.first == "__metadata__") {
                continue;
            }
            const Json &meta = entry.second;
            Require(meta.is_object() && meta["dtype"].is_string(), "invalid tensor " + entry.first);
            SourceTensor tensor;
            tensor.file = path;
            tensor.fileSize = fileSize;
            tensor.dtype = meta["dtype"].string_value();
            uint64_t elements = 1;
            Require(meta["shape"].is_array() && !meta["shape"].array_items().empty() &&
                    meta["shape"].array_items().size() <= 2, "expected vector/matrix: " + entry.first);
            for (const auto &dimension : meta["shape"].array_items()) {
                uint64_t dim = Integer(dimension, entry.first + ".shape");
                Require(dim > 0, "zero dimension: " + entry.first);
                tensor.shape.push_back(dim);
                elements = Multiply(elements, dim);
            }
            Require(meta["data_offsets"].is_array() && meta["data_offsets"].array_items().size() == 2,
                    "invalid offsets: " + entry.first);
            uint64_t begin = Integer(meta["data_offsets"][0], entry.first + ".offset_begin");
            uint64_t end = Integer(meta["data_offsets"][1], entry.first + ".offset_end");
            tensor.bytes = Multiply(elements, ItemSize(tensor.dtype));
            Require(begin <= end && end <= fileSize - base && end - begin == tensor.bytes,
                    "tensor size/offset mismatch: " + entry.first);
            tensor.offset = base + begin;
            Require(sources.emplace(entry.first, tensor).second, "duplicate tensor: " + entry.first);
            intervals.push_back({begin, end});
        }

        std::sort(intervals.begin(), intervals.end());
        uint64_t end = 0;
        for (const auto &interval : intervals) {
            Require(interval.first == end, "overlapping or noncontiguous safetensors payload: " + path.string());
            end = interval.second;
        }
        Require(end == fileSize - base, "unindexed safetensors payload: " + path.string());
    }

    /** 打开标准 HF 单文件或 weight_map 分片索引，验证映射与真实文件头一致。 */
    void IndexDirectory(const fs::path &directory) {
        fs::path index = directory / "model.safetensors.index.json";
        if (!fs::exists(index)) {
            IndexFile(directory / "model.safetensors");
            return;
        }
        Json mapping = ReadJson(index)["weight_map"];
        Require(mapping.is_object() && !mapping.object_items().empty(), "invalid weight_map");
        std::set<std::string> files;
        for (const auto &entry : mapping.object_items()) {
            std::string name = entry.second.string_value();
            Require(entry.second.is_string() && !name.empty() && fs::path(name).filename() == fs::path(name) &&
                    fs::path(name).extension() == ".safetensors", "invalid weight_map file: " + name);
            files.insert(name);
        }
        for (const auto &name : files) {
            IndexFile(directory / name);
        }
        Require(sources.size() == mapping.object_items().size(), "weight_map/header tensor count mismatch");
        for (const auto &entry : mapping.object_items()) {
            auto found = sources.find(entry.first);
            Require(found != sources.end() && found->second.file.filename().string() == entry.second.string_value(),
                    "weight_map/header mismatch: " + entry.first);
        }
    }

    /** 根据配置构造所有目标权重，先验证完整模型再允许任何 payload 读取。 */
    void Build(const Json &config, int rank, int world, ShardModel model) {
        const bool moe = model == ShardModel::Qwen2Moe;
        Require(config["model_type"].string_value() == (moe ? "qwen2_moe" : "qwen3"),
                moe ? "expected model_type=qwen2_moe" : "only dense model_type=qwen3 is supported");
        plan.model = model;
        Require(config["quantization_config"].is_null(), "quantized checkpoints require scale-layout support");
        auto positive = [&](const std::string &key) {
            uint64_t n = Integer(config[key], key);
            Require(n > 0 && n <= std::numeric_limits<int>::max(), "invalid model dimension " + key);
            return n;
        };
        uint64_t layers = positive("num_hidden_layers"), hidden = positive("hidden_size");
        uint64_t qHeads = positive("num_attention_heads"), kvHeads = positive("num_key_value_heads");
        uint64_t intermediate = positive(moe ? "moe_intermediate_size" : "intermediate_size");
        uint64_t vocab = positive("vocab_size");
        if (moe) {
            Require(world == 1 || world == 2 || world == 4, "EP size must be 1, 2 or 4");
            plan.expertCount = positive("num_experts");
            plan.expertsPerToken = positive("num_experts_per_tok");
            plan.sharedIntermediateSize = positive("shared_expert_intermediate_size");
            Require(plan.expertsPerToken <= plan.expertCount && plan.expertCount <= 256 &&
                    intermediate <= 262144 && plan.sharedIntermediateSize <= 262144,
                    "unsupported MoE expert dimensions");
            Require(config["norm_topk_prob"].is_null() || config["norm_topk_prob"].is_bool(),
                    "invalid norm_topk_prob");
            plan.normTopkProb = config["norm_topk_prob"].bool_value();
            Require(config["decoder_sparse_step"].is_null() ||
                    Integer(config["decoder_sparse_step"], "decoder_sparse_step") == 1,
                    "only all-MoE layers (decoder_sparse_step=1) are supported");
            Require(config["mlp_only_layers"].is_null() ||
                    (config["mlp_only_layers"].is_array() && config["mlp_only_layers"].array_items().empty()),
                    "dense mlp_only_layers are unsupported");
        }
        Require(qHeads % kvHeads == 0, "Q heads must be divisible by KV heads");
        Require(!config["head_dim"].is_null() || hidden % qHeads == 0, "head_dim is required");
        uint64_t headDim = config["head_dim"].is_null() ? hidden / qHeads : positive("head_dim");
        Require(headDim > 0, "invalid head_dim");
        Require(config["tie_word_embeddings"].is_null() || config["tie_word_embeddings"].is_bool(),
                "invalid tie_word_embeddings");
        bool tied = config["tie_word_embeddings"].bool_value();

        plan.rank = rank;
        plan.worldSize = world;
        plan.hiddenSize = hidden;
        plan.layerCount = layers;
        plan.headDim = headDim;
        plan.queryHeads = qHeads;
        plan.kvHeads = kvHeads;
        plan.vocabSize = vocab;
        auto forwardNumber = [&](const std::string &key, float fallback) {
            if (config[key].is_null()) {
                return fallback;
            }
            double value = config[key].number_value();
            if (!config[key].is_number() || !std::isfinite(value) || value <= 0 ||
                value > std::numeric_limits<float>::max() || static_cast<float>(value) <= 0) {
                plan.forwardUnsupportedReason = "invalid forward config: " + key;
                return fallback;
            }
            return static_cast<float>(value);
        };
        plan.rmsNormEps = forwardNumber("rms_norm_eps", 1e-6f);
        plan.ropeTheta = forwardNumber("rope_theta", moe ? 10000.0f : 1000000.0f);
        if ((!config["hidden_act"].is_null() && config["hidden_act"].string_value() != "silu") ||
            !config["rope_scaling"].is_null() || !config["rope_parameters"].is_null() ||
            (!config["use_sliding_window"].is_null() &&
             (!config["use_sliding_window"].is_bool() || config["use_sliding_window"].bool_value())) ||
            (!config["partial_rotary_factor"].is_null() &&
             (!config["partial_rotary_factor"].is_number() || config["partial_rotary_factor"].number_value() != 1.0))) {
            plan.forwardUnsupportedReason = "unsupported forward activation/rope/sliding-window configuration";
        }
        if (!config["layer_types"].is_null()) {
            if (!config["layer_types"].is_array() || config["layer_types"].array_items().size() != layers) {
                plan.forwardUnsupportedReason = "invalid forward layer_types";
            }
            for (const auto &type : config["layer_types"].array_items()) {
                if (type.string_value() != "full_attention") {
                    plan.forwardUnsupportedReason = "only full_attention layers are supported";
                }
            }
        }
        auto mlp = Partition(intermediate, moe ? 0 : rank, moe ? 1 : world);
        auto vocabulary = Partition(vocab, moe ? 0 : rank, moe ? 1 : world);
        Partition(qHeads, moe ? 0 : rank, moe ? 1 : world);
        uint64_t group = qHeads / kvHeads;
        std::pair<uint64_t, uint64_t> q, kv;
        if (moe) {
            q = {0, qHeads};
            kv = {0, kvHeads};
        } else if (static_cast<uint64_t>(world) <= kvHeads) {
            kv = Partition(kvHeads, rank, world);
            q = {kv.first * group, kv.second * group};
        } else {
            // 对同一 KV 头分配多个 rank；这些 rank 分担 Q 头并各自保留该 KV 头。
            int offset = 0;
            for (uint64_t head = 0; head < kvHeads; ++head) {
                int count = static_cast<int>((world - offset + kvHeads - head - 1) / (kvHeads - head));
                if (rank >= offset && rank < offset + count) {
                    auto local = Partition(group, rank - offset, count);
                    q = {head * group + local.first, head * group + local.second};
                    kv = {head, head + 1};
                    break;
                }
                offset += count;
            }
        }
        plan.queryHeadBegin = q.first;
        plan.queryHeadCount = q.second - q.first;
        plan.kvHeadBegin = kv.first;
        plan.kvHeadCount = kv.second - kv.first;
        plan.intermediateBegin = mlp.first;
        plan.intermediateCount = mlp.second - mlp.first;
        plan.vocabBegin = vocabulary.first;
        plan.vocabCount = vocabulary.second - vocabulary.first;

        std::set<std::string> used;
        auto check = [&](const std::string &name, const std::vector<uint64_t> &shape) {
            auto found = sources.find(name);
            Require(found != sources.end(), "missing tensor: " + name);
            Require(found->second.shape == shape, "shape mismatch: " + name);
            used.insert(name);
        };
        auto add = [&](const std::string &name, const std::vector<uint64_t> &shape,
                       const std::vector<ShardSlice> &slices, bool zero = false) {
            ShardTensorPlan tensor;
            tensor.name = name;
            tensor.shape = shape;
            tensor.slices = slices;
            tensor.zeroFill = zero;
            uint64_t rows = 0, columns = shape.size() == 2 ? shape[1] : 1;
            for (const auto &slice : slices) {
                const auto &source = sources.at(slice.source);
                uint64_t sourceColumns = source.shape.size() == 2 ? source.shape[1] : 1;
                Require(slice.rowBegin < slice.rowEnd && slice.rowEnd <= source.shape[0] &&
                        slice.columnBegin < slice.columnEnd && slice.columnEnd <= sourceColumns &&
                        slice.columnEnd - slice.columnBegin == columns, "invalid slice: " + name);
                Require(tensor.dtype.empty() || tensor.dtype == source.dtype, "mixed fused dtype: " + name);
                tensor.dtype = source.dtype;
                rows += slice.rowEnd - slice.rowBegin;
            }
            Require(rows == shape[0], "merged shape mismatch: " + name);
            tensor.bytes = Multiply(Multiply(rows, columns), ItemSize(tensor.dtype));
            Require(plan.tensorBytes <= std::numeric_limits<uint64_t>::max() - tensor.bytes, "plan size overflow");
            plan.tensorBytes += tensor.bytes;
            plan.tensors.push_back(std::move(tensor));
        };
        auto replicate = [&](const std::string &name, const std::vector<uint64_t> &shape) {
            check(name, shape);
            add(name, shape, {{name, 0, shape[0], 0, shape.size() == 2 ? shape[1] : 1}});
        };

        replicate("model.embed_tokens.weight", {vocab, hidden});
        replicate("model.norm.weight", {hidden});
        std::string lmSource = sources.count("lm_head.weight") ? "lm_head.weight" : "model.embed_tokens.weight";
        Require(tied || lmSource == "lm_head.weight", "untied model is missing lm_head.weight");
        check(lmSource, {vocab, hidden});
        add("lm_head.weight", {plan.vocabCount, hidden}, {{lmSource, vocabulary.first, vocabulary.second, 0, hidden}});

        uint64_t qWidth = Multiply(qHeads, headDim), kvWidth = Multiply(kvHeads, headDim);
        for (uint64_t layer = 0; layer < layers; ++layer) {
            std::string prefix = "model.layers." + std::to_string(layer) + ".";
            std::string attn = prefix + "self_attn.", mlpPrefix = prefix + "mlp.";
            replicate(prefix + "input_layernorm.weight", {hidden});
            replicate(prefix + "post_attention_layernorm.weight", {hidden});
            if (!moe) {
                replicate(attn + "q_norm.weight", {headDim});
                replicate(attn + "k_norm.weight", {headDim});
            }

            for (const std::string suffix : {"weight", "bias"}) {
                bool bias = suffix == "bias";
                int present = 0;
                for (const std::string projection : {"q_proj.", "k_proj.", "v_proj."}) {
                    present += sources.count(attn + projection + suffix) ? 1 : 0;
                }
                if (bias && present == 0) {
                    continue;
                }
                Require(present == 3, "incomplete QKV " + suffix + ": " + attn);
                std::vector<ShardSlice> slices;
                for (const std::string projection : {"q_proj.", "k_proj.", "v_proj."}) {
                    bool isQ = projection == "q_proj.";
                    std::string name = attn + projection + suffix;
                    uint64_t width = isQ ? qWidth : kvWidth;
                    check(name, bias ? std::vector<uint64_t>{width} : std::vector<uint64_t>{width, hidden});
                    auto range = isQ ? q : kv;
                    slices.push_back({name, range.first * headDim, range.second * headDim, 0, bias ? 1 : hidden});
                }
                uint64_t localWidth = (plan.queryHeadCount + 2 * plan.kvHeadCount) * headDim;
                add(attn + "mergeqkv." + suffix,
                    bias ? std::vector<uint64_t>{localWidth} : std::vector<uint64_t>{localWidth, hidden}, slices);
            }

            check(attn + "o_proj.weight", {hidden, qWidth});
            add(attn + "o_proj.weight", {hidden, plan.queryHeadCount * headDim},
                {{attn + "o_proj.weight", 0, hidden, q.first * headDim, q.second * headDim}});
            if (moe) {
                replicate(mlpPrefix + "gate.weight", {plan.expertCount, hidden});
                replicate(mlpPrefix + "shared_expert_gate.weight", {1, hidden});
                // 所有专家先验证布局；只有归属本 rank 的专家才进入目标计划。
                for (uint64_t expert = 0; expert <= plan.expertCount; ++expert) {
                    bool shared = expert == plan.expertCount;
                    uint64_t width = shared ? plan.sharedIntermediateSize : intermediate;
                    std::string name = mlpPrefix + (shared ? "shared_expert." :
                        "experts." + std::to_string(expert) + ".");
                    check(name + "gate_proj.weight", {width, hidden});
                    check(name + "up_proj.weight", {width, hidden});
                    check(name + "down_proj.weight", {hidden, width});
                    if (shared || expert % world == uint64_t(rank)) {
                        add(name + "gateup_proj.weight", {2 * width, hidden}, {
                            {name + "gate_proj.weight", 0, width, 0, hidden},
                            {name + "up_proj.weight", 0, width, 0, hidden}});
                        replicate(name + "down_proj.weight", {hidden, width});
                    }
                }
                if (sources.count(attn + "o_proj.bias")) {
                    replicate(attn + "o_proj.bias", {hidden});
                }
                continue;
            }

            std::vector<ShardSlice> gateUp;
            for (const std::string projection : {"gate_proj.weight", "up_proj.weight"}) {
                check(mlpPrefix + projection, {intermediate, hidden});
                gateUp.push_back({mlpPrefix + projection, mlp.first, mlp.second, 0, hidden});
            }
            add(mlpPrefix + "gateup_proj.weight", {2 * plan.intermediateCount, hidden}, gateUp);
            check(mlpPrefix + "down_proj.weight", {hidden, intermediate});
            add(mlpPrefix + "down_proj.weight", {hidden, plan.intermediateCount},
                {{mlpPrefix + "down_proj.weight", 0, hidden, mlp.first, mlp.second}});

            for (const std::string name : {attn + "o_proj.bias", mlpPrefix + "down_proj.bias"}) {
                if (sources.count(name)) {
                    check(name, {hidden});
                    add(name, {hidden}, {{name, 0, hidden, 0, 1}}, rank != 0);
                }
            }
        }
        for (const auto &source : sources) {
            Require(used.count(source.first) != 0, "unexpected tensors: " + source.first +
                    " (fused/quantized/MLP-bias checkpoints are not supported)");
        }
    }
};

Qwen3ShardLoader::Qwen3ShardLoader(const std::string &directory, int rank, int worldSize, ShardModel model)
    : impl(new Impl) {
    Require(worldSize > 0 && rank >= 0 && rank < worldSize, "invalid rank/world size");
    // 保存绝对路径，调用方后续切换工作目录不能改变分片的来源文件。
    const fs::path root = fs::absolute(directory);
    Json config = ReadJson(root / "config.json");
    impl->IndexDirectory(root);
    impl->Build(config, rank, worldSize, model);
}

Qwen3ShardLoader::~Qwen3ShardLoader() = default;

const Qwen3ShardPlan &Qwen3ShardLoader::Plan() const {
    return impl->plan;
}

LoadedShardTensor Qwen3ShardLoader::LoadTensor(const std::string &name, uint64_t maxBytes) const {
    auto found = std::find_if(impl->plan.tensors.begin(), impl->plan.tensors.end(),
                             [&](const ShardTensorPlan &tensor) { return tensor.name == name; });
    Require(found != impl->plan.tensors.end(), "unknown target tensor: " + name);
    const ShardTensorPlan &tensor = *found;
    Require(tensor.bytes <= maxBytes && tensor.bytes <= std::numeric_limits<size_t>::max(),
            "allocation limit exceeded: " + name);
    LoadedShardTensor result;
    result.data.resize(static_cast<size_t>(tensor.bytes), 0);
    if (tensor.zeroFill) {
        return result;
    }

    uint64_t destination = 0, itemSize = ItemSize(tensor.dtype);
    for (const auto &slice : tensor.slices) {
        const auto &source = impl->sources.at(slice.source);
        Require(fs::file_size(source.file) == source.fileSize, "source file size changed: " + source.file.string());
        std::ifstream stream(source.file, std::ios::binary);
        uint64_t columns = source.shape.size() == 2 ? source.shape[1] : 1;
        uint64_t rowBytes = (slice.columnEnd - slice.columnBegin) * itemSize;
        if (slice.columnBegin == 0 && slice.columnEnd == columns) {
            uint64_t bytes = (slice.rowEnd - slice.rowBegin) * rowBytes;
            ReadAt(stream, source.offset + slice.rowBegin * columns * itemSize,
                   result.data.data() + destination, bytes);
            destination += bytes;
            result.bytesRead += bytes;
        } else {
            for (uint64_t row = slice.rowBegin; row < slice.rowEnd; ++row) {
                ReadAt(stream, source.offset + (row * columns + slice.columnBegin) * itemSize,
                       result.data.data() + destination, rowBytes);
                destination += rowBytes;
                result.bytesRead += rowBytes;
            }
        }
    }
    Require(destination == tensor.bytes, "internal output size mismatch: " + name);
    return result;
}

} // namespace distributed
} // namespace fastllm
