#include "models/duo/model.h"
#include "ffn_peer.h"
#include "models/minimax_h3_image/fp8.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <stdexcept>
#include <tuple>

namespace fastllm::duo {
namespace fs = std::filesystem;
namespace {
constexpr int minPeerFfnRows = 512;

void require(bool ok, const std::string &why) {
    if (!ok)
        throw std::invalid_argument("Duo: " + why);
}
std::string replace(std::string s, const std::string &a, const std::string &b) {
    auto pos = s.find(a);
    if (pos != std::string::npos)
        s.replace(pos, a.size(), b);
    return s;
}
std::string logical(std::string name) {
    if (name.rfind("blocks.", 0) == 0)
        name = "transformer_" + name;
    name = replace(name, "token_refiner.blocks.", "token_refiner.refiner_blocks.");
    for (const auto &p :
         std::vector<std::pair<std::string, std::string>>{{".attn.out_proj", ".attn.to_out.0"},
                                                          {".mlp.fc1", ".ff.net.0.proj"},
                                                          {".mlp.fc2", ".ff.net.2"},
                                                          {"condition_proj", "context_embedder"},
                                                          {"video_patch_proj", "proj_in"},
                                                          {"audio_patch_proj", "audio_proj_in"},
                                                          {"final_layer.video_out", "proj_out"},
                                                          {"final_layer.audio_out", "audio_proj_out"}})
        name = replace(name, p.first, p.second);
    return name;
}
void finite(const std::vector<float> &values) {
    for (float x : values)
        require(std::isfinite(x), "nonfinite input features");
}

Tensor selectRows(Runtime &rt, const Tensor &input, const std::vector<RowRange> &ranges) {
    if (ranges.size() == 1)
        return input.rowSlice(ranges[0].first, ranges[0].count);
    std::vector<Tensor> slices;
    for (const auto &range : ranges)
        slices.push_back(input.rowSlice(range.first, range.count));
    return rt.concat(slices);
}

// Attention has finished reading the joined tensor on the inference stream.
// Keep its existing history in place and compact only the retained current rows.
Tensor commitRows(Runtime &rt, const Tensor &current, Tensor joined, const std::vector<RowRange> &ranges) {
    if (ranges.size() == 1 && ranges.front().count == current.rows)
        return joined;
    auto keep = selectRows(rt, current, ranges);
    int pastRows = joined.rows - current.rows;
    if (!pastRows)
        return keep;
    auto tail = joined.rowSlice(pastRows, keep.rows);
    rt.concatInto({keep}, tail);
    return joined.rowSlice(0, pastRows + keep.rows);
}

} // namespace
json11::Json readJson(const std::string &path) {
    std::ifstream file(path);
    require(bool(file), "cannot open " + path);
    std::string data((std::istreambuf_iterator<char>(file)), {}), error;
    require(data.size() < 64u * 1024 * 1024, "oversized JSON " + path);
    auto value = json11::Json::parse(data, error);
    require(error.empty() && value.is_object(), "invalid JSON " + path + ": " + error);
    return value;
}
Config::Config(const json11::Json &c) {
    auto integer = [&](const char *k, int &v) {
        require(c[k].is_number() && c[k].number_value() == c[k].int_value(),
                std::string("missing/integral config ") + k);
        v = c[k].int_value();
        require(v > 0 && v < 1000000, "config dimensions out of range");
    };
    integer("hidden_size", hidden);
    integer("num_attention_heads", heads);
    integer("attention_head_dim", headDim);
    integer("num_layers", layers);
    integer("num_refiner_layers", refiners);
    integer("ffn_dim", ffn);
    integer("text_dim", text);
    integer("freq_dim", freq);
    integer("time_embed_dim", time);
    integer("rope_freq_dim", rope);
    auto real = [&](const char *k, float &v) {
        require(c[k].is_number() && c[k].number_value() > 0, k);
        v = c[k].number_value();
    };
    real("norm_eps", eps);
    real("qk_norm_eps", qkEps);
    real("final_norm_eps", finalEps);
    real("rope_theta", theta);
    require(layers <= 128 && refiners <= 8 && hidden <= 16384 && heads * int64_t(headDim) <= 32768 &&
                freq % 2 == 0 && rope * 6 <= headDim && c["in_channels"].int_value() == 24 &&
                c["audio_in_channels"].int_value() == 32 && c["patch_size"] == json11::Json::array{1, 2, 2},
            "unsupported model geometry");
}
ModelSource::ModelSource(const std::string &model, const std::string &imported) {
    auto root = fs::absolute(model).lexically_normal().string();
    if (fs::is_regular_file(root + "/realization/complete.json"))
        root += "/realization";
    const auto manifest = readJson(root + "/complete.json");
    format = manifest["format"].string_value();
    stage = manifest["stage"].string_value();
    require(stage == "causal" || stage == "dmd", "unsupported stage");
    const auto &models = manifest["models"].object_items();
    require(models.size() == 1, "expected one released model family");
    entry = models.begin()->second;
    Config checked(entry["config"]);
    require(!entry["role_embedding"].bool_value(),
            "role embeddings are not supported by this release adapter");
    const auto &codec = manifest["identity"]["codec"];
    require(codec["mode"].string_value() == "duo_5_17" && codec["frames_per_block"].int_value() == 17 &&
                codec["fps"].int_value() == 24,
            "checkpoint codec is not Duo 5-to-17");
    steps = manifest["inference"]["sampling_steps"].int_value();
    shift = manifest["inference"]["flow_shift"].number_value();
    schedule(steps, shift);
    if (format == "duo-release-modelopt-v1") {
        require(entry["mode"].string_value() == "full" &&
                    entry["storage"]["format"].string_value() == "modelopt_fp8",
                "invalid ModelOpt release");
        require(imported.empty(), "FP8 release is already merged; do not apply the DCP adapter again");
        weights = (fs::path(root) / entry["storage"]["path"].string_value()).lexically_normal().string();
    } else {
        require(format == "duo-release-dcp-v1" && entry["mode"].string_value() == "lora",
                "unsupported weight format");
        require(entry["base"]["format"].string_value() == "original_duo", "unsupported backbone format");
        weights = (fs::path(root) / entry["base"]["path"].string_value()).lexically_normal().string();
        require(
            !imported.empty(),
            "DCP needs the offline adapter import; run tools/duo/import_checkpoint.py and pass --adapter");
        adapter = fs::absolute(imported).string();
        auto data = readJson(adapter + "/adapter.json");
        require(data["format"].string_value() == "fastllm-duo-adapter-v1" &&
                    data["source_manifest"] == manifest,
                "imported adapter identity differs from the selected checkpoint");
        loraScale = float(entry["lora"]["alpha"].number_value() / entry["lora"]["rank"].number_value());
        require(std::isfinite(loraScale) && loraScale > 0, "invalid LoRA scale");
    }
    require(fs::is_directory(weights), "missing model weights");
}
size_t Cache::bytes() const {
    size_t n = 0;
    for (const auto &x : layers)
        n += x.first.bytes() + x.second.bytes();
    return n;
}
void Cache::evict(Runtime &rt, int first) {
    std::vector<Group> kept;
    std::vector<int> rows;
    int cursor = 0;
    for (auto g : groups)
        if (g.kind != Kind::Clean || g.block >= first) {
            for (int i = 0; i < g.count; ++i)
                rows.push_back(g.first + i);
            g.first = cursor;
            cursor += g.count;
            kept.push_back(g);
        }
    if (cursor == groupRows(groups))
        return;
    require(cursor > 0, "cannot evict the complete Duo prefix");
    auto indices = rt.indices(rows);
    std::vector<std::pair<Tensor, Tensor>> values;
    for (const auto &x : layers)
        values.emplace_back(rt.gather(x.first, indices), rt.gather(x.second, indices));
    layers = std::move(values);
    groups = std::move(kept);
}
struct Model::Impl {
    ModelSource source;
    Config cfg;
    Runtime rt, full;
    nomas::Weights weights, fullWeights;
    std::unique_ptr<nomas::Weights> lora;
    std::map<std::string, std::unique_ptr<minimax_h3_image::Fp8Weights>> quant;
    std::map<std::string, std::string> quantKeys;
    std::set<std::string> loraKeys;
    std::vector<float> times;
    std::vector<Tensor> modulation;
    Tensor outputMod;
    bool lowMemory;
    std::unique_ptr<FfnPeer> ffnPeer;
    Impl(int device, const ModelSource &s, bool low)
        : source(s), cfg(s.entry["config"]), rt(device, DType::BF16, {true, false}),
          full(device, DType::FP32), weights(rt, s.weights), fullWeights(full, s.weights), lowMemory(low) {
        if (!s.adapter.empty()) {
            lora = std::make_unique<nomas::Weights>(full, s.adapter + "/adapter.safetensors");
            auto metadata = readJson(s.adapter + "/adapter.json");
            for (const auto &x : metadata["tensors"].object_items())
                loraKeys.insert(x.first);
        }
        if (s.format == "duo-release-modelopt-v1") {
            minimax_h3_image::Fp8Weights::validateDevice(device);
            auto index =
                readJson(s.weights + "/diffusion_pytorch_model.safetensors.index.json")["weight_map"];
            require(index.is_object(), "missing ModelOpt weight map");
            for (const auto &x : index.object_items())
                if (x.first.size() > 13 && x.first.substr(x.first.size() - 13) == ".weight_scale") {
                    std::string name = x.first.substr(0, x.first.size() - 13), file = x.second.string_value();
                    require(!file.empty() && fs::path(file).filename() == file &&
                                index[name + ".weight"].string_value() == file,
                            "FP8 scale must be colocated with its weight");
                    quantKeys[name] = file;
                }
            require(quantKeys.size() == size_t(cfg.layers) * 4,
                    "expected four ModelOpt projections per main block");
            minimax_h3_image::ProjectionOptions options;
            options.modelopt = true;
            for (const auto &x : quantKeys)
                if (!quant.count(x.second))
                    quant[x.second] =
                        std::make_unique<minimax_h3_image::Fp8Weights>(s.weights + "/" + x.second, options);
        }
    }
    Tensor addAdapter(const Tensor &x, const std::string &name, Tensor out, bool swap = false) {
        if (!lora)
            return out;
        std::string a = name + ".lora_A.default.weight", b = name + ".lora_B.default.weight";
        if (!loraKeys.count(a))
            return out;
        require(loraKeys.count(b), "incomplete adapter pair");
        auto delta = full.linear(full.linear(full.cast(x, DType::FP32), lora->get(a)), lora->get(b));
        if (source.loraScale != 1)
            delta = full.affine(delta, source.loraScale);
        if (swap)
            delta = swapHalves(full, delta);
        auto type = out.dtype;
        out = full.add(full.cast(out, DType::FP32), delta);
        return rt.cast(out, type);
    }
    Tensor projection(const Tensor &x, const std::string &name, bool fp32 = false) {
        Tensor out;
        auto q = quantKeys.find(name);
        if (q != quantKeys.end())
            out = quant.at(q->second)->linearPacked(x, {name});
        else
            out = (fp32 ? fullWeights : weights).linear(x, name);
        const bool swap = name.size() >= 8 && name.substr(name.size() - 8) == ".mlp.fc1";
        return addAdapter(x, logical(name), out, swap);
    }
    std::array<Tensor, 3> prepareQkv(const Tensor &x, const std::string &p,
                                     const std::pair<Tensor, Tensor> *rope) {
        auto packed = projection(x, p + ".qkv_proj");
        const auto &qw = weights.get(p + ".q_norm.weight"), &kw = weights.get(p + ".k_norm.weight");
        require(packed.cols == 3 * cfg.heads * cfg.headDim, "invalid Duo QKV projection width");
        if (!lora && packed.dtype == DType::BF16 && cfg.headDim == 128)
            return rt.unpackHeadQkvNormRotary128(packed, qw, kw, cfg.qkEps, rope ? &rope->first : nullptr,
                                                rope ? &rope->second : nullptr);
        Tensor q = splitQkv(rt, packed, cfg.heads, cfg.headDim, 0),
               k = splitQkv(rt, packed, cfg.heads, cfg.headDim, 1),
               v = splitQkv(rt, packed, cfg.heads, cfg.headDim, 2);
        const std::string target = logical(p);
        q = addAdapter(x, target + ".to_q", q);
        k = addAdapter(x, target + ".to_k", k);
        v = addAdapter(x, target + ".to_v", v);
        if (rope)
            std::tie(q, k) = rt.qkNormRotary(q, k, qw, kw, cfg.qkEps, rope->first, rope->second, cfg.headDim);
        else {
            q = rt.rms(q, qw, cfg.qkEps, cfg.headDim);
            k = rt.rms(k, kw, cfg.qkEps, cfg.headDim);
        }
        return {q, k, v};
    }
    Tensor attend(const Tensor &x, const std::string &p, const std::vector<AttentionDomain> &domains,
                  const std::pair<Tensor, Tensor> *rope, const std::pair<Tensor, Tensor> *past,
                  const std::vector<RowRange> &retain, std::pair<Tensor, Tensor> *committed) {
        auto [q, k, v] = prepareQkv(x, p, rope);
        auto allK = past ? rt.concat({past->first, k}) : k;
        auto allV = past ? rt.concat({past->second, v}) : v;
        std::vector<Tensor> attended;
        for (const auto &domain : domains) {
            const auto &range = domain.queries;
            attended.push_back(rt.attention(q.rowSlice(range.first, range.count),
                                            selectRows(rt, allK, domain.keys), selectRows(rt, allV, domain.keys),
                                            cfg.heads));
        }
        if (committed)
            *committed = {commitRows(rt, k, std::move(allK), retain), commitRows(rt, v, std::move(allV), retain)};
        auto output = attended.size() == 1 ? attended.front() : rt.concat(attended);
        return projection(output, p + ".out_proj");
    }
    Tensor feedForward(const Tensor &x, const std::string &p) {
        if (ffnPeer && x.rows >= minPeerFfnRows && p.compare(0, 7, "blocks.") == 0) {
            const auto up = p + ".fc1", down = p + ".fc2";
            return ffnPeer->run(rt, x, up, down, *quant.at(quantKeys.at(up)), *quant.at(quantKeys.at(down)));
        }
        return projection(rt.siluMultiply(projection(x, p + ".fc1"), true), p + ".fc2");
    }
    void release(const std::string &p) {
        if (!lowMemory)
            return;
        weights.erasePrefix(p);
        if (lora)
            lora->erasePrefix(logical(p));
        // Direct packed keys include their length; clear per shard only in the
        // explicit streaming mode, never evict another session's state.
        for (auto &x : quant)
            x.second->clear();
    }
    Tensor refine(const Tensor &input, const Layout &layout) {
        auto x = projection(input, "condition_proj");
        auto domains = attentionDomains(layout.refinerGroups, layout.refinerGroups, true);
        for (int i = 0; i < cfg.refiners; ++i) {
            auto p = "token_refiner.blocks." + std::to_string(i);
            auto n = rt.rms(x, weights.get(p + ".norm1.weight"), cfg.eps);
            x = rt.add(x, attend(n, p + ".attn", domains, nullptr, nullptr, {}, nullptr));
            n = rt.rms(x, weights.get(p + ".norm2.weight"), cfg.eps);
            x = rt.add(x, feedForward(n, p + ".mlp"));
            release(p);
        }
        return rt.rms(x, weights.get("token_refiner.final_norm.weight"), cfg.finalEps);
    }
};
Model::Model(int device, const ModelSource &s, bool low) : impl_(new Impl(device, s, low)) {}
Model::~Model() = default;
const Config &Model::config() const {
    return impl_->cfg;
}
Runtime &Model::runtime() {
    return impl_->rt;
}
void Model::prepare() {
    auto &m = *impl_;
    if (!m.modulation.empty())
        return;
    m.times = schedule(m.source.steps, m.source.shift);
    std::vector<Tensor> inputs, plans;
    for (float time : m.times) {
        auto x = m.full.timestepEmbedding({1.f, time}, m.cfg.freq);
        x = m.full.activation(m.fullWeights.linear(x, "time_embedder.proj_in"));
        x = m.fullWeights.linear(x, "time_embedder.proj_out");
        inputs.push_back(m.rt.cast(m.full.activation(x), DType::BF16));
    }
    for (int i = 0; i < m.cfg.layers; ++i) {
        std::string p = "blocks." + std::to_string(i) + ".adaln_proj.linear";
        std::vector<Tensor> rows;
        for (const auto &x : inputs) {
            auto v = m.weights.linear(x, p, true);
            v.rows *= 3;
            v.cols /= 3;
            rows.push_back(v);
        }
        plans.push_back(m.rt.concat(rows));
        m.weights.erasePrefix(p);
    }
    std::vector<Tensor> out;
    for (const auto &x : inputs)
        out.push_back(m.weights.linear(x, "final_layer.adaln_proj.linear", true));
    auto output = m.rt.concat(out);
    m.weights.erasePrefix("final_layer.adaln_proj");
    m.outputMod = std::move(output);
    m.modulation = std::move(plans);
}
std::pair<Tensor, Tensor> Model::forward(const Layout &s, const Tensor &video, const Tensor &audio,
                                         const Tensor &text, float time, const Cache &past, Cache *committed,
                                         std::function<void()> checkpoint) {
    return forwardPrepared(s, video, audio, prepareText(s, text), time, past, committed, std::move(checkpoint));
}
Tensor Model::prepareText(const Layout &s, const Tensor &text) {
    auto &m = *impl_;
    require(text.cols == m.cfg.text && text.rows == groupRows(s.refinerGroups), "forward text shape differs");
    if (s.text.empty())
        return {};
    return m.rt.gather(m.refine(text, s), m.rt.indices(s.textSelection));
}
std::pair<Tensor, Tensor> Model::forwardPrepared(const Layout &s, const Tensor &video, const Tensor &audio,
                                                 const Tensor &condition, float time, const Cache &past,
                                                 Cache *committed, std::function<void()> checkpoint) {
    auto &m = *impl_;
    prepare();
    auto &rt = m.rt;
    require(video.rows == int(s.video.size()) && video.cols == 96 && audio.rows == int(s.audio.size()) &&
                audio.cols == 32 && condition.rows == int(s.text.size()) &&
                (s.text.empty() || condition.cols == m.cfg.hidden),
            "forward input shape differs");
    require(past.layers.empty() || past.layers.size() == size_t(m.cfg.layers), "KV layer count differs");
    auto found = std::find(m.times.begin(), m.times.end(), time);
    require(found != m.times.end(), "unprepared sampling timestep");
    int plan = int(found - m.times.begin());
    auto keys = past.groups;
    int pastRows = groupRows(keys);
    for (auto g : s.groups) {
        g.first += pastRows;
        keys.push_back(g);
    }
    auto domains = attentionDomains(s.groups, keys);
    auto vi = s.video.empty() ? Tensor{} : rt.indices(s.video);
    auto ai = s.audio.empty() ? Tensor{} : rt.indices(s.audio);
    auto ti = s.text.empty() ? Tensor{} : rt.indices(s.text);
    Tensor x(s.rows(), m.cfg.hidden);
    if (!s.text.empty())
        rt.scatter(x, condition, ti);
    if (video.rows)
        rt.scatter(x, rt.cast(m.projection(video, "video_patch_proj", true), DType::BF16), vi);
    if (audio.rows)
        rt.scatter(x, rt.cast(m.projection(audio, "audio_patch_proj", true), DType::BF16), ai);
    auto rope = rt.ropeTable(s.positions, m.cfg.rope, m.cfg.theta);
    std::vector<int> modulation(s.rows()), output(s.rows());
    std::vector<RowRange> retain;
    Cache next;
    for (int i = 0; i < s.rows(); ++i) {
        output[i] = plan * 2 + s.time[i];
        modulation[i] = output[i] * 3 + s.tags[i];
    }
    if (committed) {
        next.groups = past.groups;
        int offset = pastRows;
        for (auto g : s.groups)
            if (g.kind == Kind::Reference || g.kind == Kind::Shared || g.kind == Kind::Clean) {
                if (!retain.empty() && retain.back().first + retain.back().count == g.first)
                    retain.back().count += g.count;
                else
                    retain.push_back({g.first, g.count});
                g.first = offset;
                offset += g.count;
                next.groups.push_back(g);
            }
        require(!retain.empty(), "empty clean commitment");
        next.layers.resize(m.cfg.layers);
    }
    Tensor idx = rt.indices(modulation);
    for (int i = 0; i < m.cfg.layers; ++i) {
        if (checkpoint)
            checkpoint();
        const auto p = "blocks." + std::to_string(i);
        auto n =
            rt.rmsModulateRows(x, m.weights.get(p + ".norm1.weight"), m.cfg.eps, m.modulation[i], idx, 0, 1);
        auto att = m.attend(n, p + ".attn", domains, &rope, past.layers.empty() ? nullptr : &past.layers[i],
                            retain, committed ? &next.layers[i] : nullptr);
        x = rt.residualRows(x, att, m.modulation[i], idx, 2);
        n = rt.rmsModulateRows(x, m.weights.get(p + ".norm2.weight"), m.cfg.eps, m.modulation[i], idx, 3, 4);
        x = rt.residualRows(x, m.feedForward(n, p + ".mlp"), m.modulation[i], idx, 5);
        m.release(p + ".");
    }
    x = rt.rmsModulateRows(x, m.weights.get("final_layer.norm.weight"), m.cfg.finalEps, m.outputMod,
                           rt.indices(output), 0, 1);
    Tensor v, a;
    v.cols = 96;
    a.cols = 32;
    if (video.rows)
        v = m.projection(m.full.cast(rt.gather(x, vi), DType::FP32), "final_layer.video_out", true);
    if (audio.rows)
        a = m.projection(m.full.cast(rt.gather(x, ai), DType::FP32), "final_layer.audio_out", true);
    if (checkpoint)
        checkpoint();
    if (committed)
        *committed = std::move(next);
    return {v, a};
}
Session::Session(int device, const ModelSource &s, int h, int w, int history, bool low)
    : source_(s), model_(device, s, low), height_(h), width_(w), history_(history) {
    prefixLayout(h, w, 1);
    require(history >= 1 && history <= 128, "historyChunks must be in [1,128]");
}
bool Session::enableFfnPeer(int device) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto &m = *model_.impl_;
    const int mediaRows = 5 * height_ * width_ / 4 + 58;
    if (!m.lowMemory && !m.ffnPeer && mediaRows >= minPeerFfnRows)
        m.ffnPeer = FfnPeer::create(m.rt.device(), device, m.source, m.cfg, m.quantKeys);
    return bool(m.ffnPeer);
}
void Session::checkpoint() const {
    if (cancelled_.load())
        throw std::runtime_error("Duo generation cancelled; reset the session");
}
void Session::reset(const std::vector<float> &reference, const std::vector<float> &shared, int rows,
                    bool reuseSystemPrefix) {
    std::lock_guard<std::mutex> lock(mutex_);
    require(reference.size() == size_t(height_) * width_ / 4 * 96 &&
                shared.size() == size_t(rows) * model_.config().text && rows > 0,
            "invalid session prefix");
    finite(reference);
    finite(shared);
    auto &rt = model_.runtime();
    rt.activate();
    {
        std::lock_guard<std::mutex> lock(commitMutex_);
        cancelled_ = false;
    }
    cache_ = {};
    timeline_.reset();
    sharedRows_ = 0;
    // Prefix KV contains only the fixed portrait and system conditioning.
    keepPrefix_ = reuseSystemPrefix;
    if (!keepPrefix_) {
        prefixCache_ = {};
        prefixReference_.clear();
        prefixShared_.clear();
    } else if (reference != prefixReference_ || shared != prefixShared_) {
        prefixCache_ = {};
        prefixReference_ = reference;
        prefixShared_ = shared;
    }
    try {
        reference_ = uploadFp32(rt, reference, height_ * width_ / 4, 96);
        shared_ = rt.upload(shared.data(), rows, model_.config().text);
        model_.prepare();
        sharedRows_ = rows;
        cache_ = prefixCache_;
    } catch (...) {
        timeline_.cancel();
        throw;
    }
}
ChunkResult Session::generate(int chunk, const std::vector<float> &features, int rows, uint64_t seed,
                              const std::vector<std::vector<float>> &noise,
                              const std::function<void(const ChunkResult &)> &mediaReady) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto ticket = timeline_.begin(chunk);
    try {
        checkpoint();
        require(sharedRows_ > 0 && rows > 0 && features.size() == size_t(rows) * model_.config().text,
                "invalid chunk Intent features");
        finite(features);
        auto &rt = model_.runtime();
        rt.activate();
        auto start = std::chrono::steady_clock::now();
        auto current = chunkLayout(height_, width_, sharedRows_, rows, chunk);
        const int nv = 5 * height_ * width_ / 4, na = 58;
        auto body = rt.upload(features.data(), rows, model_.config().text), text = rt.concat({shared_, body});
        if (cache_.layers.empty()) {
            Tensor empty;
            empty.cols = 32;
            Cache initial;
            model_.forward(prefixLayout(height_, width_, sharedRows_), reference_, empty, shared_, 1.f, {},
                           &initial, [&] { checkpoint(); });
            cache_ = std::move(initial);
            if (keepPrefix_)
                prefixCache_ = cache_;
        }
        cache_.evict(rt, std::max(0, chunk - history_ + 1));
        std::mt19937_64 generator(seed);
        std::normal_distribution<float> normal;
        size_t noiseIndex = 0;
        auto random = [&](int n, int c) {
            std::vector<float> values;
            if (!noise.empty()) {
                require(noiseIndex < noise.size() && noise[noiseIndex].size() == size_t(n) * c,
                        "noise fixture shape/count differs");
                values = noise[noiseIndex];
            } else {
                values.resize(size_t(n) * c);
                for (float &v : values)
                    v = normal(generator);
            }
            ++noiseIndex;
            finite(values);
            // uploadFp32 preserves sampler precision independently of model compute dtype.
            return uploadFp32(rt, values, n, c);
        };
        auto v = random(nv, 96), a = random(na, 32);
        // All sampling steps and the clean KV pass use the same text rows and visibility.
        auto condition = model_.prepareText(current, text);
        auto grid = schedule(source_.steps, source_.shift);
        for (int i = 0; i < source_.steps; ++i) {
            checkpoint();
            auto pred = model_.forwardPrepared(current, v, a, condition, grid[i], cache_, nullptr, [&] { checkpoint(); });
            float amount = source_.stage == "causal" ? grid[i + 1] - grid[i] : 1.f - grid[i];
            v = flowUpdate(rt, v, pred.first, amount);
            a = flowUpdate(rt, a, pred.second, amount);
            if (source_.stage == "dmd" && i + 1 < source_.steps) {
                v = flowNoise(rt, v, random(nv, 96), grid[i + 1]);
                a = flowNoise(rt, a, random(na, 32), grid[i + 1]);
            }
        }
        require(noise.empty() || noiseIndex == noise.size(), "unused noise fixture tensors");
        ChunkResult out;
        out.chunk = chunk;
        out.firstFrame = frameStart(chunk);
        out.firstSample = sampleStart(chunk);
        out.endSample = sampleStart(int64_t(chunk) + 1);
        auto downloadMedia = [&] {
            out.video = rt.download(v);
            out.audio = rt.download(a);
            finite(out.video);
            finite(out.audio);
        };
        if (mediaReady) {
            downloadMedia();
            checkpoint();
            mediaReady(out);
        }
        Cache committed;
        model_.forwardPrepared(chunkLayout(height_, width_, sharedRows_, rows, chunk, true), v, a, condition,
                               1.f, cache_, &committed, [&] { checkpoint(); });
        if (mediaReady) {
            // Downloads above have already synchronized the denoising passes. Finish
            // queued KV writes before handing the committed block back to the caller.
            rt.synchronize();
        } else {
            downloadMedia();
        }
        {
            std::lock_guard<std::mutex> lock(commitMutex_);
            checkpoint();
            timeline_.committed(ticket);
            cache_ = std::move(committed);
        }
        out.cacheBytes = cache_.bytes();
        out.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return out;
    } catch (...) {
        timeline_.fail(ticket);
        cache_ = {};
        throw;
    }
}
} // namespace fastllm::duo
