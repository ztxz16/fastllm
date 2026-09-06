#include "contextconfig.h"
#include "executor.h"
#include "json11.hpp"
#include "models/basellm.h"
#include <algorithm>
#include <cmath>
#include <climits>
#include <set>
#include <stdexcept>

namespace fastllm {
    namespace {
        using Json = json11::Json;
        using Object = Json::object;
        void Require(bool ok, const std::string &message) {
            if (!ok)
                throw std::invalid_argument("Context/RoPE: " + message);
        }
        Json Parse(const std::string &s, const std::string &field) {
            std::string error;
            auto value = Json::parse(s, error);
            Require(error.empty(), "invalid " + field + ": " + error);
            return value;
        }
        Json Read(const std::map<std::string, std::string> &dict, const std::string &key) {
            auto it = dict.find(key);
            if (it == dict.end())
                return Json();
            std::string error;
            auto value = Json::parse(it->second, error);
            return error.empty() ? value : Json(it->second);
        }
        double Number(const Object &o, const std::string &key, double fallback, double minimum = 0) {
            auto it = o.find(key);
            if (it == o.end())
                return fallback;
            Require(it->second.is_number() && std::isfinite(it->second.number_value()) &&
                        it->second.number_value() > minimum && it->second.number_value() <= INT_MAX,
                    key + " must be a finite number greater than " + std::to_string(minimum));
            return it->second.number_value();
        }
        int Integer(const Object &o, const std::string &key, int fallback) {
            double n = Number(o, key, fallback);
            Require(n == std::floor(n), key + " must be an integer");
            return (int)n;
        }
        Object Normalize(Object o, const std::string &field) {
            auto old = o.find("type"), modern = o.find("rope_type");
            if (old != o.end()) {
                Require(modern == o.end() || modern->second == old->second, field + ".type conflicts with rope_type");
                o["rope_type"] = old->second;
                o.erase("type");
            }
            return o;
        }
        Object RopeObject(const std::map<std::string, std::string> &dict, const std::string &prefix) {
            Object o;
            Json root = Read(dict, prefix);
            if (root.is_object())
                o = root.object_items();
            // Configuration dictionaries can contain JSON objects or flattened fields.
            const std::string fieldPrefix = prefix + ".";
            for (auto &entry : dict) {
                if (entry.first.compare(0, fieldPrefix.size(), fieldPrefix) == 0) {
                    auto key = entry.first.substr(fieldPrefix.size());
                    o[key] = Read(dict, entry.first);
                }
            }
            return Normalize(o, prefix);
        }
    } // namespace

    ContextPlan ResolveContextOptions(const std::map<std::string, std::string> &dict, const ModelContextSpec &spec,
                                      const ContextOptions &options) {
        Require(options.maxLength == -1 || options.maxLength > 0, "max_context_length must be positive");
        ContextPlan plan;
        const std::string p =
            Read(dict, spec.configPrefix + "max_position_embeddings").is_null() ? "" : spec.configPrefix;
        Object geometry;
        for (auto key : {"max_position_embeddings", "head_dim", "hidden_size", "num_attention_heads", "rope_theta",
                         "partial_rotary_factor"}) {
            Json v = Read(dict, p + key);
            if (!v.is_null())
                geometry[key] = v;
        }
        plan.declaredLength = Integer(geometry, "max_position_embeddings", 0);
        plan.requestedLength = options.maxLength;
        plan.effectiveLength = options.maxLength > 0 ? options.maxLength : plan.declaredLength;
        if (!spec.supported && options.ropeScaling.empty() && options.maxLength == -1)
            return plan;
        Object modern = RopeObject(dict, p + "rope_parameters");
        Object legacy = RopeObject(dict, p + "rope_scaling");
        bool configuredYarn = (modern.count("rope_type") && modern.at("rope_type").string_value() == "yarn") ||
                              (legacy.count("rope_type") && legacy.at("rope_type").string_value() == "yarn");
        plan.configured = !options.ropeScaling.empty() || options.maxLength > 0 || (spec.supported && configuredYarn);
        if (!plan.configured)
            return plan;
        Require(plan.declaredLength > 0, "model has no declared max_position_embeddings");
        if (!spec.supported) {
            Require(options.ropeScaling.empty() && plan.effectiveLength <= plan.declaredLength,
                    "context extension is not implemented for this model layout");
            return plan;
        }

        Object override;
        if (!options.ropeScaling.empty()) {
            if (options.ropeScaling == "yarn")
                override["rope_type"] = "yarn";
            else {
                auto parsed = Parse(options.ropeScaling, "rope_scaling");
                Require(parsed.is_object(), "rope_scaling must be yarn or a JSON object");
                override = Normalize(parsed.object_items(), "rope_scaling");
                if (override.count("full_attention")) {
                    Require(override.size() == 1 && override.at("full_attention").is_object(),
                            "this layout only supports the full_attention RoPE group");
                    override = Normalize(override.at("full_attention").object_items(), "full_attention");
                }
            }
        }
        Object rope = legacy;
        for (auto &it : modern) {
            Require(!rope.count(it.first) || rope.at(it.first) == it.second || override.count(it.first),
                    p + "rope_parameters." + it.first + " conflicts with rope_scaling");
            rope[it.first] = it.second;
        }
        if (override.count("rope_type") && rope.count("rope_type") &&
            override.at("rope_type") != rope.at("rope_type")) {
            // Only layout fields survive an algorithm change. Do not inherit old algorithm-specific factors.
            for (auto it = rope.begin(); it != rope.end();) {
                if (it->first != "rope_theta" && it->first != "partial_rotary_factor" && it->first != "mrope_section" &&
                    it->first != "mrope_interleaved")
                    it = rope.erase(it);
                else
                    ++it;
            }
        }
        for (auto &it : override)
            rope[it.first] = it.second;
        auto &r = plan.rope;
        r.type = rope.count("rope_type") ? rope.at("rope_type").string_value() : "default";
        // Keep old NTK inference unchanged when no RoPE override or extension is requested.
        if (r.type == "dynamic" && options.ropeScaling.empty() && plan.effectiveLength <= plan.declaredLength)
            return plan;
        Require(r.type == "default" || r.type == "linear" || r.type == "yarn", "unsupported rope_type: " + r.type);
        const std::set<std::string> fields = {
            "rope_type", "rope_theta", "partial_rotary_factor", "factor",        "original_max_position_embeddings",
            "beta_fast", "beta_slow",  "attention_factor",      "mrope_section", "mrope_interleaved"};
        for (auto &it : rope)
            Require(fields.count(it.first), "unsupported RoPE field: " + it.first);
        Require(r.IsYarn() || (!rope.count("beta_fast") && !rope.count("beta_slow") && !rope.count("attention_factor")),
                "beta_fast, beta_slow and attention_factor require YaRN");
        Require(r.type != "default" || Number(rope, "factor", 1) == 1, "default RoPE cannot apply a scaling factor");
        r.theta = Number(rope, "rope_theta", Number(geometry, "rope_theta", 10000, 1), 1);
        int head = Integer(geometry, "head_dim", 0);
        if (!head) {
            int heads = Integer(geometry, "num_attention_heads", 0), hidden = Integer(geometry, "hidden_size", 0);
            Require(heads > 0 && hidden > 0 && hidden % heads == 0, "invalid attention head dimensions");
            head = hidden / heads;
        }
        double partial = Number(rope, "partial_rotary_factor",
                                Number(geometry, "partial_rotary_factor", spec.interleavedMRope ? 0.25 : 1.0));
        Require(partial <= 1.0, "partial_rotary_factor must be <= 1");
        Require(head * partial == std::floor(head * partial),
                "partial_rotary_factor must produce an integer rotary dimension");
        r.rotaryDim = (int)(head * partial);
        Require(r.rotaryDim > 0 && r.rotaryDim % 2 == 0 && r.rotaryDim <= head, "invalid rotary dimension");
        if (spec.interleavedMRope) {
            auto sections = rope.find("mrope_section");
            Require(sections != rope.end() && sections->second.is_array() && sections->second.array_items().size() == 3,
                    "interleaved M-RoPE requires three mrope_section values");
            int sum = 0;
            for (auto &v : sections->second.array_items()) {
                Require(v.is_number() && v.number_value() >= 0 && v.number_value() <= r.rotaryDim &&
                            v.number_value() == v.int_value(),
                        "invalid mrope_section");
                r.mropeSections.push_back(v.int_value());
                sum += v.int_value();
            }
            Require(sum == r.rotaryDim / 2, "mrope_section sum must equal rotary_dim / 2");
            Require(rope.count("mrope_interleaved") && rope.at("mrope_interleaved").is_bool() &&
                        rope.at("mrope_interleaved").bool_value(),
                    "only interleaved M-RoPE is supported");
        } else
            Require(!rope.count("mrope_section") && !rope.count("mrope_interleaved"),
                    "M-RoPE requires a multimodal model adapter");

        r.originalMaxPosition = Integer(rope, "original_max_position_embeddings", 0);
        if (r.IsYarn()) {
            // Verified Qwen3.5 recipe; ordinary Qwen3's declared 40960 is NOT its YaRN original length.
            if (!r.originalMaxPosition && spec.interleavedMRope && plan.declaredLength == 262144 &&
                r.theta == 10000000.0f && r.rotaryDim == 64 && r.mropeSections == std::vector<int>({11, 11, 10}))
                r.originalMaxPosition = 262144;
            Require(r.originalMaxPosition > 0,
                    "YaRN requires original_max_position_embeddings; it cannot be inferred from the serving window");
            r.factor =
                Number(rope, "factor", std::max(1.0, std::ceil((double)plan.effectiveLength / r.originalMaxPosition)));
            Require(r.factor >= 1, "YaRN factor must be >= 1");
            r.betaFast = Number(rope, "beta_fast", 32);
            r.betaSlow = Number(rope, "beta_slow", 1);
            Require(r.betaFast >= r.betaSlow, "beta_fast must be >= beta_slow");
            r.attentionFactor = Number(rope, "attention_factor", 1.0 + 0.1 * std::log(r.factor));
            auto correction = [&](float rotations) {
                return (r.rotaryDim *
                        std::log((float)r.originalMaxPosition / (rotations * 2.0f * 3.14159265358979323846f))) /
                       (2.0f * std::log(r.theta));
            };
            r.correctionLow = std::max(0.0f, std::floor(correction(r.betaFast)));
            r.correctionHigh = std::min((float)r.rotaryDim - 1, std::ceil(correction(r.betaSlow)));
            if (r.correctionLow == r.correctionHigh)
                r.correctionHigh += 0.001f;
            Require(r.correctionHigh > r.correctionLow, "invalid YaRN correction range");
        } else if (r.type == "linear") {
            r.factor = Number(rope, "factor", 1);
            Require(r.factor >= 1, "linear factor must be >= 1");
            if (plan.effectiveLength > plan.declaredLength)
                Require(r.originalMaxPosition > 0, "linear extension requires original_max_position_embeddings");
        }
        double supported = plan.declaredLength;
        if (r.type != "default" && r.originalMaxPosition > 0)
            supported = (double)r.originalMaxPosition * r.factor;
        Require(plan.effectiveLength <= supported,
                "requested " + std::to_string(plan.effectiveLength) + " tokens exceeds the configured RoPE window " +
                    std::to_string((int64_t)supported) + "; provide an explicit rope_scaling configuration");
        Require(plan.effectiveLength <= (1 << 24), "current FLOAT32 position ids support at most 16777216 tokens");
        return plan;
    }

    FloatDict RopeConfig::Params() const {
        return {{"ropeTheta", theta},
                {"factor", factor},
                {"attentionFactor", attentionFactor},
                {"correctionLow", correctionLow},
                {"correctionHigh", correctionHigh}};
    }
    IntDict RopeConfig::PositionParams(bool multimodal) const {
        IntDict p = {{"rotaryDim", rotaryDim}};
        if (multimodal && !mropeSections.empty()) {
            p["mrope"] = 1;
            p["sectionH"] = mropeSections[1];
            p["sectionW"] = mropeSections[2];
        }
        return p;
    }
    void RopeConfig::AddFusedParams(FloatDict &floats, IntDict &ints) const {
        if (!IsYarn())
            return;
        ints["useYarn"] = 1;
        floats["yarnFactor"] = factor;
        floats["yarnAttentionFactor"] = attentionFactor;
        floats["yarnCorrectionLow"] = correctionLow;
        floats["yarnCorrectionHigh"] = correctionHigh;
    }
    void ApplyYarnRope(Data &input, const Data &positions, const RopeConfig &rope) {
        static_cast<Executor *>(GetExecutor())
            ->Run("YarnRopeEncoding", {{"input", &input}, {"positionIds", (Data *)&positions}}, rope.Params(),
                  rope.PositionParams(positions.dims.size() == 2 && positions.dims[0] == 3));
    }
    std::string ContextPlan::ToJson() const {
        Json::object r = {
            {"rope_type", rope.type},       {"rope_theta", rope.theta},
            {"factor", rope.factor},        {"original_max_position_embeddings", rope.originalMaxPosition},
            {"rotary_dim", rope.rotaryDim}, {"attention_factor", rope.attentionFactor},
            {"beta_fast", rope.betaFast},   {"beta_slow", rope.betaSlow}};
        if (!rope.mropeSections.empty()) {
            r["mrope_section"] = Json(rope.mropeSections);
            r["mrope_interleaved"] = true;
        }
        return Json(Json::object{{"configured", configured},
                                 {"model_context_window", declaredLength},
                                 {"configured_context_window_limit", requestedLength},
                                 {"context_window", effectiveLength},
                                 {"rope_parameters", r}})
            .dump();
    }

    void basellm::ConfigureContext(const ContextOptions &options) {
        contextPlan = ResolveContextOptions(weight.dicts, GetContextSpec(), options);
    }
    void basellm::InitContextParams(RoPEType &type, float &theta, float &factor, int &rotaryDim) {
        if (!contextPlan.configured)
            return;
        max_positions = contextPlan.effectiveLength;
        const auto &r = contextPlan.rope;
        if (r.type == "dynamic")
            return;
        type = r.IsYarn() ? RoPEType::YARN : r.type == "linear" ? RoPEType::LINEAR_SCALE : RoPEType::BASE;
        theta = r.theta;
        factor = r.factor;
        rotaryDim = r.rotaryDim;
    }
    void basellm::ValidateContextCapacity() {
        if (!contextPlan.configured)
            return;
        int capacity = tokensLimit > 0 ? tokensLimit : GetMaxTokens();
        // All initially supported layouts include token-growing full attention.
        // Future bounded-only adapters must provide their own capacity semantics.
        if (capacity > 0 && contextPlan.requestedLength > capacity) {
            throw std::runtime_error(
                "Context capacity insufficient: requested " + std::to_string(contextPlan.requestedLength) +
                " tokens, calibrated KV capacity " + std::to_string(capacity) +
                ". Increase --tokens if it was set below the target, reduce the requested context or use more memory.");
        }
        contextPlan.effectiveLength = capacity > 0 ? std::min(max_positions, capacity) : max_positions;
    }
} // namespace fastllm
