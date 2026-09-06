#include "contextconfig.h"
#include "executor.h"
#include "models/basellm.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#endif

using namespace fastllm;
namespace {
    void Check(bool ok, const std::string &message) {
        if (!ok)
            throw std::runtime_error(message);
    }
    template <class F> void Reject(F f, const std::string &message) {
        try {
            f();
        } catch (const std::invalid_argument &) {
            return;
        }
        throw std::runtime_error("Expected rejection: " + message);
    }
    std::map<std::string, std::string> TextConfig() {
        return {{"max_position_embeddings", "40960"}, {"head_dim", "128"}, {"rope_theta", "1000000"}};
    }
    std::map<std::string, std::string> MMConfig() {
        return {{"text_config.max_position_embeddings", "262144"},
                {"text_config.head_dim", "256"},
                {"text_config.rope_parameters.rope_type", "default"},
                {"text_config.rope_parameters.rope_theta", "10000000"},
                {"text_config.rope_parameters.partial_rotary_factor", "0.25"},
                {"text_config.rope_parameters.mrope_interleaved", "true"},
                {"text_config.rope_parameters.mrope_section", "[11,11,10]"}};
    }
    void ConfigTests() {
        ModelContextSpec text{true, false, ""}, mm{true, true, "text_config."};
        const std::string yarn = R"({"rope_type":"yarn","factor":4,"original_max_position_embeddings":32768})";
        auto d = TextConfig();
        Check(!ResolveContextOptions(d, text, {}).configured, "default changed");
        auto r = ResolveContextOptions(d, text, {131072, yarn});
        Check(r.declaredLength == 40960 && r.rope.originalMaxPosition == 32768 && r.effectiveLength == 131072 &&
                  r.rope.rotaryDim == 128,
              "Qwen3 original/reference length");
        Reject([&] { ResolveContextOptions(d, text, {131072, "yarn"}); }, "unknown original");
        Reject([&] { ResolveContextOptions(d, text, {131072, ""}); }, "unscaled extension");
        Reject([&] { ResolveContextOptions(d, text, {131073, yarn}); }, "factor insufficient");
        Check(ResolveContextOptions(d, text, {8192, ""}).effectiveLength == 8192, "shorter context");
        d["rope_scaling"] = yarn;
        Check(ResolveContextOptions(d, text, {}).rope.IsYarn(), "config-only YaRN");
        d["max_position_embeddings"] = "131072";
        Check(ResolveContextOptions(d, text, {}).effectiveLength == 131072, "already scaled model");
        Reject([&] { ResolveContextOptions(d, text, {524288, ""}); }, "double scaling");
        d["rope_parameters.factor"] = "8";
        Reject([&] { ResolveContextOptions(d, text, {131072, ""}); }, "conflicting aliases");
        Check(ResolveContextOptions(d, text, {131072, R"({"factor":4})"}).rope.factor == 4,
              "explicit conflict resolution");
        auto m = MMConfig();
        auto mr = ResolveContextOptions(m, mm, {1000000, "yarn"});
        Check(mr.rope.factor == 4 && mr.rope.originalMaxPosition == 262144 && mr.rope.rotaryDim == 64 &&
                  mr.rope.mropeSections == std::vector<int>({11, 11, 10}),
              "Qwen3.5 recipe");
        for (auto override : {R"({"factor":0})", R"({"factor":-1})", R"({"factor":1e100})", R"({"factor":"4"})",
                              R"({"rope_type":"default","factor":4})", R"({"rope_theta":1})", R"({"beta_fast":32})",
                              R"({"rope_type":"unknown"})", R"({"unknown":4})",
                              R"({"rope_type":"yarn","type":"linear"})", R"({"original_max_position_embeddings":2.5})"})
            Reject([&] { ResolveContextOptions(m, mm, {1000000, override}); }, override);
        auto grouped = ResolveContextOptions(m, mm, {1000000, R"({"full_attention":{"rope_type":"yarn"}})"});
        Check(grouped.rope.factor == 4, "named group");
        m["text_config.rope_parameters.rope_type"] = "yarn";
        for (auto override : {R"({"factor":0.5})", R"({"factor":"4"})", R"({"beta_fast":0})",
                              R"({"beta_fast":1,"beta_slow":32})", R"({"attention_factor":-1})"}) {
            Reject([&] { ResolveContextOptions(m, mm, {1000000, override}); }, override);
        }
        m["text_config.rope_parameters.mrope_section"] = "[11,11,11]";
        Reject([&] { ResolveContextOptions(m, mm, {1000000, "yarn"}); }, "invalid sections");
        Reject([&] { ResolveContextOptions(TextConfig(), {}, {131072, yarn}); }, "unadapted model");
        Reject([&] { ResolveContextOptions(TextConfig(), text, {0, ""}); }, "zero length");
        std::cout << "Context configuration cases passed\n";
    }
    void Device(const std::string &name) { static_cast<Executor *>(GetExecutor())->SetFirstDevice(name); }
    std::vector<float> Floats(Data data) {
        data.ToDevice(DataDevice::CPU);
        ToDataTypeForceCPU(data, DataType::FLOAT32);
        return std::vector<float>((float *)data.cpuData, (float *)data.cpuData + data.Count(0));
    }
    void Near(const std::vector<float> &a, const std::vector<float> &b, float tol, const std::string &label) {
        Check(a.size() == b.size(), label + " shape");
        for (size_t i = 0; i < a.size(); ++i)
            Check(std::isfinite(b[i]) && std::abs(a[i] - b[i]) <= tol + tol * std::abs(a[i]),
                  label + " at " + std::to_string(i) + ": " + std::to_string(a[i]) + " != " + std::to_string(b[i]));
    }
    // Independent scalar reference, evaluated on the actually rounded input dtype.
    std::vector<float> Reference(const std::vector<float> &a, int tokens, int heads, int dim,
                                 const std::vector<float> &pos, const RopeConfig &r, bool mm) {
        auto out = a;
        auto correction = [&](double beta) {
            return r.rotaryDim * std::log(r.originalMaxPosition / (beta * 2 * 3.14159265358979323846)) /
                   (2 * std::log(r.theta));
        };
        double lo = std::max(0.0, std::floor(correction(r.betaFast)));
        double hi = std::min((double)r.rotaryDim - 1, std::ceil(correction(r.betaSlow)));
        if (lo == hi)
            hi += 0.001;
        for (int t = 0; t < tokens; ++t)
            for (int j = 0; j < r.rotaryDim / 2; ++j) {
                int row = mm ? (j % 3 == 1 && j < r.mropeSections[1] * 3   ? 1
                                : j % 3 == 2 && j < r.mropeSections[2] * 3 ? 2
                                                                           : 0)
                             : 0;
                float f = (float)std::pow((double)r.theta, 2.0 * j / r.rotaryDim);
                float ramp = (float)std::max(0.0, std::min(1.0, (j - lo) / (hi - lo)));
                float extrapolation = 1 - ramp;
                float freq = (1.0f / (r.factor * f)) * (1 - extrapolation) + (1.0f / f) * extrapolation;
                float angle = pos[row * tokens + t] * freq;
                float sn = std::sin(angle) * r.attentionFactor, cs = std::cos(angle) * r.attentionFactor;
                for (int h = 0; h < heads; ++h) {
                    size_t off = ((size_t)t * heads + h) * dim + j;
                    out[off] = a[off] * cs - a[off + r.rotaryDim / 2] * sn;
                    out[off + r.rotaryDim / 2] = a[off] * sn + a[off + r.rotaryDim / 2] * cs;
                }
            }
        return out;
    }
    void RotaryTests(bool cuda) {
        for (bool mm : {false, true}) {
            auto r = ResolveContextOptions(
                         mm ? MMConfig() : TextConfig(), {true, mm, mm ? "text_config." : ""},
                         {mm ? 1000000 : 131072,
                          mm ? "yarn" : R"({"rope_type":"yarn","factor":4,"original_max_position_embeddings":32768})"})
                         .rope;
            std::vector<float> pos = {0, 32767, 32768, 40960, 262143, 262144, 999999, 1000000};
            const int tokens = pos.size(), heads = 3, dim = mm ? 256 : 128;
            if (mm) {
                auto original = pos;
                for (int row = 1; row < 3; ++row)
                    for (auto v : original)
                        pos.push_back(v + row * 7);
            }
            Data positions(DataType::FLOAT32, {mm ? 3 : 1, tokens}, pos);
            std::vector<float> values(tokens * heads * dim);
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = std::sin(i * 0.17f);
            for (auto type : {DataType::FLOAT32, DataType::FLOAT16, DataType::BFLOAT16}) {
                Device("cpu");
                Data input(type, {1, tokens, heads, dim}, values);
                auto ref = Reference(Floats(input), tokens, heads, dim, pos, r, mm);
                for (auto device : cuda ? std::vector<std::string>{"cpu", "cuda"} : std::vector<std::string>{"cpu"}) {
                    Device("cpu");
                    Data data(type, {1, tokens, heads, dim}, values);
                    if (device == "cuda")
                        data.ToDevice(DataDevice::CUDA);
                    Device(device);
                    ApplyYarnRope(data, positions, r);
                    float tol = type == DataType::FLOAT32 ? 0.002f : type == DataType::FLOAT16 ? 0.003f : 0.02f;
                    Near(ref, Floats(data), tol, device + (mm ? " M-RoPE" : " RoPE"));
                }
            }
        }
        std::cout << "RoPE/M-RoPE high-position cases passed\n";
    }
    void FusedTests(bool cuda) {
        for (bool mm : {false, true})
            for (auto type : {DataType::FLOAT32, DataType::FLOAT16, DataType::BFLOAT16}) {
                auto r =
                    ResolveContextOptions(
                        mm ? MMConfig() : TextConfig(), {true, mm, mm ? "text_config." : ""},
                        {mm ? 1000000 : 131072,
                         mm ? "yarn" : R"({"rope_type":"yarn","factor":4,"original_max_position_embeddings":32768})"})
                        .rope;
                int tokens = 4, qheads = 2, kheads = 1, dim = mm ? 256 : 128, pageLen = 2, pages = 3;
                int total = (mm ? 2 * qheads + 2 * kheads : qheads + 2 * kheads) * dim;
                std::vector<float> raw(tokens * total), pos = {32768, 262144, 999999, 1000000};
                for (size_t i = 0; i < raw.size(); ++i)
                    raw[i] = std::sin(i * 0.13f) * 0.7f;
                if (mm) {
                    auto original = pos;
                    for (int row = 1; row < 3; ++row)
                        for (auto v : original)
                            pos.push_back(v + row * 11);
                }
                Device("cpu");
                Data packed(type, {tokens, 1, total}, raw);
                raw = Floats(packed);
                std::vector<float> qv, kv, vv, gv;
                for (int t = 0; t < tokens; ++t) {
                    for (int h = 0; h < qheads; ++h)
                        for (int j = 0; j < dim; ++j) {
                            qv.push_back(raw[t * total + h * dim * (mm ? 2 : 1) + j]);
                            if (mm)
                                gv.push_back(raw[t * total + h * dim * 2 + dim + j]);
                        }
                    int kstart = t * total + qheads * dim * (mm ? 2 : 1);
                    for (int j = 0; j < kheads * dim; ++j) {
                        kv.push_back(raw[kstart + j]);
                        vv.push_back(raw[kstart + kheads * dim + j]);
                    }
                }
                Data q(type, {1, tokens, qheads, dim}, qv), k(type, {1, tokens, kheads, dim}, kv);
                Data norm(DataType::FLOAT32, {dim}, std::vector<float>(dim, 1));
                Data positions(DataType::FLOAT32, {mm ? 3 : 1, tokens}, pos);
                RMSNorm(q, norm, 1e-6, q);
                RMSNorm(k, norm, 1e-6, k);
                ApplyYarnRope(q, positions, r);
                ApplyYarnRope(k, positions, r);
                auto refq = Floats(q), refk = Floats(k);
                // Plain CPU append has a decomposed fallback; gated append is a CUDA operator.
                for (auto device : cuda ? std::vector<std::string>{"cpu", "cuda"} : std::vector<std::string>{"cpu"}) {
                    if (mm && device == "cpu")
                        continue;
                    Device("cpu");
                    Data input(type, {tokens, 1, total}, raw), out(type, {tokens * qheads, 1, dim}),
                        gate(type, {tokens, 1, qheads * dim});
                    Data kc(type, {pages, pageLen, kheads, dim}, std::vector<float>(pages * pageLen * kheads * dim, 0));
                    Data vc(type, {pages, pageLen, kheads, dim}, std::vector<float>(pages * pageLen * kheads * dim, 0));
                    Data index(DataType::INT32, {tokens}), offset(DataType::INT32, {tokens});
                    index.Allocate();
                    offset.Allocate();
                    int indices[] = {2, 0, 1, 2}, offsets[] = {0, 1, 1, 1};
                    std::memcpy(index.cpuData, indices, sizeof(indices));
                    std::memcpy(offset.cpuData, offsets, sizeof(offsets));
                    Data weights(DataType::FLOAT32, {dim}, std::vector<float>(dim, 1));
                    // Ordinary fused decode uses [batch,1] positions; M-RoPE selects its packed rows.
                    Data pp(DataType::FLOAT32, mm ? std::vector<int>{3, tokens} : std::vector<int>{tokens, 1}, pos);
                    if (device == "cuda")
                        for (Data *d : {&input, &out, &gate, &kc, &vc, &index, &offset, &weights, &pp})
                            d->ToDevice(DataDevice::CUDA);
                    Device(device);
                    FloatDict fp = {{"eps", 1e-6}, {"ropeTheta", r.theta}, {"ropeScale", 1}};
                    IntDict ip = {
                        {"q_heads", qheads},  {"k_heads", kheads}, {"head_dim", dim}, {"rotaryDim", r.rotaryDim},
                        {"pageLen", pageLen}, {"batch", tokens},   {"doQKNorm", 1},   {"sectionT", 11},
                        {"sectionH", 11},     {"sectionW", 10}};
                    r.AddFusedParams(fp, ip);
                    DataDict data = {{mm ? "qgatekv" : "qkv", &input},
                                     {"qNormWeight", &weights},
                                     {"kNormWeight", &weights},
                                     {"positionIds", &pp},
                                     {"qOutput", &out},
                                     {"pagedKCacheData", &kc},
                                     {"pagedVCacheData", &vc},
                                     {"insertIndexs", &index},
                                     {"insertPositions", &offset}};
                    if (mm)
                        data["gateOutput"] = &gate;
                    static_cast<Executor *>(GetExecutor())
                        ->Run(mm ? "Qwen35QGateKVRMSNormRopeSplitAppendPagedCache"
                                 : "QKVRMSNormRopeSplitAppendPagedCache",
                              data, fp, ip);
                    float tol = type == DataType::FLOAT32 ? 0.003f : type == DataType::FLOAT16 ? 0.005f : 0.03f;
                    Near(refq, Floats(out), tol, "fused Q " + device);
                    auto actualk = Floats(kc), actualv = Floats(vc);
                    for (int t = 0; t < tokens; ++t)
                        for (int j = 0; j < kheads * dim; ++j) {
                            int dst = (indices[t] * pageLen + offsets[t]) * kheads * dim + j;
                            Check(std::abs(refk[t * kheads * dim + j] - actualk[dst]) <=
                                      tol * (1 + std::abs(refk[t * kheads * dim + j])),
                                  "fused K");
                            Check(vv[t * kheads * dim + j] == actualv[dst], "V changed");
                        }
                    if (mm)
                        Near(gv, Floats(gate), 0, "gate changed");
                }
            }
        std::cout << "Fused Q/K, gate and paged V cases passed\n";
    }
} // namespace
int main(int argc, char **argv) {
    try {
        bool cuda = argc > 1 && std::string(argv[1]) == "--cuda";
#ifdef USE_CUDA
        if (cuda) {
            int count = 0;
            if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0)
                return 77;
        }
#else
        if (cuda)
            return 77;
#endif
        SetThreads(4);
        ConfigTests();
        RotaryTests(cuda);
        FusedTests(cuda);
        std::cout << "Context extension regression passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
