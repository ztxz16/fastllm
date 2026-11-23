#include "align.h"
#include <cmath>
namespace fastllm::duo::live {
using native::Tensor;
namespace {
void fixTimestamps(std::vector<int> &a) {
    int n = a.size();
    if (!n)
        return;
    std::vector<int> dp(n, 1), parent(n, -1);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < i; ++j)
            if (a[j] <= a[i] && dp[j] + 1 > dp[i]) {
                dp[i] = dp[j] + 1;
                parent[i] = j;
            }
    std::vector<bool> keep(n);
    for (int k = std::max_element(dp.begin(), dp.end()) - dp.begin(); k >= 0; k = parent[k])
        keep[k] = true;
    for (int i = 0; i < n;) {
        if (keep[i]) {
            ++i;
            continue;
        }
        int j = i + 1;
        while (j < n && !keep[j])
            ++j;
        for (int k = i; k < j; ++k) {
            if (!i)
                a[k] = a[j];
            else if (j == n)
                a[k] = a[i - 1];
            else if (j - i <= 2)
                a[k] = (k - i + 1 <= j - k) ? a[i - 1] : a[j];
            else
                a[k] = a[i - 1] + double(a[j] - a[i - 1]) * (k - i + 1) / (j - i + 1);
        }
        i = j;
    }
}
} // namespace
Aligner::Aligner(const std::string &root, int device) : weights(device) {
    weights.load(root + "/speech/forced-aligner/model.safetensors", {"thinker."});
}
Tensor Aligner::mel(const std::vector<float> &pcm) {
    if (pcm.size() < 400 || pcm.size() > 16000 * 31)
        throw std::invalid_argument("invalid audio duration");
    auto powers = native::melPower(pcm, weights.device);
    auto invMel = [](double m) {
        return m < 15 ? m * (200. / 3) : 1000. * std::exp(std::log(6.4) / 27. * (m - 15));
    };
    double top = 15 + std::log(8.) * 27. / std::log(6.4);
    std::vector<float> filter(128 * 201);
    for (int i = 0; i < 128; ++i) {
        double l = invMel(i * top / 129), c = invMel((i + 1) * top / 129), r = invMel((i + 2) * top / 129);
        for (int j = 0; j < 201; ++j)
            filter[i * 201 + j] =
                2. / (r - l) * std::max(0., std::min((40 * j - l) / (c - l), (r - 40 * j) / (r - c)));
    }
    auto s = native::linear(powers.transpose(0, 1),
                            native::from_blob(filter.data(), {128, 201}, native::kFloat).to(weights.device))
                 .transpose(0, 1)
                 .clamp_min(1e-10)
                 .log10();
    return ((native::maximum(s, s.max() - 8.) + 4.) / 4.).to(native::kBFloat16);
}
Tensor Aligner::encode(const Tensor &features) {
    const std::string p = "thinker.audio_tower.";
    int nf = features.size(1), width = std::min(nf, 100), groups = (nf + 99) / 100;
    auto padded = native::zeros({128, groups * width}, features.options());
    padded.narrow(1, 0, nf).copy_(features);
    auto x = padded.reshape({128, groups, width}).transpose(0, 1).unsqueeze(1);
    for (int i = 1; i <= 3; ++i) {
        auto k = p + "conv2d" + std::to_string(i);
        x = native::gelu(native::conv2dStride2(x, weights.get(k + ".weight"), weights.get(k + ".bias")));
    }
    int t = x.size(3), tail = ((nf - 1) % 100 + 1 + 7) / 8;
    x = weights.linear(x.permute({0, 3, 1, 2}).reshape({groups, t, -1}), p + "conv_out");
    auto opts = x.options().dtype(native::kFloat);
    auto theta =
        native::arange(t, opts).unsqueeze(1) * (-std::log(10000.) * native::arange(512, opts) / 511.).exp();
    x = (x + native::cat({theta.sin(), theta.cos()}, 1).to(x.scalar_type()))
            .reshape({groups * t, 1024})
            .narrow(0, 0, (groups - 1) * t + tail);
    auto ln = [&](const Tensor &a, const std::string &s) {
        return native::normalize(a, weights.get(s + ".weight"), weights.get(s + ".bias"), true, 1e-5f);
    };
    // Independent bidirectional encoder windows span eight 100-frame input slices.
    std::vector<Tensor> spans;
    for (auto v : x.split(t * 8)) {
        for (int i = 0; i < 24; ++i) {
            auto k = p + "layers." + std::to_string(i) + ".";
            auto y = ln(v, k + "self_attn_layer_norm");
            auto proj = [&](std::string a) {
                return weights.linear(y, k + "self_attn." + a + "_proj")
                    .reshape({1, -1, 16, 64})
                    .transpose(1, 2);
            };
            auto z = native::attention(proj("q"), proj("k"), proj("v")).transpose(1, 2).reshape(y.sizes());
            v = v + weights.linear(z, k + "self_attn.out_proj");
            v = v + weights.linear(native::gelu(weights.linear(ln(v, k + "final_layer_norm"), k + "fc1")),
                                   k + "fc2");
        }
        spans.push_back(v);
    }
    return weights.linear(native::gelu(weights.linear(ln(native::cat(spans), p + "ln_post"), p + "proj1")),
                          p + "proj2");
}
std::vector<int> Aligner::align(const std::vector<float> &pcm, const std::vector<std::vector<int>> &words) {

    native::DeviceGuard guard(weights.device);
    if (words.empty())
        throw std::invalid_argument("empty speech");
    auto audio = encode(mel(pcm));
    std::vector<int> ids{151669}, selected;
    ids.insert(ids.end(), audio.size(0), 151676);
    ids.push_back(151670);
    for (auto &w : words) {
        ids.insert(ids.end(), w.begin(), w.end());
        for (int i = 0; i < 2; ++i) {
            selected.push_back(ids.size());
            ids.push_back(151705);
        }
    }
    auto indices = [&](const std::vector<int> &i) {
        return native::rowIndices(i, weights.device);
    };
    const std::string p = "thinker.model.";
    auto x = weights.get(p + "embed_tokens.weight").index_select(0, indices(ids));
    x.narrow(0, 1, audio.size(0)).copy_(audio);
    auto opts = x.options().dtype(native::kFloat);
    auto angles = native::arange(x.size(0), opts).unsqueeze(1) *
                  native::pow(1000000., native::arange(64, opts) / 64.).reciprocal();
    angles = native::cat({angles, angles}, 1).unsqueeze(0).unsqueeze(0);
    auto c = angles.cos().to(x.scalar_type()), s = angles.sin().to(x.scalar_type());
    auto norm = [&](const Tensor &a, const std::string &key) {
        return native::normalize(a, weights.get(key + ".weight"), {}, false, 1e-6f, true);
    };
    for (int i = 0; i < 28; ++i) {
        auto k = p + "layers." + std::to_string(i) + ".";
        auto y = norm(x, k + "input_layernorm");
        auto head = [&](std::string kind, int h) {
            auto a =
                weights.linear(y, k + "self_attn." + kind + "_proj").reshape({1, -1, h, 128}).transpose(1, 2);
            if (kind == "v")
                return a;
            a = norm(a, k + "self_attn." + kind + "_norm");
            auto half = a.chunk(2, -1);
            return a * c + native::cat({-half[1], half[0]}, -1) * s;
        };
        auto z = native::attention(head("q", 16), head("k", 8), head("v", 8), true);
        x = x + weights.linear(z.transpose(1, 2).reshape({x.size(0), 2048}), k + "self_attn.o_proj");
        y = norm(x, k + "post_attention_layernorm");
        x = x + weights.linear(native::silu(weights.linear(y, k + "mlp.gate_proj")) *
                                   weights.linear(y, k + "mlp.up_proj"),
                               k + "mlp.down_proj");
    }
    auto scores = weights.linear(norm(x.index_select(0, indices(selected)), p + "norm"), "thinker.lm_head");
    auto stamps = (scores.argmax() * 80).to(native::kCPU, native::kInt).contiguous();
    auto ptr = stamps.data_ptr<int>();
    std::vector<int> out(ptr, ptr + stamps.numel());
    fixTimestamps(out);
    return out;
}
} // namespace fastllm::duo::live
