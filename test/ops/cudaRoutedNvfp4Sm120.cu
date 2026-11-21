// Compare the routed SM120 block-scaled MMA with an independent FP64 dot.
// Include shuffled routes, signed block scales and independently scaled experts.
#include "../../src/models/inkling2/runtime.cuh"
#include "../../src/devices/cuda/moe/fastllm-nvfp4-routed-sm120.cuh"
#include <cstdio>
#include <random>

namespace {
namespace op = fastllm::cuda::routed_fp4;
using fastllm::inkling2::Buffer;
using fastllm::inkling2::check;

template <class T> Buffer<T> copy(const std::vector<T> &v) {
    Buffer<T> d(v.size());
    check(cudaMemcpy(d.p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
double fp4(unsigned packed, int pos) {
    const double levels[] = {0, .5, 1, 1.5, 2, 3, 4, 6};
    unsigned v = (packed >> (4 * pos)) & 15;
    return (v & 8 ? -1 : 1) * levels[v & 7];
}
double scale(unsigned byte) {
    int a = byte & 127, exp = a / 8;
    double v = exp ? std::ldexp(1. + (a % 8) / 8., exp - 7) : a / 512.;
    return byte & 128 ? -v : v;
}
void test(int tokens, bool up, bool skew) {
    const int experts = 6, top = 3, n = 48, k = 192, routes = tokens * top, div = up ? top : 1;
    const int capacity = (routes + (op::RouteTile - 1) * std::min(experts, routes)) / op::RouteTile;
    std::mt19937 rng(107);
    std::vector<uint32_t> w(size_t(experts) * n * k / 8);
    std::vector<unsigned char> sf(size_t(experts) * n * k / 16);
    std::vector<float> global(experts);
    std::vector<op::Block16> x(size_t(routes / div) * k / 16);
    std::vector<int> ids(routes);
    for (auto &i : w)
        i = rng();
    for (auto &i : sf)
        i = (rng() % 82) | ((rng() % 2) * 128);
    for (int i = 0; i < experts; ++i)
        global[i] = float(i) / 7.f;
    for (size_t i = 0; i < x.size(); ++i) {
        auto &b = x[i];
        b.code[0] = rng();
        b.code[1] = rng();
        b.scale = rng() % 82;
        b.outer = float((i / (k / 16) + 1) % 9) / 7.f;
    }
    for (int i = 0; i < routes; ++i)
        ids[i] = skew ? 4 : (i / top * 5 + i % top * 2) % experts;
    auto wi = copy(w);
    auto si = copy(sf);
    auto gi = copy(global);
    auto xi = copy(x);
    auto di = copy(ids);
    Buffer<uint4> codes(w.size() / 4);
    Buffer<uint32_t> scales(sf.size() / 4);
    Buffer<op::Tile> plan(capacity);
    Buffer<int> used(1);
    Buffer<op::InputTile> packed(size_t(capacity) * (op::RouteTile / 8) * (k / 64));
    Buffer<__nv_bfloat16> y(size_t(routes) * n);
    const size_t words = size_t(n) * k / 8;
    for (int e = 0; e < experts; ++e) {
        op::pack<<<(words / 4 + 255) / 256, 256>>>(wi.p + e * words,
                                                   reinterpret_cast<uint32_t *>(si.p) + e * words / 8,
                                                   codes.p + e * words / 4, scales.p + e * words / 8, n, k, up);
    }
    op::Weights view{codes.p, scales.p, gi.p};
    for (int pass = 0; pass < 2; ++pass) {
        std::reverse(ids.begin(), ids.end());
        check(cudaMemcpy(di.p, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice));
        check(cudaMemset(y.p, 0xff, size_t(routes) * n * sizeof(__nv_bfloat16)));
        if (tokens == 1) {
            op::decodeProject<4><<<dim3(n / 16, routes), 128>>>(view, xi.p, di.p, y.p, n, k, div);
        } else {
            check(cudaMemsetAsync(used.p, 0, sizeof(int), cudaStreamPerThread));
            op::groupRoutes<<<experts, 256>>>(di.p, routes, plan.p, used.p);
            op::packInput<<<dim3((k / 64 + 7) / 8, capacity, op::RouteTile / 8), 256>>>(plan.p, used.p, xi.p,
                                                                                        packed.p, k, div);
            op::groupedProject<8>
                <<<dim3((n / 16 + 7) / 8, capacity), 256>>>(view, xi.p, packed.p, plan.p, used.p, y.p, n, k, div);
        }
        std::vector<__nv_bfloat16> got(routes * n);
        check(cudaMemcpy(got.data(), y.p, got.size() * sizeof(got[0]), cudaMemcpyDeviceToHost));
        double error = 0, norm = 0;
        for (int i = 0; i < routes; ++i)
            for (int j = 0; j < n; ++j) {
                const int expertRow = ids[i] * n + (up ? 2 * (j % (n / 2)) + j / (n / 2) : j);
                double ref = 0;
                for (int c = 0; c < k; ++c) {
                    const auto &b = x[(i / div) * (k / 16) + c / 16];
                    ref += fp4(b.code[(c % 16) / 8], c % 8) * scale(b.scale) *
                           fp4(w[expertRow * (k / 8) + c / 8], c % 8) * scale(sf[expertRow * (k / 16) + c / 16]);
                }
                ref *= global[ids[i]] * double(x[(i / div) * (k / 16)].outer);
                const double diff = float(got[i * n + j]) - ref;
                error += diff * diff;
                norm += ref * ref;
                if (!std::isfinite(diff) || std::abs(diff) > std::abs(ref) * .004 + 1e-4)
                    throw std::runtime_error("Incorrect routed NVFP4 output");
            }
        printf("tokens=%d gate=%d skew=%d pass=%d relL2=%.6g\n", tokens, up, skew, pass,
               std::sqrt(error / std::max(norm, 1e-30)));
    }
}
}
int main() {
    int count = 0, image = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    check(cudaMemcpyFromSymbol(&image, op::nativeImage, sizeof(image)));
    if (!image) return 77;
    for (bool up : {false, true})
        for (int rows : {1, 3, 31, 257})
            for (bool skew : {false, true})
                test(rows, up, skew);
}
