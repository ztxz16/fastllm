// Cover the compressed CUTLASS iterator with empty experts, partial tiles,
// E4M3 subnormals and different power-of-two scales on adjacent K groups.
#include "../../src/models/inkling2/runtime.cuh"
#include "../../src/models/inkling2/mxfp8-sm75.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cmath>
#include <cstdio>

using namespace fastllm::inkling2;
namespace mx = fastllm::inkling2::sm75;

template<class T> struct Managed {
    T *p;
    explicit Managed(size_t n) { check(cudaMallocManaged(&p, n * sizeof(T))); }
    ~Managed() { cudaFree(p); }
    Managed(const Managed &) = delete;
};

float decode(uint8_t code) {
    int e = (code >> 3) & 15, m = code & 7;
    float value = e ? std::ldexp(1.f + m / 8.f, e - 7) : std::ldexp(float(m), -9);
    return code & 128 ? -value : value;
}

int main() {
    if (FastllmCudaRuntimeArch() != 75) return 77;
    constexpr int n = 160, k = 96, rows = 169;
    Managed<mx::Plan> plan(1);
    Managed<uint8_t> codes(mx::Experts * n * k);
    Managed<float> scales(mx::Experts * n * k / 32);
    Managed<uint8_t *> weightTable(mx::Experts);
    Managed<float *> scaleTable(mx::Experts);
    Managed<half> input(rows * k);
    Managed<float> output(rows * n);
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < k; ++j) input.p[r*k+j] = __float2half(j == r % k ? 1.f : 0.f);
    for (int e = 0; e < mx::Experts; ++e) {
        plan.p->counts[e] = e == 0 ? 1 : e == 42 ? 31 : e == 127 ? 137 : 0;
        weightTable.p[e] = codes.p + e*n*k;
        scaleTable.p[e] = scales.p + e*n*k/32;
        for (int i = 0; i < n*k; ++i) {
            int q = (17*i + e) & 255;
            weightTable.p[e][i] = (q & 127) == 127 ? q-1 : q;
            if (i % 32 == 0) scaleTable.p[e][i/32] = std::ldexp(1.f, (i/32 + e) % 23 - 15);
        }
    }
    const auto stream = cudaStreamPerThread;
    mx::prefixRoutes<<<1,1,0,stream>>>(plan.p);
    auto verify = [&](auto tile) {
        constexpr int m = decltype(tile)::value;
        mx::project<m>(plan.p, input.p, output.p, weightTable.p, scaleTable.p, n, k,
                       mx::Grouped<m>::Op::sufficient(), stream);
        check(cudaStreamSynchronize(stream));
        int r = 0;
        for (int e = 0; e < mx::Experts; ++e)
            for (int i = 0; i < plan.p->counts[e]; ++i, ++r)
                for (int j = 0; j < n; ++j) {
                    const int col = j*k + r%k;
                    const float ref = decode(weightTable.p[e][col]) * scaleTable.p[e][col/32];
                    if (output.p[r*n+j] != ref) throw std::runtime_error("SM75 MXFP8 weight or scale mismatch");
                }
    };
    verify(std::integral_constant<int,16>{});
    verify(std::integral_constant<int,128>{});
    printf("Compared %d FP32 outputs per tile shape\n", rows*n);
}
