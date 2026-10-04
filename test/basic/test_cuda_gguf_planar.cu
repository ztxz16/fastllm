#include "../../src/devices/cuda/fastllm-gguf-planar-dot.cuh"
#include "cuda_gguf_t8_test.cuh"
#include "cuda_gguf_weights_test.cuh"
#include "fastllm-cuda-gguf-planar.h"
#include <functional>

static int cases = 0, graphCases = 0;
__global__ void CheckIQ1CodebookKernel(int *errors) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2048)
        return;
    const uint64_t canonical = iq1s_grid[i];
    const uint32_t packed = fastllm_gguf_planar::PackIQ1Grid(canonical);
    for (int sign = -1; sign <= 1; sign += 2) {
        const uint32_t offset = sign < 0 ? 0x77777777u : 0x79797979u;
        const uint32_t lo = fastllm_gguf_planar::ExpandIQ1Values(packed & 0x0f0f0f0fu, offset);
        const uint32_t hi = fastllm_gguf_planar::ExpandIQ1Values((packed >> 4) & 0x0f0f0f0fu, offset);
        for (int j = 0; j < 8; ++j) {
            const int expected = int(int8_t(canonical >> (8 * j))) * 8 + sign;
            const int actual = int(int8_t((j < 4 ? lo : hi) >> (8 * (j % 4))));
            if (expected != actual)
                atomicAdd(errors, 1);
        }
    }
}
static void CheckIQ1Codebook() {
    Allocation errors(sizeof(int));
    Cuda(cudaMemset(errors.p, 0, sizeof(int)));
    CheckIQ1CodebookKernel<<<8, 256, 0, cudaStreamPerThread>>>(errors.as<int>());
    int count = 0;
    Cuda(cudaMemcpy(&count, errors.p, sizeof(int), cudaMemcpyDeviceToHost));
    Check(count == 0, "compact IQ1 codebook differs from canonical signed integers");
    ++cases;
}
template <typename Input, int Kind> static void QuantizeFormats() {
    constexpr int k = 768, t = 13, keyHeads = 3, groups = 2, headDim = 128;
    const size_t bytes = FastllmGgufPlanarBytes(t, k);
    Allocation dx(size_t(t) * k * sizeof(Input)), dq(bytes + 16);
    std::vector<Input> input(t * k);
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = Input(i % k < 32 ? 0.f : std::sin(float(i) * .09f));
    // Exact half-way cases and a negative maximum in a nonzero block.
    for (int i = 32; i < 64; ++i)
        input[i] = Input(i == 32 ? -127.f : (i % 2 ? 1.5f : -1.5f));
    Cuda(cudaMemcpy(dx.p, input.data(), input.size() * sizeof(Input), cudaMemcpyHostToDevice));
    std::vector<uint8_t> actual(bytes + 16), expected(bytes + 16, 0xcd);
    for (bool permuted : {false, true}) {
        Cuda(cudaMemset(dq.p, 0xcd, bytes + 16));
        Check(FastllmGgufQuantizePlanar(dx.p, Kind, dq.p, t, k, cudaStreamPerThread, permuted ? keyHeads : 0,
                                        groups, headDim),
              "input kind / permutation rejected");
        Cuda(cudaMemcpy(actual.data(), dq.p, actual.size(), cudaMemcpyDeviceToHost));
        for (int b = 0; b < t * k / 32; ++b) {
            float x[32], sum[32], a = 0;
            for (int l = 0; l < 32; ++l) {
                const int at = b * 32 + l, i = at % k, head = i / headDim;
                const int source =
                    permuted ? ((head % keyHeads) * groups + head / keyHeads) * headDim + i % headDim : i;
                x[l] = sum[l] = float(input[(at / k) * k + source]);
                a = std::max(a, std::fabs(x[l]));
            }
            for (int d = 16; d; d >>= 1) {
                float next[32];
                for (int l = 0; l < 32; ++l)
                    next[l] = sum[l] + sum[l ^ d];
                std::memcpy(sum, next, sizeof(sum));
            }
            const float scale = a / 127;
            int lo = 0, hi = 0;
            for (int l = 0; l < 32; ++l) {
                int8_t q = a == 0 ? 0 : int8_t(std::round(x[l] / scale));
                expected[b * 32 + l] = uint8_t(q);
                (l < 16 ? lo : hi) += q;
            }
            const half ds[2] = {__float2half_rn(scale), __float2half_rn(sum[0])};
            const short sums[2] = {short(lo), short(hi)};
            std::memcpy(expected.data() + t * k + b * 4, ds, 4);
            std::memcpy(expected.data() + t * k + (t * k / 32) * 4 + b * 4, sums, 4);
        }
        Check(actual == expected, "input conversion / head permutation differs from CPU oracle");
        ++cases;
    }
    const auto before = actual;
    Check(!FastllmGgufQuantizePlanar(dx.p, Kind, dq.p, t, k, cudaStreamPerThread, 3, 2, 64),
          "invalid permutation accepted");
    Check(!FastllmGgufQuantizePlanar(dx.p, 3, dq.p, t, k, cudaStreamPerThread),
          "invalid input kind accepted");
    Check(!FastllmGgufProjectPlanar(GGML_TYPE_Q8_0, 0, dx.p, nullptr, dq.p, dq.p, t, k, 1, 1,
                                    cudaStreamPerThread),
          "invalid type accepted");
    Check(FastllmGgufPlanarBytes(0, k) == 0 && FastllmGgufPlanarBytes(17, k) == 0 &&
              FastllmGgufPlanarBytes(1, 255) == 0,
          "invalid workspace shape accepted");
    Cuda(cudaMemcpy(actual.data(), dq.p, actual.size(), cudaMemcpyDeviceToHost));
    Check(actual == before, "rejected API wrote output");
}
__global__ void Epilogue(half *gate, const half *up, const half *initial, int mode, int tokens, int n,
                         int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tokens * n)
        return;
    int at = (i / n) * stride + i % n;
    if (mode == 1)
        gate[at] = __hadd(initial[at], gate[at]);
    else {
        half g = mode == 2 ? initial[at] : gate[at];
        half u = mode == 2 ? gate[at] : up[at];
        gate[at] = __hmul(__hdiv(g, __hadd(__float2half(1), hexp(-g))), u);
    }
}
template <ggml_type Type>
static void Reference(const void *w, const block_q8_1 *x, half *y, int t, int k, int n, int stride) {
    for (int first = 0; first < t; first += 8) {
#define TILE(T)                                                                                              \
    case T:                                                                                                  \
        fastllm_gguf_small_mmvq::LaunchBatchTokens<Type, T, half, 0>(                                        \
            w, x + first * (k / 32), y + first * stride, k, n, k, stride, cudaStreamPerThread);              \
        break
        switch (std::min(t - first, 8)) {
            TILE(1);
            TILE(2);
            TILE(3);
            TILE(4);
            TILE(5);
            TILE(6);
            TILE(7);
            TILE(8);
        }
#undef TILE
    }
}
template <ggml_type Type> static void Test(int t, int k, int n, bool graph) {
    const int stride = n + 11, blocks = t * k / 32;
    const auto w = Weights(Type, n, k);
    const size_t qbytes = FastllmGgufPlanarBytes(t, k), count = size_t(t) * stride + 16;
    Allocation dw(w.size()), du(w.size()), dx(size_t(t) * k * 2), dq(qbytes + 16),
        da(size_t(blocks) * sizeof(block_q8_1)), dy(count * 2), dr(count * 2), dur(count * 2), di(count * 2);
    Cuda(cudaMemcpy(dw.p, w.data(), w.size(), cudaMemcpyHostToDevice));
    auto up = w;
    std::rotate(up.begin(), up.begin() + ggml_row_size(Type, k), up.end());
    Cuda(cudaMemcpy(du.p, up.data(), up.size(), cudaMemcpyHostToDevice));
    std::vector<half> x(size_t(t) * k), initial(count), old(count), now(count);
    for (size_t i = 0; i < count; ++i)
        initial[i] = __float2half_rn(.2f * std::sin(float(i) * .031f));
    Cuda(cudaMemcpy(di.p, initial.data(), count * 2, cudaMemcpyHostToDevice));
    std::vector<block_q8_1> aos(blocks);
    std::vector<uint8_t> expected(qbytes + 16, 0xcd), quantized(expected.size());
    auto prepare = [&](int seed) {
        for (size_t i = 0; i < x.size(); ++i)
            x[i] = __float2half_rn(i % k < 32 || (t > 1 && i / k == size_t(t - 1))
                                       ? 0
                                       : std::sin(float(i + seed) * .117f) * (.5f + float(i % 7) / 5));
        for (int b = 0; b < blocks; ++b) {
            float a = 0, sum[32];
            for (int l = 0; l < 32; ++l) {
                sum[l] = float(x[b * 32 + l]);
                a = std::max(a, std::fabs(sum[l]));
            }
            for (int d = 16; d; d >>= 1) {
                float next[32];
                for (int l = 0; l < 32; ++l)
                    next[l] = sum[l] + sum[l ^ d];
                std::memcpy(sum, next, sizeof(sum));
            }
            int lo = 0, hi = 0;
            const float scale = a / 127;
            for (int l = 0; l < 32; ++l) {
                int8_t q = a == 0 ? 0 : int8_t(std::round(float(x[b * 32 + l]) / scale));
                aos[b].qs[l] = q;
                expected[b * 32 + l] = uint8_t(q);
                (l < 16 ? lo : hi) += int(q);
            }
            aos[b].ds = __floats2half2_rn(scale, sum[0]);
            std::memcpy(expected.data() + t * k + b * 4, &aos[b].ds, 4);
            const short sums[2] = {short(lo), short(hi)};
            std::memcpy(expected.data() + t * k + blocks * 4 + b * 4, sums, 4);
        }
        Cuda(cudaMemcpy(dx.p, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(da.p, aos.data(), aos.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice));
    };
    auto compare = [&] {
        Cuda(cudaDeviceSynchronize());
        Cuda(cudaMemcpy(quantized.data(), dq.p, quantized.size(), cudaMemcpyDeviceToHost));
        Check(quantized == expected, "quantization differs from independent CPU oracle / tail overwritten");
        Cuda(cudaMemcpy(old.data(), dr.p, count * 2, cudaMemcpyDeviceToHost));
        Cuda(cudaMemcpy(now.data(), dy.p, count * 2, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i)
            if (std::memcmp(&old[i], &now[i], 2)) {
                std::cerr << "type=" << ggml_type_name(Type) << " T=" << t << " K=" << k << " N=" << n
                          << " at=" << i << " old=" << float(old[i]) << " got=" << float(now[i]) << std::endl;
                throw std::runtime_error("bitwise reference or output guard failed");
            }
    };
    for (int mode = 0; mode < 4; ++mode) {
        auto reference = [&] {
            Cuda(cudaMemcpy(dr.p, initial.data(), count * 2, cudaMemcpyHostToDevice));
            Cuda(cudaMemcpy(dur.p, initial.data(), count * 2, cudaMemcpyHostToDevice));
            Reference<Type>(dw.p, da.as<block_q8_1>(), dr.as<half>(), t, k, n, stride);
            if (mode == 3)
                Reference<Type>(du.p, da.as<block_q8_1>(), dur.as<half>(), t, k, n, stride);
            if (mode)
                Epilogue<<<(t * n + 255) / 256, 256, 0, cudaStreamPerThread>>>(
                    dr.as<half>(), dur.as<half>(), di.as<half>(), mode, t, n, stride);
        };
        auto run = [&] {
            Check(FastllmGgufQuantizePlanar(dx.p, 1, dq.p, t, k, cudaStreamPerThread), "quantize rejected");
            Check(FastllmGgufProjectPlanar(Type, mode, dw.p, du.p, dq.p, dy.p, t, k, n, stride,
                                           cudaStreamPerThread),
                  "projection rejected");
        };
        auto reset = [&] {
            Cuda(cudaMemset(dq.p, 0xcd, qbytes + 16));
            Cuda(cudaMemcpy(dy.p, initial.data(), count * 2, cudaMemcpyHostToDevice));
        };
        prepare(0);
        reset();
        reference();
        run();
        compare();
        ++cases;
        if (mode == 0) {
            std::vector<float> decoded(k);
            for (int row : {0, n / 2, n - 1}) {
                ggml_type_to_float(Type)(w.data() + size_t(row) * ggml_row_size(Type, k), decoded.data(), k);
                for (int token = 0; token < t; ++token) {
                    double dot = 0, magnitude = 0;
                    for (int j = 0; j < k; ++j) {
                        const auto &q = aos[token * (k / 32) + j / 32];
                        const double term = double(decoded[j]) * float(__low2half(q.ds)) * int(q.qs[j % 32]);
                        dot += term;
                        magnitude += std::fabs(term);
                    }
                    const double value = float(now[token * stride + row]);
                    // Legacy IQ integer scaling rounds before the float dot.
                    Check(std::fabs(value - dot) <= .002 * magnitude + .001 * std::fabs(dot) + .00005,
                          "independent GGUF dequantized FP64 oracle failed");
                }
            }
        }
        if (graph) {
            cudaGraph_t g;
            cudaGraphExec_t exec;
            Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            run();
            Cuda(cudaStreamEndCapture(cudaStreamPerThread, &g));
            Cuda(cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0));
            prepare(83);
            reset();
            reference();
            Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
            compare();
            Cuda(cudaGraphExecDestroy(exec));
            Cuda(cudaGraphDestroy(g));
            ++graphCases;
        }
    }
}
static void MixedReference(int type, const void *w, const block_q8_1 *x, half *y, int t, int k, int n,
                           int stride) {
#define TYPE(G)                                                                                              \
    case G:                                                                                                  \
        Reference<G>(w, x, y, t, k, n, stride);                                                              \
        break
    switch (static_cast<ggml_type>(type)) {
        TYPE(GGML_TYPE_IQ3_S);
        TYPE(GGML_TYPE_IQ3_XXS);
        TYPE(GGML_TYPE_IQ4_XS);
        TYPE(GGML_TYPE_Q4_K);
        TYPE(GGML_TYPE_Q2_K);
        TYPE(GGML_TYPE_IQ2_S);
        TYPE(GGML_TYPE_IQ2_XS);
        TYPE(GGML_TYPE_IQ2_XXS);
        TYPE(GGML_TYPE_IQ1_M);
    default:
        throw std::runtime_error("invalid reference type");
    }
#undef TYPE
}

static void Mixed(int gateType, int upType, int t, int k, int n, bool graph) {
    const int stride = n + 7, blocks = t * k / 32;
    const size_t count = size_t(t) * stride + 16, qbytes = FastllmGgufPlanarBytes(t, k);
    auto gate = Weights(static_cast<ggml_type>(gateType), n, k);
    auto up = Weights(static_cast<ggml_type>(upType), n, k);
    std::rotate(up.begin(), up.begin() + ggml_row_size(static_cast<ggml_type>(upType), k), up.end());
    Allocation dg(gate.size()), du(up.size()), dq(qbytes), da(size_t(blocks) * sizeof(block_q8_1)),
        dy(count * 2), dr(count * 2), dur(count * 2);
    Cuda(cudaMemcpy(dg.p, gate.data(), gate.size(), cudaMemcpyHostToDevice));
    Cuda(cudaMemcpy(du.p, up.data(), up.size(), cudaMemcpyHostToDevice));
    std::vector<block_q8_1> aos(blocks);
    std::vector<uint8_t> planar(qbytes);
    std::vector<half> expected(count), actual(count);
    auto prepare = [&](uint32_t seed) {
        for (int b = 0; b < blocks; ++b) {
            int lo = 0, hi = 0;
            for (int j = 0; j < 32; ++j) {
                seed = seed * 1664525u + 1013904223u;
                const int8_t q = b % 7 == 0 ? 0 : int(seed >> 24) - 128;
                aos[b].qs[j] = q;
                planar[b * 32 + j] = uint8_t(q);
                (j < 16 ? lo : hi) += q;
            }
            aos[b].ds = __floats2half2_rn(.007f, (lo + hi) * .007f);
            const short sums[2] = {short(lo), short(hi)};
            std::memcpy(planar.data() + t * k + b * 4, &aos[b].ds, 4);
            std::memcpy(planar.data() + t * k + blocks * 4 + b * 4, sums, 4);
        }
        Cuda(cudaMemcpy(dq.p, planar.data(), qbytes, cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(da.p, aos.data(), aos.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice));
        Cuda(cudaMemset(dy.p, 0xcd, count * 2));
        Cuda(cudaMemset(dr.p, 0xcd, count * 2));
    };
    auto reference = [&] {
        MixedReference(gateType, dg.p, da.as<block_q8_1>(), dr.as<half>(), t, k, n, stride);
        MixedReference(upType, du.p, da.as<block_q8_1>(), dur.as<half>(), t, k, n, stride);
        Epilogue<<<(t * n + 255) / 256, 256, 0, cudaStreamPerThread>>>(dr.as<half>(), dur.as<half>(), nullptr,
                                                                       3, t, n, stride);
    };
    auto run = [&] {
        Check(FastllmGgufGateUpPlanar(gateType, upType, dg.p, du.p, dq.p, dy.p, t, k, n, stride,
                                      cudaStreamPerThread),
              "mixed projection rejected");
    };
    auto compare = [&] {
        Cuda(cudaMemcpy(expected.data(), dr.p, count * 2, cudaMemcpyDeviceToHost));
        Cuda(cudaMemcpy(actual.data(), dy.p, count * 2, cudaMemcpyDeviceToHost));
        if (std::memcmp(expected.data(), actual.data(), count * 2)) {
            std::cerr << "mixed gate=" << gateType << " up=" << upType << " T=" << t << " K=" << k
                      << " N=" << n << std::endl;
            throw std::runtime_error("mixed projection / guards differ from separate reference");
        }
    };
    prepare(18379);
    reference();
    run();
    compare();
    ++cases;
    if (graph) {
        cudaGraph_t g;
        cudaGraphExec_t exec;
        Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        run();
        Cuda(cudaStreamEndCapture(cudaStreamPerThread, &g));
        Cuda(cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0));
        prepare(9183);
        reference();
        Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
        compare();
        Cuda(cudaGraphExecDestroy(exec));
        Cuda(cudaGraphDestroy(g));
        ++graphCases;
    }
    Check(!FastllmGgufGateUpPlanar(gateType, GGML_TYPE_Q8_0, dg.p, du.p, dq.p, dy.p, t, k, n, stride,
                                   cudaStreamPerThread),
          "invalid mixed type accepted");
    Check(!FastllmGgufGateUpPlanar(gateType, upType, dg.p, du.p, dq.p, dy.p, t, k, n, n - 1,
                                   cudaStreamPerThread),
          "invalid mixed stride accepted");
    compare();
}

static void AllMixed() {
    constexpr ggml_type types[] = {GGML_TYPE_IQ3_S,  GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS,
                                   GGML_TYPE_Q4_K,   GGML_TYPE_Q2_K,    GGML_TYPE_IQ2_S,
                                   GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ1_M};
    for (auto gate : types) {
        for (auto up : types) {
            for (int t = 1; t <= 16; ++t)
                Mixed(gate, up, t, t % 2 ? 768 : 1280, 19, t % 2 == 0);
            for (int k : {4096, 5120, 6144, 7168, 17408})
                Mixed(gate, up, 8, k, 131, false);
        }
        std::cout << "PASS mixed gate=" << ggml_type_name(gate) << std::endl;
    }
}

template <ggml_type Type> static void All() {
    if constexpr (Type == GGML_TYPE_IQ3_S || Type == GGML_TYPE_IQ3_XXS) {
        // Exercise the single-row expansion dispatch, padded output strides
        // and a final one-row tile after an eight-row tile, including K tails.
        Test<Type>(1, 256, 257, true);
        Test<Type>(1, 768, 769, false);
        Test<Type>(9, 1280, 1537, true);
        Test<Type>(1, 4096, 4097, false);
        Test<Type>(1, 6144, 6145, false);
        Mixed(Type, Type, 1, 256, 257, true);
        Mixed(Type, Type, 9, 1280, 1537, true);
    }
    for (int t = 1; t <= 16; ++t)
        Test<Type>(t, t % 2 ? 768 : 1280, 19, true);
    for (int t : {1, 2, 3, 4, 5, 6, 7, 8, 12, 16})
        Test<Type>(t, 6144, 4097, false);
    for (int k : {256, 4096, 5120, 7168, 17408, 20480})
        Test<Type>(8, k, 131, true);
    std::cout << "PASS " << ggml_type_name(Type) << std::endl;
}
static void StreamingIQ4() {
    int device = 0, l2Bytes = 0;
    Cuda(cudaGetDevice(&device));
    Cuda(cudaDeviceGetAttribute(&l2Bytes, cudaDevAttrL2CacheSize, device));
    if (l2Bytes <= 0)
        return;
    // Select a pair larger than this device's L2, so the public API actually
    // exercises streaming loads. Odd N checks output tails; T>8 checks tiling.
    constexpr int k = 5120;
    const size_t pairRowBytes = 2 * ggml_row_size(GGML_TYPE_IQ4_XS, k);
    const int n = int(size_t(l2Bytes) / pairRowBytes + 1) | 1;
    for (int t = 1; t <= 16; ++t)
        Test<GGML_TYPE_IQ4_XS>(t, k, n, true);
    std::cout << "PASS streaming IQ4 rows=1..16 N=" << n << " L2=" << l2Bytes << std::endl;
}

int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            return 77;
        Cuda(cudaSetDevice(0));
        CheckIQ1Codebook();
        QuantizeFormats<float, 0>();
        QuantizeFormats<half, 1>();
        QuantizeFormats<__nv_bfloat16, 2>();
        All<GGML_TYPE_IQ3_S>();
        All<GGML_TYPE_IQ3_XXS>();
        All<GGML_TYPE_IQ4_XS>();
        All<GGML_TYPE_Q4_K>();
        All<GGML_TYPE_Q2_K>();
        All<GGML_TYPE_IQ2_S>();
        All<GGML_TYPE_IQ2_XS>();
        All<GGML_TYPE_IQ2_XXS>();
        All<GGML_TYPE_IQ1_M>();
        AllMixed();
        StreamingIQ4();
        std::cout << "PASS planar general cases=" << cases << " graphs=" << graphCases << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
