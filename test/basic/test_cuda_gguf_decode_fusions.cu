#include "cuda_gguf_fusion_test.cuh"
#include <cuda_bf16.h>

static void Shared(ggml_type a, ggml_type b, ggml_type c, int t, int k, int n) {
    Data x(FLOAT16, {1, t, k});
    Allocate(x);
    Projection p(a, t, k, n), q(b, t, k, n + 1), r(c, t, k, n + 8);
    Data *weights[] = {&p.w, &q.w, &r.w}, *outputs[] = {&p.y, &q.y, &r.y};
    Graph([&] { Check(FastllmCudaGGUFLinearShared(x, weights, outputs, 3), "shared rejected"); },
          [&] {
              p.Compare();
              q.Compare();
              r.Compare();
          },
          [&](int seed) {
              Upload(x, Input(t, k, seed * 53));
              p.Reference(x);
              q.Reference(x);
              r.Reference(x);
          });
}
// Mixed output widths and quantization formats must preserve ordinary BF16
// Linear bit-for-bit, including fresh inputs on each graph replay.
static void SharedBfloat16(int rows, int columns, int count, bool forceMmvq) {
    context = "SharedBfloat16 rows=" + std::to_string(rows) + " count=" + std::to_string(count);
    Data x(BFLOAT16, {1, rows, columns}), bias;
    Allocate(x);
    std::vector<std::unique_ptr<Data>> weights, outputs, references;
    std::vector<Data *> w, o;
    const ggml_type types[] = {GGML_TYPE_Q5_K, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K};
    for (int i = 0; i < count; ++i) {
        const int n = 64 + i * 13;
        weights.emplace_back(new Data(DATA_GGUF_FORMAT, int(types[i % 3]), {n, columns}));
        std::vector<float> values(size_t(n) * columns);
        for (size_t j = 0; j < values.size(); ++j) values[j] = .15f * std::sin(float(j + 71 * i) * .19f);
        weights.back()->CreateFromOriData(WeightType::LINEAR, FLOAT32,
            reinterpret_cast<uint8_t *>(values.data()), nullptr, nullptr);
        weights.back()->ToDevice(DataDevice::CUDA, {0}, true);
        weights.back()->forceGGUFFp32Dequant = forceMmvq;
        outputs.emplace_back(new Data(BFLOAT16, {1, rows, n}));
        references.emplace_back(new Data(BFLOAT16, {1, rows, n}));
        Allocate(*outputs.back()); Allocate(*references.back());
        w.push_back(weights.back().get()); o.push_back(outputs.back().get());
    }
    Graph([&] { Check(FastllmCudaGGUFLinearShared(x, w.data(), o.data(), count), "BF16 group rejected"); },
        [&] {
            for (int i = 0; i < count; ++i)
                Same(*o[i], Download(*references[i], references[i]->Count(0)), "BF16 shared projection");
        }, [&](int seed) {
            std::vector<__nv_bfloat16> values(size_t(rows) * columns);
            for (size_t j = 0; j < values.size(); ++j)
                values[j] = __float2bfloat16_rn(.8f * std::sin(float(j + seed * 71) * .11f));
            Upload(x, values);
            for (int i = 0; i < count; ++i)
                Check(FastllmCudaBFloat16MatMulGGUF(x, *w[i], bias, *references[i],
                    rows, columns, w[i]->dims[0]), "BF16 reference rejected");
        });
    auto before = Download(*o[0], o[0]->Count(0));
    Data *saved = o[1]; o[1] = o[0];
    Check(!FastllmCudaGGUFLinearShared(x, w.data(), o.data(), count), "BF16 output alias accepted");
    o[1] = saved;
    w.back()->dataDeviceIds = {1};
    Check(!FastllmCudaGGUFLinearShared(x, w.data(), o.data(), count), "BF16 wrong device accepted");
    w.back()->dataDeviceIds = {0};
    Same(*o[0], before, "BF16 rejection modified output");
}
static void Permuted(ggml_type type, int t, int kh, int groups) {
    const int hd = 128, k = kh * groups * hd, n = 129;
    Data x(FLOAT16, {1, t, k}), ordered(FLOAT16, {1, t, k}), middle(FLOAT16, {1, t, n}), bias(FLOAT32);
    Allocate(x);
    Allocate(ordered);
    Allocate(middle);
    Projection p(type, t, k, n);
    Graph(
        [&] {
            Check(FastllmCudaGGUFLinearAddPermuted(x, p.w, bias, p.y, kh, kh * groups, hd),
                  "permute rejected");
        },
        [&] { p.Compare(); },
        [&](int seed) {
            const auto values = Input(t, k, seed * 29);
            auto shuffled = values;
            for (int row = 0; row < t; ++row)
                for (int j = 0; j < k; ++j) {
                    const int head = j / hd, source = ((head % kh) * groups + head / kh) * hd + j % hd;
                    shuffled[size_t(row) * k + j] = values[size_t(row) * k + source];
                }
            Upload(x, values);
            Upload(ordered, shuffled);
            auto residual = Input(t, n, seed + 9);
            Upload(p.y, residual);
            Upload(p.ref, residual);
            Check(FastllmCudaHalfMatMulGGUF(ordered, p.w, bias, middle, t, k, n),
                  "permuted reference rejected");
            FastllmCudaAddTo(p.ref, middle, 1);
        });
}
static void MergedGate(ggml_type type, int t, int k, int n) {
    context = "MergedGate type=" + std::to_string(type) + " T=" + std::to_string(t) +
              " K=" + std::to_string(k);
    Data x(FLOAT16, {1, t, k}), merged(DATA_GGUF_FORMAT, int(type), {2 * n, k});
    Allocate(x);
    Allocate(merged);
    merged.strides = {1};
    merged.forceGGUFFp32Dequant = true;
    Projection p(type, t, k, n), q(type, t, k, n);
    auto gate = Weights(type, n, k), up = gate;
    std::rotate(up.begin(), up.begin() + ggml_row_size(type, k), up.end());
    Upload(q.w, up);
    gate.insert(gate.end(), up.begin(), up.end());
    Upload(merged, gate);
    Graph([&] {
        Check(FastllmCudaHalfGgufMergedGateUpSiluMul(x, merged, p.y, t, k, n), "merged gate rejected");
    }, [&] { p.Compare(); }, [&](int seed) {
        Upload(x, Input(t, k, seed * 37));
        p.Reference(x);
        q.Reference(x);
        FastllmCudaSilu(p.ref, p.ref);
        FastllmCudaMulTo(p.ref, q.ref, 1.0f);
    });
}

static void MixedNorm(int t, int n, bool gguf) {
    context =
        "MixedNorm T=" + std::to_string(t) + " N=" + std::to_string(n) + " GGUF=" + std::to_string(gguf);
    const int k = 5120;
    Data x(FLOAT16, {1, t, k}), norm(FLOAT32, {k}), bias(FLOAT32);
    Data y(FLOAT16, {1, t, k}), o(FLOAT16, {1, t, n}), yr(FLOAT16, {1, t, k}), orr(FLOAT16, {1, t, n});
    std::unique_ptr<Data> holder(gguf ? new Data(DATA_GGUF_FORMAT, int(GGML_TYPE_BF16), {n, k})
                                      : new Data(BFLOAT16, {n, k}));
    Data &w = *holder;
    for (auto *d : {&x, &norm, &w, &y, &o, &yr, &orr})
        Allocate(*d);
    if (gguf)
        w.strides = {1};
    std::vector<float> norms(k);
    for (int j = 0; j < k; ++j)
        norms[j] = 1 + .1f * std::cos(float(j));
    Upload(norm, norms);
    std::vector<__nv_bfloat16> weights(n * k);
    for (size_t j = 0; j < weights.size(); ++j)
        weights[j] = __float2bfloat16_rn(.07f * std::sin(float(j) * .619f));
    Upload(w, weights);
    Graph([&] { Check(CudaRMSNormSmallLinearBlock(x, norm, w, bias, y, o, 1e-6f), "mixed norm fell back"); },
          [&] {
              Same(y, Download(yr, t * k), "mixed normalized");
              Same(o, Download(orr, t * n), "mixed projection");
          },
          [&](int seed) {
              Upload(x, Input(t, k, seed * 19));
              FastllmCudaRMSNorm(x, norm, yr, 1e-6f);
              if (gguf)
                  Check(FastllmCudaHalfMatMulGGUF(yr, w, bias, orr, t, k, n), "mixed reference rejected");
              else
                  Check(FastllmCudaHalfMatMulBFloat16(yr, w, bias, orr, t, k, n), "mixed reference rejected");
          });
}
static void Reject() {
    Data x(FLOAT16, {1, 256}), bias(FLOAT32);
    Allocate(x);
    Upload(x, Input(1, 256));
    Projection a(GGML_TYPE_IQ3_S, 1, 256, 129), b(GGML_TYPE_IQ2_S, 1, 256, 129);
    Data *w[] = {&a.w, &b.w}, *o[] = {&a.y, &b.y};
    auto before = Download(a.storage, 137);
    Check(!FastllmCudaGGUFLinearShared(x, w, o, 1), "single projection accepted");
    o[1] = o[0];
    Check(!FastllmCudaGGUFLinearShared(x, w, o, 2), "aliased outputs accepted");
    o[1] = &b.y;
    x.strides.back() = 2;
    Check(!FastllmCudaGGUFLinearShared(x, w, o, 2), "strided input accepted");
    x.strides.back() = 1;
    b.w.dataDeviceIds = {1};
    Check(!FastllmCudaGGUFLinearShared(x, w, o, 2), "wrong device accepted");
    b.w.dataDeviceIds = {0};
    b.w.ggmlType = GGML_TYPE_Q8_0;
    Check(!FastllmCudaGGUFLinearShared(x, w, o, 2), "unsupported type accepted");
    b.w.ggmlType = GGML_TYPE_IQ2_S;
    auto *tensor = static_cast<ggml_tensor *>(b.w.ggmlTensor);
    ++tensor->nb[1];
    Check(!FastllmCudaGGUFLinearShared(x, w, o, 2), "strided weight accepted");
    --tensor->nb[1];
    Check(!FastllmCudaGGUFLinearAddPermuted(x, a.w, bias, a.y, 3, 4, 64), "invalid grouping accepted");
    Check(!FastllmCudaGGUFLinearAddPermuted(x, a.w, bias, a.y, 1, 16, 16), "unaligned head accepted");
    Same(a.storage, before, "rejected calls modified output");
}
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            return 77;
        Cuda(cudaSetDevice(0));
        Reject();
        cudaDeviceProp properties{};
        Cuda(cudaGetDeviceProperties(&properties, 0));
        for (int rows = 1; rows <= 8; ++rows)
            for (int columns : {256, 4096})
                for (int count : {2, 6})
                    for (bool force : {false, true}) {
                        if (rows == 8 && !force && properties.major >= 10) continue;
                        SharedBfloat16(rows, columns, count, force);
                    }
        std::cout << "PASS BF16 shared Q8, 1..8 rows, 2/6 mixed projections" << std::endl;
        const ggml_type types[] = {GGML_TYPE_IQ3_S,  GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS,
                                   GGML_TYPE_Q4_K,   GGML_TYPE_Q2_K,    GGML_TYPE_IQ2_XXS,
                                   GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,   GGML_TYPE_IQ1_M};
        constexpr int nt = sizeof(types) / sizeof(types[0]);
        for (int t = 1; t <= 8; ++t)
            for (int i = 0; i < nt; ++i)
                Shared(types[i], types[(i + 3) % nt], types[(i + 5) % nt], t, 5120, 129);
        for (int i = 0; i < nt; ++i)
            Shared(types[i], types[(i + 1) % nt], types[(i + 2) % nt], 1, 256, 13);
        std::cout << "PASS shared Q8" << std::endl;
        for (int a = 0; a < nt; ++a)
            for (int b = 0; b < nt; ++b)
                if (a != b)
                    for (int t : {1, 3, 8})
                        Gate(types[a], types[b], t, 5120, 129);
        for (int t : {2, 4, 5, 6, 7})
            for (int a = 0; a < nt; ++a)
                Gate(types[a], types[(a + 1) % nt], t, 5120, 129);
        std::cout << "PASS mixed gate epilogue" << std::endl;
        for (auto type : types)
            for (int t = 1; t <= 8; ++t)
                for (int k : {768, 4096, 5120, 6144})
                    MergedGate(type, t, k, 129);
        std::cout << "PASS merged gate epilogue" << std::endl;
        for (auto type : types)
            if (type != GGML_TYPE_IQ2_XXS && type != GGML_TYPE_IQ1_M)
                for (int t = 1; t <= 8; ++t)
                    Permuted(type, t, 16, 3);
        for (auto type : types)
            if (type != GGML_TYPE_IQ2_XXS && type != GGML_TYPE_IQ1_M)
                Permuted(type, 1, 2, 2);
        std::cout << "PASS permuted Q8 residual" << std::endl;
        for (bool gguf : {false, true})
            for (int t = 1; t <= 8; ++t)
                for (int n : {1, 95, 96, 256})
                    MixedNorm(t, n, gguf);
        std::cout << "PASS mixed norm projection" << std::endl;
        std::cout << "PASS GGUF decode fusions cases=" << cases << " graph_cases=" << graphs << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
