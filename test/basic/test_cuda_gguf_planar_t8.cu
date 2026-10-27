#include "cuda_gguf_fusion_test.cuh"
#include "fastllm-cuda-gguf-planar-t8.h"

static void Quantization() {
    constexpr int k = 5120, t = 8;
    Data x(FLOAT16, {t, k});
    Allocate(x);
    void *q = nullptr;
    Cuda(cudaMalloc(&q, FASTLLM_GGUF_PLANAR_T8_BYTES + 16));
    std::vector<uint8_t> bytes(FASTLLM_GGUF_PLANAR_T8_BYTES + 16);
    for (int seed = 0; seed < 3; ++seed) {
        auto input = Input(t, k, seed * 17);
        // Exact half-way rounding cases, signed values and zero groups.
        for (int j = 32; j < 64; ++j)
            input[j] = __float2half_rn((j == 32 ? 127.f : (j % 2 ? 1.5f : -1.5f)) / 256.f);
        Upload(x, input);
        Cuda(cudaMemset(q, 0xcd, bytes.size()));
        FastllmGgufQuantizePlanarT8(x.cudaData, q, cudaStreamPerThread);
        Cuda(cudaMemcpy(bytes.data(), q, bytes.size(), cudaMemcpyDeviceToHost));
        for (int b = 0; b < t * k / 32; ++b) {
            float vals[32], sums[32], a = 0;
            for (int l = 0; l < 32; ++l) {
                vals[l] = float(input[b * 32 + l]);
                sums[l] = vals[l];
                a = std::max(a, std::fabs(vals[l]));
            }
            for (int shift = 16; shift; shift >>= 1) {
                float next[32];
                for (int l = 0; l < 32; ++l)
                    next[l] = sums[l] + sums[l ^ shift];
                std::memcpy(sums, next, sizeof(sums));
            }
            const float d = a / 127;
            for (int l = 0; l < 32; ++l) {
                const int8_t want = a == 0 ? 0 : int8_t(std::round(vals[l] / d));
                Check(int8_t(bytes[b * 32 + l]) == want, "planar quantized byte differs from CPU oracle");
            }
            const half expected[2] = {__float2half_rn(d), __float2half_rn(sums[0])};
            Check(!std::memcmp(expected, bytes.data() + t * k + b * 4, 4),
                  "planar scales/sum differ from CPU oracle");
        }
        for (size_t i = FASTLLM_GGUF_PLANAR_T8_BYTES; i < bytes.size(); ++i)
            Check(bytes[i] == 0xcd, "quantization tail overwritten");
    }
    Cuda(cudaFree(q));
    ++cases;
}

static void Merged(ggml_type type, int n) {
    constexpr int k = 5120, t = 8;
    context = "merged type=" + std::to_string(type) + " N=" + std::to_string(n);
    Data x(FLOAT16, {1, t, k});
    Allocate(x);
    Projection w(type, t, k, 2 * n);
    Data storage(FLOAT16, {t * n + 8}), y(FLOAT16), reference(FLOAT16, {1, t, n});
    Allocate(storage);
    Allocate(reference);
    y.FakeFrom(storage, 0);
    y.Resize({1, t, n});
    y.dataDeviceIds = {0};
    Upload(storage, std::vector<half>(t * n + 8, __float2half_rn(42)));
    Graph(
        [&] { Check(FastllmCudaHalfGgufMergedGateUpSiluMul(x, w.w, y, t, k, n), "merged planar rejected"); },
        [&] {
            Same(y, Download(reference, t * n), "merged planar");
            const auto tail = Download(storage, t * n + 8);
            for (int i = t * n; i < t * n + 8; ++i)
                Check(float(tail[i]) == 42, "merged tail overwritten");
        },
        [&](int seed) {
            Upload(x, Input(t, k, seed * 89));
            w.Reference(x);
            FastllmCudaSwiglu(w.ref, reference);
        });
    const auto before = Download(storage, t * n + 8);
    x.strides.back() = 2;
    Check(!FastllmCudaHalfGgufMergedGateUpSiluMul(x, w.w, y, t, k, n), "strided merged input accepted");
    x.strides.back() = 1;
    auto *tensor = static_cast<ggml_tensor *>(w.w.ggmlTensor);
    ++tensor->nb[1];
    Check(!FastllmCudaHalfGgufMergedGateUpSiluMul(x, w.w, y, t, k, n), "strided merged weight accepted");
    --tensor->nb[1];
    Same(storage, before, "rejected merged calls wrote output");
}

static void RawProjection(ggml_type type) {
    constexpr int k = 5120, t = 8, n = 4097;
    context = "raw type=" + std::to_string(type);
    Data x(FLOAT16, {1, t, k});
    Allocate(x);
    Projection w(type, t, k, n);
    void *q = nullptr;
    Cuda(cudaMalloc(&q, FASTLLM_GGUF_PLANAR_T8_BYTES));
    Graph(
        [&] {
            FastllmGgufQuantizePlanarT8(x.cudaData, q, cudaStreamPerThread);
            Check(FastllmGgufProjectPlanarT8(type, 0, w.w.cudaData, nullptr, q, w.y.cudaData, n, n,
                                             cudaStreamPerThread),
                  "raw planar rejected");
        },
        [&] { w.Compare(); },
        [&](int seed) {
            Upload(x, Input(t, k, seed * 43));
            w.Reference(x);
        });
    Cuda(cudaFree(q));
}

int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            return 77;
        Cuda(cudaSetDevice(0));
        Quantization();
        const ggml_type types[] = {GGML_TYPE_IQ3_S,  GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS,
                                   GGML_TYPE_Q4_K,   GGML_TYPE_Q2_K,    GGML_TYPE_IQ2_XXS,
                                   GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,   GGML_TYPE_IQ1_M};
        for (int i = 0; i < 9; ++i) {
            RawProjection(types[i]);
            context = "mixed type=" + std::to_string(types[i]);
            Gate(types[(i + 1) % 9], types[i], 8, 5120, 4097);
            Merged(types[i], 4097);
            std::cout << "PASS planar type=" << ggml_type_name(types[i]) << std::endl;
        }
        for (auto type : {GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS})
            Merged(type, 17408);
        context = "mixed real shape";
        Gate(GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS, 8, 5120, 17408);
        Check(!FastllmGgufPlanarT8Supported(GGML_TYPE_Q8_0), "unexpected type support");
        std::cout << "PASS planar T8 cases=" << cases << " graph_cases=" << graphs << std::endl;
    } catch (const std::exception &e) {
        std::cerr << context << ": " << e.what() << std::endl;
        return 1;
    }
}
