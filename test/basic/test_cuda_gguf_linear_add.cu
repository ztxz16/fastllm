#include "cuda_gguf_data_test.cuh"
#include "fastllm-cuda-gguf-linear-add.h"
#include <utility>

static double maxOracleError = 0;
static int cases = 0, graphCases = 0;
static void Test(ggml_type type, int tokens, int k, int n) {
    const size_t count = size_t(tokens) * n;
    Data weight(DATA_GGUF_FORMAT, int(type), {n, k});
    Data input(FLOAT16, {1, tokens, k}), middle(FLOAT16, {1, tokens, n});
    Data reference(FLOAT16, {1, tokens, n}), storage(FLOAT16, {int(count) + 16}), output;
    Data bias(FLOAT32);
    Allocate(weight);
    Allocate(input);
    Allocate(middle);
    Allocate(reference);
    Allocate(storage);
    // Match the loader's block metadata and its deliberately non-dense Data strides.
    weight.strides = {1};
    weight.forceGGUFFp32Dequant = true;
    output.FakeFrom(storage, 0);
    output.Resize({1, tokens, n});
    output.dataDeviceIds = {0};
    const auto packed = Weights(type, n, k);
    Upload(weight, packed);
    std::vector<half> x(size_t(tokens) * k), residual(count + 16);
    uint32_t random = 313;
    for (size_t i = 0; i < x.size(); ++i) {
        random = random * 1664525U + 1013904223U;
        x[i] = __float2half_rn(i % k < 32 || (tokens == 8 && i / size_t(k) == 7)
                                   ? 0.0f
                                   : float(int(random >> 24) - 128) / 256.0f);
    }
    for (size_t i = 0; i < count; ++i)
        residual[i] = __float2half_rn(.23f * std::cos(float(i) * .017f));
    for (size_t i = count; i < residual.size(); ++i)
        residual[i] = __float2half_rn(42.0f);
    Upload(input, x);
    Upload(storage, residual);
    Cuda(cudaMemcpy(reference.cudaData, residual.data(), count * sizeof(half), cudaMemcpyHostToDevice));
    // Compare against the unchanged, materialized FP16 boundary and separate AddTo.
    Check(FastllmCudaHalfMatMulGGUF(input, weight, bias, middle, tokens, k, n), "reference linear rejected");
    FastllmCudaAddTo(reference, middle, 1.0f);
    Check(FastllmCudaGGUFLinearAdd(input, weight, bias, output), "fused shape rejected");
    Cuda(cudaGetLastError());
    Cuda(cudaDeviceSynchronize());
    const auto expected = Download(reference, count);
    auto compare = [&]() {
        const auto actual = Download(storage, count + 16);
        if (std::memcmp(actual.data(), expected.data(), count * sizeof(half))) {
            for (size_t i = 0; i < count; ++i)
                if (std::memcmp(&actual[i], &expected[i], 2)) {
                    std::cerr << "mismatch " << ggml_type_name(type) << " T=" << tokens << " K=" << k
                              << " N=" << n << " at=" << i << " got=" << float(actual[i])
                              << " expected=" << float(expected[i]) << '\n';
                    break;
                }
            throw std::runtime_error("fused result is not bitwise equal to Linear+AddTo");
        }
        Check(!std::memcmp(actual.data() + count, residual.data() + count, 16 * sizeof(half)),
              "tail overwritten");
    };
    compare();
    // Exercise the CUDA operator entry, not only a raw kernel.
    Upload(storage, residual);
    Executor executor;
    executor.RunOnDevice(
        "cuda", "LinearAdd",
        {{"input", &input}, {"weight", &weight}, {"bias", &bias}, {"middle", &middle}, {"output", &output}},
        {}, {});
    Cuda(cudaDeviceSynchronize());
    compare();

    // Every token count is captured and replayed twice, with a restored residual.
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    Check(FastllmCudaGGUFLinearAdd(input, weight, bias, output), "graph capture rejected");
    Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Cuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int i = 0; i < 2; ++i) {
        Upload(storage, residual);
        Cuda(cudaGraphLaunch(executable, cudaStreamPerThread));
        Cuda(cudaDeviceSynchronize());
        compare();
    }
    Cuda(cudaGraphExecDestroy(executable));
    Cuda(cudaGraphDestroy(graph));
    ++graphCases;

    // Independent CPU GGUF decoder + FP64 dot against the represented Q8 activation.
    // Q4_K/Q2_K retain the original activation-sum correction, so this oracle uses
    // a quantization-aware bound in addition to the exact compatibility check above.
    std::vector<float> qx(x.size()), decoded(k);
    for (size_t b = 0; b < x.size(); b += 32) {
        float amax = 0;
        for (int i = 0; i < 32; ++i)
            amax = std::max(amax, std::fabs(float(x[b + i])));
        const float d = amax / 127.0f, stored = float(__float2half_rn(d));
        for (int i = 0; i < 32; ++i)
            qx[b + i] = amax == 0 ? 0 : std::round(float(x[b + i]) / d) * stored;
    }
    for (int row : {0, 1, n / 2, n - 1}) {
        ggml_type_to_float(type)(packed.data() + size_t(row) * ggml_row_size(type, k), decoded.data(), k);
        for (int t = 0; t < tokens; ++t) {
            double sum = 0, magnitude = 0;
            for (int j = 0; j < k; ++j) {
                const double v = double(decoded[j]) * qx[size_t(t) * k + j];
                sum += v;
                magnitude += std::fabs(v);
            }
            const size_t at = size_t(t) * n + row;
            const double want =
                float(__float2half_rn(float(residual[at]) + float(__float2half_rn(float(sum)))));
            const double error = std::fabs(double(float(expected[at])) - want);
            maxOracleError = std::max(maxOracleError, error);
            Check(error <= .002 * magnitude + .001 * std::fabs(want) + .00005,
                  "independent dequantized FP64 oracle failed");
        }
    }
    ++cases;
}

static void Reject() {
    Data x(FLOAT16, {9, 256}), w(DATA_GGUF_FORMAT, int(GGML_TYPE_IQ3_S), {128, 256});
    Data y(FLOAT16, {9, 128}), b(FLOAT32);
    Allocate(x);
    Allocate(w);
    Allocate(y);
    Cuda(cudaMemset(y.cudaData, 0x5a, y.Count(0) * 2));
    const auto before = Download(y, y.Count(0));
    Check(!FastllmCudaGGUFLinearAdd(x, w, b, y), "B9 admitted");
    x.Resize({1, 256});
    y.Resize({1, 128});
    b.Resize({128});
    Check(!FastllmCudaGGUFLinearAdd(x, w, b, y), "bias admitted");
    b.dims.clear();
    b.strides.clear();
    y.dataType = BFLOAT16;
    Check(!FastllmCudaGGUFLinearAdd(x, w, b, y), "BF16 residual admitted");
    y.dataType = FLOAT16;
    x.strides.back() = 2;
    Check(!FastllmCudaGGUFLinearAdd(x, w, b, y), "strided input admitted");
    x.strides.back() = 1;
    w.ggmlType = GGML_TYPE_Q8_0;
    Check(!FastllmCudaGGUFLinearAdd(x, w, b, y), "unsupported type admitted");
    w.ggmlType = GGML_TYPE_IQ3_S;
    auto *tensor = static_cast<ggml_tensor *>(w.ggmlTensor);
    const auto rowBytes = tensor->nb[1];
    ++tensor->nb[1];
    Check(!FastllmCudaGGUFLinearAdd(x, w, b, y), "strided GGUF blocks admitted");
    tensor->nb[1] = rowBytes;
    auto *saved = y.cudaData;
    y.cudaData = x.cudaData;
    Check(!FastllmCudaGGUFLinearAdd(x, w, b, y), "input/output overlap admitted");
    y.cudaData = saved;
    y.cudaData = w.cudaData;
    Check(!FastllmCudaGGUFLinearAdd(x, w, b, y), "weight/output overlap admitted");
    y.cudaData = saved;
    x.dataDeviceIds = {1};
    Check(!FastllmCudaGGUFLinearAdd(x, w, b, y), "wrong device admitted");
    x.dataDeviceIds = {0};
    const auto after = Download(y, before.size());
    Check(!std::memcmp(before.data(), after.data(), before.size() * 2), "rejected call modified residual");
}

int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            return 77;
        Cuda(cudaSetDevice(0));
        Reject();
        const std::pair<int, int> shapes[] = {{256, 13}, {5120, 129}, {6144, 5120}, {17408, 5121}};
        for (auto type : {GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K,
                          GGML_TYPE_Q2_K, GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS}) {
            for (auto shape : shapes)
                for (int tokens = 1; tokens <= 8; ++tokens)
                    Test(type, tokens, shape.first, shape.second);
            std::cout << "PASS " << ggml_type_name(type)
                      << " T=1..8, K/N=256/13,5120/129,6144/5120,17408/5121" << std::endl;
        }
        std::cout << "PASS GGUF LinearAdd cases=" << cases << " graph_cases=" << graphCases
                  << " max_oracle_abs_error=" << maxOracleError << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
