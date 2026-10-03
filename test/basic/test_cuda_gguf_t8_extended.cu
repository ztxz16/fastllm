#include "cuda_gguf_t8_test.cuh"

static int cases = 0;
static int selectedLaunches = 0;
static double maxError = 0;
template <ggml_type Type, int K, typename Output, int StoreMode = 0> static void Test(int n) {
    constexpr int tokens = 8;
    const int inputStride = K / 32 + 4, outputStride = n + 17;
    std::vector<uint8_t> w(size_t(n) * ggml_row_size(Type, K));
    uint32_t random = 83291;
    auto next = [&]() {
        random = random * 1664525u + 1013904223u;
        return random;
    };
    for (auto &b : w)
        b = next() >> 24;
    for (size_t i = 0; i < w.size(); i += ggml_type_size(Type)) {
        const half d = __float2half_rn(float(1 + (i / ggml_type_size(Type)) % 4) / 16384.f);
        if constexpr (Type == GGML_TYPE_Q2_K)
            reinterpret_cast<block_q2_K *>(w.data() + i)->dm = __halves2half2(d, __float2half_rn(.00013f));
        else if constexpr (Type == GGML_TYPE_IQ2_S)
            reinterpret_cast<block_iq2_s *>(w.data() + i)->d = d;
        else if constexpr (Type == GGML_TYPE_IQ2_XS)
            reinterpret_cast<block_iq2_xs *>(w.data() + i)->d = d;
        else if constexpr (Type == GGML_TYPE_IQ2_XXS)
            reinterpret_cast<block_iq2_xxs *>(w.data() + i)->d = d;
        else if constexpr (Type == GGML_TYPE_IQ3_S)
            reinterpret_cast<block_iq3_s *>(w.data() + i)->d = d;
        else if constexpr (Type == GGML_TYPE_IQ3_XXS)
            reinterpret_cast<block_iq3_xxs *>(w.data() + i)->d = d;
    }
    std::vector<block_q8_1> x(size_t(tokens) * inputStride);
    for (int t = 0; t < tokens; ++t)
        for (int b = 0; b < inputStride; ++b) {
            auto &q = x[t * inputStride + b];
            q.ds = __halves2half2(__float2half_rn(.006f * (1 + b % 5)), __float2half_rn(123.f));
            for (auto &v : q.qs)
                v = (t == 7 || b == 0) ? 0 : int(next() >> 24) - 128;
        }
    std::vector<Output> initial(size_t(tokens) * outputStride + 8), old(initial.size()), now(initial.size());
    for (size_t i = 0; i < initial.size(); ++i)
        initial[i] = Output(.2f * std::sin(float(i) * .031f));
    Allocation dw(w.size()), dx(x.size() * sizeof(block_q8_1)), dy(initial.size() * sizeof(Output)),
        dr(initial.size() * sizeof(Output));
    Cuda(cudaMemcpy(dw.p, w.data(), w.size(), cudaMemcpyHostToDevice));
    Cuda(cudaMemcpy(dx.p, x.data(), x.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice));
    auto reset = [&] {
        Cuda(cudaMemcpy(dy.p, initial.data(), initial.size() * sizeof(Output), cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(dr.p, initial.data(), initial.size() * sizeof(Output), cudaMemcpyHostToDevice));
    };
    auto legacy = [&](Output *destination) {
        if constexpr (Type == GGML_TYPE_IQ3_S || Type == GGML_TYPE_IQ3_XXS) {
            if (n >= 4096) {
                fastllm_gguf_small_mmvq::IQ3PairSharedGemvKernel<Type, 8, 8, 1024, Output, StoreMode>
                    <<<(n + 15) / 16, 256, 0, cudaStreamPerThread>>>(dw.p, dx.as<block_q8_1>(), destination,
                                                                     K, n, inputStride, outputStride);
                return;
            }
        }
        fastllm_gguf_small_mmvq::LaunchRows<Type, 8, 16, Output, StoreMode>(
            dw.p, dx.as<block_q8_1>(), destination, K, n, inputStride * 32, outputStride,
            cudaStreamPerThread);
    };
    auto reference = [&] { legacy(dr.as<Output>()); };
    auto candidate = [&] {
        constexpr int outputKind = std::is_same<Output, half>::value            ? 1
                                   : std::is_same<Output, __nv_bfloat16>::value ? 2
                                                                                : 0;
        if (fastllm_gguf_small_mmvq::DispatchExtendedT8(Type, outputKind, StoreMode, dw.p,
                                                        dx.as<block_q8_1>(), dy.p, K, n, inputStride * 32,
                                                        outputStride, cudaStreamPerThread))
            ++selectedLaunches;
        else
            legacy(dy.as<Output>());
    };
    auto compare = [&] {
        Cuda(cudaMemcpy(old.data(), dr.p, old.size() * sizeof(Output), cudaMemcpyDeviceToHost));
        Cuda(cudaMemcpy(now.data(), dy.p, now.size() * sizeof(Output), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < now.size(); ++i)
            if (std::memcmp(&now[i], &old[i], sizeof(Output))) {
                std::cerr << "T8 mismatch type=" << Type << " K=" << K << " N=" << n << " store=" << StoreMode
                          << " index=" << i << " got=" << float(now[i]) << " old=" << float(old[i])
                          << std::endl;
                throw std::runtime_error("bitwise compatibility failed");
            }
        for (int t = 0; t < tokens; ++t)
            for (int i = n; i < outputStride; ++i)
                Check(
                    !std::memcmp(&now[t * outputStride + i], &initial[t * outputStride + i], sizeof(Output)),
                    "padding overwritten");
    };
    reset();
    reference();
    candidate();
    Cuda(cudaDeviceSynchronize());
    compare();
    // Graph replay with changed activations proves the captured route rereads operands.
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    candidate();
    Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Cuda(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int replay = 0; replay < 2; ++replay) {
        for (auto &q : x)
            for (auto &v : q.qs)
                v = v == -128 ? 127 : -v;
        Cuda(cudaMemcpy(dx.p, x.data(), x.size() * sizeof(block_q8_1), cudaMemcpyHostToDevice));
        reset();
        reference();
        Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
        Cuda(cudaDeviceSynchronize());
        compare();
    }
    Cuda(cudaGraphExecDestroy(exec));
    Cuda(cudaGraphDestroy(graph));
    if constexpr (StoreMode == 0) {
        // Independent GGUF CPU decoder and FP64 dot of represented quantized inputs.
        std::vector<float> decoded(K);
        for (int row : {0, 1, n / 2, n - 1}) {
            ggml_type_to_float(Type)(w.data() + size_t(row) * ggml_row_size(Type, K), decoded.data(), K);
            for (int t = 0; t < tokens; ++t) {
                double sum = 0, magnitude = 0;
                for (int j = 0; j < K; ++j) {
                    const auto &q = x[t * inputStride + j / 32];
                    double v = double(decoded[j]) * float(__low2half(q.ds)) * int(q.qs[j % 32]);
                    sum += v;
                    magnitude += std::fabs(v);
                }
                const double want = float(Output(float(sum))),
                             error = std::fabs(float(now[t * outputStride + row]) - want);
                maxError = std::max(maxError, error);
                const double castTolerance = std::is_same<Output, float>::value  ? 0
                                             : std::is_same<Output, half>::value ? .001
                                                                                 : .008;
                // IQ2/IQ3_XXS legacy dot arithmetic rounds integer scale terms;
                // retain that contract and bound its difference from exact decoded weights.
                constexpr bool integerRounding = Type == GGML_TYPE_IQ2_S || Type == GGML_TYPE_IQ2_XS ||
                                                 Type == GGML_TYPE_IQ2_XXS || Type == GGML_TYPE_IQ3_XXS;
                Check(error <= (integerRounding ? .002 : 2e-6) * magnitude + castTolerance * std::fabs(want) +
                                   1e-5,
                      "independent FP64 oracle failed");
            }
        }
    }
    ++cases;
}
template <ggml_type Type> static void Suite() {
    Test<Type, 5120, float>(4097);
    Test<Type, 5120, half>(4097);
    Test<Type, 5120, half, 1>(4097);
    Test<Type, 5120, half, 2>(4097);
    Test<Type, 5120, __nv_bfloat16>(4097);
    Test<Type, 5120, half>(6144);
    Test<Type, 5120, half>(17408);
    Test<Type, 6144, half, 1>(5120);
    Test<Type, 17408, half, 1>(5120);
}
int main() {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
            return 77;
        Cuda(cudaSetDevice(0));
        Suite<GGML_TYPE_IQ2_S>();
        Suite<GGML_TYPE_IQ2_XS>();
        Suite<GGML_TYPE_IQ2_XXS>();
        Suite<GGML_TYPE_IQ3_S>();
        Suite<GGML_TYPE_IQ3_XXS>();
        Suite<GGML_TYPE_Q2_K>();
        Check(selectedLaunches > 0, "no production extended launches");
        std::cout << "PASS T8 extended projections cases=" << cases << " max_oracle_abs=" << maxError
                  << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
