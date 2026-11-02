// Included by cudaGgufMmqRegression.cu. Uses the same binary fixture format,
// with model-width Q5_K matrices to exercise INT8 MMA small-row dispatch.
struct SmallQ5Graph {
    void *graph = nullptr, *exec = nullptr;
    std::vector<void *> reserved;

    template<class F> explicit SmallQ5Graph(F run) {
        auto check = [](bool ok) {
            if (!ok) throw std::runtime_error("Small Q5_K graph capture failed");
        };
        std::vector<size_t> allocations;
        { FastllmCudaAllocationRecorder record(&allocations); run(); }
        check(cudaStreamSynchronize(cudaStreamPerThread) == cudaSuccess);
        check(FastllmCudaReserveGraphWorkspace(allocations));
        check(FastllmCudaGraphMemoryPoolBegin() && FastllmCudaGraphBeginCapture());
        run();
        check(FastllmCudaGraphEndCapture(&graph));
        check(FastllmCudaGraphMemoryPoolEnd(reserved));
        check(FastllmCudaGraphInstantiate(graph, &exec));
    }
    void Launch() {
        if (!FastllmCudaGraphLaunch(exec)) throw std::runtime_error("Small Q5_K graph replay failed");
    }
    ~SmallQ5Graph() {
        cudaStreamSynchronize(cudaStreamPerThread);
        FastllmCudaGraphExecDestroy(exec);
        FastllmCudaGraphDestroy(graph);
        FastllmCudaGraphMemoryPoolRelease(reserved);
    }
};

int CheckSmallQ5(const char *path, int tokens) {
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
    if (major * 10 + minor < 75) return 77;
    const int ordinaryMinRows = major >= 12 ? 8 : 9;
    auto check = [](bool ok, const char *message) {
        if (!ok) throw std::runtime_error(message);
    };
    auto read = [&](const fastllm::Data &data) {
        std::vector<__nv_bfloat16> packed(data.Count(0));
        check(cudaMemcpy(packed.data(), data.cudaData, packed.size()*2,
                         cudaMemcpyDeviceToHost) == cudaSuccess, "Small Q5_K output copy failed");
        std::vector<float> result(packed.size());
        for (size_t i = 0; i < packed.size(); ++i) result[i] = __bfloat162float(packed[i]);
        return result;
    };
    std::ifstream file(path, std::ios::binary);
    file.exceptions(std::ios::failbit | std::ios::badbit);
    uint32_t cases;
    file.read(reinterpret_cast<char *>(&cases), sizeof(cases));
    int accepted = 0, rejected = 0;
    double worst = 0;
    for (uint32_t test = 0; test < cases; ++test) {
        uint32_t header[4];
        file.read(reinterpret_cast<char *>(header), sizeof(header));
        const int type = header[0], outputs = header[1], columns = header[2];
        const bool eligible = type == GGML_TYPE_Q5_K && tokens >= 4 && columns >= 1280 &&
            columns % 256 == 0 && outputs >= (tokens == 4 ? 5120 : 4096);
        if (!eligible) {
            // Blackwell already admits general MMQ at eight rows. Only
            // assert rejection below that pre-existing dispatch threshold.
            if (tokens < ordinaryMinRows)
                check(!FastllmCudaBFloat16MatMulGGUFMMQ(nullptr, nullptr, nullptr, type,
                    tokens, columns, outputs, cudaStreamPerThread), "Small MMQ accepted excluded shape/type");
            file.seekg(header[3], std::ios::cur);
            ++rejected;
            continue;
        }
        check(header[3] == size_t(outputs)*ggml_row_size(GGML_TYPE_Q5_K, columns), "Invalid Q5_K fixture");
        if (tokens < ordinaryMinRows) {
            check(!FastllmCudaHalfMatMulGGUFMMQ(nullptr, nullptr, nullptr, type, tokens, columns,
                outputs, cudaStreamPerThread), "Small MMQ changed FP16 dispatch");
            check(!FastllmCudaFloatMatMulGGUFMMQ(nullptr, nullptr, nullptr, type, tokens, columns,
                outputs, cudaStreamPerThread), "Small MMQ changed FP32 dispatch");
        }
        std::vector<char> packed(header[3]);
        file.read(packed.data(), packed.size());
        using namespace fastllm;
        Data weight(DATA_GGUF_FORMAT, type, {outputs, columns});
        weight.disableGGUFRepack = true;
        weight.Allocate();
        std::memcpy(weight.cpuData, packed.data(), packed.size());
        weight.ToDevice(CUDA, {0}, true);
        Data input(BFLOAT16, {tokens, columns}), reference(BFLOAT16, {tokens, outputs});
        Data actual(BFLOAT16, {tokens, outputs}), bias;
        for (Data *data : {&input, &reference, &actual}) { data->ToDevice(CUDA, {0}, true); data->Allocate(); }
        std::vector<__nv_bfloat16> values(size_t(tokens)*columns, __float2bfloat16(0.f));
        check(cudaMemcpy(input.cudaData, values.data(), values.size()*2, cudaMemcpyHostToDevice) == cudaSuccess,
              "Small Q5_K initial input copy failed");
        SmallQ5Graph baseline([&] {
            // Existing small-row safety flag selects the original MMVQ math.
            weight.forceGGUFFp32Dequant = true;
            check(FastllmCudaBFloat16MatMulGGUF(input, weight, bias, reference, tokens, columns, outputs),
                  "Q5_K MMVQ oracle failed");
            weight.forceGGUFFp32Dequant = false;
        });
        SmallQ5Graph candidate([&] {
            check(FastllmCudaBFloat16MatMulGGUFMMQ(input.cudaData, weight.cudaData, actual.cudaData,
                type, tokens, columns, outputs, cudaStreamPerThread), "Small Q5_K MMQ was not selected");
        });
        std::mt19937 rng(711 + columns + test);
        std::normal_distribution<float> normal(0.f, .2f);
        for (int pattern = 0; pattern < 3; ++pattern) {
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = __float2bfloat16(pattern == 0 ? normal(rng) : pattern == 1 ? (int(i%3)-1)*.125f : 0.f);
            check(cudaMemcpy(input.cudaData, values.data(), values.size()*2, cudaMemcpyHostToDevice) == cudaSuccess,
                  "Small Q5_K changed input copy failed");
            baseline.Launch(); candidate.Launch();
            auto a = read(actual), b = read(reference);
            double error = 0, norm = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                check(std::isfinite(a[i]) && std::isfinite(b[i]), "Nonfinite small Q5_K result");
                error += std::pow(a[i]-b[i], 2); norm += std::pow(b[i], 2);
            }
            const double relative = std::sqrt(error/std::max(norm, 1e-30));
            worst = std::max(worst, relative);
            check(relative <= .003163 && (pattern != 2 || error == 0), "Small Q5_K MMVQ relative L2 failed");
            // Independent CPU decoding/FP64 dots, sampled across every output
            // matrix and every input row. Include intentional Q8 activation error.
            double denseError = 0, denseNorm = 0, absoluteNorm = 0;
            std::vector<float> decoded(columns);
            for (int sample = 0; sample < 32; ++sample) {
                const int row = sample*(outputs-1)/31;
                Decode(type, packed.data()+size_t(row)*ggml_row_size(GGML_TYPE_Q5_K, columns), decoded.data(), columns);
                for (int token = 0; token < tokens; ++token) {
                    double expected = 0, absolute = 0;
                    for (int col = 0; col < columns; ++col) {
                        const double term = double(__bfloat162float(values[size_t(token)*columns+col]))*decoded[col];
                        expected += term; absolute += std::abs(term);
                    }
                    denseError += std::pow(a[size_t(token)*outputs+row]-expected, 2);
                    denseNorm += expected*expected; absoluteNorm += absolute*absolute;
                }
            }
            check(std::sqrt(denseError) <= .025*std::sqrt(denseNorm)+.00005*std::sqrt(absoluteNorm)+1e-6,
                  "Small Q5_K independent CPU reference failed");
        }
        ++accepted;
    }
    check(tokens < 4 || accepted > 0, "Small Q5_K regression needs model-width Q5_K fixtures");
    std::cout << "PASS small Q5_K rows=" << tokens << " accepted=" << accepted
              << " rejected=" << rejected << " worst_l2=" << worst << std::endl;
    return 0;
}
