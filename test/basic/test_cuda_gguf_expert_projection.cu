#include "cuda_gguf_data_test.cuh"
#include "devices/cuda/fastllm-cuda-gguf-projections.h"

// Exercise both halves of each packed pair, the four independent block scales,
// non-consecutive expert IDs and shared input rows.
static void Run(bool useOuter, int copies) {
    constexpr int experts = 3, rows = 5, cols = 192, routes = 6;
    const size_t stride = ggml_row_size(GGML_TYPE_NVFP4, cols);
    std::vector<uint8_t> packed(experts * rows * stride);
    std::vector<float> reference(experts * rows * cols), input(routes / copies * cols);
    const std::vector<int> ids{2, 0, 2, 1, 0, 1};
    const std::vector<float> outer{.015625f, .25f, .0625f};
    const float magnitudes[]{0, .5f, 1, 1.5f, 2, 3, 4, 6};
    for (int row = 0; row < experts * rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const int group = (col % 64) / 16, index = col % 16;
            auto *block = packed.data() + row * stride + (col / 64) * 36;
            const int exponent = (row + group + col / 64) % 11 - 5;
            const bool zeroScale = exponent == -5;
            block[group] = zeroScale ? 0x7f : (exponent + 7) * 8;
            const int code = (row + col * 7 + index / 8) % 16;
            block[4 + 8 * group + index % 8] |= code << (4 * (index / 8));
            reference[row * cols + col] = zeroScale ? 0.f : std::ldexp(magnitudes[code & 7] * ((code & 8) ? -1 : 1), exponent);
        }
    }
    for (size_t i = 0; i < input.size(); ++i) input[i] = float(int(i * 23 % 59) - 29) / 32;
    Data w(INT8, {int(packed.size())}), s(FLOAT32, {experts}), indices(INT32, {routes});
    Data x(FLOAT32, {routes / copies, cols}), y(FLOAT32, {routes, rows});
    for (Data *d : {&w, &s, &indices, &x, &y}) Allocate(*d);
    Upload(w, packed); Upload(s, outer); Upload(indices, ids); Upload(x, input);
    FastllmCudaGetGGUFExpertProjection(GGML_TYPE_NVFP4)(w.cudaData,
        useOuter ? static_cast<float *>(s.cudaData) : nullptr, static_cast<int *>(indices.cudaData),
        static_cast<float *>(x.cudaData), static_cast<float *>(y.cudaData), routes, copies, cols, rows, stride);
    Cuda(cudaDeviceSynchronize());
    std::vector<float> actual(routes * rows);
    Cuda(cudaMemcpy(actual.data(), y.cudaData, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (int r = 0; r < routes; ++r) for (int j = 0; j < rows; ++j) {
        float sum = 0;
        for (int i = 0; i < cols; ++i) sum += reference[(ids[r] * rows + j) * cols + i] * input[r / copies * cols + i];
        Check(actual[r * rows + j] == sum * (useOuter ? outer[ids[r]] : 1.f), "NVFP4 routed projection mismatch");
    }
}
// Compare routed MMQ with the original full-precision-activation dot. Reuse a
// captured gate/down pair with new route IDs, including an empty expert.
static void Grouped(ggml_type type, int rows) {
    constexpr int experts = 5, top = 3, hidden = 256, inter = 160;
    constexpr auto downType = GGML_TYPE_IQ4_NL;
    const int routes = rows * top;
    Check(FastllmCudaGGUFExpertMMQMaxRows(type, downType, hidden, inter, experts, top) >= rows,
          "Expected grouped expert MMQ for supported shape");
    const auto gateWeights = Weights(type, experts * 2 * inter, hidden);
    const auto downWeights = Weights(downType, experts * hidden, inter);
    const std::vector<float> scales{1.f, .25f, 1.75f, 1.25f, 1.5f};
    std::vector<float> x(rows * hidden), mid(routes * inter);
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = std::sin(.041f * i);
    for (size_t i = 0; i < mid.size(); ++i)
        mid[i] = std::cos(.037f * i);
    const auto bytes = FastllmCudaGGUFExpertMMQBytes(rows, hidden, inter, experts, top);
    Data gate(INT8, {int(gateWeights.size())}), down(INT8, {int(downWeights.size())});
    Data table(INT8, {int(2 * experts * sizeof(void *))}), outer(FLOAT32, {experts});
    Data ids(INT32, {routes}), input(FLOAT32, {rows, hidden}), middle(FLOAT32, {routes, inter});
    Data pair(FLOAT32, {routes, 2 * inter}), projected(FLOAT32, {routes, hidden}), ref(FLOAT32, {routes, 2 * inter});
    Data scratch(INT8, {int(bytes)});
    for (Data *d : {&gate, &down, &table, &outer, &ids, &input, &middle, &pair, &projected, &ref, &scratch})
        Allocate(*d);
    Upload(gate, gateWeights);
    Upload(down, downWeights);
    Upload(outer, scales);
    Upload(input, x);
    Upload(middle, mid);
    std::vector<const uint8_t *> ptrs(2 * experts);
    for (int e = 0; e < experts; ++e) {
        ptrs[2 * e] = static_cast<const uint8_t *>(gate.cudaData) + e * (gateWeights.size() / experts);
        ptrs[2 * e + 1] = static_cast<const uint8_t *>(down.cudaData) + e * (downWeights.size() / experts);
    }
    Upload(table, ptrs);
    auto *index = static_cast<int *>(ids.cudaData);
    auto mmq = [&]() {
        FastllmCudaGGUFExpertMMQ(true, type, table.cudaData, static_cast<float *>(outer.cudaData), index,
                                 static_cast<float *>(input.cudaData), static_cast<float *>(pair.cudaData),
                                 scratch.cudaData, rows, hidden, inter, experts, top);
        FastllmCudaGGUFExpertMMQ(false, downType, table.cudaData, nullptr, index, static_cast<float *>(middle.cudaData),
                                 static_cast<float *>(projected.cudaData), scratch.cudaData, rows, hidden, inter,
                                 experts, top);
    };
    std::vector<int> route(routes, 0);
    Upload(ids, route);
    mmq();
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    mmq();
    Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Cuda(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int iteration = 0; iteration < 2; ++iteration) {
        for (int r = 0; r < routes; ++r)
            route[r] = (iteration + r % top + r / top) % 4;
        Upload(ids, route);
        Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
        for (bool isGate : {true, false}) {
            const int cols = isGate ? hidden : inter, width = isGate ? 2 * inter : hidden;
            const auto t = isGate ? type : downType;
            FastllmCudaGetGGUFExpertProjection(t)(
                isGate ? gate.cudaData : down.cudaData, isGate ? static_cast<float *>(outer.cudaData) : nullptr,
                index, static_cast<float *>(isGate ? input.cudaData : middle.cudaData),
                static_cast<float *>(ref.cudaData), routes, isGate ? top : 1, cols, width, ggml_row_size(t, cols));
            std::vector<float> expected(size_t(routes) * width), actual(expected.size());
            Cuda(cudaMemcpy(expected.data(), ref.cudaData, expected.size() * sizeof(float), cudaMemcpyDeviceToHost));
            Cuda(cudaMemcpy(actual.data(), isGate ? pair.cudaData : projected.cudaData, actual.size() * sizeof(float),
                            cudaMemcpyDeviceToHost));
            double diff = 0, norm = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                diff += std::pow(double(expected[i]) - actual[i], 2);
                norm += double(expected[i]) * expected[i];
            }
            const double l2 = std::sqrt(diff / norm);
            Check(l2 < .015, "GGUF expert MMQ mismatch");
        }
    }
    Cuda(cudaGraphExecDestroy(exec));
    Cuda(cudaGraphDestroy(graph));
}
int main() {
    Run(false, 1);
    Run(true, 3);
    int device;
    cudaDeviceProp props;
    Cuda(cudaGetDevice(&device));
    Cuda(cudaGetDeviceProperties(&props, device));
    if (10 * props.major + props.minor >= 75) {
        for (auto type : {GGML_TYPE_Q3_K, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS,
                          GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL})
            Grouped(type, 37);
        Grouped(GGML_TYPE_IQ3_XXS, 1031);
    }
    std::cout << "PASS GGUF expert projection\n";
}
