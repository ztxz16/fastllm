#include "cuda_gguf_data_test.cuh"
#include "devices/cuda/fastllm-cuda-gguf-projections.h"
#include "devices/cuda/fastllm-cuda-activation.cuh"
#include "devices/cuda/fastllm-moe.h"

static void MakeNVFP4Weights(std::vector<uint8_t> &packed, std::vector<float> &reference, int rows, int cols) {
    const size_t stride = ggml_row_size(GGML_TYPE_NVFP4, cols);
    const float magnitudes[]{0, .5f, 1, 1.5f, 2, 3, 4, 6};
    for (int row = 0; row < rows; ++row) {
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
}

// Check gate/up (one input shared by several routes) and down (one input per route).
static void CheckRoutedGemv(ggml_type type, int cols, bool useOuter, int copies) {
    constexpr int experts = 3, rows = 5, routes = 6;
    const size_t stride = ggml_row_size(type, cols);
    std::vector<uint8_t> packed;
    std::vector<float> reference(experts * rows * cols), input(routes / copies * cols);
    if (type == GGML_TYPE_NVFP4) {
        packed.resize(experts * rows * stride);
        MakeNVFP4Weights(packed, reference, experts * rows, cols);
    } else {
        packed = Weights(type, experts * rows, cols);
        if (type == GGML_TYPE_MXFP4) {
            constexpr float values[]{0, .5f, 1, 1.5f, 2, 3, 4, 6, 0, -.5f, -1, -1.5f, -2, -3, -4, -6};
            for (size_t i = 0; i < reference.size(); ++i) {
                const uint8_t *block = packed.data() + (i / 32) * 17;
                const uint8_t byte = block[1 + i % 16];
                reference[i] = std::ldexp(values[i % 32 < 16 ? byte & 15 : byte >> 4], int(block[0]) - 127);
            }
        } else if (type == GGML_TYPE_BF16) {
            for (size_t i = 0; i < reference.size(); ++i) {
                uint16_t word;
                std::memcpy(&word, packed.data() + 2 * i, sizeof(word));
                uint32_t bits = uint32_t(word) << 16;
                std::memcpy(&reference[i], &bits, sizeof(bits));
            }
        } else {
            const auto unpack = ggml_type_to_float(type);
            Check(unpack != nullptr, "Missing GGUF reference decoder");
            for (int row = 0; row < experts * rows; ++row)
                unpack(packed.data() + row * stride, reference.data() + row * cols, cols);
        }
    }
    const std::vector<int> ids{2, -1, 2, 1, 0, -1};
    const std::vector<float> outer{.015625f, .25f, .0625f};
    for (size_t i = 0; i < input.size(); ++i) input[i] = float(int(i * 23 % 59) - 29) / 32;
    Data w(INT8, {int(packed.size())}), s(FLOAT32, {experts}), indices(INT32, {routes});
    Data x(FLOAT32, {routes / copies, cols}), y(FLOAT32, {routes, rows});
    for (Data *d : {&w, &s, &indices, &x, &y}) Allocate(*d);
    Upload(w, packed); Upload(s, outer); Upload(indices, ids); Upload(x, input);
    FastllmCudaGetGGUFExpertProjection(type)(w.cudaData,
        useOuter ? static_cast<float *>(s.cudaData) : nullptr, static_cast<int *>(indices.cudaData),
        static_cast<float *>(x.cudaData), static_cast<float *>(y.cudaData), routes, copies, cols, rows, stride);
    Cuda(cudaDeviceSynchronize());
    std::vector<float> actual(routes * rows);
    Cuda(cudaMemcpy(actual.data(), y.cudaData, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    // Interleave another projection's payload and permute expert-to-slot ownership.
    // The host branch owns two routes; its outputs must survive the GPU projection.
    const size_t payload = rows * stride, scaleOffset = (payload + 3) & ~size_t(3);
    const size_t pitch = (scaleOffset + sizeof(float) + 255) & ~size_t(255);
    std::vector<uint8_t> records(experts * pitch);
    std::vector<int> slots = ids;
    for (int e = 0; e < experts; ++e) {
        auto *record = records.data() + (experts - 1 - e) * pitch;
        std::memcpy(record, packed.data() + e * payload, payload);
        const float factor = useOuter ? outer[e] : 1.f;
        std::memcpy(record + scaleOffset, &factor, sizeof(factor));
    }
    for (int &slot : slots) if (slot >= 0) slot = experts - 1 - slot;
    Data cache(INT8, {int(records.size())}), slotIds(INT32, {routes});
    Allocate(cache); Upload(cache, records); Allocate(slotIds); Upload(slotIds, slots);
    std::vector<float> expectedSlots(routes * rows, -1234.f);
    Upload(y, expectedSlots);
    FastllmCudaGetGGUFSlotProjection(type)(static_cast<uint8_t *>(cache.cudaData), pitch,
        static_cast<int *>(slotIds.cudaData), static_cast<float *>(x.cudaData), static_cast<float *>(y.cudaData),
        routes, copies, cols, rows, stride, cudaStreamPerThread);
    Cuda(cudaStreamSynchronize(cudaStreamPerThread));
    for (int r = 0; r < routes; ++r) if (slots[r] >= 0)
        std::copy_n(actual.data() + r * rows, rows, expectedSlots.data() + r * rows);
    std::vector<float> slotResult(routes * rows);
    Cuda(cudaMemcpy(slotResult.data(), y.cudaData, slotResult.size() * sizeof(float), cudaMemcpyDeviceToHost));
    Check(slotResult == expectedSlots, "Masked GGUF slot projection mismatch");
    double diff = 0, norm = 0;
    for (int r = 0; r < routes; ++r) for (int j = 0; j < rows; ++j) {
        if (ids[r] < 0) {
            Check(actual[r * rows + j] == 0.f, "Nonlocal expert route must produce zero");
            continue;
        }
        double expected = 0;
        for (int i = 0; i < cols; ++i)
            expected += double(reference[(ids[r] * rows + j) * cols + i]) * input[r / copies * cols + i];
        if (useOuter) expected *= outer[ids[r]];
        const double error = actual[r * rows + j] - expected;
        if (type == GGML_TYPE_NVFP4) Check(error == 0, "NVFP4 routed projection mismatch");
        diff += error * error;
        norm += expected * expected;
    }
    if (!(norm > 0 && diff <= 1e-10 * norm))
        throw std::runtime_error(std::string("Routed GEMV mismatch for ") + ggml_type_name(type));
    if (copies > 1 && (type == GGML_TYPE_Q3_K || type == GGML_TYPE_IQ3_XXS)) {
        const int tokens = routes / copies;
        Data scratch(INT8, {int(FastllmCudaQ8ProjectionScratchBytes(tokens, cols))});
        Allocate(scratch);
        auto *weights = static_cast<const uint8_t *>(w.cudaData);
        auto *scales = static_cast<const float *>(s.cudaData);
        auto *routeIds = static_cast<const int *>(indices.cudaData);
        auto *inputs = static_cast<const float *>(x.cudaData);
        auto *outputs = static_cast<float *>(y.cudaData);
        Check(FastllmCudaGGUFSingleExpertProjection(type, weights, scales, routeIds, inputs, outputs,
                  scratch.cudaData, copies, cols, rows, stride, tokens), "Missing indexed GGUF projection");
        Cuda(cudaMemcpy(actual.data(), outputs, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
        Cuda(cudaMemset(outputs, 0xff, actual.size() * sizeof(float)));
        for (int t = 0; t < tokens; ++t)
            Check(FastllmCudaGGUFSingleExpertProjection(type, weights, scales, routeIds + t * copies,
                      inputs + t * cols, outputs + t * copies * rows, scratch.cudaData, copies, cols, rows, stride),
                  "Missing single-token GGUF projection");
        std::vector<float> single(actual.size());
        Cuda(cudaMemcpy(single.data(), outputs, single.size() * sizeof(float), cudaMemcpyDeviceToHost));
        Check(actual == single, "Indexed verifier projection differs from single-token decode");
        for (size_t i = 0; i < actual.size(); ++i)
            if (ids[i / rows] < 0) Check(actual[i] == 0.f, "Indexed nonlocal route must produce zero");
    }
}
// Materialize the original route-order intermediate to check the fused Q8 handoff.
__global__ void ReferenceSiTU(fastllm::cuda::RoutedSiTU<float> src, float *dst, int rows, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * width) dst[i] = src.get(i / width, i % width, width);
}
// Compare routed MMQ with the original full-precision-activation dot. Reuse a
// captured gate/down pair with new route IDs, including an empty expert.
static void Grouped(ggml_type type, int rows, ggml_type downType = GGML_TYPE_IQ4_NL, int inter = 160) {
    constexpr int experts = 5, top = 3, hidden = 256;
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
        if (iteration) std::fill_n(route.begin(), top + 1, -1);
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
            if (!isGate) {
                const auto groups = FastllmCudaGGUFExpertGroupsView(scratch.cudaData, rows, hidden, inter, experts, top);
                auto *sum = static_cast<float *>(ref.cudaData);
                const bool scaled = iteration != 0;
                FastllmCudaGGUFSumExpertGroups(groups, index, scaled ? static_cast<float *>(outer.cudaData) : nullptr,
                                               sum, rows, hidden, top);
                std::vector<float> reduced(size_t(rows) * hidden);
                Cuda(cudaMemcpy(reduced.data(), sum, reduced.size() * sizeof(float), cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < reduced.size(); ++i) {
                    float reference = 0.f;
                    for (int k = 0; k < top; ++k) {
                        const size_t r = (i / hidden) * top + k;
                        if (route[r] < 0) continue;
                        volatile float part = actual[r * hidden + i % hidden] * (scaled ? scales[route[r]] : 1.f);
                        reference += part;
                    }
                    Check(reduced[i] == reference, "Sorted down changed the route reduction");
                }
            }
            double diff = 0, norm = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                diff += std::pow(double(expected[i]) - actual[i], 2);
                norm += double(expected[i]) * expected[i];
            }
            const double l2 = std::sqrt(diff / norm);
            if (!(l2 < .015))
                throw std::runtime_error(std::string("Grouped ") + (isGate ? "gate " : "down ") +
                    ggml_type_name(t) + " relative L2 " + std::to_string(l2));
        }
    }
    if (downType == GGML_TYPE_Q2_0) {
        Data scores(FLOAT32, {routes});
        Allocate(scores);
        std::vector<float> routeScores(routes);
        for (int r = 0; r < routes; ++r) routeScores[r] = (r % 13) / 12.f;
        Upload(scores, routeScores);
        auto *score = static_cast<float *>(scores.cudaData);
        auto *scale = static_cast<float *>(outer.cudaData);
        const auto groups = FastllmCudaGGUFExpertGroupsView(scratch.cudaData, rows, hidden, inter, experts, top);
        std::vector<float> outputs[2];
        for (int fused = 0; fused < 2; ++fused) {
            FastllmCudaGGUFExpertMMQ(true, type, table.cudaData, scale, index, static_cast<float *>(input.cudaData),
                fused ? nullptr : static_cast<float *>(pair.cudaData), scratch.cudaData, rows, hidden, inter, experts, top);
            if (fused) {
                FastllmCudaGGUFExpertSiTUDown(downType, table.cudaData, index, score, scale, 4.f, 6.f,
                    scratch.cudaData, rows, hidden, inter, experts, top);
            } else {
                ReferenceSiTU<<<(routes * inter + 255) / 256, 256, 0, cudaStreamPerThread>>>(
                    {static_cast<float *>(pair.cudaData), score, 4.f, 6.f},
                    static_cast<float *>(middle.cudaData), routes, inter);
                FastllmCudaGGUFExpertMMQ(false, downType, table.cudaData, nullptr, index,
                    static_cast<float *>(middle.cudaData), nullptr, scratch.cudaData, rows, hidden, inter, experts, top);
            }
            FastllmCudaGGUFSumExpertGroups(groups, index, scale, static_cast<float *>(projected.cudaData), rows, hidden, top);
            outputs[fused].resize(size_t(rows) * hidden);
            Cuda(cudaMemcpy(outputs[fused].data(), projected.cudaData, outputs[fused].size() * sizeof(float), cudaMemcpyDeviceToHost));
        }
        Check(outputs[0] == outputs[1], "Sorted SiTU/Q8 down changed the expert output");
    }
    Cuda(cudaGraphExecDestroy(exec));
    Cuda(cudaGraphDestroy(graph));
}
int main() {
    CheckRoutedGemv(GGML_TYPE_NVFP4, 192, false, 1);
    CheckRoutedGemv(GGML_TYPE_NVFP4, 192, true, 3);
    CheckRoutedGemv(GGML_TYPE_Q3_K, 2304, true, 3);
    CheckRoutedGemv(GGML_TYPE_IQ3_XXS, 2304, true, 3);
    for (auto type : {GGML_TYPE_IQ1_S, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q2_0, GGML_TYPE_IQ4_XS,
                      GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0, GGML_TYPE_MXFP4, GGML_TYPE_BF16}) {
        CheckRoutedGemv(type, 2304, true, 3);
        // The real down dimension has a partial 256-column tile for short-block formats.
        CheckRoutedGemv(type, ggml_blck_size(type) < 256 ? 1152 : 2304, false, 1);
    }
    int device;
    cudaDeviceProp props;
    Cuda(cudaGetDevice(&device));
    Cuda(cudaGetDeviceProperties(&props, device));
    if (10 * props.major + props.minor >= 75) {
        for (auto type : {GGML_TYPE_Q2_0, GGML_TYPE_Q3_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
                          GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_Q8_0})
            Grouped(type, 37);
        Grouped(GGML_TYPE_IQ1_S, 37, GGML_TYPE_Q2_0, 1152);
        Grouped(GGML_TYPE_IQ1_S, 1031, GGML_TYPE_Q2_0, 192);
        Grouped(GGML_TYPE_IQ2_XXS, 37, GGML_TYPE_Q2_0, 192);
        Grouped(GGML_TYPE_Q8_0, 37, GGML_TYPE_Q8_0);
        Grouped(GGML_TYPE_Q5_K, 37, GGML_TYPE_Q5_0, 1152);
        Grouped(GGML_TYPE_Q4_K, 37, GGML_TYPE_Q5_0);
        Grouped(GGML_TYPE_IQ2_S, 37, GGML_TYPE_MXFP4, 1152);
        if (FastllmCudaRoutedBF16GemmSupported()) {
            Grouped(GGML_TYPE_BF16, 37, GGML_TYPE_BF16, 179);
            Grouped(GGML_TYPE_BF16, 37, GGML_TYPE_BF16, 1152);
        } else {
            Check(FastllmCudaGGUFExpertMMQMaxRows(GGML_TYPE_BF16, GGML_TYPE_BF16, 256, 160, 5, 3) == 0,
                  "BF16 Tensor Core unavailable: expected routed GEMV");
        }
        Grouped(GGML_TYPE_IQ3_XXS, 1031);
    }
    std::cout << "PASS GGUF expert projection\n";
}
