#include "devices/cuda/fastllm-mxfp4-ep.h"
#include <cuda_runtime_api.h>
#include <array>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
using Bytes = std::vector<uint8_t>;
using DeviceBuffer = std::unique_ptr<void, decltype(&cudaFree)>;

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

DeviceBuffer allocate(size_t bytes) {
    void *p = nullptr;
    check(cudaMalloc(&p, bytes));
    return DeviceBuffer(p, cudaFree);
}

Bytes expert(int n, int k, int seed) {
    const size_t values = size_t(n) * k / 2;
    Bytes bytes(values + size_t(n) * k / 32);
    for (size_t i = 0; i < values; ++i) bytes[i] = uint8_t(i * 73 + (i >> 7) + seed);
    for (size_t i = values; i < bytes.size(); ++i) bytes[i] = uint8_t(115 + seed + (i % 12));
    return bytes;
}
void routedRows(void *layer) {
    constexpr int hidden = 4096, experts = 256, top = 8;
    constexpr int maxRows = kFastllmMxfp4LocalRoutedMaxRows;
    std::vector<uint16_t> x(maxRows * hidden), expected(x.size()), actual(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = uint16_t(0x3c40 + (i * 271 + i / hidden * 13) % 192 + ((i % 3) ? 0 : 0x8000));
    std::vector<float> probabilities(maxRows * experts), bias(experts), scores(maxRows * top);
    std::vector<int32_t> ids(maxRows * top);
    auto in = allocate(x.size() * 2), out = allocate(x.size() * 2), baseline = allocate(x.size() * 2);
    auto probs = allocate(probabilities.size() * 4), correction = allocate(experts * 4);
    auto routes = allocate(ids.size() * 4), weights = allocate(scores.size() * 4);
    check(cudaMemcpy(in.get(), x.data(), x.size() * 2, cudaMemcpyHostToDevice));
    for (int e = 0; e < experts; ++e) bias[e] = (e % 3 - 1) * 0.001f;
    check(cudaMemcpy(correction.get(), bias.data(), experts * 4, cudaMemcpyHostToDevice));
    // Exercise unaligned BF16 views and replay one graph with changing routes.
    auto graphInput = allocate((x.size() + 1) * sizeof(uint16_t));
    auto graphOutput = allocate((x.size() + 2) * sizeof(uint16_t));
    auto *offsetInput = static_cast<uint16_t *>(graphInput.get()) + 1;
    auto *offsetOutput = static_cast<uint16_t *>(graphOutput.get()) + 1;
    check(cudaMemcpy(offsetInput, x.data(), x.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t replay = nullptr;
    std::vector<uint16_t> guarded(x.size() + 2);
    // Alternate shared, overlapping and disjoint routes while reusing the same
    // arena. A shorter replay must not consume the preceding call's rows.
    std::vector<int> rowCounts;
    for (int lo = 1, hi = maxRows; lo <= hi; ++lo, --hi) {
        rowCounts.push_back(hi);
        if (lo < hi) rowCounts.push_back(lo);
    }
    for (int shift : {0, 17, 29, 8, 1}) for (int rows : rowCounts) {
        std::fill(probabilities.begin(), probabilities.end(), 0.05f);
        for (int r = 0; r < rows; ++r) {
            float sum = 0.f;
            for (int t = 0; t < top; ++t) {
                int e = shift == 1 ? r * top + t : (251 + shift * r + t * 17) % experts;
                float v = (16 - t) / 32.f;
                probabilities[r * experts + e] = v;
                ids[r * top + t] = e;
                scores[r * top + t] = v;
                sum += v;
            }
            for (int t = 0; t < top; ++t) scores[r * top + t] /= sum;
        }
        check(cudaMemcpy(probs.get(), probabilities.data(), rows * experts * 4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(routes.get(), ids.data(), rows * top * 4, cudaMemcpyHostToDevice));
        check(cudaMemcpy(weights.get(), scores.data(), rows * top * 4, cudaMemcpyHostToDevice));
        for (int first = 0; first < rows; first += 8) {
            const int count = std::min(8, rows - first);
            if (!fastllm_mxfp4_ep_run(layer, static_cast<uint16_t *>(in.get()) + first * hidden,
                    static_cast<uint16_t *>(baseline.get()) + first * hidden,
                    static_cast<int32_t *>(routes.get()) + first * top, static_cast<float *>(weights.get()) + first * top,
                    count, count, nullptr)) throw std::runtime_error(fastllm_mxfp4_ep_error());
        }
        check(cudaMemsetAsync(out.get(), 0x5a, x.size() * 2, cudaStreamPerThread));
        if (!fastllm_mxfp4_tp8_run_routed(layer, in.get(), out.get(), static_cast<float *>(probs.get()),
                static_cast<float *>(correction.get()), rows))
            throw std::runtime_error(fastllm_mxfp4_ep_error());
        check(cudaStreamSynchronize(cudaStreamPerThread));
        check(cudaMemcpy(expected.data(), baseline.get(), rows * hidden * 2, cudaMemcpyDeviceToHost));
        check(cudaMemcpy(actual.data(), out.get(), x.size() * 2, cudaMemcpyDeviceToHost));
        for (int i = 0; i < rows * hidden; ++i) if (actual[i] != expected[i]) {
            std::fprintf(stderr, "rows=%d shift=%d element=%d expected=%04x actual=%04x\n",
                         rows,shift,i,expected[i],actual[i]);
            throw std::runtime_error("routed expert output differs from explicit TopK");
        }
        for (int i = rows * hidden; i < int(x.size()); ++i) if (actual[i] != 0x5a5a)
            throw std::runtime_error("routed expert output overwrote a padded row");
        if (rows == maxRows) {
            if (!replay) {
                check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
                if (!fastllm_mxfp4_tp8_run_routed(layer, offsetInput, offsetOutput,
                        static_cast<float *>(probs.get()), static_cast<float *>(correction.get()), rows))
                    throw std::runtime_error(fastllm_mxfp4_ep_error());
                check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
                check(cudaGraphInstantiateWithFlags(&replay, graph, 0));
            }
            check(cudaMemsetAsync(graphOutput.get(), 0x5a, guarded.size() * sizeof(uint16_t), cudaStreamPerThread));
            check(cudaGraphLaunch(replay, cudaStreamPerThread));
            check(cudaStreamSynchronize(cudaStreamPerThread));
            check(cudaMemcpy(guarded.data(), graphOutput.get(), guarded.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
            if (!std::equal(expected.begin(), expected.end(), guarded.begin() + 1) ||
                    guarded.front() != 0x5a5a || guarded.back() != 0x5a5a)
                throw std::runtime_error("graph replay differs from explicit TopK or overwrote an output guard");
        }
    }
    check(cudaGraphExecDestroy(replay));
    check(cudaGraphDestroy(graph));
}

}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    try {
        size_t workspaceBytes = fastllm_mxfp4_tp8_workspace_bytes(0);
        if (!workspaceBytes) return 77;
        constexpr int experts = 256;
        std::array<Bytes, 4> reference{
            expert(512, 4096, 0), expert(512, 4096, 1),
            expert(4096, 256, 2), expert(4096, 256, 3)};
        std::vector<DeviceBuffer> inputs;
        for (const auto &bytes : reference) {
            inputs.push_back(allocate(bytes.size()));
            check(cudaMemcpy(inputs.back().get(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
        }
        std::array<const void *, experts> gate, down;
        for (int e = 0; e < experts; ++e) {
            gate[e] = inputs[e % 2].get();
            down[e] = inputs[2 + e % 2].get();
        }
        std::array<int, 2 * experts> shifts;
        // Reject an unsupported scale before consuming any source.
        auto *firstScale = static_cast<uint8_t *>(inputs.front().get()) + 512 * 4096 / 2;
        check(cudaMemset(firstScale, 0, 1));
        if (fastllm_mxfp4_tp8_validate(gate.data(), down.data(), shifts.data()) != 0)
            throw std::runtime_error("accepted invalid scale");
        check(cudaMemcpy(firstScale, &reference.front()[512 * 4096 / 2], 1, cudaMemcpyHostToDevice));
        if (fastllm_mxfp4_tp8_validate(gate.data(), down.data(), shifts.data()) != 1)
            throw std::runtime_error(fastllm_mxfp4_ep_error());

        auto workspace = allocate(workspaceBytes);
        check(cudaMemset(workspace.get(), 0, workspaceBytes));
        auto layer = std::unique_ptr<void, decltype(&fastllm_mxfp4_ep_destroy)>(
            fastllm_mxfp4_tp8_create(gate.data(), down.data(), shifts.data(), workspace.get(), workspaceBytes),
            fastllm_mxfp4_ep_destroy);
        if (!layer) throw std::runtime_error(fastllm_mxfp4_ep_error());
        routedRows(layer.get());
        inputs.clear();
        auto staging = allocate(experts * (reference[0].size() + reference[2].size()));
        if (!fastllm_mxfp4_tp8_restore(layer.get(), staging.get()))
            throw std::runtime_error(fastllm_mxfp4_ep_error());
        check(cudaStreamSynchronize(cudaStreamPerThread));
        auto *source = static_cast<const uint8_t *>(staging.get());
        for (int leg = 0; leg < 2; ++leg) {
            Bytes actual(reference[2 * leg].size());
            for (int e = 0; e < experts; ++e) {
                check(cudaMemcpy(actual.data(), source, actual.size(), cudaMemcpyDeviceToHost));
                if (actual != reference[2 * leg + e % 2])
                    throw std::runtime_error("restored expert differs from the canonical bytes");
                source += actual.size();
            }
        }
        std::puts("TP8 PACKED STAGE PASS");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
