#pragma once

#ifdef USE_CUDA
#include "fastllm.h"
#include "utils/utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-token-constraint.h"
#include "utils/decode_graph.h"
#include <cstring>
#include <map>
#include <random>
#include <tuple>

namespace fastllm {

// Reusable across models/SMs: startup-only sampling graphs, request-owned RNG,
// and two result slots. No weights, device allocation or capture on Submit().
// The owner serializes submission and drains before releasing logits/storage.
class CudaDecodeSampling {
public:
    struct Request {
        uint64_t identity, step;
        float temperature, topP;
        int topK;
        std::shared_ptr<const TokenConstraintProgram> constraint;
        // Used on admission/reorder/rewind. Consecutive steps advance their
        // private GPU cursor, independently of CPU publication/lookahead.
        int constraintState = 0;
        bool operator==(const Request &r) const {
            return identity == r.identity && step == r.step && temperature == r.temperature &&
                   topP == r.topP && topK == r.topK && constraint == r.constraint;
        }
    };
    CudaDecodeSampling() = default;
    CudaDecodeSampling(const CudaDecodeSampling &) = delete;
    CudaDecodeSampling &operator=(const CudaDecodeSampling &) = delete;
    ~CudaDecodeSampling() {
        if (device < 0) return;
        FastllmCudaSetDevice(device);
        Drain();
        for (auto &entry : graphs) FastllmCudaGraphExecDestroy(entry.second);
        for (int slot = 0; slot < 2; ++slot) {
            if (ready[slot]) FastllmCudaEventDestroy(ready[slot]);
            if (hostReady[slot]) FastllmCudaEventDestroy(hostReady[slot]);
            if (hostTokens[slot]) FastllmCudaHostFree(hostTokens[slot]);
        }
        if (hostParams) FastllmCudaHostFree(hostParams);
    }
    // A sampling-only owner may consume its logits in place. Softmax finishes
    // all reduction reads before writing probabilities; the next model must
    // wait for Ready before overwriting this storage. No second vocab buffer.
    bool Prepare(Data &logits, const std::vector<int> &shapes, uint64_t salt = 0,
                 bool consumeLogits = false) {
        if (device >= 0 || shapes.empty() || logits.dataDevice != DataDevice::CUDA ||
            logits.dataType != FLOAT32 || logits.dataDeviceIds.size() != 1 || !logits.cudaData ||
            logits.dims.size() != 2 || logits.dims[1] <= 0 ||
            shapes.front() <= 0 || shapes.back() > logits.dims[0] ||
            !std::is_sorted(shapes.begin(), shapes.end()) ||
            std::adjacent_find(shapes.begin(), shapes.end()) != shapes.end()) return false;
        device = logits.dataDeviceIds.at(0);
        FastllmCudaSetDevice(device);
        maxRows = shapes.back(); vocab = logits.dims.back(); buckets = shapes;
        cudaLogits = (float *)logits.cudaData;
        nonce = salt ? salt : ((uint64_t)std::random_device{}() << 32) ^ std::random_device{}();
        auto allocate = [&](Data &data, DataType type, std::vector<int> dims) {
            data.dataType = type; data.Resize(dims);
            data.ToDevice(DataDevice::CUDA, {device}, false); data.Allocate(true);
        };
        if (!consumeLogits) allocate(probs, FLOAT32, {maxRows, vocab});
        cudaProbs = consumeLogits ? cudaLogits : (float *)probs.cudaData;
        allocate(params, INT32, {8, maxRows});
        allocate(tokens, INT32, {2, maxRows});
        allocate(floatTokens, FLOAT32, {2, maxRows});
        constraints = FastllmCudaCreateTokenConstraintBatch(maxRows, 1);
        FastllmCudaPrepareTokenConstraintBatch(constraints.get(),
            std::vector<std::shared_ptr<const TokenConstraintProgram>>(maxRows), std::vector<int>(maxRows), vocab);
        hostParams = FastllmCudaHostMalloc(params.GetBytes());
        for (int slot = 0; slot < 2; ++slot) {
            hostTokens[slot] = (int *)FastllmCudaHostMalloc(maxRows * sizeof(int));
            ready[slot] = FastllmCudaEventCreate();
            hostReady[slot] = FastllmCudaEventCreate();
        }
        if (!hostParams || !hostTokens[0] || !hostTokens[1] || !ready[0] || !ready[1] ||
            !hostReady[0] || !hostReady[1]) return false;
        std::vector<Request> initial(maxRows, {0, 0, 1.0f, 1.0f, vocab});
        Upload(initial);
        // Warm every shape before capturing startup-only sampling graphs.
        for (bool constrained : {false, true}) for (int rows : buckets) {
            if (!Run(rows, 0, constrained)) {
                FastllmCudaClearLastError();
                FastllmCudaSyncCurrentThreadStream();
                return false;
            }
        }
        FastllmCudaSyncCurrentThreadStream();
        for (bool constrained : {false, true}) for (int rows : buckets) for (int slot = 0; slot < 2; ++slot) {
            void *graph = nullptr, *exec = nullptr;
            if (!FastllmCudaGraphPrepareCaptureDevice() || !FastllmCudaGraphBeginCapture()) {
                FastllmCudaClearLastError();
                continue;
            }
            bool ok = Run(rows, slot, constrained);
            ok = FastllmCudaGraphEndCapture(&graph) && ok;
            if (ok) ok = FastllmCudaGraphInstantiate(graph, &exec);
            if (graph) FastllmCudaGraphDestroy(graph);
            if (ok) graphs[{rows, slot, constrained}] = exec;
            else if (exec) FastllmCudaGraphExecDestroy(exec);
            FastllmCudaClearLastError(); // Optional capture failures must not poison the fallback.
        }
        expected.clear(); // Warmup RNG is never used for a real request.
        return std::any_of(buckets.begin(), buckets.end(), [this](int rows) { return Supports(rows); });
    }
    bool Supports(int rows, bool constrained = false) const {
        auto shape = DecodeGraphBatch::Select(rows, buckets);
        return shape && graphs.count({shape.graphRows, 0, constrained}) &&
            graphs.count({shape.graphRows, 1, constrained});
    }
    // A false result has submitted no work. Runtime misses never capture.
    bool Submit(const std::vector<Request> &requests, int slot) {
        const bool constrained = std::any_of(requests.begin(), requests.end(),
            [](const Request &r) { return bool(r.constraint); });
        if (slot < 0 || slot > 1 || queued[slot] || !Supports(requests.size(), constrained)) return false;
        FastllmCudaSetDevice(device);
        if (requests != expected) {
            // Rare rebind (admission, reorder, changed params or fallback): the
            // single pinned staging buffer cannot be overwritten while in use.
            Drain();
            Upload(requests);
        }
        const int rows = DecodeGraphBatch::Select(requests.size(), buckets).graphRows;
        AssertInFastLLM(FastllmCudaGraphLaunch(graphs.at({rows, slot, constrained})), "Decode sampling graph launch failed.");
        // The next model depends on the GPU result, not its host readback.
        FastllmCudaEventRecordCurrentThread(ready[slot]);
        AssertInFastLLM(FastllmCudaCopyFromDeviceToHostAsyncCurrentThread(hostTokens[slot],
            (int *)tokens.cudaData + slot * maxRows, requests.size() * sizeof(int)),
            "Decode token readback failed.");
        FastllmCudaEventRecordCurrentThread(hostReady[slot]);
        queued[slot] = true; counts[slot] = requests.size();
        expected = requests;
        for (auto &request : expected) ++request.step;
        return true;
    }
    std::vector<int> Collect(int slot) {
        AssertInFastLLM(slot >= 0 && slot < 2 && queued[slot], "Invalid decode result ticket.");
        FastllmCudaSetDevice(device);
        FastllmCudaEventSynchronize(hostReady[slot]);
        queued[slot] = false;
        for (int row = 0; row < counts[slot]; ++row)
            AssertInFastLLM(hostTokens[slot][row] >= 0 && hostTokens[slot][row] < vocab,
                           "Decode sampling produced an invalid token.");
        return {hostTokens[slot], hostTokens[slot] + counts[slot]};
    }
    void Drain() {
        for (int slot = 0; slot < 2; ++slot)
            if (queued[slot]) FastllmCudaEventSynchronize(hostReady[slot]);
    }
    // Owner-only cancellation/error boundary; no consumer retains these tickets.
    void Discard() {
        Drain();
        queued[0] = queued[1] = false;
    }
    void *Ready(int slot) const { return ready[slot]; }
    const float *DeviceTokens(int slot) const { return (float *)floatTokens.cudaData + slot * maxRows; }
    int Capacity() const { return maxRows; }
private:
    static uint64_t Mix(uint64_t x) {
        x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
        return x ^ (x >> 31);
    }
    void Upload(const std::vector<Request> &requests) {
        auto *p = (uint32_t *)hostParams;
        std::memset(p, 0, params.GetBytes());
        for (int row = 0; row < maxRows; ++row) {
            Request r = row < (int)requests.size() ? requests[row] : Request{0, 0, 1, 1, 1};
            ((float *)p)[row] = r.temperature;
            ((int *)p)[maxRows + row] = r.topK;
            ((float *)p)[2 * maxRows + row] = r.topP;
            ((uint64_t *)(p + 4 * maxRows))[row] = Mix(nonce + r.identity);
            ((uint64_t *)(p + 6 * maxRows))[row] = r.step << 32;
        }
        if (std::any_of(requests.begin(), requests.end(), [](const Request &r) { return bool(r.constraint); })) {
            std::vector<std::shared_ptr<const TokenConstraintProgram>> programs(maxRows);
            std::vector<int> states(maxRows, 0);
            for (size_t row = 0; row < requests.size(); ++row) {
                programs[row] = requests[row].constraint; states[row] = requests[row].constraintState;
            }
            FastllmCudaPrepareTokenConstraintBatch(constraints.get(), programs, states, vocab);
        }
        AssertInFastLLM(FastllmCudaCopyFromPinnedHostToDeviceAsyncCurrentThread(
            params.cudaData, hostParams, params.GetBytes()), "Decode sampling parameters upload failed.");
    }
    bool Run(int rows, int slot, bool constrained) {
        auto *p = (uint32_t *)params.cudaData;
        if (constrained) FastllmCudaTokenConstraintApplyRows(constraints.get(), cudaLogits, rows);
        const bool ok = FastllmCudaTopKTopPSamplingToDeviceWithRng(cudaLogits, cudaProbs,
            (float *)p, (int *)p + maxRows, (float *)p + 2 * maxRows,
            nullptr, nullptr, 0, (int *)tokens.cudaData + slot * maxRows,
            (float *)floatTokens.cudaData + slot * maxRows, rows, vocab,
            (uint64_t *)(p + 4 * maxRows), (uint64_t *)(p + 6 * maxRows));
        if (ok && constrained) FastllmCudaTokenConstraintAdvanceInPlace(constraints.get(),
            (int *)tokens.cudaData + slot * maxRows, (float *)floatTokens.cudaData + slot * maxRows, rows);
        return ok;
    }
    int device = -1, maxRows = 0, vocab = 0;
    uint64_t nonce = 0;
    float *cudaLogits = nullptr, *cudaProbs = nullptr;
    Data probs, params, tokens, floatTokens;
    std::shared_ptr<void> constraints;
    void *hostParams = nullptr, *ready[2]{}, *hostReady[2]{};
    int *hostTokens[2]{};
    bool queued[2]{};
    int counts[2]{};
    std::vector<int> buckets;
    std::vector<Request> expected;
    std::map<std::tuple<int, int, bool>, void *> graphs;
};
}
#endif
