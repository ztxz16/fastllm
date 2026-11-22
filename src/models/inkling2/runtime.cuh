#pragma once
#include "inkling2.h"
#include "devices/cuda/cudaworkspace.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace fastllm {
namespace inkling2 {
inline void check(cudaError_t status) {
    if (status == cudaErrorMemoryAllocation)
        throw CudaWorkspaceExhausted("Insufficient CUDA memory for the Inkling request");
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Inkling CUDA: ") + cudaGetErrorString(status));
}
// A captured single-stream graph can reuse a block after its last host
// tensor view goes out of scope. Keep every backing address until graph teardown.
class DecodeGraphScratch {
    struct Block { size_t bytes; std::shared_ptr<void> storage; };
    // Shapes replay serially on the decoder stream. They can own the same
    // backing blocks while recording independent, immutable allocation orders.
    using Blocks = std::vector<Block>;
    std::shared_ptr<Blocks> backing = std::make_shared<Blocks>();
    std::vector<size_t> order;
    size_t cursor = 0;
    bool sealed = false;
public:
    static thread_local DecodeGraphScratch *active;
    struct Scope {
        DecodeGraphScratch *previous;
        explicit Scope(DecodeGraphScratch &pool) : previous(active) { active = &pool; }
        ~Scope() { active = previous; }
        Scope(const Scope &) = delete;
    };
    void ShareStorage(const DecodeGraphScratch &other) { backing = other.backing; }
    std::shared_ptr<void> Acquire(size_t bytes);
    template <class Body> void Warmup(Body &&body) {
        Scope scope(*this);
        body();
        check(cudaStreamSynchronize(cudaStreamPerThread));
        sealed = true;
    }
    template <class Body> void Capture(cudaGraphExec_t &exec, Body &&body, size_t *nodes = nullptr) {
        cudaGraph_t graph = nullptr;
        check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        bool capturing = true;
        try {
            {
                Scope scope(*this);
                body();
            }
            auto status = cudaStreamEndCapture(cudaStreamPerThread, &graph);
            capturing = false;
            check(status);
            if (cursor != order.size()) throw std::runtime_error("Inkling graph scratch plan was not consumed");
            if (nodes) check(cudaGraphGetNodes(graph, nullptr, nodes));
            check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        } catch (...) {
            if (capturing) cudaStreamEndCapture(cudaStreamPerThread, &graph);
            if (graph) cudaGraphDestroy(graph);
            throw;
        }
        cudaGraphDestroy(graph);
        std::vector<size_t>().swap(order);
    }
    size_t Bytes() const {
        size_t bytes = 0;
        for (const auto &block : *backing) bytes += block.bytes;
        return bytes;
    }
};
template <typename T> struct Buffer {
    T *p = nullptr;
    std::shared_ptr<void> owner;
    Buffer() = default;
    explicit Buffer(size_t n) {
        if (!n)
            return;
        if (DecodeGraphScratch::active) {
            owner = DecodeGraphScratch::active->Acquire(n * sizeof(T));
            p = static_cast<T *>(owner.get());
            return;
        }
#if CUDART_VERSION >= 11020
        auto result = cudaMallocAsync(reinterpret_cast<void **>(&p), n * sizeof(T), cudaStreamPerThread);
        if (result == cudaSuccess) {
            owner = std::shared_ptr<void>(p, [](void *ptr) { cudaFreeAsync(ptr, cudaStreamPerThread); });
            return;
        }
        if (result != cudaErrorNotSupported)
            check(result);
        cudaGetLastError();
#endif
        check(cudaMalloc(reinterpret_cast<void **>(&p), n * sizeof(T)));
        owner = std::shared_ptr<void>(p, [](void *ptr) { cudaFree(ptr); });
    }
};
template <typename T> struct Rows : Buffer<T> {
    int rows = 0, cols = 0;
    Rows() = default;
    Rows(int r, int c) : Buffer<T>(size_t(r) * c), rows(r), cols(c) {}
    Rows slice(int first, int count) const {
        if (first < 0 || count < 0 || first > rows - count)
            throw std::runtime_error("Inkling row slice out of bounds");
        Rows v = *this;
        v.p += size_t(first) * cols;
        v.rows = count;
        return v;
    }
};
using Matrix = Rows<float>;
constexpr int threads = 128;
__device__ inline float blockSum(float v, float *scratch) {
    scratch[threadIdx.x] = v;
    __syncthreads();
    for (int n = blockDim.x / 2; n; n /= 2) {
        if (threadIdx.x < n)
            scratch[threadIdx.x] += scratch[threadIdx.x + n];
        __syncthreads();
    }
    float sum = scratch[0];
    __syncthreads();
    return sum;
}

class ExpertBackend {
public:
    virtual ~ExpertBackend() = default;
    virtual Matrix run(const Config &, const Matrix &input, const float *bias, bool decodeArithmetic = false) const = 0;
};
std::vector<std::unique_ptr<ExpertBackend>> loadExperts(const Config &, const Placement &,
                                                        const std::map<std::string, TensorRange> &, const WeightMap &);
} // namespace inkling2
} // namespace fastllm
