#pragma once
#include "inkling2.h"
#include "devices/cuda/fastllm-hybrid-moe-graph.cuh"
#include "devices/cuda/fastllm-prefill-workspace.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdexcept>
#include <type_traits>
#include <functional>
#include <exception>

namespace fastllm {
namespace inkling2 {
inline void check(cudaError_t status) {
    if (status == cudaErrorMemoryAllocation)
        throw CudaWorkspaceExhausted("Insufficient CUDA memory for the Inkling request");
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Inkling CUDA: ") + cudaGetErrorString(status));
}
// Captured branches join before their tensor views go out of scope. Reuse the
// block only after that join and keep its backing address until graph teardown.
class DecodeGraphScratch {
    struct Block { size_t bytes; std::shared_ptr<void> storage; };
    // Shapes replay serially on the decoder stream. They can own the same
    // backing blocks while recording independent, immutable allocation orders.
    using Blocks = std::vector<Block>;
    std::shared_ptr<Blocks> backing = std::make_shared<Blocks>();
    std::vector<size_t> order;
    size_t cursor = 0;
    bool sealed = false;
    struct HostTask {
        std::function<void()> run;
        std::exception_ptr failure;
        static void CUDART_CB invoke(void *data) noexcept {
            auto &task = *static_cast<HostTask *>(data);
            try { task.run(); }
            catch (...) { task.failure = std::current_exception(); }
        }
    };
    std::vector<std::unique_ptr<HostTask>> hostTasks;
public:
    std::unique_ptr<cuda::HybridMoeGraph> hybrid;
    static thread_local DecodeGraphScratch *active;
    struct Scope {
        DecodeGraphScratch *previous;
        explicit Scope(DecodeGraphScratch &pool) : previous(active) { active = &pool; }
        ~Scope() { active = previous; }
        Scope(const Scope &) = delete;
    };
    bool recording() const { return sealed; }
    void hostNode(std::function<void()> fn, cudaStream_t stream = cudaStreamPerThread) {
        auto task = std::make_unique<HostTask>();
        task->run = std::move(fn);
        hostTasks.push_back(std::move(task));
        check(cudaLaunchHostFunc(stream, HostTask::invoke, hostTasks.back().get()));
    }
    void finishHostTasks() {
        if (hostTasks.empty()) return;
        check(cudaStreamSynchronize(cudaStreamPerThread));
        std::exception_ptr first;
        for (auto &task : hostTasks) {
            if (!first) first = task->failure;
            task->failure = nullptr;
        }
        if (first) std::rethrow_exception(first);
    }
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
        cudaGraphExec_t candidate = nullptr;
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
            check(cudaGraphInstantiate(&candidate, graph, nullptr, nullptr, 0));
            if (hybrid) hybrid->seal(graph);
        } catch (...) {
            if (capturing) cudaStreamEndCapture(cudaStreamPerThread, &graph);
            if (candidate) cudaGraphExecDestroy(candidate);
            if (graph) cudaGraphDestroy(graph);
            throw;
        }
        exec = candidate;
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
    explicit Buffer(size_t n, const std::shared_ptr<CudaWorkspace> &workspace = {}) {
        if (!n)
            return;
        if (DecodeGraphScratch::active) {
            owner = DecodeGraphScratch::active->Acquire(n * sizeof(T));
            p = static_cast<T *>(owner.get());
            return;
        }
        if (workspace) {
            p = static_cast<T *>(workspace->Allocate(n * sizeof(T)));
            if (p) {
                owner = std::shared_ptr<void>(p, [workspace](void *ptr) {
                    workspace->FreeAfterStream(ptr, cudaStreamPerThread);
                });
                return;
            }
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
    Rows(int r, int c, const std::shared_ptr<CudaWorkspace> &arena = {})
        : Buffer<T>(size_t(r) * c, arena), rows(r), cols(c) {}
    static Rows temporary(int r, int c) { return Rows(r, c, CurrentPrefillWorkspace()); }
    Rows slice(int first, int count) const {
        if (first < 0 || count < 0 || first > rows - count)
            throw std::runtime_error("Inkling row slice out of bounds");
        Rows v = *this;
        v.p += size_t(first) * cols;
        v.rows = count;
        return v;
    }
};
template <typename T, typename U>
__global__ void castProjection(const T *input, U *output, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) output[i] = U(input[i]);
}

template <typename U, typename T> Rows<U> castRows(const Rows<T> &x) {
    if constexpr (std::is_same<T, U>::value) return x;
    Rows<U> out(x.rows, x.cols);
    const size_t n = size_t(x.rows) * x.cols;
    castProjection<<<(n + 255) / 256, 256, 0, cudaStreamPerThread>>>(x.p, out.p, n);
    return out;
}

using Matrix = Rows<float>;
struct ImageSpan { int start; Matrix features; };
std::vector<ImageSpan> encodeImages(const ImageInputs &, const Config &, WeightMap &, int gpu,
                                    DataType precision, int promptLength);
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

enum class ExpertPhase { Prefill, Decode, Verify };

class ExpertBackend {
public:
    virtual ~ExpertBackend() = default;
    virtual Matrix run(const Config &, const Matrix &input, const float *bias, ExpertPhase phase,
                       Rows<__nv_bfloat16> *unreduced = nullptr) const = 0;
};

// A decode step spends one admission budget across all host layers and repeated visits.
class ExpertCacheCycle {
public:
    virtual ~ExpertCacheCycle() = default;
    virtual std::shared_ptr<CudaWorkspace> borrowPrefillWorkspace(size_t bytes, int keep) = 0;
    virtual void beginStep() = 0;
    virtual void endStep() = 0;
};
struct ExpertLayers {
    std::unique_ptr<ExpertCacheCycle> cache;
    std::vector<std::unique_ptr<ExpertBackend>> layers;
};
ExpertLayers loadExperts(const Config &, const Placement &, const std::map<std::string, TensorRange> &,
                         const WeightMap &);
} // namespace inkling2
} // namespace fastllm
