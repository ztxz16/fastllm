#pragma once
#include "models/glm5_next.h"
#if defined(USE_CUDA) && !defined(USE_ROCM)
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/cudaworkspace.h"
#include <exception>

namespace fastllm {
// Only the hybrid single-request path owns these segments. Request state is
// validated before replay; CPU experts, batching and MTP retain their paths.
struct Glm5NextModel::HybridDecodeState {
    struct Segment {
        int device;
        Data input;
        std::vector<Data> outputs;
        std::vector<const void *> bindings;
        std::vector<std::vector<int>> shapes;
        void *graph = nullptr, *exec = nullptr;
        std::vector<void *> reserved;
        bool warmed = false, disabled = false;

        explicit Segment(int device, int count) : device(device), outputs(count) {}
        static void CopyActivation(Data &target, const Data &source) {
            // Stable graph boundaries already have matching, owned storage.
            // Keep these small copies on the computation stream without
            // repeating the general clone's device and allocation handling.
            if (target.cudaData && !target.cudaDataBorrowed &&
                target.dataDevice == DataDevice::CUDA && source.dataDevice == DataDevice::CUDA &&
                target.dataDeviceIds == source.dataDeviceIds && target.dataType == source.dataType &&
                target.dims == source.dims && target.expansionBytes >= source.GetBytes()) {
                AssertInFastLLM(FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(
                    target.cudaData, source.cudaData, source.GetBytes()), "GLM hybrid graph input copy failed.");
            } else {
                target.CopyFrom(source);
            }
        }
        ~Segment() {
            const int previous = FastllmCudaGetDevice();
            FastllmCudaSetDevice(device);
            if (exec) FastllmCudaGraphExecDestroy(exec);
            if (graph) FastllmCudaGraphDestroy(graph);
            FastllmCudaGraphMemoryPoolRelease(reserved);
            FastllmCudaSetDevice(previous);
        }
        void Run(Data &source, const std::function<void(Data &, std::vector<Data> &)> &body) {
            CudaWorkspaceScope workspace(nullptr);
            CopyActivation(input, source);
            RunBound([&](std::vector<Data> &out) { body(input, out); }, source.GetBytes());
        }
        // Callers validate all bound tensor addresses before replay. This is
        // also used when the preceding graph already owns stable inputs.
        void RunBound(const std::function<void(std::vector<Data> &)> &body, size_t inputBytes) {
            CudaWorkspaceScope workspace(nullptr);
            struct PointerTables {
                explicit PointerTables(const void *owner) { FastllmCudaGraphPointerTablesBegin(owner); }
                ~PointerTables() { FastllmCudaGraphPointerTablesEnd(); }
            } tables(this);
            if (exec) {
                AssertInFastLLM(FastllmCudaGraphLaunch(exec), "GLM hybrid graph replay failed.");
                return;
            }
            if (!warmed || disabled) {
                body(outputs);
                warmed = true;
                return;
            }
            // The shared allocator cannot grow during capture. Eager warmup
            // prepares weights; these reusable blocks cover capture scratch.
            std::vector<void *> scratch;
            for (int i = 0; i < 32; ++i)
                scratch.push_back(FastllmCudaMalloc(std::max<size_t>(65536, inputBytes * 2)));
            for (void *p : scratch) FastllmCudaFree(p);
            FastllmCudaSyncCurrentThreadStream();
            if (!FastllmCudaGraphMemoryPoolBegin()) {
                body(outputs);
                return;
            }
            FastllmCudaClearThreadError();
            const bool begun = FastllmCudaGraphBeginCapture();
            bool ok = begun;
            std::exception_ptr error;
            if (begun) {
                try { body(outputs); }
                catch (...) { error = std::current_exception(); ok = false; }
                const bool ended = FastllmCudaGraphEndCapture(&graph);
                ok = ok && ended && graph && !FastllmCudaGetThreadError();
            }
            const bool pooled = FastllmCudaGraphMemoryPoolEnd(reserved);
            ok = ok && pooled;
            if (ok) ok = FastllmCudaGraphInstantiate(graph, &exec);
            if (!ok) {
                if (exec) FastllmCudaGraphExecDestroy(exec);
                if (graph) FastllmCudaGraphDestroy(graph);
                exec = graph = nullptr;
                FastllmCudaGraphMemoryPoolRelease(reserved);
                reserved.clear();
                disabled = true;
                FastllmCudaClearGraphError();
                FastllmCudaClearLastError();
                FastllmCudaClearThreadError();
                if (error) std::rethrow_exception(error);
                body(outputs);
            } else {
                AssertInFastLLM(FastllmCudaGraphLaunch(exec), "GLM hybrid graph first replay failed.");
            }
        }
    };
    struct MlaSegment {
        Data position;
        std::unique_ptr<Segment> graph;
        int nextPosition = -1;
        bool Run(Glm5NextIndexerCache &cache, Data &key, Data &gate,
            Data &query, Data &weights, Data &ape, const Data &pagedCache,
            const std::vector<const Data *> &extraBindings,
            const std::function<void(Data &, Data &)> &finish, Data &output);
    };
    using Cache = std::vector<std::pair<Data, Data>>;
    std::mutex mutex;
    std::map<const Cache *, std::map<int, std::unique_ptr<Segment>>> requests;
    std::map<int, std::unique_ptr<Segment>> mlaProjections;
    std::map<const Cache *, std::map<int, std::unique_ptr<MlaSegment>>> mlaRequests;
};
}
#endif
