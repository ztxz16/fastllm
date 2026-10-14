#pragma once
#include "models/glm5_next.h"
#include "utils/persistent_worker_group.h"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include "devices/cuda/cudaworkspace.h"
#endif
#include <algorithm>
#include <condition_variable>
#include <chrono>
#include <cstdio>
#include <stdexcept>

namespace fastllm {
    struct Glm5NextModel::ThreadTpState {
        using Cache = std::vector<std::pair<Data, Data>>;
        std::vector<int> devices;
        std::vector<std::vector<int>> workerCpus;
        std::vector<unsigned char> workerBoundToSpareCores;
        std::vector<std::unique_ptr<Glm5NextModel>> ranks;
        std::map<const Cache *, std::vector<Cache>> requests;
        std::map<const Cache *, std::vector<std::vector<KdaReplayCapture>>> replays;
        std::mutex forwardMutex, barrierMutex;
        std::condition_variable barrierCv;
        unsigned generation = 0;
        int arrived = 0;
        bool failed = false;
        PersistentWorkerGroup workers;
        std::vector<void *> moeReady, callerReady;
        std::vector<bool> shardedMoeLayers;
        int loadingLayer = -1;
#ifdef USE_CUDA
        std::vector<std::vector<Data>> reductionBuffers;
        // Graphs belong to the request whose recurrent states they reference.
        struct KdaGraphLayer {
            struct Rank {
                Data input, output;
                void *graph = nullptr, *exec = nullptr;
                bool warmed = false, begun = false, ok = false;
                std::exception_ptr error;
            };
            std::vector<Rank> ranks;
            std::vector<int> devices;
            std::vector<void *> reserved;
            bool pool = false, disabled = false;
            explicit KdaGraphLayer(const std::vector<int> &ids) : ranks(ids.size()), devices(ids) {}
            ~KdaGraphLayer() {
                const int previous = FastllmCudaGetDevice();
                for (size_t r = 0; r < ranks.size(); ++r) {
                    FastllmCudaSetDevice(devices[r]);
                    if (ranks[r].exec) FastllmCudaGraphExecDestroy(ranks[r].exec);
                    if (ranks[r].graph) FastllmCudaGraphDestroy(ranks[r].graph);
                }
                FastllmCudaGraphMemoryPoolRelease(reserved);
                FastllmCudaSetDevice(previous);
            }
        };
        using RequestGraphs = std::vector<std::unique_ptr<KdaGraphLayer>>;
        std::map<const Cache *, RequestGraphs> graphs;
        RequestGraphs *activeGraphs = nullptr;
#endif

        ~ThreadTpState() {
            workers.Stop();
#ifdef USE_CUDA
            graphs.clear();
            const int previous = FastllmCudaGetDevice();
            for (size_t r = 0; r < moeReady.size(); ++r) if (moeReady[r]) {
                FastllmCudaSetDevice(devices[r]);
                FastllmCudaEventDestroy(moeReady[r]);
            }
            for (size_t r = 0; r < callerReady.size(); ++r) if (callerReady[r]) {
                FastllmCudaSetDevice(devices[r]);
                FastllmCudaEventDestroy(callerReady[r]);
            }
            FastllmCudaSetDevice(previous);
#endif
        }

        void Abort() {
            std::lock_guard<std::mutex> lock(barrierMutex);
            failed = true;
            barrierCv.notify_all();
        }

        void Barrier() {
            std::unique_lock<std::mutex> lock(barrierMutex);
            if (failed) throw std::runtime_error("GLM TP peer failed");
            unsigned previous = generation;
            if (++arrived == (int)devices.size()) {
                arrived = 0;
                ++generation;
                barrierCv.notify_all();
            } else if (!barrierCv.wait_for(lock, std::chrono::seconds(120), [&] {
                    return failed || generation != previous;
                })) {
                failed = true;
                barrierCv.notify_all();
                throw std::runtime_error("GLM TP rank rendezvous timed out");
            }
            if (failed) throw std::runtime_error("GLM TP peer failed");
        }

#ifdef USE_CUDA
        void RunKdaGraph(int rank, int layer, Data &input, Data &output,
                const std::function<void(Data &, Data &)> &body) {
            CudaWorkspaceScope graphWorkspace(nullptr);
            auto &g = *activeGraphs->at(layer);
            auto &r = g.ranks[rank];
            r.input.CopyFrom(input);
            struct PointerTables {
                explicit PointerTables(const void *owner) { FastllmCudaGraphPointerTablesBegin(owner); }
                ~PointerTables() { FastllmCudaGraphPointerTablesEnd(); }
            } tables(&r);
            auto expose = [&] { output.FakeFrom(r.output, 0); output.Resize(r.output.dims); };
            if (r.exec) {
                AssertInFastLLM(FastllmCudaGraphLaunch(r.exec), "GLM KDA graph replay failed.");
                expose();
                return;
            }
            if (g.disabled || !r.warmed) {
                body(r.input, r.output);
                r.warmed = true;
                expose();
                return;
            }
            // Capture cannot grow the CUDA pool, and the register scan needs
            // graph-owned scratch in place of its eager borrowed workspace.
            std::vector<void *> reserve;
            for (int i = 0; i < 32; ++i)
                reserve.push_back(FastllmCudaMalloc(std::max<size_t>(65536, input.GetBytes()*2)));
            for (void *p : reserve) FastllmCudaFree(p);
            // The allocator's capture reservation is shared by all ranks.
            // Capture attention and its reduction, then publish both graphs
            // together. CPU MoE stays outside; KDA cache addresses stay fixed.
            FastllmCudaSyncCurrentThreadStream();
            Barrier();
            if (rank == 0) g.pool = FastllmCudaGraphMemoryPoolBegin();
            Barrier();
            FastllmCudaClearThreadError();
            r.begun = g.pool && FastllmCudaGraphBeginCapture();
            r.ok = r.begun;
            Barrier();
            const bool allBegun = std::all_of(g.ranks.begin(), g.ranks.end(),
                [](const auto &peer) { return peer.begun; });
            if (allBegun) {
                try { body(r.input, r.output); }
                catch (...) { r.error = std::current_exception(); r.ok = false; }
            } else r.ok = false;
            if (r.begun) {
                const bool ended = FastllmCudaGraphEndCapture(&r.graph);
                r.ok = r.ok && ended && r.graph && !FastllmCudaGetThreadError();
                if (!r.ok) std::fprintf(stderr, "[GLM TP graph] rank %d end=%d graph=%p error=%s\n",
                    rank, int(ended), r.graph, FastllmCudaGraphLastError());
            }
            Barrier();
            if (rank == 0) {
                if (g.pool) g.pool = FastllmCudaGraphMemoryPoolEnd(g.reserved);
                if (!g.pool) std::fprintf(stderr, "[GLM TP graph] pool failed: %s\n", FastllmCudaGraphLastError());
                g.disabled = !g.pool || !std::all_of(g.ranks.begin(), g.ranks.end(),
                    [](const auto &peer) { return peer.ok; });
            }
            Barrier();
            if (!g.disabled) {
                r.ok = FastllmCudaGraphInstantiate(r.graph, &r.exec);
                if (!r.ok) std::fprintf(stderr, "[GLM TP graph] rank %d instantiate: %s\n",
                    rank, FastllmCudaGraphLastError());
            }
            Barrier();
            if (rank == 0) {
                g.disabled = g.disabled || !std::all_of(g.ranks.begin(), g.ranks.end(),
                    [](const auto &peer) { return peer.ok && peer.exec; });
            }
            Barrier();
            if (g.disabled) {
                if (r.exec) { FastllmCudaGraphExecDestroy(r.exec); r.exec = nullptr; }
                if (r.graph) { FastllmCudaGraphDestroy(r.graph); r.graph = nullptr; }
                Barrier();
                if (rank == 0) {
                    FastllmCudaGraphMemoryPoolRelease(g.reserved);
                    g.reserved.clear();
                    FastllmCudaClearGraphError();
                }
                Barrier();
                FastllmCudaClearLastError();
                FastllmCudaClearThreadError();
                if (r.error) std::rethrow_exception(r.error);
                body(r.input, r.output);
            } else {
                AssertInFastLLM(FastllmCudaGraphLaunch(r.exec), "GLM KDA graph first launch failed.");
            }
            expose();
        }
#endif
    };

} // namespace fastllm
