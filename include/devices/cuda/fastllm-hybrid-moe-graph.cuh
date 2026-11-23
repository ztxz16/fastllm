#pragma once
#include "devices/cuda/fastllm-moe-graph-timing.cuh"
#include "utils/hybrid_moe_routes.h"
#include <chrono>
#include <memory>
#include <stdexcept>

namespace fastllm { namespace cuda {

// Captured CPU/NUMA and CUDA expert branches. The caller supplies route metadata,
// numerical operators and their buffers; this object owns the dependencies and
// collects timings once the enclosing target graph has completed.
class HybridMoeGraph {
    static void check(cudaError_t e) {
        if (e != cudaSuccess) throw std::runtime_error(std::string("Hybrid MoE graph: ") + cudaGetErrorString(e));
    }
    struct Event {
        cudaEvent_t handle = nullptr;
        void create(unsigned flags = cudaEventDefault) { check(cudaEventCreateWithFlags(&handle, flags)); }
        ~Event() { if (handle) cudaEventDestroy(handle); }
    };
public:
    struct Visit {
        MoeDecodeOverlapScheduler &cost;
        using Clock = std::chrono::steady_clock;
        const HybridMoeGraph &graph;
        const size_t index;
        uint64_t lastUploadSample = 0;
        int residentRows = 0, stagedRows = 0, uploads = 0;
        Clock::time_point hostStart;
        Event begin, residentDone, copied, computed;
        Visit(MoeDecodeOverlapScheduler &cost, const HybridMoeGraph &graph, size_t index)
            : cost(cost), graph(graph), index(index) {
            begin.create(); residentDone.create(); copied.create(); computed.create();
        }
        bool probeUpload() const {
            // Rotate a small real transfer across visits on instrumented launches.
            // Natural transfers refresh the same sample without an extra probe.
            const auto sample = graph.samples_;
            return graph.sampled_ && (sample + index) % 4 == 0 &&
                   (!lastUploadSample || sample - lastUploadSample >= 4);
        }
        void routes(const HybridMoeRoutes &plan) {
            residentRows = plan.residentRows;
            stagedRows = plan.stagedRows;
            uploads = 0;
        }
        void beginHost() { hostStart = Clock::now(); }
        void finishHost(int rows, int routes) {
            HybridMoeRoutes::ObserveCpu(cost, rows, routes,
                std::chrono::duration<double, std::micro>(Clock::now() - hostStart).count());
        }
        void observe() {
            auto elapsed = [](const Event &a, const Event &b) {
                float ms;
                check(cudaEventElapsedTime(&ms, a.handle, b.handle));
                return 1000. * ms;
            };
            if (uploads) {
                cost.copiedExpert.Observe(elapsed(begin, copied) / uploads);
                lastUploadSample = graph.samples_;
            }
            if (residentRows) cost.residentExpert.Observe(elapsed(begin, residentDone) / residentRows);
            if (stagedRows) cost.stagedExpert.Observe(elapsed(copied, computed) / stagedRows);
        }
    };
    HybridMoeGraph() {
        fork_.create(cudaEventDisableTiming);
        joined_.create(cudaEventDisableTiming);
        gpuFork_.create(cudaEventDisableTiming);
        stagedDone_.create(cudaEventDisableTiming);
        check(cudaStreamCreateWithFlags(&host_, cudaStreamNonBlocking));
        auto status = cudaStreamCreateWithFlags(&staged_, cudaStreamNonBlocking);
        if (status != cudaSuccess) { cudaStreamDestroy(host_); check(status); }
    }
    ~HybridMoeGraph() {
        if (steady_) cudaGraphExecDestroy(steady_);
        if (staged_) cudaStreamDestroy(staged_);
        if (host_) cudaStreamDestroy(host_);
    }
    HybridMoeGraph(const HybridMoeGraph &) = delete;
    Visit &add(MoeDecodeOverlapScheduler &cost) {
        visits_.push_back(std::make_unique<Visit>(cost, *this, visits_.size()));
        return *visits_.back();
    }
    // The plan publishes disjoint route masks. Only staged routes depend on
    // refill. Both GPU branches wait for any publication from the previous step.
    template<class Host, class PrepareSlots, class Upload, class Compute>
    void parallel(cudaStream_t main, Visit &visit, Host host, PrepareSlots prepare, Upload upload, Compute compute) {
        check(cudaEventRecord(fork_.handle, main));
        check(cudaStreamWaitEvent(host_, fork_.handle, 0));
        host(host_);
        check(cudaEventRecord(joined_.handle, host_));
        prepare();
        check(cudaEventRecordWithFlags(visit.begin.handle, main, cudaEventRecordExternal));
        check(cudaEventRecord(gpuFork_.handle, main));
        check(cudaStreamWaitEvent(staged_, gpuFork_.handle, 0));
        upload(staged_);
        check(cudaEventRecordWithFlags(visit.copied.handle, staged_, cudaEventRecordExternal));
        compute(staged_, true);
        check(cudaEventRecordWithFlags(visit.computed.handle, staged_, cudaEventRecordExternal));
        check(cudaEventRecord(stagedDone_.handle, staged_));
        compute(main, false);
        check(cudaEventRecordWithFlags(visit.residentDone.handle, main, cudaEventRecordExternal));
        check(cudaStreamWaitEvent(main, stagedDone_.handle, 0));
        check(cudaStreamWaitEvent(main, joined_.handle, 0));
    }
    void seal(cudaGraph_t graph) {
        std::vector<cudaEvent_t> events;
        for (const auto &v : visits_)
            events.insert(events.end(), {v->begin.handle, v->residentDone.handle, v->copied.handle, v->computed.handle});
        if (!events.empty()) steady_ = UntimedMoeGraph(graph, events);
    }
    cudaGraphExec_t select(cudaGraphExec_t instrumented) {
        sampled_ = !steady_ || SampleHybridMoeReplay(replays_);
        if (sampled_) ++samples_;
        ++replays_;
        return sampled_ ? instrumented : steady_;
    }
    void observe() {
        if (sampled_) for (auto &v : visits_) v->observe();
    }
private:
    cudaStream_t host_ = nullptr, staged_ = nullptr;
    Event fork_, joined_, gpuFork_, stagedDone_;
    std::vector<std::unique_ptr<Visit>> visits_;
    cudaGraphExec_t steady_ = nullptr;
    uint64_t replays_ = 0, samples_ = 0;
    bool sampled_ = false;
};
}} // namespace fastllm::cuda
