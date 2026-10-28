#include "utils/host_moe_decode.h"
#include "devices/cpu/cpudevice.h"
#include "utils/nsys.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <mutex>
#include <thread>

namespace fastllm {
namespace {
struct CpuDecodeSubset : CpuMergeMOE { using CpuMergeMOE::Run; };

// CpuMergeMOE dispatches into the global CPU pool itself. Keep its outer
// coordinator on a dedicated worker, rather than occupying a pool worker or
// paying Windows thread startup/teardown on every expert layer.
class CpuDecodeWorker {
    std::mutex mutex;
    std::condition_variable workReady, stateChanged;
    std::function<void()> task;
    bool stopping = false, busy = false, started = false;
    std::thread thread;

    void Run() {
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            workReady.wait(lock, [&] { return stopping || bool(task); });
            if (stopping) return;
            auto next = std::move(task);
            started = true;
            stateChanged.notify_one();
            lock.unlock();
            next(); // The submitted CPU operation captures its own exception.
            next = nullptr;
            lock.lock();
            busy = false;
            stateChanged.notify_one();
        }
    }
public:
    CpuDecodeWorker() : thread([this] { Run(); }) {}
    ~CpuDecodeWorker() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        workReady.notify_one();
        thread.join();
    }
    void Submit(const std::function<void()> &fn) {
        std::unique_lock<std::mutex> lock(mutex);
        task = fn;
        busy = true;
        started = false;
        workReady.notify_one();
        // Wake the existing worker before submitting GPU work. A newly
        // created std::thread often started after GPU submission on Windows.
        stateChanged.wait(lock, [&] { return started; });
    }
    void Wait() {
        std::unique_lock<std::mutex> lock(mutex);
        stateChanged.wait(lock, [&] { return !busy; });
    }
};
}
bool CanRunHostMoeDecodeExperts(Data *const *weights, int count) {
    if (!weights || count<4 || count%2 || weights[0] || weights[1] ||
        !weights[2] || !weights[3] || weights[3]->dims.size()!=2) return false;
    const int hidden=weights[3]->dims[0],inter=weights[3]->dims[1];
    if (hidden<=0 || inter<=0) return false;
    for(int i=2;i<count;++i) {
        const auto *w=weights[i];
        if (!w || w->dataDevice!=CPU || !w->cpuData || w->isDiskWeight ||
            !w->numasData.empty() || w->dataType!=DATA_GGUF_FORMAT || !w->ggmlTensor ||
            w->dims!=std::vector<int>({i%2?hidden:2*inter,i%2?inter:hidden})) return false;
    }
    return true;
}
void PrepareHostMoeDecodeWeights(Data *const *weights,int count) {
    // In-place CPU packing must finish before publishing borrowed source pointers.
    // Each weight retains its allocation and its ordinary gate/up row order.
    for(int i=2;i<count;++i) weights[i]->Repack();
}
void HostMoeDecodeExperts(const float *input,float *output,Data **weights,
    const int32_t *indices,const int32_t *gpuIndices,int topk,
    const std::function<void()> &submitGpu,double *cpuElapsedUs) {
    const int hidden=weights[2]->dims[1];
    std::vector<int> routes;
    for(int r=0;r<topk;++r) if(gpuIndices[r]<0) routes.push_back(r);
    if(cpuElapsedUs) *cpuElapsedUs=0;
    if(routes.empty()) {if(submitGpu) {NsysRange range("host_moe_decode/gpu_submit");submitGpu();}return;}
    std::exception_ptr cpuError,gpuError;
    auto cpu=[&] {
        try {
            NsysRange range("host_moe_decode/cpu");
            const auto start=std::chrono::steady_clock::now();
            const int n=routes.size();
            // Preserve unweighted per-route outputs for the common CUDA
            // reduction. Duplicate experts still produce one result per route.
            Data x(FLOAT32,{1,hidden}),ids(INT32,{1,n}),scores(FLOAT32,{1,n}),result(FLOAT32,{n,hidden});
            x.Allocate();ids.Allocate();scores.Allocate();
            std::memcpy(x.cpuData,input,hidden*sizeof(float));
            for(int j=0;j<n;++j) {
                reinterpret_cast<int32_t *>(ids.cpuData)[j]=indices[routes[j]];
                reinterpret_cast<float *>(scores.cpuData)[j]=1;
            }
            Data w1,w2,w3,tmpIn,tmpOut;
            std::vector<Data *> bias(2*(int(*std::max_element(indices,indices+topk))+2),nullptr);
            const int count=bias.size();
            CpuDecodeSubset op;
            op.Run("MergeMOE",{{"input",&x},{"output",&result},{"index",&ids},{"score",&scores},
                {"weights",reinterpret_cast<Data *>(weights)},{"biass",reinterpret_cast<Data *>(bias.data())},
                {"w1",&w1},{"w2",&w2},{"w3",&w3},{"temporaryInput",&tmpIn},{"temporaryOutput",&tmpOut}},
                {},{{"weights___batch",count},{"biass___batch",count},{"hostMoeDecodeRoutes",1}});
            for(int j=0;j<n;++j) std::memcpy(output+size_t(routes[j])*hidden,
                result.cpuData+size_t(j)*hidden*sizeof(float),hidden*sizeof(float));
            if(cpuElapsedUs) *cpuElapsedUs=std::chrono::duration<double,std::micro>(
                std::chrono::steady_clock::now()-start).count();
        } catch(...) {cpuError=std::current_exception();}
    };
    if(submitGpu) {
        static thread_local CpuDecodeWorker worker;
        {NsysRange range("host_moe_decode/wake_cpu");worker.Submit(cpu);}
        try {NsysRange range("host_moe_decode/gpu_submit");submitGpu();}
        catch(...) {gpuError=std::current_exception();}
        {NsysRange range("host_moe_decode/wait_cpu");worker.Wait();}
    } else cpu();
    if(cpuError) std::rethrow_exception(cpuError);
    if(gpuError) std::rethrow_exception(gpuError);
}
}
