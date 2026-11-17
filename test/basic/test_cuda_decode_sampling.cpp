#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#endif
#include "utils/cuda_decode_sampling.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include <set>
#include <cmath>
#include <thread>
#include <stdexcept>
#ifdef FASTLLM_TEST_DECODE_SAMPLING_HOOKS
#include <dlfcn.h>
#include <atomic>
enum class CaptureFailure { None, Begin, SecondSlot };
static CaptureFailure captureFailure = CaptureFailure::None;
static std::atomic<int> captureCalls{0}, instantiateCalls{0};
static void LeaveCudaError() {
    cudaLaunchKernel(nullptr, dim3(1), dim3(1), nullptr, 0, cudaStreamPerThread);
}
extern "C" bool FastllmCudaGraphBeginCapture() {
    ++captureCalls;
    if (captureFailure == CaptureFailure::Begin) { LeaveCudaError(); return false; }
    static auto fn = (bool (*)())dlsym(RTLD_NEXT, "FastllmCudaGraphBeginCapture");
    return fn();
}
extern "C" bool FastllmCudaGraphInstantiate(void *graph, void **exec) {
    ++instantiateCalls;
    if (captureFailure == CaptureFailure::SecondSlot && instantiateCalls == 2) {
        LeaveCudaError(); *exec = nullptr; return false;
    }
    static auto fn = (bool (*)(void *, void **))dlsym(RTLD_NEXT, "FastllmCudaGraphInstantiate");
    return fn(graph, exec);
}
#endif
using namespace fastllm;
static void Check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }

static void ConcurrentSamplers(int vocab) {
    std::exception_ptr failures[2];
    auto run=[&](int worker) {try {
        FastllmCudaSetDevice(0);
        std::vector<float> values(3*vocab,-100.f);
        for(int r=0;r<3;++r) values[r*vocab+17+worker]=100.f;
        Data logits(FLOAT32,{3,vocab},values);logits.ToDevice(DataDevice::CUDA,std::vector<int>{0},true);
        CudaDecodeSampling sampler;
        Check(sampler.Prepare(logits,{1,2,3},12345+worker),"concurrent sampler prepare");
        std::vector<CudaDecodeSampling::Request> requests;
        for(int r=0;r<3;++r) requests.push_back({uint64_t(100+worker*10+r),0,.4f,.95f,vocab});
        for(int step=0;step<16;++step) {
            Check(sampler.Submit(requests,step%2),"concurrent submit");
            for(int token:sampler.Collect(step%2)) Check(token==17+worker,"sampler scratch crossed instances");
            for(auto &r:requests) ++r.step;
        }
    }catch(...) {failures[worker]=std::current_exception();}};
    std::thread a(run,0),b(run,1);a.join();b.join();
    for(auto e:failures) if(e) std::rethrow_exception(e);
}
static void InPlaceSamplingChecks() {
    constexpr int capacity = 65, vocab = 65536;
    std::vector<float> values((size_t)capacity * vocab);
    for (size_t i = 0; i < values.size(); ++i) values[i] = 3.f * std::sin(i * .071f);
    Data input(FLOAT32, {capacity, vocab}, values), reference(FLOAT32, {capacity, vocab}, values);
    input.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
    reference.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
    CudaDecodeSampling consumed, preserved;
    Check(consumed.Prepare(input, {1, 3, capacity}, 4321, true) &&
          preserved.Prepare(reference, {1, 3, capacity}, 4321), "in-place sampler prepare");
#ifdef FASTLLM_TEST_DECODE_SAMPLING_HOOKS
    const int captures = captureCalls;
#endif
    int step = 0;
    for (int rows : {1, 3, capacity, 3, 1, capacity}) {
        FastllmCudaCopyFromHostToDevice(input.cudaData, values.data(), values.size() * sizeof(float));
        std::vector<CudaDecodeSampling::Request> requests;
        for (int row = 0; row < rows; ++row)
            requests.push_back({uint64_t(1000 + rows - row), uint64_t(step), .4f, .95f, vocab});
        const int slot = step++ % 2;
        Check(consumed.Submit(requests, slot) && preserved.Submit(requests, slot), "in-place sampler submit");
        Check(consumed.Collect(slot) == preserved.Collect(slot), "in-place sampling changed token/RNG");
    }
#ifdef FASTLLM_TEST_DECODE_SAMPLING_HOOKS
    Check(captureCalls == captures, "in-place sampler recaptured on batch change");
#endif
}

int main(int argc, char **argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) { std::cout << "FASTLLM_TEST_SKIP_NO_DEVICE\n"; return 0; }
    FastllmCudaSetDevice(0);
    const int rows = 8, vocab = argc>1?std::stoi(argv[1]):4096;
    Data logits(FLOAT32, {rows,vocab}, std::vector<float>(rows*vocab,0));
    logits.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
    if(vocab>4096) ConcurrentSamplers(vocab);
    InPlaceSamplingChecks();
    CudaDecodeSampling a,b,invalid;
    Check(!invalid.Prepare(logits,{0,8}) && !invalid.Prepare(logits,{4,2}) &&
          !invalid.Prepare(logits,{1,1}) && !invalid.Prepare(logits,{1,16}), "invalid startup buckets");
    Check(a.Prepare(logits,{1,2,4,8},12345) && b.Prepare(logits,{1,2,4,8},12345),"startup sampler capture");
    Check(!a.Prepare(logits,{1,2,4,8},12345),"sampler recaptured after sealing");
    Check(!a.Supports(0) && !a.Supports(9),"unsupported batch admission");
#ifdef FASTLLM_TEST_DECODE_SAMPLING_HOOKS
    for (auto failure : {CaptureFailure::Begin, CaptureFailure::SecondSlot}) {
        captureFailure = failure; instantiateCalls = 0;
        CudaDecodeSampling failed;
        Check(!failed.Prepare(logits,{1},12345),"incomplete sampler slots admitted");
        Check(cudaGetLastError()==cudaSuccess,"startup capture left a CUDA error");
        const int captures = captureCalls;
        Check(!failed.Supports(1) && !failed.Submit({{1,0,1,1,vocab}},0),"failed sampler did not fall back");
        Check(!failed.Prepare(logits,{1},12345) && captureCalls==captures,"failed sampler recaptured at runtime");
    }
    captureFailure = CaptureFailure::None;
    const int runtimeCaptures = captureCalls;
#endif
    std::vector<CudaDecodeSampling::Request> requests;
    for(int row=0;row<3;++row) requests.push_back({uint64_t(100+row),0,1,1,vocab});
    std::set<int> values;
    for(int step=0;step<64;++step) {
        auto shuffled=requests;
        std::reverse(shuffled.begin(),shuffled.end());
        Check(a.Submit(requests,step%2) && b.Submit(shuffled,step%2),"sampling submit");
        auto x=a.Collect(step%2),y=b.Collect(step%2);
        std::reverse(y.begin(),y.end());
        Check(x==y,"request RNG changed after batch reorder");
        for(int token:x) Check(token>=0 && token<vocab,"invalid token");
        values.insert(x[0]);
        for(auto &r:requests) ++r.step;
    }
    Check(values.size()>48,"graph replay repeats RNG");
    // A row removed from a padded batch keeps the same sequence when compacted.
    auto one=std::vector<CudaDecodeSampling::Request>{requests[2]};
    Check(a.Submit(requests,0) && b.Submit(one,0),"compaction submit");
    auto x=a.Collect(0),y=b.Collect(0);
    Check(x[2]==y[0],"RNG depends on physical batch row");
    one[0].identity=10000;one[0].step=0;one[0].topK=1;
    Check(b.Submit(one,1),"parameter and identity rebind");b.Collect(1);
    ++one[0].step;
    Check(b.Submit(one,0),"queued token before discard");
    b.Discard();
    ++one[0].step;
    Check(b.Submit(one,0),"reusing slot after discard");b.Collect(0);
    // The input update uses zero live length as its padding contract.
    Data state(FLOAT32,{3,8},std::vector<float>(24,0));
    auto *p=(float*)state.cpuData;auto *n=(int*)(p+16);
    p[8]=11;p[9]=101;n[0]=12;n[1]=102;
    state.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
    Data next(FLOAT32,{4},std::vector<float>{17,18,19,20});
    next.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
    const float *mapped[4]{(float *)next.cudaData + 3, (float *)next.cudaData, nullptr, nullptr};
    Data mapping(INT32, {int(sizeof(mapped) / sizeof(int))});
    mapping.ToDevice(DataDevice::CUDA, std::vector<int>{0}, false); mapping.Allocate();
    FastllmCudaCopyFromHostToDevice(mapping.cudaData, mapped, sizeof(mapped));
    Data localPos(FLOAT32, {2}, std::vector<float>{11, -1});
    Data localLens(INT32, {2}); localLens.Allocate();
    ((int *)localLens.cpuData)[0] = 12; ((int *)localLens.cpuData)[1] = 0;
    localPos.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
    localLens.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
    Check(FastllmCudaAdvanceMappedDecodeInput(state.cudaData, 8, 4, (const float *const *)mapping.cudaData,
        (float *)localPos.cudaData, (int *)localLens.cudaData, 2), "mapped input advance");
    state.ToDevice(DataDevice::CPU); localPos.ToDevice(DataDevice::CPU); localLens.ToDevice(DataDevice::CPU);
    p=(float *)state.cpuData; n=(int *)(p+16);
    Check(p[0]==20 && p[1]==17 && p[2]==0 && p[8]==12 && n[0]==13 && n[2]==0 &&
        ((float *)localPos.cpuData)[0]==12 && ((float *)localPos.cpuData)[1]==-1 &&
        ((int *)localLens.cpuData)[0]==13 && ((int *)localLens.cpuData)[1]==0,
        "mapped token permutation/local positions/padding semantics");
#ifdef FASTLLM_TEST_DECODE_SAMPLING_HOOKS
    Check(captureCalls==runtimeCaptures,"sampling replay recaptured a graph");
#endif
    std::cout << "CUDA DECODE SAMPLING PASS\n";
}
