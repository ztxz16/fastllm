#include "devices/cuda/low_latency/norm_linear.h"
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>
using namespace fastllm::cuda::low_latency;
using Bf16 = __nv_bfloat16;
static void Check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
static void Cuda(cudaError_t error) { Check(error == cudaSuccess,cudaGetErrorString(error)); }
static float Round(float x) { return float(Bf16(x)); }
template<class T> struct Allocation {
    T *data = nullptr;
    size_t count;
    explicit Allocation(size_t n):count(n) { Cuda(cudaMalloc(&data,n*sizeof(T))); }
    ~Allocation() { cudaFree(data); }
    Allocation(const Allocation &) = delete;
    ConstBuffer Read() const { return {data,count*sizeof(T)}; }
    Buffer Write() const { return {data,count*sizeof(T)}; }
    void Put(const std::vector<T> &v) { Check(v.size()==count,"upload size"); Cuda(cudaMemcpy(data,v.data(),count*sizeof(T),cudaMemcpyHostToDevice)); }
    std::vector<T> Get() const { std::vector<T> v(count); Cuda(cudaMemcpy(v.data(),data,count*sizeof(T),cudaMemcpyDeviceToHost)); return v; }
};
static float Random(unsigned &state) { state=1664525*state+1013904223; return (int(state>>16)-32768)/32768.f; }
struct Case {
    NormLinearShape shape;
    Allocation<Bf16> input, branch, weight;
    Allocation<float> gamma, output, mirror;
    std::vector<Bf16> x,r,w;
    std::vector<float> g;
    NormLinearPlan plan;
    NormLinearBindings bindings;
    Case(int rows,int inner,int channels,bool padded,bool residual,bool mirrored,unsigned seed=42):
        shape{rows,inner,channels,inner+(padded?3:0),inner+(padded?5:0),inner+(padded?8:0),channels+(padded?7:0)},
        input(rows*shape.inputStride),branch(rows*shape.branchStride),weight(channels*shape.weightStride),
        gamma(inner),output(rows*shape.outputStride),mirror(rows*(channels+19)+13),
        x(input.count),r(branch.count),w(weight.count),g(inner) {
        for (auto &v:x) v=Bf16(Random(seed));
        for (auto &v:r) v=Bf16(Random(seed)*.2f);
        for (auto &v:w) v=Bf16(Random(seed)*.1f);
        for (auto &v:g) v=1+Random(seed)*.1f;
        input.Put(x);branch.Put(r);weight.Put(w);gamma.Put(g);
        output.Put(std::vector<float>(output.count,-1234));mirror.Put(std::vector<float>(mirror.count,-1234));
        bindings.input=input.Read();bindings.branch=residual?branch.Read():ConstBuffer{};
        bindings.weight=weight.Read();bindings.gamma=gamma.Read();bindings.output=output.Write();
        if (mirrored) { bindings.mirror={mirror.data+7,(mirror.count-7)*sizeof(float)};bindings.mirrorStride=channels+19; }
    }
    void Prepare(cudaStream_t stream) { Cuda(PrepareNormLinear(shape,bool(bindings.mirror.data),plan,stream)); }
    void Run(cudaStream_t stream) { Cuda(RunNormLinear(plan,bindings,stream)); }
    double Verify() {
        auto actual=output.Get();auto copied=mirror.Get();double worst=0;
        for (int row=0;row<shape.rows;++row) {
            // Independent scalar, double-precision norm and dot reference,
            // explicitly reproducing only the public BF16 rounding boundaries.
            std::vector<float> h(shape.inner);double sum=0;
            for (int k=0;k<shape.inner;++k) {
                float value=float(x[row*shape.inputStride+k]);
                if (bindings.branch.data) value=Round(value+float(r[row*shape.branchStride+k]));
                h[k]=value;sum+=double(value)*value;
            }
            const float inverse=float(1/std::sqrt(sum/shape.inner+bindings.epsilon));
            for (int k=0;k<shape.inner;++k) h[k]=Round(Round(h[k]*inverse)*g[k]);
            double error=0,magnitude=0;
            for (int col=0;col<shape.channels;++col) {
                double dot=0;
                for (int k=0;k<shape.inner;++k) dot+=double(h[k])*float(w[col*shape.weightStride+k]);
                const float expected=Round(float(dot)),value=actual[row*shape.outputStride+col];
                Check(std::isfinite(value),"nonfinite output");
                error+=std::pow(double(value)-expected,2);magnitude+=double(expected)*expected;
            }
            const double l2=std::sqrt(error/(magnitude+1e-30));
            Check(l2<.001,"per-row normalized L2 >= 0.001");worst=std::max(worst,l2);
            for (int col=shape.channels;col<shape.outputStride;++col)
                Check(actual[row*shape.outputStride+col]==-1234,"output padding overwritten");
        }
        for (size_t i=0;i<copied.size();++i) {
            const int64_t pos=int64_t(i)-7, row=pos/(shape.channels+19),col=pos%(shape.channels+19);
            const bool active=bindings.mirror.data && pos>=0 && row<shape.rows && col<shape.channels;
            Check(copied[i]==(active?actual[row*shape.outputStride+col]:-1234),"mirror guard or value");
        }
        Check(input.Get()==x && branch.Get()==r && weight.Get()==w && gamma.Get()==g,"input mutated");
        return worst;
    }
    void Reject(cudaStream_t stream) {
        auto saved=output.Get();NormLinearBindings bad=bindings;
        --bad.output.bytes; // Use the exact required span, including the row tail.
        bad.output.bytes=((shape.rows-1)*shape.outputStride+shape.channels)*sizeof(float)-1;
        Check(!CanRunNormLinear(plan,bad),"short output accepted");
        Check(RunNormLinear(plan,bad,stream)==cudaErrorInvalidValue,"short output launched");
        bad=bindings;bad.output={input.data,input.count*sizeof(Bf16)};
        Check(!CanRunNormLinear(plan,bad),"alias accepted");
        bad=bindings;bad.epsilon=NAN;Check(!CanRunNormLinear(plan,bad),"NaN epsilon accepted");
        bad=bindings;bad.weight.data=weight.data+1;Check(!CanRunNormLinear(plan,bad),"unaligned weight accepted");
        bad=bindings;bad.input.bytes=0;Check(!CanRunNormLinear(plan,bad),"missing input capacity accepted");
        bad=bindings;bad.mirror=bindings.mirror.data?Buffer{}:output.Write();
        Check(!CanRunNormLinear(plan,bad),"mirror mode mismatch accepted");
        NormLinearShape unsupported=shape;unsupported.inner=257;
        Check(!SupportsNormLinearShape(unsupported),"K tail accepted");
        Check(PrepareNormLinear(unsupported,false,plan,stream)==cudaErrorNotSupported,"invalid prepare");
        Check(plan.Ready() && plan.Shape().inner==shape.inner,"failed preparation destroyed plan");
        Cuda(cudaStreamSynchronize(stream));Check(saved==output.Get(),"rejection wrote output");
    }
    void Graph(cudaStream_t stream) {
        cudaGraph_t graph;cudaGraphExec_t exec;
        Cuda(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        NormLinearPlan rejected;
        Check(PrepareNormLinear(shape,bool(bindings.mirror.data),rejected,stream)==cudaErrorStreamCaptureUnsupported,
              "prepare allowed inside capture");
        Run(stream);Cuda(cudaStreamEndCapture(stream,&graph));
        size_t nodes=0;Cuda(cudaGraphGetNodes(graph,nullptr,&nodes));Check(nodes==1,"logical batch split launches");
        Cuda(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        for (int i=0;i<7;++i) Cuda(cudaGraphLaunch(exec,stream));
        Cuda(cudaStreamSynchronize(stream));Cuda(cudaGraphExecDestroy(exec));Cuda(cudaGraphDestroy(graph));
    }
};
int main() {
    try {
        int count=0; if (cudaGetDeviceCount(&count)!=cudaSuccess || !count) return 77;
        cudaDeviceProp properties{};Cuda(cudaGetDeviceProperties(&properties,0));if(properties.major<7)return 77;
        cudaStream_t stream;Cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        double worst=0;int cases=0;
        for (int rows:{1,2,3,4,7,8,9,16,17,32,64,65}) for (bool residual:{false,true}) {
            Case test(rows,256,17,true,residual,true);test.Prepare(stream);test.Run(stream);
            Cuda(cudaStreamSynchronize(stream));test.Reject(stream);test.Graph(stream);
            worst=std::max(worst,test.Verify());++cases;
            // Same plan, fresh bindings: no preparation/capture for request changes.
            Case rebound(rows,256,17,true,residual,true,57);rebound.plan=test.plan;
            rebound.Run(stream);Cuda(cudaStreamSynchronize(stream));worst=std::max(worst,rebound.Verify());
        }
        for (int rows:{1,2,4,8}) for(int inner:{1024,4096,8192}) {
            Case test(rows,inner,48,false,true,false);
            auto status=PrepareNormLinear(test.shape,false,test.plan,stream);
            if (status==cudaErrorNotSupported) continue; // Resource fallback on smaller-SMEM GPUs.
            Cuda(status);test.Graph(stream);worst=std::max(worst,test.Verify());++cases;
        }
        // Force a smaller physical tile on this device too. Dense strides with
        // logical rows <=8 used to ignore grid.y when resource-limited.
        for (int rows : {3, 7, 8, 9, 17}) for (size_t limit : {size_t(48*1024), size_t(99*1024)}) {
            Case test(rows,8192,48,false,true,true);
            Cuda(PrepareNormLinear(test.shape,true,test.plan,stream,{limit}));
            Check(test.plan.RowTile() < 8,"resource cap did not reduce tile");
            test.Graph(stream);worst=std::max(worst,test.Verify());++cases;
            auto saved=test.output.Get();
            Check(PrepareNormLinear(test.shape,true,test.plan,stream,{1024})==cudaErrorNotSupported,
                  "impossible resource budget accepted");
            test.Run(stream);Cuda(cudaStreamSynchronize(stream));
            Check(test.output.Get()==saved,"failed preparation changed reusable plan");
        }
        Check(BackendUnavailable(cudaErrorNotSupported) && BackendUnavailable(cudaErrorNoKernelImageForDevice),
              "capability miss must fall back");
        Check(BackendUnavailable(cudaErrorUnsupportedPtxVersion) && BackendUnavailable(cudaErrorJitCompilerNotFound),
              "unavailable PTX compiler must fall back");
#if CUDART_VERSION >= 11060
        Check(BackendUnavailable(cudaErrorJitCompilationDisabled), "disabled PTX JIT must fall back");
#endif
        Check(!BackendUnavailable(cudaErrorInvalidPtx) && !BackendUnavailable(cudaErrorIllegalAddress) && !BackendUnavailable(cudaErrorMemoryAllocation) &&
              !BackendUnavailable(cudaErrorLaunchFailure),"execution failure disguised as fallback");
        // A second prepare using the same kernel image must not lower its
        // shared-memory ceiling and invalidate an existing larger capture.
        Case large(7,7680,16,false,true,false),small(7,5120,16,false,true,false);
        auto status=PrepareNormLinear(large.shape,false,large.plan,stream);
        if(status==cudaSuccess) {
            cudaGraph_t graph;cudaGraphExec_t exec;
            Cuda(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));large.Run(stream);
            Cuda(cudaStreamEndCapture(stream,&graph));Cuda(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
            small.Prepare(stream);Cuda(cudaGraphLaunch(exec,stream));Cuda(cudaStreamSynchronize(stream));
            worst=std::max(worst,large.Verify());Cuda(cudaGraphExecDestroy(exec));Cuda(cudaGraphDestroy(graph));
        } else Check(status==cudaErrorNotSupported,"large plan unexpected error");
        NormLinearShape overflow{2,256,17,INT64_MAX,256,256,17};
        Check(!SupportsNormLinearShape(overflow),"overflowing span accepted");
        int peer=0;
        if(count>1) Cuda(cudaDeviceCanAccessPeer(&peer,0,1));
        if(peer) {
            Case test(3,1024,17,true,true,false);
            Cuda(cudaSetDevice(1));
            {
                Allocation<float> remote(3*36+13);remote.Put(std::vector<float>(remote.count,-1234));
                Cuda(cudaSetDevice(0));const auto enabled=cudaDeviceEnablePeerAccess(1,0);
                if(enabled==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();else Cuda(enabled);
                test.bindings.mirror={remote.data+7,(remote.count-7)*sizeof(float)};
                test.bindings.mirrorStride=36;test.Prepare(stream);test.Graph(stream);
                const auto actual=test.output.Get();Cuda(cudaSetDevice(1));const auto copied=remote.Get();
                for(size_t i=0;i<copied.size();++i) {
                    const int64_t pos=int64_t(i)-7,row=pos/36,col=pos%36;
                    const bool active=pos>=0&&row<3&&col<17;
                    Check(copied[i]==(active?actual[row*test.shape.outputStride+col]:-1234),"peer mirror guard/visibility");
                }
            }
            Cuda(cudaSetDevice(0));
        }
        NormLinearPlan sharedPlan;
        Cuda(PrepareNormLinear({9,4096,48,4099,4101,4104,55},true,sharedPlan,stream));
        std::exception_ptr errors[2];
        auto concurrent=[&](int worker) { try {
            Cuda(cudaSetDevice(0));cudaStream_t s;Cuda(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
            Case test(9,4096,48,true,true,true,100+worker);test.plan=sharedPlan;
            for(int i=0;i<4;++i)test.Run(s);Cuda(cudaStreamSynchronize(s));test.Verify();Cuda(cudaStreamDestroy(s));
        } catch(...) { errors[worker]=std::current_exception(); } };
        std::thread a(concurrent,0),b(concurrent,1);a.join();b.join();for(auto error:errors)if(error)std::rethrow_exception(error);
        Cuda(cudaStreamDestroy(stream));
        std::printf("PASS generic norm linear cases=%d max_row_l2=%.9g workspace_bytes=0\n",cases,worst);
        return 0;
    } catch(const std::exception &error) { std::fprintf(stderr,"FAIL: %s\n",error.what());return 1; }
}
