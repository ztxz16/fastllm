// Standalone ROCm sampling regression test. Link against libfastllm_tools.
#include "devices/cuda/fastllm-cuda.cuh"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

void Check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
void Hip(hipError_t e) { Check(e == hipSuccess, hipGetErrorString(e)); }
template<class T> struct Buffer {
    T *p = nullptr; size_t n;
    explicit Buffer(size_t n) : n(n) { Hip(hipMalloc((void**)&p, n * sizeof(T))); }
    ~Buffer() { (void)hipFree(p); }
    void Put(const std::vector<T> &v) { Check(v.size() == n, "upload shape"); Hip(hipMemcpy(p, v.data(), n*sizeof(T), hipMemcpyHostToDevice)); }
    std::vector<T> Get() { Hip(hipStreamSynchronize(hipStreamPerThread)); std::vector<T> v(n); Hip(hipMemcpy(v.data(), p, n*sizeof(T), hipMemcpyDeviceToHost)); return v; }
};

void Distribution(const char *name, float temperature, int k, float topP,
        const std::vector<double> &expected, bool ties=false) {
    const int batch=4096, vocab=4;
    std::vector<float> logits(batch*vocab);
    double original[]={0.4,0.3,0.2,0.1};
    for (int b=0;b<batch;++b) for(int i=0;i<vocab;++i) logits[b*vocab+i]=ties?0:std::log(original[i]);
    Buffer<float> gpu(logits.size()); gpu.Put(logits);
    std::vector<float> temperatures(batch,temperature), ps(batch,topP);
    std::vector<int> ks(batch,k), output(batch,-99);
    Check(FastllmCudaTopKTopPSampling(gpu.p,temperatures.data(),ks.data(),ps.data(),output.data(),batch,vocab),"host sampling failed");
    int counts[4]={};
    for (int id:output) { Check(id>=0 && id<vocab,"invalid sampled id"); ++counts[id]; }
    for(int i=0;i<vocab;++i) {
        if(expected[i]==0) Check(counts[i]==0,"sample outside permitted set");
        else Check(std::fabs((double)counts[i]/batch-expected[i])<0.04,"sampling distribution mismatch");
    }
    std::cout<<name<<" PASS: "; for(int n:counts) std::cout<<n<<' '; std::cout<<std::endl;
}

void DeviceAndGraph() {
    const int batch=64,vocab=4;
    std::vector<float> logits(batch*vocab);
    for(int b=0;b<batch;++b) for(int j=0;j<vocab;++j) logits[b*vocab+j]=std::log((float)(4-j));
    Buffer<float> x(logits.size()),p(logits.size()),t(batch),ps(batch),fo(batch);
    Buffer<int> k(batch),out(batch);
    x.Put(logits);t.Put(std::vector<float>(batch,1));ps.Put(std::vector<float>(batch,1));k.Put(std::vector<int>(batch,0));
    auto launch=[&]() { return FastllmCudaTopKTopPSamplingToDevice(x.p,p.p,t.p,k.p,ps.p,nullptr,nullptr,0,out.p,fo.p,batch,vocab); };
    Check(launch(),"device sampling failed");
    Hip(hipStreamSynchronize(hipStreamPerThread));
    hipGraph_t graph; hipGraphExec_t executable;
    Hip(hipStreamBeginCapture(hipStreamPerThread,hipStreamCaptureModeThreadLocal));
    Check(launch(),"capture launch failed");
    Hip(hipStreamEndCapture(hipStreamPerThread,&graph));
    Hip(hipGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    std::vector<int> first; bool changed=false; int counts[4]={};
    for(int iteration=0;iteration<64;++iteration) {
        Hip(hipGraphLaunch(executable,hipStreamPerThread));
        auto ids=out.Get();auto floats=fo.Get();
        if(iteration==0)first=ids;else changed|=ids!=first;
        for(int i=0;i<batch;++i) { Check(ids[i]>=0&&ids[i]<vocab,"graph sampled invalid id");Check(floats[i]==ids[i],"float output mismatch");++counts[ids[i]]; }
    }
    Check(changed,"graph replay repeated fixed random draws");
    for(int i=0;i<4;++i)Check(std::fabs(counts[i]/4096.0-(4-i)/10.0)<0.04,"graph distribution mismatch");
    Hip(hipGraphExecDestroy(executable));Hip(hipGraphDestroy(graph));
    // Signed and duplicate repetition penalties, including padded invalid ids.
    Buffer<float> a(6),prob(6),temps(2),topPs(2),fout(2),factors(6);
    Buffer<int> topKs(2),ids(2),penalties(6);
    a.Put({3,2,-4,-1,-1.5f,-3});temps.Put({1,1});topPs.Put({1,1});topKs.Put({1,1});
    penalties.Put({0,0,-1,0,-1,99});factors.Put({2,2,1,2,1,1});
    Check(FastllmCudaTopKTopPSamplingToDevice(a.p,prob.p,temps.p,topKs.p,topPs.p,penalties.p,factors.p,3,ids.p,fout.p,2,3),"penalty launch failed");
    Check(ids.Get()==std::vector<int>({1,1}),"signed repetition penalty mismatch");
    std::cout<<"device output / graph replay / repeated signed penalties PASS"<<std::endl;
}

void TypicalAndInvalid() {
    Buffer<float> x(8);x.Put({std::log(0.4f),std::log(0.3f),std::log(0.2f),std::log(0.1f),0,1,2,3});
    float temps[]={1,1},ps[]={1,1};int ks[]={2,1},out[2],candidates[]={3,1},rows[]={1,0},recovered[2];unsigned char accepted[2];
    Check(FastllmCudaTopKTopPSamplingWithTypicalAcceptance(x.p,temps,ks,ps,out,2,4,candidates,rows,accepted,recovered,2,0.09f,0.3f),"typical acceptance failed");
    Check(accepted[0]==1&&accepted[1]==1&&recovered[0]==3&&recovered[1]==0,"typical acceptance mismatch");
    candidates[0]=0;candidates[1]=-1;
    Check(FastllmCudaTopKTopPSamplingWithTypicalAcceptance(x.p,temps,ks,ps,out,2,4,candidates,rows,accepted,recovered,2,0.09f,0.3f),"typical rejection launch failed");
    Check(accepted[0]==0&&accepted[1]==0,"typical rejection mismatch");
    rows[0]=2;
    Check(!FastllmCudaTopKTopPSamplingWithTypicalAcceptance(x.p,temps,ks,ps,out,2,4,candidates,rows,accepted,recovered,2,0.09f,0.3f),"invalid row accepted");
    x.Put(std::vector<float>(8,-std::numeric_limits<float>::infinity()));
    Check(!FastllmCudaTopKTopPSampling(x.p,temps,ks,ps,out,2,4),"all-masked distribution accepted");
    x.Put({NAN,0,1,2,0,1,2,3});
    Check(!FastllmCudaTopKTopPSampling(x.p,temps,ks,ps,out,2,4),"NaN distribution accepted");
    std::cout<<"typical acceptance / invalid inputs PASS"<<std::endl;
}

void ImplicitStreamCapture() {
    using namespace fastllm;
    const int width = 128;
    const float eps = 1.0e-5f;
    std::vector<float> values(width), ones(width, 1.0f);
    for (int i = 0; i < width; ++i) values[i] = (i + 1) / 128.0f;
    Data input(DataType::FLOAT32, {1, width}, values);
    Data weight(DataType::FLOAT32, {width}, ones);
    Data output(DataType::FLOAT32, {1, width});
    output.Allocate();
    input.ToDevice(DataDevice::CUDA);
    weight.ToDevice(DataDevice::CUDA);
    output.ToDevice(DataDevice::CUDA);
    Check(FastllmCudaRMSNorm(input, weight, output, eps), "RMSNorm warmup failed");
    Hip(hipDeviceSynchronize());
    hipGraph_t graph;
    hipGraphExec_t executable;
    Hip(hipStreamBeginCapture(hipStreamPerThread, hipStreamCaptureModeThreadLocal));
    Check(FastllmCudaRMSNorm(input, weight, output, eps), "RMSNorm capture failed");
    Hip(hipStreamEndCapture(hipStreamPerThread, &graph));
    size_t nodes = 0;
    Hip(hipGraphGetNodes(graph, nullptr, &nodes));
    Check(nodes > 0, "implicit HIP launches missing from captured graph");
    Hip(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    for (int iteration = 0; iteration < 3; ++iteration) {
        double squareSum = 0;
        for (int i = 0; i < width; ++i) {
            values[i] = std::sin(float(i + 1) * (iteration + 1));
            squareSum += (double)values[i] * values[i];
        }
        Hip(hipMemcpy(input.cudaData, values.data(), width * sizeof(float), hipMemcpyHostToDevice));
        Hip(hipGraphLaunch(executable, hipStreamPerThread));
        Hip(hipStreamSynchronize(hipStreamPerThread));
        std::vector<float> actual(width);
        Hip(hipMemcpy(actual.data(), output.cudaData, width * sizeof(float), hipMemcpyDeviceToHost));
        double scale = 1.0 / std::sqrt(squareSum / width + eps);
        for (int i = 0; i < width; ++i) Check(std::isfinite(actual[i]) && std::fabs(actual[i] - values[i] * scale) < 1.0e-5, "captured graph did not recompute changed input");
    }
    Hip(hipGraphExecDestroy(executable));
    Hip(hipGraphDestroy(graph));
    std::cout << "implicit kernel capture and replay with changed inputs PASS" << std::endl;
}

int main() { try {
    Distribution("greedy",1,1,1,{1,0,0,0});
    Distribution("temperature-zero greedy",0,0,1,{1,0,0,0});
    Distribution("top-k",1,2,1,{4.0/7,3.0/7,0,0});
    Distribution("top-p",1,0,0.5f,{4.0/7,3.0/7,0,0});
    Distribution("joint top-k/top-p",1,3,0.3f,{1,0,0,0});
    Distribution("temperature",0.5f,0,1,{16.0/30,9.0/30,4.0/30,1.0/30});
    Distribution("ties",1,2,1,{0.25,0.25,0.25,0.25},true);
    DeviceAndGraph();TypicalAndInvalid();ImplicitStreamCapture();
    std::cout<<"ALL SAMPLING TESTS PASS"<<std::endl;
    return 0;
} catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;} }
