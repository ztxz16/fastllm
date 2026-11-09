#define main ExistingExl3MoeTests
#include "test_exl3_moe.cpp"
#undef main
#include "devices/cuda/fastllm-cuda.cuh"
#include <thread>

static void CheckResidentGraph(int rows, int hidden=256, int inter=128, int topk=3) {
    const int experts=std::max(4,topk+1);
    std::mt19937 rng(871+rows);
    std::vector<std::unique_ptr<Data>> storage;
    std::vector<Data *> weights(2+2*experts,nullptr);
    for (int e=0;e<experts;++e) for (int role=0;role<2;++role) {
        auto w=std::make_unique<Data>(EXL3);
        w->blockK=4+2*(e%3);w->blockM=role?1:2;
        w->Resize(role?std::vector<int>{hidden,inter}:std::vector<int>{2*inter,hidden});
        Fill(*w,rng);w->ToDevice(CUDA,{0},true);
        weights[2+e*2+role]=w.get();storage.push_back(std::move(w));
    }
    Data x(BFLOAT16,{rows,hidden}),ids(INT32,{rows,topk}),scores(FLOAT32,{rows,topk}),out;
    for (auto *d:{&x,&ids,&scores}) { d->ToDevice(CUDA,{0},false);d->Allocate(false); }
    auto run=[&] { Require(FastllmCudaExl3Moe(x,out,ids,scores,weights.data(),weights.size(),.01f),"resident dispatch"); };
    auto change=[&](int turn) {
        std::vector<uint16_t> a(rows*hidden);
        std::vector<int> indices(rows*topk);
        std::vector<float> scales(rows*topk);
        for (auto &v:a) v=Float32ToBFloat16RNEBits(turn==2?0.f:float(int(rng()%63)-31)/32);
        for (int i=0;i<rows*topk;++i) {
            indices[i]=turn==1?2:(i+turn)%experts;
            scales[i]=float(int(rng()%11)-5)/8;
        }
        FastllmCudaCopyFromHostToDevice(x.cudaData,a.data(),a.size()*2);
        FastllmCudaCopyFromHostToDevice(ids.cudaData,indices.data(),indices.size()*4);
        FastllmCudaCopyFromHostToDevice(scores.cudaData,scales.data(),scales.size()*4);
    };
    auto read=[&] { Require(cudaDeviceSynchronize()==cudaSuccess,"read completion");std::vector<float> v(rows*hidden);FastllmCudaCopyFromDeviceToHost(v.data(),out.cudaData,v.size()*4);return v; };
    Require(!FastllmCudaExl3MoeGraphReady(weights[2],rows,topk),"unwarmed graph admitted");
    change(0);run();for(float v:read())Require(std::isfinite(v),"nonfinite warm output");
    Require(FastllmCudaExl3MoeGraphReady(weights[2],rows,topk),"warmed graph rejected");
    Require(!FastllmCudaExl3MoeGraphReady(weights[2],rows+1,topk),"different rows admitted");
    Require(!FastllmCudaExl3MoeGraphReady(weights[2],rows,topk-1),"different top-k admitted");
    bool otherThreadReady=true;
    std::thread worker([&] {FastllmCudaSetDevice(0);otherThreadReady=FastllmCudaExl3MoeGraphReady(weights[2],rows,topk);});
    worker.join();Require(!otherThreadReady,"another thread's private scratch admitted");
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    Require(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)==cudaSuccess,"capture begin");
    run();Require(cudaStreamEndCapture(cudaStreamPerThread,&graph)==cudaSuccess,"capture end");
    Require(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0)==cudaSuccess,"instantiate");
    for (int turn=0;turn<5;++turn) {
        change(turn);
        Require(cudaGraphLaunch(executable,cudaStreamPerThread)==cudaSuccess,"replay");
        const auto captured=read();run();const auto eager=read();
        double error=0,norm=0;
        for (size_t i=0;i<eager.size();++i) {error+=std::pow(double(captured[i])-eager[i],2);norm+=double(eager[i])*eager[i];}
        const double l2=std::sqrt(error/std::max(norm,1e-30));
        std::cout<<"REPLAY rows="<<rows<<" turn="<<turn<<" l2="<<l2<<std::endl;
        Require(captured==eager,"graph did not consume changing routes/activations/scores");
        for (float v:eager) Require(std::isfinite(v),"nonfinite output");
    }
    cudaGraphExecDestroy(executable);cudaGraphDestroy(graph);
    FastllmCudaReleaseExl3(weights[3]);
    Require(!FastllmCudaExl3MoeGraphReady(weights[2],rows,topk),"non-key weight retirement missed");
}
int main() {
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;
    SetThreads(8);
    for (int rows:{1,2,7,8,9,12,17,24,33,48,65,97}) {
        Check(rows,256,128,true);
        CheckResidentGraph(rows);
    }
    for (int rows:{9,12,24,48}) Check(rows,5120,2304,true);
    CheckResidentGraph(48,5120,2304,6);
    CheckResidentGraph(17,256,128,8);
    std::cout<<"EXL3 RESIDENT GRAPH PASS: independent CPU oracle, 70 changed-input replays, retirement"<<std::endl;
}
