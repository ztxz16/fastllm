#define main ExistingExl3MoeTests
#include "test_exl3_moe.cpp"
#undef main
#include "devices/cuda/fastllm-cuda.cuh"

static void CacheBatch(int rows, int hidden, int inter) {
    constexpr int experts=8, topk=6, gateBits=4, downBits=6;
    std::mt19937 rng(872+rows);
    std::vector<std::unique_ptr<Data>> storage;
    std::vector<Data *> weights(2+2*experts,nullptr);
    const size_t downOffset=2*Exl3PartBytes(inter,hidden,gateBits);
    const size_t stride=(downOffset+Exl3PartBytes(hidden,inter,downBits)+127)&~size_t(127);
    Data records(INT8,{int(stride*experts)});records.Allocate(false);
    for(int e=0;e<experts;++e)for(int role=0;role<2;++role) {
        auto w=std::make_unique<Data>(EXL3);w->blockK=role?downBits:gateBits;w->blockM=role?1:2;
        w->Resize(role?std::vector<int>{hidden,inter}:std::vector<int>{2*inter,hidden});Fill(*w,rng);
        std::memcpy(records.cpuData+e*stride+(role?downOffset:0),w->cpuData,w->GetBytes());
        w->ToDevice(CUDA,{0},true);weights[2+2*e+role]=w.get();storage.push_back(std::move(w));
    }
    records.ToDevice(CUDA,{0},true);
    std::vector<int> map(rows*topk),slots(rows*topk),ids(rows*topk);
    std::vector<float> scales(rows*topk),input(rows*hidden);
    for(auto &v:input)v=float(int(rng()%63)-31)/32;
    for(int r=0;r<rows*topk;++r) {map[r]=rows*topk-1-r;ids[r]=r%experts;scales[r]=float(int(r%7)-3)/8;}
    for(int r=0;r<rows*topk;++r)slots[r]=ids[map[r]];
    Data x(FLOAT32,{rows,hidden},input),index(INT32,{rows,topk},CPU,ids.data());ToDataType(x,BFLOAT16);
    Data score(FLOAT32,{rows,topk},scales),mapping(INT32,{rows*topk},CPU,map.data());
    Data selected(INT32,{rows*topk},CPU,slots.data()),partial(FLOAT32,{rows*topk,hidden}),reference;
    for(auto *v:{&x,&index,&score,&mapping,&selected,&partial}){v->ToDevice(CUDA,{0},true);v->Allocate(false);}
    auto run=[&](bool preserve=false) {
        Require(FastllmCudaExl3Cache(x,static_cast<const uint8_t *>(records.cudaData),
            static_cast<const int *>(selected.cudaData),nullptr,stride,downOffset,inter,gateBits,downBits,
            static_cast<const float *>(score.cudaData),topk,.01f,static_cast<float *>(partial.cudaData),0,
            static_cast<const int *>(mapping.cudaData),rows*topk,preserve),"full-route cache rejected");
    };
    auto read=[&]{std::vector<float> v(rows*topk*hidden);Require(cudaStreamSynchronize(cudaStreamPerThread)==cudaSuccess,"cache sync");FastllmCudaCopyFromDeviceToHost(v.data(),partial.cudaData,v.size()*4);return v;};
    run();cudaGraph_t graph;cudaGraphExec_t exec;
    Require(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal)==cudaSuccess,"capture begin");run();
    Require(cudaStreamEndCapture(cudaStreamPerThread,&graph)==cudaSuccess,"capture end");
    Require(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0)==cudaSuccess,"instantiate");
    if(rows==9) {
        // Keep the nine-row graph alive while warming twelve rows in the same
        // workspace capacity class. Its captured addresses must remain valid.
        const int larger=rows+3;
        std::vector<int> otherIds(larger*topk,0);
        Data otherX(BFLOAT16,{larger,hidden});otherX.Allocate(0.f);
        Data otherSlots(INT32,{larger*topk},CPU,otherIds.data());
        Data otherScores(FLOAT32,{larger*topk},std::vector<float>(larger*topk,.25f));
        Data otherOut(FLOAT32,{larger*topk,hidden});
        for(auto *v:{&otherX,&otherSlots,&otherScores,&otherOut}){v->ToDevice(CUDA,{0},true);v->Allocate(false);}
        Require(FastllmCudaExl3Cache(otherX,static_cast<const uint8_t *>(records.cudaData),
            static_cast<const int *>(otherSlots.cudaData),nullptr,stride,downOffset,inter,gateBits,downBits,
            static_cast<const float *>(otherScores.cudaData),topk,.01f,static_cast<float *>(otherOut.cudaData),0),
            "larger cache shape rejected");
        Require(cudaStreamSynchronize(cudaStreamPerThread)==cudaSuccess,"larger cache shape completion");
    }
    for(int turn=0;turn<4;++turn) {
        for(int r=0;r<rows*topk;++r)ids[r]=turn==1?-1:(r+turn)%experts;
        for(int r=0;r<rows*topk;++r)slots[r]=ids[map[r]];
        FastllmCudaCopyFromHostToDevice(index.cudaData,ids.data(),ids.size()*4);
        FastllmCudaCopyFromHostToDevice(selected.cudaData,slots.data(),slots.size()*4);
        Require(cudaGraphLaunch(exec,cudaStreamPerThread)==cudaSuccess,"replay");const auto captured=read();run();
        Require(read()==captured,"cache graph retained stale groups");
        Require(FastllmCudaExl3Moe(x,reference,index,score,weights.data(),weights.size(),.01f),"resident oracle");
        std::vector<float> expected(rows*hidden);FastllmCudaCopyFromDeviceToHost(expected.data(),reference.cudaData,expected.size()*4);
        double err=0,norm=0;
        for(int r=0;r<rows;++r)for(int c=0;c<hidden;++c){float sum=0;for(int j=0;j<topk;++j)sum+=captured[(r*topk+j)*hidden+c];const double v=expected[r*hidden+c];err+=(sum-v)*(sum-v);norm+=v*v;}
        const double l2=std::sqrt(err/std::max(norm,1e-30));Require(l2<.008,"cache differs from independent resident experts");
        std::cout<<"CACHE rows="<<rows<<" turn="<<turn<<" l2="<<l2<<std::endl;
    }
    std::fill(slots.begin(),slots.end(),-1);FastllmCudaCopyFromHostToDevice(selected.cudaData,slots.data(),slots.size()*4);
    const auto before=read();run(true);Require(read()==before,"missing routes overwrote other branch partials");
    cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);
}
int main(){int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return 77;SetThreads(8);
    for(int rows:{1,9,17,33,65})CacheBatch(rows,256,128);CacheBatch(17,5120,2304);
    std::cout<<"EXL3 CACHE BATCH PASS"<<std::endl;return 0;}
