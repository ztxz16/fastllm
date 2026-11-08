#define main ExistingExl3MoeTests
#include "test_exl3_moe.cpp"
#undef main
#include "devices/numas/numasdevice.h"
#include "devices/numas/numas.h"
#include "gguf.h"

static void CheckBatch(Data **weights, int weightCount, int hidden, int topk, int rows,
        int turn, float limit, bool limited, bool throwing) {
    const int experts=(weightCount-2)/2;
    std::vector<uint16_t> input(size_t(rows)*hidden);
    std::vector<float> floats(input.size()), scores(rows*topk), expected(size_t(rows)*topk*hidden, -123.f), actual(expected);
    std::vector<int32_t> ids(rows*topk), owners(ids.size());
    for(size_t i=0;i<input.size();++i) {
        input[i]=Float32ToBFloat16RNEBits(float(int((i*17+turn*23)%191)-95)/64.f);
        uint32_t bits=uint32_t(input[i])<<16;std::memcpy(&floats[i],&bits,4);
    }
    for(int i=0;i<rows*topk;++i) {
        ids[i]=(turn%2 ? (i/topk+turn+i%2) : (turn+i%2))%experts;
        owners[i]=turn%4==0 ? ids[i] : (turn%4==1 || (i+turn)%3==0 ? -1:ids[i]);
        scores[i]=float(int((i*3+turn)%11)-4)/8.f;
    }
    for(int row=0;row<rows;++row)
        NumasMoeDecodeExperts(floats.data()+size_t(row)*hidden,expected.data()+size_t(row)*topk*hidden,
            weights,ids.data()+row*topk,owners.data()+row*topk,topk,0,scores.data()+row*topk,limit);
    auto *pool=GetAlivePool();const auto saved=pool->curActivateThreadInterval;
    if(limited)pool->curActivateThreadInterval={0,1};
    if(throwing) {
        bool caught=false;
        try {NumasMoeVerifyExpertsWithOverlap(input.data(),actual.data(),rows,weights,weightCount,
            ids.data(),owners.data(),scores.data(),topk,0,limit,true,32,[]{throw std::runtime_error("submit");},nullptr,true);}
        catch(const std::runtime_error&){caught=true;}
        Require(caught,"batch lost submit exception");std::fill(actual.begin(),actual.end(),-123.f);
    }
    int submissions=0;
    NumasMoeVerifyExpertsWithOverlap(input.data(),actual.data(),rows,weights,weightCount,
        ids.data(),owners.data(),scores.data(),topk,0,limit,true,32,[&]{++submissions;},nullptr,true);
    pool->curActivateThreadInterval=saved;
    Require(submissions==1,"batch submit count");
    for(size_t i=0;i<actual.size();++i)if(std::memcmp(&actual[i],&expected[i],4)) {
        std::fprintf(stderr,"Mismatch rows=%d turn=%d limit=%g slot=%zu old=%g new=%g\n",rows,turn,limit,i,expected[i],actual[i]);
        Require(false,"packed batch differs from single-row oracle");
    }
}

// Cover both storage formats and the model's expert shape without test-only
// environment switches. Clear shared workspaces before fixture weights die.
static void CheckFormat(bool gguf, int hidden, int inter) {
    SetThreads(8);
    HostMoePlacementScope placement(false);
    constexpr int experts=6,topk=4;
    const int bits[experts]={2,3,4,5,8,16};
    std::mt19937 rng(43711);
    std::vector<std::unique_ptr<Data>> owned;
    std::vector<Data*> weights(2+2*experts,nullptr);
    std::vector<std::pair<Data*,std::string>> pending;
    for(int e=0;e<experts;++e)for(int part=0;part<2;++part) {
        const auto dims=part?std::vector<int>{hidden,inter}:std::vector<int>{2*inter,hidden};
        auto w=gguf?std::make_unique<Data>(DATA_GGUF_FORMAT,GGML_TYPE_IQ2_XS,dims):std::make_unique<Data>(EXL3);
        if(gguf) {
            w->Allocate();
            for(size_t i=0;i<w->GetBytes();++i)w->cpuData[i]=rng()%256;
            auto *blocks=reinterpret_cast<block_iq2_xs*>(w->cpuData);
            for(size_t i=0;i<w->GetBytes()/sizeof(block_iq2_xs);++i)blocks[i].d=float_to_half(.00390625f*(1+e%3));
        }else {
            w->blockM=part?1:2;w->blockK=bits[e];w->Resize(dims);Fill(*w,rng);
        }
        weights[2+2*e+part]=w.get();pending.push_back({w.get(),part?"linear":"linearSwiglu"});owned.push_back(std::move(w));
    }
    RegisterNumasMoeWeightBatch(pending);
    Require(CanRunNumasMoeDecodeExperts(weights.data(),weights.size()),"decode capability");
    int cases=0;
    for(int rows:{2,3,5,6,8,1})for(int turn=0;turn<8;++turn)for(float limit:{0.f,.013f,10.f}) {
        CheckBatch(weights.data(),weights.size(),hidden,topk,rows,turn,limit,false,rows==3&&turn==3&&limit==0.f);++cases;
    }
    for(int threads:{4,12,8}) {
        SetThreads(threads);
        for(bool limited:{false,true}) {
            std::fprintf(stderr,"RESIZE threads=%d limited=%d\n",threads,int(limited));
            CheckBatch(weights.data(),weights.size(),hidden,topk,6,5,.013f,limited,false);++cases;
        }
    }
    std::printf("PASS %s packed verifier %d bitwise cases hidden=%d inter=%d\n",gguf?"IQ2_XS":"EXL3",cases,hidden,inter);
    ClearNumasMoeRuntimeCache();
}

int main() {
    try {
        EnableAMX(false);
        for (bool gguf : {false, true}) {
            CheckFormat(gguf, 1024, 1024);
            CheckFormat(gguf, 5120, 2304);
        }
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
