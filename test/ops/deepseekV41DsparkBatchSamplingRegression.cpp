// Batched Markov proposals: conditional q and independent, fresh graph RNG.
#include "fastllm.h"
#include "utils.h"
#include <cstring>
#include "devices/cuda/fastllm-cuda-dspark.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace fastllm;
static void Check(bool ok, const char *msg) {
    if (!ok)
        throw std::runtime_error(msg);
}
static void Cuda(cudaError_t e) { Check(e == cudaSuccess, cudaGetErrorString(e)); }
struct Fixture {
    int vocab, rank, block, batch;
    int topK = 50; float topP = .95f; Data batchScores;
    float temperature;
    std::vector<float> e, h, x;
    Data embed, head, logits, tokens, probs, samplingScratch, out;
    Fixture(int v, int r, int b, DataType type, float t, int batches)
        : vocab(v), rank(r), block(b), batch(batches), temperature(t), e(v * r), h(v * r), x(batches * b * v),
          tokens(DataType::INT32, {batches * (b + 1)}) {
        for (int i = 0; i < v * r; ++i) {
            e[i] = ((i * 13 + i / r * 3) % 31 - 15) / 32.0f;
            h[i] = ((i * 7 + i / r * 5) % 29 - 14) / 32.0f;
        }
        for (int i = 0; i < batches * b * v; ++i)
            x[i] = ((i * 11 + i / v * 7) % 23 - 11) / 8.0f;
        embed.CopyFrom(Data(type, {v, r}, e));
        head.CopyFrom(Data(type, {v, r}, h));
        logits.CopyFrom(Data(FLOAT32, {batches, b, v}, x));
        for (Data *d : {&embed, &head, &logits, &tokens})
            d->ToDevice(DataDevice::CUDA, std::vector<int>{0});
        tokens.Allocate();
        Anchor(0);
    }
    void Anchor(int a) { std::vector<int> ids(batch*(block+1),-1);
        for(int b=0;b<batch;++b)ids[b*(block+1)]=(a+7*b)%vocab;
        Cuda(cudaMemcpy(tokens.cudaData,ids.data(),ids.size()*sizeof(int),cudaMemcpyHostToDevice)); }
    bool Run() { return FastllmCudaDeepSeekV41MarkovFilteredSampleBatchDevice(logits, embed, head,
            block, temperature, topK, topP, tokens, probs, out, batchScores, samplingScratch); }
    std::vector<int> Tokens() {
        std::vector<int> t(batch * (block + 1));
        Cuda(cudaMemcpy(t.data(), tokens.cudaData, t.size() * sizeof(int), cudaMemcpyDeviceToHost));
        return t;
    }
    std::vector<double> Q(int step, int previous, int request) {
        std::vector<double> q(vocab);
        double maximum = -INFINITY, total = 0;
        for (int v = 0; v < vocab; ++v) {
            double dot = 0;
            for (int d = 0; d < rank; ++d)
                dot += (double)e[previous * rank + d] * h[v * rank + d];
            float projection = float(dot);
            if ( embed.dataType == FLOAT16) projection = half_to_float(float_to_half(projection));
            if ( embed.dataType == BFLOAT16) {
                uint32_t bits = uint32_t(Float32ToBFloat16RNEBits(projection)) << 16;
                std::memcpy(&projection, &bits, sizeof(projection));
            }
            q[v] = x[(request * block + step) * vocab + v] + projection;
            maximum = std::max(maximum, q[v]);
        }
        for (double &v : q) {
            v = std::exp((v - maximum) / std::max(temperature, 1.0e-6f));
            total += v;
        }
        for (double &v : q)
            v /= total;
        auto sorted=q;std::sort(sorted.rbegin(),sorted.rend());
        double cumulative=0,cutoff=0;
        for(double p:sorted){cumulative+=p;cutoff=p;if(cumulative>=topP)break;}
        if(topP>=1)cutoff=0;
        for(double &p:q)if(p<cutoff)p=0;
        total=0;for(double p:q)total+=p;for(double&p:q)p/=total;
        sorted=q;std::sort(sorted.rbegin(),sorted.rend());
        cutoff=(topK>0&&topK<vocab)?sorted[topK-1]:0;
        total=0;for(double&p:q){if(p<cutoff)p=0;total+=p;}for(double&p:q)p/=total;
        return q;
    }
    void Numeric() {
        auto t = Tokens();
        std::vector<float> p(batch * block * vocab), embeddings(batch * block * rank);
        Cuda(cudaMemcpy(p.data(), probs.cudaData, p.size() * sizeof(float), cudaMemcpyDeviceToHost));
        Cuda(cudaMemcpy(embeddings.data(), out.cudaData, embeddings.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
        for(int request=0;request<batch;++request)for (int s = 0; s < block; ++s) {
            const int row=request*block+s, tokenRow=request*(block+1)+s;
            Check(t[tokenRow + 1] >= 0 && t[tokenRow + 1] < vocab, "sampled token out of range");
            auto q = Q(s, t[tokenRow], request);
            double sum = 0, l1 = 0;
            for (int v = 0; v < vocab; ++v) {
                sum += p[row * vocab + v];
                l1 += std::abs(p[row * vocab + v] - q[v]);
            }
            if (std::abs(sum - 1) >= 1e-5 || l1 >= 1e-5)
                std::cerr << "dtype=" << embed.dataType << " vocab=" << vocab << " block=" << block
                          << " step=" << s << " anchor=" << t[tokenRow] << " sum=" << sum << " L1=" << l1
                          << " q0=" << p[row * vocab] << " ref0=" << q[0] << '\n';
            Check(std::abs(sum - 1) < 1e-5 && l1 < 1e-5, "stored q differs from conditional softmax");
            for (int d = 0; d < rank; ++d) {
                if(embeddings[row * rank + d] != e[t[tokenRow] * rank + d])
                    std::cerr << "embedding mismatch dtype=" << embed.dataType << " request=" << request
                              << " step=" << s << " token=" << t[tokenRow] << " d=" << d
                              << " actual=" << embeddings[row*rank+d] << " expected=" << e[t[tokenRow]*rank+d] << '\n';
                Check(embeddings[row * rank + d] == e[t[tokenRow] * rank + d], "wrong conditional embedding");
            }
        }
    }
};

static void RunCase(int batch,int block,int vocab,int rank,DataType type,int k,float p){
    Fixture f(vocab,rank,block,type,.8f,batch);f.topK=k;f.topP=p;
    Check(f.Run(),"batch eager");f.Numeric();Cuda(cudaDeviceSynchronize());
    cudaGraph_t g;cudaGraphExec_t e;Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
    Check(f.Run(),"batch capture");Cuda(cudaStreamEndCapture(cudaStreamPerThread,&g));Cuda(cudaGraphInstantiate(&e,g,nullptr,nullptr,0));
    for(int pass=0;pass<4;++pass){
        f.Anchor(pass*3);std::rotate(f.x.begin(),f.x.begin()+1,f.x.end());
        Cuda(cudaMemcpy(f.logits.cudaData,f.x.data(),f.x.size()*sizeof(float),cudaMemcpyHostToDevice));
        Cuda(cudaGraphLaunch(e,cudaStreamPerThread));f.Numeric();
    }
    Cuda(cudaGraphExecDestroy(e));Cuda(cudaGraphDestroy(g));
}
int main(){try{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    Cuda(cudaSetDevice(0));int cases=0;
    for(auto type:{FLOAT32,FLOAT16,BFLOAT16})for(int batch:{1,2,3,8,17})for(int block:{1,5,9}){
        RunCase(batch,block,37,17,type,11,.8f);++cases;
    }
    for(int batch:{2,8}){RunCase(batch,5,32769,256,BFLOAT16,50,.95f);++cases;}
    RunCase(4,5,129280,256,BFLOAT16,50,.95f);++cases;
    Fixture f(17,7,3,FLOAT32,1,8);f.topK=17;f.topP=1;
    std::fill(f.e.begin(),f.e.end(),0);std::fill(f.h.begin(),f.h.end(),0);std::fill(f.x.begin(),f.x.end(),0);
    for(Data*d:{&f.embed,&f.head,&f.logits})Cuda(cudaMemset(d->cudaData,0,d->GetBytes()));
    Check(f.Run(),"rng warmup");cudaGraph_t g;cudaGraphExec_t e;
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));Check(f.Run(),"rng capture");
    Cuda(cudaStreamEndCapture(cudaStreamPerThread,&g));Cuda(cudaGraphInstantiate(&e,g,nullptr,nullptr,0));
    int changed=0,equal=0,previous=-1;
    for(int i=0;i<256;++i){Cuda(cudaGraphLaunch(e,cudaStreamPerThread));auto t=f.Tokens();changed+=t[1]!=previous;previous=t[1];equal+=t[1]==t[5];}
    Check(changed>200&&equal<40,"RNG reused across replays or requests");
    Cuda(cudaGraphExecDestroy(e));Cuda(cudaGraphDestroy(g));
    std::cout<<"BATCH_SAMPLING_PASS cases="<<cases<<" replays="<<cases*4<<" RNG_changes="<<changed<<" equal_rows="<<equal<<std::endl;
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}}
