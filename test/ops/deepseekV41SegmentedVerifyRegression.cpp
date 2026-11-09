// Compare one segmented verification graph with independent causal graphs.
// Replay with different partitions and KV addresses; check per-request commits.
#include "fastllm.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-v41-decode.cuh"
#include "devices/cuda/fastllm-v41-full-decode.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <memory>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace fastllm;
static void Check(bool ok, const char *what) {
    if (!ok)
        throw std::runtime_error(what);
}
static void Cuda(cudaError_t e) {
    Check(e == cudaSuccess, cudaGetErrorString(e));
}
static float amplitude = .3f;
static double worstAttentionL2 = 0;
static void Fill(Data &d, const std::vector<int> &dims, int seed,
                 DataType type = DataType::BFLOAT16) {
    if (type == DataType::INT32 || type == DataType::INT8) {
        d.dataType = type;
        d.Resize(dims);
        d.Allocate();
        std::memset(d.cpuData, 0, d.GetBytes());
        d.ToDevice(DataDevice::CUDA, std::vector<int>{0});
        return;
    }
    size_t n = 1;
    for (int v : dims)
        n *= v;
    std::vector<float> values(n);
    float pattern[1009];
    for (int i = 0; i < 1009; ++i)
        pattern[i] = std::sin(float(i)) * amplitude;
    for (size_t i = 0; i < n; ++i)
        values[i] = pattern[(i * 17 + seed * 13) % 1009];
    Data cpu(DataType::FLOAT32, dims, values);
    ToDataType(cpu, type);
    d.CopyFrom(cpu);
    d.ToDevice(DataDevice::CUDA, std::vector<int>{0});
}
static void View(Data &v, const Data &d, size_t offset, const std::vector<int> &dims) {
    v.FakeFrom(d, offset);
    v.expansionDims.clear();
    v.Resize(dims);
    v.expansionDims = dims;
    v.expansionSize = v.Count(0);
    v.expansionBytes = v.GetBytes();
}
static void CopyDevice(Data &dst, const Data &src) {
    Cuda(cudaMemcpyAsync(dst.cudaData, src.cudaData, src.GetBytes(), cudaMemcpyDeviceToDevice,
                         cudaStreamPerThread));
}
static void Equal(const Data &a, const Data &b, const char *name) {
    Check(a.GetBytes() == b.GetBytes(), "shape changed");
    std::vector<unsigned char> x(a.GetBytes()), y(b.GetBytes());
    Cuda(cudaMemcpy(x.data(), a.cudaData, x.size(), cudaMemcpyDeviceToHost));
    Cuda(cudaMemcpy(y.data(), b.cudaData, y.size(), cudaMemcpyDeviceToHost));
    if (x != y)
        throw std::runtime_error(std::string("bits changed: ") + name);
}
struct Buffers {
    Data q, kv, out, raw, score, joined, scores, latent, keys, iq, iw, top, cand, norm, sink, saved, savedRaw, savedScore, requests;
    FastllmV41FullDecodeBuffers b;
    explicit Buffers(const FastllmV41FullDecodeConfig &c) {
        const int n = c.batch, blocks = (c.maxKeys + c.blockSize - 1) / c.blockSize;
        Fill(q, {1,n,c.heads,512}, 1); Fill(kv,{1,n,512},2); Fill(out,q.dims,3);
        Fill(raw,{1,n,512},4,FLOAT32); Fill(score,raw.dims,5,FLOAT32);
        Fill(joined,{n,std::max(1,c.ratio),512},6,FLOAT32); Fill(scores,joined.dims,7,FLOAT32);
        Fill(latent,{1,n,512},8); Fill(keys,{1,n,128},9); Fill(iq,{1,n,32,128},10);
        Fill(iw,{1,n,32},11,FLOAT32); Fill(top,{1,n,512},0,INT32); Fill(cand,{1,n,blocks},0,INT8);
        Fill(norm,{512},12,FLOAT32); Fill(sink,{c.heads},13,FLOAT32);
        Fill(saved,c.ringMode?std::vector<int>{1,n,c.ringRowBytes}:kv.dims,0,c.ringMode?INT8:BFLOAT16);
        Fill(savedRaw,{1,2*n,512},0,FLOAT32); Fill(savedScore,savedRaw.dims,0,FLOAT32);
        Fill(requests,{int(n*sizeof(FastllmV41FullDecodeRequest))},0,INT8);
        b.q=q.cudaData; b.kv=kv.cudaData; b.output=out.cudaData; b.rawKV=raw.cudaData; b.rawScore=score.cudaData;
        b.joined=joined.cudaData; b.scores=scores.cudaData; b.latents=latent.cudaData; b.newIndexKeys=keys.cudaData;
        b.indexQ=iq.cudaData; b.indexWeights=iw.cudaData; b.topk=top.cudaData; b.candidates=cand.cudaData;
        b.compressNorm=norm.cudaData; b.sink=sink.cudaData; b.savedKV=saved.cudaData;
        b.savedRawKV=savedRaw.cudaData; b.savedRawScore=savedScore.cudaData; b.requests=requests.cudaData;
    }
    void CopyInputs(const Buffers &source, int begin, int count) {
        for (auto pair : {std::make_pair(&q,&source.q),{&kv,&source.kv},{&raw,&source.raw},
                          {&score,&source.score},{&keys,&source.keys},{&iq,&source.iq},{&iw,&source.iw}}) {
            const size_t bytes = pair.second->GetBytes()/source.q.dims[1];
            Cuda(cudaMemcpyAsync(pair.first->cudaData,(char*)pair.second->cudaData+begin*bytes,
                count*bytes,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
        }
        CopyDevice(norm,source.norm); CopyDevice(sink,source.sink);
    }
};
struct Cache {
    Data ring, cmp, keys, tail, score;
    Cache(const FastllmV41FullDecodeConfig &c,int seed) {
        Fill(ring,{1,128,512},seed); Fill(cmp,{1,c.maxKeys,512},seed+1); Fill(keys,{1,c.maxKeys,128},seed+2);
        Fill(tail,{1,1,512},seed+3,FLOAT32); Fill(score,tail.dims,seed+4,FLOAT32);
        for (auto pair : {std::make_pair(&ring,c.ringMode),{&cmp,c.cmpMode},{&keys,c.indexMode}}) {
            if (!pair.second) continue;
            Data packed; Check(FastllmCudaDeepSeekV41QuantizeKV(*pair.first,packed,pair.second,pair.second==3?16:32),"pack");
            pair.first->CopyFrom(packed);
        }
    }
    FastllmV41FullDecodeRequest Request(int pos,int begin,int length) {
        return {ring.cudaData,cmp.cudaData,keys.cudaData,tail.cudaData,score.cudaData,pos,begin,length};
    }
};
static std::vector<unsigned char> Read(const Data &x) {
    std::vector<unsigned char> bytes(x.GetBytes());
    Cuda(cudaMemcpy(bytes.data(),x.cudaData,bytes.size(),cudaMemcpyDeviceToHost)); return bytes;
}
static void CompareAttention(const Data &all, const Data &one, int begin) {
    const auto a=Read(all),b=Read(one); const size_t offset=begin*b.size()/one.dims[1];
    const auto *x=reinterpret_cast<const __nv_bfloat16 *>(a.data()+offset),*y=reinterpret_cast<const __nv_bfloat16 *>(b.data());
    double error=0,norm=0;
    for (size_t i=0;i<b.size()/2;++i) { double u=__bfloat162float(x[i]),v=__bfloat162float(y[i]);
        Check(std::isfinite(u)&&std::isfinite(v),"nonfinite attention");error+=(u-v)*(u-v);norm+=v*v; }
    const double l2=std::sqrt(error/std::max(norm,1e-30));worstAttentionL2=std::max(worstAttentionL2,l2);
    Check(l2<.0032,"segmented attention L2");
}
static void Run(int packed,int ratio,int heads,int total) {
    FastllmV41FullDecodeConfig c;c.batch=total;c.heads=heads;c.verifying=c.segmentedVerify=true;
    c.ratio=ratio;c.kvSource=c.indexSource=ratio>0;c.maxKeys=2048;c.softmaxScale=1/std::sqrt(512.f);
    if(packed) {c.ringMode=1;c.ringRowBytes=528;c.cmpMode=packed;c.cmpRowBytes=packed==3?288:528;
        c.indexMode=packed==3?2:1;c.indexRowBytes=packed==3?68:132;}
    Buffers batch(c),initial(c);void *plan=FastllmCudaV41CreateFullDecode(c,batch.b);Check(plan,"segmented plan");
    cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    for(int replay=0;replay<4;++replay) {
        std::vector<int> lengths;
        if(total==12) lengths=replay%2?std::vector<int>{1,4,2,5}:std::vector<int>{6,6};
        else if(total==24) lengths=replay%2?std::vector<int>{5,7,8,4}:std::vector<int>{6,6,6,6};
        else lengths=replay%2?std::vector<int>{1,3}:std::vector<int>{2,2};
        batch.CopyInputs(initial,0,total);
        std::vector<std::unique_ptr<Cache>> cache;
        std::vector<FastllmV41FullDecodeRequest> requests;
        int begin=0;
        for(size_t i=0;i<lengths.size();++i) {
            cache.emplace_back(new Cache(c,29+i*37+replay*11));
            const int pos=(i%2?127:510)+replay;
            for(int row=0;row<lengths[i];++row) requests.push_back(cache.back()->Request(pos+row,begin,lengths[i]));
            begin+=lengths[i];
        }
        Check(FastllmCudaV41UploadFullDecodeBatch(batch.requests.cudaData,&plan,1,requests.data(),total),"upload");
        if(!graph) {
            Check(FastllmCudaV41PrepareFullDecode(plan)&&FastllmCudaV41FinishFullDecode(plan),"warm");
            Cuda(cudaDeviceSynchronize());batch.CopyInputs(initial,0,total);Cuda(cudaDeviceSynchronize());
            Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
            Check(FastllmCudaV41PrepareFullDecode(plan)&&FastllmCudaV41FinishFullDecode(plan),"capture");
            Cuda(cudaStreamEndCapture(cudaStreamPerThread,&graph));Cuda(cudaGraphInstantiate(&exec,graph,0));
        }
        Cuda(cudaGraphLaunch(exec,cudaStreamPerThread));Cuda(cudaDeviceSynchronize());begin=0;
        for(size_t i=0;i<lengths.size();++i) {
            auto single=c;single.batch=lengths[i];single.segmentedVerify=false;
            Buffers ref(single);ref.CopyInputs(initial,begin,single.batch);Cache expected(single,29+i*37+replay*11);
            std::vector<FastllmV41FullDecodeRequest> rr;
            for(int row=0;row<single.batch;++row)rr.push_back(expected.Request(requests[begin].position+row,0,single.batch));
            void *rp=FastllmCudaV41CreateFullDecode(single,ref.b);Check(rp,"single plan");
            Check(FastllmCudaV41UploadFullDecodeBatch(ref.requests.cudaData,&rp,1,rr.data(),rr.size()),"single upload");
            Check(FastllmCudaV41PrepareFullDecode(rp)&&FastllmCudaV41FinishFullDecode(rp),"single run");
            Cuda(cudaDeviceSynchronize());CompareAttention(batch.out,ref.out,begin);
            Equal(cache[i]->ring,expected.ring,"deferred ring");Equal(cache[i]->tail,expected.tail,"deferred tail");
            Equal(cache[i]->score,expected.score,"deferred score");
            Equal(cache[i]->cmp,expected.cmp,"compressed KV");Equal(cache[i]->keys,expected.keys,"index KV");
            const auto saved=Read(batch.saved),raw=Read(batch.savedRaw),scores=Read(batch.savedScore);
            auto ring=Read(cache[i]->ring),tail=Read(cache[i]->tail),tailScore=Read(cache[i]->score);
            const int accept=1+(replay+i)%single.batch,pos=requests[begin].position,bytes=c.ringRowBytes;
            for(int row=0;row<accept;++row)std::memcpy(ring.data()+((pos+row)%128)*bytes,saved.data()+(begin+row)*bytes,bytes);
            if(ratio==2&&(pos+accept)%2) {const int row=2*begin+pos%2+accept-1;
                std::memcpy(tail.data(),raw.data()+row*512*4,512*4);std::memcpy(tailScore.data(),scores.data()+row*512*4,512*4);}
            FastllmV41VerifyCommitLayer descriptor{static_cast<const FastllmV41FullDecodeRequest*>(batch.requests.cudaData),
                batch.saved.cudaData,batch.savedRaw.cudaData,batch.savedScore.cudaData,128,ratio,bytes};
            void *desc=nullptr;Cuda(cudaMalloc(&desc,sizeof(descriptor)));Cuda(cudaMemcpy(desc,&descriptor,sizeof(descriptor),cudaMemcpyHostToDevice));
            for (int invalidOffset : {-1, 2147483647})
                Check(!FastllmCudaV41CommitVerifiedPrefix(static_cast<const FastllmV41VerifyCommitLayer*>(desc),1,accept,single.batch,invalidOffset),"invalid commit offset accepted");
            Check(FastllmCudaV41CommitVerifiedPrefix(static_cast<const FastllmV41VerifyCommitLayer*>(desc),1,accept,single.batch,begin),"commit");
            Check(Read(cache[i]->ring)==ring&&Read(cache[i]->tail)==tail&&Read(cache[i]->score)==tailScore,"independent prefix commit");
            Cuda(cudaFree(desc));FastllmCudaV41DestroyFullDecode(rp);begin+=single.batch;
        }
        auto bad=requests;bad[0].chunkBegin=1;Check(!FastllmCudaV41UploadFullDecodeBatch(batch.requests.cudaData,&plan,1,bad.data(),total),"reject overlapping chunks");
    }
    Cuda(cudaGraphExecDestroy(exec));Cuda(cudaGraphDestroy(graph));FastllmCudaV41DestroyFullDecode(plan);
}
int main() {try {SetThreads(4);Cuda(cudaSetDevice(0));int cases=0;
    for(int packed:{0,1,3})for(int ratio:{0,1,2})for(int heads:{8,32})for(int total:{4,12,24}) {
        Run(packed,ratio,heads,total);++cases;
        std::cout<<"PASS packed="<<packed<<" ratio="<<ratio<<" heads="<<heads<<" rows="<<total<<std::endl;
    }
    std::cout<<"PASS cases="<<cases<<" replays="<<cases*4<<" worst_attention_l2="<<worstAttentionL2<<std::endl;return 0;
}catch(const std::exception &e){std::cerr<<"FAIL "<<e.what()<<std::endl;return 1;}}
