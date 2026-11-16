#include "../../src/devices/cuda/models/naive-n05-topk-compact-select.cuh"
#include <climits>
#include <functional>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <limits>
#include "../../src/devices/cuda/models/naive-n05-topk-compact-sort.cuh"
#include <vector>
#include <cstdio>
#include <stdexcept>
#include "../../src/devices/cuda/models/naive-n05-indexer-batch.cuh"
#include "../../src/devices/cuda/models/naive-n05-indexer-sortpack.cuh"
using namespace naive_indexer_batch;
namespace sp = naive_indexer_sortpack;
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> T *alloc(size_t n) { T *p; ck(cudaMalloc(&p,n*sizeof(T))); return p; }
static void TestCompactSort() {
    constexpr int rows=12, capacity=4096, topK=2048, keyCount=69632;
    auto input=alloc<unsigned long long>(rows*capacity);
    auto output=alloc<int>(rows*topK), ends=alloc<int>(rows*2);
    std::vector<unsigned long long> values(rows*capacity);
    std::vector<int> actual(rows*topK), offsets(rows*2);
    auto ordered=[](float value) {
        unsigned bits;std::memcpy(&bits,&value,4);if(value==0.f)bits=0;
        return bits>>31?~bits:bits^0x80000000u;
    };
    for(int count : {0,1,31,2048,2049,3072,3073,4096}) {
        unsigned seed=71;
        std::vector<std::vector<std::pair<unsigned,int>>> expected(rows);
        for(int row=0;row<rows;++row) {
            offsets[row]=row*capacity;offsets[rows+row]=row*capacity+count;
            for(int col=0;col<capacity;++col) {
                seed=seed*1664525u+1013904223u;
                float score=32.f+float(seed>>8)/16777216.f*8.f;
                if(row==1)score=0;
                if(row==2)score=col%64;
                if(row==3)score=col%2048;
                if(row==4)score=col/2;
                if(row==5)score=-score;
                if(row==6)score=col%2?-0.f:0.f;
                if(row==7&&col%7==0)score=-INFINITY;
                if(row==8&&col%7==0)score=INFINITY;
                if(row==9&&col%7==0)score=std::numeric_limits<float>::quiet_NaN();
                if(row==10)score=col<2031?64.f:col<2063?32.f:16.f;
                if(row==11)score=col%2?score:-score;
                unsigned bits;std::memcpy(&bits,&score,4);int index=(col*17+row)%keyCount;
                values[size_t(row)*capacity+col]=(static_cast<unsigned long long>(bits)<<32)|(0xffffffffu-unsigned(index));
                if(col<count)expected[row].emplace_back(ordered(score),index);
            }
            std::sort(expected[row].begin(),expected[row].end(),[](auto a,auto b){return a.first==b.first?a.second<b.second:a.first>b.first;});
        }
        ck(cudaMemcpy(input,values.data(),values.size()*8,cudaMemcpyHostToDevice));
        ck(cudaMemcpy(ends,offsets.data(),offsets.size()*4,cudaMemcpyHostToDevice));
        for(int threads : {128,256}) {
            if(threads==128)naive_topk::SortTopKCompact<true,128><<<rows,128>>>(input,ends,output,rows,capacity,topK,keyCount);
            else naive_topk::SortTopKCompact<true><<<rows,256>>>(input,ends,output,rows,capacity,topK,keyCount);
            ck(cudaGetLastError());ck(cudaDeviceSynchronize());
            ck(cudaMemcpy(actual.data(),output,actual.size()*4,cudaMemcpyDeviceToHost));
            for(int row=0;row<rows;++row)for(int col=0;col<topK;++col) {
                int want=col<count?expected[row][col].second:-1;
                if(actual[size_t(row)*topK+col]!=want)throw std::runtime_error("Compact sort differs from exact CPU score/index order");
            }
        }
    }
    ck(cudaFree(input));ck(cudaFree(output));ck(cudaFree(ends));
}


static void TestCompactSelection() {
    constexpr int rows=17, capacity=4096, topK=3072;
    auto output=alloc<unsigned long long>(size_t(rows)*capacity);
    auto ends=alloc<int>(rows*2);
    for(int keys : {8192,16384,32768,65536,65537,131072}) {
        auto input=alloc<float>(size_t(rows)*keys);
        std::vector<float> values(size_t(rows)*keys);
        std::vector<unsigned long long> expected(size_t(rows)*capacity), actual(expected.size());
        std::vector<int> expectedEnds(rows*2), actualEnds(rows*2);
        const int past=keys-rows;
        for(int pattern=0;pattern<7;++pattern) {
            unsigned seed=73;
            std::vector<std::vector<unsigned long long>> winners(rows);
            for(int r=0;r<rows;++r) {
                const int count=past+r+1;
                float *v=values.data()+size_t(r)*keys;
                for(int i=0;i<keys;++i) {
                    seed=seed*1664525u+1013904223u;
                    v[i]=float(seed>>8)/16777216.f;
                    if(pattern==2)v[i]=7.f;
                    if(pattern==3)v[i]=i<3100?2.f:1.f;
                    if(pattern==4)v[i]=-(32.f+v[i]);
                    if(pattern==5) {
                        unsigned bits[]={0x7fc00002u,0x7fc00001u,0x7f800000u,0x3f800000u,0u,
                            0x80000000u,0x80000001u,0xbf800000u,0xff800000u,0xffc00001u,0xffffffffu};
                        std::memcpy(v+i,bits+(i*5+r)%11,4);
                    }
                    if(pattern==6)v[i]=i%2?0.f:-0.f;
                }
                if(pattern<2)for(int i=0;i<2048;++i)
                    v[int(int64_t(i)*count/2048)]=pattern==0?10000.f:-10000.f;
                for(int i=0;i<count;++i) {
                    unsigned bits;std::memcpy(&bits,v+i,4);if(v[i]==0.f)bits=0;
                    bits=bits>>31?~bits:bits^0x80000000u;
                    winners[r].push_back((static_cast<unsigned long long>(bits)<<32)|(0xffffffffu-unsigned(i)));
                }
                std::partial_sort(winners[r].begin(),winners[r].begin()+topK,winners[r].end(),std::greater<unsigned long long>());
                winners[r].resize(topK);
            }
            ck(cudaMemcpy(input,values.data(),values.size()*4,cudaMemcpyHostToDevice));
            for(int mode=0;mode<2;++mode) {
                auto launch=[&]{
                    if(mode==0)naive_topk::SelectTopKCompact<true,4,512><<<rows,512>>>(input,output,ends,rows,keys,past,topK,capacity);
                    else naive_topk::SelectTopKCompact<true,2,1024><<<rows,1024>>>(input,output,ends,rows,keys,past,topK,capacity);
                };
                if(keys==65537 && mode==1) {
                    cudaGraph_t graph;cudaGraphExec_t exec;
                    ck(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));launch();
                    ck(cudaStreamEndCapture(cudaStreamPerThread,&graph));ck(cudaGraphInstantiate(&exec,graph,0));
                    ck(cudaGraphLaunch(exec,cudaStreamPerThread));ck(cudaStreamSynchronize(cudaStreamPerThread));
                    ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));
                } else launch();
                ck(cudaGetLastError());ck(cudaDeviceSynchronize());
                ck(cudaMemcpy(actual.data(),output,actual.size()*8,cudaMemcpyDeviceToHost));
                ck(cudaMemcpy(actualEnds.data(),ends,actualEnds.size()*4,cudaMemcpyDeviceToHost));
                for(int r=0;r<rows;++r) {
                    int n=actualEnds[rows+r]-r*capacity;
                    if(n<topK||n>capacity)throw std::runtime_error("invalid compact candidate bounds");
                    auto first=actual.begin()+size_t(r)*capacity;
                    std::sort(first,first+n,std::greater<unsigned long long>());
                    if(!std::equal(winners[r].begin(),winners[r].end(),first))
                        throw std::runtime_error("compact selector lost an exact CPU TopK winner");
                    if(mode&&(actualEnds!=expectedEnds||!std::equal(first,first+n,expected.begin()+size_t(r)*capacity)))
                        throw std::runtime_error("wide selector changed refinement candidate set");
                }
                if(!mode){expected=actual;expectedEnds=actualEnds;}
            }
        }
        ck(cudaFree(input));
    }
    ck(cudaFree(output));ck(cudaFree(ends));
    puts("PASS: wide compact selection CPU TopK and original candidate-set equality, tails/ties/NaN/graph");
}

int main() {
 try {
  int devices=0; if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
  cudaDeviceProp prop;ck(cudaGetDeviceProperties(&prop,0));if(prop.major<7)return 77;
  TestCompactSelection();
  TestCompactSort();
  for (int count : {1,17,BatchLimit}) for (int length : {0,31,2047,2048,2301,4097,16384}) {
   const int stride=std::max(1,length); Batch batch{};
   auto scores=alloc<unsigned>(size_t(count)*stride); auto live=alloc<int>(count);
   auto work=alloc<naive_topk::Workspace>(count); auto indices=alloc<int>(size_t(count)*2048);
   std::vector<int> sizes(count,length), result(size_t(count)*2048), expected(result.size(),-1);
   std::vector<unsigned> host(size_t(count)*stride);
   for(int r=0;r<count;++r) {
    sizes[r]=std::max(0,length-r%7);
    unsigned state=71+r; std::vector<int> order(sizes[r]); std::iota(order.begin(),order.end(),0);
    for(int j=0;j<sizes[r];++j) {
     state=state*1664525u+1013904223u;
     // Uniform rows, heavy boundary ties, narrow score ranges and arbitrary bits.
     host[size_t(r)*stride+j]=r%4==0?0x80000000u:r%4==1?0x80000000u+(state%11):r%4==2?0x80000000u+(state%65536):state;
    }
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return host[size_t(r)*stride+a]>host[size_t(r)*stride+b];});
    for(int j=0;j<std::min(2048,sizes[r]);++j)expected[size_t(r)*2048+j]=order[j];
    batch.rows[r].scores=scores+size_t(r)*stride;batch.rows[r].live=live+r;
    batch.rows[r].workspace=work+r;batch.rows[r].indices=indices+size_t(r)*2048;
   }
   ck(cudaMemcpy(scores,host.data(),host.size()*4,cudaMemcpyHostToDevice));
   ck(cudaMemcpy(live,sizes.data(),count*4,cudaMemcpyHostToDevice));
   for(int mode=0;mode<3;++mode) {
    auto run=[&] {
     if(mode==0) { Select<><<<count,256>>>(batch); sp::SortPack<256,5,true,false><<<count,256,sp::SortBytes<256,5,true,false>()>>>(batch); }
     else if(mode==1) sp::SelectSort<4096><<<count,256,sp::SortBytes<256,5,true,false,4096>()>>>(batch);
     else if(mode==2) sp::SelectSort<2048><<<count,256,sp::SortBytes<256,5,true,false,2048>()>>>(batch);
    };
    run();ck(cudaGetLastError());ck(cudaDeviceSynchronize());
    ck(cudaMemcpy(result.data(),indices,result.size()*4,cudaMemcpyDeviceToHost));
    if(result!=expected) {
     size_t i=0;while(i<result.size()&&result[i]==expected[i])++i;
     std::fprintf(stderr,"FAIL count=%d len=%d mode=%d row=%zu slot=%zu got=%d expected=%d\n",count,length,mode,i/2048,i%2048,result[i],expected[i]);return 1;
    }
    cudaGraph_t graph;cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));run();ck(cudaStreamEndCapture(cudaStreamPerThread,&graph));ck(cudaGraphInstantiate(&exec,graph,0));
    // Reuse the executable after changing live lengths, including empty rows.
    for (int replay=0;replay<2;++replay) {
     for (int r=0;r<count;++r) sizes[r]=replay ? std::max(0,length-r%7) : std::min(31,std::max(0,length-r%7));
     std::fill(expected.begin(),expected.end(),-1);
     for (int r=0;r<count;++r) {
      std::vector<int> order(sizes[r]);std::iota(order.begin(),order.end(),0);
      std::stable_sort(order.begin(),order.end(),[&](int a,int b){return host[size_t(r)*stride+a]>host[size_t(r)*stride+b];});
      for(int j=0;j<std::min(2048,sizes[r]);++j)expected[size_t(r)*2048+j]=order[j];
     }
     ck(cudaMemcpy(live,sizes.data(),count*4,cudaMemcpyHostToDevice));
     ck(cudaGraphLaunch(exec,cudaStreamPerThread));ck(cudaStreamSynchronize(cudaStreamPerThread));
     ck(cudaMemcpy(result.data(),indices,result.size()*4,cudaMemcpyDeviceToHost));
     if(result!=expected)throw std::runtime_error("TopK graph replay differs from stable CPU sort");
    }
    ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));
   }
   ck(cudaFree(scores));ck(cudaFree(live));ck(cudaFree(work));ck(cudaFree(indices));
  }
  std::puts("PASS: exact fused TopK and dynamic graph replay");
 } catch(const std::exception&e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
