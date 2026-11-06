#include <climits>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cstdio>
#include <stdexcept>
#include "../../src/devices/cuda/models/naive-n05-indexer-batch.cuh"
#include "../../src/devices/cuda/models/naive-n05-indexer-sortpack.cuh"
using namespace naive_indexer_batch;
namespace sp = naive_indexer_sortpack;
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> T *alloc(size_t n) { T *p; ck(cudaMalloc(&p,n*sizeof(T))); return p; }
int main() {
 try {
  int devices=0; if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
  cudaDeviceProp prop;ck(cudaGetDeviceProperties(&prop,0));if(prop.major<7)return 77;
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
