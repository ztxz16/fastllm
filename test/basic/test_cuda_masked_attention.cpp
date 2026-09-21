#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "utils/utils.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace fastllm;
static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static float H(float x) { return half_to_float(float_to_half(x)); }
static float Q(int h, int r, int d) { return H(((h * 17 + r * 13 + d * 7) % 53 - 26) / 32.f); }
static float K(int h, int t, int d) { return H(((h * 11 + t * 7 + d * 3) % 31 - 15) / 32.f); }
static float V(int h, int t, int d) { return H(((h * 3 + t * 13 + d * 5) % 43 - 21) / 32.f); }
static bool Mask(int b, int r, int t, int keys, int rows, bool allMasked) {
    return allMasked || t > keys - rows + r || (t + b * 3) % 7 == 0;
}
template<class F> static void Init(Data &x, int heads, int rows, int dim, int capacity, F f) {
    x.Expansion({heads, capacity, dim}); x.Resize({heads, rows, dim});
    auto *p=(uint16_t *)x.cpuData;
    std::fill_n(p,x.expansionBytes/2,float_to_half(std::numeric_limits<float>::quiet_NaN()));
    for(int h=0;h<heads;h++)for(int r=0;r<rows;r++)for(int d=0;d<dim;d++)
        p[h*x.strides[0]+r*x.strides[1]+d]=float_to_half(f(h,r,d));
    x.ToDevice(DataDevice::CUDA,{0},true);
}
static void Run(int rows,int keys,int dim,int group,int batches,bool paddedQ,bool allMasked,
                std::ofstream *dump,bool bench) {
    int kvHeads=2*batches,heads=kvHeads*group;
    Data q(FLOAT16),k(FLOAT16),v(FLOAT16),mask(FLOAT16),out(FLOAT16);
    Init(q,heads,rows,dim,rows+(paddedQ?3:0),Q);
    Init(k,kvHeads,keys,dim,keys+17,K);
    Init(v,kvHeads,keys,dim,keys+31,V);
    Init(mask,batches,rows,keys,rows,[&](int b,int r,int t){return Mask(b,r,t,keys,rows,allMasked)?1.f:0.f;});
    Init(out,heads,rows,dim,rows,[](int,int,int){return 0.f;});
    float scale=1.f/std::sqrt((float)dim);
    auto run=[&]{Require(FastllmCudaHalfAttention(q,k,v,mask,out,group,scale,1),"attention rejected");};
    run();
    std::vector<uint16_t> actual(heads*rows*dim);
    Require(cudaMemcpy(actual.data(),out.cudaData,actual.size()*2,cudaMemcpyDeviceToHost)==cudaSuccess,"copy failed");
    if(dump)dump->write((char *)actual.data(),actual.size()*2);
    // Independent FP64 oracle; reproduce only the public FP16 score and
    // probability boundaries, without using the CUDA reduction or GEMM.
    double maxError=0;
    for(int h=0;h<heads;h++)for(int r=0;r<rows;r++) {
        std::vector<double> scores(keys);double maximum=-1e30;
        for(int t=0;t<keys;t++) {
            double dot=0;for(int d=0;d<dim;d++)dot+=(double)Q(h,r,d)*K(h/group,t,d);
            scores[t]=Mask(h/(heads/batches),r,t,keys,rows,allMasked)?-10000.:H(dot*H(scale));
            maximum=std::max(maximum,scores[t]);
        }
        double sum=0;for(double &x:scores){x=std::exp(x-maximum);sum+=x;}
        for(double &x:scores)x=H(x/sum);
        for(int d=0;d<dim;d++) {
            double ref=0;for(int t=0;t<keys;t++)ref+=scores[t]*V(h/group,t,d);
            double got=half_to_float(actual[(h*rows+r)*dim+d]);
            maxError=std::max(maxError,std::abs(got-ref));
            Require(std::isfinite(got)&&std::abs(got-ref)<0.002+0.003*std::abs(ref),"masked attention differs from reference");
        }
    }
    double ms=0;
    if(bench){
        for(int i=0;i<8;i++)run();cudaDeviceSynchronize();
        auto start=std::chrono::steady_clock::now();
        for(int i=0;i<100;i++)run();cudaDeviceSynchronize();
        ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/100;
    }
    std::printf("rows=%d keys=%d dim=%d group=%d batch=%d padded_q=%d all_masked=%d max_error=%.8g wall_ms=%.6f PASS\n",rows,keys,dim,group,batches,paddedQ,allMasked,maxError,ms);
}
int main(int argc,char **argv) {
    try {
        if(FastllmCudaGetDeviceCount()<1)return 77;
        FastllmCudaSetDevice(0);SetThreads(2);
        std::ofstream dump;
        if(argc>1)dump.open(argv[1],std::ios::binary);
        struct Case{int rows,keys,dim,group,batch;bool padded,masked;};
        Case cases[]={{1,2049,128,6,1,false,false},{2,1023,64,2,2,false,false},
            {3,1024,128,1,1,false,false},{5,2049,256,6,1,false,false},
            {8,4097,128,8,1,false,false},{16,513,32,2,2,false,false},
            {33,257,64,1,1,false,false},{5,2049,128,6,1,true,false},
            {3,31,64,2,2,false,true},{2,7,32,2,1,false,false},
            {2,63,32,2,1,false,false},{2,511,32,2,1,false,false},
            {9,8193,128,6,1,false,false}};
        for(auto c:cases)Run(c.rows,c.keys,c.dim,c.group,c.batch,c.padded,c.masked,dump.is_open()?&dump:nullptr,c.rows>1&&c.keys>=1024);
    }catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
