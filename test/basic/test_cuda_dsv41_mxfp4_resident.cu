#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "../../src/devices/cuda/moe/fastllm-moe-deepseekv41-cache.cuh"
static cudaError_t LaunchGate(bool resident, const __nv_bfloat16 *input,
        const int32_t *slots, const uint8_t *weights, const float *scores,
        __nv_bfloat16 *output, int hidden, int inter, int topk, size_t stride,
        cudaStream_t stream) {
    namespace op = fastllm::cuda::dsv41_cache;
    const dim3 grid(inter / 32, topk);
    if (resident)
        op::GateResident<true><<<grid, 512, 0, stream>>>(
            input, slots, weights, scores, output, hidden, inter, stride, 10);
    else
        op::Gate<<<grid, 512, 0, stream>>>(
            input, slots, weights, scores, output, hidden, inter, stride, 10);
    return cudaGetLastError();
}
#define CK(x) do { cudaError_t e=(cudaError_t)(x); if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} } while(0)
template<class T> T* alloc(size_t n) { T* p; CK(cudaMalloc(&p,n*sizeof(T))); return p; }
int main() {
    int devices = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (status == cudaErrorNoDevice || (status == cudaSuccess && !devices)) return 77;
    CK(status);
    int device=0; cudaDeviceProp prop{};
    CK(cudaGetDevice(&device)); CK(cudaGetDeviceProperties(&prop,device));
    if(prop.major!=12 || prop.minor!=0) return 77;
    std::mt19937 rng(771);
    std::uniform_real_distribution<float> val(-1,1);
    const int shapes[][3]={{256,32,1},{256,256,3},{768,512,6},{1280,256,16},{5120,2304,6}};
    for(auto &sh:shapes) {
        int H=sh[0],J=sh[1],K=sh[2];
        size_t gate=size_t(2)*J*(H/32)*17, stride=gate+size_t(H)*(J/32)*17;
        std::vector<unsigned char> wc(K*stride),wp(K*stride);
        for(size_t n=0;n<wc.size();++n) wc[n]=(n%17==16)?113+rng()%11:rng()%256;
        for(int k=0;k<K;++k) for(int part=0;part<2;++part) {
            size_t begin=part?gate:0, bytes=part?stride-gate:gate, blocks=bytes/17;
            for(size_t b=0;b<blocks;++b) {
                memcpy(wp.data()+k*stride+begin+b*16,wc.data()+k*stride+begin+b*17,16);
                wp[k*stride+begin+blocks*16+b]=wc[k*stride+begin+b*17+16];
            }
        }
        auto *w=alloc<unsigned char>(wc.size()),*p=alloc<unsigned char>(wp.size());
        CK(cudaMemcpy(w,wc.data(),wc.size(),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(p,wp.data(),wp.size(),cudaMemcpyHostToDevice));
        auto *x=alloc<__nv_bfloat16>(H),*y=alloc<__nv_bfloat16>(K*J),*gold=alloc<__nv_bfloat16>(K*J);
        auto *scores=alloc<float>(K); auto *slots=alloc<int>(K);
        std::vector<__nv_bfloat16> hx(H),hy(K*J),hg(K*J);
        std::vector<float> hs(K); std::vector<int> hi(K);
        cudaStream_t stream;CK(cudaStreamCreate(&stream));
        auto inputs=[&](int active,bool reverse) {
            for(auto &a:hx) a=__float2bfloat16(val(rng));
            for(int k=0;k<K;++k){hs[k]=(val(rng)+1)*.5f/K;hi[k]=k<active?(reverse?K-1-k:k):-1;}
            CK(cudaMemcpy(x,hx.data(),H*2,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(scores,hs.data(),K*4,cudaMemcpyHostToDevice));
            CK(cudaMemcpy(slots,hi.data(),K*4,cudaMemcpyHostToDevice));
        };
        auto call = [&](bool resident) {
            CK(LaunchGate(resident, x, slots, resident ? p : w, scores,
                resident ? y : gold, H, J, K, stride, stream));
        };
        auto reset=[&](){CK(cudaMemsetAsync(y,0x32,K*J*2,stream));CK(cudaMemsetAsync(gold,0x32,K*J*2,stream));};
        auto equal=[&](){
            CK(cudaStreamSynchronize(stream));
            CK(cudaMemcpy(hy.data(),y,K*J*2,cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hg.data(),gold,K*J*2,cudaMemcpyDeviceToHost));
            if(memcmp(hy.data(),hg.data(),K*J*2)){
                int mismatches=0;double sq=0,ref=0;float mx=0;
                for(int i=0;i<K*J;++i){float a=__bfloat162float(hy[i]),b=__bfloat162float(hg[i]);if(a!=b)++mismatches;sq+=(a-b)*(a-b);ref+=b*b;mx=std::max(mx,std::abs(a-b));}
                fprintf(stderr,"MISMATCH H%d I%d K%d count%d max%g l2%g\n",H,J,K,mismatches,mx,sqrt(sq/std::max(ref,1e-30)));exit(2);
            }
        };
        // Missing routes must leave the prefilled output untouched.
        for (int active : {K, std::max(1, K / 2), 0}) {
            inputs(active, false); reset(); call(false); call(true); equal();
        }
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        call(true);
        CK(cudaStreamEndCapture(stream, &graph));
        CK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (int replay = 0; replay < 3; ++replay) {
            inputs(std::max(0, K - replay), true); reset(); call(false);
            CK(cudaGraphLaunch(exec, stream)); equal();
        }
        CK(cudaGraphExecDestroy(exec));
        CK(cudaGraphDestroy(graph));
        printf("PASS H%d I%d K%d bitwise and graph\n",H,J,K);fflush(stdout);
        CK(cudaStreamDestroy(stream));for(void* a:{(void*)w,(void*)p,(void*)x,(void*)y,(void*)gold,(void*)scores,(void*)slots})CK(cudaFree(a));
    }
    printf("PLANAR_RESIDENT_PASS\n"); return 0;
}
