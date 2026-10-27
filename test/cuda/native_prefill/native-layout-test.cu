#include "fastllm-native-nvfp4-layout.cuh"
#include <vector>
#include <random>
#include <cstring>
#include <cstdio>
#include <cmath>
using namespace fastllm_native_layout;
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){printf("CUDA %s line %d\n",cudaGetErrorString(e),__LINE__);return 2;}}while(0)
int main(int argc, char **argv){
 int N=256,K=argc>1?atoi(argv[1]):256,M=8;size_t groups=size_t(N)*K/16,rawBytes=groups*12,nativeBytes=groups*9;
 std::mt19937 rng(123);std::vector<uint8_t>raw(rawBytes),back(rawBytes);std::vector<half>x(M*K),out(M*N),bias(N),initial(M*N);float global=.03125f,processed=global*128;
 for(size_t i=0;i<groups;i++){for(int j=0;j<8;j++)raw[i*12+j]=rng();__nv_fp8_e4m3 s(float((rng()%32)+1)/8);float f=float(s)*global;memcpy(raw.data()+i*12+8,&f,4);}
 for(auto &v:x)v=__float2half(float(int(rng()%200)-100)/100);
 for(auto &v:bias)v=__float2half(float(int(rng()%200)-100)/100);
 for(auto &v:initial)v=__float2half(float(int(rng()%200)-100)/100);
 uint8_t *dr,*dn,*restore;half *dx,*dy,*db;float *dg;int *bad;
 CK(cudaMalloc(&dr,rawBytes));CK(cudaMalloc(&dn,nativeBytes));CK(cudaMalloc(&restore,rawBytes));CK(cudaMalloc(&dx,x.size()*2));CK(cudaMalloc(&dy,out.size()*2));CK(cudaMalloc(&db,N*2));CK(cudaMalloc(&dg,4));CK(cudaMalloc(&bad,4));
 CK(cudaMemcpy(dr,raw.data(),rawBytes,cudaMemcpyHostToDevice));CK(cudaMemcpy(dx,x.data(),x.size()*2,cudaMemcpyHostToDevice));CK(cudaMemcpy(db,bias.data(),N*2,cudaMemcpyHostToDevice));CK(cudaMemcpy(dg,&processed,4,cudaMemcpyHostToDevice));CK(cudaMemset(bad,0,4));
 Pack<<<(groups+255)/256,256>>>(dr,dn,dn+N*K/2,N,K,global,bad);CK(cudaDeviceSynchronize());int invalid;CK(cudaMemcpy(&invalid,bad,4,cudaMemcpyDeviceToHost));if(invalid)return 3;
 Restore<<<(groups+255)/256,256>>>(dn,dn+N*K/2,dg,restore,N,K);CK(cudaMemcpy(back.data(),restore,rawBytes,cudaMemcpyDeviceToHost));if(back!=raw){puts("raw roundtrip mismatch");return 4;}puts("raw roundtrip exact");
 int errors=0;
 for(int m:{1,2,3,8})for(int mode:{0,1,2}){
  int width=mode==1?N/2:N;CK(cudaMemcpy(dy,initial.data(),initial.size()*2,cudaMemcpyHostToDevice));dim3 grid((width+127)/128*32,m);
#define LAUNCH_TEST(KSIZE) \
  if(mode==0)Gemv<0,8,KSIZE><<<dim3((width+127)/128*16,m),256>>>(dx,dn,dn+N*K/2,dg,db,dy,m,N,K); \
  if(mode==1)Gemv<1,8,KSIZE><<<dim3((width+127)/128*16,m),256>>>(dx,dn,dn+N*K/2,dg,db,dy,m,N,K); \
  if(mode==2)Gemv<2,8,KSIZE><<<dim3((width+127)/128*16,m),256>>>(dx,dn,dn+N*K/2,dg,db,dy,m,N,K);
  if(m==1 && K==5120) { LAUNCH_TEST(5120) }
  else if(m==1 && K==17408) { LAUNCH_TEST(17408) }
  else {
   if(mode==0)Gemv<0><<<grid,128>>>(dx,dn,dn+N*K/2,dg,db,dy,m,N,K);
   if(mode==1)Gemv<1><<<grid,128>>>(dx,dn,dn+N*K/2,dg,db,dy,m,N,K);
   if(mode==2)Gemv<2><<<grid,128>>>(dx,dn,dn+N*K/2,dg,db,dy,m,N,K);
  }
#undef LAUNCH_TEST
  CK(cudaDeviceSynchronize());CK(cudaMemcpy(out.data(),dy,m*width*2,cudaMemcpyDeviceToHost));int wrong=0;float maxe=0;
  auto dot=[&](int t,int row){double sum=0;for(int g=0;g<K/16;g++){float s;memcpy(&s,raw.data()+(row*K/16+g)*12+8,4);for(int p=0;p<8;p++){__nv_fp4x2_e2m1 q;q.__x=raw[(row*K/16+g)*12+p];float2 v=static_cast<float2>(q);sum+=double(__half2float(x[t*K+g*16+p*2]))*v.x*s+double(__half2float(x[t*K+g*16+p*2+1]))*v.y*s;}}return __hadd(__float2half(float(sum)),bias[row]);};
  for(int t=0;t<m;t++)for(int row=0;row<width;row++) {half v=dot(t,row);if(mode==1){half u=dot(t,row+width);v=__hmul(__hdiv(v,__hadd(__float2half(1),__float2half(expf(-__half2float(v))))),u);}if(mode==2)v=__hadd(v,initial[t*width+row]);float a=__half2float(v),b=__half2float(out[t*width+row]),e=fabs(a-b);maxe=fmax(maxe,e);wrong+=!std::isfinite(b)||e>.002f+.004f*fabs(a);}
  printf("M=%d mode=%d bad=%d max_abs=%g\n",m,mode,wrong,maxe);errors+=wrong;
 }
 // Same native layout works under graph capture for small-batch decode.
 cudaGraph_t graph;cudaGraphExec_t exec;CK(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));Gemv<0><<<dim3(64,1),128,0,cudaStreamPerThread>>>(dx,dn,dn+N*K/2,dg,db,dy,1,N,K);CK(cudaStreamEndCapture(cudaStreamPerThread,&graph));CK(cudaGraphInstantiate(&exec,graph,0));CK(cudaGraphLaunch(exec,cudaStreamPerThread));CK(cudaStreamSynchronize(cudaStreamPerThread));CK(cudaGraphExecDestroy(exec));CK(cudaGraphDestroy(graph));puts("graph replay passed");
 cudaFree(dr);cudaFree(dn);cudaFree(restore);cudaFree(dx);cudaFree(dy);cudaFree(db);cudaFree(dg);cudaFree(bad);
 return errors?1:0;
}
