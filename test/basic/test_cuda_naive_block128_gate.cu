// Standalone SM103 regression: exact routing, changing Graph inputs, expert
// membership, and gate output against the existing MXFP8/32 implementation.
#include "../../src/devices/cuda/moe/fastllm-shared-flex-tp.cu"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include <random>
#include <cstring>
using namespace fastllm_shared_flex_tp;
namespace b128 = fastllm_shared_flex_tp::block128_gate_detail;
using Args = Block128GateArgs;
void need(bool v,const char*s){if(!v)throw std::runtime_error(s);}
__device__ unsigned randomBits(unsigned x){x=x*747796405u+2891336453u;return ((x>>((x>>28)+4))^x)*277803737u;}
__global__ void fillWeights(uint8_t*w,uint8_t*s,unsigned seed){
 for(size_t i=blockIdx.x*256+threadIdx.x;i<size_t(256)*512*4096;i+=size_t(gridDim.x)*256){
  unsigned v=randomBits(unsigned(i)+seed);w[i]=__nv_fp8_e4m3(float(int(v%15)-7)).__x;}
 for(size_t i=blockIdx.x*256+threadIdx.x;i<size_t(256)*512*128;i+=size_t(gridDim.x)*256)
  s[i]=119+randomBits(unsigned(i)+seed)%5;
}
__global__ void fillInput(Rank a,unsigned seed){
 int i=blockIdx.x*256+threadIdx.x;
 if(i<32768)a.xq[i]=__nv_fp8_e4m3(float(int(randomBits(i+seed)%15)-7)).__x;
 if(i<1024)a.xs[i]=121+randomBits(i+seed)%4;
}
__global__ void prepare128(const uint8_t*x,const uint8_t*xs,uint8_t*out,int rows){
  int row=blockIdx.x/32,kb=blockIdx.x%32,t=threadIdx.x;
  __nv_fp8_e4m3 v;v.__x=row<rows?x[row*4096+kb*128+t]:0;
  unsigned old=row<rows?xs[row*128+kb*4+t/32]:127;
  __shared__ unsigned ex[4];
  if(t%32==0)ex[t/32]=old;__syncthreads();
  unsigned e=__reduce_max_sync(0xffffffff,t%32<4?ex[t%32]:0);
  float scaled=float(v)*sf(127+int(old)-int(e));
  out[row*4112+kb*128+t]=__nv_fp8_e4m3(scaled).__x;
  if(t==0)reinterpret_cast<float*>(out+32896)[kb*8+row]=sf(e);
}
__global__ void reset(Rank a){a.gcount[threadIdx.x]=0;}
double err(const std::vector<uint8_t>&x,const std::vector<uint8_t>&xs,const std::vector<uint8_t>&y,const std::vector<uint8_t>&ys,int rows){
 double worst=0;for(int row=0;row<rows;++row){double dif=0,mag=0;
  for(int i=row*2048;i<(row+1)*2048;++i){__nv_fp8_e4m3 a,b;a.__x=x[i];b.__x=y[i];double v=std::ldexp(float(a),int(xs[i/32])-127),w=std::ldexp(float(b),int(ys[i/32])-127);dif+=(v-w)*(v-w);mag+=v*v;}
  worst=std::max(worst,std::sqrt(dif/std::max(mag,1e-30)));}return worst;
}
template<int Mode> __global__ void selectRows(const float*x,const float*b,int*ids,float*p,int rows,int norm,float scale){
 __shared__ fastllm::cuda::RouterTopKScratch<8> scratch[8];
 const int warp=threadIdx.x/32,lane=threadIdx.x%32,row=blockIdx.x*8+warp;
 if(row>=rows)return;
 auto&s=scratch[warp];
 if constexpr(Mode==0)fastllm::cuda::SelectRouterTopK<256,8>(x+row*256,b?b+row*256:nullptr,ids+row*8,s.probabilities,b!=nullptr,norm,scale,s);
 if constexpr(Mode!=0)fastllm_shared_flex_tp::block128_gate_detail::SelectGateRoutes<Mode==2>(x+row*256,b?b+row*256:nullptr,ids+row*8,s.probabilities,norm,scale,s);

 if(lane<8)p[row*8+lane]=s.probabilities[lane];
}
float frombits(unsigned x){float v;memcpy(&v,&x,4);return v;}

void ValidateRouting(){
 constexpr int N=4096;std::mt19937 rng(18291);std::uniform_real_distribution<float> uni(-1,1);
 std::vector<float>x(N*256),bias(N*256);
 auto*dx=(float*)alloc(x.size()*4),*db=(float*)alloc(bias.size()*4);
 int*ids[3];float*p[3];for(int k=0;k<3;++k){ids[k]=(int*)alloc(N*32);p[k]=(float*)alloc(N*32);}
 size_t cases=0;
 for(int seed=0;seed<3;++seed){
  for(int row=0;row<N;++row)for(int e=0;e<256;++e){
   float v=uni(rng),b=uni(rng)*.05f;
   switch(row%16){
    case 0:v=(v+1)*.5f;break;
    case 1:break;
    case 2:v=std::ldexp(v,int(rng()%241)-120);b=0;break;
    case 3:v=.5f;b=0;break;
    case 4:v=e<2?.9f:float(256-e)/512;b=0;break;
    case 5:v=e<7?1.f-e*.01f:e<10?.5f:.1f;b=0;break;
    case 6:v=frombits(0x3f800000u+(rng()%256));b=0;break;
    case 7:v=frombits(0x3f8000f0u+(rng()%32));b=0;break;
    case 8:v=e%2?-0.f:0.f;b=0;break;
    case 9:v=e%17==0?NAN:e%19==0?INFINITY:e%23==0?-INFINITY:v;break;
    case 10:v=NAN;b=0;break;
    case 11:v=-INFINITY;b=0;break;
    case 12:v=frombits(0xff7fffffu-(rng()%512));b=0;break;
    case 13:v=e<8?1.f-e*.02f:.01f;b=0;break;
    case 14:b=-v+uni(rng)*1.e-7f;break;
    case 15:b=e%7==0?NAN:e%9==0?INFINITY:b;break;
   }
   x[row*256+e]=v;bias[row*256+e]=b;
  }
  check(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));check(cudaMemcpy(db,bias.data(),bias.size()*4,cudaMemcpyHostToDevice));
  for(int hasBias:{0,1})for(int norm:{0,1})for(float scale:{0.f,-.75f,.75f,2.5f}){
   selectRows<0><<<N/8,256,0,cudaStreamPerThread>>>(dx,hasBias?db:nullptr,ids[0],p[0],N,norm,scale);
   selectRows<1><<<N/8,256,0,cudaStreamPerThread>>>(dx,hasBias?db:nullptr,ids[1],p[1],N,norm,scale);
   selectRows<2><<<N/8,256,0,cudaStreamPerThread>>>(dx,hasBias?db:nullptr,ids[2],p[2],N,norm,scale);

   check(cudaStreamSynchronize(cudaStreamPerThread));
   std::vector<int>ref(N*8),got(N*8);std::vector<unsigned>rp(N*8),gp(N*8);
   check(cudaMemcpy(ref.data(),ids[0],N*32,cudaMemcpyDeviceToHost));check(cudaMemcpy(rp.data(),p[0],N*32,cudaMemcpyDeviceToHost));
   for(int k=1;k<3;++k){check(cudaMemcpy(got.data(),ids[k],N*32,cudaMemcpyDeviceToHost));check(cudaMemcpy(gp.data(),p[k],N*32,cudaMemcpyDeviceToHost));
    for(int i=0;i<N*8;++i)if(ref[i]!=got[i]||rp[i]!=gp[i]){fprintf(stderr,"FAIL seed=%d row=%d pattern=%d slot=%d mode=%d bias=%d norm=%d scale=%g ids=%d/%d bits=%x/%x\n",seed,i/8,(i/8)%16,i%8,k,hasBias,norm,scale,ref[i],got[i],rp[i],gp[i]);throw std::runtime_error("router IDs/probabilities differ");}
   }
   cases+=N;
  }
  printf("ROUTE_SEED_PASS seed=%d cases=%zu\n",seed,cases);fflush(stdout);
 }
 printf("ROUTE_VALIDATE_PASS cases=%zu variants=2 IDs_probabilities_bitwise=1\n",cases);check(cudaFree(dx));check(cudaFree(db));for(int k=0;k<3;++k){check(cudaFree(ids[k]));check(cudaFree(p[k]));}
}
int main(){try{
 int count=0;cudaDeviceProp prop{};
 if(cudaGetDeviceCount(&count)!=cudaSuccess||count==0||cudaGetDeviceProperties(&prop,0)!=cudaSuccess||prop.major!=10||prop.minor!=3){puts("SKIP: SM103 gate regression");return 77;}
 check(cudaSetDevice(0));ValidateRouting();
 Rank a{};a.gate=(uint8_t*)alloc(size_t(256)*512*4096);a.gs=(uint8_t*)alloc(size_t(256)*512*128);
 a.xq=(uint8_t*)alloc(32768);a.xs=(uint8_t*)alloc(1024);a.mid=(uint8_t*)alloc(16384);a.ms=(uint8_t*)alloc(512);
 a.gpartial=(float*)alloc(64*8*8*512*4);a.gcount=(int*)alloc(1024);a.epoch=(uint64_t*)alloc(8);
 a.count=(int*)alloc(4);a.nuniq=(int*)alloc(4);a.uniq=(int*)alloc(256);a.members=(int*)alloc(2048);a.rowcounts=(int*)alloc(256);a.jobs=(int*)alloc(256);
 a.gmap=(CUtensorMap*)alloc(128);a.xmap=(CUtensorMap*)alloc(128);
 auto*router=(float*)alloc(8192);auto*bias=(float*)alloc(1024);auto*ids=(int*)alloc(256);auto*pr=(float*)alloc(256);
 auto*packed=(uint8_t*)alloc(size_t(256)*512*4096);auto*prepared=(uint8_t*)alloc(33920);
 auto*coarseScales=(uint8_t*)alloc(size_t(256)*512*128);auto*coarseMap=(CUtensorMap*)alloc(128);
 CUtensorMap cm;cuuint64_t cd[]={128,512,32,256},cs[]={4096,128,512*4096};cuuint32_t cb[]={128,32,8,1},ct[]={1,1,1,1};
 need(cuTensorMapEncodeTiled(&cm,CU_TENSOR_MAP_DATA_TYPE_UINT8,4,packed,cd,cs,cb,ct,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)==CUDA_SUCCESS,"coarse map");check(cudaMemcpy(coarseMap,&cm,128,cudaMemcpyHostToDevice));
 CUtensorMap m;cuuint64_t gd[]={4096,512,256},gs[]={4096,4096*512};cuuint32_t gb[]={128,128,1},go[]={1,1,1};
 need(cuTensorMapEncodeTiled(&m,CU_TENSOR_MAP_DATA_TYPE_UINT8,3,(void*)a.gate,gd,gs,gb,go,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)==CUDA_SUCCESS,"gmap");check(cudaMemcpy(a.gmap,&m,128,cudaMemcpyHostToDevice));
 cuuint64_t xd[]={4096,8},xt[]={4096};cuuint32_t xb[]={128,8},xx[]={1,1};
 need(cuTensorMapEncodeTiled(&m,CU_TENSOR_MAP_DATA_TYPE_UINT8,2,a.xq,xd,xt,xb,xx,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)==CUDA_SUCCESS,"xmap");check(cudaMemcpy(a.xmap,&m,128,cudaMemcpyHostToDevice));
 check(cudaFuncSetAttribute(routed_gate,cudaFuncAttributeMaxDynamicSharedMemorySize,RoutedGateBytes));
 check(cudaFuncSetAttribute(block128_gate,cudaFuncAttributeMaxDynamicSharedMemorySize,Block128GateBytes));
 check(cudaFuncSetAttribute(block128_gate_primary,cudaFuncAttributeMaxDynamicSharedMemorySize,Block128GateBytes));
 Args args{};args.rank=a;args.scores=router;args.bias=bias;args.expert_indices=ids;args.expert_probs=pr;
 args.needNorm=1;args.route_scale=2.5f;args.weightMap=coarseMap;args.weightScales=coarseScales;args.prepared=prepared;
 int gateCases=0;
 for(int layer=0;layer<3;++layer){
  fillWeights<<<2048,256,0,cudaStreamPerThread>>>((uint8_t*)a.gate,(uint8_t*)a.gs,19+layer*71);
  check(cudaMemcpyAsync(packed,a.gate,size_t(256)*512*4096,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
  check(cudaMemcpyAsync(coarseScales,a.gs,size_t(256)*512*128,cudaMemcpyDeviceToDevice,cudaStreamPerThread));
  coarsen_gate_weights<<<256*16*32*2,256,0,cudaStreamPerThread>>>(packed,coarseScales);check(cudaDeviceSynchronize());
  for(int rows:{7,8})for(int pattern=-1;pattern<9;++pattern){a.rows=rows;args.rank.rows=rows;
   fillInput<<<128,256,0,cudaStreamPerThread>>>(a,37+pattern*19+layer*23);
   std::vector<float>scores(2048),correction(256);
   for(int i=0;i<2048;++i)scores[i]=float((i*79+layer*37)%257)/257;
   for(int i=0;i<256;++i)correction[i]=.01f*sinf(float(i));
   if(pattern>=0){std::fill(scores.begin(),scores.end(),.01f);std::fill(correction.begin(),correction.end(),0.f);
    for(int row=0;row<rows;++row)for(int j=0;j<8;++j)scores[row*256+row*pattern+j]=.9f-.01f*j;}
   check(cudaMemcpy(router,scores.data(),rows*1024,cudaMemcpyHostToDevice));
   check(cudaMemcpy(bias,correction.data(),1024,cudaMemcpyHostToDevice));
   args.bias=pattern%3==0?nullptr:bias;args.needNorm=pattern%2!=0;args.route_scale=pattern%3==0?.75f:2.5f;
   prepare128<<<256,128,0,cudaStreamPerThread>>>(a.xq,a.xs,prepared,rows);
   auto reference=[&](){
     reset<<<1,256,0,cudaStreamPerThread>>>(a);
     routed_gate<<<296,256,RoutedGateBytes,cudaStreamPerThread>>>(a,router,args.bias,ids,pr,args.needNorm,args.route_scale);
   };
   auto launch=[&](bool primary){
     reset<<<1,256,0,cudaStreamPerThread>>>(a);
     if(primary)block128_gate_primary<<<148,384,Block128GateBytes,cudaStreamPerThread>>>(args);
     else block128_gate<<<148,384,Block128GateBytes,cudaStreamPerThread>>>(args);
   };
   reference();check(cudaDeviceSynchronize());std::vector<uint8_t>ref(rows*2048),rs(rows*64),got(ref.size()),ss(rs.size());std::vector<int>refids(rows*8),gotids(rows*8);
   check(cudaMemcpy(ref.data(),a.mid,ref.size(),cudaMemcpyDeviceToHost));check(cudaMemcpy(rs.data(),a.ms,rs.size(),cudaMemcpyDeviceToHost));check(cudaMemcpy(refids.data(),ids,rows*32,cudaMemcpyDeviceToHost));
   std::vector<float>refprob(rows*8),gotprob(rows*8);check(cudaMemcpy(refprob.data(),pr,rows*32,cudaMemcpyDeviceToHost));
   for(bool primary:{false,true}){
    if(primary && rows!=8)continue;
    ++gateCases;
    launch(primary);check(cudaDeviceSynchronize());
    check(cudaMemcpy(got.data(),a.mid,got.size(),cudaMemcpyDeviceToHost));check(cudaMemcpy(ss.data(),a.ms,ss.size(),cudaMemcpyDeviceToHost));check(cudaMemcpy(gotids.data(),ids,rows*32,cudaMemcpyDeviceToHost));need(refids==gotids,"route IDs");double l2=err(ref,rs,got,ss,rows);need(std::isfinite(l2),"nonfinite error");
    need(l2<.003,"exact-rebase gate L2 exceeds 0.003");
    check(cudaMemcpy(gotprob.data(),pr,rows*32,cudaMemcpyDeviceToHost));need(refprob==gotprob,"routing probabilities");
    int count=0;check(cudaMemcpy(&count,a.nuniq,4,cudaMemcpyDeviceToHost));need(count>=8&&count<=rows*8,"unique count");
    std::vector<int>experts(count),members(count*8),counts(count);
    check(cudaMemcpy(experts.data(),a.uniq,count*4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(members.data(),a.members,count*32,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(counts.data(),a.rowcounts,count*4,cudaMemcpyDeviceToHost));
    int total=0;std::vector<int>seen(rows*8);
    for(int i=0;i<count;++i){need(counts[i]>0&&counts[i]<=rows,"membership count");
     for(int j=0;j<counts[i];++j){int m=members[i*8+j];need(m>=0&&m<rows*8&&refids[m]==experts[i],"membership value");++seen[m];++total;}}
    need(total==rows*8,"membership total");for(int n:seen)need(n==1,"duplicate/missing member");
    cudaGraph_t graph;cudaGraphExec_t exec;
    check(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));launch(primary);
    check(cudaStreamEndCapture(cudaStreamPerThread,&graph));check(cudaGraphInstantiate(&exec,graph,0));
    for(int it=0;it<3;++it){
     fillInput<<<128,256,0,cudaStreamPerThread>>>(a,113+it*71+layer*13);
     reference();check(cudaStreamSynchronize(cudaStreamPerThread));
     check(cudaMemcpy(ref.data(),a.mid,ref.size(),cudaMemcpyDeviceToHost));
     check(cudaMemcpy(rs.data(),a.ms,rs.size(),cudaMemcpyDeviceToHost));
     prepare128<<<256,128,0,cudaStreamPerThread>>>(a.xq,a.xs,prepared,rows);
     check(cudaMemsetAsync(a.mid,0x55,16384,cudaStreamPerThread));
     check(cudaGraphLaunch(exec,cudaStreamPerThread));check(cudaStreamSynchronize(cudaStreamPerThread));
     check(cudaMemcpy(got.data(),a.mid,got.size(),cudaMemcpyDeviceToHost));
     check(cudaMemcpy(ss.data(),a.ms,ss.size(),cudaMemcpyDeviceToHost));
     need(err(ref,rs,got,ss,rows)<.003,"changing-input gate graph L2 exceeds 0.003");
    }
    check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));
   }
  }
 }
 printf("GATE_VALIDATE_PASS cases=%d\n",gateCases);return 0;
}catch(const std::exception&e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}
