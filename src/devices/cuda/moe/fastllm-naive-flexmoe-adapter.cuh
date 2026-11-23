// Experimental FastLLM adapter, SM103 TP8 EP1, <=8 rows. Built in with the existing CUDA/CUTLASS toolchain.
#include <vector>
#include <cstdio>
#include <stdexcept>
#include "naivert/entry/expert_down_runtime_domain.hpp"

namespace fastflex {
static void check(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static void* alloc(size_t n) { void* p=nullptr; check(cudaMalloc(&p,n)); return p; }
struct Layer { unsigned char *gate=nullptr,*down=nullptr; float* scale=nullptr; };
struct Rank {
  int device; std::vector<Layer> layers; float *up,*probs; int* ids;
  uint64_t* epoch; unsigned char* mailbox; Domain* domains;
};
struct State { Rank ranks[8]; };

// Input experts are FastLLM's original row-major FP8 bytes and FP32 scales.
// Pack only once; activation quantization remains inside the arithmetic bodies.
__global__ void pack_gate(const uint64_t* weights, const float* scales, unsigned char* packed) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  constexpr size_t count=size_t(256)*16*4*32896;
  if(i>=count)return;
  int v=i%32896, page=i/32896%4, group=i/(32896*4)%16, e=i/(32896*4*16);
  if(v<32768){
    int part=v/16384, z=v%16384, kc=z/512, s=z%512;
    int b=s/64, d=s/16%4, c=s/8%2, a=s/4%2, tail=s%4;
    int row=group*16+a*8+b+part*256, col=page*1024+kc*32+c*16+d*4+tail;
    packed[i]=reinterpret_cast<const unsigned char*>(weights[e*2])[row*4096+col];
  } else if(v<32832) {
    int s=(v-32768)/4, part=s/8, k=s%8;
    float val=scales[e*128+(part*2+group/8)*32+page*8+k];
    packed[i]=reinterpret_cast<unsigned char*>(&val)[(v-32768)%4];
  } else packed[i]=0;
}
__global__ void pack_down(const uint64_t* weights, const float* scales,
                           unsigned char* packed,float* out_scale) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  constexpr size_t count=size_t(256)*4096*256;
  if(i<count){
    int e=i/(4096*256), z=i%(4096*256), tile=z/16384, p=z%16384;
    int kg=p/8192, row=p/128%64, chunk=p/16%8, tail=p%16;
    packed[i]=reinterpret_cast<const unsigned char*>(weights[e*2+1])[(tile*64+row)*256+kg*128+(chunk^(row&7))*16+tail];
  }
  if(i<size_t(256)*64*16){
    int e=i/(64*16), tile=i/16%64, s=i%16;
    out_scale[i]=scales[e*64+(tile/2)*2+s%2];
  }
}
__global__ void advance(uint64_t* epoch) { ++*epoch; }
}

extern "C" int fastflex_supported(const int*,int);
extern "C" void fastflex_destroy(void*);
extern "C" int fastflex_create(const int* devices,int layers,void** result) {
  using namespace fastflex;
  if(!devices||!result||layers<=0)return 1;
  *result=nullptr;
  if(!fastflex_supported(devices,8))return 1;
  State* st=nullptr;
  try {
    st=new State{};
    for(int r=0;r<8;++r){
      check(cudaSetDevice(devices[r])); cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop,devices[r]));
      if(prop.major!=10 || prop.minor!=3 || prop.multiProcessorCount!=148) throw std::runtime_error("requires SM103 with 148 SMs");
      for(int peer=0;peer<8;++peer) if(peer!=r){
        auto e=cudaDeviceEnablePeerAccess(devices[peer],0);
        if(e==cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError(); else check(e);
      }
      auto& a=st->ranks[r]; a.device=devices[r]; a.layers.resize(layers);
      a.up=(float*)alloc(8*8*256*4); a.probs=(float*)alloc(8*8*4);a.ids=(int*)alloc(8*8*4);
      a.epoch=(uint64_t*)alloc(8);check(cudaMemset(a.epoch,0,8));
      a.mailbox=(unsigned char*)alloc(8*down::kExpertDownMailboxRowBytes);
      check(cudaMemset(a.mailbox,0,8*down::kExpertDownMailboxRowBytes));
      a.domains=(Domain*)alloc(8*sizeof(Domain));
    }
    for(int r=0;r<8;++r){
      auto&a=st->ranks[r];check(cudaSetDevice(a.device));Domain domains[8]{};
      for(int row=0;row<8;++row){
        auto& d=domains[row];d.magic=down::kExpertDownDomainMagic;d.version=down::kExpertDownDomainVersion;
        d.rank=r;d.nranks=8;d.native_mailbox_tag=5;d.step_block_ptr=(uint64_t)a.epoch;
        for(int peer=0;peer<8;++peer)d.peer_mailbox_base_ptrs[peer]=(uint64_t)(st->ranks[peer].mailbox+row*down::kExpertDownMailboxRowBytes);
      }
      check(cudaMemcpy(a.domains,domains,sizeof(domains),cudaMemcpyHostToDevice));
    }
    *result=st;return 0;
  }catch(const std::exception&e){fastflex_destroy(st);fprintf(stderr,"[fastflex] create: %s\n",e.what());return 1;}
}
extern "C" int fastflex_pack(void* state,int rank,int layer,const uint64_t* weights,
                              const float* gate_scales,const float* down_scales){
  using namespace fastflex;
  if(!state||rank<0||rank>=8||!weights||!gate_scales||!down_scales)return 1;
  auto&a=((State*)state)->ranks[rank];
  if(layer<0||layer>=(int)a.layers.size())return 1;
  auto&w=a.layers[layer];if(w.gate||w.down||w.scale)return 1;
  uint64_t*p=nullptr;float*gs=nullptr;float*ds=nullptr;
  try {
    check(cudaSetDevice(a.device));
    w.gate=(unsigned char*)alloc(size_t(256)*16*4*32896);
    w.down=(unsigned char*)alloc(size_t(256)*4096*256);w.scale=(float*)alloc(256*64*16*4);
    p=(uint64_t*)alloc(512*8);gs=(float*)alloc(256*128*4);ds=(float*)alloc(256*64*4);
    check(cudaMemcpy(p,weights,512*8,cudaMemcpyHostToDevice));
    check(cudaMemcpy(gs,gate_scales,256*128*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(ds,down_scales,256*64*4,cudaMemcpyHostToDevice));
    pack_gate<<<(size_t(256)*16*4*32896+255)/256,256>>>(p,gs,w.gate);
    pack_down<<<(size_t(256)*4096*256+255)/256,256>>>(p,ds,w.down,w.scale);
    check(cudaGetLastError());check(cudaDeviceSynchronize());
    check(cudaFree(p));p=nullptr;check(cudaFree(gs));gs=nullptr;check(cudaFree(ds));ds=nullptr;return 0;
  }catch(const std::exception&e){cudaFree(p);cudaFree(gs);cudaFree(ds);fprintf(stderr,"[fastflex] pack: %s\n",e.what());return 1;}
}
extern "C" int fastflex_run(void* state,int rank,int layer,int rows,
                             const void* x,const void* probabilities,const void* bias,void* hidden){
  if(!state||rank<0||rank>=8||rows<1||rows>8||!x||!probabilities||!bias||!hidden)return 1;
  auto&a=((fastflex::State*)state)->ranks[rank];
  if(layer<0||layer>=(int)a.layers.size())return 1;
  auto&w=a.layers[layer];if(!w.gate||!w.down||!w.scale)return 1;
  auto stream=cudaStreamPerThread;
  fastflex::advance<<<1,1,0,stream>>>(a.epoch);
  uint64_t gp[]={(uint64_t)x,(uint64_t)probabilities,(uint64_t)bias,(uint64_t)w.gate,
                (uint64_t)a.up,(uint64_t)a.probs,(uint64_t)a.ids};
  int e=LaunchGate(gp,rows,stream);if(e)return e;
  // <=8 uses the fused body: each element reads its residual before storing.
  uint64_t dp[]={(uint64_t)a.up,(uint64_t)w.down,(uint64_t)w.scale,(uint64_t)a.ids,
                (uint64_t)a.probs,(uint64_t)hidden,(uint64_t)hidden,(uint64_t)a.domains};
  return LaunchDown(dp,rows,rank,stream);
}
extern "C" void fastflex_destroy(void* state){
  using namespace fastflex;
  auto*st=(State*)state;if(!st)return;
  for(auto&a:st->ranks){cudaSetDevice(a.device);cudaDeviceSynchronize();
    for(auto&w:a.layers){cudaFree(w.gate);cudaFree(w.down);cudaFree(w.scale);}
    cudaFree(a.up);cudaFree(a.probs);cudaFree(a.ids);cudaFree(a.epoch);cudaFree(a.mailbox);cudaFree(a.domains);
  }
  delete st;
}
