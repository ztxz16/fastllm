// Blackwell MiMo EP4/EP8: pinned upstream FP8/MXFP4 MegaMoE, no Python/JIT dependency.
// FP8/FP32 block128 source weights are explicitly requantized on load to the
// FP8/UE8M0-per32 format used by the historical NaiveSglang SM103 adapter.
#ifndef FASTLLM_MIMO_MEGA_SM100
#define FASTLLM_MIMO_MEGA_SM100
#endif
#include "devices/cuda/fastllm-mimo-mega-backends.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <array>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>
// NVCC 13 can emit an unqualified inf literal in its host stub.
static constexpr float inf = __builtin_inff();
namespace {
namespace dg = fastllm_blackwell_mega;
constexpr int H=4096, I=2048, E=256, K=8, CAP=128;
constexpr int BN=128, BK=128, SMS=148, SMALL_CAP=32;
// Keep every buffer address identical when switching tiles, including the
// scale-ring stride. Both variants reuse the existing arena and packed weights.
constexpr int RING=1920, SFRING=RING/16*128;
template<int BM> struct Tile {
    static_assert(BM==16 || BM==32);
    static constexpr int STORE=BM/2, STAGES=BM==16 ? 11 : 10;
    static constexpr int SMEM = 1024 + 4*H + STORE*128*4 + 8*STORE*4 +
        2*sizeof(dg::sched::TaskInfo<true>) + (4+4+16+4)*8 + 4 +
        STAGES*(BM/2*128 + 128*128 + 128*4*2 + 16);
    static_assert(SMEM <= 232448);
};
static_assert(Tile<32>::SMEM <= Tile<16>::SMEM);
thread_local std::string error;
void check(cudaError_t s) { if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
void *allocate(size_t bytes) { void *p=nullptr; check(cudaMalloc(&p,bytes)); return p; }
using Desc=cute::TmaDescriptor;
void descriptor(Desc &out, CUtensorMapDataType type, void *p, int rank,
    const cuuint64_t *dims, const cuuint64_t *strides, const cuuint32_t *box, int swizzle) {
    cuuint32_t steps[]={1,1,1,1};
    auto mode=swizzle==128 ? CU_TENSOR_MAP_SWIZZLE_128B :
        swizzle==64 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_NONE;
    auto s=cuTensorMapEncodeTiled(reinterpret_cast<CUtensorMap *>(&out),type,rank,p,
        dims,strides,box,steps,CU_TENSOR_MAP_INTERLEAVE_NONE,mode,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (s!=CUDA_SUCCESS) throw std::runtime_error("Blackwell MegaMoE TMA error "+std::to_string(s));
}
void desc2(Desc &out,void *p,int inner,int outer,int bi,int bo,int bytes,int swizzle) {
    cuuint64_t dims[]={(uint64_t)inner,(uint64_t)outer}, strides[]={(uint64_t)inner*bytes};
    cuuint32_t box[]={(uint32_t)bi,(uint32_t)bo};
    descriptor(out,bytes==4 ? CU_TENSOR_MAP_DATA_TYPE_INT32 : CU_TENSOR_MAP_DATA_TYPE_UINT8,
        p,2,dims,strides,box,swizzle);
}
void descWeight(Desc &out,void *p,int n,int k,int localExperts,bool mxfp4) {
    cuuint64_t dims[]={(uint64_t)k,(uint64_t)n/2,2,(uint64_t)localExperts};
    const cuuint64_t rowBytes=mxfp4 ? k/2 : k;
    cuuint64_t strides[]={rowBytes,n*rowBytes/2,n*rowBytes};
    cuuint32_t box[]={BK,BN,1,1};
    descriptor(out,mxfp4 ? CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B : CU_TENSOR_MAP_DATA_TYPE_UINT8,p,4,dims,strides,box,128);
}
__device__ int sourceRow(int row,int n,bool gate) {
    return gate ? row/16*8+row%8+(row%16>=8 ? n/2 : 0) : row;
}
__device__ size_t scaleOffset(int row,int col,int n) {
    int transposed=(row&~127)+(row%32)*4+(row%128)/32;
    return ((size_t)(col/4)*n+transposed)*4+col%4;
}
// Native E2M1 nibbles and K32 UE8M0 exponents arrive in one allocation.
// Only permute rows and scales; TMA unpacks the weights into shared memory.
__global__ void packMxfp4(const uint8_t *src,uint8_t *dst,uint8_t *sf,int n,int k,bool gate) {
    int group=blockIdx.x*blockDim.x+threadIdx.x;
    if (group>=n*(k/32)) return;
    int row=group/(k/32), col=group%(k/32);
    int source=sourceRow(row,n,gate)*(k/32)+col;
    reinterpret_cast<uint4 *>(dst)[group]=reinterpret_cast<const uint4 *>(src)[source];
    sf[scaleOffset(row,col,n)]=src[(size_t)n*k/2+source];
}
// One warp per 32 source weights. Combine requantization, gate/up gran8
// interleave and UTCCP scale transpose in one load-time pass.
__global__ void pack(const uint8_t *src,const float *scales,uint8_t *dst,uint8_t *sf,
    int n,int k,bool gate) {
    int group=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if (group>=n*(k/32)) return;
    int lane=threadIdx.x%32, row=group/(k/32), col=group%(k/32);
    int source=sourceRow(row,n,gate);
    __nv_fp8_e4m3 q; q.__x=src[(size_t)source*k+col*32+lane];
    float value=float(q)*scales[(source/128)*(k/128)+col/4];
    float amax=fabsf(value);
    for (int mask=16;mask;mask>>=1) amax=fmaxf(amax,__shfl_xor_sync(0xffffffffu,amax,mask));
    uint32_t exponent=dg::math::get_ue8m0_sf_exp(amax);
    float inverse=dg::math::get_ue8m0_sf_inv<float>(exponent);
    dst[(size_t)row*k+col*32+lane]=__nv_fp8_e4m3(value*inverse).__x;
    if (lane==0) sf[scaleOffset(row,col,n)]=exponent;
}
__global__ void prepare(const __nv_bfloat16 *x,const int32_t *idx,const float *score,
    uint8_t *qx,uint8_t *sf,int64_t *qi,float *qs,int rows,const int *live,
    uint64_t *combineReady,int ranks) {
    // CUDA Graph replay reuses %gridid. The upstream combine protocol tags
    // readiness with gridid+1, so clear our receive flags on every invocation.
    // This kernel precedes MegaMoE on the same stream; its initial cross-rank
    // barrier guarantees every clear finishes before any new ready signal.
    // Zero-token ranks must execute this too. No extra launch is required.
    if (blockIdx.x==0 && threadIdx.x<ranks) combineReady[threadIdx.x]=0;
    int group=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if (group>=rows*(H/32)) return;
    int lane=threadIdx.x%32,row=group/(H/32),col=group%(H/32);
    bool valid=!live || live[row]>0;
    float value=valid ? __bfloat162float(x[row*H+col*32+lane]) : 0.f;
    float amax=fabsf(value);
    for (int mask=16;mask;mask>>=1) amax=fmaxf(amax,__shfl_xor_sync(0xffffffffu,amax,mask));
    uint32_t exponent=dg::math::get_ue8m0_sf_exp(amax);
    qx[row*H+col*32+lane]=__nv_fp8_e4m3(value*dg::math::get_ue8m0_sf_inv<float>(exponent)).__x;
    if (lane==0) sf[row*(H/32)+col]=exponent;
    if (col==0 && lane<K) {
        qi[row*K+lane]=valid ? idx[row*K+lane] : -1;
        qs[row*K+lane]=valid ? score[row*K+lane] : 0.f;
    }
}
struct LayerBase { int ranks; };
template<int R> struct Mega {
    static_assert(R==4 || R==8);
    static constexpr int LE=E/R;
    static dg::layout::MegaMoEBuffer buffer(void *p) { return {p,H,I,R,E,CAP,K,RING,SFRING,true,0}; }
    static size_t bytes() { return buffer(nullptr).get_num_bytes(); }
    template<int BM=16,typename Weight=cutlass::float_e4m3_t> static auto kernel() {
        return &dg::sm100_fp8_fp4_mega_moe_impl<CAP,H,I,E,0,K,BM,BN,BK,
            Tile<BM>::STORE,Tile<BM>::STORE,128,128,RING,SFRING,Tile<BM>::STAGES,H,128,128,256,SMS,R,
            cute::numeric_limits<float>::infinity(),true,Weight>;
    }
    template<int BM> static auto selectKernel(bool mxfp4) {
        return mxfp4 ? kernel<BM,cutlass::detail::float_e2m1_unpacksmem_t>() : kernel<BM>();
    }
    struct Layer:LayerBase {
        Layer():LayerBase{R} {}
        int device=-1,rank=-1;
        bool mxfp4=false;
        void *workspace=nullptr,*gate=nullptr,*down=nullptr,*gateSf=nullptr,*downSf=nullptr;
        uint8_t *domains=nullptr;
        std::array<int64_t,R> peers{};
        Desc a1,s1,w1,ws1,d1,a2,s2,w2,ws2;
        Desc batchA1,batchD1,batchA2;
        ~Layer() {
            int previous=0; cudaGetDevice(&previous);
            if (device>=0) cudaSetDevice(device);
            for (void *p:{gate,down,gateSf,downSf,(void *)domains}) if (p) cudaFree(p);
            cudaSetDevice(previous);
        }
    };
    static int init(void *w,size_t size,const int *devices,int rank) {
        try {
            int device=-1; check(cudaGetDevice(&device));
            if (!w || size<bytes() || !devices || rank<0 || rank>=R || devices[rank]!=device ||
                !fastllm_sm100_mega_moe_supported()) throw std::runtime_error("Invalid Blackwell MegaMoE workspace/device");
            for (int r=0;r<R;++r) if (r!=rank) {
                int accessible=0; check(cudaDeviceCanAccessPeer(&accessible,device,devices[r]));
                if (!accessible) throw std::runtime_error("MegaMoE requires all-to-all CUDA peer access");
                auto s=cudaDeviceEnablePeerAccess(devices[r],0);
                if (s==cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError(); else check(s);
            }
            check(cudaMemset(w,0,size)); check(cudaDeviceSynchronize()); return 1;
        } catch (const std::exception &e) { error=e.what(); return 0; }
    }
    static void *create(const void *const *g,const void *const *d,const void *const *gs,const void *const *ds,
        void *w,size_t size,const void *const *peers,int rank,int slot,bool mxfp4=false) {
        try {
            if (!fastllm_sm100_mega_moe_supported() || !w || size<bytes() ||
                !g || !d || (!mxfp4 && (!gs || !ds)) || !peers || rank<0 || rank>=R ||
                peers[rank]!=w || slot<0 || slot>=48 || (uintptr_t)w%256)
                throw std::runtime_error("Invalid Blackwell MegaMoE layer configuration");
            auto l=std::make_unique<Layer>(); check(cudaGetDevice(&l->device));
            l->rank=rank; l->workspace=w; l->mxfp4=mxfp4;
            for (int r=0;r<R;++r) {
                if (!peers[r]) throw std::runtime_error("Missing MegaMoE peer workspace");
                l->peers[r]=reinterpret_cast<int64_t>(peers[r]);
            }
            const size_t downBytes=(size_t)H*I/(mxfp4 ? 2 : 1);
            l->gate=allocate(LE*2*downBytes); l->down=allocate(LE*downBytes);
            l->gateSf=allocate((size_t)LE*2*I*H/32); l->downSf=allocate((size_t)LE*H*I/32);
            l->domains=static_cast<uint8_t *>(allocate(2048));
            std::array<uint8_t,2048> domains{};
            for (size_t j=0;j<domains.size();++j) domains[j]=(j/2)%2;
            check(cudaMemcpy(l->domains,domains.data(),domains.size(),cudaMemcpyHostToDevice));
            for (int e=0;e<LE;++e) {
                if (!g[e] || !d[e] || (!mxfp4 && (!gs[e] || !ds[e])))
                    throw std::runtime_error("Missing MegaMoE expert weights or scales");
                auto packExpert=[&](const void *src,const void *scale,void *dst,void *sf,int n,int k,bool gate) {
                    auto *values=(uint8_t *)dst+(size_t)e*n*k/(mxfp4 ? 2 : 1);
                    auto *exponents=(uint8_t *)sf+(size_t)e*n*k/32;
                    if (mxfp4)
                        packMxfp4<<<(n*k/32+127)/128,128,0,cudaStreamPerThread>>>(
                            (const uint8_t *)src,values,exponents,n,k,gate);
                    else
                        pack<<<(n*k+127)/128,128,0,cudaStreamPerThread>>>(
                            (const uint8_t *)src,(const float *)scale,values,exponents,n,k,gate);
                };
                packExpert(g[e],mxfp4 ? nullptr : gs[e],l->gate,l->gateSf,2*I,H,true);
                packExpert(d[e],mxfp4 ? nullptr : ds[e],l->down,l->downSf,H,I,false);
            }
            check(cudaGetLastError()); check(cudaDeviceSynchronize());
            auto b=buffer(w);
            desc2(l->a1,b.l1_token_buffer.base,H,RING,BK,8,1,128);
            desc2(l->s1,b.l1_sf_buffer.base,SFRING,H/128,128,1,4,0);
            descWeight(l->w1,l->gate,2*I,H,LE,mxfp4);
            desc2(l->ws1,l->gateSf,2*I,H/128*LE,128,1,4,0);
            desc2(l->d1,b.l2_token_buffer.base,I,RING,64,8,1,64);
            desc2(l->a2,b.l2_token_buffer.base,I,RING,BK,8,1,128);
            desc2(l->s2,b.l2_sf_buffer.base,SFRING,I/128,128,1,4,0);
            descWeight(l->w2,l->down,H,I,LE,mxfp4);
            desc2(l->ws2,l->downSf,H,I/128*LE,128,1,4,0);
            desc2(l->batchA1,b.l1_token_buffer.base,H,RING,BK,16,1,128);
            desc2(l->batchD1,b.l2_token_buffer.base,I,RING,64,16,1,64);
            desc2(l->batchA2,b.l2_token_buffer.base,I,RING,BK,16,1,128);
            check(cudaFuncSetAttribute(selectKernel<16>(mxfp4),cudaFuncAttributeMaxDynamicSharedMemorySize,Tile<16>::SMEM));
            check(cudaFuncSetAttribute(selectKernel<32>(mxfp4),cudaFuncAttributeMaxDynamicSharedMemorySize,Tile<32>::SMEM));
            return l.release();
        } catch (const std::exception &e) { error=e.what(); return nullptr; }
    }
    template<int BM=16> static int run(void *p,const void *x,void *y,const int32_t *idx,const float *scores,int rows,const int *live) {
        try {
            if (!p || !x || !y || !idx || !scores || rows<0 || rows>CAP)
                throw std::runtime_error("Invalid Blackwell MegaMoE pointers/row count");
            auto &l=*static_cast<Layer *>(p); int device=-1; check(cudaGetDevice(&device));
            if (device!=l.device) throw std::runtime_error("MegaMoE device mismatch");
            auto b=buffer(l.workspace); auto stream=cudaStreamPerThread;
            prepare<<<rows ? (rows*H+127)/128 : 1,128,0,stream>>>((const __nv_bfloat16 *)x,idx,scores,
                b.input_token_buffer.template get_base_ptr<uint8_t>(),b.input_sf_buffer.template get_base_ptr<uint8_t>(),
                b.input_topk_idx_buffer.template get_base_ptr<int64_t>(),b.input_topk_weights_buffer.template get_base_ptr<float>(),rows,live,
                b.workspace.signals->combine_ready_grid_idx,R);
            cudaLaunchAttribute attr{}; attr.id=cudaLaunchAttributeClusterDimension; attr.val.clusterDim={2,1,1};
            cudaLaunchConfig_t config{}; config.gridDim=dim3(SMS); config.blockDim=dim3(512);
            config.dynamicSmemBytes=Tile<BM>::SMEM; config.stream=stream; config.attrs=&attr; config.numAttrs=1;
            const auto &a1=BM==32 ? l.batchA1 : l.a1;
            const auto &d1=BM==32 ? l.batchD1 : l.d1;
            const auto &a2=BM==32 ? l.batchA2 : l.a2;
            check(cudaLaunchKernelEx(&config,selectKernel<BM>(l.mxfp4),y,(int *)nullptr,(uint32_t)rows,
                dg::layout::SymBuffer<R>(l.peers,l.rank),
                a1,l.s1,l.w1,l.ws1,d1,a2,l.s2,l.w2,l.ws2,
                a1,l.s1,l.w1,l.ws1,d1,a2,l.s2,l.w2,l.ws2,(const uint8_t *)l.domains));
            check(cudaGetLastError()); return 1;
        } catch (const std::exception &e) { error=e.what(); return 0; }
    }
};
}
extern "C" int fastllm_sm100_mega_moe_supported() {
    int device=-1; if (cudaGetDevice(&device)!=cudaSuccess) return 0;
    static thread_local std::map<int,bool> cache;
    auto it=cache.find(device); if (it!=cache.end()) return it->second;
    cudaDeviceProp p{}; bool ok=cudaGetDeviceProperties(&p,device)==cudaSuccess &&
        p.major==10 && (p.minor==0 || p.minor==3) && p.multiProcessorCount==SMS && p.sharedMemPerBlockOptin>=Tile<16>::SMEM;
    // SM100a and SM103a cubins are architecture-specific. A multi-architecture
    // build may contain only one; reject missing device code before allocation.
    if (ok) {
        cudaFuncAttributes attributes{};
        ok=cudaFuncGetAttributes(&attributes,Mega<8>::kernel())==cudaSuccess;
        if (!ok) cudaGetLastError();
    }
    cache.emplace(device,ok); return ok;
}
extern "C" const char *fastllm_sm100_mega_moe_error() { return error.c_str(); }
extern "C" int fastllm_sm100_mega_moe_max_rows() { return CAP; }
extern "C" int fastllm_sm100_mega_moe_decode_capacity(int inputRows,int ranks) {
    if (ranks!=4 && ranks!=8) return 0;
    // The common batch/Graph shape selects both the bound and the tile. Never
    // select from local rows: idle and uneven ranks must launch the same tile.
    return inputRows<=ranks*SMALL_CAP ? SMALL_CAP : CAP;
}
extern "C" size_t fastllm_sm100_mega_moe_workspace_bytes(int ranks) {
    return ranks==4 ? Mega<4>::bytes() : ranks==8 ? Mega<8>::bytes() : 0;
}
extern "C" int fastllm_sm100_mega_moe_init_workspace(void *w,size_t b,const int *d,int r,int n) {
    if (n==4) return Mega<4>::init(w,b,d,r); if (n==8) return Mega<8>::init(w,b,d,r);
    error="MegaMoE requires 4 or 8 ranks"; return 0;
}
extern "C" void *fastllm_sm100_mega_moe_create(const void *const *g,const void *const *d,
    const void *const *gs,const void *const *ds,void *w,size_t b,const void *const *p,int r,int s,int n) {
    if (n==4) return Mega<4>::create(g,d,gs,ds,w,b,p,r,s);
    if (n==8) return Mega<8>::create(g,d,gs,ds,w,b,p,r,s);
    error="MegaMoE requires 4 or 8 ranks"; return nullptr;
}
extern "C" void *fastllm_sm100_mega_moe_create_mxfp4(const void *const *g,const void *const *d,
    void *w,size_t b,const void *const *p,int r,int s,int n) {
    if (n==4) return Mega<4>::create(g,d,nullptr,nullptr,w,b,p,r,s,true);
    if (n==8) return Mega<8>::create(g,d,nullptr,nullptr,w,b,p,r,s,true);
    error="MegaMoE requires 4 or 8 ranks"; return nullptr;
}
extern "C" int fastllm_sm100_mega_moe_run_masked(void *p,const void *x,void *y,const int32_t *i,const float *s,int rows,const int *live) {
    if (!p) { error="Missing MegaMoE handle"; return 0; }
    return static_cast<LayerBase *>(p)->ranks==4 ? Mega<4>::run(p,x,y,i,s,rows,live) : Mega<8>::run(p,x,y,i,s,rows,live);
}
extern "C" int fastllm_sm100_mega_moe_run(void *p,const void *x,void *y,const int32_t *i,const float *s,int rows) {
    return fastllm_sm100_mega_moe_run_masked(p,x,y,i,s,rows,nullptr);
}
extern "C" int fastllm_sm100_mega_moe_run_batched(void *p,const void *x,void *y,const int32_t *i,const float *s,int rows,int capacity,const int *live) {
    if ((capacity!=SMALL_CAP && capacity!=CAP) || rows<0 || rows>capacity || !p) {
        error="Invalid Blackwell MegaMoE capacity/row count/handle"; return 0;
    }
    if (capacity==SMALL_CAP)
        return fastllm_sm100_mega_moe_run_masked(p,x,y,i,s,rows,live);
    return static_cast<LayerBase *>(p)->ranks==4 ? Mega<4>::run<32>(p,x,y,i,s,rows,live)
                                               : Mega<8>::run<32>(p,x,y,i,s,rows,live);
}
// Reuse the same bounded persistent ring for arbitrary prefill sizes. The
// caller already synchronizes equal-count chunks across every EP rank.
extern "C" size_t fastllm_sm100_mega_moe_prefill_workspace_bytes(int,int) { return 0; }
extern "C" int fastllm_sm100_mega_moe_prepare_prefill(void *,void *,size_t,int) { return 0; }
extern "C" int fastllm_sm100_mega_moe_run_prefill(void *,const void *const *,const void *,void *,const int32_t *,const float *,int) {
    error="Blackwell MegaMoE uses the persistent bounded arena"; return 0;
}
extern "C" void fastllm_sm100_mega_moe_destroy(void *p) {
    if (!p) return;
    if (static_cast<LayerBase *>(p)->ranks==4) delete static_cast<Mega<4>::Layer *>(p);
    else delete static_cast<Mega<8>::Layer *>(p);
}

extern "C" int fastllm_sm100_mega_moe_weights(void*p,const void**out){
 if(!p||!out||static_cast<LayerBase*>(p)->ranks!=8)return 0;
 auto*l=static_cast<Mega<8>::Layer*>(p);
 if(l->mxfp4)return 0; // Flex consumes FP8 weight views.
 out[0]=l->gate;out[1]=l->down;out[2]=l->gateSf;out[3]=l->downSf;return 1;
}
