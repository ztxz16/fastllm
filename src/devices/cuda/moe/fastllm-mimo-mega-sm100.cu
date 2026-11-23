#include "fastllm-mxfp4-weight-pack.cuh"
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
#include <algorithm>
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
constexpr int H=4096, I=2048, E=256, K=8, CAP=256;
constexpr int BN=128, BK=128, SMS=148, SMALL_CAP=32, MID_CAP=128;
// Keep every buffer address identical when switching tiles, including the
// scale-ring stride. All decode tiles share one arena and the same packed weights.
constexpr int RING=1920, SFRING=RING/16*128;
template<int BM> struct Tile {
    static constexpr int STORE = BM <= 16 ? 8 : BM <= 64 ? 16 : BM <= 192 ? 32 : 24;
    static constexpr int STORE_L2 = BM == 240 ? 8 : STORE;
    static constexpr int SF_M = (BM + 127) / 128 * 128;
    static constexpr int CD = std::max(2 * STORE * 64 * 2, 2 * STORE_L2 * 128 * 2);
    static constexpr int FIXED = 1024 + 4*H + ((CD+1023)/1024)*1024 + 8*STORE*4 +
        2*sizeof(dg::sched::TaskInfo<true>) + (4+4+16+4)*8 + 4;
    static constexpr int PER_STAGE = BM/2*128 + 128*128 + (SF_M+128)*4 + 16;
    static constexpr int STAGES = (232448-FIXED)/PER_STAGE;
    static constexpr int SMEM = FIXED + STAGES*PER_STAGE;
    static_assert(STAGES >= 2);
    static_assert(SMEM <= 232448);
};
static_assert(Tile<16>::STAGES == 11 && Tile<32>::STAGES == 10);
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
// One warp per 32 source weights. Combine requantization, gate/up gran8
// interleave and UTCCP scale transpose in one load-time pass.
__global__ void pack(const uint8_t *src,const float *scales,uint8_t *dst,uint8_t *sf,
    int n,int k,bool gate) {
    int group=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if (group>=n*(k/32)) return;
    int lane=threadIdx.x%32, row=group/(k/32), col=group%(k/32);
    int source=fastllm_mxfp4::sourceRow(row,n,gate);
    __nv_fp8_e4m3 q; q.__x=src[(size_t)source*k+col*32+lane];
    float value=float(q)*scales[(source/128)*(k/128)+col/4];
    float amax=fabsf(value);
    for (int mask=16;mask;mask>>=1) amax=fmaxf(amax,__shfl_xor_sync(0xffffffffu,amax,mask));
    uint32_t exponent=dg::math::get_ue8m0_sf_exp(amax);
    float inverse=dg::math::get_ue8m0_sf_inv<float>(exponent);
    dst[(size_t)row*k+col*32+lane]=__nv_fp8_e4m3(value*inverse).__x;
    if (lane==0) sf[fastllm_mxfp4::scaleOffset(row,col,n)]=exponent;
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
// Each large prefill tile borrows the dead FP32 router tensor. Its capacity,
// token-ring geometry and TMA descriptors must agree on all participants.
template<int R, int Capacity, int BM> struct Geometry {
    static constexpr int maxBlocks = (Capacity*R*K+BM-1)/BM + E/R;
    static constexpr int ring = Capacity == CAP ? RING :
        BM * dg::sched::get_num_max_live_pool_blocks(maxBlocks, SMS, H, I);
    static constexpr int sfRing = Capacity == CAP ? SFRING :
        dg::layout::get_num_sf_ring_tokens(ring,BM);
    static_assert(ring % BM == 0 && ring / BM >= dg::sched::get_num_max_live_pool_blocks(maxBlocks, SMS, H, I));
    static_assert(sfRing >= dg::layout::get_num_sf_ring_tokens(ring, BM));
    static_assert(ring <= dg::layout::MegaMoESignals::kNumMaxRingBlocks * dg::layout::kMinCandidateBlockM);
    static dg::layout::MegaMoEBuffer buffer(void *p) { return {p,H,I,R,E,Capacity,K,ring,sfRing,true,0}; }
};
struct LayerBase { int ranks; };
template<int R> struct Mega {
    static_assert(R==4 || R==8);
    static constexpr int LE=E/R;
    static dg::layout::MegaMoEBuffer buffer(void *p) { return {p,H,I,R,E,CAP,K,RING,SFRING,true,0}; }
    static size_t bytes() { return buffer(nullptr).get_num_bytes(); }
    template<int BM=16,typename Weight=cutlass::float_e4m3_t,int Capacity=CAP> static auto kernel() {
        using G = Geometry<R,Capacity,BM>;
        return &dg::sm100_fp8_fp4_mega_moe_impl<Capacity,H,I,E,0,K,BM,BN,BK,
            Tile<BM>::STORE,Tile<BM>::STORE_L2,Tile<BM>::SF_M,128,G::ring,G::sfRing,Tile<BM>::STAGES,H,128,128,256,SMS,R,
            cute::numeric_limits<float>::infinity(),true,Weight>;
    }
    template<int BM,int Capacity=CAP> static auto selectKernel(bool mxfp4) {
        return mxfp4 ? kernel<BM,cutlass::detail::float_e2m1_unpacksmem_t,Capacity>() : kernel<BM,cutlass::float_e4m3_t,Capacity>();
    }
    struct ActivationMaps {
        Desc a1, d1, a2;
        template<int BM>
        void bind(const dg::layout::MegaMoEBuffer &buffer, int ring) {
            desc2(a1,buffer.l1_token_buffer.base,H,ring,BK,BM/2,1,128);
            desc2(d1,buffer.l2_token_buffer.base,I,ring,64,Tile<BM>::STORE,1,64);
            desc2(a2,buffer.l2_token_buffer.base,I,ring,BK,BM/2,1,128);
        }
    };
    struct PrefillPlan;
    struct Layer:LayerBase {
        Layer():LayerBase{R} {}
        int device=-1,rank=-1;
        bool mxfp4=false;
        void *workspace=nullptr,*gate=nullptr,*down=nullptr,*gateSf=nullptr,*downSf=nullptr;
        uint8_t *domains=nullptr;
        std::array<int64_t,R> peers{};
        ActivationMaps smallMaps, batchMaps, wideMaps, prefillMaps;
        Desc s1,w1,ws1,s2,w2,ws2,preS1,preS2;
        void *prefillBase=nullptr;
        const PrefillPlan *preparedPrefill=nullptr;
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
                        fastllm_mxfp4::packMxfp4<<<(n*k/32+127)/128,128,0,cudaStreamPerThread>>>(
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
            l->smallMaps.template bind<16>(b,RING);
            l->batchMaps.template bind<32>(b,RING);
            l->wideMaps.template bind<128>(b,RING);
            desc2(l->s1,b.l1_sf_buffer.base,SFRING,H/128,128,1,4,0);
            descWeight(l->w1,l->gate,2*I,H,LE,mxfp4);
            desc2(l->ws1,l->gateSf,2*I,H/128*LE,128,1,4,0);
            desc2(l->s2,b.l2_sf_buffer.base,SFRING,I/128,128,1,4,0);
            descWeight(l->w2,l->down,H,I,LE,mxfp4);
            desc2(l->ws2,l->downSf,H,I/128*LE,128,1,4,0);
            check(cudaFuncSetAttribute(selectKernel<128>(mxfp4),cudaFuncAttributeMaxDynamicSharedMemorySize,Tile<128>::SMEM));
            check(cudaFuncSetAttribute(selectKernel<16>(mxfp4),cudaFuncAttributeMaxDynamicSharedMemorySize,Tile<16>::SMEM));
            check(cudaFuncSetAttribute(selectKernel<32>(mxfp4),cudaFuncAttributeMaxDynamicSharedMemorySize,Tile<32>::SMEM));
            return l.release();
        } catch (const std::exception &e) { error=e.what(); return nullptr; }
    }
    struct PrefillPlan {
        int capacity;
        int smallerCapacity;
        size_t bytes;
        bool (*bind)(Layer &, void *);
        int (*launch)(void *, const void *, void *, const int32_t *, const float *,
                      int, const int *, const void *const *);
    };
    template<int BM,int Capacity>
    static PrefillPlan makePrefillPlan(int smallerCapacity) {
        return {Capacity, smallerCapacity, Geometry<R,Capacity,BM>::buffer(nullptr).get_num_bytes(),
                &bindPrefill<BM,Capacity>, &run<BM,Capacity>};
    }
    static const PrefillPlan *prefillPlan(int total) {
        if (total<=CAP*R) return nullptr;
        // The 512-row tile fits the 64 MiB router tensor at EP8. Larger
        // tiles amortize weight loads across tokens routed to the same expert.
        static const PrefillPlan plans[] = {
            makePrefillPlan<240,32768>(16384), makePrefillPlan<240,16384>(8192),
            makePrefillPlan<240,8192>(4096), makePrefillPlan<R==8 ? 240 : 192,4096>(2048),
            makePrefillPlan<192,2048>(512), makePrefillPlan<R==8 ? 64 : 128,512>(CAP)
        };
        const size_t available = size_t(total)*H*sizeof(float);
        for (const auto &plan : plans) {
            if (total>plan.smallerCapacity*R && plan.bytes<=available) return &plan;
        }
        return nullptr;
    }
    static size_t prefillBytes(int total) {
        const auto *plan=prefillPlan(total);
        return plan ? plan->bytes : 0;
    }
    template<int BM,int Capacity>
    static bool bindPrefill(Layer &l, void *workspace) {
        using G=Geometry<R,Capacity,BM>;
        auto b=G::buffer(workspace);
        if (l.prefillBase!=workspace || !l.preparedPrefill || l.preparedPrefill->capacity!=Capacity) {
            auto status=cudaFuncSetAttribute(selectKernel<BM,Capacity>(l.mxfp4),
                cudaFuncAttributeMaxDynamicSharedMemorySize,Tile<BM>::SMEM);
            if (status==cudaErrorInvalidValue || status==cudaErrorInvalidDeviceFunction ||
                    status==cudaErrorNoKernelImageForDevice) {
                cudaGetLastError();
                return false;
            }
            check(status);
            l.prefillMaps.template bind<BM>(b,G::ring);
            desc2(l.preS1,b.l1_sf_buffer.base,G::sfRing,H/128,Tile<BM>::SF_M,1,4,0);
            desc2(l.preS2,b.l2_sf_buffer.base,G::sfRing,I/128,Tile<BM>::SF_M,1,4,0);
            l.prefillBase=workspace;
        }
        // Router activations overwrite the borrowed control words between layers.
        // The caller publishes an event after this reset on every EP rank.
        check(cudaMemsetAsync(workspace,0,b.workspace.get_num_bytes(),cudaStreamPerThread));
        return true;
    }
    static int preparePrefill(void *p,void *workspace,size_t bytes,int total) {
        try {
            const auto *plan=prefillPlan(total);
            if (!plan || !workspace || uintptr_t(workspace)%256 || bytes<plan->bytes) return 0;
            auto &l=*static_cast<Layer *>(p);
            if (!plan->bind(l, workspace)) return 0;
            l.preparedPrefill=plan;
            return plan->capacity;
        } catch (const std::exception &e) { error=e.what(); return -1; }
    }
    template<int BM=16,int Capacity=CAP> static int run(void *p,const void *x,void *y,const int32_t *idx,const float *scores,int rows,const int *live,const void *const *peers=nullptr) {
        try {
            if (!p || !x || !y || !idx || !scores || rows<0 || rows>Capacity)
                throw std::runtime_error("Invalid Blackwell MegaMoE pointers/row count");
            auto &l=*static_cast<Layer *>(p); int device=-1; check(cudaGetDevice(&device));
            if (device!=l.device) throw std::runtime_error("MegaMoE device mismatch");
            using G=Geometry<R,Capacity,BM>;
            constexpr bool prefill = Capacity!=CAP;
            auto b=G::buffer(prefill ? l.prefillBase : l.workspace);
            auto addresses=l.peers;
            if (prefill) for (int r=0;r<R;++r) addresses[r]=reinterpret_cast<int64_t>(peers[r]);
            auto stream=cudaStreamPerThread;
            prepare<<<rows ? (rows*H+127)/128 : 1,128,0,stream>>>((const __nv_bfloat16 *)x,idx,scores,
                b.input_token_buffer.template get_base_ptr<uint8_t>(),b.input_sf_buffer.template get_base_ptr<uint8_t>(),
                b.input_topk_idx_buffer.template get_base_ptr<int64_t>(),b.input_topk_weights_buffer.template get_base_ptr<float>(),rows,live,
                b.workspace.signals->combine_ready_grid_idx,R);
            cudaLaunchAttribute attr{}; attr.id=cudaLaunchAttributeClusterDimension; attr.val.clusterDim={2,1,1};
            cudaLaunchConfig_t config{}; config.gridDim=dim3(SMS); config.blockDim=dim3(512);
            config.dynamicSmemBytes=Tile<BM>::SMEM; config.stream=stream; config.attrs=&attr; config.numAttrs=1;
            const auto &maps = [&]() -> const ActivationMaps & {
                if constexpr (prefill) return l.prefillMaps;
                if constexpr (BM == 128) return l.wideMaps;
                if constexpr (BM == 32) return l.batchMaps;
                return l.smallMaps;
            }();
            const auto &s1=prefill ? l.preS1 : l.s1;
            const auto &s2=prefill ? l.preS2 : l.s2;
            check(cudaLaunchKernelEx(&config,selectKernel<BM,Capacity>(l.mxfp4),y,(int *)nullptr,(uint32_t)rows,
                dg::layout::SymBuffer<R>(addresses,l.rank),
                maps.a1,s1,l.w1,l.ws1,maps.d1,maps.a2,s2,l.w2,l.ws2,
                maps.a1,s1,l.w1,l.ws1,maps.d1,maps.a2,s2,l.w2,l.ws2,(const uint8_t *)l.domains));
            check(cudaGetLastError()); return 1;
        } catch (const std::exception &e) { error=e.what(); return 0; }
    }
    static int runPrefill(void *p,const void *const *peers,const void *x,void *y,
            const int32_t *idx,const float *scores,int rows) {
        const auto *plan=static_cast<Layer *>(p)->preparedPrefill;
        if (!plan) { error="Unprepared Blackwell MegaMoE prefill"; return 0; }
        return plan->launch(p,x,y,idx,scores,rows,nullptr,peers);
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
    if (inputRows <= ranks * SMALL_CAP) return SMALL_CAP;
    // A full EP8 128-row shard amortizes BM128; EP4 still favors BM32 there.
    return inputRows < ranks * MID_CAP || (ranks == 4 && inputRows == ranks * MID_CAP) ? MID_CAP : CAP;
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
    if ((capacity!=SMALL_CAP && capacity!=MID_CAP && capacity!=CAP) || rows<0 || rows>capacity || !p) {
        error="Invalid Blackwell MegaMoE capacity/row count/handle"; return 0;
    }
    if (capacity==SMALL_CAP)
        return fastllm_sm100_mega_moe_run_masked(p,x,y,i,s,rows,live);
    if (capacity==CAP)
        return static_cast<LayerBase *>(p)->ranks==4 ? Mega<4>::run<128>(p,x,y,i,s,rows,live)
                                                   : Mega<8>::run<128>(p,x,y,i,s,rows,live);
    return static_cast<LayerBase *>(p)->ranks==4 ? Mega<4>::run<32>(p,x,y,i,s,rows,live)
                                               : Mega<8>::run<32>(p,x,y,i,s,rows,live);
}
extern "C" size_t fastllm_sm100_mega_moe_prefill_workspace_bytes(int total,int ranks) {
    return ranks==8 ? Mega<8>::prefillBytes(total) : ranks==4 ? Mega<4>::prefillBytes(total) : 0;
}
extern "C" int fastllm_sm100_mega_moe_prepare_prefill(void *p,void *w,size_t bytes,int total) {
    if (!p) return 0;
    return static_cast<LayerBase *>(p)->ranks==8 ? Mega<8>::preparePrefill(p,w,bytes,total) :
            Mega<4>::preparePrefill(p,w,bytes,total);
}
extern "C" int fastllm_sm100_mega_moe_run_prefill(void *p,const void *const *peers,const void *x,
        void *y,const int32_t *idx,const float *scores,int rows) {
    if (!p || !peers) { error="Invalid Blackwell MegaMoE prefill handle or peers"; return 0; }
    return static_cast<LayerBase *>(p)->ranks==8 ? Mega<8>::runPrefill(p,peers,x,y,idx,scores,rows) :
            Mega<4>::runPrefill(p,peers,x,y,idx,scores,rows);
}
extern "C" void fastllm_sm100_mega_moe_destroy(void *p) {
    if (!p) return;
    if (static_cast<LayerBase *>(p)->ranks==4) delete static_cast<Mega<4>::Layer *>(p);
    else delete static_cast<Mega<8>::Layer *>(p);
}

extern "C" int fastllm_sm100_mega_moe_weight_view(void*p,const void**out,bool *mxfp4){
 if(!p||!out||static_cast<LayerBase*>(p)->ranks!=8)return 0;
 auto*l=static_cast<Mega<8>::Layer*>(p);
 if (!mxfp4 && l->mxfp4) return 0;
 if (mxfp4) *mxfp4 = l->mxfp4;
 out[0]=l->gate;out[1]=l->down;out[2]=l->gateSf;out[3]=l->downSf;return 1;
}
