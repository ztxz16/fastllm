#include "naive-n05-draft-fused.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <functional>
#include "naivert/body/dspark_attention/body.cuh"
#include "naivert/body/dspark_mlp/body.cuh"
#include "naivert/body/dspark_common/rendezvous.cuh"
#include "naivert/body/dspark_context/body.cuh"
#include "naivert/body/dspark_head_markov/head_body.cuh"
#include "naivert/body/dspark_head_markov/cooperative.cuh"
#include <vector>
#include <algorithm>
#include <memory>
#include <string>
#include <stdexcept>
#include <cmath>
#include <cstring>

namespace fastllm_draft_fused {
using namespace fastllm_draft_ops;
using B = __nv_bfloat16;
thread_local std::string error;
void Check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
struct Allocation {
    void *ptr = nullptr; int device = -1;
    Allocation(size_t bytes, int gpu) : device(gpu) {
        Check(cudaSetDevice(gpu)); Check(cudaMalloc(&ptr, bytes));
        Check(cudaMemset(ptr, 0, bytes));
    }
    ~Allocation() { if (ptr) { cudaSetDevice(device); cudaFree(ptr); } }
    Allocation(const Allocation &) = delete;
};
struct Weights {
    const float *inputNorm, *qNorm, *kNorm;
    B *q, *k, *v, *o, *gate, *up, *down;
    const B *rawK, *rawV;
    float *postNorm;
};
struct Lane {
    B *attention, *intermediate, *qk, *k, *v, *attn, *padded;
    float *split, *cosine, *sine;
    int64_t *positions;
    int *epoch;
    unsigned *barrier, *watchdog;
    uint8_t *mailbox;
    B *ctxInput, *ctxSum, *ctxValid;
    float *ctxSsq, *ctxPartials;
    int64_t *ctxDest;
    int64_t *headIds = nullptr, *sampling = nullptr;
    float *headLogits = nullptr, *corrected = nullptr, *samplingParams = nullptr;
    float *softmax = nullptr, *confidence = nullptr;
    unsigned long long *headControl = nullptr;
};
struct ContextPlans {
    int device;
    std::vector<cudaGraphExec_t> execs;
    explicit ContextPlans(int d) : device(d) {}
    ~ContextPlans() {
        cudaSetDevice(device);
        for (auto exec : execs) if (exec) cudaGraphExecDestroy(exec);
    }
};
// The device packet includes an optional accepted count without growing the
// existing allocation. Rows <= 8 and oldLength <= 1023 fit in 16 bits; the
// public host descriptor ABI and its full-width validation remain unchanged.
struct PreparedContextRequest {
    const void *features;
    const void *const *cacheTable;
    const int *accepted;
    int position;
    uint16_t rows, oldLength;
    int tableRequests, tableRequest;
};
static_assert(sizeof(PreparedContextRequest) == sizeof(FastllmDraftContextRequest));
struct ContextInputs {
    int device;
    PreparedContextRequest *host=nullptr,*gpu=nullptr;
    unsigned *alias=nullptr;
    cudaStream_t stream=nullptr;
    cudaEvent_t ready=nullptr,acceptedReady=nullptr;
    bool inFlight=false,hasAccepted=false;
    explicit ContextInputs(int d):device(d) {}
    ~ContextInputs() {
        cudaSetDevice(device);
        if(inFlight)cudaEventSynchronize(ready);
        if(stream)cudaStreamDestroy(stream);
        if(ready)cudaEventDestroy(ready);
        if(acceptedReady)cudaEventDestroy(acceptedReady);
        if(host)cudaFreeHost(host);
        if(gpu)cudaFree(gpu);
        if(alias)cudaFree(alias);
    }
};
struct BeginArgs;
struct HeadBatchLane;
struct Blas {
    cublasHandle_t handle=nullptr;
    int device;
    explicit Blas(int d):device(d) {
        Check(cudaSetDevice(d));
        if(cublasCreate(&handle)!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("draft BLAS creation failed");
    }
    ~Blas(){if(handle){cudaSetDevice(device);cublasDestroy(handle);}}
};
struct Rank {
    std::vector<std::unique_ptr<Allocation>> allocations;
    std::vector<Weights> weights;
    std::vector<Lane> lanes;
    int device;
    B *fc = nullptr;
    float *hiddenNorm = nullptr;
    B *head = nullptr, *biasTable = nullptr;
    const B *rawHead=nullptr;
    float *headLogits=nullptr,*headConfidence=nullptr;
    int64_t *headIds=nullptr;
    HeadBatchLane *headBatch=nullptr;
    int headCapacity=0;
    int contextCapacity=0;
    B *contextInput=nullptr,*contextSum=nullptr;
    float *contextPartial=nullptr;
    BeginArgs *contextBatch=nullptr;
    const B *markovW1 = nullptr;
    const float *confidenceWeight = nullptr, *confidenceBias = nullptr;
    std::unique_ptr<ContextPlans> contextPlans;
    template<class T> T *Alloc(size_t count) {
        allocations.emplace_back(new Allocation(count * sizeof(T), device));
        return static_cast<T *>(allocations.back()->ptr);
    }
};
struct State {
    std::unique_ptr<ContextInputs> contextInputs;
    std::vector<Rank> ranks;
    int layers, rows, window, prepared = 0; float eps, theta;
    bool contextReady = false, headReady = false;
    bool programmatic = false;
    bool batchMlp = true, batchMlp4 = true;
    bool batchAttention = true, batchAttention4 = true;
    bool batchHead = true, batchHead4 = true;
    bool batchContext = true, batchContext4 = true;
};
void PrepareContextPlans(State &s, int requests);
__global__ void Pack(const B *source, B *out, int rows, int width, bool interleave) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(rows) * width) return;
    int row = i / width, col = i % width;
    int srcRow = interleave ? row / 2 + (row % 2) * (rows / 2) : row;
    size_t dst = ((size_t(row / 16) * (width / 32) + col / 32) * 16 + row % 16) * 32 + col % 32;
    out[dst] = source[size_t(srcRow) * width + col];
}
__global__ void PackNorm(const float *source, float *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 4096) out[naivert::attn_rows::PackedNormGamma::offset(i)] = source[i];
}
__global__ void ShardFc(const B *source, B *out, int rank) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 4096 * 4096) return;
    int row = i / 4096, col = i % 4096;
    out[i] = source[size_t(row) * 32768 + rank * 4096 + col];
}
__device__ void ContextInputBody(const B *input, B *out, B *valid, int rows,
        int position, int keep, int window, int64_t *positions, int64_t *destinations,
        float *cosine, float *sine, float theta) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 8 * 4096; i += blockDim.x * gridDim.x)
        out[i] = i / 4096 < rows ? input[i] : __float2bfloat16_rn(0.f);
    if (!blockIdx.x) {
        for (int i = threadIdx.x; i < 8 * 128; i += blockDim.x) {
            int row = i / 128, col = i % 128;
            float angle = float(position + row) * powf(theta, -2.f * (col % 64) / 128.f);
            cosine[i] = float(__float2bfloat16_rn(cosf(angle)));
            sine[i] = float(__float2bfloat16_rn(sinf(angle)));
            if (!col) {
                positions[row] = position + row;
                destinations[row] = row < rows ? keep + row : window + row - 1;
                valid[row] = __float2bfloat16_rn(row < rows ? 1.f : 0.f);
            }
        }
    }
}
__global__ void ContextInput(const B *input, B *out, B *valid, int rows,
        int position, int keep, int window, int64_t *positions, int64_t *destinations,
        float *cosine, float *sine, float theta) {
    ContextInputBody(input,out,valid,rows,position,keep,window,positions,destinations,cosine,sine,theta);
}
__device__ void ContextTrimBody(const void *const *table, int count, int request, int keep, int drop,
        uint4 *values) {
    int layer = blockIdx.x / 8, kind = (blockIdx.x / 4) % 2, columns = (blockIdx.x % 4) * 32;
    B *base = static_cast<B *>(const_cast<void *>(table[(layer * count + request) * 2 + kind]));
    for (int i = threadIdx.x; i < keep * 4; i += blockDim.x)
        values[i] = *reinterpret_cast<const uint4 *>(base + (i / 4 + drop) * 128 + columns + i % 4 * 8);
    __syncthreads();
    for (int i = threadIdx.x; i < keep * 4; i += blockDim.x)
        *reinterpret_cast<uint4 *>(base + (i / 4) * 128 + columns + i % 4 * 8) = values[i];
}
__global__ void ContextTrim(const void *const *table, int count, int request, int keep, int drop) {
    extern __shared__ uint4 values[];
    ContextTrimBody(table,count,request,keep,drop,values);
}
constexpr int ContextSmem = kD12WeightOffset + kD12WeightBytes;
__global__ __launch_bounds__(kThreads, 1) void FusedDraftContext(const __grid_constant__ FcCtxKvArgs a) {
    extern __shared__ __align__(128) unsigned char memory[];
    __syncthreads(); FcCtxKvBody::run(memory, a);
}
__device__ void ContextPreparedBody(unsigned char *memory, const FcCtxKvArgs &base, const PreparedContextRequest *request,
        int window, float theta) {
    const auto &d = *request;
    const int rows = d.accepted ? *d.accepted+1 : d.rows;
    if (rows < 1 || rows > d.rows) { asm volatile("trap;"); return; }
    const int keep = min(int(d.oldLength),window-1-rows), drop = d.oldLength-keep;
    static_assert(ContextSmem >= 1024*32*2);
    if (blockIdx.x < 40 && keep && drop)
        ContextTrimBody(d.cacheTable,d.tableRequests,d.tableRequest,keep,drop,
            reinterpret_cast<uint4 *>(memory));
    // FC consumes the request's immutable features directly and pads in each
    // CTA's shared memory. No cross-CTA input preparation remains. Join only
    // local trim readers before reusing shared memory; FC's existing grid join
    // retires every trim before any later KV output can overwrite its source.
    __syncthreads();
    FcCtxKvArgs a = base;
    a.fc.target_hidden=static_cast<const B*>(d.features);
    a.kv.cache_table = request->cacheTable;
    a.kv.table_requests = request->tableRequests; a.kv.table_request = request->tableRequest;
    FcCtxKvBody::run<true>(memory,a,{rows,d.position,keep,window,theta});
}
__global__ __launch_bounds__(kThreads, 1) void FusedDraftContextPrepared(
        const __grid_constant__ FcCtxKvArgs base, const PreparedContextRequest *request,
        int window, float theta,const unsigned *fallback=nullptr) {
    if(fallback && !*fallback)return;
    extern __shared__ __align__(128) unsigned char memory[];
    ContextPreparedBody(memory,base,request,window,theta);
}
struct ContextTileArgs { FcCtxKvArgs requests[2]; };
__device__ bool ContextOverlap(const void *a,size_t na,const void *b,size_t nb) {
    const uintptr_t x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    return x<=y ? y-x<na : x-y<nb;
}
__global__ __launch_bounds__(kThreads,1) void FusedDraftContextTile(
        const __grid_constant__ ContextTileArgs tile, const PreparedContextRequest *requests,
        int window,float theta,const unsigned *fallback=nullptr) {
    if(fallback && !*fallback)return;
    extern __shared__ __align__(128) unsigned char memory[];
    const int tid=threadIdx.x;
    PreparedContextRequest d[2]={requests[0],requests[1]};
    int rows[2],keep[2];
#pragma unroll
    for(int request=0;request<2;++request) {
        rows[request]=d[request].accepted ? *d[request].accepted+1 : d[request].rows;
        if(rows[request]<1 || rows[request]>d[request].rows){asm volatile("trap;");return;}
        keep[request]=min(int(d[request].oldLength),window-1-rows[request]);
    }
    // Retain sequential semantics when requests share cache storage, or a
    // later request consumes features inside another request's cache. The
    // comparison is uniform across the fully resident grid and adds no buffer.
    bool alias=false;
    constexpr int caches=10;
    const size_t cacheBytes=size_t(window+8)*128*sizeof(B);
    if(tid<caches*caches) {
        int first=tid/caches,second=tid%caches;
        const void *a=d[0].cacheTable[((first/2)*d[0].tableRequests+d[0].tableRequest)*2+first%2];
        const void *b=d[1].cacheTable[((second/2)*d[1].tableRequests+d[1].tableRequest)*2+second%2];
        alias=ContextOverlap(a,cacheBytes,b,cacheBytes);
    }
    if(tid<2*caches) {
        int request=tid/caches,index=tid%caches;
        const void *cache=d[request].cacheTable[((index/2)*d[request].tableRequests+d[request].tableRequest)*2+index%2];
        alias=alias || ContextOverlap(cache,cacheBytes,d[1-request].features,size_t(rows[1-request])*4096*sizeof(B));
    }
    if(__syncthreads_or(alias)) {
        ContextPreparedBody(memory,tile.requests[0],requests,window,theta);
        grid_barrier(tile.requests[0].fc.barrier);
        ContextPreparedBody(memory,tile.requests[1],requests+1,window,theta);
        return;
    }
#pragma unroll
    for(int request=0;request<2;++request) {
        int drop=d[request].oldLength-keep[request];
        if(blockIdx.x<40 && keep[request] && drop)
            ContextTrimBody(d[request].cacheTable,d[request].tableRequests,d[request].tableRequest,
                keep[request],drop,reinterpret_cast<uint4*>(memory));
        __syncthreads();
        if(!blockIdx.x && !tid) {
            const auto &a=tile.requests[request].fc;
            rendezvous_announce(a.mailbox_peers,a.rank,unsigned(*a.replay_generation));
        }
    }
    constexpr int stride=4096+32;
    constexpr int inputBytes=16*stride*sizeof(B);
    constexpr int partialBytes=2*kShellWarps*128*sizeof(float);
    static_assert(inputBytes+partialBytes+2*8*32*sizeof(float)<=ContextSmem);
    auto *input=reinterpret_cast<B*>(memory);
    auto *partial=reinterpret_cast<float*>(memory+inputBytes);
    auto *projected=reinterpret_cast<float*>(memory+inputBytes+partialBytes);
    for(int i=tid;i<16*512;i+=kThreads) {
        const int row=i/512,col=i%512,request=row/8,local=row%8;
        *reinterpret_cast<uint4*>(input+row*stride+col*8)=local<rows[request] ?
            reinterpret_cast<const uint4*>(d[request].features)[local*512+col] : make_uint4(0,0,0,0);
    }
    __syncthreads();
    naivert::attn_rows::project_strided_rows_mma<16,4096,4,2,kShellWarps,stride>(0,4096/16,
        [&](int column) {
            return tile.requests[0].fc.fc+size_t(column)*4096;
        },input,partial,[&](int column,int row,float value) {
            projected[(row/8)*256+(row%8)*32+column-int(blockIdx.x)*32]=value;
        });
    const float values[2]={tid<256?projected[tid]:0.f,tid<256?projected[256+tid]:0.f};
    __syncthreads();
#pragma unroll
    for(int request=0;request<2;++request) {
        auto a=tile.requests[request];
        a.kv.cache_table=d[request].cacheTable;
        a.kv.table_requests=d[request].tableRequests;a.kv.table_request=d[request].tableRequest;
        FcCtxKvBody::run<true,true>(memory,a,{rows[request],d[request].position,keep[request],window,theta},values[request]);
        __syncthreads();
    }
}

template<int Requests> struct ContextRequests { FcCtxKvArgs lanes[Requests]; };
template<int Requests>
__global__ __launch_bounds__(kThreads,1) void FusedDraftContextRequests(
        const __grid_constant__ ContextRequests<Requests> tile,const PreparedContextRequest* requests,
        int window,float theta,const unsigned *fallback=nullptr) {
    if(fallback && !*fallback)return;
    extern __shared__ __align__(128) unsigned char memory[];
    const int tid=threadIdx.x;
    bool alias=false;
    const size_t cacheBytes=size_t(window+8)*128*sizeof(B);
    for(int first=0;first<Requests;++first){
        const auto &d=requests[first];const int rows=d.accepted?*d.accepted+1:d.rows;
        if(rows<1 || rows>d.rows){asm volatile("trap;");return;}
        for(int second=first+1;second<Requests;++second){
            const auto &e=requests[second];const int otherRows=e.accepted?*e.accepted+1:e.rows;
            if(tid<100){int a=tid/10,b=tid%10;
                const void *x=d.cacheTable[((a/2)*d.tableRequests+d.tableRequest)*2+a%2];
                const void *y=e.cacheTable[((b/2)*e.tableRequests+e.tableRequest)*2+b%2];
                alias=alias || ContextOverlap(x,cacheBytes,y,cacheBytes);
            }
            if(tid<10){
                const void *x=d.cacheTable[((tid/2)*d.tableRequests+d.tableRequest)*2+tid%2];
                const void *y=e.cacheTable[((tid/2)*e.tableRequests+e.tableRequest)*2+tid%2];
                alias=alias || ContextOverlap(x,cacheBytes,e.features,size_t(otherRows)*4096*sizeof(B)) ||
                    ContextOverlap(y,cacheBytes,d.features,size_t(rows)*4096*sizeof(B));
            }
        }
    }
    if(__syncthreads_or(alias)){
        for(int request=0;request<Requests;++request){
            ContextPreparedBody(memory,tile.lanes[request],requests+request,window,theta);
            if(request+1<Requests)grid_barrier(tile.lanes[0].fc.barrier);
        }
        return;
    }
#pragma unroll
    for(int request=0;request<Requests;++request){
        const auto &d=requests[request];const int rows=d.accepted?*d.accepted+1:d.rows;
        const int keep=min(int(d.oldLength),window-1-rows),drop=d.oldLength-keep;
        if(blockIdx.x<40 && keep && drop)ContextTrimBody(d.cacheTable,d.tableRequests,d.tableRequest,keep,drop,reinterpret_cast<uint4*>(memory));
        __syncthreads();
        if(!blockIdx.x && !tid){const auto &a=tile.lanes[request].fc;
            rendezvous_announce(a.mailbox_peers,a.rank,unsigned(*a.replay_generation));}
    }
    FcCtxKvBody::run_requests<Requests>(memory,tile.lanes,requests,window,theta);
}

struct BeginArgs {
    const uint8_t *peers[8]; int rank; int *epoch; unsigned *watchdog;
    const int *live; int64_t *positions; float *cosine, *sine; float theta;
    const B *input; B *output; int rows;
    int64_t *headIds = nullptr, *sampling = nullptr;
    float *samplingParams = nullptr;
    int anchor = 0; uint64_t seed = 0; float inverseTemperature = 1;
};
__device__ __forceinline__ void BeginBody(BeginArgs a) {
    naivert::pdl::trigger_dependents<true>();
    // The preceding kernel boundary retires every old mailbox reader. Publish
    // that fact before local preparation so the peer round trip can overlap it.
    // The next collective still waits for this entire kernel to complete.
    const unsigned old = !threadIdx.x ? *a.epoch : 0;
    if (!threadIdx.x) rendezvous_announce(a.peers,a.rank,old);
    if (a.input) {
        if (!(reinterpret_cast<uintptr_t>(a.input) & 15) &&
            !(reinterpret_cast<uintptr_t>(a.output) & 15)) {
            const auto *input = reinterpret_cast<const uint4 *>(a.input);
            auto *output = reinterpret_cast<uint4 *>(a.output);
            for (int i = threadIdx.x; i < 8 * 512; i += blockDim.x)
                output[i] = i < a.rows * 512 ? input[i] : make_uint4(0,0,0,0);
        } else {
            for (int i = threadIdx.x; i < 8 * 4096; i += blockDim.x)
                a.output[i] = i < a.rows * 4096 ? a.input[i] : __float2bfloat16_rn(0.f);
        }
    }
    if (a.live) for (int i = threadIdx.x; i < 8 * 128; i += blockDim.x) {
        int row = i / 128, col = i % 128;
        int position = *a.live - 1 + row;
        float angle = float(position) * powf(a.theta, -2.f * (col % 64) / 128.f);
        a.cosine[i] = float(__float2bfloat16_rn(cosf(angle)));
        a.sine[i] = float(__float2bfloat16_rn(sinf(angle)));
        if (!col) a.positions[row] = position;
    }
    if (!threadIdx.x) {
        rendezvous_wait(a.peers[a.rank],a.rank,old,
            naivert::MailboxWatchdog{5000000000ull,a.watchdog});
        *a.epoch = int(old + 1);
        if (a.headIds) {
            a.headIds[0] = a.anchor; a.sampling[0] = int64_t(a.seed); a.sampling[1] = 0;
            a.samplingParams[4] = a.inverseTemperature;
        }
    }
}
__global__ void Begin(BeginArgs a) { BeginBody(a); }
__global__ void BeginHeadPrepared(BeginArgs a,const FastllmDraftHeadRequest *request) {
    a.anchor=request->anchor; a.seed=request->seed; a.inverseTemperature=__fdiv_rn(1.f,request->temperature);
    BeginBody(a);
}
template<int Ctas> __device__ void DraftAttentionBody(SwaAttnArgs a,
        const void *const *table,const int *live,int request,int requests,unsigned char *memory) {
    a.kv_k = static_cast<const B *>(table[(a.layer * requests + request) * 2]);
    a.kv_v = static_cast<const B *>(table[(a.layer * requests + request) * 2 + 1]);
    a.ncache = max(1, min(*live - 1, a.window_left));
    a.total_cols = a.ncache + a.valid_rows;
    a.padded_cols = (a.total_cols + 255) / 256 * 256;
    AttnRowsBody<Ctas>::template run<true>(a, memory);
    // O projection has acquired all four merged attention heads: every KV
    // reader has finished. Publish the speculative suffix only after that
    // join, including the zero-history case where physical row zero is reused.
    if (!blockIdx.x) for (int i = threadIdx.x; i < a.valid_rows * 128; i += blockDim.x) {
        int offset = min(*live - 1, a.window_left) * 128 + i;
        const_cast<B *>(a.kv_k)[offset] = a.k_blk[i];
        const_cast<B *>(a.kv_v)[offset] = a.v_blk[i];
    }
}
__global__ __launch_bounds__(kAttnThreads, 1) void FusedDraftAttention(
        SwaAttnArgs a,const void *const *table,const int *live,int request,int requests) {
    extern __shared__ __align__(128) unsigned char memory[];
    DraftAttentionBody<128>(a,table,live,request,requests,memory);
}
struct AttentionTileArgs { SwaAttnArgs requests[2]; };
__global__ __launch_bounds__(kAttnThreads, 1) void FusedDraftAttentionTile(
        const __grid_constant__ AttentionTileArgs tile,const void *const *table,
        const int *live,int first,int requests) {
    extern __shared__ __align__(128) unsigned char memory[];
    const int request=first+blockIdx.y;
    DraftAttentionBody<64>(tile.requests[blockIdx.y],table,live+request,request,requests,memory);
}
template<int Requests> struct AttentionRequests { SwaAttnArgs lanes[Requests]; };
template<int Requests>
__global__ __launch_bounds__(kAttnThreads,1) void FusedDraftAttentionRequests(
        const __grid_constant__ AttentionRequests<Requests> tile,const void *const *table,
        const int *live,int first,int count) {
    extern __shared__ __align__(128) unsigned char memory[];
    const int local=blockIdx.x/(128/Requests),request=first+local;
    SwaAttnArgs a=tile.lanes[local];
    a.kv_k=static_cast<const B*>(table[(a.layer*count+request)*2]);
    a.kv_v=static_cast<const B*>(table[(a.layer*count+request)*2+1]);
    a.ncache=max(1,min(live[request]-1,a.window_left));
    a.total_cols=a.ncache+a.valid_rows;a.padded_cols=(a.total_cols+255)/256*256;
    AttnRowsBody<128/Requests,Requests>::template run<true>(a,memory,tile.lanes);
    if(blockIdx.x%(128/Requests)==0){
        for(int i=threadIdx.x;i<a.valid_rows*128;i+=blockDim.x){
            int at=min(live[request]-1,a.window_left)*128+i;
            const_cast<B*>(a.kv_k)[at]=a.k_blk[i];const_cast<B*>(a.kv_v)[at]=a.v_blk[i];
        }
    }
}
__global__ __launch_bounds__(kMlpThreads, 1) void FusedDraftMlp(const __grid_constant__ MlpArgs a) {
    extern __shared__ __align__(128) unsigned char memory[];
    MlpBody::run(a, memory);
}
struct MlpTileArgs { MlpArgs requests[2]; };
__global__ __launch_bounds__(kMlpThreads, 1) void FusedDraftMlpTile(
        const __grid_constant__ MlpTileArgs tile) {
    extern __shared__ __align__(128) unsigned char memory[];
    MlpRowsBody<2>::run(tile.requests[0], memory, tile.requests + 1);
}
struct MlpQuadArgs { MlpArgs requests[4]; };
__global__ __launch_bounds__(kMlpThreads, 1) void FusedDraftMlpQuad(
        const __grid_constant__ MlpQuadArgs tile) {
    extern __shared__ __align__(128) unsigned char memory[];
    MlpRowsBody<4>::run(tile.requests[0],memory,tile.requests+1);
}
__global__ void FinishResult(const B *input, B *output, int count, const unsigned *watchdog) {
    // A timed-out peer exchange cannot fall back after partial writes. Surface
    // the asynchronous CUDA error before any consumer can use an invalid result.
    if (*watchdog & naivert::kMailboxWatchdogTripped) asm volatile("trap;");
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) output[i] = input[i];
}
__global__ void FinishNormalized(const B *input,B *output,B *normalized,
        const float *gamma,float eps,const unsigned *watchdog) {
    if (*watchdog & naivert::kMailboxWatchdogTripped) asm volatile("trap;");
    const int row=blockIdx.x,tid=threadIdx.x;
    const int base=row*4096+tid*16;
    uint4 raw[2];float values[16],weights[16],sum=0.f;
    #pragma unroll
    for(int part=0;part<2;++part) {
        raw[part]=*reinterpret_cast<const uint4*>(input+base+part*8);
        const auto *words=reinterpret_cast<const unsigned*>(raw+part);
        #pragma unroll
        for(int i=0;i<4;++i) {
            float2 v=__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(words+i));
            values[part*8+i*2]=v.x;values[part*8+i*2+1]=v.y;
        }
    }
    #pragma unroll
    for(int part=0;part<4;++part) {
        const float4 g=*reinterpret_cast<const float4*>(gamma+tid*16+part*4);
        weights[part*4]=g.x;weights[part*4+1]=g.y;weights[part*4+2]=g.z;weights[part*4+3]=g.w;
    }
    #pragma unroll
    for(int i=0;i<16;++i) {
        sum+=values[i]*values[i];
    }
    #pragma unroll
    for(int delta=16;delta;delta/=2) sum+=__shfl_xor_sync(0xffffffffu,sum,delta);
    __shared__ float sums[8];
    if(!(tid&31))sums[tid/32]=sum;
    __syncthreads();
    if(tid<32) {
        sum=tid<8?sums[tid]:0.f;
        #pragma unroll
        for(int delta=16;delta;delta/=2)sum+=__shfl_xor_sync(0xffffffffu,sum,delta);
        if(!tid)sums[0]=rsqrtf(sum/4096.f+eps);
    }
    __syncthreads();
    const float scale=sums[0];
    #pragma unroll
    for(int part=0;part<2;++part) {
        uint4 packed;auto *words=reinterpret_cast<unsigned*>(&packed);
        #pragma unroll
        for(int i=0;i<4;++i) {
            const int j=part*8+i*2;
            // Preserve the BF16 boundary before multiplication by FP32 gamma.
            const auto v=__floats2bfloat162_rn(
                float(__float2bfloat16_rn(values[j]*scale))*weights[j],
                float(__float2bfloat16_rn(values[j+1]*scale))*weights[j+1]);
            words[i]=*reinterpret_cast<const unsigned*>(&v);
        }
        *reinterpret_cast<uint4*>(output+base+part*8)=raw[part];
        *reinterpret_cast<uint4*>(normalized+base+part*8)=packed;
    }
}
template<class Kernel,class... Args>
void LaunchDependent(bool enabled,Kernel kernel,dim3 grid,dim3 block,int shared,
        cudaStream_t stream,Args... args) {
#if CUDART_VERSION >= 12030
    if(enabled) {
        cudaLaunchConfig_t config{};
        config.gridDim=grid;config.blockDim=block;config.dynamicSmemBytes=shared;config.stream=stream;
        cudaLaunchAttribute attribute{};attribute.id=cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute.val.programmaticStreamSerializationAllowed=1;
        config.attrs=&attribute;config.numAttrs=1;
        Check(cudaLaunchKernelEx(&config,kernel,args...));return;
    }
#endif
    void *parameters[]={static_cast<void*>(&args)...};
    Check(cudaLaunchKernel(reinterpret_cast<const void*>(kernel),grid,block,parameters,shared,stream));
}
void Configure(int device) {
    Check(cudaSetDevice(device));
    Check(cudaFuncSetAttribute(FusedDraftAttention, cudaFuncAttributeMaxDynamicSharedMemorySize, kAttnSmem));
    Check(cudaFuncSetAttribute(FusedDraftMlp, cudaFuncAttributeMaxDynamicSharedMemorySize, kMlpSmem));
    int sms, attn, mlp;
    Check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&attn, FusedDraftAttention, kAttnThreads, kAttnSmem));
    Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mlp, FusedDraftMlp, kMlpThreads, kMlpSmem));
    if (attn * sms < kAttnCtas || mlp * sms < kMlpCtas)
        throw std::runtime_error("fused draft requires a fully resident grid");
}
template <class Kernel>
bool ConfigureOptionalKernel(int device, Kernel kernel, int threads, int shared, int ctas) {
    int blocks = 0, sms = 0;
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, shared) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) {
        cudaGetLastError(); return false;
    }
    return blocks * sms >= ctas;
}
void Start(State &s, int rank, int request, const int *live, cudaStream_t stream, const B *input) {
    auto &l = s.ranks[rank].lanes[request];
    BeginArgs a{};
    for (int i = 0; i < 8; ++i) a.peers[i] = s.ranks[i].lanes[request].mailbox;
    a.rank = rank; a.epoch = l.epoch; a.watchdog = l.watchdog; a.live = live;
    a.positions = l.positions; a.cosine = l.cosine; a.sine = l.sine; a.theta = s.theta;
    a.input = input; a.output = l.padded; a.rows = s.rows;
    Begin<<<1, 256, 0, stream>>>(a);
    Check(cudaGetLastError());
}
SwaAttnArgs AttentionArguments(State &s,int rank,int request,int layer,const B *input,B *output) {
    auto &r = s.ranks[rank]; auto &l = r.lanes[request]; auto &w = r.weights[layer];
    SwaAttnArgs a{};
    a.resid_in = input; a.input_ln = w.inputNorm; a.q_w = w.q; a.k_w = w.k;
    a.v_w = w.v; a.o_w = w.o; a.q_norm = w.qNorm; a.k_norm = w.kNorm;
    a.rope_cos = l.cosine; a.rope_sin = l.sine; a.blk_pos = l.positions;
    a.qk_raw = l.qk; a.k_blk = l.k; a.v_blk = l.v; a.attn_out = l.attn; a.hid_a = output;
    a.barrier = l.barrier; a.replay_generation = l.epoch;
    for (int i = 0; i < 8; ++i) a.mailbox_peers[i] = s.ranks[i].lanes[request].mailbox;
    a.generation_salt = kSaltD3Base + 2 * layer; a.attn_scale = float(1. / std::sqrt(128.));
    a.layer = layer; a.cache_rows = s.window + 8; a.rank = rank;
    a.watchdog_limit_ns = 5000000000ull; a.watchdog_flag = l.watchdog;
    a.split_scratch = l.split; a.window_left = s.window - 1; a.rms_epsilon = s.eps;
    a.valid_rows = s.rows;
    return a;
}
void Attention(State &s,int rank,int request,int layer,const B *input,B *output,
        const int *live,const void *const *table,int requests,cudaStream_t stream) {
    LaunchDependent(s.programmatic,FusedDraftAttention,kAttnCtas,kAttnThreads,kAttnSmem,stream,
        AttentionArguments(s,rank,request,layer,input,output),table,live,request,requests);
}
template<int Requests>
void AttentionShared(State &s,int rank,int first,int layer,const int *live,
        const void *const *table,int count,cudaStream_t stream) {
    AttentionRequests<Requests> tile{};
    for(int i=0;i<Requests;++i){auto &l=s.ranks[rank].lanes[first+i];
        tile.lanes[i]=AttentionArguments(s,rank,first+i,layer,l.padded,l.attention);}
    LaunchDependent(s.programmatic,FusedDraftAttentionRequests<Requests>,128,kAttnThreads,
        AttnRowsBody<128/Requests,Requests>::kSharedBytes,stream,tile,table,live,first,count);
}
void AttentionTile(State &s,int rank,int first,int layer,const int *live,
        const void *const *table,int requests,cudaStream_t stream) {
    AttentionTileArgs tile{};
    for(int i=0;i<2;++i) {
        auto &l=s.ranks[rank].lanes[first+i];
        tile.requests[i]=AttentionArguments(s,rank,first+i,layer,l.padded,l.attention);
    }
    LaunchDependent(s.programmatic,FusedDraftAttentionTile,dim3(64,2),kAttnThreads,
        AttnRowsBody<64>::kSharedBytes,stream,tile,table,live,first,requests);
}
MlpArgs MlpArguments(State &s, int rank, int request, int layer, const B *input, B *output) {
    auto &r = s.ranks[rank]; auto &l = r.lanes[request]; auto &w = r.weights[layer];
    MlpArgs a{};
    a.hid_a = input; a.post_ln = w.postNorm; a.gate_w = w.gate; a.up_w = w.up;
    a.down_w = w.down; a.mlp_h = l.intermediate; a.resid_out = output; a.barrier = l.barrier;
    for (int i = 0; i < 8; ++i) a.mailbox_peers[i] = s.ranks[i].lanes[request].mailbox;
    a.replay_generation = l.epoch; a.generation_salt = kSaltD4Base + 2 * layer; a.rank = rank;
    a.watchdog_limit_ns = 5000000000ull; a.watchdog_flag = l.watchdog; a.rms_epsilon = s.eps;
    return a;
}
void Mlp(State &s, int rank, int request, int layer, const B *input, B *output, cudaStream_t stream) {
    LaunchDependent(s.programmatic,FusedDraftMlp,kMlpCtas,kMlpThreads,kMlpSmem,stream,
        MlpArguments(s,rank,request,layer,input,output));
}
void MlpTile(State &s,int rank,int first,int layer,bool probe,cudaStream_t stream) {
    MlpTileArgs tile{};
    for(int i=0;i<2;++i) {
        auto &lane=s.ranks[rank].lanes[first+i];
        tile.requests[i]=MlpArguments(s,rank,first+i,layer,
            probe?lane.padded:lane.attention,probe?lane.attention:lane.padded);
    }
    LaunchDependent(s.programmatic,FusedDraftMlpTile,kMlpCtas,kMlpThreads,
        MlpRowsBody<2>::kSharedBytes,stream,tile);
}
void MlpQuad(State &s,int rank,int first,int layer,bool probe,cudaStream_t stream) {
    MlpQuadArgs tile{};
    for(int i=0;i<4;++i) {
        auto &lane=s.ranks[rank].lanes[first+i];
        tile.requests[i]=MlpArguments(s,rank,first+i,layer,
            probe?lane.padded:lane.attention,probe?lane.attention:lane.padded);
    }
    LaunchDependent(s.programmatic,FusedDraftMlpQuad,kMlpCtas,kMlpThreads,
        MlpRowsBody<4>::kSharedBytes,stream,tile);
}
void RunBody(State &s,int rank,int requests,B *hidden,const int *live,
        const void *const *table,const float *gamma,B *normalized,cudaStream_t stream) {
    for(int first=0;first<requests;) {
        const int count=s.batchMlp4 && first+3<requests ? 4 :
            (s.batchMlp || s.batchAttention) && first+1<requests ? 2 : 1;
        for(int request=first;request<first+count;++request)
            Start(s,rank,request,live+request,stream,hidden+size_t(request)*s.rows*4096);
        for(int layer=0;layer<s.layers;++layer) {
            if(count==4 && s.batchAttention4) AttentionShared<4>(s,rank,first,layer,live,table,requests,stream);
            else if(count>=2 && s.batchAttention) {
                for(int offset=0;offset<count;offset+=2)
                    AttentionTile(s,rank,first+offset,layer,live,table,requests,stream);
            }
            else for(int request=first;request<first+count;++request) {
                auto &l=s.ranks[rank].lanes[request];
                Attention(s,rank,request,layer,l.padded,l.attention,live+request,table,requests,stream);
            }
            if(count==4) MlpQuad(s,rank,first,layer,false,stream);
            else if(count==2 && s.batchMlp) MlpTile(s,rank,first,layer,false,stream);
            else for(int request=first;request<first+count;++request) {
                auto &l=s.ranks[rank].lanes[request];
                Mlp(s,rank,request,layer,l.attention,l.padded,stream);
            }
        }
        for(int request=first;request<first+count;++request) {
            auto &l=s.ranks[rank].lanes[request];
            const size_t offset=size_t(request)*s.rows*4096;
            if(normalized)
                FinishNormalized<<<s.rows,256,0,stream>>>(l.padded,hidden+offset,
                    normalized+offset,gamma,s.eps,l.watchdog);
            else
                FinishResult<<<s.rows*4096/256,256,0,stream>>>(l.padded,hidden+offset,s.rows*4096,l.watchdog);
            Check(cudaGetLastError());
        }
        first+=count;
    }
}
}
using namespace fastllm_draft_fused;
extern "C" bool FastllmDraftFusedSupported(const int *devices, int ranks) {
    if (!devices || ranks != 8) return false;
    for (int i = 0; i < ranks; ++i) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, devices[i]) != cudaSuccess || p.major != 10 || p.minor != 3) return false;
        for (int j = 0; j < i; ++j) {
            int access = 0;
            if (devices[i] == devices[j] || cudaDeviceCanAccessPeer(&access, devices[i], devices[j]) != cudaSuccess || !access) return false;
        }
    }
    return true;
}
extern "C" void *FastllmDraftFusedCreate(const int *devices, int ranks, int layers,
        const FastllmDraftFusedWeight *weights, int rows, int window, float eps, float theta) {
    if (!FastllmDraftFusedSupported(devices, ranks) || !weights || layers < 1 || layers > 5 ||
        rows < 2 || rows > 8 || window < 2 || window > 16384 || !std::isfinite(eps) || eps <= 0 || !std::isfinite(theta) || theta <= 0) return nullptr;
    try {
        std::unique_ptr<State> s(new State()); s->ranks.resize(ranks);
        s->layers = layers; s->rows = rows; s->window = window; s->eps = eps; s->theta = theta;
#if CUDART_VERSION >= 12030
        int runtime=0,driver=0;
        Check(cudaRuntimeGetVersion(&runtime));Check(cudaDriverGetVersion(&driver));
        s->programmatic=runtime>=12030 && driver>=12030;
#endif
        for (int rank = 0; rank < ranks; ++rank) {
            auto &r = s->ranks[rank]; r.device = devices[rank]; Configure(r.device);
            s->batchMlp4 = ConfigureOptionalKernel(r.device, FusedDraftMlpQuad,
                kMlpThreads,MlpRowsBody<4>::kSharedBytes,kMlpCtas) && s->batchMlp4;
            s->batchMlp = ConfigureOptionalKernel(r.device, FusedDraftMlpTile,
                kMlpThreads, MlpRowsBody<2>::kSharedBytes, kMlpCtas) && s->batchMlp;
            s->batchAttention4 = ConfigureOptionalKernel(r.device,FusedDraftAttentionRequests<4>,
                kAttnThreads,AttnRowsBody<32,4>::kSharedBytes,128) && s->batchAttention4 && window<=AttnRowsBody<32,4>::kMaxHistory;
            s->batchAttention = ConfigureOptionalKernel(r.device, FusedDraftAttentionTile,
                kAttnThreads, AttnRowsBody<64>::kSharedBytes, kAttnCtas) && s->batchAttention &&
                window <= AttnRowsBody<64>::kMaxHistory;
            for (int peer = 0; peer < ranks; ++peer) if (peer != rank) {
                auto result = cudaDeviceEnablePeerAccess(devices[peer], 0);
                if (result == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError(); else Check(result);
            }
            for (int layer = 0; layer < layers; ++layer) {
                const auto &x = weights[rank * layers + layer]; Weights w{};
                if (!x.qkv || !x.o || !x.inputNorm || !x.qNorm || !x.kNorm || !x.gateUp || !x.down || !x.postNorm)
                    throw std::runtime_error("null fused draft weight");
                auto pack = [&](const void *p, int n, int k, bool gu = false) {
                    B *out = r.Alloc<B>(size_t(n) * k);
                    Pack<<<(size_t(n) * k + 255) / 256, 256>>>(static_cast<const B *>(p), out, n, k, gu);
                    Check(cudaGetLastError()); return out;
                };
                w.q = pack(x.qkv, 512, 4096);
                w.k = pack(static_cast<const B *>(x.qkv) + 512 * 4096, 128, 4096);
                w.v = pack(static_cast<const B *>(x.qkv) + 640 * 4096, 128, 4096);
                w.o = pack(x.o, 4096, 512); w.gate = pack(x.gateUp, 1024, 4096, true);
                w.up = w.gate + 512 * 4096; w.down = pack(x.down, 4096, 512);
                w.postNorm = r.Alloc<float>(4096);
                PackNorm<<<16, 256>>>(static_cast<const float *>(x.postNorm), w.postNorm);
                w.inputNorm = static_cast<const float *>(x.inputNorm);
                w.qNorm = static_cast<const float *>(x.qNorm); w.kNorm = static_cast<const float *>(x.kNorm);
                w.rawK = static_cast<const B *>(x.qkv) + 512 * 4096;
                w.rawV = static_cast<const B *>(x.qkv) + 640 * 4096;
                r.weights.push_back(w);
            }
            Check(cudaDeviceSynchronize());
        }
        Check(cudaSetDevice(devices[0])); return s.release();
    } catch (const std::exception &e) { error = e.what(); return nullptr; }
}
namespace fastllm_draft_fused {
void AllocateHeadLane(Rank &r, Lane &l, int index) {
    const bool contiguous=index<r.headCapacity;
    l.headIds = contiguous ? r.headIds+size_t(index)*8 : r.Alloc<int64_t>(8); l.sampling = r.Alloc<int64_t>(2);
    l.samplingParams = r.Alloc<float>(5);
    l.headLogits = contiguous ? r.headLogits+size_t(index)*7*19072 : r.Alloc<float>(7 * 19072); l.corrected = r.Alloc<float>(7 * 19072);
    l.headControl = r.Alloc<unsigned long long>(2 * 148 + 2);
    l.softmax = r.Alloc<float>(7 * 38 * 2); l.confidence = contiguous ? r.headConfidence+size_t(index)*7 : r.Alloc<float>(7);
}
}
extern "C" bool FastllmDraftFusedPrepare(void *opaque, int requests) {
    if (!opaque || requests < 1 || requests > 64) return false;
    try {
        auto &s = *static_cast<State *>(opaque);
        if (requests <= s.prepared) return true;
        for (auto &r : s.ranks) {
            Check(cudaSetDevice(r.device));
            if(!r.contextCapacity) {
                r.contextCapacity=requests;
                r.contextInput=r.Alloc<B>(size_t(requests)*8*4096);
                r.contextSum=r.Alloc<B>(size_t(requests)*8*4096);
                r.contextPartial=r.Alloc<float>(size_t(requests)*kD12Splits*kTotalCols*kWindow);
            }
            while (int(r.lanes.size()) < requests) {
                const size_t index=r.lanes.size();const bool contiguous=index<size_t(r.contextCapacity);
                Lane l{};
                l.padded = r.Alloc<B>(8 * 4096);
                l.attention = r.Alloc<B>(8 * 4096); l.intermediate = r.Alloc<B>(8 * 512);
                l.qk = r.Alloc<B>(8 * 640); l.k = r.Alloc<B>(8 * 128);
                l.v = r.Alloc<B>(8 * 128); l.attn = r.Alloc<B>(8 * 512);
                l.split = r.Alloc<float>(kSplitScratch + kReadyWords);
                l.cosine = r.Alloc<float>(8 * 128); l.sine = r.Alloc<float>(8 * 128);
                l.positions = r.Alloc<int64_t>(8); l.epoch = r.Alloc<int>(1);
                l.barrier = r.Alloc<unsigned>(2); l.watchdog = r.Alloc<unsigned>(1);
                l.mailbox = r.Alloc<uint8_t>(kMailboxSlabsPerRank * naivert::kMailboxSlabStride);
                l.ctxInput = contiguous ? r.contextInput+index*8*4096 : r.Alloc<B>(8 * 4096);
                l.ctxSum = contiguous ? r.contextSum+index*8*4096 : r.Alloc<B>(8 * 4096);
                l.ctxValid = r.Alloc<B>(8); l.ctxSsq = r.Alloc<float>(8 * 128);
                l.ctxPartials = contiguous ? r.contextPartial+index*kD12Splits*kTotalCols*kWindow :
                    r.Alloc<float>(kD12Splits * kTotalCols * kWindow);
                l.ctxDest = r.Alloc<int64_t>(8);
                if (s.headReady) AllocateHeadLane(r, l, r.lanes.size());
                r.lanes.push_back(l);
            }
            Check(cudaDeviceSynchronize());
        }
        s.prepared = requests;
        if (s.contextReady) PrepareContextPlans(s, requests);
        Check(cudaSetDevice(s.ranks[0].device)); return true;
    } catch (const std::exception &e) { error = e.what(); return false; }
}
extern "C" void FastllmDraftFusedDestroy(void *state) { delete static_cast<State *>(state); }
extern "C" const char *FastllmDraftFusedError() { return error.c_str(); }
extern "C" bool FastllmDraftFusedRun(void *opaque, int rank, int requests, void *hidden,
        const int *live, const void *const *table, void *rawStream) {
    if (!opaque || rank < 0 || rank >= 8 || !hidden || !live || !table) return false;
    try {
        auto &s = *static_cast<State *>(opaque); auto &r = s.ranks[rank];
        if (requests < 1 || requests > int(r.lanes.size())) return false;
        cudaStream_t stream = (rawStream ? static_cast<cudaStream_t>(rawStream) : cudaStreamPerThread);
        RunBody(s,rank,requests,static_cast<B*>(hidden),live,table,nullptr,nullptr,stream);
        return true;
    } catch (const std::exception &e) { error = e.what(); return false; }
}
extern "C" bool FastllmDraftFusedRunNormalized(void *opaque,int rank,int requests,void *hidden,
        const int *live,const void *const *table,const float *gamma,void *normalized,void *rawStream) {
    if(!opaque || rank<0 || rank>=8 || !hidden || !live || !table || !gamma || !normalized ||
        hidden==normalized || ((reinterpret_cast<uintptr_t>(hidden)|reinterpret_cast<uintptr_t>(normalized)|
        reinterpret_cast<uintptr_t>(gamma))&15))return false;
    try {
        auto &s=*static_cast<State*>(opaque);auto &r=s.ranks[rank];
        if(requests<1 || requests>int(r.lanes.size()))return false;
        auto stream=rawStream?static_cast<cudaStream_t>(rawStream):cudaStreamPerThread;
        RunBody(s,rank,requests,static_cast<B*>(hidden),live,table,gamma,static_cast<B*>(normalized),stream);
        return true;
    }catch(const std::exception &e){error=e.what();return false;}
}
extern "C" bool FastllmDraftFusedMlpProbe(void *opaque, int rank, int request, int layer,
        const void *input, void *output, void *rawStream) {
    try {
        auto &s = *static_cast<State *>(opaque); cudaStream_t stream = (rawStream ? static_cast<cudaStream_t>(rawStream) : cudaStreamPerThread);
        auto &l = s.ranks[rank].lanes[request];
        Start(s, rank, request, nullptr, stream, static_cast<const B *>(input));
        Mlp(s, rank, request, layer, l.padded, l.attention, stream);
        Check(cudaMemcpyAsync(output, l.attention, s.rows * 4096 * sizeof(B), cudaMemcpyDeviceToDevice, stream));
        return true;
    } catch (const std::exception &e) { error = e.what(); return false; }
}
extern "C" bool FastllmDraftFusedMlpBatchProbe(void *opaque,int rank,int requests,int layer,
        const void *input,void *output,void *rawStream) {
    if(!opaque || rank<0 || rank>=8 || !input || !output) return false;
    try {
        auto &s=*static_cast<State*>(opaque);
        if(requests<1 || requests>int(s.ranks[rank].lanes.size()) || layer<0 || layer>=s.layers) return false;
        auto stream=rawStream?static_cast<cudaStream_t>(rawStream):cudaStreamPerThread;
        for(int first=0;first<requests;) {
            const int count=s.batchMlp4 && first+3<requests?4:s.batchMlp && first+1<requests?2:1;
            for(int request=first;request<first+count;++request)
                Start(s,rank,request,nullptr,stream,static_cast<const B*>(input)+size_t(request)*s.rows*4096);
            if(count==4) MlpQuad(s,rank,first,layer,true,stream);
            else if(count==2) MlpTile(s,rank,first,layer,true,stream);
            else {
                auto &l=s.ranks[rank].lanes[first];
                Mlp(s,rank,first,layer,l.padded,l.attention,stream);
            }
            for(int request=first;request<first+count;++request) {
                auto &l=s.ranks[rank].lanes[request];
                FinishResult<<<s.rows*4096/256,256,0,stream>>>(l.attention,
                    static_cast<B*>(output)+size_t(request)*s.rows*4096,s.rows*4096,l.watchdog);
            }
            first+=count;
        }
        Check(cudaGetLastError()); return true;
    }catch(const std::exception &e){error=e.what();return false;}
}
extern "C" bool FastllmDraftFusedAttentionProbe(void *opaque, int rank, int request, int layer,
        const void *input, void *output, const int *live, const void *const *table,
        int requests, void *rawStream) {
    try {
        auto &s = *static_cast<State *>(opaque); cudaStream_t stream = (rawStream ? static_cast<cudaStream_t>(rawStream) : cudaStreamPerThread);
        auto &l = s.ranks[rank].lanes[request];
        Start(s, rank, request, live, stream, static_cast<const B *>(input));
        Attention(s, rank, request, layer, l.padded, l.attention, live, table, requests, stream);
        Check(cudaMemcpyAsync(output, l.attention, s.rows * 4096 * sizeof(B), cudaMemcpyDeviceToDevice, stream));
        return true;
    } catch (const std::exception &e) { error = e.what(); return false; }
}
extern "C" bool FastllmDraftFusedAttentionBatchProbe(void *opaque,int rank,int requests,int layer,
        const void *input,void *output,const int *live,const void *const *table,void *rawStream) {
    if(!opaque || rank<0 || rank>=8 || !input || !output || !live || !table) return false;
    try {
        auto &s=*static_cast<State*>(opaque);
        if(requests<1 || requests>int(s.ranks[rank].lanes.size()) || layer<0 || layer>=s.layers) return false;
        auto stream=rawStream?static_cast<cudaStream_t>(rawStream):cudaStreamPerThread;
        for(int first=0;first<requests;) {
            const int count=s.batchAttention4 && first+3<requests?4:s.batchAttention && first+1<requests?2:1;
            for(int request=first;request<first+count;++request)
                Start(s,rank,request,live+request,stream,static_cast<const B*>(input)+size_t(request)*s.rows*4096);
            if(count==4) AttentionShared<4>(s,rank,first,layer,live,table,requests,stream);
            else if(count==2) AttentionTile(s,rank,first,layer,live,table,requests,stream);
            else {
                auto &l=s.ranks[rank].lanes[first];
                Attention(s,rank,first,layer,l.padded,l.attention,live+first,table,requests,stream);
            }
            for(int request=first;request<first+count;++request) {
                auto &l=s.ranks[rank].lanes[request];
                FinishResult<<<s.rows*4096/256,256,0,stream>>>(l.attention,
                    static_cast<B*>(output)+size_t(request)*s.rows*4096,s.rows*4096,l.watchdog);
            }
            first+=count;
        }
        Check(cudaGetLastError());return true;
    }catch(const std::exception &e){error=e.what();return false;}
}
extern "C" unsigned FastllmDraftFusedWatchdog(void *opaque) {
    auto &s = *static_cast<State *>(opaque); unsigned result = 0;
    for (auto &r : s.ranks) { cudaSetDevice(r.device);
        for (auto &l : r.lanes) { unsigned flag = 0; cudaMemcpy(&flag, l.watchdog, sizeof(flag), cudaMemcpyDeviceToHost); result |= flag; }
    }
    return result;
}
extern "C" bool FastllmDraftFusedContextPrepare(void *opaque, const void *fc, const void *norm, int owner) {
    if (!opaque || !fc || !norm) return false;
    try {
        auto &s = *static_cast<State *>(opaque);
        if (s.layers != 5 || s.window != 1024) return false;
        if (s.contextReady) return true;
        for (int rank = 0; rank < 8; ++rank) {
            auto &r = s.ranks[rank]; Check(cudaSetDevice(r.device));
            Check(cudaFuncSetAttribute(FusedDraftContext, cudaFuncAttributeMaxDynamicSharedMemorySize, ContextSmem));
            Check(cudaFuncSetAttribute(ContextTrim, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024 * 32 * 2));
            int perSm = 0, sms = 0;
            Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, FusedDraftContext, kThreads, ContextSmem));
            Check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, r.device));
            if (perSm * sms < kCtas) throw std::runtime_error("context grid is not resident");
            r.fc = r.Alloc<B>(4096 * 4096); r.hiddenNorm = r.Alloc<float>(4096);
            ShardFc<<<4096 * 4096 / 256, 256>>>(static_cast<const B *>(fc), r.fc, rank);
            Check(cudaMemcpyPeer(r.hiddenNorm, r.device, norm, owner, 4096 * sizeof(float)));
            Check(cudaDeviceSynchronize());
        }
        s.contextReady = true;
        if (s.prepared) PrepareContextPlans(s, s.prepared);
        Check(cudaSetDevice(owner)); return true;
    } catch (const std::exception &e) { error = e.what(); return false; }
}
namespace fastllm_draft_fused {
FcCtxKvArgs ContextArguments(State &s, int rank, int request) {
    auto &r = s.ranks[rank]; auto &l = r.lanes[request];
    FcCtxKvArgs a{};
    a.fc.target_hidden = l.ctxInput; a.fc.fc = r.fc; a.fc.hidden_norm = r.hiddenNorm;
    a.fc.ctx_valid = l.ctxValid; a.fc.ctx_sum = l.ctxSum;
    a.fc.ssq = l.ctxSsq; a.fc.barrier = l.barrier; a.fc.replay_generation = l.epoch;
    a.fc.generation_salt = kSaltD1; a.fc.rank = rank;
    a.fc.watchdog_limit_ns = 5000000000ull; a.fc.watchdog_flag = l.watchdog;
    a.fc.rms_epsilon = s.eps;
    for (int peer = 0; peer < 8; ++peer) a.fc.mailbox_peers[peer] = s.ranks[peer].lanes[request].mailbox;
    for (int i = 0; i < 5; ++i) {
        a.kv.k_w[i] = r.weights[i].rawK; a.kv.v_w[i] = r.weights[i].rawV;
        a.kv.k_norm[i] = r.weights[i].kNorm;
    }
    a.kv.ctx_pos = l.positions; a.kv.kv_dst = l.ctxDest;
    a.kv.rope_cos = l.cosine; a.kv.rope_sin = l.sine; a.kv.rms_epsilon = s.eps;
    a.kv_partials = l.ctxPartials;
    return a;
}
struct ContextBatchArgs {
    const PreparedContextRequest *requests;
    const BeginArgs *lanes;
    const unsigned *alias;
    const B *kv[5];
    void **table;
    const float *partials[8],*norm,*keyNorm[5];
    B *input,*normalized;
    float *work;
    int count,window;
    float eps,theta;
};
__global__ void ContextBatchAliases(const PreparedContextRequest *requests,int count,unsigned *alias) {
    const auto *d=requests+blockIdx.x*count;
    constexpr size_t bytes=1031*128*sizeof(B);
    bool overlap=false;
    for(int index=threadIdx.x;index<count*10;index+=blockDim.x) {
        const int request=index/10,slot=index%10;
        const void *x=d[request].cacheTable[(slot/2*d[request].tableRequests+d[request].tableRequest)*2+slot%2];
        for(int other=0;other<count;++other) {
            overlap=overlap || ContextOverlap(x,bytes,d[other].features,size_t(d[other].rows)*4096*sizeof(B));
            if(other<=request)continue;
            for(int j=0;j<10;++j) {
                const void *y=d[other].cacheTable[(j/2*d[other].tableRequests+d[other].tableRequest)*2+j%2];
                overlap=overlap || ContextOverlap(x,bytes,y,bytes);
            }
        }
    }
    if(__syncthreads_or(overlap) && !threadIdx.x)atomicOr(alias,1u);
}
__global__ void ContextBatchInput(ContextBatchArgs a) {
    const int request=blockIdx.x,tid=threadIdx.x;
    const auto &d=a.requests[request];const int rows=d.accepted?*d.accepted+1:d.rows;
    if(rows<1 || rows>d.rows){asm volatile("trap;");return;}
    const auto *src=reinterpret_cast<const uint4*>(d.features);
    auto *dst=reinterpret_cast<uint4*>(a.input+size_t(request)*8*4096);
    for(int i=tid;i<8*512;i+=blockDim.x)dst[i]=!(*a.alias) && i<rows*512?src[i]:make_uint4(0,0,0,0);
}
__global__ void ContextBatchTrim(ContextBatchArgs a) {
    if(*a.alias)return;
    const auto &d=a.requests[blockIdx.y];const int rows=d.accepted?*d.accepted+1:d.rows;
    const int keep=min(int(d.oldLength),a.window-1-rows),drop=d.oldLength-keep;
    if(keep && drop){extern __shared__ uint4 values[];ContextTrimBody(d.cacheTable,d.tableRequests,d.tableRequest,keep,drop,values);}
}
__global__ void ContextBatchJoin(ContextBatchArgs a) {
    if(*a.alias)return;
    const auto &b=a.lanes[blockIdx.x];
    if(!threadIdx.x) {
        // Use a fresh monotone generation even on the first replay. One join
        // publishes FC writes; the second retires all peer readers before the
        // same scratch becomes KV GEMM output.
        const unsigned generation=unsigned(*b.epoch)+1;
        rendezvous_announce(b.peers,b.rank,generation);
        rendezvous_wait(b.peers[b.rank],b.rank,generation,naivert::MailboxWatchdog{5000000000ull,b.watchdog});
        *b.epoch=int(generation);
    }
}
__global__ void ContextBatchNormalize(ContextBatchArgs a) {
    if(*a.alias)return;
    const int row=blockIdx.x,tid=threadIdx.x;
    __shared__ B values[4096];__shared__ float sums[256];
    float ssq=0;
    for(int col=tid*4;col<4096;col+=256*4) {
        float4 peers[8];
#pragma unroll
        for(int peer=0;peer<8;++peer)peers[peer]=*reinterpret_cast<const float4*>(a.partials[peer]+size_t(row)*4096+col);
        float4 sum=make_float4(0,0,0,0);
#pragma unroll
        for(int peer=0;peer<8;++peer) {
            sum.x=__fadd_rn(sum.x,peers[peer].x);sum.y=__fadd_rn(sum.y,peers[peer].y);
            sum.z=__fadd_rn(sum.z,peers[peer].z);sum.w=__fadd_rn(sum.w,peers[peer].w);
        }
        const float parts[4]={sum.x,sum.y,sum.z,sum.w};
#pragma unroll
        for(int i=0;i<4;++i){const B value=__float2bfloat16_rn(parts[i]);values[col+i]=value;ssq+=float(value)*float(value);}
    }
    sums[tid]=ssq;__syncthreads();
    for(int d=128;d;d/=2){if(tid<d)sums[tid]+=sums[tid+d];__syncthreads();}
    const float scale=rsqrtf(sums[0]/4096.f+a.eps);
    for(int col=tid;col<4096;col+=256)
        a.normalized[size_t(row)*4096+col]=__float2bfloat16_rn(float(__float2bfloat16_rn(float(values[col])*scale))*a.norm[col]);
}
__global__ void ContextBatchKVTable(ContextBatchArgs a) {
    const int i=threadIdx.x;
    if(i<5) {
        a.table[i]=const_cast<B*>(a.kv[i]);a.table[5+i]=a.normalized;
        a.table[10+i]=reinterpret_cast<B*>(a.work)+size_t(i)*a.count*8*256;
    }
}
__global__ void ContextBatchKVFinish(ContextBatchArgs a) {
    if(*a.alias)return;
    const int layer=blockIdx.x/(a.count*8),row=blockIdx.x%(a.count*8),request=row/8,local=row%8,tid=threadIdx.x;
    const auto &d=a.requests[request];const int rows=d.accepted?*d.accepted+1:d.rows;
    const int keep=min(int(d.oldLength),a.window-1-rows),destination=local<rows?keep+local:a.window+local-1;
    const B *raw=reinterpret_cast<const B*>(a.work)+(size_t(layer)*a.count*8+row)*256;
    __shared__ float partial[4];__shared__ B head[128];
    const float value=tid<128?float(raw[tid]):0.f;
    float square=value*value;
    for(int delta=16;delta;delta/=2)square+=__shfl_xor_sync(0xffffffffu,square,delta);
    if(tid<128 && tid%32==0)partial[tid/32]=square;
    if(tid>=128)static_cast<B*>(const_cast<void*>(d.cacheTable[(layer*d.tableRequests+d.tableRequest)*2+1]))[size_t(destination)*128+tid-128]=raw[tid];
    __syncthreads();
    const float scale=rsqrtf(((partial[0]+partial[1])+partial[2]+partial[3])/128.f+a.eps);
    if(tid<128)head[tid]=__float2bfloat16_rn(float(__float2bfloat16_rn(value*scale))*a.keyNorm[layer][tid]);
    __syncthreads();
    if(tid<128) {
        const float angle=float(d.position+local)*powf(a.theta,-2.f*(tid%64)/128.f);
        const float cosine=float(__float2bfloat16_rn(cosf(angle))),sine=float(__float2bfloat16_rn(sinf(angle)));
        const float rotated=tid<64?-float(head[tid+64]):float(head[tid-64]);
        const B result=__float2bfloat16_rn(__fadd_rn(float(__float2bfloat16_rn(__fmul_rn(float(head[tid]),cosine))),float(__float2bfloat16_rn(__fmul_rn(rotated,sine)))));
        static_cast<B*>(const_cast<void*>(d.cacheTable[(layer*d.tableRequests+d.tableRequest)*2]))[size_t(destination)*128+tid]=result;
    }
}
// The batched body has its own communication domain. Reuse context scratch at
// this disjoint phase boundary instead of activating an extra NCCL work arena.
struct DraftReduceArgs {
    BeginArgs join;
    float *peers[8],*data;
    int elements,rank;
};
__global__ void DraftReduceJoin(DraftReduceArgs a) {
    if(threadIdx.x)return;
    const auto &b=a.join;const unsigned generation=unsigned(*b.epoch)+1;
    rendezvous_announce(b.peers,b.rank,generation);
    rendezvous_wait(b.peers[b.rank],b.rank,generation,naivert::MailboxWatchdog{5000000000ull,b.watchdog});
    *b.epoch=int(generation);
    if(*b.watchdog)asm volatile("trap;");
}
__global__ void DraftReduceScatter(DraftReduceArgs a) {
    const int chunk=a.elements/8;
    const int col=(blockIdx.x*blockDim.x+threadIdx.x)*4+a.rank*chunk;
    if(col>=(a.rank+1)*chunk)return;
    float4 peers[8];
#pragma unroll
    for(int peer=0;peer<8;++peer)peers[peer]=*reinterpret_cast<const float4*>(a.peers[peer]+col);
    float4 sum=make_float4(0,0,0,0);
#pragma unroll
    for(int peer=0;peer<8;++peer){sum.x=__fadd_rn(sum.x,peers[peer].x);sum.y=__fadd_rn(sum.y,peers[peer].y);sum.z=__fadd_rn(sum.z,peers[peer].z);sum.w=__fadd_rn(sum.w,peers[peer].w);}
    *reinterpret_cast<float4*>(a.peers[a.rank]+col)=sum;
}
__global__ void DraftReduceGather(DraftReduceArgs a) {
    const int col=(blockIdx.x*blockDim.x+threadIdx.x)*4;
    if(col<a.elements)*reinterpret_cast<float4*>(a.data+col)=
        *reinterpret_cast<const float4*>(a.peers[col/(a.elements/8)]+col);
}
void AddBlasNodes(cudaGraph_t graph,cudaGraphNode_t &previous,std::vector<cudaGraphNode_t> *prefix,
    const std::function<void(cudaStream_t)> &launch);
void SetDraftBlasStream(Rank &r,cublasHandle_t blas,cudaStream_t stream);
void PrepareWideContext(State &s,int rank) {
    auto &r=s.ranks[rank];if(r.contextBatch)return;
    std::vector<BeginArgs> lanes(r.contextCapacity);
    for(int i=0;i<r.contextCapacity;++i) {
        auto &a=lanes[i];auto &l=r.lanes[i];a.rank=rank;a.epoch=l.epoch;a.watchdog=l.watchdog;
        for(int peer=0;peer<8;++peer)a.peers[peer]=s.ranks[peer].lanes[i].mailbox;
    }
    r.contextBatch=r.Alloc<BeginArgs>(lanes.size());
    Check(cudaMemcpy(r.contextBatch,lanes.data(),lanes.size()*sizeof(BeginArgs),cudaMemcpyHostToDevice));
}
void AddWideContext(State &s,int rank,int count,cudaGraph_t graph,cudaGraphNode_t &previous,cublasHandle_t blas) {
    auto &r=s.ranks[rank];ContextBatchArgs a{};
    a.requests=s.contextInputs->gpu+rank*count;a.lanes=r.contextBatch;a.alias=s.contextInputs->alias;
    a.norm=r.hiddenNorm;a.input=r.contextInput;a.normalized=r.contextSum;a.work=r.contextPartial;
    a.count=count;a.window=s.window;a.eps=s.eps;a.theta=s.theta;
    for(int peer=0;peer<8;++peer)a.partials[peer]=s.ranks[peer].contextPartial;
    for(int layer=0;layer<5;++layer){a.keyNorm[layer]=r.weights[layer].kNorm;a.kv[layer]=r.weights[layer].rawK;}
    a.table=reinterpret_cast<void**>(r.contextPartial+size_t(r.contextCapacity)*8*4096);
    auto add=[&](void *function,dim3 blocks,int threads,int smem) {
        void *args[]={&a};cudaKernelNodeParams params{};
        params.func=function;params.gridDim=blocks;params.blockDim=dim3(threads);params.sharedMemBytes=smem;params.kernelParams=args;
        cudaGraphNode_t node;Check(cudaGraphAddKernelNode(&node,graph,previous?&previous:nullptr,previous?1:0,&params));previous=node;
    };
    add((void*)ContextBatchInput,dim3(count),256,0);
    add((void*)ContextBatchTrim,dim3(40,count),256,1024*32*2);
    AddBlasNodes(graph,previous,nullptr,[&](cudaStream_t stream) {
        const float alpha=1,beta=0;SetDraftBlasStream(r,blas,stream);
        if(cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,4096,count*8,4096,&alpha,
            r.fc,CUDA_R_16BF,4096,r.contextInput,CUDA_R_16BF,4096,&beta,r.contextPartial,CUDA_R_32F,4096,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP)!=CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("batched context FC projection failed");
    });
    add((void*)ContextBatchJoin,dim3(count),32,0);
    add((void*)ContextBatchNormalize,dim3(count*8),256,0);
    add((void*)ContextBatchJoin,dim3(count),32,0);
    AddBlasNodes(graph,previous,nullptr,[&](cudaStream_t stream) {
        const float alpha=1,beta=0;
        SetDraftBlasStream(r,blas,stream);
        ContextBatchKVTable<<<1,32,0,stream>>>(a);Check(cudaGetLastError());
        if(cublasGemmBatchedEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,256,count*8,4096,&alpha,
            (const void *const*)a.table,CUDA_R_16BF,4096,(const void *const*)(a.table+5),CUDA_R_16BF,4096,&beta,
            (void *const*)(a.table+10),CUDA_R_16BF,256,5,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP)!=CUBLAS_STATUS_SUCCESS)
                throw std::runtime_error("batched context KV projection failed");
    });
    add((void*)ContextBatchKVFinish,dim3(5*count*8),256,0);
}

void PrepareContextPlans(State &s, int requests) {
    // All addresses are per-request lanes. Only this bounded descriptor array
    // changes between launches; graphs never retain caller-owned host memory.
    try {
        if(!s.contextInputs) {
            s.contextInputs.reset(new ContextInputs(s.ranks[0].device));
            auto &p=*s.contextInputs;Check(cudaSetDevice(p.device));
            Check(cudaMallocHost(&p.host,8*64*sizeof(*p.host)));
            Check(cudaMalloc(&p.gpu,8*64*sizeof(*p.gpu)));
            Check(cudaMalloc(&p.alias,sizeof(unsigned)));
            Check(cudaStreamCreateWithFlags(&p.stream,cudaStreamNonBlocking));
            Check(cudaEventCreateWithFlags(&p.ready,cudaEventDisableTiming));
            Check(cudaEventCreateWithFlags(&p.acceptedReady,cudaEventDisableTiming));
        }
        for (int rank=0;rank<8;++rank) {
            Check(cudaSetDevice(s.ranks[rank].device));
            s.batchContext4=ConfigureOptionalKernel(s.ranks[rank].device,FusedDraftContextRequests<4>,
                kThreads,ContextSmem,kCtas) && s.batchContext4;
            s.batchContext=ConfigureOptionalKernel(s.ranks[rank].device,FusedDraftContextTile,
                kThreads,ContextSmem,kCtas) && s.batchContext;
        }
        for (int rank = 0; rank < 8; ++rank) {
            auto &r = s.ranks[rank]; Check(cudaSetDevice(r.device));
            PrepareWideContext(s,rank);
            Check(cudaFuncSetAttribute(ContextBatchTrim,cudaFuncAttributeMaxDynamicSharedMemorySize,1024*32*2));
            cudaFuncAttributes attrs{};Check(cudaFuncGetAttributes(&attrs,ContextBatchAliases));
            Check(cudaFuncSetAttribute(FusedDraftContextPrepared,cudaFuncAttributeMaxDynamicSharedMemorySize,ContextSmem));
            int active = 0, sms = 0;
            Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,FusedDraftContextPrepared,kThreads,ContextSmem));
            Check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,r.device));
            if (active*sms < kCtas) throw std::runtime_error("context graph requires a resident grid");
            if (!r.contextPlans) {
                r.contextPlans.reset(new ContextPlans(r.device));
                r.contextPlans->execs.resize(1,nullptr);
            }
            auto &p = *r.contextPlans;
            // Only graph construction needs a cuBLAS handle. Every captured
            // matrix and workspace belongs to State; release the handle's
            // otherwise idle internal arena before serving graph replays.
            std::unique_ptr<Blas> blas;
            while (int(p.execs.size()) <= requests) {
                const int count = p.execs.size();
                cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr;
                try {
                    Check(cudaGraphCreate(&graph,0));
                    cudaGraphNode_t previous = nullptr;
                    auto add = [&](void *function, int blocks, int threads, int smem, void **args) {
                        cudaKernelNodeParams params{};
                        params.func = function; params.gridDim = dim3(blocks); params.blockDim = dim3(threads);
                        params.sharedMemBytes = smem; params.kernelParams = args;
                        cudaGraphNode_t node = nullptr;
                        Check(cudaGraphAddKernelNode(&node,graph,previous?&previous:nullptr,previous?1:0,&params));
                        previous = node;
                    };
                    const unsigned *fallback=nullptr;
                    bool wide=count>=8;
                    for(const auto &peer:s.ranks)wide=wide && count<=peer.contextCapacity;
                    if(wide) {
                        if(!blas)blas.reset(new Blas(r.device));
                        AddWideContext(s,rank,count,graph,previous,blas->handle);
                        fallback=s.contextInputs->alias;
                    }
                    for (int first=0;first<count;) {
                        const int requests=s.batchContext4 && first+3<count ? 4 : s.batchContext && first+1<count ? 2 : 1;
                        auto *d=s.contextInputs->gpu+rank*count+first;
                        if(requests==4) {
                            ContextRequests<4> tile{};
                            for(int i=0;i<4;++i)tile.lanes[i]=ContextArguments(s,rank,first+i);
                            void *body[]={&tile,&d,&s.window,&s.theta,&fallback};
                            add((void*)FusedDraftContextRequests<4>,kCtas,kThreads,ContextSmem,body);
                        } else if(requests==2) {
                            ContextTileArgs tile{{ContextArguments(s,rank,first),ContextArguments(s,rank,first+1)}};
                            void *body[]={&tile,&d,&s.window,&s.theta,&fallback};
                            add((void*)FusedDraftContextTile,kCtas,kThreads,ContextSmem,body);
                        } else {
                            auto a=ContextArguments(s,rank,first);
                            void *body[]={&a,&d,&s.window,&s.theta,&fallback};
                            add((void*)FusedDraftContextPrepared,kCtas,kThreads,ContextSmem,body);
                        }
                        for(int request=first;request<first+requests;++request) {
                            auto &l=r.lanes[request];B *none=nullptr;int zero=0;
                            void *finish[]={&none,&none,&zero,&l.watchdog};add((void*)FinishResult,1,32,0,finish);
                        }
                        first+=requests;
                    }
                    Check(cudaGraphInstantiateWithFlags(&exec,graph,0));
                    Check(cudaGraphUpload(exec,cudaStreamPerThread));
                    Check(cudaGraphDestroy(graph)); graph = nullptr;
                    p.execs.push_back(exec); exec = nullptr;
                } catch (...) {
                    if (exec) cudaGraphExecDestroy(exec);
                    if (graph) cudaGraphDestroy(graph);
                    throw;
                }
            }
            Check(cudaDeviceSynchronize());
        }
    } catch (const std::exception &e) {
        error = e.what();
        // Preparation has not run a collective. Disable plans consistently on
        // every rank and retain the already-tested eager context implementation.
        for (auto &r : s.ranks) r.contextPlans.reset();
        s.contextInputs.reset();
        cudaGetLastError();
    }
}
}
extern "C" bool FastllmDraftFusedReduceSupported(void *opaque,int requests) {
    if(!opaque || requests<1 || requests>64)return false;
    const auto &s=*static_cast<State*>(opaque);
    for(const auto &peer:s.ranks)if(requests>peer.contextCapacity)return false;
    return true;
}
extern "C" bool FastllmDraftFusedReduce(void *opaque,int rank,int requests,void *data,int elements,void *rawStream) {
    if(!FastllmDraftFusedReduceSupported(opaque,requests) || rank<0 || rank>=8 || !data ||
        (reinterpret_cast<uintptr_t>(data)&15))return false;
    auto &s=*static_cast<State*>(opaque);
    if(elements!=requests*s.rows*4096)return false;
    try {
        auto &r=s.ranks[rank];auto &lane=r.lanes.front();DraftReduceArgs a{};
        a.rank=a.join.rank=rank;a.join.epoch=lane.epoch;a.join.watchdog=lane.watchdog;
        a.data=static_cast<float*>(data);a.elements=elements;
        for(int peer=0;peer<8;++peer){a.peers[peer]=s.ranks[peer].contextPartial;a.join.peers[peer]=s.ranks[peer].lanes.front().mailbox;}
        auto stream=rawStream?static_cast<cudaStream_t>(rawStream):cudaStreamPerThread;
        Check(cudaMemcpyAsync(a.peers[rank],data,size_t(elements)*sizeof(float),cudaMemcpyDeviceToDevice,stream));
        DraftReduceJoin<<<1,32,0,stream>>>(a);
        DraftReduceScatter<<<(elements/8+1023)/1024,256,0,stream>>>(a);
        DraftReduceJoin<<<1,32,0,stream>>>(a);
        DraftReduceGather<<<(elements+1023)/1024,256,0,stream>>>(a);
        // Retire all peer reads before a following body/head/context operation
        // reuses the same scratch. No timing assumption replaces this join.
        DraftReduceJoin<<<1,32,0,stream>>>(a);
        Check(cudaGetLastError());return true;
    }catch(const std::exception &e){error=e.what();return false;}
}
extern "C" bool FastllmDraftFusedContextSharedSupported(void *opaque,int count) {
    if (!opaque || count < 1 || count > 64) return false;
    auto &s = *static_cast<State *>(opaque);
    if (!s.contextInputs) return false;
    for (const auto &r : s.ranks)
        if (!r.contextPlans || count >= int(r.contextPlans->execs.size()) || !r.contextPlans->execs[count]) return false;
    return true;
}
namespace {
bool UploadContextDescriptors(void *opaque,const FastllmDraftContextRequest *requests,
        int count,const int *accepted,void *producerStream) {
    if(!requests || !FastllmDraftFusedContextSharedSupported(opaque,count))return false;
    for(int i=0;i<8*count;++i) {
        const auto &d=requests[i];
        if(!d.features || !d.cacheTable || d.rows<1 || d.rows>8 || d.position<0 ||
            d.oldLength<0 || d.oldLength>1023 || d.tableRequests<1 || d.tableRequest<0 || d.tableRequest>=d.tableRequests)return false;
    }
    try {
        auto &p=*static_cast<State*>(opaque)->contextInputs;Check(cudaSetDevice(p.device));
        if(p.inFlight)Check(cudaEventSynchronize(p.ready));
        for(int i=0;i<8*count;++i) {
            const auto &d=requests[i];
            p.host[i]={d.features,d.cacheTable,accepted?accepted+i%count:nullptr,d.position,
                uint16_t(d.rows),uint16_t(d.oldLength),d.tableRequests,d.tableRequest};
        }
        Check(cudaMemcpyAsync(p.gpu,p.host,8*count*sizeof(*p.host),cudaMemcpyHostToDevice,p.stream));
        if(count>=8) {
            Check(cudaMemsetAsync(p.alias,0,sizeof(unsigned),p.stream));
            ContextBatchAliases<<<8,256,0,p.stream>>>(p.gpu,count,p.alias);
            Check(cudaGetLastError());
        }
        // The explicit source stream contains a real copy before publication;
        // peer waits never inherit a newly created per-thread stream dependency.
        Check(cudaEventRecord(p.ready,p.stream));p.inFlight=true;
        p.hasAccepted=accepted!=nullptr;
        if(accepted) {
            auto producer=producerStream?static_cast<cudaStream_t>(producerStream):cudaStreamPerThread;
            Check(cudaEventRecord(p.acceptedReady,producer));
        }
        return true;
    }catch(const std::exception &e){error=e.what();return false;}
}
}
extern "C" bool FastllmDraftFusedContextSharedUpload(void *opaque,
        const FastllmDraftContextRequest *requests,int count) {
    return UploadContextDescriptors(opaque,requests,count,nullptr,nullptr);
}
extern "C" bool FastllmDraftFusedContextSharedUploadAccepted(void *opaque,
        const FastllmDraftContextRequest *requests,int count,const int *accepted,void *producerStream) {
    return accepted && UploadContextDescriptors(opaque,requests,count,accepted,producerStream);
}
extern "C" bool FastllmDraftFusedContextSharedRun(void *opaque,int rank,int count,void *rawStream) {
    if(rank<0 || rank>=8 || !FastllmDraftFusedContextSharedSupported(opaque,count))return false;
    try {
        auto &s=*static_cast<State*>(opaque);
        auto stream=rawStream?static_cast<cudaStream_t>(rawStream):cudaStreamPerThread;
        Check(cudaStreamWaitEvent(stream,s.contextInputs->ready,0));
        // The descriptor copy and accepted counts have independent producers.
        // Join both directly; no patch kernel or copy-stream relay is required.
        if(s.contextInputs->hasAccepted)
            Check(cudaStreamWaitEvent(stream,s.contextInputs->acceptedReady,0));
        Check(cudaGraphLaunch(s.ranks[rank].contextPlans->execs[count],stream));return true;
    }catch(const std::exception &e){error=e.what();return false;}
}
extern "C" bool FastllmDraftFusedContextRun(void *opaque, int rank, int request, const void *features,
        int rows, int position, int oldLength, const void *const *table,
        int tableRequests, int tableRequest, void *rawStream) {
    if (!opaque || rank < 0 || rank >= 8 || !features || !table || rows < 1 || rows > 8 ||
        position < 0 || oldLength < 0 || oldLength > 1023 || tableRequest < 0 || tableRequest >= tableRequests) return false;
    try {
        auto &s = *static_cast<State *>(opaque); auto &r = s.ranks[rank];
        if (!s.contextReady || request < 0 || request >= int(r.lanes.size())) return false;
        auto &l = r.lanes[request];
        cudaStream_t stream = rawStream ? static_cast<cudaStream_t>(rawStream) : cudaStreamPerThread;
        int keep = std::min(oldLength, s.window - 1 - rows);
        ContextInput<<<32,256,0,stream>>>(static_cast<const B *>(features), l.ctxInput, l.ctxValid, rows,
            position, keep, s.window, l.positions, l.ctxDest, l.cosine, l.sine, s.theta);
        if (keep && oldLength > keep)
            ContextTrim<<<40,256,1024*32*2,stream>>>(table,tableRequests,tableRequest,keep,oldLength-keep);
        Check(cudaGetLastError());
        auto a = ContextArguments(s,rank,request);
        a.kv.cache_table = table; a.kv.table_requests = tableRequests; a.kv.table_request = tableRequest;
        FusedDraftContext<<<kCtas,kThreads,ContextSmem,stream>>>(a);
        FinishResult<<<1,32,0,stream>>>(nullptr,nullptr,0,l.watchdog);
        Check(cudaGetLastError()); return true;
    } catch (const std::exception &e) { error = e.what(); return false; }
}

namespace fastllm_draft_fused {
constexpr int HeadStride = 146;
constexpr int HeadSmem = kHeadSmem + 7 * HeadStride * sizeof(float);
constexpr int HeadQuadSmem = kHeadSmem + 4 * 7 * HeadStride * sizeof(float);
static_assert(HeadQuadSmem <= 232448);
constexpr int HeadTileSmem = kHeadSmem + 2 * 7 * HeadStride * sizeof(float);
static_assert(HeadTileSmem <= 232448, "paired head must fit one resident CTA per SM");
__global__ void PackHead(const B *source, B *out) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= size_t(19072) * 4096) return;
    int row = i / 4096, col = i % 4096;
    size_t dst = (((size_t(row / 16) * 4 + col / 1024) * 32 + (col % 1024) / 32) * 16 + row % 16) * 32 + col % 32;
    out[dst] = source[i];
}
__global__ __launch_bounds__(128) void PrecomputeDraftBias(const B *w1, const B *w2, B *out) {
    __shared__ B queries[8 * 256];
    __shared__ float partials[4 * 128];
    const int first = int(blockIdx.y) * 8;
    for (int i = threadIdx.x; i < 8 * 256; i += 128) queries[i] = w1[size_t(first) * 256 + i];
    __syncthreads();
    naivert::attn_rows::project_strided_rows_mma<8,256,1,4,4>(0,19072/16,
        [&](int n) { return w2 + size_t(n) * 256; }, queries, partials,
        [&](int n,int row,float value) { out[size_t(first+row)*19072+n] = __float2bfloat16_rn(value); });
}
__global__ __launch_bounds__(kHeadThreads,1) void FusedDraftHeadProposal(
        const __grid_constant__ HeadArgs h, const __grid_constant__ MarkovArgs m,
        unsigned long long *control,int steps) {
    extern __shared__ __align__(128) unsigned char memory[];
    HeadBody::run<true,HeadStride>(memory,h,nullptr,steps);
    markov_cooperative_steps<160,false,HeadStride>(m,
        reinterpret_cast<const float*>(memory+kHeadSmem),control,control+2*kHeadCtas,steps);
}
struct HeadTileArgs {
    HeadArgs head[2];
    MarkovArgs markov[2];
    unsigned long long *control[2];
    int steps;
};
__global__ __launch_bounds__(kHeadThreads,1) void FusedDraftHeadTile(
        const __grid_constant__ HeadTileArgs tile) {
    extern __shared__ __align__(128) unsigned char memory[];
    HeadBody::run<true,HeadStride,2>(memory,tile.head[0],tile.head+1,tile.steps);
    // Independent request groups advance their own Markov chain concurrently.
    // The preceding head CTA join retired all projection producers. Each group
    // retains its request's RNG, mailbox, epoch and candidate buffers.
    const int request=threadIdx.x/192;
    auto *control=tile.control[request];
    markov_cooperative_steps<192,false,HeadStride,2>(tile.markov[request],
        reinterpret_cast<const float*>(memory+kHeadSmem)+request*7*HeadStride,
        control,control+2*kHeadCtas,tile.steps);
}
struct HeadQuadArgs {
    HeadArgs head[4];
    MarkovArgs markov[4];
    unsigned long long *control[4];
    int steps;
};
__global__ __launch_bounds__(kHeadThreads,1) void FusedDraftHeadQuad(
        const __grid_constant__ HeadQuadArgs tile) {
    extern __shared__ __align__(128) unsigned char memory[];
    HeadBody::run<true,HeadStride,4>(memory,tile.head[0],tile.head+1,tile.steps);
    // Independent request groups advance their own Markov chain concurrently.
    // The preceding head CTA join retired all projection producers. Each group
    // retains its request's RNG, mailbox, epoch and candidate buffers.
    const int request=threadIdx.x/96;
    auto *control=tile.control[request];
    markov_cooperative_steps<96,false,HeadStride,4>(tile.markov[request],
        reinterpret_cast<const float*>(memory+kHeadSmem)+request*7*HeadStride,
        control,control+2*kHeadCtas,tile.steps);
}
// Wide batches share one dense projection. Parallel vocabulary tiles feed an
// independent reduction per Markov chain; lane seeds and BF16 boundaries remain.
struct HeadBatchLane { BeginArgs begin; MarkovArgs markov; float *logits; unsigned long long *control; };
void PrepareHeadBatch(State &s,int rank) {
    auto &r=s.ranks[rank];Check(cudaSetDevice(r.device));
    std::vector<HeadBatchLane> lanes(r.headCapacity);
    for(int i=0;i<r.headCapacity;++i) {
        auto &l=r.lanes[i];auto &a=lanes[i];
        auto &b=a.begin;b.rank=rank;b.epoch=l.epoch;b.watchdog=l.watchdog;
        b.headIds=l.headIds;b.sampling=l.sampling;b.samplingParams=l.samplingParams;
        auto &m=a.markov;m.block_ids=l.headIds;m.draft_tokens=l.headIds+1;m.prev_token=l.headIds+7;
        m.logit_trace=l.corrected;m.bias_table=r.biasTable;m.replay_generation=l.epoch;
        m.rank=rank;m.apply_markov=1;m.watchdog_limit_ns=5000000000ull;m.watchdog_flag=l.watchdog;
        m.sampling_state=l.sampling;m.sampling_params=l.samplingParams;
        for(int peer=0;peer<8;++peer) b.peers[peer]=m.mailbox_peers[peer]=s.ranks[peer].lanes[i].mailbox;
        a.logits=l.headLogits;a.control=l.headControl;
    }
    r.headBatch=r.Alloc<HeadBatchLane>(lanes.size());
    Check(cudaMemcpy(r.headBatch,lanes.data(),lanes.size()*sizeof(HeadBatchLane),cudaMemcpyHostToDevice));
}
__global__ void BeginHeadBatch(const HeadBatchLane *lanes,const FastllmDraftHeadRequest *requests) {
    auto b=lanes[blockIdx.x].begin;const auto &d=requests[blockIdx.x];
    b.anchor=d.anchor;b.seed=d.seed;b.inverseTemperature=__fdiv_rn(1.f,d.temperature);
    BeginBody(b);
}
constexpr int HeadMarkovTiles=16;
__global__ __launch_bounds__(256) void DraftHeadMarkovScores(const HeadBatchLane *lanes,int step) {
    const auto &d=lanes[blockIdx.y];const auto &a=d.markov;
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32,prev=int(a.block_ids[step]);
    __shared__ float values[8];__shared__ int indices[8];
    float best=-INFINITY;int index=INT_MAX;
    for(int n=blockIdx.x*256+tid;n<19072;n+=256*HeadMarkovTiles) {
        const float base=float(__float2bfloat16_rn(d.logits[step*19072+n]));
        d.logits[step*19072+n]=base;
        const float value=float(__float2bfloat16_rn(base+float(a.bias_table[size_t(prev)*19072+n])));
        a.logit_trace[step*19072+n]=value;
        const float score=naivert_dspark::tempered(value,a.sampling_params[4])+
            sampling_gumbel(a.sampling_state,step,a.rank*19072+n);
        markov_take(best,index,score,n);
    }
    for(int off=16;off;off>>=1){const float v=__shfl_xor_sync(0xffffffffu,best,off);const int i=__shfl_xor_sync(0xffffffffu,index,off);markov_take(best,index,v,i);}
    if(!lane){values[warp]=best;indices[warp]=index;}
    __syncthreads();
    if(!warp) {
        float v=lane<8?values[lane]:-INFINITY;int i=lane<8?indices[lane]:INT_MAX;
        for(int off=4;off;off>>=1){const float ov=__shfl_xor_sync(0xffffffffu,v,off);const int oi=__shfl_xor_sync(0xffffffffu,i,off);markov_take(v,i,ov,oi);}
        if(!lane)reinterpret_cast<uint2*>(d.control)[blockIdx.x]=make_uint2(__float_as_uint(v),unsigned(i));
    }
}
__global__ void DraftHeadMarkovPick(const HeadBatchLane *lanes,int step) {
    const auto &d=lanes[blockIdx.x];const auto &a=d.markov;const int lane=threadIdx.x;
    const auto p=lane<HeadMarkovTiles?reinterpret_cast<const uint2*>(d.control)[lane]:make_uint2(__float_as_uint(-INFINITY),INT_MAX);
    float v=__uint_as_float(p.x);int i=int(p.y);
    for(int off=16;off;off>>=1){const float ov=__shfl_xor_sync(0xffffffffu,v,off);const int oi=__shfl_xor_sync(0xffffffffu,i,off);markov_take(v,i,ov,oi);}
    const unsigned gen=naivert::mailbox_generation(unsigned(*a.replay_generation),kSaltD6Cand+unsigned(step));
    const auto mailbox=naivert::TileMailbox<MarkovLayout>::make(a.mailbox_peers,a.rank,gen,kMkSlabCand*naivert::kMailboxSlabStride);
    const int gi=i+a.rank*19072;
    if(!lane)mailbox.publish_pair(0,0,__float_as_uint(v),unsigned(gi));
    const int pick=markov_pick<false>(a,mailbox,nullptr,gen,v,gi,step);
    if(!lane){a.draft_tokens[step]=pick;a.block_ids[step+1]=pick;a.prev_token[0]=pick;}
}

__global__ void VerificationInputBatch(const HeadBatchLane *lanes,const FastllmDraftHeadRequest *requests,
        float *input,int stride,int rows) {
    const int request=blockIdx.x,row=threadIdx.x;
    if(!row && *lanes[request].begin.watchdog)asm volatile("trap;");
    if(row<rows) {
        const int offset=request*rows+row,position=requests[request].position+row;
        input[offset]=float(lanes[request].markov.block_ids[row]);
        input[stride+offset]=float(position);reinterpret_cast<int*>(input+2*stride)[offset]=position+1;
    }
}
void SetDraftBlasStream(Rank &r,cublasHandle_t blas,cudaStream_t stream) {
    // FC and head/KV projection run at disjoint phase boundaries. The tail of
    // the existing context partial allocation is not a matrix output; lend it
    // to cuBLAS instead of allocating a persistent auxiliary workspace.
    const size_t first=size_t(r.contextCapacity)*8*4096*sizeof(float);
    const size_t total=size_t(r.contextCapacity)*kD12Splits*kTotalCols*kWindow*sizeof(float);
    if(cublasSetStream(blas,stream)!=CUBLAS_STATUS_SUCCESS ||
       cublasSetWorkspace(blas,reinterpret_cast<char*>(r.contextPartial)+first+256,total-first-256)!=CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("draft BLAS workspace binding failed");
}
void HeadProjection(Rank &r,cublasHandle_t blas,const void *hidden,int count,cudaStream_t stream) {
    const float alpha=1.f,beta=0.f;
    SetDraftBlasStream(r,blas,stream);
    if(cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,19072,count*7,4096,
            &alpha,r.rawHead,CUDA_R_16BF,4096,hidden,CUDA_R_16BF,4096,&beta,
            r.headLogits,CUDA_R_32F,19072,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP)!=CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("batched draft head projection failed");
}
void AddBlasNodes(cudaGraph_t graph,cudaGraphNode_t &previous,std::vector<cudaGraphNode_t> *prefix,
        const std::function<void(cudaStream_t)> &launch) {
    cudaStream_t stream=nullptr;cudaGraph_t captured=nullptr;
    try {
        Check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        launch(stream);Check(cudaStreamSynchronize(stream));
        Check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        launch(stream);
        Check(cudaStreamEndCapture(stream,&captured));
        size_t size=0;Check(cudaGraphGetNodes(captured,nullptr,&size));
        std::vector<cudaGraphNode_t> pending(size),done;
        Check(cudaGraphGetNodes(captured,pending.data(),&size));
        // cuBLAS may split K. Preserve its dependency order and keep every
        // prefix node independently disableable for ordinary verification.
        while(!pending.empty()) {
            bool progress=false;
            for(auto it=pending.begin();it!=pending.end();) {
                size_t n=0;Check(cudaGraphNodeGetDependencies(*it,nullptr,&n));
                std::vector<cudaGraphNode_t> deps(n);Check(cudaGraphNodeGetDependencies(*it,deps.data(),&n));
                bool ready=true;for(auto dep:deps)if(std::find(done.begin(),done.end(),dep)==done.end())ready=false;
                if(!ready){++it;continue;}
                cudaGraphNodeType type;Check(cudaGraphNodeGetType(*it,&type));cudaGraphNode_t node;
                if(type==cudaGraphNodeTypeMemset) {
                    cudaMemsetParams params{};Check(cudaGraphMemsetNodeGetParams(*it,&params));
                    Check(cudaGraphAddMemsetNode(&node,graph,previous?&previous:nullptr,previous?1:0,&params));
                } else if(type==cudaGraphNodeTypeMemcpy) {
                    cudaMemcpy3DParms params{};Check(cudaGraphMemcpyNodeGetParams(*it,&params));
                    Check(cudaGraphAddMemcpyNode(&node,graph,previous?&previous:nullptr,previous?1:0,&params));
                } else if(type==cudaGraphNodeTypeEmpty) {
                    Check(cudaGraphAddEmptyNode(&node,graph,previous?&previous:nullptr,previous?1:0));
                } else if(type==cudaGraphNodeTypeKernel) {
                    CUDA_KERNEL_NODE_PARAMS params{};
                    CUgraphNode added=nullptr,dependency=reinterpret_cast<CUgraphNode>(previous);
                    // cuBLAS registers driver functions, not host runtime stubs.
                    if(cuGraphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(*it),&params)!=CUDA_SUCCESS ||
                       cuGraphAddKernelNode(&added,reinterpret_cast<CUgraph>(graph),previous?&dependency:nullptr,previous?1:0,&params)!=CUDA_SUCCESS)
                        throw std::runtime_error("cannot compose draft BLAS graph");
                    node=reinterpret_cast<cudaGraphNode_t>(added);
                    Check(cudaGraphKernelNodeCopyAttributes(node,*it));
                } else throw std::runtime_error("unsupported draft BLAS graph node type "+std::to_string(int(type)));
                previous=node;
                if(prefix && type!=cudaGraphNodeTypeEmpty)prefix->push_back(node);
                done.push_back(*it);it=pending.erase(it);progress=true;
            }
            if(!progress)throw std::runtime_error("invalid draft BLAS graph dependencies");
        }
        Check(cudaGraphDestroy(captured));captured=nullptr;Check(cudaStreamDestroy(stream));stream=nullptr;
    } catch(...) {
        if(stream){cudaStreamCaptureStatus status;cudaStreamIsCapturing(stream,&status);if(status!=cudaStreamCaptureStatusNone){cudaGraph_t g=nullptr;cudaStreamEndCapture(stream,&g);if(g)cudaGraphDestroy(g);}}
        if(captured)cudaGraphDestroy(captured);if(stream)cudaStreamDestroy(stream);throw;
    }
}

struct ProbabilityArgs { const float *logits[8]; const unsigned *watchdogs[8]; };
struct ProbabilityRequest {
    ProbabilityArgs peers;
    float *out;float2 *partial;
    const int64_t *ids;
    float inverseTemperature;
    int steps;
};
struct ProbabilityBatch { ProbabilityRequest requests[64]; };
__global__ void DraftProbabilityBatchPartial(const __grid_constant__ ProbabilityBatch batch) {
    const auto &a=batch.requests[blockIdx.z];const int row=blockIdx.y,chunk=blockIdx.x,tid=threadIdx.x;
    if(row>=a.steps)return;
    __shared__ float reduce[256];
    if(tid<8 && *a.peers.watchdogs[tid])asm volatile("trap;");
    float maximum=-INFINITY;
    for(int col=chunk*4096+tid;col<(chunk+1)*4096 && col<152576;col+=256) {
        const float value=a.peers.logits[col/19072][row*19072+col%19072]*a.inverseTemperature;
        a.out[size_t(row)*152576+col]=value;maximum=fmaxf(maximum,value);
    }
    reduce[tid]=maximum;__syncthreads();
    for(int d=128;d;d/=2){if(tid<d)reduce[tid]=fmaxf(reduce[tid],reduce[tid+d]);__syncthreads();}
    maximum=reduce[0];float sum=0;
    for(int col=chunk*4096+tid;col<(chunk+1)*4096 && col<152576;col+=256)
        sum+=expf(a.out[size_t(row)*152576+col]-maximum);
    reduce[tid]=sum;__syncthreads();
    for(int d=128;d;d/=2){if(tid<d)reduce[tid]+=reduce[tid+d];__syncthreads();}
    if(!tid)a.partial[row*38+chunk]=make_float2(maximum,reduce[0]);
}
__global__ void DraftProbabilityBatchStats(const __grid_constant__ ProbabilityBatch batch) {
    const auto &a=batch.requests[blockIdx.x];const int row=threadIdx.x;
    if(row>=a.steps)return;
    float maximum=-INFINITY,denominator=0;
    for(int i=0;i<38;++i)maximum=fmaxf(maximum,a.partial[row*38+i].x);
    for(int i=0;i<38;++i){const auto p=a.partial[row*38+i];denominator+=p.y*expf(p.x-maximum);}
    a.partial[row*38]=make_float2(maximum,denominator);
}
__global__ void DraftProbabilityBatchNormalize(const __grid_constant__ ProbabilityBatch batch) {
    const auto &a=batch.requests[blockIdx.z];const int row=blockIdx.y;
    if(row>=a.steps)return;
    const auto stats=a.partial[row*38];
    for(int col=blockIdx.x*4096+threadIdx.x;col<(blockIdx.x+1)*4096 && col<152576;col+=256)
        a.out[size_t(row)*152576+col]=expf(a.out[size_t(row)*152576+col]-stats.x)/stats.y;
}
__global__ void DraftConfidenceBatch(const __grid_constant__ ProbabilityBatch batch,
        const B *hidden,const B *w1,const float *weight,const float *bias,float *out) {
    const int request=blockIdx.y,row=blockIdx.x,tid=threadIdx.x;
    const auto &a=batch.requests[request];if(row>=a.steps)return;
    __shared__ float partial[256];float sum=0;
    if(weight)for(int i=tid;i<4352;i+=256) {
        const float value=i<4096?float(hidden[(size_t(request)*7+row)*4096+i]):float(w1[size_t(a.ids[row])*256+i-4096]);
        sum+=value*weight[i];
    }
    partial[tid]=sum;__syncthreads();
    for(int d=128;d;d/=2){if(tid<d)partial[tid]+=partial[tid+d];__syncthreads();}
    if(!tid)out[request*7+row]=partial[0]+(bias?*bias:0.f);
}

__global__ void DraftProbabilityPartial(ProbabilityArgs a, float *out, float2 *partial, float inverseTemperature) {
    __shared__ float reduce[256];
    const int tid=threadIdx.x, row=blockIdx.y, chunk=blockIdx.x;
    for (int rank=tid;rank<8;rank+=256) if (*a.watchdogs[rank]) asm volatile("trap;");
    float maximum=-INFINITY;
    for (int col=chunk*4096+tid;col<(chunk+1)*4096 && col<152576;col+=256) {
        float value=a.logits[col/19072][row*19072+col%19072]*inverseTemperature;
        out[size_t(row)*152576+col]=value; maximum=fmaxf(maximum,value);
    }
    reduce[tid]=maximum; __syncthreads();
    for (int d=128;d;d/=2) { if(tid<d) reduce[tid]=fmaxf(reduce[tid],reduce[tid+d]); __syncthreads(); }
    maximum=reduce[0]; float sum=0;
    for (int col=chunk*4096+tid;col<(chunk+1)*4096 && col<152576;col+=256)
        sum+=expf(out[size_t(row)*152576+col]-maximum);
    reduce[tid]=sum; __syncthreads();
    for(int d=128;d;d/=2) { if(tid<d) reduce[tid]+=reduce[tid+d]; __syncthreads(); }
    if(!tid) partial[row*38+chunk]=make_float2(maximum,reduce[0]);
}
__global__ void DraftProbabilityNormalize(float *out,const float2 *partial) {
    const int row=blockIdx.y;
    float maximum=-INFINITY,denominator=0;
    for(int chunk=0;chunk<38;++chunk) maximum=fmaxf(maximum,partial[row*38+chunk].x);
    for(int chunk=0;chunk<38;++chunk) { auto p=partial[row*38+chunk]; denominator+=p.y*expf(p.x-maximum); }
    for(int col=blockIdx.x*4096+threadIdx.x;col<(blockIdx.x+1)*4096 && col<152576;col+=256)
        out[size_t(row)*152576+col]=expf(out[size_t(row)*152576+col]-maximum)/denominator;
}
__global__ void DraftConfidence(const B *hidden,const B *w1,const float *weight,const float *bias,
        const int64_t *ids,float *out) {
    __shared__ float partial[256];
    const int row=blockIdx.x,tid=threadIdx.x;
    float sum=0;
    if(weight) for(int i=tid;i<4352;i+=256) {
        float value=i<4096 ? float(hidden[row*4096+i]) : float(w1[size_t(ids[row])*256+i-4096]);
        sum+=value*weight[i];
    }
    partial[tid]=sum; __syncthreads();
    for(int d=128;d;d/=2) { if(tid<d) partial[tid]+=partial[tid+d]; __syncthreads(); }
    if(!tid) out[row]=partial[0]+(bias?*bias:0.f);
}
__global__ void VerificationInput(const int64_t *tokens, float *input, int stride, int offset, int position, int rows) {
    const int row = threadIdx.x;
    if (row < rows) {
        input[offset+row] = float(tokens[row]);
        input[stride+offset+row] = float(position+row);
        reinterpret_cast<int *>(input+2*stride)[offset+row] = position+row+1;
    }
}
__global__ void VerificationInputPrepared(const int64_t *tokens,float *input,int stride,int offset,
        const FastllmDraftHeadRequest *request,int rows) {
    const int row=threadIdx.x, position=request->position;
    if(row<rows) {
        input[offset+row]=float(tokens[row]); input[stride+offset+row]=float(position+row);
        reinterpret_cast<int *>(input+2*stride)[offset+row]=position+row+1;
    }
}
struct VerificationPlan {
    int device,count;
    FastllmDraftHeadRequest *host=nullptr,*gpu=nullptr;
    cudaEvent_t uploaded=nullptr;
    cudaGraphExec_t exec=nullptr;
    cudaGraph_t graph=nullptr;
    std::vector<cudaGraphNode_t> prefix;
    bool inFlight=false,head=true;
    ~VerificationPlan() {
        cudaSetDevice(device);
        if(inFlight) cudaEventSynchronize(uploaded);
        if(exec) cudaGraphExecDestroy(exec);
        if(graph) cudaGraphDestroy(graph);
        if(uploaded) cudaEventDestroy(uploaded);
        if(host) cudaFreeHost(host);
        if(gpu) cudaFree(gpu);
    }
};
}
extern "C" void *FastllmDraftFusedVerificationCreate(void *opaque,int rank,int count,const void *hidden,
        void *input,int stride,void *headReady,void *targetGraph,int verifyRows) {
    if(!opaque || rank<0 || rank>=8 || count<1 || count>64 || !hidden || !input ||
        verifyRows<2 || verifyRows>8 || stride<count*verifyRows ||
        !headReady || !targetGraph) return nullptr;
    auto &s=*static_cast<State*>(opaque); auto &r=s.ranks[rank];
    if(!s.headReady || count>int(r.lanes.size())) return nullptr;
    std::unique_ptr<VerificationPlan> p(new VerificationPlan{r.device,count});
    try {
        Check(cudaSetDevice(r.device));
        cudaFuncAttributes attrs{};
        Check(cudaFuncGetAttributes(&attrs,BeginHeadPrepared));
        Check(cudaFuncGetAttributes(&attrs,VerificationInputPrepared));
        Check(cudaFuncGetAttributes(&attrs,BeginHeadBatch));
        Check(cudaFuncGetAttributes(&attrs,DraftHeadMarkovScores));
        Check(cudaFuncGetAttributes(&attrs,DraftHeadMarkovPick));
        Check(cudaFuncGetAttributes(&attrs,VerificationInputBatch));
        Check(cudaMallocHost(&p->host,count*sizeof(*p->host)));
        Check(cudaMalloc(&p->gpu,count*sizeof(*p->gpu)));
        Check(cudaEventCreateWithFlags(&p->uploaded,cudaEventDisableTiming));
        Check(cudaGraphCreate(&p->graph,0));
        cudaGraphNode_t previous=nullptr;
        auto add=[&](void *function,dim3 blocks,int threads,int smem,void **args) {
            cudaKernelNodeParams params{};
            params.func=function;params.gridDim=dim3(blocks);params.blockDim=dim3(threads);
            params.sharedMemBytes=smem;params.kernelParams=args;
            cudaGraphNode_t node;
            Check(cudaGraphAddKernelNode(&node,p->graph,previous?&previous:nullptr,previous?1:0,&params));
            previous=node;p->prefix.push_back(node);
        };
        if(count>=8 && count<=r.headCapacity && r.headBatch) {
            auto *lanes=r.headBatch;auto *requests=p->gpu;int steps=verifyRows-1;
            void *begin[]={&lanes,&requests};add((void*)BeginHeadBatch,count,256,0,begin);
            {
                Blas blas(r.device);
                AddBlasNodes(p->graph,previous,&p->prefix,[&](cudaStream_t stream){HeadProjection(r,blas.handle,hidden,count,stream);});
            }
            for(int step=0;step<steps;++step) {
                void *markov[]={&lanes,&step};
                add((void*)DraftHeadMarkovScores,dim3(HeadMarkovTiles,count),256,0,markov);
                add((void*)DraftHeadMarkovPick,count,32,0,markov);
            }
            auto *packed=static_cast<float*>(input);
            void *finish[]={&lanes,&requests,&packed,&stride,&verifyRows};
            add((void*)VerificationInputBatch,count,32,0,finish);
        } else for(int first=0;first<count;) {
            const int requests=s.batchHead4 && first+3<count ? 4 : s.batchHead && first+1<count ? 2 : 1;
            HeadQuadArgs tile{};tile.steps=verifyRows-1;
            for(int requestIndex=0;requestIndex<requests;++requestIndex) {
                const int row=first+requestIndex;
                auto &l=r.lanes[row];auto *request=p->gpu+row;
                BeginArgs b{};
                for(int peer=0;peer<8;++peer)b.peers[peer]=s.ranks[peer].lanes[row].mailbox;
                b.rank=rank;b.epoch=l.epoch;b.watchdog=l.watchdog;
                b.input=static_cast<const B*>(hidden)+size_t(row)*7*4096;b.output=l.padded;b.rows=tile.steps;
                b.headIds=l.headIds;b.sampling=l.sampling;b.samplingParams=l.samplingParams;
                void *begin[]={&b,&request};add((void*)BeginHeadPrepared,1,256,0,begin);
                auto &h=tile.head[requestIndex];h.resid_in=l.padded;h.lm_head=r.head;h.logits=l.headLogits;
                h.watchdog_limit_ns=5000000000ull;h.watchdog_flag=l.watchdog;h.first_logit_row=0;
                auto &m=tile.markov[requestIndex];m.block_ids=l.headIds;m.draft_tokens=l.headIds+1;m.prev_token=l.headIds+7;
                m.logit_trace=l.corrected;m.bias_table=r.biasTable;m.replay_generation=l.epoch;
                m.rank=rank;m.apply_markov=1;m.watchdog_limit_ns=5000000000ull;m.watchdog_flag=l.watchdog;
                m.sampling_state=l.sampling;m.sampling_params=l.samplingParams;
                for(int peer=0;peer<8;++peer)m.mailbox_peers[peer]=s.ranks[peer].lanes[row].mailbox;
                tile.control[requestIndex]=l.headControl;
            }
            if(requests==4) {
                void *head[]={&tile};add((void*)FusedDraftHeadQuad,kHeadCtas,kHeadThreads,HeadQuadSmem,head);
            } else if(requests==2) {
                HeadTileArgs pair{};pair.steps=tile.steps;
                for(int i=0;i<2;++i){pair.head[i]=tile.head[i];pair.markov[i]=tile.markov[i];pair.control[i]=tile.control[i];}
                void *head[]={&pair};add((void*)FusedDraftHeadTile,kHeadCtas,kHeadThreads,HeadTileSmem,head);
            } else {
                void *head[]={tile.head,tile.markov,tile.control,&tile.steps};
                add((void*)FusedDraftHeadProposal,kHeadCtas,kHeadThreads,HeadSmem,head);
            }
            for(int requestIndex=0;requestIndex<requests;++requestIndex) {
                const int row=first+requestIndex;auto &l=r.lanes[row];auto *request=p->gpu+row;
                B *none=nullptr;int zero=0;
                void *finish[]={&none,&none,&zero,&l.watchdog};add((void*)FinishResult,1,32,0,finish);
                int offset=row*verifyRows;auto *packed=static_cast<float*>(input);
                void *adapter[]={&l.headIds,&packed,&stride,&offset,&request,&verifyRows};
                add((void*)VerificationInputPrepared,1,32,0,adapter);
            }
            first+=requests;
        }
        cudaGraphNode_t ready,target;
        Check(cudaGraphAddEventRecordNode(&ready,p->graph,&previous,1,static_cast<cudaEvent_t>(headReady)));
        Check(cudaGraphAddChildGraphNode(&target,p->graph,&ready,1,static_cast<cudaGraph_t>(targetGraph)));
        Check(cudaGraphInstantiateWithFlags(&p->exec,p->graph,0));
        Check(cudaGraphUpload(p->exec,cudaStreamPerThread));
        Check(cudaStreamSynchronize(cudaStreamPerThread));
        return p.release();
    } catch(const std::exception &e) {
        error=e.what();cudaGetLastError();return nullptr;
    }
}
extern "C" void FastllmDraftFusedVerificationDestroy(void *p) { delete static_cast<VerificationPlan*>(p); }
extern "C" void *FastllmDraftFusedVerificationExec(void *opaque,bool head) {
    if(!opaque)return nullptr;
    auto &p=*static_cast<VerificationPlan*>(opaque);
    try {
        if(p.head!=head) {
            for(auto node:p.prefix)Check(cudaGraphNodeSetEnabled(p.exec,node,head?1u:0u));
            p.head=head;
        }
        return p.exec;
    }catch(const std::exception &e){error=e.what();return nullptr;}
}
extern "C" bool FastllmDraftFusedVerificationUpload(void *opaque,const FastllmDraftHeadRequest *requests,
        int count,void *rawStream) {
    if(!opaque || !requests) return false;
    auto &p=*static_cast<VerificationPlan*>(opaque);
    if(count!=p.count) return false;
    for(int i=0;i<count;++i)
        if(requests[i].anchor<0 || requests[i].anchor>=152576 || requests[i].position<0 ||
            !std::isfinite(requests[i].temperature) || requests[i].temperature<=0) return false;
    try {
        auto stream=rawStream?static_cast<cudaStream_t>(rawStream):cudaStreamPerThread;
        if(p.inFlight)Check(cudaEventSynchronize(p.uploaded));
        std::memcpy(p.host,requests,count*sizeof(*requests));
        Check(cudaMemcpyAsync(p.gpu,p.host,count*sizeof(*requests),cudaMemcpyHostToDevice,stream));
        Check(cudaEventRecord(p.uploaded,stream));p.inFlight=true;return true;
    } catch(const std::exception &e) {error=e.what();return false;}
}
extern "C" const int64_t *FastllmDraftFusedHeadTokens(void *opaque, int rank, int request) {
    if (!opaque || rank < 0 || rank >= 8 || request < 0) return nullptr;
    auto &s = *static_cast<State *>(opaque);
    return s.headReady && request < int(s.ranks[rank].lanes.size()) ? s.ranks[rank].lanes[request].headIds : nullptr;
}
extern "C" bool FastllmDraftFusedVerificationInput(const int64_t *tokens, void *input,
        int stride, int offset, int position, int rows, void *rawStream) {
    if (!tokens || !input || offset < 0 || rows < 2 || rows > 8 || stride < offset+rows || position < 0) return false;
    VerificationInput<<<1,32,0,rawStream?static_cast<cudaStream_t>(rawStream):cudaStreamPerThread>>>(
        tokens,static_cast<float *>(input),stride,offset,position,rows);
    return cudaGetLastError() == cudaSuccess;
}
extern "C" bool FastllmDraftFusedHeadPrepare(void *opaque,const void *const *heads,
        const void *w1,const void *w2,const float *confidenceWeight,const float *confidenceBias,int owner) {
    return FastllmDraftFusedHeadPrepareWithBudget(opaque,heads,w1,w2,confidenceWeight,confidenceBias,owner,1.0);
}
extern "C" bool FastllmDraftFusedHeadPrepareWithBudget(void *opaque,const void *const *heads,
        const void *w1,const void *w2,const float *confidenceWeight,const float *confidenceBias,
        int owner,double gpuMemRatio) {
    if(!opaque || !heads || !w1 || !w2 || !(gpuMemRatio>0.0 && gpuMemRatio<=1.0)) return false;
    auto &s=*static_cast<State*>(opaque);
    if(s.rows!=7 || owner!=s.ranks[0].device) return false;
    if(s.headReady) return true;
    std::vector<size_t> before;
    for(auto &r:s.ranks) before.push_back(r.allocations.size());
    try {
        const size_t tableBytes=size_t(152576)*19072*sizeof(B);
        for(int rank=0;rank<8;++rank) {
            if(!heads[rank]) return false;
            auto &r=s.ranks[rank]; Check(cudaSetDevice(r.device));
            size_t free=0,total=0;Check(cudaMemGetInfo(&free,&total));
            // The extra GiB covers the packed head, staging and bounded lane scratch.
            const size_t reserve=static_cast<size_t>(total*(1.0-gpuMemRatio));
            if(free<=reserve || free-reserve<tableBytes+(size_t(1)<<30)) {
                Check(cudaSetDevice(owner));
                return false;
            }
            Check(cudaFuncSetAttribute(FusedDraftHeadProposal,cudaFuncAttributeMaxDynamicSharedMemorySize,HeadSmem));
            int active=0,sms=0;
            Check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,FusedDraftHeadProposal,kHeadThreads,HeadSmem));
            Check(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,r.device));
            if(active*sms<kHeadCtas) return false;
            s.batchHead4=ConfigureOptionalKernel(r.device,FusedDraftHeadQuad,kHeadThreads,
                HeadQuadSmem,kHeadCtas) && s.batchHead4;
            s.batchHead=ConfigureOptionalKernel(r.device,FusedDraftHeadTile,kHeadThreads,
                HeadTileSmem,kHeadCtas) && s.batchHead;
        }
        for(int rank=0;rank<8;++rank) {
            auto &r=s.ranks[rank]; Check(cudaSetDevice(r.device));
            r.head=r.Alloc<B>(size_t(19072)*4096);
            r.biasTable=r.Alloc<B>(size_t(152576)*19072);
            Allocation first(size_t(152576)*256*sizeof(B),r.device),second(size_t(19072)*256*sizeof(B),r.device);
            Check(cudaMemcpyPeer(first.ptr,r.device,w1,owner,size_t(152576)*256*sizeof(B)));
            Check(cudaMemcpyPeer(second.ptr,r.device,static_cast<const B*>(w2)+size_t(rank)*19072*256,owner,size_t(19072)*256*sizeof(B)));
            PackHead<<<(size_t(19072)*4096+255)/256,256>>>(static_cast<const B*>(heads[rank]),r.head);
            PrecomputeDraftBias<<<dim3(19072/64,19072),128>>>(static_cast<const B*>(first.ptr),static_cast<const B*>(second.ptr),r.biasTable);
            Check(cudaGetLastError()); Check(cudaDeviceSynchronize());
            r.rawHead=static_cast<const B*>(heads[rank]);
            r.headCapacity=r.lanes.size();
            r.headLogits=r.Alloc<float>(size_t(r.headCapacity)*7*19072);
            r.headIds=r.Alloc<int64_t>(size_t(r.headCapacity)*8);
            r.headConfidence=r.Alloc<float>(size_t(r.headCapacity)*7);
            for(int i=0;i<r.headCapacity;++i) AllocateHeadLane(r,r.lanes[i],i);
            // Resolve and exercise the input adapter before serving collectives.
            // Loading a new CUDA function while peers wait can deadlock the driver.
            if (!r.lanes.empty()) {
                Allocation warm(3*8*sizeof(float),r.device);
                VerificationInput<<<1,32>>>(r.lanes[0].headIds,static_cast<float *>(warm.ptr),8,0,0,8);
                Check(cudaGetLastError()); Check(cudaDeviceSynchronize());
            }
            // Publish initialized control buffers before nonblocking worker streams can use them.
            Check(cudaDeviceSynchronize());
        }
        for(int rank=0;rank<8;++rank) PrepareHeadBatch(s,rank);
        auto &r=s.ranks[0]; r.markovW1=static_cast<const B*>(w1);
        r.confidenceWeight=confidenceWeight; r.confidenceBias=confidenceBias;
        s.headReady=true; Check(cudaSetDevice(owner));return true;
    } catch(const std::exception &e) {
        error=e.what();
        for(int rank=0;rank<8;++rank) { auto &r=s.ranks[rank];r.allocations.resize(before[rank]);r.head=nullptr;r.biasTable=nullptr; }
        cudaSetDevice(owner);return false;
    }
}
extern "C" bool FastllmDraftFusedHeadRun(void *opaque,int rank,int request,const void *hidden,
        int anchor,uint64_t seed,float temperature,int steps,void *rawStream) {
    if(!opaque || rank<0 || rank>=8 || !hidden || anchor<0 || anchor>=152576 ||
        !std::isfinite(temperature) || temperature<=0 || steps<1 || steps>7) return false;
    try {
        auto &s=*static_cast<State*>(opaque);auto &r=s.ranks[rank];
        if(!s.headReady || request<0 || request>=int(r.lanes.size())) return false;
        auto &l=r.lanes[request];cudaStream_t stream=rawStream?static_cast<cudaStream_t>(rawStream):cudaStreamPerThread;
        BeginArgs b{};for(int peer=0;peer<8;++peer)b.peers[peer]=s.ranks[peer].lanes[request].mailbox;
        b.rank=rank;b.epoch=l.epoch;b.watchdog=l.watchdog;b.input=static_cast<const B*>(hidden);b.output=l.padded;b.rows=steps;
        b.headIds=l.headIds;b.sampling=l.sampling;b.samplingParams=l.samplingParams;b.anchor=anchor;b.seed=seed;b.inverseTemperature=1.f/temperature;
        Begin<<<1,256,0,stream>>>(b);
        HeadArgs h{};h.resid_in=l.padded;h.lm_head=r.head;h.logits=l.headLogits;
        h.watchdog_limit_ns=5000000000ull;h.watchdog_flag=l.watchdog;h.first_logit_row=0;
        MarkovArgs m{};m.block_ids=l.headIds;m.draft_tokens=l.headIds+1;m.prev_token=l.headIds+7;
        m.logit_trace=l.corrected;m.bias_table=r.biasTable;m.replay_generation=l.epoch;
        m.rank=rank;m.apply_markov=1;m.watchdog_limit_ns=5000000000ull;m.watchdog_flag=l.watchdog;
        m.sampling_state=l.sampling;m.sampling_params=l.samplingParams;
        for(int peer=0;peer<8;++peer)m.mailbox_peers[peer]=s.ranks[peer].lanes[request].mailbox;
        FusedDraftHeadProposal<<<kHeadCtas,kHeadThreads,HeadSmem,stream>>>(h,m,l.headControl,steps);
        FinishResult<<<1,32,0,stream>>>(nullptr,nullptr,0,l.watchdog);
        Check(cudaGetLastError());return true;
    } catch(const std::exception &e) {error=e.what();return false;}
}
extern "C" bool FastllmDraftFusedHeadSubmit(void *opaque,int request,const void *hidden,
        float temperature,float *probabilities,int64_t *tokens,float *scores,int steps,void *rawStream) {
    if(!opaque || !hidden || !probabilities || !tokens || !std::isfinite(temperature) ||
        temperature<=0 || steps<1 || steps>7) return false;
    try {
        auto &s=*static_cast<State*>(opaque);auto &r=s.ranks[0];
        if(!s.headReady || request<0 || request>=int(r.lanes.size())) return false;
        auto &l=r.lanes[request];cudaStream_t stream=rawStream?static_cast<cudaStream_t>(rawStream):cudaStreamPerThread;
        ProbabilityArgs p{};
        for(int peer=0;peer<8;++peer) {p.logits[peer]=s.ranks[peer].lanes[request].corrected;p.watchdogs[peer]=s.ranks[peer].lanes[request].watchdog;}
        DraftProbabilityPartial<<<dim3(38,steps),256,0,stream>>>(p,probabilities,reinterpret_cast<float2*>(l.softmax),1.f/temperature);
        DraftProbabilityNormalize<<<dim3(38,steps),256,0,stream>>>(probabilities,reinterpret_cast<float2*>(l.softmax));
        if(scores) DraftConfidence<<<steps,256,0,stream>>>(static_cast<const B*>(hidden),r.markovW1,r.confidenceWeight,r.confidenceBias,l.headIds,l.confidence);
        Check(cudaGetLastError());
        Check(cudaMemcpyAsync(tokens,l.headIds+1,steps*sizeof(int64_t),cudaMemcpyDeviceToHost,stream));
        if(scores) Check(cudaMemcpyAsync(scores,l.confidence,steps*sizeof(float),cudaMemcpyDeviceToHost,stream));
        return true;
    } catch(const std::exception &e) {error=e.what();return false;}
}
extern "C" bool FastllmDraftFusedHeadSubmitBatch(void *opaque,int count,const void *hidden,
        const float *temperatures,const int *steps,float *probabilities,int64_t *tokens,float *scores,void *rawStream) {
    if(!opaque || count<1 || count>64 || !hidden || !temperatures || !steps || !probabilities || !tokens)return false;
    auto &s=*static_cast<State*>(opaque);auto &r=s.ranks[0];
    if(!s.headReady || count>int(r.lanes.size()))return false;
    for(int i=0;i<count;++i)
        if(!std::isfinite(temperatures[i]) || temperatures[i]<=0 || steps[i]<1 || steps[i]>7)return false;
    if(count>r.headCapacity) {
        for(int i=0;i<count;++i)
            if(!FastllmDraftFusedHeadSubmit(opaque,i,static_cast<const B*>(hidden)+size_t(i)*7*4096,
                temperatures[i],probabilities+size_t(i)*7*152576,tokens+i*7,scores?scores+i*7:nullptr,steps[i],rawStream))return false;
        return true;
    }
    ProbabilityBatch batch{};
    for(int i=0;i<count;++i) {
        auto &a=batch.requests[i];auto &l=r.lanes[i];
        for(int peer=0;peer<8;++peer){a.peers.logits[peer]=s.ranks[peer].lanes[i].corrected;a.peers.watchdogs[peer]=s.ranks[peer].lanes[i].watchdog;}
        a.out=probabilities+size_t(i)*7*152576;a.partial=reinterpret_cast<float2*>(l.softmax);
        a.ids=l.headIds;a.inverseTemperature=1.f/temperatures[i];a.steps=steps[i];
    }
    try {
        const auto stream=rawStream?static_cast<cudaStream_t>(rawStream):cudaStreamPerThread;
        DraftProbabilityBatchPartial<<<dim3(38,7,count),256,0,stream>>>(batch);
        DraftProbabilityBatchStats<<<count,32,0,stream>>>(batch);
        DraftProbabilityBatchNormalize<<<dim3(38,7,count),256,0,stream>>>(batch);
        if(scores)DraftConfidenceBatch<<<dim3(7,count),256,0,stream>>>(batch,static_cast<const B*>(hidden),r.markovW1,r.confidenceWeight,r.confidenceBias,r.headConfidence);
        Check(cudaGetLastError());
        Check(cudaMemcpy2DAsync(tokens,7*sizeof(int64_t),r.headIds+1,8*sizeof(int64_t),7*sizeof(int64_t),count,cudaMemcpyDeviceToHost,stream));
        if(scores)Check(cudaMemcpyAsync(scores,r.headConfidence,size_t(count)*7*sizeof(float),cudaMemcpyDeviceToHost,stream));
        return true;
    }catch(const std::exception &e){error=e.what();return false;}
}

extern "C" bool FastllmDraftFusedHeadFinish(void *opaque,int request,const void *hidden,
        float temperature,float *probabilities,int64_t *tokens,float *scores,int steps,void *rawStream) {
    if (!FastllmDraftFusedHeadSubmit(opaque,request,hidden,temperature,probabilities,tokens,scores,steps,rawStream)) return false;
    try {
        Check(cudaStreamSynchronize(rawStream?static_cast<cudaStream_t>(rawStream):cudaStreamPerThread));
        return true;
    } catch(const std::exception &e) {error=e.what();return false;}
}
extern "C" bool FastllmDraftFusedHeadProbe(void *opaque,int rank,int request,void *base,void *corrected,
        void *ids,void *biasTableRows,const int64_t *previous,int rows,void *rawStream) {
    if(!opaque || rank<0 || rank>=8) return false;
    try {
        auto &s=*static_cast<State*>(opaque);auto &r=s.ranks[rank];
        if(!s.headReady || request<0 || request>=int(r.lanes.size())) return false;
        auto &l=r.lanes[request];cudaStream_t stream=rawStream?static_cast<cudaStream_t>(rawStream):cudaStreamPerThread;
        if(base)Check(cudaMemcpyAsync(base,l.headLogits,7*19072*sizeof(float),cudaMemcpyDeviceToDevice,stream));
        if(corrected)Check(cudaMemcpyAsync(corrected,l.corrected,7*19072*sizeof(float),cudaMemcpyDeviceToDevice,stream));
        if(ids)Check(cudaMemcpyAsync(ids,l.headIds,8*sizeof(int64_t),cudaMemcpyDeviceToDevice,stream));
        for(int i=0;i<rows;++i) {
            if(previous[i]<0 || previous[i]>=152576)return false;
            Check(cudaMemcpyAsync(static_cast<B*>(biasTableRows)+size_t(i)*19072,r.biasTable+size_t(previous[i])*19072,
                19072*sizeof(B),cudaMemcpyDeviceToDevice,stream));
        }
        return true;
    }catch(const std::exception &e){error=e.what();return false;}
}
