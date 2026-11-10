#pragma once

// IQ block/codebook arithmetic follows fastllm-moe-gguf-q8.cuh and
// fastllm-gguf-iq2-dot.cuh (ggml authors / Iwan Kawrakow, MIT).
// Scheduling and cross-route reuse are local FastLLM changes.

// Reuse the format decoders while retaining ordinary GGUF's floating-point
// fractional scales (the V4.1 IQ2 dot deliberately truncates those scales).
template<ggml_type Type>
struct GGUFDecoded32 {
    int values[8], lo=0, hi=0;
    float scale;
    __device__ GGUFDecoded32(const void *weight, int k, const uint64_t *grid,
            bool packed, int packedRow) {
        using namespace gguf_cache_q8;
        if constexpr (Type==GGML_TYPE_IQ2_XXS || Type==GGML_TYPE_IQ2_XS || Type==GGML_TYPE_IQ2_S) {
            iq2_decode::Decoded32 d;
            if constexpr(Type==GGML_TYPE_IQ2_XXS)
                d=packed?iq2_decode::DecodeXXS<true>(weight,k/8,2*(k%8),grid,packedRow):
                    iq2_decode::DecodeXXS<false>(weight,k/8,2*(k%8),grid);
            else if constexpr(Type==GGML_TYPE_IQ2_XS)
                d=packed?iq2_decode::DecodeXS<true>(weight,k/8,2*(k%8),grid,packedRow):
                    iq2_decode::DecodeXS<false>(weight,k/8,2*(k%8),grid);
            else d=packed?iq2_decode::DecodeS<true>(weight,k/8,2*(k%8),grid,packedRow):
                    iq2_decode::DecodeS<false>(weight,k/8,2*(k%8),grid);
#pragma unroll
            for(int j=0;j<8;++j)values[j]=d.values[j];
            lo=d.lo;hi=d.hi;scale=d.scale;
        } else if constexpr(Type==GGML_TYPE_IQ3_XXS || Type==GGML_TYPE_IQ3_S) {
            int2 indices;unsigned signs,high=0;
            if constexpr(Type==GGML_TYPE_IQ3_S) {
                const auto &q=static_cast<const block_iq3_s *>(weight)[k/8];
                const int part=k%8;
                indices=make_int2(get_int_b2(q.qs,2*part),get_int_b2(q.qs,2*part+1));
                signs=get_int_b2(q.signs,part);high=q.qh[part];
                lo=1+2*((q.scales[part/2]>>(4*(part%2)))&15);scale=__half2float(q.d);
            } else {
                unsigned aux;const int part=k%8;
                if(packed) {
                    const auto &q=static_cast<const block_iq3_xxs_r4 *>(weight)[k/8];
                    indices=make_int2(get_int_b4(q.qs,8*part+2*packedRow),get_int_b4(q.qs,8*part+2*packedRow+1));
                    aux=get_int_b4(q.sas,4*part+packedRow);scale=__half2float(q.d[packedRow]);
                    lo=(aux&1)|((aux>>7)&2)|((aux>>14)&4)|((aux>>21)&8);
                } else {
                    const auto &q=static_cast<const block_iq3_xxs *>(weight)[k/8];
                    indices=make_int2(get_int_b2(q.qs,2*part),get_int_b2(q.qs,2*part+1));
                    aux=get_int_b2(q.qs,QK_K/16+part);scale=__half2float(q.d);lo=aux>>28;
                }
                signs=0;
#pragma unroll
                for(int j=0;j<4;++j) {
                    const unsigned v=packed?(aux>>(8*j))&255:(aux>>(7*j))&127;
                    signs|=(packed?((v>>1)^(v&254)):(v|((__popc(v)&1)<<7)))<<(8*j);
                }
            }
            const auto *q=reinterpret_cast<const uint8_t *>(&indices);
            const auto *g=reinterpret_cast<const uint32_t *>(grid);
#pragma unroll
            for(int j=0;j<8;++j) {
                const unsigned index=q[j]|((high<<(8-j))&256u),mask=SignMask4(signs>>(4*j));
                values[j]=(g[index]^mask)+(mask&0x01010101u);
            }
        } else {
            const uint8_t *qs;
            if constexpr(Type==GGML_TYPE_IQ4_NL) {
                const auto &q=static_cast<const block_iq4_nl *>(weight)[k];qs=q.qs;scale=__half2float(q.d);
            } else {
                const auto &q=static_cast<const block_iq4_xs *>(weight)[k/8];const int part=k%8;
                qs=q.qs+16*part;scale=__half2float(q.d)*((((q.scales_l[part/2]>>(4*(part%2)))&15)|
                    (((q.scales_h>>(2*part))&3)<<4))-32);
            }
#pragma unroll
            for(int j=0;j<4;++j) {const int2 v=UnpackIQ4(get_int_b2(qs,j));values[j]=v.x;values[j+4]=v.y;}
        }
    }
    __device__ float Dot(const block_q8_1 &x) const {
        using namespace gguf_cache_q8;
        int a=0,b=0;
#pragma unroll
        for(int j=0;j<4;++j)a=ggml_cuda_dp4a(values[j],get_int_b4(x.qs,j),a);
#pragma unroll
        for(int j=4;j<8;++j)b=ggml_cuda_dp4a(values[j],get_int_b4(x.qs,j),b);
        float v;
        if constexpr(Type==GGML_TYPE_IQ2_XXS)v=(lo+0.5f)*(a+b)*0.25f;
        else if constexpr(Type==GGML_TYPE_IQ2_XS || Type==GGML_TYPE_IQ2_S)v=((lo+0.5f)*a+(hi+0.5f)*b)*0.25f;
        else if constexpr(Type==GGML_TYPE_IQ3_XXS)v=(lo+0.5f)*0.5f*(a+b);
        else if constexpr(Type==GGML_TYPE_IQ3_S)v=(a+b)*lo;
        else v=a+b;
        return (scale*__low2float(x.ds))*v;
    }
};

template<typename T, typename View>
__global__ void BindGGUFGroups(View view, T *activation, float *partial, uint64_t *keys, int *entries, int first) {
    const int r = first + blockIdx.x;
    const uint8_t *weight = ActiveRoute(view,r) ? ExpertWeight<true>(view,r) : nullptr;
    if (!threadIdx.x) {keys[r]=reinterpret_cast<uintptr_t>(weight);entries[r]=r;}
    if (!weight) {
        for (int col=threadIdx.x;col<view.inter;col+=blockDim.x)
            activation[size_t(r)*view.inter+col]=DequantizeCast<T>::cast(0);
        if (partial && ActiveRoute(view,r))
            for (int col=threadIdx.x;col<view.hidden;col+=blockDim.x)
                partial[size_t(OriginalRoute(view,r))*view.hidden+col]=0;
    }
}

template<ggml_type Type, bool Gate, typename T, typename View, int Tile = 2, int Lanes = Gate ? 16 : 8>
__global__ void GGUFGroupedProjection(const block_q8_1 *input, T *activation, float *partial, View view, int topk,
        fastllm::moe_routes::Plan plan, size_t rowBytes, int firstGroup, bool sharedInput) {
    const int group = firstGroup + blockIdx.y;
    if (group >= *plan.groups) return;
    extern __shared__ uint32_t stagedInput[];
    __shared__ uint64_t grid[gguf_cache_q8::Format<Type>::gridSize ? gguf_cache_q8::Format<Type>::gridSize : 1];
    gguf_cache_q8::StageGrid<Type>(grid);
    __syncthreads();
    constexpr int virtualLanes = 32 / Lanes;
    const int row=blockIdx.x*(256/Lanes)+threadIdx.x/Lanes, lane=threadIdx.x%Lanes;
    if(row>=(Gate?view.inter:view.hidden))return;
    const int storage=NumaType<Gate>(view);
    const bool interleaved=Gate && storage>=0;
    const auto *record=ExpertWeight<Gate>(view,plan.entries[plan.starts[group]]);
    const bool packed=storage>=0 && storage!=Type;
    const int gRow=interleaved?2*row:row,uRow=interleaved?2*row+1:row+view.inter;
    const auto *gate=record+size_t(packed?gRow/4*4:gRow)*rowBytes;
    const auto *up=record+size_t(packed?uRow/4*4:uRow)*rowBytes;
    const int columns=Gate?view.hidden:view.inter;
    const int end=plan.starts[group+1];
    for(int first=plan.starts[group];first<end;first+=Tile) {
        int routes[Tile];const block_q8_1 *xs[Tile];float g[Tile][virtualLanes]{},u[Tile][virtualLanes]{};
#pragma unroll
        for(int j=0;j<Tile;++j) {
            routes[j]=plan.entries[min(first+j,end-1)];
            xs[j]=input+size_t(Gate?OriginalRoute(view,routes[j])/topk:routes[j])*(columns/32);
        }
        if (sharedInput) {
            const int words = columns / 32 * sizeof(block_q8_1) / sizeof(uint32_t);
#pragma unroll
            for (int j=0;j<Tile;++j) if (first+j<end) {
                const auto *source = reinterpret_cast<const uint32_t *>(xs[j]);
                for (int i=threadIdx.x;i<words;i+=blockDim.x)
                    stagedInput[j*words+i] = source[i];
                xs[j] = reinterpret_cast<const block_q8_1 *>(stagedInput+j*words);
            }
            __syncthreads();
        }
        // Each physical lane retains the original warp lanes separately.
        // Fold virtual lanes before shuffling so accumulation stays bit exact.
        for(int base=lane;base<columns/32;base+=32) {
#pragma unroll
            for(int v=0;v<virtualLanes;++v) {
                const int k=base+v*Lanes;
                if(k>=columns/32)continue;
                GGUFDecoded32<Type> dg(gate,k,grid,packed,gRow%4);
#pragma unroll
                for(int j=0;j<Tile;++j)if(first+j<end)g[j][v]+=dg.Dot(xs[j][k]);
                if constexpr(Gate) {
                    GGUFDecoded32<Type> du(up,k,grid,packed,uRow%4);
#pragma unroll
                    for(int j=0;j<Tile;++j)if(first+j<end)u[j][v]+=du.Dot(xs[j][k]);
                }
            }
        }
#pragma unroll
        for(int offset=virtualLanes/2;offset;offset>>=1) {
#pragma unroll
            for(int v=0;v<offset;++v) {
#pragma unroll
                for(int j=0;j<Tile;++j) if(first+j<end) {
                    g[j][v]+=g[j][v+offset];
                    if constexpr(Gate)u[j][v]+=u[j][v+offset];
                }
            }
        }
#pragma unroll
        for(int offset=Lanes/2;offset;offset>>=1) {
#pragma unroll
            for(int j=0;j<Tile;++j) if(first+j<end) {
                g[j][0]+=__shfl_xor_sync(0xffffffff,g[j][0],offset,Lanes);
                if constexpr(Gate)u[j][0]+=__shfl_xor_sync(0xffffffff,u[j][0],offset,Lanes);
            }
        }
        if(!lane) for(int j=0;j<Tile && first+j<end;++j) {
            const float a=float(DequantizeCast<T>::cast(g[j][0]));
            if constexpr(Gate) {
                const float b=float(DequantizeCast<T>::cast(u[j][0]));
                activation[size_t(routes[j])*view.inter+row]=DequantizeCast<T>::cast(a/(1.0f+expf(-a))*b);
            } else partial[size_t(OriginalRoute(view,routes[j]))*view.hidden+row]=a;
        }
        if (sharedInput) __syncthreads();
    }
}

template<bool Gate, typename T, typename View>
bool TryGGUFGroupedProjection(const block_q8_1 *input, T *activation, float *partial,
        const View &view, int rows, int topk, int routes, bool prepared=false, bool stageInput=false) {
    const int type=Gate?view.gateType:view.downType,columns=Gate?view.hidden:view.inter;
    const bool supported=type==GGML_TYPE_IQ2_XXS || type==GGML_TYPE_IQ2_XS || type==GGML_TYPE_IQ2_S ||
        type==GGML_TYPE_IQ3_XXS || type==GGML_TYPE_IQ3_S || type==GGML_TYPE_IQ4_NL || type==GGML_TYPE_IQ4_XS;
    if(rows<=1 || !supported || columns%(type==GGML_TYPE_IQ4_NL?32:256) || !view.workspace || (!Gate && !partial))return false;
    const size_t offset=(Q8WorkspaceBytes(rows,view.hidden,view.inter,topk)+255)&~size_t(255);
    const size_t temporary=fastllm::moe_routes::Plan::TemporaryBytes(routes);
    const size_t bytes=temporary?fastllm::moe_routes::Plan(nullptr,routes,temporary).bytes:0;
    if(!bytes || offset+bytes>view.workspaceBytes)return false;
    fastllm::moe_routes::Plan plan(static_cast<uint8_t *>(view.workspace)+offset,routes,temporary);
    if(!prepared) {
        for(int first=0;first<routes;first+=65535)
            BindGGUFGroups<<<std::min(routes-first,65535),128,0,cudaStreamPerThread>>>(view,activation,partial,plan.keys,plan.routes,first);
        if(!plan.Build(cudaStreamPerThread))return false;
    }
    int sharedLimit = 0;
    if (stageInput && cudaDeviceGetAttribute(&sharedLimit, cudaDevAttrMaxSharedMemoryPerBlock,
            FastllmCudaGetDevice()) != cudaSuccess) return false;
    const size_t inputBytes = size_t(2) * columns / 32 * sizeof(block_q8_1);
    // Reserve the largest supported IQ codebook as well. Large dimensions
    // retain global input reads; this is a storage constraint, not a row cap.
    const bool sharedInput = inputBytes + 1024 * sizeof(uint64_t) <= size_t(sharedLimit);
    for(int first=0;first<routes;first+=65535) {
        constexpr int outputRows = Gate ? 16 : 32;
        const dim3 grid(((Gate?view.inter:view.hidden)+outputRows-1)/outputRows,std::min(routes-first,65535));
        const size_t rowBytes=ggml_row_size(static_cast<ggml_type>(type),columns);
#define GROUPED_CASE(NAME) case GGML_TYPE_##NAME: GGUFGroupedProjection<GGML_TYPE_##NAME,Gate><<<grid,256,sharedInput?inputBytes:0,cudaStreamPerThread>>>(input,activation,partial,view,topk,plan,rowBytes,first,sharedInput);break;
        switch(type) {
            GROUPED_CASE(IQ2_XXS) GROUPED_CASE(IQ2_XS) GROUPED_CASE(IQ2_S)
            GROUPED_CASE(IQ3_XXS) GROUPED_CASE(IQ3_S) GROUPED_CASE(IQ4_NL) GROUPED_CASE(IQ4_XS)
            default:return false;
        }
#undef GROUPED_CASE
    }
    return cudaGetLastError()==cudaSuccess;
}
