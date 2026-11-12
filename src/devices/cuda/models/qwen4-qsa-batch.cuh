// Packed QSA requests. Attention chunk/merge is adapted from Strata's
// qsa_decode_attn.cu, commit dbb2d168cffa8cf793d9325358809d5263c7c6b0.
// Copyright (c) 2026 Niko1221 and the Strata contributors, MIT license.
// See third_party/strata/{LICENSE,README.md}. No Strata runtime is used.
#pragma once
#include <set>

namespace fastllm_qsa_batch {
struct Request {
    half *key, *value;
    float *tail, *tailPositions, *blocks;
    const float *positions;
    const int32_t *meta, *valid;
    float *scores;
    int32_t *selected;
    float *capture;
    uint64_t keyStride, valueStride;
    int start, rows, capacity;
    Qwen4CacheAddress address;
};
struct Requests {
    // CUDA kernel argument storage is bounded; logical batches are not.
    // Large batches are submitted in physical packs without changing math.
    static constexpr int capacity = (3072 - 3 * sizeof(int)) / sizeof(Request);
    Request items[capacity];
    int count, first, rows;
    __device__ const Request &Find(int row) const {
        int i = 0;
        while (i + 1 < count && row >= items[i + 1].start) ++i;
        return items[i];
    }
};
static_assert(sizeof(Requests) < 3500, "QSA descriptors must fit CUDA arguments");
struct Geometry {
    int heads, keyHeads, dim, indexHeads, indexDim, ratio, budget, rotary;
    float theta, eps;
};
__device__ float Sum(float x) {
    for (int n = 16; n; n >>= 1) x += __shfl_xor_sync(0xffffffffu, x, n);
    return x;
}
__device__ float Max(float x) {
    for (int n = 16; n; n >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, n));
    return x;
}

// Read strided projection fields directly, normalize/rotate Q, K and index Q,
// and publish K/V in each request's cache. No Split/Permute/Pack copies.
template<bool Index>
__global__ void Prepare(Requests batch, Geometry g, const half *qgate,
        const half *key, const half *value, const half *index,
        const float *qn, const float *kn, const float *iqn,
        half *query, half *indexQuery, int firstRow) {
    const int row = firstRow + blockIdx.y;
    const Request &r = batch.Find(row);
    const int token = row - r.start, role = blockIdx.x, tid = threadIdx.x;
    const bool isIndex = Index, isQ = !Index && role < g.heads;
    const int h = isIndex || isQ ? role : role - g.heads;
    const int dim = isIndex ? g.indexDim : g.dim;
    const half *src = isQ ? qgate + (uint64_t(row) * g.heads + h) * 2 * dim :
        isIndex ? index + uint64_t(row) * (g.indexHeads + 1) * dim + h * dim :
        key + (uint64_t(row) * g.keyHeads + h) * dim;
    const float *weight = isQ ? qn : isIndex ? iqn : kn;
    const bool valid = !r.valid || token < *r.valid;
    if (!valid) return;
    __shared__ float values[256], sums[2], scale;
    float sum = 0;
    for (int d = tid; d < dim; d += 64) {
        const float v = __half2float(src[d]);
        sum += v * v;
    }
    sum = Sum(sum);
    if (!(tid & 31)) sums[tid / 32] = sum;
    __syncthreads();
    if (!tid) scale = rsqrtf((sums[0] + sums[1]) / dim + g.eps);
    __syncthreads();
    for (int d = tid; d < dim; d += 64)
        values[d] = __half2float(__float2half_rn(__half2float(src[d]) * scale * weight[d]));
    __syncthreads();
    half *dst = isQ ? query + (uint64_t(row) * g.heads + h) * dim :
        isIndex ? indexQuery + (uint64_t(row) * g.indexHeads + h) * dim :
        nullptr;
    for (int d = tid; d < dim; d += 64) {
        float v = values[d];
        if (d < g.rotary) {
            const int k = d % (g.rotary / 2);
            const float angle = FastllmPreciseRopeAngle(float(int(r.positions[token])), k, g.rotary, g.theta);
            const float a = values[k], b = values[k + g.rotary / 2];
            v = d < g.rotary / 2 ? a * cosf(angle) - b * sinf(angle) : a * sinf(angle) + b * cosf(angle);
        }
        if (dst) dst[d] = __float2half_rn(v);
        else {
            r.address.Store(false,r.key,h,r.meta[0]+token,r.keyStride,d,__float2half_rn(v));
            r.address.Store(true,r.value,h,r.meta[0]+token,r.valueStride,d,
                value[(uint64_t(row) * g.keyHeads + h) * dim + d]);
        }
    }
}

// A CTA owns one request's mutable tail. It can commit any number of groups;
// padded verifier/draft rows never update persistent state.
__global__ void Compress(Requests batch, Geometry g, const half *index, const float *norm) {
    const Request &r = batch.items[blockIdx.x];
    const int tid = threadIdx.x, dim = g.indexDim;
    const int valid = r.valid ? min(r.rows, *r.valid) : r.rows;
    __shared__ float values[256], sums[2], scale;
    for (int token = 0; token < r.rows; ++token) {
        const half *raw = index + uint64_t(r.start + token) * (g.indexHeads + 1) * dim + g.indexHeads * dim;
        for (int d = tid; d < dim; d += 64) {
            if (r.capture) r.capture[uint64_t(token) * dim + d] = __half2float(raw[d]);
            if (token < valid) r.tail[uint64_t((r.meta[0] + token) % g.ratio) * dim + d] = __half2float(raw[d]);
        }
        if (!tid && token < valid) r.tailPositions[(r.meta[0] + token) % g.ratio] = r.positions[token];
        __syncthreads();
        if (token >= valid || (r.meta[0] + token + 1) % g.ratio) continue;
        float sum = 0;
        for (int d = tid; d < dim; d += 64) {
            float v = r.tail[d];
            for (int j = 1; j < g.ratio; ++j) v += r.tail[uint64_t(j) * dim + d];
            values[d] = v / g.ratio;
            sum += values[d] * values[d];
        }
        sum = Sum(sum);
        if (!(tid & 31)) sums[tid / 32] = sum;
        __syncthreads();
        if (!tid) scale = rsqrtf((sums[0] + sums[1]) / dim + g.eps);
        __syncthreads();
        for (int d = tid; d < dim; d += 64) values[d] = values[d] * scale * norm[d];
        __syncthreads();
        float *dst = r.blocks + uint64_t((r.meta[0] + token + 1) / g.ratio - 1) * dim;
        for (int d = tid; d < dim; d += 64) {
            float v = values[d];
            if (d < g.rotary) {
                const int k = d % (g.rotary / 2);
                const float angle = FastllmPreciseRopeAngle(float(int(r.tailPositions[0])), k, g.rotary, g.theta);
                const float a = values[k], b = values[k + g.rotary / 2];
                v = d < g.rotary / 2 ? a * cosf(angle) - b * sinf(angle) : a * sinf(angle) + b * cosf(angle);
            }
            dst[d] = v;
        }
        __syncthreads();
    }
}

__global__ void Scores(Requests batch, Geometry g, const half *query, int firstRow) {
    const int row = firstRow + blockIdx.y;
    const Request &r = batch.Find(row);
    const int token = row - r.start, lane = threadIdx.x & 31;
    const int block = blockIdx.x * 8 + threadIdx.x / 32;
    const int blocks = (r.meta[0] + token + 1) / g.ratio;
    if ((r.valid && token >= *r.valid) || block >= blocks || block >= r.capacity) return;
    float score = 0;
    for (int h = 0; h < g.indexHeads; ++h) {
        float dot = 0;
        for (int d = lane; d < g.indexDim; d += 32)
            dot += __half2float(query[(uint64_t(row) * g.indexHeads + h) * g.indexDim + d]) *
                r.blocks[uint64_t(block) * g.indexDim + d];
        score += fmaxf(Sum(dot), 0.f);
    }
    if (!lane) r.scores[uint64_t(token) * r.capacity + block] = score * rsqrtf(float(g.indexDim));
}

__global__ void SelectExpand(Requests batch, Geometry g, int32_t *indices, int firstRow) {
    const int row = firstRow + blockIdx.x;
    const Request &r = batch.Find(row);
    const int token = row - r.start, selected = g.budget / g.ratio;
    const int width = g.budget + g.ratio - 1;
    const int length = r.valid && token >= *r.valid ? 0 : r.meta[0] + token + 1;
    int32_t *blocks = r.selected + uint64_t(token) * selected;
    Qwen4QSARadixSelectRow<0>(r.scores + uint64_t(token) * r.capacity,
        blocks, length / g.ratio, selected);
    __syncthreads();
    const int complete = length / g.ratio, chosen = min(complete, selected) * g.ratio;
    for (int i = threadIdx.x; i < width; i += blockDim.x)
        indices[uint64_t(row) * width + i] = i < chosen ? blocks[i / g.ratio] * g.ratio + i % g.ratio :
            i - chosen < length % g.ratio ? complete * g.ratio + i - chosen : -1;
}

// Strata chunked attention, generalized to independent cache strides, ragged
// rows and head groups. Each KV tile is reused across query heads. The merge
// also applies the existing FastLLM FP16 output/sigmoid rounding boundaries.
template<int D, int H, int Chunk>
__global__ __launch_bounds__(256) void AttentionReference(Requests batch, Geometry g,
        const half *query, const int32_t *indices, float *partials, int firstRow) {
    const int row = firstRow + blockIdx.y;
    const Request &r = batch.Find(row);
    const int groups = g.heads / g.keyHeads, tiles = (groups + H - 1) / H;
    const int kvh = blockIdx.z / tiles, firstHead = (blockIdx.z % tiles) * H;
    const int heads = min(H, groups - firstHead), width = g.budget + g.ratio - 1;
    const int splits = (width + Chunk - 1) / Chunk, chunk = blockIdx.x;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid / 32;
    const int token = row - r.start, length = r.meta[0] + token + 1;
    const bool valid = !r.valid || token < *r.valid;
    __shared__ float q[H][D], probabilities[H][Chunk];
    __shared__ int ids[Chunk];
    __shared__ uint64_t keyOffsets[Chunk], valueOffsets[Chunk];
    for (int i = tid; i < H * D; i += 256)
        q[i / D][i % D] = valid && i / D < heads ? __half2float(query[
            (uint64_t(row) * g.heads + kvh * groups + firstHead + i / D) * D + i % D]) : 0;
    for (int i = tid; i < Chunk; i += 256) {
        const int id = valid && chunk * Chunk + i < width ? indices[uint64_t(row) * width + chunk * Chunk + i] : -1;
        ids[i] = id >= 0 && id < length ? id : -1;
        if (ids[i] >= 0) {
            keyOffsets[i] = r.address.Offset(false,kvh,id,r.keyStride);
            valueOffsets[i] = r.address.Offset(true,kvh,id,r.valueStride);
        }
    }
    __syncthreads();
    for (int c = warp; c < Chunk; c += 8) {
        float k[D / 32];
#pragma unroll
        for (int i = 0; i < D / 32; ++i)
            k[i] = ids[c] < 0 ? 0 : __half2float(r.key[keyOffsets[c] + lane * (D / 32) + i]);
#pragma unroll
        for (int h = 0; h < H; ++h) {
            float dot = 0;
#pragma unroll
            for (int i = 0; i < D / 32; ++i) dot += k[i] * q[h][lane * (D / 32) + i];
            dot = Sum(dot);
            if (!lane) probabilities[h][c] = ids[c] < 0 ? -FLT_MAX : dot * rsqrtf(float(D));
        }
    }
    __syncthreads();
    for (int h = warp; h < heads; h += 8) {
        float maximum = -FLT_MAX;
#pragma unroll
        for (int c = lane; c < Chunk; c += 32) maximum = fmaxf(maximum, probabilities[h][c]);
        maximum = Max(maximum);
        float sum = 0;
#pragma unroll
        for (int c = lane; c < Chunk; c += 32) {
            const float p = ids[c] < 0 ? 0 : __expf(probabilities[h][c] - maximum);
            probabilities[h][c] = p; sum += p;
        }
        sum = Sum(sum);
        if (!lane) {
            float *dst = partials + ((uint64_t(row) * splits + chunk) * g.heads + kvh * groups + firstHead + h) * (D + 2);
            dst[D] = maximum; dst[D + 1] = sum;
        }
    }
    __syncthreads();
    if (tid < D) {
        float acc[H] = {};
        for (int c = 0; c < Chunk; ++c) {
            if (ids[c] < 0) continue;
            const float v = __half2float(r.value[valueOffsets[c] + tid]);
#pragma unroll
            for (int h = 0; h < H; ++h) acc[h] = fmaf(probabilities[h][c], v, acc[h]);
        }
        for (int h = 0; h < heads; ++h)
            partials[((uint64_t(row) * splits + chunk) * g.heads + kvh * groups + firstHead + h) * (D + 2) + tid] = acc[h];
    }
}

// Strata's portable score reduce-scatter and cell-major value folding,
// generalized to this entry's head tiles and 128/256-dimensional FP16 queries.
__device__ __forceinline__ float ReduceHeads(const float (&part)[16], int lane) {
    float a[8], b[4], c[2];
#pragma unroll
    for (int i=0;i<8;++i) {
        const bool high=lane&16;
        a[i]=(high?part[i+8]:part[i])+__shfl_xor_sync(0xffffffffu,high?part[i]:part[i+8],16);
    }
#pragma unroll
    for (int i=0;i<4;++i) {
        const bool high=lane&8;
        b[i]=(high?a[i+4]:a[i])+__shfl_xor_sync(0xffffffffu,high?a[i]:a[i+4],8);
    }
#pragma unroll
    for (int i=0;i<2;++i) {
        const bool high=lane&4;
        c[i]=(high?b[i+2]:b[i])+__shfl_xor_sync(0xffffffffu,high?b[i]:b[i+2],4);
    }
    const bool high=lane&2;
    const float v=(high?c[1]:c[0])+__shfl_xor_sync(0xffffffffu,high?c[0]:c[1],2);
    return v+__shfl_xor_sync(0xffffffffu,v,1);
}
template<int D>
__device__ __forceinline__ void LoadKey(const half *p, float (&out)[D/32]) {
    using Vector=typename std::conditional<D==256,uint4,uint2>::type;
    if (reinterpret_cast<uintptr_t>(p) % sizeof(Vector)) {
#pragma unroll
        for (int j=0;j<D/32;++j) out[j]=__half2float(p[j]);
        return;
    }
    const Vector raw=*reinterpret_cast<const Vector*>(p);
    const half2 *h=reinterpret_cast<const half2*>(&raw);
#pragma unroll
    for (int j=0;j<D/64;++j) {
        const float2 f=__half22float2(h[j]);out[2*j]=f.x;out[2*j+1]=f.y;
    }
}
template<int H>
__device__ __forceinline__ void FoldCell(float (&acc)[H], const float *weights, float value) {
#pragma unroll
    for (int i=0;i<H/4;++i) {
        const float4 p=reinterpret_cast<const float4*>(weights)[i];
        acc[4*i]=fmaf(p.x,value,acc[4*i]);acc[4*i+1]=fmaf(p.y,value,acc[4*i+1]);
        acc[4*i+2]=fmaf(p.z,value,acc[4*i+2]);acc[4*i+3]=fmaf(p.w,value,acc[4*i+3]);
    }
}
template<int D,int H,int Chunk>
__global__ __launch_bounds__(256) void Attention(Requests batch,Geometry g,
        const half *query,const int32_t *indices,float *partials,int firstRow) {
    const int row=firstRow+blockIdx.y;
    const Request &r=batch.Find(row);
    const int groups=g.heads/g.keyHeads,tiles=(groups+H-1)/H;
    const int kvh=blockIdx.z/tiles,firstHead=(blockIdx.z%tiles)*H,heads=min(H,groups-firstHead);
    const int width=g.budget+g.ratio-1,splits=(width+Chunk-1)/Chunk,chunk=blockIdx.x;
    const int tid=threadIdx.x,lane=tid&31,warp=tid/32,token=row-r.start,length=r.meta[0]+token+1;
    const bool valid=!r.valid || token<*r.valid;
    __shared__ __align__(16) float q[H][D];
    __shared__ float scores[H][Chunk];
    __shared__ int ids[Chunk];
    __shared__ uint64_t ko[Chunk],vo[Chunk];
    // Query storage is dead before the cell-major probabilities are written.
    float (*prob)[H]=reinterpret_cast<float (*)[H]>(&q[0][0]);
    static_assert(Chunk<=D && H<=16 && H%4==0,"QSA tile storage");
    for(int i=tid;i<H*D;i+=256)
        q[i/D][i%D]=valid&&i/D<heads ? __half2float(query[(uint64_t(row)*g.heads+kvh*groups+firstHead+i/D)*D+i%D]):0;
    for(int i=tid;i<Chunk;i+=256) {
        const int id=valid&&chunk*Chunk+i<width?indices[uint64_t(row)*width+chunk*Chunk+i]:-1;
        ids[i]=id>=0&&id<length?id:-1;
        if(ids[i]>=0){ko[i]=r.address.Offset(false,kvh,id,r.keyStride);vo[i]=r.address.Offset(true,kvh,id,r.valueStride);}
    }
    __syncthreads();
    // Two cells share each query read. Masked tails use zero keys, with the
    // exact same reduction and explicit invalid-score mask.
    for(int first=warp;first<Chunk;first+=16) {
        float keys[2][D/32],parts[2][16];
#pragma unroll
        for(int j=0;j<2;++j) {
            const int c=first+j*8;
            if(c<Chunk&&ids[c]>=0)LoadKey<D>(r.key+ko[c]+lane*(D/32),keys[j]);
            else {
#pragma unroll
                for(int d=0;d<D/32;++d)keys[j][d]=0;
            }
        }
#pragma unroll
        for(int h=0;h<16;++h) {
            float sum[2]={};
            if(h<H) {
#pragma unroll
                for(int v=0;v<D/128;++v) {
                    const float4 a=*reinterpret_cast<const float4*>(&q[h][lane*(D/32)+v*4]);
#pragma unroll
                    for(int j=0;j<2;++j) {
                        sum[j]+=keys[j][v*4]*a.x+keys[j][v*4+1]*a.y+keys[j][v*4+2]*a.z+keys[j][v*4+3]*a.w;
                    }
                }
            }
            parts[0][h]=sum[0];parts[1][h]=sum[1];
        }
#pragma unroll
        for(int j=0;j<2;++j) {
            const int c=first+j*8,h=lane/2;
            const float score=ReduceHeads(parts[j],lane);
            if(!(lane&1)&&h<H&&c<Chunk)scores[h][c]=ids[c]<0?-FLT_MAX:score*rsqrtf(float(D));
        }
    }
    __syncthreads();
    for(int h=warp;h<H;h+=8) {
        float maximum=-FLT_MAX;
#pragma unroll
        for(int c=lane;c<Chunk;c+=32)maximum=fmaxf(maximum,scores[h][c]);
        maximum=Max(maximum);
        float sum=0;
#pragma unroll
        for(int c=lane;c<Chunk;c+=32) {
            const float p=ids[c]<0||h>=heads?0:__expf(scores[h][c]-maximum);
            prob[c][h]=p;sum+=p;
        }
        sum=Sum(sum);
        if(!lane&&h<heads) {
            float *dst=partials+((uint64_t(row)*splits+chunk)*g.heads+kvh*groups+firstHead+h)*(D+2);
            dst[D]=maximum;dst[D+1]=sum;
        }
    }
    __syncthreads();
    if(tid<D) {
        float acc[H]={};
        for(int first=0;first<Chunk;first+=4) {
            float values[4];
#pragma unroll
            for(int j=0;j<4;++j)values[j]=ids[first+j]<0?0:__half2float(r.value[vo[first+j]+tid]);
#pragma unroll
            for(int j=0;j<4;++j)FoldCell<H>(acc,prob[first+j],values[j]);
        }
        for(int h=0;h<heads;++h)partials[((uint64_t(row)*splits+chunk)*g.heads+kvh*groups+firstHead+h)*(D+2)+tid]=acc[h];
    }
}

template<int D, int Chunk>
__global__ void MergeGate(Geometry g, const float *partials, const half *qgate, half *output, int firstRow) {
    const int row = firstRow + blockIdx.y, head = blockIdx.x, d = threadIdx.x;
    const int splits = (g.budget + g.ratio - 2 + Chunk) / Chunk;
    const uint64_t stride = uint64_t(g.heads) * (D + 2);
    const float *src = partials + (uint64_t(row) * splits * g.heads + head) * (D + 2);
    float maximum = -FLT_MAX;
    for (int c = 0; c < splits; ++c) maximum = fmaxf(maximum, src[c * stride + D]);
    float sum = 0, value = 0;
    for (int first = 0; first < splits; first += 8) {
        float m[8], l[8], a[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const bool valid = first + i < splits;
            m[i] = valid ? src[(first + i) * stride + D] : -FLT_MAX;
            l[i] = valid ? src[(first + i) * stride + D + 1] : 0;
            a[i] = valid ? src[(first + i) * stride + d] : 0;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float w = __expf(m[i] - maximum);
            if (m[i] != -FLT_MAX) { sum = fmaf(l[i], w, sum); value = fmaf(a[i], w, value); }
        }
    }
    const half context = __float2half_rn(sum > 0 ? value / sum : 0);
    const half gate = qgate[(uint64_t(row) * g.heads + head) * 2 * D + D + d];
    output[(uint64_t(row) * g.heads + head) * D + d] = Qwen4AttentionGate(context, gate);
}

struct AttentionCase {
    const std::vector<Requests> *packs;
    Geometry g;
    const half *query, *gate;
    const int32_t *indices;
    float *scratch;
    half *output;
};
template<int D, int H, int Chunk, bool Reference = false>
void LaunchAttention(const AttentionCase &c) {
    const int splits = (c.g.budget + c.g.ratio - 2 + Chunk) / Chunk;
    const int tiles = (c.g.heads / c.g.keyHeads + H - 1) / H;
    for (const auto &pack : *c.packs) {
        for (int first = pack.first; first < pack.first + pack.rows;) {
            const int count = std::min(65535, pack.first + pack.rows - first);
            if constexpr (Reference) AttentionReference<D,H,Chunk><<<dim3(splits,count,c.g.keyHeads * tiles),256,0,cudaStreamPerThread>>>(
                pack,c.g,c.query,c.indices,c.scratch,first);
            else Attention<D,H,Chunk><<<dim3(splits,count,c.g.keyHeads * tiles),256,0,cudaStreamPerThread>>>(
                pack,c.g,c.query,c.indices,c.scratch,first);
            MergeGate<D,Chunk><<<dim3(c.g.heads,count),D,0,cudaStreamPerThread>>>(
                c.g,c.scratch,c.gate,c.output,first);
            first += count;
        }
    }
}
bool RunAttention(void *opaque, int impl) {
    const auto &c = *static_cast<AttentionCase*>(opaque);
    auto launch = [&](auto dimension, auto heads, auto chunk) {
        if (impl==2) LaunchAttention<decltype(dimension)::value,decltype(heads)::value,decltype(chunk)::value,true>(c);
        else LaunchAttention<decltype(dimension)::value,decltype(heads)::value,decltype(chunk)::value>(c);
    };
    auto geometry = [&](auto chunk) {
        if (c.g.dim == 256) {
            if (c.g.heads / c.g.keyHeads == 12) launch(std::integral_constant<int,256>{},std::integral_constant<int,12>{},chunk);
            else launch(std::integral_constant<int,256>{},std::integral_constant<int,16>{},chunk);
        } else {
            if (c.g.heads / c.g.keyHeads == 12) launch(std::integral_constant<int,128>{},std::integral_constant<int,12>{},chunk);
            else launch(std::integral_constant<int,128>{},std::integral_constant<int,16>{},chunk);
        }
    };
    if (impl!=1) geometry(std::integral_constant<int,64>{});
    else geometry(std::integral_constant<int,128>{});
    return cudaPeekAtLastError() == cudaSuccess;
}
const fastllm::kernels::Family family{15,"attention.qsa.requests.fp16","2",{
    {"cuda.qsa.requests.chunk64",json11::Json::object{{"backend","cuda"},{"source","Strata MIT, dbb2d168"},{"architecture","portable CUDA SIMT"},{"head_dim","128 or 256"},{"rows","arbitrary ragged requests"},{"fusion","indexed QK/softmax/PV; merge/output gate"},{"chunk",64}}},
    {"cuda.qsa.requests.chunk128",json11::Json::object{{"backend","cuda"},{"source","Strata MIT, dbb2d168"},{"architecture","portable CUDA SIMT"},{"head_dim","128 or 256"},{"rows","arbitrary ragged requests"},{"fusion","indexed QK/softmax/PV; merge/output gate"},{"chunk",128}}}
,
    {"cuda.qsa.requests.reference64",json11::Json::object{{"backend","cuda"},{"architecture","portable CUDA SIMT"},{"rows","arbitrary ragged requests"},{"source","basic chunked reference"}}}
}};
const bool registered = [] {fastllm::kernels::Register(&family);return true;}();
}

bool FastllmCudaQwen4QSABatchGraph(
        const std::vector<FastllmCudaQwen4QSARequest> &requests,
        const fastllm::Data &queryGate, const fastllm::Data &key, const fastllm::Data &value,
        const fastllm::Data &indexProjection, const fastllm::Data &queryNorm,
        const fastllm::Data &keyNorm, const fastllm::Data &indexQueryNorm,
        const fastllm::Data &indexKeyNorm, fastllm::Data &output,
        int heads, int keyHeads, int headDim, int indexHeads, int indexDim,
        int ratio, int budget, int rotaryDim, float theta, float eps) {
    using namespace fastllm;
    using namespace fastllm_qsa_batch;
    if (requests.empty() || heads <= 0 || keyHeads <= 0 || heads % keyHeads ||
        (headDim != 128 && headDim != 256) || indexHeads <= 0 || indexHeads > 32 ||
        indexDim <= 0 || indexDim > 256 || indexDim % 32 || ratio <= 0 || budget <= 0 ||
        budget % ratio || int64_t(budget) + ratio > INT_MAX - 128 || rotaryDim <= 0 || rotaryDim % 2 ||
        heads > INT_MAX / headDim / 2 || int64_t(keyHeads) * ((heads / keyHeads + 15) / 16) > 65535 ||
        rotaryDim > std::min(headDim,indexDim) || !std::isfinite(theta) || !std::isfinite(eps) ||
        !(theta > 0) || !(eps > 0)) return false;
    const int device = FastllmCudaGetDevice();
    auto onDevice = [&](const Data &d, DataType type) {
        return d.dataDevice == CUDA && d.cudaData && d.dataType == type && !d.multiDeviceData &&
            (d.dataDeviceIds.empty() || d.dataDeviceIds == std::vector<int>{device});
    };
    auto dense = [](const Data &d) {
        uint64_t stride = 1;
        if (d.dims.empty() || d.strides.size() != d.dims.size()) return false;
        for (int i = int(d.dims.size()) - 1; i >= 0; --i) {
            if (d.dims[i] <= 0 || (d.dims[i] > 1 && d.strides[i] != stride)) return false;
            stride *= d.dims[i];
        }
        return true;
    };
    for (const Data *d : {&queryGate,&key,&value,&indexProjection})
        if (!onDevice(*d,FLOAT16) || !dense(*d) || &output == d) return false;
    for (const Data *d : {&queryNorm,&keyNorm,&indexQueryNorm,&indexKeyNorm})
        if (!onDevice(*d,FLOAT32) || !dense(*d)) return false;
    if (queryNorm.Count(0) != headDim || keyNorm.Count(0) != headDim ||
        indexQueryNorm.Count(0) != indexDim || indexKeyNorm.Count(0) != indexDim) return false;
    std::vector<Requests> packs;
    int rows = 0, capacity = 0, cacheLayouts = 0, maxPageLength = 0, maxRequestRows = 0;
    std::set<const void*> owners;
    for (const auto &r : requests) {
        if (!r.key || !r.value || !r.tailKeys || !r.tailPositions || !r.blocks || !r.positions ||
            !r.meta || !r.scores || !r.selected || r.rows <= 0 || r.rows > INT_MAX - rows) return false;
        for (Data *d : {r.key,r.value})
            if (!onDevice(*d,FLOAT16) || Qwen4GetKvMem(*d) ||
                d->dims.size() != 3 || d->dims[0] != keyHeads || d->dims[2] != headDim ||
                d->strides.size() != 3 || d->strides[1] != headDim || d->strides[2] != 1 ||
                d->dims[1] < 0 || d->strides[0] < uint64_t(d->dims[1]) * headDim ||
                !owners.insert(d->cudaData).second || &output == d) return false;
        if (r.key->isPagedKVCache != r.value->isPagedKVCache) return false;
        if (r.key->isPagedKVCache && (!r.key->pagedKVCacheData || !r.value->pagedKVCacheData ||
            r.key->pageLen <= 0 || r.key->pageLen != r.value->pageLen ||
            !onDevice(*r.key->pagedKVCacheData,FLOAT16) || !onDevice(*r.value->pagedKVCacheData,FLOAT16))) return false;
        for (const Data *d : std::vector<const Data*>{r.tailKeys,r.tailPositions,r.blocks,r.positions,r.scores})
            if (!onDevice(*d,FLOAT32)) return false;
        for (const Data *d : {r.tailKeys,r.tailPositions,r.blocks,r.scores,r.selected})
            if (!owners.insert(d->cudaData).second || &output == d) return false;
        const auto &tailCapacity = r.tailKeys->expansionDims.empty() ? r.tailKeys->dims : r.tailKeys->expansionDims;
        const auto &positionCapacity = r.tailPositions->expansionDims.empty() ? r.tailPositions->dims : r.tailPositions->expansionDims;
        if (!onDevice(*r.meta,INT32) || r.meta->Count(0) < 1 || !onDevice(*r.selected,INT32) ||
            !dense(*r.positions) || r.positions->Count(0) != r.rows ||
            !dense(*r.scores) || !dense(*r.selected) ||
            tailCapacity.size() != 2 || tailCapacity[0] < ratio || tailCapacity[1] != indexDim ||
            positionCapacity.size() != 1 || positionCapacity[0] < ratio ||
            r.blocks->dims.size() != 2 || r.blocks->dims[1] != indexDim ||
            r.tailKeys->strides != std::vector<uint64_t>{uint64_t(indexDim),1} ||
            r.blocks->strides != std::vector<uint64_t>{uint64_t(indexDim),1}) return false;
        const int cap = r.blocks->expansionDims.size() == 2 ? r.blocks->expansionDims[0] : r.blocks->dims[0];
        if (cap < budget / ratio || r.scores->dims != std::vector<int>{r.rows,cap} ||
            r.selected->dims != std::vector<int>{r.rows,budget/ratio} ||
            (r.rawKeyCapture && (!onDevice(*r.rawKeyCapture,FLOAT32) ||
                !dense(*r.rawKeyCapture) || r.rawKeyCapture->Count(0) != uint64_t(r.rows)*indexDim ||
                !owners.insert(r.rawKeyCapture->cudaData).second || &output == r.rawKeyCapture))) return false;
        if (packs.empty() || packs.back().count == Requests::capacity) {
            packs.emplace_back(Requests{});packs.back().first = rows;
        }
        auto &p = packs.back();
        p.items[p.count++] = {(half*)Qwen4CachePayload(*r.key),(half*)Qwen4CachePayload(*r.value),
            (float*)r.tailKeys->cudaData,(float*)r.tailPositions->cudaData,(float*)r.blocks->cudaData,
            (const float*)r.positions->cudaData,(const int32_t*)r.meta->cudaData,
            r.meta->Count(0)>1 ? (const int32_t*)r.meta->cudaData+1 : nullptr,
            (float*)r.scores->cudaData,(int32_t*)r.selected->cudaData,
            r.rawKeyCapture ? (float*)r.rawKeyCapture->cudaData : nullptr,
            r.key->strides[0],r.value->strides[0],rows,r.rows,cap,Qwen4GetCacheAddress(*r.key,*r.value)};
        p.rows += r.rows;rows += r.rows;capacity = std::max(capacity,cap);
        cacheLayouts |= r.key->isPagedKVCache ? 2 : 1;
        maxPageLength = std::max(maxPageLength,r.key->isPagedKVCache ? r.key->pageLen : 0);
        maxRequestRows = std::max(maxRequestRows,r.rows);
    }
    if (queryGate.Count(0) != uint64_t(rows)*heads*headDim*2 ||
        key.Count(0) != uint64_t(rows)*keyHeads*headDim || value.Count(0) != key.Count(0) ||
        indexProjection.Count(0) != uint64_t(rows)*(indexHeads+1)*indexDim) return false;
    const int width = budget + ratio - 1, splits = (width + 63) / 64;
    Data query(FLOAT16,{rows,heads,headDim}), indexQuery(FLOAT16,{rows,indexHeads,indexDim}),
        indices(INT32,{rows,width}), scratch(FLOAT32,{rows,splits,heads,headDim+2});
    output.dataType = FLOAT16;output.UpdateUnitSize();output.Resize({1,rows,heads*headDim});
    for (Data *d : {&query,&indexQuery,&indices,&scratch,&output}) {
        d->ToDevice(CUDA,std::vector<int>{device},false);d->Allocate(false);
    }
    Geometry g{heads,keyHeads,headDim,indexHeads,indexDim,ratio,budget,rotaryDim,theta,eps};
    const int region = FastllmCudaGraphBeginIndependentRegion();
    for (const auto &p : packs) {
        Compress<<<p.count,64,0,cudaStreamPerThread>>>(p,g,(const half*)indexProjection.cudaData,(const float*)indexKeyNorm.cudaData);
        for (int first=p.first;first<p.first+p.rows;) {
            const int count=std::min(65535,p.first+p.rows-first);
            Prepare<true><<<dim3(indexHeads,count),64,0,cudaStreamPerThread>>>(p,g,
                nullptr,nullptr,nullptr,(const half*)indexProjection.cudaData,nullptr,nullptr,
                (const float*)indexQueryNorm.cudaData,nullptr,(half*)indexQuery.cudaData,first);
            Scores<<<dim3((capacity+7)/8,count),256,0,cudaStreamPerThread>>>(p,g,(const half*)indexQuery.cudaData,first);
            SelectExpand<<<count,256,0,cudaStreamPerThread>>>(p,g,(int32_t*)indices.cudaData,first);
            first+=count;
        }
    }
    if (region) {FastllmCudaGraphMarkParallelFirstDone(region);FastllmCudaGraphMarkParallelSecondBegin(region);}
    for (const auto &p : packs) {
        for (int first=p.first;first<p.first+p.rows;) {
            const int count=std::min(65535,p.first+p.rows-first);
            Prepare<false><<<dim3(heads+keyHeads,count),64,0,cudaStreamPerThread>>>(p,g,
                (const half*)queryGate.cudaData,(const half*)key.cudaData,(const half*)value.cudaData,
                nullptr,(const float*)queryNorm.cudaData,(const float*)keyNorm.cudaData,nullptr,
                (half*)query.cudaData,nullptr,first);
            first+=count;
        }
    }
    if (region) FastllmCudaGraphMarkParallelJoin(region);
    // Only stateless attention is benchmarked: mutable KV/tails above execute
    // once, while every candidate reads the same frozen state and selection.
    AttentionCase operation{&packs,g,(const half*)query.cudaData,(const half*)queryGate.cudaData,
        (const int32_t*)indices.cudaData,(float*)scratch.cudaData,(half*)output.cudaData};
    int selected = 0;
    if (kernels::Enabled()) {
        kernels::Signature signature{{15,FastllmCudaRuntimeArch(),device,rows,heads,keyHeads,headDim,width,
            int64_t(requests.size()),cacheLayouts,maxPageLength,maxRequestRows}};
        kernels::CudaOutputsCase trial{{&operation,RunAttention,[](void*,int){return std::string{};},nullptr,0,false},
            {{output.cudaData,output.Count(0),FLOAT16}}};
        selected = kernels::SelectCudaOutputs(family,signature,trial);
    }
    AssertInFastLLM(RunAttention(&operation,std::max(0,selected)) && cudaGetLastError() == cudaSuccess,
        "Qwen packed QSA graph launch failed");
    DeviceSync();
    return true;
}
