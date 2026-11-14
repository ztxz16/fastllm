#pragma once

// Compact NUMA block-32 E2M1/UE8M0 weights. Cache copies restore the original
// gate/up halves. The caller quantizes the BF16 input to FP8 in groups of 32.
// All routes include their score before down-input quantization and round the
// down projection to BF16 before the ordered FP32 expert sum.
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "../fastllm-bfloat16-silu.cuh"

namespace fastllm {
namespace cuda {
namespace dsv41_cache {
// Stable prefix of nonresident routes, matching the CPU admission order.
__device__ inline int NativeStagedSlot(unsigned resident, int route, int selected) {
    if (resident & (1u << route)) return -1;
    const int slot = route - __popc(resident & ((1u << route) - 1));
    return slot < selected ? slot : -1;
}
__device__ inline float FP4(unsigned value, unsigned exponent) {
    // Normal E2M1 magnitudes 2..7 map to consecutive FP32 bit
    // patterns spaced by 1 << 22. Handle zero and the .5 subnormal
    // explicitly, preserving signed zero and the UE8M0 edge scales.
    const unsigned magnitude = value & 7;
    const unsigned bits = magnitude >= 2 ? (magnitude + 252) << 22 : magnitude * 0x3f000000u;
    const float decoded = __uint_as_float(bits | ((value & 8) << 28));
    const float scale = exponent == 0     ? __int_as_float(0x00400000)
                        : exponent == 255 ? __int_as_float(0x7fc00000)
                                          : __int_as_float(exponent << 23);
    return __bfloat162float(__float2bfloat16_rn(__fmul_rn(decoded, scale)));
}
__device__ inline float BFloat(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}
// Keep the scale edge cases and signed zero exact without divergent branches.
__device__ __forceinline__ unsigned SelectBits(unsigned a, unsigned b, bool condition) {
    unsigned result;
    asm("{ .reg .pred p; setp.ne.u32 p,%3,0; selp.u32 %0,%1,%2,p; }"
        : "=r"(result) : "r"(a), "r"(b), "r"((unsigned)condition));
    return result;
}
__device__ __forceinline__ float2 FP4Pair(unsigned packed, unsigned exponent) {
#if defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && \
    __CUDA_ARCH_FAMILY_SPECIFIC__ >= 1200 && __CUDA_ARCH_FAMILY_SPECIFIC__ < 1300
    // As in FlashInfer sparse_mla_sm120, unpack and scale a pair in BF16.
    // E2M1 values are exact; power-of-two scaling preserves the old FP32
    // product followed by BF16 rounding, including UE8M0 edge encodings.
    unsigned halves, scaleBits;
    asm("{ .reg .b8 lo, hi; mov.b16 {lo,hi}, %1; cvt.rn.f16x2.e2m1x2 %0, lo; }"
        : "=r"(halves) : "h"((unsigned short)packed));
    asm("cvt.rn.bf16x2.ue8m0x2 %0,%1;" : "=r"(scaleBits)
        : "h"((unsigned short)(exponent * 0x101u)));
    const __nv_bfloat162 values = __float22bfloat162_rn(
        __half22float2(*reinterpret_cast<const __half2 *>(&halves)));
    const __nv_bfloat162 scale = *reinterpret_cast<const __nv_bfloat162 *>(&scaleBits);
    return __bfloat1622float2(__hmul2(values, scale));
#else
    unsigned bits = SelectBits(exponent << 23, 0x00400000u, exponent != 0);
    bits = SelectBits(bits, 0x7fc00000u, exponent != 255);
    const float scale = __uint_as_float(bits);
    const unsigned lut = __byte_perm(0x3e3c3800u, 0x46444240u, packed & 0x77u);
    unsigned halves = __byte_perm(lut, 0u, 0x1404u);
    halves |= ((packed & 8u) << 12) | ((packed & 128u) << 24);
    const float2 values = __half22float2(*reinterpret_cast<const __half2 *>(&halves));
    return make_float2(BFloat(__fmul_rn(values.x,scale)), BFloat(__fmul_rn(values.y,scale)));
#endif
}
template<bool Planar = false>
__device__ inline float Dot(const __nv_bfloat16 *input, const uint8_t *weight, int columns,
                            const uint8_t *scales = nullptr) {
    const int lane = threadIdx.x & 15;
    float sum = 0;
    // Independent loads/decodes overlap; additions retain the original order.
    #pragma unroll 8
    for (int block = 0; block < columns / 32; ++block) {
        const uint8_t *w = weight + block * (Planar ? 16 : 17);
        const float2 value = FP4Pair(w[lane], Planar ? scales[block] : w[16]);
        // Each adjacent BF16 input pair is aligned, including every block.
        const unsigned x = reinterpret_cast<const unsigned *>(input)[block * 16 + lane];
        const float x0 = __uint_as_float(x << 16), x1 = __uint_as_float(x & 0xffff0000u);
        const float pair = __fadd_rn(__fmul_rn(x0, value.x), __fmul_rn(x1, value.y));
        sum = __fadd_rn(sum, pair);
    }
    for (int delta = 8; delta; delta >>= 1)
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, delta, 16));
    return sum;
}
// Reuse each input pair for both projections without changing either dot's
// accumulation order. Dot remains the independent single-projection path.
template<bool Planar = false>
__device__ inline float2 DotGateUp(const __nv_bfloat16 *input, const uint8_t *gate,
        const uint8_t *up, int columns, const uint8_t *gateScales = nullptr, const uint8_t *upScales = nullptr) {
    const int lane = threadIdx.x & 15;
    float gs = 0, us = 0;
    #pragma unroll 4
    for (int block = 0; block < columns / 32; ++block) {
        const uint8_t *g = gate + block * (Planar ? 16 : 17), *u = up + block * (Planar ? 16 : 17);
        const float2 gv = FP4Pair(g[lane], Planar ? gateScales[block] : g[16]);
        const float2 uv = FP4Pair(u[lane], Planar ? upScales[block] : u[16]);
        const unsigned x = reinterpret_cast<const unsigned *>(input)[block * 16 + lane];
        const float x0 = __uint_as_float(x << 16), x1 = __uint_as_float(x & 0xffff0000u);
        gs = __fadd_rn(gs, __fadd_rn(__fmul_rn(x0, gv.x), __fmul_rn(x1, gv.y)));
        us = __fadd_rn(us, __fadd_rn(__fmul_rn(x0, uv.x), __fmul_rn(x1, uv.y)));
    }
    for (int delta = 8; delta; delta >>= 1) {
        gs = __fadd_rn(gs, __shfl_down_sync(0xffffffff, gs, delta, 16));
        us = __fadd_rn(us, __shfl_down_sync(0xffffffff, us, delta, 16));
    }
    return make_float2(gs, us);
}
// Shared activation epilogue for ordinary and tiled single-token projections.
__device__ inline void StoreGate(float2 pair, float score, __nv_bfloat16 *activation,
        int route, int inter, float limit, const float *siluTable) {
    float gate = BFloat(pair.x), up = BFloat(pair.y);
    __shared__ float values[32];
    if ((threadIdx.x & 15) == 0) {
        if (limit > 0) {
            gate = fminf(gate, limit);
            up = fmaxf(-limit, fminf(up, limit));
        }
        float value = __fmul_rn(FastllmCudaBFloat16Silu(gate, siluTable), up);
        values[threadIdx.x / 16] = BFloat(__fmul_rn(value, score));
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float value = values[threadIdx.x];
        float amax = fmaxf(1e-4f, fabsf(value));
        for (int delta = 16; delta; delta >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, delta));
        const unsigned bits = __float_as_uint(amax / 448.0f);
        const int exponent = int((bits >> 23) & 255) - 127 + ((bits & 0x7fffff) != 0);
        const float scale = exp2f(float(exponent));
        const float q = fmaxf(-448.0f, fminf(448.0f, value / scale));
        activation[route * inter + blockIdx.x * 32 + threadIdx.x] =
            __float2bfloat16_rn(float(__nv_fp8_e4m3(q)) * scale);
    }
}
template <bool Batched = false, bool Disk = false>
__global__ void Gate(const __nv_bfloat16 *input, const int32_t *slots, const uint8_t *records, const float *scores,
                     __nv_bfloat16 *activation, int hidden, int inter, size_t stride, float limit, int topk = 1,
                     const int32_t *routeMap = nullptr, bool planar = false,
                     const uint8_t *const *recordPointers = nullptr, const float *siluTable = nullptr) {
    const int selected = blockIdx.y, route = routeMap ? routeMap[selected] : selected;
    const int slot = slots[selected];
    if constexpr (Batched)
        input += (size_t)(route / topk) * hidden;
    if (slot < 0)
        return;
    const int column = blockIdx.x * 32 + threadIdx.x / 16;
    const size_t pitch = size_t(hidden / 32) * 17;
    const uint8_t *record = Disk ? recordPointers[slot] : records + size_t(slot) * stride;
    const size_t qpitch = hidden / 2, spitch = hidden / 32;
    const uint8_t *scales = record + size_t(2 * inter) * qpitch;
    const float2 pair = planar ? DotGateUp<true>(input, record + column * qpitch,
        record + (column + inter) * qpitch, hidden, scales + column * spitch,
        scales + (column + inter) * spitch) :
        DotGateUp(input, record + (Disk ? 2*column : column) * pitch,
            record + (Disk ? 2*column+1 : column+inter) * pitch, hidden);
    StoreGate(pair, scores[route], activation, route, inter, limit, siluTable);
}
// Coalesced weight staging avoids repeatedly fetching small fragments of
// block-32 records from VRAM. Two stages overlap the next tile's copies with
// the current tile's arithmetic. FP4 decode, accumulation and shuffles are
// identical to DotGateUp. Call with 512 threads and columns divisible by 256
// so every record row and partial tile stay aligned to the 8-byte copies.
// The end of each iteration needs no extra barrier:
// the next iteration's wait + CTA barrier retires reads before buffer reuse.
template <bool Planar = false>
__device__ inline float2 DotGateUpTiled(const __nv_bfloat16 *input,
        const uint8_t *gate, const uint8_t *up, int columns,
        const uint8_t *gateScale = nullptr, const uint8_t *upScale = nullptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int tileBlocks = 16, tileBytes = tileBlocks * 17, rows = 32;
    const int lane = threadIdx.x & 15, row = threadIdx.x / 16;
    const int blocks = columns / 32, pitch = blocks * (Planar ? 16 : 17);
    // The planar value tile uses the same padded stride to avoid shared-memory
    // bank conflicts. Scale copies stay 8-byte aligned for 256-column tails.
    __shared__ __align__(16) uint8_t tiles[2][2][rows][tileBytes];
    __shared__ __align__(16) uint8_t scales[2][2][rows][Planar ? tileBlocks : 1];
    auto prefetch = [&](int tile) {
        if constexpr (Planar) {
            for (int item = threadIdx.x; item < rows * tileBlocks; item += 512) {
                const int r = item / tileBlocks, offset = item % tileBlocks * 16;
                const int column = tile * tileBlocks * 16 + offset;
                const int valid = column < pitch ? 16 : 0;
                const size_t source = size_t(r) * pitch + column;
                const unsigned g = __cvta_generic_to_shared(&tiles[tile & 1][0][r][offset]);
                const unsigned u = __cvta_generic_to_shared(&tiles[tile & 1][1][r][offset]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::
                    "r"(g), "l"(gate + source), "r"(valid) : "memory");
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;" ::
                    "r"(u), "l"(up + source), "r"(valid) : "memory");
            }
            for (int item = threadIdx.x; item < rows * (tileBlocks / 8); item += 512) {
                const int r = item / (tileBlocks / 8), offset = item % (tileBlocks / 8) * 8;
                const int column = tile * tileBlocks + offset;
                const int valid = column < blocks ? 8 : 0;
                const size_t source = size_t(r) * blocks + column;
                const unsigned g = __cvta_generic_to_shared(&scales[tile & 1][0][r][offset]);
                const unsigned u = __cvta_generic_to_shared(&scales[tile & 1][1][r][offset]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::
                    "r"(g), "l"(gateScale + source), "r"(valid) : "memory");
                asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::
                    "r"(u), "l"(upScale + source), "r"(valid) : "memory");
            }
        } else {
            for (int item = threadIdx.x; item < rows * (tileBytes / 8); item += 512) {
                const int r = item / (tileBytes / 8), offset = item % (tileBytes / 8) * 8;
                const int columnOffset = tile * tileBytes + offset;
                const int validBytes = columnOffset < pitch ? 8 : 0;
                const size_t source = size_t(r) * pitch + columnOffset;
                const unsigned g = __cvta_generic_to_shared(&tiles[tile & 1][0][r][offset]);
                const unsigned u = __cvta_generic_to_shared(&tiles[tile & 1][1][r][offset]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::
                    "r"(g), "l"(gate + source), "r"(validBytes) : "memory");
                asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" ::
                    "r"(u), "l"(up + source), "r"(validBytes) : "memory");
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    float gs = 0.0f, us = 0.0f;
    prefetch(0);
    for (int tile = 0; tile < (blocks + tileBlocks - 1) / tileBlocks; ++tile) {
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        if ((tile + 1) * tileBlocks < blocks) prefetch(tile + 1);
#pragma unroll
        for (int b = 0; b < tileBlocks; ++b) if (tile * tileBlocks + b < blocks) {
            const unsigned x = reinterpret_cast<const unsigned *>(input)[(tile * tileBlocks + b) * 16 + lane];
            const float x0 = __uint_as_float(x << 16), x1 = __uint_as_float(x & 0xffff0000u);
            const uint8_t *g = &tiles[tile & 1][0][row][b * (Planar ? 16 : 17)];
            const uint8_t *u = &tiles[tile & 1][1][row][b * (Planar ? 16 : 17)];
            const float2 gv = FP4Pair(g[lane], Planar ? scales[tile & 1][0][row][b] : g[16]);
            const float2 uv = FP4Pair(u[lane], Planar ? scales[tile & 1][1][row][b] : u[16]);
            gs = __fadd_rn(gs, __fadd_rn(__fmul_rn(x0, gv.x), __fmul_rn(x1, gv.y)));
            us = __fadd_rn(us, __fadd_rn(__fmul_rn(x0, uv.x), __fmul_rn(x1, uv.y)));
        }
    }
    for (int delta = 8; delta; delta >>= 1) {
        gs = __fadd_rn(gs, __shfl_down_sync(0xffffffff, gs, delta, 16));
        us = __fadd_rn(us, __shfl_down_sync(0xffffffff, us, delta, 16));
    }
    return make_float2(gs, us);
#else
    const size_t pitch = size_t(columns / 32) * (Planar ? 16 : 17);
    const int row = threadIdx.x / 16;
    if constexpr (Planar)
        return DotGateUp<true>(input, gate + row * pitch, up + row * pitch, columns,
            gateScale + size_t(row) * (columns / 32), upScale + size_t(row) * (columns / 32));
    else
        return DotGateUp(input, gate + row * pitch, up + row * pitch, columns);
#endif
}

// Resident experts rotate through a cache much larger than L2. Staged experts
// keep the ordinary path because their freshly copied weights can be warm in L2.
template <bool Planar = false>
static __global__ void GateResident(const __nv_bfloat16 *input, const int32_t *slots,
        const uint8_t *records, const float *scores, __nv_bfloat16 *activation,
        int hidden, int inter, size_t stride, float limit, const float *siluTable = nullptr) {
    const int route = blockIdx.y, slot = slots[route];
    if (slot < 0) return;
    const size_t pitch = size_t(hidden / 32) * (Planar ? 16 : 17);
    const uint8_t *record = records + size_t(slot) * stride;
    const uint8_t *gate = record + size_t(blockIdx.x * 32) * pitch;
    const uint8_t *up = gate + size_t(inter) * pitch;
    const uint8_t *scale = Planar ? record + size_t(2 * inter) * pitch +
        size_t(blockIdx.x * 32) * (hidden / 32) : nullptr;
    const float2 pair = DotGateUpTiled<Planar>(input, gate, up, hidden, scale,
        Planar ? scale + size_t(inter) * (hidden / 32) : nullptr);
    StoreGate(pair, scores[route], activation, route, inter, limit, siluTable);
}
template<bool Indirect = false>
static __global__ void Down(const __nv_bfloat16 *activation, const int32_t *slots, const uint8_t *records,
                            float *output, int hidden, int inter, size_t stride, size_t downOffset,
                            const int32_t *routeMap = nullptr, bool planar = false,
                            const uint8_t *const *recordPointers = nullptr) {
    const int selected = blockIdx.y, route = routeMap ? routeMap[selected] : selected;
    const int slot = slots[selected];
    if (slot < 0)
        return;
    const int column = blockIdx.x * 8 + threadIdx.x / 16;
    const uint8_t *matrix = (Indirect ? recordPointers[slot] : records + size_t(slot) * stride) + downOffset;
    const uint8_t *weight = matrix + size_t(column) * (inter / 32) * 17;
    const float value = planar ? Dot<true>(activation + route * inter,
        matrix + size_t(column) * (inter / 2), inter,
        matrix + size_t(hidden) * (inter / 2) + size_t(column) * (inter / 32)) :
        Dot(activation + route * inter, weight, inter);
    if ((threadIdx.x & 15) == 0)
        output[route * hidden + column] = BFloat(value);
}
// Reuse a decoded expert row across a bounded register tile of routes. Tile
// boundaries are independent of request count; partial and nonrepeated groups
// retain the same accumulation and BF16/FP8 rounding as the scalar route path.
constexpr int kExpertRouteTile = 4;
constexpr int kExpertMmaRouteTile = 16;
template<int Capacity = kExpertRouteTile>
__device__ inline int ExpertRouteTile(const int32_t *slots, int selected, int routes) {
    const int slot = slots[selected];
    if (slot < 0) return 0;
    int first = selected;
    while (first % Capacity && slots[first - 1] == slot) --first;
    int count = 1;
    while (count < Capacity - first % Capacity && first + count < routes &&
           slots[first + count] == slot) ++count;
    // A pair is already served efficiently from L2 by the scalar CTAs.
    // Register reuse pays off only when it removes more repeated decodes.
    if (count <= 2) return 1;
    return first == selected ? count : 0;
}
template<bool SmallOnly = false>
__global__ void GateReuse(const __nv_bfloat16 *input, const int32_t *slots,
        const uint8_t *records, const float *scores, __nv_bfloat16 *activation,
        int hidden, int inter, size_t stride, float limit, int topk,
        const int32_t *routeMap, int routes, bool planar, const float *siluTable = nullptr) {
    const int selected = blockIdx.y;
    if constexpr (SmallOnly) if (ExpertRouteTile<kExpertMmaRouteTile>(slots,selected,routes) != 1) return;
    const int count = ExpertRouteTile(slots, selected, routes);
    if (!count) return;
    const int lane = threadIdx.x & 15, column = blockIdx.x * 32 + threadIdx.x / 16;
    const uint8_t *record = records + size_t(slots[selected]) * stride;
    const int blocks = hidden / 32, pitch = blocks * (planar ? 16 : 17);
    const uint8_t *g = record + size_t(column) * pitch, *u = g + size_t(inter) * pitch;
    const uint8_t *scales = record + size_t(2 * inter) * hidden / 2;
    int route[kExpertRouteTile];const unsigned *x[kExpertRouteTile];
    float gs[kExpertRouteTile]{},us[kExpertRouteTile]{};
#pragma unroll
    for(int r=0;r<kExpertRouteTile;++r)if(r<count){
        route[r]=routeMap ? routeMap[selected+r] : selected+r;
        x[r]=reinterpret_cast<const unsigned *>(input+size_t(route[r]/topk)*hidden);
    }
    if(count==1){
        const auto *a=reinterpret_cast<const __nv_bfloat16 *>(x[0]);
        const float2 value=planar ? DotGateUp<true>(a,g,u,hidden,scales+size_t(column)*blocks,
            scales+size_t(column+inter)*blocks) : DotGateUp(a,g,u,hidden);
        StoreGate(value,scores[route[0]],activation,route[0],inter,limit,siluTable);return;
    }
#pragma unroll 4
    for(int b=0;b<blocks;++b){
        const int at=b*(planar?16:17);
        const float2 gv=FP4Pair(g[at+lane],planar?scales[size_t(column)*blocks+b]:g[at+16]);
        const float2 uv=FP4Pair(u[at+lane],planar?scales[size_t(column+inter)*blocks+b]:u[at+16]);
#pragma unroll
        for(int r=0;r<kExpertRouteTile;++r)if(r<count){
            const unsigned value=x[r][b*16+lane];
            const float x0=__uint_as_float(value<<16),x1=__uint_as_float(value&0xffff0000u);
            gs[r]=__fadd_rn(gs[r],__fadd_rn(__fmul_rn(x0,gv.x),__fmul_rn(x1,gv.y)));
            us[r]=__fadd_rn(us[r],__fadd_rn(__fmul_rn(x0,uv.x),__fmul_rn(x1,uv.y)));
        }
    }
#pragma unroll
    for(int r=0;r<kExpertRouteTile;++r)if(r<count){
        for(int delta=8;delta;delta>>=1){
            gs[r]=__fadd_rn(gs[r],__shfl_down_sync(0xffffffff,gs[r],delta,16));
            us[r]=__fadd_rn(us[r],__shfl_down_sync(0xffffffff,us[r],delta,16));
        }
        StoreGate(make_float2(gs[r],us[r]),scores[route[r]],activation,route[r],inter,limit,siluTable);
        __syncthreads();
    }
}
template<bool SmallOnly = false>
__global__ void DownReuse(const __nv_bfloat16 *activation, const int32_t *slots,
        const uint8_t *records, float *output, int hidden, int inter, size_t stride,
        size_t downOffset, const int32_t *routeMap, int routes, bool planar) {
    const int selected=blockIdx.y;
    if constexpr (SmallOnly) if (ExpertRouteTile<kExpertMmaRouteTile>(slots,selected,routes) != 1) return;
    const int count=ExpertRouteTile(slots,selected,routes);
    if(!count)return;
    const int lane=threadIdx.x&15,column=blockIdx.x*8+threadIdx.x/16,blocks=inter/32;
    const uint8_t *matrix=records+size_t(slots[selected])*stride+downOffset;
    const uint8_t *w=matrix+size_t(column)*blocks*(planar?16:17);
    const uint8_t *scales=matrix+size_t(hidden)*inter/2+size_t(column)*blocks;
    int route[kExpertRouteTile];const unsigned *x[kExpertRouteTile];float sum[kExpertRouteTile]{};
#pragma unroll
    for(int r=0;r<kExpertRouteTile;++r)if(r<count){
        route[r]=routeMap?routeMap[selected+r]:selected+r;
        x[r]=reinterpret_cast<const unsigned *>(activation+size_t(route[r])*inter);
    }
    if(count==1){
        const auto *a=reinterpret_cast<const __nv_bfloat16 *>(x[0]);
        const float value=planar?Dot<true>(a,w,inter,scales):Dot(a,w,inter);
        if(!lane)output[size_t(route[0])*hidden+column]=BFloat(value);return;
    }
#pragma unroll 8
    for(int b=0;b<blocks;++b){
        const int at=b*(planar?16:17);const float2 value=FP4Pair(w[at+lane],planar?scales[b]:w[at+16]);
#pragma unroll
        for(int r=0;r<kExpertRouteTile;++r)if(r<count){
            const unsigned v=x[r][b*16+lane];
            sum[r]=__fadd_rn(sum[r],__fadd_rn(__fmul_rn(__uint_as_float(v<<16),value.x),
                __fmul_rn(__uint_as_float(v&0xffff0000u),value.y)));
        }
    }
#pragma unroll
    for(int r=0;r<kExpertRouteTile;++r)if(r<count){
        for(int delta=8;delta;delta>>=1)sum[r]=__fadd_rn(sum[r],__shfl_down_sync(0xffffffff,sum[r],delta,16));
        if(!lane)output[size_t(route[r])*hidden+column]=BFloat(sum[r]);
    }
}

template <bool flatGrid = false>
static __global__ void Reduce(const float *cpu, const float *gpu, const int32_t *gpuIndices, const int32_t *indices,
                              __nv_bfloat16 *output, int hidden, int topk) {
    const int columns = (hidden + blockDim.x - 1) / blockDim.x;
    const int column = (flatGrid ? blockIdx.x % columns : blockIdx.x) * blockDim.x + threadIdx.x;
    const int row = flatGrid ? blockIdx.x / columns : blockIdx.y;
    cpu += (size_t)row * topk * hidden;
    gpu += (size_t)row * topk * hidden;
    gpuIndices += row * topk;
    indices += row * topk;
    output += (size_t)row * hidden;
    // Sort once per CTA, preserving ascending expert IDs and route order on
    // ties. All threads must reach the barrier, including the final partial CTA.
    __shared__ int order[16], fromGpu[16];
    if (threadIdx.x < topk) {
        const int route = threadIdx.x, id = indices[route];
        int rank = 0;
        for (int r = 0; r < topk; ++r)
            rank += indices[r] < id || (indices[r] == id && r < route);
        order[rank] = route;
        fromGpu[rank] = gpuIndices[route] >= 0;
    }
    __syncthreads();
    if (column >= hidden) return;
    float sum = 0;
    for (int r = 0; r < topk; ++r)
        sum = __fadd_rn(sum, (fromGpu[r] ? gpu : cpu)[order[r] * hidden + column]);
    output[column] = __float2bfloat16_rn(sum);
}
}
}
}
