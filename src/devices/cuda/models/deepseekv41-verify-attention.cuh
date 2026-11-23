// SM103 verification adapter. The split MMA and rounding of every partial
// match the ordinary FlashInfer path. A resident grid barrier separates history
// from the causal merge. Request rows need not belong to the same chunk.
struct VerifyQueryRope {
    const float *angles;

    __device__ __forceinline__ uint4 operator()(const bf16 *q, int row, int d) const {
        uint4 result = *reinterpret_cast<const uint4 *>(q);
        if (d >= 448) {
            auto *pairs = reinterpret_cast<__nv_bfloat162 *>(&result);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float2 v = __bfloat1622float2(pairs[i]);
                const float *a = angles + row * 64 + (d - 448) / 2 + i;
                pairs[i] = __floats2bfloat162_rn(v.x * a[0] - v.y * a[32],
                                                __fmaf_rn(v.y, a[0], __fmul_rn(v.x, a[32])));
            }
        }
        return result;
    }
};

__device__ __forceinline__ float VerifyOutputScale(float peak) {
    const unsigned bits = __float_as_uint(fmaxf(peak, 1e-4f) * (1.f / 448.f));
    return __uint_as_float((bits & 0x7f800000u) + ((bits & 0x007fffffu) ? 0x00800000u : 0));
}

template<bool Extra>
__global__ void __launch_bounds__(256, 1) V41VerifySparseAttention(
        FastllmV41FullDecodeBuffers b, const FastllmV41FullDecodeRequest *requests,
        bf16 *partial, float *lse, float *angles, int rows, int width, int ratio, int splits, float scale, bool packedOutput) {
    const auto *q = static_cast<const bf16 *>(b.q);
    const int tasks = rows * 4 * splits;
    for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
        const unsigned row = task / (4 * splits), split = (task / 4) % splits, group = task % 4;
        const auto r = requests[row];
        int active = 0;
        if constexpr (Extra) {
            const int end = requests[r.chunkBegin].position + (r.chunkLength ? r.chunkLength : rows);
            active = (min(width, end / ratio) + 63) / 64;
        }
        if (split >= 2 + active) {
            if (threadIdx.x < 16) lse[(row * 64 + group * 16 + threadIdx.x) * splits + split] = -1e30f;
        } else {
            dsv41_attention_splits<64, Extra, true, true, false>(
                q, nullptr, nullptr, partial, lse, nullptr, nullptr,
                static_cast<const int *>(b.topk), nullptr, width, 1, 288, rows, 64, 128,
                splits, 1, scale, 528, 128, width, Dsv41Pages{1, false}, &requests->windowKV,
                Extra ? &requests->compressedKV : nullptr, sizeof(*requests), &requests->position,
                ratio, &requests->chunkBegin, &requests->chunkLength, VerifyQueryRope{angles}, make_uint3(row, group, split));
        }
        __syncthreads();
    }
    cooperative_groups::this_grid().sync();
    const int half = threadIdx.x / 128, t = threadIdx.x % 128;
    extern __shared__ __align__(16) unsigned char storage[];
    auto *queryHeads = reinterpret_cast<bf16 (*)[512]>(storage);
    auto *weights = reinterpret_cast<float (*)[160]>(queryHeads + 2);
    for (int index = blockIdx.x * 2 + half; index < rows * 64; index += gridDim.x * 2) {
        const int row = index / 64, head = index % 64;
        const int first = requests[row].chunkBegin;
        if (t < 64) {
            reinterpret_cast<uint4 *>(queryHeads[half])[t] =
                VerifyQueryRope{angles}(q + size_t(index) * 512 + t * 8, row, t * 8);
        }
        fastllm_v41_verify::MergeBarrier<true>();
        uint2 result = fastllm_v41_verify::MergeHead<true>(queryHeads[half],
            static_cast<const bf16 *>(b.kv) + size_t(first) * 512,
            partial + size_t(index) * splits * 512, lse + index * splits,
            static_cast<const float *>(b.sink)[head], scale, splits, row - first,
            weights[half], weights[half] + 128);
        auto *pairs = reinterpret_cast<__nv_bfloat162 *>(&result);
        float vals[4];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            float2 v = __bfloat1622float2(pairs[k]);
            if (t >= 112) {
                const float *a = angles + row * 64 + (t - 112) * 2 + k;
                v = __bfloat1622float2(__floats2bfloat162_rn(v.x * a[0] - v.y * (-a[32]), v.x * (-a[32]) + v.y * a[0]));
            }
            vals[2 * k] = v.x; vals[2 * k + 1] = v.y;
        }
        float peak = fmaxf(fmaxf(fabsf(vals[0]), fabsf(vals[1])), fmaxf(fabsf(vals[2]), fabsf(vals[3])));
#pragma unroll
        for (int step = 4; step; step >>= 1) peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, step));
        const float sf = VerifyOutputScale(peak);
        const uint32_t exponent = __float_as_uint(sf);
        uint32_t codes = 0;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float code = exponent >= 0x00800000u && exponent < 0x7f000000u ?
                vals[k] * __uint_as_float(0x7f000000u - exponent) : vals[k] / sf;
            const __nv_fp8_e4m3 value(fminf(448.f, fmaxf(-448.f, code)));
            if (packedOutput) codes |= uint32_t(value.__x) << (8 * k);
            else vals[k] = float(value) * sf;
        }
        if (packedOutput) {
            static_cast<uint32_t *>(b.output)[size_t(index) * 128 + t] = codes;
            if ((t & 7) == 0) {
                const int block = head * 16 + t / 8, pitch = (rows + 3) / 4 * 4;
                auto *sfOut = static_cast<uint8_t *>(b.output) + size_t(rows) * 32768;
                sfOut[(size_t(block / 4) * pitch + row) * 4 + block % 4] = exponent >> 23;
            }
        } else {
            pairs[0] = __floats2bfloat162_rn(vals[0], vals[1]);
            pairs[1] = __floats2bfloat162_rn(vals[2], vals[3]);
            reinterpret_cast<uint2 *>(b.output)[size_t(index) * 128 + t] = result;
        }
        fastllm_v41_verify::MergeBarrier<true>();
    }
}
