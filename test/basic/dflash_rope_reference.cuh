// Regression oracle: the previous table-based fused kernels, with tables
// containing only the tested positions. This preserves their normalization,
// BF16 rounding, layout conversion and cache-write semantics independently of
// the new position-independent implementation.
namespace reference {
__global__ void FastllmDFlashPrepareQkvBf16Kernel(
        const __nv_bfloat16 *__restrict__ qkv,
        const float *__restrict__ qNormWeight,
        const float *__restrict__ kNormWeight,
        const float *__restrict__ positionIds,
        const float *__restrict__ sinData,
        const float *__restrict__ cosData,
        half *__restrict__ query,
        half *__restrict__ key,
        half *__restrict__ value,
        int tokens, int projectionStride, int queryHeads,
        int kvHeads, int headDim, int sinCosStride, float eps) {
    const int outputHead = blockIdx.x / tokens;
    const int token = blockIdx.x % tokens;
    int kind;
    int head;
    if (outputHead < queryHeads) {
        kind = 0;
        head = outputHead;
    } else if (outputHead < queryHeads + kvHeads) {
        kind = 1;
        head = outputHead - queryHeads;
    } else {
        kind = 2;
        head = outputHead - queryHeads - kvHeads;
    }

    const int sourceHead = kind == 0 ? head :
        (kind == 1 ? queryHeads + head : queryHeads + kvHeads + head);
    const __nv_bfloat16 *source = qkv +
        (size_t)token * projectionStride + (size_t)sourceHead * headDim;
    half *destination = (kind == 0 ? query : (kind == 1 ? key : value)) +
        (size_t)(head * tokens + token) * headDim;
    const int tid = threadIdx.x;

    if (kind == 2) {
        for (int channel = tid; channel < headDim; channel += blockDim.x) {
            destination[channel] = __float2half_rz(
                __bfloat162float(source[channel]));
        }
        return;
    }

    __shared__ float warpSums[2];
    __shared__ float scale;
    __shared__ __nv_bfloat16 normalized[128];
    const __nv_bfloat162 *source2 =
        reinterpret_cast<const __nv_bfloat162 *>(source);
    float sum2 = 0.0f;
    for (int channel = tid; channel < headDim / 2;
         channel += blockDim.x) {
        __nv_bfloat162 pair = source2[channel];
        const float low = __bfloat162float(pair.x);
        const float high = __bfloat162float(pair.y);
        sum2 += low * low + high * high;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum2 += __shfl_down_sync(0xffffffff, sum2, offset);
    }
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (lane == 0) {
        warpSums[warp] = sum2;
    }
    __syncthreads();
    if (warp == 0) {
        float sum = lane < 2 ? warpSums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            scale = rsqrtf(sum / headDim + eps);
        }
    }
    __syncthreads();

    const float *normWeight = kind == 0 ? qNormWeight : kNormWeight;
    for (int channel = tid; channel < headDim / 2;
         channel += blockDim.x) {
        __nv_bfloat162 pair = source2[channel];
        normalized[channel * 2] = __float2bfloat16_rn(
            __bfloat162float(pair.x) * scale *
                __ldg(normWeight + channel * 2));
        normalized[channel * 2 + 1] = __float2bfloat16_rn(
            __bfloat162float(pair.y) * scale *
                __ldg(normWeight + channel * 2 + 1));
    }
    __syncthreads();

    const int halfHeadDim = headDim / 2;
    if (tid < halfHeadDim) {
        const int position = (int)positionIds[token];
        const float sine = sinData[(size_t)position * sinCosStride + tid];
        const float cosine =
            cosData[(size_t)position * sinCosStride + tid];
        const float first = __bfloat162float(normalized[tid]);
        const float second =
            __bfloat162float(normalized[tid + halfHeadDim]);
        const __nv_bfloat16 rotatedFirst = __float2bfloat16_rn(
            first * cosine - second * sine);
        const __nv_bfloat16 rotatedSecond = __float2bfloat16_rn(
            first * sine + second * cosine);
        destination[tid] = __float2half_rz(
            __bfloat162float(rotatedFirst));
        destination[tid + halfHeadDim] = __float2half_rz(
            __bfloat162float(rotatedSecond));
    }
}


struct FastllmDFlashKvCacheOutput {
    half *pointers[10];
    size_t headStride;
    int tokenOffset;
};

__global__ void FastllmDFlashMaterializeKvBf16Kernel(
        const __nv_bfloat16 *__restrict__ projectedKv,
        const float *__restrict__ kNormWeights,
        const float *__restrict__ positionIds,
        const float *__restrict__ sinData,
        const float *__restrict__ cosData,
        half *__restrict__ output,
        FastllmDFlashKvCacheOutput cacheOutput,
        int tokens, int projectionStride, int kvHeads,
        int headDim, int sinCosStride, float eps) {
    int item = blockIdx.x;
    int kind = item & 1;
    item >>= 1;
    int head = item % kvHeads;
    item /= kvHeads;
    int token = item % tokens;
    int layer = item / tokens;

    const int sourceChannel =
        ((layer * 2 + kind) * kvHeads + head) * headDim;
    const __nv_bfloat16 *source =
        projectedKv + (size_t)token * projectionStride + sourceChannel;
    half *destination;
    if (output != nullptr) {
        destination = output +
            (size_t)(((layer * 2 + kind) * kvHeads + head) * tokens +
                     token) * headDim;
    } else {
        destination = cacheOutput.pointers[layer * 2 + kind] +
            (size_t)head * cacheOutput.headStride +
            (size_t)(cacheOutput.tokenOffset + token) * headDim;
    }
    const int tid = threadIdx.x;

    if (kind != 0) {
        for (int channel = tid; channel < headDim; channel += blockDim.x) {
            destination[channel] = __float2half_rz(
                __bfloat162float(source[channel]));
        }
        return;
    }

    __shared__ float warpSums[2];
    __shared__ float scale;
    __shared__ __nv_bfloat16 normalized[128];
    const __nv_bfloat162 *source2 =
        reinterpret_cast<const __nv_bfloat162 *>(source);
    float sum2 = 0.0f;
    for (int channel = tid; channel < headDim / 2;
         channel += blockDim.x) {
        __nv_bfloat162 value = source2[channel];
        float low = __bfloat162float(value.x);
        float high = __bfloat162float(value.y);
        sum2 += low * low + high * high;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum2 += __shfl_down_sync(0xffffffff, sum2, offset);
    }
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (lane == 0) {
        warpSums[warp] = sum2;
    }
    __syncthreads();
    if (warp == 0) {
        float value = lane < 2 ? warpSums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        if (lane == 0) {
            scale = rsqrtf(value / headDim + eps);
        }
    }
    __syncthreads();

    const float *normWeight = kNormWeights + (size_t)layer * headDim;
    for (int channel = tid; channel < headDim / 2;
         channel += blockDim.x) {
        __nv_bfloat162 value = source2[channel];
        float low = __bfloat162float(value.x);
        float high = __bfloat162float(value.y);
        normalized[channel * 2] = __float2bfloat16_rn(
            low * scale * __ldg(normWeight + channel * 2));
        normalized[channel * 2 + 1] = __float2bfloat16_rn(
            high * scale * __ldg(normWeight + channel * 2 + 1));
    }
    __syncthreads();

    const int halfHeadDim = headDim / 2;
    if (tid < halfHeadDim) {
        const int position = (int)positionIds[token];
        const float sine = sinData[(size_t)position * sinCosStride + tid];
        const float cosine = cosData[(size_t)position * sinCosStride + tid];
        const float first = __bfloat162float(normalized[tid]);
        const float second =
            __bfloat162float(normalized[tid + halfHeadDim]);
        const __nv_bfloat16 rotatedFirst = __float2bfloat16_rn(
            first * cosine - second * sine);
        const __nv_bfloat16 rotatedSecond = __float2bfloat16_rn(
            first * sine + second * cosine);
        destination[tid] = __float2half_rz(
            __bfloat162float(rotatedFirst));
        destination[tid + halfHeadDim] = __float2half_rz(
            __bfloat162float(rotatedSecond));
    }
}


}
