// Private eager TP2 acceleration. The CPU rendezvous in multicuda.cu owns
// ordering; these buffers are reused only after both ranks finish reading.
struct FastllmHostStagedInput {
    uint8_t *data = nullptr;
    size_t capacity = 0;

    ~FastllmHostStagedInput() {
        if (data != nullptr)
            cudaFreeHost(data);
    }

    bool Reserve(size_t bytes) {
#if defined(_WIN32) && !defined(USE_ROCM)
        int device = 0, canMap = 0;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&canMap, cudaDevAttrCanMapHostMemory, device) != cudaSuccess ||
            !canMap) {
            cudaGetLastError();
            return false;
        }
        if (capacity >= bytes)
            return true;
        void *next = nullptr;
        cudaError_t status = cudaHostAlloc(
            &next, bytes, cudaHostAllocPortable | cudaHostAllocMapped);
        if (status != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        if (data != nullptr)
            cudaFreeHost(data);
        data = static_cast<uint8_t *>(next);
        capacity = bytes;
        return true;
#else
        return false;
#endif
    }

    FastllmHostStagedInput() = default;
    FastllmHostStagedInput(const FastllmHostStagedInput &) = delete;
    FastllmHostStagedInput &operator=(const FastllmHostStagedInput &) = delete;
};

#if defined(_WIN32) && !defined(USE_ROCM)
// Match the existing host fallback's software FP16 conversions, including
// its halfway rounding and exponent-31 handling. Using native half addition
// here would silently change eager SUM results for some representable inputs.
__device__ __forceinline__ float FastllmHostStagedHalfToFloat(uint16_t x) {
    const uint32_t sign = (uint32_t)(x & 0x8000) << 16;
    const uint32_t e = (x & 0x7c00) >> 10;
    const uint32_t m = (x & 0x03ff) << 13;
    if (e != 0)
        return __uint_as_float(sign | ((e + 112) << 23) | m);
    if (m == 0)
        return __uint_as_float(sign);
    const uint32_t v = __float_as_uint((float)m) >> 23;
    return __uint_as_float(sign | ((v - 37) << 23) |
                           ((m << (150 - v)) & 0x007fe000));
}

__device__ __forceinline__ uint16_t FastllmHostStagedFloatToHalf(float x) {
    const uint32_t b = __float_as_uint(x) + 0x00001000;
    const uint32_t sign = (b & 0x80000000) >> 16;
    const uint32_t e = (b & 0x7f800000) >> 23;
    const uint32_t m = b & 0x007fffff;
    if (e > 143)
        return (uint16_t)(sign | 0x7fff);
    if (e > 112)
        return (uint16_t)(sign | (((e - 112) << 10) & 0x7c00) | (m >> 13));
    if (e > 101)
        return (uint16_t)(sign | ((((0x007ff000 + m) >> (125 - e)) + 1) >> 1));
    return (uint16_t)sign;
}

template <typename T, int DataType>
__global__ void FastllmHostStagedSumKernel(
        const T *local, const T *peer, T *output,
        int count, int rank) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < (size_t)count; i += (size_t)blockDim.x * gridDim.x) {
        const T a = rank == 0 ? local[i] : peer[i];
        const T b = rank == 0 ? peer[i] : local[i];
        if constexpr (DataType == fastllm::FLOAT16) {
            const float sum = __fadd_rn(__fadd_rn(0.0f, FastllmHostStagedHalfToFloat(a)),
                                        FastllmHostStagedHalfToFloat(b));
            output[i] = FastllmHostStagedFloatToHalf(sum);
        } else if constexpr (DataType == fastllm::BFLOAT16) {
            const float sum = __fadd_rn(__fadd_rn(0.0f, __uint_as_float((uint32_t)a << 16)),
                                        __uint_as_float((uint32_t)b << 16));
            uint32_t bits = __float_as_uint(sum);
            // CUDA may canonicalize NaNs to 0x7fffffff. Applying the normal
            // rounding bias to that encoding would wrap it into signed zero.
            if ((bits & 0x7fffffffU) > 0x7f800000U) {
                output[i] = (uint16_t)((bits >> 16) | 0x0040U);
            } else {
                bits += 0x7fffU + ((bits >> 16) & 1U);
                output[i] = (uint16_t)(bits >> 16);
            }
        } else if constexpr (DataType == fastllm::FLOAT32) {
            // Preserve rank order, initial +0, and FP32 rounding/subnormals.
            output[i] = __fadd_rn(__fadd_rn(0.0f, a), b);
        } else {
            // As in the CPU fallback, accumulate before narrowing to INT8/32.
            output[i] = (T)((int64_t)a + (int64_t)b);
        }
    }
}

template <typename T, int DataType>
static void FastllmLaunchHostStagedSum(
        const void *send, void *recv, const void *peer, int count, int rank) {
    const int blocks = (int)std::min<size_t>(((size_t)count + 255) / 256, 1024);
    FastllmHostStagedSumKernel<T, DataType><<<blocks, 256, 0, cudaStreamPerThread>>>(
        static_cast<const T *>(send), static_cast<const T *>(peer),
        static_cast<T *>(recv), count, rank);
}
#endif

static bool FastllmHostStagedSum(
        const void *send, void *recv, FastllmHostStagedInput &peer,
        int count, int dataType, int rank) {
#if defined(_WIN32) && !defined(USE_ROCM)
    void *mapped = nullptr;
    cudaError_t status = cudaHostGetDevicePointer(&mapped, peer.data, 0);
    if (status == cudaSuccess) {
        switch (dataType) {
        case fastllm::FLOAT16:
            FastllmLaunchHostStagedSum<uint16_t, fastllm::FLOAT16>(send, recv, mapped, count, rank);
            break;
        case fastllm::BFLOAT16:
            FastllmLaunchHostStagedSum<uint16_t, fastllm::BFLOAT16>(send, recv, mapped, count, rank);
            break;
        case fastllm::FLOAT32:
            FastllmLaunchHostStagedSum<float, fastllm::FLOAT32>(send, recv, mapped, count, rank);
            break;
        case fastllm::INT8:
            FastllmLaunchHostStagedSum<int8_t, fastllm::INT8>(send, recv, mapped, count, rank);
            break;
        case fastllm::INT32:
            FastllmLaunchHostStagedSum<int32_t, fastllm::INT32>(send, recv, mapped, count, rank);
            break;
        default:
            return false;
        }
        status = cudaGetLastError();
    }
    if (status == cudaSuccess)
        status = cudaStreamSynchronize(cudaStreamPerThread);
    if (status != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
#else
    return false;
#endif
}
