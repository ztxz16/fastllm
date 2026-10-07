#include "fastllm-cuda-gguf-projections.h"
#include "fastllm-cuda-gguf-linear-add.h"
#include "fastllm-gguf-mmvq-dispatch.cuh"

#include <cuda_bf16.h>

namespace fastllm_gguf_mmq {

#include "mmq.cuh"
#include "../moe/fastllm-moe-gguf-q8.cuh"

constexpr int kQuantizeBlockSize = 128;
constexpr int kBlackwellMinMmqRows = 8;
constexpr int kDefaultMinMmqRows = 9;
constexpr int kMaxMmqRows = 1024;

#include "fastllm-gguf-mmq-io.cuh"

static bool is_extended_mmvq_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
            return true;
        default:
            return false;
    }
}

// Check the actual weight block size, not only the Q8 activation block.
// The caller must first establish that the type has an implemented kernel.
static bool supports_mmvq_shape(ggml_type type, int rows, int columns, int output_rows) {
    return rows > 0 && rows <= 8 && columns > 0 &&
           columns % ggml_blck_size(type) == 0 && output_rows > 0;
}

template <typename OutputType, int StoreMode = 0>
static void dispatch_extended_mmvq(
        ggml_type type, const void *weight, const block_q8_1 *input,
        OutputType *output, int rows, int input_columns, int output_rows,
        cudaStream_t stream) {
// Instantiate fused stores only for formats admitted by the public entry points.
#define FASTLLM_DISPATCH_EXTENDED_MMVQ(type_name)                        \
    case type_name:                                                       \
        if constexpr (StoreMode == 0 || type_name == GGML_TYPE_IQ1_M ||   \
                      type_name == GGML_TYPE_IQ2_XXS ||                  \
                      type_name == GGML_TYPE_IQ2_XS || type_name == GGML_TYPE_IQ2_S) { \
            launch_extended_mmvq_type<type_name, OutputType, StoreMode>(  \
                weight, input, output, rows, input_columns, output_rows, stream); \
        }                                                                \
        break
    switch (type) {
        FASTLLM_DISPATCH_EXTENDED_MMVQ(GGML_TYPE_Q4_0);
        FASTLLM_DISPATCH_EXTENDED_MMVQ(GGML_TYPE_Q4_1);
        FASTLLM_DISPATCH_EXTENDED_MMVQ(GGML_TYPE_IQ2_XXS);
        FASTLLM_DISPATCH_EXTENDED_MMVQ(GGML_TYPE_IQ2_XS);
        FASTLLM_DISPATCH_EXTENDED_MMVQ(GGML_TYPE_IQ2_S);
        FASTLLM_DISPATCH_EXTENDED_MMVQ(GGML_TYPE_IQ1_S);
        FASTLLM_DISPATCH_EXTENDED_MMVQ(GGML_TYPE_IQ1_M);
        default: break;
    }
#undef FASTLLM_DISPATCH_EXTENDED_MMVQ
}

static void dispatch_extended_gate_up(
        ggml_type type, const void *gate_weight, const void *up_weight,
        const block_q8_1 *input, half *output, int rows,
        int input_columns, int output_rows, cudaStream_t stream) {
#define FASTLLM_DISPATCH_EXTENDED_GATE_UP(type_name)                     \
    case type_name:                                                       \
        launch_extended_gate_up_type<type_name>(                          \
            gate_weight, up_weight, input, output, rows, input_columns,   \
            output_rows, stream);                                         \
        break
    switch (type) {
        FASTLLM_DISPATCH_EXTENDED_GATE_UP(GGML_TYPE_IQ2_XXS);
        FASTLLM_DISPATCH_EXTENDED_GATE_UP(GGML_TYPE_IQ2_XS);
        FASTLLM_DISPATCH_EXTENDED_GATE_UP(GGML_TYPE_IQ2_S);
        default: break;
    }
#undef FASTLLM_DISPATCH_EXTENDED_GATE_UP
}

// Ordinary Q2_0 rows can use the same Q8/DP4A arithmetic as resident experts.
// A warp computes one output row; the block shares its quantized activation.
template<typename OutputType>
static __global__ void q2_mmvq(
        const block_q2_0 *weight, const block_q8_1 *input,
        OutputType *output, int columns, int output_rows) {
    extern __shared__ uint32_t activation[];
    const int input_blocks = columns/QK8_1;
    const auto *source = reinterpret_cast<const uint32_t *>(
        input + size_t(blockIdx.y)*input_blocks);
    for (int i = threadIdx.x; i < input_blocks*int(sizeof(block_q8_1))/4;
         i += blockDim.x) activation[i] = source[i];
    __syncthreads();
    const int row = blockIdx.x*8+threadIdx.x/32;
    if (row >= output_rows) return;
    const float value = gguf_cache_q8::RowDot<GGML_TYPE_Q2_0>(
        weight + size_t(row)*(columns/QK2_0),
        reinterpret_cast<const block_q8_1 *>(activation), columns, nullptr);
    if (threadIdx.x%32 == 0)
        output[size_t(blockIdx.y)*output_rows+row] = mmq_io<OutputType>::from_float(value);
}

template <typename InputType, typename OutputType, int StoreMode = 0>
static bool matmul_mmvq(
        const InputType *input, const void *weight, OutputType *output,
        ggml_type type, int rows, int input_columns, int output_rows,
        cudaStream_t stream) {
    const bool q2 = type == GGML_TYPE_Q2_0;
    if constexpr (StoreMode == 1) {
        if (type != GGML_TYPE_IQ2_S && type != GGML_TYPE_IQ2_XS) return false;
    }
    if ((!q2 && !is_extended_mmvq_type(type)) ||
        !supports_mmvq_shape(type, rows, input_columns, output_rows) ||
        (q2 && size_t(input_columns/QK8_1)*sizeof(block_q8_1) > 32*1024)) {
        return false;
    }
    if (type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M) {
        ensure_extended_iq1s_grid(stream);
    }

    const size_t block_count =
        static_cast<size_t>(rows) * input_columns / QK8_1;
    block_q8_1 *quantized = nullptr;
    if (FastllmCudaTryMalloc(reinterpret_cast<void **>(&quantized),
                             block_count * sizeof(block_q8_1)) !=
        FASTLLM_CUDA_TRY_MALLOC_SUCCESS) {
        return false;
    }
    constexpr int threads = 256;
    const dim3 blocks(
        (input_columns + threads - 1) / threads, rows, 1);
    quantize_mmvq_q8_1<<<blocks, threads, 0, stream>>>(
        input, quantized, input_columns);

    if (q2) {
        q2_mmvq<<<dim3((output_rows+7)/8, rows), 256,
            size_t(input_columns/QK8_1)*sizeof(block_q8_1), stream>>>(
                static_cast<const block_q2_0 *>(weight), quantized, output,
                input_columns, output_rows);
    } else {
        dispatch_extended_mmvq<OutputType, StoreMode>(
            type, weight, quantized, output, rows, input_columns, output_rows,
            stream);
    }
    FastllmCudaFree(quantized);
    return true;
}

static bool gate_up_mmvq(
        const half *input, const void *gate_weight, const void *up_weight,
        half *output, ggml_type type, int rows, int input_columns,
        int output_rows, cudaStream_t stream) {
    // dispatch_extended_gate_up implements only the IQ2 family. Reject other
    // formats before allocating/quantizing so callers can use their fallback.
    if ((type != GGML_TYPE_IQ2_XXS && type != GGML_TYPE_IQ2_XS &&
         type != GGML_TYPE_IQ2_S) ||
        !supports_mmvq_shape(type, rows, input_columns, output_rows)) {
        return false;
    }

    const size_t block_count =
        static_cast<size_t>(rows) * input_columns / QK8_1;
    block_q8_1 *quantized = nullptr;
    if (FastllmCudaTryMalloc(reinterpret_cast<void **>(&quantized),
                             block_count * sizeof(block_q8_1)) !=
        FASTLLM_CUDA_TRY_MALLOC_SUCCESS) {
        return false;
    }
    constexpr int threads = 256;
    const dim3 blocks(
        (input_columns + threads - 1) / threads, rows, 1);
    quantize_mmvq_q8_1<<<blocks, threads, 0, stream>>>(
        input, quantized, input_columns);
    dispatch_extended_gate_up(
        type, gate_weight, up_weight, quantized, output, rows,
        input_columns, output_rows, stream);
    FastllmCudaFree(quantized);
    return true;
}

template <mmq_q8_1_ds_layout layout, typename InputType>
static __global__ void quantize_mmq_q8_1(
        const InputType *__restrict__ input, void *__restrict__ quantized,
        int64_t cols, int64_t rows, int64_t padded_cols) {
    constexpr int values_per_scale =
        layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int values_per_sum =
        layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t col =
        (static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x) * 4;
    if (col >= padded_cols) {
        return;
    }

    const int64_t row = rows * blockIdx.z + blockIdx.y;
    block_q8_1_mmq *output = static_cast<block_q8_1_mmq *>(quantized);
    const int64_t first_block =
        blockIdx.z * (static_cast<int64_t>(gridDim.y) * gridDim.x *
                      blockDim.x / QK8_1);
    const int64_t block = first_block + (col / (4 * QK8_1)) * rows +
                          blockIdx.y;
    const int64_t quant_index = col % (4 * QK8_1);

    float4 values = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (col < cols) {
        const InputType *source = input + row * cols + col;
        values.x = mmq_io<InputType>::to_float(source[0]);
        values.y = mmq_io<InputType>::to_float(source[1]);
        values.z = mmq_io<InputType>::to_float(source[2]);
        values.w = mmq_io<InputType>::to_float(source[3]);
    }

    float abs_max = fabsf(values.x);
    abs_max = fmaxf(abs_max, fabsf(values.y));
    abs_max = fmaxf(abs_max, fabsf(values.z));
    abs_max = fmaxf(abs_max, fabsf(values.w));
#pragma unroll
    for (int mask = values_per_scale / 8; mask > 0; mask >>= 1) {
        abs_max = fmaxf(
            abs_max,
            __shfl_xor_sync(0xffffffff, abs_max, mask, WARP_SIZE));
    }

    float sum = 0.0f;
    if constexpr (layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = values.x + values.y + values.z + values.w;
#pragma unroll
        for (int mask = values_per_sum / 8; mask > 0; mask >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, mask, WARP_SIZE);
        }
    }

    float scale = abs_max / 127.0f;
    const float inverse_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
    char4 quants;
    quants.x = static_cast<int8_t>(roundf(values.x * inverse_scale));
    quants.y = static_cast<int8_t>(roundf(values.y * inverse_scale));
    quants.z = static_cast<int8_t>(roundf(values.z * inverse_scale));
    quants.w = static_cast<int8_t>(roundf(values.w * inverse_scale));
    reinterpret_cast<char4 *>(output[block].qs)[quant_index / 4] = quants;

    if constexpr (layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (quant_index % 16 != 0 || quant_index >= 96) {
            return;
        }
        output[block].d2s6[2 + quant_index / 16] = __float2half(sum);
        if (quant_index % 64 == 0) {
            output[block].d2s6[quant_index / 64] = __float2half(scale);
        }
    } else {
        if (quant_index % 32 != 0) {
            return;
        }
        if constexpr (layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
            scale = fmaxf(-65504.0f, fminf(65504.0f, scale));
            sum = fmaxf(-65504.0f, fminf(65504.0f, sum));
            output[block].ds4[quant_index / 32] =
                make_half2(__float2half(scale), __float2half(sum));
        } else {
            output[block].d4[quant_index / 32] = scale;
        }
    }
}

static bool is_supported_type(ggml_type type) {
    return type == GGML_TYPE_Q2_0 || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ||
           type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_Q3_K ||
           type == GGML_TYPE_Q6_K || type == GGML_TYPE_IQ3_S ||
           type == GGML_TYPE_IQ3_XXS ||
           type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_Q4_0 ||
           type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 ||
           type == GGML_TYPE_Q5_1 || type == GGML_TYPE_Q8_0 ||
           type == GGML_TYPE_Q2_K ||
           type == GGML_TYPE_IQ1_M ||
           type == GGML_TYPE_IQ2_XXS ||
           type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S ||
           type == GGML_TYPE_IQ1_S;
}

template <typename InputType>
static void launch_quantize(
        ggml_type type, const InputType *input,
        block_q8_1_mmq *quantized,
        int rows, int cols, int padded_cols, cudaStream_t stream) {
    const dim3 blocks((padded_cols + 4 * kQuantizeBlockSize - 1) /
                          (4 * kQuantizeBlockSize),
                      rows, 1);
    const dim3 threads(kQuantizeBlockSize, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, InputType>
                <<<blocks, threads, 0, stream>>>(
                    input, quantized, cols, rows, padded_cols);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4, InputType>
                <<<blocks, threads, 0, stream>>>(
                    input, quantized, cols, rows, padded_cols);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6, InputType>
                <<<blocks, threads, 0, stream>>>(
                    input, quantized, cols, rows, padded_cols);
            break;
    }
}

template <ggml_type type, typename OutputType>
void launch_mmq_type(ggml_backend_cuda_context &context, const mmq_args &args,
                     OutputType *output, cudaStream_t stream);

template <typename OutputType>
static void launch_mmq(
        ggml_type type, ggml_backend_cuda_context &context,
        const mmq_args &args, OutputType *output, cudaStream_t stream) {
    switch (type) {
        case GGML_TYPE_Q2_0:
            launch_mmq_type<GGML_TYPE_Q2_0, OutputType>(
                context, args, output, stream);
            break;
        case GGML_TYPE_Q5_0:
            launch_mmq_type<GGML_TYPE_Q5_0, OutputType>(
                context, args, output, stream);
            break;
        case GGML_TYPE_Q5_1:
            launch_mmq_type<GGML_TYPE_Q5_1, OutputType>(
                context, args, output, stream);
            break;
        case GGML_TYPE_Q2_K:
            launch_mmq_type<GGML_TYPE_Q2_K, OutputType>(
                context, args, output, stream);
            break;
        case GGML_TYPE_IQ1_M:
            launch_mmq_type<GGML_TYPE_IQ1_M, OutputType>(
                context, args, output, stream);
            break;
        case GGML_TYPE_Q4_K:
            launch_mmq_type<GGML_TYPE_Q4_K, OutputType>(
                context, args, output, stream);
            break;
        case GGML_TYPE_Q5_K:
            launch_mmq_type<GGML_TYPE_Q5_K, OutputType>(
                context, args, output, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            launch_mmq_type<GGML_TYPE_IQ4_XS, OutputType>(
                context, args, output, stream);
            break;
        case GGML_TYPE_Q3_K:
            launch_mmq_type<GGML_TYPE_Q3_K, OutputType>(
                context, args, output, stream);
            break;
        case GGML_TYPE_Q6_K:
            launch_mmq_type<GGML_TYPE_Q6_K, OutputType>(
                context, args, output, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            launch_mmq_type<GGML_TYPE_IQ3_XXS, OutputType>(
                context, args, output, stream);
            break;
        case GGML_TYPE_IQ3_S:
            launch_mmq_type<GGML_TYPE_IQ3_S, OutputType>(
                context, args, output, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            launch_mmq_type<GGML_TYPE_IQ4_NL, OutputType>(
                context, args, output, stream);
            break;
        case GGML_TYPE_Q4_0:
            launch_mmq_type<GGML_TYPE_Q4_0, OutputType>(
                context, args, output, stream);
            break;
        case GGML_TYPE_Q4_1:
            launch_mmq_type<GGML_TYPE_Q4_1, OutputType>(
                context, args, output, stream);
            break;
        case GGML_TYPE_Q8_0:
            launch_mmq_type<GGML_TYPE_Q8_0, OutputType>(
                context, args, output, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            launch_mmq_type<GGML_TYPE_IQ2_XXS, OutputType>(
                context, args, output, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            launch_mmq_type<GGML_TYPE_IQ2_XS, OutputType>(
                context, args, output, stream);
            break;
        case GGML_TYPE_IQ2_S:
            launch_mmq_type<GGML_TYPE_IQ2_S, OutputType>(
                context, args, output, stream);
            break;
        case GGML_TYPE_IQ1_S:
            launch_mmq_type<GGML_TYPE_IQ1_S, OutputType>(
                context, args, output, stream);
            break;
        default:
            break;
    }
}

template <typename InputType, typename OutputType>
static bool matmul(
        const InputType *input, const void *weight, OutputType *output,
        ggml_type type,
        int rows, int cols, int output_cols, cudaStream_t stream) {
    if (!is_supported_type(type) || rows <= 0 || cols <= 0 ||
        cols % ggml_blck_size(type) != 0 || output_cols <= 0) {
        return false;
    }
    // Only these tile loaders guard a partial K tile. The quantized activation
    // allocation must include its zero padding as well as the row-tile guard.
    if (cols % MMQ_ITER_K != 0 &&
        type != GGML_TYPE_Q2_0 && type != GGML_TYPE_IQ4_NL) return false;
    const int padded_cols = ((cols + MMQ_ITER_K - 1) / MMQ_ITER_K) * MMQ_ITER_K;

    const int device = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[device].cc;
    const int min_rows = cc >= 1200 ?
        kBlackwellMinMmqRows : kDefaultMinMmqRows;
    if (rows < min_rows || rows > kMaxMmqRows ||
        !int8_mma_available(cc)) {
        return false;
    }

    // MMQ always loads a complete row tile. The final tile can therefore read
    // past the last real row in each K block and, for the final K block, past
    // the packed activation payload. ggml reserves one maximum-size tile as a
    // guard region for the same reason. Its values need not be initialized:
    // they belong only to output rows masked by the checked write-back path.
    const size_t quantized_payload_count =
        static_cast<size_t>(rows) * padded_cols / (4 * QK8_1);
    const size_t quantized_count = quantized_payload_count +
        static_cast<size_t>(get_mmq_x_max_host(
            ggml_cuda_info().devices[device].cc));
    const size_t output_count = static_cast<size_t>(rows) * output_cols;
    block_q8_1_mmq *quantized = nullptr;
    float *float_output = nullptr;
    if (FastllmCudaTryMalloc(reinterpret_cast<void **>(&quantized),
                             quantized_count * sizeof(block_q8_1_mmq)) !=
        FASTLLM_CUDA_TRY_MALLOC_SUCCESS) {
        return false;
    }
    if (FastllmCudaTryMalloc(reinterpret_cast<void **>(&float_output),
                             output_count * sizeof(float)) !=
        FASTLLM_CUDA_TRY_MALLOC_SUCCESS) {
        FastllmCudaFree(quantized);
        return false;
    }

    launch_quantize(type, input, quantized, rows, cols, padded_cols, stream);

    mmq_args args{};
    args.x = static_cast<const char *>(weight);
    args.y = reinterpret_cast<const char *>(quantized);
    args.dst = float_output;
    args.ne00 = padded_cols;
    args.ne01 = output_cols;
    args.stride01 = ggml_row_size(type, cols);
    args.ne10 = padded_cols;
    args.ne11 = rows;
    args.stride11 = rows;
    args.ne0 = output_cols;

    ggml_backend_cuda_context context;
    launch_mmq(type, context, args, output, stream);

    FastllmCudaFree(float_output);
    FastllmCudaFree(quantized);
    return true;
}


} // namespace fastllm_gguf_mmq

bool FastllmCudaFloatMatMulGGUFMMQ(
        const void *input, const void *weight, void *output, int weight_type,
        int n, int m, int k, void *stream) {
    return fastllm_gguf_mmq::matmul(
        static_cast<const float *>(input), weight, static_cast<float *>(output),
        static_cast<ggml_type>(weight_type), n, m, k,
        reinterpret_cast<cudaStream_t>(stream));
}

bool FastllmCudaHalfMatMulGGUFMMQ(
        const void *input, const void *weight, void *output, int weight_type,
        int n, int m, int k, void *stream) {
    return fastllm_gguf_mmq::matmul(
        static_cast<const half *>(input), weight, static_cast<half *>(output),
        static_cast<ggml_type>(weight_type), n, m, k,
        reinterpret_cast<cudaStream_t>(stream));
}

bool FastllmCudaBFloat16MatMulGGUFMMQ(
        const void *input, const void *weight, void *output, int weight_type,
        int n, int m, int k, void *stream) {
    return fastllm_gguf_mmq::matmul(
        static_cast<const __nv_bfloat16 *>(input), weight,
        static_cast<__nv_bfloat16 *>(output),
        static_cast<ggml_type>(weight_type), n, m, k,
        reinterpret_cast<cudaStream_t>(stream));
}

bool FastllmCudaHalfMatMulGGUFMMVQ(
        const void *input, const void *weight, void *output, int weight_type,
        int n, int m, int k, void *stream) {
    return fastllm_gguf_mmq::matmul_mmvq(
        static_cast<const half *>(input), weight, static_cast<half *>(output),
        static_cast<ggml_type>(weight_type), n, m, k,
        reinterpret_cast<cudaStream_t>(stream));
}

bool FastllmCudaGGUFExtendedFromQ8(const void *input, const void *weight,
        void *output, int weightType, int rows, int columns, int outputRows,
        int storeMode, void *stream) {
    using namespace fastllm_gguf_mmq;
    const auto type = static_cast<ggml_type>(weightType);
    if ((type != GGML_TYPE_IQ2_XXS && type != GGML_TYPE_IQ2_XS && type != GGML_TYPE_IQ2_S && type != GGML_TYPE_IQ1_M) ||
        !supports_mmvq_shape(type, rows, columns, outputRows) || storeMode < 0 || storeMode > 2)
        return false;
    if (type == GGML_TYPE_IQ1_M) ensure_extended_iq1s_grid(reinterpret_cast<cudaStream_t>(stream));
#define SHARED_CASE(MODE) \
    case MODE: dispatch_extended_mmvq<half, MODE>(type, weight, \
        static_cast<const block_q8_1 *>(input), static_cast<half *>(output), \
        rows, columns, outputRows, reinterpret_cast<cudaStream_t>(stream)); break
    switch (storeMode) { SHARED_CASE(0); SHARED_CASE(1); SHARED_CASE(2); }
#undef SHARED_CASE
    return true;
}

bool FastllmCudaHalfMatMulGGUFMMVQAddTo(
        const void *input, const void *weight, void *output, int weightType,
        int rows, int columns, int outputRows, void *stream) {
    return fastllm_gguf_mmq::matmul_mmvq<half, half, true>(
        static_cast<const half *>(input), weight, static_cast<half *>(output),
        static_cast<ggml_type>(weightType), rows, columns, outputRows,
        reinterpret_cast<cudaStream_t>(stream));
}

bool FastllmCudaBFloat16MatMulGGUFMMVQ(
        const void *input, const void *weight, void *output, int weight_type,
        int n, int m, int k, void *stream) {
    return fastllm_gguf_mmq::matmul_mmvq(
        static_cast<const __nv_bfloat16 *>(input), weight,
        static_cast<__nv_bfloat16 *>(output),
        static_cast<ggml_type>(weight_type), n, m, k,
        reinterpret_cast<cudaStream_t>(stream));
}

bool FastllmCudaFloatMatMulGGUFMMVQ(
        const void *input, const void *weight, void *output, int weight_type,
        int n, int m, int k, void *stream) {
    return fastllm_gguf_mmq::matmul_mmvq(
        static_cast<const float *>(input), weight,
        static_cast<float *>(output), static_cast<ggml_type>(weight_type),
        n, m, k, reinterpret_cast<cudaStream_t>(stream));
}

bool FastllmCudaHalfGgufGateUpSiluMulMMVQ(
        const void *input, const void *gate_weight, const void *up_weight,
        void *output, int weight_type, int n, int m, int k, void *stream) {
    return fastllm_gguf_mmq::gate_up_mmvq(
        static_cast<const half *>(input), gate_weight, up_weight,
        static_cast<half *>(output), static_cast<ggml_type>(weight_type),
        n, m, k, reinterpret_cast<cudaStream_t>(stream));
}
