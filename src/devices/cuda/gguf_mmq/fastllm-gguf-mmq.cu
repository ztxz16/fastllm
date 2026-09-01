#include "fastllm-gguf-mmq-common.cuh"

#include <cuda_bf16.h>

#include <mutex>

namespace fastllm_gguf_mmq {

#include "mmq.cuh"

constexpr int kQuantizeBlockSize = 128;
constexpr int kBlackwellDirectTileThreshold = 1000;
constexpr int kBlackwellMinMmqRows = 8;
constexpr int kDefaultMinMmqRows = 9;
constexpr int kMaxMmqRows = 1024;

template <typename T>
struct mmq_io;

template <>
struct mmq_io<half> {
    static __device__ __forceinline__ float to_float(half value) {
        return __half2float(value);
    }
    static __device__ __forceinline__ half from_float(float value) {
        return __float2half_rn(value);
    }
};

template <>
struct mmq_io<__nv_bfloat16> {
    static __device__ __forceinline__ float to_float(__nv_bfloat16 value) {
        return __bfloat162float(value);
    }
    static __device__ __forceinline__ __nv_bfloat16 from_float(float value) {
        return __float2bfloat16_rn(value);
    }
};

template <>
struct mmq_io<float> {
    static __device__ __forceinline__ float to_float(float value) {
        return value;
    }
    static __device__ __forceinline__ float from_float(float value) {
        return value;
    }
};

static __global__ void initialize_iq1s_grid_gpu() {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= NGRID_IQ1S) {
        return;
    }

    // The canonical table stores eight signed {-1, 0, 1} bytes in a uint64.
    // ik_llama's CUDA table places values 0..3 in the low nibble of four
    // bytes and values 4..7 in their high nibble.  vec_dot_iq1_* extracts
    // those two four-value vectors with masks before feeding them to dp4a.
    const uint64_t source = iq1s_grid[index];
    uint32_t packed = 0;
#pragma unroll
    for (int value = 0; value < 8; ++value) {
        const int8_t signed_value =
            static_cast<int8_t>(source >> (8 * value));
        const int shift = value < 4 ? 8 * value : 8 * (value - 4) + 4;
        packed |= static_cast<uint32_t>(signed_value + 1) << shift;
    }
    iq1s_grid_gpu[index] = packed;
}

static void ensure_iq1s_grid(cudaStream_t stream) {
    static std::once_flag initialized[GGML_CUDA_MAX_DEVICES];
    const int device = ggml_cuda_get_device();
    std::call_once(initialized[device], [stream]() {
        constexpr int threads = 256;
        initialize_iq1s_grid_gpu<<<
            (NGRID_IQ1S + threads - 1) / threads, threads, 0, stream>>>();
        CUDA_CHECK(cudaGetLastError());
        // Initialization is outside steady-state execution and must be visible
        // to every per-thread stream that can subsequently launch an IQ1 op.
        CUDA_CHECK(cudaStreamSynchronize(stream));
    });
}

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

template <ggml_type type>
struct fastllm_mmvq_type_traits;

#define FASTLLM_GGUF_MMVQ_TRAITS(type_name, vdr_value, dot_function)       \
    template <>                                                            \
    struct fastllm_mmvq_type_traits<type_name> {                           \
        static constexpr int vdr = vdr_value;                              \
        static __device__ __forceinline__ float dot(                       \
                const void *weight, const block_q8_1 *input,               \
                const int &block, const int &quant) {                       \
            return dot_function(weight, input, block, quant);              \
        }                                                                  \
    }

FASTLLM_GGUF_MMVQ_TRAITS(
    GGML_TYPE_Q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1);
FASTLLM_GGUF_MMVQ_TRAITS(
    GGML_TYPE_Q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1);
FASTLLM_GGUF_MMVQ_TRAITS(
    GGML_TYPE_IQ2_XXS, VDR_IQ2_XXS_Q8_1_MMVQ,
    vec_dot_iq2_xxs_q8_1);
FASTLLM_GGUF_MMVQ_TRAITS(
    GGML_TYPE_IQ2_XS, VDR_IQ2_XS_Q8_1_MMVQ,
    vec_dot_iq2_xs_q8_1);
FASTLLM_GGUF_MMVQ_TRAITS(
    GGML_TYPE_IQ2_S, VDR_IQ2_S_Q8_1_MMVQ, vec_dot_iq2_s_q8_1);
FASTLLM_GGUF_MMVQ_TRAITS(
    GGML_TYPE_IQ1_S, VDR_IQ1_S_Q8_1_MMVQ, vec_dot_iq1_s_q8_1);
FASTLLM_GGUF_MMVQ_TRAITS(
    GGML_TYPE_IQ1_M, VDR_IQ1_M_Q8_1_MMVQ, vec_dot_iq1_m_q8_1);

#undef FASTLLM_GGUF_MMVQ_TRAITS

template <typename InputType>
static __global__ void quantize_mmvq_q8_1(
        const InputType *__restrict__ input,
        block_q8_1 *__restrict__ quantized, int cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) {
        return;
    }
    const int row = blockIdx.y;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const int block = row * (cols / QK8_1) + col / QK8_1;
    const float value = mmq_io<InputType>::to_float(
        input[static_cast<size_t>(row) * cols + col]);

    float abs_max = warp_reduce_max(fabsf(value));
    float sum = warp_reduce_sum(value);
    const float scale = abs_max / 127.0f;
    quantized[block].qs[lane] = static_cast<int8_t>(
        abs_max == 0.0f ? 0 : roundf(value / scale));
    if (lane == 0) {
        quantized[block].ds = make_half2(
            __float2half(scale), __float2half(sum));
    }
}

template <ggml_type type, int input_rows, int nwarps, typename OutputType>
__launch_bounds__(nwarps * WARP_SIZE, 1)
static __global__ void mul_mat_vec_extended(
        const void *__restrict__ weight,
        const block_q8_1 *__restrict__ input,
        OutputType *__restrict__ output, int input_columns,
        int output_rows) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = fastllm_mmvq_type_traits<type>::vdr;
    constexpr int rows_per_block = input_rows < 4 ? 1 : 2;
    constexpr int blocks_per_iteration =
        vdr * nwarps * WARP_SIZE / qi;

    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
    const int output_row0 = rows_per_block * blockIdx.x;
    const int weight_blocks_per_row = input_columns / qk;
    const int input_blocks_per_row = input_columns / QK8_1;
    float sums[input_rows][rows_per_block] = {0.0f};

    for (int weight_block = tid / (qi / vdr);
         weight_block < weight_blocks_per_row;
         weight_block += blocks_per_iteration) {
        const int input_block = weight_block * (qk / QK8_1);
        const int quant = vdr * (tid % (qi / vdr));
#pragma unroll
        for (int input_row = 0; input_row < input_rows; ++input_row) {
#pragma unroll
            for (int output_offset = 0;
                 output_offset < rows_per_block; ++output_offset) {
                if (output_row0 + output_offset < output_rows) {
                    sums[input_row][output_offset] +=
                        fastllm_mmvq_type_traits<type>::dot(
                            weight,
                            input + input_row * input_blocks_per_row +
                                input_block,
                            (output_row0 + output_offset) *
                                weight_blocks_per_row + weight_block,
                            quant);
                }
            }
        }
    }

    __shared__ float partial[nwarps > 1 ? nwarps - 1 : 1]
                            [input_rows][rows_per_block][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int input_row = 0; input_row < input_rows; ++input_row) {
#pragma unroll
            for (int output_offset = 0;
                 output_offset < rows_per_block; ++output_offset) {
                partial[threadIdx.y - 1][input_row][output_offset]
                       [threadIdx.x] = sums[input_row][output_offset];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

#pragma unroll
    for (int input_row = 0; input_row < input_rows; ++input_row) {
#pragma unroll
        for (int output_offset = 0;
             output_offset < rows_per_block; ++output_offset) {
#pragma unroll
            for (int warp = 0; warp < nwarps - 1; ++warp) {
                sums[input_row][output_offset] +=
                    partial[warp][input_row][output_offset][threadIdx.x];
            }
            sums[input_row][output_offset] =
                warp_reduce_sum(sums[input_row][output_offset]);
        }

        if (threadIdx.x < rows_per_block &&
            output_row0 + threadIdx.x < output_rows) {
            output[static_cast<size_t>(input_row) * output_rows +
                   output_row0 + threadIdx.x] =
                mmq_io<OutputType>::from_float(
                    sums[input_row][threadIdx.x]);
        }
    }
}

template <ggml_type type, int input_rows, int nwarps>
__launch_bounds__(nwarps * WARP_SIZE, 1)
static __global__ void mul_mat_vec_gate_up_extended(
        const void *__restrict__ gate_weight,
        const void *__restrict__ up_weight,
        const block_q8_1 *__restrict__ input,
        half *__restrict__ output, int input_columns, int output_rows) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int qi = ggml_cuda_type_traits<type>::qi;
    constexpr int vdr = fastllm_mmvq_type_traits<type>::vdr;
    constexpr int rows_per_block = input_rows < 4 ? 1 : 2;
    constexpr int blocks_per_iteration =
        vdr * nwarps * WARP_SIZE / qi;

    const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
    const int output_row0 = rows_per_block * blockIdx.x;
    const int weight_blocks_per_row = input_columns / qk;
    const int input_blocks_per_row = input_columns / QK8_1;
    float gate_sums[input_rows][rows_per_block] = {0.0f};
    float up_sums[input_rows][rows_per_block] = {0.0f};

    for (int weight_block = tid / (qi / vdr);
         weight_block < weight_blocks_per_row;
         weight_block += blocks_per_iteration) {
        const int input_block = weight_block * (qk / QK8_1);
        const int quant = vdr * (tid % (qi / vdr));
#pragma unroll
        for (int input_row = 0; input_row < input_rows; ++input_row) {
#pragma unroll
            for (int output_offset = 0;
                 output_offset < rows_per_block; ++output_offset) {
                if (output_row0 + output_offset < output_rows) {
                    const block_q8_1 *row_input =
                        input + input_row * input_blocks_per_row +
                        input_block;
                    const int packed_row =
                        (output_row0 + output_offset) *
                            weight_blocks_per_row + weight_block;
                    gate_sums[input_row][output_offset] +=
                        fastllm_mmvq_type_traits<type>::dot(
                            gate_weight, row_input, packed_row, quant);
                    up_sums[input_row][output_offset] +=
                        fastllm_mmvq_type_traits<type>::dot(
                            up_weight, row_input, packed_row, quant);
                }
            }
        }
    }

    __shared__ float gate_partial[nwarps > 1 ? nwarps - 1 : 1]
                                 [input_rows][rows_per_block][WARP_SIZE];
    __shared__ float up_partial[nwarps > 1 ? nwarps - 1 : 1]
                               [input_rows][rows_per_block][WARP_SIZE];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int input_row = 0; input_row < input_rows; ++input_row) {
#pragma unroll
            for (int output_offset = 0;
                 output_offset < rows_per_block; ++output_offset) {
                gate_partial[threadIdx.y - 1][input_row][output_offset]
                            [threadIdx.x] =
                    gate_sums[input_row][output_offset];
                up_partial[threadIdx.y - 1][input_row][output_offset]
                          [threadIdx.x] =
                    up_sums[input_row][output_offset];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) {
        return;
    }

#pragma unroll
    for (int input_row = 0; input_row < input_rows; ++input_row) {
#pragma unroll
        for (int output_offset = 0;
             output_offset < rows_per_block; ++output_offset) {
#pragma unroll
            for (int warp = 0; warp < nwarps - 1; ++warp) {
                gate_sums[input_row][output_offset] +=
                    gate_partial[warp][input_row][output_offset]
                                [threadIdx.x];
                up_sums[input_row][output_offset] +=
                    up_partial[warp][input_row][output_offset]
                              [threadIdx.x];
            }
            gate_sums[input_row][output_offset] =
                warp_reduce_sum(gate_sums[input_row][output_offset]);
            up_sums[input_row][output_offset] =
                warp_reduce_sum(up_sums[input_row][output_offset]);
        }

        if (threadIdx.x < rows_per_block &&
            output_row0 + threadIdx.x < output_rows) {
            const half gate =
                __float2half_rn(gate_sums[input_row][threadIdx.x]);
            const half up =
                __float2half_rn(up_sums[input_row][threadIdx.x]);
            const half activated = __hdiv(
                gate, __hadd(__float2half(1.0f), hexp(-gate)));
            output[static_cast<size_t>(input_row) * output_rows +
                   output_row0 + threadIdx.x] = __hmul(activated, up);
        }
    }
}

template <ggml_type type, int nwarps, typename OutputType>
static void launch_extended_mmvq_rows(
        const void *weight, const block_q8_1 *input, OutputType *output,
        int rows, int input_columns, int output_rows,
        cudaStream_t stream) {
    constexpr int threads_x = WARP_SIZE;
    const int rows_per_block = rows < 4 ? 1 : 2;
    const dim3 blocks(
        (output_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    const dim3 threads(threads_x, nwarps, 1);
#define FASTLLM_LAUNCH_MMVQ_ROWS(row_count)                              \
    mul_mat_vec_extended<type, row_count, nwarps, OutputType>            \
        <<<blocks, threads, 0, stream>>>(                                 \
            weight, input, output, input_columns, output_rows)
    switch (rows) {
        case 1: FASTLLM_LAUNCH_MMVQ_ROWS(1); break;
        case 2: FASTLLM_LAUNCH_MMVQ_ROWS(2); break;
        case 3: FASTLLM_LAUNCH_MMVQ_ROWS(3); break;
        case 4: FASTLLM_LAUNCH_MMVQ_ROWS(4); break;
        case 5: FASTLLM_LAUNCH_MMVQ_ROWS(5); break;
        case 6: FASTLLM_LAUNCH_MMVQ_ROWS(6); break;
        case 7: FASTLLM_LAUNCH_MMVQ_ROWS(7); break;
        case 8: FASTLLM_LAUNCH_MMVQ_ROWS(8); break;
        default: break;
    }
#undef FASTLLM_LAUNCH_MMVQ_ROWS
}

template <ggml_type type, int nwarps>
static void launch_extended_gate_up_rows(
        const void *gate_weight, const void *up_weight,
        const block_q8_1 *input, half *output, int rows,
        int input_columns, int output_rows, cudaStream_t stream) {
    const int rows_per_block = rows < 4 ? 1 : 2;
    const dim3 blocks(
        (output_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    const dim3 threads(WARP_SIZE, nwarps, 1);
#define FASTLLM_LAUNCH_GATE_UP_ROWS(row_count)                           \
    mul_mat_vec_gate_up_extended<type, row_count, nwarps>                \
        <<<blocks, threads, 0, stream>>>(                                 \
            gate_weight, up_weight, input, output,                       \
            input_columns, output_rows)
    switch (rows) {
        case 1: FASTLLM_LAUNCH_GATE_UP_ROWS(1); break;
        case 2: FASTLLM_LAUNCH_GATE_UP_ROWS(2); break;
        case 3: FASTLLM_LAUNCH_GATE_UP_ROWS(3); break;
        case 4: FASTLLM_LAUNCH_GATE_UP_ROWS(4); break;
        case 5: FASTLLM_LAUNCH_GATE_UP_ROWS(5); break;
        case 6: FASTLLM_LAUNCH_GATE_UP_ROWS(6); break;
        case 7: FASTLLM_LAUNCH_GATE_UP_ROWS(7); break;
        case 8: FASTLLM_LAUNCH_GATE_UP_ROWS(8); break;
        default: break;
    }
#undef FASTLLM_LAUNCH_GATE_UP_ROWS
}

template <ggml_type type, typename OutputType>
static void launch_extended_mmvq_type(
        const void *weight, const block_q8_1 *input, OutputType *output,
        int rows, int input_columns, int output_rows,
        cudaStream_t stream) {
    const int nwarps = rows <= 4 ? 4 : 1;
    if (nwarps == 4) {
        launch_extended_mmvq_rows<type, 4>(
            weight, input, output, rows, input_columns, output_rows,
            stream);
    } else {
        launch_extended_mmvq_rows<type, 1>(
            weight, input, output, rows, input_columns, output_rows,
            stream);
    }
}

template <ggml_type type>
static void launch_extended_gate_up_type(
        const void *gate_weight, const void *up_weight,
        const block_q8_1 *input, half *output, int rows,
        int input_columns, int output_rows, cudaStream_t stream) {
    const int nwarps = rows <= 4 ? 4 : 1;
    if (nwarps == 4) {
        launch_extended_gate_up_rows<type, 4>(
            gate_weight, up_weight, input, output, rows,
            input_columns, output_rows, stream);
    } else {
        launch_extended_gate_up_rows<type, 1>(
            gate_weight, up_weight, input, output, rows,
            input_columns, output_rows, stream);
    }
}

template <typename OutputType>
static void dispatch_extended_mmvq(
        ggml_type type, const void *weight, const block_q8_1 *input,
        OutputType *output, int rows, int input_columns, int output_rows,
        cudaStream_t stream) {
#define FASTLLM_DISPATCH_EXTENDED_MMVQ(type_name)                        \
    case type_name:                                                       \
        launch_extended_mmvq_type<type_name>(                             \
            weight, input, output, rows, input_columns, output_rows,      \
            stream);                                                      \
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
        FASTLLM_DISPATCH_EXTENDED_GATE_UP(GGML_TYPE_Q4_0);
        FASTLLM_DISPATCH_EXTENDED_GATE_UP(GGML_TYPE_Q4_1);
        FASTLLM_DISPATCH_EXTENDED_GATE_UP(GGML_TYPE_IQ2_XXS);
        FASTLLM_DISPATCH_EXTENDED_GATE_UP(GGML_TYPE_IQ2_XS);
        FASTLLM_DISPATCH_EXTENDED_GATE_UP(GGML_TYPE_IQ2_S);
        FASTLLM_DISPATCH_EXTENDED_GATE_UP(GGML_TYPE_IQ1_S);
        FASTLLM_DISPATCH_EXTENDED_GATE_UP(GGML_TYPE_IQ1_M);
        default: break;
    }
#undef FASTLLM_DISPATCH_EXTENDED_GATE_UP
}

template <typename InputType, typename OutputType>
static bool matmul_mmvq(
        const InputType *input, const void *weight, OutputType *output,
        ggml_type type, int rows, int input_columns, int output_rows,
        cudaStream_t stream) {
    if (!is_extended_mmvq_type(type) || rows <= 0 || rows > 8 ||
        input_columns <= 0 || input_columns % QK8_1 != 0 ||
        output_rows <= 0) {
        return false;
    }
    if (type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M) {
        ensure_iq1s_grid(stream);
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

    dispatch_extended_mmvq(
        type, weight, quantized, output, rows, input_columns, output_rows,
        stream);
    FastllmCudaFree(quantized);
    return true;
}

static bool gate_up_mmvq(
        const half *input, const void *gate_weight, const void *up_weight,
        half *output, ggml_type type, int rows, int input_columns,
        int output_rows, cudaStream_t stream) {
    if (!is_extended_mmvq_type(type) || rows <= 0 || rows > 8 ||
        input_columns <= 0 || input_columns % QK8_1 != 0 ||
        output_rows <= 0) {
        return false;
    }
    if (type == GGML_TYPE_IQ1_S || type == GGML_TYPE_IQ1_M) {
        ensure_iq1s_grid(stream);
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

template <typename OutputType>
static __global__ void convert_float_to_output(
        const float *__restrict__ input, OutputType *__restrict__ output,
        int64_t count) {
    const int64_t index =
        static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (index < count) {
        output[index] = mmq_io<OutputType>::from_float(input[index]);
    }
}

template <ggml_type type, int mmq_x, int nwarps, bool need_check>
static __global__ void mul_mat_q_xy(
        const char *__restrict__ x, const char *__restrict__ y,
        float *__restrict__ output, int ne00, int ne01, int stride01,
        int ne10, int ne11, int stride11, int ne0) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr bool fixup = false;
    mul_mat_q_process_tile<type, mmq_x, nwarps, need_check, fixup>(
        x, y, output, nullptr, ne00, ne01, stride01, ne10, ne11,
        stride11, ne0, blockIdx.x, blockIdx.y, 0, ne00 / qk);
}

template <ggml_type type, int mmq_x, int nwarps, bool need_check,
          typename OutputType>
static __global__ void stream_k_fixup_to_output(
        const float *__restrict__ source,
        const float *__restrict__ last_tile,
        OutputType *__restrict__ output,
        int ne00, int ne01, int ne11, int ne0, int block_num_mmq) {
    constexpr int mmq_y = get_mmq_y_device();
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int blocks_per_iter = MMQ_ITER_K / qk;
    const int64_t blocks_per_ne00 = ne00 / qk;

    float sum[mmq_x * mmq_y / (nwarps * WARP_SIZE)] = {0.0f};
    const int ntx = (ne11 + mmq_x - 1) / mmq_x;
    const int nty = (ne01 + mmq_y - 1) / mmq_y;
    bool any_fixup = false;

    const int bidx_start =
        ((blockIdx.y * nty + blockIdx.x) * block_num_mmq) /
        (gridDim.y * gridDim.x);
    const int bidx_stop =
        ((blockIdx.y * nty + blockIdx.x + 1) * block_num_mmq +
         gridDim.y * gridDim.x - 1) /
        (gridDim.y * gridDim.x);

    int64_t kbc_0;
    int64_t kbc_stop_0 =
        (int64_t)bidx_start * blocks_per_ne00 * ntx * nty /
        block_num_mmq;
    for (int bidx = bidx_start; bidx < bidx_stop; ++bidx) {
        kbc_0 = kbc_stop_0;
        kbc_stop_0 =
            (int64_t)(bidx + 1) * blocks_per_ne00 * ntx * nty /
            block_num_mmq;

        const int64_t kbc = kbc_0 -
            (kbc_0 % blocks_per_ne00) % blocks_per_iter;
        const int64_t kbc_stop = kbc_stop_0 -
            (kbc_stop_0 % blocks_per_ne00) % blocks_per_iter;
        if (kbc == kbc_stop || kbc_stop % blocks_per_ne00 == 0) {
            continue;
        }

        const int jt = kbc_stop / (blocks_per_ne00 * nty);
        const int it =
            (kbc_stop - jt * (blocks_per_ne00 * nty)) /
            blocks_per_ne00;
        if (it != blockIdx.x || jt != blockIdx.y) {
            continue;
        }

        any_fixup = true;
#pragma unroll
        for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
            const int j = j0 + threadIdx.y;
#pragma unroll
            for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
                const int i = i0 + threadIdx.x;
                sum[(j0 / nwarps) * (mmq_y / WARP_SIZE) +
                    i0 / WARP_SIZE] +=
                    last_tile[bidx * (mmq_x * mmq_y) + j * mmq_y + i];
            }
        }
    }

    const int output_base =
        blockIdx.y * mmq_x * ne0 + blockIdx.x * mmq_y;
    const int i_max = ne01 - blockIdx.x * mmq_y - 1;
    const int j_max = ne11 - blockIdx.y * mmq_x - 1;
#pragma unroll
    for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
        const int j = j0 + threadIdx.y;
        if (j > j_max) {
            return;
        }
#pragma unroll
        for (int i0 = 0; i0 < mmq_y; i0 += WARP_SIZE) {
            const int i = i0 + threadIdx.x;
            if (need_check && i > i_max) {
                continue;
            }
            float value = source[output_base + j * ne0 + i];
            if (any_fixup) {
                value += sum[(j0 / nwarps) * (mmq_y / WARP_SIZE) +
                             i0 / WARP_SIZE];
            }
            output[output_base + j * ne0 + i] =
                mmq_io<OutputType>::from_float(value);
        }
    }
}

template <ggml_type type, int mmq_x, typename OutputType>
static void launch_mul_mat_q_to_output(
        ggml_backend_cuda_context &context, const mmq_args &args,
        OutputType *output, cudaStream_t stream) {
    const int device = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[device].cc;
    const int nsm = ggml_cuda_info().devices[device].nsm;
    const int mmq_y = get_mmq_y_host(cc);
    const dim3 threads(WARP_SIZE, MMQ_NWARPS, 1);
    const int shared_bytes = mmq_get_shmem<type>(mmq_x, mmq_y, cc);

    static bool shared_limit_raised[GGML_CUDA_MAX_DEVICES] = {false};
    if (!shared_limit_raised[device]) {
        CUDA_CHECK(cudaFuncSetAttribute(
            mul_mat_q<type, mmq_x, MMQ_NWARPS, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
        CUDA_CHECK(cudaFuncSetAttribute(
            mul_mat_q<type, mmq_x, MMQ_NWARPS, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
        if constexpr (mmq_x == 8) {
            CUDA_CHECK(cudaFuncSetAttribute(
                mul_mat_q_xy<
                    type, mmq_x, MMQ_NWARPS, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
            CUDA_CHECK(cudaFuncSetAttribute(
                mul_mat_q_xy<
                    type, mmq_x, MMQ_NWARPS, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
        }
        shared_limit_raised[device] = true;
    }

    const int nty = (args.ne01 + mmq_y - 1) / mmq_y;
    const int ntx = (args.ne11 + mmq_x - 1) / mmq_x;
    const dim3 output_tiles(nty, ntx, 1);

    // A 5090 can cover the very wide vocabulary projection directly: its
    // 1940 output tiles provide enough parallel work without stream-K.  This
    // saves one partial-tile reduction and is ~33 us faster per Q6_K head
    // projection.  Narrower matrices still need stream-K to fill all SMs.
    if constexpr (mmq_x == 8) {
        if (cc >= 1200 &&
            output_tiles.x * output_tiles.y >=
                kBlackwellDirectTileThreshold) {
            if (args.ne01 % mmq_y == 0) {
                constexpr bool need_check = false;
                mul_mat_q_xy<
                    type, mmq_x, MMQ_NWARPS, need_check><<<
                    output_tiles, threads, shared_bytes, stream>>>(
                        args.x, args.y, args.dst, args.ne00, args.ne01,
                        args.stride01, args.ne10, args.ne11, args.stride11,
                        args.ne0);
            } else {
                constexpr bool need_check = true;
                mul_mat_q_xy<
                    type, mmq_x, MMQ_NWARPS, need_check><<<
                    output_tiles, threads, shared_bytes, stream>>>(
                        args.x, args.y, args.dst, args.ne00, args.ne01,
                        args.stride01, args.ne10, args.ne11, args.stride11,
                        args.ne0);
            }
            constexpr int convert_threads = 256;
            const size_t count = (size_t)args.ne11 * args.ne0;
            convert_float_to_output<OutputType><<<
                (count + convert_threads - 1) / convert_threads,
                convert_threads, 0, stream>>>(args.dst, output, count);
            return;
        }
    }

    // The wrapper is enabled only on NVIDIA devices with INT8 MMA, which all
    // use the stream-K path. Keep a defensive conventional fallback for any
    // future backend that reuses this translation unit.
    if (!(cc >= CC_VOLTA && cc < CC_OFFSET_AMD)) {
        launch_mul_mat_q<type, mmq_x>(context, args, stream);
        constexpr int convert_threads = 256;
        const size_t count = (size_t)args.ne11 * args.ne0;
        convert_float_to_output<OutputType><<<
            (count + convert_threads - 1) / convert_threads,
            convert_threads, 0, stream>>>(args.dst, output, count);
        return;
    }

    const dim3 mmq_blocks(nsm, 1, 1);
    ggml_cuda_pool_alloc<float> last_tile(
        context.pool(device), mmq_blocks.x * mmq_x * mmq_y);
    if (args.ne01 % mmq_y == 0) {
        constexpr bool need_check = false;
        mul_mat_q<type, mmq_x, MMQ_NWARPS, need_check><<<
            mmq_blocks, threads, shared_bytes, stream>>>(
                args.x, args.y, args.dst, last_tile.ptr,
                args.ne00, args.ne01, args.stride01,
                args.ne10, args.ne11, args.stride11, args.ne0);
        stream_k_fixup_to_output<
            type, mmq_x, MMQ_NWARPS, need_check, OutputType><<<
            output_tiles, threads, 0, stream>>>(
                args.dst, last_tile.ptr, output,
                args.ne00, args.ne01, args.ne11, args.ne0, mmq_blocks.x);
    } else {
        constexpr bool need_check = true;
        mul_mat_q<type, mmq_x, MMQ_NWARPS, need_check><<<
            mmq_blocks, threads, shared_bytes, stream>>>(
                args.x, args.y, args.dst, last_tile.ptr,
                args.ne00, args.ne01, args.stride01,
                args.ne10, args.ne11, args.stride11, args.ne0);
        stream_k_fixup_to_output<
            type, mmq_x, MMQ_NWARPS, need_check, OutputType><<<
            output_tiles, threads, 0, stream>>>(
                args.dst, last_tile.ptr, output,
                args.ne00, args.ne01, args.ne11, args.ne0, mmq_blocks.x);
    }
}

static bool is_supported_type(ggml_type type) {
    return type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ||
           type == GGML_TYPE_IQ4_XS || type == GGML_TYPE_Q3_K ||
           type == GGML_TYPE_Q6_K || type == GGML_TYPE_IQ3_S ||
           type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_Q4_0 ||
           type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q8_0 ||
           type == GGML_TYPE_IQ2_XXS ||
           type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S ||
           type == GGML_TYPE_IQ1_S;
}

template <typename InputType>
static void launch_quantize(
        ggml_type type, const InputType *input,
        block_q8_1_mmq *quantized,
        int rows, int cols, cudaStream_t stream) {
    const dim3 blocks((cols + 4 * kQuantizeBlockSize - 1) /
                          (4 * kQuantizeBlockSize),
                      rows, 1);
    const dim3 threads(kQuantizeBlockSize, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, InputType>
                <<<blocks, threads, 0, stream>>>(
                    input, quantized, cols, rows, cols);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4, InputType>
                <<<blocks, threads, 0, stream>>>(
                    input, quantized, cols, rows, cols);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6, InputType>
                <<<blocks, threads, 0, stream>>>(
                    input, quantized, cols, rows, cols);
            break;
    }
}

template <ggml_type type, typename OutputType>
static void launch_mmq_type(
        ggml_backend_cuda_context &context, const mmq_args &args,
        OutputType *output, cudaStream_t stream) {
    // The verifier always supplies exactly eight rows.  The imported chooser
    // would select mmq_x=8 here; using the prefill-oriented 128-row tile makes
    // the tensor-core kernel calculate 120 masked rows for every output tile.
    if (args.ne11 <= 8) {
        launch_mul_mat_q_to_output<type, 8, OutputType>(
            context, args, output, stream);
    } else {
        launch_mul_mat_q_to_output<type, 128, OutputType>(
            context, args, output, stream);
    }
}

template <typename OutputType>
static void launch_mmq(
        ggml_type type, ggml_backend_cuda_context &context,
        const mmq_args &args, OutputType *output, cudaStream_t stream) {
    switch (type) {
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
        cols % (4 * QK8_1) != 0 || output_cols <= 0) {
        return false;
    }

    if (type == GGML_TYPE_IQ1_S) {
        ensure_iq1s_grid(stream);
    }

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
        static_cast<size_t>(rows) * cols / (4 * QK8_1);
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

    launch_quantize(type, input, quantized, rows, cols, stream);

    mmq_args args{};
    args.x = static_cast<const char *>(weight);
    args.y = reinterpret_cast<const char *>(quantized);
    args.dst = float_output;
    args.ne00 = cols;
    args.ne01 = output_cols;
    args.stride01 = ggml_row_size(type, cols);
    args.ne10 = cols;
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
