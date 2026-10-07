#pragma once
#include "fastllm-gguf-mmvq-dispatch.cuh"
#include "../fastllm-gguf-store.cuh"
#include <cuda_bf16.h>
#include <mutex>
#include <type_traits>

namespace fastllm_gguf_mmq {
#include "vecdotq.cuh"
#include "fastllm-gguf-iq2-gemv.cuh"
#include "../fastllm-gguf-small-mmvq.cuh"

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

template <ggml_type type, int input_rows, int nwarps, typename OutputType, int StoreMode = 0>
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
            FastllmGgufStore<StoreMode>(output + static_cast<size_t>(input_row) * output_rows +
                   output_row0 + threadIdx.x, sums[input_row][threadIdx.x]);
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

template <ggml_type type, int nwarps, typename OutputType, int StoreMode = 0>
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
    mul_mat_vec_extended<type, row_count, nwarps, OutputType, StoreMode>            \
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

template <ggml_type type, typename OutputType, int StoreMode>
void launch_extended_mmvq_type(
        const void *weight, const block_q8_1 *input, OutputType *output,
        int rows, int input_columns, int output_rows,
        cudaStream_t stream) {
    if constexpr (type == GGML_TYPE_IQ1_M || type == GGML_TYPE_IQ2_XS ||
                  type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_S) {
        if (rows >= 2 && rows <= 8 && fastllm_gguf_small_mmvq::Supports(
                input, input_columns, output_rows, input_columns, output_rows)) {
            fastllm_gguf_small_mmvq::LaunchBatch<type, OutputType, StoreMode>(weight, input, output,
                input_columns, output_rows, rows, input_columns, output_rows, stream);
            return;
        }
    }
    if constexpr (type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_S) {
        if (rows == 1 && iq2_decode::Supports(input, input_columns, output_rows)) {
            iq2_decode::Launch<type, false, OutputType, StoreMode>(weight, nullptr, input, output,
                                          input_columns, output_rows, stream);
            return;
        }
    }
    const int nwarps = rows <= 4 || rows < fastllm::FastllmCudaGetLinearExactBatchThreshold() ? 4 : 1;
    if (nwarps == 4) {
        launch_extended_mmvq_rows<type, 4, OutputType, StoreMode>(
            weight, input, output, rows, input_columns, output_rows,
            stream);
    } else {
        launch_extended_mmvq_rows<type, 1, OutputType, StoreMode>(
            weight, input, output, rows, input_columns, output_rows,
            stream);
    }
}

template <ggml_type type>
void launch_extended_gate_up_type(
        const void *gate_weight, const void *up_weight,
        const block_q8_1 *input, half *output, int rows,
        int input_columns, int output_rows, cudaStream_t stream) {
    if constexpr (type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_S) {
        if (rows == 1 && iq2_decode::Supports(input, input_columns, output_rows)) {
            iq2_decode::Launch<type, true>(gate_weight, up_weight, input, output,
                                          input_columns, output_rows, stream);
            return;
        }
    }
    const int nwarps = rows <= 4 || rows < fastllm::FastllmCudaGetLinearExactBatchThreshold() ? 4 : 1;
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

#define FASTLLM_INSTANTIATE_EXTENDED_MMVQ_OUTPUT(Type, Output, Store) \
    template void launch_extended_mmvq_type<Type, Output, Store>( \
        const void *, const block_q8_1 *, Output *, int, int, int, cudaStream_t);

#define FASTLLM_INSTANTIATE_EXTENDED_MMVQ(Type) \
    FASTLLM_INSTANTIATE_EXTENDED_MMVQ_OUTPUT(Type, float, 0) \
    FASTLLM_INSTANTIATE_EXTENDED_MMVQ_OUTPUT(Type, half, 0) \
    FASTLLM_INSTANTIATE_EXTENDED_MMVQ_OUTPUT(Type, __nv_bfloat16, 0)

// Only IQ2 and IQ1_M are accepted by the fused-store public entry points.
#define FASTLLM_INSTANTIATE_EXTENDED_MMVQ_STORES(Type) \
    FASTLLM_INSTANTIATE_EXTENDED_MMVQ_OUTPUT(Type, half, 1) \
    FASTLLM_INSTANTIATE_EXTENDED_MMVQ_OUTPUT(Type, half, 2)

#define FASTLLM_INSTANTIATE_EXTENDED_GATE_UP(Type) \
    template void launch_extended_gate_up_type<Type>( \
        const void *, const void *, const block_q8_1 *, half *, int, int, int, cudaStream_t);
} // namespace fastllm_gguf_mmq
