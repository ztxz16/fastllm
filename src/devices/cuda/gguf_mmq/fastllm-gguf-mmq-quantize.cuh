#pragma once
// Included inside fastllm_gguf_mmq. Shared MMQ activation packing for LLM and image backends.
constexpr int kQuantizeBlockSize = 128;

template <mmq_q8_1_ds_layout layout, typename InputType, bool QuantizedSum = false>
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
    if constexpr (layout != MMQ_Q8_1_DS_LAYOUT_D4 && !QuantizedSum) {
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
    if constexpr (QuantizedSum) {
        // Q5_K MMVQ applies the minimum term to quantized activations. Match
        // that arithmetic instead of using the unquantized activation sum.
        int isum = int(quants.x) + int(quants.y) + int(quants.z) + int(quants.w);
        #pragma unroll
        for (int mask = 4; mask > 0; mask >>= 1)
            isum += __shfl_xor_sync(0xffffffff, isum, mask, WARP_SIZE);
        sum = __half2float(__float2half(scale)) * float(isum);
    }

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
