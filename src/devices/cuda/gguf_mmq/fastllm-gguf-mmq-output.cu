#include "fastllm-gguf-mmq-common.cuh"
#include "fastllm-gguf-mmq-output.cuh"
#include <cuda_bf16.h>

namespace fastllm_gguf_mmq {
#include "mmq.cuh"
#include "fastllm-gguf-mmq-io.cuh"

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

template <int qk, int mmq_x, int nwarps, bool need_check,
          typename OutputType>
static __global__ void stream_k_fixup_to_output(
        const float *__restrict__ source,
        const float *__restrict__ last_tile,
        OutputType *__restrict__ output,
        int ne00, int ne01, int ne11, int ne0, int block_num_mmq) {
    constexpr int mmq_y = get_mmq_y_device();
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

template <int qk, int mmq_x, bool need_check, typename OutputType>
void launch_mmq_fixup_to_output(
        const mmq_args &args, const float *last_tile, OutputType *output,
        dim3 output_tiles, int block_num_mmq, cudaStream_t stream) {
    const dim3 threads(WARP_SIZE, MMQ_NWARPS, 1);
    stream_k_fixup_to_output<qk, mmq_x, MMQ_NWARPS, need_check, OutputType>
        <<<output_tiles, threads, 0, stream>>>(
            args.dst, last_tile, output,
            args.ne00, args.ne01, args.ne11, args.ne0, block_num_mmq);
}

template <typename OutputType>
void launch_mmq_output_conversion(
        const float *input, OutputType *output, int64_t count, cudaStream_t stream) {
    constexpr int threads = 256;
    convert_float_to_output<OutputType><<<
        (count + threads - 1) / threads, threads, 0, stream>>>(input, output, count);
}

#define FASTLLM_INSTANTIATE_MMQ_FIXUP(QK, X, CHECK, OUTPUT) \
    template void launch_mmq_fixup_to_output<QK, X, CHECK, OUTPUT>( \
        const mmq_args &, const float *, OUTPUT *, dim3, int, cudaStream_t);
#define FASTLLM_INSTANTIATE_MMQ_FIXUP_OUTPUTS(QK, X, CHECK) \
    FASTLLM_INSTANTIATE_MMQ_FIXUP(QK, X, CHECK, float) \
    FASTLLM_INSTANTIATE_MMQ_FIXUP(QK, X, CHECK, half) \
    FASTLLM_INSTANTIATE_MMQ_FIXUP(QK, X, CHECK, __nv_bfloat16)
#define FASTLLM_INSTANTIATE_MMQ_FIXUP_TILE(QK, X) \
    FASTLLM_INSTANTIATE_MMQ_FIXUP_OUTPUTS(QK, X, false) \
    FASTLLM_INSTANTIATE_MMQ_FIXUP_OUTPUTS(QK, X, true)
#define FASTLLM_INSTANTIATE_MMQ_FIXUP_BLOCK(QK) \
    FASTLLM_INSTANTIATE_MMQ_FIXUP_TILE(QK, 8) \
    FASTLLM_INSTANTIATE_MMQ_FIXUP_TILE(QK, 32) \
    FASTLLM_INSTANTIATE_MMQ_FIXUP_TILE(QK, 64) \
    FASTLLM_INSTANTIATE_MMQ_FIXUP_TILE(QK, 128)

FASTLLM_INSTANTIATE_MMQ_FIXUP_BLOCK(32)
FASTLLM_INSTANTIATE_MMQ_FIXUP_BLOCK(64)
FASTLLM_INSTANTIATE_MMQ_FIXUP_BLOCK(256)

#undef FASTLLM_INSTANTIATE_MMQ_FIXUP_BLOCK
#undef FASTLLM_INSTANTIATE_MMQ_FIXUP_TILE
#undef FASTLLM_INSTANTIATE_MMQ_FIXUP_OUTPUTS
#undef FASTLLM_INSTANTIATE_MMQ_FIXUP

template void launch_mmq_output_conversion<float>(const float *, float *, int64_t, cudaStream_t);
template void launch_mmq_output_conversion<half>(const float *, half *, int64_t, cudaStream_t);
template void launch_mmq_output_conversion<__nv_bfloat16>(const float *, __nv_bfloat16 *, int64_t, cudaStream_t);
} // namespace fastllm_gguf_mmq
