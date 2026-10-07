#pragma once
#include "fastllm-gguf-mmq-common.cuh"
#include <cuda_bf16.h>
#include <mutex>
#include <type_traits>

namespace fastllm_gguf_mmq {
#include "mmq.cuh"
#include "fastllm-gguf-mmq-io.cuh"
// Defined only in the IQ1 instantiation unit; other types discard the call.
static void ensure_iq1s_grid(cudaStream_t stream);
constexpr int kBlackwellDirectTileThreshold = 1000;

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

template <ggml_type type, typename OutputType>
void launch_mmq_type(
        ggml_backend_cuda_context &context, const mmq_args &args,
        OutputType *output, cudaStream_t stream) {
    if constexpr (type == GGML_TYPE_IQ1_S) ensure_iq1s_grid(stream);
    // Match the token tile to small verifier and prefill batches. A fixed
    // 128-row tile also computes the rows masked out during write-back.
    if (args.ne11 <= 8) {
        launch_mul_mat_q_to_output<type, 8, OutputType>(
            context, args, output, stream);
    } else if (args.ne11 <= 32) {
        launch_mul_mat_q_to_output<type, 32, OutputType>(
            context, args, output, stream);
    } else if (args.ne11 <= 64) {
        launch_mul_mat_q_to_output<type, 64, OutputType>(
            context, args, output, stream);
    } else {
        launch_mul_mat_q_to_output<type, 128, OutputType>(
            context, args, output, stream);
    }
}


// Keep each quantization family in a separate translation unit. The public
// dispatcher references these explicit instantiations without recompiling them.
#define FASTLLM_INSTANTIATE_MMQ(TYPE) \
    template void launch_mmq_type<TYPE, float>(ggml_backend_cuda_context &, const mmq_args &, float *, cudaStream_t); \
    template void launch_mmq_type<TYPE, half>(ggml_backend_cuda_context &, const mmq_args &, half *, cudaStream_t); \
    template void launch_mmq_type<TYPE, __nv_bfloat16>(ggml_backend_cuda_context &, const mmq_args &, __nv_bfloat16 *, cudaStream_t);
} // namespace fastllm_gguf_mmq
