#pragma once
#include "fastllm-gguf-mmq-common.cuh"
#include "fastllm-gguf-mmq-output.cuh"
#include <cuda_bf16.h>
#include <mutex>

namespace fastllm_gguf_mmq {
#include "mmq.cuh"
// Defined only in the IQ1 instantiation unit; other types discard the call.
static void ensure_iq1s_grid(cudaStream_t stream);
constexpr int kBlackwellDirectTileThreshold = 1000;

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
            launch_mmq_output_conversion(args.dst, output,
                (size_t)args.ne11 * args.ne0, stream);
            return;
        }
    }

    // matmul() admits only NVIDIA INT8 MMA devices, all of which use stream-K.
    // Instantiating the conventional launcher here also emits its unreachable
    // legacy fixup kernels for every quantization format and tile size.
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
        launch_mmq_fixup_to_output<
            ggml_cuda_type_traits<type>::qk, mmq_x, need_check>(
                args, last_tile.ptr, output, output_tiles, mmq_blocks.x, stream);
    } else {
        constexpr bool need_check = true;
        mul_mat_q<type, mmq_x, MMQ_NWARPS, need_check><<<
            mmq_blocks, threads, shared_bytes, stream>>>(
                args.x, args.y, args.dst, last_tile.ptr,
                args.ne00, args.ne01, args.stride01,
                args.ne10, args.ne11, args.stride11, args.ne0);
        launch_mmq_fixup_to_output<
            ggml_cuda_type_traits<type>::qk, mmq_x, need_check>(
                args, last_tile.ptr, output, output_tiles, mmq_blocks.x, stream);
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
