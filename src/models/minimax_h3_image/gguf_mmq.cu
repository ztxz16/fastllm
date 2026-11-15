#include "quantized_backends.h"
#include "fastllm-gguf-mmq-kernels.cuh"
#include <algorithm>
#include <stdexcept>

namespace fastllm_gguf_mmq {
#include "fastllm-gguf-mmq-io.cuh"
#include "fastllm-gguf-mmq-quantize.cuh"
} // namespace fastllm_gguf_mmq

namespace fastllm::minimax_h3_image::detail {
namespace {
using minimax_h3::Tensor;
using minimax_h3::DType;
using B = __nv_bfloat16;
using namespace fastllm_gguf_mmq;

void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
__global__ void convert(const float *input, B *output, size_t count) {
    for (size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x; i < count;
         i += size_t(blockDim.x)*gridDim.x)
        output[i] = __float2bfloat16_rn(input[i]);
}

// The wide Qwen gate/up projection otherwise needs two FP32 staging
// chunks. Keep accumulation in registers and write the requested dtype once.
// Other MMQ consumers keep their existing output and arithmetic policy.
template<int X, int Y, int Warps, bool Check, ggml_type Type>
struct WideGateTraits : mmq_type_traits<X, Y, Warps, Check, Type> {
    static constexpr vec_dot_mmq_t vec_dot_mma = vec_dot_q8_1_q8_1_mma<X, Y, Warps, true>;
};
template<typename Output>
__global__ void wideGateKernel(const char *packed, const char *quantized, Output *output,
                               int columns, int outputs, int stride, int rows) {
    mul_mat_q_process_tile<GGML_TYPE_Q4_K, 128, MMQ_NWARPS, false, false,
                          WideGateTraits, true, Output>(
        packed, quantized, output, nullptr, columns, outputs, stride, columns, rows,
        rows, outputs, blockIdx.x, blockIdx.y, 0, columns/256);
}
template<typename Output>
void wideGate(const Tensor &input, const Tensor &packed, Tensor &output, int cc) {
    // Dispatch admits at most 1024 x 4096: 4.5 MiB of Q8 scratch.
    const size_t bytes = size_t(input.rows)*(input.cols/128)*sizeof(block_q8_1_mmq);
    Tensor quantized(1, int(bytes/2));
    quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4, B>
        <<<dim3((input.cols+4*kQuantizeBlockSize-1)/(4*kQuantizeBlockSize), input.rows),
           kQuantizeBlockSize, 0, cudaStreamPerThread>>>(
            reinterpret_cast<const B *>(input.data), quantized.data, input.cols, input.rows, input.cols);
    check(cudaGetLastError());
    const int shared = mmq_get_shmem<GGML_TYPE_Q4_K>(128, get_mmq_y_host(cc), cc);
    check(cudaFuncSetAttribute(wideGateKernel<Output>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
    wideGateKernel<Output><<<dim3(output.cols/128, (input.rows+127)/128),
                             dim3(WARP_SIZE, MMQ_NWARPS), shared, cudaStreamPerThread>>>(
        reinterpret_cast<const char *>(packed.data), reinterpret_cast<const char *>(quantized.data),
        reinterpret_cast<Output *>(output.data), input.cols, output.cols, packed.cols*2, input.rows);
    check(cudaGetLastError());
}

template<ggml_type Type, int Tile, bool Check>
void multiply(const Tensor &packed, const Tensor &quantized, Tensor &accumulator,
              int columns, int cc, int rows) {
    const int tileY = get_mmq_y_host(cc);
    const int shared = mmq_get_shmem<Type>(Tile, tileY, cc);
    // No global allocator/context: all scratch belongs to the image TensorPool.
    check(cudaFuncSetAttribute(mul_mat_q_xy<Type, Tile, MMQ_NWARPS, Check>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize, shared));
    mul_mat_q_xy<Type, Tile, MMQ_NWARPS, Check>
        <<<dim3((accumulator.cols+tileY-1)/tileY, (rows+Tile-1)/Tile),
           dim3(WARP_SIZE, MMQ_NWARPS), shared, cudaStreamPerThread>>>(
            reinterpret_cast<const char *>(packed.data), reinterpret_cast<const char *>(quantized.data),
            reinterpret_cast<float *>(accumulator.data), columns, accumulator.cols, packed.cols*2,
            columns, rows, rows, accumulator.cols);
    check(cudaGetLastError());
}

template<ggml_type Type, int Tile>
void multiply(const Tensor &packed, const Tensor &quantized, Tensor &accumulator,
              int columns, int cc, int rows) {
    if (accumulator.cols % get_mmq_y_host(cc))
        multiply<Type, Tile, true>(packed, quantized, accumulator, columns, cc, rows);
    else
        multiply<Type, Tile, false>(packed, quantized, accumulator, columns, cc, rows);
}

template<ggml_type Type>
void run(const B *input, const Tensor &packed, const Tensor &quantized, Tensor &accumulator,
         int columns, const cudaDeviceProp &prop, int rows) {
    constexpr auto layout = Type == GGML_TYPE_Q2_K ? MMQ_Q8_1_DS_LAYOUT_D2S6 :
        (Type == GGML_TYPE_Q4_0 || Type == GGML_TYPE_Q4_K || Type == GGML_TYPE_Q5_K) ?
            MMQ_Q8_1_DS_LAYOUT_DS4 : MMQ_Q8_1_DS_LAYOUT_D4;
    const dim3 blocks((columns+4*kQuantizeBlockSize-1)/(4*kQuantizeBlockSize), rows);
    quantize_mmq_q8_1<layout, B><<<blocks, kQuantizeBlockSize, 0, cudaStreamPerThread>>>(
        input, quantized.data, columns, rows, columns);
    check(cudaGetLastError());
    const int cc = prop.major*100+prop.minor*10;
    if (rows <= 8) multiply<Type, 8>(packed, quantized, accumulator, columns, cc, rows);
    else if (rows <= 32) multiply<Type, 32>(packed, quantized, accumulator, columns, cc, rows);
    else if (rows <= 64 ||
             // Qwen's 4096-output projections underfill SM120 with 128-row
             // tiles at these batch sizes. The 64-row tile keeps identical math.
             (prop.major == 12 && (Type == GGML_TYPE_Q4_K || Type == GGML_TYPE_Q5_K) &&
              accumulator.cols == 4096 && (columns == 4096 || columns == 12288) &&
              (rows <= 128 || (rows > 384 && rows <= 512))) ||
             mmq_get_shmem<Type>(128, get_mmq_y_host(cc), cc) > prop.sharedMemPerBlockOptin)
        multiply<Type, 64>(packed, quantized, accumulator, columns, cc, rows);
    else multiply<Type, 128>(packed, quantized, accumulator, columns, cc, rows);
}
} // namespace

bool ggufMmq(const Tensor &input, const Tensor &packed, Tensor &output, uint32_t kind) {
    // Qwen's fused Q4_K gate/up benefits from MMQ through 1024 tokens on
    // SM120, including ordinary 512-square editing. Larger canvases still
    // favor bounded decoding + cuBLAS. Retain H3's cutoff for other shapes.
    const bool isWideGate = kind == 12 && input.cols == 4096 && output.cols == 24576 && input.rows <= 1024;
    if (input.rows < 8 || (input.rows > 512 && !isWideGate) || input.cols % MMQ_ITER_K != 0 ||
        (kind != 2 && kind != 8 && (kind < 10 || kind > 14))) return false;
    int device = 0;
    check(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, device));
    if (prop.major < 8) return false;
    if (input.rows > 512 && prop.major != 12) return false;
    if (input.rows > 512) {
        if (output.isFloat32()) wideGate<float>(input, packed, output, prop.major*100+prop.minor*10);
        else wideGate<B>(input, packed, output, prop.major*100+prop.minor*10);
        return true;
    }
    const auto type = static_cast<ggml_type>(kind);
    const size_t rowBytes = size_t(input.cols/128)*sizeof(block_q8_1_mmq);
    const size_t guard = size_t(get_mmq_x_max_host(prop.major*100+prop.minor*10))*sizeof(block_q8_1_mmq);
    // Bound both Q8 and FP32 staging to 64 MiB, independent of canvas size.
    constexpr size_t limit = size_t(64) << 20;
    size_t maxRows = std::min({size_t(65535), (limit-guard)/rowBytes,
                               limit/(size_t(output.cols)*sizeof(float))});
    if (maxRows >= 128) maxRows = maxRows/128*128;
    if (!maxRows) return false;
    const int chunk = int(std::min(size_t(input.rows), maxRows));
    Tensor quantized(1, int((size_t(chunk)*rowBytes+guard)/2));
    Tensor accumulator(chunk, output.cols, DType::FP32);
    for (int offset = 0; offset < input.rows; offset += chunk) {
        const int rows = std::min(chunk, input.rows-offset);
        const B *source = reinterpret_cast<const B *>(input.data)+size_t(offset)*input.cols;
#define MMQ_CASE(T) case T: run<T>(source, packed, quantized, accumulator, input.cols, prop, rows); break
        switch (type) {
            MMQ_CASE(GGML_TYPE_Q4_0);
            MMQ_CASE(GGML_TYPE_Q8_0);
            MMQ_CASE(GGML_TYPE_Q2_K);
            MMQ_CASE(GGML_TYPE_Q3_K);
            MMQ_CASE(GGML_TYPE_Q4_K);
            MMQ_CASE(GGML_TYPE_Q5_K);
            MMQ_CASE(GGML_TYPE_Q6_K);
            default: throw std::logic_error("unsupported image MMQ type");
        }
#undef MMQ_CASE
        const size_t count = size_t(rows)*output.cols;
        if(output.isFloat32())
            check(cudaMemcpyAsync(reinterpret_cast<float *>(output.data)+size_t(offset)*output.cols,
                                  accumulator.data,count*sizeof(float),cudaMemcpyDeviceToDevice,cudaStreamPerThread));
        else {
            convert<<<int(std::min(size_t(4096), (count+255)/256)), 256, 0, cudaStreamPerThread>>>(
                reinterpret_cast<const float *>(accumulator.data),
                reinterpret_cast<B *>(output.data)+size_t(offset)*output.cols, count);
            check(cudaGetLastError());
        }
    }
    return true;
}
} // namespace fastllm::minimax_h3_image::detail
