#include "devices/cuda/glm5-next-cuda.cuh"

#ifdef FASTLLM_ENABLE_GLM53_FLASHINFER_SM120
#include "devices/cuda/fastllm-cuda.cuh"
#include "glm5-next-dsa.cuh"
#include <climits>
#include <cmath>
#include <stdexcept>

namespace {
using namespace fastllm;

bool ContiguousCuda(const Data &x) {
    if (x.dataDevice != DataDevice::CUDA || !x.cudaData ||
        x.dims.empty() || x.dims.size() != x.strides.size()) return false;
    uint64_t stride = 1;
    for (int i = int(x.dims.size()) - 1; i >= 0; --i) {
        if (x.dims[i] <= 0 || (x.dims[i] > 1 && x.strides[i] != stride)) return false;
        stride *= x.dims[i];
    }
    return true;
}

void Check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("GLM DSA FlashInfer: ") +
                                 cudaGetErrorString(status));
    }
}

void Prepare(Data &x, DataType type, const std::vector<int> &dims, int device) {
    x.dataType = type;
    x.UpdateUnitSize();
    x.Resize(dims);
    x.ToDevice(DataDevice::CUDA, {device}, false);
    x.Allocate(false);
    if (!x.cudaData) throw std::runtime_error("GLM DSA FlashInfer allocation failed.");
}
} // namespace
#endif

bool FastllmCudaGlm5NextDsaPrefill(const fastllm::Data &query,
        const fastllm::Data &latent, const fastllm::Data &indices,
        float scale, fastllm::Data &output) {
#ifdef FASTLLM_ENABLE_GLM53_FLASHINFER_SM120
    using namespace fastllm;
    if (query.dataType != BFLOAT16 || query.dims.size() != 4 ||
        query.dims[0] != 1 || query.dims[1] < 64 || query.dims[2] != 64 ||
        query.dims[3] != 512 || latent.dataType != BFLOAT16 ||
        latent.dims.size() != 3 || latent.dims[0] != 1 || latent.dims[2] != 512 ||
        latent.dims[1] <= 0 || latent.dims[1] > INT_MAX / 4 ||
        indices.dataType != INT32 || indices.dims.size() != 3 ||
        indices.dims[0] != 1 || indices.dims[1] != query.dims[1] ||
        indices.dims[2] <= 0 || indices.dims[2] > INT_MAX - 63 ||
        !std::isfinite(scale) || scale <= 0 || &output == &query ||
        &output == &latent || &output == &indices ||
        !ContiguousCuda(query) || !ContiguousCuda(latent) || !ContiguousCuda(indices)) {
        return false;
    }
    const int device = GetPointerDeviceId(query.cudaData);
    if (device < 0 || GetPointerDeviceId(latent.cudaData) != device ||
        GetPointerDeviceId(indices.cudaData) != device) return false;
    int major = 0, minor = 0;
    Check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    Check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
    // The cubin uses architecture-specific instructions; major == 12 alone
    // would incorrectly admit SM121 devices to an sm_120a binary.
    if (major != 12 || minor != 0) return false;
    FastllmCudaSetDevice(device);
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    Check(cudaStreamIsCapturing(cudaStreamPerThread, &capture));
    if (capture != cudaStreamCaptureStatusNone) return false;

    const int rows = query.dims[1], tokens = latent.dims[1];
    const int width = indices.dims[2], paddedWidth = (width + 63) / 64 * 64;
    Data bytes, scales, packed, padded, lse;
    Prepare(bytes, INT8, {tokens, 512}, device);
    Prepare(scales, FLOAT32, {tokens, 4}, device);
    Prepare(packed, INT8, {tokens, 528}, device);
    Prepare(padded, INT32, {rows, paddedWidth}, device);
    Prepare(lse, FLOAT32, {rows, 64}, device);
    Prepare(output, BFLOAT16, query.dims, device);

    // Keep BF16 paged state authoritative, including restored history. No
    // persistent FP8 cache is added: quantize the gathered latent per chunk.
    Check(FastllmCudaGlm5NextQuantizeLatentRaw(latent.cudaData,
        bytes.cudaData, static_cast<float *>(scales.cudaData), tokens * 4,
        cudaStreamPerThread));
    Check(cudaMemcpy2DAsync(packed.cudaData, 528, bytes.cudaData, 512,
        512, tokens, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    Check(cudaMemcpy2DAsync(static_cast<uint8_t *>(packed.cudaData) + 512, 528,
        scales.cudaData, 16, 16, tokens, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    // Preserve all 2048 selected tokens and the causal tail (up to 3 tokens).
    Check(cudaMemsetAsync(padded.cudaData, 0xff,
        size_t(rows) * paddedWidth * sizeof(int32_t), cudaStreamPerThread));
    Check(cudaMemcpy2DAsync(padded.cudaData, size_t(paddedWidth) * sizeof(int32_t),
        indices.cudaData, size_t(width) * sizeof(int32_t),
        size_t(width) * sizeof(int32_t), rows, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    Check(FastllmCudaGlm5NextDsaPrefillSm120Raw(query.cudaData, packed.cudaData,
        static_cast<const int32_t *>(padded.cudaData), output.cudaData,
        static_cast<float *>(lse.cudaData), rows, paddedWidth, scale, cudaStreamPerThread));
    return true;
#else
    return false;
#endif
}
