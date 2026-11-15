#include "gguf_checkpoint.h"
#include "quantized_backends.h"
#include "quantized_partition.h"
#include <algorithm>
#include <climits>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <map>
#include <stdexcept>

namespace fastllm::minimax_h3_image::detail {
namespace {
using minimax_h3::Tensor;
using minimax_h3::DType;
using B = __nv_bfloat16;
void check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
void check(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("GGUF cuBLAS status " + std::to_string(int(status)));
}
void require(bool ok, const char *message) {
    if (!ok)
        throw std::invalid_argument(message);
}
__device__ float halfAt(const uint8_t *p) { return __half2float(*reinterpret_cast<const __half *>(p)); }
// GGML block layouts (Q*_K uses 256 values). Decode to BF16 without another
// lossy weight quantization. CUDA and CPU oracle are tested independently.
template <int Type> __device__ float value(const uint8_t *p, int i) {
    if constexpr (Type == 0)
        return reinterpret_cast<const float *>(p)[i];
    if constexpr (Type == 1)
        return halfAt(p + i * 2);
    if constexpr (Type == 30)
        return float(reinterpret_cast<const B *>(p)[i]);
    if constexpr (Type == 2) {
        const int q = (p[2 + i % 16] >> (4 * (i / 16))) & 15;
        return halfAt(p) * (q - 8);
    }
    if constexpr (Type == 8)
        return halfAt(p) * reinterpret_cast<const int8_t *>(p + 2)[i];
    if constexpr (Type == 10) {
        const int sc = p[i / 16];
        const int q = (p[16 + (i / 128) * 32 + i % 32] >> (2 * ((i % 128) / 32))) & 3;
        return (halfAt(p + 80) * (sc & 15)) * q - halfAt(p + 82) * (sc >> 4);
    }
    if constexpr (Type == 11) {
        const int group = i / 16;
        const int sc = ((p[96 + group % 8] >> (4 * (group / 8))) & 15) |
                       (((p[104 + group % 4] >> (2 * (group / 4))) & 3) << 4);
        const int low = (p[32 + (i / 128) * 32 + i % 32] >> (2 * ((i % 128) / 32))) & 3;
        const int q = low - ((p[i % 32] & (1 << (i / 32))) ? 0 : 4);
        return (halfAt(p + 108) * (sc - 32)) * q;
    }
    if constexpr (Type == 12 || Type == 13) {
        const int g = i / 32;
        const uint8_t *s = p + 4;
        const int sc = g < 4 ? s[g] & 63 : (s[g + 4] & 15) | ((s[g - 4] >> 6) << 4);
        const int mn = g < 4 ? s[g + 4] & 63 : (s[g + 4] >> 4) | ((s[g] >> 6) << 4);
        const uint8_t *q = p + (Type == 12 ? 16 : 48);
        int v = (q[(i / 64) * 32 + i % 32] >> (4 * ((i / 32) % 2))) & 15;
        if constexpr (Type == 13)
            v += ((p[16 + i % 32] >> g) & 1) * 16;
        return (halfAt(p) * sc) * v - halfAt(p + 2) * mn;
    }
    if constexpr (Type == 14) {
        const int sub = i % 128, part = sub / 32, lane = i % 32;
        const int low = (p[(i / 128) * 64 + lane + (part % 2) * 32] >> (4 * (part / 2))) & 15;
        const int high = (p[128 + (i / 128) * 32 + lane] >> (2 * part)) & 3;
        return (halfAt(p + 208) * reinterpret_cast<const int8_t *>(p + 192)[i / 16]) *
               ((low | (high << 4)) - 32);
    }
    return 0;
}
template <int Type, int Values, int Bytes>
__global__ void decode(const uint8_t *packed, B *out, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(blockDim.x) * gridDim.x)
        out[i] = B(value<Type>(packed + (i / Values) * Bytes, i % Values));
}
void dequantize(const uint8_t *packed, Tensor &out, uint32_t type) {
    const int grid = int(std::min(size_t(4096), (out.size() + 255) / 256));
#define GGUF_DECODE(T, V, S)                                                                                 \
    case T:                                                                                                  \
        decode<T, V, S>                                                                                      \
            <<<grid, 256, 0, cudaStreamPerThread>>>(packed, reinterpret_cast<B *>(out.data), out.size());    \
        break
    switch (type) {
        GGUF_DECODE(0, 1, 4);
        GGUF_DECODE(1, 1, 2);
        GGUF_DECODE(30, 1, 2);
        GGUF_DECODE(2, 32, 18);
        GGUF_DECODE(8, 32, 34);
        GGUF_DECODE(10, 256, 84);
        GGUF_DECODE(11, 256, 110);
        GGUF_DECODE(12, 256, 144);
        GGUF_DECODE(13, 256, 176);
        GGUF_DECODE(14, 256, 210);
    default:
        throw std::invalid_argument("unsupported GGUF decode type");
    }
#undef GGUF_DECODE
    check(cudaGetLastError());
}
class GgufWeights final : public QuantizedWeights {
    struct Weight {
        Tensor packed;
        int rows, cols;
        uint32_t type;
        size_t rowBytes;
    };
    GgufCheckpoint checkpoint_;
    ProjectionOptions options_;
    std::map<std::string, Weight> weights_;
    cublasHandle_t handle_ = nullptr;
    Tensor workspace_;
    int device_ = -1, major_ = 0;
    bool prefill_ = false;
    void bindDevice() {
        int device = 0;
        check(cudaGetDevice(&device));
        require(device_ < 0 || device == device_, "GGUF backend used on a different device");
        if (device_ < 0) {
            validateGgufDevice(device);
            cudaDeviceProp prop{};
            check(cudaGetDeviceProperties(&prop, device));
            device_ = device;
            major_ = prop.major;
        }
    }
    Weight &get(const std::string &name) {
        auto found = weights_.find(name);
        if (found != weights_.end())
            return found->second;
        const auto &d = checkpoint_.tensor(name + ".weight");
        require(d.dimensions.size() == 2, "GGUF projection rank");
        Weight w;
        w.cols = int(d.dimensions[0]);
        w.rows = int(d.dimensions[1]);
        w.type = d.type;
        const ProjectionSlice slice(name,w.rows,w.cols,options_);
        const auto block=ggufBlock(w.type);
        require(slice.cols%block.values==0 && slice.firstColumn()%block.values==0,
                "GGUF TP partition must align to quantization blocks; use TP2/TP4 or SP");
        const size_t sourceStride=size_t(d.bytes/w.rows);
        w.rows=slice.rows; w.cols=slice.cols;
        w.rowBytes = size_t(w.cols/block.values)*block.bytes;
        require(w.rowBytes % 2 == 0 && w.rowBytes / 2 <= size_t(INT_MAX), "GGUF packed row size");
        w.packed = Tensor(w.rows, int(w.rowBytes / 2));
        if(options_.size>1) {
            slice.transfer([&](size_t offset,void *host,size_t bytes) {checkpoint_.read(d,offset,host,bytes);},
                [&](size_t offset,const void *host,size_t bytes) {
                    check(cudaMemcpyAsync(reinterpret_cast<char *>(w.packed.data)+offset,host,bytes,
                                          cudaMemcpyHostToDevice,cudaStreamPerThread));
                    check(cudaStreamSynchronize(cudaStreamPerThread));
                },sourceStride,w.rowBytes,size_t(slice.firstColumn()/block.values)*block.bytes);
        } else {
        std::vector<char> staging(std::min(d.bytes, uint64_t(8) << 20));
        for (uint64_t offset = 0; offset < d.bytes;) {
            const size_t n = std::min(uint64_t(staging.size()), d.bytes - offset);
            checkpoint_.read(d, offset, staging.data(), n);
            check(cudaMemcpyAsync(reinterpret_cast<char *>(w.packed.data) + offset, staging.data(), n,
                                  cudaMemcpyHostToDevice, cudaStreamPerThread));
            check(cudaStreamSynchronize(cudaStreamPerThread));
            offset += n;
        }
        }
        return weights_.emplace(name, std::move(w)).first->second;
    }

    void decodedLinear(const Tensor &x, const Weight &w, Tensor &out, int firstColumn) {
        if (!handle_) {
            check(cublasCreate(&handle_));
            check(cublasSetStream(handle_, cudaStreamPerThread));
            check(cublasSetMathMode(handle_, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
        }
        if (!workspace_.data) {
            workspace_ = Tensor(1, (4 << 20) / 2);
            check(cublasSetWorkspace(handle_, workspace_.data, workspace_.bytes()));
        }
        // Keep the decode buffer bounded, and avoid tiny GEMMs for Qwen's
        // middle-sized batches. H3 retains its existing 2048-row schedule.
        int chunk = 2048;
        if (major_ == 12 && !prefill_ && x.rows > 512 && x.rows <= 1024 &&
            w.rows == 4096 && w.cols == 4096)
            chunk = 4096;
        for (int row = 0; row < w.rows; row += chunk) {
            const int n = std::min(chunk, w.rows - row);
            Tensor decoded(n, w.cols);
            dequantize(reinterpret_cast<const uint8_t *>(w.packed.data) + size_t(row) * w.rowBytes, decoded,
                       w.type);
            float one = 1, zero = 0;
            check(cublasGemmEx(handle_, CUBLAS_OP_T, CUBLAS_OP_N, n, x.rows, x.cols, &one, decoded.data,
                               CUDA_R_16BF, w.cols, x.data, CUDA_R_16BF, x.cols, &zero,
                               reinterpret_cast<char *>(out.data)+size_t(firstColumn+row)*(out.isFloat32()?4:2),
                               out.isFloat32()?CUDA_R_32F:CUDA_R_16BF, out.cols,
                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    }

  public:
    GgufWeights(const std::string &path, const ProjectionOptions &options)
        : QuantizedWeights(WeightFormat::GGUF), checkpoint_(path), options_(options) {}
    ~GgufWeights() override {
        if (device_ >= 0)
            cudaSetDevice(device_);
        if (handle_) {
            cudaStreamSynchronize(cudaStreamPerThread);
            cublasDestroy(handle_);
        }
    }
    void validateModel() const override {
        if(options_.comfyVideo) checkpoint_.validateVideoModel();
        else checkpoint_.validateModel();
    }
    void setPrefill(bool prefill) override { prefill_ = prefill; }
    Tensor linear(const Tensor &x, const std::string &name, Tensor *destination) override {
        bindDevice();
        auto &w = get(name);
        require(x.data && x.rows > 0 && x.dtype == DType::BF16 && x.cols == w.cols, "GGUF input shape/dtype");
        Tensor out = destination ? *destination : Tensor(x.rows, w.rows);
        require(out.data && out.rows == x.rows && out.cols == w.rows &&
                    (out.dtype == DType::BF16 || out.dtype == DType::FP32) &&
                    out.storage != x.storage && out.storage != w.packed.storage,
                "GGUF output shape/alias");
#ifdef FASTLLM_IMAGE_GGUF_MMQ
        if ((!prefill_ || x.rows <= 512) && ggufMmq(x, w.packed, out, w.type))
            return out;
#endif
        decodedLinear(x, w, out, 0);
        return out;
    }
    Tensor linearPacked(const Tensor &x, const std::vector<std::string> &names, Tensor *destination) override {
        require(!names.empty() && options_.size == 1, "invalid direct GGUF projection list");
        if (names.size() == 1)
            return linear(x, names.front(), destination);
        bindDevice();
        require(x.data && x.rows > 0 && x.dtype == DType::BF16, "GGUF packed input shape/dtype");
        int columns = 0;
        for (const auto &name : names) {
            const auto &w = get(name);
            require(w.cols == x.cols && w.rows <= INT_MAX - columns, "inconsistent GGUF packed projection");
            columns += w.rows;
        }
        Tensor out = destination ? *destination : Tensor(x.rows, columns);
        require(out.data && out.rows == x.rows && out.cols == columns &&
                    (out.dtype == DType::BF16 || out.dtype == DType::FP32) && out.storage != x.storage,
                "GGUF packed destination shape/alias");
        const size_t element = out.isFloat32() ? 4 : 2;
        int offset = 0;
        // Mixed-bit checkpoints may use different GGML types for Q/K/V.
        // Keep each original block format and concatenate only the outputs.
        for (const auto &name : names) {
            const auto &w = get(name);
            require(out.storage != w.packed.storage, "GGUF packed output/weight alias");
            // Q/K/V with more than 512 tokens use decoded GEMM. Write to
            // their final columns directly instead of staging and copying.
            if (!prefill_ && x.rows > 512 && w.rows == 4096 && w.cols == 4096) {
                decodedLinear(x, w, out, offset);
            } else {
                Tensor part(x.rows, w.rows, out.dtype);
                linear(x, name, &part);
                check(cudaMemcpy2DAsync(reinterpret_cast<char *>(out.data) + size_t(offset)*element,
                    size_t(columns)*element, part.data, size_t(part.cols)*element,
                    size_t(part.cols)*element, x.rows, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
            }
            offset += w.rows;
        }
        return out;
    }
    size_t loadedBytes() const override {
        size_t bytes = workspace_.bytes();
        for (const auto &entry : weights_)
            bytes += entry.second.packed.bytes();
        return bytes;
    }
    void erasePrefix(const std::string &prefix) override {
        for (auto it = weights_.begin(); it != weights_.end();)
            if (it->first.compare(0, prefix.size(), prefix) == 0)
                it = weights_.erase(it);
            else
                ++it;
    }
    void clear() override {
        weights_.clear();
        workspace_ = {};
    }
};
} // namespace
void validateGgufDevice(int device) {
    cudaDeviceProp p{};
    check(cudaGetDeviceProperties(&p, device));
    require(p.major >= 8, "GGUF image inference requires SM80+ for BF16 stages");
}
std::unique_ptr<QuantizedWeights> loadGguf(const std::string &path, const ProjectionOptions &options) {
    return std::make_unique<GgufWeights>(path, options);
}
} // namespace fastllm::minimax_h3_image::detail
