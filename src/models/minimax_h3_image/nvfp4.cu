#include "quantized_backends.h"
#include "quantized_checkpoint.h"
#include "quantized_kernels.h"
#if defined(FASTLLM_IMAGE_NVFP4_SM120)
#include "json11.hpp"
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <map>
#include <stdexcept>

namespace fastllm::minimax_h3_image {
using minimax_h3::DType;
using minimax_h3::Tensor;
using B = __nv_bfloat16;
static void require(bool v, const std::string &s) {
    if (!v)
        throw std::runtime_error(s);
}
static void ck(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
__device__ size_t scaleIndex(int row, int group, int k) {
    return (size_t(row / 128) * (k / 64) + group / 4) * 512 + (row % 32) * 16 + (row % 128 / 32) * 4 +
           group % 4;
}
__global__ void maximum(const B *x, float *maxval, size_t size) {
    float a = 0;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size;
         i += size_t(gridDim.x) * blockDim.x)
        a = fmaxf(a, fabsf(float(x[i])));
    for (int d = 16; d; d /= 2)
        a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, d));
    __shared__ float s[8];
    if (threadIdx.x % 32 == 0)
        s[threadIdx.x / 32] = a;
    __syncthreads();
    if (threadIdx.x < 32) {
        a = threadIdx.x < 8 ? s[threadIdx.x] : 0;
        for (int d = 16; d; d /= 2)
            a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, d));
        if (!threadIdx.x)
            atomicMax(reinterpret_cast<int *>(maxval), __float_as_int(a));
    }
}
__global__ void quant(const B *x, uint8_t *q, uint8_t *scales, float *global, float wg, int m, int padded,
                      int k) {
    float gs = fmaxf(global[0] / 2688.f, 1e-12f);
    if (blockIdx.x == 0 && threadIdx.x == 0)
        global[1] = gs * wg;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size_t(padded) * (k / 16);
         i += size_t(gridDim.x) * blockDim.x) {
        int row = i / (k / 16), group = i % (k / 16);
        B v[16];
        float a = 0;
        if (row < m) {
            *reinterpret_cast<uint4 *>(v) = *reinterpret_cast<const uint4 *>(x + i * 16);
            *reinterpret_cast<uint4 *>(v + 8) = *reinterpret_cast<const uint4 *>(x + i * 16 + 8);
        } else {
            *reinterpret_cast<uint4 *>(v) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4 *>(v + 8) = make_uint4(0, 0, 0, 0);
        }
        for (int j = 0; j < 16; ++j)
            a = fmaxf(a, fabsf(float(v[j])));
        __nv_fp8_e4m3 sf;
        sf.__x = __nv_cvt_float_to_fp8(a / (6.f * gs), __NV_SATFINITE, __NV_E4M3);
        scales[scaleIndex(row, group, k)] = sf.__x;
        float inv = float(sf) > 0 ? 1.f / (gs * float(sf)) : 0.f;
        uint64_t data = 0;
        for (int j = 0; j < 8; ++j) {
            auto pair = __nv_cvt_float2_to_fp4x2(
                make_float2(float(v[2 * j]) * inv, float(v[2 * j + 1]) * inv), __NV_E2M1, cudaRoundNearest);
            data |= uint64_t(pair) << (j * 8);
        }
        *reinterpret_cast<uint64_t *>(q + i * 8) = data;
    }
}
__global__ void swapNibbles(uint8_t *p, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x)
        p[i] = (p[i] << 4) | (p[i] >> 4);
}
class Nvfp4Weights final : public QuantizedWeights {
  public:
    Nvfp4Weights(const std::string &path, const ProjectionOptions &);
    ~Nvfp4Weights() override;
    void validateModel() const override;
    Tensor linear(const Tensor &, const std::string &, Tensor *) override;
    void erasePrefix(const std::string &) override;
    void clear() override;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
struct Nvfp4Weights::Impl : detail::QuantizedCheckpoint {
    ProjectionOptions options;
    struct Weight {
        Tensor q, s;
        int n = 0, k = 0;
        float global = 0;
    };
    json11::Json metadata;
    std::map<std::string, Weight> weights;
    Impl(const std::string &path, const ProjectionOptions &o) : QuantizedCheckpoint(path), options(o) {
        std::string error;
        metadata =
            json11::Json::parse(header["__metadata__"]["_quantization_metadata"].string_value(), error);
        require(error.empty() && metadata["layers"].is_object(), "invalid NVFP4 metadata");
    }
    ~Impl() { cudaStreamSynchronize(cudaStreamPerThread); }
    void validate(const std::string &name, int n, int k) const {
        const auto d = metadata["layers"][name];
        require(d["format"].string_value() == "nvfp4" && d["group_size"].number_value() == 16 &&
                    ((options.comfyVideo && d["orig_shape"].is_null()) ||
                     (d["orig_shape"].array_items().size() == 2 && d["orig_shape"][0].number_value() == n &&
                      d["orig_shape"][1].number_value() == k)) && n > 0 && n % 128 == 0 && k > 0 && k % 64 == 0,
                "invalid NVFP4 layer metadata: " + name);
        auto q = desc(name + ".weight", "U8", size_t(n) * k / 2);
        auto scale = desc(name + ".weight_scale", "F8_E4M3", size_t(n) * k / 16);
        require(q["shape"].array_items().size() == 2 && q["shape"][0].number_value() == n &&
                    q["shape"][1].number_value() == k / 2 && scale["shape"].array_items().size() == 2 &&
                    scale["shape"][0].number_value() == n && scale["shape"][1].number_value() == k / 16,
                "invalid NVFP4 packed shape: " + name);
        desc(name + ".weight_scale_2", "F32", 4);
    }
    Weight &get(const std::string &name) {
        auto it = weights.find(name);
        if (it != weights.end())
            return it->second;
        Weight w;
        auto shape = metadata["layers"][name]["orig_shape"];
        w.n = shape[0].int_value();
        w.k = shape[1].int_value();
        if(options.comfyVideo && shape.is_null()) {
            const auto packed = descriptor(name+".weight","U8")["shape"];
            require(packed.array_items().size()==2 && packed[1].number_value()<=INT32_MAX/2,
                    "invalid NVFP4 logical shape");
            w.n=packed[0].int_value(); w.k=packed[1].int_value()*2;
        }
        validate(name, w.n, w.k);
        detail::ProjectionSlice slice(name,w.n,w.k,options);
        require(slice.rows%128==0 && slice.cols%64==0 && slice.firstColumn()%64==0,
                "NVFP4 TP partition must align to scale tiles");
        const auto matrix=desc(name+".weight","U8",size_t(w.n)*w.k/2);
        const auto scales=desc(name+".weight_scale","F8_E4M3",size_t(w.n)*w.k/16);
        w.n=slice.rows; w.k=slice.cols;
        w.q = Tensor(w.n, w.k / 4);
        w.s = Tensor(w.n, w.k / 32);
        uploadSlice(matrix,w.q.data,slice,slice.fullCols/2,slice.cols/2,slice.firstColumn()/2);
        // Comfy NVFP4 scales already use 128x4-group CUTLASS tiles. Slice
        // whole tiles and retain the intra-tile permutation, not logical rows.
        uploadSlice(scales,w.s.data,slice,size_t(slice.fullCols/64)*512,size_t(slice.cols/64)*512,
                    size_t(slice.firstColumn()/64)*512,128,true);
        auto d = desc(name + ".weight_scale_2", "F32", 4);
        read(d, &w.global, 4);
        require(std::isfinite(w.global) && w.global > 0, "NVFP4 global scale");
        // Comfy files pack the even column in the high nibble; CUDA/CUTLASS use the
        // low nibble.
        swapNibbles<<<4096, 256, 0, cudaStreamPerThread>>>((uint8_t *)w.q.data, w.q.bytes());
        ck(cudaGetLastError());
        return weights.emplace(name, std::move(w)).first->second;
    }
    Tensor run(const Tensor &x, const std::string &name, Tensor *dst) {
        auto &w = get(name);
        require(x.dtype == DType::BF16 && x.cols == w.k, "NVFP4 input shape");
        int padded = (x.rows + 127) / 128 * 128;
        Tensor q(padded, w.k / 4), s(padded, w.k / 32), global(1, 2, true),
            out = dst ? *dst : Tensor(x.rows, w.n);
        require(out.rows == x.rows && out.cols == w.n &&
                (out.dtype == DType::BF16 || out.dtype == DType::FP32), "NVFP4 output shape");
        ck(cudaMemsetAsync(global.data, 0, 8, cudaStreamPerThread));
        maximum<<<256, 256, 0, cudaStreamPerThread>>>((B *)x.data, (float *)global.data, x.size());
        if(options.activationMaximum) {
            Tensor amax=global; amax.cols=1;
            options.activationMaximum(amax);
        }
        quant<<<4096, 256, 0, cudaStreamPerThread>>>((B *)x.data, (uint8_t *)q.data, (uint8_t *)s.data,
                                                     (float *)global.data, w.global, x.rows, padded, w.k);
        ck(cudaGetLastError());
        detail::nvfp4Gemm(out.data, q.data, w.q.data, s.data, w.s.data, (float *)global.data + 1, x.rows, w.n,
                          w.k, cudaStreamPerThread, out.isFloat32());
        return out;
    }
};
Nvfp4Weights::Nvfp4Weights(const std::string &path, const ProjectionOptions &options)
    : QuantizedWeights(WeightFormat::NVFP4), impl_(std::make_unique<Impl>(path,options)) {}
Nvfp4Weights::~Nvfp4Weights() = default;
void detail::validateNvfp4Device(int device) {
    cudaDeviceProp p{};
    ck(cudaGetDeviceProperties(&p, device));
    require(p.major == 12 && p.minor == 0, "image NVFP4 requires SM120");
}
void Nvfp4Weights::validateModel() const {
    for (int l = 0; l < 50; ++l) {
        auto prefix = "blocks." + std::to_string(l);
        impl_->validate(prefix + ".attn.qkv_proj", 21504, 5376);
        impl_->validate(prefix + ".attn.out_proj", 5376, 7168);
        impl_->validate(prefix + ".mlp.fc1", 28672, 5376);
        impl_->validate(prefix + ".mlp.fc2", 5376, 14336);
    }
}

Tensor Nvfp4Weights::linear(const Tensor &x, const std::string &n, Tensor *out) {
    return impl_->run(x, n, out);
}
void Nvfp4Weights::erasePrefix(const std::string &p) {
    for (auto i = impl_->weights.begin(); i != impl_->weights.end();)
        if (i->first.compare(0, p.size(), p) == 0)
            i = impl_->weights.erase(i);
        else
            ++i;
}
void Nvfp4Weights::clear() { impl_->weights.clear(); }
std::unique_ptr<QuantizedWeights> detail::loadNvfp4(const std::string &path, const ProjectionOptions &options) {
    return std::make_unique<Nvfp4Weights>(path, options);
}
} // namespace fastllm::minimax_h3_image
#else
#include <stdexcept>
namespace fastllm::minimax_h3_image::detail {
void validateNvfp4Device(int) {
    throw std::runtime_error("image NVFP4 requires CUDA 12.8+ and the SM120 CUTLASS backend");
}
std::unique_ptr<QuantizedWeights> loadNvfp4(const std::string &, const ProjectionOptions &) {
    validateNvfp4Device(0);
    return {};
}
} // namespace fastllm::minimax_h3_image::detail
#endif
