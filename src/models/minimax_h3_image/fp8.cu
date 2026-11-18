#include "quantized_checkpoint.h"
#include "models/minimax_h3_image/fp8.h"
#include <cuda_runtime.h>
#if CUDART_VERSION >= 11080
#include <algorithm>
#include <cmath>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>
#ifdef FASTLLM_IMAGE_FP8_SM120
#include "fp8_scaled.h"
#endif
namespace fastllm::minimax_h3_image {
using minimax_h3::DType;
using minimax_h3::Tensor;
namespace {
using B = __nv_bfloat16;
void checked(cudaError_t x, int line) {
    if (x != cudaSuccess)
        throw std::runtime_error("CUDA line " + std::to_string(line) + ": " + cudaGetErrorString(x));
}
void checked(cublasStatus_t x, int line) {
    if (x != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("FP8 cuBLAS line " + std::to_string(line) + " status " +
                                 std::to_string(int(x)));
}
#define ck(call) checked((call), __LINE__)
bool usesScaledSm120(const cudaDeviceProp &properties) {
#ifdef FASTLLM_IMAGE_FP8_SM120
    return properties.major == 12 && properties.minor == 0;
#else
    (void)properties;
    return false;
#endif
}
void require(bool x, const std::string &message) {
    if (!x)
        throw std::runtime_error(message);
}
__device__ __forceinline__ float activationScale(float largest, bool bf16Scale) {
    // TorchAO rounds amax / 448 in BF16; H3 retains its FP32-scale policy.
    return bf16Scale ? float(B(fmaxf(largest, float(B(1e-12f))) / 448.f))
                     : fmaxf(largest / 448.f, 1e-12f);
}
__global__ void quantize(const B *x, uint8_t *q, float *scales, int rows, int cols, bool bf16Scale, bool scaleOnly = false) {
    int row = blockIdx.x;
    float largest = 0;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        largest = fmaxf(largest, row < rows ? fabsf(float(x[size_t(row) * cols + c])) : 0.f);
    for (int mask = 16; mask; mask /= 2)
        largest = fmaxf(largest, __shfl_xor_sync(0xffffffff, largest, mask));
    __shared__ float maxima[8];
    if (threadIdx.x % 32 == 0)
        maxima[threadIdx.x / 32] = largest;
    __syncthreads();
    largest = threadIdx.x < 8 ? maxima[threadIdx.x] : 0;
    for (int mask = 16; mask; mask /= 2)
        largest = fmaxf(largest, __shfl_xor_sync(0xffffffff, largest, mask));
    if (threadIdx.x == 0)
        maxima[0] = activationScale(largest, bf16Scale);
    __syncthreads();
    float scale = maxima[0];
    if (threadIdx.x == 0)
        scales[row] = scale;
    if (scaleOnly) return;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = row < rows ? float(x[size_t(row) * cols + c]) / scale : 0;
        q[size_t(row) * cols + c] = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    }
}
// Keep each row's input in registers across the reduction. Aligned vector
// loads and paired FP8 conversions avoid the scalar kernel's second read.
template <int Cols> __global__ void quantizeVector(const B *x, uint8_t *q, float *scales, int rows, bool bf16Scale) {
    constexpr int chunks = (Cols + 2047) / 2048;
    const int row = blockIdx.x;
    uint4 values[chunks];
    float largest = 0;
#pragma unroll
    for (int c = 0; c < chunks; ++c) {
        const int column = c * 2048 + threadIdx.x * 8;
        values[c] = row < rows && column < Cols
                        ? *reinterpret_cast<const uint4 *>(x + size_t(row) * Cols + column)
                        : make_uint4(0, 0, 0, 0);
        const auto *v = reinterpret_cast<const B *>(&values[c]);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            largest = fmaxf(largest, fabsf(float(v[j])));
    }
    for (int mask = 16; mask; mask /= 2)
        largest = fmaxf(largest, __shfl_xor_sync(0xffffffff, largest, mask));
    __shared__ float maxima[8];
    if (threadIdx.x % 32 == 0)
        maxima[threadIdx.x / 32] = largest;
    __syncthreads();
    largest = threadIdx.x < 8 ? maxima[threadIdx.x] : 0;
    for (int mask = 16; mask; mask /= 2)
        largest = fmaxf(largest, __shfl_xor_sync(0xffffffff, largest, mask));
    if (threadIdx.x == 0)
        maxima[0] = activationScale(largest, bf16Scale);
    __syncthreads();
    const float scale = maxima[0], inverse = 1.f / scale;
    if (threadIdx.x == 0)
        scales[row] = scale;
#pragma unroll
    for (int c = 0; c < chunks; ++c) {
        const int column = c * 2048 + threadIdx.x * 8;
        if (column < Cols) {
            const auto *v = reinterpret_cast<const B *>(&values[c]);
            uint32_t packed[2]{};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                auto pair = __nv_cvt_float2_to_fp8x2(
                    make_float2(float(v[j * 2]) * inverse, float(v[j * 2 + 1]) * inverse), __NV_SATFINITE,
                    __NV_E4M3);
                packed[j / 2] |= uint32_t(pair) << ((j % 2) * 16);
            }
            *reinterpret_cast<uint2 *>(q + size_t(row) * Cols + column) = make_uint2(packed[0], packed[1]);
        }
    }
}
// Row-parallel inputs share the full row's scale. Requantize the original
// BF16 values after the small collective, preserving vector-kernel arithmetic.
__global__ void quantizeShared(const B *x, uint8_t *q, const float *scales, int rows, int cols) {
    const int row = blockIdx.x;
    const float inverse = 1.f / scales[row];
    for (int c = threadIdx.x * 8; c < cols; c += blockDim.x * 8) {
        uint4 values = row < rows ? *reinterpret_cast<const uint4 *>(x + size_t(row)*cols+c)
                                 : make_uint4(0,0,0,0);
        const auto *v = reinterpret_cast<const B *>(&values);
        uint32_t packed[2]{};
#pragma unroll
        for (int j=0;j<4;++j) {
            auto pair=__nv_cvt_float2_to_fp8x2(make_float2(float(v[j*2])*inverse,float(v[j*2+1])*inverse),
                                               __NV_SATFINITE,__NV_E4M3);
            packed[j/2] |= uint32_t(pair)<<((j%2)*16);
        }
        *reinterpret_cast<uint2 *>(q+size_t(row)*cols+c)=make_uint2(packed[0],packed[1]);
    }
}
template<class Input, class Output>
__global__ void finish(const Input *raw, const float *xs, const float *ws, Output *out, size_t count, int cols) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(blockDim.x) * gridDim.x)
        out[i] = Output(float(raw[i]) * xs[i / cols] * ws[i % cols]);
}
__global__ void packQkv(const uint32_t *src, uint32_t *dst, int rows, int words) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size_t(rows) * words;
         i += size_t(blockDim.x) * gridDim.x) {
        int row = i / words, part = row / (56 * 128), within = row % (56 * 128),
            source = (within / 128) * 384 + part * 128 + within % 128;
        dst[i] = src[size_t(source) * words + i % words];
    }
}
struct Plan {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    ~Plan() {
        if (op)
            cublasLtMatmulDescDestroy(op);
        if (a)
            cublasLtMatrixLayoutDestroy(a);
        if (b)
            cublasLtMatrixLayoutDestroy(b);
        if (c)
            cublasLtMatrixLayoutDestroy(c);
        if (pref)
            cublasLtMatmulPreferenceDestroy(pref);
    }
};
struct Weight {
    Tensor bytes, scales;
    int rows = 0, cols = 0;
    bool rowParallel = false;
};
} // namespace
struct Fp8Weights::Impl : detail::QuantizedCheckpoint {
    ProjectionOptions options;
    bool unsloth = false, nativePackedQkv = false;
    json11::Json quantMetadata;
    cublasLtHandle_t handle = nullptr;
    Tensor workspace, one;
    int device = -1, multiprocessors = 0;
    bool scaledSm120 = false;
    std::map<std::string, Weight> weights;
    std::map<std::string, std::unique_ptr<Plan>> plans;
    Impl(const std::string &path, const ProjectionOptions &o) : QuantizedCheckpoint(path), options(o) {
        std::string error;
        unsloth = header["transformer_blocks.0.attn.to_q._weight_qdata"].is_object();
        const auto metadata = header["__metadata__"]["_quantization_metadata"].string_value();
        if (!unsloth && !metadata.empty()) {
            quantMetadata = json11::Json::parse(metadata, error);
            require(error.empty() && quantMetadata.is_object(), "invalid FP8 quantization metadata");
            // Comfy's exported H3 matrices already use [all Q, all K, all V].
            // Original H3 matrices interleave Q/K/V inside each head instead.
            nativePackedQkv =
                quantMetadata["layers"]["blocks.0.attn.qkv_proj"]["format"].string_value() == "float8_e4m3fn";
        }
        if(options.comfyVideo) {
            require(!unsloth, "video projections require the validated Comfy layout");
            nativePackedQkv = true;
        }
    }
    ~Impl() {
        cudaStreamSynchronize(cudaStreamPerThread);
        plans.clear();
        if (handle)
            cublasLtDestroy(handle);
    }
    Weight &get(const std::string &name, const std::vector<std::string> *direct = nullptr) {
        auto found = weights.find(name);
        if (found != weights.end())
            return found->second;
        std::vector<std::string> keys;
        bool qkv = !direct && name.find(".qkv_proj") != std::string::npos;
        if (direct)
            keys = *direct;
        else if (unsloth) {
            auto dot = name.find('.', 7);
            require(dot != std::string::npos, "invalid FP8 layer name");
            auto prefix = "transformer_blocks." + name.substr(7, dot - 7);
            if (qkv)
                keys = {prefix + ".attn.to_q", prefix + ".attn.to_k", prefix + ".attn.to_v"};
            else if (name.find(".out_proj") != std::string::npos)
                keys = {prefix + ".attn.to_out.0"};
            else if (name.find(".fc1") != std::string::npos)
                keys = {prefix + ".ff.net.0.proj"};
            else if (name.find(".fc2") != std::string::npos)
                keys = {prefix + ".ff.net.2"};
            else
                throw std::runtime_error("unsupported FP8 projection");
        } else
            keys = {name};
        Weight w;
        std::vector<json11::Json> matrices;
        std::vector<float> scales;
        for (auto &key : keys) {
            const bool torchao = !options.modelopt && (direct ? header[key + "._weight_qdata"].is_object() : unsloth);
            const bool rowScaled = torchao || options.modelopt;
            auto matrix = descriptor(key + (torchao ? "._weight_qdata" : ".weight"), "F8_E4M3");
            auto shape = matrix["shape"].array_items();
            require(shape.size() == 2 && shape[0].int_value() > 0 && shape[1].int_value() > 0,
                    "FP8 matrix shape");
            int rows = shape[0].int_value(), cols = shape[1].int_value();
            require(!w.cols || w.cols == cols, "inconsistent FP8 QKV");
            w.cols = cols;
            w.rows += rows;
            matrices.push_back(matrix);
            auto sd = descriptor(key + (torchao ? "._weight_scale" : ".weight_scale"), "F32");
            std::vector<float> part(rowScaled ? rows : 1);
            read(sd, part.data(), part.size() * 4);
            for (float v : part)
                require(std::isfinite(v) && v > 0, "invalid FP8 weight scale");
            if (!rowScaled)
                part.resize(rows, part[0]);
            scales.insert(scales.end(), part.begin(), part.end());
        }
        require(w.cols % 16 == 0 && w.rows % 16 == 0, "unaligned FP8 matrix");
        if (qkv)
            require(w.rows == 21504 && w.cols == 5376, "FP8 QKV layout");
        detail::ProjectionSlice slice(direct ? direct->front() : name, w.rows, w.cols, options,
                                      direct ? int(direct->size()) : 1);
        w.rows=slice.rows; w.cols=slice.cols; w.rowParallel=slice.rowParallel;
        if(options.size>1) scales=slice.scales(scales);
        w.bytes = Tensor(w.rows, w.cols / 2);
        const bool interleavedQkv = qkv && !unsloth && !nativePackedQkv;
        uploadByteMatrices(matrices, w.bytes.data, slice, interleavedQkv);
        // Diffusers stores [value, gate]; the original H3 uses [gate, value].
        // Apply the identical row permutation to both payload and per-row scales.
        if (!direct && unsloth && name.find(".fc1") != std::string::npos) {
            Tensor ordered(w.rows, w.cols / 2);
            size_t half = w.bytes.bytes() / 2;
            ck(cudaMemcpyAsync(ordered.data, reinterpret_cast<char *>(w.bytes.data) + half, half,
                               cudaMemcpyDeviceToDevice, cudaStreamPerThread));
            ck(cudaMemcpyAsync(reinterpret_cast<char *>(ordered.data) + half, w.bytes.data, half,
                               cudaMemcpyDeviceToDevice, cudaStreamPerThread));
            w.bytes = std::move(ordered);
            std::rotate(scales.begin(), scales.begin() + w.rows / 2, scales.end());
        }
        if (interleavedQkv && options.size == 1) {
            Tensor ordered(w.rows, w.cols / 2);
            packQkv<<<4096, 256, 0, cudaStreamPerThread>>>(reinterpret_cast<uint32_t *>(w.bytes.data),
                                                           reinterpret_cast<uint32_t *>(ordered.data), w.rows,
                                                           w.cols / 4);
            ck(cudaGetLastError());
            w.bytes = std::move(ordered);
        }
        w.scales = Tensor(1, w.rows, true);
        ck(cudaMemcpyAsync(w.scales.data, scales.data(), scales.size() * 4, cudaMemcpyHostToDevice,
                           cudaStreamPerThread));
        ck(cudaStreamSynchronize(cudaStreamPerThread));
        return weights.emplace(name, std::move(w)).first->second;
    }
    Tensor run(const Tensor &x, const std::string &name, Tensor *destination,
               const std::vector<std::string> *direct = nullptr) {
        if (device < 0) {
            ck(cudaGetDevice(&device));
            cudaDeviceProp properties{};
            ck(cudaGetDeviceProperties(&properties, device));
            multiprocessors = properties.multiProcessorCount;
            scaledSm120 = usesScaledSm120(properties);
        }
        if (!workspace.data)
            workspace = Tensor(1, (32 << 20) / 2);
        if (!scaledSm120 && !handle) {
            ck(cublasLtCreate(&handle));
            one = Tensor(1, 1, true);
            float value = 1;
            ck(cudaMemcpyAsync(one.data, &value, 4, cudaMemcpyHostToDevice, cudaStreamPerThread));
            ck(cudaStreamSynchronize(cudaStreamPerThread));
        }
        auto &w = get(name, direct);
        require(x.dtype == DType::BF16 && x.cols == w.cols, "FP8 input dtype/shape");
        int padded = (x.rows + 15) / 16 * 16;
        Tensor q(padded, x.cols / 2), scale(1, padded, true);
        Tensor out = destination ? *destination : Tensor(x.rows, w.rows);
        require(out.rows == x.rows && out.cols == w.rows && (out.dtype == DType::BF16 || out.dtype == DType::FP32), "FP8 output shape");
        auto *input = reinterpret_cast<const B *>(x.data);
        auto *quantized = reinterpret_cast<uint8_t *>(q.data);
        auto *scales = reinterpret_cast<float *>(scale.data);
        const bool bf16Scale = direct != nullptr && !options.modelopt;
        // H3's callback calibrates NVFP4; shared FP8 scales are Qwen-specific.
        if (w.rowParallel && options.qwenImage && options.activationMaximum) {
            quantize<<<padded,256,0,cudaStreamPerThread>>>(input,quantized,scales,x.rows,x.cols,
                                                         bf16Scale,true);
            ck(cudaGetLastError());
            options.activationMaximum(scale);
            quantizeShared<<<padded,256,0,cudaStreamPerThread>>>(input,quantized,scales,x.rows,x.cols);
        } else if (x.cols == 4096)
            quantizeVector<4096><<<padded, 256, 0, cudaStreamPerThread>>>(input, quantized, scales, x.rows, bf16Scale);
        else if (x.cols == 12288)
            quantizeVector<12288><<<padded, 256, 0, cudaStreamPerThread>>>(input, quantized, scales, x.rows, bf16Scale);
        else if (x.cols == 5376)
            quantizeVector<5376><<<padded, 256, 0, cudaStreamPerThread>>>(input, quantized, scales, x.rows, bf16Scale);
        else if (x.cols == 7168)
            quantizeVector<7168><<<padded, 256, 0, cudaStreamPerThread>>>(input, quantized, scales, x.rows, bf16Scale);
        else if (x.cols == 14336)
            quantizeVector<14336><<<padded, 256, 0, cudaStreamPerThread>>>(input, quantized, scales, x.rows, bf16Scale);
        else
            quantize<<<padded, 256, 0, cudaStreamPerThread>>>(input, quantized, scales, x.rows, x.cols, bf16Scale);
        ck(cudaGetLastError());
#ifdef FASTLLM_IMAGE_FP8_SM120
        if (scaledSm120) {
            detail::fp8ScaledSm120(q.data, w.bytes.data, scales,
                                   reinterpret_cast<const float *>(w.scales.data), out.data, x.rows, w.rows,
                                   x.cols, device, multiprocessors, workspace.data, workspace.bytes(),
                                   cudaStreamPerThread, out.isFloat32());
            return out;
        }
#endif
        Tensor raw(padded, w.rows, out.dtype);
        auto key = std::to_string(padded) + ":" + std::to_string(w.rows) + ":" + std::to_string(w.cols) + (out.isFloat32() ? ":fp32" : "");
        auto &plan = plans[key];
        if (!plan) {
            auto p = std::make_unique<Plan>();
            ck(cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
            cublasOperation_t transpose = CUBLAS_OP_T;
            ck(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &transpose,
                                              sizeof(transpose)));
            void *unit = one.data;
            ck(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &unit,
                                              sizeof(unit)));
            ck(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &unit,
                                              sizeof(unit)));
            ck(cublasLtMatrixLayoutCreate(&p->a, CUDA_R_8F_E4M3, w.cols, w.rows, w.cols));
            ck(cublasLtMatrixLayoutCreate(&p->b, CUDA_R_8F_E4M3, x.cols, padded, x.cols));
            ck(cublasLtMatrixLayoutCreate(&p->c, out.isFloat32() ? CUDA_R_32F : CUDA_R_16BF, w.rows, padded, w.rows));
            ck(cublasLtMatmulPreferenceCreate(&p->pref));
            size_t bytes = workspace.bytes();
            ck(cublasLtMatmulPreferenceSetAttribute(p->pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &bytes,
                                                    sizeof(bytes)));
            cublasLtMatmulHeuristicResult_t choice{};
            int count = 0;
            ck(cublasLtMatmulAlgoGetHeuristic(handle, p->op, p->a, p->b, p->c, p->c, p->pref, 1, &choice,
                                              &count));
            require(count == 1, "no native FP8 cuBLASLt algorithm: " + key);
            p->algorithm = choice.algo;
            plan = std::move(p);
        }
        float alpha = 1, beta = 0;
        ck(cublasLtMatmul(handle, plan->op, &alpha, w.bytes.data, plan->a, q.data, plan->b, &beta, raw.data,
                          plan->c, raw.data, plan->c, &plan->algorithm, workspace.data, workspace.bytes(),
                          cudaStreamPerThread));
        if(out.isFloat32())
            finish<<<4096,256,0,cudaStreamPerThread>>>((float *)raw.data,(float *)scale.data,
                (float *)w.scales.data,(float *)out.data,out.size(),w.rows);
        else
        finish<<<4096, 256, 0, cudaStreamPerThread>>>(
            reinterpret_cast<B *>(raw.data), reinterpret_cast<float *>(scale.data),
            reinterpret_cast<float *>(w.scales.data), reinterpret_cast<B *>(out.data), out.size(), w.rows);
        ck(cudaGetLastError());
        return out;
    }
};
Fp8Weights::Fp8Weights(const std::string &path) : Fp8Weights(path,{}) {}
Fp8Weights::Fp8Weights(const std::string &path, const ProjectionOptions &options)
    : QuantizedWeights(WeightFormat::FP8), impl_(std::make_unique<Impl>(path,options)) {}
Fp8Weights::~Fp8Weights() = default;
void Fp8Weights::validateDevice(int device, bool multipleDevices) {
    cudaDeviceProp properties{};
    ck(cudaGetDeviceProperties(&properties, device));
    require(properties.major * 10 + properties.minor >= 89,
            "image FP8 requires an SM89 or newer CUDA device");
    // Older cuBLAS can fail FP8 launches across Blackwell devices (CUB-9487).
    // The fused CUTLASS path does not issue cuBLAS FP8 calls.
    require(!multipleDevices || properties.major < 12 || usesScaledSm120(properties) ||
                cublasLtGetVersion() >= 130201,
            "multi-GPU Blackwell cuBLAS FP8 requires cuBLAS 13.2.1 or newer; update the "
            "cuBLAS runtime libraries");
}
void Fp8Weights::validateModel() const {
    for (int layer = 0; layer < 50; ++layer) {
        const std::string native = "blocks." + std::to_string(layer),
                          diffusers = "transformer_blocks." + std::to_string(layer);
        std::vector<std::tuple<std::string, int, int>> tensors;
        if (impl_->unsloth) {
            for (auto part : {"q", "k", "v"})
                tensors.emplace_back(diffusers + ".attn.to_" + part, 7168, 5376);
            tensors.emplace_back(diffusers + ".attn.to_out.0", 5376, 7168);
            tensors.emplace_back(diffusers + ".ff.net.0.proj", 28672, 5376);
            tensors.emplace_back(diffusers + ".ff.net.2", 5376, 14336);
        } else {
            tensors.emplace_back(native + ".attn.qkv_proj", 21504, 5376);
            tensors.emplace_back(native + ".attn.out_proj", 5376, 7168);
            tensors.emplace_back(native + ".mlp.fc1", 28672, 5376);
            tensors.emplace_back(native + ".mlp.fc2", 5376, 14336);
        }
        for (const auto &[name, rows, cols] : tensors) {
            if (impl_->nativePackedQkv)
                require(impl_->quantMetadata["layers"][name]["format"].string_value() == "float8_e4m3fn",
                        "inconsistent Comfy FP8 layout: " + name);
            auto d = impl_->descriptor(name + (impl_->unsloth ? "._weight_qdata" : ".weight"), "F8_E4M3");
            require(d["shape"].array_items().size() == 2 && d["shape"][0].number_value() == rows &&
                        d["shape"][1].number_value() == cols,
                    "invalid H3 Image FP8 matrix: " + name);
            auto scale =
                impl_->descriptor(name + (impl_->unsloth ? "._weight_scale" : ".weight_scale"), "F32");
            require(scale["shape"].is_array() && scale["shape"].array_items().size() <= 2,
                    "invalid FP8 scale rank: " + name);
            require(impl_->payloadBytes(scale) == size_t(impl_->unsloth ? rows : 1) * sizeof(float),
                    "invalid H3 Image FP8 scales: " + name);
        }
    }
}
Tensor Fp8Weights::linear(const Tensor &x, const std::string &name, Tensor *output) {
    return impl_->run(x, name, output);
}
Tensor Fp8Weights::linearPacked(const Tensor &x, const std::vector<std::string> &prefixes, Tensor *output) {
    require(!prefixes.empty(), "empty FP8 projection list");
    std::string key = "packed:";
    for (const auto &prefix : prefixes)
        key += std::to_string(prefix.size()) + ":" + prefix;
    return impl_->run(x, key, output, &prefixes);
}
size_t Fp8Weights::loadedBytes() const {
    size_t bytes = impl_->workspace.bytes() + impl_->one.bytes();
    for (const auto &entry : impl_->weights)
        bytes += entry.second.bytes.bytes() + entry.second.scales.bytes();
    return bytes;
}
void Fp8Weights::erasePrefix(const std::string &prefix) {
    for (auto i = impl_->weights.begin(); i != impl_->weights.end();)
        if (i->first.compare(0, prefix.size(), prefix) == 0)
            i = impl_->weights.erase(i);
        else
            ++i;
}
void Fp8Weights::clear() {
    impl_->weights.clear();
    impl_->plans.clear();
    impl_->workspace = {};
    impl_->one = {};
    if (impl_->handle) {
        cublasLtDestroy(impl_->handle);
        impl_->handle = nullptr;
    }
}
} // namespace fastllm::minimax_h3_image
#else
#include <stdexcept>
namespace fastllm::minimax_h3_image {
struct Fp8Weights::Impl {};
namespace {
void unavailable() { throw std::runtime_error("image FP8 requires a build with CUDA 11.8 or newer"); }
} // namespace
Fp8Weights::Fp8Weights(const std::string &) : QuantizedWeights(WeightFormat::FP8) { unavailable(); }
Fp8Weights::Fp8Weights(const std::string &path, const ProjectionOptions &) : Fp8Weights(path) {}
Fp8Weights::~Fp8Weights() = default;
void Fp8Weights::validateDevice(int, bool) { unavailable(); }
void Fp8Weights::validateModel() const { unavailable(); }
minimax_h3::Tensor Fp8Weights::linear(const minimax_h3::Tensor &, const std::string &, minimax_h3::Tensor *) {
    unavailable();
    return {};
}
void Fp8Weights::erasePrefix(const std::string &) {}
void Fp8Weights::clear() {}
minimax_h3::Tensor Fp8Weights::linearPacked(const minimax_h3::Tensor &, const std::vector<std::string> &,
                                          minimax_h3::Tensor *) { unavailable(); return {}; }
size_t Fp8Weights::loadedBytes() const { return 0; }
} // namespace fastllm::minimax_h3_image
#endif
