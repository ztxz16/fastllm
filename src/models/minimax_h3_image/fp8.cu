#include "models/minimax_h3_image/fp8.h"
#include "json11.hpp"
#include <cuda_runtime.h>
#if CUDART_VERSION >= 11080
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <stdexcept>
#include <vector>
#include <tuple>
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
void require(bool x, const std::string &message) {
    if (!x)
        throw std::runtime_error(message);
}
__global__ void quantize(const B *x, uint8_t *q, float *scales, int rows, int cols, int padded) {
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
        maxima[0] = fmaxf(largest / 448.f, 1e-12f);
    __syncthreads();
    float scale = maxima[0];
    if (threadIdx.x == 0)
        scales[row] = scale;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = row < rows ? float(x[size_t(row) * cols + c]) / scale : 0;
        q[size_t(row) * cols + c] = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    }
}
__global__ void finish(const B *raw, const float *xs, const float *ws, B *out, size_t count, int cols) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(blockDim.x) * gridDim.x)
        out[i] = B(float(raw[i]) * xs[i / cols] * ws[i % cols]);
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
};
} // namespace
struct Fp8Weights::Impl {
    std::ifstream file;
    json11::Json header;
    uint64_t base = 0, length = 0;
    bool unsloth = false, nativePackedQkv = false;
    json11::Json quantMetadata;
    cublasLtHandle_t handle = nullptr;
    Tensor workspace, one;
    std::map<std::string, Weight> weights;
    std::map<std::string, std::unique_ptr<Plan>> plans;
    explicit Impl(const std::string &path) : file(path, std::ios::binary) {
        require(bool(file), "cannot open FP8 checkpoint");
        file.seekg(0, std::ios::end);
        length = uint64_t(file.tellg());
        file.seekg(0);
        uint64_t n = 0;
        file.read(reinterpret_cast<char *>(&n), 8);
        require(n > 0 && n < (16 << 20) && n + 8 < length, "invalid FP8 header size");
        std::string data(n, '\0'), error;
        file.read(data.data(), n);
        require(bool(file), "truncated FP8 header");
        header = json11::Json::parse(data, error);
        require(error.empty() && header.is_object(), "invalid FP8 JSON");
        base = n + 8;
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
    }
    ~Impl() {
        cudaStreamSynchronize(cudaStreamPerThread);
        plans.clear();
        if (handle)
            cublasLtDestroy(handle);
    }
    json11::Json descriptor(const std::string &name, const std::string &dtype) const {
        auto d = header[name];
        require(d.is_object() && d["dtype"].string_value() == dtype, "invalid FP8 tensor " + name);
        auto offsets = d["data_offsets"].array_items();
        require(offsets.size() == 2 && offsets[0].is_number() && offsets[1].is_number() &&
                    offsets[0].number_value() >= 0 &&
                    std::floor(offsets[0].number_value()) == offsets[0].number_value() &&
                    std::floor(offsets[1].number_value()) == offsets[1].number_value() &&
                    offsets[1].number_value() >= offsets[0].number_value() &&
                    offsets[1].number_value() <= double(length - base),
                "invalid offsets " + name);
        return d;
    }
    void read(const json11::Json &d, void *dest, size_t bytes) {
        auto offsets = d["data_offsets"].array_items();
        require(size_t(offsets[1].number_value() - offsets[0].number_value()) == bytes,
                "FP8 tensor byte size mismatch");
        file.clear();
        file.seekg(base + uint64_t(offsets[0].number_value()));
        file.read(static_cast<char *>(dest), bytes);
        require(bool(file), "truncated FP8 payload");
    }
    void upload(const json11::Json &d, void *dest, size_t bytes) {
        auto offsets = d["data_offsets"].array_items();
        require(size_t(offsets[1].number_value() - offsets[0].number_value()) == bytes,
                "FP8 matrix byte size mismatch");
        file.clear();
        file.seekg(base + uint64_t(offsets[0].number_value()));
        std::vector<char> buffer(std::min(bytes, size_t(8) << 20));
        for (size_t pos = 0; pos < bytes;) {
            size_t n = std::min(buffer.size(), bytes - pos);
            file.read(buffer.data(), n);
            require(bool(file), "truncated FP8 matrix");
            ck(cudaMemcpyAsync(static_cast<char *>(dest) + pos, buffer.data(), n, cudaMemcpyHostToDevice,
                               cudaStreamPerThread));
            ck(cudaStreamSynchronize(cudaStreamPerThread));
            pos += n;
        }
    }
    Weight &get(const std::string &name, bool packed) {
        auto found = weights.find(name);
        if (found != weights.end())
            return found->second;
        std::vector<std::string> keys;
        bool qkv = name.find(".qkv_proj") != std::string::npos;
        if (unsloth) {
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
            auto matrix = descriptor(key + (unsloth ? "._weight_qdata" : ".weight"), "F8_E4M3");
            auto shape = matrix["shape"].array_items();
            require(shape.size() == 2 && shape[0].int_value() > 0 && shape[1].int_value() > 0,
                    "FP8 matrix shape");
            int rows = shape[0].int_value(), cols = shape[1].int_value();
            require(!w.cols || w.cols == cols, "inconsistent FP8 QKV");
            w.cols = cols;
            w.rows += rows;
            matrices.push_back(matrix);
            auto sd = descriptor(key + (unsloth ? "._weight_scale" : ".weight_scale"), "F32");
            std::vector<float> part(unsloth ? rows : 1);
            read(sd, part.data(), part.size() * 4);
            for (float v : part)
                require(std::isfinite(v) && v > 0, "invalid FP8 weight scale");
            if (!unsloth)
                part.resize(rows, part[0]);
            scales.insert(scales.end(), part.begin(), part.end());
        }
        require(w.cols % 16 == 0 && w.rows % 16 == 0, "unaligned FP8 matrix");
        if (qkv)
            require(packed && w.rows == 21504 && w.cols == 5376, "FP8 QKV layout");
        w.bytes = Tensor(w.rows, w.cols / 2);
        size_t offset = 0;
        for (auto &d : matrices) {
            size_t bytes = size_t(d["shape"][0].int_value()) * w.cols;
            upload(d, reinterpret_cast<char *>(w.bytes.data) + offset, bytes);
            offset += bytes;
        }
        // Diffusers stores [value, gate]; the original H3 uses [gate, value].
        // Apply the identical row permutation to both payload and per-row scales.
        if (unsloth && name.find(".fc1") != std::string::npos) {
            Tensor ordered(w.rows, w.cols / 2);
            size_t half = w.bytes.bytes() / 2;
            ck(cudaMemcpyAsync(ordered.data, reinterpret_cast<char *>(w.bytes.data) + half, half,
                               cudaMemcpyDeviceToDevice, cudaStreamPerThread));
            ck(cudaMemcpyAsync(reinterpret_cast<char *>(ordered.data) + half, w.bytes.data, half,
                               cudaMemcpyDeviceToDevice, cudaStreamPerThread));
            w.bytes = std::move(ordered);
            std::rotate(scales.begin(), scales.begin() + w.rows / 2, scales.end());
        }
        if (qkv && !unsloth && !nativePackedQkv) {
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
    Tensor run(const Tensor &x, const std::string &name, bool packed, Tensor *destination) {
        if (!handle) {
            ck(cublasLtCreate(&handle));
            workspace = Tensor(1, (32 << 20) / 2);
            one = Tensor(1, 1, true);
            float value = 1;
            ck(cudaMemcpyAsync(one.data, &value, 4, cudaMemcpyHostToDevice, cudaStreamPerThread));
            ck(cudaStreamSynchronize(cudaStreamPerThread));
        }
        auto &w = get(name, packed);
        require(x.dtype == DType::BF16 && x.cols == w.cols, "FP8 input dtype/shape");
        int padded = (x.rows + 15) / 16 * 16;
        Tensor q(padded, x.cols / 2), scale(1, padded, true), raw(padded, w.rows);
        Tensor out = destination ? *destination : Tensor(x.rows, w.rows);
        require(out.rows == x.rows && out.cols == w.rows && out.dtype == DType::BF16, "FP8 output shape");
        quantize<<<padded, 256, 0, cudaStreamPerThread>>>(
            reinterpret_cast<B *>(x.data), reinterpret_cast<uint8_t *>(q.data),
            reinterpret_cast<float *>(scale.data), x.rows, x.cols, padded);
        ck(cudaGetLastError());
        auto key = std::to_string(padded) + ":" + std::to_string(w.rows) + ":" + std::to_string(w.cols);
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
            ck(cublasLtMatrixLayoutCreate(&p->c, CUDA_R_16BF, w.rows, padded, w.rows));
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
        finish<<<4096, 256, 0, cudaStreamPerThread>>>(
            reinterpret_cast<B *>(raw.data), reinterpret_cast<float *>(scale.data),
            reinterpret_cast<float *>(w.scales.data), reinterpret_cast<B *>(out.data), out.size(), w.rows);
        ck(cudaGetLastError());
        return out;
    }
};
Fp8Weights::Fp8Weights(const std::string &path) : impl_(std::make_unique<Impl>(path)) {}
Fp8Weights::~Fp8Weights() = default;
void Fp8Weights::validateDevice(int device, bool multipleDevices) {
    cudaDeviceProp properties{};
    ck(cudaGetDeviceProperties(&properties, device));
    require(properties.major * 10 + properties.minor >= 89,
            "image FP8 requires an SM89 or newer CUDA device");
    // CUDA 13.0's cuBLAS can fail FP8 launches across Blackwell devices (CUB-9487).
    require(!multipleDevices || properties.major < 12 || cublasLtGetVersion() >= 130201,
            "multi-GPU Blackwell FP8 requires cuBLAS 13.2.1 or newer; update the cuBLAS runtime libraries");
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
                        d["shape"][1].number_value() == cols &&
                        d["data_offsets"][1].number_value() - d["data_offsets"][0].number_value() ==
                            double(rows) * cols,
                    "invalid H3 Image FP8 matrix: " + name);
            auto scale =
                impl_->descriptor(name + (impl_->unsloth ? "._weight_scale" : ".weight_scale"), "F32");
            require(scale["shape"].is_array() && scale["shape"].array_items().size() <= 2,
                    "invalid FP8 scale rank: " + name);
            size_t elements = 1;
            for (const auto &dimension : scale["shape"].array_items()) {
                require(dimension.is_number() && dimension.number_value() >= 1 &&
                            dimension.number_value() <= rows &&
                            std::floor(dimension.number_value()) == dimension.number_value() &&
                            elements <= size_t(rows) / size_t(dimension.number_value()),
                        "invalid FP8 scale shape: " + name);
                elements *= size_t(dimension.number_value());
            }
            require(elements == size_t(impl_->unsloth ? rows : 1) &&
                        scale["data_offsets"][1].number_value() - scale["data_offsets"][0].number_value() ==
                            elements * 4,
                    "invalid H3 Image FP8 scales: " + name);
        }
    }
}
Tensor Fp8Weights::linear(const Tensor &x, const std::string &name, Tensor *output) {
    return impl_->run(x, name, true, output);
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
Fp8Weights::Fp8Weights(const std::string &) { unavailable(); }
Fp8Weights::~Fp8Weights() = default;
void Fp8Weights::validateDevice(int, bool) { unavailable(); }
void Fp8Weights::validateModel() const { unavailable(); }
minimax_h3::Tensor Fp8Weights::linear(const minimax_h3::Tensor &, const std::string &,
                                      minimax_h3::Tensor *) {
    unavailable();
    return {};
}
void Fp8Weights::erasePrefix(const std::string &) {}
void Fp8Weights::clear() {}
} // namespace fastllm::minimax_h3_image
#endif
