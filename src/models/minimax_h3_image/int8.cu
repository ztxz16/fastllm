#include "json11.hpp"
#include "quantized_backends.h"
#include "quantized_checkpoint.h"
#include "quantized_kernels.h"
#include <cuda_runtime.h>
#if CUDART_VERSION >= 11080
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>
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
        throw std::runtime_error("INT8 cuBLAS line " + std::to_string(line) + " status " +
                                 std::to_string(int(x)));
}
#define ck(call) checked((call), __LINE__)
void require(bool x, const std::string &message) {
    if (!x)
        throw std::runtime_error(message);
}
// Each warp owns a 256-element H4 Kronecker group. Values stay in
// registers through all four stages and row-wise quantization; no rotated
// BF16 matrix or block-wide transpose is materialized. Preserve BF16 rounding
// at the rotation boundary used by the native ConvRot reference.
template <int GroupsPerWarp, int Mode = 0>
__global__ void rotateQuantize(const B *x, int8_t *q, float *scales, int rows, int cols) {
    const int row = blockIdx.x, lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    constexpr int warps = 8;
    float v[GroupsPerWarp][8];
    float largest = 0;
#pragma unroll
    for (int group = 0; group < GroupsPerWarp; ++group) {
        const int offset = (group * warps + warp) * 256;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            v[group][j] =
                row < rows && offset < cols ? float(x[size_t(row) * cols + offset + j * 32 + lane]) : 0;
#pragma unroll
        for (int stride = 1; stride <= 4; stride *= 4) {
            const int base = lane & ~(3 * stride), digit = (lane / stride) % 4;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float a = __shfl_sync(0xffffffff, v[group][j], base);
                const float b = __shfl_sync(0xffffffff, v[group][j], base + stride);
                const float c = __shfl_sync(0xffffffff, v[group][j], base + 2 * stride);
                const float d = __shfl_sync(0xffffffff, v[group][j], base + 3 * stride);
                const float negative = __shfl_sync(0xffffffff, v[group][j], base + (3 - digit) * stride);
                v[group][j] = ((a + b) + c) + d - 2.f * negative;
            }
        }
// Stride 16 spans the high lane bit and the low register bit.
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
            const float a = __shfl_sync(0xffffffff, v[group][j], lane & 15);
            const float b = __shfl_sync(0xffffffff, v[group][j], (lane & 15) + 16);
            const float c = __shfl_sync(0xffffffff, v[group][j + 1], lane & 15);
            const float d = __shfl_sync(0xffffffff, v[group][j + 1], (lane & 15) + 16);
            const float sum = ((a + b) + c) + d;
            v[group][j] = sum - 2.f * (lane < 16 ? d : c);
            v[group][j + 1] = sum - 2.f * (lane < 16 ? b : a);
        }
// Stride 64 is entirely within each lane's registers.
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float a = v[group][j], b = v[group][j + 2], c = v[group][j + 4], d = v[group][j + 6];
            const float sum = ((a + b) + c) + d;
            v[group][j] = float(B((sum - 2.f * d) * (1.f / 16)));
            v[group][j + 2] = float(B((sum - 2.f * c) * (1.f / 16)));
            v[group][j + 4] = float(B((sum - 2.f * b) * (1.f / 16)));
            v[group][j + 6] = float(B((sum - 2.f * a) * (1.f / 16)));
        }
#pragma unroll
        for (int j = 0; j < 8; ++j)
            largest = fmaxf(largest, fabsf(v[group][j]));
    }
    float scale;
    if constexpr (Mode == 2) {
        scale = scales[row];
    } else {
        for (int d = 16; d; d /= 2)
            largest = fmaxf(largest, __shfl_xor_sync(0xffffffff, largest, d));
        __shared__ float maxima[warps];
        if (!lane)
            maxima[warp] = largest;
        __syncthreads();
        largest = lane < warps ? maxima[lane] : 0;
        for (int d = 16; d; d /= 2)
            largest = fmaxf(largest, __shfl_xor_sync(0xffffffff, largest, d));
        scale = fmaxf(largest / 127.f, 1e-12f);
        if (!threadIdx.x)
            scales[row] = scale;
    }
    if constexpr (Mode == 1) return;
#pragma unroll
    for (int group = 0; group < GroupsPerWarp; ++group) {
        const int offset = (group * warps + warp) * 256;
        if (offset < cols) {
#pragma unroll
            for (int j = 0; j < 8; ++j)
                q[size_t(row) * cols + offset + j * 32 + lane] =
                    int8_t(max(-128, min(127, __float2int_rn(v[group][j] / scale))));
        }
    }
}
// Mode 0 is the original fused path. TP row projections first compute the
// scale (1), share it across ranks, then rotate/quantize with that scale (2).
template<int Mode = 0>
void launchRotateQuantize(const B *input, int8_t *quantized, float *scales, int rows, int cols, int padded) {
    const int groups = (cols / 256 + 7) / 8;
    switch (groups) {
    case 1:
        rotateQuantize<1,Mode><<<padded,256,0,cudaStreamPerThread>>>(input,quantized,scales,rows,cols);
        break;
    case 2:
        rotateQuantize<2,Mode><<<padded,256,0,cudaStreamPerThread>>>(input,quantized,scales,rows,cols);
        break;
    case 3:
        rotateQuantize<3,Mode><<<padded,256,0,cudaStreamPerThread>>>(input,quantized,scales,rows,cols);
        break;
    case 4:
        rotateQuantize<4,Mode><<<padded,256,0,cudaStreamPerThread>>>(input,quantized,scales,rows,cols);
        break;
    case 6:
        rotateQuantize<6,Mode><<<padded,256,0,cudaStreamPerThread>>>(input,quantized,scales,rows,cols);
        break;
    case 7:
        rotateQuantize<7,Mode><<<padded,256,0,cudaStreamPerThread>>>(input,quantized,scales,rows,cols);
        break;
    default: throw std::runtime_error("unsupported ConvRot width");
    }
    ck(cudaGetLastError());
}
template<class Output>
__global__ void finish(const int32_t *raw, const float *xs, const float *ws, Output *out, size_t count, int cols) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(blockDim.x) * gridDim.x)
        out[i] = Output(float(raw[i]) * xs[i / cols] * ws[i % cols]);
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
class Int8Weights final : public QuantizedWeights {
  public:
    Int8Weights(const std::string &path, const ProjectionOptions &);
    ~Int8Weights() override;
    void validateModel() const override;
    Tensor linear(const Tensor &, const std::string &, Tensor *) override;
    Tensor linearPacked(const Tensor &, const std::vector<std::string> &, Tensor *) override;
    size_t loadedBytes() const override;
    void erasePrefix(const std::string &) override;
    void clear() override;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
struct Int8Weights::Impl : detail::QuantizedCheckpoint {
    ProjectionOptions options;
    bool unsloth = false, rotationVerified = false;
    std::string checkpointPath;
    cublasLtHandle_t handle = nullptr;
    Tensor workspace;
    int device = -1;
    bool fusedGemm = false;
    std::map<std::string, Weight> weights;
    std::map<std::string, std::unique_ptr<Plan>> plans;
    Impl(const std::string &path, const ProjectionOptions &o) : QuantizedCheckpoint(path), options(o), checkpointPath(path) {
        unsloth = header["transformer_blocks.0.attn.to_q._weight_qdata"].is_object();
        require(!options.comfyVideo || !unsloth, "video projections require the validated Comfy layout");
    }
    ~Impl() {
        cudaStreamSynchronize(cudaStreamPerThread);
        plans.clear();
        if (handle)
            cublasLtDestroy(handle);
    }
    void validateMarker(const std::string &key) {
        auto metadata = marker(key + ".comfy_quant");
        require(metadata["format"].string_value() == "int8_tensorwise" && metadata["convrot"].bool_value() &&
                    metadata["convrot_groupsize"].number_value() == 256,
                "unsupported INT8 ConvRot metadata: " + key);
    }
    void verifyUnslothRotation() {
        if (!unsloth || rotationVerified)
            return;
        // Some exports lose the ConvRot marker. Infer nothing from the file
        // name: verify representative rows of every projection against the
        // matching BF16 checkpoint before enabling the H256 transform.
        if (header["__metadata__"]["h3_convrot_groupsize"].string_value() == "256") {
            rotationVerified = true;
            return;
        }
        const auto directory =
            std::filesystem::path(resolveModelSource(checkpointPath).directory) / "transformer";
        std::ifstream indexFile(directory / "model.safetensors.index.json");
        std::string text((std::istreambuf_iterator<char>(indexFile)), {}), error;
        auto index = json11::Json::parse(text, error)["weight_map"];
        require(error.empty() && index.is_object(),
                "cannot verify unmarked INT8 rotation: missing BF16 weight index");
        std::map<std::string, std::unique_ptr<detail::QuantizedCheckpoint>> shards;
        for (int layer = 0; layer < 50; ++layer) {
            auto native = "blocks." + std::to_string(layer),
                 diffusers = "transformer_blocks." + std::to_string(layer);
            for (int projection = 0; projection < 6; ++projection) {
                const std::string suffixes[] = {".attn.to_q",     ".attn.to_k",     ".attn.to_v",
                                                ".attn.to_out.0", ".ff.net.0.proj", ".ff.net.2"};
                const std::string natives[] = {".attn.qkv_proj", ".attn.qkv_proj", ".attn.qkv_proj",
                                               ".attn.out_proj", ".mlp.fc1",       ".mlp.fc2"};
                const auto key = diffusers + suffixes[projection],
                           baseKey = native + natives[projection] + ".weight";
                const auto shardName = index[baseKey].string_value();
                require(!shardName.empty() && std::filesystem::path(shardName).filename() == shardName,
                        "invalid BF16 shard for INT8 verification");
                auto &shard = shards[shardName];
                if (!shard)
                    shard = std::make_unique<detail::QuantizedCheckpoint>((directory / shardName).string());
                auto original = shard->descriptor(baseKey, "BF16");
                auto quantized = descriptor(key + "._weight_qdata", "I8");
                auto scale = descriptor(key + "._weight_scale", "F32");
                const int rows = quantized["shape"][0].int_value(), cols = quantized["shape"][1].int_value();
                require(cols > 0 && cols % 256 == 0 && original["shape"][1].number_value() == cols,
                        "INT8 rotation verification shape mismatch");
                for (int row : {0, rows - 1}) {
                    int sourceRow = row;
                    if (projection < 3)
                        sourceRow = row / 128 * 384 + projection * 128 + row % 128;
                    if (projection == 4)
                        sourceRow = (row + rows / 2) % rows;
                    std::vector<uint16_t> bf16(cols);
                    std::vector<int8_t> payload(cols);
                    float multiplier = 0;
                    shard->readRange(original, size_t(sourceRow) * cols * 2, bf16.data(), size_t(cols) * 2);
                    readRange(quantized, size_t(row) * cols, payload.data(), cols);
                    readRange(scale, size_t(row) * 4, &multiplier, 4);
                    require(std::isfinite(multiplier) && multiplier > 0, "invalid INT8 weight scale");
                    std::vector<float> rotated(cols), expected(cols);
                    for (int c = 0; c < cols; ++c) {
                        uint32_t bits = uint32_t(bf16[c]) << 16;
                        std::memcpy(&expected[c], &bits, 4);
                        rotated[c] = float(payload[c]) * multiplier;
                    }
                    for (int group = 0; group < cols; group += 256) {
                        for (int stride = 1; stride <= 64; stride *= 4) {
                            for (int b = 0; b < 256; b += 4 * stride)
                                for (int j = 0; j < stride; ++j) {
                                    const int i = group + b + j;
                                    float a = rotated[i], b = rotated[i + stride],
                                          c = rotated[i + 2 * stride], d = rotated[i + 3 * stride];
                                    float sum = ((a + b) + c) + d;
                                    rotated[i] = sum - 2 * d;
                                    rotated[i + stride] = sum - 2 * c;
                                    rotated[i + 2 * stride] = sum - 2 * b;
                                    rotated[i + 3 * stride] = sum - 2 * a;
                                }
                        }
                    }
                    double err = 0, plainErr = 0, norm = 0;
                    for (int c = 0; c < cols; ++c) {
                        double delta = rotated[c] / 16.f - expected[c],
                               plain = float(payload[c]) * multiplier - expected[c];
                        err += delta * delta;
                        plainErr += plain * plain;
                        norm += double(expected[c]) * expected[c];
                    }
                    require(norm > 1e-20 && std::sqrt(err / norm) < .12 && err < plainErr * .0625,
                            "unmarked INT8 checkpoint does not match H256 ConvRot: " + key);
                }
            }
        }
        rotationVerified = true;
    }
    Weight &get(const std::string &name, const std::vector<std::string> *direct = nullptr) {
        require(!direct || !unsloth, "direct INT8 requires Comfy weights");
        verifyUnslothRotation();
        auto found = weights.find(name);
        if (found != weights.end())
            return found->second;
        std::vector<std::string> keys;
        bool qkv = name.find(".qkv_proj") != std::string::npos;
        if (direct) {
            keys = *direct;
            qkv = false;
        } else if (unsloth) {
            auto dot = name.find('.', 7);
            require(dot != std::string::npos, "invalid INT8 layer name");
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
                throw std::runtime_error("unsupported INT8 projection");
        } else
            keys = {name};
        Weight w;
        std::vector<json11::Json> matrices;
        std::vector<float> scales;
        for (auto &key : keys) {
            auto matrix = descriptor(key + (unsloth ? "._weight_qdata" : ".weight"), "I8");
            auto shape = matrix["shape"].array_items();
            require(shape.size() == 2 && shape[0].int_value() > 0 && shape[1].int_value() > 0,
                    "INT8 matrix shape");
            int rows = shape[0].int_value(), cols = shape[1].int_value();
            require(!w.cols || w.cols == cols, "inconsistent INT8 QKV");
            w.cols = cols;
            w.rows += rows;
            matrices.push_back(matrix);
            auto sd = descriptor(key + (unsloth ? "._weight_scale" : ".weight_scale"), "F32");
            std::vector<float> part(rows);
            read(sd, part.data(), part.size() * 4);
            for (float v : part)
                require(std::isfinite(v) && v > 0, "invalid INT8 weight scale");
            if (unsloth) {
                std::vector<int8_t> zeros(rows);
                read(descriptor(key + "._weight_zero_point", "I8"), zeros.data(), zeros.size());
                require(std::all_of(zeros.begin(), zeros.end(), [](int8_t x) { return x == 0; }),
                        "asymmetric INT8 weights are not supported");
            } else
                validateMarker(key);
            scales.insert(scales.end(), part.begin(), part.end());
        }
        require(w.cols % 16 == 0 && w.rows % 16 == 0, "unaligned INT8 matrix");
        if (qkv)
            require(w.rows == 21504 && w.cols == 5376, "INT8 QKV layout");
        detail::ProjectionSlice slice(direct ? direct->front() : name, w.rows, w.cols, options,
                                      direct ? int(direct->size()) : 1);
        w.rows=slice.rows; w.cols=slice.cols; w.rowParallel=slice.rowParallel;
        if(options.size>1) scales=slice.scales(scales);
        w.bytes = Tensor(w.rows, w.cols / 2);
        uploadByteMatrices(matrices, w.bytes.data, slice);
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
#ifdef FASTLLM_IMAGE_INT8_SCALED
            cudaDeviceProp properties{};
            ck(cudaGetDeviceProperties(&properties, device));
            fusedGemm = properties.major >= 8;
#endif
        }
        auto &w = get(name, direct);
        require(x.dtype == DType::BF16 && x.cols == w.cols, "INT8 input dtype/shape");
        int padded = (x.rows + 15) / 16 * 16;
        Tensor q(padded, x.cols / 2), scale(1, padded, true);
        Tensor out = destination ? *destination : Tensor(x.rows, w.rows);
        require(out.rows == x.rows && out.cols == w.rows && (out.dtype == DType::BF16 || out.dtype == DType::FP32), "INT8 output shape");
        auto *input = reinterpret_cast<const B *>(x.data);
        auto *quantized = reinterpret_cast<int8_t *>(q.data);
        auto *scales = reinterpret_cast<float *>(scale.data);
        require(x.cols % 256 == 0, "unaligned ConvRot input");
        // H3's callback calibrates NVFP4; shared INT8 scales are Qwen-specific.
        if (w.rowParallel && options.qwenImage && options.activationMaximum) {
            launchRotateQuantize<1>(input,quantized,scales,x.rows,x.cols,padded);
            options.activationMaximum(scale);
            launchRotateQuantize<2>(input,quantized,scales,x.rows,x.cols,padded);
        } else {
            launchRotateQuantize(input,quantized,scales,x.rows,x.cols,padded);
        }
#ifdef FASTLLM_IMAGE_INT8_SCALED
        if (fusedGemm && w.rowParallel && options.rowReduction) {
            Tensor raw(x.rows,w.rows,true);
            detail::int8RawGemm(raw.data,q.data,w.bytes.data,x.rows,w.rows,w.cols,cudaStreamPerThread);
            options.rowReduction(raw,scale,w.scales,out);
            return out;
        }
        if (fusedGemm) {
            detail::int8ScaledGemm(out.data, q.data, w.bytes.data, scales,
                                   reinterpret_cast<const float *>(w.scales.data), x.rows, w.rows, w.cols,
                                   cudaStreamPerThread, out.isFloat32());
            return out;
        }
#endif
        if (!workspace.data)
            workspace = Tensor(1, (32 << 20) / 2);
        if (!handle)
            ck(cublasLtCreate(&handle));
        Tensor raw(padded, w.rows, true);
        auto key = std::to_string(padded) + ":" + std::to_string(w.rows) + ":" + std::to_string(w.cols);
        auto &plan = plans[key];
        if (!plan) {
            auto p = std::make_unique<Plan>();
            ck(cublasLtMatmulDescCreate(&p->op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
            cublasOperation_t transpose = CUBLAS_OP_T;
            ck(cublasLtMatmulDescSetAttribute(p->op, CUBLASLT_MATMUL_DESC_TRANSA, &transpose,
                                              sizeof(transpose)));
            ck(cublasLtMatrixLayoutCreate(&p->a, CUDA_R_8I, w.cols, w.rows, w.cols));
            ck(cublasLtMatrixLayoutCreate(&p->b, CUDA_R_8I, x.cols, padded, x.cols));
            ck(cublasLtMatrixLayoutCreate(&p->c, CUDA_R_32I, w.rows, padded, w.rows));
            ck(cublasLtMatmulPreferenceCreate(&p->pref));
            size_t bytes = workspace.bytes();
            ck(cublasLtMatmulPreferenceSetAttribute(p->pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &bytes,
                                                    sizeof(bytes)));
            cublasLtMatmulHeuristicResult_t choice{};
            int count = 0;
            ck(cublasLtMatmulAlgoGetHeuristic(handle, p->op, p->a, p->b, p->c, p->c, p->pref, 1, &choice,
                                              &count));
            require(count == 1, "no native INT8 cuBLASLt algorithm: " + key);
            p->algorithm = choice.algo;
            plan = std::move(p);
        }
        int32_t alpha = 1, beta = 0;
        ck(cublasLtMatmul(handle, plan->op, &alpha, w.bytes.data, plan->a, q.data, plan->b, &beta, raw.data,
                          plan->c, raw.data, plan->c, &plan->algorithm, workspace.data, workspace.bytes(),
                          cudaStreamPerThread));
        if (w.rowParallel && options.rowReduction) {
            options.rowReduction(raw.rowSlice(0,x.rows),scale,w.scales,out);
            return out;
        }
        if(out.isFloat32())
            finish<<<4096,256,0,cudaStreamPerThread>>>((int32_t *)raw.data,(float *)scale.data,
                (float *)w.scales.data,(float *)out.data,out.size(),w.rows);
        else
        finish<<<4096, 256, 0, cudaStreamPerThread>>>(
            reinterpret_cast<int32_t *>(raw.data), reinterpret_cast<float *>(scale.data),
            reinterpret_cast<float *>(w.scales.data), reinterpret_cast<B *>(out.data), out.size(), w.rows);
        ck(cudaGetLastError());
        return out;
    }
};
Int8Weights::Int8Weights(const std::string &path, const ProjectionOptions &options)
    : QuantizedWeights(WeightFormat::INT8), impl_(std::make_unique<Impl>(path,options)) {}
Int8Weights::~Int8Weights() = default;
void detail::validateInt8Device(int device) {
    cudaDeviceProp properties{};
    ck(cudaGetDeviceProperties(&properties, device));
    require(properties.major >= 8, "image INT8 requires SM80 or newer for BF16 activations");
}
void Int8Weights::validateModel() const {
    impl_->verifyUnslothRotation();
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
            if (!impl_->unsloth)
                impl_->validateMarker(name);
            auto d = impl_->descriptor(name + (impl_->unsloth ? "._weight_qdata" : ".weight"), "I8");
            require(d["shape"].array_items().size() == 2 && d["shape"][0].number_value() == rows &&
                        d["shape"][1].number_value() == cols,
                    "invalid H3 Image INT8 matrix: " + name);
            auto scale =
                impl_->descriptor(name + (impl_->unsloth ? "._weight_scale" : ".weight_scale"), "F32");
            require(scale["shape"].is_array() && scale["shape"].array_items().size() <= 2,
                    "invalid INT8 scale rank: " + name);
            require(impl_->payloadBytes(scale) == size_t(rows) * sizeof(float),
                    "invalid H3 Image INT8 scales: " + name);
        }
    }
}
Tensor Int8Weights::linear(const Tensor &x, const std::string &name, Tensor *output) {
    return impl_->run(x, name, output);
}
Tensor Int8Weights::linearPacked(const Tensor &x, const std::vector<std::string> &names, Tensor *output) {
    require(!names.empty(), "empty INT8 projection list");
    std::string key = "packed:";
    for (const auto &name : names)
        key += std::to_string(name.size()) + ":" + name;
    return impl_->run(x, key, output, &names);
}
size_t Int8Weights::loadedBytes() const {
    size_t bytes = impl_->workspace.bytes();
    for (const auto &entry : impl_->weights)
        bytes += entry.second.bytes.bytes() + entry.second.scales.bytes();
    return bytes;
}
void Int8Weights::erasePrefix(const std::string &prefix) {
    for (auto i = impl_->weights.begin(); i != impl_->weights.end();)
        if (i->first.compare(0, prefix.size(), prefix) == 0)
            i = impl_->weights.erase(i);
        else
            ++i;
}
void Int8Weights::clear() {
    impl_->weights.clear();
    impl_->plans.clear();
    impl_->workspace = {};
    if (impl_->handle) {
        cublasLtDestroy(impl_->handle);
        impl_->handle = nullptr;
    }
}
std::unique_ptr<QuantizedWeights> detail::loadInt8(const std::string &path, const ProjectionOptions &options) {
    return std::make_unique<Int8Weights>(path, options);
}
} // namespace fastllm::minimax_h3_image
#else
#include <stdexcept>
namespace fastllm::minimax_h3_image::detail {
void validateInt8Device(int) { throw std::runtime_error("image INT8 requires CUDA 11.8 or newer"); }
std::unique_ptr<QuantizedWeights> loadInt8(const std::string &, const ProjectionOptions &) {
    validateInt8Device(0);
    return {};
}
} // namespace fastllm::minimax_h3_image::detail
#endif
