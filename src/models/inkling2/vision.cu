// Inkling 2 vision tower and image adapter. Checkpoint and processor specification:
// https://huggingface.co/thinkingmachines/Inkling-2-Mini
// https://github.com/huggingface/transformers/tree/main/src/transformers/models/inkling2
#include "runtime.cuh"
#include "devices/cpu/cpudevice.h"
#include "devices/cuda/cudadevice.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <algorithm>
#include <climits>
#include <cmath>
#include <type_traits>

namespace fastllm::inkling2 {
namespace {
constexpr int imageThreads = 256;

template <class T>
__global__ void packPixels(const uint8_t *rgb, T *patches, int height, int width, int channels, int columns, int patch,
                           size_t elements) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const int area = patch * patch, tile = i / (3 * area), pixel = (i / 3) % area;
    const int y = (tile / columns) * patch + pixel / patch, x = (tile % columns) * patch + pixel % patch;
    int value = -1; // Padding on the original 0..255 scale.
    if (y < height && x < width) {
        const auto *p = rgb + (size_t(y) * width + x) * channels;
        value = p[i % 3];
        if (channels == 4) {
            const int bg = ((y / 8 + x / 8) % 2 == 0) ? 204 : 153;
            value = (value * p[3] + bg * (255 - p[3]) + 127) / 255;
        }
    }
    patches[i] = T(value * (2.f / 255.f) - 1.f);
}

template <class T>
__global__ void addPositions(T *hidden, const float *table, const float *cls, const float *clsPos, int rows, int cols,
                             int grid, int width) {
    const int row = blockIdx.x;
    const float y = fminf(grid - 1.f, fmaxf(0.f, ((row - 1) / cols + .5f) * grid / rows - .5f));
    const float x = fminf(grid - 1.f, fmaxf(0.f, ((row - 1) % cols + .5f) * grid / cols - .5f));
    const int y0 = int(y), x0 = int(x);
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
        if (!row) {
            hidden[c] = T(cls[c] + clsPos[c]);
            continue;
        }
        const float *a = table + size_t(y0 * grid) * width + c;
        const float *b = table + size_t(min(y0 + 1, grid - 1) * grid) * width + c;
        const int j = x0 * width, k = min(x0 + 1, grid - 1) * width;
        const float ax = x - x0, ay = y - y0;
        const T pos = T((1 - ay) * ((1 - ax) * a[j] + ax * a[k]) + ay * ((1 - ax) * b[j] + ax * b[k]));
        const size_t i = size_t(row) * width + c;
        hidden[i] = T(float(hidden[i]) + float(pos));
    }
}

template <class T>
__global__ void visionNorm(const T *input, T *out, const float *w, const float *b, int width, float eps) {
    __shared__ float scratch[imageThreads];
    const size_t start = size_t(blockIdx.x) * width;
    float sum = 0.f;
    for (int c = threadIdx.x; c < width; c += blockDim.x)
        sum += float(input[start + c]);
    const float mean = blockSum(sum, scratch) / width;
    sum = 0.f;
    for (int c = threadIdx.x; c < width; c += blockDim.x) {
        const float v = float(input[start + c]) - mean;
        sum += v * v;
    }
    const float scale = rsqrtf(blockSum(sum, scratch) / width + eps);
    for (int c = threadIdx.x; c < width; c += blockDim.x)
        out[start + c] = T((float(input[start + c]) - mean) * scale * w[c] + b[c]);
}

template <class T> __global__ void rotateQK(T *q, T *k, size_t pairs, int dim, int width, int columns) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= pairs) return;
    const int row = (2 * i) / width, pair = i % (dim / 2);
    if (!row) return;
    const int coord = 1 + (pair < dim / 4 ? (row - 1) % columns : (row - 1) / columns);
    float sn, cs;
    sincosf(coord * powf(10000.f, -4.f * (pair % (dim / 4)) / dim), &sn, &cs);
    auto rotate = [&](T *p) {
        const float a = float(p[2 * i]), b = float(p[2 * i + 1]);
        p[2 * i] = T(a * cs - b * sn);
        p[2 * i + 1] = T(a * sn + b * cs);
    };
    rotate(q);
    rotate(k);
}

template <class T> __global__ void residual(T *hidden, const T *value, const float *gamma, size_t n, int width) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) hidden[i] = T(float(hidden[i]) + float(T(float(value[i]) * gamma[i % width])));
}

template <class T> __global__ void gelu(T *values, size_t count) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        const float v = float(values[i]);
        values[i] = T(.5f * v * erfcf(-v * .7071067811865475f));
    }
}

template <class T>
__global__ void mergePatches(const T *hidden, T *merged, size_t elements, int width, int columns, int merge) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const int patch = i / width, token = patch / (merge * merge), sub = patch % (merge * merge);
    const int y = token / columns * merge + sub / merge, x = token % columns * merge + sub % merge;
    merged[i] = hidden[(size_t(1) + size_t(y) * columns * merge + x) * width + i % width];
}

unsigned blocks(size_t n) { return unsigned((n + imageThreads - 1) / imageThreads); }

template <class T> constexpr DataType dtype = std::is_same_v<T, half> ? FLOAT16 : BFLOAT16;

// Reuse FastLLM's noncausal Attention. Head-major FP32 also runs on GPUs without BF16 MMA.
// Query tiles bound the score workspace of very large images.
template <class T, bool Unpack>
__global__ void headRows(T *packed, float *heads, size_t count, int rows, int width, int dim) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int row = i / width, col = i % width;
    const size_t j = (size_t(col / dim) * rows + row) * dim + col % dim;
    if constexpr (Unpack)
        heads[j] = float(packed[i]);
    else
        packed[i] = T(heads[j]);
}

template <class T> Rows<T> attend(Rows<T> &q, Rows<T> &k, Rows<T> &v, int heads, int columns) {
    const int n = q.rows, width = q.cols, dim = width / heads;
    const size_t size = size_t(n) * width;
    rotateQK<<<blocks(size / 2), imageThreads, 0, cudaStreamPerThread>>>(q.p, k.p, size / 2, dim, width, columns);
    auto unpack = [&](Rows<T> &input) {
        Matrix result(n, width);
        headRows<T, true>
            <<<blocks(size), imageThreads, 0, cudaStreamPerThread>>>(input.p, result.p, size, n, width, dim);
        return result;
    };
    auto qh = unpack(q), kh = unpack(k), vh = unpack(v);
    Matrix out(n, width);
    auto run = [&](int h, int count, int first, int length) {
        const size_t base = size_t(h) * n * dim, pos = base + size_t(first) * dim;
        Data query(FLOAT32, {count, length, dim}, CUDA, qh.p + pos), key(FLOAT32, {count, n, dim}, CUDA, kh.p + base);
        Data value(FLOAT32, {count, n, dim}, CUDA, vh.p + base),
            result(FLOAT32, {count, length, dim}, CUDA, out.p + pos);
        if (!FastllmCudaAttention(query, key, value, *GetEmptyData(), result, 1, 1.f / std::sqrt(float(dim)), 2))
            throw std::runtime_error("Inkling vision attention failed");
    };
    if (n <= 256)
        run(0, heads, 0, n);
    else
        for (int h = 0; h < heads; ++h)
            for (int start = 0; start < n; start += 256)
                run(h, 1, start, std::min(256, n - start));
    Rows<T> result(n, width);
    headRows<T, false><<<blocks(size), imageThreads, 0, cudaStreamPerThread>>>(result.p, out.p, size, n, width, dim);
    return result;
}

template <class T>
Matrix encodeImage(Data &image, const Config &config, WeightMap &weights, int gpu, int rows, int cols) {
    const auto &c = config.vision;
    const int w = c.width, tokens = rows * cols, ny = rows * c.merge, nx = cols * c.merge, n = ny * nx + 1;
    auto weight = [&](const std::string &key, DataType type, std::vector<int> dims) -> Data & {
        auto it = weights.weight.find("model.visual." + key);
        if (it == weights.weight.end() || it->second.dataType != type || it->second.dims != dims)
            throw std::runtime_error("Missing or incompatible Inkling vision weight: " + key);
        it->second.ToDevice(CUDA, std::vector<int>{gpu}, true);
        return it->second;
    };
    auto vector = [&](const std::string &key) {
        return static_cast<const float *>(weight(key, FLOAT32, {w}).cudaData);
    };
    auto linear = [&](const Rows<T> &in, const std::string &key, int outWidth, bool bias) {
        auto &matrix = weight(key + ".weight", dtype<T>, {outWidth, in.cols});
        auto &b = bias ? weight(key + ".bias", FLOAT32, {outWidth}) : *GetEmptyData();
        Rows<T> result(in.rows, outWidth);
        Data input(dtype<T>, {in.rows, in.cols}, CUDA, in.p), output(dtype<T>, {in.rows, outWidth}, CUDA, result.p);
        DoCudaLinear(input, matrix, b, output);
        return result;
    };
    auto norm = [&](const Rows<T> &in, const std::string &key) {
        Rows<T> result(n, w);
        visionNorm<<<n, imageThreads, 0, cudaStreamPerThread>>>(in.p, result.p, vector(key + ".weight"),
                                                                vector(key + ".bias"), w, c.eps);
        return result;
    };
    const int patchWidth = 3 * c.patch * c.patch;
    Rows<T> pixels(n - 1, patchWidth);
    // Keep the request-owned HTTP payload on the host after image prefill.
    image.ToDevice(CPU);
    Buffer<uint8_t> rgb(image.Count(0));
    check(cudaMemcpyAsync(rgb.p, image.cpuData, image.Count(0), cudaMemcpyHostToDevice, cudaStreamPerThread));
    const size_t pixelCount = size_t(n - 1) * patchWidth;
    packPixels<<<blocks(pixelCount), imageThreads, 0, cudaStreamPerThread>>>(
        rgb.p, pixels.p, image.dims[0], image.dims[1], image.dims[2], nx, c.patch, pixelCount);
    const std::string emb = "backbone.embeddings.";
    rgb = {};
    auto embedded = linear(pixels, emb + "patch_embedding", w, false);
    Rows<T> hidden(n, w);
    check(cudaMemcpyAsync(hidden.p + w, embedded.p, size_t(n - 1) * w * sizeof(T), cudaMemcpyDeviceToDevice,
                          cudaStreamPerThread));
    pixels = {};
    embedded = {};
    addPositions<<<n, imageThreads, 0, cudaStreamPerThread>>>(
        hidden.p,
        static_cast<const float *>(weight(emb + "position_embedding", FLOAT32, {c.grid * c.grid, w}).cudaData),
        vector(emb + "class_embedding"), vector(emb + "cls_position_embedding"), ny, nx, c.grid, w);
    hidden = norm(hidden, "backbone.ln_pre");
    const size_t count = size_t(n) * w;
    for (int l = 0; l < c.layers; ++l) {
        const std::string pre = "backbone.layers." + std::to_string(l) + ".";
        {
            auto x = norm(hidden, pre + "ln_1");
            auto q = linear(x, pre + "self_attn.q_proj", w, true), k = linear(x, pre + "self_attn.k_proj", w, true);
            auto v = linear(x, pre + "self_attn.v_proj", w, true);
            auto attended = attend(q, k, v, c.heads, nx);
            auto proj = linear(attended, pre + "self_attn.out_proj", w, true);
            residual<<<blocks(count), imageThreads, 0, cudaStreamPerThread>>>(hidden.p, proj.p, vector(pre + "ls_1.gamma"),
                                                                              count, w);
        }
        auto x = norm(hidden, pre + "ln_2");
        const auto &ff = weights.weight.at("model.visual." + pre + "mlp.c_fc.weight");
        if (ff.dims.size() != 2 || ff.dims[0] <= 0) throw std::runtime_error("Invalid Inkling vision MLP");
        auto fc = linear(x, pre + "mlp.c_fc", ff.dims[0], true);
        const size_t nf = size_t(n) * fc.cols;
        gelu<<<blocks(nf), imageThreads, 0, cudaStreamPerThread>>>(fc.p, nf);
        auto mlp = linear(fc, pre + "mlp.c_proj", w, true);
        residual<<<blocks(count), imageThreads, 0, cudaStreamPerThread>>>(hidden.p, mlp.p, vector(pre + "ls_2.gamma"),
                                                                          count, w);
    }
    Rows<T> merged(tokens, w * c.merge * c.merge);
    const size_t nm = size_t(tokens) * merged.cols;
    mergePatches<<<blocks(nm), imageThreads, 0, cudaStreamPerThread>>>(hidden.p, merged.p, nm, w, cols, c.merge);
    auto proj = linear(merged, "proj_in", c.projection, false);
    const size_t np = size_t(tokens) * proj.cols;
    gelu<<<blocks(np), imageThreads, 0, cudaStreamPerThread>>>(proj.p, np);
    auto features = linear(proj, "proj_out", config.hidden, false);
    auto result = castRows<float>(features);
    check(cudaGetLastError());
    return result;
}
} // namespace

std::vector<ImageSpan> encodeImages(const ImageInputs &input, const Config &cfg, WeightMap &weights, int gpu,
                                    DataType precision, int promptLength) {
    const auto &c = cfg.vision;
    if (c.width <= 0 || c.layers <= 0 || c.heads <= 0 || c.width % c.heads || (c.width / c.heads) % 4 || c.grid <= 0 ||
        c.grid > 46340 || c.patch <= 0 || c.patch > INT_MAX / 3 / c.patch || c.merge <= 0 ||
        c.merge > INT_MAX / c.merge / c.width || c.patch > INT_MAX / c.merge || c.projection <= 0 || !(c.eps > 0))
        throw std::runtime_error("Unsupported Inkling vision geometry");
    const auto images = input.find("images"), offsets = input.find("image_offsets");
    if (images == input.end() || images->second.empty() || offsets == input.end() || offsets->second.size() != 1 ||
        !offsets->second[0] || offsets->second[0]->dataType != INT32 ||
        offsets->second[0]->Count(0) != images->second.size())
        throw std::runtime_error("Invalid Inkling image spans");
    auto &starts = *offsets->second[0];
    starts.ToDevice(CPU);
    const auto *pos = reinterpret_cast<const int *>(starts.cpuData);
    int end = 0;
    std::vector<ImageSpan> spans;
    for (size_t i = 0; i < images->second.size(); ++i) {
        auto *image = images->second[i];
        if (!image || image->dataType != INT8 || image->dims.size() != 3 || image->dims[0] <= 0 ||
            image->dims[1] <= 0 || (image->dims[2] != 3 && image->dims[2] != 4))
            throw std::runtime_error("Expected an Inkling RGB or RGBA image");
        auto [rows, cols] = c.tokenGrid(image->dims[0], image->dims[1]);
        int64_t nt = int64_t(rows) * cols;
        if (pos[i] < end || nt > promptLength - pos[i] || nt > (INT_MAX - 1) / c.merge / c.merge)
            throw std::runtime_error("Inkling image span exceeds prompt");
        end = pos[i] + int(nt);
        auto features = precision == BFLOAT16 ? encodeImage<__nv_bfloat16>(*image, cfg, weights, gpu, rows, cols)
                                              : encodeImage<half>(*image, cfg, weights, gpu, rows, cols);
        spans.push_back({pos[i], std::move(features)});
    }
    return spans;
}
} // namespace fastllm::inkling2
