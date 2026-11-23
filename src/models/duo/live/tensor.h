#pragma once
#include <cuda_runtime_api.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fastllm::duo::native {
using Shape = std::vector<int64_t>;
enum class Type { Undefined, BF16, F16, F32, U8, I32, I64 };
constexpr Type kBFloat16 = Type::BF16, kHalf = Type::F16, kFloat = Type::F32, kByte = Type::U8,
               kInt = Type::I32, kLong = Type::I64;
constexpr int kCPU = -1;
struct Options {
    Type type = kFloat;
    int gpu = kCPU;
    Options() = default;
    Options(Type t, int d = kCPU) : type(t), gpu(d) {}
    Options dtype(Type t) const {
        return {t, gpu};
    }
};
void check(cudaError_t status);
int elementSize(Type type);
class DeviceGuard {
    int previous;

  public:
    explicit DeviceGuard(int gpu);
    ~DeviceGuard();
    DeviceGuard(const DeviceGuard &) = delete;
};
class Tensor {
    std::shared_ptr<void> owner;
    void *ptr = nullptr;
    Shape shape, stride;
    Options opts;
    int axis(int d) const;

  public:
    Tensor() = default;
    Tensor(std::shared_ptr<void> owner, void *ptr, Shape sizes, Options opts);
    const Shape &sizes() const {
        return shape;
    }
    const Shape &strides() const {
        return stride;
    }
    int64_t size(int d) const {
        return shape.at(axis(d));
    }
    int dim() const {
        return int(shape.size());
    }
    int device() const {
        return opts.gpu;
    }
    Type scalar_type() const {
        return opts.type;
    }
    Options options() const {
        return opts;
    }
    bool defined() const {
        return ptr != nullptr;
    }
    int64_t numel() const;
    size_t nbytes() const {
        return numel() * elementSize(opts.type);
    }
    void *data_ptr() const {
        return ptr;
    }
    template <typename T> T *data_ptr() const {
        return static_cast<T *>(ptr);
    }
    Tensor contiguous() const;
    Tensor clone() const;
    Tensor reshape(Shape target) const;
    Tensor narrow(int d, int64_t start, int64_t count) const;
    Tensor slice(int d, int64_t start, int64_t end = INT64_MAX, int64_t step = 1) const;
    Tensor operator[](int64_t index) const;
    Tensor unsqueeze(int d) const;
    Tensor transpose(int a, int b) const;
    Tensor permute(const Shape &axes) const;
    Tensor expand(Shape target) const;
    Tensor flatten(int first = 0) const;
    std::vector<Tensor> split(int64_t count, int d = 0) const;
    std::vector<Tensor> chunk(int count, int d) const;
    void copy_(const Tensor &source) const;
    void add_(const Tensor &source) const;
    Tensor to(int gpu, Type t) const;
    Tensor to(Type t) const {
        return to(device(), t);
    }
    Tensor to(int gpu) const {
        return to(gpu, opts.type);
    }
    Tensor cpu() const {
        return to(kCPU);
    }
    Tensor sin() const;
    Tensor cos() const;
    Tensor exp() const;
    Tensor reciprocal() const;
    Tensor clamp(float lo, float hi) const;
    Tensor clamp_min(float v) const;
    Tensor log10() const;
    Tensor round() const;
    Tensor max() const;
    Tensor argmax() const;
    Tensor index_select(int dim, const Tensor &ids) const;
    void index_copy_(int dim, const Tensor &ids, const Tensor &rows) const;
    bool allFinite() const;
};
Tensor empty(Shape shape, Options opts);
Tensor zeros(Shape shape, Options opts);
Tensor from_blob(void *data, Shape shape, Type type);
Tensor rowIndices(const std::vector<int> &ids, int device);
Tensor tensor(const std::vector<int64_t> &values);
Tensor cat(const std::vector<Tensor> &tensors, int dim = 0);
Tensor stack(const std::vector<Tensor> &tensors, int dim);
Tensor arange(double start, double end, Options options);
inline Tensor arange(double end, Options opts) {
    return arange(0, end, opts);
}
Tensor operator+(const Tensor &, const Tensor &);
Tensor operator-(const Tensor &, const Tensor &);
Tensor operator*(const Tensor &, const Tensor &);
Tensor operator*(const Tensor &, double);
inline Tensor operator*(double x, const Tensor &t) {
    return t * x;
}
Tensor operator+(const Tensor &, double);
Tensor operator-(const Tensor &, double);
Tensor operator-(double, const Tensor &);
Tensor operator/(const Tensor &, double);
Tensor operator-(const Tensor &);
Tensor pow(double base, const Tensor &);
Tensor maximum(const Tensor &, const Tensor &);
Tensor silu(const Tensor &);
Tensor gelu(const Tensor &);
Tensor linear(const Tensor &x, const Tensor &w, const Tensor &bias = {});
Tensor normalize(const Tensor &x, const Tensor &weight, const Tensor &bias, bool center, float eps,
                 bool roundBeforeWeight = false);
Tensor attention(const Tensor &q, const Tensor &k, const Tensor &v, bool causal = false);
Tensor conv2dStride2(const Tensor &x, const Tensor &w, const Tensor &bias);
Tensor melPower(const std::vector<float> &pcm, int device);
void quantizeScales(Tensor &scales, const Tensor &map);
} // namespace fastllm::duo::native
