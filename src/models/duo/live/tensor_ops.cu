#include "tensor.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <map>

namespace fastllm::nomas {
cudaError_t fusedAttention(const void *, const void *, const void *, void *, int, int, int, int, int, bool,
                           bool, int, int, int, float, bool, cudaStream_t, int, int, int);
}
namespace fastllm::duo::native {
namespace {
constexpr int threads = 256;
int blocks(int64_t n) {
    return int(std::min<int64_t>(65535, (n + threads - 1) / threads));
}
void launched() {
    check(cudaGetLastError());
}
void require(bool ok, const char *message) {
    if (!ok)
        throw std::invalid_argument(message);
}
struct View {
    void *ptr;
    Type type;
    int rank = 0;
    int64_t dims[8], strides[8];
    explicit View(const Tensor &t) : ptr(t.data_ptr()), type(t.scalar_type()) {
        require(t.dim() <= 8, "Duo tensor rank exceeds eight");
        for (int i = 0; i < t.dim(); ++i) {
            auto n = t.size(i), s = t.strides()[i];
            if (n == 1)
                continue;
            if (rank && strides[rank - 1] == n * s) {
                dims[rank - 1] *= n;
                strides[rank - 1] = s;
            } else {
                dims[rank] = n;
                strides[rank++] = s;
            }
        }
    }
    __device__ int64_t offset(int64_t i) const {
        if (rank == 0)
            return 0;
        if (rank == 1)
            return i * strides[0];
        int64_t pos = 0;
        for (int d = rank - 1; d > 0; --d) {
            pos += (i % dims[d]) * strides[d];
            i /= dims[d];
        }
        return pos + i * strides[0];
    }
    __device__ float get(int64_t i) const {
        i = offset(i);
        switch (type) {
        case Type::F32:
            return ((float *)ptr)[i];
        case Type::BF16:
            return __bfloat162float(((__nv_bfloat16 *)ptr)[i]);
        case Type::F16:
            return __half2float(((__half *)ptr)[i]);
        case Type::I64:
            return ((int64_t *)ptr)[i];
        case Type::I32:
            return ((int *)ptr)[i];
        case Type::U8:
            return ((uint8_t *)ptr)[i];
        default:
            return NAN;
        }
    }
    __device__ void put(int64_t i, float x) const {
        i = offset(i);
        switch (type) {
        case Type::F32:
            ((float *)ptr)[i] = x;
            break;
        case Type::BF16:
            ((__nv_bfloat16 *)ptr)[i] = __float2bfloat16(x);
            break;
        case Type::F16:
            ((__half *)ptr)[i] = __float2half(x);
            break;
        case Type::I64:
            ((int64_t *)ptr)[i] = int64_t(x);
            break;
        case Type::I32:
            ((int *)ptr)[i] = int(x);
            break;
        case Type::U8:
            ((uint8_t *)ptr)[i] = uint8_t(x);
            break;
        default:
            break;
        }
    }
};
template <typename T> __global__ void copyStrided(View dst, View src, int64_t count) {
    auto *target = static_cast<T *>(dst.ptr);
    auto *source = static_cast<const T *>(src.ptr);
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x)
        target[dst.offset(i)] = source[src.offset(i)];
}
enum class Op {
    Copy,
    Add,
    Sub,
    Mul,
    Scale,
    Shift,
    ReverseSub,
    Maximum,
    Sin,
    Cos,
    Exp,
    Reciprocal,
    Clamp,
    Log10,
    Round,
    Pow,
    Silu,
    Gelu,
    Range
};
__global__ void elementwise(View out, View a, View b, int64_t n, Op op, float p, float q) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float x = op == Op::Range ? 0 : a.get(i), v = x;
        switch (op) {
        case Op::Copy:
            break;
        case Op::Add:
            v = x + b.get(i);
            break;
        case Op::Sub:
            v = x - b.get(i);
            break;
        case Op::Mul:
            v = x * b.get(i);
            break;
        case Op::Maximum:
            v = fmaxf(x, b.get(i));
            break;
        case Op::Scale:
            v = x * p;
            break;
        case Op::Shift:
            v = x + p;
            break;
        case Op::ReverseSub:
            v = p - x;
            break;
        case Op::Sin:
            v = sinf(x);
            break;
        case Op::Cos:
            v = cosf(x);
            break;
        case Op::Exp:
            v = expf(x);
            break;
        case Op::Reciprocal:
            v = 1.f / x;
            break;
        case Op::Clamp:
            v = fminf(q, fmaxf(p, x));
            break;
        case Op::Log10:
            v = log10f(x);
            break;
        case Op::Round:
            v = nearbyintf(x);
            break;
        case Op::Pow:
            v = powf(p, x);
            break;
        case Op::Silu:
            v = x / (1.f + expf(-x));
            break;
        case Op::Gelu:
            v = .5f * x * (1.f + erff(x * .7071067811865475f));
            break;
        case Op::Range:
            v = p + float(i);
            break;
        }
        out.put(i, v);
    }
}
void run(const Tensor &out, const Tensor &a, const Tensor &b, Op op, float p = 0, float q = 0) {
    if (!out.numel())
        return;
    DeviceGuard guard(out.device());
    elementwise<<<blocks(out.numel()), threads>>>(View(out), View(a), View(b), out.numel(), op, p, q);
    launched();
}
Tensor unary(const Tensor &a, Op op, float p = 0, float q = 0) {
    auto y = empty(a.sizes(), a.options());
    run(y, a, a, op, p, q);
    return y;
}
Tensor binary(const Tensor &a, const Tensor &b, Op op) {
    require(a.device() == b.device(), "Duo elementwise inputs on different devices");
    Shape shape(std::max(a.dim(), b.dim()), 1);
    for (auto *t : {&a, &b})
        for (int i = 0; i < t->dim(); ++i) {
            auto &d = shape[shape.size() - t->dim() + i];
            auto n = t->size(i);
            require(d == 1 || n == 1 || n == d, "Duo elementwise shape mismatch");
            if (n != 1)
                d = n;
        }
    Type type = a.scalar_type() == kFloat || b.scalar_type() == kFloat ? kFloat : a.scalar_type();
    auto y = empty(shape, {type, a.device()});
    run(y, a.expand(shape), b.expand(shape), op);
    return y;
}
__device__ float sumBlock(float val, float *smem) {
    for (int d = 16; d; d /= 2)
        val += __shfl_down_sync(0xffffffff, val, d);
    if (threadIdx.x % 32 == 0)
        smem[threadIdx.x / 32] = val;
    __syncthreads();
    val = threadIdx.x < blockDim.x / 32 ? smem[threadIdx.x] : 0;
    if (threadIdx.x < 32) {
        for (int d = 16; d; d /= 2)
            val += __shfl_down_sync(0xffffffff, val, d);
    }
    if (threadIdx.x == 0)
        smem[0] = val;
    __syncthreads();
    val = smem[0];
    __syncthreads();
    return val;
}
__global__ void normRows(View out, View x, View w, View b, int rows, int cols, bool center, float eps,
                         bool roundFirst) {
    __shared__ float smem[32];
    for (int r = blockIdx.x; r < rows; r += gridDim.x) {
        int64_t begin = int64_t(r) * cols;
        float mean = 0, sum = 0;
        if (center) {
            for (int j = threadIdx.x; j < cols; j += blockDim.x)
                mean += x.get(begin + j);
            mean = sumBlock(mean, smem) / cols;
        }
        for (int j = threadIdx.x; j < cols; j += blockDim.x) {
            float v = x.get(begin + j) - mean;
            sum += v * v;
        }
        float scale = rsqrtf(sumBlock(sum, smem) / cols + eps);
        for (int j = threadIdx.x; j < cols; j += blockDim.x) {
            float v = (x.get(begin + j) - mean) * scale;
            if (roundFirst && x.type == Type::BF16)
                v = __bfloat162float(__float2bfloat16(v));
            if (w.ptr)
                v *= w.get(j);
            if (b.ptr)
                v += b.get(j);
            out.put(begin + j, v);
        }
    }
}
__global__ void checkFinite(View x, int64_t n, int *invalid) {
    bool bad = false;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        bad |= !isfinite(x.get(i));
    if (__any_sync(0xffffffff, bad) && threadIdx.x % 32 == 0)
        atomicExch(invalid, 1);
}
enum class Reduction { Max, Argmax };
__global__ void reduceRows(View out, View x, int64_t rows, int64_t width, Reduction op) {
    __shared__ float values[threads];
    __shared__ int indices[threads];
    int lane = threadIdx.x;
    for (int64_t r = blockIdx.x; r < rows; r += gridDim.x) {
        float best = -INFINITY;
        int idx = 0;
        for (int64_t j = lane; j < width; j += blockDim.x) {
            float v = x.get(r * width + j);
            if (v > best) {
                best = v;
                idx = int(j);
            }
        }
        values[lane] = best;
        indices[lane] = idx;
        __syncthreads();
        for (int d = blockDim.x / 2; d; d /= 2) {
            if (lane < d && (values[lane + d] > values[lane] ||
                             (values[lane + d] == values[lane] && indices[lane + d] < indices[lane]))) {
                values[lane] = values[lane + d];
                indices[lane] = indices[lane + d];
            }
            __syncthreads();
        }
        if (!lane)
            out.put(r, op == Reduction::Argmax ? float(indices[0]) : values[0]);
        __syncthreads();
    }
}
Tensor reduce(const Tensor &x, Reduction op) {
    bool rows = op == Reduction::Argmax;
    Shape s = rows ? x.sizes() : Shape{};
    if (rows)
        s.pop_back();
    auto y = empty(s, {rows ? kLong : kFloat, x.device()});
    DeviceGuard guard(x.device());
    reduceRows<<<std::min<int64_t>(65535, y.numel()), threads>>>(View(y), View(x), y.numel(),
                                                                 x.numel() / y.numel(), op);
    launched();
    return y;
}
__global__ void indexedRows(View out, View src, const int64_t *ids, int count, int cols, int limit,
                            bool scatter, int *invalid) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < int64_t(count) * cols;
         i += gridDim.x * blockDim.x) {
        auto row = ids[i / cols];
        if (row < 0 || row >= limit) {
            atomicExch(invalid, 1);
            continue;
        }
        auto pos = row * cols + i % cols;
        out.put(scatter ? pos : i, src.get(scatter ? i : pos));
    }
}
void indexed(const Tensor &out, const Tensor &src, const Tensor &ids, bool scatter) {
    require(src.dim() == 2 && out.dim() == 2 && ids.scalar_type() == kLong && ids.dim() == 1 &&
                ids.device() == src.device(),
            "invalid Duo row lookup");
    DeviceGuard guard(src.device());
    auto indices = ids.contiguous();
    if (!ids.numel())
        return;
    auto invalid = zeros({1}, {kInt, src.device()});
    indexedRows<<<blocks(ids.numel() * src.size(1)), threads>>>(
        View(out), View(src), indices.data_ptr<int64_t>(), ids.numel(), src.size(1),
        scatter ? out.size(0) : src.size(0), scatter, invalid.data_ptr<int>());
    launched();
    if (*invalid.cpu().data_ptr<int>())
        throw std::out_of_range("Duo row index exceeds embedding size");
}
struct Blas {
    cublasHandle_t h;
    Blas() {
        if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLAS initialization failed");
    }
    ~Blas() {
        cublasDestroy(h);
    }
};
cudaDataType_t cudaType(Type t) {
    if (t == kFloat)
        return CUDA_R_32F;
    if (t == kBFloat16)
        return CUDA_R_16BF;
    if (t == kHalf)
        return CUDA_R_16F;
    throw std::invalid_argument("invalid GEMM dtype");
}
} // namespace
void copyOnDevice(const Tensor &out, const Tensor &src) {
    if (out.scalar_type() != src.scalar_type()) {
        run(out, src, src, Op::Copy);
        return;
    }
    auto n = src.numel();
    View dst(out), input(src);
    switch (elementSize(src.scalar_type())) {
    case 1:
        copyStrided<uint8_t><<<blocks(n), threads>>>(dst, input, n);
        break;
    case 2:
        copyStrided<uint16_t><<<blocks(n), threads>>>(dst, input, n);
        break;
    case 4:
        copyStrided<uint32_t><<<blocks(n), threads>>>(dst, input, n);
        break;
    case 8:
        copyStrided<uint64_t><<<blocks(n), threads>>>(dst, input, n);
        break;
    }
    launched();
}
Tensor operator+(const Tensor &a, const Tensor &b) {
    return binary(a, b, Op::Add);
}
Tensor operator-(const Tensor &a, const Tensor &b) {
    return binary(a, b, Op::Sub);
}
Tensor operator*(const Tensor &a, const Tensor &b) {
    return binary(a, b, Op::Mul);
}
Tensor maximum(const Tensor &a, const Tensor &b) {
    return binary(a, b, Op::Maximum);
}
Tensor operator*(const Tensor &a, double b) {
    return unary(a, Op::Scale, b);
}
Tensor operator+(const Tensor &a, double b) {
    return unary(a, Op::Shift, b);
}
Tensor operator-(const Tensor &a, double b) {
    return a + (-b);
}
Tensor operator-(double a, const Tensor &b) {
    return unary(b, Op::ReverseSub, a);
}
Tensor operator/(const Tensor &a, double b) {
    return a * (1. / b);
}
Tensor operator-(const Tensor &a) {
    return a * (-1.);
}
Tensor Tensor::sin() const {
    return unary(*this, Op::Sin);
}
Tensor Tensor::cos() const {
    return unary(*this, Op::Cos);
}
Tensor Tensor::exp() const {
    return unary(*this, Op::Exp);
}
Tensor Tensor::reciprocal() const {
    return unary(*this, Op::Reciprocal);
}
Tensor Tensor::clamp(float a, float b) const {
    return unary(*this, Op::Clamp, a, b);
}
Tensor Tensor::clamp_min(float v) const {
    return clamp(v, INFINITY);
}
Tensor Tensor::log10() const {
    return unary(*this, Op::Log10);
}
Tensor Tensor::round() const {
    return unary(*this, Op::Round);
}
Tensor pow(double base, const Tensor &x) {
    return unary(x, Op::Pow, base);
}
Tensor silu(const Tensor &x) {
    return unary(x, Op::Silu);
}
Tensor gelu(const Tensor &x) {
    return unary(x, Op::Gelu);
}
Tensor arange(double start, double end, Options opts) {
    auto out = empty({int64_t(std::ceil(end - start))}, opts);
    run(out, out, out, Op::Range, start);
    return out;
}
Tensor normalize(const Tensor &x, const Tensor &weight, const Tensor &bias, bool center, float eps,
                 bool roundBeforeWeight) {
    int cols = x.size(-1), rows = x.numel() / cols;
    auto out = empty(x.sizes(), x.options());
    DeviceGuard guard(x.device());
    normRows<<<std::min(rows, 65535), threads>>>(View(out), View(x), View(weight), View(bias), rows, cols,
                                                 center, eps, roundBeforeWeight);
    launched();
    return out;
}
Tensor Tensor::max() const {
    return reduce(*this, Reduction::Max);
}
Tensor Tensor::argmax() const {
    return reduce(*this, Reduction::Argmax);
}
bool Tensor::allFinite() const {
    if (!numel())
        return true;
    DeviceGuard guard(device());
    auto invalid = zeros({1}, {kInt, device()});
    checkFinite<<<blocks(numel()), threads>>>(View(*this), numel(), invalid.data_ptr<int>());
    launched();
    return !*invalid.cpu().data_ptr<int>();
}
Tensor Tensor::index_select(int d, const Tensor &ids) const {
    require(d == 0, "expected row selection");
    auto out = empty({ids.numel(), size(1)}, opts);
    indexed(out, *this, ids, false);
    return out;
}
void Tensor::index_copy_(int d, const Tensor &ids, const Tensor &rows) const {
    require(d == 0 && size(1) == rows.size(1) && ids.numel() == rows.size(0), "invalid embedding update");
    indexed(*this, rows, ids, true);
}
Tensor linear(const Tensor &input, const Tensor &weights, const Tensor &bias) {
    require(weights.dim() == 2 && input.size(-1) == weights.size(1) && input.device() == weights.device() &&
                input.scalar_type() == weights.scalar_type(),
            "Duo projection geometry/dtype mismatch");
    auto x = input.contiguous(), w = weights.contiguous();
    Shape s = x.sizes();
    int k = s.back(), n = w.size(0), m = x.numel() / k;
    s.back() = n;
    DeviceGuard guard(x.device());
    auto y = empty(s, x.options());
    thread_local std::map<int, Blas> handles;
    float one = 1, zero = 0;
    auto type = cudaType(x.scalar_type());
    auto status = cublasGemmEx(handles[x.device()].h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &one, w.data_ptr(),
                               type, k, x.data_ptr(), type, k, &zero, y.data_ptr(), type, n,
                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Duo projection failed: " + std::to_string(status));
    return bias.defined() ? y + bias : y;
}
Tensor attention(const Tensor &q, const Tensor &k, const Tensor &v, bool causal) {
    require(q.dim() == 4 && k.dim() == 4 && k.sizes() == v.sizes() && q.size(0) == k.size(0) &&
                q.size(3) == k.size(3) && k.size(1) > 0 && q.size(1) % k.size(1) == 0 &&
                q.device() == k.device() && q.device() == v.device() && q.scalar_type() == kBFloat16 &&
                k.scalar_type() == kBFloat16 && v.scalar_type() == kBFloat16,
            "invalid Duo attention geometry");
    DeviceGuard guard(q.device());
    auto Q = q.transpose(1, 2).contiguous(), K = k.transpose(1, 2).contiguous(),
         V = v.transpose(1, 2).contiguous();
    auto out = empty(Q.sizes(), Q.options());
    for (int i = 0; i < Q.size(0); ++i)
        check(nomas::fusedAttention(Q[i].data_ptr(), K[i].data_ptr(), V[i].data_ptr(), out[i].data_ptr(),
                                    q.size(2), k.size(2), q.size(1), k.size(1), q.size(3), false, causal, 0,
                                    0, 0, 1.f / std::sqrt(float(q.size(3))), true, nullptr, 0, 0, 1));
    return out.transpose(1, 2);
}
} // namespace fastllm::duo::native
