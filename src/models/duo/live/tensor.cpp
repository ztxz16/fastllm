#include "tensor.h"

namespace fastllm::duo::native {
namespace {
int64_t elements(const Shape &s) {
    int64_t n = 1;
    for (auto d : s) {
        if (d < 0 || (d && n > INT64_MAX / d))
            throw std::invalid_argument("invalid Duo tensor dimensions");
        n *= d;
    }
    return n;
}
Shape strides(const Shape &s) {
    Shape out(s.size());
    int64_t n = 1;
    for (int i = int(s.size()) - 1; i >= 0; --i) {
        out[i] = n;
        n *= s[i];
    }
    return out;
}
bool compact(const Tensor &t) {
    int64_t n = 1;
    for (int d = t.dim() - 1; d >= 0; --d) {
        if (t.size(d) > 1 && t.strides()[d] != n)
            return false;
        n *= t.size(d);
    }
    return true;
}
int64_t offset(const Tensor &t, int64_t i) {
    int64_t pos = 0;
    for (int d = t.dim() - 1; d >= 0; --d) {
        pos += (i % t.size(d)) * t.strides()[d];
        i /= t.size(d);
    }
    return pos;
}
} // namespace
void check(cudaError_t s) {
    if (s != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(s));
}
DeviceGuard::DeviceGuard(int gpu) {
    check(cudaGetDevice(&previous));
    if (gpu < 0)
        throw std::invalid_argument("expected CUDA device");
    if (gpu != previous)
        check(cudaSetDevice(gpu));
}
DeviceGuard::~DeviceGuard() {
    cudaSetDevice(previous);
}
int elementSize(Type t) {
    switch (t) {
    case Type::BF16:
    case Type::F16:
        return 2;
    case Type::F32:
    case Type::I32:
        return 4;
    case Type::I64:
        return 8;
    case Type::U8:
        return 1;
    default:
        throw std::invalid_argument("unknown tensor dtype");
    }
}
Tensor::Tensor(std::shared_ptr<void> owner, void *ptr, Shape sizes, Options opts)
    : owner(std::move(owner)), ptr(ptr), shape(std::move(sizes)), stride(native::strides(shape)), opts(opts) {
}
int Tensor::axis(int d) const {
    if (d < 0)
        d += dim();
    if (d < 0 || d >= dim())
        throw std::out_of_range("invalid Duo tensor axis");
    return d;
}
int64_t Tensor::numel() const {
    return defined() ? elements(shape) : 0;
}
Tensor empty(Shape shape, Options opts) {
    auto count = elements(shape);
    int bytes = elementSize(opts.type);
    if (uint64_t(count) > SIZE_MAX / bytes)
        throw std::overflow_error("Duo tensor allocation overflow");
    size_t n = std::max<size_t>(1, size_t(count) * bytes);
    void *ptr;
    std::shared_ptr<void> storage;
    if (opts.gpu < 0) {
        ptr = ::operator new(n);
        storage = {ptr, [](void *p) { ::operator delete(p); }};
    } else {
        DeviceGuard guard(opts.gpu);
        check(cudaMallocAsync(&ptr, n, nullptr));
        storage = {ptr, [d = opts.gpu](void *p) {
                       int old;
                       cudaGetDevice(&old);
                       cudaSetDevice(d);
                       cudaFreeAsync(p, nullptr);
                       cudaSetDevice(old);
                   }};
    }
    return {std::move(storage), ptr, std::move(shape), opts};
}
Tensor zeros(Shape s, Options opts) {
    auto out = empty(s, opts);
    if (opts.gpu < 0)
        std::memset(out.data_ptr(), 0, out.nbytes());
    else {
        DeviceGuard guard(opts.gpu);
        check(cudaMemsetAsync(out.data_ptr(), 0, out.nbytes()));
    }
    return out;
}
Tensor from_blob(void *p, Shape s, Type t) {
    elements(s);
    return {{}, p, std::move(s), Options(t)};
}
Tensor rowIndices(const std::vector<int> &ids, int device) {
    return tensor(std::vector<int64_t>(ids.begin(), ids.end())).to(device);
}
Tensor tensor(const std::vector<int64_t> &values) {
    auto host = empty({int64_t(values.size())}, kLong);
    if (!values.empty())
        std::memcpy(host.data_ptr(), values.data(), host.nbytes());
    return host;
}
Tensor Tensor::reshape(Shape s) const {
    if (!compact(*this))
        return contiguous().reshape(std::move(s));
    auto it = std::find(s.begin(), s.end(), -1);
    if (it != s.end()) {
        *it = 1;
        int64_t other = elements(s);
        if (!other || numel() % other)
            throw std::invalid_argument("invalid inferred tensor shape");
        *it = numel() / other;
    }
    if (elements(s) != numel())
        throw std::invalid_argument("Duo reshape changes element count");
    return {owner, ptr, std::move(s), opts};
}
Tensor Tensor::slice(int d, int64_t start, int64_t end, int64_t step) const {
    d = axis(d);
    auto n = size(d);
    if (step <= 0)
        throw std::invalid_argument("invalid slice stride");
    start = std::clamp(start < 0 ? n + start : start, int64_t(0), n);
    end = std::clamp(end < 0 ? n + end : end, start, n);
    Tensor t = *this;
    t.ptr = static_cast<char *>(ptr) + start * stride[d] * elementSize(opts.type);
    t.shape[d] = (end - start + step - 1) / step;
    t.stride[d] *= step;
    return t;
}
Tensor Tensor::narrow(int d, int64_t start, int64_t n) const {
    if (start < 0 || n < 0 || start > size(d) - n)
        throw std::out_of_range("invalid Duo tensor slice");
    return slice(d, start, start + n);
}
Tensor Tensor::operator[](int64_t i) const {
    Tensor t = narrow(0, i, 1);
    t.shape.erase(t.shape.begin());
    t.stride.erase(t.stride.begin());
    return t;
}
Tensor Tensor::unsqueeze(int d) const {
    if (d < 0)
        d += dim() + 1;
    if (d < 0 || d > dim())
        throw std::out_of_range("invalid unsqueeze axis");
    Tensor t = *this;
    t.shape.insert(t.shape.begin() + d, 1);
    t.stride.insert(t.stride.begin() + d, d == dim() ? 1 : stride[d] * shape[d]);
    return t;
}
Tensor Tensor::permute(const Shape &axes) const {
    if (int(axes.size()) != dim())
        throw std::invalid_argument("invalid permutation rank");
    Tensor out = *this;
    std::vector<bool> used(dim());
    for (int i = 0; i < dim(); ++i) {
        int d = axis(axes[i]);
        if (used[d])
            throw std::invalid_argument("duplicate axis");
        used[d] = true;
        out.shape[i] = shape[d];
        out.stride[i] = stride[d];
    }
    return out;
}
Tensor Tensor::transpose(int a, int b) const {
    Tensor out = *this;
    a = axis(a);
    b = axis(b);
    std::swap(out.shape[a], out.shape[b]);
    std::swap(out.stride[a], out.stride[b]);
    return out;
}
Tensor Tensor::expand(Shape s) const {
    if (s.size() < shape.size())
        throw std::invalid_argument("invalid broadcast rank");
    Tensor t = *this;
    while (t.shape.size() < s.size())
        t = t.unsqueeze(0);
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == -1 || s[i] == t.shape[i])
            continue;
        if (s[i] < 0 || t.shape[i] != 1)
            throw std::invalid_argument("invalid broadcast dimension");
        t.shape[i] = s[i];
        t.stride[i] = 0;
    }
    return t;
}
Tensor Tensor::flatten(int first) const {
    first = axis(first);
    Shape s(shape.begin(), shape.begin() + first);
    s.push_back(-1);
    return reshape(s);
}
std::vector<Tensor> Tensor::split(int64_t n, int d) const {
    if (n <= 0)
        throw std::invalid_argument("invalid split size");
    std::vector<Tensor> out;
    for (int64_t i = 0; i < size(d); i += n)
        out.push_back(narrow(d, i, std::min(n, size(d) - i)));
    return out;
}
std::vector<Tensor> Tensor::chunk(int n, int d) const {
    return split((size(d) + n - 1) / n, d);
}
Tensor Tensor::contiguous() const {
    return compact(*this) ? *this : clone();
}
Tensor Tensor::clone() const {
    auto out = empty(shape, opts);
    out.copy_(*this);
    return out;
}
void copyOnDevice(const Tensor &out, const Tensor &src);
void Tensor::copy_(const Tensor &src) const {
    if (shape != src.shape)
        throw std::invalid_argument("Duo copy shape mismatch");
    if (!numel())
        return;
    if (opts.gpu != src.device()) {
        if (opts.type != src.opts.type || !compact(*this) || !compact(src)) {
            // Move a contiguous buffer first; cast on the CUDA side.
            auto a = src.device() < 0 ? src.contiguous() : src.to(opts.type).contiguous();
            auto tmp = empty(shape, Options(a.scalar_type(), device()));
            tmp.copy_(a);
            copy_(tmp);
            return;
        }
        DeviceGuard guard(std::max(device(), src.device()));
        if (device() >= 0 && src.device() >= 0) {
            {
                DeviceGuard from(src.device());
                check(cudaStreamSynchronize(nullptr));
            }
            DeviceGuard dest(device());
            check(cudaMemcpyPeerAsync(ptr, device(), src.ptr, src.device(), nbytes()));
            check(cudaStreamSynchronize(nullptr));
        } else {
            check(cudaMemcpyAsync(ptr, src.ptr, nbytes(),
                                  device() < 0 ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice));
            check(cudaStreamSynchronize(nullptr));
        }
    } else if (device() >= 0) {
        DeviceGuard guard(device());
        if (compact(*this) && compact(src) && opts.type == src.opts.type)
            check(cudaMemcpyAsync(ptr, src.ptr, nbytes(), cudaMemcpyDeviceToDevice));
        else
            copyOnDevice(*this, src);
    } else if (opts.type != src.opts.type) {
        int gpu;
        check(cudaGetDevice(&gpu));
        copy_(src.to(gpu).to(opts.type).cpu());
    } else if (compact(*this) && compact(src))
        std::memcpy(ptr, src.ptr, nbytes());
    else {
        int bytes = elementSize(opts.type);
        for (int64_t i = 0; i < numel(); ++i)
            std::memcpy(static_cast<char *>(ptr) + offset(*this, i) * bytes,
                        static_cast<char *>(src.ptr) + offset(src, i) * bytes, bytes);
    }
}
Tensor Tensor::to(int gpu, Type t) const {
    if (device() == gpu && opts.type == t)
        return *this;
    auto out = empty(shape, {t, gpu});
    out.copy_(*this);
    return out;
}
Tensor cat(const std::vector<Tensor> &items, int d) {
    if (items.empty())
        throw std::invalid_argument("empty Duo concatenation");
    if (items.size() == 1)
        return items.front();
    Shape s = items.front().sizes();
    if (d < 0)
        d += int(s.size());
    s.at(d) = 0;
    for (auto &t : items)
        s[d] += t.size(d);
    auto out = empty(s, items.front().options());
    int64_t pos = 0;
    for (auto &t : items) {
        out.narrow(d, pos, t.size(d)).copy_(t);
        pos += t.size(d);
    }
    return out;
}
Tensor stack(const std::vector<Tensor> &items, int d) {
    std::vector<Tensor> slices;
    for (auto &t : items)
        slices.push_back(t.unsqueeze(d));
    return cat(slices, d);
}
void Tensor::add_(const Tensor &t) const {
    copy_(*this + t);
}
} // namespace fastllm::duo::native
