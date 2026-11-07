#include "exl3.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_fp16.h>
#include <map>
#include <memory>
#include <tuple>
#include <type_traits>
#include <mutex>

#ifdef FASTLLM_USE_MAYA_EXL3
bool FastllmCudaExl3MayaGrouped(const fastllm::Data &, const fastllm::Data &, fastllm::Data &);
bool FastllmCudaExl3MayaLinear(const fastllm::Data &, fastllm::Data &, const fastllm::Data &, fastllm::Data &);
bool FastllmCudaExl3MayaMoe(const fastllm::Data &, fastllm::Data &, const fastllm::Data &, const fastllm::Data &, fastllm::Data **, int, float);
bool FastllmCudaExl3MayaCache(const fastllm::Data &, const uint8_t *, const int32_t *, const uint64_t *, size_t, size_t, int, int, int, const float *, int, float, float *, int);
void FastllmCudaReleaseExl3Maya(const fastllm::Data *);
#endif

#include "fastllm-exl3-shards.cuh"
namespace {
constexpr float kNormalization = 0.08838834764831845f;
__device__ float ReadActivation(const void *data, int type, size_t at) {
    if (type == fastllm::FLOAT32) return static_cast<const float *>(data)[at];
    const unsigned short bits = static_cast<const unsigned short *>(data)[at];
    return type == fastllm::FLOAT16 ? __half2float(__ushort_as_half(bits)) : __uint_as_float(unsigned(bits) << 16);
}
__device__ void WriteActivation(void *data, int type, size_t at, float value) {
    if (type == fastllm::FLOAT32) { static_cast<float *>(data)[at] = value; return; }
    unsigned short bits;
    if (type == fastllm::FLOAT16) bits = __half_as_ushort(__float2half_rn(value));
    else {
        unsigned word = __float_as_uint(value);
        word += 0x7fff + ((word >> 16) & 1);
        bits = word >> 16;
    }
    static_cast<unsigned short *>(data)[at] = bits;
}
__device__ float Rotate(float value, float *shared) {
    const int lane = threadIdx.x;
    // Five butterfly stages stay within a warp on every supported SM.
    for (int bit = 1; bit < 32; bit <<= 1) {
        float peer = __shfl_xor_sync(0xffffffffu, value, bit);
        value = (lane & bit) ? peer - value : value + peer;
    }
    for (int bit = 32; bit < 128; bit <<= 1) {
        shared[lane] = value;
        __syncthreads();
        float peer = shared[lane ^ bit];
        __syncthreads();
        value = (lane & bit) ? peer - value : value + peer;
    }
    return value * kNormalization;
}
template<int twiceBits>
__device__ float DecodeWeight(const unsigned *trellis, int inputs, int outputs, int r, int c) {
    const int lr = r & 15, lc = c & 15;
    const int step = (lr & 1) | ((lr >> 3) << 1) | ((lc >> 3) << 2) |
                     (((lr >> 1) & 3) << 3) | ((lc & 7) << 5);
    const int ring = 128 * twiceBits;
    const int begin = (((step + 1) * twiceBits / 2) + ring - 16) % ring;
    const unsigned *tile = trellis + (size_t(r / 16) * (outputs / 16) + c / 16) * (4 * twiceBits);
    const int index = begin / 32;
    const unsigned long long pair = (static_cast<unsigned long long>(tile[index]) << 32) |
                                     tile[(index + 1) % (4 * twiceBits)];
    const unsigned state = (pair >> (48 - (begin & 31))) & 65535;
    const unsigned mixed = state * 0x83dcd12du;
    const unsigned bytes = (mixed & 255) + ((mixed >> 8) & 255) + ((mixed >> 16) & 255) + (mixed >> 24);
    const float code = fmaf(1024.0f + bytes, __half2float(__ushort_as_half(0x1eee)),
                           __half2float(__ushort_as_half(0xc931)));
    return __half2float(__float2half_rn(code));
}
template<class T>
__global__ void PrepareInput(const void *input, int type, const unsigned short *scale, T *rotated, int inputs, int rowStride) {
    __shared__ float shared[128];
    const int column = blockIdx.x * 128 + threadIdx.x;
    const size_t at = size_t(blockIdx.y) * inputs + column;
    float value = ReadActivation(input, type, size_t(blockIdx.y) * rowStride + column) * __half2float(__ushort_as_half(scale[column]));
    value = Rotate(value, shared);
    const half rounded = __float2half_rn(value);
    if constexpr (std::is_same<T, half>::value) rotated[at] = rounded;
    else rotated[at] = __half2float(rounded);
}
// The small-batch path never expands weights. Splitting the input supplies
// enough independent blocks even for narrow expert projections. Each output
// reduces its pieces in a fixed order; no floating-point atomics are required.
template<int twiceBits>
__global__ void DirectProduct(const unsigned *trellis, const float *input, float *partial,
                              int inputs, int outputs, int splits) {
    const int c = blockIdx.x * 128 + threadIdx.x;
    const int split = blockIdx.y, row = blockIdx.z;
    const int first = split * 256, end = min(first + 256, inputs);
    float sum = 0;
    for (int r = first; r < end; ++r)
        sum = fmaf(input[size_t(row) * inputs + r], DecodeWeight<twiceBits>(trellis, inputs, outputs, r, c), sum);
    partial[(size_t(row) * splits + split) * outputs + c] = sum;
}
template<int twiceBits>
__global__ void ExpandInner(const unsigned *trellis, unsigned short *matrix,
                            int inputs, int outputs) {
    const size_t at = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= size_t(inputs) * outputs) return;
    const int c = at / inputs, r = at % inputs;
    matrix[at] = __half_as_ushort(__float2half_rn(DecodeWeight<twiceBits>(trellis, inputs, outputs, r, c)));
}
__global__ void FinishOutput(const float *partial, const unsigned short *scale, const float *bias,
                             void *output, int type, int outputs, int totalOutputs, int offset, int splits) {
    __shared__ float shared[128];
    const int c = blockIdx.x * 128 + threadIdx.x, row = blockIdx.y;
    float value = 0;
    for (int split = 0; split < splits; ++split)
        value += partial[(size_t(row) * splits + split) * outputs + c];
    value = Rotate(value, shared) * __half2float(__ushort_as_half(scale[c]));
    if (bias) value += bias[offset + c];
    WriteActivation(output, type, size_t(row) * totalOutputs + offset + c, value);
}
template<int twiceBits>
void Product(const unsigned *trellis, const float *input, float *partial, unsigned short *decoded,
             int k, int n, int rows, int splits, cudaStream_t stream) {
    if (rows <= 8) DirectProduct<twiceBits><<<dim3(n / 128, splits, rows), 128, 0, stream>>>(
        trellis, input, partial, k, n, splits);
    else ExpandInner<twiceBits><<<(uint64_t(k) * n + 255) / 256, 256, 0, stream>>>(trellis, decoded, k, n);
}
void LaunchProduct(int bits, const unsigned *trellis, const float *input, float *partial,
                   unsigned short *decoded, int k, int n, int rows, int splits, cudaStream_t stream) {
#define EXL3_CASE(B) case B: Product<B>(trellis, input, partial, decoded, k, n, rows, splits, stream); break
    switch (bits) {
        EXL3_CASE(2); EXL3_CASE(3); EXL3_CASE(4); EXL3_CASE(5); EXL3_CASE(6); EXL3_CASE(7);
        EXL3_CASE(8); EXL3_CASE(10); EXL3_CASE(12); EXL3_CASE(14); EXL3_CASE(16);
    }
#undef EXL3_CASE
}
struct Workspace {
    fastllm::Data input{fastllm::FLOAT32}, partial{fastllm::FLOAT32}, decoded{fastllm::FLOAT16}, halfInput{fastllm::FLOAT16};
    explicit Workspace(int device) {
        for (auto *value : {&input, &partial, &decoded, &halfInput}) {
            value->dataDevice = fastllm::DataDevice::CUDA;
            value->dataDeviceIds = {device};
        }
    }
};
}

static bool Exl3LinearStrided(const fastllm::Data &input, fastllm::Data &weight,
                              const fastllm::Data &bias, fastllm::Data &output,
                              int inputStride, int outputStride) {
    using namespace fastllm;
    if (weight.dataType != EXL3) return false;
    AssertInFastLLM(weight.cudaData && input.cudaData && output.cudaData &&
                    (input.dataType == FLOAT32 || input.dataType == FLOAT16 || input.dataType == BFLOAT16),
                    "EXL3: invalid CUDA linear inputs.");
    Exl3StorageBytes(weight);
    const int k = weight.dims[1], total = weight.dims[0], n = total / weight.blockM;
    const int rows = input.Count(0) / k;
    AssertInFastLLM(input.dims.back() == k && output.dims.back() == total,
                    "EXL3: CUDA linear dimension mismatch.");
    const int device = FastllmCudaGetDevice();
    // Small batches can be captured: keep scratch addresses stable per shape.
    // Eager prefill reuses one bounded workspace per device.
    static thread_local std::map<std::tuple<int, int, int, int>, std::unique_ptr<Workspace>> workspaces;
    const auto key = rows <= 8 ? std::make_tuple(device, k, n, rows) : std::make_tuple(device, 0, 0, 0);
    auto &owned = workspaces[key];
    if (!owned) owned.reset(new Workspace(device));
    Workspace &work = *owned;
    if (rows <= 8) { work.input.Resize({rows, k}); work.input.Allocate(false); }
    else { work.halfInput.Resize({rows, k}); work.halfInput.Allocate(false); }
    const int splits = rows <= 8 ? (k + 255) / 256 : 1;
    work.partial.Resize({rows * splits, n}); work.partial.Allocate(false);
    const size_t trellisBytes = uint64_t(n) * k * weight.blockK / 16;
    const size_t partBytes = Exl3PartBytes(n, k, weight.blockK);
    const float *biasData = bias.dims.empty() ? nullptr : static_cast<const float *>(bias.cudaData);
    cudaStream_t stream = cudaStreamPerThread;
    for (int part = 0; part < weight.blockM; ++part) {
        const uint8_t *base = static_cast<const uint8_t *>(weight.cudaData) + part * partBytes;
        const auto *trellis = reinterpret_cast<const unsigned *>(base);
        const auto *inScale = reinterpret_cast<const unsigned short *>(base + trellisBytes);
        const auto *outScale = inScale + k;
        // GEMM consumes FP16 directly. Writing its rounded rotation here
        // avoids a temporary FP32 activation and a second conversion pass.
        if (rows <= 8) PrepareInput<float><<<dim3(k / 128, rows), 128, 0, stream>>>(input.cudaData, input.dataType, inScale,
            static_cast<float *>(work.input.cudaData), k, inputStride);
        else PrepareInput<half><<<dim3(k / 128, rows), 128, 0, stream>>>(input.cudaData, input.dataType, inScale,
            static_cast<half *>(work.halfInput.cudaData), k, inputStride);
        if (rows > 8) { work.decoded.Resize({n, k}); work.decoded.Allocate(false); }
        LaunchProduct(weight.blockK, trellis, static_cast<const float *>(work.input.cudaData),
            static_cast<float *>(work.partial.cudaData), static_cast<unsigned short *>(work.decoded.cudaData),
            k, n, rows, splits, stream);
        if (rows > 8) {
            const float alpha = 1.0f, beta = 0.0f;
            // Keep FP32 accumulation through the output rotation. A FP16
            // intermediate here would add rounding not present in the format.
            auto handle = getFastllmCublasHandle();
            auto status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, rows, k,
                &alpha, work.decoded.cudaData, CUDA_R_16F, k,
                work.halfInput.cudaData, CUDA_R_16F, k, &beta,
                work.partial.cudaData, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
            AssertInFastLLM(status == CUBLAS_STATUS_SUCCESS, "EXL3: half GEMM failed.");
        }
        FinishOutput<<<dim3(n / 128, rows), 128, 0, stream>>>(static_cast<const float *>(work.partial.cudaData),
            outScale, biasData, output.cudaData, output.dataType, n, outputStride, part * n, splits);
    }
    AssertInFastLLM(cudaGetLastError() == cudaSuccess, "EXL3: CUDA kernel launch failed.");
    return true;
}

bool FastllmCudaExl3Linear(const fastllm::Data &input, fastllm::Data &weight,
                          const fastllm::Data &bias, fastllm::Data &output) {
#ifdef FASTLLM_USE_MAYA_EXL3
    if (FastllmCudaExl3MayaLinear(input, weight, bias, output)) return true;
#endif
    if (weight.dataType != fastllm::EXL3 || weight.dims.size() != 2) return false;
    return Exl3LinearStrided(input, weight, bias, output, weight.dims[1], weight.dims[0]);
}

namespace {
struct Exl3Projection {
    const unsigned *trellis;
    const unsigned short *inputScale, *outputScale;
    int bits;
};
struct ExpertDirectory {
    std::vector<Exl3Projection> host;
    fastllm::Data device{fastllm::INT8};
};
using DirectoryKey = std::pair<const fastllm::Data *, int>;
std::mutex &DirectoryMutex() { static auto *mutex = new std::mutex; return *mutex; }
std::map<DirectoryKey, std::shared_ptr<ExpertDirectory>> &Directories() {
    static auto *map = new std::map<DirectoryKey, std::shared_ptr<ExpertDirectory>>;
    return *map;
}
__device__ float RoundBfloat(float value) {
    unsigned word = __float_as_uint(value);
    word += 0x7fff + ((word >> 16) & 1);
    return __uint_as_float(word & 0xffff0000u);
}
__global__ void MoePrepare(const void *input, int type, const int *routes, const Exl3Projection *weights,
                            float *rotated, int k, int topk) {
    __shared__ float shared[128];
    const int c = blockIdx.x * 128 + threadIdx.x, task = blockIdx.y, row = blockIdx.z;
    const int part = task & 1, route = task / 2;
    if (routes[row*topk+route] < 0) return;
    const Exl3Projection projection = weights[routes[row * topk + route] * 3 + part];
    float value = ReadActivation(input, type, size_t(row) * k + c) *
                  __half2float(__ushort_as_half(projection.inputScale[c]));
    rotated[((size_t(row) * topk + route) * 2 + part) * k + c] =
        __half2float(__float2half_rn(Rotate(value, shared)));
}
template<int bits>
__device__ float Segment(const Exl3Projection &weight, const float *input, int k, int n, int c, int split) {
    float sum = 0;
    for (int r = split * 256; r < min(k, split * 256 + 256); ++r)
        sum = fmaf(input[r], DecodeWeight<bits>(weight.trellis, k, n, r, c), sum);
    return sum;
}
__global__ void MoeProduct(const float *input, const int *routes, const Exl3Projection *weights,
                            float *partial, int k, int n, int topk, int splits, bool down) {
    const int c = blockIdx.x * 128 + threadIdx.x, split = blockIdx.y;
    const int projections = down ? 1 : 2, task = blockIdx.z, routeRow = task / projections;
    const int part = down ? 2 : task % 2;
    if (routes[routeRow] < 0) return;
    const Exl3Projection weight = weights[routes[routeRow] * 3 + part];
    const float *x = input + size_t(task) * k;
    float sum = 0;
#define EXL3_SEGMENT(B) case B: sum = Segment<B>(weight, x, k, n, c, split); break
    switch (weight.bits) {
        EXL3_SEGMENT(2); EXL3_SEGMENT(3); EXL3_SEGMENT(4); EXL3_SEGMENT(5); EXL3_SEGMENT(6); EXL3_SEGMENT(7);
        EXL3_SEGMENT(8); EXL3_SEGMENT(10); EXL3_SEGMENT(12); EXL3_SEGMENT(14); EXL3_SEGMENT(16);
    }
#undef EXL3_SEGMENT
    partial[(size_t(task) * splits + split) * n + c] = sum;
}
__global__ void MoeGateAndPrepareDown(const float *partial, const int *routes, const Exl3Projection *weights,
                                      float *downInput, int mid, int splits, float limit) {
    __shared__ float shared[128];
    const int c = blockIdx.x * 128 + threadIdx.x, task = blockIdx.y;
    if (routes[task] < 0) return;
    const Exl3Projection *expert = weights + routes[task] * 3;
    float gate = 0, up = 0;
    for (int split = 0; split < splits; ++split) {
        gate += partial[((size_t(task) * 2) * splits + split) * mid + c];
        up += partial[((size_t(task) * 2 + 1) * splits + split) * mid + c];
    }
    gate = RoundBfloat(Rotate(gate, shared) * __half2float(__ushort_as_half(expert[0].outputScale[c])));
    up = RoundBfloat(Rotate(up, shared) * __half2float(__ushort_as_half(expert[1].outputScale[c])));
    if (limit > 0) { gate = fminf(gate, limit); up = fmaxf(-limit, fminf(up, limit)); }
    float activated = RoundBfloat((gate / (1.0f + expf(-gate))) * up);
    activated *= __half2float(__ushort_as_half(expert[2].inputScale[c]));
    downInput[size_t(task) * mid + c] = __half2float(__float2half_rn(Rotate(activated, shared)));
}
__global__ void MoeFinish(const float *partial, const int *routes, const float *scores,
                          const Exl3Projection *weights, float *output, int k, int topk, int splits, float *perExpert = nullptr) {
    __shared__ float shared[128];
    const int c = blockIdx.x * 128 + threadIdx.x, row = blockIdx.y;
    float total = 0;
    for (int route = 0; route < topk; ++route) {
        const int task = row * topk + route;
        if (routes[task] < 0) { if (perExpert) perExpert[size_t(task)*k+c]=0; continue; }
        float value = 0;
        for (int split = 0; split < splits; ++split)
            value += partial[(size_t(task) * splits + split) * k + c];
        value = Rotate(value, shared) * __half2float(__ushort_as_half(weights[routes[task] * 3 + 2].outputScale[c]));
        // Match the existing BF16 expert boundary and FP32 routed reduction.
        value = RoundBfloat(value) * scores[task];
        if (perExpert) perExpert[size_t(task)*k+c] = value;
        total += value;
    }
    if (output) output[size_t(row) * k + c] = total;
}
struct MoeWorkspace {
    fastllm::Data input{fastllm::FLOAT32}, gate{fastllm::FLOAT32}, down{fastllm::FLOAT32}, result{fastllm::FLOAT32};
    explicit MoeWorkspace(int device) {
        for (auto *value : {&input, &gate, &down, &result}) {
            value->dataDevice = fastllm::DataDevice::CUDA; value->dataDeviceIds = {device};
        }
    }
};
}
void FastllmCudaReleaseExl3(const fastllm::Data *owner) {
#ifdef FASTLLM_USE_MAYA_EXL3
    FastllmCudaReleaseExl3Maya(owner);
#endif
    std::lock_guard<std::mutex> lock(DirectoryMutex());
    auto &directories = Directories();
    for (auto it = directories.begin(); it != directories.end(); ) {
        if (it->first.first == owner) it = directories.erase(it); else ++it;
    }
}
bool FastllmCudaExl3Moe(const fastllm::Data &input, fastllm::Data &output,
                        const fastllm::Data &index, const fastllm::Data &score,
                        fastllm::Data **weights, int count, float limit) {
#ifdef FASTLLM_USE_MAYA_EXL3
    if (FastllmCudaExl3MayaMoe(input, output, index, score, weights, count, limit)) return true;
#endif
    using namespace fastllm;
    if (input.dataType != BFLOAT16 || input.dims.size() != 2 || input.dims[0] > 8 || count < 4 ||
        weights[0] || !weights[2] || weights[2]->dataType != EXL3 || index.dataDevice != DataDevice::CUDA ||
        score.dataDevice != DataDevice::CUDA || index.dataType != INT32 || score.dataType != FLOAT32) return false;
    const int k = input.dims[1], mid = weights[2]->dims[0] / 2, rows = input.dims[0], topk = index.dims.back();
    const int experts = count / 2 - 1, device = FastllmCudaGetDevice();
    std::vector<Exl3Projection> descriptors(experts * 3);
    for (int expert = 0; expert < experts; ++expert) {
        Data *gate = weights[2 + expert * 2], *down = weights[3 + expert * 2];
        if (!gate || !down || gate->dataType != EXL3 || down->dataType != EXL3 ||
            !gate->cudaData || !down->cudaData || gate->dims.size() != 2 || down->dims.size() != 2 ||
            gate->dims[0] != 2 * mid || gate->dims[1] != k || down->dims[0] != k || down->dims[1] != mid || gate->blockM != 2 || down->blockM != 1) return false;
        for (int part = 0; part < 3; ++part) {
            Data &weight = part == 2 ? *down : *gate;
            const int inputs = part == 2 ? mid : k, outputs = part == 2 ? k : mid;
            const auto *base = static_cast<const uint8_t *>(weight.cudaData) +
                (part == 1 ? uint64_t(mid) * k * weight.blockK / 16 + 2 * uint64_t(mid + k) : 0);
            auto &desc = descriptors[expert * 3 + part];
            desc.trellis = reinterpret_cast<const unsigned *>(base);
            desc.inputScale = reinterpret_cast<const unsigned short *>(base + uint64_t(inputs) * outputs * weight.blockK / 16);
            desc.outputScale = desc.inputScale + inputs; desc.bits = weight.blockK;
        }
    }
    std::shared_ptr<ExpertDirectory> directory;
    {
        std::lock_guard<std::mutex> lock(DirectoryMutex());
        auto &entry = Directories()[{weights[2], device}];
        if (!entry) entry.reset(new ExpertDirectory);
        directory = entry;
        bool changed = entry->host.size() != descriptors.size();
        for (size_t i = 0; !changed && i < descriptors.size(); ++i) {
            const auto &a = entry->host[i], &b = descriptors[i];
            changed = a.trellis != b.trellis || a.inputScale != b.inputScale || a.outputScale != b.outputScale || a.bits != b.bits;
        }
        if (changed) {
            entry->host = std::move(descriptors);
            entry->device.dataDevice = DataDevice::CUDA; entry->device.dataDeviceIds = {device};
            entry->device.Resize({int(entry->host.size() * sizeof(Exl3Projection))}); entry->device.Allocate(false);
            FastllmCudaCopyFromHostToDevice(entry->device.cudaData, entry->host.data(), entry->device.GetBytes());
        }
    }
    using Key = std::tuple<int,int,int,int,int>;
    static thread_local std::map<Key, std::unique_ptr<MoeWorkspace>> workspaces;
    auto &owned = workspaces[{device,k,mid,rows,topk}];
    if (!owned) owned.reset(new MoeWorkspace(device));
    auto &work = *owned;
    const int tasks = rows * topk, gateSplits = (k + 255) / 256, downSplits = (mid + 255) / 256;
    work.input.Resize({tasks * 2, k}); work.input.Allocate(false);
    work.gate.Resize({tasks * 2 * gateSplits, mid}); work.gate.Allocate(false);
    work.down.Resize({tasks, mid}); work.down.Allocate(false);
    work.result.Resize({tasks * downSplits, k}); work.result.Allocate(false);
    if (output.dataType != FLOAT32) { output.FreeSpace(); output.dataType = FLOAT32; }
    output.Resize({rows,k}); output.ToDevice(DataDevice::CUDA, {device}, false); output.Allocate(false);
    const auto *table = static_cast<const Exl3Projection *>(directory->device.cudaData);
    const auto *routes = static_cast<const int *>(index.cudaData);
    const auto stream = cudaStreamPerThread;
    MoePrepare<<<dim3(k / 128, topk * 2, rows),128,0,stream>>>(input.cudaData, input.dataType, routes, table,
        static_cast<float *>(work.input.cudaData),k,topk);
    MoeProduct<<<dim3(mid / 128, gateSplits, tasks * 2),128,0,stream>>>(static_cast<float *>(work.input.cudaData),
        routes,table,static_cast<float *>(work.gate.cudaData),k,mid,topk,gateSplits,false);
    MoeGateAndPrepareDown<<<dim3(mid / 128,tasks),128,0,stream>>>(static_cast<float *>(work.gate.cudaData),routes,table,
        static_cast<float *>(work.down.cudaData),mid,gateSplits,limit);
    MoeProduct<<<dim3(k / 128,downSplits,tasks),128,0,stream>>>(static_cast<float *>(work.down.cudaData),routes,table,
        static_cast<float *>(work.result.cudaData),mid,k,topk,downSplits,true);
    MoeFinish<<<dim3(k / 128,rows),128,0,stream>>>(static_cast<float *>(work.result.cudaData),routes,
        static_cast<const float *>(score.cudaData),table,static_cast<float *>(output.cudaData),k,topk,downSplits);
    AssertInFastLLM(cudaGetLastError() == cudaSuccess, "EXL3: indexed MoE launch failed.");
    return true;
}

namespace {
__global__ void CacheDescriptors(const uint8_t *records, const int32_t *slots,
    const uint64_t *offsets, size_t stride, size_t downOffset, int k, int mid,
    int gateBits, int downBits, int count, Exl3Projection *table, int *routes) {
    const int at=threadIdx.x;
    if (at>=count) return;
    const int slot=slots[at]; routes[at]=slot<0?-1:at;
    if (slot<0) return;
    const auto *record=records+(offsets?offsets[slot]:size_t(slot)*stride);
    for(int part=0;part<3;++part) {
        const int inputs=part==2?mid:k, outputs=part==2?k:mid, bits=part==2?downBits:gateBits;
        const size_t partBytes=size_t(mid)*k*gateBits/16+2*size_t(mid+k);
        const auto *base=record+(part==2?downOffset:part*partBytes);
        table[at*3+part]={reinterpret_cast<const unsigned *>(base),
            reinterpret_cast<const unsigned short *>(base+size_t(inputs)*outputs*bits/16),
            reinterpret_cast<const unsigned short *>(base+size_t(inputs)*outputs*bits/16)+inputs,bits};
    }
}
struct CacheWorkspace: MoeWorkspace {
    fastllm::Data table{fastllm::INT8}, routes{fastllm::INT32};
    explicit CacheWorkspace(int device):MoeWorkspace(device) {
        for (auto *d:{&table,&routes}) {d->dataDevice=fastllm::CUDA;d->dataDeviceIds={device};}
    }
};
}
bool FastllmCudaExl3Cache(const fastllm::Data &input, const uint8_t *records,
    const int32_t *slots, const uint64_t *offsets, size_t stride, size_t downOffset,
    int mid, int gateBits, int downBits, const float *scores, int topk,
    float limit, float *perExpert, int workspace) {
    if (input.dims.size()!=2 || input.dims[1]<128 || input.dims[1]%128 || mid<128 || mid%128 ||
        input.dataDevice!=fastllm::CUDA || !input.cudaData || !records || !slots || !scores || !perExpert || workspace<0 || workspace>1) return false;
#ifdef FASTLLM_USE_MAYA_EXL3
    if(FastllmCudaExl3MayaCache(input,records,slots,offsets,stride,downOffset,mid,
        gateBits,downBits,scores,topk,limit,perExpert,workspace)) return true;
#endif
    using namespace fastllm;
    if(input.dataType!=BFLOAT16 || input.dims.size()!=2 || input.dims[0]<1 || input.dims[0]>8 ||
        topk<1 || topk>8 || !Exl3BitRateSupported(gateBits) || !Exl3BitRateSupported(downBits)) return false;
    const int k=input.dims[1],rows=input.dims[0],tasks=rows*topk,device=FastllmCudaGetDevice();
    using Key=std::tuple<int,int,int,int,int,int>;
    static thread_local std::map<Key,std::unique_ptr<CacheWorkspace>> workspaces;
    auto &owned=workspaces[{device,k,mid,rows,topk,workspace}];if(!owned)owned.reset(new CacheWorkspace(device));auto &work=*owned;
    const int gateSplits=(k+255)/256,downSplits=(mid+255)/256;
    work.input.Resize({tasks*2,k});work.gate.Resize({tasks*2*gateSplits,mid});
    work.down.Resize({tasks,mid});work.result.Resize({tasks*downSplits,k});
    work.table.Resize({int(tasks*3*sizeof(Exl3Projection))});work.routes.Resize({tasks});
    for(auto *d:{&work.input,&work.gate,&work.down,&work.result,&work.table,&work.routes})d->Allocate(false);
    auto *table=static_cast<Exl3Projection *>(work.table.cudaData);auto *routes=static_cast<int *>(work.routes.cudaData);
    const auto stream=cudaStreamPerThread;
    CacheDescriptors<<<1,64,0,stream>>>(records,slots,offsets,stride,downOffset,k,mid,gateBits,downBits,tasks,table,routes);
    MoePrepare<<<dim3(k/128,topk*2,rows),128,0,stream>>>(input.cudaData,input.dataType,routes,table,
        static_cast<float *>(work.input.cudaData),k,topk);
    MoeProduct<<<dim3(mid/128,gateSplits,tasks*2),128,0,stream>>>(static_cast<float *>(work.input.cudaData),
        routes,table,static_cast<float *>(work.gate.cudaData),k,mid,topk,gateSplits,false);
    MoeGateAndPrepareDown<<<dim3(mid/128,tasks),128,0,stream>>>(static_cast<float *>(work.gate.cudaData),routes,table,
        static_cast<float *>(work.down.cudaData),mid,gateSplits,limit);
    MoeProduct<<<dim3(k/128,downSplits,tasks),128,0,stream>>>(static_cast<float *>(work.down.cudaData),routes,table,
        static_cast<float *>(work.result.cudaData),mid,k,topk,downSplits,true);
    MoeFinish<<<dim3(k/128,rows),128,0,stream>>>(static_cast<float *>(work.result.cudaData),routes,scores,table,
        nullptr,k,topk,downSplits,perExpert);
    return cudaGetLastError()==cudaSuccess;
}

bool FastllmCudaExl3Grouped(const fastllm::Data &input, const fastllm::Data &weight, fastllm::Data &output) {
    using namespace fastllm;
    if (weight.dataType != EXL3 || weight.dims.size() != 3) return false;
    Exl3StorageBytes(weight);
    const int groups = weight.dims[0], n = weight.dims[1], k = weight.dims[2];
    const int rows = input.Count(0) / (groups * k);
    AssertInFastLLM(rows > 0 && input.Count(0) == uint64_t(rows) * groups * k &&
        output.Count(0) == uint64_t(rows) * groups * n &&
        (input.dataType == FLOAT32 || input.dataType == FLOAT16 || input.dataType == BFLOAT16) &&
        (output.dataType == FLOAT32 || output.dataType == FLOAT16 || output.dataType == BFLOAT16),
        "EXL3: invalid grouped CUDA projection");
#ifdef FASTLLM_USE_MAYA_EXL3
    if (FastllmCudaExl3MayaGrouped(input, weight, output)) return true;
#endif
    Data empty;
    // The row stride is known from the grouped layout. Read and write each
    // slice in place, avoiding a gather/scatter pair for every prefill group.
    for (int g = 0; g < groups; ++g) {
        Data w, x, y;
        w.FakeFrom(weight, size_t(g) * Exl3PartBytes(n,k,weight.blockK));
        w.blockK = weight.blockK; w.blockM = 1; w.Resize({n,k});
        x.FakeFrom(input, size_t(g) * k * input.unitSize); x.Resize({rows,k});
        y.FakeFrom(output, size_t(g) * n * output.unitSize); y.Resize({rows,n});
        if (!Exl3LinearStrided(x,w,empty,y,groups*k,groups*n)) return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

void FastllmCudaRestoreExl3Weight(const fastllm::Data &weight, void *destination, void *stream) {
    const int nodes = weight.numasData.size();
    bool mapped = weight.isPinned && nodes > 0 && nodes <= 32;
    fastllm_exl3_shards::Matrix view{};
    view.rows = weight.dims[0]; view.columns = weight.dims[1];
    view.bits = weight.blockK; view.parts = weight.blockM; view.shards = nodes;
    if (mapped) for (int node = 0; node < nodes; ++node) {
        if (cudaHostGetDevicePointer(&view.pointers[node], weight.numasData[node], 0) != cudaSuccess) {
            cudaGetLastError(); mapped = false; break;
        }
    }
    if (mapped) {
        fastllm_exl3_shards::MatrixRestore<<<64, 256, 0, static_cast<cudaStream_t>(stream)>>>(
            view, static_cast<uint8_t *>(destination));
        fastllm::AssertInFastLLM(cudaGetLastError() == cudaSuccess, "EXL3 row-shard GPU restore failed");
    } else {
        // Rare unpinned fallback owns only this matrix until its copy completes.
        std::vector<uint8_t> canonical(weight.GetBytes());
        fastllm::Exl3CopyCanonical(weight, canonical.data());
        FastllmCudaCopyFromHostToDeviceAsync(destination, canonical.data(), canonical.size(), stream);
        FastllmCudaStreamSynchronize(stream);
    }
}
