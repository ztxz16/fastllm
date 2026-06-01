//
// SM70 (V100) AWQ W4A16 GEMM bridge over the ported TurboMind s884 kernels.
//
// The logic mirrors 1Cat-vLLM's csrc/quantization/awq/awq_sm70_gemm.cu, but is
// rewritten to use raw CUDA pointers instead of torch::Tensor so it can be
// driven directly from fastllm's INT4_GROUP weights.
//

#include "devices/cuda/fastllm-awq-sm70.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <set>
#include <tuple>
#include <unordered_map>

#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/gemm/cast.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/gemm.h"
#include "src/turbomind/kernels/gemm/types.h"
#include "src/turbomind/kernels/gemm/utils.h"
#include "fastllm-cuda.cuh"

namespace fastllm {
namespace awq_sm70 {

namespace tm = turbomind::gemm;

namespace {

struct Handle {
    void *tmWeight = nullptr;   // device, uint4 packed
    void *tmScales = nullptr;   // device, int32 packed
    int K = 0;                  // input dim
    int N = 0;                  // output dim
    int kLd = 0;
    int qLd = 0;
    int groupSize = 0;
};

struct WorkspaceHolder {
    void *barriers = nullptr;
    void *partials = nullptr;
    void *tensormaps = nullptr;
    int *flags = nullptr;
    tm::Workspace workspace{};
};

std::mutex g_mutex;
std::unordered_map<int, std::unique_ptr<tm::Gemm>> g_gemm;
std::unordered_map<int, WorkspaceHolder> g_workspace;
// 已自动调优过的 (device, m, n, k, group) 解码 shape；首次 kMeasure，之后 kReuse。
std::set<std::tuple<int, int, int, int, int>> g_tunedShapes;

int DenseTuneMaxM() {
    static int v = []() {
        const char *raw = getenv("FASTLLM_SM70_AWQ_TUNE_MAX_M");
        return raw ? std::max(0, atoi(raw)) : 16;
    }();
    return v;
}

tm::DispatchPolicy SelectDispatch(int device, int m, int n, int k, int group_size) {
    if (m > DenseTuneMaxM()) {
        return tm::DispatchPolicy::kDefault;
    }
    auto key = std::make_tuple(device, m, n, k, group_size);
    if (g_tunedShapes.count(key)) {
        return tm::DispatchPolicy::kReuse;
    }
    g_tunedShapes.insert(key);
    return tm::DispatchPolicy::kMeasure;
}

int CurrentDevice() {
    int dev = 0;
    cudaGetDevice(&dev);
    return dev;
}

tm::Gemm &GetGemm(int device) {
    auto it = g_gemm.find(device);
    if (it != g_gemm.end()) {
        return *it->second;
    }
    auto gemm = std::make_unique<tm::Gemm>();
    auto *raw = gemm.get();
    g_gemm.emplace(device, std::move(gemm));
    return *raw;
}

WorkspaceHolder &GetWorkspace(int device) {
    auto it = g_workspace.find(device);
    if (it != g_workspace.end()) {
        return it->second;
    }
    WorkspaceHolder holder;
    cudaMalloc(&holder.barriers, tm::Gemm::kBarriersSize);
    cudaMalloc(&holder.partials, tm::Gemm::kPartialsSize);
    cudaMalloc(&holder.tensormaps, (size_t)8192 * 128);
    cudaMalloc((void **)&holder.flags, sizeof(int));
    cudaMemset(holder.barriers, 0, tm::Gemm::kBarriersSize);
    cudaMemset(holder.partials, 0, tm::Gemm::kPartialsSize);
    cudaMemset(holder.flags, 0, sizeof(int));
    holder.workspace.barriers = holder.barriers;
    holder.workspace.barriers_size = tm::Gemm::kBarriersSize;
    holder.workspace.partials = holder.partials;
    holder.workspace.partials_size = tm::Gemm::kPartialsSize;
    holder.workspace.tensormaps = holder.tensormaps;
    holder.workspace.tensormaps_size = (size_t)8192 * 128;
    holder.workspace.flags = holder.flags;
    auto [insert_it, _] = g_workspace.emplace(device, holder);
    return insert_it->second;
}

// dst[N, K] = src[K, N] (uint16, row-major transpose)
__global__ void TransposeU16Kernel(uint16_t *dst, const uint16_t *src, int K, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total = K * N;
    if (idx >= total) {
        return;
    }
    int kk = idx / N;
    int nn = idx - kk * N;
    dst[(size_t)nn * K + kk] = src[idx];
}

// 把当前二进制在该 device 上实际执行的内核编译期 __CUDA_ARCH__ 写回主机。
// 与 TurboMind s884 GEMM 内核走相同编译路径：若用 -DCUDA_ARCH=60 之类未包含 sm_70
// 的目标编译，运行在 7.0 设备上时此处取到的也是 600，恰好对应 GEMM 内核被
// if constexpr (is_compatible(__CUDA_ARCH__)) 编译为空壳的情况。
__global__ void DetectCompiledArchKernel(int *out) {
#if defined(__CUDA_ARCH__)
    *out = __CUDA_ARCH__;
#else
    *out = 0;
#endif
}

// 返回当前 device 上实际执行内核的编译 arch（如 700）。失败返回 0。结果按进程缓存。
int CompiledArchOnCurrentDevice() {
    int *d = nullptr;
    if (cudaMalloc(&d, sizeof(int)) != cudaSuccess) {
        return 0;
    }
    int h = 0;
    DetectCompiledArchKernel<<<1, 1>>>(d);
    cudaError_t e = cudaDeviceSynchronize();
    if (e == cudaSuccess) {
        if (cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
            h = 0;
        }
    } else {
        h = 0;
    }
    cudaFree(d);
    return h;
}

}  // namespace

bool Supported() {
    int dev = 0;
    int major = 0, minor = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess) {
        return false;
    }
    if (major != 7 || minor != 0) {
        return false;
    }
    // 设备算力为 7.0 还不够：必须确认当前二进制确实为 sm_70 编译了 tensor core 内核。
    // 若用未包含 sm_70 的目标（如 -DCUDA_ARCH=60）编译，TurboMind s884 内核体会被
    // if constexpr (is_compatible(__CUDA_ARCH__)) 整段编译为空，GEMM 不写输出 -> 乱码。
    // 这里运行时探测实际编译的 __CUDA_ARCH__，<700 则禁用本路径、退回原始 int4 实现。
    static const int compiledArch = CompiledArchOnCurrentDevice();
    if (compiledArch < 700) {
        static bool warned = false;
        if (!warned) {
            printf("[Fastllm] SM70 AWQ disabled: binary not compiled for sm_70 "
                   "(detected __CUDA_ARCH__=%d). Falling back to native int4 path.\n",
                   compiledArch);
            warned = true;
        }
        return false;
    }
    bool grouped = true;
    auto converters = tm::GetConverters(turbomind::kHalf, turbomind::kUint4, turbomind::kHalf, grouped, 70);
    return converters[0] != nullptr && converters[1] != nullptr;
}

void *Prepare(const uint16_t *d_qvals_u16, const half *d_scales, const half *d_zeros,
              int K, int N, int num_groups, int group_size, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(g_mutex);

    // 入参判空：任意源指针为空都会让后续 TurboMind Convert 命中
    // "'S' Must be non NULL" 断言并 abort，这里先打印出来便于定位。
    if (d_qvals_u16 == nullptr || d_scales == nullptr || d_zeros == nullptr) {
        printf("FastllmAwqSm70 Prepare error: null input pointer "
               "(qvals=%p scales=%p zeros=%p) K=%d N=%d num_groups=%d group_size=%d\n",
               (const void *)d_qvals_u16, (const void *)d_scales, (const void *)d_zeros,
               K, N, num_groups, group_size);
        return nullptr;
    }

    const bool grouped = (group_size != K);
    auto converters = tm::GetConverters(turbomind::kHalf, turbomind::kUint4, turbomind::kHalf, grouped, 70);
    const auto *conv_w = converters[0];
    const auto *conv_s = converters[1];
    if (conv_w == nullptr || conv_s == nullptr) {
        printf("FastllmAwqSm70 Prepare error: no compatible TurboMind converters.\n");
        return nullptr;
    }

    // ---- weight ----
    const auto order_w = conv_w->order;
    const bool is_A_w = tm::get_operand_tag(conv_w->pack) == tm::OPERAND_A;
    const bool is_B_w = !is_A_w;

    const uint16_t *srcU16 = d_qvals_u16;  // [K, N]
    uint16_t *transposed = nullptr;
    if (order_w == tm::kRowMajor) {
        cudaError_t e = cudaMalloc((void **)&transposed, (size_t)K * N * sizeof(uint16_t));
        if (e != cudaSuccess || transposed == nullptr) {
            printf("FastllmAwqSm70 Prepare error: cudaMalloc(transposed, %zu bytes) failed: %s\n",
                   (size_t)K * N * sizeof(uint16_t), cudaGetErrorString(e));
            return nullptr;
        }
        int total = K * N;
        int threads = 256;
        TransposeU16Kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(transposed, d_qvals_u16, K, N);
        srcU16 = transposed;  // [N, K]
    }

    tm::MatrixLayout w_desc{
        turbomind::kHalf,
        order_w,
        N,
        K,
        order_w == tm::kRowMajor ? K : N,
    };
    if (is_B_w) {
        std::swap(w_desc.rows, w_desc.cols);
        w_desc.order = ~w_desc.order;
    }

    tm::MatrixLayout k_desc = w_desc;
    k_desc.type = turbomind::data_type_v<turbomind::uint4_t>;
    k_desc.pack = conv_w->pack;
    if (is_A_w) {
        k_desc = tm::transpose(k_desc);
    }

    void *tmWeight = nullptr;
    {
        cudaError_t e = cudaMalloc(&tmWeight, (size_t)K * N / 2);  // uint4 packed
        if (e != cudaSuccess || tmWeight == nullptr) {
            printf("FastllmAwqSm70 Prepare error: cudaMalloc(tmWeight, %zu bytes) failed: %s\n",
                   (size_t)K * N / 2, cudaGetErrorString(e));
            if (transposed) cudaFree(transposed);
            return nullptr;
        }
    }
    if (srcU16 == nullptr) {
        printf("FastllmAwqSm70 Prepare error: weight source pointer is null before Convert "
               "(order_w=%d K=%d N=%d).\n", (int)order_w, K, N);
        if (transposed) cudaFree(transposed);
        cudaFree(tmWeight);
        return nullptr;
    }
    if (conv_w->Convert(srcU16, w_desc, tmWeight, k_desc, stream) != 0) {
        printf("FastllmAwqSm70 Prepare error: weight conversion failed.\n");
        if (transposed) cudaFree(transposed);
        cudaFree(tmWeight);
        return nullptr;
    }
    if (transposed) {
        cudaFree(transposed);
    }

    // ---- scales + zeros ----
    half *fused = nullptr;
    {
        cudaError_t e = cudaMalloc((void **)&fused, (size_t)num_groups * N * 2 * sizeof(half));
        if (e != cudaSuccess || fused == nullptr) {
            printf("FastllmAwqSm70 Prepare error: cudaMalloc(fused, %zu bytes) failed: %s\n",
                   (size_t)num_groups * N * 2 * sizeof(half), cudaGetErrorString(e));
            cudaFree(tmWeight);
            return nullptr;
        }
    }
    turbomind::fuse_scales_and_zeros(fused, d_scales, const_cast<half *>(d_zeros),
                                     (size_t)num_groups * N, stream);

    const auto order_s = conv_s->order;
    const bool is_A_s = tm::get_operand_tag(conv_s->pack) == tm::OPERAND_U;
    const bool is_B_s = !is_A_s;

    tm::MatrixLayout s_desc{
        turbomind::kUint32,
        order_s,
        N,
        num_groups,
        N,
    };
    if (is_B_s) {
        std::swap(s_desc.rows, s_desc.cols);
        s_desc.order = ~s_desc.order;
    }

    tm::MatrixLayout q_desc = s_desc;
    q_desc.pack = conv_s->pack;
    if (is_A_s) {
        q_desc = tm::transpose(q_desc);
    }

    void *tmScales = nullptr;
    {
        cudaError_t e = cudaMalloc(&tmScales, (size_t)num_groups * N * sizeof(int32_t));
        if (e != cudaSuccess || tmScales == nullptr) {
            printf("FastllmAwqSm70 Prepare error: cudaMalloc(tmScales, %zu bytes) failed: %s\n",
                   (size_t)num_groups * N * sizeof(int32_t), cudaGetErrorString(e));
            cudaFree(fused);
            cudaFree(tmWeight);
            return nullptr;
        }
    }
    if (fused == nullptr) {
        printf("FastllmAwqSm70 Prepare error: scale source pointer is null before Convert "
               "(num_groups=%d N=%d).\n", num_groups, N);
        cudaFree(fused);
        cudaFree(tmWeight);
        cudaFree(tmScales);
        return nullptr;
    }
    if (conv_s->Convert(fused, s_desc, tmScales, q_desc, stream) != 0) {
        printf("FastllmAwqSm70 Prepare error: scale conversion failed.\n");
        cudaFree(fused);
        cudaFree(tmWeight);
        cudaFree(tmScales);
        return nullptr;
    }
    cudaFree(fused);

    auto *handle = new Handle();
    handle->tmWeight = tmWeight;
    handle->tmScales = tmScales;
    handle->K = K;
    handle->N = N;
    handle->kLd = k_desc.ld;
    handle->qLd = q_desc.ld;
    handle->groupSize = group_size;
    return handle;
}

bool Gemm(void *handlePtr, const half *in, half *out, int tokens, cudaStream_t stream) {
    if (handlePtr == nullptr) {
        return false;
    }
    auto *handle = static_cast<Handle *>(handlePtr);
    const int m = tokens;          // tokens
    const int k = handle->K;       // input dim
    const int n = handle->N;       // output dim
    const int group_size = handle->groupSize;

    std::lock_guard<std::mutex> lock(g_mutex);
    const int device = CurrentDevice();

    const bool grouped = (group_size != k);
    auto converters = tm::GetConverters(turbomind::kHalf, turbomind::kUint4, turbomind::kHalf, grouped, 70);
    const auto *conv_w = converters[0];
    const auto *conv_s = converters[1];
    if (conv_w == nullptr || conv_s == nullptr) {
        return false;
    }

    tm::MatrixLayout desc_A{turbomind::kHalf, tm::kRowMajor, m, k, k};
    tm::MatrixLayout desc_U{};

    const auto order_w = conv_w->order;
    const bool is_A_w = tm::get_operand_tag(conv_w->pack) == tm::OPERAND_A;
    const bool is_B_w = !is_A_w;

    tm::MatrixLayout w_desc{turbomind::kHalf, order_w, n, k, order_w == tm::kRowMajor ? k : n};
    if (is_B_w) {
        std::swap(w_desc.rows, w_desc.cols);
        w_desc.order = ~w_desc.order;
    }
    tm::MatrixLayout desc_B = w_desc;
    desc_B.type = turbomind::data_type_v<turbomind::uint4_t>;
    desc_B.pack = conv_w->pack;
    if (is_A_w) {
        desc_B = tm::transpose(desc_B);
    }
    desc_B.ld = handle->kLd;

    const auto order_s = conv_s->order;
    const bool is_A_s = tm::get_operand_tag(conv_s->pack) == tm::OPERAND_U;
    const bool is_B_s = !is_A_s;
    const int num_groups = k / group_size;

    tm::MatrixLayout s_desc{turbomind::kUint32, order_s, n, num_groups, n};
    if (is_B_s) {
        std::swap(s_desc.rows, s_desc.cols);
        s_desc.order = ~s_desc.order;
    }
    tm::MatrixLayout desc_V = s_desc;
    desc_V.pack = conv_s->pack;
    if (is_A_s) {
        desc_V = tm::transpose(desc_V);
    }
    desc_V.ld = handle->qLd;

    tm::MatrixLayout desc_D{turbomind::kHalf, tm::kRowMajor, m, n, n};

    tm::Operation op{};
    op.dispatch = SelectDispatch(device, m, n, k, group_size);
    op.epilogue = tm::Epilogue::kNone;
    op.quant_a = {tm::QuantType::kNone, 0};
    op.quant_b = {tm::QuantType::kK, group_size};
    op.batch_dim = 0;

    auto &workspace = GetWorkspace(device);
    auto &gemm = GetGemm(device);

    const int ec = gemm.Run(op, 1.f, in, desc_A, nullptr, desc_U, handle->tmWeight, desc_B,
                            handle->tmScales, desc_V, 0.f, out, desc_D, out, desc_D,
                            workspace.workspace, stream);
    if (ec != 0) {
        printf("FastllmAwqSm70 Gemm error: TurboMind GEMM failed (ec=%d).\n", ec);
        return false;
    }
    return true;
}

void Free(void *handlePtr) {
    if (handlePtr == nullptr) {
        return;
    }
    auto *handle = static_cast<Handle *>(handlePtr);
    if (handle->tmWeight) cudaFree(handle->tmWeight);
    if (handle->tmScales) cudaFree(handle->tmScales);
    delete handle;
}

}  // namespace awq_sm70
}  // namespace fastllm
