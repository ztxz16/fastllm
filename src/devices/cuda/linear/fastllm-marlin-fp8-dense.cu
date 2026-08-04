/*
 * Dense Marlin W8A16 FP8 and W4A16 NVFP4 launchers for FastLLM.
 * Kernel body vendored from vLLM csrc/quantization/marlin (Apache-2.0).
 * SM75 uses a two-stage pipeline; NVFP4 on SM80+ uses four stages, matching
 * vLLM's ops.marlin_gemm(b_q_type=float4_e2m1f) dispatch.
 */

#include "fastllm-cuda.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <mutex>

#define MARLIN_NAMESPACE_NAME fastllm_marlin_dense_fp8
#include "marlin_dense_fp8/kernel.h"
#include "marlin_dense_fp8/marlin_template.h"

namespace {

using KernelFn = void (*)(MARLIN_KERNEL_PARAMS);

static bool DeviceOk() {
#ifdef CUDA_NO_TENSOR_CORE
    return false;
#else
    int dev = 0, major = 0, minor = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess)
        return false;
    if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess)
        return false;
    return major * 10 + minor >= 75;
#endif
}

static int DeviceArch() {
    int dev = 0, major = 0, minor = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                               dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                               dev) != cudaSuccess) {
        return 0;
    }
    return major * 10 + minor;
}

// Explicit FE4M3 group128 (group_blocks=8) + channelwise (-1) SM75 stages=2.
#define RET_K(THREADS, TM, TN, TK, M8, GB)                                      \
    return MARLIN_NAMESPACE_NAME::Marlin<                                       \
        vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(),         \
        vllm::kFloat16.id(), (THREADS), (TM), (TN), (TK), (M8), 2, (GB), false>

static KernelFn PickKernel(int sizeM, int threadK, int threadN, int groupBlocks,
                           bool m8, int &threads) {
    threads = 0;
    const int tm = (m8 || sizeM <= 8) ? 1 : std::min(4, (sizeM + 15) / 16);
    const bool useM8 = m8 || sizeM <= 8;

    if (groupBlocks != 8 && groupBlocks != -1) return nullptr;

    if (useM8) {
        if (threadK == 128 && threadN == 128) {
            threads = 256;
            if (groupBlocks == 8) RET_K(256, 1, 8, 8, true, 8);
            RET_K(256, 1, 8, 8, true, -1);
        }
        if (threadK == 64 && threadN == 128) {
            threads = 128;
            if (groupBlocks == 8) RET_K(128, 1, 8, 4, true, 8);
            RET_K(128, 1, 8, 4, true, -1);
        }
        if (threadK == 128 && threadN == 64) {
            threads = 128;
            if (groupBlocks == 8) RET_K(128, 1, 4, 8, true, 8);
            RET_K(128, 1, 4, 8, true, -1);
        }
        return nullptr;
    }

    if (threadK == 64 && threadN == 256 && tm >= 1 && tm <= 4) {
        threads = 256;
        if (groupBlocks == 8) {
            if (tm == 1) RET_K(256, 1, 16, 4, false, 8);
            if (tm == 2) RET_K(256, 2, 16, 4, false, 8);
            if (tm == 3) RET_K(256, 3, 16, 4, false, 8);
            if (tm == 4) RET_K(256, 4, 16, 4, false, 8);
        } else {
            if (tm == 1) RET_K(256, 1, 16, 4, false, -1);
            if (tm == 2) RET_K(256, 2, 16, 4, false, -1);
            if (tm == 3) RET_K(256, 3, 16, 4, false, -1);
            if (tm == 4) RET_K(256, 4, 16, 4, false, -1);
        }
    }
    if (threadK == 64 && threadN == 128) {
        threads = 128;
        if (groupBlocks == 8) {
            if (tm == 1) RET_K(128, 1, 8, 4, false, 8);
            if (tm == 2) RET_K(128, 2, 8, 4, false, 8);
            if (tm == 3) RET_K(128, 3, 8, 4, false, 8);
            if (tm == 4) RET_K(128, 4, 8, 4, false, 8);
        } else {
            if (tm == 1) RET_K(128, 1, 8, 4, false, -1);
            if (tm == 2) RET_K(128, 2, 8, 4, false, -1);
            if (tm == 3) RET_K(128, 3, 8, 4, false, -1);
            if (tm == 4) RET_K(128, 4, 8, 4, false, -1);
        }
    }
    if (threadK == 128 && threadN == 64) {
        threads = 128;
        if (groupBlocks == 8) {
            if (tm == 1) RET_K(128, 1, 4, 8, false, 8);
            if (tm == 2) RET_K(128, 2, 4, 8, false, 8);
            if (tm == 3) RET_K(128, 3, 4, 8, false, 8);
            if (tm == 4) RET_K(128, 4, 4, 8, false, 8);
        } else {
            if (tm == 1) RET_K(128, 1, 4, 8, false, -1);
            if (tm == 2) RET_K(128, 2, 4, 8, false, -1);
            if (tm == 3) RET_K(128, 3, 4, 8, false, -1);
            if (tm == 4) RET_K(128, 4, 4, 8, false, -1);
        }
    }
    if (threadK == 128 && threadN == 128 && tm == 1) {
        threads = 256;
        if (groupBlocks == 8) RET_K(256, 1, 8, 8, false, 8);
        RET_K(256, 1, 8, 8, false, -1);
    }
    return nullptr;
}
#undef RET_K

// Explicit FE2M1 + special FE4M3 scale, group16 (group_blocks=1).  vLLM uses
// stages=2 on SM75 and stages=4 on every newer architecture.
#define RET_FP4(STAGES, THREADS, TM, TN, TK, M8)                               \
    return MARLIN_NAMESPACE_NAME::Marlin<                                     \
        vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(),        \
        vllm::kFE4M3fn.id(), (THREADS), (TM), (TN), (TK), (M8),              \
        (STAGES), 1, false>
#define RET_FP4_FOR_ARCH(THREADS, TM, TN, TK, M8)                             \
    do {                                                                      \
        if (stages == 2) RET_FP4(2, THREADS, TM, TN, TK, M8);                \
        RET_FP4(4, THREADS, TM, TN, TK, M8);                                 \
    } while (0)

static KernelFn PickFp4Kernel(int sizeM, int threadK, int threadN,
                              bool m8, int stages, int &threads) {
    threads = 0;
    if (stages != 2 && stages != 4) return nullptr;
    const int tm = (m8 || sizeM <= 8) ? 1 :
                   std::min(4, (sizeM + 15) / 16);
    const bool useM8 = m8 || sizeM <= 8;

    if (useM8) {
        if (threadK == 128 && threadN == 128) {
            threads = 256;
            RET_FP4_FOR_ARCH(256, 1, 8, 8, true);
        }
        if (threadK == 64 && threadN == 128) {
            threads = 128;
            RET_FP4_FOR_ARCH(128, 1, 8, 4, true);
        }
        if (threadK == 128 && threadN == 64) {
            threads = 128;
            RET_FP4_FOR_ARCH(128, 1, 4, 8, true);
        }
        return nullptr;
    }

    // M=9..16 still uses the small-batch 128x128 priority in vLLM, but with
    // the regular (non-m8) output path.
    if (threadK == 128 && threadN == 128 && tm == 1) {
        threads = 256;
        RET_FP4_FOR_ARCH(256, 1, 8, 8, false);
    }
    if (threadK == 64 && threadN == 256 && tm >= 1 && tm <= 4) {
        threads = 256;
        if (tm == 1) RET_FP4_FOR_ARCH(256, 1, 16, 4, false);
        if (tm == 2) RET_FP4_FOR_ARCH(256, 2, 16, 4, false);
        if (tm == 3) RET_FP4_FOR_ARCH(256, 3, 16, 4, false);
        RET_FP4_FOR_ARCH(256, 4, 16, 4, false);
    }
    if (threadK == 64 && threadN == 128) {
        threads = 128;
        if (tm == 1) RET_FP4_FOR_ARCH(128, 1, 8, 4, false);
        if (tm == 2) RET_FP4_FOR_ARCH(128, 2, 8, 4, false);
        if (tm == 3) RET_FP4_FOR_ARCH(128, 3, 8, 4, false);
        RET_FP4_FOR_ARCH(128, 4, 8, 4, false);
    }
    if (threadK == 128 && threadN == 64) {
        threads = 128;
        if (tm == 1) RET_FP4_FOR_ARCH(128, 1, 4, 8, false);
        if (tm == 2) RET_FP4_FOR_ARCH(128, 2, 4, 8, false);
        if (tm == 3) RET_FP4_FOR_ARCH(128, 3, 4, 8, false);
        RET_FP4_FOR_ARCH(128, 4, 4, 8, false);
    }
    return nullptr;
}
#undef RET_FP4_FOR_ARCH
#undef RET_FP4

static bool SelectTile(int sizeM, int sizeN, int sizeK, int &threadK, int &threadN) {
    static const int smallM[][2] = {{64, 128}, {128, 64}, {128, 128}};
    static const int largeM[][2] = {{64, 128}, {128, 64}, {64, 256}, {128, 128}};
    const int (*cfgs)[2] = sizeM <= 8 ? smallM : largeM;
    int n = sizeM <= 8 ? 3 : 4;
    for (int i = 0; i < n; i++) {
        if (sizeK % cfgs[i][0] == 0 && sizeN % cfgs[i][1] == 0) {
            threadK = cfgs[i][0];
            threadN = cfgs[i][1];
            return true;
        }
    }
    return false;
}

static bool SelectFp4Tile(int sizeM, int sizeN, int sizeK,
                          int &threadK, int &threadN) {
    static const int smallM[][2] = {{128, 128}, {64, 128}, {128, 64}};
    static const int largeM[][2] = {{64, 256}, {64, 128}, {128, 64}};
    const int (*configs)[2] = sizeM <= 16 ? smallM : largeM;
    for (int i = 0; i < 3; i++) {
        if (sizeK % configs[i][0] == 0 &&
            sizeN % configs[i][1] == 0) {
            threadK = configs[i][0];
            threadN = configs[i][1];
            return true;
        }
    }
    return false;
}

static bool PrepareKernels(int device) {
    int prev = -1;
    cudaGetDevice(&prev);
    if (prev != device && cudaSetDevice(device) != cudaSuccess) return false;

    int maxShared = 0;
    bool ok = cudaDeviceGetAttribute(&maxShared,
                                     cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                     device) == cudaSuccess &&
              maxShared > 0;
    static const int tiles[][2] = {{64, 128}, {128, 64}, {128, 128}, {64, 256}};
    if (ok) {
        // Cover every non-m8 thread_m specialization. In particular, rows
        // 33..48 select thread_m=3; without preparing that specialization its
        // first real launch can fail with too much dynamic shared memory.
        for (int m : {4, 8, 16, 32, 48, 64}) {
            for (auto &t : tiles) {
                for (int gb : {8, -1}) {
                    int threads = 0;
                    KernelFn k = PickKernel(m, t[0], t[1], gb, m <= 8, threads);
                    if (k == nullptr) continue;
                    if (cudaFuncSetAttribute(
                            k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            maxShared) != cudaSuccess) {
                        ok = false;
                        break;
                    }
                }
                if (!ok) break;
            }
            if (!ok) break;
        }
    }
    if (prev >= 0 && prev != device) cudaSetDevice(prev);
    return ok;
}

static bool EnsureKernels(int device) {
    static std::mutex mu;
    static std::vector<int8_t> ready;
    std::lock_guard<std::mutex> lock(mu);
    if (device < 0) return false;
    if ((int)ready.size() <= device) ready.resize(device + 1, -1);
    if (ready[device] < 0) ready[device] = PrepareKernels(device) ? 1 : 0;
    return ready[device] == 1;
}

static bool PrepareFp4Kernels(int device) {
    int previous = -1;
    cudaGetDevice(&previous);
    if (previous != device && cudaSetDevice(device) != cudaSuccess) return false;

    int major = 0, minor = 0, maxShared = 0;
    bool ok = cudaDeviceGetAttribute(&major,
                                     cudaDevAttrComputeCapabilityMajor,
                                     device) == cudaSuccess &&
              cudaDeviceGetAttribute(&minor,
                                     cudaDevAttrComputeCapabilityMinor,
                                     device) == cudaSuccess &&
              cudaDeviceGetAttribute(&maxShared,
                                     cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                     device) == cudaSuccess &&
              major * 10 + minor >= 75 && maxShared > 0;
    const int stages = major == 7 && minor == 5 ? 2 : 4;
    static const int tiles[][2] = {
        {128, 128}, {64, 128}, {128, 64}, {64, 256}};
    if (ok) {
        for (int m : {4, 8, 16, 32, 48, 64}) {
            for (auto &tile : tiles) {
                int threads = 0;
                KernelFn kernel = PickFp4Kernel(
                    m, tile[0], tile[1], m <= 8, stages, threads);
                if (kernel == nullptr) continue;
                if (cudaFuncSetAttribute(
                        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                        maxShared) != cudaSuccess) {
                    ok = false;
                    break;
                }
            }
            if (!ok) break;
        }
    }
    if (previous >= 0 && previous != device) cudaSetDevice(previous);
    return ok;
}

static bool EnsureFp4Kernels(int device) {
    static std::mutex mutex;
    static std::vector<int8_t> ready;
    std::lock_guard<std::mutex> lock(mutex);
    if (device < 0) return false;
    if ((int)ready.size() <= device) ready.resize(device + 1, -1);
    if (ready[device] < 0) {
        ready[device] = PrepareFp4Kernels(device) ? 1 : 0;
    }
    return ready[device] == 1;
}

struct CTmpBuf {
    float *ptr = nullptr;
    size_t elems = 0;
    int device = -1;
};

static CTmpBuf &GetCTmp(int device) {
    static CTmpBuf bufs[16];
    if (device < 0 || device >= 16) device = 0;
    return bufs[device];
}

static bool EnsureCTmp(int device, size_t elems) {
    CTmpBuf &b = GetCTmp(device);
    if (b.device == device && b.ptr != nullptr && b.elems >= elems) return true;
    int prev = -1;
    cudaGetDevice(&prev);
    if (prev != device) cudaSetDevice(device);
    if (b.ptr) FastllmCudaFree(b.ptr);
    FastllmCudaClearThreadError();
    b.ptr = (float *)FastllmCudaMalloc(elems * sizeof(float));
    b.elems = (b.ptr && !FastllmCudaGetThreadError()) ? elems : 0;
    b.device = device;
    if (!b.ptr) FastllmCudaClearThreadError();
    if (prev >= 0 && prev != device) cudaSetDevice(prev);
    return b.ptr != nullptr;
}

}  // namespace

extern "C" bool FastllmCudaMarlinNVFP4Supported(int size_n, int size_k) {
    if (!DeviceOk() || size_n <= 0 || size_k <= 0 ||
        size_n % 64 != 0 || size_k % 64 != 0 ||
        !((size_k % 64 == 0 && size_n % 128 == 0) ||
          (size_k % 128 == 0 && size_n % 64 == 0))) {
        return false;
    }
    int device = 0;
    return cudaGetDevice(&device) == cudaSuccess &&
           EnsureFp4Kernels(device);
}

extern "C" bool FastllmCudaMarlinHalfFP8Gemm(
        const void *a, const uint32_t *b_q_weight, const void *b_scales,
        void *c, int size_m, int size_n, int size_k, int group_size,
        int *workspace) {
    if (!DeviceOk() || size_m <= 0 || size_n <= 0 || size_k <= 0) return false;
    if (group_size != 128 && group_size != -1) return false;
    if (size_n % 64 != 0 || size_k % 64 != 0) return false;
    if (group_size == 128 && size_k % 128 != 0) return false;

    int dev = 0, sms = 0, maxShared = 0;
    cudaGetDevice(&dev);
    if (!EnsureKernels(dev)) return false;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&maxShared, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (sms <= 0 || maxShared <= 0) return false;

    // vLLM: c_tmp = sms * max_m_block * max_thread_n
    int maxMBlock = size_m <= 8 ? 8 : std::min(64, ((size_m + 15) / 16) * 16);
    size_t cTmpElems = (size_t)sms * maxMBlock * 256;
    if (!EnsureCTmp(dev, cTmpElems)) return false;

    int numGroups = (group_size == -1) ? 1 : (size_k / group_size);
    int groupBlocks = (group_size == -1) ? -1 : (group_size / 16);

    // A Marlin kernel invocation can process many complete 64-row blocks, but
    // its internal parallel count is deliberately integral. Launch the bulk
    // and the final partial block separately; otherwise e.g. M=534 computes
    // only 512 rows and leaves the tail uninitialised.
    int row = 0;
    int remaining = size_m;
    const int maxParallel = size_n <= 4096 ? 128 : 16;
    while (remaining > 0) {
        int chunkM = remaining;
        if (remaining >= 64) {
            chunkM = std::min(remaining / 64, maxParallel) * 64;
        }

        int threadK = 0, threadN = 0;
        if (!SelectTile(chunkM, size_n, size_k, threadK, threadN)) return false;

        int threads = 0;
        bool m8 = chunkM <= 8;
        KernelFn kernel = PickKernel(chunkM, threadK, threadN, groupBlocks, m8, threads);
        if (kernel == nullptr) return false;

        const half *chunkA = reinterpret_cast<const half *>(a) +
                             (size_t)row * size_k;
        half *chunkC = reinterpret_cast<half *>(c) + (size_t)row * size_n;
        kernel<<<sms, threads, maxShared, cudaStreamPerThread>>>(
            reinterpret_cast<const int4 *>(chunkA),
            reinterpret_cast<const int4 *>(b_q_weight),
            reinterpret_cast<int4 *>(chunkC),
            reinterpret_cast<int4 *>(GetCTmp(dev).ptr),
            nullptr, nullptr,
            reinterpret_cast<const int4 *>(b_scales),
            nullptr, nullptr, nullptr,
            numGroups, chunkM, size_n, size_k, size_k, workspace,
            /*has_bias=*/false, /*use_atomic_add=*/false,
            /*use_fp32_reduce=*/true, maxShared);
        if (cudaPeekAtLastError() != cudaSuccess) return false;

        row += chunkM;
        remaining -= chunkM;
    }
    return true;
}

extern "C" bool FastllmCudaMarlinHalfNVFP4Gemm(
        const void *a, const uint32_t *b_q_weight, const void *b_scales,
        const float *global_scale, void *c,
        int size_m, int size_n, int size_k, int *workspace, void *c_tmp) {
    if (!FastllmCudaMarlinNVFP4Supported(size_n, size_k) ||
        a == nullptr || b_q_weight == nullptr ||
        b_scales == nullptr || global_scale == nullptr || c == nullptr ||
        workspace == nullptr || size_m <= 0 || size_n <= 0 || size_k <= 0) {
        return false;
    }

    int device = 0, sms = 0, maxShared = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    cudaDeviceGetAttribute(&maxShared,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (sms <= 0 || maxShared <= 0) return false;

    int maxMBlock = size_m <= 8
        ? 8 : std::min(64, ((size_m + 15) / 16) * 16);
    size_t cTmpElems = (size_t)sms * maxMBlock * 256;
    float *cTmp = static_cast<float *>(c_tmp);
    if (cTmp == nullptr) {
        if (!EnsureCTmp(device, cTmpElems)) return false;
        cTmp = GetCTmp(device).ptr;
    }

    const int arch = DeviceArch();
    const int stages = arch == 75 ? 2 : 4;
    const int numGroups = size_k / 16;
    int row = 0;
    int remaining = size_m;
    const int maxParallel = size_n <= 4096 ? 128 : 16;
    while (remaining > 0) {
        int chunkM = remaining;
        if (remaining >= 64) {
            chunkM = std::min(remaining / 64, maxParallel) * 64;
        }

        int threadK = 0, threadN = 0;
        if (!SelectFp4Tile(chunkM, size_n, size_k, threadK, threadN)) {
            return false;
        }
        int threads = 0;
        KernelFn kernel = PickFp4Kernel(
            chunkM, threadK, threadN, chunkM <= 8, stages, threads);
        if (kernel == nullptr) return false;

        const half *chunkA = reinterpret_cast<const half *>(a) +
                             (size_t)row * size_k;
        half *chunkC = reinterpret_cast<half *>(c) +
                       (size_t)row * size_n;
        kernel<<<sms, threads, maxShared, cudaStreamPerThread>>>(
            reinterpret_cast<const int4 *>(chunkA),
            reinterpret_cast<const int4 *>(b_q_weight),
            reinterpret_cast<int4 *>(chunkC),
            reinterpret_cast<int4 *>(cTmp),
            nullptr, nullptr,
            reinterpret_cast<const int4 *>(b_scales), global_scale,
            nullptr, nullptr,
            numGroups, chunkM, size_n, size_k, size_k, workspace,
            /*has_bias=*/false, /*use_atomic_add=*/false,
            /*use_fp32_reduce=*/true, maxShared);
        if (cudaPeekAtLastError() != cudaSuccess) return false;

        row += chunkM;
        remaining -= chunkM;
    }
    return true;
}
