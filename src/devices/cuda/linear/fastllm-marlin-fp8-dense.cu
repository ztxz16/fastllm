/*
 * Dense Marlin W8A16 FP8 and W4A16 NVFP4 launchers for FastLLM.
 * Kernel body vendored from vLLM csrc/quantization/marlin (Apache-2.0).
 * FP8 uses four stages for the SM80/SM86 64x256x64 prefill tile and two
 * stages otherwise. NVFP4 on SM80+ uses four stages, matching
 * vLLM's ops.marlin_gemm(b_q_type=float4_e2m1f) dispatch.
 */

#include "fastllm-cuda.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <map>
#include <set>

#define MARLIN_NAMESPACE_NAME fastllm_marlin_dense_fp8
#include "marlin_dense_fp8/kernel.h"
#include "marlin_dense_fp8/marlin_template.h"

namespace {

using KernelFn = void (*)(MARLIN_KERNEL_PARAMS);

static bool DeviceOk(int *deviceArch = nullptr) {
#ifdef CUDA_NO_TENSOR_CORE
    if (deviceArch != nullptr) *deviceArch = 0;
    return false;
#else
    int dev = 0, major = 0, minor = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess)
        return false;
    if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev) != cudaSuccess)
        return false;
    const int arch = major * 10 + minor;
    if (deviceArch != nullptr) *deviceArch = arch;
    return arch >= 75;
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

static bool IsAmpereFp8Device(int deviceArch) {
    return deviceArch == 80 || deviceArch == 86;
}

// Explicit FE4M3 group128 (group_blocks=8) + channelwise (-1).
#define RET_K_STAGES(STAGES, THREADS, TM, TN, TK, M8, GB)                        \
    return MARLIN_NAMESPACE_NAME::Marlin<                                       \
        vllm::kFloat16.id(), vllm::kFE4M3fn.id(), vllm::kFloat16.id(),         \
        vllm::kFloat16.id(), (THREADS), (TM), (TN), (TK), (M8), (STAGES), (GB), false>
#define RET_K(THREADS, TM, TN, TK, M8, GB)                                      \
    RET_K_STAGES(2, THREADS, TM, TN, TK, M8, GB)

static KernelFn PickKernel(int sizeM, int threadK, int threadN, int groupBlocks,
                           bool m8, int deviceArch, int &threads) {
    threads = 0;
    const int tm = (m8 || sizeM <= 8) ? 1 : std::min(4, (sizeM + 15) / 16);
    const bool useM8 = m8 || sizeM <= 8;

    if (groupBlocks != 8 && groupBlocks != -1) return nullptr;

    if (!useM8 && sizeM >= 64 && IsAmpereFp8Device(deviceArch) &&
        threadK == 64 && threadN == 256) {
        // M64/N256/K64, 256 threads, four asynchronous copy stages.
        threads = 256;
        if (groupBlocks == 8) RET_K_STAGES(4, 256, 4, 16, 4, false, 8);
        RET_K_STAGES(4, 256, 4, 16, 4, false, -1);
    }

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
#undef RET_K_STAGES

// Explicit FE2M1 + special FE4M3 scale, group16 (group_blocks=1).
// Only SM75 selects stages=2; newer architectures select stages=4.
// The SM75 64x256x64 prefill tile specializes the fixed dense reduction
// flags to remove unused bias/atomic paths and register spills. Bias
// remains in the outer epilogue. Small-M and SM80+ keep their pipeline.
#define RET_FP4(STAGES, THREADS, TM, TN, TK, M8)                            \
    return MARLIN_NAMESPACE_NAME::Marlin<                                   \
        vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(),       \
        vllm::kFE4M3fn.id(), (THREADS), (TM), (TN), (TK), (M8),             \
        (STAGES), 1, false,                                                 \
        ((STAGES) == 2 && (THREADS) == 256 && (TM) == 4 &&                  \
         (TN) == 16 && (TK) == 4 && !(M8))>
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
        // These dense calls always have no bias/atomics and use FP32 reduction.
        // Specialize those fixed flags without changing the tile, weight layout,
        // or scratch requirements. Keep other architectures on their old path.
        if (stages == 4 && DeviceArch() == 120) {
#define RET_FP4_DENSE(THREADS, TN, TK) \
            return MARLIN_NAMESPACE_NAME::Marlin< \
                vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(), \
                vllm::kFE4M3fn.id(), THREADS, 1, TN, TK, true, 4, 1, false, true>
            if (threadK == 128 && threadN == 128) {
                threads = 256; RET_FP4_DENSE(256, 8, 8);
            }
            if (threadK == 64 && threadN == 128) {
                threads = 128; RET_FP4_DENSE(128, 8, 4);
            }
            if (threadK == 128 && threadN == 64) {
                threads = 128; RET_FP4_DENSE(128, 4, 8);
            }
#undef RET_FP4_DENSE
        }
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
// Only instantiate the m<=8 residual variants. The original kernel table
// and its prefill / non-residual dispatch remain unchanged.
static KernelFn PickFp4AddKernel(int threadK, int threadN, int stages, int &threads) {
#define ADD_KERNEL(STAGES, THREADS, TN, TK) \
    MARLIN_NAMESPACE_NAME::Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), \
        vllm::kFloat16.id(), vllm::kFE4M3fn.id(), THREADS, 1, TN, TK, true, \
        STAGES, 1, false, true, true>
#define PICK_ADD(THREADS, TN, TK) \
    do { threads = THREADS; return stages == 2 ? ADD_KERNEL(2, THREADS, TN, TK) \
                                              : ADD_KERNEL(4, THREADS, TN, TK); } while (0)
    if (stages != 2 && stages != 4) return nullptr;
    if (threadK == 128 && threadN == 128) PICK_ADD(256, 8, 8);
    if (threadK == 64 && threadN == 128) PICK_ADD(128, 8, 4);
    if (threadK == 128 && threadN == 64) PICK_ADD(128, 4, 8);
    return nullptr;
#undef PICK_ADD
#undef ADD_KERNEL
}

static KernelFn PickFp4SwigluKernel(int stages) {
#define FP4_SWIGLU(STAGES) \
    MARLIN_NAMESPACE_NAME::Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), \
        vllm::kFloat16.id(), vllm::kFE4M3fn.id(), 256, 1, 8, 8, true, \
        STAGES, 1, false, true, false, true>
    return stages == 2 ? FP4_SWIGLU(2) : FP4_SWIGLU(4);
#undef FP4_SWIGLU
}

#undef RET_FP4_FOR_ARCH
#undef RET_FP4

static bool SelectTile(int sizeM, int sizeN, int sizeK, int deviceArch,
                       int &threadK, int &threadN) {
    // Prefer the wider prefill tile on SM75 (two copy stages) and
    // SM80/SM86 (four stages) only when M/N/K amortize its overhead.
    if ((deviceArch == 75 || IsAmpereFp8Device(deviceArch)) &&
        sizeM >= 256 && sizeN >= 4096 && sizeK >= 1024 &&
        sizeK % 64 == 0 && sizeN % 256 == 0) {
        threadK = 64;
        threadN = 256;
        return true;
    }
    static const int smallM[][2] = {{64, 128}, {128, 64}, {128, 128}};
    // On Turing, the 256-thread tile amortizes the fixed M8 reduction cost
    // better than either 128-thread tile. Keep the established priority on
    // newer architectures until it is benchmarked there.
    static const int smallMTuring[][2] = {{128, 128}, {64, 128}, {128, 64}};
    static const int largeM[][2] = {{64, 128}, {128, 64}, {64, 256}, {128, 128}};
    const int (*cfgs)[2] = sizeM <= 8
        ? (deviceArch == 75 ? smallMTuring : smallM) : largeM;
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
    // A narrower N tile gives small-M matrices more independent output
    // stripes and less cross-CTA reduction. Bound this policy to the tested
    // SM120 shape range; wide matrices and other architectures keep their
    // original priority. The limit scales with the device's SM count.
    if (sizeM >= 1 && sizeM <= 8 && sizeN >= 2048 && sizeN % 64 == 0 &&
        sizeK >= 1024 && sizeK <= 32768 && sizeK % 128 == 0 && DeviceArch() == 120) {
        int device = 0, sms = 0;
        if (cudaGetDevice(&device) == cudaSuccess &&
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess &&
            sizeN / 64 <= sms) {
            threadK = 128;
            threadN = 64;
            return true;
        }
    }
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

// The original launch reserves all 64 KiB of SM75 shared memory for one
// resident CTA. These measured M=1..8 shapes can use two resident CTAs without
// changing the weight layout or growing the per-weight reduction scratch.
// Enable tuning by default only for the measured 68-SM Turing device and
// shapes. Other shapes, prefill, and architectures keep their dispatch;
// FASTLLM_CUDA_NVFP4_SM75_DECODE_TUNE=0 restores the untuned launch.
static int Sm75Nvfp4DecodeTuneMode() {
    static const int mode = []() {
        const char *value = std::getenv("FASTLLM_CUDA_NVFP4_SM75_DECODE_TUNE");
        if (value == nullptr) return 3;
        if (!std::strcmp(value, "1")) return 3;
        if (!std::strcmp(value, "linear")) return 1;
        if (!std::strcmp(value, "swiglu")) return 2;
        return 0;
    }();
    return mode;
}

static bool Sm75Nvfp4DecodeTuneShape(int sizeM, int sizeN, int sizeK, bool swiglu) {
    if (sizeM < 1 || sizeM > 8 ||
        !(Sm75Nvfp4DecodeTuneMode() & (swiglu ? 2 : 1))) return false;
    return swiglu ? sizeN == 17408 && sizeK == 5120 :
        sizeN == 5120 && (sizeK == 3072 || sizeK == 8704);
}

static bool TuneSm75Nvfp4DecodeLaunch(
        int device, int arch, int sizeM, int sizeN, int sizeK,
        int sms, bool swiglu, KernelFn &kernel, int &threads,
        int &blocks, int &shared) {
    if (arch != 75 || sms != 68 ||
        !Sm75Nvfp4DecodeTuneShape(sizeM, sizeN, sizeK, swiglu)) return false;

    int tunedThreads = threads;
    KernelFn tunedKernel = swiglu ? kernel :
        PickFp4Kernel(sizeM, 128, 64, true, 2, tunedThreads);
    if (tunedKernel == nullptr) return false;
    constexpr int tunedShared = 32 * 1024;
    // The paired N128/K128 epilogue needs 22 KiB; the N64/K128 projection
    // needs 13 KiB. 32 KiB admits two CTAs while retaining a safety margin.
    // At two CTAs/SM, N<=128 uses at most the existing sms*8*256 FP32
    // scratch elements and fewer than the existing sms*4 lock entries.
    static thread_local std::map<std::pair<int, KernelFn>, int> residency;
    const auto key = std::make_pair(device, tunedKernel);
    auto it = residency.find(key);
    if (it == residency.end()) {
        int active = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active, tunedKernel, tunedThreads, tunedShared) != cudaSuccess) {
            return false;
        }
        it = residency.emplace(key, active).first;
    }
    // Marlin uses inter-CTA spin barriers. Never launch more persistent
    // CTAs than can be resident together on the device.
    if (it->second < 2) return false;
    kernel = tunedKernel;
    threads = tunedThreads;
    blocks = 2 * sms;
    shared = tunedShared;
    static thread_local std::set<std::pair<int, std::pair<int, int>>> reported;
    if (reported.emplace(device, std::make_pair(sizeM, swiglu ? -sizeK : sizeK)).second) {
        printf("[Fastllm] SM75 NVFP4 decode tuned GPU %d: M=%d N=%d K=%d "
               "swiglu=%d blocks=%d threads=%d shared=%d.\n",
               device, sizeM, sizeN, sizeK, swiglu ? 1 : 0,
               blocks, threads, shared);
    }
    return true;
}

static bool PrepareKernels(int device) {
    int prev = -1;
    cudaGetDevice(&prev);
    if (prev != device && cudaSetDevice(device) != cudaSuccess) return false;

    const int deviceArch = DeviceArch();
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
                    KernelFn k = PickKernel(m, t[0], t[1], gb, m <= 8,
                                            deviceArch, threads);
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
    // A captured graph keeps this address even when later eager prefills use
    // larger M tiles. Reserve the largest reduction tile on first use so the
    // cached buffer never moves underneath an existing FP8/NVFP4 graph.
    int sms = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) !=
            cudaSuccess || sms <= 0) return false;
    elems = std::max(elems, (size_t)sms * 64 * 256);
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

extern "C" bool FastllmCudaMarlinNVFP4DecodeTuneEnabled(
        int size_m, int size_n, int size_k, bool swiglu) {
    if (!Sm75Nvfp4DecodeTuneShape(size_m, size_n, size_k, swiglu)) return false;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    static thread_local std::map<int, bool> devices;
    auto it = devices.find(device);
    if (it == devices.end()) {
        int arch = 0, sms = 0;
        bool supported = DeviceOk(&arch) && arch == 75 &&
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess &&
            sms == 68;
        it = devices.emplace(device, supported).first;
    }
    return it->second;
}

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
    int deviceArch = 0;
    if (!DeviceOk(&deviceArch) || size_m <= 0 || size_n <= 0 || size_k <= 0) {
        return false;
    }
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
        if (!SelectTile(chunkM, size_n, size_k, deviceArch,
                        threadK, threadN)) {
            return false;
        }

        int threads = 0;
        bool m8 = chunkM <= 8;
        KernelFn kernel = PickKernel(chunkM, threadK, threadN, groupBlocks,
                                     m8, deviceArch, threads);
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

        int blocks = sms;
        int launchShared = maxShared;
        if (size_m <= 8) {
            TuneSm75Nvfp4DecodeLaunch(
                device, arch, chunkM, size_n, size_k, sms, false,
                kernel, threads, blocks, launchShared);
        }

        const half *chunkA = reinterpret_cast<const half *>(a) +
                             (size_t)row * size_k;
        half *chunkC = reinterpret_cast<half *>(c) +
                       (size_t)row * size_n;
        kernel<<<blocks, threads, launchShared, cudaStreamPerThread>>>(
            reinterpret_cast<const int4 *>(chunkA),
            reinterpret_cast<const int4 *>(b_q_weight),
            reinterpret_cast<int4 *>(chunkC),
            reinterpret_cast<int4 *>(cTmp),
            nullptr, nullptr,
            reinterpret_cast<const int4 *>(b_scales), global_scale,
            nullptr, nullptr,
            numGroups, chunkM, size_n, size_k, size_k, workspace,
            /*has_bias=*/false, /*use_atomic_add=*/false,
            /*use_fp32_reduce=*/true, launchShared);
        if (cudaPeekAtLastError() != cudaSuccess) return false;

        row += chunkM;
        remaining -= chunkM;
    }
    return true;
}

extern "C" bool FastllmCudaMarlinNVFP4AddSupported(int size_n, int size_k) {
    if (!DeviceOk() || size_n <= 0 || size_k <= 0 || size_n % 128 || size_k % 128)
        return false;
    int device = 0, maxShared = 0, threadK = 0, threadN = 0, threads = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        !SelectFp4Tile(8, size_n, size_k, threadK, threadN)) return false;
    KernelFn kernel = PickFp4AddKernel(threadK, threadN, DeviceArch() == 75 ? 2 : 4, threads);
    if (!kernel) return false;
    static thread_local std::map<std::pair<int, KernelFn>, bool> ready;
    const auto key = std::make_pair(device, kernel);
    auto it = ready.find(key);
    if (it != ready.end()) return it->second;
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) return false;
    cudaFuncAttributes attr{};
    bool ok = cudaDeviceGetAttribute(&maxShared, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                     device) == cudaSuccess && maxShared > 0 &&
        cudaFuncGetAttributes(&attr, kernel) == cudaSuccess && attr.maxThreadsPerBlock >= threads &&
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             maxShared) == cudaSuccess;
    if (!ok) cudaGetLastError();
    ready.emplace(key, ok);
    return ok;
}

extern "C" bool FastllmCudaMarlinHalfNVFP4Add(
        const void *a, const uint32_t *b_q_weight, const void *b_scales,
        const float *global_scale, void *c, int size_m, int size_n, int size_k,
        int *workspace, void *c_tmp) {
    if (size_m < 2 || size_m > 8 || !a || !b_q_weight || !b_scales ||
        !global_scale || !c || !workspace || !c_tmp ||
        !FastllmCudaMarlinNVFP4AddSupported(size_n, size_k)) return false;
    int device = 0, sms = 0, maxShared = 0, threadK = 0, threadN = 0, threads = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    cudaDeviceGetAttribute(&maxShared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (sms <= 0 || maxShared <= 0 ||
        !SelectFp4Tile(size_m, size_n, size_k, threadK, threadN)) return false;
    KernelFn kernel = PickFp4AddKernel(threadK, threadN, DeviceArch() == 75 ? 2 : 4, threads);
    if (!kernel) return false;
    kernel<<<sms, threads, maxShared, cudaStreamPerThread>>>(
        reinterpret_cast<const int4*>(a), reinterpret_cast<const int4*>(b_q_weight),
        reinterpret_cast<int4*>(c), reinterpret_cast<int4*>(c_tmp), nullptr, nullptr,
        reinterpret_cast<const int4*>(b_scales), global_scale, nullptr, nullptr,
        size_k / 16, size_m, size_n, size_k, size_k, workspace,
        false, false, true, maxShared);
    // After a launch, failures are errors: never retry by adding residual twice.
    if (cudaGetLastError() != cudaSuccess) throw "NVFP4 residual GEMM launch failed";
    return true;
}

extern "C" bool FastllmCudaMarlinNVFP4SwigluSupported(int size_n, int size_k) {
    const int arch = DeviceArch();
    if (!DeviceOk() || (arch != 75 && arch != 120) || size_n < 256 || size_n % 256 ||
        size_k < 128 || size_k % 128) return false;
    // Pair gate/up tiles only in the measured wide-matrix range on SM120.
    // Small matrices keep their existing narrow tile and separate activation.
    if (arch == 120 && (size_n < 16384 || size_n > 65536 ||
                       size_k < 4096 || size_k > 32768)) return false;
    int device = 0, maxShared = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    static thread_local std::map<int, bool> ready;
    auto it = ready.find(device);
    if (it != ready.end()) return it->second;
    cudaStreamCaptureStatus capture;
    if (cudaStreamIsCapturing(cudaStreamPerThread, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) return false;
    cudaFuncAttributes attr{};
    KernelFn kernel = PickFp4SwigluKernel(DeviceArch() == 75 ? 2 : 4);
    bool ok = cudaDeviceGetAttribute(&maxShared, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                     device) == cudaSuccess && maxShared > 0 &&
        cudaFuncGetAttributes(&attr, kernel) == cudaSuccess && attr.maxThreadsPerBlock >= 256 &&
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maxShared) == cudaSuccess;
    if (!ok) cudaGetLastError();
    ready.emplace(device, ok);
    return ok;
}

extern "C" bool FastllmCudaMarlinHalfNVFP4Swiglu(
        const void *a, const uint32_t *b_q_weight, const void *b_scales,
        const float *global_scale, void *c, int size_m, int size_n, int size_k,
        int *workspace, void *c_tmp) {
    if (size_m < 1 || size_m > 8 ||
        (size_m == 1 && !FastllmCudaMarlinNVFP4DecodeTuneEnabled(1, size_n, size_k, true)) ||
        !a || !b_q_weight || !b_scales ||
        !global_scale || !c || !workspace || !c_tmp ||
        !FastllmCudaMarlinNVFP4SwigluSupported(size_n, size_k)) return false;
    int device = 0, sms = 0, maxShared = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    cudaDeviceGetAttribute(&maxShared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (sms <= 0 || maxShared <= 0) return false;
    KernelFn kernel = PickFp4SwigluKernel(DeviceArch() == 75 ? 2 : 4);
    int threads = 256, blocks = sms, launchShared = maxShared;
    TuneSm75Nvfp4DecodeLaunch(
        device, DeviceArch(), size_m, size_n, size_k, sms, true,
        kernel, threads, blocks, launchShared);
    kernel<<<blocks, threads, launchShared, cudaStreamPerThread>>>(
        reinterpret_cast<const int4*>(a), reinterpret_cast<const int4*>(b_q_weight),
        reinterpret_cast<int4*>(c), reinterpret_cast<int4*>(c_tmp), nullptr, nullptr,
        reinterpret_cast<const int4*>(b_scales), global_scale, nullptr, nullptr,
        size_k / 16, size_m, size_n, size_k, size_k, workspace,
        false, false, true, launchShared);
    if (cudaGetLastError() != cudaSuccess) throw "NVFP4 SwiGLU GEMM launch failed";
    return true;
}
