//
// DeepSeek-V4.1 系列模型实现。架构说明见 include/models/deepseekv41.h。
//
// 本文件保留共享模型数学、注意力、KV 与 Graph 状态；MoE 执行策略独立于此：
//   * deepseekv41_gpu.cpp：GPU 专家执行；deepseekv41_hybrid.cpp：CPU/NUMA 混合执行。
//   * 注意力、Hyper-Connections、indexer 等在执行器选择的设备上运行（通常是 GPU）；
//   * 路由专家通过 MergeMOEBlock 交给 MoE 设备（cpu / numa / cuda）；
//   * Engram 哈希表以 FP8 + UE8M0 scale 常驻 CPU 内存或通过磁盘 Embedding 按行读取，
//     查表在 CPU 完成，后续 wkv 投影与门控在 GPU 完成。
//

#include "deepseekv41.h"
#include "deepseekv41_execution.h"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-v41-full-decode.h"
#endif
#include "utils/hybrid_moe.h"
#include <optional>

#include "baseblock.h"
#include "executor.h"
#include "devices/cpu/deepseekv41-reference-math.h"
#include "utils.h"
#include "json11.hpp"

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#include "fastllm-v41-decode.cuh"
#include "devices/cuda/fastllm-v41-engram.h"
#include "deepseekv41-reference.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>

#if !defined(_WIN32) && !defined(_WIN64)
#include <fcntl.h>
#include <pwd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace fastllm {
    namespace {
#ifdef USE_CUDA
        // A one-token slice of a dense [1, tokens, ...] activation is contiguous
        // on every TP rank. Borrow it for this forward instead of copying it.
        // The forward scope drains all ranks before releasing the parent.
        bool V41DecodeSliceView(Data &dst, const Data &src, int offset,
                                const std::vector<int> &devices) {
            if (devices.empty() || !src.multiDeviceData || src.dims.size() < 3 || src.dims[0] != 1 ||
                offset < 0 || offset >= src.dims[1] || src.tpAxis == 1 ||
                (!src.IsTensorParallelReplicated() && !src.IsTensorParallelSharded())) return false;
            auto sameSliceShape = [](const std::vector<int> &view, const std::vector<int> &full) {
                if (view.size() != full.size() || view.size() < 3 || view[1] != 1) return false;
                for (size_t axis = 0; axis < view.size(); ++axis)
                    if (axis != 1 && view[axis] != full[axis]) return false;
                return true;
            };
            bool cachedView = dst.multiDeviceData && dst.multiDeviceDatas.size() == devices.size() &&
                dst.dataDevice == DataDevice::CUDA && dst.dataType == src.dataType &&
                dst.dataDeviceIds == devices && dst.tpLayout == src.tpLayout && dst.tpAxis == src.tpAxis &&
                dst.tpRanges == src.tpRanges && sameSliceShape(dst.dims, src.dims) &&
                dst.tpGlobalDims == dst.dims;
            for (int device : devices) {
                auto found = src.multiDeviceDatas.find(device);
                if (found == src.multiDeviceDatas.end() || !found->second) return false;
                const Data &local = *found->second;
                if (!local.cudaData || local.dataDevice != DataDevice::CUDA || local.isKVCache ||
                    local.isGGUFData || local.dims.size() != src.dims.size() || local.dims[0] != 1 ||
                    local.dims[1] != src.dims[1] || local.strides.size() != local.dims.size() ||
                    (local.dataType != DataType::BFLOAT16 && local.dataType != DataType::FLOAT16 &&
                     local.dataType != DataType::FLOAT32 && local.dataType != DataType::INT32)) return false;
                uint64_t stride = 1;
                for (int axis = (int)local.dims.size() - 1; axis >= 0; --axis) {
                    if (local.strides[axis] != stride) return false;
                    stride *= local.dims[axis];
                }
                if (cachedView) {
                    auto previous = dst.multiDeviceDatas.find(device);
                    const Data *view = previous == dst.multiDeviceDatas.end() ? nullptr : previous->second;
                    cachedView = view && view->isFake && view->dataType == local.dataType &&
                        view->dataDevice == local.dataDevice && view->dataDeviceIds == local.dataDeviceIds &&
                        view->cudaData == (uint8_t*)local.cudaData + (size_t)offset * local.strides[1] * local.unitSize &&
                        sameSliceShape(view->dims, local.dims) && view->expansionSize == local.strides[1];
                }
            }
            // Graph boundaries retain their allocations between layers and
            // tokens. A matching borrowed row needs no metadata reconstruction.
            if (cachedView) return true;
            bool reusable = dst.multiDeviceData && dst.multiDeviceDatas.size() == devices.size();
            for (int device : devices) {
                auto found = dst.multiDeviceDatas.find(device);
                reusable &= found != dst.multiDeviceDatas.end() && found->second && found->second->isFake;
            }
            if (!reusable) {
                V41ResetMultiDevice(dst);
                dst.FreeSpace();
            }
            dst.dataType = src.dataType;
            dst.UpdateUnitSize();
            auto dims = src.dims; dims[1] = 1;
            dst.Resize(dims);
            dst.dataDevice = DataDevice::CUDA;
            dst.dataDeviceIds = devices;
            dst.multiDeviceData = true;
            dst.tpLayout = src.tpLayout; dst.tpAxis = src.tpAxis;
            dst.tpGlobalDims = dims; dst.tpRanges = src.tpRanges;
            for (int device : devices) {
                const Data &source = *src.multiDeviceDatas.at(device);
                Data *&view = dst.multiDeviceDatas[device];
                if (!view) view = new Data();
                view->FakeFrom(source, (size_t)offset * source.strides[1] * source.unitSize);
                auto localDims = source.dims; localDims[1] = 1;
                view->expansionDims.clear();
                view->Resize(localDims);
                // TP validation checks the accessible capacity even for a
                // borrowed view. Publish only this row, never the parent's
                // remaining capacity (which would permit overwriting siblings).
                view->expansionDims = localDims;
                view->expansionSize = view->Count(0);
                view->expansionBytes = view->GetBytes();
            }
            return true;
        }
#endif

        // ---------------- 配置读取 ----------------

        bool V41HasKey(const WeightMap &weight, const std::string &key) {
            return weight.dicts.find(key) != weight.dicts.end();
        }

        int V41DictInt(const WeightMap &weight, const std::string &key, int fallback) {
            auto it = weight.dicts.find(key);
            return it == weight.dicts.end() ? fallback : atoi(it->second.c_str());
        }

        float V41DictFloat(const WeightMap &weight, const std::string &key, float fallback) {
            auto it = weight.dicts.find(key);
            return it == weight.dicts.end() ? fallback : (float)atof(it->second.c_str());
        }

        std::vector<int64_t> V41DictInt64Array(const WeightMap &weight, const std::string &key) {
            std::vector<int64_t> ret;
            auto it = weight.dicts.find(key);
            if (it == weight.dicts.end()) {
                return ret;
            }
            std::string err;
            auto json = json11::Json::parse(it->second, err);
            if (!err.empty() || !json.is_array()) {
                return ret;
            }
            for (auto &item : json.array_items()) {
                ret.push_back((int64_t)item.number_value());
            }
            return ret;
        }

        std::vector<int> V41DictIntArray(const WeightMap &weight, const std::string &key) {
            std::vector<int> ret;
            for (int64_t v : V41DictInt64Array(weight, key)) {
                ret.push_back((int)v);
            }
            return ret;
        }

        bool V41Contains(const std::vector<int> &values, int v) {
            return std::find(values.begin(), values.end(), v) != values.end();
        }

        bool V41StartsWith(const std::string &s, const std::string &prefix) {
            return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
        }

        bool V41EndsWith(const std::string &s, const std::string &suffix) {
            return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
        }

        // 默认开启的开关：显式设成 0 才关掉
        bool V41EnvFlagOn(const char *name) {
            const char *v = std::getenv(name);
            return v == nullptr || v[0] == '\0' || strcmp(v, "0") != 0;
        }

        // ---------------- Engram 计时 ----------------
        // FASTLLM_DSV41_ENGRAM_PROFILE=1 累计统计，=2 额外逐次打印。
        // 分成：hash（算行号）、prep（准备输出 Data）、gather（读表 + FP8→BF16）、
        // wkv（投影）、apply（门控写回）。
        // wkv / apply 落在 GPU 上且是异步下发的，要拿到真实耗时必须同时设 FASTLLM_CUDA_SYNC=1，
        // 否则这两项只反映 kernel launch 的时间。
        // FASTLLM_DSV41_ENGRAM_PROFILE_EVERY=N 控制每累计 N 次打印一行（默认 64，0 表示只在退出时打印）。

        double V41NowMs() {
            return std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now().time_since_epoch()).count();
        }

        struct V41EngramStat {
            uint64_t calls = 0;
            uint64_t tokens = 0;
            double hash = 0.0, prep = 0.0, gather = 0.0, wkv = 0.0, apply = 0.0, wait = 0.0;
        };

        struct V41EngramProfiler {
            int level = 0;
            uint64_t reportEvery = 64;
            std::mutex mutex;
            V41EngramStat decode, prefill;
            uint64_t sinceReport = 0;

            V41EngramProfiler() {
                const char *v = std::getenv("FASTLLM_DSV41_ENGRAM_PROFILE");
                if (v != nullptr && v[0] != '\0' && strcmp(v, "0") != 0) {
                    level = atoi(v);
                    if (level <= 0) {
                        level = 1;
                    }
                }
                const char *e = std::getenv("FASTLLM_DSV41_ENGRAM_PROFILE_EVERY");
                if (e != nullptr && e[0] != '\0') {
                    long long n = atoll(e);
                    reportEvery = n > 0 ? (uint64_t)n : 0;
                }
            }

            ~V41EngramProfiler() {
                if (level > 0) {
                    Report("汇总");
                }
            }

            void Add(int layer, int tokens, double hash, double prep, double gather, double wkv, double apply, double wait) {
                std::lock_guard<std::mutex> guard(mutex);
                V41EngramStat &s = tokens > 1 ? prefill : decode;
                s.calls++;
                s.tokens += (uint64_t)tokens;
                s.hash += hash;
                s.prep += prep;
                s.gather += gather;
                s.wkv += wkv;
                s.apply += apply;
                s.wait += wait;
                if (level >= 2) {
                    // 服务端的 stdout 是重定向到文件的块缓冲，不 flush 的话最后一批
                    // 逐次记录会一直留在缓冲区里（进程被 kill 时直接丢掉）
                    printf("[Engram] layer %d tokens %d | hash %.3f prep %.3f gather %.3f wkv %.3f apply %.3f wait %.3f ms\n",
                           layer, tokens, hash, prep, gather, wkv, apply, wait);
                    fflush(stdout);
                }
                if (reportEvery > 0 && ++sinceReport >= reportEvery) {
                    sinceReport = 0;
                    ReportLocked("进行中");
                }
            }

            void Report(const char *tag) {
                std::lock_guard<std::mutex> guard(mutex);
                ReportLocked(tag);
            }

            void ReportLocked(const char *tag) {
                Line(tag, "decode ", decode);
                Line(tag, "prefill", prefill);
                fflush(stdout);
            }

            static void Line(const char *tag, const char *name, const V41EngramStat &s) {
                if (s.calls == 0) {
                    return;
                }
                double n = (double)s.calls;
                double sum = s.hash + s.prep + s.gather + s.wkv + s.apply;
                printf("[Engram %s] %s %llu 次 / %llu token：每次 hash %.3f + prep %.3f + gather %.3f + wkv %.3f "
                       "+ apply %.3f = %.3f ms（等待预取 %.3f，累计 %.1f ms）\n",
                       tag, name, (unsigned long long)s.calls, (unsigned long long)s.tokens,
                       s.hash / n, s.prep / n, s.gather / n, s.wkv / n, s.apply / n, sum / n, s.wait / n, sum);
            }
        };

        V41EngramProfiler &V41Profiler() {
            static V41EngramProfiler profiler;
            return profiler;
        }

        // ---------------- Engram 表的内存访问提示 ----------------
        // FASTLLM_DSV41_ENGRAM_MADVISE=random / hugepage / both（默认 off，行为不变）。
        //   random   ：对常驻表设置 MADV_RANDOM 提示。
        //   hugepage ：常驻模式下改用匿名 mmap + MADV_HUGEPAGE 分配 100 GB 的表，
        //              4 KB 页要 2500 万个 PTE，随机查表几乎每次都 TLB miss；
        //              顺带省掉 std::vector 的 100 GB 清零，加载也更快。

        struct V41EngramMadviseCfg {
            bool random = false;
            bool hugePage = false;
        };

        V41EngramMadviseCfg V41EngramMadvise() {
            V41EngramMadviseCfg cfg;
            const char *v = std::getenv("FASTLLM_DSV41_ENGRAM_MADVISE");
            if (v == nullptr || v[0] == '\0') {
                return cfg;
            }
            std::string s = v;
            cfg.random = s.find("random") != std::string::npos || s.find("both") != std::string::npos ||
                         s == "1" || s.find("all") != std::string::npos;
            cfg.hugePage = s.find("huge") != std::string::npos || s.find("both") != std::string::npos ||
                           s.find("all") != std::string::npos;
            return cfg;
        }

        void V41MadviseRandom(const void *ptr, uint64_t bytes) {
#if !defined(_WIN32) && !defined(_WIN64) && defined(MADV_RANDOM)
            if (ptr == nullptr || bytes == 0) {
                return;
            }
            long pageSize = sysconf(_SC_PAGESIZE);
            uintptr_t start = (uintptr_t)ptr / pageSize * pageSize;
            size_t len = (size_t)(bytes + ((uintptr_t)ptr - start));
            madvise((void*)start, len, MADV_RANDOM);
#endif
        }

        // 匿名大页分配：常驻表用它替代 std::vector，可以在填充之前打上 MADV_HUGEPAGE。
        bool V41AllocAnon(uint64_t bytes, bool hugePage, void *&mapping, size_t &mapLen, uint8_t *&ptr) {
#if defined(_WIN32) || defined(_WIN64)
            return false;
#else
            void *p = mmap(nullptr, (size_t)bytes, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p == MAP_FAILED) {
                return false;
            }
#ifdef MADV_HUGEPAGE
            if (hugePage) {
                madvise(p, (size_t)bytes, MADV_HUGEPAGE);
            }
#endif
            mapping = p;
            mapLen = (size_t)bytes;
            ptr = (uint8_t*)p;
            return true;
#endif
        }

        // ---------------- 算子封装 ----------------

        // indexer 分数矩阵 [token, m] 在长上下文下会非常大（1M 上下文的 ratio-1 层每个 token
        // 有 1M 个候选，4096 token 的分块就是 16 GB）。按固定字节预算切 token 维，
        // 让峰值显存与上下文长度解耦。FASTLLM_DSV41_INDEX_SCORE_MB 可调（默认 128 MB）。
        int V41IndexScoreChunk(int segLen, int m) {
            if (segLen <= 1 || m <= 0) {
                return segLen;
            }
            static int forced = -2;
            if (forced == -2) {
                const char *env = std::getenv("FASTLLM_DSV41_INDEX_CHUNK");
                forced = env != nullptr && env[0] != '\0' ? atoi(env) : -1;
            }
            if (forced > 0) {
                return std::min(segLen, forced);
            }
            static size_t budget = 0;
            if (budget == 0) {
                const char *env = std::getenv("FASTLLM_DSV41_INDEX_SCORE_MB");
                long mb = env != nullptr && env[0] != '\0' ? atol(env) : 0;
                if (mb <= 0) {
                    mb = 128;
                }
                budget = (size_t)mb * 1024 * 1024;
            }
            // 分数矩阵之外，候选块打分与掩码另外约占 m / blockSize 的量级，留 1.5 倍余量
            const size_t perToken = (size_t)m * sizeof(float) * 3 / 2;
            const size_t chunk = budget / std::max<size_t>(perToken, 1);
            if (chunk >= (size_t)segLen) {
                return segLen;
            }
            int c = (int)std::max<size_t>(chunk, 1);
            if (c > 64) {
                c = (c / 64) * 64;
            }
            return c;
        }

        Executor &V41Executor() {
            return *((Executor*)GetExecutor());
        }

        bool V41DeviceMapUsesMultiCuda(const std::map<std::string, int> &deviceMap) {
            for (const auto &it : deviceMap) {
                if (V41DeviceSpecUsesType(it.first, "multicuda")) {
                    return true;
                }
            }
            return false;
        }

        // 从 deviceMap 的 key（如 "multicuda:0,1"）直接数出 TP rank 数。
        // InitParams 阶段执行器还没设置过 multicuda 设备表，只能这样拿。
        int V41MultiCudaRankCount(const std::map<std::string, int> &deviceMap) {
            for (const auto &it : deviceMap) {
                if (!V41DeviceSpecUsesType(it.first, "multicuda")) {
                    continue;
                }
                size_t pos = it.first.find(':');
                if (pos == std::string::npos) {
#ifdef USE_CUDA
                    return std::max(1, FastllmCudaGetDeviceCount());
#else
                    return 1;
#endif
                }
                int count = 0;
                std::string spec = it.first.substr(pos + 1);
                size_t start = 0;
                while (start <= spec.size()) {
                    size_t end = spec.find(',', start);
                    std::string item = spec.substr(start, end == std::string::npos ? std::string::npos : end - start);
                    if (!item.empty()) {
                        count++;
                    }
                    if (end == std::string::npos) {
                        break;
                    }
                    start = end + 1;
                }
                return std::max(1, count);
            }
            return 1;
        }

        // 返回当前生效的多卡设备列表；不是多卡张量并行时返回空
        std::vector<int> V41TpDevices(const std::map<std::string, int> &deviceMap) {
            std::vector<int> devices;
#ifdef USE_CUDA
            if (!V41DeviceMapUsesMultiCuda(deviceMap)) {
                return devices;
            }
            // FastllmGetMulticudaDeviceAndRatio 读的是"上一个跑过的 multicuda 算子"
            // 发布的全局设备表，前向开始时可能还没被写过；先按执行器里解析好的
            // deviceIds 取，取不到再退回全局表。
            devices = V41Executor().GetDeviceIds("multicuda");
            if (devices.size() <= 1) {
                std::map<int, int> ratios;
                devices.clear();
                FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
            }
            if (devices.size() <= 1) {
                devices.clear();
            }
#endif
            return devices;
        }

#ifdef USE_CUDA

        // 路由分数变换与专家选择在每张卡上各算一份：路由结果必须逐位一致，
        // 否则两张卡的共享专家 / all-reduce 会对不上。
        bool V41RouteScoreTransformTp(Data &logits, int mode, const std::vector<int> &devices) {
            if (devices.empty() || !logits.multiDeviceData ||
                !logits.IsTensorParallelReplicated()) {
                return FastllmCudaDeepSeekV4RouteScoreTransform(logits, mode);
            }
            std::vector<char> ok(devices.size(), 0);
            if (!MultiCudaRunDeviceCallbacks(devices, [&](int rank, int device) {
                    auto it = logits.multiDeviceDatas.find(device);
                    if (it != logits.multiDeviceDatas.end() && it->second != nullptr) {
                        ok[rank] = FastllmCudaDeepSeekV4RouteScoreTransform(*it->second, mode);
                    }
                })) {
                return false;
            }
            return std::all_of(ok.begin(), ok.end(), [](char v) { return v != 0; });
        }
#endif

        // 分配一个（可能是复制布局的）空张量
        void V41AllocLike(Data &dst, DataType type, const std::vector<int> &dims,
                          const Data &reference, const std::vector<int> &tpDevices) {
#ifdef USE_CUDA
            if (!tpDevices.empty() && reference.multiDeviceData &&
                reference.IsTensorParallelReplicated()) {
                // These outputs are fully overwritten on ordered TP streams.
                // Reuse capacity when the logical Top-K/candidate width changes.
                uint64_t required = 1;
                for (int dim : dims) required *= dim;
                bool reuse = dst.multiDeviceData && dst.IsTensorParallelReplicated() &&
                    !dst.isFake && dst.dataType == type &&
                    dst.dataDeviceIds == tpDevices && dst.multiDeviceDatas.size() == tpDevices.size();
                if (reuse) {
                    for (int device : tpDevices) {
                        auto it = dst.multiDeviceDatas.find(device);
                        if (it == dst.multiDeviceDatas.end() || !it->second ||
                            !it->second->cudaData || it->second->isFake ||
                            it->second->dataDevice != DataDevice::CUDA ||
                            it->second->dataType != type || it->second->expansionSize < required) {
                            reuse = false;
                            break;
                        }
                    }
                }
                if (reuse) {
                    for (int device : tpDevices) {
                        Data &local = *dst.multiDeviceDatas.at(device);
                        if (local.dims != dims) local.Resize(dims);
                    }
                    if (dst.dims != dims) V41SyncRootFromReplica(dst, tpDevices);
                    return;
                }
                V41ResetMultiDevice(dst);
                dst.FreeSpace();
                dst.dataType = type;
                dst.UpdateUnitSize();
                dst.Resize(dims);
                dst.dataDevice = DataDevice::CUDA;
                dst.dataDeviceIds = tpDevices;
                PrepareMultiCudaReplicatedData(dst, tpDevices, false);
                const int oriDevice = FastllmCudaGetDevice();
                std::vector<int> capacityDims = dims;
                if (dims.size() == 3 && dims[0] == 1 && dims[1] == 1 && dims[2] > 0 &&
                    (type == DataType::INT32 || type == DataType::INT8)) {
                    int capacity = 64;
                    while (capacity < dims[2] && capacity <= (1 << 29)) capacity *= 2;
                    capacityDims[2] = std::max(capacity, dims[2]);
                }
                for (int device : tpDevices) {
                    FastllmCudaSetDevice(device);
                    Data &local = *dst.multiDeviceDatas.at(device);
                    local.Resize(capacityDims);
                    local.Allocate(false);
                    if (capacityDims != dims) local.Resize(dims);
                }
                FastllmCudaSetDevice(oriDevice);
                V41SyncRootFromReplica(dst, tpDevices);
                return;
            }
#endif
            dst.dataType = type;
            dst.Resize(dims);
            dst.ToDevice(reference.dataDevice);
            dst.Allocate();
        }

        std::vector<int> V41ReadTokenIds(const Data &inputIds) {
            Data cpuIds;
            const Data *src = &inputIds;
            if (inputIds.dataDevice != DataDevice::CPU) {
                cpuIds.CopyFrom(inputIds);
                cpuIds.ToDevice(DataDevice::CPU);
                src = &cpuIds;
            }
            std::vector<int> ret;
            uint64_t n = src->Count(0);
            for (uint64_t i = 0; i < n; i++) {
                if (src->dataType == DataType::FLOAT32) {
                    ret.push_back((int)(((const float*)src->cpuData)[i] + 1e-6));
                } else if (src->dataType == DataType::INT32) {
                    ret.push_back(((const int32_t*)src->cpuData)[i]);
                } else {
                    ErrorInFastLLM("DeepSeekV41: unsupported inputIds dtype.");
                }
            }
            return ret;
        }

        void V41RMSNormBF16(const Data &input, Data &weight, float eps, Data &output) {
#ifdef USE_CUDA
            if (V41EnvFlag("FASTLLM_DSV41_REFERENCE_MATH") && FastllmCudaV41ReferenceNorm(input,weight,eps,output)) return;
#endif
            RMSNorm(input, weight, eps, output);
            ToDataType(output, DataType::BFLOAT16);
        }

#ifdef USE_CUDA
        bool V41PrefillHcMixTp(const Data &input, Data &hcFn, Data &hcScale, Data &hcBase,
                              int hcMult, int sinkhornIters, float eps, float normEps,
                              Data &pre, Data &post, Data &comb) {
            // A metadata-only replicated root often names just its first GPU.
            // The replica map, not root.dataDeviceIds, defines the participating ranks.
            std::vector<int> devices;
            if (input.multiDeviceData && input.IsTensorParallelReplicated()) {
                for (const auto &entry : input.multiDeviceDatas) devices.push_back(entry.first);
            }
            if (devices.size() < 2 || !input.multiDeviceData || !input.IsTensorParallelReplicated() ||
                input.dims.size() != 4 || input.dims[0] != 1 || hcMult != 4 ||
                input.dims[2] != hcMult || input.dims[1] < (int)devices.size() * 256 ||
                (input.dataType != DataType::BFLOAT16 && input.dataType != DataType::FLOAT16 &&
                 input.dataType != DataType::FLOAT32) || FastllmCudaGraphIsCapturingFast() ||
                V41EnvFlag("FASTLLM_DSV41_REFERENCE_MATH")) return false;
            for (const Data *data : {&input, (const Data *)&hcFn, (const Data *)&hcScale, (const Data *)&hcBase}) {
                if (!data->multiDeviceData || !data->IsTensorParallelReplicated()) return false;
                for (int device : devices) {
                    auto it = data->multiDeviceDatas.find(device);
                    if (it == data->multiDeviceDatas.end() || !it->second || !it->second->cudaData ||
                        it->second->dims != data->dims) return false;
                }
            }
            // Collective counts are signed ints; use wide arithmetic before
            // selecting the optional partitioned path.
            const int tokens = input.dims[1];
            const int64_t rows = ((int64_t)tokens + devices.size() - 1) / devices.size();
            if (rows * devices.size() > INT_MAX || rows * hcMult * hcMult > INT_MAX) return false;
            if (!FastllmInitNccl(devices)) return false;
            const int perRank = (int)rows, padded = perRank * devices.size();
            V41AllocLike(pre, DataType::FLOAT32, {1, padded, hcMult}, input, devices);
            V41AllocLike(post, DataType::FLOAT32, {1, padded, hcMult}, input, devices);
            V41AllocLike(comb, DataType::FLOAT32, {1, padded, hcMult, hcMult}, input, devices);
            auto viewRows = [](Data &view, const Data &parent, int first, int count) {
                view.FakeFrom(parent, (size_t)first * parent.strides[1] * parent.unitSize);
                auto dims = parent.dims; dims[1] = count;
                view.expansionDims.clear(); view.Resize(dims);
                view.expansionDims = dims; view.expansionSize = view.Count(0);
                view.expansionBytes = view.GetBytes();
            };
            bool dispatched = MultiCudaRunDeviceCallbacks(devices, [&](int rank, int device) {
                const int begin = rank * perRank, count = std::min(perRank, tokens - begin);
                Data x, p, s, c;
                viewRows(x, *input.multiDeviceDatas.at(device), begin, count);
                viewRows(p, *pre.multiDeviceDatas.at(device), begin, count);
                viewRows(s, *post.multiDeviceDatas.at(device), begin, count);
                viewRows(c, *comb.multiDeviceDatas.at(device), begin, count);
                if (count < perRank) {
                    for (Data *output : {&pre, &post, &comb}) {
                        Data &local = *output->multiDeviceDatas.at(device);
                        FastllmCudaMemset0((float *)local.cudaData + (size_t)(begin + count) * local.strides[1],
                            (size_t)(perRank - count) * local.strides[1] * sizeof(float));
                    }
                }
                bool ok = FastllmCudaDeepSeekV41HcMix(x, *hcFn.multiDeviceDatas.at(device),
                    *hcScale.multiDeviceDatas.at(device), *hcBase.multiDeviceDatas.at(device),
                    hcMult, sinkhornIters, eps, normEps, p, s, c);
                AssertInFastLLM(ok, "DeepSeekV41 prefill HC query partition failed.");
            });
            AssertInFastLLM(dispatched, "DeepSeekV41 prefill HC dispatch failed.");
            dispatched = MultiCudaRunDeviceCallbacks(devices, [&](int rank, int device) {
                bool ok = true;
                for (Data *output : {&pre, &post, &comb}) {
                    Data &local = *output->multiDeviceDatas.at(device);
                    const int count = perRank * local.strides[1];
                    const bool gathered = FastllmNcclAllGather((float *)local.cudaData + (size_t)rank * count,
                        local.cudaData, count, (int)DataType::FLOAT32, device);
                    ok = gathered && ok;
                }
                AssertInFastLLM(ok, "DeepSeekV41 prefill HC all-gather failed.");
            });
            AssertInFastLLM(dispatched, "DeepSeekV41 prefill HC gather dispatch failed.");
            for (Data *output : {&pre, &post, &comb}) {
                for (int device : devices) {
                    auto dims = output->multiDeviceDatas.at(device)->dims; dims[1] = tokens;
                    output->multiDeviceDatas.at(device)->Resize(dims);
                }
                V41SyncRootFromReplica(*output, devices);
            }
            return true;
        }
#endif

        void V41HcMix(const Data &input, Data &hcFn, Data &hcScale, Data &hcBase,
                      int hcMult, int sinkhornIters, float eps, float normEps,
                      Data &pre, Data &post, Data &comb) {
#ifdef USE_CUDA
            if (V41PrefillHcMixTp(input, hcFn, hcScale, hcBase, hcMult,
                                 sinkhornIters, eps, normEps, pre, post, comb)) return;
#endif
            V41Executor().Run("DeepSeekV41HcMix", {
                {"input", (Data*)&input}, {"hcFn", &hcFn}, {"hcScale", &hcScale}, {"hcBase", &hcBase},
                {"pre", &pre}, {"post", &post}, {"comb", &comb}
            }, {{"eps", eps}, {"normEps", normEps}}, {{"hcMult", hcMult}, {"sinkhornIters", sinkhornIters}});
        }

        void V41HcPost(Data &input, Data &residual, Data &post, Data &comb, Data &output) {
            V41Executor().Run("DeepSeekV41HcPost", {{"input", &input}, {"residual", &residual},
                {"post", &post}, {"comb", &comb}, {"output", &output}}, {}, {});
        }

        void V41HcApplyPre(const Data &input, const Data &pre, Data &output) {
            V41Executor().Run("DeepSeekV41HcApplyPre", {
                {"input", (Data*)&input}, {"pre", (Data*)&pre}, {"output", &output}
            }, {}, {});
        }

#ifdef USE_CUDA
        // 融合 kernel 是**直接调用**的，不经过执行器，所以不会像普通算子那样自动把
        // 输入搬到本层所属的卡上。按层切分时，stage 边界那一层的 hidden / pre 还在
        // 上一张卡，而本层的 norm 权重已经加载到下一张卡，混着用直接触发
        // Warp MMU Fault（现场会以 "moving bias to device" 之类的无关报错浮现）。
        // 因此只有当所有张量都和当前 CUDA 设备一致时才走融合路径。
        // 用 dataDeviceIds 判断而不是 cudaPointerGetAttributes：后者每次都是一次
        // runtime 调用，这个函数每层要走两次。
        bool V41FusedPreNormSameDevice(const Data &input, const Data &pre, const Data &normWeight) {
            const int current = FastllmCudaGetDevice();
            if (current < 0) {
                return false;
            }
            auto deviceOf = [](const Data &data) -> int {
                if (data.dataDevice != DataDevice::CUDA || data.dataDeviceIds.empty()) {
                    return -1;
                }
                return data.dataDeviceIds[0];
            };
            if (deviceOf(input) != current || deviceOf(pre) != current) {
                return false;
            }
            // 还在 CPU 上的 norm 权重会被 kernel 内部搬到当前卡，是安全的
            const int weightDevice = deviceOf(normWeight);
            return weightDevice < 0 || weightDevice == current;
        }
#endif

        // 子层入口的「HcApplyPre -> RMSNorm」：中间张量只被紧接着的 RMSNorm 读一次，
        // 单卡 CUDA 上用融合 kernel 一次算完（结果逐 bit 相同）；其余情况退回两步。
        void V41HcApplyPreNorm(const Data &input, const Data &pre, Data &normWeight, float eps,
                               Data &tmp, Data &output) {
#ifdef USE_CUDA
            static const bool disableFused = V41EnvFlag("FASTLLM_DSV41_DISABLE_HCPRENORM");
            if (!disableFused && input.multiDeviceData && input.IsTensorParallelReplicated() &&
                pre.multiDeviceData && pre.IsTensorParallelReplicated() &&
                input.dataType == DataType::BFLOAT16 && input.dims.size() == 4 &&
                pre.dataType == DataType::FLOAT32 && normWeight.dataType == DataType::FLOAT32) {
                // A replicated root may advertise only its primary device;
                // the replica map is the complete TP group.
                std::vector<int> devices;
                for (const auto &replica : input.multiDeviceDatas) devices.push_back(replica.first);
                bool compatible = !devices.empty();
                for (int device : devices) {
                    const auto xi = input.multiDeviceDatas.find(device), pi = pre.multiDeviceDatas.find(device);
                    compatible &= xi != input.multiDeviceDatas.end() && xi->second && xi->second->cudaData &&
                                  pi != pre.multiDeviceDatas.end() && pi->second && pi->second->cudaData;
                }
                if (compatible) {
                    if (!normWeight.multiDeviceData)
                        PrepareMultiCudaReplicatedData(normWeight, devices, true);
                    compatible = normWeight.IsTensorParallelReplicated();
                    for (int device : devices) {
                        const auto wi = normWeight.multiDeviceDatas.find(device);
                        compatible &= wi != normWeight.multiDeviceDatas.end() && wi->second && wi->second->cudaData;
                    }
                }
                if (compatible) {
                    V41AllocLike(output, DataType::BFLOAT16,
                        {input.dims[0], input.dims[1], input.dims[3]}, input, devices);
                    std::vector<int> ok(devices.size(), 0);
                    const bool submitted = MultiCudaRunDeviceCallbacks(devices, [&](int rank, int device) {
                        ok[rank] = FastllmCudaDeepSeekV41HcPreNorm(*input.multiDeviceDatas.at(device),
                            *pre.multiDeviceDatas.at(device), *normWeight.multiDeviceDatas.at(device),
                            eps, *output.multiDeviceDatas.at(device));
                    });
                    if (submitted && std::all_of(ok.begin(), ok.end(), [](int v) { return v != 0; })) return;
                }
            }
            if (!disableFused && input.dataDevice == DataDevice::CUDA && !input.multiDeviceData &&
                V41FusedPreNormSameDevice(input, pre, normWeight) &&
                FastllmCudaDeepSeekV41HcPreNorm(input, pre, normWeight, eps, output)) {
                return;
            }
#endif
            V41HcApplyPre(input, pre, tmp);
            V41RMSNormBF16(tmp, normWeight, eps, output);
        }

        void V41HcMixPreNorm(const Data &input, const Data &previousPre, Data &fn, Data &scale,
                            Data &base, Data &normWeight, int hcMult, int iters, float mixEps,
                            float normEps, Data &pre, Data &post, Data &comb, Data &tmp, Data &output) {
#ifdef USE_CUDA
            const bool eligible = !V41EnvFlag("FASTLLM_DSV41_REFERENCE_MATH") &&
                !V41EnvFlag("FASTLLM_DSV41_LEGACY_HCMIX") &&
                !V41EnvFlag("FASTLLM_DSV41_LEGACY_HCMIX_DECODE") &&
                !V41EnvFlag("FASTLLM_DSV41_DISABLE_HCPRENORM") && hcMult == 4 &&
                input.dataType == DataType::BFLOAT16 && input.dims.size() == 4 &&
                input.dims[0] * input.dims[1] > 0 && input.dims[0] * input.dims[1] < 16 &&
                input.dims[3] >= 4096 && input.dims[3] <= 8192 && !(input.dims[3] & 1);
            if (eligible && input.multiDeviceData && input.IsTensorParallelReplicated() &&
                previousPre.multiDeviceData && previousPre.IsTensorParallelReplicated()) {
                std::vector<int> devices;
                for (const auto &replica : input.multiDeviceDatas) devices.push_back(replica.first);
                bool compatible = !devices.empty();
                for (Data *w : {&fn, &scale, &base, &normWeight}) {
                    if (!w->multiDeviceData) PrepareMultiCudaReplicatedData(*w, devices, true);
                    compatible &= w->IsTensorParallelReplicated() && w->dataType == DataType::FLOAT32;
                }
                const Data *inputs[] = {&input, &previousPre, &fn, &scale, &base, &normWeight};
                for (int device : devices) {
                    for (const Data *v : inputs) {
                        auto it = v->multiDeviceDatas.find(device);
                        compatible &= it != v->multiDeviceDatas.end() && it->second && it->second->cudaData;
                    }
                }
                if (compatible) {
                    const int b = input.dims[0], n = input.dims[1];
                    V41AllocLike(pre, DataType::FLOAT32, {b, n, 4}, input, devices);
                    V41AllocLike(post, DataType::FLOAT32, {b, n, 4}, input, devices);
                    V41AllocLike(comb, DataType::FLOAT32, {b, n, 4, 4}, input, devices);
                    V41AllocLike(output, DataType::BFLOAT16, {b, n, input.dims[3]}, input, devices);
                    std::vector<int> ok(devices.size(), 0);
                    const bool submitted = MultiCudaRunDeviceCallbacks(devices, [&](int rank, int device) {
                        ok[rank] = FastllmCudaDeepSeekV41HcMixPreNorm(*input.multiDeviceDatas.at(device),
                            *previousPre.multiDeviceDatas.at(device), *fn.multiDeviceDatas.at(device),
                            *scale.multiDeviceDatas.at(device), *base.multiDeviceDatas.at(device),
                            *normWeight.multiDeviceDatas.at(device), iters, mixEps, normEps,
                            *pre.multiDeviceDatas.at(device), *post.multiDeviceDatas.at(device),
                            *comb.multiDeviceDatas.at(device), *output.multiDeviceDatas.at(device));
                    });
                    if (submitted && std::all_of(ok.begin(), ok.end(), [](int v) { return v != 0; })) return;
                }
            }
            if (eligible && !input.multiDeviceData &&
                V41FusedPreNormSameDevice(input, previousPre, normWeight) &&
                FastllmCudaDeepSeekV41HcMixPreNorm(input, previousPre, fn, scale, base, normWeight,
                    iters, mixEps, normEps, pre, post, comb, output)) return;
#endif
            V41HcMix(input, fn, scale, base, hcMult, iters, mixEps, normEps, pre, post, comb);
            V41HcApplyPreNorm(input, previousPre, normWeight, normEps, tmp, output);
        }

#ifdef USE_CUDA
        bool V41RouterScores(const Data &input, Data &weight, float temperature, Data &output) {
            if (input.dataType != DataType::BFLOAT16 || input.dims.empty() ||
                input.dims.back() <= 0 || input.Count(0) / input.dims.back() > 8 ||
                weight.dataType != DataType::FLOAT32 || weight.dims.size() != 2 ||
                weight.dims[1] != input.dims.back() || weight.dims[0] > 1024) return false;
            if (!input.multiDeviceData) {
                const int current = FastllmCudaGetDevice();
                if (input.dataDevice != DataDevice::CUDA || input.dataDeviceIds.empty() ||
                    input.dataDeviceIds[0] != current ||
                    (weight.dataDevice == DataDevice::CUDA && !weight.dataDeviceIds.empty() &&
                     weight.dataDeviceIds[0] != current)) return false;
                return FastllmCudaV41RouterScores(input, weight, temperature, output);
            }
            if (!input.IsTensorParallelReplicated()) return false;
            std::vector<int> devices;
            for (const auto &replica : input.multiDeviceDatas) devices.push_back(replica.first);
            if (devices.empty()) return false;
            if (!weight.multiDeviceData) PrepareMultiCudaReplicatedData(weight, devices, true);
            if (!weight.IsTensorParallelReplicated()) return false;
            for (int device : devices) {
                auto w = weight.multiDeviceDatas.find(device);
                if (w == weight.multiDeviceDatas.end() || !w->second || !w->second->cudaData ||
                    !input.multiDeviceDatas.at(device) || !input.multiDeviceDatas.at(device)->cudaData)
                    return false;
            }
            auto dims = input.dims;
            dims.back() = weight.dims[0];
            V41AllocLike(output, DataType::FLOAT32, dims, input, devices);
            std::vector<int> ok(devices.size(), 0);
            const bool submitted = MultiCudaRunDeviceCallbacks(devices, [&](int rank, int device) {
                ok[rank] = FastllmCudaV41RouterScores(*input.multiDeviceDatas.at(device),
                    *weight.multiDeviceDatas.at(device), temperature, *output.multiDeviceDatas.at(device));
            });
            return submitted && std::all_of(ok.begin(), ok.end(), [](int v) { return v != 0; });
        }

        // Decode partitions independent output channels. Keeping the weight
        // replicas also preserves the existing prefill and unsupported-layout
        // paths. Each owner computes full K, then gathers rounded BF16 outputs.
        bool V41EngramTpLinear(Data &input, Data &weight, Data &output) {
            if (!input.multiDeviceData || !input.IsTensorParallelReplicated() ||
                input.dataType != DataType::BFLOAT16 || weight.dataType != DataType::FP8_E4M3 ||
                weight.dims.size() != 2 || input.dims.empty() ||
                input.Count(0) != (uint64_t)weight.dims[1] || weight.IsRepacked ||
                weight.blockK <= 0 || weight.blockM <= 0 ||
                weight.dims[1] % weight.blockM != 0) return false;
            std::vector<int> devices;
            for (const auto &local : input.multiDeviceDatas) devices.push_back(local.first);
            if (devices.size() < 2 || weight.dims[0] % (devices.size() * weight.blockK) != 0 ||
                V41EnvFlag("FASTLLM_DISABLE_NCCL") || !FastllmInitNccl(devices)) return false;
            if (!weight.multiDeviceData) PrepareMultiCudaReplicatedData(weight, devices, true);
            if (!weight.IsTensorParallelReplicated()) return false;
            for (int device : devices) {
                auto w = weight.multiDeviceDatas.find(device);
                if (w == weight.multiDeviceDatas.end() || !w->second || !w->second->cudaData ||
                    !input.multiDeviceDatas.at(device) || !input.multiDeviceDatas.at(device)->cudaData)
                    return false;
            }
            auto dims = input.dims;
            dims.back() = weight.dims[0];
            V41AllocLike(output, DataType::BFLOAT16, dims, input, devices);
            const int width = weight.dims[0] / devices.size();
            std::vector<int> ready(devices.size(), 0);
            bool dispatched = MultiCudaRunDeviceCallbacks(devices, [&](int rank, int device) {
                ready[rank] = FastllmCudaBFloat16FP8ReplicatedRowSlice(
                    *input.multiDeviceDatas.at(device), *weight.multiDeviceDatas.at(device),
                    *output.multiDeviceDatas.at(device), rank * width, width);
            });
            if (!dispatched || !std::all_of(ready.begin(), ready.end(), [](int v) { return v != 0; }))
                return false;
            dispatched = MultiCudaRunDeviceCallbacks(devices, [&](int rank, int device) {
                auto *dest = (uint16_t *)output.multiDeviceDatas.at(device)->cudaData;
                const bool ok = FastllmNcclAllGather(dest + (size_t)rank * width, dest,
                    width, DataType::BFLOAT16, device);
                AssertInFastLLM(ok, "DeepSeekV41 Engram projection all-gather failed");
            });
            AssertInFastLLM(dispatched, "DeepSeekV41 Engram projection dispatch failed");
            return true;
        }
#endif

        void V41EngramApply(Data &hidden, const Data &kv, Data &qWeight, Data &kWeight,
                            const Data *mask, float eps) {
            DataDict datas = {
                {"hidden", &hidden}, {"kv", (Data*)&kv}, {"qWeight", &qWeight}, {"kWeight", &kWeight}
            };
            if (mask != nullptr) {
                datas["mask"] = (Data*)mask;
            }
            V41Executor().Run("DeepSeekV41EngramApply", datas, {{"eps", eps}}, {});
        }

        struct V41RopeParams {
            int ropeDim;
            float base;
            int originalSeqLen;
            float factor;
            int betaFast;
            int betaSlow;
        };

#ifdef USE_CUDA
        // Amortize worker/event handoffs across independent decode requests.
        // Each rank still runs the portable native kernels in the same stream.
        using V41DecodeTransform = FastllmCudaV41DecodeTransform;

        bool V41PrepareDecodeTransforms(const std::vector<V41DecodeTransform> &items,
                                 const V41RopeParams &rope, int windowSize,
                                 const std::vector<int> &devices,
                                 std::function<void(int, int)> &task) {
            if (devices.empty() || items.empty()) return false;
            for (const auto &item : items) {
                if (!item.input || !item.input->multiDeviceData) return false;
                for (int device : devices) {
                    auto it = item.input->multiDeviceDatas.find(device);
                    if (it == item.input->multiDeviceDatas.end() || !it->second) return false;
                    const Data &x = *it->second;
                    if (!x.cudaData || x.dataDevice != DataDevice::CUDA ||
                        (x.dims.size() != 3 && x.dims.size() != 4) || x.dims[1] != 1 ||
                        x.dims.back() > 1024 || x.dims.back() % 32 ||
                        rope.ropeDim <= 0 || rope.ropeDim > 128 || rope.ropeDim > x.dims.back() ||
                        (rope.ropeDim & 1) ||
                        (x.dataType != DataType::BFLOAT16 && x.dataType != DataType::FLOAT16 &&
                         x.dataType != DataType::FLOAT32)) return false;
                    if (item.window) {
                        auto ring = item.window->multiDeviceDatas.find(device);
                        if (x.dims.size() != 3 || ring == item.window->multiDeviceDatas.end() ||
                            !ring->second || ring->second->dataType != x.dataType ||
                            !ring->second->cudaData || ring->second->dims.size() != 3 ||
                            ring->second->dims[0] != x.dims[0] ||
                            ring->second->dims[1] != windowSize ||
                            ring->second->dims[2] != x.dims[2])
                            return false;
                    }
                }
            }
            static const bool disableFakeQuant = V41EnvFlag("FASTLLM_DSV41_DISABLE_FAKE_QUANT");
            task = [&items, rope, windowSize](int, int device) {
                std::vector<FastllmCudaV41DecodeTransform> local;
                local.reserve(items.size());
                for (const auto &item : items) {
                    local.push_back({item.input->multiDeviceDatas.at(device),
                        item.window ? item.window->multiDeviceDatas.at(device) : nullptr,
                        item.startPos, disableFakeQuant ? 0 : item.quantMode, item.inverse});
                }
                if (FastllmCudaV41DecodeTransformsBatch(local, rope.ropeDim, rope.base,
                        rope.originalSeqLen, rope.factor, rope.betaFast, rope.betaSlow, windowSize)) return;

                for (const auto &item : items) {
                    Data &x = *item.input->multiDeviceDatas.at(device);
                    bool ok;
                    if (item.window) {
                        ok = FastllmCudaDeepSeekV41WindowStore(x, *item.window->multiDeviceDatas.at(device),
                                                             item.startPos, windowSize);
                    } else {
                        ok = FastllmCudaDeepSeekV41RotaryQuant(x, rope.ropeDim, rope.base, item.startPos, 1,
                            item.inverse, rope.originalSeqLen, rope.factor, rope.betaFast, rope.betaSlow,
                            disableFakeQuant ? 0 : item.quantMode, x.dims.back(), 32);
                    }
                    AssertInFastLLM(ok, "DeepSeekV41 batched decode transform failed.");
                }
            };
            return true;
        }

        bool V41DecodeTransforms(const std::vector<V41DecodeTransform> &items,
                                 const V41RopeParams &rope, int windowSize,
                                 const std::vector<int> &devices, bool workerOrdered = false) {
            std::function<void(int, int)> task;
            if (!V41PrepareDecodeTransforms(items, rope, windowSize, devices, task)) return false;
            if (workerOrdered && MultiCudaRunDeviceCallbacksEnqueueOnly(devices, task)) return true;
            return MultiCudaRunDeviceCallbacks(devices, task);
        }

        using V41DecodeIndexTask = FastllmCudaV41DecodeIndexTask;

        bool V41PrepareDecodeIndexer(const std::vector<V41DecodeIndexTask> &items,
                                   const V41RopeParams &rope, int ratio, int topK,
                                   int blockSize, int candidateTopK,
                                   const std::vector<int> &devices,
                                   std::function<void(int, int)> &task) {
            if (items.empty() || devices.empty()) return false;
            // Validate before mutating queries. Unsupported layouts retain the
            // existing executor path, including its architecture fallbacks.
            for (const auto &r : items) {
                for (Data *data : {r.query, r.weights, r.keys}) {
                    if (!data || !data->multiDeviceData || !data->IsTensorParallelReplicated()) return false;
                    for (int device : devices) {
                        auto it = data->multiDeviceDatas.find(device);
                        if (it == data->multiDeviceDatas.end() || !it->second || !it->second->cudaData) return false;
                    }
                }
                if (r.query->dataType != DataType::BFLOAT16 || r.weights->dataType != DataType::FLOAT32 ||
                    r.keys->dataType != DataType::BFLOAT16 || r.query->dims.size() != 4 ||
                    r.query->dims[0] != 1 || r.query->dims[1] != 1 || r.query->dims[3] != 128 ||
                    r.keys->dims.size() != 3 || r.keys->dims[0] != 1 || r.keys->dims[2] != 128 ||
                    rope.ropeDim <= 0 || rope.ropeDim > 128 || (rope.ropeDim & 1)) return false;
                if (r.useCandidates) {
                    for (int device : devices) {
                        auto it = r.candidates->multiDeviceDatas.find(device);
                        if (it == r.candidates->multiDeviceDatas.end() || !it->second || !it->second->cudaData)
                            return false;
                    }
                }
            }
            for (const auto &r : items) {
                V41AllocLike(*r.topk, DataType::INT32, {1, 1, std::min(topK, r.keys->dims[1])},
                             *r.query, devices);
                if (r.writeCandidates) {
                    V41AllocLike(*r.candidates, DataType::INT8,
                        {1, 1, (r.keys->dims[1] + blockSize - 1) / blockSize}, *r.query, devices);
                }
            }
            static const bool disableFakeQuant = V41EnvFlag("FASTLLM_DSV41_DISABLE_FAKE_QUANT");
            task = [&items, rope, ratio, topK, blockSize, candidateTopK](int, int device) {
                std::vector<FastllmCudaV41DecodeIndexTask> local;
                local.reserve(items.size());
                for (const auto &r : items) local.push_back({r.query->multiDeviceDatas.at(device),
                    r.weights->multiDeviceDatas.at(device), r.keys->multiDeviceDatas.at(device),
                    (r.writeCandidates || r.useCandidates) ? r.candidates->multiDeviceDatas.at(device) : nullptr,
                    r.topk->multiDeviceDatas.at(device), r.startPos, r.writeCandidates, r.useCandidates});
                if (FastllmCudaV41DecodeIndexerBatch(local, ratio, topK, blockSize, candidateTopK,
                        rope.ropeDim, rope.base, rope.originalSeqLen, rope.factor, rope.betaFast, rope.betaSlow,
                        !disableFakeQuant)) return;
                for (const auto &r : items) {
                    Data &q = *r.query->multiDeviceDatas.at(device);
                    Data &w = *r.weights->multiDeviceDatas.at(device);
                    Data &k = *r.keys->multiDeviceDatas.at(device);
                    Data &out = *r.topk->multiDeviceDatas.at(device);
                    Data score;
                    bool ok = FastllmCudaDeepSeekV41RotaryQuant(q, rope.ropeDim, rope.base, r.startPos, 1, false,
                        rope.originalSeqLen, rope.factor, rope.betaFast, rope.betaSlow,
                        disableFakeQuant ? 0 : 2, 128, 32);
                    ok = ok && FastllmCudaDeepSeekV41IndexerScore(q, w, k, ratio, r.startPos, score);
                    Data *candidate = nullptr;
                    if (r.writeCandidates || r.useCandidates) candidate = r.candidates->multiDeviceDatas.at(device);
                    if (r.writeCandidates)
                        ok = ok && FastllmCudaDeepSeekV41CandidateBlocks(score, blockSize, candidateTopK,
                                                                       ratio, r.startPos, *candidate);
                    // The source layer publishes candidates for subsequent
                    // layers, but does not apply its own new mask to top-k.
                    ok = ok && FastllmCudaDeepSeekV41IndexerTopK(score, r.useCandidates ? candidate : nullptr,
                        topK, ratio, r.startPos, std::max(1, blockSize), out);
                    AssertInFastLLM(ok, "DeepSeekV41 batched indexer failed.");
                    if (score.cudaData && FastllmCudaFreeAfterCurrentThreadStream(score.cudaData))
                        score.cudaData = nullptr;
                }
            };
            return true;
        }

        bool V41DecodeIndexerBatch(const std::vector<V41DecodeIndexTask> &items,
                                   const V41RopeParams &rope, int ratio, int topK,
                                   int blockSize, int candidateTopK,
                                   const std::vector<int> &devices) {
            std::function<void(int, int)> task;
            return V41PrepareDecodeIndexer(items, rope, ratio, topK, blockSize,
                                          candidateTopK, devices, task) &&
                   MultiCudaRunDeviceCallbacks(devices, task);
        }
#endif

        void V41RotaryQuant(Data &x, const V41RopeParams &rope, int startPos, int posStep,
                            bool inverse, int quantMode, int quantBlock, int quantDim = -1) {
            static const bool disableFakeQuant = V41EnvFlag("FASTLLM_DSV41_DISABLE_FAKE_QUANT");
            if (disableFakeQuant) {
                quantMode = 0;
            }
            IntDict ints = {
                {"ropeDim", rope.ropeDim}, {"startPos", startPos}, {"posStep", posStep},
                {"inverse", inverse ? 1 : 0}, {"originalSeqLen", rope.originalSeqLen},
                {"betaFast", rope.betaFast}, {"betaSlow", rope.betaSlow},
                {"quantMode", quantMode}, {"quantBlock", quantBlock}
            };
            if (quantDim > 0) {
                ints["quantDim"] = quantDim;
            }
            V41Executor().Run("DeepSeekV41RotaryQuant", {{"input", &x}},
                              {{"ropeBase", rope.base}, {"ropeFactor", rope.factor}}, ints);
        }

        void V41Compress(const Data &kv, const Data *score, Data &normWeight, int ratio, float normEps, Data &output) {
            DataDict datas = {{"kv", (Data*)&kv}, {"normWeight", &normWeight}, {"output", &output}};
            if (score != nullptr) {
                datas["score"] = (Data*)score;
            }
            V41Executor().Run("DeepSeekV41Compress", datas, {{"normEps", normEps}}, {{"compressRatio", ratio}});
        }

        void V41IndexerScore(const Data &q, const Data &weights, const Data &k, int ratio, int startPos,
                             Data &output) {
            V41Executor().Run("DeepSeekV41IndexerScore", {
                {"q", (Data*)&q}, {"weights", (Data*)&weights}, {"k", (Data*)&k}, {"output", &output}
            }, {}, {{"compressRatio", ratio}, {"startPos", startPos}});
        }

        void V41CandidateBlocks(const Data &score, int blockSize, int topkBlocks, int ratio, int startPos, Data &output) {
            V41Executor().Run("DeepSeekV41CandidateBlocks", {
                {"score", (Data*)&score}, {"output", &output}
            }, {}, {{"blockSize", blockSize}, {"topkBlocks", topkBlocks},
                    {"compressRatio", ratio}, {"startPos", startPos}});
        }

        void V41IndexerTopK(const Data &score, const Data *candidates, int topK, int ratio, int startPos,
                            int blockSize, Data &output) {
            DataDict datas = {{"score", (Data*)&score}, {"output", &output}};
            if (candidates != nullptr) {
                datas["candidates"] = (Data*)candidates;
            }
            V41Executor().Run("DeepSeekV41IndexerTopK", datas, {},
                              {{"topK", topK}, {"compressRatio", ratio}, {"startPos", startPos},
                               {"blockSize", blockSize}});
        }

#ifdef USE_CUDA
        // Split independent prefill query rows across the TP group. Each rank
        // keeps the original score/top-k kernels and causal positions; only the
        // compact selected indices and candidate masks are exchanged.
        bool V41PrefillIndexerTp(const Data &q, const Data &weights, const Data &keys,
                                 Data &candidates, Data &topk, bool writeCandidates,
                                 bool useCandidates, int ratio, int startPos, int topK,
                                 int blockSize, int candidateTopK,
                                 const std::vector<int> &devices) {
            if (devices.size() < 2 || q.dims.size() != 4 || q.dims[0] != 1 ||
                q.dims[1] < 1024 || q.dataType != DataType::BFLOAT16 ||
                weights.dataType != DataType::FLOAT32 || weights.dims.size() != 3 ||
                weights.dims[0] != 1 || weights.dims[1] != q.dims[1] ||
                weights.dims[2] != q.dims[2] || keys.dataType != DataType::BFLOAT16 ||
                keys.dims.size() != 3 || keys.dims[0] != 1 || keys.dims[1] <= 0 ||
                keys.dims[2] != q.dims[3] || topK <= 0 || ratio <= 0 ||
                (writeCandidates && useCandidates) || FastllmCudaGraphIsCapturingFast()) return false;
            const int tokens = q.dims[1], width = std::min(topK, keys.dims[1]);
            const int ranks = (int)devices.size(), perRank = (tokens + ranks - 1) / ranks;
            const int paddedTokens = perRank * ranks;
            const int maskWidth = blockSize > 0 ? (keys.dims[1] + blockSize - 1) / blockSize : 0;
            if ((writeCandidates || useCandidates) && (maskWidth <= 0 || candidateTopK <= 0)) return false;
            if ((int64_t)perRank * std::max(width, maskWidth) > std::numeric_limits<int>::max()) return false;
            if (useCandidates && (candidates.dataType != DataType::INT8 ||
                candidates.dims != std::vector<int>({1, tokens, maskWidth}))) return false;
            std::vector<const Data *> inputs = {&q, &weights, &keys};
            if (useCandidates) inputs.push_back(&candidates);
            for (const Data *data : inputs) {
                if (!data->multiDeviceData || !data->IsTensorParallelReplicated()) return false;
                for (int device : devices) {
                    auto it = data->multiDeviceDatas.find(device);
                    if (it == data->multiDeviceDatas.end() || !it->second ||
                        !it->second->cudaData || it->second->dims != data->dims ||
                        it->second->dataType != data->dataType) return false;
                }
            }
            if (!FastllmInitNccl(devices)) return false;
            V41AllocLike(topk, DataType::INT32, {1, paddedTokens, width}, q, devices);
            if (writeCandidates)
                V41AllocLike(candidates, DataType::INT8, {1, paddedTokens, maskWidth}, q, devices);
            auto rowsView = [](Data &view, const Data &source, int first, int count) {
                view.FakeFrom(source, (size_t)first * source.strides[1] * source.unitSize);
                auto dims = source.dims; dims[1] = count;
                view.expansionDims.clear(); view.Resize(dims);
                view.expansionDims = dims; view.expansionSize = view.Count(0);
                view.expansionBytes = view.GetBytes();
            };
            std::vector<Data> scores(devices.size());
            const int chunkTokens = V41IndexScoreChunk(perRank, keys.dims[1]);
            bool dispatched = MultiCudaRunDeviceCallbacks(devices, [&](int rank, int device) {
                const int begin = rank * perRank, end = std::min(tokens, begin + perRank);
                Data &output = *topk.multiDeviceDatas.at(device);
                Data *mask = (writeCandidates || useCandidates) ? candidates.multiDeviceDatas.at(device) : nullptr;
                // The last rank may have padding; initialise only the bytes
                // exchanged by the collective, never another rank's range.
                FastllmCudaMemset0((int32_t *)output.cudaData + (size_t)begin * width,
                                  (size_t)perRank * width * sizeof(int32_t));
                if (writeCandidates)
                    FastllmCudaMemset0((uint8_t *)mask->cudaData + (size_t)begin * maskWidth,
                                      (size_t)perRank * maskWidth);
                for (int first = begin; first < end; first += chunkTokens) {
                    const int count = std::min(chunkTokens, end - first);
                    Data query, weight, selected, candidate;
                    rowsView(query, *q.multiDeviceDatas.at(device), first, count);
                    rowsView(weight, *weights.multiDeviceDatas.at(device), first, count);
                    rowsView(selected, output, first, count);
                    if (mask) rowsView(candidate, *mask, first, count);
                    bool ok = FastllmCudaDeepSeekV41IndexerScore(query, weight,
                        *keys.multiDeviceDatas.at(device), ratio, startPos + first, scores[rank]);
                    if (writeCandidates)
                        ok = ok && FastllmCudaDeepSeekV41CandidateBlocks(scores[rank], blockSize,
                            candidateTopK, ratio, startPos + first, candidate);
                    ok = ok && FastllmCudaDeepSeekV41IndexerTopK(scores[rank],
                        useCandidates ? &candidate : nullptr, topK, ratio, startPos + first,
                        std::max(1, blockSize), selected);
                    AssertInFastLLM(ok, "DeepSeekV41 prefill indexer query partition failed.");
                }
            });
            AssertInFastLLM(dispatched, "DeepSeekV41 prefill indexer dispatch failed.");
            // Separate submission from allocation/destruction on every rank.
            // No worker can enter a new allocation while peers submit NCCL.
            dispatched = MultiCudaRunDeviceCallbacks(devices, [&](int rank, int device) {
                void *out = topk.multiDeviceDatas.at(device)->cudaData;
                bool ok = FastllmNcclAllGather((int32_t *)out + (size_t)rank * perRank * width,
                    out, perRank * width, (int)DataType::INT32, device);
                if (writeCandidates) {
                    void *mask = candidates.multiDeviceDatas.at(device)->cudaData;
                    ok = ok && FastllmNcclAllGather((uint8_t *)mask + (size_t)rank * perRank * maskWidth,
                        mask, perRank * maskWidth, (int)DataType::INT8, device);
                }
                AssertInFastLLM(ok, "DeepSeekV41 prefill indexer all-gather failed.");
            });
            AssertInFastLLM(dispatched, "DeepSeekV41 prefill indexer gather dispatch failed.");
            dispatched = MultiCudaRunDeviceCallbacks(devices, [&](int rank, int) {
                if (scores[rank].cudaData &&
                    FastllmCudaFreeAfterCurrentThreadStream(scores[rank].cudaData))
                    scores[rank].cudaData = nullptr;
                else if (scores[rank].cudaData)
                    FastllmCudaSyncCurrentThreadStream();
            });
            AssertInFastLLM(dispatched, "DeepSeekV41 prefill indexer cleanup dispatch failed.");
            for (int device : devices) {
                topk.multiDeviceDatas.at(device)->Resize({1, tokens, width});
                if (writeCandidates) candidates.multiDeviceDatas.at(device)->Resize({1, tokens, maskWidth});
            }
            V41SyncRootFromReplica(topk, devices);
            if (writeCandidates) V41SyncRootFromReplica(candidates, devices);
            return true;
        }
#endif

        void V41SparseAttention(const Data &q, const Data &chunkKV, const Data *ringKV,
                                const Data *compressedKV, const Data *cmpIdx, Data &attnSink,
                                int windowSize, int startPos, float softmaxScale, Data &output) {
            DataDict datas = {
                {"q", (Data*)&q}, {"chunkKV", (Data*)&chunkKV}, {"attnSink", &attnSink}, {"output", &output}
            };
            if (ringKV != nullptr && ringKV->dims.size() == 3) {
                datas["ringKV"] = (Data*)ringKV;
            }
            if (compressedKV != nullptr && cmpIdx != nullptr && compressedKV->dims.size() == 3 &&
                compressedKV->dims[1] > 0 && cmpIdx->dims.size() == 3) {
                datas["compressedKV"] = (Data*)compressedKV;
                datas["cmpIdx"] = (Data*)cmpIdx;
            }
            V41Executor().Run("DeepSeekV41SparseAttention", datas, {{"softmaxScale", softmaxScale}},
                              {{"windowSize", windowSize}, {"startPos", startPos}});
        }

        // float KV 行 -> 量化缓存行（INT8 Data）。(quantMode, quantBlock) 与写入前的伪量化一致：
        //   1 / 32：FP8 E4M3 + UE8M0      （滑窗 KV）
        //   3 / 16：FP4 E2M1 + E4M3 scale （压缩 KV）
        //   2 / 32：FP4 E2M1 + UE8M0      （indexer key）
        void V41QuantizeKV(const Data &input, Data &output, int quantMode = 1, int quantBlock = 32) {
            V41Executor().Run("DeepSeekV41QuantizeKV", {{"input", (Data*)&input}, {"output", &output}}, {},
                              {{"quantMode", quantMode}, {"quantBlock", quantBlock}});
        }

        void V41WindowStore(const Data &chunkKV, Data &ring, int startPos, int windowSize) {
            V41Executor().Run("DeepSeekV41WindowStore", {
                {"chunk", (Data*)&chunkKV}, {"ring", &ring}
            }, {}, {{"startPos", startPos}, {"windowSize", windowSize}});
        }

        // 缓存在 axis=1 上的行容量。expansionDims 为空表示"没有富余容量"，
        // 此时容量就等于逻辑行数（而不是一个越界读到的垃圾值）。
        int V41CacheRowCapacity(const Data &cache) {
            if (cache.expansionDims.size() >= 2) {
                return cache.expansionDims[1];
            }
            if (cache.dims.size() >= 2) {
                return cache.dims[1];
            }
            return 0;
        }

        // 在 axis=1 上追加行（预扩容 + CatDirect），cache 需为 [b, cap, d]
        // 张量并行时缓存是每卡一份：扩容必须逐卡执行，root 只保留形状信息。
        void V41AppendRows(Data &cache, const Data &rows,
                           const std::vector<int> &tpDevices = std::vector<int>()) {
            const int unitLen = 256;
            if (cache.dims.size() == 0 && cache.expansionDims.size() == 0) {
                cache.dataType = rows.dataType;
                cache.UpdateUnitSize();
            }
            bool tp = false;
#ifdef USE_CUDA
            tp = !tpDevices.empty() && rows.multiDeviceData && rows.IsTensorParallelReplicated();
            if (tp && !cache.multiDeviceData) {
                if (cache.Count(0) == 0) {
                    // 空缓存：直接在每张卡上建副本（前缀缓存恢复出来的缓存在 CPU 上，
                    // 由 CopyToMultiDevices 从 CPU 广播，不能把 dataDevice 强改成 CUDA）
                    cache.dataDevice = DataDevice::CUDA;
                    cache.dataDeviceIds = tpDevices;
                }
                PrepareMultiCudaReplicatedData(cache, tpDevices, cache.Count(0) > 0);
                V41SyncRootFromReplica(cache, tpDevices);
            }
            tp = tp && cache.multiDeviceData && cache.IsTensorParallelReplicated();
#endif
            // 已有行数与需要的总容量。注意 expansionDims 可能是空的：Data::CopyFrom 在
            // 源张量没有富余容量时会 clear() 掉它（前缀缓存恢复、以及 CopyToMultiDevices
            // 的 CPU 分支建出来的副本都是这种），原来的写法直接读 expansionDims[1] 属于
            // 越界读——读到的垃圾值偏大时就不扩容，CatDirect 直接写出界。
            const int haveRows = cache.dims.size() >= 2 ? cache.dims[1] : 0;
            const int needRows = haveRows + rows.dims[1];
            while (V41CacheRowCapacity(cache) < needRows) {
                std::vector<int> newDims;
                if (cache.Count(0) == 0 || cache.dims.size() == 0) {
                    newDims = {rows.dims[0], ((rows.dims[1] - 1) / unitLen + 1) * unitLen, rows.dims[2]};
                } else {
                    newDims = cache.dims;
                    newDims[1] += std::max(((rows.dims[1] - 1) / unitLen + 1) * unitLen, cache.dims[1] / 2);
                }
#ifdef USE_CUDA
                if (tp) {
                    const int oriDevice = FastllmCudaGetDevice();
                    for (int device : tpDevices) {
                        Data *local = cache.multiDeviceDatas.at(device);
                        FastllmCudaSetDevice(device);
                        local->Expansion(newDims);
                    }
                    FastllmCudaSetDevice(oriDevice);
                    V41SyncRootFromReplica(cache, tpDevices);
                    continue;
                }
#endif
                cache.Expansion(newDims);
            }
#ifdef USE_CUDA
            // 追加前把容量 / 行宽 / 类型核一遍。越界写在 CUDA 上只会在之后某个
            // 不相关的地方报 illegal address，这里提前拦住并说清是哪一项不对。
            if (tp) {
                for (int device : tpDevices) {
                    Data *local = cache.multiDeviceDatas.at(device);
                    AssertInFastLLM(
                        local != nullptr && local->dataType == rows.dataType &&
                        (local->dims.size() == 0 || local->dims[2] == rows.dims[2]) &&
                        V41CacheRowCapacity(*local) >= needRows,
                        "DeepSeekV41: TP cache replica on device " + std::to_string(device) +
                        " can't take " + std::to_string(rows.dims[1]) + " more rows of width " +
                        std::to_string(rows.dims[2]) + " (capacity " +
                        std::to_string(V41CacheRowCapacity(*local)) + " rows, need " +
                        std::to_string(needRows) + ").");
                }
            }
#endif
            CatDirect(cache, rows, 1);
        }

#ifdef USE_CUDA
        // Reserve without appending: copies are submitted together on the
        // persistent rank streams after all request metadata has been checked.
        void V41ReserveDecodeRows(Data &data, DataType type, int width, int required,
                                  const std::vector<int> &devices) {
            if (data.dims.empty()) {
                data.dataType = type;
                data.UpdateUnitSize();
                data.Resize({1, 0, width});
            }
            if (!data.multiDeviceData) {
                if (data.Count(0) == 0) {
                    data.dataDevice = DataDevice::CUDA;
                    data.dataDeviceIds = devices;
                }
                PrepareMultiCudaReplicatedData(data, devices, data.Count(0) > 0);
            }
            const int original = FastllmCudaGetDevice();
            for (int device : devices) {
                Data &local = *data.multiDeviceDatas.at(device);
                if (V41CacheRowCapacity(local) < required) {
                    FastllmCudaSetDevice(device);
                    int capacity = std::max(required, V41CacheRowCapacity(local) * 3 / 2);
                    if (required > 2) capacity = (capacity + 255) / 256 * 256;
                    local.Expansion({1, capacity, width});
                }
            }
            FastllmCudaSetDevice(original);
            V41SyncRootFromReplica(data, devices);
        }

        bool V41CompressDecodeBatch(const std::vector<DeepSeekV41Segment> &segments,
                int layer, int ratio, Data &rawKV, Data &rawScore, Data &norm,
                float eps, const V41RopeParams &rope, bool indexSource,
                const std::function<void(Data &, Data &)> &projectIndex,
                const std::vector<int> &devices, DeepSeekV41DecodeWorkspace &scratch) {
            auto replicated = [&](const Data &data) {
                if (!data.multiDeviceData || !data.IsTensorParallelReplicated()) return false;
                for (int d : devices) {
                    auto it = data.multiDeviceDatas.find(d);
                    if (it == data.multiDeviceDatas.end() || !it->second ||
                        (data.Count(0) && !it->second->cudaData)) return false;
                }
                return true;
            };
            if (devices.empty() || ratio < 1 || ratio > 2 ||
                rawKV.dataType != DataType::FLOAT32 || rawKV.dims.size() != 3 ||
                rawKV.dims[2] != 512 || !replicated(rawKV) ||
                (ratio > 1 && (rawScore.dataType != DataType::FLOAT32 ||
                              rawScore.dims != rawKV.dims || !replicated(rawScore)))) return false;
            std::vector<int> ready, rowOf(segments.size(), -1);
            for (int i = 0; i < (int)segments.size(); ++i) {
                const auto &seg = segments[i];
                auto &cache = seg.state->layers[layer];
                if (seg.seqlen != 1 || seg.startPos <= 0 || seg.spec ||
                    cache.rawTail != seg.startPos % ratio ||
                    cache.compressedBlocks != seg.startPos / ratio) return false;
                for (Data *data : {&cache.compressedKV, &cache.indexK,
                                   &cache.rawTailKV, &cache.rawTailScore}) {
                    // Count(0) is 1 for an unshaped Data. Prefill Split also
                    // legitimately returns a root-only tail. Neither means
                    // that the request cannot use the batched decode path.
                    if (data->dims.empty()) continue;
                    if (data->dims.size() != 3) return false;
                    if (data->dims[1] == 0 || replicated(*data)) continue;
                    const bool payload = data->dataDevice == DataDevice::CUDA ?
                        data->cudaData != nullptr : data->cpuData != nullptr;
                    if (data->IsTensorParallelSharded() || !payload) return false;
                    PrepareMultiCudaReplicatedData(*data, devices, true);
                    if (!replicated(*data)) return false;
                }
                if (cache.rawTail + 1 == ratio) {
                    rowOf[i] = (int)ready.size();
                    ready.push_back(i);
                }
            }
            const int rows = (int)ready.size();
            for (int i = 0; i < (int)segments.size(); ++i) {
                auto &c = segments[i].state->layers[layer];
                if (rowOf[i] < 0) {
                    V41ReserveDecodeRows(c.rawTailKV, DataType::FLOAT32, 512, ratio, devices);
                    V41ReserveDecodeRows(c.rawTailScore, DataType::FLOAT32, 512, ratio, devices);
                } else {
                    V41ReserveDecodeRows(c.compressedKV, DataType::BFLOAT16, 512, c.compressedBlocks + 1, devices);
                    if (indexSource) V41ReserveDecodeRows(c.indexK, DataType::BFLOAT16, 128, c.compressedBlocks + 1, devices);
                }
            }
            // Scratch remains alive until this forward's TP streams drain;
            // graph workspaces retain it across tokens. Every rank reuses it
            // only after its preceding cache append on the same worker stream.
            auto &joined = scratch.compressJoined;
            auto &scores = scratch.compressScores;
            auto &latents = scratch.compressLatents;
            auto &keys = scratch.compressKeys;
            if (rows) {
                V41AllocLike(joined, DataType::FLOAT32, {rows, ratio, 512}, rawKV, devices);
                if (ratio > 1) V41AllocLike(scores, DataType::FLOAT32, {rows, ratio, 512}, rawKV, devices);
                V41AllocLike(latents, DataType::BFLOAT16, {rows, 1, 512}, rawKV, devices);
                PrepareMultiCudaReplicatedData(norm, devices, true);
            }
            MultiCudaRunDeviceCallbacks(devices, [&](int, int device) {
                std::vector<void *> destinations;
                std::vector<const void *> sources;
                std::vector<size_t> sizes;
                auto copy = [&](void *dst, const void *src, size_t bytes) {
                    destinations.push_back(dst); sources.push_back(src); sizes.push_back(bytes);
                };
                for (int i = 0; i < (int)segments.size(); ++i) {
                    auto &c = segments[i].state->layers[layer];
                    const size_t offset = (size_t)segments[i].offset * 512 * sizeof(float);
                    for (int part = 0; part < (ratio > 1 ? 2 : 1); ++part) {
                        Data &source = part ? rawScore : rawKV;
                        Data &tail = part ? c.rawTailScore : c.rawTailKV;
                        const void *token = (const uint8_t *)source.multiDeviceDatas.at(device)->cudaData + offset;
                        if (rowOf[i] < 0) {
                            copy((uint8_t *)tail.multiDeviceDatas.at(device)->cudaData +
                                 (size_t)c.rawTail * 512 * sizeof(float), token, 512 * sizeof(float));
                        } else {
                            Data &target = part ? scores : joined;
                            auto *dst = (uint8_t *)target.multiDeviceDatas.at(device)->cudaData +
                                (size_t)rowOf[i] * ratio * 512 * sizeof(float);
                            if (c.rawTail) copy(dst, tail.multiDeviceDatas.at(device)->cudaData,
                                                (size_t)c.rawTail * 512 * sizeof(float));
                            copy(dst + (size_t)c.rawTail * 512 * sizeof(float), token, 512 * sizeof(float));
                        }
                    }
                }
                AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(
                    destinations.data(), sources.data(), sizes.data(), sizes.size()),
                    "DeepSeekV41 decode compressor gather failed.");
                if (rows) {
                    AssertInFastLLM(FastllmCudaDeepSeekV41Compress(*joined.multiDeviceDatas.at(device),
                        ratio > 1 ? scores.multiDeviceDatas.at(device) : nullptr,
                        *norm.multiDeviceDatas.at(device), ratio, eps, *latents.multiDeviceDatas.at(device)),
                        "DeepSeekV41 batched compressor failed.");
                }
            });
            if (rows) {
                latents.Reshape({1, rows, 512});
                if (indexSource) projectIndex(latents, keys);
                auto &latentViews = scratch.compressLatentViews;
                auto &keyViews = scratch.compressKeyViews;
                latentViews.resize(rows);
                keyViews.resize(rows);
                for (int row = 0; row < rows; ++row) {
                    AssertInFastLLM(V41DecodeSliceView(latentViews[row], latents, row, devices), "compressor latent view failed");
                    if (indexSource) AssertInFastLLM(V41DecodeSliceView(keyViews[row], keys, row, devices), "compressor key view failed");
                }
                const bool disableQuant = V41EnvFlag("FASTLLM_DSV41_DISABLE_FAKE_QUANT");
                MultiCudaRunDeviceCallbacks(devices, [&](int, int device) {
                    std::vector<FastllmCudaV41DecodeTransform> transforms;
                    std::vector<void *> destinations;
                    std::vector<const void *> sources;
                    std::vector<size_t> sizes;
                    for (int row = 0; row < rows; ++row) {
                        auto &c = segments[ready[row]].state->layers[layer];
                        auto store = [&](Data &view, Data &cache, int mode, int block) {
                            Data &local = *view.multiDeviceDatas.at(device);
                            transforms.push_back({&local, nullptr, c.compressedBlocks * ratio,
                                disableQuant ? 0 : mode, false, block});
                            const size_t bytes = (size_t)local.dims.back() * 2;
                            destinations.push_back((uint8_t *)cache.multiDeviceDatas.at(device)->cudaData +
                                                   (size_t)c.compressedBlocks * bytes);
                            sources.push_back(local.cudaData); sizes.push_back(bytes);
                        };
                        if (indexSource) store(keyViews[row], c.indexK, 2, 32);
                        store(latentViews[row], c.compressedKV, 3, 16);
                    }
                    if (!FastllmCudaV41DecodeTransformsBatch(transforms, rope.ropeDim, rope.base,
                            rope.originalSeqLen, rope.factor, rope.betaFast, rope.betaSlow, 1)) {
                        for (const auto &t : transforms)
                            AssertInFastLLM(FastllmCudaDeepSeekV41RotaryQuant(*t.input, rope.ropeDim, rope.base,
                                t.startPos, 1, false, rope.originalSeqLen, rope.factor, rope.betaFast, rope.betaSlow,
                                t.quantMode, t.input->dims.back(), t.quantBlock), "DeepSeekV41 compressor RoPE failed.");
                    }
                    AssertInFastLLM(FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(
                        destinations.data(), sources.data(), sizes.data(), sizes.size()),
                        "DeepSeekV41 batched compressed cache append failed.");
                });
            }
            auto setRows = [&](Data &data, int n) {
                if (data.dims.empty()) return;
                auto dims = data.dims; dims[1] = n;
                for (int d : devices) data.multiDeviceDatas.at(d)->Resize(dims);
                V41SyncRootFromReplica(data, devices);
            };
            for (int i = 0; i < (int)segments.size(); ++i) {
                auto &c = segments[i].state->layers[layer];
                c.rawTail = (c.rawTail + 1) % ratio;
                if (rowOf[i] >= 0) {
                    ++c.compressedBlocks;
                    setRows(c.compressedKV, c.compressedBlocks);
                    if (indexSource) setRows(c.indexK, c.compressedBlocks);
                }
                setRows(c.rawTailKV, c.rawTail);
                setRows(c.rawTailScore, c.rawTail);
            }
            return true;
        }
#endif

        // 把 [1, n, w] 的分块结果写进 [1, N, w] 的整段缓冲的第 offset 行
        void V41WriteRows(Data &dst, const Data &src, int offset,
                          const std::vector<int> &tpDevices = std::vector<int>()) {
            const uint64_t rowBytes = (uint64_t)src.dims[2] * src.unitSize / src.unitSizeDiv;
            const uint64_t bytes = (uint64_t)src.dims[1] * rowBytes;
            if (bytes == 0) {
                return;
            }
#ifdef USE_CUDA
            if (!tpDevices.empty() && dst.multiDeviceData && dst.IsTensorParallelReplicated() &&
                src.multiDeviceData && src.IsTensorParallelReplicated()) {
                const int oriDevice = FastllmCudaGetDevice();
                for (int device : tpDevices) {
                    Data *dstLocal = dst.multiDeviceDatas.at(device);
                    Data *srcLocal = src.multiDeviceDatas.at(device);
                    FastllmCudaSetDevice(device);
                    FastllmCudaCopyFromDeviceToDevice(
                        (uint8_t*)dstLocal->cudaData + (uint64_t)offset * rowBytes,
                        (uint8_t*)srcLocal->cudaData, bytes);
                }
                FastllmCudaSetDevice(oriDevice);
                return;
            }
#endif
            if (dst.dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
                FastllmCudaCopyFromDeviceToDevice((uint8_t*)dst.cudaData + (uint64_t)offset * rowBytes,
                                                  (uint8_t*)src.cudaData, bytes);
#endif
            } else {
                memcpy(dst.cpuData + (uint64_t)offset * rowBytes, src.cpuData, bytes);
            }
        }

        // 调试：把张量以 float32 原始字节写到文件（FASTLLM_DSV41_DUMP_DIR）
        void V41DumpTensor(const Data &data, const std::string &name) {
            const char *dir = std::getenv("FASTLLM_DSV41_DUMP_DIR");
            if (dir == nullptr || dir[0] == '\0') {
                return;
            }
            Data cpu;
#ifdef USE_CUDA
            if (data.multiDeviceData && !data.IsTensorParallelReplicated()) {
                // Sharded roots have no payload. Record each owned shard
                // instead of issuing an invalid copy from the empty root.
                for (const auto &part : data.multiDeviceDatas) {
                    if (part.second != nullptr) {
                        V41DumpTensor(*part.second, name + "_rank" + std::to_string(part.first));
                    }
                }
                return;
            }
            // 复制布局下 root 只有形状信息，CopyFrom 会导出一份垃圾——多卡排查时
            // 浮点张量会假发散（相关系数接近 0），非常容易误导。从副本取数。
            if (data.multiDeviceData && data.IsTensorParallelReplicated()) {
                std::vector<int> replicaDevices;
                for (const auto &it : data.multiDeviceDatas) {
                    if (it.second != nullptr) {
                        replicaDevices.push_back(it.first);
                    }
                }
                V41ReplicaToCpu(cpu, data, replicaDevices);
            } else {
                cpu.CopyFrom(data);
            }
#else
            cpu.CopyFrom(data);
#endif
            cpu.ToDevice(DataDevice::CPU);
            Data f32;
            if (cpu.dataType == DataType::FLOAT32) {
                f32.CopyFrom(cpu);
            } else if (cpu.dataType == DataType::INT32 || cpu.dataType == DataType::INT8) {
                f32 = Data(DataType::FLOAT32, cpu.dims);
                f32.Allocate();
                for (uint64_t i = 0; i < cpu.Count(0); i++) {
                    ((float*)f32.cpuData)[i] = cpu.dataType == DataType::INT32 ?
                        (float)((int32_t*)cpu.cpuData)[i] : (float)((uint8_t*)cpu.cpuData)[i];
                }
            } else {
                ToDataType(cpu, f32, DataType::FLOAT32);
            }
            f32.ToDevice(DataDevice::CPU);
            if (f32.cpuData == nullptr) {
                return;
            }
            std::string path = std::string(dir) + "/" + name + ".bin";
            FILE *fo = fopen(path.c_str(), "wb");
            if (fo != nullptr) {
                fwrite(f32.cpuData, 1, f32.GetBytes(), fo);
                fclose(fo);
            }
        }

        inline float V41Softplus(float x) {
            if (x > 20.0f) {
                return x;
            }
            if (x < -20.0f) {
                return std::exp(x);
            }
            return std::log1p(std::exp(x));
        }

        // ---------------- Engram 表 ----------------

        struct V41EngramTable {
            int64_t rows = 0;
            int dim = 0;
            int scaleBlock = 32;
            Data *diskWeight = nullptr;
            Data diskScale;
            const uint8_t *data = nullptr;      // FP8 E4M3, [rows, dim]
            const uint8_t *scale = nullptr;     // UE8M0, [rows, dim / scaleBlock]
            std::vector<uint8_t> dataStorage;
            std::vector<uint8_t> scaleStorage;
            void *mmapData = nullptr;
            size_t mmapDataLen = 0;
            void *mmapScale = nullptr;
            size_t mmapScaleLen = 0;

#ifdef USE_CUDA
            void *gpuTable = nullptr;
            std::vector<int> gpuDevices;
#endif

            ~V41EngramTable() {
#ifdef USE_CUDA
                FastllmCudaV41FreeEngram(gpuTable);
#endif
#if !defined(_WIN32) && !defined(_WIN64)
                if (mmapData != nullptr) {
                    munmap(mmapData, mmapDataLen);
                }
                if (mmapScale != nullptr) {
                    munmap(mmapScale, mmapScaleLen);
                }
#endif
            }
        };

        struct V41SafeTensorInfo {
            std::string fileName;
            std::string dtype;
            std::vector<int64_t> shape;
            uint64_t offset = 0;   // 绝对文件偏移
            uint64_t bytes = 0;
        };

        bool V41FindSafeTensor(const std::string &dir, const std::string &tensorName, V41SafeTensorInfo &info) {
            // 1. 通过 index 定位文件；没有 index 时尝试 model.safetensors
            std::vector<std::string> candidates;
            {
                std::ifstream fin(dir + "model.safetensors.index.json");
                if (fin.good()) {
                    std::stringstream ss;
                    ss << fin.rdbuf();
                    std::string err;
                    auto json = json11::Json::parse(ss.str(), err);
                    if (err.empty()) {
                        auto file = json["weight_map"][tensorName];
                        if (file.is_string()) {
                            candidates.push_back(dir + file.string_value());
                        }
                    }
                }
            }
            if (candidates.empty()) {
                candidates.push_back(dir + "model.safetensors");
            }
            for (auto &fileName : candidates) {
                std::ifstream fin(fileName, std::ios::binary);
                if (!fin.good()) {
                    continue;
                }
                uint64_t headerLen = 0;
                fin.read((char*)&headerLen, sizeof(headerLen));
                if (!fin.good() || headerLen == 0 || headerLen > (1ULL << 31)) {
                    continue;
                }
                std::string header(headerLen, '\0');
                fin.read(&header[0], headerLen);
                std::string err;
                auto json = json11::Json::parse(header, err);
                if (!err.empty()) {
                    continue;
                }
                auto item = json[tensorName];
                if (!item.is_object()) {
                    continue;
                }
                info.fileName = fileName;
                info.dtype = item["dtype"].string_value();
                info.shape.clear();
                for (auto &d : item["shape"].array_items()) {
                    info.shape.push_back((int64_t)d.number_value());
                }
                uint64_t st = (uint64_t)item["data_offsets"][0].number_value();
                uint64_t end = (uint64_t)item["data_offsets"][1].number_value();
                info.offset = 8 + headerLen + st;
                info.bytes = end - st;
                return true;
            }
            return false;
        }

        void V41ReadFileRange(const std::string &fileName, uint64_t offset, uint8_t *dst, uint64_t bytes) {
            FILE *fi = fopen(fileName.c_str(), "rb");
            AssertInFastLLM(fi != nullptr, "DeepSeekV41: can't open " + fileName);
#if defined(_WIN32) || defined(_WIN64)
            _fseeki64(fi, offset, SEEK_SET);
#else
            fseeko(fi, (off_t)offset, SEEK_SET);
#endif
            const uint64_t chunk = 256ULL << 20;
            uint64_t done = 0;
            while (done < bytes) {
                uint64_t cur = std::min(chunk, bytes - done);
                uint64_t got = fread(dst + done, 1, cur, fi);
                AssertInFastLLM(got == cur, "DeepSeekV41: short read from " + fileName);
                done += cur;
            }
            fclose(fi);
        }

        bool V41IsPrime(int64_t x) {
            if (x < 2) {
                return false;
            }
            if (x % 2 == 0) {
                return x == 2;
            }
            for (int64_t d = 3; d * d <= x; d += 2) {
                if (x % d == 0) {
                    return false;
                }
            }
            return true;
        }

        inline float V41E8M0ToFloat(uint8_t v) {
            if (v == 0xFF) {
                return std::numeric_limits<float>::quiet_NaN();
            }
            return std::ldexp(1.0f, (int)v - 127);
        }
    }

    // ==================== 构造 / 参数 ====================

    DeepSeekV41Model::DeepSeekV41Model() {
        this->model_type = "deepseek_v41";
        this->model_struct = "deepseek_v41";
        this->canDoBatchForward = true;      // 多请求 decode 共享一次前向（见 ForwardSegments）
        this->canDoConcurrentForward = true;
        this->defaultChunkedPrefillSize = 4096;

        weight.embeddingNames.clear();
        weight.embeddingNames.insert("embed.weight");
        weight.linearNames = {
            "head.weight",
            "layers.*.attn.wq_a.weight", "layers.*.attn.wq_b.weight",
            "layers.*.attn.wkv.weight",
            "layers.*.attn.wo_a.weight", "layers.*.attn.wo_b.weight",
            "layers.*.attn.indexer.wq_b.weight",
            "layers.*.attn.indexer.weights_proj.weight",
            "layers.*.attn.indexer.wk.weight",
            "layers.*.attn.compressor.wkv.weight",
            "layers.*.attn.compressor.wgate.weight",
            "layers.*.engram.wkv.weight",
            "layers.*.ffn.gate.weight",
            "layers.*.ffn.experts.*.w1.weight",
            "layers.*.ffn.experts.*.w2.weight",
            "layers.*.ffn.experts.*.w3.weight",
            "layers.*.ffn.shared_experts.w1.weight",
            "layers.*.ffn.shared_experts.w2.weight",
            "layers.*.ffn.shared_experts.w3.weight",
            // DSpark 草稿层（mtp.*，见 src/models/deepseekv41_dspark.cpp）
            "mtp.*.attn.wq_a.weight", "mtp.*.attn.wq_b.weight",
            "mtp.*.attn.wkv.weight",
            "mtp.*.attn.wo_a.weight", "mtp.*.attn.wo_b.weight",
            "mtp.*.ffn.gate.weight",
            "mtp.*.ffn.experts.*.w1.weight",
            "mtp.*.ffn.experts.*.w2.weight",
            "mtp.*.ffn.experts.*.w3.weight",
            "mtp.*.ffn.shared_experts.w1.weight",
            "mtp.*.ffn.shared_experts.w2.weight",
            "mtp.*.ffn.shared_experts.w3.weight",
            "mtp.*.main_proj.weight",
            "mtp.*.markov_head.head.weight",
            "mtp.*.confidence_head.proj.weight",
            "vision.patch_embed.proj.weight",
            "vision.blocks.*.attn.wqkv.weight", "vision.blocks.*.attn.wo.weight",
            "vision.blocks.*.mlp.w1.weight", "vision.blocks.*.mlp.w2.weight",
            "aligner.w1.weight", "aligner.w2.weight",
        };
    }

    DeepSeekV41Model::~DeepSeekV41Model() {
        ShutdownRuntime();
        DsparkReportStats();
        ReleaseMoeCudaCache(weights);
        {
            std::lock_guard<std::mutex> guard(v41StateMutex);
            v41States.clear();
        }
        engramTables.clear();
    }

    void DeepSeekV41Model::InitParams() {
        // 1. 把 text_config.* 展平到顶层（HF config 为嵌套结构）
        {
            std::vector<std::pair<std::string, std::string> > flattened;
            for (auto &it : this->weight.dicts) {
                if (V41StartsWith(it.first, "text_config.")) {
                    flattened.push_back({it.first.substr(strlen("text_config.")), it.second});
                }
            }
            for (auto &kv : flattened) {
                if (!V41HasKey(this->weight, kv.first)) {
                    this->weight.AddDict(kv.first, kv.second);
                }
            }
            if (!V41HasKey(this->weight, "rope_scaling.type") && V41HasKey(this->weight, "rope_scaling.rope_type")) {
                this->weight.AddDict("rope_scaling.type", this->weight.dicts["rope_scaling.rope_type"]);
            }
            // V4.1 没有 hash 路由层
            if (!V41HasKey(this->weight, "num_hash_layers")) {
                this->weight.AddDict("num_hash_layers", "0");
            }
        }

        // 2. 复用 V4 的基础解析（尺寸、RoPE、MoE 合并规则、特殊权重注册等）
        DeepSeekV4Model::InitParams();
        this->rms_norm_eps = V41DictFloat(this->weight, "rms_norm_eps", this->rms_norm_eps);

        // 3. V4.1 专有参数
        kv_source_layer_ids = V41DictIntArray(this->weight, "kv_source_layer_ids");
        index_source_layer_ids = V41DictIntArray(this->weight, "index_source_layer_ids");
        candidate_source_layer_id = V41DictInt(this->weight, "candidate_source_layer_id", -1);
        candidate_topk_blocks = V41DictInt(this->weight, "candidate_topk_blocks", 0);
        candidate_block_size = V41DictInt(this->weight, "candidate_block_size", 0);
        gate_temp = V41DictFloat(this->weight, "gate_temp", 1.0f);
        image_token_id = V41DictInt(this->weight, "image_token_id", -1);

        engram_layer_ids = V41DictIntArray(this->weight, "engram_layer_ids");
        engram_num_embeddings = V41DictInt64Array(this->weight, "engram_num_embeddings");
        engram_max_ngram_size = V41DictInt(this->weight, "engram_max_ngram_size", 4);
        engram_vocab_size = V41DictInt(this->weight, "engram_vocab_size", 0);
        engram_n_heads = V41DictInt(this->weight, "engram_n_heads", 0);
        engram_head_dim = V41DictInt(this->weight, "engram_head_dim", 0);
        engram_pad_token_id = V41DictInt(this->weight, "engram_pad_token_id", 2);
        engram_compressed_vocab_size = V41DictInt(this->weight, "engram_compressed_vocab_size", 0);

        // 4. 每层的跨层共享关系
        kvSourceOf.assign(block_cnt, -1);
        indexSourceOf.assign(block_cnt, -1);
        isKvSource.assign(block_cnt, 0);
        isIndexSource.assign(block_cnt, 0);
        int curKv = -1, curIndex = -1;
        for (int layer = 0; layer < block_cnt; layer++) {
            int ratio = compress_ratios.size() > (size_t)layer ? compress_ratios[layer] : 0;
            if (V41Contains(kv_source_layer_ids, layer)) {
                curKv = layer;
                isKvSource[layer] = 1;
            }
            if (V41Contains(index_source_layer_ids, layer)) {
                curIndex = layer;
                isIndexSource[layer] = 1;
            }
            if (ratio > 0) {
                AssertInFastLLM(curKv >= 0 && curIndex >= 0 &&
                                compress_ratios[curKv] == ratio,
                                "DeepSeekV41: layer " + std::to_string(layer) +
                                " uses compressed attention but has no matching kv/index source layer.");
                kvSourceOf[layer] = curKv;
                indexSourceOf[layer] = curIndex;
            }
            AssertInFastLLM(ratio == 0 || ratio == 1 || ratio == 2,
                            "DeepSeekV41: unsupported compress ratio " + std::to_string(ratio));
        }
        for (int layer : kv_source_layer_ids) {
            AssertInFastLLM(layer >= 0 && layer < block_cnt && V41Contains(index_source_layer_ids, layer),
                            "DeepSeekV41: every kv source layer must also be an index source layer.");
        }

        decoderSwaTailLayer = -1;
        if (V41EnvFlag("FASTLLM_DSV41_DECODER_SWA_BOUNDED_REPLAY")) {
            AssertInFastLLM(!kv_source_layer_ids.empty() && window_size > 0,
                            "DeepSeekV41: decoder SWA bounded replay needs KV sources and a sliding window.");
            decoderSwaTailLayer = *std::max_element(kv_source_layer_ids.begin(), kv_source_layer_ids.end()) + 1;
            for (int layer = decoderSwaTailLayer; layer < block_cnt; layer++) {
                AssertInFastLLM(compress_ratios[layer] <= 1,
                                "DeepSeekV41: decoder SWA bounded replay requires late layers with ratio 0 or 1.");
            }
            printf("[Fastllm] DeepSeek-V4.1 approximate decoder SWA bounded replay: layer %d onward, window %d\n",
                   decoderSwaTailLayer, window_size);
        }

        // 5. 这些小权重保持源精度
        for (int i = 0; i < block_cnt; i++) {
            std::string pre = "layers." + std::to_string(i);
            this->cantQuantLinears.insert(pre + ".attn.compressor.wkv.weight");
            this->cantQuantLinears.insert(pre + ".attn.compressor.wgate.weight");
            this->cantQuantLinears.insert(pre + ".attn.indexer.wk.weight");
            this->cantQuantLinears.insert(pre + ".attn.indexer.weights_proj.weight");
            this->cantQuantLinears.insert(pre + ".ffn.gate.weight");
        }

        // 稀疏注意力支持不足一个 tile 的 head 分片；wo_a 仍要求区间对齐到 o_group。
        // 不满足时撤销 DeepSeekV4Model::InitParams
        // 注册的注意力 TP 权重，让它们整块加载，前向里注意力也退回复制布局。
        if (V41DeviceMapUsesMultiCuda(this->deviceMap)) {
            const int ranks = V41MultiCudaRankCount(this->deviceMap);
            const bool aligned =
                ranks > 1 && o_groups > 0 &&
                num_attention_heads % ranks == 0 &&
                num_attention_heads % o_groups == 0 &&
                (num_attention_heads / ranks) % (num_attention_heads / o_groups) == 0;
            if (!aligned) {
                // 注意力切不开时整体退回单卡：只切 FFN / head 的"半 TP"没有可验证的
                // 收益（注意力仍要在每张卡上各算一份），而视觉编码器也还不是 TP 感知的。
                for (int i = 0; i < block_cnt; i++) {
                    for (const char *suffix : {".attn.wq_b.weight", ".attn.wo_a.weight",
                                               ".attn.wo_b.weight"}) {
                        std::string name = "layers." + std::to_string(i) + suffix;
                        this->specialWeights.erase(name);
                        this->specialWeightLayerIds.erase(name);
                    }
                }
                this->specialWeights.erase("head.weight");
                this->specialWeightLayerIds.erase("head.weight");
                std::string fallbackDevice = "cuda";
                for (const auto &it : this->deviceMap) {
                    if (!V41DeviceSpecUsesType(it.first, "multicuda")) {
                        continue;
                    }
                    size_t pos = it.first.find(':');
                    if (pos != std::string::npos) {
                        std::string spec = it.first.substr(pos + 1);
                        size_t comma = spec.find(',');
                        std::string first = comma == std::string::npos ? spec : spec.substr(0, comma);
                        size_t slash = first.find('/');
                        if (slash != std::string::npos) {
                            first = first.substr(0, slash);
                        }
                        if (!first.empty()) {
                            fallbackDevice = "cuda:" + first;
                        }
                    }
                    break;
                }
                printf("[Fastllm] DeepSeek-V4.1 tensor parallel needs equal head shards "
                       "aligned to o_groups (got %d heads, %d ranks, o_groups=%d); "
                       "falling back to %s.\n",
                       num_attention_heads, ranks, o_groups, fallbackDevice.c_str());
                fflush(stdout);
                this->deviceMap = std::map<std::string, int>{{fallbackDevice, 1}};
                if (V41DeviceMapUsesMultiCuda(this->moeDeviceMap)) {
                    this->moeDeviceMap = this->deviceMap;
                }
                if (V41DeviceMapUsesMultiCuda(this->layeredMoeDeviceMap)) {
                    this->layeredMoeDeviceMap = this->deviceMap;
                }
            }
        }

        InitExecutionMode();
        LoadEngramMeta();
        InitVisionParams();
        InitDsparkParams();

        printf("[Fastllm] DeepSeek-V4.1: %d layers, %d experts (top-%d), kv sources = %d, index sources = %d, "
               "engram layers = %d%s, vision layers = %d\n",
               block_cnt, num_experts, num_experts_per_tok, (int)kv_source_layer_ids.size(),
               (int)index_source_layer_ids.size(), (int)engram_layer_ids.size(),
               engramMeta.loaded ? "" : " (engram meta NOT loaded)", vision_n_layers);
        fflush(stdout);
    }

    void DeepSeekV41Model::InitExecutionMode() {
        std::vector<std::string> placements;
        v41MoeCudaGraphCompatible = true;
        const std::string defaultDevice = V41Executor().GetFirstDeviceType();
        for (int layer = 0; layer < block_cnt; ++layer) {
            auto main = SelectDeviceFromMap(deviceMap, layer + 1, block_cnt);
            if (main.empty()) main = defaultDevice;
            auto moe = SelectMoeDeviceForLayer(layer);
            if (moe.empty()) moe = main;
            placements.push_back(main);
            placements.push_back(moe);
            v41MoeCudaGraphCompatible &= !V41DeviceSpecUsesType(moe, "disk");
        }
        const char *requested = std::getenv("FASTLLM_EXECUTION_MODE");
        executionMode = ResolveExecutionMode(requested, placements);
        printf("[Fastllm] FASTLLM_EXECUTION_MODE=%s -> %s (DeepSeek-V4.1)\n",
               NormalizeExecutionMode(requested ? requested : "auto").c_str(), ExecutionModeName(executionMode));
        if (!v41MoeCudaGraphCompatible) {
            // Disk experts load/release temporary weights during a forward.
            // They do not satisfy the persistent graph workspace contract.
            // Engram disk lookups are independent and remain graph-compatible.
            printf("[Fastllm] DeepSeek-V4.1: CUDA Graph bypassed for disk expert weights; using eager execution.\n");
        }
        fflush(stdout);
    }

    // ==================== Engram 元数据 ====================

    void DeepSeekV41Model::BuildEngramPrimes() {
        // 与 engram.py::EngramLayout.from_args 一致：所有层共享一个 seen 集合，
        // 逐层、逐 n-gram 大小、逐 head 取"下一个未使用的素数"。
        engramMeta.primes.clear();
        engramMeta.offsets.clear();
        std::set<int64_t> seen;
        for (size_t l = 0; l < engram_layer_ids.size(); l++) {
            std::vector<int64_t> flat;
            for (int n = 0; n < engram_max_ngram_size - 1; n++) {
                int64_t current = (int64_t)engram_vocab_size - 1;
                for (int h = 0; h < engram_n_heads; h++) {
                    int64_t candidate = current + 1;
                    while (!V41IsPrime(candidate) || seen.count(candidate)) {
                        candidate++;
                    }
                    seen.insert(candidate);
                    current = candidate;
                    flat.push_back(candidate);
                }
            }
            std::vector<int64_t> offsets(flat.size(), 0);
            int64_t total = 0;
            for (size_t i = 0; i < flat.size(); i++) {
                offsets[i] = total;
                total += flat[i];
            }
            if (l < engram_num_embeddings.size()) {
                AssertInFastLLM(total == engram_num_embeddings[l],
                                "DeepSeekV41: engram prime layout mismatch (layer " + std::to_string(l) +
                                ": " + std::to_string(total) + " vs " + std::to_string(engram_num_embeddings[l]) + ").");
            }
            engramMeta.primes.push_back(flat);
            engramMeta.offsets.push_back(offsets);
        }
    }

    void DeepSeekV41Model::LoadEngramMeta() {
        engramMeta.loaded = false;
        if (engram_layer_ids.empty()) {
            return;
        }
        std::string metaPath;
        if (V41HasKey(this->weight, "engram_meta_path")) {
            metaPath = this->weight.dicts["engram_meta_path"];
        } else if (V41HasKey(this->weight, "model_directory")) {
            metaPath = this->weight.dicts["model_directory"] + "engram_meta.json";
            if (!FileExists(metaPath)) {
                // 与 Python ensure_engram_meta 的只读模型目录回退路径一致。
#if defined(_WIN32) || defined(_WIN64)
                const char *homeDir = std::getenv("USERPROFILE");
#else
                const char *homeDir = std::getenv("HOME");
                if (homeDir == nullptr) {
                    const auto *user = getpwuid(getuid());
                    homeDir = user != nullptr ? user->pw_dir : nullptr;
                }
#endif
                if (homeDir != nullptr) {
                    const auto modelName = fs::absolute(metaPath)
                        .lexically_normal().parent_path().filename();
                    const auto cachedPath = fs::path(homeDir) / ".cache" / "fastllm" /
                        "engram" / (modelName.string() + "_engram_meta.json");
                    if (FileExists(cachedPath.string())) {
                        metaPath = cachedPath.string();
                    }
                }
            }
        }
        if (metaPath.empty()) {
            return;
        }
        std::ifstream fin(metaPath);
        if (!fin.good()) {
            printf("[Fastllm] DeepSeek-V4.1: engram meta file not found: %s\n", metaPath.c_str());
            return;
        }
        std::stringstream ss;
        ss << fin.rdbuf();
        std::string err;
        auto json = json11::Json::parse(ss.str(), err);
        AssertInFastLLM(err.empty() && json.is_object(), "DeepSeekV41: failed to parse engram meta " + metaPath);
        engramMeta.tokenMap.clear();
        for (auto &v : json["token_map"].array_items()) {
            engramMeta.tokenMap.push_back(v.int_value());
        }
        engramMeta.compressedVocabSize = json["compressed_vocab_size"].int_value();
        AssertInFastLLM(engram_compressed_vocab_size == 0 ||
                        engramMeta.compressedVocabSize == engram_compressed_vocab_size,
                        "DeepSeekV41: engram compressed vocab size mismatch (" +
                        std::to_string(engramMeta.compressedVocabSize) + " vs " +
                        std::to_string(engram_compressed_vocab_size) + ").");
        engramMeta.multipliers.clear();
        for (auto &row : json["multipliers"].array_items()) {
            std::vector<int64_t> values;
            for (auto &v : row.array_items()) {
                // JSON 数字用 double 存放，multiplier 上界约 2^63 / vocab / 2，精度不够；
                // 因此 Python 侧以字符串写出，这里兼容两种写法
                if (v.is_string()) {
                    values.push_back(std::strtoll(v.string_value().c_str(), nullptr, 10));
                } else {
                    values.push_back((int64_t)v.number_value());
                }
            }
            engramMeta.multipliers.push_back(values);
        }
        AssertInFastLLM(engramMeta.multipliers.size() == engram_layer_ids.size() &&
                        (int)engramMeta.tokenMap.size() > engram_pad_token_id,
                        "DeepSeekV41: engram meta is incomplete.");
        engramMeta.padCompressedId = engramMeta.tokenMap[engram_pad_token_id];
        BuildEngramPrimes();
        engramMeta.loaded = true;
    }

    // ==================== 权重映射 ====================

    std::map<std::string, std::vector<std::pair<std::string, DataType> > >
    DeepSeekV41Model::GetTensorMap(const std::vector<std::string> &tensorNames) {
        std::map<std::string, std::vector<std::pair<std::string, DataType> > > result;
        std::vector<std::string> ordinary;
        // engram.wkv 在真实权重里本来就是 F8_E4M3 + UE8M0 块 scale（block 32x32），
        // 默认会被解量化成启动 dtype（float16），权重体积翻倍。
        // FASTLLM_DSV41_ENGRAM_WKV_FP8=1 时按原样保留 FP8，不做任何重量化，
        // 数值上就是 checkpoint 里的那份权重（比解成 float16 少一次舍入）。
        // 只有伴随 .scale 张量存在时才生效，BF16 权重的迷你模型不受影响。
        // 默认开启：保留 checkpoint 里的 FP8 比解量化成 float16 少一次舍入，且省约 300 MB 显存。
        // FASTLLM_DSV41_ENGRAM_WKV_FP8=0 可退回解量化。
        static const bool wkvFp8 = V41EnvFlagOn("FASTLLM_DSV41_ENGRAM_WKV_FP8") && !V41EnvFlag("FASTLLM_DSV41_REFERENCE_MATH");
        std::set<std::string> tensorNameSet(tensorNames.begin(), tensorNames.end());
        auto markQuantizedLinear = [&](const std::string &name) {
            quantizedLinearNames.insert(name);
            if (V41EndsWith(name, ".w1.weight")) {
                quantizedLinearNames.insert(name.substr(0, name.size() - strlen("w1.weight")) + "gateup.weight");
            }
        };
        // FastLLM exports keep scales inside each quantized weight. Preserve
        // the source activation boundaries even when the separate .scale
        // tensors are absent, or weights were requantized to Q2/Q4.
        auto activationNames = this->weight.dicts.find("fastllm_activation_quantized_linears");
        if (activationNames != this->weight.dicts.end()) {
            std::string error;
            auto names = json11::Json::parse(activationNames->second, error);
            AssertInFastLLM(error.empty() && names.is_array(),
                            "DeepSeekV41: invalid exported activation quantization metadata.");
            for (const auto &name : names.array_items()) {
                AssertInFastLLM(name.is_string(),
                                "DeepSeekV41: exported activation quantization names must be strings.");
                if (tensorNameSet.count(name.string_value())) {
                    markQuantizedLinear(name.string_value());
                }
            }
        }
        for (const std::string &name : tensorNames) {
            if (V41EndsWith(name, ".weight") &&
                tensorNameSet.count(name.substr(0, name.size() - strlen("weight")) + "scale")) {
                markQuantizedLinear(name);
            }
        }
        for (const std::string &name : tensorNames) {
            // DSpark 草稿层：只有开启投机解码时才加载（默认跳过，省下约 30 GB 权重）
            if (V41StartsWith(name, "mtp.")) {
                if (!DsparkTensorNeeded(name)) {
                    continue;
                }
                if (V41EndsWith(name, ".confidence_head.proj.weight")) {
                    // 置信度是调度用的概率而不是激活，参考实现用 fp32 计算
                    result[name].push_back({name, DataType::FLOAT32});
                    continue;
                }
                // markov head 的两张表都保持 checkpoint 的 BF16：一来不做多余的重量化，
                // 二来融合的 markov kernel 要求 embed 与 head 的 dtype 一致（各 66 MB）
                if (V41EndsWith(name, ".markov_head.embed.weight") ||
                    V41EndsWith(name, ".markov_head.head.weight")) {
                    result[name].push_back({name, DataType::BFLOAT16});
                    continue;
                }
                // 其余（attn_sink / gate.bias / gate.weight / 线性层）走下面的通用规则
            }
            // 视觉编码器：线性层权重走通用映射（float16），其余（norm / bias / 分隔符嵌入）保持 float32
            if (IsVisionTensor(name)) {
                if (!VisionEnabled()) {
                    continue;
                }
                if (V41EndsWith(name, ".weight") && this->weight.GetWeightType(name) == WeightType::LINEAR) {
                    ordinary.push_back(name);
                } else {
                    result[name].push_back({name, DataType::FLOAT32});
                }
                continue;
            }
            if (name == "head.weight" && !quantizedLinearNames.count(name)) {
                // The original ParallelHead promotes the BF16 checkpoint
                // weight to FP32 and returns FP32 logits without BF16 rounding.
                result[name].push_back({name, DataType::FLOAT32});
                continue;
            }
            if (V41EndsWith(name, ".attn.wkv.weight") && quantizedLinearNames.count(name)) {
                // V4's unquantized-KV exception must not dequantize the V4.1
                // checkpoint to FP16: small weights lose exponent range.
                result[name].push_back({name, DataType::FP8_E4M3});
                continue;
            }
            if (V41EndsWith(name, ".attn.wo_a.weight")) {
                // The original converter dequantizes this grouped projection
                // to BF16, while retaining its explicit activation quantizer.
                result[name].push_back({name, DataType::BFLOAT16});
                continue;
            }
            // 磁盘表复用通用惰性加载；常驻表保留紧凑 UE8M0 scale，避免展开整张 scale 表。
            if (name.find(".engram.embed.") != std::string::npos) {
                if (this->ngramDevice == "disk" && V41EndsWith(name, ".weight")) {
                    result[name].push_back({name, DataType::FP8_E4M3});
                    this->ngramWeights.insert(name);
                }
                continue;
            }
            if (wkvFp8 && V41EndsWith(name, ".engram.wkv.weight") &&
                tensorNameSet.count(name.substr(0, name.size() - strlen("weight")) + "scale") > 0) {
                result[name].push_back({name, DataType::FP8_E4M3});
                continue;
            }
            if (name.find(".engram.q_weight") != std::string::npos ||
                name.find(".engram.k_weight") != std::string::npos ||
                V41EndsWith(name, ".attn_sink") ||
                name.find(".ffn.gate.bias") != std::string::npos) {
                result[name].push_back({name, DataType::FLOAT32});
                continue;
            }
            if (V41EndsWith(name, ".ffn.gate.weight") ||
                name.find(".attn.compressor.wkv.weight") != std::string::npos ||
                name.find(".attn.compressor.wgate.weight") != std::string::npos) {
                result[name].push_back({name, DataType::FLOAT32});
                continue;
            }
            if (name.find(".attn.indexer.wk.weight") != std::string::npos ||
                name.find(".attn.indexer.weights_proj.weight") != std::string::npos) {
                result[name].push_back({name, DataType::BFLOAT16});
                continue;
            }
            ordinary.push_back(name);
        }
        auto mapped = basellm::GetTensorMap(ordinary);
        for (auto &it : mapped) {
            if (V41EnvFlag("FASTLLM_DSV41_REFERENCE_MATH") &&
                it.first.find(".ffn.shared_experts.") != std::string::npos &&
                quantizedLinearNames.count(it.first)) {
                // Preserve checkpoint block scales and block-32 accumulation.
                for (auto &target : it.second) target.second = DataType::FP8_E4M3;
            }
            result[it.first] = it.second;
        }
        return result;
    }

    void DeepSeekV41Model::LinearWithActivationQuant(Data &input, const std::string &weightName,
                                                    Data &output, bool replicated, Data *scratch,
                                                    bool inputQuantized, Data *reductionScratch) {
        Data local;
        Data *source = &input;
        if (quantizedLinearNames.count(weightName)) {
            if (!inputQuantized) {
                source = scratch != nullptr ? scratch : &local;
                V41Executor().Run("DeepSeekV41QuantizeActivation", {{"input", &input}, {"output", source}}, {}, {});
            }
#ifdef USE_CUDA
            // Keep checkpoint block boundaries when checking numerical alignment.
            // A full-K GEMM can cross a BF16 rounding boundary even though both
            // its operands are exactly representable after dequantization.
            static const bool block32 = V41EnvFlag("FASTLLM_DSV41_REFERENCE_MATH");
            if (block32 && !replicated && !source->multiDeviceData &&
                !weight[weightName].multiDeviceData &&
                FastllmCudaDeepSeekV41LinearBlock32(*source, weight[weightName], output)) {
                return;
            }
#endif
        }
#ifdef USE_CUDA
        if (V41EnvFlag("FASTLLM_DSV41_REFERENCE_MATH") && !replicated &&
            FastllmCudaV41ReferenceLinear(*source,weight[weightName],output,1,source->dataType==DataType::BFLOAT16)) return;
        if (replicated && V41EndsWith(weightName, ".engram.wkv.weight") &&
            !V41EnvFlag("FASTLLM_DSV41_REFERENCE_MATH") &&
            V41EngramTpLinear(*source, weight[weightName], output)) return;
#endif
        // Match RowParallelLinear: each local GEMM rounds to the activation
        // dtype, the collective sums in FP32, and only its result rounds back.
#ifdef USE_CUDA
        if (source->IsTensorParallelSharded() &&
            weight[weightName].tpLinearType == TP_LINEAR_COLUMN &&
            (source->dataType == DataType::BFLOAT16 || source->dataType == DataType::FLOAT16)) {
            Data localReduction;
            Data &sum = reductionScratch != nullptr ? *reductionScratch : localReduction;
            if (MultiCudaLinearColumnLocal(*source, weight[weightName], *GetEmptyData(), output)) {
                // The custom collective accumulates half/BF16 operands in
                // FP32 and rounds only once. Avoid the two staging casts and
                // halve communication bytes when that exact path is available.
                if (MultiCudaTryReduceReplicatedFp32Accumulated(output)) return;
                ToDataType(output, sum, DataType::FLOAT32);
                AssertInFastLLM(MultiCudaReduceReplicated(sum), "V4.1 FP32 TP reduction failed.\n");
                ToDataType(sum, output, source->dataType);
                return;
            }
        }
#endif
        Linear(*source, weight[weightName], Data(), output, replicated);
    }

    void DeepSeekV41Model::OnModelWeightsLoaded() {
        if (engram_layer_ids.empty()) {
            return;
        }
        AssertInFastLLM(V41HasKey(this->weight, "model_directory"),
                        "DeepSeekV41: model directory is unknown, can't load engram tables.");
        std::string dir = this->weight.dicts["model_directory"];
        V41EngramMadviseCfg madviseCfg = V41EngramMadvise();
        engramTables.clear();
        for (size_t l = 0; l < engram_layer_ids.size(); l++) {
            int layer = engram_layer_ids[l];
            std::string base = "layers." + std::to_string(layer) + ".engram.embed.";
            auto table = std::make_shared<V41EngramTable>();
            if (this->ngramDevice == "disk") {
                Data &embedding = this->weight[base + "weight"];
                AssertInFastLLM(embedding.isDiskWeight && embedding.cpuData == nullptr &&
                                embedding.dataType == DataType::FP8_E4M3 && embedding.dims.size() == 2 &&
                                embedding.dims[1] == engram_head_dim && embedding.blockK == 1 &&
                                embedding.blockM > 0 && embedding.dims[1] % embedding.blockM == 0 &&
                                embedding.diskWeightParts.size() == 2,
                                "DeepSeekV41: invalid disk engram table " + base + "weight");
                table->rows = embedding.dims[0];
                table->dim = embedding.dims[1];
                table->scaleBlock = embedding.blockM;
                table->diskWeight = &embedding;
                auto scalePart = embedding.diskWeightParts[1];
                AssertInFastLLM(scalePart.isScalePart && scalePart.sourceDataType == DataType::INT8 &&
                                scalePart.bytes == (uint64_t)table->rows * (table->dim / table->scaleBlock),
                                "DeepSeekV41: invalid disk engram scale " + base + "scale");
                scalePart.isScalePart = false;
                table->diskScale = Data(DataType::INT8, scalePart.dims);
                table->diskScale.isDiskWeight = true;
                table->diskScale.weightType = WeightType::EMBEDDING;
                table->diskScale.diskWeightParts.push_back(std::move(scalePart));
                printf("[Fastllm] DeepSeek-V4.1: engram table for layer %d ready (disk).\n", layer);
                engramTables.push_back(std::static_pointer_cast<void>(table));
                continue;
            }
            V41SafeTensorInfo weightInfo, scaleInfo;
            AssertInFastLLM(V41FindSafeTensor(dir, base + "weight", weightInfo) &&
                            V41FindSafeTensor(dir, base + "scale", scaleInfo),
                            "DeepSeekV41: can't locate engram table " + base + "weight in " + dir);
            AssertInFastLLM(weightInfo.dtype == "F8_E4M3" && weightInfo.shape.size() == 2 &&
                            (scaleInfo.dtype == "F8_E8M0" || scaleInfo.dtype == "U8") &&
                            scaleInfo.shape.size() == 2 && scaleInfo.shape[0] == weightInfo.shape[0] &&
                            weightInfo.shape[1] % scaleInfo.shape[1] == 0,
                            "DeepSeekV41: unsupported engram table format for " + base + "weight");
            table->rows = weightInfo.shape[0];
            table->dim = (int)weightInfo.shape[1];
            table->scaleBlock = (int)(weightInfo.shape[1] / scaleInfo.shape[1]);
            AssertInFastLLM(table->dim == engram_head_dim && weightInfo.bytes == (uint64_t)table->rows * table->dim &&
                            scaleInfo.bytes == (uint64_t)table->rows * (table->dim / table->scaleBlock),
                            "DeepSeekV41: engram table byte count mismatch for " + base + "weight");
            printf("[Fastllm] DeepSeek-V4.1: loading engram table for layer %d (%.1f GB, resident%s%s)...\n",
                   layer, (weightInfo.bytes + scaleInfo.bytes) / 1e9,
                   madviseCfg.random ? " +random" : "", madviseCfg.hugePage ? " +hugepage" : "");
            fflush(stdout);
            {
                // 常驻：默认用 std::vector；开了 hugepage 提示时改用匿名 mmap，
                // 这样可以在读入之前 madvise(MADV_HUGEPAGE)，还省掉 vector 的清零。
                uint8_t *dataPtr = nullptr, *scalePtr = nullptr;
                bool anon = madviseCfg.hugePage &&
                            V41AllocAnon(weightInfo.bytes, true, table->mmapData, table->mmapDataLen, dataPtr) &&
                            V41AllocAnon(scaleInfo.bytes, true, table->mmapScale, table->mmapScaleLen, scalePtr);
                if (!anon) {
                    table->dataStorage.resize(weightInfo.bytes);
                    table->scaleStorage.resize(scaleInfo.bytes);
                    dataPtr = table->dataStorage.data();
                    scalePtr = table->scaleStorage.data();
                }
                V41ReadFileRange(weightInfo.fileName, weightInfo.offset, dataPtr, weightInfo.bytes);
                V41ReadFileRange(scaleInfo.fileName, scaleInfo.offset, scalePtr, scaleInfo.bytes);
                table->data = dataPtr;
                table->scale = scalePtr;
            }
            if (madviseCfg.random) {
                // 表是按哈希随机访问的，顺序预读只会白白占用内存带宽 / 页缓存
                V41MadviseRandom(table->data, weightInfo.bytes);
                V41MadviseRandom(table->scale, scaleInfo.bytes);
            }
#ifdef USE_CUDA
            if (this->ngramDevice == "cuda") {
                // Weight loading precedes the first forward, so the executor's
                // TP device list may not have been initialized yet.
                ApplyDeviceMap(this->deviceMap, 1, block_cnt);
                table->gpuDevices = V41TpDevices(this->deviceMap);
                if (!table->gpuDevices.empty()) {
                    table->gpuTable = FastllmCudaV41CreateEngram(table->gpuDevices,
                        table->data, table->scale, table->rows, table->dim, table->scaleBlock);
                }
                printf("[Fastllm] DeepSeek-V4.1: layer %d Engram %s.\n", layer,
                    table->gpuTable ? "sharded GPU table ready" :
                    "using CPU fallback (GPU topology or capacity unavailable)");
                fflush(stdout);
            }
#endif
            engramTables.push_back(std::static_pointer_cast<void>(table));
        }
        printf("[Fastllm] DeepSeek-V4.1: engram tables ready.\n");
        fflush(stdout);
    }

    // ==================== Engram 前向 ====================

    namespace {
        // 自由函数版的哈希计算：预取线程不持有 model，只需要元数据与两个尺寸。
        void V41ComputeEngramHashes(const DeepSeekV41EngramMeta &engramMeta, int engramLayerIndex,
                                    int maxNgram, int heads,
                                    const std::vector<int> &history, int startPos, int seqlen,
                                    std::vector<int64_t> &rows) {
        const int cols = (maxNgram - 1) * heads;
        const auto &multipliers = engramMeta.multipliers[engramLayerIndex];
        const auto &primes = engramMeta.primes[engramLayerIndex];
        const auto &offsets = engramMeta.offsets[engramLayerIndex];
        rows.assign((size_t)seqlen * cols, 0);
        std::vector<int64_t> tokens(maxNgram);
        for (int i = 0; i < seqlen; i++) {
            int pos = startPos + i;
            bool blocked = false;
            for (int shift = 0; shift < maxNgram; shift++) {
                int p = pos - shift;
                int source = p >= 0 ? history[p] : -1;
                blocked = blocked || p < 0 || source < 0;
                tokens[shift] = blocked ? engramMeta.padCompressedId : source;
            }
            // rolling XOR：第 i 步之后的值是 (i+1)-gram 的哈希
            uint64_t rolling = (uint64_t)tokens[0] * (uint64_t)multipliers[0];
            for (int n = 1; n < maxNgram; n++) {
                rolling ^= (uint64_t)tokens[n] * (uint64_t)multipliers[n];
                int64_t signedRolling = (int64_t)rolling;
                for (int h = 0; h < heads; h++) {
                    int col = (n - 1) * heads + h;
                    int64_t prime = primes[col];
                    int64_t bucket = signedRolling % prime;
                    if (bucket < 0) {
                        bucket += prime;   // Python 取模语义
                    }
                    rows[(size_t)i * cols + col] = offsets[col] + bucket;
                }
            }
        }
        }
    }

    void DeepSeekV41Model::ComputeEngramHashes(int engramLayerIndex,
                                               const std::vector<int> &history, int startPos, int seqlen,
                                               std::vector<int64_t> &rows) const {
        V41ComputeEngramHashes(engramMeta, engramLayerIndex, engram_max_ngram_size, engram_n_heads,
                               history, startPos, seqlen, rows);
    }

    namespace {
        // 片段的历史窗口快照：预取线程不引用请求状态，只看这份副本，
        // 这样即使前向提前退出、请求被回收，后台线程也不会读到失效内存。
        struct V41EngramSegSnapshot {
            std::vector<int> history;   // [startPos - maxNgram + 1, startPos + seqlen) 的副本，索引已平移
            int startPos = 0;           // 平移后的起点
            int seqlen = 0;
            int offset = 0;             // 在整批里的 token 偏移
        };

        std::vector<V41EngramSegSnapshot> V41SnapshotSegments(const std::vector<DeepSeekV41Segment> &segments,
                                                              int maxNgram) {
            std::vector<V41EngramSegSnapshot> ret(segments.size());
            for (size_t i = 0; i < segments.size(); i++) {
                const DeepSeekV41Segment &seg = segments[i];
                const std::vector<int> &history = seg.state->engramHistory;
                int lo = std::max(0, seg.startPos - maxNgram + 1);
                int hi = std::min((int)history.size(), seg.startPos + seg.seqlen);
                ret[i].history.assign(history.begin() + lo, history.begin() + std::max(lo, hi));
                ret[i].startPos = seg.startPos - lo;
                ret[i].seqlen = seg.seqlen;
                ret[i].offset = seg.offset;
            }
            return ret;
        }

        // 由快照算出整批的行号与 mask（dead token）。lo > 0 时窗口内不会出现 p < 0，
        // lo == 0 时窗口就是原始历史，两种情况下平移都不改变语义。
        void V41BuildEngramInputs(const DeepSeekV41EngramMeta &meta, int engramLayerIndex,
                                  int maxNgram, int heads,
                                  const std::vector<V41EngramSegSnapshot> &snapshots, int total,
                                  std::vector<int64_t> &rows, std::vector<float> &maskValues, bool &hasDead) {
            const int cols = (maxNgram - 1) * heads;
            rows.clear();
            rows.reserve((size_t)total * cols);
            maskValues.assign(total, 1.0f);
            hasDead = false;
            std::vector<int64_t> segRows;
            for (const V41EngramSegSnapshot &seg : snapshots) {
                V41ComputeEngramHashes(meta, engramLayerIndex, maxNgram, heads,
                                       seg.history, seg.startPos, seg.seqlen, segRows);
                rows.insert(rows.end(), segRows.begin(), segRows.end());
                for (int i = 0; i < seg.seqlen; i++) {
                    if (seg.history[seg.startPos + i] < 0) {
                        maskValues[seg.offset + i] = 0.0f;
                        hasDead = true;
                    }
                }
            }
        }

        // 把要用到的表行摸一遍（每 64 字节一次），把页表项与 cache line 提前拉进来。
        // 常驻表超过缓存预算或使用磁盘 Embedding 时，后台只计算行号。
        void V41TouchEngramRows(const V41EngramTable &table, const std::vector<int64_t> &rows) {
#ifdef USE_CUDA
            if (table.gpuTable) return;
#endif
            const uint64_t budget = 32ULL << 20;
            if (table.diskWeight != nullptr || rows.size() * (uint64_t)table.dim > budget) {
                return;
            }
            const int dim = table.dim;
            const int scaleCols = dim / std::max(1, table.scaleBlock);
            volatile uint64_t sink = 0;
            for (size_t i = 0; i < rows.size(); i++) {
                int64_t row = rows[i];
                if (row < 0 || row >= table.rows) {
                    continue;
                }
                const uint8_t *src = table.data + (uint64_t)row * dim;
                for (int d = 0; d < dim; d += 64) {
                    sink += src[d];
                }
                sink += table.scale[(uint64_t)row * scaleCols];
            }
            (void)sink;
        }

        // 常驻线程池上的一段 token 区间
        struct V41EngramGatherOp : MultiThreadBaseOp {
            const std::function<void(int, int)> *worker;
            int st, end;
            V41EngramGatherOp(const std::function<void(int, int)> *worker, int st, int end)
                : worker(worker), st(st), end(end) {}
            void Run() override {
                (*worker)(st, end);
            }
        };

        // 一次跨层预取任务。生命周期由模型持有，析构时保证 join。
        struct V41EngramPrefetchJob {
            std::thread worker;
            int engramLayerIndex = -1;
            int total = 0;
            std::vector<int64_t> rows;
            std::vector<float> maskValues;
            bool hasDead = false;
            bool ready = false;

            void Join() {
                if (worker.joinable()) {
                    worker.join();
                }
            }

            void Reset() {
                Join();
                engramLayerIndex = -1;
                total = 0;
                ready = false;
                hasDead = false;
                rows.clear();
                maskValues.clear();
            }

            ~V41EngramPrefetchJob() {
                Join();
            }
        };
    }

    void DeepSeekV41Model::GatherEngramRows(int layer, const std::vector<int64_t> &rows, int tokens,
                                           Data &output, double *prepMs) {
        int engramLayerIndex = -1;
        for (size_t l = 0; l < engram_layer_ids.size(); l++) {
            if (engram_layer_ids[l] == layer) {
                engramLayerIndex = (int)l;
            }
        }
        AssertInFastLLM(engramLayerIndex >= 0 && engramLayerIndex < (int)engramTables.size(),
                        "DeepSeekV41: engram table for layer " + std::to_string(layer) + " is not loaded.");
        V41EngramTable &table = *std::static_pointer_cast<V41EngramTable>(engramTables[engramLayerIndex]);
        const int cols = (int)(rows.size() / std::max(1, tokens));
        const int dim = table.dim;
        const int scaleCols = dim / table.scaleBlock;
        double prepStart = V41Profiler().level > 0 ? V41NowMs() : 0.0;
#ifdef USE_CUDA
        if (table.gpuTable && rows.size() <= 1024 * 1024) {
            V41ResetMultiDevice(output);
            output = Data(DataType::BFLOAT16, {1, tokens, cols * dim});
            output.dataDevice = DataDevice::CUDA;
            output.dataDeviceIds = table.gpuDevices;
            PrepareMultiCudaReplicatedData(output, table.gpuDevices, false);
            const bool dispatched = MultiCudaRunDeviceCallbacks(table.gpuDevices, [&](int, int device) {
                Data &local = *output.multiDeviceDatas.at(device);
                local.Allocate(false);
                AssertInFastLLM(FastllmCudaV41GatherEngram(table.gpuTable, rows, local),
                    "DeepSeekV41: GPU Engram gather failed.");
            });
            AssertInFastLLM(dispatched, "DeepSeekV41: GPU Engram dispatch failed.");
            V41SyncRootFromReplica(output, table.gpuDevices);
            if (prepMs) *prepMs = 0;
            return;
        }
#endif
        output.ToDevice(DataDevice::CPU);
        output = Data(DataType::BFLOAT16, {1, tokens, cols * dim});
        output.Allocate(false);
        if (prepMs != nullptr) {
            *prepMs = V41NowMs() - prepStart;
        }
        uint16_t *dst = (uint16_t*)output.cpuData;
        static const FP8E4M3ToFP32Manager fp8;
        Data diskValues, diskScales;
        if (table.diskWeight != nullptr) {
            Data lookupRows(DataType::INT32, {tokens, cols});
            lookupRows.Allocate(false);
            for (size_t i = 0; i < rows.size(); i++) {
                AssertInFastLLM(rows[i] >= 0 && rows[i] < table.rows, "DeepSeekV41: engram hash out of range.");
                ((int32_t*)lookupRows.cpuData)[i] = (int32_t)rows[i];
            }
            V41Executor().RunOnDevice("disk", "EmbeddingDirect",
                {{"input", &lookupRows}, {"weight", table.diskWeight}, {"output", &diskValues}}, {}, {});
            V41Executor().RunOnDevice("disk", "EmbeddingDirect",
                {{"input", &lookupRows}, {"weight", &table.diskScale}, {"output", &diskScales}}, {}, {});
        }

        const std::function<void(int, int)> worker = [&](int st, int end) {
            for (int t = st; t < end; t++) {
                for (int c = 0; c < cols; c++) {
                    int64_t row = rows[(size_t)t * cols + c];
                    AssertInFastLLM(row >= 0 && row < table.rows, "DeepSeekV41: engram hash out of range.");
                    const uint64_t sourceRow = table.diskWeight != nullptr ? (uint64_t)t * cols + c : row;
                    const uint8_t *src = (table.diskWeight != nullptr ? diskValues.cpuData : table.data) + sourceRow * dim;
                    const uint8_t *sc = (table.diskWeight != nullptr ? diskScales.cpuData : table.scale) + sourceRow * scaleCols;
                    uint16_t *out = dst + ((uint64_t)t * cols + c) * dim;
                    for (int d = 0; d < dim; d++) {
                        float v = fp8.dict[src[d]] * V41E8M0ToFloat(sc[d / table.scaleBlock]);
                        out[d] = Float32ToBFloat16RNEBits(v);
                    }
                }
            }
        };
        // FASTLLM_DSV41_ENGRAM_POOL=1：改用 fastllm 常驻线程池。原来的实现每次调用都
        // 现场 create/join 最多 32 个 std::thread，prefill 时这笔固定开销比查表本身还大。
        // 结果逐位相同（只是换了执行 worker 的线程），默认开启；=0 可退回每次现场建线程。
        static const bool usePool = V41EnvFlagOn("FASTLLM_DSV41_ENGRAM_POOL");
        if (usePool) {
            AliveThreadPool *pool = GetAlivePool();
            int threadSt = pool->curActivateThreadInterval.first;
            int threadLen = pool->curActivateThreadInterval.second - threadSt;
            int threads = std::min(tokens, std::max(1, threadLen));
            if (threads <= 1 || tokens < 8) {
                worker(0, tokens);
            } else {
                std::vector<V41EngramGatherOp*> ops;
                int per = (tokens + threads - 1) / threads;
                for (int i = 0; i < threads; i++) {
                    int st = i * per, end = std::min(tokens, st + per);
                    if (st < end) {
                        ops.push_back(new V41EngramGatherOp(&worker, st, end));
                    }
                }
                for (size_t i = 0; i < ops.size(); i++) {
                    pool->PushOp(threadSt + (int)i, ops[i]);
                }
                for (size_t i = 0; i < ops.size(); i++) {
                    pool->Wait(threadSt + (int)i);
                    delete ops[i];
                }
            }
            return;
        }
        // std::thread::hardware_concurrency() 在 glibc 上会去读 /sys/devices/system/cpu/online，
        // 原来每次查表都调一次，单次 decode 就要 0.2 ms——比查表本身贵一个数量级。只算一次。
        static const int hardwareThreads = std::max(1, (int)std::thread::hardware_concurrency() / 2);
        int threads = std::min(tokens, hardwareThreads);
        threads = std::min(threads, 32);
        if (threads <= 1 || tokens < 8) {
            worker(0, tokens);
        } else {
            std::vector<std::thread> pool;
            int per = (tokens + threads - 1) / threads;
            for (int i = 0; i < threads; i++) {
                int st = i * per, end = std::min(tokens, st + per);
                if (st < end) {
                    pool.emplace_back(worker, st, end);
                }
            }
            for (auto &th : pool) {
                th.join();
            }
        }
    }

    void DeepSeekV41Model::RunEngram(int layer, int engramLayerIndex, const std::vector<DeepSeekV41Segment> &segments,
                                     Data &hiddenStates) {
        AssertInFastLLM(engramMeta.loaded,
                        "DeepSeekV41: engram meta is not loaded. Generate engram_meta.json with "
                        "`python -m ftllm.deepseek_v41_engram <model_dir>`.");
        static const bool prefetchEnabled = V41EnvFlag("FASTLLM_DSV41_ENGRAM_PREFETCH");
        V41EngramProfiler &profiler = V41Profiler();
        const bool profiling = profiler.level > 0;
        double tHash = 0.0, tPrep = 0.0, tGather = 0.0, tWkv = 0.0, tApply = 0.0, tWait = 0.0, mark = 0.0;

        std::string pre = "layers." + std::to_string(layer) + ".engram";
        int total = 0;
        for (auto &seg : segments) {
            total += seg.seqlen;
        }

        // ---- 第一段：算行号（查表索引）----
        std::vector<int64_t> rows;
        std::vector<float> maskValues;
        bool hasDead = false;

        V41EngramPrefetchJob *job = nullptr;
        if (prefetchEnabled) {
            if (this->engramPrefetch == nullptr) {
                this->engramPrefetch = std::make_shared<V41EngramPrefetchJob>();
            }
            job = (V41EngramPrefetchJob*)this->engramPrefetch.get();
            if (engramLayerIndex == 0) {
                // 新的一次前向：丢掉上一次可能残留的任务（前向中途异常退出时会留下）
                job->Reset();
            }
        }

        bool tookPrefetched = false;
        if (job != nullptr && job->engramLayerIndex == engramLayerIndex) {
            mark = profiling ? V41NowMs() : 0.0;
            job->Join();
            if (profiling) {
                tWait = V41NowMs() - mark;
            }
            if (job->ready && job->total == total && (int)job->maskValues.size() == total) {
                rows.swap(job->rows);
                maskValues.swap(job->maskValues);
                hasDead = job->hasDead;
                tookPrefetched = true;
            }
            job->Reset();
        }
        std::vector<V41EngramSegSnapshot> snapshots;
        if (!tookPrefetched || job != nullptr) {
            snapshots = V41SnapshotSegments(segments, engram_max_ngram_size);
        }
        if (!tookPrefetched) {
            mark = profiling ? V41NowMs() : 0.0;
            V41BuildEngramInputs(engramMeta, engramLayerIndex, engram_max_ngram_size, engram_n_heads,
                                 snapshots, total, rows, maskValues, hasDead);
            if (profiling) {
                tHash = V41NowMs() - mark;
            }
        }

        // ---- 顺手把下一个 engram 层的行号与表行放到后台算 ----
        // 哈希只依赖 token 历史，进入第 0 层之前就已经全部确定，所以这里不需要任何中间激活。
        if (job != nullptr && engramLayerIndex + 1 < (int)engram_layer_ids.size() &&
            engramLayerIndex + 1 < (int)engramTables.size()) {
            int nextIndex = engramLayerIndex + 1;
            job->Reset();   // 可能还挂着上一次前向留下的任务，先 join 再复用
            job->engramLayerIndex = nextIndex;
            job->total = total;
            job->ready = false;
            const DeepSeekV41EngramMeta &meta = engramMeta;
            int maxNgram = engram_max_ngram_size, heads = engram_n_heads;
            auto tablePtr = std::static_pointer_cast<V41EngramTable>(engramTables[nextIndex]);
            job->worker = std::thread([job, nextIndex, total, snapshots, &meta, maxNgram, heads, tablePtr]() {
                V41BuildEngramInputs(meta, nextIndex, maxNgram, heads, snapshots, total,
                                     job->rows, job->maskValues, job->hasDead);
                V41TouchEngramRows(*tablePtr, job->rows);
                job->ready = true;
            });
        }

        // ---- 第二段：查表 + FP8→BF16 ----
        mark = profiling ? V41NowMs() : 0.0;
        Data gathered;
        GatherEngramRows(layer, rows, total, gathered, profiling ? &tPrep : nullptr);
        if (profiling) {
            tGather = V41NowMs() - mark - tPrep;
        }

        // ---- 第三段：wkv 投影与门控写回（GPU，异步）----
        // 张量并行：gathered / mask 是 CPU 查表产物，而 hiddenStates 是每卡一份的复制布局。
        // 必须先把它们广播成复制布局，wkv 投影才会逐卡各算一份、kv 也才是复制的；
        // 否则第二张卡上的 EngramApply 拿不到有效的 kv（读到未初始化显存 -> NaN，
        // 从 engram 层开始一路传播）。这与 CPU/NUMA MoE 输入、图像 token CPU 路由
        // 是同一类问题（CPU 产出的数据要喂给复制布局张量）。
        mark = profiling ? V41NowMs() : 0.0;
        const std::vector<int> tpDevices = V41TpDevices(this->deviceMap);
        const bool tp = !tpDevices.empty();
        Data kv;
#ifdef USE_CUDA
        if (tp && !gathered.multiDeviceData) {
            // Large CPU Engram rows need one PCIe upload, not one complete
            // host copy/upload per rank. NCCL preserves the original BF16 bits.
            if (gathered.GetBytes() < 1024 * 1024 ||
                !MultiCudaBroadcastCpuActivation(gathered, tpDevices)) {
                PrepareMultiCudaReplicatedData(gathered, tpDevices, true);
            }
        }
#endif
        LinearWithActivationQuant(gathered, pre + ".wkv.weight", kv, tp);
        if (profiling) {
            tWkv = V41NowMs() - mark;
            mark = V41NowMs();
        }
        Data mask;
        if (hasDead) {
            mask.CopyFrom(Data(DataType::FLOAT32, {1, total}, maskValues));
#ifdef USE_CUDA
            if (tp) {
                PrepareMultiCudaReplicatedData(mask, tpDevices, true);
            }
#endif
        }
        V41EngramApply(hiddenStates, kv, weight[pre + ".q_weight"], weight[pre + ".k_weight"],
                       hasDead ? &mask : nullptr, rms_norm_eps);
        if (profiling) {
            tApply = V41NowMs() - mark;
            profiler.Add(layer, total, tHash, tPrep, tGather, tWkv, tApply, tWait);
        }
    }

    // ==================== 请求状态 ====================

    namespace {
        // 原地清空请求状态（layers 里的 Data 没有深拷贝赋值，只能重建）
        bool V41HasImageToken(const std::vector<int> &tokens, int imageTokenId) {
            if (imageTokenId < 0) {
                return false;
            }
            for (int tok : tokens) {
                if (tok == imageTokenId) {
                    return true;
                }
            }
            return false;
        }

        void V41ResetState(DeepSeekV41RequestState &state, int blockCnt) {
            state.layers.clear();
            state.layers.resize(blockCnt);
            state.engramHistory.clear();
            state.totalLen = 0;
            state.restoredLen = 0;
            // pendingMultimodal 由 ResponseContext 持有，重建时保留；图像嵌入需要重新编码
            state.imagesEncoded = false;
            state.imageSpans.clear();
            // DSpark：滑窗与待发队列都是相对旧缓存的，一并丢弃
            state.dspark.reset();
        }
    }

    void DeepSeekV41Model::RegisterState(const void *vectorKey, const void *firstKey,
                                         const std::shared_ptr<DeepSeekV41RequestState> &state) {
        if (vectorKey != nullptr) {
            v41States[vectorKey] = state;
        }
        if (firstKey != nullptr) {
            v41StatesByFirstKey[firstKey] = state;
        }
    }

    std::shared_ptr<DeepSeekV41RequestState> DeepSeekV41Model::GetOrCreateState(
            std::vector<std::pair<Data, Data> > &pastKeyValues, bool reset) {
        const void *key = (const void*)&pastKeyValues;
        const void *firstKey = pastKeyValues.empty() ? nullptr : (const void*)&pastKeyValues[0].first;
        std::lock_guard<std::mutex> guard(v41StateMutex);
        auto it = v41States.find(key);
        if (it != v41States.end()) {
            if (reset) {
                V41ResetState(*it->second, block_cnt);
            }
            if (firstKey != nullptr) {
                v41StatesByFirstKey[firstKey] = it->second;
            }
            return it->second;
        }
        auto state = std::make_shared<DeepSeekV41RequestState>();
        state->layers.resize(block_cnt);
        RegisterState(key, firstKey, state);
        return state;
    }

    std::shared_ptr<DeepSeekV41RequestState> DeepSeekV41Model::GetStateByFirstKey(const Data *firstKey) {
        std::lock_guard<std::mutex> guard(v41StateMutex);
        auto it = v41StatesByFirstKey.find((const void*)firstKey);
        return it == v41StatesByFirstKey.end() ? nullptr : it->second;
    }

    void DeepSeekV41Model::OnResponseContextCreated(ResponseContext *context) {
        if (context == nullptr) {
            return;
        }
        const void *key = (const void*)&context->pastKeyValues;
        const void *firstKey = context->pastKeyValues.empty() ? nullptr : (const void*)&context->pastKeyValues[0].first;
        std::lock_guard<std::mutex> guard(v41StateMutex);
        std::shared_ptr<DeepSeekV41RequestState> state;
        if (v41PendingRestoredState) {
            state = v41PendingRestoredState;
            v41PendingRestoredState.reset();
        } else {
            auto existing = v41States.find(key);
            if (existing != v41States.end()) {
                state = existing->second;
            } else {
                state = std::make_shared<DeepSeekV41RequestState>();
                state->layers.resize(block_cnt);
            }
        }
        RegisterState(key, firstKey, state);
        // 图文请求：调度器可能只用普通 Forward 逐块 prefill，因此在这里就把多模态输入记到请求状态里，
        // 由第一个 prefill 块编码图像（见 ForwardSegments 的调用方）
        if (!context->multimodalInput.empty()) {
            state->pendingMultimodal = &context->multimodalInput;
        }
    }

    void DeepSeekV41Model::OnResponseContextRemoved(ResponseContext *context) {
        if (context == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> guard(v41StateMutex);
        v41States.erase((const void*)&context->pastKeyValues);
        if (!context->pastKeyValues.empty()) {
            v41StatesByFirstKey.erase((const void*)&context->pastKeyValues[0].first);
        }
    }

    // ==================== 前缀缓存 ====================

    namespace {
        bool V41PrefixCacheDisabled() {
            static const bool disabled = V41EnvFlag("FASTLLM_DSV41_DISABLE_PREFIX_CACHE");
            return disabled;
        }

        bool V41PrefixCacheDebug() {
            static const bool debug = V41EnvFlag("FASTLLM_DSV41_PREFIX_CACHE_DEBUG");
            return debug;
        }

        int V41EnvInt(const char *name, int fallback) {
            const char *v = std::getenv(name);
            if (v == nullptr || v[0] == '\0') {
                return fallback;
            }
            return atoi(v);
        }

        // 快照：深拷贝到 CPU（保留 expansion 容量，restore 后可继续追加）
        void V41SnapshotTensor(Data &dst, const Data &src) {
            if (src.dims.size() == 0 || src.Count(0) == 0) {
                return;
            }
#ifdef USE_CUDA
            // 张量并行下缓存是每卡一份，root 只保留形状；快照必须从某张卡的副本上取。
            if (src.multiDeviceData && src.IsTensorParallelReplicated()) {
                for (const auto &it : src.multiDeviceDatas) {
                    if (it.second == nullptr || it.second->cudaData == nullptr) {
                        continue;
                    }
                    const int oriDevice = FastllmCudaGetDevice();
                    FastllmCudaSetDevice(it.first);
                    Data local;
                    local.CopyFrom(*it.second);
                    local.ToDevice(DataDevice::CPU);
                    FastllmCudaSetDevice(oriDevice);
                    dst.CopyFrom(local);
                    return;
                }
            }
#endif
            dst.CopyFrom(src);
            dst.ToDevice(DataDevice::CPU);
        }

        // 恢复：CPU -> CPU 深拷贝，随后由执行器在首次使用时搬到计算设备
        void V41RestoreTensor(Data &dst, const Data &src) {
            if (src.dims.size() == 0 || src.Count(0) == 0) {
                return;
            }
            dst.CopyFrom(src);
            dst.SetKVCache();
        }
    }

    void DeepSeekV41HistoryCacheManager::Record(const std::shared_ptr<DeepSeekV41HistoryMemory> &memory) {
        if (!memory || memory->totalLen <= 0 || (int)memory->tokens.size() != memory->totalLen) {
            return;
        }
        std::lock_guard<std::mutex> guard(this->locker);
        int commonMax = V41EnvInt("FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_RECORDS", this->maxRecordNum);
        this->maxRecordNum = std::max(1, V41EnvInt("FASTLLM_DSV41_PREFIX_CACHE_MAX_RECORDS", commonMax));
        auto old = this->memorys.find(memory->tokens);
        if (old != this->memorys.end()) {
            memory->recordTimes = old->second->recordTimes + 1;
            memory->flushTime = ++this->flushTime;
            old->second = memory;
            return;
        }
        while ((int)this->memorys.size() >= this->maxRecordNum) {
            auto eraseIt = this->memorys.end();
            long long minFlushTime = (1LL << 60);
            for (auto it = this->memorys.begin(); it != this->memorys.end(); ++it) {
                if (it->second->flushTime < minFlushTime) {
                    minFlushTime = it->second->flushTime;
                    eraseIt = it;
                }
            }
            if (eraseIt == this->memorys.end()) {
                break;
            }
            this->memorys.erase(eraseIt);
        }
        memory->recordTimes = 1;
        memory->flushTime = ++this->flushTime;
        this->memorys[memory->tokens] = memory;
    }

    std::vector<std::pair<std::shared_ptr<DeepSeekV41HistoryMemory>, int> >
    DeepSeekV41HistoryCacheManager::GetCandidates(const std::vector<int> &inputTokens) {
        std::vector<std::pair<std::shared_ptr<DeepSeekV41HistoryMemory>, int> > candidates;
        std::lock_guard<std::mutex> guard(this->locker);
        // 至少留一个 token 给本次前向
        const int maxLen = (int)inputTokens.size() - 1;
        for (auto &it : this->memorys) {
            const std::vector<int> &tokens = it.first;
            int limit = std::min(maxLen, (int)tokens.size());
            int len = 0;
            while (len < limit && tokens[len] == inputTokens[len]) {
                len++;
            }
            if (len > 0) {
                candidates.push_back({it.second, len});
            }
        }
        std::stable_sort(candidates.begin(), candidates.end(),
                         [](const std::pair<std::shared_ptr<DeepSeekV41HistoryMemory>, int> &a,
                            const std::pair<std::shared_ptr<DeepSeekV41HistoryMemory>, int> &b) {
                             if (a.second != b.second) {
                                 return a.second > b.second;
                             }
                             return a.first->totalLen < b.first->totalLen;
                         });
        return candidates;
    }

    bool DeepSeekV41Model::CanTruncateHistory(const DeepSeekV41HistoryMemory &memory, int len) const {
        const int total = memory.totalLen;
        if (len <= 0 || len > total) {
            return false;
        }
        // 滑窗环形缓存只保留最后 window_size 个位置；截断到 len 后需要 [len - W + 1, len) 仍然完整
        if (std::max(0, len - window_size + 1) < std::max(0, total - window_size)) {
            return false;
        }
        // 压缩 KV：凑不满一组的原始尾块只在 len == total 时可用
        for (int layer : kv_source_layer_ids) {
            int ratio = compress_ratios[layer];
            if (ratio > 1 && len % ratio != 0 && len != total) {
                return false;
            }
        }
        return true;
    }

    std::shared_ptr<DeepSeekV41HistoryMemory> DeepSeekV41Model::SnapshotState(
            const DeepSeekV41RequestState &state, const std::vector<int> &allTokens) {
        const int totalLen = state.totalLen;
        if (totalLen <= 0 || (int)allTokens.size() < totalLen || (int)state.layers.size() != block_cnt) {
            return nullptr;
        }
        if (!engram_layer_ids.empty() && (int)state.engramHistory.size() < totalLen) {
            return nullptr;
        }
        auto memory = std::make_shared<DeepSeekV41HistoryMemory>();
        memory->totalLen = totalLen;
        memory->tokens.assign(allTokens.begin(), allTokens.begin() + totalLen);
        memory->engramHistory.assign(state.engramHistory.begin(),
                                     state.engramHistory.begin() + std::min((int)state.engramHistory.size(), totalLen));
        memory->layers.resize(block_cnt);
        for (int layer = 0; layer < block_cnt; layer++) {
            const DeepSeekV41LayerCache &src = state.layers[layer];
            DeepSeekV41LayerCache &dst = memory->layers[layer];
            if (src.totalLen != totalLen) {
                return nullptr;
            }
            dst.totalLen = src.totalLen;
            dst.compressedBlocks = src.compressedBlocks;
            dst.rawTail = src.rawTail;
            V41SnapshotTensor(dst.windowKV, src.windowKV);
            if (isKvSource[layer]) {
                V41SnapshotTensor(dst.compressedKV, src.compressedKV);
                V41SnapshotTensor(dst.indexK, src.indexK);
                if (src.rawTail > 0) {
                    V41SnapshotTensor(dst.rawTailKV, src.rawTailKV);
                    V41SnapshotTensor(dst.rawTailScore, src.rawTailScore);
                }
            }
        }
        return memory;
    }

    std::shared_ptr<DeepSeekV41RequestState> DeepSeekV41Model::RestoreState(
            const DeepSeekV41HistoryMemory &memory, int len) {
        auto state = std::make_shared<DeepSeekV41RequestState>();
        state->layers.resize(block_cnt);
        state->totalLen = len;
        state->restoredLen = len;
        if (!engram_layer_ids.empty()) {
            state->engramHistory.assign(memory.engramHistory.begin(), memory.engramHistory.begin() + len);
        }
        const bool exact = len == memory.totalLen;
        for (int layer = 0; layer < block_cnt; layer++) {
            const DeepSeekV41LayerCache &src = memory.layers[layer];
            DeepSeekV41LayerCache &dst = state->layers[layer];
            dst.totalLen = len;
            // 环形缓存整体恢复；位置 >= len 的行不会被读取，之后会被新 token 覆盖
            V41RestoreTensor(dst.windowKV, src.windowKV);
            if (!isKvSource[layer]) {
                continue;
            }
            const int ratio = compress_ratios[layer];
            const int blocks = len / ratio;
            dst.compressedBlocks = blocks;
            if (blocks > 0) {
                V41RestoreTensor(dst.compressedKV, src.compressedKV);
                if (dst.compressedKV.dims.size() == 3 && dst.compressedKV.dims[1] > blocks) {
                    dst.compressedKV.Resize({dst.compressedKV.dims[0], blocks, dst.compressedKV.dims[2]});
                }
                if (isIndexSource[layer]) {
                    V41RestoreTensor(dst.indexK, src.indexK);
                    if (dst.indexK.dims.size() == 3 && dst.indexK.dims[1] > blocks) {
                        dst.indexK.Resize({dst.indexK.dims[0], blocks, dst.indexK.dims[2]});
                    }
                }
            }
            if (exact && src.rawTail > 0) {
                dst.rawTail = src.rawTail;
                V41RestoreTensor(dst.rawTailKV, src.rawTailKV);
                V41RestoreTensor(dst.rawTailScore, src.rawTailScore);
            } else {
                dst.rawTail = 0;
            }
        }
        return state;
    }

    void DeepSeekV41Model::TryRecordResponseContext(ResponseContext *context) {
        if (context == nullptr || !this->saveHistoryChat || V41PrefixCacheDisabled()) {
            return;
        }
        std::shared_ptr<DeepSeekV41RequestState> state;
        {
            std::lock_guard<std::mutex> guard(v41StateMutex);
            auto it = v41States.find((const void*)&context->pastKeyValues);
            if (it != v41States.end()) {
                state = it->second;
            }
        }
        if (!state || state->totalLen <= 0 || context->allTokens.empty()) {
            return;
        }
        // 图文请求的状态不进前缀缓存（原因同 TryRestoreHistoryCache）
        if (!context->multimodalInput.empty() || V41HasImageToken(context->allTokens, image_token_id)) {
            return;
        }
        auto memory = SnapshotState(*state, context->allTokens);
        if (!memory) {
            if (V41PrefixCacheDebug()) {
                printf("[fastllm-dsv41-prefix-cache] skip record: total_len=%d all_tokens=%d\n",
                       state->totalLen, (int)context->allTokens.size());
                fflush(stdout);
            }
            return;
        }
        v41HistoryCache.Record(memory);
        if (V41PrefixCacheDebug()) {
            printf("[fastllm-dsv41-prefix-cache] record tokens=%d records=%d\n",
                   memory->totalLen, (int)v41HistoryCache.memorys.size());
            fflush(stdout);
        }
    }

    bool DeepSeekV41Model::TryRestoreHistoryCache(std::vector<int> &inputTokens, int &cacheLen) {
        cacheLen = 0;
        if (!this->saveHistoryChat || V41PrefixCacheDisabled()) {
            return false;
        }
        const int minTokens = std::max(1, V41EnvInt("FASTLLM_DSV41_PREFIX_CACHE_MIN_TOKENS", 16));
        if ((int)inputTokens.size() <= minTokens) {
            return false;
        }
        // 图文请求不复用前缀：图像占位 token 的 id 与图像内容无关，同样的文字配不同的图会误命中
        if (V41HasImageToken(inputTokens, image_token_id)) {
            return false;
        }
        auto candidates = v41HistoryCache.GetCandidates(inputTokens);
        std::shared_ptr<DeepSeekV41HistoryMemory> memory;
        int hitLen = 0, len = 0;
        for (auto &candidate : candidates) {
            if (candidate.second < minTokens) {
                break;
            }
            // 截断约束（滑窗 / 压缩尾块）最多需要回退几个 token
            int cur = candidate.second;
            for (int step = 0; step < 4 && cur >= minTokens && !CanTruncateHistory(*candidate.first, cur); step++) {
                cur--;
            }
            if (cur >= minTokens && CanTruncateHistory(*candidate.first, cur)) {
                memory = candidate.first;
                hitLen = candidate.second;
                len = cur;
                break;
            }
        }
        if (!memory) {
            if (V41PrefixCacheDebug()) {
                printf("[fastllm-dsv41-prefix-cache] miss input_tokens=%d candidates=%d best_lcp=%d\n",
                       (int)inputTokens.size(), (int)candidates.size(),
                       candidates.empty() ? 0 : candidates[0].second);
                fflush(stdout);
            }
            return false;
        }
        {
            std::lock_guard<std::mutex> guard(v41HistoryCache.locker);
            memory->flushTime = ++v41HistoryCache.flushTime;
        }
        auto state = RestoreState(*memory, len);
        {
            std::lock_guard<std::mutex> guard(v41StateMutex);
            v41PendingRestoredState = state;
        }
        inputTokens.erase(inputTokens.begin(), inputTokens.begin() + len);
        cacheLen = len;
        if (V41PrefixCacheDebug()) {
            printf("[fastllm-dsv41-prefix-cache] hit len=%d (lcp=%d record=%d) remaining=%d\n",
                   len, hitLen, memory->totalLen, (int)inputTokens.size());
            fflush(stdout);
        }
        return true;
    }

    void DeepSeekV41Model::TryRecordHistoryCache(const std::vector<int> &allTokens) {
        // 状态与 ResponseContext 绑定，记录在 TryRecordResponseContext 中完成
        (void)allTokens;
    }

    // ==================== 前向 ====================

    namespace {
        // 只更新调度器读取的第 0 层占位 KV（kvCacheId == 0）
        void V41UpdateStubKV(Data &key, Data &value, int totalLen) {
            int paddedLen = (std::max(totalLen, 1) / 128 + 1) * 128;
            std::vector<float> zeros((uint64_t)totalLen, 0.0f);
            Data stubKey(DataType::FLOAT32, {1, totalLen, 1}, zeros);
            Data stubValue(DataType::FLOAT32, {1, totalLen, 1}, zeros);
            stubKey.SetKVCache();
            stubValue.SetKVCache();
            stubKey.Expansion({1, paddedLen, 1});
            stubValue.Expansion({1, paddedLen, 1});
            key.FreeSpace();
            value.FreeSpace();
            key = Data();
            value = Data();
            key.CopyFrom(stubKey);
            value.CopyFrom(stubValue);
            key.SetKVCache();
            value.SetKVCache();
        }

        int V41FirstPosition(const Data *positionIds) {
            if (positionIds == nullptr || positionIds->dims.size() == 0 || positionIds->Count(0) == 0) {
                return 0;
            }
            auto pids = V41ReadTokenIds(*positionIds);
            return pids.empty() ? 0 : pids[0];
        }
    }

    bool DeepSeekV41Model::TryForwardChunkedPrefill(
            const Data &inputIds, const Data &attentionMask, const Data &positionIds,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
            std::vector<float> *logits, int &outputToken) {
        if (executionMode == ExecutionMode::Hybrid &&
            TryForwardHybridChunkedPrefill(inputIds, attentionMask, positionIds,
                pastKeyValues, generationConfig, lastTokens, logits, outputToken)) {
            auto state = GetStateByFirstKey(&pastKeyValues[0].first);
            V41UpdateStubKV(pastKeyValues[0].first, pastKeyValues[0].second, state->totalLen);
            return true;
        }
        const int chunk = GetChunkedPrefillSize();
        if (decoderSwaTailLayer <= 0 || decoderSwaTailLayer >= block_cnt ||
            v41DsparkEnabled || chunk <= window_size || inputIds.dims.size() != 2 ||
            inputIds.dims[0] != 1 || inputIds.dims[1] <= chunk ||
            positionIds.dims.size() != 2 || positionIds.dims != inputIds.dims ||
            V41FirstPosition(&positionIds) != 0) return false;
        const int length = inputIds.dims[1];
        const int lastChunk = (length - 1) % chunk + 1;
        // A short final chunk would consume the preceding decoder window.
        // Only bounded replay reconstructs that window without old tail KV.
        if (lastChunk <= window_size ||
            V41HasImageToken(V41ReadTokenIds(inputIds), image_token_id)) return false;
        for (int layer : engram_layer_ids) if (layer >= decoderSwaTailLayer) return false;
        std::vector<std::vector<float>*> finalLogits{logits};
        for (int start = 0; start < length; start += chunk) {
            const auto begin = std::chrono::steady_clock::now();
            const int count = std::min(chunk, length - start);
            const bool cacheOnly = start + count < length;
            Data ids, positions;
            Split(inputIds, 1, start, start + count, ids);
            Split(positionIds, 1, start, start + count, positions);
            auto result = ForwardSingle(ids, positions, pastKeyValues, generationConfig,
                lastTokens, cacheOnly ? nullptr : &finalLogits, nullptr, nullptr, cacheOnly);
            if (!cacheOnly) outputToken = result[0];
            if (verbose) {
                const double seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - begin).count();
                printf("[Prompt] Long Prefill ... (%d/%d, %d%%). Speed: %.2f tokens / s.\n",
                    start + count, length, (start + count) * 100 / length,
                    seconds > 0 ? count / seconds : 0);
                fflush(stdout);
            }
        }
        return true;
    }

    int DeepSeekV41Model::Forward(const Data &inputIds, const Data &attentionMask, const Data &positionIds,
                                  std::vector<std::pair<Data, Data> > &pastKeyValues,
                                  const GenerationConfig &generationConfig,
                                  const LastTokensManager &lastTokens,
                                  std::vector<float> *retLogits) {
        std::vector<std::vector<float>*> batchLogits;
        batchLogits.push_back(retLogits);
        return ForwardBatch(1, inputIds, attentionMask, positionIds, pastKeyValues,
                            generationConfig, lastTokens, &batchLogits)[0];
    }

    std::vector<int> DeepSeekV41Model::ForwardBatch(int batch, const Data &inputIds, const Data &attentionMask,
                                                    const Data &positionIds,
                                                    std::vector<std::pair<Data, Data> > &pastKeyValues,
                                                    const GenerationConfig &generationConfig,
                                                    const LastTokensManager &lastTokens,
                                                    std::vector<std::vector<float>*> *retLogits) {
        (void)attentionMask;
        AssertInFastLLM(batch == 1 && inputIds.dims.size() == 2 && inputIds.dims[0] == 1,
                        "DeepSeekV41Model::ForwardBatch only supports one sequence per call.");
        return ForwardSingle(inputIds, positionIds, pastKeyValues, generationConfig, lastTokens, retLogits,
                             nullptr, nullptr);
    }

    std::vector<int> DeepSeekV41Model::ForwardSingle(const Data &inputIds, const Data &positionIds,
                                                     std::vector<std::pair<Data, Data> > &pastKeyValues,
                                                     const GenerationConfig &generationConfig,
                                                     const LastTokensManager &lastTokens,
                                                     std::vector<std::vector<float>*> *retLogits,
                                                     const Data *inputEmbeds,
                                                     const std::vector<int> *imageMask, bool prefillCacheOnly) {
        const int seqlen = inputIds.dims[1];
        const int startPos = V41FirstPosition(&positionIds);
        AssertInFastLLM(inputEmbeds == nullptr ||
                        (inputEmbeds->dims.size() == 3 && inputEmbeds->dims[1] == seqlen &&
                         inputEmbeds->dims[2] == embed_dim),
                        "DeepSeekV41Model: inputEmbeds must be [1, seqlen, dim].");
        AssertInFastLLM(imageMask == nullptr || (int)imageMask->size() == seqlen,
                        "DeepSeekV41Model: imageMask length mismatch.");
        // 重建缓存时 V41ResetState 保留 pendingMultimodal（图像会重新编码）
        auto state = GetOrCreateState(pastKeyValues, startPos == 0);

        // ---- DSpark：已经校验通过的 token 直接出队，不需要前向 ----
        if (v41DsparkEnabled && state->dspark && !state->dspark->pending.empty()) {
            int queued = DsparkTakePending(*state, inputIds, seqlen);
            if (queued >= 0) {
                if (!pastKeyValues.empty()) {
                    V41UpdateStubKV(pastKeyValues[0].first, pastKeyValues[0].second, state->totalLen);
                }
                return std::vector<int>{queued};
            }
        }
        AssertInFastLLM(state->totalLen == startPos,
                        "DeepSeekV41Model: position mismatch (cache has " + std::to_string(state->totalLen) +
                        " tokens, request starts at " + std::to_string(startPos) + ").");

        // 图文请求：第一个块编码全部图像，之后每个块把与图像 span 重叠的位置换成图像嵌入
        Data imageEmbeds;
        std::vector<int> imageMaskStorage;
        if (inputEmbeds == nullptr && state->pendingMultimodal != nullptr) {
            if (!state->imagesEncoded) {
                EncodeImageSpans(*state->pendingMultimodal, *state);
            }
            if (PrepareImageEmbeds(inputIds, startPos, *state, imageEmbeds, imageMaskStorage)) {
                inputEmbeds = &imageEmbeds;
                imageMask = &imageMaskStorage;
            }
        }

        std::vector<DeepSeekV41Segment> segments(1);
        segments[0].state = state;
        segments[0].startPos = startPos;
        segments[0].seqlen = seqlen;
        segments[0].offset = 0;
        segments[0].prefillCacheOnly = prefillCacheOnly;
        std::vector<GenerationConfig> generationConfigs(1, generationConfig);
        std::vector<std::pair<Data*, Data*> > samplingPastKeyValues;
        for (auto &kvPair : pastKeyValues) {
            samplingPastKeyValues.push_back(std::make_pair(&kvPair.first, &kvPair.second));
        }
        std::vector<int> ret = ForwardSegmentsWithDspark(segments, inputIds, inputEmbeds, imageMask,
                                                         generationConfigs, lastTokens, retLogits,
                                                         samplingPastKeyValues, true);
        if (!pastKeyValues.empty()) {
            V41UpdateStubKV(pastKeyValues[0].first, pastKeyValues[0].second, state->totalLen);
        }
        return ret;
    }

    std::vector<int> DeepSeekV41Model::ForwardBatch(int batch, const Data &inputIds,
                                                    const std::vector<Data*> &attentionMask,
                                                    const std::vector<Data*> &positionIds,
                                                    const std::vector<int> &seqLens,
                                                    std::vector<std::pair<Data*, Data*> > &pastKeyValues,
                                                    const std::vector<GenerationConfig> &generationConfigs,
                                                    const LastTokensManager &lastTokens,
                                                    std::vector<std::vector<float>*> *retLogits) {
        (void)attentionMask;
        AssertInFastLLM(batch >= 1 && (int)seqLens.size() == batch && (int)positionIds.size() == batch &&
                        (int)pastKeyValues.size() == batch * block_cnt &&
                        (int)generationConfigs.size() == batch,
                        "DeepSeekV41Model::ForwardBatch: inconsistent batch arguments.");
        int total = 0;
        for (int len : seqLens) {
            total += len;
        }
        AssertInFastLLM(inputIds.Count(0) == (uint64_t)total,
                        "DeepSeekV41Model::ForwardBatch: inputIds length does not match seqLens.");

        // ---- DSpark：已经校验通过的 token 直接出队，这些请求不参与本次前向 ----
        // （批量前向里不做投机，但仍然要采集 main hidden，让草稿侧的滑窗跟上目标缓存）
        std::vector<int> pendingRet(batch, -1);
        std::vector<char> isPending(batch, 0);
        int pendingCount = 0;
        if (v41DsparkEnabled) {
            Data cpuIds;
            cpuIds.CopyFrom(inputIds);
            cpuIds.ToDevice(DataDevice::CPU);
            int scan = 0;
            for (int i = 0; i < batch; i++) {
                auto state = GetStateByFirstKey(pastKeyValues[(size_t)i * block_cnt].first);
                if (state && state->dspark && !state->dspark->pending.empty()) {
                    Data one;
                    Split(cpuIds, cpuIds.dims.size() - 1, scan, scan + seqLens[i], one);
                    int queued = DsparkTakePending(*state, one, seqLens[i]);
                    if (queued >= 0) {
                        pendingRet[i] = queued;
                        isPending[i] = 1;
                        pendingCount++;
                    }
                }
                scan += seqLens[i];
            }
        }
        if (pendingCount == batch) {
            for (int i = 0; i < batch; i++) {
                auto state = GetStateByFirstKey(pastKeyValues[(size_t)i * block_cnt].first);
                if (state) {
                    V41UpdateStubKV(*pastKeyValues[(size_t)i * block_cnt].first,
                                    *pastKeyValues[(size_t)i * block_cnt].second, state->totalLen);
                }
            }
            return pendingRet;
        }
        // 有请求出队时，把剩下的请求重新拼成一次前向
        if (pendingCount > 0) {
            Data cpuIds;
            cpuIds.CopyFrom(inputIds);
            cpuIds.ToDevice(DataDevice::CPU);
            std::vector<Data*> activeMask, activePosition;
            std::vector<int> activeSeqLens, activeIndex;
            std::vector<std::pair<Data*, Data*> > activePast;
            std::vector<GenerationConfig> activeConfigs;
            std::vector<std::vector<float>*> activeLogits;
            LastTokensManager activeTokens;
            Data activeIds, part, catTmp;
            int scan = 0;
            bool first = true;
            for (int i = 0; i < batch; i++) {
                if (!isPending[i]) {
                    Split(cpuIds, cpuIds.dims.size() - 1, scan, scan + seqLens[i], part);
                    if (first) {
                        activeIds.CopyFrom(part);
                        first = false;
                    } else {
                        Cat(activeIds, part, activeIds.dims.size() - 1, catTmp);
                        activeIds.CopyFrom(catTmp);
                    }
                    activePosition.push_back(positionIds[i]);
                    activeSeqLens.push_back(seqLens[i]);
                    activeConfigs.push_back(generationConfigs[i]);
                    activeIndex.push_back(i);
                    for (int l = 0; l < block_cnt; l++) {
                        activePast.push_back(pastKeyValues[(size_t)i * block_cnt + l]);
                    }
                    if ((int)lastTokens.units.size() > i) {
                        activeTokens.units.push_back(lastTokens.units[i]);
                    }
                    if (retLogits != nullptr && (int)retLogits->size() > i) {
                        activeLogits.push_back((*retLogits)[i]);
                    }
                }
                scan += seqLens[i];
            }
            std::vector<Data*> emptyMask;
            auto sub = ForwardBatch((int)activeSeqLens.size(), activeIds, emptyMask, activePosition,
                                    activeSeqLens, activePast, activeConfigs, activeTokens,
                                    retLogits != nullptr ? &activeLogits : nullptr);
            for (size_t k = 0; k < activeIndex.size(); k++) {
                pendingRet[activeIndex[k]] = sub[k];
            }
            for (int i = 0; i < batch; i++) {
                if (isPending[i]) {
                    auto state = GetStateByFirstKey(pastKeyValues[(size_t)i * block_cnt].first);
                    if (state) {
                        V41UpdateStubKV(*pastKeyValues[(size_t)i * block_cnt].first,
                                        *pastKeyValues[(size_t)i * block_cnt].second, state->totalLen);
                    }
                }
            }
            return pendingRet;
        }

        std::vector<DeepSeekV41Segment> segments(batch);
        int offset = 0;
        for (int i = 0; i < batch; i++) {
            const int startPos = V41FirstPosition(positionIds[i]);
            auto state = GetStateByFirstKey(pastKeyValues[(size_t)i * block_cnt].first);
            if (!state) {
                AssertInFastLLM(startPos == 0,
                                "DeepSeekV41Model::ForwardBatch: request state is missing for a continuing request.");
                state = std::make_shared<DeepSeekV41RequestState>();
                state->layers.resize(block_cnt);
                std::lock_guard<std::mutex> guard(v41StateMutex);
                RegisterState(nullptr, (const void*)pastKeyValues[(size_t)i * block_cnt].first, state);
            } else if (startPos == 0 && state->totalLen != 0) {
                V41ResetState(*state, block_cnt);
            }
            AssertInFastLLM(state->totalLen == startPos,
                            "DeepSeekV41Model: position mismatch in batch (cache has " +
                            std::to_string(state->totalLen) + " tokens, request starts at " +
                            std::to_string(startPos) + ").");
            segments[i].state = state;
            segments[i].startPos = startPos;
            segments[i].seqlen = seqLens[i];
            segments[i].offset = offset;
            offset += seqLens[i];
        }

        std::vector<int> ret;
        if (batch > 1) {
            static bool announced = false;
            if (!announced) {
                announced = true;
                printf("[Fastllm] DeepSeek-V4.1: batched forward active (batch = %d).\n", batch);
                fflush(stdout);
            }
        }
        // 含长 prefill 片段的混合批次退回为逐个前向，避免一次前向的激活内存过大
        bool splitBatch = false;
        if (batch > 1) {
            const int chunkLimit = std::max(1, GetChunkedPrefillSize());
            for (auto &seg : segments) {
                if (seg.seqlen > chunkLimit) {
                    splitBatch = true;
                }
            }
        }
        if (splitBatch) {
            Data cpuIds;
            cpuIds.CopyFrom(inputIds);
            cpuIds.ToDevice(DataDevice::CPU);
            for (int i = 0; i < batch; i++) {
                std::vector<DeepSeekV41Segment> one(1);
                one[0] = segments[i];
                one[0].offset = 0;
                Data curIds;
                Split(cpuIds, 1, segments[i].offset, segments[i].offset + segments[i].seqlen, curIds);
                std::vector<GenerationConfig> curConfigs(1, generationConfigs[i]);
                LastTokensManager curTokens;
                if ((int)lastTokens.units.size() > i) {
                    curTokens.units.push_back(lastTokens.units[i]);
                }
                std::vector<std::vector<float>*> curLogits;
                if (retLogits != nullptr && (int)retLogits->size() > i) {
                    curLogits.push_back((*retLogits)[i]);
                }
                std::vector<std::pair<Data*, Data*> > curPastKeyValues(
                        pastKeyValues.begin() + (size_t)i * block_cnt,
                        pastKeyValues.begin() + (size_t)(i + 1) * block_cnt);
                auto cur = ForwardSegmentsWithDspark(one, curIds, nullptr, nullptr, curConfigs, curTokens,
                                                     retLogits != nullptr ? &curLogits : nullptr,
                                                     curPastKeyValues, true);
                ret.push_back(cur[0]);
            }
        } else {
            ret = ForwardSegmentsWithDspark(segments, inputIds, nullptr, nullptr, generationConfigs, lastTokens,
                                            retLogits, pastKeyValues, batch == 1);
        }
        for (int i = 0; i < batch; i++) {
            V41UpdateStubKV(*pastKeyValues[(size_t)i * block_cnt].first, *pastKeyValues[(size_t)i * block_cnt].second,
                            segments[i].state->totalLen);
        }
        return ret;
    }

    // ==================== decode / DSpark 校验的 CUDA Graph ====================
    //
    // 整段前向没办法一次捕获：Engram 查表在 CPU 上做、压缩 KV 与 indexer key 用
    // Expansion + CatDirect 追加（写偏移是 host 状态、容量按 1.5 倍增长会重新分配）、
    // 滑窗 KV 是环形缓冲（写位置随 token 变）、两级 indexer 的候选数随上下文增长、
    // 路由专家还可能落在 cpu / numa 上。这些"随 token 变化"的部分全部集中在每层的
    // 注意力核心与 MoE 两处。
    //
    // 因此这里按层做分段捕获，每层只捕获两段与位置无关的纯 GPU 计算：
    //   pre  : hc_attn 混合 -> attn_norm -> wq_a / q_norm / wq_b、wkv / kv_norm
    //          -> compressor 的 wkv / wgate 投影、indexer 的 wq_b / weights_proj 投影
    //   post : wo_a -> wo_b -> hc 残差 -> hc_ffn 混合 -> ffn_norm
    //          -> 路由 gate + SelectExpert -> 共享专家
    // 两段之间（RoPE、压缩追加、indexer 打分与 top-k、稀疏注意力、滑窗写入）以及
    // post 之后（MergeMOEBlock、共享专家相加、hc 残差）保持逐算子执行。
    //
    // 这样图里不含任何 startPos、缓存长度或缓存指针：图一旦捕获，在权重与设备布局
    // 不变的前提下一直有效，上下文增长不会让它失效，也不需要为 KV 预分配上下文上限。
    // 代价是每步仍有约 9 个逐算子调用/层（注意力核心 + MoE 前后），捕获掉的是
    // 每层约 20 个算子里的全部稠密计算。
    //
    // 图只读写权重和下面这个常驻工作区，不碰任何请求私有的状态，所以整个模型共用
    // 每种 token 数一份图。并发前向用 try_lock 抢工作区，抢不到就走逐算子。

    // 每层的分段数：pre / post / route / sharedExpert
    static constexpr int kV41GraphSegmentsPerLayer = 4;

    // 回放时被跳过的 host 侧形状变更。捕获时记下，回放时原样补上。
    struct DeepSeekV41GraphSegmentMeta {
        bool containsGpuMoe = false;
        int composedLast = -1;
        std::vector<int> qDims, kvDims, qIdxDims;
        std::vector<int> ffnDims, ffnInputDims, expertIndexDims, expertScoreDims;
        bool swapHidden = false;
    };

#ifdef USE_CUDA
    DeepSeekV41Model::ScopedTpDispatch::ScopedTpDispatch(
            bool enabled, const std::vector<int> &devices) : enabled(enabled), devices(devices) {
        if (enabled) previous = MultiCudaSetPersistentAsyncDispatch(true);
    }

    DeepSeekV41Model::ScopedTpDispatch::~ScopedTpDispatch() {
        if (enabled) {
            const int originalDevice = FastllmCudaGetDevice();
            // Keep the device-wide lifetime barrier, but let each persistent
            // worker drain its own device concurrently. Serial device switches
            // on the caller can cost more than the final decode operations.
            if (!MultiCudaRunDeviceCallbacksEnqueueOnly(devices,
                    [](int, int device) { FastllmCudaSyncDevice(device); })) {
                for (int device : devices) FastllmCudaSyncDevice(device);
            }
            FastllmCudaSetDevice(originalDevice);
            MultiCudaSetPersistentAsyncDispatch(previous);
        }
    }

    namespace {
        int V41DecodeCudaGraphWarmupRounds() {
            static const int rounds = []() -> int {
                const char *env = std::getenv("FASTLLM_DSV41_CUDA_GRAPH_WARMUP");
                int v = env != nullptr && env[0] != '\0' ? atoi(env) : 0;
                return v > 0 ? v : 2;
            }();
            return rounds;
        }

        // 排查用：位 0～3 分别回放 pre / post / route / sharedExpert，默认全开。
        int V41DecodeCudaGraphReplayMask() {
            static const int mask = []() -> int {
                const char *env = std::getenv("FASTLLM_DSV41_CUDA_GRAPH_REPLAY_MASK");
                return env != nullptr && env[0] != 0 ? atoi(env) : 15;
            }();
            return mask;
        }

        // 排查用：让第 N 段的捕获强制失败，验证"就地退回逐算子"这条回退路径
        int V41DecodeCudaGraphFailAt() {
            static const int at = []() -> int {
                const char *env = std::getenv("FASTLLM_DSV41_CUDA_GRAPH_FAIL_AT");
                return env != nullptr && env[0] != '\0' ? atoi(env) : -1;
            }();
            return at;
        }

        // 排查用：每 N 次回放模拟地址失效；普通 decode 退回 eager，verify 可重新捕获
        int V41DecodeCudaGraphInvalidateEvery() {
            static const int n = []() -> int {
                const char *env = std::getenv("FASTLLM_DSV41_CUDA_GRAPH_INVALIDATE_EVERY");
                return env != nullptr && env[0] != '\0' ? atoi(env) : 0;
            }();
            return n;
        }

        bool V41DecodeCudaGraphVerbose() {
            static const bool v = V41EnvFlag("FASTLLM_DSV41_CUDA_GRAPH_DEBUG");
            return v;
        }

        struct DeepSeekV41GraphDeviceState {
            int device = -1;
            void *workerStartEvent = nullptr;
            void *workerEndEvent = nullptr;
        };

        // 一段 = 每个 TP rank 一张图，外加回放时要补上的 host 侧形状变更。
        struct DeepSeekV41GraphSegment : DeepSeekV41GraphSegmentMeta {
            std::vector<void*> graphs;
            std::vector<void*> execs;
            std::vector<void*> composedGraphs, composedExecs;
            // 这一段本来就进不了图（例如专家数超过 CUDA 路由 kernel 的上限、
            // 只能走 CPU 参考路由），占位保持段号对齐，捕获与回放时都逐算子执行。
            bool passthrough = false;
        };

        struct DeepSeekV41CudaGraphState {
            bool disabled = false;             // 捕获或回放失败后永久退回逐算子
            bool captured = false;
            bool capturing = false;
            bool replaying = false;
            bool captureFailed = false;
            int warmupRounds = 0;
            int blockCnt = 0;
            std::string signature;             // 设备布局 / dtype 等，变化时重新捕获
            std::vector<int> devices;          // 参与的 CUDA 设备（单卡时只有一个）
            bool tensorParallel = false;
            std::vector<DeepSeekV41GraphDeviceState> deviceStates;
            std::vector<DeepSeekV41GraphSegment> segments;
            std::vector<void*> reservedPointers;
            // 捕获时工作区里各边界张量的设备地址。回放前逐一核对，任何一个搬了家
            // 都说明图里烤死的地址已经失效，销毁重捕获；反复失效就彻底关掉。
            std::vector<const void*> boundaryPointers;
            int recaptureCount = 0;
            long long replayCount = 0;
            int composedThrough = -1;
            std::unique_ptr<DeepSeekV41DecodeWorkspace> workspace;

            DeepSeekV41CudaGraphState() : workspace(new DeepSeekV41DecodeWorkspace()) {}

            std::shared_ptr<void> fullDecode;

            void DestroyCapturedGraph() {
                fullDecode.reset();
                const int oriDevice = FastllmCudaGetDevice();
                for (auto &segment : segments) {
                    for (size_t i = 0; i < segment.composedExecs.size(); ++i) {
                        FastllmCudaSetDevice(deviceStates[i].device);
                        if (segment.composedExecs[i]) FastllmCudaGraphExecDestroy(segment.composedExecs[i]);
                        if (segment.composedGraphs[i]) FastllmCudaGraphDestroy(segment.composedGraphs[i]);
                    }
                    for (size_t i = 0; i < segment.execs.size() && i < deviceStates.size(); i++) {
                        if (segment.execs[i] != nullptr) {
                            FastllmCudaSetDevice(deviceStates[i].device);
                            FastllmCudaGraphExecDestroy(segment.execs[i]);
                        }
                    }
                    for (size_t i = 0; i < segment.graphs.size() && i < deviceStates.size(); i++) {
                        if (segment.graphs[i] != nullptr) {
                            FastllmCudaSetDevice(deviceStates[i].device);
                            FastllmCudaGraphDestroy(segment.graphs[i]);
                        }
                    }
                }
                segments.clear();
                boundaryPointers.clear();
                FastllmCudaSetDevice(oriDevice);
                if (!reservedPointers.empty()) {
                    FastllmCudaGraphMemoryPoolRelease(reservedPointers);
                    reservedPointers.clear();
                }
                captured = false;
                capturing = false;
                replaying = false;
                captureFailed = false;
                warmupRounds = 0;
            }

            void ResetWorkspace() {
                workspace.reset(new DeepSeekV41DecodeWorkspace());
            }

            ~DeepSeekV41CudaGraphState() {
                DestroyCapturedGraph();
                workspace.reset();
                const int oriDevice = FastllmCudaGetDevice();
                for (auto &deviceState : deviceStates) {
                    FastllmCudaSetDevice(deviceState.device);
                    if (deviceState.workerStartEvent != nullptr) {
                        FastllmCudaEventDestroy(deviceState.workerStartEvent);
                    }
                    if (deviceState.workerEndEvent != nullptr) {
                        FastllmCudaEventDestroy(deviceState.workerEndEvent);
                    }
                }
                FastllmCudaSetDevice(oriDevice);
            }

            void PrepareDevices(const std::vector<int> &nextDevices, bool tp) {
                if (!deviceStates.empty()) {
                    return;
                }
                tensorParallel = tp;
                devices = nextDevices;
                const int oriDevice = FastllmCudaGetDevice();
                for (int device : nextDevices) {
                    DeepSeekV41GraphDeviceState deviceState;
                    deviceState.device = device;
                    if (tp) {
                        FastllmCudaSetDevice(device);
                        deviceState.workerStartEvent = FastllmCudaEventCreate();
                        deviceState.workerEndEvent = FastllmCudaEventCreate();
                    }
                    deviceStates.push_back(deviceState);
                }
                FastllmCudaSetDevice(oriDevice);
            }
        };

        std::shared_ptr<DeepSeekV41CudaGraphState> V41GetCudaGraphState(std::shared_ptr<void> &slot) {
            if (slot == nullptr) {
                slot = std::shared_ptr<void>(new DeepSeekV41CudaGraphState(), [](void *ptr) {
                    delete (DeepSeekV41CudaGraphState*)ptr;
                });
            }
            return std::shared_ptr<DeepSeekV41CudaGraphState>(
                slot, (DeepSeekV41CudaGraphState*)slot.get());
        }

        void V41ComposeGraphSegments(DeepSeekV41CudaGraphState &state,
                                     bool crossLayer, const std::vector<int> &engramLayers) {
            if (!state.tensorParallel || (V41DecodeCudaGraphReplayMask() & 14) != 14) return;
            const int original = FastllmCudaGetDevice();
            for (int first = 1; first + 2 < (int)state.segments.size(); first += 4) {
                if (!state.segments[first + 2].containsGpuMoe) continue;
                bool eligible = true;
                for (int i = first; i <= first + 2; ++i)
                    eligible &= !state.segments[i].passthrough &&
                        state.segments[i].graphs.size() == state.deviceStates.size();
                if (!eligible) continue;
                int last = first + 2;
                const int nextLayer = (first + 3) / kV41GraphSegmentsPerLayer;
                if (crossLayer && (V41DecodeCudaGraphReplayMask() & 1) &&
                    first + 3 < (int)state.segments.size() &&
                    std::find(engramLayers.begin(), engramLayers.end(), nextLayer) == engramLayers.end() &&
                    !state.segments[first + 3].passthrough &&
                    state.segments[first + 3].graphs.size() == state.deviceStates.size()) last = first + 3;
                auto &target = state.segments[first];
                target.composedLast = last;
                const int ranks = (int)state.deviceStates.size();
                target.composedGraphs.assign(ranks, nullptr);
                target.composedExecs.assign(ranks, nullptr);
                for (int rank = 0; rank < ranks && eligible; ++rank) {
                    FastllmCudaSetDevice(state.deviceStates[rank].device);
                    std::vector<const void *> children;
                    for (int child = first; child <= last; ++child)
                        children.push_back(state.segments[child].graphs[rank]);
                    eligible = FastllmCudaGraphCompose(children.data(), (int)children.size(),
                        &target.composedGraphs[rank], &target.composedExecs[rank]);
                }
                if (!eligible) {
                    for (int rank = 0; rank < ranks; ++rank) {
                        FastllmCudaSetDevice(state.deviceStates[rank].device);
                        if (target.composedExecs[rank]) FastllmCudaGraphExecDestroy(target.composedExecs[rank]);
                        if (target.composedGraphs[rank]) FastllmCudaGraphDestroy(target.composedGraphs[rank]);
                    }
                    target.composedExecs.clear();
                    target.composedGraphs.clear();
                }
            }
            FastllmCudaSetDevice(original);
        }

        // 回放一段：单卡直接 Launch；TP 下每个 rank 一张图，必须由各自的常驻 worker
        // 线程同时发射（图里含 rank 间的集合通信，只发射 root 会把其它 rank 挂死）。
        bool V41LaunchGraphSegment(DeepSeekV41CudaGraphState &state, int index,
                                   bool workerOrderedReplay = false) {
            if (index < 0 || index >= (int)state.segments.size() || state.deviceStates.empty()) {
                return false;
            }
            if (index == 0) state.composedThrough = -1;
            if (state.captured && index <= state.composedThrough) return true;
            DeepSeekV41GraphSegment &segment = state.segments[index];
            const bool composed = state.captured && !segment.composedExecs.empty();
            const auto &execs = composed ? segment.composedExecs : segment.execs;
            if (execs.size() != state.deviceStates.size()) {
                return false;
            }
            if (!state.tensorParallel) {
                FastllmCudaSetDevice(state.deviceStates[0].device);
                return execs[0] != nullptr && FastllmCudaGraphLaunch(execs[0]);
            }
            std::vector<int> devices;
            devices.reserve(state.deviceStates.size());
            for (auto &deviceState : state.deviceStates) {
                devices.push_back(deviceState.device);
            }
            const int launchCount = (int)devices.size();
            std::vector<int> launchOk(devices.size(), 0);
            std::atomic<int> launchReady{0};
            std::function<void(int, int)> launchOne = [&](int rank, int device) {
                bool ready = rank >= 0 && rank < launchCount &&
                             state.deviceStates[rank].device == device &&
                             execs[rank] != nullptr;
                launchReady.fetch_add(1, std::memory_order_release);
                while (launchReady.load(std::memory_order_acquire) < launchCount) {
                    std::this_thread::yield();
                }
                if (ready) {
                    FastllmCudaSetDevice(device);
                    launchOk[rank] = FastllmCudaGraphLaunch(execs[rank]) ? 1 : 0;
                }
            };
            bool previousAsync = MultiCudaSetPersistentAsyncDispatch(true);
            // Ordinary GPU decode produces and consumes graph boundaries on
            // the same persistent rank streams. The host only updates metadata
            // between them, so no round trip through the caller streams is
            // needed. Capture validation and paths with host consumers retain
            // the normal bidirectional handoff.
            bool launched = workerOrderedReplay && state.captured &&
                MultiCudaRunDeviceCallbacksEnqueueOnly(devices, launchOne);
            if (!launched) launched = MultiCudaRunDeviceCallbacks(devices, launchOne);
            MultiCudaSetPersistentAsyncDispatch(previousAsync);
            const bool ok = launched && std::all_of(launchOk.begin(), launchOk.end(),
                                                    [](int v) { return v != 0; });
            if (ok && composed) state.composedThrough = segment.composedLast;
            return ok;
        }

        // 捕获一段。TP 下用 workerStart / workerEnd 事件把常驻 worker 的 stream 接进
        // 调用线程正在捕获的图（V4 的做法），捕获结束后立刻回放一次校验。
        bool V41CaptureGraphSegment(DeepSeekV41CudaGraphState &state,
                                    const std::function<void()> &body) {
            if (state.deviceStates.empty() || state.captureFailed) {
                return false;
            }
            const int originalDevice = FastllmCudaGetDevice();
            const int deviceCount = (int)state.deviceStates.size();
            DeepSeekV41GraphSegment segment;
            segment.graphs.assign(deviceCount, nullptr);
            segment.execs.assign(deviceCount, nullptr);

            int begunCaptures = 0;
            bool captureOk = true;
            for (auto &deviceState : state.deviceStates) {
                FastllmCudaSetDevice(deviceState.device);
                if (!FastllmCudaGraphBeginCapture()) {
                    captureOk = false;
                    break;
                }
                begunCaptures++;
            }

            std::vector<int> devices;
            std::vector<void*> workerStartEvents, workerEndEvents;
            for (auto &deviceState : state.deviceStates) {
                devices.push_back(deviceState.device);
                workerStartEvents.push_back(deviceState.workerStartEvent);
                workerEndEvents.push_back(deviceState.workerEndEvent);
            }

            bool workersJoined = false;
            if (captureOk && state.tensorParallel) {
                for (auto &deviceState : state.deviceStates) {
                    FastllmCudaSetDevice(deviceState.device);
                    FastllmCudaEventRecordCurrentThread(deviceState.workerStartEvent);
                }
                workersJoined = MultiCudaGraphWorkersWaitEvents(devices, workerStartEvents);
                captureOk = workersJoined;
            }

            if (captureOk) {
                state.capturing = true;
                body();
                state.capturing = false;
                captureOk = !FastllmCudaGetThreadError() && !FastllmCudaGetGraphError();
            }

            if (workersJoined) {
                if (!MultiCudaGraphWorkersRecordEvents(devices, workerEndEvents)) {
                    captureOk = false;
                }
                for (auto &deviceState : state.deviceStates) {
                    FastllmCudaSetDevice(deviceState.device);
                    FastllmCudaCurrentThreadStreamWaitEvent(deviceState.workerEndEvent);
                }
            }

            if (begunCaptures == deviceCount) {
                for (auto &deviceState : state.deviceStates) {
                    FastllmCudaSetDevice(deviceState.device);
                    if (FastllmCudaGraphCaptureInvalidated()) {
                        captureOk = false;
                    }
                }
            } else {
                captureOk = false;
            }

            bool endOk = begunCaptures == deviceCount;
            for (int index = 0; index < begunCaptures; index++) {
                FastllmCudaSetDevice(state.deviceStates[index].device);
                void *graph = nullptr;
                bool oneEndOk = FastllmCudaGraphEndCapture(&graph) && graph != nullptr;
                if (oneEndOk) {
                    segment.graphs[index] = graph;
                } else if (graph != nullptr) {
                    FastllmCudaGraphDestroy(graph);
                }
                endOk &= oneEndOk;
            }
            captureOk &= endOk;

            if (captureOk) {
                for (int index = 0; index < deviceCount; index++) {
                    FastllmCudaSetDevice(state.deviceStates[index].device);
                    if (!FastllmCudaGraphInstantiate(segment.graphs[index], &segment.execs[index]) ||
                        segment.execs[index] == nullptr) {
                        captureOk = false;
                        break;
                    }
                }
            }

            if (captureOk) {
                state.segments.push_back(segment);
                captureOk = V41LaunchGraphSegment(state, (int)state.segments.size() - 1);
                if (!captureOk) {
                    segment = state.segments.back();
                    state.segments.pop_back();
                }
            }

            if (!captureOk) {
                for (int index = 0; index < deviceCount; index++) {
                    FastllmCudaSetDevice(state.deviceStates[index].device);
                    if (segment.execs[index] != nullptr) {
                        FastllmCudaGraphExecDestroy(segment.execs[index]);
                    }
                    if (segment.graphs[index] != nullptr) {
                        FastllmCudaGraphDestroy(segment.graphs[index]);
                    }
                }
                state.captureFailed = true;
            }
            state.capturing = false;
            FastllmCudaSetDevice(originalDevice);
            return captureOk;
        }
    }
#endif

    struct DeepSeekV41Model::ForwardWorkspace {
        DeepSeekV41Model &model;
        HybridPrefillStage *prefill = nullptr;
        const std::vector<DeepSeekV41Segment> &inputSegments;
        const Data &inputIds;
        const Data *inputEmbeds;
        const std::vector<int> *imageMask;
        const std::vector<GenerationConfig> &generationConfigsIn;
        const LastTokensManager &lastTokens;
        std::vector<std::vector<float>*> *retLogits;
        std::vector<std::pair<Data*, Data*> > &samplingPastKeyValues;
        int numSegments = 0;
        bool boundedPrefill = false;
        int total = 0;
        std::vector<DeepSeekV41Segment> tailSegments;
        bool single = false;
        bool dsparkVerify = false;
        int seqlen = 0;
        bool hasImageTokens = false;
        bool fp8KV = false;
        bool fp4KV = false;
        bool quantKV = false;
        int cmpQuantMode = 1;
        int cmpQuantBlock = 32;
        int idxQuantMode = 1;
        int idxQuantBlock = 32;
        std::vector<int> tpDevices;
        bool tp = false;
        bool tpAttention = false;
        DeepSeekV41DecodeWorkspace localWorkspace;
        DeepSeekV41DecodeWorkspace *ws = &localWorkspace;
        bool graphActive = false;
        int dim = 0;
        int headDim = 0;
        float softmaxScale = 0;
        int indexTopK = 0;
        V41RopeParams windowRope{};
        V41RopeParams compressRope{};
        std::optional<HybridMoeDecodeStep> moeStep;
        Data *curHidden = nullptr;
        Data *nextHidden = nullptr;
        bool dumpDebug = false;
        int startPos0 = 0;
        std::string dumpSuffix;
        std::vector<char> segHasCandidates;
        std::vector<int> segHasTopK;
        Data *preMixPtr = nullptr;
        Data catTmp[2];
        bool cudaRouteCapturable = false;
        std::vector<Data*> moeWeights;
        bool hasSharedExpertOut = false;
        std::vector<int> ffnDims;
        bool combineFfnPostDispatch = false;
        bool captureGpuMoeTogether = false;
        int endedLayerInGraph = -1;
#ifdef USE_CUDA
        std::shared_ptr<DeepSeekV41CudaGraphState> graphState;
        std::unique_lock<std::mutex> graphLock;
        bool graphReplay = false;
        bool graphCapture = false;
        Data *fullGraphHead = nullptr, *fullGraphLogits = nullptr;
        bool graphPoolOpen = false;
        bool workerOrderedGraphReplay = false;
        bool decodeQkvViewsPrepared = false;
        bool decodeOutputViewsPrepared = false;
        std::unique_ptr<ScopedTpDispatch> tpDecodeDispatch;
#endif
        ForwardWorkspace(DeepSeekV41Model &model, const std::vector<DeepSeekV41Segment> &inputSegments,
                                                       const Data &inputIds,
                                                       const Data *inputEmbeds,
                                                       const std::vector<int> *imageMask,
                                                       const std::vector<GenerationConfig> &generationConfigsIn,
                                                       const LastTokensManager &lastTokens,
                                                       std::vector<std::vector<float>*> *retLogits,
                                                       std::vector<std::pair<Data*, Data*> > &samplingPastKeyValues)
          : model(model), inputSegments(inputSegments), inputIds(inputIds), inputEmbeds(inputEmbeds), imageMask(imageMask), generationConfigsIn(generationConfigsIn), lastTokens(lastTokens), retLogits(retLogits), samplingPastKeyValues(samplingPastKeyValues) {}
    };
    DeepSeekV41Model::ForwardState::ForwardState(DeepSeekV41Model &model, const std::vector<DeepSeekV41Segment> &inputSegments,
                                                       const Data &inputIds,
                                                       const Data *inputEmbeds,
                                                       const std::vector<int> *imageMask,
                                                       const std::vector<GenerationConfig> &generationConfigsIn,
                                                       const LastTokensManager &lastTokens,
                                                       std::vector<std::vector<float>*> *retLogits,
                                                       std::vector<std::pair<Data*, Data*> > &samplingPastKeyValues, HybridPrefillStage *prefill)
        : work(new ForwardWorkspace(model, inputSegments, inputIds, inputEmbeds, imageMask, generationConfigsIn, lastTokens, retLogits, samplingPastKeyValues)) {
        work->prefill = prefill;
        model.PrepareForward(*work);
    }
    DeepSeekV41Model::ForwardState::~ForwardState() = default;
    void DeepSeekV41Model::ForwardState::PrepareInputs() { work->model.PrepareForwardInputs(*work); }
    bool DeepSeekV41Model::ForwardState::FullGraph() {
#ifdef USE_CUDA
        return work->model.TryFullDecodeGraph(*work);
#else
        return false;
#endif
    }
    int DeepSeekV41Model::ForwardState::Layers() const {
        const int end = work->inputSegments[0].prefillCacheOnly ?
            work->model.decoderSwaTailLayer : work->model.block_cnt;
        return work->prefill ? std::min(end, work->prefill->endLayer) : end;
    }
    void DeepSeekV41Model::ForwardState::BeginLayer(int layer) { work->model.BeginForwardLayer(*work, layer); }
    void DeepSeekV41Model::ForwardState::EndLayer(int layer) {
        if (work->endedLayerInGraph != layer) work->model.EndForwardLayer(*work, layer);
    }
    void DeepSeekV41Model::ForwardState::BeginHybridStep() {
        if (work->model.executionMode == ExecutionMode::Hybrid) {
            work->moeStep.emplace();
            work->model.BeginHybridMoeStep(*work->moeStep, work->tp, work->single, work->seqlen);
        }
    }
    void DeepSeekV41Model::ForwardState::GpuExperts(int layer) {
        auto ctx=work->model.MakeForwardMoeContext(*work, layer);
#ifdef USE_CUDA
        bool ready = work->graphActive && work->tp && work->cudaRouteCapturable &&
            work->ws->ffnInput.dataType == DataType::BFLOAT16 &&
            work->model.quantizedLinearNames.count("layers." + std::to_string(layer) + ".ffn.experts.0.w1.weight");
        if (ready) {
            for (int device : work->tpDevices) {
                const Data *key = nullptr;
                for (size_t slot = 2; slot < ctx.moeWeights.size(); slot += 2) {
                    if (!ctx.moeWeights[slot]) continue;
                    auto it = ctx.moeWeights[slot]->multiDeviceDatas.find(device);
                    if (it != ctx.moeWeights[slot]->multiDeviceDatas.end() && it->second) {
                        key = it->second;
                        break;
                    }
                }
                ready &= key && FastllmCudaV41MarlinGraphReady(key, work->seqlen);
            }
        }
        if (work->graphReplay) {
            const int index = kV41GraphSegmentsPerLayer * layer + 3;
            const bool captured = index < (int)work->graphState->segments.size() &&
                                  work->graphState->segments[index].containsGpuMoe;
            if (captured && !ready) work->model.GiveUpForwardGraph(*work, "MoE buffers");
            if (!captured) ready = false;
        }
        if (ready) {
            work->captureGpuMoeTogether = true;
            work->combineFfnPostDispatch = false;
            work->model.RunForwardGraphSegment(*work, [&] {
                    work->combineFfnPostDispatch = work->model.RunGpuMoe(ctx);
                    work->model.EndForwardLayer(*work, layer);
                }, kV41GraphSegmentsPerLayer * layer + 3, true,
                [](DeepSeekV41GraphSegmentMeta &meta) { meta.containsGpuMoe = true; },
                [&](const DeepSeekV41GraphSegmentMeta &) {
                    work->ws->ffnOut.Reshape(work->ffnDims);
                    std::swap(work->curHidden, work->nextHidden);
                    work->preMixPtr = &work->ws->ffnPre;
                });
            work->endedLayerInGraph = layer;
            work->captureGpuMoeTogether = false;
            return;
        }
#endif
        work->combineFfnPostDispatch=work->model.RunGpuMoe(ctx);
    }
    void DeepSeekV41Model::ForwardState::HybridExperts(int layer) {
        auto ctx=work->model.MakeForwardMoeContext(*work, layer);
        work->combineFfnPostDispatch=work->model.RunHybridMoe(ctx, work->moeStep ? &*work->moeStep : nullptr);
    }
    std::vector<int> DeepSeekV41Model::ForwardState::Finish() { return work->model.FinishForward(*work); }
    void DeepSeekV41Model::PrepareForward(ForwardWorkspace &f) {
        auto &inputSegments = f.inputSegments;
        auto &inputIds = f.inputIds;
        auto &inputEmbeds = f.inputEmbeds;
        auto &imageMask = f.imageMask;
        auto &generationConfigsIn = f.generationConfigsIn;
        auto &numSegments = f.numSegments;
        auto &boundedPrefill = f.boundedPrefill;
        auto &total = f.total;
        auto &tailSegments = f.tailSegments;
        auto &single = f.single;
        auto &dsparkVerify = f.dsparkVerify;
        auto &seqlen = f.seqlen;
        auto &hasImageTokens = f.hasImageTokens;
        auto &fp8KV = f.fp8KV;
        auto &fp4KV = f.fp4KV;
        auto &quantKV = f.quantKV;
        auto &cmpQuantMode = f.cmpQuantMode;
        auto &cmpQuantBlock = f.cmpQuantBlock;
        auto &idxQuantMode = f.idxQuantMode;
        auto &idxQuantBlock = f.idxQuantBlock;
        auto &tpDevices = f.tpDevices;
        auto &tp = f.tp;
        auto &tpAttention = f.tpAttention;
        auto &ws = f.ws;
        auto &graphActive = f.graphActive;
        auto &dim = f.dim;
        auto &headDim = f.headDim;
        auto &softmaxScale = f.softmaxScale;
        auto &indexTopK = f.indexTopK;
        auto &windowRope = f.windowRope;
        auto &compressRope = f.compressRope;
#ifdef USE_CUDA
        auto &graphState = f.graphState;
        auto &graphLock = f.graphLock;
        auto &graphReplay = f.graphReplay;
        auto &graphCapture = f.graphCapture;
        auto &graphPoolOpen = f.graphPoolOpen;
        auto &tpDecodeDispatch = f.tpDecodeDispatch;
#endif
        numSegments = (int)inputSegments.size();
        AssertInFastLLM(numSegments >= 1 && (int)generationConfigsIn.size() == numSegments,
                        "DeepSeekV41Model::ForwardSegments: bad segments.");
        boundedPrefill = decoderSwaTailLayer >= 0 && decoderSwaTailLayer < block_cnt &&
                              inputEmbeds == nullptr && imageMask == nullptr;
        bool hasLongSegment = false;
        total = 0;
        for (const auto &seg : inputSegments) {
            AssertInFastLLM(seg.state && seg.seqlen > 0 && seg.offset == total && seg.state->totalLen == seg.startPos,
                            "DeepSeekV41Model::ForwardSegments: inconsistent segment.");
            total += seg.seqlen;
            hasLongSegment |= seg.seqlen > window_size;
            if (seg.spec != nullptr && (seg.spec->wantAllTokens || seg.spec->deferWindow)) {
                boundedPrefill = false; // verify 必须计算每个候选位置
            }
        }
        boundedPrefill &= hasLongSegment;
        for (const auto &seg : inputSegments) if (seg.prefillCacheOnly) {
            AssertInFastLLM(numSegments == 1 && boundedPrefill && !v41DsparkEnabled && !seg.spec,
                "DeepSeekV41: cache-only chunks require single-sequence bounded prefill.");
        }
        // 只裁剪本次前向的视图，调用方仍按完整片段提交请求 / DSpark 的逻辑位置。

        if (boundedPrefill) {
            tailSegments = inputSegments;
        }

        const auto &segments = boundedPrefill ? tailSegments : inputSegments;
        single = numSegments == 1;
        dsparkVerify = single && segments[0].spec != nullptr && segments[0].spec->wantAllTokens;
        seqlen = total;   // 当前层实际计算的 token 数；请求长度仍由 inputSegments 保留
        // 图像 token（掩码按拼接后的全局下标）：Engram 置 -1，专家选择改用 gate.bias_vl
        AssertInFastLLM(imageMask == nullptr || (int)imageMask->size() == seqlen,
                        "DeepSeekV41Model::ForwardSegments: imageMask length mismatch.");
        hasImageTokens = false;
        if (imageMask != nullptr) {
            for (int v : *imageMask) {
                if (v != 0) {
                    hasImageTokens = true;
                    break;
                }
            }
        }
        // --kv_cache_dtype 控制长期 KV 的存储精度（默认 BF16）：
        //   fp8_e4m3：滑窗 / 压缩 KV 为 FP8 + UE8M0，indexer key 为 FP8 + UE8M0
        //   fp4_e2m1：滑窗 KV 仍为 FP8（它本来就在 FP8 网格上），压缩 KV 为 FP4 + E4M3（每 16 个一组），
        //             indexer key 为 FP4 + UE8M0（每 32 个一组），与伪量化的网格 / 分组完全一致
        fp8KV = this->kvCacheDataType == DataType::FP8_E4M3;
        fp4KV = this->kvCacheDataType == DataType::FP4_E2M1;
        quantKV = fp8KV || fp4KV;
        // (quantMode, quantBlock)：压缩 KV 与 indexer key
        cmpQuantMode = fp4KV ? 3 : 1; cmpQuantBlock = fp4KV ? 16 : 32;
        idxQuantMode = fp4KV ? 2 : 1; idxQuantBlock = 32;

        // ---- 张量并行 ----
        // deviceMap 为 multicuda 且不止一张卡时启用：q / attn_sink / 注意力输出 / wo_a
        // 按 query head 切分，wo_b 按列切并 all-reduce，共享专家保持复制。
        // FastllmGetMulticudaDeviceAndRatio 读的是 ApplyDeviceMap 设置的全局设备表，
        // 首次前向前必须先应用一次主 device map，否则拿到的是空列表。
        ApplyDeviceMap(this->deviceMap, f.prefill ? f.prefill->firstLayer + 1 : 1, block_cnt);
        tpDevices = V41TpDevices(this->deviceMap);
        tp = !tpDevices.empty();
        // 与 InitParams 使用相同的 head/o_group 对齐条件；kernel 内部处理尾部 head。
        const int tpRanks = (int)tpDevices.size();
        const bool tpHeadsAligned =
            tpRanks > 0 && o_groups > 0 &&
            num_attention_heads % tpRanks == 0 &&
            num_attention_heads % o_groups == 0 &&
            (num_attention_heads / tpRanks) % (num_attention_heads / o_groups) == 0;
        // InitParams 已经在 head 数切不开时把 device map 改回单卡（那时 tp 为假），
        // 这里的 tpHeadsAligned 只是同一条约束的兜底。
        // 排查时可关闭注意力 head 切分，改为每卡复制计算。
        tpAttention = tp && tpHeadsAligned &&
                                 !V41EnvFlag("FASTLLM_DSV41_DISABLE_TP_ATTENTION");
        if (std::getenv("FASTLLM_TRACE_OPS") != nullptr) {
            static bool printed = false;
            if (!printed) {
                printed = true;
                fprintf(stderr, "[v41tp] tp=%d devices=%d deviceMap:", (int)tp, (int)tpDevices.size());
                for (auto &it : this->deviceMap) {
                    fprintf(stderr, " '%s'=%d", it.first.c_str(), it.second);
                }
                fprintf(stderr, "\n");
                fflush(stderr);
            }
        }

        // ---- Decode batch / DSpark verification CUDA graphs ----
        // 按本次 token 数缓存图，不按上下文长度捕获；特征采集与 KV 回滚留在图外。

        graphActive = false;      // 本次前向使用常驻工作区（预热 / 捕获 / 回放）
#ifdef USE_CUDA

        graphReplay = false;      // 本次前向回放已捕获的图
        graphCapture = false;     // 本次前向做捕获
        // 张量 dump 与逐算子同步都会在捕获中插入 host 侧拷贝 / 同步，直接关掉图。
        const bool decodeBatch = std::all_of(segments.begin(), segments.end(),
            [](const DeepSeekV41Segment &seg) { return seg.seqlen == 1 && seg.startPos > 0; });
        const bool verifyGraph = dsparkVerify && segments[0].spec->deferWindow && segments[0].startPos > 0;
        if (v41MoeCudaGraphCompatible && (decodeBatch || verifyGraph) &&
            inputEmbeds == nullptr && !hasImageTokens &&
            std::getenv("FASTLLM_DSV41_DUMP_DIR") == nullptr &&
            !GetFastllmEnv().cudaSync && !GetFastllmEnv().printProfile &&
            (this->deviceMap.empty() || V41DeviceMapUsesCuda(this->deviceMap)) &&
            // 按层切分不支持分段图；仅允许单卡或 multicuda 张量并行。
            (V41DeviceMapUsesMultiCuda(this->deviceMap) ||
             V41DeviceMapCudaDeviceCount(this->deviceMap) <= 1) &&
            !V41EnvFlag("FASTLLM_DSV41_REFERENCE_MATH") && GetFastllmEnv().cudaGraph) {
            graphLock = std::unique_lock<std::mutex>(v41CudaGraphMutex, std::try_to_lock);
            if (graphLock.owns_lock()) {
                // Verification and independent requests have different host metadata.
                const int graphKey = verifyGraph ? -seqlen : seqlen;
                auto found = v41CudaGraphSlots.find(graphKey);
                if (v41CudaGraphWarmingUp || verifyGraph || found != v41CudaGraphSlots.end()) {
                    graphState = V41GetCudaGraphState(v41CudaGraphSlots[graphKey]);
                }
            }
            if (!graphState || graphState->disabled) {
                graphState.reset();
            } else {
                // 设备布局 / dtype 变化时丢弃旧图（图里烤死了每张卡上的权重地址）
                std::string signature = std::to_string(block_cnt) + "|" + std::to_string((int)tp) +
                                        "|" + std::to_string((int)tpAttention) +
                                        "|" + std::to_string((int)this->kvCacheDataType) +
                                        "|" + std::to_string(num_experts) +
                                        "|" + std::to_string((int)GetCudaSharedExpert()) + "|";
                for (int device : tpDevices) {
                    signature += std::to_string(device) + ",";
                }
                if (graphState->signature != signature) {
                    graphState->DestroyCapturedGraph();
                    graphState->ResetWorkspace();
                    graphState->signature = signature;
                    graphState->blockCnt = block_cnt;
                }
                std::vector<int> graphDevices = tpDevices;
                if (graphDevices.empty()) {
                    int device = FastllmCudaGetDevice();
                    if (device >= 0) {
                        graphDevices.push_back(device);
                    }
                }
                if (graphDevices.empty()) {
                    graphState.reset();
                } else {
                    graphState->PrepareDevices(graphDevices, tp);
                    ws = graphState->workspace.get();
                    graphActive = true;
                    if (graphState->captured &&
                        (int)graphState->segments.size() == kV41GraphSegmentsPerLayer * block_cnt) {
                        graphReplay = true;
                    } else if ((v41CudaGraphWarmingUp || verifyGraph) &&
                               graphState->warmupRounds >= V41DecodeCudaGraphWarmupRounds()) {
                        graphCapture = true;
                    }
                }
            }
        }
        bool tpVerifyWorkspaceActive = false;
        if (!graphActive && tp && dsparkVerify && seqlen > 1 && seqlen <= 8 &&
            segments[0].spec->deferWindow && segments[0].startPos > 0 &&
            inputEmbeds == nullptr && !hasImageTokens &&
            std::getenv("FASTLLM_DSV41_DUMP_DIR") == nullptr &&
            !GetFastllmEnv().cudaSync && !GetFastllmEnv().printProfile &&
            !V41ReferenceMathEnabled() &&
            !V41EnvFlag("FASTLLM_DSV41_DISABLE_TP_VERIFY_ASYNC")) {
            if (!graphLock.owns_lock()) {
                graphLock = std::unique_lock<std::mutex>(v41CudaGraphMutex, std::try_to_lock);
            }
            if (graphLock.owns_lock()) {
                if (!v41TpVerifyWorkspace || v41TpVerifyDevices != tpDevices) {
                    v41TpVerifyWorkspace = std::make_shared<DeepSeekV41DecodeWorkspace>();
                    v41TpVerifyDevices = tpDevices;
                }
                ws = v41TpVerifyWorkspace.get();
                tpVerifyWorkspaceActive = true;
            }
        }
        // The GPU execution path keeps producer/consumer ordering through
        // bidirectional stream events, including eager prefill. Every dispatch
        // publishes its completion event to the caller before temporary tensors
        // can be reused; the scope drains all ranks before workspace destruction.
        // Hybrid CPU expert paths retain their existing synchronization policy.
        tpDecodeDispatch.reset(new ScopedTpDispatch(
            tp && (executionMode == ExecutionMode::Gpu || (single && seqlen == 1) ||
                   graphActive || tpVerifyWorkspaceActive), tpDevices));
        graphPoolOpen = false;
        if (graphCapture) {
            const int oriDevice = FastllmCudaGetDevice();
            for (int device : graphState->devices) {
                FastllmCudaSyncDevice(device);
            }
            bool prepared = true;
            for (int device : graphState->devices) {
                FastllmCudaSetDevice(device);
                if (!FastllmCudaGraphPrepareCaptureDevice()) {
                    prepared = false;
                    break;
                }
            }
            FastllmCudaSetDevice(oriDevice);
            if (prepared) {
                FastllmCudaClearThreadError();
                FastllmCudaClearGraphError();
                prepared = FastllmCudaGraphMemoryPoolBegin();
            }
            if (!prepared) {
                GiveUpForwardGraph(f, "prepare capture");
            } else {
                graphPoolOpen = true;
                graphState->segments.clear();
                graphState->captureFailed = false;
            }
        }
#endif

        // ---- Engram 历史 ----
        std::vector<int> tokenIds = V41ReadTokenIds(inputIds);
        AssertInFastLLM((int)tokenIds.size() == total, "DeepSeekV41Model::ForwardSegments: inputIds length mismatch.");
        if (!engram_layer_ids.empty()) {
            for (auto &seg : segments) {
                AssertInFastLLM((int)seg.state->engramHistory.size() == seg.startPos,
                                "DeepSeekV41: engram history is out of sync with the cache.");
                for (int i = 0; i < seg.seqlen; i++) {
                    int tok = tokenIds[seg.offset + i];
                    int compressed = -1;
                    bool isImage = tok == image_token_id ||
                                   (imageMask != nullptr && (int)imageMask->size() > seg.offset + i &&
                                    (*imageMask)[seg.offset + i] != 0);
                    if (!isImage && engramMeta.loaded && tok >= 0 && tok < (int)engramMeta.tokenMap.size()) {
                        compressed = engramMeta.tokenMap[tok];
                    }
                    seg.state->engramHistory.push_back(compressed);
                }
            }
        }

        dim = embed_dim;
        headDim = head_dim_full;
        softmaxScale = 1.0f / std::sqrt((float)headDim);
        indexTopK = index_topk;
        windowRope = {qk_rope_head_dim, rope_base, 0, rope_factor,
                                    rope_scaling_beta_fast, rope_scaling_beta_slow};
        compressRope = {qk_rope_head_dim, compress_rope_theta,
                                      (int)rope_scaling_original_max_position_embeddings, rope_factor,
                                      rope_scaling_beta_fast, rope_scaling_beta_slow};

        PrepareForwardWeights();
    }
    void DeepSeekV41Model::PrepareHybridPrefill() {
        PrepareForwardWeights();
        // Resolve legacy lazy helpers only before concurrent hybrid stages.
        V41IndexScoreChunk(2, 1);
        V41Profiler();
    }
    void DeepSeekV41Model::PrepareForwardWeights() {
        // MoE 权重表（首次构建）
        if (weights.empty()) {
            weights.resize(block_cnt);
            biass.resize(block_cnt);
            auto getWeightPtr = [&](const std::string &name) -> Data* {
                auto it = weight.weight.find(name);
                return it == weight.weight.end() ? nullptr : &it->second;
            };
            for (int layer = 0; layer < block_cnt; layer++) {
                std::string pre = "layers." + std::to_string(layer) + ".ffn";
                weights[layer].push_back(getWeightPtr(pre + ".shared_experts.gateup.weight"));
                weights[layer].push_back(getWeightPtr(pre + ".shared_experts.w2.weight"));
                biass[layer].push_back(nullptr);
                biass[layer].push_back(nullptr);
                for (int expert = 0; expert < num_experts; expert++) {
                    weights[layer].push_back(getWeightPtr(pre + ".experts." + std::to_string(expert) + ".gateup.weight"));
                    weights[layer].push_back(getWeightPtr(pre + ".experts." + std::to_string(expert) + ".w2.weight"));
                    biass[layer].push_back(nullptr);
                    biass[layer].push_back(nullptr);
                }
            }
        }

    }
    void DeepSeekV41Model::PrepareForwardInputs(ForwardWorkspace &f) {
        const auto &segments = f.boundedPrefill ? f.tailSegments : f.inputSegments;
        auto &inputIds = f.inputIds;
        auto &inputEmbeds = f.inputEmbeds;
        auto &numSegments = f.numSegments;
        auto &single = f.single;
        auto &seqlen = f.seqlen;
        auto &hasImageTokens = f.hasImageTokens;
        auto &tp = f.tp;
        auto &ws = f.ws;
        auto &graphActive = f.graphActive;
        auto &dim = f.dim;
        auto &curHidden = f.curHidden;
        auto &nextHidden = f.nextHidden;
        auto &dumpDebug = f.dumpDebug;
        auto &startPos0 = f.startPos0;
        auto &dumpSuffix = f.dumpSuffix;
        auto &segTopK = f.ws->segTopK;
        auto &segCandidate = f.ws->segCandidate;
        auto &segHasCandidates = f.segHasCandidates;
        auto &segHasTopK = f.segHasTopK;
        auto &preMixPtr = f.preMixPtr;
        auto &segQ = f.ws->segQ;
        auto &segKV = f.ws->segKV;
        auto &segAttnOut = f.ws->segAttnOut;
        auto &cudaRouteCapturable = f.cudaRouteCapturable;
#ifdef USE_CUDA
        auto &graphState = f.graphState;
        auto &graphReplay = f.graphReplay;
        auto &graphCapture = f.graphCapture;
        auto &graphPoolOpen = f.graphPoolOpen;
#endif
        // ---- embedding -> hc 份 ----
        // 图模式下这些张量来自常驻工作区，地址必须在两次 decode 之间保持不变。
        Data &hiddenStates = ws->hiddenStates;
        Data &hiddenTemp = ws->hiddenTemp;
        if (f.prefill && f.prefill->firstLayer > 0) {
            Data view;
            view.FakeFrom(*f.prefill->payload, 0);
            view.Resize({1, seqlen, hc_mult, dim});
            hiddenStates.CopyFrom(view);
        } else {
            // embedOut 只活在前导段里，不是图的边界张量，必须保持为局部量：
            // 常驻之后 ToDataType 的原地降精度会往上一轮留下的、更小的设备缓冲里拷贝。
            Data embedOut;
            if (inputEmbeds != nullptr) {
                AssertInFastLLM(inputEmbeds->Count(0) == (uint64_t)seqlen * dim,
                                "DeepSeekV41Model::ForwardSegments: inputEmbeds shape mismatch.");
                embedOut.CopyFrom(*inputEmbeds);
                ToDataType(embedOut, DataType::BFLOAT16);
            } else {
                Embedding(inputIds, weight["embed.weight"], embedOut);
                ToDataType(embedOut, DataType::BFLOAT16);
            }
            embedOut.Reshape({1, seqlen, 1, dim});
            bool repeated = false;
#ifdef USE_CUDA
            if (tp) {
                // Repeat 没有 multicuda 实现，直接走"广播到每卡副本后逐卡 Repeat"，
                // 否则只有 root 被写入、两张卡的副本会失配。
                repeated = MultiCudaRepeatToReplicated(embedOut, 2, hc_mult, hiddenStates);
            }
#endif
            if (!repeated) {
                Repeat(embedOut, 2, hc_mult, hiddenStates);
            }
        }
        curHidden = &hiddenStates;
        nextHidden = &hiddenTemp;
        dumpDebug = single && std::getenv("FASTLLM_DSV41_DUMP_DIR") != nullptr;
        startPos0 = segments[0].startPos;
        dumpSuffix = startPos0 == 0 ? std::string("") : "_p" + std::to_string(startPos0);
        if (dumpDebug) {
            V41DumpTensor(hiddenStates, "fl_embed" + dumpSuffix);
        }

        Data &preMix = ws->preMix;
        // 内容是常量 one-hot，长度没变就不重建：CopyFrom 会先把它搬回 CPU 再搬上卡，
        // 每次换一个设备地址，图就废了。
        if (f.prefill && f.prefill->firstLayer > 0) {
            Data view;
            view.FakeFrom(*f.prefill->payload, uint64_t(seqlen) * hc_mult * dim * 2);
            view.dataType = FLOAT32;
            view.UpdateUnitSize();
            view.Resize({1, seqlen, hc_mult});
            preMix.CopyFrom(view);
        } else if (ws->preMixSeqlen != seqlen) {
            std::vector<float> values((uint64_t)seqlen * hc_mult, 0.0f);
            for (int i = 0; i < seqlen; i++) {
                values[(uint64_t)i * hc_mult] = 1.0f;
            }
            preMix.CopyFrom(Data(DataType::FLOAT32, {1, seqlen, hc_mult}, values));
            ws->preMixSeqlen = seqlen;
        }

        // 片段切片：单片段时直接使用整批张量，多片段时按 offset 拷贝出来
        // 按 axis=1 拼接多个片段（两块临时缓冲交替使用，避免输出与输入别名）

        // 本次前向内跨层共享的 indexer 结果（按片段）
        segTopK.resize(numSegments); segCandidate.resize(numSegments);
        segHasCandidates.assign(numSegments, 0);
        segHasTopK.assign(numSegments, 0);

        // 上一子层产出的 pre 系数。原来用 preMix.CopyFrom(ffnPre) 传递，
        // 但复制布局下 CopyFrom 只会拷贝可能已失效的 root，这里改成指针传递
        // （ffnPre 在被下一层 attention 消费之后才会被覆盖）。
        preMixPtr = &preMix;
        // pre 段产出、注意力核心逐算子消费的张量：也必须常驻（各层形状相同可以共用，
        // 只有写它的那一层会读它）
        // TP + cpu/numa MoE 时，从卡上副本拷出来的路由专家输入（复用同一组缓冲，
        // 避免每层重新分配、也避免栈上 Data 的地址被下游缓存复用）
        segQ.resize(numSegments); segKV.resize(numSegments); segAttnOut.resize(numSegments);
        f.ws->segIndexQ.resize(numSegments); f.ws->segIndexWeight.resize(numSegments);

#ifdef USE_CUDA
        // 工作区里所有跨段传递的张量：图里烤死的就是它们的设备地址。
        if (graphActive && hiddenStates.dataDevice != DataDevice::CUDA) {
            // 残差流没在卡上：deviceMap 为空但执行器选了 CPU，捕获只会得到空图
            if (graphPoolOpen) {
                FastllmCudaGraphMemoryPoolAbort();
                graphPoolOpen = false;
            }
            graphState->DestroyCapturedGraph();
            graphState->disabled = true;
            graphCapture = false;
            graphReplay = false;
        }
        if (graphReplay) {
            std::vector<const void*> current;
            CollectForwardGraphPointers(f, current);
            const int invalidateEvery = V41DecodeCudaGraphInvalidateEvery();
            ++graphState->replayCount;
            const bool forceInvalidate = invalidateEvery > 0 &&
                                         graphState->replayCount % invalidateEvery == 0;
            if (forceInvalidate || current != graphState->boundaryPointers) {
                graphState->DestroyCapturedGraph();
                graphState->recaptureCount++;
                graphReplay = false;
                if ((!v41CudaGraphWarmingUp && !f.dsparkVerify) ||
                    (graphState->recaptureCount > 3 && !forceInvalidate)) {
                    graphState->disabled = true;
                }
                if (V41DecodeCudaGraphVerbose()) {
                    fprintf(stderr, "[Fastllm] DeepSeek-V4.1 decode CUDA graph invalidated "
                                    "(workspace addresses moved, recapture #%d%s)\n",
                            graphState->recaptureCount,
                            graphState->disabled ? ", giving up" : "");
                    fflush(stderr);
                }
            }
            if (graphReplay && graphState->replayCount == 1 && V41DecodeCudaGraphVerbose()) {
                fprintf(stderr, "[Fastllm] DeepSeek-V4.1 decode CUDA graph replay: tokens=%d\n", seqlen);
            }
        }
#endif

        // Both native CUDA score transformation and TopK iterate over the
        // expert dimension; 384 experts do not require a host round-trip.
        cudaRouteCapturable =
            V41EnvFlag("FASTLLM_DSV41_CUDA_GRAPH_FORCE_ROUTE_CAPTURE") ||
            (!hasImageTokens && num_experts > 0 && num_experts_per_tok <= 50 &&
             !V41EnvFlag("FASTLLM_DSV41_DISABLE_CUDA_ROUTE"));

#ifdef USE_CUDA
        // Only the fully GPU, ordinary decode path has worker-only graph
        // boundaries. CPU experts, speculative feature copies, packed KV and
        // partially captured graphs keep the conservative caller-stream order.
        f.workerOrderedGraphReplay = graphReplay && tp && f.tpAttention &&
            V41DecodeCudaGraphReplayMask() == 15 &&
            executionMode == ExecutionMode::Gpu && !v41DsparkEnabled &&
            !f.quantKV && cudaRouteCapturable &&
            std::all_of(f.inputSegments.begin(), f.inputSegments.end(),
                [](const DeepSeekV41Segment &seg) {
                    return seg.seqlen == 1 && seg.startPos > 0 && !seg.spec;
                }) &&
            std::none_of(graphState->segments.begin(), graphState->segments.end(),
                [](const DeepSeekV41GraphSegment &seg) { return seg.passthrough; });
        // A captured shared expert does not imply a captured routed MoE. The
        // generic routed path may use other streams or host staging and needs
        // the caller/worker handoff. Skip it only when every complete MoE is
        // captured on the persistent rank streams.
        if (f.workerOrderedGraphReplay) {
            for (int index = kV41GraphSegmentsPerLayer - 1;
                 index < (int)graphState->segments.size(); index += kV41GraphSegmentsPerLayer) {
                if (!graphState->segments[index].containsGpuMoe) {
                    f.workerOrderedGraphReplay = false;
                    break;
                }
            }
        }
#endif

        // 分段调度：捕获 / 回放 / 逐算子。record 保存回放时被跳过的 host 侧形状变更，
        // restore 在回放时把它们补回来。任何一步失败都就地退回逐算子并永久关掉图。

    }
    void DeepSeekV41Model::BeginForwardLayer(ForwardWorkspace &f, int layer) {
        auto &inputSegments = f.inputSegments;
        auto &imageMask = f.imageMask;
        auto &numSegments = f.numSegments;
        auto &boundedPrefill = f.boundedPrefill;
        auto &tailSegments = f.tailSegments;
        auto &single = f.single;
        auto &seqlen = f.seqlen;
        auto &hasImageTokens = f.hasImageTokens;
        auto &quantKV = f.quantKV;
        auto &cmpQuantMode = f.cmpQuantMode;
        auto &cmpQuantBlock = f.cmpQuantBlock;
        auto &idxQuantMode = f.idxQuantMode;
        auto &idxQuantBlock = f.idxQuantBlock;
        auto &tpDevices = f.tpDevices;
        auto &tp = f.tp;
        auto &tpAttention = f.tpAttention;
        auto &localWorkspace = f.localWorkspace;
        auto &ws = f.ws;
        auto &dim = f.dim;
        auto &headDim = f.headDim;
        auto &softmaxScale = f.softmaxScale;
        auto &indexTopK = f.indexTopK;
        auto &windowRope = f.windowRope;
        auto &compressRope = f.compressRope;
        auto &curHidden = f.curHidden;
        auto &nextHidden = f.nextHidden;
        auto &dumpDebug = f.dumpDebug;
        auto &dumpSuffix = f.dumpSuffix;
        auto &segTopK = f.ws->segTopK;
        auto &segCandidate = f.ws->segCandidate;
        auto &segHasCandidates = f.segHasCandidates;
        auto &segHasTopK = f.segHasTopK;
        auto &preMixPtr = f.preMixPtr;
        auto &segQ = f.ws->segQ;
        auto &segKV = f.ws->segKV;
        auto &segAttnOut = f.ws->segAttnOut;
        auto &catTmp = f.catTmp;
        auto &cudaRouteCapturable = f.cudaRouteCapturable;
        auto &moeWeights = f.moeWeights;
        auto &hasSharedExpertOut = f.hasSharedExpertOut;
        auto &ffnDims = f.ffnDims;
        const auto &segments = boundedPrefill ? tailSegments : inputSegments;
        auto &preMix = ws->preMix;
        auto &attnPre = ws->attnPre;
        auto &attnPost = ws->attnPost;
        auto &attnComb = ws->attnComb;
        auto &ffnPre = ws->ffnPre;
        auto &ffnPost = ws->ffnPost;
        auto &ffnComb = ws->ffnComb;
        auto &x = ws->x;
        auto &attnInput = ws->attnInput;
        auto &qr = ws->qr;
        auto &qNorm = ws->qNorm;
        auto &q = ws->q;
        auto &kv = ws->kv;
        auto &attnOut = ws->attnOut;
        auto &woAOut = ws->woAOut;
        auto &attnProj = ws->attnProj;
        auto &ffnInput = ws->ffnInput;
        auto &expertIndex = ws->expertIndex;
        auto &expertScore = ws->expertScore;
        auto &rawKVAll = ws->rawKVAll;
        auto &rawScoreAll = ws->rawScoreAll;
        auto &qIdxAll = ws->qIdxAll;
        auto &idxWeightsAll = ws->idxWeightsAll;
        auto &cpuMoeInput = ws->cpuMoeInput;
        auto &cpuMoeIndex = ws->cpuMoeIndex;
        auto &cpuMoeScore = ws->cpuMoeScore;
        auto sliceOf = [&](Data &full, int i, Data &tmp) -> Data* {
            if (single) {
                return &full;
            }
#ifdef USE_CUDA
            if (segments[i].seqlen == 1 && segments[i].startPos > 0 &&
                V41DecodeSliceView(tmp, full, segments[i].offset, tpDevices)) return &tmp;
#endif
            Split(full, 1, segments[i].offset, segments[i].offset + segments[i].seqlen, tmp);
            return &tmp;
        };
        auto catSegments = [&](std::vector<Data> &parts, Data *tmp) -> Data* {
            Data *acc = &parts[0];
            int t = 0;
            for (size_t i = 1; i < parts.size(); i++) {
                Cat(*acc, parts[i], 1, tmp[t]);
                acc = &tmp[t];
                t ^= 1;
            }
            return acc;
        };
        auto quantizedLinear = [&](Data &input, const std::string &name, Data &output, bool replicated = false,
                                   Data *scratch = nullptr, bool inputQuantized = false) {
            // Graphs and asynchronous TP decode need stable per-weight buffers.
            // Synchronous TP prefill can release each full-chunk temporary as
            // soon as its linear completes, instead of retaining all layers.
            LinearWithActivationQuant(input, name, output, replicated,
                scratch != nullptr ? scratch : (ws == &localWorkspace && (!tp || seqlen > 1) ?
                    (tp ? nullptr : &ws->quantizedActivation) : &ws->quantizedActivations[name]), inputQuantized,
                ws == &localWorkspace && seqlen > 1 ? nullptr : &ws->reductionBuffers[name]);
        };
            ApplyDeviceMap(this->deviceMap, layer + 1, block_cnt);
            std::string pre = "layers." + std::to_string(layer);
            const int ratio = compress_ratios[layer];
            const V41RopeParams &rope = ratio > 0 ? compressRope : windowRope;

            if (boundedPrefill && layer == decoderSwaTailLayer) {
                // 跨层共享 KV 已完整生成；hidden / HC pre 与逐 query 的索引同步取尾部。
                auto trimBatch = [&](Data &data) {
                    std::vector<Data> parts(numSegments);
                    for (int s = 0; s < numSegments; s++) {
                        const auto &seg = segments[s];
                        const int end = seg.offset + seg.seqlen;
                        Split(data, 1, end - std::min(seg.seqlen, window_size), end, parts[s]);
                    }
                    Data tmp[2];
                    V41Assign(data, *catSegments(parts, tmp), tpDevices);
                };
                trimBatch(*curHidden);
                trimBatch(*preMixPtr);
                // Other intermediates were released after their last use. Drop
                // the remaining full-chunk scratch before switching to tail rows.
                ReleaseForwardPrefill(f, {&preMix, &attnPre, &expertIndex, &expertScore,
                                &ws->w1, &ws->w2, &ws->w3, &ws->tempInput, &ws->tempOutput,
                                &ws->moeInputTemp, &ws->moeOutputTemp,
                                &cpuMoeInput, &cpuMoeIndex, &cpuMoeScore});
                seqlen = 0;
                for (int s = 0; s < numSegments; s++) {
                    auto &seg = tailSegments[s];
                    const int skip = std::max(0, seg.seqlen - window_size);
                    if (skip > 0) {
                        auto trimRows = [&](Data &data) {
                            if (data.dims.size() >= 3 && data.dims[1] == seg.seqlen) {
                                Data tmp;
                                Split(data, 1, skip, seg.seqlen, tmp);
                                V41Assign(data, tmp, tpDevices);
                            }
                        };
                        trimRows(segTopK[s]);
                        trimRows(segCandidate[s]);
                        if (seg.spec != nullptr && seg.spec->captureMain) {
                            for (Data &hidden : seg.spec->mainHidden) {
                                trimRows(hidden); // 目标层也可能位于裁剪边界之前
                            }
                            seg.spec->mainHiddenStartPos = seg.startPos + skip;
                        }
                        seg.startPos += skip;
                        seg.seqlen -= skip;
                    }
                    seg.offset = seqlen;
                    seqlen += seg.seqlen;
                }
            }

            // ---- Engram ----
            for (size_t l = 0; l < engram_layer_ids.size(); l++) {
                if (engram_layer_ids[l] == layer) {
                    // Engram gather and NUMA MoE share the global CPU worker pool.
                    std::unique_lock<std::mutex> hostLock;
                    if (f.prefill) hostLock = std::unique_lock<std::mutex>(*f.prefill->hostExperts);
                    RunEngram(layer, (int)l, segments, *curHidden);
                }
            }

            // ---- DSpark：目标层的 main hidden（attention 的输入，对 hc 份取均值）----
            if (v41DsparkEnabled && (int)v41IsDsparkTarget.size() == block_cnt && v41IsDsparkTarget[layer]) {
                int slot = 0;
                for (size_t k = 0; k < v41DsparkTargetLayerIds.size(); k++) {
                    if (v41DsparkTargetLayerIds[k] == layer) {
                        slot = (int)k;
                        break;
                    }
                }
                for (int s = 0; s < numSegments; s++) {
                    DeepSeekV41SpecScratch *spec = segments[s].spec;
                    if (spec == nullptr || !spec->captureMain) {
                        continue;
                    }
                    if (spec->mainHidden.size() != v41DsparkTargetLayerIds.size()) {
                        spec->mainHidden.resize(v41DsparkTargetLayerIds.size());
                    }
                    Data segHidden;
                    Data *src = sliceOf(*curHidden, s, segHidden);
                    // TP decode can leave the kernel queued on a worker stream.
                    // Retain its constant coefficients until the forward drains.
                    Data localUniform;
                    Data &uniform = single ? ws->dsparkPreMix : localUniform;
                    const std::vector<int> uniformDims = {1, segments[s].seqlen, hc_mult};
                    if (uniform.dims != uniformDims) {
                        std::vector<float> uniformValues((uint64_t)segments[s].seqlen * hc_mult,
                                                         1.0f / (float)hc_mult);
                        uniform.CopyFrom(Data(DataType::FLOAT32, uniformDims, uniformValues));
                    }
                    V41HcApplyPre(*src, uniform, spec->mainHidden[slot]);
                }
            }

            // ---- attention（整批部分）----
            // 【CUDA Graph 的 pre 段】这里到 RoPE 之前全部与 token 位置无关，整段可捕获。
            bool needIndexer = ratio > 0 && isIndexSource[layer];
            auto runAttentionPre = [&]() {
            // Both projections read the same fake-FP8 values. The workspace
            // owns these buffers across asynchronous TP dispatch and graph replay.
            const bool reuseQuantized = !V41EnvFlag("FASTLLM_DSV41_REFERENCE_MATH");
            Data *attnQuant = reuseQuantized &&
                quantizedLinearNames.count(pre + ".attn.wq_a.weight") &&
                quantizedLinearNames.count(pre + ".attn.wkv.weight") ? &ws->attnInputQuantized : nullptr;
            Data *qQuant = reuseQuantized && needIndexer &&
                quantizedLinearNames.count(pre + ".attn.wq_b.weight") &&
                quantizedLinearNames.count(pre + ".attn.indexer.wq_b.weight") ? &ws->qNormQuantized : nullptr;
            V41HcMixPreNorm(*curHidden, *preMixPtr, weight[pre + ".hc_attn_fn"],
                weight[pre + ".hc_attn_scale"], weight[pre + ".hc_attn_base"],
                weight[pre + ".attn_norm.weight"], hc_mult, hc_sinkhorn_iters, hc_eps, rms_norm_eps,
                attnPre, attnPost, attnComb, x, attnInput);

            // wq_a / wkv 是复制的（KV 是 MLA 式的单份 latent，与 head 无关）；
            // wq_b 按行切 -> q 的 head 维分片，Reshape 会把 tpAxis 从最后一维换算到 head 维。
            quantizedLinear(attnInput, pre + ".attn.wq_a.weight", qr, tp, attnQuant);
            V41RMSNormBF16(qr, weight[pre + ".attn.q_norm.weight"], rms_norm_eps, qNorm);
            if (tpAttention) {
                weight[pre + ".attn.wq_b.weight"].tpLinearType = TP_LINEAR_ROW;
            }
            // 不切分注意力时 wq_b 也必须显式走复制布局：否则 MultiCudaLinearOp 会按
            // "大权重通用切分 + gather"处理，而那条路径读的是复制张量已失效的 root。
            quantizedLinear(qNorm, pre + ".attn.wq_b.weight", q, tp && !tpAttention, qQuant);
            if (dumpDebug) {
                const std::string tag = "fl_layer" + std::to_string(layer);
                V41DumpTensor(qr, tag + "_qr" + dumpSuffix);
                V41DumpTensor(qNorm, tag + "_qnorm" + dumpSuffix);
                V41DumpTensor(q, tag + "_qproj" + dumpSuffix);
                V41DumpTensor(attnPre, tag + "_attn_pre" + dumpSuffix);
                V41DumpTensor(attnPost, tag + "_attn_post" + dumpSuffix);
                V41DumpTensor(attnComb, tag + "_attn_comb" + dumpSuffix);
            }
            q.Reshape({1, seqlen, num_attention_heads, headDim});

            quantizedLinear(attnQuant ? *attnQuant : attnInput, pre + ".attn.wkv.weight", kv,
                            tp, nullptr, attnQuant != nullptr);
            if (dumpDebug) V41DumpTensor(kv, "fl_layer" + std::to_string(layer) + "_kv_proj" + dumpSuffix);
            V41RMSNormBF16(kv, weight[pre + ".attn.kv_norm.weight"], rms_norm_eps, kv);
            if (dumpDebug) V41DumpTensor(kv, "fl_layer" + std::to_string(layer) + "_kv_norm" + dumpSuffix);
            kv.Reshape({1, seqlen, headDim});

            // kv source 层：compressor 的投影按整批算，分组 / 追加按片段做
            // （rawKVAll / rawScoreAll / qIdxAll / idxWeightsAll 见上面的工作区绑定）
            if (ratio > 0 && isKvSource[layer]) {
                std::string cpre = pre + ".attn.compressor";
                Data &xFloat = ws->attnInputFloat;
                ToDataType(attnInput, xFloat, DataType::FLOAT32);
                quantizedLinear(xFloat, cpre + ".wkv.weight", rawKVAll, tp);
                ToDataType(rawKVAll, DataType::FLOAT32);
                if (ratio > 1) {
                    quantizedLinear(xFloat, cpre + ".wgate.weight", rawScoreAll, tp);
                    ToDataType(rawScoreAll, DataType::FLOAT32);
                }
            }
            if (ratio > 0 && isIndexSource[layer]) {
                std::string ipre = pre + ".attn.indexer";
                // indexer 在每张卡上各算一份：它选出的候选块要供后续所有层复用，
                // 切 index head 就得对 [token, m] 的分数矩阵做 all-reduce，
                // 通信量远大于重复计算，而且两卡 top-k 必须逐位一致。
                quantizedLinear(qQuant ? *qQuant : qNorm, ipre + ".wq_b.weight", qIdxAll,
                                tp, nullptr, qQuant != nullptr);
                qIdxAll.Reshape({1, seqlen, index_n_heads, index_head_dim});
                Data &idxWeights = ws->idxWeights;
                quantizedLinear(attnInput, ipre + ".weights_proj.weight", idxWeights, tp);
                ToDataType(idxWeights, DataType::FLOAT32);
                Mul(idxWeights, (1.0f / std::sqrt((float)index_head_dim)) * (1.0f / std::sqrt((float)index_n_heads)),
                    idxWeightsAll);
            }
            };   // runAttentionPre
            RunForwardGraphSegment(f, runAttentionPre, kV41GraphSegmentsPerLayer * layer, true,
                [&](DeepSeekV41GraphSegmentMeta &meta) {
                    meta.qDims = q.dims;
                    meta.kvDims = kv.dims;
                    meta.qIdxDims = qIdxAll.dims;
                },
                [&](const DeepSeekV41GraphSegmentMeta &meta) {
                    if (q.dims != meta.qDims) {
                        q.Reshape(meta.qDims);
                    }
                    if (kv.dims != meta.kvDims) {
                        kv.Reshape(meta.kvDims);
                    }
                    if (!meta.qIdxDims.empty() && qIdxAll.dims != meta.qIdxDims) {
                        qIdxAll.Reshape(meta.qIdxDims);
                    }
                });

            ReleaseForwardPrefill(f, {&x, &qr, &qNorm, &ws->attnInputFloat, &ws->idxWeights});
            // Batch independent decode requests without changing their QK /
            // softmax / BF16-probability arithmetic or per-request KV positions.
            bool batchAttention = false;
            bool batchTransforms = false, mergedAttention = false;
#ifdef USE_CUDA
            batchAttention = tp && !dumpDebug &&
                std::all_of(segments.begin(), segments.end(), [](const DeepSeekV41Segment &seg) {
                    return seg.seqlen == 1 && seg.startPos > 0;
                });
            std::vector<FastllmCudaV41AttentionRequest> attentionRequests;
            std::vector<V41DecodeIndexTask> indexTasks;
            auto &segIndexQ = f.ws->segIndexQ;
            auto &segIndexWeight = f.ws->segIndexWeight;
            if (batchAttention) {
                std::vector<V41DecodeTransform> transforms;
                transforms.reserve(2 * numSegments);
                // A fully captured ordinary decode has fixed boundary pointers
                // and uniform Q/KV shapes across layers. Validate/build its row
                // views once per forward; only their contents change per layer.
                // Eager, capture, invalidated and speculative paths still check
                // every row. These flags never survive a forward/request batch.
                const bool reuseRows = f.graphReplay && f.workerOrderedGraphReplay &&
                    f.decodeQkvViewsPrepared;
                for (int s = 0; s < numSegments; ++s) {
                    if (reuseRows) {
                        // The same borrowed rows were checked in an earlier layer.
                    } else if (single) {
                        AssertInFastLLM(V41DecodeSliceView(segQ[s], q, segments[s].offset, tpDevices) &&
                                        V41DecodeSliceView(segKV[s], kv, segments[s].offset, tpDevices),
                                        "DeepSeekV41 decode row view failed.");
                    } else {
                        sliceOf(q, s, segQ[s]);
                        sliceOf(kv, s, segKV[s]);
                    }
                    transforms.push_back({&segQ[s], nullptr, segments[s].startPos, 0, false});
                    transforms.push_back({&segKV[s], nullptr, segments[s].startPos, 1, false});
                }
                f.decodeQkvViewsPrepared = true;
                // These producers and consumers all use the dedicated rank
                // stream during ordinary replay. Preserve the general handoff
                // on source layers, where cache growth may use the caller stream.
                batchTransforms = V41DecodeTransforms(transforms, rope, window_size, tpDevices,
                    f.workerOrderedGraphReplay && !isKvSource[layer]);
                // Independent rows can be written directly into the persistent
                // graph boundary, avoiding an O(batch^2) chain of concatenations.
                mergedAttention = MultiCudaCopyShardedActivation(attnOut, q, tpDevices, false);
                if (mergedAttention && !(f.graphReplay && f.workerOrderedGraphReplay &&
                                         f.decodeOutputViewsPrepared)) {
                    for (int s = 0; s < numSegments; ++s) {
                        AssertInFastLLM(V41DecodeSliceView(segAttnOut[s], attnOut, segments[s].offset, tpDevices),
                                        "DeepSeekV41 attention output view failed.");
                    }
                    f.decodeOutputViewsPrepared = true;
                }
            }
#endif
            bool batchedCompression = false;
#ifdef USE_CUDA
            if (batchAttention && !quantKV && ratio > 0 && isKvSource[layer]) {
                batchedCompression = V41CompressDecodeBatch(segments, layer, ratio, rawKVAll, rawScoreAll,
                    weight[pre + ".attn.compressor.norm.weight"], rms_norm_eps, rope, isIndexSource[layer],
                    [&](Data &latent, Data &keys) {
                        const std::string ipre = pre + ".attn.indexer";
                        quantizedLinear(latent, ipre + ".wk.weight", keys, tp);
                        V41RMSNormBF16(keys, weight[ipre + ".k_norm.weight"], rms_norm_eps, keys);
                    }, tpDevices, *ws);
            }
#endif
            auto finishAttention = [&](int s, Data &kvRows, Data &attentionOutput) {
                const auto &seg = segments[s];
                auto &cache = (f.prefill ? f.prefill->request->layers : seg.state->layers)[layer];
                const int startPos = seg.startPos, segLen = seg.seqlen;
                Data *kvSeg = &kvRows, *attnOutSeg = &attentionOutput;
                Data kvSeg8;
                const Data *windowRows = kvSeg;
                if (quantKV) {
                    // 滑窗 KV 在两档下都用 FP8：它本来就在 FP8 网格上，改 FP4 会真的损失精度
                    V41QuantizeKV(*kvSeg, kvSeg8, 1, 32);
                    windowRows = &kvSeg8;
                }
                if (seg.spec != nullptr && seg.spec->deferWindow) {
                    // DSpark 校验：环形缓冲的写入推迟到接受长度确定之后。片段内的位置
                    // 一律从 chunkKV 读取，因此推迟写入不改变本次前向的任何结果，
                    // 也就不需要为回滚保存被覆盖的旧行。
                    V41Assign(seg.spec->windowKV[layer], *windowRows, tpDevices);
                } else {
                    V41WindowStore(*windowRows, cache.windowKV, startPos, window_size);
                }
                // V4.1 quantizes the inverse-RoPE attention output in blocks
                // of 32 before the grouped wo_a projection on every SM.
                V41RotaryQuant(*attnOutSeg, rope, startPos, 1, true, 1, 32);
                cache.totalLen = startPos + segLen;
            };
            // ---- attention（按片段准备，decode 合并提交）----
            for (int s = 0; s < numSegments; s++) {
                const DeepSeekV41Segment &seg = segments[s];
                DeepSeekV41LayerCache &cache = (f.prefill ? f.prefill->request->layers : seg.state->layers)[layer];
                const int startPos = seg.startPos;
                const int segLen = seg.seqlen;

                Data *qSeg = batchAttention ? &segQ[s] : sliceOf(q, s, segQ[s]);
                if (!batchTransforms) V41RotaryQuant(*qSeg, rope, startPos, 1, false, 0, 32);
                Data *kvSeg = batchAttention ? &segKV[s] : sliceOf(kv, s, segKV[s]);
                if (!batchTransforms) V41RotaryQuant(*kvSeg, rope, startPos, 1, false, 1, 32);

                Data *compressedKV = nullptr;
                Data *cmpIdx = nullptr;
                if (ratio > 0) {
                    const int src = kvSourceOf[layer];
                    DeepSeekV41LayerCache &srcCache = (f.prefill ? f.prefill->request->layers : seg.state->layers)[src];
                    if (isKvSource[layer] && !batchedCompression) {
                        std::string cpre = pre + ".attn.compressor";
                        Data rawKVSeg, rawScoreSeg, allKV, allScore;
                        Data *rawKV = sliceOf(rawKVAll, s, rawKVSeg);
                        Data *rawScore = ratio > 1 ? sliceOf(rawScoreAll, s, rawScoreSeg) : nullptr;
                        Data *kvAll = rawKV, *scoreAll = rawScore;
                        const int specPrevTail = cache.rawTail;
                        const int specPrevBlocks = cache.compressedBlocks;
                        if (cache.rawTail > 0) {
                            Cat(cache.rawTailKV, *rawKV, 1, allKV);
                            kvAll = &allKV;
                            if (ratio > 1) {
                                Cat(cache.rawTailScore, *rawScore, 1, allScore);
                                scoreAll = &allScore;
                            }
                        }
                        // DSpark 校验：保存压缩器的原始输入流与回滚点，接受长度确定后重建
                        if (seg.spec != nullptr && seg.spec->deferWindow) {
                            seg.spec->prevRawTail[layer] = specPrevTail;
                            seg.spec->prevBlocks[layer] = specPrevBlocks;
                            V41Assign(seg.spec->rawKV[layer], *kvAll, tpDevices);
                            if (ratio > 1) {
                                V41Assign(seg.spec->rawScore[layer], *scoreAll, tpDevices);
                            }
                        }
                        const int n = kvAll->dims[1];
                        const int full = n - n % ratio;
                        const int blocks = full / ratio;
                        if (blocks > 0) {
                            Data kvPart, scorePart, latent;
                            Data *kvPartPtr = kvAll, *scorePartPtr = ratio > 1 ? scoreAll : nullptr;
                            if (full != n) {
                                Split(*kvAll, 1, 0, full, kvPart);
                                kvPartPtr = &kvPart;
                                if (ratio > 1) {
                                    Split(*scoreAll, 1, 0, full, scorePart);
                                    scorePartPtr = &scorePart;
                                }
                            }
                            V41Compress(*kvPartPtr, scorePartPtr, weight[cpre + ".norm.weight"], ratio, rms_norm_eps, latent);
                            const int blockStart = cache.compressedBlocks;
                            // indexer key 由 pre-RoPE latent 派生
                            if (isIndexSource[layer]) {
                                std::string ipre = pre + ".attn.indexer";
                                Data kIdx;
                                quantizedLinear(latent, ipre + ".wk.weight", kIdx, tp);
                                V41RMSNormBF16(kIdx, weight[ipre + ".k_norm.weight"], rms_norm_eps, kIdx);
                                kIdx.Reshape({1, blocks, index_head_dim});
                                V41RotaryQuant(kIdx, rope, blockStart * ratio, ratio, false, 2, 32);
                                if (quantKV) {
                                    Data kIdxQ;
                                    V41QuantizeKV(kIdx, kIdxQ, idxQuantMode, idxQuantBlock);
                                    V41AppendRows(cache.indexK, kIdxQ, tpDevices);
                                } else {
                                    V41AppendRows(cache.indexK, kIdx, tpDevices);
                                }
                            }
                            V41RotaryQuant(latent, rope, blockStart * ratio, ratio, false, 3, 16);
                            if (quantKV) {
                                Data latentQ;
                                V41QuantizeKV(latent, latentQ, cmpQuantMode, cmpQuantBlock);
                                V41AppendRows(cache.compressedKV, latentQ, tpDevices);
                            } else {
                                V41AppendRows(cache.compressedKV, latent, tpDevices);
                            }
                            cache.compressedBlocks += blocks;
                        }
                        const int rem = n - full;
                        if (rem > 0) {
                            Data tailKV, tailScore;
                            Split(*kvAll, 1, full, n, tailKV);
                            V41Assign(cache.rawTailKV, tailKV, tpDevices);
                            if (ratio > 1) {
                                Split(*scoreAll, 1, full, n, tailScore);
                                V41Assign(cache.rawTailScore, tailScore, tpDevices);
                            }
                        }
                        cache.rawTail = rem;
                    }
                    AssertInFastLLM(srcCache.compressedBlocks == (startPos + segLen) / ratio,
                                    "DeepSeekV41: compressed cache is out of sync at layer " + std::to_string(layer));
                    if (srcCache.compressedBlocks > 0) {
                        compressedKV = &srcCache.compressedKV;
                        if (needIndexer) {
#ifdef USE_CUDA
                            if (batchAttention && !quantKV) {
                                const bool writeCandidates = layer == candidate_source_layer_id &&
                                    candidate_block_size > 0 && candidate_topk_blocks > 0;
                                const bool useCandidates = segHasCandidates[s] && candidate_source_layer_id >= 0 &&
                                    candidate_source_layer_id < layer;
                                indexTasks.push_back({sliceOf(qIdxAll, s, segIndexQ[s]),
                                    sliceOf(idxWeightsAll, s, segIndexWeight[s]), &srcCache.indexK,
                                    &segCandidate[s], &segTopK[s], startPos, writeCandidates, useCandidates});
                                if (writeCandidates) segHasCandidates[s] = 1;
                                cmpIdx = &segTopK[s];
                            } else
#endif
                            {
                            Data qIdxSeg, idxWeightsSeg;
                            Data *qIdx = sliceOf(qIdxAll, s, qIdxSeg);
                            V41RotaryQuant(*qIdx, rope, startPos, 1, false, 2, 32);
                            Data *idxWeightsScaled = sliceOf(idxWeightsAll, s, idxWeightsSeg);
                            const bool isCandidateLayer = layer == candidate_source_layer_id &&
                                                          candidate_block_size > 0 && candidate_topk_blocks > 0;
                            const bool useCandidates = segHasCandidates[s] && candidate_source_layer_id >= 0 &&
                                                       candidate_source_layer_id < layer;
                            // 分数矩阵是 [token, m]，长上下文下会非常大（1M 上下文时每个 token 512 K 个候选）。
                            // 按 token 维分块调用打分 / 候选块 / top-k，把峰值显存钉在一个固定预算内。
                            const int scoreM = srcCache.indexK.dims.size() == 3 ? srcCache.indexK.dims[1] : 0;
                            int chunkTokens = V41IndexScoreChunk(segLen, scoreM);
                            bool partitionedIndexer = false;
#ifdef USE_CUDA
                            partitionedIndexer = !dumpDebug && V41PrefillIndexerTp(
                                *qIdx, *idxWeightsScaled, srcCache.indexK, segCandidate[s], segTopK[s],
                                isCandidateLayer, useCandidates, ratio, startPos, indexTopK,
                                candidate_block_size, candidate_topk_blocks, tpDevices);
#endif
                            if (partitionedIndexer) {
                                if (isCandidateLayer) segHasCandidates[s] = 1;
                            } else if (chunkTokens >= segLen) {
                                Data score;
                                V41IndexerScore(*qIdx, *idxWeightsScaled, srcCache.indexK, ratio, startPos, score);
                                if (dumpDebug) {
                                    V41DumpTensor(*qIdx, "fl_layer" + std::to_string(layer) + "_idxq" + dumpSuffix);
                                    V41DumpTensor(*idxWeightsScaled, "fl_layer" + std::to_string(layer) + "_idxw" + dumpSuffix);
                                    V41DumpTensor(score, "fl_layer" + std::to_string(layer) + "_score" + dumpSuffix);
                                }
                                if (isCandidateLayer) {
                                    V41CandidateBlocks(score, candidate_block_size, candidate_topk_blocks, ratio,
                                                       startPos, segCandidate[s]);
                                    segHasCandidates[s] = 1;
                                    if (dumpDebug) {
                                        V41DumpTensor(segCandidate[s], "fl_layer" + std::to_string(layer) + "_cand" + dumpSuffix);
                                    }
                                }
                                V41IndexerTopK(score, useCandidates ? &segCandidate[s] : nullptr, indexTopK, ratio,
                                               startPos, std::max(1, candidate_block_size), segTopK[s]);
                            } else {
                                const int numChunks = (segLen + chunkTokens - 1) / chunkTokens;
                                // 整段输出先按第一块的宽度分配好，各块直接写进对应行区间（避免逐块 Cat）
                                Data candAll, topkAll;
                                Data prevCandidate;
                                if (useCandidates) {
                                    V41Assign(prevCandidate, segCandidate[s], tpDevices);
                                }
                                for (int c = 0; c < numChunks; c++) {
                                    const int c0 = c * chunkTokens;
                                    const int c1 = std::min(segLen, c0 + chunkTokens);
                                    Data qChunk, wChunk, score, candChunk, topkChunk;
                                    Split(*qIdx, 1, c0, c1, qChunk);
                                    Split(*idxWeightsScaled, 1, c0, c1, wChunk);
                                    V41IndexerScore(qChunk, wChunk, srcCache.indexK, ratio, startPos + c0, score);
                                    Data *candPtr = nullptr;
                                    if (isCandidateLayer) {
                                        V41CandidateBlocks(score, candidate_block_size, candidate_topk_blocks, ratio,
                                                           startPos + c0, candChunk);
                                        candPtr = &candChunk;
                                    } else if (useCandidates) {
                                        Split(prevCandidate, 1, c0, c1, candChunk);
                                        candPtr = &candChunk;
                                    }
                                    V41IndexerTopK(score, candPtr, indexTopK, ratio, startPos + c0,
                                                   std::max(1, candidate_block_size), topkChunk);
                                    if (c == 0) {
                                        if (isCandidateLayer) {
                                            V41AllocLike(candAll, candChunk.dataType,
                                                         {candChunk.dims[0], segLen, candChunk.dims[2]},
                                                         candChunk, tpDevices);
                                        }
                                        V41AllocLike(topkAll, topkChunk.dataType,
                                                     {topkChunk.dims[0], segLen, topkChunk.dims[2]},
                                                     topkChunk, tpDevices);
                                    }
                                    if (isCandidateLayer) {
                                        V41WriteRows(candAll, candChunk, c0, tpDevices);
                                    }
                                    V41WriteRows(topkAll, topkChunk, c0, tpDevices);
                                }
                                if (isCandidateLayer) {
                                    V41Assign(segCandidate[s], candAll, tpDevices);
                                    segHasCandidates[s] = 1;
                                }
                                V41Assign(segTopK[s], topkAll, tpDevices);
                            }
                            }
                        }
                        if (needIndexer) segHasTopK[s] = 1;
                        if (segHasTopK[s] && segTopK[s].dims.size() == 3) {
                            cmpIdx = &segTopK[s];
                        }
                    }
                }

                Data *attnOutSeg = single ? &attnOut : &segAttnOut[s];
                // 裁剪后边界之前的局部 KV 未计算，不能读取上一 chunk 的旧环形槽。
                // attention 的 startPos 只用于局部窗口寻址；RoPE、indexer 和压缩 KV
                // 的因果筛选已经按绝对位置完成。以新窗口起点 0 执行局部 attention。
                const int windowStart = seg.seqlen < inputSegments[s].seqlen ? 0 : startPos;
#ifdef USE_CUDA
                if (batchAttention) {
                    attentionRequests.push_back({qSeg, kvSeg, windowStart > 0 ? &cache.windowKV : nullptr,
                        compressedKV, cmpIdx, attnOutSeg, windowStart});
                } else
#endif
                {
                V41SparseAttention(*qSeg, *kvSeg, windowStart > 0 ? &cache.windowKV : nullptr, compressedKV, cmpIdx,
                                   weight[pre + ".attn.attn_sink"], window_size, windowStart, softmaxScale, *attnOutSeg);
                }
                if (dumpDebug) {
                    std::string tag = "fl_layer" + std::to_string(layer);
                    V41DumpTensor(*qSeg, tag + "_q" + dumpSuffix);
                    V41DumpTensor(*kvSeg, tag + "_kv" + dumpSuffix);
                    V41DumpTensor(*attnOutSeg, tag + "_attn_o_raw" + dumpSuffix);
                    if (compressedKV != nullptr) {
                        V41DumpTensor(*compressedKV, tag + "_ckv" + dumpSuffix);
                    }
                    if (ratio > 0 && isKvSource[layer]) {
                        V41DumpTensor(cache.indexK, tag + "_idxk" + dumpSuffix);
                    }
                    if (startPos > 0) {
                        V41DumpTensor(cache.windowKV, tag + "_ring" + dumpSuffix);
                    }
                }
                if (!batchAttention) finishAttention(s, *kvSeg, *attnOutSeg);
            }

#ifdef USE_CUDA
            if (batchAttention) {
                // Index selection, attention, and ring update have device-local
                // dependencies. Submit them together instead of handing the
                // same stream back to the caller between all three stages.
                bool completedTogether = false;
                std::vector<V41DecodeTransform> finishTransforms;
                std::function<void(int, int)> beforeAttention, afterAttention;
                if (!quantKV && std::all_of(segments.begin(), segments.end(), [](const DeepSeekV41Segment &seg) {
                        return !seg.spec || !seg.spec->deferWindow;
                    })) {
                    for (int s = 0; s < numSegments; ++s) {
                        finishTransforms.push_back({&segKV[s], &(f.prefill ? f.prefill->request->layers : segments[s].state->layers)[layer].windowKV,
                                                    segments[s].startPos, 0, false});
                        finishTransforms.push_back({&segAttnOut[s], nullptr, segments[s].startPos, 1, true});
                    }
                    if (V41PrepareDecodeTransforms(finishTransforms, rope, window_size, tpDevices, afterAttention) &&
                        (indexTasks.empty() || V41PrepareDecodeIndexer(indexTasks, rope, ratio, indexTopK,
                            std::max(1, candidate_block_size), candidate_topk_blocks, tpDevices, beforeAttention))) {
                        completedTogether = MultiCudaDeepSeekV41SparseAttentionBatch(attentionRequests,
                            weight[pre + ".attn.attn_sink"], window_size, softmaxScale, tpDevices,
                            beforeAttention, afterAttention,
                            f.workerOrderedGraphReplay && !isKvSource[layer]);
                    }
                }
                if (!completedTogether) {
                    if (!indexTasks.empty() && !V41DecodeIndexerBatch(indexTasks, rope, ratio, indexTopK,
                        std::max(1, candidate_block_size), candidate_topk_blocks, tpDevices)) {
                        for (const auto &r : indexTasks) {
                            V41RotaryQuant(*r.query, rope, r.startPos, 1, false, 2, 32);
                            Data score;
                            V41IndexerScore(*r.query, *r.weights, *r.keys, ratio, r.startPos, score);
                            if (r.writeCandidates)
                                V41CandidateBlocks(score, candidate_block_size, candidate_topk_blocks,
                                                   ratio, r.startPos, *r.candidates);
                            V41IndexerTopK(score, r.useCandidates ? r.candidates : nullptr, indexTopK,
                                ratio, r.startPos, std::max(1, candidate_block_size), *r.topk);
                        }
                    }
                    if (!MultiCudaDeepSeekV41SparseAttentionBatch(attentionRequests,
                        weight[pre + ".attn.attn_sink"], window_size, softmaxScale, tpDevices)) {
                        for (const auto &r : attentionRequests) {
                            V41SparseAttention(*r.q, *r.chunkKV, r.ringKV, r.compressedKV, r.indices,
                                weight[pre + ".attn.attn_sink"], window_size, r.startPos, softmaxScale, *r.output);
                        }
                    }
                    // Do not overwrite a ring row until attention has read it.
                    bool finished = false;
                    if (!quantKV && std::all_of(segments.begin(), segments.end(), [](const DeepSeekV41Segment &seg) {
                            return !seg.spec || !seg.spec->deferWindow;
                        })) {
                        std::vector<V41DecodeTransform> transforms;
                        for (int s = 0; s < numSegments; ++s) {
                            transforms.push_back({&segKV[s], &(f.prefill ? f.prefill->request->layers : segments[s].state->layers)[layer].windowKV,
                                                  segments[s].startPos, 0, false});
                            transforms.push_back({&segAttnOut[s], nullptr, segments[s].startPos, 1, true});
                        }
                        finished = V41DecodeTransforms(transforms, rope, window_size, tpDevices);
                    }
                    if (!finished) for (int s = 0; s < numSegments; ++s)
                        finishAttention(s, segKV[s], segAttnOut[s]);
                }
                for (int s = 0; s < numSegments; ++s) {
                    (f.prefill ? f.prefill->request->layers : segments[s].state->layers)[layer].totalLen = segments[s].startPos + 1;
                }
            }
#endif
            ReleaseForwardPrefill(f, {&q, &kv, &attnInput, &rawKVAll, &rawScoreAll, &qIdxAll, &idxWeightsAll});
            for (int s = 0; s < numSegments; s++) ReleaseForwardPrefill(f, {&segQ[s], &segKV[s]});
            Data *attnOutAll = (single || mergedAttention) ? &attnOut : catSegments(segAttnOut, catTmp);
            if (f.graphActive && !single && !mergedAttention) {
                // The concatenation temporaries belong to this forward only.
                // A graph must read the persistent boundary buffer instead.
#ifdef USE_CUDA
                if (tp && attnOutAll->IsTensorParallelSharded()) {
                    AssertInFastLLM(MultiCudaCopyShardedActivation(attnOut, *attnOutAll, tpDevices),
                                    "V4.1 graph attention boundary copy failed.");
                } else
#endif
                {
                    V41Assign(attnOut, *attnOutAll, tpDevices);
                }
                attnOutAll = &attnOut;
            }
            // 【CUDA Graph 的 post 段】wo_a 到共享专家之间同样与 token 位置无关。
            // MergeMOEBlock（可能在 CPU / NUMA 上）与其后的 hc 残差留在段外。
            moeWeights = weights[layer];
            hasSharedExpertOut = false;
            auto sharedGateupIt = weight.weight.find(pre + ".ffn.shared_experts.gateup.weight");
            auto sharedDownIt = weight.weight.find(pre + ".ffn.shared_experts.w2.weight");
            // Keep the shared expert on CUDA in TP as well. Mixing its FP16
            // weights with the routed NVFP4 experts disables the NUMA fast path.
            if (GetCudaSharedExpert() && !V41ReferenceMathEnabled() &&
                sharedGateupIt != weight.weight.end() &&
                sharedDownIt != weight.weight.end() && !sharedGateupIt->second.isDiskWeight &&
                !sharedDownIt->second.isDiskWeight) {
                moeWeights[0] = moeWeights[1] = nullptr;
                hasSharedExpertOut = true;
            }
            ffnDims.clear();
            auto runAttentionPost = [&]() {
            if (tpAttention) {
                // wo_a 按 head 组切（与 q 的分片一致），wo_b 按列切；
                // wo_b 的输入是分片的，MultiCudaLinearOp 会自动走 column + all-reduce。
                weight[pre + ".attn.wo_a.weight"].tpLinearType = TP_LINEAR_ROW;
                weight[pre + ".attn.wo_b.weight"].tpLinearType = TP_LINEAR_COLUMN;
            }
            bool referenceWoA = false;
#ifdef USE_CUDA
            if (V41EnvFlag("FASTLLM_DSV41_REFERENCE_MATH") && !attnOutAll->multiDeviceData) {
                ToDataType(weight[pre+".attn.wo_a.weight"], DataType::BFLOAT16);
                auto originalDims = attnOutAll->dims;
                attnOutAll->Reshape({1,seqlen,num_attention_heads*headDim});
                referenceWoA = FastllmCudaV41ReferenceLinear(*attnOutAll,weight[pre+".attn.wo_a.weight"],woAOut,o_groups,true);
                attnOutAll->Reshape(originalDims);
            }
#endif
            if (!referenceWoA) DeepSeekV4WoA(*attnOutAll, weight[pre + ".attn.wo_a.weight"], o_groups, o_lora_rank, woAOut);
            ReleaseForwardPrefill(f, {&attnOut, &catTmp[0], &catTmp[1]});
            for (Data &part : segAttnOut) ReleaseForwardPrefill(f, {&part});
            // 切分时 woAOut 是分片的，MultiCudaLinearOp 自动走 column + all-reduce；
            // 不切分时 woAOut 是复制的，必须显式要求复制布局，否则会退回单卡 CUDA
            // 读到已经失效的 root。
            quantizedLinear(woAOut, pre + ".attn.wo_b.weight", attnProj, tp && !tpAttention);
            if (dumpDebug) {
                V41DumpTensor(woAOut, "fl_layer" + std::to_string(layer) + "_woa" + dumpSuffix);
                V41DumpTensor(attnInput, "fl_layer" + std::to_string(layer) + "_attn_in" + dumpSuffix);
                V41DumpTensor(*attnOutAll, "fl_layer" + std::to_string(layer) + "_attn_o" + dumpSuffix);
                V41DumpTensor(attnProj, "fl_layer" + std::to_string(layer) + "_attn" + dumpSuffix);
                if (segTopK[0].dims.size() == 3) {
                    V41DumpTensor(segTopK[0], "fl_layer" + std::to_string(layer) + "_topk" + dumpSuffix);
                }
            }
            V41HcPost(attnProj, *curHidden, attnPost, attnComb, *nextHidden);
            std::swap(curHidden, nextHidden);
            ReleaseForwardPrefill(f, {&woAOut, &attnProj, &attnPost, &attnComb});
            if (dumpDebug) {
                V41DumpTensor(*curHidden, "fl_layer" + std::to_string(layer) + "_hidden_attn" + dumpSuffix);
            }

            // ---- FFN (MoE)：整批 ----
            V41HcMixPreNorm(*curHidden, attnPre, weight[pre + ".hc_ffn_fn"],
                weight[pre + ".hc_ffn_scale"], weight[pre + ".hc_ffn_base"],
                weight[pre + ".ffn_norm.weight"], hc_mult, hc_sinkhorn_iters, hc_eps, rms_norm_eps,
                ffnPre, ffnPost, ffnComb, x, ffnInput);
            ffnDims = ffnInput.dims;
            ffnInput.Reshape({seqlen, dim});
            };   // runAttentionPost
            RunForwardGraphSegment(f, runAttentionPost, kV41GraphSegmentsPerLayer * layer + 1, true,
                [&](DeepSeekV41GraphSegmentMeta &meta) {
                    meta.ffnDims = ffnDims;
                    meta.ffnInputDims = ffnInput.dims;
                    meta.swapHidden = true;
                },
                [&](const DeepSeekV41GraphSegmentMeta &meta) {
                    ffnDims = meta.ffnDims;
                    if (ffnInput.dims != meta.ffnInputDims) {
                        ffnInput.Reshape(meta.ffnInputDims);
                    }
                    if (meta.swapHidden) {
                        std::swap(curHidden, nextHidden);
                    }
                });

            auto runRoute = [&]() {
            // 路由：sqrt(softplus(logits / gate_temp))，bias 只参与选择
            {
                std::string gpre = pre + ".ffn.gate";
                Data &xFloat = ws->gateInput;
                Data &logits = ws->gateLogits;
                bool scoresReady = false;
#ifdef USE_CUDA
                if (!hasImageTokens && !V41ReferenceMathEnabled() &&
                    !V41EnvFlag("FASTLLM_DSV41_DISABLE_CUDA_ROUTE"))
                    scoresReady = V41RouterScores(ffnInput, weight[gpre + ".weight"], gate_temp, logits);
#endif
                if (!scoresReady) {
                    ToDataType(ffnInput, xFloat, DataType::FLOAT32);
                    quantizedLinear(xFloat, gpre + ".weight", logits, tp);
                    ToDataType(logits, DataType::FLOAT32);
                    if (std::fabs(gate_temp - 1.0f) > 1e-6f) {
                        Mul(logits, 1.0f / gate_temp, logits);
                    }
                }
                Data &gateBias = weight[gpre + ".bias"];
                // 图像 token 使用 bias_vl 做专家选择（只影响 prefill，走 CPU 参考路径）
                Data *gateBiasVl = nullptr;
                if (hasImageTokens) {
                    auto vlIt = weight.weight.find(gpre + ".bias_vl");
                    AssertInFastLLM(vlIt != weight.weight.end(),
                                    "DeepSeekV41: " + gpre + ".bias_vl is required for image tokens.");
                    gateBiasVl = &vlIt->second;
                    gateBiasVl->ToDevice(DataDevice::CPU);
                }
                bool routed = false;
#ifdef USE_CUDA
                if (!hasImageTokens && !V41ReferenceMathEnabled() &&
                    (logits.dataDevice == DataDevice::CUDA || (tp && logits.multiDeviceData)) &&
                    !V41EnvFlag("FASTLLM_DSV41_DISABLE_CUDA_ROUTE") &&
                    (scoresReady || V41RouteScoreTransformTp(logits, 2, tpDevices))) {
                    gateBias.ToDevice(DataDevice::CUDA);
                    SelectExpert(logits, expertIndex, expertScore, num_experts_per_tok, true,
                                 routed_scaling_factor, &gateBias);
                    routed = true;
                }
#endif
                if (!routed) {
                    // 图像 token 走 CPU 参考路由。复制布局下 logits 的 root 只有形状信息，
                    // 直接 ToDevice(CPU) 会读到失效指针，必须从卡上副本拷出来。
                    Data cpuLogits;
                    const Data *logitsCpu = &logits;
                    if (tp && logits.multiDeviceData && logits.IsTensorParallelReplicated()) {
                        V41ReplicaToCpu(cpuLogits, logits, tpDevices);
                        logitsCpu = &cpuLogits;
                    } else {
                        logits.ToDevice(DataDevice::CPU);
                    }
                    gateBias.ToDevice(DataDevice::CPU);
                    const float *raw = (const float*)logitsCpu->cpuData;
                    const float *bias = (const float*)gateBias.cpuData;
                    std::vector<int> indices((uint64_t)seqlen * num_experts_per_tok);
                    std::vector<float> scores((uint64_t)seqlen * num_experts_per_tok);
                    std::vector<float> original(num_experts), select(num_experts);
                    for (int t = 0; t < seqlen; t++) {
                        const float *tokenBias = (gateBiasVl != nullptr && (*imageMask)[t] != 0) ?
                                                 (const float*)gateBiasVl->cpuData : bias;
                        for (int e = 0; e < num_experts; e++) {
                            const float value = raw[(uint64_t)t * num_experts + e];
                            original[e] = V41ReferenceMathEnabled()
                                ? V41CPUMath().Sqrt(value > 20.0f ? value : V41CPUMath().Log1p(V41CPUMath().Exp(value)))
                                : std::sqrt(V41Softplus(value));
                            select[e] = original[e] + tokenBias[e];
                        }
                        float sum = 0.0f;
                        for (int k = 0; k < num_experts_per_tok; k++) {
                            int best = 0;
                            for (int e = 1; e < num_experts; e++) {
                                if (select[e] > select[best]) {
                                    best = e;
                                }
                            }
                            indices[(uint64_t)t * num_experts_per_tok + k] = best;
                            scores[(uint64_t)t * num_experts_per_tok + k] = original[best];
                            sum += original[best];
                            select[best] = -std::numeric_limits<float>::infinity();
                        }
                        if (V41ReferenceMathEnabled()) {
                            sum = V41ReferenceSum(num_experts_per_tok, [&](size_t k) {
                                return scores[(uint64_t)t * num_experts_per_tok + k];
                            });
                        }
                        for (int k = 0; k < num_experts_per_tok; k++) {
                            float &v = scores[(uint64_t)t * num_experts_per_tok + k];
                            if (norm_topk_prob && num_experts_per_tok > 1) {
                                v /= (sum + 1e-20f);
                            }
                            v *= routed_scaling_factor;
                        }
                    }
                    Data idxData(DataType::INT32, {seqlen, num_experts_per_tok});
                    idxData.Allocate();
                    memcpy(idxData.cpuData, indices.data(), indices.size() * sizeof(int));
#ifdef USE_CUDA
                    // 之前几层可能把它们做成了复制布局，CopyFrom 只会写 root，
                    // 留下的旧副本会被后面的 multicuda 算子当成有效数据。
                    V41ResetMultiDevice(expertIndex);
                    V41ResetMultiDevice(expertScore);
#endif
                    expertIndex.CopyFrom(idxData);
                    expertScore.CopyFrom(Data(DataType::FLOAT32, {seqlen, num_experts_per_tok}, scores));
                }
            }

            };   // runRoute
            // Image-dependent routing keeps its existing eager fallback.
            RunForwardGraphSegment(f, runRoute, kV41GraphSegmentsPerLayer * layer + 2, cudaRouteCapturable,
                [&](DeepSeekV41GraphSegmentMeta &meta) {
                    meta.expertIndexDims = expertIndex.dims;
                    meta.expertScoreDims = expertScore.dims;
                },
                [&](const DeepSeekV41GraphSegmentMeta &meta) {
                    if (expertIndex.dims != meta.expertIndexDims) {
                        expertIndex.Reshape(meta.expertIndexDims);
                    }
                    if (expertScore.dims != meta.expertScoreDims) {
                        expertScore.Reshape(meta.expertScoreDims);
                    }
                });

    }
    void DeepSeekV41Model::EndForwardLayer(ForwardWorkspace &f, int layer) {
        auto &inputSegments = f.inputSegments;
        auto &boundedPrefill = f.boundedPrefill;
        auto &tailSegments = f.tailSegments;
        auto &tpDevices = f.tpDevices;
        auto &ws = f.ws;
        auto &curHidden = f.curHidden;
        auto &nextHidden = f.nextHidden;
        auto &dumpDebug = f.dumpDebug;
        auto &dumpSuffix = f.dumpSuffix;
        auto &preMixPtr = f.preMixPtr;
        auto &ffnDims = f.ffnDims;
        auto &combineFfnPostDispatch = f.combineFfnPostDispatch;
        const auto &segments = boundedPrefill ? tailSegments : inputSegments;
        auto &ffnPre = ws->ffnPre;
        auto &ffnPost = ws->ffnPost;
        auto &ffnComb = ws->ffnComb;
        auto &ffnInput = ws->ffnInput;
        auto &ffnOut = ws->ffnOut;
        auto &expertIndex = ws->expertIndex;
        auto &expertScore = ws->expertScore;
        auto &sharedExpertOut = ws->sharedExpertOut;

            // HcPost uses residual's shape and input's element count. Keep the
            // CPU MoE output flat so next layer's MergeMOEBlock does not discard
            // its replicas solely because [tokens, dim] became [1, tokens, dim].
            // The joint callback refreshes these buffers before every AddTo.
            const bool retainCpuMoeReplicas = combineFfnPostDispatch &&
                ffnOut.dataDevice == DataDevice::CPU && ffnOut.cpuData != nullptr;
            if (!retainCpuMoeReplicas) ffnOut.Reshape(ffnDims);
            if (dumpDebug) {
                V41DumpTensor(ffnInput, "fl_layer" + std::to_string(layer) + "_ffn_in" + dumpSuffix);
                V41DumpTensor(ffnOut, "fl_layer" + std::to_string(layer) + "_ffn" + dumpSuffix);
                V41DumpTensor(expertIndex, "fl_layer" + std::to_string(layer) + "_expert_idx" + dumpSuffix);
                V41DumpTensor(expertScore, "fl_layer" + std::to_string(layer) + "_expert_score" + dumpSuffix);
                V41DumpTensor(ffnPre, "fl_layer" + std::to_string(layer) + "_ffn_pre" + dumpSuffix);
                V41DumpTensor(ffnPost, "fl_layer" + std::to_string(layer) + "_ffn_post" + dumpSuffix);
                V41DumpTensor(ffnComb, "fl_layer" + std::to_string(layer) + "_ffn_comb" + dumpSuffix);
            }
            bool ffnPostDone = false;
#ifdef USE_CUDA
            if (combineFfnPostDispatch) {
                ffnPostDone = MultiCudaDeepSeekV41AddHcPost(
                    ffnOut, sharedExpertOut, *curHidden, ffnPost, ffnComb, *nextHidden);
                if (!ffnPostDone) {
                    if (ffnOut.dataDevice == DataDevice::CPU && ffnOut.cpuData != nullptr) {
                        // Rejection occurs before AddTo. Rebuild the old view
                        // and upload fresh data before using the separate operators.
                        V41ResetMultiDevice(ffnOut);
                        ffnOut.Reshape(ffnDims);
                        PrepareMultiCudaReplicatedData(ffnOut, tpDevices, true);
                    }
                    if (ffnOut.dataType == DataType::FLOAT32 &&
                        sharedExpertOut.dataType == DataType::BFLOAT16) {
                        Data &sharedFloat = ws->reductionBuffers["moe_shared_sum"];
                        Data &rounded = ws->reductionBuffers["moe_rounded_sum"];
                        ToDataType(sharedExpertOut, sharedFloat, DataType::FLOAT32);
                        AddTo(ffnOut, sharedFloat);
                        ToDataType(ffnOut, rounded, DataType::BFLOAT16);
                        rounded.Reshape(ffnDims);
                        V41HcPost(rounded, *curHidden, ffnPost, ffnComb, *nextHidden);
                        ffnPostDone = true;
                    } else {
                        AddTo(ffnOut, sharedExpertOut);
                    }
                }
            }
#endif
            if (!ffnPostDone) V41HcPost(ffnOut, *curHidden, ffnPost, ffnComb, *nextHidden);
            std::swap(curHidden, nextHidden);
            preMixPtr = &ffnPre;
            ReleaseForwardPrefill(f, {&ffnInput, &ffnOut, &ffnPost, &ffnComb, &ws->sharedExpertOut, nextHidden});
            if (dumpDebug) {
                V41DumpTensor(*curHidden, "fl_layer" + std::to_string(layer) + dumpSuffix);
            }
    }
    std::vector<int> DeepSeekV41Model::FinishForward(ForwardWorkspace &f) {
        auto &inputSegments = f.inputSegments;
        auto &generationConfigsIn = f.generationConfigsIn;
        auto &lastTokens = f.lastTokens;
        auto &retLogits = f.retLogits;
        auto &samplingPastKeyValues = f.samplingPastKeyValues;
        auto &numSegments = f.numSegments;
        auto &boundedPrefill = f.boundedPrefill;
        auto &tailSegments = f.tailSegments;
        auto &single = f.single;
        auto &dsparkVerify = f.dsparkVerify;
        auto &seqlen = f.seqlen;
        auto &tp = f.tp;
        auto &moeStep = f.moeStep;
        auto &curHidden = f.curHidden;
        auto &preMixPtr = f.preMixPtr;
#ifdef USE_CUDA
        auto &graphState = f.graphState;
        auto &graphReplay = f.graphReplay;
        auto &graphCapture = f.graphCapture;
        auto &graphPoolOpen = f.graphPoolOpen;
#endif
        const auto &segments = boundedPrefill ? tailSegments : inputSegments;
        auto catSegments = [&](std::vector<Data> &parts, Data *tmp) -> Data* {
            Data *acc = &parts[0];
            int t = 0;
            for (size_t i = 1; i < parts.size(); i++) {
                Cat(*acc, parts[i], 1, tmp[t]);
                acc = &tmp[t];
                t ^= 1;
            }
            return acc;
        };
        if (moeStep) moeStep->Finish();
#ifdef USE_CUDA
        if (f.prefill && !f.prefill->sampleOutput) {
            auto &payload = *f.prefill->payload;
            AssertInFastLLM(curHidden->dataType == BFLOAT16 && preMixPtr->dataType == FLOAT32,
                "DeepSeek hybrid PP payload dtype mismatch.");
            curHidden->ToDevice(CUDA);
            preMixPtr->ToDevice(CUDA);
            payload.Resize({1, seqlen, hc_mult * (embed_dim + 2)});
            FastllmCudaCopyFromDeviceToDevice(payload.cudaData, curHidden->cudaData, curHidden->GetBytes());
            FastllmCudaCopyFromDeviceToDevice((uint8_t*)payload.cudaData + curHidden->GetBytes(),
                preMixPtr->cudaData, preMixPtr->GetBytes());
            for (const auto &seg : inputSegments) seg.state->totalLen += seg.seqlen;
            return std::vector<int>{-1};
        }
#endif
        if (inputSegments[0].prefillCacheOnly) {
            // The last chunk rebuilds all late SWA windows in bounded replay.
            // No intermediate logits are consumed by the scheduler.
            for (const auto &seg : inputSegments) seg.state->totalLen += seg.seqlen;
            return std::vector<int>{-1};
        }
#ifdef USE_CUDA
        // ---- 收尾：结束捕获 / 累计预热轮次 ----
        if (graphState) {
            if (graphCapture) {
                bool ok = !graphState->captureFailed &&
                          (int)graphState->segments.size() == kV41GraphSegmentsPerLayer * block_cnt &&
                          !FastllmCudaGetThreadError() && !FastllmCudaGetGraphError();
                if (ok && graphPoolOpen) {
                    ok = FastllmCudaGraphMemoryPoolEnd(graphState->reservedPointers);
                    graphPoolOpen = false;
                }
                if (ok) {
                    V41ComposeGraphSegments(*graphState, !dsparkVerify && !v41DsparkEnabled, engram_layer_ids);
                    graphState->captured = true;
                    CollectForwardGraphPointers(f, graphState->boundaryPointers);
                    if (V41DecodeCudaGraphVerbose()) {
                        fprintf(stderr, "[Fastllm] DeepSeek-V4.1 decode CUDA graph captured: "
                                        "%d segments on %d device(s), tp=%d, tokens=%d\n",
                                (int)graphState->segments.size(),
                                (int)graphState->devices.size(), (int)graphState->tensorParallel, seqlen);
                        fflush(stderr);
                    }
                } else {
                    if (graphPoolOpen) {
                        FastllmCudaGraphMemoryPoolAbort();
                        graphPoolOpen = false;
                    }
                    GiveUpForwardGraph(f, "finalize capture");
                }
            } else if (!graphReplay && !graphState->disabled) {
                graphState->warmupRounds++;
            }
        }
#endif

        // ---- DSpark 校验片段：计算每个位置的目标 logits ----
        // Greedy uses argmax; sampling uses the same target filtering as
        // LLMSamplingBlock and the shared Qwen chain-rejection implementation.
        if (dsparkVerify) {
            Data allHidden, normed, allLogits, topk;
            V41HcApplyPre(*curHidden, *preMixPtr, allHidden);
            RMSNorm(allHidden, weight["norm.weight"], rms_norm_eps, normed);
            // Reuse the sharded head and gather complete logits for DSpark.
            DsparkProjectHead(normed, allLogits);
            ToDataType(allLogits, DataType::FLOAT32);
            segments[0].spec->tokens.resize(seqlen);
            const GenerationConfig sampling = DsparkSamplingConfig(generationConfigsIn[0]);
            if (sampling.top_k > 1) {
                DsparkSampleVerify(allLogits, *segments[0].state->dspark,
                                    *segments[0].spec, sampling);
            } else {
                DsparkMaskToolLogits(allLogits, sampling, segments[0].spec->draftTokens);
                TopK(allLogits, topk, 1);
                topk.ToDevice(DataDevice::CPU);
                const int stride = topk.dims[topk.dims.size() - 1];
                const float *topkData = (const float*)topk.cpuData;
                for (int i = 0; i < seqlen; i++) {
                    segments[0].spec->tokens[i] = (int)(topkData[(uint64_t)i * stride] + 1e-3);
                }
            }
            for (auto &seg : segments) {
                seg.state->totalLen += seg.seqlen;
            }
            const int last = segments[0].spec->acceptedDrafts >= 0 ?
                segments[0].spec->acceptedDrafts : seqlen - 1;
            return std::vector<int>{segments[0].spec->tokens[last]};
        }

        // ---- head（每个片段只取最后一个 token）----
        Data headInput;
        Data *samplingInput = &headInput;
#ifdef USE_CUDA
        if (f.fullGraphHead && f.fullGraphLogits) {
            // Precomputed logits make sampling read-only with respect to the
            // hidden state. Borrow the graph-owned tensor instead of allocating
            // and copying a temporary replica on every device and decode step.
            samplingInput = f.fullGraphHead;
        } else
#endif
        if (std::all_of(segments.begin(), segments.end(),
                       [](const DeepSeekV41Segment &seg) { return seg.seqlen == 1; })) {
            V41HcApplyPre(*curHidden, *preMixPtr, headInput);
        } else {
            std::vector<Data> lastHidden(numSegments), lastPre(numSegments);
            for (int s = 0; s < numSegments; s++) {
                int end = segments[s].offset + segments[s].seqlen;
                Split(*curHidden, 1, end - 1, end, lastHidden[s]);
                Split(*preMixPtr, 1, end - 1, end, lastPre[s]);
            }
            Data hiddenTmp[2], preTmp[2];
            Data *hiddenAll = catSegments(lastHidden, hiddenTmp);
            Data *preAll = catSegments(lastPre, preTmp);
            V41HcApplyPre(*hiddenAll, *preAll, headInput);
        }

        std::vector<int> ret;
        std::vector<int> samplingSeqLens(numSegments, 1);
        std::vector<GenerationConfig> generationConfigs = generationConfigsIn;
        for (auto &config : generationConfigs) {
            config = DsparkSamplingConfig(config);
        }
        // head 按行切分：两张卡各算一半词表，LLMSamplingBlock 的
        // SampleTensorParallelGreedyLogits / GatherTensorParallelLogitsToRoot 负责合并。
        // 不切分时 head 的 Linear 会走 multicuda 的"通用切分 + gather"路径，
        // 那条路径读的是复制张量已经失效的 root，结果是错的。
        if (tp) {
            weight["head.weight"].tpLinearType = TP_LINEAR_ROW;
        }
        Data referenceHeadNorm, referenceHeadLogits;
        Data *precomputedHeadLogits = nullptr;
        bool preserveHeadLogits = false;
#ifdef USE_CUDA
        if (f.fullGraphLogits) {
            precomputedHeadLogits = f.fullGraphLogits;
            preserveHeadLogits = true;
        }
#endif
        if (!precomputedHeadLogits && weight["head.weight"].dataType == DataType::FLOAT32) {
            V41RMSNormBF16(headInput, weight["norm.weight"], rms_norm_eps, referenceHeadNorm);
            ToDataType(referenceHeadNorm, DataType::FLOAT32);
            Linear(referenceHeadNorm, weight["head.weight"], *GetEmptyData(), referenceHeadLogits);
            precomputedHeadLogits = &referenceHeadLogits;
        }
#ifdef USE_CUDA
        if (precomputedHeadLogits == nullptr && !tp && V41EnvFlag("FASTLLM_DSV41_REFERENCE_MATH")) {
            headInput.ToDevice(DataDevice::CUDA);
            V41RMSNormBF16(headInput, weight["norm.weight"], rms_norm_eps, referenceHeadNorm);
            if (FastllmCudaV41ReferenceLinear(referenceHeadNorm, weight["head.weight"],
                                             referenceHeadLogits, 1, false)) {
                precomputedHeadLogits = &referenceHeadLogits;
            }
        }
#endif
        LLMSamplingBlock(this, samplingInput, &weight["norm.weight"], &weight["head.weight"],
                         rms_norm_eps, numSegments, true, samplingSeqLens, samplingPastKeyValues,
                         generationConfigs, lastTokens, retLogits, ret, precomputedHeadLogits,
                         nullptr, nullptr, nullptr, preserveHeadLogits);

        for (const auto &seg : inputSegments) {
            seg.state->totalLen += seg.seqlen;
        }
        // FASTLLM_DSV41_KV_STATS=1：打印实际占用的长期 KV 字节数（用于核对每 token 的缓存开销）
        static const bool kvStats = V41EnvFlag("FASTLLM_DSV41_KV_STATS");
        if (kvStats && !segments.empty() && segments[0].state != nullptr) {
            const DeepSeekV41RequestState *st = segments[0].state.get();
            long long cmpBytes = 0, idxBytes = 0, winBytes = 0;
            auto usedBytes = [](const Data &d, int rows) -> long long {
                if (d.dims.size() != 3 || rows <= 0) {
                    return 0;
                }
                return (long long)rows * d.dims[2] * d.unitSize / d.unitSizeDiv;
            };
            for (const auto &layer : st->layers) {
                cmpBytes += usedBytes(layer.compressedKV, layer.compressedBlocks);
                idxBytes += usedBytes(layer.indexK, layer.compressedBlocks);
                winBytes += usedBytes(layer.windowKV, layer.windowKV.dims.size() == 3 ? layer.windowKV.dims[1] : 0);
            }
            printf("[V41 KV] tokens=%d compressedKV=%lld B indexK=%lld B (long-term %lld B, %.1f B/token) window=%lld B model-estimate=%lld B/token\n",
                   st->totalLen, cmpBytes, idxBytes, cmpBytes + idxBytes,
                   st->totalLen > 0 ? (double)(cmpBytes + idxBytes) / st->totalLen : 0.0,
                   winBytes, KVCacheBytesPerToken());
        }
        return ret;
    }
    void DeepSeekV41Model::RunForwardSharedExpert(ForwardWorkspace &f, int layer) {
        auto &seqlen = f.seqlen;
        auto &tp = f.tp;
        auto &localWorkspace = f.localWorkspace;
        auto &ws = f.ws;
        auto &hasSharedExpertOut = f.hasSharedExpertOut;
        auto &ffnInput = ws->ffnInput;
        auto &sharedExpertOut = ws->sharedExpertOut;
        auto quantizedLinear = [&](Data &input, const std::string &name, Data &output, bool replicated = false,
                                   Data *scratch = nullptr, bool inputQuantized = false) {
            // Graphs and asynchronous TP decode need stable per-weight buffers.
            // Synchronous TP prefill can release each full-chunk temporary as
            // soon as its linear completes, instead of retaining all layers.
            LinearWithActivationQuant(input, name, output, replicated,
                scratch != nullptr ? scratch : (ws == &localWorkspace && (!tp || seqlen > 1) ?
                    (tp ? nullptr : &ws->quantizedActivation) : &ws->quantizedActivations[name]), inputQuantized,
                ws == &localWorkspace && seqlen > 1 ? nullptr : &ws->reductionBuffers[name]);
        };

        const std::string pre = "layers." + std::to_string(layer);
        auto body = [&] {

            if (hasSharedExpertOut) {
                // Keep a complete shared expert on each rank, preserving
                // the original BF16 rounding boundary without a collective.
                Data &ww1 = ws->sharedSwiglu, &ww3 = ws->sharedGateup;
                bool sharedQuantized = false;
                quantizedLinear(ffnInput, pre + ".ffn.shared_experts.gateup.weight", ww3,
                                tp);
#ifdef USE_CUDA
                if (!V41EnvFlag("FASTLLM_DSV41_REFERENCE_MATH") &&
                    quantizedLinearNames.count(pre + ".ffn.shared_experts.w2.weight") &&
                    !ww3.dims.empty() && ww3.dims.back() % 64 == 0) {
                    sharedQuantized = tp ?
                        MultiCudaDeepSeekV41SharedSwigluQuantized(ww3, swiglu_limit, ww1) :
                        FastllmCudaDeepSeekV41SharedSwigluQuantized(ww3, swiglu_limit, ww1);
                }
                if (!sharedQuantized) {
                    AssertInFastLLM(tp ? MultiCudaDeepSeekV41SharedSwiglu(ww3, swiglu_limit, ww1) :
                        FastllmCudaDeepSeekV41SharedSwiglu(ww3, swiglu_limit, ww1),
                        "DeepSeekV41: CUDA shared expert activation rejected input.");
                }
#else
                Swiglu(ww3, ww1);
#endif
                quantizedLinear(ww1, pre + ".ffn.shared_experts.w2.weight", sharedExpertOut,
                                tp, nullptr, sharedQuantized);
                ToDataType(sharedExpertOut, DataType::BFLOAT16);
            }

        };
        if (f.captureGpuMoeTogether) {
            body();
            return;
        }
        RunForwardGraphSegment(f, body, kV41GraphSegmentsPerLayer * layer + 3, hasSharedExpertOut,
            [](DeepSeekV41GraphSegmentMeta &) {}, [](const DeepSeekV41GraphSegmentMeta &) {});
    }
    DeepSeekV41MoeContext DeepSeekV41Model::MakeForwardMoeContext(ForwardWorkspace &f, int layer) {
        auto &inputSegments = f.inputSegments;
        auto &boundedPrefill = f.boundedPrefill;
        auto &tailSegments = f.tailSegments;
        auto &single = f.single;
        auto &dsparkVerify = f.dsparkVerify;
        auto &seqlen = f.seqlen;
        auto &tpDevices = f.tpDevices;
        auto &tp = f.tp;
        auto &ws = f.ws;
        auto &graphActive = f.graphActive;
        auto &dumpDebug = f.dumpDebug;
        auto &moeWeights = f.moeWeights;
        auto &hasSharedExpertOut = f.hasSharedExpertOut;
        const auto &segments = boundedPrefill ? tailSegments : inputSegments;

        return DeepSeekV41MoeContext {
            *ws, moeWeights, tpDevices, layer, seqlen,
            tp, single, dsparkVerify, graphActive, dumpDebug, hasSharedExpertOut,
            single && segments[0].spec != nullptr && segments[0].spec->wantAllTokens,
            [this, &f, layer] { RunForwardSharedExpert(f, layer); },
            [this, &f] { auto *ws=f.ws;
                ReleaseForwardPrefill(f, {&ws->x, &ws->gateInput, &ws->gateLogits,
                    &ws->sharedGateup, &ws->sharedSwiglu, &ws->quantizedActivation}); },
            f.prefill ? f.prefill->hostExperts : nullptr,
            f.prefill ? f.prefill->firstLayer : -1
        };
    }
    void DeepSeekV41Model::GiveUpForwardGraph(ForwardWorkspace &f, const char *stage) {
        auto &inputSegments = f.inputSegments;
        auto &boundedPrefill = f.boundedPrefill;
        auto &tailSegments = f.tailSegments;
#ifdef USE_CUDA
        auto &graphState = f.graphState;
        auto &graphReplay = f.graphReplay;
        auto &graphCapture = f.graphCapture;
#endif
        const auto &segments = boundedPrefill ? tailSegments : inputSegments;
#ifdef USE_CUDA
            if (!graphState) {
                return;
            }
            const int oriDevice = FastllmCudaGetDevice();
            for (int device : graphState->devices) {
                FastllmCudaSetDevice(device);
                // 捕获被判废后运行时的 last-error 会一直粘着，回退到逐算子之前必须清掉，
                // 否则第一个算子的 cudaGetLastError 会误判为 kernel 失败。
                FastllmCudaClearLastError();
                FastllmCudaSyncDevice(device);
                FastllmCudaClearLastError();
            }
            FastllmCudaSetDevice(oriDevice);
            if (graphState->tensorParallel && graphState->devices.size() > 1) {
                // 常驻 worker 线程有各自的 last-error，也要在它们自己的线程上清
                MultiCudaRunDeviceCallbacks(graphState->devices, [](int rank, int device) {
                    (void)rank;
                    FastllmCudaSetDevice(device);
                    FastllmCudaClearLastError();
                });
            }
            graphState->DestroyCapturedGraph();
            graphState->disabled = true;
            fprintf(stderr, "[Fastllm] DeepSeek-V4.1 decode CUDA graph disabled at %s: %s\n",
                    stage, FastllmCudaGraphLastError());
            fflush(stderr);
            FastllmCudaClearThreadError();
            FastllmCudaClearGraphError();
            graphReplay = false;
            graphCapture = false;
#endif
    }
    void DeepSeekV41Model::CollectForwardGraphPointers(ForwardWorkspace &f, std::vector<const void*> &out) {
        auto &inputSegments = f.inputSegments;
        auto &boundedPrefill = f.boundedPrefill;
        auto &tailSegments = f.tailSegments;
        auto &tpDevices = f.tpDevices;
        auto &ws = f.ws;
        const auto &segments = boundedPrefill ? tailSegments : inputSegments;
        auto &hiddenStates = ws->hiddenStates;
        auto &hiddenTemp = ws->hiddenTemp;
        auto &preMix = ws->preMix;
        auto &attnPre = ws->attnPre;
        auto &attnPost = ws->attnPost;
        auto &attnComb = ws->attnComb;
        auto &ffnPre = ws->ffnPre;
        auto &ffnPost = ws->ffnPost;
        auto &ffnComb = ws->ffnComb;
        auto &x = ws->x;
        auto &attnInput = ws->attnInput;
        auto &qr = ws->qr;
        auto &qNorm = ws->qNorm;
        auto &q = ws->q;
        auto &kv = ws->kv;
        auto &attnOut = ws->attnOut;
        auto &woAOut = ws->woAOut;
        auto &attnProj = ws->attnProj;
        auto &ffnInput = ws->ffnInput;
        auto &expertIndex = ws->expertIndex;
        auto &expertScore = ws->expertScore;
        auto &rawKVAll = ws->rawKVAll;
        auto &rawScoreAll = ws->rawScoreAll;
        auto &qIdxAll = ws->qIdxAll;
        auto &idxWeightsAll = ws->idxWeightsAll;
        auto &sharedExpertOut = ws->sharedExpertOut;
#ifdef USE_CUDA
            out.clear();
            std::vector<const Data *> tensors = {
                &hiddenStates, &hiddenTemp, &preMix,
                &attnPre, &attnPost, &attnComb, &ffnPre, &ffnPost, &ffnComb,
                &x, &attnInput, &qr, &qNorm, &q, &kv, &attnOut, &woAOut, &attnProj,
                &rawKVAll, &rawScoreAll, &qIdxAll, &idxWeightsAll,
                &ffnInput, &expertIndex, &expertScore, &ws->sharedExpertOut,
                &ws->ffnOut, &ws->w1, &ws->w2, &ws->w3};
            for (const auto &entry : ws->reductionBuffers) tensors.push_back(&entry.second);
            for (const Data *data : tensors) {
                out.push_back(data->cudaData);
                // 非空却不在卡上，说明它被某个 host 侧算子搬走过，显存已经还给内存池，
                // 图里烤死的地址已经指向别人。用一个哨兵值让核对必然失败。
                if (data->Count(0) > 0 && data->dataDevice != DataDevice::CUDA) {
                    out.push_back((const void*)(intptr_t)-1);
                }
                for (int device : tpDevices) {
                    auto it = data->multiDeviceDatas.find(device);
                    out.push_back(it == data->multiDeviceDatas.end() || it->second == nullptr ?
                                  nullptr : it->second->cudaData);
                }
            }
#endif
    }
    void DeepSeekV41Model::ReleaseForwardPrefill(ForwardWorkspace &f, std::initializer_list<Data *> tensors) {
        auto &inputSegments = f.inputSegments;
        auto &boundedPrefill = f.boundedPrefill;
        auto &tailSegments = f.tailSegments;
        auto &seqlen = f.seqlen;
        auto &tp = f.tp;
        auto &localWorkspace = f.localWorkspace;
        auto &ws = f.ws;
        auto &dumpDebug = f.dumpDebug;
        auto &preMixPtr = f.preMixPtr;
        const auto &segments = boundedPrefill ? tailSegments : inputSegments;
            if (ws != &localWorkspace || tp || dumpDebug || seqlen <= window_size) return;
            for (Data *data : tensors) {
                if (data->isFake || data == preMixPtr) continue;
#ifdef USE_CUDA
                // MoE workers may allocate on other streams. Wait for this
                // tensor's consumers via a pool event before allowing reuse.
                if (data->cudaData != nullptr && !data->cudaDataBorrowed) {
                    const int original = FastllmCudaGetDevice();
                    const int device = data->dataDeviceIds.empty() ? original : data->dataDeviceIds[0];
                    FastllmCudaSetDevice(device);
                    if (!data->directMemory && FastllmCudaFreeAfterCurrentThreadStream(data->cudaData)) {
                        data->cudaData = nullptr;
                    } else {
                        FastllmCudaSyncDevice(device);
                    }
                    FastllmCudaSetDevice(original);
                }
#endif
                data->FreeSpace();
                data->expansionDims.clear();
                data->dims.clear();
                data->strides.clear();
            }
    }
    void DeepSeekV41Model::RunForwardGraphSegment(ForwardWorkspace &f, const std::function<void()> &body, int index, bool capturable, const std::function<void(DeepSeekV41GraphSegmentMeta&)> &record, const std::function<void(const DeepSeekV41GraphSegmentMeta&)> &restore) {
        auto &inputSegments = f.inputSegments;
        auto &boundedPrefill = f.boundedPrefill;
        auto &tailSegments = f.tailSegments;
#ifdef USE_CUDA
        auto &graphState = f.graphState;
        auto &graphReplay = f.graphReplay;
        auto &graphCapture = f.graphCapture;
        auto &graphPoolOpen = f.graphPoolOpen;
#endif
        const auto &segments = boundedPrefill ? tailSegments : inputSegments;
#ifdef USE_CUDA
            if (graphReplay) {
                if ((V41DecodeCudaGraphReplayMask() & (1 << (index % kV41GraphSegmentsPerLayer))) == 0 ||
                    (index < (int)graphState->segments.size() &&
                     graphState->segments[index].passthrough)) {
                    body();
                    return;
                }
                if (index < (int)graphState->segments.size() &&
                    V41LaunchGraphSegment(*graphState, index, f.workerOrderedGraphReplay)) {
                    restore(graphState->segments[index]);
                    return;
                }
                GiveUpForwardGraph(f, "replay");
                body();
                return;
            }
            if (graphCapture) {
                if (!capturable) {
                    // 占位段：逐算子跑，但要占住段号，回放时也走同一条路
                    body();
                    DeepSeekV41GraphSegment placeholder;
                    placeholder.passthrough = true;
                    graphState->segments.push_back(placeholder);
                    record(graphState->segments.back());
                    return;
                }
                if (index != V41DecodeCudaGraphFailAt() &&
                    V41CaptureGraphSegment(*graphState, body)) {
                    record(graphState->segments.back());
                    return;
                }
                if (graphPoolOpen) {
                    FastllmCudaGraphMemoryPoolAbort();
                    graphPoolOpen = false;
                }
                GiveUpForwardGraph(f, "capture");
                // 捕获时没有真正执行任何 kernel，输入也没被改动，直接逐算子重跑一遍。
                body();
                return;
            }
#else
            (void)index;
            (void)capturable;
            (void)record;
            (void)restore;
#endif
            body();
    }
    std::vector<int> DeepSeekV41Model::ForwardSegments(const std::vector<DeepSeekV41Segment> &inputSegments,
                                                       const Data &inputIds,
                                                       const Data *inputEmbeds,
                                                       const std::vector<int> *imageMask,
                                                       const std::vector<GenerationConfig> &generationConfigsIn,
                                                       const LastTokensManager &lastTokens,
                                                       std::vector<std::vector<float>*> *retLogits,
                                                       std::vector<std::pair<Data*, Data*> > &samplingPastKeyValues) {
        if (executionMode == ExecutionMode::Gpu) return ForwardGpu(inputSegments, inputIds, inputEmbeds, imageMask, generationConfigsIn, lastTokens, retLogits, samplingPastKeyValues);
        return ForwardHybrid(inputSegments, inputIds, inputEmbeds, imageMask, generationConfigsIn, lastTokens, retLogits, samplingPastKeyValues);
    }

#ifdef USE_CUDA
#include "deepseekv41-full-graph.inc"
#endif

    void DeepSeekV41Model::WarmUp() {
        printf("Warmup...\n");
        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {1});
        Data attentionMask = Data(DataType::FLOAT32, {1, 1}, {0});
        Data positionIds = Data(DataType::FLOAT32, {1, 1}, {0});
        std::vector<std::pair<Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(this->dataType), Data(this->dataType)));
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        {
            std::lock_guard<std::mutex> guard(v41StateMutex);
            v41States.erase((const void*)&pastKeyValues);
            v41StatesByFirstKey.erase((const void*)&pastKeyValues[0].first);
        }
#ifdef USE_CUDA
        if (GetFastllmEnv().cudaGraph && v41MoeCudaGraphCompatible &&
            V41DeviceMapUsesCuda(this->deviceMap) &&
            std::getenv("FASTLLM_DSV41_DUMP_DIR") == nullptr) {
            // Capture common decode sizes before accepting requests. Unknown
            // batch sizes and invalidated graphs fall back to eager execution.
            std::vector<int> batches;
            const int limit = std::max(1, this->maxBatch);
            for (int batch = 1; batch <= limit; batch *= 2) {
                batches.push_back(batch);
                if (batch > limit / 2) break;
            }
            if (batches.back() != limit) batches.push_back(limit);
            v41CudaGraphWarmingUp = true;
            try {
                for (int batch : batches) {
                    std::vector<std::shared_ptr<DeepSeekV41RequestState>> states;
                    for (int i = 0; i < batch; ++i) {
                        states.emplace_back(std::make_shared<DeepSeekV41RequestState>());
                        states.back()->layers.resize(block_cnt);
                    }
                    std::vector<Data> dummy(2 * block_cnt * batch);
                    std::vector<std::pair<Data*, Data*>> past;
                    for (size_t i = 0; i < dummy.size(); i += 2) past.emplace_back(&dummy[i], &dummy[i + 1]);
                    std::vector<GenerationConfig> configs(batch);
                    for (auto &config : configs) { config.do_sample = false; config.top_k = 1; }
                    LastTokensManager last(batch, 64);
                    Data ids(DataType::FLOAT32, {1, batch}, std::vector<float>(batch, 1));
                    // Initial token, eager warmups, capture, and a replay.
                    const int rounds = V41DecodeCudaGraphWarmupRounds() + 3;
                    for (int step = 0; step < rounds; ++step) {
                        std::vector<DeepSeekV41Segment> segments(batch);
                        for (int i = 0; i < batch; ++i) {
                            segments[i].state = states[i];
                            segments[i].startPos = states[i]->totalLen;
                            segments[i].offset = i;
                            segments[i].seqlen = 1;
                        }
                        ForwardSegments(segments, ids, nullptr, nullptr, configs, last, nullptr, past);
                    }
                    if (batch == 1) BuildFullDecodeGraph(batch, states);
                }
            } catch (...) {
                v41CudaGraphWarmingUp = false;
                throw;
            }
            v41CudaGraphWarmingUp = false;
        }
#endif
        this->kvCacheId = 0;
        // 占位 KV 不反映真实占用；按模型几何估算每个 token 的长期缓存字节数
        // （压缩 KV + indexer key；滑窗缓存长度固定，不计入），
        // 折算成 kvCacheDataType 的元素数供调度器估算上下文预算。
        const long long bytesPerToken = std::max(1LL, KVCacheBytesPerToken());
        // 调度器算的是 GetDataBytes(kvCacheDataType, 1, elementsInKVCachePerToken)，这里反解元素数
        if (this->kvCacheDataType == DataType::FP4_E2M1) {
            elementsInKVCachePerToken = ((bytesPerToken + 8) / 9) * 16;      // 每 16 个元素 9 字节
        } else if (this->kvCacheDataType == DataType::FP8_E4M3) {
            elementsInKVCachePerToken = bytesPerToken;
        } else {
            long long unitBytes = std::max(1LL, (long long)GetDataBytes(this->kvCacheDataType, 1, 1));
            elementsInKVCachePerToken = (bytesPerToken + unitBytes - 1) / unitBytes;
        }
        elementsInKVCachePerToken = std::max(1LL, elementsInKVCachePerToken);
        printf("finish.\n");
    }

    // 每个 token 的长期 KV 缓存字节数（压缩 KV + indexer key，按 kv_source 层与 compress_ratio 累加）
    long long DeepSeekV41Model::KVCacheBytesPerToken() const {
        const bool fp8KV = this->kvCacheDataType == DataType::FP8_E4M3;
        const bool fp4KV = this->kvCacheDataType == DataType::FP4_E2M1;
        auto rowBytes = [](int dim, int quantMode, int blockSize) -> long long {
            if (quantMode <= 0) {
                return (long long)dim * 2;                                  // BF16
            }
            return (quantMode == 1 ? dim : dim / 2) + dim / blockSize;
        };
        const long long cmpRow = rowBytes(head_dim_full, fp4KV ? 3 : (fp8KV ? 1 : 0), fp4KV ? 16 : 32);
        const long long idxRow = rowBytes(index_head_dim, fp4KV ? 2 : (fp8KV ? 1 : 0), 32);
        long long bytesPerToken = 0;
        for (int layer = 0; layer < block_cnt; layer++) {
            if (!isKvSource[layer]) {
                continue;
            }
            int ratio = std::max(1, compress_ratios[layer]);
            bytesPerToken += cmpRow / ratio;
            if (isIndexSource[layer]) {
                bytesPerToken += idxRow / ratio;
            }
        }
        return bytesPerToken;
    }
}
