//
// DeepSeek-V4.1 系列模型实现。架构说明见 include/models/deepseekv41.h。
//
// 本文件实现通用（CUDA / CPU 混合）路径：
//   * 注意力、Hyper-Connections、indexer 等在执行器选择的设备上运行（通常是 GPU）；
//   * 路由专家通过 MergeMOEBlock 交给 MoE 设备（cpu / numa / cuda）；
//   * Engram 哈希表以 FP8 + UE8M0 scale 原样保存在 CPU 内存中（每层约 100GB），
//     查表在 CPU 完成，后续 wkv 投影与门控在 GPU 完成。
//

#include "deepseekv41.h"

#include "baseblock.h"
#include "executor.h"
#include "utils.h"
#include "json11.hpp"

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>

#if !defined(_WIN32) && !defined(_WIN64)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace fastllm {
    namespace {
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

        bool V41EnvFlag(const char *name) {
            const char *v = std::getenv(name);
            return v != nullptr && v[0] != '\0' && strcmp(v, "0") != 0;
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
        //   random   ：表是纯随机访问，MADV_RANDOM 关掉内核的顺序预读（mmap 模式下最有用）。
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

        // ---------------- 张量并行（multicuda）辅助 ----------------
        //
        // V4.1 的切分方式与 V4 一致：只有 query head 维被切分（wq_b 按行切 ->
        // 稀疏注意力 -> wo_a 按 head 组切 -> wo_b 按列切 + all-reduce），
        // 其余激活全部在每张卡上保留一份完整副本。跨层共享的压缩 KV / indexer key /
        // 滑窗 KV 缓存也是每卡一份：它们与 head 无关，切分后每一个后续层（以及
        // 后续 index source 层）都要 all-gather 才能用，而复制只多花一份算力。
        bool V41DeviceSpecUsesType(const std::string &spec, const std::string &type) {
            std::string normalized = spec;
            std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                           [](unsigned char c) { return (char)std::tolower(c); });
            return normalized == type || normalized.rfind(type + ":", 0) == 0;
        }

        // 模型主体是否真的跑在 CUDA 上。--device cpu 时 FastllmCudaGetDevice() 依然返回 0，
        // 只看它会把 CPU 路径也当成可捕获，捕出来的是空图，回放等于什么都没算。
        bool V41DeviceMapUsesCuda(const std::map<std::string, int> &deviceMap) {
            for (const auto &it : deviceMap) {
                if (V41DeviceSpecUsesType(it.first, "cuda") ||
                    V41DeviceSpecUsesType(it.first, "multicuda")) {
                    return true;
                }
            }
            return false;
        }

        // 按层切分（流水线 / 模型分片）：device map 里出现了不止一张卡，但不是 multicuda。
        // 这种布局下每层跑在不同的卡上，一段图只能属于一张卡，当前的"整趟捕获都在同一张卡上
        // 开始"的做法会把另一张卡的算子漏在图外（那些流根本没在捕获），回放就少算了。
        int V41DeviceMapCudaDeviceCount(const std::map<std::string, int> &deviceMap) {
            std::set<int> devices;
            for (const auto &it : deviceMap) {
                const std::string &spec = it.first;
                if (!V41DeviceSpecUsesType(spec, "cuda")) {
                    continue;
                }
                size_t pos = spec.find(':');
                if (pos == std::string::npos) {
                    devices.insert(0);
                } else {
                    devices.insert(atoi(spec.c_str() + pos + 1));
                }
            }
            return (int)devices.size();
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
        void V41ResetMultiDevice(Data &data) {
            if (!data.multiDeviceData) {
                return;
            }
            for (auto &it : data.multiDeviceDatas) {
                delete it.second;
            }
            data.multiDeviceDatas.clear();
            data.multiDeviceData = false;
            data.ClearTensorParallelLayout();
        }

        // 把 root 的形状 / 扩容信息与 0 号卡副本对齐（副本才是真实数据）
        void V41SyncRootFromReplica(Data &data, const std::vector<int> &devices) {
            if (!data.multiDeviceData || devices.empty()) {
                return;
            }
            auto it = data.multiDeviceDatas.find(devices[0]);
            if (it == data.multiDeviceDatas.end() || it->second == nullptr) {
                return;
            }
            Data *first = it->second;
            data.dataType = first->dataType;
            data.UpdateUnitSize();
            if (data.dims != first->dims) {
                data.Resize(first->dims);
            }
            data.expansionDims = first->expansionDims;
            data.strides = first->strides;
            data.expansionSize = first->expansionSize;
            data.expansionBytes = first->expansionBytes;
            data.dataDevice = DataDevice::CUDA;
            data.dataDeviceIds = devices;
            data.tpLayout = TP_LAYOUT_REPLICATED;
            data.tpAxis = -1;
            data.tpGlobalDims = data.dims;
        }

        // 复制布局下的张量拷贝：逐卡拷贝，不经过可能已经失效的 root
        void V41CopyTensor(Data &dst, const Data &src, const std::vector<int> &devices) {
            if (devices.empty() || !src.multiDeviceData || !src.IsTensorParallelReplicated()) {
                dst.CopyFrom(src);
                return;
            }
            V41ResetMultiDevice(dst);
            dst.FreeSpace();
            dst.dataType = src.dataType;
            dst.UpdateUnitSize();
            dst.Resize(src.dims);
            dst.dataDevice = DataDevice::CUDA;
            dst.dataDeviceIds = devices;
            dst.multiDeviceData = true;
            const int oriDevice = FastllmCudaGetDevice();
            for (int device : devices) {
                auto it = src.multiDeviceDatas.find(device);
                AssertInFastLLM(it != src.multiDeviceDatas.end() && it->second != nullptr,
                                "DeepSeekV41: missing TP replica while copying a tensor.");
                FastllmCudaSetDevice(device);
                Data *local = new Data();
                local->CopyFrom(*it->second);
                local->dataDeviceIds = {device};
                dst.multiDeviceDatas[device] = local;
            }
            FastllmCudaSetDevice(oriDevice);
            dst.tpLayout = TP_LAYOUT_REPLICATED;
            dst.tpAxis = -1;
            dst.tpGlobalDims = dst.dims;
            V41SyncRootFromReplica(dst, devices);
        }

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

        // 张量并行感知的赋值：复制布局下逐卡拷贝，否则退回普通 CopyFrom
        void V41Assign(Data &dst, const Data &src, const std::vector<int> &tpDevices) {
#ifdef USE_CUDA
            if (!tpDevices.empty() && src.multiDeviceData && src.IsTensorParallelReplicated()) {
                V41CopyTensor(dst, src, tpDevices);
                return;
            }
#endif
            dst.CopyFrom(src);
        }

        // 从复制布局的某一张卡副本拷到 CPU（root 在复制布局下只有形状信息）
        void V41ReplicaToCpu(Data &dst, const Data &src, const std::vector<int> &tpDevices) {
#ifdef USE_CUDA
            if (!tpDevices.empty() && src.multiDeviceData && src.IsTensorParallelReplicated()) {
                for (int device : tpDevices) {
                    auto it = src.multiDeviceDatas.find(device);
                    if (it == src.multiDeviceDatas.end() || it->second == nullptr ||
                        it->second->cudaData == nullptr) {
                        continue;
                    }
                    const int oriDevice = FastllmCudaGetDevice();
                    FastllmCudaSetDevice(device);
                    dst.CopyFrom(*it->second);
                    dst.ToDevice(DataDevice::CPU);
                    FastllmCudaSetDevice(oriDevice);
                    return;
                }
            }
#endif
            dst.CopyFrom(src);
            dst.ToDevice(DataDevice::CPU);
        }

        // 分配一个（可能是复制布局的）空张量
        void V41AllocLike(Data &dst, DataType type, const std::vector<int> &dims,
                          const Data &reference, const std::vector<int> &tpDevices) {
#ifdef USE_CUDA
            if (!tpDevices.empty() && reference.multiDeviceData &&
                reference.IsTensorParallelReplicated()) {
                V41ResetMultiDevice(dst);
                dst.FreeSpace();
                dst.dataType = type;
                dst.UpdateUnitSize();
                dst.Resize(dims);
                dst.dataDevice = DataDevice::CUDA;
                dst.dataDeviceIds = tpDevices;
                PrepareMultiCudaReplicatedData(dst, tpDevices, false);
                const int oriDevice = FastllmCudaGetDevice();
                for (int device : tpDevices) {
                    FastllmCudaSetDevice(device);
                    dst.multiDeviceDatas.at(device)->Allocate();
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
            RMSNorm(input, weight, eps, output);
            ToDataType(output, DataType::BFLOAT16);
        }

        void V41HcMix(const Data &input, Data &hcFn, Data &hcScale, Data &hcBase,
                      int hcMult, int sinkhornIters, float eps, float normEps,
                      Data &pre, Data &post, Data &comb) {
            V41Executor().Run("DeepSeekV41HcMix", {
                {"input", (Data*)&input}, {"hcFn", &hcFn}, {"hcScale", &hcScale}, {"hcBase", &hcBase},
                {"pre", &pre}, {"post", &post}, {"comb", &comb}
            }, {{"eps", eps}, {"normEps", normEps}}, {{"hcMult", hcMult}, {"sinkhornIters", sinkhornIters}});
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
            if (!disableFused && input.dataDevice == DataDevice::CUDA && !input.multiDeviceData &&
                V41FusedPreNormSameDevice(input, pre, normWeight) &&
                FastllmCudaDeepSeekV41HcPreNorm(input, pre, normWeight, eps, output)) {
                return;
            }
#endif
            V41HcApplyPre(input, pre, tmp);
            V41RMSNormBF16(tmp, normWeight, eps, output);
        }

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
            bool fileMapped = false;            // 表是文件 mmap（缺页可能要读盘）还是常驻内存
            const uint8_t *data = nullptr;      // FP8 E4M3, [rows, dim]
            const uint8_t *scale = nullptr;     // UE8M0, [rows, dim / scaleBlock]
            std::vector<uint8_t> dataStorage;
            std::vector<uint8_t> scaleStorage;
            void *mmapData = nullptr;
            size_t mmapDataLen = 0;
            void *mmapScale = nullptr;
            size_t mmapScaleLen = 0;

            ~V41EngramTable() {
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

        bool V41MapFileRange(const std::string &fileName, uint64_t offset, uint64_t bytes,
                             void *&mapping, size_t &mapLen, const uint8_t *&ptr) {
#if defined(_WIN32) || defined(_WIN64)
            return false;
#else
            int fd = open(fileName.c_str(), O_RDONLY);
            if (fd < 0) {
                return false;
            }
            long pageSize = sysconf(_SC_PAGESIZE);
            uint64_t alignedOffset = offset / pageSize * pageSize;
            mapLen = (size_t)(bytes + (offset - alignedOffset));
            mapping = mmap(nullptr, mapLen, PROT_READ, MAP_PRIVATE, fd, (off_t)alignedOffset);
            close(fd);
            if (mapping == MAP_FAILED) {
                mapping = nullptr;
                return false;
            }
            ptr = (const uint8_t*)mapping + (offset - alignedOffset);
            return true;
#endif
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

        // 5. 这些小权重保持源精度
        for (int i = 0; i < block_cnt; i++) {
            std::string pre = "layers." + std::to_string(i);
            this->cantQuantLinears.insert(pre + ".attn.compressor.wkv.weight");
            this->cantQuantLinears.insert(pre + ".attn.compressor.wgate.weight");
            this->cantQuantLinears.insert(pre + ".attn.indexer.wk.weight");
            this->cantQuantLinears.insert(pre + ".attn.indexer.weights_proj.weight");
            this->cantQuantLinears.insert(pre + ".ffn.gate.weight");
        }

        // 注意力的 head 切分有硬约束（CUDA 稀疏注意力 kernel 每 block 32 个 head，
        // wo_a 要求区间对齐到 o_group）。不满足时撤销 DeepSeekV4Model::InitParams
        // 注册的注意力 TP 权重，让它们整块加载，前向里注意力也退回复制布局。
        if (V41DeviceMapUsesMultiCuda(this->deviceMap)) {
            const int ranks = V41MultiCudaRankCount(this->deviceMap);
            const bool aligned =
                ranks > 1 && o_groups > 0 &&
                num_attention_heads % ranks == 0 &&
                (num_attention_heads / ranks) % 32 == 0 &&
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
                printf("[Fastllm] DeepSeek-V4.1 tensor parallel needs num_attention_heads / tp to be a "
                       "multiple of 32 and aligned to o_groups (got %d heads, %d ranks, o_groups=%d); "
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
        if (const char *env = std::getenv("FASTLLM_DSV41_ENGRAM_META")) {
            metaPath = env;
        } else if (V41HasKey(this->weight, "engram_meta_path")) {
            metaPath = this->weight.dicts["engram_meta_path"];
        } else if (V41HasKey(this->weight, "model_directory")) {
            metaPath = this->weight.dicts["model_directory"] + "engram_meta.json";
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
        static const bool wkvFp8 = V41EnvFlagOn("FASTLLM_DSV41_ENGRAM_WKV_FP8");
        std::set<std::string> tensorNameSet;
        if (wkvFp8) {
            tensorNameSet.insert(tensorNames.begin(), tensorNames.end());
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
            // Engram 表由模型自行读取（超出通用加载器的 int32 scale 索引范围）
            if (name.find(".engram.embed.") != std::string::npos) {
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
            result[it.first] = it.second;
        }
        return result;
    }

    void DeepSeekV41Model::OnModelWeightsLoaded() {
        if (engram_layer_ids.empty()) {
            return;
        }
        AssertInFastLLM(V41HasKey(this->weight, "model_directory"),
                        "DeepSeekV41: model directory is unknown, can't load engram tables.");
        std::string dir = this->weight.dicts["model_directory"];
        bool useMmap = V41EnvFlag("FASTLLM_DSV41_ENGRAM_MMAP");
        V41EngramMadviseCfg madviseCfg = V41EngramMadvise();
        engramTables.clear();
        for (size_t l = 0; l < engram_layer_ids.size(); l++) {
            int layer = engram_layer_ids[l];
            std::string base = "layers." + std::to_string(layer) + ".engram.embed.";
            V41SafeTensorInfo weightInfo, scaleInfo;
            AssertInFastLLM(V41FindSafeTensor(dir, base + "weight", weightInfo) &&
                            V41FindSafeTensor(dir, base + "scale", scaleInfo),
                            "DeepSeekV41: can't locate engram table " + base + "weight in " + dir);
            AssertInFastLLM(weightInfo.dtype == "F8_E4M3" && weightInfo.shape.size() == 2 &&
                            (scaleInfo.dtype == "F8_E8M0" || scaleInfo.dtype == "U8") &&
                            scaleInfo.shape.size() == 2 && scaleInfo.shape[0] == weightInfo.shape[0] &&
                            weightInfo.shape[1] % scaleInfo.shape[1] == 0,
                            "DeepSeekV41: unsupported engram table format for " + base + "weight");
            auto table = std::make_shared<V41EngramTable>();
            table->rows = weightInfo.shape[0];
            table->dim = (int)weightInfo.shape[1];
            table->scaleBlock = (int)(weightInfo.shape[1] / scaleInfo.shape[1]);
            AssertInFastLLM(table->dim == engram_head_dim && weightInfo.bytes == (uint64_t)table->rows * table->dim &&
                            scaleInfo.bytes == (uint64_t)table->rows * (table->dim / table->scaleBlock),
                            "DeepSeekV41: engram table byte count mismatch for " + base + "weight");
            printf("[Fastllm] DeepSeek-V4.1: loading engram table for layer %d (%.1f GB, %s%s%s)...\n",
                   layer, (weightInfo.bytes + scaleInfo.bytes) / 1e9, useMmap ? "mmap" : "resident",
                   madviseCfg.random ? " +random" : "", madviseCfg.hugePage ? " +hugepage" : "");
            fflush(stdout);
            bool mapped = false;
            if (useMmap) {
                mapped = V41MapFileRange(weightInfo.fileName, weightInfo.offset, weightInfo.bytes,
                                         table->mmapData, table->mmapDataLen, table->data) &&
                         V41MapFileRange(scaleInfo.fileName, scaleInfo.offset, scaleInfo.bytes,
                                         table->mmapScale, table->mmapScaleLen, table->scale);
                table->fileMapped = mapped;
            }
            if (!mapped) {
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
        // mmap 模式下这一步把缺页代价挪到后台线程，多大的批都值得做；
        // 常驻模式下靠的是 cache/TLB 命中，一旦要摸的数据超过末级缓存，等真正查表时
        // 早就被挤出去了，白白多跑一遍内存带宽——所以给一个预算，超了就只算行号不摸表。
        void V41TouchEngramRows(const V41EngramTable &table, const std::vector<int64_t> &rows) {
            const uint64_t budget = 32ULL << 20;
            if (!table.fileMapped && rows.size() * (uint64_t)table.dim > budget) {
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
        const V41EngramTable &table = *std::static_pointer_cast<V41EngramTable>(engramTables[engramLayerIndex]);
        const int cols = (int)(rows.size() / std::max(1, tokens));
        const int dim = table.dim;
        const int scaleCols = dim / table.scaleBlock;
        double prepStart = V41Profiler().level > 0 ? V41NowMs() : 0.0;
        output.ToDevice(DataDevice::CPU);
        output = Data(DataType::BFLOAT16, {1, tokens, cols * dim});
        output.Allocate(false);
        if (prepMs != nullptr) {
            *prepMs = V41NowMs() - prepStart;
        }
        uint16_t *dst = (uint16_t*)output.cpuData;
        static const FP8E4M3ToFP32Manager fp8;

        const std::function<void(int, int)> worker = [&](int st, int end) {
            for (int t = st; t < end; t++) {
                for (int c = 0; c < cols; c++) {
                    int64_t row = rows[(size_t)t * cols + c];
                    AssertInFastLLM(row >= 0 && row < table.rows, "DeepSeekV41: engram hash out of range.");
                    const uint8_t *src = table.data + (uint64_t)row * dim;
                    const uint8_t *sc = table.scale + (uint64_t)row * scaleCols;
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
                        "`python -m ftllm.deepseek_v41_engram <model_dir>` or set FASTLLM_DSV41_ENGRAM_META.");
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
        if (tp) {
            PrepareMultiCudaReplicatedData(gathered, tpDevices, true);
        }
#endif
        Linear(gathered, weight[pre + ".wkv.weight"], Data(), kv, tp);
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
                                                     const std::vector<int> *imageMask) {
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

    // ==================== 单 token decode 的 CUDA Graph ====================
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
    // 一份图（V4 是每个请求一份）。并发前向用 try_lock 抢工作区，抢不到就走逐算子。
    struct DeepSeekV41DecodeWorkspace {
        Data hiddenStates, hiddenTemp, preMix;
        int preMixSeqlen = -1;      // preMix 只在长度变化时重建（内容是常量 one-hot）
        Data attnPre, attnPost, attnComb, ffnPre, ffnPost, ffnComb;
        Data x, attnInput, qr, qNorm, q, kv, attnOut, woAOut, attnProj;
        Data attnInputFloat, rawKVAll, rawScoreAll, qIdxAll, idxWeights, idxWeightsAll;
        Data ffnInput, ffnOut, gateInput, gateLogits, expertIndex, expertScore;
        Data sharedGateup, sharedSwiglu, sharedExpertOut;
        Data w1, w2, w3, tempInput, tempOutput, moeInputTemp, moeOutputTemp;
        Data cpuMoeInput, cpuMoeIndex, cpuMoeScore;
    };

    // 每层的分段数：pre / post / route / sharedExpert
    static constexpr int kV41GraphSegmentsPerLayer = 4;

    // 回放时被跳过的 host 侧形状变更。捕获时记下，回放时原样补上。
    struct DeepSeekV41GraphSegmentMeta {
        std::vector<int> qDims, kvDims, qIdxDims;
        std::vector<int> ffnDims, ffnInputDims, expertIndexDims, expertScoreDims;
        bool swapHidden = false;
    };

#ifdef USE_CUDA
    namespace {
        // 开关：FASTLLM_DSV41_CUDA_GRAPH=1/0 显式开关；未设置时跟随全局 FASTLLM_CUDA_GRAPH。
        bool V41DecodeCudaGraphEnabled() {
            static const int mode = []() -> int {
                const char *env = std::getenv("FASTLLM_DSV41_CUDA_GRAPH");
                if (env == nullptr || env[0] == '\0') {
                    return -1;
                }
                return strcmp(env, "0") == 0 ? 0 : 1;
            }();
            if (mode >= 0) {
                return mode != 0;
            }
            return GetFastllmEnv().cudaGraph;
        }

        int V41DecodeCudaGraphWarmupRounds() {
            static const int rounds = []() -> int {
                const char *env = std::getenv("FASTLLM_DSV41_CUDA_GRAPH_WARMUP");
                int v = env != nullptr && env[0] != '\0' ? atoi(env) : 0;
                return v > 0 ? v : 2;
            }();
            return rounds;
        }

        // 排查用：位 0 = 回放 pre 段，位 1 = 回放 post 段，其余走逐算子（默认 3 全开）
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

        // 排查用：每 N 次回放强制判定一次"地址搬家"，验证失效 -> 重捕获这条路径
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
            // 这一段本来就进不了图（例如专家数超过 CUDA 路由 kernel 的上限、
            // 只能走 CPU 参考路由），占位保持段号对齐，捕获与回放时都逐算子执行。
            bool passthrough = false;
        };

        struct DeepSeekV41CudaGraphState {
            std::mutex mutex;
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
            std::unique_ptr<DeepSeekV41DecodeWorkspace> workspace;

            DeepSeekV41CudaGraphState() : workspace(new DeepSeekV41DecodeWorkspace()) {}

            void DestroyCapturedGraph() {
                const int oriDevice = FastllmCudaGetDevice();
                for (auto &segment : segments) {
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

        // 回放一段：单卡直接 Launch；TP 下每个 rank 一张图，必须由各自的常驻 worker
        // 线程同时发射（图里含 rank 间的集合通信，只发射 root 会把其它 rank 挂死）。
        bool V41LaunchGraphSegment(DeepSeekV41CudaGraphState &state, int index) {
            if (index < 0 || index >= (int)state.segments.size() || state.deviceStates.empty()) {
                return false;
            }
            DeepSeekV41GraphSegment &segment = state.segments[index];
            if (segment.execs.size() != state.deviceStates.size()) {
                return false;
            }
            if (!state.tensorParallel) {
                FastllmCudaSetDevice(state.deviceStates[0].device);
                return segment.execs[0] != nullptr && FastllmCudaGraphLaunch(segment.execs[0]);
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
                             segment.execs[rank] != nullptr;
                launchReady.fetch_add(1, std::memory_order_release);
                while (launchReady.load(std::memory_order_acquire) < launchCount) {
                    std::this_thread::yield();
                }
                if (ready) {
                    FastllmCudaSetDevice(device);
                    launchOk[rank] = FastllmCudaGraphLaunch(segment.execs[rank]) ? 1 : 0;
                }
            };
            bool previousAsync = MultiCudaSetPersistentAsyncDispatch(true);
            bool launched = MultiCudaRunDeviceCallbacks(devices, launchOne);
            MultiCudaSetPersistentAsyncDispatch(previousAsync);
            return launched && std::all_of(launchOk.begin(), launchOk.end(),
                                           [](int v) { return v != 0; });
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

    std::vector<int> DeepSeekV41Model::ForwardSegments(std::vector<DeepSeekV41Segment> &segments,
                                                       const Data &inputIds,
                                                       const Data *inputEmbeds,
                                                       const std::vector<int> *imageMask,
                                                       const std::vector<GenerationConfig> &generationConfigsIn,
                                                       const LastTokensManager &lastTokens,
                                                       std::vector<std::vector<float>*> *retLogits,
                                                       std::vector<std::pair<Data*, Data*> > &samplingPastKeyValues) {
        const int numSegments = (int)segments.size();
        AssertInFastLLM(numSegments >= 1 && (int)generationConfigsIn.size() == numSegments,
                        "DeepSeekV41Model::ForwardSegments: bad segments.");
        int total = 0;
        for (auto &seg : segments) {
            AssertInFastLLM(seg.state && seg.seqlen > 0 && seg.offset == total && seg.state->totalLen == seg.startPos,
                            "DeepSeekV41Model::ForwardSegments: inconsistent segment.");
            total += seg.seqlen;
        }
        const bool single = numSegments == 1;
        const int seqlen = total;   // 拼接后的 token 总数
        // 图像 token（掩码按拼接后的全局下标）：Engram 置 -1，专家选择改用 gate.bias_vl
        AssertInFastLLM(imageMask == nullptr || (int)imageMask->size() == seqlen,
                        "DeepSeekV41Model::ForwardSegments: imageMask length mismatch.");
        bool hasImageTokens = false;
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
        const bool fp8KV = this->kvCacheDataType == DataType::FP8_E4M3;
        const bool fp4KV = this->kvCacheDataType == DataType::FP4_E2M1;
        const bool quantKV = fp8KV || fp4KV;
        // (quantMode, quantBlock)：压缩 KV 与 indexer key
        const int cmpQuantMode = fp4KV ? 3 : 1, cmpQuantBlock = fp4KV ? 16 : 32;
        const int idxQuantMode = fp4KV ? 2 : 1, idxQuantBlock = 32;

        // ---- 张量并行 ----
        // deviceMap 为 multicuda 且不止一张卡时启用：q / attn_sink / 注意力输出 / wo_a
        // 按 query head 切分，wo_b 与共享专家 down 按列切并 all-reduce，其余全部复制。
        // FastllmGetMulticudaDeviceAndRatio 读的是 ApplyDeviceMap 设置的全局设备表，
        // 首次前向前必须先应用一次主 device map，否则拿到的是空列表。
        ApplyDeviceMap(this->deviceMap, 1, block_cnt);
        const std::vector<int> tpDevices = V41TpDevices(this->deviceMap);
        const bool tp = !tpDevices.empty();
        // 注意力能否按 head 切分：CUDA 稀疏注意力 kernel 每个 block 处理 32 个 head，
        // 要求每张卡分到的 head 数是 32 的倍数；wo_a 又要求 head 区间对齐到 o_group。
        // 不满足时（例如 32 头的迷你模型、或 64 头开 TP=4）注意力退回"每卡各算一份"，
        // 稠密 FFN / head 仍然切分。
        const int tpRanks = (int)tpDevices.size();
        const bool tpHeadsAligned =
            tpRanks > 0 && o_groups > 0 &&
            num_attention_heads % tpRanks == 0 &&
            (num_attention_heads / tpRanks) % 32 == 0 &&
            num_attention_heads % o_groups == 0 &&
            (num_attention_heads / tpRanks) % (num_attention_heads / o_groups) == 0;
        // InitParams 已经在 head 数切不开时把 device map 改回单卡（那时 tp 为假），
        // 这里的 tpHeadsAligned 只是同一条约束的兜底。
        // 排查用开关：关掉注意力 head 切分 / 共享专家切分后，全部算子退化为"两卡各算一份"，
        // 可以把数值问题定位到切分路径还是复制路径。
        const bool tpAttention = tp && tpHeadsAligned &&
                                 !V41EnvFlag("FASTLLM_DSV41_DISABLE_TP_ATTENTION");
        const bool tpSharedExpert = tp && !V41EnvFlag("FASTLLM_DSV41_DISABLE_TP_SHARED_EXPERT");
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

        // ---- 单 token decode 的 CUDA Graph ----
        // 只在"单请求 / 单 token / 已有缓存 / 纯文本 / 非 DSpark 校验"时启用。
        // 图里没有任何位置相关的东西，所以不需要按上下文长度重新捕获。
        DeepSeekV41DecodeWorkspace localWorkspace;
        DeepSeekV41DecodeWorkspace *ws = &localWorkspace;
#ifdef USE_CUDA
        std::shared_ptr<DeepSeekV41CudaGraphState> graphState;
        std::unique_lock<std::mutex> graphLock;
        bool graphActive = false;      // 本次前向使用常驻工作区（预热 / 捕获 / 回放）
        bool graphReplay = false;      // 本次前向回放已捕获的图
        bool graphCapture = false;     // 本次前向做捕获
        // 张量 dump 与逐算子同步都会在捕获中插入 host 侧拷贝 / 同步，直接关掉图。
        if (single && seqlen == 1 && segments[0].startPos > 0 && segments[0].spec == nullptr &&
            inputEmbeds == nullptr && !hasImageTokens &&
            std::getenv("FASTLLM_DSV41_DUMP_DIR") == nullptr &&
            !GetFastllmEnv().cudaSync && !GetFastllmEnv().printProfile &&
            (this->deviceMap.empty() || V41DeviceMapUsesCuda(this->deviceMap)) &&
            // 按层切分（非 multicuda 的多卡 device map）默认不启用，见 V41DeviceMapCudaDeviceCount；
            // FASTLLM_DSV41_CUDA_GRAPH_ALLOW_PIPELINE=1 可以强行打开（排查 / 评估用）
            (V41DeviceMapUsesMultiCuda(this->deviceMap) ||
             V41DeviceMapCudaDeviceCount(this->deviceMap) <= 1 ||
             V41EnvFlag("FASTLLM_DSV41_CUDA_GRAPH_ALLOW_PIPELINE")) &&
            V41DecodeCudaGraphEnabled()) {
            graphState = V41GetCudaGraphState(this->v41CudaGraphSlot);
            graphLock = std::unique_lock<std::mutex>(graphState->mutex, std::try_to_lock);
            if (!graphLock.owns_lock() || graphState->disabled) {
                graphState.reset();
            } else {
                // 设备布局 / dtype 变化时丢弃旧图（图里烤死了每张卡上的权重地址）
                std::string signature = std::to_string(block_cnt) + "|" + std::to_string((int)tp) +
                                        "|" + std::to_string((int)tpAttention) +
                                        "|" + std::to_string((int)tpSharedExpert) +
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
                    } else if (graphState->warmupRounds >= V41DecodeCudaGraphWarmupRounds()) {
                        graphCapture = true;
                    }
                }
            }
        }
        // 捕获期间 multicuda 算子必须复用常驻 worker stream，否则算子边界上的
        // device synchronize 会让捕获失效。
        bool graphPreviousAsyncDispatch = false;
        bool graphAsyncDispatchChanged = false;
        bool graphPoolOpen = false;
        if (graphCapture && graphState && graphState->tensorParallel) {
            graphPreviousAsyncDispatch = MultiCudaSetPersistentAsyncDispatch(true);
            graphAsyncDispatchChanged = true;
        }
        auto graphGiveUp = [&](const char *stage) {
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
        };
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
                graphGiveUp("prepare capture");
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

        const int dim = embed_dim;
        const int headDim = head_dim_full;
        const float softmaxScale = 1.0f / std::sqrt((float)headDim);
        const int indexTopK = index_topk;
        V41RopeParams windowRope = {qk_rope_head_dim, rope_base, 0, rope_factor,
                                    rope_scaling_beta_fast, rope_scaling_beta_slow};
        V41RopeParams compressRope = {qk_rope_head_dim, compress_rope_theta,
                                      (int)rope_scaling_original_max_position_embeddings, rope_factor,
                                      rope_scaling_beta_fast, rope_scaling_beta_slow};

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

        // ---- embedding -> hc 份 ----
        // 图模式下这些张量来自常驻工作区，地址必须在两次 decode 之间保持不变。
        Data &hiddenStates = ws->hiddenStates;
        Data &hiddenTemp = ws->hiddenTemp;
        {
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
        Data *curHidden = &hiddenStates;
        Data *nextHidden = &hiddenTemp;
        const bool dumpDebug = single && std::getenv("FASTLLM_DSV41_DUMP_DIR") != nullptr;
        const int startPos0 = segments[0].startPos;
        const std::string dumpSuffix = startPos0 == 0 ? std::string("") : "_p" + std::to_string(startPos0);
        if (dumpDebug) {
            V41DumpTensor(hiddenStates, "fl_embed" + dumpSuffix);
        }

        Data &preMix = ws->preMix;
        // 内容是常量 one-hot，长度没变就不重建：CopyFrom 会先把它搬回 CPU 再搬上卡，
        // 每次换一个设备地址，图就废了。
        if (ws->preMixSeqlen != seqlen) {
            std::vector<float> values((uint64_t)seqlen * hc_mult, 0.0f);
            for (int i = 0; i < seqlen; i++) {
                values[(uint64_t)i * hc_mult] = 1.0f;
            }
            preMix.CopyFrom(Data(DataType::FLOAT32, {1, seqlen, hc_mult}, values));
            ws->preMixSeqlen = seqlen;
        }

        // 片段切片：单片段时直接使用整批张量，多片段时按 offset 拷贝出来
        auto sliceOf = [&](Data &full, int i, Data &tmp) -> Data* {
            if (single) {
                return &full;
            }
            Split(full, 1, segments[i].offset, segments[i].offset + segments[i].seqlen, tmp);
            return &tmp;
        };
        // 按 axis=1 拼接多个片段（两块临时缓冲交替使用，避免输出与输入别名）
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

        // 本次前向内跨层共享的 indexer 结果（按片段）
        std::vector<Data> segTopK(numSegments), segCandidate(numSegments);
        std::vector<char> segHasCandidates(numSegments, 0);

        Data &attnPre = ws->attnPre, &attnPost = ws->attnPost, &attnComb = ws->attnComb;
        Data &ffnPre = ws->ffnPre, &ffnPost = ws->ffnPost, &ffnComb = ws->ffnComb;
        // 上一子层产出的 pre 系数。原来用 preMix.CopyFrom(ffnPre) 传递，
        // 但复制布局下 CopyFrom 只会拷贝可能已失效的 root，这里改成指针传递
        // （ffnPre 在被下一层 attention 消费之后才会被覆盖）。
        Data *preMixPtr = &preMix;
        Data &x = ws->x, &attnInput = ws->attnInput, &qr = ws->qr, &qNorm = ws->qNorm;
        Data &q = ws->q, &kv = ws->kv, &attnOut = ws->attnOut;
        Data &woAOut = ws->woAOut, &attnProj = ws->attnProj;
        Data &ffnInput = ws->ffnInput, &ffnOut = ws->ffnOut;
        Data &expertIndex = ws->expertIndex, &expertScore = ws->expertScore;
        Data &w1 = ws->w1, &w2 = ws->w2, &w3 = ws->w3;
        Data &tempInput = ws->tempInput, &tempOutput = ws->tempOutput;
        Data &moeInputTemp = ws->moeInputTemp, &moeOutputTemp = ws->moeOutputTemp;
        // pre 段产出、注意力核心逐算子消费的张量：也必须常驻（各层形状相同可以共用，
        // 只有写它的那一层会读它）
        Data &rawKVAll = ws->rawKVAll, &rawScoreAll = ws->rawScoreAll;
        Data &qIdxAll = ws->qIdxAll, &idxWeightsAll = ws->idxWeightsAll;
        // TP + cpu/numa MoE 时，从卡上副本拷出来的路由专家输入（复用同一组缓冲，
        // 避免每层重新分配、也避免栈上 Data 的地址被下游缓存复用）
        Data &cpuMoeInput = ws->cpuMoeInput, &cpuMoeIndex = ws->cpuMoeIndex, &cpuMoeScore = ws->cpuMoeScore;
        std::vector<Data> segQ(numSegments), segKV(numSegments), segAttnOut(numSegments);
        Data catTmp[2];

#ifdef USE_CUDA
        // 工作区里所有跨段传递的张量：图里烤死的就是它们的设备地址。
        auto graphCollectBoundaryPointers = [&](std::vector<const void*> &out) {
            out.clear();
            const Data *tensors[] = {
                &hiddenStates, &hiddenTemp, &preMix,
                &attnPre, &attnPost, &attnComb, &ffnPre, &ffnPost, &ffnComb,
                &x, &attnInput, &qr, &qNorm, &q, &kv, &attnOut, &woAOut, &attnProj,
                &rawKVAll, &rawScoreAll, &qIdxAll, &idxWeightsAll,
                &ffnInput, &expertIndex, &expertScore, &ws->sharedExpertOut};
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
        };
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
            graphCollectBoundaryPointers(current);
            const int invalidateEvery = V41DecodeCudaGraphInvalidateEvery();
            const bool forceInvalidate = invalidateEvery > 0 &&
                                         ++graphState->replayCount % invalidateEvery == 0;
            if (forceInvalidate || current != graphState->boundaryPointers) {
                graphState->DestroyCapturedGraph();
                graphState->recaptureCount++;
                graphReplay = false;
                if (graphState->recaptureCount > 3 && !forceInvalidate) {
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
        }
#endif

        // CUDA 路由能不能进图：FastllmCudaDeepSeekV4RouteScoreTransform 的 kernel 只支持
        // <= 256 个专家（真实模型是 384），超了就整段走 CPU 参考路由，里面有同步 D2H。
        // FASTLLM_DSV41_CUDA_GRAPH_FORCE_ROUTE_CAPTURE=1：排查用，强行把不可捕获的路由
        // 也当成可捕获，用来验证"捕获中撞上非法的同步 D2H 也只是丢图回退、不会崩"。
        const bool cudaRouteCapturable =
            V41EnvFlag("FASTLLM_DSV41_CUDA_GRAPH_FORCE_ROUTE_CAPTURE") ||
            (!hasImageTokens && num_experts <= 256 &&
             !V41EnvFlag("FASTLLM_DSV41_DISABLE_CUDA_ROUTE"));

        // 分段调度：捕获 / 回放 / 逐算子。record 保存回放时被跳过的 host 侧形状变更，
        // restore 在回放时把它们补回来。任何一步失败都就地退回逐算子并永久关掉图。
        auto V41RunGraphSegment = [&](const std::function<void()> &body, int index, bool capturable,
                                      const std::function<void(DeepSeekV41GraphSegmentMeta&)> &record,
                                      const std::function<void(const DeepSeekV41GraphSegmentMeta&)> &restore) {
#ifdef USE_CUDA
            if (graphReplay) {
                if ((V41DecodeCudaGraphReplayMask() & (1 << (index % kV41GraphSegmentsPerLayer))) == 0 ||
                    (index < (int)graphState->segments.size() &&
                     graphState->segments[index].passthrough)) {
                    body();
                    return;
                }
                if (index < (int)graphState->segments.size() &&
                    V41LaunchGraphSegment(*graphState, index)) {
                    restore(graphState->segments[index]);
                    return;
                }
                graphGiveUp("replay");
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
                graphGiveUp("capture");
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
        };

        for (int layer = 0; layer < block_cnt; layer++) {
            ApplyDeviceMap(this->deviceMap, layer + 1, block_cnt);
            std::string pre = "layers." + std::to_string(layer);
            const int ratio = compress_ratios[layer];
            const V41RopeParams &rope = ratio > 0 ? compressRope : windowRope;

            // ---- Engram ----
            for (size_t l = 0; l < engram_layer_ids.size(); l++) {
                if (engram_layer_ids[l] == layer) {
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
                    Data uniform;
                    std::vector<float> uniformValues((uint64_t)segments[s].seqlen * hc_mult,
                                                     1.0f / (float)hc_mult);
                    uniform.CopyFrom(Data(DataType::FLOAT32, {1, segments[s].seqlen, hc_mult}, uniformValues));
                    V41HcApplyPre(*src, uniform, spec->mainHidden[slot]);
                }
            }

            // ---- attention（整批部分）----
            // 【CUDA Graph 的 pre 段】这里到 RoPE 之前全部与 token 位置无关，整段可捕获。
            bool needIndexer = ratio > 0 && isIndexSource[layer];
            auto runAttentionPre = [&]() {
            V41HcMix(*curHidden, weight[pre + ".hc_attn_fn"], weight[pre + ".hc_attn_scale"],
                     weight[pre + ".hc_attn_base"], hc_mult, hc_sinkhorn_iters, hc_eps, rms_norm_eps,
                     attnPre, attnPost, attnComb);
            V41HcApplyPreNorm(*curHidden, *preMixPtr, weight[pre + ".attn_norm.weight"], rms_norm_eps,
                              x, attnInput);

            // wq_a / wkv 是复制的（KV 是 MLA 式的单份 latent，与 head 无关）；
            // wq_b 按行切 -> q 的 head 维分片，Reshape 会把 tpAxis 从最后一维换算到 head 维。
            Linear(attnInput, weight[pre + ".attn.wq_a.weight"], Data(), qr, tp);
            V41RMSNormBF16(qr, weight[pre + ".attn.q_norm.weight"], rms_norm_eps, qNorm);
            if (tpAttention) {
                weight[pre + ".attn.wq_b.weight"].tpLinearType = TP_LINEAR_ROW;
            }
            // 不切分注意力时 wq_b 也必须显式走复制布局：否则 MultiCudaLinearOp 会按
            // "大权重通用切分 + gather"处理，而那条路径读的是复制张量已失效的 root。
            Linear(qNorm, weight[pre + ".attn.wq_b.weight"], Data(), q, tp && !tpAttention);
            q.Reshape({1, seqlen, num_attention_heads, headDim});

            Linear(attnInput, weight[pre + ".attn.wkv.weight"], Data(), kv, tp);
            V41RMSNormBF16(kv, weight[pre + ".attn.kv_norm.weight"], rms_norm_eps, kv);
            kv.Reshape({1, seqlen, headDim});

            // kv source 层：compressor 的投影按整批算，分组 / 追加按片段做
            // （rawKVAll / rawScoreAll / qIdxAll / idxWeightsAll 见上面的工作区绑定）
            if (ratio > 0 && isKvSource[layer]) {
                std::string cpre = pre + ".attn.compressor";
                Data &xFloat = ws->attnInputFloat;
                ToDataType(attnInput, xFloat, DataType::FLOAT32);
                Linear(xFloat, weight[cpre + ".wkv.weight"], Data(), rawKVAll, tp);
                ToDataType(rawKVAll, DataType::FLOAT32);
                if (ratio > 1) {
                    Linear(xFloat, weight[cpre + ".wgate.weight"], Data(), rawScoreAll, tp);
                    ToDataType(rawScoreAll, DataType::FLOAT32);
                }
            }
            if (ratio > 0 && isIndexSource[layer]) {
                std::string ipre = pre + ".attn.indexer";
                // indexer 在每张卡上各算一份：它选出的候选块要供后续所有层复用，
                // 切 index head 就得对 [token, m] 的分数矩阵做 all-reduce，
                // 通信量远大于重复计算，而且两卡 top-k 必须逐位一致。
                Linear(qNorm, weight[ipre + ".wq_b.weight"], Data(), qIdxAll, tp);
                qIdxAll.Reshape({1, seqlen, index_n_heads, index_head_dim});
                Data &idxWeights = ws->idxWeights;
                Linear(attnInput, weight[ipre + ".weights_proj.weight"], Data(), idxWeights, tp);
                ToDataType(idxWeights, DataType::FLOAT32);
                Mul(idxWeights, (1.0f / std::sqrt((float)index_head_dim)) * (1.0f / std::sqrt((float)index_n_heads)),
                    idxWeightsAll);
            }
            };   // runAttentionPre
            V41RunGraphSegment(runAttentionPre, kV41GraphSegmentsPerLayer * layer, true,
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

            // ---- attention（按片段）----
            for (int s = 0; s < numSegments; s++) {
                DeepSeekV41Segment &seg = segments[s];
                DeepSeekV41LayerCache &cache = seg.state->layers[layer];
                const int startPos = seg.startPos;
                const int segLen = seg.seqlen;

                Data *qSeg = sliceOf(q, s, segQ[s]);
                V41RotaryQuant(*qSeg, rope, startPos, 1, false, 0, 32);
                Data *kvSeg = sliceOf(kv, s, segKV[s]);
                V41RotaryQuant(*kvSeg, rope, startPos, 1, false, 1, 32);

                Data *compressedKV = nullptr;
                Data *cmpIdx = nullptr;
                if (ratio > 0) {
                    const int src = kvSourceOf[layer];
                    DeepSeekV41LayerCache &srcCache = seg.state->layers[src];
                    if (isKvSource[layer]) {
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
                                Linear(latent, weight[ipre + ".wk.weight"], Data(), kIdx, tp);
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
                            if (chunkTokens >= segLen) {
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
                        if (segTopK[s].dims.size() == 3) {
                            cmpIdx = &segTopK[s];
                        }
                    }
                }

                Data *attnOutSeg = single ? &attnOut : &segAttnOut[s];
                V41SparseAttention(*qSeg, *kvSeg, startPos > 0 ? &cache.windowKV : nullptr, compressedKV, cmpIdx,
                                   weight[pre + ".attn.attn_sink"], window_size, startPos, softmaxScale, *attnOutSeg);
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
                V41RotaryQuant(*attnOutSeg, rope, startPos, 1, true, 0, 32);
                cache.totalLen += segLen;
            }

            Data *attnOutAll = single ? &attnOut : catSegments(segAttnOut, catTmp);
            // 【CUDA Graph 的 post 段】wo_a 到共享专家之间同样与 token 位置无关。
            // MergeMOEBlock（可能在 CPU / NUMA 上）与其后的 hc 残差留在段外。
            std::vector<Data*> moeWeights = weights[layer];
            bool hasSharedExpertOut = false;
            auto sharedGateupIt = weight.weight.find(pre + ".ffn.shared_experts.gateup.weight");
            auto sharedDownIt = weight.weight.find(pre + ".ffn.shared_experts.w2.weight");
            if (GetCudaSharedExpert() && sharedGateupIt != weight.weight.end() &&
                sharedDownIt != weight.weight.end() && !sharedGateupIt->second.isDiskWeight &&
                !sharedDownIt->second.isDiskWeight) {
                moeWeights[0] = moeWeights[1] = nullptr;
                hasSharedExpertOut = true;
            }
            Data &sharedExpertOut = ws->sharedExpertOut;
            std::vector<int> ffnDims;
            auto runAttentionPost = [&]() {
            if (tpAttention) {
                // wo_a 按 head 组切（与 q 的分片一致），wo_b 按列切；
                // wo_b 的输入是分片的，MultiCudaLinearOp 会自动走 column + all-reduce。
                weight[pre + ".attn.wo_a.weight"].tpLinearType = TP_LINEAR_ROW;
                weight[pre + ".attn.wo_b.weight"].tpLinearType = TP_LINEAR_COLUMN;
            }
            DeepSeekV4WoA(*attnOutAll, weight[pre + ".attn.wo_a.weight"], o_groups, o_lora_rank, woAOut);
            // 切分时 woAOut 是分片的，MultiCudaLinearOp 自动走 column + all-reduce；
            // 不切分时 woAOut 是复制的，必须显式要求复制布局，否则会退回单卡 CUDA
            // 读到已经失效的 root。
            Linear(woAOut, weight[pre + ".attn.wo_b.weight"], Data(), attnProj, tp && !tpAttention);
            if (dumpDebug) {
                V41DumpTensor(attnInput, "fl_layer" + std::to_string(layer) + "_attn_in" + dumpSuffix);
                V41DumpTensor(*attnOutAll, "fl_layer" + std::to_string(layer) + "_attn_o" + dumpSuffix);
                V41DumpTensor(attnProj, "fl_layer" + std::to_string(layer) + "_attn" + dumpSuffix);
                if (segTopK[0].dims.size() == 3) {
                    V41DumpTensor(segTopK[0], "fl_layer" + std::to_string(layer) + "_topk" + dumpSuffix);
                }
            }
            DeepSeekV4HcPost(attnProj, *curHidden, attnPost, attnComb, *nextHidden);
            std::swap(curHidden, nextHidden);
            if (dumpDebug) {
                V41DumpTensor(*curHidden, "fl_layer" + std::to_string(layer) + "_hidden_attn" + dumpSuffix);
            }

            // ---- FFN (MoE)：整批 ----
            V41HcMix(*curHidden, weight[pre + ".hc_ffn_fn"], weight[pre + ".hc_ffn_scale"],
                     weight[pre + ".hc_ffn_base"], hc_mult, hc_sinkhorn_iters, hc_eps, rms_norm_eps,
                     ffnPre, ffnPost, ffnComb);
            V41HcApplyPreNorm(*curHidden, attnPre, weight[pre + ".ffn_norm.weight"], rms_norm_eps,
                              x, ffnInput);
            ffnDims = ffnInput.dims;
            ffnInput.Reshape({seqlen, dim});
            };   // runAttentionPost
            V41RunGraphSegment(runAttentionPost, kV41GraphSegmentsPerLayer * layer + 1, true,
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
                ToDataType(ffnInput, xFloat, DataType::FLOAT32);
                Linear(xFloat, weight[gpre + ".weight"], Data(), logits, tp);
                ToDataType(logits, DataType::FLOAT32);
                if (std::fabs(gate_temp - 1.0f) > 1e-6f) {
                    Mul(logits, 1.0f / gate_temp, logits);
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
                if (!hasImageTokens &&
                    (logits.dataDevice == DataDevice::CUDA || (tp && logits.multiDeviceData)) &&
                    !V41EnvFlag("FASTLLM_DSV41_DISABLE_CUDA_ROUTE") &&
                    V41RouteScoreTransformTp(logits, 2, tpDevices)) {
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
                            original[e] = std::sqrt(V41Softplus(raw[(uint64_t)t * num_experts + e]));
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
            // 专家数超过 CUDA 路由 kernel 的 256 上限时（真实模型是 384），路由整段只能
            // 走 CPU 参考实现：里面的 logits.ToDevice(CPU) 是同步 D2H，捕获期间非法。
            // 这种情况下把这一段标成占位段逐算子执行，而不是让捕获在半路炸掉。
            V41RunGraphSegment(runRoute, kV41GraphSegmentsPerLayer * layer + 2, cudaRouteCapturable,
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

            auto runSharedExpert = [&]() {
            if (hasSharedExpertOut) {
                if (tpSharedExpert) {
                    sharedGateupIt->second.tpLinearType = TP_LINEAR_ROW;
                    sharedGateupIt->second.tpPackType = TP_PACK_GATEUP;
                    sharedDownIt->second.tpLinearType = TP_LINEAR_COLUMN;
                }
                Data &ww1 = ws->sharedSwiglu, &ww3 = ws->sharedGateup;
                LinearSwigluBlock(&ffnInput, &sharedGateupIt->second, GetEmptyData(), &ww3, &ww1);
                Linear(ww1, sharedDownIt->second, *GetEmptyData(), sharedExpertOut);
            }
            };   // runSharedExpert
            V41RunGraphSegment(runSharedExpert, kV41GraphSegmentsPerLayer * layer + 3, hasSharedExpertOut,
                [](DeepSeekV41GraphSegmentMeta &) {},
                [](const DeepSeekV41GraphSegmentMeta &) {});

            {
                this->ApplyMoeDeviceMapForLayer(layer);
                // 路由专家在 multicuda 上按专家并行（每卡一部分专家 + all-reduce），
                // 在 cpu / numa 上仍然是单份计算，结果随后广播回两张卡。
                const std::string moeDeviceSpec = this->SelectMoeDeviceForLayer(layer);
                const bool routedExpertParallel = V41DeviceSpecUsesType(moeDeviceSpec, "multicuda");
                const bool routedExpertOnCuda = routedExpertParallel ||
                                                V41DeviceSpecUsesType(moeDeviceSpec, "cuda");
                Data *moeInputPtr = &ffnInput;
                Data *moeIndexPtr = &expertIndex;
                Data *moeScorePtr = &expertScore;
                // MoE 落在 cpu / numa 时，输入必须从某张卡的副本拷出来：直接交给 CPU 算子
                // 会让 Data::ToDevice 从复制布局已经失效的 root 上读，直接段错误。
                const bool tpStageMoe = tp && !routedExpertParallel;
                bool stageMoeInput = tpStageMoe;
                bool stageMoeRoute = tpStageMoe;
#ifdef USE_CUDA
                // 图模式下 ffnInput / expertIndex / expertScore 是 post 段的输出，地址被烤
                // 进图里；而 DoCudaMergeMOE 一定会把 index / score 搬到 CPU 上分桶
                // （cpu / numa 上的 MoE 还会连输入一起搬走），Data::ToDevice 顺手就把显存
                // 释放了，下一次回放于是写到别人的缓冲上。所以先拷到独立的暂存缓冲再交出去。
                stageMoeRoute = stageMoeRoute || graphActive;
                stageMoeInput = stageMoeInput || (graphActive && !routedExpertOnCuda);
#endif
                if (stageMoeInput) {
                    V41ReplicaToCpu(cpuMoeInput, ffnInput, tpDevices);
                    moeInputPtr = &cpuMoeInput;
                }
                if (stageMoeRoute) {
                    if (routedExpertParallel) {
                        // 专家并行要的是每卡一份的复制布局，不能拉到 CPU
                        V41Assign(cpuMoeIndex, expertIndex, tpDevices);
                        V41Assign(cpuMoeScore, expertScore, tpDevices);
                    } else {
                        V41ReplicaToCpu(cpuMoeIndex, expertIndex, tpDevices);
                        V41ReplicaToCpu(cpuMoeScore, expertScore, tpDevices);
                    }
                    moeIndexPtr = &cpuMoeIndex;
                    moeScorePtr = &cpuMoeScore;
                }
                MergeMOEBlock(moeInputPtr, moeIndexPtr, moeScorePtr, &moeWeights, &biass[layer],
                              &w1, &w2, &w3, &tempInput, &tempOutput, 1.0f, &ffnOut, layer,
                              ffnInput.dataType, ffnInput.dataType, &moeInputTemp, &moeOutputTemp,
                              MoeGateSwiglu, routedExpertParallel, swiglu_limit, true);
                ApplyDeviceMap(this->deviceMap, layer + 1, block_cnt);
#ifdef USE_CUDA
                if (tp && ffnOut.dataDevice == DataDevice::CPU && ffnOut.cpuData != nullptr) {
                    // CPU / NUMA 上算出的路由专家结果只有一份，广播到两张卡后才能
                    // 与复制布局的共享专家输出、hc 残差相加。
                    PrepareMultiCudaReplicatedData(ffnOut, tpDevices, true);
                }
#endif
                if (hasSharedExpertOut) {
                    if (!(tp && ffnOut.multiDeviceData && sharedExpertOut.multiDeviceData)) {
                        ffnOut.ToDevice(sharedExpertOut.dataDevice);
                    }
                    AddTo(ffnOut, sharedExpertOut);
                }
#ifdef USE_CUDA
                if (graphActive && !tp && ffnOut.dataDevice == DataDevice::CPU) {
                    // 之后的 hc 残差还要读 curHidden（图的输出，必须留在卡上），
                    // 否则 DeepSeekV4HcPost 会把整条残差流搬到 CPU 上。
                    ffnOut.ToDevice(DataDevice::CUDA);
                }
#endif
            }
            ffnOut.Reshape(ffnDims);
            if (dumpDebug) {
                V41DumpTensor(ffnInput, "fl_layer" + std::to_string(layer) + "_ffn_in" + dumpSuffix);
                V41DumpTensor(ffnOut, "fl_layer" + std::to_string(layer) + "_ffn" + dumpSuffix);
                V41DumpTensor(expertIndex, "fl_layer" + std::to_string(layer) + "_expert_idx" + dumpSuffix);
            }
            DeepSeekV4HcPost(ffnOut, *curHidden, ffnPost, ffnComb, *nextHidden);
            std::swap(curHidden, nextHidden);
            preMixPtr = &ffnPre;
            if (dumpDebug) {
                V41DumpTensor(*curHidden, "fl_layer" + std::to_string(layer) + dumpSuffix);
            }
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
                    graphState->captured = true;
                    graphCollectBoundaryPointers(graphState->boundaryPointers);
                    if (V41DecodeCudaGraphVerbose()) {
                        fprintf(stderr, "[Fastllm] DeepSeek-V4.1 decode CUDA graph captured: "
                                        "%d segments on %d device(s), tp=%d\n",
                                (int)graphState->segments.size(),
                                (int)graphState->devices.size(), (int)graphState->tensorParallel);
                        fflush(stderr);
                    }
                } else {
                    if (graphPoolOpen) {
                        FastllmCudaGraphMemoryPoolAbort();
                        graphPoolOpen = false;
                    }
                    graphGiveUp("finalize capture");
                }
            } else if (!graphReplay && !graphState->disabled) {
                graphState->warmupRounds++;
            }
            if (graphAsyncDispatchChanged) {
                MultiCudaSetPersistentAsyncDispatch(graphPreviousAsyncDispatch);
            }
        }
#endif

        // ---- DSpark 校验片段：对本片段的每个位置都出贪心 token ----
        // 只在单片段（单请求）时启用，见 ForwardSingle。要求请求是简单贪心，因此
        // 这里的 RMSNorm + head + TopK 与 LLMSamplingBlock 的 allSimple 分支等价。
        if (numSegments == 1 && segments[0].spec != nullptr && segments[0].spec->wantAllGreedy) {
            Data allHidden, normed, allLogits, topk;
            V41HcApplyPre(*curHidden, *preMixPtr, allHidden);
            RMSNorm(allHidden, weight["norm.weight"], rms_norm_eps, normed);
            // DSpark 校验分支自己做 TopK，需要完整 logits，这里让 head 走复制布局
            if (tp) {
                weight["head.weight"].tpLinearType = TP_LINEAR_NONE;
            }
            Linear(normed, weight["head.weight"], *GetEmptyData(), allLogits, tp);
            ToDataType(allLogits, DataType::FLOAT32);
            segments[0].spec->greedy.resize(seqlen);
            if (tp && allLogits.multiDeviceData && allLogits.IsTensorParallelReplicated()) {
                // TopK 没有 multicuda 实现，会退回单卡读复制布局已失效的 root；
                // 这里直接从副本拷到 CPU 上自己取 argmax。
                Data cpuLogits;
                V41ReplicaToCpu(cpuLogits, allLogits, tpDevices);
                const int vocab = cpuLogits.dims.back();
                const float *values = (const float*)cpuLogits.cpuData;
                for (int i = 0; i < seqlen; i++) {
                    const float *row = values + (uint64_t)i * vocab;
                    int best = 0;
                    for (int v = 1; v < vocab; v++) {
                        if (row[v] > row[best]) {
                            best = v;
                        }
                    }
                    segments[0].spec->greedy[i] = best;
                }
            } else {
                TopK(allLogits, topk, 1);
                topk.ToDevice(DataDevice::CPU);
                const int stride = topk.dims[topk.dims.size() - 1];
                const float *topkData = (const float*)topk.cpuData;
                for (int i = 0; i < seqlen; i++) {
                    segments[0].spec->greedy[i] = (int)(topkData[(uint64_t)i * stride] + 1e-3);
                }
            }
            for (auto &seg : segments) {
                seg.state->totalLen += seg.seqlen;
            }
            return std::vector<int>{segments[0].spec->greedy[seqlen - 1]};
        }

        // ---- head（每个片段只取最后一个 token）----
        Data headInput;
        if (single && seqlen == 1) {
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
            if (config.do_sample && config.top_k <= 1 && config.temperature > 1e-6f) {
                config.top_k = 5;
            }
        }
        // head 按行切分：两张卡各算一半词表，LLMSamplingBlock 的
        // SampleTensorParallelGreedyLogits / GatherTensorParallelLogitsToRoot 负责合并。
        // 不切分时 head 的 Linear 会走 multicuda 的"通用切分 + gather"路径，
        // 那条路径读的是复制张量已经失效的 root，结果是错的。
        if (tp) {
            weight["head.weight"].tpLinearType = TP_LINEAR_ROW;
        }
        LLMSamplingBlock(this, &headInput, &weight["norm.weight"], &weight["head.weight"],
                         rms_norm_eps, numSegments, true, samplingSeqLens, samplingPastKeyValues,
                         generationConfigs, lastTokens, retLogits, ret);

        for (auto &seg : segments) {
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
