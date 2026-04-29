//
// Created by huangyuyang on 4/24/26.
//
// DeepSeek-V4 系列模型的 fastllm 适配框架。
//
// 当前文件提供：
//   - 模型类型 / 权重前缀的注册；
//   - 全部超参解析（包括 HC、Indexer、Compress、MTP 等 V4 特有字段）；
//   - YaRN RoPE（主分支与 compress 分支）的预计算；
//   - Forward / ForwardBatch 等接口的占位实现（暂未支持完整推理，
//     直接调用会抛出未实现错误，便于后续按层逐步填充）。
//
// 完整 forward 涉及 Hyper-Connections / 稀疏 attention / hash gate / MTP，
// 这些算子在 fastllm 中尚无对应实现，将会作为后续 PR 单独提交。
//

#include "deepseekv4.h"

#include "baseblock.h"
#include "executor.h"
#include "utils.h"

#include <sstream>
#include <random>
#include <unordered_map>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <limits>
#include <cstdlib>
#include <cctype>
#include <memory>
#include <mutex>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

#include "json11.hpp"

namespace fastllm {
    // 复用 deepseekv2.cpp 中的 yarn 工具函数
    extern float yarn_find_correction_dim(int num_rotations, int dim, float base, int max_position_embeddings);
    extern void yarn_find_correction_range(int low_rot, int high_rot, int dim, float base, int max_position_embeddings, int &low, int &high);
    extern float yarn_get_mscale(float scale, float mscale);
    extern std::vector <float> yarn_linear_ramp_mask(float min, float max, int dim);

    namespace {
        static int GetIntWithFallback(const WeightMap &weight, const std::vector<std::string> &keys, int fallback) {
            for (auto &key : keys) {
                auto it = weight.dicts.find(key);
                if (it != weight.dicts.end() && !it->second.empty()) {
                    return atoi(it->second.c_str());
                }
            }
            return fallback;
        }

        static float GetFloatWithFallback(const WeightMap &weight, const std::vector<std::string> &keys, float fallback) {
            for (auto &key : keys) {
                auto it = weight.dicts.find(key);
                if (it != weight.dicts.end() && !it->second.empty()) {
                    return atof(it->second.c_str());
                }
            }
            return fallback;
        }

        static std::string GetStringWithFallback(const WeightMap &weight, const std::vector<std::string> &keys, const std::string &fallback) {
            for (auto &key : keys) {
                auto it = weight.dicts.find(key);
                if (it != weight.dicts.end() && !it->second.empty()) {
                    return it->second;
                }
            }
            return fallback;
        }

        static bool EnvFlagEnabled(const char *name) {
            const char *v = std::getenv(name);
            if (v == nullptr) {
                return false;
            }
            std::string s(v);
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
                return (char)std::tolower(c);
            });
            return !(s.empty() || s == "0" || s == "false" || s == "off" || s == "no");
        }

        static int EnvInt(const char *name, int fallback) {
            const char *v = std::getenv(name);
            return v == nullptr ? fallback : atoi(v);
        }

        static bool DeepSeekV4PreferCuda() {
#ifdef USE_CUDA
            auto *executor = (Executor*)GetExecutor();
            return executor != nullptr && executor->firstDevice == "cuda";
#else
            return false;
#endif
        }

        static std::vector<float> ReadFloatData(const Data &input);

        static double NowMs() {
            using Clock = std::chrono::steady_clock;
            return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
        }

        static Executor *GetProfilerExecutor() {
            return (Executor*)GetExecutor();
        }

        static float ExecutorProfileTotal() {
            auto *executor = GetProfilerExecutor();
            return executor == nullptr ? 0.0f : executor->GetProfilerTotal();
        }

        struct ScopedExecutorProfiler {
            std::string opType;
            double startMs = 0.0;
            float startProfile = 0.0f;
            bool active = false;

            ScopedExecutorProfiler(const std::string &opType)
                : opType(opType), startMs(NowMs()), startProfile(ExecutorProfileTotal()),
                  active(GetProfilerExecutor() != nullptr) {}

            ~ScopedExecutorProfiler() {
                auto *executor = GetProfilerExecutor();
                if (!active || executor == nullptr) {
                    return;
                }
                float elapsed = (float)((NowMs() - startMs) * 0.001);
                float alreadyProfiled = ExecutorProfileTotal() - startProfile;
                float unprofiled = elapsed - alreadyProfiled;
                if (unprofiled > 1e-7f) {
                    executor->AddProfiler(opType, unprofiled);
                }
            }
        };

        static void ProfileSyncIfNeeded(bool sync) {
#ifdef USE_CUDA
            if (sync) {
                ForceDeviceSync();
            }
#else
            (void)sync;
#endif
        }

        static uint64_t CachedWeightMaxBytes() {
            static uint64_t maxBytes = []() -> uint64_t {
                int mb = EnvInt("FASTLLM_WEIGHT_CACHE_MAX_MB", 256);
                if (mb <= 0) {
                    return 0;
                }
                return (uint64_t)mb * 1024ULL * 1024ULL;
            }();
            return maxBytes;
        }

        struct CachedFloatTensor {
            DataType dataType = DataType::FLOAT32;
            DataDevice dataDevice = DataDevice::CPU;
            uint64_t count = 0;
            uint64_t bytes = 0;
            const uint8_t *cpuData = nullptr;
            const void *cudaData = nullptr;
            std::vector<int> dims;
            std::shared_ptr<const std::vector<float>> values;
        };

        static std::mutex &CachedWeightMutex() {
            static std::mutex mutex;
            return mutex;
        }

        static std::unordered_map<const Data*, CachedFloatTensor> &CachedWeightFloats() {
            static std::unordered_map<const Data*, CachedFloatTensor> cache;
            return cache;
        }

        static bool CachedFloatTensorMatches(const CachedFloatTensor &cached, const Data &input) {
            return cached.dataType == input.dataType &&
                   cached.dataDevice == input.dataDevice &&
                   cached.count == input.Count(0) &&
                   cached.bytes == input.GetBytes() &&
                   cached.cpuData == input.cpuData &&
                   cached.cudaData == input.cudaData &&
                   cached.dims == input.dims &&
                   cached.values != nullptr;
        }

        static std::shared_ptr<const std::vector<float>> ReadWeightFloatDataCached(const Data &input) {
            uint64_t bytes = input.GetBytes();
            if (CachedWeightMaxBytes() == 0 || bytes == 0 || bytes > CachedWeightMaxBytes() || input.multiDeviceData) {
                return std::make_shared<const std::vector<float>>(ReadFloatData(input));
            }

            {
                std::lock_guard<std::mutex> guard(CachedWeightMutex());
                auto &cache = CachedWeightFloats();
                auto it = cache.find(&input);
                if (it != cache.end() && CachedFloatTensorMatches(it->second, input)) {
                    return it->second.values;
                }
            }

            auto values = std::make_shared<const std::vector<float>>(ReadFloatData(input));
            CachedFloatTensor cached;
            cached.dataType = input.dataType;
            cached.dataDevice = input.dataDevice;
            cached.count = input.Count(0);
            cached.bytes = bytes;
            cached.cpuData = input.cpuData;
            cached.cudaData = input.cudaData;
            cached.dims = input.dims;
            cached.values = values;
            {
                std::lock_guard<std::mutex> guard(CachedWeightMutex());
                CachedWeightFloats()[&input] = std::move(cached);
            }
            return values;
        }

        struct DeepSeekV4LayerProfile {
            double attnPrep = 0.0;
            double cache = 0.0;
            double sparseAttn = 0.0;
            double attnOut = 0.0;
            double route = 0.0;
            double moeMove = 0.0;
            double moe = 0.0;
            double ffnPost = 0.0;
            double attnHcPre = 0.0;
            double attnNorm = 0.0;
            double qProj = 0.0;
            double qPost = 0.0;
            double kvProjNorm = 0.0;
            double kvPost = 0.0;
            double attnWoA = 0.0;
            double attnWoB = 0.0;
            double attnHcPost = 0.0;
            double ffnHcPre = 0.0;
            double ffnNorm = 0.0;
            double routeGate = 0.0;
            double routeScore = 0.0;
        };

        static double LayerProfileTotal(const DeepSeekV4LayerProfile &p) {
            return p.attnPrep + p.cache + p.sparseAttn + p.attnOut +
                   p.route + p.moeMove + p.moe + p.ffnPost;
        }

        static uint64_t CountDims(const std::vector<int> &dims) {
            uint64_t ret = 1;
            for (int v : dims) {
                ret *= (uint64_t)v;
            }
            return ret;
        }

        static float BFloat16ToFloat(uint16_t v);

        static std::vector<float> ReadFloatData(const Data &input) {
            if (input.dataType == DataType::INT32 || input.dataType == DataType::INT32PARAM) {
                Data tmp;
                tmp.CopyFrom(input);
                tmp.ToDevice(DataDevice::CPU);
                uint64_t cnt = tmp.Count(0);
                std::vector<float> ret(cnt);
                int32_t *p = (int32_t*)tmp.cpuData;
                for (uint64_t i = 0; i < cnt; i++) {
                    ret[i] = (float)p[i];
                }
                return ret;
            }
            if (input.dataType == DataType::BFLOAT16 || input.dataType == DataType::FLOAT16) {
                Data tmp;
                tmp.CopyFrom(input);
                tmp.ToDevice(DataDevice::CPU);
                uint64_t cnt = tmp.Count(0);
                std::vector<float> ret(cnt);
                uint16_t *p = (uint16_t*)tmp.cpuData;
                if (input.dataType == DataType::BFLOAT16) {
                    for (uint64_t i = 0; i < cnt; i++) {
                        ret[i] = BFloat16ToFloat(p[i]);
                    }
                } else {
                    for (uint64_t i = 0; i < cnt; i++) {
                        ret[i] = half_to_float(p[i]);
                    }
                }
                return ret;
            }
            Data tmp;
            ToDataType(input, tmp, DataType::FLOAT32);
            tmp.ToDevice(DataDevice::CPU);
            uint64_t cnt = tmp.Count(0);
            std::vector<float> ret(cnt);
            memcpy(ret.data(), tmp.cpuData, cnt * sizeof(float));
            return ret;
        }

        static std::vector<int> ReadTokenIds(const Data &inputIds) {
            Data tmp;
            ToDataType(inputIds, tmp, DataType::FLOAT32);
            tmp.ToDevice(DataDevice::CPU);
            int cnt = (int)tmp.Count(0);
            std::vector<int> ret(cnt);
            float *p = (float*)tmp.cpuData;
            for (int i = 0; i < cnt; i++) {
                ret[i] = (int)(p[i] + 0.5f);
            }
            return ret;
        }

        static uint16_t FloatToBFloat16(float v) {
            uint32_t x;
            memcpy(&x, &v, sizeof(uint32_t));
            x += 0x7FFF + ((x >> 16) & 1);
            return (uint16_t)(x >> 16);
        }

        static float BFloat16ToFloat(uint16_t v) {
            uint32_t x = ((uint32_t)v) << 16;
            float ret;
            memcpy(&ret, &x, sizeof(float));
            return ret;
        }

        static void ResetData(Data &data) {
            data.FreeSpace();
            data = Data();
        }

        static void WriteFloatData(const std::vector<float> &values, const std::vector<int> &dims,
                                   Data &output, DataType dtype = DataType::FLOAT32) {
            Data tmp(dtype, dims);
            tmp.Allocate();
            if (dtype == DataType::FLOAT32) {
                memcpy(tmp.cpuData, values.data(), values.size() * sizeof(float));
            } else if (dtype == DataType::BFLOAT16) {
                uint16_t *dst = (uint16_t*)tmp.cpuData;
                for (size_t i = 0; i < values.size(); i++) {
                    dst[i] = FloatToBFloat16(values[i]);
                }
            } else if (dtype == DataType::FLOAT16) {
                uint16_t *dst = (uint16_t*)tmp.cpuData;
                for (size_t i = 0; i < values.size(); i++) {
                    dst[i] = float_to_half(values[i]);
                }
            } else {
                ErrorInFastLLM("DeepSeekV4Model: unsupported WriteFloatData dtype.");
            }
            ResetData(output);
            output.CopyFrom(tmp);
        }

        static void WriteIntData(const std::vector<int> &values, const std::vector<int> &dims,
                                 Data &output) {
            Data tmp(DataType::INT32, dims);
            tmp.Allocate();
            memcpy(tmp.cpuData, values.data(), values.size() * sizeof(int));
            ResetData(output);
            output.CopyFrom(tmp);
        }

#ifdef USE_CUDA
        static bool PrepareCudaData(Data &output, DataType dtype, const std::vector<int> &dims) {
            ResetData(output);
            output.dataType = dtype;
            output.Resize(dims);
            output.ToDevice(DataDevice::CUDA, false);
            output.Allocate(false);
            return output.cudaData != nullptr;
        }
#endif

        static void UpdateDebugPastKeyValues(std::vector<std::pair<Data, Data>> &pastKeyValues,
                                             int bsz, int totalLen, int blocks) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4PastKVStub");
            if (pastKeyValues.empty()) {
                return;
            }
            int paddedLen = ((std::max(totalLen, 1) - 1) / 128 + 1) * 128;
            std::vector<float> zeros((uint64_t)bsz * totalLen, 0.0f);
            for (int i = 0; i < std::min(blocks, (int)pastKeyValues.size()); i++) {
                Data key(DataType::FLOAT32, {bsz, totalLen, 1}, zeros);
                Data value(DataType::FLOAT32, {bsz, totalLen, 1}, zeros);
                key.SetKVCache();
                value.SetKVCache();
                key.Expansion({bsz, paddedLen, 1});
                value.Expansion({bsz, paddedLen, 1});
                ResetData(pastKeyValues[i].first);
                ResetData(pastKeyValues[i].second);
                pastKeyValues[i].first.CopyFrom(key);
                pastKeyValues[i].second.CopyFrom(value);
                pastKeyValues[i].first.SetKVCache();
                pastKeyValues[i].second.SetKVCache();
            }
        }

        static float SigmoidFloat(float x) {
            if (x >= 0.0f) {
                float z = std::exp(-x);
                return 1.0f / (1.0f + z);
            }
            float z = std::exp(x);
            return z / (1.0f + z);
        }

        static float SoftplusFloat(float x) {
            if (x > 20.0f) {
                return x;
            }
            if (x < -20.0f) {
                return std::exp(x);
            }
            return std::log1p(std::exp(x));
        }

        static void DebugDumpData(const std::string &name, const Data &input, int startPos = -1) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4DebugDump");
            Data tmp;
            ToDataType(input, tmp, DataType::FLOAT32);
            tmp.ToDevice(DataDevice::CPU);
            printf("[fastllm-debug] %-10s shape=(", name.c_str());
            for (int i = 0; i < (int)tmp.dims.size(); i++) {
                if (i) {
                    printf(", ");
                }
                printf("%d", tmp.dims[i]);
            }
            printf(") dtype=%s", GetDataTypeName(input.dataType).c_str());
            if (startPos >= 0) {
                printf(" start_pos=%d", startPos);
            }
            printf("\n");

            float *p = (float*)tmp.cpuData;
            uint64_t cnt = tmp.Count(0);
            if (cnt == 0) {
                return;
            }
            float mn = std::numeric_limits<float>::infinity();
            float mx = -std::numeric_limits<float>::infinity();
            double sum = 0.0, sq = 0.0;
            bool hasNan = false, hasInf = false;
            uint64_t finiteCount = 0, nanCount = 0, infCount = 0;
            uint64_t firstNan = cnt, firstInf = cnt;
            for (uint64_t i = 0; i < cnt; i++) {
                float v = p[i];
                if (std::isnan(v)) {
                    hasNan = true;
                    nanCount++;
                    if (firstNan == cnt) {
                        firstNan = i;
                    }
                    continue;
                }
                if (std::isinf(v)) {
                    hasInf = true;
                    infCount++;
                    if (firstInf == cnt) {
                        firstInf = i;
                    }
                    continue;
                }
                mn = std::min(mn, v);
                mx = std::max(mx, v);
                sum += v;
                sq += (double)v * (double)v;
                finiteCount++;
            }
            if (finiteCount == 0) {
                mn = mx = std::numeric_limits<float>::quiet_NaN();
            }
            double mean = finiteCount == 0 ? std::numeric_limits<double>::quiet_NaN() : sum / finiteCount;
            double var = finiteCount == 0 ? std::numeric_limits<double>::quiet_NaN()
                                          : std::max(0.0, sq / finiteCount - mean * mean);
            printf("[fastllm-debug]            min=%.4f max=%.4f mean=%.4f std=%.4f nan=%s inf=%s\n",
                   mn, mx, (float)mean, (float)std::sqrt(var), hasNan ? "true" : "false", hasInf ? "true" : "false");
            if (hasNan || hasInf) {
                auto printLocation = [&](uint64_t flat) {
                    printf("[");
                    std::vector<int> coords(tmp.dims.size(), 0);
                    for (int d = (int)tmp.dims.size() - 1; d >= 0; d--) {
                        coords[d] = flat % tmp.dims[d];
                        flat /= tmp.dims[d];
                    }
                    for (int d = 0; d < (int)coords.size(); d++) {
                        if (d) {
                            printf(", ");
                        }
                        printf("%d", coords[d]);
                    }
                    printf("]");
                };
                printf("[fastllm-debug]            finite=%llu nan_count=%llu inf_count=%llu",
                       (unsigned long long)finiteCount,
                       (unsigned long long)nanCount,
                       (unsigned long long)infCount);
                if (firstNan != cnt) {
                    printf(" first_nan=");
                    printLocation(firstNan);
                }
                if (firstInf != cnt) {
                    printf(" first_inf=");
                    printLocation(firstInf);
                }
                printf("\n");
            }

            uint64_t sampleOffset = 0;
            int sampleLen = 8;
            if (tmp.dims.size() == 4) {
                sampleOffset = (((uint64_t)0 * tmp.dims[1] + (tmp.dims[1] - 1)) * tmp.dims[2] + 0) * tmp.dims[3];
                sampleLen = std::min(sampleLen, tmp.dims[3]);
            } else if (tmp.dims.size() == 3) {
                sampleOffset = ((uint64_t)0 * tmp.dims[1] + (tmp.dims[1] - 1)) * tmp.dims[2];
                sampleLen = std::min(sampleLen, tmp.dims[2]);
            } else if (tmp.dims.size() == 2) {
                sampleOffset = 0;
                sampleLen = std::min(sampleLen, tmp.dims[1]);
            } else {
                sampleOffset = 0;
                sampleLen = std::min<uint64_t>(sampleLen, cnt);
            }
            printf("[fastllm-debug]            sample=[");
            for (int i = 0; i < sampleLen; i++) {
                if (i) {
                    printf(", ");
                }
                printf("%.9g", p[sampleOffset + i]);
            }
            printf("]\n");
        }

        static void DebugDumpInputIds(const Data &inputIds, int startPos) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4DebugDump");
            auto ids = ReadTokenIds(inputIds);
            printf("[fastllm-debug] input_ids  shape=(");
            for (int i = 0; i < (int)inputIds.dims.size(); i++) {
                if (i) {
                    printf(", ");
                }
                printf("%d", inputIds.dims[i]);
            }
            printf(") start_pos=%d\n[fastllm-debug]            values=[", startPos);
            int idx = 0;
            for (int b = 0; b < inputIds.dims[0]; b++) {
                if (b) {
                    printf(", ");
                }
                printf("[");
                for (int s = 0; s < inputIds.dims[1]; s++, idx++) {
                    if (s) {
                        printf(", ");
                    }
                    printf("%d", ids[idx]);
                }
                printf("]");
            }
            printf("]\n");
        }

        struct HcMix {
            Data y;
            Data postData;
            Data combData;
            std::vector<float> post;
            std::vector<float> comb;
            int b = 0, s = 0, hc = 0;
        };

        static bool HcPreCudaIfAvailable(const Data &x, Data &hcFn, Data &hcScale, Data &hcBase,
                                         int hcMult, int sinkhornIters, float eps, float normEps,
                                         HcMix &ret) {
#ifdef USE_CUDA
            auto fail = [](const char *) {
                return false;
            };
            if (EnvFlagEnabled("FASTLLM_DSV4_DISABLE_CUDA_HCPRE") ||
                x.dims.size() != 4 || (x.dataDevice != DataDevice::CUDA && !DeepSeekV4PreferCuda())) {
                return fail("disabled_or_not_cuda");
            }
            if (hcScale.dataType != DataType::FLOAT32 || hcBase.dataType != DataType::FLOAT32 ||
                (hcFn.dataType != DataType::FLOAT32 && hcFn.dataType != DataType::FLOAT16 &&
                 hcFn.dataType != DataType::BFLOAT16)) {
                return fail("unsupported_dtype");
            }
            int bsz = x.dims[0], seqlen = x.dims[1], dim = x.dims[3];
            int flatDim = hcMult * dim;
            int mixHc = (2 + hcMult) * hcMult;
            if (x.dims[2] != hcMult || hcFn.Count(0) != (uint64_t)mixHc * flatDim ||
                hcScale.Count(0) < 3 || hcBase.Count(0) < (uint64_t)mixHc ||
                bsz <= 0 || seqlen <= 0 || dim <= 0 || sinkhornIters <= 0) {
                return fail("shape_mismatch");
            }
            Data cudaX;
            const Data *cudaInput = &x;
            if (x.dataDevice != DataDevice::CUDA) {
                cudaX.CopyFrom(x);
                cudaX.ToDevice(DataDevice::CUDA);
                cudaInput = &cudaX;
            }
            hcFn.ToDevice(DataDevice::CUDA);
            hcScale.ToDevice(DataDevice::CUDA);
            hcBase.ToDevice(DataDevice::CUDA);
            if (!FastllmCudaDeepSeekV4HcPre(*cudaInput, hcFn, hcScale, hcBase, hcMult,
                                            sinkhornIters, eps, normEps,
                                            ret.y, ret.postData, ret.combData)) {
                ErrorInFastLLM("DeepSeekV4HcPre CUDA error: kernel rejected valid input.\n");
            }
            ret.b = bsz;
            ret.s = seqlen;
            ret.hc = hcMult;
            return true;
#else
            (void)x;
            (void)hcFn;
            (void)hcScale;
            (void)hcBase;
            (void)hcMult;
            (void)sinkhornIters;
            (void)eps;
            (void)normEps;
            (void)ret;
            return false;
#endif
        }

        struct HcPreDotsOp : MultiThreadBaseOp {
            const float *xrow;
            const float *fn;
            float *mixes;
            float rsqrt;
            int flatDim, mixSt, mixEnd;

            HcPreDotsOp(const float *xrow, const float *fn, float *mixes, float rsqrt,
                        int flatDim, int mixSt, int mixEnd)
                : xrow(xrow), fn(fn), mixes(mixes), rsqrt(rsqrt),
                  flatDim(flatDim), mixSt(mixSt), mixEnd(mixEnd) {}

            void Run() override {
                for (int m = mixSt; m < mixEnd; m++) {
                    double v = 0.0;
                    const float *w = fn + (uint64_t)m * flatDim;
                    for (int k = 0; k < flatDim; k++) {
                        v += (double)xrow[k] * w[k];
                    }
                    mixes[m] = (float)v * rsqrt;
                }
            }
        };

        static void HcPreComputeDotsCpu(const float *xrow, const float *fn, float *mixes,
                                        float rsqrt, int flatDim, int mixHc) {
            auto *pool = GetAlivePool();
            int threadNum = std::min((int)pool->threads.size(), mixHc);
            if (threadNum <= 1 || mixHc < 8 || EnvFlagEnabled("FASTLLM_DSV4_DISABLE_CPU_HCPRE_PARALLEL")) {
                HcPreDotsOp(xrow, fn, mixes, rsqrt, flatDim, 0, mixHc).Run();
                return;
            }
            std::vector<HcPreDotsOp*> ops;
            int per = (mixHc + threadNum - 1) / threadNum;
            for (int i = 0; i < threadNum; i++) {
                int st = i * per;
                int end = std::min(mixHc, st + per);
                if (st >= end) {
                    break;
                }
                ops.push_back(new HcPreDotsOp(xrow, fn, mixes, rsqrt, flatDim, st, end));
            }
            for (int i = 0; i < (int)ops.size(); i++) {
                pool->PushOp(i, ops[i]);
            }
            for (int i = 0; i < (int)ops.size(); i++) {
                pool->Wait(i);
                delete ops[i];
            }
        }

        static HcMix HcPreReference(const Data &x, Data &hcFn, Data &hcScale, Data &hcBase,
                                    int hcMult, int sinkhornIters, float eps, float normEps) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4HcPre");
            HcMix ret;
            if (HcPreCudaIfAvailable(x, hcFn, hcScale, hcBase, hcMult, sinkhornIters, eps, normEps, ret)) {
                return ret;
            }
            int bsz = x.dims[0], seqlen = x.dims[1], dim = x.dims[3];
            int flatDim = hcMult * dim;
            int mixHc = (2 + hcMult) * hcMult;
            bool profileHcPre = EnvFlagEnabled("FASTLLM_PROFILE_HCPRE");
            double profileLast = profileHcPre ? NowMs() : 0.0;
            double profileRead = 0.0, profileCudaDots = 0.0, profileAlloc = 0.0;
            double profileNorm = 0.0, profileDots = 0.0, profileGates = 0.0;
            double profileSinkhorn = 0.0, profileY = 0.0, profileWrite = 0.0;
            auto profileLap = [&](double &bucket) {
                if (!profileHcPre) {
                    return;
                }
                double now = NowMs();
                bucket += now - profileLast;
                profileLast = now;
            };
            auto xv = ReadFloatData(x);
            auto scalePtr = ReadWeightFloatDataCached(hcScale);
            auto basePtr = ReadWeightFloatDataCached(hcBase);
            const auto &scale = *scalePtr;
            const auto &base = *basePtr;
            std::shared_ptr<const std::vector<float>> fnPtr;
            const std::vector<float> *fn = nullptr;
            std::vector<float> dotCache;
            profileLap(profileRead);

#ifdef USE_CUDA
            if (EnvFlagEnabled("FASTLLM_DSV4_ENABLE_CUDA_HCPRE_DOTS") && x.dataDevice == DataDevice::CUDA) {
                Data dots;
                hcFn.ToDevice(DataDevice::CUDA);
                if (FastllmCudaDeepSeekV4HcPreDots(x, hcFn, hcMult, dots)) {
                    dotCache = ReadFloatData(dots);
                    if ((int)dotCache.size() != bsz * seqlen * mixHc) {
                        dotCache.clear();
                    }
                }
                profileLap(profileCudaDots);
            }
#endif

            if (dotCache.empty()) {
                fnPtr = ReadWeightFloatDataCached(hcFn);
                fn = fnPtr.get();
                profileLap(profileRead);
            }

            std::vector<float> y((uint64_t)bsz * seqlen * dim, 0.0f);
            std::vector<float> post((uint64_t)bsz * seqlen * hcMult, 0.0f);
            std::vector<float> comb((uint64_t)bsz * seqlen * hcMult * hcMult, 0.0f);
            std::vector<float> mixes(mixHc);
            std::vector<float> pre(hcMult);
            std::vector<float> combLocal(hcMult * hcMult);
            profileLap(profileAlloc);

            for (int t = 0; t < bsz * seqlen; t++) {
                const float *xrow = xv.data() + (uint64_t)t * flatDim;
                double ss = 0.0;
                for (int k = 0; k < flatDim; k++) {
                    ss += (double)xrow[k] * xrow[k];
                }
                float rsqrt = 1.0f / std::sqrt((float)(ss / flatDim) + normEps);
                profileLap(profileNorm);
                for (int m = 0; m < mixHc; m++) {
                    if (!dotCache.empty()) {
                        mixes[m] = dotCache[(uint64_t)t * mixHc + m] * rsqrt;
                    } else {
                        HcPreComputeDotsCpu(xrow, fn->data(), mixes.data(), rsqrt, flatDim, mixHc);
                        break;
                    }
                }
                profileLap(profileDots);
                for (int h = 0; h < hcMult; h++) {
                    pre[h] = SigmoidFloat(mixes[h] * scale[0] + base[h]) + eps;
                    post[(uint64_t)t * hcMult + h] =
                        2.0f * SigmoidFloat(mixes[h + hcMult] * scale[1] + base[h + hcMult]);
                }
                profileLap(profileGates);
                for (int r = 0; r < hcMult; r++) {
                    float rowMax = -std::numeric_limits<float>::infinity();
                    for (int c = 0; c < hcMult; c++) {
                        int idx = r * hcMult + c + 2 * hcMult;
                        combLocal[r * hcMult + c] = mixes[idx] * scale[2] + base[idx];
                        rowMax = std::max(rowMax, combLocal[r * hcMult + c]);
                    }
                    float rowSum = 0.0f;
                    for (int c = 0; c < hcMult; c++) {
                        float v = std::exp(combLocal[r * hcMult + c] - rowMax);
                        combLocal[r * hcMult + c] = v;
                        rowSum += v;
                    }
                    for (int c = 0; c < hcMult; c++) {
                        combLocal[r * hcMult + c] = combLocal[r * hcMult + c] / rowSum + eps;
                    }
                }
                for (int c = 0; c < hcMult; c++) {
                    float colSum = 0.0f;
                    for (int r = 0; r < hcMult; r++) {
                        colSum += combLocal[r * hcMult + c];
                    }
                    for (int r = 0; r < hcMult; r++) {
                        combLocal[r * hcMult + c] /= (colSum + eps);
                    }
                }
                for (int it = 1; it < sinkhornIters; it++) {
                    for (int r = 0; r < hcMult; r++) {
                        float rowSum = 0.0f;
                        for (int c = 0; c < hcMult; c++) {
                            rowSum += combLocal[r * hcMult + c];
                        }
                        for (int c = 0; c < hcMult; c++) {
                            combLocal[r * hcMult + c] /= (rowSum + eps);
                        }
                    }
                    for (int c = 0; c < hcMult; c++) {
                        float colSum = 0.0f;
                        for (int r = 0; r < hcMult; r++) {
                            colSum += combLocal[r * hcMult + c];
                        }
                        for (int r = 0; r < hcMult; r++) {
                            combLocal[r * hcMult + c] /= (colSum + eps);
                        }
                    }
                }
                memcpy(comb.data() + (uint64_t)t * hcMult * hcMult, combLocal.data(),
                       hcMult * hcMult * sizeof(float));
                profileLap(profileSinkhorn);
                for (int d = 0; d < dim; d++) {
                    double v = 0.0;
                    for (int h = 0; h < hcMult; h++) {
                        v += (double)pre[h] * xrow[(uint64_t)h * dim + d];
                    }
                    y[(uint64_t)t * dim + d] = (float)v;
                }
                profileLap(profileY);
            }

            WriteFloatData(y, {bsz, seqlen, dim}, ret.y, x.dataType);
#ifdef USE_CUDA
            if (x.dataDevice == DataDevice::CUDA) {
                ret.y.ToDevice(DataDevice::CUDA);
            }
#endif
            ret.post = std::move(post);
            ret.comb = std::move(comb);
            ret.b = bsz;
            ret.s = seqlen;
            ret.hc = hcMult;
            profileLap(profileWrite);
            if (profileHcPre) {
                double total = profileRead + profileCudaDots + profileAlloc + profileNorm +
                               profileDots + profileGates + profileSinkhorn + profileY + profileWrite;
                printf("[fastllm-profile-hcpre] rows=%d hc=%d dim=%d cuda_dots=%d read=%.3f cuda=%.3f alloc=%.3f norm=%.3f dots=%.3f gates=%.3f sinkhorn=%.3f y=%.3f write=%.3f total=%.3f\n",
                       bsz * seqlen, hcMult, dim, dotCache.empty() ? 0 : 1, profileRead,
                       profileCudaDots, profileAlloc, profileNorm, profileDots, profileGates,
                       profileSinkhorn, profileY, profileWrite, total);
                fflush(stdout);
            }
            return ret;
        }

        static bool HcPostCudaIfAvailable(Data &x, Data &residual, const HcMix &mix, Data &output) {
#ifdef USE_CUDA
            auto fail = [](const char *) {
                return false;
            };
            if (EnvFlagEnabled("FASTLLM_DSV4_DISABLE_CUDA_WOA_HCPOST")) {
                return fail("disabled");
            }
            if (x.dataDevice != DataDevice::CUDA || x.dims.size() < 2 || residual.dims.size() != 4) {
                return fail("shape_or_device");
            }
            int bsz = residual.dims[0], seqlen = residual.dims[1], hcMult = residual.dims[2], dim = residual.dims[3];
            if (x.Count(0) != (uint64_t)bsz * seqlen * dim) {
                return fail("x_count");
            }
            residual.ToDevice(DataDevice::CUDA);
            if (mix.postData.dataDevice == DataDevice::CUDA && mix.combData.dataDevice == DataDevice::CUDA &&
                mix.postData.Count(0) == (uint64_t)bsz * seqlen * hcMult &&
                mix.combData.Count(0) == (uint64_t)bsz * seqlen * hcMult * hcMult) {
                if (FastllmCudaDeepSeekV4HcPostCudaMix(x, residual, mix.postData, mix.combData,
                                                       bsz, seqlen, hcMult, dim, output)) {
                    return true;
                }
                return fail("cuda_mix_kernel");
            }
            if ((int)mix.post.size() != bsz * seqlen * hcMult ||
                (int)mix.comb.size() != bsz * seqlen * hcMult * hcMult) {
                return fail("host_mix_shape");
            }
            return FastllmCudaDeepSeekV4HcPost(x, residual, mix.post.data(), mix.comb.data(),
                                               bsz, seqlen, hcMult, dim, output);
#else
            (void)x;
            (void)residual;
            (void)mix;
            (void)output;
            return false;
#endif
        }

        static void HcPostReference(const Data &x, const Data &residual, const HcMix &mix, Data &output) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4HcPost");
#ifdef USE_CUDA
            if (x.dataDevice == DataDevice::CUDA || DeepSeekV4PreferCuda()) {
                Data cudaX, cudaResidual;
                cudaX.CopyFrom(x);
                cudaX.ToDevice(DataDevice::CUDA);
                cudaResidual.CopyFrom(residual);
                cudaResidual.ToDevice(DataDevice::CUDA);
                if (HcPostCudaIfAvailable(cudaX, cudaResidual, mix, output)) {
                    return;
                }
            }
#endif

            int bsz = residual.dims[0], seqlen = residual.dims[1], hcMult = residual.dims[2], dim = residual.dims[3];
            auto xv = ReadFloatData(x);
            auto rv = ReadFloatData(residual);
            std::vector<float> postHost, combHost;
            const std::vector<float> *postVec = &mix.post;
            const std::vector<float> *combVec = &mix.comb;
            if ((int)postVec->size() != bsz * seqlen * hcMult && mix.postData.Count(0) == (uint64_t)bsz * seqlen * hcMult) {
                postHost = ReadFloatData(mix.postData);
                postVec = &postHost;
            }
            if ((int)combVec->size() != bsz * seqlen * hcMult * hcMult &&
                mix.combData.Count(0) == (uint64_t)bsz * seqlen * hcMult * hcMult) {
                combHost = ReadFloatData(mix.combData);
                combVec = &combHost;
            }
            if ((int)postVec->size() != bsz * seqlen * hcMult ||
                (int)combVec->size() != bsz * seqlen * hcMult * hcMult) {
                ErrorInFastLLM("DeepSeekV4HcPost error: invalid hc mix shape.\n");
            }
            std::vector<float> y((uint64_t)bsz * seqlen * hcMult * dim, 0.0f);
            for (int t = 0; t < bsz * seqlen; t++) {
                const float *xrow = xv.data() + (uint64_t)t * dim;
                const float *rrow = rv.data() + (uint64_t)t * hcMult * dim;
                const float *post = postVec->data() + (uint64_t)t * hcMult;
                const float *comb = combVec->data() + (uint64_t)t * hcMult * hcMult;
                for (int target = 0; target < hcMult; target++) {
                    for (int d = 0; d < dim; d++) {
                        double v = (double)post[target] * xrow[d];
                        for (int src = 0; src < hcMult; src++) {
                            v += (double)comb[src * hcMult + target] * rrow[(uint64_t)src * dim + d];
                        }
                        y[((uint64_t)t * hcMult + target) * dim + d] = (float)v;
                    }
                }
            }
            WriteFloatData(y, {bsz, seqlen, hcMult, dim}, output, x.dataType);
        }

        static bool RMSNormCudaIfAvailable(const Data &input, Data &weight, float eps, Data &output, DataType dtype) {
#ifdef USE_CUDA
            if (EnvFlagEnabled("FASTLLM_DSV4_DISABLE_CUDA_PREP") ||
                weight.dataType != DataType::FLOAT32 || input.dims.empty()) {
                return false;
            }
            if (input.dataType != DataType::FLOAT32 && input.dataType != DataType::FLOAT16 &&
                input.dataType != DataType::BFLOAT16) {
                return false;
            }
            if (dtype != DataType::FLOAT32 && dtype != DataType::FLOAT16 && dtype != DataType::BFLOAT16) {
                return false;
            }
            weight.ToDevice(DataDevice::CUDA);
            Data cudaInput;
            const Data *cudaInputPtr = &input;
            if (input.dataDevice != DataDevice::CUDA) {
                if (!DeepSeekV4PreferCuda()) {
                    return false;
                }
                cudaInput.CopyFrom(input);
                cudaInput.ToDevice(DataDevice::CUDA);
                cudaInputPtr = &cudaInput;
            }
            if (&input == &output && dtype != input.dataType) {
                Data tmp;
                if (!FastllmCudaDeepSeekV4RMSNorm(*cudaInputPtr, weight, eps, tmp, dtype)) {
                    return false;
                }
                output.CopyFrom(tmp);
                return true;
            }
            return FastllmCudaDeepSeekV4RMSNorm(*cudaInputPtr, weight, eps, output, dtype);
#else
            (void)input;
            (void)weight;
            (void)eps;
            (void)output;
            (void)dtype;
            return false;
#endif
        }

        static void RMSNormReference(const Data &input, Data &weight, float eps, Data &output, DataType dtype) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4RMSNorm");
            if (RMSNormCudaIfAvailable(input, weight, eps, output, dtype)) {
                return;
            }

            auto xv = ReadFloatData(input);
            auto wvPtr = ReadWeightFloatDataCached(weight);
            const auto &wv = *wvPtr;
            int dim = input.dims.back();
            int rows = (int)(xv.size() / dim);
            std::vector<float> y(xv.size());
            for (int r = 0; r < rows; r++) {
                const float *src = xv.data() + (uint64_t)r * dim;
                double ss = 0.0;
                for (int d = 0; d < dim; d++) {
                    ss += (double)src[d] * src[d];
                }
                float scale = 1.0f / std::sqrt((float)(ss / dim) + eps);
                for (int d = 0; d < dim; d++) {
                    y[(uint64_t)r * dim + d] = src[d] * scale * wv[d];
                }
            }
            WriteFloatData(y, input.dims, output, dtype);
        }

        static std::vector<float> BuildInvFreqReference(int ropeDim, float base, int originalSeqLen,
                                                        float factor, int betaFast, int betaSlow) {
            std::vector<float> invFreq;
            for (int i = 0; i < ropeDim; i += 2) {
                invFreq.push_back(1.0f / std::pow(base, (float)i / ropeDim));
            }
            if (originalSeqLen > 0) {
                float lowF = ropeDim * std::log((float)originalSeqLen / (betaFast * 2.0f * (float)M_PI)) /
                             (2.0f * std::log(base));
                float highF = ropeDim * std::log((float)originalSeqLen / (betaSlow * 2.0f * (float)M_PI)) /
                              (2.0f * std::log(base));
                int low = std::max((int)std::floor(lowF), 0);
                int high = std::min((int)std::ceil(highF), ropeDim - 1);
                if (low == high) {
                    high++;
                }
                for (int i = 0; i < (int)invFreq.size(); i++) {
                    float ramp = std::max(0.0f, std::min(1.0f, ((float)i - low) / (high - low)));
                    float smooth = 1.0f - ramp;
                    invFreq[i] = invFreq[i] / factor * (1.0f - smooth) + invFreq[i] * smooth;
                }
            }
            return invFreq;
        }

        static void ApplyRotaryReference(std::vector<float> &x, const std::vector<int> &dims,
                                         int ropeDim, float base, int startPos, bool inverse,
                                         int originalSeqLen = 0, float factor = 1.0f,
                                         int betaFast = 32, int betaSlow = 1, int posStep = 1) {
            int bsz = dims[0], seqlen = dims[1];
            int heads = (dims.size() == 4) ? dims[2] : 1;
            int dim = (dims.size() == 4) ? dims[3] : dims[2];
            int off = dim - ropeDim;
            auto invFreq = BuildInvFreqReference(ropeDim, base, originalSeqLen, factor, betaFast, betaSlow);
            for (int b = 0; b < bsz; b++) {
                for (int s = 0; s < seqlen; s++) {
                    int pos = startPos + s * posStep;
                    for (int h = 0; h < heads; h++) {
                        uint64_t rowIndex = dims.size() == 4 ? (((uint64_t)b * seqlen + s) * heads + h)
                                                             : ((uint64_t)b * seqlen + s);
                        float *row = x.data() + rowIndex * dim + off;
                        for (int i = 0; i < ropeDim; i += 2) {
                            float ang = pos * invFreq[i / 2];
                            float c = std::cos(ang), sn = std::sin(ang);
                            if (inverse) {
                                sn = -sn;
                            }
                            float a = row[i], bb = row[i + 1];
                            row[i] = a * c - bb * sn;
                            row[i + 1] = a * sn + bb * c;
                        }
                    }
                }
            }
        }

        static void ScaleQReference(Data &q, float eps) {
            auto qv = ReadFloatData(q);
            int dim = q.dims.back();
            int rows = (int)(qv.size() / dim);
            for (int r = 0; r < rows; r++) {
                float *row = qv.data() + (uint64_t)r * dim;
                double ss = 0.0;
                for (int d = 0; d < dim; d++) {
                    ss += (double)row[d] * row[d];
                }
                float scale = 1.0f / std::sqrt((float)(ss / dim) + eps);
                for (int d = 0; d < dim; d++) {
                    row[d] *= scale;
                }
            }
            WriteFloatData(qv, q.dims, q, DataType::BFLOAT16);
        }

        static bool ScaleQRotaryCudaIfAvailable(Data &q, int ropeDim, float ropeBase, int startPos,
                                                int originalSeqLen, float ropeFactor,
                                                int betaFast, int betaSlow, float eps) {
#ifdef USE_CUDA
            if (EnvFlagEnabled("FASTLLM_DSV4_DISABLE_CUDA_PREP") ||
                q.dataDevice != DataDevice::CUDA || q.dims.size() != 4 ||
                q.dataType != DataType::BFLOAT16) {
                return false;
            }
            return FastllmCudaDeepSeekV4ScaleQRotary(q, ropeDim, ropeBase, startPos,
                                                     originalSeqLen, ropeFactor, betaFast, betaSlow, eps);
#else
            (void)q;
            (void)ropeDim;
            (void)ropeBase;
            (void)startPos;
            (void)originalSeqLen;
            (void)ropeFactor;
            (void)betaFast;
            (void)betaSlow;
            (void)eps;
            return false;
#endif
        }

        static void ScaleQRotary(Data &q, float eps, int ropeDim, float ropeBase, int startPos,
                                 int originalSeqLen, float ropeFactor, int betaFast, int betaSlow) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4ScaleQRotary");
            if (ScaleQRotaryCudaIfAvailable(q, ropeDim, ropeBase, startPos, originalSeqLen,
                                            ropeFactor, betaFast, betaSlow, eps)) {
                return;
            }
            ScaleQReference(q, eps);
            auto qv = ReadFloatData(q);
            ApplyRotaryReference(qv, q.dims, ropeDim, ropeBase, startPos, false,
                                 originalSeqLen, ropeFactor, betaFast, betaSlow);
            WriteFloatData(qv, q.dims, q, DataType::BFLOAT16);
        }

        static void ActQuantInplaceReference(std::vector<float> &x, const std::vector<int> &dims,
                                             int quantDim, int blockSize) {
            int dim = dims.back();
            int rows = (int)(x.size() / dim);
            for (int r = 0; r < rows; r++) {
                float *row = x.data() + (uint64_t)r * dim;
                for (int start = 0; start < quantDim; start += blockSize) {
                    int end = std::min(start + blockSize, quantDim);
                    float amax = 1e-4f;
                    for (int d = start; d < end; d++) {
                        amax = std::max(amax, std::fabs(row[d]));
                    }
                    float scale = std::pow(2.0f, std::ceil(std::log2(amax / 448.0f)));
                    for (int d = start; d < end; d++) {
                        float q = std::max(-448.0f, std::min(448.0f, row[d] / scale));
                        row[d] = BFloat16ToFloat(FloatToBFloat16(q)) * scale;
                    }
                }
            }
        }

        static bool RotaryQuantCudaIfAvailable(Data &x, int ropeDim, float ropeBase, int startPos,
                                               int originalSeqLen, float ropeFactor,
                                               int betaFast, int betaSlow, int quantDim,
                                               int blockSize, int posStep = 1) {
#ifdef USE_CUDA
            if (EnvFlagEnabled("FASTLLM_DSV4_DISABLE_CUDA_PREP") ||
                x.dataDevice != DataDevice::CUDA || x.dataType != DataType::BFLOAT16 ||
                x.dims.size() < 3 || x.dims.size() > 4) {
                return false;
            }
            return FastllmCudaDeepSeekV4RotaryQuant(x, ropeDim, ropeBase, startPos,
                                                   originalSeqLen, ropeFactor, betaFast, betaSlow,
                                                   quantDim, blockSize, posStep);
#else
            (void)x;
            (void)ropeDim;
            (void)ropeBase;
            (void)startPos;
            (void)originalSeqLen;
            (void)ropeFactor;
            (void)betaFast;
            (void)betaSlow;
            (void)quantDim;
            (void)blockSize;
            (void)posStep;
            return false;
#endif
        }

        static void RotaryQuant(Data &x, int ropeDim, float ropeBase, int startPos,
                                int originalSeqLen, float ropeFactor, int betaFast, int betaSlow,
                                int quantDim, int blockSize, int posStep = 1) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4RotaryQuant");
            if (RotaryQuantCudaIfAvailable(x, ropeDim, ropeBase, startPos, originalSeqLen,
                                           ropeFactor, betaFast, betaSlow, quantDim, blockSize, posStep)) {
                return;
            }
            auto xv = ReadFloatData(x);
            ApplyRotaryReference(xv, x.dims, ropeDim, ropeBase, startPos, false,
                                 originalSeqLen, ropeFactor, betaFast, betaSlow, posStep);
            ActQuantInplaceReference(xv, x.dims, quantDim, blockSize);
            WriteFloatData(xv, x.dims, x, DataType::BFLOAT16);
        }

        static void StoreWindowKVCache(const std::vector<float> &kv, int bsz, int seqlen, int headDim,
                                       int startPos, int windowSize, std::vector<float> &windowKV) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4KVCache");
            windowKV.assign((uint64_t)bsz * windowSize * headDim, 0.0f);
            if (startPos == 0) {
                if (seqlen <= windowSize) {
                    for (int b = 0; b < bsz; b++) {
                        memcpy(windowKV.data() + (uint64_t)b * windowSize * headDim,
                               kv.data() + (uint64_t)b * seqlen * headDim,
                               (uint64_t)seqlen * headDim * sizeof(float));
                    }
                } else {
                    int cutoff = seqlen % windowSize;
                    int first = windowSize - cutoff;
                    for (int b = 0; b < bsz; b++) {
                        const float *src = kv.data() + ((uint64_t)b * seqlen + seqlen - windowSize) * headDim;
                        memcpy(windowKV.data() + ((uint64_t)b * windowSize + cutoff) * headDim,
                               src, (uint64_t)first * headDim * sizeof(float));
                        if (cutoff > 0) {
                            memcpy(windowKV.data() + (uint64_t)b * windowSize * headDim,
                                   src + (uint64_t)first * headDim,
                                   (uint64_t)cutoff * headDim * sizeof(float));
                        }
                    }
                }
                return;
            }
            for (int b = 0; b < bsz; b++) {
                memcpy(windowKV.data() + ((uint64_t)b * windowSize + (startPos % windowSize)) * headDim,
                       kv.data() + (uint64_t)b * headDim,
                       (uint64_t)headDim * sizeof(float));
            }
        }

        static void UpdateWindowKVCache(const std::vector<float> &kv, int bsz, int headDim,
                                        int startPos, int windowSize, std::vector<float> &windowKV) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4KVCache");
            int seqlen = (int)(kv.size() / ((uint64_t)bsz * headDim));
            for (int b = 0; b < bsz; b++) {
                for (int s = 0; s < seqlen; s++) {
                    memcpy(windowKV.data() + ((uint64_t)b * windowSize + ((startPos + s) % windowSize)) * headDim,
                           kv.data() + ((uint64_t)b * seqlen + s) * headDim,
                           (uint64_t)headDim * sizeof(float));
                }
            }
        }

        static int BuildWindowKVPrefixData(const std::vector<float> &windowKV, int bsz, int headDim,
                                           int startPos, int windowSize, Data &output) {
            int prefixLen = std::min(windowSize, startPos);
            if (prefixLen <= 0 || windowKV.empty()) {
                return 0;
            }
            ScopedExecutorProfiler executorProfile("DeepSeekV4KVCache");
            std::vector<float> prefix((uint64_t)bsz * prefixLen * headDim);
            int firstPos = startPos - prefixLen;
            for (int b = 0; b < bsz; b++) {
                for (int s = 0; s < prefixLen; s++) {
                    int srcSlot = (firstPos + s) % windowSize;
                    memcpy(prefix.data() + ((uint64_t)b * prefixLen + s) * headDim,
                           windowKV.data() + ((uint64_t)b * windowSize + srcSlot) * headDim,
                           (uint64_t)headDim * sizeof(float));
                }
            }
            WriteFloatData(prefix, {bsz, prefixLen, headDim}, output, DataType::FLOAT32);
            return prefixLen;
        }

        static void ComputeCompressorRaw(WeightMap &weight, const std::string &prefix, const Data &x,
                                         std::vector<float> &kv, std::vector<float> &score) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4CompressorRaw");
            Data kvData, scoreData;
            Linear((Data&)x, weight[prefix + ".wkv.weight"], Data(), kvData);
            Linear((Data&)x, weight[prefix + ".wgate.weight"], Data(), scoreData);
            kv = ReadFloatData(kvData);
            score = ReadFloatData(scoreData);
        }

        static void AppendCompressorRaw(const std::vector<float> &kv, const std::vector<float> &score,
                                        int bsz, int seqlen, int wideDim,
                                        std::vector<float> &allKV, std::vector<float> &allScore) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4CompressorAppend");
            int oldLen = allKV.empty() ? 0 : (int)(allKV.size() / ((uint64_t)bsz * wideDim));
            if (bsz == 1) {
                uint64_t oldSize = (uint64_t)oldLen * wideDim;
                uint64_t addSize = (uint64_t)seqlen * wideDim;
                allKV.resize(oldSize + addSize);
                allScore.resize(oldSize + addSize);
                memcpy(allKV.data() + oldSize, kv.data(), addSize * sizeof(float));
                memcpy(allScore.data() + oldSize, score.data(), addSize * sizeof(float));
                return;
            }
            std::vector<float> nextKV((uint64_t)bsz * (oldLen + seqlen) * wideDim);
            std::vector<float> nextScore((uint64_t)bsz * (oldLen + seqlen) * wideDim);
            for (int b = 0; b < bsz; b++) {
                if (oldLen > 0) {
                    memcpy(nextKV.data() + (uint64_t)b * (oldLen + seqlen) * wideDim,
                           allKV.data() + (uint64_t)b * oldLen * wideDim,
                           (uint64_t)oldLen * wideDim * sizeof(float));
                    memcpy(nextScore.data() + (uint64_t)b * (oldLen + seqlen) * wideDim,
                           allScore.data() + (uint64_t)b * oldLen * wideDim,
                           (uint64_t)oldLen * wideDim * sizeof(float));
                }
                memcpy(nextKV.data() + ((uint64_t)b * (oldLen + seqlen) + oldLen) * wideDim,
                       kv.data() + (uint64_t)b * seqlen * wideDim,
                       (uint64_t)seqlen * wideDim * sizeof(float));
                memcpy(nextScore.data() + ((uint64_t)b * (oldLen + seqlen) + oldLen) * wideDim,
                       score.data() + (uint64_t)b * seqlen * wideDim,
                       (uint64_t)seqlen * wideDim * sizeof(float));
            }
            allKV.swap(nextKV);
            allScore.swap(nextScore);
        }

        struct BuildCompressedKVRangeOp : MultiThreadBaseOp {
            const float *kv;
            const float *score;
            const float *ape;
            float *compressed;
            uint64_t st, end;
            int bsz, rawLen, blockStart, blockCount, compressRatio, headDim, wideDim;
            bool overlap;

            BuildCompressedKVRangeOp(const float *kv, const float *score, const float *ape,
                                     float *compressed, uint64_t st, uint64_t end,
                                     int bsz, int rawLen, int blockStart, int blockCount,
                                     int compressRatio, int headDim, int wideDim, bool overlap)
                : kv(kv), score(score), ape(ape), compressed(compressed), st(st), end(end),
                  bsz(bsz), rawLen(rawLen), blockStart(blockStart), blockCount(blockCount),
                  compressRatio(compressRatio), headDim(headDim), wideDim(wideDim),
                  overlap(overlap) {}

            void ScanTerms(int b, int block, int d, float &mx) const {
                if (overlap) {
                    if (block > 0) {
                        for (int r = 0; r < compressRatio; r++) {
                            int tok = (block - 1) * compressRatio + r;
                            uint64_t off = ((uint64_t)b * rawLen + tok) * wideDim + d;
                            mx = std::max(mx, score[off] + ape[(uint64_t)r * wideDim + d]);
                        }
                    }
                    for (int r = 0; r < compressRatio; r++) {
                        int tok = block * compressRatio + r;
                        uint64_t off = ((uint64_t)b * rawLen + tok) * wideDim + headDim + d;
                        mx = std::max(mx, score[off] + ape[(uint64_t)r * wideDim + headDim + d]);
                    }
                } else {
                    for (int r = 0; r < compressRatio; r++) {
                        int tok = block * compressRatio + r;
                        uint64_t off = ((uint64_t)b * rawLen + tok) * wideDim + d;
                        mx = std::max(mx, score[off] + ape[(uint64_t)r * wideDim + d]);
                    }
                }
            }

            void AccumulateTerms(int b, int block, int d, float mx, double &sum, double &value) const {
                if (overlap) {
                    if (block > 0) {
                        for (int r = 0; r < compressRatio; r++) {
                            int tok = (block - 1) * compressRatio + r;
                            uint64_t off = ((uint64_t)b * rawLen + tok) * wideDim + d;
                            double e = std::exp((double)(score[off] + ape[(uint64_t)r * wideDim + d]) - mx);
                            sum += e;
                            value += e * kv[off];
                        }
                    }
                    for (int r = 0; r < compressRatio; r++) {
                        int tok = block * compressRatio + r;
                        uint64_t off = ((uint64_t)b * rawLen + tok) * wideDim + headDim + d;
                        double e = std::exp((double)(score[off] + ape[(uint64_t)r * wideDim + headDim + d]) - mx);
                        sum += e;
                        value += e * kv[off];
                    }
                } else {
                    for (int r = 0; r < compressRatio; r++) {
                        int tok = block * compressRatio + r;
                        uint64_t off = ((uint64_t)b * rawLen + tok) * wideDim + d;
                        double e = std::exp((double)(score[off] + ape[(uint64_t)r * wideDim + d]) - mx);
                        sum += e;
                        value += e * kv[off];
                    }
                }
            }

            void Run() override {
                (void)bsz;
                for (uint64_t idx = st; idx < end; idx++) {
                    int d = (int)(idx % headDim);
                    uint64_t tmp = idx / headDim;
                    int localBlock = (int)(tmp % blockCount);
                    int b = (int)(tmp / blockCount);
                    int block = blockStart + localBlock;

                    float mx = -std::numeric_limits<float>::infinity();
                    ScanTerms(b, block, d, mx);

                    double sum = 0.0, value = 0.0;
                    AccumulateTerms(b, block, d, mx, sum, value);
                    compressed[((uint64_t)b * blockCount + localBlock) * headDim + d] =
                        (float)(value / std::max(sum, 1e-30));
                }
            }
        };

        static void ComputeCompressedKVRangeCpu(const std::vector<float> &kv,
                                                const std::vector<float> &score,
                                                const std::vector<float> &ape,
                                                int bsz, int rawLen, int blockStart, int blockCount,
                                                int compressRatio, int headDim, int wideDim, bool overlap,
                                                std::vector<float> &compressed) {
            compressed.assign((uint64_t)bsz * blockCount * headDim, 0.0f);
            uint64_t total = (uint64_t)bsz * blockCount * headDim;
            if (total == 0) {
                return;
            }
            auto *pool = GetAlivePool();
            int threadNum = std::min((int)pool->threads.size(), (int)std::min<uint64_t>(total, 64));
            if (threadNum <= 1 || total < 4096 ||
                EnvFlagEnabled("FASTLLM_DSV4_DISABLE_CPU_COMPRESSKV_PARALLEL")) {
                BuildCompressedKVRangeOp(kv.data(), score.data(), ape.data(), compressed.data(), 0, total,
                                         bsz, rawLen, blockStart, blockCount, compressRatio,
                                         headDim, wideDim, overlap).Run();
                return;
            }

            std::vector<BuildCompressedKVRangeOp*> ops;
            uint64_t per = (total + threadNum - 1) / threadNum;
            for (int i = 0; i < threadNum; i++) {
                uint64_t st = (uint64_t)i * per;
                uint64_t end = std::min(total, st + per);
                if (st >= end) {
                    break;
                }
                ops.push_back(new BuildCompressedKVRangeOp(
                    kv.data(), score.data(), ape.data(), compressed.data(), st, end,
                    bsz, rawLen, blockStart, blockCount, compressRatio, headDim, wideDim, overlap));
            }
            for (int i = 0; i < (int)ops.size(); i++) {
                pool->PushOp(i, ops[i]);
            }
            for (int i = 0; i < (int)ops.size(); i++) {
                pool->Wait(i);
                delete ops[i];
            }
        }

        static void FinalizeCompressedKVRows(WeightMap &weight, const std::string &prefix,
                                             std::vector<float> &compressed, int bsz, int blockStart,
                                             int blockCount, int compressRatio, int headDim,
                                             int ropeDim, float ropeBase, float ropeFactor,
                                             int betaFast, int betaSlow, int originalSeqLen,
                                             Data &output) {
            Data compressedData, normed;
            WriteFloatData(compressed, {bsz, blockCount, headDim}, compressedData, DataType::BFLOAT16);
            RMSNormReference(compressedData, weight[prefix + ".norm.weight"], 1e-6f, normed, DataType::BFLOAT16);
            auto out = ReadFloatData(normed);
            ApplyRotaryReference(out, normed.dims, ropeDim, ropeBase, blockStart * compressRatio, false,
                                 originalSeqLen, ropeFactor, betaFast, betaSlow, compressRatio);
            ActQuantInplaceReference(out, normed.dims, headDim - ropeDim, 64);
            WriteFloatData(out, normed.dims, output, DataType::BFLOAT16);
        }

        static int GetReusableCompressedBlocks(const Data &output, int bsz, int blocks, int headDim) {
            if (output.dataType != DataType::BFLOAT16 || output.dims.size() != 3 ||
                output.dims[0] != bsz || output.dims[2] != headDim ||
                output.dims[1] < 0 || output.dims[1] > blocks) {
                return 0;
            }
            return output.dims[1];
        }

        static void AppendCompressedKVRows(Data &output, const Data &newRows,
                                           int bsz, int oldBlocks, int addBlocks, int headDim) {
            if (oldBlocks <= 0) {
                output.CopyFrom(newRows);
                return;
            }

            Data oldCpu, rowsCpu;
            const Data *oldPtr = &output;
            const Data *rowsPtr = &newRows;
            if (output.dataDevice != DataDevice::CPU) {
                oldCpu.CopyFrom(output);
                oldCpu.ToDevice(DataDevice::CPU);
                oldPtr = &oldCpu;
            }
            if (newRows.dataDevice != DataDevice::CPU) {
                rowsCpu.CopyFrom(newRows);
                rowsCpu.ToDevice(DataDevice::CPU);
                rowsPtr = &rowsCpu;
            }

            int blocks = oldBlocks + addBlocks;
            Data merged(DataType::BFLOAT16, {bsz, blocks, headDim});
            merged.Allocate(false);
            uint16_t *dst = (uint16_t*)merged.cpuData;
            const uint16_t *oldData = (const uint16_t*)oldPtr->cpuData;
            const uint16_t *newData = (const uint16_t*)rowsPtr->cpuData;
            for (int b = 0; b < bsz; b++) {
                memcpy(dst + (uint64_t)b * blocks * headDim,
                       oldData + (uint64_t)b * oldBlocks * headDim,
                       (uint64_t)oldBlocks * headDim * sizeof(uint16_t));
                memcpy(dst + ((uint64_t)b * blocks + oldBlocks) * headDim,
                       newData + (uint64_t)b * addBlocks * headDim,
                       (uint64_t)addBlocks * headDim * sizeof(uint16_t));
            }
            output.CopyFrom(merged);
        }

        static bool BuildCompressedKVFromRaw(WeightMap &weight, const std::string &prefix,
                                             const std::vector<float> &kv, const std::vector<float> &score,
                                             int bsz, int totalLen, int compressRatio,
                                             int headDim, int ropeDim, float ropeBase,
                                             float ropeFactor, int betaFast, int betaSlow,
                                             int originalSeqLen, Data &output) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4BuildCompressedKV");
            if (compressRatio <= 0 || totalLen < compressRatio) {
                return false;
            }
            int cutoff = totalLen - (totalLen % compressRatio);
            int blocks = cutoff / compressRatio;
            if (blocks <= 0) {
                return false;
            }
            bool overlap = (compressRatio == 4);
            int coff = overlap ? 2 : 1;
            int wideDim = coff * headDim;
            auto apePtr = ReadWeightFloatDataCached(weight[prefix + ".ape"]);
            const auto &ape = *apePtr;

            int reusableBlocks = GetReusableCompressedBlocks(output, bsz, blocks, headDim);
            if (reusableBlocks == blocks) {
                return true;
            }

            int addBlocks = blocks - reusableBlocks;
            std::vector<float> compressed;
            ComputeCompressedKVRangeCpu(kv, score, ape, bsz, totalLen, reusableBlocks, addBlocks,
                                        compressRatio, headDim, wideDim, overlap, compressed);

            Data newRows;
            FinalizeCompressedKVRows(weight, prefix, compressed, bsz, reusableBlocks, addBlocks,
                                     compressRatio, headDim, ropeDim, ropeBase, ropeFactor,
                                     betaFast, betaSlow, originalSeqLen, newRows);
            AppendCompressedKVRows(output, newRows, bsz, reusableBlocks, addBlocks, headDim);
            return true;
        }

        static void BuildDecodeKVData(const std::vector<float> &windowKV, const Data &compressedKV,
                                      int bsz, int windowSize, int compressedCount, int headDim,
                                      Data &output) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4BuildDecodeKV");
            std::vector<float> y((uint64_t)bsz * (windowSize + compressedCount) * headDim, 0.0f);
            for (int b = 0; b < bsz; b++) {
                memcpy(y.data() + (uint64_t)b * (windowSize + compressedCount) * headDim,
                       windowKV.data() + (uint64_t)b * windowSize * headDim,
                       (uint64_t)windowSize * headDim * sizeof(float));
            }
            if (compressedCount > 0) {
                auto cv = ReadFloatData(compressedKV);
                for (int b = 0; b < bsz; b++) {
                    memcpy(y.data() + ((uint64_t)b * (windowSize + compressedCount) + windowSize) * headDim,
                           cv.data() + (uint64_t)b * compressedCount * headDim,
                           (uint64_t)compressedCount * headDim * sizeof(float));
                }
            }
            WriteFloatData(y, {bsz, windowSize + compressedCount, headDim}, output, DataType::BFLOAT16);
        }

        static void SparseAttentionReference(Data &q, Data &kv, Data &attnSink, int windowSize,
                                             int ropeDim, float ropeBase, int startPos, float softmaxScale,
                                             Data &output, int compressRatio = 0, int originalSeqLen = 0,
                                             float ropeFactor = 1.0f, int betaFast = 32, int betaSlow = 1,
                                             int prefixLen = 0) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4SparseAttention");
#ifdef USE_CUDA
            if (!EnvFlagEnabled("FASTLLM_DSV4_DISABLE_CUDA_SPARSE_PREFILL") &&
                DeepSeekV4PreferCuda() && q.dims.size() == 4 && kv.dims.size() == 3) {
                Data qCuda, kvCuda;
                const Data *qForCuda = &q;
                const Data *kvForCuda = &kv;
                if (q.dataDevice != DataDevice::CUDA) {
                    qCuda.CopyFrom(q);
                    qCuda.ToDevice(DataDevice::CUDA);
                    qForCuda = &qCuda;
                }
                if (kv.dataDevice != DataDevice::CUDA) {
                    kvCuda.CopyFrom(kv);
                    kvCuda.ToDevice(DataDevice::CUDA);
                    kvForCuda = &kvCuda;
                }
                attnSink.ToDevice(DataDevice::CUDA);
                if (FastllmCudaDeepSeekV4SparseAttentionPrefill(
                        *qForCuda, *kvForCuda, attnSink, windowSize, startPos, compressRatio,
                        ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow,
                        softmaxScale, output, prefixLen)) {
                    return;
                }
            }
#endif
            auto qv = ReadFloatData(q);
            auto kvv = ReadFloatData(kv);
            auto sinkPtr = ReadWeightFloatDataCached(attnSink);
            const auto &sink = *sinkPtr;
            int bsz = q.dims[0], seqlen = q.dims[1], heads = q.dims[2], dim = q.dims[3];
            int realPrefixLen = std::max(0, std::min(prefixLen, kv.dims[1] - seqlen));
            int compressedStart = realPrefixLen + seqlen;
            int compressedCount = std::max(0, kv.dims[1] - compressedStart);
            int prefixStartPos = startPos - realPrefixLen;
            std::vector<float> out((uint64_t)bsz * seqlen * heads * dim, 0.0f);
            for (int b = 0; b < bsz; b++) {
                for (int s = 0; s < seqlen; s++) {
                    int liveWindow = std::min(windowSize, realPrefixLen + s + 1);
                    std::vector<int> idxs(liveWindow, -1);
                    int beginPos = startPos + s - liveWindow + 1;
                    for (int k = 0; k < liveWindow; k++) {
                        int pos = beginPos + k;
                        idxs[k] = (pos < startPos) ? (pos - prefixStartPos) : (realPrefixLen + pos - startPos);
                    }
                    if (compressRatio > 0) {
                        int availableCompressed = (startPos + s + 1) / compressRatio;
                        for (int k = 0; k < compressedCount; k++) {
                            idxs.push_back(k < availableCompressed ? compressedStart + k : -1);
                        }
                    }
                    std::vector<float> scores(idxs.size());
                    for (int h = 0; h < heads; h++) {
                        const float *qrow = qv.data() + (((uint64_t)b * seqlen + s) * heads + h) * dim;
                        float mx = -std::numeric_limits<float>::infinity();
                        for (int k = 0; k < (int)idxs.size(); k++) {
                            if (idxs[k] < 0) {
                                scores[k] = -std::numeric_limits<float>::infinity();
                                continue;
                            }
                            const float *kvrow = kvv.data() + ((uint64_t)b * kv.dims[1] + idxs[k]) * dim;
                            double dot = 0.0;
                            for (int d = 0; d < dim; d++) {
                                dot += (double)qrow[d] * kvrow[d];
                            }
                            scores[k] = (float)dot * softmaxScale;
                            mx = std::max(mx, scores[k]);
                        }
                        float safeMx = std::isfinite(mx) ? mx : 0.0f;
                        double denom = std::exp((double)sink[h] - safeMx);
                        for (int k = 0; k < (int)idxs.size(); k++) {
                            if (std::isfinite(scores[k])) {
                                denom += std::exp((double)scores[k] - safeMx);
                            }
                        }
                        float *orow = out.data() + (((uint64_t)b * seqlen + s) * heads + h) * dim;
                        for (int k = 0; k < (int)idxs.size(); k++) {
                            if (!std::isfinite(scores[k])) {
                                continue;
                            }
                            float w = (float)(std::exp((double)scores[k] - safeMx) / std::max(denom, 1e-30));
                            const float *kvrow = kvv.data() + ((uint64_t)b * kv.dims[1] + idxs[k]) * dim;
                            for (int d = 0; d < dim; d++) {
                                orow[d] += w * kvrow[d];
                            }
                        }
                    }
                }
            }
            ApplyRotaryReference(out, {bsz, seqlen, heads, dim}, ropeDim, ropeBase, startPos, true,
                                 originalSeqLen, ropeFactor, betaFast, betaSlow);
            WriteFloatData(out, {bsz, seqlen, heads, dim}, output, DataType::BFLOAT16);
        }

        static void SparseAttentionDecodeReference(Data &q, Data &cacheKV, Data &attnSink,
                                                   int windowSize, int startPos, int compressedCount,
                                                   int ropeDim, float ropeBase, float softmaxScale,
                                                   Data &output, int originalSeqLen = 0,
                                                   float ropeFactor = 1.0f, int betaFast = 32, int betaSlow = 1) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4SparseDecode");
            auto qv = ReadFloatData(q);
            auto kvv = ReadFloatData(cacheKV);
            auto sinkPtr = ReadWeightFloatDataCached(attnSink);
            const auto &sink = *sinkPtr;
            int bsz = q.dims[0], heads = q.dims[2], dim = q.dims[3];
            std::vector<float> out((uint64_t)bsz * heads * dim, 0.0f);
            std::vector<int> idxs;
            if (startPos >= windowSize - 1) {
                int pos = startPos % windowSize;
                for (int i = pos + 1; i < windowSize; i++) {
                    idxs.push_back(i);
                }
                for (int i = 0; i <= pos; i++) {
                    idxs.push_back(i);
                }
            } else {
                for (int i = 0; i <= startPos; i++) {
                    idxs.push_back(i);
                }
            }
            for (int i = 0; i < compressedCount; i++) {
                idxs.push_back(windowSize + i);
            }

            for (int b = 0; b < bsz; b++) {
                std::vector<float> scores(idxs.size());
                for (int h = 0; h < heads; h++) {
                    const float *qrow = qv.data() + ((uint64_t)b * heads + h) * dim;
                    float mx = -std::numeric_limits<float>::infinity();
                    for (int k = 0; k < (int)idxs.size(); k++) {
                        const float *kvrow = kvv.data() + ((uint64_t)b * cacheKV.dims[1] + idxs[k]) * dim;
                        double dot = 0.0;
                        for (int d = 0; d < dim; d++) {
                            dot += (double)qrow[d] * kvrow[d];
                        }
                        scores[k] = (float)dot * softmaxScale;
                        mx = std::max(mx, scores[k]);
                    }
                    float safeMx = std::isfinite(mx) ? mx : 0.0f;
                    double denom = std::exp((double)sink[h] - safeMx);
                    for (float score : scores) {
                        denom += std::exp((double)score - safeMx);
                    }
                    float *orow = out.data() + ((uint64_t)b * heads + h) * dim;
                    for (int k = 0; k < (int)idxs.size(); k++) {
                        float w = (float)(std::exp((double)scores[k] - safeMx) / std::max(denom, 1e-30));
                        const float *kvrow = kvv.data() + ((uint64_t)b * cacheKV.dims[1] + idxs[k]) * dim;
                        for (int d = 0; d < dim; d++) {
                            orow[d] += w * kvrow[d];
                        }
                    }
                }
            }
            ApplyRotaryReference(out, {bsz, 1, heads, dim}, ropeDim, ropeBase, startPos, true,
                                 originalSeqLen, ropeFactor, betaFast, betaSlow);
            WriteFloatData(out, {bsz, 1, heads, dim}, output, DataType::BFLOAT16);
        }

        static void SparseAttentionDecodeCachedReference(Data &q,
                                                         const std::vector<float> &windowKV,
                                                         const Data *windowKVData,
                                                         const Data &compressedKV,
                                                         Data &attnSink,
                                                         int windowSize, int startPos, int compressedCount,
                                                         int ropeDim, float ropeBase, float softmaxScale,
                                                         Data &output, int originalSeqLen = 0,
                                                         float ropeFactor = 1.0f, int betaFast = 32, int betaSlow = 1) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4SparseDecodeCached");
#ifdef USE_CUDA
            if (!EnvFlagEnabled("FASTLLM_DSV4_DISABLE_CUDA_SPARSE_DECODE") && q.dims[1] == 1) {
                Data qCuda, compressedCuda;
                const Data *qForCuda = &q;
                if (q.dataDevice != DataDevice::CUDA) {
                    qCuda.CopyFrom(q);
                    qCuda.ToDevice(DataDevice::CUDA);
                    qForCuda = &qCuda;
                }
                const Data *compressedForCuda = &compressedKV;
                if (compressedCount > 0 && compressedKV.dataDevice != DataDevice::CUDA) {
                    compressedCuda.CopyFrom(compressedKV);
                    compressedCuda.ToDevice(DataDevice::CUDA);
                    compressedForCuda = &compressedCuda;
                }
                attnSink.ToDevice(DataDevice::CUDA);
                if (FastllmCudaDeepSeekV4SparseAttentionDecodeCached(*qForCuda, windowKVData, windowKV.data(),
                                                                     *compressedForCuda,
                                                                     attnSink, windowSize, startPos,
                                                                     compressedCount, ropeDim, ropeBase,
                                                                     originalSeqLen, ropeFactor,
                                                                     betaFast, betaSlow, softmaxScale, output)) {
                    return;
                }
                if (windowKVData != nullptr && windowKVData->dataDevice == DataDevice::CUDA) {
                    // windowKVData on CUDA may be out of sync with the CPU windowKV vector
                    // (e.g. when an earlier decode step updated only the CPU side).
                    // Move to CPU so the retry below re-uploads the fresh CPU data.
                    const_cast<Data *>(windowKVData)->ToDevice(DataDevice::CPU);
                    // Retry with the now-CPU windowKVData — the wrapper will fall through
                    // to the CPU windowKV pointer and upload it to a temp CUDA buffer.
                    if (FastllmCudaDeepSeekV4SparseAttentionDecodeCached(*qForCuda, windowKVData, windowKV.data(),
                                                                         *compressedForCuda,
                                                                         attnSink, windowSize, startPos,
                                                                         compressedCount, ropeDim, ropeBase,
                                                                         originalSeqLen, ropeFactor,
                                                                         betaFast, betaSlow, softmaxScale, output)) {
                        return;
                    }
                }
            }
#endif
            auto qv = ReadFloatData(q);
            std::vector<float> compressed;
            if (compressedCount > 0) {
                compressed = ReadFloatData(compressedKV);
            }
            auto sinkPtr = ReadWeightFloatDataCached(attnSink);
            const auto &sink = *sinkPtr;
            int bsz = q.dims[0], heads = q.dims[2], dim = q.dims[3];
            std::vector<float> out((uint64_t)bsz * heads * dim, 0.0f);
            std::vector<int> idxs;
            if (startPos >= windowSize - 1) {
                int pos = startPos % windowSize;
                for (int i = pos + 1; i < windowSize; i++) {
                    idxs.push_back(i);
                }
                for (int i = 0; i <= pos; i++) {
                    idxs.push_back(i);
                }
            } else {
                for (int i = 0; i <= startPos; i++) {
                    idxs.push_back(i);
                }
            }
            for (int i = 0; i < compressedCount; i++) {
                idxs.push_back(windowSize + i);
            }

            auto getKVRow = [&](int b, int idx) -> const float* {
                if (idx < windowSize) {
                    return windowKV.data() + ((uint64_t)b * windowSize + idx) * dim;
                }
                return compressed.data() + ((uint64_t)b * compressedCount + (idx - windowSize)) * dim;
            };

            for (int b = 0; b < bsz; b++) {
                std::vector<float> scores(idxs.size());
                for (int h = 0; h < heads; h++) {
                    const float *qrow = qv.data() + ((uint64_t)b * heads + h) * dim;
                    float mx = -std::numeric_limits<float>::infinity();
                    for (int k = 0; k < (int)idxs.size(); k++) {
                        const float *kvrow = getKVRow(b, idxs[k]);
                        double dot = 0.0;
                        for (int d = 0; d < dim; d++) {
                            dot += (double)qrow[d] * kvrow[d];
                        }
                        scores[k] = (float)dot * softmaxScale;
                        mx = std::max(mx, scores[k]);
                    }
                    float safeMx = std::isfinite(mx) ? mx : 0.0f;
                    double denom = std::exp((double)sink[h] - safeMx);
                    for (float score : scores) {
                        denom += std::exp((double)score - safeMx);
                    }
                    float *orow = out.data() + ((uint64_t)b * heads + h) * dim;
                    for (int k = 0; k < (int)idxs.size(); k++) {
                        float w = (float)(std::exp((double)scores[k] - safeMx) / std::max(denom, 1e-30));
                        const float *kvrow = getKVRow(b, idxs[k]);
                        for (int d = 0; d < dim; d++) {
                            orow[d] += w * kvrow[d];
                        }
                    }
                }
            }
            ApplyRotaryReference(out, {bsz, 1, heads, dim}, ropeDim, ropeBase, startPos, true,
                                 originalSeqLen, ropeFactor, betaFast, betaSlow);
            WriteFloatData(out, {bsz, 1, heads, dim}, output, DataType::BFLOAT16);
        }

        static bool CompressKVReference(WeightMap &weight, const std::string &prefix, const Data &x,
                                        int compressRatio, int headDim, int ropeDim, float ropeBase,
                                        float ropeFactor, int betaFast, int betaSlow, int originalSeqLen,
                                        int startPos, Data &output) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4CompressKV");
            if (startPos != 0 || compressRatio <= 0) {
                return false;
            }
            int bsz = x.dims[0], seqlen = x.dims[1];
            if (seqlen < compressRatio) {
                return false;
            }
            int cutoff = seqlen - (seqlen % compressRatio);
            if (cutoff <= 0) {
                return false;
            }
            int blocks = cutoff / compressRatio;
            bool overlap = (compressRatio == 4);
            int coff = overlap ? 2 : 1;

            Data kvData, scoreData;
            Linear((Data&)x, weight[prefix + ".wkv.weight"], Data(), kvData);
            Linear((Data&)x, weight[prefix + ".wgate.weight"], Data(), scoreData);
            auto kv = ReadFloatData(kvData);
            auto score = ReadFloatData(scoreData);
            auto apePtr = ReadWeightFloatDataCached(weight[prefix + ".ape"]);
            const auto &ape = *apePtr;

            int wideDim = coff * headDim;
            std::vector<float> compressed;
            ComputeCompressedKVRangeCpu(kv, score, ape, bsz, seqlen, 0, blocks,
                                        compressRatio, headDim, wideDim, overlap, compressed);
            FinalizeCompressedKVRows(weight, prefix, compressed, bsz, 0, blocks,
                                     compressRatio, headDim, ropeDim, ropeBase, ropeFactor,
                                     betaFast, betaSlow, originalSeqLen, output);
            return true;
        }

        static void ConcatSeqReference(const Data &a, const Data &b, Data &output) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4ConcatSeq");
            auto av = ReadFloatData(a);
            auto bv = ReadFloatData(b);
            int bsz = a.dims[0], aSeq = a.dims[1], bSeq = b.dims[1], dim = a.dims[2];
            std::vector<float> y((uint64_t)bsz * (aSeq + bSeq) * dim);
            for (int batch = 0; batch < bsz; batch++) {
                memcpy(y.data() + (uint64_t)batch * (aSeq + bSeq) * dim,
                       av.data() + (uint64_t)batch * aSeq * dim,
                       (uint64_t)aSeq * dim * sizeof(float));
                memcpy(y.data() + ((uint64_t)batch * (aSeq + bSeq) + aSeq) * dim,
                       bv.data() + (uint64_t)batch * bSeq * dim,
                       (uint64_t)bSeq * dim * sizeof(float));
            }
            WriteFloatData(y, {bsz, aSeq + bSeq, dim}, output, a.dataType);
        }

        struct WoAReferenceOp : MultiThreadBaseOp {
            const std::vector<float> *ov;
            const std::vector<float> *wv;
            float *y;
            int st, end;
            int bsz, seqlen, heads, headDim, groups, oRank, headsPerGroup, groupDim;

            WoAReferenceOp(const std::vector<float> *ov, const std::vector<float> *wv, float *y,
                           int st, int end, int bsz, int seqlen, int heads, int headDim,
                           int groups, int oRank)
                : ov(ov), wv(wv), y(y), st(st), end(end), bsz(bsz), seqlen(seqlen),
                  heads(heads), headDim(headDim), groups(groups), oRank(oRank) {
                headsPerGroup = heads / groups;
                groupDim = headsPerGroup * headDim;
            }

            void Run() override {
                const float *ovData = ov->data();
                const float *wvData = wv->data();
                for (int idx = st; idx < end; idx++) {
                    int r = idx % oRank;
                    int tmp = idx / oRank;
                    int g = tmp % groups;
                    tmp /= groups;
                    int s = tmp % seqlen;
                    int b = tmp / seqlen;
                    const float *w = wvData + ((uint64_t)g * oRank + r) * groupDim;
                    double v = 0.0;
                    int d = 0;
                    for (int hh = 0; hh < headsPerGroup; hh++) {
                        const float *src = ovData + (((uint64_t)b * seqlen + s) * heads +
                                                     g * headsPerGroup + hh) * headDim;
                        for (int localD = 0; localD < headDim; localD++, d++) {
                            v += (double)src[localD] * w[d];
                        }
                    }
                    y[idx] = (float)v;
                }
            }
        };

        static void WoAReference(Data &o, Data &woA, int groups, int oRank, Data &output) {
            auto ov = ReadFloatData(o);
            auto wvPtr = ReadWeightFloatDataCached(woA);
            const auto &wv = *wvPtr;
            int bsz = o.dims[0], seqlen = o.dims[1], heads = o.dims[2], headDim = o.dims[3];
            int headsPerGroup = heads / groups;
            int groupDim = headsPerGroup * headDim;
            std::vector<float> y((uint64_t)bsz * seqlen * groups * oRank, 0.0f);
            int total = bsz * seqlen * groups * oRank;
            auto *pool = GetAlivePool();
            int threadNum = std::min((int)pool->threads.size(), total);
            if (threadNum <= 1 || total < 1024) {
                WoAReferenceOp(&ov, &wv, y.data(), 0, total, bsz, seqlen, heads, headDim, groups, oRank).Run();
            } else {
                std::vector<WoAReferenceOp*> ops;
                int per = total / threadNum;
                int cur = 0;
                for (int i = 0; i < threadNum; i++) {
                    int end = (i == threadNum - 1) ? total : cur + per;
                    ops.push_back(new WoAReferenceOp(&ov, &wv, y.data(), cur, end,
                                                     bsz, seqlen, heads, headDim, groups, oRank));
                    cur = end;
                }
                for (int i = 0; i < (int)ops.size(); i++) {
                    pool->PushOp(i, ops[i]);
                }
                for (int i = 0; i < (int)ops.size(); i++) {
                    pool->Wait(i);
                    delete ops[i];
                }
            }
            WriteFloatData(y, {bsz, seqlen, groups * oRank}, output, DataType::BFLOAT16);
        }

        static bool WoACudaIfAvailable(Data &o, Data &woA, int groups, int oRank, Data &output) {
#ifdef USE_CUDA
            if (EnvFlagEnabled("FASTLLM_DSV4_DISABLE_CUDA_WOA_HCPOST")) {
                return false;
            }
            if (o.dims.size() != 4 || groups <= 0 || oRank <= 0) {
                return false;
            }
            int heads = o.dims[2], headDim = o.dims[3];
            if (heads % groups != 0 || woA.Count(0) != (uint64_t)groups * oRank * (heads / groups) * headDim) {
                return false;
            }
            o.ToDevice(DataDevice::CUDA);
            woA.ToDevice(DataDevice::CUDA);
            return FastllmCudaDeepSeekV4WoA(o, woA, groups, oRank, output);
#else
            (void)o;
            (void)woA;
            (void)groups;
            (void)oRank;
            (void)output;
            return false;
#endif
        }

        static void WoA(Data &o, Data &woA, int groups, int oRank, Data &output) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4WoA");
            if (WoACudaIfAvailable(o, woA, groups, oRank, output)) {
                return;
            }
            WoAReference(o, woA, groups, oRank, output);
        }

        static void RunExpertReference(WeightMap &weight, const std::string &prefix, const Data &x,
                                       float routeWeight, float swigluLimit, std::vector<float> &accum) {
            Data gate, up, gateup;
            int interDim = 0;
            if (weight.weight.find(prefix + ".gateup.weight") != weight.weight.end()) {
                Linear((Data&)x, weight[prefix + ".gateup.weight"], Data(), gateup);
                auto gv = ReadFloatData(gateup);
                interDim = gateup.dims.back() / 2;
                std::vector<float> act(interDim);
                for (int i = 0; i < interDim; i++) {
                    float g = gv[i];
                    float u = gv[interDim + i];
                    if (swigluLimit > 0.0f) {
                        g = std::min(g, swigluLimit);
                        u = std::max(-swigluLimit, std::min(u, swigluLimit));
                    }
                    act[i] = (g * SigmoidFloat(g)) * u * routeWeight;
                }
                WriteFloatData(act, {1, interDim}, gate, DataType::BFLOAT16);
            } else {
                Linear((Data&)x, weight[prefix + ".w1.weight"], Data(), gate);
                Linear((Data&)x, weight[prefix + ".w3.weight"], Data(), up);
                auto gv = ReadFloatData(gate);
                auto uv = ReadFloatData(up);
                interDim = (int)gv.size();
                std::vector<float> act(interDim);
                for (int i = 0; i < interDim; i++) {
                    float g = gv[i], u = uv[i];
                    if (swigluLimit > 0.0f) {
                        g = std::min(g, swigluLimit);
                        u = std::max(-swigluLimit, std::min(u, swigluLimit));
                    }
                    act[i] = (g * SigmoidFloat(g)) * u * routeWeight;
                }
                WriteFloatData(act, {1, interDim}, gate, DataType::BFLOAT16);
            }
            Data down;
            Linear(gate, weight[prefix + ".w2.weight"], Data(), down);
            auto dv = ReadFloatData(down);
            for (int i = 0; i < (int)accum.size(); i++) {
                accum[i] += dv[i];
            }
        }

        static void BuildMoERoutingData(WeightMap &weight, const std::string &prefix, const Data &x,
                                        const std::vector<int> &inputIds, int nRoutedExperts,
                                        int topk, const std::string &scoreFunc, float routeScale,
                                        Data &expertIndex, Data &expertScore,
                                        DeepSeekV4LayerProfile *profile = nullptr, bool profileSync = false) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4RouteScore");
            if (profile != nullptr) {
                ProfileSyncIfNeeded(profileSync);
            }
            double detailLastMs = NowMs();
            auto detailLap = [&](double &bucket) {
                if (profile == nullptr) {
                    return;
                }
                ProfileSyncIfNeeded(profileSync);
                double now = NowMs();
                bucket += now - detailLastMs;
                detailLastMs = now;
            };

            Data xFloat, routerLogits;
            ToDataType(x, xFloat, DataType::FLOAT32);
            Linear(xFloat, weight[prefix + ".gate.weight"], Data(), routerLogits, true);
            detailLap(profile != nullptr ? profile->routeGate : detailLastMs);

#ifdef USE_CUDA
            bool hashRoutingForCuda = weight.weight.find(prefix + ".gate.tid2eid") != weight.weight.end();
            if (!EnvFlagEnabled("FASTLLM_DSV4_DISABLE_CUDA_ROUTE") && hashRoutingForCuda &&
                routerLogits.dataDevice == DataDevice::CUDA && routerLogits.dataType == DataType::FLOAT32 &&
                inputIds.size() >= (size_t)x.dims[0]) {
                int scoreFuncMode = scoreFunc == "softmax" ? 0 : (scoreFunc == "sigmoid" ? 1 : 2);
                if (FastllmCudaDeepSeekV4HashRouteScore(routerLogits, weight[prefix + ".gate.tid2eid"],
                                                        inputIds.data(), x.dims[0], topk,
                                                        scoreFuncMode, routeScale,
                                                        expertIndex, expertScore)) {
                    detailLap(profile != nullptr ? profile->routeScore : detailLastMs);
                    return;
                }
            }
            if (!EnvFlagEnabled("FASTLLM_DSV4_DISABLE_CUDA_ROUTE") && !hashRoutingForCuda &&
                routerLogits.dataDevice == DataDevice::CUDA && routerLogits.dataType == DataType::FLOAT32) {
                int scoreFuncMode = scoreFunc == "softmax" ? 0 : (scoreFunc == "sigmoid" ? 1 : 2);
                Data *gateBiasData = nullptr;
                if (weight.weight.find(prefix + ".gate.bias") != weight.weight.end()) {
                    gateBiasData = &weight[prefix + ".gate.bias"];
                    gateBiasData->ToDevice(DataDevice::CUDA);
                }
                if (FastllmCudaDeepSeekV4RouteScoreTransform(routerLogits, scoreFuncMode)) {
                    int tokens = x.dims[0];
                    if (!PrepareCudaData(expertIndex, DataType::INT32, {tokens, topk}) ||
                        !PrepareCudaData(expertScore, DataType::FLOAT32, {tokens, topk})) {
                        ErrorInFastLLM("DeepSeekV4RouteScore CUDA error: failed to allocate route outputs.");
                    }
                    bool needNorm = scoreFunc != "softmax";
                    if (FastllmCudaSelectExpert(routerLogits, gateBiasData, expertIndex, expertScore,
                                                topk, needNorm, routeScale)) {
                        detailLap(profile != nullptr ? profile->routeScore : detailLastMs);
                        return;
                    }
                }
            }
#endif
            auto rawScores = ReadFloatData(routerLogits);

            bool hashRouting = weight.weight.find(prefix + ".gate.tid2eid") != weight.weight.end();
            std::shared_ptr<const std::vector<float>> tid2eidPtr, gateBiasPtr;
            const std::vector<float> *tid2eid = nullptr;
            const std::vector<float> *gateBias = nullptr;
            if (hashRouting) {
                tid2eidPtr = ReadWeightFloatDataCached(weight[prefix + ".gate.tid2eid"]);
                tid2eid = tid2eidPtr.get();
            } else if (weight.weight.find(prefix + ".gate.bias") != weight.weight.end()) {
                gateBiasPtr = ReadWeightFloatDataCached(weight[prefix + ".gate.bias"]);
                gateBias = gateBiasPtr.get();
            }

            int tokens = x.dims[0];
            std::vector<int> indices((uint64_t)tokens * topk);
            std::vector<float> weights((uint64_t)tokens * topk);

            for (int t = 0; t < tokens; t++) {
                std::vector<float> originalScores(nRoutedExperts);
                if (scoreFunc == "softmax") {
                    float mx = -std::numeric_limits<float>::infinity();
                    for (int e = 0; e < nRoutedExperts; e++) {
                        mx = std::max(mx, rawScores[(uint64_t)t * nRoutedExperts + e]);
                    }
                    double sum = 0.0;
                    for (int e = 0; e < nRoutedExperts; e++) {
                        double v = std::exp((double)rawScores[(uint64_t)t * nRoutedExperts + e] - mx);
                        originalScores[e] = (float)v;
                        sum += v;
                    }
                    for (int e = 0; e < nRoutedExperts; e++) {
                        originalScores[e] /= (float)sum;
                    }
                } else {
                    for (int e = 0; e < nRoutedExperts; e++) {
                        float raw = rawScores[(uint64_t)t * nRoutedExperts + e];
                        originalScores[e] = (scoreFunc == "sigmoid") ? SigmoidFloat(raw) : std::sqrt(SoftplusFloat(raw));
                    }
                }

                std::vector<int> curIndices(topk);
                if (hashRouting) {
                    for (int k = 0; k < topk; k++) {
                        curIndices[k] = (int)((*tid2eid)[(uint64_t)inputIds[t] * topk + k] + 0.5f);
                    }
                } else {
                    std::vector<float> selectScores = originalScores;
                    if (gateBias != nullptr) {
                        for (int e = 0; e < nRoutedExperts; e++) {
                            selectScores[e] += (*gateBias)[e];
                        }
                    }
                    for (int k = 0; k < topk; k++) {
                        int best = 0;
                        float bestScore = -std::numeric_limits<float>::infinity();
                        for (int e = 0; e < nRoutedExperts; e++) {
                            if (selectScores[e] > bestScore) {
                                bestScore = selectScores[e];
                                best = e;
                            }
                        }
                        curIndices[k] = best;
                        selectScores[best] = -std::numeric_limits<float>::infinity();
                    }
                }

                float sum = 0.0f;
                for (int k = 0; k < topk; k++) {
                    sum += originalScores[curIndices[k]];
                }
                for (int k = 0; k < topk; k++) {
                    float v = originalScores[curIndices[k]];
                    if (scoreFunc != "softmax") {
                        v = v / sum;
                    }
                    indices[(uint64_t)t * topk + k] = curIndices[k];
                    weights[(uint64_t)t * topk + k] = v * routeScale;
                }
            }

            WriteIntData(indices, {tokens, topk}, expertIndex);
            WriteFloatData(weights, {tokens, topk}, expertScore, DataType::FLOAT32);
            detailLap(profile != nullptr ? profile->routeScore : detailLastMs);
        }

        static void MoEReference(WeightMap &weight, const std::string &prefix, const Data &x,
                                 const std::vector<int> &inputIds, int nRoutedExperts,
                                 int topk, const std::string &scoreFunc, float routeScale,
                                 float swigluLimit, Data &output) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4MoEReference");
            Data xFloat, routerLogits;
            ToDataType(x, xFloat, DataType::FLOAT32);
            Linear(xFloat, weight[prefix + ".gate.weight"], Data(), routerLogits, true);
            auto rawScores = ReadFloatData(routerLogits);
            bool hashRouting = weight.weight.find(prefix + ".gate.tid2eid") != weight.weight.end();
            std::shared_ptr<const std::vector<float>> tid2eidPtr, gateBiasPtr;
            const std::vector<float> *tid2eid = nullptr;
            const std::vector<float> *gateBias = nullptr;
            if (hashRouting) {
                tid2eidPtr = ReadWeightFloatDataCached(weight[prefix + ".gate.tid2eid"]);
                tid2eid = tid2eidPtr.get();
            } else if (weight.weight.find(prefix + ".gate.bias") != weight.weight.end()) {
                gateBiasPtr = ReadWeightFloatDataCached(weight[prefix + ".gate.bias"]);
                gateBias = gateBiasPtr.get();
            }
            int tokens = x.dims[0] * x.dims[1];
            int dim = x.dims.back();
            std::vector<float> y((uint64_t)tokens * dim, 0.0f);
            for (int t = 0; t < tokens; t++) {
                std::vector<float> originalScores(nRoutedExperts);
                if (scoreFunc == "softmax") {
                    float mx = -std::numeric_limits<float>::infinity();
                    for (int e = 0; e < nRoutedExperts; e++) {
                        mx = std::max(mx, rawScores[(uint64_t)t * nRoutedExperts + e]);
                    }
                    double sum = 0.0;
                    for (int e = 0; e < nRoutedExperts; e++) {
                        double v = std::exp((double)rawScores[(uint64_t)t * nRoutedExperts + e] - mx);
                        originalScores[e] = (float)v;
                        sum += v;
                    }
                    for (int e = 0; e < nRoutedExperts; e++) {
                        originalScores[e] /= (float)sum;
                    }
                } else {
                    for (int e = 0; e < nRoutedExperts; e++) {
                        float raw = rawScores[(uint64_t)t * nRoutedExperts + e];
                        if (scoreFunc == "sigmoid") {
                            originalScores[e] = SigmoidFloat(raw);
                        } else {
                            originalScores[e] = std::sqrt(SoftplusFloat(raw));
                        }
                    }
                }

                std::vector<int> indices(topk);
                if (hashRouting) {
                    for (int k = 0; k < topk; k++) {
                        indices[k] = (int)((*tid2eid)[(uint64_t)inputIds[t] * topk + k] + 0.5f);
                    }
                } else {
                    std::vector<float> selectScores = originalScores;
                    if (gateBias != nullptr) {
                        for (int e = 0; e < nRoutedExperts; e++) {
                            selectScores[e] += (*gateBias)[e];
                        }
                    }
                    for (int k = 0; k < topk; k++) {
                        int best = 0;
                        float bestScore = -std::numeric_limits<float>::infinity();
                        for (int e = 0; e < nRoutedExperts; e++) {
                            if (selectScores[e] > bestScore) {
                                bestScore = selectScores[e];
                                best = e;
                            }
                        }
                        indices[k] = best;
                        selectScores[best] = -std::numeric_limits<float>::infinity();
                    }
                }
                std::vector<float> weights(topk);
                float sum = 0.0f;
                for (int k = 0; k < topk; k++) {
                    float v = originalScores[indices[k]];
                    weights[k] = v;
                    sum += v;
                }
                if (scoreFunc != "softmax") {
                    for (int k = 0; k < topk; k++) {
                        weights[k] = weights[k] / sum * routeScale;
                    }
                } else {
                    for (int k = 0; k < topk; k++) {
                        weights[k] *= routeScale;
                    }
                }
                Data tokenX;
                Split(x, 1, t, t + 1, tokenX);
                tokenX.Reshape({1, dim});
                std::vector<float> accum(dim, 0.0f);
                for (int k = 0; k < topk; k++) {
                    RunExpertReference(weight, prefix + ".experts." + std::to_string(indices[k]), tokenX,
                                       weights[k], swigluLimit, accum);
                }
                RunExpertReference(weight, prefix + ".shared_experts", tokenX, 1.0f, 0.0f, accum);
                memcpy(y.data() + (uint64_t)t * dim, accum.data(), dim * sizeof(float));
            }
            WriteFloatData(y, x.dims, output, x.dataType);
        }

        static void HcHeadReference(const Data &x, Data &hcFn, Data &hcScale, Data &hcBase,
                                    int hcMult, float eps, float normEps, Data &output) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4HcHead");
            int bsz = x.dims[0], seqlen = x.dims[1], dim = x.dims[3];
            int flatDim = hcMult * dim;
            auto xv = ReadFloatData(x);
            auto fnPtr = ReadWeightFloatDataCached(hcFn);
            auto scalePtr = ReadWeightFloatDataCached(hcScale);
            auto basePtr = ReadWeightFloatDataCached(hcBase);
            const auto &fn = *fnPtr;
            const auto &scale = *scalePtr;
            const auto &base = *basePtr;
            std::vector<float> y((uint64_t)bsz * seqlen * dim, 0.0f);
            for (int t = 0; t < bsz * seqlen; t++) {
                const float *xrow = xv.data() + (uint64_t)t * flatDim;
                double ss = 0.0;
                for (int k = 0; k < flatDim; k++) {
                    ss += (double)xrow[k] * xrow[k];
                }
                float rsqrt = 1.0f / std::sqrt((float)(ss / flatDim) + normEps);
                for (int h = 0; h < hcMult; h++) {
                    const float *w = fn.data() + (uint64_t)h * flatDim;
                    double mix = 0.0;
                    for (int k = 0; k < flatDim; k++) {
                        mix += (double)xrow[k] * w[k];
                    }
                    float pre = SigmoidFloat((float)mix * rsqrt * scale[0] + base[h]) + eps;
                    for (int d = 0; d < dim; d++) {
                        y[(uint64_t)t * dim + d] += pre * xrow[(uint64_t)h * dim + d];
                    }
                }
            }
            WriteFloatData(y, {bsz, seqlen, dim}, output, x.dataType);
        }
    }

    static uint64_t DeepSeekV4TokenBlockHash(const std::vector<int> &tokens, int len, int blockSize) {
        uint64_t h = 1469598103934665603ULL;
        auto mix = [&](uint64_t v) {
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h *= 1099511628211ULL;
        };
        mix((uint64_t)blockSize);
        for (int i = 0; i < len; i++) {
            mix((uint64_t)(uint32_t)tokens[i]);
            if ((i + 1) % blockSize == 0) {
                mix(0xff51afd7ed558ccdULL ^ (uint64_t)((i + 1) / blockSize));
            }
        }
        return h;
    }

    static bool DeepSeekV4PrefixCacheDebugEnabled() {
        return EnvFlagEnabled("FASTLLM_DSV4_PREFIX_CACHE_DEBUG") ||
               EnvFlagEnabled("FASTLLM_DSV4_DEBUG_PREFIX_CACHE");
    }

    static bool DeepSeekV4PrefixCacheDisabled() {
        return EnvFlagEnabled("FASTLLM_DSV4_DISABLE_PREFIX_CACHE");
    }

    void DeepSeekV4HistoryCacheManager::SetMaxRecordNum(int maxRecordNum) {
        std::lock_guard<std::mutex> guard(this->locker);
        this->maxRecordNum = std::max(1, maxRecordNum);
    }

    void DeepSeekV4HistoryCacheManager::Record(const DeepSeekV4HistoryCacheMemory &memory) {
        if (memory.tokens <= 0 || memory.tokens % this->logicalBlockSize != 0 ||
            (int)memory.inputToken.size() != memory.tokens || memory.layers.empty()) {
            return;
        }
        std::lock_guard<std::mutex> guard(this->locker);
        int envMax = EnvInt("FASTLLM_DSV4_PREFIX_CACHE_MAX_RECORDS", this->maxRecordNum);
        this->maxRecordNum = std::max(1, envMax);

        auto old = this->memorys.find(memory.inputToken);
        if (old != this->memorys.end()) {
            old->second = memory;
            old->second.recordTimes++;
            old->second.flushTime = ++this->flushTime;
            this->blockIndex[old->second.blockHash] = old->second.inputToken;
            return;
        }

        while ((int)this->memorys.size() >= this->maxRecordNum) {
            auto eraseIt = this->memorys.end();
            long long minFlushTime = (1LL << 60);
            for (auto it = this->memorys.begin(); it != this->memorys.end(); ++it) {
                if (it->second.flushTime < minFlushTime) {
                    minFlushTime = it->second.flushTime;
                    eraseIt = it;
                }
            }
            if (eraseIt == this->memorys.end()) {
                break;
            }
            auto blockIt = this->blockIndex.find(eraseIt->second.blockHash);
            if (blockIt != this->blockIndex.end() && blockIt->second == eraseIt->second.inputToken) {
                this->blockIndex.erase(blockIt);
            }
            this->memorys.erase(eraseIt);
        }

        auto inserted = this->memorys.emplace(memory.inputToken, memory);
        inserted.first->second.recordTimes = 1;
        inserted.first->second.flushTime = ++this->flushTime;
        this->blockIndex[inserted.first->second.blockHash] = inserted.first->second.inputToken;
    }

    bool DeepSeekV4HistoryCacheManager::Get(const std::vector<int> &inputToken,
                                            DeepSeekV4HistoryCacheMemory &memory,
                                            int &hitLen) {
        hitLen = 0;
        if ((int)inputToken.size() <= this->logicalBlockSize) {
            return false;
        }
        std::lock_guard<std::mutex> guard(this->locker);
        if (this->memorys.empty()) {
            return false;
        }

        int maxAligned = ((int)inputToken.size() - 1) / this->logicalBlockSize * this->logicalBlockSize;
        for (int len = maxAligned; len >= this->logicalBlockSize; len -= this->logicalBlockSize) {
            uint64_t hash = DeepSeekV4TokenBlockHash(inputToken, len, this->logicalBlockSize);
            auto idxIt = this->blockIndex.find(hash);
            if (idxIt == this->blockIndex.end() || (int)idxIt->second.size() != len) {
                continue;
            }
            auto memIt = this->memorys.find(idxIt->second);
            if (memIt == this->memorys.end()) {
                continue;
            }
            bool match = true;
            for (int i = 0; i < len; i++) {
                if (inputToken[i] != memIt->second.inputToken[i]) {
                    match = false;
                    break;
                }
            }
            if (!match) {
                continue;
            }
            memIt->second.flushTime = ++this->flushTime;
            memory = memIt->second;
            hitLen = len;
            return true;
        }

        for (auto &it : this->memorys) {
            int len = (int)it.first.size();
            if (len <= hitLen || len > maxAligned) {
                continue;
            }
            bool match = true;
            for (int i = 0; i < len; i++) {
                if (inputToken[i] != it.first[i]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                hitLen = len;
                memory = it.second;
            }
        }
        if (hitLen > 0) {
            this->memorys[memory.inputToken].flushTime = ++this->flushTime;
            return true;
        }
        return false;
    }

    bool DeepSeekV4Model::RestoreHistoryCacheMemory(const DeepSeekV4HistoryCacheMemory &memory) {
        if (memory.tokens <= 0 || memory.layers.empty()) {
            return false;
        }
        this->decodeLayerCaches.clear();
        this->decodeLayerCaches.resize(memory.layers.size());
        for (int i = 0; i < (int)memory.layers.size(); i++) {
            const auto &src = memory.layers[i];
            auto &dst = this->decodeLayerCaches[i];
            dst.initialized = src.initialized;
            dst.bsz = src.bsz;
            dst.totalLen = src.totalLen;
            dst.headDim = src.headDim;
            dst.windowSize = src.windowSize;
            dst.compressRatio = src.compressRatio;
            dst.compressorWideDim = src.compressorWideDim;
            dst.windowKV = src.windowKV;
            dst.compressedBlocks = src.compressedBlocks;
            dst.compressedTokenBase = src.compressedTokenBase;
            dst.rawTailStartPos = src.rawTailStartPos;
            dst.compressorTailKV = src.compressorTailKV;
            dst.compressorTailScore = src.compressorTailScore;

            ResetData(dst.windowKVData);
            ResetData(dst.compressedKV);
            if (src.compressedBlocks > 0 && src.compressedKV.dims.size() >= 2) {
                dst.compressedKV.CopyFrom(src.compressedKV);
            }

            if (!src.compressorKVRaw.empty()) {
                dst.compressorKVRaw = src.compressorKVRaw;
                dst.compressorScoreRaw = src.compressorScoreRaw;
            } else if (src.compressRatio > 0 && src.compressorWideDim > 0 &&
                       !src.compressorTailKV.empty() && !src.compressorTailScore.empty()) {
                uint64_t rawSize = (uint64_t)src.bsz * src.totalLen * src.compressorWideDim;
                dst.compressorKVRaw.assign(rawSize, 0.0f);
                dst.compressorScoreRaw.assign(rawSize, 0.0f);
                int tailTokens = (int)(src.compressorTailKV.size() /
                                       ((uint64_t)std::max(1, src.bsz) * src.compressorWideDim));
                int tailStart = std::max(0, std::min(src.rawTailStartPos, src.totalLen));
                tailTokens = std::min(tailTokens, src.totalLen - tailStart);
                for (int b = 0; b < src.bsz; b++) {
                    memcpy(dst.compressorKVRaw.data() + ((uint64_t)b * src.totalLen + tailStart) * src.compressorWideDim,
                           src.compressorTailKV.data() + (uint64_t)b * tailTokens * src.compressorWideDim,
                           (uint64_t)tailTokens * src.compressorWideDim * sizeof(float));
                    memcpy(dst.compressorScoreRaw.data() + ((uint64_t)b * src.totalLen + tailStart) * src.compressorWideDim,
                           src.compressorTailScore.data() + (uint64_t)b * tailTokens * src.compressorWideDim,
                           (uint64_t)tailTokens * src.compressorWideDim * sizeof(float));
                }
            } else {
                dst.compressorKVRaw.clear();
                dst.compressorScoreRaw.clear();
            }

#ifdef USE_CUDA
            if (!EnvFlagEnabled("FASTLLM_DSV4_DISABLE_CUDA_SPARSE_DECODE") &&
                DeepSeekV4PreferCuda() && !dst.windowKV.empty() && dst.bsz > 0 &&
                dst.windowSize > 0 && dst.headDim > 0) {
                WriteFloatData(dst.windowKV, {dst.bsz, dst.windowSize, dst.headDim},
                               dst.windowKVData, DataType::FLOAT32);
                dst.windowKVData.ToDevice(DataDevice::CUDA);
            }
#endif
        }
        this->deepseekV4HistoryTokens = memory.inputToken;
        if (DeepSeekV4PrefixCacheDebugEnabled()) {
            printf("[fastllm-dsv4-prefix-cache] restore hit_len=%d blocks=%d layers=%d\n",
                   memory.tokens, memory.blockCount, (int)memory.layers.size());
            for (int i = 0; i < (int)this->decodeLayerCaches.size(); i++) {
                const auto &layer = this->decodeLayerCaches[i];
                printf("[fastllm-dsv4-prefix-cache]   layer=%02d ratio=%d total_len=%d compressed_blocks=%d window=%d raw_tail_start=%d tail_tokens=%d\n",
                       i, layer.compressRatio, layer.totalLen, layer.compressedBlocks,
                       (int)(layer.windowKV.size() / std::max(1, layer.headDim)),
                       layer.rawTailStartPos,
                       layer.compressorWideDim > 0 && layer.bsz > 0 ?
                           (int)(layer.compressorTailKV.size() / ((uint64_t)layer.bsz * layer.compressorWideDim)) : 0);
            }
            fflush(stdout);
        }
        return true;
    }

    void DeepSeekV4Model::RecordHistorySnapshot(const std::vector<int> &tokens, int totalLen) {
        if (!this->saveHistoryChat || DeepSeekV4PrefixCacheDisabled() ||
            totalLen <= 0 || totalLen % 256 != 0 || (int)tokens.size() < totalLen ||
            this->decodeLayerCaches.empty()) {
            return;
        }
        DeepSeekV4HistoryCacheMemory memory;
        memory.tokens = totalLen;
        memory.blockCount = totalLen / 256;
        memory.inputToken.assign(tokens.begin(), tokens.begin() + totalLen);
        memory.blockHash = DeepSeekV4TokenBlockHash(memory.inputToken, totalLen, 256);
        memory.layers.resize(this->decodeLayerCaches.size());
        bool storeFullRaw = EnvFlagEnabled("FASTLLM_DSV4_PREFIX_CACHE_FULL_RAW");
        for (int i = 0; i < (int)this->decodeLayerCaches.size(); i++) {
            const auto &src = this->decodeLayerCaches[i];
            auto &dst = memory.layers[i];
            if (!src.initialized || src.totalLen != totalLen) {
                return;
            }
            dst.initialized = src.initialized;
            dst.bsz = src.bsz;
            dst.totalLen = src.totalLen;
            dst.headDim = src.headDim;
            dst.windowSize = src.windowSize;
            dst.compressRatio = src.compressRatio;
            dst.compressorWideDim = src.compressorWideDim;
            dst.windowKV = src.windowKV;
            dst.compressedBlocks = src.compressedBlocks;
            dst.compressedTokenBase = src.compressedBlocks * std::max(1, src.compressRatio);
            dst.compressedKV = Data();
            if (src.compressedBlocks > 0 && src.compressedKV.dims.size() >= 2) {
                dst.compressedKV.CopyFrom(src.compressedKV);
                dst.compressedKV.ToDevice(DataDevice::CPU);
                dst.compressedKV.lockInCPU = true;
            }

            if (src.compressRatio > 0 && src.compressorWideDim > 0) {
                int tailTokens = src.compressRatio == 4 ? 8 : (src.compressRatio == 128 ? 128 : src.compressRatio);
                tailTokens = std::min(tailTokens, src.totalLen);
                dst.rawTailStartPos = src.totalLen - tailTokens;
                if (storeFullRaw) {
                    dst.compressorKVRaw = src.compressorKVRaw;
                    dst.compressorScoreRaw = src.compressorScoreRaw;
                }
                if (!src.compressorKVRaw.empty() && !src.compressorScoreRaw.empty()) {
                    dst.compressorTailKV.assign((uint64_t)src.bsz * tailTokens * src.compressorWideDim, 0.0f);
                    dst.compressorTailScore.assign((uint64_t)src.bsz * tailTokens * src.compressorWideDim, 0.0f);
                    for (int b = 0; b < src.bsz; b++) {
                        memcpy(dst.compressorTailKV.data() + (uint64_t)b * tailTokens * src.compressorWideDim,
                               src.compressorKVRaw.data() + ((uint64_t)b * src.totalLen + dst.rawTailStartPos) * src.compressorWideDim,
                               (uint64_t)tailTokens * src.compressorWideDim * sizeof(float));
                        memcpy(dst.compressorTailScore.data() + (uint64_t)b * tailTokens * src.compressorWideDim,
                               src.compressorScoreRaw.data() + ((uint64_t)b * src.totalLen + dst.rawTailStartPos) * src.compressorWideDim,
                               (uint64_t)tailTokens * src.compressorWideDim * sizeof(float));
                    }
                }
            }
        }
        this->deepseekV4HistoryCacheManager.Record(memory);
        if (DeepSeekV4PrefixCacheDebugEnabled()) {
            printf("[fastllm-dsv4-prefix-cache] record tokens=%d blocks=%d layers=%d residual_only=%d\n",
                   totalLen, memory.blockCount, (int)memory.layers.size(), storeFullRaw ? 0 : 1);
            fflush(stdout);
        }
    }

    bool DeepSeekV4Model::TryRestoreHistoryCache(std::vector<int> &inputTokens, int &cacheLen) {
        bool debugPrefixCache = DeepSeekV4PrefixCacheDebugEnabled();
        if (!this->saveHistoryChat) {
            if (debugPrefixCache) {
                printf("[fastllm-dsv4-prefix-cache] disabled: cache_history is off input_tokens=%d\n",
                       (int)inputTokens.size());
                fflush(stdout);
            }
            return false;
        }
        if (DeepSeekV4PrefixCacheDisabled()) {
            if (debugPrefixCache) {
                printf("[fastllm-dsv4-prefix-cache] disabled: FASTLLM_DSV4_DISABLE_PREFIX_CACHE input_tokens=%d\n",
                       (int)inputTokens.size());
                fflush(stdout);
            }
            return false;
        }
        if (inputTokens.size() <= 256) {
            if (debugPrefixCache) {
                printf("[fastllm-dsv4-prefix-cache] skip: input_tokens=%d <= logical_block=256\n",
                       (int)inputTokens.size());
                fflush(stdout);
            }
            return false;
        }
        DeepSeekV4HistoryCacheMemory memory;
        int hitLen = 0;
        if (!this->deepseekV4HistoryCacheManager.Get(inputTokens, memory, hitLen) || hitLen <= 0) {
            if (debugPrefixCache) {
                int alignedProbeLen = ((int)inputTokens.size() - 1) / 256 * 256;
                printf("[fastllm-dsv4-prefix-cache] miss input_tokens=%d aligned_probe=%d\n",
                       (int)inputTokens.size(), alignedProbeLen);
                fflush(stdout);
            }
            return false;
        }
        if (!this->RestoreHistoryCacheMemory(memory)) {
            return false;
        }
        inputTokens.erase(inputTokens.begin(), inputTokens.begin() + hitLen);
        cacheLen = hitLen;
        if (DeepSeekV4PrefixCacheDebugEnabled()) {
            printf("[fastllm-dsv4-prefix-cache] hit lcp_aligned=%d remaining=%d\n",
                   hitLen, (int)inputTokens.size());
            fflush(stdout);
        }
        return true;
    }

    void DeepSeekV4Model::TryRecordHistoryCache(const std::vector<int> &allTokens) {
        if (!this->saveHistoryChat || DeepSeekV4PrefixCacheDisabled() ||
            this->decodeLayerCaches.empty() || allTokens.empty()) {
            if (DeepSeekV4PrefixCacheDebugEnabled()) {
                printf("[fastllm-dsv4-prefix-cache] skip record: save=%d disabled=%d caches=%d tokens=%d\n",
                       this->saveHistoryChat ? 1 : 0, DeepSeekV4PrefixCacheDisabled() ? 1 : 0,
                       (int)this->decodeLayerCaches.size(), (int)allTokens.size());
                fflush(stdout);
            }
            return;
        }
        int totalLen = this->decodeLayerCaches[0].totalLen;
        if (totalLen > 0 && totalLen % 256 == 0 && (int)allTokens.size() >= totalLen) {
            this->RecordHistorySnapshot(allTokens, totalLen);
        } else if (DeepSeekV4PrefixCacheDebugEnabled()) {
            printf("[fastllm-dsv4-prefix-cache] skip record: total_len=%d aligned=%d all_tokens=%d\n",
                   totalLen, totalLen / 256 * 256, (int)allTokens.size());
            fflush(stdout);
        }
    }

    DeepSeekV4Model::DeepSeekV4Model() {
        this->model_type = "deepseek_v4";
        this->model_struct = "deepseek_v4";
        this->defaultChunkedPrefillSize = 4096;

        // V4 推荐 thinking 模式，需配合外部 chat_template；这里给一份最小默认值
        this->pre_prompt = "";
        this->user_role = "<|User|>";
        this->bot_role = "<|Assistant|>";
        this->history_sep = "";

        // 与 model.py 对齐：embed -> layers.X.attn / ffn -> head -> mtp.Z.*
        // 注意 V4 ckpt 的命名前缀直接是 layers / mtp / embed / head（无 model. 前缀）
        weight.embeddingNames.insert("embed.weight");
        weight.linearNames = {
            "head.weight",
            // attention 主权重
            "layers.*.attn.wq_a.weight", "layers.*.attn.wq_b.weight",
            "layers.*.attn.wkv.weight",
            "layers.*.attn.wo_a.weight", "layers.*.attn.wo_b.weight",
            // indexer / compressor 子权重
            "layers.*.attn.indexer.wq_b.weight",
            "layers.*.attn.indexer.weights_proj.weight",
            "layers.*.attn.indexer.compressor.wkv.weight",
            "layers.*.attn.indexer.compressor.wgate.weight",
            "layers.*.attn.compressor.wkv.weight",
            "layers.*.attn.compressor.wgate.weight",
            // moe gate / experts
            "layers.*.ffn.gate.weight",
            "layers.*.ffn.experts.*.w1.weight",
            "layers.*.ffn.experts.*.w2.weight",
            "layers.*.ffn.experts.*.w3.weight",
            "layers.*.ffn.shared_experts.w1.weight",
            "layers.*.ffn.shared_experts.w2.weight",
            "layers.*.ffn.shared_experts.w3.weight",
            // mtp 同构权重
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
            "mtp.*.e_proj.weight", "mtp.*.h_proj.weight",
        };
    }

    void DeepSeekV4Model::InitParams() {
        basellm::InitParams();

        // -------- 基础尺寸 --------
        block_cnt = GetIntWithFallback(this->weight, {"num_hidden_layers", "n_layers"}, block_cnt > 0 ? block_cnt : 1);
        embed_dim = GetIntWithFallback(this->weight, {"hidden_size", "dim"}, embed_dim > 0 ? embed_dim : 4096);
        num_attention_heads = GetIntWithFallback(this->weight, {"num_attention_heads", "n_heads"}, num_attention_heads > 0 ? num_attention_heads : 64);
        max_positions = GetIntWithFallback(this->weight, {"max_position_embeddings", "max_seq_len"}, max_positions > 0 ? max_positions : 4096);
        max_position_embeddings = max_positions;
        if (this->weight.dicts.find("rms_norm_eps") != this->weight.dicts.end()) {
            rms_norm_eps = atof(this->weight.dicts["rms_norm_eps"].c_str());
        }
        rms_norm_eps = GetFloatWithFallback(this->weight, {"rms_norm_eps", "norm_eps"}, rms_norm_eps);

        // num_attention_heads / num_key_value_heads / embed_dim / block_cnt 已由 basellm 解析
        if (this->weight.dicts.find("num_key_value_heads") != this->weight.dicts.end()) {
            num_key_value_heads = atoi(this->weight.dicts["num_key_value_heads"].c_str());
        } else {
            num_key_value_heads = 1;
        }
        if (this->weight.dicts.find("max_position_embeddings") != this->weight.dicts.end()) {
            max_positions = atoi(this->weight.dicts["max_position_embeddings"].c_str());
        }

        // -------- Attention 维度 --------
        q_lora_rank = GetIntWithFallback(this->weight, {"q_lora_rank"}, q_lora_rank);
        o_lora_rank = GetIntWithFallback(this->weight, {"o_lora_rank"}, o_lora_rank);
        o_groups = GetIntWithFallback(this->weight, {"o_groups"}, o_groups);
        head_dim_full = GetIntWithFallback(this->weight, {"head_dim"}, head_dim_full);
        qk_rope_head_dim = GetIntWithFallback(this->weight, {"qk_rope_head_dim", "rope_head_dim"}, qk_rope_head_dim);
        qk_nope_head_dim = head_dim_full - qk_rope_head_dim;
        head_dim = head_dim_full;
        rotary_dim = qk_rope_head_dim;

        if (this->weight.dicts.find("sliding_window") != this->weight.dicts.end()) {
            window_size = atoi(this->weight.dicts["sliding_window"].c_str());
        }

        // -------- Indexer --------
        if (this->weight.dicts.find("index_n_heads") != this->weight.dicts.end()) {
            index_n_heads = atoi(this->weight.dicts["index_n_heads"].c_str());
        }
        if (this->weight.dicts.find("index_head_dim") != this->weight.dicts.end()) {
            index_head_dim = atoi(this->weight.dicts["index_head_dim"].c_str());
        }
        if (this->weight.dicts.find("index_topk") != this->weight.dicts.end()) {
            index_topk = atoi(this->weight.dicts["index_topk"].c_str());
        }

        // -------- compress_ratios（数组） --------
        compress_ratios.clear();
        if (this->weight.dicts.find("compress_ratios") != this->weight.dicts.end()) {
            std::string err;
            auto j = json11::Json::parse(this->weight.dicts["compress_ratios"], err);
            if (j.is_array()) {
                for (auto &v : j.array_items()) {
                    compress_ratios.push_back(v.int_value());
                }
            }
        }
        if ((int)compress_ratios.size() < block_cnt) {
            compress_ratios.resize(block_cnt, 0);
        }

        // -------- MoE --------
        moe_intermediate_size = GetIntWithFallback(this->weight, {"moe_intermediate_size", "moe_inter_dim"}, moe_intermediate_size);
        n_shared_experts = GetIntWithFallback(this->weight, {"n_shared_experts"}, n_shared_experts);
        num_experts = GetIntWithFallback(this->weight, {"n_routed_experts", "num_experts"}, num_experts);
        num_experts_per_tok = GetIntWithFallback(this->weight, {"num_experts_per_tok", "n_activated_experts"}, num_experts_per_tok);
        norm_topk_prob = (this->weight.dicts.find("norm_topk_prob") != this->weight.dicts.end() &&
                          this->weight.dicts["norm_topk_prob"] == "true");
        routed_scaling_factor = GetFloatWithFallback(this->weight, {"routed_scaling_factor", "route_scale"}, routed_scaling_factor);
        scoring_func = GetStringWithFallback(this->weight, {"scoring_func", "score_func"}, scoring_func);
        if (this->weight.dicts.find("topk_method") != this->weight.dicts.end()) {
            topk_method = this->weight.dicts["topk_method"];
        }
        if (this->weight.dicts.find("swiglu_limit") != this->weight.dicts.end()) {
            swiglu_limit = atof(this->weight.dicts["swiglu_limit"].c_str());
        }

        // -------- Hash 路由 / MTP --------
        num_hash_layers = GetIntWithFallback(this->weight, {"num_hash_layers", "n_hash_layers"}, num_hash_layers);
        num_nextn_predict_layers = GetIntWithFallback(this->weight, {"num_nextn_predict_layers", "n_mtp_layers"}, num_nextn_predict_layers);

        // -------- Hyper-Connections --------
        hc_mult = GetIntWithFallback(this->weight, {"hc_mult"}, hc_mult);
        hc_sinkhorn_iters = GetIntWithFallback(this->weight, {"hc_sinkhorn_iters"}, hc_sinkhorn_iters);
        hc_eps = GetFloatWithFallback(this->weight, {"hc_eps"}, hc_eps);

        // -------- RoPE / YaRN --------
        rope_base = GetFloatWithFallback(this->weight, {"rope_theta"}, rope_base);
        compress_rope_theta = GetFloatWithFallback(this->weight, {"compress_rope_theta"}, compress_rope_theta);
        if (this->weight.dicts.find("rope_scaling.type") != this->weight.dicts.end()) {
            rope_scaling_type = this->weight.dicts["rope_scaling.type"];
            if (rope_scaling_type == "yarn") {
                rope_type = RoPEType::YARN;
            } else if (rope_scaling_type == "linear") {
                rope_type = RoPEType::LINEAR_SCALE;
            } else if (rope_scaling_type == "dynamic") {
                rope_type = RoPEType::DYMAMIC_NTK;
            }
        }
        if (this->weight.dicts.find("rope_scaling.factor") != this->weight.dicts.end()) {
            rope_factor = atof(this->weight.dicts["rope_scaling.factor"].c_str());
        }
        rope_factor = GetFloatWithFallback(this->weight, {"rope_scaling.factor", "rope_factor"}, rope_factor);
        if (this->weight.dicts.find("rope_scaling.beta_fast") != this->weight.dicts.end()) {
            rope_scaling_beta_fast = atoi(this->weight.dicts["rope_scaling.beta_fast"].c_str());
        }
        rope_scaling_beta_fast = GetIntWithFallback(this->weight, {"rope_scaling.beta_fast", "beta_fast"}, rope_scaling_beta_fast);
        if (this->weight.dicts.find("rope_scaling.beta_slow") != this->weight.dicts.end()) {
            rope_scaling_beta_slow = atoi(this->weight.dicts["rope_scaling.beta_slow"].c_str());
        }
        rope_scaling_beta_slow = GetIntWithFallback(this->weight, {"rope_scaling.beta_slow", "beta_slow"}, rope_scaling_beta_slow);
        if (this->weight.dicts.find("rope_scaling.original_max_position_embeddings") != this->weight.dicts.end()) {
            rope_scaling_original_max_position_embeddings = atof(this->weight.dicts["rope_scaling.original_max_position_embeddings"].c_str());
        }
        rope_scaling_original_max_position_embeddings =
            GetFloatWithFallback(this->weight, {"rope_scaling.original_max_position_embeddings", "original_seq_len"},
                                 rope_scaling_original_max_position_embeddings);
        if (this->weight.dicts.find("rope_scaling.mscale") != this->weight.dicts.end()) {
            rope_scaling_mscale = atof(this->weight.dicts["rope_scaling.mscale"].c_str());
        } else {
            rope_scaling_mscale = 1.0f;
        }
        if (this->weight.dicts.find("rope_scaling.mscale_all_dim") != this->weight.dicts.end()) {
            rope_scaling_mscale_all_dim = atof(this->weight.dicts["rope_scaling.mscale_all_dim"].c_str());
        } else {
            rope_scaling_mscale_all_dim = rope_scaling_mscale;
        }

        // 预计算 RoPE：主分支 (rope_theta) + compress 分支 (compress_rope_theta)
        auto pair = this->UpdateRotaryPosEmb(rope_base, rope_factor);
        sinData.ToDevice(DataDevice::CPU);
        cosData.ToDevice(DataDevice::CPU);
        sinData.CopyFrom(Data(DataType::FLOAT32, { (int)this->sin.size(), (int)this->sin[0].size() }, pair.first));
        cosData.CopyFrom(Data(DataType::FLOAT32, { (int)this->cos.size(), (int)this->cos[0].size() }, pair.second));

        auto cpair = this->UpdateCompressRotaryPosEmb(compress_rope_theta, rope_factor);
        compressSinData.ToDevice(DataDevice::CPU);
        compressCosData.ToDevice(DataDevice::CPU);
        compressSinData.CopyFrom(Data(DataType::FLOAT32, { (int)this->compressSin.size(), (int)this->compressSin[0].size() }, cpair.first));
        compressCosData.CopyFrom(Data(DataType::FLOAT32, { (int)this->compressCos.size(), (int)this->compressCos[0].size() }, cpair.second));

        // -------- 注册 expert merge / 特殊层（与 V2 类似，用 V4 的命名） --------
        for (int i = 0; i < block_cnt; i++) {
            for (int j = -1; j < this->num_experts; j++) {
                std::string w1Name, w3Name, swigluName, downName;
                if (j == -1) {
                    w1Name = "layers." + std::to_string(i) + ".ffn.shared_experts.w1.weight";
                    w3Name = "layers." + std::to_string(i) + ".ffn.shared_experts.w3.weight";
                    swigluName = "layers." + std::to_string(i) + ".ffn.shared_experts.gateup.weight";
                    downName = "layers." + std::to_string(i) + ".ffn.shared_experts.w2.weight";
                } else {
                    w1Name = "layers." + std::to_string(i) + ".ffn.experts." + std::to_string(j) + ".w1.weight";
                    w3Name = "layers." + std::to_string(i) + ".ffn.experts." + std::to_string(j) + ".w3.weight";
                    swigluName = "layers." + std::to_string(i) + ".ffn.experts." + std::to_string(j) + ".gateup.weight";
                    downName = "layers." + std::to_string(i) + ".ffn.experts." + std::to_string(j) + ".w2.weight";
                }
                this->weightMergeRules.push_back(
                    WeightMergeRule({WeightMergeRuleSingle({w1Name, w3Name}, swigluName, std::string("linearSwiglu"))})
                );
                if (j != -1 || !GetCudaSharedExpert()) {
                    this->specialWeights[swigluName] = "linearSwiglu";
                    this->specialWeights[downName] = "linearColumn";
                }
                this->moeLinears.insert(w1Name);
                this->moeLinears.insert(w3Name);
                this->moeLinears.insert(downName);
            }
            // wkv / wo_a / indexer 的 latent 投影需保持高精度
            this->cantQuantLinears.insert("layers." + std::to_string(i) + ".attn.wkv.weight");
            this->cantQuantLinears.insert("layers." + std::to_string(i) + ".attn.wo_a.weight");
        }
    }

    std::pair<std::vector<float>, std::vector<float>> DeepSeekV4Model::UpdateRotaryPosEmb(float base, float factor, int seqLen) {
        int dim = rotary_dim;
        std::vector <float> freqExtra, freqInter;
        for (int i = 0; i < dim; i += 2) {
            freqExtra.push_back(1.0 / pow(base, (float)i / rotary_dim));
            freqInter.push_back(1.0 / (rope_factor * pow(base, (float)i / rotary_dim)));
        }

        int low, high;
        yarn_find_correction_range(
            rope_scaling_beta_fast,
            rope_scaling_beta_slow,
            dim, base,
            (int)rope_scaling_original_max_position_embeddings,
            low, high
        );
        std::vector <float> invFreqMask = yarn_linear_ramp_mask(low, high, dim / 2);
        for (size_t i = 0; i < invFreqMask.size(); i++) {
            invFreqMask[i] = 1.0 - invFreqMask[i];
        }
        std::vector <float> invFreq;
        for (size_t i = 0; i < freqInter.size(); i++) {
            invFreq.push_back(freqInter[i] * (1.0 - invFreqMask[i]) + freqExtra[i] * invFreqMask[i]);
        }

        float _mscale = yarn_get_mscale(rope_factor, rope_scaling_mscale) /
                        yarn_get_mscale(rope_factor, rope_scaling_mscale_all_dim);

        int positions = std::max(max_positions, seqLen);
        sin.resize(positions);
        cos.resize(positions);
        for (int i = 0; i < positions; i++) {
            sin[i].resize(rotary_dim);
            cos[i].resize(rotary_dim);
            for (int j = 0; j < (int)invFreq.size() * 2; j++) {
                sin[i][j] = ::sin((float)i * invFreq[j % invFreq.size()]) * _mscale;
                cos[i][j] = ::cos((float)i * invFreq[j % invFreq.size()]) * _mscale;
            }
        }
        std::vector <float> fsin, fcos;
        for (size_t i = 0; i < sin.size(); i++) {
            fsin.insert(fsin.end(), sin[i].begin(), sin[i].end());
        }
        for (size_t i = 0; i < cos.size(); i++) {
            fcos.insert(fcos.end(), cos[i].begin(), cos[i].end());
        }
        return std::make_pair(fsin, fcos);
    }

    std::pair<std::vector<float>, std::vector<float>> DeepSeekV4Model::UpdateCompressRotaryPosEmb(float base, float factor, int seqLen) {
        // compress 分支不做 YaRN 插值，对应 model.py 中 original_seq_len > 0 才开启的逻辑
        // 这里只生成纯 RoPE 频率（base 使用 compress_rope_theta）
        int dim = rotary_dim;
        std::vector <float> invFreq;
        for (int i = 0; i < dim; i += 2) {
            invFreq.push_back(1.0 / pow(base, (float)i / rotary_dim));
        }

        int positions = std::max(max_positions, seqLen);
        compressSin.resize(positions);
        compressCos.resize(positions);
        for (int i = 0; i < positions; i++) {
            compressSin[i].resize(rotary_dim);
            compressCos[i].resize(rotary_dim);
            for (int j = 0; j < (int)invFreq.size() * 2; j++) {
                compressSin[i][j] = ::sin((float)i * invFreq[j % invFreq.size()]);
                compressCos[i][j] = ::cos((float)i * invFreq[j % invFreq.size()]);
            }
        }
        std::vector <float> fsin, fcos;
        for (size_t i = 0; i < compressSin.size(); i++) {
            fsin.insert(fsin.end(), compressSin[i].begin(), compressSin[i].end());
        }
        for (size_t i = 0; i < compressCos.size(); i++) {
            fcos.insert(fcos.end(), compressCos[i].begin(), compressCos[i].end());
        }
        return std::make_pair(fsin, fcos);
    }

    int DeepSeekV4Model::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                                 const fastllm::Data &positionIds,
                                 std::vector<std::pair<Data, Data>> &pastKeyValues,
                                 const GenerationConfig &generationConfig,
                                 const LastTokensManager &lastTokens,
                                 std::vector <float> *retLogits) {
        std::vector <std::vector <float>*> batchLogits;
        batchLogits.push_back(retLogits);
        return ForwardBatch(1, inputIds, attentionMask, positionIds, pastKeyValues,
                            generationConfig, lastTokens, &batchLogits)[0];
    }

    std::vector <int> DeepSeekV4Model::ForwardBatch(int batch,
                                                   const fastllm::Data &inputIds,
                                                   const fastllm::Data &attentionMask,
                                                   const fastllm::Data &positionIds,
                                                   std::vector<std::pair<Data, Data>> &pastKeyValues,
                                                   const GenerationConfig &generationConfig,
                                                   const LastTokensManager &lastTokens,
                                                   std::vector <std::vector <float>*> *retLogits) {
        ScopedExecutorProfiler forwardOtherProfile("DeepSeekV4ForwardOther");
        bool profileEnabled = EnvFlagEnabled("FASTLLM_PROFILE") || EnvFlagEnabled("FASTLLM_PROFILE_DEEPSEEKV4");
        bool profileSync = profileEnabled && !EnvFlagEnabled("FASTLLM_PROFILE_NO_SYNC");
        double profileLastMs = NowMs();
        auto profileLap = [&](double &bucket) {
            if (!profileEnabled) {
                return;
            }
            ProfileSyncIfNeeded(profileSync);
            double now = NowMs();
            bucket += now - profileLastMs;
            profileLastMs = now;
        };
        auto profileLapDetail = [&](double &bucket, double &detailBucket) {
            if (!profileEnabled) {
                return;
            }
            ProfileSyncIfNeeded(profileSync);
            double now = NowMs();
            double span = now - profileLastMs;
            bucket += span;
            detailBucket += span;
            profileLastMs = now;
        };
        double profilePrepareMs = 0.0;
        double profileEmbedMs = 0.0;
        double profileDebugDumpMs = 0.0;
        double profileHcExpandMs = 0.0;
        double profileSetupMs = 0.0;
        double profileLayersMs = 0.0;
        double profileHeadMs = 0.0;
        double profileLogitsDumpMs = 0.0;
        double profileTopkMs = 0.0;
        double profilePastMs = 0.0;
        std::vector<DeepSeekV4LayerProfile> layerProfiles;
        bool debugStopNow = false;
        Data hiddenStates;
        int startPos = 0;
        if (positionIds.dims.size() >= 2 && positionIds.Count(0) > 0) {
            auto pids = ReadTokenIds(positionIds);
            startPos = pids.empty() ? 0 : pids[0];
        }
        int originalStartPos = startPos;
        bool debugFullRecomputeDecode = EnvFlagEnabled("FASTLLM_DEBUG_FULL_RECOMPUTE_DECODE") && batch == 1;
        Data recomputeInputIds;
        const Data *forwardInputIds = &inputIds;
        if (debugFullRecomputeDecode) {
            auto ids = ReadTokenIds(inputIds);
            if (originalStartPos == 0) {
                debugFullRecomputeTokens = ids;
                debugGeneratedTokens = 0;
            } else {
                debugFullRecomputeTokens.insert(debugFullRecomputeTokens.end(), ids.begin(), ids.end());
                std::vector<float> fullIds(debugFullRecomputeTokens.begin(), debugFullRecomputeTokens.end());
                recomputeInputIds.CopyFrom(Data(DataType::FLOAT32, {1, (int)fullIds.size()}, fullIds));
                forwardInputIds = &recomputeInputIds;
                startPos = 0;
            }
        }
        if (!debugFullRecomputeDecode && this->saveHistoryChat && !DeepSeekV4PrefixCacheDisabled() &&
            batch == 1 && inputIds.dims.size() >= 2 && inputIds.dims[1] > 1 &&
            !EnvFlagEnabled("FASTLLM_DSV4_PREFIX_CACHE_DISABLE_CHUNK_SPLIT")) {
            int seq = inputIds.dims[1];
            int nextBoundary = ((originalStartPos / 256) + 1) * 256;
            if (originalStartPos + seq > nextBoundary) {
                if (DeepSeekV4PrefixCacheDebugEnabled()) {
                    printf("[fastllm-dsv4-prefix-cache] split prefill start=%d seq=%d next_boundary=%d\n",
                           originalStartPos, seq, nextBoundary);
                    fflush(stdout);
                }
                std::vector<int> ret(1, 0);
                for (int offset = 0; offset < seq; ) {
                    int pos = originalStartPos + offset;
                    int boundary = ((pos / 256) + 1) * 256;
                    int curLen = std::min(seq - offset, boundary - pos);
                    if (curLen <= 0) {
                        curLen = std::min(256, seq - offset);
                    }
                    Data curInputIds, curPositionIds;
                    Split(inputIds, 1, offset, offset + curLen, curInputIds);
                    if (positionIds.dims.size() >= 2) {
                        Split(positionIds, 1, offset, offset + curLen, curPositionIds);
                    } else {
                        std::vector<float> pids(curLen);
                        for (int i = 0; i < curLen; i++) {
                            pids[i] = (float)(pos + i);
                        }
                        curPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, curLen}, pids));
                    }
                    ret = ForwardBatch(1, curInputIds, Data(), curPositionIds, pastKeyValues,
                                       generationConfig, lastTokens,
                                       (offset + curLen == seq) ? retLogits : nullptr);
                    offset += curLen;
                }
                return ret;
            }
        }
        if (!debugFullRecomputeDecode && batch == 1 && inputIds.dims.size() >= 2 &&
            inputIds.dims[1] > 1 && originalStartPos > 0 &&
            EnvFlagEnabled("FASTLLM_DSV4_ENABLE_PREFIX_CACHE_SEQUENTIAL")) {
            std::vector<int> ret(1, 0);
            int seq = inputIds.dims[1];
            for (int s = 0; s < seq; s++) {
                Data curInputIds, curPositionIds;
                Split(inputIds, 1, s, s + 1, curInputIds);
                if (positionIds.dims.size() >= 2) {
                    Split(positionIds, 1, s, s + 1, curPositionIds);
                } else {
                    curPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, 1}, {(float)(originalStartPos + s)}));
                }
                ret = ForwardBatch(1, curInputIds, Data(), curPositionIds, pastKeyValues,
                                   generationConfig, lastTokens,
                                   (s + 1 == seq) ? retLogits : nullptr);
            }
            return ret;
        }
        bool debugThisStep = (originalStartPos == 0) || EnvFlagEnabled("FASTLLM_DEBUG_ALL_STEPS");
        bool debugLogitsThisStep = debugThisStep || EnvFlagEnabled("FASTLLM_DEBUG_LOGITS_ALL_STEPS") || debugFullRecomputeDecode;
        bool debugDumpStates = EnvFlagEnabled("FASTLLM_DEBUG_DUMP_STATES");
        bool useDecodeCache = (!debugFullRecomputeDecode && batch == 1);
        int debugStopAfterTokens = EnvInt("FASTLLM_DEBUG_STOP_AFTER_TOKENS", 0);
        if (!debugFullRecomputeDecode && originalStartPos == 0 && debugStopAfterTokens > 0) {
            debugGeneratedTokens = 0;
        }

        profileLap(profilePrepareMs);
        if (debugThisStep || debugFullRecomputeDecode) {
            DebugDumpInputIds(*forwardInputIds, originalStartPos);
            profileLap(profileDebugDumpMs);
        }
        Embedding(*forwardInputIds, weight["embed.weight"], hiddenStates);
        profileLap(profileEmbedMs);
        if (debugThisStep && debugDumpStates) {
            DebugDumpData("embed", hiddenStates);
            profileLap(profileDebugDumpMs);
        }

        int bsz = forwardInputIds->dims[0];
        int seqlen = forwardInputIds->dims[1];
        int dim = embed_dim;
        {
            ScopedExecutorProfiler executorProfile("DeepSeekV4HcExpand");
            auto hv = ReadFloatData(hiddenStates);
            std::vector<float> expanded((uint64_t)bsz * seqlen * hc_mult * dim);
            for (int b = 0; b < bsz; b++) {
                for (int s = 0; s < seqlen; s++) {
                    const float *src = hv.data() + ((uint64_t)b * seqlen + s) * dim;
                    for (int h = 0; h < hc_mult; h++) {
                        memcpy(expanded.data() + (((uint64_t)b * seqlen + s) * hc_mult + h) * dim,
                               src, dim * sizeof(float));
                    }
                }
            }
            WriteFloatData(expanded, {bsz, seqlen, hc_mult, dim}, hiddenStates, hiddenStates.dataType);
        }
        profileLap(profileHcExpandMs);
        if (debugThisStep && debugDumpStates) {
            DebugDumpData("hc_expand", hiddenStates);
            profileLap(profileDebugDumpMs);
        }

        if (block_cnt <= 0) {
            ErrorInFastLLM("DeepSeekV4Model: invalid block_cnt.");
        }
        if (useDecodeCache && originalStartPos == 0) {
            decodeLayerCaches.clear();
            decodeLayerCaches.resize(block_cnt);
        }

        // 默认跑完全部主层；FASTLLM_DEBUG_LAYERS 可限制调试层数，便于逐层排查。
        int debugLayers = block_cnt;
        if (const char *env = std::getenv("FASTLLM_DEBUG_LAYERS")) {
            int limit = atoi(env);
            if (limit > 0) {
                debugLayers = std::min(limit, block_cnt);
            }
        }
        int debugDetailLayer = -1;
        if (const char *env = std::getenv("FASTLLM_DEBUG_DETAIL_LAYER")) {
            debugDetailLayer = atoi(env);
        }
        std::vector<int> tokenIds = ReadTokenIds(*forwardInputIds);
        if (!debugFullRecomputeDecode && this->saveHistoryChat && !DeepSeekV4PrefixCacheDisabled() && batch == 1) {
            if (originalStartPos == 0) {
                this->deepseekV4HistoryTokens = tokenIds;
            } else if ((int)this->deepseekV4HistoryTokens.size() == originalStartPos) {
                this->deepseekV4HistoryTokens.insert(this->deepseekV4HistoryTokens.end(), tokenIds.begin(), tokenIds.end());
            } else if ((int)this->deepseekV4HistoryTokens.size() < originalStartPos) {
                if (DeepSeekV4PrefixCacheDebugEnabled()) {
                    printf("[fastllm-dsv4-prefix-cache] reset token history: history=%d start=%d add=%d\n",
                           (int)this->deepseekV4HistoryTokens.size(), originalStartPos, (int)tokenIds.size());
                    fflush(stdout);
                }
                this->deepseekV4HistoryTokens.clear();
            }
        }
        if (weights.empty()) {
            auto getWeightPtr = [this](const std::string &name) -> Data* {
                auto it = this->weight.weight.find(name);
                return it == this->weight.weight.end() ? nullptr : &it->second;
            };
            weights.resize(block_cnt);
            biass.resize(block_cnt);
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
        profileLap(profileSetupMs);
        for (int layer = 0; layer < debugLayers; layer++) {
            DeepSeekV4LayerProfile layerProfile;
            std::string pre = "layers." + std::to_string(layer);
            int compressRatio = compress_ratios.size() > layer ? compress_ratios[layer] : 0;
            bool useCompressRope = compressRatio != 0;
            float layerRopeBase = useCompressRope ? compress_rope_theta : rope_base;
            int layerOriginalSeqLen = useCompressRope ? (int)rope_scaling_original_max_position_embeddings : 0;

            Data residual;
            residual.CopyFrom(hiddenStates);
            HcMix attnMix = HcPreReference(hiddenStates, weight[pre + ".hc_attn_fn"],
                                           weight[pre + ".hc_attn_scale"], weight[pre + ".hc_attn_base"],
                                           hc_mult, hc_sinkhorn_iters, hc_eps, rms_norm_eps);
            profileLapDetail(layerProfile.attnPrep, layerProfile.attnHcPre);
            Data attnInput;
            RMSNormReference(attnMix.y, weight[pre + ".attn_norm.weight"], rms_norm_eps, attnInput, DataType::BFLOAT16);
            profileLapDetail(layerProfile.attnPrep, layerProfile.attnNorm);

            Data qr, qNorm, q;
            Linear(attnInput, weight[pre + ".attn.wq_a.weight"], Data(), qr);
            RMSNormReference(qr, weight[pre + ".attn.q_norm.weight"], rms_norm_eps, qNorm, DataType::BFLOAT16);
            Linear(qNorm, weight[pre + ".attn.wq_b.weight"], Data(), q);
            profileLapDetail(layerProfile.attnPrep, layerProfile.qProj);
            q.Reshape({bsz, seqlen, num_attention_heads, head_dim_full});
            ScaleQRotary(q, rms_norm_eps, qk_rope_head_dim, layerRopeBase, startPos,
                         layerOriginalSeqLen, rope_factor, rope_scaling_beta_fast,
                         rope_scaling_beta_slow);
            profileLapDetail(layerProfile.attnPrep, layerProfile.qPost);

            Data kv;
            Linear(attnInput, weight[pre + ".attn.wkv.weight"], Data(), kv);
            RMSNormReference(kv, weight[pre + ".attn.kv_norm.weight"], rms_norm_eps, kv, DataType::BFLOAT16);
            profileLapDetail(layerProfile.attnPrep, layerProfile.kvProjNorm);
            kv.Reshape({bsz, seqlen, 1, head_dim_full});
            RotaryQuant(kv, qk_rope_head_dim, layerRopeBase, startPos,
                        layerOriginalSeqLen, rope_factor, rope_scaling_beta_fast,
                        rope_scaling_beta_slow, head_dim_full - qk_rope_head_dim, 64);
            kv.Reshape({bsz, seqlen, head_dim_full});
            profileLapDetail(layerProfile.attnPrep, layerProfile.kvPost);
            DeepSeekV4DecodeLayerCache *decodeCache = nullptr;
            if (useDecodeCache && layer < (int)decodeLayerCaches.size()) {
                decodeCache = &decodeLayerCaches[layer];
            }
            Data chunkPrefixKV;
            int chunkPrefixLen = 0;
            int decodeCompressedCount = 0;
            if (decodeCache != nullptr) {
                if (startPos == 0) {
                    auto kvValues = ReadFloatData(kv);
                    decodeCache->initialized = true;
                    decodeCache->bsz = bsz;
                    decodeCache->totalLen = seqlen;
                    decodeCache->headDim = head_dim_full;
                    decodeCache->windowSize = window_size;
                    decodeCache->compressRatio = compressRatio;
                    decodeCache->compressorWideDim = (compressRatio == 4 ? 2 : 1) * head_dim_full;
                    StoreWindowKVCache(kvValues, bsz, seqlen, head_dim_full, startPos, window_size,
                                       decodeCache->windowKV);
#ifdef USE_CUDA
                    if (!EnvFlagEnabled("FASTLLM_DSV4_DISABLE_CUDA_SPARSE_DECODE") && DeepSeekV4PreferCuda()) {
                        WriteFloatData(decodeCache->windowKV, {bsz, window_size, head_dim_full},
                                       decodeCache->windowKVData, DataType::FLOAT32);
                        decodeCache->windowKVData.ToDevice(DataDevice::CUDA);
                    }
#endif
                } else {
                    if (!decodeCache->initialized) {
                        ErrorInFastLLM("DeepSeekV4Model: decode cache is not initialized.");
                    }
                    if (seqlen > 1) {
                        chunkPrefixLen = BuildWindowKVPrefixData(decodeCache->windowKV, bsz, head_dim_full,
                                                                 startPos, window_size, chunkPrefixKV);
                    }
                    decodeCache->totalLen = startPos + seqlen;
                    bool updatedWindowKVData = false;
#ifdef USE_CUDA
                    if (!EnvFlagEnabled("FASTLLM_DSV4_DISABLE_CUDA_SPARSE_DECODE") &&
                        seqlen == 1 &&
                        kv.dataDevice == DataDevice::CUDA &&
                        decodeCache->windowKVData.dataDevice == DataDevice::CUDA) {
                        updatedWindowKVData = FastllmCudaDeepSeekV4UpdateWindowKVCache(
                            kv, startPos, window_size, decodeCache->windowKVData);
                    }
#endif
                    if (!updatedWindowKVData) {
                        auto kvValues = ReadFloatData(kv);
                        UpdateWindowKVCache(kvValues, bsz, head_dim_full, startPos, window_size,
                                            decodeCache->windowKV);
    #ifdef USE_CUDA
                        // Keep CUDA windowKVData in sync with the CPU windowKV just updated.
                        if (decodeCache->windowKVData.dataDevice == DataDevice::CUDA) {
                            WriteFloatData(decodeCache->windowKV, {bsz, window_size, head_dim_full},
                                           decodeCache->windowKVData, DataType::FLOAT32);
                            decodeCache->windowKVData.ToDevice(DataDevice::CUDA);
                        }
    #endif
                    }
                }
            }
            if (compressRatio > 0) {
                if (decodeCache != nullptr) {
                    std::vector<float> compressorKV, compressorScore;
                    ComputeCompressorRaw(weight, pre + ".attn.compressor", attnInput, compressorKV, compressorScore);
                    if (startPos == 0) {
                        decodeCache->compressorKVRaw = std::move(compressorKV);
                        decodeCache->compressorScoreRaw = std::move(compressorScore);
                    } else {
                        AppendCompressorRaw(compressorKV, compressorScore, bsz, seqlen,
                                            decodeCache->compressorWideDim,
                                            decodeCache->compressorKVRaw,
                                            decodeCache->compressorScoreRaw);
                    }
                    int compressedCutoff = decodeCache->totalLen - (decodeCache->totalLen % compressRatio);
                    int targetCompressedBlocks = compressRatio > 0 ? compressedCutoff / compressRatio : 0;
                    if (targetCompressedBlocks > 0 &&
                        decodeCache->compressedBlocks == targetCompressedBlocks &&
                        decodeCache->compressedKV.dims.size() >= 2) {
                        decodeCompressedCount = decodeCache->compressedBlocks;
                    } else if (BuildCompressedKVFromRaw(weight, pre + ".attn.compressor",
                                                        decodeCache->compressorKVRaw,
                                                        decodeCache->compressorScoreRaw,
                                                        bsz, decodeCache->totalLen, compressRatio,
                                                        head_dim_full, qk_rope_head_dim, layerRopeBase,
                                                        rope_factor, rope_scaling_beta_fast, rope_scaling_beta_slow,
                                                        layerOriginalSeqLen, decodeCache->compressedKV)) {
                        decodeCache->compressedBlocks = decodeCache->compressedKV.dims[1];
                        decodeCompressedCount = decodeCache->compressedBlocks;
                        if (startPos == 0) {
                            Data catKV;
                            ConcatSeqReference(kv, decodeCache->compressedKV, catKV);
                            kv.CopyFrom(catKV);
                        }
                    }
                } else {
                    Data compressedKV;
                    if (CompressKVReference(weight, pre + ".attn.compressor", attnInput, compressRatio,
                                            head_dim_full, qk_rope_head_dim, layerRopeBase, rope_factor,
                                            rope_scaling_beta_fast, rope_scaling_beta_slow,
                                            layerOriginalSeqLen, startPos, compressedKV)) {
                        Data catKV;
                        ConcatSeqReference(kv, compressedKV, catKV);
                        kv.CopyFrom(catKV);
                    }
                }
            }
            profileLap(layerProfile.cache);

            Data attnOut4, woAOut, attnOut;
            Data sparsePrefillKV;
            Data *sparsePrefillKVPtr = &kv;
            int sparsePrefillPrefixLen = 0;
            if (decodeCache != nullptr && startPos > 0 && seqlen > 1) {
                sparsePrefillPrefixLen = chunkPrefixLen;
                if (chunkPrefixLen > 0) {
                    ConcatSeqReference(chunkPrefixKV, kv, sparsePrefillKV);
                } else {
                    sparsePrefillKV.CopyFrom(kv);
                }
                if (decodeCompressedCount > 0 && decodeCache->compressedKV.dims.size() >= 2) {
                    Data catKV;
                    ConcatSeqReference(sparsePrefillKV, decodeCache->compressedKV, catKV);
                    sparsePrefillKV.CopyFrom(catKV);
                }
                sparsePrefillKVPtr = &sparsePrefillKV;
            }
            if (decodeCache != nullptr && startPos > 0 && seqlen == 1) {
                SparseAttentionDecodeCachedReference(q, decodeCache->windowKV, &decodeCache->windowKVData,
                                                     decodeCache->compressedKV, weight[pre + ".attn.attn_sink"],
                                                     window_size, startPos, decodeCompressedCount,
                                                     qk_rope_head_dim, layerRopeBase,
                                                     1.0f / std::sqrt((float)head_dim_full), attnOut4,
                                                     layerOriginalSeqLen, rope_factor,
                                                     rope_scaling_beta_fast, rope_scaling_beta_slow);
            } else {
                SparseAttentionReference(q, *sparsePrefillKVPtr, weight[pre + ".attn.attn_sink"], window_size,
                                         qk_rope_head_dim, layerRopeBase, startPos,
                                         1.0f / std::sqrt((float)head_dim_full), attnOut4,
                                         compressRatio, layerOriginalSeqLen, rope_factor,
                                         rope_scaling_beta_fast, rope_scaling_beta_slow,
                                         sparsePrefillPrefixLen);
            }
            profileLap(layerProfile.sparseAttn);
            WoA(attnOut4, weight[pre + ".attn.wo_a.weight"], o_groups, o_lora_rank, woAOut);
            profileLapDetail(layerProfile.attnOut, layerProfile.attnWoA);
            Linear(woAOut, weight[pre + ".attn.wo_b.weight"], Data(), attnOut);
            profileLapDetail(layerProfile.attnOut, layerProfile.attnWoB);
            if (debugThisStep && layer == debugDetailLayer) {
                DebugDumpData("layer" + std::to_string(layer) + "_attn_out", attnOut);
            }
            HcPostReference(attnOut, residual, attnMix, hiddenStates);
            profileLapDetail(layerProfile.attnOut, layerProfile.attnHcPost);
            if (debugThisStep && layer == debugDetailLayer) {
                DebugDumpData("layer" + std::to_string(layer) + "_after_attn", hiddenStates);
            }

            residual.CopyFrom(hiddenStates);
            HcMix ffnMix = HcPreReference(hiddenStates, weight[pre + ".hc_ffn_fn"],
                                          weight[pre + ".hc_ffn_scale"], weight[pre + ".hc_ffn_base"],
                                          hc_mult, hc_sinkhorn_iters, hc_eps, rms_norm_eps);
            profileLapDetail(layerProfile.route, layerProfile.ffnHcPre);
            Data ffnInput, ffnOut;
            RMSNormReference(ffnMix.y, weight[pre + ".ffn_norm.weight"], rms_norm_eps, ffnInput, DataType::BFLOAT16);
            profileLapDetail(layerProfile.route, layerProfile.ffnNorm);
            if (debugThisStep && layer == debugDetailLayer) {
                DebugDumpData("layer" + std::to_string(layer) + "_ffn_in", ffnInput);
            }
            std::vector<int> ffnDims = ffnInput.dims;
            ffnInput.Reshape({bsz * seqlen, dim});
            Data expertIndex, expertScore;
            BuildMoERoutingData(weight, pre + ".ffn", ffnInput, tokenIds, num_experts,
                                num_experts_per_tok, scoring_func, routed_scaling_factor,
                                expertIndex, expertScore,
                                profileEnabled ? &layerProfile : nullptr, profileSync);
            profileLap(layerProfile.route);
            if (layer >= (int)weights.size() || weights[layer].empty() ||
                weights[layer][0] == nullptr || weights[layer][1] == nullptr ||
                weights[layer].size() < 4 || weights[layer][2] == nullptr || weights[layer][3] == nullptr ||
                !CanRunMergeMOE(ffnInput, biass[layer])) {
                ffnInput.Reshape(ffnDims);
                MoEReference(weight, pre + ".ffn", ffnInput, tokenIds, num_experts,
                             num_experts_per_tok, scoring_func, routed_scaling_factor, swiglu_limit, ffnOut);
                profileLap(layerProfile.moe);
            } else {
                Data w1, w2, w3, tempInput, tempOutput, moeInputTemp, moeOutputTemp;
                ApplyDeviceMap(this->moeDeviceMap, layer + 1, block_cnt);
                // NumasMergeMOE 的小 batch 路径直接读取 cpuData，先保证输入在 CPU 可见。
                ffnInput.ToDevice(DataDevice::CPU);
                expertIndex.ToDevice(DataDevice::CPU);
                expertScore.ToDevice(DataDevice::CPU);
                profileLap(layerProfile.moeMove);
                MergeMOEBlock(&ffnInput, &expertIndex, &expertScore,
                              &weights[layer], &biass[layer],
                              &w1, &w2, &w3, &tempInput, &tempOutput,
                              1.0f, &ffnOut, layer,
                              ffnInput.dataType, this->moeAtype,
                              &moeInputTemp, &moeOutputTemp);
                profileLap(layerProfile.moe);
                ApplyDeviceMap(this->deviceMap, layer + 1, block_cnt);
                profileLap(layerProfile.moeMove);
            }
            ffnOut.Reshape(ffnDims);
            if (debugThisStep && layer == debugDetailLayer) {
                DebugDumpData("layer" + std::to_string(layer) + "_ffn_out", ffnOut);
            }
            HcPostReference(ffnOut, residual, ffnMix, hiddenStates);
            if (debugThisStep && debugDumpStates) {
                DebugDumpData("layer" + std::to_string(layer), hiddenStates);
            }
            profileLap(layerProfile.ffnPost);
            if (profileEnabled) {
                profileLayersMs += LayerProfileTotal(layerProfile);
                layerProfiles.push_back(layerProfile);
            }
        }

        Data headStates, headInput, normed, logits;
        const Data *headSource = &hiddenStates;
        if (seqlen > 1) {
            Split(hiddenStates, 1, seqlen - 1, seqlen, headStates);
            headSource = &headStates;
        }
        HcHeadReference(*headSource, weight["hc_head_fn"], weight["hc_head_scale"], weight["hc_head_base"],
                        hc_mult, hc_eps, rms_norm_eps, headInput);
        RMSNormReference(headInput, weight["norm.weight"], rms_norm_eps, normed, DataType::BFLOAT16);
        Linear(normed, weight["head.weight"], Data(), logits);
        ToDataType(logits, DataType::FLOAT32);
        profileLap(profileHeadMs);
        if (debugLogitsThisStep) {
            DebugDumpData("logits", logits);
            profileLap(profileLogitsDumpMs);
        }
        if (originalStartPos == 0 && std::getenv("FASTLLM_DEBUG_EXIT_AFTER_PREFILL") != nullptr) {
            fflush(stdout);
            std::_Exit(0);
        }

        if (generationConfig.output_logits && retLogits != nullptr) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4LogitsRead");
            logits.ToDevice(DataDevice::CPU);
            int vocabSize = logits.dims.back();
            for (int b = 0; b < batch; b++) {
                (*retLogits)[b]->resize(vocabSize);
                memcpy((float*)(*retLogits)[b]->data(),
                       ((float*)logits.cpuData) + ((uint64_t)b * logits.dims[1] + logits.dims[1] - 1) * vocabSize,
                       vocabSize * sizeof(float));
            }
        }

        Data topk;
        TopK(logits, topk, 1);
        std::vector<int> ret;
        {
            ScopedExecutorProfiler executorProfile("DeepSeekV4TopKRead");
            topk.ToDevice(DataDevice::CPU);
            for (int b = 0; b < batch; b++) {
                ret.push_back((int)(((float*)topk.cpuData)[b * 2] + 1e-3));
            }
        }
        profileLap(profileTopkMs);

        if (debugFullRecomputeDecode) {
            debugGeneratedTokens++;
            printf("[fastllm-debug] generated step=%d start_pos=%d token=%d\n",
                   debugGeneratedTokens, originalStartPos, ret.empty() ? -1 : ret[0]);
            if (debugStopAfterTokens > 0 && debugGeneratedTokens >= debugStopAfterTokens) {
                debugStopNow = true;
            }
        } else if (debugStopAfterTokens > 0) {
            debugGeneratedTokens++;
            printf("[fastllm-debug] generated step=%d start_pos=%d token=%d\n",
                   debugGeneratedTokens, originalStartPos, ret.empty() ? -1 : ret[0]);
            if (debugGeneratedTokens >= debugStopAfterTokens) {
                debugStopNow = true;
            }
        }

        int finalTotalLen = originalStartPos + inputIds.dims[1];
        UpdateDebugPastKeyValues(pastKeyValues, bsz, finalTotalLen, block_cnt);
        if (!debugFullRecomputeDecode && this->saveHistoryChat && !DeepSeekV4PrefixCacheDisabled() &&
            batch == 1 && finalTotalLen % 256 == 0 &&
            (int)this->deepseekV4HistoryTokens.size() >= finalTotalLen) {
            this->RecordHistorySnapshot(this->deepseekV4HistoryTokens, finalTotalLen);
        } else if (!debugFullRecomputeDecode && this->saveHistoryChat && !DeepSeekV4PrefixCacheDisabled() &&
                   batch == 1 && finalTotalLen % 256 == 0 && DeepSeekV4PrefixCacheDebugEnabled()) {
            printf("[fastllm-dsv4-prefix-cache] skip boundary record: final_len=%d history_tokens=%d\n",
                   finalTotalLen, (int)this->deepseekV4HistoryTokens.size());
            fflush(stdout);
        }
        profileLap(profilePastMs);
        if (profileEnabled) {
            double totalMs = profilePrepareMs + profileEmbedMs + profileDebugDumpMs +
                             profileHcExpandMs + profileSetupMs + profileLayersMs +
                             profileHeadMs + profileLogitsDumpMs + profileTopkMs + profilePastMs;
            printf("[fastllm-profile] step start_pos=%d seqlen=%d batch=%d mode=%s token=%d total_ms=%.3f\n",
                   originalStartPos, seqlen, batch, originalStartPos == 0 ? "prefill" : "decode",
                   ret.empty() ? -1 : ret[0], totalMs);
            printf("[fastllm-profile]   prepare=%.3f embed=%.3f debug_dump=%.3f hc_expand=%.3f setup=%.3f layers=%.3f head=%.3f logits_dump=%.3f topk=%.3f past=%.3f\n",
                   profilePrepareMs, profileEmbedMs, profileDebugDumpMs, profileHcExpandMs,
                   profileSetupMs, profileLayersMs, profileHeadMs, profileLogitsDumpMs,
                   profileTopkMs, profilePastMs);
            for (int i = 0; i < (int)layerProfiles.size(); i++) {
                const auto &p = layerProfiles[i];
                printf("[fastllm-profile]   layer=%02d total=%.3f attn_prep=%.3f cache=%.3f sparse_attn=%.3f attn_out=%.3f route=%.3f moe_move=%.3f moe=%.3f ffn_post=%.3f\n",
                       i, LayerProfileTotal(p), p.attnPrep, p.cache, p.sparseAttn,
                       p.attnOut, p.route, p.moeMove, p.moe, p.ffnPost);
                if (EnvFlagEnabled("FASTLLM_PROFILE_DETAIL")) {
                    printf("[fastllm-profile-detail] layer=%02d attn_hc_pre=%.3f attn_norm=%.3f q_proj=%.3f q_post=%.3f kv_proj_norm=%.3f kv_post=%.3f attn_woa=%.3f attn_wob=%.3f attn_hc_post=%.3f ffn_hc_pre=%.3f ffn_norm=%.3f route_gate=%.3f route_score=%.3f\n",
                           i, p.attnHcPre, p.attnNorm, p.qProj, p.qPost,
                           p.kvProjNorm, p.kvPost, p.attnWoA, p.attnWoB,
                           p.attnHcPost, p.ffnHcPre, p.ffnNorm, p.routeGate,
                           p.routeScore);
                }
            }
            fflush(stdout);
        }
        if (debugStopNow) {
            fflush(stdout);
            std::_Exit(0);
        }
        return ret;
    }

    std::vector <int> DeepSeekV4Model::ForwardBatch(int batch,
                                                   const Data &inputIds,
                                                   const std::vector <Data*> &attentionMask,
                                                   const std::vector <Data*> &positionIds,
                                                   const std::vector <int> &seqLens,
                                                   std::vector <std::pair <Data*, Data*> > &pastKeyValues,
                                                   const std::vector <GenerationConfig> &generationConfigs,
                                                   const LastTokensManager &lastTokens,
                                                   std::vector <std::vector <float>*> *retLogits) {
        ErrorInFastLLM("DeepSeekV4Model::ForwardBatch (multi-prompt) is not implemented yet.");
        return std::vector<int>(batch, 0);
    }

    bool DeepSeekV4Model::NeedAttentionMask(int qlen, int klen) {
        // 滑窗 + sparse 索引下，mask 由 sparse_attn 内部处理
        return false;
    }

    void DeepSeekV4Model::FillLLMInputsBatch(std::vector<std::vector<float>> &inputTokens,
                                             const std::vector<std::map<std::string, int>> &params,
                                             fastllm::Data &inputIds, fastllm::Data &attentionMask,
                                             fastllm::Data &positionIds) {
        // 先复用 DeepSeekV2 的 batch 填充逻辑：左填充 + 因果 mask + 顺序 positionIds
        // 后续若 hash gate 需要原始 input_ids，可以保持当前 inputIds 直接被消费
        inputIds.ToDevice(DataDevice::CPU);
        attentionMask.ToDevice(DataDevice::CPU);
        positionIds.ToDevice(DataDevice::CPU);

        int batch = (int)inputTokens.size();
        int index = params[0].find("index")->second;
        if (index == 0) {
            int maxLen = 0;
            for (int i = 0; i < batch; i++) {
                maxLen = std::max(maxLen, (int)inputTokens[i].size());
            }
            std::vector <float> ids(batch * maxLen, 0);
            std::vector <float> vpids(batch * maxLen, 0);
            std::vector <float> vmask(batch * maxLen * maxLen, 0);
            for (int i = 0; i < batch; i++) {
                auto &tokens = inputTokens[i];
                int len = (int)tokens.size();
                int base = maxLen - len;
                for (int j = 0; j < len; j++) {
                    ids[i * maxLen + base + j] = tokens[j];
                    vpids[i * maxLen + base + j] = j;
                }
                std::fill(vmask.data() + i * maxLen * maxLen,
                          vmask.data() + i * maxLen * maxLen + (maxLen - len) * maxLen, 1.0);
                for (int j = maxLen - len; j < maxLen; j++) {
                    std::fill(vmask.data() + i * maxLen * maxLen + j * maxLen,
                              vmask.data() + i * maxLen * maxLen + j * maxLen + maxLen - len, 1.0);
                }
                for (int j = 0; j < len; j++) {
                    for (int k = j + 1; k < len; k++) {
                        vmask[i * maxLen * maxLen + (base + j) * maxLen + base + k] = 1;
                    }
                }
            }
            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen}, ids));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen, maxLen}, vmask));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch, maxLen}, vpids));
        } else {
            std::vector <float> pids(batch);
            std::vector <float> fret;
            for (int i = 0; i < batch; i++) {
                fret.push_back(inputTokens[i][0]);
            }
            int maxLen = 0;
            for (int i = 0; i < batch; i++) {
                int promptLen = params[i].find("promptLen")->second;
                maxLen = std::max(promptLen, maxLen);
                pids[i] = promptLen + index - 1;
            }
            maxLen += index;
            std::vector <float> vmasks(batch * maxLen, 0.0f);
            for (int i = 0; i < batch; i++) {
                int curLen = params[i].find("promptLen")->second + index;
                for (int j = 0; j < maxLen - curLen; j++) {
                    vmasks[i * maxLen + j] = 1.0f;
                }
            }
            inputIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, fret));
            attentionMask.CopyFrom(Data(DataType::FLOAT32, {batch, 1, maxLen}, vmasks));
            positionIds.CopyFrom(Data(DataType::FLOAT32, {batch, 1}, pids));
        }
    }

    void DeepSeekV4Model::WarmUp() {
        // forward 尚未实现，warmup 暂时只打印提示，保持构造期间不崩溃
        printf("DeepSeekV4Model warmup skipped: forward not implemented yet.\n");
    }

    std::string DeepSeekV4Model::MakeInput(const std::string &history, int round, const std::string &input) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
    }

    std::string DeepSeekV4Model::MakeHistory(const std::string &history, int round,
                                             const std::string &input, const std::string &output) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role + output + history_sep;
    }
}
