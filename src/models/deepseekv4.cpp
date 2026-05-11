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

        static bool HasTensorData(const Data &data) {
            return !data.dims.empty() && data.Count(0) > 0 &&
                   (data.cpuData != nullptr || data.cudaData != nullptr);
        }

        static void CopyTensorData(Data &dst, const Data &src) {
            ResetData(dst);
            if (HasTensorData(src)) {
                Copy(src, dst);
            }
        }

        static int GetDataSeqLen(const Data &data, int bsz, int dim) {
            if (!HasTensorData(data) || data.dims.size() < 3 || bsz <= 0 || dim <= 0 ||
                data.dims[0] != bsz || data.dims[2] != dim) {
                return 0;
            }
            return data.dims[1];
        }

        static int RoundUpToBlock(int value, int block) {
            return ((std::max(value, 1) - 1) / block + 1) * block;
        }

        static void EnsureCompressorRawCapacity(Data &data, int targetLen) {
            if (!HasTensorData(data) || data.dims.size() != 3 || targetLen <= 0) {
                return;
            }
            int targetCapacity = RoundUpToBlock(targetLen, 128);
            int currentCapacity = data.dims[1];
            if (data.expansionDims.size() == data.dims.size()) {
                currentCapacity = data.expansionDims[1];
            }
            if (currentCapacity >= targetCapacity) {
                return;
            }
            std::vector<int> newDims = data.dims;
            newDims[1] = targetCapacity;
            data.Expansion(newDims);
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

        struct HcMix {
            Data y;
            Data postData;
            Data combData;
            std::vector<float> post;
            std::vector<float> comb;
            int b = 0, s = 0, hc = 0;
        };

        static void RMSNormReference(const Data &input, Data &weight, float eps, Data &output, DataType dtype) {
            RMSNorm(input, weight, eps, output);
            ToDataType(output, dtype);
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

        static void StoreWindowKVCache(const Data &kv, int bsz, int seqlen, int headDim,
                                       int startPos, int windowSize, Data &windowKV) {
            (void)bsz;
            (void)seqlen;
            (void)headDim;
            Executor &executor = *((Executor*)GetExecutor());
            executor.Run("DeepSeekV4StoreWindowKVCache", {
                {"input", (Data*)&kv}, {"cache", &windowKV}
            }, {}, {{"startPos", startPos}, {"windowSize", windowSize}});
        }

        static void UpdateWindowKVCache(const Data &kv, int bsz, int headDim,
                                        int startPos, int windowSize, Data &windowKV) {
            (void)bsz;
            (void)headDim;
            Executor &executor = *((Executor*)GetExecutor());
            executor.Run("DeepSeekV4UpdateWindowKVCache", {
                {"input", (Data*)&kv}, {"cache", &windowKV}
            }, {}, {{"startPos", startPos}, {"windowSize", windowSize}});
        }

        static int BuildWindowKVPrefixData(const Data &windowKV, int bsz, int headDim,
                                           int startPos, int windowSize, Data &output) {
            int prefixLen = std::min(windowSize, startPos);
            if (prefixLen <= 0 || !HasTensorData(windowKV)) {
                return 0;
            }
            ScopedExecutorProfiler executorProfile("DeepSeekV4KVCache");
#ifdef USE_CUDA
            if (DeepSeekV4PreferCuda() && windowKV.dataDevice == DataDevice::CUDA &&
                FastllmCudaDeepSeekV4BuildWindowKVPrefix(windowKV, startPos, windowSize, prefixLen, output)) {
                return prefixLen;
            }
#endif
            auto cached = ReadFloatData(windowKV);
            std::vector<float> prefix((uint64_t)bsz * prefixLen * headDim);
            int firstPos = startPos - prefixLen;
            for (int b = 0; b < bsz; b++) {
                for (int s = 0; s < prefixLen; s++) {
                    int srcSlot = (firstPos + s) % windowSize;
                    memcpy(prefix.data() + ((uint64_t)b * prefixLen + s) * headDim,
                           cached.data() + ((uint64_t)b * windowSize + srcSlot) * headDim,
                           (uint64_t)headDim * sizeof(float));
                }
            }
            WriteFloatData(prefix, {bsz, prefixLen, headDim}, output, DataType::FLOAT32);
            return prefixLen;
        }

        static void ComputeCompressorRaw(WeightMap &weight, const std::string &prefix, const Data &x,
                                         Data &kv, Data &score) {
            Linear((Data&)x, weight[prefix + ".wkv.weight"], Data(), kv);
            Linear((Data&)x, weight[prefix + ".wgate.weight"], Data(), score);
        }

        static void AppendCompressorRaw(const Data &kv, const Data &score,
                                        int bsz, int seqlen, int wideDim,
                                        Data &allKV, Data &allScore) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4CompressorAppend");
            if (seqlen <= 0 || wideDim <= 0 || kv.dims.size() != 3 || score.dims != kv.dims) {
                return;
            }
            if (!HasTensorData(allKV)) {
                Copy(kv, allKV);
                Copy(score, allScore);
                EnsureCompressorRawCapacity(allKV, kv.dims[1]);
                EnsureCompressorRawCapacity(allScore, score.dims[1]);
                return;
            }
            int oldLen = GetDataSeqLen(allKV, bsz, wideDim);
            if (oldLen <= 0 || GetDataSeqLen(allScore, bsz, wideDim) != oldLen) {
                Copy(kv, allKV);
                Copy(score, allScore);
                EnsureCompressorRawCapacity(allKV, kv.dims[1]);
                EnsureCompressorRawCapacity(allScore, score.dims[1]);
                return;
            }
            EnsureCompressorRawCapacity(allKV, oldLen + kv.dims[1]);
            EnsureCompressorRawCapacity(allScore, oldLen + score.dims[1]);
            CatDirect(allKV, kv, 1);
            CatDirect(allScore, score, 1);
        }

        static int GetCompressorRawLen(const Data &raw, int bsz, int wideDim) {
            return GetDataSeqLen(raw, bsz, wideDim);
        }

        static void TrimCompressorRawCache(int bsz, int totalLen, int compressRatio, int wideDim,
                                           int compressedBlocks, Data &allKV,
                                           Data &allScore, int &rawTokenBase) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4CompressorTrim");
            int oldLen = GetCompressorRawLen(allKV, bsz, wideDim);
            if (oldLen <= 0 || GetCompressorRawLen(allScore, bsz, wideDim) != oldLen) {
                ResetData(allKV);
                ResetData(allScore);
                rawTokenBase = std::max(0, totalLen);
                return;
            }

            int rawEnd = rawTokenBase + oldLen;
            int retainStart = compressedBlocks * std::max(1, compressRatio);
            if (compressRatio == 4 && compressedBlocks > 0) {
                retainStart = (compressedBlocks - 1) * compressRatio;
            }
            retainStart = std::max(rawTokenBase, std::min(retainStart, rawEnd));
            if (retainStart <= rawTokenBase) {
                return;
            }

            int newLen = rawEnd - retainStart;
            if (newLen <= 0) {
                ResetData(allKV);
                ResetData(allScore);
                rawTokenBase = retainStart;
                return;
            }

            int dropLen = retainStart - rawTokenBase;
            Data nextKV, nextScore;
            Split(allKV, 1, dropLen, oldLen, nextKV);
            Split(allScore, 1, dropLen, oldLen, nextScore);
            CopyTensorData(allKV, nextKV);
            CopyTensorData(allScore, nextScore);
            rawTokenBase = retainStart;
        }

        struct BuildCompressedKVRangeOp : MultiThreadBaseOp {
            const float *kv;
            const float *score;
            const float *ape;
            float *compressed;
            uint64_t st, end;
            int bsz, rawTokenBase, rawLen, blockStart, blockCount, compressRatio, headDim, wideDim;
            bool overlap;

            BuildCompressedKVRangeOp(const float *kv, const float *score, const float *ape,
                                     float *compressed, uint64_t st, uint64_t end,
                                     int bsz, int rawTokenBase, int rawLen, int blockStart, int blockCount,
                                     int compressRatio, int headDim, int wideDim, bool overlap)
                : kv(kv), score(score), ape(ape), compressed(compressed), st(st), end(end),
                  bsz(bsz), rawTokenBase(rawTokenBase), rawLen(rawLen),
                  blockStart(blockStart), blockCount(blockCount),
                  compressRatio(compressRatio), headDim(headDim), wideDim(wideDim),
                  overlap(overlap) {}

            uint64_t RawOffset(int b, int token, int dimOffset) const {
                int localToken = token - rawTokenBase;
                return ((uint64_t)b * rawLen + localToken) * wideDim + dimOffset;
            }

            void ScanTerms(int b, int block, int d, float &mx) const {
                if (overlap) {
                    if (block > 0) {
                        for (int r = 0; r < compressRatio; r++) {
                            int tok = (block - 1) * compressRatio + r;
                            uint64_t off = RawOffset(b, tok, d);
                            mx = std::max(mx, score[off] + ape[(uint64_t)r * wideDim + d]);
                        }
                    }
                    for (int r = 0; r < compressRatio; r++) {
                        int tok = block * compressRatio + r;
                        uint64_t off = RawOffset(b, tok, headDim + d);
                        mx = std::max(mx, score[off] + ape[(uint64_t)r * wideDim + headDim + d]);
                    }
                } else {
                    for (int r = 0; r < compressRatio; r++) {
                        int tok = block * compressRatio + r;
                        uint64_t off = RawOffset(b, tok, d);
                        mx = std::max(mx, score[off] + ape[(uint64_t)r * wideDim + d]);
                    }
                }
            }

            void AccumulateTerms(int b, int block, int d, float mx, double &sum, double &value) const {
                if (overlap) {
                    if (block > 0) {
                        for (int r = 0; r < compressRatio; r++) {
                            int tok = (block - 1) * compressRatio + r;
                            uint64_t off = RawOffset(b, tok, d);
                            double e = std::exp((double)(score[off] + ape[(uint64_t)r * wideDim + d]) - mx);
                            sum += e;
                            value += e * kv[off];
                        }
                    }
                    for (int r = 0; r < compressRatio; r++) {
                        int tok = block * compressRatio + r;
                        uint64_t off = RawOffset(b, tok, headDim + d);
                        double e = std::exp((double)(score[off] + ape[(uint64_t)r * wideDim + headDim + d]) - mx);
                        sum += e;
                        value += e * kv[off];
                    }
                } else {
                    for (int r = 0; r < compressRatio; r++) {
                        int tok = block * compressRatio + r;
                        uint64_t off = RawOffset(b, tok, d);
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
                                                int bsz, int rawTokenBase, int rawLen,
                                                int blockStart, int blockCount,
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
                                         bsz, rawTokenBase, rawLen, blockStart, blockCount, compressRatio,
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
                    bsz, rawTokenBase, rawLen, blockStart, blockCount,
                    compressRatio, headDim, wideDim, overlap));
            }
            for (int i = 0; i < (int)ops.size(); i++) {
                pool->PushOp(i, ops[i]);
            }
            for (int i = 0; i < (int)ops.size(); i++) {
                pool->Wait(i);
                delete ops[i];
            }
        }

        static bool ComputeCompressedKVRangeData(WeightMap &weight, const std::string &prefix,
                                                 const Data &kv, const Data &score,
                                                 int bsz, int rawTokenBase, int rawLen,
                                                 int blockStart, int blockCount,
                                                 int compressRatio, int headDim, int wideDim, bool overlap,
                                                 Data &compressed) {
#ifdef USE_CUDA
            if (DeepSeekV4PreferCuda() && kv.dataDevice == DataDevice::CUDA &&
                score.dataDevice == DataDevice::CUDA) {
                Data ape, apeCuda;
                ToDataType(weight[prefix + ".ape"], ape, DataType::FLOAT32);
                apeCuda.CopyFrom(ape);
                apeCuda.ToDevice(DataDevice::CUDA);
                if (FastllmCudaDeepSeekV4BuildCompressedKV(
                        kv, score, apeCuda, rawTokenBase, rawLen, blockStart, blockCount,
                        compressRatio, headDim, wideDim, overlap, compressed)) {
                    return true;
                }
            }
#endif
            auto kvValues = ReadFloatData(kv);
            auto scoreValues = ReadFloatData(score);
            auto apePtr = ReadWeightFloatDataCached(weight[prefix + ".ape"]);
            std::vector<float> compressedValues;
            ComputeCompressedKVRangeCpu(kvValues, scoreValues, *apePtr, bsz, rawTokenBase, rawLen,
                                        blockStart, blockCount, compressRatio, headDim, wideDim,
                                        overlap, compressedValues);
            WriteFloatData(compressedValues, {bsz, blockCount, headDim}, compressed, DataType::FLOAT32);
            return true;
        }

        static void FinalizeCompressedKVRows(WeightMap &weight, const std::string &prefix,
                                             const Data &compressedData, int blockStart,
                                             int compressRatio, int headDim,
                                             int ropeDim, float ropeBase, float ropeFactor,
                                             int betaFast, int betaSlow, int originalSeqLen,
                                             Data &output) {
            Data compressedForNorm, normed;
            ToDataType(compressedData, compressedForNorm, DataType::BFLOAT16);
            if (compressedData.dataDevice == DataDevice::CUDA) {
                compressedForNorm.ToDevice(DataDevice::CUDA);
            }
            RMSNormReference(compressedForNorm, weight[prefix + ".norm.weight"], 1e-6f, normed, DataType::BFLOAT16);
#ifdef USE_CUDA
            if (normed.dataDevice == DataDevice::CUDA &&
                FastllmCudaDeepSeekV4RotaryQuant(normed, ropeDim, ropeBase, blockStart * compressRatio,
                                                 originalSeqLen, ropeFactor, betaFast, betaSlow,
                                                 headDim - ropeDim, 64, compressRatio)) {
                CopyTensorData(output, normed);
                return;
            }
#endif
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

        static bool AppendCompressedKVRowsCuda(Data &output, const Data &newRows,
                                               int bsz, int oldBlocks, int addBlocks, int headDim) {
#ifdef USE_CUDA
            if (!DeepSeekV4PreferCuda()) {
                return false;
            }
            Data rowsCuda;
            const Data *rowsPtr = &newRows;
            if (newRows.dataDevice != DataDevice::CUDA) {
                rowsCuda.CopyFrom(newRows);
                rowsCuda.ToDevice(DataDevice::CUDA);
                rowsPtr = &rowsCuda;
            }
            if (oldBlocks <= 0 || output.dims.size() < 2 || output.Count(0) <= 0 ||
                (output.cpuData == nullptr && output.cudaData == nullptr)) {
                ResetData(output);
                output.CopyFrom(*rowsPtr);
                output.SetKVCache();
                output.ToDevice(DataDevice::CUDA);
                return true;
            }
            if (output.dataDevice != DataDevice::CUDA || output.cudaData == nullptr) {
                return false;
            }
            int blocks = oldBlocks + addBlocks;
            Data merged;
            if (!PrepareCudaData(merged, DataType::BFLOAT16, {bsz, blocks, headDim})) {
                return false;
            }
            merged.SetKVCache();

            size_t unit = sizeof(uint16_t);
            size_t oldPitch = (size_t)oldBlocks * headDim * unit;
            size_t addPitch = (size_t)addBlocks * headDim * unit;
            size_t mergedPitch = (size_t)blocks * headDim * unit;
            FastllmCudaMemcpy2DDeviceToDevice(
                merged.cudaData, mergedPitch,
                output.cudaData, oldPitch,
                oldPitch, bsz);
            FastllmCudaMemcpy2DDeviceToDevice(
                (uint8_t*)merged.cudaData + (size_t)oldBlocks * headDim * unit, mergedPitch,
                rowsPtr->cudaData, addPitch,
                addPitch, bsz);
            ResetData(output);
            output.CopyFrom(merged);
            output.SetKVCache();
            output.ToDevice(DataDevice::CUDA);
            return true;
#else
            (void)output;
            (void)newRows;
            (void)bsz;
            (void)oldBlocks;
            (void)addBlocks;
            (void)headDim;
            return false;
#endif
        }

        static bool HasCompressedKVData(const Data &data) {
            return data.dims.size() >= 2 && data.Count(0) > 0 &&
                   (data.cpuData != nullptr || data.cudaData != nullptr);
        }

        static bool EnsureCompressedKVOnCpu(DeepSeekV4DecodeLayerCache &cache) {
            if (!HasCompressedKVData(cache.compressedKV)) {
                return false;
            }
            cache.compressedKV.ToDevice(DataDevice::CPU);
            return HasCompressedKVData(cache.compressedKV);
        }

        static bool EnsureCompressedKVOnCuda(DeepSeekV4DecodeLayerCache &cache) {
#ifdef USE_CUDA
            if (!DeepSeekV4PreferCuda()) {
                return false;
            }
            if (!HasCompressedKVData(cache.compressedKV)) {
                return false;
            }
            cache.compressedKV.SetKVCache();
            cache.compressedKV.ToDevice(DataDevice::CUDA);
            return HasCompressedKVData(cache.compressedKV);
#else
            (void)cache;
            return false;
#endif
        }

        static bool BuildCompressedKVFromRaw(WeightMap &weight, const std::string &prefix,
                                             const Data &kv, const Data &score,
                                             int bsz, int rawTokenBase, int totalLen, int compressRatio,
                                             int headDim, int ropeDim, float ropeBase,
                                             float ropeFactor, int betaFast, int betaSlow,
                                             int originalSeqLen, Data &output, bool preferCudaOutput = false) {
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
            int rawLen = GetCompressorRawLen(kv, bsz, wideDim);
            if (rawLen <= 0 || GetCompressorRawLen(score, bsz, wideDim) != rawLen) {
                return false;
            }

#ifdef USE_CUDA
            if (preferCudaOutput && HasCompressedKVData(output) && output.dataDevice != DataDevice::CUDA) {
                output.SetKVCache();
                output.ToDevice(DataDevice::CUDA);
            }
#else
            (void)preferCudaOutput;
#endif
            int reusableBlocks = GetReusableCompressedBlocks(output, bsz, blocks, headDim);
            if (reusableBlocks == blocks) {
                return true;
            }

            int firstNeededToken = reusableBlocks * compressRatio;
            if (overlap && reusableBlocks > 0) {
                firstNeededToken = (reusableBlocks - 1) * compressRatio;
            }
            int lastNeededToken = blocks * compressRatio;
            if (rawTokenBase > firstNeededToken || rawTokenBase + rawLen < lastNeededToken) {
                return false;
            }

            int addBlocks = blocks - reusableBlocks;
            Data ape;
            ToDataType(weight[prefix + ".ape"], ape, DataType::FLOAT32);
            DeepSeekV4BuildCompressedKVFromRaw(kv, score, ape, weight[prefix + ".norm.weight"],
                                               rawTokenBase, rawLen, reusableBlocks, addBlocks,
                                               compressRatio, headDim, ropeDim, ropeBase, ropeFactor,
                                               betaFast, betaSlow, originalSeqLen, overlap,
                                               preferCudaOutput, output);
            return true;
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

        static void SparseAttentionDecodeCachedReference(Data &q,
                                                         const Data &windowKV,
                                                         const Data &compressedKV,
                                                         Data &attnSink,
                                                         int windowSize, int startPos, int compressedCount,
                                                         int ropeDim, float ropeBase, float softmaxScale,
                                                         Data &output, int originalSeqLen = 0,
                                                         float ropeFactor = 1.0f, int betaFast = 32, int betaSlow = 1) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4SparseDecodeCached");
#ifdef USE_CUDA
            if (q.dims[1] == 1) {
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
                if (FastllmCudaDeepSeekV4SparseAttentionDecodeCached(*qForCuda, windowKV,
                                                                     *compressedForCuda,
                                                                     attnSink, windowSize, startPos,
                                                                     compressedCount, ropeDim, ropeBase,
                                                                     originalSeqLen, ropeFactor,
                                                                     betaFast, betaSlow, softmaxScale, output)) {
                    return;
                }
            }
#endif
            auto qv = ReadFloatData(q);
            auto windowValues = ReadFloatData(windowKV);
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
                    return windowValues.data() + ((uint64_t)b * windowSize + idx) * dim;
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
            Data kv, score;
            Linear((Data&)x, weight[prefix + ".wkv.weight"], Data(), kv);
            Linear((Data&)x, weight[prefix + ".wgate.weight"], Data(), score);

            int wideDim = coff * headDim;
            Data compressed;
            if (!ComputeCompressedKVRangeData(weight, prefix, kv, score, bsz, 0, seqlen, 0, blocks,
                                              compressRatio, headDim, wideDim, overlap, compressed)) {
                return false;
            }
            FinalizeCompressedKVRows(weight, prefix, compressed, 0,
                                     compressRatio, headDim, ropeDim, ropeBase, ropeFactor,
                                     betaFast, betaSlow, originalSeqLen, output);
            return true;
        }

        static void ConcatSeqReference(const Data &a, const Data &b, Data &output) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4ConcatSeq");
            if (a.dims.size() == 3 && b.dims.size() == 3 &&
                a.dims[0] == b.dims[0] && a.dims[2] == b.dims[2] &&
                a.dataType == b.dataType) {
                Cat(a, b, 1, output);
                return;
            }
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

        static void BuildMoERoutingData(WeightMap &weight, const std::string &prefix, const Data &x,
                                        const std::vector<int> &inputIds, int nRoutedExperts,
                                        int topk, const std::string &scoreFunc, float routeScale,
                                        Data &expertIndex, Data &expertScore) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4RouteScore");
            Data xFloat, routerLogits;
            ToDataType(x, xFloat, DataType::FLOAT32);
            Linear(xFloat, weight[prefix + ".gate.weight"], Data(), routerLogits, true);

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
                auto selectByScore = [&]() {
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
                };
                bool useHashRow = hashRouting && inputIds.size() > (size_t)t && inputIds[t] >= 0 &&
                                  tid2eid != nullptr &&
                                  tid2eid->size() >= (uint64_t)(inputIds[t] + 1) * topk;
                if (useHashRow) {
                    uint64_t routeOffset = (uint64_t)inputIds[t] * topk;
                    for (int k = 0; k < topk; k++) {
                        int expert = (int)((*tid2eid)[routeOffset + k] + 0.5f);
                        curIndices[k] = std::max(0, std::min(expert, nRoutedExperts - 1));
                    }
                } else {
                    selectByScore();
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

    static bool DeepSeekV4PrefixCacheEveryBlockSplitEnabled() {
        return EnvFlagEnabled("FASTLLM_DSV4_PREFIX_CACHE_ENABLE_CHUNK_SPLIT") &&
               !EnvFlagEnabled("FASTLLM_DSV4_PREFIX_CACHE_DISABLE_CHUNK_SPLIT");
    }

    static thread_local int gDeepSeekV4SuppressHistorySnapshot = 0;

    struct ScopedDeepSeekV4HistorySnapshotSuppress {
        bool active = false;

        explicit ScopedDeepSeekV4HistorySnapshotSuppress(bool active) : active(active) {
            if (this->active) {
                gDeepSeekV4SuppressHistorySnapshot++;
            }
        }

        ~ScopedDeepSeekV4HistorySnapshotSuppress() {
            if (active) {
                gDeepSeekV4SuppressHistorySnapshot--;
            }
        }
    };

    static bool DeepSeekV4HistorySnapshotSuppressed() {
        return gDeepSeekV4SuppressHistorySnapshot > 0 &&
               !EnvFlagEnabled("FASTLLM_DSV4_PREFIX_CACHE_RECORD_INTERMEDIATE_CHUNKS");
    }

    DeepSeekV4DecodeLayerCache::DeepSeekV4DecodeLayerCache(const DeepSeekV4DecodeLayerCache &other) {
        *this = other;
    }

    DeepSeekV4DecodeLayerCache &DeepSeekV4DecodeLayerCache::operator=(const DeepSeekV4DecodeLayerCache &other) {
        if (this == &other) {
            return *this;
        }
        initialized = other.initialized;
        bsz = other.bsz;
        totalLen = other.totalLen;
        headDim = other.headDim;
        windowSize = other.windowSize;
        compressRatio = other.compressRatio;
        compressorWideDim = other.compressorWideDim;
        compressorRawTokenBase = other.compressorRawTokenBase;
        compressedBlocks = other.compressedBlocks;
        compressedTokenBase = other.compressedTokenBase;
        rawTailStartPos = other.rawTailStartPos;
        CopyTensorData(windowKV, other.windowKV);
        CopyTensorData(compressorKVRaw, other.compressorKVRaw);
        CopyTensorData(compressorScoreRaw, other.compressorScoreRaw);
        CopyTensorData(compressedKV, other.compressedKV);
        CopyTensorData(compressorTailKV, other.compressorTailKV);
        CopyTensorData(compressorTailScore, other.compressorTailScore);
        return *this;
    }

    DeepSeekV4HistoryLayerCache::DeepSeekV4HistoryLayerCache(const DeepSeekV4HistoryLayerCache &other) {
        *this = other;
    }

    DeepSeekV4HistoryLayerCache &DeepSeekV4HistoryLayerCache::operator=(const DeepSeekV4HistoryLayerCache &other) {
        if (this == &other) {
            return *this;
        }
        initialized = other.initialized;
        bsz = other.bsz;
        totalLen = other.totalLen;
        headDim = other.headDim;
        windowSize = other.windowSize;
        compressRatio = other.compressRatio;
        compressorWideDim = other.compressorWideDim;
        compressorRawTokenBase = other.compressorRawTokenBase;
        compressedBlocks = other.compressedBlocks;
        compressedTokenBase = other.compressedTokenBase;
        rawTailStartPos = other.rawTailStartPos;
        CopyTensorData(windowKV, other.windowKV);
        CopyTensorData(compressorKVRaw, other.compressorKVRaw);
        CopyTensorData(compressorScoreRaw, other.compressorScoreRaw);
        CopyTensorData(compressedKV, other.compressedKV);
        CopyTensorData(compressorTailKV, other.compressorTailKV);
        CopyTensorData(compressorTailScore, other.compressorTailScore);
        return *this;
    }

    void DeepSeekV4HistoryCacheManager::SetMaxRecordNum(int maxRecordNum) {
        std::lock_guard<std::mutex> guard(this->locker);
        this->maxRecordNum = std::max(1, maxRecordNum);
    }

    void DeepSeekV4HistoryCacheManager::Record(const DeepSeekV4HistoryCacheMemory &memory) {
        if (memory.tokens <= 0 || (int)memory.inputToken.size() != memory.tokens || memory.layers.empty()) {
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

        int maxProbeLen = (int)inputToken.size() - 1;
        int maxAligned = maxProbeLen / this->logicalBlockSize * this->logicalBlockSize;
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
            if (len <= hitLen || len > maxProbeLen) {
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
        DeepSeekV4RequestState state;
        if (!RestoreHistoryCacheMemory(memory, state)) {
            return false;
        }
        this->decodeLayerCaches = state.decodeLayerCaches;
        this->deepseekV4HistoryTokens = state.historyTokens;
        return true;
    }

    bool DeepSeekV4Model::RestoreHistoryCacheMemory(const DeepSeekV4HistoryCacheMemory &memory,
                                                    DeepSeekV4RequestState &state) {
        if (memory.tokens <= 0 || memory.layers.empty()) {
            return false;
        }
        state.decodeLayerCaches.clear();
        state.decodeLayerCaches.resize(memory.layers.size());
        for (int i = 0; i < (int)memory.layers.size(); i++) {
            const auto &src = memory.layers[i];
            auto &dst = state.decodeLayerCaches[i];
            dst.initialized = src.initialized;
            dst.bsz = src.bsz;
            dst.totalLen = src.totalLen;
            dst.headDim = src.headDim;
            dst.windowSize = src.windowSize;
            dst.compressRatio = src.compressRatio;
            dst.compressorWideDim = src.compressorWideDim;
            CopyTensorData(dst.windowKV, src.windowKV);
            dst.compressedBlocks = src.compressedBlocks;
            dst.compressedTokenBase = src.compressedTokenBase;
            dst.rawTailStartPos = src.rawTailStartPos;
            dst.compressorRawTokenBase = src.compressorRawTokenBase;
            CopyTensorData(dst.compressorTailKV, src.compressorTailKV);
            CopyTensorData(dst.compressorTailScore, src.compressorTailScore);

            ResetData(dst.compressedKV);
            if (src.compressedBlocks > 0 && src.compressedKV.dims.size() >= 2) {
                dst.compressedKV.CopyFrom(src.compressedKV);
                EnsureCompressedKVOnCuda(dst);
            }

            if (HasTensorData(src.compressorKVRaw)) {
                CopyTensorData(dst.compressorKVRaw, src.compressorKVRaw);
                CopyTensorData(dst.compressorScoreRaw, src.compressorScoreRaw);
            } else if (src.compressRatio > 0 && src.compressorWideDim > 0 &&
                       HasTensorData(src.compressorTailKV) && HasTensorData(src.compressorTailScore)) {
                int tailTokens = GetDataSeqLen(src.compressorTailKV, std::max(1, src.bsz), src.compressorWideDim);
                int tailStart = std::max(0, std::min(src.rawTailStartPos, src.totalLen));
                tailTokens = std::min(tailTokens, src.totalLen - tailStart);
                CopyTensorData(dst.compressorKVRaw, src.compressorTailKV);
                CopyTensorData(dst.compressorScoreRaw, src.compressorTailScore);
                dst.compressorRawTokenBase = tailStart;
            } else {
                ResetData(dst.compressorKVRaw);
                ResetData(dst.compressorScoreRaw);
                dst.compressorRawTokenBase = src.totalLen;
            }

#ifdef USE_CUDA
            if (DeepSeekV4PreferCuda() && HasTensorData(dst.windowKV) && dst.bsz > 0 &&
                dst.windowSize > 0 && dst.headDim > 0) {
                dst.windowKV.SetKVCache();
                dst.windowKV.ToDevice(DataDevice::CUDA);
            }
#endif
        }
        state.historyTokens = memory.inputToken;
        if (DeepSeekV4PrefixCacheDebugEnabled()) {
            printf("[fastllm-dsv4-prefix-cache] restore hit_len=%d blocks=%d layers=%d\n",
                   memory.tokens, memory.blockCount, (int)memory.layers.size());
            for (int i = 0; i < (int)state.decodeLayerCaches.size(); i++) {
                const auto &layer = state.decodeLayerCaches[i];
                printf("[fastllm-dsv4-prefix-cache]   layer=%02d ratio=%d total_len=%d compressed_blocks=%d window=%d raw_tail_start=%d tail_tokens=%d\n",
                       i, layer.compressRatio, layer.totalLen, layer.compressedBlocks,
                       GetDataSeqLen(layer.windowKV, std::max(1, layer.bsz), std::max(1, layer.headDim)),
                       layer.rawTailStartPos,
                       layer.compressorWideDim > 0 && layer.bsz > 0 ?
                           GetDataSeqLen(layer.compressorTailKV, layer.bsz, layer.compressorWideDim) : 0);
            }
            fflush(stdout);
        }
        return true;
    }

    void DeepSeekV4Model::RecordHistorySnapshot(const std::vector<int> &tokens, int totalLen) {
        RecordHistorySnapshot(tokens, totalLen, this->decodeLayerCaches);
    }

    void DeepSeekV4Model::RecordHistorySnapshot(const std::vector<int> &tokens,
                                                int totalLen,
                                                const std::vector<DeepSeekV4DecodeLayerCache> &decodeCaches) {
        if (DeepSeekV4HistorySnapshotSuppressed()) {
            return;
        }
        if (!this->saveHistoryChat || DeepSeekV4PrefixCacheDisabled() ||
            totalLen <= 0 || (int)tokens.size() < totalLen ||
            decodeCaches.empty()) {
            return;
        }
        DeepSeekV4HistoryCacheMemory memory;
        memory.tokens = totalLen;
        memory.blockCount = (totalLen + 255) / 256;
        memory.inputToken.assign(tokens.begin(), tokens.begin() + totalLen);
        memory.blockHash = DeepSeekV4TokenBlockHash(memory.inputToken, totalLen, 256);
        memory.layers.resize(decodeCaches.size());
        bool storeFullRaw = EnvFlagEnabled("FASTLLM_DSV4_PREFIX_CACHE_FULL_RAW");
        for (int i = 0; i < (int)decodeCaches.size(); i++) {
            const auto &src = decodeCaches[i];
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
            CopyTensorData(dst.windowKV, src.windowKV);
            dst.compressedBlocks = src.compressedBlocks;
            dst.compressedTokenBase = src.compressedBlocks * std::max(1, src.compressRatio);
            dst.compressorRawTokenBase = src.compressorRawTokenBase;
            ResetData(dst.compressedKV);
            if (src.compressedBlocks > 0 && HasCompressedKVData(src.compressedKV)) {
                bool copiedCompressed = false;
#ifdef USE_CUDA
                if (DeepSeekV4PreferCuda()) {
                    CopyTensorData(dst.compressedKV, src.compressedKV);
                    copiedCompressed = true;
                }
#endif
                if (!copiedCompressed) {
                    Data compressedCpuTemp;
                    compressedCpuTemp.CopyFrom(src.compressedKV);
                    compressedCpuTemp.ToDevice(DataDevice::CPU);
                    dst.compressedKV.CopyFrom(compressedCpuTemp);
                }
            }
            if (HasCompressedKVData(dst.compressedKV)) {
#ifdef USE_CUDA
                if (DeepSeekV4PreferCuda()) {
                    dst.compressedKV.SetKVCache();
                    dst.compressedKV.ToDevice(DataDevice::CUDA);
                } else
#endif
                {
                    dst.compressedKV.ToDevice(DataDevice::CPU);
                    dst.compressedKV.lockInCPU = true;
                }
            }

            if (src.compressRatio > 0 && src.compressorWideDim > 0) {
                int tailTokens = src.compressRatio == 4 ? 8 : (src.compressRatio == 128 ? 128 : src.compressRatio);
                tailTokens = std::min(tailTokens, src.totalLen);
                if (storeFullRaw) {
                    CopyTensorData(dst.compressorKVRaw, src.compressorKVRaw);
                    CopyTensorData(dst.compressorScoreRaw, src.compressorScoreRaw);
                    dst.compressorRawTokenBase = src.compressorRawTokenBase;
                }
                if (HasTensorData(src.compressorKVRaw) && HasTensorData(src.compressorScoreRaw)) {
                    int rawLen = GetCompressorRawLen(src.compressorKVRaw, src.bsz, src.compressorWideDim);
                    int rawEnd = src.compressorRawTokenBase + rawLen;
                    int tailStart = std::max(src.compressorRawTokenBase, src.totalLen - tailTokens);
                    tailStart = std::min(tailStart, rawEnd);
                    tailTokens = std::max(0, rawEnd - tailStart);
                    dst.rawTailStartPos = tailStart;
                    int rawOffset = tailStart - src.compressorRawTokenBase;
                    if (tailTokens > 0) {
                        Data tailKV, tailScore;
                        Split(src.compressorKVRaw, 1, rawOffset, rawOffset + tailTokens, tailKV);
                        Split(src.compressorScoreRaw, 1, rawOffset, rawOffset + tailTokens, tailScore);
                        CopyTensorData(dst.compressorTailKV, tailKV);
                        CopyTensorData(dst.compressorTailScore, tailScore);
                    } else {
                        ResetData(dst.compressorTailKV);
                        ResetData(dst.compressorTailScore);
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
        auto restoredState = std::make_shared<DeepSeekV4RequestState>();
        if (!this->RestoreHistoryCacheMemory(memory, *restoredState)) {
            return false;
        }
        {
            std::lock_guard<std::mutex> guard(this->requestStateMutex);
            this->pendingRequestState = restoredState;
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
        auto releaseDecodeCaches = [&]() {
            this->decodeLayerCaches.clear();
            std::vector<DeepSeekV4DecodeLayerCache>().swap(this->decodeLayerCaches);
            this->deepseekV4HistoryTokens.clear();
            std::vector<int>().swap(this->deepseekV4HistoryTokens);
        };

        if (!this->saveHistoryChat || DeepSeekV4PrefixCacheDisabled() ||
            this->decodeLayerCaches.empty() || allTokens.empty()) {
            if (DeepSeekV4PrefixCacheDebugEnabled()) {
                printf("[fastllm-dsv4-prefix-cache] skip record: save=%d disabled=%d caches=%d tokens=%d\n",
                       this->saveHistoryChat ? 1 : 0, DeepSeekV4PrefixCacheDisabled() ? 1 : 0,
                       (int)this->decodeLayerCaches.size(), (int)allTokens.size());
                fflush(stdout);
            }
            releaseDecodeCaches();
            return;
        }
        int totalLen = this->decodeLayerCaches[0].totalLen;
        if (totalLen > 0 && (int)allTokens.size() >= totalLen) {
            this->RecordHistorySnapshot(allTokens, totalLen);
        } else if (DeepSeekV4PrefixCacheDebugEnabled()) {
            printf("[fastllm-dsv4-prefix-cache] skip record: total_len=%d all_tokens=%d\n",
                   totalLen, (int)allTokens.size());
            fflush(stdout);
        }
        releaseDecodeCaches();
    }

    std::shared_ptr<DeepSeekV4RequestState> DeepSeekV4Model::GetRequestState(std::vector<std::pair<Data, Data> > &pastKeyValues) {
        const void *key = (const void*)&pastKeyValues;
        std::lock_guard<std::mutex> guard(this->requestStateMutex);
        auto it = this->requestStates.find(key);
        if (it == this->requestStates.end()) {
            return nullptr;
        }
        return it->second;
    }

    void DeepSeekV4Model::OnResponseContextCreated(ResponseContext *context) {
        if (context == nullptr) {
            return;
        }
        const void *key = (const void*)&context->pastKeyValues;
        std::lock_guard<std::mutex> guard(this->requestStateMutex);
        if (this->pendingRequestState) {
            this->requestStates[key] = this->pendingRequestState;
            this->pendingRequestState.reset();
        } else {
            this->requestStates[key] = std::make_shared<DeepSeekV4RequestState>();
        }
    }

    void DeepSeekV4Model::OnResponseContextRemoved(ResponseContext *context) {
        if (context == nullptr) {
            return;
        }
        const void *key = (const void*)&context->pastKeyValues;
        std::lock_guard<std::mutex> guard(this->requestStateMutex);
        this->requestStates.erase(key);
    }

    void DeepSeekV4Model::TryRecordResponseContext(ResponseContext *context) {
        if (context == nullptr) {
            return;
        }
        const void *key = (const void*)&context->pastKeyValues;
        std::shared_ptr<DeepSeekV4RequestState> state;
        {
            std::lock_guard<std::mutex> guard(this->requestStateMutex);
            auto it = this->requestStates.find(key);
            if (it != this->requestStates.end()) {
                state = it->second;
            }
        }
        if (!state) {
            TryRecordHistoryCache(context->allTokens);
            return;
        }
        if (!this->saveHistoryChat || DeepSeekV4PrefixCacheDisabled() ||
            state->decodeLayerCaches.empty() || context->allTokens.empty()) {
            if (DeepSeekV4PrefixCacheDebugEnabled()) {
                printf("[fastllm-dsv4-prefix-cache] skip record: save=%d disabled=%d caches=%d tokens=%d\n",
                       this->saveHistoryChat ? 1 : 0, DeepSeekV4PrefixCacheDisabled() ? 1 : 0,
                       (int)state->decodeLayerCaches.size(), (int)context->allTokens.size());
                fflush(stdout);
            }
            return;
        }
        int totalLen = state->decodeLayerCaches[0].totalLen;
        if (totalLen > 0 && (int)context->allTokens.size() >= totalLen) {
            this->RecordHistorySnapshot(context->allTokens, totalLen, state->decodeLayerCaches);
        } else if (DeepSeekV4PrefixCacheDebugEnabled()) {
            printf("[fastllm-dsv4-prefix-cache] skip record: total_len=%d all_tokens=%d\n",
                   totalLen, (int)context->allTokens.size());
            fflush(stdout);
        }
    }

    void DeepSeekV4Model::RunModelSpecificScheduler() {
        DeepSeekV4Model *model = this;
        long long kvCacheLimit = 16LL << 30;
#ifdef USE_CUDA
        auto freeSizes = FastllmCudaGetFreeSizes();
        auto dmap = GetDeviceMap();
        std::set<int> deviceIds;
        std::map<int, int> ratios;
        for (auto &it : dmap) {
            if (StartWith(it.first, "cuda")) {
                for (int id : ParseDeviceIds(it.first, "cuda", ratios)) {
                    deviceIds.insert(id);
                }
            }
        }
        if (deviceIds.empty()) {
            deviceIds.insert(0);
        }
        kvCacheLimit = 0;
        for (int id : deviceIds) {
            if (id < (int)freeSizes.size()) {
                kvCacheLimit += std::max(freeSizes[id] * 3 / 4, freeSizes[id] - (2LL << 30));
            }
        }
        if (kvCacheLimit == 0) {
            kvCacheLimit = 16LL << 30;
        }
#endif
        if (model->kvCacheLimit > 0) {
            kvCacheLimit = model->kvCacheLimit;
        }

        int maxTotalLens = kvCacheLimit / 1024 / 1024;
        if (model->elementsInKVCachePerToken > 0) {
            long long bytesPerToken = GetDataBytes(model->kvCacheDataType, 1, model->elementsInKVCachePerToken);
            if (bytesPerToken > 0) {
                maxTotalLens = kvCacheLimit / bytesPerToken;
            }
        }
        if (model->tokensLimit > 0) {
            maxTotalLens = model->tokensLimit;
        }

        int maxBatch = std::max(1, std::min(512, maxTotalLens / 128));
        if (model->maxBatch > 0) {
            maxBatch = model->maxBatch;
        }
        if (!model->canDoBatchForward && !model->canDoConcurrentForward) {
            maxBatch = 1;
        }
        maxBatch = std::max(1, maxBatch);

        model->tokensLimit = maxTotalLens;
        int limit = maxTotalLens;
        model->promptLimit = limit * 3 / 4;
        int prefillChunkSize = model->GetChunkedPrefillSize();

        auto getContextLen = [&](ResponseContext *ctx) -> int {
            if (ctx == nullptr) {
                return 0;
            }
            auto state = model->GetRequestState(ctx->pastKeyValues);
            if (state && !state->decodeLayerCaches.empty()) {
                int totalLen = state->decodeLayerCaches[0].totalLen;
                if (totalLen > 0) {
                    return totalLen;
                }
            }
            if ((int)ctx->pastKeyValues.size() > model->kvCacheId) {
                const Data &kv = ctx->pastKeyValues[model->kvCacheId].first;
                if (kv.expansionDims.size() > 1) {
                    return kv.expansionDims[1];
                }
                if (kv.dims.size() > 1) {
                    return kv.dims[1];
                }
            }
            return ctx->cacheLen + ctx->preTokens;
        };

        if (model->verbose) {
            printf("Fastllm KV Cache Limit: %f MB.\n", (double)kvCacheLimit / 1e6);
            printf("Fastllm KV Cache Token limit: %d tokens.\n", maxTotalLens);
            printf("Fastllm Prompt Token limit: %d tokens.\n", std::min(model->max_positions, model->promptLimit));
            printf("Fastllm Batch limit: %d.\n", maxBatch);
            printf("Fastllm Scheduler: DeepSeekV4.\n");
        }

        auto lastRecordTime = std::chrono::system_clock::now();
        long long genTokens = 0;
        while (true) {
            if (model->isFree) {
                break;
            }

            std::vector<Data*> attentionMasks;
            std::vector<Data*> positionIds;
            std::vector<float> ids;
            std::vector<int> seqLens;
            std::vector<int> handles;
            std::vector<GenerationConfig> generationConfigs;
            LastTokensManager tokensManager;
            std::vector<std::vector<float>*> logits;

            std::unique_lock<std::mutex> dictLocker(model->dictLocker);
            auto &forwardLocker = model->forwardLocker;

            std::set<int> abortHandles;
            for (auto &it : model->responseContextDict.dicts) {
                if (it.second->isAbort) {
                    it.second->TryRecord(model);
                    abortHandles.insert(it.first);
                }
            }
            for (auto &it : abortHandles) {
                model->RemoveResponseContext(it);
            }

            int lenSum = 0, currentActivate = 0;
            for (auto &it : model->responseContextDict.dicts) {
                if (it.second->isEnding) {
                    continue;
                }
                int ctxLen = getContextLen(it.second);
                if (it.second->preTokens > 0 || ctxLen > 0) {
                    lenSum += ctxLen;
                    currentActivate++;
                }
            }

            std::vector<std::pair<int, int> > orders;
            for (auto &it : model->responseContextDict.dicts) {
                orders.push_back(std::make_pair(-(int)it.second->currentTokens.size(), it.first));
            }
            sort(orders.begin(), orders.end());

            for (int isPrompt = 1; isPrompt >= 0; isPrompt--) {
                if (isPrompt == 0 && !seqLens.empty()) {
                    continue;
                }

                for (auto &ii : orders) {
                    auto contextIt = model->responseContextDict.dicts.find(ii.second);
                    if (contextIt == model->responseContextDict.dicts.end()) {
                        continue;
                    }
                    auto &it = *contextIt;
                    ResponseContext *ctx = it.second;

                    if (ctx->isEnding) {
                        continue;
                    }
                    if (isPrompt && ctx->preTokens != 0) {
                        continue;
                    }
                    if (!isPrompt && ctx->preTokens == 0) {
                        continue;
                    }
                    if (isPrompt && !seqLens.empty()) {
                        continue;
                    }
                    if (isPrompt && currentActivate >= maxBatch) {
                        continue;
                    }

                    if ((maxTotalLens > 0 && ctx->cacheLen + (int)ctx->currentTokens.size() > maxTotalLens) ||
                        ctx->cacheLen + (int)ctx->currentTokens.size() > model->max_positions) {
                        ctx->isEnding = true;
                        ctx->error = ResponseContextErrorPromptTooLong;
                        continue;
                    }

                    if (!isPrompt) {
                        int sur = ctx->generationConfig.output_token_limit - ctx->curTokens;
                        int predictLen = 256;
                        if (sur > 0) {
                            predictLen = std::min(predictLen, ((sur - 1) / 128 + 1) * 128);
                        }
                        if (maxTotalLens > 0 && lenSum + predictLen > maxTotalLens) {
                            continue;
                        }
                        lenSum += predictLen;
                    } else {
                        lenSum += ctx->currentTokens.size();
                        currentActivate++;
                    }

                    generationConfigs.push_back(ctx->generationConfig);
                    if (ctx->generationConfig.output_logits) {
                        ctx->resultLogits.push(new std::vector<float>());
                        logits.push_back(ctx->resultLogits.back());
                    } else {
                        logits.push_back(nullptr);
                    }

                    tokensManager.units.push_back(ctx->tokens);
                    handles.push_back(it.first);

                    if (ctx->preTokens == 0) {
                        ctx->intParams["add_special_tokens"] =
                            ctx->cacheLen > 0 ? false : ctx->generationConfig.add_special_tokens;
                        ctx->intParams["promptLen"] = ctx->cacheLen + ctx->currentTokens.size();
                        ctx->intParams["index"] = 0;
                    } else {
                        ctx->intParams["index"]++;
                    }

                    Data inputIds, attentionMask, curPositionIds;
                    std::vector<std::vector<float> > tokens(1);
                    for (int token : ctx->currentTokens) {
                        tokens[0].push_back(token);
                    }
                    model->FillLLMInputs(tokens, ctx->intParams, inputIds, attentionMask, curPositionIds);
                    ToDataType(attentionMask, model->dataType);

                    seqLens.push_back(inputIds.Count(0));
                    for (int i = 0; i < inputIds.Count(0); i++) {
                        ids.push_back(((float*)inputIds.cpuData)[i]);
                    }
                    if (attentionMask.dims.empty()) {
                        attentionMasks.push_back(nullptr);
                    } else {
                        attentionMasks.push_back(new Data());
                        attentionMask.ToDevice(DataDevice::CPU);
                        attentionMasks.back()->CopyFrom(attentionMask);
                    }
                    if (curPositionIds.dims.empty()) {
                        positionIds.push_back(nullptr);
                    } else {
                        positionIds.push_back(new Data());
                        positionIds.back()->CopyFrom(curPositionIds);
                    }
                    ctx->preTokens += seqLens.back();

                    if (isPrompt) {
                        break;
                    }
                    if ((int)seqLens.size() >= maxBatch ||
                        (maxTotalLens > 0 && lenSum + (int)seqLens.size() * 128 > maxTotalLens)) {
                        break;
                    }
                }
            }

            if (!seqLens.empty()) {
                dictLocker.unlock();
                forwardLocker.lock();
#ifdef USE_CUDA
                FastllmCudaClearBigBuffer();
#endif
                Data inputIds = Data(DataType::FLOAT32, {1, (int)ids.size()}, ids);
                std::vector<int> ret;

                if (seqLens.size() > 1) {
                    for (int i = 0; i < (int)handles.size(); i++) {
                        Data inputIdNow = Data(DataType::FLOAT32, {1, 1}, {ids[i]});
                        LastTokensManager singleTokens;
                        singleTokens.units.push_back(tokensManager.units[i]);
                        Data emptyAttention, emptyPosition;
                        dictLocker.lock();
                        auto contextIt = model->responseContextDict.dicts.find(handles[i]);
                        if (contextIt == model->responseContextDict.dicts.end()) {
                            dictLocker.unlock();
                            ret.push_back(model->eos_token_id);
                            continue;
                        }
                        ResponseContext *ctx = contextIt->second;
                        ret.push_back(model->Forward(inputIdNow,
                                                     attentionMasks[i] == nullptr ? emptyAttention : *attentionMasks[i],
                                                     positionIds[i] == nullptr ? emptyPosition : *positionIds[i],
                                                     ctx->pastKeyValues,
                                                     generationConfigs[i], singleTokens, logits[i]));
                        dictLocker.unlock();
                    }
                } else {
                    dictLocker.lock();
                    auto contextIt = model->responseContextDict.dicts.find(handles[0]);
                    ResponseContext *ctx = contextIt == model->responseContextDict.dicts.end() ? nullptr : contextIt->second;
                    std::vector<std::pair<Data, Data> > *pastKeyValue = ctx == nullptr ? nullptr : &ctx->pastKeyValues;
                    bool isMultimodal = ctx != nullptr && !ctx->multimodalInput.empty();
                    dictLocker.unlock();

                    if (ctx == nullptr || pastKeyValue == nullptr) {
                        ret.push_back(model->eos_token_id);
                    } else if (isMultimodal) {
                        Data emptyAttention, emptyPosition;
                        ret = model->ForwardMultimodal(inputIds,
                                                       attentionMasks[0] == nullptr ? emptyAttention : *attentionMasks[0],
                                                       positionIds[0] == nullptr ? emptyPosition : *positionIds[0],
                                                       *pastKeyValue, ctx->multimodalInput,
                                                       ctx->generationConfig, tokensManager, &logits);
                    } else if (seqLens[0] > prefillChunkSize) {
                        int len = seqLens[0];
                        for (int st = 0; st < len; ) {
                            if (model->verbose) {
                                genTokens += seqLens.size();
                                auto nowTime = std::chrono::system_clock::now();
                                float spend = GetSpan(lastRecordTime, nowTime);
                                if (spend > 1) {
                                    printf("Long Prefill ... (%d%%)\n", st * 100 / len);
                                    lastRecordTime = nowTime;
                                }
                            }
                            int curLen = std::min(st == 0 ? prefillChunkSize : prefillChunkSize, len - st);
                            Data curInput, curPositionIds;
                            Split(inputIds, 1, st, st + curLen, curInput);
                            if (positionIds[0] != nullptr) {
                                Split(*positionIds[0], 1, st, st + curLen, curPositionIds);
                            }
                            Data emptyAttention;
                            bool lastChunk = st + curLen >= len;
                            ScopedDeepSeekV4HistorySnapshotSuppress suppressSnapshot(!lastChunk);
                            ret = std::vector<int>{model->Forward(curInput, emptyAttention, curPositionIds,
                                                                  *pastKeyValue, generationConfigs[0],
                                                                  tokensManager, logits[0])};
                            st += curLen;
                        }
                    } else {
                        Data emptyAttention, emptyPosition;
                        ret = std::vector<int>{model->Forward(inputIds,
                                                              attentionMasks[0] == nullptr ? emptyAttention : *attentionMasks[0],
                                                              positionIds[0] == nullptr ? emptyPosition : *positionIds[0],
                                                              *pastKeyValue, generationConfigs[0],
                                                              tokensManager, logits[0])};
                    }
                }

                forwardLocker.unlock();
                dictLocker.lock();

                if (model->verbose) {
                    genTokens += seqLens.size();
                    auto nowTime = std::chrono::system_clock::now();
                    float spend = GetSpan(lastRecordTime, nowTime);
                    if (spend > 1) {
                        int alive = 0, pending = 0, aliveLen = 0;
                        for (auto &it : model->responseContextDict.dicts) {
                            if (it.second->isEnding) {
                                continue;
                            }
                            int ctxLen = getContextLen(it.second);
                            if (it.second->preTokens > 0 || ctxLen > 0) {
                                alive++;
                                aliveLen += ctxLen;
                            } else {
                                pending++;
                            }
                        }
                        printf("[DeepSeekV4 Decode] alive = %d, pending = %d, contextLen = %d, Speed: %f tokens / s.\n",
                               alive, pending, aliveLen, (float)genTokens / spend);
                        lastRecordTime = nowTime;
                        genTokens = 0;
                    }
                }

                int resultCount = std::min((int)handles.size(), (int)ret.size());
                for (int i = 0; i < resultCount; i++) {
                    auto contextIt = model->responseContextDict.dicts.find(handles[i]);
                    if (contextIt == model->responseContextDict.dicts.end()) {
                        continue;
                    }
                    ResponseContext *ctx = contextIt->second;
                    int curRet = ret[i];
                    if (curRet == model->eos_token_id ||
                        model->eos_token_ids.find(curRet) != model->eos_token_ids.end()) {
                        ctx->isEnding = true;
                        ctx->TryRecord(model);
                    } else {
                        auto itStopTk = ctx->generationConfig.stop_token_ids.find(curRet);
                        if (itStopTk != ctx->generationConfig.stop_token_ids.end()) {
                            ctx->isEnding = true;
                            ctx->TryRecord(model);
                        }
                    }
                    if (!ctx->isEnding) {
                        ctx->currentTokens = std::vector<int>{curRet};
                        ctx->resultTokenQueue.push(curRet);
                        ctx->allTokens.push_back(curRet);
                        ctx->tokens.Push(curRet);
                        ctx->curTokens++;
                        if (ctx->curTokens == ctx->generationConfig.output_token_limit ||
                            ctx->allTokens.size() >= model->max_positions) {
                            ctx->isEnding = true;
                            ctx->TryRecord(model);
                        }
                    }
                }
            } else {
                int maxLen = -1, select = -1;
                for (auto &it : model->responseContextDict.dicts) {
                    if (it.second->isEnding) {
                        continue;
                    }
                    int ctxLen = getContextLen(it.second);
                    if (ctxLen > maxLen) {
                        maxLen = ctxLen;
                        select = it.first;
                    }
                }
                if (select != -1 && maxTotalLens > 0 && maxLen >= maxTotalLens) {
                    model->responseContextDict.dicts[select]->isEnding = true;
                }
            }

            for (int i = 0; i < (int)attentionMasks.size(); i++) {
                delete attentionMasks[i];
            }
            for (int i = 0; i < (int)positionIds.size(); i++) {
                delete positionIds[i];
            }

            if (seqLens.empty()) {
                model->dictCV.wait(dictLocker);
            }
        }
    }

    DeepSeekV4Model::DeepSeekV4Model() {
        this->canDoBatchForward = false;
        this->canDoConcurrentForward = true;
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
        Data hiddenStates, hiddenStatesBeforeHcExpand;
        int startPos = 0;
        if (positionIds.dims.size() >= 2 && positionIds.Count(0) > 0) {
            auto pids = ReadTokenIds(positionIds);
            startPos = pids.empty() ? 0 : pids[0];
        }
        int originalStartPos = startPos;
        std::shared_ptr<DeepSeekV4RequestState> requestState = GetRequestState(pastKeyValues);
        std::vector<DeepSeekV4DecodeLayerCache> *decodeCachesPtr =
            requestState == nullptr ? &this->decodeLayerCaches : &requestState->decodeLayerCaches;
        std::vector<int> *historyTokensPtr =
            requestState == nullptr ? &this->deepseekV4HistoryTokens : &requestState->historyTokens;
        auto &activeDecodeLayerCaches = *decodeCachesPtr;
        auto &activeHistoryTokens = *historyTokensPtr;
        if (this->saveHistoryChat && !DeepSeekV4PrefixCacheDisabled() &&
            batch == 1 && inputIds.dims.size() >= 2 && inputIds.dims[1] > 1 &&
            !EnvFlagEnabled("FASTLLM_DSV4_PREFIX_CACHE_DISABLE_CHUNK_SPLIT")) {
            int seq = inputIds.dims[1];
            int finalTotalLen = originalStartPos + seq;
            int lastRecordBoundary = (finalTotalLen / 256) * 256;
            if (!DeepSeekV4PrefixCacheEveryBlockSplitEnabled() &&
                lastRecordBoundary > originalStartPos && lastRecordBoundary < finalTotalLen) {
                int prefixLen = lastRecordBoundary - originalStartPos;
                Data prefixInputIds, prefixPositionIds;
                Split(inputIds, 1, 0, prefixLen, prefixInputIds);
                if (positionIds.dims.size() >= 2) {
                    Split(positionIds, 1, 0, prefixLen, prefixPositionIds);
                } else {
                    std::vector<float> pids(prefixLen);
                    for (int i = 0; i < prefixLen; i++) {
                        pids[i] = (float)(originalStartPos + i);
                    }
                    prefixPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, prefixLen}, pids));
                }
                ForwardBatch(1, prefixInputIds, Data(), prefixPositionIds, pastKeyValues,
                             generationConfig, lastTokens, nullptr);

                Data suffixInputIds, suffixPositionIds;
                Split(inputIds, 1, prefixLen, seq, suffixInputIds);
                if (positionIds.dims.size() >= 2) {
                    Split(positionIds, 1, prefixLen, seq, suffixPositionIds);
                } else {
                    int suffixLen = seq - prefixLen;
                    std::vector<float> pids(suffixLen);
                    for (int i = 0; i < suffixLen; i++) {
                        pids[i] = (float)(lastRecordBoundary + i);
                    }
                    suffixPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, suffixLen}, pids));
                }
                return ForwardBatch(1, suffixInputIds, Data(), suffixPositionIds, pastKeyValues,
                                    generationConfig, lastTokens, retLogits);
            }

            int nextBoundary = ((originalStartPos / 256) + 1) * 256;
            if (DeepSeekV4PrefixCacheEveryBlockSplitEnabled() && originalStartPos + seq > nextBoundary) {
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
        bool useDecodeCache = batch == 1;
        Embedding(inputIds, weight["embed.weight"], hiddenStatesBeforeHcExpand);

        int bsz = inputIds.dims[0];
        int seqlen = inputIds.dims[1];
        int dim = embed_dim;


        {
            // hc expand
            hiddenStatesBeforeHcExpand.Reshape({bsz, seqlen, 1, dim});
            Repeat(hiddenStatesBeforeHcExpand, 2, hc_mult, hiddenStates);
        }
        
        if (useDecodeCache && originalStartPos == 0) {
            activeDecodeLayerCaches.clear();
            activeDecodeLayerCaches.resize(block_cnt);
        }

        std::vector<int> tokenIds = ReadTokenIds(inputIds);
        if (this->saveHistoryChat && !DeepSeekV4PrefixCacheDisabled() && batch == 1) {
            if (originalStartPos == 0) {
                activeHistoryTokens = tokenIds;
            } else if ((int)activeHistoryTokens.size() == originalStartPos) {
                activeHistoryTokens.insert(activeHistoryTokens.end(), tokenIds.begin(), tokenIds.end());
            } else if ((int)activeHistoryTokens.size() < originalStartPos) {
                if (DeepSeekV4PrefixCacheDebugEnabled()) {
                    printf("[fastllm-dsv4-prefix-cache] reset token history: history=%d start=%d add=%d\n",
                           (int)activeHistoryTokens.size(), originalStartPos, (int)tokenIds.size());
                    fflush(stdout);
                }
                activeHistoryTokens.clear();
            }
        }
        bool cudaSe = GetCudaSharedExpert();
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

        Data attnInput;
        Data qr, qNorm, q, kv;
        HcMix attnMix, ffnMix;
        Data hiddenStatesTemp;
        Data *curHiddenStates = &hiddenStates;
        Data *nextHiddenStates = &hiddenStatesTemp;
        auto runHcPost = [&](Data &input, const HcMix &mix) {
            DeepSeekV4HcPost(input, *curHiddenStates, mix.postData, mix.combData, *nextHiddenStates);
            std::swap(curHiddenStates, nextHiddenStates);
        };

        for (int layer = 0; layer < block_cnt; layer++) {
            std::string pre = "layers." + std::to_string(layer);
            int compressRatio = compress_ratios.size() > layer ? compress_ratios[layer] : 0;
            bool useCompressRope = compressRatio != 0;
            float layerRopeBase = useCompressRope ? compress_rope_theta : rope_base;
            int layerOriginalSeqLen = useCompressRope ? (int)rope_scaling_original_max_position_embeddings : 0;
            DeepSeekV4HcPre(*curHiddenStates, weight[pre + ".hc_attn_fn"],
                            weight[pre + ".hc_attn_scale"], weight[pre + ".hc_attn_base"],
                            hc_mult, hc_sinkhorn_iters, hc_eps, rms_norm_eps,
                            attnMix.y, attnMix.postData, attnMix.combData);
            attnMix.b = bsz;
            attnMix.s = seqlen;
            attnMix.hc = hc_mult;

            RMSNormReference(attnMix.y, weight[pre + ".attn_norm.weight"], rms_norm_eps, attnInput, DataType::BFLOAT16);
            Linear(attnInput, weight[pre + ".attn.wq_a.weight"], Data(), qr);
            RMSNormReference(qr, weight[pre + ".attn.q_norm.weight"], rms_norm_eps, qNorm, DataType::BFLOAT16);
            Linear(qNorm, weight[pre + ".attn.wq_b.weight"], Data(), q);
            q.Reshape({bsz, seqlen, num_attention_heads, head_dim_full});
            ScaleQRatory(q, rms_norm_eps, qk_rope_head_dim, layerRopeBase, startPos,
                         layerOriginalSeqLen, rope_factor, rope_scaling_beta_fast,
                         rope_scaling_beta_slow);
            Linear(attnInput, weight[pre + ".attn.wkv.weight"], Data(), kv);
            RMSNormReference(kv, weight[pre + ".attn.kv_norm.weight"], rms_norm_eps, kv, DataType::BFLOAT16);
            kv.Reshape({bsz, seqlen, 1, head_dim_full});
            DeepSeekV4RotaryQuant(kv, qk_rope_head_dim, layerRopeBase, startPos,
                                  layerOriginalSeqLen, rope_factor, rope_scaling_beta_fast,
                                  rope_scaling_beta_slow, head_dim_full - qk_rope_head_dim, 64);
            kv.Reshape({bsz, seqlen, head_dim_full});
            DeepSeekV4DecodeLayerCache *decodeCache = nullptr;
            if (useDecodeCache && layer < (int)activeDecodeLayerCaches.size()) {
                decodeCache = &activeDecodeLayerCaches[layer];
            }
            Data chunkPrefixKV;
            int chunkPrefixLen = 0;
            int decodeCompressedCount = 0;
            if (decodeCache != nullptr) {
                if (startPos == 0) {
                    decodeCache->initialized = true;
                    decodeCache->bsz = bsz;
                    decodeCache->totalLen = seqlen;
                    decodeCache->headDim = head_dim_full;
                    decodeCache->windowSize = window_size;
                    decodeCache->compressRatio = compressRatio;
                    decodeCache->compressorWideDim = (compressRatio == 4 ? 2 : 1) * head_dim_full;
                    decodeCache->compressorRawTokenBase = 0;
                    StoreWindowKVCache(kv, bsz, seqlen, head_dim_full, startPos, window_size,
                                       decodeCache->windowKV);
                } else {
                    if (!decodeCache->initialized) {
                        ErrorInFastLLM("DeepSeekV4Model: decode cache is not initialized.");
                    }
                    if (seqlen > 1) {
                        chunkPrefixLen = BuildWindowKVPrefixData(decodeCache->windowKV, bsz, head_dim_full,
                                                                 startPos, window_size, chunkPrefixKV);
                    }
                    decodeCache->totalLen = startPos + seqlen;
                    UpdateWindowKVCache(kv, bsz, head_dim_full, startPos, window_size,
                                        decodeCache->windowKV);
                }
            }
            const Data *decodeCompressedKVForAttention = nullptr;
            if (decodeCache != nullptr) {
                decodeCompressedKVForAttention = &decodeCache->compressedKV;
            }
            if (compressRatio > 0) {
                if (decodeCache != nullptr) {
                    Data compressorKV, compressorScore;
                    ComputeCompressorRaw(weight, pre + ".attn.compressor", attnInput, compressorKV, compressorScore);
                    int compressedCutoff = decodeCache->totalLen - (decodeCache->totalLen % compressRatio);
                    int targetCompressedBlocks = compressRatio > 0 ? compressedCutoff / compressRatio : 0;
                    bool targetCompressedReady = targetCompressedBlocks > 0 &&
                        decodeCache->compressedBlocks == targetCompressedBlocks &&
                        HasCompressedKVData(decodeCache->compressedKV);

                    const Data *compressorKVForBuild = &decodeCache->compressorKVRaw;
                    const Data *compressorScoreForBuild = &decodeCache->compressorScoreRaw;
                    int compressorRawTokenBaseForBuild = decodeCache->compressorRawTokenBase;
                    bool transientCompressorRaw = false;
                    if (startPos == 0) {
                        Copy(compressorKV, decodeCache->compressorKVRaw);
                        Copy(compressorScore, decodeCache->compressorScoreRaw);
                        decodeCache->compressorRawTokenBase = 0;
                    } else if (!targetCompressedReady && seqlen > 1 &&
                               !HasTensorData(decodeCache->compressorKVRaw) &&
                               !HasTensorData(decodeCache->compressorScoreRaw)) {
                        int reusableBlocks = GetReusableCompressedBlocks(decodeCache->compressedKV,
                                                                         bsz, targetCompressedBlocks,
                                                                         head_dim_full);
                        int firstNeededToken = reusableBlocks * compressRatio;
                        if (compressRatio == 4 && reusableBlocks > 0) {
                            firstNeededToken = (reusableBlocks - 1) * compressRatio;
                        }
                        int lastNeededToken = targetCompressedBlocks * compressRatio;
                        if (targetCompressedBlocks > reusableBlocks &&
                            startPos <= firstNeededToken && startPos + seqlen >= lastNeededToken) {
                            compressorKVForBuild = &compressorKV;
                            compressorScoreForBuild = &compressorScore;
                            compressorRawTokenBaseForBuild = startPos;
                            transientCompressorRaw = true;
                        } else {
                            AppendCompressorRaw(compressorKV, compressorScore, bsz, seqlen,
                                                decodeCache->compressorWideDim,
                                                decodeCache->compressorKVRaw,
                                                decodeCache->compressorScoreRaw);
                        }
                    } else {
                        AppendCompressorRaw(compressorKV, compressorScore, bsz, seqlen,
                                            decodeCache->compressorWideDim,
                                            decodeCache->compressorKVRaw,
                                            decodeCache->compressorScoreRaw);
                    }
                    if (targetCompressedReady) {
                        decodeCompressedCount = decodeCache->compressedBlocks;
                        decodeCompressedKVForAttention = &decodeCache->compressedKV;
                    } else {
                        bool builtCompressed = BuildCompressedKVFromRaw(
                            weight, pre + ".attn.compressor", *compressorKVForBuild,
                            *compressorScoreForBuild, bsz, compressorRawTokenBaseForBuild,
                            decodeCache->totalLen, compressRatio, head_dim_full,
                            qk_rope_head_dim, layerRopeBase, rope_factor,
                            rope_scaling_beta_fast, rope_scaling_beta_slow,
                            layerOriginalSeqLen, decodeCache->compressedKV, true);
                        if (!builtCompressed && transientCompressorRaw) {
                            AppendCompressorRaw(compressorKV, compressorScore, bsz, seqlen,
                                                decodeCache->compressorWideDim,
                                                decodeCache->compressorKVRaw,
                                                decodeCache->compressorScoreRaw);
                            builtCompressed = BuildCompressedKVFromRaw(
                                weight, pre + ".attn.compressor",
                                decodeCache->compressorKVRaw,
                                decodeCache->compressorScoreRaw, bsz,
                                decodeCache->compressorRawTokenBase,
                                decodeCache->totalLen, compressRatio, head_dim_full,
                                qk_rope_head_dim, layerRopeBase, rope_factor,
                                rope_scaling_beta_fast, rope_scaling_beta_slow,
                                layerOriginalSeqLen, decodeCache->compressedKV, true);
                            transientCompressorRaw = false;
                        }
                        if (builtCompressed) {
                            int builtBlocks = GetReusableCompressedBlocks(decodeCache->compressedKV,
                                                                          bsz, targetCompressedBlocks,
                                                                          head_dim_full);
                            decodeCache->compressedBlocks = builtBlocks;
                            decodeCompressedCount = decodeCache->compressedBlocks;
                            if (transientCompressorRaw) {
                                int retainStart = decodeCache->compressedBlocks * std::max(1, compressRatio);
                                if (compressRatio == 4 && decodeCache->compressedBlocks > 0) {
                                    retainStart = (decodeCache->compressedBlocks - 1) * compressRatio;
                                }
                                int rawEnd = startPos + seqlen;
                                retainStart = std::max(startPos, std::min(retainStart, rawEnd));
                                int tailLen = rawEnd - retainStart;
                                if (tailLen > 0) {
                                    Data tailKV, tailScore;
                                    int tailOffset = retainStart - startPos;
                                    Split(compressorKV, 1, tailOffset, seqlen, tailKV);
                                    Split(compressorScore, 1, tailOffset, seqlen, tailScore);
                                    CopyTensorData(decodeCache->compressorKVRaw, tailKV);
                                    CopyTensorData(decodeCache->compressorScoreRaw, tailScore);
                                    EnsureCompressorRawCapacity(decodeCache->compressorKVRaw, tailLen);
                                    EnsureCompressorRawCapacity(decodeCache->compressorScoreRaw, tailLen);
                                } else {
                                    ResetData(decodeCache->compressorKVRaw);
                                    ResetData(decodeCache->compressorScoreRaw);
                                }
                                decodeCache->compressorRawTokenBase = retainStart;
                            } else {
                                TrimCompressorRawCache(bsz, decodeCache->totalLen, compressRatio,
                                                       decodeCache->compressorWideDim,
                                                       decodeCache->compressedBlocks,
                                                       decodeCache->compressorKVRaw,
                                                       decodeCache->compressorScoreRaw,
                                                       decodeCache->compressorRawTokenBase);
                            }
                            if (startPos == 0) {
                                Data catKV;
                                const Data *prefillCompressed = &decodeCache->compressedKV;
                                if (HasCompressedKVData(*prefillCompressed)) {
                                    Cat(kv, *prefillCompressed, 1, catKV);
                                    Copy(catKV, kv);
                                }
                            }
                            decodeCompressedKVForAttention = &decodeCache->compressedKV;
                        }
                    }
                } else {
                    Data compressedKV;
                    if (CompressKVReference(weight, pre + ".attn.compressor", attnInput, compressRatio,
                                            head_dim_full, qk_rope_head_dim, layerRopeBase, rope_factor,
                                            rope_scaling_beta_fast, rope_scaling_beta_slow,
                                            layerOriginalSeqLen, startPos, compressedKV)) {
                        Data catKV;
                        Cat(kv, compressedKV, 1, catKV);
                        Copy(catKV, kv);
                    }
                }
            }

            Data attnOut4, woAOut, attnOut;
            Data sparsePrefillKV;
            Data *sparsePrefillKVPtr = &kv;
            int sparsePrefillPrefixLen = 0;
            if (decodeCache != nullptr && startPos > 0 && seqlen > 1) {
                sparsePrefillPrefixLen = chunkPrefixLen;
                if (chunkPrefixLen > 0) {
                    const Data *chunkPrefixForAttention = &chunkPrefixKV;
                    Data chunkPrefixTyped;
                    if (chunkPrefixKV.dataType != kv.dataType) {
                        ToDataType(chunkPrefixKV, chunkPrefixTyped, kv.dataType);
                        if (chunkPrefixTyped.dataDevice != kv.dataDevice) {
                            chunkPrefixTyped.ToDevice(kv.dataDevice);
                        }
                        chunkPrefixForAttention = &chunkPrefixTyped;
                    }
                    ConcatSeqReference(*chunkPrefixForAttention, kv, sparsePrefillKV);
                } else {
                    sparsePrefillKV.CopyFrom(kv);
                }
                if (decodeCompressedCount > 0 &&
                    HasCompressedKVData(decodeCache->compressedKV)) {
                    if (sparsePrefillKV.dataDevice != DataDevice::CUDA) {
                        EnsureCompressedKVOnCpu(*decodeCache);
                    }
                    const Data *prefillCompressed = &decodeCache->compressedKV;
                    Data catKV;
                    if (HasCompressedKVData(*prefillCompressed)) {
                        ConcatSeqReference(sparsePrefillKV, *prefillCompressed, catKV);
                        sparsePrefillKV.CopyFrom(catKV);
                    }
                }
                sparsePrefillKVPtr = &sparsePrefillKV;
            }
            if (decodeCache != nullptr && startPos > 0 && seqlen == 1) {
                SparseAttentionDecodeCachedReference(q, decodeCache->windowKV,
                                                     *decodeCompressedKVForAttention, weight[pre + ".attn.attn_sink"],
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
            DeepSeekV4WoA(attnOut4, weight[pre + ".attn.wo_a.weight"], o_groups, o_lora_rank, woAOut);
            Linear(woAOut, weight[pre + ".attn.wo_b.weight"], Data(), attnOut);
            runHcPost(attnOut, attnMix);
            DeepSeekV4HcPre(*curHiddenStates, weight[pre + ".hc_ffn_fn"],
                            weight[pre + ".hc_ffn_scale"], weight[pre + ".hc_ffn_base"],
                            hc_mult, hc_sinkhorn_iters, hc_eps, rms_norm_eps,
                            ffnMix.y, ffnMix.postData, ffnMix.combData);
            ffnMix.b = bsz;
            ffnMix.s = seqlen;
            ffnMix.hc = hc_mult;
            Data ffnInput, ffnOut;
            RMSNormReference(ffnMix.y, weight[pre + ".ffn_norm.weight"], rms_norm_eps, ffnInput, DataType::BFLOAT16);
            std::vector<int> ffnDims = ffnInput.dims;
            ffnInput.Reshape({bsz * seqlen, dim});
            Data expertIndex, expertScore;
            BuildMoERoutingData(weight, pre + ".ffn", ffnInput, tokenIds, num_experts,
                                num_experts_per_tok, scoring_func, routed_scaling_factor,
                                expertIndex, expertScore);
            {
                // MOE
                Data w1, w2, w3, tempInput, tempOutput, moeInputTemp, moeOutputTemp, sharedExpertOut;
                Data ww1, ww3;
                if (cudaSe &&
                    weight.weight.find(pre + ".ffn.shared_experts.gateup.weight") != weight.weight.end() &&
                    weight.weight.find(pre + ".ffn.shared_experts.w2.weight") != weight.weight.end()) {
                    LinearSwigluBlock(&ffnInput, &weight[pre + ".ffn.shared_experts.gateup.weight"], GetEmptyData(), &ww3, &ww1);
                    Linear(ww1, weight[pre + ".ffn.shared_experts.w2.weight"], *GetEmptyData(), sharedExpertOut);
                    weights[layer][0] = weights[layer][1] = nullptr;
                }
                ApplyDeviceMap(this->moeDeviceMap, layer + 1, block_cnt);
                // NumasMergeMOE 的小 batch 路径直接读取 cpuData，先保证输入在 CPU 可见。
                ffnInput.ToDevice(DataDevice::CPU);
                expertIndex.ToDevice(DataDevice::CPU);
                expertScore.ToDevice(DataDevice::CPU);
                MergeMOEBlock(&ffnInput, &expertIndex, &expertScore,
                              &weights[layer], &biass[layer],
                              &w1, &w2, &w3, &tempInput, &tempOutput,
                              1.0f, &ffnOut, layer,
                              ffnInput.dataType, this->moeAtype,
                              &moeInputTemp, &moeOutputTemp);
                ApplyDeviceMap(this->deviceMap, layer + 1, block_cnt);
                if (sharedExpertOut.dims.size() > 0) {
                    ffnOut.ToDevice(sharedExpertOut.dataDevice);
                    AddTo(ffnOut, sharedExpertOut);
                }
            }
            ffnOut.Reshape(ffnDims);
            runHcPost(ffnOut, ffnMix);
        }

        Data headStates, headInput;
        const Data *headSource = curHiddenStates;
        if (seqlen > 1) {
            Split(*curHiddenStates, 1, seqlen - 1, seqlen, headStates);
            headSource = &headStates;
        }
        HcHeadReference(*headSource, weight["hc_head_fn"], weight["hc_head_scale"], weight["hc_head_base"],
                        hc_mult, hc_eps, rms_norm_eps, headInput);

        std::vector<int> ret;
        std::vector<int> samplingSeqLens(batch, 1);
        std::vector<GenerationConfig> generationConfigs(batch, generationConfig);
        if (generationConfig.top_k <= 1) {
            generationConfigs[0].top_k = 5;
        }
        std::vector<std::pair<Data*, Data*> > samplingPastKeyValues;
        samplingPastKeyValues.reserve(pastKeyValues.size());
        for (auto &kv : pastKeyValues) {
            samplingPastKeyValues.push_back(std::make_pair(&kv.first, &kv.second));
        }
        LastTokensManager samplingLastTokens;
        const LastTokensManager *samplingLastTokensPtr = &lastTokens;
        if ((int)lastTokens.units.size() < batch) {
            samplingLastTokens = LastTokensManager(batch, generationConfig.last_n);
            samplingLastTokensPtr = &samplingLastTokens;
        }
        LLMSamplingBlock(this, &headInput, &weight["norm.weight"], &weight["head.weight"],
                         rms_norm_eps, batch, true, samplingSeqLens, samplingPastKeyValues,
                         generationConfigs, *samplingLastTokensPtr, retLogits, ret);

        int finalTotalLen = originalStartPos + inputIds.dims[1];
        UpdateDebugPastKeyValues(pastKeyValues, bsz, finalTotalLen, block_cnt);
        if (this->saveHistoryChat && !DeepSeekV4PrefixCacheDisabled() &&
            batch == 1 && finalTotalLen % 256 == 0 &&
            (int)activeHistoryTokens.size() >= finalTotalLen) {
            this->RecordHistorySnapshot(activeHistoryTokens, finalTotalLen, activeDecodeLayerCaches);
        } else if (this->saveHistoryChat && !DeepSeekV4PrefixCacheDisabled() &&
                   batch == 1 && finalTotalLen % 256 == 0 && DeepSeekV4PrefixCacheDebugEnabled()) {
            printf("[fastllm-dsv4-prefix-cache] skip boundary record: final_len=%d history_tokens=%d\n",
                   finalTotalLen, (int)activeHistoryTokens.size());
            fflush(stdout);
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
        printf("Warmup...\n");

        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {1});
        Data attentionMask = Data(DataType::FLOAT32, {1, 1}, {0});
        Data positionIds = Data(DataType::FLOAT32, {1, 1}, {0});

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(this->dataType),
                                                   Data(this->dataType)));
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);

        this->kvCacheId = 0;
        elementsInKVCachePerToken = 0;
        bool foundTokenGrowingCache = false;
        for (int i = 0; i < block_cnt; i++) {
            if (pastKeyValues[i].first.isLinearAttention || pastKeyValues[i].second.isLinearAttention) {
                continue;
            }
            if (pastKeyValues[i].first.dims.size() < 3 || pastKeyValues[i].second.dims.size() < 3) {
                continue;
            }
            if (!foundTokenGrowingCache) {
                this->kvCacheId = i;
                foundTokenGrowingCache = true;
            }
            elementsInKVCachePerToken +=
                (long long)pastKeyValues[i].first.dims[0] * pastKeyValues[i].first.dims[2] +
                (long long)pastKeyValues[i].second.dims[0] * pastKeyValues[i].second.dims[2];
        }
        printf("finish.\n");
    }

    std::string DeepSeekV4Model::MakeInput(const std::string &history, int round, const std::string &input) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
    }

    std::string DeepSeekV4Model::MakeHistory(const std::string &history, int round,
                                             const std::string &input, const std::string &output) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role + output + history_sep;
    }
}
