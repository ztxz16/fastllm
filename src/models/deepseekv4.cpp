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
#include "devices/cpu/computeutils.h"
#include "executor.h"
#include "utils.h"

#include <sstream>
#include <random>
#include <unordered_map>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <cstdlib>
#include <cctype>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <cstdio>

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif

#include "json11.hpp"

namespace fastllm {
#ifdef USE_CUDA
    void DoCudaCatDirect(Data &input0, Data &input1, int axis);
#endif
    // 复用 deepseekv2.cpp 中的 yarn 工具函数
    extern float yarn_find_correction_dim(int num_rotations, int dim, float base, int max_position_embeddings);
    extern void yarn_find_correction_range(int low_rot, int high_rot, int dim, float base, int max_position_embeddings, int &low, int &high);
    extern float yarn_get_mscale(float scale, float mscale);
    extern std::vector <float> yarn_linear_ramp_mask(float min, float max, int dim);

    namespace {
        static constexpr int DEEPSEEK_V4_DSPARK_LOG_INTERVAL = 64;
        static constexpr int DEEPSEEK_V4_DSPARK_CALIBRATION_ROUNDS = 7;

        // A DSpark target verification writes a speculative multi-token suffix
        // into the ordinary decode cache.  Keep the compressor raw rows alive
        // until the verifier decides how much of that suffix to commit; the
        // normal trim point may otherwise advance past a rejected token.
        static thread_local int gDeepSeekV4DsparkVerificationDepth = 0;

        struct ScopedDeepSeekV4DsparkVerification {
            ScopedDeepSeekV4DsparkVerification() {
                gDeepSeekV4DsparkVerificationDepth++;
            }

            ~ScopedDeepSeekV4DsparkVerification() {
                gDeepSeekV4DsparkVerificationDepth--;
            }
        };

        static bool DeepSeekV4DsparkVerificationActive() {
            return gDeepSeekV4DsparkVerificationDepth > 0;
        }

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

        static bool DeepSeekV4SparseMlaSm120Enabled() {
            const char *v = std::getenv(
                "FASTLLM_CUDA_DSV4_SPARSE_MLA_SM120");
            return v == nullptr || EnvFlagEnabled(
                "FASTLLM_CUDA_DSV4_SPARSE_MLA_SM120");
        }

        static bool DeepSeekV4PairedAllReduceEnabled() {
#ifdef USE_CUDA
            return FastllmCudaCustomAllReduceEnabled() &&
                   !EnvFlagEnabled(
                       "FASTLLM_DSV4_DISABLE_PAIRED_ALLREDUCE");
#else
            return false;
#endif
        }

        static bool DeepSeekV4LinearColumnLocal(
                Data &input, Data &weight, Data &bias, Data &output) {
#ifdef USE_CUDA
            return MultiCudaLinearColumnLocal(input, weight, bias, output);
#else
            return false;
#endif
        }

        static void PrepareDeepSeekV4SamplingConfig(GenerationConfig &config) {
            // top_k=1 is the public greedy default. Only widen it for an
            // explicitly sampled request; otherwise batch decode would turn a
            // deterministic request into top-k sampling.
            if (config.do_sample && config.top_k <= 1 &&
                config.temperature > 1e-6f) {
                config.top_k = 5;
            }
        }

        static int EnvInt(const char *name, int fallback) {
            const char *v = std::getenv(name);
            return v == nullptr ? fallback : atoi(v);
        }

        static bool DeepSeekV4DeviceSpecUsesType(const std::string &deviceSpec,
                                                 const std::string &deviceType) {
            std::string normalized = deviceSpec;
            std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                           [](unsigned char c) { return (char)std::tolower(c); });
            return normalized == deviceType ||
                   normalized.rfind(deviceType + ":", 0) == 0;
        }

        static bool DeepSeekV4DeviceMapUsesMultiCuda(const std::map<std::string, int> &deviceMap) {
            for (const auto &it : deviceMap) {
                if (DeepSeekV4DeviceSpecUsesType(it.first, "multicuda")) {
                    return true;
                }
            }
            return false;
        }

        static bool DeepSeekV4DeviceMapUsesSingleCuda(
                const std::map<std::string, int> &deviceMap) {
#ifdef USE_CUDA
            if (deviceMap.size() != 1) {
                return false;
            }
            const std::string &device = deviceMap.begin()->first;
            if (device == "cuda") {
                return true;
            }
            if (device.rfind("cuda:", 0) != 0) {
                return false;
            }
            const std::string id = device.substr(5);
            return !id.empty() &&
                   std::all_of(
                       id.begin(), id.end(),
                       [](unsigned char c) { return std::isdigit(c); });
#else
            return false;
#endif
        }

        static bool DeepSeekV4PreferCuda() {
#ifdef USE_CUDA
            auto *executor = (Executor*)GetExecutor();
            return executor != nullptr &&
                   (executor->firstDevice == "cuda" ||
                    executor->firstDevice.rfind("cuda:", 0) == 0 ||
                    executor->firstDevice == "multicuda" ||
                    executor->firstDevice.rfind("multicuda:", 0) == 0);
#else
            return false;
#endif
        }

        static std::vector<float> ReadFloatData(const Data &input);
        static void WriteFloatData(const std::vector<float> &values,
                                   const std::vector<int> &dims,
                                   Data &output, DataType dtype);

#ifdef USE_CUDA
        static int GetTensorCudaDevice(const Data &data) {
            if (data.cudaData != nullptr) {
                int device = GetPointerDeviceId(data.cudaData);
                if (device >= 0) {
                    return device;
                }
            }
            if (data.multiDeviceData) {
                for (const auto &deviceData : data.multiDeviceDatas) {
                    if (deviceData.second != nullptr &&
                        deviceData.second->cudaData != nullptr) {
                        int device = GetPointerDeviceId(deviceData.second->cudaData);
                        if (device >= 0) {
                            return device;
                        }
                    }
                }
            }
            return -1;
        }

        static Data *GetTensorCudaReplica(Data &data, int device) {
            if (data.multiDeviceData) {
                auto it = data.multiDeviceDatas.find(device);
                if (it != data.multiDeviceDatas.end() && it->second != nullptr &&
                    it->second->cudaData != nullptr &&
                    GetPointerDeviceId(it->second->cudaData) == device) {
                    return it->second;
                }
                return nullptr;
            }
            if (data.cudaData != nullptr && GetPointerDeviceId(data.cudaData) == device) {
                return &data;
            }
            return nullptr;
        }

        static const Data *GetTensorCudaReplica(const Data &data, int device) {
            return GetTensorCudaReplica(const_cast<Data&>(data), device);
        }

        static bool CopyDeepSeekV4CudaTensorToCpu(
            const Data &source, Data &destination
        ) {
            int device = GetTensorCudaDevice(source);
            if (device < 0) {
                return false;
            }
            const Data *replica =
                GetTensorCudaReplica(source, device);
            if (replica == nullptr || replica->cudaData == nullptr ||
                replica->dataType != source.dataType ||
                replica->dims != source.dims) {
                return false;
            }
            destination.dataType = source.dataType;
            destination.Resize(source.dims);
            destination.Allocate(false);
            FastllmCudaSetDevice(device);
            // Persistent tensor-parallel operators publish their results from
            // per-device worker streams.  RunMultiCudaDeviceOps wires those
            // completion events into the caller's per-thread stream, but the
            // synchronous cudaMemcpy used below runs on CUDA's legacy copy
            // path and does not reliably observe that stream dependency.
            // Finish the caller stream explicitly before handing routing data
            // to the NUMA MoE implementation.
            if (MultiCudaCurrentThreadWaitForWorker(device)) {
                FastllmCudaSyncCurrentThreadStream();
            } else {
                FastllmCudaSyncDevice(device);
            }
            FastllmCudaCopyFromDeviceToHost(
                destination.cpuData, replica->cudaData,
                destination.GetBytes());
            // Keep the source allocation as a non-owning mirror.  NUMA MoE
            // uses the CPU copy for its local experts, while mixed inference
            // can still recognize that this tensor genuinely came from CUDA
            // and reuse the existing replica instead of staging it again.
            destination.cudaData = replica->cudaData;
            destination.cudaDataBorrowed = true;
            destination.dataDeviceIds = {device};
            return true;
        }

        static bool DeepSeekV4NumasGpuPrefillEnabled() {
            const char *value = std::getenv("FT_GPU_PREFILL");
            if (value == nullptr) {
                return true;
            }
            std::string normalized(value);
            std::transform(
                normalized.begin(), normalized.end(), normalized.begin(),
                [](unsigned char c) { return (char)std::tolower(c); });
            return normalized != "0" && normalized != "false" &&
                   normalized != "off";
        }

        // NUMA evaluates the official FP8 activation boundary on CPU, while
        // its large-prefill path may stream selected experts to CUDA.  Publish
        // the already quantized/dequantized BF16 bytes on that CUDA device;
        // borrowing the original pre-quantization activation would make the
        // CPU and GPU expert subsets observe different model inputs.
        static bool AddDeepSeekV4QuantizedCudaReplica(
            Data &activation, int device
        ) {
            if (device < 0 || activation.dataDevice != DataDevice::CPU ||
                activation.cpuData == nullptr || activation.cudaData != nullptr) {
                return false;
            }
            activation.ToDevice(
                DataDevice::CUDA, std::vector<int>{device}, true);
            if (activation.cudaData == nullptr ||
                GetPointerDeviceId(activation.cudaData) != device) {
                return false;
            }
            // Activation storage is not a model weight or KV cache, so the
            // CPU allocation remains valid.  Switching the authoritative view
            // back does not copy data and retains the owned CUDA allocation.
            activation.ToDevice(
                DataDevice::CPU, std::vector<int>{device}, false);
            return activation.cpuData != nullptr &&
                   activation.cudaData != nullptr &&
                   !activation.cudaDataBorrowed;
        }

        static std::vector<int> GetReplicatedCudaDevices(const Data &data) {
            std::vector<int> devices;
            if (data.multiDeviceData && data.IsTensorParallelReplicated()) {
                for (const auto &it : data.multiDeviceDatas) {
                    if (it.second != nullptr && it.second->cudaData != nullptr &&
                        GetPointerDeviceId(it.second->cudaData) == it.first) {
                        devices.push_back(it.first);
                    }
                }
            } else if (data.cudaData != nullptr) {
                int device = GetPointerDeviceId(data.cudaData);
                if (device >= 0) {
                    devices.push_back(device);
                }
            }
            return devices;
        }

        static std::vector<int> GetTensorCudaDevices(const Data &data) {
            std::vector<int> devices;
            if (data.multiDeviceData) {
                for (const auto &it : data.multiDeviceDatas) {
                    if (it.second != nullptr && it.second->cudaData != nullptr &&
                        GetPointerDeviceId(it.second->cudaData) == it.first) {
                        devices.push_back(it.first);
                    }
                }
            } else if (data.cudaData != nullptr) {
                int device = GetPointerDeviceId(data.cudaData);
                if (device >= 0) {
                    devices.push_back(device);
                }
            }
            return devices;
        }

        static std::vector<int> GetDeepSeekV4TensorParallelDevices(
                const std::map<std::string, int> &deviceMap) {
            std::vector<int> devices;
            if (!DeepSeekV4DeviceMapUsesMultiCuda(deviceMap)) {
                return devices;
            }
            std::map<int, int> ratios;
            FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
            return devices;
        }

        static void SelectDeepSeekV4TensorParallelRoot(
                const std::map<std::string, int> &deviceMap) {
            std::vector<int> devices = GetDeepSeekV4TensorParallelDevices(deviceMap);
            if (!devices.empty()) {
                FastllmCudaSetDevice(devices.front());
            }
        }

        static void SynchronizeDeepSeekV4TensorParallelDevices(
                const std::map<std::string, int> &deviceMap) {
            std::vector<int> devices = GetDeepSeekV4TensorParallelDevices(deviceMap);
            if (devices.empty()) {
                ForceDeviceSync();
                return;
            }
            for (int device : devices) {
                FastllmCudaSetDevice(device);
                // cudaDeviceSynchronize only observes work that a worker has
                // already submitted.  Persistent MultiCUDA operators publish
                // a completion event from the worker stream, so first attach
                // that event to this thread's stream and wait for it here.
                // This is required at cache snapshot/restore boundaries where
                // the producer workspace can otherwise be reused too early.
                if (MultiCudaCurrentThreadWaitForWorker(device)) {
                    FastllmCudaSyncCurrentThreadStream();
                }
                FastllmCudaSyncDevice(device);
            }
            // A chunked prefill recursively enters ForwardBatch again.  Keep its
            // embedding/root allocation deterministic even if a cache helper's
            // final local operation ran on the last TP rank.
            FastllmCudaSetDevice(devices.front());
        }

#endif

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
            if (input.multiDeviceData && !input.multiDeviceDatas.empty()) {
                if (!input.IsTensorParallelSharded()) {
                    for (const auto &it : input.multiDeviceDatas) {
                        if (it.second != nullptr) {
                            return ReadFloatData(*it.second);
                        }
                    }
                    return {};
                }

                const std::vector<int> &globalDims = input.tpGlobalDims.empty() ? input.dims : input.tpGlobalDims;
                if (globalDims.empty()) {
                    return {};
                }
                int axis = input.tpAxis;
                axis = (axis % (int)globalDims.size() + (int)globalDims.size()) % (int)globalDims.size();
                uint64_t outer = 1, inner = 1;
                for (int i = 0; i < axis; i++) {
                    outer *= (uint64_t)globalDims[i];
                }
                for (int i = axis + 1; i < (int)globalDims.size(); i++) {
                    inner *= (uint64_t)globalDims[i];
                }
                int globalAxis = globalDims[axis];
                std::vector<float> ret(outer * (uint64_t)globalAxis * inner, 0.0f);
                for (const auto &it : input.multiDeviceDatas) {
                    Data *local = it.second;
                    auto rangeIt = input.tpRanges.find(it.first);
                    if (local == nullptr || rangeIt == input.tpRanges.end() || rangeIt->second.empty()) {
                        continue;
                    }
                    std::vector<float> localValues = ReadFloatData(*local);
                    int localAxis = 0;
                    for (const auto &range : rangeIt->second) {
                        localAxis += std::max(0, range.second - range.first);
                    }
                    if (localAxis <= 0 || localValues.size() < outer * (uint64_t)localAxis * inner) {
                        continue;
                    }
                    int localOffset = 0;
                    for (const auto &range : rangeIt->second) {
                        int width = std::max(0, range.second - range.first);
                        for (uint64_t o = 0; o < outer; o++) {
                            memcpy(ret.data() + (o * (uint64_t)globalAxis + range.first) * inner,
                                   localValues.data() + (o * (uint64_t)localAxis + localOffset) * inner,
                                   (uint64_t)width * inner * sizeof(float));
                        }
                        localOffset += width;
                    }
                }
                return ret;
            }
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

        static bool IsDeepSeekV4QuantizedLinearWeight(const Data &weight) {
            return weight.dataType == DataType::FP8_E4M3 ||
                   weight.dataType == DataType::FP8_E4M3_BLOCK_128 ||
                   weight.dataType == DataType::FP8_E4M3_PERCHANNEL ||
                   weight.dataType == DataType::NVFP4 ||
                   weight.dataType == DataType::NVFP4_BLOCK_16 ||
                   weight.dataType == DataType::NVFP4_BLOCK_16_E8M0 ||
                   weight.dataType == DataType::NVFP4_BLOCK_32_E8M0;
        }

        static void ResetData(Data &data);

        // Match inference/kernel.py::act_quant for a non-inplace quantized
        // Linear: BF16 -> FP8 E4M3FN with one UE8M0 (power-of-two) scale per
        // 128 values, then dequantize back to BF16 for the CPU weight kernels.
        // The dequantized representation is mathematically equivalent to the
        // FP8 values and scales consumed by the official GEMM.
        static void DeepSeekV4QuantizeLinearActivationCpu(const Data &input, Data &output) {
            AssertInFastLLM(input.dataType == DataType::BFLOAT16 && !input.dims.empty() &&
                            input.dims.back() % 128 == 0,
                            "DeepSeek-V4 FP8 activation quantization expects BF16 rows divisible by 128.\n");
            if (input.dataDevice == DataDevice::CPU && input.cpuData != nullptr &&
                std::getenv("FASTLLM_DSV4_DISABLE_CPU_ACT_QUANT_FAST") == nullptr) {
                ResetData(output);
                output = Data(DataType::BFLOAT16, input.dims);
                output.Allocate(false);
                RunCpuDeepSeekV4ActivationQuantization(
                    (const uint16_t*)input.cpuData,
                    (uint16_t*)output.cpuData, input.Count(0));
                return;
            }
            std::vector<float> values = ReadFloatData(input);
            int dim = input.dims.back();
            int rows = (int)(values.size() / dim);
            for (int row = 0; row < rows; row++) {
                float *rowData = values.data() + (uint64_t)row * dim;
                QuantizeDequantizeFP8E4M3Block128(rowData, dim);
            }
            WriteFloatData(values, input.dims, output, DataType::BFLOAT16);
        }

        static void DeepSeekV4Linear(Data &input, Data &weight, const Data &bias,
                                     Data &output, bool keepTpReplicated = false) {
            if (!DeepSeekV4PreferCuda() && input.dataDevice == DataDevice::CPU &&
                input.dataType == DataType::BFLOAT16 &&
                IsDeepSeekV4QuantizedLinearWeight(weight)) {
                Data quantizedInput;
                DeepSeekV4QuantizeLinearActivationCpu(input, quantizedInput);
                Linear(quantizedInput, weight, bias, output, keepTpReplicated);
            } else {
                Linear(input, weight, bias, output, keepTpReplicated);
            }
        }

        static void ResetData(Data &data) {
            if (data.multiDeviceData) {
                for (auto &it : data.multiDeviceDatas) {
                    delete it.second;
                }
                data.multiDeviceDatas.clear();
                data.multiDeviceData = false;
                data.ClearTensorParallelLayout();
            }
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
            if (data.dims.empty() || data.Count(0) == 0) {
                return false;
            }
            if (data.cpuData != nullptr || data.cudaData != nullptr) {
                return true;
            }
            if (data.multiDeviceData) {
                for (const auto &it : data.multiDeviceDatas) {
                    if (it.second != nullptr &&
                        (it.second->cpuData != nullptr || it.second->cudaData != nullptr)) {
                        return true;
                    }
                }
            }
            return false;
        }

        static bool IsDiskWeight(const Data *weight) {
            return weight != nullptr && weight->isDiskWeight;
        }

#ifdef USE_CUDA
        static std::vector<int> GetCudaDeviceIdsForData(const Data &data) {
            if (data.cudaData != nullptr) {
                int realDevice = GetPointerDeviceId(data.cudaData);
                if (realDevice >= 0) {
                    return {realDevice};
                }
            }
            if (!data.dataDeviceIds.empty()) {
                return data.dataDeviceIds;
            }
            return {FastllmCudaGetDevice()};
        }

        static void PublishReplicatedCudaRootMetadata(
                Data &data, const std::vector<int> &devices,
                bool releaseRootPayload) {
            if (!data.multiDeviceData || !data.IsTensorParallelReplicated() ||
                devices.empty()) {
                return;
            }
            Data *first = nullptr;
            for (int device : devices) {
                auto it = data.multiDeviceDatas.find(device);
                if (it != data.multiDeviceDatas.end() && it->second != nullptr) {
                    first = it->second;
                    break;
                }
            }
            AssertInFastLLM(first != nullptr,
                            "DeepSeek V4 replicated CUDA tensor has no local metadata.\n");
            if (releaseRootPayload) {
                // Once every rank owns an independent copy, the root is metadata
                // only.  Keeping the restored CPU allocation here is unsafe: its
                // 128-row backing store becomes stale as the local tensors expand
                // to 256 rows and later generic layout checks may treat that CPU
                // pointer as a valid source for the enlarged logical tensor.
                data.FreeSpace();
            }
            data.dataType = first->dataType;
            data.UpdateUnitSize();
            data.dims = first->dims;
            data.strides = first->strides;
            data.expansionDims = first->expansionDims;
            data.expansionSize = first->expansionSize;
            data.expansionBytes = first->expansionBytes;
            data.dataDevice = DataDevice::CUDA;
            data.dataDeviceIds = devices;
            data.tpLayout = TP_LAYOUT_REPLICATED;
            data.tpAxis = -1;
            data.tpGlobalDims = data.dims;
        }

        static void EnsureTensorOnSameCudaDevice(Data &data, const Data &reference) {
            if (!DeepSeekV4PreferCuda() || reference.dataDevice != DataDevice::CUDA ||
                !HasTensorData(data)) {
                return;
            }
            if (reference.multiDeviceData && reference.IsTensorParallelReplicated()) {
                std::vector<int> devices = GetReplicatedCudaDevices(reference);
                if (!devices.empty()) {
                    // Prefix-cache restoration keeps one CPU/GPU payload to
                    // avoid storing an identical snapshot per TP rank.  Before
                    // a cache update, recreate the replicated physical layout
                    // of the new KV tensor; otherwise executor device selection
                    // can fall back to the CPU op with CUDA-only input buffers.
                    data.lockInCPU = false;
                    PrepareMultiCudaReplicatedData(data, devices, true);
                    PublishReplicatedCudaRootMetadata(data, devices, true);
                    return;
                }
            }
            data.lockInCPU = false;
            data.ToDevice(DataDevice::CUDA, GetCudaDeviceIdsForData(reference));
        }
#endif

        static void CopyTensorData(Data &dst, const Data &src) {
            ResetData(dst);
            if (!HasTensorData(src)) {
                return;
            }
#ifdef USE_CUDA
            if (src.multiDeviceData) {
                const int originalDevice = FastllmCudaGetDevice();
                std::vector<int> devices;
                for (const auto &it : src.multiDeviceDatas) {
                    if (it.second != nullptr) {
                        devices.push_back(it.first);
                    }
                }
                dst.dataType = src.dataType;
                dst.Resize(src.dims);
                if (src.IsTensorParallelSharded()) {
                    PrepareMultiCudaShardedData(dst, devices,
                        src.tpGlobalDims.empty() ? src.dims : src.tpGlobalDims,
                        src.tpAxis, src.tpRanges);
                } else {
                    PrepareMultiCudaReplicatedData(dst, devices, false);
                }
                for (int device : devices) {
                    Data *srcLocal = src.multiDeviceDatas.at(device);
                    Data *dstLocal = dst.multiDeviceDatas.at(device);
                    FastllmCudaSetDevice(device);
                    dstLocal->dataType = srcLocal->dataType;
                    dstLocal->UpdateUnitSize();
                    dstLocal->Resize(srcLocal->dims);
                    if (!srcLocal->expansionDims.empty() &&
                        srcLocal->expansionSize > 0) {
                        // KV-cache tensors keep a small logical shape inside a
                        // much larger strided backing allocation.  GetBytes()
                        // follows those expanded strides, so allocating only
                        // the logical shape before the copy corrupts the next
                        // CUDA allocation (typically the residual prefill
                        // workspace after a 256-token prefix snapshot).
                        dstLocal->strides = srcLocal->strides;
                        dstLocal->expansionDims = srcLocal->expansionDims;
                        dstLocal->MallocSpace(srcLocal->expansionSize, false);
                    } else {
                        dstLocal->Allocate(false);
                    }
                    AssertInFastLLM(
                        dstLocal->expansionBytes >= srcLocal->GetBytes(),
                        "DeepSeek V4 cache snapshot allocation is smaller than its source layout.\n");
                    if (srcLocal->cudaData != nullptr) {
                        FastllmCudaCopyFromDeviceToDevice(dstLocal->cudaData, srcLocal->cudaData,
                                                         srcLocal->GetBytes());
                    } else if (srcLocal->cpuData != nullptr) {
                        memcpy(dstLocal->cpuData, srcLocal->cpuData, srcLocal->GetBytes());
                    }
                }
                dst.strides = src.strides;
                dst.expansionDims = src.expansionDims;
                dst.expansionSize = src.expansionSize;
                dst.expansionBytes = src.expansionBytes;
                PublishReplicatedCudaRootMetadata(dst, devices, false);
                FastllmCudaSetDevice(originalDevice);
                return;
            }
#endif
            Copy(src, dst);
        }

        // Reuse an already matching CUDA allocation and enqueue the copy on
        // the per-thread streams that carry MultiCUDA event dependencies.
        // CopyTensorData is kept as the one-time/layout-changing fallback, but
        // its ResetData + synchronous copies are too expensive in every
        // speculative decode round.
        static bool CopyTensorDataInPlaceCuda(Data &dst, const Data &src) {
#ifdef USE_CUDA
            if (dst.dataType != src.dataType || dst.dims != src.dims ||
                dst.multiDeviceData != src.multiDeviceData) {
                return false;
            }
            if (src.multiDeviceData) {
                if (dst.tpLayout != src.tpLayout ||
                    dst.tpAxis != src.tpAxis ||
                    dst.tpGlobalDims != src.tpGlobalDims ||
                    dst.tpRanges != src.tpRanges ||
                    dst.multiDeviceDatas.size() !=
                        src.multiDeviceDatas.size()) {
                    return false;
                }
                std::vector<int> devices;
                for (const auto &it : src.multiDeviceDatas) {
                    auto dstIt = dst.multiDeviceDatas.find(it.first);
                    if (it.second == nullptr ||
                        dstIt == dst.multiDeviceDatas.end() ||
                        dstIt->second == nullptr ||
                        it.second->cudaData == nullptr ||
                        dstIt->second->cudaData == nullptr ||
                        it.second->dataType != dstIt->second->dataType ||
                        it.second->dims != dstIt->second->dims ||
                        it.second->GetBytes() !=
                            dstIt->second->GetBytes()) {
                        return false;
                    }
                    devices.push_back(it.first);
                }
                std::vector<char> copied(devices.size(), 0);
                std::function<void(int, int)> task =
                    [&](int rank, int device) {
                    const Data *source = src.multiDeviceDatas.at(device);
                    Data *destination = dst.multiDeviceDatas.at(device);
                    copied[rank] =
                        FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(
                            destination->cudaData, source->cudaData,
                            source->GetBytes());
                };
                return !devices.empty() &&
                    MultiCudaRunDeviceCallbacks(devices, task) &&
                    std::all_of(copied.begin(), copied.end(),
                        [](char state) { return state != 0; });
            }
            if (src.dataDevice != DataDevice::CUDA ||
                dst.dataDevice != DataDevice::CUDA ||
                src.cudaData == nullptr || dst.cudaData == nullptr ||
                src.GetBytes() != dst.GetBytes()) {
                return false;
            }
            const int sourceDevice = GetPointerDeviceId(src.cudaData);
            const int destinationDevice = GetPointerDeviceId(dst.cudaData);
            if (sourceDevice < 0 || sourceDevice != destinationDevice) {
                return false;
            }
            const int originalDevice = FastllmCudaGetDevice();
            FastllmCudaSetDevice(sourceDevice);
            const bool copied =
                FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(
                    dst.cudaData, src.cudaData, src.GetBytes());
            FastllmCudaSetDevice(originalDevice);
            return copied;
#else
            (void)dst;
            (void)src;
            return false;
#endif
        }

        static void CopyHistoryTensorData(Data &dst, const Data &src) {
            ResetData(dst);
            if (!HasTensorData(src)) {
                return;
            }

            const Data *payload = &src;
#ifdef USE_CUDA
            if (src.multiDeviceData) {
                payload = nullptr;
                for (const auto &it : src.multiDeviceDatas) {
                    if (it.second != nullptr &&
                        (it.second->cudaData != nullptr || it.second->cpuData != nullptr)) {
                        payload = it.second;
                        break;
                    }
                }
                AssertInFastLLM(payload != nullptr,
                                "DeepSeek V4 history snapshot has no replicated payload.\n");
            }
#endif

            Data compactPayload;
            if (payload->dims.size() >= 2 &&
                !payload->expansionDims.empty()) {
                // GetBytes() follows expanded strides. Materialize the logical
                // sequence before copying so a graph-sized reserve is not
                // mistaken for live prefix data.
                Split(*payload, 1, 0, payload->dims[1], compactPayload);
                payload = &compactPayload;
            }

            dst.dataType = payload->dataType;
            dst.UpdateUnitSize();
            dst.Resize(payload->dims);
            dst.dataDevice = DataDevice::CPU;
            dst.dataDeviceIds.clear();
            // CUDA Graph expands compressed caches to the request's planned
            // output bound.  A prefix snapshot only owns the logical rows;
            // retaining graph capacity here can copy gigabytes of unused
            // storage per record and ties the next request to stale addresses.
            dst.Allocate(false);
            AssertInFastLLM(dst.expansionBytes >= payload->GetBytes(),
                            "DeepSeek V4 CPU history snapshot allocation is too small.\n");

            if (payload->cpuData != nullptr) {
                memcpy(dst.cpuData, payload->cpuData, payload->GetBytes());
            }
#ifdef USE_CUDA
            else if (payload->cudaData != nullptr) {
                const int originalDevice = FastllmCudaGetDevice();
                int sourceDevice = GetPointerDeviceId(payload->cudaData);
                AssertInFastLLM(sourceDevice >= 0,
                                "DeepSeek V4 history snapshot has an invalid CUDA pointer.\n");
                FastllmCudaSetDevice(sourceDevice);
                FastllmCudaCopyFromDeviceToHost(dst.cpuData, payload->cudaData,
                                                payload->GetBytes());
                FastllmCudaSetDevice(originalDevice);
            }
#endif
            else {
                ErrorInFastLLM("DeepSeek V4 history snapshot has no readable payload.\n");
            }
            dst.isKVCache = payload->isKVCache;
            dst.lockInCPU = true;
            dst.ClearTensorParallelLayout();
        }

        static void CatDirectTensor(Data &dst, const Data &src, int axis) {
#ifdef USE_CUDA
            if (dst.multiDeviceData && src.multiDeviceData) {
                const int originalDevice = FastllmCudaGetDevice();
                for (auto &it : dst.multiDeviceDatas) {
                    int device = it.first;
                    auto srcIt = src.multiDeviceDatas.find(device);
                    AssertInFastLLM(it.second != nullptr && srcIt != src.multiDeviceDatas.end() &&
                                    srcIt->second != nullptr,
                                    "DeepSeek V4 MultiCuda cache append is missing a local tensor.\n");
                    FastllmCudaSetDevice(device);
                    DoCudaCatDirect(*it.second, *srcIt->second, axis);
                }
                Data *first = dst.multiDeviceDatas.begin()->second;
                (void)first;
                std::vector<int> devices;
                devices.reserve(dst.multiDeviceDatas.size());
                for (const auto &local : dst.multiDeviceDatas) {
                    devices.push_back(local.first);
                }
                PublishReplicatedCudaRootMetadata(dst, devices, false);
                FastllmCudaSetDevice(originalDevice);
                return;
            }
            if (!dst.multiDeviceData && src.multiDeviceData) {
                AssertInFastLLM(src.IsTensorParallelReplicated(),
                                "DeepSeek V4 cache append cannot collapse a sharded source tensor.\n");
                int device = dst.dataDeviceIds.empty() ? 0 : dst.dataDeviceIds[0];
                auto srcIt = src.multiDeviceDatas.find(device);
                AssertInFastLLM(srcIt != src.multiDeviceDatas.end() && srcIt->second != nullptr &&
                                srcIt->second->cudaData != nullptr,
                                "DeepSeek V4 cache append is missing the source replica on the target device.\n");
                const int originalDevice = FastllmCudaGetDevice();
                FastllmCudaSetDevice(device);
                DoCudaCatDirect(dst, *srcIt->second, axis);
                FastllmCudaSetDevice(originalDevice);
                return;
            }
#endif
            CatDirect(dst, src, axis);
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
#ifdef USE_CUDA
            if (data.multiDeviceData) {
                const int originalDevice = FastllmCudaGetDevice();
                std::vector<int> devices;
                for (auto &it : data.multiDeviceDatas) {
                    if (it.second != nullptr) {
                        // Expansion allocates and frees on the current CUDA
                        // device.  Each replicated cache owns a rank-local
                        // pointer, so expanding every local tensor while the
                        // scheduler happens to be on the last-used rank puts
                        // several allocations on the wrong GPU and leaves the
                        // advertised device metadata invalid.
                        FastllmCudaSetDevice(it.first);
                        devices.push_back(it.first);
                        EnsureCompressorRawCapacity(*it.second, targetLen);
                    }
                }
                PublishReplicatedCudaRootMetadata(data, devices, false);
                FastllmCudaSetDevice(originalDevice);
                return;
            }
#endif
            data.Expansion(newDims);
        }

        static void ResizeTensorSequenceInPlace(Data &data, int sequenceLen) {
            AssertInFastLLM(sequenceLen >= 0,
                            "DeepSeek V4 cache sequence length cannot be negative.\n");
            if (data.dims.size() != 3) {
                AssertInFastLLM(!HasTensorData(data),
                                "DeepSeek V4 cache tensor must have rank three.\n");
                return;
            }
            std::vector<int> dims = data.dims;
            dims[1] = sequenceLen;
#ifdef USE_CUDA
            if (data.multiDeviceData) {
                AssertInFastLLM(data.IsTensorParallelReplicated(),
                                "DeepSeek V4 cache truncation expects replicated CUDA data.\n");
                std::vector<int> devices;
                for (auto &it : data.multiDeviceDatas) {
                    AssertInFastLLM(it.second != nullptr &&
                                    it.second->dims.size() == 3,
                                    "DeepSeek V4 cache truncation is missing a CUDA replica.\n");
                    std::vector<int> localDims = it.second->dims;
                    localDims[1] = sequenceLen;
                    it.second->Resize(localDims);
                    devices.push_back(it.first);
                }
                PublishReplicatedCudaRootMetadata(data, devices, false);
                return;
            }
#endif
            data.Resize(dims);
        }

        static void TruncateDeepSeekV4DecodeCache(
                DeepSeekV4DecodeLayerCache &cache, int totalLen) {
            AssertInFastLLM(cache.initialized && totalLen >= 0 &&
                            totalLen <= cache.totalLen,
                            "DeepSeek V4 cannot grow a cache through truncation.\n");
            cache.totalLen = totalLen;
            if (cache.compressRatio <= 0) {
                return;
            }

            if (!cache.cudaGraphCacheReady) {
                const int rawLen = GetDataSeqLen(
                    cache.compressorKVRaw, cache.bsz,
                    cache.compressorWideDim);
                const int scoreLen = GetDataSeqLen(
                    cache.compressorScoreRaw, cache.bsz,
                    cache.compressorWideDim);
                if (rawLen > 0 || scoreLen > 0) {
                    AssertInFastLLM(
                        rawLen == scoreLen &&
                        cache.compressorRawTokenBase <= totalLen &&
                        cache.compressorRawTokenBase + rawLen >= totalLen,
                        "DeepSeek V4 speculative compressor cache does not "
                        "cover the committed prefix.\n");
                    const int committedRawLen =
                        totalLen - cache.compressorRawTokenBase;
                    ResizeTensorSequenceInPlace(
                        cache.compressorKVRaw, committedRawLen);
                    ResizeTensorSequenceInPlace(
                        cache.compressorScoreRaw, committedRawLen);
                }
            }

            const int committedBlocks = totalLen / cache.compressRatio;
            if (cache.compressedKV.dims.size() == 3) {
                AssertInFastLLM(cache.compressedKV.dims[1] >= committedBlocks,
                                "DeepSeek V4 speculative compressed cache is shorter than the committed prefix.\n");
                ResizeTensorSequenceInPlace(
                    cache.compressedKV, committedBlocks);
            } else {
                AssertInFastLLM(committedBlocks == 0,
                                "DeepSeek V4 committed prefix is missing compressed KV rows.\n");
            }
            cache.compressedBlocks = committedBlocks;
            cache.compressedTokenBase =
                committedBlocks * cache.compressRatio;
            if (cache.compressRatio == 4 &&
                cache.indexerCompressorWideDim > 0) {
                if (!cache.cudaGraphCacheReady) {
                    const int indexerRawLen = GetDataSeqLen(
                        cache.indexerCompressorKVRaw, cache.bsz,
                        cache.indexerCompressorWideDim);
                    const int indexerScoreLen = GetDataSeqLen(
                        cache.indexerCompressorScoreRaw, cache.bsz,
                        cache.indexerCompressorWideDim);
                    if (indexerRawLen > 0 || indexerScoreLen > 0) {
                        AssertInFastLLM(
                            indexerRawLen == indexerScoreLen &&
                            cache.indexerCompressorRawTokenBase <= totalLen &&
                            cache.indexerCompressorRawTokenBase +
                                indexerRawLen >= totalLen,
                            "DeepSeek V4 speculative indexer compressor cache "
                            "does not cover the committed prefix.\n");
                        const int committedRawLen =
                            totalLen - cache.indexerCompressorRawTokenBase;
                        ResizeTensorSequenceInPlace(
                            cache.indexerCompressorKVRaw, committedRawLen);
                        ResizeTensorSequenceInPlace(
                            cache.indexerCompressorScoreRaw, committedRawLen);
                    }
                }
                if (cache.indexerCompressedKV.dims.size() == 3) {
                    AssertInFastLLM(
                        cache.indexerCompressedKV.dims[1] >= committedBlocks,
                        "DeepSeek V4 speculative indexer compressed cache is "
                        "shorter than the committed prefix.\n");
                    ResizeTensorSequenceInPlace(
                        cache.indexerCompressedKV, committedBlocks);
                } else {
                    AssertInFastLLM(
                        committedBlocks == 0,
                        "DeepSeek V4 committed prefix is missing indexer "
                        "compressed rows.\n");
                }
                cache.indexerCompressedBlocks = committedBlocks;
            }
            cache.rawTailStartPos = std::min(
                cache.rawTailStartPos, totalLen);
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

        static void UpdateDebugPastKeyValues(std::vector<std::pair<Data*, Data*>> &pastKeyValues,
                                             int batchIndex, int bsz, int totalLen, int blocks) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4PastKVStub");
            int offset = batchIndex * blocks;
            if (blocks <= 0 || offset < 0 || offset >= (int)pastKeyValues.size()) {
                return;
            }
            int paddedLen = ((std::max(totalLen, 1) - 1) / 128 + 1) * 128;
            std::vector<float> zeros((uint64_t)bsz * totalLen, 0.0f);
            int limit = std::min(blocks, (int)pastKeyValues.size() - offset);
            for (int i = 0; i < limit; i++) {
                Data *keyPtr = pastKeyValues[offset + i].first;
                Data *valuePtr = pastKeyValues[offset + i].second;
                if (keyPtr == nullptr || valuePtr == nullptr) {
                    continue;
                }
                Data key(DataType::FLOAT32, {bsz, totalLen, 1}, zeros);
                Data value(DataType::FLOAT32, {bsz, totalLen, 1}, zeros);
                key.SetKVCache();
                value.SetKVCache();
                key.Expansion({bsz, paddedLen, 1});
                value.Expansion({bsz, paddedLen, 1});
                ResetData(*keyPtr);
                ResetData(*valuePtr);
                keyPtr->CopyFrom(key);
                valuePtr->CopyFrom(value);
                keyPtr->SetKVCache();
                valuePtr->SetKVCache();
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
            if (!DeepSeekV4PreferCuda() && input.dataDevice == DataDevice::CPU &&
                input.cpuData != nullptr && input.dataType == DataType::BFLOAT16 &&
                dtype == DataType::BFLOAT16 && !input.dims.empty()) {
                const int channels = input.dims.back();
                const int rows = (int)(input.Count(0) / channels);
                auto weightValues = ReadWeightFloatDataCached(weight);
                const uint16_t *source = (const uint16_t*)input.cpuData;
                if (&input != &output) {
                    ResetData(output);
                    output = Data(DataType::BFLOAT16, input.dims);
                    output.Allocate(false);
                }
                RunCpuDeepSeekV4RMSNormBFloat16(
                    source, weightValues->data(),
                    (uint16_t*)output.cpuData, rows, channels, eps);
                return;
            }
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
            int pairs = ropeDim / 2;
            std::vector<float> cosValues((uint64_t)seqlen * pairs);
            std::vector<float> sinValues((uint64_t)seqlen * pairs);
            for (int s = 0; s < seqlen; s++) {
                int pos = startPos + s * posStep;
                for (int p = 0; p < pairs; p++) {
                    float ang = pos * invFreq[p];
                    cosValues[(uint64_t)s * pairs + p] = std::cos(ang);
                    float sn = std::sin(ang);
                    sinValues[(uint64_t)s * pairs + p] =
                        inverse ? -sn : sn;
                }
            }
            for (int b = 0; b < bsz; b++) {
                for (int s = 0; s < seqlen; s++) {
                    for (int h = 0; h < heads; h++) {
                        uint64_t rowIndex = dims.size() == 4 ? (((uint64_t)b * seqlen + s) * heads + h)
                                                             : ((uint64_t)b * seqlen + s);
                        float *row = x.data() + rowIndex * dim + off;
                        for (int i = 0; i < ropeDim; i += 2) {
                            float c = cosValues[(uint64_t)s * pairs + i / 2];
                            float sn = sinValues[(uint64_t)s * pairs + i / 2];
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
                                        int startPos, int windowSize, Data &windowKV,
                                        const Data *decodeMeta = nullptr) {
            (void)bsz;
            (void)headDim;
#ifdef USE_CUDA
            EnsureTensorOnSameCudaDevice(windowKV, kv);
            if (kv.multiDeviceData && kv.IsTensorParallelReplicated()) {
                std::vector<int> devices = GetReplicatedCudaDevices(kv);
                if (!windowKV.multiDeviceData ||
                    !windowKV.IsTensorParallelReplicated()) {
                    PrepareMultiCudaReplicatedData(windowKV, devices, true);
                }
                std::vector<char> ok(devices.size(), 0);
                std::function<void(int, int)> task = [&](int rank, int device) {
                    const Data *localKV = GetTensorCudaReplica(kv, device);
                    Data *localWindow = GetTensorCudaReplica(windowKV, device);
                    const Data *localMeta = decodeMeta == nullptr ? nullptr :
                        GetTensorCudaReplica(*decodeMeta, device);
                    if (localKV != nullptr && localWindow != nullptr) {
                        if (localMeta != nullptr) {
                            ok[rank] = FastllmCudaDeepSeekV4UpdateWindowKVCacheGraph(
                                *localKV, (const int32_t*)localMeta->cudaData,
                                windowSize, *localWindow);
                        } else if (decodeMeta == nullptr) {
                            ok[rank] = FastllmCudaDeepSeekV4UpdateWindowKVCache(
                                *localKV, startPos, windowSize, *localWindow);
                        }
                    }
                };
                if (!devices.empty() &&
                    MultiCudaRunDeviceCallbacks(devices, task) &&
                    std::all_of(ok.begin(), ok.end(),
                                [](char state) { return state != 0; })) {
                    return;
                }
                FastllmCudaSetThreadError();
                ErrorInFastLLM(
                    "DeepSeekV4UpdateWindowKVCache: replicated CUDA update failed.\n");
            }
            if (decodeMeta != nullptr && decodeMeta->dataDevice == DataDevice::CUDA &&
                decodeMeta->cudaData != nullptr &&
                FastllmCudaDeepSeekV4UpdateWindowKVCacheGraph(
                    kv, (const int32_t*)decodeMeta->cudaData, windowSize, windowKV)) {
                return;
            }
#endif
            Executor &executor = *((Executor*)GetExecutor());
            executor.Run("DeepSeekV4UpdateWindowKVCache", {
                {"input", (Data*)&kv}, {"cache", &windowKV}
            }, {}, {{"startPos", startPos}, {"windowSize", windowSize}});
        }

        static bool CanAppendFullWindowKVCache(const Data &kv,
                                               int appendTokens,
                                               int windowSize,
                                               const Data &windowKV) {
#ifdef USE_CUDA
            if (!DeepSeekV4PreferCuda() || appendTokens <= 0 ||
                kv.dims.size() != 3 || windowKV.dims.size() != 3 ||
                kv.dims[0] != windowKV.dims[0] ||
                kv.dims[2] != windowKV.dims[2] ||
                kv.dims[1] < appendTokens ||
                windowKV.dims[1] != windowSize ||
                kv.dataType != windowKV.dataType) {
                return false;
            }

            if (kv.multiDeviceData && windowKV.multiDeviceData &&
                kv.IsTensorParallelReplicated() &&
                windowKV.IsTensorParallelReplicated()) {
                std::vector<int> devices = GetReplicatedCudaDevices(windowKV);
                return !devices.empty() &&
                    devices == GetReplicatedCudaDevices(kv);
            }
            const int device = GetTensorCudaDevice(windowKV);
            return device >= 0 &&
                GetTensorCudaReplica(kv, device) != nullptr &&
                GetTensorCudaReplica(windowKV, device) != nullptr;
#else
            (void)kv;
            (void)appendTokens;
            (void)windowSize;
            (void)windowKV;
            return false;
#endif
        }

        static bool AppendFullWindowKVCache(const Data &kv, int appendTokens,
                                            int windowSize, Data &windowKV) {
#ifdef USE_CUDA
            if (!CanAppendFullWindowKVCache(
                    kv, appendTokens, windowSize, windowKV)) {
                return false;
            }
            if (kv.multiDeviceData && windowKV.multiDeviceData &&
                kv.IsTensorParallelReplicated() &&
                windowKV.IsTensorParallelReplicated()) {
                std::vector<int> devices = GetReplicatedCudaDevices(windowKV);
                std::vector<char> ok(devices.size(), 0);
                std::function<void(int, int)> task = [&](int rank, int device) {
                    const Data *localKV = GetTensorCudaReplica(kv, device);
                    Data *localWindow = GetTensorCudaReplica(windowKV, device);
                    if (localKV != nullptr && localWindow != nullptr) {
                        ok[rank] =
                            FastllmCudaDeepSeekV4AppendFullWindowKVCache(
                                *localKV, appendTokens, *localWindow);
                    }
                };
                return MultiCudaRunDeviceCallbacks(devices, task) &&
                    std::all_of(ok.begin(), ok.end(),
                        [](char state) { return state != 0; });
            }

            const int device = GetTensorCudaDevice(windowKV);
            const Data *localKV = GetTensorCudaReplica(kv, device);
            Data *localWindow = GetTensorCudaReplica(windowKV, device);
            const int originalDevice = FastllmCudaGetDevice();
            FastllmCudaSetDevice(device);
            const bool ok = FastllmCudaDeepSeekV4AppendFullWindowKVCache(
                *localKV, appendTokens, *localWindow);
            FastllmCudaSetDevice(originalDevice);
            return ok;
#else
            (void)kv;
            (void)appendTokens;
            (void)windowSize;
            (void)windowKV;
            return false;
#endif
        }

        static bool AppendFullWindowKVCacheBatch(
                const std::vector<Data*> &kvs, int appendTokens,
                int windowSize, std::vector<Data> &windowKVs) {
#ifdef USE_CUDA
            if (kvs.empty() || kvs.size() != windowKVs.size()) {
                return false;
            }
            for (int stage = 0; stage < (int)kvs.size(); ++stage) {
                if (kvs[stage] == nullptr ||
                    !CanAppendFullWindowKVCache(
                        *kvs[stage], appendTokens, windowSize,
                        windowKVs[stage])) {
                    return false;
                }
            }

            bool replicated = true;
            std::vector<int> devices;
            for (int stage = 0; stage < (int)kvs.size(); ++stage) {
                const Data &kv = *kvs[stage];
                Data &window = windowKVs[stage];
                if (!kv.multiDeviceData || !window.multiDeviceData ||
                    !kv.IsTensorParallelReplicated() ||
                    !window.IsTensorParallelReplicated()) {
                    replicated = false;
                    break;
                }
                const std::vector<int> stageDevices =
                    GetReplicatedCudaDevices(window);
                if (stage == 0) {
                    devices = stageDevices;
                } else if (stageDevices != devices ||
                           GetReplicatedCudaDevices(kv) != devices) {
                    replicated = false;
                    break;
                }
            }
            if (replicated && !devices.empty()) {
                std::vector<char> ok(devices.size(), 0);
                std::function<void(int, int)> task =
                    [&](int rank, int device) {
                        bool localOk = true;
                        for (int stage = 0;
                             stage < (int)kvs.size() && localOk; ++stage) {
                            const Data *localKV =
                                GetTensorCudaReplica(*kvs[stage], device);
                            Data *localWindow = GetTensorCudaReplica(
                                windowKVs[stage], device);
                            localOk = localKV != nullptr &&
                                localWindow != nullptr &&
                                FastllmCudaDeepSeekV4AppendFullWindowKVCache(
                                    *localKV, appendTokens, *localWindow);
                        }
                        ok[rank] = localOk;
                    };
                return MultiCudaRunDeviceCallbacks(devices, task) &&
                    std::all_of(ok.begin(), ok.end(),
                        [](char state) { return state != 0; });
            }

            for (int stage = 0; stage < (int)kvs.size(); ++stage) {
                if (!AppendFullWindowKVCache(
                        *kvs[stage], appendTokens, windowSize,
                        windowKVs[stage])) {
                    return false;
                }
            }
            return true;
#else
            (void)kvs;
            (void)appendTokens;
            (void)windowSize;
            (void)windowKVs;
            return false;
#endif
        }

        static int BuildWindowKVPrefixData(const Data &windowKV, int bsz, int headDim,
                                           int startPos, int windowSize, Data &output) {
            int prefixLen = std::min(windowSize, startPos);
            if (prefixLen <= 0 || !HasTensorData(windowKV)) {
                return 0;
            }
            ScopedExecutorProfiler executorProfile("DeepSeekV4KVCache");
#ifdef USE_CUDA
            if (DeepSeekV4PreferCuda() && windowKV.multiDeviceData &&
                windowKV.IsTensorParallelReplicated()) {
                std::vector<int> devices = GetReplicatedCudaDevices(windowKV);
                ResetData(output);
                output.dataType = DataType::FLOAT32;
                output.UpdateUnitSize();
                output.Resize({bsz, prefixLen, headDim});
                output.dataDevice = DataDevice::CUDA;
                output.dataDeviceIds = devices;
                PrepareMultiCudaReplicatedData(output, devices, false);

                std::vector<char> ok(devices.size(), 0);
                std::function<void(int, int)> task = [&](int rank, int device) {
                    const Data *localWindow =
                        GetTensorCudaReplica(windowKV, device);
                    auto outputIt = output.multiDeviceDatas.find(device);
                    Data *localOutput = outputIt == output.multiDeviceDatas.end() ?
                        nullptr : outputIt->second;
                    if (localWindow != nullptr && localOutput != nullptr) {
                        ok[rank] = FastllmCudaDeepSeekV4BuildWindowKVPrefix(
                            *localWindow, startPos, windowSize, prefixLen,
                            *localOutput);
                    }
                };
                if (!devices.empty() &&
                    MultiCudaRunDeviceCallbacks(devices, task) &&
                    std::all_of(ok.begin(), ok.end(),
                                [](char state) { return state != 0; })) {
                    Data *first = output.multiDeviceDatas.at(devices.front());
                    output.Resize(first->dims);
                    output.strides = first->strides;
                    output.expansionDims = first->expansionDims;
                    output.expansionSize = first->expansionSize;
                    output.expansionBytes = first->expansionBytes;
                    output.tpGlobalDims = output.dims;
                    return prefixLen;
                }
                FastllmCudaSetThreadError();
                ErrorInFastLLM(
                    "DeepSeekV4BuildWindowKVPrefix: replicated CUDA build failed.\n");
            }
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
            Data &wkv = weight[prefix + ".wkv.weight"];
            Data &wgate = weight[prefix + ".wgate.weight"];
            // The official Compressor explicitly promotes both the activation
            // and its checkpoint-BF16 weights to FP32 before these projections.
            Data xFloat;
            ToDataType(x, xFloat, DataType::FLOAT32);
            Linear(xFloat, wkv, Data(), kv, true);
            Linear(xFloat, wgate, Data(), score, true);
        }

        static void AppendCompressorRaw(const Data &kv, const Data &score,
                                        int bsz, int seqlen, int wideDim,
                                        Data &allKV, Data &allScore) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4CompressorAppend");
            if (seqlen <= 0 || wideDim <= 0 || kv.dims.size() != 3 || score.dims != kv.dims) {
                return;
            }
            if (!HasTensorData(allKV)) {
                CopyTensorData(allKV, kv);
                CopyTensorData(allScore, score);
                EnsureCompressorRawCapacity(allKV, kv.dims[1]);
                EnsureCompressorRawCapacity(allScore, score.dims[1]);
                return;
            }
            int oldLen = GetDataSeqLen(allKV, bsz, wideDim);
            if (oldLen <= 0 || GetDataSeqLen(allScore, bsz, wideDim) != oldLen) {
                CopyTensorData(allKV, kv);
                CopyTensorData(allScore, score);
                EnsureCompressorRawCapacity(allKV, kv.dims[1]);
                EnsureCompressorRawCapacity(allScore, score.dims[1]);
                return;
            }
#ifdef USE_CUDA
            // History snapshots intentionally retain a single CPU copy of the
            // compressor tail.  A TP cache hit resumes with replicated CUDA
            // projections, so restore the cached destination on every rank
            // before CatDirect.  Otherwise the first compressed-attention
            // layer attempts to append a CUDA replica to CPU storage.
            if (!allKV.multiDeviceData &&
                allKV.dataDevice == DataDevice::CPU) {
                EnsureTensorOnSameCudaDevice(allKV, kv);
            }
            if (!allScore.multiDeviceData &&
                allScore.dataDevice == DataDevice::CPU) {
                EnsureTensorOnSameCudaDevice(allScore, score);
            }
#endif
            EnsureCompressorRawCapacity(allKV, oldLen + kv.dims[1]);
            EnsureCompressorRawCapacity(allScore, oldLen + score.dims[1]);
            CatDirectTensor(allKV, kv, 1);
            CatDirectTensor(allScore, score, 1);
        }

        static int GetCompressorRawLen(const Data &raw, int bsz, int wideDim) {
            return GetDataSeqLen(raw, bsz, wideDim);
        }

#ifdef USE_CUDA
        static bool CompactCompressorRawInPlaceCuda(Data &kv, Data &score,
                                                     int dropLen) {
            if (kv.multiDeviceData || score.multiDeviceData) {
                // Keep replicated caches on the established Split path until
                // this optimization has a multi-device atomic-commit path.
                return false;
            }
            if (kv.dataDevice != DataDevice::CUDA ||
                score.dataDevice != DataDevice::CUDA ||
                kv.cudaData == nullptr || score.cudaData == nullptr) {
                return false;
            }
            int device = GetPointerDeviceId(kv.cudaData);
            if (device < 0 || GetPointerDeviceId(score.cudaData) != device) {
                return false;
            }
            int originalDevice = FastllmCudaGetDevice();
            FastllmCudaSetDevice(device);
            bool ok = FastllmCudaDeepSeekV4CompactCompressorRaw(
                kv, score, dropLen);
            FastllmCudaSetDevice(originalDevice);
            return ok;
        }
#endif

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
#ifdef USE_CUDA
            // The ratio-4 overlap tail is disjoint from the dropped prefix
            // (normally rows [4, 8) -> [0, 4)).  Keep the 128-row reserve and
            // copy both FP32 projections in one launch instead of allocating,
            // splitting and replacing two tensors every fourth token.
            if (dropLen >= newLen &&
                CompactCompressorRawInPlaceCuda(allKV, allScore, dropLen)) {
                rawTokenBase = retainStart;
                return;
            }
#endif
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
            // The main MLA cache quantizes only its NoPE prefix in 64-value
            // groups.  The C4 learned indexer instead stores one 128-value
            // FP8 block after applying RoPE to the final 64 values.  Keep the
            // common compressor pipeline, but select the cache ABI from the
            // head width so the low-SM/reference path matches vLLM as well.
            const int quantDim = headDim == 128 ? headDim : headDim - ropeDim;
            const int quantBlock = headDim == 128 ? 128 : 64;
#ifdef USE_CUDA
            if (normed.dataDevice == DataDevice::CUDA &&
                FastllmCudaDeepSeekV4RotaryQuant(normed, ropeDim, ropeBase, blockStart * compressRatio,
                                                 originalSeqLen, ropeFactor, betaFast, betaSlow,
                                                 quantDim, quantBlock, compressRatio)) {
                CopyTensorData(output, normed);
                return;
            }
#endif
            auto out = ReadFloatData(normed);
            ApplyRotaryReference(out, normed.dims, ropeDim, ropeBase, blockStart * compressRatio, false,
                                 originalSeqLen, ropeFactor, betaFast, betaSlow, compressRatio);
            ActQuantInplaceReference(out, normed.dims, quantDim, quantBlock);
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
            return data.dims.size() >= 2 && HasTensorData(data);
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
                                             int originalSeqLen, Data &output,
                                             bool preferCudaOutput = false,
                                             bool indexer = false) {
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
                                               preferCudaOutput, output, indexer);
            return true;
        }

#ifdef USE_CUDA
        struct DeepSeekV4CudaWorker {
            int device;
            std::thread thread;
            std::mutex mutex;
            std::condition_variable cv;
            std::atomic<uint64_t> publishId{0};
            std::atomic<uint64_t> doneId{0};
            std::atomic<bool> stop{false};
            std::function<void(int, int)> *task = nullptr;
            int rank = 0;
            std::exception_ptr error;

            explicit DeepSeekV4CudaWorker(int device) : device(device) {
                thread = std::thread([this]() { Loop(); });
            }

            void Loop() {
                FastllmCudaSetDevice(device);
                uint64_t lastId = 0;
                constexpr int spinLimit = 100000;
                int spins = 0;
                while (!stop.load(std::memory_order_acquire)) {
                    uint64_t currentId = publishId.load(std::memory_order_acquire);
                    if (currentId != lastId) {
                        lastId = currentId;
                        error = nullptr;
                        try {
                            (*task)(rank, device);
                            // CUDA is built with per-thread default streams.  A persistent
                            // producer thread must explicitly complete its stream before
                            // a MultiCUDA worker on another host thread consumes the tensor.
                            FastllmCudaSyncCurrentThreadStream();
                        } catch (...) {
                            error = std::current_exception();
                        }
                        doneId.store(currentId, std::memory_order_release);
                        spins = 0;
                        continue;
                    }
                    if (spins++ < spinLimit) {
                        if ((spins & 8191) == 0) {
                            std::this_thread::yield();
                        }
                        continue;
                    }
                    std::unique_lock<std::mutex> lock(mutex);
                    cv.wait(lock, [this, lastId]() {
                        return stop.load(std::memory_order_acquire) ||
                               publishId.load(std::memory_order_acquire) != lastId;
                    });
                    spins = 0;
                }
            }

            uint64_t Submit(std::function<void(int, int)> *nextTask, int nextRank) {
                while (doneId.load(std::memory_order_acquire) !=
                       publishId.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                uint64_t nextId;
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    task = nextTask;
                    rank = nextRank;
                    nextId = publishId.load(std::memory_order_relaxed) + 1;
                    publishId.store(nextId, std::memory_order_release);
                }
                cv.notify_one();
                return nextId;
            }

            void Wait(uint64_t targetId) {
                while (doneId.load(std::memory_order_acquire) < targetId) {
                    std::this_thread::yield();
                }
            }
        };

        struct DeepSeekV4CudaWorkerPool {
            std::mutex mutex;
            std::mutex dispatchMutex;
            std::map<int, std::unique_ptr<DeepSeekV4CudaWorker> > workers;

            DeepSeekV4CudaWorker *Get(int device) {
                std::lock_guard<std::mutex> lock(mutex);
                auto it = workers.find(device);
                if (it != workers.end()) {
                    return it->second.get();
                }
                auto worker = std::make_unique<DeepSeekV4CudaWorker>(device);
                DeepSeekV4CudaWorker *result = worker.get();
                workers[device] = std::move(worker);
                return result;
            }
        };

        static DeepSeekV4CudaWorkerPool &GetDeepSeekV4CudaWorkerPool() {
            // CUDA may already be shutting down when function statics are destroyed.
            // Keep these process-lifetime workers alive, matching the MultiCUDA pool.
            static auto *pool = new DeepSeekV4CudaWorkerPool();
            return *pool;
        }

        template <typename F>
        static void RunDeepSeekV4MultiCuda(const std::vector<int> &devices, F &&fn) {
            std::function<void(int, int)> task(std::forward<F>(fn));
            if (MultiCudaRunDeviceCallbacks(devices, task)) {
                return;
            }
            auto &pool = GetDeepSeekV4CudaWorkerPool();
            std::lock_guard<std::mutex> dispatchLock(pool.dispatchMutex);
            std::vector<DeepSeekV4CudaWorker*> workers;
            std::vector<uint64_t> targetIds;
            workers.reserve(devices.size());
            targetIds.reserve(devices.size());
            for (int i = 0; i < (int)devices.size(); i++) {
                auto *worker = pool.Get(devices[i]);
                workers.push_back(worker);
                targetIds.push_back(worker->Submit(&task, i));
            }
            std::exception_ptr firstError;
            for (int i = 0; i < (int)devices.size(); i++) {
                workers[i]->Wait(targetIds[i]);
                if (firstError == nullptr && workers[i]->error != nullptr) {
                    firstError = workers[i]->error;
                }
            }
            if (firstError != nullptr) {
                std::rethrow_exception(firstError);
            }
        }

        static bool DeepSeekV4ScaleQRotaryGraphMultiCuda(
                Data &q, int ropeDim, float ropeBase, const Data &decodeMeta,
                int originalSeqLen, float ropeFactor, int betaFast,
                int betaSlow, float eps) {
            if (!q.multiDeviceData) {
                return decodeMeta.cudaData != nullptr &&
                    FastllmCudaDeepSeekV4ScaleQRotaryGraph(
                        q, ropeDim, ropeBase,
                        (const int32_t*)decodeMeta.cudaData, originalSeqLen,
                        ropeFactor, betaFast, betaSlow, eps);
            }
            std::vector<int> devices = GetTensorCudaDevices(q);
            std::vector<char> ok(devices.size(), 0);
            RunDeepSeekV4MultiCuda(devices, [&](int rank, int device) {
                Data *localQ = GetTensorCudaReplica(q, device);
                const Data *localMeta = GetTensorCudaReplica(decodeMeta, device);
                if (localQ != nullptr && localMeta != nullptr) {
                    ok[rank] = FastllmCudaDeepSeekV4ScaleQRotaryGraph(
                        *localQ, ropeDim, ropeBase,
                        (const int32_t*)localMeta->cudaData, originalSeqLen,
                        ropeFactor, betaFast, betaSlow, eps);
                }
            });
            return !devices.empty() &&
                   std::all_of(ok.begin(), ok.end(),
                               [](char state) { return state != 0; });
        }

        static bool DeepSeekV4RotaryQuantGraphMultiCuda(
                Data &x, int ropeDim, float ropeBase, const Data &decodeMeta,
                int originalSeqLen, float ropeFactor, int betaFast,
                int betaSlow, int quantDim, int blockSize, int posStep) {
            if (!x.multiDeviceData) {
                return decodeMeta.cudaData != nullptr &&
                    FastllmCudaDeepSeekV4RotaryQuantGraph(
                        x, ropeDim, ropeBase,
                        (const int32_t*)decodeMeta.cudaData, originalSeqLen,
                        ropeFactor, betaFast, betaSlow, quantDim, blockSize,
                        posStep);
            }
            std::vector<int> devices = GetTensorCudaDevices(x);
            std::vector<char> ok(devices.size(), 0);
            RunDeepSeekV4MultiCuda(devices, [&](int rank, int device) {
                Data *localX = GetTensorCudaReplica(x, device);
                const Data *localMeta = GetTensorCudaReplica(decodeMeta, device);
                if (localX != nullptr && localMeta != nullptr) {
                    ok[rank] = FastllmCudaDeepSeekV4RotaryQuantGraph(
                        *localX, ropeDim, ropeBase,
                        (const int32_t*)localMeta->cudaData, originalSeqLen,
                        ropeFactor, betaFast, betaSlow, quantDim, blockSize,
                        posStep);
                }
            });
            return !devices.empty() &&
                   std::all_of(ok.begin(), ok.end(),
                               [](char state) { return state != 0; });
        }

        static bool DeepSeekV4UpdateCompressedKVGraphMultiCuda(
                const Data &kv, const Data &score, const Data &ape,
                const Data &normWeight, const Data &decodeMeta,
                int compressRatio, int headDim, int ropeDim, float ropeBase,
                int originalSeqLen, float ropeFactor, int betaFast,
                int betaSlow, Data &kvRing, Data &scoreRing,
                Data &compressedKV) {
            if (!kv.multiDeviceData) {
                return decodeMeta.cudaData != nullptr &&
                    FastllmCudaDeepSeekV4UpdateCompressedKVGraph(
                        kv, score, ape, normWeight,
                        (const int32_t*)decodeMeta.cudaData, compressRatio,
                        headDim, ropeDim, ropeBase, originalSeqLen, ropeFactor,
                        betaFast, betaSlow, kvRing, scoreRing, compressedKV);
            }
            std::vector<int> devices = GetTensorCudaDevices(kv);
            std::vector<char> ok(devices.size(), 0);
            RunDeepSeekV4MultiCuda(devices, [&](int rank, int device) {
                const Data *localKV = GetTensorCudaReplica(kv, device);
                const Data *localScore = GetTensorCudaReplica(score, device);
                const Data *localApe = GetTensorCudaReplica(ape, device);
                const Data *localNorm = GetTensorCudaReplica(normWeight, device);
                const Data *localMeta = GetTensorCudaReplica(decodeMeta, device);
                Data *localKVRing = GetTensorCudaReplica(kvRing, device);
                Data *localScoreRing = GetTensorCudaReplica(scoreRing, device);
                Data *localCompressed = GetTensorCudaReplica(compressedKV, device);
                if (localKV != nullptr && localScore != nullptr &&
                    localApe != nullptr && localNorm != nullptr &&
                    localMeta != nullptr && localKVRing != nullptr &&
                    localScoreRing != nullptr && localCompressed != nullptr) {
                    ok[rank] = FastllmCudaDeepSeekV4UpdateCompressedKVGraph(
                        *localKV, *localScore, *localApe, *localNorm,
                        (const int32_t*)localMeta->cudaData, compressRatio,
                        headDim, ropeDim, ropeBase, originalSeqLen, ropeFactor,
                        betaFast, betaSlow, *localKVRing, *localScoreRing,
                        *localCompressed);
                }
            });
            return !devices.empty() &&
                   std::all_of(ok.begin(), ok.end(),
                               [](char state) { return state != 0; });
        }

        static bool DeepSeekV4BuildIndexerTopKGraphMultiCuda(
                Data &q, Data &weights, Data &compressedKV,
                Data &decodeMeta, int compressRatio, float ropeBase,
                int originalSeqLen, float ropeFactor, int betaFast,
                int betaSlow, Data &indices, Data &lengths,
                const std::vector<int> &preferredDevices) {
            if (!q.multiDeviceData) {
                return decodeMeta.cudaData != nullptr &&
                    FastllmCudaDeepSeekV4BuildIndexerTopKGraph(
                        q, weights, compressedKV,
                        (const int32_t *)decodeMeta.cudaData, compressRatio,
                        ropeBase, originalSeqLen, ropeFactor, betaFast,
                        betaSlow, indices, lengths);
            }
            std::vector<int> devices = preferredDevices.empty() ?
                GetTensorCudaDevices(q) : preferredDevices;
            if (devices.empty() || !q.IsTensorParallelReplicated()) {
                return false;
            }
            PrepareMultiCudaReplicatedData(q, devices, true);
            PrepareMultiCudaReplicatedData(weights, devices, true);
            PrepareMultiCudaReplicatedData(compressedKV, devices, true);
            PrepareMultiCudaReplicatedData(decodeMeta, devices, true);
            if (!indices.multiDeviceData ||
                !indices.IsTensorParallelReplicated() ||
                !lengths.multiDeviceData ||
                !lengths.IsTensorParallelReplicated()) {
                return false;
            }
            std::vector<char> ok(devices.size(), 0);
            RunDeepSeekV4MultiCuda(devices, [&](int rank, int device) {
                Data *localQ = GetTensorCudaReplica(q, device);
                Data *localWeights = GetTensorCudaReplica(weights, device);
                Data *localCompressed =
                    GetTensorCudaReplica(compressedKV, device);
                Data *localMeta = GetTensorCudaReplica(decodeMeta, device);
                Data *localIndices = GetTensorCudaReplica(indices, device);
                Data *localLengths = GetTensorCudaReplica(lengths, device);
                if (localQ != nullptr && localWeights != nullptr &&
                    localCompressed != nullptr && localMeta != nullptr &&
                    localIndices != nullptr && localLengths != nullptr) {
                    ok[rank] = FastllmCudaDeepSeekV4BuildIndexerTopKGraph(
                        *localQ, *localWeights, *localCompressed,
                        (const int32_t *)localMeta->cudaData, compressRatio,
                        ropeBase, originalSeqLen, ropeFactor, betaFast,
                        betaSlow, *localIndices, *localLengths);
                }
            });
            return std::all_of(ok.begin(), ok.end(),
                               [](char state) { return state != 0; });
        }

        static bool PrepareDeepSeekV4AttentionTp(Data &q, Data &kv, Data &attnSink,
                                                 Data &output, std::vector<int> &devices) {
            if (!q.multiDeviceData || !q.IsTensorParallelSharded() ||
                q.dims.size() != 4 || q.tpAxis != 2) {
                return false;
            }
            std::map<int, int> ratios;
            FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
            if (devices.size() <= 1) {
                return false;
            }
            PrepareMultiCudaReplicatedData(kv, devices, true);
            DivisionScheme headScheme = q.tpRanges;
            if (!attnSink.multiDeviceData) {
                if (!SplitMultiCudaWeight1D(attnSink, devices, headScheme)) {
                    return false;
                }
            }
            output.dataType = DataType::BFLOAT16;
            PrepareMultiCudaShardedData(output, devices, q.dims, 2, q.tpRanges);
            return true;
        }
#endif

        static void SparseAttentionReference(Data &q, Data &kv, Data &attnSink, int windowSize,
                                             int ropeDim, float ropeBase, int startPos, float softmaxScale,
                                             Data &output, int compressRatio = 0, int originalSeqLen = 0,
                                             float ropeFactor = 1.0f, int betaFast = 32, int betaSlow = 1,
                                             int prefixLen = 0,
                                             bool nonCausalBlock = false,
                                             const Data *decodeMeta = nullptr,
                                             const Data *compressedTopK = nullptr) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4SparseAttention");
#ifdef USE_CUDA
            std::vector<int> tpDevices;
            if (compressedTopK == nullptr &&
                PrepareDeepSeekV4AttentionTp(q, kv, attnSink, output, tpDevices)) {
                std::vector<char> ok(tpDevices.size(), 0);
                RunDeepSeekV4MultiCuda(tpDevices, [&](int rank, int device) {
                    const Data *localMeta = decodeMeta == nullptr ? nullptr :
                        GetTensorCudaReplica(*decodeMeta, device);
                    ok[rank] = FastllmCudaDeepSeekV4SparseAttentionPrefill(
                        *q.multiDeviceDatas.at(device), *kv.multiDeviceDatas.at(device),
                        *attnSink.multiDeviceDatas.at(device), windowSize, startPos, compressRatio,
                        ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow,
                        softmaxScale, *output.multiDeviceDatas.at(device), prefixLen,
                        nonCausalBlock,
                        localMeta == nullptr ? nullptr :
                            (const int32_t *)localMeta->cudaData);
                });
                for (char state : ok) {
                    AssertInFastLLM(state != 0,
                                    "DeepSeek V4 MultiCuda sparse prefill rejected a local shard.\n");
                }
                return;
            }
            if (compressedTopK == nullptr &&
                !EnvFlagEnabled("FASTLLM_DSV4_DISABLE_CUDA_SPARSE_PREFILL") &&
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
                        softmaxScale, output, prefixLen, nonCausalBlock,
                        decodeMeta == nullptr ? nullptr :
                            (const int32_t *)decodeMeta->cudaData)) {
                    return;
                }
            }
#endif
            if (compressedTopK != nullptr) {
                DeepSeekV4SparseAttention(
                    q, kv, attnSink, windowSize, ropeDim, ropeBase,
                    startPos, softmaxScale, output, compressRatio,
                    originalSeqLen, ropeFactor, betaFast, betaSlow,
                    prefixLen, compressedTopK);
                return;
            }
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
                    int liveWindow = nonCausalBlock ?
                        realPrefixLen + seqlen :
                        std::min(windowSize, realPrefixLen + s + 1);
                    std::vector<int> idxs(liveWindow, -1);
                    int beginPos = startPos + s - liveWindow + 1;
                    for (int k = 0; k < liveWindow; k++) {
                        if (nonCausalBlock) {
                            idxs[k] = k;
                        } else {
                            int pos = beginPos + k;
                            idxs[k] = (pos < startPos) ?
                                (pos - prefixStartPos) :
                                (realPrefixLen + pos - startPos);
                        }
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

        struct SparseAttentionDecodeCachedHeadsOp : MultiThreadBaseOp {
            const float *q;
            const float *window;
            const float *compressed;
            const float *sink;
            const int *idxs;
            float *output;
            int flatHeadSt, flatHeadEnd;
            int heads, dim, windowSize, compressedCount, idxCount;
            float softmaxScale;

            SparseAttentionDecodeCachedHeadsOp(
                const float *q, const float *window, const float *compressed,
                const float *sink, const int *idxs, float *output,
                int flatHeadSt, int flatHeadEnd, int heads, int dim,
                int windowSize, int compressedCount, int idxCount,
                float softmaxScale
            ) : q(q), window(window), compressed(compressed), sink(sink),
                idxs(idxs), output(output), flatHeadSt(flatHeadSt),
                flatHeadEnd(flatHeadEnd), heads(heads), dim(dim),
                windowSize(windowSize), compressedCount(compressedCount),
                idxCount(idxCount), softmaxScale(softmaxScale) {}

            const float *GetKVRow(int batch, int idx) const {
                if (idx < windowSize) {
                    return window +
                        ((uint64_t)batch * windowSize + idx) * dim;
                }
                return compressed +
                    ((uint64_t)batch * compressedCount +
                     (idx - windowSize)) * dim;
            }

            void Run() override {
                std::vector<float> scores(idxCount);
                for (int flatHead = flatHeadSt;
                     flatHead < flatHeadEnd; flatHead++) {
                    int batch = flatHead / heads;
                    int head = flatHead % heads;
                    const float *qrow =
                        q + (uint64_t)flatHead * dim;
                    float mx = -std::numeric_limits<float>::infinity();
                    for (int k = 0; k < idxCount; k++) {
                        const float *kvrow = GetKVRow(batch, idxs[k]);
                        double dot = 0.0;
                        for (int d = 0; d < dim; d++) {
                            dot += (double)qrow[d] * kvrow[d];
                        }
                        scores[k] = (float)dot * softmaxScale;
                        mx = std::max(mx, scores[k]);
                    }
                    float safeMx = std::isfinite(mx) ? mx : 0.0f;
                    double denom =
                        std::exp((double)sink[head] - safeMx);
                    for (float score : scores) {
                        denom += std::exp((double)score - safeMx);
                    }
                    float *orow =
                        output + (uint64_t)flatHead * dim;
                    for (int k = 0; k < idxCount; k++) {
                        float w = (float)(
                            std::exp((double)scores[k] - safeMx) /
                            std::max(denom, 1e-30));
                        const float *kvrow =
                            GetKVRow(batch, idxs[k]);
                        for (int d = 0; d < dim; d++) {
                            orow[d] += w * kvrow[d];
                        }
                    }
                }
            }
        };

        static void SparseAttentionDecodeCachedReference(Data &q,
                                                         const Data &windowKV,
                                                         const Data &compressedKV,
                                                         Data &attnSink,
                                                         int windowSize, int startPos, int compressedCount,
                                                         int ropeDim, float ropeBase, float softmaxScale,
                                                         Data &output, int originalSeqLen = 0,
                                                         float ropeFactor = 1.0f, int betaFast = 32,
                                                         int betaSlow = 1,
                                                         const Data *decodeMeta = nullptr,
                                                         int compressRatio = 0,
                                                         Data *packedWindowKV = nullptr,
                                                         Data *packedCompressedKV = nullptr,
                                                         Data *sm120Scratch = nullptr,
                                                         const Data *compressedIndices = nullptr,
                                                         const Data *compressedLengths = nullptr,
                                                         const Data *compressedTopK = nullptr) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4SparseDecodeCached");
#ifdef USE_CUDA
            Data &windowMutable = (Data&)windowKV;
            Data &compressedMutable = (Data&)compressedKV;
            std::vector<int> tpDevices;
            if (compressedTopK == nullptr &&
                PrepareDeepSeekV4AttentionTp(q, windowMutable, attnSink, output, tpDevices)) {
                PrepareMultiCudaReplicatedData(compressedMutable, tpDevices, true);
                if (compressedIndices != nullptr) {
                    PrepareMultiCudaReplicatedData(
                        *(Data *)compressedIndices, tpDevices, true);
                }
                if (compressedLengths != nullptr) {
                    PrepareMultiCudaReplicatedData(
                        *(Data *)compressedLengths, tpDevices, true);
                }
                std::vector<char> ok(tpDevices.size(), 0);
                RunDeepSeekV4MultiCuda(tpDevices, [&](int rank, int device) {
                    Data *localQ = q.multiDeviceDatas.at(device);
                    Data *localWindow = windowMutable.multiDeviceDatas.at(device);
                    Data *localCompressed = compressedMutable.multiDeviceDatas.at(device);
                    Data *localSink = attnSink.multiDeviceDatas.at(device);
                    Data *localOutput = output.multiDeviceDatas.at(device);
                    const Data *localMeta = decodeMeta == nullptr ? nullptr :
                        GetTensorCudaReplica(*decodeMeta, device);
                    const Data *localIndices = compressedIndices == nullptr ?
                        nullptr : GetTensorCudaReplica(*compressedIndices, device);
                    const Data *localLengths = compressedLengths == nullptr ?
                        nullptr : GetTensorCudaReplica(*compressedLengths, device);
                    if (localMeta != nullptr) {
                        Data *localPackedWindow = packedWindowKV == nullptr ?
                            nullptr : GetTensorCudaReplica(*packedWindowKV, device);
                        Data *localPackedCompressed =
                            packedCompressedKV == nullptr ? nullptr :
                            GetTensorCudaReplica(*packedCompressedKV, device);
                        Data *localScratch = sm120Scratch == nullptr ?
                            nullptr : GetTensorCudaReplica(
                                *sm120Scratch, device);
                        if (DeepSeekV4SparseMlaSm120Enabled() &&
                            localPackedWindow != nullptr &&
                            localPackedCompressed != nullptr &&
                            (sm120Scratch == nullptr ||
                             localScratch != nullptr)) {
                            ok[rank] =
                                FastllmCudaDeepSeekV4SparseAttentionDecodeCachedGraphSm120(
                                    *localQ, *localWindow, *localCompressed,
                                    localIndices, localLengths,
                                    *localPackedWindow, *localPackedCompressed,
                                    localScratch, *localSink, windowSize,
                                    compressRatio,
                                    (const int32_t*)localMeta->cudaData, ropeDim,
                                    ropeBase, originalSeqLen, ropeFactor,
                                    betaFast, betaSlow, softmaxScale,
                                    *localOutput);
                        }
                        if (!ok[rank]) {
                            ok[rank] =
                                FastllmCudaDeepSeekV4SparseAttentionDecodeCachedGraph(
                                    *localQ, *localWindow, *localCompressed,
                                    localIndices, localLengths,
                                    *localSink, windowSize, compressRatio,
                                    (const int32_t*)localMeta->cudaData, ropeDim,
                                    ropeBase, originalSeqLen, ropeFactor,
                                    betaFast, betaSlow, softmaxScale,
                                    *localOutput);
                        }
                    } else {
                        ok[rank] = FastllmCudaDeepSeekV4SparseAttentionDecodeCached(
                            *localQ, *localWindow, *localCompressed, *localSink,
                            windowSize, startPos, compressedCount, ropeDim,
                            ropeBase, originalSeqLen, ropeFactor, betaFast,
                            betaSlow, softmaxScale, *localOutput);
                    }
                });
                for (char state : ok) {
                    AssertInFastLLM(state != 0,
                                    "DeepSeek V4 MultiCuda sparse decode rejected a local shard.\n");
                }
                return;
            }
            if (compressedTopK == nullptr && q.dims[1] == 1) {
                Data qCuda, windowCuda, compressedCuda;
                const Data *qForCuda = &q;
                if (q.dataDevice != DataDevice::CUDA) {
                    qCuda.CopyFrom(q);
                    qCuda.ToDevice(DataDevice::CUDA);
                    qForCuda = &qCuda;
                }
                std::vector<int> targetDeviceIds = qForCuda->dataDeviceIds;
                if (qForCuda->cudaData != nullptr) {
                    int realQDevice = GetPointerDeviceId(qForCuda->cudaData);
                    if (realQDevice >= 0) {
                        targetDeviceIds = {realQDevice};
                    }
                }
                const Data *windowForCuda = &windowKV;
                bool needWindowCopy = windowKV.dataDevice != DataDevice::CUDA;
                if (!needWindowCopy && windowKV.cudaData != nullptr && !targetDeviceIds.empty()) {
                    int realWindowDevice = GetPointerDeviceId(windowKV.cudaData);
                    needWindowCopy = realWindowDevice >= 0 && realWindowDevice != targetDeviceIds[0];
                }
                if (needWindowCopy) {
                    windowCuda.CopyFrom(windowKV);
                    windowCuda.ToDevice(DataDevice::CUDA, targetDeviceIds);
                    windowForCuda = &windowCuda;
                }
                const Data *compressedForCuda = &compressedKV;
                bool needCompressedCopy = compressedCount > 0 && compressedKV.dataDevice != DataDevice::CUDA;
                if (!needCompressedCopy && compressedCount > 0 &&
                    compressedKV.cudaData != nullptr && !targetDeviceIds.empty()) {
                    int realCompressedDevice = GetPointerDeviceId(compressedKV.cudaData);
                    needCompressedCopy = realCompressedDevice >= 0 && realCompressedDevice != targetDeviceIds[0];
                }
                if (needCompressedCopy) {
                    compressedCuda.CopyFrom(compressedKV);
                    compressedCuda.ToDevice(DataDevice::CUDA, targetDeviceIds);
                    compressedForCuda = &compressedCuda;
                }
                attnSink.ToDevice(DataDevice::CUDA, targetDeviceIds);
                if (decodeMeta != nullptr && decodeMeta->dataDevice == DataDevice::CUDA &&
                    decodeMeta->cudaData != nullptr) {
                    if (DeepSeekV4SparseMlaSm120Enabled() &&
                        packedWindowKV != nullptr &&
                        packedCompressedKV != nullptr &&
                        FastllmCudaDeepSeekV4SparseAttentionDecodeCachedGraphSm120(
                            *qForCuda, *windowForCuda, *compressedForCuda,
                            compressedIndices, compressedLengths,
                            *packedWindowKV, *packedCompressedKV,
                            sm120Scratch, attnSink, windowSize, compressRatio,
                            (const int32_t*)decodeMeta->cudaData, ropeDim,
                            ropeBase, originalSeqLen, ropeFactor, betaFast,
                            betaSlow, softmaxScale, output)) {
                        return;
                    }
                    if (FastllmCudaDeepSeekV4SparseAttentionDecodeCachedGraph(
                            *qForCuda, *windowForCuda, *compressedForCuda,
                            compressedIndices, compressedLengths,
                            attnSink, windowSize, compressRatio,
                            (const int32_t*)decodeMeta->cudaData, ropeDim,
                            ropeBase, originalSeqLen, ropeFactor,
                            betaFast, betaSlow, softmaxScale, output)) {
                        return;
                    }
                }
                if (FastllmCudaDeepSeekV4SparseAttentionDecodeCached(*qForCuda, *windowForCuda,
                                                                     *compressedForCuda,
                                                                     attnSink, windowSize, startPos,
                                                                     compressedCount, ropeDim, ropeBase,
                                                                     originalSeqLen, ropeFactor,
                                                                     betaFast, betaSlow, softmaxScale, output)) {
                    return;
                }
            }
#endif
            if (q.dataDevice == DataDevice::CPU &&
                windowKV.dataDevice == DataDevice::CPU &&
                attnSink.dataDevice == DataDevice::CPU &&
                (compressedCount <= 0 ||
                 compressedKV.dataDevice == DataDevice::CPU) &&
                (compressedTopK == nullptr ||
                 compressedTopK->dataDevice == DataDevice::CPU)) {
                DeepSeekV4SparseAttentionDecodeCached(
                    q, windowKV, compressedKV, attnSink, windowSize,
                    startPos, compressedCount, ropeDim, ropeBase,
                    softmaxScale, output, originalSeqLen, ropeFactor,
                    betaFast, betaSlow, compressedTopK);
                return;
            }
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
            if (compressedTopK != nullptr && compressedTopK->Count(0) > 0) {
                AssertInFastLLM(
                    compressedTopK->dataDevice == DataDevice::CPU &&
                        compressedTopK->dataType == DataType::INT32 &&
                        compressedTopK->cpuData != nullptr,
                    "DeepSeekV4 sparse decode received invalid top-k indices.\n");
                const int32_t *topKData =
                    (const int32_t*)compressedTopK->cpuData;
                int topKCount = (int)compressedTopK->Count(0);
                for (int i = 0; i < topKCount; i++) {
                    int idx = topKData[i];
                    if (idx >= 0 && idx < compressedCount) {
                        idxs.push_back(windowSize + idx);
                    }
                }
            } else {
                for (int i = 0; i < compressedCount; i++) {
                    idxs.push_back(windowSize + i);
                }
            }

            auto getKVRow = [&](int b, int idx) -> const float* {
                if (idx < windowSize) {
                    return windowValues.data() + ((uint64_t)b * windowSize + idx) * dim;
                }
                return compressed.data() + ((uint64_t)b * compressedCount + (idx - windowSize)) * dim;
            };

            bool useParallelHeads =
                bsz * heads > 1 &&
                !EnvFlagEnabled(
                    "FASTLLM_DSV4_DISABLE_CPU_SPARSE_DECODE_PARALLEL");
            if (useParallelHeads) {
                auto *pool = GetAlivePool();
                int sparseThreads = idxs.size() <= 8 ? 12 : 24;
                sparseThreads = std::min(
                    {(int)pool->threads.size(), sparseThreads,
                     bsz * heads});
                std::vector<SparseAttentionDecodeCachedHeadsOp*> ops;
                int totalHeads = bsz * heads;
                int per = totalHeads / sparseThreads;
                int cur = 0;
                for (int i = 0; i < sparseThreads; i++) {
                    int end = cur + per +
                        (cur + per * (sparseThreads - i) < totalHeads);
                    if (i == sparseThreads - 1) {
                        end = totalHeads;
                    }
                    ops.push_back(new SparseAttentionDecodeCachedHeadsOp(
                        qv.data(), windowValues.data(), compressed.data(),
                        sink.data(), idxs.data(), out.data(), cur, end,
                        heads, dim, windowSize, compressedCount,
                        (int)idxs.size(), softmaxScale));
                    cur = end;
                }
                for (int i = 0; i < sparseThreads; i++) {
                    pool->PushOp(i, ops[i]);
                }
                for (int i = 0; i < sparseThreads; i++) {
                    pool->Wait(i);
                    delete ops[i];
                }
            } else for (int b = 0; b < bsz; b++) {
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

        static bool SparseAttentionDecodeCachedBatch(Data &attnSink,
                                                     int windowSize,
                                                     int ropeDim, float ropeBase, float softmaxScale,
                                                     const std::vector<int> &startPositions,
                                                     const std::vector<int> &compressedCounts,
                                                     std::vector<Data*> &qParts,
                                                     std::vector<Data*> &windowKVs,
                                                     std::vector<Data*> &compressedKVs,
                                                     Data &output, int originalSeqLen = 0,
                                                     float ropeFactor = 1.0f,
                                                     int betaFast = 32, int betaSlow = 1) {
            ScopedExecutorProfiler executorProfile("DeepSeekV4SparseDecodeCachedBatch");
#ifdef USE_CUDA
            if (!DeepSeekV4PreferCuda() || qParts.empty() ||
                qParts.size() != windowKVs.size() ||
                qParts.size() != compressedKVs.size() ||
                qParts.size() != startPositions.size() ||
                qParts.size() != compressedCounts.size()) {
                return false;
            }
            for (size_t i = 0; i < qParts.size(); i++) {
                if (qParts[i] == nullptr || windowKVs[i] == nullptr || compressedKVs[i] == nullptr ||
                    qParts[i]->dataDevice != DataDevice::CUDA ||
                    windowKVs[i]->dataDevice != DataDevice::CUDA ||
                    (compressedCounts[i] > 0 && compressedKVs[i]->dataDevice != DataDevice::CUDA)) {
                    return false;
                }
            }
            attnSink.ToDevice(DataDevice::CUDA);
            return FastllmCudaDeepSeekV4SparseAttentionDecodeCachedBatch(
                qParts, windowKVs, compressedKVs, attnSink, windowSize,
                startPositions, compressedCounts, ropeDim, ropeBase, originalSeqLen,
                ropeFactor, betaFast, betaSlow, softmaxScale, output);
#else
            (void)attnSink;
            (void)windowSize;
            (void)ropeDim;
            (void)ropeBase;
            (void)softmaxScale;
            (void)startPositions;
            (void)compressedCounts;
            (void)qParts;
            (void)windowKVs;
            (void)compressedKVs;
            (void)output;
            (void)originalSeqLen;
            (void)ropeFactor;
            (void)betaFast;
            (void)betaSlow;
            return false;
#endif
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
            Data xFloat;
            ToDataType(x, xFloat, DataType::FLOAT32);
            Linear(xFloat, weight[prefix + ".wkv.weight"], Data(), kv, true);
            Linear(xFloat, weight[prefix + ".wgate.weight"], Data(), score, true);

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

        static bool HasCpuIndexerWeights(WeightMap &weight,
                                         const std::string &attentionPrefix) {
            static const char *suffixes[] = {
                ".indexer.wq_b.weight",
                ".indexer.weights_proj.weight",
                ".indexer.compressor.wkv.weight",
                ".indexer.compressor.wgate.weight",
                ".indexer.compressor.ape",
                ".indexer.compressor.norm.weight"
            };
            for (const char *suffix : suffixes) {
                if (weight.weight.find(attentionPrefix + suffix) ==
                    weight.weight.end()) {
                    return false;
                }
            }
            return true;
        }

        static bool PrepareCpuIndexerTopK(
                WeightMap &weight, const std::string &attentionPrefix,
                Data &attnInput, Data &qNorm, int bsz, int seqlen,
                int startPos, int compressRatio, int indexHeads,
                int indexHeadDim, int indexTopK, int ropeDim,
                float ropeBase, int originalSeqLen, float ropeFactor,
                int betaFast, int betaSlow,
                DeepSeekV4DecodeLayerCache *decodeCache,
                Data &topKIndices) {
            if (compressRatio != 4 || DeepSeekV4PreferCuda() ||
                EnvFlagEnabled("FASTLLM_DSV4_DISABLE_CPU_INDEXER") ||
                !HasCpuIndexerWeights(weight, attentionPrefix)) {
                return false;
            }
            ScopedExecutorProfiler executorProfile("DeepSeekV4Indexer");
            const std::string indexerPrefix = attentionPrefix + ".indexer";
            const std::string compressorPrefix = indexerPrefix + ".compressor";

            Data indexQ, indexWeights;
            weight[indexerPrefix + ".wq_b.weight"].tpLinearType =
                TP_LINEAR_ROW;
            DeepSeekV4Linear(
                qNorm, weight[indexerPrefix + ".wq_b.weight"], Data(),
                indexQ);
            indexQ.Reshape({bsz, seqlen, indexHeads, indexHeadDim});
            weight[indexerPrefix + ".weights_proj.weight"].tpLinearType =
                TP_LINEAR_ROW;
            DeepSeekV4Linear(
                attnInput, weight[indexerPrefix + ".weights_proj.weight"],
                Data(), indexWeights);
            indexWeights.Reshape({bsz, seqlen, indexHeads});
            if (indexWeights.dataType != DataType::BFLOAT16) {
                ToDataType(indexWeights, DataType::BFLOAT16);
            }

            Data rawKV, rawScore;
            ComputeCompressorRaw(
                weight, compressorPrefix, attnInput, rawKV, rawScore);
            const int wideDim = 2 * indexHeadDim;
            const int totalLen = startPos + seqlen;
            const int targetBlocks = totalLen / compressRatio;
            if (targetBlocks <= 0) {
                return false;
            }

            Data transientCompressed;
            Data *compressed = &transientCompressed;
            if (decodeCache == nullptr) {
                if (startPos != 0 || !BuildCompressedKVFromRaw(
                        weight, compressorPrefix, rawKV, rawScore, bsz, 0,
                        totalLen, compressRatio, indexHeadDim, ropeDim,
                        ropeBase, ropeFactor, betaFast, betaSlow,
                        originalSeqLen, transientCompressed, false, true)) {
                    return false;
                }
            } else {
                if (startPos == 0) {
                    decodeCache->indexerCompressorWideDim = wideDim;
                    decodeCache->indexerCompressorRawTokenBase = 0;
                    decodeCache->indexerCompressedBlocks = 0;
                    ResetData(decodeCache->indexerCompressedKV);
                    CopyTensorData(decodeCache->indexerCompressorKVRaw, rawKV);
                    CopyTensorData(decodeCache->indexerCompressorScoreRaw,
                                   rawScore);
                    EnsureCompressorRawCapacity(
                        decodeCache->indexerCompressorKVRaw, seqlen);
                    EnsureCompressorRawCapacity(
                        decodeCache->indexerCompressorScoreRaw, seqlen);
                } else {
                    AppendCompressorRaw(
                        rawKV, rawScore, bsz, seqlen, wideDim,
                        decodeCache->indexerCompressorKVRaw,
                        decodeCache->indexerCompressorScoreRaw);
                }

                bool targetReady =
                    decodeCache->indexerCompressedBlocks == targetBlocks &&
                    HasCompressedKVData(decodeCache->indexerCompressedKV);
                if (!targetReady) {
                    bool built = BuildCompressedKVFromRaw(
                        weight, compressorPrefix,
                        decodeCache->indexerCompressorKVRaw,
                        decodeCache->indexerCompressorScoreRaw, bsz,
                        decodeCache->indexerCompressorRawTokenBase, totalLen,
                        compressRatio, indexHeadDim, ropeDim, ropeBase,
                        ropeFactor, betaFast, betaSlow, originalSeqLen,
                        decodeCache->indexerCompressedKV, false, true);
                    if (!built) {
                        return false;
                    }
                    decodeCache->indexerCompressedBlocks =
                        GetReusableCompressedBlocks(
                            decodeCache->indexerCompressedKV, bsz,
                            targetBlocks, indexHeadDim);
                }
                TrimCompressorRawCache(
                    bsz, totalLen, compressRatio, wideDim,
                    decodeCache->indexerCompressedBlocks,
                    decodeCache->indexerCompressorKVRaw,
                    decodeCache->indexerCompressorScoreRaw,
                    decodeCache->indexerCompressorRawTokenBase);
                compressed = &decodeCache->indexerCompressedKV;
            }

            if (!HasCompressedKVData(*compressed) ||
                compressed->dataDevice != DataDevice::CPU) {
                return false;
            }
            DeepSeekV4IndexerTopK(
                indexQ, indexWeights, *compressed, indexTopK,
                compressRatio, ropeDim, ropeBase, startPos,
                originalSeqLen, ropeFactor, betaFast, betaSlow,
                topKIndices);
            return topKIndices.dataType == DataType::INT32 &&
                topKIndices.dims.size() == 3 &&
                topKIndices.Count(0) > 0;
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

#ifdef USE_CUDA
        static bool DeepSeekV4RouteScoreTransformMultiCuda(Data &logits, int scoreFuncMode) {
            if (!logits.multiDeviceData || !logits.IsTensorParallelReplicated()) {
                return FastllmCudaDeepSeekV4RouteScoreTransform(logits, scoreFuncMode);
            }

            std::vector<int> devices;
            std::map<int, int> ratios;
            FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
            if (devices.size() <= 1) {
                return FastllmCudaDeepSeekV4RouteScoreTransform(logits, scoreFuncMode);
            }

            std::vector<char> ok(devices.size(), 0);
            RunDeepSeekV4MultiCuda(devices, [&](int rank, int device) {
                auto it = logits.multiDeviceDatas.find(device);
                if (it != logits.multiDeviceDatas.end() && it->second != nullptr) {
                    ok[rank] = FastllmCudaDeepSeekV4RouteScoreTransform(
                        *it->second, scoreFuncMode);
                }
            });
            return std::all_of(ok.begin(), ok.end(), [](char state) { return state != 0; });
        }

        static bool DeepSeekV4SqrtSoftplusRouterMultiCuda(
                Data &logits, Data &gateBias, float routeScale,
                Data &expertIndex, Data &expertScore) {
            constexpr int topk = 6;
            if (!logits.multiDeviceData ||
                !logits.IsTensorParallelReplicated()) {
                return FastllmCudaDeepSeekV4SqrtSoftplusRouter(
                    logits, gateBias, routeScale,
                    expertIndex, expertScore);
            }

            std::vector<int> devices;
            std::map<int, int> ratios;
            FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
            if (devices.size() <= 1) {
                return FastllmCudaDeepSeekV4SqrtSoftplusRouter(
                    logits, gateBias, routeScale,
                    expertIndex, expertScore);
            }

            PrepareMultiCudaReplicatedData(gateBias, devices, true);
            int tokens = logits.dims.empty() ? 0 :
                (int)(logits.Count(0) / logits.dims.back());
            expertIndex.dataType = DataType::INT32;
            expertIndex.Resize({tokens, topk});
            expertIndex.dataDevice = DataDevice::CUDA;
            expertIndex.dataDeviceIds = {devices[0]};
            PrepareMultiCudaReplicatedData(expertIndex, devices, false);

            expertScore.dataType = DataType::FLOAT32;
            expertScore.Resize({tokens, topk});
            expertScore.dataDevice = DataDevice::CUDA;
            expertScore.dataDeviceIds = {devices[0]};
            PrepareMultiCudaReplicatedData(expertScore, devices, false);

            std::vector<char> ok(devices.size(), 0);
            RunDeepSeekV4MultiCuda(devices, [&](int rank, int device) {
                auto logitsIt = logits.multiDeviceDatas.find(device);
                auto biasIt = gateBias.multiDeviceDatas.find(device);
                auto indexIt = expertIndex.multiDeviceDatas.find(device);
                auto scoreIt = expertScore.multiDeviceDatas.find(device);
                if (logitsIt != logits.multiDeviceDatas.end() &&
                    logitsIt->second != nullptr &&
                    biasIt != gateBias.multiDeviceDatas.end() &&
                    biasIt->second != nullptr &&
                    indexIt != expertIndex.multiDeviceDatas.end() &&
                    indexIt->second != nullptr &&
                    scoreIt != expertScore.multiDeviceDatas.end() &&
                    scoreIt->second != nullptr) {
                    ok[rank] = FastllmCudaDeepSeekV4SqrtSoftplusRouter(
                        *logitsIt->second, *biasIt->second, routeScale,
                        *indexIt->second, *scoreIt->second);
                }
            });
            return std::all_of(ok.begin(), ok.end(),
                               [](char state) { return state != 0; });
        }

        static bool DeepSeekV4HashRouteScoreMultiCuda(
                Data &logits, Data &tid2eid, const int *inputIds, int tokens,
                int topk, int scoreFuncMode, float routeScale,
                Data &expertIndex, Data &expertScore,
                const Data *decodeMeta = nullptr) {
            if (!logits.multiDeviceData || !logits.IsTensorParallelReplicated()) {
                if (decodeMeta != nullptr && decodeMeta->dataDevice == DataDevice::CUDA &&
                    decodeMeta->cudaData != nullptr) {
                    return FastllmCudaDeepSeekV4HashRouteScoreGraph(
                        logits, tid2eid, (const int32_t*)decodeMeta->cudaData,
                        tokens, topk, scoreFuncMode, routeScale,
                        expertIndex, expertScore);
                }
                return FastllmCudaDeepSeekV4HashRouteScore(
                    logits, tid2eid, inputIds, tokens, topk, scoreFuncMode,
                    routeScale, expertIndex, expertScore);
            }

            std::vector<int> devices;
            std::map<int, int> ratios;
            FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
            if (devices.size() <= 1) {
                return FastllmCudaDeepSeekV4HashRouteScore(
                    logits, tid2eid, inputIds, tokens, topk, scoreFuncMode,
                    routeScale, expertIndex, expertScore);
            }

            bool cpuRouteTable = tid2eid.dataDevice == DataDevice::CPU &&
                                 (tid2eid.dataType == DataType::INT32PARAM ||
                                  tid2eid.dataType == DataType::INT32);
            if (!cpuRouteTable) {
                PrepareMultiCudaReplicatedData(tid2eid, devices, true);
            }

            expertIndex.dataType = DataType::INT32;
            expertIndex.Resize({tokens, topk});
            expertIndex.dataDevice = DataDevice::CUDA;
            expertIndex.dataDeviceIds = {devices[0]};
            PrepareMultiCudaReplicatedData(expertIndex, devices, false);

            expertScore.dataType = DataType::FLOAT32;
            expertScore.Resize({tokens, topk});
            expertScore.dataDevice = DataDevice::CUDA;
            expertScore.dataDeviceIds = {devices[0]};
            PrepareMultiCudaReplicatedData(expertScore, devices, false);

            std::vector<char> ok(devices.size(), 0);
            RunDeepSeekV4MultiCuda(devices, [&](int rank, int device) {
                auto logitsIt = logits.multiDeviceDatas.find(device);
                auto indexIt = expertIndex.multiDeviceDatas.find(device);
                auto scoreIt = expertScore.multiDeviceDatas.find(device);
                Data *routeData = &tid2eid;
                auto routeIt = tid2eid.multiDeviceDatas.find(device);
                if (routeIt != tid2eid.multiDeviceDatas.end() && routeIt->second != nullptr) {
                    routeData = routeIt->second;
                }
                if (logitsIt != logits.multiDeviceDatas.end() && logitsIt->second != nullptr &&
                    indexIt != expertIndex.multiDeviceDatas.end() && indexIt->second != nullptr &&
                    scoreIt != expertScore.multiDeviceDatas.end() && scoreIt->second != nullptr) {
                    const Data *localMeta = decodeMeta == nullptr ? nullptr :
                        GetTensorCudaReplica(*decodeMeta, device);
                    if (localMeta != nullptr) {
                        ok[rank] = FastllmCudaDeepSeekV4HashRouteScoreGraph(
                            *logitsIt->second, *routeData,
                            (const int32_t*)localMeta->cudaData, tokens, topk,
                            scoreFuncMode, routeScale, *indexIt->second,
                            *scoreIt->second);
                    } else {
                        ok[rank] = FastllmCudaDeepSeekV4HashRouteScore(
                            *logitsIt->second, *routeData, inputIds, tokens,
                            topk, scoreFuncMode, routeScale,
                            *indexIt->second, *scoreIt->second);
                    }
                }
            });
            return std::all_of(ok.begin(), ok.end(), [](char state) { return state != 0; });
        }
#endif

        static void BuildMoERoutingData(WeightMap &weight, const std::string &prefix, const Data &x,
                                        const std::vector<int> &inputIds, int nRoutedExperts,
                                        int topk, const std::string &scoreFunc, float routeScale,
                                        Data &expertIndex, Data &expertScore,
                                        const Data *decodeMeta = nullptr) {
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
                if (DeepSeekV4HashRouteScoreMultiCuda(routerLogits, weight[prefix + ".gate.tid2eid"],
                                                      inputIds.data(), x.dims[0], topk,
                                                      scoreFuncMode, routeScale,
                                                      expertIndex, expertScore, decodeMeta)) {
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
                if (scoreFuncMode == 2 && topk == 6 &&
                    gateBiasData != nullptr &&
                    DeepSeekV4SqrtSoftplusRouterMultiCuda(
                        routerLogits, *gateBiasData, routeScale,
                        expertIndex, expertScore)) {
                    return;
                }
                if (DeepSeekV4RouteScoreTransformMultiCuda(routerLogits, scoreFuncMode)) {
                    bool needNorm = scoreFunc != "softmax";
                    SelectExpert(routerLogits, expertIndex, expertScore, topk, needNorm, routeScale, gateBiasData);
                    return;
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
#ifdef USE_CUDA
            const Data *cudaX = &x;
            if (x.multiDeviceData && x.IsTensorParallelReplicated()) {
                for (const auto &deviceData : x.multiDeviceDatas) {
                    if (deviceData.second != nullptr &&
                        deviceData.second->dataDevice == DataDevice::CUDA &&
                        deviceData.second->cudaData != nullptr) {
                        cudaX = deviceData.second;
                        break;
                    }
                }
            }
            if (cudaX->dataDevice == DataDevice::CUDA && cudaX->cudaData != nullptr) {
                std::vector<int> targetDevices = cudaX->dataDeviceIds;
                int realDevice = GetPointerDeviceId(cudaX->cudaData);
                if (realDevice >= 0) {
                    targetDevices = {realDevice};
                    FastllmCudaSetDevice(realDevice);
                }
                hcFn.ToDevice(DataDevice::CUDA, targetDevices);
                hcScale.ToDevice(DataDevice::CUDA, targetDevices);
                hcBase.ToDevice(DataDevice::CUDA, targetDevices);
                if (FastllmCudaDeepSeekV4HcHead(
                        *cudaX, hcFn, hcScale, hcBase, hcMult, eps, normEps, output)) {
                    return;
                }
            }
#endif
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
                    float scaledMix = (float)mix * rsqrt;
                    float pre = SigmoidFloat(scaledMix * scale[0] + base[h]) + eps;
                    for (int d = 0; d < dim; d++) {
                        y[(uint64_t)t * dim + d] += pre * xrow[(uint64_t)h * dim + d];
                    }
                }
            }
            WriteFloatData(y, {bsz, seqlen, dim}, output, x.dataType);
        }

        static bool DeepSeekV4HeadLogitsCpu(Data &headInput, Data &normWeight,
                                            Data &headWeight, float normEps,
                                            Data &logits) {
            if (headInput.dataDevice != DataDevice::CPU || headInput.cpuData == nullptr ||
                headInput.multiDeviceData) {
                return false;
            }
            // ParallelHead stores checkpoint-BF16 values in an FP32 parameter
            // and evaluates F.linear(norm(x).float(), weight.float()).  Keep
            // norm's BF16 boundary, then use the CPU F32 x BF16 GEMM so logits
            // remain FP32 instead of being rounded to BF16 before sampling.
            Data normalized, normalizedFloat;
            RMSNormReference(headInput, normWeight, normEps, normalized, DataType::BFLOAT16);
            ToDataType(normalized, normalizedFloat, DataType::FLOAT32);
            Linear(normalizedFloat, headWeight, *GetEmptyData(), logits);
            ToDataType(logits, DataType::FLOAT32);
            return true;
        }
    }

#ifdef USE_CUDA
    bool DeepSeekV4CopyCudaTensorToCpuForTest(
            const Data &source, Data &destination) {
        return CopyDeepSeekV4CudaTensorToCpu(source, destination);
    }

    bool DeepSeekV4AddQuantizedCudaReplicaForTest(
            Data &activation, int device) {
        return AddDeepSeekV4QuantizedCudaReplica(activation, device);
    }

    int DeepSeekV4BuildWindowKVPrefixForTest(
            const Data &windowKV, int bsz, int headDim, int startPos,
            int windowSize, Data &output) {
        return BuildWindowKVPrefixData(
            windowKV, bsz, headDim, startPos, windowSize, output);
    }

    void DeepSeekV4UpdateWindowKVCacheForTest(
            const Data &kv, int bsz, int headDim, int startPos,
            int windowSize, Data &windowKV) {
        UpdateWindowKVCache(
            kv, bsz, headDim, startPos, windowSize, windowKV);
    }

    void DeepSeekV4AppendCompressorRawForTest(
            const Data &kv, const Data &score, int bsz, int seqlen,
            int wideDim, Data &allKV, Data &allScore) {
        AppendCompressorRaw(
            kv, score, bsz, seqlen, wideDim, allKV, allScore);
    }

    void DeepSeekV4TrimCompressorRawForTest(
            int bsz, int totalLen, int compressRatio, int wideDim,
            int compressedBlocks, Data &allKV, Data &allScore,
            int &rawTokenBase) {
        TrimCompressorRawCache(
            bsz, totalLen, compressRatio, wideDim, compressedBlocks,
            allKV, allScore, rawTokenBase);
    }
#endif

    struct DeepSeekV4DecodeWorkspace {
        Data hiddenStates;
        Data hiddenStatesBeforeHcExpand;
        Data hiddenStatesTemp;
        Data attnInput;
        Data qr;
        Data qNorm;
        Data q;
        Data kv;
        HcMix attnMix;
        HcMix ffnMix;
        Data ffnInput;
        Data ffnOut;
        Data expertIndex;
        Data expertScore;
        Data w1;
        Data w2;
        Data w3;
        Data tempInput;
        Data tempOutput;
        Data moeInputTemp;
        Data moeOutputTemp;
        Data compressorKV;
        Data compressorScore;
        Data indexerCompressorKV;
        Data indexerCompressorScore;
        Data indexerQ;
        Data indexerWeights;
        // SM120 sparse-MLA decode uses one temporary buffer per TP rank. Keep
        // it request-owned so CUDA Graph capture never depends on incidental
        // allocator-pool sizes left behind by prefill or KV-cache dtype.
        Data sparseMlaScratch;
        Data attnOut4;
        Data woAOut;
        Data attnOut;
        Data sharedExpertOut;
        Data sharedW1;
        Data sharedW3;
        Data headInput;
        Data samplingHeadRoot;
        Data samplingHeadReplicated;
        Data samplingHeadNorm;
        Data samplingLogits;
        Data samplingLogitsFloat;
        Data samplingGreedyIds;
        Data samplingGreedyScores;
        Data dsparkTargetCombinedTemp;
        Data dsparkTargetCombined;
        Data dsparkTargetProjected;
        Data dsparkTargetMainHidden;
        std::vector<Data> dsparkTargetStageKV;
        Data dsparkMarkovPreviousId;
        Data dsparkMarkovLocalCandidates;
        Data dsparkMarkovGatheredCandidates;
        Data dsparkMarkovCandidatePointers;
        Data dsparkMarkovLatentSignal;
        Data dsparkMarkovLatentSeen;
        Data dsparkMarkovLatentReplicas;
        Data dsparkMarkovCandidateSignals;
        Data dsparkMarkovCandidateSignalPointers;
        Data dsparkMarkovCandidateSeen;
        Data dsparkMarkovGlobalOffsets;
        Data dsparkMarkovProposalIds;
        Data dsparkMarkovProposalSignal;
        Data dsparkMarkovProposalSeen;
        std::vector<Data> dsparkMarkovLatents;
        std::vector<Data> dsparkMarkovBiasesRaw;
        std::vector<Data> dsparkMarkovBiasesFloat;
        std::vector<int> dsparkMarkovDevices;
        std::vector<int> dsparkMarkovOffsets;
        int dsparkMarkovRootDevice = -1;
        bool dsparkMarkovPrepared = false;
        bool dsparkMarkovProposalPeerReady = false;
    };

    // The scheduler uses the pointer-based batched overload even when batch=1.
    // Let that overload borrow the request's real KV holders and enter the
    // single-request implementation without losing its per-request DSV4 cache.
    static thread_local std::shared_ptr<DeepSeekV4RequestState>
        deepSeekV4ForwardRequestStateOverride;

    struct DeepSeekV4RequestStateOverrideScope {
        std::shared_ptr<DeepSeekV4RequestState> previous;

        explicit DeepSeekV4RequestStateOverrideScope(
                const std::shared_ptr<DeepSeekV4RequestState> &next) :
                previous(deepSeekV4ForwardRequestStateOverride) {
            deepSeekV4ForwardRequestStateOverride = next;
        }

        ~DeepSeekV4RequestStateOverrideScope() {
            deepSeekV4ForwardRequestStateOverride = previous;
        }
    };

    // A target verification needs every row of the target head and the three
    // auxiliary HC states.  Keep this invocation-local so concurrent requests
    // never publish captures into one another.
    static thread_local DeepSeekV4DsparkTargetCapture *
        deepSeekV4DsparkTargetCapture = nullptr;

    struct DeepSeekV4DsparkTargetCaptureScope {
        DeepSeekV4DsparkTargetCapture *previous;

        explicit DeepSeekV4DsparkTargetCaptureScope(
                DeepSeekV4DsparkTargetCapture *next) :
                previous(deepSeekV4DsparkTargetCapture) {
            deepSeekV4DsparkTargetCapture = next;
        }

        ~DeepSeekV4DsparkTargetCaptureScope() {
            deepSeekV4DsparkTargetCapture = previous;
        }
    };

#ifdef USE_CUDA
    struct DeepSeekV4DsparkTargetGpuInput {
        Data *proposalIds = nullptr;
        Data *readySignal = nullptr;
        Data *readySeen = nullptr;
        int anchorToken = 0;
        int startPos = 0;
        int proposalCount = 0;
    };

    static thread_local const DeepSeekV4DsparkTargetGpuInput *
        deepSeekV4DsparkTargetGpuInput = nullptr;

    struct DeepSeekV4DsparkTargetGpuInputScope {
        const DeepSeekV4DsparkTargetGpuInput *previous;

        explicit DeepSeekV4DsparkTargetGpuInputScope(
                const DeepSeekV4DsparkTargetGpuInput *next) :
                previous(deepSeekV4DsparkTargetGpuInput) {
            deepSeekV4DsparkTargetGpuInput = next;
        }

        ~DeepSeekV4DsparkTargetGpuInputScope() {
            deepSeekV4DsparkTargetGpuInput = previous;
        }
    };

    struct DeepSeekV4DsparkDraftGpuInput {
        Data *acceptanceResult = nullptr;
        Data *readySignal = nullptr;
        Data *readySeen = nullptr;
        std::vector<Data*> stageKV;
        int baseCommittedTokens = 0;
        int rows = 0;
    };

    static thread_local const DeepSeekV4DsparkDraftGpuInput *
        deepSeekV4DsparkDraftGpuInput = nullptr;

    struct DeepSeekV4DsparkDraftGpuInputScope {
        const DeepSeekV4DsparkDraftGpuInput *previous;

        explicit DeepSeekV4DsparkDraftGpuInputScope(
                const DeepSeekV4DsparkDraftGpuInput *next) :
                previous(deepSeekV4DsparkDraftGpuInput) {
            deepSeekV4DsparkDraftGpuInput = next;
        }

        ~DeepSeekV4DsparkDraftGpuInputScope() {
            deepSeekV4DsparkDraftGpuInput = previous;
        }
    };
#endif

#ifdef USE_CUDA
    static bool DeepSeekV4HcMeanCuda(const Data &input, Data &output);
#endif

    static void DeepSeekV4HcMean(const Data &input, Data &output) {
        AssertInFastLLM(
            input.dims.size() == 4 && input.dims[2] > 0,
            "DeepSeek-V4 DSpark HC capture expects [b,s,hc,d].");
#ifdef USE_CUDA
        if (DeepSeekV4HcMeanCuda(input, output)) {
            return;
        }
#endif
        const int hc = input.dims[2];
        // The generic scalar Mul/Add operators do not accept BF16.  Accumulate
        // in FP32 (the same reduction precision used by torch.mean) and restore
        // the model's BF16 activation boundary before main_proj.
        Data inputFloat, meanFloat;
        ToDataType(input, inputFloat, DataType::FLOAT32);
        for (int index = 0; index < hc; index++) {
            Data selected;
            Split(inputFloat, 2, index, index + 1, selected);
            selected.Reshape(
                {input.dims[0], input.dims[1], input.dims[3]});
            if (index == 0) {
                Mul(selected, 1.0f / hc, meanFloat);
            } else {
                AddTo(meanFloat, selected, 1.0f / hc);
            }
        }
        ToDataType(meanFloat, output, DataType::BFLOAT16);
    }

#ifdef USE_CUDA
    static constexpr int kDeepSeekV4CudaGraphMetaInts = 64;

    struct DeepSeekV4CudaGraphDeviceState {
        int device = -1;
        void *graph = nullptr;
        void *exec = nullptr;
        void *workerStartEvent = nullptr;
        void *workerEndEvent = nullptr;
        void *replayDoneEvent = nullptr;
        std::vector<void*> markovLatentReadyEvents;
        std::vector<void*> markovCandidateReadyEvents;
        Data *decodeMeta = nullptr;
    };

    struct DeepSeekV4CudaGraphState {
        std::mutex mutex;
        bool warmed = false;
        int warmupRounds = 0;
        bool captured = false;
        bool disabled = false;
        bool capturing = false;
        bool indexerScorerMode = false;
        int graphMaxTokens = 0;
        int inputDevice = -1;
        Data inputIds;
        Data graphInputIds;
        Data decodeMeta;
        void *pinnedMeta = nullptr;
        void *pinnedInputIds = nullptr;
        bool replayInputsPending = false;
        std::unique_ptr<DeepSeekV4DecodeWorkspace> workspace;
        std::vector<std::unique_ptr<DeepSeekV4CudaGraphDeviceState> > devices;
        std::map<int, DeepSeekV4CudaGraphDeviceState*> deviceIndex;
        std::vector<int> launchOrder;
        std::vector<void*> reservedPointers;

        DeepSeekV4CudaGraphState() : workspace(new DeepSeekV4DecodeWorkspace()) {}

        void DestroyCapturedGraph() {
            for (auto &device : devices) {
                if (device->exec != nullptr) {
                    FastllmCudaSetDevice(device->device);
                    FastllmCudaGraphExecDestroy(device->exec);
                    device->exec = nullptr;
                }
                if (device->graph != nullptr) {
                    FastllmCudaSetDevice(device->device);
                    FastllmCudaGraphDestroy(device->graph);
                    device->graph = nullptr;
                }
            }
            if (!reservedPointers.empty()) {
                FastllmCudaGraphMemoryPoolRelease(reservedPointers);
                reservedPointers.clear();
            }
            captured = false;
            warmed = false;
            warmupRounds = 0;
            capturing = false;
        }

        ~DeepSeekV4CudaGraphState() {
            DestroyCapturedGraph();
            workspace.reset();
            for (auto &device : devices) {
                FastllmCudaSetDevice(device->device);
                if (device->workerStartEvent != nullptr) {
                    FastllmCudaEventDestroy(device->workerStartEvent);
                }
                if (device->workerEndEvent != nullptr) {
                    FastllmCudaEventDestroy(device->workerEndEvent);
                }
                if (device->replayDoneEvent != nullptr) {
                    FastllmCudaEventDestroy(device->replayDoneEvent);
                }
                for (void *event : device->markovLatentReadyEvents) {
                    FastllmCudaEventDestroy(event);
                }
                for (void *event : device->markovCandidateReadyEvents) {
                    FastllmCudaEventDestroy(event);
                }
            }
            FastllmCudaHostFree(pinnedMeta);
            FastllmCudaHostFree(pinnedInputIds);
        }

        void PrepareDevices(const std::vector<int> &nextDevices) {
            if (!devices.empty()) {
                return;
            }
            decodeMeta.dataType = DataType::INT32;
            decodeMeta.Resize({kDeepSeekV4CudaGraphMetaInts});
            decodeMeta.dataDevice = DataDevice::CUDA;
            decodeMeta.dataDeviceIds = nextDevices;
            PrepareMultiCudaReplicatedData(decodeMeta, nextDevices, false);
            pinnedMeta = FastllmCudaHostMalloc(
                kDeepSeekV4CudaGraphMetaInts * sizeof(int32_t));
            pinnedInputIds = FastllmCudaHostMalloc(
                kDeepSeekV4CudaGraphMetaInts * sizeof(float));
            for (int id : nextDevices) {
                std::unique_ptr<DeepSeekV4CudaGraphDeviceState> state(
                    new DeepSeekV4CudaGraphDeviceState());
                state->device = id;
                FastllmCudaSetDevice(id);
                state->workerStartEvent = FastllmCudaEventCreate();
                state->workerEndEvent = FastllmCudaEventCreate();
                state->replayDoneEvent = FastllmCudaEventCreate();
                state->decodeMeta = decodeMeta.multiDeviceDatas.at(id);
                state->decodeMeta->dataType = DataType::INT32;
                state->decodeMeta->Resize({kDeepSeekV4CudaGraphMetaInts});
                state->decodeMeta->dataDevice = DataDevice::CUDA;
                state->decodeMeta->dataDeviceIds = {id};
                state->decodeMeta->Allocate(false);
                deviceIndex[id] = state.get();
                devices.push_back(std::move(state));
            }
        }

        Data *GetDecodeMeta(int device) {
            auto it = deviceIndex.find(device);
            return it == deviceIndex.end() ? nullptr : it->second->decodeMeta;
        }

        Data *GetReplicatedDecodeMeta() {
            return &decodeMeta;
        }

        void SetLaunchOrder(const std::vector<int> &pipelineDevices) {
            launchOrder.clear();
            auto appendUnique = [&](int device) {
                if (deviceIndex.find(device) != deviceIndex.end() &&
                    std::find(launchOrder.begin(), launchOrder.end(), device) ==
                        launchOrder.end()) {
                    launchOrder.push_back(device);
                }
            };
            for (int device : pipelineDevices) {
                appendUnique(device);
            }
            for (auto &device : devices) {
                appendUnique(device->device);
            }
        }

    };

    // DSpark's three-layer draft has a fixed seven-row shape, but its anchor,
    // absolute position and 128-row main-model KV window change every round.
    // Keep those mutable inputs in stable allocations and capture the complete
    // draft backbone once per request.  Intermediate allocations made during
    // capture are pinned by FastLLM's graph memory pool.
    struct DeepSeekV4DsparkCudaGraphState {
        std::mutex mutex;
        bool warmed = false;
        int warmupRounds = 0;
        bool captured = false;
        bool disabled = false;
        bool capturing = false;
        bool hasEnqueueOnlyReplay = false;
        int inputDevice = -1;
        Data inputIds;
        Data decodeMeta;
        std::vector<Data> mainWindowKV;
        Data baseLogits;
        std::unique_ptr<DeepSeekV4DecodeWorkspace> workspace;
        std::vector<std::unique_ptr<DeepSeekV4CudaGraphDeviceState> > devices;
        std::map<int, DeepSeekV4CudaGraphDeviceState*> deviceIndex;
        std::vector<int> launchOrder;
        std::vector<void*> reservedPointers;
        std::vector<uint64_t> directWindowSignature;
        void *pinnedMeta = nullptr;
        void *pinnedInputIds = nullptr;

        DeepSeekV4DsparkCudaGraphState() :
                workspace(new DeepSeekV4DecodeWorkspace()) {}

        void DestroyCapturedGraph() {
            // Enqueue-only draft replay intentionally leaves GPU execution
            // outstanding after ForwardDspark returns.  Its callbacks have
            // already recorded replayDoneEvent, so wait for the newest record
            // before releasing graph-owned workspaces during request teardown
            // or graph recovery.
            if (hasEnqueueOnlyReplay) {
                for (auto &device : devices) {
                    if (device->replayDoneEvent != nullptr) {
                        FastllmCudaSetDevice(device->device);
                        FastllmCudaEventSynchronize(
                            device->replayDoneEvent);
                    }
                }
                hasEnqueueOnlyReplay = false;
            }
            for (auto &device : devices) {
                if (device->exec != nullptr) {
                    FastllmCudaSetDevice(device->device);
                    FastllmCudaGraphExecDestroy(device->exec);
                    device->exec = nullptr;
                }
                if (device->graph != nullptr) {
                    FastllmCudaSetDevice(device->device);
                    FastllmCudaGraphDestroy(device->graph);
                    device->graph = nullptr;
                }
            }
            if (!reservedPointers.empty()) {
                FastllmCudaGraphMemoryPoolRelease(reservedPointers);
                reservedPointers.clear();
            }
            captured = false;
            warmed = false;
            warmupRounds = 0;
            capturing = false;
        }

        ~DeepSeekV4DsparkCudaGraphState() {
            DestroyCapturedGraph();
            workspace.reset();
            for (auto &device : devices) {
                FastllmCudaSetDevice(device->device);
                if (device->workerStartEvent != nullptr) {
                    FastllmCudaEventDestroy(device->workerStartEvent);
                }
                if (device->workerEndEvent != nullptr) {
                    FastllmCudaEventDestroy(device->workerEndEvent);
                }
                if (device->replayDoneEvent != nullptr) {
                    FastllmCudaEventDestroy(device->replayDoneEvent);
                }
                for (void *event : device->markovLatentReadyEvents) {
                    FastllmCudaEventDestroy(event);
                }
                for (void *event : device->markovCandidateReadyEvents) {
                    FastllmCudaEventDestroy(event);
                }
            }
            FastllmCudaHostFree(pinnedMeta);
            FastllmCudaHostFree(pinnedInputIds);
        }

        void PrepareDevices(const std::vector<int> &nextDevices,
                            int layers, int markovSteps) {
            if (!devices.empty()) {
                return;
            }
            decodeMeta.dataType = DataType::INT32;
            decodeMeta.Resize({kDeepSeekV4CudaGraphMetaInts});
            decodeMeta.dataDevice = DataDevice::CUDA;
            decodeMeta.dataDeviceIds = nextDevices;
            PrepareMultiCudaReplicatedData(decodeMeta, nextDevices, false);
            pinnedMeta = FastllmCudaHostMalloc(
                kDeepSeekV4CudaGraphMetaInts * sizeof(int32_t));
            pinnedInputIds = FastllmCudaHostMalloc(
                std::max(1, markovSteps) * sizeof(float));
            for (int id : nextDevices) {
                std::unique_ptr<DeepSeekV4CudaGraphDeviceState> state(
                    new DeepSeekV4CudaGraphDeviceState());
                state->device = id;
                FastllmCudaSetDevice(id);
                state->workerStartEvent = FastllmCudaEventCreate();
                state->workerEndEvent = FastllmCudaEventCreate();
                state->replayDoneEvent = FastllmCudaEventCreate();
                for (int step = 0; step < markovSteps; step++) {
                    state->markovLatentReadyEvents.push_back(
                        FastllmCudaEventCreate());
                    state->markovCandidateReadyEvents.push_back(
                        FastllmCudaEventCreate());
                }
                state->decodeMeta = decodeMeta.multiDeviceDatas.at(id);
                state->decodeMeta->dataType = DataType::INT32;
                state->decodeMeta->Resize({kDeepSeekV4CudaGraphMetaInts});
                state->decodeMeta->dataDevice = DataDevice::CUDA;
                state->decodeMeta->dataDeviceIds = {id};
                state->decodeMeta->Allocate(false);
                deviceIndex[id] = state.get();
                devices.push_back(std::move(state));
            }
            mainWindowKV.resize(layers);
            launchOrder = nextDevices;
        }
    };

    static std::shared_ptr<DeepSeekV4DsparkCudaGraphState>
    GetDeepSeekV4DsparkCudaGraphState(std::shared_ptr<void> &slot) {
        if (!slot) {
            slot = std::shared_ptr<void>(
                new DeepSeekV4DsparkCudaGraphState(), [](void *ptr) {
                    delete (DeepSeekV4DsparkCudaGraphState*)ptr;
                });
        }
        return std::shared_ptr<DeepSeekV4DsparkCudaGraphState>(
            slot, (DeepSeekV4DsparkCudaGraphState*)slot.get());
    }

    static std::vector<uint64_t> DeepSeekV4DsparkWindowSignature(
            const std::vector<Data> &windows) {
        std::vector<uint64_t> signature;
        for (const Data &window : windows) {
            signature.push_back((uint64_t)window.dataType);
            signature.push_back((uint64_t)window.dims.size());
            for (int dim : window.dims) {
                signature.push_back((uint64_t)(uint32_t)dim);
            }
            signature.push_back(
                (uint64_t)(uintptr_t)window.cudaData);
            signature.push_back(
                (uint64_t)window.multiDeviceDatas.size());
            for (const auto &local : window.multiDeviceDatas) {
                signature.push_back((uint64_t)(uint32_t)local.first);
                signature.push_back((uint64_t)(uintptr_t)
                    (local.second == nullptr ? nullptr :
                        local.second->cudaData));
            }
        }
        return signature;
    }

    static bool DeepSeekV4DecodeCudaGraphEnabled() {
        return GetFastllmEnv().cudaGraph;
    }

    static std::shared_ptr<DeepSeekV4CudaGraphState> GetDeepSeekV4CudaGraphState(
            std::shared_ptr<void> &slot) {
        if (!slot) {
            slot = std::shared_ptr<void>(new DeepSeekV4CudaGraphState(), [](void *ptr) {
                delete (DeepSeekV4CudaGraphState*)ptr;
            });
        }
        return std::shared_ptr<DeepSeekV4CudaGraphState>(
            slot, (DeepSeekV4CudaGraphState*)slot.get());
    }

    static int DeepSeekV4GraphInitialMaxTokens(int currentTokens) {
        int configured = 4096;
        while (configured < currentTokens) {
            configured *= 2;
        }
        return configured;
    }

    static bool DeepSeekV4GraphTensorMatches(
            const Data &data, DataType type, const std::vector<int> &dims,
            const std::vector<int> &devices) {
        if (data.dataType != type || data.dims != dims || devices.empty()) {
            return false;
        }
        if (devices.size() == 1 && !data.multiDeviceData) {
            return data.dataDevice == DataDevice::CUDA && data.cudaData != nullptr &&
                   GetPointerDeviceId(data.cudaData) == devices[0];
        }
        if (!data.multiDeviceData || !data.IsTensorParallelReplicated()) {
            return false;
        }
        for (int device : devices) {
            auto it = data.multiDeviceDatas.find(device);
            if (it == data.multiDeviceDatas.end() || it->second == nullptr ||
                it->second->dataType != type || it->second->dims != dims ||
                it->second->dataDevice != DataDevice::CUDA ||
                it->second->cudaData == nullptr ||
                GetPointerDeviceId(it->second->cudaData) != device) {
                return false;
            }
        }
        return true;
    }

    static bool DeepSeekV4AllocateGraphTensor(
            Data &data, DataType type, const std::vector<int> &dims,
            const std::vector<int> &devices, bool zero) {
        if (devices.empty()) {
            return false;
        }
        ResetData(data);
        data.dataType = type;
        data.Resize(dims);
        data.dataDevice = DataDevice::CUDA;
        data.dataDeviceIds = devices;
        if (devices.size() == 1) {
            FastllmCudaSetDevice(devices[0]);
            data.dataDeviceIds = {devices[0]};
            data.Allocate(zero);
            return data.cudaData != nullptr;
        }
        PrepareMultiCudaReplicatedData(data, devices, false);
        for (int device : devices) {
            Data *local = data.multiDeviceDatas.at(device);
            FastllmCudaSetDevice(device);
            local->dataType = type;
            local->Resize(dims);
            local->dataDevice = DataDevice::CUDA;
            local->dataDeviceIds = {device};
            local->Allocate(zero);
            if (local->cudaData == nullptr) {
                return false;
            }
        }
        data.dataDevice = DataDevice::CUDA;
        data.dataDeviceIds = devices;
        data.tpLayout = TP_LAYOUT_REPLICATED;
        data.tpAxis = -1;
        data.tpGlobalDims = dims;
        return true;
    }

    static bool DeepSeekV4HcMeanCuda(const Data &input, Data &output) {
        std::vector<int> devices = GetTensorCudaDevices(input);
        if (devices.empty()) {
            return false;
        }
        if (devices.size() == 1 && !input.multiDeviceData) {
            FastllmCudaSetDevice(devices[0]);
            return FastllmCudaDeepSeekV4HcMean(input, output);
        }
        if (!input.multiDeviceData || !input.IsTensorParallelReplicated()) {
            return false;
        }

        std::vector<int> outputDims = {
            input.dims[0], input.dims[1], input.dims[3]};
        if (!DeepSeekV4GraphTensorMatches(
                output, DataType::BFLOAT16, outputDims, devices) &&
            !DeepSeekV4AllocateGraphTensor(
                output, DataType::BFLOAT16, outputDims, devices, false)) {
            return false;
        }

        std::vector<char> ok(devices.size(), 0);
        RunDeepSeekV4MultiCuda(devices, [&](int rank, int device) {
            const Data *localInput = GetTensorCudaReplica(input, device);
            Data *localOutput = GetTensorCudaReplica(output, device);
            if (localInput != nullptr && localOutput != nullptr) {
                ok[rank] = FastllmCudaDeepSeekV4HcMean(
                    *localInput, *localOutput);
            }
        });
        return std::all_of(ok.begin(), ok.end(),
                           [](char state) { return state != 0; });
    }

    static bool DeepSeekV4HcPreNormMultiCuda(
            const Data &x, Data &hcFn, Data &hcScale, Data &hcBase,
            Data &normWeight, int hcMult, int sinkhornIters,
            float eps, float normEps, Data &normOutput, Data &post,
            Data &comb, const std::vector<int> &devices) {
        if (devices.size() <= 1) {
            return FastllmCudaDeepSeekV4HcPreNorm(
                x, hcFn, hcScale, hcBase, normWeight, hcMult,
                sinkhornIters, eps, normEps, normOutput, post, comb);
        }
        if (!x.multiDeviceData || !x.IsTensorParallelReplicated()) {
            return false;
        }

        PrepareMultiCudaReplicatedData(hcFn, devices, true);
        PrepareMultiCudaReplicatedData(hcScale, devices, true);
        PrepareMultiCudaReplicatedData(hcBase, devices, true);
        PrepareMultiCudaReplicatedData(normWeight, devices, true);
        std::vector<int> normDims = {x.dims[0], x.dims[1], x.dims[3]};
        std::vector<int> postDims = {x.dims[0], x.dims[1], hcMult};
        std::vector<int> combDims = {x.dims[0], x.dims[1], hcMult, hcMult};
        if ((!DeepSeekV4GraphTensorMatches(
                 normOutput, DataType::BFLOAT16, normDims, devices) &&
             !DeepSeekV4AllocateGraphTensor(
                 normOutput, DataType::BFLOAT16, normDims, devices, false)) ||
            (!DeepSeekV4GraphTensorMatches(
                 post, DataType::FLOAT32, postDims, devices) &&
             !DeepSeekV4AllocateGraphTensor(
                 post, DataType::FLOAT32, postDims, devices, false)) ||
            (!DeepSeekV4GraphTensorMatches(
                 comb, DataType::FLOAT32, combDims, devices) &&
             !DeepSeekV4AllocateGraphTensor(
                 comb, DataType::FLOAT32, combDims, devices, false))) {
            return false;
        }

        std::vector<char> ok(devices.size(), 0);
        RunDeepSeekV4MultiCuda(devices, [&](int rank, int device) {
            const Data *localX = GetTensorCudaReplica(x, device);
            Data *localFn = GetTensorCudaReplica(hcFn, device);
            Data *localScale = GetTensorCudaReplica(hcScale, device);
            Data *localBase = GetTensorCudaReplica(hcBase, device);
            Data *localNormWeight = GetTensorCudaReplica(normWeight, device);
            Data *localNormOutput = GetTensorCudaReplica(normOutput, device);
            Data *localPost = GetTensorCudaReplica(post, device);
            Data *localComb = GetTensorCudaReplica(comb, device);
            if (localX != nullptr && localFn != nullptr &&
                localScale != nullptr && localBase != nullptr &&
                localNormWeight != nullptr && localNormOutput != nullptr &&
                localPost != nullptr && localComb != nullptr) {
                ok[rank] = FastllmCudaDeepSeekV4HcPreNorm(
                    *localX, *localFn, *localScale, *localBase,
                    *localNormWeight, hcMult, sinkhornIters, eps, normEps,
                    *localNormOutput, *localPost, *localComb);
            }
        });
        return std::all_of(ok.begin(), ok.end(),
                           [](char state) { return state != 0; });
    }

    static bool DeepSeekV4HcPostPreNormMultiCuda(
            const Data &x, const Data &residual, const Data &previousPost,
            const Data &previousComb, Data &nextHcFn, Data &nextHcScale,
            Data &nextHcBase, Data &nextNormWeight, int hcMult,
            int sinkhornIters, float eps, float normEps,
            Data &residualOutput, Data &normOutput, Data &nextPost,
            Data &nextComb, const std::vector<int> &devices) {
        if (devices.size() <= 1) {
            return FastllmCudaDeepSeekV4HcPostPreNorm(
                x, residual, previousPost, previousComb, nextHcFn,
                nextHcScale, nextHcBase, nextNormWeight, hcMult,
                sinkhornIters, eps, normEps, residualOutput, normOutput,
                nextPost, nextComb);
        }
        if (!x.multiDeviceData || !x.IsTensorParallelReplicated() ||
            !residual.multiDeviceData ||
            !residual.IsTensorParallelReplicated() ||
            !previousPost.multiDeviceData ||
            !previousPost.IsTensorParallelReplicated() ||
            !previousComb.multiDeviceData ||
            !previousComb.IsTensorParallelReplicated()) {
            return false;
        }

        PrepareMultiCudaReplicatedData(nextHcFn, devices, true);
        PrepareMultiCudaReplicatedData(nextHcScale, devices, true);
        PrepareMultiCudaReplicatedData(nextHcBase, devices, true);
        PrepareMultiCudaReplicatedData(nextNormWeight, devices, true);
        std::vector<int> residualDims = {
            residual.dims[0], residual.dims[1], hcMult, residual.dims[3]};
        std::vector<int> normDims = {
            residual.dims[0], residual.dims[1], residual.dims[3]};
        std::vector<int> postDims = {
            residual.dims[0], residual.dims[1], hcMult};
        std::vector<int> combDims = {
            residual.dims[0], residual.dims[1], hcMult, hcMult};
        if ((!DeepSeekV4GraphTensorMatches(
                 residualOutput, DataType::BFLOAT16, residualDims, devices) &&
             !DeepSeekV4AllocateGraphTensor(
                 residualOutput, DataType::BFLOAT16, residualDims, devices,
                 false)) ||
            (!DeepSeekV4GraphTensorMatches(
                 normOutput, DataType::BFLOAT16, normDims, devices) &&
             !DeepSeekV4AllocateGraphTensor(
                 normOutput, DataType::BFLOAT16, normDims, devices, false)) ||
            (!DeepSeekV4GraphTensorMatches(
                 nextPost, DataType::FLOAT32, postDims, devices) &&
             !DeepSeekV4AllocateGraphTensor(
                 nextPost, DataType::FLOAT32, postDims, devices, false)) ||
            (!DeepSeekV4GraphTensorMatches(
                 nextComb, DataType::FLOAT32, combDims, devices) &&
             !DeepSeekV4AllocateGraphTensor(
                 nextComb, DataType::FLOAT32, combDims, devices, false))) {
            return false;
        }

        std::vector<char> ok(devices.size(), 0);
        RunDeepSeekV4MultiCuda(devices, [&](int rank, int device) {
            const Data *localX = GetTensorCudaReplica(x, device);
            const Data *localResidual = GetTensorCudaReplica(residual, device);
            const Data *localPreviousPost =
                GetTensorCudaReplica(previousPost, device);
            const Data *localPreviousComb =
                GetTensorCudaReplica(previousComb, device);
            Data *localFn = GetTensorCudaReplica(nextHcFn, device);
            Data *localScale = GetTensorCudaReplica(nextHcScale, device);
            Data *localBase = GetTensorCudaReplica(nextHcBase, device);
            Data *localNormWeight =
                GetTensorCudaReplica(nextNormWeight, device);
            Data *localResidualOutput =
                GetTensorCudaReplica(residualOutput, device);
            Data *localNormOutput = GetTensorCudaReplica(normOutput, device);
            Data *localNextPost = GetTensorCudaReplica(nextPost, device);
            Data *localNextComb = GetTensorCudaReplica(nextComb, device);
            if (localX != nullptr && localResidual != nullptr &&
                localPreviousPost != nullptr && localPreviousComb != nullptr &&
                localFn != nullptr && localScale != nullptr &&
                localBase != nullptr && localNormWeight != nullptr &&
                localResidualOutput != nullptr && localNormOutput != nullptr &&
                localNextPost != nullptr && localNextComb != nullptr) {
                ok[rank] = FastllmCudaDeepSeekV4HcPostPreNorm(
                    *localX, *localResidual, *localPreviousPost,
                    *localPreviousComb, *localFn, *localScale, *localBase,
                    *localNormWeight, hcMult, sinkhornIters, eps, normEps,
                    *localResidualOutput, *localNormOutput, *localNextPost,
                    *localNextComb);
            }
        });
        return std::all_of(ok.begin(), ok.end(),
                           [](char state) { return state != 0; });
    }

    static bool DeepSeekV4FusedQKVRopeCacheMultiCuda(
            Data &q, Data &kv, Data &kvNormWeight, Data &decodeMeta,
            int ropeDim, float ropeBase, int originalSeqLen,
            float ropeFactor, int betaFast, int betaSlow, float eps,
            int quantDim, int quantBlockSize, int windowSize,
            Data &windowKV, const std::vector<int> &devices) {
        if (devices.size() <= 1) {
            return FastllmCudaDeepSeekV4FusedQKVRopeCacheGraph(
                q, kv, kvNormWeight, (const int32_t*)decodeMeta.cudaData,
                ropeDim, ropeBase, originalSeqLen, ropeFactor,
                betaFast, betaSlow, eps, quantDim, quantBlockSize,
                windowSize, windowKV);
        }
        if (!q.multiDeviceData || !q.IsTensorParallelSharded() ||
            q.tpAxis != 2) {
            return false;
        }
        PrepareMultiCudaReplicatedData(kv, devices, true);
        PrepareMultiCudaReplicatedData(kvNormWeight, devices, true);
        PrepareMultiCudaReplicatedData(decodeMeta, devices, true);
        PrepareMultiCudaReplicatedData(windowKV, devices, true);

        std::vector<char> ok(devices.size(), 0);
        RunDeepSeekV4MultiCuda(devices, [&](int rank, int device) {
            Data *localQ = GetTensorCudaReplica(q, device);
            Data *localKV = GetTensorCudaReplica(kv, device);
            Data *localNormWeight = GetTensorCudaReplica(kvNormWeight, device);
            Data *localMeta = GetTensorCudaReplica(decodeMeta, device);
            Data *localWindow = GetTensorCudaReplica(windowKV, device);
            if (localQ != nullptr && localKV != nullptr &&
                localNormWeight != nullptr && localMeta != nullptr &&
                localWindow != nullptr) {
                ok[rank] = FastllmCudaDeepSeekV4FusedQKVRopeCacheGraph(
                    *localQ, *localKV, *localNormWeight,
                    (const int32_t*)localMeta->cudaData,
                    ropeDim, ropeBase, originalSeqLen, ropeFactor,
                    betaFast, betaSlow, eps, quantDim, quantBlockSize,
                    windowSize, *localWindow);
            }
        });
        return std::all_of(ok.begin(), ok.end(),
                           [](char state) { return state != 0; });
    }

    static bool DeepSeekV4PrepareGraphWeight(
            Data &output, const Data &weight, DataType outputType,
            const std::vector<int> &devices, bool &addressChanged) {
        if (DeepSeekV4GraphTensorMatches(output, outputType, weight.dims, devices)) {
            return true;
        }
        std::vector<float> values = ReadFloatData(weight);
        if (values.size() != weight.Count(0)) {
            return false;
        }
        Data cpu;
        WriteFloatData(values, weight.dims, cpu, outputType);
        ResetData(output);
        output.dataType = outputType;
        output.Resize(weight.dims);
        output.dataDevice = DataDevice::CUDA;
        output.dataDeviceIds = devices;
        if (devices.size() == 1) {
            output.CopyFrom(cpu);
            output.ToDevice(DataDevice::CUDA, std::vector<int>{devices[0]});
        } else {
            PrepareMultiCudaReplicatedData(output, devices, false);
            for (int device : devices) {
                Data *local = output.multiDeviceDatas.at(device);
                local->CopyFrom(cpu);
                local->ToDevice(DataDevice::CUDA, std::vector<int>{device});
            }
            output.dataDevice = DataDevice::CUDA;
            output.dataDeviceIds = devices;
            output.tpLayout = TP_LAYOUT_REPLICATED;
            output.tpAxis = -1;
            output.tpGlobalDims = output.dims;
        }
        addressChanged = true;
        return DeepSeekV4GraphTensorMatches(output, outputType, weight.dims, devices);
    }

    static bool DeepSeekV4EnsureGraphCompressedCapacity(
            Data &compressedKV, int bsz, int logicalBlocks, int headDim,
            int requiredCapacity, const std::vector<int> &devices,
            bool &addressChanged) {
        std::vector<int> logicalDims = {
            bsz, std::max(1, logicalBlocks), headDim
        };
        if (devices.size() == 1 && !compressedKV.multiDeviceData) {
            int device = devices[0];
            if (compressedKV.dataDevice != DataDevice::CUDA ||
                compressedKV.cudaData == nullptr ||
                GetPointerDeviceId(compressedKV.cudaData) != device) {
                if (HasTensorData(compressedKV)) {
                    compressedKV.SetKVCache();
                    compressedKV.ToDevice(DataDevice::CUDA,
                                          std::vector<int>{device});
                } else {
                    compressedKV.dataType = DataType::BFLOAT16;
                    compressedKV.Resize(logicalDims);
                    compressedKV.dataDevice = DataDevice::CUDA;
                    compressedKV.dataDeviceIds = {device};
                    compressedKV.Allocate(true);
                }
                addressChanged = true;
            }
            if (compressedKV.dims.size() != 3 ||
                compressedKV.dims[0] != bsz ||
                compressedKV.dims[2] != headDim) {
                return false;
            }
            int capacity = compressedKV.dims[1];
            if (compressedKV.expansionDims.size() >= 3) {
                capacity = std::max(capacity, compressedKV.expansionDims[1]);
            }
            if (capacity < requiredCapacity) {
                compressedKV.Expansion({bsz, requiredCapacity, headDim});
                addressChanged = true;
            }
            return compressedKV.cudaData != nullptr;
        }

        if (!compressedKV.multiDeviceData) {
            if (HasTensorData(compressedKV)) {
                PrepareMultiCudaReplicatedData(compressedKV, devices, true);
            } else if (!DeepSeekV4AllocateGraphTensor(
                           compressedKV, DataType::BFLOAT16, logicalDims,
                           devices, true)) {
                return false;
            }
            addressChanged = true;
        }
        if (!compressedKV.IsTensorParallelReplicated()) {
            return false;
        }
        Data *first = nullptr;
        for (int device : devices) {
            auto it = compressedKV.multiDeviceDatas.find(device);
            if (it == compressedKV.multiDeviceDatas.end() || it->second == nullptr) {
                return false;
            }
            Data *local = it->second;
            FastllmCudaSetDevice(device);
            if (local->dataDevice != DataDevice::CUDA || local->cudaData == nullptr) {
                if (HasTensorData(*local)) {
                    local->SetKVCache();
                    local->ToDevice(DataDevice::CUDA, std::vector<int>{device});
                } else {
                    local->dataType = DataType::BFLOAT16;
                    local->Resize(logicalDims);
                    local->dataDevice = DataDevice::CUDA;
                    local->dataDeviceIds = {device};
                    local->Allocate(true);
                }
                addressChanged = true;
            }
            if (local->dataType != DataType::BFLOAT16 || local->dims.size() != 3 ||
                local->dims[0] != bsz || local->dims[2] != headDim) {
                return false;
            }
            int capacity = local->dims[1];
            if (local->expansionDims.size() >= 3) {
                capacity = std::max(capacity, local->expansionDims[1]);
            }
            if (capacity < requiredCapacity) {
                local->Expansion({bsz, requiredCapacity, headDim});
                addressChanged = true;
            }
            first = first == nullptr ? local : first;
        }
        if (first == nullptr) {
            return false;
        }
        compressedKV.dataType = DataType::BFLOAT16;
        compressedKV.Resize(first->dims);
        compressedKV.dataDevice = DataDevice::CUDA;
        compressedKV.dataDeviceIds = devices;
        compressedKV.tpLayout = TP_LAYOUT_REPLICATED;
        compressedKV.tpAxis = -1;
        compressedKV.tpGlobalDims = compressedKV.dims;
        compressedKV.expansionDims = first->expansionDims;
        compressedKV.strides = first->strides;
        compressedKV.expansionSize = first->expansionSize;
        compressedKV.expansionBytes = first->expansionBytes;
        return true;
    }

    static bool DeepSeekV4PrepareFixedGraphLayerCache(
            DeepSeekV4DecodeLayerCache &cache, Data &apeWeight, Data &normWeight,
            Data *indexerApeWeight, Data *indexerNormWeight,
            int graphMaxTokens, int graphSequenceLen,
            const std::vector<int> &devices,
            bool &addressChanged) {
        addressChanged = false;
        if (devices.empty() || !cache.initialized || cache.bsz != 1 ||
            cache.headDim <= 0 || cache.windowSize <= 0 ||
            cache.windowKV.dataDevice != DataDevice::CUDA) {
            return false;
        }
        auto preparePackedSm120Caches = [&](int compressedCapacity) {
            if (!DeepSeekV4SparseMlaSm120Enabled()) {
                return true;
            }
            bool useSm120 = true;
            int originalDevice = FastllmCudaGetDevice();
            for (int device : devices) {
                FastllmCudaSetDevice(device);
                if (!FastllmCudaDeepSeekV4SparseMlaSm120Available()) {
                    useSm120 = false;
                    break;
                }
            }
            FastllmCudaSetDevice(originalDevice);
            if (!useSm120) {
                return true;
            }

            constexpr int pageSize = 64;
            constexpr int pageBytes = pageSize * 584;
            int windowPages = std::max(1, (graphMaxTokens + pageSize - 1) /
                                           pageSize);
            int compressedPages = std::max(
                1, (compressedCapacity + pageSize - 1) / pageSize);
            bool windowStale = !DeepSeekV4GraphTensorMatches(
                cache.cudaGraphPackedWindowKV, DataType::INT8,
                {windowPages, pageBytes}, devices);
            bool compressedStale = !DeepSeekV4GraphTensorMatches(
                cache.cudaGraphPackedCompressedKV, DataType::INT8,
                {compressedPages, pageBytes}, devices);
            if (windowStale && !DeepSeekV4AllocateGraphTensor(
                    cache.cudaGraphPackedWindowKV, DataType::INT8,
                    {windowPages, pageBytes}, devices, true)) {
                return false;
            }
            if (compressedStale && !DeepSeekV4AllocateGraphTensor(
                    cache.cudaGraphPackedCompressedKV, DataType::INT8,
                    {compressedPages, pageBytes}, devices, true)) {
                return false;
            }
            if (windowStale || compressedStale) {
                for (int device : devices) {
                    const Data *window = GetTensorCudaReplica(
                        cache.windowKV, device);
                    const Data *compressed = GetTensorCudaReplica(
                        cache.compressedKV, device);
                    Data *packedWindow = GetTensorCudaReplica(
                        cache.cudaGraphPackedWindowKV, device);
                    Data *packedCompressed = GetTensorCudaReplica(
                        cache.cudaGraphPackedCompressedKV, device);
                    FastllmCudaSetDevice(device);
                    if (window == nullptr || compressed == nullptr ||
                        packedWindow == nullptr || packedCompressed == nullptr ||
                        !FastllmCudaDeepSeekV4PrepareSparseMlaSm120Cache(
                            *window, cache.totalLen, cache.windowSize,
                            *compressed, cache.compressedBlocks,
                            *packedWindow, *packedCompressed)) {
                        FastllmCudaSetDevice(originalDevice);
                        return false;
                    }
                }
                addressChanged = true;
            }
            cache.cudaGraphPackedWindowCapacity = windowPages * pageSize;
            cache.cudaGraphPackedCompressedCapacity =
                compressedPages * pageSize;
            FastllmCudaSetDevice(originalDevice);
            return true;
        };
        FastllmCudaSetDevice(devices[0]);
        if (cache.compressRatio <= 0) {
            bool compressedChanged = false;
            if (!DeepSeekV4EnsureGraphCompressedCapacity(
                    cache.compressedKV, 1, 1, cache.headDim, 1,
                    devices, compressedChanged)) {
                return false;
            }
            if (compressedChanged) {
                addressChanged = true;
            }
            cache.cudaGraphCompressedCapacity = 1;
            cache.cudaGraphCacheReady = true;
            return preparePackedSm120Caches(1);
        }

        int compressRatio = cache.compressRatio;
        int requiredCapacity = std::max(
            1, (graphMaxTokens + compressRatio - 1) / compressRatio);
        requiredCapacity = std::min(
            requiredCapacity, 320 * 1024 - cache.windowSize);
        if (requiredCapacity <= cache.compressedBlocks) {
            requiredCapacity = cache.compressedBlocks + 1;
        }

        // Prefill can leave the compressor tail as a single CUDA tensor even
        // when attention itself is tensor-parallel.  Promote that immutable
        // tail to one replica per rank before creating the fixed graph rings;
        // all TP replicas consume the same compressed history.
        if (devices.size() > 1 &&
            HasTensorData(cache.compressorKVRaw) &&
            HasTensorData(cache.compressorScoreRaw)) {
            if (!cache.compressorKVRaw.multiDeviceData) {
                PrepareMultiCudaReplicatedData(cache.compressorKVRaw,
                                               devices, true);
                addressChanged = true;
            }
            if (!cache.compressorScoreRaw.multiDeviceData) {
                PrepareMultiCudaReplicatedData(cache.compressorScoreRaw,
                                               devices, true);
                addressChanged = true;
            }
        }
        if (compressRatio == 4 && devices.size() > 1 &&
            HasTensorData(cache.indexerCompressorKVRaw) &&
            HasTensorData(cache.indexerCompressorScoreRaw)) {
            if (!cache.indexerCompressorKVRaw.multiDeviceData) {
                PrepareMultiCudaReplicatedData(
                    cache.indexerCompressorKVRaw, devices, true);
                addressChanged = true;
            }
            if (!cache.indexerCompressorScoreRaw.multiDeviceData) {
                PrepareMultiCudaReplicatedData(
                    cache.indexerCompressorScoreRaw, devices, true);
                addressChanged = true;
            }
        }

        // A replicated raw tensor can be published one worker stream at a time.
        // Preflight every local pointer before resizing any persistent cache so a
        // failed first attempt leaves the eager path untouched.
        if (HasTensorData(cache.compressorKVRaw) &&
            HasTensorData(cache.compressorScoreRaw)) {
            for (int device : devices) {
                if (GetTensorCudaReplica(cache.compressorKVRaw, device) == nullptr ||
                    GetTensorCudaReplica(cache.compressorScoreRaw, device) == nullptr) {
                    return false;
                }
            }
        }
        if (compressRatio == 4 &&
            HasTensorData(cache.indexerCompressorKVRaw) &&
            HasTensorData(cache.indexerCompressorScoreRaw)) {
            for (int device : devices) {
                if (GetTensorCudaReplica(
                        cache.indexerCompressorKVRaw, device) == nullptr ||
                    GetTensorCudaReplica(
                        cache.indexerCompressorScoreRaw, device) == nullptr) {
                    return false;
                }
            }
        }

        bool compressedChanged = false;
        int logicalBlocks = cache.compressedKV.dims.size() >= 3 ?
                            cache.compressedKV.dims[1] :
                            std::max(1, cache.compressedBlocks);
        if (!DeepSeekV4EnsureGraphCompressedCapacity(
                cache.compressedKV, cache.bsz, logicalBlocks, cache.headDim,
                requiredCapacity, devices, compressedChanged)) {
            return false;
        }
        if (compressedChanged) {
            addressChanged = true;
        }
        cache.cudaGraphCompressedCapacity = requiredCapacity;

        if (compressRatio == 4) {
            if (indexerApeWeight == nullptr || indexerNormWeight == nullptr ||
                !HasTensorData(*indexerApeWeight) ||
                !HasTensorData(*indexerNormWeight)) {
                return false;
            }
            bool indexerCompressedChanged = false;
            int indexerLogicalBlocks =
                cache.indexerCompressedKV.dims.size() >= 3 ?
                cache.indexerCompressedKV.dims[1] :
                std::max(1, cache.indexerCompressedBlocks);
            if (!DeepSeekV4EnsureGraphCompressedCapacity(
                    cache.indexerCompressedKV, cache.bsz,
                    indexerLogicalBlocks, 128, requiredCapacity, devices,
                    indexerCompressedChanged)) {
                return false;
            }
            if (indexerCompressedChanged) {
                addressChanged = true;
            }
            cache.cudaGraphIndexerCompressedCapacity = requiredCapacity;
        }

        int wideDim = (compressRatio == 4 ? 2 : 1) * cache.headDim;
        // Store the complete fixed-shape verification chunk before building
        // any newly completed compressed block. Preserve the preceding raw
        // block(s) as well, otherwise later rows can wrap around and overwrite
        // inputs needed by an earlier row in the same graph replay.
        int rawHistory = compressRatio == 4 ?
                         2 * compressRatio : compressRatio;
        int rawCapacity = rawHistory + std::max(0, graphSequenceLen - 1);
        // ComputeCompressorRaw explicitly promotes both projections to FP32.
        // A ratio-128 prefill can consume the complete raw tail, leaving no
        // compressorKVRaw tensor from which to infer the graph ring type.  The
        // old BF16 default then made every ratio-128 decode reject the fixed
        // ring even though the live compressor output is FP32.
        DataType ringType = HasTensorData(cache.compressorKVRaw) ?
                            cache.compressorKVRaw.dataType : DataType::FLOAT32;
        bool ringStale = !cache.cudaGraphCacheReady ||
                         cache.cudaGraphRawCapacity != rawCapacity ||
                         !DeepSeekV4GraphTensorMatches(
                             cache.cudaGraphCompressorKVRing, ringType,
                             {1, rawCapacity, wideDim}, devices) ||
                         !DeepSeekV4GraphTensorMatches(
                             cache.cudaGraphCompressorScoreRing, ringType,
                             {1, rawCapacity, wideDim}, devices);
        if (ringStale) {
            if (!DeepSeekV4AllocateGraphTensor(
                    cache.cudaGraphCompressorKVRing, ringType,
                    {1, rawCapacity, wideDim}, devices, true) ||
                !DeepSeekV4AllocateGraphTensor(
                    cache.cudaGraphCompressorScoreRing, ringType,
                    {1, rawCapacity, wideDim}, devices, true)) {
                return false;
            }
            if (HasTensorData(cache.compressorKVRaw) &&
                HasTensorData(cache.compressorScoreRaw)) {
                for (int device : devices) {
                    const Data *rawKV = GetTensorCudaReplica(
                        cache.compressorKVRaw, device);
                    const Data *rawScore = GetTensorCudaReplica(
                        cache.compressorScoreRaw, device);
                    Data *kvRing = GetTensorCudaReplica(
                        cache.cudaGraphCompressorKVRing, device);
                    Data *scoreRing = GetTensorCudaReplica(
                        cache.cudaGraphCompressorScoreRing, device);
                    FastllmCudaSetDevice(device);
                    if (rawKV == nullptr || rawScore == nullptr ||
                        kvRing == nullptr || scoreRing == nullptr ||
                        !FastllmCudaDeepSeekV4InitGraphRawRing(
                            *rawKV, cache.compressorRawTokenBase, *kvRing) ||
                        !FastllmCudaDeepSeekV4InitGraphRawRing(
                            *rawScore, cache.compressorRawTokenBase, *scoreRing)) {
                        return false;
                    }
                }
            }
            cache.cudaGraphRawCapacity = rawCapacity;
            addressChanged = true;
        }

        if (compressRatio == 4) {
            constexpr int indexerWideDim = 256;
            DataType indexerRingType =
                HasTensorData(cache.indexerCompressorKVRaw) ?
                cache.indexerCompressorKVRaw.dataType : DataType::FLOAT32;
            bool indexerRingStale =
                cache.cudaGraphIndexerRawCapacity != rawCapacity ||
                !DeepSeekV4GraphTensorMatches(
                    cache.cudaGraphIndexerCompressorKVRing,
                    indexerRingType,
                    {1, rawCapacity, indexerWideDim}, devices) ||
                !DeepSeekV4GraphTensorMatches(
                    cache.cudaGraphIndexerCompressorScoreRing,
                    indexerRingType,
                    {1, rawCapacity, indexerWideDim}, devices);
            if (indexerRingStale) {
                if (!DeepSeekV4AllocateGraphTensor(
                        cache.cudaGraphIndexerCompressorKVRing,
                        indexerRingType,
                        {1, rawCapacity, indexerWideDim}, devices, true) ||
                    !DeepSeekV4AllocateGraphTensor(
                        cache.cudaGraphIndexerCompressorScoreRing,
                        indexerRingType,
                        {1, rawCapacity, indexerWideDim}, devices, true)) {
                    return false;
                }
                if (HasTensorData(cache.indexerCompressorKVRaw) &&
                    HasTensorData(cache.indexerCompressorScoreRaw)) {
                    for (int device : devices) {
                        const Data *rawKV = GetTensorCudaReplica(
                            cache.indexerCompressorKVRaw, device);
                        const Data *rawScore = GetTensorCudaReplica(
                            cache.indexerCompressorScoreRaw, device);
                        Data *kvRing = GetTensorCudaReplica(
                            cache.cudaGraphIndexerCompressorKVRing, device);
                        Data *scoreRing = GetTensorCudaReplica(
                            cache.cudaGraphIndexerCompressorScoreRing, device);
                        FastllmCudaSetDevice(device);
                        if (rawKV == nullptr || rawScore == nullptr ||
                            kvRing == nullptr || scoreRing == nullptr ||
                            !FastllmCudaDeepSeekV4InitGraphRawRing(
                                *rawKV,
                                cache.indexerCompressorRawTokenBase,
                                *kvRing) ||
                            !FastllmCudaDeepSeekV4InitGraphRawRing(
                                *rawScore,
                                cache.indexerCompressorRawTokenBase,
                                *scoreRing)) {
                            return false;
                        }
                    }
                }
                cache.cudaGraphIndexerRawCapacity = rawCapacity;
                addressChanged = true;
            }
        }

        bool apeChanged = false;
        if (!DeepSeekV4PrepareGraphWeight(
                cache.cudaGraphApe, apeWeight, DataType::FLOAT32,
                devices, apeChanged)) {
            return false;
        }
        if (apeChanged) {
            addressChanged = true;
        }
        DataType normType = normWeight.dataType == DataType::BFLOAT16 ||
                            normWeight.dataType == DataType::FLOAT16 ||
                            normWeight.dataType == DataType::FLOAT32 ?
                            normWeight.dataType : DataType::FLOAT32;
        bool normChanged = false;
        if (!DeepSeekV4PrepareGraphWeight(
                cache.cudaGraphNormWeight, normWeight, normType,
                devices, normChanged)) {
            return false;
        }
        if (normChanged) {
            addressChanged = true;
        }
        DataType indexerNormType = DataType::FLOAT32;
        if (compressRatio == 4) {
            bool indexerApeChanged = false;
            if (!DeepSeekV4PrepareGraphWeight(
                    cache.cudaGraphIndexerApe, *indexerApeWeight,
                    DataType::FLOAT32, devices, indexerApeChanged)) {
                return false;
            }
            indexerNormType =
                indexerNormWeight->dataType == DataType::BFLOAT16 ||
                indexerNormWeight->dataType == DataType::FLOAT16 ||
                indexerNormWeight->dataType == DataType::FLOAT32 ?
                indexerNormWeight->dataType : DataType::FLOAT32;
            bool indexerNormChanged = false;
            if (!DeepSeekV4PrepareGraphWeight(
                    cache.cudaGraphIndexerNormWeight, *indexerNormWeight,
                    indexerNormType, devices, indexerNormChanged)) {
                return false;
            }
            bool indicesStale = !DeepSeekV4GraphTensorMatches(
                cache.cudaGraphIndexerIndices, DataType::INT32,
                {graphSequenceLen, 512}, devices);
            bool lengthsStale = !DeepSeekV4GraphTensorMatches(
                cache.cudaGraphIndexerLengths, DataType::INT32,
                {graphSequenceLen}, devices);
            if (indicesStale && !DeepSeekV4AllocateGraphTensor(
                    cache.cudaGraphIndexerIndices, DataType::INT32,
                    {graphSequenceLen, 512}, devices, false)) {
                return false;
            }
            if (lengthsStale && !DeepSeekV4AllocateGraphTensor(
                    cache.cudaGraphIndexerLengths, DataType::INT32,
                    {graphSequenceLen}, devices, false)) {
                return false;
            }
            if (indexerApeChanged || indexerNormChanged || indicesStale ||
                lengthsStale) {
                addressChanged = true;
            }
        }
        cache.cudaGraphCacheReady = true;
        bool baseReady = DeepSeekV4GraphTensorMatches(
                   cache.cudaGraphCompressorKVRing, ringType,
                   {1, rawCapacity, wideDim}, devices) &&
               DeepSeekV4GraphTensorMatches(
                   cache.cudaGraphCompressorScoreRing, ringType,
                   {1, rawCapacity, wideDim}, devices) &&
               DeepSeekV4GraphTensorMatches(
                   cache.cudaGraphApe, DataType::FLOAT32,
                   apeWeight.dims, devices) &&
               DeepSeekV4GraphTensorMatches(
                   cache.cudaGraphNormWeight, normType,
                   normWeight.dims, devices);
        bool indexerReady = compressRatio != 4 ||
            (DeepSeekV4GraphTensorMatches(
                 cache.cudaGraphIndexerCompressorKVRing,
                 HasTensorData(cache.indexerCompressorKVRaw) ?
                     cache.indexerCompressorKVRaw.dataType : DataType::FLOAT32,
                 {1, rawCapacity, 256}, devices) &&
             DeepSeekV4GraphTensorMatches(
                 cache.cudaGraphIndexerCompressorScoreRing,
                 HasTensorData(cache.indexerCompressorKVRaw) ?
                     cache.indexerCompressorKVRaw.dataType : DataType::FLOAT32,
                 {1, rawCapacity, 256}, devices) &&
             DeepSeekV4GraphTensorMatches(
                 cache.cudaGraphIndexerApe, DataType::FLOAT32,
                 indexerApeWeight->dims, devices) &&
             DeepSeekV4GraphTensorMatches(
                 cache.cudaGraphIndexerNormWeight, indexerNormType,
                 indexerNormWeight->dims, devices) &&
             DeepSeekV4GraphTensorMatches(
                 cache.cudaGraphIndexerIndices, DataType::INT32,
                 {graphSequenceLen, 512}, devices) &&
             DeepSeekV4GraphTensorMatches(
                 cache.cudaGraphIndexerLengths, DataType::INT32,
                 {graphSequenceLen}, devices));
        return baseReady && indexerReady &&
               preparePackedSm120Caches(cache.cudaGraphCompressedCapacity);
    }
#endif

#ifndef USE_CUDA
    static bool DeepSeekV4DecodeCudaGraphEnabled() {
        return false;
    }
#endif

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

    static bool DeepSeekV4PrefixCacheEnabledByEnv() {
        const char *value = std::getenv("FASTLLM_PREFIX_CACHE");
        if (value == nullptr || value[0] == 0) {
            return true;
        }
        std::string normalized(value);
        std::transform(normalized.begin(), normalized.end(),
                       normalized.begin(), [](unsigned char c) {
                           return (char)std::tolower(c);
                       });
        return normalized != "0" && normalized != "false" &&
               normalized != "off" && normalized != "no";
    }

    static bool DeepSeekV4PrefixCacheDisabled() {
        // Match Qwen3.5 prefix-cache semantics: enabled by default and
        // explicitly disabled through the common flag.  Restored V4 caches
        // acquire a fresh request-local graph, so their old addresses never
        // leak into CUDA Graph replay.
        return EnvFlagEnabled("FASTLLM_DSV4_DISABLE_PREFIX_CACHE") ||
               !DeepSeekV4PrefixCacheEnabledByEnv();
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
        indexerCompressorWideDim = other.indexerCompressorWideDim;
        indexerCompressorRawTokenBase = other.indexerCompressorRawTokenBase;
        indexerCompressedBlocks = other.indexerCompressedBlocks;
        CopyTensorData(indexerCompressorKVRaw,
                       other.indexerCompressorKVRaw);
        CopyTensorData(indexerCompressorScoreRaw,
                       other.indexerCompressorScoreRaw);
        CopyTensorData(indexerCompressedKV, other.indexerCompressedKV);
        cudaGraphCacheReady = false;
        cudaGraphRawCapacity = 0;
        cudaGraphCompressedCapacity = 0;
        cudaGraphIndexerRawCapacity = 0;
        cudaGraphIndexerCompressedCapacity = 0;
        cudaGraphPackedWindowCapacity = 0;
        cudaGraphPackedCompressedCapacity = 0;
        ResetData(cudaGraphCompressorKVRing);
        ResetData(cudaGraphCompressorScoreRing);
        ResetData(cudaGraphApe);
        ResetData(cudaGraphNormWeight);
        ResetData(cudaGraphIndexerCompressorKVRing);
        ResetData(cudaGraphIndexerCompressorScoreRing);
        ResetData(cudaGraphIndexerApe);
        ResetData(cudaGraphIndexerNormWeight);
        ResetData(cudaGraphIndexerFp8KV);
        ResetData(cudaGraphIndexerFp8Scale);
        ResetData(cudaGraphIndexerIndices);
        ResetData(cudaGraphIndexerLengths);
        ResetData(cudaGraphPackedWindowKV);
        ResetData(cudaGraphPackedCompressedKV);
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
        CopyHistoryTensorData(windowKV, other.windowKV);
        CopyHistoryTensorData(compressorKVRaw, other.compressorKVRaw);
        CopyHistoryTensorData(compressorScoreRaw, other.compressorScoreRaw);
        CopyHistoryTensorData(compressedKV, other.compressedKV);
        CopyHistoryTensorData(compressorTailKV, other.compressorTailKV);
        CopyHistoryTensorData(compressorTailScore, other.compressorTailScore);
        indexerCompressorWideDim = other.indexerCompressorWideDim;
        indexerCompressorRawTokenBase = other.indexerCompressorRawTokenBase;
        indexerCompressedBlocks = other.indexerCompressedBlocks;
        CopyHistoryTensorData(indexerCompressorKVRaw,
                              other.indexerCompressorKVRaw);
        CopyHistoryTensorData(indexerCompressorScoreRaw,
                              other.indexerCompressorScoreRaw);
        CopyHistoryTensorData(indexerCompressedKV,
                              other.indexerCompressedKV);
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
        int commonMax = EnvInt("FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_RECORDS",
                               this->maxRecordNum);
        int envMax = EnvInt("FASTLLM_DSV4_PREFIX_CACHE_MAX_RECORDS",
                            commonMax);
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

    bool DeepSeekV4HistoryCacheManager::Get(
            const std::vector<int> &inputToken,
            DeepSeekV4HistoryCacheMemory &memory,
            int &hitLen, bool requireDspark) {
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
            if (requireDspark && !memIt->second.dsparkValid) {
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
            if (requireDspark && !it.second.dsparkValid) {
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
            CopyHistoryTensorData(dst.windowKV, src.windowKV);
            dst.compressedBlocks = src.compressedBlocks;
            dst.compressedTokenBase = src.compressedTokenBase;
            dst.rawTailStartPos = src.rawTailStartPos;
            dst.compressorRawTokenBase = src.compressorRawTokenBase;
            dst.indexerCompressorWideDim = src.indexerCompressorWideDim;
            dst.indexerCompressorRawTokenBase =
                src.indexerCompressorRawTokenBase;
            dst.indexerCompressedBlocks = src.indexerCompressedBlocks;
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

            ResetData(dst.indexerCompressedKV);
            if (src.indexerCompressedBlocks > 0 &&
                HasCompressedKVData(src.indexerCompressedKV)) {
                CopyTensorData(dst.indexerCompressedKV,
                               src.indexerCompressedKV);
#ifdef USE_CUDA
                if (DeepSeekV4PreferCuda()) {
                    dst.indexerCompressedKV.SetKVCache();
                    dst.indexerCompressedKV.ToDevice(DataDevice::CUDA);
                }
#endif
            }
            CopyTensorData(dst.indexerCompressorKVRaw,
                           src.indexerCompressorKVRaw);
            CopyTensorData(dst.indexerCompressorScoreRaw,
                           src.indexerCompressorScoreRaw);

#ifdef USE_CUDA
            if (DeepSeekV4PreferCuda() && HasTensorData(dst.windowKV) && dst.bsz > 0 &&
                dst.windowSize > 0 && dst.headDim > 0) {
                dst.windowKV.SetKVCache();
                dst.windowKV.ToDevice(DataDevice::CUDA);
            }
#endif
        }
        if (this->dsparkEnabled) {
            if (!memory.dsparkValid ||
                memory.dsparkCommittedTokens != memory.tokens ||
                (int)memory.dsparkHistoryTokens.size() < memory.tokens ||
                (int)memory.dsparkMainWindowKV.size() != this->dsparkLayers) {
                return false;
            }
            auto dspark = std::make_shared<DeepSeekV4DsparkContext>();
            dspark->initialized = true;
            dspark->committedTokens = memory.dsparkCommittedTokens;
            dspark->historyTokens.assign(
                memory.dsparkHistoryTokens.begin(),
                memory.dsparkHistoryTokens.begin() + memory.tokens);
            dspark->mainWindowKV.resize(this->dsparkLayers);
            for (int stage = 0; stage < this->dsparkLayers; ++stage) {
                const Data &src = memory.dsparkMainWindowKV[stage];
                Data &dst = dspark->mainWindowKV[stage];
                if (!HasTensorData(src) || src.dims.size() != 3 ||
                    src.dims[0] != 1 || src.dims[1] > this->window_size ||
                    src.dims[2] != this->head_dim_full) {
                    return false;
                }
                CopyTensorData(dst, src);
#ifdef USE_CUDA
                if (DeepSeekV4PreferCuda()) {
                    std::vector<int> devices;
                    std::map<int, int> ratios;
                    FastllmGetMulticudaDeviceAndRatio(
                        devices, ratios, true);
                    if (devices.empty()) {
                        int rootDevice = GetTensorCudaDevice(
                            this->weight["embed.weight"]);
                        if (rootDevice >= 0) {
                            devices.push_back(rootDevice);
                        }
                    }
                    if (!devices.empty()) {
                        dst.lockInCPU = false;
                        PrepareMultiCudaReplicatedData(dst, devices, true);
                        PublishReplicatedCudaRootMetadata(
                            dst, devices, true);
                    }
                }
#endif
            }
            state.dspark = std::move(dspark);
        } else {
            state.dspark.reset();
        }
        state.historyTokens = memory.inputToken;
        state.restoredHistoryCache = true;
        if (DeepSeekV4PrefixCacheDebugEnabled()) {
            printf("[fastllm-dsv4-prefix-cache] restore hit_len=%d blocks=%d layers=%d dspark=%d\n",
                   memory.tokens, memory.blockCount,
                   (int)memory.layers.size(), memory.dsparkValid ? 1 : 0);
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
        RecordHistorySnapshot(tokens, totalLen, this->decodeLayerCaches,
                              nullptr);
    }

    void DeepSeekV4Model::RecordHistorySnapshot(const std::vector<int> &tokens,
                                                int totalLen,
                                                const std::vector<DeepSeekV4DecodeLayerCache> &decodeCaches,
                                                const DeepSeekV4DsparkContext *dsparkContext) {
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
            dst.indexerCompressorWideDim = src.indexerCompressorWideDim;
            CopyTensorData(dst.windowKV, src.windowKV);
            dst.compressedBlocks = src.compressedBlocks;
            dst.compressedTokenBase = src.compressedBlocks * std::max(1, src.compressRatio);
            dst.compressorRawTokenBase = src.compressorRawTokenBase;
            dst.indexerCompressorRawTokenBase =
                src.indexerCompressorRawTokenBase;
            dst.indexerCompressedBlocks = src.indexerCompressedBlocks;
            ResetData(dst.compressedKV);
            if (src.compressedBlocks > 0 && HasCompressedKVData(src.compressedKV)) {
                CopyHistoryTensorData(dst.compressedKV, src.compressedKV);
            }
            if (HasCompressedKVData(dst.compressedKV)) {
                dst.compressedKV.ToDevice(DataDevice::CPU);
                dst.compressedKV.lockInCPU = true;
            }
            ResetData(dst.indexerCompressedKV);
            if (src.indexerCompressedBlocks > 0 &&
                HasCompressedKVData(src.indexerCompressedKV)) {
                CopyHistoryTensorData(dst.indexerCompressedKV,
                                      src.indexerCompressedKV);
                dst.indexerCompressedKV.ToDevice(DataDevice::CPU);
                dst.indexerCompressedKV.lockInCPU = true;
            }
            if (storeFullRaw || src.compressRatio == 4) {
                CopyHistoryTensorData(dst.indexerCompressorKVRaw,
                                      src.indexerCompressorKVRaw);
                CopyHistoryTensorData(dst.indexerCompressorScoreRaw,
                                      src.indexerCompressorScoreRaw);
            }

            if (src.compressRatio > 0 && src.compressorWideDim > 0) {
                int tailTokens = src.compressRatio == 4 ? 8 : (src.compressRatio == 128 ? 128 : src.compressRatio);
                tailTokens = std::min(tailTokens, src.totalLen);
                if (storeFullRaw) {
                    CopyHistoryTensorData(dst.compressorKVRaw, src.compressorKVRaw);
                    CopyHistoryTensorData(dst.compressorScoreRaw, src.compressorScoreRaw);
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
                        CopyHistoryTensorData(dst.compressorTailKV, tailKV);
                        CopyHistoryTensorData(dst.compressorTailScore, tailScore);
                    } else {
                        ResetData(dst.compressorTailKV);
                        ResetData(dst.compressorTailScore);
                    }
                }
            }
        }
        if (this->dsparkEnabled && dsparkContext != nullptr &&
            dsparkContext->initialized &&
            dsparkContext->committedTokens == totalLen &&
            (int)dsparkContext->historyTokens.size() >= totalLen &&
            (int)dsparkContext->mainWindowKV.size() == this->dsparkLayers) {
            memory.dsparkValid = true;
            memory.dsparkCommittedTokens = totalLen;
            memory.dsparkHistoryTokens.assign(
                dsparkContext->historyTokens.begin(),
                dsparkContext->historyTokens.begin() + totalLen);
            memory.dsparkMainWindowKV.resize(this->dsparkLayers);
            for (int stage = 0; stage < this->dsparkLayers; ++stage) {
                const Data &src = dsparkContext->mainWindowKV[stage];
                if (!HasTensorData(src) || src.dims.size() != 3 ||
                    src.dims[0] != 1 || src.dims[1] > this->window_size ||
                    src.dims[2] != this->head_dim_full) {
                    memory.dsparkValid = false;
                    memory.dsparkMainWindowKV.clear();
                    break;
                }
                CopyHistoryTensorData(
                    memory.dsparkMainWindowKV[stage], src);
            }
        }
        this->deepseekV4HistoryCacheManager.Record(memory);
        if (DeepSeekV4PrefixCacheDebugEnabled()) {
            printf("[fastllm-dsv4-prefix-cache] record tokens=%d blocks=%d layers=%d residual_only=%d dspark=%d\n",
                   totalLen, memory.blockCount, (int)memory.layers.size(),
                   storeFullRaw ? 0 : 1, memory.dsparkValid ? 1 : 0);
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
                printf("[fastllm-dsv4-prefix-cache] disabled by environment input_tokens=%d\n",
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
        if (!this->deepseekV4HistoryCacheManager.Get(
                inputTokens, memory, hitLen, this->dsparkEnabled) ||
            hitLen <= 0) {
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

    std::shared_ptr<DeepSeekV4RequestState> DeepSeekV4Model::GetRequestStateByFirstKey(const Data *firstPastKey) {
        if (firstPastKey == nullptr) {
            return nullptr;
        }
        const void *key = (const void*)firstPastKey;
        std::lock_guard<std::mutex> guard(this->requestStateMutex);
        auto it = this->requestStatesByFirstKey.find(key);
        if (it == this->requestStatesByFirstKey.end()) {
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
        std::shared_ptr<DeepSeekV4RequestState> state;
        auto existing = this->requestStates.find(key);
        if (existing != this->requestStates.end()) {
            // Forward() is also used by warmup and a few synchronous callers
            // before a ResponseContext is published.  DSpark lazily creates
            // the state in that case; keep it when the scheduler subsequently
            // attaches the real response context.
            state = existing->second;
        } else if (this->pendingRequestState) {
            state = this->pendingRequestState;
            this->requestStates[key] = state;
            this->pendingRequestState.reset();
        } else {
            state = std::make_shared<DeepSeekV4RequestState>();
            this->requestStates[key] = state;
        }
        if (!context->pastKeyValues.empty()) {
            this->requestStatesByFirstKey[(const void*)&context->pastKeyValues[0].first] = state;
        }
    }

    void DeepSeekV4Model::OnResponseContextRemoved(ResponseContext *context) {
        if (context == nullptr) {
            return;
        }
        const void *key = (const void*)&context->pastKeyValues;
        std::lock_guard<std::mutex> guard(this->requestStateMutex);
        this->requestStates.erase(key);
        if (!context->pastKeyValues.empty()) {
            this->requestStatesByFirstKey.erase((const void*)&context->pastKeyValues[0].first);
        }
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
#ifdef USE_CUDA
            if (DeepSeekV4PreferCuda()) {
                SynchronizeDeepSeekV4TensorParallelDevices(this->deviceMap);
            }
#endif
            this->RecordHistorySnapshot(
                context->allTokens, totalLen, state->decodeLayerCaches,
                state->dspark.get());
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
        if (fastllm::GetMaxTokens() > 0) {
            maxTotalLens = fastllm::GetMaxTokens();
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
            printf("Fastllm Prefix Cache: %s (history=%s, CUDA graph=%s, DSpark state=%s).\n",
                   (!model->saveHistoryChat || DeepSeekV4PrefixCacheDisabled()) ?
                       "disabled" : "enabled",
                   model->saveHistoryChat ? "on" : "off",
                   DeepSeekV4DecodeCudaGraphEnabled() ? "on" : "off",
                   model->dsparkEnabled ? "included" : "unused");
        }

        auto lastRecordTime = std::chrono::system_clock::now();
        long long genTokens = 0;
        while (true) {
            if (model->isFree) {
                break;
            }

            std::vector<Data*> attentionMasks;
            std::vector<Data*> positionIds;
            std::vector<std::pair<Data*, Data*> > pastKeyValues;
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
                // A newly restored prefix has model cache state but its prompt
                // suffix has not been scheduled yet (preTokens == 0).  Counting
                // it as an active request makes max_batch=1 reject that same
                // request forever.  It becomes active only after the scheduler
                // has actually submitted its first prompt/suffix chunk.
                if (it.second->preTokens > 0) {
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
                        // Restored pages are not part of currentTokens, but
                        // they still consume the shared cache token budget.
                        lenSum += ctx->cacheLen +
                                  ctx->currentTokens.size();
                        currentActivate++;
                    }

                    generationConfigs.push_back(ctx->generationConfig);
                    model->PrepareToolCallConstraint(ctx, generationConfigs.back());
                    if (ctx->generationConfig.output_logits) {
                        ctx->resultLogits.push(new std::vector<float>());
                        logits.push_back(ctx->resultLogits.back());
                    } else {
                        logits.push_back(nullptr);
                    }

                    tokensManager.units.push_back(ctx->tokens);
                    handles.push_back(it.first);
                    for (int layer = 0; layer < model->block_cnt; layer++) {
                        pastKeyValues.push_back(std::make_pair(&ctx->pastKeyValues[layer].first,
                                                              &ctx->pastKeyValues[layer].second));
                    }

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
                bool allDecodeTokens = true;
                for (int seqLen : seqLens) {
                    allDecodeTokens &= (seqLen == 1);
                }
#ifdef USE_CUDA
                // DSpark decode has fixed graph shapes and request-persistent
                // workspaces. Purging its idle big-buffer pool here performs
                // real cudaFree calls (and their implicit device barriers),
                // only to allocate the same blocks again in the next
                // speculative round. Keep the established reclamation policy
                // for prefill and non-DSpark execution.
                const bool persistentDsparkDecode =
                    model->dsparkEnabled && seqLens.size() == 1 &&
                    allDecodeTokens;
                if (!persistentDsparkDecode) {
                    FastllmCudaClearBigBuffer();
                }
#endif
                std::chrono::system_clock::time_point profileStartTime;
                const bool printProfile = GetFastllmEnv().printProfile;
                if (printProfile) {
                    profileStartTime = std::chrono::system_clock::now();
                    ClearProfiler();
                }
                Data inputIds = (seqLens.size() > 1 && allDecodeTokens) ?
                                Data(DataType::FLOAT32, {(int)seqLens.size(), 1}, ids) :
                                Data(DataType::FLOAT32, {1, (int)ids.size()}, ids);
                std::vector<int> ret;
                if (seqLens.size() > 1) {
                    ret = model->ForwardBatch((int)seqLens.size(), inputIds, attentionMasks,
                                              positionIds, seqLens, pastKeyValues, generationConfigs,
                                              tokensManager, &logits);
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
                            int curLen = std::min(prefillChunkSize, len - st);
                            auto chunkStartTime = std::chrono::system_clock::now();
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
                            if (model->verbose) {
                                auto chunkEndTime = std::chrono::system_clock::now();
                                float chunkSpend = GetSpan(chunkStartTime, chunkEndTime);
                                float chunkSpeed = chunkSpend > 0 ? curLen / chunkSpend : 0;
                                printf("[Prompt] Long Prefill ... (%d/%d, %d%%). Speed: %f tokens / s.\n",
                                       st, len, st * 100 / len, chunkSpeed);
                                lastRecordTime = chunkEndTime;
                                genTokens = 0;
                            }
                        }
                    } else {
                        Data emptyAttention, emptyPosition;
                        bool recordPrefillSpeed = model->verbose && seqLens[0] > 1;
                        std::chrono::system_clock::time_point prefillStartTime;
                        if (recordPrefillSpeed) {
                            prefillStartTime = std::chrono::system_clock::now();
                        }
                        ret = std::vector<int>{model->Forward(inputIds,
                                                              attentionMasks[0] == nullptr ? emptyAttention : *attentionMasks[0],
                                                              positionIds[0] == nullptr ? emptyPosition : *positionIds[0],
                                                              *pastKeyValue, generationConfigs[0],
                                                              tokensManager, logits[0])};
                        if (recordPrefillSpeed) {
                            auto prefillEndTime = std::chrono::system_clock::now();
                            float prefillSpend = GetSpan(prefillStartTime, prefillEndTime);
                            float prefillSpeed = prefillSpend > 0 ? seqLens[0] / prefillSpend : 0;
                            printf("[Prompt] %d Tokens. Speed: %f tokens / s.\n", seqLens[0], prefillSpeed);
                            lastRecordTime = prefillEndTime;
                            genTokens = 0;
                        }
                    }
                }

                if (printProfile) {
                    PrintProfiler();
                    int inputTokens = 0;
                    for (int seqLen : seqLens) {
                        inputTokens += seqLen;
                    }
                    float spend = GetSpan(profileStartTime, std::chrono::system_clock::now());
                    float tokenPerSecond = spend > 0.0f ? (float)inputTokens / spend : 0.0f;
                    printf("[fastllm-profile] loop = deepseekv4, batch = %d, input tokens = %d, "
                           "output tokens = %d, spend = %f s, tokens / s = %f\n",
                           (int)seqLens.size(), inputTokens, (int)ret.size(),
                           spend, tokenPerSecond);
                    fflush(stdout);
                }

                forwardLocker.unlock();
                dictLocker.lock();

                int resultCount = std::min((int)handles.size(), (int)ret.size());
                for (int i = 0; i < resultCount; i++) {
                    auto contextIt = model->responseContextDict.dicts.find(handles[i]);
                    if (contextIt == model->responseContextDict.dicts.end()) {
                        continue;
                    }
                    ResponseContext *ctx = contextIt->second;
                    std::vector<int> generatedTokens{ret[i]};

                    // ForwardDspark verifies and commits a whole accepted
                    // block at once.  Its scalar Forward() ABI used to expose
                    // the remaining tokens through one no-op scheduler turn
                    // per token.  Those turns still rebuilt CPU inputs and
                    // cleared CUDA workspaces, leaving a large gap between
                    // otherwise fast draft/target graph replays.  Drain the
                    // already-verified chain directly into this scheduler
                    // iteration.  Keep the pending representation in the
                    // model so direct scalar Forward() callers remain fully
                    // backward compatible.
                    if (model->dsparkEnabled) {
                        auto requestState = model->GetRequestState(
                            ctx->pastKeyValues);
                        if (requestState && requestState->dspark) {
                            auto &pending = requestState->dspark->pending;
                            int expectedInput = generatedTokens.back();
                            while (!pending.empty()) {
                                AssertInFastLLM(
                                    pending.front().expectedInput ==
                                        expectedInput,
                                    "DSpark scheduler pending output stream "
                                    "is out of sync.");
                                expectedInput = pending.front().outputToken;
                                generatedTokens.push_back(expectedInput);
                                pending.pop_front();
                            }

                            // The target cache already consumed the first
                            // token of every extra output edge.  Account for
                            // those logical decode inputs just as the old
                            // scalar pending turns did, so the next position
                            // and scheduler length bookkeeping stay exact.
                            const int extraTokens =
                                (int)generatedTokens.size() - 1;
                            ctx->preTokens += extraTokens;
                            auto indexIt = ctx->intParams.find("index");
                            if (indexIt != ctx->intParams.end()) {
                                indexIt->second += extraTokens;
                            }
                        }
                    }

                    for (int curRet : generatedTokens) {
                        if (curRet == model->eos_token_id ||
                            model->eos_token_ids.find(curRet) !=
                                model->eos_token_ids.end()) {
                            ctx->isEnding = true;
                            ctx->TryRecord(model);
                        } else {
                            auto itStopTk =
                                ctx->generationConfig.stop_token_ids.find(
                                    curRet);
                            if (itStopTk !=
                                ctx->generationConfig.stop_token_ids.end()) {
                                ctx->isEnding = true;
                                ctx->TryRecord(model);
                            }
                        }

                        if (!ctx->isEnding) {
                            model->UpdateToolCallConstraintState(ctx, curRet);
                            ctx->currentTokens = std::vector<int>{curRet};
                            ctx->resultTokenQueue.push(curRet);
                            ctx->allTokens.push_back(curRet);
                            ctx->tokens.Push(curRet);
                            ctx->curTokens++;
                            genTokens++;
                            if ((ctx->generationConfig.output_token_limit > 0 &&
                                 ctx->curTokens >=
                                    ctx->generationConfig.output_token_limit) ||
                                ctx->allTokens.size() >=
                                    model->max_positions) {
                                ctx->isEnding = true;
                                ctx->TryRecord(model);
                            }
                        }
                        if (ctx->isEnding) {
                            break;
                        }
                    }
                }
                if (model->verbose) {
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
                        printf("[Decode] alive = %d, pending = %d, "
                               "context len: %d, Speed: %f tokens / s.\n",
                               alive, pending, aliveLen,
                               spend > 0 ? (float)genTokens / spend : 0.0f);
                        lastRecordTime = nowTime;
                        genTokens = 0;
                    }
                }
                model->dictCV.notify_all();
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
        weight.embeddingNames.insert("mtp.2.markov_head.markov_w1.weight");
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
            // DeepSeek-V4 Flash 内置 DSpark 的 stage-0 主特征投影，
            // 以及 stage-2 Markov / confidence heads。
            "mtp.*.main_proj.weight",
            "mtp.*.markov_head.markov_w2.weight",
            "mtp.*.confidence_head.proj.weight",
        };
    }

    DeepSeekV4Model::~DeepSeekV4Model() {
        ShutdownRuntime();
    }

    std::map<std::string,
             std::vector<std::pair<std::string, DataType> > >
    DeepSeekV4Model::GetTensorMap(
            const std::vector<std::string> &tensorNames) {
        auto mapped = basellm::GetTensorMap(tensorNames);
#ifdef USE_CUDA
        // vLLM keeps the unquantized DSv4 auxiliary projections in BF16 and
        // requests FP32 accumulation/output for the compressor GEMMs.  The
        // generic FastLLM auto dtype otherwise converts these checkpoint-BF16
        // linears to FP16.  Besides moving target logits farther away, that can
        // change greedy tokens when the two leading logits are close.
        // Apply the source-precision contract automatically for embedded
        // DSpark TP decode; ordinary models and non-CUDA fallbacks retain the
        // established global dtype policy.
        const bool keepDsparkAuxBf16 =
            dsparkEnabled &&
            DeepSeekV4DeviceMapUsesMultiCuda(this->deviceMap);
        if (keepDsparkAuxBf16) {
            auto isDsparkAux = [](const std::string &name) {
                const bool compressorProjection =
                    name.find(".attn.compressor.wkv.weight") !=
                        std::string::npos ||
                    name.find(".attn.compressor.wgate.weight") !=
                        std::string::npos ||
                    name.find(".attn.indexer.compressor.wkv.weight") !=
                        std::string::npos ||
                    name.find(".attn.indexer.compressor.wgate.weight") !=
                        std::string::npos;
                return compressorProjection ||
                    name.find(".attn.indexer.weights_proj.weight") !=
                        std::string::npos ||
                    name.find(".ffn.gate.weight") != std::string::npos;
            };
            for (auto &source : mapped) {
                for (auto &destination : source.second) {
                    if (destination.first ==
                        "mtp.2.confidence_head.proj.weight") {
                        // The reference implementation promotes this tiny
                        // BF16 checkpoint tensor to FP32 because its sigmoid
                        // output is a scheduling probability, not an
                        // activation passed into the next model layer.
                        destination.second = DataType::FLOAT32;
                    } else if (isDsparkAux(destination.first)) {
                        destination.second = DataType::BFLOAT16;
                    }
                }
            }
        }
#endif
        return mapped;
    }

    std::string DeepSeekV4Model::SelectSpecialWeightDevice(
            const std::string &weightName, int layerId) const {
        const bool routedExpert = weightName.find(".ffn.experts.") != std::string::npos;
        const bool sharedExpert =
            weightName.find(".ffn.shared_experts.") != std::string::npos;
        if (routedExpert || (sharedExpert && !GetCudaSharedExpert())) {
            return this->SelectMoeDeviceForLayer(layerId);
        }

        if (this->deviceMap.empty()) {
            return "";
        }
        const int totalLayers = std::max(1, this->block_cnt);
        const int currentLayer = weightName == "head.weight" ? totalLayers :
            std::min(std::max(layerId + 1, 1), totalLayers);
        return SelectDeviceFromMap(this->deviceMap, currentLayer, totalLayers);
    }

    void DeepSeekV4Model::OnModelWeightsLoaded() {
#ifdef USE_CUDA
        if (!EnvFlagEnabled("FASTLLM_TP") ||
            !DeepSeekV4DeviceMapUsesMultiCuda(this->deviceMap)) {
            return;
        }

        int expectedShards = 0;
        int validShards = 0;
        int expectedRoutedShards = 0;
        int validRoutedShards = 0;
        std::set<int> devices;
        std::string firstInvalidWeight;
        for (const auto &specialWeight : this->specialWeights) {
            const std::string &name = specialWeight.first;
            const bool routedExpert =
                name.find(".ffn.experts.") != std::string::npos;
            if (routedExpert && !UseTensorParallelRoutedExperts()) {
                continue;
            }
            auto layerIt = this->specialWeightLayerIds.find(name);
            if (layerIt == this->specialWeightLayerIds.end() ||
                !DeepSeekV4DeviceSpecUsesType(
                    this->SelectSpecialWeightDevice(name, layerIt->second),
                    "multicuda")) {
                continue;
            }
            expectedShards++;
            if (routedExpert) {
                expectedRoutedShards++;
            }
            auto weightIt = this->weight.weight.find(name);
            bool valid = weightIt != this->weight.weight.end();
            const std::string &splitType = specialWeight.second;
            const int splitAxis = splitType == "linearColumn" ? 1 : 0;
            const TensorParallelLinearType expectedLinearType =
                splitType == "linearColumn" ? TP_LINEAR_COLUMN : TP_LINEAR_ROW;
            if (valid) {
                const Data &shardedWeight = weightIt->second;
                valid = shardedWeight.multiDeviceData &&
                        shardedWeight.multiDeviceDatas.size() >= 2 &&
                        shardedWeight.dims.size() == 2 &&
                        shardedWeight.tpLinearType == expectedLinearType;
                int splitLength = 0;
                for (const auto &local : shardedWeight.multiDeviceDatas) {
                    const Data *localWeight = local.second;
                    bool localValid =
                        localWeight != nullptr && localWeight->dims.size() == 2 &&
                        localWeight->dataDevice == DataDevice::CUDA &&
                        localWeight->cudaData != nullptr &&
                        GetPointerDeviceId(localWeight->cudaData) == local.first &&
                        localWeight->tpLinearType == expectedLinearType &&
                        localWeight->dims[1 - splitAxis] ==
                            shardedWeight.dims[1 - splitAxis];
                    if (!localValid) {
                        valid = false;
                        break;
                    }
                    splitLength += localWeight->dims[splitAxis];
                }
                valid = valid && splitLength == shardedWeight.dims[splitAxis];
            }
            if (!valid) {
                if (firstInvalidWeight.empty()) {
                    firstInvalidWeight = name;
                }
                continue;
            }
            validShards++;
            if (routedExpert) {
                validRoutedShards++;
            }
            for (const auto &local : weightIt->second.multiDeviceDatas) {
                if (local.second != nullptr) {
                    devices.insert(local.first);
                }
            }
        }

        AssertInFastLLM(
            expectedShards > 0 && validShards == expectedShards && devices.size() > 1,
            "DeepSeek-V4 tensor parallel weight validation failed: " +
            std::to_string(validShards) + "/" + std::to_string(expectedShards) +
            " selected weights are sharded" +
            (firstInvalidWeight.empty() ? std::string(".") :
             std::string(", first invalid weight: ") + firstInvalidWeight + "."));
        AssertInFastLLM(
            !UseTensorParallelRoutedExperts() ||
                (expectedRoutedShards > 0 &&
                 validRoutedShards == expectedRoutedShards),
            "DeepSeek-V4 DSpark routed expert tensor parallel validation failed: " +
            std::to_string(validRoutedShards) + "/" +
            std::to_string(expectedRoutedShards) + " weights are sharded.");

        std::ostringstream deviceNames;
        bool first = true;
        for (int device : devices) {
            if (!first) {
                deviceNames << ",";
            }
            first = false;
            deviceNames << device;
        }
        printf("[Fastllm] DeepSeek-V4 TP validated: %d weights are sharded "
               "across CUDA devices [%s] (%d routed expert weights); "
               "routed experts use %s.\n",
               validShards, deviceNames.str().c_str(), validRoutedShards,
               this->SelectMoeDeviceForLayer(0).c_str());
        fflush(stdout);
#endif
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

        dsparkTokens = std::max(0, EnvInt("FASTLLM_DSPARK_TOKENS", 0));
        dsparkEnabled = dsparkTokens > 0;
        if (dsparkEnabled) {
            // The model-specific scheduler can drain several already-verified
            // tokens from one request, so do not mix those pending queues in
            // the ordinary batched-decode overload.
            this->canDoConcurrentForward = false;
            // NVIDIA's DeepSeek-V4 DSpark checkpoint contains three stages
            // under mtp.0/1/2.  num_nextn_predict_layers describes the legacy
            // MTP head and is therefore not the number of DSpark stages.
            dsparkLayers = 3;
            dsparkNoiseTokenId = GetIntWithFallback(
                this->weight, {"dspark_noise_token_id"}, -1);
            dsparkMarkovRank = GetIntWithFallback(
                this->weight, {"dspark_markov_rank"}, 0);
            int trainedBlock = GetIntWithFallback(
                this->weight, {"dspark_block_size"}, 0);
            dsparkTargetLayerIds.clear();
            auto targetIt = this->weight.dicts.find(
                "dspark_target_layer_ids");
            if (targetIt != this->weight.dicts.end()) {
                std::string parseError;
                auto targetJson = json11::Json::parse(
                    targetIt->second, parseError);
                if (parseError.empty() && targetJson.is_array()) {
                    for (const auto &item : targetJson.array_items()) {
                        dsparkTargetLayerIds.push_back(item.int_value());
                    }
                }
            }
            AssertInFastLLM(
                dsparkLayers == 3 && trainedBlock > 0 &&
                dsparkTokens >= trainedBlock && dsparkNoiseTokenId >= 0 &&
                dsparkMarkovRank > 0 &&
                dsparkTargetLayerIds == std::vector<int>({40, 41, 42}),
                "The embedded DeepSeek-V4 DSpark configuration is unsupported.");
            const char *confidenceThresholdEnv = std::getenv(
                "FASTLLM_DSPARK_CONFIDENCE_THRESHOLD");
            if (confidenceThresholdEnv != nullptr &&
                confidenceThresholdEnv[0] != '\0') {
                char *end = nullptr;
                dsparkConfidenceThreshold = std::strtof(
                    confidenceThresholdEnv, &end);
                AssertInFastLLM(
                    end != confidenceThresholdEnv && *end == '\0' &&
                    std::isfinite(dsparkConfidenceThreshold) &&
                    dsparkConfidenceThreshold >= 0.0f &&
                    dsparkConfidenceThreshold <= 1.0f,
                    "FASTLLM_DSPARK_CONFIDENCE_THRESHOLD must be in [0, 1].");
            }
        }

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
        bool tensorParallelLoad = EnvFlagEnabled("FASTLLM_TP");
        bool tensorParallelAttention = tensorParallelLoad &&
                                       DeepSeekV4DeviceMapUsesMultiCuda(this->deviceMap);
        // wo_a is bandwidth-bound during single-token CUDA decode.  Keep its
        // checkpoint FP8 representation on a single CUDA device and dequantize
        // to the legacy FP16 value in the CUDA kernel.  Other device maps
        // retain the established load-time FP16 conversion.
        const bool cudaWoADevice =
            tensorParallelAttention ||
            DeepSeekV4DeviceMapUsesSingleCuda(this->deviceMap);
        const bool keepCudaFp8WoA =
            cudaWoADevice &&
            (dsparkEnabled ||
             DeepSeekV4DeviceMapUsesSingleCuda(this->deviceMap));
        for (int i = 0; i < block_cnt; i++) {
            if (tensorParallelAttention) {
                this->AddSpecialWeight("layers." + std::to_string(i) + ".attn.wq_b.weight", "linearRow", i);
                this->AddSpecialWeight("layers." + std::to_string(i) + ".attn.wo_a.weight", "linearRow", i);
                this->AddSpecialWeight("layers." + std::to_string(i) + ".attn.wo_b.weight", "linearColumn", i);
            }
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
                if (j != -1 || !GetCudaSharedExpert() || tensorParallelLoad) {
                    this->AddSpecialWeight(swigluName, "linearSwiglu", i);
                    this->AddSpecialWeight(downName, "linearColumn", i);
                }
                this->moeLinears.insert(w1Name);
                this->moeLinears.insert(w3Name);
                this->moeLinears.insert(downName);
            }
            this->cantQuantLinears.insert("layers." + std::to_string(i) + ".attn.wkv.weight");
            if (!keepCudaFp8WoA) {
                this->cantQuantLinears.insert(
                    "layers." + std::to_string(i) +
                    ".attn.wo_a.weight");
            }
        }
        if (dsparkEnabled) {
            for (int stage = 0; stage < dsparkLayers; stage++) {
                const int layerId = std::max(0, block_cnt - dsparkLayers + stage);
                const std::string prefix = "mtp." + std::to_string(stage);
                if (tensorParallelAttention) {
                    this->AddSpecialWeight(
                        prefix + ".attn.wq_b.weight", "linearRow", layerId);
                    this->AddSpecialWeight(
                        prefix + ".attn.wo_a.weight", "linearRow", layerId);
                    this->AddSpecialWeight(
                        prefix + ".attn.wo_b.weight", "linearColumn", layerId);
                }
                for (int expert = -1; expert < this->num_experts; expert++) {
                    std::string expertPrefix = prefix + ".ffn.";
                    if (expert < 0) {
                        expertPrefix += "shared_experts";
                    } else {
                        expertPrefix += "experts." + std::to_string(expert);
                    }
                    const std::string w1Name = expertPrefix + ".w1.weight";
                    const std::string w3Name = expertPrefix + ".w3.weight";
                    const std::string gateupName =
                        expertPrefix + ".gateup.weight";
                    const std::string downName = expertPrefix + ".w2.weight";
                    this->weightMergeRules.push_back(WeightMergeRule({
                        WeightMergeRuleSingle(
                            {w1Name, w3Name}, gateupName,
                            std::string("linearSwiglu"))}));
                    if (expert >= 0 || !GetCudaSharedExpert() ||
                        tensorParallelLoad) {
                        this->AddSpecialWeight(
                            gateupName, "linearSwiglu", layerId);
                        this->AddSpecialWeight(
                            downName, "linearColumn", layerId);
                    }
                    this->moeLinears.insert(w1Name);
                    this->moeLinears.insert(w3Name);
                    this->moeLinears.insert(downName);
                }
                this->cantQuantLinears.insert(
                    prefix + ".attn.wkv.weight");
                if (!keepCudaFp8WoA) {
                    this->cantQuantLinears.insert(
                        prefix + ".attn.wo_a.weight");
                }
            }
            if (tensorParallelAttention) {
                this->AddSpecialWeight(
                    "mtp.2.markov_head.markov_w2.weight",
                    "linearRow", block_cnt - 1);
            }
            std::printf(
                "[Fastllm] DeepSeek-V4 embedded DSpark enabled: "
                "3 stages, %d draft tokens, target layers [40,41,42].\n",
                dsparkTokens);
            std::fflush(stdout);
        }
        if (tensorParallelAttention) {
            this->AddSpecialWeight("head.weight", "linearRow", 0);
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

    std::vector<int> DeepSeekV4Model::RunDsparkTarget(
            const std::vector<int> &tokenIds, int startPos,
            std::vector<std::pair<Data, Data> > &pastKeyValues,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens,
            std::vector<float> *retLogits,
            DeepSeekV4DsparkTargetCapture *capture) {
        AssertInFastLLM(!tokenIds.empty(),
                        "DSpark target run cannot be empty.");
        if (capture != nullptr) {
            capture->samplingLogitsFloat = nullptr;
            capture->samplingGreedyIds = nullptr;
            capture->samplingGreedyScores = nullptr;
            capture->samplingReadyEvents.clear();
            capture->samplingReady = false;
            capture->samplingDevicesDrained = false;
            capture->contextStageKV.clear();
            capture->contextRows = 0;
            capture->contextReady = false;
        }
        std::vector<float> ids(tokenIds.begin(), tokenIds.end());
        std::vector<float> positions(tokenIds.size());
        for (int index = 0; index < (int)tokenIds.size(); index++) {
            positions[index] = (float)(startPos + index);
        }
        Data input(DataType::FLOAT32, {1, (int)tokenIds.size()}, ids);
        Data position(DataType::FLOAT32,
                      {1, (int)tokenIds.size()}, positions);
        std::vector<std::vector<float>*> logits{retLogits};
        DeepSeekV4DsparkTargetCaptureScope captureScope(capture);
        return ForwardBatch(1, input, Data(), position, pastKeyValues,
                            generationConfig, lastTokens, &logits);
    }

    void DeepSeekV4Model::AppendDsparkTargetHidden(
            const DeepSeekV4DsparkTargetCapture &capture, int tokens,
            DeepSeekV4DsparkContext &context) {
        AssertInFastLLM(tokens > 0,
                        "DSpark must append at least one target token.");
        bool fullWindowBefore =
            (int)context.mainWindowKV.size() == dsparkLayers;
        for (int stage = 0; stage < dsparkLayers && fullWindowBefore;
             ++stage) {
            fullWindowBefore =
                context.mainWindowKV[stage].dims ==
                    std::vector<int>({1, window_size, head_dim_full});
        }
        int capturedTokens = -1;
        std::vector<const Data*> capturedFeatures;
        capturedFeatures.reserve(dsparkTargetLayerIds.size());
        for (int feature = 0;
             feature < (int)dsparkTargetLayerIds.size(); feature++) {
            auto hiddenIt = capture.targetHidden.find(
                dsparkTargetLayerIds[feature]);
            AssertInFastLLM(
                hiddenIt != capture.targetHidden.end() &&
                hiddenIt->second.dims.size() == 3 &&
                hiddenIt->second.dims[1] >= tokens &&
                hiddenIt->second.dims[2] == embed_dim,
                "DSpark target hidden capture is incomplete.");
            if (capturedTokens < 0) {
                capturedTokens = hiddenIt->second.dims[1];
            }
            AssertInFastLLM(
                hiddenIt->second.dims[1] == capturedTokens,
                "DSpark target hidden captures have inconsistent lengths.");
            capturedFeatures.push_back(&hiddenIt->second);
        }
        AssertInFastLLM(capturedTokens >= tokens,
                        "DSpark target hidden capture is too short.");
        AssertInFastLLM(!capturedFeatures.empty(),
                        "DSpark target hidden feature list is empty.");

        if ((int)context.mainWindowKV.size() != dsparkLayers) {
            context.mainWindowKV.clear();
            context.mainWindowKV.resize(dsparkLayers);
        }
        std::vector<Data*> stageKVs;
        const bool useGraphContext =
            capture.contextReady && capture.contextRows == capturedTokens &&
            (int)capture.contextStageKV.size() == dsparkLayers &&
            std::all_of(
                capture.contextStageKV.begin(),
                capture.contextStageKV.end(),
                [capturedTokens, this](const Data *kv) {
                    return kv != nullptr && kv->dims ==
                        std::vector<int>({1, capturedTokens, head_dim_full});
                });
        if (useGraphContext) {
            stageKVs = capture.contextStageKV;
        } else {
            Data *combined = nullptr;
            if (capturedFeatures.size() == 1) {
                Copy(*capturedFeatures[0], context.targetCombined);
                combined = &context.targetCombined;
            } else {
                Cat(*capturedFeatures[0], *capturedFeatures[1], -1,
                    context.targetCombinedTemp);
                combined = &context.targetCombinedTemp;
                for (int feature = 2;
                     feature < (int)capturedFeatures.size(); ++feature) {
                    Data *output =
                        combined == &context.targetCombinedTemp ?
                            &context.targetCombined :
                            &context.targetCombinedTemp;
                    Cat(*combined, *capturedFeatures[feature], -1, *output);
                    combined = output;
                }
            }

            // vLLM projects every target row in the padded verification batch
            // before applying num_rejected.  Keep the same ordering here: FP8
            // projection/normalization is performed for all captured rows, and
            // only the committed prefix is sliced from the resulting context KV.
            ApplyDeviceMap(this->deviceMap, block_cnt, block_cnt);
            DeepSeekV4Linear(*combined, weight["mtp.0.main_proj.weight"],
                             Data(), context.targetProjected, true);
            RMSNormReference(context.targetProjected,
                             weight["mtp.0.main_norm.weight"],
                             rms_norm_eps, context.targetMainHidden,
                             DataType::BFLOAT16);

            if ((int)context.targetStageKV.size() != dsparkLayers) {
                context.targetStageKV.resize(dsparkLayers);
            }
            for (int stage = 0; stage < dsparkLayers; stage++) {
                const int layerId = std::max(
                    0, block_cnt - dsparkLayers + stage);
                ApplyDeviceMap(this->deviceMap, layerId + 1, block_cnt);
                const std::string prefix =
                    "mtp." + std::to_string(stage) + ".attn";
                Data &kv = context.targetStageKV[stage];
                DeepSeekV4Linear(context.targetMainHidden,
                                 weight[prefix + ".wkv.weight"],
                                 Data(), kv, true);
                kv.Reshape({1, capturedTokens, 1, head_dim_full});
                RMSNormReference(kv, weight[prefix + ".kv_norm.weight"],
                                 rms_norm_eps, kv, DataType::BFLOAT16);
                DeepSeekV4RotaryQuant(
                    kv, qk_rope_head_dim, rope_base,
                    context.committedTokens, 0, rope_factor,
                    rope_scaling_beta_fast, rope_scaling_beta_slow,
                    head_dim_full - qk_rope_head_dim, 64);
                kv.Reshape({1, capturedTokens, head_dim_full});
                stageKVs.push_back(&kv);
            }
        }

        bool useFullWindowUpdate = fullWindowBefore;
        for (int stage = 0; stage < dsparkLayers && useFullWindowUpdate;
             ++stage) {
            useFullWindowUpdate = CanAppendFullWindowKVCache(
                *stageKVs[stage], tokens, window_size,
                context.mainWindowKV[stage]);
        }
        if (useFullWindowUpdate) {
            AssertInFastLLM(
                AppendFullWindowKVCacheBatch(
                    stageKVs, tokens, window_size,
                    context.mainWindowKV),
                "DSpark full-window in-place KV update failed.");
        } else {
            // Generic path for CPU, lower-SM CUDA, short prefixes and unusual
            // tensor layouts.  It intentionally retains the established
            // Cat/Split behavior so the optimized steady-state path does not
            // narrow model or device compatibility.
            for (int stage = 0; stage < dsparkLayers; stage++) {
                Data &kv = *stageKVs[stage];

                Data committedKV;
                if (capturedTokens == tokens) {
                    CopyTensorData(committedKV, kv);
                } else {
                    Split(kv, 1, 0, tokens, committedKV);
                }

                Data appended;
                if (context.mainWindowKV[stage].dims.empty()) {
                    CopyTensorData(appended, committedKV);
                } else {
                    ConcatSeqReference(
                        context.mainWindowKV[stage], committedKV, appended);
                }
                if (appended.dims[1] > window_size) {
                    Data tail;
                    Split(appended, 1,
                          appended.dims[1] - window_size,
                          appended.dims[1], tail);
                    CopyTensorData(context.mainWindowKV[stage], tail);
                } else {
                    CopyTensorData(context.mainWindowKV[stage], appended);
                }
            }
        }
        context.committedTokens += tokens;
    }

    std::vector<int> DeepSeekV4Model::SampleDsparkTargetRows(
            Data &headInput,
            DeepSeekV4DsparkContext *persistentContext) {
        AssertInFastLLM(
            headInput.dims.size() == 3 && headInput.dims[0] == 1 &&
            headInput.dims[1] > 0 && headInput.dims[2] == embed_dim,
            "DSpark target head capture has an invalid shape.");
        const int rows = headInput.dims[1];
#ifdef USE_CUDA
        struct ScopedDsparkSamplingAsync {
            bool active = false;
            bool previous = false;
            explicit ScopedDsparkSamplingAsync(bool enabled) :
                    active(enabled) {
                if (active) {
                    previous = MultiCudaSetPersistentAsyncDispatch(true);
                }
            }
            ~ScopedDsparkSamplingAsync() {
                if (active) {
                    MultiCudaSetPersistentAsyncDispatch(previous);
                }
            }
        } samplingAsync(persistentContext != nullptr &&
                        DeepSeekV4PreferCuda());
#endif
        DeepSeekV4DsparkTargetCapture *graphCapture =
            persistentContext == nullptr ? nullptr :
                &persistentContext->targetCapture;
        const bool useGraphSampling =
            graphCapture != nullptr && graphCapture->samplingReady &&
            graphCapture->samplingLogitsFloat != nullptr &&
            graphCapture->samplingGreedyIds != nullptr &&
            graphCapture->samplingGreedyScores != nullptr &&
            !graphCapture->samplingReadyEvents.empty();

        // Do not mutate the persistent verification capture: a CUDA Graph
        // replay writes to this exact allocation and expects its original
        // [1, tokens, hidden] layout to remain stable between rounds.  A
        // steady-state graph has already normalized and projected these rows,
        // so it can skip this copy entirely.
        Data localSamplingInput;
        Data &samplingInput = persistentContext == nullptr ?
            localSamplingInput : persistentContext->targetSamplingInput;
        Data *samplingInputPtr = &headInput;
        if (!useGraphSampling) {
            if (!CopyTensorDataInPlaceCuda(samplingInput, headInput)) {
                CopyTensorData(samplingInput, headInput);
            }
            samplingInput.Reshape({rows, 1, embed_dim});
            samplingInputPtr = &samplingInput;
        }
        GenerationConfig greedy;
        greedy.do_sample = false;
        greedy.top_k = 1;
        greedy.repeat_penalty = 1.0f;
        greedy.output_token_least = 0;
        greedy.output_logits = false;
        std::vector<GenerationConfig> configs(rows, greedy);
        std::vector<int> seqLens(rows, 1);
        LastTokensManager emptyLastTokens(rows, greedy.last_n);
        std::vector<std::pair<Data*, Data*> > emptyPast;
        std::vector<int> sampled;
        Data *precomputedLogits = useGraphSampling ?
            graphCapture->samplingLogitsFloat : nullptr;
        Data *precomputedGreedyIds = useGraphSampling ?
            graphCapture->samplingGreedyIds : nullptr;
        Data *precomputedGreedyScores = useGraphSampling ?
            graphCapture->samplingGreedyScores : nullptr;
        const std::map<int, void*> *precomputedReadyEvents =
            useGraphSampling ? &graphCapture->samplingReadyEvents : nullptr;
        if (!useGraphSampling && persistentContext != nullptr) {
            RMSNorm(samplingInput, weight["norm.weight"], rms_norm_eps,
                    samplingInput);
            Linear(samplingInput, weight["head.weight"], *GetEmptyData(),
                   persistentContext->targetSamplingLogits);
            ToDataType(persistentContext->targetSamplingLogits,
                       persistentContext->targetSamplingLogitsFloat,
                       DataType::FLOAT32);
            precomputedLogits =
                &persistentContext->targetSamplingLogitsFloat;
        }
        LLMSamplingBlock(
            this, samplingInputPtr, &weight["norm.weight"],
            &weight["head.weight"], rms_norm_eps, rows, true,
            seqLens, emptyPast, configs, emptyLastTokens,
            nullptr, sampled, precomputedLogits,
            precomputedGreedyIds, precomputedGreedyScores,
            precomputedReadyEvents);
        if (useGraphSampling) {
            // The root gather waits every replay-done event and synchronizes
            // its stream before returning.  All verifier ranks are therefore
            // complete; rejection rollback need not synchronize eight devices
            // once more.
            graphCapture->samplingDevicesDrained = true;
        }
        AssertInFastLLM((int)sampled.size() == rows,
                        "DSpark target row sampling failed.");
        return sampled;
    }

    DeepSeekV4DsparkProposal DeepSeekV4Model::RunDsparkDraft(
            int anchorToken, DeepSeekV4DsparkContext &context,
            bool forceEager, bool deferGpuCopy) {
        AssertInFastLLM(
            context.initialized &&
            (int)context.mainWindowKV.size() == dsparkLayers,
            "DSpark draft context is not initialized.");
        if (dsparkWeights.empty()) {
            auto getWeightPtr = [this](const std::string &name) -> Data* {
                auto it = this->weight.weight.find(name);
                return it == this->weight.weight.end() ? nullptr : &it->second;
            };
            dsparkWeights.resize(dsparkLayers);
            dsparkBiass.resize(dsparkLayers);
            for (int stage = 0; stage < dsparkLayers; stage++) {
                const std::string ffn =
                    "mtp." + std::to_string(stage) + ".ffn";
                dsparkWeights[stage].push_back(getWeightPtr(
                    ffn + ".shared_experts.gateup.weight"));
                dsparkWeights[stage].push_back(getWeightPtr(
                    ffn + ".shared_experts.w2.weight"));
                dsparkBiass[stage].push_back(nullptr);
                dsparkBiass[stage].push_back(nullptr);
                for (int expert = 0; expert < num_experts; expert++) {
                    const std::string expertPrefix = ffn + ".experts." +
                        std::to_string(expert);
                    dsparkWeights[stage].push_back(getWeightPtr(
                        expertPrefix + ".gateup.weight"));
                    dsparkWeights[stage].push_back(getWeightPtr(
                        expertPrefix + ".w2.weight"));
                    dsparkBiass[stage].push_back(nullptr);
                    dsparkBiass[stage].push_back(nullptr);
                }
            }
        }

        std::vector<float> inputValues(
            dsparkTokens, (float)dsparkNoiseTokenId);
        inputValues[0] = (float)anchorToken;
        std::vector<int> routingTokenIds(
            dsparkTokens, dsparkNoiseTokenId);
        routingTokenIds[0] = anchorToken;
        Data inputIds(DataType::FLOAT32, {1, dsparkTokens}, inputValues);

        std::string draftBackboneFailure;
        auto runDraftBackbone = [&](const Data &activeInputIds,
                                    const std::vector<Data> &activeMainWindowKV,
                                    const Data *decodeMeta,
                                    Data &baseLogits,
                                    DeepSeekV4DecodeWorkspace *persistentWorkspace,
                                    Data *confidenceHidden)
                                    -> bool {
            draftBackboneFailure.clear();
#ifdef USE_CUDA
            struct ScopedDraftPersistentAsync {
                bool previous = false;
                bool active = false;
                explicit ScopedDraftPersistentAsync(bool enabled) :
                        active(enabled) {
                    if (active) {
                        previous = MultiCudaSetPersistentAsyncDispatch(true);
                    }
                }
                ~ScopedDraftPersistentAsync() {
                    if (active) {
                        MultiCudaSetPersistentAsyncDispatch(previous);
                    }
                }
            } persistentAsync(decodeMeta != nullptr);
            auto draftGraphHealthy = [&](const char *stage) {
                if (decodeMeta != nullptr &&
                    (FastllmCudaGetThreadError() ||
                     FastllmCudaGetGraphError())) {
                    draftBackboneFailure = stage;
                    return false;
                }
                return true;
            };
#endif
            DeepSeekV4DecodeWorkspace localWorkspace;
            DeepSeekV4DecodeWorkspace &draftWorkspace =
                persistentWorkspace == nullptr ? localWorkspace :
                    *persistentWorkspace;
            Data &embedded = draftWorkspace.hiddenStatesBeforeHcExpand;
            Data &hiddenStates = draftWorkspace.hiddenStates;
            Data &hiddenStatesTemp = draftWorkspace.hiddenStatesTemp;
            Data &attnInput = draftWorkspace.attnInput;
            Data &qr = draftWorkspace.qr;
            Data &qNorm = draftWorkspace.qNorm;
            Data &q = draftWorkspace.q;
            Data &kv = draftWorkspace.kv;
            HcMix &attnMix = draftWorkspace.attnMix;
            Data &attentionKV = draftWorkspace.compressorKV;
            Data &attnOut4 = draftWorkspace.attnOut4;
            Data &woAOut = draftWorkspace.woAOut;
            Data &attnOut = draftWorkspace.attnOut;
            HcMix &ffnMix = draftWorkspace.ffnMix;
            Data &ffnInput = draftWorkspace.ffnInput;
            Data &expertIndex = draftWorkspace.expertIndex;
            Data &expertScore = draftWorkspace.expertScore;
            Data &sharedExpertOut = draftWorkspace.sharedExpertOut;
            Data &sharedW1 = draftWorkspace.sharedW1;
            Data &sharedW3 = draftWorkspace.sharedW3;
            Data &w1 = draftWorkspace.w1;
            Data &w2 = draftWorkspace.w2;
            Data &w3 = draftWorkspace.w3;
            Data &tempInput = draftWorkspace.tempInput;
            Data &tempOutput = draftWorkspace.tempOutput;
            Data &moeInputTemp = draftWorkspace.moeInputTemp;
            Data &moeOutputTemp = draftWorkspace.moeOutputTemp;
            Data &ffnOut = draftWorkspace.ffnOut;
            Data &headInput = draftWorkspace.headInput;
            Data &samplingHeadRoot = draftWorkspace.samplingHeadRoot;
            Data &samplingHeadReplicated =
                draftWorkspace.samplingHeadReplicated;
            Data &samplingHeadNorm = draftWorkspace.samplingHeadNorm;
            EmbeddingDirect(activeInputIds, weight["embed.weight"], embedded);
            embedded.Reshape({1, dsparkTokens, 1, embed_dim});
            bool repeatedToTpReplicas = false;
#ifdef USE_CUDA
            if (decodeMeta != nullptr) {
                repeatedToTpReplicas = MultiCudaRepeatToReplicated(
                    embedded, 2, hc_mult, hiddenStates);
            }
#endif
            if (!repeatedToTpReplicas) {
                Repeat(embedded, 2, hc_mult, hiddenStates);
            }
#ifdef USE_CUDA
            if (!draftGraphHealthy("embedding")) {
                return false;
            }
#endif
            Data *curHiddenStates = &hiddenStates;
            Data *nextHiddenStates = &hiddenStatesTemp;

            for (int stage = 0; stage < dsparkLayers; stage++) {
            const int layerId = std::max(
                0, block_cnt - dsparkLayers + stage);
            ApplyDeviceMap(this->deviceMap, layerId + 1, block_cnt);
            const std::string pre = "mtp." + std::to_string(stage);

            DeepSeekV4HcPre(
                *curHiddenStates, weight[pre + ".hc_attn_fn"],
                weight[pre + ".hc_attn_scale"],
                weight[pre + ".hc_attn_base"], hc_mult,
                hc_sinkhorn_iters, hc_eps, rms_norm_eps,
                attnMix.y, attnMix.postData, attnMix.combData);
            RMSNormReference(
                attnMix.y, weight[pre + ".attn_norm.weight"],
                rms_norm_eps, attnInput, DataType::BFLOAT16);

            DeepSeekV4Linear(
                attnInput, weight[pre + ".attn.wq_a.weight"],
                Data(), qr, true);
            RMSNormReference(
                qr, weight[pre + ".attn.q_norm.weight"],
                rms_norm_eps, qNorm, DataType::BFLOAT16);
            weight[pre + ".attn.wq_b.weight"].tpLinearType =
                TP_LINEAR_ROW;
            DeepSeekV4Linear(
                qNorm, weight[pre + ".attn.wq_b.weight"],
                Data(), q);
            q.Reshape(
                {1, dsparkTokens, num_attention_heads, head_dim_full});
            if (decodeMeta != nullptr) {
#ifdef USE_CUDA
                if (!DeepSeekV4ScaleQRotaryGraphMultiCuda(
                        q, qk_rope_head_dim, rope_base, *decodeMeta, 0,
                        rope_factor, rope_scaling_beta_fast,
                        rope_scaling_beta_slow, rms_norm_eps)) {
                    draftBackboneFailure = "q-rope";
                    FastllmCudaSetThreadError();
                    return false;
                }
#else
                return false;
#endif
            } else {
                ScaleQRatory(
                    q, rms_norm_eps, qk_rope_head_dim, rope_base,
                    context.committedTokens, 0, rope_factor,
                    rope_scaling_beta_fast, rope_scaling_beta_slow);
            }

            DeepSeekV4Linear(
                attnInput, weight[pre + ".attn.wkv.weight"],
                Data(), kv, true);
            kv.Reshape({1, dsparkTokens, 1, head_dim_full});
            RMSNormReference(
                kv, weight[pre + ".attn.kv_norm.weight"],
                rms_norm_eps, kv, DataType::BFLOAT16);
            if (decodeMeta != nullptr) {
#ifdef USE_CUDA
                if (!DeepSeekV4RotaryQuantGraphMultiCuda(
                        kv, qk_rope_head_dim, rope_base, *decodeMeta, 0,
                        rope_factor, rope_scaling_beta_fast,
                        rope_scaling_beta_slow,
                        head_dim_full - qk_rope_head_dim, 64, 1)) {
                    draftBackboneFailure = "kv-rope-quant";
                    FastllmCudaSetThreadError();
                    return false;
                }
#else
                return false;
#endif
            } else {
                DeepSeekV4RotaryQuant(
                    kv, qk_rope_head_dim, rope_base,
                    context.committedTokens, 0, rope_factor,
                    rope_scaling_beta_fast, rope_scaling_beta_slow,
                    head_dim_full - qk_rope_head_dim, 64);
            }
            kv.Reshape({1, dsparkTokens, head_dim_full});

            const int prefixLen = activeMainWindowKV[stage].dims.empty() ?
                0 : activeMainWindowKV[stage].dims[1];
            AssertInFastLLM(
                prefixLen == std::min(window_size,
                                      context.committedTokens),
                "DSpark main KV window is out of sync.");
            if (prefixLen > 0) {
                ConcatSeqReference(
                    activeMainWindowKV[stage], kv, attentionKV);
            } else {
                CopyTensorData(attentionKV, kv);
            }
            SparseAttentionReference(
                q, attentionKV, weight[pre + ".attn.attn_sink"],
                window_size, qk_rope_head_dim, rope_base,
                context.committedTokens,
                1.0f / std::sqrt((float)head_dim_full), attnOut4,
                0, 0, rope_factor, rope_scaling_beta_fast,
                rope_scaling_beta_slow, prefixLen, true, decodeMeta);
            DeepSeekV4WoA(
                attnOut4, weight[pre + ".attn.wo_a.weight"],
                o_groups, o_lora_rank, woAOut);
            DeepSeekV4Linear(
                woAOut, weight[pre + ".attn.wo_b.weight"],
                Data(), attnOut);
            DeepSeekV4HcPost(
                attnOut, *curHiddenStates, attnMix.postData,
                attnMix.combData, *nextHiddenStates);
            std::swap(curHiddenStates, nextHiddenStates);

            DeepSeekV4HcPre(
                *curHiddenStates, weight[pre + ".hc_ffn_fn"],
                weight[pre + ".hc_ffn_scale"],
                weight[pre + ".hc_ffn_base"], hc_mult,
                hc_sinkhorn_iters, hc_eps, rms_norm_eps,
                ffnMix.y, ffnMix.postData, ffnMix.combData);
            RMSNormReference(
                ffnMix.y, weight[pre + ".ffn_norm.weight"],
                rms_norm_eps, ffnInput, DataType::BFLOAT16);
            const std::vector<int> ffnDims = ffnInput.dims;
            ffnInput.Reshape({dsparkTokens, embed_dim});

            BuildMoERoutingData(
                weight, pre + ".ffn", ffnInput, routingTokenIds,
                num_experts, num_experts_per_tok, scoring_func,
                routed_scaling_factor, expertIndex, expertScore,
                decodeMeta);

            std::vector<Data*> moeWeights = dsparkWeights[stage];
            bool hasSharedExpertOut = false;
            const bool routedTensorParallel =
                this->UseTensorParallelRoutedExperts();
            auto sharedGateupIt = weight.weight.find(
                pre + ".ffn.shared_experts.gateup.weight");
            auto sharedDownIt = weight.weight.find(
                pre + ".ffn.shared_experts.w2.weight");
            if (GetCudaSharedExpert() &&
                sharedGateupIt != weight.weight.end() &&
                sharedDownIt != weight.weight.end() &&
                !IsDiskWeight(&sharedGateupIt->second) &&
                !IsDiskWeight(&sharedDownIt->second)) {
                sharedGateupIt->second.tpLinearType = TP_LINEAR_ROW;
                sharedGateupIt->second.tpPackType = TP_PACK_GATEUP;
                sharedDownIt->second.tpLinearType = TP_LINEAR_COLUMN;
                LinearSwigluBlock(
                    &ffnInput, &sharedGateupIt->second, GetEmptyData(),
                    &sharedW3, &sharedW1);
                // Draft stages stay byte-for-byte on the established two
                // collectives.  The paired reduction targets M=8 validation,
                // where the saved collective is material; changing draft
                // launch timing can otherwise perturb proposal acceptance even
                // when the arithmetic itself is identical.
                DeepSeekV4Linear(
                    sharedW1, sharedDownIt->second, *GetEmptyData(),
                    sharedExpertOut);
                hasSharedExpertOut = true;
                moeWeights[0] = moeWeights[1] = nullptr;
            }

            this->ApplyMoeDeviceMapForLayer(layerId);
            const bool routedExpertParallel =
                !routedTensorParallel && DeepSeekV4DeviceSpecUsesType(
                    this->SelectMoeDeviceForLayer(layerId), "multicuda");
            MergeMOEBlock(
                &ffnInput, &expertIndex, &expertScore,
                &moeWeights, &dsparkBiass[stage],
                &w1, &w2, &w3, &tempInput, &tempOutput,
                1.0f, &ffnOut, layerId,
                ffnInput.dataType, ffnInput.dataType,
                &moeInputTemp, &moeOutputTemp,
                MoeGateSwiglu, routedExpertParallel, swiglu_limit, true);
            ApplyDeviceMap(this->deviceMap, layerId + 1, block_cnt);
            if (hasSharedExpertOut) {
                if (!(ffnOut.multiDeviceData &&
                      sharedExpertOut.multiDeviceData)) {
                    ffnOut.ToDevice(sharedExpertOut.dataDevice);
                }
                AddTo(ffnOut, sharedExpertOut);
            }
            ffnOut.Reshape(ffnDims);
            DeepSeekV4HcPost(
                ffnOut, *curHiddenStates, ffnMix.postData,
                ffnMix.combData, *nextHiddenStates);
            std::swap(curHiddenStates, nextHiddenStates);
#ifdef USE_CUDA
            if (!draftGraphHealthy("draft-layer")) {
                return false;
            }
#endif
            }

#ifdef USE_CUDA
            // The ordinary path predates persistent worker streams and keeps
            // its established boundary.  A graph body must not synchronize a
            // captured stream; graph dependencies order the following head.
            if (decodeMeta == nullptr) {
                SynchronizeDeepSeekV4TensorParallelDevices(this->deviceMap);
            }
#endif
            HcHeadReference(
                *curHiddenStates, weight["mtp.2.hc_head_fn"],
                weight["mtp.2.hc_head_scale"],
                weight["mtp.2.hc_head_base"], hc_mult, hc_eps,
                rms_norm_eps, headInput);
            if (confidenceHidden != nullptr) {
                // Confidence is trained on the unnormalized HcHead output.
                // The eager LM-head path normalizes headInput in place, so
                // preserve this small [1, block, hidden] tensor first.
                Copy(headInput, *confidenceHidden);
            }
            Data *normalizedHead = &headInput;
#ifdef USE_CUDA
            if (decodeMeta != nullptr &&
                headInput.dataDevice == DataDevice::CUDA &&
                headInput.cudaData != nullptr) {
                // Tensor-parallel RMSNorm/LM-head preparation promotes its
                // input to a replicated tensor.  Keep HcHead's output separate
                // from that mutable layout: reusing the promoted tensor as the
                // next HcHead output leaves rank-local replicas stale and makes
                // the second graph warmup diverge from eager execution.
                const int headDevice =
                    GetPointerDeviceId(headInput.cudaData);
                bool staleSamplingRoot =
                    samplingHeadRoot.cudaData == nullptr ||
                    samplingHeadRoot.dataType != headInput.dataType ||
                    samplingHeadRoot.dims != headInput.dims ||
                    GetPointerDeviceId(samplingHeadRoot.cudaData) !=
                        headDevice;
                if (staleSamplingRoot) {
                    ResetData(samplingHeadRoot);
                    samplingHeadRoot.dataType = headInput.dataType;
                    samplingHeadRoot.Resize(headInput.dims);
                    samplingHeadRoot.dataDevice = DataDevice::CUDA;
                    samplingHeadRoot.dataDeviceIds = {headDevice};
                    FastllmCudaSetDevice(headDevice);
                    samplingHeadRoot.Allocate(false);
                }
                FastllmCudaSetDevice(headDevice);
                FastllmCudaCopyFromDeviceToDevice(
                    samplingHeadRoot.cudaData, headInput.cudaData,
                    headInput.GetBytes());
                if (!MultiCudaRepeatToReplicated(
                        samplingHeadRoot,
                        (int)samplingHeadRoot.dims.size() - 1, 1,
                        samplingHeadReplicated)) {
                    draftBackboneFailure = "head-replication";
                    FastllmCudaSetThreadError();
                    return false;
                }
                RMSNorm(
                    samplingHeadReplicated,
                    weight["mtp.2.norm.weight"], rms_norm_eps,
                    samplingHeadNorm);
                normalizedHead = &samplingHeadNorm;
            } else
#endif
            {
                RMSNormReference(
                    headInput, weight["mtp.2.norm.weight"],
                    rms_norm_eps, headInput, DataType::BFLOAT16);
            }
            // Keep the LM-head output and its FP32 conversion in separate
            // persistent tensors.  In-place BF16/FP16 -> FP32 conversion must
            // allocate a second buffer before releasing the source.  Short
            // prefills do not necessarily leave a large enough idle block in
            // FastLLM's pool, so doing that allocation for the first time
            // during CUDA graph capture used to reject the capture at the
            // 128-token DSpark window boundary.
            Linear(*normalizedHead, weight["head.weight"], Data(),
                   draftWorkspace.samplingLogits);
            ToDataType(draftWorkspace.samplingLogits, baseLogits,
                       DataType::FLOAT32);
#ifdef USE_CUDA
            if (!draftGraphHealthy("head")) {
                return false;
            }
#endif
            return true;
        };

#ifdef USE_CUDA
        auto runGpuMarkovProposal = [&] (
                const Data &activeInputIds, Data &baseLogits,
                DeepSeekV4DecodeWorkspace &draftWorkspace,
                DeepSeekV4DsparkCudaGraphState &graphState,
                bool synchronizeAtEnd) -> Data* {
            if (!baseLogits.multiDeviceData ||
                !baseLogits.IsTensorParallelSharded() ||
                baseLogits.dataType != DataType::FLOAT32) {
                return nullptr;
            }
            std::vector<int> markovDevices =
                GetTensorCudaDevices(baseLogits);
            const int rootDevice = GetTensorCudaDevice(activeInputIds);
            std::vector<int> globalOffsets;
            bool supported = rootDevice >= 0 &&
                activeInputIds.cudaData != nullptr &&
                std::find(markovDevices.begin(), markovDevices.end(),
                          rootDevice) != markovDevices.end();
            for (int device : markovDevices) {
                auto localIt = baseLogits.multiDeviceDatas.find(device);
                auto rangeIt = baseLogits.tpRanges.find(device);
                Data *local = localIt == baseLogits.multiDeviceDatas.end() ?
                    nullptr : localIt->second;
                supported = supported && local != nullptr &&
                    local->cudaData != nullptr &&
                    local->dataType == DataType::FLOAT32 &&
                    !local->dims.empty() && local->dims.back() > 0 &&
                    local->Count(0) == (uint64_t)dsparkTokens *
                        local->dims.back() &&
                    rangeIt != baseLogits.tpRanges.end() &&
                    rangeIt->second.size() == 1 &&
                    rangeIt->second[0].second -
                        rangeIt->second[0].first == local->dims.back() &&
                    graphState.deviceIndex.find(device) !=
                        graphState.deviceIndex.end() &&
                    graphState.deviceIndex.at(device) != nullptr &&
                    graphState.deviceIndex.at(device)->
                        markovCandidateReadyEvents.size() >=
                            (size_t)dsparkTokens;
                if (!supported) {
                    break;
                }
                globalOffsets.push_back(rangeIt->second[0].first);
            }
            supported = supported &&
                graphState.deviceIndex.at(rootDevice)->
                    markovLatentReadyEvents.size() >=
                        (size_t)dsparkTokens;

            Data &previousId = draftWorkspace.dsparkMarkovPreviousId;
            Data &localCandidates =
                draftWorkspace.dsparkMarkovLocalCandidates;
            Data &gatheredCandidates =
                draftWorkspace.dsparkMarkovGatheredCandidates;
            Data &candidatePointers =
                draftWorkspace.dsparkMarkovCandidatePointers;
            Data &latentSignal =
                draftWorkspace.dsparkMarkovLatentSignal;
            Data &latentSeen =
                draftWorkspace.dsparkMarkovLatentSeen;
            Data &latentReplicas =
                draftWorkspace.dsparkMarkovLatentReplicas;
            Data &candidateSignals =
                draftWorkspace.dsparkMarkovCandidateSignals;
            Data &candidateSignalPointers =
                draftWorkspace.dsparkMarkovCandidateSignalPointers;
            Data &candidateSeen =
                draftWorkspace.dsparkMarkovCandidateSeen;
            Data &globalOffsetData =
                draftWorkspace.dsparkMarkovGlobalOffsets;
            Data &proposalIds = draftWorkspace.dsparkMarkovProposalIds;
            Data &proposalSignal =
                draftWorkspace.dsparkMarkovProposalSignal;
            Data &proposalSeen =
                draftWorkspace.dsparkMarkovProposalSeen;
            const Data &markovW2ForShape =
                weight["mtp.2.markov_head.markov_w2.weight"];
            const int markovHidden = markovW2ForShape.dims.size() == 2 ?
                markovW2ForShape.dims[1] : 0;
            supported = supported && markovHidden > 0;
            if (supported && !draftWorkspace.dsparkMarkovPrepared) {
                const bool allocated =
                    DeepSeekV4AllocateGraphTensor(
                        previousId, DataType::FLOAT32, {1, 1},
                        {rootDevice}, false) &&
                    DeepSeekV4AllocateGraphTensor(
                        localCandidates, DataType::INT32, {2},
                        markovDevices, false) &&
                    DeepSeekV4AllocateGraphTensor(
                        gatheredCandidates, DataType::INT32,
                        {(int)markovDevices.size(), 2},
                        {rootDevice}, false) &&
                    DeepSeekV4AllocateGraphTensor(
                        candidatePointers, DataType::INT32,
                        {(int)markovDevices.size(), 2},
                        {rootDevice}, false) &&
                    DeepSeekV4AllocateGraphTensor(
                        latentSignal, DataType::INT32, {dsparkTokens},
                        {rootDevice}, true) &&
                    DeepSeekV4AllocateGraphTensor(
                        latentSeen, DataType::INT32, {dsparkTokens},
                        markovDevices, true) &&
                    DeepSeekV4AllocateGraphTensor(
                        latentReplicas, DataType::FLOAT32, {markovHidden},
                        markovDevices, false) &&
                    DeepSeekV4AllocateGraphTensor(
                        candidateSignals, DataType::INT32,
                        {dsparkTokens}, markovDevices, true) &&
                    DeepSeekV4AllocateGraphTensor(
                        candidateSignalPointers, DataType::INT32,
                        {(int)markovDevices.size(), 2},
                        {rootDevice}, false) &&
                    DeepSeekV4AllocateGraphTensor(
                        candidateSeen, DataType::INT32,
                        {(int)markovDevices.size(), dsparkTokens},
                        {rootDevice}, true) &&
                    DeepSeekV4AllocateGraphTensor(
                        globalOffsetData, DataType::INT32,
                        {(int)markovDevices.size()},
                        {rootDevice}, false) &&
                    DeepSeekV4AllocateGraphTensor(
                        proposalIds, DataType::INT32, {dsparkTokens},
                        {rootDevice}, false) &&
                    DeepSeekV4AllocateGraphTensor(
                        proposalSignal, DataType::INT32, {1},
                        {rootDevice}, true) &&
                    DeepSeekV4AllocateGraphTensor(
                        proposalSeen, DataType::INT32, {1},
                        markovDevices, true);
                if (allocated) {
                    std::vector<uint64_t> pointerValues;
                    std::vector<uint64_t> signalPointerValues;
                    pointerValues.reserve(markovDevices.size());
                    signalPointerValues.reserve(markovDevices.size());
                    for (int device : markovDevices) {
                        pointerValues.push_back((uint64_t)(uintptr_t)
                            localCandidates.multiDeviceDatas.at(device)->
                                cudaData);
                        signalPointerValues.push_back((uint64_t)(uintptr_t)
                            candidateSignals.multiDeviceDatas.at(device)->
                                cudaData);
                    }
                    FastllmCudaSetDevice(rootDevice);
                    FastllmCudaCopyFromHostToDevice(
                        globalOffsetData.cudaData, globalOffsets.data(),
                        globalOffsets.size() * sizeof(int));
                    FastllmCudaCopyFromHostToDevice(
                        candidatePointers.cudaData, pointerValues.data(),
                        pointerValues.size() * sizeof(uint64_t));
                    FastllmCudaCopyFromHostToDevice(
                        candidateSignalPointers.cudaData,
                        signalPointerValues.data(),
                        signalPointerValues.size() * sizeof(uint64_t));
                    draftWorkspace.dsparkMarkovDevices = markovDevices;
                    draftWorkspace.dsparkMarkovOffsets = globalOffsets;
                    draftWorkspace.dsparkMarkovRootDevice = rootDevice;
                    draftWorkspace.dsparkMarkovPrepared = true;
                } else {
                    supported = false;
                }
            }
            supported = supported &&
                draftWorkspace.dsparkMarkovPrepared &&
                draftWorkspace.dsparkMarkovDevices == markovDevices &&
                draftWorkspace.dsparkMarkovOffsets == globalOffsets &&
                draftWorkspace.dsparkMarkovRootDevice == rootDevice &&
                DeepSeekV4GraphTensorMatches(
                    previousId, DataType::FLOAT32, {1, 1},
                    {rootDevice}) &&
                DeepSeekV4GraphTensorMatches(
                    localCandidates, DataType::INT32, {2},
                    markovDevices) &&
                DeepSeekV4GraphTensorMatches(
                    gatheredCandidates, DataType::INT32,
                    {(int)markovDevices.size(), 2}, {rootDevice}) &&
                DeepSeekV4GraphTensorMatches(
                    candidatePointers, DataType::INT32,
                    {(int)markovDevices.size(), 2}, {rootDevice}) &&
                DeepSeekV4GraphTensorMatches(
                    latentSignal, DataType::INT32, {dsparkTokens},
                    {rootDevice}) &&
                DeepSeekV4GraphTensorMatches(
                    latentSeen, DataType::INT32, {dsparkTokens},
                    markovDevices) &&
                DeepSeekV4GraphTensorMatches(
                    latentReplicas, DataType::FLOAT32, {markovHidden},
                    markovDevices) &&
                DeepSeekV4GraphTensorMatches(
                    candidateSignals, DataType::INT32, {dsparkTokens},
                    markovDevices) &&
                DeepSeekV4GraphTensorMatches(
                    candidateSignalPointers, DataType::INT32,
                    {(int)markovDevices.size(), 2}, {rootDevice}) &&
                DeepSeekV4GraphTensorMatches(
                    candidateSeen, DataType::INT32,
                    {(int)markovDevices.size(), dsparkTokens},
                    {rootDevice}) &&
                DeepSeekV4GraphTensorMatches(
                    globalOffsetData, DataType::INT32,
                    {(int)markovDevices.size()}, {rootDevice}) &&
                DeepSeekV4GraphTensorMatches(
                    proposalIds, DataType::INT32, {dsparkTokens},
                    {rootDevice}) &&
                DeepSeekV4GraphTensorMatches(
                    proposalSignal, DataType::INT32, {1},
                    {rootDevice}) &&
                DeepSeekV4GraphTensorMatches(
                    proposalSeen, DataType::INT32, {1},
                    markovDevices);
            if (!supported) {
                return nullptr;
            }

            Data &markovW2 =
                weight["mtp.2.markov_head.markov_w2.weight"];
            std::map<int, int> markovRatios;
            std::vector<int> configuredMarkovDevices;
            FastllmGetMulticudaDeviceAndRatio(
                configuredMarkovDevices, markovRatios, true);
            DivisionScheme markovRanges;
            if (configuredMarkovDevices == markovDevices &&
                markovW2.dims.size() == 2) {
                markovRanges = BuildMultiCudaRowSplitScheme(
                    markovW2, configuredMarkovDevices, markovRatios);
                BalanceMultiCudaDivisionSchemeByLayer(
                    markovW2.name, configuredMarkovDevices,
                    markovRanges, false);
            }
            bool peerLinearReady =
                FastllmCudaDeepSeekV4DsparkMarkovPeerAvailable() &&
                configuredMarkovDevices == markovDevices &&
                markovW2.multiDeviceData &&
                markovW2.dataType == DataType::FLOAT16 &&
                markovW2.dims.size() == 2 &&
                !baseLogits.tpGlobalDims.empty() &&
                markovW2.dims[0] == baseLogits.tpGlobalDims.back() &&
                markovRanges == baseLogits.tpRanges &&
                FastllmCudaPeerAccessInit(markovDevices);
            if (graphState.capturing && !peerLinearReady) {
                // The eager fallback records one event per GPU and makes the
                // root wait for all of them.  Those waits are valid outside a
                // graph, but would merge independently captured GPU streams.
                draftBackboneFailure = "markov-peer-access";
                return nullptr;
            }
            draftWorkspace.dsparkMarkovProposalPeerReady = false;
            for (int device : markovDevices) {
                auto weightIt = markovW2.multiDeviceDatas.find(device);
                const int localVocab =
                    baseLogits.multiDeviceDatas.at(device)->dims.back();
                peerLinearReady = peerLinearReady &&
                    weightIt != markovW2.multiDeviceDatas.end() &&
                    weightIt->second != nullptr &&
                    weightIt->second->dataDevice == DataDevice::CUDA &&
                    weightIt->second->dataType == DataType::FLOAT16 &&
                    weightIt->second->cudaData != nullptr &&
                    weightIt->second->dims ==
                        std::vector<int>({localVocab, markovW2.dims[1]});
            }
            struct ScopedMarkovPersistentAsync {
                bool previous;
                ScopedMarkovPersistentAsync() :
                    previous(MultiCudaSetPersistentAsyncDispatch(true)) {}
                ~ScopedMarkovPersistentAsync() {
                    MultiCudaSetPersistentAsyncDispatch(previous);
                }
            } persistentAsync;
            if (draftWorkspace.dsparkMarkovLatents.empty() &&
                draftWorkspace.dsparkMarkovBiasesRaw.empty() &&
                draftWorkspace.dsparkMarkovBiasesFloat.empty()) {
                draftWorkspace.dsparkMarkovLatents.resize(dsparkTokens);
                draftWorkspace.dsparkMarkovBiasesRaw.resize(dsparkTokens);
                draftWorkspace.dsparkMarkovBiasesFloat.resize(dsparkTokens);
            }
            if (draftWorkspace.dsparkMarkovLatents.size() !=
                    (size_t)dsparkTokens ||
                draftWorkspace.dsparkMarkovBiasesRaw.size() !=
                    (size_t)dsparkTokens ||
                draftWorkspace.dsparkMarkovBiasesFloat.size() !=
                    (size_t)dsparkTokens) {
                return nullptr;
            }
            std::vector<Data> &markovLatents =
                draftWorkspace.dsparkMarkovLatents;
            std::vector<Data> &markovBiasesRaw =
                draftWorkspace.dsparkMarkovBiasesRaw;
            std::vector<Data> &markovBiasesFloat =
                draftWorkspace.dsparkMarkovBiasesFloat;
            auto syncMarkovDevices = [&]() {
                int oldDevice = FastllmCudaGetDevice();
                for (int device : markovDevices) {
                    FastllmCudaSyncDevice(device);
                }
                FastllmCudaSetDevice(oldDevice);
            };
            auto markovGraphHealthy = [&](const char *stage, int step) {
                if (!graphState.capturing) {
                    return true;
                }
                int oldDevice = FastllmCudaGetDevice();
                bool healthy = true;
                for (int device : markovDevices) {
                    FastllmCudaSetDevice(device);
                    if (FastllmCudaGraphCaptureInvalidated()) {
                        healthy = false;
                        break;
                    }
                }
                FastllmCudaSetDevice(oldDevice);
                if (!healthy) {
                    draftBackboneFailure = std::string("markov-") + stage;
                    if (step >= 0) {
                        draftBackboneFailure += "[" +
                            std::to_string(step) + "]";
                    }
                }
                return healthy;
            };

            FastllmCudaSetDevice(rootDevice);
            FastllmCudaCopyFromDeviceToDevice(
                previousId.cudaData, activeInputIds.cudaData,
                sizeof(float));
            if (!markovGraphHealthy("seed-copy", -1)) {
                return nullptr;
            }
            for (int step = 0; step < dsparkTokens; step++) {
                Data &markovLatent = markovLatents[step];
                Data &markovBiasRaw = markovBiasesRaw[step];
                Data &markovBias = markovBiasesFloat[step];
                EmbeddingDirect(
                    previousId,
                    weight["mtp.2.markov_head.markov_w1.weight"],
                    markovLatent);
                if (!markovGraphHealthy("embedding", step)) {
                    return nullptr;
                }
                bool usedPeerLinear = peerLinearReady &&
                    markovLatent.dataDevice == DataDevice::CUDA &&
                    markovLatent.dataType == DataType::FLOAT32 &&
                    markovLatent.cudaData != nullptr &&
                    !markovLatent.dims.empty() &&
                    markovLatent.Count(0) ==
                        (uint64_t)markovW2.dims[1];
                if (usedPeerLinear) {
                    FastllmCudaSetDevice(rootDevice);
                    usedPeerLinear =
                        FastllmCudaDeepSeekV4DsparkMarkovSignal(
                            (uint32_t*)latentSignal.cudaData, step);
                    if (!markovGraphHealthy("latent-signal", step)) {
                        return nullptr;
                    }
                    markovBias.dataType = DataType::FLOAT32;
                    std::vector<int> markovBiasDims =
                        markovLatent.dims;
                    markovBiasDims.back() = markovW2.dims[0];
                    PrepareMultiCudaShardedData(
                        markovBias, markovDevices,
                        markovBiasDims,
                        (int)markovBiasDims.size() - 1, markovRanges);
                    for (int device : markovDevices) {
                        Data *local =
                            markovBias.multiDeviceDatas.at(device);
                        FastllmCudaSetDevice(device);
                        local->dataType = DataType::FLOAT32;
                        local->dataDevice = DataDevice::CUDA;
                        local->dataDeviceIds = {device};
                        local->Allocate(false);
                        usedPeerLinear = usedPeerLinear &&
                            local->cudaData != nullptr;
                    }
                    std::vector<char> linearReady(
                        markovDevices.size(), 0);
                    if (usedPeerLinear) {
                        RunDeepSeekV4MultiCuda(
                            markovDevices, [&](int rank, int device) {
                            Data *localSeen =
                                latentSeen.multiDeviceDatas.at(device);
                            Data *localWeight =
                                markovW2.multiDeviceDatas.at(device);
                            Data *localOutput =
                                markovBias.multiDeviceDatas.at(device);
                            Data *localLatent =
                                latentReplicas.multiDeviceDatas.at(device);
                            linearReady[rank] =
                                FastllmCudaDeepSeekV4DsparkMarkovCopyPeer(
                                    (const uint32_t*)latentSignal.cudaData,
                                    (uint32_t*)localSeen->cudaData, step,
                                    (const float*)markovLatent.cudaData,
                                    (float*)localLatent->cudaData,
                                    markovW2.dims[1]) &&
                                FastllmCudaDeepSeekV4DsparkMarkovLinearPeer(
                                    (const float*)localLatent->cudaData,
                                    *localWeight,
                                    (float*)localOutput->cudaData,
                                    markovW2.dims[1],
                                    localOutput->dims.back());
                        });
                    }
                    usedPeerLinear = usedPeerLinear && std::all_of(
                        linearReady.begin(), linearReady.end(),
                        [](char ready) { return ready != 0; });
                    if (!markovGraphHealthy("peer-linear", step)) {
                        return nullptr;
                    }
                } else {
                    Linear(
                        markovLatent, markovW2,
                        Data(), markovBiasRaw);
                    ToDataType(
                        markovBiasRaw, markovBias, DataType::FLOAT32);
                }

                bool biasLayoutReady = markovBias.multiDeviceData &&
                    markovBias.IsTensorParallelSharded() &&
                    (!peerLinearReady || usedPeerLinear);
                std::vector<char> localReady(markovDevices.size(), 0);
                if (biasLayoutReady) {
                    RunDeepSeekV4MultiCuda(
                        markovDevices, [&](int rank, int device) {
                        auto baseIt =
                            baseLogits.multiDeviceDatas.find(device);
                        auto biasIt =
                            markovBias.multiDeviceDatas.find(device);
                        auto candidateIt =
                            localCandidates.multiDeviceDatas.find(device);
                        if (baseIt == baseLogits.multiDeviceDatas.end() ||
                            biasIt == markovBias.multiDeviceDatas.end() ||
                            candidateIt ==
                                localCandidates.multiDeviceDatas.end() ||
                            baseIt->second == nullptr ||
                            biasIt->second == nullptr ||
                            candidateIt->second == nullptr) {
                            return;
                        }
                        Data *localBase = baseIt->second;
                        Data *localBias = biasIt->second;
                        const int localVocab = localBase->dims.back();
                        if (localBias->dataType != DataType::FLOAT32 ||
                            localBias->dims.empty() ||
                            localBias->dims.back() != localVocab ||
                            localBias->Count(0) !=
                                (uint64_t)localVocab) {
                            return;
                        }
                        const float *baseRow =
                            (const float*)localBase->cudaData +
                            (size_t)step * localVocab;
                        localReady[rank] =
                            FastllmCudaDeepSeekV4DsparkMarkovLocalArgmax(
                                baseRow,
                                (const float*)localBias->cudaData,
                                (int*)candidateIt->second->cudaData,
                                localVocab);
                        if (localReady[rank]) {
                            if (peerLinearReady && usedPeerLinear) {
                                Data *localSignal =
                                    candidateSignals.multiDeviceDatas.at(
                                        device);
                                localReady[rank] =
                                    FastllmCudaDeepSeekV4DsparkMarkovSignal(
                                        (uint32_t*)localSignal->cudaData,
                                        step);
                            } else {
                                FastllmCudaEventRecordCurrentThread(
                                    graphState.deviceIndex.at(device)->
                                        markovCandidateReadyEvents[step]);
                            }
                        }
                    });
                }
                supported = biasLayoutReady && std::all_of(
                    localReady.begin(), localReady.end(),
                    [](char ready) { return ready != 0; });
                if (!markovGraphHealthy("local-argmax", step)) {
                    return nullptr;
                }
                if (!supported) {
                    break;
                }

                FastllmCudaSetDevice(rootDevice);
                if (peerLinearReady && usedPeerLinear) {
                    supported =
                        FastllmCudaDeepSeekV4DsparkMarkovSelectPeer(
                            (const uint64_t*)candidatePointers.cudaData,
                            (const uint64_t*)
                                candidateSignalPointers.cudaData,
                            (uint32_t*)candidateSeen.cudaData,
                            (const int*)globalOffsetData.cudaData,
                            (int)markovDevices.size(), dsparkTokens,
                            (int*)proposalIds.cudaData,
                            (float*)previousId.cudaData, step) && supported;
                } else {
                    for (int device : markovDevices) {
                        FastllmCudaCurrentThreadStreamWaitEvent(
                            graphState.deviceIndex.at(device)->
                                markovCandidateReadyEvents[step]);
                    }
                    for (int rank = 0;
                         rank < (int)markovDevices.size(); rank++) {
                        const int device = markovDevices[rank];
                        Data *source =
                            localCandidates.multiDeviceDatas.at(device);
                        void *destination =
                            (uint8_t*)gatheredCandidates.cudaData +
                            (size_t)rank * 2 * sizeof(int);
                        supported = FastllmCudaMemcpyPeerAsyncCurrentThread(
                            rootDevice, destination, device,
                            source->cudaData, 2 * sizeof(int)) && supported;
                    }
                    supported = supported &&
                        FastllmCudaDeepSeekV4DsparkMarkovSelect(
                            (const int*)gatheredCandidates.cudaData,
                            (const int*)globalOffsetData.cudaData,
                            (int)markovDevices.size(),
                            (int*)proposalIds.cudaData,
                            (float*)previousId.cudaData, step);
                }
                if (!supported) {
                    break;
                }
                if (!markovGraphHealthy("select", step)) {
                    return nullptr;
                }
            }
            if (supported && peerLinearReady) {
                FastllmCudaSetDevice(rootDevice);
                draftWorkspace.dsparkMarkovProposalPeerReady =
                    FastllmCudaDeepSeekV4DsparkMarkovSignal(
                        (uint32_t*)proposalSignal.cudaData, 0);
                if (!markovGraphHealthy("proposal-signal", dsparkTokens)) {
                    return nullptr;
                }
            }
            // The invocation-local Linear buffers cannot return to FastLLM's
            // pool until every rank has completed the final Markov step.  This
            // is one synchronization per seven-token block, replacing seven
            // host sampling synchronizations.
            if (synchronizeAtEnd) {
                syncMarkovDevices();
            }
            return supported ? &proposalIds : nullptr;
        };
#endif

        Data eagerBaseLogits;
        Data eagerConfidenceHidden;
        Data *confidenceHiddenPtr = nullptr;
        Data *baseLogitsPtr = &eagerBaseLogits;
        Data *gpuProposalIdsPtr = nullptr;
#ifdef USE_CUDA
        void *gpuProposalReadyEvent = nullptr;
#endif
        bool draftBackboneReady = false;

#ifdef USE_CUDA
        std::shared_ptr<DeepSeekV4DsparkCudaGraphState> draftGraphState;
        std::unique_lock<std::mutex> draftGraphLock;
        std::vector<int> draftGraphDevices;
        bool fullWindow =
            (int)context.mainWindowKV.size() == dsparkLayers;
        for (int stage = 0; stage < dsparkLayers && fullWindow; stage++) {
            fullWindow = context.mainWindowKV[stage].dims.size() == 3 &&
                context.mainWindowKV[stage].dims[0] == 1 &&
                context.mainWindowKV[stage].dims[1] == window_size &&
                context.mainWindowKV[stage].dims[2] == head_dim_full;
        }
        const bool draftGraphRequested =
            !forceEager && DeepSeekV4DecodeCudaGraphEnabled() && fullWindow &&
            DeepSeekV4PreferCuda();
        if (draftGraphRequested) {
            draftGraphState = GetDeepSeekV4DsparkCudaGraphState(
                context.cudaGraphState);
            draftGraphLock =
                std::unique_lock<std::mutex>(draftGraphState->mutex);
            std::map<int, int> draftGraphRatios;
            FastllmGetMulticudaDeviceAndRatio(
                draftGraphDevices, draftGraphRatios, true);
            bool prepared = draftGraphDevices.size() > 1;
            if (prepared) {
                // Markov's sharded output path may be the first operation that
                // needs the ordinary TP NCCL domain.  Initialize it before any
                // stream capture; lazy communicator creation invalidates every
                // participating CUDA graph stream.
                prepared = FastllmInitNccl(draftGraphDevices);
            }
            bool graphSafePeerReady = true;
            if (prepared) {
                // DSpark's Markov proposal uses GPU-side peer signals to keep
                // each device in its own capture sequence.  This only needs
                // CUDA P2P; it must not depend on custom all-reduce being
                // enabled because NCCL remains a valid reduction backend.
                graphSafePeerReady =
                    FastllmCudaDeepSeekV4DsparkMarkovPeerAvailable() &&
                    FastllmCudaPeerAccessInit(draftGraphDevices);
                prepared = graphSafePeerReady;
            }
            if (!graphSafePeerReady) {
                static std::once_flag peerFallbackLog;
                std::call_once(peerFallbackLog, []() {
                    std::fprintf(
                        stderr,
                        "[Fastllm] DeepSeek-V4 DSpark draft CUDA graph "
                        "disabled: graph-safe full-mesh CUDA peer access is "
                        "unavailable; using eager NCCL fallback.\n");
                    std::fflush(stderr);
                });
            }
            if (prepared && draftGraphState->devices.empty()) {
                draftGraphState->PrepareDevices(
                    draftGraphDevices, dsparkLayers, dsparkTokens);
                draftGraphState->inputDevice =
                    GetTensorCudaDevice(weight["embed.weight"]);
                prepared = draftGraphState->inputDevice >= 0;
                if (prepared) {
                    draftGraphState->inputIds.dataType = DataType::FLOAT32;
                    draftGraphState->inputIds.Resize({1, dsparkTokens});
                    draftGraphState->inputIds.dataDevice = DataDevice::CUDA;
                    draftGraphState->inputIds.dataDeviceIds = {
                        draftGraphState->inputDevice};
                    FastllmCudaSetDevice(draftGraphState->inputDevice);
                    draftGraphState->inputIds.Allocate(false);
                    prepared =
                        draftGraphState->inputIds.cudaData != nullptr;
                }
                for (int stage = 0;
                     stage < dsparkLayers && prepared; stage++) {
                    int windowDevice = GetTensorCudaDevice(
                        context.mainWindowKV[stage]);
                    prepared = windowDevice >= 0;
                    if (!prepared) {
                        break;
                    }
                    for (int device : draftGraphDevices) {
                        if (prepared && device != windowDevice) {
                            prepared = FastllmInitNcclGraphPeer(
                                windowDevice, device);
                        }
                    }
                }
                for (int device : draftGraphDevices) {
                    if (prepared &&
                        device != draftGraphState->inputDevice) {
                        prepared = FastllmInitNcclGraphPeer(
                            draftGraphState->inputDevice, device);
                    }
                }
            }
            prepared = prepared &&
                draftGraphState->inputIds.cudaData != nullptr &&
                draftGraphState->inputIds.dims ==
                    std::vector<int>({1, dsparkTokens}) &&
                draftGraphState->pinnedMeta != nullptr &&
                draftGraphState->pinnedInputIds != nullptr &&
                context.mainWindowKV.size() == (size_t)dsparkLayers;
            const std::vector<uint64_t> directWindowSignature =
                DeepSeekV4DsparkWindowSignature(context.mainWindowKV);
            if (draftGraphState->captured) {
                prepared = prepared &&
                    draftGraphState->directWindowSignature ==
                        directWindowSignature;
            } else {
                // Warmup may promote a root-only cache to TP replicas. Record
                // the current physical layout again until capture freezes it.
                draftGraphState->directWindowSignature =
                    directWindowSignature;
            }
            int32_t draftMetaHost[kDeepSeekV4CudaGraphMetaInts] = {};
            draftMetaHost[0] = context.committedTokens;
            for (int token = 0; token < dsparkTokens; token++) {
                draftMetaHost[token + 1] = token == 0 ?
                    anchorToken : dsparkNoiseTokenId;
            }
            const DeepSeekV4DsparkDraftGpuInput *draftGpuInput =
                deepSeekV4DsparkDraftGpuInput;
            bool draftGpuInputReady = draftGpuInput == nullptr;
            if (draftGpuInput != nullptr) {
                draftGpuInputReady =
                    draftGpuInput->acceptanceResult != nullptr &&
                    draftGpuInput->acceptanceResult->dataType ==
                        DataType::INT32 &&
                    draftGpuInput->acceptanceResult->cudaData != nullptr &&
                    draftGpuInput->acceptanceResult->Count(0) >=
                        (uint64_t)(3 + 2 * draftGpuInput->rows - 1) &&
                    draftGpuInput->readySignal != nullptr &&
                    draftGpuInput->readySignal->dataType == DataType::INT32 &&
                    draftGpuInput->readySignal->cudaData != nullptr &&
                    draftGpuInput->readySeen != nullptr &&
                    draftGpuInput->readySeen->dataType == DataType::INT32 &&
                    draftGpuInput->readySeen->multiDeviceData &&
                    draftGpuInput->readySeen->IsTensorParallelReplicated() &&
                    draftGpuInput->readySeen->Count(0) >= 4 &&
                    draftGpuInput->rows == dsparkTokens + 1 &&
                    draftGpuInput->stageKV.size() ==
                        (size_t)dsparkLayers &&
                    context.mainWindowKV.size() ==
                        (size_t)dsparkLayers;
                for (int stage = 0;
                     stage < dsparkLayers && draftGpuInputReady; ++stage) {
                    draftGpuInputReady =
                        draftGpuInput->stageKV[stage] != nullptr &&
                        draftGpuInput->stageKV[stage]->dataType ==
                            DataType::BFLOAT16 &&
                        draftGpuInput->stageKV[stage]->dims ==
                            std::vector<int>({1, draftGpuInput->rows,
                                              head_dim_full}) &&
                        context.mainWindowKV[stage].dataType ==
                            DataType::BFLOAT16 &&
                        context.mainWindowKV[stage].dims ==
                            std::vector<int>({1, window_size,
                                              head_dim_full});
                }
            }
            auto stageDraftInput = [&](int device) {
                auto it = draftGraphState->deviceIndex.find(device);
                DeepSeekV4CudaGraphDeviceState *deviceState =
                    it == draftGraphState->deviceIndex.end() ?
                        nullptr : it->second;
                bool ok = deviceState != nullptr &&
                    deviceState->decodeMeta != nullptr &&
                    deviceState->decodeMeta->cudaData != nullptr;
                if (ok && draftGpuInput != nullptr) {
                    ok = draftGpuInputReady;
                }
                if (ok && draftGpuInput != nullptr) {
                    Data *localSeen = GetTensorCudaReplica(
                        *draftGpuInput->readySeen, device);
                    Data *localStage0 = GetTensorCudaReplica(
                        *draftGpuInput->stageKV[0], device);
                    Data *localStage1 = GetTensorCudaReplica(
                        *draftGpuInput->stageKV[1], device);
                    Data *localStage2 = GetTensorCudaReplica(
                        *draftGpuInput->stageKV[2], device);
                    Data *localWindow0 = GetTensorCudaReplica(
                        context.mainWindowKV[0], device);
                    Data *localWindow1 = GetTensorCudaReplica(
                        context.mainWindowKV[1], device);
                    Data *localWindow2 = GetTensorCudaReplica(
                        context.mainWindowKV[2], device);
                    float *localInputIds =
                        device == draftGraphState->inputDevice ?
                            (float*)draftGraphState->inputIds.cudaData :
                            nullptr;
                    ok = draftGpuInput->acceptanceResult != nullptr &&
                        draftGpuInput->acceptanceResult->cudaData != nullptr &&
                        draftGpuInput->readySignal != nullptr &&
                        draftGpuInput->readySignal->cudaData != nullptr &&
                        localSeen != nullptr && localStage0 != nullptr &&
                        localStage1 != nullptr && localStage2 != nullptr &&
                        localWindow0 != nullptr && localWindow1 != nullptr &&
                        localWindow2 != nullptr &&
                        FastllmCudaDeepSeekV4DsparkPrepareDraftPeer(
                            (const uint32_t*)
                                draftGpuInput->readySignal->cudaData,
                            (uint32_t*)localSeen->cudaData,
                            (const int*)
                                draftGpuInput->acceptanceResult->cudaData,
                            draftGpuInput->baseCommittedTokens,
                            localStage0->cudaData, localWindow0->cudaData,
                            localStage1->cudaData, localWindow1->cudaData,
                            localStage2->cudaData, localWindow2->cudaData,
                            draftGpuInput->rows, window_size, head_dim_full,
                            dsparkNoiseTokenId, dsparkTokens,
                            (int32_t*)deviceState->decodeMeta->cudaData,
                            localInputIds);
                } else if (ok) {
                    ok = FastllmCudaCopyFromPinnedHostToDeviceAsyncCurrentThread(
                        deviceState->decodeMeta->cudaData,
                        draftGraphState->pinnedMeta,
                        sizeof(draftMetaHost));
                    if (ok && device == draftGraphState->inputDevice) {
                        ok =
                            FastllmCudaCopyFromPinnedHostToDeviceAsyncCurrentThread(
                                draftGraphState->inputIds.cudaData,
                                draftGraphState->pinnedInputIds,
                                inputValues.size() * sizeof(float));
                    }
                }
                return ok;
            };
            if (prepared) {
                std::memcpy(draftGraphState->pinnedMeta, draftMetaHost,
                            sizeof(draftMetaHost));
                std::memcpy(draftGraphState->pinnedInputIds,
                            inputValues.data(),
                            inputValues.size() * sizeof(float));
            }
            // Eager warmup/capture executes the graph body through several
            // worker callbacks, so publish its input before entering that
            // path.  A steady-state replay can stage the same pinned input in
            // the launch callback itself: both copies and cudaGraphLaunch then
            // share the worker stream and no intermediate eight-rank
            // caller/worker event handoff is needed.
            if (prepared && !draftGraphState->captured) {
                std::vector<char> staged(draftGraphDevices.size(), 0);
                const bool previousAsync =
                    MultiCudaSetPersistentAsyncDispatch(true);
                RunDeepSeekV4MultiCuda(
                    draftGraphDevices, [&](int rank, int device) {
                    staged[rank] = stageDraftInput(device);
                });
                MultiCudaSetPersistentAsyncDispatch(previousAsync);
                prepared = std::all_of(
                    staged.begin(), staged.end(),
                    [](char state) { return state != 0; });
            }

            auto syncDraftDevices = [&]() {
                int oldDevice = FastllmCudaGetDevice();
                for (int device : draftGraphDevices) {
                    FastllmCudaSyncDevice(device);
                }
                FastllmCudaSetDevice(oldDevice);
            };
            auto launchDraftGraphs = [&]() {
                std::vector<int> launched(draftGraphDevices.size(), 0);
                std::atomic<int> launchReady{0};
                const int launchCount = (int)draftGraphDevices.size();
                const bool alignGraphLaunch = launchCount > 1;
                std::function<void(int, int)> launchOne =
                    [&](int rank, int device) {
                    auto it = draftGraphState->deviceIndex.find(device);
                    DeepSeekV4CudaGraphDeviceState *deviceState =
                        it == draftGraphState->deviceIndex.end() ?
                            nullptr : it->second;
                    bool ready = stageDraftInput(device) &&
                        deviceState != nullptr &&
                        deviceState->exec != nullptr;
                    if (alignGraphLaunch) {
                        launchReady.fetch_add(
                            1, std::memory_order_release);
                        while (launchReady.load(
                                   std::memory_order_acquire) < launchCount) {
                            std::this_thread::yield();
                        }
                    }
                    launched[rank] = ready &&
                        FastllmCudaGraphLaunch(deviceState->exec);
                    if (launched[rank] &&
                        deviceState->replayDoneEvent != nullptr) {
                        FastllmCudaEventRecordCurrentThread(
                            deviceState->replayDoneEvent);
                    }
                };
                bool previousAsync =
                    MultiCudaSetPersistentAsyncDispatch(true);
                // A deferred proposal is consumed through the graph's
                // proposal-ready signal on the same persistent worker streams.
                // Adding the generic caller-stream completion waits here made
                // ForwardDspark's acceptance-stream synchronize wait for the
                // entire next draft.  That defeated speculative lookahead and
                // left a host-sized idle gap before every verifier replay.
                const bool enqueueOnly =
                    deferGpuCopy && draftGraphState->captured;
                bool parallel = enqueueOnly &&
                    MultiCudaRunDeviceCallbacksEnqueueOnly(
                        draftGraphDevices, launchOne);
                if (!parallel) {
                    // Dedicated workers can be disabled on generic/lower-SM
                    // configurations.  EnqueueOnly guarantees that a false
                    // return submitted nothing, so the established ordered
                    // callback path remains a safe compatibility fallback.
                    parallel = MultiCudaRunDeviceCallbacks(
                        draftGraphDevices, launchOne);
                }
                if (parallel && enqueueOnly) {
                    draftGraphState->hasEnqueueOnlyReplay = true;
                }
                MultiCudaSetPersistentAsyncDispatch(previousAsync);
                return parallel && std::all_of(
                    launched.begin(), launched.end(),
                    [](int state) { return state != 0; });
            };
            auto runPersistentDraftBackbone = [&]() {
                try {
                    bool ready = runDraftBackbone(
                        draftGraphState->inputIds,
                        context.mainWindowKV,
                        &draftGraphState->decodeMeta,
                        draftGraphState->baseLogits,
                        draftGraphState->workspace.get(), nullptr);
                    if (ready && draftGraphState->workspace) {
                        gpuProposalIdsPtr = runGpuMarkovProposal(
                            draftGraphState->inputIds,
                            draftGraphState->baseLogits,
                            *draftGraphState->workspace,
                            *draftGraphState,
                            !draftGraphState->capturing);
                        ready = gpuProposalIdsPtr != nullptr;
                    }
                    return ready;
                } catch (const char *message) {
                    draftBackboneFailure = message == nullptr ?
                        "unknown FastLLM exception" : message;
                } catch (const std::exception &exception) {
                    draftBackboneFailure = exception.what();
                } catch (...) {
                    draftBackboneFailure = "unknown C++ exception";
                }
                FastllmCudaSetThreadError();
                return false;
            };

            if (!prepared || draftGraphState->disabled) {
                draftGraphState->disabled = true;
            } else if (draftGraphState->captured) {
                if (launchDraftGraphs()) {
                    baseLogitsPtr = &draftGraphState->baseLogits;
                    draftBackboneReady = true;
                    if (draftGraphState->workspace) {
                        gpuProposalIdsPtr =
                            &draftGraphState->workspace->dsparkMarkovProposalIds;
                        const int proposalDevice = GetTensorCudaDevice(
                            *gpuProposalIdsPtr);
                        auto readyIt = draftGraphState->deviceIndex.find(
                            proposalDevice);
                        if (readyIt !=
                                draftGraphState->deviceIndex.end() &&
                            readyIt->second != nullptr) {
                            gpuProposalReadyEvent =
                                readyIt->second->replayDoneEvent;
                        }
                    }
                } else {
                    draftGraphState->DestroyCapturedGraph();
                    draftGraphState->disabled = true;
                }
            } else if (draftGraphState->warmupRounds < 2) {
                syncDraftDevices();
                FastllmCudaClearThreadError();
                FastllmCudaClearGraphError();
                draftBackboneReady = runPersistentDraftBackbone();
                syncDraftDevices();
                if (!draftBackboneReady ||
                    FastllmCudaGetThreadError() ||
                    FastllmCudaGetGraphError()) {
                    draftGraphState->disabled = true;
                    draftBackboneReady = false;
                } else {
                    draftGraphState->warmed = true;
                    draftGraphState->warmupRounds++;
                    baseLogitsPtr = &draftGraphState->baseLogits;
                }
            } else {
                syncDraftDevices();
                bool captureOk = true;
                const char *failureStage = nullptr;
                for (auto &deviceState : draftGraphState->devices) {
                    FastllmCudaSetDevice(deviceState->device);
                    if (!FastllmCudaGraphPrepareCaptureDevice()) {
                        captureOk = false;
                        failureStage = "prepare capture devices";
                        break;
                    }
                }
                if (captureOk && !FastllmCudaGraphMemoryPoolBegin()) {
                    captureOk = false;
                    failureStage = "workspace reservation";
                }
                int begunCaptures = 0;
                if (captureOk) {
                    for (auto &deviceState : draftGraphState->devices) {
                        FastllmCudaSetDevice(deviceState->device);
                        if (!FastllmCudaGraphBeginCapture()) {
                            captureOk = false;
                            failureStage = "begin capture";
                            break;
                        }
                        begunCaptures++;
                    }
                }
                std::vector<void*> workerStartEvents;
                std::vector<void*> workerEndEvents;
                for (auto &deviceState : draftGraphState->devices) {
                    workerStartEvents.push_back(
                        deviceState->workerStartEvent);
                    workerEndEvents.push_back(deviceState->workerEndEvent);
                }
                bool workersJoined = false;
                if (captureOk) {
                    for (auto &deviceState : draftGraphState->devices) {
                        FastllmCudaSetDevice(deviceState->device);
                        FastllmCudaEventRecordCurrentThread(
                            deviceState->workerStartEvent);
                    }
                    workersJoined = MultiCudaGraphWorkersWaitEvents(
                        draftGraphDevices, workerStartEvents);
                    if (!workersJoined) {
                        captureOk = false;
                        failureStage = "join persistent workers";
                    }
                }
                if (captureOk) {
                    FastllmCudaClearThreadError();
                    draftGraphState->capturing = true;
                    captureOk = runPersistentDraftBackbone();
                    draftGraphState->capturing = false;
                    if (!captureOk || FastllmCudaGetThreadError() ||
                        FastllmCudaGetGraphError()) {
                        captureOk = false;
                        failureStage = "captured draft body";
                    }
                }
                bool captureInvalidated = false;
                if (begunCaptures ==
                    (int)draftGraphState->devices.size()) {
                    for (auto &deviceState : draftGraphState->devices) {
                        FastllmCudaSetDevice(deviceState->device);
                        captureInvalidated |=
                            FastllmCudaGraphCaptureInvalidated();
                    }
                }
                if (captureInvalidated) {
                    captureOk = false;
                    if (failureStage == nullptr) {
                        failureStage = "invalidated captured draft body";
                    }
                }
                if (workersJoined && !captureInvalidated) {
                    if (!MultiCudaGraphWorkersRecordEvents(
                            draftGraphDevices, workerEndEvents)) {
                        captureOk = false;
                        failureStage = "rejoin persistent workers";
                    }
                    for (auto &deviceState : draftGraphState->devices) {
                        FastllmCudaSetDevice(deviceState->device);
                        FastllmCudaCurrentThreadStreamWaitEvent(
                            deviceState->workerEndEvent);
                    }
                }
                if (!captureInvalidated && begunCaptures ==
                    (int)draftGraphState->devices.size()) {
                    for (auto &deviceState : draftGraphState->devices) {
                        FastllmCudaSetDevice(deviceState->device);
                        if (FastllmCudaGraphCaptureInvalidated()) {
                            captureOk = false;
                            failureStage = "invalidated capture";
                        }
                    }
                }
                bool endOk = begunCaptures ==
                    (int)draftGraphState->devices.size();
                for (int index = 0; index < begunCaptures; index++) {
                    auto &deviceState = draftGraphState->devices[index];
                    FastllmCudaSetDevice(deviceState->device);
                    void *capturedGraph = nullptr;
                    bool oneEndOk = FastllmCudaGraphEndCapture(
                        &capturedGraph) && capturedGraph != nullptr;
                    if (oneEndOk) {
                        deviceState->graph = capturedGraph;
                    } else if (capturedGraph != nullptr) {
                        FastllmCudaGraphDestroy(capturedGraph);
                    }
                    endOk &= oneEndOk;
                }
                if (!endOk) {
                    captureOk = false;
                    if (failureStage == nullptr) {
                        failureStage = "end capture";
                    }
                }
                if (captureOk) {
                    captureOk = FastllmCudaGraphMemoryPoolEnd(
                        draftGraphState->reservedPointers);
                    if (!captureOk) {
                        failureStage = "pin captured workspace";
                    }
                } else {
                    FastllmCudaGraphMemoryPoolAbort();
                }
                if (captureOk) {
                    for (auto &deviceState : draftGraphState->devices) {
                        FastllmCudaSetDevice(deviceState->device);
                        if (!FastllmCudaGraphInstantiate(
                                deviceState->graph,
                                &deviceState->exec) ||
                            deviceState->exec == nullptr) {
                            captureOk = false;
                            failureStage = "instantiate";
                            break;
                        }
                    }
                }
                if (captureOk) {
                    draftGraphState->directWindowSignature =
                        DeepSeekV4DsparkWindowSignature(
                            context.mainWindowKV);
                    draftGraphState->captured = true;
                    captureOk = launchDraftGraphs();
                    if (!captureOk) {
                        failureStage = "first launch";
                    }
                }
                if (captureOk) {
                    syncDraftDevices();
                    baseLogitsPtr = &draftGraphState->baseLogits;
                    draftBackboneReady = true;
                } else {
                    draftGraphState->capturing = false;
                    syncDraftDevices();
                    draftGraphState->DestroyCapturedGraph();
                    draftGraphState->disabled = true;
                    std::fprintf(
                        stderr,
                        "[Fastllm] DeepSeek-V4 DSpark draft CUDA graph "
                        "disabled at %s (body=%s, thread_error=%d, "
                        "graph_error=%d): %s\n",
                        failureStage == nullptr ? "unknown stage" :
                            failureStage,
                        draftBackboneFailure.empty() ? "unknown" :
                            draftBackboneFailure.c_str(),
                        FastllmCudaGetThreadError() ? 1 : 0,
                        FastllmCudaGetGraphError() ? 1 : 0,
                        FastllmCudaGraphLastError());
                    std::fflush(stderr);
                }
            }
        }
#endif

        if (!draftBackboneReady) {
#ifdef USE_CUDA
            FastllmCudaClearThreadError();
#endif
            AssertInFastLLM(
                runDraftBackbone(
                    inputIds, context.mainWindowKV, nullptr,
                    eagerBaseLogits, nullptr, &eagerConfidenceHidden),
                "DSpark draft backbone failed.");
            baseLogitsPtr = &eagerBaseLogits;
            confidenceHiddenPtr = &eagerConfidenceHidden;
        }
#ifdef USE_CUDA
        if (draftBackboneReady && draftGraphState != nullptr &&
            draftGraphState->workspace != nullptr) {
            // The persistent path keeps HcHead output separate from its
            // normalized LM-head workspace, so no additional graph node or
            // copy is needed.
            confidenceHiddenPtr = &draftGraphState->workspace->headInput;
        }
#endif
        Data &baseLogits = *baseLogitsPtr;

        DeepSeekV4DsparkProposal proposal;
        proposal.tokens.reserve(dsparkTokens);
        auto runHostMarkov = [&](Data &hostBaseLogits) {
            std::vector<int> tokens;
            tokens.reserve(dsparkTokens);
            int previousToken = anchorToken;
            for (int step = 0; step < dsparkTokens; step++) {
                Data stepLogits;
                Split(hostBaseLogits, 1, step, step + 1, stepLogits);
                Data previousIds(
                    DataType::FLOAT32, {1, 1}, {(float)previousToken});
                Data markovLatent, markovBias;
                EmbeddingDirect(
                    previousIds,
                    weight["mtp.2.markov_head.markov_w1.weight"],
                    markovLatent);
                Linear(
                    markovLatent,
                    weight["mtp.2.markov_head.markov_w2.weight"],
                    Data(), markovBias);
                ToDataType(markovBias, DataType::FLOAT32);
                AddTo(stepLogits, markovBias);

                GenerationConfig greedy;
                greedy.do_sample = false;
                greedy.top_k = 1;
                greedy.repeat_penalty = 1.0f;
                greedy.output_token_least = 0;
                greedy.output_logits = false;
                std::vector<GenerationConfig> configs{greedy};
                std::vector<int> seqLens{1};
                LastTokensManager emptyLastTokens(1, greedy.last_n);
                std::vector<std::pair<Data*, Data*> > emptyPast;
                std::vector<int> sampled;
                LLMSamplingBlock(
                    this, &stepLogits, &weight["norm.weight"],
                    &weight["head.weight"], rms_norm_eps, 1, true,
                    seqLens, emptyPast, configs, emptyLastTokens, nullptr,
                    sampled, &stepLogits);
                AssertInFastLLM(sampled.size() == 1,
                                "DSpark draft sampling failed.");
                previousToken = sampled[0];
                tokens.push_back(previousToken);
            }
            return tokens;
        };
        auto populateProposalConfidence = [&] (
                DeepSeekV4DsparkProposal &activeProposal) {
            if (dsparkConfidenceThreshold <= 0.0f ||
                confidenceHiddenPtr == nullptr) {
                activeProposal.confidence.assign(dsparkTokens, 1.0f);
                return;
            }
            AssertInFastLLM(
                (int)activeProposal.tokens.size() == dsparkTokens &&
                confidenceHiddenPtr->dims ==
                    std::vector<int>({1, dsparkTokens, embed_dim}),
                "DSpark confidence input has an invalid shape.");

            // The confidence head consumes the unnormalized draft backbone
            // state and the Markov embedding of [anchor, draft[:-1]].  Batch
            // all positions into one tiny projection so scheduling adds a
            // single device-to-host synchronization per proposal.
            std::vector<float> previousTokens(dsparkTokens);
            previousTokens[0] = (float)anchorToken;
            for (int step = 1; step < dsparkTokens; ++step) {
                previousTokens[step] =
                    (float)activeProposal.tokens[step - 1];
            }
            Data previousIds(
                DataType::FLOAT32, {1, dsparkTokens}, previousTokens);
            Data markovEmbeddings;
            EmbeddingDirect(
                previousIds,
                weight["mtp.2.markov_head.markov_w1.weight"],
                markovEmbeddings);
            Data confidenceHidden, confidenceMarkov;
            Copy(*confidenceHiddenPtr, confidenceHidden);
            Copy(markovEmbeddings, confidenceMarkov);
            ToDataType(confidenceHidden, DataType::FLOAT32);
            ToDataType(confidenceMarkov, DataType::FLOAT32);
            Data confidenceFeatures;
            Cat(confidenceHidden, confidenceMarkov, -1,
                confidenceFeatures);
            Data confidenceLogits;
            Linear(
                confidenceFeatures,
                weight["mtp.2.confidence_head.proj.weight"],
                Data(), confidenceLogits);
            ToDataType(confidenceLogits, DataType::FLOAT32);
            confidenceLogits.ToDevice(DataDevice::CPU);
            AssertInFastLLM(
                confidenceLogits.dims ==
                    std::vector<int>({1, dsparkTokens, 1}),
                "DSpark confidence head output has an invalid shape.");
            activeProposal.confidence.resize(dsparkTokens);
            const float *confidenceData =
                (const float*)confidenceLogits.cpuData;
            for (int step = 0; step < dsparkTokens; ++step) {
                const float value = confidenceData[step];
                const float probability = value >= 0.0f ?
                    1.0f / (1.0f + std::exp(-value)) :
                    std::exp(value) / (1.0f + std::exp(value));
                AssertInFastLLM(
                    std::isfinite(probability),
                    "DSpark confidence head produced NaN or Inf.");
                activeProposal.confidence[step] = probability;
            }
        };
#ifdef USE_CUDA
        if (gpuProposalIdsPtr != nullptr &&
            gpuProposalIdsPtr->dataDevice == DataDevice::CUDA &&
            gpuProposalIdsPtr->cudaData != nullptr &&
            gpuProposalIdsPtr->dataType == DataType::INT32 &&
            gpuProposalIdsPtr->Count(0) >= (uint64_t)dsparkTokens) {
            if (deferGpuCopy && draftGraphState != nullptr &&
                draftGraphState->captured &&
                draftGraphState->workspace != nullptr &&
                draftGraphState->workspace->
                    dsparkMarkovProposalPeerReady) {
                Data &readySignal = draftGraphState->workspace->
                    dsparkMarkovProposalSignal;
                Data &readySeen = draftGraphState->workspace->
                    dsparkMarkovProposalSeen;
                if (readySignal.dataDevice == DataDevice::CUDA &&
                    readySignal.cudaData != nullptr &&
                    readySignal.dataType == DataType::INT32 &&
                    readySeen.multiDeviceData &&
                    readySeen.IsTensorParallelReplicated()) {
                    proposal.gpuTokens = gpuProposalIdsPtr;
                    proposal.gpuReadySignal = &readySignal;
                    proposal.gpuReadySeen = &readySeen;
                    proposal.gpuDeferred = true;
                    // Deferred graph proposals remain fixed-shape.  Their
                    // confidence tensor is intentionally not synchronized to
                    // the host, preserving the existing overlapped pipeline.
                    proposal.confidence.assign(dsparkTokens, 1.0f);
                    return proposal;
                }
            }
            proposal.tokens.resize(dsparkTokens);
            const int proposalDevice =
                GetPointerDeviceId(gpuProposalIdsPtr->cudaData);
            FastllmCudaSetDevice(proposalDevice);
            bool copiedAsync = false;
            if (gpuProposalReadyEvent != nullptr) {
                FastllmCudaCurrentThreadStreamWaitEvent(
                    gpuProposalReadyEvent);
                copiedAsync =
                    FastllmCudaCopyFromDeviceToHostAsyncCurrentThread(
                        proposal.tokens.data(),
                        gpuProposalIdsPtr->cudaData,
                        proposal.tokens.size() * sizeof(int));
                if (copiedAsync) {
                    FastllmCudaSyncCurrentThreadStream();
                }
            }
            if (!copiedAsync) {
                FastllmCudaCopyFromDeviceToHost(
                    proposal.tokens.data(), gpuProposalIdsPtr->cudaData,
                    proposal.tokens.size() * sizeof(int));
            }
            populateProposalConfidence(proposal);
            return proposal;
        }
#endif
        proposal.tokens = runHostMarkov(baseLogits);
        populateProposalConfidence(proposal);
        return proposal;
    }

    int DeepSeekV4Model::SelectDsparkVerifyDrafts(
            const DeepSeekV4DsparkProposal &proposal,
            const DeepSeekV4DsparkContext &context,
            bool *preferTargetOnly) const {
        (void)context;
        if (preferTargetOnly != nullptr) {
            *preferTargetOnly = false;
        }
        if (proposal.gpuDeferred) {
            return dsparkTokens;
        }
        AssertInFastLLM(
            (int)proposal.tokens.size() == dsparkTokens &&
            (int)proposal.confidence.size() == dsparkTokens,
            "DSpark proposal has an invalid confidence length.");
        // Threshold zero is the exact fixed-block compatibility mode used by
        // acceptance and performance regressions.
        if (dsparkConfidenceThreshold <= 0.0f) {
            return dsparkTokens;
        }

        // Start with a few complete blocks.  Besides warming the verifier,
        // these rounds provide an unbiased prefix-survival sample for online
        // confidence calibration.  A 32-token application warmup normally
        // covers these five full-block probes plus two target-only probes.
        if (dsparkValidationCount.load(std::memory_order_relaxed) <
                DEEPSEEK_V4_DSPARK_CALIBRATION_ROUNDS - 2) {
            return dsparkTokens;
        }

        std::vector<double> calibratedSurvival(dsparkTokens, 0.0);
        double rawSurvival = 1.0;
        double previousSurvival = 1.0;
        int confidenceLimit = dsparkTokens;
        auto clampProbability = [](double value) {
            return std::max(1.0e-4, std::min(1.0 - 1.0e-4, value));
        };
        for (int position = 0; position < dsparkTokens; ++position) {
            rawSurvival *= clampProbability(
                proposal.confidence[position]);
            double survival = rawSurvival;
            const long long attempts =
                dsparkDraftPositionAttempts[position].load(
                    std::memory_order_relaxed);
            const long long accepted =
                dsparkDraftPositionAccepts[position].load(
                    std::memory_order_relaxed);
            const long long predictedScaled =
                dsparkDraftPositionPredictedSurvival[position].load(
                    std::memory_order_relaxed);
            if (attempts > 0 && predictedScaled > 0) {
                const double observed = clampProbability(
                    (double)accepted / (double)attempts);
                const double averagePrediction = clampProbability(
                    (double)predictedScaled /
                    (1000000.0 * (double)attempts));
                const double ratioCalibrated = clampProbability(
                    rawSurvival * observed / averagePrediction);
                // Retain proposal-specific confidence while smoothly moving
                // toward the observed curve.  Four virtual raw samples keep
                // the short warmup from overreacting.
                const double weight = (double)attempts /
                    ((double)attempts + 4.0);
                survival = rawSurvival +
                    weight * (ratioCalibrated - rawSurvival);
            }
            survival = std::min(previousSurvival,
                                clampProbability(survival));
            calibratedSurvival[position] = survival;
            const double conditional = survival / previousSurvival;
            if (conditional < dsparkConfidenceThreshold &&
                confidenceLimit == dsparkTokens) {
                confidenceLimit = position;
            }
            previousSurvival = survival;
        }

        const long long draftUs = dsparkDraftBestUs.load(
            std::memory_order_relaxed);
        struct VerifyPoint {
            int drafts;
            double milliseconds;
        };
        std::vector<VerifyPoint> observed;
        observed.reserve(2);
        for (int drafts = dsparkTokens;
             drafts >= 0 && observed.size() < 2; --drafts) {
            const long long verifyUs = dsparkVerifyBestUs[drafts].load(
                std::memory_order_relaxed);
            // Kernel compilation and graph preparation can make the first
            // encounter tens of seconds long.  Such startup work is not the
            // steady SPS curve the scheduler is meant to optimize.
            if (verifyUs > 0 && verifyUs < 5000000) {
                observed.push_back(
                    {drafts, (double)verifyUs / 1000.0});
            }
        }
        // Before the warmup request has populated a throughput curve, use the
        // calibrated confidence limit.  Once prefix timing is available, fit
        // the NUMA verifier's marginal token cost without cold compilation.
        if (draftUs <= 0 || observed.empty()) {
            return confidenceLimit;
        }

        const double draftMs = (double)draftUs / 1000.0;
        double slopeMs = 0.0;
        double interceptMs = 0.0;
        bool linearCurve = observed.size() >= 2 &&
            observed[0].drafts != observed[1].drafts;
        if (linearCurve) {
            slopeMs =
                (observed[0].milliseconds - observed[1].milliseconds) /
                (double)(observed[0].drafts - observed[1].drafts);
            interceptMs = observed[0].milliseconds -
                slopeMs * observed[0].drafts;
            linearCurve = slopeMs > 0.0 && interceptMs > 0.0 &&
                std::isfinite(slopeMs) && std::isfinite(interceptMs);
        }
        auto estimateTargetMs = [&](int drafts) {
            if (linearCurve) {
                return interceptMs + slopeMs * drafts;
            }
            // One observed point is enough for a conservative bootstrap.  A
            // NUMA MoE verifier is dominated by expert-weight traffic and is
            // close to linear; 0.7 is the measured marginal/full-decode ratio
            // and is refined as soon as the second point arrives.
            const VerifyPoint &reference = observed[0];
            return reference.milliseconds *
                (1.0 + 0.7 * drafts) /
                (1.0 + 0.7 * reference.drafts);
        };

        double expectedCommit = 1.0;
        double bestThroughput = 0.0;
        int bestDrafts = 0;
        for (int drafts = 1; drafts <= confidenceLimit; ++drafts) {
            expectedCommit += calibratedSurvival[drafts - 1];
            const double throughput = expectedCommit /
                (draftMs + estimateTargetMs(drafts));
            if (throughput > bestThroughput) {
                bestThroughput = throughput;
                bestDrafts = drafts;
            }
        }

        // Compare speculation with an actual one-row verifier observation
        // when available.  Otherwise the fitted curve's intercept is the best
        // target-only estimate.  The recommendation is returned separately
        // because the current round has already paid its draft cost.
        const long long targetOnlyUs = dsparkVerifyBestUs[0].load(
            std::memory_order_relaxed);
        const double targetOnlyMs = targetOnlyUs > 0 ?
            (double)targetOnlyUs / 1000.0 : estimateTargetMs(0);
        const double targetOnlyThroughput = 1.0 / targetOnlyMs;
        if (preferTargetOnly != nullptr) {
            *preferTargetOnly = bestDrafts == 0 ||
                bestThroughput <= targetOnlyThroughput;
        }
        // The draft cost is already sunk in the current probe, so verify the
        // best immediate prefix even when future rounds should run target
        // only.  This avoids turning a probe into draft + one-token decode.
        return bestDrafts;
    }

    int DeepSeekV4Model::ForwardDspark(
            const Data &inputIds,
            std::vector<std::pair<Data, Data> > &pastKeyValues,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens,
            std::vector<float> *retLogits) {
        std::shared_ptr<DeepSeekV4RequestState> requestState =
            GetRequestState(pastKeyValues);
        if (!requestState) {
            // Model warmup and direct synchronous inference do not necessarily
            // create a ResponseContext before their first Forward() call.  Use
            // the same KV-vector identity as the scheduler lifecycle hooks so
            // both paths share the exact same request-local caches.
            const void *key = (const void*)&pastKeyValues;
            std::lock_guard<std::mutex> guard(this->requestStateMutex);
            auto &slot = this->requestStates[key];
            if (!slot) {
                slot = std::make_shared<DeepSeekV4RequestState>();
            }
            requestState = slot;
            if (!pastKeyValues.empty()) {
                this->requestStatesByFirstKey[
                    (const void*)&pastKeyValues[0].first] = requestState;
            }
        }
        if (!requestState->dspark) {
            requestState->dspark =
                std::make_shared<DeepSeekV4DsparkContext>();
        }
        DeepSeekV4DsparkContext &context = *requestState->dspark;
        std::vector<int> tokenIds = ReadTokenIds(inputIds);

        if (!context.pending.empty()) {
            AssertInFastLLM(
                tokenIds.size() == 1 &&
                tokenIds[0] == context.pending.front().expectedInput,
                "DSpark pending output stream is out of sync.");
            const int output = context.pending.front().outputToken;
            context.pending.pop_front();
            return output;
        }

        int targetStart = 0;
        if (!requestState->decodeLayerCaches.empty()) {
            targetStart = requestState->decodeLayerCaches[0].totalLen;
        }
        if (!context.initialized || tokenIds.size() != 1) {
            AssertInFastLLM(
                !context.initialized ||
                targetStart == context.committedTokens,
                "DSpark target and draft context lengths diverged.");
            AssertInFastLLM(
                context.initialized || targetStart == 0,
                "DSpark cannot restore a target-only prefix cache; disable "
                "prefix caching for this request.");
            const bool prefixCacheEnabled =
                this->saveHistoryChat && !DeepSeekV4PrefixCacheDisabled();
            auto publishPromptSnapshot = [&]() {
                // DSpark may verify several decode tokens ahead of the HTTP
                // response.  Publish snapshots while target and draft state
                // still describe an exact prompt prefix, rather than waiting
                // for request teardown where allTokens can lag the caches.
                if (!prefixCacheEnabled ||
                    context.committedTokens <=
                        this->deepseekV4HistoryCacheManager.logicalBlockSize) {
                    return;
                }
#ifdef USE_CUDA
                if (DeepSeekV4PreferCuda()) {
                    SynchronizeDeepSeekV4TensorParallelDevices(
                        this->deviceMap);
                }
#endif
                this->RecordHistorySnapshot(
                    context.historyTokens, context.committedTokens,
                    requestState->decodeLayerCaches, &context);
            };
            auto runPrefillChunk = [&](int offset, int count,
                                       bool finalChunk) {
                std::vector<int> chunk(
                    tokenIds.begin() + offset,
                    tokenIds.begin() + offset + count);
                DeepSeekV4DsparkTargetCapture capture;
                std::vector<int> chunkResult = RunDsparkTarget(
                    chunk, context.committedTokens, pastKeyValues,
                    generationConfig, lastTokens,
                    finalChunk ? retLogits : nullptr, &capture);
                AppendDsparkTargetHidden(capture, count, context);
                context.historyTokens.insert(
                    context.historyTokens.end(), chunk.begin(), chunk.end());
                context.initialized = true;
                publishPromptSnapshot();
                return chunkResult;
            };

            // Chat templates usually replace the previous request's final
            // assistant-generation marker, so a snapshot keyed only by the
            // complete prompt often misses on the next turn.  Materialize the
            // last shared 256-token page boundary with synchronized DSpark
            // state, then finish the short tail.  This keeps DSpark hidden
            // capture complete for each target invocation and gives the next
            // request a stable Qwen-style page prefix to match.
            const int finalPromptLen = targetStart + (int)tokenIds.size();
            const int pageSize =
                this->deepseekV4HistoryCacheManager.logicalBlockSize;
            const int lastPageBoundary = finalPromptLen / pageSize * pageSize;
            const int firstChunkTokens = lastPageBoundary - targetStart;
            std::vector<int> result;
            if (prefixCacheEnabled && firstChunkTokens > 0 &&
                firstChunkTokens < (int)tokenIds.size()) {
                runPrefillChunk(0, firstChunkTokens, false);
                result = runPrefillChunk(
                    firstChunkTokens,
                    (int)tokenIds.size() - firstChunkTokens, true);
            } else {
                result = runPrefillChunk(
                    0, (int)tokenIds.size(), true);
            }
            AssertInFastLLM(result.size() == 1,
                            "DSpark target prefill sampling failed.");
            return result[0];
        }

        const int anchorToken = tokenIds[0];
        const int oldTokens = context.committedTokens;
        using DsparkProfileClock = std::chrono::steady_clock;
        const int dsparkTargetOnlyCooldownRounds = std::max(
            0, std::min(
                1024,
                EnvInt(
                    "FASTLLM_DSV4_DSPARK_TARGET_ONLY_COOLDOWN_ROUNDS",
                    0)));
        const auto dsparkRoundBegin = DsparkProfileClock::now();
        double dsparkDraftSubmitMs = 0.0;
        double dsparkTargetSubmitMs = 0.0;
        double dsparkTargetFinishMs = 0.0;
        auto dsparkElapsedMs = [](const auto &begin, const auto &end) {
            return std::chrono::duration<double, std::milli>(
                end - begin).count();
        };
        if (!dsparkLogPrinted.exchange(true)) {
            std::printf(
                "[DeepSeek-V4 DSpark] enabled: layers=%d, "
                "drafts_per_step=%d, acceptance=exact, "
                "log_interval=%d validations.\n",
                dsparkLayers, dsparkTokens,
                DEEPSEEK_V4_DSPARK_LOG_INTERVAL);
            std::fflush(stdout);
        }
#ifdef USE_CUDA
        bool deferGpuProposal = false;
        if (requestState->cudaGraphState != nullptr) {
            DeepSeekV4CudaGraphState *targetGraphState =
                (DeepSeekV4CudaGraphState*)
                    requestState->cudaGraphState.get();
            std::lock_guard<std::mutex> graphGuard(
                targetGraphState->mutex);
            deferGpuProposal = targetGraphState->captured &&
                !targetGraphState->disabled &&
                targetGraphState->graphMaxTokens >=
                    targetStart + dsparkTokens + 1 &&
                targetGraphState->inputDevice >= 0 &&
                targetGraphState->workspace != nullptr &&
                targetGraphState->pinnedMeta != nullptr &&
                targetGraphState->pinnedInputIds != nullptr &&
                targetGraphState->inputIds.dataType == DataType::FLOAT32 &&
                targetGraphState->inputIds.dims ==
                    std::vector<int>({1, dsparkTokens + 1}) &&
                targetGraphState->inputIds.cudaData != nullptr &&
                targetGraphState->graphInputIds.dataType == DataType::FLOAT32 &&
                targetGraphState->graphInputIds.dims ==
                    std::vector<int>({1, dsparkTokens + 1}) &&
                targetGraphState->graphInputIds.cudaData != nullptr &&
                !targetGraphState->devices.empty() &&
                !targetGraphState->launchOrder.empty();
            for (const auto &deviceState : targetGraphState->devices) {
                deferGpuProposal = deferGpuProposal &&
                    deviceState != nullptr && deviceState->device >= 0 &&
                    deviceState->exec != nullptr &&
                    deviceState->decodeMeta != nullptr &&
                    deviceState->decodeMeta->cudaData != nullptr;
            }
        }
#endif
        DeepSeekV4DsparkProposal proposal;
        // Reserve the last two calibration rounds for genuine no-draft target
        // decode.  The first compiles the one-row verifier shape; the second
        // supplies its steady baseline for the throughput comparison.
        const long long completedCalibrations =
            dsparkValidationCount.load(std::memory_order_relaxed);
        const bool calibrationTargetOnlyRound =
            dsparkConfidenceThreshold > 0.0f &&
            completedCalibrations >=
                DEEPSEEK_V4_DSPARK_CALIBRATION_ROUNDS - 2 &&
            completedCalibrations <
                DEEPSEEK_V4_DSPARK_CALIBRATION_ROUNDS;
        const bool targetOnlyRound = calibrationTargetOnlyRound ||
            context.targetOnlyRoundsRemaining > 0;
        if (targetOnlyRound && !calibrationTargetOnlyRound) {
            context.targetOnlyRoundsRemaining--;
        }
        bool usedPrefetchedProposal = !targetOnlyRound &&
            context.prefetchedProposalReady &&
            context.prefetchedAnchorToken == anchorToken &&
            context.prefetchedCommittedTokens == oldTokens;
        if (usedPrefetchedProposal) {
            proposal = std::move(context.prefetchedProposal);
        }
        context.prefetchedProposal = DeepSeekV4DsparkProposal();
        context.prefetchedAnchorToken = -1;
        context.prefetchedCommittedTokens = -1;
        context.prefetchedProposalReady = false;
        if (targetOnlyRound) {
            proposal.tokens.assign(dsparkTokens, dsparkNoiseTokenId);
            proposal.confidence.assign(dsparkTokens, 0.0f);
        } else if (!usedPrefetchedProposal) {
            const auto draftBegin = DsparkProfileClock::now();
            proposal = RunDsparkDraft(
                anchorToken, context, false,
#ifdef USE_CUDA
                deferGpuProposal
#else
                false
#endif
            );
            dsparkDraftSubmitMs = dsparkElapsedMs(
                draftBegin, DsparkProfileClock::now());
        }
        AssertInFastLLM(
            proposal.gpuDeferred ||
                ((int)proposal.tokens.size() == dsparkTokens &&
                 (int)proposal.confidence.size() == dsparkTokens),
            "DSpark proposal has an invalid length.");
        bool preferTargetOnly = false;
        const int verifyDrafts = targetOnlyRound ? 0 :
            SelectDsparkVerifyDrafts(
                proposal, context, &preferTargetOnly);
        // Confidence and acceptance can change at every token as a response
        // moves between prose and code, so retry on the next token by default.
        // An explicitly configured cooldown performs target-only rounds
        // without paying for draft construction.
        if (!targetOnlyRound && preferTargetOnly) {
            context.targetOnlyRoundsRemaining =
                dsparkTargetOnlyCooldownRounds;
        } else if (!targetOnlyRound && verifyDrafts > 0) {
            context.targetOnlyRoundsRemaining = 0;
        }
        const int verifyTokens = verifyDrafts + 1;
        AssertInFastLLM(
            verifyDrafts >= 0 && verifyDrafts <= dsparkTokens,
            "DSpark confidence scheduler returned an invalid prefix.");

        std::vector<int> verifyIds;
        verifyIds.reserve(verifyTokens);
        verifyIds.push_back(anchorToken);
        if (proposal.gpuDeferred) {
            verifyIds.insert(
                verifyIds.end(), verifyDrafts, dsparkNoiseTokenId);
        } else {
            verifyIds.insert(verifyIds.end(), proposal.tokens.begin(),
                             proposal.tokens.begin() + verifyDrafts);
        }
        DeepSeekV4DsparkTargetCapture &capture = context.targetCapture;
        ScopedDeepSeekV4DsparkVerification verificationScope;
        ScopedDeepSeekV4HistorySnapshotSuppress historyScope(true);
#ifdef USE_CUDA
        DeepSeekV4DsparkTargetGpuInput targetGpuInput;
        targetGpuInput.proposalIds = proposal.gpuTokens;
        targetGpuInput.readySignal = proposal.gpuReadySignal;
        targetGpuInput.readySeen = proposal.gpuReadySeen;
        targetGpuInput.anchorToken = anchorToken;
        targetGpuInput.startPos = oldTokens;
        targetGpuInput.proposalCount = verifyDrafts;
        DeepSeekV4DsparkTargetGpuInputScope gpuInputScope(
            proposal.gpuDeferred ? &targetGpuInput : nullptr);
#endif
        const auto targetSubmitBegin = DsparkProfileClock::now();
        RunDsparkTarget(
            verifyIds, oldTokens, pastKeyValues, generationConfig,
            lastTokens, nullptr, &capture);
        dsparkTargetSubmitMs = dsparkElapsedMs(
            targetSubmitBegin, DsparkProfileClock::now());
        const auto targetFinishBegin = DsparkProfileClock::now();
        std::vector<int> targetRows;
        DeepSeekV4DsparkProposal pipelinedProposal;
        bool gpuPostprocess = false;
        int gpuAccepted = -1;
        int gpuCommitTokens = -1;
        int gpuNextToken = -1;
#ifdef USE_CUDA
        const int verifyRows = verifyTokens;
        bool gpuPostprocessEligible =
            verifyDrafts == dsparkTokens && proposal.gpuDeferred &&
            capture.samplingReady &&
            capture.contextReady && capture.contextRows == verifyRows &&
            FastllmCudaDeepSeekV4DsparkMarkovPeerAvailable() &&
            proposal.gpuTokens != nullptr &&
            proposal.gpuTokens->dataType == DataType::INT32 &&
            proposal.gpuTokens->cudaData != nullptr &&
            proposal.gpuTokens->Count(0) >= (uint64_t)dsparkTokens &&
            capture.samplingLogitsFloat != nullptr &&
            capture.samplingGreedyIds != nullptr &&
            capture.samplingGreedyScores != nullptr &&
            (int)capture.contextStageKV.size() == dsparkLayers &&
            (int)context.mainWindowKV.size() == dsparkLayers;
        std::vector<int> acceptanceDevices;
        std::vector<int> acceptanceOffsets;
        int acceptanceRootDevice = -1;
        if (gpuPostprocessEligible) {
            for (const auto &ready : capture.samplingReadyEvents) {
                acceptanceDevices.push_back(ready.first);
            }
            acceptanceRootDevice =
                GetPointerDeviceId(proposal.gpuTokens->cudaData);
            gpuPostprocessEligible = !acceptanceDevices.empty() &&
                acceptanceRootDevice >= 0 &&
                std::find(acceptanceDevices.begin(),
                          acceptanceDevices.end(),
                          acceptanceRootDevice) != acceptanceDevices.end() &&
                DeepSeekV4GraphTensorMatches(
                    *capture.samplingGreedyIds, DataType::INT32,
                    {verifyRows}, acceptanceDevices) &&
                DeepSeekV4GraphTensorMatches(
                    *capture.samplingGreedyScores, DataType::FLOAT32,
                    {verifyRows}, acceptanceDevices) &&
                capture.samplingLogitsFloat->multiDeviceData &&
                capture.samplingLogitsFloat->IsTensorParallelSharded();
        }
        for (int device : acceptanceDevices) {
            if (!gpuPostprocessEligible) {
                break;
            }
            auto logitsIt =
                capture.samplingLogitsFloat->multiDeviceDatas.find(device);
            auto rangesIt =
                capture.samplingLogitsFloat->tpRanges.find(device);
            Data *localLogits = logitsIt ==
                    capture.samplingLogitsFloat->multiDeviceDatas.end() ?
                nullptr : logitsIt->second;
            gpuPostprocessEligible =
                localLogits != nullptr && localLogits->cudaData != nullptr &&
                localLogits->dataType == DataType::FLOAT32 &&
                !localLogits->dims.empty() &&
                localLogits->Count(0) ==
                    (uint64_t)verifyRows * localLogits->dims.back() &&
                rangesIt != capture.samplingLogitsFloat->tpRanges.end() &&
                rangesIt->second.size() == 1 &&
                rangesIt->second[0].second - rangesIt->second[0].first ==
                    localLogits->dims.back();
            if (gpuPostprocessEligible) {
                acceptanceOffsets.push_back(rangesIt->second[0].first);
            }
        }
        for (int stage = 0;
             stage < dsparkLayers && gpuPostprocessEligible; ++stage) {
            gpuPostprocessEligible =
                capture.contextStageKV[stage] != nullptr &&
                DeepSeekV4GraphTensorMatches(
                    *capture.contextStageKV[stage], DataType::BFLOAT16,
                    {1, verifyRows, head_dim_full}, acceptanceDevices) &&
                DeepSeekV4GraphTensorMatches(
                    context.mainWindowKV[stage], DataType::BFLOAT16,
                    {1, window_size, head_dim_full}, acceptanceDevices);
        }

        const int acceptanceResultInts =
            3 + verifyRows + dsparkTokens;
        if (gpuPostprocessEligible) {
            bool workspaceMatches = context.targetAcceptanceReady &&
                context.targetAcceptanceDevices == acceptanceDevices &&
                context.targetAcceptanceOffsets == acceptanceOffsets &&
                context.targetAcceptanceRootDevice ==
                    acceptanceRootDevice &&
                context.targetAcceptanceHost != nullptr &&
                DeepSeekV4GraphTensorMatches(
                    context.targetAcceptanceCandidateIds,
                    DataType::INT32,
                    {(int)acceptanceDevices.size() * verifyRows},
                    {acceptanceRootDevice}) &&
                DeepSeekV4GraphTensorMatches(
                    context.targetAcceptanceCandidateScores,
                    DataType::FLOAT32,
                    {(int)acceptanceDevices.size() * verifyRows},
                    {acceptanceRootDevice}) &&
                DeepSeekV4GraphTensorMatches(
                    context.targetAcceptanceGlobalOffsets,
                    DataType::INT32,
                    {(int)acceptanceDevices.size()},
                    {acceptanceRootDevice}) &&
                DeepSeekV4GraphTensorMatches(
                    context.targetAcceptanceResult, DataType::INT32,
                    {acceptanceResultInts}, {acceptanceRootDevice}) &&
                DeepSeekV4GraphTensorMatches(
                    context.targetAcceptanceSignal, DataType::INT32,
                    {1}, {acceptanceRootDevice}) &&
                DeepSeekV4GraphTensorMatches(
                    context.targetAcceptanceSeen, DataType::INT32,
                    {4}, acceptanceDevices);
            if (!workspaceMatches) {
                context.targetAcceptanceReady = false;
                context.targetAcceptanceHost.reset();
                const bool allocated =
                    DeepSeekV4AllocateGraphTensor(
                        context.targetAcceptanceCandidateIds,
                        DataType::INT32,
                        {(int)acceptanceDevices.size() * verifyRows},
                        {acceptanceRootDevice}, false) &&
                    DeepSeekV4AllocateGraphTensor(
                        context.targetAcceptanceCandidateScores,
                        DataType::FLOAT32,
                        {(int)acceptanceDevices.size() * verifyRows},
                        {acceptanceRootDevice}, false) &&
                    DeepSeekV4AllocateGraphTensor(
                        context.targetAcceptanceGlobalOffsets,
                        DataType::INT32,
                        {(int)acceptanceDevices.size()},
                        {acceptanceRootDevice}, false) &&
                    DeepSeekV4AllocateGraphTensor(
                        context.targetAcceptanceResult,
                        DataType::INT32, {acceptanceResultInts},
                        {acceptanceRootDevice}, false) &&
                    DeepSeekV4AllocateGraphTensor(
                        context.targetAcceptanceSignal,
                        DataType::INT32, {1},
                        {acceptanceRootDevice}, true) &&
                    DeepSeekV4AllocateGraphTensor(
                        context.targetAcceptanceSeen,
                        DataType::INT32, {4},
                        acceptanceDevices, true);
                void *hostResult = allocated ? FastllmCudaHostMalloc(
                    (size_t)acceptanceResultInts * sizeof(int)) : nullptr;
                if (hostResult != nullptr) {
                    context.targetAcceptanceHost = std::shared_ptr<void>(
                        hostResult, [](void *pointer) {
                            FastllmCudaHostFree(pointer);
                        });
                    FastllmCudaSetDevice(acceptanceRootDevice);
                    FastllmCudaCopyFromHostToDevice(
                        context.targetAcceptanceGlobalOffsets.cudaData,
                        acceptanceOffsets.data(),
                        acceptanceOffsets.size() * sizeof(int));
                    context.targetAcceptanceDevices = acceptanceDevices;
                    context.targetAcceptanceOffsets = acceptanceOffsets;
                    context.targetAcceptanceRootDevice =
                        acceptanceRootDevice;
                    context.targetAcceptanceReady = true;
                    workspaceMatches = true;
                }
            }
            gpuPostprocessEligible = workspaceMatches;
        }

        if (gpuPostprocessEligible) {
            FastllmCudaSetDevice(acceptanceRootDevice);
            bool enqueued = true;
            for (int device : acceptanceDevices) {
                auto ready = capture.samplingReadyEvents.find(device);
                if (ready == capture.samplingReadyEvents.end() ||
                    ready->second == nullptr) {
                    enqueued = false;
                    break;
                }
                FastllmCudaCurrentThreadStreamWaitEvent(ready->second);
            }
            const size_t rowBytes =
                (size_t)verifyRows * sizeof(int);
            for (int rank = 0;
                 rank < (int)acceptanceDevices.size() && enqueued; ++rank) {
                const int device = acceptanceDevices[rank];
                Data *localIds = GetTensorCudaReplica(
                    *capture.samplingGreedyIds, device);
                Data *localScores = GetTensorCudaReplica(
                    *capture.samplingGreedyScores, device);
                enqueued = localIds != nullptr && localScores != nullptr &&
                    FastllmCudaMemcpyPeerAsyncCurrentThread(
                        acceptanceRootDevice,
                        (int*)context.targetAcceptanceCandidateIds.cudaData +
                            (size_t)rank * verifyRows,
                        device, localIds->cudaData, rowBytes) &&
                    FastllmCudaMemcpyPeerAsyncCurrentThread(
                        acceptanceRootDevice,
                        (float*)context.targetAcceptanceCandidateScores.cudaData +
                            (size_t)rank * verifyRows,
                        device, localScores->cudaData, rowBytes);
            }
            if (enqueued) {
                enqueued = FastllmCudaDeepSeekV4DsparkAcceptPeer(
                    (const int*)
                        context.targetAcceptanceCandidateIds.cudaData,
                    (const float*)
                        context.targetAcceptanceCandidateScores.cudaData,
                    (const int*)
                        context.targetAcceptanceGlobalOffsets.cudaData,
                    (const int*)proposal.gpuTokens->cudaData,
                    (int)acceptanceDevices.size(), verifyRows,
                    (int*)context.targetAcceptanceResult.cudaData,
                    (uint32_t*)context.targetAcceptanceSignal.cudaData);
            }
            if (enqueued) {
                enqueued =
                    FastllmCudaCopyFromDeviceToHostAsyncCurrentThread(
                        context.targetAcceptanceHost.get(),
                        context.targetAcceptanceResult.cudaData,
                        (size_t)acceptanceResultInts * sizeof(int));
            }
            if (enqueued) {
                DeepSeekV4DsparkDraftGpuInput draftGpuInput;
                draftGpuInput.acceptanceResult =
                    &context.targetAcceptanceResult;
                draftGpuInput.readySignal =
                    &context.targetAcceptanceSignal;
                draftGpuInput.readySeen =
                    &context.targetAcceptanceSeen;
                draftGpuInput.stageKV = capture.contextStageKV;
                draftGpuInput.baseCommittedTokens = oldTokens;
                draftGpuInput.rows = verifyRows;
                DeepSeekV4DsparkDraftGpuInputScope gpuInputScope(
                    &draftGpuInput);
                pipelinedProposal = RunDsparkDraft(
                    dsparkNoiseTokenId, context, false, true);
                AssertInFastLLM(
                    pipelinedProposal.gpuDeferred,
                    "DSpark GPU postprocess could not launch the next "
                    "deferred draft.");
                FastllmCudaSetDevice(acceptanceRootDevice);
                FastllmCudaSyncCurrentThreadStream();

                const int *result = (const int*)
                    context.targetAcceptanceHost.get();
                gpuAccepted = result[0];
                gpuCommitTokens = result[1];
                gpuNextToken = result[2];
                AssertInFastLLM(
                    gpuAccepted >= 0 && gpuAccepted <= dsparkTokens &&
                    gpuCommitTokens == gpuAccepted + 1 &&
                    gpuCommitTokens <= verifyRows,
                    "DSpark GPU acceptance returned an invalid prefix.");
                targetRows.assign(result + 3,
                                  result + 3 + verifyRows);
                proposal.tokens.assign(
                    result + 3 + verifyRows,
                    result + 3 + verifyRows + dsparkTokens);
                proposal.gpuDeferred = false;
                proposal.gpuTokens = nullptr;
                proposal.gpuReadySignal = nullptr;
                proposal.gpuReadySeen = nullptr;
                for (int token = 0; token < dsparkTokens; ++token) {
                    verifyIds[token + 1] = proposal.tokens[token];
                }
                capture.samplingDevicesDrained = true;
                gpuPostprocess = true;
            } else {
                // Drain any event waits or partial copies before handing the
                // same verifier workspaces to the established CPU fallback.
                FastllmCudaSetDevice(acceptanceRootDevice);
                FastllmCudaSyncCurrentThreadStream();
            }
        }

        if (!gpuPostprocess) {
            targetRows = SampleDsparkTargetRows(capture.headInput, &context);
        }
        if (!gpuPostprocess && proposal.gpuDeferred) {
            AssertInFastLLM(
                proposal.gpuTokens != nullptr &&
                proposal.gpuTokens->dataDevice == DataDevice::CUDA &&
                proposal.gpuTokens->dataType == DataType::INT32 &&
                proposal.gpuTokens->cudaData != nullptr,
                "DSpark deferred proposal has an invalid CUDA tensor.");
            proposal.tokens.resize(dsparkTokens);
            FastllmCudaSetDevice(
                GetPointerDeviceId(proposal.gpuTokens->cudaData));
            FastllmCudaCopyFromDeviceToHost(
                proposal.tokens.data(), proposal.gpuTokens->cudaData,
                proposal.tokens.size() * sizeof(int));
            proposal.gpuDeferred = false;
            for (int token = 0; token < dsparkTokens; ++token) {
                verifyIds[token + 1] = proposal.tokens[token];
            }
        }
#else
        targetRows = SampleDsparkTargetRows(capture.headInput, &context);
#endif
        dsparkTargetFinishMs = dsparkElapsedMs(
            targetFinishBegin, DsparkProfileClock::now());
        auto updateBestLatency = [](
                std::atomic<long long> &slot, double milliseconds) {
            const long long candidate = (long long)std::llround(
                milliseconds * 1000.0);
            if (candidate <= 0) {
                return;
            }
            long long current = slot.load(std::memory_order_relaxed);
            while ((current == 0 || candidate < current) &&
                   !slot.compare_exchange_weak(
                       current, candidate,
                       std::memory_order_relaxed,
                       std::memory_order_relaxed)) {
            }
        };
        if (!usedPrefetchedProposal && dsparkDraftSubmitMs > 0.0) {
            updateBestLatency(dsparkDraftBestUs, dsparkDraftSubmitMs);
        }
        updateBestLatency(
            dsparkVerifyBestUs[verifyDrafts],
            dsparkTargetSubmitMs + dsparkTargetFinishMs);
        AssertInFastLLM(
            (int)proposal.tokens.size() == dsparkTokens,
            "DSpark proposal materialization returned an invalid length.");
        AssertInFastLLM(
            (int)targetRows.size() == verifyTokens,
            "DSpark target verification returned an invalid row count.");

        int accepted = 0;
        while (accepted < verifyDrafts &&
               proposal.tokens[accepted] == targetRows[accepted]) {
            accepted++;
        }
        const int nextToken = targetRows[accepted];
        const int commitTokens = accepted + 1;
        if (gpuPostprocess) {
            AssertInFastLLM(
                accepted == gpuAccepted &&
                commitTokens == gpuCommitTokens &&
                nextToken == gpuNextToken,
                "DSpark GPU and CPU acceptance decisions diverged.");
        }

        if (commitTokens < verifyTokens) {
#ifdef USE_CUDA
            // MultiCUDA worker streams finish the speculative writes before
            // their logical shapes are published back to every replica.
            if (!capture.samplingDevicesDrained) {
                SynchronizeDeepSeekV4TensorParallelDevices(
                    this->deviceMap);
            }
#endif
            const int committedEnd = oldTokens + commitTokens;
            for (DeepSeekV4DecodeLayerCache &cache :
                 requestState->decodeLayerCaches) {
                TruncateDeepSeekV4DecodeCache(cache, committedEnd);
            }
            // Fixed graph rings are position-addressed. Rejected future rows
            // are ignored by dynamic metadata and overwritten by the next
            // verification, so keep the expensive graph executable alive.
            if ((int)requestState->historyTokens.size() > committedEnd) {
                requestState->historyTokens.resize(committedEnd);
            }
            for (auto &past : pastKeyValues) {
                if (past.first.dims.size() == 3 &&
                    past.first.dims[1] >= committedEnd) {
                    ResizeTensorSequenceInPlace(
                        past.first, committedEnd);
                }
                if (past.second.dims.size() == 3 &&
                    past.second.dims[1] >= committedEnd) {
                    ResizeTensorSequenceInPlace(
                        past.second, committedEnd);
                }
            }
        }
        if (gpuPostprocess) {
            // The next-draft preamble has already shifted all three replicated
            // windows and appended this accepted verifier prefix on GPU.
            context.committedTokens += commitTokens;
        } else {
            AppendDsparkTargetHidden(capture, commitTokens, context);
        }
        context.historyTokens.insert(
            context.historyTokens.end(), verifyIds.begin(),
            verifyIds.begin() + commitTokens);
        context.proposedTokens += verifyDrafts;
        context.acceptedTokens += accepted;
        context.verifyRounds++;
        const int trackedDrafts = std::min(
            verifyDrafts,
            (int)dsparkDraftPositionAttempts.size());
        double predictedSurvival = 1.0;
        for (int position = 0; position < trackedDrafts; ++position) {
            predictedSurvival *= std::max(
                1.0e-4, std::min(
                    1.0 - 1.0e-4,
                    (double)proposal.confidence[position]));
            dsparkDraftPositionAttempts[position].fetch_add(
                1, std::memory_order_relaxed);
            dsparkDraftPositionPredictedSurvival[position].fetch_add(
                (long long)std::llround(predictedSurvival * 1000000.0),
                std::memory_order_relaxed);
            if (accepted > position) {
                dsparkDraftPositionAccepts[position].fetch_add(
                    1, std::memory_order_relaxed);
            }
        }
        dsparkProposedTokenCount.fetch_add(
            verifyDrafts, std::memory_order_relaxed);
        dsparkAcceptedTokenCount.fetch_add(
            accepted, std::memory_order_relaxed);
        const long long validations =
            dsparkValidationCount.fetch_add(
                1, std::memory_order_relaxed) + 1;
        if (!targetOnlyRound && verifyDrafts > 0 &&
            !preferTargetOnly &&
            validations > DEEPSEEK_V4_DSPARK_CALIBRATION_ROUNDS) {
            context.speculativeWindowMs += dsparkElapsedMs(
                dsparkRoundBegin, DsparkProfileClock::now());
            context.speculativeWindowCommitted += commitTokens;
            context.speculativeWindowRounds++;
            if (context.speculativeWindowRounds >= 2) {
                const long long targetOnlyUs =
                    dsparkVerifyBestUs[0].load(
                        std::memory_order_relaxed);
                if (targetOnlyUs > 0 &&
                    context.speculativeWindowMs > 0.0) {
                    const double speculativeTokensPerSecond =
                        (double)context.speculativeWindowCommitted * 1000.0 /
                        context.speculativeWindowMs;
                    const double targetOnlyTokensPerSecond =
                        1000000.0 / (double)targetOnlyUs;
                    // Require a small positive margin so measurement noise
                    // cannot keep a non-improving speculative path enabled.
                    if (speculativeTokensPerSecond <=
                            targetOnlyTokensPerSecond * 1.01) {
                        context.targetOnlyRoundsRemaining =
                            dsparkTargetOnlyCooldownRounds;
                    }
                }
                context.speculativeWindowMs = 0.0;
                context.speculativeWindowCommitted = 0;
                context.speculativeWindowRounds = 0;
            }
        } else if (targetOnlyRound || preferTargetOnly ||
                   verifyDrafts == 0) {
            context.speculativeWindowMs = 0.0;
            context.speculativeWindowCommitted = 0;
            context.speculativeWindowRounds = 0;
        }
        if (validations % DEEPSEEK_V4_DSPARK_LOG_INTERVAL == 0) {
            const long long proposed = dsparkProposedTokenCount.load(
                std::memory_order_relaxed);
            const long long acceptedTotal = dsparkAcceptedTokenCount.load(
                std::memory_order_relaxed);
            const double acceptRate = proposed > 0 ?
                (double)acceptedTotal * 100.0 / (double)proposed : 0.0;
            const double averageAccepted = validations > 0 ?
                (double)acceptedTotal / (double)validations : 0.0;
            std::printf(
                "[DeepSeek-V4 DSpark] validations=%lld, "
                "accept_rate=%.2f%%, avg_accepted=%.2f/%d, "
                "pos_accept_rate=[",
                validations, acceptRate, averageAccepted, dsparkTokens);
            const int loggedDrafts = std::min(
                dsparkTokens,
                (int)dsparkDraftPositionAttempts.size());
            for (int position = 0; position < loggedDrafts; ++position) {
                const long long attempts =
                    dsparkDraftPositionAttempts[position].load(
                        std::memory_order_relaxed);
                const long long positionAccepts =
                    dsparkDraftPositionAccepts[position].load(
                        std::memory_order_relaxed);
                const double positionRate = attempts > 0 ?
                    (double)positionAccepts * 100.0 /
                        (double)attempts : 0.0;
                std::printf("%s%.2f%%",
                            position == 0 ? "" : ", ", positionRate);
            }
            std::printf("].\n");
            std::fflush(stdout);
        }

        std::vector<int> outputs;
        outputs.insert(outputs.end(), proposal.tokens.begin(),
                       proposal.tokens.begin() + accepted);
        outputs.push_back(nextToken);
#ifdef USE_CUDA
        if (gpuPostprocess) {
            context.prefetchedProposal = std::move(pipelinedProposal);
            context.prefetchedAnchorToken = nextToken;
            context.prefetchedCommittedTokens = context.committedTokens;
            context.prefetchedProposalReady = true;
        }
#endif
        for (int index = 1; index < (int)outputs.size(); index++) {
            context.pending.push_back(
                {outputs[index - 1], outputs[index]});
        }
        AssertInFastLLM(!outputs.empty(),
                        "DSpark produced no output token.");
        return outputs[0];
    }

    int DeepSeekV4Model::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                                 const fastllm::Data &positionIds,
                                 std::vector<std::pair<Data, Data>> &pastKeyValues,
                                 const GenerationConfig &generationConfig,
                                 const LastTokensManager &lastTokens,
                                 std::vector <float> *retLogits) {
        if (dsparkEnabled && generationConfig.IsSimpleGreedy() &&
            !generationConfig.output_logits) {
            return ForwardDspark(inputIds, pastKeyValues,
                                 generationConfig, lastTokens, retLogits);
        }
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
        int startPos = 0;
        if (positionIds.dims.size() >= 2 && positionIds.Count(0) > 0) {
            auto pids = ReadTokenIds(positionIds);
            startPos = pids.empty() ? 0 : pids[0];
        }
        int originalStartPos = startPos;
#ifdef USE_CUDA
        // MultiCUDA helpers operate on one physical rank at a time and some
        // requests recursively split prefill at a prefix-cache boundary.  Every
        // ForwardBatch invocation starts from the same physical root so a new
        // workspace can never be allocated on the rank selected by the preceding
        // cache operation.
        SelectDeepSeekV4TensorParallelRoot(this->deviceMap);
#endif
        std::shared_ptr<DeepSeekV4RequestState> requestState =
            deepSeekV4ForwardRequestStateOverride ?
                deepSeekV4ForwardRequestStateOverride : GetRequestState(pastKeyValues);
        std::vector<DeepSeekV4DecodeLayerCache> *decodeCachesPtr =
            requestState == nullptr ? &this->decodeLayerCaches : &requestState->decodeLayerCaches;
        std::vector<int> *historyTokensPtr =
            requestState == nullptr ? &this->deepseekV4HistoryTokens : &requestState->historyTokens;
        auto &activeDecodeLayerCaches = *decodeCachesPtr;
        auto &activeHistoryTokens = *historyTokensPtr;
        if (!this->dsparkEnabled && this->saveHistoryChat &&
            !DeepSeekV4PrefixCacheDisabled() &&
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
        int bsz = inputIds.dims[0];
        int seqlen = inputIds.dims[1];
        int dim = embed_dim;
        if (useDecodeCache && originalStartPos == 0) {
#ifdef USE_CUDA
            if (!activeDecodeLayerCaches.empty() && DeepSeekV4PreferCuda()) {
                // MultiCUDA uses per-thread default streams.  Sampling can finish on the root device
                // while cache-update kernels on peer devices are still in flight.  A new request owns
                // fresh decode caches, so drain every TP stream before releasing the previous request's
                // cache tensors.
                std::vector<int> devices;
                std::map<int, int> ratios;
                FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
                if (devices.empty()) {
                    ForceDeviceSync();
                } else {
                    int originalDevice = FastllmCudaGetDevice();
                    for (int device : devices) {
                        FastllmCudaSyncDevice(device);
                    }
                    FastllmCudaSetDevice(originalDevice);
                }
            }
            if (requestState != nullptr) {
                requestState->cudaGraphState.reset();
            } else {
                this->fallbackCudaGraphState.reset();
            }
#endif
            activeDecodeLayerCaches.clear();
            activeDecodeLayerCaches.resize(block_cnt);
            if (requestState != nullptr) {
                requestState->restoredHistoryCache = false;
            }
        }

        std::vector<int> tokenIds;
        tokenIds = ReadTokenIds(inputIds);
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

        DeepSeekV4DecodeWorkspace localDecodeWorkspace;
        DeepSeekV4DecodeWorkspace *decodeWorkspace = &localDecodeWorkspace;
        const Data *modelInputIds = &inputIds;
        bool graphSafeDecode = false;
        bool dsparkGraphVerification = false;
#ifdef USE_CUDA
        std::shared_ptr<DeepSeekV4CudaGraphState> graphState;
        std::unique_lock<std::mutex> graphLock;
        bool graphGpuReplayInputs = false;
        const DeepSeekV4DsparkTargetGpuInput *targetGpuInput =
            deepSeekV4DsparkTargetGpuInput;
        std::vector<int> graphDevices;
        dsparkGraphVerification =
            DeepSeekV4DsparkVerificationActive() && dsparkEnabled &&
            dsparkTokens > 0 && seqlen == dsparkTokens + 1;
        // Size and specialize a request-local decode graph for the complete
        // generation, rather than only for the position at which it happens
        // to be captured.  DSpark makes this especially important: rebuilding
        // the eight-row target graph at the C4 indexer threshold (~2048
        // tokens), and then growing its fixed KV storage at 4096, introduces
        // second-scale holes even though every individual draft/verify replay
        // is fast.  output_token_limit is an upper bound (not a remaining-token
        // count), so adding it at a later step may over-reserve; only the first
        // graph preparation consumes this value and the result is capped by the
        // model context limit.
        long long plannedGraphEnd = (long long)originalStartPos + seqlen;
        if (generationConfig.output_token_limit > 0) {
            plannedGraphEnd = std::max(
                plannedGraphEnd,
                (long long)originalStartPos +
                    generationConfig.output_token_limit);
        }
        if (max_positions > 0) {
            plannedGraphEnd = std::min(
                plannedGraphEnd, (long long)max_positions);
        }
        const int graphPlannedMaxTokens = (int)std::max(
            (long long)originalStartPos + seqlen, plannedGraphEnd);
        bool graphRequested = DeepSeekV4DecodeCudaGraphEnabled();
        const bool graphSequenceSupported =
            seqlen == 1 || dsparkGraphVerification;
        if (graphRequested && batch == 1 && graphSequenceSupported &&
            seqlen + 1 <= kDeepSeekV4CudaGraphMetaInts &&
            originalStartPos > 0 && useDecodeCache &&
            DeepSeekV4PreferCuda() &&
            (int)activeDecodeLayerCaches.size() == block_cnt) {
            std::shared_ptr<void> &graphSlot = requestState != nullptr ?
                requestState->cudaGraphState : this->fallbackCudaGraphState;
            graphState = GetDeepSeekV4CudaGraphState(graphSlot);
            graphLock = std::unique_lock<std::mutex>(graphState->mutex);
            // vLLM bypasses the learned q/scorer/top-k path while every C4
            // compressed row has at most 512 candidates.  The scorer changes
            // the graph structure.  Select it from the request's planned end
            // position on the first capture, so a long generation does not
            // destroy and recapture the full target graph at the 2048-token
            // boundary.  Short requests retain the cheaper direct path.
            const bool desiredIndexerScorerMode =
                graphPlannedMaxTokens / 4 > 512;
            const bool actualIndexerScorerMode =
                (originalStartPos + seqlen) / 4 > 512;
            if (!graphState->captured && !graphState->warmed &&
                graphState->graphMaxTokens == 0) {
                // Latch the request-wide plan before its first warmup.  A long
                // request starts with the scorer enabled so it never needs a
                // mid-generation recapture.
                graphState->indexerScorerMode = desiredIndexerScorerMode;
            } else if ((graphState->captured || graphState->warmed) &&
                       !graphState->indexerScorerMode &&
                       actualIndexerScorerMode) {
                // With no usable output bound, preserve correctness when the
                // real context eventually crosses the C4 scorer threshold.
                graphState->DestroyCapturedGraph();
                graphState->indexerScorerMode = true;
            }
            // Once a request's graph is instantiated, its workspace, fixed KV
            // rings, weights and device placement cannot move before that
            // request is released.  Re-running the full 61-layer preflight on
            // every token calls cudaPointerGetAttributes hundreds of times and
            // costs several milliseconds on the decode critical path.  The
            // captured fast path only stages its fixed-shape mutable inputs. Shape or
            // capacity changes fall through to the complete validation below.
            bool capturedReplayPrepared = graphState->captured &&
                graphState->graphMaxTokens >= originalStartPos + seqlen &&
                graphState->inputDevice >= 0 &&
                graphState->workspace != nullptr &&
                graphState->pinnedMeta != nullptr &&
                graphState->pinnedInputIds != nullptr &&
                graphState->inputIds.cudaData != nullptr &&
                graphState->graphInputIds.cudaData != nullptr &&
                graphState->inputIds.dataType == inputIds.dataType &&
                graphState->inputIds.dims == inputIds.dims &&
                graphState->graphInputIds.dataType == inputIds.dataType &&
                graphState->graphInputIds.dims == inputIds.dims &&
                !graphState->devices.empty() &&
                !graphState->launchOrder.empty();
            if (capturedReplayPrepared) {
                for (const auto &deviceState : graphState->devices) {
                    if (deviceState == nullptr || deviceState->device < 0 ||
                        deviceState->exec == nullptr ||
                        deviceState->decodeMeta == nullptr ||
                        deviceState->decodeMeta->cudaData == nullptr) {
                        capturedReplayPrepared = false;
                        graphDevices.clear();
                        break;
                    }
                    graphDevices.push_back(deviceState->device);
                }
            }
            if (capturedReplayPrepared) {
                graphState->replayInputsPending = false;
                int inputDevice = graphState->inputDevice;
                FastllmCudaSetDevice(inputDevice);
                const bool gpuInputShapeReady =
                    targetGpuInput != nullptr && dsparkGraphVerification &&
                    targetGpuInput->proposalCount == seqlen - 1 &&
                    targetGpuInput->startPos == originalStartPos &&
                    targetGpuInput->proposalIds != nullptr &&
                    targetGpuInput->proposalIds->dataDevice ==
                        DataDevice::CUDA &&
                    targetGpuInput->proposalIds->dataType ==
                        DataType::INT32 &&
                    targetGpuInput->proposalIds->cudaData != nullptr &&
                    targetGpuInput->proposalIds->Count(0) >=
                        (uint64_t)targetGpuInput->proposalCount &&
                    targetGpuInput->readySignal != nullptr &&
                    targetGpuInput->readySignal->dataDevice ==
                        DataDevice::CUDA &&
                    targetGpuInput->readySignal->dataType ==
                        DataType::INT32 &&
                    targetGpuInput->readySignal->cudaData != nullptr &&
                    targetGpuInput->readySeen != nullptr &&
                    targetGpuInput->readySeen->multiDeviceData &&
                    targetGpuInput->readySeen->IsTensorParallelReplicated() &&
                    graphState->inputIds.dataType == DataType::FLOAT32;
                if (gpuInputShapeReady) {
                    graphGpuReplayInputs = true;
                    graphState->replayInputsPending = true;
                } else if (targetGpuInput != nullptr) {
                    capturedReplayPrepared = false;
                } else if (inputIds.dataDevice == DataDevice::CPU &&
                    inputIds.cpuData != nullptr &&
                    inputIds.GetBytes() <=
                        kDeepSeekV4CudaGraphMetaInts * sizeof(float)) {
                    std::memcpy(graphState->pinnedInputIds,
                                inputIds.cpuData, inputIds.GetBytes());
                    graphState->replayInputsPending = true;
                } else if (inputIds.dataDevice == DataDevice::CUDA &&
                           inputIds.cudaData != nullptr) {
                    int sourceDevice = GetPointerDeviceId(inputIds.cudaData);
                    if (sourceDevice == inputDevice) {
                        FastllmCudaCopyFromDeviceToDevice(
                            graphState->inputIds.cudaData, inputIds.cudaData,
                            inputIds.GetBytes());
                    } else if (sourceDevice >= 0) {
                        FastllmCudaMemcpyBetweenDevices(
                            inputDevice, graphState->inputIds.cudaData,
                            sourceDevice, inputIds.cudaData,
                            inputIds.GetBytes());
                    } else {
                        capturedReplayPrepared = false;
                    }
                } else {
                    capturedReplayPrepared = false;
                }
            }
            if (capturedReplayPrepared) {
                int32_t decodeMetaHost[kDeepSeekV4CudaGraphMetaInts] = {};
                decodeMetaHost[0] = (int32_t)originalStartPos;
                for (int token = 0; token < seqlen; token++) {
                    decodeMetaHost[token + 1] = (int32_t)(
                        token < (int)tokenIds.size() ? tokenIds[token] : 0);
                }
                if (graphGpuReplayInputs) {
                    // The launch worker fills both buffers after observing the
                    // draft graph's system-scope proposal-ready epoch.
                } else if (graphState->replayInputsPending) {
                    std::memcpy(graphState->pinnedMeta, decodeMetaHost,
                                sizeof(decodeMetaHost));
                } else {
                    for (auto &deviceState : graphState->devices) {
                        FastllmCudaSetDevice(deviceState->device);
                        FastllmCudaCopyFromHostToDevice(
                            deviceState->decodeMeta->cudaData,
                            decodeMetaHost, sizeof(decodeMetaHost));
                    }
                }
                graphSafeDecode = true;
                decodeWorkspace = graphState->workspace.get();
                modelInputIds = &graphState->inputIds;
            }

            AssertInFastLLM(
                targetGpuInput == nullptr || graphGpuReplayInputs,
                "DSpark GPU proposal requires an instantiated target CUDA graph.");

            if (!graphSafeDecode) {
                graphDevices.clear();
                std::map<int, int> graphRatios;
                FastllmGetMulticudaDeviceAndRatio(
                    graphDevices, graphRatios, true);
            }
            if (!graphSafeDecode && !graphDevices.empty()) {
                graphState->PrepareDevices(graphDevices);
                int graphMaxTokens = graphState->graphMaxTokens > 0 ?
                    graphState->graphMaxTokens :
                    DeepSeekV4GraphInitialMaxTokens(
                        graphPlannedMaxTokens);
                while (graphMaxTokens < originalStartPos + seqlen) {
                    graphMaxTokens *= 2;
                }
                bool graphPrepared = true;
                bool addressChanged = graphState->graphMaxTokens != 0 &&
                                      graphState->graphMaxTokens != graphMaxTokens;
                // Build every layer-boundary P2P communicator before capture.
                // The embedding output is the source of the first boundary.
                Data &embeddingWeight = weight["embed.weight"];
                int embeddingDevice = GetTensorCudaDevice(embeddingWeight);
                int previousLayerDevice = embeddingDevice;
                std::vector<int> pipelineDevices;
                if (embeddingDevice >= 0) {
                    pipelineDevices.push_back(embeddingDevice);
                }
                bool fullTensorParallelGraph = !activeDecodeLayerCaches.empty() &&
                    activeDecodeLayerCaches[0].windowKV.multiDeviceData &&
                    activeDecodeLayerCaches[0].windowKV.IsTensorParallelReplicated();
                if (fullTensorParallelGraph && embeddingDevice >= 0) {
                    for (int device : graphDevices) {
                        if (device != embeddingDevice &&
                            !FastllmInitNcclGraphPeer(embeddingDevice, device)) {
                            graphPrepared = false;
                            break;
                        }
                    }
                }
                for (int layer = 0; layer < block_cnt && graphPrepared; layer++) {
                    DeepSeekV4DecodeLayerCache &cache = activeDecodeLayerCaches[layer];
                    int layerDevice = GetTensorCudaDevice(cache.windowKV);
                    if (layerDevice < 0 || graphState->GetDecodeMeta(layerDevice) == nullptr) {
                        graphPrepared = false;
                        break;
                    }
                    std::vector<int> layerGraphDevices{layerDevice};
                    if (cache.windowKV.multiDeviceData &&
                        cache.windowKV.IsTensorParallelReplicated()) {
                        layerGraphDevices.clear();
                        for (int device : graphDevices) {
                            if (GetTensorCudaReplica(cache.windowKV, device) != nullptr) {
                                layerGraphDevices.push_back(device);
                            }
                        }
                        if (layerGraphDevices.size() != graphDevices.size()) {
                            graphPrepared = false;
                            break;
                        }
                    }
                    bool layerAddressChanged = false;
                    Data emptyApe, emptyNorm;
                    Data &apeWeight = cache.compressRatio > 0 ?
                        weight["layers." + std::to_string(layer) + ".attn.compressor.ape"] :
                        emptyApe;
                    Data &normWeight = cache.compressRatio > 0 ?
                        weight["layers." + std::to_string(layer) +
                               ".attn.compressor.norm.weight"] : emptyNorm;
                    Data *indexerApeWeight = cache.compressRatio == 4 ?
                        &weight["layers." + std::to_string(layer) +
                                ".attn.indexer.compressor.ape"] : nullptr;
                    Data *indexerNormWeight = cache.compressRatio == 4 ?
                        &weight["layers." + std::to_string(layer) +
                                ".attn.indexer.compressor.norm.weight"] : nullptr;
                    if (!DeepSeekV4PrepareFixedGraphLayerCache(
                            cache, apeWeight, normWeight, indexerApeWeight,
                            indexerNormWeight, graphMaxTokens, seqlen,
                            layerGraphDevices,
                            layerAddressChanged)) {
                        graphPrepared = false;
                        break;
                    }
                    addressChanged |= layerAddressChanged;
                    if (previousLayerDevice >= 0 && previousLayerDevice != layerDevice) {
                        if (!FastllmInitNcclGraphPeer(previousLayerDevice, layerDevice)) {
                            graphPrepared = false;
                            break;
                        }
                    }
                    if (pipelineDevices.empty() || pipelineDevices.back() != layerDevice) {
                        pipelineDevices.push_back(layerDevice);
                    }
                    previousLayerDevice = layerDevice;
                }
                if (graphPrepared) {
                    const size_t sparseMlaScratchBytes =
                        DeepSeekV4SparseMlaSm120Enabled() ?
                        FastllmCudaDeepSeekV4SparseAttentionDecodeCachedGraphSm120ScratchBytes(
                            seqlen, num_attention_heads, 4) : 0;
                    if (sparseMlaScratchBytes >
                            (size_t)std::numeric_limits<int>::max()) {
                        graphPrepared = false;
                    } else if (sparseMlaScratchBytes > 0) {
                        const std::vector<int> scratchDims = {
                            (int)sparseMlaScratchBytes};
                        const bool scratchStale =
                            !DeepSeekV4GraphTensorMatches(
                                graphState->workspace->sparseMlaScratch,
                                DataType::INT8, scratchDims, graphDevices);
                        if (scratchStale &&
                            !DeepSeekV4AllocateGraphTensor(
                                graphState->workspace->sparseMlaScratch,
                                DataType::INT8, scratchDims, graphDevices,
                                false)) {
                            graphPrepared = false;
                        }
                        addressChanged |= scratchStale;
                    }
                }
                if (graphPrepared) {
                    // A full-TP capture owns one graph per GPU.  Every graph
                    // must be launched; launching only the metadata/root GPU
                    // would strand custom all-reduce barriers on its peers.
                    graphState->SetLaunchOrder(fullTensorParallelGraph ?
                                               graphDevices : pipelineDevices);
                }

                // Embedding is dispatched before ApplyDeviceMap selects the first
                // transformer layer.  Its CUDA operator therefore runs where the
                // embedding weight lives (normally GPU 0), which is not necessarily
                // the device that owns layer 0's window KV.  Keep the captured token
                // staging tensor on the embedding device so Executor::Run does not
                // migrate and reallocate it on every decode step.
                int inputDevice = embeddingDevice;
                if (inputDevice < 0 && graphPrepared &&
                    !activeDecodeLayerCaches.empty() &&
                    GetTensorCudaDevice(activeDecodeLayerCaches[0].windowKV) >= 0) {
                    inputDevice = GetTensorCudaDevice(
                        activeDecodeLayerCaches[0].windowKV);
                }
                if (graphPrepared && inputDevice >= 0) {
                    graphState->inputDevice = inputDevice;
                    bool inputStale = graphState->inputIds.cudaData == nullptr ||
                                      graphState->inputIds.dataType != inputIds.dataType ||
                                      graphState->inputIds.dims != inputIds.dims ||
                                      GetPointerDeviceId(graphState->inputIds.cudaData) != inputDevice;
                    bool graphInputStale =
                        graphState->graphInputIds.cudaData == nullptr ||
                        graphState->graphInputIds.dataType != inputIds.dataType ||
                        graphState->graphInputIds.dims != inputIds.dims ||
                        GetPointerDeviceId(graphState->graphInputIds.cudaData) != inputDevice;
                    if (inputStale || graphInputStale) {
                        addressChanged = true;
                    }
                    if (addressChanged) {
                        graphState->DestroyCapturedGraph();
                    }
                    if (inputStale) {
                        ResetData(graphState->inputIds);
                        graphState->inputIds.dataType = inputIds.dataType;
                        graphState->inputIds.Resize(inputIds.dims);
                        graphState->inputIds.dataDevice = DataDevice::CUDA;
                        graphState->inputIds.dataDeviceIds = {inputDevice};
                        FastllmCudaSetDevice(inputDevice);
                        graphState->inputIds.Allocate(false);
                    }
                    if (graphInputStale) {
                        ResetData(graphState->graphInputIds);
                        graphState->graphInputIds.dataType = inputIds.dataType;
                        graphState->graphInputIds.Resize(inputIds.dims);
                        graphState->graphInputIds.dataDevice = DataDevice::CUDA;
                        graphState->graphInputIds.dataDeviceIds = {inputDevice};
                        FastllmCudaSetDevice(inputDevice);
                        graphState->graphInputIds.Allocate(false);
                    }
                    graphState->graphMaxTokens = graphMaxTokens;
                    FastllmCudaSetDevice(inputDevice);
                    if (inputIds.dataDevice == DataDevice::CPU && inputIds.cpuData != nullptr) {
                        FastllmCudaCopyFromHostToDevice(
                            graphState->inputIds.cudaData, inputIds.cpuData,
                            inputIds.GetBytes());
                    } else if (inputIds.dataDevice == DataDevice::CUDA &&
                               inputIds.cudaData != nullptr) {
                        int sourceDevice = GetPointerDeviceId(inputIds.cudaData);
                        if (sourceDevice == inputDevice) {
                            FastllmCudaCopyFromDeviceToDevice(
                                graphState->inputIds.cudaData, inputIds.cudaData,
                                inputIds.GetBytes());
                        } else if (sourceDevice >= 0) {
                            FastllmCudaMemcpyBetweenDevices(
                                inputDevice, graphState->inputIds.cudaData,
                                sourceDevice, inputIds.cudaData, inputIds.GetBytes());
                        } else {
                            graphPrepared = false;
                        }
                    } else {
                        graphPrepared = false;
                    }
                } else {
                    graphPrepared = false;
                }

                if (graphPrepared) {
                    int32_t decodeMetaHost[kDeepSeekV4CudaGraphMetaInts] = {};
                    decodeMetaHost[0] = (int32_t)originalStartPos;
                    for (int token = 0; token < seqlen; token++) {
                        decodeMetaHost[token + 1] = (int32_t)(
                            token < (int)tokenIds.size() ? tokenIds[token] : 0);
                    }
                    for (auto &deviceState : graphState->devices) {
                        FastllmCudaSetDevice(deviceState->device);
                        FastllmCudaCopyFromHostToDevice(
                            deviceState->decodeMeta->cudaData, decodeMetaHost,
                            sizeof(decodeMetaHost));
                    }
                    graphSafeDecode = true;
                    decodeWorkspace = graphState->workspace.get();
                    modelInputIds = &graphState->inputIds;
                }
            }
        }
#endif

        Data &hiddenStates = decodeWorkspace->hiddenStates;
        Data &hiddenStatesBeforeHcExpand = decodeWorkspace->hiddenStatesBeforeHcExpand;
        Data &hiddenStatesTemp = decodeWorkspace->hiddenStatesTemp;
        Data &attnInput = decodeWorkspace->attnInput;
        Data &qr = decodeWorkspace->qr;
        Data &qNorm = decodeWorkspace->qNorm;
        Data &q = decodeWorkspace->q;
        Data &kv = decodeWorkspace->kv;
        HcMix &attnMix = decodeWorkspace->attnMix;
        HcMix &ffnMix = decodeWorkspace->ffnMix;
        Data &ffnInput = decodeWorkspace->ffnInput;
        Data &ffnOut = decodeWorkspace->ffnOut;
        Data &expertIndex = decodeWorkspace->expertIndex;
        Data &expertScore = decodeWorkspace->expertScore;
        Data &w1 = decodeWorkspace->w1;
        Data &w2 = decodeWorkspace->w2;
        Data &w3 = decodeWorkspace->w3;
        Data &tempInput = decodeWorkspace->tempInput;
        Data &tempOutput = decodeWorkspace->tempOutput;
        Data &moeInputTemp = decodeWorkspace->moeInputTemp;
        Data &moeOutputTemp = decodeWorkspace->moeOutputTemp;
        Data &compressorKV = decodeWorkspace->compressorKV;
        Data &compressorScore = decodeWorkspace->compressorScore;
        Data &indexerCompressorKV = decodeWorkspace->indexerCompressorKV;
        Data &indexerCompressorScore = decodeWorkspace->indexerCompressorScore;
        Data &indexerQ = decodeWorkspace->indexerQ;
        Data &indexerWeights = decodeWorkspace->indexerWeights;
        Data &attnOut4 = decodeWorkspace->attnOut4;
        Data &woAOut = decodeWorkspace->woAOut;
        Data &attnOut = decodeWorkspace->attnOut;
        Data &sharedExpertOut = decodeWorkspace->sharedExpertOut;
        Data &sharedW1 = decodeWorkspace->sharedW1;
        Data &sharedW3 = decodeWorkspace->sharedW3;
        Data &headInput = decodeWorkspace->headInput;
        Data &samplingHeadRoot = decodeWorkspace->samplingHeadRoot;
        Data &samplingHeadReplicated = decodeWorkspace->samplingHeadReplicated;
        Data &samplingHeadNorm = decodeWorkspace->samplingHeadNorm;
        Data &samplingLogits = decodeWorkspace->samplingLogits;
        Data &samplingLogitsFloat = decodeWorkspace->samplingLogitsFloat;
        Data &samplingGreedyIds = decodeWorkspace->samplingGreedyIds;
        Data &samplingGreedyScores = decodeWorkspace->samplingGreedyScores;
        Data &dsparkTargetCombinedTemp =
            decodeWorkspace->dsparkTargetCombinedTemp;
        Data &dsparkTargetCombined =
            decodeWorkspace->dsparkTargetCombined;
        Data &dsparkTargetProjected =
            decodeWorkspace->dsparkTargetProjected;
        Data &dsparkTargetMainHidden =
            decodeWorkspace->dsparkTargetMainHidden;
        std::vector<Data> &dsparkTargetStageKV =
            decodeWorkspace->dsparkTargetStageKV;

#ifdef USE_CUDA
        const bool multiCudaTensorParallel =
            DeepSeekV4DeviceMapUsesMultiCuda(this->deviceMap);
        // The fused decode helpers use graphDevices as their TP rank set even
        // when full-step CUDA graph capture is disabled.  Leaving it empty made
        // HcPre+Norm silently execute on one root GPU, and the following Linear
        // then consumed a cross-device/root-only tensor instead of TP replicas.
        if (multiCudaTensorParallel && graphDevices.empty()) {
            std::map<int, int> tpRatios;
            FastllmGetMulticudaDeviceAndRatio(
                graphDevices, tpRatios, true);
        }
        bool persistentTpAsync = multiCudaTensorParallel && batch == 1 &&
                                 ((seqlen == 1 && originalStartPos > 0) ||
                                  seqlen > 1);
        // The decode-only HcPre+RMSNorm kernel preserves the BF16 tensor
        // boundary between both operators while avoiding a second global
        // memory round trip.  Keep prefill and non-CUDA execution on the
        // established generic path, and retain an explicit A/B fallback.
        bool cudaHcPreNormDecode =
            graphSafeDecode ||
            (batch == 1 && seqlen == 1 && originalStartPos > 0 &&
             DeepSeekV4PreferCuda() &&
             !EnvFlagEnabled("FASTLLM_DSV4_DISABLE_CUDA_HCPRENORM"));
#else
        bool persistentTpAsync = false;
#endif

        auto runModelBody = [&]() {
#ifdef USE_CUDA
            struct ScopedMultiCudaPersistentAsync {
                bool previous;
                explicit ScopedMultiCudaPersistentAsync(bool enabled) :
                        previous(MultiCudaSetPersistentAsyncDispatch(enabled)) {}
                ~ScopedMultiCudaPersistentAsync() {
                    MultiCudaSetPersistentAsyncDispatch(previous);
                }
            } persistentAsyncScope(persistentTpAsync);
#endif
            Data *curHiddenStates = &hiddenStates;
            Data *nextHiddenStates = &hiddenStatesTemp;
            auto runHcPost = [&](Data &input, const HcMix &mix) {
                DeepSeekV4HcPost(input, *curHiddenStates, mix.postData, mix.combData,
                                 *nextHiddenStates);
                std::swap(curHiddenStates, nextHiddenStates);
            };
#ifdef USE_CUDA
            auto graphCaptureHealthy = [&](const char *stage, int layer) {
                if (!graphState || !graphState->capturing) {
                    return true;
                }
                if (FastllmCudaGetThreadError() ||
                    FastllmCudaGetGraphError()) {
                    std::fprintf(
                        stderr,
                        "[fastllm-dspark-graph] model body rejected at "
                        "layer=%d stage=%s thread_error=%d "
                        "graph_error=%d\n",
                        layer, stage,
                        FastllmCudaGetThreadError() ? 1 : 0,
                        FastllmCudaGetGraphError() ? 1 : 0);
                    std::fflush(stderr);
                    return false;
                }
                int originalDevice = FastllmCudaGetDevice();
                for (int device : graphDevices) {
                    FastllmCudaSetDevice(device);
                    if (FastllmCudaGraphCaptureInvalidated()) {
                        std::fprintf(
                            stderr,
                            "[fastllm-dspark-graph] capture invalidated at "
                            "layer=%d stage=%s device=%d\n",
                            layer, stage, device);
                        std::fflush(stderr);
                        FastllmCudaSetThreadError();
                        FastllmCudaSetDevice(originalDevice);
                        return false;
                    }
                }
                FastllmCudaSetDevice(originalDevice);
                return true;
            };
#endif
            const Data *embeddingInputIds = modelInputIds;
#ifdef USE_CUDA
            if (graphSafeDecode && graphState && modelInputIds != nullptr &&
                modelInputIds->cudaData != nullptr &&
                graphState->graphInputIds.cudaData != nullptr) {
                // Make the staged token copy an explicit root of the captured
                // graph.  Without this tiny D2D node, the first embedding node
                // can start from the previous decode token on graph replay.
                int inputDevice = GetPointerDeviceId(modelInputIds->cudaData);
                FastllmCudaSetDevice(inputDevice);
                FastllmCudaCopyFromDeviceToDevice(
                    graphState->graphInputIds.cudaData,
                    modelInputIds->cudaData, modelInputIds->GetBytes());
                embeddingInputIds = &graphState->graphInputIds;
            }
#endif
            // ParallelEmbedding returns the checkpoint dtype.  EmbeddingDirect
            // preserves the BF16 payload instead of materializing FP32 and
            // converting it back in-place.  Besides avoiding that round trip,
            // the fixed dtype keeps the persistent decode workspace allocation
            // stable across CUDA-graph warmup and capture.
            EmbeddingDirect(*embeddingInputIds, weight["embed.weight"],
                            hiddenStatesBeforeHcExpand);
            hiddenStatesBeforeHcExpand.Reshape({bsz, seqlen, 1, dim});
            bool repeatedToTpReplicas = false;
#ifdef USE_CUDA
            if (graphSafeDecode && persistentTpAsync) {
                repeatedToTpReplicas = MultiCudaRepeatToReplicated(
                    hiddenStatesBeforeHcExpand, 2, hc_mult, hiddenStates);
            }
#endif
            if (!repeatedToTpReplicas) {
                Repeat(hiddenStatesBeforeHcExpand, 2, hc_mult, hiddenStates);
            }
#ifdef USE_CUDA
            if (!graphCaptureHealthy("embedding", -1)) {
                return;
            }
#endif
            bool prefetchedAttnHcPreNorm = false;
            for (int layer = 0; layer < block_cnt; layer++) {
            std::string pre = "layers." + std::to_string(layer);
#ifdef USE_CUDA
            int layerDevice = -1;
            Data *decodeMeta = nullptr;
            if (graphSafeDecode && layer < (int)activeDecodeLayerCaches.size()) {
                DeepSeekV4DecodeLayerCache &cache = activeDecodeLayerCaches[layer];
                layerDevice = GetTensorCudaDevice(cache.windowKV);
                decodeMeta = cache.windowKV.multiDeviceData ?
                    graphState->GetReplicatedDecodeMeta() :
                    graphState->GetDecodeMeta(layerDevice);
            }
#endif
            ApplyDeviceMap(this->deviceMap, layer + 1, block_cnt);
            int compressRatio = compress_ratios.size() > layer ? compress_ratios[layer] : 0;
            bool useCompressRope = compressRatio != 0;
            float layerRopeBase = useCompressRope ? compress_rope_theta : rope_base;
            int layerOriginalSeqLen = useCompressRope ? (int)rope_scaling_original_max_position_embeddings : 0;
            bool fusedAttnHcPreNorm = prefetchedAttnHcPreNorm;
            prefetchedAttnHcPreNorm = false;
#ifdef USE_CUDA
            if (!fusedAttnHcPreNorm && cudaHcPreNormDecode) {
                fusedAttnHcPreNorm = DeepSeekV4HcPreNormMultiCuda(
                    *curHiddenStates, weight[pre + ".hc_attn_fn"],
                    weight[pre + ".hc_attn_scale"], weight[pre + ".hc_attn_base"],
                    weight[pre + ".attn_norm.weight"], hc_mult, hc_sinkhorn_iters,
                    hc_eps, rms_norm_eps, attnInput, attnMix.postData,
                    attnMix.combData, graphDevices);
            }
#endif
            if (!fusedAttnHcPreNorm) {
                DeepSeekV4HcPre(*curHiddenStates, weight[pre + ".hc_attn_fn"],
                                weight[pre + ".hc_attn_scale"], weight[pre + ".hc_attn_base"],
                                hc_mult, hc_sinkhorn_iters, hc_eps, rms_norm_eps,
                                attnMix.y, attnMix.postData, attnMix.combData);
                RMSNormReference(attnMix.y, weight[pre + ".attn_norm.weight"],
                                 rms_norm_eps, attnInput, DataType::BFLOAT16);
            }
            attnMix.b = bsz;
            attnMix.s = seqlen;
            attnMix.hc = hc_mult;
            DeepSeekV4Linear(attnInput, weight[pre + ".attn.wq_a.weight"], Data(), qr, true);
            RMSNormReference(qr, weight[pre + ".attn.q_norm.weight"], rms_norm_eps, qNorm, DataType::BFLOAT16);
            bool hasLearnedIndexer = compressRatio == 4 &&
                HasTensorData(weight[pre + ".attn.indexer.wq_b.weight"]) &&
                HasTensorData(weight[pre + ".attn.indexer.weights_proj.weight"]);
            bool useLearnedIndexerScores = false;
#ifdef USE_CUDA
            useLearnedIndexerScores = hasLearnedIndexer && graphSafeDecode &&
                graphState != nullptr && graphState->indexerScorerMode;
#endif
            if (useLearnedIndexerScores) {
                DeepSeekV4Linear(
                    qNorm, weight[pre + ".attn.indexer.wq_b.weight"],
                    Data(), indexerQ, true);
                indexerQ.Reshape(
                    {bsz, seqlen, index_n_heads, index_head_dim});
                DeepSeekV4Linear(
                    attnInput,
                    weight[pre + ".attn.indexer.weights_proj.weight"],
                    Data(), indexerWeights, true);
                indexerWeights.Reshape({bsz, seqlen, index_n_heads});
            }
            weight[pre + ".attn.wq_b.weight"].tpLinearType = TP_LINEAR_ROW;
            DeepSeekV4Linear(qNorm, weight[pre + ".attn.wq_b.weight"], Data(), q);
            q.Reshape({bsz, seqlen, num_attention_heads, head_dim_full});
            DeepSeekV4Linear(attnInput, weight[pre + ".attn.wkv.weight"], Data(), kv, true);
            kv.Reshape({bsz, seqlen, 1, head_dim_full});
#ifdef USE_CUDA
            if (!graphCaptureHealthy("qkv", layer)) {
                return;
            }
#endif
            DeepSeekV4DecodeLayerCache *decodeCache = nullptr;
            if (useDecodeCache && layer < (int)activeDecodeLayerCaches.size()) {
                decodeCache = &activeDecodeLayerCaches[layer];
            }
            bool fusedQKVRopeCache = false;
#ifdef USE_CUDA
            if (graphSafeDecode && decodeMeta != nullptr && decodeCache != nullptr &&
                decodeCache->initialized && startPos > 0) {
                EnsureTensorOnSameCudaDevice(decodeCache->windowKV, kv);
                fusedQKVRopeCache = DeepSeekV4FusedQKVRopeCacheMultiCuda(
                    q, kv, weight[pre + ".attn.kv_norm.weight"], *decodeMeta,
                    qk_rope_head_dim, layerRopeBase, layerOriginalSeqLen,
                    rope_factor, rope_scaling_beta_fast, rope_scaling_beta_slow,
                    rms_norm_eps, head_dim_full - qk_rope_head_dim, 64,
                    window_size, decodeCache->windowKV, graphDevices);
            }
#endif

            if (!fusedQKVRopeCache) {
                RMSNormReference(kv, weight[pre + ".attn.kv_norm.weight"],
                                 rms_norm_eps, kv, DataType::BFLOAT16);
#ifdef USE_CUDA
                bool graphQRotaryDone = graphSafeDecode && decodeMeta != nullptr &&
                    DeepSeekV4ScaleQRotaryGraphMultiCuda(
                        q, qk_rope_head_dim, layerRopeBase, *decodeMeta,
                        layerOriginalSeqLen,
                        rope_factor, rope_scaling_beta_fast, rope_scaling_beta_slow,
                        rms_norm_eps);
                if (!graphQRotaryDone) {
                    if (graphSafeDecode) {
                        FastllmCudaSetThreadError();
                    }
#endif
                    ScaleQRatory(q, rms_norm_eps, qk_rope_head_dim, layerRopeBase, startPos,
                                 layerOriginalSeqLen, rope_factor, rope_scaling_beta_fast,
                                 rope_scaling_beta_slow);
#ifdef USE_CUDA
                }
                bool graphKvRotaryDone = graphSafeDecode && decodeMeta != nullptr &&
                    DeepSeekV4RotaryQuantGraphMultiCuda(
                        kv, qk_rope_head_dim, layerRopeBase, *decodeMeta,
                        layerOriginalSeqLen,
                        rope_factor, rope_scaling_beta_fast, rope_scaling_beta_slow,
                        head_dim_full - qk_rope_head_dim, 64, 1);
                if (!graphKvRotaryDone) {
                    if (graphSafeDecode) {
                        FastllmCudaSetThreadError();
                    }
#endif
                    DeepSeekV4RotaryQuant(kv, qk_rope_head_dim, layerRopeBase, startPos,
                                          layerOriginalSeqLen, rope_factor,
                                          rope_scaling_beta_fast, rope_scaling_beta_slow,
                                          head_dim_full - qk_rope_head_dim, 64);
#ifdef USE_CUDA
                }
#endif
            }
            kv.Reshape({bsz, seqlen, head_dim_full});
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
                    decodeCache->indexerCompressorWideDim =
                        hasLearnedIndexer ? 2 * index_head_dim : 0;
                    decodeCache->indexerCompressorRawTokenBase = 0;
                    StoreWindowKVCache(kv, bsz, seqlen, head_dim_full, startPos, window_size,
                                       decodeCache->windowKV);
                } else {
                    if (!decodeCache->initialized) {
                        ErrorInFastLLM("DeepSeekV4Model: decode cache is not initialized.");
                    }
                    if (seqlen > 1 && !graphSafeDecode) {
#ifdef USE_CUDA
                        EnsureTensorOnSameCudaDevice(decodeCache->windowKV, kv);
#endif
                        chunkPrefixLen = BuildWindowKVPrefixData(decodeCache->windowKV, bsz, head_dim_full,
                                                                 startPos, window_size, chunkPrefixKV);
                    }
                    decodeCache->totalLen = startPos + seqlen;
                    if (!fusedQKVRopeCache) {
                        UpdateWindowKVCache(kv, bsz, head_dim_full, startPos, window_size,
                                            decodeCache->windowKV
#ifdef USE_CUDA
                                            , graphSafeDecode ? decodeMeta : nullptr
#endif
                                            );
                    }
                }
            }
            const Data *decodeCompressedKVForAttention = nullptr;
            if (decodeCache != nullptr) {
                decodeCompressedKVForAttention = &decodeCache->compressedKV;
            }
            if (compressRatio > 0) {
                if (decodeCache != nullptr) {
                    ComputeCompressorRaw(weight, pre + ".attn.compressor", attnInput, compressorKV, compressorScore);
                    if (hasLearnedIndexer) {
                        ComputeCompressorRaw(
                            weight, pre + ".attn.indexer.compressor",
                            attnInput, indexerCompressorKV,
                            indexerCompressorScore);
                    }
                    bool restoredMultiTokenCompressedBuild =
                        requestState != nullptr &&
                        requestState->restoredHistoryCache &&
                        startPos > 0 && seqlen > 1;
#ifdef USE_CUDA
                    if (restoredMultiTokenCompressedBuild) {
                        // Compressor projections finish on the persistent
                        // MultiCUDA worker streams, while the restored raw-tail
                        // append below is issued by this scheduler thread.  Make
                        // the producer-to-copy handoff explicit; synchronizing
                        // only after CatDirect can preserve data that was already
                        // copied before its producer completed.
                        SynchronizeDeepSeekV4TensorParallelDevices(
                            this->deviceMap);
                    }
                    bool graphCompressedUpdated = false;
                    if (graphSafeDecode) {
                        graphCompressedUpdated = decodeMeta != nullptr &&
                            DeepSeekV4UpdateCompressedKVGraphMultiCuda(
                                compressorKV, compressorScore, decodeCache->cudaGraphApe,
                                decodeCache->cudaGraphNormWeight,
                                *decodeMeta, compressRatio,
                                head_dim_full, qk_rope_head_dim, layerRopeBase,
                                layerOriginalSeqLen, rope_factor,
                                rope_scaling_beta_fast, rope_scaling_beta_slow,
                                decodeCache->cudaGraphCompressorKVRing,
                                decodeCache->cudaGraphCompressorScoreRing,
                                decodeCache->compressedKV);
                        if (!graphCompressedUpdated) {
                            FastllmCudaSetThreadError();
                        } else {
                            decodeCache->compressedBlocks =
                                (startPos + seqlen) / compressRatio;
                            decodeCompressedCount = decodeCache->compressedBlocks;
                            decodeCompressedKVForAttention = &decodeCache->compressedKV;
                        }
                    }
                    if (!graphSafeDecode || !graphCompressedUpdated)
#endif
                    {
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
                        CopyTensorData(decodeCache->compressorKVRaw, compressorKV);
                        CopyTensorData(decodeCache->compressorScoreRaw, compressorScore);
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
#ifdef USE_CUDA
                        if (restoredMultiTokenCompressedBuild) {
                            // CatDirect appends each restored compressor tail on
                            // the scheduler thread's per-device stream.  The
                            // compressed-KV operator is dispatched to dedicated
                            // MultiCUDA worker streams, so make the handoff
                            // explicit before either worker consumes the newly
                            // appended rows.  Synchronizing after the build is
                            // too late: it only waits for a consumer that may
                            // already have read the old allocation contents.
                            SynchronizeDeepSeekV4TensorParallelDevices(
                                this->deviceMap);
                        }
#endif
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
                                if (!DeepSeekV4DsparkVerificationActive()) {
                                    TrimCompressorRawCache(bsz, decodeCache->totalLen, compressRatio,
                                                           decodeCache->compressorWideDim,
                                                           decodeCache->compressedBlocks,
                                                           decodeCache->compressorKVRaw,
                                                           decodeCache->compressorScoreRaw,
                                                           decodeCache->compressorRawTokenBase);
                                }
                            }
                            if (startPos == 0) {
                                Data catKV;
                                const Data *prefillCompressed = &decodeCache->compressedKV;
                                if (HasCompressedKVData(*prefillCompressed)) {
                                    Cat(kv, *prefillCompressed, 1, catKV);
                                    CopyTensorData(kv, catKV);
                                }
                            }
                            decodeCompressedKVForAttention = &decodeCache->compressedKV;
                        }
                    }
                    }

                    if (hasLearnedIndexer) {
                        bool graphIndexerUpdated = false;
#ifdef USE_CUDA
                        if (graphSafeDecode) {
                            graphIndexerUpdated = decodeMeta != nullptr &&
                                DeepSeekV4UpdateCompressedKVGraphMultiCuda(
                                    indexerCompressorKV,
                                    indexerCompressorScore,
                                    decodeCache->cudaGraphIndexerApe,
                                    decodeCache->cudaGraphIndexerNormWeight,
                                    *decodeMeta, 4, index_head_dim,
                                    qk_rope_head_dim, layerRopeBase,
                                    layerOriginalSeqLen, rope_factor,
                                    rope_scaling_beta_fast,
                                    rope_scaling_beta_slow,
                                    decodeCache->cudaGraphIndexerCompressorKVRing,
                                    decodeCache->cudaGraphIndexerCompressorScoreRing,
                                    decodeCache->indexerCompressedKV);
                            if (!graphIndexerUpdated) {
                                FastllmCudaSetThreadError();
                            } else {
                                decodeCache->indexerCompressedBlocks =
                                    (startPos + seqlen) / 4;
                            }
                        }
#endif
                        if (!graphSafeDecode || !graphIndexerUpdated) {
                            const int targetIndexerBlocks =
                                decodeCache->totalLen / 4;
                            const bool targetIndexerReady =
                                targetIndexerBlocks > 0 &&
                                decodeCache->indexerCompressedBlocks ==
                                    targetIndexerBlocks &&
                                HasCompressedKVData(
                                    decodeCache->indexerCompressedKV);
                            if (startPos == 0) {
                                CopyTensorData(
                                    decodeCache->indexerCompressorKVRaw,
                                    indexerCompressorKV);
                                CopyTensorData(
                                    decodeCache->indexerCompressorScoreRaw,
                                    indexerCompressorScore);
                                decodeCache->indexerCompressorRawTokenBase = 0;
                            } else {
                                AppendCompressorRaw(
                                    indexerCompressorKV,
                                    indexerCompressorScore, bsz, seqlen,
                                    decodeCache->indexerCompressorWideDim,
                                    decodeCache->indexerCompressorKVRaw,
                                    decodeCache->indexerCompressorScoreRaw);
                            }
                            if (!targetIndexerReady) {
                                bool builtIndexer = BuildCompressedKVFromRaw(
                                    weight,
                                    pre + ".attn.indexer.compressor",
                                    decodeCache->indexerCompressorKVRaw,
                                    decodeCache->indexerCompressorScoreRaw,
                                    bsz,
                                    decodeCache->indexerCompressorRawTokenBase,
                                    decodeCache->totalLen, 4,
                                    index_head_dim, qk_rope_head_dim,
                                    layerRopeBase, rope_factor,
                                    rope_scaling_beta_fast,
                                    rope_scaling_beta_slow,
                                    layerOriginalSeqLen,
                                    decodeCache->indexerCompressedKV, true);
                                if (builtIndexer) {
                                    decodeCache->indexerCompressedBlocks =
                                        GetReusableCompressedBlocks(
                                            decodeCache->indexerCompressedKV,
                                            bsz, targetIndexerBlocks,
                                            index_head_dim);
                                }
                            }
                            if (!DeepSeekV4DsparkVerificationActive()) {
                                TrimCompressorRawCache(
                                    bsz, decodeCache->totalLen, 4,
                                    decodeCache->indexerCompressorWideDim,
                                    decodeCache->indexerCompressedBlocks,
                                    decodeCache->indexerCompressorKVRaw,
                                    decodeCache->indexerCompressorScoreRaw,
                                    decodeCache->indexerCompressorRawTokenBase);
                            }
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
                        CopyTensorData(kv, catKV);
                    }
                }
            }
#ifdef USE_CUDA
            if (!graphCaptureHealthy("kv-cache-compressor", layer)) {
                return;
            }
#endif
            const Data *activeIndexerIndices = nullptr;
            const Data *activeIndexerLengths = nullptr;
#ifdef USE_CUDA
            if (graphSafeDecode && useLearnedIndexerScores &&
                decodeCache != nullptr &&
                decodeMeta != nullptr) {
                bool builtIndexerTopK =
                    DeepSeekV4BuildIndexerTopKGraphMultiCuda(
                        indexerQ, indexerWeights,
                        decodeCache->indexerCompressedKV, *decodeMeta, 4,
                        layerRopeBase, layerOriginalSeqLen, rope_factor,
                        rope_scaling_beta_fast, rope_scaling_beta_slow,
                        decodeCache->cudaGraphIndexerIndices,
                        decodeCache->cudaGraphIndexerLengths, graphDevices);
                if (!builtIndexerTopK) {
                    FastllmCudaSetThreadError();
                } else {
                    activeIndexerIndices =
                        &decodeCache->cudaGraphIndexerIndices;
                    activeIndexerLengths =
                        &decodeCache->cudaGraphIndexerLengths;
                }
            }
            if (!graphCaptureHealthy("indexer-topk", layer)) {
                return;
            }
#endif
#ifdef USE_CUDA
            if (requestState != nullptr && requestState->restoredHistoryCache &&
                decodeCache != nullptr && compressRatio > 0 &&
                startPos > 0 && seqlen > 1) {
                // Restoring a prefix moves the compressed cache back to both
                // CUDA ranks.  The rebuild above runs on the per-rank worker
                // streams, while the following Cat can be dispatched from the
                // caller stream immediately.  Complete both producers before
                // the restored cache is consumed by sparse prefill attention.
                SynchronizeDeepSeekV4TensorParallelDevices(this->deviceMap);
            }
#endif
            Data sparsePrefillKV;
            Data *sparsePrefillKVPtr = &kv;
            int sparsePrefillPrefixLen = 0;
            if (decodeCache != nullptr && startPos > 0 && seqlen > 1 &&
                !graphSafeDecode) {
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
                    CopyTensorData(sparsePrefillKV, kv);
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
                        CopyTensorData(sparsePrefillKV, catKV);
                    }
                }
                sparsePrefillKVPtr = &sparsePrefillKV;
            }
            Data compressedTopK;
            const Data *compressedTopKPtr = nullptr;
            if (compressRatio == 4 && !DeepSeekV4PreferCuda() &&
                !EnvFlagEnabled("FASTLLM_DSV4_DISABLE_CPU_INDEXER") &&
                startPos + seqlen >= compressRatio) {
                bool indexerReady = PrepareCpuIndexerTopK(
                    weight, pre + ".attn", attnInput, qNorm, bsz, seqlen,
                    startPos, compressRatio, index_n_heads, index_head_dim,
                    index_topk, qk_rope_head_dim, layerRopeBase,
                    layerOriginalSeqLen, rope_factor,
                    rope_scaling_beta_fast, rope_scaling_beta_slow,
                    decodeCache, compressedTopK);
                AssertInFastLLM(
                    indexerReady,
                    "DeepSeekV4Model: CPU CSA indexer failed to produce top-k indices.\n");
                compressedTopKPtr = &compressedTopK;
            }
            if (decodeCache != nullptr && startPos > 0 &&
                (seqlen == 1 || graphSafeDecode)) {
                SparseAttentionDecodeCachedReference(q, decodeCache->windowKV,
                                                     *decodeCompressedKVForAttention, weight[pre + ".attn.attn_sink"],
                                                     window_size, startPos, decodeCompressedCount,
                                                     qk_rope_head_dim, layerRopeBase,
                                                     1.0f / std::sqrt((float)head_dim_full), attnOut4,
                                                     layerOriginalSeqLen, rope_factor,
                                                     rope_scaling_beta_fast, rope_scaling_beta_slow
#ifdef USE_CUDA
                                                     , graphSafeDecode ? decodeMeta : nullptr,
                                                     graphSafeDecode ? compressRatio : 0,
                                                     graphSafeDecode ?
                                                        &decodeCache->cudaGraphPackedWindowKV : nullptr,
                                                     graphSafeDecode ?
                                                        &decodeCache->cudaGraphPackedCompressedKV : nullptr,
                                                     graphSafeDecode ?
                                                        &decodeWorkspace->sparseMlaScratch : nullptr,
                                                     activeIndexerIndices,
                                                     activeIndexerLengths,
                                                     compressedTopKPtr
#else
                                                     , nullptr, 0,
                                                     nullptr, nullptr,
                                                     nullptr, nullptr,
                                                     compressedTopKPtr
#endif
                                                     );
            } else {
                SparseAttentionReference(q, *sparsePrefillKVPtr, weight[pre + ".attn.attn_sink"], window_size,
                                         qk_rope_head_dim, layerRopeBase, startPos,
                                         1.0f / std::sqrt((float)head_dim_full), attnOut4,
                                         compressRatio, layerOriginalSeqLen, rope_factor,
                                         rope_scaling_beta_fast, rope_scaling_beta_slow,
                                         sparsePrefillPrefixLen,
                                         false, nullptr, compressedTopKPtr);
            }
            DeepSeekV4WoA(attnOut4, weight[pre + ".attn.wo_a.weight"], o_groups, o_lora_rank, woAOut);
            DeepSeekV4Linear(woAOut, weight[pre + ".attn.wo_b.weight"], Data(), attnOut);
#ifdef USE_CUDA
            if (!graphCaptureHealthy("attention-output", layer)) {
                return;
            }
#endif
            bool fusedFfnHcPostPreNorm = false;
#ifdef USE_CUDA
            if (graphSafeDecode) {
                fusedFfnHcPostPreNorm = DeepSeekV4HcPostPreNormMultiCuda(
                    attnOut, *curHiddenStates, attnMix.postData,
                    attnMix.combData, weight[pre + ".hc_ffn_fn"],
                    weight[pre + ".hc_ffn_scale"],
                    weight[pre + ".hc_ffn_base"],
                    weight[pre + ".ffn_norm.weight"], hc_mult,
                    hc_sinkhorn_iters, hc_eps, rms_norm_eps,
                    *nextHiddenStates, ffnInput, ffnMix.postData,
                    ffnMix.combData, graphDevices);
                if (fusedFfnHcPostPreNorm) {
                    std::swap(curHiddenStates, nextHiddenStates);
                }
            }
#endif
            if (!fusedFfnHcPostPreNorm) {
                runHcPost(attnOut, attnMix);
            }
            bool fusedFfnHcPreNorm = fusedFfnHcPostPreNorm;
            {
#ifdef USE_CUDA
                if (!fusedFfnHcPreNorm && cudaHcPreNormDecode) {
                    fusedFfnHcPreNorm = DeepSeekV4HcPreNormMultiCuda(
                        *curHiddenStates, weight[pre + ".hc_ffn_fn"],
                        weight[pre + ".hc_ffn_scale"], weight[pre + ".hc_ffn_base"],
                        weight[pre + ".ffn_norm.weight"], hc_mult, hc_sinkhorn_iters,
                        hc_eps, rms_norm_eps, ffnInput, ffnMix.postData,
                        ffnMix.combData, graphDevices);
                }
#endif
                if (!fusedFfnHcPreNorm) {
                    DeepSeekV4HcPre(*curHiddenStates, weight[pre + ".hc_ffn_fn"],
                                    weight[pre + ".hc_ffn_scale"], weight[pre + ".hc_ffn_base"],
                                    hc_mult, hc_sinkhorn_iters, hc_eps, rms_norm_eps,
                                    ffnMix.y, ffnMix.postData, ffnMix.combData);
                }
                ffnMix.b = bsz;
                ffnMix.s = seqlen;
                ffnMix.hc = hc_mult;
            }
            if (!fusedFfnHcPreNorm) {
                RMSNormReference(ffnMix.y, weight[pre + ".ffn_norm.weight"], rms_norm_eps,
                                 ffnInput, DataType::BFLOAT16);
            }
            std::vector<int> ffnDims = ffnInput.dims;
            ffnInput.Reshape({bsz * seqlen, dim});
            {
                BuildMoERoutingData(weight, pre + ".ffn", ffnInput, tokenIds, num_experts,
                                    num_experts_per_tok, scoring_func, routed_scaling_factor,
                                    expertIndex, expertScore
#ifdef USE_CUDA
                                    , graphSafeDecode ? decodeMeta : nullptr
#endif
                                    );
            }
#ifdef USE_CUDA
            if (!graphCaptureHealthy("router", layer)) {
                return;
            }
#endif
            {
                // MOE
                bool hasSharedExpertOut = false;
                bool fuseSharedExpert = false;
                Data *pairedReduceInput = nullptr;
                std::vector<Data*> moeWeights = weights[layer];
                auto sharedGateupIt = weight.weight.find(pre + ".ffn.shared_experts.gateup.weight");
                auto sharedDownIt = weight.weight.find(pre + ".ffn.shared_experts.w2.weight");
                Data preCopiedMoeInput;
                Data preCopiedExpertIndex;
                Data preCopiedExpertScore;
                Data *routedMoeInput = &ffnInput;
                Data *routedExpertIndex = &expertIndex;
                Data *routedExpertScore = &expertScore;
#ifdef USE_CUDA
                bool preCopiedNumasMoeInputs = false;
                bool canPreCopyNumasMoeInputs =
                    std::getenv(
                        "FASTLLM_DSV4_DISABLE_NUMAS_MOE_PRECOPY") == nullptr &&
                    ffnInput.dims.size() > 0 &&
                    moeWeights.size() > 2 &&
                    moeWeights[2] != nullptr &&
                    !moeWeights[2]->IsTensorParallelSharded() &&
                    GetTensorCudaDevice(ffnInput) >= 0;
                if (canPreCopyNumasMoeInputs) {
                    std::string selectedMoeDevice =
                        this->SelectMoeDeviceForLayer(layer);
                    std::transform(
                        selectedMoeDevice.begin(),
                        selectedMoeDevice.end(),
                        selectedMoeDevice.begin(),
                        [](unsigned char c) {
                            return (char)std::tolower(c);
                        });
                    canPreCopyNumasMoeInputs =
                        selectedMoeDevice == "numa" ||
                        selectedMoeDevice.rfind("numa:", 0) == 0;
                }
                if (canPreCopyNumasMoeInputs) {
                    preCopiedNumasMoeInputs =
                        CopyDeepSeekV4CudaTensorToCpu(
                            ffnInput, preCopiedMoeInput) &&
                        CopyDeepSeekV4CudaTensorToCpu(
                            expertIndex, preCopiedExpertIndex) &&
                        CopyDeepSeekV4CudaTensorToCpu(
                            expertScore, preCopiedExpertScore);
                    AssertInFastLLM(
                        preCopiedNumasMoeInputs,
                        "DeepSeek-V4 failed to copy tensor-parallel MoE inputs to NUMA.");
                    routedMoeInput = &preCopiedMoeInput;
                    routedExpertIndex = &preCopiedExpertIndex;
                    routedExpertScore = &preCopiedExpertScore;
                }
#endif
                const bool routedTensorParallel =
                    this->UseTensorParallelRoutedExperts();
                const bool routedExpertParallel =
                    !routedTensorParallel && DeepSeekV4DeviceSpecUsesType(
                        this->SelectMoeDeviceForLayer(layer), "multicuda");
                {
                    if (cudaSe && sharedGateupIt != weight.weight.end() && sharedDownIt != weight.weight.end() &&
                        !IsDiskWeight(&sharedGateupIt->second) && !IsDiskWeight(&sharedDownIt->second)) {
                        sharedGateupIt->second.tpLinearType = TP_LINEAR_ROW;
                        sharedGateupIt->second.tpPackType = TP_PACK_GATEUP;
                        sharedDownIt->second.tpLinearType = TP_LINEAR_COLUMN;
                        // DSpark verifies the target token together with all draft
                        // tokens, so its decode-shaped target pass has 1 + N rows.
                        // MultiCuda's expert-parallel path can accumulate each
                        // rank's shared-expert shard into its routed-expert partial
                        // before the final all-reduce for those rows as well.  Keep
                        // ordinary multi-token execution on the established path,
                        // while avoiding a second per-layer reduction in DSpark.
                        const bool fuseSharedExpertRows =
                            ffnInput.dims.size() > 0 &&
                            (ffnInput.dims[0] == 1 ||
                             dsparkGraphVerification);
                        // The expert-parallel path owns a complete routed
                        // expert partial per rank, so it can safely pre-add the
                        // local shared shard.  Tensor parallel must instead
                        // preserve two independent rank-ordered FP32 sums and
                        // their BF16 rounding boundaries; the M=8 verifier uses
                        // the paired collective below for that exact contract.
                        fuseSharedExpert =
                            routedExpertParallel && fuseSharedExpertRows;
                        if (!fuseSharedExpert) {
                            Data sharedInput;
                            Data *sharedInputPtr = &ffnInput;
                            if (!DeepSeekV4PreferCuda() && ffnInput.dataDevice == DataDevice::CPU &&
                                IsDeepSeekV4QuantizedLinearWeight(sharedGateupIt->second)) {
                                DeepSeekV4QuantizeLinearActivationCpu(ffnInput, sharedInput);
                                sharedInputPtr = &sharedInput;
                            }
                            LinearSwigluBlock(sharedInputPtr, &sharedGateupIt->second, GetEmptyData(),
                                              &sharedW3, &sharedW1);
                            const bool pairSharedReduction =
                                routedTensorParallel &&
                                dsparkGraphVerification &&
                                DeepSeekV4PairedAllReduceEnabled();
                            if (pairSharedReduction) {
                                AssertInFastLLM(
                                    DeepSeekV4LinearColumnLocal(
                                        sharedW1, sharedDownIt->second,
                                        *GetEmptyData(), sharedExpertOut),
                                    "DeepSeek-V4 failed to defer the shared-expert TP reduction.\n");
                                pairedReduceInput = &sharedExpertOut;
                            } else {
                                DeepSeekV4Linear(
                                    sharedW1, sharedDownIt->second,
                                    *GetEmptyData(), sharedExpertOut);
                                hasSharedExpertOut = true;
                            }
                            moeWeights[0] = moeWeights[1] = nullptr;
                        }
                    }
                }
                {
                    this->ApplyMoeDeviceMapForLayer(layer);
                    DataType effectiveMoeAtype = ffnInput.dataType;
                    Data moeQuantizedInput;
                    Data *moeInput = routedMoeInput;
                    if (!DeepSeekV4PreferCuda() && moeInput->dataDevice == DataDevice::CPU &&
                        moeWeights.size() > 2 && moeWeights[2] != nullptr &&
                        IsDeepSeekV4QuantizedLinearWeight(*moeWeights[2])) {
#ifdef USE_CUDA
                        int moeInputCudaDevice = GetTensorCudaDevice(*moeInput);
#endif
                        DeepSeekV4QuantizeLinearActivationCpu(*moeInput, moeQuantizedInput);
#ifdef USE_CUDA
                        if (moeInputCudaDevice >= 0 &&
                            moeQuantizedInput.dims.size() > 0 &&
                            moeQuantizedInput.dims[0] >= 32 &&
                            DeepSeekV4NumasGpuPrefillEnabled()) {
                            AssertInFastLLM(
                                AddDeepSeekV4QuantizedCudaReplica(
                                    moeQuantizedInput, moeInputCudaDevice),
                                "DeepSeek-V4 failed to stage its quantized NUMA MoE activation on CUDA.");
                        }
#endif
                        moeInput = &moeQuantizedInput;
                    }
                    MergeMOEBlock(moeInput, routedExpertIndex, routedExpertScore,
                                  &moeWeights, &biass[layer],
                                  &w1, &w2, &w3, &tempInput, &tempOutput,
                                  1.0f, &ffnOut, layer,
                                  ffnInput.dataType, effectiveMoeAtype,
                                  &moeInputTemp, &moeOutputTemp,
                                  MoeGateSwiglu, routedExpertParallel,
                                  swiglu_limit, true,
                                  pairedReduceInput);
                    ApplyDeviceMap(this->deviceMap, layer + 1, block_cnt);
                }
#ifdef USE_CUDA
                if (!graphCaptureHealthy("moe", layer)) {
                    return;
                }
#endif
                {
                    if (hasSharedExpertOut) {
                        if (!(ffnOut.multiDeviceData && sharedExpertOut.multiDeviceData)) {
                            ffnOut.ToDevice(sharedExpertOut.dataDevice);
                        }
                        AddTo(ffnOut, sharedExpertOut);
                    }
                }
            }
            {
                ffnOut.Reshape(ffnDims);
#ifdef USE_CUDA
                if (DeepSeekV4PreferCuda() && ffnOut.dataDevice == DataDevice::CPU && ffnOut.cpuData != nullptr) {
                    ffnOut.ToDevice(DataDevice::CUDA);
                }
#endif
                bool fusedNextAttnHcPreNorm = false;
#ifdef USE_CUDA
                if (layer + 1 < block_cnt && graphSafeDecode) {
                    std::string nextPre =
                        "layers." + std::to_string(layer + 1);
                    fusedNextAttnHcPreNorm =
                        DeepSeekV4HcPostPreNormMultiCuda(
                            ffnOut, *curHiddenStates, ffnMix.postData,
                            ffnMix.combData,
                            weight[nextPre + ".hc_attn_fn"],
                            weight[nextPre + ".hc_attn_scale"],
                            weight[nextPre + ".hc_attn_base"],
                            weight[nextPre + ".attn_norm.weight"], hc_mult,
                            hc_sinkhorn_iters, hc_eps, rms_norm_eps,
                            *nextHiddenStates, attnInput, attnMix.postData,
                            attnMix.combData, graphDevices);
                    if (fusedNextAttnHcPreNorm) {
                        std::swap(curHiddenStates, nextHiddenStates);
                    }
                }
#endif
                if (!fusedNextAttnHcPreNorm) {
                    runHcPost(ffnOut, ffnMix);
                }
                if (deepSeekV4DsparkTargetCapture != nullptr &&
                    std::find(dsparkTargetLayerIds.begin(),
                              dsparkTargetLayerIds.end(), layer) !=
                        dsparkTargetLayerIds.end()) {
                    DeepSeekV4HcMean(
                        *curHiddenStates,
                        deepSeekV4DsparkTargetCapture->targetHidden[layer]);
                }
                prefetchedAttnHcPreNorm = fusedNextAttnHcPreNorm;
            }
#ifdef USE_CUDA
            if (!graphCaptureHealthy("layer-end", layer)) {
                return;
            }
#endif
        }

            const bool dsparkVerification =
                DeepSeekV4DsparkVerificationActive();
            Data headStates;
            const Data *headSource = curHiddenStates;
            if (!dsparkVerification && seqlen > 1) {
                Split(*curHiddenStates, 1, seqlen - 1, seqlen, headStates);
                headSource = &headStates;
            }
#ifdef USE_CUDA
            if (!dsparkVerification && persistentTpAsync &&
                headSource->multiDeviceData &&
                headSource->IsTensorParallelReplicated()) {
                for (const auto &deviceData : headSource->multiDeviceDatas) {
                    if (deviceData.second == nullptr ||
                        deviceData.second->cudaData == nullptr) {
                        continue;
                    }
                    int device = GetPointerDeviceId(deviceData.second->cudaData);
                    if (device >= 0 &&
                        !MultiCudaCurrentThreadWaitForWorker(device)) {
                        FastllmCudaSyncDevice(device);
                    }
                    break;
                }
            }
#endif
            // Verification consumes the full-row HC head below.  Its ordinary
            // last-row head and sampling result are discarded by DSpark.
            if (!dsparkVerification) {
                HcHeadReference(
                    *headSource, weight["hc_head_fn"],
                    weight["hc_head_scale"], weight["hc_head_base"],
                    hc_mult, hc_eps, rms_norm_eps, headInput);
            }
            if (deepSeekV4DsparkTargetCapture != nullptr) {
#ifdef USE_CUDA
                if (persistentTpAsync && !graphSafeDecode) {
                    SynchronizeDeepSeekV4TensorParallelDevices(
                        this->deviceMap);
                }
#endif
                HcHeadReference(
                    *curHiddenStates, weight["hc_head_fn"],
                    weight["hc_head_scale"], weight["hc_head_base"],
                    hc_mult, hc_eps, rms_norm_eps,
                    deepSeekV4DsparkTargetCapture->headInput);
            }
#ifdef USE_CUDA
            if (!graphCaptureHealthy("head", block_cnt)) {
                return;
            }
#endif
#ifdef USE_CUDA
            const Data *graphSamplingHeadInput =
                dsparkVerification &&
                    deepSeekV4DsparkTargetCapture != nullptr ?
                    &deepSeekV4DsparkTargetCapture->headInput : &headInput;
            const int graphSamplingRows =
                dsparkVerification ? bsz * seqlen : bsz;
            if (graphSafeDecode && persistentTpAsync &&
                graphSamplingHeadInput->dataDevice == DataDevice::CUDA &&
                graphSamplingHeadInput->cudaData != nullptr) {
                int headDevice = GetPointerDeviceId(
                    graphSamplingHeadInput->cudaData);
                bool staleSamplingRoot = samplingHeadRoot.cudaData == nullptr ||
                    samplingHeadRoot.dataType !=
                        graphSamplingHeadInput->dataType ||
                    samplingHeadRoot.dims !=
                        graphSamplingHeadInput->dims ||
                    GetPointerDeviceId(samplingHeadRoot.cudaData) != headDevice;
                if (staleSamplingRoot) {
                    if (graphState && graphState->capturing) {
                        FastllmCudaSetThreadError();
                        return;
                    }
                    ResetData(samplingHeadRoot);
                    samplingHeadRoot.dataType =
                        graphSamplingHeadInput->dataType;
                    samplingHeadRoot.Resize(
                        graphSamplingHeadInput->dims);
                    samplingHeadRoot.dataDevice = DataDevice::CUDA;
                    samplingHeadRoot.dataDeviceIds = {headDevice};
                    FastllmCudaSetDevice(headDevice);
                    samplingHeadRoot.Allocate(false);
                }
                FastllmCudaSetDevice(headDevice);
                FastllmCudaCopyFromDeviceToDevice(
                    samplingHeadRoot.cudaData,
                    graphSamplingHeadInput->cudaData,
                    graphSamplingHeadInput->GetBytes());
                bool repeatedSamplingHead = MultiCudaRepeatToReplicated(
                    samplingHeadRoot, (int)samplingHeadRoot.dims.size() - 1,
                    1, samplingHeadReplicated);
                if (!repeatedSamplingHead) {
                    FastllmCudaSetThreadError();
                    return;
                }
                RMSNorm(samplingHeadReplicated, weight["norm.weight"],
                        rms_norm_eps, samplingHeadNorm);
                Linear(samplingHeadNorm, weight["head.weight"],
                       *GetEmptyData(), samplingLogits);
                ToDataType(samplingLogits, samplingLogitsFloat,
                           DataType::FLOAT32);
                bool greedyCandidatesReady =
                    samplingLogitsFloat.multiDeviceData &&
                    samplingLogitsFloat.IsTensorParallelSharded() &&
                    DeepSeekV4GraphTensorMatches(
                        samplingGreedyIds, DataType::INT32,
                        {graphSamplingRows}, graphDevices) &&
                    DeepSeekV4GraphTensorMatches(
                        samplingGreedyScores, DataType::FLOAT32,
                        {graphSamplingRows}, graphDevices);
                if (!greedyCandidatesReady && !graphState->capturing) {
                    greedyCandidatesReady = DeepSeekV4AllocateGraphTensor(
                        samplingGreedyIds, DataType::INT32,
                        {graphSamplingRows},
                        graphDevices, false) &&
                        DeepSeekV4AllocateGraphTensor(
                            samplingGreedyScores, DataType::FLOAT32,
                            {graphSamplingRows},
                            graphDevices, false);
                }
                if (greedyCandidatesReady) {
                    std::vector<char> sampled(graphDevices.size(), 0);
                    RunDeepSeekV4MultiCuda(graphDevices, [&](int rank, int device) {
                        auto logitsIt = samplingLogitsFloat.multiDeviceDatas.find(device);
                        auto idsIt = samplingGreedyIds.multiDeviceDatas.find(device);
                        auto scoresIt = samplingGreedyScores.multiDeviceDatas.find(device);
                        if (logitsIt == samplingLogitsFloat.multiDeviceDatas.end() ||
                            idsIt == samplingGreedyIds.multiDeviceDatas.end() ||
                            scoresIt == samplingGreedyScores.multiDeviceDatas.end() ||
                            logitsIt->second == nullptr || idsIt->second == nullptr ||
                            scoresIt->second == nullptr) {
                            return;
                        }
                        int localVocab = logitsIt->second->dims.back();
                        sampled[rank] = FastllmCudaGreedySamplingWithScores(
                            (float*)logitsIt->second->cudaData,
                            (int*)idsIt->second->cudaData,
                            (float*)scoresIt->second->cudaData,
                            graphSamplingRows, localVocab);
                    });
                    if (!std::all_of(sampled.begin(), sampled.end(),
                                     [](int state) { return state != 0; })) {
                        FastllmCudaSetThreadError();
                        return;
                    }
                } else {
                    FastllmCudaSetThreadError();
                    return;
                }
            }
#endif
#ifdef USE_CUDA
            // The three HC captures, main projection and three stage-KV
            // projections have a fixed verifier shape.  Running them after
            // graph replay used to spend several milliseconds dispatching a
            // few hundred microseconds of GPU work rank by rank.  Capture the
            // fixed work with the target model; rejection handling below only
            // commits the accepted prefix to the rolling draft windows.
            if (dsparkVerification && graphSafeDecode && persistentTpAsync &&
                deepSeekV4DsparkTargetCapture != nullptr) {
                std::vector<const Data*> capturedFeatures;
                capturedFeatures.reserve(dsparkTargetLayerIds.size());
                bool graphContextReady = !dsparkTargetLayerIds.empty();
                for (int layerId : dsparkTargetLayerIds) {
                    auto hiddenIt =
                        deepSeekV4DsparkTargetCapture->targetHidden.find(
                            layerId);
                    if (hiddenIt ==
                            deepSeekV4DsparkTargetCapture->targetHidden.end() ||
                        hiddenIt->second.dims !=
                            std::vector<int>({bsz, seqlen, embed_dim})) {
                        graphContextReady = false;
                        break;
                    }
                    capturedFeatures.push_back(&hiddenIt->second);
                }
                if (!graphContextReady) {
                    FastllmCudaSetThreadError();
                    return;
                }

                Data *combined = nullptr;
                if (capturedFeatures.size() == 1) {
                    Copy(*capturedFeatures[0], dsparkTargetCombined);
                    combined = &dsparkTargetCombined;
                } else {
                    Cat(*capturedFeatures[0], *capturedFeatures[1], -1,
                        dsparkTargetCombinedTemp);
                    combined = &dsparkTargetCombinedTemp;
                    for (int feature = 2;
                         feature < (int)capturedFeatures.size(); ++feature) {
                        Data *output =
                            combined == &dsparkTargetCombinedTemp ?
                                &dsparkTargetCombined :
                                &dsparkTargetCombinedTemp;
                        Cat(*combined, *capturedFeatures[feature], -1,
                            *output);
                        combined = output;
                    }
                }

                ApplyDeviceMap(this->deviceMap, block_cnt, block_cnt);
                DeepSeekV4Linear(
                    *combined, weight["mtp.0.main_proj.weight"], Data(),
                    dsparkTargetProjected, true);
                RMSNormReference(
                    dsparkTargetProjected,
                    weight["mtp.0.main_norm.weight"], rms_norm_eps,
                    dsparkTargetMainHidden, DataType::BFLOAT16);

                if ((int)dsparkTargetStageKV.size() != dsparkLayers) {
                    if (graphState != nullptr && graphState->capturing) {
                        FastllmCudaSetThreadError();
                        return;
                    }
                    dsparkTargetStageKV.resize(dsparkLayers);
                }
                Data *contextDecodeMeta = graphState == nullptr ? nullptr :
                    graphState->GetReplicatedDecodeMeta();
                if (contextDecodeMeta == nullptr) {
                    FastllmCudaSetThreadError();
                    return;
                }
                for (int stage = 0; stage < dsparkLayers; ++stage) {
                    const int layerId = std::max(
                        0, block_cnt - dsparkLayers + stage);
                    ApplyDeviceMap(this->deviceMap, layerId + 1, block_cnt);
                    const std::string prefix =
                        "mtp." + std::to_string(stage) + ".attn";
                    Data &stageKV = dsparkTargetStageKV[stage];
                    DeepSeekV4Linear(
                        dsparkTargetMainHidden,
                        weight[prefix + ".wkv.weight"], Data(), stageKV,
                        true);
                    stageKV.Reshape({bsz, seqlen, 1, head_dim_full});
                    RMSNormReference(
                        stageKV, weight[prefix + ".kv_norm.weight"],
                        rms_norm_eps, stageKV, DataType::BFLOAT16);
                    if (!DeepSeekV4RotaryQuantGraphMultiCuda(
                            stageKV, qk_rope_head_dim, rope_base,
                            *contextDecodeMeta, 0, rope_factor,
                            rope_scaling_beta_fast,
                            rope_scaling_beta_slow,
                            head_dim_full - qk_rope_head_dim, 64, 1)) {
                        FastllmCudaSetThreadError();
                        return;
                    }
                    stageKV.Reshape({bsz, seqlen, head_dim_full});
                }
                if (!graphCaptureHealthy("dspark-context", block_cnt)) {
                    return;
                }
            }
#endif
#ifdef USE_CUDA
            if (persistentTpAsync && seqlen > 1 && !graphSafeDecode) {
                // Multi-token TP uses the same ordered worker-stream handoff as
                // decode.  Drain once at the model-body boundary so prefill
                // temporaries and cache views cannot be released while a rank
                // still consumes them, without paying a device synchronize at
                // every replicated/sharded operator.
                SynchronizeDeepSeekV4TensorParallelDevices(this->deviceMap);
            }
#endif
        };

        bool modelBodyDone = false;
#ifdef USE_CUDA
        // Candidates are consumed asynchronously only when this invocation
        // entered with an already-instantiated graph.  The first launch at the
        // end of capture is numerically valid too, but keeping that token on the
        // regular sampling path leaves an unpinned set of sampling temporaries
        // in the CUDA pool.  A following request can then warm and capture its
        // own graph without attempting cudaMalloc while a stream is capturing.
        bool graphReplayReadyForSampling = graphSafeDecode && graphState &&
            graphState->captured;
        if (graphSafeDecode) {
            // Draft and target graphs share FastLLM's CUDA allocation pool.
            // DSpark therefore warms the target once more after the draft has
            // captured (and pinned its own temporary blocks): rounds one and
            // two heat both workspaces, round three captures draft and performs
            // the target's final warmup, and round four can capture target from
            // a disjoint set of idle blocks. Ordinary one-token decode has no
            // second graph competing for the pool and retains one warmup.
            const int requiredGraphWarmups =
                dsparkGraphVerification ? 3 : 1;
            auto syncGraphDevices = [&]() {
                int oldDevice = FastllmCudaGetDevice();
                for (int device : graphDevices) {
                    FastllmCudaSyncDevice(device);
                }
                FastllmCudaSetDevice(oldDevice);
            };
            auto launchGraphs = [&]() {
                bool ok = true;
                auto stageReplayInputs = [&](
                        int device,
                        DeepSeekV4CudaGraphDeviceState *deviceState) {
                    if (!graphState->replayInputsPending) {
                        return true;
                    }
                    if (graphGpuReplayInputs) {
                        if (targetGpuInput == nullptr ||
                            targetGpuInput->readySeen == nullptr ||
                            deviceState == nullptr ||
                            deviceState->decodeMeta == nullptr) {
                            return false;
                        }
                        auto seenIt = targetGpuInput->readySeen->
                            multiDeviceDatas.find(device);
                        if (seenIt == targetGpuInput->readySeen->
                                multiDeviceDatas.end() ||
                            seenIt->second == nullptr ||
                            seenIt->second->cudaData == nullptr) {
                            return false;
                        }
                        float *deviceInputIds =
                            device == graphState->inputDevice ?
                                (float*)graphState->inputIds.cudaData :
                                nullptr;
                        return
                            FastllmCudaDeepSeekV4DsparkPrepareTargetPeer(
                                (const uint32_t*)targetGpuInput->
                                    readySignal->cudaData,
                                (uint32_t*)seenIt->second->cudaData,
                                (const int*)targetGpuInput->
                                    proposalIds->cudaData,
                                targetGpuInput->proposalCount,
                                targetGpuInput->anchorToken,
                                targetGpuInput->startPos,
                                (int32_t*)deviceState->decodeMeta->cudaData,
                                deviceInputIds);
                    }
                    bool staged = deviceState != nullptr &&
                        deviceState->decodeMeta != nullptr &&
                        FastllmCudaCopyFromPinnedHostToDeviceAsyncCurrentThread(
                            deviceState->decodeMeta->cudaData,
                            graphState->pinnedMeta,
                            kDeepSeekV4CudaGraphMetaInts * sizeof(int32_t));
                    if (staged && device == graphState->inputDevice) {
                        staged =
                            FastllmCudaCopyFromPinnedHostToDeviceAsyncCurrentThread(
                                graphState->inputIds.cudaData,
                                graphState->pinnedInputIds,
                                graphState->inputIds.GetBytes());
                    }
                    return staged;
                };
                bool parallelLaunch = graphState->launchOrder.size() > 1;
                bool launchedInParallel = false;
                if (parallelLaunch) {
                    std::vector<int> launchOk(graphState->launchOrder.size(), 0);
                    std::atomic<int> launchReady{0};
                    const int launchCount =
                        (int)graphState->launchOrder.size();
                    std::function<void(int, int)> launchOne =
                        [&](int rank, int device) {
                            auto it = graphState->deviceIndex.find(device);
                            DeepSeekV4CudaGraphDeviceState *deviceState =
                                it == graphState->deviceIndex.end() ? nullptr : it->second;
                            bool ready = stageReplayInputs(
                                    device, deviceState) &&
                                deviceState != nullptr &&
                                deviceState->exec != nullptr;
                            // Enqueue every rank's replay-input copies first.
                            // Releasing the graph launches together keeps rank 0
                            // from reaching the first TP collective while peer
                            // worker threads are still staging or dispatching.
                            launchReady.fetch_add(
                                1, std::memory_order_release);
                            while (launchReady.load(
                                       std::memory_order_acquire) <
                                   launchCount) {
                                std::this_thread::yield();
                            }
                            launchOk[rank] = ready &&
                                FastllmCudaGraphLaunch(deviceState->exec);
                            if (launchOk[rank] &&
                                deviceState->replayDoneEvent != nullptr) {
                                FastllmCudaEventRecordCurrentThread(
                                    deviceState->replayDoneEvent);
                            }
                        };
                    bool previousAsync =
                        MultiCudaSetPersistentAsyncDispatch(true);
                    launchedInParallel = MultiCudaRunDeviceCallbacks(
                        graphState->launchOrder, launchOne);
                    MultiCudaSetPersistentAsyncDispatch(previousAsync);
                    if (launchedInParallel) {
                        ok = std::all_of(
                            launchOk.begin(), launchOk.end(),
                            [](int state) { return state != 0; });
                    }
                }
                if (!launchedInParallel) {
                    for (int device : graphState->launchOrder) {
                        auto it = graphState->deviceIndex.find(device);
                        DeepSeekV4CudaGraphDeviceState *deviceState =
                            it == graphState->deviceIndex.end() ? nullptr : it->second;
                        if (deviceState == nullptr) {
                            ok = false;
                            continue;
                        }
                        FastllmCudaSetDevice(deviceState->device);
                        bool launched = stageReplayInputs(
                                device, deviceState) &&
                            deviceState->exec != nullptr &&
                            FastllmCudaGraphLaunch(deviceState->exec);
                        if (launched && deviceState->replayDoneEvent != nullptr) {
                            FastllmCudaEventRecordCurrentThread(
                                deviceState->replayDoneEvent);
                        }
                        ok = launched && ok;
                    }
                }
                graphState->replayInputsPending = false;
                return ok;
            };
            auto disableCapturedGraph = [&](const char *stage) {
                syncGraphDevices();
                graphState->DestroyCapturedGraph();
                graphState->disabled = true;
                std::fprintf(stderr,
                             "[Fastllm] DeepSeek-V4 full decode CUDA graph disabled at %s: %s\n",
                             stage, FastllmCudaGraphLastError());
                std::fflush(stderr);
            };

            if (graphState->disabled) {
                FastllmCudaClearThreadError();
                runModelBody();
                modelBodyDone = true;
            } else if (graphState->captured) {
                if (launchGraphs()) {
                    modelBodyDone = true;
                } else {
                    AssertInFastLLM(
                        !graphGpuReplayInputs,
                        "DSpark GPU proposal handoff failed during target replay.");
                    disableCapturedGraph("replay");
                    FastllmCudaClearThreadError();
                    runModelBody();
                    modelBodyDone = true;
                }
            } else if (graphState->warmupRounds <
                       requiredGraphWarmups) {
                FastllmCudaClearThreadError();
                FastllmCudaClearGraphError();
                runModelBody();
                modelBodyDone = true;
                if (FastllmCudaGetThreadError() || FastllmCudaGetGraphError()) {
                    graphState->disabled = true;
                    std::fprintf(stderr,
                                 "[Fastllm] DeepSeek-V4 full decode CUDA graph disabled: "
                                 "a graph-safe kernel rejected warmup.\n");
                    std::fflush(stderr);
                } else {
                    graphState->warmupRounds++;
                    graphState->warmed =
                        graphState->warmupRounds >=
                        requiredGraphWarmups;
                }
            } else {
                syncGraphDevices();
                // The graph allocation-failure sentinel is one real cudaMalloc
                // per device. Resolve all of them before starting rank 0's
                // stream capture; lazily allocating rank 1's sentinel while
                // rank 0 is already capturing invalidates rank 0's graph.
                bool captureOk = true;
                const char *failureStage = nullptr;
                for (auto &deviceState : graphState->devices) {
                    FastllmCudaSetDevice(deviceState->device);
                    if (!FastllmCudaGraphPrepareCaptureDevice()) {
                        captureOk = false;
                        failureStage = "prepare capture devices";
                        break;
                    }
                }
                if (captureOk && !FastllmCudaGraphMemoryPoolBegin()) {
                    captureOk = false;
                    failureStage = "workspace reservation";
                }
                int begunCaptures = 0;
                if (captureOk) {
                    for (auto &deviceState : graphState->devices) {
                        FastllmCudaSetDevice(deviceState->device);
                        if (!FastllmCudaGraphBeginCapture()) {
                            captureOk = false;
                            failureStage = "begin capture";
                            break;
                        }
                        begunCaptures++;
                    }
                }

                std::vector<void*> workerStartEvents;
                std::vector<void*> workerEndEvents;
                workerStartEvents.reserve(graphState->devices.size());
                workerEndEvents.reserve(graphState->devices.size());
                for (auto &deviceState : graphState->devices) {
                    workerStartEvents.push_back(deviceState->workerStartEvent);
                    workerEndEvents.push_back(deviceState->workerEndEvent);
                }
                bool workersJoined = false;
                if (captureOk) {
                    for (auto &deviceState : graphState->devices) {
                        FastllmCudaSetDevice(deviceState->device);
                        FastllmCudaEventRecordCurrentThread(deviceState->workerStartEvent);
                    }
                    workersJoined = MultiCudaGraphWorkersWaitEvents(
                        graphDevices, workerStartEvents);
                    if (!workersJoined) {
                        captureOk = false;
                        failureStage = "join persistent workers";
                    }
                }

                if (captureOk) {
                    FastllmCudaClearThreadError();
                    graphState->capturing = true;
                    runModelBody();
                    graphState->capturing = false;
                    if (FastllmCudaGetThreadError() || FastllmCudaGetGraphError()) {
                        captureOk = false;
                        failureStage = "captured model body";
                    }
                }

                if (workersJoined) {
                    if (!MultiCudaGraphWorkersRecordEvents(graphDevices, workerEndEvents)) {
                        captureOk = false;
                        failureStage = "rejoin persistent workers";
                    }
                    for (auto &deviceState : graphState->devices) {
                        FastllmCudaSetDevice(deviceState->device);
                        FastllmCudaCurrentThreadStreamWaitEvent(deviceState->workerEndEvent);
                    }
                }

                if (begunCaptures == (int)graphState->devices.size()) {
                    for (auto &deviceState : graphState->devices) {
                        FastllmCudaSetDevice(deviceState->device);
                        if (FastllmCudaGraphCaptureInvalidated()) {
                            captureOk = false;
                            failureStage = "invalidated capture";
                        }
                    }
                }

                bool endOk = begunCaptures == (int)graphState->devices.size();
                for (int i = 0; i < begunCaptures; i++) {
                    auto &deviceState = graphState->devices[i];
                    FastllmCudaSetDevice(deviceState->device);
                    void *capturedGraph = nullptr;
                    bool oneEndOk = FastllmCudaGraphEndCapture(&capturedGraph) &&
                                    capturedGraph != nullptr;
                    if (oneEndOk) {
                        deviceState->graph = capturedGraph;
                    } else if (capturedGraph != nullptr) {
                        FastllmCudaGraphDestroy(capturedGraph);
                    }
                    endOk &= oneEndOk;
                }
                if (!endOk) {
                    captureOk = false;
                    if (failureStage == nullptr) {
                        failureStage = "end capture";
                    }
                }

                if (captureOk) {
                    captureOk = FastllmCudaGraphMemoryPoolEnd(
                        graphState->reservedPointers);
                    if (!captureOk) {
                        failureStage = "pin captured workspace";
                    }
                } else {
                    FastllmCudaGraphMemoryPoolAbort();
                }

                if (captureOk) {
                    for (auto &deviceState : graphState->devices) {
                        FastllmCudaSetDevice(deviceState->device);
                        if (!FastllmCudaGraphInstantiate(
                                deviceState->graph, &deviceState->exec) ||
                            deviceState->exec == nullptr) {
                            captureOk = false;
                            failureStage = "instantiate";
                            break;
                        }
                    }
                }

                if (captureOk) {
                    graphState->captured = true;
                    if (launchGraphs()) {
                        modelBodyDone = true;
                    } else {
                        failureStage = "first launch";
                        captureOk = false;
                    }
                }

                if (!captureOk) {
                    graphState->capturing = false;
                    disableCapturedGraph(failureStage == nullptr ?
                                         "unknown capture stage" : failureStage);
                    FastllmCudaClearThreadError();
                    runModelBody();
                    modelBodyDone = true;
                }
            }

            for (int layer = 0; layer < (int)activeDecodeLayerCaches.size(); layer++) {
                DeepSeekV4DecodeLayerCache &cache = activeDecodeLayerCaches[layer];
                cache.totalLen = originalStartPos + seqlen;
                if (cache.compressRatio > 0) {
                    cache.compressedBlocks = cache.totalLen / cache.compressRatio;
                    if (cache.compressedKV.dims.size() == 3 &&
                        cache.compressedKV.dims[1] !=
                            cache.compressedBlocks) {
                        ResizeTensorSequenceInPlace(
                            cache.compressedKV,
                            cache.compressedBlocks);
                    }
                    if (cache.compressRatio == 4) {
                        cache.indexerCompressedBlocks =
                            cache.compressedBlocks;
                        if (cache.indexerCompressedKV.dims.size() == 3 &&
                            cache.indexerCompressedKV.dims[1] !=
                                cache.indexerCompressedBlocks) {
                            ResizeTensorSequenceInPlace(
                                cache.indexerCompressedKV,
                                cache.indexerCompressedBlocks);
                        }
                    }
                }
            }
        }
#endif
        if (!modelBodyDone) {
            runModelBody();
        }

        if (DeepSeekV4DsparkVerificationActive()) {
#ifdef USE_CUDA
            // A replayed verifier graph has already run final RMSNorm, the
            // tensor-parallel LM head and each rank's local top-1 reduction.
            // Publish those request-owned workspaces to the DSpark sampler;
            // the replay-done events let its root gather wait without a
            // device-wide synchronization.  Warmup, first capture, eager and
            // non-CUDA paths intentionally leave samplingReady false.
            bool graphSamplingReady =
                deepSeekV4DsparkTargetCapture != nullptr &&
                dsparkGraphVerification && graphSafeDecode &&
                persistentTpAsync && graphReplayReadyForSampling &&
                samplingLogitsFloat.dataType == DataType::FLOAT32 &&
                samplingLogitsFloat.multiDeviceData &&
                samplingLogitsFloat.IsTensorParallelSharded() &&
                DeepSeekV4GraphTensorMatches(
                    samplingGreedyIds, DataType::INT32,
                    {bsz * seqlen}, graphDevices) &&
                DeepSeekV4GraphTensorMatches(
                    samplingGreedyScores, DataType::FLOAT32,
                    {bsz * seqlen}, graphDevices);
            std::map<int, void*> samplingReadyEvents;
            if (graphSamplingReady) {
                for (int device : graphDevices) {
                    auto stateIt = graphState->deviceIndex.find(device);
                    if (stateIt == graphState->deviceIndex.end() ||
                        stateIt->second == nullptr ||
                        stateIt->second->replayDoneEvent == nullptr) {
                        graphSamplingReady = false;
                        samplingReadyEvents.clear();
                        break;
                    }
                    samplingReadyEvents[device] =
                        stateIt->second->replayDoneEvent;
                }
            }
            if (graphSamplingReady) {
                deepSeekV4DsparkTargetCapture->samplingLogitsFloat =
                    &samplingLogitsFloat;
                deepSeekV4DsparkTargetCapture->samplingGreedyIds =
                    &samplingGreedyIds;
                deepSeekV4DsparkTargetCapture->samplingGreedyScores =
                    &samplingGreedyScores;
                deepSeekV4DsparkTargetCapture->samplingReadyEvents =
                    std::move(samplingReadyEvents);
                deepSeekV4DsparkTargetCapture->samplingReady = true;

                bool graphContextReady =
                    (int)dsparkTargetStageKV.size() == dsparkLayers;
                for (int stage = 0;
                     stage < dsparkLayers && graphContextReady; ++stage) {
                    graphContextReady = DeepSeekV4GraphTensorMatches(
                        dsparkTargetStageKV[stage], DataType::BFLOAT16,
                        {bsz, seqlen, head_dim_full}, graphDevices);
                }
                if (graphContextReady) {
                    deepSeekV4DsparkTargetCapture->contextStageKV.clear();
                    deepSeekV4DsparkTargetCapture->contextStageKV.reserve(
                        dsparkLayers);
                    for (int stage = 0; stage < dsparkLayers; ++stage) {
                        deepSeekV4DsparkTargetCapture->contextStageKV.push_back(
                            &dsparkTargetStageKV[stage]);
                    }
                    deepSeekV4DsparkTargetCapture->contextRows = seqlen;
                    deepSeekV4DsparkTargetCapture->contextReady = true;
                }
            }
#endif
            // The speculative caller ignores ForwardBatch's sampled token, but
            // the public KV holders still need their logical length updated so
            // rejection rollback can truncate them to the accepted prefix.
            const int finalTotalLen = originalStartPos + inputIds.dims[1];
            UpdateDebugPastKeyValues(
                pastKeyValues, bsz, finalTotalLen, block_cnt);
            return std::vector<int>(batch, 0);
        }

        std::vector<int> ret;
        std::vector<int> samplingSeqLens(batch, 1);
        std::vector<GenerationConfig> generationConfigs(batch, generationConfig);
        PrepareDeepSeekV4SamplingConfig(generationConfigs[0]);
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
        Data samplingHeadInput;
        Data *samplingHeadInputPtr = &headInput;
#ifdef USE_CUDA
        // Keep both graph warmup and the first launch after capture on the
        // regular sampling path.  Steady-state replays use the graph-produced
        // sharded candidates without repeating norm / LM head / sampling.
        bool asyncShardedGreedy = graphSafeDecode && persistentTpAsync &&
            graphReplayReadyForSampling &&
            batch == 1 && generationConfigs[0].IsSimpleGreedy() &&
            generationConfigs[0].output_token_least <= 0 &&
            !generationConfigs[0].output_logits &&
            samplingLogitsFloat.dataType == DataType::FLOAT32 &&
            samplingLogitsFloat.multiDeviceData &&
            samplingLogitsFloat.IsTensorParallelSharded() &&
            DeepSeekV4DeviceMapUsesMultiCuda(this->deviceMap);
        std::map<int, void*> samplingGreedyReadyEvents;
        if (asyncShardedGreedy) {
            for (int device : graphDevices) {
                auto stateIt = graphState->deviceIndex.find(device);
                if (stateIt == graphState->deviceIndex.end() ||
                    stateIt->second == nullptr ||
                    stateIt->second->replayDoneEvent == nullptr) {
                    asyncShardedGreedy = false;
                    samplingGreedyReadyEvents.clear();
                    break;
                }
                samplingGreedyReadyEvents[device] =
                    stateIt->second->replayDoneEvent;
            }
        }
        if (graphSafeDecode && !asyncShardedGreedy &&
            headInput.dataDevice == DataDevice::CUDA &&
            headInput.cudaData != nullptr) {
            int headDevice = GetPointerDeviceId(headInput.cudaData);
            samplingHeadInput.dataType = headInput.dataType;
            samplingHeadInput.Resize(headInput.dims);
            samplingHeadInput.dataDevice = DataDevice::CUDA;
            samplingHeadInput.dataDeviceIds = {headDevice};
            FastllmCudaSetDevice(headDevice);
            samplingHeadInput.Allocate(false);
            FastllmCudaCopyFromDeviceToDevice(
                samplingHeadInput.cudaData, headInput.cudaData,
                headInput.GetBytes());
            samplingHeadInputPtr = &samplingHeadInput;
        }
#endif
        Data officialCpuSamplingLogits;
        Data *samplingPrecomputedLogits = nullptr;
        if (DeepSeekV4HeadLogitsCpu(*samplingHeadInputPtr, weight["norm.weight"],
                                    weight["head.weight"], rms_norm_eps,
                                    officialCpuSamplingLogits)) {
            samplingPrecomputedLogits = &officialCpuSamplingLogits;
        }
#ifdef USE_CUDA
        if (asyncShardedGreedy) {
            samplingPrecomputedLogits = &samplingLogitsFloat;
        }
#endif
        {
#ifdef USE_CUDA
            struct ScopedSamplingPersistentAsync {
                bool enabled;
                bool previous;
                explicit ScopedSamplingPersistentAsync(bool enabled) :
                        enabled(enabled),
                        previous(enabled ?
                            MultiCudaSetPersistentAsyncDispatch(true) : false) {}
                ~ScopedSamplingPersistentAsync() {
                    if (enabled) {
                        MultiCudaSetPersistentAsyncDispatch(previous);
                    }
                }
            } samplingAsyncScope(asyncShardedGreedy);
#endif
            LLMSamplingBlock(this, samplingHeadInputPtr,
                             &weight["norm.weight"], &weight["head.weight"],
                             rms_norm_eps, batch, true, samplingSeqLens,
                             samplingPastKeyValues, generationConfigs,
                             *samplingLastTokensPtr, retLogits, ret,
                             samplingPrecomputedLogits,
#ifdef USE_CUDA
                             asyncShardedGreedy ? &samplingGreedyIds : nullptr,
                             asyncShardedGreedy ? &samplingGreedyScores : nullptr,
                             asyncShardedGreedy ? &samplingGreedyReadyEvents : nullptr
#else
                             nullptr, nullptr, nullptr
#endif
            );
        }

        int finalTotalLen = originalStartPos + inputIds.dims[1];
        UpdateDebugPastKeyValues(pastKeyValues, bsz, finalTotalLen, block_cnt);
        if (!this->dsparkEnabled && this->saveHistoryChat &&
            !DeepSeekV4PrefixCacheDisabled() &&
            batch == 1 && finalTotalLen % 256 == 0 &&
            (int)activeHistoryTokens.size() >= finalTotalLen) {
#ifdef USE_CUDA
            // The recursive chunk owns a temporary workspace, while MultiCUDA
            // kernels use per-thread streams.  Drain every rank before copying
            // the cache snapshot and before that workspace is destroyed/reused
            // by the suffix chunk.
            SynchronizeDeepSeekV4TensorParallelDevices(this->deviceMap);
#endif
            this->RecordHistorySnapshot(activeHistoryTokens, finalTotalLen, activeDecodeLayerCaches);
        } else if (!this->dsparkEnabled && this->saveHistoryChat &&
                   !DeepSeekV4PrefixCacheDisabled() &&
                   batch == 1 && finalTotalLen % 256 == 0 && DeepSeekV4PrefixCacheDebugEnabled()) {
            printf("[fastllm-dsv4-prefix-cache] skip boundary record: final_len=%d history_tokens=%d\n",
                   finalTotalLen, (int)activeHistoryTokens.size());
            fflush(stdout);
        }
        if (requestState != nullptr && requestState->restoredHistoryCache &&
            originalStartPos > 0) {
            // Restore-specific raw-tail rebuilding is required only for the
            // first suffix.  Leaving the flag set makes every DSpark verifier
            // chunk take the expensive restored-prefill synchronization path.
            requestState->restoredHistoryCache = false;
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
        if (batch <= 0) {
            return {};
        }
        bool allDecodeTokens = (int)seqLens.size() == batch;
        for (int i = 0; i < batch; i++) {
            allDecodeTokens &= (seqLens[i] == 1);
        }
        if (!allDecodeTokens) {
            ErrorInFastLLM("DeepSeekV4Model::ForwardBatch only supports batched decode for now.");
        }
        if ((int)pastKeyValues.size() < batch * block_cnt ||
            (int)generationConfigs.size() < batch ||
            (int)lastTokens.units.size() < batch) {
            ErrorInFastLLM("DeepSeekV4Model::ForwardBatch got invalid batched decode inputs.");
        }

        Data decodeInputIds;
        Copy(inputIds, decodeInputIds);
        if ((int)decodeInputIds.Count(0) != batch) {
            ErrorInFastLLM("DeepSeekV4Model::ForwardBatch decode input size mismatch.");
        }
        decodeInputIds.Reshape({batch, 1});

        if (batch == 1) {
            std::shared_ptr<DeepSeekV4RequestState> requestState =
                GetRequestStateByFirstKey(pastKeyValues[0].first);
            if (!requestState) {
                ErrorInFastLLM("DeepSeekV4Model::ForwardBatch missing request state.");
            }

            std::vector<std::pair<Data, Data> > borrowedPastKeyValues(block_cnt);
            struct BorrowPastKeyValuesScope {
                std::vector<std::pair<Data*, Data*> > &source;
                std::vector<std::pair<Data, Data> > &borrowed;
                int count;

                BorrowPastKeyValuesScope(
                        std::vector<std::pair<Data*, Data*> > &source,
                        std::vector<std::pair<Data, Data> > &borrowed,
                        int count) : source(source), borrowed(borrowed), count(count) {
                    for (int i = 0; i < count; i++) {
                        AssertInFastLLM(source[i].first != nullptr &&
                                        source[i].second != nullptr,
                                        "DeepSeekV4Model::ForwardBatch got null KV holders.\n");
                        std::swap(borrowed[i].first, *source[i].first);
                        std::swap(borrowed[i].second, *source[i].second);
                    }
                }

                ~BorrowPastKeyValuesScope() {
                    for (int i = 0; i < count; i++) {
                        std::swap(borrowed[i].first, *source[i].first);
                        std::swap(borrowed[i].second, *source[i].second);
                    }
                }
            } borrowScope(pastKeyValues, borrowedPastKeyValues, block_cnt);
            DeepSeekV4RequestStateOverrideScope requestScope(requestState);

            Data emptyAttentionMask;
            const Data &singleAttentionMask =
                !attentionMask.empty() && attentionMask[0] != nullptr ?
                    *attentionMask[0] : emptyAttentionMask;
            int startPos = requestState->decodeLayerCaches.empty() ? 0 :
                           requestState->decodeLayerCaches[0].totalLen;
            if (!positionIds.empty() && positionIds[0] != nullptr &&
                positionIds[0]->Count(0) > 0) {
                std::vector<int> ids = ReadTokenIds(*positionIds[0]);
                if (!ids.empty()) {
                    startPos = ids[0];
                }
            }
            // The scheduler's per-sequence position tensor may be rank-1,
            // while the legacy single-request path expects [1, 1].
            Data singlePositionIds(
                DataType::FLOAT32, {1, 1}, {(float)startPos});
            return ForwardBatch(1, decodeInputIds, singleAttentionMask,
                                singlePositionIds, borrowedPastKeyValues,
                                generationConfigs[0], lastTokens, retLogits);
        }

        std::vector<int> startPositions(batch, 0);
        std::vector<std::shared_ptr<DeepSeekV4RequestState> > requestStates(batch);
        for (int b = 0; b < batch; b++) {
            requestStates[b] = GetRequestStateByFirstKey(pastKeyValues[b * block_cnt].first);
            if (!requestStates[b]) {
                ErrorInFastLLM("DeepSeekV4Model::ForwardBatch missing request state.");
            }
            if (positionIds.size() > b && positionIds[b] != nullptr && positionIds[b]->Count(0) > 0) {
                auto pids = ReadTokenIds(*positionIds[b]);
                startPositions[b] = pids.empty() ? 0 : pids[0];
            } else if (!requestStates[b]->decodeLayerCaches.empty()) {
                startPositions[b] = requestStates[b]->decodeLayerCaches[0].totalLen;
            }
            if (startPositions[b] == 0) {
                requestStates[b]->decodeLayerCaches.clear();
                requestStates[b]->decodeLayerCaches.resize(block_cnt);
            } else if ((int)requestStates[b]->decodeLayerCaches.size() < block_cnt) {
                ErrorInFastLLM("DeepSeekV4Model::ForwardBatch decode cache is not initialized.");
            }
        }

        Data hiddenStates, hiddenStatesBeforeHcExpand;
        Embedding(decodeInputIds, weight["embed.weight"], hiddenStatesBeforeHcExpand);
        if (hiddenStatesBeforeHcExpand.dataType != DataType::BFLOAT16) {
            ToDataType(hiddenStatesBeforeHcExpand, DataType::BFLOAT16);
        }

        int bsz = batch;
        int seqlen = 1;
        int dim = embed_dim;
        hiddenStatesBeforeHcExpand.Reshape({bsz, seqlen, 1, dim});
        Repeat(hiddenStatesBeforeHcExpand, 2, hc_mult, hiddenStates);

        std::vector<int> tokenIds = ReadTokenIds(decodeInputIds);
        if (this->saveHistoryChat && !DeepSeekV4PrefixCacheDisabled()) {
            for (int b = 0; b < batch; b++) {
                auto &historyTokens = requestStates[b]->historyTokens;
                if (startPositions[b] == 0) {
                    historyTokens = std::vector<int>{tokenIds[b]};
                } else if ((int)historyTokens.size() == startPositions[b]) {
                    historyTokens.push_back(tokenIds[b]);
                } else if ((int)historyTokens.size() < startPositions[b]) {
                    if (DeepSeekV4PrefixCacheDebugEnabled()) {
                        printf("[fastllm-dsv4-prefix-cache] reset token history: history=%d start=%d add=1\n",
                               (int)historyTokens.size(), startPositions[b]);
                        fflush(stdout);
                    }
                    historyTokens.clear();
                }
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
        Data ffnInput, ffnOut, expertIndex, expertScore;
        Data w1, w2, w3, tempInput, tempOutput, moeInputTemp, moeOutputTemp;
        auto runHcPost = [&](Data &input, const HcMix &mix) {
            DeepSeekV4HcPost(input, *curHiddenStates, mix.postData, mix.combData, *nextHiddenStates);
            std::swap(curHiddenStates, nextHiddenStates);
        };

        for (int layer = 0; layer < block_cnt; layer++) {
            std::string pre = "layers." + std::to_string(layer);
            ApplyDeviceMap(this->deviceMap, layer + 1, block_cnt);
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
            DeepSeekV4Linear(attnInput, weight[pre + ".attn.wq_a.weight"], Data(), qr, true);
            RMSNormReference(qr, weight[pre + ".attn.q_norm.weight"], rms_norm_eps, qNorm, DataType::BFLOAT16);
            weight[pre + ".attn.wq_b.weight"].tpLinearType = TP_LINEAR_ROW;
            DeepSeekV4Linear(qNorm, weight[pre + ".attn.wq_b.weight"], Data(), q);
            q.Reshape({bsz, seqlen, num_attention_heads, head_dim_full});

            std::vector<Data> qParts(batch);
            std::vector<Data*> qPartPtrs(batch);
            bool cpuIndexerLayer =
                compressRatio == 4 && !DeepSeekV4PreferCuda() &&
                !EnvFlagEnabled("FASTLLM_DSV4_DISABLE_CPU_INDEXER");
            std::vector<Data> indexerAttnInputParts(batch);
            std::vector<Data> indexerQNormParts(batch);
            std::vector<Data> compressedTopKParts(batch);
            std::vector<Data*> compressedTopKPtrs(batch, nullptr);
            for (int b = 0; b < batch; b++) {
                Split(q, 0, b, b + 1, qParts[b]);
                ScaleQRatory(qParts[b], rms_norm_eps, qk_rope_head_dim, layerRopeBase,
                             startPositions[b], layerOriginalSeqLen, rope_factor,
                             rope_scaling_beta_fast, rope_scaling_beta_slow);
                qPartPtrs[b] = &qParts[b];
                if (cpuIndexerLayer) {
                    Split(attnInput, 0, b, b + 1,
                          indexerAttnInputParts[b]);
                    Split(qNorm, 0, b, b + 1,
                          indexerQNormParts[b]);
                }
            }

            DeepSeekV4Linear(attnInput, weight[pre + ".attn.wkv.weight"], Data(), kv, true);
            RMSNormReference(kv, weight[pre + ".attn.kv_norm.weight"], rms_norm_eps, kv, DataType::BFLOAT16);
            kv.Reshape({bsz, seqlen, 1, head_dim_full});
            std::vector<Data> kvParts(batch);
            for (int b = 0; b < batch; b++) {
                Split(kv, 0, b, b + 1, kvParts[b]);
                DeepSeekV4RotaryQuant(kvParts[b], qk_rope_head_dim, layerRopeBase,
                                      startPositions[b], layerOriginalSeqLen, rope_factor,
                                      rope_scaling_beta_fast, rope_scaling_beta_slow,
                                      head_dim_full - qk_rope_head_dim, 64);
                kvParts[b].Reshape({1, 1, head_dim_full});
            }

            Data compressorKVAll, compressorScoreAll;
            if (compressRatio > 0) {
                ComputeCompressorRaw(weight, pre + ".attn.compressor", attnInput,
                                     compressorKVAll, compressorScoreAll);
            }

            std::vector<Data> attnOutParts(batch);
            std::vector<Data*> attnOutPtrs(batch);
            std::vector<Data*> windowKVPtrs(batch, nullptr);
            std::vector<Data*> compressedKVPtrs(batch, nullptr);
            std::vector<int> layerCompressedCounts(batch, 0);
            std::vector<char> cachedDecode(batch, 0);
            for (int b = 0; b < batch; b++) {
                auto &decodeCaches = requestStates[b]->decodeLayerCaches;
                DeepSeekV4DecodeLayerCache &decodeCache = decodeCaches[layer];
                int startPos = startPositions[b];
                int decodeCompressedCount = 0;

                if (startPos == 0) {
                    decodeCache.initialized = true;
                    decodeCache.bsz = 1;
                    decodeCache.totalLen = seqlen;
                    decodeCache.headDim = head_dim_full;
                    decodeCache.windowSize = window_size;
                    decodeCache.compressRatio = compressRatio;
                    decodeCache.compressorWideDim = (compressRatio == 4 ? 2 : 1) * head_dim_full;
                    decodeCache.compressorRawTokenBase = 0;
                    StoreWindowKVCache(kvParts[b], 1, seqlen, head_dim_full, startPos, window_size,
                                       decodeCache.windowKV);
                } else {
                    if (!decodeCache.initialized) {
                        ErrorInFastLLM("DeepSeekV4Model::ForwardBatch decode cache is not initialized.");
                    }
                    decodeCache.totalLen = startPos + seqlen;
                    UpdateWindowKVCache(kvParts[b], 1, head_dim_full, startPos, window_size,
                                        decodeCache.windowKV);
                }

                const Data *decodeCompressedKVForAttention = &decodeCache.compressedKV;
                if (compressRatio > 0) {
                    Data compressorKV, compressorScore;
                    Split(compressorKVAll, 0, b, b + 1, compressorKV);
                    Split(compressorScoreAll, 0, b, b + 1, compressorScore);
                    int compressedCutoff = decodeCache.totalLen - (decodeCache.totalLen % compressRatio);
                    int targetCompressedBlocks = compressRatio > 0 ? compressedCutoff / compressRatio : 0;
                    bool targetCompressedReady = targetCompressedBlocks > 0 &&
                        decodeCache.compressedBlocks == targetCompressedBlocks &&
                        HasCompressedKVData(decodeCache.compressedKV);

                    const Data *compressorKVForBuild = &decodeCache.compressorKVRaw;
                    const Data *compressorScoreForBuild = &decodeCache.compressorScoreRaw;
                    int compressorRawTokenBaseForBuild = decodeCache.compressorRawTokenBase;
                    if (startPos == 0) {
                        CopyTensorData(decodeCache.compressorKVRaw, compressorKV);
                        CopyTensorData(decodeCache.compressorScoreRaw, compressorScore);
                        decodeCache.compressorRawTokenBase = 0;
                    } else if (!targetCompressedReady) {
                        AppendCompressorRaw(compressorKV, compressorScore, 1, seqlen,
                                            decodeCache.compressorWideDim,
                                            decodeCache.compressorKVRaw,
                                            decodeCache.compressorScoreRaw);
                    }

                    if (targetCompressedReady) {
                        decodeCompressedCount = decodeCache.compressedBlocks;
                        decodeCompressedKVForAttention = &decodeCache.compressedKV;
                    } else {
                        bool builtCompressed = BuildCompressedKVFromRaw(
                            weight, pre + ".attn.compressor", *compressorKVForBuild,
                            *compressorScoreForBuild, 1, compressorRawTokenBaseForBuild,
                            decodeCache.totalLen, compressRatio, head_dim_full,
                            qk_rope_head_dim, layerRopeBase, rope_factor,
                            rope_scaling_beta_fast, rope_scaling_beta_slow,
                            layerOriginalSeqLen, decodeCache.compressedKV, true);
                        if (builtCompressed) {
                            int builtBlocks = GetReusableCompressedBlocks(decodeCache.compressedKV,
                                                                          1, targetCompressedBlocks,
                                                                          head_dim_full);
                            decodeCache.compressedBlocks = builtBlocks;
                            decodeCompressedCount = decodeCache.compressedBlocks;
                            TrimCompressorRawCache(1, decodeCache.totalLen, compressRatio,
                                                   decodeCache.compressorWideDim,
                                                   decodeCache.compressedBlocks,
                                                   decodeCache.compressorKVRaw,
                                                   decodeCache.compressorScoreRaw,
                                                   decodeCache.compressorRawTokenBase);
                            decodeCompressedKVForAttention = &decodeCache.compressedKV;
                        }
                    }
                }

                windowKVPtrs[b] = &decodeCache.windowKV;
                compressedKVPtrs[b] = (Data*)decodeCompressedKVForAttention;
                layerCompressedCounts[b] = decodeCompressedCount;
                cachedDecode[b] = startPos > 0 ? 1 : 0;
                if (cpuIndexerLayer &&
                    startPos + seqlen >= compressRatio) {
                    bool indexerReady = PrepareCpuIndexerTopK(
                        weight, pre + ".attn", indexerAttnInputParts[b],
                        indexerQNormParts[b], 1, seqlen, startPos,
                        compressRatio, index_n_heads, index_head_dim,
                        index_topk, qk_rope_head_dim, layerRopeBase,
                        layerOriginalSeqLen, rope_factor,
                        rope_scaling_beta_fast, rope_scaling_beta_slow,
                        &decodeCache, compressedTopKParts[b]);
                    AssertInFastLLM(
                        indexerReady,
                        "DeepSeekV4Model::ForwardBatch: CPU CSA indexer failed.\n");
                    compressedTopKPtrs[b] = &compressedTopKParts[b];
                }
            }

            Data attnOut4, woAOut, attnOut;
            bool allCachedDecode = batch > 1;
            for (int b = 0; b < batch; b++) {
                allCachedDecode = allCachedDecode && cachedDecode[b] != 0 &&
                    compressedTopKPtrs[b] == nullptr;
            }
            bool usedBatchSparseDecode = false;
            if (allCachedDecode) {
                usedBatchSparseDecode = SparseAttentionDecodeCachedBatch(
                    weight[pre + ".attn.attn_sink"], window_size,
                    qk_rope_head_dim, layerRopeBase,
                    1.0f / std::sqrt((float)head_dim_full),
                    startPositions, layerCompressedCounts, qPartPtrs,
                    windowKVPtrs, compressedKVPtrs, attnOut4,
                    layerOriginalSeqLen, rope_factor,
                    rope_scaling_beta_fast, rope_scaling_beta_slow);
            }

            if (!usedBatchSparseDecode) {
                for (int b = 0; b < batch; b++) {
                    if (startPositions[b] > 0) {
                        SparseAttentionDecodeCachedReference(qParts[b], *windowKVPtrs[b],
                                                             *compressedKVPtrs[b],
                                                             weight[pre + ".attn.attn_sink"],
                                                             window_size, startPositions[b],
                                                             layerCompressedCounts[b],
                                                             qk_rope_head_dim, layerRopeBase,
                                                             1.0f / std::sqrt((float)head_dim_full),
                                                             attnOutParts[b], layerOriginalSeqLen,
                                                             rope_factor, rope_scaling_beta_fast,
                                                             rope_scaling_beta_slow,
                                                             nullptr, 0,
                                                             nullptr, nullptr,
                                                             nullptr, nullptr,
                                                             compressedTopKPtrs[b]);
                    } else {
                        SparseAttentionReference(qParts[b], kvParts[b], weight[pre + ".attn.attn_sink"], window_size,
                                                 qk_rope_head_dim, layerRopeBase, startPositions[b],
                                                 1.0f / std::sqrt((float)head_dim_full), attnOutParts[b],
                                                 compressRatio, layerOriginalSeqLen, rope_factor,
                                                 rope_scaling_beta_fast, rope_scaling_beta_slow,
                                                 0, false, nullptr,
                                                 compressedTopKPtrs[b]);
                    }
                    attnOutPtrs[b] = &attnOutParts[b];
                }
                CatBatch(attnOutPtrs, 0, attnOut4);
            }

            DeepSeekV4WoA(attnOut4, weight[pre + ".attn.wo_a.weight"], o_groups, o_lora_rank, woAOut);
            DeepSeekV4Linear(woAOut, weight[pre + ".attn.wo_b.weight"], Data(), attnOut);
            runHcPost(attnOut, attnMix);

            DeepSeekV4HcPre(*curHiddenStates, weight[pre + ".hc_ffn_fn"],
                            weight[pre + ".hc_ffn_scale"], weight[pre + ".hc_ffn_base"],
                            hc_mult, hc_sinkhorn_iters, hc_eps, rms_norm_eps,
                            ffnMix.y, ffnMix.postData, ffnMix.combData);
            ffnMix.b = bsz;
            ffnMix.s = seqlen;
            ffnMix.hc = hc_mult;
            RMSNormReference(ffnMix.y, weight[pre + ".ffn_norm.weight"], rms_norm_eps, ffnInput, DataType::BFLOAT16);
            std::vector<int> ffnDims = ffnInput.dims;
            ffnInput.Reshape({bsz * seqlen, dim});
            BuildMoERoutingData(weight, pre + ".ffn", ffnInput, tokenIds, num_experts,
                                num_experts_per_tok, scoring_func, routed_scaling_factor,
                                expertIndex, expertScore);
            {
                Data sharedExpertOut;
                bool hasSharedExpertOut = false;
                bool fuseSharedExpert = false;
                std::vector<Data*> moeWeights = weights[layer];
                auto sharedGateupIt = weight.weight.find(pre + ".ffn.shared_experts.gateup.weight");
                auto sharedDownIt = weight.weight.find(pre + ".ffn.shared_experts.w2.weight");
                Data preCopiedMoeInput;
                Data preCopiedExpertIndex;
                Data preCopiedExpertScore;
                Data *routedMoeInput = &ffnInput;
                Data *routedExpertIndex = &expertIndex;
                Data *routedExpertScore = &expertScore;
#ifdef USE_CUDA
                bool canPreCopyNumasMoeInputs =
                    std::getenv(
                        "FASTLLM_DSV4_DISABLE_NUMAS_MOE_PRECOPY") == nullptr &&
                    ffnInput.dims.size() > 0 &&
                    moeWeights.size() > 2 &&
                    moeWeights[2] != nullptr &&
                    !moeWeights[2]->IsTensorParallelSharded() &&
                    GetTensorCudaDevice(ffnInput) >= 0 &&
                    DeepSeekV4DeviceSpecUsesType(
                        this->SelectMoeDeviceForLayer(layer), "numa");
                if (canPreCopyNumasMoeInputs) {
                    bool copied =
                        CopyDeepSeekV4CudaTensorToCpu(
                            ffnInput, preCopiedMoeInput) &&
                        CopyDeepSeekV4CudaTensorToCpu(
                            expertIndex, preCopiedExpertIndex) &&
                        CopyDeepSeekV4CudaTensorToCpu(
                            expertScore, preCopiedExpertScore);
                    AssertInFastLLM(
                        copied,
                        "DeepSeek-V4 failed to copy batched tensor-parallel MoE inputs to NUMA.");
                    routedMoeInput = &preCopiedMoeInput;
                    routedExpertIndex = &preCopiedExpertIndex;
                    routedExpertScore = &preCopiedExpertScore;
                }
#endif
                const bool routedTensorParallel =
                    this->UseTensorParallelRoutedExperts();
                const bool routedExpertParallel =
                    !routedTensorParallel && DeepSeekV4DeviceSpecUsesType(
                        this->SelectMoeDeviceForLayer(layer), "multicuda");
                if (cudaSe && sharedGateupIt != weight.weight.end() && sharedDownIt != weight.weight.end() &&
                    !IsDiskWeight(&sharedGateupIt->second) && !IsDiskWeight(&sharedDownIt->second)) {
                    sharedGateupIt->second.tpLinearType = TP_LINEAR_ROW;
                    sharedGateupIt->second.tpPackType = TP_PACK_GATEUP;
                    sharedDownIt->second.tpLinearType = TP_LINEAR_COLUMN;
                    fuseSharedExpert = routedExpertParallel &&
                                       ffnInput.dims.size() > 0 &&
                                       ffnInput.dims[0] == 1;
                    if (!fuseSharedExpert) {
                        Data ww1, ww3;
                        Data sharedInput;
                        Data *sharedInputPtr = &ffnInput;
                        if (!DeepSeekV4PreferCuda() && ffnInput.dataDevice == DataDevice::CPU &&
                            IsDeepSeekV4QuantizedLinearWeight(sharedGateupIt->second)) {
                            DeepSeekV4QuantizeLinearActivationCpu(ffnInput, sharedInput);
                            sharedInputPtr = &sharedInput;
                        }
                        LinearSwigluBlock(sharedInputPtr, &sharedGateupIt->second, GetEmptyData(), &ww3, &ww1);
                        DeepSeekV4Linear(ww1, sharedDownIt->second, *GetEmptyData(), sharedExpertOut);
                        moeWeights[0] = moeWeights[1] = nullptr;
                        hasSharedExpertOut = true;
                    }
                }
                this->ApplyMoeDeviceMapForLayer(layer);
                {
                    DataType effectiveMoeAtype = ffnInput.dataType;
                    Data moeQuantizedInput;
                    Data *moeInput = routedMoeInput;
                    if (!DeepSeekV4PreferCuda() && moeInput->dataDevice == DataDevice::CPU &&
                        moeWeights.size() > 2 && moeWeights[2] != nullptr &&
                        IsDeepSeekV4QuantizedLinearWeight(*moeWeights[2])) {
#ifdef USE_CUDA
                        int moeInputCudaDevice = GetTensorCudaDevice(*moeInput);
#endif
                        DeepSeekV4QuantizeLinearActivationCpu(*moeInput, moeQuantizedInput);
#ifdef USE_CUDA
                        if (moeInputCudaDevice >= 0 &&
                            moeQuantizedInput.dims.size() > 0 &&
                            moeQuantizedInput.dims[0] >= 32 &&
                            DeepSeekV4NumasGpuPrefillEnabled()) {
                            AssertInFastLLM(
                                AddDeepSeekV4QuantizedCudaReplica(
                                    moeQuantizedInput, moeInputCudaDevice),
                                "DeepSeek-V4 failed to stage its quantized batched NUMA MoE activation on CUDA.");
                        }
#endif
                        moeInput = &moeQuantizedInput;
                    }
                    MergeMOEBlock(moeInput, routedExpertIndex, routedExpertScore,
                                  &moeWeights, &biass[layer],
                                  &w1, &w2, &w3, &tempInput, &tempOutput,
                                  1.0f, &ffnOut, layer,
                                  ffnInput.dataType, effectiveMoeAtype,
                                  &moeInputTemp, &moeOutputTemp,
                                  MoeGateSwiglu, routedExpertParallel,
                                  swiglu_limit, true);
                }
                ApplyDeviceMap(this->deviceMap, layer + 1, block_cnt);
                if (hasSharedExpertOut) {
                    ffnOut.ToDevice(sharedExpertOut.dataDevice);
                    AddTo(ffnOut, sharedExpertOut);
                }
            }
            ffnOut.Reshape(ffnDims);
#ifdef USE_CUDA
            if (DeepSeekV4PreferCuda() && ffnOut.dataDevice == DataDevice::CPU && ffnOut.cpuData != nullptr) {
                ffnOut.ToDevice(DataDevice::CUDA);
            }
#endif
            runHcPost(ffnOut, ffnMix);
        }

        Data headInput;
        HcHeadReference(*curHiddenStates, weight["hc_head_fn"], weight["hc_head_scale"], weight["hc_head_base"],
                        hc_mult, hc_eps, rms_norm_eps, headInput);

        std::vector<int> ret;
        std::vector<int> samplingSeqLens(batch, 1);
        std::vector<GenerationConfig> samplingGenerationConfigs = generationConfigs;
        for (int b = 0; b < batch; b++) {
            PrepareDeepSeekV4SamplingConfig(samplingGenerationConfigs[b]);
        }
        Data officialCpuSamplingLogits;
        Data *samplingPrecomputedLogits = nullptr;
        if (DeepSeekV4HeadLogitsCpu(headInput, weight["norm.weight"],
                                    weight["head.weight"], rms_norm_eps,
                                    officialCpuSamplingLogits)) {
            samplingPrecomputedLogits = &officialCpuSamplingLogits;
        }
        LLMSamplingBlock(this, &headInput, &weight["norm.weight"], &weight["head.weight"],
                         rms_norm_eps, batch, true, samplingSeqLens, pastKeyValues,
                         samplingGenerationConfigs, lastTokens, retLogits, ret,
                         samplingPrecomputedLogits);

        for (int b = 0; b < batch; b++) {
            int finalTotalLen = startPositions[b] + 1;
            UpdateDebugPastKeyValues(pastKeyValues, b, 1, finalTotalLen, block_cnt);
            if (this->saveHistoryChat && !DeepSeekV4PrefixCacheDisabled() &&
                finalTotalLen % 256 == 0 &&
                (int)requestStates[b]->historyTokens.size() >= finalTotalLen) {
#ifdef USE_CUDA
                SynchronizeDeepSeekV4TensorParallelDevices(this->deviceMap);
#endif
                this->RecordHistorySnapshot(requestStates[b]->historyTokens, finalTotalLen,
                                            requestStates[b]->decodeLayerCaches);
            } else if (this->saveHistoryChat && !DeepSeekV4PrefixCacheDisabled() &&
                       finalTotalLen % 256 == 0 && DeepSeekV4PrefixCacheDebugEnabled()) {
                printf("[fastllm-dsv4-prefix-cache] skip boundary record: final_len=%d history_tokens=%d\n",
                       finalTotalLen, (int)requestStates[b]->historyTokens.size());
                fflush(stdout);
            }
        }
        return ret;
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
