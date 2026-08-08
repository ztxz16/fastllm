//
// Created by huangyuyang on 10/15/24.
//

#include <sys/mman.h>
#include <fcntl.h>

#include "devices/numas/numasdevice.h"
#include "devices/cpu/cpudevice.h"
#include "devices/cpu/alivethreadpool.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <array>
#include <algorithm>
#include <atomic>
#include <thread>
#include <mutex>
#include <memory>
#include <chrono>
#if defined(__linux__) && defined(__GLIBC__)
#include <malloc.h>
#endif

#include <cfloat>
#include <climits>
#include <cmath>
#include <map>
#include <set>
#include <numeric>

#ifdef __aarch64__
#include <arm_neon.h>
#include "armMath.h"
#endif

#include "utils.h"
#include "computeutils.h"
#include "numas.h"
#include "gguf.h"

#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/cudadevice.h"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif

namespace fastllm {
    extern CPUInstructInfo *GetCPUInstructInfo();
    extern bool LinearBFloat16Float16Decode_AVX512F_Kernel(
        uint16_t *inputData, uint16_t *weightData,
        float *biasData, float *outputData,
        int n, int m, int k, int st, int end);

    static double NumasProfileNowMs() {
        using Clock = std::chrono::steady_clock;
        return std::chrono::duration<double, std::milli>(Clock::now().time_since_epoch()).count();
    }

    static void CopyTensorToCPU(const Data &src, Data &dst, const char *name) {
        dst.dataType = src.dataType;
        dst.Resize(src.dims);
        dst.Allocate(false);

        if (src.dataDevice == DataDevice::CPU && src.cpuData != nullptr) {
            memcpy(dst.cpuData, src.cpuData, src.GetBytes());
            return;
        }

#ifdef USE_CUDA
        const Data *cudaSrc = &src;
        int deviceId = src.dataDeviceIds.empty() ? FastllmCudaGetDevice() : src.dataDeviceIds[0];
        if (src.multiDeviceData && !src.multiDeviceDatas.empty()) {
            auto it = src.multiDeviceDatas.begin();
            if (it->second != nullptr) {
                deviceId = it->first;
                cudaSrc = it->second;
            }
        }
        if (cudaSrc != nullptr && cudaSrc->cudaData != nullptr) {
            FastllmCudaSetDevice(deviceId);
            FastllmCudaCopyFromDeviceToHost(dst.cpuData, cudaSrc->cudaData, dst.GetBytes());
            return;
        }
#endif

        ErrorInFastLLM(std::string("NumasMergeMOE: tensor ") + name + " has no readable CPU/CUDA data.\n");
    }

    static Data *GetCpuTensor(Data &src, Data &scratch, const char *name) {
        if (src.dataDevice == DataDevice::CPU && src.cpuData != nullptr) {
            return &src;
        }
        CopyTensorToCPU(src, scratch, name);
        return &scratch;
    }

#ifdef USE_CUDA
    struct NumasMoeCudaInputReplica {
        int deviceId = -1;
        void *cudaData = nullptr;
    };

    static std::vector<NumasMoeCudaInputReplica>
    GetNumasMoeCudaInputReplicas(const Data &input) {
        std::vector<NumasMoeCudaInputReplica> replicas;
        if (input.multiDeviceData && input.IsTensorParallelReplicated()) {
            for (const auto &it : input.multiDeviceDatas) {
                const Data *local = it.second;
                if (local == nullptr || local->cudaData == nullptr ||
                    local->dataType != input.dataType ||
                    local->dims != input.dims ||
                    GetPointerDeviceId(local->cudaData) != it.first) {
                    continue;
                }
                replicas.push_back({it.first, local->cudaData});
            }
        }
        if (replicas.empty() && input.cudaData != nullptr) {
            int deviceId = GetPointerDeviceId(input.cudaData);
            if (deviceId < 0 && !input.dataDeviceIds.empty()) {
                deviceId = input.dataDeviceIds[0];
            }
            if (deviceId >= 0) {
                replicas.push_back({deviceId, input.cudaData});
            }
        }
        std::sort(
            replicas.begin(), replicas.end(),
            [](const NumasMoeCudaInputReplica &a,
               const NumasMoeCudaInputReplica &b) {
                return a.deviceId < b.deviceId;
            });
        return replicas;
    }
#endif

    class MoeEnvConfig {
    public:
        static MoeEnvConfig& GetInstance() {
            static MoeEnvConfig instance;
            return instance;
        }

        int GetExpertLimit() const { return expertLimit; }
        bool HasExpertLimitOverride() const { return hasExpertLimitOverride; }
        bool GetGpuPrefill() const { return gpuPrefill; }
        bool GetPinnedWeight() const { return pinnedWeight; }

    private:
        MoeEnvConfig() {
            expertLimit = 128;
            hasExpertLimitOverride = false;
            gpuPrefill = true;
            pinnedWeight = true;

            const char *expertLimitEnv = std::getenv("FT_EXPERT_LIMIT");
            if (expertLimitEnv != nullptr) {
                expertLimit = std::atoi(expertLimitEnv);
                hasExpertLimitOverride = true;
            }

            const char *gpuPrefillEnv = std::getenv("FT_GPU_PREFILL");
            if (gpuPrefillEnv != nullptr) {
                std::string val(gpuPrefillEnv);
                std::transform(val.begin(), val.end(), val.begin(), ::tolower);
                if (val == "0" || val == "false" || val == "off") {
                    gpuPrefill = false;
                }
            }

            const char *pinnedWeightEnv = std::getenv("FT_PINNED_WEIGHT");
            if (pinnedWeightEnv != nullptr) {
                std::string val(pinnedWeightEnv);
                std::transform(val.begin(), val.end(), val.begin(), ::tolower);
                if (val == "0" || val == "false" || val == "off") {
                    pinnedWeight = false;
                }
            }

            if (gpuPrefill) {
                printf("Activate GPU Prefill\n");
            } else {
                printf("Disable GPU Prefill\n");
            }
            if (pinnedWeight) {
                printf("Activate Pinned Weight\n");
            }
        }

        MoeEnvConfig(const MoeEnvConfig&) = delete;
        MoeEnvConfig& operator=(const MoeEnvConfig&) = delete;

        int expertLimit;
        bool hasExpertLimitOverride;
        bool gpuPrefill;
        bool pinnedWeight;
    };

    static MachineNumaInfo machineNumaInfo;
    NumaConfig *fastllmNumaConfig = nullptr;
    std::mutex numaConfigLocker;

    NumaConfig *GetNumaConfig() {
        numaConfigLocker.lock();
        auto *pool = GetAlivePool();
        if (fastllmNumaConfig == nullptr) {
            // AliveThreadPool keeps previously-created worker objects when
            // SetThreads() shrinks the active range.  Using vector::size()
            // here silently ignored a later `-t N` and reactivated every
            // old worker for NUMA.  NumaConfig uses worker ids from zero, so
            // the active interval end is the requested pool size.
            int activeThreads = pool->curActivateThreadInterval.second;
            fastllmNumaConfig = new NumaConfig(
                activeThreads, pool, &machineNumaInfo);
        }
        numaConfigLocker.unlock();
        
        return fastllmNumaConfig;
    }

    NumasDevice::NumasDevice() {
        this->deviceType = "numa";
        this->ops["Linear"] = (BaseOperator *) (new NumasLinearOp());
        this->ops["MergeMOE"] = (BaseOperator*)(new NumasMergeMOE());
        this->ops["DeepSeekV4WoA"] =
            (BaseOperator*)(new NumasDeepSeekV4WoAOp());
        this->ops["FusedMOE"] = (BaseOperator*)(new NumasFusedMOE());
        this->ops["KimiK3RoutedExperts"] =
            (BaseOperator*)(new NumasKimiK3RoutedExpertsOp());

        /*this->ops["CatDirect"] = (BaseOperator *) (new NumaCatDirectOp());
        this->ops["Attention"] = (BaseOperator *) (new NumaAttention());

        this->ops["AttentionBatch"] = (BaseOperator *) (new NumaAttentionBatchOp());
        this->ops["CatDirectBatch"] = (BaseOperator *) (new NumaCatDirectBatchOp());*/
    }

    bool NumasDevice::Malloc(void **ret, size_t size) {
        *ret = (void*)new uint8_t [size];
        return true;
    }

    bool NumasDevice::Free(void *ret) {
        delete[] (uint8_t *)ret;
        return true;
    }

    bool NumasDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
        return true;
    }
    
    bool NumasDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
        return true;
    }

    extern void Float16ToFloat32(uint16_t *float16, float *float32, int len);
    extern void Float32ToFloat16(float *float32, uint16_t *float16, int len);
    extern void Float32ToBFloat16(float *float32, uint16_t *bfloat16, int len);

    static void BFloat16ToFloat32(uint16_t *bfloat16, float *float32, int len) {
        for (int i = 0; i < len; i++) {
            uint32_t x = (uint32_t)bfloat16[i] << 16;
            float32[i] = *(float*)&x;
        }
    }

    static bool IsDeepSeekV4QuantizedWeight(const Data &weight) {
        return weight.dataType == DataType::FP8_E4M3 ||
               weight.dataType == DataType::FP8_E4M3_BLOCK_128 ||
               weight.dataType == DataType::FP8_E4M3_PERCHANNEL ||
               weight.dataType == DataType::NVFP4 ||
               weight.dataType == DataType::NVFP4_BLOCK_16 ||
               weight.dataType == DataType::NVFP4_BLOCK_16_E8M0 ||
               weight.dataType == DataType::NVFP4_BLOCK_32_E8M0;
    }

    static const std::array<float, 65536> &
    GetDeepSeekV4BFloat16SiluLookup() {
        static const std::array<float, 65536> lookup = []() {
            std::array<float, 65536> values {};
            for (uint32_t bits = 0; bits < values.size(); bits++) {
                float gate = BFloat16BitsToFloat32((uint16_t)bits);
                values[bits] =
                    gate / (1.0f + std::exp(-gate));
            }
            return values;
        }();
        return lookup;
    }

    static void QuantizeDeepSeekV4FP8ActivationBFloat16(
        uint16_t *values, int len
    ) {
        for (int start = 0; start < len; start += 128) {
            int end = std::min(start + 128, len);
            float amax = 1e-4f;
            for (int i = start; i < end; i++) {
                amax = std::max(
                    amax,
                    std::fabs(BFloat16BitsToFloat32(values[i])));
            }
            float normalized = amax / 448.0f;
            uint32_t bits;
            memcpy(&bits, &normalized, sizeof(bits));
            int exponent = (int)((bits >> 23) & 0xFF) - 127 +
                           ((bits & ((1u << 23) - 1)) != 0);
            const uint16_t *lookupRow =
                GetFP8E4M3BFloat16QuantizeLookupRow(exponent);
            if (lookupRow != nullptr) {
                for (int i = start; i < end; i++) {
                    values[i] = lookupRow[values[i]];
                }
            } else {
                static const FP8E4M3ToFP32Manager fp8;
                float scale = std::ldexp(1.0f, exponent);
                for (int i = start; i < end; i++) {
                    float value = BFloat16BitsToFloat32(values[i]);
                    float q = std::max(
                        -448.0f, std::min(448.0f, value / scale));
                    values[i] = Float32ToBFloat16RNEBits(
                        fp8.quantizeDequantize(q) * scale);
                }
            }
        }
    }

    // NUMA gate-up weights are cross-reordered, so each pair is [gate, up].
    // Recreate inference/model.py::Expert.forward including all observable
    // BF16 boundaries before preparing the input for the down projection.
    static void PrepareDeepSeekV4DownInput(
        const float *gateUp, float *swiglu, uint8_t *downInput,
        DataType downInputType, const Data &downWeight,
        int interDim, bool routed, float routeWeight, float swigluLimit
    ) {
        for (int i = 0; i < interDim; i++) {
            float gate = RoundFloat32ToBFloat16RNE(gateUp[i * 2]);
            float up = RoundFloat32ToBFloat16RNE(gateUp[i * 2 + 1]);
            if (routed && swigluLimit > 0.0f) {
                gate = std::min(gate, swigluLimit);
                up = std::max(-swigluLimit, std::min(up, swigluLimit));
            }
            float h = (gate / (1.0f + std::exp(-gate))) * up;
            swiglu[i] = RoundFloat32ToBFloat16RNE(routeWeight * h);
        }
        if (IsDeepSeekV4QuantizedWeight(downWeight)) {
            QuantizeDequantizeFP8E4M3Block128(swiglu, interDim);
        }

        if (downInputType == DataType::BFLOAT16) {
            uint16_t *dst = (uint16_t*)downInput;
            for (int i = 0; i < interDim; i++) {
                dst[i] = Float32ToBFloat16RNEBits(swiglu[i]);
            }
        } else if (downInputType == DataType::FLOAT32) {
            memcpy(downInput, swiglu, (size_t)interDim * sizeof(float));
        } else if (downInputType == DataType::FLOAT16) {
            Float32ToFloat16(swiglu, (uint16_t*)downInput, interDim);
        } else {
            ErrorInFastLLM("DeepSeek-V4 NUMA MoE requires a floating-point down activation type.\n");
        }
    }

    struct MultiThreadDeepSeekV4NumasDownPrepareOp : MultiThreadBaseOp {
        const float *gateUpData;
        float *swigluData;
        uint8_t *downInputData;
        DataType downInputType;
        int st, end;
        bool routed, quantize, useBFloat16SiluLookup;
        bool useDirectBFloat16Prepare;
        float routeWeight, swigluLimit;

        MultiThreadDeepSeekV4NumasDownPrepareOp(
            const float *gateUpData, float *swigluData,
            uint8_t *downInputData, DataType downInputType,
            int st, int end, bool routed, float routeWeight,
            float swigluLimit, bool quantize,
            bool useBFloat16SiluLookup, bool useDirectBFloat16Prepare
        ) : gateUpData(gateUpData), swigluData(swigluData),
            downInputData(downInputData), downInputType(downInputType),
            st(st), end(end), routed(routed), quantize(quantize),
            useBFloat16SiluLookup(useBFloat16SiluLookup),
            useDirectBFloat16Prepare(useDirectBFloat16Prepare),
            routeWeight(routeWeight), swigluLimit(swigluLimit) {}

        void Run() override {
            const std::array<float, 65536> *siluLookup =
                useBFloat16SiluLookup ?
                    &GetDeepSeekV4BFloat16SiluLookup() : nullptr;
            uint16_t *directBFloat16Output =
                useDirectBFloat16Prepare &&
                        downInputType == DataType::BFLOAT16 ?
                    (uint16_t*)downInputData : nullptr;
            for (int i = st; i < end; i++) {
                uint16_t gateBits =
                    Float32ToBFloat16RNEBits(gateUpData[i * 2]);
                float gate = BFloat16BitsToFloat32(gateBits);
                float up =
                    RoundFloat32ToBFloat16RNE(gateUpData[i * 2 + 1]);
                bool gateIsBFloat16 = true;
                if (routed && swigluLimit > 0.0f) {
                    if (gate > swigluLimit) {
                        gate = swigluLimit;
                        gateBits = Float32ToBFloat16RNEBits(gate);
                        gateIsBFloat16 =
                            BFloat16BitsToFloat32(gateBits) == gate;
                    }
                    up = std::max(
                        -swigluLimit, std::min(up, swigluLimit));
                }
                float silu = siluLookup != nullptr && gateIsBFloat16 ?
                    (*siluLookup)[gateBits] :
                    gate / (1.0f + std::exp(-gate));
                float h = silu * up;
                uint16_t outputBits =
                    Float32ToBFloat16RNEBits(routeWeight * h);
                if (directBFloat16Output != nullptr) {
                    directBFloat16Output[i] = outputBits;
                } else {
                    swigluData[i] =
                        BFloat16BitsToFloat32(outputBits);
                }
            }
            if (directBFloat16Output != nullptr) {
                if (quantize) {
                    QuantizeDeepSeekV4FP8ActivationBFloat16(
                        directBFloat16Output + st, end - st);
                }
                return;
            }
            if (quantize) {
                // Task boundaries follow the official FP8 activation blocks,
                // so parallel quantization keeps the exact per-block scale.
                QuantizeDequantizeFP8E4M3Block128(
                    swigluData + st, end - st);
            }

            if (downInputType == DataType::BFLOAT16) {
                Float32ToBFloat16(
                    swigluData + st,
                    (uint16_t*)downInputData + st, end - st);
            } else if (downInputType == DataType::FLOAT32) {
                memcpy(
                    (float*)downInputData + st, swigluData + st,
                    (size_t)(end - st) * sizeof(float));
            } else if (downInputType == DataType::FLOAT16) {
                Float32ToFloat16(
                    swigluData + st,
                    (uint16_t*)downInputData + st, end - st);
            } else {
                ErrorInFastLLM(
                    "DeepSeek-V4 NUMA MoE requires a floating-point "
                    "down activation type.\n");
            }
        }
    };

    struct MultiThreadDeepSeekV4NumasRoundOutputOp : MultiThreadBaseOp {
        float *data;
        size_t st, end;

        MultiThreadDeepSeekV4NumasRoundOutputOp(
            float *data, size_t st, size_t end
        ) : data(data), st(st), end(end) {}

        void Run() override {
            for (size_t i = st; i < end; i++) {
                data[i] = RoundFloat32ToBFloat16RNE(data[i]);
            }
        }
    };

    struct MultiThreadDeepSeekV4NumasStoreOutputOp : MultiThreadBaseOp {
        const float *input;
        uint16_t *output;
        size_t st, end;

        MultiThreadDeepSeekV4NumasStoreOutputOp(
            const float *input, uint16_t *output, size_t st, size_t end
        ) : input(input), output(output), st(st), end(end) {}

        void Run() override {
            for (size_t i = st; i < end; i++) {
                output[i] = Float32ToBFloat16RNEBits(input[i]);
            }
        }
    };

    struct MultiThreadDeepSeekV4NumasReduceOp : MultiThreadBaseOp {
        float *downOutput;
        float *output;
        int expertCount, outputDim, st, end;

        MultiThreadDeepSeekV4NumasReduceOp(
            float *downOutput, float *output,
            int expertCount, int outputDim, int st, int end
        ) : downOutput(downOutput), output(output),
            expertCount(expertCount), outputDim(outputDim),
            st(st), end(end) {}

        void Run() override {
            if (expertCount <= 0) {
                return;
            }

            float *first = downOutput;
            for (int d = st; d < end; d++) {
                first[d] = RoundFloat32ToBFloat16RNE(first[d]);
                output[d] = first[d];
            }
            for (int expert = 1; expert < expertCount; expert++) {
                float *curOutput =
                    downOutput + (size_t)expert * outputDim;
                for (int d = st; d < end; d++) {
                    curOutput[d] =
                        RoundFloat32ToBFloat16RNE(curOutput[d]);
                }
                int d = st;
#ifdef __AVX2__
                for (; d + 7 < end; d += 8) {
                    __m256 cur = _mm256_loadu_ps(curOutput + d);
                    __m256 sum = _mm256_loadu_ps(output + d);
                    _mm256_storeu_ps(
                        output + d, _mm256_add_ps(sum, cur));
                }
#endif
                for (; d < end; d++) {
                    output[d] += curOutput[d];
                }
            }
        }
    };

    struct MultiThreadDeepSeekV4NumasTaskQueueOp : MultiThreadBaseOp {
        std::vector<MultiThreadBaseOp*> *tasks;
        std::atomic<int> *next;
        int firstTask;

        MultiThreadDeepSeekV4NumasTaskQueueOp(
            std::vector<MultiThreadBaseOp*> *tasks,
            std::atomic<int> *next, int firstTask
        ) : tasks(tasks), next(next), firstTask(firstTask) {}

        void Run() override {
            const int taskCount = (int)tasks->size();
            if (firstTask < taskCount) {
                (*tasks)[firstTask]->Run();
            }
            while (true) {
                int taskId =
                    next->fetch_add(1, std::memory_order_relaxed);
                if (taskId >= taskCount) {
                    break;
                }
                (*tasks)[taskId]->Run();
            }
        }
    };

    struct DeepSeekV4NumasSchedulerWorkspace {
        std::vector<std::unique_ptr<std::atomic<int>>> next;
        std::vector<MultiThreadDeepSeekV4NumasTaskQueueOp> workers;
        std::vector<int> workerThreadIds;
    };

    static void ScheduleDeepSeekV4NumasMoeTasks(
        std::vector<std::vector<MultiThreadBaseOp*>> &tasks,
        bool deleteTasks = true
    ) {
        auto *pool = GetAlivePool();
        auto *numaConfig = GetNumaConfig();
        DeepSeekV4NumasSchedulerWorkspace localWorkspace;
        static thread_local DeepSeekV4NumasSchedulerWorkspace workspace;
        bool reuseWorkspace =
            std::getenv(
                "FASTLLM_DSV4_DISABLE_NUMAS_MOE_TASK_CACHE") ==
            nullptr;
        auto &activeWorkspace =
            reuseWorkspace ? workspace : localWorkspace;
        activeWorkspace.next.resize(numaConfig->numaCnt);
        auto &next = activeWorkspace.next;
        auto &workers = activeWorkspace.workers;
        auto &workerThreadIds = activeWorkspace.workerThreadIds;
        workers.clear();
        workerThreadIds.clear();
        workers.reserve(numaConfig->threads);
        workerThreadIds.reserve(numaConfig->threads);

        for (int nid = 0; nid < numaConfig->numaCnt; nid++) {
            if (nid >= (int)tasks.size() || tasks[nid].empty()) {
                continue;
            }
            const auto &nodeThreads = numaConfig->numaToCpuDict[nid];
            int workerCount = std::min(
                (int)nodeThreads.size(), (int)tasks[nid].size());
            if (!next[nid]) {
                next[nid] =
                    std::make_unique<std::atomic<int>>(workerCount);
            } else {
                next[nid]->store(
                    workerCount, std::memory_order_relaxed);
            }
            for (int i = 0; i < workerCount; i++) {
                workers.emplace_back(
                    &tasks[nid], next[nid].get(), i);
                workerThreadIds.push_back(nodeThreads[i].first);
            }
        }

        for (int i = 0; i < (int)workers.size(); i++) {
            pool->PushOp(workerThreadIds[i], &workers[i]);
        }
        for (int i = 0; i < (int)workers.size(); i++) {
            pool->Wait(workerThreadIds[i]);
        }

        if (deleteTasks) {
            for (auto &nodeTasks : tasks) {
                for (auto *task : nodeTasks) {
                    delete task;
                }
            }
        }
    }

    struct DeepSeekV4NumasGemmQueueContext {
        const std::vector<std::pair<int, float>> *experts;
        const std::vector<int> *expertOrder;
        Data **weights;
        uint8_t *inputData;
        size_t inputStrideBytes;
        DataType inputDataType;
        float *outputData;
        int weightOffset;
        int nid;
        int m, k, kPer;
        int rowsPerTask;
        int chunksPerExpert;

        DeepSeekV4NumasGemmQueueContext(
            const std::vector<std::pair<int, float>> *experts,
            const std::vector<int> *expertOrder,
            Data **weights,
            uint8_t *inputData, size_t inputStrideBytes,
            DataType inputDataType, float *outputData,
            int weightOffset, int nid,
            int m, int k, int kPer, int rowsPerTask
        ) : experts(experts), expertOrder(expertOrder),
            weights(weights), inputData(inputData),
            inputStrideBytes(inputStrideBytes),
            inputDataType(inputDataType), outputData(outputData),
            weightOffset(weightOffset), nid(nid),
            m(m), k(k), kPer(kPer), rowsPerTask(rowsPerTask),
            chunksPerExpert(
                (kPer + rowsPerTask - 1) / rowsPerTask) {}

        int TaskCount() const {
            return (int)experts->size() * chunksPerExpert;
        }
    };

    struct MultiThreadDeepSeekV4NumasGemmQueueOp :
            MultiThreadBaseOp {
        DeepSeekV4NumasGemmQueueContext *context;
        std::atomic<int> *next;
        int firstTask;

        MultiThreadDeepSeekV4NumasGemmQueueOp(
            DeepSeekV4NumasGemmQueueContext *context,
            std::atomic<int> *next, int firstTask
        ) : context(context), next(next), firstTask(firstTask) {}

        void RunTask(int taskId) {
            int orderIndex = taskId / context->chunksPerExpert;
            int chunk = taskId -
                orderIndex * context->chunksPerExpert;
            int expertIdx =
                (*context->expertOrder)[orderIndex];
            int expert = (*context->experts)[expertIdx].first;
            Data *weight =
                context->weights[expert * 2 + context->weightOffset];
            int st = chunk * context->rowsPerTask;
            int end = std::min(
                st + context->rowsPerTask, context->kPer);
            uint8_t *input =
                context->inputData +
                (size_t)expertIdx * context->inputStrideBytes;
            uint8_t *output = (uint8_t*)(
                context->outputData +
                (size_t)expertIdx * context->k +
                (size_t)context->nid * context->kPer);
            FastllmGemm(
                1, context->m, context->k,
                input,
                GetDataBytes(
                    context->inputDataType, 1, context->m),
                weight->numasData[context->nid],
                GetDataBytes(
                    weight->GetDataType(), 1, context->m),
                output,
                GetDataBytes(DataType::FLOAT32, 1, context->k),
                st, end,
                context->inputDataType, weight->GetDataType(),
                DataType::FLOAT32);
        }

        void Run() override {
            int taskCount = context->TaskCount();
            if (firstTask < taskCount) {
                RunTask(firstTask);
            }
            while (true) {
                int taskId =
                    next->fetch_add(1, std::memory_order_relaxed);
                if (taskId >= taskCount) {
                    break;
                }
                RunTask(taskId);
            }
        }
    };

    struct DeepSeekV4NumasGemmSchedulerWorkspace {
        std::vector<std::unique_ptr<std::atomic<int>>> next;
        std::vector<MultiThreadDeepSeekV4NumasGemmQueueOp> workers;
        std::vector<int> workerThreadIds;
    };

    static void ScheduleDeepSeekV4NumasGemmQueue(
        std::vector<DeepSeekV4NumasGemmQueueContext> &contexts
    ) {
        auto *pool = GetAlivePool();
        auto *numaConfig = GetNumaConfig();
        DeepSeekV4NumasGemmSchedulerWorkspace localWorkspace;
        static thread_local DeepSeekV4NumasGemmSchedulerWorkspace
            workspace;
        bool reuseWorkspace =
            std::getenv(
                "FASTLLM_DSV4_DISABLE_NUMAS_MOE_TASK_CACHE") ==
            nullptr;
        auto &activeWorkspace =
            reuseWorkspace ? workspace : localWorkspace;
        activeWorkspace.next.resize(numaConfig->numaCnt);
        auto &workers = activeWorkspace.workers;
        auto &workerThreadIds = activeWorkspace.workerThreadIds;
        workers.clear();
        workerThreadIds.clear();
        workers.reserve(numaConfig->threads);
        workerThreadIds.reserve(numaConfig->threads);

        for (int nid = 0; nid < numaConfig->numaCnt; nid++) {
            auto &context = contexts[nid];
            const auto &nodeThreads =
                numaConfig->numaToCpuDict[nid];
            int workerCount = std::min(
                (int)nodeThreads.size(), context.TaskCount());
            if (!activeWorkspace.next[nid]) {
                activeWorkspace.next[nid] =
                    std::make_unique<std::atomic<int>>(workerCount);
            } else {
                activeWorkspace.next[nid]->store(
                    workerCount, std::memory_order_relaxed);
            }
            for (int i = 0; i < workerCount; i++) {
                workers.emplace_back(
                    &context, activeWorkspace.next[nid].get(), i);
                workerThreadIds.push_back(nodeThreads[i].first);
            }
        }

        for (int i = 0; i < (int)workers.size(); i++) {
            pool->PushOp(workerThreadIds[i], &workers[i]);
        }
        for (int i = 0; i < (int)workers.size(); i++) {
            pool->Wait(workerThreadIds[i]);
        }
    }

    void Fp8ToFastllmFP8_E4M3_BLOCK128(int experts, int k, int m, uint8_t *fp8, float *scales, int blockK, int blockM, std::vector <uint8_t> &fp8Packed) {
        int ks = (k - 1) / blockK + 1;
        int ms = (m - 1) / blockM + 1;
        
        // 计算每行需要的总字节数
        // 每128个fp8需要1个float的scale，所以需要 (m + 127) / 128 个scale
        int numScalesPerRow = (m + 127) / 128;
        int rowSize = m + numScalesPerRow * sizeof(float);
        fp8Packed.resize((size_t)experts * k * rowSize);
        for (size_t i = 0; i < experts; i++) {
            for (size_t j = 0; j < k; j++) {
                size_t rowIdx = i * k + j;
                size_t packedOffset = rowIdx * rowSize;
                
                // 按照每128个fp8后接一个scale的格式打包
                size_t currentPos = packedOffset;
                
                for (int blockIdx = 0; blockIdx < numScalesPerRow; blockIdx++) {
                    size_t blockStart = blockIdx * 128;
                    size_t blockEnd = std::min(blockStart + 128, (size_t)m);
                    size_t blockSize = blockEnd - blockStart;
                    
                    // 复制当前block的fp8数据
                    for (size_t l = blockStart; l < blockEnd; l++) {
                        size_t srcIdx = i * k * m + j * m + l;
                        fp8Packed[currentPos++] = fp8[srcIdx];
                    }
                    
                    // 在这个block后面添加对应的scale
                    // scale的索引计算：需要根据当前block在整个矩阵中的位置
                    size_t scaleRow = j / blockK;  // 当前行属于哪个scale行块
                    size_t scaleCol = blockStart / blockM;  // 当前block属于哪个scale列块
                    size_t scaleIdx = i * ks * ms + scaleRow * ms + scaleCol;
                    
                    float* scalePtr = (float*)(&fp8Packed[currentPos]);
                    *scalePtr = scales[scaleIdx];
                    currentPos += sizeof(float);
                }
            }
        }
    }

    void Fp8PerchannelToFastllmFP8_E4M3_PERCHANNEL(int experts, int k, int m, uint8_t *fp8, float *scales, int blockK, int blockM, std::vector <uint8_t> &fp8Packed) {
        int ks = (k - 1) / blockK + 1;
        int ms = (m - 1) / blockM + 1;
        int rowSize = m + sizeof(float);
        fp8Packed.resize((size_t)experts * k * rowSize);
        for (size_t i = 0; i < experts; i++) {
            for (size_t j = 0; j < k; j++) {
                size_t rowIdx = i * k + j;
                size_t packedOffset = rowIdx * rowSize;
                size_t currentPos = packedOffset;
                    
                // 复制当前block的fp8数据
                for (size_t l = 0; l < m; l++) {
                    size_t srcIdx = i * k * m + j * m + l;
                    fp8Packed[currentPos++] = fp8[srcIdx];
                }
                    
                // 在这个block后面添加对应的scale
                // scale的索引计算：需要根据当前block在整个矩阵中的位置
                size_t scaleRow = j / blockK;  // 当前行属于哪个scale行块
                size_t scaleCol = 0;
                size_t scaleIdx = i * ks * ms + scaleRow * ms + scaleCol;
                    
                float* scalePtr = (float*)(&fp8Packed[currentPos]);
                *scalePtr = scales[scaleIdx];
                currentPos += sizeof(float);
            }
        }
    }

    void Nvfp4ToFastllmNVFP4_BLOCK16(int experts, int k, int m, uint8_t *nvfp4, float *scales, uint8_t *scaleBytes,
                                     int blockK, int blockM, std::vector<uint8_t> &nvfp4Packed) {
        const int packBlockM = 16;
        const int fp4BytesPerBlock = packBlockM / 2;
        int scaleKs = (k - 1) / blockK + 1;
        int scaleMs = (m - 1) / blockM + 1;
        int packedBlocks = (m - 1) / packBlockM + 1;
        size_t rawBytesPerRow = GetDataBytes(DataType::NVFP4, 1, m);
        size_t packedBytesPerRow = GetDataBytes(DataType::NVFP4_BLOCK_16, 1, m);
        nvfp4Packed.assign((size_t)experts * k * packedBytesPerRow, 0);

        for (int e = 0; e < experts; e++) {
            for (int row = 0; row < k; row++) {
                uint8_t *dst = nvfp4Packed.data() + ((size_t)e * k + row) * packedBytesPerRow;
                uint8_t *src = nvfp4 + ((size_t)e * k + row) * rawBytesPerRow;
                for (int block = 0; block < packedBlocks; block++) {
                    int blockStart = block * packBlockM;
                    int blockElems = std::min(packBlockM, m - blockStart);
                    int blockBytes = blockElems / 2;
                    memcpy(dst, src + blockStart / 2, blockBytes);
                    dst += fp4BytesPerBlock;

                    int scaleCol = blockStart / blockM;
                    size_t scaleIdx = (size_t)e * scaleKs * scaleMs + (row / blockK) * scaleMs + scaleCol;
                    float scale = scales != nullptr ? scales[scaleIdx] : NVFP4E8M0ScaleToFloat(scaleBytes[scaleIdx]);
                    memcpy(dst, &scale, sizeof(float));
                    dst += sizeof(float);
                }
            }
        }
    }

    static int GetCrossSwigluSourceRow(int dstRow, int k) {
        int half = k / 2;
        return (dstRow & 1) ? (half + dstRow / 2) : (dstRow / 2);
    }

    void Nvfp4ToFastllmNVFP4_BLOCK16_E8M0_Rows(
        int k, int m, uint8_t *nvfp4, uint8_t *scaleBytes,
        int blockK, int blockM, uint8_t *dst, int dstRowStart, int dstRows, bool isCrossSwiglu
    ) {
        const int packBlockM = 16;
        const int fp4BytesPerBlock = packBlockM / 2;
        const int scaleMs = (m - 1) / blockM + 1;
        const int packedBlocks = (m - 1) / packBlockM + 1;
        size_t rawBytesPerRow = GetDataBytes(DataType::NVFP4, 1, m);
        size_t packedBytesPerRow = GetDataBytes(DataType::NVFP4_BLOCK_16_E8M0, 1, m);

        for (int localRow = 0; localRow < dstRows; localRow++) {
            int dstRow = dstRowStart + localRow;
            int srcRow = isCrossSwiglu ? GetCrossSwigluSourceRow(dstRow, k) : dstRow;
            uint8_t *rowDst = dst + (size_t)localRow * packedBytesPerRow;
            uint8_t *src = nvfp4 + (size_t)srcRow * rawBytesPerRow;
            for (int block = 0; block < packedBlocks; block++) {
                int blockStart = block * packBlockM;
                int blockElems = std::min(packBlockM, m - blockStart);
                int blockBytes = blockElems / 2;
                memcpy(rowDst, src + blockStart / 2, blockBytes);
                rowDst += fp4BytesPerBlock;

                int scaleCol = blockStart / blockM;
                size_t scaleIdx = (size_t)(srcRow / blockK) * scaleMs + scaleCol;
                *rowDst++ = scaleBytes[scaleIdx];
            }
        }
    }

    void Nvfp4ToFastllmNVFP4_BLOCK32_E8M0_Rows(
        int k, int m, uint8_t *nvfp4, uint8_t *scaleBytes,
        int blockK, int blockM, uint8_t *dst, int dstRowStart,
        int dstRows, bool isCrossSwiglu
    ) {
        AssertInFastLLM(blockM == 32,
                        "NVFP4 block32 packing requires blockM = 32.\n");
        const int packBlockM = 32;
        const int fp4BytesPerBlock = packBlockM / 2;
        const int scaleMs = (m - 1) / blockM + 1;
        const int packedBlocks = (m - 1) / packBlockM + 1;
        const size_t rawBytesPerRow =
            GetDataBytes(DataType::NVFP4, 1, m);
        const size_t packedBytesPerRow =
            GetDataBytes(DataType::NVFP4_BLOCK_32_E8M0, 1, m);

        for (int localRow = 0; localRow < dstRows; localRow++) {
            const int dstRow = dstRowStart + localRow;
            const int srcRow = isCrossSwiglu ?
                GetCrossSwigluSourceRow(dstRow, k) : dstRow;
            uint8_t *rowDst =
                dst + (size_t)localRow * packedBytesPerRow;
            const uint8_t *src =
                nvfp4 + (size_t)srcRow * rawBytesPerRow;
            for (int block = 0; block < packedBlocks; block++) {
                const int blockStart = block * packBlockM;
                const int blockElems = std::min(packBlockM, m - blockStart);
                const int blockBytes = (blockElems + 1) / 2;
                memset(rowDst, 0, fp4BytesPerBlock);
                memcpy(rowDst, src + blockStart / 2, blockBytes);
                rowDst += fp4BytesPerBlock;

                const size_t scaleIdx =
                    (size_t)(srcRow / blockK) * scaleMs + block;
                *rowDst++ = scaleBytes[scaleIdx];
            }
        }
    }

    void Int4ToFastllmInt4PerchannelRow(uint8_t *newWeight, uint8_t *oldWeight, int m) {
        if (GetCPUInstructInfo()->hasAVX512VNNI) {
            uint8_t *temp = new uint8_t[64];
            uint8_t *repack = new uint8_t[64];
            for (int i = 0; i < m; i += 64) {
                int len = std::min(m - i, 64);
                if (len == 64) {
                    for (int k = 0; k < 32; k++) {
                        temp[k * 2] = oldWeight[i / 2 + k] >> 4;
                        temp[k * 2 + 1] = oldWeight[i / 2 + k] & 0xF;
                    }
                            
                    for (int k = 0; k < 32; k++) {
                        repack[k * 2 + 1] = temp[k];
                        repack[k * 2] = temp[k + 32];
                    }

                    for (int k = 0; k < 32; k++) {
                        newWeight[i / 2 + k] = (repack[k * 2] << 4) + (repack[k * 2 + 1]);
                    }
                } else {
                    memcpy(newWeight + i / 2, oldWeight + i / 2, len / 2);
                }
            }
            delete[] temp;
            delete[] repack;
        } else if (GetCPUInstructInfo()->hasAVX2) {
            uint8_t *temp = new uint8_t[32];
            uint8_t *repack = new uint8_t[32];
            for (int i = 0; i < m; i += 32) {
                int len = std::min(m - i, 32);
                if (len == 32) {
                    for (int k = 0; k < 16; k++) {
                        temp[k * 2] = oldWeight[i / 2 + k] >> 4;
                        temp[k * 2 + 1] = oldWeight[i / 2 + k] & 0xF;
                    }
                            
                    for (int k = 0; k < 16; k++) {
                        repack[k * 2 + 1] = temp[k];
                        repack[k * 2] = temp[k + 16];
                    }

                    for (int k = 0; k < 16; k++) {
                        newWeight[i / 2 + k] = (repack[k * 2] << 4) + (repack[k * 2 + 1]);
                    }
                } else {
                    memcpy(newWeight + i / 2, oldWeight + i / 2, len / 2);
                }
            }
            delete[] temp;
            delete[] repack;
        } else {
            memcpy(newWeight, oldWeight, m / 2);
        }
    }

    void Int4ToFastllmInt4PerchannelPacked(int experts, int n, int m, uint8_t *qweight, float *mins, float *scales, std::vector <uint8_t> &int4Packed) {
        // 每行需要的字节数：m个uint4需要m/2个uint8，加上一个min和一个scale
        int int4BytesPerRow = m / 2;  // m是偶数，所以直接除以2
        int rowSize = int4BytesPerRow + sizeof(float) * 2;  // m/2个uint8 + 1个min + 1个scale
        
        // 调整输出vector大小
        int4Packed.resize((size_t)experts * n * rowSize);
        
        for (size_t i = 0; i < experts; i++) {
            for (size_t j = 0; j < n; j++) {
                size_t rowIdx = i * n + j;
                size_t packedOffset = rowIdx * rowSize;
                
                size_t currentPos = packedOffset;
                
                // 直接使用当前行的int4数据
                size_t srcOffset = i * n * int4BytesPerRow + j * int4BytesPerRow;
                Int4ToFastllmInt4PerchannelRow(&int4Packed[currentPos], &qweight[srcOffset], m);
                currentPos += int4BytesPerRow;
                
                // 添加当前行的min值
                float* minPtr = (float*)(&int4Packed[currentPos]);
                *minPtr = mins[rowIdx];
                currentPos += sizeof(float);
                
                // 添加当前行的scale值
                float* scalePtr = (float*)(&int4Packed[currentPos]);
                *scalePtr = scales[rowIdx];
                currentPos += sizeof(float);
            }
        }
    }

    void Int4ToFastllmInt4Group32Compact(int rows, int columns,
                                         const uint8_t *qweight,
                                         const float *mins,
                                         const float *scales,
                                         std::vector<uint8_t> &packed) {
        AssertInFastLLM(rows > 0 && columns > 0 && columns % 32 == 0,
                        "INT4_GROUP32 compact pack requires columns divisible by 32.");
        const int groups = columns / 32;
        for (size_t i = 0; i < (size_t)rows * groups; i++) {
            const float expectedMin = -8.0f * scales[i];
            const float tolerance = std::max(1e-6f, std::fabs(expectedMin) * 1e-6f);
            AssertInFastLLM(std::fabs(mins[i] - expectedMin) <= tolerance,
                            "INT4_GROUP32 compact pack only supports symmetric weights with zero point 8.");
        }
        const size_t quantBytes = (size_t)columns / 2;
        const size_t rowBytes = GetDataBytes(DataType::INT4_GROUP32, 1, columns);
        packed.resize((size_t)rows * rowBytes);
        for (int row = 0; row < rows; row++) {
            uint8_t *dst = packed.data() + (size_t)row * rowBytes;
            for (int group = 0; group < groups; group++) {
                uint8_t *dstBlock = dst +
                    GetInt4Group32DataOffset(group, groups);
                memcpy(dstBlock,
                       qweight + (size_t)row * quantBytes + (size_t)group * 16,
                       16);
                const uint16_t scale = Float32ToBFloat16RNEBits(
                    scales[(size_t)row * groups + group]);
                memcpy(dst + GetInt4Group32ScaleOffset(group, groups),
                       &scale, sizeof(scale));
            }
        }
    }

    void Int8ToFastllmInt8PerchannelPacked(int experts, int n, int m, uint8_t *qweight, int *zeros, float *scales, std::vector <uint8_t> &int8Packed) {
        int int8BytesPerRow = m;
        int rowSize = int8BytesPerRow + sizeof(float) * 2;  // m个uint8 + 1个min + 1个scale
        
        // 调整输出vector大小
        int8Packed.resize((size_t)experts * n * rowSize);
        
        for (size_t i = 0; i < experts; i++) {
            for (size_t j = 0; j < n; j++) {
                size_t rowIdx = i * n + j;
                size_t packedOffset = rowIdx * rowSize;
                
                size_t currentPos = packedOffset;
                
                // 直接使用当前行的int4数据
                size_t srcOffset = i * n * int8BytesPerRow + j * int8BytesPerRow;
                memcpy(&int8Packed[currentPos], &qweight[srcOffset], m);
                currentPos += int8BytesPerRow;
                
                // 添加当前行的min值
                float* minPtr = (float*)(&int8Packed[currentPos]);
                *minPtr = ((float)0 - zeros[rowIdx]) * scales[rowIdx];
                currentPos += sizeof(float);
                
                // 添加当前行的scale值
                float* scalePtr = (float*)(&int8Packed[currentPos]);
                *scalePtr = scales[rowIdx];
                currentPos += sizeof(float);
            }
        }
    }

    #ifdef USE_CUDA
    struct FastllmCudaHostFreeDeleter {
        void operator()(void *ptr) const {
            if (ptr != nullptr) {
                FastllmCudaHostFree(ptr);
            }
        }
    };

    struct FastllmCudaFreeDeleter {
        void operator()(void *ptr) const {
            if (ptr != nullptr) {
                FastllmCudaFree(ptr);
            }
        }
    };

    struct FastllmCudaStreamDestroyDeleter {
        int device = -1;

        void operator()(void *stream) const {
            if (stream == nullptr) {
                return;
            }
            int oriDevice = FastllmCudaGetDevice();
            if (device >= 0 && oriDevice != device) {
                FastllmCudaSetDevice(device);
            }
            FastllmCudaStreamDestroy(stream);
            if (device >= 0 && oriDevice != device) {
                FastllmCudaSetDevice(oriDevice);
            }
        }
    };
    #endif

    struct FastllmMoeDataManagerNumas {
            std::vector <float, alignedAllocator<float, 64> > gateUpOutput, swigluOutput, downOutput, reduceOutput;
            std::vector <float, alignedAllocator<float, 64> > inputFloat32;  // 当 input 非 FLOAT32 时暂存转成 float32 的数据
            std::vector <uint8_t, alignedAllocator<uint8_t, 64> > realInput, expandInput, downInput;
            std::vector<int> activeExperts;
            std::vector<std::pair<int, float>> selectedExperts;
            std::vector<int> expertOrder;
            std::vector<std::vector<MultiThreadGemmOp>> gemmTaskStorage;
            std::vector<std::vector<MultiThreadDeepSeekV4NumasDownPrepareOp>>
                prepareTaskStorage;
            std::vector<std::vector<MultiThreadBaseOp*>> taskPointers;
            std::vector<MultiThreadDeepSeekV4NumasRoundOutputOp> roundOps;
            std::vector<MultiThreadDeepSeekV4NumasStoreOutputOp> storeOps;
            std::vector<MultiThreadDeepSeekV4NumasReduceOp> reduceOps;
    #ifdef USE_CUDA
            std::unique_ptr<void, FastllmCudaHostFreeDeleter> pinnedOutput {nullptr};
            size_t pinnedOutputBytes = 0;
            std::unique_ptr<void, FastllmCudaFreeDeleter> gpuOutputStaging {nullptr};
            size_t gpuOutputStagingBytes = 0;
            int gpuOutputStagingDevice = -1;
            std::map<int, std::unique_ptr<Data>> gpuInputReplicas;
            std::unique_ptr<void, FastllmCudaStreamDestroyDeleter> gpuOutputCopyStream {
                nullptr, FastllmCudaStreamDestroyDeleter {}
            };

            uint8_t *EnsurePinnedOutput(size_t bytes) {
                if (pinnedOutputBytes < bytes || pinnedOutput.get() == nullptr) {
                    pinnedOutput.reset(FastllmCudaHostMalloc(bytes));
                    pinnedOutputBytes = bytes;
                }
                return (uint8_t*)pinnedOutput.get();
            }

            void *EnsureGpuOutputStaging(size_t bytes, int gpuId) {
                if (gpuOutputStagingDevice != gpuId) {
                    gpuOutputStaging.reset(nullptr);
                    gpuOutputStagingBytes = 0;
                    gpuOutputStagingDevice = gpuId;
                }
                if (gpuOutputStagingBytes < bytes || gpuOutputStaging.get() == nullptr) {
                    int oriDevice = FastllmCudaGetDevice();
                    if (oriDevice != gpuId) {
                        FastllmCudaSetDevice(gpuId);
                    }
                    gpuOutputStaging.reset(FastllmCudaMalloc(bytes));
                    if (oriDevice != gpuId) {
                        FastllmCudaSetDevice(oriDevice);
                    }
                    gpuOutputStagingBytes = bytes;
                }
                return gpuOutputStaging.get();
            }

            Data *StageGpuInputReplica(const Data &input, int gpuId) {
                auto &replica = gpuInputReplicas[gpuId];
                bool stale = replica == nullptr ||
                    replica->dataType != input.dataType ||
                    replica->dims != input.dims ||
                    replica->cudaData == nullptr ||
                    GetPointerDeviceId(replica->cudaData) != gpuId;
                if (stale) {
                    replica.reset(new Data(input.dataType, input.dims));
                    replica->dataDevice = DataDevice::CUDA;
                    replica->dataDeviceIds = {gpuId};
                    int oriDevice = FastllmCudaGetDevice();
                    if (oriDevice != gpuId) {
                        FastllmCudaSetDevice(gpuId);
                    }
                    replica->Allocate(false);
                    if (oriDevice != gpuId) {
                        FastllmCudaSetDevice(oriDevice);
                    }
                }
                AssertInFastLLM(
                    input.cpuData != nullptr,
                    "NUMA MoE cannot stage a CUDA input replica without CPU data.");
                int oriDevice = FastllmCudaGetDevice();
                if (oriDevice != gpuId) {
                    FastllmCudaSetDevice(gpuId);
                }
                FastllmCudaCopyFromHostToDevice(
                    replica->cudaData, input.cpuData, input.GetBytes());
                if (oriDevice != gpuId) {
                    FastllmCudaSetDevice(oriDevice);
                }
                return replica.get();
            }

            void *EnsureGpuOutputCopyStream(int gpuId) {
                if (gpuOutputCopyStream.get() == nullptr ||
                    gpuOutputCopyStream.get_deleter().device != gpuId) {
                    gpuOutputCopyStream.reset(nullptr);
                    int oriDevice = FastllmCudaGetDevice();
                    if (oriDevice != gpuId) {
                        FastllmCudaSetDevice(gpuId);
                    }
                    gpuOutputCopyStream = std::unique_ptr<void, FastllmCudaStreamDestroyDeleter>(
                        FastllmCudaStreamCreate(true), FastllmCudaStreamDestroyDeleter {gpuId}
                    );
                    if (oriDevice != gpuId) {
                        FastllmCudaSetDevice(oriDevice);
                    }
                }
                return gpuOutputCopyStream.get();
            }
    #endif
    };
    // 每层一个 MoE 缓存，避免层间共享导致的数据竞争。缓存中包含 CUDA
    // staging buffers，因此不能依赖跨翻译单元的静态析构顺序：CUDA 内存池可能
    // 先被析构。使用进程生命周期的容器，并由 release_memory 在 CUDA 仍可用时显式清理。
    static std::unordered_map<int, FastllmMoeDataManagerNumas> &
    GetNumasMoeRuntimeCache() {
        static auto *cache =
            new std::unordered_map<int, FastllmMoeDataManagerNumas>();
        return *cache;
    }

    void ClearNumasMoeRuntimeCache() {
        GetNumasMoeRuntimeCache().clear();
    }

    // CrossSwiglu 重排：将 a[n, m] 重排为 b[n, m]
    // b[0] = a[0], b[1] = a[n/2], b[2] = a[1], b[3] = a[n/2+1], ...
    // 即前半部分和后半部分行交错排列
    void CrossSwigluReorder(uint8_t *src, int k, size_t bytesPerRow, std::vector<uint8_t> &dst) {
        dst.resize((size_t)k * bytesPerRow);
        int half = k / 2;
        for (int i = 0; i < half; i++) {
            memcpy(dst.data() + (size_t)(2 * i) * bytesPerRow, 
                   src + (size_t)i * bytesPerRow, bytesPerRow);
            memcpy(dst.data() + (size_t)(2 * i + 1) * bytesPerRow, 
                   src + (size_t)(half + i) * bytesPerRow, bytesPerRow);
        }
    }

    static void RegisterNumasImpl(fastllm::Data *data,
                                  const std::string &weightType,
                                  bool allowPinned) {
        auto *numaConfig = GetNumaConfig();
        if (data == nullptr) {
            return;
        }
        if (data->numasData.size() == 0) {
            data->numasData.resize(numaConfig->numaCnt);

            int k = data->dims[0], m = data->dims[1];
            if (k % numaConfig->numaCnt != 0) {
                ErrorInFastLLM("Linear weight's size %% numaCnt != 0.");
            }
            int kPerNuma = k / numaConfig->numaCnt;

            bool isCrossSwiglu = (weightType == "linearSwiglu");
            bool usePinned = allowPinned &&
                             MoeEnvConfig::GetInstance().GetPinnedWeight();

            auto allocFunc = [usePinned](size_t size, int node) -> void* {
                if (usePinned) {
                    return allocate_pinned_numa(size, node);
                }
                return allocate_aligned_numa(size, node);
            };

            if (data->dataType == DataType::DATA_GGUF_FORMAT) {
                // GGUF格式需要先交错存储，然后再repack
                // 因为repack会将多行打包在一起，打包后行之间的数据互相依赖，无法再做行交错
                size_t bytesPerRow = GetDataBytes((DataType)((int)data->dataType + data->ggmlType), 1, m);
                if (isCrossSwiglu) {
                    std::vector<uint8_t> reordered;
                    CrossSwigluReorder(data->cpuData, k, bytesPerRow, reordered);
                    memcpy(data->cpuData, reordered.data(), (size_t)k * bytesPerRow);
                }
                // 交错存储完成后再repack
                data->Repack();
                // repack后bytesPerRow可能变化，重新获取
                bytesPerRow = GetDataBytes((DataType)((int)data->dataType + data->ggmlType), 1, m);
                uint8_t *srcData = data->cpuData;
                for (int i = 0; i < numaConfig->numaCnt; i++) {
                    data->numasData[i] = (uint8_t*)allocFunc(kPerNuma * bytesPerRow, i);
                    memcpy(data->numasData[i], srcData + (size_t)i * kPerNuma * bytesPerRow, kPerNuma * bytesPerRow);
                }
            } else {
                data->Repack();

                if (data->dataType == DataType::FLOAT32 ||
                    data->dataType == DataType::BFLOAT16 ||
                    data->dataType == DataType::FLOAT16 ||
                    data->dataType == DataType::FP8_E4M3_BLOCK_128 ||
                    data->dataType == DataType::FP8_E4M3_PERCHANNEL ||
                    data->dataType == DataType::INT8_PERCHANNEL ||
                    data->dataType == DataType::INT4_PERCHANNEL ||
                    data->dataType == DataType::INT4_GROUP128 ||
                    data->dataType == DataType::NVFP4_BLOCK_16 ||
                    data->dataType == DataType::NVFP4_BLOCK_16_E8M0 ||
                    data->dataType == DataType::NVFP4_BLOCK_32_E8M0 ||
                    data->dataType == DataType::INT4_GROUP32) {
                    size_t bytesPerRow = GetDataBytes(data->dataType, 1, m);
                    uint8_t *srcData = data->cpuData;
                    std::vector<uint8_t> reordered;
                    if (isCrossSwiglu) {
                        CrossSwigluReorder(data->cpuData, k, bytesPerRow, reordered);
                        srcData = reordered.data();
                    }
                    for (int i = 0; i < numaConfig->numaCnt; i++) {
                        data->numasData[i] = (uint8_t*)allocFunc(kPerNuma * bytesPerRow, i);
                        memcpy(data->numasData[i], srcData + (size_t)i * kPerNuma * bytesPerRow, kPerNuma * bytesPerRow);
                    }
                } else if (data->dataType == DataType::FP8_E4M3) {
                    std::vector <uint8_t> fp8Packed;
                    if (data->blockM == 128) {
                        data->dataType = DataType::FP8_E4M3_BLOCK_128;
                        Fp8ToFastllmFP8_E4M3_BLOCK128(1, k, m, (uint8_t*)data->cpuData, data->scales.data(), data->blockK, data->blockM, fp8Packed);
                    } else if (data->blockM == m) {
                        data->dataType = DataType::FP8_E4M3_PERCHANNEL;
                        Fp8PerchannelToFastllmFP8_E4M3_PERCHANNEL(1, k, m, (uint8_t*)data->cpuData, data->scales.data(), data->blockK, data->blockM, fp8Packed);
                    } else {
                        ErrorInFastLLM("RegisterNumas can't support fp8 with blockM = " + std::to_string(data->blockM));    
                    }

                    size_t bytesPerRow = GetDataBytes(data->dataType, 1, m);
                    if (isCrossSwiglu) {
                        std::vector<uint8_t> reordered;
                        CrossSwigluReorder(fp8Packed.data(), k, bytesPerRow, reordered);
                        fp8Packed.swap(reordered);
                    }
                    for (int i = 0; i < numaConfig->numaCnt; i++) {
                        data->numasData[i] = (uint8_t*)allocFunc(kPerNuma * bytesPerRow, i);
                        memcpy(data->numasData[i], fp8Packed.data() + (size_t)i * kPerNuma * bytesPerRow, kPerNuma * bytesPerRow);
                    }
                } else if (data->dataType == DataType::NVFP4) {
                    if (data->blockM < 16 || (data->blockM % 16 != 0 && data->blockM != m)) {
                        ErrorInFastLLM("RegisterNumas can't support nvfp4 with blockM = " + std::to_string(data->blockM));
                    }
                    float *scaleFloats = data->scales.empty() ? nullptr : data->scales.data();
                    uint8_t *scaleBytes = GetNVFP4ScaleData(*data);
                    AssertInFastLLM(scaleFloats != nullptr || scaleBytes != nullptr,
                                    "RegisterNumas can't find nvfp4 scale data.");
                    if (scaleBytes != nullptr) {
                        const bool useBlock32 = data->blockM == 32 &&
                            GetCPUInstructInfo()->hasAVX512BF16;
                        data->dataType = useBlock32 ?
                            DataType::NVFP4_BLOCK_32_E8M0 :
                            DataType::NVFP4_BLOCK_16_E8M0;
                        const size_t packedBytesPerRow =
                            GetDataBytes(data->dataType, 1, m);
                        for (int i = 0; i < numaConfig->numaCnt; i++) {
                            data->numasData[i] = (uint8_t*)allocFunc(
                                kPerNuma * packedBytesPerRow, i);
                            if (useBlock32) {
                                Nvfp4ToFastllmNVFP4_BLOCK32_E8M0_Rows(
                                    k, m, (uint8_t*)data->cpuData,
                                    scaleBytes, data->blockK, data->blockM,
                                    data->numasData[i], i * kPerNuma,
                                    kPerNuma, isCrossSwiglu);
                            } else {
                                Nvfp4ToFastllmNVFP4_BLOCK16_E8M0_Rows(
                                    k, m, (uint8_t*)data->cpuData,
                                    scaleBytes, data->blockK, data->blockM,
                                    data->numasData[i], i * kPerNuma,
                                    kPerNuma, isCrossSwiglu);
                            }
                        }
                    } else {
                        std::vector<uint8_t> nvfp4Packed;
                        Nvfp4ToFastllmNVFP4_BLOCK16(1, k, m, (uint8_t*)data->cpuData, scaleFloats, scaleBytes, data->blockK, data->blockM, nvfp4Packed);
                        data->dataType = DataType::NVFP4_BLOCK_16;
                        size_t packedBytesPerRow = GetDataBytes(DataType::NVFP4_BLOCK_16, 1, m);
                        if (isCrossSwiglu) {
                            std::vector<uint8_t> reordered;
                            CrossSwigluReorder(nvfp4Packed.data(), k, packedBytesPerRow, reordered);
                            nvfp4Packed.swap(reordered);
                        }
                        for (int i = 0; i < numaConfig->numaCnt; i++) {
                            data->numasData[i] = (uint8_t*)allocFunc(kPerNuma * packedBytesPerRow, i);
                            memcpy(data->numasData[i], nvfp4Packed.data() + (size_t)i * kPerNuma * packedBytesPerRow, kPerNuma * packedBytesPerRow);
                        }
                    }
                } else if (data->dataType == DataType::INT8) {
                    std::vector <uint8_t> int8Packed;
                    Int8ToFastllmInt8PerchannelPacked(1, k, m, (uint8_t*)data->cpuData, data->zeros.data(), data->scales.data(), int8Packed);
                    data->dataType = DataType::INT8_PERCHANNEL;

                    size_t bytesPerRow = GetDataBytes(data->dataType, 1, m);
                    if (isCrossSwiglu) {
                        std::vector<uint8_t> reordered;
                        CrossSwigluReorder(int8Packed.data(), k, bytesPerRow, reordered);
                        int8Packed.swap(reordered);
                    }
                    for (int i = 0; i < numaConfig->numaCnt; i++) {
                        data->numasData[i] = (uint8_t*)allocFunc(kPerNuma * bytesPerRow, i);
                        memcpy(data->numasData[i], int8Packed.data() + (size_t)i * kPerNuma * bytesPerRow, kPerNuma * bytesPerRow);
                    }
                } else if (data->dataType == DataType::INT4_NOZERO) {
                    std::vector <uint8_t> int4Packed;
                    Int4ToFastllmInt4PerchannelPacked(1, k, m, (uint8_t*)data->cpuData, data->mins.data(), data->scales.data(), int4Packed);
                    data->dataType = DataType::INT4_PERCHANNEL;

                    size_t bytesPerRow = GetDataBytes(data->dataType, 1, m);
                    if (isCrossSwiglu) {
                        std::vector<uint8_t> reordered;
                        CrossSwigluReorder(int4Packed.data(), k, bytesPerRow, reordered);
                        int4Packed.swap(reordered);
                    }
                    for (int i = 0; i < numaConfig->numaCnt; i++) {
                        data->numasData[i] = (uint8_t*)allocFunc(kPerNuma * bytesPerRow, i);
                        memcpy(data->numasData[i], int4Packed.data() + (size_t)i * kPerNuma * bytesPerRow, kPerNuma * bytesPerRow);
                    }
                } else if (data->dataType == DataType::INT4_GROUP) {
                    if (m % data->groupCnt > 0) {
                        ErrorInFastLLM("RegisterNumas can't support data type int4g when m % groupCnt > 0.");
                    }

                    std::vector <uint8_t> int4Packed;
                    if (data->groupCnt == 128) {
                        int groups = m / data->groupCnt;
                        Int4ToFastllmInt4PerchannelPacked(
                            1, k * groups, data->groupCnt, (uint8_t*)data->cpuData,
                            data->mins.data(), data->scales.data(), int4Packed);
                        data->dataType = DataType::INT4_GROUP128;
                    } else if (data->groupCnt == 32) {
                        Int4ToFastllmInt4Group32Compact(
                            k, m, (const uint8_t*)data->cpuData,
                            data->mins.data(),
                            data->scales.data(), int4Packed);
                        data->dataType = DataType::INT4_GROUP32;
                    } else {
                        ErrorInFastLLM("RegisterNumas can't support data type " + GetDataTypeName(data->dataType));
                    }

                    size_t bytesPerRow = GetDataBytes(data->dataType, 1, m);
                    if (isCrossSwiglu) {
                        std::vector<uint8_t> reordered;
                        CrossSwigluReorder(int4Packed.data(), k, bytesPerRow, reordered);
                        int4Packed.swap(reordered);
                    }
                    for (int i = 0; i < numaConfig->numaCnt; i++) {
                        data->numasData[i] = (uint8_t*)allocFunc(kPerNuma * bytesPerRow, i);
                        memcpy(data->numasData[i], int4Packed.data() + (size_t)i * kPerNuma * bytesPerRow, kPerNuma * bytesPerRow);
                    }
                } else {
                    ErrorInFastLLM("RegisterNumas can't support data type " + GetDataTypeName(data->dataType));
                }
            }
            data->expansionBytes = data->GetBytes();
            data->isPinned = usePinned;
        }

        if (data->dataType == DataType::INT4_GROUP32) {
            std::vector<float>().swap(data->scales);
            std::vector<float>().swap(data->mins);
            std::vector<int>().swap(data->zeros);
            std::vector<int>().swap(data->weightSum);
        }

        delete[] data->cpuData;
        data->cpuData = nullptr;
    }

    void RegisterNumas(fastllm::Data *data, std::string weightType) {
        RegisterNumasImpl(data, weightType, true);
    }

    namespace {
        struct NumasDeepSeekV4WoAPackOp : MultiThreadBaseOp {
            const uint8_t *input;
            float *packed;
            DataType inputType;
            int tokens, inputStride, groupDim, group;
            int tokenBegin, tokenEnd;

            NumasDeepSeekV4WoAPackOp(
                    const uint8_t *input, float *packed,
                    DataType inputType, int tokens, int inputStride,
                    int groupDim, int group,
                    int tokenBegin, int tokenEnd)
                : input(input), packed(packed), inputType(inputType),
                  tokens(tokens), inputStride(inputStride),
                  groupDim(groupDim), group(group),
                  tokenBegin(tokenBegin), tokenEnd(tokenEnd) {}

            void Run() override {
                for (int token = tokenBegin; token < tokenEnd; token++) {
                    size_t sourceOffset =
                        (size_t)token * inputStride +
                        (size_t)group * groupDim;
                    float *destination = packed +
                        ((size_t)group * tokens + token) * groupDim;
                    if (inputType == DataType::FLOAT32) {
                        memcpy(
                            destination,
                            (const float*)input + sourceOffset,
                            (size_t)groupDim * sizeof(float));
                    } else if (inputType == DataType::BFLOAT16) {
                        const uint16_t *source =
                            (const uint16_t*)input + sourceOffset;
                        for (int d = 0; d < groupDim; d++) {
                            destination[d] =
                                BFloat16BitsToFloat32(source[d]);
                        }
                    } else {
                        const uint16_t *source =
                            (const uint16_t*)input + sourceOffset;
                        for (int d = 0; d < groupDim; d++) {
                            destination[d] = half_to_float(source[d]);
                        }
                    }
                }
            }
        };

        struct NumasDeepSeekV4WoAGemmOp : MultiThreadBaseOp {
            uint8_t *input;
            DataType inputType;
            uint16_t *weight;
            float *output;
            int tokens, groupDim, oRank, rowBegin, rowEnd;

            NumasDeepSeekV4WoAGemmOp(
                    uint8_t *input, DataType inputType,
                    uint16_t *weight, float *output,
                    int tokens, int groupDim, int oRank,
                    int rowBegin, int rowEnd)
                : input(input), inputType(inputType), weight(weight),
                  output(output), tokens(tokens), groupDim(groupDim),
                  oRank(oRank), rowBegin(rowBegin), rowEnd(rowEnd) {}

            void Run() override {
                if (inputType == DataType::BFLOAT16) {
                    AssertInFastLLM(
                        LinearBFloat16Float16Decode_AVX512F_Kernel(
                            (uint16_t*)input, weight, nullptr, output,
                            tokens, groupDim, oRank, rowBegin, rowEnd),
                        "NUMA DeepSeekV4WoA BF16 decode kernel is unavailable.\n");
                } else {
                    MultiThreadLinearFloat32Float16Op(
                        (float*)input, weight, nullptr, output,
                        tokens, groupDim, oRank,
                        rowBegin, rowEnd).Run();
                }
            }
        };

        struct NumasDeepSeekV4WoAOutputOp : MultiThreadBaseOp {
            const float *groupOutput;
            uint16_t *output;
            int tokens, groups, oRank, group;
            int tokenBegin, tokenEnd;

            NumasDeepSeekV4WoAOutputOp(
                    const float *groupOutput, uint16_t *output,
                    int tokens, int groups, int oRank, int group,
                    int tokenBegin, int tokenEnd)
                : groupOutput(groupOutput), output(output),
                  tokens(tokens), groups(groups), oRank(oRank),
                  group(group), tokenBegin(tokenBegin),
                  tokenEnd(tokenEnd) {}

            void Run() override {
                for (int token = tokenBegin; token < tokenEnd; token++) {
                    const float *source = groupOutput +
                        ((size_t)group * tokens + token) * oRank;
                    uint16_t *destination = output +
                        ((size_t)token * groups + group) * oRank;
                    for (int row = 0; row < oRank; row++) {
                        destination[row] =
                            Float32ToBFloat16RNEBits(source[row]);
                    }
                }
            }
        };

        struct NumasDeepSeekV4WoAWorkspace {
            std::unique_ptr<float[]> groupInput;
            size_t groupInputCapacity = 0;
            std::unique_ptr<float[]> groupOutput;
            size_t groupOutputCapacity = 0;
            std::vector<std::vector<NumasDeepSeekV4WoAPackOp>> pack;
            std::vector<std::vector<NumasDeepSeekV4WoAGemmOp>> gemm;
            std::vector<std::vector<NumasDeepSeekV4WoAOutputOp>> finalize;
            std::vector<std::vector<MultiThreadBaseOp*>> taskPointers;

            float *EnsureGroupInput(size_t count) {
                if (groupInputCapacity < count) {
                    groupInput.reset(new float[count]);
                    groupInputCapacity = count;
                }
                return groupInput.get();
            }

            float *EnsureGroupOutput(size_t count) {
                if (groupOutputCapacity < count) {
                    groupOutput.reset(new float[count]);
                    groupOutputCapacity = count;
                }
                return groupOutput.get();
            }
        };

        template <typename Task>
        void BuildNumasDeepSeekV4WoATaskPointers(
                std::vector<std::vector<Task>> &storage,
                std::vector<std::vector<MultiThreadBaseOp*>> &pointers) {
            pointers.resize(storage.size());
            for (size_t node = 0; node < storage.size(); node++) {
                pointers[node].clear();
                pointers[node].reserve(storage[node].size());
                for (Task &task : storage[node]) {
                    pointers[node].push_back(&task);
                }
            }
        }

        bool IsNumasDeepSeekV4WoAWeightRegistered(
                const Data &weight, const NumaConfig *numaConfig) {
            if ((int)weight.numasData.size() != numaConfig->numaCnt) {
                return false;
            }
            for (const uint8_t *shard : weight.numasData) {
                if (shard == nullptr) {
                    return false;
                }
            }
            return true;
        }

        void EnsureNumasDeepSeekV4WoAWeightRegistered(
                Data &weight, int groups, int oRank, int groupDim,
                NumaConfig *numaConfig) {
            if (IsNumasDeepSeekV4WoAWeightRegistered(
                    weight, numaConfig)) {
                return;
            }
            static std::mutex registrationMutex;
            std::lock_guard<std::mutex> guard(registrationMutex);
            if (IsNumasDeepSeekV4WoAWeightRegistered(
                    weight, numaConfig)) {
                return;
            }
            AssertInFastLLM(
                weight.cpuData != nullptr && !weight.isDiskWeight &&
                    weight.dataType == DataType::FLOAT16,
                "NUMA DeepSeekV4WoA requires a resident FP16 weight before "
                "its first invocation.\n");
            const int groupsPerNode = groups / numaConfig->numaCnt;
            const size_t elementsPerGroup =
                (size_t)oRank * groupDim;
            const size_t shardBytes =
                (size_t)groupsPerNode * elementsPerGroup *
                sizeof(uint16_t);
            weight.numasData.resize(numaConfig->numaCnt);
            for (int node = 0; node < numaConfig->numaCnt; node++) {
                weight.numasData[node] = (uint8_t*)allocate_aligned_numa(
                    shardBytes, node);
                memcpy(
                    weight.numasData[node],
                    (const uint16_t*)weight.cpuData +
                        (size_t)node * groupsPerNode * elementsPerGroup,
                    shardBytes);
            }
            delete[] weight.cpuData;
            weight.cpuData = nullptr;
            weight.isPinned = false;
        }

        bool CanUseNumasDeepSeekV4WoA(
                const Data &input, const Data &weight,
                int groups, int oRank, const NumaConfig *numaConfig) {
            if (input.dataDevice != DataDevice::CPU ||
                input.cpuData == nullptr || input.dims.size() != 4 ||
                (input.dataType != DataType::FLOAT32 &&
                 input.dataType != DataType::FLOAT16 &&
                 input.dataType != DataType::BFLOAT16) ||
                weight.dataType != DataType::FLOAT16 ||
                groups <= 0 || oRank <= 0 ||
                groups % numaConfig->numaCnt != 0) {
                return false;
            }
            const int heads = input.dims[2];
            if (heads <= 0 || heads % groups != 0 ||
                input.dims[0] <= 0 || input.dims[1] <= 0 ||
                input.dims[3] <= 0) {
                return false;
            }
            const int groupDim = heads / groups * input.dims[3];
            return weight.Count(0) ==
                (uint64_t)groups * oRank * groupDim;
        }
    }

    void NumasDeepSeekV4WoAOp::Run(
            const std::string &opType,
            const fastllm::DataDict &datas,
            const fastllm::FloatDict &floatParams,
            const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &output = *(datas.find("output")->second);
        const int groups = intParams.find("groups") != intParams.end() ?
            intParams.find("groups")->second : 1;
        const int oRank = intParams.find("oRank") != intParams.end() ?
            intParams.find("oRank")->second : 1;
        NumaConfig *numaConfig = GetNumaConfig();

        if (!CanUseNumasDeepSeekV4WoA(
                input, weight, groups, oRank, numaConfig)) {
            CpuDeepSeekV4WoAOp::Run(
                opType, datas, floatParams, intParams);
            return;
        }

        const bool profile = GetFastllmEnv().printProfile;
        const auto begin = profile ? std::chrono::steady_clock::now() :
            std::chrono::steady_clock::time_point();
        const int bsz = input.dims[0];
        const int seqlen = input.dims[1];
        const int heads = input.dims[2];
        const int headDim = input.dims[3];
        const int tokens = bsz * seqlen;
        const int inputStride = heads * headDim;
        const int groupDim = heads / groups * headDim;
        const int groupsPerNode = groups / numaConfig->numaCnt;
        const bool directBFloat16Decode =
            tokens == 1 && input.dataType == DataType::BFLOAT16 &&
            GetCPUInstructInfo()->hasAVX512F;

        EnsureNumasDeepSeekV4WoAWeightRegistered(
            weight, groups, oRank, groupDim, numaConfig);
        const auto registrationEnd = profile ?
            std::chrono::steady_clock::now() :
            std::chrono::steady_clock::time_point();

        static thread_local NumasDeepSeekV4WoAWorkspace workspace;
        float *groupInput = directBFloat16Decode ? nullptr :
            workspace.EnsureGroupInput(
                (size_t)groups * tokens * groupDim);
        float *groupOutput = workspace.EnsureGroupOutput(
            (size_t)groups * tokens * oRank);
        if (!directBFloat16Decode) {
            workspace.pack.clear();
            workspace.pack.resize(numaConfig->numaCnt);

            if (tokens == 1) {
                for (int group = 0; group < groups; group++) {
                    NumasDeepSeekV4WoAPackOp(
                        input.cpuData, groupInput, input.dataType,
                        tokens, inputStride, groupDim, group, 0, 1).Run();
                }
            } else {
                for (int node = 0; node < numaConfig->numaCnt; node++) {
                    int workers = std::max(
                        1, (int)numaConfig->numaToCpuDict[node].size());
                    int chunksPerGroup = std::max(
                        1, (workers + groupsPerNode - 1) / groupsPerNode);
                    auto &tasks = workspace.pack[node];
                    tasks.reserve((size_t)groupsPerNode * chunksPerGroup);
                    for (int localGroup = 0;
                         localGroup < groupsPerNode; localGroup++) {
                        int group = node * groupsPerNode + localGroup;
                        for (int chunk = 0; chunk < chunksPerGroup; chunk++) {
                            int tokenBegin = (int)(
                                (int64_t)tokens * chunk / chunksPerGroup);
                            int tokenEnd = (int)(
                                (int64_t)tokens * (chunk + 1) /
                                chunksPerGroup);
                            if (tokenBegin < tokenEnd) {
                                tasks.emplace_back(
                                    input.cpuData, groupInput,
                                    input.dataType, tokens, inputStride,
                                    groupDim, group, tokenBegin, tokenEnd);
                            }
                        }
                    }
                }
                BuildNumasDeepSeekV4WoATaskPointers(
                    workspace.pack, workspace.taskPointers);
                ScheduleDeepSeekV4NumasMoeTasks(
                    workspace.taskPointers, false);
            }
        }
        const auto prepareEnd = profile ?
            std::chrono::steady_clock::now() :
            std::chrono::steady_clock::time_point();

        workspace.gemm.clear();
        workspace.gemm.resize(numaConfig->numaCnt);
        constexpr int rowsPerTask = 64;
        const size_t elementsPerGroup = (size_t)oRank * groupDim;
        for (int node = 0; node < numaConfig->numaCnt; node++) {
            auto &tasks = workspace.gemm[node];
            tasks.reserve(
                (size_t)groupsPerNode *
                ((oRank + rowsPerTask - 1) / rowsPerTask));
            uint16_t *nodeWeight =
                (uint16_t*)weight.numasData[node];
            for (int localGroup = 0;
                 localGroup < groupsPerNode; localGroup++) {
                int group = node * groupsPerNode + localGroup;
                uint8_t *gemmInput = directBFloat16Decode ?
                    input.cpuData +
                        (size_t)group * groupDim * sizeof(uint16_t) :
                    (uint8_t*)(groupInput +
                        (size_t)group * tokens * groupDim);
                DataType gemmInputType = directBFloat16Decode ?
                    DataType::BFLOAT16 : DataType::FLOAT32;
                for (int row = 0; row < oRank; row += rowsPerTask) {
                    tasks.emplace_back(
                        gemmInput, gemmInputType,
                        nodeWeight +
                            (size_t)localGroup * elementsPerGroup,
                        groupOutput + (size_t)group * tokens * oRank,
                        tokens, groupDim, oRank, row,
                        std::min(row + rowsPerTask, oRank));
                }
            }
        }
        BuildNumasDeepSeekV4WoATaskPointers(
            workspace.gemm, workspace.taskPointers);
        ScheduleDeepSeekV4NumasMoeTasks(
            workspace.taskPointers, false);
        const auto gemmEnd = profile ?
            std::chrono::steady_clock::now() :
            std::chrono::steady_clock::time_point();

        output.dataType = DataType::BFLOAT16;
        output.Resize({bsz, seqlen, groups * oRank});
        output.Allocate(false);
        workspace.finalize.clear();
        workspace.finalize.resize(numaConfig->numaCnt);
        if (tokens == 1) {
            for (int group = 0; group < groups; group++) {
                NumasDeepSeekV4WoAOutputOp(
                    groupOutput, (uint16_t*)output.cpuData,
                    tokens, groups, oRank, group, 0, 1).Run();
            }
        } else {
            for (int node = 0; node < numaConfig->numaCnt; node++) {
                int workers = std::max(
                    1, (int)numaConfig->numaToCpuDict[node].size());
                int chunksPerGroup = std::max(
                    1, (workers + groupsPerNode - 1) / groupsPerNode);
                auto &tasks = workspace.finalize[node];
                tasks.reserve((size_t)groupsPerNode * chunksPerGroup);
                for (int localGroup = 0;
                     localGroup < groupsPerNode; localGroup++) {
                    int group = node * groupsPerNode + localGroup;
                    for (int chunk = 0; chunk < chunksPerGroup; chunk++) {
                        int tokenBegin = (int)(
                            (int64_t)tokens * chunk / chunksPerGroup);
                        int tokenEnd = (int)(
                            (int64_t)tokens * (chunk + 1) /
                            chunksPerGroup);
                        if (tokenBegin < tokenEnd) {
                            tasks.emplace_back(
                                groupOutput, (uint16_t*)output.cpuData,
                                tokens, groups, oRank, group,
                                tokenBegin, tokenEnd);
                        }
                    }
                }
            }
            BuildNumasDeepSeekV4WoATaskPointers(
                workspace.finalize, workspace.taskPointers);
            ScheduleDeepSeekV4NumasMoeTasks(
                workspace.taskPointers, false);
        }

        if (profile) {
            const auto end = std::chrono::steady_clock::now();
            const double registrationSeconds =
                std::chrono::duration<double>(
                    registrationEnd - begin).count();
            const double prepareSeconds =
                std::chrono::duration<double>(
                    prepareEnd - registrationEnd).count();
            const double gemmSeconds =
                std::chrono::duration<double>(
                    gemmEnd - prepareEnd).count();
            const double outputSeconds =
                std::chrono::duration<double>(end - gemmEnd).count();
            const double flops = 2.0 * (double)tokens * groups *
                                 oRank * groupDim;
            printf(
                "[fastllm-profile-dsv4-woa-numas] tokens=%d groups=%d "
                "m=%d k=%d register=%.6f prepare=%.6f gemm=%.6f "
                "output=%.6f compute=%.6f total=%.6f gflops=%.3f\n",
                tokens, groups, groupDim, oRank,
                registrationSeconds, prepareSeconds, gemmSeconds,
                outputSeconds,
                prepareSeconds + gemmSeconds + outputSeconds,
                registrationSeconds + prepareSeconds + gemmSeconds +
                    outputSeconds,
                flops / gemmSeconds / 1.0e9);
        }
    }

    static DataType GetNumasLinearActDataType(Data *weight, int batchSize) {
        // Mixed-quant MoE grouping queries the activation type before the
        // selected weights are registered. Mirror RegisterNumas' native
        // quantized-weight conversions so the pre-registration and registered
        // forms select the same activation buffer format.
        if (weight != nullptr &&
            (weight->dataType == DataType::INT8 ||
             weight->dataType == DataType::INT4_NOZERO)) {
            return DataType::INF_INT8_PERCHANNEL;
        }
        if (weight != nullptr && weight->dataType == DataType::INT4_GROUP) {
            if (weight->groupCnt == 128) {
                return DataType::INF_INT8_GROUP128;
            }
            if (weight->groupCnt == 32) {
                return DataType::INF_INT8_GROUP32;
            }
        }
        if (weight != nullptr && weight->dataType == DataType::INT4_GROUP32) {
            return DataType::INF_INT8_GROUP32;
        }
        if (weight != nullptr && weight->dataType == DataType::FLOAT32) {
            return DataType::FLOAT32;
        }
        if (weight != nullptr &&
            (weight->dataType == DataType::NVFP4 ||
             weight->dataType == DataType::NVFP4_BLOCK_16 ||
             weight->dataType == DataType::NVFP4_BLOCK_16_E8M0 ||
             weight->dataType == DataType::NVFP4_BLOCK_32_E8M0)) {
            return DataType::BFLOAT16;
        }
        return weight->GetLinearActDataType(batchSize);
    }

    bool IsNumasLinearWeightSupported(const Data *weight) {
        if (weight == nullptr || weight->dims.size() != 2) {
            return false;
        }
        if (weight->dataType == DataType::DATA_GGUF_FORMAT) {
            return true;
        }
        if (weight->dataType == DataType::FP8_E4M3) {
            return weight->blockM == 128 ||
                   weight->blockM == weight->dims[1];
        }
        if (weight->dataType == DataType::NVFP4) {
            return weight->blockM >= 16 &&
                   (weight->blockM % 16 == 0 ||
                    weight->blockM == weight->dims[1]);
        }
        if (weight->dataType == DataType::INT4_GROUP) {
            return (weight->groupCnt == 32 || weight->groupCnt == 128) &&
                   weight->dims[1] % weight->groupCnt == 0;
        }
        switch (weight->dataType) {
            case DataType::FLOAT32:
            case DataType::BFLOAT16:
            case DataType::FLOAT16:
            case DataType::FP8_E4M3_BLOCK_128:
            case DataType::FP8_E4M3_PERCHANNEL:
            case DataType::NVFP4_BLOCK_16:
            case DataType::NVFP4_BLOCK_16_E8M0:
            case DataType::NVFP4_BLOCK_32_E8M0:
            case DataType::INT8:
            case DataType::INT8_PERCHANNEL:
            case DataType::INT4_NOZERO:
            case DataType::INT4_PERCHANNEL:
            case DataType::INT4_GROUP128:
            case DataType::INT4_GROUP32:
                return true;
            default:
                return false;
        }
    }

    bool IsNumasLinearWeightRegistered(const Data *weight) {
        if (!IsNumasLinearWeightSupported(weight)) {
            return false;
        }
        const NumaConfig *numaConfig = GetNumaConfig();
        if ((int)weight->numasData.size() != numaConfig->numaCnt) {
            return false;
        }
        for (const uint8_t *shard : weight->numasData) {
            if (shard == nullptr) {
                return false;
            }
        }
        return true;
    }

    namespace {
        struct NumasLinearGemmOp : MultiThreadBaseOp {
            const uint8_t *inputData;
            const uint8_t *weightData;
            float *outputData;
            void *finalOutput;
            const float *biasData;
            const float *finalizeBias;
            DataType inputType;
            DataType weightType;
            DataType outputType;
            int n, m, k;
            int localStart, localEnd, globalStart;

            NumasLinearGemmOp(
                    const uint8_t *inputData, DataType inputType,
                    const uint8_t *weightData, DataType weightType,
                    float *outputData, void *finalOutput,
                    DataType outputType, const float *biasData,
                    const float *finalizeBias,
                    int n, int m, int k,
                    int localStart, int localEnd, int globalStart)
                : inputData(inputData), weightData(weightData),
                  outputData(outputData), finalOutput(finalOutput),
                  biasData(biasData), finalizeBias(finalizeBias),
                  inputType(inputType), weightType(weightType),
                  outputType(outputType),
                  n(n), m(m), k(k),
                  localStart(localStart), localEnd(localEnd),
                  globalStart(globalStart) {}

            void Finalize() {
                const int start = globalStart + localStart;
                const int end = globalStart + localEnd;
                if (outputType == DataType::BFLOAT16) {
                    uint16_t *destination = (uint16_t*)finalOutput;
                    for (int row = 0; row < n; row++) {
                        const float *source =
                            outputData + (size_t)row * k;
                        uint16_t *current =
                            destination + (size_t)row * k;
                        if (finalizeBias == nullptr) {
                            for (int column = start; column < end;
                                 column++) {
                                current[column] =
                                    Float32ToBFloat16RNEBits(
                                        source[column]);
                            }
                        } else {
                            for (int column = start; column < end;
                                 column++) {
                                current[column] =
                                    Float32ToBFloat16RNEBits(
                                        source[column] +
                                        finalizeBias[column]);
                            }
                        }
                    }
                } else if (finalizeBias != nullptr) {
                    float *destination = (float*)finalOutput;
                    for (int row = 0; row < n; row++) {
                        float *current =
                            destination + (size_t)row * k;
                        for (int column = start; column < end;
                             column++) {
                            current[column] += finalizeBias[column];
                        }
                    }
                }
            }

            void Run() override {
                if (inputType == DataType::FLOAT32 &&
                    weightType == DataType::FLOAT32) {
                    MultiThreadLinearFloat32Float32Op(
                        (float*)inputData, (float*)weightData,
                        biasData == nullptr ? nullptr :
                            (float*)biasData + globalStart,
                        outputData + globalStart,
                        n, m, k, localStart, localEnd).Run();
                } else if (inputType == DataType::FLOAT32 &&
                           weightType == DataType::FLOAT16) {
                    MultiThreadLinearFloat32Float16Op(
                        (float*)inputData, (uint16_t*)weightData,
                        biasData == nullptr ? nullptr :
                            (float*)biasData + globalStart,
                        outputData + globalStart,
                        n, m, k, localStart, localEnd).Run();
                } else if (inputType == DataType::BFLOAT16 &&
                           weightType == DataType::FLOAT16) {
                    AssertInFastLLM(
                        LinearBFloat16Float16Decode_AVX512F_Kernel(
                            (uint16_t*)inputData,
                            (uint16_t*)weightData, nullptr,
                            outputData + globalStart,
                            n, m, k, localStart, localEnd),
                        "NUMA BF16 x FP16 decode kernel is unavailable.\n");
                } else {
                    FastllmGemm(
                        n, m, k,
                        inputData, GetDataBytes(inputType, 1, m),
                        weightData, GetDataBytes(weightType, 1, m),
                        outputData + globalStart,
                        GetDataBytes(DataType::FLOAT32, 1, k),
                        localStart, localEnd,
                        inputType, weightType, DataType::FLOAT32);
                }
                Finalize();
            }
        };

        struct NumasLinearWorkspace {
            std::unique_ptr<uint16_t[]> convertedBFloatInput;
            size_t convertedBFloatInputCapacity = 0;
            std::unique_ptr<float[]> convertedFloatInput;
            size_t convertedFloatInputCapacity = 0;
            std::unique_ptr<float[]> floatOutput;
            size_t floatOutputCapacity = 0;
            std::vector<std::vector<NumasLinearGemmOp>> gemmStorage;
            std::vector<std::vector<MultiThreadBaseOp*>> taskPointers;

            uint16_t *EnsureConvertedBFloatInput(size_t count) {
                if (convertedBFloatInputCapacity < count) {
                    convertedBFloatInput.reset(new uint16_t[count]);
                    convertedBFloatInputCapacity = count;
                }
                return convertedBFloatInput.get();
            }

            float *EnsureConvertedFloatInput(size_t count) {
                if (convertedFloatInputCapacity < count) {
                    convertedFloatInput.reset(new float[count]);
                    convertedFloatInputCapacity = count;
                }
                return convertedFloatInput.get();
            }

            float *EnsureFloatOutput(size_t count) {
                if (floatOutputCapacity < count) {
                    floatOutput.reset(new float[count]);
                    floatOutputCapacity = count;
                }
                return floatOutput.get();
            }
        };

        NumasLinearWorkspace &GetNumasLinearWorkspace() {
            static thread_local NumasLinearWorkspace workspace;
            return workspace;
        }

        bool CanUseNumasLinearPath(
                const Data &input, const Data &weight,
                const Data &output, const NumaConfig *numaConfig) {
            if ((input.dataType != DataType::FLOAT32 &&
                 input.dataType != DataType::BFLOAT16) ||
                output.dataType != input.dataType ||
                weight.dims.size() != 2 ||
                weight.dims[0] % numaConfig->numaCnt != 0) {
                return false;
            }
            switch (weight.dataType) {
                case DataType::FLOAT16:
                case DataType::BFLOAT16:
                case DataType::FP8_E4M3:
                case DataType::FP8_E4M3_BLOCK_128:
                case DataType::FP8_E4M3_PERCHANNEL:
                    return true;
                case DataType::FLOAT32:
                    return input.dataType == DataType::FLOAT32;
                default:
                    return false;
            }
        }

        void EnsureNumasLinearWeightRegistered(Data &weight) {
            if (IsNumasLinearWeightRegistered(&weight)) {
                return;
            }
            static std::mutex registrationMutex;
            std::lock_guard<std::mutex> guard(registrationMutex);
            if (IsNumasLinearWeightRegistered(&weight)) {
                return;
            }
            AssertInFastLLM(
                weight.cpuData != nullptr && !weight.isDiskWeight,
                "NUMA Linear requires a resident CPU weight before its first "
                "invocation.\n");
            // Ordinary Linear weights stay on CPU.  Unlike streamed MoE
            // weights, pinning them would defeat NUMA first-touch placement,
            // so always create node-local shards here.
            RegisterNumasImpl(&weight, "linearColumn", false);
        }

        template <typename Task>
        void BuildNumasLinearTaskPointers(
                std::vector<std::vector<Task>> &storage,
                std::vector<std::vector<MultiThreadBaseOp*>> &pointers) {
            pointers.resize(storage.size());
            for (size_t node = 0; node < storage.size(); node++) {
                pointers[node].clear();
                pointers[node].reserve(storage[node].size());
                for (Task &task : storage[node]) {
                    pointers[node].push_back(&task);
                }
            }
        }
    }

    void NumasLinearOp::Run(
            const std::string &opType, const fastllm::DataDict &datas,
            const fastllm::FloatDict &floatParams,
            const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);
        auto *numaConfig = GetNumaConfig();

        if (!CanUseNumasLinearPath(input, weight, output, numaConfig)) {
            CpuLinearOp::Run(opType, datas, floatParams, intParams);
            return;
        }
        AssertInFastLLM(
            bias.dataType == DataType::FLOAT32,
            "Linear's bias' type should be float32.\n");

        const bool profile = GetFastllmEnv().printProfile;
        const auto begin = profile ? std::chrono::steady_clock::now() :
            std::chrono::steady_clock::time_point();
        EnsureNumasLinearWeightRegistered(weight);
        const auto registrationEnd = profile ?
            std::chrono::steady_clock::now() :
            std::chrono::steady_clock::time_point();
        output.Allocate();

        int n = input.Count(0) / input.dims.back();
        int m = input.dims.back();
        int k = output.dims.back();
        NumasLinearWorkspace &workspace = GetNumasLinearWorkspace();

        const float *biasData = bias.dims.empty() ? nullptr :
            (const float*)bias.cpuData;
        const uint8_t *gemmInput = input.cpuData;
        DataType gemmInputType = input.dataType;
        const DataType weightType = weight.GetDataType();
        if (input.dataType == DataType::FLOAT32 &&
            weightType != DataType::FLOAT32 &&
            weightType != DataType::FLOAT16) {
            uint16_t *converted = workspace.EnsureConvertedBFloatInput(
                (size_t)n * m);
            RunMultiThreadConvertFromFloat32(
                converted, DataType::BFLOAT16,
                (const float*)input.cpuData, n, m, GetAlivePool());
            gemmInput = (const uint8_t*)converted;
            gemmInputType = DataType::BFLOAT16;
        } else if (input.dataType == DataType::BFLOAT16 &&
                   weightType == DataType::FLOAT16 &&
                   (n != 1 ||
                    !GetCPUInstructInfo()->hasAVX512F)) {
            float *converted = workspace.EnsureConvertedFloatInput(
                (size_t)n * m);
            BFloat16ToFloat32(
                (uint16_t*)input.cpuData, converted, n * m);
            gemmInput = (const uint8_t*)converted;
            gemmInputType = DataType::FLOAT32;
        }

        float *floatOutput = output.dataType == DataType::FLOAT32 ?
            (float*)output.cpuData :
            workspace.EnsureFloatOutput((size_t)n * k);
        const bool biasAppliedInGemm =
            gemmInputType == DataType::FLOAT32 &&
            (weight.GetDataType() == DataType::FLOAT32 ||
             weight.GetDataType() == DataType::FLOAT16);
        const float *finalizeBias = biasAppliedInGemm ? nullptr : biasData;
        const auto prepareEnd = profile ? std::chrono::steady_clock::now() :
            std::chrono::steady_clock::time_point();

        const int columnsPerNode = k / numaConfig->numaCnt;
        workspace.gemmStorage.resize(numaConfig->numaCnt);
        for (int node = 0; node < numaConfig->numaCnt; node++) {
            auto &storage = workspace.gemmStorage[node];
            storage.clear();
            int workers = std::max(
                1, (int)numaConfig->numaToCpuDict[node].size());
            int taskCount = std::min(workers, columnsPerNode);
            // Only collapse extreme one-column/two-column shards.  Wider
            // projections still benefit from all available memory channels.
            if (columnsPerNode < workers * 2) {
                constexpr int minColumnsPerTask = 8;
                int usefulTasks =
                    (columnsPerNode + minColumnsPerTask - 1) /
                    minColumnsPerTask;
                taskCount = std::min(workers, usefulTasks);
            }
            int columnsPerTask =
                (columnsPerNode + taskCount - 1) / taskCount;
            storage.reserve(taskCount);
            int globalStart = node * columnsPerNode;
            for (int start = 0; start < columnsPerNode;
                 start += columnsPerTask) {
                storage.emplace_back(
                    gemmInput, gemmInputType,
                    weight.numasData[node], weight.GetDataType(),
                    floatOutput, output.cpuData, output.dataType,
                    biasData, finalizeBias, n, m, k, start,
                    std::min(start + columnsPerTask, columnsPerNode),
                    globalStart);
            }
        }
        BuildNumasLinearTaskPointers(
            workspace.gemmStorage, workspace.taskPointers);
        ScheduleDeepSeekV4NumasMoeTasks(workspace.taskPointers, false);
        const auto gemmEnd = profile ? std::chrono::steady_clock::now() :
            std::chrono::steady_clock::time_point();

        if (profile) {
            const auto end = std::chrono::steady_clock::now();
            double registrationSeconds =
                std::chrono::duration<double>(registrationEnd - begin).count();
            double prepareSeconds =
                std::chrono::duration<double>(
                    prepareEnd - registrationEnd).count();
            double gemmSeconds =
                std::chrono::duration<double>(gemmEnd - prepareEnd).count();
            double finalizeSeconds =
                std::chrono::duration<double>(end - gemmEnd).count();
            double flops = 2.0 * (double)n * m * k;
            printf(
                "[fastllm-profile-numas-linear] n=%d m=%d k=%d "
                "input=%s weight=%s register=%.6f prepare=%.6f "
                "gemm=%.6f finalize=%.6f compute=%.6f total=%.6f "
                "gflops=%.3f\n",
                n, m, k, GetDataTypeName(input.dataType).c_str(),
                GetDataTypeName(weight.GetDataType()).c_str(),
                registrationSeconds, prepareSeconds, gemmSeconds,
                finalizeSeconds,
                prepareSeconds + gemmSeconds + finalizeSeconds,
                registrationSeconds + prepareSeconds + gemmSeconds +
                    finalizeSeconds,
                flops / gemmSeconds / 1.0e9);
        }
    }

    static bool CanUseCudaMoePrefill(DataType inputType, Data **weights, int weightsBatch) {
#ifndef USE_CUDA
        return false;
#else
        for (int i = 0; i < weightsBatch; i++) {
            if (weights[i] == nullptr) {
                continue;
            }
            // Match DoCudaLinear's dispatch key. GetDataType() expands GGUF into
            // its concrete ggml subtype, while the CUDA path accepts the common
            // DATA_GGUF_FORMAT storage type.
            DataType weightType = weights[i]->dataType;
            // Compact group-32 weights already have a bandwidth-efficient
            // AVX512-VNNI path.  Streaming many small experts over PCIe and
            // dequantizing them on every prefill is slower, and the dynamic
            // curve calibration itself can take minutes.  This only disables
            // NUMA-to-CUDA streaming; CUDA-resident INT4_GROUP32 weights still
            // use the native CUDA kernels below the normal Linear/MergeMOE path.
            if (weightType == DataType::INT4_GROUP32 &&
                GetCPUInstructInfo()->hasAVX512VNNI) {
                static std::once_flag compactWarningOnce;
                std::call_once(compactWarningOnce, []() {
                    printf("[Fastllm] NUMA GPU prefill keeps compact int4_group32 experts on NUMA: AVX512-VNNI is faster than sparse PCIe weight streaming.\n");
                });
                return false;
            }
            if (!IsCudaLinearDataTypeSupported(inputType, weightType, DataType::FLOAT32)) {
                static std::once_flag warningOnce;
                std::call_once(warningOnce, [inputType, weightType]() {
                    printf("[Fastllm] NUMA GPU prefill disabled: CUDA Linear does not support %s activations with %s MoE weights.\n",
                           GetDataTypeName(inputType).c_str(), GetDataTypeName(weightType).c_str());
                });
                return false;
            }
        }
        return true;
#endif
    }

    struct NumaWorkStealingOp : MultiThreadBaseOp {
        struct alignas(64) TaskState {
            std::atomic<int> curr;
            int end;
            std::vector<MultiThreadBaseOp*> tasks;
            std::atomic<bool> completed;
        };
        
        int threadId;
        int numaId;
        std::vector<TaskState*>* allStates;
        TaskState* myState;
        NumaConfig* numaConfig;
        
        NumaWorkStealingOp(int tid, int nid, std::vector<TaskState*>* states, 
                      TaskState* state, NumaConfig* config) 
            : threadId(tid), numaId(nid), allStates(states), 
              myState(state), numaConfig(config) {}
        
        void Run() override {
            // 首先执行自己的任务
            processOwnTasks();
            
            // 然后从同一NUMA节点的其他线程偷取任务
            stealFromSameNuma();
            
            // 标记完成
            myState->completed.store(true, std::memory_order_release);
        }
        
    private:
        void processOwnTasks() {
            while (true) {
                int taskId = myState->curr.fetch_add(1, std::memory_order_acq_rel);
                if (taskId >= myState->end) {
                    break;
                }
                if (taskId < myState->tasks.size()) {
                    myState->tasks[taskId]->Run();
                }
            }
        }
        
        void stealFromSameNuma() {
            // 获取同一NUMA节点的所有线程
            auto& numaThreads = numaConfig->numaToCpuDict[numaId];
            
            // 利用连续性计算位置：当前线程ID - NUMA节点第一个线程ID
            int numaStartThread = numaThreads[0].first;
            int myPos = threadId - numaStartThread;
            
            // 从当前线程开始，环形遍历其他线程
            for (int offset = 1; offset < numaThreads.size(); offset++) {
                int targetPos = (myPos + offset) % numaThreads.size();
                int tid = numaThreads[targetPos].first;
                
                TaskState* otherState = (*allStates)[tid];
                if (otherState == nullptr) continue;
                
                // 检查是否还有任务可偷
                while (true) {
                    int taskId = otherState->curr.fetch_add(1, std::memory_order_acq_rel);
                    if (taskId >= otherState->end) {
                        break;
                    }
                    if (taskId < otherState->tasks.size()) {
                        otherState->tasks[taskId]->Run();
                    }
                }
            }
        }
    };
    
    // 重构的动态任务调度函数，支持work-stealing
    void DynamicScheduleTasks(std::vector<std::vector<MultiThreadBaseOp*>>& ops) {
        auto *pool = GetAlivePool();
        auto *numaConfig = GetNumaConfig();
        
        // 创建任务状态数组
        using TaskState = typename NumaWorkStealingOp::TaskState;
        std::vector<TaskState*> taskStates(numaConfig->threads, nullptr);
        
        // 为每个线程分配任务状态
        for (int i = 0; i < numaConfig->threads; i++) {
            taskStates[i] = new (std::align_val_t{64}) TaskState();
            taskStates[i]->curr.store(0, std::memory_order_relaxed);
            taskStates[i]->end = 0;
            taskStates[i]->completed.store(false, std::memory_order_relaxed);
        }
        
        // 分配任务到各个线程
        int totalOps = 0;
        for (int nid = 0; nid < numaConfig->numaCnt; nid++) {
            totalOps += ops[nid].size();
            
            if (ops[nid].empty()) continue;
            
            int threadNum = numaConfig->numaToCpuDict[nid].size();
            if (threadNum == 0) continue;
            
            // 计算每个线程的任务数量
            int tasksPerThread = ops[nid].size() / threadNum;
            int remainingTasks = ops[nid].size() % threadNum;
            
            int taskIndex = 0;
            for (int i = 0; i < threadNum; i++) {
                int tid = numaConfig->numaToCpuDict[nid][i].first;
                int numTasks = tasksPerThread + (i < remainingTasks ? 1 : 0);
                
                if (numTasks > 0) {
                    // 分配任务到该线程
                    taskStates[tid]->tasks.clear();
                    taskStates[tid]->tasks.reserve(numTasks);
                    
                    for (int j = 0; j < numTasks && taskIndex < ops[nid].size(); j++) {
                        taskStates[tid]->tasks.push_back(ops[nid][taskIndex++]);
                    }
                    
                    taskStates[tid]->curr.store(0, std::memory_order_relaxed);
                    taskStates[tid]->end = taskStates[tid]->tasks.size();
                } else {
                    taskStates[tid]->end = 0;
                }
            }
        }
        
        // 创建work-stealing ops并提交到线程池
        std::vector<NumaWorkStealingOp*> wsOps(numaConfig->threads);
        for (int i = 0; i < numaConfig->threads; i++) {
            int numaId = numaConfig->threadIdToNumaDict[i];
            wsOps[i] = new NumaWorkStealingOp (
                i, numaId, &taskStates, taskStates[i], numaConfig
            );
            
            // 只有有任务的线程才启动
            if (taskStates[i] != nullptr && taskStates[i]->end > 0) {
                pool->PushOp(i, wsOps[i]);
            } else {
                // 没有任务的线程也要启动，以便参与work-stealing
                taskStates[i]->completed.store(true, std::memory_order_release);
                pool->PushOp(i, wsOps[i]);
            }
        }
        // 等待所有线程完成
        for (int i = 0; i < numaConfig->threads; i++) {
            pool->Wait(i);
        }
        
        // 清理资源
        for (int i = 0; i < numaConfig->threads; i++) {
            delete wsOps[i];
            if (taskStates[i] != nullptr) {
                taskStates[i]->~TaskState();
                #if __cpp_aligned_new >= 201606
                    operator delete(taskStates[i], std::align_val_t{64});
                #else
                    free_aligned(taskStates[i], sizeof(TaskState));
                #endif
            }
        }
        
        // 删除原始ops
        for (int nid = 0; nid < numaConfig->numaCnt; nid++) {
            for (auto* op : ops[nid]) {
                delete op;
            }
        }
    }

    namespace {
        struct NumasBatchMemcpyOp : MultiThreadBaseOp {
            uint8_t *dst;
            const uint8_t *src;
            size_t bytes;

            NumasBatchMemcpyOp(uint8_t *dst, const uint8_t *src, size_t bytes)
                : dst(dst), src(src), bytes(bytes) {}

            void Run() override {
                memcpy(dst, src, bytes);
            }
        };

        size_t AlignNumaArenaOffset(size_t value) {
            return (value + 63) & ~(size_t)63;
        }

        struct NumasLinearArenaEntry {
            Data *weight;
            size_t shardBytes;
            size_t offset;
        };

        DataType PredictRegisteredNumasLinearType(const Data &weight) {
            if (weight.dataType == DataType::FP8_E4M3) {
                return weight.blockM == 128 ?
                    DataType::FP8_E4M3_BLOCK_128 :
                    DataType::FP8_E4M3_PERCHANNEL;
            }
            if (weight.dataType == DataType::NVFP4) {
                if (GetNVFP4ScaleData(weight) == nullptr) {
                    return DataType::NVFP4_BLOCK_16;
                }
                const bool useBlock32 = weight.blockM == 32 &&
                    GetCPUInstructInfo()->hasAVX512BF16;
                return useBlock32 ?
                    DataType::NVFP4_BLOCK_32_E8M0 :
                    DataType::NVFP4_BLOCK_16_E8M0;
            }
            if (weight.dataType == DataType::INT8) {
                return DataType::INT8_PERCHANNEL;
            }
            if (weight.dataType == DataType::INT4_NOZERO) {
                return DataType::INT4_PERCHANNEL;
            }
            if (weight.dataType == DataType::INT4_GROUP) {
                return weight.groupCnt == 128 ?
                    DataType::INT4_GROUP128 : DataType::INT4_GROUP32;
            }
            return weight.dataType;
        }

        struct NumasBatchRegisterAndCopyOp : MultiThreadBaseOp {
            Data *weight;
            std::vector<uint8_t*> *arenas;
            size_t offset;
            size_t shardBytes;
            int numaCount;

            NumasBatchRegisterAndCopyOp(
                    Data *weight, std::vector<uint8_t*> *arenas,
                    size_t offset, size_t shardBytes, int numaCount)
                : weight(weight), arenas(arenas), offset(offset),
                  shardBytes(shardBytes), numaCount(numaCount) {}

            void Run() override {
                // Reuse the exact conversion and format dispatch used by
                // MergeMOE.  The short-lived shards are deliberately
                // unpinned: they are copied into the layer arena immediately.
                RegisterNumasImpl(weight, "linearColumn", false);
                AssertInFastLLM(
                    (int)weight->numasData.size() == numaCount,
                    "RegisterNumasLinearWeightBatch produced an incomplete "
                    "temporary shard table.");
                int rowsPerNode = weight->dims[0] / numaCount;
                size_t actualShardBytes = (size_t)rowsPerNode *
                    GetDataBytes(weight->GetDataType(), 1, weight->dims[1]);
                AssertInFastLLM(
                    actualShardBytes == shardBytes,
                    "RegisterNumasLinearWeightBatch predicted an invalid "
                    "packed row size.");
                for (int node = 0; node < numaCount; node++) {
                    uint8_t *temporary = weight->numasData[node];
                    uint8_t *destination = (*arenas)[node] + offset;
                    AssertInFastLLM(
                        temporary != nullptr,
                        "RegisterNumasLinearWeightBatch produced a missing "
                        "temporary NUMA shard.");
                    memcpy(destination, temporary, shardBytes);
                    free_aligned_numa(temporary, shardBytes);
                    weight->numasData[node] = destination;
                }
                weight->isPinned = false;
            }
        };
    }

    void RegisterNumasLinearWeightBatch(const std::vector<Data*> &weights) {
        auto *numaConfig = GetNumaConfig();
        std::vector<Data*> ggufPending;
        std::vector<Data*> genericPending;
        std::set<Data*> seen;
        ggufPending.reserve(weights.size());
        genericPending.reserve(weights.size());
        for (Data *weight : weights) {
            if (weight == nullptr || !weight->numasData.empty() ||
                !seen.insert(weight).second) {
                continue;
            }
            AssertInFastLLM(
                !weight->isDiskWeight && weight->cpuData != nullptr &&
                IsNumasLinearWeightSupported(weight),
                "RegisterNumasLinearWeightBatch received an unsupported "
                "or non-resident linear weight.");
            AssertInFastLLM(
                weight->dims[0] % numaConfig->numaCnt == 0,
                "RegisterNumasLinearWeightBatch requires rows divisible by "
                "the NUMA node count.");
            if (weight->dataType == DataType::DATA_GGUF_FORMAT) {
                ggufPending.push_back(weight);
            } else {
                genericPending.push_back(weight);
            }
        }
        if (ggufPending.empty() && genericPending.empty()) {
            return;
        }

        AliveThreadPool *pool = GetAlivePool();
        if (!ggufPending.empty()) {
            // Repack every GGUF tensor before any row shard is copied.
            // Q*_K_R4 packs groups of rows together, so repacking a shard
            // independently would change its layout at the node boundary.
            int threadCount = std::min(
                (int)ggufPending.size(), numaConfig->threads);
            std::vector<MultiThreadRepackWeightsOp*> repackOps;
            repackOps.reserve(threadCount);
            for (int thread = 0; thread < threadCount; thread++) {
                int start = (int)((int64_t)ggufPending.size() * thread /
                                  threadCount);
                int end = (int)((int64_t)ggufPending.size() * (thread + 1) /
                                threadCount);
                repackOps.push_back(new MultiThreadRepackWeightsOp(
                    ggufPending.data(), start, end));
                pool->PushOp(thread, repackOps.back());
            }
            for (int thread = 0; thread < threadCount; thread++) {
                pool->Wait(thread);
                delete repackOps[thread];
            }

            std::vector<NumasLinearArenaEntry> entries;
            entries.reserve(ggufPending.size());
            size_t arenaBytes = 0;
            for (Data *weight : ggufPending) {
                int rowsPerNode = weight->dims[0] / numaConfig->numaCnt;
                size_t rowBytes = GetDataBytes(
                    weight->GetDataType(), 1, weight->dims[1]);
                size_t shardBytes = (size_t)rowsPerNode * rowBytes;
                arenaBytes = AlignNumaArenaOffset(arenaBytes);
                entries.push_back({weight, shardBytes, arenaBytes});
                arenaBytes += shardBytes;
            }
            arenaBytes = AlignNumaArenaOffset(arenaBytes);

            std::vector<uint8_t*> arenas(
                numaConfig->numaCnt, nullptr);
            for (int node = 0; node < numaConfig->numaCnt; node++) {
                arenas[node] = (uint8_t*)allocate_aligned_numa(
                    arenaBytes, node);
                AssertInFastLLM(
                    arenas[node] != nullptr,
                    "RegisterNumasLinearWeightBatch failed to allocate a "
                    "GGUF NUMA weight arena.");
            }

            std::vector<std::vector<MultiThreadBaseOp*>> copyOps(
                numaConfig->numaCnt);
            for (const NumasLinearArenaEntry &entry : entries) {
                Data *weight = entry.weight;
                weight->numasData.resize(numaConfig->numaCnt);
                for (int node = 0; node < numaConfig->numaCnt; node++) {
                    uint8_t *destination = arenas[node] + entry.offset;
                    const uint8_t *source = weight->cpuData +
                        (size_t)node * entry.shardBytes;
                    weight->numasData[node] = destination;
                    copyOps[node].push_back(new NumasBatchMemcpyOp(
                        destination, source, entry.shardBytes));
                }
            }
            DynamicScheduleTasks(copyOps);

            for (Data *weight : ggufPending) {
                weight->expansionBytes = weight->GetBytes();
                weight->isPinned = false;
                delete[] weight->cpuData;
                weight->cpuData = nullptr;
            }
        }

        if (!genericPending.empty()) {
            // RegisterNumas owns all non-GGUF conversions used by MergeMOE.
            // Predict only the final row size here so that one layer arena can
            // be allocated up front; each worker converts into short-lived
            // shards and immediately consolidates them into that arena.
            std::vector<NumasLinearArenaEntry> entries;
            entries.reserve(genericPending.size());
            size_t arenaBytes = 0;
            for (Data *weight : genericPending) {
                DataType packedType =
                    PredictRegisteredNumasLinearType(*weight);
                int rowsPerNode = weight->dims[0] / numaConfig->numaCnt;
                size_t rowBytes = GetDataBytes(
                    packedType, 1, weight->dims[1]);
                size_t shardBytes = (size_t)rowsPerNode * rowBytes;
                arenaBytes = AlignNumaArenaOffset(arenaBytes);
                entries.push_back({weight, shardBytes, arenaBytes});
                arenaBytes += shardBytes;
            }
            arenaBytes = AlignNumaArenaOffset(arenaBytes);

            std::vector<uint8_t*> arenas(
                numaConfig->numaCnt, nullptr);
            for (int node = 0; node < numaConfig->numaCnt; node++) {
                arenas[node] = (uint8_t*)allocate_aligned_numa(
                    arenaBytes, node);
                AssertInFastLLM(
                    arenas[node] != nullptr,
                    "RegisterNumasLinearWeightBatch failed to allocate a "
                    "generic NUMA weight arena.");
            }

            std::vector<std::vector<MultiThreadBaseOp*>> registerOps(
                numaConfig->numaCnt);
            for (size_t i = 0; i < entries.size(); i++) {
                const NumasLinearArenaEntry &entry = entries[i];
                registerOps[i % numaConfig->numaCnt].push_back(
                    new NumasBatchRegisterAndCopyOp(
                        entry.weight, &arenas, entry.offset,
                        entry.shardBytes, numaConfig->numaCnt));
            }
            DynamicScheduleTasks(registerOps);
        }
#if defined(__linux__) && defined(__GLIBC__)
        // Safetensors may place hundreds of gigabytes of small expert
        // allocations in glibc arenas.  Returning the just-freed source
        // pages after every layer keeps warmup RSS close to the final NUMA
        // shard size instead of retaining both complete copies.
        malloc_trim(0);
#endif
    }

    namespace {
        bool IsKimiK3NumaWeight(const Data *weight) {
            return IsNumasLinearWeightRegistered(weight);
        }

        struct KimiK3NumaFormatGroup {
            DataType inputType = DataType::FLOAT32;
            std::vector<int> experts;
            int lines = 0;
        };

        bool IsDirectBFloat16Q8KType(DataType dataType) {
#ifdef __AVX2__
            if (dataType >= DataType::DATA_GGUF_FORMAT &&
                dataType < DataType::DATA_GGUF_FORMAT_END) {
                const ggml_type type = (ggml_type)(
                    (int)dataType - (int)DataType::DATA_GGUF_FORMAT);
                return type == GGML_TYPE_Q8_K ||
                       type == GGML_TYPE_Q8_K32;
            }
#endif
            return false;
        }

        inline float KimiK3MulAddPreserveOrder(
                float left, float right, float sum) {
            float product = left * right;
#if (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__))
            __asm__ __volatile__("" : "+x"(product));
#elif (defined(__GNUC__) || defined(__clang__)) && defined(__aarch64__)
            __asm__ __volatile__("" : "+w"(product));
#elif defined(__GNUC__) || defined(__clang__)
            __asm__ __volatile__("" : "+m"(product));
#endif
            return sum + product;
        }

        struct MultiThreadKimiK3NumaActivationOp : MultiThreadBaseOp {
            const float *gate;
            const float *up;
            uint16_t *activated;
            size_t start;
            size_t end;
            float beta;
            float linearBeta;

            MultiThreadKimiK3NumaActivationOp(
                    const float *gate, const float *up,
                    uint16_t *activated,
                    size_t start, size_t end, float beta, float linearBeta)
                : gate(gate), up(up), activated(activated),
                  start(start), end(end), beta(beta),
                  linearBeta(linearBeta) {}

            void Run() override {
                for (size_t i = start; i < end; i++) {
                    float gateValue = RoundFloat32ToBFloat16RNE(gate[i]);
                    float upValue = RoundFloat32ToBFloat16RNE(up[i]);
                    float sigmoid =
                        1.0f / (1.0f + std::exp(-gateValue));
                    float situ = beta * std::tanh(gateValue / beta) *
                                 sigmoid;
                    float boundedUp = linearBeta > 0.0f ?
                        linearBeta * std::tanh(upValue / linearBeta) :
                        upValue;
                    uint16_t bits = Float32ToBFloat16RNEBits(
                        situ * boundedUp);
                    activated[i] = bits;
                }
            }
        };

#ifdef __AVX2__
        inline __m256 RoundKimiK3Float32ToBFloat16Avx2(__m256 value) {
            __m256i bits = _mm256_castps_si256(value);
            const __m256i one = _mm256_set1_epi32(1);
            const __m256i bias = _mm256_add_epi32(
                _mm256_set1_epi32(0x7fff),
                _mm256_and_si256(_mm256_srli_epi32(bits, 16), one));
            bits = _mm256_add_epi32(bits, bias);
            bits = _mm256_slli_epi32(
                _mm256_srli_epi32(bits, 16), 16);
            return _mm256_castsi256_ps(bits);
        }

        inline void StoreKimiK3Float32AsBFloat16Avx2(
                uint16_t *destination, __m256 value) {
            __m256i bits = _mm256_castps_si256(value);
            const __m256i one = _mm256_set1_epi32(1);
            const __m256i bias = _mm256_add_epi32(
                _mm256_set1_epi32(0x7fff),
                _mm256_and_si256(_mm256_srli_epi32(bits, 16), one));
            bits = _mm256_srli_epi32(
                _mm256_add_epi32(bits, bias), 16);
            const __m128i packed = _mm_packus_epi32(
                _mm256_castsi256_si128(bits),
                _mm256_extracti128_si256(bits, 1));
            _mm_storeu_si128((__m128i*)destination, packed);
        }

        inline __m256 KimiK3MulAddPreserveOrderAvx2(
                __m256 left, __m256 right, __m256 sum) {
            __m256 product = _mm256_mul_ps(left, right);
#if defined(__GNUC__) || defined(__clang__)
            // The reference path rounds the multiplication to FLOAT32
            // before the addition.  A compiler is otherwise allowed to
            // contract these intrinsics into FMA and change the final BF16
            // bit by one ULP in cancellation-heavy rows.
            __asm__ __volatile__("" : "+x"(product));
#endif
            return _mm256_add_ps(sum, product);
        }
#endif

        struct MultiThreadKimiK3NumaFusedReduceOp : MultiThreadBaseOp {
            const float *downOutput;
            const int *routedLineOffsets;
            const float *score;
            uint16_t *output;
            int token;
            int topk;
            int outputDimension;
            int startChannel;
            int endChannel;

            MultiThreadKimiK3NumaFusedReduceOp(
                    const float *downOutput,
                    const int *routedLineOffsets,
                    const float *score, uint16_t *output,
                    int token, int topk, int outputDimension,
                    int startChannel, int endChannel)
                : downOutput(downOutput),
                  routedLineOffsets(routedLineOffsets), score(score),
                  output(output), token(token), topk(topk),
                  outputDimension(outputDimension),
                  startChannel(startChannel), endChannel(endChannel) {}

            void Run() override {
                const int routeBase = token * topk;
                uint16_t *destination =
                    output + (size_t)token * outputDimension;
                int channel = startChannel;
#ifdef __AVX2__
                for (; channel + 7 < endChannel; channel += 8) {
                    __m256 sum = _mm256_setzero_ps();
                    for (int slot = 0; slot < topk; slot++) {
                        const int line =
                            routedLineOffsets[routeBase + slot];
                        if (line < 0) {
                            continue;
                        }
                        const float *source = downOutput +
                            (size_t)line * outputDimension + channel;
                        const __m256 rounded =
                            RoundKimiK3Float32ToBFloat16Avx2(
                                _mm256_loadu_ps(source));
                        sum = KimiK3MulAddPreserveOrderAvx2(
                            rounded,
                            _mm256_set1_ps(score[routeBase + slot]), sum);
                    }
                    StoreKimiK3Float32AsBFloat16Avx2(
                        destination + channel, sum);
                }
#endif
                for (; channel < endChannel; channel++) {
                    float sum = 0.0f;
                    for (int slot = 0; slot < topk; slot++) {
                        const int line =
                            routedLineOffsets[routeBase + slot];
                        if (line < 0) {
                            continue;
                        }
                        sum = KimiK3MulAddPreserveOrder(
                            RoundFloat32ToBFloat16RNE(
                                downOutput[(size_t)line *
                                           outputDimension + channel]),
                            score[routeBase + slot], sum);
                    }
                    destination[channel] =
                        Float32ToBFloat16RNEBits(sum);
                }
            }
        };

        struct KimiK3NumaGemmInvocation {
            const std::vector<uint8_t*> *convertedInputs = nullptr;
            const std::vector<int> *routeCounts = nullptr;
            const std::vector<int> *expertLineOffsets = nullptr;
            const std::vector<int> *convertedLineOffsets = nullptr;
            float *firstOutput = nullptr;
            float *secondOutput = nullptr;
            size_t inputRowBytes = 0;
        };

        // Weight pointers, dimensions and output-column ranges never change
        // after NUMA warmup.  Keep them in a per-expert plan and read the
        // small invocation context below at execution time.  This avoids
        // rewriting every GEMM descriptor for every decoded token while
        // retaining the same FastllmGemm call and scheduling granularity.
        struct MultiThreadKimiK3NumaCachedGemmOp : MultiThreadBaseOp {
            KimiK3NumaGemmInvocation *invocation;
            uint8_t *weightData;
            DataType inputDataType;
            DataType weightDataType;
            int node;
            int expert;
            int outputSlot;
            int m;
            int k;
            int st;
            int end;
            int globalOutputOffset;

            MultiThreadKimiK3NumaCachedGemmOp(
                    KimiK3NumaGemmInvocation *invocation,
                    uint8_t *weightData, DataType inputDataType,
                    DataType weightDataType, int node, int expert,
                    int outputSlot, int m, int k, int st, int end,
                    int globalOutputOffset)
                : invocation(invocation), weightData(weightData),
                  inputDataType(inputDataType),
                  weightDataType(weightDataType), node(node), expert(expert),
                  outputSlot(outputSlot), m(m), k(k), st(st), end(end),
                  globalOutputOffset(globalOutputOffset) {}

            void Run() override {
                AssertInFastLLM(
                    invocation != nullptr &&
                    invocation->convertedInputs != nullptr &&
                    node < (int)invocation->convertedInputs->size(),
                    "KimiK3 NUMA cached GEMM has no invocation input.");
                const int rows = (*invocation->routeCounts)[expert];
                const int inputLine =
                    (*invocation->convertedLineOffsets)[expert];
                const int outputLine =
                    (*invocation->expertLineOffsets)[expert];
                uint8_t *input =
                    (*invocation->convertedInputs)[node] +
                    (size_t)inputLine * invocation->inputRowBytes;
                float *outputBase = outputSlot == 0 ?
                    invocation->firstOutput : invocation->secondOutput;
                AssertInFastLLM(
                    rows > 0 && inputLine >= 0 && outputLine >= 0 &&
                    outputBase != nullptr,
                    "KimiK3 NUMA cached GEMM has an invalid invocation.");
                uint8_t *output = (uint8_t*)(
                    outputBase + (size_t)outputLine * k +
                    globalOutputOffset);
                FastllmGemm(
                    rows, m, k,
                    input, invocation->inputRowBytes,
                    weightData, GetDataBytes(weightDataType, 1, m),
                    output, GetDataBytes(DataType::FLOAT32, 1, k),
                    st, end, inputDataType, weightDataType,
                    DataType::FLOAT32);
            }
        };

        struct KimiK3NumaExpertGemmTaskPlan {
            Data *firstWeight = nullptr;
            Data *secondWeight = nullptr;
            DataType firstWeightType = DataType::FLOAT32;
            DataType secondWeightType = DataType::FLOAT32;
            std::vector<uint8_t*> firstWeightData;
            std::vector<uint8_t*> secondWeightData;
            std::vector<std::vector<MultiThreadKimiK3NumaCachedGemmOp>>
                storage;
        };

        struct KimiK3NumaGemmTaskCache {
            KimiK3NumaGemmInvocation invocation;
            std::vector<std::unique_ptr<KimiK3NumaExpertGemmTaskPlan>>
                expertPlans;
            std::vector<std::vector<MultiThreadBaseOp*>> pointers;
        };

        struct MultiThreadKimiK3NumaGatherConvertOp : MultiThreadBaseOp {
            uint8_t *destination;
            DataType destinationType;
            const uint16_t *source;
            const int *sourceRows;
            size_t columns;
            size_t startRow;
            size_t endRow;

            MultiThreadKimiK3NumaGatherConvertOp(
                    uint8_t *destination, DataType destinationType,
                    const uint16_t *source, const int *sourceRows,
                    size_t columns, size_t startRow, size_t endRow)
                : destination(destination),
                  destinationType(destinationType), source(source),
                  sourceRows(sourceRows), columns(columns),
                  startRow(startRow), endRow(endRow) {}

            void Run() override {
                const size_t rowBytes =
                    GetDataBytes(destinationType, 1, columns);
                for (size_t row = startRow; row < endRow; row++) {
                    ConvertFromBFloat16(
                        destination + row * rowBytes, destinationType,
                        source + (size_t)sourceRows[row] * columns,
                        1, columns);
                }
            }
        };

        class KimiK3NumaLocalBufferSet {
        public:
            KimiK3NumaLocalBufferSet() = default;
            KimiK3NumaLocalBufferSet(
                const KimiK3NumaLocalBufferSet &) = delete;
            KimiK3NumaLocalBufferSet &operator=(
                const KimiK3NumaLocalBufferSet &) = delete;

            ~KimiK3NumaLocalBufferSet() {
                Release();
            }

            void Ensure(int numaCount, size_t bytes) {
                if ((int)pointers.size() != numaCount) {
                    Release();
                    pointers.assign(numaCount, nullptr);
                    capacities.assign(numaCount, 0);
                }
                for (int node = 0; node < numaCount; node++) {
                    if (capacities[node] >= bytes) {
                        continue;
                    }
                    if (pointers[node] != nullptr) {
                        free_aligned_numa(pointers[node], capacities[node]);
                    }
                    pointers[node] = (uint8_t*)allocate_aligned_numa(
                        bytes, node);
                    AssertInFastLLM(
                        pointers[node] != nullptr,
                        "KimiK3 failed to allocate a node-local activation "
                        "buffer.");
                    capacities[node] = bytes;
                }
            }

            const std::vector<uint8_t*> &Pointers() const {
                return pointers;
            }

        private:
            void Release() {
                for (size_t node = 0; node < pointers.size(); node++) {
                    if (pointers[node] != nullptr) {
                        free_aligned_numa(pointers[node], capacities[node]);
                    }
                }
                pointers.clear();
                capacities.clear();
            }

            std::vector<uint8_t*> pointers;
            std::vector<size_t> capacities;
        };

        struct KimiK3NumaWorkspace {
            std::vector<float, alignedAllocator<float, 64>> gateOutput;
            std::vector<float, alignedAllocator<float, 64>> upOutput;
            std::vector<float, alignedAllocator<float, 64>> downOutput;
            std::vector<uint16_t, alignedAllocator<uint16_t, 64>>
                gatheredInput;
            std::vector<uint16_t, alignedAllocator<uint16_t, 64>>
                conversionBfloat16;
            std::vector<uint16_t, alignedAllocator<uint16_t, 64>> activated;
            std::vector<uint8_t, alignedAllocator<uint8_t, 64>>
                convertedInput;
            KimiK3NumaLocalBufferSet numaConvertedInput;
            std::vector<uint8_t*> activeConvertedInputs;
            std::vector<std::vector<std::pair<int, int>>> routes;
            std::vector<int> activeExperts;
            std::vector<int> routeCounts;
            std::vector<int> expertLineOffsets;
            std::vector<int> convertedLineOffsets;
            std::vector<int> conversionSourceRows;
            std::vector<int> expertLineTokens;
            std::vector<int> routedLineOffsets;
            std::map<std::array<uintptr_t, 8>, KimiK3NumaGemmTaskCache>
                gemmTaskCaches;
            std::vector<
                std::vector<MultiThreadKimiK3NumaGatherConvertOp>>
                convertTaskStorage;
            std::vector<std::vector<MultiThreadKimiK3NumaActivationOp>>
                activationTaskStorage;
            std::vector<std::vector<MultiThreadKimiK3NumaFusedReduceOp>>
                reduceTaskStorage;
            std::vector<std::vector<MultiThreadBaseOp*>> taskPointers;
#ifdef USE_CUDA
            std::unique_ptr<void, FastllmCudaHostFreeDeleter>
                pinnedInput {nullptr};
            size_t pinnedInputBytes = 0;
            std::unique_ptr<void, FastllmCudaHostFreeDeleter>
                pinnedOutput {nullptr};
            size_t pinnedOutputBytes = 0;
            std::unique_ptr<void, FastllmCudaFreeDeleter>
                gpuOutputStaging {nullptr};
            size_t gpuOutputStagingBytes = 0;
            int gpuOutputStagingDevice = -1;
            std::unique_ptr<void, FastllmCudaStreamDestroyDeleter>
                inputCopyStream {
                    nullptr, FastllmCudaStreamDestroyDeleter {}
                };
            std::unique_ptr<void, FastllmCudaStreamDestroyDeleter>
                routeCopyStream {
                    nullptr, FastllmCudaStreamDestroyDeleter {}
                };
            std::unique_ptr<void, FastllmCudaStreamDestroyDeleter>
                outputCopyStream {
                    nullptr, FastllmCudaStreamDestroyDeleter {}
                };

            uint8_t *EnsurePinnedInput(size_t bytes) {
                if (pinnedInput.get() == nullptr ||
                    pinnedInputBytes < bytes) {
                    pinnedInput.reset(FastllmCudaHostMalloc(bytes));
                    pinnedInputBytes = bytes;
                }
                return (uint8_t*)pinnedInput.get();
            }

            uint8_t *EnsurePinnedOutput(size_t bytes) {
                if (pinnedOutput.get() == nullptr ||
                    pinnedOutputBytes < bytes) {
                    pinnedOutput.reset(FastllmCudaHostMalloc(bytes));
                    pinnedOutputBytes = bytes;
                }
                return (uint8_t*)pinnedOutput.get();
            }

            void *EnsureGpuOutputStaging(size_t bytes, int gpuId) {
                if (gpuOutputStagingDevice != gpuId) {
                    gpuOutputStaging.reset(nullptr);
                    gpuOutputStagingBytes = 0;
                    gpuOutputStagingDevice = gpuId;
                }
                if (gpuOutputStaging.get() == nullptr ||
                    gpuOutputStagingBytes < bytes) {
                    int originalDevice = FastllmCudaGetDevice();
                    if (originalDevice != gpuId) {
                        FastllmCudaSetDevice(gpuId);
                    }
                    gpuOutputStaging.reset(FastllmCudaMalloc(bytes));
                    if (originalDevice != gpuId) {
                        FastllmCudaSetDevice(originalDevice);
                    }
                    gpuOutputStagingBytes = bytes;
                }
                return gpuOutputStaging.get();
            }

            void *EnsureStream(
                    std::unique_ptr<void,
                        FastllmCudaStreamDestroyDeleter> &stream,
                    int gpuId) {
                if (stream.get() == nullptr ||
                    stream.get_deleter().device != gpuId) {
                    stream.reset(nullptr);
                    int originalDevice = FastllmCudaGetDevice();
                    if (originalDevice != gpuId) {
                        FastllmCudaSetDevice(gpuId);
                    }
                    stream = std::unique_ptr<
                        void, FastllmCudaStreamDestroyDeleter>(
                            FastllmCudaStreamCreate(true),
                            FastllmCudaStreamDestroyDeleter {gpuId});
                    if (originalDevice != gpuId) {
                        FastllmCudaSetDevice(originalDevice);
                    }
                }
                return stream.get();
            }

            void *EnsureInputCopyStream(int gpuId) {
                return EnsureStream(inputCopyStream, gpuId);
            }

            void *EnsureRouteCopyStream(int gpuId) {
                return EnsureStream(routeCopyStream, gpuId);
            }

            void *EnsureOutputCopyStream(int gpuId) {
                return EnsureStream(outputCopyStream, gpuId);
            }
#endif
        };

        KimiK3NumaWorkspace &GetKimiK3NumaWorkspace() {
            static thread_local KimiK3NumaWorkspace workspace;
            return workspace;
        }

        std::vector<KimiK3NumaFormatGroup> BuildKimiK3NumaFormatGroups(
                const std::vector<int> &activeExperts,
                const std::vector<int> &routeCounts,
                Data **firstWeights, Data **secondWeights) {
            std::vector<KimiK3NumaFormatGroup> groups;
            std::map<int, size_t> typeToGroup;
            for (int expert : activeExperts) {
                int rows = routeCounts[expert];
                DataType inputType = GetNumasLinearActDataType(
                    firstWeights[expert], rows);
                if (secondWeights != nullptr) {
                    AssertInFastLLM(
                        inputType == GetNumasLinearActDataType(
                            secondWeights[expert], rows),
                        "KimiK3 NUMA paired expert weights require the same "
                        "activation type.");
                }
                int typeKey = (int)inputType;
                auto it = typeToGroup.find(typeKey);
                if (it == typeToGroup.end()) {
                    size_t groupIndex = groups.size();
                    typeToGroup[typeKey] = groupIndex;
                    groups.push_back({inputType, {}, 0});
                    it = typeToGroup.find(typeKey);
                }
                KimiK3NumaFormatGroup &group = groups[it->second];
                group.experts.push_back(expert);
                group.lines += rows;
            }
            return groups;
        }

        void ConvertKimiK3NumaGroupInput(
                const KimiK3NumaFormatGroup &group,
                const std::vector<int> &routeCounts,
                const std::vector<int> &expertLineOffsets,
                const uint16_t *source, int columns,
                const std::vector<int> *sourceRowMap,
                std::vector<int> &convertedLineOffsets,
                KimiK3NumaWorkspace &workspace) {
            size_t elementCount = (size_t)group.lines * columns;
            int convertedLine = 0;
            workspace.conversionSourceRows.resize(group.lines);
            for (int expert : group.experts) {
                int rows = routeCounts[expert];
                convertedLineOffsets[expert] = convertedLine;
                for (int row = 0; row < rows; row++) {
                    int sourceRow = expertLineOffsets[expert] + row;
                    if (sourceRowMap != nullptr) {
                        AssertInFastLLM(
                            sourceRow >= 0 &&
                            sourceRow < (int)sourceRowMap->size(),
                            "KimiK3 NUMA gather source row is out of range.");
                        sourceRow = (*sourceRowMap)[sourceRow];
                    }
                    workspace.conversionSourceRows[convertedLine++] =
                        sourceRow;
                }
            }
            AssertInFastLLM(
                convertedLine == group.lines,
                "KimiK3 NUMA activation format group has an invalid row "
                "count.");

            const size_t convertedBytes = GetDataBytes(
                group.inputType, group.lines, columns);
            const bool directQ8K =
                IsDirectBFloat16Q8KType(group.inputType);
            NumaConfig *numaConfig = GetNumaConfig();
            if (directQ8K) {
                workspace.numaConvertedInput.Ensure(
                    numaConfig->numaCnt, convertedBytes);
                workspace.activeConvertedInputs.assign(
                    workspace.numaConvertedInput.Pointers().begin(),
                    workspace.numaConvertedInput.Pointers().end());
                workspace.convertTaskStorage.resize(numaConfig->numaCnt);
                workspace.taskPointers.resize(numaConfig->numaCnt);
                for (int node = 0; node < numaConfig->numaCnt; node++) {
                    auto &storage = workspace.convertTaskStorage[node];
                    auto &pointers = workspace.taskPointers[node];
                    storage.clear();
                    pointers.clear();
                    const int workers = std::max(
                        1, (int)numaConfig->numaToCpuDict[node].size());
                    const int desiredTasks = std::max(
                        1, std::min(group.lines, workers * 2));
                    const int rowsPerTask =
                        (group.lines + desiredTasks - 1) / desiredTasks;
                    const int taskCount =
                        (group.lines + rowsPerTask - 1) / rowsPerTask;
                    storage.reserve(taskCount);
                    pointers.reserve(taskCount);
                    for (int row = 0; row < group.lines;
                         row += rowsPerTask) {
                        storage.emplace_back(
                            workspace.activeConvertedInputs[node],
                            group.inputType, source,
                            workspace.conversionSourceRows.data(), columns,
                            row, std::min(row + rowsPerTask, group.lines));
                    }
                    for (MultiThreadKimiK3NumaGatherConvertOp &task :
                         storage) {
                        pointers.push_back(&task);
                    }
                }
                ScheduleDeepSeekV4NumasMoeTasks(
                    workspace.taskPointers, false);
            } else {
                int firstSourceRow = workspace.conversionSourceRows[0];
                bool sourceIsContiguous = true;
                for (int row = 1; row < group.lines; row++) {
                    sourceIsContiguous = sourceIsContiguous &&
                        workspace.conversionSourceRows[row] ==
                            firstSourceRow + row;
                }
                const uint16_t *conversionSource = nullptr;
                if (sourceIsContiguous) {
                    conversionSource = source +
                        (size_t)firstSourceRow * columns;
                } else {
                    workspace.conversionBfloat16.resize(elementCount);
                    for (int row = 0; row < group.lines; row++) {
                        memcpy(
                            workspace.conversionBfloat16.data() +
                                (size_t)row * columns,
                            source +
                                (size_t)workspace.conversionSourceRows[row] *
                                    columns,
                            (size_t)columns * sizeof(uint16_t));
                    }
                    conversionSource = workspace.conversionBfloat16.data();
                }
                workspace.convertedInput.resize(convertedBytes);
                RunMultiThreadConvertFromBFloat16(
                    workspace.convertedInput.data(), group.inputType,
                    conversionSource, group.lines, columns, GetAlivePool());
                workspace.activeConvertedInputs.assign(
                    numaConfig->numaCnt, workspace.convertedInput.data());
            }
        }

        void RunKimiK3NumaGemmGroup(
                const KimiK3NumaFormatGroup &group,
                const std::vector<int> &routeCounts,
                const std::vector<int> &expertLineOffsets,
                const std::vector<int> &convertedLineOffsets,
                int inputDimension, int outputDimension,
                Data **firstWeights, float *firstOutput,
                Data **secondWeights, float *secondOutput,
                KimiK3NumaWorkspace &workspace) {
            NumaConfig *numaConfig = GetNumaConfig();
            AssertInFastLLM(
                outputDimension % numaConfig->numaCnt == 0,
                "KimiK3 NUMA output rows must be divisible by the NUMA "
                "node count.");
            int outputPerNode = outputDimension / numaConfig->numaCnt;
            int weightsPerExpert = secondWeights == nullptr ? 1 : 2;

            // Keep two waves of equally sized work per node.  Four-row
            // alignment is required by repacked GGUF R4 kernels and is
            // harmless for the other supported formats.
            int workersPerNode = 1;
            for (int node = 0; node < numaConfig->numaCnt; node++) {
                workersPerNode = std::max(
                    workersPerNode,
                    (int)numaConfig->numaToCpuDict[node].size());
            }
            int weightsPerNode = std::max(
                1, (int)group.experts.size() * weightsPerExpert);
            constexpr int taskWaves = 2;
            int desiredChunks = std::max(
                1, (workersPerNode * taskWaves + weightsPerNode - 1) /
                       weightsPerNode);
            int desiredRows =
                (outputPerNode + desiredChunks - 1) / desiredChunks;
            int rowsPerTask =
                std::max(4, (desiredRows + 3) / 4 * 4);
            int chunksPerWeight =
                (outputPerNode + rowsPerTask - 1) / rowsPerTask;
            size_t tasksPerNode = (size_t)group.experts.size() *
                                  weightsPerExpert * chunksPerWeight;
            std::array<uintptr_t, 8> cacheKey = {
                reinterpret_cast<uintptr_t>(firstWeights),
                reinterpret_cast<uintptr_t>(secondWeights),
                (uintptr_t)numaConfig->numaCnt,
                (uintptr_t)(int)group.inputType,
                (uintptr_t)inputDimension, (uintptr_t)outputDimension,
                (uintptr_t)weightsPerExpert, (uintptr_t)rowsPerTask
            };
            auto cacheIt = workspace.gemmTaskCaches.find(cacheKey);
            if (cacheIt == workspace.gemmTaskCaches.end()) {
                cacheIt = workspace.gemmTaskCaches.emplace(
                    cacheKey, KimiK3NumaGemmTaskCache()).first;
            }
            KimiK3NumaGemmTaskCache *taskCache = &cacheIt->second;
            AssertInFastLLM(
                (int)workspace.activeConvertedInputs.size() ==
                    numaConfig->numaCnt,
                "KimiK3 NUMA GEMM has no node-local activation table.");
            taskCache->invocation.convertedInputs =
                &workspace.activeConvertedInputs;
            taskCache->invocation.routeCounts = &routeCounts;
            taskCache->invocation.expertLineOffsets = &expertLineOffsets;
            taskCache->invocation.convertedLineOffsets =
                &convertedLineOffsets;
            taskCache->invocation.firstOutput = firstOutput;
            taskCache->invocation.secondOutput = secondOutput;
            taskCache->invocation.inputRowBytes = GetDataBytes(
                group.inputType, 1, inputDimension);

            int maximumExpert = *std::max_element(
                group.experts.begin(), group.experts.end());
            if ((int)taskCache->expertPlans.size() <= maximumExpert) {
                taskCache->expertPlans.resize(maximumExpert + 1);
            }
            taskCache->pointers.resize(numaConfig->numaCnt);
            for (int node = 0; node < numaConfig->numaCnt; node++) {
                auto &pointers = taskCache->pointers[node];
                pointers.clear();
                pointers.reserve(tasksPerNode);
            }
            for (int expert : group.experts) {
                Data *firstWeight = firstWeights[expert];
                Data *secondWeight = secondWeights == nullptr ? nullptr :
                    secondWeights[expert];
                AssertInFastLLM(
                    IsKimiK3NumaWeight(firstWeight) &&
                    firstWeight->dims[0] == outputDimension &&
                    firstWeight->dims[1] == inputDimension &&
                    (secondWeight == nullptr ||
                     (IsKimiK3NumaWeight(secondWeight) &&
                      secondWeight->dims[0] == outputDimension &&
                      secondWeight->dims[1] == inputDimension)),
                    "KimiK3 NUMA expert weight is not registered or has an "
                    "incompatible shape.");
                auto &plan = taskCache->expertPlans[expert];
                bool planHit =
                    plan != nullptr && plan->firstWeight == firstWeight &&
                    plan->secondWeight == secondWeight &&
                    plan->firstWeightType == firstWeight->GetDataType() &&
                    (secondWeight == nullptr ||
                     plan->secondWeightType == secondWeight->GetDataType()) &&
                    (int)plan->storage.size() == numaConfig->numaCnt;
                if (planHit) {
                    for (int node = 0; node < numaConfig->numaCnt; node++) {
                        planHit = planHit &&
                            plan->firstWeightData[node] ==
                                firstWeight->numasData[node] &&
                            (secondWeight == nullptr ||
                             plan->secondWeightData[node] ==
                                secondWeight->numasData[node]);
                    }
                }
                if (!planHit) {
                    plan = std::make_unique<KimiK3NumaExpertGemmTaskPlan>();
                    plan->firstWeight = firstWeight;
                    plan->secondWeight = secondWeight;
                    plan->firstWeightType = firstWeight->GetDataType();
                    plan->secondWeightType = secondWeight == nullptr ?
                        DataType::FLOAT32 : secondWeight->GetDataType();
                    plan->firstWeightData = firstWeight->numasData;
                    if (secondWeight != nullptr) {
                        plan->secondWeightData = secondWeight->numasData;
                    }
                    plan->storage.resize(numaConfig->numaCnt);
                    for (int node = 0; node < numaConfig->numaCnt; node++) {
                        auto &storage = plan->storage[node];
                        storage.reserve(
                            (size_t)weightsPerExpert * chunksPerWeight);
                        const int globalOutputOffset =
                            node * outputPerNode;
                        auto appendWeight = [&](Data *weight,
                                                int outputSlot) {
                            for (int row = 0; row < outputPerNode;
                                 row += rowsPerTask) {
                                storage.emplace_back(
                                    &taskCache->invocation,
                                    weight->numasData[node], group.inputType,
                                    weight->GetDataType(), node, expert,
                                    outputSlot, inputDimension,
                                    outputDimension, row,
                                    std::min(row + rowsPerTask,
                                             outputPerNode),
                                    globalOutputOffset);
                            }
                        };
                        appendWeight(firstWeight, 0);
                        if (secondWeight != nullptr) {
                            appendWeight(secondWeight, 1);
                        }
                    }
                }
                for (int node = 0; node < numaConfig->numaCnt; node++) {
                    for (MultiThreadKimiK3NumaCachedGemmOp &task :
                         plan->storage[node]) {
                        taskCache->pointers[node].push_back(&task);
                    }
                }
            }
            for (const auto &pointers : taskCache->pointers) {
                AssertInFastLLM(
                    pointers.size() == tasksPerNode,
                    "KimiK3 NUMA GEMM task plan shape mismatch.");
            }
            ScheduleDeepSeekV4NumasMoeTasks(
                taskCache->pointers, false);
        }

        void RunKimiK3NumaActivation(
                int totalLines, int intermediateDimension,
                float beta, float linearBeta,
                KimiK3NumaWorkspace &workspace) {
            NumaConfig *numaConfig = GetNumaConfig();
            workspace.activationTaskStorage.resize(numaConfig->numaCnt);
            workspace.taskPointers.resize(numaConfig->numaCnt);
            int columnsPerNode =
                (intermediateDimension + numaConfig->numaCnt - 1) /
                numaConfig->numaCnt;
            const int columnsPerTask = 512;
            for (int node = 0; node < numaConfig->numaCnt; node++) {
                auto &storage = workspace.activationTaskStorage[node];
                auto &pointers = workspace.taskPointers[node];
                storage.clear();
                pointers.clear();
                int nodeStart = std::min(
                    node * columnsPerNode, intermediateDimension);
                int nodeEnd = std::min(
                    nodeStart + columnsPerNode, intermediateDimension);
                int chunks =
                    (nodeEnd - nodeStart + columnsPerTask - 1) /
                    columnsPerTask;
                storage.reserve((size_t)totalLines * chunks);
                pointers.reserve((size_t)totalLines * chunks);
                for (int line = 0; line < totalLines; line++) {
                    for (int column = nodeStart; column < nodeEnd;
                         column += columnsPerTask) {
                        size_t start =
                            (size_t)line * intermediateDimension + column;
                        size_t end =
                            (size_t)line * intermediateDimension +
                            std::min(column + columnsPerTask, nodeEnd);
                        storage.emplace_back(
                            workspace.gateOutput.data(),
                            workspace.upOutput.data(),
                            workspace.activated.data(),
                            start, end, beta, linearBeta);
                    }
                }
                for (MultiThreadKimiK3NumaActivationOp &task : storage) {
                    pointers.push_back(&task);
                }
            }
            ScheduleDeepSeekV4NumasMoeTasks(
                workspace.taskPointers, false);
        }

        void RunKimiK3NumaFusedReduce(
                const float *downOutput,
                const std::vector<int> &routedLineOffsets,
                const float *scoreData, int tokens, int topk,
                int outputDimension, uint16_t *outputData,
                KimiK3NumaWorkspace &workspace) {
            NumaConfig *numaConfig = GetNumaConfig();
            AssertInFastLLM(
                outputDimension % numaConfig->numaCnt == 0 &&
                routedLineOffsets.size() == (size_t)tokens * topk,
                "KimiK3 fused reduce has an incompatible NUMA layout.");
            const int columnsPerNode =
                outputDimension / numaConfig->numaCnt;
            constexpr int columnsPerTask = 256;
            workspace.reduceTaskStorage.resize(numaConfig->numaCnt);
            workspace.taskPointers.resize(numaConfig->numaCnt);
            for (int node = 0; node < numaConfig->numaCnt; node++) {
                auto &storage = workspace.reduceTaskStorage[node];
                auto &pointers = workspace.taskPointers[node];
                storage.clear();
                pointers.clear();
                const int nodeStart = node * columnsPerNode;
                const int nodeEnd = nodeStart + columnsPerNode;
                const int chunks =
                    (columnsPerNode + columnsPerTask - 1) /
                    columnsPerTask;
                storage.reserve((size_t)tokens * chunks);
                pointers.reserve((size_t)tokens * chunks);
                for (int token = 0; token < tokens; token++) {
                    for (int start = nodeStart; start < nodeEnd;
                         start += columnsPerTask) {
                        storage.emplace_back(
                            downOutput, routedLineOffsets.data(), scoreData,
                            outputData, token, topk, outputDimension, start,
                            std::min(start + columnsPerTask, nodeEnd));
                    }
                }
                for (MultiThreadKimiK3NumaFusedReduceOp &task : storage) {
                    pointers.push_back(&task);
                }
            }
            ScheduleDeepSeekV4NumasMoeTasks(
                workspace.taskPointers, false);
        }

    }

    extern void DoCudaKimiK3RoutedExpertsFromCPU(
        Data &input, Data &output,
        const int32_t *indexData, const float *scoreData, int topk,
        Data **w1s, Data **w2s, Data **w3s, int expertCount,
        float beta, float linearBeta, bool setZero,
        const std::unordered_set<int> &experts);

    static bool CanUseCudaKimiK3Prefill(
            DataType inputType, const std::vector<int> &activeExperts,
            Data **w1s, Data **w2s, Data **w3s) {
#ifndef USE_CUDA
        return false;
#else
        for (int expert : activeExperts) {
            Data *weights[] = {w1s[expert], w2s[expert], w3s[expert]};
            for (Data *weight : weights) {
                if (weight == nullptr ||
                    !IsCudaLinearDataTypeSupported(
                        inputType, weight->dataType, DataType::FLOAT32)) {
                    static std::once_flag warningOnce;
                    std::call_once(warningOnce, [inputType, weight]() {
                        const std::string weightType = weight == nullptr ?
                            "null" : GetDataTypeName(weight->dataType);
                        printf("[Fastllm] KimiK3 NUMA GPU prefill disabled: "
                               "CUDA Linear does not support %s activations "
                               "with %s expert weights.\n",
                               GetDataTypeName(inputType).c_str(),
                               weightType.c_str());
                    });
                    return false;
                }
            }
        }
        return true;
#endif
    }

    bool NumasKimiK3RoutedExpertsOp::CanRun(
            const std::string &, const DataDict &datas,
            const FloatDict &, const IntDict &intParams) {
        auto inputIt = datas.find("input");
        auto weightsIt = datas.find("w1s");
        auto countIt = intParams.find("experts___batch");
        if (inputIt == datas.end() || weightsIt == datas.end() ||
            countIt == intParams.end() || countIt->second <= 0 ||
            inputIt->second == nullptr ||
            inputIt->second->dataType != DataType::BFLOAT16) {
            return false;
        }
        Data **weights = (Data**)weightsIt->second;
        return IsKimiK3NumaWeight(weights[0]);
    }

    void NumasKimiK3RoutedExpertsOp::Reshape(
            const std::string &, const DataDict &datas,
            const FloatDict &, const IntDict &) {
        Data &input = *datas.find("input")->second;
        Data &output = *datas.find("output")->second;
        output.dataType = DataType::BFLOAT16;
        output.Resize(input.dims);
    }

    void NumasKimiK3RoutedExpertsOp::Run(
            const std::string &, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *datas.find("input")->second;
        Data &index = *datas.find("index")->second;
        Data &score = *datas.find("score")->second;
        Data &output = *datas.find("output")->second;
        Data **w1s = (Data**)datas.find("w1s")->second;
        Data **w2s = (Data**)datas.find("w2s")->second;
        Data **w3s = (Data**)datas.find("w3s")->second;
        int expertCount = intParams.find("experts___batch")->second;
        float beta = floatParams.find("beta") == floatParams.end() ?
                     1.0f : floatParams.find("beta")->second;
        float linearBeta =
            floatParams.find("linearBeta") == floatParams.end() ?
            0.0f : floatParams.find("linearBeta")->second;

        AssertInFastLLM(
            input.dataType == DataType::BFLOAT16 &&
            input.dims.size() == 2 && index.dims.size() == 2 &&
            score.dims == index.dims &&
            index.dataType == DataType::INT32 &&
            score.dataType == DataType::FLOAT32 &&
            index.dims[0] == input.dims[0] && expertCount > 0 &&
            beta > 0.0f,
            "KimiK3 NUMA routed-expert input/index/score mismatch.");
        int tokens = input.dims[0];
        int inputDimension = input.dims[1];
        int topk = index.dims[1];
        AssertInFastLLM(
            IsKimiK3NumaWeight(w1s[0]) &&
            IsKimiK3NumaWeight(w2s[0]) &&
            IsKimiK3NumaWeight(w3s[0]),
            "KimiK3 NUMA routed experts require warmup-registered Linear "
            "weights.");
        int intermediateDimension = w1s[0]->dims[0];
        int outputDimension = w2s[0]->dims[0];
        AssertInFastLLM(
            w1s[0]->dims[1] == inputDimension &&
            w3s[0]->dims == w1s[0]->dims &&
            w2s[0]->dims[1] == intermediateDimension &&
            outputDimension == inputDimension,
            "KimiK3 NUMA routed-expert weight shape mismatch.");

        KimiK3NumaWorkspace &workspace = GetKimiK3NumaWorkspace();
        const int32_t *indexData = nullptr;
        const float *scoreData = nullptr;
        const uint16_t *inputData = nullptr;
        bool returnOutputToCuda = false;
        int cudaDeviceId = -1;
#ifdef USE_CUDA
        void *inputCopyStream = nullptr;
        bool inputCopyPending = false;
        const bool inputOnCuda =
            input.dataDevice == DataDevice::CUDA && input.cudaData != nullptr;
        const bool indexOnCuda =
            index.dataDevice == DataDevice::CUDA && index.cudaData != nullptr;
        const bool scoreOnCuda =
            score.dataDevice == DataDevice::CUDA && score.cudaData != nullptr;
        if (inputOnCuda || indexOnCuda || scoreOnCuda) {
            auto cudaId = [](const Data &data) {
                return data.dataDeviceIds.empty() ? FastllmCudaGetDevice() :
                       data.dataDeviceIds[0];
            };
            cudaDeviceId = inputOnCuda ? cudaId(input) :
                           (indexOnCuda ? cudaId(index) : cudaId(score));
            AssertInFastLLM(
                (!inputOnCuda || cudaId(input) == cudaDeviceId) &&
                (!indexOnCuda || cudaId(index) == cudaDeviceId) &&
                (!scoreOnCuda || cudaId(score) == cudaDeviceId) &&
                !input.multiDeviceData && !index.multiDeviceData &&
                !score.multiDeviceData,
                "KimiK3 NUMA CUDA staging requires single-device inputs.");
            FastllmCudaSetDevice(cudaDeviceId);

            const size_t inputBytes = input.GetBytes();
            const size_t indexBytes = index.GetBytes();
            const size_t scoreBytes = score.GetBytes();
            uint8_t *staging = workspace.EnsurePinnedInput(
                inputBytes + indexBytes + scoreBytes);
            uint8_t *inputHost = staging;
            uint8_t *indexHost = inputHost + inputBytes;
            uint8_t *scoreHost = indexHost + indexBytes;
            inputCopyStream = workspace.EnsureInputCopyStream(cudaDeviceId);
            void *routeCopyStream =
                workspace.EnsureRouteCopyStream(cudaDeviceId);
            void *sourceReadyEvent = FastllmCudaEventCreate();
            FastllmCudaEventRecordCurrentThread(sourceReadyEvent);
            FastllmCudaStreamWaitEvent(
                inputCopyStream, sourceReadyEvent);
            FastllmCudaStreamWaitEvent(
                routeCopyStream, sourceReadyEvent);
            auto stage = [](const Data &data, bool onCuda,
                            uint8_t *destination, size_t bytes,
                            void *stream) {
                if (onCuda) {
                    AssertInFastLLM(
                        FastllmCudaCopyFromDeviceToPinnedHostAsync(
                            destination, data.cudaData, bytes, stream),
                        "KimiK3 NUMA failed to enqueue a CUDA D2H copy.");
                } else {
                    AssertInFastLLM(
                        data.cpuData != nullptr,
                        "KimiK3 NUMA input has no readable host data.");
                    memcpy(destination, data.cpuData, bytes);
                }
            };
            stage(input, inputOnCuda, inputHost, inputBytes,
                  inputCopyStream);
            stage(index, indexOnCuda, indexHost, indexBytes,
                  routeCopyStream);
            stage(score, scoreOnCuda, scoreHost, scoreBytes,
                  routeCopyStream);
            FastllmCudaEventDestroy(sourceReadyEvent);
            if (indexOnCuda || scoreOnCuda) {
                FastllmCudaStreamSynchronize(routeCopyStream);
            }
            inputCopyPending = inputOnCuda;
            inputData = (const uint16_t*)inputHost;
            indexData = (const int32_t*)indexHost;
            scoreData = (const float*)scoreHost;
            returnOutputToCuda = inputOnCuda;
        }
#endif
        if (inputData == nullptr) {
            inputData = (const uint16_t*)input.cpuData;
        }
        if (indexData == nullptr) {
            indexData = (const int32_t*)index.cpuData;
        }
        if (scoreData == nullptr) {
            scoreData = (const float*)score.cpuData;
        }
        AssertInFastLLM(
            inputData != nullptr && indexData != nullptr &&
            scoreData != nullptr,
            "KimiK3 NUMA routed experts require readable activations and "
            "routing tensors.");
        workspace.routes.resize(expertCount);
        for (auto &expertRoutes : workspace.routes) {
            expertRoutes.clear();
        }
        for (int token = 0; token < tokens; token++) {
            for (int slot = 0; slot < topk; slot++) {
                int expert = indexData[(size_t)token * topk + slot];
                AssertInFastLLM(
                    expert >= 0 && expert < expertCount,
                    "KimiK3 NUMA routed-expert index is out of range.");
                workspace.routes[expert].emplace_back(token, slot);
            }
        }
        std::vector<int> routedExperts;
        workspace.routeCounts.assign(expertCount, 0);
        for (int expert = 0; expert < expertCount; expert++) {
            int routeCount = (int)workspace.routes[expert].size();
            workspace.routeCounts[expert] = routeCount;
            if (routeCount == 0) {
                continue;
            }
            routedExperts.push_back(expert);
        }
        AssertInFastLLM(
            std::accumulate(
                workspace.routeCounts.begin(),
                workspace.routeCounts.end(), 0) == tokens * topk,
            "KimiK3 NUMA routed-expert route table is incomplete.");

        for (int expert : routedExperts) {
            Data &w1 = *w1s[expert];
            Data &w2 = *w2s[expert];
            Data &w3 = *w3s[expert];
            AssertInFastLLM(
                IsKimiK3NumaWeight(&w1) &&
                IsKimiK3NumaWeight(&w2) &&
                IsKimiK3NumaWeight(&w3) &&
                w1.dims == w1s[0]->dims &&
                w2.dims == w2s[0]->dims &&
                w3.dims == w3s[0]->dims,
                "KimiK3 NUMA routed experts require registered weights with "
                "uniform shapes.");
        }

        workspace.activeExperts = routedExperts;
#ifdef USE_CUDA
        std::unordered_set<int> gpuExperts;
        std::thread gpuThread;
        bool hybridGpuPrefill = false;
        bool gpuPrefill = inputOnCuda &&
            MoeEnvConfig::GetInstance().GetGpuPrefill() &&
            CanUseCudaKimiK3Prefill(
                input.dataType, routedExperts, w1s, w2s, w3s);
        if (gpuPrefill) {
            const int expertLimit =
                MoeEnvConfig::GetInstance().GetExpertLimit();
            workspace.activeExperts.clear();
            for (int expert : routedExperts) {
                if (workspace.routeCounts[expert] < expertLimit) {
                    workspace.activeExperts.push_back(expert);
                } else {
                    gpuExperts.insert(expert);
                }
            }
        }
        if (!gpuExperts.empty() && workspace.activeExperts.empty()) {
            DoCudaKimiK3RoutedExpertsFromCPU(
                input, output, indexData, scoreData, topk,
                w1s, w2s, w3s, expertCount, beta, linearBeta, true,
                gpuExperts);
            if (inputCopyPending) {
                FastllmCudaStreamSynchronize(inputCopyStream);
            }
            output.dataDevice = DataDevice::CUDA;
            output.dataDeviceIds = {cudaDeviceId};
            return;
        }
        if (!gpuExperts.empty()) {
            hybridGpuPrefill = true;
            gpuThread = std::thread([
                    &, cudaDeviceId, gpuExperts]() {
                FastllmCudaSetDevice(cudaDeviceId);
                DoCudaKimiK3RoutedExpertsFromCPU(
                    input, output, indexData, scoreData, topk,
                    w1s, w2s, w3s, expertCount, beta, linearBeta, true,
                    gpuExperts);
            });
        }
        if (inputCopyPending) {
            FastllmCudaStreamSynchronize(inputCopyStream);
        }
#endif

        workspace.expertLineOffsets.assign(expertCount, -1);
        int totalLines = 0;
        for (int expert : workspace.activeExperts) {
            workspace.expertLineOffsets[expert] = totalLines;
            totalLines += workspace.routeCounts[expert];
        }
        workspace.routedLineOffsets.assign((size_t)tokens * topk, -1);
        workspace.expertLineTokens.assign(totalLines, -1);
        for (int expert : workspace.activeExperts) {
            const int lineOffset = workspace.expertLineOffsets[expert];
            const int routeCount = workspace.routeCounts[expert];
            for (int row = 0; row < routeCount; row++) {
                const int token = workspace.routes[expert][row].first;
                const int slot = workspace.routes[expert][row].second;
                workspace.routedLineOffsets[(size_t)token * topk + slot] =
                    lineOffset + row;
                workspace.expertLineTokens[lineOffset + row] = token;
            }
        }
        AssertInFastLLM(
            std::find(workspace.expertLineTokens.begin(),
                      workspace.expertLineTokens.end(), -1) ==
                workspace.expertLineTokens.end(),
            "KimiK3 NUMA routed-expert line map is incomplete.");
        const size_t intermediateCount =
            (size_t)totalLines * intermediateDimension;
        const size_t downCount = (size_t)totalLines * outputDimension;
        // Group before allocating the gate staging table: direct Q8_K groups
        // gather from token-major BF16 while quantizing into each NUMA node,
        // so they do not need the intermediate expert-major BF16 copy.
        std::vector<KimiK3NumaFormatGroup> gateGroups =
            BuildKimiK3NumaFormatGroups(
                workspace.activeExperts, workspace.routeCounts, w1s, w3s);
        std::vector<KimiK3NumaFormatGroup> downGroups =
            BuildKimiK3NumaFormatGroups(
                workspace.activeExperts, workspace.routeCounts, w2s,
                nullptr);
        bool gateNeedsBfloat16Gather = false;
        for (const KimiK3NumaFormatGroup &group : gateGroups) {
            gateNeedsBfloat16Gather = gateNeedsBfloat16Gather ||
                !IsDirectBFloat16Q8KType(group.inputType);
        }
        if (gateNeedsBfloat16Gather) {
            workspace.gatheredInput.resize(
                (size_t)totalLines * inputDimension);
        }
        workspace.gateOutput.resize(intermediateCount);
        workspace.upOutput.resize(intermediateCount);
        workspace.activated.resize(intermediateCount);
        workspace.downOutput.resize(downCount);
        workspace.convertedLineOffsets.assign(expertCount, -1);

        // Validate every active expert even when the fused gather/convert path
        // bypasses the shared expert-major table.
        for (int expert : workspace.activeExperts) {
            Data &w1 = *w1s[expert];
            Data &w2 = *w2s[expert];
            Data &w3 = *w3s[expert];
            AssertInFastLLM(
                IsKimiK3NumaWeight(&w1) &&
                IsKimiK3NumaWeight(&w2) &&
                IsKimiK3NumaWeight(&w3) &&
                w1.dims == w1s[0]->dims &&
                w2.dims == w2s[0]->dims &&
                w3.dims == w3s[0]->dims,
                "KimiK3 NUMA routed experts require registered weights with "
                "uniform shapes.");
        }

        // Generic formats retain a shared expert-major staging table.  The
        // direct Q8_K path gathers inside node-bound conversion tasks below.
        if (gateNeedsBfloat16Gather) {
            for (int expert : workspace.activeExperts) {
                int routeCount = workspace.routeCounts[expert];
                int lineOffset = workspace.expertLineOffsets[expert];
                for (int row = 0; row < routeCount; row++) {
                    int token = workspace.routes[expert][row].first;
                    const uint16_t *source = inputData +
                        (size_t)token * inputDimension;
                    uint16_t *destination =
                        workspace.gatheredInput.data() +
                        (size_t)(lineOffset + row) * inputDimension;
                    memcpy(destination, source,
                           (size_t)inputDimension * sizeof(uint16_t));
                }
            }
        }

        // MergeMOE-style format grouping: a model may use GGUF Q2/Q4,
        // NVFP4, FP8, INT8/INT4, or floating-point experts.  Each activation
        // representation is converted once for all matching active experts.
        for (const KimiK3NumaFormatGroup &group : gateGroups) {
            std::fill(workspace.convertedLineOffsets.begin(),
                      workspace.convertedLineOffsets.end(), -1);
            const bool useDirectTokenSource =
                IsDirectBFloat16Q8KType(group.inputType);
            ConvertKimiK3NumaGroupInput(
                group, workspace.routeCounts,
                workspace.expertLineOffsets,
                useDirectTokenSource ? inputData :
                    workspace.gatheredInput.data(),
                inputDimension,
                useDirectTokenSource ? &workspace.expertLineTokens :
                    nullptr,
                workspace.convertedLineOffsets, workspace);
            RunKimiK3NumaGemmGroup(
                group, workspace.routeCounts, workspace.expertLineOffsets,
                workspace.convertedLineOffsets, inputDimension,
                intermediateDimension, w1s,
                workspace.gateOutput.data(), w3s,
                workspace.upOutput.data(), workspace);
        }

        RunKimiK3NumaActivation(
            totalLines, intermediateDimension, beta, linearBeta, workspace);

        for (const KimiK3NumaFormatGroup &group : downGroups) {
            std::fill(workspace.convertedLineOffsets.begin(),
                      workspace.convertedLineOffsets.end(), -1);
            ConvertKimiK3NumaGroupInput(
                group, workspace.routeCounts,
                workspace.expertLineOffsets,
                workspace.activated.data(), intermediateDimension,
                nullptr, workspace.convertedLineOffsets, workspace);
            RunKimiK3NumaGemmGroup(
                group, workspace.routeCounts, workspace.expertLineOffsets,
                workspace.convertedLineOffsets, intermediateDimension,
                outputDimension, w2s, workspace.downOutput.data(),
                nullptr, nullptr, workspace);
        }

        uint16_t *outputData = nullptr;
#ifdef USE_CUDA
        if (returnOutputToCuda) {
            outputData = (uint16_t*)workspace.EnsurePinnedOutput(
                output.GetBytes());
        } else
#endif
        {
            output.Allocate(false);
            outputData = (uint16_t*)output.cpuData;
        }

        RunKimiK3NumaFusedReduce(
            workspace.downOutput.data(), workspace.routedLineOffsets,
            scoreData, tokens, topk, outputDimension, outputData,
            workspace);
#ifdef USE_CUDA
        if (returnOutputToCuda) {
            if (hybridGpuPrefill) {
                FastllmCudaSetDevice(cudaDeviceId);
                void *cpuOutputStaging =
                    workspace.EnsureGpuOutputStaging(
                        output.GetBytes(), cudaDeviceId);
                void *outputCopyStream =
                    workspace.EnsureOutputCopyStream(cudaDeviceId);
                FastllmCudaCopyFromPinnedHostToDeviceAsync(
                    cpuOutputStaging, outputData, output.GetBytes(),
                    outputCopyStream);
                gpuThread.join();
                FastllmCudaStreamSynchronize(outputCopyStream);
                Data gpuOutputAlias(
                    output.dataType, output.dims, DataDevice::CUDA,
                    output.cudaData);
                Data cpuOutputAlias(
                    output.dataType, output.dims, DataDevice::CUDA,
                    cpuOutputStaging);
                FastllmCudaAddTo(gpuOutputAlias, cpuOutputAlias, 1.0f);
                output.dataDevice = DataDevice::CUDA;
                output.dataDeviceIds = {cudaDeviceId};
            } else {
                output.ToDevice(
                    DataDevice::CUDA, std::vector<int>{cudaDeviceId}, false);
                output.Allocate(false);
                AssertInFastLLM(
                    output.cudaData != nullptr &&
                    FastllmCudaCopyFromPinnedHostToDeviceAsyncCurrentThread(
                        output.cudaData, outputData, output.GetBytes()),
                    "KimiK3 NUMA failed to enqueue a CUDA H2D copy.");
            }
        }
#endif
    }

    extern void DoCudaMergeMOEFromCPU (Data &input, Data &output, Data &index, Data &score, Data &w1, Data &w2, Data &w3, 
        Data **weights, Data **biass, float sharedScale, bool setZero, const std::unordered_set<int> &experts,
        bool isCrossSwiglu, MoeGateType gateType = MoeGateSwiglu,
        bool deepSeekV4Mode = false, float swigluLimit = 0.0f);
    extern void ReduceSumFromCPU(Data &output);
    void DoNumasMergeMOEOnCPU(
        Data &input, Data &output,
        Data &index, Data &score,
        Data **weights, Data **biass,
        float sharedScale,
        int weightsBatch, int topk,
        const std::unordered_set<int> &cpuExperts,
        FastllmMoeDataManagerNumas &fastllmMoeDataManagerNumas,
        uint8_t *cpuOutputBuffer,
        float swigluLimit = 0.0f,
        bool deepSeekV4Mode = false
    );

    struct MoeBenchmarkShapeKey {
        int inputDim = 0;
        int interDim = 0;
        int outputDim = 0;
        int topk = 0;
        int inputType = 0;
        int outputType = 0;
        int gpuId = 0;

        bool operator==(const MoeBenchmarkShapeKey &other) const {
            return inputDim == other.inputDim &&
                   interDim == other.interDim &&
                   outputDim == other.outputDim &&
                   topk == other.topk &&
                   inputType == other.inputType &&
                   outputType == other.outputType &&
                   gpuId == other.gpuId;
        }
    };

    struct MoeBenchmarkShapeKeyHasher {
        size_t operator()(const MoeBenchmarkShapeKey &key) const {
            size_t h = std::hash<int>()(key.inputDim);
            h = h * 131 + std::hash<int>()(key.interDim);
            h = h * 131 + std::hash<int>()(key.outputDim);
            h = h * 131 + std::hash<int>()(key.topk);
            h = h * 131 + std::hash<int>()(key.inputType);
            h = h * 131 + std::hash<int>()(key.outputType);
            h = h * 131 + std::hash<int>()(key.gpuId);
            return h;
        }
    };

    class MoeExpertSpeedEstimator {
    public:
        static MoeExpertSpeedEstimator& GetInstance() {
            static MoeExpertSpeedEstimator instance;
            return instance;
        }

        int GetDynamicExpertLimit(
            Data &input, Data &output, Data &w1, Data &w2, Data &w3,
            Data **weights, Data **biass, int weightsBatch, int topk, float sharedScale,
            const std::vector<std::vector<std::pair<int, float> > > &expertTasks,
            int defaultExpertLimit, int gpuCount
        ) {
#ifdef USE_CUDA
            if (input.cudaData == nullptr || input.cpuData == nullptr) {
                return defaultExpertLimit;
            }
            gpuCount = std::max(1, gpuCount);

            int m = weightsBatch / 2 - 1;
            int maxTaskSize = 0;
            for (int e = 0; e < (int)expertTasks.size(); e++) {
                if (e * 2 >= weightsBatch || weights[e * 2] == nullptr) {
                    continue;
                }
                maxTaskSize = std::max(maxTaskSize, (int)expertTasks[e].size());
            }
            if (maxTaskSize <= 0) {
                return defaultExpertLimit;
            }

            int adaptiveN = std::min(maxTaskSize, hardMaxBenchmarkN);
            adaptiveN = std::min(adaptiveN, input.dims[0]);
            if (adaptiveN <= 0) {
                return defaultExpertLimit;
            }

            MoeBenchmarkShapeKey key;
            key.inputDim = input.dims[1];
            key.interDim = weights[2]->dims[0] / 2;
            key.outputDim = output.dims[1];
            key.topk = topk;
            key.inputType = (int)input.dataType;
            key.outputType = (int)output.dataType;
            key.gpuId = FastllmCudaGetDevice();

            std::lock_guard<std::mutex> lock(profileLocker);
            auto &profile = profiles[key];
            if (!profile.initialized || profile.maxN < adaptiveN) {
                if (!BuildProfile(profile, input, output, w1, w2, w3, weights, biass,
                                  weightsBatch, sharedScale, adaptiveN, m)) {
                    return defaultExpertLimit;
                }
                // printf("MoE dynamic expertLimit benchmark initialized: N=%d, bs=%d, topk=%d\n", profile.maxN, input.dims[0], topk);
            }
            // Precompute per-expert interpolated times to avoid repeated map lookups
            int numExperts = (int)expertTasks.size();
            std::vector<int> expertSz(numExperts);
            std::vector<double> expertCpu(numExperts, 0.0);
            std::vector<double> expertGpu(numExperts, 0.0);
            std::vector<bool> expertValid(numExperts, false);
            for (int e = 0; e < numExperts; e++) {
                if (e * 2 >= weightsBatch || weights[e * 2] == nullptr) {
                    continue;
                }
                expertValid[e] = true;
                expertSz[e] = (int)expertTasks[e].size();
                expertCpu[e] = InterpolateFromMap(profile.cpuTimeUs, expertSz[e]);
                expertGpu[e] = InterpolateFromMap(profile.gpuTimeUs, expertSz[e]);
            }

            int bestLimit = defaultExpertLimit;
            double bestMetric = DBL_MAX;
            double bestCpuTime = 0.0;
            double bestGpuTime = 0.0;
            int activeExpertCount = 0;
            for (int e = 0; e < numExperts; e++) {
                if (expertValid[e] && expertSz[e] > 0) {
                    activeExpertCount++;
                }
            }
            maxTaskSize = std::min(maxTaskSize, defaultExpertLimit);
            for (int t = 1; t <= maxTaskSize + 1; t++) {
                double cpuTime = 0.0;
                std::vector<std::pair<double, int> > gpuJobs;
                for (int e = 0; e < numExperts; e++) {
                    if (!expertValid[e]) {
                        continue;
                    }
                    if (expertSz[e] < t) {
                        cpuTime += expertCpu[e];
                    } else {
                        gpuJobs.push_back({expertGpu[e], e});
                    }
                }
                if (gpuCount > 1 && activeExpertCount >= gpuCount &&
                    (int)gpuJobs.size() < gpuCount) {
                    continue;
                }
                std::sort(
                    gpuJobs.begin(), gpuJobs.end(),
                    [](const std::pair<double, int> &a,
                       const std::pair<double, int> &b) {
                        if (a.first != b.first) {
                            return a.first > b.first;
                        }
                        return a.second < b.second;
                    });
                std::vector<double> gpuLoads(gpuCount, 0.0);
                for (const auto &job : gpuJobs) {
                    auto loadIt = std::min_element(
                        gpuLoads.begin(), gpuLoads.end());
                    *loadIt += job.first;
                }
                double gpuTime = gpuLoads.empty() ? 0.0 :
                    *std::max_element(gpuLoads.begin(), gpuLoads.end());
                double metric = gpuCount == 1 ?
                    std::fabs(cpuTime - gpuTime) :
                    std::max(cpuTime, gpuTime);
                if (metric < bestMetric) {
                    bestMetric = metric;
                    bestLimit = t;
                    bestCpuTime = cpuTime;
                    bestGpuTime = gpuTime;
                }
            }
            if (profile.lastPrintedLimit != bestLimit) {
                // printf("MoE dynamic expertLimit=%d, predict cpu=%.2fus gpu=%.2fus\n", bestLimit, bestCpuTime, bestGpuTime);
                profile.lastPrintedLimit = bestLimit;
            }
            return bestLimit;
#else
            (void)input;
            (void)output;
            (void)w1;
            (void)w2;
            (void)w3;
            (void)weights;
            (void)biass;
            (void)weightsBatch;
            (void)topk;
            (void)sharedScale;
            (void)expertTasks;
            (void)gpuCount;
            return defaultExpertLimit;
#endif
        }

    private:
        struct BenchmarkProfile {
            int maxN = 0;
            bool initialized = false;
            int lastPrintedLimit = -1;
            std::map<int, double> cpuTimeUs;
            std::map<int, double> gpuTimeUs;
        };

        std::unordered_map<MoeBenchmarkShapeKey, BenchmarkProfile, MoeBenchmarkShapeKeyHasher> profiles;
        std::mutex profileLocker;
        const int warmupRounds = 2;
        const int measureRounds = 8;
        const int hardMaxBenchmarkN = 512;

        static int GetStepForN(int n, bool isCpu) {
            int baseStep;
            if (n <= 16) {
                baseStep = 1;
            } else if (n <= 64) {
                baseStep = 4;
            } else if (n <= 256) {
                baseStep = 16;
            } else if (n <= 1024) {
                baseStep = 64;
            } else {
                baseStep = 128;
            }
            return isCpu ? std::max(baseStep * 2, 2) : baseStep;
        }

        static std::vector<int> GenerateSamplePoints(int maxN, bool isCpu) {
            std::vector<int> points;
            int n = 1;
            while (n <= maxN) {
                points.push_back(n);
                int step = GetStepForN(n, isCpu);
                n += step;
            }
            if (points.empty() || points.back() != maxN) {
                points.push_back(maxN);
            }
            return points;
        }

        static double InterpolateFromMap(const std::map<int, double> &curve, int size) {
            if (size <= 0 || curve.empty()) {
                return 0.0;
            }
            auto it = curve.find(size);
            if (it != curve.end()) {
                return it->second;
            }
            auto upper = curve.upper_bound(size);
            if (upper == curve.begin()) {
                return upper->second;
            }
            if (upper == curve.end()) {
                auto last = std::prev(upper);
                if (last == curve.begin()) {
                    return last->second;
                }
                auto prev2 = std::prev(last);
                double slope = (last->second - prev2->second) / (last->first - prev2->first);
                double ret = last->second + slope * (size - last->first);
                return std::max(ret, 0.0);
            }
            auto lower = std::prev(upper);
            double t = (double)(size - lower->first) / (upper->first - lower->first);
            return lower->second + t * (upper->second - lower->second);
        }

        static int PickBenchmarkExpert(Data **weights, int weightsBatch, int m) {
            for (int e = 1; e <= m; e++) {
                if (e * 2 < weightsBatch && weights[e * 2] != nullptr) {
                    return e;
                }
            }
            return -1;
        }

        bool BuildProfile(
            BenchmarkProfile &profile,
            Data &input, Data &output, Data &w1, Data &w2, Data &w3,
            Data **weights, Data **biass, int weightsBatch, float sharedScale,
            int adaptiveN, int m
        ) {
            if (input.dims.size() < 2 || output.dims.size() < 2 || input.cpuData == nullptr) {
                return false;
            }
            if (adaptiveN <= 0 || adaptiveN > input.dims[0]) {
                return false;
            }

            int benchExpert = PickBenchmarkExpert(weights, weightsBatch, m);
            if (benchExpert < 1) {
                return false;
            }

            int inputDim = input.dims[1];
            int outputDim = output.dims[1];
            std::unordered_set<int> cpuExperts = {benchExpert};
            std::unordered_set<int> gpuExperts = {benchExpert};
            FastllmMoeDataManagerNumas cpuBenchDataManager;

            Data benchIndex(DataType::INT32, {adaptiveN, 1});
            benchIndex.Allocate();
            Data benchScore(DataType::FLOAT32, {adaptiveN, 1});
            benchScore.Allocate();
            int32_t *benchIndexData = (int32_t*)benchIndex.cpuData;
            float *benchScoreData = (float*)benchScore.cpuData;
            for (int i = 0; i < adaptiveN; i++) {
                benchIndexData[i] = benchExpert - 1;
                benchScoreData[i] = 1.0f;
            }

            profile.maxN = adaptiveN;
            profile.cpuTimeUs.clear();
            profile.gpuTimeUs.clear();
            profile.cpuTimeUs[0] = 0.0;
            profile.gpuTimeUs[0] = 0.0;

            std::vector<int> cpuSamples = GenerateSamplePoints(adaptiveN, true);
            std::vector<int> gpuSamples = GenerateSamplePoints(adaptiveN, false);
            std::set<int> allSamples(cpuSamples.begin(), cpuSamples.end());
            allSamples.insert(gpuSamples.begin(), gpuSamples.end());
            std::set<int> cpuSampleSet(cpuSamples.begin(), cpuSamples.end());
            std::set<int> gpuSampleSet(gpuSamples.begin(), gpuSamples.end());

            for (auto it = allSamples.rbegin(); it != allSamples.rend(); ++it) {
                int k = *it;
                Data benchInput(input.dataType, {k, inputDim}, DataDevice::CPU, input.cpuData);
                benchInput.cudaData = input.cudaData;
                benchInput.cudaDataBorrowed = true;
                benchInput.dataDeviceIds = input.dataDeviceIds;
                Data benchIndexK(DataType::INT32, {k, 1}, DataDevice::CPU, benchIndex.cpuData);
                Data benchScoreK(DataType::FLOAT32, {k, 1}, DataDevice::CPU, benchScore.cpuData);

                if (cpuSampleSet.count(k)) {
                    Data cpuOutput(output.dataType, {k, outputDim});
                    cpuOutput.Allocate();
                    for (int i = 0; i < warmupRounds; i++) {
                        DoNumasMergeMOEOnCPU(
                            benchInput, cpuOutput, benchIndexK, benchScoreK, weights, biass,
                            sharedScale, weightsBatch, 1, cpuExperts, cpuBenchDataManager, nullptr
                        );
                    }
                    double cpuElapsed = 0.0;
                    for (int i = 0; i < measureRounds; i++) {
                        auto st = std::chrono::steady_clock::now();
                        DoNumasMergeMOEOnCPU(
                            benchInput, cpuOutput, benchIndexK, benchScoreK, weights, biass,
                            sharedScale, weightsBatch, 1, cpuExperts, cpuBenchDataManager, nullptr
                        );
                        auto ed = std::chrono::steady_clock::now();
                        cpuElapsed += std::chrono::duration<double, std::micro>(ed - st).count();
                    }
                    profile.cpuTimeUs[k] = cpuElapsed / measureRounds;
                }

                if (gpuSampleSet.count(k)) {
#ifdef USE_CUDA
                    Data gpuOutput(output.dataType, {k, outputDim});
                    gpuOutput.Allocate();
                    if (gpuOutput.cudaData == nullptr) {
                        gpuOutput.ToDevice(DataDevice::CUDA);
                        gpuOutput.ToDevice(DataDevice::CPU);
                    }
                    for (int i = 0; i < warmupRounds; i++) {
                        DoCudaMergeMOEFromCPU(
                            benchInput, gpuOutput, benchIndexK, benchScoreK, w1, w2, w3, weights, biass,
                            sharedScale, true, gpuExperts, true
                        );
#ifdef USE_CUDA
                        FastllmCudaStreamSynchronize(nullptr);
#endif
                    }
                    double gpuElapsed = 0.0;
                    for (int i = 0; i < measureRounds; i++) {
                        auto st = std::chrono::steady_clock::now();
                        DoCudaMergeMOEFromCPU(
                            benchInput, gpuOutput, benchIndexK, benchScoreK, w1, w2, w3, weights, biass,
                            sharedScale, true, gpuExperts, true
                        );
#ifdef USE_CUDA
                        FastllmCudaStreamSynchronize(nullptr);
#endif
                        auto ed = std::chrono::steady_clock::now();
                        gpuElapsed += std::chrono::duration<double, std::micro>(ed - st).count();
                    }
                    profile.gpuTimeUs[k] = gpuElapsed / measureRounds;
#else
                    profile.gpuTimeUs[k] = profile.cpuTimeUs.count(k) ? profile.cpuTimeUs[k] : 0.0;
#endif
                }
            }

            // printf("MoE benchmark: cpu samples=%d, gpu samples=%d\n", (int)profile.cpuTimeUs.size(), (int)profile.gpuTimeUs.size());
            profile.initialized = true;
            profile.lastPrintedLimit = -1;
            return true;
        }
    };

    void DoNumasMergeMOEOnCPU(
        Data &input, Data &output,
        Data &index, Data &score,
        Data **weights, Data **biass,
        float sharedScale,
        int weightsBatch, int topk,
        const std::unordered_set<int> &cpuExperts,
        FastllmMoeDataManagerNumas &fastllmMoeDataManagerNumas,
        uint8_t *cpuOutputBuffer,
        float swigluLimit,
        bool deepSeekV4Mode
    ) {
        int bs = input.dims[0];
        int m = weightsBatch / 2 - 1; // num experts
        int32_t *indexData = (int32_t*)index.cpuData;
        float *scoreData = (float*)score.cpuData;

        auto *pool = GetAlivePool();

        int dim = output.dims[1];
        int inputDim = input.dims[1];
        int interDim = weights[2]->dims[0] / 2;
        int outputDim = output.dims[1];

        std::vector <std::vector <std::pair <int, float> > > expertTasks; // expertTasks[i]代表专家i的task, expertTasks[i][j] = (第j个任务对应的行数， 权重)
        expertTasks.resize(m + 1);
        for (int b = 0; b < bs; b++) {
            expertTasks[0].push_back(std::make_pair(b, sharedScale));
            for (int j = 0; j < topk; j++) {
                int expertIdx = indexData[b * topk + j];
                float value = scoreData[b * topk + j];
                expertTasks[expertIdx + 1].push_back(std::make_pair(b, value));
            }
        }

        // GGUF MoE weights in the same layer do not necessarily use the same
        // quantization type.  In particular, mixed quant models commonly keep
        // the shared expert in Q8_0 while routed experts use Q4_K/Q5_K.  Their
        // dot kernels require different activation row formats (Q8_0 vs Q8_K),
        // so one converted input/down-input buffer cannot be shared between
        // those experts.  Split active CPU experts by both required activation
        // formats and accumulate the partial outputs.
        std::map<std::pair<int, int>, std::unordered_set<int> > expertTypeGroups;
        for (int e : cpuExperts) {
            if (e < 0 || e >= (int)expertTasks.size() || expertTasks[e].empty() ||
                weights[e * 2] == nullptr || weights[e * 2 + 1] == nullptr) {
                continue;
            }
            DataType gateActType = GetNumasLinearActDataType(weights[e * 2], bs);
            DataType downActType = GetNumasLinearActDataType(weights[e * 2 + 1], bs);
            expertTypeGroups[std::make_pair((int)gateActType, (int)downActType)].insert(e);
        }
        if (expertTypeGroups.empty()) {
            uint8_t *emptyOutput = cpuOutputBuffer != nullptr ? cpuOutputBuffer : output.cpuData;
            AssertInFastLLM(emptyOutput != nullptr,
                            "NumasMergeMOE has no writable CPU output.\n");
            memset(emptyOutput, 0, output.GetBytes());
            return;
        }
        if (expertTypeGroups.size() > 1) {
            bool firstGroup = true;
            for (auto &group : expertTypeGroups) {
                if (firstGroup) {
                    DoNumasMergeMOEOnCPU(
                        input, output, index, score, weights, biass,
                        sharedScale, weightsBatch, topk, group.second,
                        fastllmMoeDataManagerNumas, nullptr,
                        swigluLimit, deepSeekV4Mode
                    );
                    firstGroup = false;
                } else {
                    Data partialOutput(output.dataType, output.dims);
                    partialOutput.Allocate();
                    DoNumasMergeMOEOnCPU(
                        input, partialOutput, index, score, weights, biass,
                        sharedScale, weightsBatch, topk, group.second,
                        fastllmMoeDataManagerNumas, nullptr,
                        swigluLimit, deepSeekV4Mode
                    );
                    AddTo(output, partialOutput);
                }
            }
            if (cpuOutputBuffer != nullptr) {
                memcpy(cpuOutputBuffer, output.cpuData, output.GetBytes());
            }
            return;
        }

        int totalLines = 0;
        for (int e = 0; e < (int)expertTasks.size(); e++) {
            if (weights[e * 2] != nullptr && cpuExperts.count(e)) {
                totalLines += expertTasks[e].size();
            }
        }

        int representativeExpert = -1;
        for (int e : cpuExperts) {
            if (e >= 0 && e < (int)expertTasks.size() && !expertTasks[e].empty() &&
                weights[e * 2] != nullptr && weights[e * 2 + 1] != nullptr) {
                representativeExpert = e;
                break;
            }
        }
        AssertInFastLLM(representativeExpert >= 0,
                        "NumasMergeMOE has no active CPU expert.\n");
        DataType startDataType = GetNumasLinearActDataType(weights[representativeExpert * 2], bs);
        DataType downInputDataType = GetNumasLinearActDataType(weights[representativeExpert * 2 + 1], bs);

        // 从 fastllmMoeDataManagerNumas 获取缓存的 vector，并根据需要调整大小
        auto& realInput = fastllmMoeDataManagerNumas.realInput;
        auto& inputFloat32 = fastllmMoeDataManagerNumas.inputFloat32;
        auto& expandInput = fastllmMoeDataManagerNumas.expandInput;
        auto& gateUpOutput = fastllmMoeDataManagerNumas.gateUpOutput;
        auto& swigluOutput = fastllmMoeDataManagerNumas.swigluOutput;
        auto& downInput = fastllmMoeDataManagerNumas.downInput;
        auto& downOutput = fastllmMoeDataManagerNumas.downOutput;
        auto& reduceOutput = fastllmMoeDataManagerNumas.reduceOutput;

        int alignTotalLines = ((totalLines - 1) / 64 + 1) * 64;
        // 计算所需大小
        size_t realInputSize = GetDataBytes(startDataType, bs, inputDim);
        size_t inputFloat32Size = (size_t)bs * inputDim;
        size_t expandInputSize = GetDataBytes(startDataType, alignTotalLines, inputDim);
        size_t gateUpOutputSize = alignTotalLines * interDim * 2;
        size_t swigluOutputSize = alignTotalLines * interDim;
        size_t downInputSize = GetDataBytes(downInputDataType, alignTotalLines, interDim);
        size_t downOutputSize = alignTotalLines * outputDim;
        size_t reduceOutputSize = bs * outputDim;

        // 只在当前容量不足时才进行 resize
        if (realInput.size() < realInputSize) {
            realInput.resize(realInputSize);
        }
        if (input.dataType != DataType::FLOAT32 && inputFloat32.size() < inputFloat32Size) {
            inputFloat32.resize(inputFloat32Size);
        }
        if (expandInput.size() < expandInputSize) {
            expandInput.resize(expandInputSize);
        }
        if (gateUpOutput.size() < gateUpOutputSize) {
            gateUpOutput.resize(gateUpOutputSize);
        }
        if (swigluOutput.size() < swigluOutputSize) {
            swigluOutput.resize(swigluOutputSize);
        }
        if (downInput.size() < downInputSize) {
            downInput.resize(downInputSize);
        }
        if (downOutput.size() < downOutputSize) {
            downOutput.resize(downOutputSize);
        }
        // downOutput 不需要 fill 0：所有有效行都会被 down 阶段完整写入；
        if (reduceOutput.size() < reduceOutputSize) {
            reduceOutput.resize(reduceOutputSize);
        }

        // 0. input -> realInput（若 input 非 FLOAT32 则先转为 float32）
        if (input.dataType == startDataType && input.dataType != DataType::FLOAT32) {
            size_t bytes = GetDataBytes(startDataType, bs, inputDim);
            RunMultiThreadMemcpy(realInput.data(), (uint8_t*)input.cpuData, bytes, GetAlivePool());
        } else {
            const float *inputF32Ptr = nullptr;
            if (input.dataType == DataType::FLOAT32) {
                inputF32Ptr = (const float*)input.cpuData;
            } else {
                int inputCount = bs * inputDim;
                if (input.dataType == DataType::FLOAT16) {
                    Float16ToFloat32((uint16_t*)input.cpuData, inputFloat32.data(), inputCount);
                } else if (input.dataType == DataType::BFLOAT16) {
                    BFloat16ToFloat32((uint16_t*)input.cpuData, inputFloat32.data(), inputCount);
                } else {
                    ErrorInFastLLM("NumasMergeMOE: unsupported input dataType.\n");
                }
                inputF32Ptr = inputFloat32.data();
            }
            RunMultiThreadConvertFromFloat32(realInput.data(), startDataType, inputF32Ptr, bs, inputDim, GetAlivePool());
        }

        // 1. realInput -> expandInput
        std::vector <MultiThreadMemcpyMultiLinesTask> memcpyTasks;
        memcpyTasks.resize(totalLines);
        {
            uint8_t* realInputPtr = realInput.data();
            uint8_t* expandInputPtr = expandInput.data();
            int bytesPerLine = GetDataBytes(startDataType, 1, inputDim);

            // 预计算每个 expert 在 expandInput 中的起始偏移（跳过不在 cpuExperts 中的专家）
            std::vector<int> curPos(expertTasks.size(), -1);
            {
                int base = 0;
                for (int e = 0; e < (int)expertTasks.size(); e++) {
                    if (weights[e * 2] != nullptr && cpuExperts.count(e)) {
                        curPos[e] = base;
                        base += expertTasks[e].size();
                    }
                }
            }

            // 按 token 顺序枚举：先 shared expert (expert 0)，再按 b * topk + j 顺序
            // 跳过不在 cpuExperts 中的专家
            int idx = 0;
            for (int b = 0; b < bs; b++) {
                // shared expert (expert 0)
                if (weights[0] != nullptr && curPos[0] >= 0) {
                    int pos = curPos[0]++;
                    memcpyTasks[idx++] = MultiThreadMemcpyMultiLinesTask(
                        expandInputPtr + pos * bytesPerLine,
                        realInputPtr + b * bytesPerLine,
                        bytesPerLine
                    );
                }
                // routed experts
                for (int j = 0; j < topk; j++) {
                    int expertIdx = indexData[b * topk + j] + 1;
                    if (weights[expertIdx * 2] != nullptr && curPos[expertIdx] >= 0) {
                        int pos = curPos[expertIdx]++;
                        memcpyTasks[idx++] = MultiThreadMemcpyMultiLinesTask(
                            expandInputPtr + pos * bytesPerLine,
                            realInputPtr + b * bytesPerLine,
                            bytesPerLine
                        );
                    }
                }
            }
            memcpyTasks.resize(idx);
        }
        RunMultiThreadMemcpyMultiLines(memcpyTasks, GetAlivePool());

        // 2. gateUp
        auto *numaConfig = GetNumaConfig();

        int offset = 0;
        int stride = 64;

        const int gateCols = interDim * 2;
        const int gateColsPerNuma = gateCols / numaConfig->numaCnt;
        const bool canFuseGroup32 =
            downInputDataType == DataType::INF_INT8_GROUP32 &&
            interDim % 32 == 0 && gateColsPerNuma % 64 == 0;
        const bool canFuseDstConvert =
            downInputDataType == DataType::FLOAT32 ||
            downInputDataType == DataType::FLOAT16 ||
            downInputDataType == DataType::BFLOAT16 ||
            canFuseGroup32;
        const bool useDeepSeekV4LargeFast =
            deepSeekV4Mode &&
            GetCPUInstructInfo()->hasAVX512BF16 &&
            std::getenv(
                "FASTLLM_DSV4_DISABLE_NUMAS_MOE_LARGE_FAST") == nullptr;
        const bool useDeepSeekV4GroupedDecodeFast =
            useDeepSeekV4LargeFast && bs > 1 && bs <= 8 &&
            std::getenv(
                "FASTLLM_DSV4_DISABLE_NUMAS_MOE_GROUPED_DECODE") ==
                nullptr;
        if (useDeepSeekV4GroupedDecodeFast) {
            // Match the tuned small-batch queue granularity.  The generic
            // grouped path's 64-column chunks create thousands of heap-backed
            // tasks for an eight-row verifier layer and erase the cache reuse
            // gained by grouping repeated experts.
            stride = 208;
        }
        const bool skipRedundantCrossSwiglu =
            useDeepSeekV4LargeFast;
        const bool useParallelDeepSeekV4Prepare =
            useDeepSeekV4LargeFast &&
            interDim % (128 * numaConfig->numaCnt) == 0;
        const bool useParallelDeepSeekV4Round =
            useDeepSeekV4LargeFast;
        const bool useParallelDeepSeekV4Store =
            useDeepSeekV4LargeFast;
        const bool useBFloat16SiluLookup =
            useParallelDeepSeekV4Prepare;
        const bool useDirectBFloat16Prepare =
            useParallelDeepSeekV4Prepare;

        std::vector<std::vector <fastllm::MultiThreadBaseOp*> > ops;
        ops.resize(numaConfig->numaCnt);
        for (int e = 0; e < (int)expertTasks.size(); e++) {
            if (weights[e * 2] != nullptr && expertTasks[e].size() > 0 && cpuExperts.count(e)) {
                if (weights[e * 2]->numasData.empty() && weights[e * 2]->cpuData != nullptr) {
                    RegisterNumas(weights[e * 2], "linearSwiglu");
                }
                int lines = expertTasks[e].size();
                // Prepare input pointer for this expert's batch
                uint16_t* expertInputPtr = (uint16_t*)(expandInput.data() + offset * GetDataBytes(startDataType, 1, inputDim));
                    
                // Prepare output pointer for this expert's batch
                float* expertGateUpOutputPtr = gateUpOutput.data() + offset * interDim * 2;
                float* expertSwigluOutputPtr = swigluOutput.data() + offset * interDim;
                uint8_t* expertDstOutputPtr = canFuseDstConvert && !deepSeekV4Mode ?
                    (uint8_t*)downInput.data() + offset * GetDataBytes(downInputDataType, 1, interDim) : nullptr;

                int k = gateCols;
                int kPer = gateColsPerNuma;
                    
                for (int nid = 0; nid < numaConfig->numaCnt; nid++) {
                    if ((int)weights[e * 2]->numasData.size() <= nid || weights[e * 2]->numasData[nid] == nullptr) {
                        ErrorInFastLLM("NumasMergeMOE gate weight missing NUMA shard: " + weights[e * 2]->name + "\n");
                    }
                    // Get weight data (assuming weights are stored as `startDataType`)
                    int base = kPer * nid;
                    size_t outputOffset = GetDataBytes(DataType::FLOAT32, 1, base);

                    for (int st = 0; st < kPer; st += stride) {
                        int end = std::min(st + stride, kPer);
                        ops[nid].push_back(new MultiThreadGemmAndCrossSwigluOp(
                            (uint8_t*)expertInputPtr, startDataType,
                            weights[e * 2]->numasData[nid], weights[e * 2]->GetDataType(),
                            (uint8_t*)expertGateUpOutputPtr + outputOffset, DataType::FLOAT32,
                            expertSwigluOutputPtr,
                            lines, inputDim, k, st, end, base,
                            expertDstOutputPtr, downInputDataType,
                            skipRedundantCrossSwiglu
                        ));
                    }
                }
                offset += lines;
            }
        }

        if (useDeepSeekV4GroupedDecodeFast) {
            ScheduleDeepSeekV4NumasMoeTasks(ops);
        } else {
            DynamicScheduleTasks(ops);
        }

        // 4. swigluOutput -> downInput. DeepSeek-V4 must redo the fused
        // activation from gateUpOutput to preserve its BF16/clamp/route order.
        if (useParallelDeepSeekV4Prepare) {
            offset = 0;
            const size_t downRowBytes =
                GetDataBytes(downInputDataType, 1, interDim);
            const int interDimPerNuma =
                interDim / numaConfig->numaCnt;
            auto &prepareTaskStorage =
                fastllmMoeDataManagerNumas.prepareTaskStorage;
            auto &prepareTasks =
                fastllmMoeDataManagerNumas.taskPointers;
            prepareTaskStorage.resize(numaConfig->numaCnt);
            prepareTasks.resize(numaConfig->numaCnt);
            size_t tasksPerNode = (size_t)totalLines;
            for (int nid = 0; nid < numaConfig->numaCnt; nid++) {
                prepareTaskStorage[nid].clear();
                prepareTasks[nid].clear();
                prepareTaskStorage[nid].reserve(tasksPerNode);
                prepareTasks[nid].reserve(tasksPerNode);
            }
            for (int e = 0; e < (int)expertTasks.size(); e++) {
                if (weights[e * 2] == nullptr ||
                    expertTasks[e].empty() || !cpuExperts.count(e)) {
                    continue;
                }
                int lines = expertTasks[e].size();
                bool quantize = IsDeepSeekV4QuantizedWeight(
                    *weights[e * 2 + 1]);
                for (int line = 0; line < lines; line++) {
                    size_t row = (size_t)offset + line;
                    bool routed = e != 0;
                    float routeWeight = routed ?
                        expertTasks[e][line].second : 1.0f;
                    for (int nid = 0; nid < numaConfig->numaCnt;
                         nid++) {
                        int st = nid * interDimPerNuma;
                        int end = st + interDimPerNuma;
                        prepareTaskStorage[nid].emplace_back(
                            gateUpOutput.data() + row * interDim * 2,
                            swigluOutput.data() + row * interDim,
                            downInput.data() + row * downRowBytes,
                            downInputDataType, st, end,
                            routed, routeWeight, swigluLimit,
                            quantize, useBFloat16SiluLookup,
                            useDirectBFloat16Prepare);
                    }
                }
                offset += lines;
            }
            for (int nid = 0; nid < numaConfig->numaCnt; nid++) {
                for (auto &task : prepareTaskStorage[nid]) {
                    prepareTasks[nid].push_back(&task);
                }
            }
            ScheduleDeepSeekV4NumasMoeTasks(prepareTasks, false);
        } else if (deepSeekV4Mode) {
            offset = 0;
            const size_t downRowBytes = GetDataBytes(downInputDataType, 1, interDim);
            for (int e = 0; e < (int)expertTasks.size(); e++) {
                if (weights[e * 2] != nullptr && expertTasks[e].size() > 0 && cpuExperts.count(e)) {
                    int lines = expertTasks[e].size();
                    for (int line = 0; line < lines; line++) {
                        PrepareDeepSeekV4DownInput(
                            gateUpOutput.data() + (size_t)(offset + line) * interDim * 2,
                            swigluOutput.data() + (size_t)(offset + line) * interDim,
                            downInput.data() + (size_t)(offset + line) * downRowBytes,
                            downInputDataType, *weights[e * 2 + 1], interDim,
                            e != 0, e == 0 ? 1.0f : expertTasks[e][line].second,
                            swigluLimit
                        );
                    }
                    offset += lines;
                }
            }
        } else if (!canFuseDstConvert) {
            offset = 0;
            for (int e = 0; e < (int)expertTasks.size(); e++) {
                if (weights[e * 2] != nullptr && expertTasks[e].size() > 0 && cpuExperts.count(e)) {
                    int lines = expertTasks[e].size();
                    float* expertSwigluOutputPtr = swigluOutput.data() + offset * interDim;
                    uint8_t* expertDstOutputPtr = (uint8_t*)downInput.data() + offset * GetDataBytes(downInputDataType, 1, interDim);
                    RunMultiThreadConvertFromFloat32(expertDstOutputPtr, downInputDataType,
                                                    expertSwigluOutputPtr, lines, interDim, GetAlivePool());
                    offset += lines;
                }
            }
        }

        // 5. down
        offset = 0;
        stride = useDeepSeekV4GroupedDecodeFast ? 128 : 64;
        ops.resize(numaConfig->numaCnt);
        for (int i = 0; i < (int)ops.size(); i++) {
            ops[i].clear();
        }

        for (int e = 0; e < (int)expertTasks.size(); e++) {
            if (weights[e * 2 + 1] != nullptr && expertTasks[e].size() > 0 && cpuExperts.count(e)) {
                if (weights[e * 2 + 1]->numasData.empty() && weights[e * 2 + 1]->cpuData != nullptr) {
                    RegisterNumas(weights[e * 2 + 1], "linearColumn");
                }
                int lines = expertTasks[e].size();
                // Prepare input pointer for this expert's batch
                uint16_t* expertDownInputPtr = (uint16_t*)(downInput.data() + offset * GetDataBytes(downInputDataType, 1, interDim));
                    
                // Prepare output pointer for this expert's batch
                float* expertDownOutputPtr = downOutput.data() + offset * dim;

                int k = dim;
                int kPer = k / numaConfig->numaCnt;
                    
                for (int nid = 0; nid < numaConfig->numaCnt; nid++) {
                    AssertInFastLLM((int)weights[e * 2 + 1]->numasData.size() > nid &&
                                    weights[e * 2 + 1]->numasData[nid] != nullptr,
                                    "NumasMergeMOE down weight missing NUMA shard: " + weights[e * 2 + 1]->name + "\n");
                    // Get weight data (assuming weights are stored as `downInputDataType`)
                    int base = kPer * nid;
                    size_t outputOffset = GetDataBytes(DataType::FLOAT32, 1, base);

                    for (int st = 0; st < kPer; st += stride) {
                        int end = std::min(st + stride, kPer);
                        ops[nid].push_back(new MultiThreadGemmOp(
                            (uint8_t*)expertDownInputPtr, downInputDataType,
                            weights[e * 2 + 1]->numasData[nid], weights[e * 2 + 1]->GetDataType(),
                            (uint8_t*)expertDownOutputPtr + outputOffset, DataType::FLOAT32,
                            lines, interDim, k, st, end
                        ));
                    }
                }
                offset += lines;
            }
        }

        if (useDeepSeekV4GroupedDecodeFast) {
            ScheduleDeepSeekV4NumasMoeTasks(ops);
        } else {
            DynamicScheduleTasks(ops);
        }

        if (deepSeekV4Mode && useParallelDeepSeekV4Round) {
            auto &roundOps = fastllmMoeDataManagerNumas.roundOps;
            roundOps.clear();
            size_t count = (size_t)totalLines * dim;
            int threads = std::min(
                (size_t)numaConfig->threads, count);
            roundOps.reserve(threads);
            size_t per = (count + threads - 1) / threads;
            for (int tid = 0; tid < threads; tid++) {
                size_t st = (size_t)tid * per;
                size_t end = std::min(st + per, count);
                if (st < end) {
                    roundOps.emplace_back(downOutput.data(), st, end);
                }
            }
            for (int tid = 0; tid < (int)roundOps.size(); tid++) {
                pool->PushOp(tid, &roundOps[tid]);
            }
            for (int tid = 0; tid < (int)roundOps.size(); tid++) {
                pool->Wait(tid);
            }
        } else if (deepSeekV4Mode) {
            for (size_t i = 0; i < (size_t)totalLines * dim; i++) {
                downOutput[i] = RoundFloat32ToBFloat16RNE(downOutput[i]);
            }
        }

        // debug: 输出指定token关联的所有专家计算结果（通过环境变量 FASTLLM_DEBUG_TOKEN_ID 指定token id，逗号分隔）
        /* {
            static std::set<int> debugTokenIds;
            static bool debugTokenIdInited = false;
            if (!debugTokenIdInited) {
                const char *env = getenv("FASTLLM_DEBUG_TOKEN_ID");
                if (env) {
                    std::string s(env);
                    size_t pos = 0;
                    while (pos < s.size()) {
                        size_t next = s.find(',', pos);
                        if (next == std::string::npos) next = s.size();
                        debugTokenIds.insert(atoi(s.substr(pos, next - pos).c_str()));
                        pos = next + 1;
                    }
                }
                debugTokenIdInited = true;
            }
            if (!debugTokenIds.empty()) {
                int debugOffset = 0;
                for (int e = 0; e < (int)expertTasks.size(); e++) {
                    if (weights[e * 2] != nullptr && expertTasks[e].size() > 0 && cpuExperts.count(e)) {
                        float* debugDownOutput = downOutput.data() + debugOffset * dim;
                        for (int i = 0; i < (int)expertTasks[e].size(); i++) {
                            int rowIdx = expertTasks[e][i].first;
                            if (debugTokenIds.count(rowIdx)) {
                                float score = expertTasks[e][i].second;
                                float sumAbs = 0.0f;
                                for (int d = 0; d < dim; d++) {
                                    sumAbs += std::abs(debugDownOutput[i * dim + d]);
                                }
                                printf("[DEBUG origToken=%d] expert=%d, score=%.6f, output_l1norm=%.6f, first5=[%.6f, %.6f, %.6f, %.6f, %.6f]\n",
                                       rowIdx, e, score, sumAbs,
                                       debugDownOutput[i * dim + 0],
                                       debugDownOutput[i * dim + 1],
                                       debugDownOutput[i * dim + 2],
                                       debugDownOutput[i * dim + 3],
                                       debugDownOutput[i * dim + 4]);
                            }
                        }
                        debugOffset += expertTasks[e].size();
                    }
                }
                fflush(stdout);
            }
        } */

        uint8_t *finalCpuOutput = cpuOutputBuffer != nullptr ? cpuOutputBuffer : (uint8_t*)output.cpuData;

        // 6. reduce
        {
            // 计算每个样本选择的专家数 k
            int k = 0;
            std::vector<int> samples_expert_count(bs, 0);
            for (int e = 0; e < (int)expertTasks.size(); e++) {
                if (weights[e * 2] != nullptr && cpuExperts.count(e)) {
                    for (auto& task : expertTasks[e]) {
                        int rowIdx = task.first;
                        samples_expert_count[rowIdx]++;
                        k = std::max(k, samples_expert_count[rowIdx]);
                    }
                }
            }

            // 分配内存: task_weights 按 totalLines 大小，以 downOutput 行号索引
            std::vector<int> pos(bs * k, -1);
            std::vector<float> task_weights(totalLines, 0.0f);
            std::vector<int> sample_expert_idx(bs, 0);

            std::vector<int> expertOffsets(expertTasks.size(), -1);
            int reduceOffset = 0;
            for (int e = 0; e < (int)expertTasks.size(); e++) {
                if (weights[e * 2] != nullptr && cpuExperts.count(e)) {
                    expertOffsets[e] = reduceOffset;
                    reduceOffset += expertTasks[e].size();
                }
            }
            std::vector<int> reduceExpertOrder;
            if (deepSeekV4Mode) {
                for (int e = 1; e < (int)expertTasks.size(); e++) {
                    reduceExpertOrder.push_back(e);
                }
                reduceExpertOrder.push_back(0);
            } else {
                for (int e = 0; e < (int)expertTasks.size(); e++) {
                    reduceExpertOrder.push_back(e);
                }
            }
            for (int e : reduceExpertOrder) {
                if (weights[e * 2] != nullptr && cpuExperts.count(e)) {
                    int line = 0;
                    for (auto& task : expertTasks[e]) {
                        int rowIdx = task.first;
                        float weight = deepSeekV4Mode ? 1.0f : task.second;
                        int idx = sample_expert_idx[rowIdx]++;
                        int outputRow = expertOffsets[e] + line++;
                        pos[rowIdx * k + idx] = outputRow;
                        task_weights[outputRow] = weight;
                    }
                }
            }

            // 有一些token不会被关联到任何专家，也需要先清零
            float *lastOutput = output.dataType == DataType::FLOAT32 ? (float*)finalCpuOutput : reduceOutput.data();
            memset(lastOutput, 0, bs * dim * sizeof(float));

            // 调用多线程函数
            MultiThreadReduceBatch(
                (uint8_t*)downOutput.data(),  // downOutData
                DataType::FLOAT32,             // downOutDataType
                task_weights.data(),           // weights
                lastOutput,                    // lastOutput
                pos.data(),                    // pos
                bs,                           // bsz
                k,                            // k (每个样本的专家数)
                dim                           // hidden_size
            );
        }

        // 7. reduceOutput -> last Output
        if (output.dataType != DataType::FLOAT32) {
            if (output.dataType == DataType::FLOAT16) {
                RunMultiThreadConvertFromFloat32((uint16_t*)finalCpuOutput, DataType::FLOAT16, 
                    reduceOutput.data(), bs, dim, GetAlivePool());
            } else if (output.dataType == DataType::BFLOAT16) {
                if (deepSeekV4Mode) {
                    uint16_t *dst = (uint16_t*)finalCpuOutput;
                    if (useParallelDeepSeekV4Store) {
                        auto &storeOps =
                            fastllmMoeDataManagerNumas.storeOps;
                        storeOps.clear();
                        size_t count = (size_t)bs * dim;
                        int threads = std::min(
                            (size_t)numaConfig->threads, count);
                        storeOps.reserve(threads);
                        size_t per =
                            (count + threads - 1) / threads;
                        for (int tid = 0; tid < threads; tid++) {
                            size_t st = (size_t)tid * per;
                            size_t end = std::min(st + per, count);
                            if (st < end) {
                                storeOps.emplace_back(
                                    reduceOutput.data(), dst, st, end);
                            }
                        }
                        for (int tid = 0;
                             tid < (int)storeOps.size(); tid++) {
                            pool->PushOp(tid, &storeOps[tid]);
                        }
                        for (int tid = 0;
                             tid < (int)storeOps.size(); tid++) {
                            pool->Wait(tid);
                        }
                    } else {
                        for (size_t i = 0; i < (size_t)bs * dim; i++) {
                            dst[i] = Float32ToBFloat16RNEBits(
                                reduceOutput[i]);
                        }
                    }
                } else {
                    RunMultiThreadConvertFromFloat32((uint16_t*)finalCpuOutput, DataType::BFLOAT16,
                        reduceOutput.data(), bs, dim, GetAlivePool());
                }
            }
        }
    }

    struct NumasFusedMoeLayerWeights {
        Data *gateSource = nullptr;
        Data *upSource = nullptr;
        Data *downSource = nullptr;
        int experts = 0;
        int inter = 0;
        int hidden = 0;
        DataType gateType = DataType::FLOAT32;
        DataType downType = DataType::FLOAT32;
        std::vector<std::unique_ptr<Data> > ownedWeights;
        std::vector<Data*> weights;
        std::vector<Data*> biass;

        bool IsReadyFor(Data &gate, Data &up, Data &down) const {
            return gateSource == &gate && upSource == &up && downSource == &down &&
                   experts > 0 && inter > 0 && hidden > 0 &&
                   gate.dims.size() == 3 && up.dims.size() == 3 && down.dims.size() == 3 &&
                   gate.dims[0] == experts && gate.dims[1] == inter && gate.dims[2] == hidden &&
                   up.dims[0] == experts && up.dims[1] == inter && up.dims[2] == hidden &&
                   down.dims[0] == experts && down.dims[1] == hidden && down.dims[2] == inter &&
                   gate.dataType == gateType && up.dataType == gateType && down.dataType == downType &&
                   weights.size() == (size_t)(experts + 1) * 2;
        }
    };

    static std::unordered_map<int, NumasFusedMoeLayerWeights> fastllmNumasFusedMoeWeightsPerLayer;

    static void CopyNumasFusedLinearMeta(Data &dst, const Data &src, const std::string &name) {
        dst.name = name;
        dst.weightType = WeightType::LINEAR;
        dst.isModelWeight = true;
        dst.blockK = src.blockK;
        dst.blockM = src.blockM;
        dst.group = src.group;
        dst.groupCnt = src.groupCnt;
        dst.perChannelAxis = src.perChannelAxis;
        dst.tpLinearType = src.tpLinearType;
        dst.tpPackType = src.tpPackType;
    }

    static void EnsureNumasFusedWeightCpuData(Data &weight, const char *name) {
        if (weight.cpuData != nullptr) {
            return;
        }
#ifdef USE_CUDA
        if (weight.cudaData != nullptr) {
            weight.ToDevice(DataDevice::CPU);
            if (weight.cpuData != nullptr) {
                return;
            }
        }
#endif
        ErrorInFastLLM(std::string("NumasFusedMOE: fused weight ") + name + " has no CPU data.\n");
    }

    static void AppendNumasFusedFp8Scales(Data &dst, const Data &src,
                                          int expert, int srcRowStart, int rows) {
        if (src.dataType != DataType::FP8_E4M3) {
            return;
        }
        AssertInFastLLM(src.dims.size() == 3 && src.blockK > 0 && src.blockM > 0 &&
                        !src.scales.empty(),
                        "NumasFusedMOE FP8 source scales are missing.\n");
        int rowsPerExpert = src.dims[1];
        int cols = src.dims[2];
        int scaleCols = (cols - 1) / src.blockM + 1;
        int scaleRowsPerExpert = (rowsPerExpert - 1) / src.blockK + 1;
        int scaleRowStart = srcRowStart / src.blockK;
        int scaleRows = (rows - 1) / src.blockK + 1;
        size_t srcOffset = ((size_t)expert * scaleRowsPerExpert + scaleRowStart) * scaleCols;
        size_t count = (size_t)scaleRows * scaleCols;
        AssertInFastLLM(expert >= 0 && expert < src.dims[0] &&
                        srcRowStart >= 0 && rows > 0 && srcRowStart + rows <= rowsPerExpert &&
                        srcRowStart % src.blockK == 0 &&
                        srcOffset + count <= src.scales.size(),
                        "NumasFusedMOE FP8 scale slice is out of bounds.\n");
        dst.scales.insert(dst.scales.end(),
                          src.scales.begin() + srcOffset,
                          src.scales.begin() + srcOffset + count);
    }

    static void CopyNumasFusedRows(Data &dst, int dstRowStart,
                                   Data &src, int expert, int srcRowStart, int rows) {
        AssertInFastLLM(src.dims.size() == 3 && dst.dims.size() == 2,
                        "NumasFusedMOE row copy expects 3D source and 2D destination.\n");
        int rowsPerExpert = src.dims[1];
        int cols = src.dims[2];
        AssertInFastLLM(dst.dims[1] == cols &&
                        expert >= 0 && expert < src.dims[0] &&
                        srcRowStart >= 0 && rows > 0 && srcRowStart + rows <= rowsPerExpert &&
                        dstRowStart >= 0 && dstRowStart + rows <= dst.dims[0],
                        "NumasFusedMOE row copy shape mismatch.\n");
        EnsureNumasFusedWeightCpuData(src, src.name.c_str());
        size_t bytesPerRow = GetDataBytes(src.dataType, 1, cols);
        memcpy(dst.cpuData + (size_t)dstRowStart * bytesPerRow,
               src.cpuData + ((size_t)expert * rowsPerExpert + srcRowStart) * bytesPerRow,
               (size_t)rows * bytesPerRow);
    }

    static NumasFusedMoeLayerWeights &GetNumasFusedMoeLayerWeights(
        int layer, Data &gate, Data &up, Data &down
    ) {
        AssertInFastLLM(gate.dims.size() == 3 && up.dims.size() == 3 && down.dims.size() == 3,
                        "NumasFusedMOE expects 3D fused weights.\n");
        int experts = gate.dims[0];
        int inter = gate.dims[1];
        int hidden = gate.dims[2];
        AssertInFastLLM(experts > 0 && inter > 0 && hidden > 0 &&
                        up.dims[0] == experts && up.dims[1] == inter && up.dims[2] == hidden &&
                        down.dims[0] == experts && down.dims[1] == hidden && down.dims[2] == inter &&
                        gate.dataType == up.dataType,
                        "NumasFusedMOE fused weight shapes mismatch.\n");

        auto &cache = fastllmNumasFusedMoeWeightsPerLayer[layer];
        if (cache.IsReadyFor(gate, up, down)) {
            return cache;
        }

        cache = NumasFusedMoeLayerWeights();
        cache.gateSource = &gate;
        cache.upSource = &up;
        cache.downSource = &down;
        cache.experts = experts;
        cache.inter = inter;
        cache.hidden = hidden;
        cache.gateType = gate.dataType;
        cache.downType = down.dataType;
        cache.weights.assign((size_t)(experts + 1) * 2, nullptr);
        cache.biass.assign((size_t)(experts + 1) * 2, nullptr);
        cache.ownedWeights.reserve((size_t)experts * 2);

        for (int expert = 0; expert < experts; expert++) {
            std::unique_ptr<Data> gateup(new Data(gate.dataType, {inter * 2, hidden}));
            CopyNumasFusedLinearMeta(*gateup, gate, gate.name + ".numas.gateup." + std::to_string(expert));
            gateup->tpLinearType = TP_LINEAR_ROW;
            gateup->tpPackType = TP_PACK_GATEUP;
            gateup->Allocate(false);
            gateup->scales.clear();
            CopyNumasFusedRows(*gateup, 0, gate, expert, 0, inter);
            CopyNumasFusedRows(*gateup, inter, up, expert, 0, inter);
            AppendNumasFusedFp8Scales(*gateup, gate, expert, 0, inter);
            AppendNumasFusedFp8Scales(*gateup, up, expert, 0, inter);
            RegisterNumas(gateup.get(), "linearSwiglu");

            std::unique_ptr<Data> downWeight(new Data(down.dataType, {hidden, inter}));
            CopyNumasFusedLinearMeta(*downWeight, down, down.name + ".numas.down." + std::to_string(expert));
            downWeight->tpLinearType = TP_LINEAR_COLUMN;
            downWeight->Allocate(false);
            downWeight->scales.clear();
            CopyNumasFusedRows(*downWeight, 0, down, expert, 0, hidden);
            AppendNumasFusedFp8Scales(*downWeight, down, expert, 0, hidden);
            RegisterNumas(downWeight.get(), "linearColumn");

            cache.weights[(expert + 1) * 2] = gateup.get();
            cache.weights[(expert + 1) * 2 + 1] = downWeight.get();
            cache.ownedWeights.push_back(std::move(gateup));
            cache.ownedWeights.push_back(std::move(downWeight));
        }

        return cache;
    }

    bool NumasFusedMOE::CanRun(const std::string &opType, const DataDict &datas,
                               const FloatDict &floatParams, const IntDict &intParams) {
        auto gateIt = datas.find("gate");
        auto upIt = datas.find("up");
        auto downIt = datas.find("down");
        if (gateIt == datas.end() || upIt == datas.end() || downIt == datas.end()) {
            return false;
        }
        Data *gate = gateIt->second;
        Data *up = upIt->second;
        Data *down = downIt->second;
        return gate != nullptr && up != nullptr && down != nullptr &&
               gate->dims.size() == 3 && up->dims.size() == 3 && down->dims.size() == 3 &&
               gate->dataType == up->dataType &&
               (gate->dataType == DataType::FP8_E4M3 ||
                gate->dataType == DataType::FP8_E4M3_BLOCK_128 ||
                gate->dataType == DataType::FP8_E4M3_PERCHANNEL ||
                gate->dataType == DataType::FLOAT32 ||
                gate->dataType == DataType::FLOAT16 ||
                gate->dataType == DataType::BFLOAT16) &&
               (down->dataType == gate->dataType ||
                down->dataType == DataType::FP8_E4M3 ||
                down->dataType == DataType::FP8_E4M3_BLOCK_128 ||
                down->dataType == DataType::FP8_E4M3_PERCHANNEL ||
                down->dataType == DataType::FLOAT32 ||
                down->dataType == DataType::FLOAT16 ||
                down->dataType == DataType::BFLOAT16);
    }

    void NumasFusedMOE::Reshape(const std::string &opType, const DataDict &datas,
                                const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &index = *(datas.find("index")->second);
        Data &gate = *(datas.find("gate")->second);
        Data &w1 = *(datas.find("w1")->second);
        Data &output = *(datas.find("output")->second);
        output.dataType = input.dataType;
        output.Resize(input.dims);
        if (index.dims.size() >= 2 && gate.dims.size() == 3) {
            w1.dataType = input.dataType;
            w1.Resize({(int)index.Count(0), gate.dims[1]});
        }
    }

    void NumasFusedMOE::Run(const std::string &opType, const DataDict &datas,
                            const FloatDict &floatParams, const IntDict &intParams) {
        MoeGateType gateType = intParams.find("gateType") != intParams.end() ?
            (MoeGateType)intParams.find("gateType")->second : MoeGateSwiglu;
        if (gateType != MoeGateSwiglu) {
            ErrorInFastLLM("NumasFusedMOE only supports swiglu gate type.\n");
        }
        float swigluLimit = floatParams.find("swigluLimit") != floatParams.end() ?
            floatParams.find("swigluLimit")->second : 0.0f;
        if (swigluLimit != 0.0f) {
            ErrorInFastLLM("NumasFusedMOE does not support non-zero swigluLimit.\n");
        }

        Data &rawInput = *(datas.find("input")->second);
        Data cpuInput;
        Data &input = *GetCpuTensor(rawInput, cpuInput, "input");
        Data &rawIndex = *(datas.find("index")->second);
        Data &rawScore = *(datas.find("score")->second);
        Data cpuIndex, cpuScore;
        Data &index = *GetCpuTensor(rawIndex, cpuIndex, "index");
        Data &score = *GetCpuTensor(rawScore, cpuScore, "score");
        Data &gate = *(datas.find("gate")->second);
        Data &up = *(datas.find("up")->second);
        Data &down = *(datas.find("down")->second);
        Data &output = *(datas.find("output")->second);

        int layer = intParams.find("layer") != intParams.end() ? intParams.find("layer")->second : 0;
        auto &layerWeights = GetNumasFusedMoeLayerWeights(layer, gate, up, down);
        FastllmMoeDataManagerNumas &manager =
            GetNumasMoeRuntimeCache()[layer % 2];

        output.dataType = input.dataType;
        output.expansionDims.clear();
        output.Resize(input.dims);
        output.Allocate();

        int topk = index.dims.size() >= 2 ? index.dims[1] : 1;
        std::unordered_set<int> cpuExperts;
        for (int expert = 1; expert <= layerWeights.experts; expert++) {
            cpuExperts.insert(expert);
        }
        DoNumasMergeMOEOnCPU(
            input, output, index, score,
            layerWeights.weights.data(), layerWeights.biass.data(),
            1.0f, (int)layerWeights.weights.size(), topk,
            cpuExperts, manager, nullptr
        );
    }

    void NumasMergeMOE::Run(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
 // auto ttt = std::chrono::system_clock::now();
 // std::vector <std::pair <std::string, float> > record;
  auto st = std::chrono::system_clock::now();
        Data &rawInput = *(datas.find("input")->second);
        Data cpuInput;
        Data &input = *GetCpuTensor(rawInput, cpuInput, "input");
        Data &output = *(datas.find("output")->second);
        Data &rawIndex = *(datas.find("index")->second);
        Data &rawScore = *(datas.find("score")->second);
        Data cpuIndex, cpuScore;
        Data &index = *GetCpuTensor(rawIndex, cpuIndex, "index");
        Data &score = *GetCpuTensor(rawScore, cpuScore, "score");
        Data &w1 = *(datas.find("w1")->second);
        Data &w2 = *(datas.find("w2")->second);
        Data &w3 = *(datas.find("w3")->second);
        Data **weights = (Data**)(datas.find("weights")->second);
        Data **biass = (Data**)(datas.find("biass")->second);

        float sharedScale = floatParams.find("sharedScale") != floatParams.end() ? floatParams.find("sharedScale")->second : 1.0f;
        float swigluLimit = floatParams.find("swigluLimit") != floatParams.end() ?
                            floatParams.find("swigluLimit")->second : 0.0f;
        bool deepSeekV4Mode = intParams.find("deepSeekV4Mode") != intParams.end() &&
                              intParams.find("deepSeekV4Mode")->second != 0;
        
        // index: [n, topk], score: [n, topk]
        int n = index.dims[0];
        int topk = index.dims[1];
        int weightsBatch = intParams.find("weights___batch") != intParams.end() ? intParams.find("weights___batch")->second : (topk + 1) * 2;
        int layer = intParams.find("layer") != intParams.end() ? intParams.find("layer")->second : 0;
        FastllmMoeDataManagerNumas &fastllmMoeDataManagerNumas =
            GetNumasMoeRuntimeCache()[layer % 2];
        // `moeFinal` is reused across layers and may have been reshaped back to
        // `[batch, seq, hidden]` by the caller. Reset it to the flattened MoE
        // input shape here before the NUMA path reads `output.dims[1]`.
        output.dataType = input.dataType;
        output.expansionDims.clear();
        output.Resize(input.dims);
        output.Allocate();
// printf("allocate spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        int32_t *indexData = (int32_t*)index.cpuData;
        float *scoreData = (float*)score.cpuData;

        bool profileDetail = std::getenv("FASTLLM_PROFILE_DETAIL") != nullptr &&
                             (std::getenv("FASTLLM_PROFILE") != nullptr ||
                              std::getenv("FASTLLM_PROFILE_DEEPSEEKV4") != nullptr ||
                              std::getenv("FASTLLM_PROFILE_NUMAS_MOE") != nullptr);
        double profileLast = NumasProfileNowMs();
        double profileRegisterMs = 0.0, profileResizeMs = 0.0, profileInputMs = 0.0;
        double profileGatePrepMs = 0.0, profileGateMs = 0.0, profileSwigluConvertMs = 0.0;
        double profileDownPrepMs = 0.0, profileDownMs = 0.0, profileReduceMs = 0.0, profileOutputMs = 0.0;
        int profileExpertCalls = 0;
        auto profileLap = [&](double &bucket) {
            if (!profileDetail) {
                return;
            }
            double now = NumasProfileNowMs();
            bucket += now - profileLast;
            profileLast = now;
        };

        // The small-batch fast path uses one activation format for all selected
        // experts.  Fall back to the grouped implementation when a mixed-quant
        // layer needs more than one format (for example Q8_0 shared + Q4_K/Q5_K
        // routed experts).  Homogeneous layers keep the existing fast path.
        auto &activeExpertList =
            fastllmMoeDataManagerNumas.activeExperts;
        activeExpertList.clear();
        activeExpertList.reserve((size_t)n * topk + 1);
        if (weights[0] != nullptr && weights[1] != nullptr) {
            activeExpertList.push_back(0);
        }
        for (int row = 0; row < n; row++) {
            for (int j = 0; j < topk; j++) {
                int expert = indexData[row * topk + j] + 1;
                if (expert >= 0 && expert * 2 + 1 < weightsBatch &&
                    weights[expert * 2] != nullptr && weights[expert * 2 + 1] != nullptr) {
                    activeExpertList.push_back(expert);
                }
            }
        }
        std::sort(activeExpertList.begin(), activeExpertList.end());
        activeExpertList.erase(
            std::unique(
                activeExpertList.begin(), activeExpertList.end()),
            activeExpertList.end());
        bool mixedActiveTypes = false;
        bool hasActiveType = false;
        std::pair<int, int> activeType;
        for (int expert : activeExpertList) {
            std::pair<int, int> curType = std::make_pair(
                (int)GetNumasLinearActDataType(weights[expert * 2], n),
                (int)GetNumasLinearActDataType(weights[expert * 2 + 1], n)
            );
            if (!hasActiveType) {
                activeType = curType;
                hasActiveType = true;
            } else if (curType != activeType) {
                mixedActiveTypes = true;
                break;
            }
        }
#ifdef USE_CUDA
        if (std::getenv("FASTLLM_NUMAS_MOE_GPU_TRACE") != nullptr) {
            std::vector<NumasMoeCudaInputReplica> traceReplicas =
                GetNumasMoeCudaInputReplicas(input);
            printf(
                "[Fastllm] NUMA MoE input layer=%d tokens=%d type=%s "
                "cpu=%d cuda=%d multi=%d replicated=%d replicas=%zu "
                "mixed_types=%d\n",
                layer, n, GetDataTypeName(input.dataType).c_str(),
                input.cpuData != nullptr, input.cudaData != nullptr,
                input.multiDeviceData,
                input.IsTensorParallelReplicated(),
                traceReplicas.size(), mixedActiveTypes);
            fflush(stdout);
        }
#endif
        if (mixedActiveTypes) {
            std::unordered_set<int> activeExperts(
                activeExpertList.begin(), activeExpertList.end());
            DoNumasMergeMOEOnCPU(
                input, output, index, score, weights, biass,
                sharedScale, weightsBatch, topk, activeExperts,
                fastllmMoeDataManagerNumas, nullptr,
                swigluLimit, deepSeekV4Mode
            );
            return;
        }

        // Verification rows are strongly correlated and commonly route to
        // the same experts.  The ordinary small-batch path below completes
        // every row independently, re-reading those expert weights once per
        // row.  Group DSpark-sized batches by expert and execute one M-row
        // GEMM so the decoded weight tiles are shared by all matching rows.
        // Keep larger small batches on their established path until they have
        // dedicated scheduling and correctness coverage.
        const bool useGroupedDecode = deepSeekV4Mode && n > 1 && n <= 8 &&
            std::getenv(
                "FASTLLM_DSV4_DISABLE_NUMAS_MOE_GROUPED_DECODE") == nullptr;
        if (useGroupedDecode) {
            std::unordered_set<int> activeExperts(
                activeExpertList.begin(), activeExpertList.end());
            DoNumasMergeMOEOnCPU(
                input, output, index, score, weights, biass,
                sharedScale, weightsBatch, topk, activeExperts,
                fastllmMoeDataManagerNumas, nullptr,
                swigluLimit, deepSeekV4Mode
            );
            return;
        }

        if (input.dims[0] < 32) {
// auto st = std::chrono::system_clock::now();
            int32_t *indexData = (int32_t*)index.cpuData;
            float *scoreData = (float*)score.cpuData;

            {
                auto *pool = GetAlivePool();

                int bs = input.dims[0];
                int inputDim = input.dims[1];
                int interDim = weights[2]->dims[0] / 2;
                int outputDim = output.dims[1];

                for (int o = 0; o < bs; o++) {
                    if (profileDetail) {
                        profileLast = NumasProfileNowMs();
                    }
                    auto &v =
                        fastllmMoeDataManagerNumas.selectedExperts;
                    v.clear();
                    v.reserve(topk + 1);
                    for (int j = 0; j < topk; j++) {
                        // index 存储的是专家索引（从0开始），需要+1因为0表示shared expert
                        int expertIdx = indexData[o * topk + j];
                        float expertScore = scoreData[o * topk + j];
                        int routedExpertCount = weightsBatch / 2 - 1;
                        if (expertIdx < 0 || expertIdx >= routedExpertCount) {
                            ErrorInFastLLM(
                                "NumasMergeMOE small batch: invalid routed expert " +
                                std::to_string(expertIdx) + " at layer " +
                                std::to_string(layer) + ", row " +
                                std::to_string(o) + ". Valid range is [0, " +
                                std::to_string(routedExpertCount) + ").\n");
                        }
                        v.push_back(std::make_pair(expertIdx + 1, expertScore));
                    }
                    if (weights[0] != nullptr) {
                        v.push_back(std::make_pair(0, sharedScale));
                    }
                    if (deepSeekV4Mode) {
                        std::stable_sort(v.begin(), v.end(), [](const auto &a, const auto &b) {
                            if (a.first == 0 || b.first == 0) {
                                return b.first == 0 && a.first != 0;
                            }
                            return a.first < b.first;
                        });
                    }

                    for (auto &expert : v) {
                        int e = expert.first;
                        if (weights[e * 2] == nullptr || weights[e * 2 + 1] == nullptr) {
                            ErrorInFastLLM("NumasMergeMOE small batch: missing expert weight.\n");
                        }
                        if (weights[e * 2]->numasData.empty() && weights[e * 2]->cpuData != nullptr) {
                            RegisterNumas(weights[e * 2], "linearSwiglu");
                        }
                        if (weights[e * 2 + 1]->numasData.empty() && weights[e * 2 + 1]->cpuData != nullptr) {
                            RegisterNumas(weights[e * 2 + 1], "linearColumn");
                        }
                    }
                    profileExpertCalls += (int)v.size();
                    profileLap(profileRegisterMs);

                    DataType startDataType = GetNumasLinearActDataType(weights[2], 1);
                    DataType downInputDataType = GetNumasLinearActDataType(weights[3], 1);

                    // 从 fastllmMoeDataManagerNumas 获取缓存的 vector，并根据需要调整大小
                    auto& realInput = fastllmMoeDataManagerNumas.realInput;
                    auto& inputFloat32 = fastllmMoeDataManagerNumas.inputFloat32;
                    auto& gateUpOutput = fastllmMoeDataManagerNumas.gateUpOutput;
                    auto& swigluOutput = fastllmMoeDataManagerNumas.swigluOutput;
                    auto& downInput = fastllmMoeDataManagerNumas.downInput;
                    auto& downOutput = fastllmMoeDataManagerNumas.downOutput;
                    auto& reduceOutput = fastllmMoeDataManagerNumas.reduceOutput;

                    // 计算所需大小
                    size_t realInputSize = GetDataBytes(startDataType, 1, inputDim);
                    size_t inputFloat32Size = 1 * inputDim;
                    size_t gateUpOutputSize = v.size() * interDim * 2;
                    size_t swigluOutputSize = v.size() * interDim;
                    size_t downInputSize = GetDataBytes(downInputDataType, v.size(), interDim);
                    size_t downOutputSize = v.size() * outputDim;
                    size_t reduceOutputSize = 1 * outputDim;

                    // 只在当前容量不足时才进行 resize
                    if (realInput.size() < realInputSize) {
                        realInput.resize(realInputSize);
                    }
                    if (input.dataType != DataType::FLOAT32 && inputFloat32.size() < inputFloat32Size) {
                        inputFloat32.resize(inputFloat32Size);
                    }
                    if (gateUpOutput.size() < gateUpOutputSize) {
                        gateUpOutput.resize(gateUpOutputSize);
                    }
                    if (swigluOutput.size() < swigluOutputSize) {
                        swigluOutput.resize(swigluOutputSize);
                    }
                    if (downInput.size() < downInputSize) {
                        downInput.resize(downInputSize);
                    }
                    if (downOutput.size() < downOutputSize) {
                        downOutput.resize(downOutputSize);
                    }
                    if (reduceOutput.size() < reduceOutputSize) {
                        reduceOutput.resize(reduceOutputSize);
                    }
                    profileLap(profileResizeMs);

// printf("malloc spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                    // 0. input -> realInput（若 input 非 FLOAT32 则先转为 float32）
                    if (input.dataType == startDataType && input.dataType != DataType::FLOAT32) {
                        size_t bytes = GetDataBytes(startDataType, 1, inputDim);
                        memcpy(realInput.data(), (uint8_t*)input.cpuData + o * bytes, bytes);
                    } else {
                        const float *inputF32Ptr = nullptr;
                        if (input.dataType == DataType::FLOAT32) {
                            inputF32Ptr = (const float*)input.cpuData + o * inputDim;
                        } else {
                            if (input.dataType == DataType::FLOAT16) {
                                Float16ToFloat32((uint16_t*)input.cpuData + o * inputDim, inputFloat32.data(), inputDim);
                            } else if (input.dataType == DataType::BFLOAT16) {
                                BFloat16ToFloat32((uint16_t*)input.cpuData + o * inputDim, inputFloat32.data(), inputDim);
                            } else {
                                ErrorInFastLLM("NumasMergeMOE: unsupported input dataType.\n");
                            }
                            inputF32Ptr = inputFloat32.data();
                        }
                        RunMultiThreadConvertFromFloat32(realInput.data(), startDataType, inputF32Ptr, 1, inputDim, GetAlivePool());
                    }
                    profileLap(profileInputMs);
// printf("RunMultiThreadConvertFromFloat32 spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));

                    // 1. gateUp + swiglu
                    auto *numaConfig = GetNumaConfig();

                    bool useDeepSeekV4MoeFast =
                        deepSeekV4Mode &&
                        GetCPUInstructInfo()->hasAVX512BF16 &&
                        std::getenv(
                            "FASTLLM_DSV4_DISABLE_NUMAS_MOE_FAST") == nullptr;
                    bool reuseMoeTaskStorage =
                        useDeepSeekV4MoeFast &&
                        std::getenv(
                            "FASTLLM_DSV4_DISABLE_NUMAS_MOE_TASK_CACHE") ==
                            nullptr;
                    bool useDirectGemmQueue =
                        useDeepSeekV4MoeFast &&
                        std::getenv(
                            "FASTLLM_DSV4_DISABLE_NUMAS_MOE_DIRECT_GEMM") ==
                            nullptr;
                    bool useBFloat16SiluLookup =
                        useDeepSeekV4MoeFast;
                    bool useDirectBFloat16Prepare =
                        useDeepSeekV4MoeFast;
                    // On the 20 workers per NUMA node used by -t 40, the gate
                    // uses ten chunks per expert (60 tasks, three waves).
                    // The smaller down chunks trade a little queue overhead
                    // for a much more even tail across those workers.
                    int gateRowsPerTask = 208;
                    int downRowsPerTask = 128;
                    // Keep each activation task on one gate-output NUMA
                    // shard. For DeepSeek-V4-Flash this is 2048 / 2 = 1024
                    // rows, while FP8 quantization remains block-128.
                    int swigluRowsPerTask = std::max(
                        128,
                        interDim / std::max(1, numaConfig->numaCnt));

                    int totalExperts = v.size();
                    int k = interDim * 2;
                    int kPer = k / numaConfig->numaCnt;
                    // With one NUMA node and 30 workers, six routed experts
                    // use 210 down tasks.  These are exactly seven full worker
                    // waves, avoiding the decode tail left by the two-NUMA
                    // default above.
                    if (numaConfig->numaCnt == 1 &&
                        numaConfig->threads == 30 &&
                        totalExperts == 6 && kPer == 4096) {
                        downRowsPerTask = 120;
                    }
                    // Group-32 quantization can be fused only when every
                    // NUMA shard begins and ends at a complete 32-value
                    // SwiGLU group (64 interleaved gate/up columns).
                    const bool canFuseGroup32 =
                        downInputDataType == DataType::INF_INT8_GROUP32 &&
                        interDim % 32 == 0 && kPer % 64 == 0;
                    const bool canFuseDstConvert =
                        downInputDataType == DataType::FLOAT32 ||
                        downInputDataType == DataType::FLOAT16 ||
                        downInputDataType == DataType::BFLOAT16 ||
                        canFuseGroup32;
                    if (useDeepSeekV4MoeFast) {
                        std::vector<int> localExpertOrder;
                        auto &expertOrder = reuseMoeTaskStorage ?
                            fastllmMoeDataManagerNumas.expertOrder :
                            localExpertOrder;
                        expertOrder.clear();
                        if (expertOrder.capacity() <
                            (size_t)totalExperts) {
                            expertOrder.reserve(totalExperts);
                        }
                        for (int i = 0; i < totalExperts; i++) {
                            expertOrder.push_back(i);
                        }
                        auto gateRowBytes = [&](int expertIdx) {
                            return GetDataBytes(
                                weights[v[expertIdx].first * 2]->
                                    GetDataType(),
                                1, inputDim);
                        };
                        bool mixedGateRows = false;
                        for (int i = 1; i < totalExperts; i++) {
                            mixedGateRows |=
                                gateRowBytes(i) != gateRowBytes(0);
                        }
                        if (mixedGateRows) {
                            std::stable_sort(
                                expertOrder.begin(), expertOrder.end(),
                                [&](int a, int b) {
                                    return gateRowBytes(a) >
                                           gateRowBytes(b);
                                });
                        }

                        if (useDirectGemmQueue) {
                            std::vector<
                                DeepSeekV4NumasGemmQueueContext>
                                contexts;
                            contexts.reserve(numaConfig->numaCnt);
                            for (int nid = 0;
                                 nid < numaConfig->numaCnt; nid++) {
                                for (int expertIdx : expertOrder) {
                                    int e = v[expertIdx].first;
                                    AssertInFastLLM(
                                        (int)weights[e * 2]->
                                                numasData.size() > nid &&
                                        weights[e * 2]->
                                                numasData[nid] != nullptr,
                                        "NumasMergeMOE small batch gate "
                                        "weight missing NUMA shard: " +
                                        weights[e * 2]->name + "\n");
                                }
                                contexts.emplace_back(
                                    &v, &expertOrder, weights,
                                    realInput.data(), 0,
                                    startDataType, gateUpOutput.data(),
                                    0, nid, inputDim, k, kPer,
                                    gateRowsPerTask);
                            }
                            profileLap(profileGatePrepMs);
                            ScheduleDeepSeekV4NumasGemmQueue(
                                contexts);
                        } else {
                            std::vector<std::vector<MultiThreadGemmOp>>
                                localGateTaskStorage;
                            std::vector<std::vector<
                                MultiThreadBaseOp*>>
                                localGateTasks;
                            auto &gateTaskStorage =
                                reuseMoeTaskStorage ?
                                    fastllmMoeDataManagerNumas
                                        .gemmTaskStorage :
                                    localGateTaskStorage;
                            auto &gateTasks = reuseMoeTaskStorage ?
                                fastllmMoeDataManagerNumas.taskPointers :
                                localGateTasks;
                            gateTaskStorage.resize(
                                numaConfig->numaCnt);
                            gateTasks.resize(numaConfig->numaCnt);
                            size_t tasksPerNode =
                                (size_t)totalExperts *
                                ((kPer + gateRowsPerTask - 1) /
                                 gateRowsPerTask);
                            for (int nid = 0;
                                 nid < numaConfig->numaCnt; nid++) {
                                gateTaskStorage[nid].clear();
                                gateTasks[nid].clear();
                                gateTaskStorage[nid].reserve(
                                    tasksPerNode);
                                int base = kPer * nid;
                                for (int expertIdx : expertOrder) {
                                    int e = v[expertIdx].first;
                                    AssertInFastLLM(
                                        (int)weights[e * 2]->
                                                numasData.size() > nid &&
                                        weights[e * 2]->
                                                numasData[nid] != nullptr,
                                        "NumasMergeMOE small batch gate "
                                        "weight missing NUMA shard: " +
                                        weights[e * 2]->name + "\n");
                                    for (int row = 0; row < kPer;
                                         row += gateRowsPerTask) {
                                        int end = std::min(
                                            row + gateRowsPerTask,
                                            kPer);
                                        gateTaskStorage[nid].
                                            emplace_back(
                                                (uint8_t*)
                                                    realInput.data(),
                                                startDataType,
                                                weights[e * 2]->
                                                    numasData[nid],
                                                weights[e * 2]->
                                                    GetDataType(),
                                                (uint8_t*)(
                                                    gateUpOutput.data() +
                                                    (size_t)expertIdx *
                                                        k +
                                                    base),
                                                DataType::FLOAT32,
                                                1, inputDim, k,
                                                row, end);
                                    }
                                }
                                gateTasks[nid].reserve(
                                    gateTaskStorage[nid].size());
                                for (auto &task :
                                     gateTaskStorage[nid]) {
                                    gateTasks[nid].push_back(&task);
                                }
                            }
                            profileLap(profileGatePrepMs);
                            ScheduleDeepSeekV4NumasMoeTasks(
                                gateTasks, false);
                        }
                    } else {
                        std::vector<MultiThreadBaseOp*> ops(
                            numaConfig->threads);
                        for (int i = 0; i < (int)ops.size(); i++) {
                            ops[i] = new MultiThreadMultiOps();
                        }
                        for (int nid = 0;
                             nid < numaConfig->numaCnt; nid++) {
                            int base = kPer * nid;
                            int threadNum =
                                numaConfig->numaToCpuDict[nid].size();
                            int totalRows = kPer * totalExperts;
                            // A fused group-32 destination must never be
                            // split between workers because its scale and sum
                            // are shared by all 32 activation values.
                            int unitRows = canFuseGroup32 ? 64 : 4;
                            int rowsPerThread =
                                (totalRows / unitRows) / threadNum;
                            int remainingRows =
                                (totalRows / unitRows) % threadNum;
                            int currentRow = 0;

                            for (int tid = 0; tid < threadNum; tid++) {
                                int threadRows =
                                    (rowsPerThread +
                                     (tid < remainingRows ? 1 : 0)) *
                                    unitRows;
                                int endRow = currentRow + threadRows;
                                int rowStart = currentRow;
                                while (rowStart < endRow) {
                                    int expertIdx = rowStart / kPer;
                                    if (expertIdx >= totalExperts) {
                                        break;
                                    }
                                    int e = v[expertIdx].first;
                                    int expertStartRow =
                                        rowStart % kPer;
                                    int expertEndRow = std::min(
                                        kPer,
                                        expertStartRow +
                                            (endRow - rowStart));
                                    size_t outputOffset =
                                        GetDataBytes(
                                            DataType::FLOAT32,
                                            expertIdx, k) +
                                        GetDataBytes(
                                            DataType::FLOAT32, 1,
                                            base);
                                    uint8_t *dstPtr =
                                        canFuseDstConvert &&
                                                !deepSeekV4Mode ?
                                            (uint8_t*)downInput.data() +
                                                expertIdx *
                                                    GetDataBytes(
                                                        downInputDataType,
                                                        1, interDim) :
                                            nullptr;
                                    AssertInFastLLM(
                                        (int)weights[e * 2]->
                                                numasData.size() >
                                            nid &&
                                        weights[e * 2]->
                                                numasData[nid] !=
                                            nullptr,
                                        "NumasMergeMOE small batch gate "
                                        "weight missing NUMA shard: " +
                                            weights[e * 2]->name +
                                            "\n");
                                    ((MultiThreadMultiOps*)ops[
                                        numaConfig->
                                            numaToCpuDict[nid][tid]
                                                .first])->
                                        ops.push_back(
                                            new MultiThreadGemmAndCrossSwigluOp(
                                                (uint8_t*)
                                                    realInput.data(),
                                                startDataType,
                                                weights[e * 2]->
                                                    numasData[nid],
                                                weights[e * 2]->
                                                    GetDataType(),
                                                (uint8_t*)
                                                        gateUpOutput.data() +
                                                    outputOffset,
                                                DataType::FLOAT32,
                                                swigluOutput.data() +
                                                    expertIdx *
                                                        interDim,
                                                1, inputDim, k,
                                                expertStartRow,
                                                expertEndRow, base,
                                                dstPtr,
                                                downInputDataType));
                                    rowStart +=
                                        expertEndRow -
                                        expertStartRow;
                                    if (expertEndRow == kPer) {
                                        rowStart =
                                            (expertIdx + 1) * kPer;
                                    }
                                }
                                currentRow = endRow;
                            }
                        }
                        profileLap(profileGatePrepMs);
                        for (int i = 0; i < (int)ops.size(); i++) {
                            pool->PushOp(i, ops[i]);
                        }
                        for (int i = 0; i < (int)ops.size(); i++) {
                            pool->Wait(i);
                            delete ops[i];
                        }
                    }
                    profileLap(profileGateMs);

                    // 4. swigluOutput -> downInput
                    if (useDeepSeekV4MoeFast) {
                        const size_t downRowBytes =
                            GetDataBytes(
                                downInputDataType, 1, interDim);
                        size_t taskCount =
                            (size_t)totalExperts *
                            ((interDim + swigluRowsPerTask - 1) /
                             swigluRowsPerTask);
                        std::vector<std::vector<
                            MultiThreadDeepSeekV4NumasDownPrepareOp>>
                            localPrepareTaskStorage;
                        std::vector<std::vector<MultiThreadBaseOp*>>
                            localPrepareTasks;
                        auto &prepareTaskStorage =
                            reuseMoeTaskStorage ?
                                fastllmMoeDataManagerNumas
                                    .prepareTaskStorage :
                                localPrepareTaskStorage;
                        auto &prepareTasks = reuseMoeTaskStorage ?
                            fastllmMoeDataManagerNumas.taskPointers :
                            localPrepareTasks;
                        prepareTaskStorage.resize(numaConfig->numaCnt);
                        prepareTasks.resize(numaConfig->numaCnt);
                        size_t tasksPerNode =
                            (taskCount + numaConfig->numaCnt - 1) /
                            numaConfig->numaCnt;
                        for (int nid = 0;
                             nid < numaConfig->numaCnt; nid++) {
                            prepareTaskStorage[nid].clear();
                            prepareTasks[nid].clear();
                            prepareTaskStorage[nid].reserve(tasksPerNode);
                            prepareTasks[nid].reserve(tasksPerNode);
                        }
                        size_t taskIndex = 0;
                        for (int expertIdx = 0;
                             expertIdx < totalExperts; expertIdx++) {
                            int e = v[expertIdx].first;
                            bool routed = e != 0;
                            float routeWeight =
                                routed ? v[expertIdx].second : 1.0f;
                            bool quantize =
                                IsDeepSeekV4QuantizedWeight(
                                    *weights[e * 2 + 1]);
                            for (int row = 0; row < interDim;
                                 row += swigluRowsPerTask) {
                                int end =
                                    std::min(
                                        row + swigluRowsPerTask,
                                        interDim);
                                int nid =
                                    taskIndex++ % numaConfig->numaCnt;
                                prepareTaskStorage[nid].emplace_back(
                                    gateUpOutput.data() +
                                        (size_t)expertIdx *
                                            interDim * 2,
                                    swigluOutput.data() +
                                        (size_t)expertIdx * interDim,
                                    downInput.data() +
                                        (size_t)expertIdx *
                                            downRowBytes,
                                    downInputDataType, row, end,
                                    routed, routeWeight, swigluLimit,
                                    quantize, useBFloat16SiluLookup,
                                    useDirectBFloat16Prepare);
                            }
                        }
                        for (int nid = 0;
                             nid < numaConfig->numaCnt; nid++) {
                            for (auto &task :
                                 prepareTaskStorage[nid]) {
                                prepareTasks[nid].push_back(&task);
                            }
                        }
                        ScheduleDeepSeekV4NumasMoeTasks(
                            prepareTasks, false);
                    } else if (deepSeekV4Mode) {
                        const size_t downRowBytes =
                            GetDataBytes(
                                downInputDataType, 1, interDim);
                        for (int expertIdx = 0;
                             expertIdx < totalExperts; expertIdx++) {
                            int e = v[expertIdx].first;
                            PrepareDeepSeekV4DownInput(
                                gateUpOutput.data() + (size_t)expertIdx * interDim * 2,
                                swigluOutput.data() + (size_t)expertIdx * interDim,
                                downInput.data() + (size_t)expertIdx * downRowBytes,
                                downInputDataType, *weights[e * 2 + 1], interDim,
                                e != 0, e == 0 ? 1.0f : v[expertIdx].second,
                                swigluLimit
                            );
                        }
                    } else if (!canFuseDstConvert) {
                        for (int expertIdx = 0; expertIdx < totalExperts; expertIdx++) {
                            RunMultiThreadConvertFromFloat32(
                                (uint8_t*)downInput.data() + expertIdx * GetDataBytes(downInputDataType, 1, interDim),
                                downInputDataType,
                                swigluOutput.data() + expertIdx * interDim,
                                1, interDim, GetAlivePool());
                        }
                    }
                    profileLap(profileSwigluConvertMs);

                    // 5. down
                    totalExperts = v.size();
                    k = outputDim;
                    kPer = k / numaConfig->numaCnt;
                    if (useDeepSeekV4MoeFast) {
                        std::vector<int> localExpertOrder;
                        auto &expertOrder = reuseMoeTaskStorage ?
                            fastllmMoeDataManagerNumas.expertOrder :
                            localExpertOrder;
                        expertOrder.clear();
                        if (expertOrder.capacity() <
                            (size_t)totalExperts) {
                            expertOrder.reserve(totalExperts);
                        }
                        for (int i = 0; i < totalExperts; i++) {
                            expertOrder.push_back(i);
                        }
                        auto downRowBytesForExpert =
                            [&](int expertIdx) {
                                return GetDataBytes(
                                    weights[
                                        v[expertIdx].first * 2 + 1]->
                                            GetDataType(),
                                    1, interDim);
                            };
                        bool mixedDownRows = false;
                        for (int i = 1; i < totalExperts; i++) {
                            mixedDownRows |=
                                downRowBytesForExpert(i) !=
                                downRowBytesForExpert(0);
                        }
                        if (mixedDownRows) {
                            std::stable_sort(
                                expertOrder.begin(), expertOrder.end(),
                                [&](int a, int b) {
                                    return downRowBytesForExpert(a) >
                                           downRowBytesForExpert(b);
                                });
                        }

                        if (useDirectGemmQueue) {
                            const size_t downRowBytes =
                                GetDataBytes(
                                    downInputDataType, 1, interDim);
                            std::vector<
                                DeepSeekV4NumasGemmQueueContext>
                                contexts;
                            contexts.reserve(numaConfig->numaCnt);
                            for (int nid = 0;
                                 nid < numaConfig->numaCnt; nid++) {
                                for (int expertIdx : expertOrder) {
                                    int e = v[expertIdx].first;
                                    AssertInFastLLM(
                                        (int)weights[e * 2 + 1]->
                                                numasData.size() > nid &&
                                        weights[e * 2 + 1]->
                                                numasData[nid] != nullptr,
                                        "NumasMergeMOE small batch down "
                                        "weight missing NUMA shard: " +
                                        weights[e * 2 + 1]->name +
                                        "\n");
                                }
                                contexts.emplace_back(
                                    &v, &expertOrder, weights,
                                    downInput.data(), downRowBytes,
                                    downInputDataType,
                                    downOutput.data(),
                                    1, nid, interDim, k, kPer,
                                    downRowsPerTask);
                            }
                            profileLap(profileDownPrepMs);
                            ScheduleDeepSeekV4NumasGemmQueue(
                                contexts);
                        } else {
                            std::vector<std::vector<MultiThreadGemmOp>>
                                localDownTaskStorage;
                            std::vector<std::vector<
                                MultiThreadBaseOp*>>
                                localDownTasks;
                            auto &downTaskStorage =
                                reuseMoeTaskStorage ?
                                    fastllmMoeDataManagerNumas
                                        .gemmTaskStorage :
                                    localDownTaskStorage;
                            auto &downTasks = reuseMoeTaskStorage ?
                                fastllmMoeDataManagerNumas.taskPointers :
                                localDownTasks;
                            downTaskStorage.resize(
                                numaConfig->numaCnt);
                            downTasks.resize(numaConfig->numaCnt);
                            size_t tasksPerNode =
                                (size_t)totalExperts *
                                ((kPer + downRowsPerTask - 1) /
                                 downRowsPerTask);
                            const size_t downRowBytes =
                                GetDataBytes(
                                    downInputDataType, 1, interDim);
                            for (int nid = 0;
                                 nid < numaConfig->numaCnt; nid++) {
                                downTaskStorage[nid].clear();
                                downTasks[nid].clear();
                                downTaskStorage[nid].reserve(
                                    tasksPerNode);
                                int base = kPer * nid;
                                for (int expertIdx : expertOrder) {
                                    int e = v[expertIdx].first;
                                    AssertInFastLLM(
                                        (int)weights[e * 2 + 1]->
                                                numasData.size() > nid &&
                                        weights[e * 2 + 1]->
                                                numasData[nid] != nullptr,
                                        "NumasMergeMOE small batch down "
                                        "weight missing NUMA shard: " +
                                        weights[e * 2 + 1]->name +
                                        "\n");
                                    for (int row = 0; row < kPer;
                                         row += downRowsPerTask) {
                                        int end = std::min(
                                            row + downRowsPerTask,
                                            kPer);
                                        downTaskStorage[nid].
                                            emplace_back(
                                                downInput.data() +
                                                    (size_t)expertIdx *
                                                        downRowBytes,
                                                downInputDataType,
                                                weights[e * 2 + 1]->
                                                    numasData[nid],
                                                weights[e * 2 + 1]->
                                                    GetDataType(),
                                                (uint8_t*)(
                                                    downOutput.data() +
                                                    (size_t)expertIdx *
                                                        k +
                                                    base),
                                                DataType::FLOAT32,
                                                1, interDim, k,
                                                row, end);
                                    }
                                }
                                downTasks[nid].reserve(
                                    downTaskStorage[nid].size());
                                for (auto &task :
                                     downTaskStorage[nid]) {
                                    downTasks[nid].push_back(&task);
                                }
                            }
                            profileLap(profileDownPrepMs);
                            ScheduleDeepSeekV4NumasMoeTasks(
                                downTasks, false);
                        }
                    } else {
                        std::vector<MultiThreadBaseOp*> ops(
                            numaConfig->threads);
                        for (int i = 0; i < (int)ops.size(); i++) {
                            ops[i] = new MultiThreadMultiOps();
                        }
                        for (int nid = 0;
                             nid < numaConfig->numaCnt; nid++) {
                            int base = kPer * nid;
                            int threadNum =
                                numaConfig->numaToCpuDict[nid].size();
                            int totalRows = kPer * totalExperts;
                            int unitRows = 4;
                            int rowsPerThread =
                                (totalRows / unitRows) / threadNum;
                            int extraRows =
                                (totalRows / unitRows) % threadNum;
                            int currentRow = 0;
                            for (int tid = 0;
                                 tid < threadNum; tid++) {
                                int threadRows =
                                    (rowsPerThread +
                                     (tid < extraRows ? 1 : 0)) *
                                    unitRows;
                                int endRow =
                                    currentRow + threadRows;
                                for (int row = currentRow;
                                     row < endRow;) {
                                    int expertIdx = row / kPer;
                                    int rowInExpert = row % kPer;
                                    int rowsToProcess = std::min(
                                        kPer - rowInExpert,
                                        endRow - row);
                                    if (expertIdx < totalExperts) {
                                        int e = v[expertIdx].first;
                                        size_t inputOffset =
                                            expertIdx *
                                            GetDataBytes(
                                                downInputDataType,
                                                1, interDim);
                                        size_t outputOffset =
                                            GetDataBytes(
                                                DataType::FLOAT32,
                                                expertIdx, k) +
                                            GetDataBytes(
                                                DataType::FLOAT32,
                                                1, base);
                                        AssertInFastLLM(
                                            (int)weights[
                                                e * 2 + 1]->
                                                    numasData.size() >
                                                nid &&
                                            weights[e * 2 + 1]->
                                                    numasData[nid] !=
                                                nullptr,
                                            "NumasMergeMOE small "
                                            "batch down weight missing "
                                            "NUMA shard: " +
                                                weights[e * 2 + 1]->
                                                    name +
                                                "\n");
                                        ((MultiThreadMultiOps*)ops[
                                            numaConfig->
                                                numaToCpuDict[nid][tid]
                                                    .first])->
                                            ops.push_back(
                                                new MultiThreadGemmOp(
                                                    downInput.data() +
                                                        inputOffset,
                                                    downInputDataType,
                                                    weights[
                                                        e * 2 + 1]->
                                                        numasData[nid],
                                                    weights[
                                                        e * 2 + 1]->
                                                        GetDataType(),
                                                    (uint8_t*)
                                                            downOutput
                                                                .data() +
                                                        outputOffset,
                                                    DataType::FLOAT32,
                                                    1, interDim, k,
                                                    rowInExpert,
                                                    rowInExpert +
                                                        rowsToProcess));
                                    }
                                    row += rowsToProcess;
                                }
                                currentRow = endRow;
                            }
                        }
                        profileLap(profileDownPrepMs);
                        for (int i = 0; i < (int)ops.size(); i++) {
                            pool->PushOp(i, ops[i]);
                        }
                        for (int i = 0; i < (int)ops.size(); i++) {
                            pool->Wait(i);
                            delete ops[i];
                        }
                    }
                    profileLap(profileDownMs);

                    if (deepSeekV4Mode &&
                        !useDeepSeekV4MoeFast) {
                        for (size_t i = 0; i < (size_t)totalExperts * outputDim; i++) {
                            downOutput[i] = RoundFloat32ToBFloat16RNE(downOutput[i]);
                        }
                    }

// printf("down spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                    float *fLastOutput = reduceOutput.data();
                    if (output.dataType == DataType::FLOAT32) {
                        fLastOutput = ((float*)output.cpuData) + o * outputDim;
                    }

                    // 6. reduce
                    if (useDeepSeekV4MoeFast) {
                        int reduceThreads = std::min(
                            {(int)pool->threads.size(),
                             16, outputDim});
                        std::vector<
                            MultiThreadDeepSeekV4NumasReduceOp>
                            localReduceOps;
                        auto &reduceOps = reuseMoeTaskStorage ?
                            fastllmMoeDataManagerNumas.reduceOps :
                            localReduceOps;
                        reduceOps.clear();
                        reduceOps.reserve(reduceThreads);
                        int per = outputDim / reduceThreads;
                        int remain = outputDim % reduceThreads;
                        int cur = 0;
                        for (int i = 0; i < reduceThreads; i++) {
                            int end =
                                cur + per + (i < remain ? 1 : 0);
                            reduceOps.emplace_back(
                                downOutput.data(), fLastOutput,
                                totalExperts, outputDim, cur, end);
                            cur = end;
                        }
                        for (int i = 0; i < reduceThreads; i++) {
                            pool->PushOp(i, &reduceOps[i]);
                        }
                        for (int i = 0; i < reduceThreads; i++) {
                            pool->Wait(i);
                        }
                    } else {
                        for (int i = 0; i < (int)v.size(); i++) {
                            float value =
                                deepSeekV4Mode ? 1.0f :
                                v[i].second;
                            float *curOutput =
                                downOutput.data() +
                                (size_t)i * outputDim;
                            if (i == 0) {
                                for (int j = 0;
                                     j < outputDim; j++) {
                                    fLastOutput[j] =
                                        curOutput[j] * value;
                                }
                            } else {
                                for (int j = 0;
                                     j < outputDim; j++) {
                                    fLastOutput[j] +=
                                        curOutput[j] * value;
                                }
                            }
                        }
                    }
                    profileLap(profileReduceMs);

// printf("reduce spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                    // 7. reduceOutput -> last Output
                    if (output.dataType != DataType::FLOAT32) {
                        if (output.dataType == DataType::FLOAT16) {
                            Float32ToFloat16(reduceOutput.data(), ((uint16_t*)output.cpuData) + o * outputDim, outputDim);
                        } else if (output.dataType == DataType::BFLOAT16) {
                            uint16_t *dst = (uint16_t*)output.cpuData + o * outputDim;
                            if (deepSeekV4Mode) {
                                for (int d = 0; d < outputDim; d++) {
                                    dst[d] = Float32ToBFloat16RNEBits(reduceOutput[d]);
                                }
                            } else {
                                Float32ToBFloat16(reduceOutput.data(), dst, outputDim);
                            }
                        }
                    }
                    profileLap(profileOutputMs);
// printf("last spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                }
                if (profileDetail) {
                    double total = profileRegisterMs + profileResizeMs + profileInputMs + profileGatePrepMs +
                                   profileGateMs + profileSwigluConvertMs + profileDownPrepMs + profileDownMs +
                                   profileReduceMs + profileOutputMs;
                    printf("[fastllm-profile-numas-moe] small_batch bs=%d topk=%d experts=%d register=%.3f resize=%.3f input=%.3f gate_prep=%.3f gate_swiglu=%.3f swiglu_convert=%.3f down_prep=%.3f down=%.3f reduce=%.3f output=%.3f total=%.3f\n",
                           bs, topk, profileExpertCalls, profileRegisterMs, profileResizeMs, profileInputMs,
                           profileGatePrepMs, profileGateMs, profileSwigluConvertMs, profileDownPrepMs,
                           profileDownMs, profileReduceMs, profileOutputMs, total);
                    fflush(stdout);
                }
            }
        } else {
            Data gate, attenPart, moePart;
            int bs = input.dims[0];
            int m = weightsBatch / 2 - 1; // num experts
            auto &moeConfig = MoeEnvConfig::GetInstance();
            int expertLimit = moeConfig.GetExpertLimit();
            bool hasExpertLimitOverride = moeConfig.HasExpertLimitOverride();
            bool gpuPrefill = moeConfig.GetGpuPrefill();
#ifdef USE_CUDA
            std::vector<NumasMoeCudaInputReplica> cudaInputReplicas =
                GetNumasMoeCudaInputReplicas(input);
            // Supplement missing TP replicas only when the input already owns
            // at least one valid CUDA mirror.  A CPU-only tensor must not be
            // promoted to GPU merely because process-global MultiCUDA state
            // still contains device ids from an earlier op.
            if (gpuPrefill && input.cpuData != nullptr &&
                !cudaInputReplicas.empty()) {
                std::vector<int> tpDevices;
                std::map<int, int> tpRatios;
                FastllmGetMulticudaDeviceAndRatio(
                    tpDevices, tpRatios, true);
                std::unordered_set<int> presentDevices;
                for (const auto &replica : cudaInputReplicas) {
                    presentDevices.insert(replica.deviceId);
                }
                for (int device : tpDevices) {
                    if (presentDevices.count(device) != 0) {
                        continue;
                    }
                    Data *staged = fastllmMoeDataManagerNumas.
                        StageGpuInputReplica(input, device);
                    cudaInputReplicas.push_back(
                        {device, staged->cudaData});
                    presentDevices.insert(device);
                }
                std::sort(
                    cudaInputReplicas.begin(),
                    cudaInputReplicas.end(),
                    [](const NumasMoeCudaInputReplica &a,
                       const NumasMoeCudaInputReplica &b) {
                        return a.deviceId < b.deviceId;
                    });
            }
            if (cudaInputReplicas.empty()) {
                gpuPrefill = false;
            }
#else
            gpuPrefill = false;
#endif
            if (gpuPrefill && !CanUseCudaMoePrefill(input.dataType, weights, weightsBatch)) {
                gpuPrefill = false;
            }

            if (std::getenv("FASTLLM_NUMAS_MOE_GPU_TRACE") != nullptr) {
                printf(
                    "[Fastllm] NUMA MoE decision layer=%d gpu_prefill=%d "
                    "replicas=%zu\n",
                    layer, gpuPrefill,
#ifdef USE_CUDA
                    cudaInputReplicas.size()
#else
                    (size_t)0
#endif
                );
                fflush(stdout);
            }

            if (!gpuPrefill) {
                expertLimit = INT_MAX; // 不使用GPU时，所有专家都由CPU处理
            }

            // 构建 expertTasks，用于根据 expertLimit 生成专家集合
            std::vector <std::vector <std::pair <int, float> > > expertTasks;
            expertTasks.resize(m + 1);
            for (int b = 0; b < bs; b++) {
                expertTasks[0].push_back(std::make_pair(b, sharedScale));
                for (int j = 0; j < topk; j++) {
                    int expertIdx = indexData[b * topk + j];
                    float value = scoreData[b * topk + j];
                    expertTasks[expertIdx + 1].push_back(std::make_pair(b, value));
                }
            }
// printf("prepare 0 spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
            // Respect an explicit FT_EXPERT_LIMIT override and skip the dynamic
            // CPU/GPU expert split benchmark in that case.
            if (gpuPrefill && !hasExpertLimitOverride) {
#ifdef USE_CUDA
                const NumasMoeCudaInputReplica &profileReplica =
                    cudaInputReplicas.front();
                FastllmCudaSetDevice(profileReplica.deviceId);
                Data profileInput(
                    input.dataType, input.dims,
                    DataDevice::CPU, input.cpuData);
                profileInput.cudaData = profileReplica.cudaData;
                profileInput.cudaDataBorrowed = true;
                profileInput.dataDeviceIds = {profileReplica.deviceId};
                expertLimit = std::min(expertLimit, 
                    MoeExpertSpeedEstimator::GetInstance().GetDynamicExpertLimit(
                        profileInput, output, w1, w2, w3,
                        weights, biass, weightsBatch, topk, sharedScale,
                        expertTasks, expertLimit,
                        (int)cudaInputReplicas.size()
                    )
                );
#endif
            }
// printf("get expertLimit spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
            // 根据 expertLimit 阈值生成 cpuExperts / gpuExperts 集合
            std::unordered_set<int> cpuExperts, gpuExperts;
            for (int e = 0; e < (int)expertTasks.size(); e++) {
                if (weights[e * 2] == nullptr) {
                    continue;
                }
                if ((int)expertTasks[e].size() < expertLimit) {
                    cpuExperts.insert(e);
                } else {
                    gpuExperts.insert(e);
                }
            }
// printf("MoE expertLimit=%d, cpuExperts=%d, gpuExperts=%d\n", expertLimit, (int)cpuExperts.size(), (int)gpuExperts.size());
// printf("prepare 1 spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
            uint8_t *cpuOutputPinned = nullptr;
#ifdef USE_CUDA
            int gpuId = -1;
            std::vector<std::unordered_set<int> > gpuExpertSets;
            std::vector<size_t> gpuExpertLoads;
            std::vector<std::unique_ptr<Data> > gpuInputAliases;
            std::vector<std::unique_ptr<Data> > gpuOutputPartials;
            std::vector<std::thread> gpuThreads;
            void *cpuOutputStaging = nullptr;
            void *cpuOutputCopyStream = nullptr;
            if (gpuPrefill && !gpuExperts.empty()) {
                int gpuWorkerCount = std::min(
                    (int)cudaInputReplicas.size(),
                    (int)gpuExperts.size());
                gpuExpertSets.resize(gpuWorkerCount);
                gpuExpertLoads.assign(gpuWorkerCount, 0);

                std::vector<int> orderedGpuExperts(
                    gpuExperts.begin(), gpuExperts.end());
                std::sort(
                    orderedGpuExperts.begin(), orderedGpuExperts.end(),
                    [&](int a, int b) {
                        if (expertTasks[a].size() != expertTasks[b].size()) {
                            return expertTasks[a].size() >
                                expertTasks[b].size();
                        }
                        return a < b;
                    });
                for (int expert : orderedGpuExperts) {
                    auto loadIt = std::min_element(
                        gpuExpertLoads.begin(), gpuExpertLoads.end());
                    int worker = (int)(loadIt - gpuExpertLoads.begin());
                    gpuExpertSets[worker].insert(expert);
                    gpuExpertLoads[worker] += expertTasks[expert].size();
                }

                gpuId = cudaInputReplicas.front().deviceId;
                FastllmCudaSetDevice(gpuId);
                output.ToDevice(
                    DataDevice::CUDA, std::vector<int>{gpuId}, false);
                output.ToDevice(
                    DataDevice::CPU, std::vector<int>{gpuId}, false);
                AssertInFastLLM(
                    output.cudaData != nullptr &&
                    GetPointerDeviceId(output.cudaData) == gpuId,
                    "NumasMergeMOE failed to allocate the root GPU partial output.");

                gpuInputAliases.reserve(gpuWorkerCount);
                gpuOutputPartials.reserve(gpuWorkerCount);
                for (int i = 0; i < gpuWorkerCount; i++) {
                    const NumasMoeCudaInputReplica &replica =
                        cudaInputReplicas[i];
                    std::unique_ptr<Data> inputAlias(new Data(
                        input.dataType, input.dims,
                        DataDevice::CUDA, replica.cudaData));
                    inputAlias->cudaDataBorrowed = true;
                    inputAlias->dataDeviceIds = {replica.deviceId};
                    inputAlias->strides = input.strides;
                    gpuInputAliases.push_back(std::move(inputAlias));

                    if (i == 0) {
                        std::unique_ptr<Data> outputAlias(new Data(
                            output.dataType, output.dims,
                            DataDevice::CUDA, output.cudaData));
                        outputAlias->cudaDataBorrowed = true;
                        outputAlias->dataDeviceIds = {replica.deviceId};
                        gpuOutputPartials.push_back(std::move(outputAlias));
                    } else {
                        std::unique_ptr<Data> outputPartial(
                            new Data(output.dataType, output.dims));
                        outputPartial->dataDevice = DataDevice::CUDA;
                        outputPartial->dataDeviceIds = {replica.deviceId};
                        FastllmCudaSetDevice(replica.deviceId);
                        outputPartial->Allocate(false);
                        gpuOutputPartials.push_back(
                            std::move(outputPartial));
                    }
                }

                if (std::getenv("FASTLLM_NUMAS_MOE_GPU_TRACE") != nullptr) {
                    size_t cpuRoutes = 0;
                    for (int expert : cpuExperts) {
                        cpuRoutes += expertTasks[expert].size();
                    }
                    printf(
                        "[Fastllm] NUMA MoE prefill layer=%d limit=%d "
                        "cpu_experts=%d cpu_routes=%zu",
                        layer, expertLimit, (int)cpuExperts.size(),
                        cpuRoutes);
                    for (int i = 0; i < gpuWorkerCount; i++) {
                        printf(
                            " gpu%d_experts=%d gpu%d_routes=%zu",
                            cudaInputReplicas[i].deviceId,
                            (int)gpuExpertSets[i].size(),
                            cudaInputReplicas[i].deviceId,
                            gpuExpertLoads[i]);
                    }
                    printf("\n");
                    fflush(stdout);
                }

                gpuThreads.reserve(gpuWorkerCount);
                for (int i = 0; i < gpuWorkerCount; i++) {
                    int workerDevice = cudaInputReplicas[i].deviceId;
                    gpuThreads.emplace_back([&, i, workerDevice]() {
                        FastllmCudaSetDevice(workerDevice);
                        DoCudaMergeMOEFromCPU(
                            *gpuInputAliases[i], *gpuOutputPartials[i],
                            index, score, w1, w2, w3, weights, biass,
                            sharedScale, true, gpuExpertSets[i], true,
                            MoeGateSwiglu, deepSeekV4Mode, swigluLimit);
                    });
                }
            }
// printf("gpu prepare spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
#endif
            if (!cpuExperts.empty()) {
#ifdef USE_CUDA
                if (gpuPrefill && !gpuExperts.empty()) {
                    cpuOutputPinned = fastllmMoeDataManagerNumas.EnsurePinnedOutput(output.GetBytes());
                }
#endif
                DoNumasMergeMOEOnCPU(
                    input, output, index, score, weights, biass,
                    sharedScale, weightsBatch, topk, cpuExperts, fastllmMoeDataManagerNumas,
                    cpuOutputPinned, swigluLimit, deepSeekV4Mode
                );
#ifdef USE_CUDA
                // CPU partial 直接写入 pinned buffer，再异步搬到复用的 GPU staging buffer。
                if (gpuPrefill && !gpuExperts.empty() && cpuOutputPinned != nullptr) {
                    FastllmCudaSetDevice(gpuId);
                    size_t outputBytes = output.GetBytes();
                    cpuOutputStaging = fastllmMoeDataManagerNumas.EnsureGpuOutputStaging(outputBytes, gpuId);
                    cpuOutputCopyStream = fastllmMoeDataManagerNumas.EnsureGpuOutputCopyStream(gpuId);
                    FastllmCudaCopyFromPinnedHostToDeviceAsync(
                        cpuOutputStaging, cpuOutputPinned, outputBytes, cpuOutputCopyStream
                    );
                }
#endif
            }
// printf("cpu spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
#ifdef USE_CUDA
            if (gpuPrefill && !gpuExperts.empty()) {
                for (std::thread &gpuThread : gpuThreads) {
                    gpuThread.join();
                }
                FastllmCudaSetDevice(gpuId);
                Data gpuOutputAlias(
                    output.dataType, output.dims,
                    DataDevice::CUDA, output.cudaData);
                if (cpuOutputStaging != nullptr) {
                    FastllmCudaStreamSynchronize(cpuOutputCopyStream);
                    Data cpuOutputAlias(output.dataType, output.dims, DataDevice::CUDA, cpuOutputStaging);
                    FastllmCudaAddTo(gpuOutputAlias, cpuOutputAlias, 1.0f);
                    FastllmCudaSyncCurrentThreadStream();
                }
                if (gpuOutputPartials.size() > 1) {
                    size_t outputBytes = output.GetBytes();
                    void *reduceStaging =
                        fastllmMoeDataManagerNumas.EnsureGpuOutputStaging(
                            outputBytes, gpuId);
                    Data reduceAlias(
                        output.dataType, output.dims,
                        DataDevice::CUDA, reduceStaging);
                    for (int i = 1;
                         i < (int)gpuOutputPartials.size(); i++) {
                        int sourceDevice =
                            cudaInputReplicas[i].deviceId;
                        FastllmCudaMemcpyBetweenDevices(
                            gpuId, reduceStaging, sourceDevice,
                            gpuOutputPartials[i]->cudaData,
                            outputBytes);
                        FastllmCudaSetDevice(gpuId);
                        FastllmCudaAddTo(
                            gpuOutputAlias, reduceAlias, 1.0f);
                        FastllmCudaSyncCurrentThreadStream();
                    }
                }
                output.dataDevice = DataDevice::CUDA;
                output.dataDeviceIds = {gpuId};
                for (int i = 1;
                     i < (int)gpuOutputPartials.size(); i++) {
                    FastllmCudaSetDevice(
                        cudaInputReplicas[i].deviceId);
                    gpuOutputPartials[i].reset();
                }
                FastllmCudaSetDevice(gpuId);
            }
#endif
// printf("last spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
            return;
        }
    }
}
