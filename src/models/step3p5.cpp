//
// Step-3.5 text model support.
//

#include "step3p5.h"
#include "utils.h"
#include "executor.h"
#include "json11.hpp"
#ifdef USE_CUDA
#include "models/qwen3_cuda_common.h"
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cmath>
#include <climits>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>
#include <tuple>

namespace fastllm {
    static const std::string STEP3P5_BOS = "<｜begin▁of▁sentence｜>";
    static const std::string STEP3P5_IM_START = "<|im_start|>";
    static const std::string STEP3P5_IM_END = "<|im_end|>";

    static bool Step3p5IsTrueString(const std::string &value) {
        std::string lowered = value;
        std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        return lowered == "1" || lowered == "true" || lowered == "on";
    }

    static bool Step3p5DisableFusedMoe() {
        const char *env = getenv("FASTLLM_STEP3P5_DISABLE_FUSED_MOE");
        return env != nullptr && Step3p5IsTrueString(env);
    }

    static bool Step3p7CudaGraphDecodeEnabled() {
        const char *env = getenv("FASTLLM_STEP3P7_CUDA_GRAPH_DECODE");
        return env == nullptr || Step3p5IsTrueString(env);
    }

#ifdef USE_CUDA
    static void Step3p7DebugCudaMemory(const char *tag) {
        const char *env = getenv("FASTLLM_STEP3P7_DEBUG");
        if (env == nullptr || !Step3p5IsTrueString(env)) {
            return;
        }
        auto freeSizes = FastllmCudaGetFreeSizes();
        printf("[Step3.7 debug] %s freeMB:", tag);
        for (int i = 0; i < (int)freeSizes.size(); i++) {
            printf(" %d=%lld", i, (long long)(freeSizes[i] / 1024 / 1024));
        }
        printf("\n");
        fflush(stdout);
    }

    static Executor &Step3p7ThreadLocalVisionExecutor() {
        static thread_local std::unique_ptr<Executor> executor;
        if (executor == nullptr) {
            executor.reset(new Executor());
        }
        return *executor;
    }

    class Step3p7ScopedVisionExecutor {
    public:
        explicit Step3p7ScopedVisionExecutor(int device) : oldExecutor(GetExecutor()) {
            Executor &executor = Step3p7ThreadLocalVisionExecutor();
            if (device >= 0) {
                executor.SetFirstDevice("cuda:" + std::to_string(device));
                FastllmCudaSetDevice(device);
            }
            SetCurrentThreadExecutor(&executor);
        }

        ~Step3p7ScopedVisionExecutor() {
            SetCurrentThreadExecutor(oldExecutor);
        }

        Step3p7ScopedVisionExecutor(const Step3p7ScopedVisionExecutor&) = delete;
        Step3p7ScopedVisionExecutor &operator=(const Step3p7ScopedVisionExecutor&) = delete;

    private:
        void *oldExecutor;
    };
#endif

    static bool Step3p5DeviceMapUsesCuda(const std::map<std::string, int> &deviceMap) {
        for (auto &it : deviceMap) {
            if (it.first.rfind("cuda", 0) == 0 || it.first.rfind("multicuda", 0) == 0) {
                return true;
            }
        }
        return false;
    }

    static bool Step3p5DeviceMapUsesDisk(const std::map<std::string, int> &deviceMap) {
        for (auto &it : deviceMap) {
            if (it.first == "disk") {
                return true;
            }
        }
        return false;
    }

    static std::string Step3p5GetDict(const std::map<std::string, std::string> &dict,
                                      const std::string &key,
                                      const std::string &defaultValue = "") {
        auto it = dict.find(key);
        return it == dict.end() ? defaultValue : it->second;
    }

    static int Step3p5GetInt(const std::map<std::string, std::string> &dict,
                             const std::string &key, int defaultValue) {
        auto it = dict.find(key);
        return it == dict.end() ? defaultValue : atoi(it->second.c_str());
    }

    static float Step3p5GetFloat(const std::map<std::string, std::string> &dict,
                                 const std::string &key, float defaultValue) {
        auto it = dict.find(key);
        return it == dict.end() ? defaultValue : atof(it->second.c_str());
    }

    static std::vector<float> Step3p5ParseFloatList(const std::string &value) {
        std::vector<float> ret;
        if (value.empty()) {
            return ret;
        }
        if (value[0] == '[') {
            std::string error;
            auto arr = json11::Json::parse(value, error);
            if (error.empty() && arr.is_array()) {
                for (auto &item : arr.array_items()) {
                    ret.push_back((float)item.number_value());
                }
            }
            return ret;
        }
        std::stringstream ss(value);
        std::string part;
        while (std::getline(ss, part, ',')) {
            if (!part.empty()) {
                ret.push_back((float)atof(part.c_str()));
            }
        }
        return ret;
    }

    static std::vector<std::string> Step3p5ParseStringList(const std::string &value) {
        std::vector<std::string> ret;
        if (value.empty() || value[0] != '[') {
            return ret;
        }
        std::string error;
        auto arr = json11::Json::parse(value, error);
        if (error.empty() && arr.is_array()) {
            for (auto &item : arr.array_items()) {
                ret.push_back(item.string_value());
            }
        }
        return ret;
    }

    static std::set<int> Step3p5ParseIntSet(const std::string &value) {
        std::set<int> ret;
        if (value.empty()) {
            return ret;
        }
        if (value[0] == '[') {
            std::string error;
            auto arr = json11::Json::parse(value, error);
            if (error.empty() && arr.is_array()) {
                for (auto &item : arr.array_items()) {
                    ret.insert(item.int_value());
                }
            }
            return ret;
        }
        std::stringstream ss(value);
        std::string part;
        while (std::getline(ss, part, ',')) {
            if (!part.empty()) {
                ret.insert(atoi(part.c_str()));
            }
        }
        return ret;
    }

    static void Step3p5FlattenTextConfig(std::map<std::string, std::string> &dicts) {
        std::map<std::string, std::string> extra;
        for (auto &it : dicts) {
            const std::string &key = it.first;
            const std::string prefix = "text_config.";
            if (key.rfind(prefix, 0) != 0) {
                continue;
            }
            std::string stripped = key.substr(prefix.size());
            if (stripped.empty() || dicts.find(stripped) != dicts.end()) {
                continue;
            }
            extra[stripped] = it.second;
        }
        for (auto &it : extra) {
            dicts[it.first] = it.second;
        }
    }

    static bool Step3p7VisionTensorType(const std::string &name, DataType &type) {
        if (name.rfind("vision_model.", 0) != 0 &&
            name.rfind("vit_large_projector", 0) != 0) {
            return false;
        }
        if (name == "vision_model.conv1.weight" ||
            name == "vision_model.vit_downsampler1.weight" ||
            name == "vision_model.vit_downsampler2.weight" ||
            name == "vit_large_projector.weight" ||
            StringEndWith(name, ".attn.in_proj_weight") ||
            StringEndWith(name, ".attn.out_proj.weight") ||
            StringEndWith(name, ".mlp.c_fc.weight") ||
            StringEndWith(name, ".mlp.c_proj.weight")) {
            type = DataType::FLOAT16;
        } else {
            type = DataType::FLOAT32;
        }
        return true;
    }

    static void Step3p5Add1(Data &input) {
        if (input.dims.empty()) {
            return;
        }
        input.ToDevice(DataDevice::CPU);
        if (input.dataType != DataType::FLOAT32) {
            ToDataType(input, DataType::FLOAT32);
            input.ToDevice(DataDevice::CPU);
        }
        float *v = (float*)input.cpuData;
        int len = input.Count(0);
        for (int i = 0; i < len; i++) {
            v[i] += 1.0f;
        }
    }

    static void Step3p5PackFp8MoeWeightToBlock128(Data &weight) {
        if (weight.dataType == DataType::FP8_E4M3_BLOCK_128) {
            return;
        }
        if (weight.dataType != DataType::FP8_E4M3) {
            return;
        }
        AssertInFastLLM(weight.dims.size() == 3, "Step3p5 FusedMOE fp8 weight should be 3D.");
        AssertInFastLLM(weight.blockK > 0 && weight.blockM == 128 && !weight.scales.empty(),
                        "Step3p5 FusedMOE fp8 weight should have block-128 scales.");
        weight.ToDevice(DataDevice::CPU);
        AssertInFastLLM(weight.cpuData != nullptr, "Step3p5 FusedMOE fp8 weight should be in CPU memory before packing.");

        int experts = weight.dims[0];
        int rows = weight.dims[1];
        int cols = weight.dims[2];
        const int blockSize = 128;
        AssertInFastLLM(cols % blockSize == 0,
                        "Step3p5 FusedMOE block128 pack currently requires columns aligned to 128.");
        size_t totalRows = (size_t)experts * rows;
        int scaleRows = (int)((totalRows - 1) / weight.blockK + 1);
        int scaleCols = (cols - 1) / weight.blockM + 1;
        AssertInFastLLM((size_t)scaleRows * scaleCols <= weight.scales.size(),
                        "Step3p5 FusedMOE fp8 scale range is out of bounds.");

        size_t rawBytesPerRow = GetDataBytes(DataType::FP8_E4M3, 1, cols);
        size_t packedBytesPerRow = GetDataBytes(DataType::FP8_E4M3_BLOCK_128, 1, cols);
        size_t packedBytes = totalRows * packedBytesPerRow;
        uint8_t *packed = new uint8_t[packedBytes];
        memset(packed, 0, packedBytes);
        int blocks = (cols - 1) / blockSize + 1;

        int threadNum = (int)std::thread::hardware_concurrency();
        if (const char *env = getenv("FT_THREADS")) {
            threadNum = atoi(env);
        }
        threadNum = std::max(1, std::min({threadNum, 32, (int)totalRows}));
        auto packRows = [&](int rowStart, int rowEnd) {
            for (int globalRowInt = rowStart; globalRowInt < rowEnd; globalRowInt++) {
                size_t globalRow = (size_t)globalRowInt;
                const uint8_t *src = weight.cpuData + globalRow * rawBytesPerRow;
                uint8_t *dst = packed + globalRow * packedBytesPerRow;
                for (int blk = 0; blk < blocks; blk++) {
                    int colStart = blk * blockSize;
                    int colsInBlock = std::min(blockSize, cols - colStart);
                    uint8_t *dstBlock = dst + blk * (blockSize + (int)sizeof(float));
                    memcpy(dstBlock, src + colStart, colsInBlock);
                    size_t scaleIndex = (globalRow / weight.blockK) * scaleCols + blk;
                    memcpy(dstBlock + blockSize, &weight.scales[scaleIndex], sizeof(float));
                }
            }
        };
        if (threadNum == 1) {
            packRows(0, totalRows);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(threadNum);
            for (int t = 0; t < threadNum; t++) {
                int rowStart = (long long)totalRows * t / threadNum;
                int rowEnd = (long long)totalRows * (t + 1) / threadNum;
                threads.emplace_back(packRows, rowStart, rowEnd);
            }
            for (auto &thread : threads) {
                thread.join();
            }
        }

        if (weight.mapFile) {
            weight.mapFile.reset();
        } else {
            delete[] weight.cpuData;
        }
        weight.cpuData = packed;
        weight.dataType = DataType::FP8_E4M3_BLOCK_128;
        weight.scales.clear();
        weight.UpdateUnitSize();
        weight.expansionSize = weight.Count(0);
        weight.expansionBytes = packedBytes;
    }

    static void Step3p5PrepareFusedMoeWeightForCuda(Data &weight, int biasK, int device = -1) {
#ifdef USE_CUDA
        (void)biasK;
        if (device >= 0) {
            FastllmCudaSetDevice(device);
            weight.ToDevice(DataDevice::CUDA, {device}, true);
        } else {
            weight.ToDevice(DataDevice::CUDA);
        }
        if (weight.dataType == DataType::FP8_E4M3 && weight.extraCudaData.empty()) {
            AssertInFastLLM(!weight.scales.empty(), "Step3p5 FusedMOE FP8 weight has no scales.\n");
            float *cudaScales = (float*)FastllmCudaMalloc(weight.scales.size() * sizeof(float));
            FastllmCudaCopyFromHostToDevice(cudaScales, (void*)weight.scales.data(),
                                            weight.scales.size() * sizeof(float));
            weight.extraCudaData.push_back((void*)cudaScales);
        }
#else
        (void)weight;
        (void)biasK;
        (void)device;
#endif
    }

    static void Step3p5MakeExpertView(Data &dst, const Data &src, const std::string &name, int expert) {
        AssertInFastLLM(src.dims.size() == 3, "Step3p5 MoE expert source weight should be 3D.");
        AssertInFastLLM(expert >= 0 && expert < src.dims[0], "Step3p5 MoE expert index out of range.");
        int rows = src.dims[1], cols = src.dims[2];
        dst = Data(src.dataType, {rows, cols});
        dst.name = name;
        dst.weightType = WeightType::LINEAR;
        dst.isModelWeight = true;
        dst.blockK = src.blockK;
        dst.blockM = src.blockM;
        dst.group = src.group;
        dst.groupCnt = src.groupCnt;
        dst.perChannelAxis = src.perChannelAxis;
        if (src.isDiskWeight) {
            dst.isDiskWeight = true;
            dst.dataDevice = DataDevice::CPU;
            dst.cpuData = nullptr;
            dst.diskWeightParts.clear();
            for (auto part : src.diskWeightParts) {
                AssertInFastLLM(part.dims.size() == 3 && part.dims[0] == src.dims[0] &&
                                part.dims[1] == rows && part.dims[2] == cols,
                                "Step3p5 disk MoE expert source part should match the fused 3D tensor.");
                uint64_t bytesPerExpert = part.bytes / part.dims[0];
                part.fileOffset += (long long)bytesPerExpert * expert;
                part.bytes = bytesPerExpert;
                part.dims = {rows, cols};
                dst.diskWeightParts.push_back(part);
            }
        } else {
            dst.FakeFrom(src, (size_t)expert * rows * cols * src.unitSize / src.unitSizeDiv);
        }
        if ((src.dataType == DataType::FP8_E4M3 || src.dataType == DataType::NVFP4) &&
            src.blockK > 0 && src.blockM > 0 && !src.scales.empty()) {
            int ks = (rows - 1) / src.blockK + 1;
            int ms = (cols - 1) / src.blockM + 1;
            int perExpert = ks * ms;
            AssertInFastLLM((expert + 1) * perExpert <= (int)src.scales.size(),
                            "Step3p5 MoE expert scale range is out of bounds.");
            dst.scales.assign(src.scales.begin() + expert * perExpert,
                              src.scales.begin() + (expert + 1) * perExpert);
        }
    }

    static void Step3p5MakeExpertCopy(Data &dst, Data &src, const std::string &name, int expert) {
        AssertInFastLLM(src.dims.size() == 3, "Step3p5 MoE expert source weight should be 3D.");
        AssertInFastLLM(expert >= 0 && expert < src.dims[0], "Step3p5 MoE expert index out of range.");
        src.ToDevice(DataDevice::CPU);
        AssertInFastLLM(src.cpuData != nullptr, "Step3p5 MoE expert source should be in CPU memory.");

        int rows = src.dims[1], cols = src.dims[2];
        dst = Data(src.dataType, {rows, cols});
        dst.name = name;
        dst.weightType = WeightType::LINEAR;
        dst.isModelWeight = true;
        dst.blockK = src.blockK;
        dst.blockM = src.blockM;
        dst.group = src.group;
        dst.groupCnt = src.groupCnt;
        dst.perChannelAxis = src.perChannelAxis;
        dst.Allocate();

        uint64_t bytesPerExpert = src.GetBytes() / src.dims[0];
        memcpy(dst.cpuData, src.cpuData + bytesPerExpert * expert, bytesPerExpert);
        if ((src.dataType == DataType::FP8_E4M3 || src.dataType == DataType::NVFP4) &&
            src.blockK > 0 && src.blockM > 0 && !src.scales.empty()) {
            int ks = (rows - 1) / src.blockK + 1;
            int ms = (cols - 1) / src.blockM + 1;
            int perExpert = ks * ms;
            AssertInFastLLM((expert + 1) * perExpert <= (int)src.scales.size(),
                            "Step3p5 MoE expert scale range is out of bounds.");
            dst.scales.assign(src.scales.begin() + expert * perExpert,
                              src.scales.begin() + (expert + 1) * perExpert);
        }
    }

    static void Step3p5MakeExpertRangeCopy(
            Data &dst, Data &src, const std::string &name, int expertStart, int expertEnd) {
        AssertInFastLLM(src.dims.size() == 3, "Step3p5 MoE expert source weight should be 3D.");
        AssertInFastLLM(expertStart >= 0 && expertStart < expertEnd && expertEnd <= src.dims[0],
                        "Step3p5 MoE expert range is out of bounds.");
        src.ToDevice(DataDevice::CPU);
        AssertInFastLLM(src.cpuData != nullptr, "Step3p5 MoE expert source should be in CPU memory.");

        int experts = expertEnd - expertStart;
        int rows = src.dims[1], cols = src.dims[2];
        dst = Data(src.dataType, {experts, rows, cols});
        dst.name = name;
        dst.weightType = WeightType::LINEAR;
        dst.isModelWeight = true;
        dst.blockK = src.blockK;
        dst.blockM = src.blockM;
        dst.group = src.group;
        dst.groupCnt = src.groupCnt;
        dst.perChannelAxis = src.perChannelAxis;
        dst.Allocate();

        uint64_t bytesPerExpert = src.GetBytes() / src.dims[0];
        memcpy(dst.cpuData, src.cpuData + bytesPerExpert * expertStart,
               bytesPerExpert * experts);
        if ((src.dataType == DataType::FP8_E4M3 || src.dataType == DataType::NVFP4) &&
            src.blockK > 0 && src.blockM > 0 && !src.scales.empty()) {
            int ks = (rows - 1) / src.blockK + 1;
            int ms = (cols - 1) / src.blockM + 1;
            int perExpert = ks * ms;
            AssertInFastLLM(expertEnd * perExpert <= (int)src.scales.size(),
                            "Step3p5 MoE expert scale range is out of bounds.");
            dst.scales.assign(src.scales.begin() + expertStart * perExpert,
                              src.scales.begin() + expertEnd * perExpert);
        }
    }

    static void Step3p5MakeGateUpWeight(Data &dst, const Data &gate, const Data &up, const std::string &name) {
        AssertInFastLLM(gate.dims.size() == 2 && up.dims.size() == 2 &&
                        gate.dims[0] == up.dims[0] && gate.dims[1] == up.dims[1],
                        "Step3p5 MoE gate/up expert weights should have the same 2D shape.");
        AssertInFastLLM(gate.dataType == up.dataType,
                        "Step3p5 MoE gate/up expert weights should have the same dtype.");
        dst = Data(gate.dataType, {gate.dims[0] + up.dims[0], gate.dims[1]});
        dst.name = name;
        dst.weightType = WeightType::LINEAR;
        dst.isModelWeight = true;
        dst.blockK = gate.blockK;
        dst.blockM = gate.blockM;
        dst.group = gate.group;
        dst.groupCnt = gate.groupCnt;
        dst.perChannelAxis = gate.perChannelAxis;
        if (gate.isDiskWeight || up.isDiskWeight) {
            AssertInFastLLM(gate.isDiskWeight && up.isDiskWeight,
                            "Step3p5 disk MoE gate/up weights should both be disk weights.");
            dst.isDiskWeight = true;
            dst.dataDevice = DataDevice::CPU;
            dst.cpuData = nullptr;
            dst.diskWeightParts = gate.diskWeightParts;
            dst.diskWeightParts.insert(dst.diskWeightParts.end(),
                                       up.diskWeightParts.begin(), up.diskWeightParts.end());
        } else {
            dst.Allocate();
            memcpy(dst.cpuData, gate.cpuData, gate.GetBytes());
            memcpy(dst.cpuData + gate.GetBytes(), up.cpuData, up.GetBytes());
        }
        dst.perChannelsConfigs = AppendVector(dst.perChannelsConfigs, gate.perChannelsConfigs);
        dst.perChannelsConfigs = AppendVector(dst.perChannelsConfigs, up.perChannelsConfigs);
        dst.zeros = AppendVector(dst.zeros, gate.zeros);
        dst.zeros = AppendVector(dst.zeros, up.zeros);
        dst.scales = AppendVector(dst.scales, gate.scales);
        dst.scales = AppendVector(dst.scales, up.scales);
        dst.mins = AppendVector(dst.mins, gate.mins);
        dst.mins = AppendVector(dst.mins, up.mins);
        dst.halfScales = AppendVector(dst.halfScales, gate.halfScales);
        dst.halfScales = AppendVector(dst.halfScales, up.halfScales);
        if (!dst.isDiskWeight) {
            dst.CalcWeightSum();
        }
    }

    static void Step3p5MakeGateUpSliceView(Data &dst, const Data &src, const std::string &name,
                                           int rowStart, int rows) {
        AssertInFastLLM(src.dims.size() == 2, "Step3p5 MoE gateup weight should be 2D.");
        AssertInFastLLM(rowStart >= 0 && rows > 0 && rowStart + rows <= src.dims[0],
                        "Step3p5 MoE gateup slice is out of range.");
        int cols = src.dims[1];
        dst = Data(src.dataType, {rows, cols});
        dst.name = name;
        dst.weightType = WeightType::LINEAR;
        dst.isModelWeight = true;
        dst.blockK = src.blockK;
        dst.blockM = src.blockM;
        dst.group = src.group;
        dst.groupCnt = src.groupCnt;
        dst.perChannelAxis = src.perChannelAxis;
        dst.FakeFrom(src, (uint64_t)rowStart * cols * src.unitSize / src.unitSizeDiv);
        if ((src.dataType == DataType::FP8_E4M3 || src.dataType == DataType::NVFP4) &&
            src.blockK > 0 && src.blockM > 0 && !src.scales.empty()) {
            AssertInFastLLM(rowStart % src.blockK == 0,
                            "Step3p5 MoE gateup scale slice should align with blockK.");
            int ks = (rows - 1) / src.blockK + 1;
            int ms = (cols - 1) / src.blockM + 1;
            int scaleOffset = (rowStart / src.blockK) * ms;
            int scaleCount = ks * ms;
            AssertInFastLLM(scaleOffset + scaleCount <= (int)src.scales.size(),
                            "Step3p5 MoE gateup scale slice is out of bounds.");
            dst.scales.assign(src.scales.begin() + scaleOffset,
                              src.scales.begin() + scaleOffset + scaleCount);
        }
    }

    static float Step3p5LayerLimit(const std::vector<float> &limits, int layer) {
        return layer >= 0 && layer < (int)limits.size() ? limits[layer] : 0.0f;
    }

    static float Step3p5ReadFloat(const Data &input, int index) {
        if (input.dataType == DataType::FLOAT32) {
            return ((float*)input.cpuData)[index];
        }
        uint16_t v = ((uint16_t*)input.cpuData)[index];
        if (input.dataType == DataType::FLOAT16) {
            return half_to_float(v);
        }
        if (input.dataType == DataType::BFLOAT16) {
            uint32_t raw = ((uint32_t)v) << 16;
            float ret;
            memcpy(&ret, &raw, sizeof(ret));
            return ret;
        }
        ErrorInFastLLM("Step3p5ReadFloat: unsupported data type.");
        return 0.0f;
    }

    static uint16_t Step3p5FloatToBFloat16(float value) {
        uint32_t raw;
        memcpy(&raw, &value, sizeof(raw));
        uint32_t rounding = ((raw >> 16) & 1) + 0x7FFF;
        return (uint16_t)((raw + rounding) >> 16);
    }

    static void Step3p5WriteFloat(Data &input, int index, float value) {
        if (input.dataType == DataType::FLOAT32) {
            ((float*)input.cpuData)[index] = value;
            return;
        }
        if (input.dataType == DataType::FLOAT16) {
            ((uint16_t*)input.cpuData)[index] = float_to_half(value);
            return;
        }
        if (input.dataType == DataType::BFLOAT16) {
            ((uint16_t*)input.cpuData)[index] = Step3p5FloatToBFloat16(value);
            return;
        }
        ErrorInFastLLM("Step3p5WriteFloat: unsupported data type.");
    }

    static void Step3p5Clamp(Data &input, bool hasMin, float minValue, bool hasMax, float maxValue) {
        DataDevice originalDevice = input.dataDevice;
        std::vector<int> originalDeviceIds = input.dataDeviceIds;
        input.ToDevice(DataDevice::CPU);
        int len = input.Count(0);
        for (int i = 0; i < len; i++) {
            float value = Step3p5ReadFloat(input, i);
            if (hasMin && value < minValue) {
                value = minValue;
            }
            if (hasMax && value > maxValue) {
                value = maxValue;
            }
            Step3p5WriteFloat(input, i, value);
        }
        input.ToDevice(originalDevice, originalDeviceIds);
    }

    static DataType Step3p5ResolvePagedAttentionQType(DataType cacheType, DataType queryType, DataType modelType) {
        if (cacheType == DataType::FLOAT16 || cacheType == DataType::BFLOAT16) {
            return cacheType;
        }
        if (queryType == DataType::FLOAT16 || queryType == DataType::BFLOAT16) {
            return queryType;
        }
        return modelType == DataType::BFLOAT16 ? DataType::BFLOAT16 : DataType::FLOAT16;
    }

    static Data &Step3p5PreparePagedAttentionQ(Data &src, DataType cacheType, DataType modelType, Data &casted) {
        DataType targetType = Step3p5ResolvePagedAttentionQType(cacheType, src.dataType, modelType);
        if (src.dataType == targetType) {
            return src;
        }
        ToDataType(src, casted, targetType);
        return casted;
    }

#ifdef USE_CUDA
    namespace {
        using namespace qwen3cuda;

        static std::atomic<int> step3p5ThreadTpNextPagedCacheBase(4000000);

        static std::string Step3p5TrimString(const std::string &s) {
            int l = 0, r = (int)s.size();
            while (l < r && std::isspace((unsigned char)s[l])) {
                l++;
            }
            while (r > l && std::isspace((unsigned char)s[r - 1])) {
                r--;
            }
            return s.substr(l, r - l);
        }

        static bool Step3p5IsDisabledTpSpec(const std::string &value) {
            std::string v = value;
            std::transform(v.begin(), v.end(), v.begin(),
                           [](unsigned char c) { return (char)std::tolower(c); });
            return v.empty() || v == "false" || v == "off" || v == "none" || v == "disable";
        }

        static bool Step3p5AppendCudaDevicesFromSpec(const std::string &spec,
                                                     const std::string &type,
                                                     int defaultRatio,
                                                     std::vector<int> &devices,
                                                     std::map<int, int> &ratios) {
            std::map<int, int> parsedRatios;
            std::vector<int> parsed = ParseDeviceIds(spec, type, parsedRatios);
            if (parsed.empty() && (spec == "cuda" || spec == "multicuda")) {
                parsed.push_back(0);
            }
            bool added = false;
            for (int device : parsed) {
                if (device < 0 || device == 99999) {
                    continue;
                }
                int ratio = defaultRatio;
                auto ratioIt = parsedRatios.find(device);
                if (ratioIt != parsedRatios.end() && ratioIt->second > 0) {
                    ratio = ratioIt->second;
                }
                if (ratio <= 0) {
                    ratio = 1;
                }
                if (std::find(devices.begin(), devices.end(), device) == devices.end()) {
                    devices.push_back(device);
                }
                ratios[device] += ratio;
                added = true;
            }
            return added;
        }

        static bool Step3p5ParseGPUForwardSpec(const std::string &rawSpec,
                                               std::vector<int> &devices,
                                               std::map<int, int> &ratios) {
            std::string spec = Step3p5TrimString(rawSpec);
            if (Step3p5IsDisabledTpSpec(spec)) {
                return false;
            }

            std::string lower = spec;
            std::transform(lower.begin(), lower.end(), lower.begin(),
                           [](unsigned char c) { return (char)std::tolower(c); });
            if (lower == "1" || lower == "true" || lower == "on" || lower == "auto") {
                int count = FastllmCudaGetDeviceCount();
                for (int i = 0; i < count; i++) {
                    devices.push_back(i);
                    ratios[i] = 1;
                }
                return !devices.empty();
            }

            std::string parseSpec = spec;
            std::string type = "cuda";
            if (StartWith(lower, "multicuda")) {
                type = "multicuda";
            } else if (!StartWith(lower, "cuda")) {
                parseSpec = "cuda:" + spec;
            }
            return Step3p5AppendCudaDevicesFromSpec(parseSpec, type, 1, devices, ratios);
        }

        static bool GetStep3p5GPUForwardDevices(const std::map<std::string, int> &deviceMap,
                                                std::vector<int> &devices,
                                                std::map<int, int> &ratios) {
            devices.clear();
            ratios.clear();
            const char *env = std::getenv("FASTLLM_TP");
            if (env == nullptr || Step3p5IsDisabledTpSpec(Step3p5TrimString(env))) {
                env = std::getenv("FASTLLM_STEP3P5_TP");
            }
            if (env == nullptr || Step3p5IsDisabledTpSpec(Step3p5TrimString(env))) {
                env = std::getenv("FASTLLM_QWEN3_THREAD_TP");
            }
            if (env != nullptr) {
                Step3p5ParseGPUForwardSpec(env, devices, ratios);
            }

            if (devices.empty() && env == nullptr) {
                std::string onlyDevice;
                int onlyRatio = 0;
                int activeDeviceEntries = 0;
                for (auto &it : deviceMap) {
                    if (it.second <= 0) {
                        continue;
                    }
                    activeDeviceEntries++;
                    onlyDevice = it.first;
                    onlyRatio = it.second;
                }

                if (activeDeviceEntries == 1) {
                    std::string lower = onlyDevice;
                    std::transform(lower.begin(), lower.end(), lower.begin(),
                                   [](unsigned char c) { return (char)std::tolower(c); });
                    if (StartWith(lower, "multicuda")) {
                        Step3p5AppendCudaDevicesFromSpec(onlyDevice, "multicuda", onlyRatio, devices, ratios);
                    } else if (StartWith(lower, "cuda")) {
                        Step3p5AppendCudaDevicesFromSpec(onlyDevice, "cuda", onlyRatio, devices, ratios);
                    }
                }
            }

            std::vector<int> uniqueDevices;
            std::set<int> seen;
            for (int device : devices) {
                if (device >= 0 && seen.insert(device).second) {
                    uniqueDevices.push_back(device);
                    if (ratios.find(device) == ratios.end() || ratios[device] <= 0) {
                        ratios[device] = 1;
                    }
                }
            }
            devices.swap(uniqueDevices);
            return !devices.empty();
        }

        static int Step3p7ResolveVisionDevice(const std::map<std::string, int> &deviceMap) {
            std::vector<int> devices;
            std::map<int, int> ratios;
            if (GetStep3p5GPUForwardDevices(deviceMap, devices, ratios) && !devices.empty()) {
                return devices[0];
            }

            int bestDevice = -1;
            for (auto &it : deviceMap) {
                if (it.second <= 0) {
                    continue;
                }
                std::string lower = it.first;
                std::transform(lower.begin(), lower.end(), lower.begin(),
                               [](unsigned char c) { return (char)std::tolower(c); });
                std::string type;
                if (StartWith(lower, "multicuda")) {
                    type = "multicuda";
                } else if (StartWith(lower, "cuda")) {
                    type = "cuda";
                } else {
                    continue;
                }

                std::vector<int> parsed;
                std::map<int, int> parsedRatios;
                Step3p5AppendCudaDevicesFromSpec(it.first, type, it.second, parsed, parsedRatios);
                for (int device : parsed) {
                    if (device >= 0 && (bestDevice < 0 || device < bestDevice)) {
                        bestDevice = device;
                    }
                }
            }
            return bestDevice;
        }

        static bool GetStep3p5ThreadTpDevices(const std::map<std::string, int> &deviceMap,
                                              std::vector<int> &devices,
                                              std::map<int, int> &ratios) {
            if (!GetStep3p5GPUForwardDevices(deviceMap, devices, ratios)) {
                return false;
            }
            return devices.size() > 1;
        }

        static DivisionScheme ExtractStep3p5FirstRangeScheme(const DivisionScheme &scheme) {
            DivisionScheme ret;
            for (auto &it : scheme) {
                ret[it.first];
                if (!it.second.empty()) {
                    ret[it.first].push_back(it.second[0]);
                }
            }
            return ret;
        }

        static DivisionScheme ExtractStep3p5KVHeadScheme(const DivisionScheme &qkvScheme,
                                                         int qWidth, int headDim) {
            DivisionScheme ret;
            for (auto &it : qkvScheme) {
                ret[it.first];
                if (it.second.size() < 2) {
                    continue;
                }
                int st = (it.second[1].first - qWidth) / headDim;
                int end = (it.second[1].second - qWidth) / headDim;
                ret[it.first].push_back({st, end});
            }
            return ret;
        }

        static DivisionScheme ExtractStep3p5QHeadScheme(const DivisionScheme &qScheme, int headDim) {
            DivisionScheme ret;
            for (auto &it : qScheme) {
                ret[it.first];
                for (auto &range : it.second) {
                    AssertInFastLLM(range.first % headDim == 0 && range.second % headDim == 0,
                                    "Step3p5 ForwardGPU got unaligned q-head split range.\n");
                    ret[it.first].push_back({range.first / headDim, range.second / headDim});
                }
            }
            return ret;
        }

        static DivisionScheme BuildStep3p5ExpertScheme(const std::vector<int> &devices,
                                                       const std::map<int, int> &ratios,
                                                       int numExperts) {
            DivisionScheme scheme;
            if (devices.empty() || numExperts <= 0) {
                return scheme;
            }
            int totalRatio = 0;
            for (int device : devices) {
                auto it = ratios.find(device);
                totalRatio += (it == ratios.end() || it->second <= 0) ? 1 : it->second;
            }
            int acc = 0;
            int start = 0;
            for (int i = 0; i < (int)devices.size(); i++) {
                int device = devices[i];
                int ratio = 1;
                auto ratioIt = ratios.find(device);
                if (ratioIt != ratios.end() && ratioIt->second > 0) {
                    ratio = ratioIt->second;
                }
                acc += ratio;
                int end = (i + 1 == (int)devices.size()) ? numExperts :
                          (int)((long long)numExperts * acc / totalRatio);
                if (end > start) {
                    scheme[device].push_back({start, end});
                } else {
                    scheme[device];
                }
                start = end;
            }
            return scheme;
        }

        static bool Step3p5ExpertOnDevice(const DivisionScheme &scheme, int device, int expert) {
            auto it = scheme.find(device);
            if (it == scheme.end()) {
                return false;
            }
            for (auto &range : it->second) {
                if (expert >= range.first && expert < range.second) {
                    return true;
                }
            }
            return false;
        }

        static DataType ResolveStep3p5ThreadTpComputeType(DataType modelType) {
            if (modelType == DataType::FLOAT32) {
                return DataType::FLOAT32;
            }
            return DataType::FLOAT16;
        }

        static DataType ResolveStep3p5ThreadTpCacheType(DataType cacheType, DataType computeType) {
            if (cacheType == DataType::FP8_E4M3) {
                return cacheType;
            }
            return computeType;
        }

        static void PrepareStep3p5EmbeddingWeightType(Data &embedWeight,
                                                      DataType outputType,
                                                      bool requireCpu) {
            if (requireCpu || embedWeight.dataType != outputType) {
                if (embedWeight.multiDeviceData) {
                    embedWeight.ResetMultiDeviceState();
                }
                if (embedWeight.dataDevice != DataDevice::CPU) {
                    embedWeight.ToDevice(DataDevice::CPU);
                }
            }
            if (embedWeight.dataType != outputType) {
                ToDataTypeForceCPU(embedWeight, outputType);
            }
        }

        static void Step3p5CpuEmbeddingDirect(Data &inputIds, Data &embedWeight,
                                              Data &hiddenStates, DataType outputType) {
            PrepareStep3p5EmbeddingWeightType(embedWeight, outputType, true);
            inputIds.ToDevice(DataDevice::CPU);
            Executor *executor = (Executor*)GetExecutor();
            executor->RunOnDevice("cpu", "EmbeddingDirect",
                                  DataDict{{"input", &inputIds},
                                           {"weight", &embedWeight},
                                           {"output", &hiddenStates}},
                                  FloatDict(), IntDict());
        }

        static void PrepareStep3p5CudaEmbeddingWeightType(Data &embedWeight,
                                                          DataType outputType) {
            if (embedWeight.dataType != outputType) {
                embedWeight.ResetMultiDeviceState();
                if (embedWeight.dataDevice != DataDevice::CPU) {
                    embedWeight.ToDevice(DataDevice::CPU);
                }
                ToDataTypeForceCPU(embedWeight, outputType);
            }
        }

        static Data *CreateStep3p5CudaReplicaLike(const Data &source, int device) {
            Data *local = new Data(source.dataType);
            local->Resize(source.dims);
            local->dataDevice = DataDevice::CUDA;
            local->dataDeviceIds = {device};
            FastllmCudaSetDevice(device);
            local->Allocate(false);
            return local;
        }

        static void PrepareStep3p5CpuEmbeddingHiddenStates(Data &hiddenStates,
                                                           const std::vector<int> &devices,
                                                           PersistentWorkerGroup &workerGroup) {
            AssertInFastLLM(!devices.empty(),
                            "Step3p5 ForwardGPU CPU embedding got empty CUDA devices.\n");
            hiddenStates.ToDevice(DataDevice::CPU);
            AssertInFastLLM(hiddenStates.cpuData != nullptr,
                            "Step3p5 ForwardGPU CPU embedding has no CPU data.\n");
            if (devices.size() == 1) {
                hiddenStates.ToDevice(DataDevice::CUDA, {devices[0]}, true);
                return;
            }

            uint64_t count = hiddenStates.Count(0);
            AssertInFastLLM(count <= (uint64_t)INT_MAX,
                            "Step3p5 ForwardGPU CPU embedding result is too large for NCCL broadcast.\n");
            hiddenStates.ResetMultiDeviceState();
            hiddenStates.multiDeviceData = true;
            hiddenStates.tpLayout = TP_LAYOUT_REPLICATED;
            hiddenStates.tpAxis = -1;
            hiddenStates.tpGlobalDims = hiddenStates.dims;
            hiddenStates.dataDevice = DataDevice::CUDA;
            hiddenStates.dataDeviceIds = devices;

            int rootDevice = devices[0];
            for (int device : devices) {
                hiddenStates.multiDeviceDatas[device] =
                    CreateStep3p5CudaReplicaLike(hiddenStates, device);
            }
            FastllmCudaSetDevice(rootDevice);
            FastllmCudaCopyFromHostToDevice(hiddenStates.multiDeviceDatas[rootDevice]->cudaData,
                                            hiddenStates.cpuData,
                                            hiddenStates.GetBytes());

            std::vector<std::exception_ptr> errors(devices.size());
            workerGroup.Run(devices, [&](int r) {
                int device = devices[r];
                auto it = hiddenStates.multiDeviceDatas.find(device);
                AssertInFastLLM(it != hiddenStates.multiDeviceDatas.end() && it->second != nullptr,
                                "Step3p5 ForwardGPU CPU embedding missing local CUDA replica.\n");
                FastllmCudaSetDevice(device);
                FastllmNcclBroadcast(it->second->cudaData, (int)count,
                                     (int)hiddenStates.dataType,
                                     rootDevice, device);
                ForceDeviceSync();
            }, errors);
            for (auto &error : errors) {
                if (error) {
                    std::rethrow_exception(error);
                }
            }
        }

        static Data *EnsureStep3p5ThreadTpLocalCache(Data &root, int device, DataType localDataType) {
            root.multiDeviceData = true;
            root.dataDevice = DataDevice::CUDA;
            auto &local = root.multiDeviceDatas[device];
            if (local == nullptr) {
                local = new Data(localDataType);
                local->SetKVCache();
                local->cacheUid = root.cacheUid;
                local->dataDevice = DataDevice::CUDA;
                local->dataDeviceIds = {device};
            } else if (local->dataType != localDataType && local->dims.empty()) {
                local->dataType = localDataType;
                local->UpdateUnitSize();
            }
            return local;
        }

        static void PrepareStep3p5SingleCudaCache(Data &cache, int device, DataType localDataType) {
            cache.isKVCache = true;
            cache.lockInCPU = false;
            if (cache.dataType != localDataType && cache.dims.empty()) {
                cache.dataType = localDataType;
                cache.UpdateUnitSize();
            }
            cache.ToDevice(DataDevice::CUDA, {device}, false);
        }

        static void SyncStep3p5ThreadTpRootCacheMetaFromLocal(Data &root,
                                                              Data *firstLocal,
                                                              const std::vector<int> &devices,
                                                              const DivisionScheme &kvHeadScheme,
                                                              int globalKVHeads,
                                                              int headDim) {
            if (firstLocal == nullptr) {
                return;
            }
            if (firstLocal->dims.size() < 3) {
                return;
            }

            std::vector<int> globalDims = {globalKVHeads, firstLocal->dims[1], headDim};
            bool samePageIndex = root.pageIndex.size() == firstLocal->pageIndex.size() &&
                (root.pageIndex.empty() || root.pageIndex.back() == firstLocal->pageIndex.back());
            bool fastMetaUpdate =
                root.multiDeviceData &&
                root.dataDevice == DataDevice::CUDA &&
                root.tpLayout == TP_LAYOUT_SHARDED &&
                root.tpAxis == 0 &&
                root.isKVCache &&
                root.isPagedKVCache == firstLocal->isPagedKVCache &&
                root.pageLen == firstLocal->pageLen &&
                root.pagedKVCacheData == firstLocal->pagedKVCacheData &&
                root.tpGlobalDims == globalDims &&
                root.dims == globalDims &&
                samePageIndex;
            if (fastMetaUpdate) {
                root.lastPageLen = firstLocal->lastPageLen;
                return;
            }

            root.dataType = firstLocal->dataType;
            root.UpdateUnitSize();
            root.dataDevice = DataDevice::CUDA;
            root.multiDeviceData = true;
            root.dataDeviceIds = devices;
            root.tpLayout = TP_LAYOUT_SHARDED;
            root.tpAxis = 0;
            root.tpRanges = kvHeadScheme;
            root.tpGlobalDims = globalDims;
            if (root.dims != globalDims) {
                root.Resize(globalDims);
            }
            root.cudaData = nullptr;
            root.isKVCache = true;
            root.isPagedKVCache = firstLocal->isPagedKVCache;
            root.pageLen = firstLocal->pageLen;
            root.pageIndex = firstLocal->pageIndex;
            root.lastPageLen = firstLocal->lastPageLen;
            root.pagedKVCacheData = firstLocal->pagedKVCacheData;
        }

        static void SyncStep3p5ThreadTpRootCacheMeta(Data &root,
                                                     const std::vector<int> &devices,
                                                     const DivisionScheme &kvHeadScheme,
                                                     int globalKVHeads,
                                                     int headDim) {
            if (devices.empty()) {
                return;
            }
            Data *firstLocal = nullptr;
            for (int device : devices) {
                auto it = root.multiDeviceDatas.find(device);
                if (it != root.multiDeviceDatas.end() && it->second != nullptr &&
                    it->second->dims.size() >= 3) {
                    firstLocal = it->second;
                    break;
                }
            }
            SyncStep3p5ThreadTpRootCacheMetaFromLocal(root, firstLocal, devices,
                                                      kvHeadScheme, globalKVHeads, headDim);
        }

        static bool Step3p5NeedRepeatPenalty(const GenerationConfig &config) {
            float diff = config.repeat_penalty - 1.0f;
            return diff > 1e-6f || diff < -1e-6f;
        }

        static bool Step3p5CanUseCudaFullLogitsSampling(
                const std::vector<GenerationConfig> &generationConfigs,
                std::vector<std::vector<float>*> *retLogits,
                int batch,
                bool &allSimple,
                int &maxTopK) {
            allSimple = true;
            maxTopK = 1;
            for (int b = 0; b < batch; b++) {
                const GenerationConfig &config = generationConfigs[b];
                allSimple &= config.IsSimpleGreedy();
                if (config.output_logits && retLogits != nullptr &&
                    b < (int)retLogits->size() && (*retLogits)[b] != nullptr) {
                    return false;
                }
                if (Step3p5NeedRepeatPenalty(config)) {
                    return false;
                }
                int curTopK = config.IsSimpleGreedy() ? 1 : config.top_k;
                if (curTopK <= 0 || curTopK > 50) {
                    return false;
                }
                maxTopK = std::max(maxTopK, curTopK);
            }
            return true;
        }

        static Data &Step3p5ThreadLocalCudaSamplingFullLogits() {
            static thread_local Data fullLogits(DataType::FLOAT32);
            return fullLogits;
        }

        static Data &Step3p5ThreadLocalCudaSamplingOutput() {
            static thread_local Data data(DataType::INT32);
            return data;
        }

        static void Step3p5GatherShardLogitsToRootCuda(
                int rootDevice,
                const std::vector<int> &devices,
                const DivisionScheme &lmHeadScheme,
                std::vector<Data> &localLogits,
                int batch,
                int vocabSize,
                Data &fullLogits) {
            FastllmCudaSetDevice(rootDevice);
            Qwen3CudaPrepareLocalOutput(fullLogits, rootDevice);
            fullLogits.dataType = DataType::FLOAT32;
            fullLogits.UpdateUnitSize();
            fullLogits.Resize({batch, vocabSize});
            fullLogits.Allocate();

            for (int r = 0; r < (int)devices.size(); r++) {
                int device = devices[r];
                auto schemeIt = lmHeadScheme.find(device);
                AssertInFastLLM(schemeIt != lmHeadScheme.end(),
                                "Step3p5 CUDA sampling: missing lm_head split range.\n");
                AssertInFastLLM(localLogits[r].dataDevice == DataDevice::CUDA &&
                                localLogits[r].cudaData != nullptr,
                                "Step3p5 CUDA sampling: local logits must stay on CUDA.\n");
                int localVocab = localLogits[r].dims.back();
                int rows = localLogits[r].Count(0) / localVocab;
                AssertInFastLLM(rows == batch,
                                "Step3p5 CUDA sampling: local logits batch mismatch.\n");

                uint8_t *dstBase = (uint8_t*)fullLogits.cudaData;
                uint8_t *srcBase = (uint8_t*)localLogits[r].cudaData;
                int localOffset = 0;
                for (auto &range : schemeIt->second) {
                    int len = range.second - range.first;
                    AssertInFastLLM(range.first >= 0 && range.second <= vocabSize &&
                                    localOffset + len <= localVocab,
                                    "Step3p5 CUDA sampling: invalid lm_head split range.\n");
                    FastllmCudaMemcpy2DDeviceToDeviceAuto(
                        dstBase + (size_t)range.first * sizeof(float),
                        (size_t)vocabSize * sizeof(float),
                        srcBase + (size_t)localOffset * sizeof(float),
                        (size_t)localVocab * sizeof(float),
                        (size_t)len * sizeof(float),
                        (size_t)batch,
                        rootDevice,
                        device);
                    localOffset += len;
                }
            }
        }

        static std::vector<int> Step3p5SampleFromRootCudaLogits(
                int rootDevice,
                Data &fullLogits,
                int batch,
                int maxTopK,
                bool allSimple,
                const std::vector<GenerationConfig> &generationConfigs) {
            FastllmCudaSetDevice(rootDevice);
            std::vector<int> lastRet;
            lastRet.reserve(batch);
            if (!allSimple) {
                static thread_local std::vector<float> temperatures;
                static thread_local std::vector<int> topKs;
                static thread_local std::vector<float> topPs;
                temperatures.resize(batch);
                topKs.resize(batch);
                topPs.resize(batch);
                for (int b = 0; b < batch; b++) {
                    temperatures[b] = std::max(generationConfigs[b].temperature, 1.0e-6f);
                    topKs[b] = generationConfigs[b].top_k;
                    topPs[b] = generationConfigs[b].top_p;
                }
                lastRet.resize(batch);
                int vocabSize = fullLogits.dims.back();
                FastllmCudaTopKTopPSampling((float*)fullLogits.cudaData,
                                            temperatures.data(), topKs.data(), topPs.data(),
                                            lastRet.data(), batch, vocabSize);
                return lastRet;
            }

            Data &cudaOutput = Step3p5ThreadLocalCudaSamplingOutput();
            Qwen3CudaPrepareLocalOutput(cudaOutput, rootDevice);
            cudaOutput.dataType = DataType::INT32;
            cudaOutput.UpdateUnitSize();
            cudaOutput.Resize({batch});
            cudaOutput.Allocate();

            int vocabSize = fullLogits.dims.back();
            FastllmCudaGreedySampling((float*)fullLogits.cudaData,
                                      (int*)cudaOutput.cudaData,
                                      batch, vocabSize);
            lastRet.resize(batch);
            FastllmCudaCopyFromDeviceToHost(lastRet.data(), cudaOutput.cudaData,
                                            (size_t)batch * sizeof(int));
            return lastRet;
        }

        static void Step3p5ZeroCudaLike(Data &dst, const Data &like, int device) {
            bool needReset = dst.isFake || dst.dataType != like.dataType ||
                             dst.dataDevice != DataDevice::CUDA || dst.dims != like.dims ||
                             (!dst.dataDeviceIds.empty() && dst.dataDeviceIds[0] != device);
            if (!needReset && dst.cudaData != nullptr) {
                int ptrDevice = GetPointerDeviceId(dst.cudaData);
                needReset = ptrDevice >= 0 && ptrDevice != device;
            }
            if (needReset) {
                if (!dst.isFake) {
                    dst.FreeSpace();
                } else {
                    dst.isFake = false;
                    dst.cpuData = nullptr;
                    dst.cudaData = nullptr;
                    dst.deviceData = nullptr;
                    dst.expansionSize = 0;
                    dst.expansionBytes = 0;
                }
                Qwen3CudaClearMultiDeviceState(dst);
                dst.dataType = like.dataType;
                dst.UpdateUnitSize();
                dst.dataDevice = DataDevice::CUDA;
                dst.dataDeviceIds = {device};
                dst.Resize(like.dims);
            }
            dst.Allocate();
            if (dst.cudaData != nullptr) {
                FastllmCudaMemset0(dst.cudaData, dst.GetBytes());
            }
        }

        struct Step3p5ForwardSingleBuffers {
            Data hiddenStates;
            Data attenInput;
            Data qkv;
            Data q;
            Data k;
            Data v;
            Data qForAttentionHolder;
            Data attenOutput;
            Data attenLastOutput;
            Data gate;
            Data ffMiddle;
            Data ffAct;
            Data ffUp;
            Data ffOut;
            Data routerLogits;
            Data routerProb;
            Data expertIndex;
            Data expertScore;
            Data w1;
            Data w2;
            Data w3;
            Data tempInput;
            Data tempOutput;
            Data moeInputTemp;
            Data moeOutputTemp;
            Data moeFinal;
            Data shareOutput;
            Data qSizes;
            Data pageSizes;
            Data pageIndexs;
            Data lastPageLens;
            Data insertIndexs;
            Data insertPositions;
            std::vector<Data*> batchPastKeys;
            std::vector<Data*> batchPastValues;

            Step3p5ForwardSingleBuffers() : batchPastKeys(1), batchPastValues(1) {}
        };

        static void Step3p5DetachFakeReusableTensor(Data &data) {
            if (!data.isFake) {
                return;
            }
            data.isFake = false;
            data.cpuData = nullptr;
            data.cudaData = nullptr;
            data.deviceData = nullptr;
            data.expansionSize = 0;
            data.expansionBytes = 0;
        }

        static void Step3p5FreeReusableTensor(Data &data) {
            if (data.isFake) {
                Step3p5DetachFakeReusableTensor(data);
            } else {
                data.FreeSpace();
            }
            data.dims.clear();
            data.strides.clear();
            data.expansionDims.clear();
            data.cpuIntDatas.clear();
            data.dataDevice = DataDevice::CPU;
            data.dataDeviceIds.clear();
            data.weightType = WeightType::NONE;
            data.lockInCPU = false;
            Qwen3CudaClearMultiDeviceState(data);
        }

        static void Step3p5FreeForwardSingleBuffers(Step3p5ForwardSingleBuffers &buf) {
            Step3p5FreeReusableTensor(buf.hiddenStates);
            Step3p5FreeReusableTensor(buf.attenInput);
            Step3p5FreeReusableTensor(buf.qkv);
            Step3p5FreeReusableTensor(buf.q);
            Step3p5FreeReusableTensor(buf.k);
            Step3p5FreeReusableTensor(buf.v);
            Step3p5FreeReusableTensor(buf.qForAttentionHolder);
            Step3p5FreeReusableTensor(buf.attenOutput);
            Step3p5FreeReusableTensor(buf.attenLastOutput);
            Step3p5FreeReusableTensor(buf.gate);
            Step3p5FreeReusableTensor(buf.ffMiddle);
            Step3p5FreeReusableTensor(buf.ffAct);
            Step3p5FreeReusableTensor(buf.ffUp);
            Step3p5FreeReusableTensor(buf.ffOut);
            Step3p5FreeReusableTensor(buf.routerLogits);
            Step3p5FreeReusableTensor(buf.routerProb);
            Step3p5FreeReusableTensor(buf.expertIndex);
            Step3p5FreeReusableTensor(buf.expertScore);
            Step3p5FreeReusableTensor(buf.w1);
            Step3p5FreeReusableTensor(buf.w2);
            Step3p5FreeReusableTensor(buf.w3);
            Step3p5FreeReusableTensor(buf.tempInput);
            Step3p5FreeReusableTensor(buf.tempOutput);
            Step3p5FreeReusableTensor(buf.moeInputTemp);
            Step3p5FreeReusableTensor(buf.moeOutputTemp);
            Step3p5FreeReusableTensor(buf.moeFinal);
            Step3p5FreeReusableTensor(buf.shareOutput);
            Step3p5FreeReusableTensor(buf.qSizes);
            Step3p5FreeReusableTensor(buf.pageSizes);
            Step3p5FreeReusableTensor(buf.pageIndexs);
            Step3p5FreeReusableTensor(buf.lastPageLens);
            Step3p5FreeReusableTensor(buf.insertIndexs);
            Step3p5FreeReusableTensor(buf.insertPositions);
        }

        static void Step3p5ReinitializeForwardSingleBuffers(Step3p5ForwardSingleBuffers &buf) {
            Step3p5FreeForwardSingleBuffers(buf);
            buf.~Step3p5ForwardSingleBuffers();
            new (&buf) Step3p5ForwardSingleBuffers();
        }

        struct Step3p5CudaGraphDecodeState {
            std::mutex mutex;
            std::string signature;
            bool warmed = false;
            bool captured = false;
            bool disabled = false;
            void *graph = nullptr;
            void *exec = nullptr;
            Data inputIds;
            Data positionIds;
            Step3p5ForwardSingleBuffers buffers;
            Step3p5ForwardSingleBuffers metaBuffers;
            Data logitsHalf;
            Data logits;
            std::vector<int> lastInsertIndexHost;
            std::vector<int> lastPageSizesHost;
            std::vector<int> lastPageIndexHost;
            std::vector<int> lastDecodePageLensHost;
            std::vector<const Data*> lastPastKeyHosts;

            ~Step3p5CudaGraphDecodeState() {
                if (exec != nullptr) {
                    FastllmCudaGraphExecDestroy(exec);
                    exec = nullptr;
                }
                if (graph != nullptr) {
                    FastllmCudaGraphDestroy(graph);
                    graph = nullptr;
                }
            }
        };

        struct Step3p5CudaGraphSyncState {
            std::mutex mutex;
            std::condition_variable cv;
            int arrived = 0;
            int generation = 0;
            bool phaseOk = true;
            bool lastPhaseOk = true;
            bool disabled = false;
        };

        static bool Step3p5CudaGraphEnabled(const Step3p5Model *model) {
            return GetFastllmEnv().cudaGraph &&
                   (model == nullptr || !model->autoWarmupRunning.load());
        }

        static Step3p5CudaGraphSyncState &GetStep3p5CudaGraphSyncState(const Step3p5Model *model) {
            static std::mutex syncsMutex;
            static std::map<const Step3p5Model*, std::unique_ptr<Step3p5CudaGraphSyncState> > syncs;
            std::lock_guard<std::mutex> guard(syncsMutex);
            auto &sync = syncs[model];
            if (sync == nullptr) {
                sync.reset(new Step3p5CudaGraphSyncState());
            }
            return *sync;
        }

        static bool Step3p5CudaGraphSyncPhase(const Step3p5Model *model, int participants, bool ok = true) {
            if (participants <= 1) {
                return ok;
            }
            Step3p5CudaGraphSyncState &sync = GetStep3p5CudaGraphSyncState(model);
            std::unique_lock<std::mutex> lock(sync.mutex);
            int generation = sync.generation;
            if (sync.arrived == 0) {
                sync.phaseOk = true;
            }
            sync.phaseOk = sync.phaseOk && ok && !sync.disabled;
            sync.arrived++;
            if (sync.arrived == participants) {
                sync.lastPhaseOk = sync.phaseOk;
                sync.arrived = 0;
                sync.generation++;
                sync.cv.notify_all();
                return sync.lastPhaseOk;
            }
            sync.cv.wait(lock, [&]() {
                return sync.generation != generation;
            });
            return sync.lastPhaseOk;
        }

        static bool Step3p5CudaGraphIsDisabled(const Step3p5Model *model) {
            Step3p5CudaGraphSyncState &sync = GetStep3p5CudaGraphSyncState(model);
            std::lock_guard<std::mutex> guard(sync.mutex);
            return sync.disabled;
        }

        static void Step3p5DisableCudaGraph(const Step3p5Model *model) {
            Step3p5CudaGraphSyncState &sync = GetStep3p5CudaGraphSyncState(model);
            {
                std::lock_guard<std::mutex> guard(sync.mutex);
                sync.disabled = true;
            }
            sync.cv.notify_all();
        }

        static void Step3p5DestroyCudaGraph(Step3p5CudaGraphDecodeState &state) {
            if (state.exec != nullptr) {
                FastllmCudaGraphExecDestroy(state.exec);
                state.exec = nullptr;
            }
            if (state.graph != nullptr) {
                FastllmCudaGraphDestroy(state.graph);
                state.graph = nullptr;
            }
            state.captured = false;
            state.warmed = false;
            state.lastInsertIndexHost.clear();
            state.lastPageSizesHost.clear();
            state.lastPageIndexHost.clear();
            state.lastDecodePageLensHost.clear();
            state.lastPastKeyHosts.clear();
        }

        static void Step3p5AbortCudaGraphCapture() {
            void *capturedGraph = nullptr;
            if (FastllmCudaGraphEndCapture(&capturedGraph) && capturedGraph != nullptr) {
                FastllmCudaGraphDestroy(capturedGraph);
            }
        }

        static void Step3p5DisableCudaGraphState(
                const Step3p5Model *model,
                Step3p5CudaGraphDecodeState &state) {
            Step3p5DestroyCudaGraph(state);
            state.disabled = true;
            Step3p5DisableCudaGraph(model);
        }

        static void Step3p5WarnCudaGraphStage(
                const char *stage,
                int gpuId,
                bool localOk) {
            if (!localOk) {
                printf("Warning: Step3p5 CUDA graph %s failed on gpu %d: %s. Disable graph for this model.\n",
                       stage, gpuId, FastllmCudaGraphLastError());
                fflush(stdout);
            }
        }

        static bool Step3p5SyncCudaGraphStage(
                const Step3p5Model *model,
                Step3p5CudaGraphDecodeState &state,
                int participants,
                const char *stage,
                int gpuId,
                bool localOk) {
            bool allOk = Step3p5CudaGraphSyncPhase(model, participants, localOk);
            if (!allOk) {
                Step3p5WarnCudaGraphStage(stage, gpuId, localOk);
                Step3p5DisableCudaGraphState(model, state);
            }
            return allOk;
        }

        using Step3p5CudaGraphStateKey = std::tuple<const Step3p5Model*, int, int>;

        static std::mutex &Step3p5CudaGraphStatesMutex() {
            static std::mutex statesMutex;
            return statesMutex;
        }

        static std::map<Step3p5CudaGraphStateKey, std::unique_ptr<Step3p5CudaGraphDecodeState> > &Step3p5CudaGraphStates() {
            static std::map<Step3p5CudaGraphStateKey, std::unique_ptr<Step3p5CudaGraphDecodeState> > states;
            return states;
        }

        static Step3p5CudaGraphDecodeState &GetStep3p5CudaGraphDecodeState(
                const Step3p5Model *model, int gpuId, int batch) {
            std::lock_guard<std::mutex> guard(Step3p5CudaGraphStatesMutex());
            auto &states = Step3p5CudaGraphStates();
            auto key = std::make_tuple(model, gpuId, batch);
            auto &state = states[key];
            if (state == nullptr) {
                state.reset(new Step3p5CudaGraphDecodeState());
            }
            return *state;
        }

        static void Step3p5DestroyCudaGraphDecodeStates(const Step3p5Model *model, int gpuId) {
            std::lock_guard<std::mutex> guard(Step3p5CudaGraphStatesMutex());
            auto &states = Step3p5CudaGraphStates();
            for (auto &it : states) {
                if (std::get<0>(it.first) == model && std::get<1>(it.first) == gpuId &&
                    it.second != nullptr) {
                    std::lock_guard<std::mutex> stateGuard(it.second->mutex);
                    Step3p5DestroyCudaGraph(*it.second);
                    Step3p5ReinitializeForwardSingleBuffers(it.second->buffers);
                    Step3p5ReinitializeForwardSingleBuffers(it.second->metaBuffers);
                }
            }
        }

        static void Step3p5PrepareGraphCudaTensorWithDims(
                Data &dst, const Data &src, int device, const std::vector<int> &dims) {
            AssertInFastLLM(src.dataDevice == DataDevice::CUDA && src.cudaData != nullptr,
                            "Step3p5 CUDA graph requires CUDA source tensor.\n");
            uint64_t count = 1;
            for (int dim : dims) {
                count *= dim;
            }
            AssertInFastLLM(count == src.Count(0),
                            "Step3p5 CUDA graph tensor reshape changes element count.\n");
            FastllmCudaSetDevice(device);

            bool needReset = dst.isFake || dst.dataDevice != DataDevice::CUDA ||
                             dst.dataType != src.dataType || dst.dims != dims ||
                             (!dst.dataDeviceIds.empty() && dst.dataDeviceIds[0] != device);
            if (!needReset && dst.cudaData != nullptr) {
                int ptrDevice = GetPointerDeviceId(dst.cudaData);
                needReset = ptrDevice >= 0 && ptrDevice != device;
            }
            if (needReset) {
                if (dst.isFake) {
                    dst.isFake = false;
                    dst.cpuData = nullptr;
                    dst.cudaData = nullptr;
                    dst.deviceData = nullptr;
                    dst.expansionSize = 0;
                    dst.expansionBytes = 0;
                } else {
                    dst.FreeSpace();
                }
                Qwen3CudaClearMultiDeviceState(dst);
                dst.dataType = src.dataType;
                dst.UpdateUnitSize();
                dst.dataDevice = DataDevice::CUDA;
                dst.dataDeviceIds = {device};
                dst.Resize(dims);
            }
            dst.Allocate(false);
            FastllmCudaCopyFromDeviceToDevice(dst.cudaData, src.cudaData, src.GetBytes());
        }

        static void Step3p5PrepareGraphCudaTensor(Data &dst, const Data &src, int device) {
            Step3p5PrepareGraphCudaTensorWithDims(dst, src, device, src.dims);
        }

        static void Step3p5PrepareGraphIntTensor(Data &dst, int device, const std::vector<int> &host) {
            AssertInFastLLM(!host.empty(), "Step3p5 CUDA graph got empty int metadata.\n");
            FastllmCudaSetDevice(device);
            bool needReset = dst.isFake || dst.dataDevice != DataDevice::CUDA ||
                             dst.dataType != DataType::INT32 ||
                             dst.dims != std::vector<int>{(int)host.size()} ||
                             (!dst.dataDeviceIds.empty() && dst.dataDeviceIds[0] != device);
            if (!needReset && dst.cudaData != nullptr) {
                int ptrDevice = GetPointerDeviceId(dst.cudaData);
                needReset = ptrDevice >= 0 && ptrDevice != device;
            }
            if (needReset) {
                if (dst.isFake) {
                    dst.isFake = false;
                    dst.cpuData = nullptr;
                    dst.cudaData = nullptr;
                    dst.deviceData = nullptr;
                    dst.expansionSize = 0;
                    dst.expansionBytes = 0;
                } else {
                    dst.FreeSpace();
                }
                Qwen3CudaClearMultiDeviceState(dst);
                dst.dataType = DataType::INT32;
                dst.UpdateUnitSize();
                dst.dataDevice = DataDevice::CUDA;
                dst.dataDeviceIds = {device};
                dst.Resize({(int)host.size()});
            }
            dst.Allocate(false);
            FastllmCudaCopyFromHostToDevice(dst.cudaData, (void*)host.data(), host.size() * sizeof(int32_t));
            dst.cpuIntDatas = host;
        }

        static bool Step3p5HasLocalMoeShard(const std::vector<Data*> &localWeights) {
            for (int i = 2; i + 1 < (int)localWeights.size(); i += 2) {
                Data *gateup = localWeights[i];
                Data *down = localWeights[i + 1];
                if (gateup != nullptr && down != nullptr &&
                    gateup->dims.size() == 2 && down->dims.size() == 2 &&
                    gateup->dims[0] > 0 && down->dims[1] > 0) {
                    return true;
                }
            }
            return false;
        }

        static void Step3p5MaskRemoteExpertsForLocalShard(
                Data &expertIndex,
                Data &expertScore,
                const std::vector<Data*> &localWeights,
                int numExperts) {
            std::vector<uint8_t> localExpert(numExperts, 0);
            int fallbackExpert = -1;
            for (int expert = 0; expert < numExperts; expert++) {
                int idx = (expert + 1) * 2;
                if (idx + 1 < (int)localWeights.size() &&
                    localWeights[idx] != nullptr &&
                    localWeights[idx + 1] != nullptr) {
                    localExpert[expert] = 1;
                    if (fallbackExpert < 0) {
                        fallbackExpert = expert;
                    }
                }
            }
            if (fallbackExpert < 0) {
                return;
            }

            expertIndex.ToDevice(DataDevice::CPU);
            ToDataType(expertIndex, DataType::INT32);
            expertScore.ToDevice(DataDevice::CPU);
            ToDataType(expertScore, DataType::FLOAT32);

            int32_t *indexData = (int32_t*)expertIndex.cpuData;
            float *scoreData = (float*)expertScore.cpuData;
            int total = expertIndex.Count(0);
            for (int i = 0; i < total; i++) {
                int expert = indexData[i];
                if (expert < 0 || expert >= numExperts || !localExpert[expert]) {
                    indexData[i] = fallbackExpert;
                    scoreData[i] = 0.0f;
                }
            }
        }

        static bool Step3p5MaskAndRemapExpertsForLocalRange(
                Data &expertIndex,
                Data &expertScore,
                int expertStart,
                int expertEnd,
                int numExperts,
                int device,
                bool moveToCuda) {
            if (expertStart < 0 || expertStart >= expertEnd || expertEnd > numExperts) {
                return false;
            }

#ifdef USE_CUDA
            if (moveToCuda &&
                expertIndex.dataDevice == DataDevice::CUDA &&
                expertScore.dataDevice == DataDevice::CUDA &&
                expertIndex.dataType == DataType::INT32 &&
                expertScore.dataType == DataType::FLOAT32 &&
                expertIndex.cudaData != nullptr &&
                expertScore.cudaData != nullptr) {
                return FastllmCudaMaskAndRemapExpertsForLocalRange(expertIndex, expertScore,
                                                                    expertStart, expertEnd);
            }
#endif

            expertIndex.ToDevice(DataDevice::CPU);
            ToDataType(expertIndex, DataType::INT32);
            expertScore.ToDevice(DataDevice::CPU);
            ToDataType(expertScore, DataType::FLOAT32);

            int32_t *indexData = (int32_t*)expertIndex.cpuData;
            float *scoreData = (float*)expertScore.cpuData;
            int total = expertIndex.Count(0);
            for (int i = 0; i < total; i++) {
                int expert = indexData[i];
                if (expert >= expertStart && expert < expertEnd) {
                    indexData[i] = expert - expertStart;
                } else {
                    indexData[i] = 0;
                    scoreData[i] = 0.0f;
                }
            }

            if (moveToCuda) {
                expertIndex.ToDevice(DataDevice::CUDA, {device}, true);
                expertScore.ToDevice(DataDevice::CUDA, {device}, true);
            }
            return true;
        }

        static void Step3p5CudaCopyTensor(Qwen3CudaDirectRunner &runner,
                                          const Data &input,
                                          Data &output) {
            runner.Run("Copy",
                       DataDict{{"input", (Data*)&input}, {"output", &output}},
                       FloatDict(), IntDict(), {"output"});
        }

        static void Step3p5CudaSigmoid(Qwen3CudaDirectRunner &runner,
                                       Data &input, Data &output) {
            runner.Run("Sigmoid",
                       DataDict{{"input", &input}, {"output", &output}},
                       FloatDict(), IntDict(), {"output"});
        }

        static void Step3p5CudaSilu(Qwen3CudaDirectRunner &runner,
                                    Data &input, Data &output) {
            runner.Run("Silu",
                       DataDict{{"input", &input}, {"output", &output}},
                       FloatDict(), IntDict(), {"output"});
        }

        static void Step3p5CudaMulTo(Qwen3CudaDirectRunner &runner,
                                     Data &input0, const Data &input1,
                                     float alpha = 1.0f) {
            runner.Run("MulTo",
                       DataDict{{"input0", &input0}, {"input1", (Data*)&input1}},
                       FloatDict{{"alpha", alpha}}, IntDict());
        }

        static void Step3p5CudaApplyRotary(Qwen3CudaDirectRunner &runner,
                                           Data &input,
                                           const Data &positionIds,
                                           int rotaryDim,
                                           float ropeTheta,
                                           bool useLlama3,
                                           float factor,
                                           float originalMaxPosition,
                                           float lowFreqFactor,
                                           float highFreqFactor) {
            if (useLlama3) {
                runner.Run("Llama3RopeEncoding",
                           DataDict{{"input", &input}, {"positionIds", (Data*)&positionIds}},
                           FloatDict{{"ropeTheta", ropeTheta},
                                     {"factor", factor},
                                     {"originalMaxPosition", originalMaxPosition},
                                     {"lowFreqFactor", lowFreqFactor},
                                     {"highFreqFactor", highFreqFactor}},
                           IntDict{{"rotaryDim", rotaryDim}});
            } else {
                Qwen3CudaRopeEncoding(runner, input, positionIds, rotaryDim, ropeTheta, 1.0f);
            }
        }

        static void Step3p5CudaQKVRMSNormRopeSplitAppendPagedCache(
                Qwen3CudaDirectRunner &runner,
                Data &qkv,
                Data &qNormWeight,
                Data &kNormWeight,
                const Data &positionIds,
                Data &qOutput,
                Data &pagedKCacheData,
                Data &pagedVCacheData,
                Data &insertIndexs,
                Data &insertPositions,
                int qHeads,
                int kHeads,
                int headDim,
                int rotaryDim,
                float eps,
                float ropeTheta,
                bool useLlama3,
                float llama3Factor,
                float llama3OriginalMaxPosition,
                float llama3LowFreqFactor,
                float llama3HighFreqFactor,
                int pageLen,
                int batch,
                Data *lastPageLens) {
            DataDict datas = {
                {"qkv", &qkv},
                {"qNormWeight", &qNormWeight},
                {"kNormWeight", &kNormWeight},
                {"positionIds", (Data*)&positionIds},
                {"qOutput", &qOutput},
                {"pagedKCacheData", &pagedKCacheData},
                {"pagedVCacheData", &pagedVCacheData},
                {"insertIndexs", &insertIndexs},
                {"insertPositions", &insertPositions}
            };
            std::vector<std::string> outputs = {"qOutput"};
            if (lastPageLens != nullptr) {
                datas["lastPageLens"] = lastPageLens;
                outputs.push_back("lastPageLens");
            }
            runner.Run("Step3p5QKVRMSNormRopeSplitAppendPagedCache",
                       datas,
                       FloatDict{{"eps", eps},
                                 {"ropeTheta", ropeTheta},
                                 {"ropeScale", 1.0f},
                                 {"llama3Factor", llama3Factor},
                                 {"llama3OriginalMaxPosition", llama3OriginalMaxPosition},
                                 {"llama3LowFreqFactor", llama3LowFreqFactor},
                                 {"llama3HighFreqFactor", llama3HighFreqFactor}},
                       IntDict{{"q_heads", qHeads}, {"k_heads", kHeads}, {"head_dim", headDim},
                               {"rotaryDim", rotaryDim}, {"pageLen", pageLen}, {"batch", batch},
                               {"doQKNorm", 1}, {"useLlama3", useLlama3 ? 1 : 0}},
                       outputs);
        }

        static void Step3p5CudaClamp(Data &input,
                                     bool hasMin, float minValue,
                                     bool hasMax, float maxValue,
                                     int device) {
            if (!hasMin && !hasMax) {
                return;
            }
            if (input.dataDevice == DataDevice::CUDA && input.cudaData != nullptr &&
                FastllmCudaClamp(input, hasMin, minValue, hasMax, maxValue)) {
                return;
            }
            input.ToDevice(DataDevice::CPU, true);
            Step3p5Clamp(input, hasMin, minValue, hasMax, maxValue);
            input.ToDevice(DataDevice::CUDA, {device}, true);
        }

        static void Step3p5CudaFusedMOE(Qwen3CudaDirectRunner &runner,
                                        Data &input, Data &expertIndex, Data &expertScore,
                                        Data &gate, Data &up, Data &down, Data &w1,
                                        Data &output, int layer, float swigluLimit) {
            runner.Run("FusedMOE",
                       DataDict{{"input", &input}, {"index", &expertIndex}, {"score", &expertScore},
                                {"gate", &gate}, {"up", &up}, {"down", &down},
                                {"w1", &w1}, {"output", &output}},
                       FloatDict{{"swigluLimit", swigluLimit}},
                       IntDict{{"layer", layer}, {"gateType", (int)MoeGateSwiglu}},
                       {"w1", "output"});
        }

        static void Step3p5CudaAttentionPagedBlock(
                Qwen3CudaDirectRunner &runner,
                Data *attenInput,
                Data *mergeQkvWeight, Data *mergeQkvBias,
                Data *qNormWeight, Data *kNormWeight,
                Data *allPositionIds,
                std::vector<std::pair<Data*, Data*>> *pastKeyValues,
                std::vector<Data*> *batchPastKeys,
                std::vector<Data*> *batchPastValues,
                Data *qkv, Data *q, Data *kBuffer, Data *vBuffer, Data *attenOutput,
                Data *qForAttentionHolder,
                Data *insertIndexs, Data *insertPositions,
                Data *qSizes, Data *pageSizes, Data *pageIndexs, Data *lastPageLens,
                bool *generatedAppendParams, bool *generatedDecodeParams,
                int batch, int blockCnt, int layerIdx,
                const std::vector<int> &seqLens,
                int numAttentionHeads, int numKeyValueHeads, int headDim,
                int rotaryDim, float rmsNormEps,
                float ropeTheta, float ropeFactor,
                bool useLlama3,
                float llama3OriginalMaxPosition,
                float llama3LowFreqFactor,
                float llama3HighFreqFactor,
                bool kvCacheInCPU,
                int pagedCacheLayerOffset,
                bool isPrefill,
                bool externalDecodeMeta,
                bool enableFlashInferCudaGraph = false,
                int flashInferCudaGraph = -1) {
            mergeQkvWeight->tpPackType = TP_PACK_QKV;
            mergeQkvWeight->tpQHeads = numAttentionHeads;
            mergeQkvWeight->tpKVHeads = numKeyValueHeads;
            mergeQkvWeight->tpHeadDim = headDim;
            Qwen3CudaLinear(runner, *attenInput, *mergeQkvWeight, *mergeQkvBias, *qkv);

            for (int b = 0; b < batch; b++) {
                Data &pastKey = *(*pastKeyValues)[b * blockCnt + layerIdx].first;
                Data &pastValue = *(*pastKeyValues)[b * blockCnt + layerIdx].second;
                if (kvCacheInCPU) {
                    pastKey.lockInCPU = true;
                    pastValue.lockInCPU = true;
                } else {
                    if (pastKey.dataDeviceIds.empty()) {
                        pastKey.dataDeviceIds = {runner.DeviceId()};
                    }
                    if (pastValue.dataDeviceIds.empty()) {
                        pastValue.dataDeviceIds = {runner.DeviceId()};
                    }
                    AssertInFastLLM(pastKey.dataDevice == DataDevice::CUDA &&
                                    pastValue.dataDevice == DataDevice::CUDA &&
                                    pastKey.dataDeviceIds[0] == runner.DeviceId() &&
                                    pastValue.dataDeviceIds[0] == runner.DeviceId(),
                                    "Step3p5 ForwardGPU cache is not on the bound CUDA device.\n");
                }
                (*batchPastKeys)[b] = (*pastKeyValues)[b * blockCnt + layerIdx].first;
                (*batchPastValues)[b] = (*pastKeyValues)[b * blockCnt + layerIdx].second;
            }

            bool useFp8KVCache = ((*batchPastKeys)[0]->dataType == DataType::FP8_E4M3 ||
                                  (*batchPastValues)[0]->dataType == DataType::FP8_E4M3);
            if (useFp8KVCache) {
                AssertInFastLLM(!kvCacheInCPU, "Step3p5 FP8 KV cache doesn't support kvCacheInCPU.\n");
                AssertInFastLLM(qkv->dataDevice == DataDevice::CUDA, "Step3p5 FP8 KV cache requires CUDA attention.\n");
                AssertInFastLLM(headDim != 64, "Step3p5 FP8 KV cache is not supported when head_dim == 64.\n");
            }

            int bsz = attenInput->dims[0], seqlen = attenInput->dims[1];
            if (!isPrefill && externalDecodeMeta) {
                AssertInFastLLM(bsz * seqlen == batch &&
                                insertIndexs != nullptr && insertPositions != nullptr &&
                                qSizes != nullptr && pageSizes != nullptr &&
                                pageIndexs != nullptr && lastPageLens != nullptr,
                                "Step3p5 CUDA graph attention requires external single-token batch decode metadata.\n");
                Data &pastKey = *(*batchPastKeys)[0];
                Data &pastValue = *(*batchPastValues)[0];
                PagedCacheManager *pagedCacheKManager = pastKey.pagedKVCacheData;
                PagedCacheManager *pagedCacheVManager = pastValue.pagedKVCacheData;
                AssertInFastLLM(pagedCacheKManager != nullptr && pagedCacheVManager != nullptr,
                                "Step3p5 CUDA graph requires paged KV cache managers.\n");

                Data &pagedKData = *(Data*)pagedCacheKManager;
                Data &pagedVData = *(Data*)pagedCacheVManager;
                AssertInFastLLM(pagedKData.dims.size() == 4 && pagedVData.dims.size() == 4 &&
                                pagedKData.cudaData != nullptr && pagedVData.cudaData != nullptr,
                                "Step3p5 CUDA graph requires allocated paged KV cache data.\n");

                q->dataType = qkv->dataType;
                q->UpdateUnitSize();
                Qwen3CudaPrepareLocalOutput(*q, runner.DeviceId());
                q->Resize({bsz * numAttentionHeads, seqlen, headDim});
                q->Allocate(false);

                FastllmCudaSetDevice(runner.DeviceId());
                AssertInFastLLM(
                    FastllmCudaQKVRMSNormRopeSplitAppendPagedCache(
                        *qkv, *qNormWeight, *kNormWeight, *allPositionIds,
                        *q,
                        (uint8_t*)pagedKData.cudaData,
                        (uint8_t*)pagedVData.cudaData,
                        (int32_t*)insertIndexs->cudaData,
                        (int32_t*)insertPositions->cudaData,
                        nullptr,
                        numAttentionHeads, numKeyValueHeads, headDim,
                        rotaryDim, rmsNormEps, ropeTheta, 1.0f,
                        pagedKData.dims[1], pagedKData.dims[0], pagedKData.dataType, batch, 1,
                        useLlama3 ? 1 : 0, ropeFactor,
                        llama3OriginalMaxPosition,
                        llama3LowFreqFactor,
                        llama3HighFreqFactor),
                    "Step3p5 CUDA graph fused QKV append failed.\n");

                Data &kCaches = *(*batchPastKeys)[0];
                Data &vCaches = *(*batchPastValues)[0];
                auto resolvePagedAttentionQType = [&](DataType cacheType, DataType queryType) -> DataType {
                    if (cacheType == DataType::FP8_E4M3) {
                        return queryType == DataType::BFLOAT16 ? DataType::BFLOAT16 : DataType::FLOAT16;
                    }
                    if (queryType == DataType::FLOAT16 || queryType == DataType::BFLOAT16) {
                        return queryType;
                    }
                    return DataType::FLOAT16;
                };
                DataType targetType = resolvePagedAttentionQType(kCaches.dataType, q->dataType);
                Data *qForAttention = q;
                if (q->dataType != targetType) {
                    Qwen3CudaConvertToDataType(runner, *q, *qForAttentionHolder, targetType);
                    qForAttention = qForAttentionHolder;
                }
                Qwen3CudaAttentionPagedBatch(runner, *qForAttention,
                    kCaches, vCaches,
                    *qSizes, *pageSizes, *pageIndexs, *lastPageLens,
                    *attenOutput, numAttentionHeads / numKeyValueHeads,
                    1.0f / std::sqrt((float)headDim), 1, layerIdx > 0,
                    enableFlashInferCudaGraph, flashInferCudaGraph);
                attenOutput->Reshape({seqlen, bsz, -1});
                Qwen3CudaPermuteSelf(runner, *attenOutput, {1, 0, 2});
                return;
            }

            if (!isPrefill && (*batchPastKeys)[0]->pagedKVCacheData == nullptr) {
                isPrefill = true;
            }

            bool singleTokenDecode = (bsz * seqlen == batch);
            for (int len : seqLens) {
                singleTokenDecode = singleTokenDecode && (len == 1);
            }

            if (!isPrefill && !kvCacheInCPU && singleTokenDecode) {
                AssertInFastLLM(qSizes != nullptr && pageSizes != nullptr &&
                                pageIndexs != nullptr && lastPageLens != nullptr,
                                "Step3p5 fused decode attention requires paged batch metadata buffers.\n");

                Data &kCaches = *(*batchPastKeys)[0];
                Data &vCaches = *(*batchPastValues)[0];
                PagedCacheManager *pagedCacheKManager = kCaches.pagedKVCacheData;
                PagedCacheManager *pagedCacheVManager = vCaches.pagedKVCacheData;
                AssertInFastLLM(pagedCacheKManager != nullptr && pagedCacheVManager != nullptr,
                                "Step3p5 fused decode attention requires paged KV cache managers.\n");

                Data &pagedKData = *(Data*)pagedCacheKManager;
                Data &pagedVData = *(Data*)pagedCacheVManager;
                AssertInFastLLM(pagedKData.dims.size() == 4 && pagedVData.dims.size() == 4 &&
                                pagedKData.cudaData != nullptr && pagedVData.cudaData != nullptr,
                                "Step3p5 fused decode attention requires allocated paged KV cache data.\n");

                Data localInsertIndexs, localInsertPositions;
                Data &decodeInsertIndexs = insertIndexs == nullptr ? localInsertIndexs : *insertIndexs;
                Data &decodeInsertPositions = insertPositions == nullptr ? localInsertPositions : *insertPositions;
                bool appendParamsReady =
                    generatedAppendParams != nullptr && *generatedAppendParams &&
                    decodeInsertIndexs.cudaData != nullptr && decodeInsertPositions.cudaData != nullptr;
                if (!appendParamsReady) {
                    Qwen3CudaGenerateAppendPagedCacheBatchParams(runner, *pagedCacheKManager,
                        *batchPastKeys, batch, decodeInsertIndexs, decodeInsertPositions);
                    if (generatedAppendParams != nullptr &&
                        insertIndexs != nullptr && insertPositions != nullptr) {
                        *generatedAppendParams = true;
                    }
                }

                q->dataType = qkv->dataType;
                q->UpdateUnitSize();
                q->Resize({bsz * numAttentionHeads, seqlen, headDim});

                bool decodeParamsReady =
                    generatedDecodeParams != nullptr && *generatedDecodeParams &&
                    qSizes->cudaData != nullptr && pageSizes->cudaData != nullptr &&
                    pageIndexs->cudaData != nullptr && lastPageLens->cudaData != nullptr;
                bool fillLastPageLensOnDevice =
                    qkv->dataDevice == DataDevice::CUDA && !qkv->multiDeviceData &&
                    !decodeParamsReady;

                Step3p5CudaQKVRMSNormRopeSplitAppendPagedCache(runner, *qkv,
                    *qNormWeight, *kNormWeight,
                    *allPositionIds,
                    *q,
                    pagedKData, pagedVData,
                    decodeInsertIndexs, decodeInsertPositions,
                    numAttentionHeads, numKeyValueHeads, headDim,
                    rotaryDim, rmsNormEps, ropeTheta,
                    useLlama3, ropeFactor,
                    llama3OriginalMaxPosition,
                    llama3LowFreqFactor,
                    llama3HighFreqFactor,
                    pagedKData.dims[1], batch,
                    fillLastPageLensOnDevice ? lastPageLens : nullptr);

                for (int b = 0; b < batch; b++) {
                    auto updatePageMeta = [](Data *cache, PagedCacheManager *mgr) {
                        if (cache->pageIndex.empty() || cache->lastPageLen >= cache->pageLen) {
                            cache->pageIndex.push_back(mgr->GetUnusedPageIndex(true));
                            cache->lastPageLen = 1;
                        } else {
                            cache->lastPageLen++;
                        }
                    };
                    updatePageMeta((*batchPastKeys)[b], pagedCacheKManager);
                    updatePageMeta((*batchPastValues)[b], pagedCacheVManager);
                }

                auto resolvePagedAttentionQType = [&](DataType cacheType, DataType queryType) -> DataType {
                    if (cacheType == DataType::FP8_E4M3) {
                        return queryType == DataType::BFLOAT16 ? DataType::BFLOAT16 : DataType::FLOAT16;
                    }
                    if (queryType == DataType::FLOAT16 || queryType == DataType::BFLOAT16) {
                        return queryType;
                    }
                    return DataType::FLOAT16;
                };
                DataType targetType = resolvePagedAttentionQType(kCaches.dataType, q->dataType);
                Data *qForAttention = q;
                if (q->dataType != targetType) {
                    Qwen3CudaConvertToDataType(runner, *q, *qForAttentionHolder, targetType);
                    qForAttention = qForAttentionHolder;
                }

                if (!decodeParamsReady) {
                    Qwen3CudaGeneratePagedBatchParams(runner, *qForAttention, *batchPastKeys, batch,
                                                      *qSizes, *pageSizes, *pageIndexs, *lastPageLens,
                                                      std::vector<int>(), fillLastPageLensOnDevice);
                    if (generatedDecodeParams != nullptr) {
                        *generatedDecodeParams = true;
                    }
                }

                Qwen3CudaAttentionPagedBatch(runner, *qForAttention,
                                             kCaches, vCaches,
                                             *qSizes, *pageSizes, *pageIndexs, *lastPageLens,
                                             *attenOutput, numAttentionHeads / numKeyValueHeads,
                                             1.0f / std::sqrt((float)headDim), 1, layerIdx > 0);
                attenOutput->Reshape({seqlen, bsz, -1});
                Qwen3CudaPermuteSelf(runner, *attenOutput, {1, 0, 2});
                return;
            }

            int per = qkv->dims.back() / (numAttentionHeads / numKeyValueHeads + 2);
            int qdim = per * (numAttentionHeads / numKeyValueHeads);
            Data localK, localV;
            Data &k = kBuffer == nullptr ? localK : *kBuffer;
            Data &v = vBuffer == nullptr ? localV : *vBuffer;
            Qwen3CudaSplit(runner, *qkv, -1, 0, qdim, *q);
            Qwen3CudaSplit(runner, *qkv, -1, qdim, qdim + per, k);
            Qwen3CudaSplit(runner, *qkv, -1, qdim + per, qdim + per * 2, v);

            std::vector<int> qkvSize = {bsz, seqlen, -1, headDim};
            q->Reshape(qkvSize);
            k.Reshape(qkvSize);
            v.Reshape(qkvSize);

            Qwen3CudaRMSNorm(runner, *q, *qNormWeight, rmsNormEps, *q);
            Qwen3CudaRMSNorm(runner, k, *kNormWeight, rmsNormEps, k);
            Step3p5CudaApplyRotary(runner, *q, *allPositionIds, rotaryDim, ropeTheta,
                                   useLlama3, ropeFactor, llama3OriginalMaxPosition,
                                   llama3LowFreqFactor, llama3HighFreqFactor);
            Step3p5CudaApplyRotary(runner, k, *allPositionIds, rotaryDim, ropeTheta,
                                   useLlama3, ropeFactor, llama3OriginalMaxPosition,
                                   llama3LowFreqFactor, llama3HighFreqFactor);

            Qwen3CudaPermuteSelf(runner, *q, {0, 2, 1, 3});
            Qwen3CudaPermuteSelf(runner, k, {0, 2, 1, 3});
            Qwen3CudaPermuteSelf(runner, v, {0, 2, 1, 3});

            k.Reshape({-1, seqlen, headDim});
            v.Reshape({-1, seqlen, headDim});
            q->Reshape({-1, seqlen, headDim});

            auto makeCacheDesc = [](const Data &src, DataType targetType) {
                Data desc(targetType);
                desc.dims = src.dims;
                desc.strides = src.strides;
                desc.dataDevice = src.dataDevice;
                desc.dataDeviceIds = src.dataDeviceIds;
                desc.multiDeviceData = src.multiDeviceData;
                desc.tpLayout = src.tpLayout;
                desc.tpAxis = src.tpAxis;
                desc.tpGlobalDims = src.tpGlobalDims;
                desc.tpRanges = src.tpRanges;
                desc.UpdateUnitSize();
                return desc;
            };

            if (batch == 1) {
                Data &pastKey = *(*batchPastKeys)[0];
                Data &pastValue = *(*batchPastValues)[0];
                Data kCacheDesc = makeCacheDesc(k, pastKey.dataType);
                Data vCacheDesc = makeCacheDesc(v, pastValue.dataType);
                int cacheLayerIdx = pagedCacheLayerOffset + layerIdx;
                PagedCacheManager *pagedCacheKManager = AllocatePagedCacheManager(
                    cacheLayerIdx * 2, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, kCacheDesc);
                PagedCacheManager *pagedCacheVManager = AllocatePagedCacheManager(
                    cacheLayerIdx * 2 + 1, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, vCacheDesc);
                Qwen3CudaAppendPagedCache(runner, *pagedCacheKManager, pastKey, k);
                Qwen3CudaAppendPagedCache(runner, *pagedCacheVManager, pastValue, v);
            } else {
                int total = 0;
                Data curK, curV;
                for (int b = 0; b < batch; b++) {
                    Data &pastKey = *(*batchPastKeys)[b];
                    Data &pastValue = *(*batchPastValues)[b];
                    Qwen3CudaSplit(runner, k, 1, total, total + seqLens[b], curK);
                    Qwen3CudaSplit(runner, v, 1, total, total + seqLens[b], curV);

                    Data kCacheDesc = makeCacheDesc(curK, pastKey.dataType);
                    Data vCacheDesc = makeCacheDesc(curV, pastValue.dataType);
                    int cacheLayerIdx = pagedCacheLayerOffset + layerIdx;
                    PagedCacheManager *pagedCacheKManager = AllocatePagedCacheManager(
                        cacheLayerIdx * 2, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, kCacheDesc);
                    PagedCacheManager *pagedCacheVManager = AllocatePagedCacheManager(
                        cacheLayerIdx * 2 + 1, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, vCacheDesc);
                    Qwen3CudaAppendPagedCache(runner, *pagedCacheKManager, pastKey, curK);
                    Qwen3CudaAppendPagedCache(runner, *pagedCacheVManager, pastValue, curV);
                    total += seqLens[b];
                }
            }

            Data &kCaches = *(*batchPastKeys)[0];
            Data &vCaches = *(*batchPastValues)[0];
            auto resolvePagedAttentionQType = [&](DataType cacheType, DataType queryType) -> DataType {
                if (cacheType == DataType::FP8_E4M3) {
                    return queryType == DataType::BFLOAT16 ? DataType::BFLOAT16 : DataType::FLOAT16;
                }
                if (queryType == DataType::FLOAT16 || queryType == DataType::BFLOAT16) {
                    return queryType;
                }
                return DataType::FLOAT16;
            };
            DataType targetType = resolvePagedAttentionQType(kCaches.dataType, q->dataType);
            Data *qForAttention = q;
            if (q->dataType != targetType) {
                Qwen3CudaConvertToDataType(runner, *q, *qForAttentionHolder, targetType);
                qForAttention = qForAttentionHolder;
            }
            Qwen3CudaGeneratePagedBatchParams(runner, *qForAttention, *batchPastKeys, batch,
                                              *qSizes, *pageSizes, *pageIndexs, *lastPageLens, seqLens);
            Qwen3CudaAttentionPagedBatch(runner, *qForAttention,
                                         kCaches, vCaches,
                                         *qSizes, *pageSizes, *pageIndexs, *lastPageLens,
                                         *attenOutput, numAttentionHeads / numKeyValueHeads,
                                         1.0f / std::sqrt((float)headDim), 1, layerIdx > 0);
            attenOutput->Reshape({1, seqlen, -1});
        }
    }
#endif

    Step3p5Model::Step3p5Model() {
        this->model_type = "step3p5";
        this->model_struct = "step3p5";
        this->use_new_engine = true;

        this->pre_prompt = STEP3P5_BOS;
        this->user_role = STEP3P5_IM_START + std::string("user\n");
        this->bot_role = STEP3P5_IM_END + std::string("\n") +
                         STEP3P5_IM_START + std::string("assistant\n<think>\n");
        this->history_sep = STEP3P5_IM_END + std::string("\n");

        weight.embeddingNames.insert("model.embed_tokens.weight");
        weight.linearNames = {
            "lm_head.weight",
            "model.layers.*.mlp.down_proj.weight",
            "model.layers.*.mlp.up_proj.weight",
            "model.layers.*.mlp.gate_proj.weight",
            "model.layers.*.mlp.gateup_proj.weight",
            "model.layers.*.share_expert.down_proj.weight",
            "model.layers.*.share_expert.up_proj.weight",
            "model.layers.*.share_expert.gate_proj.weight",
            "model.layers.*.share_expert.gateup_proj.weight",
            "model.layers.*.moe.gate.weight",
            "model.layers.*.moe.gate_proj.weight",
            "model.layers.*.moe.up_proj.weight",
            "model.layers.*.moe.down_proj.weight",
            "model.layers.*.self_attn.o_proj.weight",
            "model.layers.*.self_attn.q_proj.weight",
            "model.layers.*.self_attn.k_proj.weight",
            "model.layers.*.self_attn.v_proj.weight",
            "model.layers.*.self_attn.g_proj.weight",
            "model.layers.*.self_attn.mergeqkv.weight"
        };
    }

    std::map <std::string, std::vector <std::pair <std::string, DataType> > >
            Step3p5Model::GetTensorMap(const std::vector <std::string> &tensorNames) {
        std::map <std::string, std::vector <std::pair <std::string, DataType> > > ret;
        std::vector<std::string> textTensorNames;
        textTensorNames.reserve(tensorNames.size());
        for (auto &name : tensorNames) {
            DataType visionType = DataType::FLOAT32;
            if (Step3p7VisionTensorType(name, visionType)) {
                ret[name].push_back(std::make_pair(name, visionType));
            } else {
                textTensorNames.push_back(name);
            }
        }
        auto textMap = basellm::GetTensorMap(textTensorNames);
        ret.insert(textMap.begin(), textMap.end());
        return ret;
    }

    bool Step3p5Model::IsThreadTensorParallelEnabled() const {
#ifdef USE_CUDA
        std::vector<int> devices;
        std::map<int, int> ratios;
        return CanUseGPUForward() &&
               GetStep3p5ThreadTpDevices(this->deviceMap, devices, ratios);
#else
        return false;
#endif
    }

    bool Step3p5Model::CanUseGPUForward() const {
#ifdef USE_CUDA
        if (GetKVCacheInCPU()) {
            return false;
        }
        std::vector<int> devices;
        std::map<int, int> ratios;
        return GetStep3p5GPUForwardDevices(this->deviceMap, devices, ratios);
#else
        return false;
#endif
    }

    void Step3p5Model::OnAutoWarmupFinished() {
#ifdef USE_CUDA
        if (GetFastllmEnv().cudaGraph) {
            if (step3p7VisionAvailable) {
                const char *forcePreCapture = std::getenv("FASTLLM_STEP3P7_CUDA_GRAPH_PRECAPTURE");
                bool force = forcePreCapture != nullptr &&
                             !(forcePreCapture[0] == '\0' ||
                               (forcePreCapture[0] == '0' && forcePreCapture[1] == '\0'));
                if (!force) {
                    if (Step3p7CudaGraphDecodeEnabled()) {
                        printf("[Fastllm] Step3.7 CUDA graph: skip startup pre-capture to keep vision prefill headroom; decode graph will warm up lazily per request.\n");
                    } else {
                        printf("[Fastllm] Step3.7 CUDA graph: skip startup pre-capture to keep vision prefill headroom; decode graph is disabled by FASTLLM_STEP3P7_CUDA_GRAPH_DECODE=0.\n");
                    }
                    return;
                }
            }
            if (threadTpWorkerGroup.HasWorkers()) {
                threadTpWorkerGroup.Stop();
            }
            ClearAllPagedCacheManagers();
            FastllmCudaClearBigBuffer();
            PreCaptureCudaGraphAfterWarmup();
        }
#endif
    }

    PagedCacheManager* Step3p5Model::GetPagedKVCacheManager(int layerIndex, bool isKey) const {
        if (layerIndex >= 0 && this->threadTpPagedCacheBase >= 0) {
            PagedCacheManager *manager = GetPagedCacheManager(
                (this->threadTpPagedCacheBase + layerIndex) * 2 + (isKey ? 0 : 1));
            if (manager != nullptr) {
                return manager;
            }
        }
        return basellm::GetPagedKVCacheManager(layerIndex, isKey);
    }

    std::vector<std::pair<int, PagedCacheManager*> > Step3p5Model::GetPagedKVCacheManagers(int layerIndex, bool isKey) const {
        if (layerIndex >= 0 && this->threadTpPagedCacheBase >= 0) {
            std::vector<std::pair<int, PagedCacheManager*> > ret;
            int ranks = this->threadTpPreparedDevices.empty() ? 1 : (int)this->threadTpPreparedDevices.size();
            for (int r = 0; r < ranks; r++) {
                PagedCacheManager *manager = GetPagedCacheManager(
                    (this->threadTpPagedCacheBase + r * this->block_cnt + layerIndex) * 2 + (isKey ? 0 : 1));
                if (manager == nullptr) {
                    ret.clear();
                    break;
                }
                int device = r < (int)this->threadTpPreparedDevices.size() ? this->threadTpPreparedDevices[r] : -1;
                if (device < 0) {
                    Data *managerData = (Data*)manager;
                    if (!managerData->dataDeviceIds.empty()) {
                        device = managerData->dataDeviceIds[0];
                    }
                }
                ret.push_back(std::make_pair(device, manager));
            }
            if (!ret.empty()) {
                return ret;
            }
            if (!this->threadTpPreparedDevices.empty()) {
                return ret;
            }
        }
        return basellm::GetPagedKVCacheManagers(layerIndex, isKey);
    }

    static std::vector<int> GetStep3p5CudaGraphWarmupBatches() {
        const int maxCudaGraphDecodeBatch = 32;
        std::vector<int> ret;
        std::set<int> seen;
        const char *env = std::getenv("FASTLLM_CUDA_GRAPH_WARMUP_BATCHES");
        std::string config = env == nullptr ? "" : env;
        if (config.empty()) {
            ret = {1};
            return ret;
        }
        std::string lowered = config;
        std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (lowered == "0" || lowered == "off" || lowered == "false" || lowered == "none") {
            return ret;
        }

        for (char &c : config) {
            if (!std::isdigit((unsigned char)c)) {
                c = ' ';
            }
        }
        std::stringstream ss(config);
        int batch = 0;
        while (ss >> batch) {
            if (batch >= 1 && batch <= maxCudaGraphDecodeBatch &&
                seen.insert(batch).second) {
                ret.push_back(batch);
            }
        }
        return ret;
    }

    void Step3p5Model::PreCaptureCudaGraphAfterWarmup() {
#ifdef USE_CUDA
        if (!GetFastllmEnv().cudaGraph || autoWarmupRunning.load() || GetKVCacheInCPU()) {
            return;
        }
        std::vector<int> devices;
        std::map<int, int> ratios;
        if (!GetStep3p5GPUForwardDevices(this->deviceMap, devices, ratios) || devices.empty()) {
            return;
        }

        std::vector<int> batches = GetStep3p5CudaGraphWarmupBatches();
        if (batches.empty()) {
            return;
        }

        std::vector<int> captureBatches;
        captureBatches.reserve(batches.size());
        for (int batch : batches) {
            if (this->maxBatch <= 0 || batch <= this->maxBatch) {
                captureBatches.push_back(batch);
            }
        }
        if (captureBatches.empty()) {
            return;
        }

        auto printProgress = [](int done, int total, int batch) {
            const int barWidth = 32;
            int filled = total > 0 ? done * barWidth / total : barWidth;
            printf("\r[Fastllm] Step3p5 CUDA graph warmup capture [");
            for (int i = 0; i < barWidth; i++) {
                putchar(i < filled ? '#' : '-');
            }
            printf("] %d/%d batch=%d%s", done, total, batch,
                   done >= total ? " done" : "     ");
            if (done >= total) {
                printf("\n");
            }
            fflush(stdout);
        };
        printProgress(0, (int)captureBatches.size(), 0);

        for (int idx = 0; idx < (int)captureBatches.size(); idx++) {
            int batch = captureBatches[idx];
            std::vector<float> inputIdsHost(batch, 1.0f);
            Data inputIds(DataType::FLOAT32, {1, batch}, inputIdsHost);
            std::vector<Data*> attentionMasks(batch, nullptr);
            std::vector<int> seqLens(batch, 1);
            std::vector<GenerationConfig> generationConfigs(batch);
            LastTokensManager lastTokens;

            std::vector<std::pair<Data, Data> > pastKeyValuesStorage;
            std::vector<std::pair<Data*, Data*> > pastKeyValues;
            pastKeyValuesStorage.reserve(batch * block_cnt);
            pastKeyValues.reserve(batch * block_cnt);
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < block_cnt; i++) {
                    pastKeyValuesStorage.push_back(std::make_pair(Data(this->kvCacheDataType),
                                                                  Data(this->kvCacheDataType)));
                    pastKeyValuesStorage.back().first.SetKVCache();
                    pastKeyValuesStorage.back().second.SetKVCache();
                    pastKeyValues.push_back(std::make_pair(&pastKeyValuesStorage.back().first,
                                                           &pastKeyValuesStorage.back().second));
                }
            }

            for (int step = 0; step < 3; step++) {
                std::vector<Data> positionIdsStorage;
                std::vector<Data*> positionIds;
                positionIdsStorage.reserve(batch);
                positionIds.reserve(batch);
                for (int b = 0; b < batch; b++) {
                    positionIdsStorage.push_back(Data(DataType::FLOAT32, {1, 1}, {(float)step}));
                    positionIds.push_back(&positionIdsStorage.back());
                }
                ForwardGPU(batch, inputIds, attentionMasks, positionIds, seqLens,
                           pastKeyValues, generationConfigs, lastTokens, nullptr);
            }
            printProgress(idx + 1, (int)captureBatches.size(), batch);
        }
#endif
    }

    Data &Step3p5Model::GetThreadTensorParallelBias(const std::string &name) {
        auto it = this->weight.weight.find(name);
        if (it != this->weight.weight.end()) {
            return it->second;
        }
        return this->threadTpEmptyBiases[name];
    }

    bool Step3p5Model::ForwardSingleGPUDecodeGraph(
            int gpuId,
            std::map <int, int> ratios,
            int batch,
            const Data &inputIds,
            const Data &positionIds,
            const std::vector <int> &seqLens,
            std::vector <std::pair <Data*, Data*> > &pastKeyValues,
            bool all1,
            bool isPrefill,
            bool tensorParallel,
            bool firstTensorParallelRank,
            int pagedCacheLayerOffset,
            Data &logits) {
#ifndef USE_CUDA
        return false;
#else
        auto rejectGraph = [](const std::string &) -> bool {
            return false;
        };
        if (step3p7VisionAvailable && !Step3p7CudaGraphDecodeEnabled()) {
            return rejectGraph("Step3.7 decode graph disabled");
        }
        if (!Step3p5CudaGraphEnabled(this)) {
            return rejectGraph("disabled");
        }
        const int maxCudaGraphDecodeBatch = 32;
        if (batch <= 0 || batch > maxCudaGraphDecodeBatch ||
            !all1 || isPrefill || (int)seqLens.size() < batch ||
            (int)pastKeyValues.size() < batch * block_cnt) {
            return rejectGraph("not single-token decode");
        }
        for (int b = 0; b < batch; b++) {
            if (seqLens[b] != 1) {
                return rejectGraph("not single-token decode");
            }
        }
        if (inputIds.Count(0) != (uint64_t)batch || positionIds.Count(0) != (uint64_t)batch) {
            return rejectGraph("input/position count mismatch");
        }

        int graphParticipants = tensorParallel ? std::max(2, (int)ratios.size()) : 1;
        auto syncGraphPeers = [&](bool ok = true) {
            return Step3p5CudaGraphSyncPhase(this, graphParticipants, ok);
        };

        auto &fusedMoeByDevice = tensorParallel ? threadTpFusedMoeWeights : singleGpuFusedMoeWeights;
        auto &fusedMoeRangesByDevice = tensorParallel ?
            threadTpFusedMoeExpertRanges : singleGpuFusedMoeExpertRanges;
        auto canUseFusedMoeGraph = [&]() -> bool {
            auto fusedIt = fusedMoeByDevice.find(gpuId);
            auto rangeIt = fusedMoeRangesByDevice.find(gpuId);
            if (fusedIt == fusedMoeByDevice.end() || rangeIt == fusedMoeRangesByDevice.end()) {
                return false;
            }
            for (int i = 0; i < block_cnt; i++) {
                if (!IsMoeLayer(i)) {
                    continue;
                }
                if (i >= (int)fusedIt->second.size() || i >= (int)rangeIt->second.size()) {
                    return false;
                }
                auto &localFusedWeights = fusedIt->second[i];
                std::pair<int, int> expertRange = rangeIt->second[i];
                if ((int)localFusedWeights.size() != 3 ||
                    localFusedWeights[0] == nullptr ||
                    localFusedWeights[1] == nullptr ||
                    localFusedWeights[2] == nullptr ||
                    expertRange.first < 0 ||
                    expertRange.first >= expertRange.second ||
                    expertRange.second > num_experts) {
                    return false;
                }
            }
            return true;
        };
        if (!syncGraphPeers(canUseFusedMoeGraph())) {
            return rejectGraph("missing fused moe graph weights");
        }

        auto requireLocal = [&](Data &data, const std::string &name) -> Data* {
            auto it = data.multiDeviceDatas.find(gpuId);
            if (it != data.multiDeviceDatas.end() && it->second != nullptr) {
                return it->second;
            }
            if (!tensorParallel) {
                if (!data.dims.empty() &&
                    (data.dataDevice != DataDevice::CUDA || data.cudaData == nullptr ||
                     data.dataDeviceIds.empty() || data.dataDeviceIds[0] != gpuId)) {
                    data.ToDevice(DataDevice::CUDA, {gpuId}, true);
                }
                return &data;
            }
            ErrorInFastLLM("Step3p5 ForwardSingleGPU graph missing local tensor: " + name + ".\n");
            return nullptr;
        };

        Data *localInputIds = requireLocal((Data&)inputIds, "inputIds");
        Data *localPositionIds = requireLocal((Data&)positionIds, "positionIds");
        bool inputOk = localInputIds->dims.size() == 2 &&
                       localInputIds->Count(0) == (uint64_t)batch &&
                       !localPositionIds->dims.empty() &&
                       localPositionIds->Count(0) == (uint64_t)batch;
        if (!syncGraphPeers(inputOk)) {
            return rejectGraph("input/position dims mismatch");
        }

        bool kvMetaOk = true;
        for (int i = 0; i < block_cnt; i++) {
            Data *firstBatchKey = pastKeyValues[i].first;
            Data *firstBatchValue = pastKeyValues[i].second;
            if (firstBatchKey == nullptr || firstBatchValue == nullptr ||
                firstBatchKey->pagedKVCacheData == nullptr ||
                firstBatchValue->pagedKVCacheData == nullptr) {
                kvMetaOk = false;
                break;
            }
            for (int b = 0; b < batch; b++) {
                Data *pastKey = pastKeyValues[b * block_cnt + i].first;
                Data *pastValue = pastKeyValues[b * block_cnt + i].second;
                if (pastKey == nullptr || pastValue == nullptr ||
                    pastKey->pagedKVCacheData == nullptr || pastValue->pagedKVCacheData == nullptr ||
                    pastKey->pagedKVCacheData != firstBatchKey->pagedKVCacheData ||
                    pastValue->pagedKVCacheData != firstBatchValue->pagedKVCacheData ||
                    pastKey->pageIndex.empty() || pastValue->pageIndex.empty() ||
                    pastKey->dataDevice != DataDevice::CUDA || pastValue->dataDevice != DataDevice::CUDA ||
                    pastKey->dataDeviceIds.empty() || pastValue->dataDeviceIds.empty() ||
                    pastKey->dataDeviceIds[0] != gpuId || pastValue->dataDeviceIds[0] != gpuId ||
                    pastKey->dataType == DataType::FP8_E4M3 || pastValue->dataType == DataType::FP8_E4M3 ||
                    pastKey->pageLen <= 0 || pastKey->pageLen != pastValue->pageLen ||
                    pastKey->pageIndex.size() != pastValue->pageIndex.size() ||
                    pastKey->lastPageLen != pastValue->lastPageLen) {
                    kvMetaOk = false;
                    break;
                }
            }
            if (!kvMetaOk) {
                break;
            }
        }
        if (!syncGraphPeers(kvMetaOk)) {
            return rejectGraph("unsupported kv cache metadata");
        }

        Step3p5CudaGraphDecodeState &state = GetStep3p5CudaGraphDecodeState(this, gpuId, batch);
        std::unique_lock<std::mutex> graphLock(state.mutex);
        if (!syncGraphPeers(!Step3p5CudaGraphIsDisabled(this))) {
            return false;
        }
        if (!syncGraphPeers(!state.disabled)) {
            if (state.disabled) {
                Step3p5DisableCudaGraph(this);
            }
            return false;
        }

        FastllmCudaSetDevice(gpuId);
        std::vector<int> graphTokenDims = {1, batch};
        Step3p5PrepareGraphCudaTensorWithDims(state.inputIds, *localInputIds, gpuId, graphTokenDims);
        Step3p5PrepareGraphCudaTensorWithDims(state.positionIds, *localPositionIds, gpuId, graphTokenDims);

        PagedCacheManager *graphPagedManager = pastKeyValues[0].first->pagedKVCacheData;
        int graphMaxPagesPerRequest = graphPagedManager != nullptr ? graphPagedManager->maxPages : 0;
        if (graphMaxPagesPerRequest <= 0 && graphPagedManager != nullptr && !graphPagedManager->dims.empty()) {
            graphMaxPagesPerRequest = graphPagedManager->dims[0];
        }
        if (graphMaxPagesPerRequest <= 0) {
            return rejectGraph("invalid paged cache capacity");
        }

        std::vector<int> insertIndexHost(batch, -1);
        std::vector<int> insertPositionHost(batch, 0);
        std::vector<int> lastPageLensHost(batch, 0);
        std::vector<int> qSizesHost(batch + 1, 0);
        std::vector<int> pageSizesHost(batch + 1, 0);
        std::vector<int> graphPlanPageSizesHost(batch + 1, 0);
        std::vector<int> pageIndexHost;
        std::vector<const Data*> currentPastKeyHosts(batch, nullptr);
        std::vector<char> needNewPage(batch, 0);
        bool anyNewPage = false;
        int maxActualPagesPerRequest = 1;

        for (int b = 0; b < batch; b++) {
            Data *firstKey = pastKeyValues[b * block_cnt].first;
            currentPastKeyHosts[b] = firstKey;
            bool need = firstKey->pageIndex.empty() || firstKey->lastPageLen >= firstKey->pageLen;
            needNewPage[b] = need ? 1 : 0;
            anyNewPage = anyNewPage || need;
            if (!need) {
                insertIndexHost[b] = firstKey->pageIndex.back();
                insertPositionHost[b] = firstKey->lastPageLen;
            }
        }

        for (int i = 0; i < block_cnt; i++) {
            for (int b = 0; b < batch; b++) {
                Data *pastKey = pastKeyValues[b * block_cnt + i].first;
                Data *pastValue = pastKeyValues[b * block_cnt + i].second;
                bool layerNeedNewPage = pastKey->pageIndex.empty() || pastKey->lastPageLen >= pastKey->pageLen;
                AssertInFastLLM(layerNeedNewPage == (needNewPage[b] != 0),
                                "Step3p5 CUDA graph requires aligned paged cache layout across layers.\n");
                if (needNewPage[b]) {
                    int keyPage = pastKey->pagedKVCacheData->GetUnusedPageIndex(true);
                    int valuePage = pastValue->pagedKVCacheData->GetUnusedPageIndex(true);
                    if (insertIndexHost[b] < 0) {
                        insertIndexHost[b] = keyPage;
                    }
                    AssertInFastLLM(keyPage == insertIndexHost[b] && valuePage == insertIndexHost[b],
                                    "Step3p5 CUDA graph requires aligned K/V page indices across layers.\n");
                    pastKey->pageIndex.push_back(keyPage);
                    pastValue->pageIndex.push_back(valuePage);
                    pastKey->lastPageLen = 1;
                    pastValue->lastPageLen = 1;
                } else {
                    AssertInFastLLM(pastKey->pageIndex.back() == insertIndexHost[b] &&
                                    pastValue->pageIndex.back() == insertIndexHost[b] &&
                                    pastKey->lastPageLen == insertPositionHost[b] &&
                                    pastValue->lastPageLen == insertPositionHost[b],
                                    "Step3p5 CUDA graph requires aligned paged cache positions across layers.\n");
                    pastKey->lastPageLen++;
                    pastValue->lastPageLen++;
                }
            }
        }

        pageIndexHost.reserve(batch);
        for (int b = 0; b < batch; b++) {
            qSizesHost[b + 1] = qSizesHost[b] + 1;
            Data *firstKey = pastKeyValues[b * block_cnt].first;
            Data *firstValue = pastKeyValues[b * block_cnt].second;
            lastPageLensHost[b] = firstKey->lastPageLen;
            int requestPages = (int)firstKey->pageIndex.size();
            AssertInFastLLM(requestPages <= graphMaxPagesPerRequest,
                            "Step3p5 CUDA graph page metadata exceeds captured capacity.\n");
            maxActualPagesPerRequest = std::max(maxActualPagesPerRequest, requestPages);
            pageSizesHost[b + 1] = pageSizesHost[b] + requestPages;
            pageIndexHost.insert(pageIndexHost.end(),
                                 firstKey->pageIndex.begin(), firstKey->pageIndex.end());
            AssertInFastLLM(firstKey->pageIndex.size() == firstValue->pageIndex.size() &&
                            firstKey->lastPageLen == firstValue->lastPageLen,
                            "Step3p5 CUDA graph requires aligned K/V page metadata.\n");
            for (int i = 1; i < block_cnt; i++) {
                Data *pastKey = pastKeyValues[b * block_cnt + i].first;
                Data *pastValue = pastKeyValues[b * block_cnt + i].second;
                AssertInFastLLM(pastKey->pageIndex == firstKey->pageIndex &&
                                pastValue->pageIndex == firstKey->pageIndex &&
                                pastKey->lastPageLen == firstKey->lastPageLen &&
                                pastValue->lastPageLen == firstKey->lastPageLen,
                                "Step3p5 CUDA graph requires aligned paged cache pages across layers.\n");
            }
        }
        int graphPlanPagesPerRequest = 1;
        while (graphPlanPagesPerRequest < maxActualPagesPerRequest &&
               graphPlanPagesPerRequest < graphMaxPagesPerRequest) {
            graphPlanPagesPerRequest <<= 1;
        }
        graphPlanPagesPerRequest = std::min(graphPlanPagesPerRequest, graphMaxPagesPerRequest);
        for (int b = 0; b < batch; b++) {
            graphPlanPageSizesHost[b + 1] =
                graphPlanPageSizesHost[b] + graphPlanPagesPerRequest;
        }
        int pageIndexCapacity = batch * graphPlanPagesPerRequest;

        std::ostringstream signature;
        signature << "gpu=" << gpuId
                  << ";tp=" << (tensorParallel ? 1 : 0)
                  << ";tpRank0=" << (firstTensorParallelRank ? 1 : 0)
                  << ";batch=" << batch
                  << ";inputDims=";
        for (int dim : state.inputIds.dims) {
            signature << dim << ",";
        }
        signature << ";posDims=";
        for (int dim : state.positionIds.dims) {
            signature << dim << ",";
        }
        signature << ";pageSizes=";
        for (int pageSize : graphPlanPageSizesHost) {
            signature << pageSize << ",";
        }
        signature << ";pages=" << pageIndexCapacity
                  << ";inputType=" << (int)state.inputIds.dataType
                  << ";posType=" << (int)state.positionIds.dataType
                  << ";kCache=" << pastKeyValues[0].first->pagedKVCacheData->cudaData
                  << ";vCache=" << pastKeyValues[0].second->pagedKVCacheData->cudaData
                  << ";lmLocal=" << requireLocal(weight["lm_head.weight"], "lm_head.weight")->dims[0];
        std::string newSignature = signature.str();
        bool signatureChanged = state.signature != newSignature;
        if (signatureChanged) {
            Step3p5DestroyCudaGraph(state);
            state.signature = newSignature;
        }

        bool requestStateChanged = state.lastPastKeyHosts != currentPastKeyHosts;
        bool graphMetaMissing =
            state.metaBuffers.insertIndexs.cudaData == nullptr ||
            state.metaBuffers.insertPositions.cudaData == nullptr ||
            state.metaBuffers.qSizes.cudaData == nullptr ||
            state.metaBuffers.pageSizes.cudaData == nullptr ||
            state.metaBuffers.pageIndexs.cudaData == nullptr ||
            state.metaBuffers.lastPageLens.cudaData == nullptr;
        bool metadataChanged =
            state.lastInsertIndexHost != insertIndexHost ||
            state.lastPageSizesHost != pageSizesHost ||
            state.lastPageIndexHost != pageIndexHost ||
            state.lastDecodePageLensHost != insertPositionHost ||
            requestStateChanged;
        bool needFullMetaCopy = graphMetaMissing || signatureChanged || anyNewPage || metadataChanged;
        if (needFullMetaCopy) {
            AssertInFastLLM((int)pageIndexHost.size() <= pageIndexCapacity,
                            "Step3p5 CUDA graph page metadata exceeds fixed graph capacity.\n");
            std::vector<int> paddedPageIndexHost = pageIndexHost;
            paddedPageIndexHost.resize(pageIndexCapacity,
                                       paddedPageIndexHost.empty() ? 0 : paddedPageIndexHost.back());

            Step3p5PrepareGraphIntTensor(state.metaBuffers.insertIndexs, gpuId, insertIndexHost);
            Step3p5PrepareGraphIntTensor(state.metaBuffers.insertPositions, gpuId, insertPositionHost);
            Step3p5PrepareGraphIntTensor(state.metaBuffers.qSizes, gpuId, qSizesHost);
            Step3p5PrepareGraphIntTensor(state.metaBuffers.pageSizes, gpuId, pageSizesHost);
            // FlashInfer graph planning uses the CPU indptr; kernels still read the real CUDA indptr.
            state.metaBuffers.pageSizes.cpuIntDatas = graphPlanPageSizesHost;
            Step3p5PrepareGraphIntTensor(state.metaBuffers.pageIndexs, gpuId, paddedPageIndexHost);
            Step3p5PrepareGraphIntTensor(state.metaBuffers.lastPageLens, gpuId, lastPageLensHost);
            state.lastInsertIndexHost = insertIndexHost;
            state.lastPageSizesHost = pageSizesHost;
            state.lastPageIndexHost = pageIndexHost;
            state.lastDecodePageLensHost = lastPageLensHost;
            state.lastPastKeyHosts = currentPastKeyHosts;
        } else {
            FastllmCudaSetDevice(gpuId);
            if (!FastllmCudaAdvanceDecodeMeta(
                    (int32_t*)state.metaBuffers.insertPositions.cudaData,
                    (int32_t*)state.metaBuffers.lastPageLens.cudaData,
                    batch)) {
                return rejectGraph("advance decode meta failed");
            }
            state.metaBuffers.insertPositions.cpuIntDatas = insertPositionHost;
            state.metaBuffers.lastPageLens.cpuIntDatas = lastPageLensHost;
            state.lastDecodePageLensHost = lastPageLensHost;
            state.lastPastKeyHosts = currentPastKeyHosts;
        }

        const DataType computeType = ResolveStep3p5ThreadTpComputeType(this->dataType);
        auto runGraphBodyWithBuffers = [&](Step3p5ForwardSingleBuffers &buf,
                                           Step3p5ForwardSingleBuffers &metaBuf) {
            Qwen3CudaDirectRunner cudaRunner(gpuId);
            if ((int)buf.batchPastKeys.size() != batch) {
                buf.batchPastKeys.resize(batch);
                buf.batchPastValues.resize(batch);
            }

            Qwen3CudaEmbeddingDirect(cudaRunner,
                                     state.inputIds,
                                     *requireLocal(weight["model.embed_tokens.weight"], "model.embed_tokens.weight"),
                                     buf.hiddenStates);
            if (buf.hiddenStates.dataType != computeType) {
                Qwen3CudaToDataType(cudaRunner, buf.hiddenStates, computeType);
            }

            auto runFeedForwardOutput = [&](Data &input,
                                            const std::string &gateupName,
                                            const std::string &gateName,
                                            const std::string &upName,
                                            const std::string &downName,
                                            float limit,
                                            Data &middle,
                                            Data &act,
                                            Data &upOut,
                                            Data &output) {
                if (limit == 0.0f && weight.weight.find(gateupName) != weight.weight.end()) {
                    Qwen3CudaLinearSwiglu(cudaRunner, input,
                                          *requireLocal(weight[gateupName], gateupName),
                                          *requireLocal(GetThreadTensorParallelBias(gateupName + ".tp_bias"),
                                                        gateupName + ".tp_bias"),
                                          middle, act);
                } else {
                    Qwen3CudaLinear(cudaRunner, input,
                                    *requireLocal(weight[gateName], gateName),
                                    *requireLocal(GetThreadTensorParallelBias(gateName + ".tp_bias"),
                                                  gateName + ".tp_bias"),
                                    act);
                    Step3p5CudaSilu(cudaRunner, act, act);
                    Qwen3CudaLinear(cudaRunner, input,
                                    *requireLocal(weight[upName], upName),
                                    *requireLocal(GetThreadTensorParallelBias(upName + ".tp_bias"),
                                                  upName + ".tp_bias"),
                                    upOut);
                    if (limit != 0.0f) {
                        Step3p5CudaClamp(act, false, 0.0f, true, limit, gpuId);
                        Step3p5CudaClamp(upOut, true, -limit, true, limit, gpuId);
                    }
                    if (upOut.dataType != act.dataType) {
                        Qwen3CudaToDataType(cudaRunner, upOut, act.dataType);
                    }
                    Step3p5CudaMulTo(cudaRunner, act, upOut);
                }
                Qwen3CudaLinear(cudaRunner, act,
                                *requireLocal(weight[downName], downName),
                                *requireLocal(GetThreadTensorParallelBias(downName + ".tp_bias"),
                                              downName + ".tp_bias"),
                                output);
            };

            auto addPartialToResidualReduce = [&](Data &partial) {
                if (partial.dataType != buf.hiddenStates.dataType) {
                    Qwen3CudaToDataType(cudaRunner, partial, buf.hiddenStates.dataType);
                }
                if (tensorParallel) {
                    if (firstTensorParallelRank) {
                        Qwen3CudaAddTo(cudaRunner, buf.hiddenStates, partial);
                    } else {
                        Step3p5CudaCopyTensor(cudaRunner, partial, buf.hiddenStates);
                    }
                    FastllmNcclAllReduce(buf.hiddenStates.cudaData, buf.hiddenStates.cudaData,
                                         buf.hiddenStates.Count(0), buf.hiddenStates.dataType, gpuId);
                } else {
                    Qwen3CudaAddTo(cudaRunner, buf.hiddenStates, partial);
                }
            };

            bool generatedAppendParams = false;
            bool generatedDecodeParams = false;
            for (int i = 0; i < block_cnt; i++) {
                std::string prefix = "model.layers." + std::to_string(i) + ".";
                std::string inputRmsName = prefix + "input_layernorm.weight";
                std::string postRmsName = prefix + "post_attention_layernorm.weight";
                std::string mergeQkvWeightName = prefix + "self_attn.mergeqkv.weight";
                std::string mergeQkvBiasName = prefix + "self_attn.mergeqkv.bias";
                std::string qNormName = prefix + "self_attn.q_norm.weight";
                std::string kNormName = prefix + "self_attn.k_norm.weight";
                std::string gProjName = prefix + "self_attn.g_proj.weight";
                std::string oWeightName = prefix + "self_attn.o_proj.weight";
                std::string oBiasName = prefix + "self_attn.o_proj.bias";
                int qHeads = LayerAttentionHeads(i);
                int kvHeads = LayerKeyValueHeads(i);
                int curRotaryDim = i < (int)layer_rotary_dims.size() ? layer_rotary_dims[i] : head_dim;
                float curTheta = i < (int)layer_rope_thetas.size() ? layer_rope_thetas[i] : rope_base;

                Qwen3CudaRMSNorm(cudaRunner, buf.hiddenStates,
                                 *requireLocal(weight[inputRmsName], inputRmsName),
                                 rms_norm_eps, buf.attenInput);

                Data *localMergeW = requireLocal(weight[mergeQkvWeightName], mergeQkvWeightName);
                int group = qHeads / kvHeads;
                int localKVHeads = localMergeW->tpKVHeads > 0 ?
                    localMergeW->tpKVHeads : localMergeW->dims[0] / ((group + 2) * head_dim);
                int localQHeads = localMergeW->tpQHeads > 0 ?
                    localMergeW->tpQHeads : localKVHeads * group;
                AssertInFastLLM(localKVHeads > 0 && localQHeads > 0,
                                "Step3p5 ForwardSingleGPU graph got empty local attention shard.\n");

                const bool enableStableFlashInferGraphPlan = true;
                const int flashInferCudaGraph = 1;
                Step3p5CudaAttentionPagedBlock(
                    cudaRunner,
                    &buf.attenInput,
                    localMergeW, requireLocal(GetThreadTensorParallelBias(mergeQkvBiasName), mergeQkvBiasName),
                    requireLocal(weight[qNormName], qNormName),
                    requireLocal(weight[kNormName], kNormName),
                    &state.positionIds,
                    &pastKeyValues,
                    &buf.batchPastKeys, &buf.batchPastValues,
                    &buf.qkv, &buf.q, &buf.k, &buf.v, &buf.attenOutput,
                    &buf.qForAttentionHolder,
                    &metaBuf.insertIndexs, &metaBuf.insertPositions,
                    &metaBuf.qSizes, &metaBuf.pageSizes, &metaBuf.pageIndexs, &metaBuf.lastPageLens,
                    &generatedAppendParams, &generatedDecodeParams,
                    batch, block_cnt, i,
                    seqLens,
                    localQHeads, localKVHeads, head_dim,
                    curRotaryDim, rms_norm_eps,
                    curTheta, rope_factor, UseLlama3Rope(i),
                    llama3_original_max_position_embeddings,
                    llama3_low_freq_factor,
                    llama3_high_freq_factor,
                    GetKVCacheInCPU(),
                    pagedCacheLayerOffset,
                    false,
                    true,
                    enableStableFlashInferGraphPlan,
                    flashInferCudaGraph
                );
                if (buf.attenOutput.dataType != computeType) {
                    Qwen3CudaToDataType(cudaRunner, buf.attenOutput, computeType);
                }

                Qwen3CudaLinear(cudaRunner, buf.attenInput,
                                *requireLocal(weight[gProjName], gProjName),
                                *GetEmptyData(), buf.gate);
                Step3p5CudaSigmoid(cudaRunner, buf.gate, buf.gate);
                int bsz = buf.attenInput.dims[0], seqlen = buf.attenInput.dims[1];
                buf.gate.Reshape({bsz, seqlen, localQHeads, 1});
                if (buf.gate.dataType != buf.attenOutput.dataType) {
                    Qwen3CudaToDataType(cudaRunner, buf.gate, buf.attenOutput.dataType);
                }
                buf.attenOutput.Reshape({bsz, seqlen, localQHeads, head_dim});
                Step3p5CudaMulTo(cudaRunner, buf.attenOutput, buf.gate);
                buf.attenOutput.Reshape({bsz, seqlen, localQHeads * head_dim});

                Qwen3CudaLinearResidualReduce(
                    cudaRunner, buf.attenOutput,
                    *requireLocal(weight[oWeightName], oWeightName),
                    *requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                    buf.attenLastOutput, buf.hiddenStates,
                    tensorParallel, firstTensorParallelRank, gpuId);

                Qwen3CudaRMSNorm(cudaRunner, buf.hiddenStates,
                                 *requireLocal(weight[postRmsName], postRmsName),
                                 rms_norm_eps, buf.attenInput);
                if (!IsMoeLayer(i)) {
                    runFeedForwardOutput(buf.attenInput,
                                         prefix + "mlp.gateup_proj.weight",
                                         prefix + "mlp.gate_proj.weight",
                                         prefix + "mlp.up_proj.weight",
                                         prefix + "mlp.down_proj.weight",
                                         Step3p5LayerLimit(swiglu_limits_shared, i),
                                         buf.ffMiddle, buf.ffAct, buf.ffUp, buf.ffOut);
                    addPartialToResidualReduce(buf.ffOut);
                } else {
                    runFeedForwardOutput(buf.attenInput,
                                         prefix + "share_expert.gateup_proj.weight",
                                         prefix + "share_expert.gate_proj.weight",
                                         prefix + "share_expert.up_proj.weight",
                                         prefix + "share_expert.down_proj.weight",
                                         Step3p5LayerLimit(swiglu_limits_shared, i),
                                         buf.ffMiddle, buf.ffAct, buf.ffUp, buf.shareOutput);
                    if (buf.shareOutput.dataType != buf.hiddenStates.dataType) {
                        Qwen3CudaToDataType(cudaRunner, buf.shareOutput, buf.hiddenStates.dataType);
                    }
                    if (tensorParallel) {
                        FastllmNcclAllReduce(buf.shareOutput.cudaData, buf.shareOutput.cudaData,
                                             buf.shareOutput.Count(0), buf.shareOutput.dataType, gpuId);
                    }
                    Qwen3CudaAddTo(cudaRunner, buf.hiddenStates, buf.shareOutput);

                    int flatBatch = buf.attenInput.dims[0];
                    int flatLen = buf.attenInput.dims[1];
                    buf.attenInput.Reshape({flatBatch * flatLen, buf.attenInput.dims[2]});
                    Qwen3CudaLinear(cudaRunner, buf.attenInput,
                                    *requireLocal(weight[prefix + "moe.gate.weight"], prefix + "moe.gate.weight"),
                                    *GetEmptyData(), buf.routerLogits, true);
                    Qwen3CudaConvertToDataType(cudaRunner, buf.routerLogits, buf.routerProb, DataType::FLOAT32);
                    Step3p5CudaSigmoid(cudaRunner, buf.routerProb, buf.routerProb);
                    Data *localGateBias = nullptr;
                    if (use_moe_router_bias &&
                        weight.weight.find(prefix + "moe.router_bias") != weight.weight.end()) {
                        localGateBias = requireLocal(weight[prefix + "moe.router_bias"], prefix + "moe.router_bias");
                    }
                    Qwen3CudaSelectExpert(cudaRunner, buf.routerProb, buf.expertIndex, buf.expertScore,
                                          num_experts_per_tok, norm_topk_prob,
                                          routed_scaling_factor, localGateBias);

                    auto &localFusedWeights = fusedMoeByDevice.at(gpuId)[i];
                    std::pair<int, int> expertRange = fusedMoeRangesByDevice.at(gpuId)[i];
                    AssertInFastLLM((int)localFusedWeights.size() == 3 &&
                                    localFusedWeights[0] != nullptr &&
                                    localFusedWeights[1] != nullptr &&
                                    localFusedWeights[2] != nullptr &&
                                    Step3p5MaskAndRemapExpertsForLocalRange(
                                        buf.expertIndex, buf.expertScore,
                                        expertRange.first, expertRange.second,
                                        num_experts, gpuId, true),
                                    "Step3p5 CUDA graph requires local fused MoE expert weights.\n");
                    Step3p5CudaFusedMOE(cudaRunner, buf.attenInput, buf.expertIndex, buf.expertScore,
                                        *localFusedWeights[0], *localFusedWeights[1],
                                        *localFusedWeights[2], buf.w1, buf.moeFinal, i,
                                        Step3p5LayerLimit(swiglu_limits, i));
                    buf.moeFinal.Reshape(buf.hiddenStates.dims);
                    if (buf.moeFinal.dataType != buf.hiddenStates.dataType) {
                        Qwen3CudaToDataType(cudaRunner, buf.moeFinal, buf.hiddenStates.dataType);
                    }
                    if (tensorParallel) {
                        FastllmNcclAllReduce(buf.moeFinal.cudaData, buf.moeFinal.cudaData,
                                             buf.moeFinal.Count(0), buf.moeFinal.dataType, gpuId);
                    }
                    Qwen3CudaAddTo(cudaRunner, buf.hiddenStates, buf.moeFinal);
                }
            }

            Qwen3CudaRMSNorm(cudaRunner, buf.hiddenStates,
                             *requireLocal(weight["model.norm.weight"], "model.norm.weight"),
                             rms_norm_eps, buf.hiddenStates);
            Qwen3CudaLinear(cudaRunner, buf.hiddenStates,
                            *requireLocal(weight["lm_head.weight"], "lm_head.weight"),
                            *requireLocal(GetThreadTensorParallelBias("lm_head.weight.tp_bias"),
                                          "lm_head.weight.tp_bias"),
                            state.logitsHalf);
            Qwen3CudaConvertToDataType(cudaRunner, state.logitsHalf, state.logits, DataType::FLOAT32);
        };

        auto runGraphBody = [&]() {
            runGraphBodyWithBuffers(state.buffers, state.metaBuffers);
        };

        auto finishWithLogits = [&]() {
            Step3p5PrepareGraphCudaTensor(logits, state.logits, gpuId);
        };

        auto runWithoutGraph = [&]() {
            runGraphBody();
            finishWithLogits();
        };

        if (state.captured) {
            if (requestStateChanged) {
                runWithoutGraph();
                return true;
            }
            if (!syncGraphPeers()) {
                return false;
            }
            bool launchOk = FastllmCudaGraphLaunch(state.exec);
            if (Step3p5SyncCudaGraphStage(this, state, graphParticipants,
                                          "replay", gpuId, launchOk)) {
                finishWithLogits();
                return true;
            }
            runWithoutGraph();
            return true;
        }

        if (!state.warmed) {
            runWithoutGraph();
            if (!syncGraphPeers()) {
                Step3p5DisableCudaGraphState(this, state);
                return true;
            }
            state.warmed = true;
            return true;
        }

        void *capturedGraph = nullptr;
        if (!syncGraphPeers()) {
            return false;
        }
        bool beginOk = FastllmCudaGraphBeginCapture();
        if (!Step3p5SyncCudaGraphStage(this, state, graphParticipants,
                                       "begin capture", gpuId, beginOk)) {
            if (beginOk) {
                Step3p5AbortCudaGraphCapture();
            }
            runWithoutGraph();
            return true;
        }
        runGraphBody();
        if (!syncGraphPeers()) {
            Step3p5AbortCudaGraphCapture();
            Step3p5DisableCudaGraphState(this, state);
            runWithoutGraph();
            return true;
        }
        bool endOk = FastllmCudaGraphEndCapture(&capturedGraph) && capturedGraph != nullptr;
        if (!Step3p5SyncCudaGraphStage(this, state, graphParticipants,
                                       "end capture", gpuId, endOk)) {
            if (capturedGraph != nullptr) {
                FastllmCudaGraphDestroy(capturedGraph);
            }
            runWithoutGraph();
            return true;
        }

        void *capturedExec = nullptr;
        bool instantiateOk = FastllmCudaGraphInstantiate(capturedGraph, &capturedExec) &&
                             capturedExec != nullptr;
        if (!Step3p5SyncCudaGraphStage(this, state, graphParticipants,
                                       "instantiate", gpuId, instantiateOk)) {
            if (capturedExec != nullptr) {
                FastllmCudaGraphExecDestroy(capturedExec);
            }
            FastllmCudaGraphDestroy(capturedGraph);
            runWithoutGraph();
            return true;
        }

        state.graph = capturedGraph;
        state.exec = capturedExec;
        state.captured = true;
        if (!syncGraphPeers()) {
            return false;
        }
        bool firstLaunchOk = FastllmCudaGraphLaunch(state.exec);
        if (!Step3p5SyncCudaGraphStage(this, state, graphParticipants,
                                       "first launch", gpuId, firstLaunchOk)) {
            runWithoutGraph();
            return true;
        }
        finishWithLogits();
        return true;
#endif
    }

    void Step3p5Model::ForwardSingleGPU(
            int gpuId,
            std::map <int, int> ratios,
            int batch,
            const Data &inputIds,
            const Data &positionIds,
            const std::vector <int> &seqLens,
            std::vector <std::pair <Data*, Data*> > &pastKeyValues,
            bool all1,
            bool isPrefill,
            bool tensorParallel,
            bool firstTensorParallelRank,
            int pagedCacheLayerOffset,
            Data &logits,
            Data *precomputedHiddenStates) {
#ifndef USE_CUDA
        ErrorInFastLLM("Step3p5 ForwardSingleGPU requires CUDA.\n");
#else
        AssertInFastLLM(ratios.find(gpuId) == ratios.end() || ratios[gpuId] > 0,
                        "Step3p5 ForwardSingleGPU got invalid GPU ratio.\n");
        FastllmCudaSetDevice(gpuId);
        Qwen3CudaDirectRunner cudaRunner(gpuId);

        auto requireLocal = [&](Data &data, const std::string &name) -> Data* {
            auto it = data.multiDeviceDatas.find(gpuId);
            if (it != data.multiDeviceDatas.end() && it->second != nullptr) {
                return it->second;
            }
            if (!tensorParallel) {
                if (!data.dims.empty() &&
                    (data.dataDevice != DataDevice::CUDA || data.cudaData == nullptr ||
                     data.dataDeviceIds.empty() || data.dataDeviceIds[0] != gpuId)) {
                    data.ToDevice(DataDevice::CUDA, {gpuId}, true);
                }
                return &data;
            }
            ErrorInFastLLM("Step3p5 ForwardSingleGPU missing local tensor: " + name + ".\n");
            return nullptr;
        };

        const DataType computeType = ResolveStep3p5ThreadTpComputeType(this->dataType);
        Data localHiddenStates;
        Data *hiddenStatesPtr = nullptr;
        if (precomputedHiddenStates != nullptr) {
            hiddenStatesPtr = requireLocal(*precomputedHiddenStates, "precomputedHiddenStates");
        } else {
            if (ForwardSingleGPUDecodeGraph(gpuId, ratios, batch, inputIds, positionIds,
                                            seqLens, pastKeyValues, all1, isPrefill,
                                            tensorParallel, firstTensorParallelRank,
                                            pagedCacheLayerOffset, logits)) {
                return;
            }
            Qwen3CudaEmbeddingDirect(cudaRunner,
                                     *requireLocal((Data&)inputIds, "inputIds"),
                                     *requireLocal(weight["model.embed_tokens.weight"], "model.embed_tokens.weight"),
                                     localHiddenStates);
            hiddenStatesPtr = &localHiddenStates;
        }
        Data &hiddenStates = *hiddenStatesPtr;
        if (hiddenStates.dataType != computeType) {
            Qwen3CudaToDataType(cudaRunner, hiddenStates, computeType);
        }

        Data attenInput, qkv, q, qForAttentionHolder, attenOutput, attenLastOutput;
        Data gate;
        Data ffMiddle, ffAct, ffUp, ffOut;
        Data routerLogits, routerProb, expertIndex, expertScore;
        Data w1, w2, w3, tempInput, tempOutput, moeInputTemp, moeOutputTemp, moeFinal, shareOutput;
        Data qSizes, pageSizes, pageIndexs, lastPageLens, insertIndexs, insertPositions;
        bool generatedAppendParams = false;
        bool generatedDecodeParams = false;
        std::vector<Data*> batchPastKeys(batch), batchPastValues(batch);
        auto &moeWeightsByDevice = tensorParallel ? threadTpMoeWeights : singleGpuMoeWeights;
        auto &moeBiassByDevice = tensorParallel ? threadTpMoeBiass : singleGpuMoeBiass;
        auto &fusedMoeByDevice = tensorParallel ? threadTpFusedMoeWeights : singleGpuFusedMoeWeights;
        auto &fusedMoeRangesByDevice = tensorParallel ?
            threadTpFusedMoeExpertRanges : singleGpuFusedMoeExpertRanges;

        auto runFeedForwardOutput = [&](Data &input,
                                        const std::string &gateupName,
                                        const std::string &gateName,
                                        const std::string &upName,
                                        const std::string &downName,
                                        float limit,
                                        Data &middle,
                                        Data &act,
                                        Data &upOut,
                                        Data &output) {
            if (limit == 0.0f && weight.weight.find(gateupName) != weight.weight.end()) {
                Qwen3CudaLinearSwiglu(cudaRunner, input,
                                      *requireLocal(weight[gateupName], gateupName),
                                      *requireLocal(GetThreadTensorParallelBias(gateupName + ".tp_bias"),
                                                    gateupName + ".tp_bias"),
                                      middle, act);
            } else {
                Qwen3CudaLinear(cudaRunner, input,
                                *requireLocal(weight[gateName], gateName),
                                *requireLocal(GetThreadTensorParallelBias(gateName + ".tp_bias"),
                                              gateName + ".tp_bias"),
                                act);
                Step3p5CudaSilu(cudaRunner, act, act);
                Qwen3CudaLinear(cudaRunner, input,
                                *requireLocal(weight[upName], upName),
                                *requireLocal(GetThreadTensorParallelBias(upName + ".tp_bias"),
                                              upName + ".tp_bias"),
                                upOut);
                if (limit != 0.0f) {
                    Step3p5CudaClamp(act, false, 0.0f, true, limit, gpuId);
                    Step3p5CudaClamp(upOut, true, -limit, true, limit, gpuId);
                }
                if (upOut.dataType != act.dataType) {
                    Qwen3CudaToDataType(cudaRunner, upOut, act.dataType);
                }
                Step3p5CudaMulTo(cudaRunner, act, upOut);
            }
            Qwen3CudaLinear(cudaRunner, act,
                            *requireLocal(weight[downName], downName),
                            *requireLocal(GetThreadTensorParallelBias(downName + ".tp_bias"),
                                          downName + ".tp_bias"),
                            output);
        };

        auto addPartialToResidualReduce = [&](Data &partial) {
            if (partial.dataType != hiddenStates.dataType) {
                Qwen3CudaToDataType(cudaRunner, partial, hiddenStates.dataType);
            }
            if (tensorParallel) {
                if (firstTensorParallelRank) {
                    Qwen3CudaAddTo(cudaRunner, hiddenStates, partial);
                } else {
                    Step3p5CudaCopyTensor(cudaRunner, partial, hiddenStates);
                }
                FastllmNcclAllReduce(hiddenStates.cudaData, hiddenStates.cudaData,
                                     hiddenStates.Count(0), hiddenStates.dataType, gpuId);
            } else {
                Qwen3CudaAddTo(cudaRunner, hiddenStates, partial);
            }
        };

        for (int i = 0; i < block_cnt; i++) {
            std::string prefix = "model.layers." + std::to_string(i) + ".";
            std::string inputRmsName = prefix + "input_layernorm.weight";
            std::string postRmsName = prefix + "post_attention_layernorm.weight";
            std::string mergeQkvWeightName = prefix + "self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = prefix + "self_attn.mergeqkv.bias";
            std::string qNormName = prefix + "self_attn.q_norm.weight";
            std::string kNormName = prefix + "self_attn.k_norm.weight";
            std::string gProjName = prefix + "self_attn.g_proj.weight";
            std::string oWeightName = prefix + "self_attn.o_proj.weight";
            std::string oBiasName = prefix + "self_attn.o_proj.bias";
            int qHeads = LayerAttentionHeads(i);
            int kvHeads = LayerKeyValueHeads(i);
            int curRotaryDim = i < (int)layer_rotary_dims.size() ? layer_rotary_dims[i] : head_dim;
            float curTheta = i < (int)layer_rope_thetas.size() ? layer_rope_thetas[i] : rope_base;

            Qwen3CudaRMSNorm(cudaRunner, hiddenStates,
                             *requireLocal(weight[inputRmsName], inputRmsName),
                             rms_norm_eps, attenInput);

            Data *localMergeW = requireLocal(weight[mergeQkvWeightName], mergeQkvWeightName);
            int group = qHeads / kvHeads;
            int localKVHeads = localMergeW->tpKVHeads > 0 ?
                localMergeW->tpKVHeads : localMergeW->dims[0] / ((group + 2) * head_dim);
            int localQHeads = localMergeW->tpQHeads > 0 ?
                localMergeW->tpQHeads : localKVHeads * group;
            AssertInFastLLM(localKVHeads > 0 && localQHeads > 0,
                            "Step3p5 ForwardSingleGPU got empty local attention shard.\n");

            Step3p5CudaAttentionPagedBlock(
                cudaRunner,
                &attenInput,
                localMergeW, requireLocal(GetThreadTensorParallelBias(mergeQkvBiasName), mergeQkvBiasName),
                requireLocal(weight[qNormName], qNormName),
                requireLocal(weight[kNormName], kNormName),
                requireLocal((Data&)positionIds, "positionIds"),
                &pastKeyValues,
                &batchPastKeys, &batchPastValues,
                &qkv, &q, nullptr, nullptr, &attenOutput,
                &qForAttentionHolder,
                &insertIndexs, &insertPositions,
                &qSizes, &pageSizes, &pageIndexs, &lastPageLens,
                &generatedAppendParams, &generatedDecodeParams,
                batch, block_cnt, i,
                seqLens,
                localQHeads, localKVHeads, head_dim,
                curRotaryDim, rms_norm_eps,
                curTheta, rope_factor, UseLlama3Rope(i),
                llama3_original_max_position_embeddings,
                llama3_low_freq_factor,
                llama3_high_freq_factor,
                GetKVCacheInCPU(),
                pagedCacheLayerOffset,
                isPrefill,
                false
            );
            if (attenOutput.dataType != computeType) {
                Qwen3CudaToDataType(cudaRunner, attenOutput, computeType);
            }

            Qwen3CudaLinear(cudaRunner, attenInput,
                            *requireLocal(weight[gProjName], gProjName),
                            *GetEmptyData(), gate);
            Step3p5CudaSigmoid(cudaRunner, gate, gate);
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];
            gate.Reshape({bsz, seqlen, localQHeads, 1});
            if (gate.dataType != attenOutput.dataType) {
                Qwen3CudaToDataType(cudaRunner, gate, attenOutput.dataType);
            }
            attenOutput.Reshape({bsz, seqlen, localQHeads, head_dim});
            Step3p5CudaMulTo(cudaRunner, attenOutput, gate);
            attenOutput.Reshape({bsz, seqlen, localQHeads * head_dim});

            Qwen3CudaLinearResidualReduce(
                cudaRunner, attenOutput,
                *requireLocal(weight[oWeightName], oWeightName),
                *requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                attenLastOutput, hiddenStates,
                tensorParallel, firstTensorParallelRank, gpuId);

            Qwen3CudaRMSNorm(cudaRunner, hiddenStates,
                             *requireLocal(weight[postRmsName], postRmsName),
                             rms_norm_eps, attenInput);
            if (!IsMoeLayer(i)) {
                std::string gateupName = prefix + "mlp.gateup_proj.weight";
                std::string gateName = prefix + "mlp.gate_proj.weight";
                std::string upName = prefix + "mlp.up_proj.weight";
                std::string downName = prefix + "mlp.down_proj.weight";
                runFeedForwardOutput(attenInput, gateupName, gateName, upName, downName,
                                     Step3p5LayerLimit(swiglu_limits_shared, i),
                                     ffMiddle, ffAct, ffUp, ffOut);
                addPartialToResidualReduce(ffOut);
            } else {
                std::string sharedGateupName = prefix + "share_expert.gateup_proj.weight";
                std::string sharedGateName = prefix + "share_expert.gate_proj.weight";
                std::string sharedUpName = prefix + "share_expert.up_proj.weight";
                std::string sharedDownName = prefix + "share_expert.down_proj.weight";
                runFeedForwardOutput(attenInput, sharedGateupName, sharedGateName, sharedUpName, sharedDownName,
                                     Step3p5LayerLimit(swiglu_limits_shared, i),
                                     ffMiddle, ffAct, ffUp, shareOutput);
                if (shareOutput.dataType != hiddenStates.dataType) {
                    Qwen3CudaToDataType(cudaRunner, shareOutput, hiddenStates.dataType);
                }
                if (tensorParallel) {
                    FastllmNcclAllReduce(shareOutput.cudaData, shareOutput.cudaData,
                                         shareOutput.Count(0), shareOutput.dataType, gpuId);
                }
                Qwen3CudaAddTo(cudaRunner, hiddenStates, shareOutput);

                int flatBatch = attenInput.dims[0];
                int flatLen = attenInput.dims[1];
                attenInput.Reshape({flatBatch * flatLen, attenInput.dims[2]});
                Qwen3CudaLinear(cudaRunner, attenInput,
                                *requireLocal(weight[prefix + "moe.gate.weight"], prefix + "moe.gate.weight"),
                                *GetEmptyData(), routerLogits, true);
                Qwen3CudaConvertToDataType(cudaRunner, routerLogits, routerProb, DataType::FLOAT32);
                Step3p5CudaSigmoid(cudaRunner, routerProb, routerProb);
                Data *localGateBias = nullptr;
                if (use_moe_router_bias &&
                    weight.weight.find(prefix + "moe.router_bias") != weight.weight.end()) {
                    localGateBias = requireLocal(weight[prefix + "moe.router_bias"], prefix + "moe.router_bias");
                }
                Qwen3CudaSelectExpert(cudaRunner, routerProb, expertIndex, expertScore,
                                      num_experts_per_tok, norm_topk_prob,
                                      routed_scaling_factor, localGateBias);

                bool ranFusedMoe = false;
                auto fusedIt = fusedMoeByDevice.find(gpuId);
                auto rangeIt = fusedMoeRangesByDevice.find(gpuId);
                if (fusedIt != fusedMoeByDevice.end() &&
                    rangeIt != fusedMoeRangesByDevice.end() &&
                    i < (int)fusedIt->second.size() &&
                    i < (int)rangeIt->second.size()) {
                    auto &localFusedWeights = fusedIt->second[i];
                    std::pair<int, int> expertRange = rangeIt->second[i];
                    if ((int)localFusedWeights.size() == 3 &&
                        localFusedWeights[0] != nullptr &&
                        localFusedWeights[1] != nullptr &&
                        localFusedWeights[2] != nullptr &&
                        Step3p5MaskAndRemapExpertsForLocalRange(
                            expertIndex, expertScore,
                            expertRange.first, expertRange.second,
                            num_experts, gpuId, true)) {
                        Step3p5CudaFusedMOE(cudaRunner, attenInput, expertIndex, expertScore,
                                            *localFusedWeights[0], *localFusedWeights[1],
                                            *localFusedWeights[2], w1, moeFinal, i,
                                            Step3p5LayerLimit(swiglu_limits, i));
                        ranFusedMoe = true;
                    }
                }

                if (!ranFusedMoe) {
                    auto deviceIt = moeWeightsByDevice.find(gpuId);
                    auto biasIt = moeBiassByDevice.find(gpuId);
                    AssertInFastLLM(deviceIt != moeWeightsByDevice.end() &&
                                    biasIt != moeBiassByDevice.end() &&
                                    i < (int)deviceIt->second.size() &&
                                    i < (int)biasIt->second.size(),
                                    "Step3p5 ForwardSingleGPU missing local MoE cache.\n");
                    auto &localWeights = deviceIt->second[i];
                    auto &localBiass = biasIt->second[i];
                    if (Step3p5HasLocalMoeShard(localWeights)) {
                        if (tensorParallel) {
                            Step3p5MaskRemoteExpertsForLocalShard(expertIndex, expertScore,
                                                                  localWeights, num_experts);
                        }
                        Qwen3CudaMergeMOEBlock(cudaRunner, &attenInput, &expertIndex, &expertScore,
                                               &localWeights, &localBiass,
                                               &w1, &w2, &w3, &tempInput, &tempOutput,
                                               1.0f, &moeFinal, i,
                                               computeType, computeType,
                                               &moeInputTemp, &moeOutputTemp);
                    } else {
                        Step3p5ZeroCudaLike(moeFinal, attenInput, gpuId);
                    }
                }
                moeFinal.Reshape(hiddenStates.dims);
                if (moeFinal.dataType != hiddenStates.dataType) {
                    Qwen3CudaToDataType(cudaRunner, moeFinal, hiddenStates.dataType);
                }
                if (tensorParallel) {
                    FastllmNcclAllReduce(moeFinal.cudaData, moeFinal.cudaData,
                                         moeFinal.Count(0), moeFinal.dataType, gpuId);
                }
                Qwen3CudaAddTo(cudaRunner, hiddenStates, moeFinal);
            }
        }

        Data lastHiddenStates;
        Data *headInput = &hiddenStates;
        if (!all1) {
            int total = 0;
            std::vector<Data> lastToks(seqLens.size());
            std::vector<Data*> lastTokPointers;
            lastTokPointers.reserve(seqLens.size());
            for (int b = 0; b < (int)seqLens.size(); b++) {
                Qwen3CudaSplit(cudaRunner, hiddenStates, 1,
                               total + seqLens[b] - 1, total + seqLens[b],
                               lastToks[b]);
                total += seqLens[b];
                lastTokPointers.push_back(&lastToks[b]);
            }
            Qwen3CudaCatBatch(cudaRunner, lastTokPointers, 1, lastHiddenStates);
            headInput = &lastHiddenStates;
        }

        Qwen3CudaRMSNorm(cudaRunner, *headInput,
                         *requireLocal(weight["model.norm.weight"], "model.norm.weight"),
                         rms_norm_eps, *headInput);
        Qwen3CudaLinear(cudaRunner, *headInput,
                        *requireLocal(weight["lm_head.weight"], "lm_head.weight"),
                        *requireLocal(GetThreadTensorParallelBias("lm_head.weight.tp_bias"),
                                      "lm_head.weight.tp_bias"),
                        logits);
        Qwen3CudaToDataType(cudaRunner, logits, DataType::FLOAT32);
#endif
    }

    void Step3p5Model::InitParams() {
        Step3p5FlattenTextConfig(weight.dicts);
        basellm::InitParams();

        base_attention_heads = Step3p5GetInt(weight.dicts, "num_attention_heads", num_attention_heads);
        base_key_value_heads = Step3p5GetInt(weight.dicts, "num_attention_groups", base_key_value_heads);
        sliding_attention_heads = Step3p5GetInt(weight.dicts, "attention_other_setting.num_attention_heads", base_attention_heads);
        sliding_key_value_heads = Step3p5GetInt(weight.dicts, "attention_other_setting.num_attention_groups", base_key_value_heads);
        num_attention_heads = base_attention_heads;
        num_key_value_heads = base_key_value_heads;
        head_dim = Step3p5GetInt(weight.dicts, "head_dim", head_dim);
        embed_dim = Step3p5GetInt(weight.dicts, "hidden_size", embed_dim);
        dense_intermediate_size = Step3p5GetInt(weight.dicts, "intermediate_size", dense_intermediate_size);
        moe_intermediate_size = Step3p5GetInt(weight.dicts, "moe_intermediate_size", moe_intermediate_size);
        shared_expert_intermediate_size = Step3p5GetInt(weight.dicts, "share_expert_dim", shared_expert_intermediate_size);
        num_experts = Step3p5GetInt(weight.dicts, "moe_num_experts", num_experts);
        num_experts_per_tok = Step3p5GetInt(weight.dicts, "moe_top_k", num_experts_per_tok);
        norm_topk_prob = Step3p5IsTrueString(Step3p5GetDict(weight.dicts, "norm_expert_weight", "true"));
        routed_scaling_factor = Step3p5GetFloat(weight.dicts, "moe_router_scaling_factor", 1.0f);
        rms_norm_eps = Step3p5GetFloat(weight.dicts, "rms_norm_eps", rms_norm_eps);
        rope_base = Step3p5GetFloat(weight.dicts, "rope_theta", 10000.0f);
        rope_factor = Step3p5GetFloat(weight.dicts, "rope_scaling.factor", 1.0f);
        sliding_window = Step3p5GetInt(weight.dicts, "sliding_window", sliding_window);
        use_moe_router_bias = Step3p5IsTrueString(Step3p5GetDict(weight.dicts, "use_moe_router_bias", "true"));
        need_fp32_gate = Step3p5IsTrueString(Step3p5GetDict(weight.dicts, "need_fp32_gate", "true"));
        llama3_original_max_position_embeddings = Step3p5GetFloat(weight.dicts, "rope_scaling.original_max_position_embeddings", 131072.0f);
        llama3_low_freq_factor = Step3p5GetFloat(weight.dicts, "rope_scaling.low_freq_factor", 1.0f);
        llama3_high_freq_factor = Step3p5GetFloat(weight.dicts, "rope_scaling.high_freq_factor", 32.0f);
        step3p7VisionAvailable =
            Step3p5GetDict(weight.dicts, "model_type", "") == "step3p7" ||
            Step3p5GetDict(weight.dicts, "vision_config.model_type", "") != "";
        if (step3p7VisionAvailable) {
            step3p7ImageTokenId = Step3p5GetInt(weight.dicts, "image_token_id", step3p7ImageTokenId);
            step3p7ImageTokenLen = Step3p5GetInt(weight.dicts, "image_token_len", step3p7ImageTokenLen);
            step3p7PatchTokenLen = Step3p5GetInt(weight.dicts, "patch_token_len", step3p7PatchTokenLen);
            step3p7VisionWidth = Step3p5GetInt(weight.dicts, "vision_config.width", step3p7VisionWidth);
            step3p7VisionLayers = Step3p5GetInt(weight.dicts, "vision_config.layers", step3p7VisionLayers);
            step3p7VisionHeads = Step3p5GetInt(weight.dicts, "vision_config.heads", step3p7VisionHeads);
            step3p7VisionImageSize = Step3p5GetInt(weight.dicts, "vision_config.image_size", step3p7VisionImageSize);
            step3p7VisionPatchSize = Step3p5GetInt(weight.dicts, "vision_config.patch_size", step3p7VisionPatchSize);
            step3p7VisionLayerNormEps = Step3p5GetFloat(weight.dicts, "vision_config.layer_norm_eps", step3p7VisionLayerNormEps);
            step3p7VisionRopeTheta = Step3p5GetFloat(weight.dicts, "vision_config.rope_theta", step3p7VisionRopeTheta);
            step3p7UseLnPre = Step3p5IsTrueString(Step3p5GetDict(weight.dicts, "vision_config.use_ln_pre", "true"));
            step3p7UseLnPost = Step3p5IsTrueString(Step3p5GetDict(weight.dicts, "vision_config.use_ln_post", "false"));
            step3p7UseAbsPosEmb = Step3p5IsTrueString(Step3p5GetDict(weight.dicts, "vision_config.use_abs_posemb", "true"));
            step3p7UseRope2d = Step3p5IsTrueString(Step3p5GetDict(weight.dicts, "vision_config.use_rope2d", "true"));
            float mlpRatio = Step3p5GetFloat(weight.dicts, "vision_config.mlp_ratio", 8960.0f / 1536.0f);
            step3p7VisionMlpHidden = (int)(step3p7VisionWidth * mlpRatio + 1e-4f);
            step3p7VisionHeadDim = step3p7VisionWidth / std::max(1, step3p7VisionHeads);
            step3p7VisionBaseGrid = step3p7VisionImageSize / std::max(1, step3p7VisionPatchSize);
            this->is_multi_modal = true;
        }

        layer_types = Step3p5ParseStringList(Step3p5GetDict(weight.dicts, "layer_types", ""));
        if ((int)layer_types.size() < block_cnt) {
            layer_types.resize(block_cnt);
            for (int i = 0; i < block_cnt; i++) {
                if (layer_types[i].empty()) {
                    layer_types[i] = (i % 4 == 0) ? "full_attention" : "sliding_attention";
                }
            }
        }

        moe_layers = Step3p5ParseIntSet(Step3p5GetDict(weight.dicts, "moe_layers_enum", ""));
        if (moe_layers.empty()) {
            for (int i = 1; i < block_cnt; i++) {
                moe_layers.insert(i);
            }
        }

        layer_rope_thetas = Step3p5ParseFloatList(Step3p5GetDict(weight.dicts, "rope_theta", ""));
        if ((int)layer_rope_thetas.size() < block_cnt) {
            layer_rope_thetas.resize(block_cnt, rope_base);
        }

        std::vector<float> partialRotary = Step3p5ParseFloatList(Step3p5GetDict(weight.dicts, "partial_rotary_factors", ""));
        layer_rotary_dims.resize(block_cnt, head_dim);
        for (int i = 0; i < block_cnt; i++) {
            float factor = (i < (int)partialRotary.size()) ? partialRotary[i] : 1.0f;
            layer_rotary_dims[i] = std::max(0, (int)(head_dim * factor + 1e-5f));
        }
        rotary_dim = layer_rotary_dims.empty() ? head_dim : layer_rotary_dims[0];

        swiglu_limits = Step3p5ParseFloatList(Step3p5GetDict(weight.dicts, "swiglu_limits", ""));
        swiglu_limits_shared = Step3p5ParseFloatList(Step3p5GetDict(weight.dicts, "swiglu_limits_shared", ""));

        for (int i = 0; i < block_cnt; i++) {
            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string qBiasName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.bias";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string kBiasName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.bias";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string vBiasName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.bias";
            std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
            this->weightMergeRules.push_back(
                WeightMergeRule({WeightMergeRuleSingle({qWeightName, kWeightName, vWeightName}, mergeQkvWeightName, std::string("linear")),
                                 WeightMergeRuleSingle({qBiasName, kBiasName, vBiasName}, mergeQkvBiasName, std::string("bias"))})
            );

            std::string denseGateName = "model.layers." + std::to_string(i) + ".mlp.gate_proj.weight";
            std::string denseUpName = "model.layers." + std::to_string(i) + ".mlp.up_proj.weight";
            std::string denseGateupName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
            if (Step3p5LayerLimit(swiglu_limits_shared, i) == 0.0f) {
                this->weightMergeRules.push_back(
                    WeightMergeRule({WeightMergeRuleSingle({denseGateName, denseUpName}, denseGateupName, std::string("linearSwiglu"))})
                );
            }

            std::string sharedGateName = "model.layers." + std::to_string(i) + ".share_expert.gate_proj.weight";
            std::string sharedUpName = "model.layers." + std::to_string(i) + ".share_expert.up_proj.weight";
            std::string sharedGateupName = "model.layers." + std::to_string(i) + ".share_expert.gateup_proj.weight";
            if (Step3p5LayerLimit(swiglu_limits_shared, i) == 0.0f) {
                this->weightMergeRules.push_back(
                    WeightMergeRule({WeightMergeRuleSingle({sharedGateName, sharedUpName}, sharedGateupName, std::string("linearSwiglu"))})
                );
            }

            if (IsMoeLayer(i)) {
                std::string moePrefix = "model.layers." + std::to_string(i) + ".moe.";
                this->moeLinears.insert(moePrefix + "gate_proj.weight");
                this->moeLinears.insert(moePrefix + "up_proj.weight");
                this->moeLinears.insert(moePrefix + "down_proj.weight");
            }
        }

        moeGateWeights.clear();
        moeUpWeights.clear();
        moeDownWeights.clear();
        weights.clear();
        biass.clear();
        moeWeightsPrepared = false;
        initialized_add1 = false;
        step3p7VisionPrepared = false;
    }

    int Step3p5Model::LayerAttentionHeads(int layer) const {
        return IsFullAttentionLayer(layer) ? base_attention_heads : sliding_attention_heads;
    }

    int Step3p5Model::LayerKeyValueHeads(int layer) const {
        return IsFullAttentionLayer(layer) ? base_key_value_heads : sliding_key_value_heads;
    }

    bool Step3p5Model::IsFullAttentionLayer(int layer) const {
        return layer >= 0 && layer < (int)layer_types.size() && layer_types[layer] == "full_attention";
    }

    bool Step3p5Model::IsMoeLayer(int layer) const {
        return moe_layers.find(layer) != moe_layers.end();
    }

    bool Step3p5Model::UseLlama3Rope(int layer) const {
        return IsFullAttentionLayer(layer) &&
               Step3p5GetDict(weight.dicts, "rope_scaling.rope_type", "") == "llama3";
    }

    void Step3p5Model::PrepareMoeWeights() {
        if (moeWeightsPrepared) {
            return;
        }
        moeGateWeights.resize(block_cnt);
        moeUpWeights.resize(block_cnt);
        moeDownWeights.resize(block_cnt);
        moeGate3DWeights.assign(block_cnt, nullptr);
        moeUp3DWeights.assign(block_cnt, nullptr);
        moeDown3DWeights.assign(block_cnt, nullptr);
        weights.clear();
        biass.clear();
        weights.resize(block_cnt);
        biass.resize(block_cnt);
        bool disableFusedMoe = Step3p5DisableFusedMoe();
        for (int i = 0; i < block_cnt; i++) {
            if (!IsMoeLayer(i)) {
                continue;
            }
            std::string prefix = "model.layers." + std::to_string(i) + ".moe.";
            std::string gateSourceName = prefix + "gate_proj.weight";
            std::string upSourceName = prefix + "up_proj.weight";
            std::string downSourceName = prefix + "down_proj.weight";
            if (weight.weight.find(gateSourceName) == weight.weight.end() ||
                weight.weight.find(upSourceName) == weight.weight.end() ||
                weight.weight.find(downSourceName) == weight.weight.end()) {
                continue;
            }
            Data &gateSource = weight[gateSourceName];
            Data &upSource = weight[upSourceName];
            Data &downSource = weight[downSourceName];
            bool useDiskMergedMoe = gateSource.isDiskWeight || upSource.isDiskWeight || downSource.isDiskWeight;
            std::string selectedMoeDevice = this->SelectMoeDeviceForLayer(i);
            bool selectedCudaMoe = selectedMoeDevice.rfind("cuda", 0) == 0 ||
                                   selectedMoeDevice.rfind("multicuda", 0) == 0 ||
                                   (selectedMoeDevice.empty() &&
                                    Step3p5DeviceMapUsesCuda(this->deviceMap));
            float expertLimit = Step3p5LayerLimit(swiglu_limits, i);
            if (!disableFusedMoe && selectedCudaMoe && !useDiskMergedMoe) {
                AssertInFastLLM(gateSource.dims.size() == 3 && upSource.dims.size() == 3 &&
                                downSource.dims.size() == 3,
                                "Step3p5 FusedMOE source weights should be 3D.");
                AssertInFastLLM(gateSource.dims[0] == num_experts &&
                                upSource.dims[0] == num_experts &&
                                downSource.dims[0] == num_experts &&
                                gateSource.dims[1] == upSource.dims[1] &&
                                gateSource.dims[2] == upSource.dims[2] &&
                                downSource.dims[0] == gateSource.dims[0] &&
                                downSource.dims[1] == gateSource.dims[2] &&
                                downSource.dims[2] == gateSource.dims[1],
                                "Step3p5 FusedMOE source weight shapes mismatch.");
                bool isFp8 = gateSource.dataType == DataType::FP8_E4M3 &&
                              upSource.dataType == DataType::FP8_E4M3 &&
                              downSource.dataType == DataType::FP8_E4M3;
                bool isFp8Block128 = gateSource.dataType == DataType::FP8_E4M3_BLOCK_128 &&
                                      upSource.dataType == DataType::FP8_E4M3_BLOCK_128 &&
                                      downSource.dataType == DataType::FP8_E4M3_BLOCK_128;
                AssertInFastLLM(isFp8 || isFp8Block128,
                                "Step3p5 FusedMOE only supports 3D FP8_E4M3 or FP8_E4M3_BLOCK_128 weights.");
                if (isFp8) {
                    AssertInFastLLM(gateSource.blockM > 0 && gateSource.blockK > 0 &&
                                    upSource.blockM == gateSource.blockM && upSource.blockK == gateSource.blockK &&
                                    downSource.blockM > 0 && downSource.blockK > 0 &&
                                    !gateSource.scales.empty() && !upSource.scales.empty() && !downSource.scales.empty(),
                                    "Step3p5 FusedMOE FP8 weights should have block scales.");
                }
                moeGate3DWeights[i] = &gateSource;
                moeUp3DWeights[i] = &upSource;
                moeDown3DWeights[i] = &downSource;
                continue;
            }
            moeGateWeights[i].resize(num_experts);
            moeUpWeights[i].resize(num_experts);
            moeDownWeights[i].resize(num_experts);
            weights[i].push_back(nullptr);
            weights[i].push_back(nullptr);
            biass[i].push_back(nullptr);
            biass[i].push_back(nullptr);
            for (int j = 0; j < num_experts; j++) {
                std::string expertPrefix = prefix + "experts." + std::to_string(j) + ".";
                std::string gateName = expertPrefix + "gate_proj.weight";
                std::string upName = expertPrefix + "up_proj.weight";
                std::string downName = expertPrefix + "down_proj.weight";
                std::string gateupName = expertPrefix + "gateup_proj.weight";
                if (useDiskMergedMoe) {
                    if (weight.weight.find(gateName) == weight.weight.end()) {
                        Step3p5MakeExpertView(weight.weight[gateName], gateSource, gateName, j);
                    }
                    if (weight.weight.find(upName) == weight.weight.end()) {
                        Step3p5MakeExpertView(weight.weight[upName], upSource, upName, j);
                    }
                    if (weight.weight.find(downName) == weight.weight.end()) {
                        Step3p5MakeExpertView(weight.weight[downName], downSource, downName, j);
                    }
                    moeGateWeights[i][j] = &weight[gateName];
                    moeUpWeights[i][j] = &weight[upName];
                    moeDownWeights[i][j] = &weight[downName];
                    if (weight.weight.find(gateupName) == weight.weight.end()) {
                        Step3p5MakeGateUpWeight(weight.weight[gateupName], weight[gateName], weight[upName], gateupName);
                    }
                    weights[i].push_back(&weight[gateupName]);
                    weights[i].push_back(&weight[downName]);
                    biass[i].push_back(nullptr);
                    biass[i].push_back(nullptr);
                } else {
                    if (weight.weight.find(gateupName) == weight.weight.end()) {
                        Data gateCopy, upCopy;
                        Step3p5MakeExpertCopy(gateCopy, gateSource, gateName, j);
                        Step3p5MakeExpertCopy(upCopy, upSource, upName, j);
                        Step3p5MakeGateUpWeight(weight.weight[gateupName], gateCopy, upCopy, gateupName);
                    }
                    if (weight.weight.find(downName) == weight.weight.end()) {
                        Step3p5MakeExpertCopy(weight.weight[downName], downSource, downName, j);
                    }
                    Data &gateup = weight[gateupName];
                    Data &down = weight[downName];
                    gateup.tpLinearType = TP_LINEAR_ROW;
                    gateup.tpPackType = TP_PACK_GATEUP;
                    down.tpLinearType = TP_LINEAR_COLUMN;
                    if (weight.weight.find(gateName) == weight.weight.end()) {
                        Step3p5MakeGateUpSliceView(weight.weight[gateName], gateup, gateName, 0, gateup.dims[0] / 2);
                    }
                    if (weight.weight.find(upName) == weight.weight.end()) {
                        Step3p5MakeGateUpSliceView(weight.weight[upName], gateup, upName, gateup.dims[0] / 2, gateup.dims[0] / 2);
                    }
                    moeGateWeights[i][j] = &weight[gateName];
                    moeUpWeights[i][j] = &weight[upName];
                    moeDownWeights[i][j] = &weight[downName];
                    weights[i].push_back(&gateup);
                    weights[i].push_back(&down);
                    biass[i].push_back(nullptr);
                    biass[i].push_back(nullptr);
                }
            }
            if (!useDiskMergedMoe) {
                weight.weight.erase(gateSourceName);
                weight.weight.erase(upSourceName);
                weight.weight.erase(downSourceName);
            }
        }
        moeWeightsPrepared = true;
    }

    void Step3p5Model::ApplyStepRotary(Data &input, const Data &positionIds, int layer) {
        int curRotaryDim = layer < (int)layer_rotary_dims.size() ? layer_rotary_dims[layer] : head_dim;
        float theta = layer < (int)layer_rope_thetas.size() ? layer_rope_thetas[layer] : rope_base;
        if (UseLlama3Rope(layer)) {
            fastllm::Llama3RopeEncoding(input, positionIds, curRotaryDim, theta, rope_factor,
                                         llama3_original_max_position_embeddings,
                                         llama3_low_freq_factor, llama3_high_freq_factor);
            return;
        }
        float ropeScale = 1.0f;
        fastllm::RopeEncoding(input, positionIds, curRotaryDim, theta, ropeScale);
    }

    void Step3p5Model::PrepareStep3p7Vision() {
        if (step3p7VisionPrepared) {
            return;
        }
        AssertInFastLLM(step3p7VisionAvailable, "Step3.7 vision is not enabled for this model.\n");
        AssertInFastLLM(weight.weight.find("vision_model.conv1.weight") != weight.weight.end() &&
                        weight.weight.find("vision_model.positional_embedding") != weight.weight.end() &&
                        weight.weight.find("vit_large_projector.weight") != weight.weight.end(),
                        "Step3.7 vision weights are incomplete.\n");

        int visionDevice = -1;
#ifdef USE_CUDA
        visionDevice = Step3p7ResolveVisionDevice(this->deviceMap);
        if (getenv("FASTLLM_STEP3P7_DEBUG") != nullptr) {
            printf("[Step3.7 debug] prepare vision device=%d\n", visionDevice);
            fflush(stdout);
        }
#endif

        auto prepareWeight = [&](const std::string &name, DataType type) {
            auto it = weight.weight.find(name);
            if (it == weight.weight.end()) {
                return;
            }
            Data &data = it->second;
            if (data.dataType != type) {
                if (data.dataDevice != DataDevice::CPU) {
                    data.ToDevice(DataDevice::CPU);
                }
                ToDataType(data, type);
            }
#ifdef USE_CUDA
            if (visionDevice >= 0) {
                data.ToDevice(DataDevice::CUDA, {visionDevice}, true);
            }
#endif
        };

        prepareWeight("vision_model.conv1.weight", DataType::FLOAT16);
        prepareWeight("vision_model.positional_embedding", DataType::FLOAT32);
        prepareWeight("vision_model.ln_pre.weight", DataType::FLOAT32);
        prepareWeight("vision_model.ln_pre.bias", DataType::FLOAT32);
        prepareWeight("vision_model.ln_post.weight", DataType::FLOAT32);
        prepareWeight("vision_model.ln_post.bias", DataType::FLOAT32);
        prepareWeight("vision_model.vit_downsampler1.weight", DataType::FLOAT16);
        prepareWeight("vision_model.vit_downsampler1.bias", DataType::FLOAT32);
        prepareWeight("vision_model.vit_downsampler2.weight", DataType::FLOAT16);
        prepareWeight("vision_model.vit_downsampler2.bias", DataType::FLOAT32);
        prepareWeight("vit_large_projector.weight", DataType::FLOAT16);
        prepareWeight("vit_large_projector.bias", DataType::FLOAT32);

        for (int i = 0; i < step3p7VisionLayers; i++) {
            std::string pre = "vision_model.transformer.resblocks." + std::to_string(i);
            prepareWeight(pre + ".ln_1.weight", DataType::FLOAT32);
            prepareWeight(pre + ".ln_1.bias", DataType::FLOAT32);
            prepareWeight(pre + ".ln_2.weight", DataType::FLOAT32);
            prepareWeight(pre + ".ln_2.bias", DataType::FLOAT32);
            prepareWeight(pre + ".attn.in_proj_weight", DataType::FLOAT16);
            prepareWeight(pre + ".attn.in_proj_bias", DataType::FLOAT32);
            prepareWeight(pre + ".attn.out_proj.weight", DataType::FLOAT16);
            prepareWeight(pre + ".attn.out_proj.bias", DataType::FLOAT32);
            prepareWeight(pre + ".mlp.c_fc.weight", DataType::FLOAT16);
            prepareWeight(pre + ".mlp.c_fc.bias", DataType::FLOAT32);
            prepareWeight(pre + ".mlp.c_proj.weight", DataType::FLOAT16);
            prepareWeight(pre + ".mlp.c_proj.bias", DataType::FLOAT32);
            prepareWeight(pre + ".ls_1.gamma", DataType::FLOAT32);
            prepareWeight(pre + ".ls_2.gamma", DataType::FLOAT32);
        }

        int ropeDim = step3p7VisionHeadDim / 4;
        std::vector<float> sinValues(step3p7VisionBaseGrid * ropeDim);
        std::vector<float> cosValues(step3p7VisionBaseGrid * ropeDim);
        int invBaseDim = step3p7VisionHeadDim / 2;
        for (int p = 0; p < step3p7VisionBaseGrid; p++) {
            for (int j = 0; j < ropeDim; j++) {
                float invFreq = 1.0f / powf(step3p7VisionRopeTheta, (float)(2 * j) / (float)invBaseDim);
                float v = (float)p * invFreq;
                sinValues[p * ropeDim + j] = sinf(v);
                cosValues[p * ropeDim + j] = cosf(v);
            }
        }
        step3p7VisionSinData.CopyFrom(Data(DataType::FLOAT32, {step3p7VisionBaseGrid, ropeDim}, sinValues));
        step3p7VisionCosData.CopyFrom(Data(DataType::FLOAT32, {step3p7VisionBaseGrid, ropeDim}, cosValues));
        step3p7VisionConv1Bias.CopyFrom(Data(DataType::FLOAT32, {step3p7VisionWidth},
                                             std::vector<float>(step3p7VisionWidth, 0.0f)));
#ifdef USE_CUDA
        if (visionDevice >= 0) {
            step3p7VisionSinData.ToDevice(DataDevice::CUDA, {visionDevice}, true);
            step3p7VisionCosData.ToDevice(DataDevice::CUDA, {visionDevice}, true);
            step3p7VisionConv1Bias.ToDevice(DataDevice::CUDA, {visionDevice}, true);
        }
#endif
        step3p7VisionPrepared = true;
#ifdef USE_CUDA
        Step3p7DebugCudaMemory("vision prepared weights");
#endif
    }

    void Step3p5Model::ReleaseStep3p7VisionCuda() {
#ifdef USE_CUDA
        if (!step3p7VisionPrepared) {
            return;
        }
        auto releaseTensor = [](Data &data) {
            if (data.dataDevice == DataDevice::CUDA) {
                std::vector<int> ids = data.dataDeviceIds.empty() ?
                                       std::vector<int>{FastllmCudaGetDevice()} : data.dataDeviceIds;
                data.ToDevice(DataDevice::CPU, ids, true);
            }
        };

        for (auto &it : weight.weight) {
            DataType type = DataType::FLOAT32;
            if (Step3p7VisionTensorType(it.first, type)) {
                releaseTensor(it.second);
            }
        }
        releaseTensor(step3p7VisionSinData);
        releaseTensor(step3p7VisionCosData);
        releaseTensor(step3p7VisionConv1Bias);
        step3p7VisionPrepared = false;
        FastllmCudaClearBigBuffer();
        Step3p7DebugCudaMemory("vision released weights");
#endif
    }

    void Step3p5Model::BuildStep3p7VisionPositionData(int gridH, int gridW,
                                                       Data &posEmb, Data &posH, Data &posW) {
        PrepareStep3p7Vision();
        Data posWeight(weight["vision_model.positional_embedding"]);
        posWeight.ToDevice(DataDevice::CPU);
        if (posWeight.dataType != DataType::FLOAT32) {
            ToDataType(posWeight, DataType::FLOAT32);
            posWeight.ToDevice(DataDevice::CPU);
        }
        AssertInFastLLM(posWeight.dims.size() == 2 && posWeight.dims[1] == step3p7VisionWidth,
                        "Step3.7 vision positional embedding shape is invalid.\n");

        int base = step3p7VisionBaseGrid;
        int hidden = step3p7VisionWidth;
        float *src = (float*)posWeight.cpuData;
        std::vector<float> emb((size_t)gridH * gridW * hidden);
        if (gridH == base && gridW == base) {
            memcpy(emb.data(), src, emb.size() * sizeof(float));
        } else {
            auto sourceIndex = [&](int y, int x, int c) {
                return ((y * base + x) * hidden + c);
            };
            for (int y = 0; y < gridH; y++) {
                float inY = ((float)y + 0.5f) * (float)base / (float)gridH - 0.5f;
                int y0 = (int)floorf(inY);
                int y1 = y0 + 1;
                float wy = inY - (float)y0;
                if (y0 < 0) {
                    y0 = 0;
                    wy = 0.0f;
                }
                if (y1 >= base) {
                    y1 = base - 1;
                }
                for (int x = 0; x < gridW; x++) {
                    float inX = ((float)x + 0.5f) * (float)base / (float)gridW - 0.5f;
                    int x0 = (int)floorf(inX);
                    int x1 = x0 + 1;
                    float wx = inX - (float)x0;
                    if (x0 < 0) {
                        x0 = 0;
                        wx = 0.0f;
                    }
                    if (x1 >= base) {
                        x1 = base - 1;
                    }
                    float w00 = (1.0f - wy) * (1.0f - wx);
                    float w01 = (1.0f - wy) * wx;
                    float w10 = wy * (1.0f - wx);
                    float w11 = wy * wx;
                    float *dst = emb.data() + ((y * gridW + x) * hidden);
                    for (int c = 0; c < hidden; c++) {
                        dst[c] = src[sourceIndex(y0, x0, c)] * w00 +
                                 src[sourceIndex(y0, x1, c)] * w01 +
                                 src[sourceIndex(y1, x0, c)] * w10 +
                                 src[sourceIndex(y1, x1, c)] * w11;
                    }
                }
            }
        }

        std::vector<float> hPos(gridH * gridW), wPos(gridH * gridW);
        for (int y = 0; y < gridH; y++) {
            for (int x = 0; x < gridW; x++) {
                hPos[y * gridW + x] = (float)y;
                wPos[y * gridW + x] = (float)x;
            }
        }
        posEmb.CopyFrom(Data(DataType::FLOAT32, {1, gridH * gridW, hidden}, emb));
        posH.CopyFrom(Data(DataType::FLOAT32, {1, gridH * gridW}, hPos));
        posW.CopyFrom(Data(DataType::FLOAT32, {1, gridH * gridW}, wPos));
        if (step3p7VisionSinData.dataDevice != DataDevice::CPU) {
            posEmb.ToDevice(step3p7VisionSinData.dataDevice, step3p7VisionSinData.dataDeviceIds);
            posH.ToDevice(step3p7VisionSinData.dataDevice, step3p7VisionSinData.dataDeviceIds);
            posW.ToDevice(step3p7VisionSinData.dataDevice, step3p7VisionSinData.dataDeviceIds);
        }
    }

    void Step3p5Model::ApplyStep3p7VisionRotary(Data &input, const Data &posH, const Data &posW) {
        if (!step3p7UseRope2d) {
            return;
        }
        AssertInFastLLM(input.dims.size() == 4 && input.dims.back() == step3p7VisionHeadDim,
                        "Step3.7 vision rotary expects [batch, seq, heads, head_dim].\n");
        int half = step3p7VisionHeadDim / 2;
        int ropeDim = half / 2;
        Data wPart, hPart, rotated;
        Split(input, -1, 0, half, wPart);
        Split(input, -1, half, step3p7VisionHeadDim, hPart);

        auto rotatePart = [&](Data &part, const Data &pos) {
            std::vector<int> dims = part.dims;
            part.Reshape({dims[0], dims[1], dims[2], ropeDim, 2});
            PermuteSelf(part, {0, 1, 2, 4, 3});
            part.Reshape({dims[0], dims[1], dims[2], half});
            LlamaRotatePosition2DPart(part, pos, step3p7VisionSinData, step3p7VisionCosData, ropeDim, half);
            part.Reshape({dims[0], dims[1], dims[2], 2, ropeDim});
            PermuteSelf(part, {0, 1, 2, 4, 3});
            part.Reshape(dims);
        };
        rotatePart(wPart, posW);
        rotatePart(hPart, posH);
        Cat(wPart, hPart, -1, rotated);
        input.CopyFrom(rotated);
    }

    void Step3p5Model::ApplyStep3p7LayerScale(Data &input, Data &gamma) {
        if (gamma.dims.empty()) {
            return;
        }
        std::vector<int> originalDims = input.dims;
        int channels = originalDims.back();
        int outer = input.Count(0) / channels;
        Data scale(gamma);
        if (scale.dataType != input.dataType) {
            ToDataType(scale, input.dataType);
        }
        if (input.dataDevice != DataDevice::CPU) {
            scale.ToDevice(input.dataDevice, input.dataDeviceIds);
        }
        input.Reshape({outer, channels});
        PermuteSelf(input, {1, 0});
        MulTo(input, scale);
        input.Reshape({channels, outer});
        PermuteSelf(input, {1, 0});
        input.Reshape(originalDims);
    }

    void Step3p5Model::Step3p7QuickGelu(Data &input) {
        Data gate;
        Mul(input, 1.702f, gate);
        Sigmoid(gate, gate);
        MulTo(input, gate);
    }

    void Step3p5Model::ProcessStep3p7ImageFeatures(const Data &hiddenStates, int grid, Data &features) {
        Data imageFeatures(hiddenStates);
        if (imageFeatures.dataType != DataType::FLOAT32) {
            ToDataType(imageFeatures, DataType::FLOAT32);
        }
        imageFeatures.Reshape({1, grid * grid, step3p7VisionWidth});
        PermuteSelf(imageFeatures, {0, 2, 1});
        imageFeatures.Reshape({1, step3p7VisionWidth, grid, grid});
        if (weight["vision_model.vit_downsampler1.weight"].dataDevice != DataDevice::CPU) {
            imageFeatures.ToDevice(weight["vision_model.vit_downsampler1.weight"].dataDevice,
                                   weight["vision_model.vit_downsampler1.weight"].dataDeviceIds);
        }
        Data down1, down2;
        Conv2D(imageFeatures,
               weight["vision_model.vit_downsampler1.weight"],
               weight["vision_model.vit_downsampler1.bias"],
               step3p7VisionWidth, step3p7VisionWidth * 2,
               3, 3, 2, 2, 1, 1, down1);
        Conv2D(down1,
               weight["vision_model.vit_downsampler2.weight"],
               weight["vision_model.vit_downsampler2.bias"],
               step3p7VisionWidth * 2, step3p7VisionWidth * 4,
               3, 3, 2, 2, 1, 1, down2);
        int outGrid = down2.dims[2];
        down2.Reshape({1, step3p7VisionWidth * 4, outGrid * outGrid});
        PermuteSelf(down2, {0, 2, 1});
        Linear(down2, weight["vit_large_projector.weight"], *GetEmptyData(), features);
    }

    void Step3p5Model::EncodeStep3p7PixelValues(const Data &pixelValues, Data &features) {
        PrepareStep3p7Vision();
        features = Data();
        if (pixelValues.dims.empty() || pixelValues.dims[0] == 0) {
            return;
        }
        AssertInFastLLM(pixelValues.dims.size() == 4 && pixelValues.dims[1] == 3,
                        "Step3.7 pixel_values should have shape [count, 3, H, W].\n");

        int count = pixelValues.dims[0];
        int imageH = pixelValues.dims[2];
        int imageW = pixelValues.dims[3];
        AssertInFastLLM(imageH == imageW && imageH % step3p7VisionPatchSize == 0,
                        "Step3.7 vision input size is invalid.\n");
        int grid = imageH / step3p7VisionPatchSize;
        int outputTokens = (grid / 4) * (grid / 4);
        std::vector<float> merged;
        merged.reserve((size_t)count * outputTokens * embed_dim);

        Data posEmb, posH, posW;
        BuildStep3p7VisionPositionData(grid, grid, posEmb, posH, posW);
        float attnScale = powf((float)step3p7VisionHeadDim, -0.5f);

        for (int item = 0; item < count; item++) {
            Data pixelInput;
            Split(pixelValues, 0, item, item + 1, pixelInput);
            if (pixelInput.dataType != DataType::FLOAT32) {
                ToDataType(pixelInput, DataType::FLOAT32);
            }
            if (weight["vision_model.conv1.weight"].dataDevice != DataDevice::CPU) {
                pixelInput.ToDevice(weight["vision_model.conv1.weight"].dataDevice,
                                    weight["vision_model.conv1.weight"].dataDeviceIds);
            }

            Data hiddenStates;
            Conv2D(pixelInput, weight["vision_model.conv1.weight"], step3p7VisionConv1Bias,
                   3, step3p7VisionWidth,
                   step3p7VisionPatchSize, step3p7VisionPatchSize,
                   step3p7VisionPatchSize, step3p7VisionPatchSize,
                   0, 0, hiddenStates);
            hiddenStates.Reshape({1, step3p7VisionWidth, grid * grid});
            PermuteSelf(hiddenStates, {0, 2, 1});

            Data localPosEmb(posEmb);
            if (hiddenStates.dataDevice != DataDevice::CPU) {
                localPosEmb.ToDevice(hiddenStates.dataDevice, hiddenStates.dataDeviceIds);
            }
            AddTo(hiddenStates, localPosEmb);
            if (step3p7UseLnPre) {
                LayerNorm(hiddenStates, weight["vision_model.ln_pre.weight"], weight["vision_model.ln_pre.bias"],
                          -1, hiddenStates);
            }

            Data blockInput, qkv, q, k, v, attnOutput, residual, mlpHidden, mlpOutput;
            for (int layer = 0; layer < step3p7VisionLayers; layer++) {
                std::string pre = "vision_model.transformer.resblocks." + std::to_string(layer);
                Mul(hiddenStates, 1.0f, residual);
                LayerNorm(hiddenStates, weight[pre + ".ln_1.weight"], weight[pre + ".ln_1.bias"],
                          -1, blockInput);
                Linear(blockInput, weight[pre + ".attn.in_proj_weight"], weight[pre + ".attn.in_proj_bias"], qkv);
                Split(qkv, -1, 0, step3p7VisionWidth, q);
                Split(qkv, -1, step3p7VisionWidth, step3p7VisionWidth * 2, k);
                Split(qkv, -1, step3p7VisionWidth * 2, step3p7VisionWidth * 3, v);
                q.Reshape({1, grid * grid, step3p7VisionHeads, step3p7VisionHeadDim});
                k.Reshape({1, grid * grid, step3p7VisionHeads, step3p7VisionHeadDim});
                v.Reshape({1, grid * grid, step3p7VisionHeads, step3p7VisionHeadDim});
                ApplyStep3p7VisionRotary(q, posH, posW);
                ApplyStep3p7VisionRotary(k, posH, posW);
                PermuteSelf(q, {0, 2, 1, 3});
                PermuteSelf(k, {0, 2, 1, 3});
                PermuteSelf(v, {0, 2, 1, 3});
                q.Reshape({step3p7VisionHeads, grid * grid, step3p7VisionHeadDim});
                k.Reshape({step3p7VisionHeads, grid * grid, step3p7VisionHeadDim});
                v.Reshape({step3p7VisionHeads, grid * grid, step3p7VisionHeadDim});
                Attention(q, k, v, *GetEmptyData(), attnOutput, 1, attnScale, 2);
                attnOutput.Reshape({1, step3p7VisionHeads, grid * grid, step3p7VisionHeadDim});
                PermuteSelf(attnOutput, {0, 2, 1, 3});
                attnOutput.Reshape({1, grid * grid, step3p7VisionWidth});
                Linear(attnOutput, weight[pre + ".attn.out_proj.weight"], weight[pre + ".attn.out_proj.bias"], attnOutput);
                ApplyStep3p7LayerScale(attnOutput, weight[pre + ".ls_1.gamma"]);
                AddTo(residual, attnOutput);
                hiddenStates.CopyFrom(residual);

                Mul(hiddenStates, 1.0f, residual);
                LayerNorm(hiddenStates, weight[pre + ".ln_2.weight"], weight[pre + ".ln_2.bias"],
                          -1, blockInput);
                Linear(blockInput, weight[pre + ".mlp.c_fc.weight"], weight[pre + ".mlp.c_fc.bias"], mlpHidden);
                Step3p7QuickGelu(mlpHidden);
                Linear(mlpHidden, weight[pre + ".mlp.c_proj.weight"], weight[pre + ".mlp.c_proj.bias"], mlpOutput);
                ApplyStep3p7LayerScale(mlpOutput, weight[pre + ".ls_2.gamma"]);
                AddTo(residual, mlpOutput);
                hiddenStates.CopyFrom(residual);
            }

            if (step3p7UseLnPost) {
                LayerNorm(hiddenStates, weight["vision_model.ln_post.weight"], weight["vision_model.ln_post.bias"],
                          -1, hiddenStates);
            }

            Data projected;
            ProcessStep3p7ImageFeatures(hiddenStates, grid, projected);
            projected.ToDevice(DataDevice::CPU);
            if (projected.dataType != DataType::FLOAT32) {
                ToDataType(projected, DataType::FLOAT32);
                projected.ToDevice(DataDevice::CPU);
            }
            AssertInFastLLM(projected.dims.size() == 3 && projected.dims[1] == outputTokens &&
                            projected.dims[2] == embed_dim,
                            "Step3.7 projected image feature shape mismatch.\n");
            float *ptr = (float*)projected.cpuData;
            merged.insert(merged.end(), ptr, ptr + projected.Count(0));
        }
        features.CopyFrom(Data(DataType::FLOAT32, {count, outputTokens, embed_dim}, merged));
    }

    void Step3p5Model::EncodeStep3p7Images(
            const std::map <std::string, std::vector <Data*> > &multimodalInput,
            Data &features) {
        auto embedIt = multimodalInput.find("image_embeds");
        if (embedIt != multimodalInput.end() && !embedIt->second.empty() && embedIt->second[0] != nullptr) {
            features.CopyFrom(*embedIt->second[0]);
            return;
        }

        auto pixelIt = multimodalInput.find("pixel_values");
        auto numIt = multimodalInput.find("num_patches");
        AssertInFastLLM(pixelIt != multimodalInput.end() && !pixelIt->second.empty() && pixelIt->second[0] != nullptr,
                        "Step3.7 multimodal requires pixel_values.\n");
        AssertInFastLLM(numIt != multimodalInput.end() && !numIt->second.empty() && numIt->second[0] != nullptr,
                        "Step3.7 multimodal requires num_patches.\n");

        Data numPatches(*numIt->second[0]);
        numPatches.ToDevice(DataDevice::CPU);
        AssertInFastLLM(numPatches.dims.size() == 1,
                        "Step3.7 num_patches should have shape [image_count].\n");
        int imageCount = numPatches.dims[0];
        std::vector<int> patchCounts(imageCount);
        int totalPatchCount = 0;
        if (numPatches.dataType == DataType::INT32) {
            int *ptr = (int*)numPatches.cpuData;
            for (int i = 0; i < imageCount; i++) {
                patchCounts[i] = ptr[i];
                totalPatchCount += ptr[i];
            }
        } else {
            if (numPatches.dataType != DataType::FLOAT32) {
                ToDataType(numPatches, DataType::FLOAT32);
                numPatches.ToDevice(DataDevice::CPU);
            }
            float *ptr = (float*)numPatches.cpuData;
            for (int i = 0; i < imageCount; i++) {
                patchCounts[i] = (int)ptr[i];
                totalPatchCount += patchCounts[i];
            }
        }

        Data imageFeatures, patchFeatures;
        EncodeStep3p7PixelValues(*pixelIt->second[0], imageFeatures);
        AssertInFastLLM(imageFeatures.dims.size() == 3 && imageFeatures.dims[0] == imageCount,
                        "Step3.7 image feature count mismatch.\n");

        auto patchIt = multimodalInput.find("patch_pixel_values");
        if (totalPatchCount > 0) {
            AssertInFastLLM(patchIt != multimodalInput.end() && !patchIt->second.empty() && patchIt->second[0] != nullptr,
                            "Step3.7 multimodal requires patch_pixel_values when num_patches is non-zero.\n");
            EncodeStep3p7PixelValues(*patchIt->second[0], patchFeatures);
            AssertInFastLLM(patchFeatures.dims.size() == 3 && patchFeatures.dims[0] == totalPatchCount,
                            "Step3.7 patch feature count mismatch.\n");
        }

        imageFeatures.ToDevice(DataDevice::CPU);
        if (imageFeatures.dataType != DataType::FLOAT32) {
            ToDataType(imageFeatures, DataType::FLOAT32);
            imageFeatures.ToDevice(DataDevice::CPU);
        }
        if (!patchFeatures.dims.empty()) {
            patchFeatures.ToDevice(DataDevice::CPU);
            if (patchFeatures.dataType != DataType::FLOAT32) {
                ToDataType(patchFeatures, DataType::FLOAT32);
                patchFeatures.ToDevice(DataDevice::CPU);
            }
        }

        int totalTokens = imageCount * step3p7ImageTokenLen + totalPatchCount * step3p7PatchTokenLen;
        std::vector<float> merged;
        merged.reserve((size_t)totalTokens * embed_dim);
        float *imagePtr = (float*)imageFeatures.cpuData;
        float *patchPtr = patchFeatures.dims.empty() ? nullptr : (float*)patchFeatures.cpuData;
        int patchOffset = 0;
        for (int i = 0; i < imageCount; i++) {
            for (int p = 0; p < patchCounts[i]; p++) {
                float *src = patchPtr + (long long)(patchOffset + p) * step3p7PatchTokenLen * embed_dim;
                merged.insert(merged.end(), src, src + (long long)step3p7PatchTokenLen * embed_dim);
            }
            patchOffset += patchCounts[i];
            float *src = imagePtr + (long long)i * step3p7ImageTokenLen * embed_dim;
            merged.insert(merged.end(), src, src + (long long)step3p7ImageTokenLen * embed_dim);
        }
        features.CopyFrom(Data(DataType::FLOAT32, {1, totalTokens, embed_dim}, merged));
    }

    void Step3p5Model::MergeStep3p7ImageFeaturesIntoText(const Data &inputIds,
                                                          const Data &imageFeatures,
                                                          Data &hiddenStates) {
        Data ids(inputIds);
        ids.ToDevice(DataDevice::CPU);
        if (ids.dataType != DataType::FLOAT32) {
            ToDataType(ids, DataType::FLOAT32);
            ids.ToDevice(DataDevice::CPU);
        }
        Data feats(imageFeatures);
        feats.ToDevice(DataDevice::CPU);
        if (feats.dataType != DataType::FLOAT32) {
            ToDataType(feats, DataType::FLOAT32);
            feats.ToDevice(DataDevice::CPU);
        }
        hiddenStates.ToDevice(DataDevice::CPU);
        if (hiddenStates.dataType != DataType::FLOAT32) {
            ToDataType(hiddenStates, DataType::FLOAT32);
            hiddenStates.ToDevice(DataDevice::CPU);
        }
        AssertInFastLLM(ids.dims.size() == 2 && ids.dims[0] == 1,
                        "Step3.7 multimodal input ids should have shape [1, seq].\n");
        AssertInFastLLM(feats.dims.size() == 3 && feats.dims[0] == 1 && feats.dims[2] == embed_dim,
                        "Step3.7 image features should have shape [1, tokens, hidden].\n");
        AssertInFastLLM(hiddenStates.dims.size() == 3 && hiddenStates.dims[0] == 1 &&
                        hiddenStates.dims[1] == ids.dims[1] && hiddenStates.dims[2] == embed_dim,
                        "Step3.7 text embedding shape mismatch.\n");

        float *idPtr = (float*)ids.cpuData;
        float *hiddenPtr = (float*)hiddenStates.cpuData;
        float *featPtr = (float*)feats.cpuData;
        int seq = ids.dims[1];
        int featureIndex = 0;
        for (int i = 0; i < seq; i++) {
            if ((int)idPtr[i] != step3p7ImageTokenId) {
                continue;
            }
            AssertInFastLLM(featureIndex < feats.dims[1],
                            "Step3.7 has more image placeholders than image features.\n");
            memcpy(hiddenPtr + (long long)i * embed_dim,
                   featPtr + (long long)featureIndex * embed_dim,
                   (size_t)embed_dim * sizeof(float));
            featureIndex++;
        }
        AssertInFastLLM(featureIndex == feats.dims[1],
                        "Step3.7 image feature count does not match image placeholders.\n");
    }

    int Step3p5Model::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <float> *retLogits) {
        Data attentionMaskCopy(attentionMask), positionIdsCopy(positionIds);
        std::vector <Data*> attentionMasks = {&attentionMaskCopy};
        std::vector <Data*> positionIdsVec = {&positionIdsCopy};
        std::vector <int> seqLens = {(int)inputIds.dims[1]};
        std::vector <GenerationConfig> generationConfigs = {generationConfig};
        std::vector <std::pair <Data*, Data*> > pagedPastKeyValues;
        for (int i = 0; i < (int)pastKeyValues.size(); i++) {
            pagedPastKeyValues.push_back(std::make_pair(&pastKeyValues[i].first, &pastKeyValues[i].second));
        }
        std::vector <std::vector <float>*> batchLogits;
        batchLogits.push_back(retLogits);
        if (CanUseGPUForward()) {
            return ForwardGPU(1, inputIds, attentionMasks, positionIdsVec, seqLens,
                              pagedPastKeyValues, generationConfigs, lastTokens, &batchLogits)[0];
        }
        return ForwardV2(1, inputIds, attentionMasks, positionIdsVec, seqLens,
                         pagedPastKeyValues, generationConfigs, lastTokens, &batchLogits)[0];
    }

    std::vector <int> Step3p5Model::ForwardMultimodal(
            const fastllm::Data &inputIds,
            const fastllm::Data &attentionMask,
            const fastllm::Data &positionIds,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            const std::map <std::string, std::vector <Data*> > &multimodalInput,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens,
            std::vector <std::vector <float>*> *retLogits) {
        std::vector<int> ret;
        std::vector<float> *logits = nullptr;
        if (retLogits != nullptr && !retLogits->empty()) {
            logits = (*retLogits)[0];
        }
        if (!step3p7VisionAvailable || multimodalInput.empty()) {
            ret.push_back(Forward(inputIds, attentionMask, positionIds, pastKeyValues,
                                  generationConfig, lastTokens, logits));
            return ret;
        }
        if (pastKeyValues.size() > 0 && pastKeyValues[0].second.dims.size() > 0) {
            ret.push_back(Forward(inputIds, attentionMask, positionIds, pastKeyValues,
                                  generationConfig, lastTokens, logits));
            return ret;
        }

        AssertInFastLLM(inputIds.dims.size() == 2 && inputIds.dims[0] == 1,
                        "Step3.7 multimodal currently supports a single prompt batch only.\n");

#ifdef USE_CUDA
        Step3p7DebugCudaMemory("multimodal before vision");
#endif
        Data imageFeatures;
        {
#ifdef USE_CUDA
            int visionDevice = -1;
            visionDevice = Step3p7ResolveVisionDevice(this->deviceMap);
            if (getenv("FASTLLM_STEP3P7_DEBUG") != nullptr) {
                printf("[Step3.7 debug] vision device=%d\n", visionDevice);
                fflush(stdout);
            }
            Step3p7ScopedVisionExecutor visionExecutor(visionDevice);
#endif
            EncodeStep3p7Images(multimodalInput, imageFeatures);
        }
#ifdef USE_CUDA
        ReleaseStep3p7VisionCuda();
        Step3p7DebugCudaMemory("multimodal after vision");
#endif

        Data embeddingResult, hiddenStates;
        Embedding(inputIds, weight["model.embed_tokens.weight"], embeddingResult);
        ToDataType(embeddingResult, hiddenStates, DataType::FLOAT32);
        MergeStep3p7ImageFeaturesIntoText(inputIds, imageFeatures, hiddenStates);
#ifdef USE_CUDA
        Step3p7DebugCudaMemory("multimodal after merge");
#endif

#ifndef USE_CUDA
        ErrorInFastLLM("Step3.7 multimodal currently requires CUDA text forward.\n");
        return ret;
#else
        bool useThreadTpForward = CanUseGPUForward();

        std::vector <std::pair <Data*, Data*> > pagedPastKeyValues;
        for (int i = 0; i < (int)pastKeyValues.size(); i++) {
            pagedPastKeyValues.push_back(std::make_pair(&pastKeyValues[i].first, &pastKeyValues[i].second));
        }

        struct PrecomputedGuard {
            Step3p5Model *model;
            explicit PrecomputedGuard(Step3p5Model *model) : model(model) {}
            ~PrecomputedGuard() { model->step3p7PrecomputedHiddenStates = nullptr; }
        } guard(this);
        std::vector <GenerationConfig> generationConfigs = {generationConfig};
        int seqLen = (int)inputIds.dims[1];
        int chunkSize = GetChunkedPrefillSize();
        if (chunkSize <= 0 || seqLen <= chunkSize) {
            Data attentionMaskCopy(attentionMask), positionIdsCopy(positionIds);
            std::vector <Data*> attentionMasks = {&attentionMaskCopy};
            std::vector <Data*> positionIdsVec = {&positionIdsCopy};
            std::vector <int> seqLens = {seqLen};
            step3p7PrecomputedHiddenStates = &hiddenStates;
            if (useThreadTpForward) {
                return ForwardGPU(1, inputIds, attentionMasks, positionIdsVec, seqLens,
                                  pagedPastKeyValues, generationConfigs, lastTokens, retLogits);
            }
            return ForwardV2(1, inputIds, attentionMasks, positionIdsVec, seqLens,
                             pagedPastKeyValues, generationConfigs, lastTokens, retLogits);
        }

        LastTokensManager emptyLastTokens;
        for (int st = 0; st < seqLen; ) {
            int curLen = std::min(chunkSize, seqLen - st);
            Data curInputIds, curPositionIds, curHiddenStates;
            Split(inputIds, 1, st, st + curLen, curInputIds);
            Split(positionIds, 1, st, st + curLen, curPositionIds);
            Split(hiddenStates, 1, st, st + curLen, curHiddenStates);

            std::vector <Data*> curAttentionMasks = {nullptr};
            std::vector <Data*> curPositionIdsVec = {&curPositionIds};
            std::vector <int> curSeqLens = {curLen};
            bool isLastChunk = (st + curLen >= seqLen);
#ifdef USE_CUDA
            if (getenv("FASTLLM_STEP3P7_DEBUG") != nullptr) {
                printf("[Step3.7 debug] text chunk %d-%d / %d\n", st, st + curLen, seqLen);
                fflush(stdout);
            }
            Step3p7DebugCudaMemory("before text chunk");
#endif
            step3p7PrecomputedHiddenStates = &curHiddenStates;
            if (useThreadTpForward) {
                ret = ForwardGPU(1, curInputIds, curAttentionMasks, curPositionIdsVec, curSeqLens,
                                 pagedPastKeyValues, generationConfigs,
                                 isLastChunk ? lastTokens : emptyLastTokens,
                                 isLastChunk ? retLogits : nullptr);
            } else {
                ret = ForwardV2(1, curInputIds, curAttentionMasks, curPositionIdsVec, curSeqLens,
                                pagedPastKeyValues, generationConfigs,
                                isLastChunk ? lastTokens : emptyLastTokens,
                                isLastChunk ? retLogits : nullptr);
            }
#ifdef USE_CUDA
            Step3p7DebugCudaMemory("after text chunk");
#endif
            st += curLen;
        }
        return ret;
#endif
    }

    std::vector <int> Step3p5Model::ForwardGPU(
        int batch,
        const Data &inputIds,
        const std::vector <Data*> &attentionMask,
        const std::vector <Data*> &positionIds,
        const std::vector <int> &seqLens,
        std::vector <std::pair <Data*, Data*> > &pastKeyValues,
        const std::vector <GenerationConfig> &generationConfigs,
        const LastTokensManager &lastTokens,
        std::vector <std::vector <float>*> *retLogits) {
#ifndef USE_CUDA
        return ForwardV2(batch, inputIds, attentionMask, positionIds, seqLens,
                         pastKeyValues, generationConfigs, lastTokens, retLogits);
#else
        (void)attentionMask;
        std::vector<int> devices;
        std::map<int, int> ratios;
        if (!CanUseGPUForward() ||
            !GetStep3p5GPUForwardDevices(this->deviceMap, devices, ratios)) {
            if (threadTpWorkerGroup.HasWorkers()) {
                threadTpWorkerGroup.Stop();
            }
            return ForwardV2(batch, inputIds, attentionMask, positionIds, seqLens,
                             pastKeyValues, generationConfigs, lastTokens, retLogits);
        }
        bool tensorParallel = devices.size() > 1;
        bool hasPrecomputedHiddenStates = step3p7PrecomputedHiddenStates != nullptr;
        bool skipCudaEmbeddingForVisionWarmup =
            step3p7VisionAvailable && autoWarmupRunning.load(std::memory_order_acquire);
        bool useCpuEmbedding = hasPrecomputedHiddenStates ||
                               skipCudaEmbeddingForVisionWarmup ||
                               !GetCudaEmbedding() || GetLowMemMode();
        const DataType computeType = ResolveStep3p5ThreadTpComputeType(this->dataType);
        if (!useCpuEmbedding) {
            PrepareStep3p5CudaEmbeddingWeightType(weight["model.embed_tokens.weight"], computeType);
        }
        AssertInFastLLM((int)pastKeyValues.size() >= batch * block_cnt,
                        "Step3p5 ForwardGPU: pastKeyValues size mismatch.\n");
        AssertInFastLLM((int)generationConfigs.size() >= batch,
                        "Step3p5 ForwardGPU: generation config size mismatch.\n");
        AssertInFastLLM((int)positionIds.size() >= batch && positionIds[0] != nullptr,
                        "Step3p5 ForwardGPU: positionIds size mismatch.\n");
        AssertInFastLLM(!GetKVCacheInCPU(),
                        "Step3p5 ForwardGPU doesn't support CPU KV cache.\n");
        if (tensorParallel) {
            AssertInFastLLM(FastllmInitNccl(devices),
                            "Step3p5 ForwardGPU requires NCCL initialization.\n");
        }

        if (threadTpPagedCacheBase < 0) {
            threadTpPagedCacheBase = step3p5ThreadTpNextPagedCacheBase.fetch_add(
                std::max(1, block_cnt * ((int)devices.size() + 1)));
        }

        int totalLen = 0;
        bool all1 = true;
        for (int b = 0; b < batch; b++) {
            totalLen += seqLens[b];
            all1 &= (seqLens[b] == 1);
        }
        bool isPrefill = !all1;

        std::vector<Data> positionIdsCpu;
        positionIdsCpu.reserve(batch);
        std::vector<float> vPositionIds;
        vPositionIds.reserve(totalLen);
        for (int b = 0; b < batch; b++) {
            AssertInFastLLM(positionIds[b] != nullptr,
                            "Step3p5 ForwardGPU: null positionIds.\n");
            positionIdsCpu.emplace_back();
            positionIdsCpu.back().CopyFrom(*positionIds[b]);
            positionIdsCpu.back().ToDevice(DataDevice::CPU);
            if (all1) {
                vPositionIds.push_back(Step3p5ReadFloat(positionIdsCpu.back(), 0));
            } else {
                for (int i = 0; i < seqLens[b]; i++) {
                    vPositionIds.push_back(Step3p5ReadFloat(positionIdsCpu.back(), i));
                }
            }
        }
        Data allPositionIds;
        allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, (int)vPositionIds.size()}, vPositionIds));

        Data gpuInputIds;
        gpuInputIds.CopyFrom(inputIds);
        if (tensorParallel) {
            PrepareMultiCudaReplicatedData(gpuInputIds, devices, true);
            PrepareMultiCudaReplicatedData(allPositionIds, devices, true);
        }

        auto ensureInitializedAdd1 = [&]() {
            if (initialized_add1) {
                return;
            }
            for (int i = 0; i < block_cnt; i++) {
                Step3p5Add1(this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"]);
                Step3p5Add1(this->weight["model.layers." + std::to_string(i) + ".self_attn.q_norm.weight"]);
                Step3p5Add1(this->weight["model.layers." + std::to_string(i) + ".self_attn.k_norm.weight"]);
                Step3p5Add1(this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"]);
                std::string routerName = "model.layers." + std::to_string(i) + ".moe.gate.weight";
                if (need_fp32_gate && weight.weight.find(routerName) != weight.weight.end()) {
                    ToDataType(this->weight[routerName], DataType::FLOAT32);
                }
            }
            Step3p5Add1(this->weight["model.norm.weight"]);
            initialized_add1 = true;
        };

        auto hasMoeCache = [&](const std::unordered_map<int, std::vector<std::vector<Data*> > > &weightCache,
                               const std::unordered_map<int, std::vector<std::vector<Data*> > > &biasCache) {
            int expectedSize = this->num_experts * 2 + 2;
            for (int device : devices) {
                auto weightIt = weightCache.find(device);
                auto biasIt = biasCache.find(device);
                if (weightIt == weightCache.end() || biasIt == biasCache.end() ||
                    (int)weightIt->second.size() != block_cnt ||
                    (int)biasIt->second.size() != block_cnt) {
                    return false;
                }
                for (int i = 0; i < block_cnt; i++) {
                    if ((int)weightIt->second[i].size() != expectedSize ||
                        (int)biasIt->second[i].size() != expectedSize) {
                        return false;
                    }
                }
            }
            return true;
        };

        DivisionScheme expertScheme = BuildStep3p5ExpertScheme(devices, ratios, num_experts);
        auto fillMoeCache = [&](std::unordered_map<int, std::vector<std::vector<Data*> > > &weightCache,
                                std::unordered_map<int, std::vector<std::vector<Data*> > > &biasCache,
                                std::unordered_map<int, std::vector<std::vector<Data> > > &ownedCache,
                                std::unordered_map<int, std::vector<std::vector<Data*> > > &fusedCache,
                                std::unordered_map<int, std::vector<std::vector<Data> > > &ownedFusedCache,
                                std::unordered_map<int, std::vector<std::pair<int, int> > > &fusedRanges,
                                bool expertParallel) {
            weightCache.clear();
            biasCache.clear();
            ownedCache.clear();
            fusedCache.clear();
            ownedFusedCache.clear();
            fusedRanges.clear();
            int expectedSize = this->num_experts * 2 + 2;
            for (int device : devices) {
                auto &deviceWeights = weightCache[device];
                auto &deviceBiass = biasCache[device];
                auto &deviceOwned = ownedCache[device];
                auto &deviceFused = fusedCache[device];
                auto &deviceOwnedFused = ownedFusedCache[device];
                auto &deviceFusedRanges = fusedRanges[device];
                deviceWeights.resize(block_cnt);
                deviceBiass.resize(block_cnt);
                deviceOwned.resize(block_cnt);
                deviceFused.resize(block_cnt);
                deviceOwnedFused.resize(block_cnt);
                deviceFusedRanges.assign(block_cnt, {-1, -1});
                for (int i = 0; i < block_cnt; i++) {
                    auto &layerWeights = deviceWeights[i];
                    auto &layerBiass = deviceBiass[i];
                    auto &layerOwned = deviceOwned[i];
                    auto &layerFused = deviceFused[i];
                    auto &layerOwnedFused = deviceOwnedFused[i];
                    layerWeights.assign(expectedSize, nullptr);
                    layerBiass.assign(expectedSize, nullptr);
                    layerOwned.resize(expectedSize);
                    layerFused.clear();
                    layerOwnedFused.clear();
                    if (!IsMoeLayer(i)) {
                        continue;
                    }
                    if (i < (int)moeGate3DWeights.size() &&
                        moeGate3DWeights[i] != nullptr && moeUp3DWeights[i] != nullptr &&
                        moeDown3DWeights[i] != nullptr) {
                        std::pair<int, int> expertRange = {0, this->num_experts};
                        if (expertParallel) {
                            auto schemeIt = expertScheme.find(device);
                            if (schemeIt == expertScheme.end() ||
                                (int)schemeIt->second.size() != 1) {
                                expertRange = {-1, -1};
                            } else {
                                expertRange = schemeIt->second[0];
                            }
                        }

                        if (expertRange.first >= 0 && expertRange.first < expertRange.second &&
                            expertRange.second <= this->num_experts) {
                            layerOwnedFused.resize(3);
                            std::string fusedPrefix = "model.layers." + std::to_string(i) +
                                ".moe.tp." + std::to_string(device) +
                                ".experts." + std::to_string(expertRange.first) + "_" +
                                std::to_string(expertRange.second) + ".";
                            Step3p5MakeExpertRangeCopy(layerOwnedFused[0], *moeGate3DWeights[i],
                                                       fusedPrefix + "gate_proj.weight",
                                                       expertRange.first, expertRange.second);
                            Step3p5MakeExpertRangeCopy(layerOwnedFused[1], *moeUp3DWeights[i],
                                                       fusedPrefix + "up_proj.weight",
                                                       expertRange.first, expertRange.second);
                            Step3p5MakeExpertRangeCopy(layerOwnedFused[2], *moeDown3DWeights[i],
                                                       fusedPrefix + "down_proj.weight",
                                                       expertRange.first, expertRange.second);
                            for (int w = 0; w < 3; w++) {
                                Step3p5PrepareFusedMoeWeightForCuda(
                                    layerOwnedFused[w], layerOwnedFused[w].dims[1], device);
                            }
                            layerFused = {&layerOwnedFused[0], &layerOwnedFused[1], &layerOwnedFused[2]};
                            deviceFusedRanges[i] = expertRange;
                            continue;
                        }
                    }
                    for (int expert = 0; expert < this->num_experts; expert++) {
                        if (expertParallel && !Step3p5ExpertOnDevice(expertScheme, device, expert)) {
                            continue;
                        }
                        Data *gateup = nullptr;
                        Data *down = nullptr;
                        if (i < (int)moeGate3DWeights.size() &&
                            moeGate3DWeights[i] != nullptr && moeUp3DWeights[i] != nullptr &&
                            moeDown3DWeights[i] != nullptr) {
                            int idx = (expert + 1) * 2;
                            Data &ownedGateup = layerOwned[idx];
                            Data &ownedDown = layerOwned[idx + 1];
                            if (ownedGateup.dims.empty()) {
                                Data gateCopy, upCopy;
                                std::string expertPrefix = "model.layers." + std::to_string(i) +
                                    ".moe.tp." + std::to_string(device) +
                                    ".experts." + std::to_string(expert) + ".";
                                Step3p5MakeExpertCopy(gateCopy, *moeGate3DWeights[i],
                                                      expertPrefix + "gate_proj.weight", expert);
                                Step3p5MakeExpertCopy(upCopy, *moeUp3DWeights[i],
                                                    expertPrefix + "up_proj.weight", expert);
                                Step3p5MakeGateUpWeight(ownedGateup, gateCopy, upCopy,
                                                        expertPrefix + "gateup_proj.weight");
                                Step3p5MakeExpertCopy(ownedDown, *moeDown3DWeights[i],
                                                      expertPrefix + "down_proj.weight", expert);
                            }
                            gateup = &ownedGateup;
                            down = &ownedDown;
                        } else if (i < (int)weights.size() &&
                                   (int)weights[i].size() >= expectedSize) {
                            int idx = (expert + 1) * 2;
                            gateup = weights[i][idx];
                            down = weights[i][idx + 1];
                        }
                        if (gateup == nullptr || down == nullptr) {
                            continue;
                        }
                        gateup->tpPackType = TP_PACK_GATEUP;
                        gateup->tpLinearType = TP_LINEAR_ROW;
                        down->tpLinearType = TP_LINEAR_COLUMN;
                        gateup->ToDevice(DataDevice::CUDA, {device}, true);
                        down->ToDevice(DataDevice::CUDA, {device}, true);
                        int idx = (expert + 1) * 2;
                        layerWeights[idx] = gateup;
                        layerWeights[idx + 1] = down;
                    }
                }
            }
        };

        std::vector<DivisionScheme> localKvHeadSchemes;
        DivisionScheme localLmHeadScheme;
        const std::vector<DivisionScheme> *kvHeadSchemes = &localKvHeadSchemes;
        const DivisionScheme *lmHeadScheme = &localLmHeadScheme;
        Data &lmHead = weight["lm_head.weight"];

        if (tensorParallel) {
            auto usePreparedThreadTpSchemes = [&]() {
                AssertInFastLLM(threadTpPreparedDevices == devices && threadTpPreparedRatios == ratios,
                                "Step3p5 ForwardGPU thread TP device config changed after weights were prepared.\n");
                AssertInFastLLM((int)threadTpKVHeadSchemes.size() == block_cnt &&
                                !threadTpLmHeadScheme.empty() &&
                                hasMoeCache(threadTpMoeWeights, threadTpMoeBiass),
                                "Step3p5 ForwardGPU thread TP cached weight schemes are incomplete.\n");
                kvHeadSchemes = &threadTpKVHeadSchemes;
                lmHeadScheme = &threadTpLmHeadScheme;
            };

            if (threadTpWeightsPrepared.load(std::memory_order_acquire)) {
                usePreparedThreadTpSchemes();
            } else {
                std::lock_guard<std::mutex> guard(threadTpWeightPrepareLock);
                ensureInitializedAdd1();
                PrepareMoeWeights();
                if (!threadTpWeightsPrepared.load(std::memory_order_relaxed)) {
                    auto prepareReplicated = [&](const std::string &name) {
                        if (weight.weight.find(name) != weight.weight.end()) {
                            PrepareMultiCudaReplicatedData(this->weight[name], devices, true);
                        }
                    };
                    auto prepareFeedForward = [&](const std::string &gateupName,
                                                  const std::string &gateName,
                                                  const std::string &upName,
                                                  const std::string &downName,
                                                  float limit) {
                        if (limit == 0.0f && weight.weight.find(gateupName) != weight.weight.end()) {
                            Data &gateup = weight[gateupName];
                            Data &gateupBias = GetThreadTensorParallelBias(gateupName + ".tp_bias");
                            gateup.tpPackType = TP_PACK_GATEUP;
                            std::vector<int> devCopy = devices;
                            DivisionScheme gateScheme = BuildMultiCudaRowSplitScheme(gateup, devCopy, ratios);
                            AssertInFastLLM(SplitMultiCudaWeight(gateup, gateupBias, devCopy, gateScheme, 0),
                                            "Step3p5 ForwardGPU failed to split " + gateupName + ".\n");

                            Data &downBias = GetThreadTensorParallelBias(downName + ".tp_bias");
                            DivisionScheme downScheme = ExtractStep3p5FirstRangeScheme(gateScheme);
                            devCopy = devices;
                            AssertInFastLLM(SplitMultiCudaWeight(weight[downName], downBias, devCopy, downScheme, 1),
                                            "Step3p5 ForwardGPU failed to split " + downName + ".\n");
                        } else {
                            AssertInFastLLM(weight.weight.find(gateName) != weight.weight.end() &&
                                            weight.weight.find(upName) != weight.weight.end(),
                                            "Step3p5 ForwardGPU requires separate gate/up weights.\n");
                            Data &gate = weight[gateName];
                            Data &gateBias = GetThreadTensorParallelBias(gateName + ".tp_bias");
                            std::vector<int> devCopy = devices;
                            DivisionScheme gateScheme = BuildMultiCudaRowSplitScheme(gate, devCopy, ratios);
                            AssertInFastLLM(SplitMultiCudaWeight(gate, gateBias, devCopy, gateScheme, 0),
                                            "Step3p5 ForwardGPU failed to split " + gateName + ".\n");

                            Data &upBias = GetThreadTensorParallelBias(upName + ".tp_bias");
                            devCopy = devices;
                            AssertInFastLLM(SplitMultiCudaWeight(weight[upName], upBias, devCopy, gateScheme, 0),
                                            "Step3p5 ForwardGPU failed to split " + upName + ".\n");

                            Data &downBias = GetThreadTensorParallelBias(downName + ".tp_bias");
                            DivisionScheme downScheme = ExtractStep3p5FirstRangeScheme(gateScheme);
                            devCopy = devices;
                            AssertInFastLLM(SplitMultiCudaWeight(weight[downName], downBias, devCopy, downScheme, 1),
                                            "Step3p5 ForwardGPU failed to split " + downName + ".\n");
                        }
                    };

                    if (!useCpuEmbedding) {
                        prepareReplicated("model.embed_tokens.weight");
                    }
                    prepareReplicated("model.norm.weight");
                    threadTpKVHeadSchemes.assign(block_cnt, DivisionScheme());

                    for (int i = 0; i < block_cnt; i++) {
                        std::string prefix = "model.layers." + std::to_string(i) + ".";
                        std::string inputRmsName = prefix + "input_layernorm.weight";
                        std::string mergeQkvWeightName = prefix + "self_attn.mergeqkv.weight";
                        std::string mergeQkvBiasName = prefix + "self_attn.mergeqkv.bias";
                        std::string qNormName = prefix + "self_attn.q_norm.weight";
                        std::string kNormName = prefix + "self_attn.k_norm.weight";
                        std::string gProjName = prefix + "self_attn.g_proj.weight";
                        std::string oWeightName = prefix + "self_attn.o_proj.weight";
                        std::string oBiasName = prefix + "self_attn.o_proj.bias";
                        std::string postRmsName = prefix + "post_attention_layernorm.weight";
                        int qHeads = LayerAttentionHeads(i);
                        int kvHeads = LayerKeyValueHeads(i);

                        AssertInFastLLM(weight.weight.find(mergeQkvWeightName) != weight.weight.end(),
                                        "Step3p5 ForwardGPU requires merged qkv weight.\n");
                        AssertInFastLLM(weight.weight.find(gProjName) != weight.weight.end(),
                                        "Step3p5 ForwardGPU requires attention gate weight.\n");

                        prepareReplicated(inputRmsName);
                        prepareReplicated(qNormName);
                        prepareReplicated(kNormName);
                        prepareReplicated(postRmsName);

                        Data &mergeW = weight[mergeQkvWeightName];
                        Data &mergeB = GetThreadTensorParallelBias(mergeQkvBiasName);
                        mergeW.tpPackType = TP_PACK_QKV;
                        mergeW.tpQHeads = qHeads;
                        mergeW.tpKVHeads = kvHeads;
                        mergeW.tpHeadDim = head_dim;
                        std::vector<int> devCopy = devices;
                        DivisionScheme qkvScheme = BuildMultiCudaRowSplitScheme(mergeW, devCopy, ratios);
                        AssertInFastLLM(SplitMultiCudaWeight(mergeW, mergeB, devCopy, qkvScheme, 0),
                                        "Step3p5 ForwardGPU failed to split " + mergeQkvWeightName + ".\n");

                        int qWidth = qHeads * head_dim;
                        DivisionScheme qScheme = ExtractStep3p5FirstRangeScheme(qkvScheme);
                        threadTpKVHeadSchemes[i] = ExtractStep3p5KVHeadScheme(qkvScheme, qWidth, head_dim);

                        Data &oB = GetThreadTensorParallelBias(oBiasName);
                        devCopy = devices;
                        AssertInFastLLM(SplitMultiCudaWeight(weight[oWeightName], oB, devCopy, qScheme, 1),
                                        "Step3p5 ForwardGPU failed to split " + oWeightName + ".\n");

                        Data &gB = GetThreadTensorParallelBias(gProjName + ".tp_bias");
                        DivisionScheme gScheme = ExtractStep3p5QHeadScheme(qScheme, head_dim);
                        devCopy = devices;
                        AssertInFastLLM(SplitMultiCudaWeight(weight[gProjName], gB, devCopy, gScheme, 0),
                                        "Step3p5 ForwardGPU failed to split " + gProjName + ".\n");

                        if (!IsMoeLayer(i)) {
                            prepareFeedForward(prefix + "mlp.gateup_proj.weight",
                                               prefix + "mlp.gate_proj.weight",
                                               prefix + "mlp.up_proj.weight",
                                               prefix + "mlp.down_proj.weight",
                                               Step3p5LayerLimit(swiglu_limits_shared, i));
                        } else {
                            prepareFeedForward(prefix + "share_expert.gateup_proj.weight",
                                               prefix + "share_expert.gate_proj.weight",
                                               prefix + "share_expert.up_proj.weight",
                                               prefix + "share_expert.down_proj.weight",
                                               Step3p5LayerLimit(swiglu_limits_shared, i));
                            prepareReplicated(prefix + "moe.gate.weight");
                            if (use_moe_router_bias &&
                                weight.weight.find(prefix + "moe.router_bias") != weight.weight.end()) {
                                prepareReplicated(prefix + "moe.router_bias");
                            }
                        }
                    }

                    fillMoeCache(threadTpMoeWeights, threadTpMoeBiass, threadTpOwnedMoeWeights,
                                 threadTpFusedMoeWeights, threadTpOwnedFusedMoeWeights,
                                 threadTpFusedMoeExpertRanges, true);

                    Data &lmHeadBias = GetThreadTensorParallelBias("lm_head.weight.tp_bias");
                    std::vector<int> devCopy = devices;
                    threadTpLmHeadScheme = BuildMultiCudaRowSplitScheme(lmHead, devCopy, ratios);
                    AssertInFastLLM(SplitMultiCudaWeight(lmHead, lmHeadBias, devCopy, threadTpLmHeadScheme, 0),
                                    "Step3p5 ForwardGPU failed to split lm_head.weight.\n");

                    threadTpPreparedDevices = devices;
                    threadTpPreparedRatios = ratios;
                    threadTpWeightsPrepared.store(true, std::memory_order_release);
                }
                usePreparedThreadTpSchemes();
            }
        } else {
            std::lock_guard<std::mutex> guard(threadTpWeightPrepareLock);
            ensureInitializedAdd1();
            PrepareMoeWeights();
            if (!singleGpuWeightsPrepared.load(std::memory_order_relaxed) ||
                !hasMoeCache(singleGpuMoeWeights, singleGpuMoeBiass)) {
                int device = devices[0];
                for (int i = 0; i < block_cnt; i++) {
                    std::string prefix = "model.layers." + std::to_string(i) + ".";
                    std::string mergeQkvWeightName = prefix + "self_attn.mergeqkv.weight";
                    AssertInFastLLM(weight.weight.find(mergeQkvWeightName) != weight.weight.end(),
                                    "Step3p5 ForwardGPU requires merged qkv weight.\n");
                    Data &mergeW = weight[mergeQkvWeightName];
                    mergeW.tpPackType = TP_PACK_QKV;
                    mergeW.tpQHeads = LayerAttentionHeads(i);
                    mergeW.tpKVHeads = LayerKeyValueHeads(i);
                    mergeW.tpHeadDim = head_dim;
                    auto markGateup = [&](const std::string &name) {
                        if (weight.weight.find(name) != weight.weight.end()) {
                            weight[name].tpPackType = TP_PACK_GATEUP;
                        }
                    };
                    markGateup(prefix + "mlp.gateup_proj.weight");
                    markGateup(prefix + "share_expert.gateup_proj.weight");
                }
                fillMoeCache(singleGpuMoeWeights, singleGpuMoeBiass, singleGpuOwnedMoeWeights,
                             singleGpuFusedMoeWeights, singleGpuOwnedFusedMoeWeights,
                             singleGpuFusedMoeExpertRanges, false);
                singleGpuWeightsPrepared.store(true, std::memory_order_release);
            }
            localKvHeadSchemes.assign(block_cnt, DivisionScheme());
            for (int i = 0; i < block_cnt; i++) {
                localKvHeadSchemes[i][devices[0]].push_back({0, LayerKeyValueHeads(i)});
            }
            localLmHeadScheme[devices[0]].push_back({0, lmHead.dims[0]});
        }

        if (tensorParallel && !useCpuEmbedding) {
            PrepareMultiCudaReplicatedData(weight["model.embed_tokens.weight"], devices, true);
        }
        Data cpuEmbeddingHiddenStates;
        Data *precomputedHiddenStates = step3p7PrecomputedHiddenStates;
        if (precomputedHiddenStates != nullptr) {
            if (precomputedHiddenStates->dataType != computeType) {
                ToDataType(*precomputedHiddenStates, computeType);
            }
            PrepareStep3p5CpuEmbeddingHiddenStates(*precomputedHiddenStates, devices, threadTpWorkerGroup);
        } else if (useCpuEmbedding) {
            Data cpuInputIds;
            cpuInputIds.CopyFrom(inputIds);
            Step3p5CpuEmbeddingDirect(cpuInputIds, weight["model.embed_tokens.weight"],
                                      cpuEmbeddingHiddenStates, computeType);
            PrepareStep3p5CpuEmbeddingHiddenStates(cpuEmbeddingHiddenStates, devices, threadTpWorkerGroup);
            precomputedHiddenStates = &cpuEmbeddingHiddenStates;
        }
        std::vector<std::vector<std::pair<Data*, Data*> > > localPastKeyValues;
        if (tensorParallel) {
            localPastKeyValues.resize(devices.size());
            for (int r = 0; r < (int)devices.size(); r++) {
                int device = devices[r];
                localPastKeyValues[r].resize(pastKeyValues.size());
                for (int i = 0; i < (int)pastKeyValues.size(); i++) {
                    DataType keyCacheType = ResolveStep3p5ThreadTpCacheType(
                        pastKeyValues[i].first->dataType, computeType);
                    DataType valueCacheType = ResolveStep3p5ThreadTpCacheType(
                        pastKeyValues[i].second->dataType, computeType);
                    localPastKeyValues[r][i].first = EnsureStep3p5ThreadTpLocalCache(
                        *pastKeyValues[i].first, device, keyCacheType);
                    localPastKeyValues[r][i].second = EnsureStep3p5ThreadTpLocalCache(
                        *pastKeyValues[i].second, device, valueCacheType);
                }
            }
        } else {
            int device = devices[0];
            for (int i = 0; i < (int)pastKeyValues.size(); i++) {
                DataType keyCacheType = ResolveStep3p5ThreadTpCacheType(
                    pastKeyValues[i].first->dataType, computeType);
                DataType valueCacheType = ResolveStep3p5ThreadTpCacheType(
                    pastKeyValues[i].second->dataType, computeType);
                PrepareStep3p5SingleCudaCache(*pastKeyValues[i].first, device, keyCacheType);
                PrepareStep3p5SingleCudaCache(*pastKeyValues[i].second, device, valueCacheType);
            }
        }

        std::vector<std::exception_ptr> errors(devices.size());
        std::vector<Data> localLogits(devices.size());
        if (devices.size() == 1) {
            if (threadTpWorkerGroup.HasWorkers()) {
                threadTpWorkerGroup.Stop();
            }
            ForwardSingleGPU(devices[0], ratios, batch, gpuInputIds, allPositionIds,
                             seqLens, pastKeyValues, all1, isPrefill,
                             false, true, threadTpPagedCacheBase, localLogits[0],
                             precomputedHiddenStates);
        } else {
            threadTpWorkerGroup.Run(devices, [&](int r) {
                ForwardSingleGPU(devices[r], ratios, batch, gpuInputIds, allPositionIds,
                                 seqLens, localPastKeyValues[r], all1, isPrefill,
                                 tensorParallel, r == 0,
                                 threadTpPagedCacheBase + r * block_cnt,
                                 localLogits[r], precomputedHiddenStates);
                FastllmCudaSetDevice(devices[r]);
                ForceDeviceSync();
            }, errors);
            for (auto &error : errors) {
                if (error) {
                    std::rethrow_exception(error);
                }
            }
        }

        if (tensorParallel) {
            auto validLocalMeta = [](Data *data) {
                return data != nullptr && data->dims.size() >= 3;
            };
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < block_cnt; i++) {
                    int idx = b * block_cnt + i;
                    Data *localKeyMeta = !localPastKeyValues.empty() &&
                        idx < (int)localPastKeyValues[0].size() ? localPastKeyValues[0][idx].first : nullptr;
                    Data *localValueMeta = !localPastKeyValues.empty() &&
                        idx < (int)localPastKeyValues[0].size() ? localPastKeyValues[0][idx].second : nullptr;
                    if ((!validLocalMeta(localKeyMeta) || !validLocalMeta(localValueMeta)) &&
                         localPastKeyValues.size() > 1) {
                        for (auto &rankPastKeyValues : localPastKeyValues) {
                            if (idx < (int)rankPastKeyValues.size()) {
                                Data *candidateKey = rankPastKeyValues[idx].first;
                                Data *candidateValue = rankPastKeyValues[idx].second;
                                if (!validLocalMeta(localKeyMeta) && validLocalMeta(candidateKey)) {
                                    localKeyMeta = candidateKey;
                                }
                                if (!validLocalMeta(localValueMeta) && validLocalMeta(candidateValue)) {
                                    localValueMeta = candidateValue;
                                }
                                if (validLocalMeta(localKeyMeta) && validLocalMeta(localValueMeta)) {
                                    break;
                                }
                            }
                        }
                    }
                    SyncStep3p5ThreadTpRootCacheMetaFromLocal(*pastKeyValues[idx].first, localKeyMeta,
                                                              devices, (*kvHeadSchemes)[i],
                                                              LayerKeyValueHeads(i), head_dim);
                    SyncStep3p5ThreadTpRootCacheMetaFromLocal(*pastKeyValues[idx].second, localValueMeta,
                                                              devices, (*kvHeadSchemes)[i],
                                                              LayerKeyValueHeads(i), head_dim);
                }
            }
        }

        int vocabSize = lmHead.dims[0];
        const char *printLogitsEnv = std::getenv("FASTLLM_PRINT_LOGITS");
        bool printLogits = GetFastllmEnv().printLogits ||
            (printLogitsEnv != nullptr && printLogitsEnv[0] != '\0' &&
             !(printLogitsEnv[0] == '0' && printLogitsEnv[1] == '\0'));
        bool allSimpleCudaSampling = true;
        int cudaSamplingTopK = 1;
        if (!printLogits && Step3p5CanUseCudaFullLogitsSampling(generationConfigs, retLogits, batch,
                                                                allSimpleCudaSampling, cudaSamplingTopK)) {
            Data &fullCudaLogits = Step3p5ThreadLocalCudaSamplingFullLogits();
            Step3p5GatherShardLogitsToRootCuda(devices[0], devices, *lmHeadScheme,
                                               localLogits, batch, vocabSize,
                                               fullCudaLogits);
            void *oldExecutor = GetExecutor();
            Executor samplingExecutor;
            samplingExecutor.SetFirstDevice("cuda:" + std::to_string(devices[0]));
            SetCurrentThreadExecutor(&samplingExecutor);
            ResetLogitsOfEOS(batch, &fullCudaLogits, pastKeyValues, generationConfigs);
            SetCurrentThreadExecutor(oldExecutor);
            std::vector<int> lastRet = Step3p5SampleFromRootCudaLogits(devices[0], fullCudaLogits, batch,
                                                                        cudaSamplingTopK, allSimpleCudaSampling,
                                                                        generationConfigs);
            return lastRet;
        }

        Data fullLogits(DataType::FLOAT32);
        fullLogits.Resize({batch, vocabSize});
        fullLogits.Allocate();
        std::fill((float*)fullLogits.cpuData,
                  (float*)fullLogits.cpuData + fullLogits.Count(0), -1.0e30f);

        for (int r = 0; r < (int)devices.size(); r++) {
            int device = devices[r];
            localLogits[r].ToDevice(DataDevice::CPU);
            int localVocab = localLogits[r].dims.back();
            int rows = localLogits[r].Count(0) / localVocab;
            AssertInFastLLM(rows == batch,
                            "Step3p5 ForwardGPU: local logits batch mismatch.\n");
            float *src = (float*)localLogits[r].cpuData;
            float *dst = (float*)fullLogits.cpuData;
            int localOffset = 0;
            auto schemeIt = lmHeadScheme->find(device);
            AssertInFastLLM(schemeIt != lmHeadScheme->end(),
                            "Step3p5 ForwardGPU: missing lm_head split scheme.\n");
            for (auto &range : schemeIt->second) {
                int len = range.second - range.first;
                AssertInFastLLM(range.first >= 0 && range.second <= vocabSize &&
                                localOffset + len <= localVocab,
                                "Step3p5 ForwardGPU: invalid lm_head split range.\n");
                for (int b = 0; b < batch; b++) {
                    memcpy(dst + (long long)b * vocabSize + range.first,
                           src + (long long)b * localVocab + localOffset,
                           (size_t)len * sizeof(float));
                }
                localOffset += len;
            }
        }

        ResetLogitsOfEOS(batch, &fullLogits, pastKeyValues, generationConfigs);
        if (printLogits) {
            printf("Step3p5 ForwardGPU logits:\n");
            fullLogits.Print();
        }

        std::vector<int> lastRet;
        LastTokensUnit emptyLastTokens;
        for (int b = 0; b < batch; b++) {
            if (generationConfigs[b].output_logits && retLogits != nullptr &&
                b < (int)retLogits->size() && (*retLogits)[b] != nullptr) {
                (*retLogits)[b]->resize(vocabSize);
                memcpy((float*)(*retLogits)[b]->data(),
                       (float*)fullLogits.cpuData + (long long)b * vocabSize,
                       (size_t)vocabSize * sizeof(float));
            }
            const LastTokensUnit &unit = b < (int)lastTokens.units.size() ?
                lastTokens.units[b] : emptyLastTokens;
            lastRet.push_back(LLMSampling(fullLogits, b, generationConfigs[b], unit));
        }
        return lastRet;
#endif
    }

    std::vector <int> Step3p5Model::ForwardV2ThreadTensorParallel(
        int batch,
        const Data &inputIds,
        const std::vector <Data*> &attentionMask,
        const std::vector <Data*> &positionIds,
        const std::vector <int> &seqLens,
        std::vector <std::pair <Data*, Data*> > &pastKeyValues,
        const std::vector <GenerationConfig> &generationConfigs,
        const LastTokensManager &lastTokens,
        std::vector <std::vector <float>*> *retLogits) {
        return ForwardGPU(batch, inputIds, attentionMask, positionIds, seqLens,
                          pastKeyValues, generationConfigs, lastTokens, retLogits);
    }

    std::vector <int> Step3p5Model::ForwardV2(int batch,
        const Data &inputIds,
        const std::vector <Data*> &attentionMask,
        const std::vector <Data*> &positionIds,
        const std::vector <int> &seqLens,
        std::vector <std::pair <Data*, Data*> > &pastKeyValues,
        const std::vector <GenerationConfig> &generationConfigs,
        const LastTokensManager &lastTokens,
        std::vector <std::vector <float>*> *retLogits) {
        AssertInFastLLM(batch > 0, "Step3p5 batch should be positive.");
        AssertInFastLLM((int)seqLens.size() >= batch, "Step3p5 seqLens missing.");
        AssertInFastLLM((int)generationConfigs.size() >= batch, "Step3p5 generation configs missing.");
        if (IsThreadTensorParallelEnabled()) {
            return ForwardGPU(batch, inputIds, attentionMask, positionIds, seqLens,
                              pastKeyValues, generationConfigs, lastTokens, retLogits);
        }
        bool all1 = true;
        int totalLen = 0;
        for (int b = 0; b < batch; b++) {
            int len = seqLens[b];
            all1 &= (len == 1);
            totalLen += len;
        }

        auto runSplitBatchForward = [&]() -> std::vector<int> {
            std::vector<int> ret;
            ret.reserve(batch);

            int inputOffset = 0;
            for (int b = 0; b < batch; b++) {
                Data curInputIds;
                Split(inputIds, 1, inputOffset, inputOffset + seqLens[b], curInputIds);
                inputOffset += seqLens[b];

                std::vector<Data*> curAttentionMask = {
                    b < (int)attentionMask.size() ? attentionMask[b] : nullptr
                };
                std::vector<Data*> curPositionIds = {
                    b < (int)positionIds.size() ? positionIds[b] : nullptr
                };
                std::vector<int> curSeqLens = {seqLens[b]};
                std::vector<GenerationConfig> curGenerationConfigs = {generationConfigs[b]};

                LastTokensManager curLastTokens;
                if (b < (int)lastTokens.units.size()) {
                    curLastTokens.units.push_back(lastTokens.units[b]);
                } else {
                    int lastN = generationConfigs[b].last_n <= 0 ? max_positions : generationConfigs[b].last_n;
                    curLastTokens = LastTokensManager(1, lastN);
                }

                std::vector<std::pair<Data*, Data*> > curPastKeyValues;
                curPastKeyValues.reserve(block_cnt);
                int pastOffset = b * block_cnt;
                AssertInFastLLM((int)pastKeyValues.size() >= pastOffset + block_cnt,
                                "Step3p5 pastKeyValues missing.");
                for (int i = 0; i < block_cnt; i++) {
                    curPastKeyValues.push_back(pastKeyValues[pastOffset + i]);
                }

                std::vector<std::vector<float>*> curLogits;
                std::vector<std::vector<float>*> *curLogitsPtr = nullptr;
                if (retLogits != nullptr) {
                    curLogits.push_back(b < (int)retLogits->size() ? (*retLogits)[b] : nullptr);
                    curLogitsPtr = &curLogits;
                }

                std::vector<int> curRet = ForwardV2(
                    1, curInputIds, curAttentionMask, curPositionIds, curSeqLens,
                    curPastKeyValues, curGenerationConfigs, curLastTokens, curLogitsPtr
                );
                ret.push_back(curRet[0]);
            }
            return ret;
        };

        auto canRunFusedDecode = [&]() -> bool {
            if (!all1 || (int)pastKeyValues.size() < batch * block_cnt) {
                return false;
            }
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < block_cnt; i++) {
                    Data *pastKey = pastKeyValues[b * block_cnt + i].first;
                    Data *pastValue = pastKeyValues[b * block_cnt + i].second;
                    if (pastKey == nullptr || pastValue == nullptr ||
                        !pastKey->isPagedKVCache || !pastValue->isPagedKVCache ||
                        pastKey->pagedKVCacheData == nullptr || pastValue->pagedKVCacheData == nullptr ||
                        pastKey->pageIndex.empty() || pastValue->pageIndex.empty()) {
                        return false;
                    }
                    Data *cacheStorage = (Data*)pastKey->pagedKVCacheData;
                    if (cacheStorage->dataDevice != DataDevice::CUDA) {
                        return false;
                    }
                }
            }
            return true;
        };

        if (batch > 1 && !canRunFusedDecode()) {
            return runSplitBatchForward();
        }

        PrepareMoeWeights();

        AssertInFastLLM((int)positionIds.size() >= batch, "Step3p5 positionIds missing.");
        std::vector <Data> positionIdsCpu;
        positionIdsCpu.reserve(batch);
        for (int b = 0; b < batch; b++) {
            AssertInFastLLM(positionIds[b] != nullptr, "Step3p5 positionIds should not be null.");
            positionIdsCpu.emplace_back();
            positionIdsCpu.back().CopyFrom(*positionIds[b]);
            positionIdsCpu.back().ToDevice(DataDevice::CPU);
        }

        Data allPositionIds;
        std::vector <float> vPositionIds;
        if (all1) {
            for (int b = 0; b < batch; b++) {
                vPositionIds.push_back(Step3p5ReadFloat(positionIdsCpu[b], 0));
            }
        } else {
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < seqLens[b]; i++) {
                    vPositionIds.push_back(Step3p5ReadFloat(positionIdsCpu[b], i));
                }
            }
        }
        allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, totalLen}, vPositionIds));

        if (!initialized_add1) {
            for (int i = 0; i < block_cnt; i++) {
                Step3p5Add1(this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"]);
                Step3p5Add1(this->weight["model.layers." + std::to_string(i) + ".self_attn.q_norm.weight"]);
                Step3p5Add1(this->weight["model.layers." + std::to_string(i) + ".self_attn.k_norm.weight"]);
                Step3p5Add1(this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"]);
                std::string routerName = "model.layers." + std::to_string(i) + ".moe.gate.weight";
                if (need_fp32_gate && weight.weight.find(routerName) != weight.weight.end()) {
                    ToDataType(this->weight[routerName], DataType::FLOAT32);
                }
            }
            Step3p5Add1(this->weight["model.norm.weight"]);
            initialized_add1 = true;
        }

        Data hiddenStates;
        if (step3p7PrecomputedHiddenStates != nullptr) {
            hiddenStates.CopyFrom(*step3p7PrecomputedHiddenStates);
            DataType hiddenType = (this->dataType == DataType::FLOAT32 ||
                                   this->dataType == DataType::FLOAT16 ||
                                   this->dataType == DataType::BFLOAT16) ?
                                  this->dataType : DataType::FLOAT16;
            if (hiddenStates.dataType != hiddenType) {
                ToDataType(hiddenStates, hiddenType);
            }
        } else {
            EmbeddingBlock((Data*)&inputIds, &this->weight["model.embed_tokens.weight"], &hiddenStates, this->dataType);
        }

        Data attenInput, q, k, v, qkv, attenOutput, gate, mergedQkv;
        Data w1, w2, w3, routerLogits, routerProb, expertIndex, expertScore;
        Data attenPart, moePart, moeFinal, shareOutput;
        Data tempInput, tempOutput;
        std::vector<Data*> batchPastKeys(batch), batchPastValues(batch);
        Data qSizes, pageSizes, pageIndexs, lastPageLens, insertIndexs, insertPositions;
        bool generatedBatchDecodeParams = false;
        bool generatedAppendPagedCacheBatchParams = false;

        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            std::string prefix = "model.layers." + std::to_string(i) + ".";
            std::string inputRmsName = prefix + "input_layernorm.weight";
            std::string postRmsName = prefix + "post_attention_layernorm.weight";
            int qHeads = LayerAttentionHeads(i);
            int kvHeads = LayerKeyValueHeads(i);
            int qDim = qHeads * head_dim;
            int kvDim = kvHeads * head_dim;

            RMSNorm(hiddenStates, this->weight[inputRmsName], rms_norm_eps, attenInput);
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];

            std::string mergeQkvWeightName = prefix + "self_attn.mergeqkv.weight";
            bool hasMergedQkv = weight.weight.find(mergeQkvWeightName) != weight.weight.end();
            if (hasMergedQkv) {
                Linear(attenInput, weight[mergeQkvWeightName], Data(), mergedQkv);
            } else {
                Linear(attenInput, weight[prefix + "self_attn.q_proj.weight"], Data(), q);
                Linear(attenInput, weight[prefix + "self_attn.k_proj.weight"], Data(), k);
                Linear(attenInput, weight[prefix + "self_attn.v_proj.weight"], Data(), v);
            }
            Linear(attenInput, weight[prefix + "self_attn.g_proj.weight"], Data(), gate);

            bool usedFusedDecodeAttention = false;
            if (hasMergedQkv && canRunFusedDecode() &&
                mergedQkv.dataDevice == DataDevice::CUDA && !mergedQkv.multiDeviceData) {
                for (int b = 0; b < batch; b++) {
                    batchPastKeys[b] = pastKeyValues[b * block_cnt + i].first;
                    batchPastValues[b] = pastKeyValues[b * block_cnt + i].second;
                }

                Data &kCaches = *batchPastKeys[0];
                Data &vCaches = *batchPastValues[0];
                PagedCacheManager *pagedCacheKManager = kCaches.pagedKVCacheData;
                PagedCacheManager *pagedCacheVManager = vCaches.pagedKVCacheData;
                AssertInFastLLM(pagedCacheKManager != nullptr && pagedCacheVManager != nullptr,
                                "Step3p5 fused batch decode requires paged KV cache.");

                Data &pagedKData = *(Data*)pagedCacheKManager;
                Data &pagedVData = *(Data*)pagedCacheVManager;
                if (!generatedAppendPagedCacheBatchParams) {
                    GenerateAppendPagedCacheBatchParams(*pagedCacheKManager, batchPastKeys, batch,
                                                        insertIndexs, insertPositions);
                    generatedAppendPagedCacheBatchParams = true;
                }

                q.dataType = mergedQkv.dataType;
                q.UpdateUnitSize();
                q.Resize({bsz * qHeads, seqlen, head_dim});
                bool fillLastPageLensOnDevice =
                    !generatedBatchDecodeParams &&
                    mergedQkv.dataDevice == DataDevice::CUDA && !mergedQkv.multiDeviceData;

                Step3p5QKVRMSNormRopeSplitAppendPagedCache(
                    mergedQkv,
                    this->weight[prefix + "self_attn.q_norm.weight"],
                    this->weight[prefix + "self_attn.k_norm.weight"],
                    allPositionIds,
                    q,
                    pagedKData, pagedVData,
                    insertIndexs, insertPositions,
                    qHeads, kvHeads, head_dim,
                    i < (int)layer_rotary_dims.size() ? layer_rotary_dims[i] : head_dim,
                    rms_norm_eps,
                    i < (int)layer_rope_thetas.size() ? layer_rope_thetas[i] : rope_base,
                    UseLlama3Rope(i), rope_factor,
                    llama3_original_max_position_embeddings,
                    llama3_low_freq_factor,
                    llama3_high_freq_factor,
                    pagedKData.dims[1], batch,
                    fillLastPageLensOnDevice ? &lastPageLens : nullptr);

                for (int b = 0; b < batch; b++) {
                    auto updatePageMeta = [](Data *cache, PagedCacheManager *mgr) {
                        if (cache->pageIndex.empty() || cache->lastPageLen >= cache->pageLen) {
                            cache->pageIndex.push_back(mgr->GetUnusedPageIndex(true));
                            cache->lastPageLen = 1;
                        } else {
                            cache->lastPageLen++;
                        }
                    };
                    updatePageMeta(batchPastKeys[b], pagedCacheKManager);
                    updatePageMeta(batchPastValues[b], pagedCacheVManager);
                }

                Data qForAttentionHolder;
                Data &qForAttention = Step3p5PreparePagedAttentionQ(q, kCaches.dataType, this->dataType, qForAttentionHolder);
                if (!generatedBatchDecodeParams) {
                    GeneratePagedBatchParams(qForAttention, batchPastKeys, batch,
                                             qSizes, pageSizes, pageIndexs, lastPageLens,
                                             std::vector<int>(), fillLastPageLensOnDevice);
                    generatedBatchDecodeParams = true;
                }
                AttentionPagedBatch(qForAttention, kCaches, vCaches,
                                    qSizes, pageSizes, pageIndexs, lastPageLens,
                                    qkv, qForAttention.dims[0] / kCaches.dims[0],
                                    1.0f / sqrt((float)head_dim), 1, false);
                usedFusedDecodeAttention = true;
            }

            if (!usedFusedDecodeAttention) {
                if (hasMergedQkv) {
                    Split(mergedQkv, -1, 0, qDim, q);
                    Split(mergedQkv, -1, qDim, qDim + kvDim, k);
                    Split(mergedQkv, -1, qDim + kvDim, qDim + kvDim * 2, v);
                }

                q.Reshape({bsz, seqlen, qHeads, head_dim});
                k.Reshape({bsz, seqlen, kvHeads, head_dim});
                v.Reshape({bsz, seqlen, kvHeads, head_dim});
                RMSNorm(q, this->weight[prefix + "self_attn.q_norm.weight"], rms_norm_eps, q);
                RMSNorm(k, this->weight[prefix + "self_attn.k_norm.weight"], rms_norm_eps, k);
                ApplyStepRotary(q, allPositionIds, i);
                ApplyStepRotary(k, allPositionIds, i);

                PermuteSelf(q, {0, 2, 1, 3});
                PermuteSelf(k, {0, 2, 1, 3});
                PermuteSelf(v, {0, 2, 1, 3});
                q.Reshape({-1, seqlen, head_dim});
                k.Reshape({-1, seqlen, head_dim});
                v.Reshape({-1, seqlen, head_dim});

                if (batch > 1 && all1) {
                    for (int b = 0; b < batch; b++) {
                        batchPastKeys[b] = pastKeyValues[b * block_cnt + i].first;
                        batchPastValues[b] = pastKeyValues[b * block_cnt + i].second;
                    }

                    Data &kCaches = *batchPastKeys[0];
                    Data &vCaches = *batchPastValues[0];
                    PagedCacheManager *pagedCacheKManager = kCaches.pagedKVCacheData;
                    PagedCacheManager *pagedCacheVManager = vCaches.pagedKVCacheData;
                    AssertInFastLLM(pagedCacheKManager != nullptr && pagedCacheVManager != nullptr,
                                    "Step3p5 fused batch decode requires paged KV cache.");

                    if (!generatedAppendPagedCacheBatchParams) {
                        GenerateAppendPagedCacheBatchParams(*pagedCacheKManager, batchPastKeys, batch,
                                                            insertIndexs, insertPositions);
                        generatedAppendPagedCacheBatchParams = true;
                    }

                    Data kAppend, vAppend;
                    Permute(k, {1, 0, 2}, kAppend);
                    Permute(v, {1, 0, 2}, vAppend);
                    AppendPagedCacheBatch(*pagedCacheKManager, batchPastKeys, kAppend, insertIndexs, insertPositions);
                    AppendPagedCacheBatch(*pagedCacheVManager, batchPastValues, vAppend, insertIndexs, insertPositions);

                    Data qForAttentionHolder;
                    Data &qForAttention = Step3p5PreparePagedAttentionQ(q, kCaches.dataType, this->dataType, qForAttentionHolder);
                    if (!generatedBatchDecodeParams) {
                        GeneratePagedBatchParams(qForAttention, batchPastKeys, batch,
                                                 qSizes, pageSizes, pageIndexs, lastPageLens);
                        generatedBatchDecodeParams = true;
                    }
                    AttentionPagedBatch(qForAttention, kCaches, vCaches,
                                        qSizes, pageSizes, pageIndexs, lastPageLens,
                                        qkv, qForAttention.dims[0] / kCaches.dims[0],
                                        1.0f / sqrt((float)head_dim), 1, false);
                } else {
                    Data &pastKey = *pastKeyValues[i].first, &pastValue = *pastKeyValues[i].second;
                    PagedCacheManager *pagedCacheKManager = AllocatePagedCacheManager(
                        i * 2, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, k);
                    PagedCacheManager *pagedCacheVManager = AllocatePagedCacheManager(
                        i * 2 + 1, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, v);
                    AppendPagedCache(*pagedCacheKManager, pastKey, k);
                    AppendPagedCache(*pagedCacheVManager, pastValue, v);
                    AttentionPaged(q, pastKey, pastValue, qkv, q.dims[0] / k.dims[0],
                                   1.0f / sqrt((float)head_dim), 1, false);
                }
            }

            if (usedFusedDecodeAttention || (batch > 1 && all1)) {
                qkv.Reshape({seqlen, bsz, -1});
                PermuteSelf(qkv, {1, 0, 2});
            } else {
                PermuteSelf(qkv, {1, 0, 2});
                qkv.Reshape({seqlen, bsz, -1});
                PermuteSelf(qkv, {1, 0, 2});
            }

            Sigmoid(gate, gate);
            gate.Reshape({bsz, seqlen, qHeads, 1});
            qkv.Reshape({bsz, seqlen, qHeads, head_dim});
            if (gate.dataType != qkv.dataType) {
                ToDataType(gate, qkv.dataType);
            }
            MulTo(qkv, gate);
            qkv.Reshape({bsz, seqlen, qDim});

            Linear(qkv, weight[prefix + "self_attn.o_proj.weight"], Data(), attenInput);
            AddTo(hiddenStates, attenInput);

            RMSNorm(hiddenStates, this->weight[postRmsName], rms_norm_eps, attenInput);
            if (!IsMoeLayer(i)) {
                std::string gateupName = prefix + "mlp.gateup_proj.weight";
                std::string downName = prefix + "mlp.down_proj.weight";
                float denseLimit = Step3p5LayerLimit(swiglu_limits_shared, i);
                if (denseLimit == 0.0f && weight.weight.find(gateupName) != weight.weight.end()) {
                    MLPBlock(&attenInput, &weight[gateupName], &weight[downName], &w3, &w1, &hiddenStates);
                } else {
                    Linear(attenInput, weight[prefix + "mlp.gate_proj.weight"], Data(), w1);
                    Silu(w1, w1);
                    Linear(attenInput, weight[prefix + "mlp.up_proj.weight"], Data(), w3);
                    if (denseLimit != 0.0f) {
                        Step3p5Clamp(w1, false, 0.0f, true, denseLimit);
                        Step3p5Clamp(w3, true, -denseLimit, true, denseLimit);
                    }
                    MulTo(w1, w3);
                    Linear(w1, weight[downName], Data(), w2);
                    AddTo(hiddenStates, w2);
                }
            } else {
                std::string sharedGateupName = prefix + "share_expert.gateup_proj.weight";
                std::string sharedDownName = prefix + "share_expert.down_proj.weight";
                float sharedLimit = Step3p5LayerLimit(swiglu_limits_shared, i);
                float expertLimit = Step3p5LayerLimit(swiglu_limits, i);
                if (sharedLimit == 0.0f && weight.weight.find(sharedGateupName) != weight.weight.end()) {
                    Linear(attenInput, weight[sharedGateupName], Data(), w3);
                    Swiglu(w3, w1);
                } else {
                    Linear(attenInput, weight[prefix + "share_expert.gate_proj.weight"], Data(), w1);
                    Silu(w1, w1);
                    Linear(attenInput, weight[prefix + "share_expert.up_proj.weight"], Data(), w3);
                    if (sharedLimit != 0.0f) {
                        Step3p5Clamp(w1, false, 0.0f, true, sharedLimit);
                        Step3p5Clamp(w3, true, -sharedLimit, true, sharedLimit);
                    }
                    MulTo(w1, w3);
                }
                Linear(w1, weight[sharedDownName], Data(), shareOutput);

                int flatBatch = attenInput.dims[0];
                int flatLen = attenInput.dims[1];
                attenInput.Reshape({flatBatch * flatLen, attenInput.dims[2]});
                Linear(attenInput, weight[prefix + "moe.gate.weight"], Data(), routerLogits);
                ToDataType(routerLogits, DataType::FLOAT32);
                Sigmoid(routerLogits, routerProb);
                Data *gateBias = nullptr;
                if (use_moe_router_bias && weight.weight.find(prefix + "moe.router_bias") != weight.weight.end()) {
                    gateBias = &weight[prefix + "moe.router_bias"];
                }
                SelectExpert(routerProb, expertIndex, expertScore, num_experts_per_tok, norm_topk_prob,
                             routed_scaling_factor, gateBias);

                bool useCudaMoe = Step3p5DeviceMapUsesCuda(this->moeDeviceMap);
                bool useDiskMoe = Step3p5DeviceMapUsesDisk(this->moeDeviceMap);
                std::string selectedMoeDevice = this->SelectMoeDeviceForLayer(i);
                bool selectedCudaMoe = selectedMoeDevice.rfind("cuda", 0) == 0;
                bool useFusedCudaMoe = selectedCudaMoe && !useDiskMoe &&
                    i < (int)moeGate3DWeights.size() &&
                    moeGate3DWeights[i] != nullptr && moeUp3DWeights[i] != nullptr && moeDown3DWeights[i] != nullptr;
                if (useFusedCudaMoe) {
                    Data expertInput;
                    expertInput.CopyFrom(attenInput);
                    this->ApplyMoeDeviceMapForLayer(i);
                    Step3p5PrepareFusedMoeWeightForCuda(*moeGate3DWeights[i], moeGate3DWeights[i]->dims[1]);
                    Step3p5PrepareFusedMoeWeightForCuda(*moeUp3DWeights[i], moeUp3DWeights[i]->dims[1]);
                    Step3p5PrepareFusedMoeWeightForCuda(*moeDown3DWeights[i], moeDown3DWeights[i]->dims[1]);
                    FusedMOE(expertInput, expertIndex, expertScore,
                             *moeGate3DWeights[i], *moeUp3DWeights[i], *moeDown3DWeights[i],
                             w1, moeFinal, i, MoeGateSwiglu, expertLimit);
                    ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
                } else {
                    routerProb.ToDevice(DataDevice::CPU);
                    expertIndex.ToDevice(DataDevice::CPU);
                    expertScore.ToDevice(DataDevice::CPU);
                    ToDataType(expertScore, DataType::FLOAT32);
                    if (i < (int)weights.size() && !weights[i].empty() && (useCudaMoe || useDiskMoe)) {
                        Data expertInput;
                        expertInput.CopyFrom(attenInput);
                        this->ApplyMoeDeviceMapForLayer(i);
                        MergeMOE(expertInput, expertIndex, expertScore,
                                 weights[i], biass[i],
                                 w1, w2, w3, tempInput, tempOutput,
                                 1.0f, moeFinal, i);
                        ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
                    } else {
                        int32_t *indexData = (int32_t*)expertIndex.cpuData;
                        float *scoreData = (float*)expertScore.cpuData;

                        std::map<std::string, int> cpuDeviceMap = {{"cpu", 1}};
                        Data expertInput;
                        expertInput.CopyFrom(attenInput);
                        expertInput.ToDevice(DataDevice::CPU);
                        moeFinal.dataType = hiddenStates.dataType;
                        moeFinal.dataDevice = expertInput.dataDevice;
                        moeFinal.dataDeviceIds = expertInput.dataDeviceIds;
                        moeFinal.UpdateUnitSize();
                        moeFinal.Resize({0, expertInput.dims[1]});
                        moeFinal.Expansion(expertInput.dims);
                        ApplyDeviceMap(cpuDeviceMap, 1, 1);
                        for (int b = 0; b < flatBatch * flatLen; b++) {
                            Data *currentData = &expertInput;
                            if (flatBatch * flatLen != 1) {
                                Split(expertInput, 0, b, b + 1, attenPart);
                                currentData = &attenPart;
                            }
                            moePart.dataType = hiddenStates.dataType;
                            moePart.dataDevice = currentData->dataDevice;
                            moePart.dataDeviceIds = currentData->dataDeviceIds;
                            moePart.UpdateUnitSize();
                            moePart.Resize(currentData->dims);
                            moePart.Allocate(0.0f);
                            for (int j = 0; j < num_experts_per_tok; j++) {
                                int expert = indexData[b * num_experts_per_tok + j];
                                float score = scoreData[b * num_experts_per_tok + j];
                                Linear(*currentData, *moeGateWeights[i][expert], Data(), w1);
                                Silu(w1, w1);
                                Linear(*currentData, *moeUpWeights[i][expert], Data(), w3);
                                if (expertLimit != 0.0f) {
                                    Step3p5Clamp(w1, false, 0.0f, true, expertLimit);
                                    Step3p5Clamp(w3, true, -expertLimit, true, expertLimit);
                                }
                                MulTo(w1, w3);
                                Linear(w1, *moeDownWeights[i][expert], Data(), w2);
                                if (w2.dataType != moePart.dataType) {
                                    ToDataType(w2, moePart.dataType);
                                }
                                AddTo(moePart, w2, score);
                            }
                            if (moePart.dataType != moeFinal.dataType) {
                                ToDataType(moePart, moeFinal.dataType);
                            }
                            CatDirect(moeFinal, moePart, 0);
                        }
                        ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
                    }
                }
                moeFinal.expansionDims.clear();
                moeFinal.Reshape(hiddenStates.dims);
                moeFinal.ToDevice(hiddenStates.dataDevice);
                if (moeFinal.dataType != hiddenStates.dataType) {
                    ToDataType(moeFinal, hiddenStates.dataType);
                }
                if (shareOutput.dataType != hiddenStates.dataType) {
                    ToDataType(shareOutput, hiddenStates.dataType);
                }
                AddTo(hiddenStates, moeFinal);
                AddTo(hiddenStates, shareOutput);
            }
        }

        std::vector <int> lastRet;
        LLMSamplingBlock(
            this, &hiddenStates,
            &weight["model.norm.weight"], &weight["lm_head.weight"],
            rms_norm_eps, batch, all1, seqLens,
            pastKeyValues, generationConfigs, lastTokens,
            retLogits, lastRet
        );
        return lastRet;
    }

    bool Step3p5Model::NeedAttentionMask(int qlen, int klen) {
        (void)qlen;
        (void)klen;
        return false;
    }

    void Step3p5Model::Prepare() {
        if (!step3p7VisionAvailable || GetMaxTokens() > 0) {
            return;
        }
        printf("[Fastllm] Step3.7 vision: auto-calculate KV cache pages with %.0f%% GPU memory budget; remaining memory is reserved for image prefill headroom.\n",
               GetGpuMemRatio() * 100.0f);
    }

    void Step3p5Model::WarmUp() {
        printf("Warmup...\n");
        float oldGpuMemRatio = GetGpuMemRatio();
        if (step3p7VisionAvailable && oldGpuMemRatio > 0.85f) {
            SetGpuMemRatio(0.85f);
        }
        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {1});
        Data attentionMask = Data(this->dataType, {1, 1}, {0});
        Data positionIds = Data(DataType::FLOAT32, {1, 1}, {0});
        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(this->dataType), Data(this->dataType)));
            pastKeyValues.back().first.SetKVCache();
            pastKeyValues.back().second.SetKVCache();
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        if (GetGpuMemRatio() != oldGpuMemRatio) {
            SetGpuMemRatio(oldGpuMemRatio);
        }
        printf("finish.\n");
    }

    std::string Step3p5Model::ApplyChatTemplate(const ChatMessages &messages) {
        std::string prompt = STEP3P5_BOS;
        for (auto &message : messages) {
            const std::string &role = message.first;
            const std::string &content = message.second;
            if (role == "system" || role == "user" || role == "assistant") {
                prompt += STEP3P5_IM_START + role + "\n" + content + STEP3P5_IM_END + "\n";
            } else if (role == "tool") {
                prompt += STEP3P5_IM_START + std::string("tool_response\n<tool_response>") +
                          content + "</tool_response>" + STEP3P5_IM_END + "\n";
            }
        }
        prompt += STEP3P5_IM_START + std::string("assistant\n<think>\n");
        return prompt;
    }

    std::string Step3p5Model::MakeInput(const std::string &history, int round, const std::string &input) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
    }

    std::string Step3p5Model::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        return (round == 0 ? pre_prompt : history) + user_role + input +
               STEP3P5_IM_END + "\n" + STEP3P5_IM_START + "assistant\n" + output + history_sep;
    }
}
