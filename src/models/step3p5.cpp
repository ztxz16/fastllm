//
// Step-3.5 text model support.
//

#include "step3p5.h"
#include "utils.h"
#include "json11.hpp"
#ifdef USE_CUDA
#include "models/qwen3_cuda_common.h"
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <exception>
#include <map>
#include <mutex>
#include <sstream>
#include <thread>

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

        static bool Step3p5DebugSyncEnabled() {
            const char *env = std::getenv("FASTLLM_STEP3P5_DEBUG_SYNC");
            return env != nullptr && Step3p5IsTrueString(env);
        }

        static void Step3p5DebugSync(int gpuId, int layer, const char *tag) {
            if (!Step3p5DebugSyncEnabled()) {
                return;
            }
            if (const char *layerEnv = std::getenv("FASTLLM_STEP3P5_DEBUG_LAYER")) {
                if (layer != atoi(layerEnv)) {
                    return;
                }
            }
            printf("[Step3p5TP] dev=%d layer=%d %s\n", gpuId, layer, tag);
            fflush(stdout);
            ForceDeviceSync();
        }

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
            if (firstLocal == nullptr) {
                return;
            }

            std::vector<int> globalDims = {globalKVHeads, firstLocal->dims[1], headDim};
            root.dataType = firstLocal->dataType;
            root.UpdateUnitSize();
            root.dataDevice = DataDevice::CUDA;
            root.multiDeviceData = false;
            root.ClearTensorParallelLayout();
            root.Resize(globalDims);
            root.multiDeviceData = true;
            root.dataDeviceIds = devices;
            root.tpLayout = TP_LAYOUT_SHARDED;
            root.tpAxis = 0;
            root.tpRanges = kvHeadScheme;
            root.tpGlobalDims = globalDims;
            root.cudaData = nullptr;
            root.isKVCache = true;
            root.isPagedKVCache = firstLocal->isPagedKVCache;
            root.pageLen = firstLocal->pageLen;
            root.pageIndex = firstLocal->pageIndex;
            root.lastPageLen = firstLocal->lastPageLen;
            root.pagedKVCacheData = firstLocal->pagedKVCacheData;
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

        static void Step3p5CudaRepeat(Qwen3CudaDirectRunner &runner,
                                      const Data &input, int axis, int repeatTimes,
                                      Data &output) {
            runner.Run("Repeat",
                       DataDict{{"input", (Data*)&input}, {"output", &output}},
                       FloatDict(), IntDict{{"axis", axis}, {"repeatTimes", repeatTimes}},
                       {"output"});
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

        static void Step3p5CudaClamp(Data &input,
                                     bool hasMin, float minValue,
                                     bool hasMax, float maxValue,
                                     int device) {
            if (!hasMin && !hasMax) {
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
                Data *qkv, Data *q, Data *attenOutput,
                Data *qForAttentionHolder,
                Data *qSizes, Data *pageSizes, Data *pageIndexs, Data *lastPageLens,
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
                int pagedCacheLayerOffset) {
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
            int per = qkv->dims.back() / (numAttentionHeads / numKeyValueHeads + 2);
            int qdim = per * (numAttentionHeads / numKeyValueHeads);
            Data k, v;
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

    bool Step3p5Model::IsThreadTensorParallelEnabled() const {
#ifdef USE_CUDA
        std::vector<int> devices;
        std::map<int, int> ratios;
        return GetStep3p5ThreadTpDevices(this->deviceMap, devices, ratios);
#else
        return false;
#endif
    }

    Data &Step3p5Model::GetThreadTensorParallelBias(const std::string &name) {
        auto it = this->weight.weight.find(name);
        if (it != this->weight.weight.end()) {
            return it->second;
        }
        return this->threadTpEmptyBiases[name];
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
            Data &logits) {
#ifndef USE_CUDA
        ErrorInFastLLM("Step3p5 ForwardSingleGPU requires CUDA.\n");
#else
        (void)isPrefill;
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
        Data hiddenStates;
        Qwen3CudaEmbeddingDirect(cudaRunner,
                                 *requireLocal((Data&)inputIds, "inputIds"),
                                 *requireLocal(weight["model.embed_tokens.weight"], "model.embed_tokens.weight"),
                                 hiddenStates);
        if (hiddenStates.dataType != computeType) {
            Qwen3CudaToDataType(cudaRunner, hiddenStates, computeType);
        }

        Data attenInput, qkv, q, qForAttentionHolder, attenOutput, attenLastOutput;
        Data gate, gateRep;
        Data ffMiddle, ffAct, ffUp, ffOut;
        Data routerLogits, routerProb, expertIndex, expertScore;
        Data w1, w2, w3, tempInput, tempOutput, moeInputTemp, moeOutputTemp, moeFinal, shareOutput;
        Data qSizes, pageSizes, pageIndexs, lastPageLens;
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

            Step3p5DebugSync(gpuId, i, "layer_begin");
            Qwen3CudaRMSNorm(cudaRunner, hiddenStates,
                             *requireLocal(weight[inputRmsName], inputRmsName),
                             rms_norm_eps, attenInput);
            Step3p5DebugSync(gpuId, i, "after_input_norm");

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
                &qkv, &q, &attenOutput,
                &qForAttentionHolder,
                &qSizes, &pageSizes, &pageIndexs, &lastPageLens,
                batch, block_cnt, i,
                seqLens,
                localQHeads, localKVHeads, head_dim,
                curRotaryDim, rms_norm_eps,
                curTheta, rope_factor, UseLlama3Rope(i),
                llama3_original_max_position_embeddings,
                llama3_low_freq_factor,
                llama3_high_freq_factor,
                GetKVCacheInCPU(),
                pagedCacheLayerOffset
            );
            Step3p5DebugSync(gpuId, i, "after_attention");
            if (attenOutput.dataType != computeType) {
                Qwen3CudaToDataType(cudaRunner, attenOutput, computeType);
            }

            Qwen3CudaLinear(cudaRunner, attenInput,
                            *requireLocal(weight[gProjName], gProjName),
                            *GetEmptyData(), gate);
            Step3p5CudaSigmoid(cudaRunner, gate, gate);
            int bsz = attenInput.dims[0], seqlen = attenInput.dims[1];
            gate.Reshape({bsz, seqlen, localQHeads, 1});
            Step3p5CudaRepeat(cudaRunner, gate, 3, head_dim, gateRep);
            if (gateRep.dataType != attenOutput.dataType) {
                Qwen3CudaToDataType(cudaRunner, gateRep, attenOutput.dataType);
            }
            attenOutput.Reshape({bsz, seqlen, localQHeads, head_dim});
            Step3p5CudaMulTo(cudaRunner, attenOutput, gateRep);
            attenOutput.Reshape({bsz, seqlen, localQHeads * head_dim});
            Step3p5DebugSync(gpuId, i, "after_attention_gate");

            Qwen3CudaLinearResidualReduce(
                cudaRunner, attenOutput,
                *requireLocal(weight[oWeightName], oWeightName),
                *requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                attenLastOutput, hiddenStates,
                tensorParallel, firstTensorParallelRank, gpuId);
            Step3p5DebugSync(gpuId, i, "after_attention_reduce");

            Qwen3CudaRMSNorm(cudaRunner, hiddenStates,
                             *requireLocal(weight[postRmsName], postRmsName),
                             rms_norm_eps, attenInput);
            Step3p5DebugSync(gpuId, i, "after_post_norm");
            if (!IsMoeLayer(i)) {
                std::string gateupName = prefix + "mlp.gateup_proj.weight";
                std::string gateName = prefix + "mlp.gate_proj.weight";
                std::string upName = prefix + "mlp.up_proj.weight";
                std::string downName = prefix + "mlp.down_proj.weight";
                runFeedForwardOutput(attenInput, gateupName, gateName, upName, downName,
                                     Step3p5LayerLimit(swiglu_limits_shared, i),
                                     ffMiddle, ffAct, ffUp, ffOut);
                Step3p5DebugSync(gpuId, i, "after_dense_ffn");
                addPartialToResidualReduce(ffOut);
                Step3p5DebugSync(gpuId, i, "after_dense_ffn_reduce");
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
                Step3p5DebugSync(gpuId, i, "after_shared_expert");

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
                Step3p5DebugSync(gpuId, i, "after_router_select");

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
                        Step3p5DebugSync(gpuId, i, "after_expert_remap");
                        Step3p5CudaFusedMOE(cudaRunner, attenInput, expertIndex, expertScore,
                                            *localFusedWeights[0], *localFusedWeights[1],
                                            *localFusedWeights[2], w1, moeFinal, i,
                                            Step3p5LayerLimit(swiglu_limits, i));
                        Step3p5DebugSync(gpuId, i, "after_fused_moe");
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
                        Step3p5DebugSync(gpuId, i, "after_merge_moe");
                    } else {
                        Step3p5ZeroCudaLike(moeFinal, attenInput, gpuId);
                        Step3p5DebugSync(gpuId, i, "after_empty_moe");
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
                Step3p5DebugSync(gpuId, i, "after_moe_reduce");
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
            std::string selectedMoeDevice = SelectDeviceFromMap(this->moeDeviceMap, i + 1, block_cnt);
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
#ifdef USE_CUDA
        std::vector<int> devices;
        std::map<int, int> ratios;
        if (GetStep3p5GPUForwardDevices(this->deviceMap, devices, ratios)) {
            return ForwardGPU(1, inputIds, attentionMasks, positionIdsVec, seqLens,
                              pagedPastKeyValues, generationConfigs, lastTokens, &batchLogits)[0];
        }
#endif
        return ForwardV2(1, inputIds, attentionMasks, positionIdsVec, seqLens,
                         pagedPastKeyValues, generationConfigs, lastTokens, &batchLogits)[0];
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
        if (!GetStep3p5GPUForwardDevices(this->deviceMap, devices, ratios)) {
            return ForwardV2(batch, inputIds, attentionMask, positionIds, seqLens,
                             pastKeyValues, generationConfigs, lastTokens, retLogits);
        }
        bool tensorParallel = devices.size() > 1;
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

        std::vector<DivisionScheme> kvHeadSchemes;
        DivisionScheme lmHeadScheme;
        Data &lmHead = weight["lm_head.weight"];

        if (tensorParallel) {
            std::lock_guard<std::mutex> guard(threadTpWeightPrepareLock);
            ensureInitializedAdd1();
            PrepareMoeWeights();
            if (threadTpWeightsPrepared) {
                AssertInFastLLM(threadTpPreparedDevices == devices && threadTpPreparedRatios == ratios,
                                "Step3p5 ForwardGPU thread TP device config changed after weights were prepared.\n");
                AssertInFastLLM((int)threadTpKVHeadSchemes.size() == block_cnt &&
                                !threadTpLmHeadScheme.empty() &&
                                hasMoeCache(threadTpMoeWeights, threadTpMoeBiass),
                                "Step3p5 ForwardGPU thread TP cached weight schemes are incomplete.\n");
            } else {
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

                prepareReplicated("model.embed_tokens.weight");
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
                threadTpWeightsPrepared = true;
            }
            kvHeadSchemes = threadTpKVHeadSchemes;
            lmHeadScheme = threadTpLmHeadScheme;
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
            kvHeadSchemes.assign(block_cnt, DivisionScheme());
            for (int i = 0; i < block_cnt; i++) {
                kvHeadSchemes[i][devices[0]].push_back({0, LayerKeyValueHeads(i)});
            }
            lmHeadScheme[devices[0]].push_back({0, lmHead.dims[0]});
        }

        const DataType computeType = ResolveStep3p5ThreadTpComputeType(this->dataType);
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
            ForwardSingleGPU(devices[0], ratios, batch, gpuInputIds, allPositionIds,
                             seqLens, pastKeyValues, all1, isPrefill,
                             false, true, threadTpPagedCacheBase, localLogits[0]);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(devices.size());
            for (int r = 0; r < (int)devices.size(); r++) {
                threads.emplace_back([&, r]() {
                    try {
                        ForwardSingleGPU(devices[r], ratios, batch, gpuInputIds, allPositionIds,
                                         seqLens, localPastKeyValues[r], all1, isPrefill,
                                         tensorParallel, r == 0,
                                         threadTpPagedCacheBase + r * block_cnt,
                                         localLogits[r]);
                    } catch (...) {
                        errors[r] = std::current_exception();
                    }
                });
            }
            for (auto &thread : threads) {
                thread.join();
            }
            for (auto &error : errors) {
                if (error) {
                    std::rethrow_exception(error);
                }
            }
            for (int device : devices) {
                FastllmCudaSetDevice(device);
                ForceDeviceSync();
            }
        }

        if (tensorParallel) {
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < block_cnt; i++) {
                    int idx = b * block_cnt + i;
                    SyncStep3p5ThreadTpRootCacheMeta(*pastKeyValues[idx].first, devices, kvHeadSchemes[i],
                                                     LayerKeyValueHeads(i), head_dim);
                    SyncStep3p5ThreadTpRootCacheMeta(*pastKeyValues[idx].second, devices, kvHeadSchemes[i],
                                                     LayerKeyValueHeads(i), head_dim);
                }
            }
        }

        int vocabSize = lmHead.dims[0];
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
            for (auto &range : lmHeadScheme[device]) {
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
        const char *printLogitsEnv = std::getenv("FASTLLM_PRINT_LOGITS");
        if (GetFastllmEnv().printLogits ||
            (printLogitsEnv != nullptr && printLogitsEnv[0] != '\0' &&
             !(printLogitsEnv[0] == '0' && printLogitsEnv[1] == '\0'))) {
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

        auto canRunFusedBatchDecode = [&]() -> bool {
            if (batch <= 1 || !all1 || (int)pastKeyValues.size() < batch * block_cnt) {
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

        if (batch > 1 && !canRunFusedBatchDecode()) {
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
        EmbeddingBlock((Data*)&inputIds, &this->weight["model.embed_tokens.weight"], &hiddenStates, this->dataType);

        Data attenInput, q, k, v, qkv, attenOutput, gate, gateRep, mergedQkv;
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
            if (weight.weight.find(mergeQkvWeightName) != weight.weight.end()) {
                Linear(attenInput, weight[mergeQkvWeightName], Data(), mergedQkv);
                Split(mergedQkv, -1, 0, qDim, q);
                Split(mergedQkv, -1, qDim, qDim + kvDim, k);
                Split(mergedQkv, -1, qDim + kvDim, qDim + kvDim * 2, v);
            } else {
                Linear(attenInput, weight[prefix + "self_attn.q_proj.weight"], Data(), q);
                Linear(attenInput, weight[prefix + "self_attn.k_proj.weight"], Data(), k);
                Linear(attenInput, weight[prefix + "self_attn.v_proj.weight"], Data(), v);
            }
            Linear(attenInput, weight[prefix + "self_attn.g_proj.weight"], Data(), gate);

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

            if (batch > 1 && all1) {
                qkv.Reshape({seqlen, bsz, -1});
                PermuteSelf(qkv, {1, 0, 2});
            } else {
                PermuteSelf(qkv, {1, 0, 2});
                qkv.Reshape({seqlen, bsz, -1});
                PermuteSelf(qkv, {1, 0, 2});
            }

            Sigmoid(gate, gate);
            gate.Reshape({bsz, seqlen, qHeads, 1});
            Repeat(gate, 3, head_dim, gateRep);
            qkv.Reshape({bsz, seqlen, qHeads, head_dim});
            if (gateRep.dataType != qkv.dataType) {
                ToDataType(gateRep, qkv.dataType);
            }
            MulTo(qkv, gateRep);
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
                std::string selectedMoeDevice = SelectDeviceFromMap(this->moeDeviceMap, i + 1, block_cnt);
                bool selectedCudaMoe = selectedMoeDevice.rfind("cuda", 0) == 0;
                bool useFusedCudaMoe = selectedCudaMoe && !useDiskMoe &&
                    i < (int)moeGate3DWeights.size() &&
                    moeGate3DWeights[i] != nullptr && moeUp3DWeights[i] != nullptr && moeDown3DWeights[i] != nullptr;
                if (useFusedCudaMoe) {
                    Data expertInput;
                    expertInput.CopyFrom(attenInput);
                    ApplyDeviceMap(this->moeDeviceMap, i + 1, block_cnt);
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
                        ApplyDeviceMap(this->moeDeviceMap, i + 1, block_cnt);
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

    void Step3p5Model::WarmUp() {
        printf("Warmup...\n");
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
