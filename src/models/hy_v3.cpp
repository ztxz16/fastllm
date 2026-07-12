//
// Created by huangyuyang on 4/29/25.
//

#include "utils.h"

#include "hy_v3.h"
#include "blocks/baseblock.h"
#include "executor.h"

#include <sstream>

#include <unordered_map>

#include <cstring>
#include <climits>
#include <atomic>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <condition_variable>
#include <set>
#include <tuple>

#ifdef USE_CUDA
#include "models/qwen3_cuda_common.h"
#endif

namespace fastllm {
    extern std::vector <float> GetInterLeavePowerOf2(int n);
    extern std::vector <float> GetInterleave(int n);

    static bool HyV3IsTrueString(const std::string &value) {
        std::string lowered = value;
        std::transform(lowered.begin(), lowered.end(), lowered.begin(),
                       [](unsigned char c) { return (char)std::tolower(c); });
        return lowered == "1" || lowered == "true" || lowered == "on";
    }

    static std::string HyV3GetDict(const std::map<std::string, std::string> &dict,
                                       const std::string &key,
                                       const std::string &fallback = "") {
        auto it = dict.find(key);
        return it == dict.end() ? fallback : it->second;
    }

    static int HyV3GetInt(const std::map<std::string, std::string> &dict,
                              const std::string &key, int fallback) {
        auto it = dict.find(key);
        return it == dict.end() ? fallback : atoi(it->second.c_str());
    }

    static float HyV3GetFloat(const std::map<std::string, std::string> &dict,
                                  const std::string &key, float fallback) {
        auto it = dict.find(key);
        return it == dict.end() ? fallback : (float)atof(it->second.c_str());
    }

    static bool HyV3ParseLayerIndex(const std::string &name, int &layer) {
        const std::string prefix = "model.layers.";
        if (!StartWith(name, prefix)) {
            return false;
        }
        size_t pos = prefix.size();
        if (pos >= name.size() || !std::isdigit((unsigned char)name[pos])) {
            return false;
        }
        layer = 0;
        while (pos < name.size() && std::isdigit((unsigned char)name[pos])) {
            layer = layer * 10 + (name[pos] - '0');
            pos++;
        }
        return pos < name.size() && name[pos] == '.';
    }

    static bool HyV3DisableFusedMoe() {
        const char *env = std::getenv("FASTLLM_DISABLE_FUSED_MOE");
        return env != nullptr && HyV3IsTrueString(env);
    }

    static std::string HyV3ExpertPrefix(int layer, int expert) {
        return "model.layers." + std::to_string(layer) + ".mlp.experts." +
               std::to_string(expert) + ".";
    }

    static std::string HyV3FusedWeightName(int layer, const std::string &kind) {
        return "model.layers." + std::to_string(layer) + ".mlp.fused_experts." +
               kind + "_proj.weight";
    }

    static std::string HyV3ExpertWeightName(int layer, int expert, const std::string &kind) {
        return HyV3ExpertPrefix(layer, expert) + kind + "_proj.weight";
    }

    static bool HyV3ParseExpertWeightName(const std::string &name,
                                              int &layer, int &expert, std::string &kind) {
        const std::string prefix = "model.layers.";
        if (!StartWith(name, prefix)) {
            return false;
        }
        size_t pos = prefix.size();
        if (pos >= name.size() || !std::isdigit((unsigned char)name[pos])) {
            return false;
        }
        layer = 0;
        while (pos < name.size() && std::isdigit((unsigned char)name[pos])) {
            layer = layer * 10 + (name[pos] - '0');
            pos++;
        }
        const std::string mid = ".mlp.experts.";
        if (name.compare(pos, mid.size(), mid) != 0) {
            return false;
        }
        pos += mid.size();
        if (pos >= name.size() || !std::isdigit((unsigned char)name[pos])) {
            return false;
        }
        expert = 0;
        while (pos < name.size() && std::isdigit((unsigned char)name[pos])) {
            expert = expert * 10 + (name[pos] - '0');
            pos++;
        }
        const std::string suffix = "_proj.weight";
        if (name.compare(pos, 1, ".") != 0 || name.size() <= pos + 1 + suffix.size() ||
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) {
            return false;
        }
        kind = name.substr(pos + 1, name.size() - pos - 1 - suffix.size());
        return kind == "gate" || kind == "up" || kind == "gateup" || kind == "down";
    }

    static int HyV3SourceLoadPriority(const std::string &name, int numExperts) {
        int layer = -1, expert = -1;
        std::string kind;
        if (!HyV3ParseExpertWeightName(name, layer, expert, kind)) {
            return 0;
        }
        (void)expert;
        (void)numExperts;
        int layerStride = 3;
        int order = (kind == "down") ? 2 : (kind == "up" ? 1 : 0);
        return -100000000 + layer * layerStride + order;
    }

    static bool HyV3IsFusedFp8Type(DataType dataType) {
        return dataType == DataType::FP8_E4M3 ||
               dataType == DataType::FP8_E4M3_BLOCK_128;
    }

    static void HyV3CopyLinearWeightMeta(Data &dst, const Data &src, const std::string &name) {
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

    static size_t HyV3BytesPerRow(const Data &weight, int columns) {
        return GetDataBytes(weight.dataType, 1, columns);
    }

    static bool HyV3CheckFp8ScaleRows(const Data &weight, int rowStart, int rows) {
        if (weight.dataType != DataType::FP8_E4M3) {
            return true;
        }
        if (weight.blockK <= 0 || weight.blockM <= 0 || weight.scales.empty() ||
            weight.dims.size() != 2) {
            return false;
        }
        int cols = weight.dims[1];
        int totalRows = weight.dims[0];
        int ms = (cols - 1) / weight.blockM + 1;
        int scaleRows = (totalRows - 1) / weight.blockK + 1;
        int scaleOffset = (rowStart / weight.blockK) * ms;
        int scaleCount = ((rows - 1) / weight.blockK + 1) * ms;
        return rowStart >= 0 && rows > 0 && rowStart + rows <= totalRows &&
               rowStart % weight.blockK == 0 &&
               scaleOffset + scaleCount <= (int)weight.scales.size() &&
               scaleRows * ms <= (int)weight.scales.size();
    }

    static void HyV3AppendFp8ScaleRows(Data &dst, const Data &src, int rowStart, int rows) {
        if (src.dataType != DataType::FP8_E4M3) {
            return;
        }
        AssertInFastLLM(HyV3CheckFp8ScaleRows(src, rowStart, rows),
                        "HY-V3 FusedMOE FP8 scale slice is out of bounds.");
        int cols = src.dims[1];
        int ms = (cols - 1) / src.blockM + 1;
        int scaleOffset = (rowStart / src.blockK) * ms;
        int scaleCount = ((rows - 1) / src.blockK + 1) * ms;
        dst.scales.insert(dst.scales.end(),
                          src.scales.begin() + scaleOffset,
                          src.scales.begin() + scaleOffset + scaleCount);
    }

    static void HyV3CopyRows(Data &dst, int dstRowStart,
                                 Data &src, int srcRowStart, int rows) {
        AssertInFastLLM(dst.dims.size() == 3 && src.dims.size() == 2,
                        "HY-V3 FusedMOE row copy expects 3D destination and 2D source.");
        int cols = src.dims[1];
        AssertInFastLLM(dst.dims[2] == cols &&
                        srcRowStart >= 0 && rows > 0 && srcRowStart + rows <= src.dims[0],
                        "HY-V3 FusedMOE row copy shape mismatch.");
        int dstRows = dst.dims[0] * dst.dims[1];
        AssertInFastLLM(dstRowStart >= 0 && dstRowStart + rows <= dstRows,
                        "HY-V3 FusedMOE destination row range is out of bounds.");
        src.ToDevice(DataDevice::CPU);
        AssertInFastLLM(src.cpuData != nullptr && dst.cpuData != nullptr,
                        "HY-V3 FusedMOE row copy requires CPU buffers.");
        size_t bytesPerRow = HyV3BytesPerRow(src, cols);
        memcpy(dst.cpuData + (size_t)dstRowStart * bytesPerRow,
               src.cpuData + (size_t)srcRowStart * bytesPerRow,
               (size_t)rows * bytesPerRow);
    }

    static bool HyV3CanBuildFusedLayer(const std::unordered_map<std::string, Data> &allWeights,
                                           int layer, int numExperts) {
        if (numExperts <= 0) {
            return false;
        }
        const Data *gateup0 = nullptr;
        const Data *down0 = nullptr;
        int inter = 0, hidden = 0;
        for (int expert = 0; expert < numExperts; expert++) {
            std::string prefix = HyV3ExpertPrefix(layer, expert);
            auto gateupIt = allWeights.find(prefix + "gateup_proj.weight");
            auto downIt = allWeights.find(prefix + "down_proj.weight");
            if (gateupIt == allWeights.end() || downIt == allWeights.end()) {
                return false;
            }
            const Data &gateup = gateupIt->second;
            const Data &down = downIt->second;
            if (gateup.isDiskWeight || down.isDiskWeight ||
                gateup.cpuData == nullptr || down.cpuData == nullptr ||
                gateup.dims.size() != 2 || down.dims.size() != 2 ||
                gateup.dims[0] <= 0 || gateup.dims[1] <= 0 ||
                (gateup.dims[0] & 1) != 0 ||
                !HyV3IsFusedFp8Type(gateup.dataType) ||
                gateup.dataType != down.dataType) {
                return false;
            }
            int curInter = gateup.dims[0] / 2;
            int curHidden = gateup.dims[1];
            if (down.dims[0] != curHidden || down.dims[1] != curInter) {
                return false;
            }
            if (gateup.dataType == DataType::FP8_E4M3 &&
                (!HyV3CheckFp8ScaleRows(gateup, 0, curInter) ||
                 !HyV3CheckFp8ScaleRows(gateup, curInter, curInter) ||
                 !HyV3CheckFp8ScaleRows(down, 0, curHidden))) {
                return false;
            }
            if (expert == 0) {
                gateup0 = &gateup;
                down0 = &down;
                inter = curInter;
                hidden = curHidden;
            } else if (gateup.dataType != gateup0->dataType ||
                       gateup.dims != gateup0->dims ||
                       gateup.blockK != gateup0->blockK ||
                       gateup.blockM != gateup0->blockM ||
                       down.dataType != down0->dataType ||
                       down.dims != down0->dims ||
                       down.blockK != down0->blockK ||
                       down.blockM != down0->blockM ||
                       curInter != inter || curHidden != hidden) {
                return false;
            }
        }
        return true;
    }

    static bool HyV3CanBuildAllFusedWeights(const std::unordered_map<std::string, Data> &allWeights,
                                                int blockCnt, int numExperts) {
        for (int layer = 0; layer < blockCnt; layer++) {
            if (!HyV3CanBuildFusedLayer(allWeights, layer, numExperts)) {
                return false;
            }
        }
        return true;
    }

    static void HyV3BuildFusedLayer(std::unordered_map<std::string, Data> &allWeights,
                                        int layer, int numExperts,
                                        Data *&gatePtr, Data *&upPtr, Data *&downPtr) {
        std::string prefix0 = HyV3ExpertPrefix(layer, 0);
        Data &gateup0 = allWeights[prefix0 + "gateup_proj.weight"];
        Data &down0 = allWeights[prefix0 + "down_proj.weight"];
        int inter = gateup0.dims[0] / 2;
        int hidden = gateup0.dims[1];
        DataType gateupType = gateup0.dataType;
        DataType downType = down0.dataType;

        std::string gate3DName = HyV3FusedWeightName(layer, "gate");
        std::string up3DName = HyV3FusedWeightName(layer, "up");
        std::string down3DName = HyV3FusedWeightName(layer, "down");
        allWeights[gate3DName] = Data(gateupType, {numExperts, inter, hidden});
        allWeights[up3DName] = Data(gateupType, {numExperts, inter, hidden});
        allWeights[down3DName] = Data(downType, {numExperts, hidden, inter});

        Data &gate3D = allWeights[gate3DName];
        Data &up3D = allWeights[up3DName];
        Data &down3D = allWeights[down3DName];
        Data &gateupMeta = allWeights[prefix0 + "gateup_proj.weight"];
        Data &downMeta = allWeights[prefix0 + "down_proj.weight"];
        HyV3CopyLinearWeightMeta(gate3D, gateupMeta, gate3DName);
        HyV3CopyLinearWeightMeta(up3D, gateupMeta, up3DName);
        HyV3CopyLinearWeightMeta(down3D, downMeta, down3DName);
        gate3D.Allocate(false);
        up3D.Allocate(false);
        down3D.Allocate(false);
        gate3D.scales.clear();
        up3D.scales.clear();
        down3D.scales.clear();

        for (int expert = 0; expert < numExperts; expert++) {
            std::string expertPrefix = HyV3ExpertPrefix(layer, expert);
            Data &gateup = allWeights[expertPrefix + "gateup_proj.weight"];
            Data &down = allWeights[expertPrefix + "down_proj.weight"];
            HyV3CopyRows(gate3D, expert * inter, gateup, 0, inter);
            HyV3CopyRows(up3D, expert * inter, gateup, inter, inter);
            HyV3CopyRows(down3D, expert * hidden, down, 0, hidden);
            HyV3AppendFp8ScaleRows(gate3D, gateup, 0, inter);
            HyV3AppendFp8ScaleRows(up3D, gateup, inter, inter);
            HyV3AppendFp8ScaleRows(down3D, down, 0, hidden);
        }

        gatePtr = &gate3D;
        upPtr = &up3D;
        downPtr = &down3D;
    }

    static void HyV3BuildFusedLayerWeight(std::unordered_map<std::string, Data> &allWeights,
                                              int layer, int numExperts, const std::string &kind,
                                              Data *&weightPtr) {
        std::string prefix0 = HyV3ExpertPrefix(layer, 0);
        Data &gateup0 = allWeights[prefix0 + "gateup_proj.weight"];
        Data &down0 = allWeights[prefix0 + "down_proj.weight"];
        int inter = gateup0.dims[0] / 2;
        int hidden = gateup0.dims[1];
        bool isDown = kind == "down";
        AssertInFastLLM(kind == "gate" || kind == "up" || kind == "down",
                        "HY-V3 fused layer weight kind is invalid.\n");

        std::string fusedName = HyV3FusedWeightName(layer, kind);
        if (isDown) {
            allWeights[fusedName] = Data(down0.dataType, {numExperts, hidden, inter});
        } else {
            allWeights[fusedName] = Data(gateup0.dataType, {numExperts, inter, hidden});
        }

        Data &fused = allWeights[fusedName];
        HyV3CopyLinearWeightMeta(fused, isDown ? down0 : gateup0, fusedName);
        fused.Allocate(false);
        fused.scales.clear();

        for (int expert = 0; expert < numExperts; expert++) {
            std::string expertPrefix = HyV3ExpertPrefix(layer, expert);
            Data &gateup = allWeights[expertPrefix + "gateup_proj.weight"];
            Data &down = allWeights[expertPrefix + "down_proj.weight"];
            if (kind == "gate") {
                HyV3CopyRows(fused, expert * inter, gateup, 0, inter);
                HyV3AppendFp8ScaleRows(fused, gateup, 0, inter);
            } else if (kind == "up") {
                HyV3CopyRows(fused, expert * inter, gateup, inter, inter);
                HyV3AppendFp8ScaleRows(fused, gateup, inter, inter);
            } else {
                HyV3CopyRows(fused, expert * hidden, down, 0, hidden);
                HyV3AppendFp8ScaleRows(fused, down, 0, hidden);
            }
        }
        weightPtr = &fused;
    }

    static void HyV3ResizeFusedFp8Scales(Data &weight) {
        if (weight.dataType != DataType::FP8_E4M3) {
            return;
        }
        AssertInFastLLM(weight.dims.size() == 3 && weight.blockK > 0 && weight.blockM > 0,
                        "HY-V3 fused FP8 scale allocation got invalid metadata.\n");
        int experts = weight.dims[0];
        int rowsPerExpert = weight.dims[1];
        int cols = weight.dims[2];
        int scaleRowsPerExpert = (rowsPerExpert - 1) / weight.blockK + 1;
        int scaleCols = (cols - 1) / weight.blockM + 1;
        size_t scaleCount = (size_t)experts * scaleRowsPerExpert * scaleCols;
        if (weight.scales.size() != scaleCount) {
            weight.scales.assign(scaleCount, 0.0f);
        }
    }

    static void HyV3CopyFp8ScaleRowsToExpert(Data &dst, const Data &src,
                                                 int expert, int srcRowStart, int rows) {
        if (src.dataType != DataType::FP8_E4M3) {
            return;
        }
        AssertInFastLLM(dst.dataType == DataType::FP8_E4M3 &&
                        dst.dims.size() == 3 && src.dims.size() == 2 &&
                        dst.blockK == src.blockK && dst.blockM == src.blockM &&
                        expert >= 0 && expert < dst.dims[0],
                        "HY-V3 fused FP8 scale copy got incompatible metadata.\n");
        AssertInFastLLM(HyV3CheckFp8ScaleRows(src, srcRowStart, rows),
                        "HY-V3 fused FP8 scale source is not ready.\n");
        int cols = src.dims[1];
        int dstRowsPerExpert = dst.dims[1];
        int scaleCols = (cols - 1) / src.blockM + 1;
        int srcScaleOffset = (srcRowStart / src.blockK) * scaleCols;
        int scaleRowCount = (rows - 1) / src.blockK + 1;
        int dstScaleRowsPerExpert = (dstRowsPerExpert - 1) / dst.blockK + 1;
        size_t dstOffset = ((size_t)expert * dstScaleRowsPerExpert) * scaleCols;
        size_t count = (size_t)scaleRowCount * scaleCols;
        AssertInFastLLM(dstOffset + count <= dst.scales.size() &&
                        srcScaleOffset + count <= src.scales.size(),
                        "HY-V3 fused FP8 scale copy is out of bounds.\n");
        memcpy(dst.scales.data() + dstOffset,
               src.scales.data() + srcScaleOffset,
               count * sizeof(float));
    }

    static void HyV3InitFusedLayerWeightMeta(std::unordered_map<std::string, Data> &allWeights,
                                                 int layer, int numExperts, const std::string &kind,
                                                 const Data &source, int rowsPerExpert, int columns,
                                                 Data *&weightPtr) {
        std::string fusedName = HyV3FusedWeightName(layer, kind);
        allWeights[fusedName] = Data(source.dataType, {numExperts, rowsPerExpert, columns});
        Data &fused = allWeights[fusedName];
        HyV3CopyLinearWeightMeta(fused, source, fusedName);
        weightPtr = &fused;
    }

    static void HyV3AllocateFusedWeightForLoad(Data *weight) {
        if (weight == nullptr || weight->cpuData != nullptr ||
            weight->multiDeviceData || weight->dataDevice != DataDevice::CPU) {
            return;
        }
        weight->Allocate(false);
    }

    static void HyV3EnsureFusedLayerWeight(std::unordered_map<std::string, Data> &allWeights,
                                               int layer, int numExperts, const std::string &kind,
                                               const Data &source, int rowsPerExpert, int columns,
                                               Data *&weightPtr) {
        if (weightPtr == nullptr) {
            HyV3InitFusedLayerWeightMeta(allWeights, layer, numExperts, kind,
                                             source, rowsPerExpert, columns, weightPtr);
        } else {
            HyV3CopyLinearWeightMeta(*weightPtr, source, weightPtr->name);
            AssertInFastLLM(weightPtr->dims.size() == 3 &&
                            weightPtr->dims[0] == numExperts &&
                            weightPtr->dims[1] == rowsPerExpert &&
                            weightPtr->dims[2] == columns &&
                            weightPtr->dataType == source.dataType,
                            "HY-V3 fused weight metadata does not match source weight.\n");
        }
        HyV3AllocateFusedWeightForLoad(weightPtr);
        HyV3ResizeFusedFp8Scales(*weightPtr);
    }

    static void HyV3ReleaseConsumedSourceWeight(Data &weight) {
        weight.FreeSpace();
        weight.scales.clear();
        weight.scales.shrink_to_fit();
        weight.mins.clear();
        weight.mins.shrink_to_fit();
        weight.zeros.clear();
        weight.zeros.shrink_to_fit();
        weight.halfScales.clear();
        weight.halfScales.shrink_to_fit();
        weight.perChannelsConfigs.clear();
        weight.perChannelsConfigs.shrink_to_fit();
        weight.weightSum.clear();
        weight.weightSum.shrink_to_fit();
    }

#ifdef USE_CUDA
    namespace {
        using namespace qwen3cuda;

        static std::atomic<int> qwen3MoeThreadTpNextPagedCacheBase(3000000);

        static std::string HyV3TrimString(const std::string &s) {
            int l = 0, r = (int)s.size();
            while (l < r && std::isspace((unsigned char)s[l])) {
                l++;
            }
            while (r > l && std::isspace((unsigned char)s[r - 1])) {
                r--;
            }
            return s.substr(l, r - l);
        }

        static bool HyV3IsDisabledTpSpec(const std::string &value) {
            std::string v = value;
            std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return (char)std::tolower(c); });
            return v.empty() || v == "false" || v == "off" || v == "none" || v == "disable";
        }

        static bool HyV3UseCudappSerialDeviceMap() {
            const char *env = std::getenv("FASTLLM_CUDAPP_SERIAL");
            return env != nullptr && HyV3IsTrueString(env);
        }

        static bool AppendHyV3CudaDevicesFromSpec(const std::string &spec,
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

        static bool ParseHyV3GPUForwardSpec(const std::string &rawSpec,
                                                std::vector<int> &devices,
                                                std::map<int, int> &ratios) {
            std::string spec = HyV3TrimString(rawSpec);
            if (HyV3IsDisabledTpSpec(spec)) {
                return false;
            }

            std::string lower = spec;
            std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return (char)std::tolower(c); });
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
            return AppendHyV3CudaDevicesFromSpec(parseSpec, type, 1, devices, ratios);
        }

        static bool GetHyV3GPUForwardDevices(const std::map<std::string, int> &deviceMap,
                                                 std::vector<int> &devices,
                                                 std::map<int, int> &ratios) {
            devices.clear();
            ratios.clear();
            const char *env = std::getenv("FASTLLM_TP");
            if (env != nullptr) {
                ParseHyV3GPUForwardSpec(env, devices, ratios);
            }

            if (devices.empty() && HyV3UseCudappSerialDeviceMap()) {
                return false;
            }

            if (devices.empty()) {
                for (auto &it : deviceMap) {
                    std::string lower = it.first;
                    std::transform(lower.begin(), lower.end(), lower.begin(),
                                   [](unsigned char c) { return (char)std::tolower(c); });
                    if (StartWith(lower, "multicuda")) {
                        AppendHyV3CudaDevicesFromSpec(it.first, "multicuda", it.second, devices, ratios);
                    }
                }
            }

            if (devices.empty()) {
                for (auto &it : deviceMap) {
                    std::string lower = it.first;
                    std::transform(lower.begin(), lower.end(), lower.begin(),
                                   [](unsigned char c) { return (char)std::tolower(c); });
                    if (StartWith(lower, "cuda")) {
                        AppendHyV3CudaDevicesFromSpec(it.first, "cuda", it.second, devices, ratios);
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

        static bool HyV3DeviceSpecStartsWith(const std::string &device, const std::string &prefix) {
            std::string lower = device;
            std::transform(lower.begin(), lower.end(), lower.begin(),
                           [](unsigned char c) { return (char)std::tolower(c); });
            return lower == prefix || lower.rfind(prefix + ":", 0) == 0;
        }

        static bool HyV3DeviceSpecIsCuda(const std::string &device) {
            return HyV3DeviceSpecStartsWith(device, "cuda") ||
                   HyV3DeviceSpecStartsWith(device, "multicuda");
        }

        static bool HyV3DeviceMapAllCuda(const std::map<std::string, int> &deviceMap) {
            bool hasCuda = false;
            for (auto &it : deviceMap) {
                if (it.second <= 0) {
                    continue;
                }
                if (!HyV3DeviceSpecIsCuda(it.first)) {
                    return false;
                }
                hasCuda = true;
            }
            return hasCuda;
        }

        static bool HyV3MoeDeviceMapAllowsCudaOnly(const std::map<std::string, int> &moeDeviceMap) {
            return moeDeviceMap.empty() || HyV3DeviceMapAllCuda(moeDeviceMap);
        }

        static bool HyV3SelectedDeviceIsCudaOrEmpty(const std::string &device) {
            return device.empty() || HyV3DeviceSpecIsCuda(device);
        }

        static bool HyV3LayerUsesMappedNonCudaMoe(const HyV3Model *model, int layer) {
            return model != nullptr &&
                   !HyV3SelectedDeviceIsCudaOrEmpty(model->SelectMoeDeviceForLayer(layer));
        }

        static bool HyV3ModelMoeLayersAllowCudaOnly(const HyV3Model *model) {
            if (model == nullptr) {
                return true;
            }
            for (int i = 0; i < model->block_cnt; i++) {
                if (HyV3LayerUsesMappedNonCudaMoe(model, i)) {
                    return false;
                }
            }
            return true;
        }

        static bool HyV3CanUseGPUForward(const std::map<std::string, int> &deviceMap,
                                             const std::map<std::string, int> &moeDeviceMap) {
            (void)moeDeviceMap;
            std::vector<int> devices;
            std::map<int, int> ratios;
            return GetHyV3GPUForwardDevices(deviceMap, devices, ratios);
        }

        static bool HyV3CanPlanFusedMoe(const std::map<std::string, int> &deviceMap,
                                            const std::map<std::string, int> &moeDeviceMap) {
#ifdef USE_CUDA
            return HyV3CanUseGPUForward(deviceMap, moeDeviceMap) &&
                   HyV3MoeDeviceMapAllowsCudaOnly(moeDeviceMap);
#else
            (void)deviceMap;
            (void)moeDeviceMap;
            return false;
#endif
        }

        static bool HyV3GenericForwardMayUseFusedMoe(const std::map<std::string, int> &deviceMap,
                                                         const std::map<std::string, int> &moeDeviceMap) {
#ifdef USE_CUDA
            return HyV3CanUseGPUForward(deviceMap, moeDeviceMap) &&
                   HyV3MoeDeviceMapAllowsCudaOnly(moeDeviceMap);
#else
            (void)deviceMap;
            (void)moeDeviceMap;
            return false;
#endif
        }

        static bool GetHyV3ThreadTpDevices(const std::map<std::string, int> &deviceMap,
                                               std::vector<int> &devices,
                                               std::map<int, int> &ratios) {
            if (!GetHyV3GPUForwardDevices(deviceMap, devices, ratios)) {
                return false;
            }
            return devices.size() > 1;
        }

        static DivisionScheme ExtractHyV3FirstRangeScheme(const DivisionScheme &scheme) {
            DivisionScheme ret;
            for (auto &it : scheme) {
                ret[it.first];
                if (!it.second.empty()) {
                    ret[it.first].push_back(it.second[0]);
                }
            }
            return ret;
        }

        static DivisionScheme ExtractHyV3KVHeadScheme(const DivisionScheme &qkvScheme,
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

        static DataType ResolveHyV3ThreadTpComputeType(DataType modelType) {
            if (modelType == DataType::FLOAT16 || modelType == DataType::BFLOAT16) {
                return modelType;
            }
            return DataType::FLOAT16;
        }

        static DataType ResolveHyV3ThreadTpCacheType(DataType cacheType, DataType computeType) {
            if (cacheType == DataType::FLOAT16 ||
                cacheType == DataType::BFLOAT16 ||
                cacheType == DataType::FP8_E4M3) {
                return cacheType;
            }
            return computeType;
        }

        static void PrepareHyV3EmbeddingWeightType(Data &embedWeight,
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

        static void HyV3CpuEmbeddingDirect(Data &inputIds, Data &embedWeight,
                                               Data &hiddenStates, DataType outputType) {
            PrepareHyV3EmbeddingWeightType(embedWeight, outputType, true);
            inputIds.ToDevice(DataDevice::CPU);
            Executor *executor = (Executor*)GetExecutor();
            executor->RunOnDevice("cpu", "EmbeddingDirect",
                                  DataDict{{"input", &inputIds},
                                           {"weight", &embedWeight},
                                           {"output", &hiddenStates}},
                                  FloatDict(), IntDict());
        }

        static void PrepareHyV3CudaEmbeddingWeightType(Data &embedWeight,
                                                           DataType outputType) {
            if (embedWeight.dataType != outputType) {
                embedWeight.ResetMultiDeviceState();
                if (embedWeight.dataDevice != DataDevice::CPU) {
                    embedWeight.ToDevice(DataDevice::CPU);
                }
                ToDataTypeForceCPU(embedWeight, outputType);
            }
        }

        static Data *CreateHyV3CudaReplicaLike(const Data &source, int device) {
            Data *local = new Data(source.dataType);
            local->Resize(source.dims);
            local->dataDevice = DataDevice::CUDA;
            local->dataDeviceIds = {device};
            FastllmCudaSetDevice(device);
            local->Allocate(false);
            return local;
        }

        static void PrepareHyV3CpuEmbeddingHiddenStates(Data &hiddenStates,
                                                            const std::vector<int> &devices,
                                                            PersistentWorkerGroup &workerGroup) {
            AssertInFastLLM(!devices.empty(),
                            "HY-V3 ForwardGPU CPU embedding got empty CUDA devices.\n");
            hiddenStates.ToDevice(DataDevice::CPU);
            AssertInFastLLM(hiddenStates.cpuData != nullptr,
                            "HY-V3 ForwardGPU CPU embedding has no CPU data.\n");
            if (devices.size() == 1) {
                hiddenStates.ToDevice(DataDevice::CUDA, {devices[0]}, true);
                return;
            }

            uint64_t count = hiddenStates.Count(0);
            AssertInFastLLM(count <= (uint64_t)INT_MAX,
                            "HY-V3 ForwardGPU CPU embedding result is too large for NCCL broadcast.\n");
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
                    CreateHyV3CudaReplicaLike(hiddenStates, device);
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
                                "HY-V3 ForwardGPU CPU embedding missing local CUDA replica.\n");
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

        static Data *EnsureHyV3ThreadTpLocalCache(Data &root, int device, DataType localDataType) {
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

        static void PrepareHyV3SingleCudaCache(Data &cache, int device, DataType localDataType) {
            cache.isKVCache = true;
            cache.lockInCPU = false;
            if (cache.dataType != localDataType && cache.dims.empty()) {
                cache.dataType = localDataType;
                cache.UpdateUnitSize();
            }
            cache.ToDevice(DataDevice::CUDA, {device}, false);
        }

        static void SyncHyV3ThreadTpRootCacheMetaFromLocal(Data &root,
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

            AssertInFastLLM(firstLocal->pageIndex.size() < 1000000,
                            "HY-V3 ForwardGPU got invalid local paged cache pageIndex metadata.\n");

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

        static void SyncHyV3ThreadTpRootCacheMeta(Data &root,
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
            SyncHyV3ThreadTpRootCacheMetaFromLocal(root, firstLocal, devices,
                                                       kvHeadScheme, globalKVHeads, headDim);
        }

        static bool HyV3CudaGraphEnabled() {
            return GetFastllmEnv().cudaGraph;
        }

        static bool HyV3NeedRepeatPenalty(const GenerationConfig &config) {
            float diff = config.repeat_penalty - 1.0f;
            return diff > 1e-6f || diff < -1e-6f;
        }

        static Executor &HyV3ThreadLocalGenericExecutor() {
            static thread_local std::unique_ptr<Executor> executor;
            if (executor == nullptr) {
                executor.reset(new Executor());
            }
            return *executor;
        }

        class HyV3ScopedGenericExecutor {
        public:
            explicit HyV3ScopedGenericExecutor(const std::string &firstDevice)
                    : oldExecutor(GetExecutor()) {
                Executor &executor = HyV3ThreadLocalGenericExecutor();
                if (!firstDevice.empty()) {
                    executor.SetFirstDevice(firstDevice);
                }
                SetCurrentThreadExecutor(&executor);
            }

            ~HyV3ScopedGenericExecutor() {
                SetCurrentThreadExecutor(oldExecutor);
            }

            HyV3ScopedGenericExecutor(const HyV3ScopedGenericExecutor&) = delete;
            HyV3ScopedGenericExecutor &operator=(const HyV3ScopedGenericExecutor&) = delete;

        private:
            void *oldExecutor;
        };

        static void HyV3CudaClearMultiDeviceState(Data &data) {
            for (auto &it : data.multiDeviceDatas) {
                delete it.second;
            }
            data.multiDeviceDatas.clear();
            data.multiDeviceData = false;
            data.ClearTensorParallelLayout();
        }

        static void HyV3ResetCpuScratch(Data &data) {
            if (data.isFake) {
                data.isFake = false;
                data.cpuData = nullptr;
                data.cudaData = nullptr;
                data.deviceData = nullptr;
                data.expansionSize = 0;
                data.expansionBytes = 0;
            } else {
                data.FreeSpace();
            }
            HyV3CudaClearMultiDeviceState(data);
            data.dataDevice = DataDevice::CPU;
            data.dataDeviceIds.clear();
            data.lockInCPU = false;
            data.expansionDims.clear();
        }

        static void HyV3CudaPrepareLocalOutput(Data &data, int device) {
            if (data.isFake) {
                data.isFake = false;
                data.cpuData = nullptr;
                data.cudaData = nullptr;
                data.deviceData = nullptr;
                data.expansionSize = 0;
                data.expansionBytes = 0;
            }

            bool needFree = false;
            if (data.dataDevice != DataDevice::CUDA) {
                needFree = data.cpuData != nullptr || data.cudaData != nullptr ||
                           data.deviceData != nullptr || data.expansionBytes != 0;
            } else if (!data.dataDeviceIds.empty() && data.dataDeviceIds[0] != device) {
                needFree = true;
            } else if (data.cudaData != nullptr) {
                int ptrDevice = GetPointerDeviceId(data.cudaData);
                needFree = ptrDevice >= 0 && ptrDevice != device;
            }
            if (needFree) {
                data.FreeSpace();
            }
            HyV3CudaClearMultiDeviceState(data);
            data.dataDevice = DataDevice::CUDA;
            data.dataDeviceIds = {device};
            data.lockInCPU = false;
        }

        static void HyV3ZeroCudaLike(Data &dst, const Data &like, int device) {
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
                HyV3CudaClearMultiDeviceState(dst);
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

        static void HyV3PrepareFusedMoeWeightForCuda(Data &weight, int device) {
            FastllmCudaSetDevice(device);
            weight.ToDevice(DataDevice::CUDA, {device}, true);
            if (weight.dataType == DataType::FP8_E4M3 && weight.extraCudaData.empty()) {
                AssertInFastLLM(!weight.scales.empty(),
                                "HY-V3 FusedMOE FP8 weight has no scales.\n");
                float *cudaScales = (float*)FastllmCudaMalloc(weight.scales.size() * sizeof(float));
                FastllmCudaCopyFromHostToDevice(cudaScales, (void*)weight.scales.data(),
                                                weight.scales.size() * sizeof(float));
                weight.extraCudaData.push_back((void*)cudaScales);
                weight.scales.clear();
                weight.scales.shrink_to_fit();
            }
        }

        static int HyV3GcdInt(int a, int b) {
            a = a < 0 ? -a : a;
            b = b < 0 ? -b : b;
            while (b != 0) {
                int t = a % b;
                a = b;
                b = t;
            }
            return a == 0 ? 1 : a;
        }

        static int HyV3LcmInt(int a, int b) {
            a = std::max(1, a);
            b = std::max(1, b);
            return a / HyV3GcdInt(a, b) * b;
        }

        static int HyV3FusedInterSplitUnit(const Data &weight) {
            int unit = weight.groupCnt <= 0 ? 128 : weight.groupCnt;
            if (weight.dataType == DataType::FP8_E4M3) {
                if (weight.blockK > 0) {
                    unit = HyV3LcmInt(unit, weight.blockK);
                }
                if (weight.blockM > 0) {
                    unit = HyV3LcmInt(unit, weight.blockM);
                }
            } else if (weight.dataType == DataType::FP8_E4M3_BLOCK_128) {
                unit = 128;
            }
            return std::max(1, unit);
        }

        static DivisionScheme HyV3BuildFusedInterScheme(const Data &gate,
                                                            const std::vector<int> &devices,
                                                            std::map<int, int> ratios) {
            AssertInFastLLM(gate.dims.size() == 3 && gate.dims[1] > 0,
                            "HY-V3 fused TP split requires 3D gate weight.\n");
            std::vector<int> devCopy = devices;
            std::vector<int> points = FastllmMultiCudaGetSplitPoints(
                devCopy, ratios, gate.dims[1], HyV3FusedInterSplitUnit(gate));
            AssertInFastLLM((int)points.size() == (int)devices.size() + 1,
                            "HY-V3 fused TP split got invalid split points.\n");
            DivisionScheme scheme;
            for (int i = 0; i < (int)devices.size(); i++) {
                scheme[devices[i]];
                if (points[i] < points[i + 1]) {
                    scheme[devices[i]].push_back({points[i], points[i + 1]});
                }
            }
            return scheme;
        }

        static int HyV3Fp8ScaleCols(int cols, int blockM) {
            return (cols - 1) / blockM + 1;
        }

        static void HyV3AppendFp8ExpertRowScales(Data &dst, const Data &src,
                                                     int expert, int expertRows, int cols,
                                                     int rowStart, int rows) {
            if (src.dataType != DataType::FP8_E4M3 || rows <= 0) {
                return;
            }
            AssertInFastLLM(src.blockK > 0 && src.blockM > 0 && !src.scales.empty(),
                            "HY-V3 fused TP FP8 weight has invalid scale metadata.\n");
            AssertInFastLLM(expert >= 0 && expert < src.dims[0] &&
                            rowStart >= 0 && rowStart + rows <= expertRows &&
                            rowStart % src.blockK == 0,
                            "HY-V3 fused TP FP8 row scale slice is unaligned.\n");
            int scaleCols = HyV3Fp8ScaleCols(cols, src.blockM);
            int scaleRowsPerExpert = (expertRows - 1) / src.blockK + 1;
            int scaleRowStart = expert * scaleRowsPerExpert + rowStart / src.blockK;
            int scaleRowCount = (rows - 1) / src.blockK + 1;
            size_t offset = (size_t)scaleRowStart * scaleCols;
            size_t count = (size_t)scaleRowCount * scaleCols;
            AssertInFastLLM(offset + count <= src.scales.size(),
                            "HY-V3 fused TP FP8 row scale slice is out of bounds.\n");
            dst.scales.insert(dst.scales.end(),
                              src.scales.begin() + offset,
                              src.scales.begin() + offset + count);
        }

        static void HyV3CopyFusedInterRows(Data &dst, Data &src,
                                               int interStart, int localInter) {
            AssertInFastLLM(dst.dims.size() == 3 && src.dims.size() == 3,
                            "HY-V3 fused TP row shard expects 3D weights.\n");
            int experts = src.dims[0], inter = src.dims[1], hidden = src.dims[2];
            AssertInFastLLM(dst.dims[0] == experts && dst.dims[1] == localInter &&
                            dst.dims[2] == hidden && interStart >= 0 &&
                            localInter >= 0 && interStart + localInter <= inter,
                            "HY-V3 fused TP row shard shape mismatch.\n");
            if (localInter == 0) {
                return;
            }
            src.ToDevice(DataDevice::CPU);
            AssertInFastLLM(src.cpuData != nullptr && dst.cpuData != nullptr,
                            "HY-V3 fused TP row shard requires CPU buffers.\n");
            size_t rowBytes = HyV3BytesPerRow(src, hidden);
            for (int expert = 0; expert < experts; expert++) {
                memcpy(dst.cpuData + (size_t)expert * localInter * rowBytes,
                       src.cpuData + ((size_t)expert * inter + interStart) * rowBytes,
                       (size_t)localInter * rowBytes);
                HyV3AppendFp8ExpertRowScales(dst, src, expert, inter, hidden,
                                                 interStart, localInter);
            }
        }

        static void HyV3AppendFp8DownColumnScales(Data &dst, const Data &src,
                                                      int interStart, int localInter) {
            if (src.dataType != DataType::FP8_E4M3 || localInter <= 0) {
                return;
            }
            int experts = src.dims[0], hidden = src.dims[1], inter = src.dims[2];
            AssertInFastLLM(src.blockK > 0 && src.blockM > 0 && !src.scales.empty() &&
                            interStart >= 0 && interStart + localInter <= inter &&
                            interStart % src.blockM == 0,
                            "HY-V3 fused TP FP8 column scale slice is unaligned.\n");
            int srcScaleCols = HyV3Fp8ScaleCols(inter, src.blockM);
            int dstScaleCols = HyV3Fp8ScaleCols(localInter, src.blockM);
            int scaleColStart = interStart / src.blockM;
            int scaleRowsPerExpert = (hidden - 1) / src.blockK + 1;
            for (int expert = 0; expert < experts; expert++) {
                for (int scaleRow = 0; scaleRow < scaleRowsPerExpert; scaleRow++) {
                    size_t offset = ((size_t)expert * scaleRowsPerExpert + scaleRow) *
                                    srcScaleCols + scaleColStart;
                    AssertInFastLLM(offset + dstScaleCols <= src.scales.size(),
                                    "HY-V3 fused TP FP8 column scale slice is out of bounds.\n");
                    dst.scales.insert(dst.scales.end(),
                                      src.scales.begin() + offset,
                                      src.scales.begin() + offset + dstScaleCols);
                }
            }
        }

        static void HyV3CopyFusedDownInterColumns(Data &dst, Data &src,
                                                      int interStart, int localInter) {
            AssertInFastLLM(dst.dims.size() == 3 && src.dims.size() == 3,
                            "HY-V3 fused TP down shard expects 3D weights.\n");
            int experts = src.dims[0], hidden = src.dims[1], inter = src.dims[2];
            AssertInFastLLM(dst.dims[0] == experts && dst.dims[1] == hidden &&
                            dst.dims[2] == localInter && interStart >= 0 &&
                            localInter >= 0 && interStart + localInter <= inter,
                            "HY-V3 fused TP down shard shape mismatch.\n");
            if (localInter == 0) {
                return;
            }
            src.ToDevice(DataDevice::CPU);
            AssertInFastLLM(src.cpuData != nullptr && dst.cpuData != nullptr,
                            "HY-V3 fused TP down shard requires CPU buffers.\n");
            int rows = experts * hidden;
            size_t srcRowBytes = HyV3BytesPerRow(src, inter);
            size_t dstRowBytes = HyV3BytesPerRow(dst, localInter);
            if (src.dataType == DataType::FP8_E4M3_BLOCK_128) {
                const int block = 128;
                const int blockBytes = block + (int)sizeof(float);
                AssertInFastLLM(interStart % block == 0,
                                "HY-V3 fused TP FP8 block shard is unaligned.\n");
                int blockStart = interStart / block;
                int blockCount = (localInter + block - 1) / block;
                for (int row = 0; row < rows; row++) {
                    memcpy(dst.cpuData + (size_t)row * dstRowBytes,
                           src.cpuData + (size_t)row * srcRowBytes + (size_t)blockStart * blockBytes,
                           (size_t)blockCount * blockBytes);
                }
            } else {
                AssertInFastLLM(src.dataType == DataType::FP8_E4M3,
                                "HY-V3 fused TP only supports FP8 fused weights.\n");
                for (int row = 0; row < rows; row++) {
                    memcpy(dst.cpuData + (size_t)row * dstRowBytes,
                           src.cpuData + (size_t)row * srcRowBytes + interStart,
                           (size_t)localInter);
                }
                HyV3AppendFp8DownColumnScales(dst, src, interStart, localInter);
            }
        }

        static Data *HyV3CreateFusedInterShard(Data &src, int axis,
                                                   int device, std::pair<int, int> range) {
            AssertInFastLLM(axis == 1 || axis == 2,
                            "HY-V3 fused TP only splits inter dimension.\n");
            int localInter = range.second - range.first;
            AssertInFastLLM(localInter >= 0,
                            "HY-V3 fused TP got invalid shard range.\n");
            std::vector<int> localDims = src.dims;
            localDims[axis] = localInter;
            Data *local = new Data(src.dataType, localDims);
            HyV3CopyLinearWeightMeta(*local, src,
                                         src.name + ".tp" + std::to_string(device));
            local->scales.clear();
            local->dataDeviceIds = {device};
            if (local->Count(0) > 0) {
                local->Allocate(false);
                if (axis == 1) {
                    HyV3CopyFusedInterRows(*local, src, range.first, localInter);
                } else {
                    HyV3CopyFusedDownInterColumns(*local, src, range.first, localInter);
                }
                HyV3PrepareFusedMoeWeightForCuda(*local, device);
            } else {
                local->dataDevice = DataDevice::CUDA;
            }
            return local;
        }

        static bool HyV3FusedShardLayoutReady(const Data &weight,
                                                  const std::vector<int> &devices,
                                                  const DivisionScheme &scheme,
                                                  int axis) {
            if (!weight.multiDeviceData || weight.tpLayout != TP_LAYOUT_SHARDED ||
                weight.tpAxis != axis || weight.tpRanges != scheme) {
                return false;
            }
            for (int device : devices) {
                auto localIt = weight.multiDeviceDatas.find(device);
                auto rangeIt = scheme.find(device);
                if (localIt == weight.multiDeviceDatas.end() || localIt->second == nullptr ||
                    rangeIt == scheme.end()) {
                    return false;
                }
                int localInter = 0;
                for (auto &range : rangeIt->second) {
                    localInter += range.second - range.first;
                }
                Data *local = localIt->second;
                if (local->dims.size() != weight.dims.size() ||
                    local->dims[axis] != localInter ||
                    local->dataDevice != DataDevice::CUDA ||
                    local->dataDeviceIds.empty() || local->dataDeviceIds[0] != device) {
                    return false;
                }
                if (local->Count(0) > 0 &&
                    (local->cudaData == nullptr ||
                     (local->dataType == DataType::FP8_E4M3 && local->extraCudaData.empty()))) {
                    return false;
                }
            }
            return true;
        }

        static void HyV3PrepareFusedShardedWeight(Data &weight,
                                                      const std::vector<int> &devices,
                                                      const DivisionScheme &scheme,
                                                      int axis) {
            if (HyV3FusedShardLayoutReady(weight, devices, scheme, axis)) {
                return;
            }
            weight.ToDevice(DataDevice::CPU);
            HyV3CudaClearMultiDeviceState(weight);
            std::map<int, Data*> localDatas;
            for (int device : devices) {
                auto rangeIt = scheme.find(device);
                AssertInFastLLM(rangeIt != scheme.end(),
                                "HY-V3 fused TP missing device range.\n");
                std::pair<int, int> range = {0, 0};
                if (!rangeIt->second.empty()) {
                    AssertInFastLLM(rangeIt->second.size() == 1,
                                    "HY-V3 fused TP expects contiguous per-device shards.\n");
                    range = rangeIt->second[0];
                }
                localDatas[device] = HyV3CreateFusedInterShard(weight, axis, device, range);
            }
            weight.multiDeviceDatas.swap(localDatas);
            weight.multiDeviceData = true;
            weight.dataDevice = DataDevice::CUDA;
            weight.dataDeviceIds = devices;
            weight.tpLayout = TP_LAYOUT_SHARDED;
            weight.tpAxis = axis;
            weight.tpGlobalDims = weight.dims;
            weight.tpRanges = scheme;
            weight.cudaData = nullptr;
            weight.deviceData = nullptr;
            if (weight.cpuData != nullptr) {
                delete[] weight.cpuData;
                weight.cpuData = nullptr;
            }
            weight.scales.clear();
            weight.scales.shrink_to_fit();
        }

        static bool HyV3HasLocalFusedMoeShard(Data *gate, Data *up, Data *down) {
            return gate != nullptr && up != nullptr && down != nullptr &&
                   gate->dims.size() == 3 && up->dims.size() == 3 && down->dims.size() == 3 &&
                   gate->dims[1] > 0 && up->dims[1] > 0 && down->dims[2] > 0 &&
                   gate->cudaData != nullptr && up->cudaData != nullptr && down->cudaData != nullptr;
        }

        static void HyV3CudaFusedMOE(Qwen3CudaDirectRunner &runner,
                                         Data &input, Data &expertIndex, Data &expertScore,
                                         Data &gate, Data &up, Data &down, Data &w1,
                                         Data &output, int layer) {
            runner.Run("FusedMOE",
                       DataDict{{"input", &input}, {"index", &expertIndex}, {"score", &expertScore},
                                {"gate", &gate}, {"up", &up}, {"down", &down},
                                {"w1", &w1}, {"output", &output}},
                       FloatDict{{"swigluLimit", 0.0f}},
                       IntDict{{"layer", layer}, {"gateType", (int)MoeGateSwiglu}},
                       {"w1", "output"});
        }

        static bool HyV3HasLocalMoeShard(const std::vector<Data*> &localWeights) {
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

        static bool HyV3IsNVFP4WeightType(DataType dataType) {
            return dataType == DataType::NVFP4 ||
                   dataType == DataType::NVFP4_BLOCK_16 ||
                   dataType == DataType::NVFP4_BLOCK_16_E8M0;
        }

        struct HyV3ForwardSingleBuffers {
            Data embedOutput;
            Data hiddenStates;
            Data attenInput;
            Data qkv;
            Data q;
            Data qForAttentionHolder;
            Data attenOutput;
            Data attenLastOutput;
            Data routerLogits;
            Data routerLogitsTemp;
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
            Data qSizes;
            Data pageSizes;
            Data pageIndexs;
            Data lastPageLens;
            Data insertIndexs;
            Data insertPositions;
            std::vector<Data*> batchPastKeys;
            std::vector<Data*> batchPastValues;

            HyV3ForwardSingleBuffers() : batchPastKeys(1), batchPastValues(1) {}
        };

        static void HyV3DetachFakeReusableTensor(Data &data) {
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

        static void HyV3FreeReusableTensor(Data &data) {
            if (data.isFake) {
                HyV3DetachFakeReusableTensor(data);
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
            HyV3CudaClearMultiDeviceState(data);
        }

        static void HyV3FreeForwardSingleBuffers(HyV3ForwardSingleBuffers &buf) {
            HyV3FreeReusableTensor(buf.embedOutput);
            HyV3FreeReusableTensor(buf.hiddenStates);
            HyV3FreeReusableTensor(buf.attenInput);
            HyV3FreeReusableTensor(buf.qkv);
            HyV3FreeReusableTensor(buf.q);
            HyV3FreeReusableTensor(buf.qForAttentionHolder);
            HyV3FreeReusableTensor(buf.attenOutput);
            HyV3FreeReusableTensor(buf.attenLastOutput);
            HyV3FreeReusableTensor(buf.routerLogits);
            HyV3FreeReusableTensor(buf.routerLogitsTemp);
            HyV3FreeReusableTensor(buf.expertIndex);
            HyV3FreeReusableTensor(buf.expertScore);
            HyV3FreeReusableTensor(buf.w1);
            HyV3FreeReusableTensor(buf.w2);
            HyV3FreeReusableTensor(buf.w3);
            HyV3FreeReusableTensor(buf.tempInput);
            HyV3FreeReusableTensor(buf.tempOutput);
            HyV3FreeReusableTensor(buf.moeInputTemp);
            HyV3FreeReusableTensor(buf.moeOutputTemp);
            HyV3FreeReusableTensor(buf.moeFinal);
            HyV3FreeReusableTensor(buf.qSizes);
            HyV3FreeReusableTensor(buf.pageSizes);
            HyV3FreeReusableTensor(buf.pageIndexs);
            HyV3FreeReusableTensor(buf.lastPageLens);
            HyV3FreeReusableTensor(buf.insertIndexs);
            HyV3FreeReusableTensor(buf.insertPositions);
        }

        static void HyV3ReinitializeForwardSingleBuffers(HyV3ForwardSingleBuffers &buf) {
            HyV3FreeForwardSingleBuffers(buf);
            buf.~HyV3ForwardSingleBuffers();
            new (&buf) HyV3ForwardSingleBuffers();
        }

        struct HyV3CudaGraphDecodeState {
            std::mutex mutex;
            std::string signature;
            bool warmed = false;
            bool captured = false;
            bool disabled = false;
            void *graph = nullptr;
            void *exec = nullptr;
            Data inputIds;
            Data positionIds;
            HyV3ForwardSingleBuffers buffers;
            HyV3ForwardSingleBuffers metaBuffers;
            Data logitsHalf;
            Data logits;
            std::vector<int> lastInsertIndexHost;
            std::vector<int> lastPageSizesHost;
            std::vector<int> lastPageIndexHost;
            std::vector<int> lastDecodePageLensHost;
            std::vector<const Data*> lastPastKeyHosts;

            ~HyV3CudaGraphDecodeState() {
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

        struct HyV3CudaGraphSyncState {
            std::mutex mutex;
            std::condition_variable cv;
            int arrived = 0;
            int generation = 0;
            bool phaseOk = true;
            bool lastPhaseOk = true;
            bool disabled = false;
        };

        static HyV3CudaGraphSyncState &GetHyV3CudaGraphSyncState(const HyV3Model *model) {
            static std::mutex syncsMutex;
            static std::map<const HyV3Model*, std::unique_ptr<HyV3CudaGraphSyncState> > syncs;
            std::lock_guard<std::mutex> guard(syncsMutex);
            auto &sync = syncs[model];
            if (sync == nullptr) {
                sync.reset(new HyV3CudaGraphSyncState());
            }
            return *sync;
        }

        static bool HyV3CudaGraphSyncPhase(const HyV3Model *model, int participants, bool ok = true) {
            if (participants <= 1) {
                return ok;
            }
            HyV3CudaGraphSyncState &sync = GetHyV3CudaGraphSyncState(model);
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
            } else {
                sync.cv.wait(lock, [&]() {
                    return sync.generation != generation;
                });
                return sync.lastPhaseOk;
            }
        }

        static bool HyV3CudaGraphIsDisabled(const HyV3Model *model) {
            HyV3CudaGraphSyncState &sync = GetHyV3CudaGraphSyncState(model);
            std::lock_guard<std::mutex> guard(sync.mutex);
            return sync.disabled;
        }

        static void HyV3DisableCudaGraph(const HyV3Model *model) {
            HyV3CudaGraphSyncState &sync = GetHyV3CudaGraphSyncState(model);
            {
                std::lock_guard<std::mutex> guard(sync.mutex);
                sync.disabled = true;
            }
            sync.cv.notify_all();
        }

        static void HyV3DestroyCudaGraph(HyV3CudaGraphDecodeState &state) {
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

        static void HyV3AbortCudaGraphCapture() {
            void *capturedGraph = nullptr;
            if (FastllmCudaGraphEndCapture(&capturedGraph) && capturedGraph != nullptr) {
                FastllmCudaGraphDestroy(capturedGraph);
            }
        }

        static void HyV3DisableCudaGraphState(
                const HyV3Model *model,
                HyV3CudaGraphDecodeState &state) {
            HyV3DestroyCudaGraph(state);
            state.disabled = true;
            HyV3DisableCudaGraph(model);
        }

        static void HyV3WarnCudaGraphStage(
                const char *stage,
                int gpuId,
                bool localOk) {
            if (!localOk) {
                printf("Warning: HY-V3 CUDA graph %s failed on gpu %d: %s. Disable graph for this model.\n",
                       stage, gpuId, FastllmCudaGraphLastError());
                fflush(stdout);
            }
        }

        static bool HyV3SyncCudaGraphStage(
                const HyV3Model *model,
                HyV3CudaGraphDecodeState &state,
                int participants,
                const char *stage,
                int gpuId,
                bool localOk) {
            bool allOk = HyV3CudaGraphSyncPhase(model, participants, localOk);
            if (!allOk) {
                HyV3WarnCudaGraphStage(stage, gpuId, localOk);
                HyV3DisableCudaGraphState(model, state);
            }
            return allOk;
        }

        using HyV3CudaGraphStateKey = std::tuple<const HyV3Model*, int, int>;

        static std::mutex &HyV3CudaGraphStatesMutex() {
            static std::mutex statesMutex;
            return statesMutex;
        }

        static std::map<HyV3CudaGraphStateKey, std::unique_ptr<HyV3CudaGraphDecodeState> > &HyV3CudaGraphStates() {
            static std::map<HyV3CudaGraphStateKey, std::unique_ptr<HyV3CudaGraphDecodeState> > states;
            return states;
        }

        static HyV3CudaGraphDecodeState &GetHyV3CudaGraphDecodeState(
                const HyV3Model *model, int gpuId, int batch) {
            std::lock_guard<std::mutex> guard(HyV3CudaGraphStatesMutex());
            auto &states = HyV3CudaGraphStates();
            auto key = std::make_tuple(model, gpuId, batch);
            auto &state = states[key];
            if (state == nullptr) {
                state.reset(new HyV3CudaGraphDecodeState());
            }
            return *state;
        }

        static void HyV3DestroyCudaGraphDecodeStates(const HyV3Model *model, int gpuId) {
            std::lock_guard<std::mutex> guard(HyV3CudaGraphStatesMutex());
            auto &states = HyV3CudaGraphStates();
            for (auto &it : states) {
                if (std::get<0>(it.first) == model && std::get<1>(it.first) == gpuId &&
                    it.second != nullptr) {
                    std::lock_guard<std::mutex> stateGuard(it.second->mutex);
                    HyV3DestroyCudaGraph(*it.second);
                    HyV3ReinitializeForwardSingleBuffers(it.second->buffers);
                    HyV3ReinitializeForwardSingleBuffers(it.second->metaBuffers);
                }
            }
        }

        static void HyV3PrepareGraphCudaTensor(Data &dst, const Data &src, int device) {
            AssertInFastLLM(src.dataDevice == DataDevice::CUDA && src.cudaData != nullptr,
                            "HY-V3 CUDA graph requires CUDA source tensor.\n");
            FastllmCudaSetDevice(device);

            bool needReset = dst.isFake || dst.dataDevice != DataDevice::CUDA ||
                             dst.dataType != src.dataType || dst.dims != src.dims ||
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
                HyV3CudaClearMultiDeviceState(dst);
                dst.dataType = src.dataType;
                dst.UpdateUnitSize();
                dst.dataDevice = DataDevice::CUDA;
                dst.dataDeviceIds = {device};
                dst.Resize(src.dims);
            }
            dst.Allocate(false);
            FastllmCudaCopyFromDeviceToDevice(dst.cudaData, src.cudaData, src.GetBytes());
        }

        static void HyV3PrepareGraphIntTensor(Data &dst, int device, const std::vector<int> &host) {
            AssertInFastLLM(!host.empty(), "HY-V3 CUDA graph got empty int metadata.\n");
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
                HyV3CudaClearMultiDeviceState(dst);
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

        static bool HyV3CanUseCudaFullLogitsSampling(
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
                if (HyV3NeedRepeatPenalty(config)) {
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

        static Data &HyV3ThreadLocalCudaSamplingFullLogits() {
            static thread_local Data fullLogits(DataType::FLOAT32);
            return fullLogits;
        }

        static Data &HyV3ThreadLocalCudaSamplingOutput() {
            static thread_local Data data(DataType::INT32);
            return data;
        }

        static void HyV3GatherShardLogitsToRootCuda(
                int rootDevice,
                const std::vector<int> &devices,
                const DivisionScheme &lmHeadScheme,
                std::vector<Data> &localLogits,
                int batch,
                int vocabSize,
                Data &fullLogits) {
            FastllmCudaSetDevice(rootDevice);
            HyV3CudaPrepareLocalOutput(fullLogits, rootDevice);
            fullLogits.dataType = DataType::FLOAT32;
            fullLogits.UpdateUnitSize();
            fullLogits.Resize({batch, vocabSize});
            fullLogits.Allocate();

            for (int r = 0; r < (int)devices.size(); r++) {
                int device = devices[r];
                auto schemeIt = lmHeadScheme.find(device);
                AssertInFastLLM(schemeIt != lmHeadScheme.end(),
                                "HY-V3 CUDA sampling: missing lm_head split range.\n");
                AssertInFastLLM(localLogits[r].dataDevice == DataDevice::CUDA &&
                                localLogits[r].cudaData != nullptr,
                                "HY-V3 CUDA sampling: local logits must stay on CUDA.\n");
                int localVocab = localLogits[r].dims.back();
                int rows = localLogits[r].Count(0) / localVocab;
                AssertInFastLLM(rows == batch,
                                "HY-V3 CUDA sampling: local logits batch mismatch.\n");

                uint8_t *dstBase = (uint8_t*)fullLogits.cudaData;
                uint8_t *srcBase = (uint8_t*)localLogits[r].cudaData;
                int localOffset = 0;
                for (auto &range : schemeIt->second) {
                    int len = range.second - range.first;
                    AssertInFastLLM(range.first >= 0 && range.second <= vocabSize &&
                                    localOffset + len <= localVocab,
                                    "HY-V3 CUDA sampling: invalid lm_head split range.\n");
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

        static std::vector<int> HyV3SampleFromRootCudaLogits(
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

            Data &cudaOutput = HyV3ThreadLocalCudaSamplingOutput();
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

        static bool HyV3CanGraphIndexedMoe(
                const std::unordered_map<int, std::vector<std::vector<Data*> > > &deviceWeights,
                int device,
                int blockCnt,
                int hidden,
                DataType moeAtype) {
            if (moeAtype != DataType::FLOAT16 && moeAtype != DataType::BFLOAT16) {
                return false;
            }
            auto deviceIt = deviceWeights.find(device);
            if (deviceIt == deviceWeights.end() || (int)deviceIt->second.size() < blockCnt) {
                return false;
            }
            for (int i = 0; i < blockCnt; i++) {
                const std::vector<Data*> &layerWeights = deviceIt->second[i];
                if ((int)layerWeights.size() < 4 || layerWeights[0] != nullptr) {
                    return false;
                }

                bool hasShard = false;
                for (int j = 2; j + 1 < (int)layerWeights.size(); j += 2) {
                    Data *gateup = layerWeights[j];
                    Data *down = layerWeights[j + 1];
                    if (gateup == nullptr && down == nullptr) {
                        continue;
                    }
                    if (gateup == nullptr || down == nullptr ||
                        gateup->dataDevice != DataDevice::CUDA ||
                        down->dataDevice != DataDevice::CUDA ||
                        gateup->cudaData == nullptr || down->cudaData == nullptr ||
                        gateup->dataType != down->dataType ||
                        gateup->dims.size() != 2 || down->dims.size() != 2 ||
                        gateup->dims[1] != hidden ||
                        down->dims[0] != hidden ||
                        gateup->dims[0] != down->dims[1] * 2) {
                        return false;
                    }
                    bool supportedWeight =
                        gateup->dataType == DataType::FP8_E4M3 ||
                        gateup->dataType == DataType::FP8_E4M3_BLOCK_128 ||
                        HyV3IsNVFP4WeightType(gateup->dataType);
                    if (!supportedWeight) {
                        return false;
                    }
                    hasShard = true;
                }
                if (!hasShard) {
                    return false;
                }
            }
            return true;
        }
    }
#endif

    HyV3Model::HyV3Model() {
        this->model_type = "hy_v3";
        this->model_struct = "hy_v3";
        this->use_new_engine = true;

        // 默认使用alpaca的提示词和instruction
        this->pre_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n";
        this->user_role = "### Instruction:\n";
        this->bot_role = "\n\n### Response:";
        this->history_sep = "</s>";

        block_cnt = 32;
        rotary_dim = 128;

        weight.embeddingNames.insert("model.embed_tokens.weight");
        weight.linearNames = {
            "lm_head.weight", "model.layers.*.down_proj.weight", "model.layers.*.up_proj.weight",
            "model.layers.*.gate_proj.weight",  "model.layers.*.gate_proj.weight", "model.layers.*.gateup_proj.weight",
            "model.layers.*.self_attn.o_proj.weight", "model.layers.*.self_attn.q_proj.weight", "model.layers.*.self_attn.k_proj.weight",
            "model.layers.*.self_attn.v_proj.weight", "model.layers.*.self_attn.mergeqkv.weight", "model.layers.*.self_attn.W_pack.weight",
            "model.layers.*.mlp.*.weight"
        };
    }

    bool HyV3Model::IsHyV3() const {
        return true;
    }

    bool HyV3Model::IsDenseMlpLayer(int layer) const {
        return layer >= 0 && layer < this->first_k_dense_replace;
    }

    std::string HyV3Model::GetSharedMlpPrefix(int layer) const {
        std::string base = "model.layers." + std::to_string(layer) + ".mlp.";
        std::string sharedMlp = base + "shared_mlp.";
        if (this->weight.weight.find(sharedMlp + "down_proj.weight") != this->weight.weight.end() ||
            this->weight.weight.find(sharedMlp + "gateup_proj.weight") != this->weight.weight.end()) {
            return sharedMlp;
        }
        std::string sharedExperts = base + "shared_experts.";
        if (this->weight.weight.find(sharedExperts + "down_proj.weight") != this->weight.weight.end() ||
            this->weight.weight.find(sharedExperts + "gateup_proj.weight") != this->weight.weight.end()) {
            return sharedExperts;
        }
        return "";
    }

    bool HyV3Model::HasSharedMlpLayer(int layer) const {
        return !GetSharedMlpPrefix(layer).empty();
    }

    std::string HyV3Model::GetMoeGateWeightName(int layer) const {
        std::string prefix = "model.layers." + std::to_string(layer) + ".mlp.";
        std::string qwenGate = prefix + "gate.weight";
        if (this->weight.weight.find(qwenGate) != this->weight.weight.end() || !IsHyV3()) {
            return qwenGate;
        }
        return prefix + "router.gate.weight";
    }

    std::string HyV3Model::GetMoeGateBiasName(int layer) const {
        std::string prefix = "model.layers." + std::to_string(layer) + ".mlp.";
        std::string qwenBias = prefix + "gate.e_score_correction_bias";
        if (this->weight.weight.find(qwenBias) != this->weight.weight.end() || !IsHyV3()) {
            return qwenBias;
        }
        return prefix + "expert_bias";
    }

    void HyV3Model::ApplyRouterActivation(Data &routerLogits) const {
        if (this->moe_router_use_sigmoid) {
            Sigmoid(routerLogits, routerLogits);
        } else {
            Softmax(routerLogits, routerLogits, -1);
        }
    }

    std::map <std::string, std::vector <std::pair <std::string, DataType> > >
    HyV3Model::GetTensorMap(const std::vector <std::string> &tensorNames) {
        auto ret = basellm::GetTensorMap(tensorNames);
        if (!IsHyV3() || this->block_cnt <= 0) {
            return ret;
        }
        for (auto &it : ret) {
            int layer = -1;
            if (HyV3ParseLayerIndex(it.first, layer) && layer >= this->block_cnt) {
                it.second.clear();
            }
        }
        return ret;
    }

    bool HyV3Model::IsThreadTensorParallelEnabled() const {
#ifdef USE_CUDA
        std::vector<int> devices;
        std::map<int, int> ratios;
        return CanUseGPUForward() &&
               GetHyV3ThreadTpDevices(this->deviceMap, devices, ratios);
#else
        return false;
#endif
    }

    void HyV3Model::OnAutoWarmupFinished() {
#ifdef USE_CUDA
        if (GetFastllmEnv().cudaGraph) {
            if (threadTpWorkerGroup.HasWorkers()) {
                threadTpWorkerGroup.Stop();
            }
            PreCaptureCudaGraphAfterWarmup();
        }
#endif
    }

    bool HyV3Model::CanUseGPUForward() const {
#ifdef USE_CUDA
        return !GetKVCacheInCPU() &&
               HyV3CanUseGPUForward(this->deviceMap, this->moeDeviceMap);
#else
        return false;
#endif
    }

    PagedCacheManager* HyV3Model::GetPagedKVCacheManager(int layerIndex, bool isKey) const {
        if (layerIndex >= 0 && this->threadTpPagedCacheBase >= 0) {
            PagedCacheManager *manager = GetPagedCacheManager(
                (this->threadTpPagedCacheBase + layerIndex) * 2 + (isKey ? 0 : 1));
            if (manager != nullptr) {
                return manager;
            }
        }
        return basellm::GetPagedKVCacheManager(layerIndex, isKey);
    }

    std::vector<std::pair<int, PagedCacheManager*> > HyV3Model::GetPagedKVCacheManagers(int layerIndex, bool isKey) const {
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

    void HyV3Model::PreCaptureCudaGraphAfterWarmup() {
#ifdef USE_CUDA
        if (!GetFastllmEnv().cudaGraph || autoWarmupRunning.load() || GetKVCacheInCPU()) {
            return;
        }
        std::vector<int> devices;
        std::map<int, int> ratios;
        if (!HyV3ModelMoeLayersAllowCudaOnly(this) ||
            !HyV3CanUseGPUForward(this->deviceMap, this->moeDeviceMap) ||
            !GetHyV3GPUForwardDevices(this->deviceMap, devices, ratios) || devices.empty()) {
            return;
        }

        const int maxCudaGraphDecodeBatch = 32;
        int maxWarmupBatch = maxCudaGraphDecodeBatch;
        if (this->maxBatch > 0) {
            maxWarmupBatch = std::min(maxWarmupBatch, this->maxBatch);
        }

        auto printProgress = [](int done, int total, int batch) {
            const int barWidth = 32;
            int filled = total > 0 ? done * barWidth / total : barWidth;
            printf("\r[Fastllm] HY-V3 CUDA graph warmup capture [");
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
        printProgress(0, maxWarmupBatch, 0);

        for (int batch = 1; batch <= maxWarmupBatch; batch++) {
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
            printProgress(batch, maxWarmupBatch, batch);
        }
#endif
    }

    Data &HyV3Model::GetThreadTensorParallelBias(const std::string &name) {
        auto it = this->weight.weight.find(name);
        if (it != this->weight.weight.end()) {
            return it->second;
        }
        return this->threadTpEmptyBiases[name];
    }

    void HyV3Model::InitParams() {
        basellm::InitParams();
        num_experts = HyV3GetInt(this->weight.dicts, "num_experts", num_experts);
        num_experts_per_tok = HyV3GetInt(this->weight.dicts, "num_experts_per_tok", num_experts_per_tok);
        norm_topk_prob = HyV3IsTrueString(
            HyV3GetDict(this->weight.dicts, "norm_topk_prob",
                            HyV3GetDict(this->weight.dicts, "route_norm", "true")));
        first_k_dense_replace = HyV3GetInt(this->weight.dicts, "first_k_dense_replace", 0);
        n_shared_experts = HyV3GetInt(this->weight.dicts, "num_shared_experts",
                                          HyV3GetInt(this->weight.dicts, "n_shared_experts", 0));
        moe_router_use_sigmoid = HyV3IsTrueString(
            HyV3GetDict(this->weight.dicts, "moe_router_use_sigmoid", "false"));
        routed_scaling_factor = HyV3GetFloat(
            this->weight.dicts, "routed_scaling_factor",
            HyV3GetFloat(this->weight.dicts, "router_scaling_factor", 1.0f));

        num_key_value_heads = num_attention_heads;
        if (this->weight.dicts.find("num_key_value_heads") != this->weight.dicts.end()) {
            num_key_value_heads = atoi(this->weight.dicts["num_key_value_heads"].c_str());
        }
        if (this->weight.dicts.find("head_dim") != this->weight.dicts.end()) {
            head_dim = atoi(this->weight.dicts["head_dim"].c_str());
        }
        embed_dim = head_dim * num_attention_heads;
        rotary_dim = head_dim;
        if (this->weight.dicts.find("max_position_embeddings") != this->weight.dicts.end()) {
            max_positions = atoi(this->weight.dicts["max_position_embeddings"].c_str());
        }
        if (this->weight.dicts.find("rms_norm_eps") != this->weight.dicts.end()) {
            rms_norm_eps = atof(this->weight.dicts["rms_norm_eps"].c_str());
        }
        if (this->weight.dicts.find("rope_scaling.type") != this->weight.dicts.end()) {
            std::string type = this->weight.dicts["rope_scaling.type"];
            if (type == "linear")
               rope_type = RoPEType::LINEAR_SCALE;
            else if (type == "dynamic")
               rope_type = RoPEType::DYMAMIC_NTK;
        }
        if (this->weight.dicts.find("rope_theta") != this->weight.dicts.end()) {
            rope_base = atof(this->weight.dicts["rope_theta"].c_str());
        } else if (this->weight.dicts.find("rope_parameters.rope_theta") != this->weight.dicts.end()) {
            rope_base = atof(this->weight.dicts["rope_parameters.rope_theta"].c_str());
        }
        if (this->weight.dicts.find("rope_scaling.factor") != this->weight.dicts.end()) {
            rope_factor = atof(this->weight.dicts["rope_scaling.factor"].c_str());
        }

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

            std::string denseGateWeightName = "model.layers." + std::to_string(i) + ".mlp.gate_proj.weight";
            std::string denseUpWeightName = "model.layers." + std::to_string(i) + ".mlp.up_proj.weight";
            std::string denseGateupWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
            this->weightMergeRules.push_back(
                WeightMergeRule({WeightMergeRuleSingle({denseGateWeightName, denseUpWeightName},
                                                       denseGateupWeightName,
                                                       std::string("linearSwiglu"))})
            );
        }

        for (int i = 0; i < block_cnt; i++) {
            for (int j = 0; j < this->num_experts; j++) {
                std::string w1WeightName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".gate_proj.weight";
                std::string w3WeightName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".up_proj.weight";
                std::string swigluWeightName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".gateup_proj.weight";
                std::string downWeightName = "model.layers." + std::to_string(i) + ".mlp.experts." + std::to_string(j) + ".down_proj.weight";
                this->weightMergeRules.push_back(
                    WeightMergeRule({WeightMergeRuleSingle({w1WeightName, w3WeightName}, swigluWeightName, std::string("linearSwiglu"))})
                );

                this->AddSpecialWeight(swigluWeightName, "linearSwiglu", i);
                this->AddSpecialWeight(downWeightName, "linearColumn", i);

                this->moeLinears.insert(w1WeightName);
                this->moeLinears.insert(w3WeightName);
                this->moeLinears.insert(downWeightName);
            }

            for (auto sharedName : {"shared_mlp", "shared_experts"}) {
                std::string sharedPrefix = "model.layers." + std::to_string(i) + ".mlp." + sharedName + ".";
                std::string w1WeightName = sharedPrefix + "gate_proj.weight";
                std::string w3WeightName = sharedPrefix + "up_proj.weight";
                std::string swigluWeightName = sharedPrefix + "gateup_proj.weight";
                std::string downWeightName = sharedPrefix + "down_proj.weight";
                this->weightMergeRules.push_back(
                    WeightMergeRule({WeightMergeRuleSingle({w1WeightName, w3WeightName},
                                                           swigluWeightName,
                                                           std::string("linearSwiglu"))})
                );

                this->AddSpecialWeight(swigluWeightName, "linearSwiglu", i);
                this->AddSpecialWeight(downWeightName, "linearColumn", i);
                this->moeLinears.insert(w1WeightName);
                this->moeLinears.insert(w3WeightName);
                this->moeLinears.insert(downWeightName);
            }
        }
    }

    bool HyV3Model::IsFusedMoeLayerPlanned(int layer) const {
        return layer >= 0 &&
               layer < (int)this->moeFusedLayerPlanned.size() &&
               this->moeFusedLayerPlanned[layer];
    }

    bool HyV3Model::HasPlannedFusedMoeLayers() const {
        for (char planned : this->moeFusedLayerPlanned) {
            if (planned) {
                return true;
            }
        }
        return false;
    }

    bool HyV3Model::ArePlannedFusedMoeLayersReady() const {
        if (!HasPlannedFusedMoeLayers()) {
            return false;
        }
        for (int i = 0; i < block_cnt; i++) {
            if (!IsFusedMoeLayerPlanned(i)) {
                continue;
            }
            if (!HasFusedMoeWeights(i) ||
                i >= (int)moeGate3DExpertReady.size() ||
                i >= (int)moeUp3DExpertReady.size() ||
                i >= (int)moeDown3DExpertReady.size() ||
                (int)moeGate3DExpertReady[i].size() != this->num_experts ||
                (int)moeUp3DExpertReady[i].size() != this->num_experts ||
                (int)moeDown3DExpertReady[i].size() != this->num_experts) {
                return false;
            }
            for (int expert = 0; expert < this->num_experts; expert++) {
                if (!moeGate3DExpertReady[i][expert] ||
                    !moeUp3DExpertReady[i][expert] ||
                    !moeDown3DExpertReady[i][expert]) {
                    return false;
                }
            }
        }
        return true;
    }

    bool HyV3Model::HasFusedMoeWeights(int layer) const {
        return layer >= 0 &&
               layer < (int)moeGate3DWeights.size() &&
               layer < (int)moeUp3DWeights.size() &&
               layer < (int)moeDown3DWeights.size() &&
               moeGate3DWeights[layer] != nullptr &&
               moeUp3DWeights[layer] != nullptr &&
               moeDown3DWeights[layer] != nullptr;
    }

    Data *HyV3Model::GetFusedMoeWeightForDevice(Data *weight, int device) const {
        AssertInFastLLM(weight != nullptr,
                        "HY-V3 fused MoE weight is missing.\n");
        if (!weight->multiDeviceData) {
            return weight;
        }
        auto it = weight->multiDeviceDatas.find(device);
        AssertInFastLLM(it != weight->multiDeviceDatas.end() && it->second != nullptr,
                        "HY-V3 fused MoE local shard is missing.\n");
        return it->second;
    }

    void HyV3Model::PrepareFusedMoeLayerForDevices(int layer,
                                                       const std::vector<int> &devices,
                                                       std::map<int, int> ratios) {
#ifdef USE_CUDA
        if (!HasFusedMoeWeights(layer) || devices.empty()) {
            return;
        }
        Data &gate = *moeGate3DWeights[layer];
        Data &up = *moeUp3DWeights[layer];
        Data &down = *moeDown3DWeights[layer];
        AssertInFastLLM(gate.dims.size() == 3 && up.dims.size() == 3 &&
                        down.dims.size() == 3 &&
                        gate.dims[1] == up.dims[1] &&
                        gate.dims[1] == down.dims[2],
                        "HY-V3 fused MoE TP weights have incompatible shapes.\n");
        if (devices.size() == 1) {
            int device = devices[0];
            HyV3PrepareFusedMoeWeightForCuda(gate, device);
            HyV3PrepareFusedMoeWeightForCuda(up, device);
            HyV3PrepareFusedMoeWeightForCuda(down, device);
        } else {
            DivisionScheme interScheme = HyV3BuildFusedInterScheme(gate, devices, ratios);
            HyV3PrepareFusedShardedWeight(gate, devices, interScheme, 1);
            HyV3PrepareFusedShardedWeight(up, devices, interScheme, 1);
            HyV3PrepareFusedShardedWeight(down, devices, interScheme, 2);
        }
#else
        (void)layer;
        (void)devices;
        (void)ratios;
#endif
    }

    void HyV3Model::PrepareFusedMoeWeightsForDevices(const std::vector<int> &devices,
                                                         std::map<int, int> ratios) {
        if (!moeFusedWeightsPrepared || !HasPlannedFusedMoeLayers() || devices.empty()) {
            return;
        }
        for (int i = 0; i < block_cnt; i++) {
            if (!IsFusedMoeLayerPlanned(i)) {
                continue;
            }
            AssertInFastLLM(HasFusedMoeWeights(i),
                            "HY-V3 fused MoE weights are incomplete.\n");
            PrepareFusedMoeLayerForDevices(i, devices, ratios);
        }
    }

    static bool HyV3AllExpertsReady(const std::vector<std::vector<char>> &ready,
                                        int layer, int numExperts) {
        if (layer < 0 || layer >= (int)ready.size() ||
            (int)ready[layer].size() != numExperts) {
            return false;
        }
        for (int expert = 0; expert < numExperts; expert++) {
            if (!ready[layer][expert]) {
                return false;
            }
        }
        return true;
    }

    static bool HyV3LayerStreamReady(const std::vector<Data*> &gateWeights,
                                         const std::vector<Data*> &upWeights,
                                         const std::vector<Data*> &downWeights,
                                         int layer,
                                         const std::vector<std::vector<char>> &gateReady,
                                         const std::vector<std::vector<char>> &upReady,
                                         const std::vector<std::vector<char>> &downReady,
                                         int numExperts) {
        return layer >= 0 &&
               layer < (int)gateWeights.size() &&
               layer < (int)upWeights.size() &&
               layer < (int)downWeights.size() &&
               gateWeights[layer] != nullptr &&
               upWeights[layer] != nullptr &&
               downWeights[layer] != nullptr &&
               HyV3AllExpertsReady(gateReady, layer, numExperts) &&
               HyV3AllExpertsReady(upReady, layer, numExperts) &&
               HyV3AllExpertsReady(downReady, layer, numExperts);
    }

    void HyV3Model::TryFinalizeFusedMoeLayerParts(int layer) {
        if (layer < 0 || layer >= block_cnt) {
            return;
        }
        if (!IsFusedMoeLayerPlanned(layer)) {
            return;
        }
#ifdef USE_CUDA
        std::vector<int> devices;
        std::map<int, int> ratios;
        bool prepareCuda = !HyV3DisableFusedMoe() &&
            !HyV3LayerUsesMappedNonCudaMoe(this, layer) &&
            HyV3CanPlanFusedMoe(this->deviceMap, this->moeDeviceMap) &&
            GetHyV3GPUForwardDevices(this->deviceMap, devices, ratios) &&
            !devices.empty();
        if (prepareCuda && HyV3AllExpertsReady(moeGate3DExpertReady, layer, this->num_experts) &&
            moeGate3DWeights[layer] != nullptr) {
            if (devices.size() == 1) {
                HyV3PrepareFusedMoeWeightForCuda(*moeGate3DWeights[layer], devices[0]);
            } else {
                DivisionScheme interScheme = HyV3BuildFusedInterScheme(*moeGate3DWeights[layer], devices, ratios);
                HyV3PrepareFusedShardedWeight(*moeGate3DWeights[layer], devices, interScheme, 1);
            }
        }
        if (prepareCuda && HyV3AllExpertsReady(moeUp3DExpertReady, layer, this->num_experts) &&
            moeUp3DWeights[layer] != nullptr) {
            if (devices.size() == 1) {
                HyV3PrepareFusedMoeWeightForCuda(*moeUp3DWeights[layer], devices[0]);
            } else if (moeGate3DWeights[layer] != nullptr &&
                       moeGate3DWeights[layer]->multiDeviceData &&
                       !moeGate3DWeights[layer]->tpRanges.empty()) {
                HyV3PrepareFusedShardedWeight(*moeUp3DWeights[layer], devices,
                                                  moeGate3DWeights[layer]->tpRanges, 1);
            }
        }
        if (prepareCuda && HyV3AllExpertsReady(moeDown3DExpertReady, layer, this->num_experts) &&
            moeDown3DWeights[layer] != nullptr) {
            if (devices.size() == 1) {
                HyV3PrepareFusedMoeWeightForCuda(*moeDown3DWeights[layer], devices[0]);
            } else if (moeGate3DWeights[layer] != nullptr &&
                       moeGate3DWeights[layer]->multiDeviceData &&
                       !moeGate3DWeights[layer]->tpRanges.empty()) {
                HyV3PrepareFusedShardedWeight(*moeDown3DWeights[layer], devices,
                                                  moeGate3DWeights[layer]->tpRanges, 2);
            }
        }
#endif
        bool layerReady = HyV3LayerStreamReady(moeGate3DWeights, moeUp3DWeights, moeDown3DWeights,
                                                   layer, moeGate3DExpertReady,
                                                   moeUp3DExpertReady,
                                                   moeDown3DExpertReady, this->num_experts);
        if (!layerReady) {
            return;
        }
        moeFusedWeightsPrepared = ArePlannedFusedMoeLayersReady();
        if (moeFusedWeightsPrepared) {
            loadFusedMoePlanned = false;
            loadFusedMoeSourceWeights.clear();
        }
    }

    bool HyV3Model::TryConsumeFusedMoeSourceWeight(const std::string &weightName) {
        if (!loadFusedMoePlanned ||
            loadFusedMoeSourceWeights.find(weightName) == loadFusedMoeSourceWeights.end()) {
            return false;
        }
        int layer = -1, expert = -1;
        std::string kind;
        if (!HyV3ParseExpertWeightName(weightName, layer, expert, kind) ||
            layer < 0 || layer >= block_cnt || expert < 0 || expert >= this->num_experts) {
            return false;
        }
        if (!IsFusedMoeLayerPlanned(layer)) {
            return false;
        }
        if (kind != "gate" && kind != "up" && kind != "gateup" && kind != "down") {
            return false;
        }
        auto weightIt = this->weight.weight.find(weightName);
        if (weightIt == this->weight.weight.end() || weightIt->second.cpuData == nullptr) {
            return false;
        }
        Data &src = weightIt->second;
        if ((int)moeGate3DWeights.size() != block_cnt) {
            moeGate3DWeights.assign(block_cnt, nullptr);
            moeUp3DWeights.assign(block_cnt, nullptr);
            moeDown3DWeights.assign(block_cnt, nullptr);
        }
        if ((int)moeGate3DExpertReady.size() != block_cnt) {
            moeGate3DExpertReady.assign(block_cnt, std::vector<char>(this->num_experts, 0));
            moeUp3DExpertReady.assign(block_cnt, std::vector<char>(this->num_experts, 0));
            moeDown3DExpertReady.assign(block_cnt, std::vector<char>(this->num_experts, 0));
        }

        if (kind == "gate" || kind == "up") {
            if (src.dims.size() != 2 || !HyV3IsFusedFp8Type(src.dataType)) {
                return false;
            }
            int inter = src.dims[0];
            int hidden = src.dims[1];
            Data *&target = kind == "gate" ? moeGate3DWeights[layer] : moeUp3DWeights[layer];
            HyV3EnsureFusedLayerWeight(this->weight.weight, layer, this->num_experts,
                                           kind, src, inter, hidden, target);
            HyV3CopyRows(*target, expert * inter, src, 0, inter);
            HyV3CopyFp8ScaleRowsToExpert(*target, src, expert, 0, inter);
            if (kind == "gate") {
                moeGate3DExpertReady[layer][expert] = 1;
            } else {
                moeUp3DExpertReady[layer][expert] = 1;
            }
            HyV3ReleaseConsumedSourceWeight(src);
            consumedFusedMoeSourceWeights.insert(weightName);
            TryFinalizeFusedMoeLayerParts(layer);
            return true;
        }

        if (kind == "gateup") {
            if (src.dims.size() != 2 || (src.dims[0] & 1) != 0 ||
                !HyV3IsFusedFp8Type(src.dataType)) {
                return false;
            }
            int inter = src.dims[0] / 2;
            int hidden = src.dims[1];
            HyV3EnsureFusedLayerWeight(this->weight.weight, layer, this->num_experts,
                                           "gate", src, inter, hidden, moeGate3DWeights[layer]);
            HyV3EnsureFusedLayerWeight(this->weight.weight, layer, this->num_experts,
                                           "up", src, inter, hidden, moeUp3DWeights[layer]);
            HyV3CopyRows(*moeGate3DWeights[layer], expert * inter, src, 0, inter);
            HyV3CopyRows(*moeUp3DWeights[layer], expert * inter, src, inter, inter);
            HyV3CopyFp8ScaleRowsToExpert(*moeGate3DWeights[layer], src, expert, 0, inter);
            HyV3CopyFp8ScaleRowsToExpert(*moeUp3DWeights[layer], src, expert, inter, inter);
            moeGate3DExpertReady[layer][expert] = 1;
            moeUp3DExpertReady[layer][expert] = 1;
            HyV3ReleaseConsumedSourceWeight(src);
            consumedFusedMoeSourceWeights.insert(weightName);
            TryFinalizeFusedMoeLayerParts(layer);
            return true;
        }

        if (src.dims.size() != 2 || !HyV3IsFusedFp8Type(src.dataType)) {
            return false;
        }
        int hidden = src.dims[0];
        int inter = src.dims[1];
        HyV3EnsureFusedLayerWeight(this->weight.weight, layer, this->num_experts,
                                       "down", src, hidden, inter, moeDown3DWeights[layer]);
        HyV3CopyRows(*moeDown3DWeights[layer], expert * hidden, src, 0, hidden);
        HyV3CopyFp8ScaleRowsToExpert(*moeDown3DWeights[layer], src, expert, 0, hidden);
        moeDown3DExpertReady[layer][expert] = 1;
        HyV3ReleaseConsumedSourceWeight(src);
        consumedFusedMoeSourceWeights.insert(weightName);
        TryFinalizeFusedMoeLayerParts(layer);
        return true;
    }

    bool HyV3Model::TryBuildFusedMoeLayerFromLoaded(int layer) {
        if (layer < 0 || layer >= block_cnt) {
            return false;
        }
        if (!IsFusedMoeLayerPlanned(layer)) {
            return true;
        }
        if (HasFusedMoeWeights(layer)) {
            return HyV3LayerStreamReady(moeGate3DWeights, moeUp3DWeights, moeDown3DWeights,
                                            layer, moeGate3DExpertReady,
                                            moeUp3DExpertReady,
                                            moeDown3DExpertReady, this->num_experts);
        }
        if ((int)moeGate3DWeights.size() != block_cnt) {
            moeGate3DWeights.assign(block_cnt, nullptr);
            moeUp3DWeights.assign(block_cnt, nullptr);
            moeDown3DWeights.assign(block_cnt, nullptr);
        }
        if ((int)moeGate3DExpertReady.size() != block_cnt) {
            moeGate3DExpertReady.assign(block_cnt, std::vector<char>(this->num_experts, 0));
            moeUp3DExpertReady.assign(block_cnt, std::vector<char>(this->num_experts, 0));
            moeDown3DExpertReady.assign(block_cnt, std::vector<char>(this->num_experts, 0));
        }
        if (!HyV3CanBuildFusedLayer(this->weight.weight, layer, this->num_experts)) {
            return false;
        }

#ifdef USE_CUDA
        std::vector<int> devices;
        std::map<int, int> ratios;
        bool prepareCuda = !HyV3DisableFusedMoe() &&
            !HyV3LayerUsesMappedNonCudaMoe(this, layer) &&
            HyV3CanPlanFusedMoe(this->deviceMap, this->moeDeviceMap) &&
            GetHyV3GPUForwardDevices(this->deviceMap, devices, ratios) &&
            !devices.empty();
        DivisionScheme interScheme;
#endif

        HyV3BuildFusedLayerWeight(this->weight.weight, layer, this->num_experts,
                                      "gate", moeGate3DWeights[layer]);
#ifdef USE_CUDA
        if (prepareCuda) {
            if (devices.size() == 1) {
                HyV3PrepareFusedMoeWeightForCuda(*moeGate3DWeights[layer], devices[0]);
            } else {
                interScheme = HyV3BuildFusedInterScheme(*moeGate3DWeights[layer], devices, ratios);
                HyV3PrepareFusedShardedWeight(*moeGate3DWeights[layer], devices, interScheme, 1);
            }
        }
#endif

        HyV3BuildFusedLayerWeight(this->weight.weight, layer, this->num_experts,
                                      "up", moeUp3DWeights[layer]);
#ifdef USE_CUDA
        if (prepareCuda) {
            if (devices.size() == 1) {
                HyV3PrepareFusedMoeWeightForCuda(*moeUp3DWeights[layer], devices[0]);
            } else {
                HyV3PrepareFusedShardedWeight(*moeUp3DWeights[layer], devices, interScheme, 1);
            }
        }
#endif

        HyV3BuildFusedLayerWeight(this->weight.weight, layer, this->num_experts,
                                      "down", moeDown3DWeights[layer]);
#ifdef USE_CUDA
        if (prepareCuda) {
            if (devices.size() == 1) {
                HyV3PrepareFusedMoeWeightForCuda(*moeDown3DWeights[layer], devices[0]);
            } else {
                HyV3PrepareFusedShardedWeight(*moeDown3DWeights[layer], devices, interScheme, 2);
            }
        }
#endif

        for (int expert = 0; expert < this->num_experts; expert++) {
            std::string expertPrefix = HyV3ExpertPrefix(layer, expert);
            this->weight.weight.erase(expertPrefix + "gate_proj.weight");
            this->weight.weight.erase(expertPrefix + "up_proj.weight");
            this->weight.weight.erase(expertPrefix + "gateup_proj.weight");
            this->weight.weight.erase(expertPrefix + "down_proj.weight");
        }
        if ((int)moeGate3DExpertReady.size() == block_cnt &&
            (int)moeUp3DExpertReady.size() == block_cnt &&
            (int)moeDown3DExpertReady.size() == block_cnt) {
            std::fill(moeGate3DExpertReady[layer].begin(), moeGate3DExpertReady[layer].end(), 1);
            std::fill(moeUp3DExpertReady[layer].begin(), moeUp3DExpertReady[layer].end(), 1);
            std::fill(moeDown3DExpertReady[layer].begin(), moeDown3DExpertReady[layer].end(), 1);
        }
        weights.clear();
        biass.clear();
        threadTpMoeWeights.clear();
        threadTpMoeBiass.clear();
        singleGpuMoeWeights.clear();
        singleGpuMoeBiass.clear();
        threadTpWeightsPrepared.store(false, std::memory_order_release);
        singleGpuWeightsPrepared.store(false, std::memory_order_release);
        moeWeightsPrepared = false;

        moeFusedWeightsPrepared = ArePlannedFusedMoeLayersReady();
        if (moeFusedWeightsPrepared) {
            loadFusedMoePlanned = false;
            loadFusedMoeSourceWeights.clear();
        }
        return true;
    }

    bool HyV3Model::TryBuildFusedMoeWeightsFromLoaded() {
        if (moeFusedWeightsPrepared && HasPlannedFusedMoeLayers()) {
            return true;
        }
        if (!HasPlannedFusedMoeLayers()) {
            return false;
        }
        for (int i = 0; i < block_cnt; i++) {
            if (!IsFusedMoeLayerPlanned(i)) {
                continue;
            }
            if (HyV3LayerStreamReady(moeGate3DWeights, moeUp3DWeights, moeDown3DWeights,
                                         i, moeGate3DExpertReady,
                                         moeUp3DExpertReady,
                                         moeDown3DExpertReady, this->num_experts)) {
                continue;
            }
            if (!TryBuildFusedMoeLayerFromLoaded(i)) {
                return false;
            }
        }
        moeFusedWeightsPrepared = ArePlannedFusedMoeLayersReady();
        if (moeFusedWeightsPrepared) {
            loadFusedMoePlanned = false;
            loadFusedMoeSourceWeights.clear();
        }
        return moeFusedWeightsPrepared;
    }

    void HyV3Model::OnWeightsCreated(const std::set<std::string> &allWeightNames) {
        loadFusedMoePlanned = false;
        loadFusedMoeSourceWeights.clear();
        consumedFusedMoeSourceWeights.clear();
        moeFusedLayerPlanned.clear();
        moeFusedWeightsPrepared = false;
#ifdef USE_CUDA
        if (HyV3DisableFusedMoe() ||
            !HyV3CanPlanFusedMoe(this->deviceMap, this->moeDeviceMap) ||
            block_cnt <= 0 || this->num_experts <= 0 ||
            this->n_shared_experts > 0) {
            return;
        }

        std::vector<bool> layerUsesGateup(block_cnt, false);
        std::vector<char> plannedLayers(block_cnt, 0);
        std::set<std::string> plannedSourceWeights;
        for (int i = 0; i < block_cnt; i++) {
            if (IsDenseMlpLayer(i)) {
                continue;
            }
            if (HyV3LayerUsesMappedNonCudaMoe(this, i)) {
                continue;
            }
            plannedLayers[i] = 1;
            int layerInter = -1, layerHidden = -1;
            DataType layerType = DataType::FLOAT32;
            for (int j = 0; j < this->num_experts; j++) {
                std::string gateName = HyV3ExpertWeightName(i, j, "gate");
                std::string upName = HyV3ExpertWeightName(i, j, "up");
                std::string gateupName = HyV3ExpertWeightName(i, j, "gateup");
                std::string downName = HyV3ExpertWeightName(i, j, "down");
                bool hasMergedGateup = allWeightNames.find(gateupName) != allWeightNames.end();
                bool hasGateAndUp = allWeightNames.find(gateName) != allWeightNames.end() &&
                                    allWeightNames.find(upName) != allWeightNames.end();
                if ((!hasMergedGateup && !hasGateAndUp) ||
                    allWeightNames.find(downName) == allWeightNames.end()) {
                    plannedSourceWeights.clear();
                    return;
                }

                auto downIt = this->weight.weight.find(downName);
                auto gateupIt = this->weight.weight.find(gateupName);
                auto gateIt = this->weight.weight.find(gateName);
                auto upIt = this->weight.weight.find(upName);
                if (downIt == this->weight.weight.end() ||
                    (hasMergedGateup && gateupIt == this->weight.weight.end()) ||
                    (!hasMergedGateup && (gateIt == this->weight.weight.end() ||
                                          upIt == this->weight.weight.end()))) {
                    plannedSourceWeights.clear();
                    return;
                }

                const Data &gateSource = hasMergedGateup ? gateupIt->second : gateIt->second;
                const Data &upSource = hasMergedGateup ? gateupIt->second : upIt->second;
                const Data &downSource = downIt->second;
                if (gateSource.dims.size() != 2 || upSource.dims.size() != 2 ||
                    downSource.dims.size() != 2 ||
                    !HyV3IsFusedFp8Type(gateSource.dataType) ||
                    gateSource.dataType != upSource.dataType ||
                    gateSource.dataType != downSource.dataType) {
                    plannedSourceWeights.clear();
                    return;
                }
                int inter = hasMergedGateup ? gateSource.dims[0] / 2 : gateSource.dims[0];
                int hidden = gateSource.dims[1];
                if (inter <= 0 || hidden <= 0 ||
                    (hasMergedGateup && ((gateSource.dims[0] & 1) != 0 ||
                                         upSource.dims[0] != gateSource.dims[0])) ||
                    (!hasMergedGateup && (upSource.dims[0] != inter ||
                                          upSource.dims[1] != hidden)) ||
                    downSource.dims[0] != hidden || downSource.dims[1] != inter) {
                    plannedSourceWeights.clear();
                    return;
                }
                if (j == 0) {
                    layerInter = inter;
                    layerHidden = hidden;
                    layerType = gateSource.dataType;
                    layerUsesGateup[i] = hasMergedGateup;
                } else if (inter != layerInter || hidden != layerHidden ||
                           gateSource.dataType != layerType ||
                           hasMergedGateup != layerUsesGateup[i]) {
                    plannedSourceWeights.clear();
                    return;
                }

                if (hasMergedGateup) {
                    plannedSourceWeights.insert(gateupName);
                } else {
                    plannedSourceWeights.insert(gateName);
                    plannedSourceWeights.insert(upName);
                }
                plannedSourceWeights.insert(downName);
            }
        }
        bool hasPlannedLayer = false;
        for (char planned : plannedLayers) {
            if (planned) {
                hasPlannedLayer = true;
                break;
            }
        }
        if (!hasPlannedLayer) {
            return;
        }
        loadFusedMoeSourceWeights.swap(plannedSourceWeights);
        moeFusedLayerPlanned = plannedLayers;
        moeGate3DWeights.assign(block_cnt, nullptr);
        moeUp3DWeights.assign(block_cnt, nullptr);
        moeDown3DWeights.assign(block_cnt, nullptr);
        moeGate3DExpertReady.assign(block_cnt, std::vector<char>(this->num_experts, 0));
        moeUp3DExpertReady.assign(block_cnt, std::vector<char>(this->num_experts, 0));
        moeDown3DExpertReady.assign(block_cnt, std::vector<char>(this->num_experts, 0));

        for (int i = 0; i < block_cnt; i++) {
            if (!IsFusedMoeLayerPlanned(i)) {
                continue;
            }
            std::string gateSourceName = layerUsesGateup[i] ?
                HyV3ExpertWeightName(i, 0, "gateup") :
                HyV3ExpertWeightName(i, 0, "gate");
            std::string upSourceName = layerUsesGateup[i] ?
                HyV3ExpertWeightName(i, 0, "gateup") :
                HyV3ExpertWeightName(i, 0, "up");
            std::string downSourceName = HyV3ExpertWeightName(i, 0, "down");
            Data &gateSource = this->weight.weight[gateSourceName];
            Data &upSource = this->weight.weight[upSourceName];
            Data &downSource = this->weight.weight[downSourceName];
            int inter = layerUsesGateup[i] ? gateSource.dims[0] / 2 : gateSource.dims[0];
            int hidden = gateSource.dims[1];
            HyV3InitFusedLayerWeightMeta(this->weight.weight, i, this->num_experts,
                                             "gate", gateSource, inter, hidden,
                                             moeGate3DWeights[i]);
            HyV3InitFusedLayerWeightMeta(this->weight.weight, i, this->num_experts,
                                             "up", upSource, inter, hidden,
                                             moeUp3DWeights[i]);
            HyV3InitFusedLayerWeightMeta(this->weight.weight, i, this->num_experts,
                                             "down", downSource, hidden, inter,
                                             moeDown3DWeights[i]);
        }

        moeFusedWeightsPrepared = false;
        loadFusedMoePlanned = true;
#endif
    }

    int HyV3Model::GetWeightLoadPriority(
            const std::string &tensorName,
            const std::vector <std::pair <std::string, DataType> > &mappedWeights) const {
        if (!loadFusedMoePlanned) {
            return 0;
        }
        if (loadFusedMoeSourceWeights.find(tensorName) != loadFusedMoeSourceWeights.end()) {
            return HyV3SourceLoadPriority(tensorName, this->num_experts);
        }
        int priority = 0;
        for (auto &mapped : mappedWeights) {
            if (loadFusedMoeSourceWeights.find(mapped.first) != loadFusedMoeSourceWeights.end()) {
                int mappedPriority = HyV3SourceLoadPriority(mapped.first, this->num_experts);
                priority = priority == 0 ? mappedPriority : std::min(priority, mappedPriority);
            }
        }
        return priority;
    }

    bool HyV3Model::ShouldLoadWeightSeriallyBeforeOthers(
            const std::string &tensorName,
            const std::vector <std::pair <std::string, DataType> > &mappedWeights) const {
        if (!loadFusedMoePlanned) {
            return false;
        }
        if (loadFusedMoeSourceWeights.find(tensorName) != loadFusedMoeSourceWeights.end()) {
            return true;
        }
        for (auto &mapped : mappedWeights) {
            if (loadFusedMoeSourceWeights.find(mapped.first) != loadFusedMoeSourceWeights.end()) {
                return true;
            }
        }
        return false;
    }

    void HyV3Model::OnWeightLoadGroupStarted(const std::set<std::string> &weightNames) {
        if (!loadFusedMoePlanned || moeFusedWeightsPrepared) {
            return;
        }
        for (auto &weightName : weightNames) {
            if (loadFusedMoeSourceWeights.find(weightName) == loadFusedMoeSourceWeights.end()) {
                continue;
            }
            int layer = -1, expert = -1;
            std::string kind;
            if (!HyV3ParseExpertWeightName(weightName, layer, expert, kind) ||
                layer < 0 || layer >= block_cnt) {
                continue;
            }
            if (kind == "gate" || kind == "gateup") {
                HyV3AllocateFusedWeightForLoad(moeGate3DWeights[layer]);
            }
            if (kind == "up" || kind == "gateup") {
                HyV3AllocateFusedWeightForLoad(moeUp3DWeights[layer]);
            }
            if (kind == "down") {
                HyV3AllocateFusedWeightForLoad(moeDown3DWeights[layer]);
            }
        }
    }

    void HyV3Model::OnWeightLoaded(const std::string &weightName,
                                       const std::set<std::string> &finishedWeightNames) {
        (void)finishedWeightNames;
        if (!loadFusedMoePlanned || moeFusedWeightsPrepared ||
            loadFusedMoeSourceWeights.find(weightName) == loadFusedMoeSourceWeights.end()) {
            return;
        }
        int layer = -1, expert = -1;
        std::string kind;
        if (!HyV3ParseExpertWeightName(weightName, layer, expert, kind)) {
            return;
        }
        if (TryConsumeFusedMoeSourceWeight(weightName)) {
            return;
        }
        TryBuildFusedMoeLayerFromLoaded(layer);
    }

    bool HyV3Model::IsWeightConsumedAfterLoad(const std::string &weightName) const {
        return consumedFusedMoeSourceWeights.find(weightName) != consumedFusedMoeSourceWeights.end();
    }

    void HyV3Model::OnWeightLoadGroupFinished() {
        if (consumedFusedMoeSourceWeights.empty()) {
            return;
        }
        for (auto &weightName : consumedFusedMoeSourceWeights) {
            this->weight.weight.erase(weightName);
        }
        consumedFusedMoeSourceWeights.clear();
    }

    bool HyV3Model::ShouldDelaySpecialWeightCudaMove(const std::string &weightName) const {
        return loadFusedMoePlanned &&
               loadFusedMoeSourceWeights.find(weightName) != loadFusedMoeSourceWeights.end();
    }

    void HyV3Model::OnModelWeightsLoaded() {
        if (!loadFusedMoePlanned || moeFusedWeightsPrepared) {
            return;
        }
        std::lock_guard<std::mutex> guard(threadTpWeightPrepareLock);
        if (TryBuildFusedMoeWeightsFromLoaded()) {
            loadFusedMoePlanned = false;
            loadFusedMoeSourceWeights.clear();
            return;
        }
        loadFusedMoePlanned = false;
        loadFusedMoeSourceWeights.clear();
    }

    void HyV3Model::PrepareMoeWeights(bool enableFusedMoe) {
        if (enableFusedMoe && !moeFusedWeightsPrepared &&
            !HyV3DisableFusedMoe() &&
            TryBuildFusedMoeWeightsFromLoaded()) {
            // Continue below: mixed placement still needs ordinary MoE pointers
            // for layers that were intentionally not fused.
        }

        if (moeWeightsPrepared) {
            return;
        }
        weights.clear();
        biass.clear();
        weights.resize(block_cnt);
        biass.resize(block_cnt);
        for (int i = 0; i < block_cnt; i++) {
            weights[i].push_back(nullptr);
            weights[i].push_back(nullptr);
            biass[i].push_back(nullptr);
            biass[i].push_back(nullptr);

            std::string sharedPrefix = GetSharedMlpPrefix(i);
            if (!sharedPrefix.empty()) {
                std::string gateupWeightName = sharedPrefix + "gateup_proj.weight";
                std::string downWeightName = sharedPrefix + "down_proj.weight";
                AssertInFastLLM(weight.weight.find(gateupWeightName) != weight.weight.end(),
                                "HY-V3 requires merged shared gateup weight.\n");
                AssertInFastLLM(weight.weight.find(downWeightName) != weight.weight.end(),
                                "HY-V3 requires shared down weight.\n");
                Data &gateup = weight[gateupWeightName];
                Data &down = weight[downWeightName];
                gateup.tpLinearType = TP_LINEAR_ROW;
                gateup.tpPackType = TP_PACK_GATEUP;
                down.tpLinearType = TP_LINEAR_COLUMN;
                weights[i][0] = &gateup;
                weights[i][1] = &down;
            }

            if (IsDenseMlpLayer(i)) {
                continue;
            }
            if (HasFusedMoeWeights(i)) {
                continue;
            }
            for (int j = 0; j < this->num_experts; j++) {
                std::string expertPrefix = HyV3ExpertPrefix(i, j);
                std::string gateupWeightName = expertPrefix + "gateup_proj.weight";
                std::string downWeightName = expertPrefix + "down_proj.weight";
                AssertInFastLLM(weight.weight.find(gateupWeightName) != weight.weight.end(),
                                "HY-V3 requires merged expert gateup weight.\n");
                AssertInFastLLM(weight.weight.find(downWeightName) != weight.weight.end(),
                                "HY-V3 requires expert down weight.\n");
                Data &gateup = weight[gateupWeightName];
                Data &down = weight[downWeightName];
                gateup.tpLinearType = TP_LINEAR_ROW;
                gateup.tpPackType = TP_PACK_GATEUP;
                down.tpLinearType = TP_LINEAR_COLUMN;
                weights[i].push_back(&gateup);
                weights[i].push_back(&down);
                biass[i].push_back(nullptr);
                biass[i].push_back(nullptr);
            }
        }
        moeWeightsPrepared = true;
    }

    int HyV3Model::Forward(const fastllm::Data &inputIds, const fastllm::Data &attentionMask,
                            const fastllm::Data &positionIds, std::vector<std::pair<Data, Data>> &pastKeyValues,
                            const GenerationConfig &generationConfig, const LastTokensManager &lastTokens,
                            std::vector <float> *retLogits) {
        Data attentionMaskCopy(attentionMask), positionIdsCopy(positionIds);
        std::vector <Data*> attentionMasks = {attentionMaskCopy.dims.empty() ? nullptr : &attentionMaskCopy};
        std::vector <Data*> positionIdsVec = {&positionIdsCopy};
        std::vector <int> seqLens = {(int)inputIds.dims[1]};
        std::vector <GenerationConfig> generationConfigs = {generationConfig};
        std::vector <std::pair <Data*, Data*> > pagedPastKeyValues;
        for (int i = 0; i < pastKeyValues.size(); i++) {
            pagedPastKeyValues.push_back(std::make_pair(&pastKeyValues[i].first, &pastKeyValues[i].second));
        }
        std::vector <std::vector <float>*> batchLogits;
        batchLogits.push_back(retLogits);
        if (CanUseGPUForward()) {
            return ForwardGPU(1, inputIds, attentionMasks, positionIdsVec, seqLens,
                              pagedPastKeyValues, generationConfigs, lastTokens,
                              &batchLogits)[0];
        }
        return ForwardV2(1, inputIds, attentionMasks, positionIdsVec, seqLens,
                         pagedPastKeyValues, generationConfigs, lastTokens, &batchLogits)[0];
    }

    bool HyV3Model::ForwardSingleGPUDecodeGraph(
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
        if (!HyV3CudaGraphEnabled()) {
            return rejectGraph("disabled");
        }
        const int maxCudaGraphDecodeBatch = 32;
        if (batch <= 0 || batch > maxCudaGraphDecodeBatch) {
            return rejectGraph("unsupported batch");
        }
        if (!all1 || isPrefill || (int)seqLens.size() < batch ||
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
        if (!HyV3ModelMoeLayersAllowCudaOnly(this)) {
            return rejectGraph("mapped non-cuda moe");
        }

        int graphParticipants = tensorParallel ? std::max(2, (int)ratios.size()) : 1;
        auto syncGraphPeers = [&](bool ok = true) {
            return HyV3CudaGraphSyncPhase(this, graphParticipants, ok);
        };

        const DataType computeType = ResolveHyV3ThreadTpComputeType(this->dataType);
        const DataType threadTpMoeAtype = (this->moeAtype == DataType::FLOAT32) ? computeType : this->moeAtype;
        auto &moeWeightsByDevice = tensorParallel ? threadTpMoeWeights : singleGpuMoeWeights;
        auto &moeBiassByDevice = tensorParallel ? threadTpMoeBiass : singleGpuMoeBiass;
        int graphHidden = embed_dim;
        auto embedIt = weight.weight.find("model.embed_tokens.weight");
        if (embedIt != weight.weight.end() && embedIt->second.dims.size() >= 2) {
            graphHidden = embedIt->second.dims.back();
        } else {
            auto normIt = weight.weight.find("model.norm.weight");
            if (normIt != weight.weight.end() && !normIt->second.dims.empty()) {
                graphHidden = normIt->second.dims[0];
            }
        }
        bool indexedMoeOk = HyV3CanGraphIndexedMoe(moeWeightsByDevice, gpuId, block_cnt,
                                                       graphHidden, threadTpMoeAtype);
        bool fusedMoeOk = moeFusedWeightsPrepared;
        if (this->first_k_dense_replace > 0 || this->n_shared_experts > 0 || this->moe_router_use_sigmoid) {
            return rejectGraph("model has non-qwen3 moe graph features");
        }
        if (!syncGraphPeers(indexedMoeOk || fusedMoeOk)) {
            return rejectGraph("unsupported moe layout");
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
            ErrorInFastLLM("HY-V3 ForwardSingleGPU graph missing local tensor: " + name + ".\n");
            return nullptr;
        };

        Data *localInputIds = requireLocal((Data&)inputIds, "inputIds");
        Data *localPositionIds = requireLocal((Data&)positionIds, "positionIds");
        if (localInputIds->dims.size() != 2 || localInputIds->Count(0) != (uint64_t)batch ||
            localPositionIds->dims.empty() || localPositionIds->Count(0) != (uint64_t)batch) {
            return rejectGraph("input/position dims mismatch");
        }

        int currentTokens = 0;
        for (int i = 0; i < block_cnt; i++) {
            Data *firstBatchKey = pastKeyValues[i].first;
            Data *firstBatchValue = pastKeyValues[i].second;
            if (firstBatchKey == nullptr || firstBatchValue == nullptr ||
                firstBatchKey->pagedKVCacheData == nullptr ||
                firstBatchValue->pagedKVCacheData == nullptr) {
                return rejectGraph("kv cache is not paged");
            }
            for (int b = 0; b < batch; b++) {
                Data *pastKey = pastKeyValues[b * block_cnt + i].first;
                Data *pastValue = pastKeyValues[b * block_cnt + i].second;
                if (pastKey == nullptr || pastValue == nullptr) {
                    return rejectGraph("null kv cache");
                }
                if (pastKey->pagedKVCacheData == nullptr || pastValue->pagedKVCacheData == nullptr ||
                    pastKey->pagedKVCacheData != firstBatchKey->pagedKVCacheData ||
                    pastValue->pagedKVCacheData != firstBatchValue->pagedKVCacheData) {
                    return rejectGraph("kv cache is not paged");
                }
                if (pastKey->pageIndex.empty() || pastValue->pageIndex.empty()) {
                    return rejectGraph("empty kv page index");
                }
                if (pastKey->dataDevice != DataDevice::CUDA || pastValue->dataDevice != DataDevice::CUDA) {
                    return rejectGraph("kv cache not on cuda");
                }
                if (pastKey->dataType == DataType::FP8_E4M3 || pastValue->dataType == DataType::FP8_E4M3) {
                    return rejectGraph("fp8 kv cache");
                }
                if (pastKey->pageLen <= 0 || pastKey->pageLen != pastValue->pageLen ||
                    pastKey->pageIndex.size() != pastValue->pageIndex.size() ||
                    pastKey->lastPageLen != pastValue->lastPageLen) {
                    return rejectGraph("unaligned kv cache metadata");
                }
                int layerTokens = ((int)pastKey->pageIndex.size() - 1) * pastKey->pageLen + pastKey->lastPageLen;
                currentTokens = std::max(currentTokens, layerTokens);
            }
        }
        if (rope_type == RoPEType::DYMAMIC_NTK && currentTokens + 1 >= max_positions) {
            return rejectGraph("dynamic ntk beyond max position");
        }

        HyV3CudaGraphDecodeState &state = GetHyV3CudaGraphDecodeState(this, gpuId, batch);
        std::unique_lock<std::mutex> graphLock(state.mutex);
        if (HyV3CudaGraphIsDisabled(this)) {
            return false;
        }
        if (!syncGraphPeers(!state.disabled)) {
            if (state.disabled) {
                HyV3DisableCudaGraph(this);
            }
            return false;
        }
        FastllmCudaSetDevice(gpuId);
        HyV3PrepareGraphCudaTensor(state.inputIds, *localInputIds, gpuId);
        HyV3PrepareGraphCudaTensor(state.positionIds, *localPositionIds, gpuId);

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
                                "HY-V3 CUDA graph requires aligned paged cache layout across layers.\n");
                if (needNewPage[b]) {
                    int keyPage = pastKey->pagedKVCacheData->GetUnusedPageIndex(true);
                    int valuePage = pastValue->pagedKVCacheData->GetUnusedPageIndex(true);
                    if (insertIndexHost[b] < 0) {
                        insertIndexHost[b] = keyPage;
                    }
                    AssertInFastLLM(keyPage == insertIndexHost[b] && valuePage == insertIndexHost[b],
                                    "HY-V3 CUDA graph requires aligned K/V page indices across layers.\n");
                    pastKey->pageIndex.push_back(keyPage);
                    pastValue->pageIndex.push_back(valuePage);
                    pastKey->lastPageLen = 1;
                    pastValue->lastPageLen = 1;
                } else {
                    AssertInFastLLM(pastKey->pageIndex.back() == insertIndexHost[b] &&
                                    pastValue->pageIndex.back() == insertIndexHost[b] &&
                                    pastKey->lastPageLen == insertPositionHost[b] &&
                                    pastValue->lastPageLen == insertPositionHost[b],
                                    "HY-V3 CUDA graph requires aligned paged cache positions across layers.\n");
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
                            "HY-V3 CUDA graph page metadata exceeds captured capacity.\n");
            maxActualPagesPerRequest = std::max(maxActualPagesPerRequest, requestPages);
            pageSizesHost[b + 1] = pageSizesHost[b] + requestPages;
            pageIndexHost.insert(pageIndexHost.end(),
                                 firstKey->pageIndex.begin(), firstKey->pageIndex.end());
            AssertInFastLLM(firstKey->pageIndex.size() == firstValue->pageIndex.size() &&
                            firstKey->lastPageLen == firstValue->lastPageLen,
                            "HY-V3 CUDA graph requires aligned K/V page metadata.\n");
            for (int i = 1; i < block_cnt; i++) {
                Data *pastKey = pastKeyValues[b * block_cnt + i].first;
                Data *pastValue = pastKeyValues[b * block_cnt + i].second;
                AssertInFastLLM(pastKey->pageIndex == firstKey->pageIndex &&
                                pastValue->pageIndex == firstKey->pageIndex &&
                                pastKey->lastPageLen == firstKey->lastPageLen &&
                                pastValue->lastPageLen == firstKey->lastPageLen,
                                "HY-V3 CUDA graph requires aligned paged cache pages across layers.\n");
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
                  << ";moeAtype=" << (int)threadTpMoeAtype
                  << ";fusedMoe=" << (fusedMoeOk ? 1 : 0)
                  << ";kCache=" << pastKeyValues[0].first->pagedKVCacheData->cudaData
                  << ";vCache=" << pastKeyValues[0].second->pagedKVCacheData->cudaData
                  << ";lmLocal=" << requireLocal(weight["lm_head.weight"], "lm_head.weight")->dims[0];
        std::string newSignature = signature.str();
        bool signatureChanged = state.signature != newSignature;
        if (signatureChanged) {
            HyV3DestroyCudaGraph(state);
            state.signature = newSignature;
        }

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
            state.lastPastKeyHosts != currentPastKeyHosts;
        bool needFullMetaCopy = graphMetaMissing || signatureChanged || anyNewPage || metadataChanged;
        if (needFullMetaCopy) {
            AssertInFastLLM((int)pageIndexHost.size() <= pageIndexCapacity,
                            "HY-V3 CUDA graph page metadata exceeds fixed graph capacity.\n");
            std::vector<int> paddedPageIndexHost = pageIndexHost;
            paddedPageIndexHost.resize(pageIndexCapacity,
                                       paddedPageIndexHost.empty() ? 0 : paddedPageIndexHost.back());

            HyV3PrepareGraphIntTensor(state.metaBuffers.insertIndexs, gpuId, insertIndexHost);
            HyV3PrepareGraphIntTensor(state.metaBuffers.insertPositions, gpuId, insertPositionHost);
            HyV3PrepareGraphIntTensor(state.metaBuffers.qSizes, gpuId, qSizesHost);
            HyV3PrepareGraphIntTensor(state.metaBuffers.pageSizes, gpuId, pageSizesHost);
            // FlashInfer graph planning uses the CPU indptr; kernels still read the real CUDA indptr.
            state.metaBuffers.pageSizes.cpuIntDatas = graphPlanPageSizesHost;
            HyV3PrepareGraphIntTensor(state.metaBuffers.pageIndexs, gpuId, paddedPageIndexHost);
            HyV3PrepareGraphIntTensor(state.metaBuffers.lastPageLens, gpuId, lastPageLensHost);
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

        auto runGraphBodyWithBuffers = [&](HyV3ForwardSingleBuffers &workBuf,
                                           HyV3ForwardSingleBuffers &metaBuf) {
            Qwen3CudaDirectRunner cudaRunner(gpuId);
                HyV3ForwardSingleBuffers &buf = workBuf;
                if ((int)buf.batchPastKeys.size() != batch) {
                    buf.batchPastKeys.resize(batch);
                    buf.batchPastValues.resize(batch);
                }

                Qwen3CudaEmbeddingDirect(cudaRunner, state.inputIds,
                                         *requireLocal(weight["model.embed_tokens.weight"], "model.embed_tokens.weight"),
                                         buf.embedOutput);
                Qwen3CudaConvertToDataType(cudaRunner, buf.embedOutput, buf.hiddenStates, computeType);

                bool generatedAppendParams = false;
                bool generatedDecodeParams = false;
                for (int i = 0; i < block_cnt; i++) {
                    std::string inputRmsName = "model.layers." + std::to_string(i) + ".input_layernorm.weight";
                    std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
                    std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
                    std::string qNormName = "model.layers." + std::to_string(i) + ".self_attn.q_norm.weight";
                    std::string kNormName = "model.layers." + std::to_string(i) + ".self_attn.k_norm.weight";
                    std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";
                    std::string oBiasName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.bias";
                    std::string postRmsName = "model.layers." + std::to_string(i) + ".post_attention_layernorm.weight";
                    std::string gateWeightName = "model.layers." + std::to_string(i) + ".mlp.gate.weight";
                    std::string gateBiasName = "model.layers." + std::to_string(i) + ".mlp.gate.e_score_correction_bias";

                    Qwen3CudaRMSNorm(cudaRunner, buf.hiddenStates,
                                     *requireLocal(weight[inputRmsName], inputRmsName),
                                     rms_norm_eps, buf.attenInput);

                    Data *localMergeW = requireLocal(weight[mergeQkvWeightName], mergeQkvWeightName);
                    int group = num_attention_heads / num_key_value_heads;
                    int localKVHeads = localMergeW->tpKVHeads > 0 ?
                        localMergeW->tpKVHeads : localMergeW->dims[0] / ((group + 2) * head_dim);
                    int localQHeads = localMergeW->tpQHeads > 0 ?
                        localMergeW->tpQHeads : localKVHeads * group;
                    AssertInFastLLM(localKVHeads > 0 && localQHeads > 0,
                                    "HY-V3 ForwardSingleGPU graph got empty local attention shard.\n");

                    const bool enableStableFlashInferGraphPlan = true;
                    const int flashInferCudaGraph = 1;
                    Qwen3CudaAttentionPagedBlock(
                        cudaRunner,
                        &buf.attenInput,
                        localMergeW, requireLocal(GetThreadTensorParallelBias(mergeQkvBiasName), mergeQkvBiasName),
                        GetEmptyData(), GetEmptyData(),
                        GetEmptyData(), GetEmptyData(),
                        GetEmptyData(), GetEmptyData(),
                        GetEmptyData(), GetEmptyData(),
                        requireLocal(weight[qNormName], qNormName),
                        requireLocal(weight[kNormName], kNormName),
                        requireLocal(weight[oWeightName], oWeightName),
                        requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                        &state.positionIds,
                        &pastKeyValues,
                        &buf.batchPastKeys, &buf.batchPastValues,
                        &buf.qkv, &buf.q, &buf.attenOutput, &buf.attenLastOutput,
                        &buf.qForAttentionHolder,
                        &metaBuf.insertIndexs, &metaBuf.insertPositions,
                        &metaBuf.qSizes, &metaBuf.pageSizes, &metaBuf.pageIndexs, &metaBuf.lastPageLens,
                        &generatedAppendParams, &generatedDecodeParams,
                        batch, block_cnt, i,
                        seqLens,
                        localQHeads, localKVHeads, head_dim,
                        rotary_dim, rms_norm_eps,
                        rope_base, rope_factor, max_positions,
                        rope_type,
                        GetKVCacheInCPU(),
                        false,
                        &buf.hiddenStates,
                        true,
                        false,
                        pagedCacheLayerOffset,
                        true,
                        true,
                        enableStableFlashInferGraphPlan,
                        flashInferCudaGraph
                    );

                    if (tensorParallel) {
                        DataType residualType = buf.hiddenStates.dataType;
                        if (firstTensorParallelRank) {
                            if (buf.attenOutput.dataType == residualType) {
                                Qwen3CudaLinearAddBlock(cudaRunner, &buf.attenOutput,
                                                       requireLocal(weight[oWeightName], oWeightName),
                                                       requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                                                       &buf.attenLastOutput, &buf.hiddenStates);
                            } else {
                                Qwen3CudaLinear(cudaRunner, buf.attenOutput,
                                                *requireLocal(weight[oWeightName], oWeightName),
                                                *requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                                                buf.attenLastOutput);
                                if (buf.attenLastOutput.dataType != residualType) {
                                    Qwen3CudaToDataType(cudaRunner, buf.attenLastOutput, residualType);
                                }
                                Qwen3CudaAddTo(cudaRunner, buf.hiddenStates, buf.attenLastOutput);
                            }
                        } else {
                            Qwen3CudaLinear(cudaRunner, buf.attenOutput,
                                            *requireLocal(weight[oWeightName], oWeightName),
                                            *requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                                            buf.hiddenStates);
                            if (buf.hiddenStates.dataType != residualType) {
                                Qwen3CudaToDataType(cudaRunner, buf.hiddenStates, residualType);
                            }
                        }
                        FastllmNcclAllReduce(buf.hiddenStates.cudaData, buf.hiddenStates.cudaData,
                                             buf.hiddenStates.Count(0), buf.hiddenStates.dataType, gpuId);
                    } else {
                        Qwen3CudaLinear(cudaRunner, buf.attenOutput,
                                        *requireLocal(weight[oWeightName], oWeightName),
                                        *requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                                        buf.attenLastOutput);
                        if (buf.attenLastOutput.dataType != buf.hiddenStates.dataType) {
                            Qwen3CudaToDataType(cudaRunner, buf.attenLastOutput, buf.hiddenStates.dataType);
                        }
                        Qwen3CudaAddTo(cudaRunner, buf.hiddenStates, buf.attenLastOutput);
                    }

                    Qwen3CudaRMSNorm(cudaRunner, buf.hiddenStates,
                                     *requireLocal(weight[postRmsName], postRmsName),
                                     rms_norm_eps, buf.attenInput);
                    int localBatch = buf.attenInput.dims[0];
                    int localLen = buf.attenInput.dims[1];
                    buf.attenInput.Reshape({localBatch * localLen, buf.attenInput.dims[2]});
                    Qwen3CudaLinear(cudaRunner, buf.attenInput,
                                    *requireLocal(weight[gateWeightName], gateWeightName),
                                    *GetEmptyData(), buf.routerLogits, true);
                    Qwen3CudaConvertToDataType(cudaRunner, buf.routerLogits,
                                               buf.routerLogitsTemp, DataType::FLOAT32);
                    Qwen3CudaSoftmax(cudaRunner, buf.routerLogitsTemp, buf.routerLogitsTemp, -1);
                    Data *localGateBias = nullptr;
                    if (weight.weight.find(gateBiasName) != weight.weight.end()) {
                        localGateBias = requireLocal(weight[gateBiasName], gateBiasName);
                    }
                    Qwen3CudaSelectExpert(cudaRunner, buf.routerLogitsTemp, buf.expertIndex, buf.expertScore,
                                          this->num_experts_per_tok, true,
                                          this->routed_scaling_factor, localGateBias);

                    if (HasFusedMoeWeights(i)) {
                        Data *localGate = GetFusedMoeWeightForDevice(moeGate3DWeights[i], gpuId);
                        Data *localUp = GetFusedMoeWeightForDevice(moeUp3DWeights[i], gpuId);
                        Data *localDown = GetFusedMoeWeightForDevice(moeDown3DWeights[i], gpuId);
                        if (HyV3HasLocalFusedMoeShard(localGate, localUp, localDown)) {
                            HyV3CudaFusedMOE(cudaRunner, buf.attenInput, buf.expertIndex, buf.expertScore,
                                                 *localGate, *localUp, *localDown,
                                                 buf.w1, buf.moeFinal, i);
                        } else {
                            HyV3ZeroCudaLike(buf.moeFinal, buf.hiddenStates, gpuId);
                        }
                    } else {
                        auto &localWeights = moeWeightsByDevice.at(gpuId)[i];
                        auto &localBiass = moeBiassByDevice.at(gpuId)[i];
                        if (HyV3HasLocalMoeShard(localWeights)) {
                            Qwen3CudaMergeMOEBlock(cudaRunner, &buf.attenInput, &buf.expertIndex, &buf.expertScore,
                                &localWeights, &localBiass,
                                &buf.w1, &buf.w2, &buf.w3,
                                &buf.tempInput, &buf.tempOutput,
                                1.0f, &buf.moeFinal, i,
                                computeType, threadTpMoeAtype,
                                &buf.moeInputTemp, &buf.moeOutputTemp);
                        } else {
                            HyV3ZeroCudaLike(buf.moeFinal, buf.hiddenStates, gpuId);
                        }
                    }
                    buf.moeFinal.Reshape(buf.hiddenStates.dims);
                    if (buf.moeFinal.dataType != buf.hiddenStates.dataType) {
                        Qwen3CudaToDataType(cudaRunner, buf.moeFinal, buf.hiddenStates.dataType);
                    }
                    if (tensorParallel) {
                        if (firstTensorParallelRank) {
                            Qwen3CudaAddTo(cudaRunner, buf.moeFinal, buf.hiddenStates);
                        }
                        FastllmNcclAllReduce(buf.moeFinal.cudaData, buf.hiddenStates.cudaData,
                                             buf.moeFinal.Count(0), buf.moeFinal.dataType, gpuId);
                    } else {
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
            HyV3PrepareGraphCudaTensor(logits, state.logits, gpuId);
        };

        auto runWithoutGraph = [&]() -> bool {
            FastllmCudaMergeMOEClearGraphUnsafeFallbackFlag();
            runGraphBody();
            bool usedUnsafeMoeFallback = FastllmCudaMergeMOEUsedGraphUnsafeFallback();
            finishWithLogits();
            return usedUnsafeMoeFallback;
        };

        if (state.captured) {
            if (!syncGraphPeers()) {
                return false;
            }
            bool launchOk = FastllmCudaGraphLaunch(state.exec);
            if (HyV3SyncCudaGraphStage(this, state, graphParticipants,
                                           "replay", gpuId, launchOk)) {
                finishWithLogits();
                return true;
            }
            runWithoutGraph();
            return true;
        }

        if (!state.warmed) {
            bool usedUnsafeMoeFallback = runWithoutGraph();
            if (!syncGraphPeers(!usedUnsafeMoeFallback)) {
                if (usedUnsafeMoeFallback) {
                    printf("Warning: HY-V3 CUDA graph disabled on gpu %d because MergeMOE used CPU expert routing fallback during warmup.\n",
                           gpuId);
                    fflush(stdout);
                }
                HyV3DisableCudaGraphState(this, state);
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
        if (!HyV3SyncCudaGraphStage(this, state, graphParticipants,
                                        "begin capture", gpuId, beginOk)) {
            if (beginOk) {
                HyV3AbortCudaGraphCapture();
            }
            runWithoutGraph();
            return true;
        }
        FastllmCudaMergeMOEClearGraphUnsafeFallbackFlag();
        runGraphBody();
        bool usedUnsafeMoeFallback = FastllmCudaMergeMOEUsedGraphUnsafeFallback();
        if (!syncGraphPeers(!usedUnsafeMoeFallback)) {
            if (usedUnsafeMoeFallback) {
                printf("Warning: HY-V3 CUDA graph disabled on gpu %d because MergeMOE used CPU expert routing fallback during capture.\n",
                       gpuId);
                fflush(stdout);
            }
            HyV3AbortCudaGraphCapture();
            HyV3DisableCudaGraphState(this, state);
            runWithoutGraph();
            return true;
        }
        syncGraphPeers();
        bool endOk = FastllmCudaGraphEndCapture(&capturedGraph) && capturedGraph != nullptr;
        if (!HyV3SyncCudaGraphStage(this, state, graphParticipants,
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
        if (!HyV3SyncCudaGraphStage(this, state, graphParticipants,
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
        if (!HyV3SyncCudaGraphStage(this, state, graphParticipants,
                                        "first launch", gpuId, firstLaunchOk)) {
            runWithoutGraph();
            return true;
        }
        finishWithLogits();
        return true;
#endif
    }

    void HyV3Model::ForwardSingleGPU(
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
        ErrorInFastLLM("HY-V3 ForwardSingleGPU requires CUDA.\n");
#else
        AssertInFastLLM(ratios.find(gpuId) == ratios.end() || ratios[gpuId] > 0,
                        "HY-V3 ForwardSingleGPU got invalid GPU ratio.\n");
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
            ErrorInFastLLM("HY-V3 ForwardSingleGPU missing local tensor: " + name + ".\n");
            return nullptr;
        };

            const DataType computeType = ResolveHyV3ThreadTpComputeType(this->dataType);
            const DataType threadTpMoeAtype = (this->moeAtype == DataType::FLOAT32) ? computeType : this->moeAtype;
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

            Data attenInput, qkv, q, attenOutput, attenLastOutput;
            Data routerLogits, routerLogitsTemp, expertIndex, expertScore;
            Data w1, w2, w3, tempInput, tempOutput, moeInputTemp, moeOutputTemp, moeFinal;
            Data qSizes, pageSizes, pageIndexs, lastPageLens, insertIndexs, insertPositions;
            std::vector<Data*> batchPastKeys(batch), batchPastValues(batch);
            bool generatedAppendParams = false;
            bool generatedDecodeParams = false;
            auto &moeWeightsByDevice = tensorParallel ? threadTpMoeWeights : singleGpuMoeWeights;
            auto &moeBiassByDevice = tensorParallel ? threadTpMoeBiass : singleGpuMoeBiass;

            for (int i = 0; i < block_cnt; i++) {
                std::string inputRmsName = "model.layers." + std::to_string(i) + ".input_layernorm.weight";
                std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
                std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
                std::string qNormName = "model.layers." + std::to_string(i) + ".self_attn.q_norm.weight";
                std::string kNormName = "model.layers." + std::to_string(i) + ".self_attn.k_norm.weight";
                std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";
                std::string oBiasName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.bias";
                std::string postRmsName = "model.layers." + std::to_string(i) + ".post_attention_layernorm.weight";
                std::string gateWeightName = GetMoeGateWeightName(i);
                std::string gateBiasName = GetMoeGateBiasName(i);

                Qwen3CudaRMSNorm(cudaRunner, hiddenStates,
                                 *requireLocal(weight[inputRmsName], inputRmsName),
                                 rms_norm_eps, attenInput);
                Data *localMergeW = requireLocal(weight[mergeQkvWeightName], mergeQkvWeightName);
                int group = num_attention_heads / num_key_value_heads;
                int localKVHeads = localMergeW->tpKVHeads > 0 ?
                    localMergeW->tpKVHeads : localMergeW->dims[0] / ((group + 2) * head_dim);
                int localQHeads = localMergeW->tpQHeads > 0 ?
                    localMergeW->tpQHeads : localKVHeads * group;
                AssertInFastLLM(localKVHeads > 0 && localQHeads > 0,
                                "HY-V3 ForwardSingleGPU got empty local attention shard.\n");
                Qwen3CudaAttentionPagedBlock(
                    cudaRunner,
                    &attenInput,
                    localMergeW, requireLocal(GetThreadTensorParallelBias(mergeQkvBiasName), mergeQkvBiasName),
                    GetEmptyData(), GetEmptyData(),
                    GetEmptyData(), GetEmptyData(),
                    GetEmptyData(), GetEmptyData(),
                    GetEmptyData(), GetEmptyData(),
                    requireLocal(weight[qNormName], qNormName),
                    requireLocal(weight[kNormName], kNormName),
                    requireLocal(weight[oWeightName], oWeightName),
                    requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                    requireLocal((Data&)positionIds, "positionIds"),
                    &pastKeyValues,
                    &batchPastKeys, &batchPastValues,
                    &qkv, &q, &attenOutput, &attenLastOutput,
                    nullptr,
                    &insertIndexs, &insertPositions,
                    &qSizes, &pageSizes, &pageIndexs, &lastPageLens,
                    &generatedAppendParams, &generatedDecodeParams,
                    batch, block_cnt, i,
                    seqLens,
                    localQHeads, localKVHeads, head_dim,
                    rotary_dim, rms_norm_eps,
                    rope_base, rope_factor, max_positions,
                    rope_type,
                    GetKVCacheInCPU(),
                    isPrefill,
                    &hiddenStates,
                    true,
                    false,
                    pagedCacheLayerOffset,
                    true,
                    false
                );
                if (tensorParallel) {
                    DataType residualType = hiddenStates.dataType;
                    if (firstTensorParallelRank) {
                        if (attenOutput.dataType == residualType) {
                            Qwen3CudaLinearAddBlock(cudaRunner, &attenOutput,
                                                   requireLocal(weight[oWeightName], oWeightName),
                                                   requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                                                   &attenLastOutput, &hiddenStates);
                        } else {
                            Qwen3CudaLinear(cudaRunner, attenOutput,
                                            *requireLocal(weight[oWeightName], oWeightName),
                                            *requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                                            attenLastOutput);
                            if (attenLastOutput.dataType != residualType) {
                                Qwen3CudaToDataType(cudaRunner, attenLastOutput, residualType);
                            }
                            Qwen3CudaAddTo(cudaRunner, hiddenStates, attenLastOutput);
                        }
                    } else {
                        Qwen3CudaLinear(cudaRunner, attenOutput,
                                        *requireLocal(weight[oWeightName], oWeightName),
                                        *requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                                        hiddenStates);
                        if (hiddenStates.dataType != residualType) {
                            Qwen3CudaToDataType(cudaRunner, hiddenStates, residualType);
                        }
                    }
                } else {
                    Qwen3CudaLinear(cudaRunner, attenOutput,
                                    *requireLocal(weight[oWeightName], oWeightName),
                                    *requireLocal(GetThreadTensorParallelBias(oBiasName), oBiasName),
                                    attenLastOutput);
                }
                if (tensorParallel) {
                    FastllmNcclAllReduce(hiddenStates.cudaData, hiddenStates.cudaData,
                                         hiddenStates.Count(0), hiddenStates.dataType, gpuId);
                } else {
                    if (attenLastOutput.dataType != hiddenStates.dataType) {
                        Qwen3CudaToDataType(cudaRunner, attenLastOutput, hiddenStates.dataType);
                    }
                    Qwen3CudaAddTo(cudaRunner, hiddenStates, attenLastOutput);
                }
                Qwen3CudaRMSNorm(cudaRunner, hiddenStates,
                                 *requireLocal(weight[postRmsName], postRmsName),
                                 rms_norm_eps, attenInput);
                int localBatch = attenInput.dims[0];
                int localLen = attenInput.dims[1];
                attenInput.Reshape({localBatch * localLen, attenInput.dims[2]});

                if (IsDenseMlpLayer(i)) {
                    std::string gateupWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
                    std::string downWeightName = "model.layers." + std::to_string(i) + ".mlp.down_proj.weight";
                    Data *gateupBias = tensorParallel ?
                        requireLocal(GetThreadTensorParallelBias(gateupWeightName + ".tp_bias"),
                                     gateupWeightName + ".tp_bias") :
                        GetEmptyData();
                    Data *downBias = tensorParallel ?
                        requireLocal(GetThreadTensorParallelBias(downWeightName + ".tp_bias"),
                                     downWeightName + ".tp_bias") :
                        GetEmptyData();
                    Qwen3CudaLinearSwiglu(cudaRunner, attenInput,
                                         *requireLocal(weight[gateupWeightName], gateupWeightName),
                                         *gateupBias, w3, w1);
                    Qwen3CudaLinear(cudaRunner, w1,
                                    *requireLocal(weight[downWeightName], downWeightName),
                                    *downBias, moeFinal);
                    moeFinal.Reshape(hiddenStates.dims);
                    if (moeFinal.dataType != hiddenStates.dataType) {
                        Qwen3CudaToDataType(cudaRunner, moeFinal, hiddenStates.dataType);
                    }
                    if (tensorParallel) {
                        FastllmNcclAllReduce(moeFinal.cudaData, moeFinal.cudaData,
                                             moeFinal.Count(0), moeFinal.dataType, gpuId);
                    }
                    Qwen3CudaAddTo(cudaRunner, hiddenStates, moeFinal);
                    continue;
                }

                Qwen3CudaLinear(cudaRunner, attenInput,
                                *requireLocal(weight[gateWeightName], gateWeightName),
                                *GetEmptyData(), routerLogits, true);
                Qwen3CudaConvertToDataType(cudaRunner, routerLogits, routerLogitsTemp, DataType::FLOAT32);
                if (this->moe_router_use_sigmoid) {
                    Qwen3CudaSigmoid(cudaRunner, routerLogitsTemp, routerLogitsTemp);
                } else {
                    Qwen3CudaSoftmax(cudaRunner, routerLogitsTemp, routerLogitsTemp, -1);
                }
                Data *localGateBias = nullptr;
                if (weight.weight.find(gateBiasName) != weight.weight.end()) {
                    localGateBias = requireLocal(weight[gateBiasName], gateBiasName);
                }
                Qwen3CudaSelectExpert(cudaRunner, routerLogitsTemp, expertIndex, expertScore,
                                      this->num_experts_per_tok, this->norm_topk_prob,
                                      this->routed_scaling_factor, localGateBias);
                bool layerMappedNonCudaMoe = HyV3LayerUsesMappedNonCudaMoe(this, i);
                if (layerMappedNonCudaMoe) {
                    if (!tensorParallel || firstTensorParallelRank) {
                        std::string selectedMoeDevice = this->SelectMoeDeviceForLayer(i);
                        HyV3ResetCpuScratch(moeFinal);
                        FastllmCudaSetDevice(gpuId);
                        HyV3ScopedGenericExecutor scopedExecutor(selectedMoeDevice);
                        MergeMOEBlock(&attenInput, &expertIndex, &expertScore,
                            &weights[i], &biass[i],
                            &w1, &w2, &w3,
                            &tempInput, &tempOutput,
                            1.0f, &moeFinal, i,
                            computeType, threadTpMoeAtype,
                            &moeInputTemp, &moeOutputTemp);
                        FastllmCudaSetDevice(gpuId);
                        if (moeFinal.dataDevice != DataDevice::CUDA || moeFinal.cudaData == nullptr ||
                            (!moeFinal.dataDeviceIds.empty() && moeFinal.dataDeviceIds[0] != gpuId)) {
                            moeFinal.ToDevice(DataDevice::CUDA, {gpuId}, true);
                        }
                    } else {
                        HyV3ZeroCudaLike(moeFinal, hiddenStates, gpuId);
                    }
                } else if (HasFusedMoeWeights(i)) {
                    Data *localGate = GetFusedMoeWeightForDevice(moeGate3DWeights[i], gpuId);
                    Data *localUp = GetFusedMoeWeightForDevice(moeUp3DWeights[i], gpuId);
                    Data *localDown = GetFusedMoeWeightForDevice(moeDown3DWeights[i], gpuId);
                    if (HyV3HasLocalFusedMoeShard(localGate, localUp, localDown)) {
                        HyV3CudaFusedMOE(cudaRunner, attenInput, expertIndex, expertScore,
                                             *localGate, *localUp, *localDown,
                                             w1, moeFinal, i);
                    } else {
                        HyV3ZeroCudaLike(moeFinal, hiddenStates, gpuId);
                    }
                } else {
                    auto &localWeights = moeWeightsByDevice.at(gpuId)[i];
                    auto &localBiass = moeBiassByDevice.at(gpuId)[i];
                    if (HyV3HasLocalMoeShard(localWeights)) {
                        Qwen3CudaMergeMOEBlock(cudaRunner, &attenInput, &expertIndex, &expertScore,
                            &localWeights, &localBiass,
                            &w1, &w2, &w3,
                            &tempInput, &tempOutput,
                            1.0f, &moeFinal, i,
                            computeType, threadTpMoeAtype,
                            &moeInputTemp, &moeOutputTemp);
                    } else {
                        HyV3ZeroCudaLike(moeFinal, hiddenStates, gpuId);
                    }
                }
                moeFinal.Reshape(hiddenStates.dims);
                if (moeFinal.dataType != hiddenStates.dataType) {
                    Qwen3CudaToDataType(cudaRunner, moeFinal, hiddenStates.dataType);
                }
                if (tensorParallel) {
                    if (firstTensorParallelRank) {
                        Qwen3CudaAddTo(cudaRunner, moeFinal, hiddenStates);
                    }
                    FastllmNcclAllReduce(moeFinal.cudaData, hiddenStates.cudaData,
                                         moeFinal.Count(0), moeFinal.dataType, gpuId);
                } else {
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

    std::vector <int> HyV3Model::ForwardGPU(
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
        std::vector<int> devices;
        std::map<int, int> ratios;
        if (!CanUseGPUForward() ||
            !GetHyV3GPUForwardDevices(this->deviceMap, devices, ratios)) {
            if (threadTpWorkerGroup.HasWorkers()) {
                threadTpWorkerGroup.Stop();
            }
            return ForwardV2(batch, inputIds, attentionMask, positionIds, seqLens,
                             pastKeyValues, generationConfigs, lastTokens, retLogits);
        }
        bool tensorParallel = devices.size() > 1;
        PrepareMoeWeights(true);
        bool useFusedMoeWeights = moeFusedWeightsPrepared && HasPlannedFusedMoeLayers();
        bool useCpuEmbedding = !GetCudaEmbedding() || GetLowMemMode();
        const DataType computeType = ResolveHyV3ThreadTpComputeType(this->dataType);
        if (!useCpuEmbedding) {
            PrepareHyV3CudaEmbeddingWeightType(weight["model.embed_tokens.weight"], computeType);
        }

        AssertInFastLLM((int)pastKeyValues.size() >= batch * block_cnt,
                        "HY-V3 ForwardGPU: pastKeyValues size mismatch.\n");
        AssertInFastLLM((int)generationConfigs.size() >= batch,
                        "HY-V3 ForwardGPU: generation config size mismatch.\n");
        AssertInFastLLM((int)positionIds.size() >= batch && positionIds[0] != nullptr,
                        "HY-V3 ForwardGPU: positionIds size mismatch.\n");
        AssertInFastLLM(!GetKVCacheInCPU(),
                        "HY-V3 ForwardGPU doesn't support CPU KV cache.\n");
        if (tensorParallel) {
            AssertInFastLLM(FastllmInitNccl(devices),
                            "HY-V3 ForwardGPU requires NCCL initialization.\n");
        }

        if (threadTpPagedCacheBase < 0) {
            threadTpPagedCacheBase = qwen3MoeThreadTpNextPagedCacheBase.fetch_add(
                std::max(1, block_cnt * ((int)devices.size() + 1)));
        }

        int seqLen = inputIds.dims[1];
        bool all1 = true;
        for (int i = 0; i < batch; i++) {
            all1 &= (seqLens[i] == 1);
        }
        bool isPrefill = !all1;

        Data allPositionIds;
        if (all1 && positionIds[0]->dataType == DataType::FLOAT32) {
            std::vector<float> vPositionIds;
            vPositionIds.reserve(batch);
            for (int b = 0; b < batch; b++) {
                vPositionIds.push_back(((float*)positionIds[b]->cpuData)[0]);
            }
            allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, vPositionIds));
        } else {
            std::vector<float> vPositionIds;
            for (int b = 0; b < batch; b++) {
                AssertInFastLLM(positionIds[b] != nullptr,
                                "HY-V3 ForwardGPU: null positionIds.\n");
                for (int i = 0; i < seqLens[b]; i++) {
                    vPositionIds.push_back(((float*)positionIds[b]->cpuData)[i]);
                }
            }
            allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, (int)vPositionIds.size()}, vPositionIds));
        }

        Data gpuInputIds;
        gpuInputIds.CopyFrom(inputIds);
        if (tensorParallel) {
            PrepareMultiCudaReplicatedData(gpuInputIds, devices, true);
            PrepareMultiCudaReplicatedData(allPositionIds, devices, true);
        }

        std::vector<DivisionScheme> localKvHeadSchemes;
        DivisionScheme localLmHeadScheme;
        const std::vector<DivisionScheme> *kvHeadSchemes = &localKvHeadSchemes;
        const DivisionScheme *lmHeadScheme = &localLmHeadScheme;
        Data &lmHead = weight["lm_head.weight"];

        auto layerNeedsCudaMoeCache = [&](int layer) {
            return !IsDenseMlpLayer(layer) &&
                   !HyV3LayerUsesMappedNonCudaMoe(this, layer) &&
                   !HasFusedMoeWeights(layer);
        };
        auto anyLayerNeedsCudaMoeCache = [&]() {
            for (int i = 0; i < block_cnt; i++) {
                if (layerNeedsCudaMoeCache(i)) {
                    return true;
                }
            }
            return false;
        };

        auto hasMoeCache = [&](const std::unordered_map<int, std::vector<std::vector<Data*> > > &weightCache,
                               const std::unordered_map<int, std::vector<std::vector<Data*> > > &biasCache) {
            if (!anyLayerNeedsCudaMoeCache()) {
                return true;
            }
            int expectedSize = this->num_experts * 2 + 2;
            for (int device : devices) {
                auto weightIt = weightCache.find(device);
                auto biasIt = biasCache.find(device);
                if (weightIt == weightCache.end() || biasIt == biasCache.end() ||
                    (int)weightIt->second.size() != block_cnt || (int)biasIt->second.size() != block_cnt) {
                    return false;
                }
                for (int i = 0; i < block_cnt; i++) {
                    if ((int)weightIt->second[i].size() != expectedSize ||
                        (int)biasIt->second[i].size() != expectedSize) {
                        return false;
                    }
                    if (!layerNeedsCudaMoeCache(i)) {
                        continue;
                    }
                    for (int j = 2; j < expectedSize; j++) {
                        if (weightIt->second[i][j] == nullptr) {
                            return false;
                        }
                    }
                }
            }
            return true;
        };

        auto fillMoeCache = [&](std::unordered_map<int, std::vector<std::vector<Data*> > > &weightCache,
                                std::unordered_map<int, std::vector<std::vector<Data*> > > &biasCache,
                                bool useLocalShards) {
            weightCache.clear();
            biasCache.clear();
            for (int device : devices) {
                auto &deviceWeights = weightCache[device];
                auto &deviceBiass = biasCache[device];
                deviceWeights.resize(block_cnt);
                deviceBiass.resize(block_cnt);
                for (int i = 0; i < block_cnt; i++) {
                    auto &layerWeights = deviceWeights[i];
                    auto &layerBiass = deviceBiass[i];
                    layerWeights.reserve(this->num_experts * 2 + 2);
                    layerBiass.reserve(this->num_experts * 2 + 2);
                    layerWeights.push_back(nullptr);
                    layerWeights.push_back(nullptr);
                    layerBiass.push_back(nullptr);
                    layerBiass.push_back(nullptr);
                    auto getLocalOrRoot = [&](Data &data) -> Data* {
                        if (useLocalShards) {
                            auto it = data.multiDeviceDatas.find(device);
                            if (it != data.multiDeviceDatas.end() && it->second != nullptr) {
                                return it->second;
                            }
                        }
                        return &data;
                    };
                    std::string sharedPrefix = GetSharedMlpPrefix(i);
                    if (!sharedPrefix.empty()) {
                        layerWeights[0] = getLocalOrRoot(weight[sharedPrefix + "gateup_proj.weight"]);
                        layerWeights[1] = getLocalOrRoot(weight[sharedPrefix + "down_proj.weight"]);
                    }
                    if (!layerNeedsCudaMoeCache(i)) {
                        layerWeights.resize(this->num_experts * 2 + 2, nullptr);
                        layerBiass.resize(this->num_experts * 2 + 2, nullptr);
                        continue;
                    }
                    for (int j = 0; j < this->num_experts; j++) {
                        std::string gateupWeightName = "model.layers." + std::to_string(i) +
                            ".mlp.experts." + std::to_string(j) + ".gateup_proj.weight";
                        std::string downWeightName = "model.layers." + std::to_string(i) +
                            ".mlp.experts." + std::to_string(j) + ".down_proj.weight";
                        layerWeights.push_back(getLocalOrRoot(weight[gateupWeightName]));
                        layerWeights.push_back(getLocalOrRoot(weight[downWeightName]));
                        layerBiass.push_back(nullptr);
                        layerBiass.push_back(nullptr);
                    }
                }
            }
        };

        if (tensorParallel) {
            auto usePreparedThreadTpSchemes = [&]() {
                AssertInFastLLM(threadTpPreparedDevices == devices && threadTpPreparedRatios == ratios,
                                "HY-V3 ForwardGPU thread TP device config changed after weights were prepared.\n");
                AssertInFastLLM((int)threadTpKVHeadSchemes.size() == block_cnt &&
                                !threadTpLmHeadScheme.empty() &&
                                hasMoeCache(threadTpMoeWeights, threadTpMoeBiass),
                                "HY-V3 ForwardGPU thread TP cached weight schemes are incomplete.\n");
                kvHeadSchemes = &threadTpKVHeadSchemes;
                lmHeadScheme = &threadTpLmHeadScheme;
            };

            if (threadTpWeightsPrepared.load(std::memory_order_acquire)) {
                usePreparedThreadTpSchemes();
            } else {
                std::lock_guard<std::mutex> guard(threadTpWeightPrepareLock);
                if (!threadTpWeightsPrepared.load(std::memory_order_relaxed)) {
                    auto prepareReplicated = [&](const std::string &name) {
                        PrepareMultiCudaReplicatedData(this->weight[name], devices, true);
                    };
                    if (!useCpuEmbedding) {
                        prepareReplicated("model.embed_tokens.weight");
                    }
                    prepareReplicated("model.norm.weight");

                    threadTpKVHeadSchemes.assign(block_cnt, DivisionScheme());
                    for (int i = 0; i < block_cnt; i++) {
                        std::string inputRmsName = "model.layers." + std::to_string(i) + ".input_layernorm.weight";
                        std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
                        std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
                        std::string qNormName = "model.layers." + std::to_string(i) + ".self_attn.q_norm.weight";
                        std::string kNormName = "model.layers." + std::to_string(i) + ".self_attn.k_norm.weight";
                        std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";
                        std::string oBiasName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.bias";
                        std::string postRmsName = "model.layers." + std::to_string(i) + ".post_attention_layernorm.weight";
                        std::string gateWeightName = GetMoeGateWeightName(i);
                        std::string gateBiasName = GetMoeGateBiasName(i);

                        AssertInFastLLM(weight.weight.find(mergeQkvWeightName) != weight.weight.end(),
                                        "HY-V3 ForwardGPU requires merged qkv weight.\n");
                        if (!IsDenseMlpLayer(i)) {
                            AssertInFastLLM(weight.weight.find(gateWeightName) != weight.weight.end(),
                                            "HY-V3 ForwardGPU requires router gate weight.\n");
                        }

                        prepareReplicated(inputRmsName);
                        prepareReplicated(qNormName);
                        prepareReplicated(kNormName);
                        prepareReplicated(postRmsName);
                        if (!IsDenseMlpLayer(i)) {
                            prepareReplicated(gateWeightName);
                        }
                        if (weight.weight.find(gateBiasName) != weight.weight.end()) {
                            prepareReplicated(gateBiasName);
                        }

                        Data &mergeW = weight[mergeQkvWeightName];
                        Data &mergeB = GetThreadTensorParallelBias(mergeQkvBiasName);
                        mergeW.tpPackType = TP_PACK_QKV;
                        mergeW.tpQHeads = num_attention_heads;
                        mergeW.tpKVHeads = num_key_value_heads;
                        mergeW.tpHeadDim = head_dim;
                        std::vector<int> devCopy = devices;
                        DivisionScheme qkvScheme = BuildMultiCudaRowSplitScheme(mergeW, devCopy, ratios);
                        AssertInFastLLM(SplitMultiCudaWeight(mergeW, mergeB, devCopy, qkvScheme, 0),
                                        "HY-V3 ForwardGPU failed to split " + mergeQkvWeightName + ".\n");

                        int qWidth = num_attention_heads * head_dim;
                        DivisionScheme qScheme = ExtractHyV3FirstRangeScheme(qkvScheme);
                        threadTpKVHeadSchemes[i] = ExtractHyV3KVHeadScheme(qkvScheme, qWidth, head_dim);
                        Data &oB = GetThreadTensorParallelBias(oBiasName);
                        devCopy = devices;
                        AssertInFastLLM(SplitMultiCudaWeight(weight[oWeightName], oB, devCopy, qScheme, 1),
                                        "HY-V3 ForwardGPU failed to split " + oWeightName + ".\n");

                        if (IsDenseMlpLayer(i)) {
                            std::string gateupWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
                            std::string downWeightName = "model.layers." + std::to_string(i) + ".mlp.down_proj.weight";
                            AssertInFastLLM(weight.weight.find(gateupWeightName) != weight.weight.end(),
                                            "HY-V3 ForwardGPU requires merged dense gateup weight.\n");
                            AssertInFastLLM(weight.weight.find(downWeightName) != weight.weight.end(),
                                            "HY-V3 ForwardGPU requires dense down weight.\n");

                            Data &gateup = weight[gateupWeightName];
                            Data &gateupBias = GetThreadTensorParallelBias(gateupWeightName + ".tp_bias");
                            gateup.tpLinearType = TP_LINEAR_ROW;
                            gateup.tpPackType = TP_PACK_GATEUP;
                            devCopy = devices;
                            DivisionScheme gateScheme = BuildMultiCudaRowSplitScheme(gateup, devCopy, ratios);
                            AssertInFastLLM(SplitMultiCudaWeight(gateup, gateupBias, devCopy, gateScheme, 0),
                                            "HY-V3 ForwardGPU failed to split " + gateupWeightName + ".\n");

                            Data &down = weight[downWeightName];
                            Data &downBias = GetThreadTensorParallelBias(downWeightName + ".tp_bias");
                            down.tpLinearType = TP_LINEAR_COLUMN;
                            DivisionScheme downScheme = ExtractHyV3FirstRangeScheme(gateScheme);
                            devCopy = devices;
                            AssertInFastLLM(SplitMultiCudaWeight(down, downBias, devCopy, downScheme, 1),
                                            "HY-V3 ForwardGPU failed to split " + downWeightName + ".\n");
                            continue;
                        }

                        std::string sharedPrefix = GetSharedMlpPrefix(i);
                        if (!sharedPrefix.empty() && !HyV3LayerUsesMappedNonCudaMoe(this, i)) {
                            std::string gateupWeightName = sharedPrefix + "gateup_proj.weight";
                            std::string downWeightName = sharedPrefix + "down_proj.weight";
                            Data &gateup = weight[gateupWeightName];
                            Data &gateupBias = GetThreadTensorParallelBias(gateupWeightName + ".tp_bias");
                            gateup.tpLinearType = TP_LINEAR_ROW;
                            gateup.tpPackType = TP_PACK_GATEUP;
                            devCopy = devices;
                            DivisionScheme gateScheme = BuildMultiCudaRowSplitScheme(gateup, devCopy, ratios);
                            AssertInFastLLM(SplitMultiCudaWeight(gateup, gateupBias, devCopy, gateScheme, 0),
                                            "HY-V3 ForwardGPU failed to split " + gateupWeightName + ".\n");

                            Data &down = weight[downWeightName];
                            Data &downBias = GetThreadTensorParallelBias(downWeightName + ".tp_bias");
                            down.tpLinearType = TP_LINEAR_COLUMN;
                            DivisionScheme downScheme = ExtractHyV3FirstRangeScheme(gateScheme);
                            devCopy = devices;
                            AssertInFastLLM(SplitMultiCudaWeight(down, downBias, devCopy, downScheme, 1),
                                            "HY-V3 ForwardGPU failed to split " + downWeightName + ".\n");
                        }

                        if (HyV3LayerUsesMappedNonCudaMoe(this, i)) {
                            continue;
                        }

                        if (HasFusedMoeWeights(i)) {
                            continue;
                        }

                        DivisionScheme gateScheme;
                        for (int j = 0; j < this->num_experts; j++) {
                            std::string gateupWeightName = "model.layers." + std::to_string(i) + ".mlp.experts." +
                                                            std::to_string(j) + ".gateup_proj.weight";
                            std::string downWeightName = "model.layers." + std::to_string(i) + ".mlp.experts." +
                                                          std::to_string(j) + ".down_proj.weight";
                            AssertInFastLLM(weight.weight.find(gateupWeightName) != weight.weight.end(),
                                            "HY-V3 ForwardGPU requires merged expert gateup weight.\n");
                            AssertInFastLLM(weight.weight.find(downWeightName) != weight.weight.end(),
                                            "HY-V3 ForwardGPU requires expert down weight.\n");

                            Data &gateup = weight[gateupWeightName];
                            Data &gateupBias = GetThreadTensorParallelBias(gateupWeightName + ".tp_bias");
                            gateup.tpLinearType = TP_LINEAR_ROW;
                            gateup.tpPackType = TP_PACK_GATEUP;
                            devCopy = devices;
                            gateScheme = BuildMultiCudaRowSplitScheme(gateup, devCopy, ratios);
                            AssertInFastLLM(SplitMultiCudaWeight(gateup, gateupBias, devCopy, gateScheme, 0),
                                            "HY-V3 ForwardGPU failed to split " + gateupWeightName + ".\n");

                            Data &down = weight[downWeightName];
                            Data &downBias = GetThreadTensorParallelBias(downWeightName + ".tp_bias");
                            down.tpLinearType = TP_LINEAR_COLUMN;
                            DivisionScheme downScheme = ExtractHyV3FirstRangeScheme(gateScheme);
                            devCopy = devices;
                            AssertInFastLLM(SplitMultiCudaWeight(down, downBias, devCopy, downScheme, 1),
                                            "HY-V3 ForwardGPU failed to split " + downWeightName + ".\n");
                        }
                    }

                    if (useFusedMoeWeights) {
                        PrepareFusedMoeWeightsForDevices(devices, ratios);
                    }
                    if (anyLayerNeedsCudaMoeCache()) {
                        fillMoeCache(threadTpMoeWeights, threadTpMoeBiass, true);
                    } else {
                        threadTpMoeWeights.clear();
                        threadTpMoeBiass.clear();
                    }

                    Data &lmHeadBias = GetThreadTensorParallelBias("lm_head.weight.tp_bias");
                    std::vector<int> devCopy = devices;
                    threadTpLmHeadScheme = BuildMultiCudaRowSplitScheme(lmHead, devCopy, ratios);
                    AssertInFastLLM(SplitMultiCudaWeight(lmHead, lmHeadBias, devCopy, threadTpLmHeadScheme, 0),
                                    "HY-V3 ForwardGPU failed to split lm_head.weight.\n");

                    threadTpPreparedDevices = devices;
                    threadTpPreparedRatios = ratios;
                    threadTpWeightsPrepared.store(true, std::memory_order_release);
                }
                usePreparedThreadTpSchemes();
            }
        } else {
            std::lock_guard<std::mutex> guard(threadTpWeightPrepareLock);
            if (!singleGpuWeightsPrepared.load(std::memory_order_relaxed) ||
                !hasMoeCache(singleGpuMoeWeights, singleGpuMoeBiass)) {
                int device = devices[0];
                for (int i = 0; i < block_cnt; i++) {
                    std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
                    std::string gateWeightName = GetMoeGateWeightName(i);
                    AssertInFastLLM(weight.weight.find(mergeQkvWeightName) != weight.weight.end(),
                                    "HY-V3 ForwardGPU requires merged qkv weight.\n");
                    if (!IsDenseMlpLayer(i)) {
                        AssertInFastLLM(weight.weight.find(gateWeightName) != weight.weight.end(),
                                        "HY-V3 ForwardGPU requires router gate weight.\n");
                    }

                    Data &mergeW = weight[mergeQkvWeightName];
                    mergeW.tpPackType = TP_PACK_QKV;
                    mergeW.tpQHeads = num_attention_heads;
                    mergeW.tpKVHeads = num_key_value_heads;
                    mergeW.tpHeadDim = head_dim;

                    if (IsDenseMlpLayer(i)) {
                        continue;
                    }

                    if (HyV3LayerUsesMappedNonCudaMoe(this, i)) {
                        continue;
                    }

                    if (HasFusedMoeWeights(i)) {
                        HyV3PrepareFusedMoeWeightForCuda(*moeGate3DWeights[i], device);
                        HyV3PrepareFusedMoeWeightForCuda(*moeUp3DWeights[i], device);
                        HyV3PrepareFusedMoeWeightForCuda(*moeDown3DWeights[i], device);
                        continue;
                    }
                    for (int j = 0; j < this->num_experts; j++) {
                        std::string gateupWeightName = "model.layers." + std::to_string(i) +
                            ".mlp.experts." + std::to_string(j) + ".gateup_proj.weight";
                        std::string downWeightName = "model.layers." + std::to_string(i) +
                            ".mlp.experts." + std::to_string(j) + ".down_proj.weight";
                        AssertInFastLLM(weight.weight.find(gateupWeightName) != weight.weight.end(),
                                        "HY-V3 ForwardGPU requires merged expert gateup weight.\n");
                        AssertInFastLLM(weight.weight.find(downWeightName) != weight.weight.end(),
                                        "HY-V3 ForwardGPU requires expert down weight.\n");
                        Data &gateup = weight[gateupWeightName];
                        Data &down = weight[downWeightName];
                        gateup.tpLinearType = TP_LINEAR_ROW;
                        gateup.tpPackType = TP_PACK_GATEUP;
                        down.tpLinearType = TP_LINEAR_COLUMN;
                        gateup.ToDevice(DataDevice::CUDA, {device}, true);
                        down.ToDevice(DataDevice::CUDA, {device}, true);
                    }
                }
                if (anyLayerNeedsCudaMoeCache()) {
                    fillMoeCache(singleGpuMoeWeights, singleGpuMoeBiass, false);
                } else {
                    singleGpuMoeWeights.clear();
                    singleGpuMoeBiass.clear();
                }
                singleGpuWeightsPrepared.store(true, std::memory_order_release);
            }
            localKvHeadSchemes.assign(block_cnt, DivisionScheme());
            for (int i = 0; i < block_cnt; i++) {
                localKvHeadSchemes[i][devices[0]].push_back({0, num_key_value_heads});
            }
            localLmHeadScheme[devices[0]].push_back({0, lmHead.dims[0]});
        }

        if (tensorParallel && !useCpuEmbedding) {
            PrepareMultiCudaReplicatedData(weight["model.embed_tokens.weight"], devices, true);
        }
        Data cpuEmbeddingHiddenStates;
        Data *precomputedHiddenStates = nullptr;
        if (useCpuEmbedding) {
            Data cpuInputIds;
            cpuInputIds.CopyFrom(inputIds);
            HyV3CpuEmbeddingDirect(cpuInputIds, weight["model.embed_tokens.weight"],
                                       cpuEmbeddingHiddenStates, computeType);
            PrepareHyV3CpuEmbeddingHiddenStates(cpuEmbeddingHiddenStates, devices, threadTpWorkerGroup);
            precomputedHiddenStates = &cpuEmbeddingHiddenStates;
        }
        std::vector<std::vector<std::pair<Data*, Data*> > > localPastKeyValues;
        if (tensorParallel) {
            localPastKeyValues.resize(devices.size());
            for (int r = 0; r < (int)devices.size(); r++) {
                int device = devices[r];
                localPastKeyValues[r].resize(pastKeyValues.size());
                for (int i = 0; i < (int)pastKeyValues.size(); i++) {
                    DataType keyCacheType = ResolveHyV3ThreadTpCacheType(
                        pastKeyValues[i].first->dataType, computeType);
                    DataType valueCacheType = ResolveHyV3ThreadTpCacheType(
                        pastKeyValues[i].second->dataType, computeType);
                    localPastKeyValues[r][i].first = EnsureHyV3ThreadTpLocalCache(
                        *pastKeyValues[i].first, device, keyCacheType);
                    localPastKeyValues[r][i].second = EnsureHyV3ThreadTpLocalCache(
                        *pastKeyValues[i].second, device, valueCacheType);
                }
            }
        } else {
            int device = devices[0];
            for (int i = 0; i < (int)pastKeyValues.size(); i++) {
                DataType keyCacheType = ResolveHyV3ThreadTpCacheType(
                    pastKeyValues[i].first->dataType, computeType);
                DataType valueCacheType = ResolveHyV3ThreadTpCacheType(
                    pastKeyValues[i].second->dataType, computeType);
                PrepareHyV3SingleCudaCache(*pastKeyValues[i].first, device, keyCacheType);
                PrepareHyV3SingleCudaCache(*pastKeyValues[i].second, device, valueCacheType);
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
                    SyncHyV3ThreadTpRootCacheMetaFromLocal(*pastKeyValues[idx].first, localKeyMeta,
                                                               devices, (*kvHeadSchemes)[i],
                                                               num_key_value_heads, head_dim);
                    SyncHyV3ThreadTpRootCacheMetaFromLocal(*pastKeyValues[idx].second, localValueMeta,
                                                               devices, (*kvHeadSchemes)[i],
                                                               num_key_value_heads, head_dim);
                }
            }
        }

        int vocabSize = lmHead.dims[0];
        bool allSimpleCudaSampling = true;
        int cudaSamplingTopK = 1;
        if (HyV3CanUseCudaFullLogitsSampling(generationConfigs, retLogits, batch,
                                                 allSimpleCudaSampling, cudaSamplingTopK)) {
            Data &fullCudaLogits = HyV3ThreadLocalCudaSamplingFullLogits();
            HyV3GatherShardLogitsToRootCuda(devices[0], devices, *lmHeadScheme,
                                                localLogits, batch, vocabSize,
                                                fullCudaLogits);
            void *oldExecutor = GetExecutor();
            Executor samplingExecutor;
            samplingExecutor.SetFirstDevice("cuda:" + std::to_string(devices[0]));
            SetCurrentThreadExecutor(&samplingExecutor);
            ResetLogitsOfEOS(batch, &fullCudaLogits, pastKeyValues, generationConfigs);
            SetCurrentThreadExecutor(oldExecutor);
            std::vector<int> lastRet = HyV3SampleFromRootCudaLogits(devices[0], fullCudaLogits, batch,
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
                            "HY-V3 ForwardGPU: local logits batch mismatch.\n");
            float *src = (float*)localLogits[r].cpuData;
            float *dst = (float*)fullLogits.cpuData;
            int localOffset = 0;
            auto schemeIt = lmHeadScheme->find(device);
            AssertInFastLLM(schemeIt != lmHeadScheme->end(),
                            "HY-V3 ForwardGPU: missing lm_head split scheme.\n");
            for (auto &range : schemeIt->second) {
                int len = range.second - range.first;
                AssertInFastLLM(range.first >= 0 && range.second <= vocabSize &&
                                localOffset + len <= localVocab,
                                "HY-V3 ForwardGPU: invalid lm_head split range.\n");
                for (int b = 0; b < batch; b++) {
                    memcpy(dst + (long long)b * vocabSize + range.first,
                           src + (long long)b * localVocab + localOffset,
                           (size_t)len * sizeof(float));
                }
                localOffset += len;
            }
        }

        ResetLogitsOfEOS(batch, &fullLogits, pastKeyValues, generationConfigs);

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

    std::vector <int> HyV3Model::ForwardV2ThreadTensorParallel(
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

    std::vector <int> HyV3Model::ForwardV2(int batch,
        const Data &inputIds,
        const std::vector <Data*> &attentionMask,
        const std::vector <Data*> &positionIds,
        const std::vector <int> &seqLens,
        std::vector <std::pair <Data*, Data*> > &pastKeyValues,
        const std::vector <GenerationConfig> &generationConfigs,
        const LastTokensManager &lastTokens,
        std::vector <std::vector <float>*> *retLogits) {
        if (IsThreadTensorParallelEnabled()) {
            return ForwardGPU(batch, inputIds, attentionMask, positionIds, seqLens,
                              pastKeyValues, generationConfigs, lastTokens, retLogits);
        }

        int seqLen = inputIds.dims[1];

        Data hiddenStates;
        Data attenInput;
        Data q, qkv;
        Data attenLastOutput;
        Data w1, w2, w3, routerLogits, gate, attenPart, moePart, moeFinal, sharedGate;
        Data tempInput, tempOutput;
        Data routerLogitsTemp;
        Data moeInputTemp, moeOutputTemp;
        Data expertIndex, expertScore;

        std::vector<Data*> batchPastKeys;
        std::vector<Data*> batchPastValues;
        batchPastKeys.resize(batch);
        batchPastValues.resize(batch);

        Data allPositionIds;
        Data qSizes, pageSizes, pageIndexs, lastPageLens;
        Data insertIndexs, insertPositions;
        Data attenOutput;
        bool generatedBatchDecodeParams = false;
        bool generatedAppendPagedCacheBatchParams = false;

        bool useFusedMoeWeights = false;
#ifdef USE_CUDA
        if (!HyV3DisableFusedMoe() &&
            HyV3GenericForwardMayUseFusedMoe(this->deviceMap, this->moeDeviceMap)) {
            useFusedMoeWeights = TryBuildFusedMoeWeightsFromLoaded();
        }
#endif
        PrepareMoeWeights(false);

        bool all1 = true;
        for (int i = 0; i < batch; i++) {
            all1 &= (seqLens[i] == 1);
        }
        bool isPrefill = !all1;
        if (all1 && positionIds[0]->dataType == DataType::FLOAT32) {
            std::vector <float> vPositionIds;
            for (int b = 0; b < batch; b++) {
                vPositionIds.push_back(((float*)positionIds[b]->cpuData)[0]);
            }
            allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, vPositionIds));
        } else {
            std::vector <float> vPositionIds;
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < seqLens[b]; i++) {
                    vPositionIds.push_back(((float*)positionIds[b]->cpuData)[i]);
                }
            }
            allPositionIds.CopyFrom(Data(DataType::FLOAT32, {1, (int)vPositionIds.size()}, vPositionIds));
        }

        EmbeddingBlock((Data*)&inputIds,
            &this->weight["model.embed_tokens.weight"], &hiddenStates, this->dataType);

        for (int i = 0; i < block_cnt; i++) {
            ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
            std::string mergeQkvWeightName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.weight";
            std::string mergeQkvBiasName = "model.layers." + std::to_string(i) + ".self_attn.mergeqkv.bias";
            std::string qWeightName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.weight";
            std::string qBiasName = "model.layers." + std::to_string(i) + ".self_attn.q_proj.bias";
            std::string kWeightName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.weight";
            std::string kBiasName = "model.layers." + std::to_string(i) + ".self_attn.k_proj.bias";
            std::string vWeightName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.weight";
            std::string vBiasName = "model.layers." + std::to_string(i) + ".self_attn.v_proj.bias";
            std::string qNormName = "model.layers." + std::to_string(i) + ".self_attn.q_norm.weight";
            std::string kNormName = "model.layers." + std::to_string(i) + ".self_attn.k_norm.weight";
            std::string oWeightName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.weight";
            std::string oBiasName = "model.layers." + std::to_string(i) + ".self_attn.o_proj.bias";

            bool hasMergeQkv = (weight.weight.find(mergeQkvWeightName) != weight.weight.end());
            Data *mergeW = hasMergeQkv ? &weight[mergeQkvWeightName] : GetEmptyData();
            Data *mergeB = hasMergeQkv ? &weight[mergeQkvBiasName] : GetEmptyData();
            Data *qW = hasMergeQkv ? GetEmptyData() : &weight[qWeightName];
            Data *qB = hasMergeQkv ? GetEmptyData() : &weight[qBiasName];
            Data *kW = hasMergeQkv ? GetEmptyData() : &weight[kWeightName];
            Data *kB = hasMergeQkv ? GetEmptyData() : &weight[kBiasName];
            Data *vW = hasMergeQkv ? GetEmptyData() : &weight[vWeightName];
            Data *vB = hasMergeQkv ? GetEmptyData() : &weight[vBiasName];

            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],
                    rms_norm_eps, attenInput);
            AttentionPagedBlock(
                &attenInput,
                mergeW, mergeB,
                qW, qB, kW, kB, vW, vB,
                GetEmptyData(), GetEmptyData(),
                &weight[qNormName], &weight[kNormName],
                &weight[oWeightName], &weight[oBiasName],
                &allPositionIds,
                &pastKeyValues, &batchPastKeys, &batchPastValues,
                &qkv, &q, &attenOutput, &attenLastOutput,
                &insertIndexs, &insertPositions,
                &qSizes, &pageSizes, &pageIndexs, &lastPageLens,
                &generatedAppendPagedCacheBatchParams, &generatedBatchDecodeParams,
                batch, block_cnt, i,
                seqLens,
                num_attention_heads, num_key_value_heads, head_dim,
                rotary_dim, rms_norm_eps,
                rope_base, rope_factor, max_positions,
                rope_type,
                GetKVCacheInCPU(),
                isPrefill,
                &hiddenStates,
                true,
                false
            );

            RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".post_attention_layernorm.weight"], rms_norm_eps, attenInput);

            // 2. moe mlp
            {
                int batch = attenInput.dims[0], len = attenInput.dims[1];
                attenInput.Reshape({batch * len, attenInput.dims[2]});

                if (IsDenseMlpLayer(i)) {
                    std::string gateupWeightName = "model.layers." + std::to_string(i) + ".mlp.gateup_proj.weight";
                    std::string downWeightName = "model.layers." + std::to_string(i) + ".mlp.down_proj.weight";
                    if (weight.weight.find(gateupWeightName) != weight.weight.end()) {
                        if (CanRunLinearEx(LinearExType::ExSwiglu)) {
                            LinearEx(attenInput, weight[gateupWeightName], Data(), w1, LinearExType::ExSwiglu);
                        } else {
                            Linear(attenInput, weight[gateupWeightName], Data(), w3);
                            Swiglu(w3, w1);
                        }
                    } else {
                        if (CanRunLinearEx(LinearExType::ExSilu)) {
                            LinearEx(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"],
                                     Data(), w1, LinearExType::ExSilu);
                        } else {
                            Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.gate_proj.weight"],
                                   Data(), w1);
                            Silu(w1, w1);
                        }
                        Linear(attenInput, weight["model.layers." + std::to_string(i) + ".mlp.up_proj.weight"],
                               Data(), w3);
                        MulTo(w1, w3);
                    }
                    Linear(w1, weight[downWeightName], Data(), w2);
                    w2.Reshape(hiddenStates.dims);
                    AddTo(hiddenStates, w2);
                    continue;
                }

                // 这里是moe mlp
                std::string gateWeightName = GetMoeGateWeightName(i);
                std::string gateBiasName = GetMoeGateBiasName(i);
                Linear(attenInput, weight[gateWeightName], Data(), routerLogits, true);

                ToDataType(routerLogits, routerLogitsTemp, DataType::FLOAT32);
                bool needNorm = this->norm_topk_prob;
                ApplyRouterActivation(routerLogitsTemp);

                bool layerHasFusedMoe = useFusedMoeWeights && HasFusedMoeWeights(i);
                bool layerHasMergeMoe = !layerHasFusedMoe &&
                    i < (int)weights.size() && i < (int)biass.size() &&
                    weight.weight.find("model.layers." + std::to_string(i) + ".mlp.experts.0.gateup_proj.weight") != weight.weight.end() &&
                    CanRunMergeMOE(attenInput, biass[i]);

                if (layerHasFusedMoe || layerHasMergeMoe) {
                    SelectExpert(routerLogitsTemp, expertIndex, expertScore, this->num_experts_per_tok, needNorm,
                                this->routed_scaling_factor, weight.weight.find(gateBiasName) != weight.weight.end() ? &weight[gateBiasName] : nullptr);
                }

                std::string selectedMoeDevice = this->SelectMoeDeviceForLayer(i);
                std::string selectedMoeDeviceLower = selectedMoeDevice;
                std::transform(selectedMoeDeviceLower.begin(), selectedMoeDeviceLower.end(),
                               selectedMoeDeviceLower.begin(),
                               [](unsigned char c) { return (char)std::tolower(c); });
                bool selectedCudaMoe = selectedMoeDeviceLower == "cuda" ||
                    selectedMoeDeviceLower.rfind("cuda:", 0) == 0;
                bool selectedNumaMoe = selectedMoeDeviceLower == "numa" ||
                    selectedMoeDeviceLower.rfind("numa:", 0) == 0;

                if (layerHasFusedMoe && (selectedCudaMoe || selectedNumaMoe)) {
                    Data expertInput;
                    expertInput.CopyFrom(attenInput);
                    this->ApplyMoeDeviceMapForLayer(i);
#ifdef USE_CUDA
                    if (selectedCudaMoe) {
                        std::map<int, int> selectedCudaRatios;
                        std::vector<int> selectedCudaDevices = ParseDeviceIds(selectedMoeDevice, "cuda", selectedCudaRatios);
                        int selectedCudaDevice = selectedCudaDevices.empty() ? 0 : selectedCudaDevices[0];
                        HyV3PrepareFusedMoeWeightForCuda(*moeGate3DWeights[i], selectedCudaDevice);
                        HyV3PrepareFusedMoeWeightForCuda(*moeUp3DWeights[i], selectedCudaDevice);
                        HyV3PrepareFusedMoeWeightForCuda(*moeDown3DWeights[i], selectedCudaDevice);
                    }
#endif
                    FusedMOE(expertInput, expertIndex, expertScore,
                             *moeGate3DWeights[i], *moeUp3DWeights[i], *moeDown3DWeights[i],
                             w1, moeFinal, i, MoeGateSwiglu, 0.0f);
                } else {
                    AssertInFastLLM(layerHasMergeMoe,
                                    "HY-V3 has no runnable MoE weights for the selected device.\n");
                    this->ApplyMoeDeviceMapForLayer(i);
                    MergeMOEBlock(&attenInput, &expertIndex, &expertScore,
                        &weights[i], &biass[i],
                        &w1, &w2, &w3,
                        &tempInput, &tempOutput,
                        1.0f, &moeFinal, i,
                        this->dataType, this->moeAtype,
                        &moeInputTemp, &moeOutputTemp);
                }
                moeFinal.Reshape(hiddenStates.dims);
                ApplyDeviceMap(this->deviceMap, i + 1, block_cnt);
                AddTo(hiddenStates, moeFinal);
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

    bool HyV3Model::NeedAttentionMask(int qlen, int klen) {
        return false;
    }

    void HyV3Model::FillLLMInputsBatch(std::vector<std::vector<float>> &inputTokens,
                                          const std::vector<std::map<std::string, int>> &params,
                                          fastllm::Data &inputIds, fastllm::Data &attentionMask,
                                          fastllm::Data &positionIds) {
        inputIds.ToDevice(DataDevice::CPU);
        attentionMask.ToDevice(DataDevice::CPU);
        positionIds.ToDevice(DataDevice::CPU);

        int batch = inputTokens.size();
        int index = params[0].find("index")->second;
        if (index == 0) {
            std::vector <int> seqLens;
            seqLens.resize(batch);
            int maxLen = 0;
            for (int i = 0; i < batch; i++) {
                maxLen = std::max(maxLen, (int)inputTokens[i].size());
                seqLens[i] = (int)inputTokens[i].size();
            }

            std::vector <float> ids = std::vector <float> (batch * maxLen, 0);
            std::vector <float> vpids = std::vector <float> (batch * maxLen, 0);
            std::vector <float> vmask = std::vector <float> (batch * maxLen * maxLen, 0);
            for (int i = 0; i < batch; i++) {
                auto &tokens = inputTokens[i];
                int len = tokens.size(), base = maxLen - len;
                for (int j = 0; j < len; j++) {
                    ids[i * maxLen + base + j] = tokens[j];
                }
                for (int j = 0; j < len; j++) {
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
            std::vector <float> pids = std::vector <float> (batch);
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
            std::vector <float> vmasks = std::vector <float> (batch * maxLen, 0.0f);
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

    std::string HyV3Model::MakeInput(const std::string &history, int round, const std::string &input) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
    }

    std::string HyV3Model::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
        return (round == 0 ? pre_prompt : history) + user_role + input + bot_role + output + history_sep;
    }

    void HyV3Model::WarmUp() {
        printf("Warmup...\n");
        int oldTopk = this->num_experts_per_tok;
        if (!IsThreadTensorParallelEnabled()) {
            this->num_experts_per_tok = this->num_experts;
        }

        Data inputIds = Data(DataType::FLOAT32, {1, 1}, {1});
        Data attentionMask = Data(this->dataType, {1, 1}, {0});
        Data positionIds = Data(this->dataType, {1, 1}, {0, 0});

        std::vector <std::pair <Data, Data> > pastKeyValues;
        for (int i = 0; i < block_cnt; i++) {
            pastKeyValues.push_back(std::make_pair(Data(this->dataType),
                                                   Data(this->dataType)));
        }
        Forward(inputIds, attentionMask, positionIds, pastKeyValues);
        this->num_experts_per_tok = oldTopk;
        elementsInKVCachePerToken = (long long)block_cnt *
            (pastKeyValues[0].first.dims[0] * pastKeyValues[0].first.dims[2] +
             pastKeyValues[0].second.dims[0] * pastKeyValues[0].second.dims[2]);
        printf("finish.\n");
    }
}
