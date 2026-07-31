#include "laguna.h"

#include "json11.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace fastllm {
    namespace {
        static std::string LagunaGet(const std::map<std::string, std::string> &dicts,
                                     const std::string &key,
                                     const std::string &defaultValue = "") {
            auto it = dicts.find(key);
            return it == dicts.end() ? defaultValue : it->second;
        }

        static int LagunaGetInt(const std::map<std::string, std::string> &dicts,
                                const std::string &key, int defaultValue) {
            std::string value = LagunaGet(dicts, key);
            return value.empty() ? defaultValue : atoi(value.c_str());
        }

        static float LagunaGetFloat(const std::map<std::string, std::string> &dicts,
                                    const std::string &key, float defaultValue) {
            std::string value = LagunaGet(dicts, key);
            return value.empty() ? defaultValue : atof(value.c_str());
        }

        static bool LagunaGetBool(const std::map<std::string, std::string> &dicts,
                                  const std::string &key, bool defaultValue) {
            std::string value = LagunaGet(dicts, key);
            if (value.empty()) {
                return defaultValue;
            }
            std::transform(value.begin(), value.end(), value.begin(), ::tolower);
            return value == "true" || value == "1";
        }

        static std::vector<int> LagunaParseIntList(const std::string &value) {
            std::vector<int> result;
            std::string error;
            json11::Json parsed = json11::Json::parse(value, error);
            if (!error.empty() || !parsed.is_array()) {
                return result;
            }
            for (const auto &item : parsed.array_items()) {
                result.push_back(item.int_value());
            }
            return result;
        }

        static std::vector<std::string> LagunaParseStringList(const std::string &value) {
            std::vector<std::string> result;
            std::string error;
            json11::Json parsed = json11::Json::parse(value, error);
            if (!error.empty() || !parsed.is_array()) {
                return result;
            }
            for (const auto &item : parsed.array_items()) {
                result.push_back(item.string_value());
            }
            return result;
        }

        static bool LagunaFusedMoeDisabled() {
            const char *value = std::getenv("FASTLLM_LAGUNA_DISABLE_FUSED_MOE");
            if (value == nullptr) {
                return false;
            }
            std::string normalized(value);
            std::transform(normalized.begin(), normalized.end(), normalized.begin(), ::tolower);
            return normalized == "1" || normalized == "true" || normalized == "on";
        }

        static bool LagunaIsCudaDevice(const std::string &device) {
            return device.rfind("cuda", 0) == 0;
        }

        static bool LagunaMoeIsFusedFp8Type(DataType dataType) {
            return dataType == DataType::FP8_E4M3 ||
                   dataType == DataType::FP8_E4M3_BLOCK_128;
        }

        static std::string LagunaMoeExpertPrefix(int layer, int expert) {
            return "model.layers." + std::to_string(layer) +
                   ".moe.experts." + std::to_string(expert) + ".";
        }

        static std::string LagunaMoeFusedWeightName(int layer, const std::string &kind) {
            return "model.layers." + std::to_string(layer) +
                   ".moe.fused_experts." + kind + "_proj.weight";
        }

        static void LagunaCopyFusedWeightMeta(Data &dst, const Data &src,
                                              const std::string &name) {
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

        static bool LagunaCheckFp8ScaleRows(const Data &weight,
                                            int rowStart, int rows) {
            if (weight.dataType != DataType::FP8_E4M3) {
                return true;
            }
            if (weight.blockK <= 0 || weight.blockM <= 0 ||
                weight.scales.empty() || weight.dims.size() != 2) {
                return false;
            }
            int columns = weight.dims[1];
            int scaleColumns = (columns - 1) / weight.blockM + 1;
            int scaleOffset = (rowStart / weight.blockK) * scaleColumns;
            int scaleCount = ((rows - 1) / weight.blockK + 1) * scaleColumns;
            return rowStart >= 0 && rows > 0 &&
                   rowStart + rows <= weight.dims[0] &&
                   rowStart % weight.blockK == 0 &&
                   scaleOffset + scaleCount <= (int)weight.scales.size();
        }

        static void LagunaAppendFp8ScaleRows(Data &dst, const Data &src,
                                             int rowStart, int rows) {
            if (src.dataType != DataType::FP8_E4M3) {
                return;
            }
            AssertInFastLLM(LagunaCheckFp8ScaleRows(src, rowStart, rows),
                            "Laguna FusedMOE FP8 scale slice is out of bounds.");
            int scaleColumns = (src.dims[1] - 1) / src.blockM + 1;
            int scaleOffset = (rowStart / src.blockK) * scaleColumns;
            int scaleCount = ((rows - 1) / src.blockK + 1) * scaleColumns;
            dst.scales.insert(dst.scales.end(),
                              src.scales.begin() + scaleOffset,
                              src.scales.begin() + scaleOffset + scaleCount);
        }

        static void LagunaCopyFusedRows(Data &dst, int dstRowStart,
                                        Data &src, int srcRowStart, int rows) {
            AssertInFastLLM(dst.dims.size() == 3 && src.dims.size() == 2,
                            "Laguna FusedMOE row copy expects 3D destination and 2D source.");
            int columns = src.dims[1];
            AssertInFastLLM(dst.dims[2] == columns &&
                            srcRowStart >= 0 && rows > 0 &&
                            srcRowStart + rows <= src.dims[0] &&
                            dstRowStart >= 0 &&
                            dstRowStart + rows <= dst.dims[0] * dst.dims[1],
                            "Laguna FusedMOE row copy shape mismatch.");
            src.ToDevice(DataDevice::CPU);
            AssertInFastLLM(src.cpuData != nullptr && dst.cpuData != nullptr,
                            "Laguna FusedMOE row copy requires CPU buffers.");
            size_t bytesPerRow = GetDataBytes(src.dataType, 1, columns);
            memcpy(dst.cpuData + (size_t)dstRowStart * bytesPerRow,
                   src.cpuData + (size_t)srcRowStart * bytesPerRow,
                   (size_t)rows * bytesPerRow);
        }

        static bool LagunaCanBuildFusedMoeLayer(
                const std::unordered_map<std::string, Data> &allWeights,
                int layer, int numExperts) {
            const Data *firstGateup = nullptr;
            const Data *firstDown = nullptr;
            for (int expert = 0; expert < numExperts; expert++) {
                std::string prefix = LagunaMoeExpertPrefix(layer, expert);
                auto gateupIt = allWeights.find(prefix + "gateup_proj.weight");
                auto downIt = allWeights.find(prefix + "down_proj.weight");
                if (gateupIt == allWeights.end() || downIt == allWeights.end()) {
                    return false;
                }
                const Data &gateup = gateupIt->second;
                const Data &down = downIt->second;
                // Tensor-parallel expert weights already own per-device shards.
                // Rebuilding them as one fused 3D tensor would duplicate the
                // complete expert layer on the root GPU and can exhaust its
                // memory before the first request finishes.
                if (gateup.multiDeviceData || down.multiDeviceData ||
                    gateup.isDiskWeight || down.isDiskWeight ||
                    gateup.dims.size() != 2 || down.dims.size() != 2 ||
                    gateup.dims[0] <= 0 || (gateup.dims[0] & 1) != 0 ||
                    !LagunaMoeIsFusedFp8Type(gateup.dataType) ||
                    gateup.dataType != down.dataType) {
                    return false;
                }
                int inter = gateup.dims[0] / 2;
                int hidden = gateup.dims[1];
                if (down.dims[0] != hidden || down.dims[1] != inter ||
                    !LagunaCheckFp8ScaleRows(gateup, 0, inter) ||
                    !LagunaCheckFp8ScaleRows(gateup, inter, inter) ||
                    !LagunaCheckFp8ScaleRows(down, 0, hidden)) {
                    return false;
                }
                if (expert == 0) {
                    firstGateup = &gateup;
                    firstDown = &down;
                } else if (gateup.dataType != firstGateup->dataType ||
                           gateup.dims != firstGateup->dims ||
                           gateup.blockK != firstGateup->blockK ||
                           gateup.blockM != firstGateup->blockM ||
                           down.dataType != firstDown->dataType ||
                           down.dims != firstDown->dims ||
                           down.blockK != firstDown->blockK ||
                           down.blockM != firstDown->blockM) {
                    return false;
                }
            }
            return firstGateup != nullptr;
        }

        static void LagunaBuildFusedMoeWeight(
                std::unordered_map<std::string, Data> &allWeights,
                int layer, int numExperts, const std::string &kind,
                Data *&weightPtr) {
            std::string firstPrefix = LagunaMoeExpertPrefix(layer, 0);
            Data &firstGateup = allWeights[firstPrefix + "gateup_proj.weight"];
            Data &firstDown = allWeights[firstPrefix + "down_proj.weight"];
            int inter = firstGateup.dims[0] / 2;
            int hidden = firstGateup.dims[1];
            bool isDown = kind == "down";
            AssertInFastLLM(kind == "gate" || kind == "up" || isDown,
                            "Laguna FusedMOE weight kind is invalid.");

            std::string fusedName = LagunaMoeFusedWeightName(layer, kind);
            allWeights[fusedName] = isDown
                ? Data(firstDown.dataType, {numExperts, hidden, inter})
                : Data(firstGateup.dataType, {numExperts, inter, hidden});
            Data &fused = allWeights[fusedName];
            LagunaCopyFusedWeightMeta(fused, isDown ? firstDown : firstGateup,
                                      fusedName);
            fused.Allocate(false);
            fused.scales.clear();

            for (int expert = 0; expert < numExperts; expert++) {
                std::string prefix = LagunaMoeExpertPrefix(layer, expert);
                Data &gateup = allWeights[prefix + "gateup_proj.weight"];
                Data &down = allWeights[prefix + "down_proj.weight"];
                if (kind == "gate") {
                    LagunaCopyFusedRows(fused, expert * inter,
                                        gateup, 0, inter);
                    LagunaAppendFp8ScaleRows(fused, gateup, 0, inter);
                } else if (kind == "up") {
                    LagunaCopyFusedRows(fused, expert * inter,
                                        gateup, inter, inter);
                    LagunaAppendFp8ScaleRows(fused, gateup, inter, inter);
                } else {
                    LagunaCopyFusedRows(fused, expert * hidden,
                                        down, 0, hidden);
                    LagunaAppendFp8ScaleRows(fused, down, 0, hidden);
                }
            }
            weightPtr = &fused;
        }

        struct LagunaPrefixSnapshotTensor {
            bool valid = false;
            bool multiDevice = false;
            Data data;
            DataDevice targetDevice = DataDevice::CPU;
            std::vector<int> targetDeviceIds;
            std::vector<int> targetDims;
            TensorParallelLayoutType targetTpLayout = TP_LAYOUT_NONE;
            int targetTpAxis = -1;
            std::vector<int> targetTpGlobalDims;
            std::map<int, std::vector<std::pair<int, int> > > targetTpRanges;
            std::map<int, std::unique_ptr<LagunaPrefixSnapshotTensor> > locals;
        };

        struct LagunaPrefixSnapshotLayer {
            bool sliding = false;
            LagunaPrefixSnapshotTensor key;
            LagunaPrefixSnapshotTensor value;
        };

        struct LagunaPrefixSnapshot {
            int cachedLen = 0;
            long long timestamp = 0;
            std::vector<int> tokens;
            std::vector<std::unique_ptr<LagunaPrefixSnapshotLayer> > layers;
        };

        static std::mutex &LagunaPrefixSnapshotsMutex() {
            static auto *mutex = new std::mutex();
            return *mutex;
        }

        static std::map<const LagunaModel*,
                        std::vector<std::unique_ptr<LagunaPrefixSnapshot> > >
                &LagunaPrefixSnapshots() {
            static auto *snapshots = new std::map<
                const LagunaModel*,
                std::vector<std::unique_ptr<LagunaPrefixSnapshot> > >();
            return *snapshots;
        }

        static long long &LagunaPrefixSnapshotTimestamp() {
            static auto *timestamp = new long long(0);
            return *timestamp;
        }

        static bool LagunaTrueString(const char *value) {
            if (value == nullptr || value[0] == 0) {
                return true;
            }
            std::string normalized(value);
            std::transform(normalized.begin(), normalized.end(),
                           normalized.begin(), ::tolower);
            return normalized == "1" || normalized == "true" ||
                   normalized == "on" || normalized == "yes";
        }

        static bool LagunaPrefixCacheEnabled() {
            return LagunaTrueString(std::getenv("FASTLLM_PREFIX_CACHE"));
        }

        static int LagunaPrefixSnapshotMaxRecords() {
            const char *value =
                std::getenv("FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_RECORDS");
            if (value == nullptr || value[0] == 0) {
                return 8;
            }
            char *end = nullptr;
            long parsed = std::strtol(value, &end, 10);
            return end == value ? 8 : std::max(1, (int)parsed);
        }

        static int LagunaCacheTokenLen(const Data &cache) {
            if (cache.isPagedKVCache) {
                if (cache.pageIndex.empty()) {
                    return 0;
                }
                return ((int)cache.pageIndex.size() - 1) * cache.pageLen +
                       cache.lastPageLen;
            }
            return cache.dims.size() > 1 ? cache.dims[1] : 0;
        }

        static bool LagunaCopySinglePrefixTensor(
                const Data &source, int begin, int end,
                LagunaPrefixSnapshotTensor &target) {
            target.valid = false;
            target.multiDevice = false;
            if (source.multiDeviceData || begin < 0 || end <= begin) {
                return false;
            }
            target.targetDevice = source.dataDevice;
            target.targetDeviceIds = source.dataDeviceIds;

            if (source.isPagedKVCache) {
                Data *storage = (Data*)source.pagedKVCacheData;
                int physicalLen = LagunaCacheTokenLen(source);
                if (storage == nullptr || storage->dims.size() < 4 ||
                    source.pageLen <= 0 || source.pageIndex.empty() ||
                    end > physicalLen) {
                    return false;
                }

                // A paged cache cannot express an offset into its first page.
                // Retain the preceding partial page as well; Laguna's windowLeft
                // still limits attention to the final sliding_window tokens.
                int firstPage = begin / source.pageLen;
                int endPage = (end + source.pageLen - 1) / source.pageLen;
                if (firstPage < 0 || endPage <= firstPage ||
                    endPage > (int)source.pageIndex.size()) {
                    return false;
                }
                int heldStart = firstPage * source.pageLen;
                int heldLen = end - heldStart;
                int heldLastPageLen = end - (endPage - 1) * source.pageLen;
                if (heldLastPageLen <= 0) {
                    heldLastPageLen = source.pageLen;
                }

                target.data.dataType = source.dataType;
                target.data.UpdateUnitSize();
                target.data.dataDevice = storage->dataDevice;
                target.data.dataDeviceIds = source.dataDeviceIds;
                target.data.Resize(
                    {storage->dims[2], heldLen, storage->dims[3]});
                target.data.isKVCache = true;
                target.data.isPagedKVCache = true;
                target.data.pagedKVCacheData = source.pagedKVCacheData;
                target.data.pageLen = source.pageLen;
                target.data.pageIndex.assign(
                    source.pageIndex.begin() + firstPage,
                    source.pageIndex.begin() + endPage);
                target.data.lastPageLen = heldLastPageLen;
                target.data.pagedKVCacheData->Pick(target.data.pageIndex);
                target.valid = true;
                return true;
            }

            if (source.dims.size() < 3 || end > source.dims[1]) {
                return false;
            }
            Data sliced;
            Split(source, 1, begin, end, sliced);
            target.data.CopyFrom(sliced);
            target.data.isKVCache = true;
            target.data.isPagedKVCache = false;
            target.data.pagedKVCacheData = nullptr;
            target.data.pageIndex.clear();
            target.data.lastPageLen = 0;
            target.data.ToDevice(DataDevice::CPU, true);
            target.data.lockInCPU = true;
            target.valid = target.data.cpuData != nullptr;
            return target.valid;
        }

        static bool LagunaCopyPrefixTensor(
                const Data &source, int begin, int end,
                LagunaPrefixSnapshotTensor &target) {
            target.valid = false;
            target.multiDevice = source.multiDeviceData;
            target.targetDevice = source.dataDevice;
            target.targetDeviceIds = source.dataDeviceIds;
            target.targetDims = source.dims;
            target.targetTpLayout = source.tpLayout;
            target.targetTpAxis = source.tpAxis;
            target.targetTpGlobalDims = source.tpGlobalDims;
            target.targetTpRanges = source.tpRanges;
            target.locals.clear();

            if (!source.multiDeviceData) {
                return LagunaCopySinglePrefixTensor(source, begin, end, target);
            }

            std::vector<int> devices = source.dataDeviceIds;
            if (devices.empty()) {
                for (const auto &it : source.multiDeviceDatas) {
                    devices.push_back(it.first);
                }
            }
            int heldLen = -1;
            for (int device : devices) {
                auto it = source.multiDeviceDatas.find(device);
                if (it == source.multiDeviceDatas.end() || it->second == nullptr) {
                    return false;
                }
                std::unique_ptr<LagunaPrefixSnapshotTensor> local(
                    new LagunaPrefixSnapshotTensor());
                if (!LagunaCopySinglePrefixTensor(
                        *it->second, begin, end, *local)) {
                    return false;
                }
                int localLen = local->data.dims.size() > 1
                    ? local->data.dims[1] : 0;
                if (heldLen < 0) {
                    heldLen = localLen;
                } else if (localLen != heldLen) {
                    return false;
                }
                target.locals[device] = std::move(local);
            }
            if (target.locals.empty() || heldLen <= 0) {
                return false;
            }
            if (target.targetDims.size() > 1) {
                target.targetDims[1] = heldLen;
            }
            if (target.targetTpGlobalDims.size() > 1) {
                target.targetTpGlobalDims[1] = heldLen;
            }
            target.valid = true;
            return true;
        }

        static void LagunaClearPrefixTensor(Data &target) {
            std::set<std::pair<PagedCacheManager*, int> > releasedPages;
            auto releasePages = [&](Data &cache) {
                if (!cache.isPagedKVCache || cache.pagedKVCacheData == nullptr) {
                    cache.pageIndex.clear();
                    cache.lastPageLen = 0;
                    return;
                }
                std::vector<int> uniquePages;
                for (int page : cache.pageIndex) {
                    auto key = std::make_pair(cache.pagedKVCacheData, page);
                    if (releasedPages.insert(key).second) {
                        uniquePages.push_back(page);
                    }
                }
                if (!uniquePages.empty()) {
                    cache.pagedKVCacheData->ReleasePageIndices(uniquePages);
                }
                cache.pageIndex.clear();
                cache.lastPageLen = 0;
            };

            releasePages(target);
            for (auto &it : target.multiDeviceDatas) {
                if (it.second != nullptr) {
                    releasePages(*it.second);
                    delete it.second;
                }
            }
            target.multiDeviceDatas.clear();
            target.multiDeviceData = false;
            target.FreeSpace();
            target.dims.clear();
            target.strides.clear();
            target.isPagedKVCache = false;
            target.pagedKVCacheData = nullptr;
            target.ClearTensorParallelLayout();
        }

        static bool LagunaRestorePrefixTensor(
                const LagunaPrefixSnapshotTensor &source, Data &target) {
            if (!source.valid) {
                return false;
            }
            long long cacheUid = target.cacheUid;

            if (source.multiDevice) {
                if (source.locals.empty() || source.targetDims.size() < 3) {
                    return false;
                }
                LagunaClearPrefixTensor(target);
                target.cacheUid = cacheUid;
                target.dataDevice = source.targetDevice;
                target.dataDeviceIds = source.targetDeviceIds;
                target.Resize(source.targetDims);
                target.multiDeviceData = true;
                target.tpLayout = source.targetTpLayout;
                target.tpAxis = source.targetTpAxis;
                target.tpGlobalDims = source.targetTpGlobalDims;
                target.tpRanges = source.targetTpRanges;
                target.isKVCache = true;

                Data *firstLocal = nullptr;
                for (const auto &it : source.locals) {
                    Data *local = new Data();
                    local->cacheUid = cacheUid;
                    if (!LagunaRestorePrefixTensor(*it.second, *local)) {
                        delete local;
                        LagunaClearPrefixTensor(target);
                        target.cacheUid = cacheUid;
                        return false;
                    }
                    target.multiDeviceDatas[it.first] = local;
                    if (firstLocal == nullptr) {
                        firstLocal = local;
                    }
                }
                if (firstLocal == nullptr) {
                    LagunaClearPrefixTensor(target);
                    target.cacheUid = cacheUid;
                    return false;
                }
                target.dataType = firstLocal->dataType;
                target.UpdateUnitSize();
                target.isPagedKVCache = firstLocal->isPagedKVCache;
                target.pagedKVCacheData = firstLocal->pagedKVCacheData;
                target.pageLen = firstLocal->pageLen;
                target.pageIndex = firstLocal->pageIndex;
                target.lastPageLen = firstLocal->lastPageLen;
                target.cudaData = nullptr;
                return true;
            }

            if (source.data.dims.size() < 3) {
                return false;
            }
            LagunaClearPrefixTensor(target);
            target.cacheUid = cacheUid;
            if (source.data.isPagedKVCache) {
                target.dataType = source.data.dataType;
                target.UpdateUnitSize();
                target.dataDevice = source.data.dataDevice;
                target.dataDeviceIds = source.data.dataDeviceIds;
                target.Resize(source.data.dims);
                target.isKVCache = true;
                target.isPagedKVCache = true;
                target.pagedKVCacheData = source.data.pagedKVCacheData;
                target.pageLen = source.data.pageLen;
                target.pageIndex = source.data.pageIndex;
                target.lastPageLen = source.data.lastPageLen;
                target.pagedKVCacheData->Pick(target.pageIndex);
                return true;
            }

            target.CopyFrom(source.data);
            target.cacheUid = cacheUid;
            target.isKVCache = true;
            target.isPagedKVCache = false;
            target.pagedKVCacheData = nullptr;
            target.pageIndex.clear();
            target.lastPageLen = 0;
            target.lockInCPU = source.targetDevice == DataDevice::CPU;
            if (source.targetDevice == DataDevice::CUDA) {
                target.ToDevice(DataDevice::CUDA,
                                source.targetDeviceIds.empty()
                                    ? std::vector<int>{0}
                                    : source.targetDeviceIds,
                                true);
            }
            return !target.dims.empty();
        }

        static const LagunaPrefixSnapshot *LagunaFindPrefixSnapshotLocked(
                const LagunaModel *model, const std::vector<int> &tokens,
                int maxCachedLen, int exactLen = -1) {
            auto allIt = LagunaPrefixSnapshots().find(model);
            if (allIt == LagunaPrefixSnapshots().end()) {
                return nullptr;
            }
            const LagunaPrefixSnapshot *best = nullptr;
            for (const auto &item : allIt->second) {
                const LagunaPrefixSnapshot *snapshot = item.get();
                if (snapshot == nullptr || snapshot->cachedLen <= 0 ||
                    snapshot->cachedLen > maxCachedLen ||
                    snapshot->cachedLen > (int)tokens.size() ||
                    (exactLen >= 0 && snapshot->cachedLen != exactLen) ||
                    (int)snapshot->tokens.size() != snapshot->cachedLen ||
                    !std::equal(snapshot->tokens.begin(), snapshot->tokens.end(),
                                tokens.begin())) {
                    continue;
                }
                if (best == nullptr || snapshot->cachedLen > best->cachedLen ||
                    (snapshot->cachedLen == best->cachedLen &&
                     snapshot->timestamp > best->timestamp)) {
                    best = snapshot;
                }
            }
            return best;
        }

        static void LagunaErasePrefixSnapshots(const LagunaModel *model) {
            std::lock_guard<std::mutex> guard(LagunaPrefixSnapshotsMutex());
            LagunaPrefixSnapshots().erase(model);
        }
    }

    LagunaModel::LagunaModel() {
        this->model_type = "laguna";
        this->model_struct = "laguna";
        this->use_new_engine = true;
        // Keep sliding-attention prefill inside the CUDA kernel's small-mask
        // path.  Larger chunks switch to a different reduction path and can
        // amplify FP16 round-off enough that a restored sliding tail chooses a
        // different next token from a fresh prefill.  256 is page-aligned and
        // also bounds the dense local-mask workspace.
        this->defaultChunkedPrefillSize = 256;
        this->pre_prompt.clear();
        this->user_role.clear();
        this->bot_role.clear();
        this->history_sep.clear();

        this->weight.linearNames.insert("model.layers.*.moe.experts.*.gate_proj.weight");
        this->weight.linearNames.insert("model.layers.*.moe.experts.*.up_proj.weight");
        this->weight.linearNames.insert("model.layers.*.moe.experts.*.down_proj.weight");
    }

    LagunaModel::~LagunaModel() {
        ShutdownRuntime();
        LagunaErasePrefixSnapshots(this);
    }

    void LagunaModel::InitParams() {
        auto &dicts = this->weight.dicts;

        // Step3p5 uses different names for the same architectural parameters.
        // Populate aliases before invoking its common initialization.
        dicts["num_attention_groups"] =
            LagunaGet(dicts, "num_key_value_heads", "8");
        dicts["attention_other_setting.num_attention_groups"] =
            LagunaGet(dicts, "num_key_value_heads", "8");
        dicts["share_expert_dim"] =
            LagunaGet(dicts, "shared_expert_intermediate_size", "1024");
        dicts["moe_num_experts"] = LagunaGet(dicts, "num_experts", "0");
        dicts["moe_top_k"] = LagunaGet(dicts, "num_experts_per_tok", "0");
        dicts["norm_expert_weight"] = LagunaGet(dicts, "norm_topk_prob", "true");
        dicts["moe_router_scaling_factor"] =
            LagunaGet(dicts, "moe_routed_scaling_factor", "1.0");
        dicts["need_fp32_gate"] = "false";

        std::vector<int> headsPerLayer =
            LagunaParseIntList(LagunaGet(dicts, "num_attention_heads_per_layer"));
        std::vector<std::string> configuredLayerTypes =
            LagunaParseStringList(LagunaGet(dicts, "layer_types"));
        int configuredBaseHeads = LagunaGetInt(dicts, "num_attention_heads", 48);
        int configuredSlidingHeads = configuredBaseHeads;
        for (int i = 0; i < (int)headsPerLayer.size(); i++) {
            if (i < (int)configuredLayerTypes.size() &&
                configuredLayerTypes[i] == "sliding_attention") {
                configuredSlidingHeads = headsPerLayer[i];
                break;
            }
        }
        dicts["attention_other_setting.num_attention_heads"] =
            std::to_string(configuredSlidingHeads);
        dicts["rope_theta"] =
            LagunaGet(dicts, "rope_parameters.full_attention.rope_theta", "500000");

        Step3p5Model::InitParams();

        this->model_type = "laguna";
        this->model_struct = "laguna";
        std::vector<int> eosTokenIds =
            LagunaParseIntList(LagunaGet(dicts, "eos_token_id"));
        if (!eosTokenIds.empty()) {
            this->eos_token_id = eosTokenIds.front();
            this->eos_token_ids.insert(eosTokenIds.begin(), eosTokenIds.end());
        }
        this->block_cnt = LagunaGetInt(dicts, "num_hidden_layers", 48);
        this->embed_dim = LagunaGetInt(dicts, "hidden_size", 3072);
        this->head_dim = LagunaGetInt(dicts, "head_dim", 128);
        this->base_attention_heads = configuredBaseHeads;
        this->sliding_attention_heads = configuredSlidingHeads;
        this->base_key_value_heads =
            LagunaGetInt(dicts, "num_key_value_heads", 8);
        this->sliding_key_value_heads = this->base_key_value_heads;
        this->num_attention_heads = this->base_attention_heads;
        this->num_key_value_heads = this->base_key_value_heads;
        this->dense_intermediate_size =
            LagunaGetInt(dicts, "intermediate_size", 12288);
        this->moe_intermediate_size =
            LagunaGetInt(dicts, "moe_intermediate_size", 1024);
        this->shared_expert_intermediate_size =
            LagunaGetInt(dicts, "shared_expert_intermediate_size", 1024);
        this->num_experts = LagunaGetInt(dicts, "num_experts", 256);
        this->num_experts_per_tok =
            LagunaGetInt(dicts, "num_experts_per_tok", 10);
        this->norm_topk_prob = LagunaGetBool(dicts, "norm_topk_prob", true);
        this->routed_scaling_factor =
            LagunaGetFloat(dicts, "moe_routed_scaling_factor", 2.5f);
        this->rms_norm_eps = LagunaGetFloat(dicts, "rms_norm_eps", 1e-6f);
        this->sliding_window = LagunaGetInt(dicts, "sliding_window", 512);
        this->max_positions =
            LagunaGetInt(dicts, "max_position_embeddings", 1048576);
        this->use_moe_router_bias = true;
        this->need_fp32_gate = false;
        this->n_shared_experts = 1;

        if (!configuredLayerTypes.empty()) {
            this->layer_types = configuredLayerTypes;
        }
        if ((int)this->layer_types.size() != this->block_cnt) {
            this->layer_types.resize(this->block_cnt);
            for (int i = 0; i < this->block_cnt; i++) {
                this->layer_types[i] =
                    (i % 4 == 0) ? "full_attention" : "sliding_attention";
            }
        }

        if ((int)headsPerLayer.size() == this->block_cnt) {
            for (int i = 0; i < this->block_cnt; i++) {
                int expected = this->IsFullAttentionLayer(i)
                    ? this->base_attention_heads : this->sliding_attention_heads;
                AssertInFastLLM(headsPerLayer[i] == expected,
                                "Laguna has unsupported per-layer attention head counts.");
            }
        }

        this->moe_layers.clear();
        std::vector<int> mlpOnlyLayers =
            LagunaParseIntList(LagunaGet(dicts, "mlp_only_layers"));
        int decoderSparseStep =
            std::max(1, LagunaGetInt(dicts, "decoder_sparse_step", 1));
        for (int i = 0; i < this->block_cnt; i++) {
            if (std::find(mlpOnlyLayers.begin(), mlpOnlyLayers.end(), i) ==
                    mlpOnlyLayers.end() &&
                (i + 1) % decoderSparseStep == 0) {
                this->moe_layers.insert(i);
            }
        }

        this->fullRopeTheta = LagunaGetFloat(
            dicts, "rope_parameters.full_attention.rope_theta", 500000.0f);
        this->fullRopeFactor = LagunaGetFloat(
            dicts, "rope_parameters.full_attention.factor", 128.0f);
        std::string attentionFactor = LagunaGet(
            dicts, "rope_parameters.full_attention.attention_factor");
        this->fullRopeAttentionFactor = attentionFactor.empty()
            ? (this->fullRopeFactor <= 1.0f
                ? 1.0f : 0.1f * std::log(this->fullRopeFactor) + 1.0f)
            : atof(attentionFactor.c_str());
        this->fullRopeBetaFast = LagunaGetFloat(
            dicts, "rope_parameters.full_attention.beta_fast", 32.0f);
        this->fullRopeBetaSlow = LagunaGetFloat(
            dicts, "rope_parameters.full_attention.beta_slow", 1.0f);
        this->fullRopeOriginalMaxPosition = LagunaGetInt(
            dicts, "rope_parameters.full_attention.original_max_position_embeddings", 8192);
        this->fullRotaryDim = std::max(2, (int)(this->head_dim * LagunaGetFloat(
            dicts, "rope_parameters.full_attention.partial_rotary_factor", 0.5f)));
        this->slidingRopeTheta = LagunaGetFloat(
            dicts, "rope_parameters.sliding_attention.rope_theta", 10000.0f);
        this->slidingRotaryDim = std::max(2, (int)(this->head_dim * LagunaGetFloat(
            dicts, "rope_parameters.sliding_attention.partial_rotary_factor", 1.0f)));
        AssertInFastLLM(this->fullRotaryDim % 2 == 0 &&
                        this->slidingRotaryDim % 2 == 0,
                        "Laguna rotary dimensions should be even.");

        this->layer_rope_thetas.resize(this->block_cnt);
        this->layer_rotary_dims.resize(this->block_cnt);
        for (int i = 0; i < this->block_cnt; i++) {
            bool full = this->IsFullAttentionLayer(i);
            this->layer_rope_thetas[i] =
                full ? this->fullRopeTheta : this->slidingRopeTheta;
            this->layer_rotary_dims[i] =
                full ? this->fullRotaryDim : this->slidingRotaryDim;
        }
        this->rotary_dim = this->fullRotaryDim;

        AssertInFastLLM(LagunaGet(dicts, "gating", "per-head") == "per-head",
                        "Laguna currently requires per-head attention gating.");
        AssertInFastLLM(!LagunaGetBool(dicts, "swa_attention_sink_enabled", false),
                        "Laguna attention sinks are not supported.");
        AssertInFastLLM(LagunaGetFloat(dicts, "moe_router_logit_softcapping", 0.0f) == 0.0f,
                        "Laguna router logit soft-capping is not supported.");
        AssertInFastLLM(!LagunaGetBool(dicts, "moe_apply_router_weight_on_input", false),
                        "Laguna input-weighted expert routing is not supported.");

        for (int i : this->moe_layers) {
            for (int expert = 0; expert < this->num_experts; expert++) {
                std::string prefix = "model.layers." + std::to_string(i) +
                                     ".moe.experts." + std::to_string(expert) + ".";
                std::string gateName = prefix + "gate_proj.weight";
                std::string upName = prefix + "up_proj.weight";
                std::string downName = prefix + "down_proj.weight";
                std::string gateupName = prefix + "gateup_proj.weight";
                this->weightMergeRules.push_back(
                    WeightMergeRule({WeightMergeRuleSingle(
                        {gateName, upName}, gateupName, "linearSwiglu")})
                );
                this->moeLinears.insert(gateName);
                this->moeLinears.insert(upName);
                this->moeLinears.insert(downName);
                this->AddSpecialWeight(gateupName, "linearSwiglu", i);
                this->AddSpecialWeight(downName, "linearColumn", i);
            }
        }

        std::vector<float> zeroGate(std::max(
            this->base_attention_heads, this->sliding_attention_heads), 0.0f);
        this->softplusALog.CopyFrom(
            Data(DataType::FLOAT32, {(int)zeroGate.size()}, zeroGate));
        this->softplusDtBias.CopyFrom(
            Data(DataType::FLOAT32, {(int)zeroGate.size()}, zeroGate));

        this->initialized_add1 = false;
        this->moeWeightsPrepared = false;
    }

    std::string LagunaModel::MapTensorName(const std::string &name) const {
        std::string mapped = name;
        const std::string sharedMarker = ".mlp.shared_expert.";
        size_t position = mapped.find(sharedMarker);
        if (position != std::string::npos) {
            mapped.replace(position, sharedMarker.size(), ".share_expert.");
            return mapped;
        }

        const std::string correctionBias =
            ".mlp.experts.e_score_correction_bias";
        position = mapped.find(correctionBias);
        if (position != std::string::npos) {
            mapped.replace(position, correctionBias.size(), ".moe.router_bias");
            return mapped;
        }

        const std::string expertsMarker = ".mlp.experts.";
        position = mapped.find(expertsMarker);
        if (position != std::string::npos) {
            mapped.replace(position, expertsMarker.size(), ".moe.experts.");
            return mapped;
        }

        const std::string routerWeight = ".mlp.gate.weight";
        position = mapped.find(routerWeight);
        if (position != std::string::npos) {
            mapped.replace(position, routerWeight.size(), ".moe.gate.weight");
        }
        return mapped;
    }

    std::map<std::string, std::vector<std::pair<std::string, DataType>>>
    LagunaModel::GetTensorMap(const std::vector<std::string> &tensorNames) {
        std::vector<std::string> mappedNames;
        mappedNames.reserve(tensorNames.size());
        for (const auto &name : tensorNames) {
            mappedNames.push_back(MapTensorName(name));
        }
        auto mappedTensorMap = basellm::GetTensorMap(mappedNames);

        std::map<std::string, std::vector<std::pair<std::string, DataType>>> result;
        for (int i = 0; i < (int)tensorNames.size(); i++) {
            result[tensorNames[i]] = mappedTensorMap[mappedNames[i]];
        }
        return result;
    }

    void LagunaModel::PrepareMoeWeights() {
        if (this->moeWeightsPrepared) {
            return;
        }
        this->moeGateWeights.assign(this->block_cnt, {});
        this->moeUpWeights.assign(this->block_cnt, {});
        this->moeDownWeights.assign(this->block_cnt, {});
        this->moeGate3DWeights.assign(this->block_cnt, nullptr);
        this->moeUp3DWeights.assign(this->block_cnt, nullptr);
        this->moeDown3DWeights.assign(this->block_cnt, nullptr);
        this->weights.assign(this->block_cnt, {});
        this->biass.assign(this->block_cnt, {});

        for (int layer : this->moe_layers) {
            for (int expert = 0; expert < this->num_experts; expert++) {
                std::string prefix = LagunaMoeExpertPrefix(layer, expert);
                std::string gateName = prefix + "gate_proj.weight";
                std::string upName = prefix + "up_proj.weight";
                std::string gateupName = prefix + "gateup_proj.weight";
                if (this->weight.weight.find(gateupName) ==
                    this->weight.weight.end()) {
                    AssertInFastLLM(
                        this->weight.weight.find(gateName) != this->weight.weight.end() &&
                        this->weight.weight.find(upName) != this->weight.weight.end(),
                        "Laguna MoE gate/up expert weights are incomplete.");
                    Step3p5MakeGateUpWeight(this->weight.weight[gateupName],
                                            this->weight[gateName],
                                            this->weight[upName], gateupName);
                }
            }

            std::string selectedMoeDevice = this->SelectMoeDeviceForLayer(layer);
            bool useFusedCudaMoe = !LagunaFusedMoeDisabled() &&
                                   LagunaIsCudaDevice(selectedMoeDevice) &&
                                   LagunaCanBuildFusedMoeLayer(
                                       this->weight.weight, layer, this->num_experts);
            if (useFusedCudaMoe) {
                LagunaBuildFusedMoeWeight(this->weight.weight, layer,
                                          this->num_experts, "gate",
                                          this->moeGate3DWeights[layer]);
                LagunaBuildFusedMoeWeight(this->weight.weight, layer,
                                          this->num_experts, "up",
                                          this->moeUp3DWeights[layer]);
                LagunaBuildFusedMoeWeight(this->weight.weight, layer,
                                          this->num_experts, "down",
                                          this->moeDown3DWeights[layer]);
                for (int expert = 0; expert < this->num_experts; expert++) {
                    std::string prefix = LagunaMoeExpertPrefix(layer, expert);
                    this->weight.weight.erase(prefix + "gate_proj.weight");
                    this->weight.weight.erase(prefix + "up_proj.weight");
                    this->weight.weight.erase(prefix + "gateup_proj.weight");
                    this->weight.weight.erase(prefix + "down_proj.weight");
                }
                continue;
            }

            this->moeDownWeights[layer].resize(this->num_experts);
            this->weights[layer].reserve((this->num_experts + 1) * 2);
            this->biass[layer].reserve((this->num_experts + 1) * 2);
            this->weights[layer].push_back(nullptr);
            this->weights[layer].push_back(nullptr);
            this->biass[layer].push_back(nullptr);
            this->biass[layer].push_back(nullptr);
            for (int expert = 0; expert < this->num_experts; expert++) {
                std::string prefix = "model.layers." + std::to_string(layer) +
                                     ".moe.experts." + std::to_string(expert) + ".";
                std::string gateName = prefix + "gate_proj.weight";
                std::string upName = prefix + "up_proj.weight";
                std::string downName = prefix + "down_proj.weight";
                std::string gateupName = prefix + "gateup_proj.weight";
                AssertInFastLLM(
                    this->weight.weight.find(downName) != this->weight.weight.end(),
                    "Laguna MoE down expert weight is missing.");
                Data &gateup = this->weight[gateupName];
                Data &down = this->weight[downName];
                gateup.tpLinearType = TP_LINEAR_ROW;
                gateup.tpPackType = TP_PACK_GATEUP;
                down.tpLinearType = TP_LINEAR_COLUMN;
                this->moeDownWeights[layer][expert] = &down;
                this->weights[layer].push_back(&gateup);
                this->weights[layer].push_back(&down);
                this->biass[layer].push_back(nullptr);
                this->biass[layer].push_back(nullptr);
                this->weight.weight.erase(gateName);
                this->weight.weight.erase(upName);
            }
        }
        this->moeWeightsPrepared = true;
    }

    void LagunaModel::PrepareRuntimeWeights() {
        // Laguna stores ordinary RMSNorm scales. Step3p5 checkpoints store
        // residual deltas and therefore add one; doing that here would corrupt
        // every normalization layer.
        this->initialized_add1 = true;
    }

    DataType LagunaModel::GPUForwardComputeType() const {
        // Laguna activations overflow in the later layers when the whole
        // direct path is kept in FP16.  Preserve BF16 for the model compute
        // path; attention capability is handled independently.
        return DataType::BFLOAT16;
    }

    DataType LagunaModel::GPUForwardCacheType(
            int layer, DataType requestedType, DataType computeType) const {
        if (requestedType == DataType::FP8_E4M3) {
            return requestedType;
        }
        return this->IsFullAttentionLayer(layer)
            ? computeType : DataType::FLOAT16;
    }

    bool LagunaModel::GPUForwardUseYarnRope(int layer) const {
        return this->IsFullAttentionLayer(layer);
    }

    void LagunaModel::GPUForwardYarnRopeParams(
            int layer, float &factor, float &attentionFactor,
            float &correctionLow, float &correctionHigh) const {
        if (!this->IsFullAttentionLayer(layer)) {
            factor = 1.0f;
            attentionFactor = 1.0f;
            correctionLow = 0.0f;
            correctionHigh = 1.0f;
            return;
        }
        auto findCorrectionDim = [&](float rotations) {
            return (this->fullRotaryDim * std::log(
                        (float)this->fullRopeOriginalMaxPosition /
                        (rotations * 2.0f * (float)M_PI))) /
                   (2.0f * std::log(this->fullRopeTheta));
        };
        factor = this->fullRopeFactor;
        attentionFactor = this->fullRopeAttentionFactor;
        correctionLow = std::max(
            0.0f, std::floor(findCorrectionDim(this->fullRopeBetaFast)));
        correctionHigh = std::min(
            (float)this->fullRotaryDim - 1.0f,
            std::ceil(findCorrectionDim(this->fullRopeBetaSlow)));
        if (correctionLow == correctionHigh) {
            correctionHigh += 0.001f;
        }
    }

    int LagunaModel::GPUForwardAttentionWindowLeft(int layer) const {
        // FlashInfer's window_left excludes the current token, while Laguna's
        // sliding_window includes it.
        return this->IsFullAttentionLayer(layer)
            ? -1 : std::max(0, this->sliding_window - 1);
    }

    int LagunaModel::GPUForwardPagedCacheMaxPages(int layer) const {
        // The Step3p5 decode graph reuses one fixed page-index metadata buffer
        // for every attention layer.  Full-attention and sliding-window cache
        // managers therefore need the same page-number space while graph mode
        // is enabled; compact per-layer pools allocate different descending
        // page IDs and make graph replay invalid.  Eager mode keeps the smaller
        // sliding-window pools below.
        if (GetFastllmEnv().cudaGraph) {
            return -1;
        }

        int retainedTokens = this->GetKVCacheRetainedTokens(layer);
        if (retainedTokens < 0) {
            return -1;
        }

        const int pageLen = std::max(1, fastllm::GetPageLen());
        const int globalMaxTokens = fastllm::GetMaxTokens();
        const int globalPages = globalMaxTokens > 0
            ? std::max(1, (int)(((long long)globalMaxTokens + pageLen - 1) /
                                pageLen))
            : 300;
        int batchLimit = this->maxBatch > 0 ? this->maxBatch : 512;
        if (globalMaxTokens > 0) {
            batchLimit = std::min(
                batchLimit, std::max(1, globalMaxTokens / 128));
        }

        int prefillChunk = this->chunkedPrefillSize >= 0
            ? this->chunkedPrefillSize : this->defaultChunkedPrefillSize;
        if (prefillChunk <= 0) {
            prefillChunk = globalMaxTokens > 0 ? globalMaxTokens : 128;
        }
        // A cache is compacted after attention. Before the next compaction it
        // can contain the retained tail, one partially used page, and one
        // complete prefill chunk. Prefix snapshots pin one additional tail per
        // retained record, so reserve those pages in the compact eager pool as
        // well. CUDA Graph uses the global token-growing pool above.
        const long long peakTokensPerRequest =
            (long long)retainedTokens + prefillChunk + pageLen - 1;
        const long long pagesPerRequest =
            (peakTokensPerRequest + pageLen - 1) / pageLen;
        long long snapshotPages = 0;
        if (LagunaPrefixCacheEnabled()) {
            const long long pagesPerSnapshot =
                ((long long)retainedTokens + pageLen - 1) / pageLen;
            snapshotPages = pagesPerSnapshot *
                            LagunaPrefixSnapshotMaxRecords();
        }
        const long long boundedPages = std::max(
            1LL, pagesPerRequest * std::max(1, batchLimit) + snapshotPages);
        return (int)std::min<long long>(globalPages, boundedPages);
    }

    bool LagunaModel::GPUForwardUseMambaSoftplusGate(int layer) const {
        (void)layer;
        return true;
    }

    Data *LagunaModel::GPUForwardMambaSoftplusALog() {
        return &this->softplusALog;
    }

    Data *LagunaModel::GPUForwardMambaSoftplusDtBias() {
        return &this->softplusDtBias;
    }

    void LagunaModel::ApplyStepRotary(Data &input, const Data &positionIds, int layer) {
        if (this->IsFullAttentionLayer(layer)) {
            fastllm::YarnRopeEncoding(
                input, positionIds, this->fullRotaryDim,
                this->fullRopeTheta, this->fullRopeFactor,
                (float)this->fullRopeOriginalMaxPosition,
                this->fullRopeBetaFast, this->fullRopeBetaSlow,
                this->fullRopeAttentionFactor);
        } else {
            fastllm::RopeEncoding(
                input, positionIds, this->slidingRotaryDim,
                this->slidingRopeTheta, 1.0f);
        }
    }

    void LagunaModel::ApplyAttentionGateActivation(Data &gate, int layer) {
        (void)layer;
        this->softplusALog.ToDevice(gate.dataDevice, gate.dataDeviceIds);
        this->softplusDtBias.ToDevice(gate.dataDevice, gate.dataDeviceIds);
        // MambaSoftplus currently has no BF16 backend.  The attention gate is
        // small and well inside the FP16 range, so evaluate only this local
        // non-linearity in FP16 and convert it back to the model activation
        // type.  Residual activations remain BF16, which is required by
        // Laguna's late layers.
        if (gate.dataType == DataType::BFLOAT16) {
            Data gateFp16, negativeSoftplus, positiveSoftplus;
            ToDataType(gate, gateFp16, DataType::FLOAT16);
            MambaSoftplus(gateFp16, this->softplusALog,
                          this->softplusDtBias, negativeSoftplus);
            Mul(negativeSoftplus, -1.0f, positiveSoftplus);
            ToDataType(positiveSoftplus, gate, DataType::BFLOAT16);
            return;
        }
        Data negativeSoftplus;
        MambaSoftplus(gate, this->softplusALog, this->softplusDtBias,
                      negativeSoftplus);
        Mul(negativeSoftplus, -1.0f, gate);
    }

    bool LagunaModel::UsePagedAttention(int layer) const {
        // Full-attention layers keep the reusable long prefix in paged KV.
        // Sliding layers use the ordinary path because they require a local
        // mask and a bounded tail rather than a token-growing page chain.
        return this->IsFullAttentionLayer(layer);
    }

    int LagunaModel::GetKVCacheRetainedTokens(int layer) const {
        if (this->IsFullAttentionLayer(layer)) {
            return -1;
        }
        // A query needs at most window-1 cached tokens in addition to itself.
        // Prefix snapshots are captured while chunked prefill passes their
        // aligned boundary, so no extra post-hoc history needs to be retained.
        return std::max(1, this->sliding_window - 1);
    }

    bool LagunaModel::BoundedKVCacheUsesTokenGrowingStorage() const {
        // The direct decode graph keeps sliding-window page ids aligned with
        // full-attention layers.  The logical attention window stays bounded,
        // but the backing page pools grow with the advertised token limit.
        return GetFastllmEnv().cudaGraph;
    }

    bool LagunaModel::TryRecordPagedPrefixCacheExtra(ResponseContext *context) {
        if (context == nullptr || !LagunaPrefixCacheEnabled() ||
            (int)context->pastKeyValues.size() < this->block_cnt) {
            return false;
        }
        int pageLen = std::max(1, fastllm::GetPageLen());
        int currentLen = 0;
        for (int i = 0; i < this->block_cnt; i++) {
            if (!this->IsFullAttentionLayer(i)) {
                continue;
            }
            currentLen = LagunaCacheTokenLen(
                context->pastKeyValues[i].first);
            if (currentLen > 0) {
                break;
            }
        }
        if (currentLen <= 0 ||
            currentLen > (int)context->allTokens.size()) {
            return false;
        }
        int chunkLen = this->chunkedPrefillSize > 0
            ? this->chunkedPrefillSize
            : std::max(1, this->defaultChunkedPrefillSize);
        // Once the sliding window is active, restoring from the middle of a
        // prefill chunk changes the FP16 attention reduction shape.  Align the
        // reusable state to both page and chunk boundaries so fresh and cached
        // requests execute the same final chunk.  Small prompts can still use
        // the finer page boundary because no sliding history is discarded.
        int totalLen = (int)context->allTokens.size();
        int snapshotAlignment = totalLen > this->sliding_window
            ? std::lcm(pageLen, chunkLen)
            : pageLen;
        int cachedLen = totalLen / snapshotAlignment * snapshotAlignment;
        // Long prefills call this hook between chunks. Capture exactly once as
        // the forward pass reaches the final reusable chunk boundary. At the
        // end of a short, non-chunked prompt the complete cache still contains
        // the preceding page boundary and can be sliced directly.
        bool shortFinalSnapshot = totalLen <= this->sliding_window &&
                                  currentLen == totalLen;
        if (currentLen != cachedLen && !shortFinalSnapshot) {
            return false;
        }
        if (cachedLen <= 0 ||
            context->intParams["laguna_prefix_snapshot_len"] == cachedLen) {
            return false;
        }

        std::unique_ptr<LagunaPrefixSnapshot> snapshot(
            new LagunaPrefixSnapshot());
        snapshot->cachedLen = cachedLen;
        snapshot->tokens.assign(context->allTokens.begin(),
                                context->allTokens.begin() + cachedLen);
        snapshot->layers.resize(this->block_cnt);
        int desiredTokens = std::min(
            std::max(1, this->sliding_window - 1), cachedLen);
        int desiredStart = cachedLen - desiredTokens;
        for (int i = 0; i < this->block_cnt; i++) {
            if (this->IsFullAttentionLayer(i)) {
                continue;
            }
            const Data &key = context->pastKeyValues[i].first;
            const Data &value = context->pastKeyValues[i].second;
            int keyLen = LagunaCacheTokenLen(key);
            int valueLen = LagunaCacheTokenLen(value);
            int keyStart = currentLen - keyLen;
            int valueStart = currentLen - valueLen;
            int keyOffset = desiredStart - keyStart;
            int valueOffset = desiredStart - valueStart;
            snapshot->layers[i].reset(new LagunaPrefixSnapshotLayer());
            auto &layer = *snapshot->layers[i];
            layer.sliding = true;
            if (!LagunaCopyPrefixTensor(
                    key, keyOffset, keyOffset + desiredTokens,
                    layer.key) ||
                !LagunaCopyPrefixTensor(
                    value, valueOffset, valueOffset + desiredTokens,
                    layer.value)) {
                return false;
            }
        }

        {
            std::lock_guard<std::mutex> guard(LagunaPrefixSnapshotsMutex());
            auto &items = LagunaPrefixSnapshots()[this];
            for (auto it = items.begin(); it != items.end();) {
                if ((*it)->cachedLen == snapshot->cachedLen &&
                    (*it)->tokens == snapshot->tokens) {
                    it = items.erase(it);
                } else {
                    ++it;
                }
            }
            snapshot->timestamp = ++LagunaPrefixSnapshotTimestamp();
            items.push_back(std::move(snapshot));
            int maxRecords = LagunaPrefixSnapshotMaxRecords();
            while ((int)items.size() > maxRecords) {
                auto oldest = std::min_element(
                    items.begin(), items.end(),
                    [](const std::unique_ptr<LagunaPrefixSnapshot> &a,
                       const std::unique_ptr<LagunaPrefixSnapshot> &b) {
                        return a->timestamp < b->timestamp;
                    });
                items.erase(oldest);
            }
        }
        context->intParams["laguna_prefix_snapshot_len"] = cachedLen;
        return true;
    }

    int LagunaModel::QueryPagedPrefixCacheExtra(
            ResponseContext *context, int maxCachedLen) const {
        if (context == nullptr || maxCachedLen <= 0) {
            return 0;
        }
        if (!LagunaPrefixCacheEnabled()) {
            return 0;
        }
        std::lock_guard<std::mutex> guard(LagunaPrefixSnapshotsMutex());
        const LagunaPrefixSnapshot *snapshot = LagunaFindPrefixSnapshotLocked(
            this, context->currentTokens, maxCachedLen);
        return snapshot == nullptr ? 0 : snapshot->cachedLen;
    }

    bool LagunaModel::RestorePagedPrefixCacheExtra(
            ResponseContext *context, int cachedLen) const {
        if (context == nullptr || cachedLen <= 0 ||
            (int)context->pastKeyValues.size() < this->block_cnt) {
            return false;
        }
        std::lock_guard<std::mutex> guard(LagunaPrefixSnapshotsMutex());
        const LagunaPrefixSnapshot *snapshot = LagunaFindPrefixSnapshotLocked(
            this, context->currentTokens, cachedLen, cachedLen);
        if (snapshot == nullptr) {
            return false;
        }
        if ((int)snapshot->layers.size() < this->block_cnt) {
            return false;
        }
        for (int i = 0; i < this->block_cnt; i++) {
            if (this->IsFullAttentionLayer(i)) {
                continue;
            }
            const auto &layer = snapshot->layers[i];
            if (layer == nullptr || !layer->sliding ||
                !LagunaRestorePrefixTensor(
                    layer->key, context->pastKeyValues[i].first) ||
                !LagunaRestorePrefixTensor(
                    layer->value, context->pastKeyValues[i].second)) {
                return false;
            }
        }
        return true;
    }

    DataType LagunaModel::NonPagedAttentionDataType(DataType inputType) const {
        // Q/K/V are normalized and stay comfortably in FP16 range. Keeping
        // the attention/cache path in FP16 avoids both the unsupported BF16
        // generic kernel and the much larger FP32 KV cache, while the
        // overflow-prone residual stream remains BF16.
        return inputType == DataType::BFLOAT16 ? DataType::FLOAT16 : inputType;
    }

    bool LagunaModel::UseHostMergeMoe() const {
        return true;
    }

    Data *LagunaModel::PrepareAttentionMask(int layer, int pastLen, int qLen,
                                           DataType attentionType, Data *inputMask,
                                           Data &generatedMask) {
        (void)inputMask;
        if (this->IsFullAttentionLayer(layer) ||
            pastLen + qLen <= this->sliding_window) {
            // The regular Attention op supplies its own causal mask when no
            // explicit mask is present.
            return nullptr;
        }

        int keyLen = pastLen + qLen;
        std::vector<float> mask((size_t)qLen * keyLen, 0.0f);
        for (int query = 0; query < qLen; query++) {
            int absoluteQuery = pastLen + query;
            int firstVisible = std::max(0, absoluteQuery - this->sliding_window + 1);
            for (int key = 0; key < keyLen; key++) {
                if (key < firstVisible || key > absoluteQuery) {
                    mask[(size_t)query * keyLen + key] = 1.0f;
                }
            }
        }
        DataType maskType =
            attentionType == DataType::FLOAT32 ? DataType::FLOAT32 : DataType::FLOAT16;
        generatedMask.CopyFrom(Data(maskType, {qLen, keyLen}, mask));
        return &generatedMask;
    }

    bool LagunaModel::CanUseGPUForward() const {
        // The numerically safe BF16 direct path requires FlashInfer on SM80+.
        // Older GPUs use the existing ForwardV2 dense-mask SWA path.
        return Step3p5Model::CanUseGPUForward() &&
               GPUForwardTargetsSupportFlashInfer();
    }

    bool LagunaModel::NeedAttentionMask(int qlen, int klen) {
        (void)qlen;
        (void)klen;
        return false;
    }

    std::string LagunaModel::ApplyChatTemplate(const ChatMessages &messages) {
        return basellm::ApplyChatTemplate(messages);
    }
}
