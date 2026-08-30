//
// Qwen4-Exp / Qwen3.8-Flash-Next text model.
//
// The implementation intentionally lives outside qwen3_next.cpp.  Qwen4-Exp
// changes the residual topology (four hyper-connection streams), uses separate
// GDN projections, and adds a very large sharded per-layer embedding (PLE).
// Keeping those rules here prevents checkpoint-specific branches from leaking
// into the established Qwen3-Next path.
//

#include "qwen4_exp.h"

#include "devices/cpu/alivethreadpool.h"
#include "executor.h"
#include "json11.hpp"
#include "utils.h"

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <set>
#include <sstream>
#include <utility>

namespace fastllm {
#ifdef USE_CUDA
    void FastllmCudaMergeMOEClearGraphUnsafeFallbackFlag();
    bool FastllmCudaMergeMOEUsedGraphUnsafeFallback();
#endif

    const std::string Qwen4ExpModel::languagePrefix = "model.language_model.";

    struct Qwen4ExpModel::QsaHostMirrorTransfer {
#ifdef USE_CUDA
        ~QsaHostMirrorTransfer() {
            Release();
        }

        void Prepare(const std::vector<float> &rawKeys,
                     const std::vector<float> &positions,
                     int expectedRows, int rowWidth) {
            AssertInFastLLM(
                expectedRows >= 0 && rowWidth > 0,
                "Qwen4-Exp QSA host mirror received an invalid history shape.");
            if (committedRows == expectedRows) {
                AssertInFastLLM(
                    committedRows == 0 || headDim == rowWidth,
                    "Qwen4-Exp QSA host mirror changed its row width.");
                return;
            }
            AssertInFastLLM(
                rawKeys.size() >= (size_t)expectedRows * rowWidth &&
                    positions.size() >= (size_t)expectedRows,
                "Qwen4-Exp QSA host mirror cannot recover its CUDA history.");
            EnsureCapacity(expectedRows, rowWidth);
            Synchronize();
            if (expectedRows > 0) {
                std::memcpy(rawStorage, rawKeys.data(),
                            (size_t)expectedRows * rowWidth * sizeof(float));
                std::memcpy(positionStorage, positions.data(),
                            (size_t)expectedRows * sizeof(float));
            }
            committedRows = expectedRows;
            headDim = rowWidth;
        }

        void Materialize(std::vector<float> &rawKeys,
                         std::vector<float> &positions) {
            Synchronize();
            if (committedRows == 0) {
                rawKeys.clear();
                positions.clear();
                return;
            }
            rawKeys.assign(rawStorage,
                           rawStorage + (size_t)committedRows * headDim);
            positions.assign(positionStorage,
                             positionStorage + committedRows);
        }

        void Queue(const Data &rawKeys,
                   const Data &positions,
                   int previousRows, int rowCount, int rowWidth) {
            AssertInFastLLM(
                committedRows == previousRows && rowCount > 0 &&
                rowWidth > 0 &&
                rawKeys.dataDevice == DataDevice::CUDA &&
                rawKeys.cudaData != nullptr &&
                positions.dataDevice == DataDevice::CUDA &&
                positions.cudaData != nullptr,
                "Qwen4-Exp QSA host mirror received an invalid CUDA append.");
            AssertInFastLLM(
                committedRows == 0 || headDim == rowWidth,
                "Qwen4-Exp QSA host mirror changed its row width.");
            device = rawKeys.dataDeviceIds.empty()
                ? FastllmCudaGetDevice() : rawKeys.dataDeviceIds[0];
            FastllmCudaSetDevice(device);
            EnsureCapacity(previousRows + rowCount, rowWidth);
            const size_t rawBytes =
                (size_t)rowCount * rowWidth * sizeof(float);
            const size_t positionBytes =
                (size_t)rowCount * sizeof(float);
            const bool rawQueued =
                FastllmCudaCopyFromDeviceToHostAsyncCurrentThread(
                    rawStorage + (size_t)previousRows * rowWidth,
                    rawKeys.cudaData, rawBytes);
            const bool positionsQueued =
                FastllmCudaCopyFromDeviceToHostAsyncCurrentThread(
                    positionStorage + previousRows,
                    positions.cudaData, positionBytes);
            AssertInFastLLM(
                rawQueued && positionsQueued,
                "Qwen4-Exp failed to stage its CUDA QSA host mirror.");
            committedRows += rowCount;
            headDim = rowWidth;
            pending = true;
        }

    private:
        void Synchronize() {
            if (!pending) {
                return;
            }
            FastllmCudaSetDevice(device);
            // Request teardown or prefix materialization can run on a
            // different host worker than the per-thread stream that queued
            // the D2H copies. Synchronize the owning device in these rare
            // paths so pinned storage is never read or freed prematurely.
            ForceDeviceSync();
            pending = false;
        }

        void EnsureCapacity(int requiredRows, int rowWidth) {
            if (rowCapacity >= requiredRows) {
                return;
            }
            Synchronize();
            // Leave one normal prefill chunk of decode headroom. Otherwise a
            // prompt ending exactly at 32K would reallocate all twelve host
            // mirrors on its first decode token.
            const int initialRows = requiredRows >= 4096 ? 36864 : 4096;
            int grownRows = std::max(requiredRows, initialRows);
            if (rowCapacity > 0) {
                grownRows = std::max(grownRows, rowCapacity * 2);
            }
            float *newRaw = reinterpret_cast<float *>(
                FastllmCudaHostMalloc(
                    (size_t)grownRows * rowWidth * sizeof(float)));
            float *newPositions = reinterpret_cast<float *>(
                FastllmCudaHostMalloc(
                    (size_t)grownRows * sizeof(float)));
            if (newRaw == nullptr || newPositions == nullptr) {
                if (newRaw != nullptr) {
                    FastllmCudaHostFree(newRaw);
                }
                if (newPositions != nullptr) {
                    FastllmCudaHostFree(newPositions);
                }
            }
            AssertInFastLLM(
                newRaw != nullptr && newPositions != nullptr,
                "Qwen4-Exp failed to allocate pinned QSA mirror memory.");
            if (committedRows > 0) {
                AssertInFastLLM(
                    headDim == rowWidth,
                    "Qwen4-Exp QSA host mirror changed its row width.");
                std::memcpy(newRaw, rawStorage,
                            (size_t)committedRows * rowWidth * sizeof(float));
                std::memcpy(newPositions, positionStorage,
                            (size_t)committedRows * sizeof(float));
            }
            if (rawStorage != nullptr) {
                FastllmCudaHostFree(rawStorage);
            }
            if (positionStorage != nullptr) {
                FastllmCudaHostFree(positionStorage);
            }
            rawStorage = newRaw;
            positionStorage = newPositions;
            rowCapacity = grownRows;
        }

        void Release() {
            if (device >= 0) {
                FastllmCudaSetDevice(device);
            }
            Synchronize();
            if (rawStorage != nullptr) {
                FastllmCudaHostFree(rawStorage);
                rawStorage = nullptr;
            }
            if (positionStorage != nullptr) {
                FastllmCudaHostFree(positionStorage);
                positionStorage = nullptr;
            }
            rowCapacity = 0;
            committedRows = 0;
            headDim = 0;
        }

        float *rawStorage = nullptr;
        float *positionStorage = nullptr;
        int rowCapacity = 0;
        int committedRows = 0;
        int headDim = 0;
        int device = -1;
        bool pending = false;
#endif
    };

    namespace {
        constexpr uint64_t kSplitMixGamma = 0x9E3779B97F4A7C15ULL;
        constexpr uint64_t kSplitMixM1 = 0xBF58476D1CE4E5B9ULL;
        constexpr uint64_t kSplitMixM2 = 0x94D049BB133111EBULL;
        constexpr uint64_t kPleLayerPrime = 10007ULL;
        constexpr int kQwen4SparsePrefillMinContext = 4096;
        constexpr int kQwen4DenseCacheMaxGrowth = 8192;

        class Qwen4RangeTask final : public MultiThreadBaseOp {
        public:
            Qwen4RangeTask(int start, int end,
                           const std::function<void(int, int)> &function)
                : start(start), end(end), function(function) {}

            void Run() override {
                function(start, end);
            }

        private:
            int start;
            int end;
            std::function<void(int, int)> function;
        };

        void Qwen4ParallelFor(int count,
                              const std::function<void(int, int)> &function) {
            if (count <= 0) {
                return;
            }
            AliveThreadPool *pool = GetAlivePool();
            const int firstThread = pool->curActivateThreadInterval.first;
            const int availableThreads = std::max(
                1, pool->curActivateThreadInterval.second - firstThread);
            const int threadCount = std::min(count, availableThreads);
            if (threadCount == 1) {
                function(0, count);
                return;
            }

            std::vector<Qwen4RangeTask*> tasks;
            tasks.reserve(threadCount);
            for (int thread = 0; thread < threadCount; thread++) {
                const int start = (int)((int64_t)count * thread / threadCount);
                const int end = (int)((int64_t)count * (thread + 1) /
                                      threadCount);
                tasks.push_back(new Qwen4RangeTask(start, end, function));
                pool->PushOp(firstThread + thread, tasks.back());
            }
            for (int thread = 0; thread < threadCount; thread++) {
                pool->Wait(firstThread + thread);
                delete tasks[thread];
            }
        }

        uint64_t Qwen4DenseLogicalBytes(const Data &data) {
            uint64_t elements = 1;
            for (int dim : data.dims) {
                elements *= (uint64_t)dim;
            }
            return (elements * data.unitSize + data.unitSizeDiv - 1) /
                   data.unitSizeDiv;
        }

        int Qwen4NextCacheCapacity(int currentCapacity,
                                   int requiredCapacity,
                                   int quantum, int maxGrowth) {
            AssertInFastLLM(currentCapacity >= 0 &&
                            requiredCapacity >= 0 && quantum > 0 &&
                            maxGrowth >= quantum,
                            "Qwen4-Exp cache growth received invalid limits.\n");
            if (requiredCapacity <= currentCapacity) {
                return currentCapacity;
            }

            const int64_t base = std::max<int64_t>(
                currentCapacity, requiredCapacity);
            const int64_t growth = std::min<int64_t>(
                maxGrowth, std::max<int64_t>(quantum, base / 2));
            const int64_t limit = std::numeric_limits<int>::max();
            const int64_t target = std::min(limit, base + growth);
            int64_t rounded = target;
            if (target <= limit - (quantum - 1)) {
                rounded = ((target + quantum - 1) / quantum) * quantum;
            }
            rounded = std::min(rounded, limit);
            return std::max(requiredCapacity, (int)rounded);
        }

        int Qwen4RoundUpCacheCapacity(int requiredCapacity, int quantum) {
            AssertInFastLLM(requiredCapacity >= 0 && quantum > 0,
                            "Qwen4-Exp cache rounding received invalid limits.\n");
            const int64_t limit = std::numeric_limits<int>::max();
            const int64_t required = requiredCapacity;
            if (required > limit - (quantum - 1)) {
                return requiredCapacity;
            }
            return (int)(((required + quantum - 1) / quantum) * quantum);
        }

        int Qwen4AxisCapacity(const Data &data, int axis) {
            if (data.dims.empty()) {
                return 0;
            }
            AssertInFastLLM(axis >= 0 && axis < (int)data.dims.size(),
                            "Qwen4-Exp cache axis is out of range.\n");
            if (data.expansionDims.size() == data.dims.size()) {
                return std::max(data.dims[axis],
                                data.expansionDims[axis]);
            }
            return data.dims[axis];
        }

        void Qwen4EnsureAppendCapacity(Data &cache, const Data &append,
                                       int axis, int quantum,
                                       int maxGrowth,
                                       bool geometricGrowth) {
            AssertInFastLLM(!append.dims.empty() && axis >= 0 &&
                            axis < (int)append.dims.size() &&
                            append.dims[axis] >= 0,
                            "Qwen4-Exp cache append received an invalid tensor.\n");
            if (!cache.dims.empty()) {
                AssertInFastLLM(cache.dims.size() == append.dims.size() &&
                                axis < (int)cache.dims.size(),
                                "Qwen4-Exp cache append rank mismatch.\n");
            }
            const int logical = cache.dims.empty() ? 0 : cache.dims[axis];
            const int64_t required64 =
                (int64_t)logical + append.dims[axis];
            AssertInFastLLM(required64 <= std::numeric_limits<int>::max(),
                            "Qwen4-Exp cache capacity overflow.\n");
            const int required = (int)required64;
            const int currentCapacity = Qwen4AxisCapacity(cache, axis);
            if (currentCapacity >= required) {
                return;
            }
            std::vector<int> expanded = cache.dims.empty()
                ? append.dims : cache.dims;
            if (geometricGrowth) {
                expanded[axis] = Qwen4NextCacheCapacity(
                    currentCapacity, required, quantum, maxGrowth);
            } else {
                const int roundedAppend = Qwen4RoundUpCacheCapacity(
                    append.dims[axis], quantum);
                const int64_t legacyCapacity =
                    (int64_t)logical + roundedAppend;
                AssertInFastLLM(
                    legacyCapacity <= std::numeric_limits<int>::max(),
                    "Qwen4-Exp cache capacity overflow.\n");
                expanded[axis] = (int)legacyCapacity;
            }
            cache.Expansion(expanded);
        }

#ifdef USE_CUDA
        class Qwen4CudaDeviceGuard {
        public:
            explicit Qwen4CudaDeviceGuard(
                    const std::vector<int> &deviceIds)
                : previousDevice(FastllmCudaGetDevice()), changed(false) {
                if (!deviceIds.empty() &&
                    deviceIds[0] != this->previousDevice) {
                    FastllmCudaSetDevice(deviceIds[0]);
                    this->changed = true;
                }
            }

            ~Qwen4CudaDeviceGuard() {
                if (this->changed) {
                    FastllmCudaSetDevice(this->previousDevice);
                }
            }

        private:
            int previousDevice;
            bool changed;
        };

        bool Qwen4BorrowCudaTensor(const Data &source, Data &destination,
                                   bool linear) {
            if (source.dataDevice != DataDevice::CUDA ||
                source.cudaData == nullptr || source.multiDeviceData) {
                return false;
            }
            const long long cacheUid = destination.cacheUid;
            destination.FreeSpace();
            destination.isFake = false;
            destination.dataType = source.dataType;
            destination.UpdateUnitSize();
            destination.dims = source.dims;
            destination.strides = source.strides;
            destination.expansionSize = source.expansionSize;
            destination.expansionBytes = source.expansionBytes;
            destination.expansionDims = source.expansionDims;
            // Data::CopyFrom intentionally drops expansionDims when capacity
            // exactly equals the logical shape.  The legacy scheduler still
            // uses expansionDims[1] as its cache-capacity marker, so restore
            // the canonical exact-capacity metadata for a borrowed view.
            if (destination.expansionDims.size() !=
                destination.dims.size()) {
                destination.expansionDims = destination.dims;
            }
            destination.dataDevice = DataDevice::CUDA;
            destination.dataDeviceIds = source.dataDeviceIds;
            destination.cudaData = source.cudaData;
            destination.cudaDataBorrowed = true;
            destination.cacheUid = cacheUid;
            destination.isKVCache = true;
            destination.isLinearAttention = linear;
            destination.isLinearAttentionTransposed =
                source.isLinearAttentionTransposed;
            destination.isPagedKVCache = false;
            destination.pagedKVCacheData = nullptr;
            destination.pageIndex.clear();
            destination.lastPageLen = 0;
            destination.multiDeviceData = false;
            destination.multiDeviceDatas.clear();
            destination.ClearTensorParallelLayout();
            return true;
        }

        void Qwen4DetachBorrowedCudaTensor(
                Data &data, bool growFullAttentionCache = true) {
            if (data.dataDevice != DataDevice::CUDA ||
                !data.cudaDataBorrowed || data.cudaData == nullptr) {
                return;
            }
            Qwen4CudaDeviceGuard deviceGuard(data.dataDeviceIds);
            if (growFullAttentionCache &&
                !data.isLinearAttention && data.dims.size() >= 2) {
                // Full-attention K/V grows along the token axis immediately
                // after restore.  Let Expansion perform COW and reserve the
                // same CUDA growth quantum used by RunFullAttention, avoiding
                // a second full-cache copy on the first appended token.
                std::vector<int> capacity = data.expansionDims;
                if (capacity.size() != data.dims.size()) {
                    capacity = data.dims;
                }
                AssertInFastLLM(
                    data.dims[1] < std::numeric_limits<int>::max(),
                    "Qwen4-Exp prefix cache capacity overflow.\n");
                capacity[1] = std::max(
                    capacity[1], data.dims[1] + 128);
                data.Expansion(capacity);
                return;
            }
            void *borrowed = data.cudaData;
            const uint64_t copyBytes = data.expansionBytes != 0
                ? data.expansionBytes : data.GetBytes();
            uint64_t allocationSize = data.expansionSize;
            if (allocationSize == 0 && !data.dims.empty()) {
                allocationSize = data.strides.empty()
                    ? data.Count(0)
                    : data.strides[0] * data.dims[0];
            }
            data.cudaData = nullptr;
            data.cudaDataBorrowed = false;
            data.expansionSize = 0;
            data.expansionBytes = 0;
            data.MallocSpace(allocationSize, false);
            AssertInFastLLM(data.cudaData != nullptr,
                            "Qwen4-Exp prefix cache COW allocation failed.\n");
            FastllmCudaCopyFromDeviceToDevice(
                data.cudaData, borrowed, copyBytes);
        }

        bool Qwen4PrepareDecodeGraphTensor(Data &destination,
                                            const Data &source,
                                            int device) {
            if (source.dataDevice != DataDevice::CUDA ||
                source.cudaData == nullptr || source.multiDeviceData) {
                return false;
            }
            FastllmCudaSetDevice(device);
            bool reset = destination.isFake ||
                         destination.dataDevice != DataDevice::CUDA ||
                         destination.dataType != source.dataType ||
                         destination.cudaData == nullptr ||
                         (!destination.dataDeviceIds.empty() &&
                          destination.dataDeviceIds[0] != device);
            if (reset) {
                if (destination.isFake) {
                    destination.isFake = false;
                    destination.cpuData = nullptr;
                    destination.cudaData = nullptr;
                    destination.deviceData = nullptr;
                    destination.expansionSize = 0;
                    destination.expansionBytes = 0;
                } else {
                    destination.FreeSpace();
                }
                destination.dataType = source.dataType;
                destination.UpdateUnitSize();
                destination.dataDevice = DataDevice::CUDA;
                destination.dataDeviceIds = {device};
                destination.multiDeviceData = false;
            }
            destination.expansionDims.clear();
            destination.Resize(source.dims);
            destination.Allocate(false);
            if (destination.cudaData == source.cudaData) {
                return true;
            }
            return FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(
                destination.cudaData, source.cudaData, source.GetBytes());
        }

        // Borrow CUDA storage while retaining normal ownership for any host
        // materialization created by Data::ToDevice(CPU).  Data::isFake is too
        // broad for this use: it suppresses both CUDA and CPU cleanup, whereas
        // cudaDataBorrowed already expresses the only ownership exception we
        // need here.
        void Qwen4BorrowCudaStorage(Data &destination,
                                    const Data &source) {
            AssertInFastLLM(source.dataDevice == DataDevice::CUDA &&
                            source.cudaData != nullptr &&
                            !source.multiDeviceData,
                            "Qwen4 received invalid borrowed CUDA storage.\n");
            AssertInFastLLM(destination.cpuData == nullptr &&
                            destination.cudaData == nullptr,
                            "Qwen4 borrowed CUDA destination must be empty.\n");
            destination.isFake = false;
            destination.dataType = source.dataType;
            destination.UpdateUnitSize();
            destination.dataDevice = DataDevice::CUDA;
            destination.dataDeviceIds = source.dataDeviceIds;
            destination.dims = source.dims;
            destination.strides = source.strides;
            destination.expansionDims = source.expansionDims;
            destination.expansionSize = source.expansionSize;
            destination.expansionBytes = source.expansionBytes;
            destination.cudaData = source.cudaData;
            destination.cudaDataBorrowed = true;
            destination.multiDeviceData = false;
        }
#endif

        bool Qwen4CudaOnlyDeviceMap(
                const std::map<std::string, int> &deviceMap) {
            for (const auto &item : deviceMap) {
                const std::string &name = item.first;
                if (item.second > 0 && name != "cuda" &&
                    !(name.size() > 5 && name.compare(0, 5, "cuda:") == 0)) {
                    return false;
                }
            }
            return true;
        }

        bool Qwen4EnvFlagEnabled(const char *name) {
            const char *value = std::getenv(name);
            return value != nullptr && value[0] != '\0' &&
                   std::strcmp(value, "0") != 0 &&
                   std::strcmp(value, "false") != 0 &&
                   std::strcmp(value, "FALSE") != 0 &&
                   std::strcmp(value, "off") != 0 &&
                   std::strcmp(value, "OFF") != 0;
        }

        int Qwen4EnvInt(const char *name, int fallback) {
            const char *value = std::getenv(name);
            if (value == nullptr || value[0] == '\0') {
                return fallback;
            }
            char *end = nullptr;
            const long parsed = std::strtol(value, &end, 10);
            return end == value ? fallback : (int)parsed;
        }

        bool Qwen4PrefixCacheEnabled() {
            const char *value = std::getenv("FASTLLM_PREFIX_CACHE");
            return value == nullptr || value[0] == '\0' ||
                   Qwen4EnvFlagEnabled("FASTLLM_PREFIX_CACHE");
        }

        bool Qwen4PrefixCacheDebugEnabled() {
            return Qwen4EnvFlagEnabled("FASTLLM_QWEN4_PREFIX_CACHE_DEBUG");
        }

        // A restored prefix may be followed by several prompt tokens in one
        // Forward. FastLLM represents that ordinary case with a dense causal
        // mask, while the incremental CUDA QSA selector represents the same
        // visibility through causalStart. Verify the mask once per Forward so
        // that path can keep its compressed state on the device. Arbitrary
        // masks deliberately return false and retain the established generic
        // CPU fallback.
        bool Qwen4IsContiguousCausalMask(const Data &mask,
                                         int previousLength,
                                         int sequence) {
            if (mask.dims.empty()) {
                return true;
            }
            if (previousLength < 0 || sequence <= 0 ||
                previousLength > std::numeric_limits<int>::max() - sequence) {
                return false;
            }
            const int keyLength = previousLength + sequence;
            const size_t expectedCount =
                (size_t)sequence * (size_t)keyLength;
            if (mask.dims.back() != keyLength ||
                mask.Count(0) < (long long)expectedCount) {
                return false;
            }

            const Data *hostMask = &mask;
            Data converted;
            if (mask.dataDevice != DataDevice::CPU ||
                mask.dataType != DataType::FLOAT32) {
                ToDataType(mask, converted, DataType::FLOAT32);
                converted.ToDevice(DataDevice::CPU);
                hostMask = &converted;
            }
            if (hostMask->cpuData == nullptr) {
                return false;
            }
            const float *values =
                reinterpret_cast<const float *>(hostMask->cpuData);
            for (int query = 0; query < sequence; query++) {
                const int lastVisible = previousLength + query;
                const float *row = values + (size_t)query * keyLength;
                for (int key = 0; key < keyLength; key++) {
                    const bool visible = std::fabs(row[key]) < 0.5f;
                    if (visible != (key <= lastVisible)) {
                        return false;
                    }
                }
            }
            return true;
        }

        uint64_t Qwen4SplitMix64(uint64_t value) {
            value += kSplitMixGamma;
            value = ((value ^ (value >> 30)) * kSplitMixM1);
            value = ((value ^ (value >> 27)) * kSplitMixM2);
            return value ^ (value >> 31);
        }

        bool Qwen4IsPrime(int64_t value) {
            if (value < 2) {
                return false;
            }
            if ((value & 1) == 0) {
                return value == 2;
            }
            for (int64_t divisor = 3; divisor * divisor <= value; divisor += 2) {
                if (value % divisor == 0) {
                    return false;
                }
            }
            return true;
        }

        int Qwen4DictInt(const std::map<std::string, std::string> &dicts,
                         const std::string &key, int defaultValue) {
            auto it = dicts.find(key);
            if (it == dicts.end() || it->second.empty() || it->second == "None") {
                return defaultValue;
            }
            return std::atoi(it->second.c_str());
        }

        float Qwen4DictFloat(const std::map<std::string, std::string> &dicts,
                             const std::string &key, float defaultValue) {
            auto it = dicts.find(key);
            if (it == dicts.end() || it->second.empty() || it->second == "None") {
                return defaultValue;
            }
            return (float)std::atof(it->second.c_str());
        }

        bool Qwen4StartsWith(const std::string &value, const std::string &prefix) {
            return value.size() >= prefix.size() &&
                   value.compare(0, prefix.size(), prefix) == 0;
        }

        bool Qwen4EndsWith(const std::string &value, const std::string &suffix) {
            return value.size() >= suffix.size() &&
                   value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
        }

        void Qwen4AddOne(Data &data) {
            if (data.dims.empty()) {
                return;
            }
            data.ToDevice(DataDevice::CPU);
            ToDataType(data, DataType::FLOAT32);
            float *values = reinterpret_cast<float *>(data.cpuData);
            for (int i = 0; i < data.Count(0); i++) {
                values[i] += 1.0f;
            }
        }

        float Qwen4Silu(float value) {
            return value / (1.0f + std::exp(-value));
        }

        void Qwen4CastLike(Data &value, const Data &reference) {
            if (value.dataType != reference.dataType) {
                ToDataType(value, reference.dataType);
            }
        }

        void Qwen4ResetCacheTensor(Data &cache, bool linear) {
            const DataType dataType = cache.dataType;
            const long long cacheUid = cache.cacheUid;
            if (cache.multiDeviceData) {
                for (auto &item : cache.multiDeviceDatas) {
                    delete item.second;
                }
                cache.multiDeviceDatas.clear();
                cache.multiDeviceData = false;
            }
            cache.FreeSpace();
            cache = Data(dataType);
            cache.cacheUid = cacheUid;
            cache.isKVCache = true;
            cache.isLinearAttention = linear;
        }
    }

    struct Qwen4ExpModel::DecodeCudaGraphState {
        struct Segment {
            bool warmed = false;
            bool captured = false;
            bool disabled = false;
            int captureFailures = 0;
            void *graph = nullptr;
            void *exec = nullptr;
            std::vector<void*> reservedPointers;
        };

        std::mutex mutex;
        int device = -1;
        std::vector<void*> linearCachePointers;
        Data hiddenStates[2];
        Data attentionInput;
        Data attentionInjection;
        Data attentionOutput;
        Data logits;
        std::map<int, Segment> segments;

        void DestroyGraphs() {
#ifdef USE_CUDA
            const int previousDevice = FastllmCudaGetDevice();
            if (device >= 0) {
                FastllmCudaSetDevice(device);
            }
            for (auto &item : segments) {
                Segment &segment = item.second;
                if (segment.exec != nullptr) {
                    FastllmCudaGraphExecDestroy(segment.exec);
                    segment.exec = nullptr;
                }
                if (segment.graph != nullptr) {
                    FastllmCudaGraphDestroy(segment.graph);
                    segment.graph = nullptr;
                }
                if (!segment.reservedPointers.empty()) {
                    FastllmCudaGraphMemoryPoolRelease(
                        segment.reservedPointers);
                    segment.reservedPointers.clear();
                }
                segment.captured = false;
            }
            if (previousDevice >= 0) {
                FastllmCudaSetDevice(previousDevice);
            }
#endif
            segments.clear();
        }

        void Reset(int newDevice) {
            DestroyGraphs();
#ifdef USE_CUDA
            const int previousDevice = FastllmCudaGetDevice();
            if (device >= 0) {
                FastllmCudaSetDevice(device);
            }
#endif
            hiddenStates[0].FreeSpace();
            hiddenStates[1].FreeSpace();
            attentionInput.FreeSpace();
            attentionInjection.FreeSpace();
            attentionOutput.FreeSpace();
            logits.FreeSpace();
#ifdef USE_CUDA
            if (previousDevice >= 0) {
                FastllmCudaSetDevice(previousDevice);
            }
#endif
            linearCachePointers.clear();
            device = newDevice;
        }

        ~DecodeCudaGraphState() {
            DestroyGraphs();
        }
    };

    Qwen4ExpModel::Qwen4ExpModel() : Qwen3NextModel() {
        this->canDoBatchForward = false;
        this->model_type = "qwen4_exp";
        this->model_struct = "qwen4_exp";
        this->defaultChunkedPrefillSize = 4096;
        this->block_cnt = 48;
        this->rotary_dim = 64;

        this->weight.embeddingNames.clear();
        this->weight.linearNames.clear();
        this->weight.embeddingNames.insert(languagePrefix + "embed_tokens.weight");
        this->weight.linearNames = {
            "lm_head.weight",
            languagePrefix + "hyper_connection_mixer.input_mix_weight_down.weight",
            languagePrefix + "hyper_connection_mixer.input_mix_weight_up.weight",
            languagePrefix + "layers.*.attn_hyper_connection.input_mix_weight_down.weight",
            languagePrefix + "layers.*.attn_hyper_connection.input_mix_weight_up.weight",
            languagePrefix + "layers.*.attn_hyper_connection.block_inject_weight.weight",
            languagePrefix + "layers.*.mlp_hyper_connection.input_mix_weight_down.weight",
            languagePrefix + "layers.*.mlp_hyper_connection.input_mix_weight_up.weight",
            languagePrefix + "layers.*.mlp_hyper_connection.block_inject_weight.weight",
            languagePrefix + "layers.*.linear_attn.in_proj_qkv.weight",
            languagePrefix + "layers.*.linear_attn.in_proj_z.weight",
            languagePrefix + "layers.*.linear_attn.in_proj_b.weight",
            languagePrefix + "layers.*.linear_attn.in_proj_a.weight",
            languagePrefix + "layers.*.linear_attn.out_proj.weight",
            languagePrefix + "layers.*.self_attn.q_proj.weight",
            languagePrefix + "layers.*.self_attn.k_proj.weight",
            languagePrefix + "layers.*.self_attn.v_proj.weight",
            languagePrefix + "layers.*.self_attn.o_proj.weight",
            languagePrefix + "layers.*.self_attn.indexer.index_qk_proj.weight",
            languagePrefix + "layers.*.mlp.gate.weight",
            languagePrefix + "layers.*.mlp.shared_expert_gate.weight",
            languagePrefix + "layers.*.mlp.shared_expert.gate_proj.weight",
            languagePrefix + "layers.*.mlp.shared_expert.up_proj.weight",
            languagePrefix + "layers.*.mlp.shared_expert.down_proj.weight",
            languagePrefix + "layers.*.mlp.shared_expert.gateup_proj.weight",
            languagePrefix + "layers.*.mlp.experts.*.gate_proj.weight",
            languagePrefix + "layers.*.mlp.experts.*.up_proj.weight",
            languagePrefix + "layers.*.mlp.experts.*.down_proj.weight",
            languagePrefix + "layers.*.mlp.experts.*.gateup_proj.weight",
            languagePrefix + "layers.*.ple.key_proj.weight",
            languagePrefix + "layers.*.ple.value_proj.weight"
        };
    }

    Qwen4ExpModel::~Qwen4ExpModel() {
        ShutdownRuntime();
        {
            std::lock_guard<std::mutex> guard(this->prefixCacheMutex);
            this->pendingPrefixRestores.clear();
            this->prefixSnapshots.clear();
        }
        {
            std::lock_guard<std::mutex> guard(this->stateMutex);
            this->decodeCudaGraphStates.clear();
            this->requestStates.clear();
        }
    }

    void Qwen4ExpModel::InitParams() {
        // The public checkpoint is multimodal and stores text settings below
        // text_config.  FastLLM's common initializer consumes flat keys.
        std::map<std::string, std::string> flattened;
        for (const auto &item : this->weight.dicts) {
            if (!Qwen4StartsWith(item.first, "text_config.")) {
                continue;
            }
            std::string key = item.first.substr(std::strlen("text_config."));
            flattened[key] = item.second;
            if (Qwen4StartsWith(key, "rope_parameters.")) {
                flattened[key.substr(std::strlen("rope_parameters."))] = item.second;
            }
        }
        for (const auto &item : flattened) {
            if (this->weight.dicts.find(item.first) == this->weight.dicts.end()) {
                this->weight.dicts[item.first] = item.second;
            }
        }

        basellm::InitParams();

        this->num_experts = Qwen4DictInt(weight.dicts, "num_experts", 512);
        this->num_experts_per_tok = Qwen4DictInt(weight.dicts, "num_experts_per_tok", 10);
        this->norm_topk_prob = true;
        this->routed_scaling_factor = Qwen4DictFloat(weight.dicts, "routed_scaling_factor", 1.0f);
        this->num_key_value_heads = Qwen4DictInt(weight.dicts, "num_key_value_heads", 2);
        this->head_dim = Qwen4DictInt(weight.dicts, "head_dim", 256);
        this->num_k_heads = Qwen4DictInt(weight.dicts, "linear_num_key_heads", 16);
        this->num_v_heads = Qwen4DictInt(weight.dicts, "linear_num_value_heads", 48);
        this->head_k_dim = Qwen4DictInt(weight.dicts, "linear_key_head_dim", 128);
        this->head_v_dim = Qwen4DictInt(weight.dicts, "linear_value_head_dim", 128);
        this->linearConvKernel = Qwen4DictInt(weight.dicts, "linear_conv_kernel_dim", 4);
        this->rms_norm_eps = Qwen4DictFloat(weight.dicts, "rms_norm_eps", 1e-6f);
        this->max_positions = Qwen4DictInt(weight.dicts, "max_position_embeddings", 262144);
        this->hcCount = Qwen4DictInt(weight.dicts, "hc_count", 4);
        this->hcLowRank = Qwen4DictInt(weight.dicts, "hc_lowrank", 320);
        this->pleEmbedDim = Qwen4DictInt(weight.dicts, "ple_embed_dim", this->embed_dim);
        this->pleConvKernel = Qwen4DictInt(weight.dicts, "ple_conv_kernel_size", 4);
        this->ngramSize = Qwen4DictInt(weight.dicts, "ngram_size", 3);
        this->headsPerNgram = Qwen4DictInt(weight.dicts, "heads_per_ngram", 8);
        this->ngramHeads = (this->ngramSize - 1) * this->headsPerNgram;
        this->ngramHeadDim = this->pleEmbedDim / this->ngramHeads;
        this->ngramVocabBase = Qwen4DictInt(weight.dicts, "ngram_vocab_size_base", 20000000);
        this->ngramShardCount = Qwen4DictInt(weight.dicts, "split_ngram_parts", 128);
        this->eosToken = Qwen4DictInt(weight.dicts, "eos_token_id", 248044);
        this->eos_token_id = this->eosToken;
        this->pleSeed = Qwen4DictInt(weight.dicts, "seed", 1234);
        this->indexerHeads = Qwen4DictInt(weight.dicts, "indexer_n_heads", 4);
        this->indexerKvHeads = Qwen4DictInt(weight.dicts, "indexer_kv_heads", 1);
        this->indexerHeadDim = Qwen4DictInt(weight.dicts, "indexer_head_dim", 128);
        this->indexerBudget = Qwen4DictInt(weight.dicts, "indexer_budget", 2048);
        this->indexerCompressRatio = Qwen4DictInt(weight.dicts, "indexer_compress_ratio", 4);

        this->rotary_dim = (int)(this->head_dim *
            Qwen4DictFloat(weight.dicts, "partial_rotary_factor", 0.25f) + 1e-5f);
        this->rope_base = Qwen4DictFloat(weight.dicts, "rope_theta", 10000000.0f);
        auto rope = this->UpdateRotaryPosEmb(this->rope_base, 1.0f);
        this->qsaSinValues = rope.first;
        this->qsaCosValues = rope.second;
        this->sinData.ToDevice(DataDevice::CPU);
        this->cosData.ToDevice(DataDevice::CPU);
        this->sinData.CopyFrom(Data(DataType::FLOAT32,
            {(int)this->sin.size(), (int)this->sin[0].size()},
            this->qsaSinValues));
        this->cosData.CopyFrom(Data(DataType::FLOAT32,
            {(int)this->cos.size(), (int)this->cos[0].size()},
            this->qsaCosValues));

        this->linearLayers.assign(this->block_cnt, true);
        for (int layer = 0; layer < this->block_cnt; layer++) {
            this->linearLayers[layer] = ((layer + 1) % 4 != 0);
        }
        auto layerTypesIt = this->weight.dicts.find("layer_types");
        if (layerTypesIt != this->weight.dicts.end()) {
            std::string error;
            json11::Json values = json11::Json::parse(layerTypesIt->second, error);
            if (error.empty() && values.is_array() &&
                values.array_items().size() == (size_t)this->block_cnt) {
                for (int layer = 0; layer < this->block_cnt; layer++) {
                    this->linearLayers[layer] =
                        values.array_items()[layer].string_value() == "linear_attention";
                }
            }
        }
        for (int layer = 0; layer < this->block_cnt; layer++) {
            if (!this->IsLinearAttentionLayer(layer)) {
                this->kvCacheId = layer;
                break;
            }
        }

        // ple_layer_ids are one-indexed.  The released checkpoint has one PLE
        // at decoder layer 2, i.e. C++ layer index 1.
        this->pleLayer = 1;
        auto pleLayersIt = this->weight.dicts.find("ple_layer_ids");
        if (pleLayersIt != this->weight.dicts.end()) {
            std::string error;
            json11::Json values = json11::Json::parse(pleLayersIt->second, error);
            if (error.empty() && values.is_array() && !values.array_items().empty()) {
                this->pleLayer = values.array_items()[0].int_value() - 1;
            }
        }

        AssertInFastLLM(this->hcCount > 1 && this->embed_dim > 0,
                        "Qwen4-Exp invalid hyper-connection configuration.");
        AssertInFastLLM(this->ngramHeads > 0 &&
                        this->pleEmbedDim % this->ngramHeads == 0,
                        "Qwen4-Exp invalid PLE embedding dimensions.");
        AssertInFastLLM(this->num_v_heads % this->num_k_heads == 0,
                        "Qwen4-Exp GDN value heads must be divisible by key heads.");
        AssertInFastLLM(this->indexerKvHeads == 1 &&
                        this->indexerBudget % this->indexerCompressRatio == 0,
                        "Qwen4-Exp QSA requires one KV head and a divisible token budget.");

        const float linearInvScale =
            1.0f / std::sqrt((float)this->head_k_dim);
        Data linearInvScaleData(
            DataType::FLOAT32, {this->head_k_dim},
            std::vector<float>(this->head_k_dim, linearInvScale));
        this->linearInvScaleData.CopyFrom(linearInvScaleData);

        // Deterministic PLE hash metadata.  Rebuilding it avoids truncating the
        // checkpoint's uint64 multipliers through FastLLM's int32 parameter
        // representation.
        this->pleMultipliers.clear();
        uint64_t unigramVocab = (uint64_t)Qwen4DictInt(weight.dicts, "vocab_size", 248320);
        uint64_t maxLong = (uint64_t)std::numeric_limits<int64_t>::max();
        uint64_t multiplierMax = maxLong / std::max<uint64_t>(unigramVocab, 1);
        uint64_t halfBound = std::max<uint64_t>(1, multiplierMax / 2);
        uint64_t baseSeed = (uint64_t)this->pleSeed; // PLE index is zero here.
        for (int index = 0; index < this->ngramSize; index++) {
            uint64_t value = baseSeed + kSplitMixGamma * (uint64_t)(index + 1);
            this->pleMultipliers.push_back(
                2 * (Qwen4SplitMix64(value) % halfBound) + 1);
        }

        this->pleHeadVocabSizes.clear();
        this->pleHeadOffsets.clear();
        int64_t candidate = (int64_t)this->ngramVocabBase - 1;
        int64_t offset = 0;
        for (int head = 0; head < this->ngramHeads; head++) {
            do {
                candidate++;
            } while (!Qwen4IsPrime(candidate));
            this->pleHeadVocabSizes.push_back(candidate);
            this->pleHeadOffsets.push_back(offset);
            offset += candidate;
        }

        this->weights.clear();
        this->biass.clear();
        this->preparedWeights = false;

        for (int layer = 0; layer < this->block_cnt; layer++) {
            const std::string mlp = languagePrefix + "layers." +
                                    std::to_string(layer) + ".mlp.";
            const std::string sharedGate = mlp + "shared_expert.gate_proj.weight";
            const std::string sharedUp = mlp + "shared_expert.up_proj.weight";
            const std::string sharedGateUp = mlp + "shared_expert.gateup_proj.weight";
            this->weightMergeRules.push_back(WeightMergeRule({
                WeightMergeRuleSingle({sharedGate, sharedUp}, sharedGateUp,
                                      std::string("linearSwiglu"))}));

            for (int expert = 0; expert < this->num_experts; expert++) {
                const std::string expertPrefix = mlp + "experts." +
                                                 std::to_string(expert) + ".";
                const std::string gate = expertPrefix + "gate_proj.weight";
                const std::string up = expertPrefix + "up_proj.weight";
                const std::string gateUp = expertPrefix + "gateup_proj.weight";
                const std::string down = expertPrefix + "down_proj.weight";
                this->weightMergeRules.push_back(WeightMergeRule({
                    WeightMergeRuleSingle({gate, up}, gateUp,
                                          std::string("linearSwiglu"))}));
                this->AddSpecialWeight(gateUp, "linearSwiglu", layer);
                this->AddSpecialWeight(down, "linearColumn", layer);
                this->moeLinears.insert(gate);
                this->moeLinears.insert(up);
                this->moeLinears.insert(down);
            }
        }
    }

    std::map<std::string, std::vector<std::pair<std::string, DataType>>>
    Qwen4ExpModel::GetTensorMap(const std::vector<std::string> &tensorNames) {
        std::map<std::string, std::vector<std::pair<std::string, DataType>>> result;
        std::vector<std::string> ordinary;
        const bool compactNvfp4Experts = std::any_of(
            tensorNames.begin(), tensorNames.end(), [](const std::string &name) {
                if (name.find(".mlp.experts.") == std::string::npos) {
                    return false;
                }
                // Scalar safetensors (including weight_scale_2) are omitted
                // from GetSortedItemNames().  The planar block-scale tensor is
                // non-scalar and therefore is the reliable model-map marker.
                return Qwen4EndsWith(name, ".weight_scale") ||
                       Qwen4EndsWith(name, ".weight_scale_2");
            });
        const bool scaledFp8Ple = std::any_of(
            tensorNames.begin(), tensorNames.end(), [](const std::string &name) {
                return name.find(
                    "ple_embedding.ngram_embedding.weight_scale") !=
                    std::string::npos;
            });
        for (const std::string &name : tensorNames) {
            if (name != "lm_head.weight" && !Qwen4StartsWith(name, languagePrefix)) {
                // The model directory also contains vision and MTP tensors.
                // The text-only FastLLM model must not keep either resident.
                continue;
            }
            if (name.find("ple_embedding.layer_multipliers") != std::string::npos ||
                name.find("ple_embedding.ngram_heads_offsets") != std::string::npos ||
                name.find("ple_embedding.ngram_heads_vocab_sizes") != std::string::npos) {
                continue;
            }
            if (name.find("ple_embedding.ngram_embedding.shard_") != std::string::npos &&
                Qwen4EndsWith(name, ".weight")) {
                // Older Qwen4-Exp checkpoints store an unscaled E4M3 payload
                // plus one common scalar.  Official Qwen3.8-Flash-Next NVFP4
                // checkpoints instead keep the large CPU-offloaded PLE table
                // in BF16 and omit that scalar.  Preserve either source
                // representation so the table is neither expanded nor
                // requantized while loading.
                result[name].push_back({
                    name,
                    scaledFp8Ple ? DataType::FP8_E4M3 : DataType::BFLOAT16});
                this->ngramWeights.insert(name);
                continue;
            }
            if (name.find("ple_embedding.ngram_embedding.weight_scale") != std::string::npos) {
                result[name].push_back({name, DataType::FLOAT32});
                continue;
            }
            ordinary.push_back(name);
        }
        auto mapped = basellm::GetTensorMap(ordinary);
        if (compactNvfp4Experts) {
            for (auto &source : mapped) {
                for (auto &target : source.second) {
                    if (this->moeLinears.find(target.first) !=
                        this->moeLinears.end()) {
                        target.second =
                            DataType::NVFP4_BLOCK_16_E4M3;
                    }
                }
            }
        }
        result.insert(mapped.begin(), mapped.end());
        return result;
    }

    void Qwen4ExpModel::OnModelWeightsLoaded() {
        if (this->ngramDevice != "disk") {
            return;
        }

        const std::string embeddingPrefix = languagePrefix + "layers." +
            std::to_string(this->pleLayer) +
            ".ple.ple_embedding.ngram_embedding.";
        int64_t totalRows = 0;
        DataType dataType = DataType::FLOAT32;
        std::vector<DiskWeightPart> parts;
        for (int shardIndex = 0; shardIndex < this->ngramShardCount;
             shardIndex++) {
            const std::string name = embeddingPrefix + "shard_" +
                std::to_string(shardIndex) + ".weight";
            auto it = this->weight.weight.find(name);
            AssertInFastLLM(it != this->weight.weight.end(),
                            "Qwen4-Exp disk PLE is missing shard metadata: " +
                                name + "\n");
            const Data &shard = it->second;
            AssertInFastLLM(
                shard.isDiskWeight && shard.cpuData == nullptr &&
                    shard.dims.size() == 2 && shard.dims[0] > 0 &&
                    shard.dims[1] == this->ngramHeadDim &&
                    !shard.diskWeightParts.empty(),
                "Qwen4-Exp disk PLE shard has invalid metadata: " + name +
                    "\n");
            if (shardIndex == 0) {
                dataType = shard.dataType;
            } else {
                AssertInFastLLM(shard.dataType == dataType,
                                "Qwen4-Exp disk PLE shards have mixed dtypes.\n");
            }
            totalRows += shard.dims[0];
            parts.insert(parts.end(), shard.diskWeightParts.begin(),
                         shard.diskWeightParts.end());
        }
        AssertInFastLLM(
            totalRows > 0 && totalRows <= std::numeric_limits<int>::max(),
            "Qwen4-Exp disk PLE table is too large for int32 row indices.\n");

        this->pleNgramDiskWeight.dataType = dataType;
        this->pleNgramDiskWeight.UpdateUnitSize();
        this->pleNgramDiskWeight.Resize(
            {(int)totalRows, this->ngramHeadDim});
        this->pleNgramDiskWeight.name = embeddingPrefix + "weight";
        this->pleNgramDiskWeight.isModelWeight = true;
        this->pleNgramDiskWeight.weightType = WeightType::EMBEDDING;
        this->pleNgramDiskWeight.dataDevice = DataDevice::CPU;
        this->pleNgramDiskWeight.isDiskWeight = true;
        this->pleNgramDiskWeight.diskWeightParts = std::move(parts);
    }

    bool Qwen4ExpModel::IsLinearAttentionLayer(int layer) const {
        return layer >= 0 && layer < (int)this->linearLayers.size() &&
               this->linearLayers[layer];
    }

    void Qwen4ExpModel::PrepareWeights() {
        std::lock_guard<std::mutex> guard(this->prepareMutex);
        if (this->preparedWeights) {
            return;
        }

        this->weights.assign(this->block_cnt, {});
        this->biass.assign(this->block_cnt, {});
        this->qsaKeyNormValues.clear();
        for (int layer = 0; layer < this->block_cnt; layer++) {
            this->weights[layer].push_back(nullptr);
            this->weights[layer].push_back(nullptr);
            this->biass[layer].push_back(nullptr);
            this->biass[layer].push_back(nullptr);
            const std::string experts = languagePrefix + "layers." +
                std::to_string(layer) + ".mlp.experts.";
            for (int expert = 0; expert < this->num_experts; expert++) {
                const std::string prefix = experts + std::to_string(expert) + ".";
                this->weights[layer].push_back(&this->weight[prefix + "gateup_proj.weight"]);
                this->weights[layer].push_back(&this->weight[prefix + "down_proj.weight"]);
                this->biass[layer].push_back(nullptr);
                this->biass[layer].push_back(nullptr);
            }

            for (const std::string &connection : {
                     "attn_hyper_connection", "mlp_hyper_connection"}) {
                const std::string connectionPrefix =
                    languagePrefix + "layers." + std::to_string(layer) +
                    "." + connection + ".";
                Qwen4AddOne(this->weight[
                    connectionPrefix + "hc_norm.weight"]);
            }
            if (!this->IsLinearAttentionLayer(layer)) {
                const std::string attention = languagePrefix + "layers." +
                    std::to_string(layer) + ".self_attn.";
                Qwen4AddOne(this->weight[attention + "q_norm.weight"]);
                Qwen4AddOne(this->weight[attention + "k_norm.weight"]);
                Qwen4AddOne(this->weight[attention + "indexer.q_layernorm.weight"]);
                Data &indexKeyNorm = this->weight[
                    attention + "indexer.k_layernorm.weight"];
                Qwen4AddOne(indexKeyNorm);
                AssertInFastLLM(
                    indexKeyNorm.dataType == DataType::FLOAT32 &&
                    indexKeyNorm.dataDevice == DataDevice::CPU &&
                    indexKeyNorm.cpuData != nullptr &&
                    indexKeyNorm.Count(0) == (uint64_t)this->indexerHeadDim,
                    "Qwen4-Exp indexer key norm has an invalid shape.");
                const float *values =
                    reinterpret_cast<const float *>(indexKeyNorm.cpuData);
                this->qsaKeyNormValues[layer].assign(
                    values, values + this->indexerHeadDim);
            }
            if (layer == this->pleLayer) {
                const std::string ple = languagePrefix + "layers." +
                    std::to_string(layer) + ".ple.";
                Qwen4AddOne(this->weight[ple + "norm_key.weight"]);
                Qwen4AddOne(this->weight[ple + "norm_query.weight"]);
                Qwen4AddOne(this->weight[ple + "norm_conv.weight"]);
            }
        }
        Qwen4AddOne(this->weight[languagePrefix +
            "hyper_connection_mixer.hc_norm.weight"]);
        this->preparedWeights = true;
    }

    void Qwen4ExpModel::GroupedRMSNorm(const Data &input, Data &normWeight,
                                        Data &output) {
        AssertInFastLLM(!input.dims.empty() &&
                        input.dims.back() == this->hcCount * this->embed_dim,
                        "Qwen4-Exp grouped RMSNorm received an invalid hidden size.");
        fastllm::Qwen4GroupedRMSNorm(input, normWeight,
                                    this->rms_norm_eps,
                                    this->hcCount, output);
    }

    void Qwen4ExpModel::HyperMixNormalized(
            Data &normalized, const std::string &prefix,
            Data &mixedInput, Data *injectionWeights) {
        Data lowRank, inputMix;
        Linear(normalized,
               this->weight[prefix + "input_mix_weight_down.weight"],
               Data(), lowRank);
        if (injectionWeights != nullptr) {
            Linear(normalized,
                   this->weight[prefix + "block_inject_weight.weight"],
                   Data(), *injectionWeights);
            fastllm::Qwen4HyperPrepare(
                lowRank, this->hcCount, lowRank);
            fastllm::Qwen4HyperInject(
                *injectionWeights, this->hcCount, *injectionWeights);
        } else {
            Mul(lowRank, 1.0f / (float)this->hcCount, lowRank);
            Silu(lowRank, lowRank);
        }
        Linear(lowRank, this->weight[prefix + "input_mix_weight_up.weight"],
               Data(), inputMix);
        fastllm::Qwen4HyperMix(normalized, inputMix,
                              this->hcCount, mixedInput);

    }

    void Qwen4ExpModel::HyperCombine(const Data &hyperInput,
                                      const Data &blockOutput,
                                      const Data &injectionWeights,
                                      Data &output) {
        // The operation rounds mixed-dtype block/injection values to the
        // residual dtype internally, matching the previous explicit casts
        // without materializing temporary tensors.
        fastllm::Qwen4HyperCombine(hyperInput, blockOutput,
                                  injectionWeights, this->hcCount,
                                  output);
    }

    void Qwen4ExpModel::HyperCombineRMSNorm(
            const Data &hyperInput, const Data &blockOutput,
            const Data &injectionWeights, Data &normWeight,
            Data &output, Data &normalized) {
        fastllm::Qwen4HyperCombineRMSNorm(
            hyperInput, blockOutput, injectionWeights, normWeight,
            this->rms_norm_eps, this->hcCount, output, normalized);
    }

    void Qwen4ExpModel::RunPLE(const Data &hyperInput,
                               const Data &inputIds,
                               RequestState &state,
                               Data &output) {
        AssertInFastLLM(inputIds.dims.size() == 2 && inputIds.dims[0] == 1,
                        "Qwen4-Exp PLE currently expects one request per forward.");
        Data idsCpu;
        ToDataType(inputIds, idsCpu, DataType::FLOAT32);
        idsCpu.ToDevice(DataDevice::CPU);
        const float *ids = reinterpret_cast<const float *>(idsCpu.cpuData);
        const int batch = inputIds.dims[0];
        const int sequence = inputIds.dims[1];
        state.processedTokens.reserve(state.processedTokens.size() + sequence);
        for (int token = 0; token < sequence; token++) {
            state.processedTokens.push_back((int)(ids[token] + 0.01f));
        }

        const std::string ple = languagePrefix + "layers." +
            std::to_string(this->pleLayer) + ".ple.";
        const std::string embeddingPrefix = ple +
            "ple_embedding.ngram_embedding.";
        Data &firstShard = this->weight[embeddingPrefix + "shard_0.weight"];
        const bool fp8Embedding = firstShard.dataType == DataType::FP8_E4M3;
        const bool bf16Embedding = firstShard.dataType == DataType::BFLOAT16;
        const bool diskEmbedding = firstShard.isDiskWeight;
        AssertInFastLLM((fp8Embedding || bf16Embedding) &&
                        firstShard.dims.size() == 2 &&
                        firstShard.dims[1] == this->ngramHeadDim,
                        "Qwen4-Exp PLE shard has an unexpected dtype or shape.");
        AssertInFastLLM(
            !diskEmbedding ||
                (this->pleNgramDiskWeight.isDiskWeight &&
                 this->pleNgramDiskWeight.dataType == firstShard.dataType &&
                 this->pleNgramDiskWeight.dims.size() == 2 &&
                 this->pleNgramDiskWeight.dims[1] == this->ngramHeadDim),
            "Qwen4-Exp disk PLE table was not initialized.");
        const int64_t rowsPerShard = firstShard.dims[0];

        float embeddingScale = 1.0f;
        if (fp8Embedding) {
            const std::string scaleName = embeddingPrefix + "weight_scale";
            auto scaleIt = this->weight.weight.find(scaleName);
            AssertInFastLLM(scaleIt != this->weight.weight.end(),
                            "Qwen4-Exp FP8 PLE shard is missing weight_scale.");
            Data &scaleWeight = scaleIt->second;
            scaleWeight.ToDevice(DataDevice::CPU);
            ToDataType(scaleWeight, DataType::FLOAT32);
            embeddingScale =
                reinterpret_cast<const float *>(scaleWeight.cpuData)[0];
        }
        static const FP8E4M3ToFP32Manager fp8Decoder;

        std::vector<float> embeddings((size_t)batch * sequence * this->pleEmbedDim);
        std::vector<int32_t> diskRows;
        if (diskEmbedding) {
            diskRows.resize((size_t)batch * sequence * this->ngramHeads);
        }
        int previous1 = state.previousToken1 < 0 ? this->eosToken : state.previousToken1;
        int previous2 = state.previousToken2 < 0 ? this->eosToken : state.previousToken2;
        for (int tokenIndex = 0; tokenIndex < sequence; tokenIndex++) {
            const int current = (int)(ids[tokenIndex] + 0.01f);
            const int shifted[3] = {current, previous1, previous2};
            for (int ngram = 2; ngram <= this->ngramSize; ngram++) {
                uint64_t mixedBits =
                    (uint64_t)(int64_t)shifted[0] * this->pleMultipliers[0];
                for (int position = 1; position < ngram; position++) {
                    mixedBits ^= (uint64_t)(int64_t)shifted[position] *
                                 this->pleMultipliers[position];
                }
                int64_t mixedSigned;
                std::memcpy(&mixedSigned, &mixedBits, sizeof(mixedSigned));
                const int headStart = (ngram - 2) * this->headsPerNgram;
                for (int localHead = 0; localHead < this->headsPerNgram;
                     localHead++) {
                    const int head = headStart + localHead;
                    const int64_t vocab = this->pleHeadVocabSizes[head];
                    int64_t remainder = mixedSigned % vocab;
                    if (remainder < 0) {
                        remainder += vocab;
                    }
                    const int64_t globalRow =
                        this->pleHeadOffsets[head] + remainder;
                    const size_t lookupIndex =
                        (size_t)tokenIndex * this->ngramHeads + head;
                    if (diskEmbedding) {
                        AssertInFastLLM(
                            globalRow >= 0 &&
                                globalRow < this->pleNgramDiskWeight.dims[0],
                            "Qwen4-Exp PLE hash selected an invalid disk row.");
                        diskRows[lookupIndex] = (int32_t)globalRow;
                        continue;
                    }
                    const int shardIndex = (int)(globalRow / rowsPerShard);
                    const int64_t shardRow = globalRow % rowsPerShard;
                    AssertInFastLLM(shardIndex >= 0 &&
                                    shardIndex < this->ngramShardCount,
                                    "Qwen4-Exp PLE hash selected an invalid shard.");
                    Data &shard = this->weight[embeddingPrefix + "shard_" +
                                               std::to_string(shardIndex) +
                                               ".weight"];
                    shard.ToDevice(DataDevice::CPU);
                    float *destination = embeddings.data() +
                        lookupIndex * this->ngramHeadDim;
                    if (fp8Embedding) {
                        const uint8_t *source = shard.cpuData +
                            (size_t)shardRow * this->ngramHeadDim;
                        for (int column = 0; column < this->ngramHeadDim; column++) {
                            destination[column] =
                                fp8Decoder.dict[source[column]] * embeddingScale;
                        }
                    } else {
                        const uint16_t *source =
                            reinterpret_cast<const uint16_t *>(shard.cpuData) +
                            (size_t)shardRow * this->ngramHeadDim;
                        for (int column = 0; column < this->ngramHeadDim; column++) {
                            destination[column] =
                                BFloat16BitsToFloat32(source[column]);
                        }
                    }
                }
            }

            if (current == this->eosToken) {
                previous1 = this->eosToken;
                previous2 = this->eosToken;
            } else {
                previous2 = previous1;
                previous1 = current;
            }
        }
        state.previousToken1 = previous1;
        state.previousToken2 = previous2;

        if (diskEmbedding) {
            Data lookupRows(DataType::INT32,
                            {batch, sequence, this->ngramHeads});
            lookupRows.Allocate(false);
            std::memcpy(lookupRows.cpuData, diskRows.data(),
                        diskRows.size() * sizeof(int32_t));
            Data diskValues;
            auto *executor = (Executor*)GetExecutor();
            executor->RunOnDevice(
                "disk", "EmbeddingDirect",
                {{"input", &lookupRows},
                 {"weight", &this->pleNgramDiskWeight},
                 {"output", &diskValues}},
                {}, {});
            AssertInFastLLM(
                diskValues.dataDevice == DataDevice::CPU &&
                    diskValues.cpuData != nullptr &&
                    diskValues.dataType == firstShard.dataType &&
                    diskValues.Count(0) ==
                        (uint64_t)diskRows.size() * this->ngramHeadDim,
                "Qwen4-Exp disk PLE lookup returned an invalid tensor.");

            const size_t valueCount =
                diskRows.size() * (size_t)this->ngramHeadDim;
            if (fp8Embedding) {
                const uint8_t *source = diskValues.cpuData;
                for (size_t i = 0; i < valueCount; i++) {
                    embeddings[i] =
                        fp8Decoder.dict[source[i]] * embeddingScale;
                }
            } else {
                const uint16_t *source =
                    reinterpret_cast<const uint16_t *>(diskValues.cpuData);
                for (size_t i = 0; i < valueCount; i++) {
                    embeddings[i] = BFloat16BitsToFloat32(source[i]);
                }
            }
        }

        Data embeddingData(DataType::FLOAT32,
                           {batch, sequence, this->pleEmbedDim}, embeddings);
        Data key, value, keyNormed, queryNormed;
        Linear(embeddingData, this->weight[ple + "key_proj.weight"], Data(), key);
        Linear(embeddingData, this->weight[ple + "value_proj.weight"], Data(), value);
        GroupedRMSNorm(key, this->weight[ple + "norm_key.weight"], keyNormed);
        GroupedRMSNorm(hyperInput, this->weight[ple + "norm_query.weight"],
                       queryNormed);

        // The ngram lookup above intentionally remains on the host. For a
        // multi-token CUDA prefill, keep the much larger projection tail on
        // the device: the legacy path copied key/query/value/normalized to
        // CPU and the final result back to CUDA (about 880 MiB at 4K).
        // Single-token decode still uses the legacy implementation until its
        // convolution history becomes a persistent device cache.
        if (sequence > 1 &&
            keyNormed.dataDevice == DataDevice::CUDA &&
            queryNormed.dataDevice == DataDevice::CUDA &&
            value.dataDevice == DataDevice::CUDA) {
            const int channels = this->hcCount * this->embed_dim;
            const int dilation = this->ngramSize;
            const int historyLength =
                (this->pleConvKernel - 1) * dilation;
            if ((int)state.convHistory.size() != historyLength * channels) {
                state.convHistory.assign(
                    (size_t)historyLength * channels, 0.0f);
            }

            Data gatedData, normalized;
            fastllm::Qwen4PLEGate(
                keyNormed, queryNormed, value,
                this->hcCount, gatedData);
            GroupedRMSNorm(
                gatedData, this->weight[ple + "norm_conv.weight"],
                normalized);

            Data &convWeight = this->weight[ple + "conv1d.weight"];
            if (convWeight.dataType != DataType::FLOAT32) {
                convWeight.ToDevice(DataDevice::CPU);
                ToDataType(convWeight, DataType::FLOAT32);
            }
            convWeight.ToDevice(normalized.dataDevice);
            AssertInFastLLM(
                convWeight.dims.size() == 3 &&
                convWeight.dims[0] == channels &&
                convWeight.dims.back() == this->pleConvKernel,
                "Qwen4-Exp PLE convolution weight has an invalid shape.");

            Data historyData(
                DataType::FLOAT32, {historyLength, channels},
                state.convHistory);
            historyData.ToDevice(normalized.dataDevice);
            Data nextHistory;
            fastllm::Qwen4PLECausalConv(
                normalized, gatedData, convWeight, historyData,
                this->pleConvKernel, dilation, output, nextHistory);
            nextHistory.ToDevice(DataDevice::CPU);
            std::memcpy(state.convHistory.data(), nextHistory.cpuData,
                        state.convHistory.size() * sizeof(float));
            ToDataType(output, hyperInput.dataType);
            output.ToDevice(hyperInput.dataDevice);
            return;
        }

        Data keyCpu, queryCpu, valueCpu;
        ToDataType(keyNormed, keyCpu, DataType::FLOAT32);
        ToDataType(queryNormed, queryCpu, DataType::FLOAT32);
        ToDataType(value, valueCpu, DataType::FLOAT32);
        keyCpu.ToDevice(DataDevice::CPU);
        queryCpu.ToDevice(DataDevice::CPU);
        valueCpu.ToDevice(DataDevice::CPU);
        const float *keyValues = reinterpret_cast<const float *>(keyCpu.cpuData);
        const float *queryValues = reinterpret_cast<const float *>(queryCpu.cpuData);
        const float *values = reinterpret_cast<const float *>(valueCpu.cpuData);

        const int channels = this->hcCount * this->embed_dim;
        std::vector<float> gated((size_t)sequence * channels);
        const float inverseSqrtHidden = 1.0f / std::sqrt((float)this->embed_dim);
        for (int token = 0; token < sequence; token++) {
            for (int stream = 0; stream < this->hcCount; stream++) {
                const size_t base = (size_t)token * channels +
                                    (size_t)stream * this->embed_dim;
                float dot = 0.0f;
                for (int column = 0; column < this->embed_dim; column++) {
                    dot += keyValues[base + column] *
                           queryValues[base + column];
                }
                float gate = dot * inverseSqrtHidden;
                if (gate != 0.0f) {
                    gate = std::copysign(
                        std::sqrt(std::max(std::fabs(gate), 1e-6f)), gate);
                }
                const float probability = 1.0f / (1.0f + std::exp(-gate));
                const float *tokenValue = values +
                    (size_t)token * this->embed_dim;
                for (int column = 0; column < this->embed_dim; column++) {
                    gated[base + column] = probability * tokenValue[column];
                }
            }
        }

        Data gatedData(DataType::FLOAT32,
                       {batch, sequence, channels}, gated);
        Data normalized;
        GroupedRMSNorm(gatedData, this->weight[ple + "norm_conv.weight"],
                       normalized);
        Data normalizedCpu;
        ToDataType(normalized, normalizedCpu, DataType::FLOAT32);
        normalizedCpu.ToDevice(DataDevice::CPU);
        const float *normalizedValues =
            reinterpret_cast<const float *>(normalizedCpu.cpuData);

        Data &convWeight = this->weight[ple + "conv1d.weight"];
        convWeight.ToDevice(DataDevice::CPU);
        ToDataType(convWeight, DataType::FLOAT32);
        AssertInFastLLM(convWeight.dims.size() == 3 &&
                        convWeight.dims[0] == channels &&
                        convWeight.dims.back() == this->pleConvKernel,
                        "Qwen4-Exp PLE convolution weight has an invalid shape.");
        const float *kernel = reinterpret_cast<const float *>(convWeight.cpuData);
        const int dilation = this->ngramSize;
        const int historyLength = (this->pleConvKernel - 1) * dilation;
        if ((int)state.convHistory.size() != historyLength * channels) {
            state.convHistory.assign((size_t)historyLength * channels, 0.0f);
        }

        std::vector<float> result((size_t)sequence * channels);
        for (int token = 0; token < sequence; token++) {
            for (int channel = 0; channel < channels; channel++) {
                float sum = 0.0f;
                for (int tap = 0; tap < this->pleConvKernel; tap++) {
                    const int relative = token - historyLength + tap * dilation;
                    float sample = 0.0f;
                    if (relative >= 0) {
                        sample = normalizedValues[(size_t)relative * channels +
                                                  channel];
                    } else {
                        sample = state.convHistory[
                            (size_t)(historyLength + relative) * channels +
                            channel];
                    }
                    sum += sample * kernel[(size_t)channel * this->pleConvKernel + tap];
                }
                const size_t index = (size_t)token * channels + channel;
                result[index] = gated[index] + Qwen4Silu(sum);
            }
        }

        std::vector<float> newHistory((size_t)historyLength * channels);
        for (int history = 0; history < historyLength; history++) {
            const int sourceToken = sequence - historyLength + history;
            const float *source;
            if (sourceToken >= 0) {
                source = normalizedValues + (size_t)sourceToken * channels;
            } else {
                source = state.convHistory.data() +
                    (size_t)(historyLength + sourceToken) * channels;
            }
            std::memcpy(newHistory.data() + (size_t)history * channels,
                        source, (size_t)channels * sizeof(float));
        }
        state.convHistory.swap(newHistory);

        output.CopyFrom(Data(DataType::FLOAT32,
                             {batch, sequence, channels}, result));
        ToDataType(output, hyperInput.dataType);
        output.ToDevice(hyperInput.dataDevice);
    }

    void Qwen4ExpModel::MaterializeQsaHostHistory(
            int layer, int length, RequestState &state) {
        AssertInFastLLM(
            length >= 0 && this->indexerCompressRatio > 0,
            "Qwen4-Exp QSA host history received an invalid length.");
        std::vector<float> &rawKeys = state.indexerRawKeys[layer];
        std::vector<float> &positions = state.indexerPositions[layer];
        const size_t fullRawCount =
            (size_t)length * this->indexerHeadDim;
        if (rawKeys.size() == fullRawCount &&
            positions.size() == (size_t)length) {
            return;
        }
#ifdef USE_CUDA
        const auto transfer =
            state.indexerHostMirrorTransfers.find(layer);
        if (transfer != state.indexerHostMirrorTransfers.end() &&
            transfer->second != nullptr) {
            transfer->second->Materialize(rawKeys, positions);
        }
#endif
        if (rawKeys.size() == fullRawCount &&
            positions.size() == (size_t)length) {
            return;
        }

        const int tailCount = length % this->indexerCompressRatio;
        const int completedLength = length - tailCount;
        AssertInFastLLM(
            rawKeys.size() ==
                    (size_t)completedLength * this->indexerHeadDim &&
                positions.size() == (size_t)completedLength &&
                tailCount > 0,
            "Qwen4-Exp QSA host history is missing completed rows.");
        const auto tailKeys = state.indexerTailKeyTensors.find(layer);
        const auto tailPositions =
            state.indexerTailPositionTensors.find(layer);
        AssertInFastLLM(
            tailKeys != state.indexerTailKeyTensors.end() &&
                tailKeys->second != nullptr &&
                tailKeys->second->dataType == DataType::FLOAT32 &&
                tailKeys->second->dims ==
                    std::vector<int>({tailCount,
                                      this->indexerHeadDim}) &&
                tailPositions !=
                    state.indexerTailPositionTensors.end() &&
                tailPositions->second != nullptr &&
                tailPositions->second->dataType == DataType::FLOAT32 &&
                tailPositions->second->dims ==
                    std::vector<int>({tailCount}),
            "Qwen4-Exp QSA device tail cannot restore its host history.");

        Data rawTail, positionTail;
#ifdef USE_CUDA
        if (tailKeys->second->dataDevice == DataDevice::CUDA) {
            Qwen4BorrowCudaStorage(rawTail, *tailKeys->second);
        } else
#endif
        {
            rawTail.CopyFrom(*tailKeys->second);
        }
#ifdef USE_CUDA
        if (tailPositions->second->dataDevice == DataDevice::CUDA) {
            Qwen4BorrowCudaStorage(positionTail,
                                   *tailPositions->second);
        } else
#endif
        {
            positionTail.CopyFrom(*tailPositions->second);
        }
        rawTail.ToDevice(DataDevice::CPU);
        positionTail.ToDevice(DataDevice::CPU);
        const float *rawValues =
            reinterpret_cast<const float *>(rawTail.cpuData);
        const float *positionValues =
            reinterpret_cast<const float *>(positionTail.cpuData);
        rawKeys.insert(
            rawKeys.end(), rawValues,
            rawValues + (size_t)tailCount * this->indexerHeadDim);
        positions.insert(positions.end(), positionValues,
                         positionValues + tailCount);
    }

    void Qwen4ExpModel::BuildQSAMask(int layer,
                                      const Data &input,
                                      const Data &baseMask,
                                      const Data &positionIds,
                                      int previousLength,
                                      bool deviceCompatibleMask,
                                      RequestState &state,
                                      Data &qsaMask,
                                      Data &qsaIndices) {
        const int batch = input.dims[0];
        const int sequence = input.dims[1];
        AssertInFastLLM(batch == 1 && sequence > 0,
                        "Qwen4-Exp QSA currently expects one non-empty request.");

        const std::string indexer = languagePrefix + "layers." +
            std::to_string(layer) + ".self_attn.indexer.";
        Data typedInput, projected, query, currentRawKeys;
        ToDataType(input, typedInput, this->dataType);
        Linear(typedInput, this->weight[indexer + "index_qk_proj.weight"],
               Data(), projected);
        const int queryColumns = this->indexerHeads * this->indexerHeadDim;
        Split(projected, -1, 0, queryColumns, query);
        Split(projected, -1, queryColumns,
              queryColumns + this->indexerKvHeads * this->indexerHeadDim,
              currentRawKeys);
        query.Reshape({batch, sequence, this->indexerHeads,
                       this->indexerHeadDim});
        currentRawKeys.Reshape({batch, sequence, this->indexerHeadDim});
        RMSNorm(query, this->weight[indexer + "q_layernorm.weight"],
                this->rms_norm_eps, query);
        LlamaRotatePosition2DPart(query, positionIds, this->sinData,
                                  this->cosData, this->rotary_dim,
                                  this->rotary_dim);

        qsaIndices = Data();
        const int keyLength = previousLength + sequence;
        const int completeKeyBlocks = keyLength /
                                      this->indexerCompressRatio;

        // Preserve the raw history needed by the generic mask path: completed
        // groups are mirrored asynchronously and the unfinished group stays
        // in the device tail. Arbitrary visible-token regrouping cannot be
        // reconstructed from compressed block keys alone.
        std::vector<float> &rawKeyCache = state.indexerRawKeys[layer];
        std::vector<float> &positionCache = state.indexerPositions[layer];
        std::vector<float> &blockKeys = state.indexerBlockKeys[layer];
        auto normIt = this->qsaKeyNormValues.find(layer);
        AssertInFastLLM(normIt != this->qsaKeyNormValues.end() &&
                        normIt->second.size() ==
                            (size_t)this->indexerHeadDim,
                        "Qwen4-Exp QSA key norm host cache is unavailable.");
        const float *keyNorm = normIt->second.data();
        const float *sinValues = this->qsaSinValues.data();
        const float *cosValues = this->qsaCosValues.data();
        const int rotaryStride = this->sinData.dims.back();
        AssertInFastLLM(!this->qsaSinValues.empty() &&
                        this->qsaSinValues.size() == this->qsaCosValues.size() &&
                        rotaryStride >= this->rotary_dim &&
                        this->rotary_dim % 2 == 0,
                        "Qwen4-Exp QSA received invalid rotary tables.");
        const int rotaryHalf = this->rotary_dim / 2;

        auto updateHostBlockCache = [&](int hostKeyLength) {
            const int hostCompleteKeyBlocks =
                hostKeyLength / this->indexerCompressRatio;
            AssertInFastLLM(
                rawKeyCache.size() ==
                        (size_t)hostKeyLength * this->indexerHeadDim &&
                    positionCache.size() == (size_t)hostKeyLength,
                "Qwen4-Exp QSA host mirror is inconsistent with the attention cache.");
            AssertInFastLLM(
                blockKeys.size() % this->indexerHeadDim == 0,
                "Qwen4-Exp compressed QSA cache has an invalid shape.");
            const int cachedBlocks = (int)(blockKeys.size() /
                                           this->indexerHeadDim);
            AssertInFastLLM(
                cachedBlocks <= hostCompleteKeyBlocks,
                "Qwen4-Exp compressed QSA cache is ahead of raw keys.");
            blockKeys.resize((size_t)hostCompleteKeyBlocks *
                             this->indexerHeadDim);
            Qwen4ParallelFor(hostCompleteKeyBlocks - cachedBlocks,
                             [&](int start, int end) {
                std::vector<float> pooled(this->indexerHeadDim);
                for (int relative = start; relative < end; relative++) {
                    const int block = cachedBlocks + relative;
                    std::fill(pooled.begin(), pooled.end(), 0.0f);
                    const int groupStart =
                        block * this->indexerCompressRatio;
                    for (int member = 0;
                         member < this->indexerCompressRatio; member++) {
                        const float *source = rawKeyCache.data() +
                            (size_t)(groupStart + member) *
                                this->indexerHeadDim;
                        for (int column = 0;
                             column < this->indexerHeadDim; column++) {
                            pooled[column] += source[column];
                        }
                    }
                    float squareSum = 0.0f;
                    for (float value : pooled) {
                        const float averaged = value /
                            (float)this->indexerCompressRatio;
                        squareSum += averaged * averaged;
                    }
                    const float inverseRms = 1.0f / std::sqrt(
                        squareSum / this->indexerHeadDim +
                        this->rms_norm_eps);
                    for (int column = 0;
                         column < this->indexerHeadDim; column++) {
                        pooled[column] = pooled[column] /
                            (float)this->indexerCompressRatio * inverseRms *
                            keyNorm[column];
                    }
                    const int position =
                        (int)(positionCache[groupStart] + 0.01f);
                    AssertInFastLLM(
                        position >= 0 && position < this->sinData.dims[0],
                        "Qwen4-Exp QSA position exceeds its rotary table.");
                    const float *sinRow = sinValues +
                        (size_t)position * rotaryStride;
                    const float *cosRow = cosValues +
                        (size_t)position * rotaryStride;
                    float *destination = blockKeys.data() +
                        (size_t)block * this->indexerHeadDim;
                    std::copy(pooled.begin(), pooled.end(), destination);
                    for (int column = 0; column < rotaryHalf; column++) {
                        destination[column] =
                            pooled[column] * cosRow[column] -
                            pooled[column + rotaryHalf] * sinRow[column];
                        destination[column + rotaryHalf] =
                            pooled[column + rotaryHalf] *
                                cosRow[column + rotaryHalf] +
                            pooled[column] * sinRow[column + rotaryHalf];
                    }
                }
            });
        };

        // The ordinary single-request CUDA path keeps QSA construction on
        // the execution device. Only the incomplete compression group is
        // retained there as raw keys; completed groups are pooled, normalized
        // and rotated immediately. A pinned host mirror preserves the raw
        // fallback history without synchronizing between attention layers.
        const bool useDeviceQsa =
            deviceCompatibleMask &&
            query.dataDevice == DataDevice::CUDA &&
            currentRawKeys.dataDevice == DataDevice::CUDA &&
            positionIds.dataDevice == DataDevice::CUDA;
        if (useDeviceQsa) {
            const int ratio = this->indexerCompressRatio;
            AssertInFastLLM(ratio > 0,
                            "Qwen4-Exp QSA compression ratio must be positive.");
            const int oldTailCount = previousLength % ratio;
            const int cachedBlocks = previousLength / ratio;
#ifdef USE_CUDA
            std::shared_ptr<QsaHostMirrorTransfer> &hostTransfer =
                state.indexerHostMirrorTransfers[layer];
            if (hostTransfer == nullptr) {
                hostTransfer =
                    std::make_shared<QsaHostMirrorTransfer>();
            }
            hostTransfer->Prepare(
                rawKeyCache, positionCache,
                previousLength - oldTailCount,
                this->indexerHeadDim);
#endif

            Data currentKeysFloat, currentPositionsFloat;
            ToDataType(currentRawKeys, currentKeysFloat,
                       DataType::FLOAT32);
            currentKeysFloat.Reshape(
                {sequence, this->indexerHeadDim});
            if (positionIds.dataType == DataType::FLOAT32) {
                currentPositionsFloat.FakeFrom(positionIds, 0);
                currentPositionsFloat.Resize(positionIds.dims);
            } else {
                ToDataType(positionIds, currentPositionsFloat,
                           DataType::FLOAT32);
            }
            currentPositionsFloat.Reshape({sequence});

            auto sameDevice = [&](const Data &data) {
                if (data.dataDevice != DataDevice::CUDA ||
                    data.cudaData == nullptr) {
                    return false;
                }
                if (data.dataDeviceIds.empty() ||
                    query.dataDeviceIds.empty()) {
                    return true;
                }
                return data.dataDeviceIds[0] == query.dataDeviceIds[0];
            };
            auto reserveFirstAxis = [](Data &data, int capacity) {
                if (data.dims.empty() || capacity <= data.dims[0]) {
                    return;
                }
                std::vector<int> expanded = data.dims;
                expanded[0] = capacity;
                data.Expansion(expanded);
            };

            std::shared_ptr<Data> &tailKeys =
                state.indexerTailKeyTensors[layer];
            std::shared_ptr<Data> &tailPositions =
                state.indexerTailPositionTensors[layer];
            std::shared_ptr<Data> &blockCache =
                state.indexerBlockKeyTensors[layer];
            const bool validTailKeys = oldTailCount == 0 ||
                (tailKeys != nullptr && sameDevice(*tailKeys) &&
                 tailKeys->dataType == DataType::FLOAT32 &&
                 tailKeys->dims ==
                     std::vector<int>({oldTailCount,
                                       this->indexerHeadDim}));
            const bool validTailPositions = oldTailCount == 0 ||
                (tailPositions != nullptr && sameDevice(*tailPositions) &&
                 tailPositions->dataType == DataType::FLOAT32 &&
                 tailPositions->dims ==
                     std::vector<int>({oldTailCount}));
            const bool validBlocks = cachedBlocks == 0 ||
                (blockCache != nullptr && sameDevice(*blockCache) &&
                 blockCache->dataType == DataType::FLOAT32 &&
                 blockCache->dims ==
                     std::vector<int>({cachedBlocks,
                                       this->indexerHeadDim}));

            // Device changes and snapshots created by an older runtime can
            // still materialize from the established host representation.
            // Normal CUDA requests never enter this compatibility branch.
            if (!validTailKeys || !validTailPositions || !validBlocks) {
                MaterializeQsaHostHistory(
                    layer, previousLength, state);
                updateHostBlockCache(previousLength);
                const auto raw = state.indexerRawKeys.find(layer);
                const auto positions = state.indexerPositions.find(layer);
                const auto blocks = state.indexerBlockKeys.find(layer);
                const size_t expectedRaw = (size_t)previousLength *
                    this->indexerHeadDim;
                const size_t expectedBlocks = (size_t)cachedBlocks *
                    this->indexerHeadDim;
                AssertInFastLLM(
                    raw != state.indexerRawKeys.end() &&
                    raw->second.size() == expectedRaw &&
                    positions != state.indexerPositions.end() &&
                    positions->second.size() == (size_t)previousLength &&
                    blocks != state.indexerBlockKeys.end() &&
                    blocks->second.size() == expectedBlocks,
                    "Qwen4-Exp QSA device cache is inconsistent with the attention cache.");

                if (oldTailCount > 0) {
                    const size_t rawOffset =
                        (size_t)(previousLength - oldTailCount) *
                        this->indexerHeadDim;
                    std::vector<float> rawTail(
                        raw->second.begin() + rawOffset,
                        raw->second.end());
                    std::vector<float> positionTail(
                        positions->second.end() - oldTailCount,
                        positions->second.end());
                    tailKeys = std::make_shared<Data>(
                        DataType::FLOAT32,
                        std::vector<int>({oldTailCount,
                                          this->indexerHeadDim}),
                        rawTail);
                    tailPositions = std::make_shared<Data>(
                        DataType::FLOAT32,
                        std::vector<int>({oldTailCount}),
                        positionTail);
                    tailKeys->ToDevice(
                        DataDevice::CUDA, query.dataDeviceIds);
                    tailPositions->ToDevice(
                        DataDevice::CUDA, query.dataDeviceIds);
                    reserveFirstAxis(*tailKeys, ratio);
                    reserveFirstAxis(*tailPositions, ratio);
                } else {
                    tailKeys.reset();
                    tailPositions.reset();
                }
                if (cachedBlocks > 0) {
                    blockCache = std::make_shared<Data>(
                        DataType::FLOAT32,
                        std::vector<int>({cachedBlocks,
                                          this->indexerHeadDim}),
                        blocks->second);
                    blockCache->ToDevice(
                        DataDevice::CUDA, query.dataDeviceIds);
                } else {
                    blockCache.reset();
                }
            }

            auto finishDeviceQsa = [&]() {
                const int completeKeyBlocks = keyLength / ratio;
                AssertInFastLLM(
                    completeKeyBlocks == 0 ||
                    (blockCache != nullptr &&
                     blockCache->dims ==
                         std::vector<int>({completeKeyBlocks,
                                           this->indexerHeadDim})),
                    "Qwen4-Exp QSA compressed device cache is incomplete.");

                if (keyLength <= this->indexerBudget) {
                    if (baseMask.dims.empty()) {
                        qsaMask = Data();
                    } else {
                        qsaMask.CopyFrom(baseMask);
                    }
                    return;
                }

                const int sparsePrefillMinContext = std::max(
                    this->indexerBudget,
                    kQwen4SparsePrefillMinContext);
                const bool useSparseAttention = sequence == 1 ||
                    (sequence > 1 &&
                     keyLength > sparsePrefillMinContext);
                Qwen4QSASelect(
                    query, *blockCache, keyLength,
                    this->indexerHeads, this->indexerHeadDim,
                    this->indexerBudget, ratio, qsaIndices,
                    sequence > 1 ? previousLength : -1);
                if (useSparseAttention) {
                    qsaMask = Data();
                } else {
                    Qwen4QSABuildMask(
                        qsaIndices, input, keyLength, qsaMask);
                    qsaIndices = Data();
                }
            };

            // Most decode steps merely extend the incomplete group. Reuse
            // its small reserved buffer directly; constructing and then
            // splitting a temporary concatenation would add four D2D copies
            // per full-attention layer without producing a compressed row.
            if (oldTailCount + sequence < ratio) {
                if (tailKeys == nullptr) {
                    tailKeys = std::make_shared<Data>();
                    tailKeys->CopyFrom(currentKeysFloat);
                    reserveFirstAxis(*tailKeys, ratio);
                } else {
                    CatDirect(*tailKeys, currentKeysFloat, 0);
                }
                if (tailPositions == nullptr) {
                    tailPositions = std::make_shared<Data>();
                    tailPositions->CopyFrom(currentPositionsFloat);
                    reserveFirstAxis(*tailPositions, ratio);
                } else {
                    CatDirect(*tailPositions,
                              currentPositionsFloat, 0);
                }
                finishDeviceQsa();
                return;
            }

            Data combinedKeysStorage, combinedPositionsStorage;
            Data *combinedKeys = &currentKeysFloat;
            Data *combinedPositions = &currentPositionsFloat;
            if (oldTailCount > 0) {
                Cat(*tailKeys, currentKeysFloat, 0,
                    combinedKeysStorage);
                Cat(*tailPositions, currentPositionsFloat, 0,
                    combinedPositionsStorage);
                combinedKeys = &combinedKeysStorage;
                combinedPositions = &combinedPositionsStorage;
            }

            const int combinedCount = oldTailCount + sequence;
            const int newBlockCount = combinedCount / ratio;
            const int tailCount = combinedCount % ratio;
            if (newBlockCount > 0) {
                Data completeKeys;
                Split(*combinedKeys, 0, 0,
                      newBlockCount * ratio, completeKeys);
                completeKeys.Reshape(
                    {newBlockCount, ratio, this->indexerHeadDim});

                Data pooled;
                Split(completeKeys, 1, 0, 1, pooled);
                pooled.Reshape(
                    {newBlockCount, this->indexerHeadDim});
                for (int member = 1; member < ratio; member++) {
                    Data memberKeys;
                    Split(completeKeys, 1, member, member + 1,
                          memberKeys);
                    memberKeys.Reshape(
                        {newBlockCount, this->indexerHeadDim});
                    AddTo(pooled, memberKeys);
                }

                Data averaged, normalized;
                Mul(pooled, 1.0f / (float)ratio, averaged);
                RMSNorm(
                    averaged,
                    this->weight[indexer + "k_layernorm.weight"],
                    this->rms_norm_eps, normalized);

                Data completePositions;
                Data firstPositions;
                Split(*combinedPositions, 0, 0,
                      newBlockCount * ratio, completePositions);
                completePositions.Reshape({newBlockCount, ratio});
                Split(completePositions, 1, 0, 1,
                      firstPositions);
                firstPositions.Reshape({1, newBlockCount});
                normalized.Reshape(
                    {1, newBlockCount, 1, this->indexerHeadDim});
                LlamaRotatePosition2DPart(
                    normalized, firstPositions,
                    this->sinData, this->cosData,
                    this->rotary_dim, this->rotary_dim);
                normalized.Reshape(
                    {newBlockCount, this->indexerHeadDim});
#ifdef USE_CUDA
                // Queue the mirror after every consumer of these rows on the
                // per-thread stream. Later work and allocator reuse on that
                // stream remain ordered behind these D2H copies.
                hostTransfer->Queue(
                    completeKeys, completePositions,
                    previousLength - oldTailCount,
                    newBlockCount * ratio,
                    this->indexerHeadDim);
#endif

                AssertInFastLLM(
                    cachedBlocks == 0 ||
                    (blockCache != nullptr &&
                     blockCache->dims ==
                         std::vector<int>({cachedBlocks,
                                           this->indexerHeadDim})),
                    "Qwen4-Exp QSA compressed cache append is inconsistent.");
                if (blockCache == nullptr) {
                    blockCache = std::make_shared<Data>();
                    blockCache->CopyFrom(normalized);
                } else {
                    const int requiredBlocks =
                        cachedBlocks + newBlockCount;
                    if (blockCache->expansionDims.empty() ||
                        blockCache->expansionDims[0] < requiredBlocks) {
                        const int capacity =
                            ((requiredBlocks + 255) / 256) * 256;
                        blockCache->Expansion(
                            {capacity, this->indexerHeadDim});
                    }
                    CatDirect(*blockCache, normalized, 0);
                }
            }

            auto storeTail = [&](Data &source,
                                 std::shared_ptr<Data> &cache,
                                 const std::vector<int> &emptyShape) {
                if (tailCount == 0) {
                    if (cache != nullptr) {
                        cache->Resize(emptyShape);
                    }
                    return;
                }
                Data newTail;
                Split(source, 0, newBlockCount * ratio,
                      combinedCount, newTail);
                if (cache == nullptr || !sameDevice(*cache) ||
                    cache->dataType != DataType::FLOAT32) {
                    cache = std::make_shared<Data>();
                    cache->CopyFrom(newTail);
                    reserveFirstAxis(*cache, ratio);
                } else {
                    cache->Resize(emptyShape);
                    CatDirect(*cache, newTail, 0);
                }
            };
            storeTail(*combinedKeys, tailKeys,
                      {0, this->indexerHeadDim});
            storeTail(*combinedPositions, tailPositions, {0});

            const int completeKeyBlocks = keyLength / ratio;
            AssertInFastLLM(
                completeKeyBlocks == cachedBlocks + newBlockCount &&
                (completeKeyBlocks == 0 ||
                 (blockCache != nullptr &&
                  blockCache->dims ==
                      std::vector<int>({completeKeyBlocks,
                                        this->indexerHeadDim}))),
                "Qwen4-Exp QSA compressed device cache is incomplete.");

            finishDeviceQsa();
            return;
        }

        MaterializeQsaHostHistory(layer, previousLength, state);
#ifdef USE_CUDA
        // The generic path appends directly to the vectors below. Retire the
        // device-side mirror so a later host step cannot overwrite those rows
        // with an older completed-group prefix.
        state.indexerHostMirrorTransfers.erase(layer);
#endif

        Data rawKeysCpu, positionsCpu;
        ToDataType(currentRawKeys, rawKeysCpu, DataType::FLOAT32);
        ToDataType(positionIds, positionsCpu, DataType::FLOAT32);
        rawKeysCpu.ToDevice(DataDevice::CPU);
        positionsCpu.ToDevice(DataDevice::CPU);
        const float *rawKeyValues =
            reinterpret_cast<const float *>(rawKeysCpu.cpuData);
        const float *currentPositions =
            reinterpret_cast<const float *>(positionsCpu.cpuData);

        AssertInFastLLM(rawKeyCache.size() % this->indexerHeadDim == 0 &&
                        (int)(rawKeyCache.size() / this->indexerHeadDim) ==
                            previousLength &&
                        (int)positionCache.size() == previousLength,
                        "Qwen4-Exp QSA cache is inconsistent with the attention cache.");
        rawKeyCache.insert(rawKeyCache.end(), rawKeyValues,
                           rawKeyValues + (size_t)sequence *
                                              this->indexerHeadDim);
        positionCache.insert(positionCache.end(), currentPositions,
                             currentPositions + sequence);

        // vLLM's QSA cache writes one compressed row only when a group is
        // completed. Keep the same invariant here: old rows are immutable and
        // a decode step computes at most one new 4-token row.
        updateHostBlockCache(keyLength);

        // Up to the configured token budget every complete block is selected,
        // so QSA is exactly the caller's ordinary causal/padding mask.  We
        // still ran the projection above because raw index keys must be cached
        // for a later decode step that crosses the budget.
        if (keyLength <= this->indexerBudget) {
            if (baseMask.dims.empty()) {
                qsaMask = Data();
            } else {
                qsaMask.CopyFrom(baseMask);
            }
            return;
        }

        // Compatibility path for callers that execute QSA projection on CUDA
        // but provide host-side position IDs (or otherwise cannot retain the
        // incremental device cache). Keep selection and mask construction on
        // CUDA after building the snapshot-safe host representation.
        const bool useLegacyDeviceQsa =
            baseMask.dims.empty() &&
            query.dataDevice == DataDevice::CUDA;
        if (useLegacyDeviceQsa) {
            std::shared_ptr<Data> &cacheTensor =
                state.indexerBlockKeyTensors[layer];
            bool rebuild = cacheTensor == nullptr ||
                cacheTensor->dataDevice != query.dataDevice ||
                cacheTensor->dataType != DataType::FLOAT32 ||
                cacheTensor->dims.size() != 2 ||
                cacheTensor->dims[1] != this->indexerHeadDim ||
                cacheTensor->dims[0] > completeKeyBlocks;
            if (rebuild) {
                cacheTensor = std::make_shared<Data>(
                    DataType::FLOAT32,
                    std::vector<int>({completeKeyBlocks,
                                      this->indexerHeadDim}),
                    blockKeys);
                cacheTensor->ToDevice(
                    query.dataDevice, query.dataDeviceIds);
                if (completeKeyBlocks > 0) {
                    const int capacity =
                        ((completeKeyBlocks + 255) / 256) * 256;
                    if (capacity > completeKeyBlocks) {
                        cacheTensor->Expansion(
                            {capacity, this->indexerHeadDim});
                    }
                }
            } else if (cacheTensor->dims[0] < completeKeyBlocks) {
                const int tensorBlocks = cacheTensor->dims[0];
                std::vector<float> deltaValues(
                    blockKeys.begin() +
                        (size_t)tensorBlocks *
                            this->indexerHeadDim,
                    blockKeys.end());
                Data delta(
                    DataType::FLOAT32,
                    {completeKeyBlocks - tensorBlocks,
                     this->indexerHeadDim},
                    deltaValues);
                delta.ToDevice(
                    query.dataDevice, query.dataDeviceIds);
                if (cacheTensor->expansionDims.empty() ||
                    cacheTensor->expansionDims[0] <
                        completeKeyBlocks) {
                    const int capacity =
                        ((completeKeyBlocks + 255) / 256) * 256;
                    cacheTensor->Expansion(
                        {capacity, this->indexerHeadDim});
                }
                CatDirect(*cacheTensor, delta, 0);
            }
            const int sparsePrefillMinContext = std::max(
                this->indexerBudget,
                kQwen4SparsePrefillMinContext);
            const bool useSparseAttention = sequence == 1 ||
                (sequence > 1 &&
                 keyLength > sparsePrefillMinContext);
            Qwen4QSASelect(
                query, *cacheTensor, keyLength,
                this->indexerHeads, this->indexerHeadDim,
                this->indexerBudget,
                this->indexerCompressRatio, qsaIndices,
                sequence > 1 ? previousLength : -1);
            if (useSparseAttention) {
                qsaMask = Data();
            } else {
                Qwen4QSABuildMask(
                    qsaIndices, input, keyLength, qsaMask);
                qsaIndices = Data();
            }
            return;
        }

        Data maskCpu;
        const float *baseMaskValues = nullptr;
        if (!baseMask.dims.empty()) {
            ToDataType(baseMask, maskCpu, DataType::FLOAT32);
            maskCpu.ToDevice(DataDevice::CPU);
            baseMaskValues = reinterpret_cast<const float *>(maskCpu.cpuData);
            AssertInFastLLM(baseMask.dims.back() == keyLength,
                            "Qwen4-Exp QSA mask length differs from its KV cache.");
        }

        Data queryCpu;
        ToDataType(query, queryCpu, DataType::FLOAT32);
        queryCpu.ToDevice(DataDevice::CPU);
        const float *queryValues =
            reinterpret_cast<const float *>(queryCpu.cpuData);

        auto isVisible = [&](int queryIndex, int keyIndex) {
            if (baseMaskValues == nullptr) {
                return keyIndex <= previousLength + queryIndex;
            }
            int maskOffset = keyIndex;
            if (maskCpu.Count(0) >= sequence * keyLength) {
                maskOffset += queryIndex * keyLength;
            }
            return std::fabs(baseMaskValues[maskOffset]) < 0.5f;
        };

        const int blockTopK = this->indexerBudget /
                              this->indexerCompressRatio;
        const float inverseSqrtIndex =
            1.0f / std::sqrt((float)this->indexerHeadDim);
        std::vector<float> maskValues((size_t)sequence * keyLength, 1.0f);

        // An ordinary single-request prefill has a contiguous causal mask.
        // The compressed block key is independent of the query, but the
        // original reference rebuilt it inside every query/block pair and ran
        // the whole QSA scorer serially.  Past the 2048-token budget that made
        // index construction quadratic on one CPU core and dominated the
        // complete forward.  Build every block key once, then score independent
        // query rows across FastLLM's CPU pool.  The arithmetic and TopK
        // tie-break remain identical to the reference path below.
        if (baseMaskValues == nullptr) {
            AssertInFastLLM(
                blockKeys.size() == (size_t)completeKeyBlocks *
                                        this->indexerHeadDim,
                "Qwen4-Exp incremental compressed cache is incomplete.");

            Qwen4ParallelFor(sequence, [&](int start, int end) {
                std::vector<std::pair<float, int>> blockScores;
                blockScores.reserve(completeKeyBlocks);
                for (int taskIndex = start; taskIndex < end; taskIndex++) {
                    // Pair early cheap rows with late expensive rows so every
                    // worker receives a similar number of scored blocks.
                    const int queryIndex = (taskIndex & 1) == 0
                        ? taskIndex / 2
                        : sequence - 1 - taskIndex / 2;
                    const int visibleTokens = previousLength + queryIndex + 1;
                    const int completeBlocks = visibleTokens /
                                               this->indexerCompressRatio;
                    float *maskRow = maskValues.data() +
                                     (size_t)queryIndex * keyLength;
                    if (completeBlocks <= blockTopK) {
                        std::fill(maskRow, maskRow + visibleTokens, 0.0f);
                        continue;
                    }

                    blockScores.clear();
                    for (int block = 0; block < completeBlocks; block++) {
                        const float *blockKey = blockKeys.data() +
                            (size_t)block * this->indexerHeadDim;
                        float score = 0.0f;
                        for (int head = 0; head < this->indexerHeads; head++) {
                            const float *headQuery = queryValues +
                                ((size_t)queryIndex * this->indexerHeads + head) *
                                    this->indexerHeadDim;
                            float headScore = 0.0f;
                            for (int column = 0;
                                 column < this->indexerHeadDim; column++) {
                                headScore += headQuery[column] * blockKey[column];
                            }
                            score += std::max(headScore, 0.0f);
                        }
                        blockScores.push_back(
                            {score * inverseSqrtIndex, block});
                    }

                    std::partial_sort(
                        blockScores.begin(),
                        blockScores.begin() + blockTopK,
                        blockScores.end(),
                        [](const std::pair<float, int> &left,
                           const std::pair<float, int> &right) {
                            if (left.first != right.first) {
                                return left.first > right.first;
                            }
                            return left.second < right.second;
                        });
                    for (int rank = 0; rank < blockTopK; rank++) {
                        const int tokenStart =
                            blockScores[rank].second *
                            this->indexerCompressRatio;
                        std::fill(maskRow + tokenStart,
                                  maskRow + tokenStart +
                                      this->indexerCompressRatio,
                                  0.0f);
                    }
                    std::fill(maskRow +
                                  completeBlocks * this->indexerCompressRatio,
                              maskRow + visibleTokens, 0.0f);
                }
            });

            Data maskFloat(DataType::FLOAT32,
                           {batch, sequence, keyLength}, maskValues);
            ToDataType(maskFloat, qsaMask, input.dataType);
            return;
        }

        std::vector<int> visibleIndices;
        std::vector<float> pooled(this->indexerHeadDim);
        std::vector<float> rotated(this->indexerHeadDim);

        for (int queryIndex = 0; queryIndex < sequence; queryIndex++) {
            visibleIndices.clear();
            for (int keyIndex = 0; keyIndex < keyLength; keyIndex++) {
                if (isVisible(queryIndex, keyIndex)) {
                    visibleIndices.push_back(keyIndex);
                }
            }

            const int completeBlocks = (int)visibleIndices.size() /
                                       this->indexerCompressRatio;
            std::vector<std::pair<float, int>> blockScores;
            blockScores.reserve(completeBlocks);
            for (int block = 0; block < completeBlocks; block++) {
                std::fill(pooled.begin(), pooled.end(), 0.0f);
                for (int member = 0; member < this->indexerCompressRatio;
                     member++) {
                    const int token = visibleIndices[
                        block * this->indexerCompressRatio + member];
                    const float *source = rawKeyCache.data() +
                        (size_t)token * this->indexerHeadDim;
                    for (int column = 0; column < this->indexerHeadDim;
                         column++) {
                        pooled[column] += source[column];
                    }
                }
                float squareSum = 0.0f;
                for (float value : pooled) {
                    const float averaged = value /
                        (float)this->indexerCompressRatio;
                    squareSum += averaged * averaged;
                }
                const float inverseRms = 1.0f / std::sqrt(
                    squareSum / this->indexerHeadDim + this->rms_norm_eps);
                for (int column = 0; column < this->indexerHeadDim;
                     column++) {
                    rotated[column] = pooled[column] /
                        (float)this->indexerCompressRatio * inverseRms *
                        keyNorm[column];
                }

                const int groupStart = visibleIndices[
                    block * this->indexerCompressRatio];
                const int position = (int)(positionCache[groupStart] + 0.01f);
                AssertInFastLLM(position >= 0 &&
                                position < this->sinData.dims[0],
                                "Qwen4-Exp QSA position exceeds its rotary table.");
                const float *sinRow = sinValues +
                    (size_t)position * rotaryStride;
                const float *cosRow = cosValues +
                    (size_t)position * rotaryStride;
                std::copy(rotated.begin(), rotated.end(), pooled.begin());
                for (int column = 0; column < rotaryHalf; column++) {
                    rotated[column] = pooled[column] * cosRow[column] -
                        pooled[column + rotaryHalf] * sinRow[column];
                    rotated[column + rotaryHalf] =
                        pooled[column + rotaryHalf] *
                            cosRow[column + rotaryHalf] +
                        pooled[column] * sinRow[column + rotaryHalf];
                }

                float score = 0.0f;
                for (int head = 0; head < this->indexerHeads; head++) {
                    const float *headQuery = queryValues +
                        ((size_t)queryIndex * this->indexerHeads + head) *
                            this->indexerHeadDim;
                    float headScore = 0.0f;
                    for (int column = 0; column < this->indexerHeadDim;
                         column++) {
                        headScore += headQuery[column] * rotated[column];
                    }
                    score += std::max(headScore, 0.0f);
                }
                blockScores.push_back({score * inverseSqrtIndex, block});
            }

            const int selectedBlockCount = std::min(blockTopK, completeBlocks);
            std::partial_sort(
                blockScores.begin(),
                blockScores.begin() + selectedBlockCount,
                blockScores.end(),
                [](const std::pair<float, int> &left,
                   const std::pair<float, int> &right) {
                    if (left.first != right.first) {
                        return left.first > right.first;
                    }
                    return left.second < right.second;
                });
            for (int rank = 0; rank < selectedBlockCount; rank++) {
                const int block = blockScores[rank].second;
                for (int member = 0; member < this->indexerCompressRatio;
                     member++) {
                    const int token = visibleIndices[
                        block * this->indexerCompressRatio + member];
                    maskValues[(size_t)queryIndex * keyLength + token] = 0.0f;
                }
            }
            for (int index = completeBlocks * this->indexerCompressRatio;
                 index < (int)visibleIndices.size(); index++) {
                maskValues[(size_t)queryIndex * keyLength +
                           visibleIndices[index]] = 0.0f;
            }
        }

        Data maskFloat(DataType::FLOAT32,
                       {batch, sequence, keyLength}, maskValues);
        // CUDA Attention reads its mask in the Q/K/V activation type. Keep
        // the QSA fallback on that same contract; leaving this temporary
        // float32 buffer uncast makes the half kernel reinterpret each float
        // as two unrelated fp16 mask entries.
        ToDataType(maskFloat, qsaMask, input.dataType);
    }

    void Qwen4ExpModel::RunFullAttention(int layer,
                                          const Data &input,
                                          const Data &attentionMask,
                                          const Data &positionIds,
                                          bool qsaDeviceCompatibleMask,
                                          Data &pastKey,
                                          Data &pastValue,
                                          RequestState &state,
                                          Data &output) {
        const std::string attention = languagePrefix + "layers." +
            std::to_string(layer) + ".self_attn.";
        const int batch = input.dims[0];
        const int sequence = input.dims[1];
        const int previousLength = pastKey.dims.empty() ? 0 : pastKey.dims[1];
        Data typedInput;
        ToDataType(input, typedInput, this->dataType);
        Data qsaMask, qsaIndices;
        BuildQSAMask(layer, typedInput, attentionMask, positionIds,
                     previousLength, qsaDeviceCompatibleMask,
                     state, qsaMask, qsaIndices);

        Data qGate, query, key, value, gate;
        Linear(typedInput, this->weight[attention + "q_proj.weight"], Data(), qGate);
        qGate.Reshape({batch, sequence, -1, this->head_dim * 2});
        Split(qGate, -1, 0, this->head_dim, query);
        Split(qGate, -1, this->head_dim, this->head_dim * 2, gate);
        gate.Reshape({batch, sequence, -1});

        Linear(typedInput, this->weight[attention + "k_proj.weight"], Data(), key);
        Linear(typedInput, this->weight[attention + "v_proj.weight"], Data(), value);
        key.Reshape({batch, sequence, -1, this->head_dim});
        value.Reshape({batch, sequence, -1, this->head_dim});

        RMSNorm(query, this->weight[attention + "q_norm.weight"],
                this->rms_norm_eps, query);
        RMSNorm(key, this->weight[attention + "k_norm.weight"],
                this->rms_norm_eps, key);
        LlamaRotatePosition2DPart(query, positionIds, this->sinData, this->cosData,
                                  this->rotary_dim, this->rotary_dim);
        LlamaRotatePosition2DPart(key, positionIds, this->sinData, this->cosData,
                                  this->rotary_dim, this->rotary_dim);

        PermuteSelf(query, {0, 2, 1, 3});
        PermuteSelf(key, {0, 2, 1, 3});
        PermuteSelf(value, {0, 2, 1, 3});
        query.Reshape({-1, sequence, this->head_dim});
        key.Reshape({-1, sequence, this->head_dim});
        value.Reshape({-1, sequence, this->head_dim});

        if (GetKVCacheInCPU()) {
            pastKey.lockInCPU = true;
            pastValue.lockInCPU = true;
        }
        if (pastKey.dims.empty() && pastKey.dataType != key.dataType) {
            pastKey.dataType = key.dataType;
            pastKey.UpdateUnitSize();
        }
        if (pastValue.dims.empty() && pastValue.dataType != value.dataType) {
            pastValue.dataType = value.dataType;
            pastValue.UpdateUnitSize();
        }
        const int unitLength = !GetKVCacheInCPU() &&
            key.dataDevice == DataDevice::CUDA ? 128 : 64;
        const bool geometricGrowth =
            state.geometricCacheGrowthReadyLayers.count(layer) != 0;
        Qwen4EnsureAppendCapacity(
            pastKey, key, 1, unitLength,
            kQwen4DenseCacheMaxGrowth, geometricGrowth);
        Qwen4EnsureAppendCapacity(
            pastValue, value, 1, unitLength,
            kQwen4DenseCacheMaxGrowth, geometricGrowth);
        CatDirect(pastKey, key, 1);
        CatDirect(pastValue, value, 1);
        if (sequence == 1) {
            state.geometricCacheGrowthReadyLayers.insert(layer);
        }

        Data context;
        const int attentionGroup = query.dims[0] / pastKey.dims[0];
        const float attentionScale =
            1.0f / std::sqrt((float)this->head_dim);
        if (!qsaIndices.dims.empty()) {
            Qwen4SparseAttention(query, pastKey, pastValue,
                                 qsaIndices, attentionGroup,
                                 attentionScale, context);
        } else {
            Attention(query, pastKey, pastValue, qsaMask, context,
                      attentionGroup, attentionScale, 1);
        }
        PermuteSelf(context, {1, 0, 2});
        context.Reshape({sequence, batch, -1});
        PermuteSelf(context, {1, 0, 2});

        Sigmoid(gate, gate);
        Qwen4CastLike(gate, context);
        MulTo(context, gate);
        Linear(context, this->weight[attention + "o_proj.weight"], Data(), output);
    }

    void Qwen4ExpModel::RunLinearAttention(int layer,
                                            const Data &input,
                                            Data &pastConv,
                                            Data &pastRecurrent,
                                            Data &output) {
        const std::string linear = languagePrefix + "layers." +
            std::to_string(layer) + ".linear_attn.";
        const int batch = input.dims[0];
        const int sequence = input.dims[1];

        Data typedInput;
        ToDataType(input, typedInput, this->dataType);
        Data qkv, z, beta, alpha;
        Linear(typedInput, this->weight[linear + "in_proj_qkv.weight"], Data(), qkv);
        Linear(typedInput, this->weight[linear + "in_proj_z.weight"], Data(), z);
        Linear(typedInput, this->weight[linear + "in_proj_b.weight"], Data(), beta);
        Linear(typedInput, this->weight[linear + "in_proj_a.weight"], Data(), alpha);
        const bool fusedDecode = sequence == 1;
        const bool mixedGdnPrefill =
            !fusedDecode && GetFastllmEnv().cudaTriton &&
            qkv.dataDevice == DataDevice::CUDA &&
            qkv.dataType == DataType::FLOAT16;
        if (!fusedDecode && !mixedGdnPrefill) {
            ToDataType(qkv, DataType::FLOAT32);
        }

        qkv.Reshape({batch, sequence, -1});
        pastConv.isKVCache = true;
        pastConv.isLinearAttention = true;
        pastRecurrent.isKVCache = true;
        pastRecurrent.isLinearAttention = true;
        if (pastConv.dims.empty() && pastConv.dataType != DataType::FLOAT32) {
            pastConv.dataType = DataType::FLOAT32;
            pastConv.UpdateUnitSize();
        }
        if (pastRecurrent.dims.empty() &&
            pastRecurrent.dataType != DataType::FLOAT32) {
            pastRecurrent.dataType = DataType::FLOAT32;
            pastRecurrent.UpdateUnitSize();
        }
        if (GetKVCacheInCPU()) {
            pastConv.lockInCPU = true;
            pastRecurrent.lockInCPU = true;
        }

        Data currentConvolved;
        if (fusedDecode) {
            PermuteSelf(qkv, {0, 2, 1});
            fastllm::CausalDepthwiseConv1DDecode(
                qkv, this->weight[linear + "conv1d.weight"],
                pastConv, this->linearConvKernel, true, currentConvolved);
            pastConv.expansionDims = pastConv.dims;
        } else {
            // Keep Q/K/V token-major during prefill.  The standard fused op is
            // mathematically identical to the old permute + Cat + Conv1D +
            // Split + SiLU chain and updates the four-value history in place.
            fastllm::CausalDepthwiseConv1DPrefill(
                qkv, this->weight[linear + "conv1d.weight"],
                pastConv, this->linearConvKernel, true, currentConvolved);
            pastConv.expansionDims = pastConv.dims;
            if (mixedGdnPrefill) {
                // The convolution accumulates into float32 alongside its
                // float32 history.  Round only its token activations before
                // the standard mixed GDN, matching Qwen3.5/vLLM-style
                // activation precision without lowering cache precision.
                ToDataType(currentConvolved, DataType::FLOAT16);
            }
        }

        if (pastRecurrent.dims.empty()) {
            pastRecurrent.Resize({batch, this->num_v_heads,
                                  this->head_k_dim, this->head_v_dim});
            pastRecurrent.ToDevice(currentConvolved.dataDevice);
            pastRecurrent.Allocate(0.0f);
        }

        // Decode is dominated by the many small Q/K normalization, head
        // repeat, recurrent-state and output-gate launches.  Keep the state in
        // float32 as required by the checkpoint.  Fuse the state transition,
        // then reuse the standard output RMSNorm/gate operations so their
        // established fp16 rounding semantics remain bit-identical.
        if (sequence == 1) {
            Data core;
            fastllm::GatedDeltaRuleDecode(
                currentConvolved, alpha, beta,
                this->weight[linear + "A_log"],
                this->weight[linear + "dt_bias"],
                pastRecurrent, this->num_k_heads, this->num_v_heads,
                this->head_k_dim, this->head_v_dim,
                1e-6f, core);
            ToDataType(core, this->dataType);
            core.Reshape({-1, this->head_v_dim});
            z.Reshape({-1, this->head_v_dim});
            RMSNorm(core, this->weight[linear + "norm.weight"],
                    this->rms_norm_eps, core);
            Sigmoid(z, z);
            Qwen4CastLike(z, core);
            MulTo(core, z);
            core.Reshape({batch, sequence,
                          this->num_v_heads * this->head_v_dim});
            Linear(core,
                   this->weight[linear + "out_proj.weight"],
                   Data(), output);
            return;
        }

        if (mixedGdnPrefill) {
            ToDataType(beta, DataType::FLOAT16);
            ToDataType(alpha, DataType::FLOAT16);
        } else {
            ToDataType(beta, DataType::FLOAT32);
            ToDataType(alpha, DataType::FLOAT32);
        }

        const int keyDimension = this->num_k_heads * this->head_k_dim;
        const int valueDimension = this->num_v_heads * this->head_v_dim;
        const float inverseHead = 1.0f / std::sqrt((float)this->head_k_dim);
        Sigmoid(beta, beta);
        Data decay;
        MambaSoftplus(alpha, this->weight[linear + "A_log"],
                      this->weight[linear + "dt_bias"], decay);

        // Qwen3.5 and Qwen4 share this standard chunk-GDN operation.  CUDA
        // prefill keeps the projected activations in float16 and uses the
        // operation's layer-wide mixed path, while the checkpoint-required
        // recurrent state remains float32.  CPU and unsupported activation
        // types retain the float32 reference path.
        constexpr int chunkSize = 64;
        const int padding = (chunkSize - sequence % chunkSize) % chunkSize;
        Data query, key, value;
        Data normalizedQuery, normalizedKey;
        Data scaledQuery, paddedKey, paddedValue, paddedBeta, paddedDecay;
        Data keyBeta, valueBeta;
        Data *chunkKey = nullptr;
        Data *chunkDecay = nullptr;
        bool fusedPostConv = false;

        if (this->linearInvScaleData.dataDevice !=
                currentConvolved.dataDevice ||
            this->linearInvScaleData.dataDeviceIds !=
                currentConvolved.dataDeviceIds) {
            this->linearInvScaleData.ToDevice(
                currentConvolved.dataDevice,
                currentConvolved.dataDeviceIds);
        }
#ifdef USE_CUDA
        if (mixedGdnPrefill) {
            fusedPostConv = FastllmCudaTryTritonChunkGdnPostConv(
                currentConvolved, this->linearInvScaleData,
                decay, beta,
                batch, sequence, this->num_k_heads, this->num_v_heads,
                this->head_k_dim, this->head_v_dim,
                1e-6f, inverseHead,
                normalizedQuery, normalizedKey,
                scaledQuery, paddedKey, paddedValue,
                paddedDecay, paddedBeta, keyBeta, valueBeta);
        }
#endif
        if (fusedPostConv) {
            chunkKey = &paddedKey;
            chunkDecay = &paddedDecay;
        } else {
            Split(currentConvolved, -1, 0, keyDimension, query);
            Split(currentConvolved, -1, keyDimension, 2 * keyDimension, key);
            Split(currentConvolved, -1, 2 * keyDimension,
                  2 * keyDimension + valueDimension, value);
            query.Reshape(
                {batch, sequence, this->num_k_heads, this->head_k_dim});
            key.Reshape(
                {batch, sequence, this->num_k_heads, this->head_k_dim});
            value.Reshape(
                {batch, sequence, this->num_v_heads, this->head_v_dim});

            const int repeat = this->num_v_heads / this->num_k_heads;
            if (repeat > 1) {
                Data expandedQuery, expandedKey;
                query.Reshape({batch, sequence, this->num_k_heads, 1,
                               this->head_k_dim});
                key.Reshape({batch, sequence, this->num_k_heads, 1,
                             this->head_k_dim});
                Repeat(query, 3, repeat, expandedQuery);
                Repeat(key, 3, repeat, expandedKey);
                expandedQuery.Reshape(
                    {batch, sequence, this->num_v_heads, this->head_k_dim});
                expandedKey.Reshape(
                    {batch, sequence, this->num_v_heads, this->head_k_dim});
                query.CopyFrom(expandedQuery);
                key.CopyFrom(expandedKey);
            }

            RMSNorm(query, this->linearInvScaleData, 1e-6f, query);
            RMSNorm(key, this->linearInvScaleData, 1e-6f, key);
            if (!mixedGdnPrefill) {
                ToDataType(query, DataType::FLOAT32);
                ToDataType(key, DataType::FLOAT32);
                ToDataType(value, DataType::FLOAT32);
                ToDataType(beta, DataType::FLOAT32);
                ToDataType(decay, DataType::FLOAT32);
            }

            PermuteSelf(query, {0, 2, 1, 3});
            PermuteSelf(key, {0, 2, 1, 3});
            PermuteSelf(value, {0, 2, 1, 3});
            PermuteSelf(beta, {0, 2, 1});
            PermuteSelf(decay, {0, 2, 1});

            Data *chunkBeta = nullptr;
            Data *chunkValue = nullptr;
            if (padding > 0) {
                Data paddedQuery;
                Pad(query, 2, padding, paddedQuery);
                Pad(key, 2, padding, paddedKey);
                Pad(value, 2, padding, paddedValue);
                Pad(beta, 2, padding, paddedBeta);
                Pad(decay, 2, padding, paddedDecay);
                Mul(paddedQuery, inverseHead, scaledQuery);
                chunkKey = &paddedKey;
                chunkValue = &paddedValue;
                chunkBeta = &paddedBeta;
                chunkDecay = &paddedDecay;
            } else {
                Mul(query, inverseHead, scaledQuery);
                chunkKey = &key;
                chunkValue = &value;
                chunkBeta = &beta;
                chunkDecay = &decay;
            }

            chunkBeta->Resize({chunkBeta->dims[0], chunkBeta->dims[1],
                               chunkBeta->dims[2], 1});
            Mul(*chunkKey, 1.0f, keyBeta);
            Mul(*chunkValue, 1.0f, valueBeta);
            MulTo(keyBeta, *chunkBeta);
            MulTo(valueBeta, *chunkBeta);
        }

        scaledQuery.Reshape({scaledQuery.dims[0], scaledQuery.dims[1], -1,
                             chunkSize, scaledQuery.dims.back()});
        chunkKey->Reshape({chunkKey->dims[0], chunkKey->dims[1], -1,
                           chunkSize, chunkKey->dims.back()});
        keyBeta.Reshape({keyBeta.dims[0], keyBeta.dims[1], -1,
                         chunkSize, keyBeta.dims.back()});
        valueBeta.Reshape({valueBeta.dims[0], valueBeta.dims[1], -1,
                           chunkSize, valueBeta.dims.back()});
        chunkDecay->Reshape({chunkDecay->dims[0], chunkDecay->dims[1], -1,
                             chunkSize});

        CumSumLastDim(*chunkDecay);
        Data decayMask;
        MakeDecayMask(*chunkDecay, decayMask);

        Data attention, negativeKeyAttention;
        MatMulTransB(keyBeta, *chunkKey, negativeKeyAttention);
        Mul(negativeKeyAttention, -1.0f, attention);
        MulTo(attention, decayMask);
        CausalMask(attention, 0, 0.0f);
        TransferAttn(attention);

        Data chunkValueOutput;
        MatMul(attention, valueBeta, chunkValueOutput);
        Data decayExp, keyCumDecay;
        Exp(*chunkDecay, decayExp);
        MulTo(keyBeta, decayExp);
        MatMul(attention, keyBeta, keyCumDecay);

        MatMulTransB(scaledQuery, *chunkKey, attention);
        MulTo(attention, decayMask);
        CausalMask(attention, 1, 0.0f);

        Data allCore;
        ChunkGatedDeltaRulePrefill(scaledQuery, *chunkKey, chunkValueOutput,
                                   *chunkDecay, attention, keyCumDecay,
                                   pastRecurrent, allCore);
        allCore.Reshape({allCore.dims[0], allCore.dims[1], -1,
                         allCore.dims.back()});
        if (padding > 0) {
            Data unpaddedCore;
            Split(allCore, 2, 0, sequence, unpaddedCore);
            PermuteSelf(unpaddedCore, {0, 2, 1, 3});
            allCore.CopyFrom(unpaddedCore);
        } else {
            PermuteSelf(allCore, {0, 2, 1, 3});
        }

        z.Reshape({batch, sequence, this->num_v_heads, this->head_v_dim});
        ToDataType(allCore, this->dataType);
        allCore.Reshape({-1, this->head_v_dim});
        z.Reshape({-1, this->head_v_dim});
        RMSNorm(allCore, this->weight[linear + "norm.weight"],
                this->rms_norm_eps, allCore);
        // Qwen4-Exp's released checkpoint explicitly chooses a sigmoid output
        // gate; Qwen3-Next's default SiLU here would noticeably change logits.
        Sigmoid(z, z);
        Qwen4CastLike(z, allCore);
        MulTo(allCore, z);
        allCore.Reshape({batch, sequence, valueDimension});
        Linear(allCore, this->weight[linear + "out_proj.weight"], Data(), output);
    }

    void Qwen4ExpModel::RunMoE(int layer, const Data &input, Data &output) {
        const std::string mlp = languagePrefix + "layers." +
                                std::to_string(layer) + ".mlp.";
        const int batch = input.dims[0];
        const int sequence = input.dims[1];
        Data flattened;
        flattened.FakeFrom(input, 0);
        flattened.Resize(input.dims);
        flattened.Reshape({batch * sequence, input.dims.back()});

        Data routerLogits, sharedGateUp, sharedHidden, sharedOutput, sharedGate;
        Linear(flattened, this->weight[mlp + "gate.weight"], Data(), routerLogits);
        // Expert selection is defined in float32 (and the generic
        // SelectExpert contract requires it).  Keep only the narrow router
        // tensor in float32 while the much larger residual/MLP activations
        // remain in the configured activation dtype.
        ToDataType(routerLogits, DataType::FLOAT32);
        Softmax(routerLogits, routerLogits, -1);
        Linear(flattened, this->weight[mlp + "shared_expert.gateup_proj.weight"],
               Data(), sharedGateUp);
        Swiglu(sharedGateUp, sharedHidden);
        Linear(sharedHidden, this->weight[mlp + "shared_expert.down_proj.weight"],
               Data(), sharedOutput);
        Linear(flattened, this->weight[mlp + "shared_expert_gate.weight"],
               Data(), sharedGate);
        Sigmoid(sharedGate, sharedGate);
        Qwen4CastLike(sharedGate, sharedOutput);
        MulTo(sharedOutput, sharedGate);

        Data expertIndex, expertScore;
        SelectExpert(routerLogits, expertIndex, expertScore,
                     this->num_experts_per_tok, this->norm_topk_prob,
                     this->routed_scaling_factor, nullptr);
        this->ApplyMoeDeviceMapForLayer(layer);

        Data w1, w2, w3, temporaryInput, temporaryOutput, routed;
        MergeMOE(flattened, expertIndex, expertScore,
                 this->weights[layer], this->biass[layer],
                 w1, w2, w3, temporaryInput, temporaryOutput,
                 1.0f, routed, layer);
        routed.Reshape(input.dims);
        sharedOutput.Reshape(input.dims);

        Data routedCopy;
        routedCopy.CopyFrom(routed);
        ApplyDeviceMap(this->deviceMap, layer + 1, this->block_cnt);
        output.CopyFrom(routedCopy);
        Qwen4CastLike(sharedOutput, output);
        AddTo(output, sharedOutput);
    }

    bool Qwen4ExpModel::TryRunDecodeCudaGraphBackbone(
            int firstFullAttentionLayer,
            const Data &hiddenStates,
            const Data &attentionOutput,
            const Data &attentionInjection,
            const Data &attentionMask,
            const Data &positionIds,
            bool qsaDeviceCompatibleMask,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            RequestState &requestState,
            Data &logits) {
#ifndef USE_CUDA
        return false;
#else
        if (!GetFastllmEnv().cudaGraph ||
            hiddenStates.dims.size() != 3 || hiddenStates.dims[0] != 1 ||
            hiddenStates.dims[1] != 1 ||
            hiddenStates.dataDevice != DataDevice::CUDA ||
            hiddenStates.cudaData == nullptr ||
            attentionOutput.dataDevice != DataDevice::CUDA ||
            attentionOutput.cudaData == nullptr ||
            attentionInjection.dataDevice != DataDevice::CUDA ||
            attentionInjection.cudaData == nullptr ||
            GetKVCacheInCPU() || firstFullAttentionLayer != this->kvCacheId ||
            this->pleLayer >= firstFullAttentionLayer ||
            std::getenv("FASTLLM_QWEN4_DUMP_DIR") != nullptr ||
            !Qwen4CudaOnlyDeviceMap(this->deviceMap) ||
            !Qwen4CudaOnlyDeviceMap(this->moeDeviceMap) ||
            !Qwen4CudaOnlyDeviceMap(this->layeredMoeDeviceMap)) {
            return false;
        }

        int device = hiddenStates.dataDeviceIds.empty()
            ? FastllmCudaGetDevice() : hiddenStates.dataDeviceIds[0];
        if (device < 0 || (int)pastKeyValues.size() < this->block_cnt) {
            return false;
        }

        std::vector<void*> linearCachePointers;
        for (int layer = firstFullAttentionLayer + 1;
             layer < this->block_cnt; layer++) {
            if (!this->IsLinearAttentionLayer(layer)) {
                continue;
            }
            Data &pastConv = pastKeyValues[layer].first;
            Data &pastRecurrent = pastKeyValues[layer].second;
            if (pastConv.dims.empty() || pastRecurrent.dims.empty() ||
                pastConv.dataDevice != DataDevice::CUDA ||
                pastRecurrent.dataDevice != DataDevice::CUDA ||
                pastConv.cudaData == nullptr ||
                pastRecurrent.cudaData == nullptr ||
                (!pastConv.dataDeviceIds.empty() &&
                 pastConv.dataDeviceIds[0] != device) ||
                (!pastRecurrent.dataDeviceIds.empty() &&
                 pastRecurrent.dataDeviceIds[0] != device)) {
                return false;
            }
            linearCachePointers.push_back(pastConv.cudaData);
            linearCachePointers.push_back(pastRecurrent.cudaData);
        }

        const Data *requestKey = &pastKeyValues[0].first;
        DecodeCudaGraphState *graphState = nullptr;
        {
            std::lock_guard<std::mutex> guard(this->stateMutex);
            std::unique_ptr<DecodeCudaGraphState> &entry =
                this->decodeCudaGraphStates[requestKey];
            if (entry == nullptr) {
                entry.reset(new DecodeCudaGraphState());
            }
            graphState = entry.get();
        }
        std::lock_guard<std::mutex> graphGuard(graphState->mutex);

        const bool workspaceShapeChanged =
            graphState->hiddenStates[0].cudaData != nullptr &&
            (graphState->hiddenStates[0].dataType != hiddenStates.dataType ||
             graphState->hiddenStates[0].dims != hiddenStates.dims);
        if (graphState->device != device || workspaceShapeChanged ||
            (!graphState->linearCachePointers.empty() &&
             graphState->linearCachePointers != linearCachePointers)) {
            graphState->Reset(device);
        }
        graphState->device = device;
        graphState->linearCachePointers = linearCachePointers;
        if (!Qwen4PrepareDecodeGraphTensor(
                graphState->hiddenStates[0], hiddenStates, device) ||
            !Qwen4PrepareDecodeGraphTensor(
                graphState->attentionOutput, attentionOutput, device) ||
            !Qwen4PrepareDecodeGraphTensor(
                graphState->attentionInjection,
                attentionInjection, device)) {
            return false;
        }

        auto destroySegment = [&](DecodeCudaGraphState::Segment &segment) {
            FastllmCudaSetDevice(device);
            if (segment.exec != nullptr) {
                FastllmCudaGraphExecDestroy(segment.exec);
                segment.exec = nullptr;
            }
            if (segment.graph != nullptr) {
                FastllmCudaGraphDestroy(segment.graph);
                segment.graph = nullptr;
            }
            if (!segment.reservedPointers.empty()) {
                FastllmCudaGraphMemoryPoolRelease(
                    segment.reservedPointers);
                segment.reservedPointers.clear();
            }
            segment.captured = false;
        };

        auto runSegmentBody = [&](int startLayer, int nextFullLayer) {
            Data carriedAttentionNorm;
            for (int layer = startLayer; layer < nextFullLayer; layer++) {
                ApplyDeviceMap(this->deviceMap, layer + 1,
                               this->block_cnt);
                const std::string layerPrefix = languagePrefix + "layers." +
                                                std::to_string(layer) + ".";

                Data linearAttentionInput;
                Data linearAttentionInjection;
                Data linearAttentionOutput;
                const Data *currentAttentionOutput = nullptr;
                const Data *currentAttentionInjection = nullptr;
                if (layer == startLayer) {
                    currentAttentionOutput = &graphState->attentionOutput;
                    currentAttentionInjection =
                        &graphState->attentionInjection;
                } else {
                    const std::string attentionHyperPrefix =
                        layerPrefix + "attn_hyper_connection.";
                    HyperMixNormalized(
                        carriedAttentionNorm, attentionHyperPrefix,
                        linearAttentionInput,
                        &linearAttentionInjection);
                    carriedAttentionNorm.FreeSpace();
                    RunLinearAttention(
                        layer, linearAttentionInput,
                        pastKeyValues[layer].first,
                        pastKeyValues[layer].second,
                        linearAttentionOutput);
                    currentAttentionOutput = &linearAttentionOutput;
                    currentAttentionInjection =
                        &linearAttentionInjection;
                }

                const std::string mlpHyperPrefix =
                    layerPrefix + "mlp_hyper_connection.";
                Data mlpNorm;
                HyperCombineRMSNorm(
                    graphState->hiddenStates[0],
                    *currentAttentionOutput,
                    *currentAttentionInjection,
                    this->weight[mlpHyperPrefix + "hc_norm.weight"],
                    graphState->hiddenStates[1], mlpNorm);

                Data mlpInput, mlpInjection, mlpOutput;
                HyperMixNormalized(mlpNorm, mlpHyperPrefix,
                                   mlpInput, &mlpInjection);
                mlpNorm.FreeSpace();
                RunMoE(layer, mlpInput, mlpOutput);

                if (layer + 1 == this->block_cnt) {
                    const std::string finalHyperPrefix =
                        languagePrefix + "hyper_connection_mixer.";
                    Data finalHyperNorm;
                    HyperCombineRMSNorm(
                        graphState->hiddenStates[1], mlpOutput,
                        mlpInjection,
                        this->weight[finalHyperPrefix +
                                     "hc_norm.weight"],
                        graphState->hiddenStates[0], finalHyperNorm);
                    Data finalHidden;
                    HyperMixNormalized(finalHyperNorm,
                                       finalHyperPrefix,
                                       finalHidden, nullptr);
                    finalHyperNorm.FreeSpace();
                    Linear(finalHidden, this->weight["lm_head.weight"],
                           Data(), graphState->logits);
                    ToDataType(graphState->logits,
                               DataType::FLOAT32);
                } else {
                    const std::string nextAttentionHyperPrefix =
                        languagePrefix + "layers." +
                        std::to_string(layer + 1) +
                        ".attn_hyper_connection.";
                    HyperCombineRMSNorm(
                        graphState->hiddenStates[1], mlpOutput,
                        mlpInjection,
                        this->weight[nextAttentionHyperPrefix +
                                     "hc_norm.weight"],
                        graphState->hiddenStates[0],
                        carriedAttentionNorm);
                }
            }

            if (nextFullLayer < this->block_cnt) {
                ApplyDeviceMap(this->deviceMap,
                               nextFullLayer + 1,
                               this->block_cnt);
                const std::string attentionHyperPrefix =
                    languagePrefix + "layers." +
                    std::to_string(nextFullLayer) +
                    ".attn_hyper_connection.";
                HyperMixNormalized(
                    carriedAttentionNorm, attentionHyperPrefix,
                    graphState->attentionInput,
                    &graphState->attentionInjection);
                carriedAttentionNorm.FreeSpace();
            }
        };

        bool captureAttempted = false;
        auto runSegment = [&](int startLayer, int nextFullLayer) {
            DecodeCudaGraphState::Segment &segment =
                graphState->segments[startLayer];
            FastllmCudaSetDevice(device);

            if (segment.captured) {
                if (FastllmCudaGraphLaunch(segment.exec)) {
                    return;
                }
                std::fprintf(
                    stderr,
                    "[Fastllm] Qwen4 decode CUDA graph replay failed "
                    "on GPU %d segment %d: %s; using eager fallback.\n",
                    device, startLayer,
                    FastllmCudaGraphLastError());
                destroySegment(segment);
                segment.disabled = true;
                FastllmCudaClearThreadError();
                FastllmCudaClearGraphError();
                runSegmentBody(startLayer, nextFullLayer);
                return;
            }

            FastllmCudaMergeMOEClearGraphUnsafeFallbackFlag();
            if (!segment.warmed || segment.disabled) {
                runSegmentBody(startLayer, nextFullLayer);
                segment.warmed = true;
                if (FastllmCudaMergeMOEUsedGraphUnsafeFallback()) {
                    segment.disabled = true;
                }
                return;
            }
            if (captureAttempted) {
                runSegmentBody(startLayer, nextFullLayer);
                return;
            }
            captureAttempted = true;

            void *capturedGraph = nullptr;
            void *capturedExec = nullptr;
            bool poolActive = false;
            bool captureActive = false;
            bool captureOk = FastllmCudaGraphPrepareCaptureDevice();
            if (captureOk) {
                poolActive = FastllmCudaGraphMemoryPoolBegin();
                captureOk = poolActive;
            }
            if (captureOk) {
                FastllmCudaClearThreadError();
                FastllmCudaClearGraphError();
                captureActive = FastllmCudaGraphBeginCapture();
                captureOk = captureActive;
            }

            bool bodyThrew = false;
            if (captureOk) {
                FastllmCudaMergeMOEClearGraphUnsafeFallbackFlag();
                try {
                    runSegmentBody(startLayer, nextFullLayer);
                } catch (...) {
                    bodyThrew = true;
                }
                const bool unsafeMoe =
                    FastllmCudaMergeMOEUsedGraphUnsafeFallback();
                const bool bodyFailed = bodyThrew ||
                    unsafeMoe || FastllmCudaGetThreadError() ||
                    FastllmCudaGetGraphError() ||
                    FastllmCudaGraphCaptureInvalidated();
                bool endOk = FastllmCudaGraphEndCapture(
                    &capturedGraph);
                captureActive = false;
                captureOk = !bodyFailed && endOk &&
                            capturedGraph != nullptr;
                if (unsafeMoe) {
                    segment.disabled = true;
                }
            }
            if (captureActive) {
                void *discardedGraph = nullptr;
                FastllmCudaGraphEndCapture(&discardedGraph);
                if (discardedGraph != nullptr) {
                    FastllmCudaGraphDestroy(discardedGraph);
                }
            }

            if (captureOk) {
                captureOk = FastllmCudaGraphMemoryPoolEnd(
                    segment.reservedPointers);
                poolActive = false;
            }
            if (poolActive) {
                FastllmCudaGraphMemoryPoolAbort();
            }
            if (captureOk) {
                captureOk = FastllmCudaGraphInstantiate(
                    capturedGraph, &capturedExec) &&
                    capturedExec != nullptr;
            }

            if (!captureOk) {
                if (capturedExec != nullptr) {
                    FastllmCudaGraphExecDestroy(capturedExec);
                }
                if (capturedGraph != nullptr) {
                    FastllmCudaGraphDestroy(capturedGraph);
                }
                if (!segment.reservedPointers.empty()) {
                    FastllmCudaGraphMemoryPoolRelease(
                        segment.reservedPointers);
                    segment.reservedPointers.clear();
                }
                segment.captureFailures++;
                if (segment.captureFailures >= 3) {
                    segment.disabled = true;
                    std::fprintf(
                        stderr,
                        "[Fastllm] Qwen4 decode CUDA graph disabled "
                        "on GPU %d segment %d after capture failure: %s\n",
                        device, startLayer,
                        FastllmCudaGraphLastError());
                }
                FastllmCudaClearThreadError();
                FastllmCudaClearGraphError();
                FastllmCudaMergeMOEClearGraphUnsafeFallbackFlag();
                runSegmentBody(startLayer, nextFullLayer);
                if (FastllmCudaMergeMOEUsedGraphUnsafeFallback()) {
                    segment.disabled = true;
                }
                return;
            }

            segment.graph = capturedGraph;
            segment.exec = capturedExec;
            segment.captured = true;
            segment.captureFailures = 0;
            if (!FastllmCudaGraphLaunch(segment.exec)) {
                std::fprintf(
                    stderr,
                    "[Fastllm] Qwen4 decode CUDA graph first launch "
                    "failed on GPU %d segment %d: %s; using eager fallback.\n",
                    device, startLayer,
                    FastllmCudaGraphLastError());
                destroySegment(segment);
                segment.disabled = true;
                FastllmCudaClearThreadError();
                FastllmCudaClearGraphError();
                runSegmentBody(startLayer, nextFullLayer);
                return;
            }
        };

        int startLayer = firstFullAttentionLayer;
        while (startLayer < this->block_cnt) {
            int nextFullLayer = startLayer + 1;
            while (nextFullLayer < this->block_cnt &&
                   this->IsLinearAttentionLayer(nextFullLayer)) {
                nextFullLayer++;
            }
            runSegment(startLayer, nextFullLayer);
            if (nextFullLayer >= this->block_cnt) {
                break;
            }
            ApplyDeviceMap(this->deviceMap,
                           nextFullLayer + 1, this->block_cnt);
            RunFullAttention(
                nextFullLayer, graphState->attentionInput,
                attentionMask, positionIds,
                qsaDeviceCompatibleMask,
                pastKeyValues[nextFullLayer].first,
                pastKeyValues[nextFullLayer].second,
                requestState, graphState->attentionOutput);
            startLayer = nextFullLayer;
        }

        if (graphState->logits.dataDevice != DataDevice::CUDA ||
            graphState->logits.cudaData == nullptr ||
            graphState->logits.dims.empty()) {
            return false;
        }
        Qwen4BorrowCudaStorage(logits, graphState->logits);
        return true;
#endif
    }

    std::shared_ptr<Qwen4ExpModel::PrefixSnapshot>
    Qwen4ExpModel::FindPrefixSnapshotLocked(
            const std::vector<int> &tokens,
            int maxCachedLen,
            int exactLen) const {
        std::shared_ptr<PrefixSnapshot> best;
        for (const auto &snapshot : this->prefixSnapshots) {
            if (snapshot == nullptr || snapshot->cachedLen <= 0 ||
                snapshot->cachedLen > maxCachedLen ||
                snapshot->cachedLen > (int)tokens.size()) {
                continue;
            }
            if (exactLen >= 0 && snapshot->cachedLen != exactLen) {
                continue;
            }
            if ((int)snapshot->tokens.size() != snapshot->cachedLen ||
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

    void Qwen4ExpModel::MaybeRecordPrefixSnapshot(
            const std::vector<std::pair<Data, Data>> &pastKeyValues,
            RequestState &state) {
        if (!Qwen4PrefixCacheEnabled() ||
            (int)pastKeyValues.size() < this->block_cnt) {
            return;
        }

        int cachedLen = 0;
        for (int layer = 0; layer < this->block_cnt; layer++) {
            if (this->IsLinearAttentionLayer(layer)) {
                continue;
            }
            const Data &key = pastKeyValues[layer].first;
            const Data &value = pastKeyValues[layer].second;
            if (key.dims.size() < 2 || value.dims.size() < 2 ||
                key.dims[1] <= 0 || key.dims[1] != value.dims[1] ||
                (cachedLen > 0 && cachedLen != key.dims[1])) {
                return;
            }
            cachedLen = key.dims[1];
        }
        if (cachedLen <= 0 ||
            cachedLen != (int)state.processedTokens.size()) {
            return;
        }

        const int pageLen = std::max(1, fastllm::GetPageLen());
        const int intervalPages = std::max(
            1, Qwen4EnvInt("FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES", 16));
        const int interval = pageLen * intervalPages;
        if (cachedLen - state.lastPrefixSnapshotLen < interval) {
            return;
        }

        // Snapshot state is copied by value. Complete the asynchronous groups
        // and the small device tail once at this boundary so the snapshot has
        // a self-sufficient generic fallback history.
        for (int layer = 0; layer < this->block_cnt; layer++) {
            if (!this->IsLinearAttentionLayer(layer)) {
                MaterializeQsaHostHistory(layer, cachedLen, state);
            }
        }

        auto validQsaTensor = [](const std::shared_ptr<Data> &tensor,
                                 const std::vector<int> &dims) {
            return tensor != nullptr && tensor->dims == dims &&
                tensor->dataType == DataType::FLOAT32 &&
                !tensor->multiDeviceData &&
                ((tensor->dataDevice == DataDevice::CUDA &&
                  tensor->cudaData != nullptr) ||
                 (tensor->dataDevice == DataDevice::CPU &&
                  tensor->cpuData != nullptr));
        };

        uint64_t tensorBytes = 0;
        for (int layer = 0; layer < this->block_cnt; layer++) {
            const Data &first = pastKeyValues[layer].first;
            const Data &second = pastKeyValues[layer].second;
            if (first.dims.empty() || second.dims.empty() ||
                first.multiDeviceData || second.multiDeviceData) {
                return;
            }
            const bool linear = this->IsLinearAttentionLayer(layer);
            tensorBytes += linear
                ? first.GetBytes() + second.GetBytes()
                : Qwen4DenseLogicalBytes(first) +
                  Qwen4DenseLogicalBytes(second);
            if (linear) {
                continue;
            }
            const int expectedTail =
                cachedLen % this->indexerCompressRatio;
            const int expectedBlockCount =
                cachedLen / this->indexerCompressRatio;
            const auto raw = state.indexerRawKeys.find(layer);
            const auto positions = state.indexerPositions.find(layer);
            const size_t expectedRaw = (size_t)cachedLen *
                this->indexerKvHeads * this->indexerHeadDim;
            if (raw == state.indexerRawKeys.end() ||
                raw->second.size() != expectedRaw ||
                positions == state.indexerPositions.end() ||
                positions->second.size() != (size_t)cachedLen) {
                return;
            }
            const auto tailKeys =
                state.indexerTailKeyTensors.find(layer);
            const auto tailPositions =
                state.indexerTailPositionTensors.find(layer);
            const auto blockTensor =
                state.indexerBlockKeyTensors.find(layer);
            const bool deviceCacheValid =
                (expectedTail == 0 ||
                 (tailKeys != state.indexerTailKeyTensors.end() &&
                  validQsaTensor(
                      tailKeys->second,
                      {expectedTail, this->indexerHeadDim}))) &&
                (expectedTail == 0 ||
                 (tailPositions !=
                      state.indexerTailPositionTensors.end() &&
                  validQsaTensor(
                      tailPositions->second, {expectedTail}))) &&
                (expectedBlockCount == 0 ||
                 (blockTensor !=
                      state.indexerBlockKeyTensors.end() &&
                  validQsaTensor(
                      blockTensor->second,
                      {expectedBlockCount,
                       this->indexerHeadDim})));
            if (deviceCacheValid) {
                tensorBytes +=
                    ((uint64_t)expectedTail *
                         (this->indexerHeadDim + 1) +
                     (uint64_t)expectedBlockCount *
                         this->indexerHeadDim) * sizeof(float);
                continue;
            }

            const auto blocks = state.indexerBlockKeys.find(layer);
            const size_t expectedBlocks =
                (size_t)(cachedLen / this->indexerCompressRatio) *
                this->indexerHeadDim;
            if (raw == state.indexerRawKeys.end() ||
                raw->second.size() != expectedRaw ||
                positions == state.indexerPositions.end() ||
                positions->second.size() != (size_t)cachedLen ||
                blocks == state.indexerBlockKeys.end() ||
                blocks->second.size() != expectedBlocks) {
                return;
            }
        }

        uint64_t stateBytes =
            ((uint64_t)state.processedTokens.size() + cachedLen) * sizeof(int) +
            (uint64_t)state.convHistory.size() * sizeof(float);
        for (const auto &item : state.indexerRawKeys) {
            stateBytes += (uint64_t)item.second.size() * sizeof(float);
        }
        for (const auto &item : state.indexerPositions) {
            stateBytes += (uint64_t)item.second.size() * sizeof(float);
        }
        for (const auto &item : state.indexerBlockKeys) {
            stateBytes += (uint64_t)item.second.size() * sizeof(float);
        }
        const uint64_t snapshotBytes = tensorBytes + stateBytes;
        const uint64_t maxSnapshotBytes =
            (uint64_t)std::max(
                1, Qwen4EnvInt(
                    "FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_MB", 4096)) *
            1024ULL * 1024ULL;
        if (snapshotBytes > maxSnapshotBytes) {
            state.lastPrefixSnapshotLen = cachedLen;
            if (Qwen4PrefixCacheDebugEnabled()) {
                std::printf(
                    "[qwen4-prefix-cache] skip tokens=%d bytes=%.3f GiB "
                    "limit=%.3f GiB\n",
                    cachedLen, snapshotBytes / 1073741824.0,
                    maxSnapshotBytes / 1073741824.0);
                std::fflush(stdout);
            }
            return;
        }

        {
            std::lock_guard<std::mutex> guard(this->prefixCacheMutex);
            if (state.prefixRequestId <= 0) {
                state.prefixRequestId = ++this->prefixRequestCounter;
                if (state.prefixRequestId <= 0) {
                    state.prefixRequestId = 1;
                    this->prefixRequestCounter = 1;
                }
            }
            auto existing = FindPrefixSnapshotLocked(
                state.processedTokens, cachedLen, cachedLen);
            if (existing != nullptr) {
                existing->timestamp = ++this->prefixSnapshotTimestamp;
                state.lastPrefixSnapshotLen = cachedLen;
                return;
            }
        }

        std::shared_ptr<PrefixSnapshot> snapshot(new PrefixSnapshot());
        snapshot->cachedLen = cachedLen;
        snapshot->requestId = state.prefixRequestId;
        snapshot->tensorBytes = tensorBytes;
        snapshot->stateBytes = stateBytes;
        snapshot->tokens = state.processedTokens;
        snapshot->state = state;
        snapshot->state.borrowedPrefixSnapshot.reset();
        snapshot->state.indexerHostMirrorTransfers.clear();
        snapshot->state.indexerTailKeyTensors.clear();
        snapshot->state.indexerTailPositionTensors.clear();
        snapshot->state.indexerBlockKeyTensors.clear();
        snapshot->state.geometricCacheGrowthReadyLayers.clear();
        snapshot->state.lastPrefixSnapshotLen = cachedLen;
        snapshot->layers.resize(this->block_cnt);

        auto copyTensorToResident = [](const Data &source,
                                       Data &destination,
                                       DataDevice &sourceDevice,
                                       std::vector<int> &sourceDeviceIds)
                                       -> bool {
            if (source.dims.empty() ||
                (source.dataDevice == DataDevice::CUDA &&
                 source.cudaData == nullptr) ||
                (source.dataDevice == DataDevice::CPU &&
                 source.cpuData == nullptr)) {
                return false;
            }
            sourceDevice = source.dataDevice;
            sourceDeviceIds = source.dataDeviceIds;
#ifdef USE_CUDA
            Qwen4CudaDeviceGuard deviceGuard(
                source.dataDevice == DataDevice::CUDA
                    ? source.dataDeviceIds : std::vector<int>());
#endif
            if (!source.isLinearAttention && source.dims.size() >= 2) {
                // Growing K/V caches retain spare token-axis capacity.  A
                // full-range Split copies only the logical tokens into tight
                // snapshot storage instead of duplicating that unused space.
                fastllm::Split(source, 1, 0, source.dims[1], destination);
            } else {
                destination.CopyFrom(source);
            }
            destination.dataDeviceIds = source.dataDeviceIds;
            destination.isKVCache = true;
            destination.isLinearAttention = source.isLinearAttention;
            destination.isLinearAttentionTransposed =
                source.isLinearAttentionTransposed;
            destination.isPagedKVCache = false;
            destination.pagedKVCacheData = nullptr;
            destination.pageIndex.clear();
            destination.lastPageLen = 0;
            destination.multiDeviceData = false;
            destination.multiDeviceDatas.clear();
            destination.ClearTensorParallelLayout();
            // Keep CUDA caches on their execution device. CopyFrom performs a
            // D2D snapshot; CPU caches retain the established host fallback.
            return (destination.dataDevice == DataDevice::CUDA &&
                    destination.cudaData != nullptr) ||
                   (destination.dataDevice == DataDevice::CPU &&
                    destination.cpuData != nullptr);
        };

        auto copyQsaTensorToResident = [](const Data &source,
                                          Data &destination) -> bool {
            if (source.dims.empty() || source.dims[0] <= 0 ||
                source.multiDeviceData ||
                (source.dataDevice == DataDevice::CUDA &&
                 source.cudaData == nullptr) ||
                (source.dataDevice == DataDevice::CPU &&
                 source.cpuData == nullptr)) {
                return false;
            }
#ifdef USE_CUDA
            Qwen4CudaDeviceGuard deviceGuard(
                source.dataDevice == DataDevice::CUDA
                    ? source.dataDeviceIds : std::vector<int>());
#endif
            fastllm::Split(source, 0, 0, source.dims[0],
                           destination);
            destination.dataDeviceIds = source.dataDeviceIds;
            destination.isKVCache = false;
            destination.isLinearAttention = false;
            destination.multiDeviceData = false;
            destination.multiDeviceDatas.clear();
            destination.ClearTensorParallelLayout();
            return (destination.dataDevice == DataDevice::CUDA &&
                    destination.cudaData != nullptr) ||
                   (destination.dataDevice == DataDevice::CPU &&
                    destination.cpuData != nullptr);
        };

        for (int layer = 0; layer < this->block_cnt; layer++) {
            PrefixLayerSnapshot &layerSnapshot = snapshot->layers[layer];
            layerSnapshot.linear = this->IsLinearAttentionLayer(layer);
            if (!copyTensorToResident(
                    pastKeyValues[layer].first,
                    layerSnapshot.first,
                    layerSnapshot.firstDevice,
                    layerSnapshot.firstDeviceIds) ||
                !copyTensorToResident(
                    pastKeyValues[layer].second,
                    layerSnapshot.second,
                    layerSnapshot.secondDevice,
                    layerSnapshot.secondDeviceIds)) {
                return;
            }
            if (layerSnapshot.linear) {
                continue;
            }

            const int tailCount =
                cachedLen % this->indexerCompressRatio;
            const int blockCount =
                cachedLen / this->indexerCompressRatio;
            const auto tailKeys =
                state.indexerTailKeyTensors.find(layer);
            const auto tailPositions =
                state.indexerTailPositionTensors.find(layer);
            const auto blockKeys =
                state.indexerBlockKeyTensors.find(layer);
            const bool hasDeviceCache =
                (tailCount == 0 ||
                 (tailKeys != state.indexerTailKeyTensors.end() &&
                  validQsaTensor(
                      tailKeys->second,
                      {tailCount, this->indexerHeadDim}))) &&
                (tailCount == 0 ||
                 (tailPositions !=
                      state.indexerTailPositionTensors.end() &&
                  validQsaTensor(
                      tailPositions->second, {tailCount}))) &&
                (blockCount == 0 ||
                 (blockKeys != state.indexerBlockKeyTensors.end() &&
                  validQsaTensor(
                      blockKeys->second,
                      {blockCount, this->indexerHeadDim})));
            if (!hasDeviceCache) {
                continue;
            }
            layerSnapshot.qsaDeviceCache = true;
            if ((tailCount > 0 &&
                 (!copyQsaTensorToResident(
                      *tailKeys->second,
                      layerSnapshot.qsaTailKeys) ||
                  !copyQsaTensorToResident(
                      *tailPositions->second,
                      layerSnapshot.qsaTailPositions))) ||
                (blockCount > 0 &&
                 !copyQsaTensorToResident(
                     *blockKeys->second,
                     layerSnapshot.qsaBlockKeys))) {
                return;
            }
        }

        size_t recordCount = 0;
        uint64_t recordBytes = 0;
        {
            std::lock_guard<std::mutex> guard(this->prefixCacheMutex);
            auto existing = FindPrefixSnapshotLocked(
                snapshot->tokens, cachedLen, cachedLen);
            if (existing != nullptr) {
                existing->timestamp = ++this->prefixSnapshotTimestamp;
            } else {
                snapshot->timestamp = ++this->prefixSnapshotTimestamp;
                this->prefixSnapshots.push_back(snapshot);
            }

            const int maxPerRequest = std::max(
                1, Qwen4EnvInt(
                    "FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_PER_REQUEST", 4));
            while (std::count_if(
                       this->prefixSnapshots.begin(),
                       this->prefixSnapshots.end(),
                       [&](const std::shared_ptr<PrefixSnapshot> &item) {
                           return item != nullptr &&
                                  item->requestId == state.prefixRequestId;
                       }) > maxPerRequest) {
                auto oldest = this->prefixSnapshots.end();
                for (auto it = this->prefixSnapshots.begin();
                     it != this->prefixSnapshots.end(); ++it) {
                    if (*it == nullptr ||
                        (*it)->requestId != state.prefixRequestId) {
                        continue;
                    }
                    if (oldest == this->prefixSnapshots.end() ||
                        (*it)->timestamp < (*oldest)->timestamp) {
                        oldest = it;
                    }
                }
                if (oldest == this->prefixSnapshots.end()) {
                    break;
                }
                this->prefixSnapshots.erase(oldest);
            }

            const int maxRecords = std::max(
                1, Qwen4EnvInt(
                    "FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_RECORDS", 8));
            for (const auto &item : this->prefixSnapshots) {
                if (item != nullptr) {
                    recordBytes += item->tensorBytes + item->stateBytes;
                }
            }
            while ((int)this->prefixSnapshots.size() > maxRecords ||
                   recordBytes > maxSnapshotBytes) {
                auto oldest = std::min_element(
                    this->prefixSnapshots.begin(),
                    this->prefixSnapshots.end(),
                    [](const std::shared_ptr<PrefixSnapshot> &left,
                       const std::shared_ptr<PrefixSnapshot> &right) {
                        if (left == nullptr) {
                            return right != nullptr;
                        }
                        if (right == nullptr) {
                            return false;
                        }
                        return left->timestamp < right->timestamp;
                    });
                if (*oldest != nullptr) {
                    const uint64_t oldestBytes =
                        (*oldest)->tensorBytes + (*oldest)->stateBytes;
                    recordBytes -= std::min(recordBytes, oldestBytes);
                }
                this->prefixSnapshots.erase(oldest);
            }
            recordCount = this->prefixSnapshots.size();
        }

        state.lastPrefixSnapshotLen = cachedLen;
        if (Qwen4PrefixCacheDebugEnabled()) {
            std::printf(
                "[qwen4-prefix-cache] record tokens=%d tensors=%.3f GiB "
                "state=%.3f GiB records=%zu resident=%.3f GiB\n",
                cachedLen, tensorBytes / 1073741824.0,
                stateBytes / 1073741824.0, recordCount,
                recordBytes / 1073741824.0);
            std::fflush(stdout);
        }
    }

    bool Qwen4ExpModel::TryRestoreHistoryCache(
            std::vector<int> &inputTokens, int &cacheLen) {
        cacheLen = 0;
        if (!Qwen4PrefixCacheEnabled() || inputTokens.size() <= 1) {
            return false;
        }

        const std::vector<int> originalTokens = inputTokens;
        std::shared_ptr<PrefixSnapshot> snapshot;
        {
            std::lock_guard<std::mutex> guard(this->prefixCacheMutex);
            snapshot = FindPrefixSnapshotLocked(
                originalTokens, (int)originalTokens.size() - 1);
            if (snapshot == nullptr) {
                return false;
            }
            snapshot->timestamp = ++this->prefixSnapshotTimestamp;
            PendingPrefixRestore pending;
            pending.cachedLen = snapshot->cachedLen;
            pending.tokens = originalTokens;
            pending.snapshot = snapshot;
            this->pendingPrefixRestores.push_back(std::move(pending));
            while (this->pendingPrefixRestores.size() > 64) {
                this->pendingPrefixRestores.pop_front();
            }
        }

        cacheLen = snapshot->cachedLen;
        inputTokens.erase(inputTokens.begin(), inputTokens.begin() + cacheLen);
        if (Qwen4PrefixCacheDebugEnabled()) {
            std::printf("[qwen4-prefix-cache] hit cached=%d remaining=%zu\n",
                        cacheLen, inputTokens.size());
            std::fflush(stdout);
        }
        return true;
    }

    bool Qwen4ExpModel::RestorePrefixSnapshot(
            ResponseContext *context,
            const std::shared_ptr<PrefixSnapshot> &snapshot) {
        if (context == nullptr || snapshot == nullptr ||
            snapshot->cachedLen <= 0 ||
            snapshot->cachedLen != context->cacheLen ||
            (int)snapshot->layers.size() < this->block_cnt ||
            (int)context->pastKeyValues.size() < this->block_cnt ||
            (int)snapshot->state.processedTokens.size() !=
                snapshot->cachedLen) {
            return false;
        }
        for (int layer = 0; layer < this->block_cnt; layer++) {
            const PrefixLayerSnapshot &layerSnapshot = snapshot->layers[layer];
            if (layerSnapshot.first.dims.empty() ||
                layerSnapshot.second.dims.empty() ||
                layerSnapshot.linear !=
                    this->IsLinearAttentionLayer(layer)) {
                return false;
            }
            if (layerSnapshot.qsaDeviceCache) {
                const int tailCount =
                    snapshot->cachedLen % this->indexerCompressRatio;
                const int blockCount =
                    snapshot->cachedLen / this->indexerCompressRatio;
                if ((tailCount > 0 &&
                     (layerSnapshot.qsaTailKeys.dims !=
                          std::vector<int>({tailCount,
                                            this->indexerHeadDim}) ||
                      layerSnapshot.qsaTailPositions.dims !=
                          std::vector<int>({tailCount}))) ||
                    (blockCount > 0 &&
                     layerSnapshot.qsaBlockKeys.dims !=
                         std::vector<int>({blockCount,
                                           this->indexerHeadDim}))) {
                    return false;
                }
            }
        }

        auto restoreTensor = [](const Data &source, Data &destination,
                                bool linear, DataDevice targetDevice,
                                const std::vector<int> &targetDeviceIds)
                                -> bool {
#ifdef USE_CUDA
            if (targetDevice == DataDevice::CUDA &&
                source.dataDevice == DataDevice::CUDA &&
                (targetDeviceIds.empty() ||
                 targetDeviceIds == source.dataDeviceIds)) {
                return Qwen4BorrowCudaTensor(
                    source, destination, linear);
            }
#endif
            const long long cacheUid = destination.cacheUid;
            destination.FreeSpace();
            destination.CopyFrom(source);
            destination.cacheUid = cacheUid;
            destination.isKVCache = true;
            destination.isLinearAttention = linear;
            destination.isLinearAttentionTransposed =
                source.isLinearAttentionTransposed;
            destination.isPagedKVCache = false;
            destination.pagedKVCacheData = nullptr;
            destination.pageIndex.clear();
            destination.lastPageLen = 0;
            destination.multiDeviceData = false;
            destination.multiDeviceDatas.clear();
            destination.ClearTensorParallelLayout();
            if (destination.expansionDims.size() != destination.dims.size()) {
                destination.expansionDims = destination.dims;
            }
            if (targetDevice == DataDevice::CUDA) {
                destination.ToDevice(
                    DataDevice::CUDA, targetDeviceIds, true);
                return destination.cudaData != nullptr;
            }
            destination.ToDevice(DataDevice::CPU, true);
            return destination.cpuData != nullptr;
        };

        auto restoreQsaTensor = [](const Data &source,
                                    std::shared_ptr<Data> &destination)
                                    -> bool {
            if (source.dims.empty()) {
                destination.reset();
                return true;
            }
            destination = std::make_shared<Data>();
#ifdef USE_CUDA
            if (source.dataDevice == DataDevice::CUDA &&
                Qwen4BorrowCudaTensor(
                    source, *destination, true)) {
                destination->isKVCache = false;
                destination->isLinearAttention = false;
                return true;
            }
#endif
            destination->CopyFrom(source);
            destination->isKVCache = false;
            destination->isLinearAttention = false;
            destination->multiDeviceData = false;
            destination->multiDeviceDatas.clear();
            destination->ClearTensorParallelLayout();
            return (destination->dataDevice == DataDevice::CUDA &&
                    destination->cudaData != nullptr) ||
                   (destination->dataDevice == DataDevice::CPU &&
                    destination->cpuData != nullptr);
        };

        for (int layer = 0; layer < this->block_cnt; layer++) {
            const PrefixLayerSnapshot &layerSnapshot = snapshot->layers[layer];
            if (!restoreTensor(
                    layerSnapshot.first,
                    context->pastKeyValues[layer].first,
                    layerSnapshot.linear,
                    layerSnapshot.firstDevice,
                    layerSnapshot.firstDeviceIds) ||
                !restoreTensor(
                    layerSnapshot.second,
                    context->pastKeyValues[layer].second,
                    layerSnapshot.linear,
                    layerSnapshot.secondDevice,
                    layerSnapshot.secondDeviceIds)) {
                return false;
            }
        }

        RequestState restored = snapshot->state;
        restored.borrowedPrefixSnapshot = snapshot;
        restored.geometricCacheGrowthReadyLayers.clear();
        restored.indexerHostMirrorTransfers.clear();
        restored.indexerTailKeyTensors.clear();
        restored.indexerTailPositionTensors.clear();
        restored.indexerBlockKeyTensors.clear();
        for (int layer = 0; layer < this->block_cnt; layer++) {
            const PrefixLayerSnapshot &layerSnapshot =
                snapshot->layers[layer];
            if (!layerSnapshot.qsaDeviceCache) {
                continue;
            }
            std::shared_ptr<Data> tailKeys;
            std::shared_ptr<Data> tailPositions;
            std::shared_ptr<Data> blockKeys;
            if (!restoreQsaTensor(
                    layerSnapshot.qsaTailKeys, tailKeys) ||
                !restoreQsaTensor(
                    layerSnapshot.qsaTailPositions,
                    tailPositions) ||
                !restoreQsaTensor(
                    layerSnapshot.qsaBlockKeys, blockKeys)) {
                return false;
            }
            if (tailKeys != nullptr) {
                restored.indexerTailKeyTensors[layer] =
                    tailKeys;
            }
            if (tailPositions != nullptr) {
                restored.indexerTailPositionTensors[layer] =
                    tailPositions;
            }
            if (blockKeys != nullptr) {
                restored.indexerBlockKeyTensors[layer] =
                    blockKeys;
            }
        }
        {
            std::lock_guard<std::mutex> guard(this->prefixCacheMutex);
            restored.prefixRequestId = ++this->prefixRequestCounter;
            if (restored.prefixRequestId <= 0) {
                restored.prefixRequestId = 1;
                this->prefixRequestCounter = 1;
            }
        }
        restored.lastPrefixSnapshotLen = snapshot->cachedLen;
        {
            std::lock_guard<std::mutex> guard(this->stateMutex);
            this->requestStates[&context->pastKeyValues[0].first] =
                std::move(restored);
        }
        return true;
    }
    int Qwen4ExpModel::Forward(
            const Data &inputIds,
            const Data &attentionMask,
            const Data &positionIds,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens,
            std::vector<float> *retLogits) {
        std::vector<std::vector<float> *> batchLogits = {retLogits};
        return ForwardBatch(1, inputIds, attentionMask, positionIds,
                            pastKeyValues, generationConfig, lastTokens,
                            &batchLogits)[0];
    }

    std::vector<int> Qwen4ExpModel::ForwardBatch(
            int batch,
            const Data &inputIds,
            const Data &attentionMask,
            const Data &positionIds,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens,
            std::vector<std::vector<float> *> *retLogits) {
        AssertInFastLLM(batch == 1 && inputIds.dims.size() == 2 &&
                        inputIds.dims[0] == 1,
                        "Qwen4-Exp currently runs one request per Forward call.");
        AssertInFastLLM((int)pastKeyValues.size() >= this->block_cnt,
                        "Qwen4-Exp received too few cache slots.");
        PrepareWeights();

        RequestState *requestState;
        {
            std::lock_guard<std::mutex> guard(this->stateMutex);
            requestState = &this->requestStates[&pastKeyValues[0].first];
        }

        const bool restoredPrefixSnapshot =
            requestState->borrowedPrefixSnapshot != nullptr;
        if (restoredPrefixSnapshot) {
#ifdef USE_CUDA
            // Restored CUDA caches initially alias immutable snapshot storage.
            // Detach all mutable layer states before any attention/GDN update;
            // other requests keep sharing the original snapshot safely.
            for (int layer = 0; layer < this->block_cnt; layer++) {
                Qwen4DetachBorrowedCudaTensor(
                    pastKeyValues[layer].first);
                Qwen4DetachBorrowedCudaTensor(
                    pastKeyValues[layer].second);
            }
            for (auto &item :
                 requestState->indexerTailKeyTensors) {
                if (item.second != nullptr) {
                    Qwen4DetachBorrowedCudaTensor(
                        *item.second, false);
                }
            }
            for (auto &item :
                 requestState->indexerTailPositionTensors) {
                if (item.second != nullptr) {
                    Qwen4DetachBorrowedCudaTensor(
                        *item.second, false);
                }
            }
            for (auto &item :
                 requestState->indexerBlockKeyTensors) {
                if (item.second != nullptr) {
                    Qwen4DetachBorrowedCudaTensor(
                        *item.second, false);
                }
            }
#endif
            requestState->borrowedPrefixSnapshot.reset();
        }

        Data embedding, hiddenBuffers[2];
        Data *hiddenStates = &hiddenBuffers[0];
        Data *nextHiddenStates = &hiddenBuffers[1];
        DumpTensorIfRequested("input_ids", inputIds);
        DumpTensorIfRequested("position_ids", positionIds);
        Embedding(inputIds, this->weight[languagePrefix + "embed_tokens.weight"],
                  embedding);
        embedding.Reshape({batch, inputIds.dims[1], 1, this->embed_dim});
        Repeat(embedding, 2, this->hcCount, *hiddenStates);
        hiddenStates->Reshape({batch, inputIds.dims[1],
                               this->hcCount * this->embed_dim});
        DumpTensorIfRequested("embedding", *hiddenStates);

        Data carriedAttentionNorm, finalHyperNorm;
        bool hasCarriedAttentionNorm = false;
        bool hasFinalHyperNorm = false;
        bool usedDecodeCudaGraph = false;
        Data logits;
        int qsaPreviousLength = 0;
        for (int layer = 0; layer < this->block_cnt; layer++) {
            if (!this->IsLinearAttentionLayer(layer)) {
                const Data &pastKey = pastKeyValues[layer].first;
                qsaPreviousLength = pastKey.dims.empty()
                    ? 0 : pastKey.dims[1];
                break;
            }
        }
        const bool qsaDeviceCompatibleMask =
            attentionMask.dims.empty() ||
            (restoredPrefixSnapshot &&
             Qwen4IsContiguousCausalMask(
                 attentionMask, qsaPreviousLength, inputIds.dims[1]));
        std::map<int, Data> cudaPositionIds;
        auto positionsFor = [&](const Data &reference) -> const Data& {
#ifdef USE_CUDA
            if (reference.dataDevice == DataDevice::CUDA) {
                const int device = reference.dataDeviceIds.empty()
                    ? FastllmCudaGetDevice()
                    : reference.dataDeviceIds[0];
                if (positionIds.dataDevice == DataDevice::CUDA &&
                    positionIds.dataType == DataType::FLOAT32 &&
                    (positionIds.dataDeviceIds.empty() ||
                     positionIds.dataDeviceIds[0] == device)) {
                    return positionIds;
                }
                Data &cached = cudaPositionIds[device];
                if (cached.dims.empty()) {
                    ToDataType(positionIds, cached,
                               DataType::FLOAT32);
                    cached.ToDevice(
                        DataDevice::CUDA,
                        std::vector<int>({device}));
                }
                return cached;
            }
#endif
            return positionIds;
        };
        for (int layer = 0; layer < this->block_cnt; layer++) {
            ApplyDeviceMap(this->deviceMap, layer + 1, this->block_cnt);

            if (layer == this->pleLayer) {
                Data pleOutput;
                RunPLE(*hiddenStates, inputIds, *requestState, pleOutput);
                AddTo(*hiddenStates, pleOutput);
                DumpTensorIfRequested("layer_" + std::to_string(layer) +
                                      "_ple", *hiddenStates);
                carriedAttentionNorm.FreeSpace();
                hasCarriedAttentionNorm = false;
            }

            const std::string layerPrefix = languagePrefix + "layers." +
                                            std::to_string(layer) + ".";
            const std::string attentionHyperPrefix =
                layerPrefix + "attn_hyper_connection.";
            Data attentionNorm, attentionInput, attentionInjection;
            Data attentionOutput;
            Data *normalizedAttention = &attentionNorm;
            if (hasCarriedAttentionNorm) {
                normalizedAttention = &carriedAttentionNorm;
            } else {
                GroupedRMSNorm(
                    *hiddenStates,
                    this->weight[attentionHyperPrefix + "hc_norm.weight"],
                    attentionNorm);
            }
            HyperMixNormalized(*normalizedAttention, attentionHyperPrefix,
                               attentionInput, &attentionInjection);
            normalizedAttention->FreeSpace();
            hasCarriedAttentionNorm = false;
            const Data *layerPositionIds = &positionIds;
            if (this->IsLinearAttentionLayer(layer)) {
                RunLinearAttention(layer, attentionInput,
                                   pastKeyValues[layer].first,
                                   pastKeyValues[layer].second,
                                   attentionOutput);
            } else {
                layerPositionIds = &positionsFor(attentionInput);
                RunFullAttention(layer, attentionInput, attentionMask,
                                 *layerPositionIds,
                                 qsaDeviceCompatibleMask,
                                 pastKeyValues[layer].first,
                                 pastKeyValues[layer].second,
                                 *requestState,
                                 attentionOutput);
            }
            DumpTensorIfRequested("layer_" + std::to_string(layer) +
                                  "_attention", attentionOutput);
            if (!this->IsLinearAttentionLayer(layer) &&
                layer == this->kvCacheId &&
                TryRunDecodeCudaGraphBackbone(
                    layer, *hiddenStates, attentionOutput,
                    attentionInjection, attentionMask,
                    *layerPositionIds,
                    qsaDeviceCompatibleMask,
                    pastKeyValues, *requestState, logits)) {
                usedDecodeCudaGraph = true;
                break;
            }
            const std::string mlpHyperPrefix =
                layerPrefix + "mlp_hyper_connection.";
            Data mlpNorm;
            HyperCombineRMSNorm(
                *hiddenStates, attentionOutput, attentionInjection,
                this->weight[mlpHyperPrefix + "hc_norm.weight"],
                *nextHiddenStates, mlpNorm);
            std::swap(hiddenStates, nextHiddenStates);

            Data mlpInput, mlpInjection, mlpOutput;
            HyperMixNormalized(mlpNorm, mlpHyperPrefix,
                               mlpInput, &mlpInjection);
            mlpNorm.FreeSpace();
            RunMoE(layer, mlpInput, mlpOutput);
            if (layer + 1 == this->block_cnt) {
                const std::string finalHyperPrefix =
                    languagePrefix + "hyper_connection_mixer.";
                HyperCombineRMSNorm(
                    *hiddenStates, mlpOutput, mlpInjection,
                    this->weight[finalHyperPrefix + "hc_norm.weight"],
                    *nextHiddenStates, finalHyperNorm);
                hasFinalHyperNorm = true;
                hasCarriedAttentionNorm = false;
            } else if (layer + 1 != this->pleLayer) {
                const std::string nextAttentionHyperPrefix =
                    languagePrefix + "layers." +
                    std::to_string(layer + 1) +
                    ".attn_hyper_connection.";
                HyperCombineRMSNorm(
                    *hiddenStates, mlpOutput, mlpInjection,
                    this->weight[nextAttentionHyperPrefix +
                                 "hc_norm.weight"],
                    *nextHiddenStates, carriedAttentionNorm);
                hasCarriedAttentionNorm = true;
            } else {
                // PLE changes the residual before the next attention norm, so
                // this single boundary cannot be normalized ahead of time.
                HyperCombine(*hiddenStates, mlpOutput, mlpInjection,
                             *nextHiddenStates);
                hasCarriedAttentionNorm = false;
            }
            std::swap(hiddenStates, nextHiddenStates);
            DumpTensorIfRequested("layer_" + std::to_string(layer) +
                                  "_output", *hiddenStates);
        }

        MaybeRecordPrefixSnapshot(pastKeyValues, *requestState);

        if (!usedDecodeCudaGraph) {
            Data finalHidden;
            AssertInFastLLM(hasFinalHyperNorm,
                            "Qwen4-Exp final hyper norm was not produced.");
            HyperMixNormalized(
                finalHyperNorm, languagePrefix + "hyper_connection_mixer.",
                finalHidden, nullptr);
            finalHyperNorm.FreeSpace();
            DumpTensorIfRequested("final_hidden", finalHidden);
            Data lastHidden;
            if (finalHidden.dims[1] > 1) {
                Split(finalHidden, 1, finalHidden.dims[1] - 1,
                      finalHidden.dims[1], lastHidden);
            } else {
                lastHidden.FakeFrom(finalHidden, 0);
                lastHidden.Resize(finalHidden.dims);
            }

            Linear(lastHidden, this->weight["lm_head.weight"], Data(), logits);
            ToDataType(logits, DataType::FLOAT32);
        }
        ResetLogitsOfEOS(1, &logits, pastKeyValues, generationConfig);
        DumpTensorIfRequested("logits", logits);

        if (generationConfig.output_logits && retLogits != nullptr &&
            !retLogits->empty() && (*retLogits)[0] != nullptr) {
            logits.ToDevice(DataDevice::CPU);
            (*retLogits)[0]->resize(logits.Count(0));
            std::memcpy((*retLogits)[0]->data(), logits.cpuData,
                        (size_t)logits.Count(0) * sizeof(float));
        }

        int token;
        if (generationConfig.IsSimpleGreedy()) {
            Data top;
            TopK(logits, top, 1);
            top.ToDevice(DataDevice::CPU);
            token = (int)(reinterpret_cast<float *>(top.cpuData)[0] + 1e-3f);
        } else {
            const LastTokensUnit &unit = lastTokens.units.empty()
                ? LastTokensUnit() : lastTokens.units[0];
            token = LLMSampling(logits, 0, generationConfig, unit);
        }
        return {token};
    }

    void Qwen4ExpModel::OnResponseContextCreated(ResponseContext *context) {
        if (context == nullptr) {
            return;
        }
        if (!context->pastKeyValues.empty()) {
            std::lock_guard<std::mutex> guard(this->stateMutex);
            this->decodeCudaGraphStates.erase(
                &context->pastKeyValues[0].first);
        }
        std::shared_ptr<PrefixSnapshot> snapshot;
        if (context->cacheLen > 0) {
            std::lock_guard<std::mutex> guard(this->prefixCacheMutex);
            for (auto it = this->pendingPrefixRestores.begin();
                 it != this->pendingPrefixRestores.end(); ++it) {
                if (it->cachedLen == context->cacheLen &&
                    it->tokens == context->allTokens) {
                    snapshot = it->snapshot;
                    this->pendingPrefixRestores.erase(it);
                    break;
                }
            }
        }

        bool restored = false;
        if (snapshot != nullptr) {
            // Cache materialization allocates and copies CUDA tensors.  Use
            // the same exclusion as the generic history-cache restore so a
            // newly launched request cannot interleave those copies with an
            // already running Forward call.
            std::lock_guard<std::mutex> guard(this->forwardLocker);
            restored = RestorePrefixSnapshot(context, snapshot);
        }
        if (restored) {
            // The legacy scheduler classifies preTokens == 0 as a new
            // prefill.  A restored request already owns cache storage, so
            // classifying it as a new prefill makes it count itself against a
            // max-batch=1 admission check and it can never run.  Mark the
            // aligned prefix as processed; the mandatory uncached suffix is
            // then evaluated by the ordinary decode path.
            context->preTokens = context->cacheLen;
            context->intParams["add_special_tokens"] = 0;
            if (context->currentTokens.size() == 1) {
                context->intParams["promptLen"] = context->cacheLen;
                context->intParams["index"] = 0;
            } else {
                context->intParams["promptLen"] =
                    context->cacheLen + context->currentTokens.size();
                context->intParams["index"] = -1;
            }
            if (Qwen4PrefixCacheDebugEnabled()) {
                std::printf("[qwen4-prefix-cache] restore tokens=%d\n",
                            context->cacheLen);
                std::fflush(stdout);
            }
            return;
        }

        if (context->cacheLen > 0) {
            for (int layer = 0;
                 layer < (int)context->pastKeyValues.size(); layer++) {
                const bool linear = layer < this->block_cnt &&
                    this->IsLinearAttentionLayer(layer);
                Qwen4ResetCacheTensor(
                    context->pastKeyValues[layer].first, linear);
                Qwen4ResetCacheTensor(
                    context->pastKeyValues[layer].second, linear);
            }
            context->cacheLen = 0;
            context->currentTokens = context->allTokens;
            context->preTokens = 0;
            context->intParams.clear();
            if (Qwen4PrefixCacheDebugEnabled()) {
                std::printf(
                    "[qwen4-prefix-cache] restore failed; recompute\n");
                std::fflush(stdout);
            }
        }

        const int layerCount = std::min(
            this->block_cnt, (int)context->pastKeyValues.size());
        for (int layer = 0; layer < layerCount; layer++) {
            if (!this->IsLinearAttentionLayer(layer)) {
                continue;
            }
            context->pastKeyValues[layer].first.isLinearAttention = true;
            context->pastKeyValues[layer].second.isLinearAttention = true;
        }
        {
            std::lock_guard<std::mutex> guard(this->stateMutex);
            if (!context->pastKeyValues.empty()) {
                this->requestStates[&context->pastKeyValues[0].first] =
                    RequestState();
            }
        }
    }

    void Qwen4ExpModel::OnResponseContextRemoved(ResponseContext *context) {
        if (context == nullptr) {
            return;
        }
        std::lock_guard<std::mutex> guard(this->stateMutex);
        if (!context->pastKeyValues.empty()) {
            this->decodeCudaGraphStates.erase(
                &context->pastKeyValues[0].first);
            this->requestStates.erase(&context->pastKeyValues[0].first);
        }
    }

    void Qwen4ExpModel::WarmUp() {
        std::printf("Warmup Qwen4-Exp...\n");
        Data inputIds(DataType::FLOAT32, {1, 1}, {(float)this->eosToken});
        Data attentionMask(DataType::FLOAT32, {1, 1, 1}, {0.0f});
        Data positionIds(DataType::FLOAT32, {1, 1}, {0.0f});
        std::vector<std::pair<Data, Data>> cache;
        for (int layer = 0; layer < this->block_cnt; layer++) {
            const DataType cacheType = this->IsLinearAttentionLayer(layer)
                ? DataType::FLOAT32 : this->dataType;
            cache.push_back({Data(cacheType), Data(cacheType)});
        }
        Forward(inputIds, attentionMask, positionIds, cache);
        {
            std::lock_guard<std::mutex> guard(this->stateMutex);
            if (!cache.empty()) {
                this->decodeCudaGraphStates.erase(&cache[0].first);
                this->requestStates.erase(&cache[0].first);
            }
        }

        this->elementsInKVCachePerToken = 0;
        for (int layer = 0; layer < this->block_cnt; layer++) {
            if (!this->IsLinearAttentionLayer(layer) &&
                cache[layer].first.dims.size() == 3) {
                this->elementsInKVCachePerToken +=
                    cache[layer].first.dims[0] * cache[layer].first.dims[2] +
                    cache[layer].second.dims[0] * cache[layer].second.dims[2];
            }
        }
        std::printf("finish.\n");
    }

    void Qwen4ExpModel::DumpTensorIfRequested(const std::string &name,
                                               const Data &data) const {
        const char *directory = std::getenv("FASTLLM_QWEN4_DUMP_DIR");
        if (directory == nullptr || directory[0] == '\0') {
            return;
        }
        const char *only = std::getenv("FASTLLM_QWEN4_DUMP_ONLY");
        if (only != nullptr && only[0] != '\0' && name != only) {
            return;
        }
        Data cpu;
        ToDataType(data, cpu, DataType::FLOAT32);
        cpu.ToDevice(DataDevice::CPU);
        std::string path = std::string(directory) + "/" + name + ".f32";
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        if (!stream.good()) {
            return;
        }
        const int32_t rank = (int32_t)cpu.dims.size();
        stream.write(reinterpret_cast<const char *>(&rank), sizeof(rank));
        for (int dimension : cpu.dims) {
            int32_t value = dimension;
            stream.write(reinterpret_cast<const char *>(&value), sizeof(value));
        }
        stream.write(reinterpret_cast<const char *>(cpu.cpuData),
                     (std::streamsize)cpu.Count(0) * sizeof(float));
    }
}
