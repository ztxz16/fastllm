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
#ifndef USE_ROCM
#include "devices/cuda/fastllm-cuda-moe-policy.h"
#endif
#endif
#ifdef USE_TFACC
#include "fastllm-tfacc.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
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
#ifdef USE_NUMAS
    void RegisterNumas(fastllm::Data *data, std::string weightType);
#endif
#ifdef USE_CUDA
    void FastllmCudaMergeMOEClearGraphUnsafeFallbackFlag();
    bool FastllmCudaMergeMOEUsedGraphUnsafeFallback();
#endif

    const std::string Qwen4ExpModel::languagePrefix = "model.language_model.";
    const std::string Qwen4ExpModel::visualPrefix = "model.visual.";

    static inline int Qwen4ClampInt(int value, int low, int high) {
        return std::max(low, std::min(value, high));
    }

    static float Qwen4BicubicWeight(float x) {
        const float a = -0.75f;
        x = std::fabs(x);
        if (x <= 1.0f) {
            return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
        }
        if (x < 2.0f) {
            return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a;
        }
        return 0.0f;
    }

    static void Qwen4ResizeRgbFrameBicubic(
            const float *source, int sourceHeight, int sourceWidth,
            int targetHeight, int targetWidth, std::vector<float> &target) {
        target.resize((size_t)targetHeight * targetWidth * 3);
        if (sourceHeight == targetHeight && sourceWidth == targetWidth) {
            std::memcpy(target.data(), source,
                        (size_t)sourceHeight * sourceWidth * 3 * sizeof(float));
            return;
        }
        const float scaleY = (float)sourceHeight / targetHeight;
        const float scaleX = (float)sourceWidth / targetWidth;
        for (int y = 0; y < targetHeight; y++) {
            const float sourceY = ((float)y + 0.5f) * scaleY - 0.5f;
            const int baseY = (int)std::floor(sourceY);
            float weightsY[4];
            int indicesY[4];
            for (int offset = 0; offset < 4; offset++) {
                indicesY[offset] = Qwen4ClampInt(
                    baseY + offset - 1, 0, sourceHeight - 1);
                weightsY[offset] = Qwen4BicubicWeight(
                    sourceY - (float)(baseY + offset - 1));
            }
            for (int x = 0; x < targetWidth; x++) {
                const float sourceX = ((float)x + 0.5f) * scaleX - 0.5f;
                const int baseX = (int)std::floor(sourceX);
                float weightsX[4];
                int indicesX[4];
                for (int offset = 0; offset < 4; offset++) {
                    indicesX[offset] = Qwen4ClampInt(
                        baseX + offset - 1, 0, sourceWidth - 1);
                    weightsX[offset] = Qwen4BicubicWeight(
                        sourceX - (float)(baseX + offset - 1));
                }
                float pixel[3] = {0.0f, 0.0f, 0.0f};
                for (int ky = 0; ky < 4; ky++) {
                    for (int kx = 0; kx < 4; kx++) {
                        const float weight = weightsY[ky] * weightsX[kx];
                        const float *sourcePixel = source +
                            ((size_t)indicesY[ky] * sourceWidth +
                             indicesX[kx]) * 3;
                        for (int channel = 0; channel < 3; channel++) {
                            pixel[channel] += sourcePixel[channel] * weight;
                        }
                    }
                }
                float *targetPixel = target.data() +
                    ((size_t)y * targetWidth + x) * 3;
                for (int channel = 0; channel < 3; channel++) {
                    targetPixel[channel] = std::max(
                        0.0f, std::min(255.0f, pixel[channel]));
                }
            }
        }
    }

    static void Qwen4BuildVisionPatches(
            const float *rawData, int sourceFrames,
            int sourceHeight, int sourceWidth,
            int gridT, int gridH, int gridW,
            int patchSize, int temporalPatchSize, int mergeSize,
            const std::vector<float> &imageMean,
            const std::vector<float> &imageStd,
            std::vector<float> &patches,
            std::vector<float> &positionH,
            std::vector<float> &positionW) {
        AssertInFastLLM(
            sourceFrames > 0 && sourceHeight > 0 && sourceWidth > 0,
            "Qwen4-Exp raw media shape is invalid.");
        AssertInFastLLM(gridT > 0 && gridH > 0 && gridW > 0,
                        "Qwen4-Exp grid_thw must be positive.");
        AssertInFastLLM(
            gridH % mergeSize == 0 && gridW % mergeSize == 0,
            "Qwen4-Exp vision grid must be divisible by spatial_merge_size.");
        AssertInFastLLM(
            gridT == (sourceFrames + temporalPatchSize - 1) /
                         temporalPatchSize,
            "Qwen4-Exp grid_thw does not match the sampled frame count.");

        const int targetHeight = gridH * patchSize;
        const int targetWidth = gridW * patchSize;
        std::vector<float> resizedFrames(
            (size_t)sourceFrames * targetHeight * targetWidth * 3);
        for (int frame = 0; frame < sourceFrames; frame++) {
            std::vector<float> resized;
            Qwen4ResizeRgbFrameBicubic(
                rawData + (size_t)frame * sourceHeight * sourceWidth * 3,
                sourceHeight, sourceWidth, targetHeight, targetWidth, resized);
            std::memcpy(
                resizedFrames.data() +
                    (size_t)frame * targetHeight * targetWidth * 3,
                resized.data(),
                (size_t)targetHeight * targetWidth * 3 * sizeof(float));
        }

        const int patchDim =
            3 * temporalPatchSize * patchSize * patchSize;
        const int patchCount = gridT * gridH * gridW;
        patches.clear();
        positionH.clear();
        positionW.clear();
        patches.reserve((size_t)patchCount * patchDim);
        positionH.reserve(patchCount);
        positionW.reserve(patchCount);

        const int blockHeight = gridH / mergeSize;
        const int blockWidth = gridW / mergeSize;
        for (int time = 0; time < gridT; time++) {
            for (int blockH = 0; blockH < blockHeight; blockH++) {
                for (int blockW = 0; blockW < blockWidth; blockW++) {
                    for (int mergeH = 0; mergeH < mergeSize; mergeH++) {
                        for (int mergeW = 0; mergeW < mergeSize; mergeW++) {
                            const int patchH = blockH * mergeSize + mergeH;
                            const int patchW = blockW * mergeSize + mergeW;
                            positionH.push_back((float)patchH);
                            positionW.push_back((float)patchW);
                            for (int channel = 0; channel < 3; channel++) {
                                for (int temporal = 0;
                                     temporal < temporalPatchSize; temporal++) {
                                    const int sourceFrame = std::min(
                                        time * temporalPatchSize + temporal,
                                        sourceFrames - 1);
                                    const float *frame = resizedFrames.data() +
                                        (size_t)sourceFrame * targetHeight *
                                            targetWidth * 3;
                                    for (int y = 0; y < patchSize; y++) {
                                        for (int x = 0; x < patchSize; x++) {
                                            const float pixel = frame[
                                                ((size_t)(patchH * patchSize + y) *
                                                     targetWidth +
                                                 patchW * patchSize + x) * 3 +
                                                channel];
                                            patches.push_back(
                                                (pixel / 255.0f -
                                                 imageMean[channel]) /
                                                imageStd[channel]);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    static void Qwen4BuildInterpolatedPositionEmbedding(
            const float *embeddingWeight, int hiddenSize,
            int gridPerSide, int gridT, int gridH, int gridW,
            int mergeSize, std::vector<float> &output) {
        AssertInFastLLM(
            gridH % mergeSize == 0 && gridW % mergeSize == 0,
            "Qwen4-Exp position interpolation needs merged grid alignment.");
        std::vector<float> heightIndices(gridH);
        std::vector<float> widthIndices(gridW);
        for (int index = 0; index < gridH; index++) {
            heightIndices[index] = gridH > 1
                ? (float)index * (gridPerSide - 1) / (gridH - 1) : 0.0f;
        }
        for (int index = 0; index < gridW; index++) {
            widthIndices[index] = gridW > 1
                ? (float)index * (gridPerSide - 1) / (gridW - 1) : 0.0f;
        }

        std::vector<float> spatial(
            (size_t)gridH * gridW * hiddenSize, 0.0f);
        for (int height = 0; height < gridH; height++) {
            const int lowH = (int)std::floor(heightIndices[height]);
            const int highH = std::min(lowH + 1, gridPerSide - 1);
            const float fractionH = heightIndices[height] - lowH;
            for (int width = 0; width < gridW; width++) {
                const int lowW = (int)std::floor(widthIndices[width]);
                const int highW = std::min(lowW + 1, gridPerSide - 1);
                const float fractionW = widthIndices[width] - lowW;
                const float highHigh = fractionH * fractionW;
                const float highLow = fractionH - highHigh;
                const float lowHigh = fractionW - highHigh;
                const float lowLow = 1.0f - fractionH - lowHigh;
                const float *v00 = embeddingWeight +
                    ((size_t)lowH * gridPerSide + lowW) * hiddenSize;
                const float *v01 = embeddingWeight +
                    ((size_t)lowH * gridPerSide + highW) * hiddenSize;
                const float *v10 = embeddingWeight +
                    ((size_t)highH * gridPerSide + lowW) * hiddenSize;
                const float *v11 = embeddingWeight +
                    ((size_t)highH * gridPerSide + highW) * hiddenSize;
                float *destination = spatial.data() +
                    ((size_t)height * gridW + width) * hiddenSize;
                for (int column = 0; column < hiddenSize; column++) {
                    destination[column] =
                        v00[column] * lowLow + v01[column] * lowHigh +
                        v10[column] * highLow + v11[column] * highHigh;
                }
            }
        }

        output.clear();
        output.reserve((size_t)gridT * gridH * gridW * hiddenSize);
        for (int time = 0; time < gridT; time++) {
            (void)time;
            for (int blockH = 0; blockH < gridH / mergeSize; blockH++) {
                for (int blockW = 0; blockW < gridW / mergeSize; blockW++) {
                    for (int mergeH = 0; mergeH < mergeSize; mergeH++) {
                        for (int mergeW = 0; mergeW < mergeSize; mergeW++) {
                            const int height = blockH * mergeSize + mergeH;
                            const int width = blockW * mergeSize + mergeW;
                            const float *source = spatial.data() +
                                ((size_t)height * gridW + width) * hiddenSize;
                            output.insert(output.end(), source,
                                          source + hiddenSize);
                        }
                    }
                }
            }
        }
    }

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

        void MarkDeviceSynchronized() {
            pending = false;
        }

        void Rollback(int rows) {
            AssertInFastLLM(
                rows >= 0 && rows <= committedRows,
                "Qwen4-Exp QSA host mirror cannot roll forward.");
            Synchronize();
            committedRows = rows;
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

        bool Qwen4PrepareCudaWorkspaceOnCurrentDevice(
                Data &workspace, DataType dataType,
                const std::vector<int> &dims, int device) {
            const bool reset = workspace.isFake ||
                workspace.dataDevice != DataDevice::CUDA ||
                workspace.dataType != dataType ||
                workspace.cudaData == nullptr ||
                (!workspace.dataDeviceIds.empty() &&
                 workspace.dataDeviceIds[0] != device);
            if (reset) {
                if (workspace.isFake) {
                    workspace.isFake = false;
                    workspace.cpuData = nullptr;
                    workspace.cudaData = nullptr;
                    workspace.deviceData = nullptr;
                    workspace.expansionSize = 0;
                    workspace.expansionBytes = 0;
                } else {
                    workspace.FreeSpace();
                }
                workspace.dataType = dataType;
                workspace.UpdateUnitSize();
                workspace.dataDevice = DataDevice::CUDA;
                workspace.dataDeviceIds = {device};
                workspace.multiDeviceData = false;
            }
            workspace.Resize(dims);
            workspace.Allocate(false);
            return workspace.cudaData != nullptr;
        }

        bool Qwen4PrepareDecodeGraphWorkspace(
                Data &workspace, DataType dataType,
                const std::vector<int> &dims, int device) {
            FastllmCudaSetDevice(device);
            return Qwen4PrepareCudaWorkspaceOnCurrentDevice(
                workspace, dataType, dims, device);
        }

        bool Qwen4TryExactCudaMultiLinear(
                const Data &input, const Data *const *weights,
                Data *const *outputs, int count) {
            if (weights == nullptr || outputs == nullptr ||
                count < 2 || count > 4 ||
                input.dataDevice != DataDevice::CUDA ||
                input.dataType != DataType::FLOAT16 ||
                input.cudaData == nullptr || input.multiDeviceData ||
                input.dims.empty() || input.dims.back() <= 0) {
                return false;
            }
            const int columns = input.dims.back();
            const uint64_t inputCount = input.Count(0);
            const int rows = (int)(inputCount / columns);
            if (rows < 1 || rows > 8 ||
                inputCount != (uint64_t)rows * columns) {
                return false;
            }
            const int device = input.dataDeviceIds.empty()
                ? FastllmCudaGetDevice() : input.dataDeviceIds[0];
            for (int i = 0; i < count; i++) {
                const Data *weight = weights[i];
                if (weight == nullptr || outputs[i] == nullptr ||
                    weight->dataDevice != DataDevice::CUDA ||
                    weight->dataType != DataType::FLOAT16 ||
                    weight->cudaData == nullptr ||
                    weight->multiDeviceData || weight->dims.size() != 2 ||
                    weight->dims[0] <= 0 || weight->dims[1] != columns ||
                    (!weight->dataDeviceIds.empty() &&
                     weight->dataDeviceIds[0] != device)) {
                    return false;
                }
            }

            Qwen4CudaDeviceGuard deviceGuard(input.dataDeviceIds);
            for (int i = 0; i < count; i++) {
                std::vector<int> dims = input.dims;
                dims.back() = weights[i]->dims[0];
                if (!Qwen4PrepareDecodeGraphWorkspace(
                        *outputs[i], DataType::FLOAT16, dims, device)) {
                    return false;
                }
            }
            return FastllmCudaHalfMultiLinearExact(
                input, weights, outputs, count);
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

        bool Qwen4CudaOrNumaOnlyDeviceMap(
                const std::map<std::string, int> &deviceMap) {
            for (const auto &item : deviceMap) {
                const std::string &name = item.first;
                const bool cuda = name == "cuda" ||
                    (name.size() > 5 && name.compare(0, 5, "cuda:") == 0);
                const bool numa = name == "numa" ||
                    (name.size() > 5 && name.compare(0, 5, "numa:") == 0);
                if (item.second > 0 && !cuda && !numa) {
                    return false;
                }
            }
            return true;
        }

        bool Qwen4EnvFlagEnabled(const char *name, bool fallback = false) {
            const char *value = std::getenv(name);
            if (value == nullptr || value[0] == '\0') {
                return fallback;
            }
            return std::strcmp(value, "0") != 0 &&
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

        int Qwen4MtpDraftsPerStep() {
            return std::max(0, std::min(
                8, Qwen4EnvInt("FASTLLM_QWEN4_ENABLE_MTP", 0)));
        }

        constexpr float QWEN4_MTP_TYPICAL_POSTERIOR_THRESHOLD = 0.09f;
        constexpr float QWEN4_MTP_TYPICAL_POSTERIOR_ALPHA = 0.3f;

        const std::string kMtpExpertPrefix = "mtp.layers.0.mlp.experts.";
        const std::string kMtpPackedGateName = kMtpExpertPrefix + "gate_up_proj";
        const std::string kMtpPackedDownName = kMtpExpertPrefix + "down_proj";

        thread_local bool qwen4MtpDecodeEquivalentTarget = false;

        struct Qwen4MtpGdnModeGuard {
            bool previous = false;
            explicit Qwen4MtpGdnModeGuard(bool enabled)
                : previous(qwen4MtpDecodeEquivalentTarget) {
                qwen4MtpDecodeEquivalentTarget = enabled;
            }
            ~Qwen4MtpGdnModeGuard() {
                qwen4MtpDecodeEquivalentTarget = previous;
            }
        };

        bool Qwen4PrefixCacheEnabled() {
            return Qwen4EnvFlagEnabled("FASTLLM_PREFIX_CACHE", true);
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

        Data Qwen4BuildCausalMask(int previousLength, int sequence) {
            const int keyLength = previousLength + sequence;
            std::vector<float> values(
                (size_t)sequence * keyLength, 0.0f);
            for (int query = 0; query < sequence; query++) {
                for (int key = previousLength + query + 1;
                     key < keyLength; key++) {
                    // FastLLM's dense Attention mask uses 1 for masked
                    // entries; zero keeps a key visible.
                    values[(size_t)query * keyLength + key] = 1.0f;
                }
            }
            return Data(
                DataType::FLOAT32, {1, sequence, keyLength}, values);
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
            bool parallelRegionsUnavailable = false;
            int captureFailures = 0;
            void *graph = nullptr;
            void *exec = nullptr;
            std::vector<void*> reservedPointers;
        };

        std::mutex mutex;
        int device = -1;
        int graphStartLayer = -1;
        bool startBeforeAttention = false;
        std::vector<void*> linearCachePointers;
        std::vector<uint64_t> fullCacheSignature;
        bool wholeGraphMode = false;
        Data hiddenStates[2];
        Data attentionInput;
        Data attentionInjection;
        Data attentionOutput;
        Data positionIds;
        Data decodeMeta;
        Data logits;
        std::map<int, Data> qsaScoreWorkspaces;
        std::map<int, Data> qsaSelectedWorkspaces;
        std::map<int, Data> qsaTailKeySnapshots;
        std::map<int, Data> qsaTailPositionSnapshots;
        int32_t *pinnedDecodeMeta = nullptr;
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
            positionIds.FreeSpace();
            decodeMeta.FreeSpace();
            logits.FreeSpace();
            for (auto &item : qsaScoreWorkspaces) {
                item.second.FreeSpace();
            }
            for (auto &item : qsaSelectedWorkspaces) {
                item.second.FreeSpace();
            }
            for (auto &item : qsaTailKeySnapshots) {
                item.second.FreeSpace();
            }
            for (auto &item : qsaTailPositionSnapshots) {
                item.second.FreeSpace();
            }
#ifdef USE_CUDA
            if (previousDevice >= 0) {
                FastllmCudaSetDevice(previousDevice);
            }
#endif
            linearCachePointers.clear();
            fullCacheSignature.clear();
            qsaScoreWorkspaces.clear();
            qsaSelectedWorkspaces.clear();
            qsaTailKeySnapshots.clear();
            qsaTailPositionSnapshots.clear();
            wholeGraphMode = false;
            graphStartLayer = -1;
            startBeforeAttention = false;
            device = newDevice;
        }

        ~DecodeCudaGraphState() {
            DestroyGraphs();
#ifdef USE_CUDA
            if (pinnedDecodeMeta != nullptr) {
                FastllmCudaHostFree(pinnedDecodeMeta);
                pinnedDecodeMeta = nullptr;
            }
#endif
        }
    };

    struct Qwen4ExpModel::MtpDraftCudaGraphState {
        struct Segment {
            bool warmed = false;
            bool captured = false;
            bool disabled = false;
            bool parallelRegionsUnavailable = false;
            int captureFailures = 0;
            void *graph = nullptr;
            void *exec = nullptr;
            std::vector<void*> reservedPointers;
            std::vector<uint64_t> inputSignature;
            Data projectedHidden;
            Data attentionOutput;
            Data attentionInjection;
            Data multiHidden;
        };

        std::mutex mutex;
        int device = -1;
        std::map<int, Segment> segments;

        void DestroySegment(Segment &segment) {
#ifdef USE_CUDA
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
#endif
            segment.warmed = false;
            segment.captured = false;
            segment.disabled = false;
            segment.parallelRegionsUnavailable = false;
            segment.captureFailures = 0;
        }

        void Reset(int newDevice) {
#ifdef USE_CUDA
            const int previousDevice = FastllmCudaGetDevice();
            if (device >= 0) {
                FastllmCudaSetDevice(device);
            }
#endif
            for (auto &item : segments) {
                DestroySegment(item.second);
            }
            segments.clear();
#ifdef USE_CUDA
            if (previousDevice >= 0) {
                FastllmCudaSetDevice(previousDevice);
            }
#endif
            device = newDevice;
        }

        ~MtpDraftCudaGraphState() {
            Reset(-1);
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
            languagePrefix + "layers.*.ple.value_proj.weight",
            "mtp.fc_embedding.weight",
            "mtp.fc_hidden.weight",
            "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
            "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
            "mtp.layers.*.attn_hyper_connection.input_mix_weight_down.weight",
            "mtp.layers.*.attn_hyper_connection.input_mix_weight_up.weight",
            "mtp.layers.*.attn_hyper_connection.block_inject_weight.weight",
            "mtp.layers.*.mlp_hyper_connection.input_mix_weight_down.weight",
            "mtp.layers.*.mlp_hyper_connection.input_mix_weight_up.weight",
            "mtp.layers.*.mlp_hyper_connection.block_inject_weight.weight",
            "mtp.layers.*.self_attn.q_proj.weight",
            "mtp.layers.*.self_attn.k_proj.weight",
            "mtp.layers.*.self_attn.v_proj.weight",
            "mtp.layers.*.self_attn.o_proj.weight",
            "mtp.layers.*.self_attn.indexer.index_qk_proj.weight",
            "mtp.layers.*.mlp.gate.weight",
            "mtp.layers.*.mlp.shared_expert_gate.weight",
            "mtp.layers.*.mlp.shared_expert.gate_proj.weight",
            "mtp.layers.*.mlp.shared_expert.up_proj.weight",
            "mtp.layers.*.mlp.shared_expert.down_proj.weight",
            "mtp.layers.*.mlp.shared_expert.gateup_proj.weight",
            "mtp.layers.*.mlp.experts.*.gate_proj.weight",
            "mtp.layers.*.mlp.experts.*.up_proj.weight",
            "mtp.layers.*.mlp.experts.*.down_proj.weight",
            "mtp.layers.*.mlp.experts.*.gateup_proj.weight"
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
        ReleaseMoeCudaCache(this->weights);
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

        auto parseIntegerArray = [&](const std::string &key,
                                     std::vector<int> &values) {
            auto iterator = this->weight.dicts.find(key);
            if (iterator == this->weight.dicts.end() ||
                iterator->second.empty() || iterator->second[0] != '[') {
                return;
            }
            std::string error;
            const json11::Json array =
                json11::Json::parse(iterator->second, error);
            if (!error.empty() || !array.is_array()) {
                return;
            }
            std::vector<int> parsed;
            for (const auto &item : array.array_items()) {
                parsed.push_back(item.int_value());
            }
            values.swap(parsed);
        };
        parseIntegerArray("mrope_section", this->mropeSections);

        this->visionDepth = Qwen4DictInt(
            weight.dicts, "vision_config.depth", 0);
        this->visionHiddenSize = Qwen4DictInt(
            weight.dicts, "vision_config.hidden_size", 0);
        this->visionNumHeads = Qwen4DictInt(
            weight.dicts, "vision_config.num_heads", 0);
        this->visionIntermediateSize = Qwen4DictInt(
            weight.dicts, "vision_config.intermediate_size", 0);
        this->visionPatchSize = Qwen4DictInt(
            weight.dicts, "vision_config.patch_size", 16);
        this->visionTemporalPatchSize = Qwen4DictInt(
            weight.dicts, "vision_config.temporal_patch_size", 2);
        this->visionSpatialMergeSize = Qwen4DictInt(
            weight.dicts, "vision_config.spatial_merge_size", 2);
        this->visionOutHiddenSize = Qwen4DictInt(
            weight.dicts, "vision_config.out_hidden_size", this->embed_dim);
        this->visionNumPositionEmbeddings = Qwen4DictInt(
            weight.dicts, "vision_config.num_position_embeddings", 0);
        this->visionNumGridPerSide = this->visionNumPositionEmbeddings > 0
            ? (int)(std::sqrt((double)this->visionNumPositionEmbeddings) + 0.5)
            : 0;
        this->visionHeadDim = this->visionNumHeads > 0
            ? this->visionHiddenSize / this->visionNumHeads : 0;
        this->imageTokenId = Qwen4DictInt(
            weight.dicts, "image_token_id", -1);
        this->videoTokenId = Qwen4DictInt(
            weight.dicts, "video_token_id", -1);
        this->visionDeepstackIndexes.clear();
        parseIntegerArray("vision_config.deepstack_visual_indexes",
                          this->visionDeepstackIndexes);

        this->rotary_dim = (int)(this->head_dim *
            Qwen4DictFloat(weight.dicts, "partial_rotary_factor", 0.25f) + 1e-5f);
        if (this->mropeSections.size() != 3 ||
            this->mropeSections[0] + this->mropeSections[1] +
                    this->mropeSections[2] != this->rotary_dim / 2) {
            const int base = this->rotary_dim / 6;
            this->mropeSections = {
                base, base, this->rotary_dim / 2 - base * 2};
        }
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
        this->mtpWeightsStatus.store(-1, std::memory_order_release);

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
        if (Qwen4MtpDraftsPerStep() > 0) {
            const std::string mlp = "mtp.layers.0.mlp.";
            const std::string sharedGate =
                mlp + "shared_expert.gate_proj.weight";
            const std::string sharedUp =
                mlp + "shared_expert.up_proj.weight";
            this->weightMergeRules.push_back(WeightMergeRule({
                WeightMergeRuleSingle(
                    {sharedGate, sharedUp},
                    mlp + "shared_expert.gateup_proj.weight",
                    std::string("linearSwiglu"))}));
            for (int expert = 0; expert < this->num_experts; expert++) {
                const std::string expertPrefix = mlp + "experts." +
                    std::to_string(expert) + ".";
                const std::string gate = expertPrefix + "gate_proj.weight";
                const std::string up = expertPrefix + "up_proj.weight";
                const std::string gateUp =
                    expertPrefix + "gateup_proj.weight";
                const std::string down = expertPrefix + "down_proj.weight";
                this->weightMergeRules.push_back(WeightMergeRule({
                    WeightMergeRuleSingle(
                        {gate, up}, gateUp,
                        std::string("linearSwiglu"))}));
                // MTP follows the last target layer's placement policy. This
                // keeps the standard --moe_device behavior while pure CUDA
                // naturally leaves all draft experts on the GPU.
                this->AddSpecialWeight(
                    gateUp, "linearSwiglu", this->block_cnt - 1);
                this->AddSpecialWeight(
                    down, "linearColumn", this->block_cnt - 1);
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
        const bool loadMtp = Qwen4MtpDraftsPerStep() > 0;
        for (const std::string &name : tensorNames) {
            const bool mtpWeight = Qwen4StartsWith(name, "mtp.");
            if (name != "lm_head.weight" &&
                !Qwen4StartsWith(name, languagePrefix) &&
                !Qwen4StartsWith(name, visualPrefix) &&
                !(mtpWeight && loadMtp)) {
                // MTP remains opt-in so normal inference keeps its established
                // memory footprint. Vision weights are loaded when present so
                // the public conditional-generation checkpoint is complete.
                continue;
            }
            if (Qwen4StartsWith(name, visualPrefix)) {
                // ModelOpt explicitly excludes model.visual.* from NVFP4.
                // Keep only matrix weights on FastLLM's standard FP16 Linear
                // path; biases, norms and positional embeddings stay FP32,
                // as required by the generic CUDA/CPU operation contracts.
                const bool visualLinear =
                    name == visualPrefix + "patch_embed.proj.weight" ||
                    name.find(".attn.qkv.weight") != std::string::npos ||
                    name.find(".attn.proj.weight") != std::string::npos ||
                    name.find(".mlp.linear_fc1.weight") != std::string::npos ||
                    name.find(".mlp.linear_fc2.weight") != std::string::npos ||
                    name.find(".merger.linear_fc1.weight") != std::string::npos ||
                    name.find(".merger.linear_fc2.weight") != std::string::npos ||
                    (name.find(".deepstack_merger_list.") != std::string::npos &&
                     (Qwen4EndsWith(name, ".linear_fc1.weight") ||
                      Qwen4EndsWith(name, ".linear_fc2.weight")));
                const DataType visualType = visualLinear
                    ? DataType::FLOAT16 : DataType::DATA_AUTO_NONE;
                result[name].push_back({name, visualType});
                continue;
            }
            if (name == kMtpPackedGateName || name == kMtpPackedDownName) {
                // ModelOpt leaves the packed MTP experts in BF16. Keep the
                // source precision and split the expert axis after loading;
                // these tensors must not inherit the target's NVFP4 layout.
                result[name].push_back({name, DataType::BFLOAT16});
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

    void Qwen4ExpModel::OnWeightLoaded(
            const std::string &weightName,
            const std::set<std::string> &finishedWeightNames) {
        if ((weightName != kMtpPackedGateName && weightName != kMtpPackedDownName) ||
            finishedWeightNames.count(kMtpPackedGateName) == 0 ||
            finishedWeightNames.count(kMtpPackedDownName) == 0) {
            return;
        }
        auto gate = this->weight.weight.find(kMtpPackedGateName);
        auto down = this->weight.weight.find(kMtpPackedDownName);
        if (gate == this->weight.weight.end() ||
            down == this->weight.weight.end()) {
            return;
        }
        const Data &gateUp = gate->second;
        const Data &downProj = down->second;
        AssertInFastLLM(
            gateUp.dims.size() == 3 && downProj.dims.size() == 3 &&
            this->num_experts > 0 &&
            gateUp.dims[0] == this->num_experts &&
            downProj.dims[0] == this->num_experts &&
            downProj.dims[1] > 0 && downProj.dims[2] > 0 &&
            gateUp.dims[1] == 2 * downProj.dims[2] &&
            gateUp.dims[2] == downProj.dims[1],
            "Qwen4-Exp packed MTP expert shapes are inconsistent.\n");
        for (const Data *source : {&gateUp, &downProj}) {
            AssertInFastLLM(
                source->dataDevice == DataDevice::CPU &&
                source->cpuData != nullptr &&
                (source->dataType == DataType::FLOAT16 ||
                 source->dataType == DataType::BFLOAT16 ||
                 source->dataType == DataType::FLOAT32),
                "Qwen4-Exp packed MTP experts require CPU floating-point weights.\n");
        }
        for (int expert = 0; expert < this->num_experts; expert++) {
            for (const Data *source : {&gateUp, &downProj}) {
                const bool isGate = source == &gateUp;
                const std::string name = kMtpExpertPrefix + std::to_string(expert) +
                    (isGate ? ".gateup_proj.weight" : ".down_proj.weight");
                AssertInFastLLM(this->weight.weight.count(name) == 0,
                    "Qwen4-Exp checkpoint mixes packed and separate MTP experts.\n");
                Data &target = this->weight.weight[name];
                target = Data(source->dataType,
                              {source->dims[1], source->dims[2]});
                target.Allocate();
                std::memcpy(target.cpuData,
                            source->cpuData + target.GetBytes() * expert,
                            target.GetBytes());
                target.name = name;
                target.weightType = WeightType::LINEAR;
                target.isModelWeight = true;
                target.CalcWeightSum();
                const std::string kind = isGate ? "linearSwiglu" : "linearColumn";
#ifdef USE_TFACC
                if (ShouldRegisterSpecialWeightForDeviceType(name, "tfacc")) {
                    target.weightSum.resize(1);
                    RegisterFastllmData(&target, kind);
                }
#endif
#ifdef USE_NUMAS
                if (ShouldRegisterSpecialWeightForDeviceType(name, "numa")) {
                    RegisterNumas(&target, kind);
                }
#endif
                MoveSpecialWeightToCudaIfNeeded(name, target);
            }
        }
        this->weight.weight.erase(kMtpPackedGateName);
        this->weight.weight.erase(kMtpPackedDownName);
    }

    void Qwen4ExpModel::OnModelWeightsLoaded() {
        if (this->ngramDevice != "disk") {
            if (MoeCudaCacheRequested()) {
                PrepareWeights();
            }
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
        if (MoeCudaCacheRequested()) {
            PrepareWeights();
        }
    }

    bool Qwen4ExpModel::ShouldDelaySpecialWeightNumaRegistration(
            const std::string &weightName) const {
        if (!MoeCudaCacheRequested()) {
            return false;
        }
        const std::string mainLayerPrefix = languagePrefix + "layers.";
        return Qwen4StartsWith(weightName, mainLayerPrefix) &&
               weightName.find(".mlp.experts.") != std::string::npos &&
               (Qwen4EndsWith(weightName, ".gateup_proj.weight") ||
                Qwen4EndsWith(weightName, ".down_proj.weight"));
    }

    bool Qwen4ExpModel::IsLinearAttentionLayer(int layer) const {
        return layer >= 0 && layer < (int)this->linearLayers.size() &&
               this->linearLayers[layer];
    }

    bool Qwen4ExpModel::HasMtpWeights() const {
        if (Qwen4MtpDraftsPerStep() <= 0) {
            return false;
        }
        const int status = this->mtpWeightsStatus.load(std::memory_order_acquire);
        if (status >= 0) {
            return status != 0;
        }
        const std::vector<std::string> required = {
            "mtp.fc_embedding.weight",
            "mtp.fc_hidden.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.hyper_connection_mixer.hc_norm.weight",
            "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
            "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
            "mtp.layers.0.attn_hyper_connection.hc_norm.weight",
            "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
            "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
            "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight",
            "mtp.layers.0.mlp_hyper_connection.hc_norm.weight",
            "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
            "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight",
            "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.k_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight",
            "mtp.layers.0.self_attn.o_proj.weight",
            "mtp.layers.0.self_attn.q_norm.weight",
            "mtp.layers.0.self_attn.k_norm.weight",
            "mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
            "mtp.layers.0.self_attn.indexer.q_layernorm.weight",
            "mtp.layers.0.self_attn.indexer.k_layernorm.weight",
            "mtp.layers.0.mlp.gate.weight",
            "mtp.layers.0.mlp.shared_expert.gateup_proj.weight",
            "mtp.layers.0.mlp.shared_expert.down_proj.weight",
            "mtp.layers.0.mlp.shared_expert_gate.weight"
        };
        for (const std::string &name : required) {
            const auto it = this->weight.weight.find(name);
            if (it == this->weight.weight.end() || it->second.dims.empty()) {
                return false;
            }
        }
        for (int expert = 0; expert < this->num_experts; expert++) {
            const std::string prefix = "mtp.layers.0.mlp.experts." +
                std::to_string(expert) + ".";
            if (this->weight.weight.find(prefix + "gateup_proj.weight") ==
                    this->weight.weight.end() ||
                this->weight.weight.find(prefix + "down_proj.weight") ==
                    this->weight.weight.end()) {
                return false;
            }
        }
        return true;
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
        if (MoeCudaCacheRequested()) {
            if (!PrepareMoeCudaCache(this->weights)) {
                std::fprintf(
                    stderr,
                    "[Fastllm] Qwen4 CUDA expert cache was "
                    "requested but could not be prepared; using the "
                    "configured MoE device.\n");
            }
        }
        Qwen4AddOne(this->weight[languagePrefix +
            "hyper_connection_mixer.hc_norm.weight"]);
        this->mtpMoeWeights.clear();
        this->mtpMoeBiass.clear();
        const bool hasMtpWeights = HasMtpWeights();
        if (hasMtpWeights) {
            for (const std::string &name : {
                     "mtp.pre_fc_norm_embedding.weight",
                     "mtp.pre_fc_norm_hidden.weight",
                     "mtp.hyper_connection_mixer.hc_norm.weight",
                     "mtp.layers.0.attn_hyper_connection.hc_norm.weight",
                     "mtp.layers.0.mlp_hyper_connection.hc_norm.weight",
                     "mtp.layers.0.self_attn.q_norm.weight",
                     "mtp.layers.0.self_attn.k_norm.weight",
                     "mtp.layers.0.self_attn.indexer.q_layernorm.weight",
                     "mtp.layers.0.self_attn.indexer.k_layernorm.weight"}) {
                Qwen4AddOne(this->weight[name]);
            }
            Data &mtpIndexKeyNorm = this->weight[
                "mtp.layers.0.self_attn.indexer.k_layernorm.weight"];
            AssertInFastLLM(
                mtpIndexKeyNorm.dataType == DataType::FLOAT32 &&
                mtpIndexKeyNorm.dataDevice == DataDevice::CPU &&
                mtpIndexKeyNorm.cpuData != nullptr &&
                mtpIndexKeyNorm.Count(0) ==
                    (uint64_t)this->indexerHeadDim,
                "Qwen4-Exp MTP indexer key norm has an invalid shape.");
            const float *mtpNormValues = reinterpret_cast<const float *>(
                mtpIndexKeyNorm.cpuData);
            this->qsaKeyNormValues[this->block_cnt].assign(
                mtpNormValues, mtpNormValues + this->indexerHeadDim);

            this->mtpMoeWeights.assign(2, nullptr);
            this->mtpMoeBiass.assign(2, nullptr);
            for (int expert = 0; expert < this->num_experts; expert++) {
                const std::string prefix =
                    "mtp.layers.0.mlp.experts." +
                    std::to_string(expert) + ".";
                this->mtpMoeWeights.push_back(
                    &this->weight[prefix + "gateup_proj.weight"]);
                this->mtpMoeWeights.push_back(
                    &this->weight[prefix + "down_proj.weight"]);
                this->mtpMoeBiass.push_back(nullptr);
                this->mtpMoeBiass.push_back(nullptr);
            }
        }
        this->mtpWeightsStatus.store(hasMtpWeights ? 1 : 0, std::memory_order_release);
        this->preparedWeights = true;
    }

    void Qwen4ExpModel::PrepareMtpDraftLmHeadWeight() {
#ifdef USE_CUDA
        if (this->mtpDraftLmHeadReady || this->mtpDraftLmHeadAttempted) {
            return;
        }
        std::lock_guard<std::mutex> guard(this->prepareMutex);
        if (this->mtpDraftLmHeadReady || this->mtpDraftLmHeadAttempted) {
            return;
        }
        // We get here after the target head has run at least once, so the
        // executor has already placed a pure-CUDA lm_head on its final GPU.
        // Delay quantization until now rather than inspecting the still-lazy
        // model weight during PrepareWeights().
        Data &lmHead = this->weight["lm_head.weight"];
        if (lmHead.dataDevice != DataDevice::CUDA ||
            lmHead.cudaData == nullptr || lmHead.dims.size() != 2 ||
            (lmHead.dataType != DataType::FLOAT16 &&
             lmHead.dataType != DataType::BFLOAT16) ||
            lmHead.dataDeviceIds.empty()) {
            this->mtpDraftLmHeadAttempted = true;
            return;
        }
        const int previousDevice = FastllmCudaGetDevice();
        FastllmCudaSetDevice(lmHead.dataDeviceIds[0]);
        const bool useNvfp4 = FastllmCudaMarlinNVFP4Supported(
            lmHead.dims[0], lmHead.dims[1]);
        if (previousDevice >= 0 &&
            previousDevice != lmHead.dataDeviceIds[0]) {
            FastllmCudaSetDevice(previousDevice);
        }
        if ((!useNvfp4 && !Qwen4EnvFlagEnabled("FASTLLM_MTP_FP8_DRAFT_HEAD", true)) ||
            lmHead.dims[1] % (useNvfp4 ? 16 : 128) != 0) {
            this->mtpDraftLmHeadAttempted = true;
            return;
        }
        this->mtpDraftLmHeadAttempted = true;
        const bool quantized = useNvfp4
            ? FastllmCudaQuantizeLinearWeightNVFP4Block16(
                  lmHead, this->mtpDraftLmHeadWeight)
            : FastllmCudaQuantizeLinearWeightFP8E4M3Block128(
                  lmHead, this->mtpDraftLmHeadWeight);
        if (!quantized) {
            return;
        }
        this->mtpDraftLmHeadWeight.name =
            lmHead.name + (useNvfp4
                ? ".mtp_draft_nvfp4" : ".mtp_draft_fp8");
        this->mtpDraftLmHeadWeight.weightType = WeightType::LINEAR;
        this->mtpDraftLmHeadWeight.isModelWeight = true;
        this->mtpDraftLmHeadReady = true;
        std::printf(
            "[Qwen4-Exp MTP] %s draft lm_head prepared "
            "(source retained: %.2f GB, extra quantized copy: %.2f GB).\n",
            useNvfp4 ? "NVFP4" : "FP8",
            lmHead.GetBytes() / 1.0e9,
            this->mtpDraftLmHeadWeight.GetBytes() / 1.0e9);
        std::fflush(stdout);
#endif
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
            Data &mixedInput, Data *injectionWeights,
            const Data *projectionStorage) {
        Data lowRank, inputMix;
        Data &downWeight =
            this->weight[prefix + "input_mix_weight_down.weight"];
        if (injectionWeights != nullptr) {
            Data &injectionWeight =
                this->weight[prefix + "block_inject_weight.weight"];
            const bool halfProjectionWeights =
                downWeight.dataType == DataType::FLOAT16 &&
                injectionWeight.dataType == DataType::FLOAT16;
            const bool useProjectionStorage =
                projectionStorage != nullptr &&
                projectionStorage->dataType == DataType::FLOAT16 &&
                halfProjectionWeights;
            if (halfProjectionWeights &&
                (normalized.dataType == DataType::FLOAT32 ||
                 useProjectionStorage)) {
                const Data &projectionSource = useProjectionStorage
                    ? *projectionStorage : normalized;
                fastllm::Qwen4HyperProject(
                    projectionSource, downWeight, injectionWeight,
                    this->hcCount, lowRank, *injectionWeights,
                    DataType::FLOAT32);
            } else {
                Linear(normalized, downWeight, Data(), lowRank);
                Linear(normalized, injectionWeight,
                       Data(), *injectionWeights);
                fastllm::Qwen4HyperPrepare(
                    lowRank, this->hcCount, lowRank);
                fastllm::Qwen4HyperInject(
                    *injectionWeights, this->hcCount, *injectionWeights);
            }
        } else {
            Linear(normalized, downWeight, Data(), lowRank);
            Mul(lowRank, 1.0f / (float)this->hcCount, lowRank);
            Silu(lowRank, lowRank);
        }
        Data &upWeight =
            this->weight[prefix + "input_mix_weight_up.weight"];
        bool mixedProjected = false;
#if defined(USE_CUDA) && !defined(CUDA_NO_TENSOR_CORE)
        const int rows = normalized.dims.empty()
            ? 0 : (int)(normalized.Count(0) / normalized.dims.back());
        if (!qwen4MtpDecodeEquivalentTarget && rows >= 8 &&
            normalized.dataDevice == DataDevice::CUDA &&
            normalized.dataType == DataType::FLOAT32 &&
            lowRank.dataType == DataType::FLOAT32 &&
            upWeight.dataType == DataType::FLOAT16) {
            fastllm::Qwen4HyperMixProjected(
                normalized, lowRank, upWeight,
                this->hcCount, mixedInput);
            mixedProjected = true;
        }
#endif
        if (!mixedProjected) {
            Linear(lowRank, upWeight, Data(), inputMix);
            fastllm::Qwen4HyperMix(normalized, inputMix,
                                  this->hcCount, mixedInput);
        }

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
            Data &output, Data &normalized,
            Data *normalizedStorage) {
        fastllm::Qwen4HyperCombineRMSNorm(
            hyperInput, blockOutput, injectionWeights, normWeight,
            this->rms_norm_eps, this->hcCount, output, normalized,
            normalizedStorage, DataType::FLOAT16);
    }

    void Qwen4ExpModel::MaterializePLEHostHistory(
            RequestState &state) {
        const int channels = this->hcCount * this->embed_dim;
        const int historyLength =
            (this->pleConvKernel - 1) * this->ngramSize;
        const size_t historyCount =
            (size_t)historyLength * channels;
        if (state.convHistory.size() != historyCount) {
            state.convHistory.assign(historyCount, 0.0f);
        }

        AssertInFastLLM(
            state.pleConvHistoryIndex == 0 ||
                state.pleConvHistoryIndex == 1,
            "Qwen4-Exp PLE device history has an invalid active index.");
        const std::shared_ptr<Data> &deviceHistory =
            state.pleConvHistoryTensors[state.pleConvHistoryIndex];
        if (deviceHistory == nullptr) {
            return;
        }
        AssertInFastLLM(
            deviceHistory->dataType == DataType::FLOAT32 &&
                deviceHistory->dims ==
                    std::vector<int>({historyLength, channels}) &&
                ((deviceHistory->dataDevice == DataDevice::CUDA &&
                  deviceHistory->cudaData != nullptr) ||
                 (deviceHistory->dataDevice == DataDevice::CPU &&
                  deviceHistory->cpuData != nullptr)),
            "Qwen4-Exp PLE device history is invalid.");

        Data hostHistory;
        hostHistory.CopyFrom(*deviceHistory);
        hostHistory.ToDevice(DataDevice::CPU);
        std::memcpy(state.convHistory.data(), hostHistory.cpuData,
                    historyCount * sizeof(float));
    }

    void Qwen4ExpModel::RunPLE(const Data &hyperInput,
                               const Data &inputIds,
                               RequestState &state,
                               Data &output,
                               const std::vector<int> *hostInputTokens) {
        AssertInFastLLM(inputIds.dims.size() == 2 && inputIds.dims[0] == 1,
                        "Qwen4-Exp PLE currently expects one request per forward.");
        const int batch = inputIds.dims[0];
        const int sequence = inputIds.dims[1];
        std::vector<int> convertedInputTokens;
        const int *ids = nullptr;
        if (hostInputTokens != nullptr &&
            hostInputTokens->size() >= (size_t)batch * sequence) {
            ids = hostInputTokens->data();
        } else {
            Data idsCpu;
            ToDataType(inputIds, idsCpu, DataType::FLOAT32);
            idsCpu.ToDevice(DataDevice::CPU);
            const float *values = reinterpret_cast<const float *>(
                idsCpu.cpuData);
            convertedInputTokens.resize((size_t)batch * sequence);
            for (size_t i = 0; i < convertedInputTokens.size(); i++) {
                convertedInputTokens[i] = (int)(values[i] + 0.01f);
            }
            ids = convertedInputTokens.data();
        }
        state.processedTokens.reserve(state.processedTokens.size() + sequence);
        for (int token = 0; token < sequence; token++) {
            state.processedTokens.push_back(ids[token]);
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
            const int current = ids[tokenIndex];
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

        // The ngram lookup above intentionally remains on the host.  Keep the
        // projection tail and the dilated short-convolution state on the
        // execution device for both prefill and decode.  This follows the
        // accelerator-resident short-conv cache used by vLLM while retaining
        // convHistory as a self-contained CPU snapshot/fallback image.
        if (keyNormed.dataDevice == DataDevice::CUDA &&
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

            AssertInFastLLM(
                state.pleConvHistoryIndex == 0 ||
                    state.pleConvHistoryIndex == 1,
                "Qwen4-Exp PLE device history has an invalid active index.");
            std::vector<int> targetDeviceIds =
                keyNormed.dataDeviceIds;
#ifdef USE_CUDA
            // Data::ToDevice canonicalizes an empty CUDA device list to the
            // current device. Canonicalize the producer metadata as well so
            // an otherwise identical request does not look like a device-map
            // migration on every token.
            if (targetDeviceIds.empty()) {
                targetDeviceIds = {FastllmCudaGetDevice()};
            }
#endif
            auto sameHistoryDevice = [&](
                    const std::shared_ptr<Data> &history) {
                return history != nullptr &&
                    history->dataDevice == DataDevice::CUDA &&
                    history->dataDeviceIds == targetDeviceIds &&
                    history->dataType == DataType::FLOAT32 &&
                    history->dims ==
                        std::vector<int>({historyLength, channels}) &&
                    history->cudaData != nullptr;
            };

            const int currentIndex = state.pleConvHistoryIndex;
            if (state.pleConvHistoryTensors[currentIndex] != nullptr &&
                !sameHistoryDevice(
                    state.pleConvHistoryTensors[currentIndex])) {
                // A device-map change is a compatibility boundary.  Preserve
                // the latest state once, then rebuild both buffers on the new
                // execution device.
                MaterializePLEHostHistory(state);
                state.pleConvHistoryTensors[0].reset();
                state.pleConvHistoryTensors[1].reset();
                state.pleConvHistoryIndex = 0;
            }

            if (!sameHistoryDevice(
                    state.pleConvHistoryTensors[
                        state.pleConvHistoryIndex])) {
                state.pleConvHistoryTensors[
                    state.pleConvHistoryIndex] =
                    std::make_shared<Data>(
                        DataType::FLOAT32,
                        std::vector<int>({historyLength, channels}),
                        state.convHistory);
                state.pleConvHistoryTensors[
                    state.pleConvHistoryIndex]->ToDevice(
                        DataDevice::CUDA, targetDeviceIds);
            }
            const int nextIndex = 1 - state.pleConvHistoryIndex;
            if (!sameHistoryDevice(
                    state.pleConvHistoryTensors[nextIndex])) {
                state.pleConvHistoryTensors[nextIndex] =
                    std::make_shared<Data>(
                        DataType::FLOAT32,
                        std::vector<int>({historyLength, channels}));
                state.pleConvHistoryTensors[nextIndex]->ToDevice(
                    DataDevice::CUDA, targetDeviceIds, false);
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

            fastllm::Qwen4PLECausalConv(
                normalized, gatedData, convWeight,
                *state.pleConvHistoryTensors[
                    state.pleConvHistoryIndex],
                this->pleConvKernel, dilation, output,
                *state.pleConvHistoryTensors[nextIndex]);
            state.pleConvHistoryIndex = nextIndex;
            ToDataType(output, hyperInput.dataType);
            output.ToDevice(hyperInput.dataDevice);
            return;
        }

        // CPU and other generic device fallbacks consume the host image.
        // Materialize only at this boundary; ordinary CUDA forwards never
        // synchronize the persistent history back to the host.
        MaterializePLEHostHistory(state);
        state.pleConvHistoryTensors[0].reset();
        state.pleConvHistoryTensors[1].reset();
        state.pleConvHistoryIndex = 0;

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

    void Qwen4ExpModel::PrepareVision() {
        std::lock_guard<std::mutex> guard(this->prepareMutex);
        if (this->visionPrepared) {
            return;
        }
        AssertInFastLLM(
            this->visionDepth > 0 && this->visionHiddenSize > 0 &&
                this->visionNumHeads > 0 &&
                this->visionIntermediateSize > 0 &&
                this->visionOutHiddenSize > 0 &&
                this->visionNumPositionEmbeddings > 0 &&
                this->visionNumGridPerSide > 0,
            "Qwen4-Exp vision_config is incomplete.");
        AssertInFastLLM(
            this->visionHiddenSize % this->visionNumHeads == 0 &&
                this->visionHeadDim > 0 && this->visionHeadDim % 4 == 0,
            "Qwen4-Exp vision hidden size must produce a head dim divisible by four.");
        AssertInFastLLM(
            this->visionNumGridPerSide * this->visionNumGridPerSide ==
                this->visionNumPositionEmbeddings,
            "Qwen4-Exp vision position embedding count must be a square.");
        AssertInFastLLM(
            this->visionImageMean.size() == 3 &&
                this->visionImageStd.size() == 3 &&
                std::all_of(this->visionImageStd.begin(),
                            this->visionImageStd.end(),
                            [](float value) { return value != 0.0f; }),
            "Qwen4-Exp vision normalization requires three nonzero channel scales.");
        AssertInFastLLM(
            this->visionDeepstackIndexes.empty(),
            "Qwen4-Exp deepstack vision is not supported yet.");
        auto hasWeight = [&](const std::string &name) {
            return this->weight.weight.find(name) != this->weight.weight.end();
        };
        AssertInFastLLM(
            hasWeight(visualPrefix + "patch_embed.proj.weight") &&
                hasWeight(visualPrefix + "patch_embed.proj.bias") &&
                hasWeight(visualPrefix + "pos_embed.weight") &&
                hasWeight(visualPrefix + "merger.norm.weight") &&
                hasWeight(visualPrefix + "merger.norm.bias") &&
                hasWeight(visualPrefix + "merger.linear_fc1.weight") &&
                hasWeight(visualPrefix + "merger.linear_fc1.bias") &&
                hasWeight(visualPrefix + "merger.linear_fc2.weight") &&
                hasWeight(visualPrefix + "merger.linear_fc2.bias"),
            "Qwen4-Exp multimodal inference needs model.visual.* weights.");

        Data &patchWeight =
            this->weight[visualPrefix + "patch_embed.proj.weight"];
        const int patchDim = 3 * this->visionTemporalPatchSize *
            this->visionPatchSize * this->visionPatchSize;
        AssertInFastLLM(
            patchWeight.Count(0) ==
                (uint64_t)this->visionHiddenSize * patchDim,
            "Qwen4-Exp vision patch embedding weight shape is invalid.");
        if (patchWeight.dims !=
                std::vector<int>({this->visionHiddenSize, patchDim})) {
            patchWeight.Reshape({this->visionHiddenSize, patchDim});
        }

        const int maxVisionPosition = 8192;
        const int rotaryQuarter = this->visionHeadDim / 4;
        std::vector<float> inverseFrequencies;
        inverseFrequencies.reserve(rotaryQuarter);
        for (int index = 0; index < this->visionHeadDim / 2; index += 2) {
            inverseFrequencies.push_back(
                1.0f / std::pow(10000.0f,
                                (float)index /
                                    (this->visionHeadDim / 2)));
        }
        std::vector<float> sine;
        std::vector<float> cosine;
        sine.reserve((size_t)maxVisionPosition * rotaryQuarter);
        cosine.reserve((size_t)maxVisionPosition * rotaryQuarter);
        for (int position = 0; position < maxVisionPosition; position++) {
            for (float inverseFrequency : inverseFrequencies) {
                const float angle = position * inverseFrequency;
                sine.push_back(std::sin(angle));
                cosine.push_back(std::cos(angle));
            }
        }
        this->visionSinData.CopyFrom(Data(
            DataType::FLOAT32, {maxVisionPosition, rotaryQuarter}, sine));
        this->visionCosData.CopyFrom(Data(
            DataType::FLOAT32, {maxVisionPosition, rotaryQuarter}, cosine));
        this->visionPrepared = true;
    }

    void Qwen4ExpModel::ApplyVisionRotary(
            Data &input, const Data &positionH, const Data &positionW) {
        AssertInFastLLM(
            input.dims.size() == 4 && input.dims.back() % 4 == 0,
            "Qwen4-Exp vision rotary expects [batch, seq, heads, dim].");
        const int axis = (int)input.dims.size() - 1;
        const int quarter = input.dims.back() / 4;
        const int half = quarter * 2;
        Data first, second, third, fourth, heightPair, widthPair;
        Split(input, axis, 0, quarter, first);
        Split(input, axis, quarter, half, second);
        Split(input, axis, half, half + quarter, third);
        Split(input, axis, half + quarter, input.dims.back(), fourth);
        Cat(first, third, axis, heightPair);
        Cat(second, fourth, axis, widthPair);
        LlamaRotatePosition2DPart(
            heightPair, positionH, this->visionSinData,
            this->visionCosData, quarter, half);
        LlamaRotatePosition2DPart(
            widthPair, positionW, this->visionSinData,
            this->visionCosData, quarter, half);

        Data rotatedFirst, rotatedSecond, rotatedThird, rotatedFourth;
        Data firstHalf, secondHalf, rotated;
        Split(heightPair, axis, 0, quarter, rotatedFirst);
        Split(heightPair, axis, quarter, half, rotatedThird);
        Split(widthPair, axis, 0, quarter, rotatedSecond);
        Split(widthPair, axis, quarter, half, rotatedFourth);
        Cat(rotatedFirst, rotatedSecond, axis, firstHalf);
        Cat(rotatedThird, rotatedFourth, axis, secondHalf);
        Cat(firstHalf, secondHalf, axis, rotated);
        input.CopyFrom(rotated);
    }

    void Qwen4ExpModel::ApplyTextRotary(
            Data &input, const Data &positionIds) {
        if (positionIds.dims.size() == 2 && positionIds.dims[0] == 3) {
            fastllm::Qwen35InterleavedRope(
                input, positionIds, this->rotary_dim,
                this->mropeSections[0], this->mropeSections[1],
                this->mropeSections[2], this->rope_base, 1.0f);
        } else {
            LlamaRotatePosition2DPart(
                input, positionIds, this->sinData, this->cosData,
                this->rotary_dim, this->rotary_dim);
        }
    }

    void Qwen4ExpModel::EncodeVisualItems(
            const std::vector<Data *> &rawInputs,
            const Data *gridThwData, bool isVideo,
            Data &features,
            std::vector<std::vector<int>> &gridThwList) {
        gridThwList.clear();
        features = Data();
        if (rawInputs.empty()) {
            return;
        }
        PrepareVision();
        AssertInFastLLM(
            gridThwData != nullptr,
            "Qwen4-Exp multimodal media needs grid_thw metadata.");

        Data gridCpu(*gridThwData);
        gridCpu.ToDevice(DataDevice::CPU);
        AssertInFastLLM(
            gridCpu.dims.size() == 2 &&
                gridCpu.dims[0] == (int)rawInputs.size() &&
                gridCpu.dims[1] == 3,
            "Qwen4-Exp grid_thw should have shape [count, 3].");
        AssertInFastLLM(
            gridCpu.dataType == DataType::FLOAT32 ||
                gridCpu.dataType == DataType::INT32 ||
                gridCpu.dataType == DataType::INT32PARAM,
            "Qwen4-Exp grid_thw should contain float32 or int32 values.");
        auto gridValue = [&](int index) {
            return gridCpu.dataType == DataType::FLOAT32
                ? (int)reinterpret_cast<float *>(gridCpu.cpuData)[index]
                : reinterpret_cast<int *>(gridCpu.cpuData)[index];
        };

        const std::string patchWeightName =
            visualPrefix + "patch_embed.proj.weight";
        const std::string patchBiasName =
            visualPrefix + "patch_embed.proj.bias";
        const float attentionScale =
            std::pow((float)this->visionHeadDim, -0.5f);
        const std::string visionDevice = SelectDeviceFromMap(
            this->deviceMap, 1, std::max(1, this->block_cnt));
        ApplyDeviceMap(this->deviceMap, 1, std::max(1, this->block_cnt));
        std::vector<float> allFeatures;
        int featureCount = 0;

        for (int media = 0; media < (int)rawInputs.size(); media++) {
            AssertInFastLLM(
                rawInputs[media] != nullptr,
                "Qwen4-Exp multimodal media contains a null tensor.");
            const std::vector<int> grid = {
                gridValue(media * 3), gridValue(media * 3 + 1),
                gridValue(media * 3 + 2)};
            gridThwList.push_back(grid);

            Data rawCpu(*rawInputs[media]);
            rawCpu.ToDevice(DataDevice::CPU);
            if (rawCpu.dataType != DataType::FLOAT32) {
                ToDataType(rawCpu, DataType::FLOAT32);
                rawCpu.ToDevice(DataDevice::CPU);
            }
            int sourceFrames = 1;
            int sourceHeight = 0;
            int sourceWidth = 0;
            if (isVideo) {
                AssertInFastLLM(
                    rawCpu.dims.size() == 4 && rawCpu.dims[3] == 3,
                    "Qwen4-Exp video tensor should have shape [T, H, W, 3].");
                sourceFrames = rawCpu.dims[0];
                sourceHeight = rawCpu.dims[1];
                sourceWidth = rawCpu.dims[2];
            } else {
                AssertInFastLLM(
                    rawCpu.dims.size() == 3 && rawCpu.dims[2] == 3,
                    "Qwen4-Exp image tensor should have shape [H, W, 3].");
                sourceHeight = rawCpu.dims[0];
                sourceWidth = rawCpu.dims[1];
            }

            std::vector<float> patches, positionsH, positionsW;
            Qwen4BuildVisionPatches(
                reinterpret_cast<float *>(rawCpu.cpuData),
                sourceFrames, sourceHeight, sourceWidth,
                grid[0], grid[1], grid[2], this->visionPatchSize,
                this->visionTemporalPatchSize,
                this->visionSpatialMergeSize,
                this->visionImageMean, this->visionImageStd,
                patches, positionsH, positionsW);

            const int patchDim = 3 * this->visionTemporalPatchSize *
                this->visionPatchSize * this->visionPatchSize;
            const int patchCount = grid[0] * grid[1] * grid[2];
            const int temporalChunks = grid[0];
            const int spatialPatchCount = grid[1] * grid[2];
            AssertInFastLLM(
                patches.size() == (size_t)patchCount * patchDim &&
                    positionsH.size() == (size_t)patchCount &&
                    positionsW.size() == (size_t)patchCount,
                "Qwen4-Exp vision patch packing size is invalid.");

            Data pixelInput(
                DataType::FLOAT32, {patchCount, patchDim}, patches);
            Data pixelOnDevice(pixelInput);
            Data &patchWeight = this->weight[patchWeightName];
            if (patchWeight.dataType == DataType::FLOAT16 ||
                patchWeight.dataType == DataType::BFLOAT16) {
                // Match Conv3d.forward in Transformers, which casts pixels to
                // the patch projection's checkpoint dtype before convolution.
                ToDataType(pixelOnDevice, patchWeight.dataType);
            }
#ifdef USE_CUDA
            if (visionDevice == "cuda" ||
                Qwen4StartsWith(visionDevice, "cuda:")) {
                int device = FastllmCudaGetDevice();
                if (Qwen4StartsWith(visionDevice, "cuda:")) {
                    try {
                        device = std::stoi(visionDevice.substr(5));
                    } catch (...) {
                        AssertInFastLLM(
                            false,
                            "Qwen4-Exp vision received an invalid CUDA device name.");
                    }
                }
                pixelOnDevice.ToDevice(
                    DataDevice::CUDA, std::vector<int>({device}));
            } else
#endif
            if (patchWeight.dataDevice != DataDevice::CPU) {
                if (!patchWeight.dataDeviceIds.empty()) {
                    pixelOnDevice.ToDevice(
                        patchWeight.dataDevice, patchWeight.dataDeviceIds);
                } else {
                    pixelOnDevice.ToDevice(patchWeight.dataDevice);
                }
            }
            Data hiddenStates;
            Linear(pixelOnDevice, patchWeight,
                   this->weight[patchBiasName], hiddenStates);
            if (hiddenStates.dataType != this->dataType) {
                ToDataType(hiddenStates, this->dataType);
            }
            hiddenStates.Reshape(
                {1, patchCount, this->visionHiddenSize});

            Data positionWeight(
                this->weight[visualPrefix + "pos_embed.weight"]);
            positionWeight.ToDevice(DataDevice::CPU);
            if (positionWeight.dataType != DataType::FLOAT32) {
                ToDataType(positionWeight, DataType::FLOAT32);
                positionWeight.ToDevice(DataDevice::CPU);
            }
            AssertInFastLLM(
                positionWeight.dims.size() == 2 &&
                    positionWeight.dims[0] ==
                        this->visionNumPositionEmbeddings &&
                    positionWeight.dims[1] == this->visionHiddenSize,
                "Qwen4-Exp vision position embedding shape is invalid.");
            std::vector<float> interpolatedPosition;
            Qwen4BuildInterpolatedPositionEmbedding(
                reinterpret_cast<float *>(positionWeight.cpuData),
                this->visionHiddenSize, this->visionNumGridPerSide,
                grid[0], grid[1], grid[2],
                this->visionSpatialMergeSize, interpolatedPosition);
            Data positionEmbedding(
                DataType::FLOAT32,
                {1, patchCount, this->visionHiddenSize},
                interpolatedPosition);
            positionEmbedding.ToDevice(
                hiddenStates.dataDevice, hiddenStates.dataDeviceIds);
            if (positionEmbedding.dataType != hiddenStates.dataType) {
                ToDataType(positionEmbedding, hiddenStates.dataType);
            }
            AddTo(hiddenStates, positionEmbedding);

            Data positionH(
                DataType::FLOAT32, {1, patchCount}, positionsH);
            Data positionW(
                DataType::FLOAT32, {1, patchCount}, positionsW);
            for (int layer = 0; layer < this->visionDepth; layer++) {
                Data blockInput, qkv, query, key, value;
                Data attentionOutput, residual, mlpHidden, mlpOutput;
                const std::string prefix = visualPrefix + "blocks." +
                    std::to_string(layer);
                Mul(hiddenStates, 1.0f, residual);
                LayerNorm(
                    hiddenStates, this->weight[prefix + ".norm1.weight"],
                    this->weight[prefix + ".norm1.bias"], -1, blockInput);
                Linear(
                    blockInput, this->weight[prefix + ".attn.qkv.weight"],
                    this->weight[prefix + ".attn.qkv.bias"], qkv);
                Split(qkv, -1, 0, this->visionHiddenSize, query);
                Split(qkv, -1, this->visionHiddenSize,
                      this->visionHiddenSize * 2, key);
                Split(qkv, -1, this->visionHiddenSize * 2,
                      this->visionHiddenSize * 3, value);
                query.Reshape({1, patchCount, this->visionNumHeads,
                               this->visionHeadDim});
                key.Reshape({1, patchCount, this->visionNumHeads,
                             this->visionHeadDim});
                value.Reshape({1, patchCount, this->visionNumHeads,
                               this->visionHeadDim});
                ApplyVisionRotary(query, positionH, positionW);
                ApplyVisionRotary(key, positionH, positionW);
                PermuteSelf(query, {0, 2, 1, 3});
                PermuteSelf(key, {0, 2, 1, 3});
                PermuteSelf(value, {0, 2, 1, 3});
                query.Reshape({this->visionNumHeads * temporalChunks,
                               spatialPatchCount, this->visionHeadDim});
                key.Reshape({this->visionNumHeads * temporalChunks,
                             spatialPatchCount, this->visionHeadDim});
                value.Reshape({this->visionNumHeads * temporalChunks,
                               spatialPatchCount, this->visionHeadDim});
                Attention(query, key, value, *GetEmptyData(),
                          attentionOutput, 1, attentionScale, 2);
                attentionOutput.Reshape(
                    {this->visionNumHeads, temporalChunks,
                     spatialPatchCount, this->visionHeadDim});
                PermuteSelf(attentionOutput, {1, 2, 0, 3});
                attentionOutput.Reshape(
                    {1, patchCount, this->visionHiddenSize});
                Linear(
                    attentionOutput,
                    this->weight[prefix + ".attn.proj.weight"],
                    this->weight[prefix + ".attn.proj.bias"],
                    attentionOutput);
                if (attentionOutput.dataType != residual.dataType) {
                    ToDataType(attentionOutput, residual.dataType);
                }
                AddTo(residual, attentionOutput);
                hiddenStates.CopyFrom(residual);

                Mul(hiddenStates, 1.0f, residual);
                LayerNorm(
                    hiddenStates, this->weight[prefix + ".norm2.weight"],
                    this->weight[prefix + ".norm2.bias"], -1, blockInput);
                Linear(
                    blockInput,
                    this->weight[prefix + ".mlp.linear_fc1.weight"],
                    this->weight[prefix + ".mlp.linear_fc1.bias"],
                    mlpHidden);
                if (mlpHidden.dataType != DataType::FLOAT32) {
                    ToDataType(mlpHidden, DataType::FLOAT32);
                }
                GeluNew(mlpHidden, mlpHidden);
                Linear(
                    mlpHidden,
                    this->weight[prefix + ".mlp.linear_fc2.weight"],
                    this->weight[prefix + ".mlp.linear_fc2.bias"],
                    mlpOutput);
                if (mlpOutput.dataType != residual.dataType) {
                    ToDataType(mlpOutput, residual.dataType);
                }
                AddTo(residual, mlpOutput);
                hiddenStates.CopyFrom(residual);
            }

            Data mergerInput;
            LayerNorm(
                hiddenStates, this->weight[visualPrefix + "merger.norm.weight"],
                this->weight[visualPrefix + "merger.norm.bias"],
                -1, mergerInput);
            const int mergeUnit = this->visionSpatialMergeSize *
                this->visionSpatialMergeSize;
            AssertInFastLLM(
                patchCount % mergeUnit == 0,
                "Qwen4-Exp vision merger received a partial merge unit.");
            mergerInput.Reshape(
                {patchCount / mergeUnit,
                 this->visionHiddenSize * mergeUnit});
            Data mergerHidden, mergerOutput;
            Linear(
                mergerInput,
                this->weight[visualPrefix + "merger.linear_fc1.weight"],
                this->weight[visualPrefix + "merger.linear_fc1.bias"],
                mergerHidden);
            if (mergerHidden.dataType != DataType::FLOAT32) {
                ToDataType(mergerHidden, DataType::FLOAT32);
            }
            Gelu(mergerHidden, mergerHidden);
            Linear(
                mergerHidden,
                this->weight[visualPrefix + "merger.linear_fc2.weight"],
                this->weight[visualPrefix + "merger.linear_fc2.bias"],
                mergerOutput);
#ifdef USE_CUDA
            if (mergerOutput.dataDevice == DataDevice::CUDA) {
                FastllmCudaSyncCurrentThreadStream();
            }
#endif
            mergerOutput.ToDevice(DataDevice::CPU);
            if (mergerOutput.dataType != DataType::FLOAT32) {
                ToDataType(mergerOutput, DataType::FLOAT32);
                mergerOutput.ToDevice(DataDevice::CPU);
            }
            AssertInFastLLM(
                mergerOutput.dims.size() == 2 &&
                    mergerOutput.dims[1] == this->visionOutHiddenSize,
                "Qwen4-Exp vision merger output shape is invalid.");
            const float *values =
                reinterpret_cast<float *>(mergerOutput.cpuData);
            allFeatures.insert(
                allFeatures.end(), values, values + mergerOutput.Count(0));
            featureCount += mergerOutput.dims[0];
        }

        if (featureCount > 0) {
            features.CopyFrom(Data(
                DataType::FLOAT32,
                {1, featureCount, this->visionOutHiddenSize},
                allFeatures));
        }
    }

    void Qwen4ExpModel::BuildMultimodalPositionData(
            const Data &inputIds,
            const std::vector<std::vector<int>> &imageGridThwList,
            const std::vector<std::vector<int>> &videoGridThwList,
            Data &mmTokenTypeIds, Data &mropePositionIds,
            Data &mropePositionDelta) {
        Data idsCpu(inputIds);
        idsCpu.ToDevice(DataDevice::CPU);
        if (idsCpu.dataType != DataType::FLOAT32) {
            ToDataType(idsCpu, DataType::FLOAT32);
            idsCpu.ToDevice(DataDevice::CPU);
        }
        AssertInFastLLM(
            idsCpu.dims.size() == 2 && idsCpu.dims[0] == 1,
            "Qwen4-Exp multimodal positions expect one text batch.");
        const int sequence = idsCpu.dims[1];
        const float *ids = reinterpret_cast<float *>(idsCpu.cpuData);
        std::vector<int> tokenTypes(sequence, 0);
        for (int index = 0; index < sequence; index++) {
            const int token = (int)ids[index];
            if (token == this->imageTokenId) {
                tokenTypes[index] = 1;
            } else if (token == this->videoTokenId) {
                tokenTypes[index] = 2;
            }
        }

        std::vector<std::vector<int>> repeatedVideoGrids;
        for (const auto &grid : videoGridThwList) {
            for (int time = 0; time < grid[0]; time++) {
                repeatedVideoGrids.push_back({1, grid[1], grid[2]});
            }
        }

        std::vector<float> typeValues(sequence, 0.0f);
        std::vector<float> temporalPositions;
        std::vector<float> heightPositions;
        std::vector<float> widthPositions;
        temporalPositions.reserve(sequence);
        heightPositions.reserve(sequence);
        widthPositions.reserve(sequence);
        int currentPosition = 0;
        int imageIndex = 0;
        int videoIndex = 0;
        for (int begin = 0; begin < sequence;) {
            const int type = tokenTypes[begin];
            int end = begin + 1;
            while (end < sequence && tokenTypes[end] == type) {
                end++;
            }
            const int length = end - begin;
            std::fill(typeValues.begin() + begin,
                      typeValues.begin() + end, (float)type);
            if (type == 0) {
                for (int offset = 0; offset < length; offset++) {
                    const float position =
                        (float)(currentPosition + offset);
                    temporalPositions.push_back(position);
                    heightPositions.push_back(position);
                    widthPositions.push_back(position);
                }
                currentPosition += length;
            } else {
                const std::vector<int> *grid = nullptr;
                if (type == 1) {
                    AssertInFastLLM(
                        imageIndex < (int)imageGridThwList.size(),
                        "Qwen4-Exp image metadata does not match placeholders.");
                    grid = &imageGridThwList[imageIndex++];
                } else {
                    AssertInFastLLM(
                        videoIndex < (int)repeatedVideoGrids.size(),
                        "Qwen4-Exp video metadata does not match placeholders.");
                    grid = &repeatedVideoGrids[videoIndex++];
                }
                const int llmGridT = (*grid)[0];
                const int llmGridH =
                    (*grid)[1] / this->visionSpatialMergeSize;
                const int llmGridW =
                    (*grid)[2] / this->visionSpatialMergeSize;
                AssertInFastLLM(
                    llmGridT > 0 && llmGridH > 0 && llmGridW > 0 &&
                        length == llmGridT * llmGridH * llmGridW,
                    "Qwen4-Exp vision token count does not match grid_thw.");
                for (int time = 0; time < llmGridT; time++) {
                    for (int height = 0; height < llmGridH; height++) {
                        for (int width = 0; width < llmGridW; width++) {
                            temporalPositions.push_back(
                                (float)(currentPosition + time));
                            heightPositions.push_back(
                                (float)(currentPosition + height));
                            widthPositions.push_back(
                                (float)(currentPosition + width));
                        }
                    }
                }
                currentPosition += std::max((*grid)[1], (*grid)[2]) /
                    this->visionSpatialMergeSize;
            }
            begin = end;
        }
        AssertInFastLLM(
            imageIndex == (int)imageGridThwList.size() &&
                videoIndex == (int)repeatedVideoGrids.size(),
            "Qwen4-Exp multimodal grid metadata was not fully consumed.");

        mmTokenTypeIds.CopyFrom(Data(
            DataType::FLOAT32, {1, sequence}, typeValues));
        std::vector<float> positions;
        positions.reserve((size_t)sequence * 3);
        positions.insert(positions.end(), temporalPositions.begin(),
                         temporalPositions.end());
        positions.insert(positions.end(), heightPositions.begin(),
                         heightPositions.end());
        positions.insert(positions.end(), widthPositions.begin(),
                         widthPositions.end());
        mropePositionIds.CopyFrom(Data(
            DataType::FLOAT32, {3, sequence}, positions));
        float maximum = 0.0f;
        if (!positions.empty()) {
            maximum = *std::max_element(positions.begin(), positions.end());
        }
        mropePositionDelta.CopyFrom(Data(
            DataType::FLOAT32, {1, 1},
            {positions.empty() ? 0.0f : maximum + 1.0f - sequence}));
    }

    void Qwen4ExpModel::MergeMultimodalFeaturesIntoText(
            const Data &mmTokenTypeIds, const Data *imageEmbeds,
            const Data *videoEmbeds, Data &hiddenStates) {
        Data types(mmTokenTypeIds);
        types.ToDevice(DataDevice::CPU);
        if (types.dataType != DataType::FLOAT32) {
            ToDataType(types, DataType::FLOAT32);
            types.ToDevice(DataDevice::CPU);
        }
        const DataType hiddenType = hiddenStates.dataType;
        hiddenStates.ToDevice(DataDevice::CPU);
        AssertInFastLLM(
            (hiddenType == DataType::FLOAT32 ||
             hiddenType == DataType::FLOAT16 ||
             hiddenType == DataType::BFLOAT16) &&
                hiddenStates.dims.size() == 3 &&
                hiddenStates.dims[0] == 1,
            "Qwen4-Exp multimodal hidden states have an invalid shape or type.");
        AssertInFastLLM(
            types.Count(0) == (uint64_t)hiddenStates.dims[1],
            "Qwen4-Exp modality types must match the input sequence.");

        const int hiddenSize = hiddenStates.dims[2];
        const size_t rowBytes = (size_t)hiddenSize * hiddenStates.unitSize /
            hiddenStates.unitSizeDiv;
        Data imageCpu, videoCpu;
        int imageCount = 0;
        int videoCount = 0;
        uint8_t *imageValues = nullptr;
        uint8_t *videoValues = nullptr;
        auto prepareFeatures = [&](const Data *source, Data &destination,
                                   int &count, uint8_t *&values) {
            if (source == nullptr || source->dims.empty()) {
                return;
            }
            destination.CopyFrom(*source);
            destination.ToDevice(DataDevice::CPU);
            if (destination.dataType != hiddenType) {
                ToDataType(destination, hiddenType);
                destination.ToDevice(DataDevice::CPU);
            }
            if (destination.dims.size() == 3 &&
                destination.dims[0] == 1) {
                destination.Reshape(
                    {destination.dims[1], destination.dims[2]});
            }
            AssertInFastLLM(
                destination.dims.size() == 2 &&
                    destination.dims[1] == hiddenSize,
                "Qwen4-Exp visual feature shape does not match text hidden size.");
            count = destination.dims[0];
            values = reinterpret_cast<uint8_t *>(destination.cpuData);
        };
        prepareFeatures(imageEmbeds, imageCpu, imageCount, imageValues);
        prepareFeatures(videoEmbeds, videoCpu, videoCount, videoValues);

        uint8_t *hiddenValues =
            reinterpret_cast<uint8_t *>(hiddenStates.cpuData);
        const float *typeValues =
            reinterpret_cast<float *>(types.cpuData);
        int nextImage = 0;
        int nextVideo = 0;
        for (int token = 0; token < hiddenStates.dims[1]; token++) {
            if ((int)typeValues[token] == 1) {
                AssertInFastLLM(
                    nextImage < imageCount,
                    "Qwen4-Exp image features do not match placeholders.");
                std::memcpy(hiddenValues + (size_t)token * rowBytes,
                            imageValues + (size_t)nextImage++ * rowBytes,
                            rowBytes);
            } else if ((int)typeValues[token] == 2) {
                AssertInFastLLM(
                    nextVideo < videoCount,
                    "Qwen4-Exp video features do not match placeholders.");
                std::memcpy(hiddenValues + (size_t)token * rowBytes,
                            videoValues + (size_t)nextVideo++ * rowBytes,
                            rowBytes);
            }
        }
        AssertInFastLLM(
            nextImage == imageCount && nextVideo == videoCount,
            "Qwen4-Exp visual features were not fully consumed.");
    }

    void Qwen4ExpModel::AdjustPositionIdsWithDelta(
            const Data &positionIds, const Data &mropePositionDelta,
            Data &adjustedPositionIds) {
        adjustedPositionIds.CopyFrom(positionIds);
        adjustedPositionIds.ToDevice(DataDevice::CPU);
        if (adjustedPositionIds.dataType != DataType::FLOAT32) {
            ToDataType(adjustedPositionIds, DataType::FLOAT32);
            adjustedPositionIds.ToDevice(DataDevice::CPU);
        }
        Data deltaCpu(mropePositionDelta);
        deltaCpu.ToDevice(DataDevice::CPU);
        if (deltaCpu.dataType != DataType::FLOAT32) {
            ToDataType(deltaCpu, DataType::FLOAT32);
            deltaCpu.ToDevice(DataDevice::CPU);
        }
        const float delta =
            reinterpret_cast<float *>(deltaCpu.cpuData)[0];
        if (adjustedPositionIds.dims.size() == 2 &&
            adjustedPositionIds.dims[0] == 3) {
            float *values =
                reinterpret_cast<float *>(adjustedPositionIds.cpuData);
            for (uint64_t index = 0;
                 index < adjustedPositionIds.Count(0); index++) {
                values[index] += delta;
            }
            return;
        }
        const int sequence = adjustedPositionIds.Count(0);
        const float *values =
            reinterpret_cast<float *>(adjustedPositionIds.cpuData);
        std::vector<float> expanded((size_t)sequence * 3);
        for (int axis = 0; axis < 3; axis++) {
            for (int token = 0; token < sequence; token++) {
                expanded[(size_t)axis * sequence + token] =
                    values[token] + delta;
            }
        }
        adjustedPositionIds.CopyFrom(Data(
            DataType::FLOAT32, {3, sequence}, expanded));
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
        const auto hasFullPositions = [&]() {
            return positions.size() == (size_t)length ||
                positions.size() == (size_t)length * 3;
        };
        if (rawKeys.size() == fullRawCount && hasFullPositions()) {
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
        if (rawKeys.size() == fullRawCount && hasFullPositions()) {
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
                                      const std::string &attentionPrefix,
                                      const Data &input,
                                      const Data &baseMask,
                                      const Data &positionIds,
                                      int previousLength,
                                      bool deviceCompatibleMask,
                                      RequestState &state,
                                      Data &qsaMask,
                                      Data &qsaIndices,
                                      Data *rawKeyCapture,
                                      bool fixedCausalWidth) {
        const int batch = input.dims[0];
        const int sequence = input.dims[1];
        AssertInFastLLM(batch == 1 && sequence > 0,
                        "Qwen4-Exp QSA currently expects one non-empty request.");

        const std::string indexer = attentionPrefix + "indexer.";
        Data typedInput, projected, query, currentRawKeys;
        ToDataType(input, typedInput, this->dataType);
        Linear(typedInput,
               this->weight[indexer + "index_qk_proj.weight"],
               Data(), projected);
        const int queryColumns =
            this->indexerHeads * this->indexerHeadDim;
        Split(projected, -1, 0, queryColumns, query);
        Split(projected, -1, queryColumns,
              queryColumns +
                  this->indexerKvHeads * this->indexerHeadDim,
              currentRawKeys);
        query.Reshape({batch, sequence, this->indexerHeads,
                       this->indexerHeadDim});
        currentRawKeys.Reshape(
            {batch, sequence, this->indexerHeadDim});
        RMSNorm(query, this->weight[indexer + "q_layernorm.weight"],
                this->rms_norm_eps, query);
        ApplyTextRotary(query, positionIds);

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
            const bool mropePositions =
                positionCache.size() == (size_t)hostKeyLength * 3;
            AssertInFastLLM(
                rawKeyCache.size() ==
                        (size_t)hostKeyLength * this->indexerHeadDim &&
                    (positionCache.size() == (size_t)hostKeyLength ||
                     mropePositions),
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
                    float *destination = blockKeys.data() +
                        (size_t)block * this->indexerHeadDim;
                    std::copy(pooled.begin(), pooled.end(), destination);
                    for (int column = 0; column < rotaryHalf; column++) {
                        int positionAxis = 0;
                        if (mropePositions && column % 3 == 1 &&
                            column < this->mropeSections[1] * 3) {
                            positionAxis = 1;
                        } else if (mropePositions && column % 3 == 2 &&
                                   column < this->mropeSections[2] * 3) {
                            positionAxis = 2;
                        }
                        const int position = (int)(
                            positionCache[mropePositions
                                ? (size_t)groupStart * 3 + positionAxis
                                : (size_t)groupStart] + 0.01f);
                        AssertInFastLLM(
                            position >= 0 &&
                                position < this->sinData.dims[0],
                            "Qwen4-Exp QSA position exceeds its rotary table.");
                        const float sine = sinValues[
                            (size_t)position * rotaryStride + column];
                        const float cosine = cosValues[
                            (size_t)position * rotaryStride + column];
                        destination[column] =
                            pooled[column] * cosine -
                            pooled[column + rotaryHalf] * sine;
                        destination[column + rotaryHalf] =
                            pooled[column + rotaryHalf] * cosine +
                            pooled[column] * sine;
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
            positionIds.dataDevice == DataDevice::CUDA &&
            !(previousLength > 0 &&
              positionCache.size() == (size_t)previousLength * 3) &&
            !(positionIds.dims.size() == 2 && positionIds.dims[0] == 3);
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
            if (rawKeyCapture != nullptr) {
                if (rawKeyCapture->dims.empty() ||
                    rawKeyCapture->dims[0] == 0) {
                    if (rawKeyCapture->dims.empty()) {
                        rawKeyCapture->CopyFrom(currentKeysFloat);
                        const int capacity = std::max(
                            sequence, Qwen4MtpDraftsPerStep() + 1);
                        if (capacity > sequence) {
                            rawKeyCapture->Expansion(
                                {capacity, this->indexerHeadDim});
                        }
                        rawKeyCapture->Resize(
                            {sequence, this->indexerHeadDim});
                    } else {
                        CatDirect(
                            *rawKeyCapture, currentKeysFloat, 0);
                    }
                } else {
                    CatDirect(*rawKeyCapture, currentKeysFloat, 0);
                }
            }
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
                    currentKeysFloat.dataDeviceIds.empty()) {
                    return true;
                }
                return data.dataDeviceIds[0] ==
                    currentKeysFloat.dataDeviceIds[0];
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
                        DataDevice::CUDA, currentKeysFloat.dataDeviceIds);
                    tailPositions->ToDevice(
                        DataDevice::CUDA, currentKeysFloat.dataDeviceIds);
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
                        DataDevice::CUDA, currentKeysFloat.dataDeviceIds);
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
                        // CUDA Attention interprets the mask in the Q/K/V
                        // activation type.  MTP builds its compact causal
                        // mask in float32, so cast it before handing it to the
                        // fp16 attention kernel.
                        ToDataType(baseMask, qsaMask, input.dataType);
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
                    fixedCausalWidth || sequence > 1
                        ? previousLength : -1);
                if (useSparseAttention) {
                    qsaMask = Data();
                } else {
                    Qwen4QSABuildMask(
                        qsaIndices, input, keyLength, qsaMask);
                    qsaIndices = Data();
                }
            };

#ifdef USE_CUDA
            // The four-row verifier always completes exactly one QSA
            // compression group. Reuse the graph-safe exact kernel on the
            // eager path as well, avoiding the temporary concatenations,
            // split reductions, normalization, RoPE and cache copies below.
            if (sequence == 4 && ratio == 4 &&
                this->indexerHeadDim == 128 &&
                this->rotary_dim == 64) {
                const int device = currentKeysFloat.dataDeviceIds.empty()
                    ? FastllmCudaGetDevice()
                    : currentKeysFloat.dataDeviceIds[0];
                auto allocateCache = [&](std::shared_ptr<Data> &cache,
                                         const std::vector<int> &capacity,
                                         const std::vector<int> &logical) {
                    cache = std::make_shared<Data>();
                    cache->dataType = DataType::FLOAT32;
                    cache->UpdateUnitSize();
                    cache->dataDevice = DataDevice::CUDA;
                    cache->dataDeviceIds = {device};
                    cache->Expansion(capacity);
                    cache->Resize(logical);
                };
                if (oldTailCount == 0 && tailKeys == nullptr) {
                    allocateCache(
                        tailKeys, {ratio, this->indexerHeadDim},
                        {0, this->indexerHeadDim});
                }
                if (oldTailCount == 0 && tailPositions == nullptr) {
                    allocateCache(tailPositions, {ratio}, {0});
                }
                if (cachedBlocks == 0 && blockCache == nullptr) {
                    allocateCache(
                        blockCache, {256, this->indexerHeadDim},
                        {0, this->indexerHeadDim});
                }

                const int requiredBlocks = cachedBlocks + 1;
                if (blockCache != nullptr && sameDevice(*blockCache) &&
                    blockCache->dataType == DataType::FLOAT32 &&
                    blockCache->dims.size() == 2 &&
                    blockCache->dims[1] == this->indexerHeadDim &&
                    Qwen4AxisCapacity(*blockCache, 0) < requiredBlocks) {
                    const int capacity =
                        ((requiredBlocks + 255) / 256) * 256;
                    blockCache->Expansion(
                        {capacity, this->indexerHeadDim});
                }

                Data &normWeight =
                    this->weight[indexer + "k_layernorm.weight"];
                const bool canFuse =
                    tailKeys != nullptr && tailPositions != nullptr &&
                    blockCache != nullptr && sameDevice(*tailKeys) &&
                    sameDevice(*tailPositions) && sameDevice(*blockCache) &&
                    sameDevice(normWeight) && sameDevice(this->sinData) &&
                    sameDevice(this->cosData) &&
                    tailKeys->dataType == DataType::FLOAT32 &&
                    tailPositions->dataType == DataType::FLOAT32 &&
                    blockCache->dataType == DataType::FLOAT32 &&
                    normWeight.dataType == DataType::FLOAT32 &&
                    this->sinData.dataType == DataType::FLOAT32 &&
                    this->cosData.dataType == DataType::FLOAT32 &&
                    Qwen4AxisCapacity(*tailKeys, 0) >= ratio &&
                    Qwen4AxisCapacity(*tailPositions, 0) >= ratio &&
                    Qwen4AxisCapacity(*blockCache, 0) >= requiredBlocks &&
                    tailKeys->strides.size() == 2 &&
                    tailKeys->strides[0] == this->indexerHeadDim &&
                    blockCache->strides.size() == 2 &&
                    blockCache->strides[0] == this->indexerHeadDim;
                if (canFuse) {
                    // Preserve the completed raw group in the pinned mirror
                    // before the fused kernel rotates the four tail slots.
                    int mirroredRows = previousLength - oldTailCount;
                    if (oldTailCount > 0) {
                        Data oldKeys, oldPositions;
                        oldKeys.FakeFrom(*tailKeys, 0);
                        oldKeys.Resize(
                            {oldTailCount, this->indexerHeadDim});
                        oldPositions.FakeFrom(*tailPositions, 0);
                        oldPositions.Resize({oldTailCount});
                        hostTransfer->Queue(
                            oldKeys, oldPositions, mirroredRows,
                            oldTailCount, this->indexerHeadDim);
                        mirroredRows += oldTailCount;
                    }
                    const int newRows = ratio - oldTailCount;
                    Data newKeys, newPositions;
                    newKeys.FakeFrom(currentKeysFloat, 0);
                    newKeys.Resize(
                        {newRows, this->indexerHeadDim});
                    newPositions.FakeFrom(currentPositionsFloat, 0);
                    newPositions.Resize({newRows});
                    hostTransfer->Queue(
                        newKeys, newPositions, mirroredRows,
                        newRows, this->indexerHeadDim);

                    const bool fused =
                        FastllmCudaQwen4QSAAppendCompress4(
                            currentKeysFloat, currentPositionsFloat,
                            normWeight, this->sinData, this->cosData,
                            previousLength, *tailKeys, *tailPositions,
                            *blockCache, this->rms_norm_eps);
                    AssertInFastLLM(
                        fused,
                        "Qwen4-Exp exact QSA compression rejected a validated input.");
                    tailKeys->Resize(
                        {oldTailCount, this->indexerHeadDim});
                    tailPositions->Resize({oldTailCount});
                    blockCache->Resize(
                        {requiredBlocks, this->indexerHeadDim});
                    finishDeviceQsa();
                    return;
                }
            }
#endif

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
                if (blockCache == nullptr || blockCache->dims.empty() ||
                    blockCache->dims[0] == 0) {
                    AssertInFastLLM(
                        cachedBlocks == 0,
                        "Qwen4-Exp QSA lost a committed compressed block.");
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

        const bool cachedMrope = previousLength > 0 &&
            positionCache.size() == (size_t)previousLength * 3;
        const bool currentMrope = positionsCpu.dims.size() == 2 &&
            positionsCpu.dims[0] == 3;
        AssertInFastLLM(
            rawKeyCache.size() % this->indexerHeadDim == 0 &&
                (int)(rawKeyCache.size() / this->indexerHeadDim) ==
                    previousLength &&
                (positionCache.size() == (size_t)previousLength ||
                 cachedMrope) &&
                (!currentMrope || previousLength == 0 || cachedMrope),
            "Qwen4-Exp QSA cache is inconsistent with the attention cache.");
        rawKeyCache.insert(rawKeyCache.end(), rawKeyValues,
                           rawKeyValues + (size_t)sequence *
                                              this->indexerHeadDim);
        if (currentMrope) {
            for (int token = 0; token < sequence; token++) {
                for (int axis = 0; axis < 3; axis++) {
                    positionCache.push_back(
                        currentPositions[(size_t)axis * sequence + token]);
                }
            }
        } else if (cachedMrope) {
            for (int token = 0; token < sequence; token++) {
                for (int axis = 0; axis < 3; axis++) {
                    positionCache.push_back(currentPositions[token]);
                }
            }
        } else {
            positionCache.insert(positionCache.end(), currentPositions,
                                 currentPositions + sequence);
        }

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
                                          Data &output,
                                          Data *qsaRawKeyCapture) {
        RunFullAttentionWithPrefix(
            layer,
            languagePrefix + "layers." + std::to_string(layer) +
                ".self_attn.",
            input, attentionMask, positionIds,
            qsaDeviceCompatibleMask, pastKey, pastValue, state, output,
            qsaRawKeyCapture);
    }

    void Qwen4ExpModel::RunFullAttentionWithPrefix(
                                          int stateLayer,
                                          const std::string &attention,
                                          const Data &input,
                                          const Data &attentionMask,
                                          const Data &positionIds,
                                          bool qsaDeviceCompatibleMask,
                                          Data &pastKey,
                                          Data &pastValue,
                                          RequestState &state,
                                          Data &output,
                                          Data *qsaRawKeyCapture) {
        const int batch = input.dims[0];
        const int sequence = input.dims[1];
        const int previousLength = pastKey.dims.empty() ? 0 : pastKey.dims[1];
        Data typedInput;
        ToDataType(input, typedInput, this->dataType);
        Data qsaMask, qsaIndices;
        BuildQSAMask(stateLayer, attention, typedInput,
                     attentionMask, positionIds,
                     previousLength, qsaDeviceCompatibleMask,
                     state, qsaMask, qsaIndices, qsaRawKeyCapture);

        Data qGate, query, key, value, gate;
        Data &qWeight = this->weight[attention + "q_proj.weight"];
        Data &kWeight = this->weight[attention + "k_proj.weight"];
        Data &vWeight = this->weight[attention + "v_proj.weight"];
        bool fusedQkvProjections = false;
#ifdef USE_CUDA
        // Small decode batches are launch-bound, especially in the MTP
        // draft/verification path. Join the three independent projection
        // grids into one launch without concatenating or repacking weights.
        // Each CTA still owns one output row and therefore keeps the native
        // compensated reduction and FP16 rounding order. Unsupported shapes,
        // types and devices retain the ordinary Linear path below.
        if (sequence == 1 || qwen4MtpDecodeEquivalentTarget) {
            const Data *weights[] = {&qWeight, &kWeight, &vWeight};
            Data *outputs[] = {&qGate, &key, &value};
            fusedQkvProjections = Qwen4TryExactCudaMultiLinear(
                typedInput, weights, outputs, 3);
        }
#endif
        if (!fusedQkvProjections) {
            Linear(typedInput, qWeight, Data(), qGate);
            Linear(typedInput, kWeight, Data(), key);
            Linear(typedInput, vWeight, Data(), value);
        }
        qGate.Reshape({batch, sequence, -1, this->head_dim * 2});
        Split(qGate, -1, 0, this->head_dim, query);
        Split(qGate, -1, this->head_dim, this->head_dim * 2, gate);
        gate.Reshape({batch, sequence, -1});

        key.Reshape({batch, sequence, -1, this->head_dim});
        value.Reshape({batch, sequence, -1, this->head_dim});

        RMSNorm(query, this->weight[attention + "q_norm.weight"],
                this->rms_norm_eps, query);
        RMSNorm(key, this->weight[attention + "k_norm.weight"],
                this->rms_norm_eps, key);
        ApplyTextRotary(query, positionIds);
        ApplyTextRotary(key, positionIds);

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
            state.geometricCacheGrowthReadyLayers.count(stateLayer) != 0;
        Qwen4EnsureAppendCapacity(
            pastKey, key, 1, unitLength,
            kQwen4DenseCacheMaxGrowth, geometricGrowth);
        Qwen4EnsureAppendCapacity(
            pastValue, value, 1, unitLength,
            kQwen4DenseCacheMaxGrowth, geometricGrowth);
        bool appendedWithStridedCudaCache = false;
#ifdef USE_CUDA
        if (!GetKVCacheInCPU() &&
            key.dataDevice == DataDevice::CUDA &&
            value.dataDevice == DataDevice::CUDA) {
            pastKey.ToDevice(
                DataDevice::CUDA, key.dataDeviceIds,
                previousLength > 0);
            pastValue.ToDevice(
                DataDevice::CUDA, value.dataDeviceIds,
                previousLength > 0);
        }
        if (!GetKVCacheInCPU() &&
            key.dataDevice == DataDevice::CUDA &&
            value.dataDevice == DataDevice::CUDA &&
            pastKey.dataDevice == DataDevice::CUDA &&
            pastValue.dataDevice == DataDevice::CUDA) {
            appendedWithStridedCudaCache = FastllmCudaQwen4KVAppend(
                key, value, previousLength, pastKey, pastValue);
        }
#endif
        if (appendedWithStridedCudaCache) {
            pastKey.Resize(
                {key.dims[0], previousLength + sequence,
                 this->head_dim});
            pastValue.Resize(
                {value.dims[0], previousLength + sequence,
                 this->head_dim});
        } else {
            CatDirect(pastKey, key, 1);
            CatDirect(pastValue, value, 1);
        }
        if (sequence == 1) {
            state.geometricCacheGrowthReadyLayers.insert(stateLayer);
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

        SigmoidMulTo(context, gate);
        Linear(context, this->weight[attention + "o_proj.weight"], Data(), output);
    }

    void Qwen4ExpModel::RunLinearAttention(int layer,
                                            const Data &input,
                                            Data &pastConv,
                                            Data &pastRecurrent,
                                            Data &output,
                                            Data *convInputCapture,
                                            Data *alphaCapture,
                                            Data *betaCapture,
                                            Data *recurrentStateOutput) {
        const std::string linear = languagePrefix + "layers." +
            std::to_string(layer) + ".linear_attn.";
        const int batch = input.dims[0];
        const int sequence = input.dims[1];

        AssertInFastLLM(
            (convInputCapture == nullptr) == (alphaCapture == nullptr) &&
            (convInputCapture == nullptr) == (betaCapture == nullptr) &&
            (recurrentStateOutput == nullptr ||
             convInputCapture != nullptr),
            "Qwen4-Exp MTP GDN state output is incomplete.");
        if (convInputCapture != nullptr) {
            AssertInFastLLM(
                qwen4MtpDecodeEquivalentTarget && sequence <= 9,
                "Qwen4-Exp can capture GDN replay inputs only during MTP verification.");
        }
        if (recurrentStateOutput != nullptr) {
            AssertInFastLLM(
                qwen4MtpDecodeEquivalentTarget && sequence > 1 &&
                sequence <= 9,
                "Qwen4-Exp can use out-of-place GDN state only for batched MTP verification.");
        }

        Data typedInput;
        ToDataType(input, typedInput, this->dataType);
        Data localQkv, z, localBeta, localAlpha;
        Data &qkv = convInputCapture == nullptr ? localQkv : *convInputCapture;
        Data &beta = betaCapture == nullptr ? localBeta : *betaCapture;
        Data &alpha = alphaCapture == nullptr ? localAlpha : *alphaCapture;
        Data &qkvWeight = this->weight[linear + "in_proj_qkv.weight"];
        Data &zWeight = this->weight[linear + "in_proj_z.weight"];
        Data &betaWeight = this->weight[linear + "in_proj_b.weight"];
        Data &alphaWeight = this->weight[linear + "in_proj_a.weight"];
        bool fusedInputProjections = false;
#ifdef USE_CUDA
        // All four projections consume the same small decode batch. Joining
        // their output-row grids removes three launches while each CTA keeps
        // the native GEMV's products, compensated reduction tree and FP16
        // rounding. Unsupported types/devices retain the ordinary operations.
        if (sequence == 1 || qwen4MtpDecodeEquivalentTarget) {
            const Data *weights[] = {
                &qkvWeight, &zWeight, &betaWeight, &alphaWeight};
            Data *outputs[] = {&qkv, &z, &beta, &alpha};
            fusedInputProjections = Qwen4TryExactCudaMultiLinear(
                typedInput, weights, outputs, 4);
        }
#endif
        bool fusedGateProjections = false;
        if (!fusedInputProjections) {
            Linear(typedInput, qkvWeight, Data(), qkv);
            Linear(typedInput, zWeight, Data(), z);
#ifdef USE_CUDA
            // Keep the narrower two-grid fusion as a fallback when the larger
            // projection group is unavailable for this device or data type.
            if (qwen4MtpDecodeEquivalentTarget) {
                const Data *weights[] = {&betaWeight, &alphaWeight};
                Data *outputs[] = {&beta, &alpha};
                fusedGateProjections = Qwen4TryExactCudaMultiLinear(
                    typedInput, weights, outputs, 2);
            }
#endif
            if (!fusedGateProjections) {
                Linear(typedInput, betaWeight, Data(), beta);
                Linear(typedInput, alphaWeight, Data(), alpha);
            }
        }
        const bool fusedDecode = sequence == 1;
        // Speculative target verification evaluates only a handful of new
        // tokens.  Advance the recurrent GDN with the exact decode operation
        // for each row so accepted logits are bit-equivalent to ordinary
        // one-token decoding instead of switching to the chunk-prefill
        // numerical path.
        const bool sequentialMtpDecode =
            sequence > 1 && sequence <= 9 &&
            qwen4MtpDecodeEquivalentTarget;
        constexpr int chunkSize = 64;
        bool mixedGdnPrefill = false;
#ifdef USE_CUDA
        mixedGdnPrefill =
            !fusedDecode && !sequentialMtpDecode &&
            qkv.dataDevice == DataDevice::CUDA &&
            qkv.dataType == DataType::FLOAT16 &&
            FastllmCudaCanUseTritonChunkGdnPrefill(
                batch, (sequence + chunkSize - 1) / chunkSize,
                chunkSize, this->head_k_dim, this->head_v_dim, true);
#endif
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
        } else if (sequentialMtpDecode) {
            // The standard token-major operation evaluates the identical
            // causal convolution for every speculative row and advances the
            // float32 history once, without per-row Split/Cat launches.
            fastllm::CausalDepthwiseConv1DPrefill(
                qkv, this->weight[linear + "conv1d.weight"],
                pastConv, this->linearConvKernel, true,
                currentConvolved);
            pastConv.expansionDims = pastConv.dims;
        } else {
            // Keep Q/K/V token-major during prefill.  The standard fused op is
            // mathematically identical to the old permute + Cat + Conv1D +
            // Split + SiLU chain and updates the four-value history in place.
            fastllm::CausalDepthwiseConv1DPrefill(
                qkv, this->weight[linear + "conv1d.weight"],
                pastConv, this->linearConvKernel, true, currentConvolved,
                mixedGdnPrefill
                    ? DataType::FLOAT16 : DataType::FLOAT32);
            pastConv.expansionDims = pastConv.dims;
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
        if (fusedDecode || sequentialMtpDecode) {
            Data core;
            if (fusedDecode) {
                fastllm::GatedDeltaRuleDecode(
                    currentConvolved, alpha, beta,
                    this->weight[linear + "A_log"],
                    this->weight[linear + "dt_bias"],
                    pastRecurrent, this->num_k_heads,
                    this->num_v_heads, this->head_k_dim,
                    this->head_v_dim, 1e-6f, core);
            } else {
                fastllm::GatedDeltaRuleSequence(
                    currentConvolved, alpha, beta,
                    this->weight[linear + "A_log"],
                    this->weight[linear + "dt_bias"],
                    pastRecurrent, this->num_k_heads,
                    this->num_v_heads, this->head_k_dim,
                    this->head_v_dim, 1e-6f, core,
                    recurrentStateOutput);
            }
            core.Reshape({-1, this->head_v_dim});
            z.Reshape({-1, this->head_v_dim});
            Data gatedCore;
            Data *outputCore = &core;
#ifdef USE_CUDA
            if (sequentialMtpDecode &&
                core.dataDevice == DataDevice::CUDA &&
                core.dataType == DataType::FLOAT32 &&
                z.dataDevice == DataDevice::CUDA &&
                z.dataType == DataType::FLOAT16) {
                const int device = core.dataDeviceIds.empty()
                    ? FastllmCudaGetDevice() : core.dataDeviceIds[0];
                Qwen4CudaDeviceGuard deviceGuard(core.dataDeviceIds);
                if (Qwen4PrepareDecodeGraphWorkspace(
                        gatedCore, DataType::FLOAT16,
                        core.dims, device) &&
                    FastllmCudaQwen4GdnOutputGateExact(
                        core, this->weight[linear + "norm.weight"],
                        z, gatedCore, this->rms_norm_eps)) {
                    outputCore = &gatedCore;
                }
            }
#endif
            if (outputCore == &core) {
                ToDataType(core, this->dataType);
                RMSNorm(core, this->weight[linear + "norm.weight"],
                        this->rms_norm_eps, core);
                SigmoidMulTo(core, z);
            }
            outputCore->Reshape({batch, sequence,
                                 this->num_v_heads * this->head_v_dim});
            Linear(*outputCore,
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
        SigmoidMulTo(allCore, z);
        allCore.Reshape({batch, sequence, valueDimension});
        Linear(allCore, this->weight[linear + "out_proj.weight"], Data(), output);
    }

    void Qwen4ExpModel::RunMoE(int layer, const Data &input, Data &output) {
        RunMoEWithPrefix(
            layer,
            languagePrefix + "layers." + std::to_string(layer) + ".mlp.",
            this->weights[layer], this->biass[layer], input, output);
    }

    void Qwen4ExpModel::RunMoEWithPrefix(
            int deviceLayer, const std::string &mlp,
            std::vector<Data *> &moeWeights,
            std::vector<Data *> &moeBiass,
            const Data &input, Data &output) {
        const int batch = input.dims[0];
        const int sequence = input.dims[1];
        Data flattened;
        flattened.FakeFrom(input, 0);
        flattened.Resize(input.dims);
        flattened.Reshape({batch * sequence, input.dims.back()});

        Data routerLogits, expertIndex, expertScore, sharedOutput;
        Linear(flattened, this->weight[mlp + "gate.weight"],
               Data(), routerLogits);
        // Expert selection is defined in float32 (and the generic
        // SelectExpert contract requires it). Keep only the narrow router
        // tensor in float32 while larger activations retain their dtype.
        ToDataType(routerLogits, DataType::FLOAT32);
#ifdef USE_CUDA
        // Shared and routed experts consume the same normalized input and
        // router result but do not depend on one another. During CUDA graph
        // capture these markers let the generic MoE graph optimizer turn the
        // following serial region into two branches. They are no-ops for
        // eager, CPU and mixed-device execution.
        FastllmCudaGraphMarkParallelFork(deviceLayer);
#endif
        // Keep every shared-branch allocation alive through the routed
        // branch's join marker.  The graph optimizer may execute those two
        // branches concurrently, so allowing the capture memory pool to
        // recycle a shared intermediate into routed-MoE workspace would turn
        // an otherwise valid dependency rewrite into a data race.
        Data sharedGateUp, sharedHidden, sharedGate;
        auto runSharedExpert = [&](Data &sharedInput,
                                   Data &sharedResult) {
            Linear(sharedInput,
                   this->weight[mlp +
                                "shared_expert.gateup_proj.weight"],
                   Data(), sharedGateUp);
            Swiglu(sharedGateUp, sharedHidden);
            Linear(sharedHidden,
                   this->weight[mlp +
                                "shared_expert.down_proj.weight"],
                   Data(), sharedResult);
            Linear(sharedInput,
                   this->weight[mlp + "shared_expert_gate.weight"],
                   Data(), sharedGate);
            SigmoidMulTo(sharedResult, sharedGate);
        };
        runSharedExpert(flattened, sharedOutput);

#ifdef USE_CUDA
        FastllmCudaGraphMarkParallelFirstDone(deviceLayer);
        FastllmCudaGraphMarkParallelSecondBegin(deviceLayer);
#endif
        bool fusedRouterSelection = false;
#ifdef USE_CUDA
        if (routerLogits.dataDevice == DataDevice::CUDA &&
            routerLogits.dataType == DataType::FLOAT32 &&
            !routerLogits.dims.empty() &&
            routerLogits.dims.back() == 512 &&
            this->num_experts_per_tok == 10) {
            FusedSoftmaxSelectExpert(
                routerLogits, expertIndex, expertScore,
                this->num_experts_per_tok, this->norm_topk_prob,
                this->routed_scaling_factor, nullptr);
            fusedRouterSelection = true;
        }
#endif
        if (!fusedRouterSelection) {
            Softmax(routerLogits, routerLogits, -1);
            SelectExpert(routerLogits, expertIndex, expertScore,
                         this->num_experts_per_tok,
                         this->norm_topk_prob,
                         this->routed_scaling_factor, nullptr);
        }
        const std::string outputDevice = SelectDeviceFromMap(
            this->deviceMap, deviceLayer + 1, this->block_cnt);
        bool hybridMoe = false;
#if defined(USE_CUDA) && !defined(USE_ROCM)
        if (Qwen4MtpDraftsPerStep() == 0 &&
            outputDevice.find("cuda") == 0) {
            hybridMoe = FastllmCudaMergeMOEHybrid(flattened, expertIndex, expertScore,
                output, moeWeights.data(), moeWeights.size(), deviceLayer);
        }
#endif
        const bool useMoeCudaCache = hybridMoe || TryApplyMoeCudaCache(
            flattened, expertIndex, expertScore, moeWeights,
            outputDevice, MoeGateSwiglu);
        const std::string routedDevice = useMoeCudaCache
            ? outputDevice : this->SelectMoeDeviceForLayer(deviceLayer);
        const bool writeRoutedDirectly = routedDevice == outputDevice;
        if (!useMoeCudaCache) {
            this->ApplyMoeDeviceMapForLayer(deviceLayer);
        }

        Data routed;
        Data &routedOutput = writeRoutedDirectly ? output : routed;
        Data w1, w2, w3, temporaryInput, temporaryOutput;
        if (!hybridMoe) {
            MergeMOE(flattened, expertIndex, expertScore,
                     moeWeights, moeBiass,
                     w1, w2, w3, temporaryInput, temporaryOutput,
                     1.0f, routedOutput, deviceLayer);
        }
#ifdef USE_CUDA
        FastllmCudaGraphMarkParallelJoin(deviceLayer);
#endif
        routedOutput.Reshape(input.dims);
        sharedOutput.Reshape(input.dims);

        if (writeRoutedDirectly) {
            ApplyDeviceMap(
                this->deviceMap, deviceLayer + 1, this->block_cnt);
        } else {
            Data routedCopy;
            routedCopy.CopyFrom(routedOutput);
            ApplyDeviceMap(
                this->deviceMap, deviceLayer + 1, this->block_cnt);
            output.CopyFrom(routedCopy);
        }
        Qwen4CastLike(sharedOutput, output);
        AddTo(output, sharedOutput);
    }

    bool Qwen4ExpModel::MtpSupportsGenerationConfig(
            const GenerationConfig &generationConfig) const {
        const bool samplingSupported =
            !generationConfig.IsSimpleGreedy() &&
            generationConfig.top_k > 1 &&
            std::isfinite(generationConfig.temperature) &&
            generationConfig.temperature > 0.0f &&
            std::isfinite(generationConfig.top_p) &&
            generationConfig.top_p > 0.0f &&
            generationConfig.top_p <= 1.0f;
        return HasMtpWeights() &&
            (generationConfig.IsSimpleGreedy() || samplingSupported) &&
            !generationConfig.output_logits &&
            generationConfig.tool_call_allowed_token_ids.empty() &&
            std::fabs(generationConfig.repeat_penalty - 1.0f) <= 1.0e-6f &&
            !generationConfig.tool_call_name_constraint_enabled &&
            !generationConfig.tool_call_parameter_name_constraint_enabled &&
            !generationConfig.tool_call_content_sampling_enabled &&
            Qwen4CudaOnlyDeviceMap(this->deviceMap) &&
            // Mixed placement is supported. Each graph entry independently
            // requires CUDA-resident experts or a usable GPU expert cache.
            Qwen4CudaOrNumaOnlyDeviceMap(this->moeDeviceMap) &&
            Qwen4CudaOrNumaOnlyDeviceMap(this->layeredMoeDeviceMap);
    }

    namespace {
        void Qwen4CopyRuntimeTensor(
                const std::shared_ptr<Data> &source,
                std::shared_ptr<Data> &destination) {
            if (source == nullptr || source->dims.empty()) {
                destination.reset();
                return;
            }
            if (destination == nullptr) {
                destination = std::make_shared<Data>();
            }
            const bool expandedSource =
                !source->expansionDims.empty() &&
                std::all_of(
                    source->expansionDims.begin(),
                    source->expansionDims.end(),
                    [](int dimension) { return dimension > 0; });
            const bool reusableExpansion =
                expandedSource && destination->expansionBytes > 0 &&
                destination->dataType == source->dataType &&
                destination->dataDevice == source->dataDevice &&
                destination->dataDeviceIds == source->dataDeviceIds &&
                destination->expansionDims == source->expansionDims;
            if (reusableExpansion) {
                // Data::CopyFrom normally rebuilds whenever logical dims
                // differ. Resize first so an append/truncate cycle keeps the
                // existing cache allocation and its padded strides.
                destination->Resize(source->dims);
            } else if (expandedSource) {
                // Preserve capacity even when the source currently fills its
                // complete expansion. CopyFrom treats dims == expansionDims
                // as a compact tensor, which would force a new allocation on
                // the next truncate.
                destination = std::make_shared<Data>();
                destination->dataType = source->dataType;
                destination->UpdateUnitSize();
                destination->ToDevice(
                    source->dataDevice, source->dataDeviceIds, false);
                destination->Expansion(source->expansionDims);
                destination->Resize(source->dims);
            }
            destination->CopyFrom(*source);
        }

        void Qwen4CopyRuntimeTensorMap(
                const std::map<int, std::shared_ptr<Data>> &source,
                std::map<int, std::shared_ptr<Data>> &destination) {
            for (auto item = destination.begin();
                 item != destination.end();) {
                auto sourceItem = source.find(item->first);
                if (sourceItem == source.end() ||
                    sourceItem->second == nullptr ||
                    sourceItem->second->dims.empty()) {
                    item = destination.erase(item);
                } else {
                    ++item;
                }
            }
            for (const auto &item : source) {
                Qwen4CopyRuntimeTensor(
                    item.second, destination[item.first]);
            }
        }

        bool Qwen4CopyCompactCacheTensor(
                const Data &source, Data &destination,
                bool allowEmpty = false) {
            if (source.dims.empty() || source.Count(0) == 0) {
                destination = Data();
                return allowEmpty;
            }
            if (source.multiDeviceData ||
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
            if (!source.isLinearAttention && source.dims.size() >= 2) {
                // Growing K/V caches may retain spare token capacity. Copy
                // only their logical rows into independently owned storage.
                fastllm::Split(
                    source, 1, 0, source.dims[1], destination);
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
            return (destination.dataDevice == DataDevice::CUDA &&
                    destination.cudaData != nullptr) ||
                   (destination.dataDevice == DataDevice::CPU &&
                    destination.cpuData != nullptr);
        }

        bool Qwen4CopyCompactQsaTensor(
                const Data &source, Data &destination) {
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
            fastllm::Split(source, 0, 0, source.dims[0], destination);
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
        }

        bool Qwen4CopyCompactQsaTensorMap(
                const std::map<int, std::shared_ptr<Data>> &source,
                std::map<int, std::shared_ptr<Data>> &destination) {
            destination.clear();
            for (const auto &item : source) {
                if (item.second == nullptr || item.second->dims.empty() ||
                    item.second->Count(0) == 0) {
                    continue;
                }
                std::shared_ptr<Data> copied = std::make_shared<Data>();
                if (!Qwen4CopyCompactQsaTensor(
                        *item.second, *copied)) {
                    destination.clear();
                    return false;
                }
                destination[item.first] = std::move(copied);
            }
            return true;
        }

        uint64_t Qwen4RuntimeTensorMapBytes(
                const std::map<int, std::shared_ptr<Data>> &tensors) {
            uint64_t bytes = 0;
            for (const auto &item : tensors) {
                if (item.second != nullptr) {
                    bytes += item.second->GetBytes();
                }
            }
            return bytes;
        }

        uint64_t Qwen4FloatVectorMapBytes(
                const std::map<int, std::vector<float>> &values) {
            uint64_t bytes = 0;
            for (const auto &item : values) {
                bytes += (uint64_t)item.second.size() * sizeof(float);
            }
            return bytes;
        }

    }

    bool Qwen4ExpModel::CanCloneMtpPrefixState(
            const MtpRuntimeState &source, int cachedLen) const {
        const int expectedMtpLength = std::max(0, cachedLen - 1);
        const auto cacheLength = [](const Data &cache) {
            return cache.dims.size() > 1 ? cache.dims[1] : 0;
        };
        return cachedLen > 0 && source.hasDeferredTargetHidden &&
            source.deferredPosition >= 0 &&
            source.deferredTargetHidden.dims ==
                std::vector<int>({1, 1, this->hcCount * this->embed_dim}) &&
            cacheLength(source.key) == expectedMtpLength &&
            cacheLength(source.value) == expectedMtpLength &&
            source.proposals.empty() && source.pendingOutputTokens.empty();
    }

    std::shared_ptr<Qwen4ExpModel::MtpRuntimeState>
    Qwen4ExpModel::CloneMtpPrefixState(
            const MtpRuntimeState &source, int cachedLen) const {
        if (!CanCloneMtpPrefixState(source, cachedLen)) {
            return nullptr;
        }

        std::shared_ptr<MtpRuntimeState> cloned =
            std::make_shared<MtpRuntimeState>();
        if (!Qwen4CopyCompactCacheTensor(
                source.key, cloned->key, true) ||
            !Qwen4CopyCompactCacheTensor(
                source.value, cloned->value, true)) {
            return nullptr;
        }
        cloned->deferredTargetHidden.CopyFrom(
            source.deferredTargetHidden);
        cloned->deferredPosition = source.deferredPosition;
        cloned->hasDeferredTargetHidden = true;

        const RequestState &sourceAttention = source.attentionState;
        RequestState &clonedAttention = cloned->attentionState;
        clonedAttention.indexerRawKeys = sourceAttention.indexerRawKeys;
        clonedAttention.indexerPositions = sourceAttention.indexerPositions;
        clonedAttention.indexerBlockKeys = sourceAttention.indexerBlockKeys;
        clonedAttention.geometricCacheGrowthReadyLayers =
            sourceAttention.geometricCacheGrowthReadyLayers;
        if (!Qwen4CopyCompactQsaTensorMap(
                sourceAttention.indexerTailKeyTensors,
                clonedAttention.indexerTailKeyTensors) ||
            !Qwen4CopyCompactQsaTensorMap(
                sourceAttention.indexerTailPositionTensors,
                clonedAttention.indexerTailPositionTensors) ||
            !Qwen4CopyCompactQsaTensorMap(
                sourceAttention.indexerBlockKeyTensors,
                clonedAttention.indexerBlockKeyTensors)) {
            return nullptr;
        }
        return cloned;
    }

    void Qwen4ExpModel::CaptureRequestRuntimeCheckpoint(
            RequestState &state,
            const std::map<int, int> &qsaLengths,
            RequestRuntimeCheckpoint &checkpoint) {
        checkpoint.previousToken1 = state.previousToken1;
        checkpoint.previousToken2 = state.previousToken2;
        // CUDA PLE restores from the two device histories below.  Its host
        // image is deliberately lazy and can be rematerialized at a snapshot
        // or device-fallback boundary.
        checkpoint.convHistory.clear();
        checkpoint.pleConvHistoryIndex = state.pleConvHistoryIndex;
        Qwen4CopyRuntimeTensor(
            state.pleConvHistoryTensors[0],
            checkpoint.pleConvHistoryTensors[0]);
        Qwen4CopyRuntimeTensor(
            state.pleConvHistoryTensors[1],
            checkpoint.pleConvHistoryTensors[1]);
        // CUDA-backed speculative checkpoints normally keep the incomplete
        // QSA group in the device tail and the completed prefix in the pinned
        // mirror. Recording the logical length is therefore sufficient;
        // copying every history to std::vector here would synchronize all
        // full-attention layers and move tens of MiB per verifier round.
        checkpoint.indexerRawKeys.clear();
        checkpoint.indexerPositions.clear();
        checkpoint.indexerBlockKeys.clear();
        checkpoint.indexerLengths = qsaLengths;
        Qwen4CopyRuntimeTensorMap(
            state.indexerTailKeyTensors,
            checkpoint.indexerTailKeyTensors);
        Qwen4CopyRuntimeTensorMap(
            state.indexerTailPositionTensors,
            checkpoint.indexerTailPositionTensors);
        // A short prompt can take the host QSA fallback before mixed MTP
        // switches its verifier to CUDA. Preserve only that fallback's
        // incomplete group (at most compression_ratio - 1 rows), so rollback
        // remains exact without copying the completed history.
        for (const auto &item : qsaLengths) {
            const int layer = item.first;
            const int length = item.second;
            const int tail = length % this->indexerCompressRatio;
            if (tail == 0) {
                continue;
            }
            const auto tailKeys =
                checkpoint.indexerTailKeyTensors.find(layer);
            const auto tailPositions =
                checkpoint.indexerTailPositionTensors.find(layer);
            const bool deviceTailReady =
                tailKeys != checkpoint.indexerTailKeyTensors.end() &&
                tailKeys->second != nullptr &&
                tailKeys->second->dims ==
                    std::vector<int>({tail, this->indexerHeadDim}) &&
                tailPositions !=
                    checkpoint.indexerTailPositionTensors.end() &&
                tailPositions->second != nullptr &&
                tailPositions->second->dims ==
                    std::vector<int>({tail});
            if (deviceTailReady) {
                continue;
            }

            auto rawKeys = state.indexerRawKeys.find(layer);
            auto rawPositions = state.indexerPositions.find(layer);
            const size_t requiredKeys =
                (size_t)length * this->indexerHeadDim;
            if (rawKeys == state.indexerRawKeys.end() ||
                rawKeys->second.size() < requiredKeys ||
                rawPositions == state.indexerPositions.end() ||
                rawPositions->second.size() < (size_t)length) {
                MaterializeQsaHostHistory(layer, length, state);
                rawKeys = state.indexerRawKeys.find(layer);
                rawPositions = state.indexerPositions.find(layer);
            }
            AssertInFastLLM(
                rawKeys != state.indexerRawKeys.end() &&
                rawKeys->second.size() >= requiredKeys &&
                rawPositions != state.indexerPositions.end() &&
                rawPositions->second.size() >= (size_t)length,
                "Qwen4-Exp MTP cannot checkpoint its host QSA tail.");
            const size_t keyBegin =
                (size_t)(length - tail) * this->indexerHeadDim;
            std::shared_ptr<Data> fallbackTailKeys =
                std::make_shared<Data>(
                    DataType::FLOAT32,
                    std::vector<int>({tail, this->indexerHeadDim}),
                    std::vector<float>(
                        rawKeys->second.begin() + keyBegin,
                        rawKeys->second.begin() + requiredKeys));
            std::shared_ptr<Data> fallbackTailPositions =
                std::make_shared<Data>(
                    DataType::FLOAT32,
                    std::vector<int>({tail}),
                    std::vector<float>(
                        rawPositions->second.begin() + length - tail,
                        rawPositions->second.begin() + length));
            checkpoint.indexerTailKeyTensors[layer] =
                std::move(fallbackTailKeys);
            checkpoint.indexerTailPositionTensors[layer] =
                std::move(fallbackTailPositions);
        }
        // Compressed QSA rows are append-only. Keep the storage identity and
        // restore its logical row count instead of copying every historical
        // block into the checkpoint. Newly appended rows never overwrite the
        // committed prefix.
        checkpoint.indexerBlockKeyTensors =
            state.indexerBlockKeyTensors;
        checkpoint.geometricCacheGrowthReadyLayers =
            state.geometricCacheGrowthReadyLayers;
        checkpoint.processedTokens = state.processedTokens;
    }

    void Qwen4ExpModel::RestoreRequestRuntimeCheckpoint(
            RequestState &state,
            const RequestRuntimeCheckpoint &checkpoint,
            bool restoreQsa) {
#ifdef USE_CUDA
        if (restoreQsa) {
            // A speculative draft advances this mirror by only a few rows
            // before rolling back to its committed checkpoint. Keep the
            // pinned allocation and restore its logical prefix in place;
            // destroying it here made every verifier round pay one large
            // cudaFreeHost/cudaHostAlloc pair.
            for (auto transfer = state.indexerHostMirrorTransfers.begin();
                 transfer != state.indexerHostMirrorTransfers.end();) {
                const auto length = checkpoint.indexerLengths.find(
                    transfer->first);
                if (transfer->second == nullptr ||
                    length == checkpoint.indexerLengths.end()) {
                    transfer = state.indexerHostMirrorTransfers.erase(
                        transfer);
                    continue;
                }
                const int completedRows =
                    length->second /
                    this->indexerCompressRatio *
                    this->indexerCompressRatio;
                transfer->second->Rollback(completedRows);
                ++transfer;
            }
        }
#endif
        state.previousToken1 = checkpoint.previousToken1;
        state.previousToken2 = checkpoint.previousToken2;
        state.convHistory = checkpoint.convHistory;
        state.pleConvHistoryIndex = checkpoint.pleConvHistoryIndex;
        Qwen4CopyRuntimeTensor(
            checkpoint.pleConvHistoryTensors[0],
            state.pleConvHistoryTensors[0]);
        Qwen4CopyRuntimeTensor(
            checkpoint.pleConvHistoryTensors[1],
            state.pleConvHistoryTensors[1]);
        if (restoreQsa) {
            state.indexerRawKeys = checkpoint.indexerRawKeys;
            state.indexerPositions = checkpoint.indexerPositions;
            state.indexerBlockKeys = checkpoint.indexerBlockKeys;
            Qwen4CopyRuntimeTensorMap(
                checkpoint.indexerTailKeyTensors,
                state.indexerTailKeyTensors);
            Qwen4CopyRuntimeTensorMap(
                checkpoint.indexerTailPositionTensors,
                state.indexerTailPositionTensors);
            state.indexerBlockKeyTensors =
                checkpoint.indexerBlockKeyTensors;
            for (const auto &item : checkpoint.indexerLengths) {
                auto blocks = state.indexerBlockKeyTensors.find(
                    item.first);
                if (blocks != state.indexerBlockKeyTensors.end() &&
                    blocks->second != nullptr) {
                    blocks->second->Resize({
                        item.second / this->indexerCompressRatio,
                        this->indexerHeadDim});
                }
            }
            state.geometricCacheGrowthReadyLayers =
                checkpoint.geometricCacheGrowthReadyLayers;
        }
        state.processedTokens = checkpoint.processedTokens;
    }

    void Qwen4ExpModel::CaptureTargetRuntimeCheckpoint(
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            RequestState &state,
            TargetRuntimeCheckpoint &checkpoint,
            bool synchronize,
            bool captureLinearRecurrent) {
        checkpoint.keyLengths.assign(this->block_cnt, 0);
        checkpoint.valueLengths.assign(this->block_cnt, 0);
        checkpoint.linearFirst.resize(this->block_cnt);
        checkpoint.linearSecond.resize(this->block_cnt);
#ifdef USE_CUDA
        std::vector<void *> checkpointDestinations;
        std::vector<const void *> checkpointSources;
        std::vector<size_t> checkpointSizes;
        auto captureLinearState = [&](Data &destination,
                                      const Data &source) {
            const bool reusable =
                destination.dataDevice == DataDevice::CUDA &&
                source.dataDevice == DataDevice::CUDA &&
                destination.cudaData != nullptr &&
                source.cudaData != nullptr &&
                destination.dataType == source.dataType &&
                destination.dims == source.dims &&
                destination.GetBytes() == source.GetBytes() &&
                destination.dataDeviceIds == source.dataDeviceIds;
            if (!reusable) {
                destination.CopyFrom(source);
                return;
            }
            destination.isKVCache = source.isKVCache;
            destination.isLinearAttention = source.isLinearAttention;
            destination.isLinearAttentionTransposed =
                source.isLinearAttentionTransposed;
            destination.cacheUid = source.cacheUid;
            destination.strides = source.strides;
            destination.expansionDims = source.expansionDims;
            destination.expansionSize = source.expansionSize;
            destination.expansionBytes = source.expansionBytes;
            checkpointDestinations.push_back(destination.cudaData);
            checkpointSources.push_back(source.cudaData);
            checkpointSizes.push_back(source.GetBytes());
        };
#endif
        std::map<int, int> qsaLengths;
        for (int layer = 0; layer < this->block_cnt; layer++) {
            Data &first = pastKeyValues[layer].first;
            Data &second = pastKeyValues[layer].second;
            if (this->IsLinearAttentionLayer(layer)) {
#ifdef USE_CUDA
                captureLinearState(checkpoint.linearFirst[layer], first);
                if (captureLinearRecurrent) {
                    captureLinearState(
                        checkpoint.linearSecond[layer], second);
                }
#else
                checkpoint.linearFirst[layer].CopyFrom(first);
                if (captureLinearRecurrent) {
                    checkpoint.linearSecond[layer].CopyFrom(second);
                }
#endif
                continue;
            }
            checkpoint.keyLengths[layer] =
                first.dims.size() > 1 ? first.dims[1] : 0;
            checkpoint.valueLengths[layer] =
                second.dims.size() > 1 ? second.dims[1] : 0;
            qsaLengths[layer] = checkpoint.keyLengths[layer];
        }
#ifdef USE_CUDA
        if (!checkpointDestinations.empty()) {
            const bool copied =
                FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(
                    checkpointDestinations.data(),
                    checkpointSources.data(), checkpointSizes.data(),
                    (int)checkpointDestinations.size());
            AssertInFastLLM(
                copied,
                "Qwen4-Exp failed to batch its MTP linear checkpoints.");
            if (synchronize) {
                // ForwardTarget may dispatch a later layer from another host
                // worker and therefore another per-thread CUDA stream.
                // Publish snapshots before recurrent state can mutate.
                FastllmCudaSyncCurrentThreadStream();
            }
        }
#endif
        CaptureRequestRuntimeCheckpoint(
            state, qsaLengths, checkpoint.request);
    }

    void Qwen4ExpModel::CommitTargetRecurrentState(
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            const TargetRuntimeCheckpoint &checkpoint) {
#ifdef USE_CUDA
        std::vector<void *> destinations;
        std::vector<const void *> sources;
        std::vector<size_t> sizes;
#endif
        for (int layer = 0; layer < this->block_cnt; layer++) {
            if (!this->IsLinearAttentionLayer(layer)) {
                continue;
            }
            AssertInFastLLM(
                layer < (int)checkpoint.linearSecond.size() &&
                !checkpoint.linearSecond[layer].dims.empty(),
                "Qwen4-Exp MTP out-of-place recurrent state is incomplete.");
            Data &destination = pastKeyValues[layer].second;
            const Data &source = checkpoint.linearSecond[layer];
#ifdef USE_CUDA
            const bool reusable =
                destination.dataDevice == DataDevice::CUDA &&
                source.dataDevice == DataDevice::CUDA &&
                destination.cudaData != nullptr &&
                source.cudaData != nullptr &&
                destination.dataType == source.dataType &&
                destination.dims == source.dims &&
                destination.GetBytes() == source.GetBytes() &&
                destination.dataDeviceIds == source.dataDeviceIds;
            if (reusable) {
                destinations.push_back(destination.cudaData);
                sources.push_back(source.cudaData);
                sizes.push_back(source.GetBytes());
                continue;
            }
#endif
            destination.CopyFrom(source);
        }
#ifdef USE_CUDA
        if (!destinations.empty()) {
            const bool copied =
                FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(
                    destinations.data(), sources.data(), sizes.data(),
                    (int)destinations.size());
            AssertInFastLLM(
                copied,
                "Qwen4-Exp failed to commit its MTP recurrent state.");
        }
#endif
    }

    void Qwen4ExpModel::CommitTargetVerificationPrefix(
            const Data &candidateIds,
            const Data &candidatePositionIds,
            const std::vector<int> &candidateTokens,
            int committedInputs,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            RequestState &state,
            const TargetRuntimeCheckpoint &checkpoint,
            TargetVerificationCapture &capture) {
        AssertInFastLLM(
            committedInputs > 0 && candidateIds.dims.size() == 2 &&
            candidatePositionIds.dims.size() == 2 &&
            committedInputs <= candidateIds.dims[1] &&
            committedInputs <= candidatePositionIds.dims[1],
            "Qwen4-Exp MTP received an invalid committed prefix.");

        Qwen4MtpGdnModeGuard mtpGdnMode(true);
        std::vector<int> linearReplayLayers;
        const bool recurrentStatePreserved =
            capture.runtimeCheckpoint == &checkpoint;
        std::vector<const Data *> linearStateCheckpoints(
            this->block_cnt, nullptr);
        for (int layer = 0; layer < this->block_cnt; layer++) {
            Data &first = pastKeyValues[layer].first;
            Data &second = pastKeyValues[layer].second;
            if (!this->IsLinearAttentionLayer(layer)) {
                AssertInFastLLM(
                    first.dims.size() > 1 && second.dims.size() > 1 &&
                    first.dims[1] >= checkpoint.keyLengths[layer] +
                        committedInputs &&
                    second.dims[1] >= checkpoint.valueLengths[layer] +
                        committedInputs,
                    "Qwen4-Exp MTP full-attention cache is incomplete.");
                std::vector<int> keyDims = first.dims;
                std::vector<int> valueDims = second.dims;
                keyDims[1] = checkpoint.keyLengths[layer] + committedInputs;
                valueDims[1] =
                    checkpoint.valueLengths[layer] + committedInputs;
                first.Resize(keyDims);
                second.Resize(valueDims);
                continue;
            }

            AssertInFastLLM(
                layer < (int)checkpoint.linearFirst.size() &&
                layer < (int)checkpoint.linearSecond.size() &&
                layer < (int)capture.linearConvInputs.size() &&
                layer < (int)capture.linearAlphas.size() &&
                layer < (int)capture.linearBetas.size(),
                "Qwen4-Exp MTP linear replay metadata is incomplete.");
            const Data *linearStateCheckpoint = recurrentStatePreserved
                ? &second : &checkpoint.linearSecond[layer];
            AssertInFastLLM(
                !checkpoint.linearFirst[layer].dims.empty() &&
                linearStateCheckpoint != nullptr &&
                !linearStateCheckpoint->dims.empty() &&
                capture.linearConvInputs[layer].dims.size() == 3 &&
                capture.linearConvInputs[layer].dims[1] >= committedInputs &&
                capture.linearAlphas[layer].dims.size() == 3 &&
                capture.linearAlphas[layer].dims[1] >= committedInputs &&
                capture.linearBetas[layer].dims.size() == 3 &&
                capture.linearBetas[layer].dims[1] >= committedInputs,
                "Qwen4-Exp MTP linear-attention replay capture is incomplete.");
            linearStateCheckpoints[layer] = linearStateCheckpoint;
            linearReplayLayers.push_back(layer);
        }

        int cudaReplayedLayerCount = 0;
#ifdef USE_CUDA
        // A rejected speculative suffix only needs the committed recurrent
        // states; its per-layer GDN output is discarded.  On a homogeneous
        // CUDA placement, restore and replay all independent layers with four
        // launches.
        // Mixed placement and unsupported geometries retain the standard
        // operation path below.
        bool cudaReplayCandidate =
            !linearReplayLayers.empty() && !GetKVCacheInCPU() &&
            this->head_k_dim == 128 && this->head_v_dim == 128 &&
            this->num_k_heads > 0 && this->num_v_heads > 0 &&
            this->num_v_heads % this->num_k_heads == 0;
        int replayDevice = -1;
        DataType replayInputType = DataType::FLOAT32;
        DataType replayGateType = DataType::FLOAT32;
        bool replayTypesReady = false;
        std::vector<FastllmCudaQwen4LinearReplayItem> replayItems;
        if (cudaReplayCandidate) {
            capture.linearReplayOutputs.resize(this->block_cnt);
            capture.linearReplayCoreOutputs.resize(this->block_cnt);
            replayItems.reserve(linearReplayLayers.size());
            const int currentDevice = FastllmCudaGetDevice();
            auto sameReplayDevice = [&](const Data &data) {
                if (data.dataDevice != DataDevice::CUDA ||
                    data.cudaData == nullptr) {
                    return false;
                }
                const int device = data.dataDeviceIds.empty()
                    ? currentDevice : data.dataDeviceIds[0];
                if (replayDevice < 0) {
                    replayDevice = device;
                }
                return device == replayDevice;
            };
            for (int layer : linearReplayLayers) {
                const std::string linear = languagePrefix + "layers." +
                    std::to_string(layer) + ".linear_attn.";
                Data &convolved = capture.linearReplayOutputs[layer];
                Data &coreOutput =
                    capture.linearReplayCoreOutputs[layer];
                const Data &convInput = capture.linearConvInputs[layer];
                const Data &alpha = capture.linearAlphas[layer];
                const Data &beta = capture.linearBetas[layer];
                Data &first = pastKeyValues[layer].first;
                Data &second = pastKeyValues[layer].second;
                Data &convWeight =
                    this->weight[linear + "conv1d.weight"];
                Data &aLog = this->weight[linear + "A_log"];
                Data &dtBias = this->weight[linear + "dt_bias"];
                cudaReplayCandidate = cudaReplayCandidate &&
                    sameReplayDevice(convInput) &&
                    sameReplayDevice(alpha) && sameReplayDevice(beta) &&
                    sameReplayDevice(first) && sameReplayDevice(second) &&
                    sameReplayDevice(checkpoint.linearFirst[layer]) &&
                    sameReplayDevice(*linearStateCheckpoints[layer]) &&
                    sameReplayDevice(convWeight) &&
                    sameReplayDevice(aLog) &&
                    sameReplayDevice(dtBias);
                if (!replayTypesReady) {
                    replayInputType = convInput.dataType;
                    replayGateType = alpha.dataType;
                    replayTypesReady = true;
                } else {
                    cudaReplayCandidate = cudaReplayCandidate &&
                        convInput.dataType == replayInputType &&
                        alpha.dataType == replayGateType;
                }
                replayItems.push_back({
                    &convInput,
                    &convWeight,
                    &checkpoint.linearFirst[layer],
                    &first,
                    &convolved,
                    &alpha,
                    &beta,
                    &aLog,
                    &dtBias,
                    linearStateCheckpoints[layer],
                    &second,
                    &coreOutput
                });
            }
        }
        if (cudaReplayCandidate) {
            const int qkvChannels =
                2 * this->num_k_heads * this->head_k_dim +
                this->num_v_heads * this->head_v_dim;
            FastllmCudaSetDevice(replayDevice);
            for (auto &item : replayItems) {
                Qwen4PrepareCudaWorkspaceOnCurrentDevice(
                    *item.convolved, DataType::FLOAT32,
                    {1, committedInputs, qkvChannels}, replayDevice);
                Qwen4PrepareCudaWorkspaceOnCurrentDevice(
                    *item.coreOutput, DataType::FLOAT32,
                    {1, committedInputs,
                     this->num_v_heads * this->head_v_dim},
                    replayDevice);
            }
            const bool replaySucceeded =
                FastllmCudaQwen4ReplayLinearAttentionBatch(
                    replayItems, capture.linearReplayPointerWorkspace,
                    capture.linearReplayPointerCache,
                    committedInputs, this->linearConvKernel, true,
                    this->num_k_heads, this->num_v_heads,
                    this->head_k_dim, this->head_v_dim, 1e-6f);
            if (replaySucceeded) {
                cudaReplayedLayerCount = (int)replayItems.size();
            }
        }
#endif
        for (int replayIndex = cudaReplayedLayerCount;
             replayIndex < (int)linearReplayLayers.size(); replayIndex++) {
                const int layer = linearReplayLayers[replayIndex];
                pastKeyValues[layer].first.CopyFrom(
                    checkpoint.linearFirst[layer]);
                if (linearStateCheckpoints[layer] !=
                    &pastKeyValues[layer].second) {
                    pastKeyValues[layer].second.CopyFrom(
                        *linearStateCheckpoints[layer]);
                }
                Data convInput, alpha, beta;
                Split(capture.linearConvInputs[layer], 1, 0,
                      committedInputs, convInput);
                Split(capture.linearAlphas[layer], 1, 0,
                      committedInputs, alpha);
                Split(capture.linearBetas[layer], 1, 0,
                      committedInputs, beta);
                const std::string linear = languagePrefix + "layers." +
                    std::to_string(layer) + ".linear_attn.";
                Data convolved, unusedCore;
                fastllm::CausalDepthwiseConv1DPrefill(
                    convInput, this->weight[linear + "conv1d.weight"],
                    pastKeyValues[layer].first,
                    this->linearConvKernel, true, convolved);
                fastllm::GatedDeltaRuleSequence(
                    convolved, alpha, beta,
                    this->weight[linear + "A_log"],
                    this->weight[linear + "dt_bias"],
                    pastKeyValues[layer].second,
                    this->num_k_heads, this->num_v_heads,
                    this->head_k_dim, this->head_v_dim, 1e-6f,
                    unusedCore);
        }
        for (int layer : linearReplayLayers) {
            Data &first = pastKeyValues[layer].first;
            Data &second = pastKeyValues[layer].second;
            first.isKVCache = true;
            first.isLinearAttention = true;
            second.isKVCache = true;
            second.isLinearAttention = true;
            first.isLinearAttentionTransposed =
                checkpoint.linearFirst[layer].isLinearAttentionTransposed;
            second.isLinearAttentionTransposed =
                checkpoint.linearSecond[layer].isLinearAttentionTransposed;
            first.cacheUid = checkpoint.linearFirst[layer].cacheUid;
            second.cacheUid = checkpoint.linearSecond[layer].cacheUid;
            first.expansionDims = first.dims;
            second.expansionDims =
                checkpoint.linearSecond[layer].expansionDims;
        }
        const int previousQsaLength =
            checkpoint.keyLengths[this->kvCacheId];
        const int committedQsaLength =
            previousQsaLength + committedInputs;
        const int candidateQsaLength =
            previousQsaLength + candidateIds.dims[1];
        const bool rollbackQsaBlock =
            committedQsaLength / this->indexerCompressRatio !=
            candidateQsaLength / this->indexerCompressRatio;
        if (rollbackQsaBlock) {
#ifdef USE_CUDA
            ForceDeviceSync();
            for (auto &item : state.indexerHostMirrorTransfers) {
                if (item.second != nullptr) {
                    item.second->MarkDeviceSynchronized();
                }
            }
#endif
            // Candidate-completed blocks may contain a rejected row. The
            // mirrors are reset in place below after this one synchronization;
            // retaining their pinned capacity avoids one free/allocation pair
            // per full-attention layer on every rejection.
        }
        RestoreRequestRuntimeCheckpoint(
            state, checkpoint.request, false);
        Data committedIds, committedPositions, committedPleInput;
        Split(candidateIds, 1, 0, committedInputs, committedIds);
        Split(candidatePositionIds, 1, 0, committedInputs,
              committedPositions);
        Split(capture.pleInput, 1, 0, committedInputs,
              committedPleInput);
        Data unusedPle;
        RunPLE(committedPleInput, committedIds, state, unusedPle,
               &candidateTokens);

        for (int layer = 0; layer < this->block_cnt; layer++) {
            if (this->IsLinearAttentionLayer(layer)) {
                continue;
            }
            if (!rollbackQsaBlock) {
                const int desiredLength =
                    checkpoint.keyLengths[layer] + committedInputs;
                const int desiredTail =
                    desiredLength % this->indexerCompressRatio;
                const int desiredBlocks =
                    desiredLength / this->indexerCompressRatio;
                std::shared_ptr<Data> &tailKeys =
                    state.indexerTailKeyTensors[layer];
                std::shared_ptr<Data> &tailPositions =
                    state.indexerTailPositionTensors[layer];
                std::shared_ptr<Data> &blockKeys =
                    state.indexerBlockKeyTensors[layer];
                AssertInFastLLM(
                    tailKeys != nullptr && tailPositions != nullptr &&
                    (desiredBlocks == 0 || blockKeys != nullptr),
                    "Qwen4-Exp MTP cannot truncate its QSA cache.");
                tailKeys->Resize(
                    {desiredTail, this->indexerHeadDim});
                tailPositions->Resize({desiredTail});
                if (blockKeys != nullptr) {
                    blockKeys->Resize(
                        {desiredBlocks, this->indexerHeadDim});
                }
                state.geometricCacheGrowthReadyLayers.insert(layer);
                continue;
            }
            auto captured = capture.qsaRawKeys.find(layer);
            AssertInFastLLM(
                captured != capture.qsaRawKeys.end() &&
                !captured->second.dims.empty(),
                "Qwen4-Exp MTP QSA capture is incomplete.");
            Data rawKeyPrefix, qsaPositions;
            Split(captured->second, 0, 0, committedInputs,
                  rawKeyPrefix);
            qsaPositions.CopyFrom(committedPositions);
#ifdef USE_CUDA
            if (rawKeyPrefix.dataDevice == DataDevice::CUDA) {
                qsaPositions.ToDevice(
                    DataDevice::CUDA, rawKeyPrefix.dataDeviceIds);
            }
#endif

            const int ratio = this->indexerCompressRatio;
            const int previousLength = checkpoint.keyLengths[layer];
            const int oldTail = previousLength % ratio;
            const int previousBlocks = previousLength / ratio;
            const int combinedCount = oldTail + committedInputs;
            const int newBlocks = combinedCount / ratio;
            const int desiredTail = combinedCount % ratio;
            const int desiredBlocks = previousBlocks + newBlocks;

            qsaPositions.Reshape({committedInputs});
            Data hostBaseKeys, hostBasePositions;
            const Data *baseKeyData = nullptr;
            const Data *basePositionData = nullptr;
            if (oldTail > 0) {
                auto baseKeys =
                    checkpoint.request.indexerTailKeyTensors.find(layer);
                auto basePositions = checkpoint.request.
                    indexerTailPositionTensors.find(layer);
                if (baseKeys != checkpoint.request.
                        indexerTailKeyTensors.end() &&
                    baseKeys->second != nullptr &&
                    baseKeys->second->dims ==
                        std::vector<int>({oldTail,
                                          this->indexerHeadDim}) &&
                    basePositions != checkpoint.request.
                        indexerTailPositionTensors.end() &&
                    basePositions->second != nullptr &&
                    basePositions->second->dims ==
                        std::vector<int>({oldTail})) {
                    baseKeyData = baseKeys->second.get();
                    basePositionData = basePositions->second.get();
                } else {
                    const auto hostKeysIt = checkpoint.request.
                        indexerRawKeys.find(layer);
                    const auto hostPositionsIt = checkpoint.request.
                        indexerPositions.find(layer);
                    AssertInFastLLM(
                        hostKeysIt != checkpoint.request.indexerRawKeys.end() &&
                        hostPositionsIt != checkpoint.request.
                            indexerPositions.end(),
                        "Qwen4-Exp MTP host QSA tail is unavailable.");
                    const std::vector<float> &hostKeys = hostKeysIt->second;
                    const std::vector<float> &hostPositions =
                        hostPositionsIt->second;
                    const size_t keyOffset =
                        (size_t)(previousLength - oldTail) *
                            this->indexerHeadDim;
                    AssertInFastLLM(
                        hostKeys.size() >= keyOffset +
                            (size_t)oldTail * this->indexerHeadDim &&
                        hostPositions.size() >=
                            (size_t)previousLength,
                        "Qwen4-Exp MTP host QSA tail is incomplete.");
                    Data hostKeyValues(
                        DataType::FLOAT32,
                        {oldTail, this->indexerHeadDim},
                        std::vector<float>(
                            hostKeys.begin() + keyOffset,
                            hostKeys.begin() + keyOffset +
                                (size_t)oldTail *
                                    this->indexerHeadDim));
                    Data hostPositionValues(
                        DataType::FLOAT32, {oldTail},
                        std::vector<float>(
                            hostPositions.begin() +
                                previousLength - oldTail,
                            hostPositions.begin() + previousLength));
                    hostBaseKeys.CopyFrom(hostKeyValues);
                    hostBasePositions.CopyFrom(hostPositionValues);
#ifdef USE_CUDA
                    if (rawKeyPrefix.dataDevice == DataDevice::CUDA) {
                        hostBaseKeys.ToDevice(
                            DataDevice::CUDA,
                            rawKeyPrefix.dataDeviceIds);
                        hostBasePositions.ToDevice(
                            DataDevice::CUDA,
                            rawKeyPrefix.dataDeviceIds);
                    }
#endif
                    baseKeyData = &hostBaseKeys;
                    basePositionData = &hostBasePositions;
                }
            }
#ifdef USE_CUDA
            if (baseKeyData != nullptr &&
                rawKeyPrefix.dataDevice == DataDevice::CUDA) {
                const int currentDevice = FastllmCudaGetDevice();
                const int targetDevice = rawKeyPrefix.dataDeviceIds.empty()
                    ? currentDevice : rawKeyPrefix.dataDeviceIds[0];
                const int baseDevice = baseKeyData->dataDeviceIds.empty()
                    ? currentDevice : baseKeyData->dataDeviceIds[0];
                if (baseKeyData->dataDevice != DataDevice::CUDA ||
                    baseDevice != targetDevice) {
                    hostBaseKeys.CopyFrom(*baseKeyData);
                    hostBasePositions.CopyFrom(*basePositionData);
                    hostBaseKeys.ToDevice(
                        DataDevice::CUDA,
                        rawKeyPrefix.dataDeviceIds);
                    hostBasePositions.ToDevice(
                        DataDevice::CUDA,
                        rawKeyPrefix.dataDeviceIds);
                    baseKeyData = &hostBaseKeys;
                    basePositionData = &hostBasePositions;
                }
            }
#endif

            std::shared_ptr<Data> &blockCache =
                state.indexerBlockKeyTensors[layer];
            AssertInFastLLM(
                desiredBlocks == 0 ||
                (blockCache != nullptr &&
                 blockCache->dims.size() == 2 &&
                 blockCache->dims[0] >= desiredBlocks &&
                 blockCache->dims[1] == this->indexerHeadDim),
                "Qwen4-Exp MTP QSA block prefix is incomplete.");
            if (blockCache != nullptr) {
                blockCache->Resize(
                    {desiredBlocks, this->indexerHeadDim});
            }

            auto appendRange = [&](const Data &source, int start, int end,
                                   std::shared_ptr<Data> &destination,
                                   int rowWidth) {
                if (end <= start) {
                    return;
                }
                Data view;
                view.FakeFrom(
                    source,
                    (size_t)start * rowWidth * sizeof(float));
                view.Resize(rowWidth == 1 ?
                    std::vector<int>({end - start}) :
                    std::vector<int>({end - start, rowWidth}));
                if (destination == nullptr) {
                    destination = std::make_shared<Data>();
                    destination->CopyFrom(view);
                    destination->Expansion(rowWidth == 1 ?
                        std::vector<int>({ratio}) :
                        std::vector<int>({ratio, rowWidth}));
                    destination->Resize(view.dims);
                } else {
                    CatDirect(*destination, view, 0);
                }
            };
            std::shared_ptr<Data> &tailKeyCache =
                state.indexerTailKeyTensors[layer];
            std::shared_ptr<Data> &tailPositionCache =
                state.indexerTailPositionTensors[layer];
            if (tailKeyCache != nullptr) {
                tailKeyCache->Resize({0, this->indexerHeadDim});
            }
            if (tailPositionCache != nullptr) {
                tailPositionCache->Resize({0});
            }
            const int tailStart = newBlocks * ratio;
            if (tailStart < oldTail) {
                appendRange(
                    *baseKeyData, tailStart, oldTail,
                    tailKeyCache, this->indexerHeadDim);
                appendRange(
                    *basePositionData, tailStart, oldTail,
                    tailPositionCache, 1);
            }
            const int rawTailStart = std::max(0, tailStart - oldTail);
            appendRange(
                rawKeyPrefix, rawTailStart, committedInputs,
                tailKeyCache, this->indexerHeadDim);
            appendRange(
                qsaPositions, rawTailStart, committedInputs,
                tailPositionCache, 1);
            AssertInFastLLM(
                (desiredTail == 0 ||
                 (tailKeyCache != nullptr &&
                  tailPositionCache != nullptr)) &&
                (tailKeyCache == nullptr ||
                 tailKeyCache->dims == std::vector<int>({
                     desiredTail, this->indexerHeadDim})) &&
                (tailPositionCache == nullptr ||
                 tailPositionCache->dims ==
                     std::vector<int>({desiredTail})),
                "Qwen4-Exp MTP QSA tail rollback failed.");

            // The pinned mirror below owns the complete raw prefix.  Leave
            // host vectors invalidated so a later snapshot/fallback boundary
            // materializes the newly committed length exactly once.
            state.indexerRawKeys[layer].clear();
            state.indexerPositions[layer].clear();
            state.indexerBlockKeys[layer].clear();
#ifdef USE_CUDA
            std::shared_ptr<QsaHostMirrorTransfer> &transfer =
                state.indexerHostMirrorTransfers[layer];
            if (transfer == nullptr) {
                transfer = std::make_shared<QsaHostMirrorTransfer>();
            }
            transfer->Rollback(previousLength - oldTail);
            if (newBlocks > 0) {
                int queuedRows = previousLength - oldTail;
                const int completeRows = newBlocks * ratio;
                const int baseRows = std::min(oldTail, completeRows);
                if (baseRows > 0) {
                    Data keyView, positionView;
                    keyView.FakeFrom(*baseKeyData, 0);
                    keyView.Resize(
                        {baseRows, this->indexerHeadDim});
                    positionView.FakeFrom(*basePositionData, 0);
                    positionView.Resize({baseRows});
                    transfer->Queue(
                        keyView, positionView, queuedRows,
                        baseRows, this->indexerHeadDim);
                    queuedRows += baseRows;
                }
                const int rawRows = completeRows - baseRows;
                if (rawRows > 0) {
                    Data keyView, positionView;
                    keyView.FakeFrom(rawKeyPrefix, 0);
                    keyView.Resize(
                        {rawRows, this->indexerHeadDim});
                    positionView.FakeFrom(qsaPositions, 0);
                    positionView.Resize({rawRows});
                    transfer->Queue(
                        keyView, positionView, queuedRows,
                        rawRows, this->indexerHeadDim);
                }
            }
#endif
            state.geometricCacheGrowthReadyLayers.insert(layer);
        }
    }

    int Qwen4ExpModel::RunMtpDraft(
            MtpRuntimeState &state,
            const Data &targetHiddenStates,
            const std::vector<int> &inputTokens,
            const std::vector<int> &positions,
            Data *nextMultiHidden,
            bool sampleToken,
            const Data *deviceTokenIds,
            Data *sampledTokenIds,
            Data *sampledTokenValues,
            int sampledTokenOffset) {
        const int sequence = (int)inputTokens.size();
        AssertInFastLLM(
            sequence > 0 && positions.size() == inputTokens.size() &&
            targetHiddenStates.dims.size() == 3 &&
            targetHiddenStates.dims[0] == 1 &&
            targetHiddenStates.dims[1] == sequence &&
            targetHiddenStates.dims[2] == this->hcCount * this->embed_dim,
            "Qwen4-Exp MTP received misaligned target states.");

        std::vector<float> tokenValues(sequence);
        std::vector<float> positionValues(sequence);
        for (int i = 0; i < sequence; i++) {
            tokenValues[i] = (float)inputTokens[i];
            positionValues[i] = (float)positions[i];
        }
        Data hostTokenIds(
            DataType::FLOAT32, {1, sequence}, tokenValues);
        Data tokenIdsView;
        Data *tokenIds = &hostTokenIds;
        if (deviceTokenIds != nullptr) {
            AssertInFastLLM(
                sequence == 1 &&
                deviceTokenIds->dataType == DataType::FLOAT32 &&
                deviceTokenIds->dataDevice == DataDevice::CUDA &&
                deviceTokenIds->cudaData != nullptr &&
                deviceTokenIds->Count(0) >= 1,
                "Qwen4-Exp MTP received an invalid device token.");
            tokenIdsView.FakeFrom(*deviceTokenIds, 0);
            tokenIdsView.Resize({1, 1});
            tokenIdsView.dataDeviceIds = deviceTokenIds->dataDeviceIds;
            tokenIds = &tokenIdsView;
        }
        Data mtpPositionIds(
            DataType::FLOAT32, {1, sequence}, positionValues);

        MtpDraftCudaGraphState::Segment *draftGraphSegment = nullptr;
        std::unique_lock<std::mutex> draftGraphLock;
#ifdef USE_CUDA
        const bool draftGraphEligible =
            GetFastllmEnv().cudaGraph && sequence == 1 && sampleToken &&
            sampledTokenOffset > 0 && deviceTokenIds != nullptr &&
            sampledTokenIds != nullptr && sampledTokenValues != nullptr &&
            targetHiddenStates.dataDevice == DataDevice::CUDA &&
            targetHiddenStates.cudaData != nullptr &&
            sampledTokenIds->dataDevice == DataDevice::CUDA &&
            sampledTokenIds->cudaData != nullptr &&
            sampledTokenValues->dataDevice == DataDevice::CUDA &&
            sampledTokenValues->cudaData != nullptr &&
            std::getenv("FASTLLM_CUDA_SYNC") == nullptr &&
            std::getenv("FASTLLM_PRINT_PROFILE") == nullptr &&
            Qwen4CudaOnlyDeviceMap(this->deviceMap) &&
            Qwen4CudaOnlyDeviceMap(this->moeDeviceMap) &&
            Qwen4CudaOnlyDeviceMap(this->layeredMoeDeviceMap);
        if (draftGraphEligible) {
            const int device = targetHiddenStates.dataDeviceIds.empty()
                ? FastllmCudaGetDevice()
                : targetHiddenStates.dataDeviceIds[0];
            if (device >= 0) {
                if (state.draftGraphState == nullptr) {
                    state.draftGraphState =
                        std::make_shared<MtpDraftCudaGraphState>();
                }
                draftGraphLock = std::unique_lock<std::mutex>(
                    state.draftGraphState->mutex);
                if (state.draftGraphState->device != device) {
                    state.draftGraphState->Reset(device);
                }
                draftGraphSegment =
                    &state.draftGraphState->segments[sampledTokenOffset];
            }
        }
#endif

        Data inputEmbeds;
        Embedding(
            *tokenIds,
            this->weight[languagePrefix + "embed_tokens.weight"],
            inputEmbeds);

        Data normalizedEmbeds, projectedEmbeds;
        RMSNorm(inputEmbeds,
                this->weight["mtp.pre_fc_norm_embedding.weight"],
                this->rms_norm_eps, normalizedEmbeds);
        Linear(normalizedEmbeds,
               this->weight["mtp.fc_embedding.weight"],
               Data(), projectedEmbeds);

        Data normalizedHidden, localProjectedHidden;
        Data &projectedHidden = draftGraphSegment == nullptr
            ? localProjectedHidden : draftGraphSegment->projectedHidden;
        RMSNorm(targetHiddenStates,
                this->weight["mtp.pre_fc_norm_hidden.weight"],
                this->rms_norm_eps, normalizedHidden);
        normalizedHidden.Reshape(
            {1, sequence * this->hcCount, this->embed_dim});
        Linear(normalizedHidden,
               this->weight["mtp.fc_hidden.weight"],
               Data(), projectedHidden);
        projectedHidden.Reshape(
            {1, sequence, this->hcCount, this->embed_dim});
        projectedEmbeds.Reshape(
            {1, sequence, 1, this->embed_dim});
        Qwen4CastLike(projectedEmbeds, projectedHidden);
        RepeatAddTo(
            projectedHidden, projectedEmbeds, 2, this->hcCount);
        projectedHidden.Reshape(
            {1, sequence, this->hcCount * this->embed_dim});

        const std::string layerPrefix = "mtp.layers.0.";
        const std::string attentionHyperPrefix =
            layerPrefix + "attn_hyper_connection.";
        Data attentionNorm, attentionInput, localAttentionInjection;
        Data &attentionInjection = draftGraphSegment == nullptr
            ? localAttentionInjection
            : draftGraphSegment->attentionInjection;
        GroupedRMSNorm(
            projectedHidden,
            this->weight[attentionHyperPrefix + "hc_norm.weight"],
            attentionNorm);
        HyperMixNormalized(
            attentionNorm, attentionHyperPrefix,
            attentionInput, &attentionInjection);

        mtpPositionIds.ToDevice(
            attentionInput.dataDevice,
            attentionInput.dataDeviceIds);
        Data localAttentionOutput;
        Data &attentionOutput = draftGraphSegment == nullptr
            ? localAttentionOutput : draftGraphSegment->attentionOutput;
        const int previousMtpLength =
            state.key.dims.size() > 1 ? state.key.dims[1] : 0;
        Data emptyAttentionMask;
        std::unique_ptr<Data> denseAttentionMask;
        const Data *mtpAttentionMask = &emptyAttentionMask;
        if (sequence > 1) {
            const int mtpKeyLength = previousMtpLength + sequence;
            std::vector<float> mtpMaskValues(
                (size_t)sequence * mtpKeyLength, 0.0f);
            for (int query = 0; query < sequence; query++) {
                for (int key = previousMtpLength + query + 1;
                     key < mtpKeyLength; key++) {
                    // FastLLM's dense Attention mask uses 1 for masked
                    // entries.  A single decode row has no future entries,
                    // so the standard empty-mask path avoids a redundant
                    // allocation and host-to-device transfer.
                    mtpMaskValues[(size_t)query * mtpKeyLength + key] = 1.0f;
                }
            }
            denseAttentionMask = std::make_unique<Data>(
                DataType::FLOAT32,
                std::vector<int>{1, sequence, mtpKeyLength},
                mtpMaskValues);
            mtpAttentionMask = denseAttentionMask.get();
        }
        RunFullAttentionWithPrefix(
            this->block_cnt, layerPrefix + "self_attn.",
            attentionInput, *mtpAttentionMask, mtpPositionIds, true,
            state.key, state.value, state.attentionState,
            attentionOutput);

#ifdef USE_CUDA
        if (draftGraphSegment != nullptr) {
            const int device = state.draftGraphState->device;
            auto graphInputReady = [&](const Data &data) {
                return data.dataDevice == DataDevice::CUDA &&
                    data.cudaData != nullptr && !data.multiDeviceData &&
                    (data.dataDeviceIds.empty() ||
                     data.dataDeviceIds[0] == device);
            };
            if (!graphInputReady(projectedHidden) ||
                !graphInputReady(attentionOutput) ||
                !graphInputReady(attentionInjection)) {
                draftGraphSegment = nullptr;
            }
        }
#endif

        const std::string mlpHyperPrefix =
            layerPrefix + "mlp_hyper_connection.";
        const std::string finalPrefix = "mtp.hyper_connection_mixer.";
        Data afterAttention, mlpNorm, mlpInput, mlpInjection, mlpOutput;
        Data localMultiHidden, finalNorm, sampleHidden, lastHidden;
        Data logits, top, halfHeadInput;
        Data &multiHidden = draftGraphSegment != nullptr
            ? draftGraphSegment->multiHidden
            : (nextMultiHidden != nullptr && sequence == 1
                ? *nextMultiHidden : localMultiHidden);

        auto runPostAttention = [&]() -> bool {
            HyperCombineRMSNorm(
                projectedHidden, attentionOutput, attentionInjection,
                this->weight[mlpHyperPrefix + "hc_norm.weight"],
                afterAttention, mlpNorm);
            HyperMixNormalized(mlpNorm, mlpHyperPrefix,
                               mlpInput, &mlpInjection);
            RunMoEWithPrefix(
                this->block_cnt - 1, layerPrefix + "mlp.",
                this->mtpMoeWeights, this->mtpMoeBiass,
                mlpInput, mlpOutput);
            HyperCombineRMSNorm(
                afterAttention, mlpOutput, mlpInjection,
                this->weight[finalPrefix + "hc_norm.weight"],
                multiHidden, finalNorm);
            HyperMixNormalized(
                finalNorm, finalPrefix, sampleHidden, nullptr);
            if (nextMultiHidden != nullptr && sequence != 1) {
                Split(multiHidden, 1, sequence - 1, sequence,
                      *nextMultiHidden);
            }
            if (!sampleToken) {
                return false;
            }

            if (sequence == 1) {
                lastHidden.FakeFrom(sampleHidden, 0);
                lastHidden.Resize(sampleHidden.dims);
            } else {
                Split(sampleHidden, 1, sequence - 1, sequence,
                      lastHidden);
            }
            PrepareMtpDraftLmHeadWeight();
            Data &draftLmHead = this->mtpDraftLmHeadReady
                ? this->mtpDraftLmHeadWeight
                : this->weight["lm_head.weight"];
            Data *headInput = &lastHidden;
            if (this->mtpDraftLmHeadReady &&
                lastHidden.dataType == DataType::FLOAT32) {
                // The packed FP8 CUDA GEMV is specialized for FP16/BF16
                // activations.  Its generic FP32 fallback materializes the
                // whole 1.27 GB head as FP16 on every call, which is slower
                // than the original head. Casting this one row selects the
                // direct packed-weight kernel.
                ToDataType(
                    lastHidden, halfHeadInput, DataType::FLOAT16);
                headInput = &halfHeadInput;
            }
            Linear(*headInput, draftLmHead, Data(), logits);
            ToDataType(logits, DataType::FLOAT32);
#ifdef USE_CUDA
            if (sampledTokenIds != nullptr &&
                sampledTokenValues != nullptr &&
                sampledTokenOffset >= 0 &&
                logits.dataDevice == DataDevice::CUDA &&
                logits.dataType == DataType::FLOAT32 &&
                logits.cudaData != nullptr &&
                sampledTokenIds->dataDevice == DataDevice::CUDA &&
                sampledTokenIds->dataType == DataType::INT32 &&
                sampledTokenIds->cudaData != nullptr &&
                sampledTokenIds->Count(0) >
                    (uint64_t)sampledTokenOffset &&
                sampledTokenValues->dataDevice == DataDevice::CUDA &&
                sampledTokenValues->dataType == DataType::FLOAT32 &&
                sampledTokenValues->cudaData != nullptr &&
                sampledTokenValues->Count(0) >
                    (uint64_t)sampledTokenOffset) {
                int *sampledId = reinterpret_cast<int *>(
                    sampledTokenIds->cudaData) + sampledTokenOffset;
                float *sampledValue = reinterpret_cast<float *>(
                    sampledTokenValues->cudaData) + sampledTokenOffset;
                if (FastllmCudaGreedySamplingWithFloatOutput(
                        reinterpret_cast<float *>(logits.cudaData),
                        sampledId, sampledValue, 1,
                        logits.dims.back())) {
                    return true;
                }
            }
#endif
            return false;
        };

        bool sampledOnDevice = false;
#ifdef USE_CUDA
        if (draftGraphSegment != nullptr) {
            MtpDraftCudaGraphState::Segment &segment =
                *draftGraphSegment;
            const int device = state.draftGraphState->device;
            std::vector<uint64_t> signature = {
                (uint64_t)(uintptr_t)projectedHidden.cudaData,
                (uint64_t)(uintptr_t)attentionOutput.cudaData,
                (uint64_t)(uintptr_t)attentionInjection.cudaData,
                (uint64_t)(uintptr_t)sampledTokenIds->cudaData,
                (uint64_t)(uintptr_t)sampledTokenValues->cudaData,
                (uint64_t)projectedHidden.dataType,
                projectedHidden.Count(0),
                attentionOutput.Count(0),
                attentionInjection.Count(0)
            };
            if (segment.inputSignature != signature) {
                state.draftGraphState->DestroySegment(segment);
                segment.inputSignature = signature;
            }
            FastllmCudaSetDevice(device);

            if (segment.captured &&
                FastllmCudaGraphLaunch(segment.exec)) {
                sampledOnDevice = true;
            } else {
                if (segment.captured) {
                    state.draftGraphState->DestroySegment(segment);
                    segment.inputSignature = signature;
                    segment.disabled = true;
                    FastllmCudaClearThreadError();
                    FastllmCudaClearGraphError();
                }
                if (segment.disabled) {
                    sampledOnDevice = runPostAttention();
                } else if (!segment.warmed) {
                    sampledOnDevice = runPostAttention();
                    segment.warmed = true;
                    if (!sampledOnDevice) {
                        segment.disabled = true;
                    }
                } else {
                    void *capturedGraph = nullptr;
                    void *capturedExec = nullptr;
                    bool poolActive = false;
                    bool captureActive = false;
                    bool recaptureWithoutParallelMarkers = false;
                    bool captureOk =
                        FastllmCudaGraphPrepareCaptureDevice();
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
                    bool bodySampled = false;
                    if (captureOk) {
                        FastllmCudaMergeMOEClearGraphUnsafeFallbackFlag();
                        const bool markersEnabled =
                            !segment.parallelRegionsUnavailable;
                        const bool previousMarkers =
                            FastllmCudaGraphSetParallelMarkersEnabled(
                                markersEnabled);
                        try {
                            bodySampled = runPostAttention();
                        } catch (...) {
                            bodyThrew = true;
                        }
                        FastllmCudaGraphSetParallelMarkersEnabled(
                            previousMarkers);
                        const bool bodyFailed = bodyThrew ||
                            !bodySampled ||
                            FastllmCudaMergeMOEUsedGraphUnsafeFallback() ||
                            FastllmCudaGetThreadError() ||
                            FastllmCudaGetGraphError() ||
                            FastllmCudaGraphCaptureInvalidated();
                        const bool endOk = FastllmCudaGraphEndCapture(
                            &capturedGraph);
                        captureActive = false;
                        captureOk = !bodyFailed && endOk &&
                            capturedGraph != nullptr;
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
                    if (captureOk &&
                        !segment.parallelRegionsUnavailable) {
                        const int optimized =
                            FastllmCudaGraphOptimizeParallelRegions(
                                capturedGraph);
                        if (optimized ==
                            FASTLLM_CUDA_GRAPH_RECAPTURE_WITHOUT_PARALLEL_MARKERS) {
                            segment.parallelRegionsUnavailable = true;
                            recaptureWithoutParallelMarkers = true;
                            captureOk = false;
                        } else if (optimized < 0) {
                            captureOk = false;
                        }
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
                        if (!recaptureWithoutParallelMarkers) {
                            segment.captureFailures++;
                            if (segment.captureFailures >= 3) {
                                segment.disabled = true;
                            }
                        }
                        FastllmCudaClearThreadError();
                        FastllmCudaClearGraphError();
                        sampledOnDevice = runPostAttention();
                        segment.warmed = true;
                    } else {
                        segment.graph = capturedGraph;
                        segment.exec = capturedExec;
                        segment.captured = true;
                        segment.captureFailures = 0;
                        sampledOnDevice =
                            FastllmCudaGraphLaunch(segment.exec);
                        if (!sampledOnDevice) {
                            state.draftGraphState->DestroySegment(
                                segment);
                            segment.inputSignature = signature;
                            segment.disabled = true;
                            FastllmCudaClearThreadError();
                            FastllmCudaClearGraphError();
                            sampledOnDevice = runPostAttention();
                        }
                    }
                }
            }
            if (nextMultiHidden != nullptr) {
                nextMultiHidden->FreeSpace();
                Qwen4BorrowCudaStorage(*nextMultiHidden, multiHidden);
            }
        } else
#endif
        {
            sampledOnDevice = runPostAttention();
        }

        if (!sampleToken || sampledOnDevice) {
            // CUDA draft sampling stays chained on the request stream. The
            // host materializes the compact proposal ids only after the
            // complete autoregressive chain.
            return -1;
        }
        TopK(logits, top, 1);
        top.ToDevice(DataDevice::CPU);
        return (int)(reinterpret_cast<float *>(top.cpuData)[0] + 1e-3f);
    }

    bool Qwen4ExpModel::TryRunDecodeCudaGraphBackbone(
            int graphStartLayer,
            bool startBeforeAttention,
            const Data &hiddenStates,
            const Data &attentionOutput,
            const Data &attentionInjection,
            const Data &attentionMask,
            const Data &positionIds,
            bool qsaDeviceCompatibleMask,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            RequestState &requestState,
            Data &logits,
            Data *targetMultiHidden,
            TargetVerificationCapture *verificationCapture) {
#ifndef USE_CUDA
        return false;
#else
        const int firstFullAttentionLayer = this->kvCacheId;
        const bool supportedStart =
            (!startBeforeAttention &&
             graphStartLayer == firstFullAttentionLayer) ||
            (startBeforeAttention &&
             graphStartLayer == this->pleLayer &&
             graphStartLayer < firstFullAttentionLayer &&
             this->IsLinearAttentionLayer(graphStartLayer));
        const bool mtpTargetGraph =
            verificationCapture != nullptr &&
            qwen4MtpDecodeEquivalentTarget &&
            hiddenStates.dims.size() == 3 &&
            hiddenStates.dims[1] > 1 &&
            hiddenStates.dims[1] <= this->indexerCompressRatio;
        // Check the supported cache shape before querying availability,
        // which lazily allocates the device cache. Cached verifier rows run
        // entirely on CUDA, so their configured NUMA fallback is graph-safe.
        const bool moeCudaCache =
            hiddenStates.dims.size() == 3 && hiddenStates.dims[1] > 0 &&
            hiddenStates.dims[1] <= FASTLLM_CUDA_MOE_CACHE_MAX_BATCH &&
            (hiddenStates.dims[1] == 1 || mtpTargetGraph) &&
            !this->weights.empty() && !this->weights[0].empty() &&
            MoeCudaCacheAvailable(this->weights[0]);
        const bool moeDeviceMapGraphCompatible =
            (Qwen4CudaOnlyDeviceMap(this->moeDeviceMap) &&
             Qwen4CudaOnlyDeviceMap(this->layeredMoeDeviceMap)) ||
            (moeCudaCache &&
             Qwen4CudaOrNumaOnlyDeviceMap(this->moeDeviceMap) &&
             Qwen4CudaOrNumaOnlyDeviceMap(this->layeredMoeDeviceMap));
        // Host decisions and NUMA execution cannot be captured in the full
        // backbone graph. MTP and prefill retain their existing paths.
        const bool hybridMoe = Qwen4MtpDraftsPerStep() == 0 &&
            !this->weights.empty() && !this->weights[0].empty() &&
            FastllmCudaUseMoeHybrid(this->weights[0].data(), this->weights[0].size());
        if (!GetFastllmEnv().cudaGraph ||
            hybridMoe ||
            !supportedStart ||
            hiddenStates.dims.size() != 3 || hiddenStates.dims[0] != 1 ||
            (hiddenStates.dims[1] != 1 && !mtpTargetGraph) ||
            hiddenStates.dataDevice != DataDevice::CUDA ||
            hiddenStates.cudaData == nullptr ||
            (!startBeforeAttention &&
             (attentionOutput.dataDevice != DataDevice::CUDA ||
              attentionOutput.cudaData == nullptr ||
              attentionInjection.dataDevice != DataDevice::CUDA ||
              attentionInjection.cudaData == nullptr)) ||
            GetKVCacheInCPU() ||
            this->pleLayer >= firstFullAttentionLayer ||
            std::getenv("FASTLLM_QWEN4_DUMP_DIR") != nullptr ||
            !Qwen4CudaOnlyDeviceMap(this->deviceMap) ||
            !moeDeviceMapGraphCompatible) {
            return false;
        }

        int device = hiddenStates.dataDeviceIds.empty()
            ? FastllmCudaGetDevice() : hiddenStates.dataDeviceIds[0];
        if (device < 0 || (int)pastKeyValues.size() < this->block_cnt) {
            return false;
        }

        const int mutableLayerStart = startBeforeAttention
            ? graphStartLayer : firstFullAttentionLayer + 1;
        const int fullAttentionCacheStartLayer = startBeforeAttention
            ? firstFullAttentionLayer : firstFullAttentionLayer + 1;
        std::vector<void*> linearCachePointers;
        for (int layer = mutableLayerStart;
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

        // Full-attention layers used to sit between independently captured
        // graph segments because their QSA and K/V cache metadata changed on
        // the host every token.  Prepare fixed-capacity storage here; the
        // graph-safe kernels below consume the logical length from one stable
        // device scalar.  A cache growth changes this signature and causes a
        // single recapture, while ordinary decode tokens keep one graph.
        const int graphSequence = hiddenStates.dims[1];
        bool wholeGraphReady =
            (graphSequence == 1 || mtpTargetGraph) &&
            qsaDeviceCompatibleMask &&
            this->indexerCompressRatio > 0 &&
            this->indexerBudget > 0 &&
            this->indexerBudget % this->indexerCompressRatio == 0 &&
            positionIds.dataDevice == DataDevice::CUDA &&
            positionIds.dataType == DataType::FLOAT32 &&
            positionIds.cudaData != nullptr;
        int decodePreviousLength = -1;
        int graphFullLayerCount = 0;
        std::vector<uint64_t> fullCacheSignature;
        auto appendSignature = [&](const void *pointer, int capacity,
                                   uint64_t stride) {
            fullCacheSignature.push_back(
                (uint64_t)(uintptr_t)pointer);
            fullCacheSignature.push_back((uint64_t)capacity);
            fullCacheSignature.push_back(stride);
        };
        auto sameCudaDevice = [&](const Data &data) {
            return data.dataDevice == DataDevice::CUDA &&
                   data.cudaData != nullptr &&
                   (data.dataDeviceIds.empty() ||
                    data.dataDeviceIds[0] == device);
        };
        auto prepareFixedTail = [&](std::shared_ptr<Data> &cache,
                                    const std::vector<int> &logicalDims,
                                    const std::vector<int> &capacityDims) {
            if (cache == nullptr) {
                cache = std::make_shared<Data>(
                    DataType::FLOAT32, logicalDims);
                cache->ToDevice(
                    DataDevice::CUDA, std::vector<int>({device}));
            }
            const bool cudaMetadataMatches =
                cache->dataDevice == DataDevice::CUDA &&
                (cache->dataDeviceIds.empty() ||
                 cache->dataDeviceIds[0] == device);
            if (!cudaMetadataMatches ||
                cache->dataType != DataType::FLOAT32 ||
                cache->dims != logicalDims) {
                return false;
            }
            bool needsGrowth =
                cache->cudaData == nullptr ||
                cache->expansionDims.size() != capacityDims.size();
            if (!needsGrowth) {
                for (int axis = 0; axis < (int)capacityDims.size(); axis++) {
                    if (cache->expansionDims[axis] < capacityDims[axis]) {
                        needsGrowth = true;
                        break;
                    }
                }
            }
            if (needsGrowth) {
                if (logicalDims[0] == 0) {
                    // A restored empty tail may own compact storage. There
                    // are no rows to preserve; avoid the strided copy used
                    // when growing a nonempty cache.
                    cache->FreeSpace();
                }
                cache->Expansion(capacityDims);
            }
            return sameCudaDevice(*cache);
        };

        if (wholeGraphReady) {
            for (int layer = fullAttentionCacheStartLayer;
                 layer < this->block_cnt; layer++) {
                if (this->IsLinearAttentionLayer(layer)) {
                    continue;
                }
                graphFullLayerCount++;
                Data &pastKey = pastKeyValues[layer].first;
                Data &pastValue = pastKeyValues[layer].second;
                if (!sameCudaDevice(pastKey) ||
                    !sameCudaDevice(pastValue) ||
                    pastKey.dims.size() != 3 ||
                    pastValue.dims != pastKey.dims ||
                    pastKey.dataType != pastValue.dataType ||
                    pastKey.dims[0] <= 0 ||
                    pastKey.dims[2] != this->head_dim ||
                    pastKey.strides.size() != 3 ||
                    pastValue.strides.size() != 3) {
                    wholeGraphReady = false;
                    break;
                }
                const int previousLength = pastKey.dims[1];
                if (decodePreviousLength < 0) {
                    decodePreviousLength = previousLength;
                } else if (decodePreviousLength != previousLength) {
                    wholeGraphReady = false;
                    break;
                }

                Data appendShape(
                    pastKey.dataType,
                    {pastKey.dims[0], graphSequence,
                     pastKey.dims[2]});
                const bool geometricGrowth =
                    requestState.geometricCacheGrowthReadyLayers.count(
                        layer) != 0;
                Qwen4EnsureAppendCapacity(
                    pastKey, appendShape, 1, 128,
                    kQwen4DenseCacheMaxGrowth, geometricGrowth);
                Qwen4EnsureAppendCapacity(
                    pastValue, appendShape, 1, 128,
                    kQwen4DenseCacheMaxGrowth, geometricGrowth);
                if (!sameCudaDevice(pastKey) ||
                    !sameCudaDevice(pastValue)) {
                    wholeGraphReady = false;
                    break;
                }

                const int ratio = this->indexerCompressRatio;
                const int oldTailCount = previousLength % ratio;
                const int oldBlocks = previousLength / ratio;
                std::shared_ptr<Data> &tailKeys =
                    requestState.indexerTailKeyTensors[layer];
                std::shared_ptr<Data> &tailPositions =
                    requestState.indexerTailPositionTensors[layer];
                std::shared_ptr<Data> &blockCache =
                    requestState.indexerBlockKeyTensors[layer];
                if (!prepareFixedTail(
                        tailKeys,
                        {oldTailCount, this->indexerHeadDim},
                        {ratio, this->indexerHeadDim}) ||
                    !prepareFixedTail(
                        tailPositions, {oldTailCount}, {ratio}) ||
                    blockCache == nullptr ||
                    !sameCudaDevice(*blockCache) ||
                    blockCache->dataType != DataType::FLOAT32 ||
                    blockCache->dims !=
                        std::vector<int>({oldBlocks,
                                          this->indexerHeadDim})) {
                    wholeGraphReady = false;
                    break;
                }

                const int kvCapacity = std::min(
                    Qwen4AxisCapacity(pastKey, 1),
                    Qwen4AxisCapacity(pastValue, 1));
                const int blockCapacity = std::max(
                    oldBlocks,
                    (kvCapacity + ratio - 1) / ratio);
                if (Qwen4AxisCapacity(*blockCache, 0) < blockCapacity) {
                    blockCache->Expansion(
                        {blockCapacity, this->indexerHeadDim});
                }

                std::vector<float> &rawKeys =
                    requestState.indexerRawKeys[layer];
                std::vector<float> &positions =
                    requestState.indexerPositions[layer];
                std::shared_ptr<QsaHostMirrorTransfer> &hostTransfer =
                    requestState.indexerHostMirrorTransfers[layer];
                if (hostTransfer == nullptr) {
                    hostTransfer =
                        std::make_shared<QsaHostMirrorTransfer>();
                }
                hostTransfer->Prepare(
                    rawKeys, positions,
                    previousLength - oldTailCount,
                    this->indexerHeadDim);

                if (mtpTargetGraph) {
                    Data &rawKeyCapture =
                        verificationCapture->qsaRawKeys[layer];
                    if (!Qwen4PrepareDecodeGraphWorkspace(
                            rawKeyCapture, DataType::FLOAT32,
                            {graphSequence, this->indexerHeadDim},
                            device)) {
                        wholeGraphReady = false;
                        break;
                    }
                    // Reservation is not a captured QSA append. Keep the
                    // logical history empty until the whole graph is chosen;
                    // segmented/eager attention appends its own rows.
                    rawKeyCapture.Resize({0, this->indexerHeadDim});
                    appendSignature(
                        rawKeyCapture.cudaData, graphSequence,
                        rawKeyCapture.strides[0]);
                }

                appendSignature(
                    pastKey.cudaData, Qwen4AxisCapacity(pastKey, 1),
                    pastKey.strides[0]);
                appendSignature(
                    pastValue.cudaData, Qwen4AxisCapacity(pastValue, 1),
                    pastValue.strides[0]);
                appendSignature(
                    tailKeys->cudaData,
                    Qwen4AxisCapacity(*tailKeys, 0),
                    tailKeys->strides[0]);
                appendSignature(
                    tailPositions->cudaData,
                    Qwen4AxisCapacity(*tailPositions, 0),
                    tailPositions->strides[0]);
                appendSignature(
                    blockCache->cudaData,
                    Qwen4AxisCapacity(*blockCache, 0),
                    blockCache->strides[0]);
            }
        }
        wholeGraphReady = wholeGraphReady && graphFullLayerCount > 0 &&
            decodePreviousLength >= this->indexerBudget;
        if (mtpTargetGraph && attentionMask.dims.empty() &&
            !wholeGraphReady) {
            // A verifier may defer construction of its dense causal mask
            // when the whole graph can encode causality from decodeMeta.
            // Segmented/eager attention still consumes an explicit mask, so
            // return before mutating caches and let the caller materialize it.
            return false;
        }

        // Keep a request on its original graph topology. Switching a live
        // request from segmented graphs to the whole-backbone graph changes
        // the full-attention execution path at the sparse threshold and can
        // perturb a close boundary logit. Short prompts that cross the
        // threshold during generation therefore retain the established
        // segmented path; requests that start at or above it use the whole
        // graph from their first decode token.
        if (wholeGraphReady && !graphState->segments.empty() &&
            !graphState->wholeGraphMode) {
            wholeGraphReady = false;
        }

        const bool workspaceShapeChanged =
            graphState->hiddenStates[0].cudaData != nullptr &&
            (graphState->hiddenStates[0].dataType != hiddenStates.dataType ||
             graphState->hiddenStates[0].dims != hiddenStates.dims);
        if (graphState->device != device || workspaceShapeChanged ||
            graphState->graphStartLayer != graphStartLayer ||
            graphState->startBeforeAttention != startBeforeAttention ||
            (!graphState->linearCachePointers.empty() &&
             graphState->linearCachePointers != linearCachePointers) ||
            (!graphState->segments.empty() &&
             graphState->wholeGraphMode != wholeGraphReady) ||
            (wholeGraphReady &&
             !graphState->fullCacheSignature.empty() &&
             graphState->fullCacheSignature != fullCacheSignature)) {
            graphState->Reset(device);
        }
        graphState->device = device;
        graphState->graphStartLayer = graphStartLayer;
        graphState->startBeforeAttention = startBeforeAttention;
        graphState->linearCachePointers = linearCachePointers;
        graphState->wholeGraphMode = wholeGraphReady;
        graphState->fullCacheSignature = fullCacheSignature;
        if (!Qwen4PrepareDecodeGraphTensor(
                graphState->hiddenStates[0], hiddenStates, device) ||
            (!startBeforeAttention &&
             (!Qwen4PrepareDecodeGraphTensor(
                  graphState->attentionOutput,
                  attentionOutput, device) ||
              !Qwen4PrepareDecodeGraphTensor(
                  graphState->attentionInjection,
                  attentionInjection, device))) ||
            (wholeGraphReady &&
             !Qwen4PrepareDecodeGraphTensor(
                 graphState->positionIds, positionIds, device))) {
            return false;
        }
        if (mtpTargetGraph) {
            // PLE may finish its CPU-offloaded ngram result on a transfer
            // stream.  The first verifier segment consumes the staged
            // residual immediately, so retire that producer and its staging
            // copy before entering/replaying the capture domain.
            FastllmCudaSyncCurrentThreadStream();
        }
        if (wholeGraphReady) {
            std::vector<void *> tailSnapshotDestinations;
            std::vector<const void *> tailSnapshotSources;
            std::vector<size_t> tailSnapshotSizes;
            if (graphState->decodeMeta.cudaData == nullptr) {
                graphState->decodeMeta.dataType = DataType::INT32;
                graphState->decodeMeta.Resize({1});
                graphState->decodeMeta.dataDevice = DataDevice::CUDA;
                graphState->decodeMeta.dataDeviceIds = {device};
                graphState->decodeMeta.Allocate(false);
            }
            if (graphState->pinnedDecodeMeta == nullptr) {
                graphState->pinnedDecodeMeta =
                    reinterpret_cast<int32_t *>(
                        FastllmCudaHostMalloc(sizeof(int32_t)));
            }
            if (graphState->decodeMeta.cudaData == nullptr ||
                graphState->pinnedDecodeMeta == nullptr) {
                return false;
            }
            graphState->pinnedDecodeMeta[0] = decodePreviousLength;
            if (!FastllmCudaCopyFromPinnedHostToDeviceAsyncCurrentThread(
                    graphState->decodeMeta.cudaData,
                    graphState->pinnedDecodeMeta,
                    sizeof(int32_t))) {
                return false;
            }

            const int selectedBlocks =
                this->indexerBudget / this->indexerCompressRatio;
            for (int layer = fullAttentionCacheStartLayer;
                 layer < this->block_cnt; layer++) {
                if (this->IsLinearAttentionLayer(layer)) {
                    continue;
                }
                auto blockCacheIt =
                    requestState.indexerBlockKeyTensors.find(layer);
                if (blockCacheIt ==
                        requestState.indexerBlockKeyTensors.end() ||
                    blockCacheIt->second == nullptr) {
                    return false;
                }
                const int blockCapacity = Qwen4AxisCapacity(
                    *blockCacheIt->second, 0);
                if (blockCapacity < selectedBlocks ||
                    !Qwen4PrepareDecodeGraphWorkspace(
                        graphState->qsaScoreWorkspaces[layer],
                        DataType::FLOAT32,
                        {graphSequence, blockCapacity}, device) ||
                    !Qwen4PrepareDecodeGraphWorkspace(
                        graphState->qsaSelectedWorkspaces[layer],
                        DataType::INT32,
                        {graphSequence, selectedBlocks}, device) ||
                    (mtpTargetGraph &&
                     (!Qwen4PrepareDecodeGraphWorkspace(
                          graphState->qsaTailKeySnapshots[layer],
                          DataType::FLOAT32,
                          {this->indexerCompressRatio,
                           this->indexerHeadDim}, device) ||
                      !Qwen4PrepareDecodeGraphWorkspace(
                          graphState->qsaTailPositionSnapshots[layer],
                          DataType::FLOAT32,
                          {this->indexerCompressRatio}, device)))) {
                    return false;
                }
                const int oldTailCount =
                    decodePreviousLength % this->indexerCompressRatio;
                if (mtpTargetGraph && oldTailCount > 0) {
                    const Data &tailKeys = *requestState.
                        indexerTailKeyTensors[layer];
                    const Data &tailPositions = *requestState.
                        indexerTailPositionTensors[layer];
                    tailSnapshotDestinations.push_back(
                        graphState->qsaTailKeySnapshots[layer].cudaData);
                    tailSnapshotSources.push_back(tailKeys.cudaData);
                    tailSnapshotSizes.push_back(
                        (size_t)oldTailCount *
                            this->indexerHeadDim * sizeof(float));
                    tailSnapshotDestinations.push_back(
                        graphState->qsaTailPositionSnapshots[layer].
                            cudaData);
                    tailSnapshotSources.push_back(
                        tailPositions.cudaData);
                    tailSnapshotSizes.push_back(
                        (size_t)oldTailCount * sizeof(float));
                }
            }
            if (mtpTargetGraph && !tailSnapshotDestinations.empty() &&
                !FastllmCudaBatchCopyFromDeviceToDeviceAsyncCurrentThread(
                    tailSnapshotDestinations.data(),
                    tailSnapshotSources.data(), tailSnapshotSizes.data(),
                    (int)tailSnapshotDestinations.size())) {
                return false;
            }
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

        auto runSegmentBody = [&](int startLayer, int nextFullLayer,
                                  bool firstAttentionReady) {
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
                if (layer == startLayer && firstAttentionReady) {
                    currentAttentionOutput = &graphState->attentionOutput;
                    currentAttentionInjection =
                        &graphState->attentionInjection;
                } else {
                    const std::string attentionHyperPrefix =
                        layerPrefix + "attn_hyper_connection.";
                    Data attentionNorm;
                    Data *normalizedAttention =
                        &carriedAttentionNorm;
                    if (layer == startLayer) {
                        GroupedRMSNorm(
                            graphState->hiddenStates[0],
                            this->weight[attentionHyperPrefix +
                                         "hc_norm.weight"],
                            attentionNorm);
                        normalizedAttention = &attentionNorm;
                    }
                    HyperMixNormalized(
                        *normalizedAttention, attentionHyperPrefix,
                        linearAttentionInput,
                        &linearAttentionInjection);
                    normalizedAttention->FreeSpace();
                    AssertInFastLLM(
                        this->IsLinearAttentionLayer(layer),
                        "Qwen4 decode graph can only enter before a linear-attention layer.");
                    RunLinearAttention(
                        layer, linearAttentionInput,
                        pastKeyValues[layer].first,
                        pastKeyValues[layer].second,
                        linearAttentionOutput,
                        verificationCapture == nullptr ? nullptr :
                            &verificationCapture->linearConvInputs[layer],
                        verificationCapture == nullptr ? nullptr :
                            &verificationCapture->linearAlphas[layer],
                        verificationCapture == nullptr ? nullptr :
                            &verificationCapture->linearBetas[layer],
                        verificationCapture == nullptr ||
                            graphSequence <= 1 ||
                            verificationCapture->runtimeCheckpoint ==
                                nullptr ||
                            layer >= (int)verificationCapture->
                                runtimeCheckpoint->linearSecond.size()
                            ? nullptr
                            : &verificationCapture->runtimeCheckpoint->
                                linearSecond[layer]);
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

        const int wholeGraphRemainder = wholeGraphReady
            ? (decodePreviousLength + graphSequence) %
                this->indexerCompressRatio
            : 0;
        const int wholeGraphSparseWidth =
            this->indexerBudget +
            (mtpTargetGraph ? this->indexerCompressRatio - 1
                            : wholeGraphRemainder);

        auto runFullAttentionGraph = [&] (
                int layer, const Data &input,
                Data &pastKey, Data &pastValue,
                Data &output) -> bool {
            const std::string attention = languagePrefix + "layers." +
                std::to_string(layer) + ".self_attn.";
            const std::string indexer = attention + "indexer.";
            constexpr int batch = 1;
            const int sequence = graphSequence;
            const int ratio = this->indexerCompressRatio;
            const int32_t *decodeMeta = reinterpret_cast<const int32_t *>(
                graphState->decodeMeta.cudaData);

            auto tailKeysIt =
                requestState.indexerTailKeyTensors.find(layer);
            auto tailPositionsIt =
                requestState.indexerTailPositionTensors.find(layer);
            auto blockCacheIt =
                requestState.indexerBlockKeyTensors.find(layer);
            auto scoreWorkspaceIt =
                graphState->qsaScoreWorkspaces.find(layer);
            auto selectedWorkspaceIt =
                graphState->qsaSelectedWorkspaces.find(layer);
            if (decodeMeta == nullptr ||
                tailKeysIt == requestState.indexerTailKeyTensors.end() ||
                tailPositionsIt ==
                    requestState.indexerTailPositionTensors.end() ||
                blockCacheIt ==
                    requestState.indexerBlockKeyTensors.end() ||
                scoreWorkspaceIt ==
                    graphState->qsaScoreWorkspaces.end() ||
                selectedWorkspaceIt ==
                    graphState->qsaSelectedWorkspaces.end() ||
                tailKeysIt->second == nullptr ||
                tailPositionsIt->second == nullptr ||
                blockCacheIt->second == nullptr) {
                return false;
            }
            Data &tailKeys = *tailKeysIt->second;
            Data &tailPositions = *tailPositionsIt->second;
            Data &blockCache = *blockCacheIt->second;
            Data &scoreWorkspace = scoreWorkspaceIt->second;
            Data &selectedWorkspace = selectedWorkspaceIt->second;

            Data typedInput, projected, indexQuery, currentRawKeys;
            ToDataType(input, typedInput, this->dataType);
            const int qsaParallelRegion = (1 << 20) + layer;
            const bool parallelQsa = mtpTargetGraph;
            if (parallelQsa) {
                FastllmCudaGraphMarkParallelFork(qsaParallelRegion);
            }
            Linear(
                typedInput,
                this->weight[indexer + "index_qk_proj.weight"],
                Data(), projected);
            const int queryColumns =
                this->indexerHeads * this->indexerHeadDim;
            Split(projected, -1, 0, queryColumns, indexQuery);
            Split(
                projected, -1, queryColumns,
                queryColumns +
                    this->indexerKvHeads * this->indexerHeadDim,
                currentRawKeys);
            indexQuery.Reshape(
                {batch, sequence, this->indexerHeads,
                 this->indexerHeadDim});
            currentRawKeys.Reshape(
                {batch, sequence, this->indexerHeadDim});
            RMSNorm(
                indexQuery,
                this->weight[indexer + "q_layernorm.weight"],
                this->rms_norm_eps, indexQuery);
            LlamaRotatePosition2DPart(
                indexQuery, graphState->positionIds,
                this->sinData, this->cosData,
                this->rotary_dim, this->rotary_dim);

            Data currentKeysFloat;
            ToDataType(
                currentRawKeys, currentKeysFloat,
                DataType::FLOAT32);
            currentKeysFloat.Reshape(
                {sequence, this->indexerHeadDim});
            if (verificationCapture != nullptr) {
                Data &rawKeyCapture =
                    verificationCapture->qsaRawKeys[layer];
                if (rawKeyCapture.dataDevice != DataDevice::CUDA ||
                    rawKeyCapture.cudaData == nullptr ||
                    rawKeyCapture.dataType != DataType::FLOAT32 ||
                    rawKeyCapture.dims != currentKeysFloat.dims ||
                    !FastllmCudaCopyFromDeviceToDeviceAsyncCurrentThread(
                        rawKeyCapture.cudaData,
                        currentKeysFloat.cudaData,
                        currentKeysFloat.GetBytes())) {
                    return false;
                }
            }

            Data tailKeysFixed, tailPositionsFixed;
            Qwen4BorrowCudaStorage(tailKeysFixed, tailKeys);
            Qwen4BorrowCudaStorage(
                tailPositionsFixed, tailPositions);
            tailKeysFixed.Resize(
                {ratio, this->indexerHeadDim});
            tailPositionsFixed.Resize({ratio});

            // Append each graph row in causal order. The graph topology is
            // fixed; the device-side base length selects the tail slot and
            // conditionally commits completed groups. Keep all per-row views
            // and work tensors alive until the QSA/attention join below.
            const bool fusedCompress =
                sequence == 4 && ratio == 4 &&
                this->indexerHeadDim == 128 &&
                this->rotary_dim == 64;
            if (fusedCompress) {
                const bool compressed =
                    FastllmCudaQwen4QSAAppendCompress4Graph(
                        currentKeysFloat, graphState->positionIds,
                        this->weight[indexer + "k_layernorm.weight"],
                        this->sinData, this->cosData, decodeMeta,
                        tailKeys, tailPositions, blockCache,
                        this->rms_norm_eps);
                if (!compressed) {
                    return false;
                }
            } else {
                std::vector<Data> tokenKeys(sequence);
                std::vector<Data> tokenPositions(sequence);
                std::vector<Data> pooledKeys(sequence);
                std::vector<Data> averagedKeys(sequence);
                std::vector<Data> normalizedKeys(sequence);
                std::vector<Data> firstPositions(sequence);
                std::vector<Data> memberKeys(
                    sequence * std::max(0, ratio - 1));
                for (int token = 0; token < sequence; token++) {
                    Data &tokenKey = tokenKeys[token];
                    Data &tokenPosition = tokenPositions[token];
                    Split(
                        currentKeysFloat, 0, token, token + 1,
                        tokenKey);
                    Split(
                        graphState->positionIds, 1, token, token + 1,
                        tokenPosition);
                    tokenKey.Reshape({1, this->indexerHeadDim});
                    tokenPosition.Reshape({1});
                    if (!FastllmCudaQwen4QSAAppendGraph(
                            tokenKey, tokenPosition, decodeMeta, token,
                            ratio, tailKeys, tailPositions)) {
                        return false;
                    }
                    // Preserve the established float32 pooling/RMSNorm/RoPE
                    // arithmetic.
                    Data &pooled = pooledKeys[token];
                    Split(tailKeysFixed, 0, 0, 1, pooled);
                    pooled.Reshape({1, this->indexerHeadDim});
                    for (int member = 1; member < ratio; member++) {
                        Data &memberKey = memberKeys[
                            token * (ratio - 1) + member - 1];
                        Split(
                            tailKeysFixed, 0, member, member + 1,
                            memberKey);
                        memberKey.Reshape({1, this->indexerHeadDim});
                        AddTo(pooled, memberKey);
                    }
                    Data &averaged = averagedKeys[token];
                    Data &normalized = normalizedKeys[token];
                    Mul(pooled, 1.0f / (float)ratio, averaged);
                    RMSNorm(
                        averaged,
                        this->weight[indexer + "k_layernorm.weight"],
                        this->rms_norm_eps, normalized);
                    Data &firstPosition = firstPositions[token];
                    Split(
                        tailPositionsFixed, 0, 0, 1,
                        firstPosition);
                    firstPosition.Reshape({1, 1});
                    normalized.Reshape(
                        {1, 1, 1, this->indexerHeadDim});
                    LlamaRotatePosition2DPart(
                        normalized, firstPosition,
                        this->sinData, this->cosData,
                        this->rotary_dim, this->rotary_dim);
                    normalized.Reshape({1, this->indexerHeadDim});
                    if (!FastllmCudaQwen4QSACommitGraph(
                            normalized, decodeMeta, token, ratio,
                            blockCache)) {
                        return false;
                    }
                }
            }

            Data qsaIndices(
                DataType::INT32,
                {sequence, wholeGraphSparseWidth});
            qsaIndices.ToDevice(
                DataDevice::CUDA, std::vector<int>({device}));
            qsaIndices.Allocate(false);
            if (!FastllmCudaQwen4QSASelectGraph(
                    indexQuery, blockCache, decodeMeta,
                    scoreWorkspace, selectedWorkspace, qsaIndices,
                    this->indexerHeads,
                    this->indexerHeadDim, this->indexerBudget,
                    ratio)) {
                return false;
            }
            if (parallelQsa) {
                FastllmCudaGraphMarkParallelFirstDone(
                    qsaParallelRegion);
                FastllmCudaGraphMarkParallelSecondBegin(
                    qsaParallelRegion);
            }

            Data qGate, query, key, value, gate;
            Linear(
                typedInput, this->weight[attention + "q_proj.weight"],
                Data(), qGate);
            qGate.Reshape(
                {batch, sequence, -1, this->head_dim * 2});
            Split(qGate, -1, 0, this->head_dim, query);
            Split(
                qGate, -1, this->head_dim,
                this->head_dim * 2, gate);
            gate.Reshape({batch, sequence, -1});
            Linear(
                typedInput, this->weight[attention + "k_proj.weight"],
                Data(), key);
            Linear(
                typedInput, this->weight[attention + "v_proj.weight"],
                Data(), value);
            key.Reshape({batch, sequence, -1, this->head_dim});
            value.Reshape({batch, sequence, -1, this->head_dim});
            RMSNorm(
                query, this->weight[attention + "q_norm.weight"],
                this->rms_norm_eps, query);
            RMSNorm(
                key, this->weight[attention + "k_norm.weight"],
                this->rms_norm_eps, key);
            LlamaRotatePosition2DPart(
                query, graphState->positionIds,
                this->sinData, this->cosData,
                this->rotary_dim, this->rotary_dim);
            LlamaRotatePosition2DPart(
                key, graphState->positionIds,
                this->sinData, this->cosData,
                this->rotary_dim, this->rotary_dim);
            PermuteSelf(query, {0, 2, 1, 3});
            PermuteSelf(key, {0, 2, 1, 3});
            PermuteSelf(value, {0, 2, 1, 3});
            query.Reshape({-1, sequence, this->head_dim});
            key.Reshape({-1, sequence, this->head_dim});
            value.Reshape({-1, sequence, this->head_dim});
            if (!FastllmCudaQwen4KVAppendGraph(
                    key, value, decodeMeta, pastKey, pastValue)) {
                return false;
            }
            if (parallelQsa) {
                FastllmCudaGraphMarkParallelJoin(qsaParallelRegion);
            }

            Data context;
            const int attentionGroup =
                query.dims[0] / pastKey.dims[0];
            const float attentionScale =
                1.0f / std::sqrt((float)this->head_dim);
            if (sequence == 1) {
                Data compactKey(
                    key.dataType,
                    {pastKey.dims[0], wholeGraphSparseWidth,
                     this->head_dim});
                Data compactValue(
                    value.dataType,
                    {pastValue.dims[0], wholeGraphSparseWidth,
                     this->head_dim});
                compactKey.ToDevice(
                    DataDevice::CUDA, std::vector<int>({device}));
                compactValue.ToDevice(
                    DataDevice::CUDA, std::vector<int>({device}));
                compactKey.Allocate(false);
                compactValue.Allocate(false);
                if (!FastllmCudaQwen4GatherKVGraph(
                        pastKey, pastValue, qsaIndices, decodeMeta,
                        compactKey, compactValue)) {
                    return false;
                }
                Attention(
                    query, compactKey, compactValue,
                    Data(), context,
                    attentionGroup, attentionScale, 1);
            } else {
                const int queryHeads = query.dims[0];
                const int keyHeads = pastKey.dims[0];
                Data packedQuery(
                    query.dataType,
                    {sequence * queryHeads, 1, this->head_dim});
                Data compactKey(
                    key.dataType,
                    {sequence * keyHeads, wholeGraphSparseWidth,
                     this->head_dim});
                Data compactValue(
                    value.dataType,
                    {sequence * keyHeads, wholeGraphSparseWidth,
                     this->head_dim});
                Data paddingMask(
                    query.dataType,
                    {sequence, 1, wholeGraphSparseWidth});
                Data packedOutput(
                    query.dataType,
                    {sequence * queryHeads, 1, this->head_dim});
                context.dataType = query.dataType;
                context.UpdateUnitSize();
                context.Resize(
                    {queryHeads, sequence, this->head_dim});
                for (Data *workspace : std::vector<Data *>({
                         &packedQuery, &compactKey, &compactValue,
                         &paddingMask, &packedOutput, &context})) {
                    workspace->ToDevice(
                        DataDevice::CUDA,
                        std::vector<int>({device}), false);
                    workspace->Allocate(false);
                }
                if (!FastllmCudaQwen4PrepareSparseBatchGraph(
                        query, pastKey, pastValue, qsaIndices,
                        decodeMeta, sequence, packedQuery,
                        compactKey, compactValue, paddingMask,
                        0, sequence)) {
                    return false;
                }
                Attention(
                    packedQuery, compactKey, compactValue,
                    paddingMask, packedOutput,
                    attentionGroup, attentionScale, 1);
                if (!FastllmCudaQwen4UnpackSparseBatch(
                        packedOutput, context, 0, sequence)) {
                    return false;
                }
            }
            PermuteSelf(context, {1, 0, 2});
            context.Reshape({sequence, batch, -1});
            PermuteSelf(context, {1, 0, 2});
            SigmoidMulTo(context, gate);
            Linear(
                context, this->weight[attention + "o_proj.weight"],
                Data(), output);
            return true;
        };

        auto runWholeGraphBody = [&]() {
            int startLayer = graphStartLayer;
            while (startLayer < this->block_cnt) {
                int nextFullLayer = startLayer + 1;
                while (nextFullLayer < this->block_cnt &&
                       this->IsLinearAttentionLayer(nextFullLayer)) {
                    nextFullLayer++;
                }
                runSegmentBody(
                    startLayer, nextFullLayer,
                    startLayer != graphStartLayer ||
                        !startBeforeAttention);
                if (nextFullLayer >= this->block_cnt) {
                    break;
                }
                ApplyDeviceMap(
                    this->deviceMap, nextFullLayer + 1,
                    this->block_cnt);
                if (!runFullAttentionGraph(
                        nextFullLayer, graphState->attentionInput,
                        pastKeyValues[nextFullLayer].first,
                        pastKeyValues[nextFullLayer].second,
                        graphState->attentionOutput)) {
                    FastllmCudaSetThreadError();
                    return;
                }
                startLayer = nextFullLayer;
            }
        };

        bool captureAttempted = false;
        auto runSegment = [&](int startLayer, int nextFullLayer,
                              bool wholeGraph, int wholeGraphVariant) {
            const int segmentKey = wholeGraph
                ? -1 - wholeGraphVariant : startLayer;
            auto runBody = [&]() {
                if (wholeGraph) {
                    runWholeGraphBody();
                } else {
                    runSegmentBody(
                        startLayer, nextFullLayer,
                        startLayer != graphStartLayer ||
                            !startBeforeAttention);
                }
            };
            DecodeCudaGraphState::Segment &segment =
                graphState->segments[segmentKey];
            FastllmCudaSetDevice(device);

            if (segment.captured) {
                if (FastllmCudaGraphLaunch(segment.exec)) {
                    return;
                }
                std::fprintf(
                    stderr,
                    "[Fastllm] Qwen4 decode CUDA graph replay failed "
                    "on GPU %d segment %d: %s; using eager fallback.\n",
                    device, segmentKey,
                    FastllmCudaGraphLastError());
                destroySegment(segment);
                segment.disabled = true;
                FastllmCudaClearThreadError();
                FastllmCudaClearGraphError();
                runBody();
                return;
            }

            FastllmCudaMergeMOEClearGraphUnsafeFallbackFlag();
            if (!segment.warmed || segment.disabled) {
                runBody();
                segment.warmed = true;
                if (FastllmCudaMergeMOEUsedGraphUnsafeFallback()) {
                    segment.disabled = true;
                }
                return;
            }
            if (captureAttempted) {
                runBody();
                return;
            }
            captureAttempted = true;

            void *capturedGraph = nullptr;
            void *capturedExec = nullptr;
            bool poolActive = false;
            bool captureActive = false;
            bool recaptureWithoutParallelMarkers = false;
            const bool enableParallelRegions =
                !segment.parallelRegionsUnavailable;
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
                const bool previousParallelMarkers =
                    FastllmCudaGraphSetParallelMarkersEnabled(
                        enableParallelRegions);
                try {
                    runBody();
                } catch (...) {
                    bodyThrew = true;
                }
                FastllmCudaGraphSetParallelMarkersEnabled(
                    previousParallelMarkers);
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
            if (captureOk && enableParallelRegions) {
                const int parallelRegions =
                    FastllmCudaGraphOptimizeParallelRegions(
                        capturedGraph);
                if (parallelRegions ==
                        FASTLLM_CUDA_GRAPH_RECAPTURE_WITHOUT_PARALLEL_MARKERS) {
                    // Some CUDA graph node types forbid marker removal. Keep
                    // the segment usable: discard this graph and capture the
                    // original serial topology without markers next token.
                    segment.parallelRegionsUnavailable = true;
                    recaptureWithoutParallelMarkers = true;
                    captureOk = false;
                } else if (parallelRegions < 0) {
                    captureOk = false;
                }
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
                if (!recaptureWithoutParallelMarkers) {
                    segment.captureFailures++;
                }
                if (!recaptureWithoutParallelMarkers &&
                    segment.captureFailures >= 3) {
                    segment.disabled = true;
                    std::fprintf(
                        stderr,
                        "[Fastllm] Qwen4 decode CUDA graph disabled "
                        "on GPU %d segment %d after capture failure: %s\n",
                        device, segmentKey,
                        FastllmCudaGraphLastError());
                }
                FastllmCudaClearThreadError();
                FastllmCudaClearGraphError();
                FastllmCudaMergeMOEClearGraphUnsafeFallbackFlag();
                runBody();
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
                    device, segmentKey,
                    FastllmCudaGraphLastError());
                destroySegment(segment);
                segment.disabled = true;
                FastllmCudaClearThreadError();
                FastllmCudaClearGraphError();
                runBody();
                return;
            }
        };

        if (wholeGraphReady) {
            if (mtpTargetGraph) {
                for (int layer = fullAttentionCacheStartLayer;
                     layer < this->block_cnt; layer++) {
                    if (!this->IsLinearAttentionLayer(layer)) {
                        verificationCapture->qsaRawKeys[layer].Resize(
                            {graphSequence, this->indexerHeadDim});
                    }
                }
            }
            runSegment(
                graphStartLayer, this->block_cnt, true,
                mtpTargetGraph ? this->indexerCompressRatio
                               : wholeGraphRemainder);

            const int keyLength =
                decodePreviousLength + graphSequence;
            const int tailCount =
                keyLength % this->indexerCompressRatio;
            const int blockCount =
                keyLength / this->indexerCompressRatio;
            const int oldTailCount =
                decodePreviousLength % this->indexerCompressRatio;
            const int newlyCompletedRows =
                ((oldTailCount + graphSequence) /
                 this->indexerCompressRatio) *
                this->indexerCompressRatio;
            for (int layer = fullAttentionCacheStartLayer;
                 layer < this->block_cnt; layer++) {
                if (this->IsLinearAttentionLayer(layer)) {
                    continue;
                }
                Data &pastKey = pastKeyValues[layer].first;
                Data &pastValue = pastKeyValues[layer].second;
                pastKey.Resize(
                    {pastKey.dims[0], keyLength,
                     pastKey.dims[2]});
                pastValue.Resize(
                    {pastValue.dims[0], keyLength,
                     pastValue.dims[2]});
                requestState.geometricCacheGrowthReadyLayers.insert(
                    layer);

                std::shared_ptr<Data> &tailKeys =
                    requestState.indexerTailKeyTensors[layer];
                std::shared_ptr<Data> &tailPositions =
                    requestState.indexerTailPositionTensors[layer];
                std::shared_ptr<Data> &blockCache =
                    requestState.indexerBlockKeyTensors[layer];
                if (mtpTargetGraph && newlyCompletedRows > 0) {
                    AssertInFastLLM(
                        verificationCapture != nullptr,
                        "Qwen4-Exp target graph lost its QSA capture state.");
                    std::shared_ptr<QsaHostMirrorTransfer> &transfer =
                        requestState.indexerHostMirrorTransfers[layer];
                    int queuedRows =
                        decodePreviousLength - oldTailCount;
                    if (oldTailCount > 0) {
                        Data oldKeys, oldPositions;
                        oldKeys.FakeFrom(
                            graphState->qsaTailKeySnapshots[layer], 0);
                        oldKeys.Resize(
                            {oldTailCount, this->indexerHeadDim});
                        oldPositions.FakeFrom(
                            graphState->qsaTailPositionSnapshots[layer],
                            0);
                        oldPositions.Resize({oldTailCount});
                        transfer->Queue(
                            oldKeys, oldPositions, queuedRows,
                            oldTailCount, this->indexerHeadDim);
                        queuedRows += oldTailCount;
                    }
                    const int candidateRows =
                        newlyCompletedRows - oldTailCount;
                    if (candidateRows > 0) {
                        Data rawKeys, candidatePositions;
                        rawKeys.FakeFrom(
                            verificationCapture->qsaRawKeys[layer], 0);
                        rawKeys.Resize(
                            {candidateRows, this->indexerHeadDim});
                        candidatePositions.FakeFrom(
                            graphState->positionIds, 0);
                        candidatePositions.Resize({candidateRows});
                        transfer->Queue(
                            rawKeys, candidatePositions, queuedRows,
                            candidateRows, this->indexerHeadDim);
                    }
                } else if (!mtpTargetGraph && tailCount == 0) {
                    const int previousRows =
                        keyLength - this->indexerCompressRatio;
                    requestState.indexerHostMirrorTransfers[layer]->Queue(
                        *tailKeys, *tailPositions,
                        previousRows, this->indexerCompressRatio,
                        this->indexerHeadDim);
                }
                tailKeys->Resize(
                    {tailCount, this->indexerHeadDim});
                tailPositions->Resize({tailCount});
                blockCache->Resize(
                    {blockCount, this->indexerHeadDim});
            }
        } else {
            int startLayer = graphStartLayer;
            while (startLayer < this->block_cnt) {
                int nextFullLayer = startLayer + 1;
                while (nextFullLayer < this->block_cnt &&
                       this->IsLinearAttentionLayer(nextFullLayer)) {
                    nextFullLayer++;
                }
                runSegment(startLayer, nextFullLayer, false, 0);
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
                    requestState, graphState->attentionOutput,
                    verificationCapture == nullptr ? nullptr :
                        &verificationCapture->qsaRawKeys[nextFullLayer]);
                startLayer = nextFullLayer;
            }
        }

        if (graphState->logits.dataDevice != DataDevice::CUDA ||
            graphState->logits.cudaData == nullptr ||
            graphState->logits.dims.empty()) {
            return false;
        }
        if (targetMultiHidden != nullptr) {
            targetMultiHidden->CopyFrom(graphState->hiddenStates[0]);
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

    bool Qwen4ExpModel::ShouldRecordPrefixSnapshot(
            const std::vector<std::pair<Data, Data>> &pastKeyValues,
            const RequestState &state, int &cachedLen) const {
        if (!Qwen4PrefixCacheEnabled() ||
            (int)pastKeyValues.size() < this->block_cnt) {
            return false;
        }

        cachedLen = 0;
        for (int layer = 0; layer < this->block_cnt; layer++) {
            if (this->IsLinearAttentionLayer(layer)) {
                continue;
            }
            const Data &key = pastKeyValues[layer].first;
            const Data &value = pastKeyValues[layer].second;
            if (key.dims.size() < 2 || value.dims.size() < 2 ||
                key.dims[1] <= 0 || key.dims[1] != value.dims[1] ||
                (cachedLen > 0 && cachedLen != key.dims[1])) {
                return false;
            }
            cachedLen = key.dims[1];
        }
        const int interval = std::max(1, fastllm::GetPageLen()) *
            std::max(1, Qwen4EnvInt(
                "FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES", 16));
        return cachedLen > 0 &&
            cachedLen == (int)state.processedTokens.size() &&
            cachedLen - state.lastPrefixSnapshotLen >= interval;
    }

    void Qwen4ExpModel::MaybeRecordPrefixSnapshot(
            const std::vector<std::pair<Data, Data>> &pastKeyValues,
            RequestState &state) {
        int cachedLen = 0;
        if (!ShouldRecordPrefixSnapshot(
                pastKeyValues, state, cachedLen)) {
            return;
        }

        {
            std::lock_guard<std::mutex> guard(this->prefixCacheMutex);
            auto existing = FindPrefixSnapshotLocked(
                state.processedTokens, cachedLen, cachedLen);
            if (existing != nullptr) {
                existing->timestamp = ++this->prefixSnapshotTimestamp;
                state.lastPrefixSnapshotLen = cachedLen;
                return;
            }
        }
        // Snapshot materialization requires the deferred MTP boundary;
        // proposal generation may already have advanced its QSA history.
        if (state.mtpState != nullptr &&
            !CanCloneMtpPrefixState(*state.mtpState, cachedLen)) {
            return;
        }

        // Snapshot state is copied by value. Complete the asynchronous groups
        // and materialize the device-resident PLE/QSA histories once at this
        // boundary so the snapshot has a self-sufficient generic fallback
        // representation.
        MaterializePLEHostHistory(state);
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
        std::shared_ptr<MtpRuntimeState> mtpSnapshotState;
        if (state.mtpState != nullptr) {
            // Unlike the target request state, the MTP attention state is not
            // visited by the target snapshot validation loop above.  Finish
            // its asynchronous QSA host mirror before dropping the transfer
            // object in CloneMtpPrefixState, so the restored request can seed
            // a new mirror from a complete host history.
            MaterializeQsaHostHistory(
                this->block_cnt, std::max(0, cachedLen - 1),
                state.mtpState->attentionState);
            mtpSnapshotState = CloneMtpPrefixState(
                *state.mtpState, cachedLen);
            if (mtpSnapshotState == nullptr) {
                return;
            }
            tensorBytes += mtpSnapshotState->key.GetBytes() +
                mtpSnapshotState->value.GetBytes() +
                mtpSnapshotState->deferredTargetHidden.GetBytes();
            const RequestState &mtpAttention =
                mtpSnapshotState->attentionState;
            tensorBytes += Qwen4RuntimeTensorMapBytes(
                mtpAttention.indexerTailKeyTensors);
            tensorBytes += Qwen4RuntimeTensorMapBytes(
                mtpAttention.indexerTailPositionTensors);
            tensorBytes += Qwen4RuntimeTensorMapBytes(
                mtpAttention.indexerBlockKeyTensors);
            stateBytes += Qwen4FloatVectorMapBytes(
                mtpAttention.indexerRawKeys);
            stateBytes += Qwen4FloatVectorMapBytes(
                mtpAttention.indexerPositions);
            stateBytes += Qwen4FloatVectorMapBytes(
                mtpAttention.indexerBlockKeys);
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
        }

        std::shared_ptr<PrefixSnapshot> snapshot(new PrefixSnapshot());
        snapshot->cachedLen = cachedLen;
        snapshot->requestId = state.prefixRequestId;
        snapshot->tensorBytes = tensorBytes;
        snapshot->stateBytes = stateBytes;
        snapshot->tokens = state.processedTokens;
        snapshot->state = state;
        snapshot->state.mtpState = mtpSnapshotState;
        snapshot->state.mtpDisabled = mtpSnapshotState == nullptr;
        snapshot->state.borrowedPrefixSnapshot.reset();
        snapshot->state.pleConvHistoryTensors[0].reset();
        snapshot->state.pleConvHistoryTensors[1].reset();
        snapshot->state.pleConvHistoryIndex = 0;
        snapshot->state.indexerHostMirrorTransfers.clear();
        snapshot->state.indexerTailKeyTensors.clear();
        snapshot->state.indexerTailPositionTensors.clear();
        snapshot->state.indexerBlockKeyTensors.clear();
        snapshot->state.geometricCacheGrowthReadyLayers.clear();
        snapshot->state.lastPrefixSnapshotLen = cachedLen;
        snapshot->layers.resize(this->block_cnt);

        for (int layer = 0; layer < this->block_cnt; layer++) {
            PrefixLayerSnapshot &layerSnapshot = snapshot->layers[layer];
            layerSnapshot.linear = this->IsLinearAttentionLayer(layer);
            layerSnapshot.firstDevice =
                pastKeyValues[layer].first.dataDevice;
            layerSnapshot.firstDeviceIds =
                pastKeyValues[layer].first.dataDeviceIds;
            layerSnapshot.secondDevice =
                pastKeyValues[layer].second.dataDevice;
            layerSnapshot.secondDeviceIds =
                pastKeyValues[layer].second.dataDeviceIds;
            if (!Qwen4CopyCompactCacheTensor(
                    pastKeyValues[layer].first,
                    layerSnapshot.first) ||
                !Qwen4CopyCompactCacheTensor(
                    pastKeyValues[layer].second,
                    layerSnapshot.second)) {
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
                 (!Qwen4CopyCompactQsaTensor(
                      *tailKeys->second,
                      layerSnapshot.qsaTailKeys) ||
                  !Qwen4CopyCompactQsaTensor(
                      *tailPositions->second,
                      layerSnapshot.qsaTailPositions))) ||
                (blockCount > 0 &&
                 !Qwen4CopyCompactQsaTensor(
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
        if (snapshot->state.mtpState == nullptr &&
            MtpSupportsGenerationConfig(context->generationConfig)) {
            // A snapshot recorded while MTP was disabled cannot reconstruct
            // the missing draft KV state. Retire it and recompute once; the
            // current MTP request will then record an MTP-capable snapshot.
            std::lock_guard<std::mutex> guard(this->prefixCacheMutex);
            this->prefixSnapshots.erase(
                std::remove(this->prefixSnapshots.begin(),
                            this->prefixSnapshots.end(), snapshot),
                this->prefixSnapshots.end());
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
        if (snapshot->state.mtpState != nullptr) {
            restored.mtpState = CloneMtpPrefixState(
                *snapshot->state.mtpState, snapshot->cachedLen);
            if (restored.mtpState == nullptr) {
                return false;
            }
            restored.mtpDisabled = false;
        } else {
            restored.mtpState.reset();
            restored.mtpDisabled = true;
        }
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
    std::vector<int> Qwen4ExpModel::ForwardMultimodal(
            const Data &inputIds, const Data &attentionMask,
            const Data &positionIds,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            const std::map<std::string, std::vector<Data *>> &multimodalInput,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens,
            std::vector<std::vector<float> *> *retLogits) {
        AssertInFastLLM(
            inputIds.dims.size() == 2 && inputIds.dims[0] == 1,
            "Qwen4-Exp multimodal inference supports one request at a time.");
        AssertInFastLLM(
            (int)pastKeyValues.size() >= this->block_cnt,
            "Qwen4-Exp multimodal inference received too few cache slots.");

        // The payload remains attached to the response context during decode.
        // Only the first call has an empty target cache and needs the visual
        // encoder; later calls use the stored M-RoPE delta with normal text
        // decode.
        if (!pastKeyValues.empty() &&
            !pastKeyValues[0].second.dims.empty()) {
            Data adjustedPositionIds;
            auto delta = multimodalInput.find("mrope_position_delta");
            if (delta != multimodalInput.end() && !delta->second.empty() &&
                delta->second[0] != nullptr) {
                AdjustPositionIdsWithDelta(
                    positionIds, *delta->second[0], adjustedPositionIds);
            } else {
                adjustedPositionIds.CopyFrom(positionIds);
            }
            std::vector<float> *logits = nullptr;
            if (retLogits != nullptr && !retLogits->empty()) {
                logits = (*retLogits)[0];
            }
            return {Forward(inputIds, attentionMask, adjustedPositionIds,
                            pastKeyValues, generationConfig,
                            lastTokens, logits)};
        }

        PrepareWeights();
        RequestState *requestState;
        {
            std::lock_guard<std::mutex> guard(this->stateMutex);
            requestState = &this->requestStates[&pastKeyValues[0].first];
            // Draft state cannot be reconstructed from the visual encoder's
            // replacement embeddings. Keep this request on exact target-model
            // decoding; ordinary text requests retain MTP.
            requestState->mtpDisabled = true;
            requestState->mtpState.reset();
        }

        auto &mutableInputs = const_cast<
            std::map<std::string, std::vector<Data *>> &>(multimodalInput);
        auto imageFrames = multimodalInput.find("image_frames");
        auto videoFrames = multimodalInput.find("video_frames");
        auto imageGrid = multimodalInput.find("image_grid_thw");
        auto videoGrid = multimodalInput.find("video_grid_thw");
        const bool hasRawMedia =
            (imageFrames != multimodalInput.end() &&
             !imageFrames->second.empty()) ||
            (videoFrames != multimodalInput.end() &&
             !videoFrames->second.empty());

        Data imageFeatures, videoFeatures;
        const Data *imageEmbeds = nullptr;
        const Data *videoEmbeds = nullptr;
        auto modalityTypes = multimodalInput.find("mm_token_type_ids");
        auto mropePositions = multimodalInput.find("mrope_position_ids");
        if (hasRawMedia) {
            std::vector<std::vector<int>> imageGrids, videoGrids;
            EncodeVisualItems(
                imageFrames == multimodalInput.end()
                    ? std::vector<Data *>() : imageFrames->second,
                imageGrid != multimodalInput.end() &&
                        !imageGrid->second.empty()
                    ? imageGrid->second[0] : nullptr,
                false, imageFeatures, imageGrids);
            EncodeVisualItems(
                videoFrames == multimodalInput.end()
                    ? std::vector<Data *>() : videoFrames->second,
                videoGrid != multimodalInput.end() &&
                        !videoGrid->second.empty()
                    ? videoGrid->second[0] : nullptr,
                true, videoFeatures, videoGrids);

            Data computedTypes, computedPositions, computedDelta;
            BuildMultimodalPositionData(
                inputIds, imageGrids, videoGrids,
                computedTypes, computedPositions, computedDelta);
            mutableInputs["mm_token_type_ids"].clear();
            mutableInputs["mrope_position_ids"].clear();
            mutableInputs["mrope_position_delta"].clear();
            mutableInputs["mm_token_type_ids"].push_back(
                new Data(computedTypes));
            mutableInputs["mrope_position_ids"].push_back(
                new Data(computedPositions));
            mutableInputs["mrope_position_delta"].push_back(
                new Data(computedDelta));
            modalityTypes = mutableInputs.find("mm_token_type_ids");
            mropePositions = mutableInputs.find("mrope_position_ids");
            imageEmbeds = imageFeatures.dims.empty()
                ? nullptr : &imageFeatures;
            videoEmbeds = videoFeatures.dims.empty()
                ? nullptr : &videoFeatures;
        } else {
            auto images = multimodalInput.find("image_embeds");
            if (images != multimodalInput.end() && !images->second.empty()) {
                imageEmbeds = images->second[0];
            }
            auto videos = multimodalInput.find("video_embeds");
            if (videos != multimodalInput.end() && !videos->second.empty()) {
                videoEmbeds = videos->second[0];
            }
        }
        AssertInFastLLM(
            modalityTypes != mutableInputs.end() &&
                !modalityTypes->second.empty() &&
                mropePositions != mutableInputs.end() &&
                !mropePositions->second.empty(),
            "Qwen4-Exp multimodal inference needs modality and M-RoPE positions.");

        Data embeddingResult, mergedEmbedding;
        Data &embeddingWeight =
            this->weight[languagePrefix + "embed_tokens.weight"];
        Embedding(inputIds, embeddingWeight, embeddingResult);
        mergedEmbedding.CopyFrom(embeddingResult);
        MergeMultimodalFeaturesIntoText(
            *modalityTypes->second[0], imageEmbeds, videoEmbeds,
            mergedEmbedding);
        if (embeddingWeight.dataDevice != DataDevice::CPU) {
            mergedEmbedding.ToDevice(
                embeddingWeight.dataDevice, embeddingWeight.dataDeviceIds);
        }

        Data mropePositionIds(*mropePositions->second[0]);
        mropePositionIds.ToDevice(DataDevice::CPU);
        if (mropePositionIds.dataType != DataType::FLOAT32) {
            ToDataType(mropePositionIds, DataType::FLOAT32);
            mropePositionIds.ToDevice(DataDevice::CPU);
        }
        return ForwardTarget(
            1, inputIds, attentionMask, mropePositionIds,
            pastKeyValues, generationConfig, lastTokens, retLogits,
            nullptr, nullptr, nullptr, nullptr,
            false, false, false, nullptr, false, &mergedEmbedding);
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
        // Target verification may have advanced several accepted inputs at
        // once. The ordinary scheduler still consumes one output per call;
        // drain those already-verified outputs without touching model state.
        if (requestState->mtpState != nullptr &&
            !requestState->mtpState->pendingOutputTokens.empty()) {
            const int token =
                requestState->mtpState->pendingOutputTokens.front();
            requestState->mtpState->pendingOutputTokens.pop_front();
            return {token};
        }

        if (requestState->mtpDisabled ||
            !MtpSupportsGenerationConfig(generationConfig)) {
            if (requestState->mtpState != nullptr) {
                requestState->mtpState->proposals.clear();
                requestState->mtpState->targetCheckpointPrepared = false;
            }
#if defined(USE_CUDA) && !defined(USE_ROCM)
            void *decodePolicy = nullptr;
            if (Qwen4MtpDraftsPerStep() == 0 && inputIds.dims[1] == 1 &&
                Qwen4CudaOnlyDeviceMap(this->deviceMap) &&
                !this->weights.empty() && !this->weights[0].empty()) {
                decodePolicy = FastllmCudaBeginMoeDecode(
                    this->weights[0].data(), this->weights[0].size(), this->num_experts_per_tok);
            }
#endif
            auto result = ForwardTarget(
                batch, inputIds, attentionMask, positionIds,
                pastKeyValues, generationConfig, lastTokens, retLogits,
                nullptr, nullptr, nullptr, nullptr,
                true, true, false);
#if defined(USE_CUDA) && !defined(USE_ROCM)
            FastllmCudaEndMoeDecode(decodePolicy);
#endif
            return result;
        }

        if (requestState->mtpState == nullptr) {
            requestState->mtpState = std::make_shared<MtpRuntimeState>();
        }
        MtpRuntimeState &mtp = *requestState->mtpState;
        const int draftCount = Qwen4MtpDraftsPerStep();

        auto dataToInts = [](const Data &source) {
            Data cpu;
            ToDataType(source, cpu, DataType::FLOAT32);
            cpu.ToDevice(DataDevice::CPU);
            const float *values =
                reinterpret_cast<const float *>(cpu.cpuData);
            std::vector<int> result(cpu.Count(0));
            for (size_t i = 0; i < result.size(); i++) {
                result[i] = (int)(values[i] + 0.01f);
            }
            return result;
        };

        auto resizeMtpCache = [](Data &cache, int length) {
            if (cache.dims.size() < 2) {
                return;
            }
            std::vector<int> dims = cache.dims;
            dims[1] = length;
            cache.Resize(dims);
        };

        auto prefixSnapshotDue = [&]() {
            int cachedLen = 0;
            if (!ShouldRecordPrefixSnapshot(
                    pastKeyValues, *requestState, cachedLen)) {
                return false;
            }
            std::lock_guard<std::mutex> guard(this->prefixCacheMutex);
            return FindPrefixSnapshotLocked(
                requestState->processedTokens,
                cachedLen, cachedLen) == nullptr;
        };

        auto generateProposalChain = [&](const Data &targetHidden,
                                         const std::vector<int> &tokens,
                                         const std::vector<int> &positions) {
            mtp.proposals.clear();
            std::vector<Data> hiddenStates(2);
            int currentHidden = 0;
            Data &sampledTokenIds = mtp.sampledTokenIds;
            Data &sampledTokenValues = mtp.sampledTokenValues;
            bool deviceChain = false;
#ifdef USE_CUDA
            Data &tokenEmbedding = this->weight[
                languagePrefix + "embed_tokens.weight"];
            if (draftCount > 0 &&
                targetHidden.dataDevice == DataDevice::CUDA &&
                targetHidden.cudaData != nullptr &&
                tokenEmbedding.dataDevice == DataDevice::CUDA &&
                tokenEmbedding.cudaData != nullptr) {
                const int device = targetHidden.dataDeviceIds.empty()
                    ? FastllmCudaGetDevice()
                    : targetHidden.dataDeviceIds[0];
                deviceChain = device >= 0 &&
                    Qwen4PrepareDecodeGraphWorkspace(
                        sampledTokenIds, DataType::INT32,
                        {draftCount}, device) &&
                    Qwen4PrepareDecodeGraphWorkspace(
                        sampledTokenValues, DataType::FLOAT32,
                        {draftCount}, device);
            }
#endif
            const int first = RunMtpDraft(
                mtp, targetHidden, tokens, positions,
                &hiddenStates[currentHidden], true, nullptr,
                deviceChain ? &sampledTokenIds : nullptr,
                deviceChain ? &sampledTokenValues : nullptr, 0);
            if (draftCount <= 1) {
                if (first >= 0) {
                    mtp.proposals.push_back(first);
                } else {
#ifdef USE_CUDA
                    mtp.proposals.resize(1);
                    FastllmCudaCopyFromDeviceToHost(
                        mtp.proposals.data(), sampledTokenIds.cudaData,
                        sizeof(int));
#endif
                }
                return;
            }

            const int committedKeyLength =
                mtp.key.dims.size() > 1 ? mtp.key.dims[1] : 0;
            const int committedValueLength =
                mtp.value.dims.size() > 1 ? mtp.value.dims[1] : 0;
            RequestRuntimeCheckpoint committedState;
            CaptureRequestRuntimeCheckpoint(
                mtp.attentionState,
                {{this->block_cnt, committedKeyLength}},
                committedState);
            bool committedStateRestored = false;
            auto restoreCommittedDraftState = [&]() {
                resizeMtpCache(mtp.key, committedKeyLength);
                resizeMtpCache(mtp.value, committedValueLength);
                RestoreRequestRuntimeCheckpoint(
                    mtp.attentionState, committedState);
                committedStateRestored = true;
            };
            auto finishHostChain = [&](int beginDraft, int previous,
                                       int nextPosition) {
                for (int draft = beginDraft;
                     draft < draftCount; draft++) {
                    const int nextHidden = 1 - currentHidden;
                    const int token = RunMtpDraft(
                        mtp, hiddenStates[currentHidden],
                        {previous}, {nextPosition},
                        &hiddenStates[nextHidden], true);
                    mtp.proposals.push_back(token);
                    currentHidden = nextHidden;
                    previous = token;
                    nextPosition++;
                }
            };

            if (first >= 0) {
                // CPU and mixed-device fallback retain the established
                // per-token sampling behavior, while alternating hidden
                // buffers avoids the old inter-draft D2D copies.
                mtp.proposals.push_back(first);
                finishHostChain(1, first, positions.back() + 1);
            } else {
#ifdef USE_CUDA
                bool completedOnDevice = true;
                int nextPosition = positions.back() + 1;
                for (int draft = 1; draft < draftCount; draft++) {
                    Data deviceToken;
                    deviceToken.FakeFrom(
                        sampledTokenValues,
                        (size_t)(draft - 1) * sizeof(float));
                    deviceToken.Resize({1, 1});
                    deviceToken.dataDeviceIds =
                        sampledTokenValues.dataDeviceIds;
                    const int nextHidden = 1 - currentHidden;
                    const int token = RunMtpDraft(
                        mtp, hiddenStates[currentHidden],
                        {0}, {nextPosition},
                        &hiddenStates[nextHidden], true,
                        &deviceToken, &sampledTokenIds,
                        &sampledTokenValues, draft);
                    currentHidden = nextHidden;
                    nextPosition++;
                    if (token >= 0) {
                        // A backend that cannot provide device-side greedy
                        // output can still complete correctly without
                        // discarding the already-computed prefix.
                        mtp.proposals.resize(draft);
                        FastllmCudaCopyFromDeviceToHost(
                            mtp.proposals.data(),
                            sampledTokenIds.cudaData,
                            (size_t)draft * sizeof(int));
                        mtp.proposals.push_back(token);
                        finishHostChain(
                            draft + 1, token, nextPosition);
                        completedOnDevice = false;
                        break;
                    }
                }
                if (completedOnDevice) {
                    // Queue the speculative-state rollback before the one
                    // synchronizing D2H.  The rollback does not depend on
                    // proposal values, and the shared CUDA stream preserves
                    // device ordering, so its host-side bookkeeping overlaps
                    // the final draft head instead of forming a serial gap.
                    restoreCommittedDraftState();
                    mtp.proposals.resize(draftCount);
                    FastllmCudaCopyFromDeviceToHost(
                        mtp.proposals.data(),
                        sampledTokenIds.cudaData,
                        (size_t)draftCount * sizeof(int));
                }
#else
                AssertInFastLLM(
                    false,
                    "Qwen4-Exp MTP returned a device token without CUDA.");
#endif
            }
            if (!committedStateRestored) {
                restoreCommittedDraftState();
            }
        };

        if (mtp.proposals.empty()) {
            Data targetHidden;
            std::vector<int> result = ForwardTarget(
                batch, inputIds, attentionMask, positionIds,
                pastKeyValues, generationConfig, lastTokens, retLogits,
                &targetHidden, nullptr, nullptr, nullptr,
                false, false, false);

            const std::vector<int> ids = dataToInts(inputIds);
            const std::vector<int> currentPositions =
                dataToInts(positionIds);
            const int sequence = inputIds.dims[1];
            const bool finalPromptChunk =
                generationConfig.input_token_length <= 0 ||
                (int)requestState->processedTokens.size() >=
                    generationConfig.input_token_length;
            std::vector<int> mtpTokens;
            std::vector<int> mtpPositions;
            Data pairedHidden;
            bool hasPairedHidden = false;
            auto appendHidden = [&](const Data &part) {
                if (part.dims.empty() || part.dims[1] <= 0) {
                    return;
                }
                if (!hasPairedHidden) {
                    pairedHidden.CopyFrom(part);
                    hasPairedHidden = true;
                    return;
                }
                Data combined;
                Cat(pairedHidden, part, 1, combined);
                pairedHidden.CopyFrom(combined);
            };

            if (mtp.hasDeferredTargetHidden) {
                appendHidden(mtp.deferredTargetHidden);
                mtpTokens.push_back(ids.front());
                mtpPositions.push_back(mtp.deferredPosition);
                mtp.hasDeferredTargetHidden = false;
                mtp.deferredTargetHidden = Data();
            }

            const int pairedCurrentRows =
                sequence - (finalPromptChunk ? 0 : 1);
            if (pairedCurrentRows > 0) {
                Data currentPart;
                Split(targetHidden, 1, 0, pairedCurrentRows, currentPart);
                appendHidden(currentPart);
                for (int row = 0; row < pairedCurrentRows; row++) {
                    mtpTokens.push_back(
                        row + 1 < sequence
                            ? ids[row + 1]
                            : result.back());
                    mtpPositions.push_back(currentPositions[row]);
                }
            }

            if (!finalPromptChunk) {
                Split(targetHidden, 1, sequence - 1, sequence,
                      mtp.deferredTargetHidden);
                mtp.deferredPosition = currentPositions.back();
                mtp.hasDeferredTargetHidden = true;
                if (hasPairedHidden) {
                    RunMtpDraft(
                        mtp, pairedHidden, mtpTokens, mtpPositions,
                        nullptr, false);
                }
                // Prefix snapshot materialization can synchronize and
                // rematerialize target recurrent state.  Keep it after the
                // independent draft state has consumed this prompt chunk so
                // snapshot bookkeeping cannot perturb prompt alignment.
                MaybeRecordPrefixSnapshot(pastKeyValues, *requestState);
                return result;
            }

            AssertInFastLLM(
                hasPairedHidden && !mtpTokens.empty(),
                "Qwen4-Exp MTP failed to align its prompt states.");
            if (prefixSnapshotDue()) {
                const int pairedRows = (int)mtpTokens.size();
                AssertInFastLLM(
                    pairedRows > 0 &&
                    pairedHidden.dims.size() == 3 &&
                    pairedHidden.dims[1] == pairedRows,
                    "Qwen4-Exp MTP prefix snapshot boundary is misaligned.");
                if (pairedRows > 1) {
                    Data prefixHidden;
                    Split(pairedHidden, 1, 0, pairedRows - 1,
                          prefixHidden);
                    RunMtpDraft(
                        mtp, prefixHidden,
                        std::vector<int>(mtpTokens.begin(),
                                         mtpTokens.end() - 1),
                        std::vector<int>(mtpPositions.begin(),
                                         mtpPositions.end() - 1),
                        nullptr, false);
                }
                Data finalPairedHidden;
                Split(pairedHidden, 1, pairedRows - 1, pairedRows,
                      finalPairedHidden);
                mtp.deferredTargetHidden.CopyFrom(finalPairedHidden);
                mtp.deferredPosition = mtpPositions.back();
                mtp.hasDeferredTargetHidden = true;
                MaybeRecordPrefixSnapshot(pastKeyValues, *requestState);
                mtp.hasDeferredTargetHidden = false;
                mtp.deferredPosition = -1;
                mtp.deferredTargetHidden = Data();
                generateProposalChain(
                    finalPairedHidden, {mtpTokens.back()},
                    {mtpPositions.back()});
            } else {
                generateProposalChain(
                    pairedHidden, mtpTokens, mtpPositions);
                MaybeRecordPrefixSnapshot(
                    pastKeyValues, *requestState);
            }
#ifdef USE_CUDA
            // Proposal state can be produced by executor workers with
            // different per-thread streams. Publish it before the scheduler
            // starts verification on a later host worker.
            ForceDeviceSync();
#endif
            return result;
        }

        // A proposal is created only after the final prompt token, so target
        // verification always starts from one scheduler-provided anchor.
        if (inputIds.dims[1] != 1) {
            mtp.proposals.clear();
            mtp.targetCheckpointPrepared = false;
            return ForwardTarget(
                batch, inputIds, attentionMask, positionIds,
                pastKeyValues, generationConfig, lastTokens, retLogits,
                nullptr, nullptr, nullptr, nullptr,
                true, true, false);
        }

        const int anchor = dataToInts(inputIds).front();
        const int firstPosition = dataToInts(positionIds).front();
        std::vector<int> candidateTokens;
        candidateTokens.reserve(1 + mtp.proposals.size());
        candidateTokens.push_back(anchor);
        candidateTokens.insert(candidateTokens.end(),
                               mtp.proposals.begin(),
                               mtp.proposals.end());
        std::vector<int> candidatePositions(candidateTokens.size());
        for (int i = 0; i < (int)candidatePositions.size(); i++) {
            candidatePositions[i] = firstPosition + i;
        }
        std::vector<float> candidateTokenValues(candidateTokens.begin(),
                                                candidateTokens.end());
        std::vector<float> candidatePositionValues(
            candidatePositions.begin(), candidatePositions.end());
        Data candidateIds(DataType::FLOAT32,
                          {1, (int)candidateTokens.size()},
                          candidateTokenValues);
        Data candidatePositionIds(
            DataType::FLOAT32,
            {1, (int)candidatePositions.size()},
            candidatePositionValues);

        if (mtp.targetCheckpointPrepared) {
            // The preceding verifier captured this exact committed target
            // state before it started the independent MTP draft chain.
            mtp.targetCheckpointPrepared = false;
        } else {
            CaptureTargetRuntimeCheckpoint(
                pastKeyValues, *requestState, mtp.targetCheckpoint);
        }
        Data candidateMask;
        Data targetHidden;
        std::vector<int> targetTokens;
        std::vector<unsigned char> verificationAccepted;
        mtp.targetCapture.runtimeCheckpoint = &mtp.targetCheckpoint;
        ForwardTarget(
            1, candidateIds, candidateMask, candidatePositionIds,
            pastKeyValues, generationConfig, lastTokens, nullptr,
            &targetHidden, &targetTokens, &verificationAccepted,
            &mtp.targetCapture,
            false, true, true, &candidateTokens, true);

        int acceptedDrafts = 0;
        const bool sampledVerification =
            !generationConfig.IsSimpleGreedy();
        AssertInFastLLM(
            !sampledVerification ||
                verificationAccepted.size() == mtp.proposals.size(),
            "Qwen4-Exp MTP rejection sampling returned an invalid shape.");
        while (acceptedDrafts < (int)mtp.proposals.size()) {
            const bool accepted = sampledVerification
                ? verificationAccepted[acceptedDrafts] != 0
                : targetTokens[acceptedDrafts] ==
                    mtp.proposals[acceptedDrafts];
            if (!accepted) {
                break;
            }
            acceptedDrafts++;
        }
        const int committedInputs = acceptedDrafts + 1;
        std::vector<int> verifiedOutputs;
        verifiedOutputs.reserve(committedInputs);
        for (int i = 0; i < acceptedDrafts; i++) {
            verifiedOutputs.push_back(mtp.proposals[i]);
        }
        verifiedOutputs.push_back(targetTokens[acceptedDrafts]);

        if (committedInputs < (int)candidateTokens.size()) {
            CommitTargetVerificationPrefix(
                candidateIds, candidatePositionIds, candidateTokens,
                committedInputs,
                pastKeyValues, *requestState, mtp.targetCheckpoint,
                mtp.targetCapture);
        } else if (mtp.targetCapture.runtimeCheckpoint ==
                   &mtp.targetCheckpoint) {
            // The verifier intentionally left the committed baseline in the
            // live recurrent caches.  A fully accepted batch needs its final
            // candidate state; commit all independent layers with one copy.
            CommitTargetRecurrentState(
                pastKeyValues, mtp.targetCheckpoint);
        }
        Data committedTargetHidden;
        if (committedInputs == (int)candidateTokens.size()) {
            committedTargetHidden.CopyFrom(targetHidden);
        } else {
            Split(targetHidden, 1, 0, committedInputs,
                  committedTargetHidden);
        }
        // The target state is final for this verifier and remains untouched by
        // RunMtpDraft, which owns separate KV/recurrent/PLE state. Queue the
        // next checkpoint now on the current per-thread stream. The draft
        // chain's mandatory device-to-host completion below publishes both
        // operations before another scheduler worker can consume it.
        CaptureTargetRuntimeCheckpoint(
            pastKeyValues, *requestState, mtp.targetCheckpoint, false,
            false);
        mtp.targetCheckpointPrepared = true;
        generateProposalChain(
            committedTargetHidden, verifiedOutputs,
            std::vector<int>(candidatePositions.begin(),
                             candidatePositions.begin() + committedInputs));
#ifdef USE_CUDA
        // The next scheduler call consumes both the proposal tokens and all
        // draft-side recurrent/KV/PLE state.
        ForceDeviceSync();
#endif
        mtp.pendingOutputTokens.clear();
        for (int i = 1; i < (int)verifiedOutputs.size(); i++) {
            mtp.pendingOutputTokens.push_back(verifiedOutputs[i]);
        }
        return {verifiedOutputs.front()};
    }

    std::vector<int> Qwen4ExpModel::ForwardTarget(
            int batch,
            const Data &inputIds,
            const Data &attentionMask,
            const Data &positionIds,
            std::vector<std::pair<Data, Data>> &pastKeyValues,
            const GenerationConfig &generationConfig,
            const LastTokensManager &lastTokens,
            std::vector<std::vector<float> *> *retLogits,
            Data *targetMultiHidden,
            std::vector<int> *allVerificationTokens,
            std::vector<unsigned char> *verificationAccepted,
            TargetVerificationCapture *verificationCapture,
            bool recordPrefixSnapshot,
            bool allowDecodeCudaGraph,
            bool decodeEquivalentGdn,
            const std::vector<int> *hostInputTokens,
            bool materializeCausalMaskOnGraphFallback,
            const Data *precomputedEmbedding) {
        Qwen4MtpGdnModeGuard mtpGdnMode(decodeEquivalentGdn);
        AssertInFastLLM(batch == 1 && inputIds.dims.size() == 2 &&
                        inputIds.dims[0] == 1,
                        "Qwen4-Exp currently runs one request per Forward call.");
        AssertInFastLLM((int)pastKeyValues.size() >= this->block_cnt,
                        "Qwen4-Exp received too few cache slots.");
        PrepareWeights();
#ifdef USE_CUDA
        // A verifier batch is still decode: keep every small FP16 projection
        // on the row-independent native path.  Besides preserving the
        // one-token reduction tree, that specialization replaces the last
        // five CTA barriers with the equivalent warp-shuffle tree.
        const int previousLinearExactBatchThreshold =
            FastllmCudaGetLinearExactBatchThreshold();
        if (verificationCapture != nullptr && inputIds.dims[1] > 1 &&
            inputIds.dims[1] <= 8) {
            FastllmCudaSetLinearExactBatchThreshold(std::max(
                previousLinearExactBatchThreshold,
                inputIds.dims[1] + 1));
        }
        struct LinearExactBatchThresholdRestore {
            int previous;
            ~LinearExactBatchThresholdRestore() {
                FastllmCudaSetLinearExactBatchThreshold(previous);
            }
        } linearExactBatchThresholdRestore{
            previousLinearExactBatchThreshold};
#endif
        if (verificationCapture != nullptr) {
            verificationCapture->linearConvInputs.resize(this->block_cnt);
            verificationCapture->linearAlphas.resize(this->block_cnt);
            verificationCapture->linearBetas.resize(this->block_cnt);
            for (int layer = 0; layer < this->block_cnt; layer++) {
                if (!this->IsLinearAttentionLayer(layer)) {
                    Data &rawKeys = verificationCapture->qsaRawKeys[layer];
                    if (!rawKeys.dims.empty()) {
                        rawKeys.Resize({0, this->indexerHeadDim});
                    }
                }
            }
        }

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
        if (precomputedEmbedding != nullptr) {
            AssertInFastLLM(
                precomputedEmbedding->dims ==
                    std::vector<int>({batch, inputIds.dims[1],
                                      this->embed_dim}),
                "Qwen4-Exp precomputed embedding shape is invalid.");
            embedding.CopyFrom(*precomputedEmbedding);
        } else {
            Embedding(
                inputIds,
                this->weight[languagePrefix + "embed_tokens.weight"],
                embedding);
        }
        embedding.Reshape({batch, inputIds.dims[1], 1, this->embed_dim});
        Repeat(embedding, 2, this->hcCount, *hiddenStates);
        hiddenStates->Reshape({batch, inputIds.dims[1],
                               this->hcCount * this->embed_dim});
        DumpTensorIfRequested("embedding", *hiddenStates);

        Data carriedAttentionNorm, carriedAttentionProjection;
        Data finalHyperNorm;
        bool hasCarriedAttentionNorm = false;
        bool hasCarriedAttentionProjection = false;
        bool hasFinalHyperNorm = false;
        bool usedDecodeCudaGraph = false;
        auto prefillProjectionStorage = [&](
                const Data &reference, const std::string &prefix,
                Data &storage) -> Data* {
#ifdef USE_CUDA
            const Data &downWeight =
                this->weight[prefix + "input_mix_weight_down.weight"];
            const Data &injectionWeight =
                this->weight[prefix + "block_inject_weight.weight"];
            if (verificationCapture == nullptr && inputIds.dims[1] >= 8 &&
                reference.dataDevice == DataDevice::CUDA &&
                reference.dataType == DataType::FLOAT32 &&
                downWeight.dataType == DataType::FLOAT16 &&
                injectionWeight.dataType == DataType::FLOAT16) {
                return &storage;
            }
#endif
            return nullptr;
        };
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
        std::unique_ptr<Data> fallbackCausalMask;
        const Data *effectiveAttentionMask = &attentionMask;
        auto ensureFallbackCausalMask = [&]() {
            if (!materializeCausalMaskOnGraphFallback ||
                !effectiveAttentionMask->dims.empty()) {
                return;
            }
            fallbackCausalMask = std::make_unique<Data>(
                Qwen4BuildCausalMask(
                    qsaPreviousLength, inputIds.dims[1]));
            effectiveAttentionMask = fallbackCausalMask.get();
        };
        const bool qsaDeviceCompatibleMask =
            materializeCausalMaskOnGraphFallback ||
            attentionMask.dims.empty() ||
            ((restoredPrefixSnapshot || verificationCapture != nullptr) &&
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
                if (verificationCapture != nullptr) {
                    verificationCapture->pleInput.CopyFrom(*hiddenStates);
                }
                RunPLE(*hiddenStates, inputIds, *requestState, pleOutput,
                       hostInputTokens);
                AddTo(*hiddenStates, pleOutput);
                DumpTensorIfRequested("layer_" + std::to_string(layer) +
                                      "_ple", *hiddenStates);
                carriedAttentionNorm.FreeSpace();
                carriedAttentionProjection.FreeSpace();
                hasCarriedAttentionNorm = false;
                hasCarriedAttentionProjection = false;

                // PLE depends on layer-0's residual and its ngram lookup can
                // remain on the host, so this is the earliest stable CUDA
                // Graph boundary.  From here the standard linear-attention,
                // MoE and graph-safe full-attention operations can share the
                // existing whole-backbone graph.
                if (allowDecodeCudaGraph &&
                    (inputIds.dims[1] == 1 ||
                     verificationCapture != nullptr) &&
                    GetFastllmEnv().cudaGraph) {
                    Data unusedAttention;
                    const Data &graphPositionIds =
                        positionsFor(*hiddenStates);
                    if (TryRunDecodeCudaGraphBackbone(
                            layer, true, *hiddenStates,
                            unusedAttention, unusedAttention,
                            *effectiveAttentionMask, graphPositionIds,
                            qsaDeviceCompatibleMask,
                            pastKeyValues, *requestState, logits,
                            targetMultiHidden, verificationCapture)) {
                        usedDecodeCudaGraph = true;
                        break;
                    }
                    ensureFallbackCausalMask();
                }
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
            Data *attentionProjection =
                hasCarriedAttentionProjection
                    ? &carriedAttentionProjection : nullptr;
            HyperMixNormalized(*normalizedAttention, attentionHyperPrefix,
                               attentionInput, &attentionInjection,
                               attentionProjection);
            normalizedAttention->FreeSpace();
            if (attentionProjection != nullptr) {
                carriedAttentionProjection.FreeSpace();
            }
            hasCarriedAttentionNorm = false;
            hasCarriedAttentionProjection = false;
            const Data *layerPositionIds = &positionIds;
            if (this->IsLinearAttentionLayer(layer)) {
                RunLinearAttention(layer, attentionInput,
                                   pastKeyValues[layer].first,
                                   pastKeyValues[layer].second,
                                   attentionOutput,
                                   verificationCapture == nullptr ? nullptr :
                                       &verificationCapture->linearConvInputs[
                                           layer],
                                   verificationCapture == nullptr ? nullptr :
                                       &verificationCapture->
                                           linearAlphas[layer],
                                   verificationCapture == nullptr ? nullptr :
                                       &verificationCapture->
                                           linearBetas[layer],
                                   verificationCapture == nullptr ||
                                       inputIds.dims[1] <= 1 ||
                                       verificationCapture->
                                           runtimeCheckpoint == nullptr ||
                                       layer >= (int)verificationCapture->
                                           runtimeCheckpoint->linearSecond.
                                               size()
                                       ? nullptr
                                       : &verificationCapture->
                                           runtimeCheckpoint->linearSecond[
                                               layer]);
            } else {
                ensureFallbackCausalMask();
                layerPositionIds = &positionsFor(attentionInput);
                RunFullAttention(layer, attentionInput,
                                 *effectiveAttentionMask,
                                 *layerPositionIds,
                                 qsaDeviceCompatibleMask,
                                 pastKeyValues[layer].first,
                                 pastKeyValues[layer].second,
                                 *requestState,
                                 attentionOutput,
                                 verificationCapture == nullptr ? nullptr :
                                     &verificationCapture->qsaRawKeys[layer]);
            }
            DumpTensorIfRequested("layer_" + std::to_string(layer) +
                                  "_attention", attentionOutput);
            if (allowDecodeCudaGraph &&
                !this->IsLinearAttentionLayer(layer) &&
                    layer == this->kvCacheId &&
                    TryRunDecodeCudaGraphBackbone(
                        layer, false, *hiddenStates, attentionOutput,
                        attentionInjection, *effectiveAttentionMask,
                    *layerPositionIds,
                    qsaDeviceCompatibleMask,
                    pastKeyValues, *requestState, logits,
                    targetMultiHidden, verificationCapture)) {
                usedDecodeCudaGraph = true;
                break;
            }
            const std::string mlpHyperPrefix =
                layerPrefix + "mlp_hyper_connection.";
            Data mlpNorm, mlpProjection;
            Data *mlpProjectionStorage =
                prefillProjectionStorage(
                    *hiddenStates, mlpHyperPrefix, mlpProjection);
            HyperCombineRMSNorm(
                *hiddenStates, attentionOutput, attentionInjection,
                this->weight[mlpHyperPrefix + "hc_norm.weight"],
                *nextHiddenStates, mlpNorm, mlpProjectionStorage);
            std::swap(hiddenStates, nextHiddenStates);

            Data mlpInput, mlpInjection, mlpOutput;
            HyperMixNormalized(mlpNorm, mlpHyperPrefix,
                               mlpInput, &mlpInjection,
                               mlpProjectionStorage);
            mlpNorm.FreeSpace();
            mlpProjection.FreeSpace();
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
                Data *carriedProjectionStorage =
                    prefillProjectionStorage(
                        *hiddenStates, nextAttentionHyperPrefix,
                        carriedAttentionProjection);
                HyperCombineRMSNorm(
                    *hiddenStates, mlpOutput, mlpInjection,
                    this->weight[nextAttentionHyperPrefix +
                                 "hc_norm.weight"],
                    *nextHiddenStates, carriedAttentionNorm,
                    carriedProjectionStorage);
                hasCarriedAttentionNorm = true;
                hasCarriedAttentionProjection =
                    carriedProjectionStorage != nullptr;
            } else {
                // PLE changes the residual before the next attention norm, so
                // this single boundary cannot be normalized ahead of time.
                HyperCombine(*hiddenStates, mlpOutput, mlpInjection,
                             *nextHiddenStates);
                hasCarriedAttentionNorm = false;
                hasCarriedAttentionProjection = false;
            }
            std::swap(hiddenStates, nextHiddenStates);
            DumpTensorIfRequested("layer_" + std::to_string(layer) +
                                  "_output", *hiddenStates);
        }

        if (recordPrefixSnapshot) {
            MaybeRecordPrefixSnapshot(pastKeyValues, *requestState);
        }

        if (!usedDecodeCudaGraph) {
            Data finalHidden;
            AssertInFastLLM(hasFinalHyperNorm,
                            "Qwen4-Exp final hyper norm was not produced.");
            if (targetMultiHidden != nullptr) {
                targetMultiHidden->CopyFrom(*hiddenStates);
            }
            HyperMixNormalized(
                finalHyperNorm, languagePrefix + "hyper_connection_mixer.",
                finalHidden, nullptr);
            finalHyperNorm.FreeSpace();
            DumpTensorIfRequested("final_hidden", finalHidden);
            Data lastHidden;
            if (allVerificationTokens != nullptr) {
                lastHidden.CopyFrom(finalHidden);
            } else if (finalHidden.dims[1] > 1) {
                Split(finalHidden, 1, finalHidden.dims[1] - 1,
                      finalHidden.dims[1], lastHidden);
            } else {
                lastHidden.FakeFrom(finalHidden, 0);
                lastHidden.Resize(finalHidden.dims);
            }

            Linear(lastHidden, this->weight["lm_head.weight"], Data(), logits);
            ToDataType(logits, DataType::FLOAT32);
        }
        if (allVerificationTokens != nullptr &&
            generationConfig.output_token_least > 0) {
#ifdef USE_CUDA
            const int rows = inputIds.dims[1];
            std::vector<int> resetLengths(rows, 0);
            std::vector<int> eosIds = {this->eos_token_id};
            eosIds.insert(eosIds.end(), this->eos_token_ids.begin(),
                          this->eos_token_ids.end());
            eosIds.insert(eosIds.end(),
                          generationConfig.stop_token_ids.begin(),
                          generationConfig.stop_token_ids.end());
            std::vector<int> eosCounts(rows, (int)eosIds.size());
            std::vector<int> flattenedEosIds;
            flattenedEosIds.reserve((size_t)rows * eosIds.size());
            bool needsReset = false;
            for (int row = 0; row < rows; row++) {
                const int rowCacheLength = qsaPreviousLength + row + 1;
                resetLengths[row] =
                    generationConfig.output_token_least - rowCacheLength +
                    generationConfig.input_token_length;
                needsReset |= resetLengths[row] > 0;
                flattenedEosIds.insert(flattenedEosIds.end(),
                                       eosIds.begin(), eosIds.end());
            }
            if (needsReset) {
                ToDataType(logits, DataType::FLOAT32);
                FastllmResetLogitsOfEOS(
                    rows, &logits, resetLengths,
                    eosCounts, flattenedEosIds);
            }
#else
            ResetLogitsOfEOS(
                1, &logits, pastKeyValues, generationConfig);
#endif
        } else {
            ResetLogitsOfEOS(
                1, &logits, pastKeyValues, generationConfig);
        }
        DumpTensorIfRequested("logits", logits);

        if (generationConfig.output_logits && retLogits != nullptr &&
            !retLogits->empty() && (*retLogits)[0] != nullptr) {
            logits.ToDevice(DataDevice::CPU);
            (*retLogits)[0]->resize(logits.Count(0));
            std::memcpy((*retLogits)[0]->data(), logits.cpuData,
                        (size_t)logits.Count(0) * sizeof(float));
        }

        int token;
        if (allVerificationTokens != nullptr) {
            const int rows = inputIds.dims[1];
            AssertInFastLLM(
                rows > 1 && hostInputTokens != nullptr &&
                (int)hostInputTokens->size() == rows,
                "Qwen4-Exp MTP target verification input is incomplete.");
            if (!generationConfig.IsSimpleGreedy()) {
                AssertInFastLLM(
                    verificationAccepted != nullptr,
                    "Qwen4-Exp MTP rejection sampling has no acceptance output.");
#ifdef USE_CUDA
                AssertInFastLLM(
                    logits.dataDevice == DataDevice::CUDA &&
                    logits.dataType == DataType::FLOAT32 &&
                    logits.cudaData != nullptr && !logits.multiDeviceData &&
                    !logits.dims.empty() && logits.dims.back() > 0 &&
                    logits.Count(0) ==
                        (uint64_t)rows * logits.dims.back(),
                    "Qwen4-Exp MTP rejection sampling requires full CUDA logits.");
                const int device = logits.dataDeviceIds.empty()
                    ? FastllmCudaGetDevice() : logits.dataDeviceIds[0];
                AssertInFastLLM(
                    device >= 0,
                    "Qwen4-Exp MTP rejection sampling has no CUDA device.");
                Qwen4CudaDeviceGuard samplingDeviceGuard({device});
                std::vector<float> temperatures(
                    rows, std::max(generationConfig.temperature, 1.0e-6f));
                std::vector<int> topKs(
                    rows, std::max(1, generationConfig.top_k));
                std::vector<float> topPs(rows, generationConfig.top_p);
                const int candidateCount = rows - 1;
                std::vector<int> candidateIds(candidateCount);
                std::vector<int> candidateRows(candidateCount);
                for (int row = 0; row < candidateCount; row++) {
                    candidateIds[row] = (*hostInputTokens)[row + 1];
                    candidateRows[row] = row;
                }
                allVerificationTokens->assign(rows, -1);
                verificationAccepted->assign(candidateCount, 0);
                std::vector<int> recoveredIds(candidateCount, -1);
                AssertInFastLLM(
                    FastllmCudaTopKTopPSamplingWithTypicalAcceptance(
                        reinterpret_cast<float *>(logits.cudaData),
                        temperatures.data(), topKs.data(), topPs.data(),
                        allVerificationTokens->data(), rows,
                        logits.dims.back(), candidateIds.data(),
                        candidateRows.data(),
                        verificationAccepted->data(), recoveredIds.data(),
                        candidateCount,
                        QWEN4_MTP_TYPICAL_POSTERIOR_THRESHOLD,
                        QWEN4_MTP_TYPICAL_POSTERIOR_ALPHA),
                    "Qwen4-Exp CUDA MTP rejection sampling failed.");
                // Match Qwen3.5: accepted rows commit their draft token;
                // the first rejected row recovers with the target argmax.
                for (int row = 0; row < candidateCount; row++) {
                    (*allVerificationTokens)[row] = recoveredIds[row];
                }
#else
                AssertInFastLLM(
                    false,
                    "Qwen4-Exp MTP rejection sampling requires CUDA.");
#endif
            } else {
                if (verificationAccepted != nullptr) {
                    verificationAccepted->clear();
                }
                bool sampledOnCuda = false;
#ifdef USE_CUDA
                if (verificationCapture != nullptr && rows <= 8 &&
                    logits.dataDevice == DataDevice::CUDA &&
                    logits.dataType == DataType::FLOAT32 &&
                    logits.cudaData != nullptr && !logits.multiDeviceData &&
                    !logits.dims.empty() &&
                    logits.Count(0) ==
                        (uint64_t)rows * logits.dims.back()) {
                    const int device = logits.dataDeviceIds.empty()
                        ? FastllmCudaGetDevice() : logits.dataDeviceIds[0];
                    Data &tokenIds = verificationCapture->greedyTokenIds;
                    Data &tokenValues = verificationCapture->greedyTokenValues;
                    if (device >= 0 &&
                        Qwen4PrepareDecodeGraphWorkspace(
                            tokenIds, DataType::INT32, {rows}, device) &&
                        Qwen4PrepareDecodeGraphWorkspace(
                            tokenValues, DataType::FLOAT32, {rows}, device) &&
                        FastllmCudaGreedySamplingWithFloatOutput(
                            reinterpret_cast<float *>(logits.cudaData),
                            reinterpret_cast<int *>(tokenIds.cudaData),
                            reinterpret_cast<float *>(tokenValues.cudaData),
                            rows, logits.dims.back())) {
                        allVerificationTokens->resize(rows);
                        FastllmCudaCopyFromDeviceToHost(
                            allVerificationTokens->data(), tokenIds.cudaData,
                            (size_t)rows * sizeof(int));
                        sampledOnCuda = true;
                    }
                }
#endif
                if (!sampledOnCuda) {
                    Data top;
                    TopK(logits, top, 1);
                    top.ToDevice(DataDevice::CPU);
                    const float *topValues =
                        reinterpret_cast<const float *>(top.cpuData);
                    allVerificationTokens->resize(rows);
                    for (int row = 0; row < rows; row++) {
                        (*allVerificationTokens)[row] =
                            (int)(topValues[(size_t)row * 2] + 1e-3f);
                    }
                }
            }
            token = allVerificationTokens->back();
        } else if (generationConfig.IsSimpleGreedy()) {
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
