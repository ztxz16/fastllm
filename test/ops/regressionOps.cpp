#include "executor.h"
#include "fastllm.h"

#if defined(USE_CUDA) && !defined(USE_ROCM)
#include <cuda_runtime_api.h>
#endif

#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
    class ScopedFirstDevice {
    public:
        explicit ScopedFirstDevice(const std::string &device) {
            executor = (fastllm::Executor*) fastllm::GetExecutor();
            previous = executor->firstDevice;
            executor->SetFirstDevice(device);
        }

        ~ScopedFirstDevice() {
            if (!previous.empty()) {
                executor->SetFirstDevice(previous);
            }
        }

    private:
        fastllm::Executor *executor = nullptr;
        std::string previous;
    };

    void Expect(bool condition, const std::string &message) {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    fastllm::Data MakeTensor(fastllm::DataType dataType, const std::vector<int> &dims, float seed = 0.0f) {
        int count = 1;
        for (int dim : dims) {
            count *= dim;
        }
        std::vector<float> values(count);
        for (int i = 0; i < count; i++) {
            values[i] = std::sin((i + 1) * 0.17f + seed) + std::cos((i + 3) * 0.11f + seed * 0.5f);
        }
        return fastllm::Data(dataType, dims, values);
    }

    fastllm::Data MakeFloatTensor(const std::vector<int> &dims, float seed = 0.0f) {
        return MakeTensor(fastllm::DataType::FLOAT32, dims, seed);
    }

    fastllm::Data MakeIntTensor(const std::vector<int> &dims, const std::vector<int32_t> &values) {
        int count = 1;
        for (int dim : dims) {
            count *= dim;
        }
        Expect(count == (int) values.size(), "INT32 tensor element count mismatch.");
        fastllm::Data data(fastllm::DataType::INT32, dims);
        data.Allocate();
        if (count > 0) {
            std::memcpy(data.cpuData, values.data(), (size_t) count * sizeof(int32_t));
        }
        return data;
    }

    std::vector<int32_t> ToIntVector(fastllm::Data data, int logicalCount = -1) {
        data.ToDevice(fastllm::DataDevice::CPU);
        int count = logicalCount >= 0 ? logicalCount : (int) data.Count(0);
        std::vector<int32_t> values(count);
        if (count > 0) {
            Expect(data.cpuData != nullptr, "INT32 tensor has no CPU buffer.");
            std::memcpy(values.data(), data.cpuData, (size_t) count * sizeof(int32_t));
        }
        return values;
    }

    std::vector<float> ToFloatVector(fastllm::Data data) {
        data.ToDevice(fastllm::DataDevice::CPU);
        if (data.dataType != fastllm::DataType::FLOAT32) {
            fastllm::ToDataTypeForceCPU(data, fastllm::DataType::FLOAT32);
        }
        int count = (int) data.Count(0);
        std::vector<float> values(count);
        Expect(data.dataType == fastllm::DataType::FLOAT32, "Only FLOAT32 tensors are supported here.");
        if (count > 0) {
            Expect(data.cpuData != nullptr, "FLOAT32 tensor has no CPU buffer.");
            std::memcpy(values.data(), data.cpuData, (size_t) count * sizeof(float));
        }
        return values;
    }

    void ExpectIntEqual(const std::vector<int32_t> &expected, const std::vector<int32_t> &actual,
                        const std::string &name) {
        Expect(expected.size() == actual.size(), name + " size mismatch.");
        for (size_t i = 0; i < expected.size(); i++) {
            if (expected[i] != actual[i]) {
                throw std::runtime_error(name + " mismatch at index " + std::to_string(i) +
                                         ": expected " + std::to_string(expected[i]) +
                                         ", got " + std::to_string(actual[i]));
            }
        }
    }

    void ExpectFloatNear(const std::vector<float> &expected, const std::vector<float> &actual,
                         float atol, float rtol, const std::string &name) {
        Expect(expected.size() == actual.size(), name + " size mismatch.");
        for (size_t i = 0; i < expected.size(); i++) {
            Expect(std::isfinite(expected[i]) && std::isfinite(actual[i]),
                   name + " contains a non-finite value at index " + std::to_string(i));
            float diff = std::fabs(expected[i] - actual[i]);
            float limit = atol + rtol * std::fabs(expected[i]);
            if (diff > limit) {
                throw std::runtime_error(name + " mismatch at index " + std::to_string(i) +
                                         ": expected " + std::to_string(expected[i]) +
                                         ", got " + std::to_string(actual[i]));
            }
        }
    }

#ifdef USE_CUDA
    fastllm::Data MakeCudaTensor(fastllm::DataType dataType, const std::vector<int> &dims,
                                 const std::vector<float> &values) {
        fastllm::Data data(dataType, dims, values);
        data.ToDevice(fastllm::DataDevice::CUDA);
        return data;
    }

    void ExpectCudaTensorMeta(const fastllm::Data &data, fastllm::DataType dataType,
                              const std::vector<int> &dims, const std::string &name) {
        Expect(data.dataType == dataType, name + " dtype mismatch");
        Expect(data.dataDevice == fastllm::DataDevice::CUDA, name + " device mismatch");
        Expect(data.dims == dims, name + " shape mismatch");
        Expect(data.strides.size() == dims.size(), name + " stride rank mismatch");
        uint64_t expectedStride = 1;
        for (int i = (int)dims.size() - 1; i >= 0; i--) {
            Expect(data.strides[i] == expectedStride,
                   name + " is not dense at axis " + std::to_string(i));
            expectedStride *= (uint64_t)dims[i];
        }
        Expect(data.cudaData != nullptr, name + " CUDA buffer is null");
    }

    std::vector<float> MakeRegressionValues(int count, float seed, float scale = 1.0f) {
        std::vector<float> values(count);
        for (int i = 0; i < count; i++) {
            values[i] = scale * (std::sin((i + 1) * 0.071f + seed) +
                                 0.5f * std::cos((i + 3) * 0.043f - seed));
        }
        return values;
    }

    std::vector<float> ExtractLastAxisToken(const std::vector<float> &values,
                                            int rows, int tokens, int token) {
        std::vector<float> result(rows);
        for (int row = 0; row < rows; row++) {
            result[row] = values[(size_t) row * tokens + token];
        }
        return result;
    }

    void RunCudaConvMultiTokenSnapshotsRegression() {
        FastllmCudaSetDevice(0);
        const int batch = 2;
        const int channels = 5;
        const int rows = batch * channels;

        std::vector<float> initialCacheValues = MakeRegressionValues(rows * 4, 0.2f, 0.35f);
        std::vector<float> weightValues = MakeRegressionValues(channels * 4, 0.7f, 0.25f);
        std::vector<float> biasValues = MakeRegressionValues(channels, 1.1f, 0.08f);
        fastllm::Data weight = MakeCudaTensor(fastllm::DataType::FLOAT32,
                                              {channels, 4}, weightValues);
        fastllm::Data bias = MakeCudaTensor(fastllm::DataType::FLOAT32,
                                            {channels}, biasValues);

        for (int tokenCount = 1; tokenCount <= 6; tokenCount++) {
            std::vector<float> tokenValues =
                MakeRegressionValues(rows * tokenCount, 1.7f + tokenCount, 0.4f);
            fastllm::Data allTokens = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                     {batch, channels, tokenCount}, tokenValues);
            fastllm::Data sequentialCache = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                           {batch, channels, 4}, initialCacheValues);
            fastllm::Data multiCache = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                      {batch, channels, 4}, initialCacheValues);

            std::vector<std::vector<float> > expectedOutputs(tokenCount);
            std::vector<std::vector<float> > expectedCaches(tokenCount);
            for (int token = 0; token < tokenCount; token++) {
                std::vector<float> singleTokenValues(rows);
                for (int row = 0; row < rows; row++) {
                    singleTokenValues[row] = tokenValues[(size_t) row * tokenCount + token];
                }
                fastllm::Data singleToken = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                           {batch, channels, 1}, singleTokenValues);
                fastllm::Data singleOutput;
                Expect(FastllmCudaShiftAppendConv1DPerChannelSiluSingleTokenFloat16(
                           sequentialCache, singleToken, weight, bias, singleOutput),
                       "single-token conv reference rejected N=" + std::to_string(tokenCount) +
                       ", token=" + std::to_string(token));
                expectedOutputs[token] = ToFloatVector(singleOutput);
                expectedCaches[token] = ToFloatVector(sequentialCache);
            }

            std::vector<fastllm::Data> snapshots(tokenCount);
            std::vector<fastllm::Data*> snapshotPtrs(tokenCount);
            for (int token = 0; token < tokenCount; token++) {
                snapshotPtrs[token] = &snapshots[token];
            }
            fastllm::Data multiOutput;
            Expect(FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
                       multiCache, allTokens, weight, bias, multiOutput,
                       snapshotPtrs.data(), tokenCount),
                   "multi-token conv rejected N=" + std::to_string(tokenCount));
            ExpectCudaTensorMeta(multiOutput, fastllm::DataType::FLOAT16,
                                 {batch, channels, tokenCount},
                                 "multi-token conv output metadata");

            std::vector<float> actualOutput = ToFloatVector(multiOutput);
            for (int token = 0; token < tokenCount; token++) {
                std::string suffix = " N=" + std::to_string(tokenCount) +
                                     ", token=" + std::to_string(token);
                ExpectFloatNear(expectedOutputs[token],
                                ExtractLastAxisToken(actualOutput, rows, tokenCount, token),
                                1e-3f, 1e-3f, "multi-token conv output" + suffix);
                ExpectFloatNear(expectedCaches[token], ToFloatVector(snapshots[token]),
                                1e-3f, 1e-3f, "multi-token conv snapshot" + suffix);
                ExpectCudaTensorMeta(snapshots[token], fastllm::DataType::FLOAT16,
                                     {batch, channels, 4},
                                     "multi-token conv snapshot metadata" + suffix);
            }
            ExpectFloatNear(ToFloatVector(sequentialCache), ToFloatVector(multiCache),
                            1e-3f, 1e-3f,
                            "multi-token conv final cache N=" + std::to_string(tokenCount));
            if (tokenCount == 6) {
                fastllm::Data partialCache = MakeCudaTensor(
                    fastllm::DataType::FLOAT16, {batch, channels, 4}, initialCacheValues);
                std::vector<fastllm::Data> partialSnapshots(5);
                std::vector<fastllm::Data*> partialSnapshotPtrs(5);
                for (int token = 0; token < 5; token++) {
                    partialSnapshotPtrs[token] = &partialSnapshots[token];
                }
                fastllm::Data partialOutput;
                Expect(FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
                           partialCache, allTokens, weight, bias, partialOutput,
                           partialSnapshotPtrs.data(), 5),
                       "multi-token conv rejected N=6 with five prefix snapshots");
                ExpectFloatNear(ToFloatVector(multiOutput), ToFloatVector(partialOutput),
                                1e-3f, 1e-3f,
                                "multi-token conv partial-snapshot output");
                ExpectFloatNear(ToFloatVector(multiCache), ToFloatVector(partialCache),
                                1e-3f, 1e-3f,
                                "multi-token conv partial-snapshot final cache");
                for (int token = 0; token < 5; token++) {
                    ExpectFloatNear(expectedCaches[token], ToFloatVector(partialSnapshots[token]),
                                    1e-3f, 1e-3f,
                                    "multi-token conv partial snapshot token=" +
                                    std::to_string(token));
                }
            }
        }

        const int tokenCount = 2;
        std::vector<float> tokenValues = MakeRegressionValues(rows * tokenCount, 3.4f, 0.4f);
        fastllm::Data allTokens = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                 {batch, channels, tokenCount}, tokenValues);
        {
            fastllm::Data cache = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                 {batch, channels, 4}, initialCacheValues);
            fastllm::Data output;
            Expect(FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
                       cache, allTokens, weight, bias, output, nullptr, 0),
                   "multi-token conv should accept nullptr snapshots when count is zero");
        }
        {
            fastllm::Data cache = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                 {batch, channels, 4}, initialCacheValues);
            fastllm::Data output;
            Expect(!FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
                       cache, allTokens, weight, bias, output, nullptr, 1),
                   "multi-token conv accepted nullptr snapshots with a positive count");
        }
        {
            fastllm::Data badCache = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                    {batch, channels * 4}, initialCacheValues);
            fastllm::Data output;
            Expect(!FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
                       badCache, allTokens, weight, bias, output, nullptr, 0),
                   "multi-token conv accepted a rank-2 cache");
        }
        {
            fastllm::Data cache = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                 {batch, channels, 4}, initialCacheValues);
            fastllm::Data badTokens = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                     {batch, channels * tokenCount}, tokenValues);
            fastllm::Data output;
            Expect(!FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
                       cache, badTokens, weight, bias, output, nullptr, 0),
                   "multi-token conv accepted rank-2 new tokens");
        }
        {
            // Keep valid backing buffers so this case specifically exercises
            // the zero-channel/grid guard rather than failing on null data.
            fastllm::Data emptyCache = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                      {batch, channels, 4}, initialCacheValues);
            fastllm::Data emptyTokens(allTokens);
            fastllm::Data emptyWeight(weight);
            fastllm::Data emptyBias(bias);
            emptyCache.Resize({batch, 0, 4});
            emptyTokens.Resize({batch, 0, 1});
            emptyWeight.Resize({0, 4});
            emptyBias.Resize({0});
            fastllm::Data output = MakeCudaTensor(
                fastllm::DataType::FLOAT16, {batch, channels, 1},
                MakeRegressionValues(rows, 6.1f, 0.1f));
            Expect(!FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
                       emptyCache, emptyTokens, emptyWeight, emptyBias, output, nullptr, 0),
                   "multi-token conv accepted zero channels");
        }
        {
            fastllm::Data cache = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                 {batch, channels, 4}, initialCacheValues);
            fastllm::Data snapshots[3];
            fastllm::Data *snapshotPtrs[3] = {&snapshots[0], &snapshots[1], &snapshots[2]};
            fastllm::Data output;
            Expect(!FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
                       cache, allTokens, weight, bias, output, snapshotPtrs, 3),
                       "multi-token conv accepted a snapshot count larger than N");
        }
        {
            const int tooManyTokens = 7;
            fastllm::Data cache = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                 {batch, channels, 4}, initialCacheValues);
            fastllm::Data tokens = MakeCudaTensor(
                fastllm::DataType::FLOAT16, {batch, channels, tooManyTokens},
                MakeRegressionValues(rows * tooManyTokens, 6.7f, 0.4f));
            fastllm::Data output;
            Expect(!FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
                       cache, tokens, weight, bias, output, nullptr, 0),
                   "multi-token conv accepted N=7");
        }
    }

    bool RunCudaCrossDeviceViewRejectionRegression() {
        if (FastllmCudaGetDeviceCount() < 2) {
            std::cout << "cross-device CUDA view regression: SKIP (two GPUs required)\n";
            return false;
        }

        const int originalDevice = FastllmCudaGetDevice();
        const int batch = 1;
        const int channels = 2;
        FastllmCudaSetDevice(0);
        fastllm::Data cache = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {batch, channels, 4},
            MakeRegressionValues(batch * channels * 4, 7.3f, 0.2f));
        fastllm::Data weight = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {channels, 4},
            MakeRegressionValues(channels * 4, 7.7f, 0.1f));
        fastllm::Data bias = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {channels},
            MakeRegressionValues(channels, 8.1f, 0.05f));

        FastllmCudaSetDevice(1);
        fastllm::Data remoteTokens = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {batch, channels, 2},
            MakeRegressionValues(batch * channels * 2, 8.5f, 0.2f));
        fastllm::Data remoteView;
        remoteView.FakeFrom(remoteTokens, 0);
        remoteView.dims = remoteTokens.dims;
        remoteView.strides = remoteTokens.strides;
        Expect(remoteView.dataDeviceIds.empty(),
               "cross-device fake view unexpectedly inherited device IDs");

        FastllmCudaSetDevice(0);
        fastllm::Data output;
        Expect(!FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
                   cache, remoteView, weight, bias, output, nullptr, 0),
               "multi-token conv accepted a CUDA view owned by another device");
        remoteView.dataDeviceIds = {0};
        Expect(!FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
                   cache, remoteView, weight, bias, output, nullptr, 0),
               "multi-token conv trusted a stale CUDA view device ID");

        // A legal view may omit metadata. Its actual pointer device must still
        // control where a reusable destination is migrated and allocated.
        FastllmCudaSetDevice(1);
        fastllm::Data localCache = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {batch, channels, 4},
            MakeRegressionValues(batch * channels * 4, 8.9f, 0.2f));
        fastllm::Data localTokens = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {batch, channels, 2},
            MakeRegressionValues(batch * channels * 2, 9.3f, 0.2f));
        fastllm::Data localWeight = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {channels, 4},
            MakeRegressionValues(channels * 4, 9.7f, 0.1f));
        fastllm::Data localBias = MakeCudaTensor(
            fastllm::DataType::FLOAT32, {channels},
            MakeRegressionValues(channels, 10.1f, 0.05f));
        localCache.dataDeviceIds.clear();

        FastllmCudaSetDevice(0);
        fastllm::Data reusableOutput = MakeCudaTensor(
            fastllm::DataType::FLOAT16, {batch, channels, 2},
            MakeRegressionValues(batch * channels * 2, 10.5f, 0.1f));
        Expect(GetPointerDeviceId(reusableOutput.cudaData) == 0,
               "cross-device output fixture was not allocated on GPU 0");
        FastllmCudaSetDevice(1);
        Expect(FastllmCudaShiftAppendConv1DPerChannelSiluMultiTokenFloat16(
                   localCache, localTokens, localWeight, localBias,
                   reusableOutput, nullptr, 0),
               "multi-token conv could not migrate a reusable output to the cache device");
        Expect(GetPointerDeviceId(reusableOutput.cudaData) == 1 &&
                   reusableOutput.dataDeviceIds == std::vector<int>({1}),
               "multi-token conv left its output on the wrong CUDA device");
        FastllmCudaSetDevice(originalDevice);
        return true;
    }

#ifndef USE_ROCM
    bool RunMultiCudaLargeWeightOffsetRegression() {
        if (FastllmCudaGetDeviceCount() < 1) {
            std::cout << "multi-CUDA large-weight offset regression: SKIP (CUDA unavailable)\n";
            return false;
        }

        constexpr int rows = 248320;
        constexpr int columns = 5120;
        constexpr int lateRow = 217280;
        const size_t rowBytes = static_cast<size_t>(columns) * sizeof(uint16_t);
        const size_t lateOffset = static_cast<size_t>(lateRow) * rowBytes;
        const size_t totalBytes = static_cast<size_t>(rows) * rowBytes;
        Expect(lateOffset > 0x7fffffffULL,
               "large-weight regression does not cross the int32 byte-offset boundary");

        const int originalDevice = FastllmCudaGetDevice();
        FastllmCudaSetDevice(0);
        void *managed = nullptr;
        cudaError_t state = cudaMallocManaged(&managed, totalBytes);
        if (state != cudaSuccess) {
            cudaGetLastError();
            FastllmCudaSetDevice(originalDevice);
            std::cout << "multi-CUDA large-weight offset regression: SKIP (managed allocation unavailable)\n";
            return false;
        }

        fastllm::Data weight(fastllm::DataType::BFLOAT16, {rows, columns});
        weight.name = "regression.large_lm_head.weight";
        weight.dataDevice = fastllm::DataDevice::CUDA;
        weight.dataDeviceIds = {0};
        weight.cudaData = managed;
        // SplitMultiCudaWeight consumes and releases the root CUDA storage.
        // Treat the managed allocation as owned so its normal cleanup path is
        // exercised without reserving the full logical tensor physically.
        weight.cudaDataBorrowed = false;
        fastllm::Data bias;
        std::vector<int> devices = {0};
        DivisionScheme scheme;
        scheme[0] = {{0, 1}, {lateRow, lateRow + 1}};

        state = cudaMemset(managed, 0x11, rowBytes);
        if (state == cudaSuccess) {
            state = cudaMemset(static_cast<uint8_t*>(managed) + lateOffset,
                               0xa5, rowBytes);
        }
        Expect(state == cudaSuccess, "failed to initialize managed large-weight rows");
        Expect(SplitMultiCudaWeight(weight, bias, devices, scheme, 0, true),
               "failed to split a weight whose source offset exceeds INT_MAX");

        auto localIt = weight.multiDeviceDatas.find(0);
        Expect(localIt != weight.multiDeviceDatas.end() && localIt->second != nullptr,
               "large-weight split did not create the local tensor");
        fastllm::Data *local = localIt->second;
        Expect(local->dims == std::vector<int>({2, columns}),
               "large-weight split produced the wrong local shape");
        std::vector<uint8_t> actual(2 * rowBytes);
        state = cudaMemcpy(actual.data(), local->cudaData, actual.size(),
                           cudaMemcpyDeviceToHost);
        Expect(state == cudaSuccess, "failed to copy the split large-weight rows");
        Expect(std::all_of(actual.begin(), actual.begin() + rowBytes,
                           [](uint8_t value) { return value == 0x11; }),
               "large-weight split corrupted the first source row");
        Expect(std::all_of(actual.begin() + rowBytes, actual.end(),
                           [](uint8_t value) { return value == 0xa5; }),
               "large-weight split used a truncated source offset");

        FastllmCudaSetDevice(originalDevice);
        return true;
    }
#endif

    void RunCudaRecurrentSnapshotsRegression() {
        FastllmCudaSetDevice(0);
        const int numKHeads = 1;
        const int numVHeads = 2;
        const int headKDim = 128;
        const int headVDim = 9;
        const int qkvDim = 2 * numKHeads * headKDim + numVHeads * headVDim;
        const float eps = 1e-6f;
        const float qScale = 1.0f / std::sqrt((float) headKDim);

        std::vector<float> normValues(headKDim);
        for (int i = 0; i < headKDim; i++) {
            normValues[i] = 0.85f + 0.08f * std::cos((i + 1) * 0.031f);
        }
        std::vector<float> initialStateValues =
            MakeRegressionValues(numVHeads * headVDim * headKDim, 0.9f, 0.025f);
        fastllm::Data normWeight = MakeCudaTensor(fastllm::DataType::FLOAT32,
                                                  {headKDim}, normValues);
        fastllm::Data aLog = MakeCudaTensor(fastllm::DataType::FLOAT32,
                                            {numVHeads}, {-0.7f, -0.55f});
        fastllm::Data dtBias = MakeCudaTensor(fastllm::DataType::FLOAT32,
                                              {numVHeads}, {0.15f, -0.08f});

        for (int tokenCount = 2; tokenCount <= 6; tokenCount++) {
            std::vector<float> convValues =
                MakeRegressionValues(tokenCount * qkvDim, 2.1f + tokenCount, 0.12f);
            std::vector<float> baValues(tokenCount * numVHeads * 2);
            for (int token = 0; token < tokenCount; token++) {
                for (int head = 0; head < numVHeads; head++) {
                    baValues[(size_t)token * numVHeads * 2 + head] =
                        -4.5f + 0.04f * token - 0.03f * head;
                    baValues[(size_t)token * numVHeads * 2 + numVHeads + head] =
                        -0.35f + 0.03f * token + 0.02f * head;
                }
            }

            fastllm::Data convSequence = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                        {1, tokenCount, qkvDim}, convValues);
            fastllm::Data baSequence = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                      {1, tokenCount, numVHeads * 2}, baValues);
            fastllm::Data sequentialState = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                           {1, numVHeads, headKDim, headVDim},
                                                           initialStateValues);
            fastllm::Data sequenceState = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                         {1, numVHeads, headKDim, headVDim},
                                                         initialStateValues);
            sequentialState.isLinearAttentionTransposed = true;
            sequenceState.isLinearAttentionTransposed = true;

            std::vector<std::vector<float> > expectedOutputs(tokenCount);
            std::vector<std::vector<float> > expectedStates(tokenCount);
            for (int token = 0; token < tokenCount; token++) {
                std::vector<float> singleConv(qkvDim);
                std::copy(convValues.begin() + (size_t) token * qkvDim,
                          convValues.begin() + (size_t) (token + 1) * qkvDim,
                          singleConv.begin());
                std::vector<float> singleBa(numVHeads * 2);
                std::copy(baValues.begin() + (size_t) token * numVHeads * 2,
                          baValues.begin() + (size_t) (token + 1) * numVHeads * 2,
                          singleBa.begin());
                fastllm::Data convToken = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                         {1, 1, qkvDim}, singleConv);
                fastllm::Data baToken = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                       {1, 1, numVHeads * 2}, singleBa);
                fastllm::Data singleOutput;
                Expect(FastllmRecurrentGatedDeltaRuleFromConvBaTransposedFloat16(
                           convToken, baToken, normWeight, aLog, dtBias,
                           sequentialState, singleOutput,
                           numKHeads, numVHeads, headKDim, headVDim, eps, qScale),
                       "single-token recurrent reference rejected N=" +
                       std::to_string(tokenCount) + ", token=" + std::to_string(token));
                expectedOutputs[token] = ToFloatVector(singleOutput);
                expectedStates[token] = ToFloatVector(sequentialState);
            }

            int snapshotCount = std::min(tokenCount, 5);
            std::vector<fastllm::Data> snapshots(snapshotCount);
            std::vector<fastllm::Data*> snapshotPtrs(snapshotCount);
            for (int token = 0; token < snapshotCount; token++) {
                snapshotPtrs[token] = &snapshots[token];
            }
            fastllm::Data sequenceOutput;
            Expect(FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16Snapshots(
                       convSequence, baSequence, normWeight, aLog, dtBias,
                       sequenceState, sequenceOutput, snapshotPtrs.data(), snapshotCount,
                       numKHeads, numVHeads, headKDim, headVDim, eps, qScale),
                   "recurrent snapshot sequence rejected N=" + std::to_string(tokenCount));
            ExpectCudaTensorMeta(sequenceOutput, fastllm::DataType::FLOAT16,
                                 {1, tokenCount, numVHeads, headVDim},
                                 "recurrent sequence output metadata");

            fastllm::Data noSnapshotState = MakeCudaTensor(
                fastllm::DataType::FLOAT16,
                {1, numVHeads, headKDim, headVDim}, initialStateValues);
            noSnapshotState.isLinearAttentionTransposed = true;
            fastllm::Data noSnapshotOutput;
            Expect(FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16(
                       convSequence, baSequence, normWeight, aLog, dtBias,
                       noSnapshotState, noSnapshotOutput,
                       numKHeads, numVHeads, headKDim, headVDim, eps, qScale),
                   "recurrent no-snapshot sequence rejected N=" +
                   std::to_string(tokenCount));
            ExpectCudaTensorMeta(noSnapshotOutput, fastllm::DataType::FLOAT16,
                                 {1, tokenCount, numVHeads, headVDim},
                                 "recurrent no-snapshot output metadata");
            ExpectFloatNear(ToFloatVector(sequenceOutput), ToFloatVector(noSnapshotOutput),
                            1e-3f, 1e-3f,
                            "recurrent no-snapshot output N=" + std::to_string(tokenCount));
            ExpectFloatNear(ToFloatVector(sequenceState), ToFloatVector(noSnapshotState),
                            1e-3f, 1e-3f,
                            "recurrent no-snapshot final state N=" + std::to_string(tokenCount));

            std::vector<float> actualOutput = ToFloatVector(sequenceOutput);
            const int outputRows = numVHeads * headVDim;
            for (int token = 0; token < tokenCount; token++) {
                std::vector<float> actualTokenOutput(
                    actualOutput.begin() + (size_t) token * outputRows,
                    actualOutput.begin() + (size_t) (token + 1) * outputRows);
                std::string suffix = " N=" + std::to_string(tokenCount) +
                                     ", token=" + std::to_string(token);
                ExpectFloatNear(expectedOutputs[token], actualTokenOutput,
                                2e-3f, 2e-3f, "recurrent sequence output" + suffix);
                if (token < snapshotCount) {
                    ExpectFloatNear(expectedStates[token], ToFloatVector(snapshots[token]),
                                    2e-3f, 2e-3f, "recurrent state snapshot" + suffix);
                    Expect(snapshots[token].isLinearAttentionTransposed,
                           "recurrent snapshot lost transposed layout marker" + suffix);
                    ExpectCudaTensorMeta(snapshots[token], fastllm::DataType::FLOAT16,
                                         {1, numVHeads, headKDim, headVDim},
                                         "recurrent snapshot metadata" + suffix);
                }
            }
            ExpectFloatNear(ToFloatVector(sequentialState), ToFloatVector(sequenceState),
                            2e-3f, 2e-3f,
                            "recurrent final state N=" + std::to_string(tokenCount));
            if (tokenCount == 6) {
                fastllm::Data rejectedState = MakeCudaTensor(
                    fastllm::DataType::FLOAT16,
                    {1, numVHeads, headKDim, headVDim}, initialStateValues);
                rejectedState.isLinearAttentionTransposed = true;
                fastllm::Data rejectedSnapshots[6];
                fastllm::Data *rejectedSnapshotPtrs[6];
                for (int token = 0; token < 6; token++) {
                    rejectedSnapshotPtrs[token] = &rejectedSnapshots[token];
                }
                fastllm::Data rejectedOutput;
                Expect(!FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16Snapshots(
                           convSequence, baSequence, normWeight, aLog, dtBias,
                           rejectedState, rejectedOutput, rejectedSnapshotPtrs, 6,
                           numKHeads, numVHeads, headKDim, headVDim, eps, qScale),
                       "recurrent sequence accepted N=6 with six snapshots");
            }
        }

        std::vector<float> oneConvValues = MakeRegressionValues(qkvDim, 5.2f, 0.12f);
        std::vector<float> oneBaValues = {-4.5f, -4.53f, -0.35f, -0.33f};
        {
            fastllm::Data conv = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                {1, 1, qkvDim}, oneConvValues);
            fastllm::Data ba = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                              {1, 1, numVHeads * 2}, oneBaValues);
            fastllm::Data state = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                 {1, numVHeads, headKDim, headVDim},
                                                 initialStateValues);
            state.isLinearAttentionTransposed = true;
            fastllm::Data snapshot;
            fastllm::Data *snapshotPtr = &snapshot;
            fastllm::Data output;
            Expect(!FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16Snapshots(
                       conv, ba, normWeight, aLog, dtBias, state, output, &snapshotPtr, 1,
                       numKHeads, numVHeads, headKDim, headVDim, eps, qScale),
                   "recurrent snapshot sequence accepted N=1");
        }
        {
            std::vector<float> twoConvValues = MakeRegressionValues(2 * qkvDim, 5.8f, 0.12f);
            std::vector<float> twoBaValues = {
                -4.5f, -4.53f, -0.35f, -0.33f,
                -4.46f, -4.49f, -0.32f, -0.30f
            };
            fastllm::Data conv = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                {1, 2, qkvDim}, twoConvValues);
            fastllm::Data ba = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                              {1, 2, numVHeads * 2}, twoBaValues);
            fastllm::Data state = MakeCudaTensor(fastllm::DataType::FLOAT16,
                                                 {1, numVHeads, headKDim, headVDim},
                                                 initialStateValues);
            state.isLinearAttentionTransposed = true;
            fastllm::Data output;
            Expect(!FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16Snapshots(
                       conv, ba, normWeight, aLog, dtBias, state, output, nullptr, 2,
                       numKHeads, numVHeads, headKDim, headVDim, eps, qScale),
                       "recurrent snapshot sequence accepted nullptr tokenStates");
        }
        {
            const int tooManyTokens = 7;
            fastllm::Data conv = MakeCudaTensor(
                fastllm::DataType::FLOAT16, {1, tooManyTokens, qkvDim},
                MakeRegressionValues(tooManyTokens * qkvDim, 6.4f, 0.12f));
            fastllm::Data ba = MakeCudaTensor(
                fastllm::DataType::FLOAT16,
                {1, tooManyTokens, numVHeads * 2},
                MakeRegressionValues(tooManyTokens * numVHeads * 2, 6.9f, 0.08f));
            fastllm::Data state = MakeCudaTensor(
                fastllm::DataType::FLOAT16,
                {1, numVHeads, headKDim, headVDim}, initialStateValues);
            state.isLinearAttentionTransposed = true;
            fastllm::Data output;
            Expect(!FastllmRecurrentGatedDeltaRuleSequenceFromConvBaTransposedFloat16(
                       conv, ba, normWeight, aLog, dtBias, state, output,
                       numKHeads, numVHeads, headKDim, headVDim, eps, qScale),
                   "recurrent sequence accepted N=7");
        }
    }
#endif

    struct PastKeyBatch {
        std::vector<fastllm::Data> keys;
        std::vector<fastllm::Data*> keyPtrs;
        std::vector<int> seqLens;
        int totalPages = 0;
        int totalSeq = 0;
    };

    PastKeyBatch BuildPastKeysForPagedRegression(int batch, int pageLen, fastllm::PagedCacheManager *manager) {
        PastKeyBatch result;
        result.keys.reserve(batch);
        result.keyPtrs.reserve(batch);
        result.seqLens.reserve(batch);

        for (int b = 0; b < batch; b++) {
            result.keys.emplace_back();
            fastllm::Data &key = result.keys.back();
            key.isKVCache = true;
            key.isPagedKVCache = true;
            key.pageLen = pageLen;
            key.pagedKVCacheData = manager;

            int mode = b % 4;
            int pageCount = mode;
            if (pageCount > 0) {
                key.pageIndex.reserve(pageCount);
                for (int i = 0; i < pageCount; i++) {
                    key.pageIndex.push_back(manager->GetUnusedPageIndex(true));
                }
            }
            if (pageCount == 0) {
                key.lastPageLen = 0;
            } else if (mode == 1) {
                key.lastPageLen = pageLen / 2;
            } else if (mode == 2) {
                key.lastPageLen = pageLen;
            } else {
                key.lastPageLen = pageLen - 3;
            }

            result.totalPages += pageCount;
            int seqLen = 1 + (b % 5);
            result.seqLens.push_back(seqLen);
            result.totalSeq += seqLen;

            result.keyPtrs.push_back(&key);
        }

        return result;
    }

    fastllm::PagedCacheManager* CreateManager(int layerIndex, int pageLen, int maxPages) {
        fastllm::Data cache = MakeFloatTensor({4, 1, 8}, 0.2f);
        return fastllm::AllocatePagedCacheManager(
            layerIndex,
            fastllm::PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE,
            cache,
            pageLen,
            maxPages
        );
    }

    void RunGenerateAppendPagedCacheBatchParams(const std::string &device, int batch) {
        const int pageLen = 128;
        fastllm::ClearAllPagedCacheManagers();
        {
            fastllm::PagedCacheManager *cpuManager = CreateManager(0, pageLen, batch * 4);
            fastllm::PagedCacheManager *deviceManager = CreateManager(1, pageLen, batch * 4);

            PastKeyBatch cpuPast = BuildPastKeysForPagedRegression(batch, pageLen, cpuManager);
            PastKeyBatch devicePast = BuildPastKeysForPagedRegression(batch, pageLen, deviceManager);

            fastllm::Data cpuInsertIndexs, cpuInsertPositions;
            {
                ScopedFirstDevice guard("cpu");
                fastllm::GenerateAppendPagedCacheBatchParams(
                    *cpuManager, cpuPast.keyPtrs, batch, cpuInsertIndexs, cpuInsertPositions);
            }

            fastllm::Data deviceInsertIndexs, deviceInsertPositions;
            {
                ScopedFirstDevice guard(device);
                fastllm::GenerateAppendPagedCacheBatchParams(
                    *deviceManager, devicePast.keyPtrs, batch, deviceInsertIndexs, deviceInsertPositions);
            }

            ExpectIntEqual(ToIntVector(cpuInsertIndexs), ToIntVector(deviceInsertIndexs), "insertIndexs");
            ExpectIntEqual(ToIntVector(cpuInsertPositions), ToIntVector(deviceInsertPositions), "insertPositions");
        }
        fastllm::ClearAllPagedCacheManagers();
    }

    void RunGeneratePagedBatchParams(const std::string &device, int batch, bool zeroPages) {
        const int pageLen = 128;
        fastllm::ClearAllPagedCacheManagers();
        {
            fastllm::PagedCacheManager *manager = CreateManager(2, pageLen, std::max(batch * 4, 16));
            PastKeyBatch past = zeroPages ? PastKeyBatch() : BuildPastKeysForPagedRegression(batch, pageLen, manager);
            if (zeroPages) {
                past.keys.reserve(batch);
                past.keyPtrs.reserve(batch);
                past.seqLens.reserve(batch);
                for (int b = 0; b < batch; b++) {
                    past.keys.emplace_back();
                    fastllm::Data &key = past.keys.back();
                    key.isKVCache = true;
                    key.isPagedKVCache = true;
                    key.pageLen = pageLen;
                    key.pagedKVCacheData = manager;
                    key.lastPageLen = 0;
                    past.keyPtrs.push_back(&key);
                    int seqLen = 1 + (b % 3);
                    past.seqLens.push_back(seqLen);
                    past.totalSeq += seqLen;
                }
                past.totalPages = 0;
            }

            fastllm::Data q = MakeFloatTensor({4, past.totalSeq, 8}, 0.3f);

            fastllm::Data cpuQSizes, cpuPageSizes, cpuPageIndexs, cpuLastPageLens;
            {
                ScopedFirstDevice guard("cpu");
                fastllm::GeneratePagedBatchParams(
                    q, past.keyPtrs, batch, cpuQSizes, cpuPageSizes, cpuPageIndexs, cpuLastPageLens, past.seqLens);
            }

            fastllm::Data deviceQSizes, devicePageSizes, devicePageIndexs, deviceLastPageLens;
            {
                ScopedFirstDevice guard(device);
                fastllm::GeneratePagedBatchParams(
                    q, past.keyPtrs, batch, deviceQSizes, devicePageSizes, devicePageIndexs, deviceLastPageLens, past.seqLens);
            }

            std::vector<int32_t> cpuPageSizesVec = ToIntVector(cpuPageSizes);
            std::vector<int32_t> devicePageSizesVec = ToIntVector(devicePageSizes);
            int logicalPages = cpuPageSizesVec.empty() ? 0 : cpuPageSizesVec.back();

            ExpectIntEqual(ToIntVector(cpuQSizes), ToIntVector(deviceQSizes), "qSizes");
            ExpectIntEqual(cpuPageSizesVec, devicePageSizesVec, "pageSizes");
            ExpectIntEqual(ToIntVector(cpuLastPageLens), ToIntVector(deviceLastPageLens), "lastPageLens");
            ExpectIntEqual(
                ToIntVector(cpuPageIndexs, logicalPages),
                ToIntVector(devicePageIndexs, logicalPages),
                "pageIndexs"
            );
            Expect(devicePageIndexs.dims.empty() || devicePageIndexs.dims[0] >= logicalPages,
                   "device pageIndexs shape is smaller than the logical page count.");
        }
        fastllm::ClearAllPagedCacheManagers();
    }

    struct MoeWeights {
        fastllm::Data routedGate;
        fastllm::Data routedDown;
    };

    MoeWeights MakeMoeWeights(int inputDim, int interDim, int outputDim, float seed) {
        MoeWeights weights {
            MakeTensor(fastllm::DataType::FLOAT16, {interDim * 2, inputDim}, seed),
            MakeTensor(fastllm::DataType::FLOAT16, {outputDim, interDim}, seed + 1.0f)
        };
        weights.routedGate.name = "test.routed_gate";
        weights.routedDown.name = "test.routed_down";
        return weights;
    }

    std::vector<float> RunMergeMoeOnDevice(const std::string &device, MoeWeights &weights) {
        const int batch = 32;
        const int inputDim = 64;
        const int outputDim = 64;

        fastllm::Data input = MakeFloatTensor({batch, inputDim}, 0.7f);
        fastllm::Data output(fastllm::DataType::FLOAT32, {batch, outputDim});
        fastllm::Data index = MakeIntTensor({batch, 1}, std::vector<int32_t>(batch, 0));
        fastllm::Data score(fastllm::DataType::FLOAT32, {batch, 1}, std::vector<float>(batch, 1.0f));
        fastllm::Data w1, w2, w3, curInput, curOutput;

        std::vector<fastllm::Data*> weightPtrs = {
            nullptr, nullptr, &weights.routedGate, &weights.routedDown
        };
        std::vector<fastllm::Data*> biasPtrs(4, nullptr);

        {
            ScopedFirstDevice guard(device);
            fastllm::MergeMOE(
                input, index, score, weightPtrs, biasPtrs,
                w1, w2, w3, curInput, curOutput,
                0.0f, output, 0
            );
        }

        Expect(output.dataType == fastllm::DataType::FLOAT32,
               "MergeMOE output dtype mismatch.");
        return ToFloatVector(output);
    }

    void RunNumasMergeMoeRegression() {
        const int inputDim = 64;
        const int interDim = 128;
        const int outputDim = 64;

        MoeWeights cpuWeights = MakeMoeWeights(inputDim, interDim, outputDim, 1.1f);
        MoeWeights numasWeights = MakeMoeWeights(inputDim, interDim, outputDim, 1.1f);

        std::vector<float> expected = RunMergeMoeOnDevice("cpu", cpuWeights);
        std::vector<float> actual = RunMergeMoeOnDevice("numa", numasWeights);

        ExpectFloatNear(expected, actual, 1e-3f, 1e-4f, "numas MergeMOE output");
        Expect(!numasWeights.routedGate.numasData.empty(), "routed gate weight was not registered to NUMA shards.");
        Expect(!numasWeights.routedDown.numasData.empty(), "routed down weight was not registered to NUMA shards.");
        Expect(numasWeights.routedGate.cpuData == nullptr, "routed gate CPU buffer should be released after NUMA registration.");
        Expect(numasWeights.routedDown.cpuData == nullptr, "routed down CPU buffer should be released after NUMA registration.");
    }
}

int main() {
    try {
        bool ranAny = false;
        bool ranCrossDeviceViewRegression = false;
#ifndef USE_ROCM
        bool ranLargeWeightOffsetRegression = false;
#endif

        if (fastllm::HasDeviceType("cuda")) {
#ifdef USE_CUDA
            Expect(FastllmCudaGraphQwen35MoeSelfTest(),
                   "Qwen3.5 CUDA graph shared/routed MoE parallelization self-test failed");
            RunCudaConvMultiTokenSnapshotsRegression();
            ranCrossDeviceViewRegression = RunCudaCrossDeviceViewRejectionRegression();
#ifndef USE_ROCM
            ranLargeWeightOffsetRegression = RunMultiCudaLargeWeightOffsetRegression();
#endif
            RunCudaRecurrentSnapshotsRegression();
#endif
            RunGenerateAppendPagedCacheBatchParams("cuda:0", 1536);
            RunGeneratePagedBatchParams("cuda:0", 1536, false);
            RunGeneratePagedBatchParams("cuda:0", 64, true);
            std::cout << "cuda snapshot and paged-batch regressions: PASS";
            if (!ranCrossDeviceViewRegression) {
                std::cout << " (cross-device view SKIPPED)";
            }
#ifndef USE_ROCM
            if (!ranLargeWeightOffsetRegression) {
                std::cout << " (large-weight offset SKIPPED)";
            }
#endif
            std::cout << "\n";
            ranAny = true;
        } else {
            std::cout << "cuda snapshot and paged-batch regressions: SKIP (cuda unavailable)\n";
        }

        if (fastllm::HasDeviceType("numa") && !fastllm::GetFastllmEnv().activateNuma) {
            RunNumasMergeMoeRegression();
            std::cout << "numa MergeMOE regression: PASS\n";
            ranAny = true;
        } else if (fastllm::HasDeviceType("numa")) {
            std::cout << "numa MergeMOE regression: SKIP (legacy numa device is active)\n";
        } else {
            std::cout << "numa MergeMOE regression: SKIP (numa unavailable)\n";
        }

        if (!ranAny) {
            std::cout << "no matching regression device paths available\n";
        }
        return 0;
    } catch (const std::exception &ex) {
        std::cerr << "regressionOps failed: " << ex.what() << "\n";
    } catch (...) {
        std::cerr << "regressionOps failed: unknown error\n";
    }
    return 1;
}
