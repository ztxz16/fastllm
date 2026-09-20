#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#define FASTLLM_CUDA_NO_MALLOC_CHECK_MACRO
#include "devices/cuda/fastllm-cuda.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#define CK(x) do { auto e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA line %d: %s\n", __LINE__, cudaGetErrorString(e)); std::abort(); } } while (0)
#define REQUIRE(x) do { if (!(x)) { \
    fprintf(stderr, "CHECK line %d: %s\n", __LINE__, #x); std::abort(); } } while (0)
using namespace fastllm;

static std::vector<unsigned char> bytes(const Data &x) {
    std::vector<unsigned char> result(x.GetBytes());
    CK(cudaMemcpy(result.data(), x.cudaData, result.size(), cudaMemcpyDeviceToHost));
    return result;
}

template<class T>
static void initialize(Data &x, DataType type, int device, bool padded) {
    x.dataType = type;
    x.UpdateUnitSize();
    x.dataDevice = CUDA;
    x.dataDeviceIds = {device};
    x.Resize({7, 64});
    if (padded) x.Expansion({7, 128});
    else x.Allocate(false);
    x.blockK = 1;
    x.blockM = 64;
    x.scales = {1, 0.5f, 2, 1, 0.25f, 1, 4};
    std::vector<T> data(7 * x.strides[0], T(64.0f));
    for (int r = 0; r < 7; ++r) {
        for (int c = 0; c < 64; ++c) {
            float value = r == 6 ? 64.0f : ((c * 13 + r * 7) % 31 - 15) * 0.125f;
            data[r * x.strides[0] + c] = T(value);
        }
    }
    CK(cudaMemcpy(x.cudaData, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template<class T>
static void checkType(DataType type, int device) {
    Data input, padded, full, output, legacy;
    initialize<T>(input, type, device, false);
    initialize<T>(padded, type, device, true);
    REQUIRE(FastllmCudaQuantizeLinearWeightNVFP4Block16Rows(input, full, {}));
    REQUIRE(FastllmCudaQuantizeLinearWeightNVFP4Block16(input, legacy));
    REQUIRE(bytes(full) == bytes(legacy) && full.scales == legacy.scales);
    // Reordered and repeated rows must match full quantization, including its
    // tensor scale when the largest-scale row (6) is excluded from the selection.
    const std::vector<int> rows = {5, 1, 1, 0};
    REQUIRE(FastllmCudaQuantizeLinearWeightNVFP4Block16Rows(input, output, rows));
    auto all = bytes(full), selected = bytes(output);
    const size_t rowBytes = all.size() / 7;
    REQUIRE(output.dims == std::vector<int>({4, 64}));
    REQUIRE(output.scales == full.scales);
    for (size_t r = 0; r < rows.size(); ++r)
        REQUIRE(std::memcmp(selected.data() + r * rowBytes, all.data() + rows[r] * rowBytes, rowBytes) == 0);

    // Unsupported inputs must be rejected before modifying a populated output.
    void *savedPointer = output.cudaData;
    auto savedDims = output.dims;
    auto savedScales = output.scales;
    auto rejected = [&](const Data &x, const std::vector<int> &ids = {0}) {
        REQUIRE(!FastllmCudaQuantizeLinearWeightNVFP4Block16Rows(x, output, ids));
        REQUIRE(output.cudaData == savedPointer && output.dims == savedDims);
        REQUIRE(output.scales == savedScales && bytes(output) == selected);
    };
    rejected(padded);
    REQUIRE(!FastllmCudaQuantizeLinearWeightNVFP4Block16(padded, output));
    auto strides = input.strides;
    input.strides.clear(); rejected(input); input.strides = strides;
    input.strides[1] = 2; rejected(input); input.strides = strides;
    input.IsRepacked = true; rejected(input); input.IsRepacked = false;
    input.multiDeviceData = true; rejected(input); input.multiDeviceData = false;
    input.dataDeviceIds.clear(); rejected(input); input.dataDeviceIds = {device, device};
    rejected(input); input.dataDeviceIds = {device};
    rejected(input, {-1}); rejected(input, {7}); rejected(input, std::vector<int>(8, 0));
    if (type == FP8_E4M3) {
        input.blockK = 2; rejected(input); input.blockK = 1;
        input.blockM = 32; rejected(input); input.blockM = 64;
        input.scales.pop_back(); rejected(input); input.scales.push_back(4);
    }
    auto original = bytes(input);
    REQUIRE(!FastllmCudaQuantizeLinearWeightNVFP4Block16Rows(input, input, rows));
    REQUIRE(bytes(input) == original);
    CK(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    REQUIRE(!FastllmCudaQuantizeLinearWeightNVFP4Block16Rows(input, output, rows));
    cudaGraph_t graph;
    CK(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    CK(cudaGraphDestroy(graph));
    REQUIRE(output.cudaData == savedPointer && bytes(output) == selected);
    std::printf("PASS device=%d dtype=%d dense/selection/scale/layout/rejection/capture\n", device, int(type));
}

static void checkInt4(DataType type, int device, int rows, int columns,
                     int groupSize, bool zeroPoint, bool perTensor = false,
                     bool configZeros = true) {
    const bool grouped = type == INT4_GROUP;
    const int groups = grouped ? (columns - 1) / groupSize + 1 : 1;
    Data input, reference, full, expected, selected;
    input.dataType = type; input.UpdateUnitSize();
    input.dataDevice = CUDA; input.dataDeviceIds = {device};
    input.Resize({rows, columns}); input.Allocate(false);
    input.perChannelAxis = perTensor ? -1 : 0;
    input.group = groups; input.groupCnt = groupSize;
    const int count = (perTensor ? 1 : rows) * groups;
    input.scales.resize(count); input.mins.resize(count);
    if (zeroPoint || type == INT4_NOZERO) input.zeros.resize(count);
    for (int i = 0; i < count; ++i) {
        input.scales[i] = 0.0713f * (1 + i % 7) * (i >= count - groups ? 32 : 1);
        input.mins[i] = -0.173f - 0.0329f * (i % 5);
        if (!input.zeros.empty()) input.zeros[i] = i % 16;
        if (type == INT4 && configZeros) {
            LowBitConfig config(-1, 1, 4, 0);
            config.zeroPoint = input.zeros[i]; config.scale = input.scales[i];
            input.perChannelsConfigs.push_back(config);
        }
    }
    std::vector<unsigned char> packed(size_t(rows) * columns / 2, 0);
    std::vector<half> decoded(size_t(rows) * columns);
    for (int r = 0; r < rows; ++r) for (int c = 0; c < columns; ++c) {
        const size_t index = size_t(r) * columns + c;
        const int q = (r * 3 + c * 7 + c / 16) % 16;
        packed[index / 2] |= q << ((c & 1) ? 0 : 4);
        const int g = (perTensor ? 0 : r) * groups + (grouped ? c / groupSize : 0);
        float scale = input.scales[g], offset = zeroPoint ? float(input.zeros[g]) : input.mins[g];
        if (grouped) { scale = float(half(scale)); offset = float(half(offset)); }
        float value = zeroPoint ? scale * (q - offset) : std::fma(scale, float(q), offset);
        if (type == INT4 && configZeros) value = input.perChannelsConfigs[g].invQuantization(q);
        decoded[index] = half(value);
    }
    CK(cudaMemcpy(input.cudaData, packed.data(), packed.size(), cudaMemcpyHostToDevice));
    reference.dataType = FLOAT16; reference.UpdateUnitSize();
    reference.dataDevice = CUDA; reference.dataDeviceIds = {device};
    reference.Resize({rows, columns}); reference.Allocate(false);
    CK(cudaMemcpy(reference.cudaData, decoded.data(), decoded.size() * sizeof(half), cudaMemcpyHostToDevice));
    REQUIRE(FastllmCudaQuantizeLinearWeightNVFP4Block16(reference, expected));
    REQUIRE(FastllmCudaQuantizeLinearWeightNVFP4Block16(input, full));
    REQUIRE(bytes(full) == bytes(expected) && full.scales == expected.scales);
    std::vector<int> ids = rows > 2 ? std::vector<int>{1, 0, 1} : std::vector<int>{0};
    REQUIRE(FastllmCudaQuantizeLinearWeightNVFP4Block16Rows(input, selected, ids));
    auto all = bytes(expected), got = bytes(selected);
    const size_t rowBytes = all.size() / rows;
    REQUIRE(selected.scales == expected.scales);
    for (size_t r = 0; r < ids.size(); ++r)
        REQUIRE(std::memcmp(got.data() + r * rowBytes, all.data() + ids[r] * rowBytes, rowBytes) == 0);
    REQUIRE(bytes(input) == packed);

    void *saved = selected.cudaData;
    auto shape = selected.dims;
    auto reject = [&]() {
        REQUIRE(!FastllmCudaQuantizeLinearWeightNVFP4Block16Rows(input, selected, ids));
        REQUIRE(selected.cudaData == saved && selected.dims == shape);
        REQUIRE(selected.scales == expected.scales && bytes(selected) == got);
    };
    input.strides[0] += 16; reject(); input.strides[0] -= 16;
    input.IsRepacked = true; reject(); input.IsRepacked = false;
    const int axis = input.perChannelAxis;
    input.perChannelAxis = 1; reject(); input.perChannelAxis = axis;
    const float scale = input.scales.back();
    input.scales.pop_back(); reject(); input.scales.push_back(scale);
    for (float invalid : {-1.0f, 1.0e10f, std::numeric_limits<float>::quiet_NaN()}) {
        input.scales.back() = invalid; reject();
    }
    input.scales.back() = scale;
    if (zeroPoint) {
        if (type == INT4 && configZeros) {
            const auto zero = input.perChannelsConfigs.back().zeroPoint;
            input.perChannelsConfigs.back().zeroPoint = 16; reject();
            input.perChannelsConfigs.back().zeroPoint = zero;
        } else {
            const int zero = input.zeros.back();
            input.zeros.back() = 16; reject(); input.zeros.back() = zero;
            if (count > 1) { input.zeros.pop_back(); reject(); input.zeros.push_back(zero); }
        }
    } else {
        const float minimum = input.mins.back();
        input.mins.pop_back(); reject(); input.mins.push_back(minimum);
        input.mins.back() = INFINITY; reject(); input.mins.back() = minimum;
    }
    if (grouped) {
        input.group++; reject(); input.group--;
        input.groupCnt = 0; reject(); input.groupCnt = groupSize;
        input.perChannelAxis = -1; reject(); input.perChannelAxis = axis;
    }
    // Zero-scale weights are valid and must not introduce NaNs or change input.
    std::fill(input.scales.begin(), input.scales.end(), 0.0f);
    std::fill(input.mins.begin(), input.mins.end(), 0.0f);
    REQUIRE(FastllmCudaQuantizeLinearWeightNVFP4Block16Rows(input, selected, ids));
    REQUIRE(std::isfinite(selected.scales[0]) && selected.scales[0] > 0);
    REQUIRE(bytes(input) == packed);
    printf("PASS INT4 device=%d dtype=%d shape=%dx%d groupSize=%d zeroPoint=%d perTensor=%d configs=%d\n",
           device, int(type), rows, columns, groupSize, zeroPoint, perTensor, configZeros);
}

int main(int argc, char **argv) {
    int device = argc > 1 ? std::atoi(argv[1]) : 0;
    CK(cudaSetDevice(device));
    checkType<half>(FLOAT16, device);
    checkType<__nv_bfloat16>(BFLOAT16, device);
    checkType<__nv_fp8_e4m3>(FP8_E4M3, device);
    checkInt4(INT4_NOZERO, device, 7, 80, 80, false);
    checkInt4(INT4_NOZERO, device, 3, 48, 48, false, true);
    checkInt4(INT4, device, 7, 80, 80, true);
    checkInt4(INT4, device, 3, 48, 48, true, true, false);
    for (bool zero : {false, true}) {
        checkInt4(INT4_GROUP, device, 7, 80, 32, zero);
        checkInt4(INT4_GROUP, device, 7, 80, 7, zero);
        checkInt4(INT4_GROUP, device, 3, 272, 128, zero);
        checkInt4(INT4_GROUP, device, 1, 16, 128, zero);
    }
    CK(cudaDeviceSynchronize());
    return 0;
}
