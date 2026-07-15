//
// Created by huangyuyang on 8/2/24.
//

#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
#include <set>

#include "fastllm-cuda.cuh"
#include "fastllm-multicuda.cuh"
#include "fastllm.h"
#include "utils.h"
#include "gguf.h"

#include "devices/cpu/alivethreadpool.h"
#include "devices/cpu/cpudevice.h"
#include "devices/cpu/computeutils.h"

#include <cuda_runtime.h>
#include <nccl.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 // support tensor core
#include "mma.h"
using namespace nvcuda;
#endif

#ifdef USE_ROCM
#include "fastllm-hip.h"
#endif

#define checkCudaErrors(message, val) showError(val, message, __FILE__, __LINE__)
extern void showError(cudaError_t result, char const* const message, const char* const file, int const line);

cudaError_t FastllmCudaMemcpy2D(void* dst, size_t dpitch, const void* src,
    size_t spitch, size_t width, size_t height, cudaMemcpyKind type, 
    int dstDeviceId, int srcDeviceId) {
#if defined(USE_ROCM) && defined(USE_MI50_WORKAROUND)
    // MI50 的跨卡 2D D2D 拷贝可能出错，保留旧的 host staging 兼容路径。
    if (type == cudaMemcpyDeviceToDevice && srcDeviceId != dstDeviceId) {
        std::vector <uint8_t> hostBuffer(height * width);
        cudaSetDevice(srcDeviceId);
        cudaError_t workaroundState =
            cudaMemcpy2D(hostBuffer.data(), width, src, spitch, width, height, cudaMemcpyDeviceToHost);

        if (workaroundState == cudaSuccess) {
            cudaSetDevice(dstDeviceId);
            workaroundState = cudaMemcpy2D(dst, dpitch, hostBuffer.data(), width, width, height, cudaMemcpyHostToDevice);
        }

        cudaDeviceSynchronize();
        if (workaroundState != cudaSuccess) {
            checkCudaErrors("Error: CUDA error when memcpy2D!", workaroundState);
        }
        return workaroundState;
    }
#endif

    cudaError_t state = cudaSuccess;
    if (type == cudaMemcpyDeviceToDevice) {
        cudaSetDevice(dstDeviceId);
        if (dstDeviceId != srcDeviceId) {
            state = cudaMemcpy2D(dst, dpitch, src, spitch,
                                 width, height, cudaMemcpyDeviceToDevice);
            if (state != cudaSuccess) {
                cudaGetLastError();
                size_t maxStagingBytes = 16 * 1024 * 1024;
                size_t chunkRows = std::max<size_t>(1, std::min(height, maxStagingBytes / std::max<size_t>(1, width)));
                std::vector<uint8_t> hostBuffer(chunkRows * width);
                for (size_t base = 0; base < height && state == cudaSuccess; base += chunkRows) {
                    size_t curRows = std::min(chunkRows, height - base);
                    cudaSetDevice(srcDeviceId);
                    for (size_t row = 0; row < curRows; row++) {
                        state = cudaMemcpy(hostBuffer.data() + row * width,
                                           (const uint8_t*)src + (base + row) * spitch,
                                           width, cudaMemcpyDeviceToHost);
                        if (state != cudaSuccess) {
                            break;
                        }
                    }
                    if (state == cudaSuccess) {
                        cudaSetDevice(dstDeviceId);
                        state = cudaMemcpy2D((uint8_t*)dst + base * dpitch, dpitch,
                                             hostBuffer.data(), width,
                                             width, curRows, cudaMemcpyHostToDevice);
                    }
                }
            }
        } else {
            state = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, type);
        }
    } else {
        state = cudaMemcpy2D(dst, dpitch, src, spitch, width, height, type);
    }

#ifdef USE_ROCM
    cudaDeviceSynchronize();
#endif

    if (state != cudaSuccess) {
        checkCudaErrors("Error: CUDA error when memcpy2D!", state);
    }
    return state;
}

void FastllmCudaMemcpy2DDeviceToDeviceAuto(void * 	dst, size_t 	dpitch, const void * 	src,
    size_t 	spitch, size_t 	width, size_t 	height, int dstDeviceId, int srcDeviceId) {
    FastllmCudaMemcpy2D(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, dstDeviceId, srcDeviceId);
}

void FastllmCudaSyncDevice(int deviceId) {
    cudaSetDevice(deviceId);
    cudaError_t state = cudaDeviceSynchronize();
    checkCudaErrors("Error: CUDA sync failed!", state);
}

std::map <int, std::string> specialDeviceIds = {
    {99999, "cpu"}
};

void SwitchDeviceAndGetInfos(int deviceId, std::string &specialId, int &mallocType) {
    specialId = "";
    if (specialDeviceIds.find(deviceId) == specialDeviceIds.end()) {
        cudaSetDevice(deviceId);
    } else {
        specialId = specialDeviceIds[deviceId];
    }
    mallocType = 1;
    if (specialId == "cpu") {
        mallocType = 0;
    }
}

/*
type: device type (0 for cpu, 1 for cuda)
*/
void *AutoMalloc(size_t size, int type) {
    if (type == 0) {
        return (void*)(new uint8_t[size]);
    } else {
        return (void*)FastllmCudaMalloc(size);
    }
}

cudaError_t AutoMemset(void *a, int value, size_t size, int type) {
    if (type == 0) {
        memset(a, value, size);
        return cudaSuccess;
    } else {
        return cudaMemset(a, value, size);
    }
}

cudaMemcpyKind GetCudaMemcpyType(int dstType, int srcType) {
    if (srcType == 0) {
        if (dstType == 0) {
            return cudaMemcpyHostToHost;
        } else {
            return cudaMemcpyHostToDevice;
        }
    } else {
        if (dstType == 0) {
            return cudaMemcpyDeviceToHost;
        } else {
            return cudaMemcpyDeviceToDevice;
        }
    }
}

std::vector <int> multiCudaCurrentDevices;
std::map <int, int> multiCudaCurrentRatios;
static std::set<int> multiCudaExplicitRatioDevices;

static bool FastllmMultiCudaLayerBalanceDisabled() {
    const char *env = std::getenv("FASTLLM_DISABLE_MULTICUDA_LAYER_BALANCE");
    if (env == nullptr || env[0] == '\0') {
        return false;
    }
    std::string value(env);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return (char)std::tolower(c);
    });
    return value != "0" && value != "false" && value != "off";
}

static bool ParseIntAt(const std::string &s, size_t pos, int &value) {
    if (pos >= s.size() || !std::isdigit((unsigned char)s[pos])) {
        return false;
    }
    long long v = 0;
    while (pos < s.size() && std::isdigit((unsigned char)s[pos])) {
        v = v * 10 + (s[pos] - '0');
        if (v > 1000000000LL) {
            return false;
        }
        pos++;
    }
    value = (int)v;
    return true;
}

static int ParseLayerIdFromWeightName(const std::string &name) {
    static const char *tags[] = {
        "layers.", "layer.", "blocks.", "block.", "h."
    };
    for (const char *tag : tags) {
        size_t tagLen = strlen(tag);
        size_t pos = 0;
        while ((pos = name.find(tag, pos)) != std::string::npos) {
            int layer = -1;
            if (ParseIntAt(name, pos + tagLen, layer)) {
                return layer;
            }
            pos += tagLen;
        }
    }
    return -1;
}

static bool MultiCudaDevicesHaveExplicitRatios(const std::vector<int> &devices) {
    if (devices.empty() || multiCudaExplicitRatioDevices.empty()) {
        return false;
    }
    for (int device : devices) {
        if (multiCudaExplicitRatioDevices.find(device) == multiCudaExplicitRatioDevices.end()) {
            return false;
        }
    }
    return true;
}

static bool MultiCudaLayerBalanceCanRotate(const std::vector<int> &devices,
                                           bool explicitDeviceRatios) {
    if (devices.size() <= 1 || FastllmMultiCudaLayerBalanceDisabled()) {
        return false;
    }
    if (explicitDeviceRatios || MultiCudaDevicesHaveExplicitRatios(devices)) {
        return false;
    }
    for (int device : devices) {
        if (specialDeviceIds.find(device) != specialDeviceIds.end()) {
            return false;
        }
    }
    int firstRatio = -1;
    for (int device : devices) {
        auto it = multiCudaCurrentRatios.find(device);
        int ratio = it == multiCudaCurrentRatios.end() ? 1 : std::max(0, it->second);
        if (firstRatio < 0) {
            firstRatio = ratio;
        } else if (ratio != firstRatio) {
            return false;
        }
    }
    return true;
}

static int FindFirstRangeOwnerIndex(const DivisionScheme &scheme, const std::vector<int> &devices) {
    bool found = false;
    int bestStart = 0;
    int bestIndex = -1;
    for (int i = 0; i < (int)devices.size(); i++) {
        auto it = scheme.find(devices[i]);
        if (it == scheme.end()) {
            continue;
        }
        for (auto &range : it->second) {
            if (range.second <= range.first) {
                continue;
            }
            if (!found || range.first < bestStart ||
                (range.first == bestStart && i < bestIndex)) {
                found = true;
                bestStart = range.first;
                bestIndex = i;
            }
        }
    }
    return bestIndex;
}

void BalanceMultiCudaDivisionSchemeByLayer(const std::string &weightName,
                                           const std::vector<int> &devices,
                                           DivisionScheme &scheme,
                                           bool explicitDeviceRatios) {
    if (!MultiCudaLayerBalanceCanRotate(devices, explicitDeviceRatios)) {
        return;
    }
    int layer = ParseLayerIdFromWeightName(weightName);
    if (layer < 0) {
        return;
    }
    int shift = layer % (int)devices.size();
    if (shift == 0) {
        return;
    }

    int firstOwner = FindFirstRangeOwnerIndex(scheme, devices);
    if (firstOwner != 0) {
        return;
    }

    DivisionScheme original = scheme;
    DivisionScheme rotated;
    std::set<int> deviceSet(devices.begin(), devices.end());
    for (int device : devices) {
        rotated[device];
    }
    for (int i = 0; i < (int)devices.size(); i++) {
        int srcDevice = devices[i];
        int dstDevice = devices[(i + shift) % (int)devices.size()];
        auto it = original.find(srcDevice);
        if (it != original.end()) {
            rotated[dstDevice] = it->second;
        }
    }
    for (auto &it : original) {
        if (deviceSet.find(it.first) == deviceSet.end()) {
            rotated[it.first] = it.second;
        }
    }
    scheme.swap(rotated);
}

void BalanceMultiCudaPairedHalfDivisionSchemeSizesByLayer(const std::string &weightName,
                                                          const std::vector<int> &devices,
                                                          DivisionScheme &scheme,
                                                          int mid,
                                                          bool explicitDeviceRatios) {
    if (!MultiCudaLayerBalanceCanRotate(devices, explicitDeviceRatios) || mid <= 0) {
        return;
    }
    int layer = ParseLayerIdFromWeightName(weightName);
    if (layer < 0) {
        return;
    }
    int shift = layer % (int)devices.size();
    if (shift == 0) {
        return;
    }

    std::vector<int> sizes(devices.size(), 0);
    int total = 0;
    for (int i = 0; i < (int)devices.size(); i++) {
        auto it = scheme.find(devices[i]);
        if (it == scheme.end()) {
            continue;
        }
        for (auto &range : it->second) {
            int l = std::max(0, std::min(mid, range.first));
            int r = std::max(0, std::min(mid, range.second));
            if (l < r) {
                sizes[i] += r - l;
                total += r - l;
            }
        }
    }
    if (total != mid) {
        return;
    }

    std::vector<int> rotatedSizes(devices.size(), 0);
    for (int i = 0; i < (int)devices.size(); i++) {
        rotatedSizes[(i + shift) % (int)devices.size()] = sizes[i];
    }

    DivisionScheme rotated;
    int offset = 0;
    for (int i = 0; i < (int)devices.size(); i++) {
        int device = devices[i];
        int len = rotatedSizes[i];
        rotated[device];
        if (len > 0) {
            rotated[device].push_back({offset, offset + len});
            rotated[device].push_back({mid + offset, mid + offset + len});
        }
        offset += len;
    }
    if (offset == mid) {
        scheme.swap(rotated);
    }
}

static void BalanceDivisionSchemeByLayer(const fastllm::Data &weight,
                                         const std::vector<int> &devices,
                                         DivisionScheme &scheme,
                                         bool explicitDeviceRatios) {
    BalanceMultiCudaDivisionSchemeByLayer(weight.name, devices, scheme, explicitDeviceRatios);
}

void FastllmMultiCudaSetDevice(std::vector <int> ids) {
    multiCudaCurrentDevices = ids;
}

void FastllmMultiCudaSetDeviceRatio(std::map <int, int> &deviceRatio) {
    multiCudaCurrentRatios = deviceRatio;
    multiCudaExplicitRatioDevices.clear();
    for (auto &it : deviceRatio) {
        multiCudaExplicitRatioDevices.insert(it.first);
    }
}

void FastllmGetMulticudaDeviceAndRatio(std::vector <int> &devices, std::map <int, int> &ratios, bool noSpecial) {
    devices.clear();
    ratios.clear();
    for (int i : multiCudaCurrentDevices) {
        if (noSpecial == false || specialDeviceIds.find(i) == specialDeviceIds.end()) {
            devices.push_back(i);
            ratios[i] = multiCudaCurrentRatios.find(i) != multiCudaCurrentRatios.end() ? multiCudaCurrentRatios[i] : 1;
        }
    }
}

// 将total个计算任务切分
// 若当前有x个设备，返回一个长度为(x + 1)的vector，第i个设备执行任务[ret[i], ret[i + 1])
std::vector <int> FastllmMultiCudaGetSplitPoints(std::vector <int> &multiCudaCurrentDevices, 
                                std::map <int, int> &multiCudaCurrentRatios, int total, int unit = 1) {
    int deviceNum = multiCudaCurrentDevices.size();
    std::vector <int> ret;
    if (deviceNum <= 0) {
        return ret;
    }
    if (total <= 0) {
        ret.resize(deviceNum + 1, 0);
        return ret;
    }
    unit = std::max(1, total > 0 ? std::min(unit, total) : unit);
    int nodes = total / unit;
    int totalRatio = 0;
    if (multiCudaCurrentRatios.size() > 0) {
        for (auto &it : multiCudaCurrentRatios) {
            totalRatio += std::max(0, it.second);
        }
    }
    if (totalRatio <= 0) {
        totalRatio = deviceNum;
    }

    std::vector<int> ratios(deviceNum, 1);
    std::vector<int> units(deviceNum, 0);
    std::vector<long long> remainders(deviceNum, 0);
    int usedNodes = 0;
    for (int i = 0; i < deviceNum; i++) {
        auto ratioIt = multiCudaCurrentRatios.find(multiCudaCurrentDevices[i]);
        ratios[i] = ratioIt == multiCudaCurrentRatios.end() ? 1 : std::max(0, ratioIt->second);
        long long scaled = (long long)nodes * ratios[i];
        units[i] = (int)(scaled / totalRatio);
        remainders[i] = scaled % totalRatio;
        usedNodes += units[i];
    }
    while (usedNodes < nodes) {
        int best = 0;
        for (int i = 1; i < deviceNum; i++) {
            if (remainders[i] > remainders[best] ||
                (remainders[i] == remainders[best] && ratios[i] > ratios[best])) {
                best = i;
            }
        }
        units[best]++;
        remainders[best] = -1;
        usedNodes++;
    }

    int cur = 0;
    for (int i = 0; i < deviceNum; i++) {
        int now = units[i] * unit;
        int end = (i == deviceNum - 1 ? total : cur + now);
        ret.push_back(cur);
        if (i == deviceNum - 1) {
            ret.push_back(end);
        }
        cur = end;
    }
    return ret;
}

static int GcdInt(int a, int b) {
    a = a < 0 ? -a : a;
    b = b < 0 ? -b : b;
    while (b != 0) {
        int t = a % b;
        a = b;
        b = t;
    }
    return a == 0 ? 1 : a;
}

static int LcmInt(int a, int b) {
    a = std::max(1, a);
    b = std::max(1, b);
    return a / GcdInt(a, b) * b;
}

static bool IsGGUFTensor(const fastllm::Data &data) {
    return data.dataType == fastllm::DataType::DATA_GGUF_FORMAT && data.ggmlType >= 0;
}

static int GetGGUFBlockSize(const fastllm::Data &data) {
    return IsGGUFTensor(data) ? (int)ggml_blck_size((ggml_type)data.ggmlType) : 1;
}

static size_t GetGGUFRowBytes(const fastllm::Data &data, int columns) {
    int blockSize = GetGGUFBlockSize(data);
    fastllm::AssertInFastLLM(blockSize > 0 && columns % blockSize == 0,
                             "GGUF split requires aligned columns, got " + std::to_string(columns) +
                             " for block size " + std::to_string(blockSize) + ".\n");
    return ggml_row_size((ggml_type)data.ggmlType, columns);
}

static bool IsRowContiguousNumaGateUpSource(const fastllm::Data &data) {
    if (data.tpPackType != fastllm::TP_PACK_GATEUP || data.numasData.empty()) {
        return false;
    }
    if (IsGGUFTensor(data)) {
        return true;
    }
    return data.dataType == fastllm::DataType::FLOAT32 ||
           data.dataType == fastllm::DataType::BFLOAT16 ||
           data.dataType == fastllm::DataType::FLOAT16 ||
           data.dataType == fastllm::DataType::FP8_E4M3_BLOCK_128 ||
           data.dataType == fastllm::DataType::FP8_E4M3_PERCHANNEL;
}

static size_t GetRowContiguousNumaGateUpRowBytes(const fastllm::Data &data, int columns) {
    return IsGGUFTensor(data) ? GetGGUFRowBytes(data, columns)
                              : fastllm::GetDataBytes(data.dataType, 1, columns);
}

static int GetMultiCudaSplitUnit(const fastllm::Data &data, int splitAxis) {
    int unit = data.groupCnt <= 0 ? 128 : data.groupCnt;
    if (data.dataType == fastllm::DataType::FP8_E4M3) {
        int blockSize = splitAxis == 0 ? data.blockK : data.blockM;
        if (blockSize > 0) {
            unit = blockSize;
        }
    } else if (data.dataType == fastllm::DataType::FP8_E4M3_BLOCK_128) {
        unit = 128;
    }
    if (data.dataType == fastllm::DataType::NVFP4) {
        if (data.blockK > 0) {
            unit = LcmInt(unit, data.blockK);
        }
        if (data.blockM > 0) {
            unit = LcmInt(unit, data.blockM);
        }
    }
    if (IsGGUFTensor(data)) {
        unit = LcmInt(unit, GetGGUFBlockSize(data));
    }
    return std::max(1, unit);
}

static int GetGGUFHeadSplitUnit(const fastllm::Data &data, int widthPerHead) {
    if (!IsGGUFTensor(data)) {
        return 1;
    }
    int blockSize = GetGGUFBlockSize(data);
    int gcd = GcdInt(blockSize, std::max(1, widthPerHead));
    return std::max(1, blockSize / gcd);
}

static void SetLocalQKVPackMeta(fastllm::Data *local, const fastllm::Data &weight,
                                const std::vector<std::pair<int, int> > &div) {
    if (local == nullptr || weight.tpPackType != fastllm::TP_PACK_QKV) {
        return;
    }
    int headDim = std::max(1, weight.tpHeadDim);
    int qWidth = weight.tpQHeads * headDim;
    int kvWidth = weight.tpKVHeads * headDim;
    int qRows = 0, kRows = 0, vRows = 0;
    for (auto &range : div) {
        if (range.first >= 0 && range.second <= qWidth) {
            qRows += range.second - range.first;
        } else if (range.first >= qWidth && range.second <= qWidth + kvWidth) {
            kRows += range.second - range.first;
        } else if (range.first >= qWidth + kvWidth && range.second <= qWidth + 2 * kvWidth) {
            vRows += range.second - range.first;
        }
    }
    if (qRows > 0 && kRows > 0 && kRows == vRows &&
        qRows % headDim == 0 && kRows % headDim == 0) {
        local->tpQHeads = qRows / headDim;
        local->tpKVHeads = kRows / headDim;
        local->tpHeadDim = headDim;
    }
}

static void InitMultiCudaLocalTensorMeta(const fastllm::Data &src, fastllm::Data &dst) {
    dst.name = src.name;
    dst.isModelWeight = src.isModelWeight;
    dst.group = src.group;
    dst.groupCnt = src.groupCnt;
    dst.blockK = src.blockK;
    dst.blockM = src.blockM;
    dst.perChannelAxis = src.perChannelAxis;
    dst.isGGUFData = src.isGGUFData || src.dataType == fastllm::DataType::DATA_GGUF_FORMAT;
    dst.ggmlType = src.ggmlType;
    dst.IsRepacked = src.IsRepacked;
    dst.tpLinearType = src.tpLinearType;
    dst.tpPackType = src.tpPackType;
    dst.tpQHeads = src.tpQHeads;
    dst.tpKVHeads = src.tpKVHeads;
    dst.tpHeadDim = src.tpHeadDim;
}

static fastllm::Data *CreateMultiCudaLocalTensor(const fastllm::Data &src, const std::vector <int> &dims) {
    fastllm::Data *local = nullptr;
    if (dims.empty()) {
        local = new fastllm::Data(src.dataType);
    } else if (src.dataType == fastllm::DataType::DATA_GGUF_FORMAT) {
        fastllm::AssertInFastLLM(src.ggmlType >= 0,
                                 "GGUF tensor \"" + src.name + "\" is missing ggmlType when creating multicuda local tensor.\n");
        local = new fastllm::Data(src.dataType, src.ggmlType, dims);
    } else {
        local = new fastllm::Data(src.dataType, dims);
    }
    InitMultiCudaLocalTensorMeta(src, *local);
    return local;
}

static void ClearGGUFExtraCudaCaches(fastllm::Data &data) {
#ifdef USE_CUDA
    if (!IsGGUFTensor(data)) {
        return;
    }
    for (void *ptr : data.extraCudaData) {
        if (ptr != nullptr) {
            FastllmCudaFree(ptr);
        }
    }
    for (void *ptr : data.extraCudaHalfData) {
        if (ptr != nullptr) {
            FastllmCudaFree(ptr);
        }
    }
#endif
    data.extraCudaData.clear();
    data.extraCudaHalfData.clear();
}

static void ResetMultiDeviceData(fastllm::Data &data) {
    if (!data.multiDeviceData) {
        data.ClearTensorParallelLayout();
        return;
    }
    for (auto &it : data.multiDeviceDatas) {
        delete it.second;
    }
    data.multiDeviceDatas.clear();
    data.multiDeviceData = false;
    data.ClearTensorParallelLayout();
}

void CopyToMultiDevices(fastllm::Data &data, std::vector <int> devices, bool copyData) {
    if (data.multiDeviceData) {
        return;
    }
    data.multiDeviceData = true;
    int oriId = FastllmCudaGetDevice();

    if (copyData) {
        bool srcOnGpu = (data.dataDevice == fastllm::DataDevice::CUDA
                         && data.cudaData != nullptr
                         && !data.dataDeviceIds.empty());
        int srcDevice = srcOnGpu ? data.dataDeviceIds[0] : -1;

        if (srcOnGpu) {
            uint64_t bytes = data.GetBytes();
            for (int device : devices) {
                FastllmCudaSetDevice(device);
                auto *local = CreateMultiCudaLocalTensor(data, data.dims);
                local->dataDevice = fastllm::DataDevice::CUDA;
                local->dataDeviceIds = {device};
                local->Allocate();

                if (device == srcDevice) {
                    cudaMemcpy(local->cudaData, data.cudaData, bytes, cudaMemcpyDeviceToDevice);
                } else {
                    cudaMemcpyPeer(local->cudaData, device, data.cudaData, srcDevice, bytes);
                }

                local->group = data.group;
                local->groupCnt = data.groupCnt;
                local->scales = data.scales;
                local->mins = data.mins;
                local->zeros = data.zeros;
                local->halfScales = data.halfScales;
                data.multiDeviceDatas[device] = local;
            }
        } else {
            data.ToDevice(fastllm::DataDevice::CPU);
            for (int device : devices) {
                int mallocType = 0;
                std::string specialId = "";
                SwitchDeviceAndGetInfos(device, specialId, mallocType);
                fastllm::DataDevice dataDevice = (mallocType == 0 ? fastllm::DataDevice::CPU :fastllm::DataDevice::CUDA);

                data.multiDeviceDatas[device] = new fastllm::Data();
                data.multiDeviceDatas[device]->CopyFrom(data);
                InitMultiCudaLocalTensorMeta(data, *data.multiDeviceDatas[device]);
                data.multiDeviceDatas[device]->ToDevice(dataDevice, std::vector <int> {device});
                data.multiDeviceDatas[device]->dataDeviceIds = {device};

                data.multiDeviceDatas[device]->group = data.group;
                data.multiDeviceDatas[device]->groupCnt = data.groupCnt;
                data.multiDeviceDatas[device]->scales = data.scales;
                data.multiDeviceDatas[device]->mins = data.mins;
                data.multiDeviceDatas[device]->zeros = data.zeros;
                data.multiDeviceDatas[device]->halfScales = data.halfScales;
            }
        }
    } else {
        for (int device : devices) {
            int mallocType = 0;
            std::string specialId = "";
            SwitchDeviceAndGetInfos(device, specialId, mallocType);
            fastllm::DataDevice dataDevice = (mallocType == 0 ? fastllm::DataDevice::CPU :fastllm::DataDevice::CUDA);
            data.multiDeviceDatas[device] = CreateMultiCudaLocalTensor(data, data.dims);
            data.multiDeviceDatas[device]->dataDevice = dataDevice;
            data.multiDeviceDatas[device]->dataDeviceIds = {device};
        }
    }
    FastllmCudaSetDevice(oriId);
}

void PrepareMultiCudaReplicatedData(fastllm::Data &data, std::vector <int> devices, bool copyData) {
    if (data.IsTensorParallelReplicated() && data.multiDeviceData) {
        return;
    }
    auto oriDevice = data.dataDevice;
    auto oriDeviceIds = data.dataDeviceIds;
    ResetMultiDeviceData(data);
    CopyToMultiDevices(data, devices, copyData);
    if (copyData && oriDevice == fastllm::DataDevice::CUDA) {
        data.ToDevice(oriDevice, oriDeviceIds, true);
    }
    data.tpLayout = fastllm::TP_LAYOUT_REPLICATED;
    data.tpAxis = -1;
    data.tpGlobalDims = data.dims;
}

void PrepareMultiCudaShardedData(fastllm::Data &data, std::vector <int> devices,
    const std::vector <int> &globalDims, int axis, DivisionScheme divisionScheme) {
    ResetMultiDeviceData(data);
    data.multiDeviceData = true;
    data.tpLayout = fastllm::TP_LAYOUT_SHARDED;
    data.tpAxis = axis;
    data.tpGlobalDims = globalDims;
    data.tpRanges = divisionScheme;
    data.dataDevice = fastllm::DataDevice::CUDA;
    data.dataDeviceIds = devices;
    data.Resize(globalDims);
    int oriId = FastllmCudaGetDevice();
    for (int device : devices) {
        int mallocType = 0;
        std::string specialId = "";
        SwitchDeviceAndGetInfos(device, specialId, mallocType);
        fastllm::DataDevice dataDevice = (mallocType == 0 ? fastllm::DataDevice::CPU : fastllm::DataDevice::CUDA);
        std::vector <int> localDims = globalDims;
        int len = 0;
        for (auto &range : divisionScheme[device]) {
            len += range.second - range.first;
        }
        localDims[axis] = len;
        data.multiDeviceDatas[device] = CreateMultiCudaLocalTensor(data, localDims);
        data.multiDeviceDatas[device]->dataDevice = dataDevice;
        data.multiDeviceDatas[device]->dataDeviceIds = {device};
    }
    data.cudaData = nullptr;
    FastllmCudaSetDevice(oriId);
}

DivisionScheme BuildMultiCudaRowSplitScheme(fastllm::Data &weight, std::vector <int> &devices, std::map <int, int> &ratios) {
    DivisionScheme divisionScheme;
    if (weight.tpPackType == fastllm::TP_PACK_QKV) {
        int kvHeads = weight.tpKVHeads;
        int qHeads = weight.tpQHeads;
        int headDim = weight.tpHeadDim;
        int group = qHeads / kvHeads;
        int qWidth = qHeads * headDim;
        int kvWidth = kvHeads * headDim;
        int headUnit = GetGGUFHeadSplitUnit(weight, group * headDim);
        fastllm::AssertInFastLLM(kvHeads > 0 && qHeads > 0 && headDim > 0 && qHeads % kvHeads == 0,
                                 "QKV tensor parallel requires valid q/kv head metadata.\n");
        fastllm::AssertInFastLLM(!IsGGUFTensor(weight) || kvHeads % headUnit == 0,
                                 "GGUF QKV tensor parallel requires kv heads aligned to head unit " +
                                 std::to_string(headUnit) + ".\n");
        if ((int)devices.size() > kvHeads) {
            fastllm::AssertInFastLLM(!IsGGUFTensor(weight),
                                     "GGUF QKV tensor parallel doesn't support splitting one KV head across devices.\n");
            fastllm::AssertInFastLLM((int)devices.size() <= qHeads,
                                     "QKV tensor parallel has more devices than query heads.\n");
            int offset = 0;
            for (int kv = 0; kv < kvHeads; kv++) {
                int leftDevices = (int)devices.size() - offset;
                int leftGroups = kvHeads - kv;
                int take = (leftDevices + leftGroups - 1) / leftGroups;
                std::vector<int> groupDevices;
                std::map<int, int> groupRatios;
                for (int j = 0; j < take; j++) {
                    int deviceId = devices[offset + j];
                    groupDevices.push_back(deviceId);
                    groupRatios[deviceId] = ratios.find(deviceId) == ratios.end() ? 1 : ratios[deviceId];
                }
                std::vector<int> qPoints = FastllmMultiCudaGetSplitPoints(groupDevices, groupRatios, group, 1);
                for (int j = 0; j < take; j++) {
                    int deviceId = groupDevices[j];
                    int qSt = kv * group + qPoints[j];
                    int qEnd = kv * group + qPoints[j + 1];
                    fastllm::AssertInFastLLM(qSt < qEnd && qEnd <= (kv + 1) * group,
                                             "QKV tensor parallel got invalid query head split.\n");
                    divisionScheme[deviceId].push_back({qSt * headDim, qEnd * headDim});
                    divisionScheme[deviceId].push_back({qWidth + kv * headDim, qWidth + (kv + 1) * headDim});
                    divisionScheme[deviceId].push_back({qWidth + kvWidth + kv * headDim,
                                                        qWidth + kvWidth + (kv + 1) * headDim});
                }
                offset += take;
            }
            return divisionScheme;
        }

        std::vector <int> points = FastllmMultiCudaGetSplitPoints(devices, ratios, kvHeads, headUnit);
        for (int i = 0; i < devices.size(); i++) {
            int deviceId = devices[i];
            int st = points[i], end = points[i + 1];
            divisionScheme[deviceId].push_back({st * group * headDim, end * group * headDim});
            divisionScheme[deviceId].push_back({qWidth + st * headDim, qWidth + end * headDim});
            divisionScheme[deviceId].push_back({qWidth + kvWidth + st * headDim, qWidth + kvWidth + end * headDim});
        }
        return divisionScheme;
    }

    if (weight.tpPackType == fastllm::TP_PACK_GATEUP) {
        int mid = weight.dims[0] / 2;
        int unit = GetMultiCudaSplitUnit(weight, 0);
        fastllm::AssertInFastLLM(!IsGGUFTensor(weight) || mid % unit == 0,
                                 "GGUF GateUp tensor parallel requires aligned split unit " +
                                 std::to_string(unit) + ".\n");
        std::vector <int> points = FastllmMultiCudaGetSplitPoints(devices, ratios, mid, unit);
        for (int i = 0; i < devices.size(); i++) {
            int deviceId = devices[i];
            int st = points[i], end = points[i + 1];
            divisionScheme[deviceId].push_back({st, end});
            divisionScheme[deviceId].push_back({mid + st, mid + end});
        }
        return divisionScheme;
    }

    int unit = GetMultiCudaSplitUnit(weight, 0);
    fastllm::AssertInFastLLM(!IsGGUFTensor(weight) || weight.dims[0] % unit == 0,
                             "GGUF row split requires aligned split unit " + std::to_string(unit) + ".\n");
    std::vector <int> points = FastllmMultiCudaGetSplitPoints(devices, ratios, weight.dims[0], unit);
    for (int i = 0; i < devices.size(); i++) {
        divisionScheme[devices[i]].push_back({points[i], points[i + 1]});
    }
    return divisionScheme;
}

DivisionScheme BuildMultiCudaColumnSplitScheme(fastllm::Data &weight, std::vector <int> &devices, std::map <int, int> &ratios) {
    DivisionScheme divisionScheme;
    fastllm::AssertInFastLLM(weight.dims.size() == 2,
                             "Column split requires 2D weight \"" + weight.name + "\".\n");
    int unit = GetMultiCudaSplitUnit(weight, 1);
    fastllm::AssertInFastLLM(!IsGGUFTensor(weight) || weight.dims[1] % unit == 0,
                             "GGUF column split requires aligned split unit " + std::to_string(unit) + ".\n");
    std::vector <int> points = FastllmMultiCudaGetSplitPoints(devices, ratios, weight.dims[1], unit);
    for (int i = 0; i < devices.size(); i++) {
        divisionScheme[devices[i]].push_back({points[i], points[i + 1]});
    }
    return divisionScheme;
}

bool SplitMultiCudaWeight(fastllm::Data &weight, fastllm::Data &bias, 
                    std::vector <int> &multiCudaCurrentDevices, DivisionScheme &divisionScheme, int splitAxis,
                    bool explicitDeviceRatios) {
    int deviceNum = multiCudaCurrentDevices.size();
    int rootDevice = deviceNum > 0 ? multiCudaCurrentDevices[0] : 0;
    BalanceDivisionSchemeByLayer(weight, multiCudaCurrentDevices, divisionScheme, explicitDeviceRatios);
    auto hasRequestedLocalTensors = [&]() {
        if (!weight.multiDeviceData) {
            return false;
        }
        for (int deviceId : multiCudaCurrentDevices) {
            auto weightIt = weight.multiDeviceDatas.find(deviceId);
            if (weightIt == weight.multiDeviceDatas.end() || weightIt->second == nullptr) {
                return false;
            }
            if (bias.multiDeviceData || bias.dims.size() > 0) {
                auto biasIt = bias.multiDeviceDatas.find(deviceId);
                if (biasIt == bias.multiDeviceDatas.end() || biasIt->second == nullptr) {
                    return false;
                }
            }
        }
        return true;
    };
    if (hasRequestedLocalTensors()) {
        return true;
    }
    fastllm::AssertInFastLLM(weight.dataType != fastllm::DataType::DATA_GGUF_FORMAT || weight.ggmlType >= 0,
                             "GGUF weight \"" + weight.name + "\" is missing ggmlType before multicuda split.\n");
    if (weight.dataDevice != fastllm::DataDevice::CUDA || weight.cudaData == nullptr ||
        weight.dataDeviceIds.size() == 0 || weight.dataDeviceIds[0] != rootDevice) {
        weight.ToDevice(fastllm::DataDevice::CUDA, {rootDevice}, true);
        if (weight.cudaData == nullptr) {
            fastllm::ErrorInFastLLM("SplitMultiCudaWeight failed to move weight \"" + weight.name +
                                    "\" to CUDA root device " + std::to_string(rootDevice) + ".\n");
        }
    }
    if (bias.dims.size() > 0 &&
        (bias.dataDevice != fastllm::DataDevice::CUDA || bias.cudaData == nullptr ||
         bias.dataDeviceIds.size() == 0 || bias.dataDeviceIds[0] != rootDevice)) {
        bias.ToDevice(fastllm::DataDevice::CUDA, {rootDevice}, true);
        if (bias.cudaData == nullptr) {
            fastllm::ErrorInFastLLM("SplitMultiCudaWeight failed to move bias \"" + bias.name +
                                    "\" to CUDA root device " + std::to_string(rootDevice) + ".\n");
        }
    }

    for (int i = 0; i < deviceNum; i++) {
        if (specialDeviceIds.find(multiCudaCurrentDevices[i]) == specialDeviceIds.end()) {
            cudaSetDevice(multiCudaCurrentDevices[i]);
        } 
        for (int j = 0; j < deviceNum; j++) {
            if (i != j) {
                if (specialDeviceIds.find(multiCudaCurrentDevices[j]) == specialDeviceIds.end()) {
                    int canPeerAccess = 0;
                    cudaError_t peerState = cudaDeviceCanAccessPeer(&canPeerAccess,
                                                                    multiCudaCurrentDevices[i],
                                                                    multiCudaCurrentDevices[j]);
                    if (peerState == cudaSuccess && canPeerAccess) {
                        peerState = cudaDeviceEnablePeerAccess(multiCudaCurrentDevices[j], 0);
                        if (peerState == cudaErrorPeerAccessAlreadyEnabled) {
                            cudaGetLastError();
                        }
                    }
                }
            }
        }
    }
    cudaSetDevice(rootDevice);

    if (weight.multiDeviceData) {
        return true;
    }
    bool hasBias = bias.dims.size() > 0;
    weight.multiDeviceData = true;
    bias.multiDeviceData = true;
    int k = weight.dims[0], m = weight.dims[1];
    const size_t kSize = static_cast<size_t>(k);
    const size_t mSize = static_cast<size_t>(m);
    const size_t unitSize = static_cast<size_t>(weight.unitSize);
    const size_t unitSizeDiv = static_cast<size_t>(weight.unitSizeDiv);
    const auto packedByteCount = [unitSize, unitSizeDiv] (size_t rows, size_t cols) -> size_t {
        return rows * cols * unitSize / unitSizeDiv;
    };
    const size_t biasBytes = kSize * sizeof(float);
    cudaError_t state = cudaSuccess;
    float *cudaBiasData = nullptr;
    if (hasBias) {
        cudaBiasData = (float*)FastllmCudaMalloc(biasBytes);
        if (cudaBiasData == nullptr) {
            fastllm::ErrorInFastLLM("SplitMultiCudaWeight failed to allocate temporary bias for \"" +
                                    weight.name + "\".\n");
        }
        state = cudaMemcpy(cudaBiasData, (uint8_t *) bias.cudaData, biasBytes, cudaMemcpyDeviceToDevice);
    }
    if (state != cudaSuccess) {
        if (cudaBiasData != nullptr) {
            FastllmCudaFree(cudaBiasData);
        }
        ResetMultiDeviceData(weight);
        ResetMultiDeviceData(bias);
        checkCudaErrors("Error: CUDA error when split weight!", state);
        return false;
    }
        
    for (int i = 0; i < multiCudaCurrentDevices.size(); i++) {
        int deviceId = multiCudaCurrentDevices[i], mallocType = 0;
        std::string specialId = "";
        SwitchDeviceAndGetInfos(deviceId, specialId, mallocType);
        fastllm::DataDevice dataDevice = (mallocType == 0 ? fastllm::DataDevice::CPU :fastllm::DataDevice::CUDA);

        auto &div = divisionScheme[deviceId];
        int len = 0;
        for (auto &it : div) {
            len += it.second - it.first;
        }

        void *deviceWeightData;
        float *deviceBiasData;
        cudaError_t state = cudaSuccess;
        if (splitAxis == 0) {
            weight.multiDeviceDatas[deviceId] = CreateMultiCudaLocalTensor(weight, {len, m});
            weight.multiDeviceDatas[deviceId]->dataDevice = dataDevice;
            weight.multiDeviceDatas[deviceId]->dataDeviceIds = {deviceId};
            bias.multiDeviceDatas[deviceId] = CreateMultiCudaLocalTensor(bias, hasBias ? std::vector<int>{len} : std::vector<int>{});
            bias.multiDeviceDatas[deviceId]->dataDevice = dataDevice;
            bias.multiDeviceDatas[deviceId]->dataDeviceIds = {deviceId};
            SetLocalQKVPackMeta(weight.multiDeviceDatas[deviceId], weight, div);
            weight.multiDeviceDatas[deviceId]->Allocate();
            if (hasBias) {
                bias.multiDeviceDatas[deviceId]->Allocate();
            }

            deviceWeightData = mallocType == 0 ? weight.multiDeviceDatas[deviceId]->cpuData : weight.multiDeviceDatas[deviceId]->cudaData;
            deviceBiasData = hasBias ? (float*)(mallocType == 0 ? bias.multiDeviceDatas[deviceId]->cpuData : bias.multiDeviceDatas[deviceId]->cudaData) : nullptr;
            bool emptyShard = len == 0;
            if (!emptyShard && (deviceWeightData == nullptr || (hasBias && deviceBiasData == nullptr))) {
                state = cudaErrorMemoryAllocation;
            }
            int curLen = 0;
            bool sourceCrossGateUp = IsRowContiguousNumaGateUpSource(weight);
            if (state != cudaSuccess) {
                // handled by the common error path below
            } else if (emptyShard) {
                // This can happen when the number of GPUs is larger than the number of aligned TP blocks.
            } else if (sourceCrossGateUp) {
                int mid = k / 2;
                size_t rowBytes = GetRowContiguousNumaGateUpRowBytes(weight, m);
                for (auto &it : div) {
                    int copyLen = it.second - it.first;
                    size_t srcRow = it.first < mid
                        ? static_cast<size_t>(it.first) * 2
                        : static_cast<size_t>(it.first - mid) * 2 + 1;
                    state = FastllmCudaMemcpy2D(
                        (uint8_t*)deviceWeightData + (size_t)curLen * rowBytes,
                        rowBytes,
                        (uint8_t*)weight.cudaData + (size_t)srcRow * rowBytes,
                        rowBytes * 2,
                        rowBytes,
                        static_cast<size_t>(copyLen),
                        GetCudaMemcpyType(mallocType, 1),
                        deviceId,
                        rootDevice);
                    if (state != cudaSuccess) {
                        break;
                    }
                    if (hasBias) {
                        state = cudaMemcpy(deviceBiasData + curLen, cudaBiasData + it.first,
                                           (size_t)copyLen * sizeof(float), GetCudaMemcpyType(mallocType, 1));
                        if (state != cudaSuccess) {
                            break;
                        }
                    }
                    curLen += copyLen;
                }
            } else if (IsGGUFTensor(weight)) {
                size_t rowBytes = GetGGUFRowBytes(weight, m);
                if (mallocType == 0) {
                    cudaSetDevice(rootDevice);
                }
                for (auto &it : div) {
                    int copyLen = it.second - it.first;
                    state = cudaMemcpy((uint8_t*)deviceWeightData + (size_t)curLen * rowBytes,
                                       (uint8_t*)weight.cudaData + (size_t)it.first * rowBytes,
                                       (size_t)copyLen * rowBytes,
                                       GetCudaMemcpyType(mallocType, 1));
                    if (state != cudaSuccess) {
                        break;
                    }
                    if (hasBias) {
                        state = cudaMemcpy(deviceBiasData + curLen, cudaBiasData + it.first,
                                           (size_t)copyLen * sizeof(float), GetCudaMemcpyType(mallocType, 1));
                        if (state != cudaSuccess) {
                            break;
                        }
                    }
                    curLen += copyLen;
                }
            } else if (weight.dataType == fastllm::DataType::FP8_E4M3_BLOCK_128) {
                size_t rowBytes = fastllm::GetDataBytes(weight.dataType, 1, m);
                if (mallocType == 0) {
                    cudaSetDevice(rootDevice);
                }
                for (auto &it : div) {
                    int copyLen = it.second - it.first;
                    state = cudaMemcpy((uint8_t*)deviceWeightData + (size_t)curLen * rowBytes,
                                       (uint8_t*)weight.cudaData + (size_t)it.first * rowBytes,
                                       (size_t)copyLen * rowBytes,
                                       GetCudaMemcpyType(mallocType, 1));
                    if (state != cudaSuccess) {
                        break;
                    }
                    if (hasBias) {
                        state = cudaMemcpy(deviceBiasData + curLen, cudaBiasData + it.first,
                                           (size_t)copyLen * sizeof(float), GetCudaMemcpyType(mallocType, 1));
                        if (state != cudaSuccess) {
                            break;
                        }
                    }
                    curLen += copyLen;
                }
            } else if (weight.dataType == fastllm::DataType::NVFP4 &&
                       weight.scales.empty() && weight.blockK > 0 && weight.blockM > 0) {
                size_t rowBytes = fastllm::GetNVFP4WeightBytes(1, m);
                size_t srcWeightBytes = fastllm::GetNVFP4WeightBytes(k, m);
                size_t dstWeightBytes = fastllm::GetNVFP4WeightBytes(len, m);
                size_t scaleCols = static_cast<size_t>((m - 1) / weight.blockM + 1);
                if (mallocType == 0) {
                    cudaSetDevice(rootDevice);
                }
                for (auto &it : div) {
                    int copyLen = it.second - it.first;
                    fastllm::AssertInFastLLM(it.first % weight.blockK == 0 && copyLen % weight.blockK == 0,
                                             "NVFP4 tensor parallel row split should align to blockK.\n");
                    state = cudaMemcpy((uint8_t*)deviceWeightData + (size_t)curLen * rowBytes,
                                       (uint8_t*)weight.cudaData + (size_t)it.first * rowBytes,
                                       (size_t)copyLen * rowBytes,
                                       GetCudaMemcpyType(mallocType, 1));
                    if (state != cudaSuccess) {
                        break;
                    }
                    state = cudaMemcpy((uint8_t*)deviceWeightData + dstWeightBytes +
                                           static_cast<size_t>(curLen / weight.blockK) * scaleCols,
                                       (uint8_t*)weight.cudaData + srcWeightBytes +
                                           static_cast<size_t>(it.first / weight.blockK) * scaleCols,
                                       static_cast<size_t>(copyLen / weight.blockK) * scaleCols,
                                       GetCudaMemcpyType(mallocType, 1));
                    if (state != cudaSuccess) {
                        break;
                    }
                    if (hasBias) {
                        state = cudaMemcpy(deviceBiasData + curLen, cudaBiasData + it.first,
                                           (size_t)copyLen * sizeof(float), GetCudaMemcpyType(mallocType, 1));
                        if (state != cudaSuccess) {
                            break;
                        }
                    }
                    curLen += copyLen;
                }
            } else if (weight.dataType == fastllm::DataType::NVFP4_BLOCK_16 ||
                       weight.dataType == fastllm::DataType::NVFP4_BLOCK_16_E8M0) {
                size_t rowBytes = fastllm::GetDataBytes(weight.dataType, 1, m);
                if (mallocType == 0) {
                    cudaSetDevice(rootDevice);
                }
                for (auto &it : div) {
                    int copyLen = it.second - it.first;
                    state = cudaMemcpy((uint8_t*)deviceWeightData + (size_t)curLen * rowBytes,
                                       (uint8_t*)weight.cudaData + (size_t)it.first * rowBytes,
                                       (size_t)copyLen * rowBytes,
                                       GetCudaMemcpyType(mallocType, 1));
                    if (state != cudaSuccess) {
                        break;
                    }
                    if (hasBias) {
                        state = cudaMemcpy(deviceBiasData + curLen, cudaBiasData + it.first,
                                           (size_t)copyLen * sizeof(float), GetCudaMemcpyType(mallocType, 1));
                        if (state != cudaSuccess) {
                            break;
                        }
                    }
                    curLen += copyLen;
                }
            } else {
                for (auto &it : div) {
                    size_t dstOffsetBytes = packedByteCount(static_cast<size_t>(curLen), mSize);
                    size_t srcOffsetBytes = packedByteCount(static_cast<size_t>(it.first), mSize);
                    size_t copyBytes = packedByteCount(static_cast<size_t>(it.second - it.first), mSize);
                    state = cudaMemcpy((uint8_t*)deviceWeightData + dstOffsetBytes,
                                       (uint8_t*)weight.cudaData + srcOffsetBytes,
                                       copyBytes, GetCudaMemcpyType(mallocType, 1));
                    if (state != cudaSuccess) {
                        break;
                    }
                    if (hasBias) {
                        state = cudaMemcpy(deviceBiasData + curLen, cudaBiasData + it.first,
                                           static_cast<size_t>(it.second - it.first) * sizeof(float),
                                           GetCudaMemcpyType(mallocType, 1));
                        if (state != cudaSuccess) {
                            break;
                        }
                    }
                    curLen += (it.second - it.first);
                }
            }
        } else {
            weight.multiDeviceDatas[deviceId] = CreateMultiCudaLocalTensor(weight, {k, len});
            weight.multiDeviceDatas[deviceId]->dataDevice = dataDevice;
            weight.multiDeviceDatas[deviceId]->dataDeviceIds = {deviceId};
            bias.multiDeviceDatas[deviceId] = CreateMultiCudaLocalTensor(bias, hasBias ? std::vector<int>{k} : std::vector<int>{});
            bias.multiDeviceDatas[deviceId]->dataDevice = dataDevice;
            bias.multiDeviceDatas[deviceId]->dataDeviceIds = {deviceId};
            weight.multiDeviceDatas[deviceId]->Allocate();
            if (hasBias) {
                bias.multiDeviceDatas[deviceId]->Allocate();
            }

            deviceWeightData = mallocType == 0 ? weight.multiDeviceDatas[deviceId]->cpuData : weight.multiDeviceDatas[deviceId]->cudaData;
            deviceBiasData = hasBias ? (float*)(mallocType == 0 ? bias.multiDeviceDatas[deviceId]->cpuData : bias.multiDeviceDatas[deviceId]->cudaData) : nullptr;
            bool emptyShard = len == 0;
            if (!emptyShard && (deviceWeightData == nullptr || (hasBias && deviceBiasData == nullptr))) {
                state = cudaErrorMemoryAllocation;
            }

            int curLen = 0;
            if (state != cudaSuccess) {
                // handled by the common error path below
            } else if (emptyShard) {
                // This can happen when the number of GPUs is larger than the number of aligned TP blocks.
            } else if (IsGGUFTensor(weight)) {
                size_t srcRowBytes = GetGGUFRowBytes(weight, m);
                size_t dstRowBytes = GetGGUFRowBytes(*weight.multiDeviceDatas[deviceId], len);
                size_t dstOffsetBytes = 0;
                if (mallocType == 0) {
                    cudaSetDevice(rootDevice);
                }
                for (auto &it : div) {
                    int copyLen = it.second - it.first;
                    size_t srcOffsetBytes = GetGGUFRowBytes(weight, it.first);
                    size_t copyBytes = GetGGUFRowBytes(weight, copyLen);
                    state = FastllmCudaMemcpy2D((uint8_t*)deviceWeightData + dstOffsetBytes,
                                                dstRowBytes,
                                                (uint8_t*)weight.cudaData + srcOffsetBytes,
                                                srcRowBytes,
                                                copyBytes,
                                                kSize, GetCudaMemcpyType(mallocType, 1), deviceId, rootDevice);
                    if (state != cudaSuccess) {
                        break;
                    }
                    dstOffsetBytes += copyBytes;
                    curLen += copyLen;
                }
            } else if (weight.dataType == fastllm::DataType::FP8_E4M3_BLOCK_128) {
                size_t srcRowBytes = fastllm::GetDataBytes(weight.dataType, 1, m);
                size_t dstRowBytes = fastllm::GetDataBytes(weight.dataType, 1, len);
                size_t dstOffsetBytes = 0;
                for (auto &it : div) {
                    int copyLen = it.second - it.first;
                    fastllm::AssertInFastLLM(it.first % 128 == 0 && copyLen % 128 == 0,
                                             "FP8 block128 tensor parallel column split should align to 128.\n");
                    if (mallocType == 0) {
                        cudaSetDevice(rootDevice);
                    }
                    size_t blockBytes = 128 + sizeof(float);
                    size_t srcOffsetBytes = (size_t)(it.first / 128) * blockBytes;
                    size_t copyBytes = (size_t)(copyLen / 128) * blockBytes;
                    state = FastllmCudaMemcpy2D((uint8_t*)deviceWeightData + dstOffsetBytes,
                                                dstRowBytes,
                                                (uint8_t*)weight.cudaData + srcOffsetBytes,
                                                srcRowBytes,
                                                copyBytes,
                                                kSize, GetCudaMemcpyType(mallocType, 1), deviceId, rootDevice);
                    if (state != cudaSuccess) {
                        break;
                    }
                    dstOffsetBytes += copyBytes;
                    curLen += copyLen;
                }
            } else if (weight.dataType == fastllm::DataType::NVFP4 &&
                       weight.scales.empty() && weight.blockK > 0 && weight.blockM > 0) {
                size_t srcRowBytes = fastllm::GetNVFP4WeightBytes(1, m);
                size_t dstRowBytes = fastllm::GetNVFP4WeightBytes(1, len);
                size_t srcWeightBytes = fastllm::GetNVFP4WeightBytes(k, m);
                size_t dstWeightBytes = fastllm::GetNVFP4WeightBytes(k, len);
                size_t srcScaleCols = static_cast<size_t>((m - 1) / weight.blockM + 1);
                size_t dstScaleCols = static_cast<size_t>((len - 1) / weight.blockM + 1);
                size_t scaleRows = static_cast<size_t>((k - 1) / weight.blockK + 1);
                for (auto &it : div) {
                    int copyLen = it.second - it.first;
                    fastllm::AssertInFastLLM((it.first & 1) == 0 && (copyLen & 1) == 0 &&
                                             it.first % weight.blockM == 0 && copyLen % weight.blockM == 0,
                                             "NVFP4 tensor parallel column split should align to packed bytes and blockM.\n");
                    if (mallocType == 0) {
                        cudaSetDevice(rootDevice);
                    }
                    state = FastllmCudaMemcpy2D((uint8_t*)deviceWeightData + (static_cast<size_t>(curLen) >> 1),
                                                dstRowBytes,
                                                (uint8_t*)weight.cudaData + (static_cast<size_t>(it.first) >> 1),
                                                srcRowBytes,
                                                static_cast<size_t>(copyLen) >> 1,
                                                kSize, GetCudaMemcpyType(mallocType, 1), deviceId, rootDevice);
                    if (state != cudaSuccess) {
                        break;
                    }
                    state = FastllmCudaMemcpy2D((uint8_t*)deviceWeightData + dstWeightBytes +
                                                    static_cast<size_t>(curLen / weight.blockM),
                                                dstScaleCols,
                                                (uint8_t*)weight.cudaData + srcWeightBytes +
                                                    static_cast<size_t>(it.first / weight.blockM),
                                                srcScaleCols,
                                                static_cast<size_t>(copyLen / weight.blockM),
                                                scaleRows, GetCudaMemcpyType(mallocType, 1), deviceId, rootDevice);
                    if (state != cudaSuccess) {
                        break;
                    }
                    curLen += copyLen;
                }
            } else if (weight.dataType == fastllm::DataType::NVFP4_BLOCK_16 ||
                       weight.dataType == fastllm::DataType::NVFP4_BLOCK_16_E8M0) {
                const size_t blockBytes = weight.dataType == fastllm::DataType::NVFP4_BLOCK_16
                    ? 8 + sizeof(float) : 8 + sizeof(uint8_t);
                size_t srcRowBytes = fastllm::GetDataBytes(weight.dataType, 1, m);
                size_t dstRowBytes = fastllm::GetDataBytes(weight.dataType, 1, len);
                for (auto &it : div) {
                    int copyLen = it.second - it.first;
                    fastllm::AssertInFastLLM(it.first % 16 == 0 && copyLen % 16 == 0,
                                             "NVFP4_BLOCK_16 tensor parallel column split should align to 16.\n");
                    if (mallocType == 0) {
                        cudaSetDevice(rootDevice);
                    }
                    size_t dstOffsetBytes = static_cast<size_t>(curLen / 16) * blockBytes;
                    size_t srcOffsetBytes = static_cast<size_t>(it.first / 16) * blockBytes;
                    size_t copyBytes = static_cast<size_t>(copyLen / 16) * blockBytes;
                    state = FastllmCudaMemcpy2D((uint8_t*)deviceWeightData + dstOffsetBytes,
                                                dstRowBytes,
                                                (uint8_t*)weight.cudaData + srcOffsetBytes,
                                                srcRowBytes,
                                                copyBytes,
                                                kSize, GetCudaMemcpyType(mallocType, 1), deviceId, rootDevice);
                    if (state != cudaSuccess) {
                        break;
                    }
                    curLen += copyLen;
                }
            } else {
                for (auto &it : div) {
                    if (mallocType == 0) {
                        cudaSetDevice(rootDevice);
                    }
                    size_t copyLen = static_cast<size_t>(it.second - it.first);
                    size_t dstOffsetBytes = packedByteCount(1, static_cast<size_t>(curLen));
                    size_t dstRowBytes = packedByteCount(1, copyLen);
                    size_t srcOffsetBytes = packedByteCount(1, static_cast<size_t>(it.first));
                    size_t srcRowBytes = packedByteCount(1, mSize);
                    state = FastllmCudaMemcpy2D((uint8_t*)deviceWeightData + dstOffsetBytes,
                                                dstRowBytes,
                                                (uint8_t*)weight.cudaData + srcOffsetBytes,
                                                srcRowBytes,
                                                dstRowBytes,
                                                kSize, GetCudaMemcpyType(mallocType, 1), deviceId, rootDevice);
                    if (state != cudaSuccess) {
                        break;
                    }
                    curLen += (it.second - it.first);
                }
            }
            if (state == cudaSuccess && !emptyShard && hasBias) {
                if (i == 0) {
                    state = cudaMemcpy(deviceBiasData, cudaBiasData, biasBytes, GetCudaMemcpyType(mallocType, 1));
                } else {
                    state = AutoMemset(deviceBiasData, 0, biasBytes, mallocType);
                }
            }
        }

        if (cudaSuccess != state) {
            if (cudaBiasData != nullptr) {
                FastllmCudaFree(cudaBiasData);
            }
            ResetMultiDeviceData(weight);
            ResetMultiDeviceData(bias);
            checkCudaErrors("Error: CUDA error when split weight!", state);
            return false;
        }
    }
    if (cudaBiasData != nullptr) {
        FastllmCudaFree(cudaBiasData);
        cudaBiasData = nullptr;
    }

    if (weight.dataType == fastllm::DataType::FP8_E4M3) {
        for (int i = 0; i < multiCudaCurrentDevices.size(); i++) {
            int deviceId = multiCudaCurrentDevices[i], mallocType = 0;
            std::string specialId = "";
            SwitchDeviceAndGetInfos(deviceId, specialId, mallocType);
            
            auto &div = divisionScheme[deviceId];
            int len = 0;
            for (auto &it : div) {
                len += it.second - it.first;
            }

            float *cudaScales;
            float *cudaMins;
            uint8_t *cudaZeropoints;
            auto curDevice = weight.multiDeviceDatas[deviceId];
            curDevice->blockK = weight.blockK;
            curDevice->blockM = weight.blockM;
            if (len == 0) {
                curDevice->scales.clear();
                continue;
            }
            size_t ks = static_cast<size_t>((curDevice->dims[0] - 1) / curDevice->blockK + 1);
            size_t ms = static_cast<size_t>((curDevice->dims[1] - 1) / curDevice->blockM + 1);
            curDevice->scales.resize(ks * ms);
            if (splitAxis == 0) {
                size_t oriKs = static_cast<size_t>((k - 1) / weight.blockK + 1);
                size_t oriMs = static_cast<size_t>((m - 1) / weight.blockM + 1);
                fastllm::AssertInFastLLM(weight.scales.size() == oriKs * oriMs && ms == oriMs,
                                         "FP8 tensor parallel row split scale shape mismatch.\n");
                std::vector<int> sourceScaleRows(ks, -1);
                size_t localRow = 0;
                for (auto &it : div) {
                    size_t sourceRow = static_cast<size_t>(it.first);
                    size_t sourceEnd = static_cast<size_t>(it.second);
                    while (sourceRow < sourceEnd) {
                        size_t localScaleRow = localRow / static_cast<size_t>(weight.blockK);
                        size_t sourceScaleRow = sourceRow / static_cast<size_t>(weight.blockK);
                        fastllm::AssertInFastLLM(localScaleRow < ks && sourceScaleRow < oriKs,
                                                 "FP8 tensor parallel row split scale index overflow.\n");
                        if (sourceScaleRows[localScaleRow] < 0) {
                            sourceScaleRows[localScaleRow] = static_cast<int>(sourceScaleRow);
                        } else {
                            fastllm::AssertInFastLLM(sourceScaleRows[localScaleRow] == (int)sourceScaleRow,
                                                     "FP8 tensor parallel row split crosses incompatible scale blocks.\n");
                        }
                        size_t localRemain = static_cast<size_t>(weight.blockK) -
                                             localRow % static_cast<size_t>(weight.blockK);
                        size_t sourceRemain = static_cast<size_t>(weight.blockK) -
                                              sourceRow % static_cast<size_t>(weight.blockK);
                        size_t step = std::min(sourceEnd - sourceRow, std::min(localRemain, sourceRemain));
                        localRow += step;
                        sourceRow += step;
                    }
                }
                fastllm::AssertInFastLLM(localRow == static_cast<size_t>(curDevice->dims[0]),
                                         "FP8 tensor parallel row split scale length mismatch.\n");
                for (size_t row = 0; row < ks; row++) {
                    fastllm::AssertInFastLLM(sourceScaleRows[row] >= 0,
                                             "FP8 tensor parallel row split has an unmapped scale block.\n");
                    memcpy(curDevice->scales.data() + row * ms,
                           weight.scales.data() + static_cast<size_t>(sourceScaleRows[row]) * oriMs,
                           ms * sizeof(float));
                }
            } else {
                size_t oriKs = static_cast<size_t>((k - 1) / weight.blockK + 1);
                size_t oriMs = static_cast<size_t>((m - 1) / weight.blockM + 1);
                fastllm::AssertInFastLLM(weight.scales.size() == oriKs * oriMs && ks == oriKs,
                                         "FP8 tensor parallel column split scale shape mismatch.\n");
                std::vector<int> sourceScaleColumns(ms, -1);
                size_t localColumn = 0;
                for (auto &it : div) {
                    size_t sourceColumn = static_cast<size_t>(it.first);
                    size_t sourceEnd = static_cast<size_t>(it.second);
                    while (sourceColumn < sourceEnd) {
                        size_t localScaleColumn = localColumn / static_cast<size_t>(weight.blockM);
                        size_t sourceScaleColumn = sourceColumn / static_cast<size_t>(weight.blockM);
                        fastllm::AssertInFastLLM(localScaleColumn < ms && sourceScaleColumn < oriMs,
                                                 "FP8 tensor parallel column split scale index overflow.\n");
                        if (sourceScaleColumns[localScaleColumn] < 0) {
                            sourceScaleColumns[localScaleColumn] = static_cast<int>(sourceScaleColumn);
                        } else {
                            fastllm::AssertInFastLLM(sourceScaleColumns[localScaleColumn] == (int)sourceScaleColumn,
                                                     "FP8 tensor parallel column split crosses incompatible scale blocks.\n");
                        }
                        size_t localRemain = static_cast<size_t>(weight.blockM) -
                                             localColumn % static_cast<size_t>(weight.blockM);
                        size_t sourceRemain = static_cast<size_t>(weight.blockM) -
                                              sourceColumn % static_cast<size_t>(weight.blockM);
                        size_t step = std::min(sourceEnd - sourceColumn, std::min(localRemain, sourceRemain));
                        localColumn += step;
                        sourceColumn += step;
                    }
                }
                fastllm::AssertInFastLLM(localColumn == static_cast<size_t>(curDevice->dims[1]),
                                         "FP8 tensor parallel column split scale length mismatch.\n");
                for (size_t column = 0; column < ms; column++) {
                    fastllm::AssertInFastLLM(sourceScaleColumns[column] >= 0,
                                             "FP8 tensor parallel column split has an unmapped scale block.\n");
                }
                for (size_t row = 0; row < ks; row++) {
                    for (size_t column = 0; column < ms; column++) {
                        curDevice->scales[row * ms + column] =
                            weight.scales[row * oriMs + static_cast<size_t>(sourceScaleColumns[column])];
                    }
                }
            }
        }
    } else {
        // 1. mins, scales
        if (weight.mins.size() > 0) {
            int weightGroup = weight.group < 0 ? 1 : weight.group;
            size_t weightGroupSize = static_cast<size_t>(weightGroup);
            size_t kGroupCount = kSize * weightGroupSize;
            std::vector <int> zeropoints = std::vector <int> (kGroupCount, 0);
            if (weight.perChannelsConfigs.size() > 0) {
                for (size_t i = 0; i < kGroupCount; i++) {
                    zeropoints[i] = weight.perChannelsConfigs[i].zeroPoint;
                }
            } else if (weight.zeros.size() > 0) {
                for (size_t i = 0; i < kGroupCount; i++) {
                    zeropoints[i] = weight.zeros[i];
                }
            } else {
                for (size_t i = 0; i < kGroupCount; i++) {
                    zeropoints[i] = 0;
                }
            }
            for (int i = 0; i < multiCudaCurrentDevices.size(); i++) {
                int deviceId = multiCudaCurrentDevices[i], mallocType = 0;
                std::string specialId = "";
                SwitchDeviceAndGetInfos(deviceId, specialId, mallocType);
                
                auto &div = divisionScheme[deviceId];
                int len = 0;
                for (auto &it : div) {
                    len += it.second - it.first;
                }

                float *cudaScales;
                float *cudaMins;
                uint8_t *cudaZeropoints;
                auto curDevice = weight.multiDeviceDatas[deviceId];
                if (splitAxis == 0) {
                    curDevice->group = weight.group;
                    curDevice->groupCnt = weight.groupCnt;
                    size_t lenGroupCount = static_cast<size_t>(len) * weightGroupSize;
                    curDevice->scales.resize(lenGroupCount);
                    curDevice->mins.resize(lenGroupCount);
                    if (weight.dataType == fastllm::DataType::INT4_GROUP) {
                        int curLen = 0;
                        for (auto &it : div) {
                            size_t dstOffset = static_cast<size_t>(curLen) * weightGroupSize;
                            size_t srcOffset = static_cast<size_t>(it.first) * weightGroupSize;
                            size_t copyCount = static_cast<size_t>(it.second - it.first) * weightGroupSize;
                            memcpy(curDevice->scales.data() + dstOffset, weight.scales.data() + srcOffset,
                                   copyCount * sizeof(float));
                            memcpy(curDevice->mins.data() + dstOffset, weight.mins.data() + srcOffset,
                                   copyCount * sizeof(float));
                            curLen += (it.second - it.first);
                        }
                    } else {
                        curDevice->zeros.resize(lenGroupCount);
                        int curLen = 0;
                        for (auto &it : div) {
                            size_t dstOffset = static_cast<size_t>(curLen) * weightGroupSize;
                            size_t srcOffset = static_cast<size_t>(it.first) * weightGroupSize;
                            size_t copyCount = static_cast<size_t>(it.second - it.first) * weightGroupSize;
                            memcpy(curDevice->scales.data() + dstOffset, weight.scales.data() + srcOffset,
                                   copyCount * sizeof(float));
                            memcpy(curDevice->mins.data() + dstOffset, weight.mins.data() + srcOffset,
                                   copyCount * sizeof(float));
                            memcpy(curDevice->zeros.data() + dstOffset, zeropoints.data() + srcOffset,
                                   copyCount * sizeof(int));
                            curLen += (it.second - it.first);
                        }
                    }
                } else {
                    curDevice->scales.resize(kGroupCount);
                    curDevice->mins.resize(kGroupCount);
                    curDevice->group = weight.group;
                    curDevice->groupCnt = weight.groupCnt;
                    if (weight.dataType == fastllm::DataType::INT4_GROUP) {
                        int base = div[0].first / weight.groupCnt;
                        std::vector <float> scales, mins;
                        size_t baseOffset = static_cast<size_t>(base);
                        for (size_t i = 0; i < weight.scales.size(); i++) {
                            size_t src = i + baseOffset;
                            scales.push_back(src < weight.scales.size() ? weight.scales[src] : 0.0f);
                        }
                        for (size_t i = 0; i < weight.mins.size(); i++) {
                            size_t src = i + baseOffset;
                            mins.push_back(src < weight.mins.size() ? weight.mins[src] : 0.0f);
                        }
                        memcpy(curDevice->scales.data(), scales.data(), kGroupCount * sizeof(float));
                        memcpy(curDevice->mins.data(), mins.data(), kGroupCount * sizeof(float));
                    } else {
                        curDevice->zeros.resize(kGroupCount);
                        memcpy(curDevice->scales.data(), weight.scales.data(), kGroupCount * sizeof(float));
                        memcpy(curDevice->mins.data(), weight.mins.data(), kGroupCount * sizeof(float));
                        memcpy(curDevice->zeros.data(), zeropoints.data(), kGroupCount * sizeof(int));
                    }
                }
            }
        }
    }

    if (cudaSuccess != state) {
        checkCudaErrors("Error: CUDA error when split weight!", state);
        return false;
    }

    cudaSetDevice(rootDevice);
    FastllmCudaFree(weight.cudaData);
    if (cudaBiasData != nullptr) {
        FastllmCudaFree(cudaBiasData);
    }
    ClearGGUFExtraCudaCaches(weight);
    weight.cudaData = nullptr;
    weight.weightSum.clear();
    return true;
}

bool PlaceMultiCudaWeightOnDevice(fastllm::Data &weight, std::vector <int> &multiCudaCurrentDevices, int targetDevice) {
    int deviceNum = multiCudaCurrentDevices.size();
    int rootDevice = deviceNum > 0 ? multiCudaCurrentDevices[0] : 0;
    fastllm::AssertInFastLLM(weight.dataType != fastllm::DataType::DATA_GGUF_FORMAT || weight.ggmlType >= 0,
                             "GGUF weight \"" + weight.name + "\" is missing ggmlType before multicuda placement.\n");
    if (weight.multiDeviceData) {
        return true;
    }
    if (weight.dataDevice != fastllm::DataDevice::CUDA || weight.cudaData == nullptr ||
        weight.dataDeviceIds.size() == 0 || weight.dataDeviceIds[0] != rootDevice) {
        weight.ToDevice(fastllm::DataDevice::CUDA, {rootDevice}, true);
    }

    int oriId = FastllmCudaGetDevice();
    int mallocType = 0;
    std::string specialId = "";
    SwitchDeviceAndGetInfos(targetDevice, specialId, mallocType);
    fastllm::DataDevice dataDevice = (mallocType == 0 ? fastllm::DataDevice::CPU : fastllm::DataDevice::CUDA);

    auto *local = CreateMultiCudaLocalTensor(weight, weight.dims);
    local->dataDevice = dataDevice;
    local->dataDeviceIds = {targetDevice};
    local->group = weight.group;
    local->groupCnt = weight.groupCnt;
    local->scales = weight.scales;
    local->mins = weight.mins;
    local->zeros = weight.zeros;
    local->halfScales = weight.halfScales;
    local->perChannelsConfigs = weight.perChannelsConfigs;
    local->Allocate();

    size_t bytes = weight.GetBytes();
    cudaError_t state = cudaSuccess;
    if (mallocType == 0) {
        cudaSetDevice(rootDevice);
        state = cudaMemcpy(local->cpuData, weight.cudaData, bytes, cudaMemcpyDeviceToHost);
    } else if (targetDevice == rootDevice) {
        cudaSetDevice(rootDevice);
        state = cudaMemcpy(local->cudaData, weight.cudaData, bytes, cudaMemcpyDeviceToDevice);
    } else {
        FastllmCudaMemcpyBetweenDevices(targetDevice, local->cudaData, rootDevice, weight.cudaData, bytes);
    }
    if (state != cudaSuccess) {
        checkCudaErrors("Error: CUDA error when placing weight on multicuda device!", state);
        delete local;
        FastllmCudaSetDevice(oriId);
        return false;
    }

    weight.multiDeviceData = true;
    weight.multiDeviceDatas[targetDevice] = local;
    weight.tpLayout = fastllm::TP_LAYOUT_NONE;
    weight.tpAxis = -1;
    weight.tpGlobalDims.clear();
    weight.tpRanges.clear();

    cudaSetDevice(rootDevice);
    FastllmCudaFree(weight.cudaData);
    ClearGGUFExtraCudaCaches(weight);
    weight.cudaData = nullptr;
    weight.weightSum.clear();
    FastllmCudaSetDevice(oriId);
    return true;
}

bool SplitMultiCudaWeight1D(fastllm::Data &bias, std::vector <int> &multiCudaCurrentDevices, DivisionScheme divisionScheme) {
    int deviceNum = multiCudaCurrentDevices.size();
    int rootDevice = deviceNum > 0 ? multiCudaCurrentDevices[0] : 0;
    if (bias.dims.size() > 0 &&
        (bias.dataDevice != fastllm::DataDevice::CUDA || bias.cudaData == nullptr ||
         bias.dataDeviceIds.size() == 0 || bias.dataDeviceIds[0] != rootDevice)) {
        bias.ToDevice(fastllm::DataDevice::CUDA, {rootDevice}, true);
    }
    for (int i = 0; i < deviceNum; i++) {
        if (specialDeviceIds.find(multiCudaCurrentDevices[i]) == specialDeviceIds.end()) {
            cudaSetDevice(multiCudaCurrentDevices[i]);
        } 
        for (int j = 0; j < deviceNum; j++) {
            if (i != j) {
                if (specialDeviceIds.find(multiCudaCurrentDevices[j]) == specialDeviceIds.end()) {
                    int canPeerAccess = 0;
                    cudaError_t peerState = cudaDeviceCanAccessPeer(&canPeerAccess,
                                                                    multiCudaCurrentDevices[i],
                                                                    multiCudaCurrentDevices[j]);
                    if (peerState == cudaSuccess && canPeerAccess) {
                        peerState = cudaDeviceEnablePeerAccess(multiCudaCurrentDevices[j], 0);
                        if (peerState == cudaErrorPeerAccessAlreadyEnabled) {
                            cudaGetLastError();
                        }
                    }
                }
            }
        }
    }
    cudaSetDevice(0);

    if (bias.multiDeviceData) {
        return true;
    }
    bias.multiDeviceData = true;
    int k = bias.dims[0];
    cudaError_t state = cudaSuccess;
    float *cudaBiasData = (float*)FastllmCudaMalloc(k * sizeof(float));
    if (bias.dims.size() > 0) {
        state = cudaMemcpy(cudaBiasData, (uint8_t *) bias.cudaData, k * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        state = cudaMemset(cudaBiasData, 0, k * sizeof(float));
    }
        
    for (int i = 0; i < multiCudaCurrentDevices.size(); i++) {
        int deviceId = multiCudaCurrentDevices[i], mallocType = 0;
        std::string specialId = "";
        SwitchDeviceAndGetInfos(deviceId, specialId, mallocType);
        fastllm::DataDevice dataDevice = (mallocType == 0 ? fastllm::DataDevice::CPU :fastllm::DataDevice::CUDA);

        auto &div = divisionScheme[deviceId];
        int len = 0;
        for (auto &it : div) {
            len += it.second - it.first;
        }

        void *deviceWeightData;
        float *deviceBiasData;
        cudaError_t state = cudaSuccess;
        {
            bias.multiDeviceDatas[deviceId] = new fastllm::Data(bias.dataType, {len});
            bias.multiDeviceDatas[deviceId]->dataDevice = dataDevice;
            bias.multiDeviceDatas[deviceId]->dataDeviceIds = {deviceId};
            bias.multiDeviceDatas[deviceId]->name = bias.name;
            bias.multiDeviceDatas[deviceId]->isModelWeight = bias.isModelWeight;
            bias.multiDeviceDatas[deviceId]->Allocate();
            deviceBiasData = (float*)(mallocType == 0 ? bias.multiDeviceDatas[deviceId]->cpuData : bias.multiDeviceDatas[deviceId]->cudaData);
            int curLen = 0;
            for (auto &it : div) {
                state = cudaMemcpy(deviceBiasData + curLen, cudaBiasData + it.first, (it.second - it.first) * sizeof(float), GetCudaMemcpyType(mallocType, 1));
                curLen += (it.second - it.first);
            }
        }

        if (cudaSuccess != state) {
            checkCudaErrors("Error: CUDA error when split weight!", state);
            return false;
        }
    }
    if (cudaSuccess != state) {
        checkCudaErrors("Error: CUDA error when split weight!", state);
        return false;
    }

    cudaSetDevice(0);
    FastllmCudaFree(cudaBiasData);
    return true;
}

std::vector <bool> streamInits = std::vector <bool> (4, 0);
cudaStream_t streams[4];

cudaStream_t *GetFastllmStream(int id) {
    if (!streamInits[id]) {
        streamInits[id] = true;
        cudaSetDevice(id);
        cudaStreamCreate(&streams[id]);
        cudaSetDevice(0);
    }
    return &streams[id];
}

// 全局变量存储通信器
// Key: deviceId, Value: ncclComm_t
static std::map<int, ncclComm_t> g_ncclComms;
static std::map<int, int> g_ncclRanks;
static bool g_ncclInitialized = false;
static int g_ncclWorldSize = 0;

static std::vector<int> FastllmUniqueNcclDevices(const std::vector<int> &devices) {
    std::vector<int> uniqueDevices;
    uniqueDevices.reserve(devices.size());
    for (int device : devices) {
        if (device >= 0 && std::find(uniqueDevices.begin(), uniqueDevices.end(), device) == uniqueDevices.end()) {
            uniqueDevices.push_back(device);
        }
    }
    return uniqueDevices;
}

static size_t FastllmNcclDataTypeBytes(int dataType) {
    if (dataType == fastllm::DataType::FLOAT32) {
        return sizeof(float);
    } else if (dataType == fastllm::DataType::FLOAT16 || dataType == fastllm::DataType::BFLOAT16) {
        return sizeof(uint16_t);
    }
    return 0;
}

static ncclComm_t FindNcclCommNoLog(int deviceId) {
    auto it = g_ncclComms.find(deviceId);
    return it == g_ncclComms.end() ? nullptr : it->second;
}

bool FastllmInitNccl(const std::vector<int>& devices) {
    std::vector<int> uniqueDevices = FastllmUniqueNcclDevices(devices);
    if (uniqueDevices.size() <= 1) {
        return false;
    }

    int numGPUs = uniqueDevices.size();
    bool ready = g_ncclInitialized && g_ncclWorldSize == numGPUs &&
                 (int)g_ncclComms.size() == numGPUs;
    if (ready) {
        for (int device : uniqueDevices) {
            auto it = g_ncclComms.find(device);
            if (it == g_ncclComms.end() || it->second == nullptr) {
                ready = false;
                break;
            }
        }
    }
    if (ready) {
        return true;
    }

    for (auto &it : g_ncclComms) {
        if (it.second != nullptr) {
            ncclCommDestroy(it.second);
        }
    }
    g_ncclComms.clear();
    g_ncclRanks.clear();
    g_ncclInitialized = false;
    g_ncclWorldSize = 0;

    std::vector<ncclComm_t> comms(numGPUs);

    for (int device : uniqueDevices) {
        cudaSetDevice(device);
        cudaFree(0);
    }

    // ncclCommInitAll 会在这一组设备之间建立通信域
    // 注意：这会阻塞，直到所有卡都就绪
    ncclResult_t initRes = ncclCommInitAll(comms.data(), numGPUs, uniqueDevices.data());
    if (initRes != ncclSuccess) {
        printf("Error: ncclCommInitAll failed: %s\n", ncclGetErrorString(initRes));
        return false;
    }

    // 将生成的 comms 存入 map，方便后续通过 deviceId 查找
    for(int i = 0; i < numGPUs; ++i) {
        g_ncclComms[uniqueDevices[i]] = comms[i];
        g_ncclRanks[uniqueDevices[i]] = i;
    }
        
    g_ncclInitialized = true;
    g_ncclWorldSize = numGPUs;
    // 通知 CUDA 分配器：NCCL 已激活。此后真实 cudaMalloc 前会先排空在途集合通信，
    // 避免 cudaMalloc 与 NCCL 主机 proxy 争用 CUDA 驱动锁导致的跨 rank 死锁。
    FastllmCudaSetNcclActive(true);
    printf("NCCL Initialized for %d devices.\n", numGPUs);
    return true;
}

ncclComm_t GetNcclComm(int deviceId) {
    if (g_ncclComms.find(deviceId) != g_ncclComms.end()) {
        return g_ncclComms[deviceId];
    }
    printf("Error: No NCCL comm found for device %d\n", deviceId);
    return nullptr;
}

// 集合通信发射后是否立即同步：warmup/权重加载阶段返回 true 以防跨 rank 死锁，
// warmup 结束后返回 false，稳态前向异步发射以恢复通信/计算重叠(见 FastllmCudaSetNcclForceSync)。
// 流捕获期间必须返回 false：cudaStreamSynchronize 属于捕获期间被禁止的调用，会直接
// invalidate 正在进行的 CUDA graph 捕获；捕获期间集合通信只是被录制、并未真正执行，
// 不存在与真实 cudaMalloc 争用驱动锁的死锁风险，跳过同步是安全的。
static bool FastllmNcclPostSyncEnabled(cudaStream_t stream) {
    if (!FastllmCudaGetNcclForceSync()) {
        return false;
    }
    cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
    cudaError_t captureState = cudaStreamIsCapturing(stream, &captureStatus);
    if (captureState != cudaSuccess) {
        FastllmCudaSetThreadError();
        cudaGetLastError();
        return false;
    }
    if (captureStatus != cudaStreamCaptureStatusNone) {
        return false;
    }
    return true;
}

namespace {

// Speculative verification uses 2--4 rows of hidden states. At TP=2 these
// tensors are only tens of KiB, where NCCL launch/protocol latency is larger
// than the actual peer transfer. Keep this path deliberately narrow: single
// process, two peer-accessible CUDA devices, residual-fused reductions, and
// tensors large enough to be the multi-token verification path. Everything
// else uses the existing NCCL implementation.
constexpr int FASTLLM_TP2_P2P_MIN_ELEMENTS = 4096;
constexpr int FASTLLM_TP2_P2P_MAX_ELEMENTS = 65536;
constexpr int FASTLLM_TP2_P2P_MAX_BLOCKS = 36;

struct alignas(128) FastllmTP2P2PSignal {
    // Two synchronization points and two alternating slots prevent a faster
    // rank from overwriting a flag before its peer has observed it.
    uint32_t flags[2][2][FASTLLM_TP2_P2P_MAX_BLOCKS][2];
    uint64_t inputPointers[2][FASTLLM_TP2_P2P_MAX_BLOCKS][2];
};

struct FastllmTP2P2PState {
    std::mutex initMutex;
    bool initialized = false;
    bool available = false;
    int devices[2] = {-1, -1};
    FastllmTP2P2PSignal *signals[2] = {nullptr, nullptr};
    std::atomic<uint32_t> sequence[2];

    FastllmTP2P2PState() {
        sequence[0].store(0, std::memory_order_relaxed);
        sequence[1].store(0, std::memory_order_relaxed);
    }
};

FastllmTP2P2PState &FastllmGetTP2P2PState() {
    static FastllmTP2P2PState state;
    return state;
}

bool FastllmTP2P2PEnabled() {
    static bool enabled = []() {
        const char *env = std::getenv("FASTLLM_CUDA_TP2_P2P_ALLREDUCE");
        if (env == nullptr || env[0] == '\0') {
            return true;
        }
        std::string value(env);
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return (char)std::tolower(c); });
        return value != "0" && value != "false" && value != "off" &&
               value != "no";
    }();
    return enabled;
}

__device__ __forceinline__ void FastllmTP2StoreRelease(uint32_t *address,
                                                       uint32_t value) {
    // Like vLLM's one-stage custom all-reduce, this barrier only coordinates
    // kernel progress; it does not publish preceding writes.
    asm volatile("st.volatile.global.u32 [%1], %0;" ::
                 "r"(value), "l"(address));
}

__device__ __forceinline__ uint32_t FastllmTP2LoadAcquire(
        const uint32_t *address) {
    uint32_t value;
    asm volatile("ld.volatile.global.u32 %0, [%1];" :
                 "=r"(value) : "l"(address));
    return value;
}

__device__ __forceinline__ void FastllmTP2StoreReleaseSystem(
        uint32_t *address, uint32_t value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("st.release.sys.global.u32 [%1], %0;" ::
                 "r"(value), "l"(address));
#else
    asm volatile("membar.sys; st.volatile.global.u32 [%1], %0;" ::
                 "r"(value), "l"(address));
#endif
}

__device__ __forceinline__ uint32_t FastllmTP2LoadAcquireSystem(
        const uint32_t *address) {
    uint32_t value;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" :
                 "=r"(value) : "l"(address));
#else
    asm volatile("ld.volatile.global.u32 %0, [%1]; membar.gl;" :
                 "=r"(value) : "l"(address));
#endif
    return value;
}

template <typename T>
__device__ __forceinline__ T FastllmTP2AddValue(T a, T b);

template <>
__device__ __forceinline__ half FastllmTP2AddValue(half a, half b) {
    return __hadd(a, b);
}

template <>
__device__ __forceinline__ __nv_bfloat16 FastllmTP2AddValue(
        __nv_bfloat16 a, __nv_bfloat16 b) {
    return __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
}

template <>
__device__ __forceinline__ float FastllmTP2AddValue(float a, float b) {
    return a + b;
}

template <typename T>
struct alignas(16) FastllmTP2Packed {
    T values[16 / sizeof(T)];
};

template <typename T>
__global__ __launch_bounds__(512, 1) void FastllmTP2P2PReduceAddDirectKernel(
        T *dst, const T *residual, const T *localInput, int count,
        FastllmTP2P2PSignal *selfSignal,
        FastllmTP2P2PSignal *peerSignal,
        int rank, uint32_t sequence) {
    const int peerRank = 1 - rank;
    const int slot = sequence & 1;
    if (threadIdx.x == 0) {
        peerSignal->inputPointers[slot][blockIdx.x][rank] =
            (uint64_t)(uintptr_t)localInput;
        // The pointer is published by this kernel. A system release/acquire
        // pair makes that remote pointer visible before either rank
        // dereferences it.
        FastllmTP2StoreReleaseSystem(
            &peerSignal->flags[0][slot][blockIdx.x][rank], sequence);
        while (FastllmTP2LoadAcquireSystem(
                   &selfSignal->flags[0][slot][blockIdx.x][peerRank]) !=
               sequence) {
        }
    }
    __syncthreads();

    using Packed = FastllmTP2Packed<T>;
    constexpr int valuesPerPack = 16 / sizeof(T);
    const int packedCount = count / valuesPerPack;
    Packed *packedDst = (Packed*)dst;
    const Packed *packedResidual = (const Packed*)residual;
    const Packed *packedLocal = (const Packed*)localInput;
    const Packed *packedPeer = (const Packed*)(uintptr_t)
        selfSignal->inputPointers[slot][blockIdx.x][peerRank];
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < packedCount; index += gridDim.x * blockDim.x) {
        Packed localValue = packedLocal[index];
        Packed peerValue = packedPeer[index];
        Packed rank0 = rank == 0 ? localValue : peerValue;
        Packed rank1 = rank == 0 ? peerValue : localValue;
        Packed residualValue = packedResidual[index];
#pragma unroll
        for (int i = 0; i < valuesPerPack; i++) {
            rank0.values[i] = FastllmTP2AddValue(rank0.values[i],
                                                 rank1.values[i]);
            rank0.values[i] = FastllmTP2AddValue(rank0.values[i],
                                                 residualValue.values[i]);
        }
        packedDst[index] = rank0;
    }

    // Keep localInput alive until the peer has completed all remote reads.
    // No memory publication is needed here, so the lightweight final barrier
    // is sufficient and matches vLLM's one-stage all-reduce protocol.
    __syncthreads();
    if (threadIdx.x == 0) {
        FastllmTP2StoreRelease(
            &peerSignal->flags[1][slot][blockIdx.x][rank], sequence);
        while (FastllmTP2LoadAcquire(
                   &selfSignal->flags[1][slot][blockIdx.x][peerRank]) !=
               sequence) {
        }
    }
}

bool FastllmInitTP2P2PState(FastllmTP2P2PState &state) {
    std::lock_guard<std::mutex> guard(state.initMutex);
    if (state.initialized) {
        return state.available;
    }
    if (!FastllmTP2P2PEnabled()) {
        state.initialized = true;
        return false;
    }
    // NCCL establishes the rank-to-device mapping. If it is not ready yet,
    // leave initialization retryable rather than permanently disabling the
    // fast path during model warmup.
    if (g_ncclWorldSize != 2 || g_ncclRanks.size() != 2) {
        return false;
    }

    for (auto &it : g_ncclRanks) {
        if (it.second < 0 || it.second >= 2) {
            state.initialized = true;
            return false;
        }
        state.devices[it.second] = it.first;
    }
    if (state.devices[0] < 0 || state.devices[1] < 0) {
        state.initialized = true;
        return false;
    }

    int canAccess01 = 0, canAccess10 = 0;
    if (cudaDeviceCanAccessPeer(&canAccess01, state.devices[0], state.devices[1]) != cudaSuccess ||
        cudaDeviceCanAccessPeer(&canAccess10, state.devices[1], state.devices[0]) != cudaSuccess ||
        !canAccess01 || !canAccess10) {
        cudaGetLastError();
        state.initialized = true;
        return false;
    }

    int oldDevice = 0;
    cudaGetDevice(&oldDevice);
    bool ok = true;
    for (int rank = 0; rank < 2; rank++) {
        cudaSetDevice(state.devices[rank]);
        cudaError_t peerState = cudaDeviceEnablePeerAccess(state.devices[1 - rank], 0);
        if (peerState == cudaErrorPeerAccessAlreadyEnabled) {
            cudaGetLastError();
        } else if (peerState != cudaSuccess) {
            cudaGetLastError();
            ok = false;
            break;
        }
        if (cudaMalloc((void**)&state.signals[rank],
                       sizeof(FastllmTP2P2PSignal)) != cudaSuccess ||
            cudaMemset(state.signals[rank], 0,
                       sizeof(FastllmTP2P2PSignal)) != cudaSuccess) {
            cudaGetLastError();
            ok = false;
            break;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            cudaGetLastError();
            ok = false;
            break;
        }
    }

    if (!ok) {
        for (int rank = 0; rank < 2; rank++) {
            if (state.devices[rank] < 0) {
                continue;
            }
            cudaSetDevice(state.devices[rank]);
            if (state.signals[rank] != nullptr) {
                cudaFree(state.signals[rank]);
                state.signals[rank] = nullptr;
            }
        }
    }
    cudaSetDevice(oldDevice);
    state.available = ok;
    state.initialized = true;
    if (state.available) {
        printf("[Fastllm] TP2 P2P small-tensor allreduce enabled for cuda:%d,%d.\n",
               state.devices[0], state.devices[1]);
        fflush(stdout);
    }
    return state.available;
}

template <typename T>
void FastllmLaunchTP2P2PReduceAddDirect(
        T *dst, const T *residual, const T *localInput, int count,
        FastllmTP2P2PSignal *selfSignal,
        FastllmTP2P2PSignal *peerSignal,
        int rank, uint32_t sequence, cudaStream_t stream) {
    constexpr int valuesPerPack = 16 / sizeof(T);
    int packedCount = count / valuesPerPack;
    int blocks = std::max(1, std::min(FASTLLM_TP2_P2P_MAX_BLOCKS,
                                     (packedCount + 511) / 512));
    FastllmTP2P2PReduceAddDirectKernel<T><<<blocks, 512, 0, stream>>>(
        dst, residual, localInput, count,
        selfSignal, peerSignal, rank, sequence);
}

} // namespace

static bool FastllmCanUseTP2P2PAllReduceAddImpl(
        int count, int dataType, int deviceId, int *rankOut) {
    if (count < FASTLLM_TP2_P2P_MIN_ELEMENTS ||
        count > FASTLLM_TP2_P2P_MAX_ELEMENTS ||
        ((size_t)count * (dataType == fastllm::DataType::FLOAT32
             ? sizeof(float) : sizeof(uint16_t))) % 16 != 0 ||
        (dataType != fastllm::DataType::FLOAT16 &&
         dataType != fastllm::DataType::BFLOAT16 &&
         dataType != fastllm::DataType::FLOAT32) ||
        FastllmCudaGetNcclForceSync() || fastllm::GetFastllmEnv().cudaGraph) {
        return false;
    }

    FastllmTP2P2PState &state = FastllmGetTP2P2PState();
    if (!FastllmInitTP2P2PState(state)) {
        return false;
    }
    auto rankIt = g_ncclRanks.find(deviceId);
    if (rankIt == g_ncclRanks.end() || rankIt->second < 0 || rankIt->second >= 2) {
        static std::atomic<bool> loggedMissingRank(false);
        if (!loggedMissingRank.exchange(true)) {
            printf("[Fastllm] TP2 P2P rank lookup failed for cuda:%d.\n", deviceId);
        }
        return false;
    }
    if (rankOut != nullptr) {
        *rankOut = rankIt->second;
    }
    return true;
}

bool FastllmCanUseTP2P2PAllReduceAdd(int count, int dataType, int deviceId) {
    return FastllmCanUseTP2P2PAllReduceAddImpl(
        count, dataType, deviceId, nullptr);
}

bool FastllmTryTP2P2PAllReduceAdd(void* data, void* dest, int count,
                                  int dataType, int deviceId) {
    if (data == nullptr || dest == nullptr) {
        return false;
    }

    int rank = -1;
    if (!FastllmCanUseTP2P2PAllReduceAddImpl(
            count, dataType, deviceId, &rank)) {
        return false;
    }
    FastllmTP2P2PState &state = FastllmGetTP2P2PState();
    uint32_t sequence = state.sequence[rank].fetch_add(
        1, std::memory_order_relaxed) + 1;
    cudaSetDevice(deviceId);
    cudaStream_t stream = cudaStreamPerThread;
    if (dataType == fastllm::DataType::FLOAT16) {
        FastllmLaunchTP2P2PReduceAddDirect(
            (half*)dest, (const half*)dest, (const half*)data, count,
            state.signals[rank], state.signals[1 - rank],
            rank, sequence, stream);
    } else if (dataType == fastllm::DataType::BFLOAT16) {
        FastllmLaunchTP2P2PReduceAddDirect(
            (__nv_bfloat16*)dest, (const __nv_bfloat16*)dest,
            (const __nv_bfloat16*)data, count,
            state.signals[rank], state.signals[1 - rank],
            rank, sequence, stream);
    } else {
        FastllmLaunchTP2P2PReduceAddDirect(
            (float*)dest, (const float*)dest, (const float*)data, count,
            state.signals[rank], state.signals[1 - rank],
            rank, sequence, stream);
    }
    cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
        printf("[Fastllm] TP2 P2P allreduce-add launch failed on cuda:%d (%s).\n",
               deviceId, cudaGetErrorString(result));
        FastllmCudaSetThreadError();
    }
    return true;
}

// 功能：将 root 设备上的 data 数据广播到所有卡 (In-place 操作)
// data: 也就是发送/接收缓冲区的首地址
// count: 元素个数
// dataType: fastllm 数据类型
// root: 源数据的 deviceId
// deviceId: 当前调用线程所属的 deviceId
void FastllmNcclBroadcast(void* data, int count, int dataType, int root, int deviceId) {
    if (data == nullptr || count <= 0) {
        return;
    }

    // 1. 获取当前设备的通信器
    ncclComm_t comm = FindNcclCommNoLog(deviceId);
    if (comm == nullptr) {
        std::vector<int> devices;
        std::map<int, int> ratios;
        FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
        if (FastllmUniqueNcclDevices(devices).size() <= 1 || !FastllmInitNccl(devices)) {
            return;
        }
        comm = FindNcclCommNoLog(deviceId);
    }
    if (comm == nullptr) {
        printf("Error: FastllmNcclBroadcast failed, comm is null for device %d\n", deviceId);
        FastllmCudaSetThreadError();
        return;
    }

    // 2. 映射数据类型
    ncclDataType_t ncclType = ncclFloat; 
    if (dataType == fastllm::DataType::FLOAT16) {
        ncclType = ncclHalf;
    } else if (dataType == fastllm::DataType::BFLOAT16) {
        ncclType = ncclBfloat16;
    } else if (dataType == fastllm::DataType::FLOAT32) {
        ncclType = ncclFloat;
    } else {
        printf("Error: Unknown dataType %d for NCCL Broadcast\n", dataType);
        FastllmCudaSetThreadError();
        return;
    }

    // 3. 执行 Broadcast
    // ncclBroadcast(sendbuff, recvbuff, ...);
    // 对于 In-place 操作，sendbuff 和 recvbuff 传同一个地址即可
    cudaStream_t stream = cudaStreamPerThread;
    int rootRank = -1;
    auto rootRankIt = g_ncclRanks.find(root);
    if (rootRankIt != g_ncclRanks.end()) {
        rootRank = rootRankIt->second;
    } else if (root >= 0 && root < g_ncclWorldSize) {
        rootRank = root;
    }
    if (rootRank < 0 || rootRank >= g_ncclWorldSize) {
        printf("Error: invalid root %d for NCCL Broadcast on device %d\n", root, deviceId);
        FastllmCudaSetThreadError();
        return;
    }
    
    ncclResult_t res = ncclBroadcast(data, data, count, ncclType, rootRank, comm, stream);
    
    if (res != ncclSuccess) {
        printf("Error: ncclBroadcast failed on device %d: %s\n", deviceId, ncclGetErrorString(res));
        FastllmCudaSetThreadError();
        return;
    }
    // 发射后立即同步：保证返回时集合通信已完成，不残留在途通信。无 P2P 的多卡(经 PCIe/SHM)下，
    // 若集合通信在途时另一线程触发真实 cudaMalloc(持 CUDA 驱动锁并隐式同步)，会与 NCCL 主机 proxy
    // 争用驱动锁形成跨 rank 死锁。同步发射可彻底消除该竞态。
    if (FastllmNcclPostSyncEnabled(stream)) {
        cudaError_t syncState = cudaStreamSynchronize(stream);
        checkCudaErrors("Error: CUDA error when synchronizing NCCL broadcast!", syncState);
    }
}

// 功能：将所有卡上的 data 数据进行 Sum 求和，结果保存在 dest 中 (支持 in-place，即 data == dest)
void FastllmNcclAllReduce(void* data, void* dest, int count, int dataType, int deviceId) {
    if (data == nullptr || dest == nullptr || count <= 0) {
        return;
    }

    // 1. 获取当前设备的通信器
    ncclComm_t comm = FindNcclCommNoLog(deviceId);
    if (comm == nullptr) {
        std::vector<int> devices;
        std::map<int, int> ratios;
        FastllmGetMulticudaDeviceAndRatio(devices, ratios, true);
        if (FastllmUniqueNcclDevices(devices).size() <= 1 || !FastllmInitNccl(devices)) {
            if (data != dest) {
                size_t bytes = (size_t)count * FastllmNcclDataTypeBytes(dataType);
                if (bytes > 0) {
                    FastllmCudaCopyFromDeviceToDevice(dest, data, bytes);
                }
            }
            return;
        }
        comm = FindNcclCommNoLog(deviceId);
    }
    if (comm == nullptr) {
        printf("Error: FastllmNcclAllReduce failed, comm is null for device %d\n", deviceId);
        FastllmCudaSetThreadError();
        return;
    }

    // 2. 映射数据类型 (Fastllm DataType -> NCCL type)
    ncclDataType_t ncclType = ncclFloat; 
    if (dataType == fastllm::DataType::FLOAT16) {
        ncclType = ncclHalf;
    } else if (dataType == fastllm::DataType::BFLOAT16) {
        ncclType = ncclBfloat16;
    } else if (dataType == fastllm::DataType::FLOAT32) {
        ncclType = ncclFloat;
    } else {
        // 如果有其他类型，需在此补充
        printf("Error: Unknown dataType %d for NCCL\n", dataType);
        FastllmCudaSetThreadError();
        return;
    }

    // 3. 执行 AllReduce
    // op: ncclSum (求和)
    cudaStream_t stream = cudaStreamPerThread;
    
    // 注意：ncclAllReduce 发射是异步的
    ncclResult_t res = ncclAllReduce(data, dest, count, ncclType, ncclSum, comm, stream);
    
    if (res != ncclSuccess) {
        printf("Error: ncclAllReduce failed on device %d: %s\n", deviceId, ncclGetErrorString(res));
        FastllmCudaSetThreadError();
        return;
    }
    // 发射后立即同步：保证返回时集合通信已完成，不残留在途通信。无 P2P 的多卡(经 PCIe/SHM)下，
    // 若集合通信在途时另一线程触发真实 cudaMalloc(持 CUDA 驱动锁并隐式同步)，会与 NCCL 主机 proxy
    // 争用驱动锁形成跨 rank 死锁。同步发射可彻底消除该竞态。
    if (FastllmNcclPostSyncEnabled(stream)) {
        cudaError_t syncState = cudaStreamSynchronize(stream);
        checkCudaErrors("Error: CUDA error when synchronizing NCCL allreduce!", syncState);
    }
}
