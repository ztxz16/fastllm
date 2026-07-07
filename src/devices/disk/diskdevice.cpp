#include "devices/disk/diskdevice.h"
#include "blocks/baseblock.h"
#include "gguf.h"
#include "utils.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <mutex>
#include <set>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#endif

namespace fastllm {
#ifdef USE_CUDA
    extern void DoCudaMergeMOEFromCPU(Data &input, Data &output, Data &index, Data &score,
                                      Data &w1, Data &w2, Data &w3,
                                      Data **weights, Data **biass, float sharedScale,
                                      bool setZero, const std::unordered_set<int> &experts,
                                      bool isCrossSwiglu, MoeGateType gateType);
#endif

    DiskDevice::DiskDevice() {
        this->deviceType = "disk";
        this->ops["MergeMOE"] = (BaseOperator*)(new DiskMergeMOE());
    }

    bool DiskDevice::Malloc(void **ret, size_t size) {
        *ret = (void*)new uint8_t[size];
        return true;
    }

    bool DiskDevice::Free(void *ret) {
        delete[] (uint8_t*)ret;
        return true;
    }

    bool DiskDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
        if (dst != src && dst != nullptr && src != nullptr) {
            memcpy(dst, src, size);
        }
        return true;
    }

    bool DiskDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
        if (dst != src && dst != nullptr && src != nullptr) {
            memcpy(dst, src, size);
        }
        return true;
    }

    static size_t DiskPartCount(const DiskWeightPart &part) {
        size_t count = 1;
        for (int dim : part.dims) {
            count *= dim;
        }
        return count;
    }

    static int DiskMoeLoadThreads() {
        static int threads = []() {
            const char *env = std::getenv("FASTLLM_DISK_MOE_LOAD_THREADS");
            int v = env == nullptr ? 4 : atoi(env);
            return std::max(1, v);
        }();
        return threads;
    }

    static bool ParseEnvFlag(const char *env, bool defaultValue) {
        if (env == nullptr) {
            return defaultValue;
        }
        std::string value(env);
        std::transform(value.begin(), value.end(), value.begin(), ::tolower);
        if (value == "0" || value == "false" || value == "off") {
            return false;
        }
        if (value == "1" || value == "true" || value == "on") {
            return true;
        }
        return defaultValue;
    }

    static bool DiskMoeGpuPrefillEnabled() {
        static bool enabled = []() {
            bool ret = ParseEnvFlag(std::getenv("FT_GPU_PREFILL"), true);
            return ParseEnvFlag(std::getenv("FASTLLM_DISK_MOE_GPU_PREFILL"), ret);
        }();
        return enabled;
    }

    static int DiskMoeGpuPrefillMinTokens() {
        static int minTokens = []() {
            const char *env = std::getenv("FASTLLM_DISK_MOE_GPU_PREFILL_MIN_TOKENS");
            int v = env == nullptr ? 32 : atoi(env);
            return std::max(1, v);
        }();
        return minTokens;
    }

    class DiskFileCache {
    public:
        ~DiskFileCache() {
            for (auto &it : fds) {
                close(it.second);
            }
        }

        int Get(const std::string &fileName) {
            std::lock_guard<std::mutex> guard(locker);
            auto it = fds.find(fileName);
            if (it != fds.end()) {
                return it->second;
            }
            int fd = open(fileName.c_str(), O_RDONLY);
            if (fd < 0) {
                ErrorInFastLLM("Disk MoE can't open weight file: " + fileName + "\n");
            }
            fds[fileName] = fd;
            return fd;
        }

    private:
        std::mutex locker;
        std::unordered_map<std::string, int> fds;
    };

    static DiskFileCache &GetDiskFileCache() {
        static DiskFileCache cache;
        return cache;
    }

    static void ReadDiskPartBytes(const DiskWeightPart &part, uint8_t *dst) {
        int fd = GetDiskFileCache().Get(part.fileName);
        uint64_t done = 0;
        while (done < part.bytes) {
            ssize_t ret = pread(fd, dst + done, part.bytes - done, part.fileOffset + done);
            if (ret < 0) {
                ErrorInFastLLM("Disk MoE read weight failed: " + part.fileName + "\n");
            }
            if (ret == 0) {
                ErrorInFastLLM("Disk MoE read EOF: " + part.fileName + "\n");
            }
            done += ret;
        }
    }

    static float BF16ToFloat(uint16_t v) {
        uint32_t u = (uint32_t)v << 16;
        return *(float*)&u;
    }

    static uint16_t FloatToBF16(float v) {
        return (uint16_t)(*(uint32_t*)&v >> 16);
    }

    static void ConvertDiskPart(uint8_t *dst, DataType dstType,
                                const uint8_t *src, DataType srcType,
                                size_t count) {
        if (dstType == srcType) {
            size_t bytes = 0;
            if (dstType == DataType::FLOAT32) {
                bytes = count * sizeof(float);
            } else if (dstType == DataType::FLOAT16 || dstType == DataType::BFLOAT16) {
                bytes = count * sizeof(uint16_t);
            }
            if (bytes > 0) {
                memcpy(dst, src, bytes);
                return;
            }
        }

        if (dstType == DataType::FLOAT32) {
            float *out = (float*)dst;
            if (srcType == DataType::FLOAT16) {
                const uint16_t *in = (const uint16_t*)src;
                for (size_t i = 0; i < count; i++) {
                    out[i] = half_to_float(in[i]);
                }
                return;
            }
            if (srcType == DataType::BFLOAT16) {
                const uint16_t *in = (const uint16_t*)src;
                for (size_t i = 0; i < count; i++) {
                    out[i] = BF16ToFloat(in[i]);
                }
                return;
            }
        } else if (dstType == DataType::FLOAT16) {
            uint16_t *out = (uint16_t*)dst;
            if (srcType == DataType::FLOAT32) {
                const float *in = (const float*)src;
                for (size_t i = 0; i < count; i++) {
                    out[i] = float_to_half(in[i]);
                }
                return;
            }
            if (srcType == DataType::BFLOAT16) {
                const uint16_t *in = (const uint16_t*)src;
                for (size_t i = 0; i < count; i++) {
                    out[i] = float_to_half(BF16ToFloat(in[i]));
                }
                return;
            }
        } else if (dstType == DataType::BFLOAT16) {
            uint16_t *out = (uint16_t*)dst;
            if (srcType == DataType::FLOAT32) {
                const float *in = (const float*)src;
                for (size_t i = 0; i < count; i++) {
                    out[i] = FloatToBF16(in[i]);
                }
                return;
            }
            if (srcType == DataType::FLOAT16) {
                const uint16_t *in = (const uint16_t*)src;
                for (size_t i = 0; i < count; i++) {
                    out[i] = FloatToBF16(half_to_float(in[i]));
                }
                return;
            }
        }
        ErrorInFastLLM("Disk MoE unsupported weight dtype conversion.\n");
    }

    static Data *LoadDiskWeight(const Data *weight) {
        if (weight == nullptr || !weight->isDiskWeight) {
            return (Data*)weight;
        }
        Data *loaded = new Data(weight->dataType);
        loaded->name = weight->name;
        loaded->isModelWeight = false;
        loaded->weightType = weight->weightType;
        loaded->tpLinearType = weight->tpLinearType;
        loaded->tpPackType = weight->tpPackType;
        loaded->perChannelAxis = weight->perChannelAxis;
        loaded->group = weight->group;
        loaded->groupCnt = weight->groupCnt;
        loaded->blockK = weight->blockK;
        loaded->blockM = weight->blockM;
        loaded->perChannelsConfigs = weight->perChannelsConfigs;
        loaded->scales = weight->scales;
        loaded->mins = weight->mins;
        loaded->zeros = weight->zeros;
        loaded->halfScales = weight->halfScales;
        loaded->isGGUFData = weight->isGGUFData;
        loaded->ggmlType = weight->ggmlType;
        loaded->IsRepacked = weight->IsRepacked;
        loaded->disableGGUFRepack = weight->disableGGUFRepack;
        loaded->forceGGUFFp32Dequant = weight->forceGGUFFp32Dequant;
        if (weight->ggmlTensor != nullptr) {
            loaded->ggmlTensor = (void*)(new ggml_tensor());
            (*(ggml_tensor*)loaded->ggmlTensor) = (*(ggml_tensor*)weight->ggmlTensor);
        }
        loaded->Resize(weight->dims);

        if (weight->dataType == DataType::DATA_GGUF_FORMAT) {
            uint64_t bytes = 0;
            for (auto &part : weight->diskWeightParts) {
                bytes += part.bytes;
            }
            loaded->expansionSize = bytes;
            loaded->expansionBytes = bytes;
            loaded->cpuData = new uint8_t[bytes];
            uint64_t dstOffset = 0;
            for (auto &part : weight->diskWeightParts) {
                ReadDiskPartBytes(part, loaded->cpuData + dstOffset);
                dstOffset += part.bytes;
            }
            loaded->IsRepacked = false;
            loaded->disableGGUFRepack = weight->disableGGUFRepack;
            loaded->forceGGUFFp32Dequant = weight->forceGGUFFp32Dequant;
            return loaded;
        }

        loaded->Allocate(false);

        uint64_t dstOffset = 0;
        std::vector<uint8_t> buffer;
        for (auto &part : weight->diskWeightParts) {
            if (part.isScalePart) {
                AssertInFastLLM(weight->dataType == DataType::NVFP4 && weight->dims.size() == 2 &&
                                weight->blockK > 0 && weight->blockM > 0,
                                "Disk MoE compact NVFP4 scale metadata is invalid.\n");
                size_t scaleBytes = GetNVFP4ScaleBytes(weight->dims[0], weight->dims[1], weight->blockK, weight->blockM);
                AssertInFastLLM(part.scaleOffset + part.bytes <= scaleBytes &&
                                loaded->expansionBytes >= GetNVFP4WeightBytes(weight->dims[0], weight->dims[1]) + scaleBytes,
                                "Disk MoE compact NVFP4 scale bytes mismatch.\n");
                ReadDiskPartBytes(part, loaded->cpuData + GetNVFP4WeightBytes(weight->dims[0], weight->dims[1]) + part.scaleOffset);
                continue;
            }
            if (weight->dataType == DataType::NVFP4 && weight->scales.empty() &&
                part.sourceDataType == DataType::NVFP4 && part.bytes == loaded->GetBytes()) {
                ReadDiskPartBytes(part, loaded->cpuData);
                dstOffset += GetNVFP4WeightBytes(weight->dims[0], weight->dims[1]);
                continue;
            }
            uint8_t *dst = loaded->cpuData + dstOffset;
            Data partData(weight->dataType, part.dims);
            uint64_t dstBytes = partData.GetBytes();
            if (part.sourceDataType == weight->dataType && part.bytes == dstBytes) {
                ReadDiskPartBytes(part, dst);
            } else {
                buffer.resize(part.bytes);
                ReadDiskPartBytes(part, buffer.data());
                ConvertDiskPart(dst, weight->dataType, buffer.data(), part.sourceDataType, DiskPartCount(part));
            }
            dstOffset += dstBytes;
        }
        return loaded;
    }

    static bool CudaSupportsDiskMoeWeight(DataType inputType, DataType weightType) {
        if (inputType != DataType::FLOAT32 &&
            inputType != DataType::FLOAT16 &&
            inputType != DataType::BFLOAT16) {
            return false;
        }
        if (weightType == DataType::FLOAT32 ||
            weightType == DataType::FLOAT16 ||
            weightType == DataType::BFLOAT16 ||
            weightType == DataType::FP8_E4M3 ||
            weightType == DataType::FP8_E4M3_BLOCK_128 ||
            weightType == DataType::NVFP4 ||
            weightType == DataType::NVFP4_BLOCK_16) {
            return true;
        }
        return false;
    }

    static bool CanPrepareCudaGateWeight(const Data &weight) {
        if (weight.dataType == DataType::FP8_E4M3) {
            return weight.blockK > 0 && weight.blockM == 128;
        }
        if (weight.dataType == DataType::NVFP4) {
            if (weight.scales.empty()) {
                return weight.blockK > 0 && weight.blockM > 0;
            }
            return weight.blockK > 0 && weight.blockM >= 16 &&
                   (weight.blockM % 16 == 0 || weight.blockM == weight.dims[1]);
        }
        return weight.dataType == DataType::FLOAT32 ||
               weight.dataType == DataType::FLOAT16 ||
               weight.dataType == DataType::BFLOAT16 ||
               weight.dataType == DataType::FP8_E4M3_BLOCK_128 ||
               weight.dataType == DataType::NVFP4_BLOCK_16;
    }

    static bool CanPrepareCudaDownWeight(const Data &weight) {
        if (weight.dataType == DataType::FP8_E4M3) {
            return weight.blockK > 0 && weight.blockM > 0;
        }
        if (weight.dataType == DataType::NVFP4) {
            if (weight.scales.empty()) {
                return weight.blockK > 0 && weight.blockM > 0;
            }
            return weight.blockK > 0 && weight.blockM >= 16 &&
                   (weight.blockM % 16 == 0 || weight.blockM == weight.dims[1]);
        }
        return true;
    }

    static bool CanUseCudaDiskMoe(Data &input, Data **weights, int weightsBatch,
                                  const std::vector<int> &loadIndices,
                                  const std::set<int> &selectedExperts,
                                  MoeGateType gateType) {
#ifndef USE_CUDA
        (void)input;
        (void)weights;
        (void)weightsBatch;
        (void)loadIndices;
        (void)selectedExperts;
        (void)gateType;
        return false;
#else
        if (!DiskMoeGpuPrefillEnabled() || input.cudaData == nullptr ||
            input.dims.size() < 2 || input.dims[0] < DiskMoeGpuPrefillMinTokens()) {
            return false;
        }
        for (int expert : selectedExperts) {
            int gate = expert * 2;
            int down = gate + 1;
            if (gate >= weightsBatch || down >= weightsBatch ||
                weights[gate] == nullptr || weights[down] == nullptr) {
                continue;
            }
            bool gateLoaded = std::find(loadIndices.begin(), loadIndices.end(), gate) != loadIndices.end();
            bool downLoaded = std::find(loadIndices.begin(), loadIndices.end(), down) != loadIndices.end();
            if ((weights[gate]->dataType == DataType::NVFP4 && !gateLoaded) ||
                (weights[down]->dataType == DataType::NVFP4 && !downLoaded)) {
                return false;
            }
            if (gateType != MoeGateGeglu) {
                if (!gateLoaded || !CanPrepareCudaGateWeight(*weights[gate])) {
                    return false;
                }
            }
            if (!CanPrepareCudaDownWeight(*weights[down])) {
                return false;
            }
            if (!CudaSupportsDiskMoeWeight(input.dataType, weights[gate]->dataType) ||
                !CudaSupportsDiskMoeWeight(input.dataType, weights[down]->dataType)) {
                return false;
            }
        }
        return true;
#endif
    }

    static void CrossSwigluReorderRows(const uint8_t *src, int rows, size_t bytesPerRow,
                                       std::vector<uint8_t> &dst) {
        AssertInFastLLM(rows % 2 == 0, "Disk MoE CrossSwiglu weight rows should be even.\n");
        dst.resize((size_t)rows * bytesPerRow);
        int half = rows / 2;
        for (int i = 0; i < half; i++) {
            memcpy(dst.data() + (size_t)(2 * i) * bytesPerRow,
                   src + (size_t)i * bytesPerRow, bytesPerRow);
            memcpy(dst.data() + (size_t)(2 * i + 1) * bytesPerRow,
                   src + (size_t)(half + i) * bytesPerRow, bytesPerRow);
        }
    }

    static size_t DiskWeightCudaRowBytes(const Data &weight) {
        int m = weight.dims[1];
        if (weight.dataType == DataType::DATA_GGUF_FORMAT) {
            return GetDataBytes((DataType)((int)DataType::DATA_GGUF_FORMAT + weight.ggmlType), 1, m);
        }
        return GetDataBytes(weight.dataType, 1, m);
    }

    static void CrossSwigluReorderWeightInPlace(Data &weight) {
        if (weight.dims.size() != 2 || weight.cpuData == nullptr) {
            return;
        }
        size_t bytesPerRow = DiskWeightCudaRowBytes(weight);
        size_t reorderBytes = (size_t)weight.dims[0] * bytesPerRow;
        AssertInFastLLM(weight.expansionBytes == 0 || reorderBytes <= weight.expansionBytes,
                        "Disk MoE CrossSwiglu weight storage is not row-contiguous.\n");
        std::vector<uint8_t> reordered;
        CrossSwigluReorderRows(weight.cpuData, weight.dims[0], bytesPerRow, reordered);
        memcpy(weight.cpuData, reordered.data(), reordered.size());
        if (weight.dataType == DataType::NVFP4 && weight.scales.empty() &&
            weight.blockK > 0 && weight.blockM > 0) {
            AssertInFastLLM(weight.blockK == 1,
                            "Disk MoE compact NVFP4 CrossSwiglu reorder requires blockK = 1.\n");
            int scaleMs = (weight.dims[1] - 1) / weight.blockM + 1;
            uint8_t *scaleData = weight.cpuData + GetNVFP4WeightBytes(weight.dims[0], weight.dims[1]);
            CrossSwigluReorderRows(scaleData, weight.dims[0], scaleMs, reordered);
            memcpy(scaleData, reordered.data(), reordered.size());
        }
    }

    static void PackFp8ToCudaBlock128(Data &weight) {
        if (weight.dataType != DataType::FP8_E4M3) {
            return;
        }
        if (weight.blockM != 128) {
            return;
        }
        AssertInFastLLM(weight.dims.size() == 2 && weight.cpuData != nullptr &&
                        weight.blockK > 0,
                        "Disk MoE FP8 weight can't be prepared for CUDA.\n");
        int k = weight.dims[0], m = weight.dims[1];
        int scaleKs = (k - 1) / weight.blockK + 1;
        int scaleMs = (m - 1) / weight.blockM + 1;
        AssertInFastLLM((int)weight.scales.size() >= scaleKs * scaleMs,
                        "Disk MoE FP8 scale metadata is invalid.\n");

        size_t rawBytesPerRow = GetDataBytes(DataType::FP8_E4M3, 1, m);
        size_t packedBytesPerRow = GetDataBytes(DataType::FP8_E4M3_BLOCK_128, 1, m);
        int packedBlocks = (m - 1) / 128 + 1;
        std::vector<uint8_t> packed((size_t)k * packedBytesPerRow, 0);

        for (int row = 0; row < k; row++) {
            uint8_t *dst = packed.data() + (size_t)row * packedBytesPerRow;
            uint8_t *src = weight.cpuData + (size_t)row * rawBytesPerRow;
            for (int block = 0; block < packedBlocks; block++) {
                int blockStart = block * 128;
                int blockElems = std::min(128, m - blockStart);
                memcpy(dst, src + blockStart, blockElems);
                dst += blockElems;

                size_t scaleIdx = (size_t)(row / weight.blockK) * scaleMs + block;
                float scale = weight.scales[scaleIdx];
                memcpy(dst, &scale, sizeof(float));
                dst += sizeof(float);
            }
        }

        delete[] weight.cpuData;
        weight.cpuData = new uint8_t[packed.size()];
        memcpy(weight.cpuData, packed.data(), packed.size());
        weight.dataType = DataType::FP8_E4M3_BLOCK_128;
        weight.UpdateUnitSize();
        weight.expansionSize = weight.Count(0);
        weight.expansionBytes = packed.size();
    }

    static void PackNvfp4ToCudaBlock16(Data &weight) {
        if (weight.dataType != DataType::NVFP4) {
            return;
        }
        AssertInFastLLM(weight.dims.size() == 2 && weight.cpuData != nullptr &&
                        weight.blockK > 0 && weight.blockM >= 16 &&
                        (weight.blockM % 16 == 0 || weight.blockM == weight.dims[1]),
                        "Disk MoE NVFP4 weight can't be prepared for CUDA.\n");
        int k = weight.dims[0], m = weight.dims[1];
        int scaleKs = (k - 1) / weight.blockK + 1;
        int scaleMs = (m - 1) / weight.blockM + 1;
        AssertInFastLLM((int)weight.scales.size() >= scaleKs * scaleMs,
                        "Disk MoE NVFP4 scale metadata is invalid.\n");

        const int packBlockM = 16;
        const int fp4BytesPerBlock = packBlockM / 2;
        int packedBlocks = (m - 1) / packBlockM + 1;
        size_t rawBytesPerRow = GetDataBytes(DataType::NVFP4, 1, m);
        size_t packedBytesPerRow = GetDataBytes(DataType::NVFP4_BLOCK_16, 1, m);
        std::vector<uint8_t> packed((size_t)k * packedBytesPerRow, 0);

        for (int row = 0; row < k; row++) {
            uint8_t *dst = packed.data() + (size_t)row * packedBytesPerRow;
            uint8_t *src = weight.cpuData + (size_t)row * rawBytesPerRow;
            for (int block = 0; block < packedBlocks; block++) {
                int blockStart = block * packBlockM;
                int blockElems = std::min(packBlockM, m - blockStart);
                int blockBytes = blockElems / 2;
                memcpy(dst, src + blockStart / 2, blockBytes);
                dst += fp4BytesPerBlock;

                int scaleCol = blockStart / weight.blockM;
                size_t scaleIdx = (size_t)(row / weight.blockK) * scaleMs + scaleCol;
                float scale = weight.scales[scaleIdx];
                memcpy(dst, &scale, sizeof(float));
                dst += sizeof(float);
            }
        }

        delete[] weight.cpuData;
        weight.cpuData = new uint8_t[packed.size()];
        memcpy(weight.cpuData, packed.data(), packed.size());
        weight.dataType = DataType::NVFP4_BLOCK_16;
        weight.UpdateUnitSize();
        weight.expansionSize = weight.Count(0);
        weight.expansionBytes = packed.size();
    }

    static void PrepareDiskWeightsForCuda(const std::vector<int> &loadIndices,
                                          std::vector<Data*> &tempWeights,
                                          MoeGateType gateType) {
        std::set<Data*> prepared;
        for (int index : loadIndices) {
            if (index >= 0 && index < (int)tempWeights.size() && tempWeights[index] != nullptr &&
                prepared.insert(tempWeights[index]).second) {
                Data &weight = *tempWeights[index];
                PackFp8ToCudaBlock128(weight);
                if (weight.dataType == DataType::NVFP4 && !weight.scales.empty()) {
                    PackNvfp4ToCudaBlock16(weight);
                }
                if (gateType != MoeGateGeglu && index % 2 == 0) {
                    CrossSwigluReorderWeightInPlace(weight);
                }
            }
        }
    }

#ifdef USE_CUDA
    static void ReleaseDiskTempWeightCudaExtras(Data *weight) {
        if (weight == nullptr) {
            return;
        }
        std::set<void*> released;
        for (void *ptr : weight->extraCudaData) {
            if (ptr != nullptr && released.insert(ptr).second) {
                FastllmCudaFree(ptr);
            }
        }
        for (void *ptr : weight->extraCudaHalfData) {
            if (ptr != nullptr && released.insert(ptr).second) {
                FastllmCudaFree(ptr);
            }
        }
        weight->extraCudaData.clear();
        weight->extraCudaHalfData.clear();
    }
#endif

    struct LoadDiskWeightsOp : MultiThreadBaseOp {
        Data **weights;
        std::vector<Data*> *tempWeights;
        const std::vector<int> *indices;
        int tid, threadCnt;

        LoadDiskWeightsOp(Data **weights, std::vector<Data*> *tempWeights,
                          const std::vector<int> *indices, int tid, int threadCnt) :
            weights(weights), tempWeights(tempWeights), indices(indices), tid(tid), threadCnt(threadCnt) {}

        void Run() {
            for (int i = tid; i < (int)indices->size(); i += threadCnt) {
                int index = (*indices)[i];
                (*tempWeights)[index] = LoadDiskWeight(weights[index]);
            }
        }
    };

    static void ConvertInputToFloat32(const Data &input, Data &output) {
        output.dataType = DataType::FLOAT32;
        output.Resize(input.dims);
        output.Allocate(false);
        int len = input.Count(0);
        float *dst = (float*)output.cpuData;
        if (input.dataType == DataType::FLOAT32) {
            memcpy(dst, input.cpuData, input.GetBytes());
        } else if (input.dataType == DataType::FLOAT16) {
            uint16_t *src = (uint16_t*)input.cpuData;
            for (int i = 0; i < len; i++) {
                dst[i] = half_to_float(src[i]);
            }
        } else if (input.dataType == DataType::BFLOAT16) {
            uint16_t *src = (uint16_t*)input.cpuData;
            for (int i = 0; i < len; i++) {
                dst[i] = BF16ToFloat(src[i]);
            }
        } else {
            ErrorInFastLLM("Disk MoE only supports FLOAT32/FLOAT16/BFLOAT16 input for quantized weights.\n");
        }
    }

    static void ConvertFloat32ToOutput(const Data &input, Data &output, DataType outputType) {
        output.dataType = outputType;
        output.Resize(input.dims);
        output.Allocate(false);
        int len = input.Count(0);
        float *src = (float*)input.cpuData;
        if (outputType == DataType::FLOAT32) {
            memcpy(output.cpuData, input.cpuData, input.GetBytes());
        } else if (outputType == DataType::FLOAT16) {
            uint16_t *dst = (uint16_t*)output.cpuData;
            for (int i = 0; i < len; i++) {
                dst[i] = float_to_half(src[i]);
            }
        } else if (outputType == DataType::BFLOAT16) {
            uint16_t *dst = (uint16_t*)output.cpuData;
            for (int i = 0; i < len; i++) {
                dst[i] = FloatToBF16(src[i]);
            }
        } else {
            ErrorInFastLLM("Disk MoE only supports FLOAT32/FLOAT16/BFLOAT16 output for quantized weights.\n");
        }
    }

    bool DiskMergeMOE::CanRun(const std::string &opType, const DataDict &datas,
                              const FloatDict &floatParams, const IntDict &intParams) {
        auto weightIt = datas.find("weights");
        if (weightIt == datas.end()) {
            return false;
        }
        Data **weights = (Data**)weightIt->second;
        if (weights == nullptr || weights[2] == nullptr) {
            return false;
        }
        auto biasIt = datas.find("biass");
        if (biasIt != datas.end()) {
            Data **biass = (Data**)biasIt->second;
            if (biass != nullptr && biass[0] != nullptr && biass[0]->dims.size() > 0) {
                return false;
            }
        }
        return weights[2]->isDiskWeight;
    }

    void DiskMergeMOE::Run(const std::string &opType, const DataDict &datas,
                           const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &index = *(datas.find("index")->second);
        Data &score = *(datas.find("score")->second);
        Data &w1 = *(datas.find("w1")->second);
        Data &w2 = *(datas.find("w2")->second);
        Data &w3 = *(datas.find("w3")->second);
        Data **weights = (Data**)datas.find("weights")->second;
        Data **biass = (Data**)datas.find("biass")->second;
        float sharedScale = floatParams.find("sharedScale") != floatParams.end() ?
            floatParams.find("sharedScale")->second : 1.0f;
        MoeGateType gateType = intParams.find("gateType") != intParams.end() ?
            (MoeGateType)intParams.find("gateType")->second : MoeGateSwiglu;
        int topk = index.dims[1];
        int weightsBatch = intParams.find("weights___batch") != intParams.end() ?
            intParams.find("weights___batch")->second : (topk + 1) * 2;

        std::set<int> selectedExperts;
        int32_t *indexData = (int32_t*)index.cpuData;
        int routedExpertCount = std::max(0, weightsBatch / 2 - 1);
        for (int i = 0; i < index.dims[0] * topk; i++) {
            int expertIdx = routedExpertCount <= 0 ? 0 : std::max(0, std::min(indexData[i], routedExpertCount - 1));
            selectedExperts.insert(expertIdx + 1);
        }
        if (weights[0] != nullptr) {
            selectedExperts.insert(0);
        }

        std::vector<Data*> tempWeights(weightsBatch, nullptr);
        std::vector<Data*> ownedWeights;
        for (int i = 0; i < weightsBatch; i++) {
            tempWeights[i] = weights[i];
        }
        std::vector<int> loadIndices;
        for (int expert : selectedExperts) {
            int gate = expert * 2;
            int down = gate + 1;
            if (gate >= weightsBatch || down >= weightsBatch || weights[gate] == nullptr || weights[down] == nullptr) {
                continue;
            }
            if (weights[gate]->isDiskWeight) {
                loadIndices.push_back(gate);
            }
            if (weights[down]->isDiskWeight) {
                loadIndices.push_back(down);
            }
        }
        if (loadIndices.size() > 0) {
            auto *pool = GetAlivePool();
            int threadCnt = std::min((int)loadIndices.size(), DiskMoeLoadThreads());
            threadCnt = std::min(threadCnt, (int)pool->threads.size());
            if (threadCnt <= 1) {
                for (int index : loadIndices) {
                    tempWeights[index] = LoadDiskWeight(weights[index]);
                }
            } else {
                std::vector<LoadDiskWeightsOp*> ops;
                for (int i = 0; i < threadCnt; i++) {
                    ops.push_back(new LoadDiskWeightsOp(weights, &tempWeights, &loadIndices, i, threadCnt));
                    pool->PushOp(i, ops.back());
                }
                for (int i = 0; i < threadCnt; i++) {
                    pool->Wait(i);
                    delete ops[i];
                }
            }
            for (int index : loadIndices) {
                ownedWeights.push_back(tempWeights[index]);
            }
        }
        auto releaseOwnedWeights = [&]() {
            std::set<Data*> releasedWeights;
            for (auto *weight : ownedWeights) {
                if (releasedWeights.insert(weight).second) {
#ifdef USE_CUDA
                    ReleaseDiskTempWeightCudaExtras(weight);
#endif
                    delete weight;
                }
            }
        };
        for (int i = 0; i < weightsBatch; i++) {
            if (tempWeights[i] != nullptr && tempWeights[i]->isDiskWeight) {
                tempWeights[i] = nullptr;
            }
        }
        if (tempWeights[2] == nullptr) {
            for (int expert : selectedExperts) {
                if (expert == 0) {
                    continue;
                }
                int gate = expert * 2;
                if (gate < weightsBatch && tempWeights[gate] != nullptr) {
                    // CpuMergeMOE uses weights[2] only as the representative dtype/shape
                    // when expert 0 is not selected. Avoid loading expert 0 just for that.
                    tempWeights[2] = tempWeights[gate];
                    break;
                }
            }
        }
        if (tempWeights[2] == nullptr) {
            ErrorInFastLLM("Disk MoE failed to load representative expert weight.\n");
        }

#ifdef USE_CUDA
        if (CanUseCudaDiskMoe(input, tempWeights.data(), weightsBatch, loadIndices, selectedExperts, gateType)) {
            PrepareDiskWeightsForCuda(loadIndices, tempWeights, gateType);
            std::unordered_set<int> cudaExperts(selectedExperts.begin(), selectedExperts.end());
            DoCudaMergeMOEFromCPU(input, output, index, score, w1, w2, w3,
                                  tempWeights.data(), biass, sharedScale,
                                  true, cudaExperts, true, gateType);
            releaseOwnedWeights();
            return;
        }
#endif

        DataDict diskDatas = datas;
        diskDatas["weights"] = (Data*)tempWeights.data();
        Data promotedInput, promotedOutput;
        DataType originalOutputType = output.dataType;
        bool promoteInput = tempWeights[2] != nullptr &&
                            (tempWeights[2]->dataType == DataType::BFLOAT16 ||
                             tempWeights[2]->dataType == DataType::FP8_E4M3 ||
                             tempWeights[2]->dataType == DataType::NVFP4) &&
                            input.dataType == DataType::FLOAT16;
        if (promoteInput) {
            ConvertInputToFloat32(input, promotedInput);
            promotedOutput.dataType = DataType::FLOAT32;
            promotedOutput.Resize(output.dims);
            diskDatas["input"] = &promotedInput;
            diskDatas["output"] = &promotedOutput;
        }
        Data sharedExpertOut;
        bool hasSharedExpertOut = false;
        if (tempWeights[0] != nullptr && tempWeights[1] != nullptr) {
            Data sharedGateOut, sharedSwigluOut;
            Data &mergeInput = promoteInput ? promotedInput : input;
            LinearSwigluBlock(&mergeInput, tempWeights[0], GetEmptyData(), &sharedGateOut, &sharedSwigluOut);
            Linear(sharedSwigluOut, *tempWeights[1], *GetEmptyData(), sharedExpertOut);
            tempWeights[0] = tempWeights[1] = nullptr;
            hasSharedExpertOut = true;
        }
        CpuMergeMOE::Run(opType, diskDatas, floatParams, intParams);
        if (hasSharedExpertOut) {
            Data &mergeOutput = promoteInput ? promotedOutput : output;
            AddTo(mergeOutput, sharedExpertOut, sharedScale);
        }
        if (promoteInput) {
            ConvertFloat32ToOutput(promotedOutput, output, originalOutputType);
        }

        releaseOwnedWeights();
    }
}
