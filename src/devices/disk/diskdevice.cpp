#include "devices/disk/diskdevice.h"
#include "blocks/baseblock.h"
#include "gguf.h"
#include "utils.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

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
        this->ops["Linear"] = (BaseOperator*)(new DiskLinearOp());
        this->ops["Embedding"] = (BaseOperator*)(new DiskEmbeddingOp(false));
        this->ops["EmbeddingDirect"] = (BaseOperator*)(new DiskEmbeddingOp(true));
        this->ops["MergeMOE"] = (BaseOperator*)(new DiskMergeMOE());
        this->ops["KimiK3RoutedExperts"] =
            (BaseOperator*)(new DiskKimiK3RoutedExpertsOp());
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

    static bool ParseEnvFlag(const char *env, bool defaultValue) {
        if (env == nullptr) {
            return defaultValue;
        }
        std::string value(env);
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return (char)std::tolower(c); });
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

    static bool DiskDirectIoEnabled() {
        static bool enabled = []() {
#ifndef O_DIRECT
            return false;
#else
            bool ret = ParseEnvFlag(std::getenv("FASTLLM_DISK_DIRECT_IO"), false);
            return ParseEnvFlag(std::getenv("FASTLLM_DISK_NO_CACHE"), ret);
#endif
        }();
        return enabled;
    }

    static int DiskMoeLoadThreads() {
        static int threads = []() {
            const char *env = std::getenv("FASTLLM_DISK_MOE_LOAD_THREADS");
            int v = env == nullptr ? (DiskDirectIoEnabled() ? 2 : 4) : atoi(env);
            return std::max(1, v);
        }();
        return threads;
    }

    class DiskFileCache {
    public:
        ~DiskFileCache() {
            for (auto &it : fds) {
                close(it.second);
            }
            for (auto &it : directFds) {
                close(it.second);
            }
        }

        int Get(const std::string &fileName, bool direct = false) {
            std::lock_guard<std::mutex> guard(locker);
            int flags = O_RDONLY;
#ifdef O_DIRECT
            if (direct) {
                flags |= O_DIRECT;
            }
#else
            direct = false;
#endif
            auto &cache = direct ? directFds : fds;
            auto it = cache.find(fileName);
            if (it != cache.end()) {
                return it->second;
            }
            int fd = open(fileName.c_str(), flags);
            if (fd < 0) {
                ErrorInFastLLM("Disk device can't open weight file: " + fileName + "\n");
            }
            cache[fileName] = fd;
            return fd;
        }

    private:
        std::mutex locker;
        std::unordered_map<std::string, int> fds;
        std::unordered_map<std::string, int> directFds;
    };

    static DiskFileCache &GetDiskFileCache() {
        static DiskFileCache cache;
        return cache;
    }

    class DiskDirectScratch {
    public:
        ~DiskDirectScratch() {
            free(buffer);
        }

        uint8_t *Acquire(size_t bytes) {
            if (inUse || bytes > 4 * 1024 * 1024) {
                return nullptr;
            }
            if (bytes > capacity) {
                free(buffer);
                buffer = nullptr;
                capacity = 0;
                void *ptr = nullptr;
                if (posix_memalign(&ptr, 4096, bytes) != 0) {
                    return nullptr;
                }
                buffer = (uint8_t*)ptr;
                capacity = bytes;
            }
            inUse = true;
            return buffer;
        }

        void Release() {
            inUse = false;
        }

    private:
        uint8_t *buffer = nullptr;
        size_t capacity = 0;
        bool inUse = false;
    };

    static thread_local DiskDirectScratch diskDirectScratch;

    class DiskDirectRange {
    public:
        DiskDirectRange(const DiskWeightPart &part, uint64_t offset, uint64_t bytes,
                        bool allowScratch = true) {
            AssertInFastLLM(part.fileOffset >= 0 && offset <= part.bytes &&
                            bytes <= part.bytes - offset && bytes > 0,
                            "Disk direct read range is invalid: " + part.fileName + "\n");
            const uint64_t alignment = 4096;
            const uint64_t maxFileOffset = (uint64_t)std::numeric_limits<off_t>::max();
            AssertInFastLLM((uint64_t)part.fileOffset <= maxFileOffset &&
                            offset <= maxFileOffset - (uint64_t)part.fileOffset &&
                            bytes <= maxFileOffset - ((uint64_t)part.fileOffset + offset),
                            "Disk direct read file offset is too large: " + part.fileName + "\n");
            uint64_t absoluteOffset = (uint64_t)part.fileOffset + offset;
            uint64_t alignedOffset = absoluteOffset / alignment * alignment;
            prefixBytes = (size_t)(absoluteOffset - alignedOffset);
            AssertInFastLLM(bytes <= std::numeric_limits<size_t>::max() - prefixBytes,
                            "Disk direct read range is too large.\n");
            size_t requiredBytes = prefixBytes + (size_t)bytes;
            AssertInFastLLM(requiredBytes <=
                                std::numeric_limits<size_t>::max() - (alignment - 1),
                            "Disk direct read aligned range is too large.\n");
            bufferBytes = (requiredBytes + alignment - 1) / alignment * alignment;
            buffer = allowScratch ? diskDirectScratch.Acquire(bufferBytes) : nullptr;
            if (buffer != nullptr) {
                usesScratch = true;
            } else {
                void *ptr = nullptr;
                AssertInFastLLM(posix_memalign(&ptr, alignment, bufferBytes) == 0 && ptr != nullptr,
                                "Disk direct read buffer allocation failed.\n");
                buffer = (uint8_t*)ptr;
            }

            int fd = GetDiskFileCache().Get(part.fileName, true);
            size_t done = 0;
            while (done < requiredBytes) {
                ssize_t ret = pread(fd, buffer + done, bufferBytes - done,
                                    (off_t)(alignedOffset + done));
                if (ret < 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    ReleaseBuffer();
                    ErrorInFastLLM("Disk direct read failed: " + part.fileName +
                                   ", errno = " + std::to_string(errno) + "\n");
                }
                if (ret == 0) {
                    ReleaseBuffer();
                    ErrorInFastLLM("Disk direct read EOF: " + part.fileName + "\n");
                }
                done += (size_t)ret;
            }
        }

        ~DiskDirectRange() {
            ReleaseBuffer();
        }

        uint8_t *Get() const {
            return buffer == nullptr ? nullptr : buffer + prefixBytes;
        }

    private:
        void ReleaseBuffer() {
            if (buffer == nullptr) {
                return;
            }
            if (usesScratch) {
                diskDirectScratch.Release();
            } else {
                free(buffer);
            }
            buffer = nullptr;
        }

        uint8_t *buffer = nullptr;
        size_t bufferBytes = 0;
        size_t prefixBytes = 0;
        bool usesScratch = false;
    };

    static void ReadDiskPartRange(const DiskWeightPart &part, uint64_t offset,
                                  uint64_t bytes, uint8_t *dst) {
        AssertInFastLLM(offset <= part.bytes && bytes <= part.bytes - offset,
                        "Disk weight read range is out of bounds: " + part.fileName + "\n");
        if (bytes == 0) {
            return;
        }
        if (DiskDirectIoEnabled()) {
            DiskDirectRange direct(part, offset, bytes);
            memcpy(dst, direct.Get(), bytes);
            return;
        }
        int fd = GetDiskFileCache().Get(part.fileName);
        uint64_t done = 0;
        while (done < bytes) {
            ssize_t ret = pread(fd, dst + done, bytes - done,
                                part.fileOffset + offset + done);
            if (ret < 0) {
                if (errno == EINTR) {
                    continue;
                }
                ErrorInFastLLM("Disk device read weight failed: " + part.fileName +
                               ", errno = " + std::to_string(errno) + "\n");
            }
            if (ret == 0) {
                ErrorInFastLLM("Disk device read EOF: " + part.fileName + "\n");
            }
            done += ret;
        }
    }

    static void ReadDiskPartBytes(const DiskWeightPart &part, uint8_t *dst) {
        ReadDiskPartRange(part, 0, part.bytes, dst);
    }

    static float BF16ToFloat(uint16_t v) {
        uint32_t u = (uint32_t)v << 16;
        float ret;
        memcpy(&ret, &u, sizeof(ret));
        return ret;
    }

    static uint16_t FloatToBF16(float v) {
        uint32_t u;
        memcpy(&u, &v, sizeof(u));
        return (uint16_t)(u >> 16);
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
                size_t i = 0;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                for (; i + 7 < count; i += 8) {
                    uint16x8_t bf16 = vld1q_u16(in + i);
                    float32x4_t low = vreinterpretq_f32_u32(
                        vshll_n_u16(vget_low_u16(bf16), 16));
                    float32x4_t high = vreinterpretq_f32_u32(
                        vshll_n_u16(vget_high_u16(bf16), 16));
                    float16x8_t fp16 = vcombine_f16(vcvt_f16_f32(low),
                                                   vcvt_f16_f32(high));
                    vst1q_u16(out + i, vreinterpretq_u16_f16(fp16));
                }
#endif
                for (; i < count; i++) {
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
        AssertInFastLLM(weight != nullptr && weight->isDiskWeight,
                        "LoadDiskWeight expects a lazy disk weight.\n");
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
        for (size_t partIndex = 0; partIndex < weight->diskWeightParts.size(); partIndex++) {
            const DiskWeightPart &part = weight->diskWeightParts[partIndex];
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
                DiskWeightPart readPart = part;
                if (DiskDirectIoEnabled() && readPart.fileOffset >= 0) {
                    // Merged gate/up projections are consecutive both in the
                    // destination tensor and in safetensors files. Reading the
                    // run at once avoids a second unaligned O_DIRECT request.
                    while (partIndex + 1 < weight->diskWeightParts.size()) {
                        const DiskWeightPart &next = weight->diskWeightParts[partIndex + 1];
                        if (next.isScalePart || next.fileName != readPart.fileName ||
                            next.fileOffset < 0 ||
                            (uint64_t)next.fileOffset !=
                                (uint64_t)readPart.fileOffset + readPart.bytes ||
                            next.sourceDataType != weight->dataType) {
                            break;
                        }
                        Data nextData(weight->dataType, next.dims);
                        uint64_t nextBytes = nextData.GetBytes();
                        if (next.bytes != nextBytes ||
                            nextBytes > std::numeric_limits<uint64_t>::max() - readPart.bytes) {
                            break;
                        }
                        readPart.bytes += nextBytes;
                        dstBytes += nextBytes;
                        partIndex++;
                    }
                }
                ReadDiskPartBytes(readPart, dst);
            } else {
                buffer.resize(part.bytes);
                ReadDiskPartBytes(part, buffer.data());
                ConvertDiskPart(dst, weight->dataType, buffer.data(), part.sourceDataType, DiskPartCount(part));
            }
            dstOffset += dstBytes;
        }
        return loaded;
    }

    static size_t DiskLinearChunkBytes() {
        static size_t bytes = []() {
            const char *env = std::getenv("FASTLLM_DISK_LINEAR_CHUNK_MB");
            unsigned long long mb = env == nullptr ? 64ULL : std::strtoull(env, nullptr, 10);
            mb = std::max(1ULL, std::min(1024ULL, mb));
            return (size_t)mb * 1024 * 1024;
        }();
        return bytes;
    }

    class DiskMappedRange {
    public:
        DiskMappedRange(const DiskWeightPart &part, uint64_t offset, uint64_t bytes) {
            long long fileOffset = part.fileOffset + offset;
            AssertInFastLLM(offset <= part.bytes && bytes <= part.bytes - offset &&
                            fileOffset >= 0 && bytes > 0,
                            "Disk Linear mmap range is invalid: " + part.fileName + "\n");
            int fd = GetDiskFileCache().Get(part.fileName);
            long pageSizeValue = sysconf(_SC_PAGESIZE);
            size_t pageSize = pageSizeValue > 0 ? (size_t)pageSizeValue : 4096;
            uint64_t alignedOffset = (uint64_t)fileOffset / pageSize * pageSize;
            size_t delta = (size_t)((uint64_t)fileOffset - alignedOffset);
            AssertInFastLLM(bytes <= std::numeric_limits<size_t>::max() - delta,
                            "Disk Linear mmap range is too large.\n");
            mappedBytes = (size_t)bytes + delta;
            mapped = mmap(nullptr, mappedBytes, PROT_READ, MAP_PRIVATE, fd, (off_t)alignedOffset);
            if (mapped == MAP_FAILED) {
                mapped = nullptr;
                mappedBytes = 0;
                return;
            }
            data = (uint8_t*)mapped + delta;
#ifdef MADV_SEQUENTIAL
            madvise(mapped, mappedBytes, MADV_SEQUENTIAL);
#endif
        }

        ~DiskMappedRange() {
            if (mapped == nullptr) {
                return;
            }
            munmap(mapped, mappedBytes);
        }

        uint8_t *Get() const {
            return data;
        }

    private:
        void *mapped = nullptr;
        size_t mappedBytes = 0;
        uint8_t *data = nullptr;
    };

    static void PrefetchDiskPartRange(const DiskWeightPart &part, uint64_t offset, uint64_t bytes) {
        if (DiskDirectIoEnabled() || bytes == 0 || offset > part.bytes || bytes > part.bytes - offset) {
            return;
        }
        int fd = GetDiskFileCache().Get(part.fileName);
        posix_fadvise(fd, (off_t)(part.fileOffset + offset), (off_t)bytes, POSIX_FADV_WILLNEED);
    }

    static size_t DiskPartRows(const DiskWeightPart &part) {
        if (part.dims.size() < 2) {
            return 0;
        }
        size_t rows = 1;
        for (int i = 0; i + 1 < (int)part.dims.size(); i++) {
            if (part.dims[i] <= 0 || rows > std::numeric_limits<size_t>::max() / part.dims[i]) {
                return 0;
            }
            rows *= part.dims[i];
        }
        return rows;
    }

    static size_t DiskTypeRowBytes(DataType type, int columns, const Data &weight) {
        if (type == DataType::DATA_GGUF_FORMAT) {
            return GetDataBytes((DataType)((int)DataType::DATA_GGUF_FORMAT + weight.ggmlType),
                                1, columns);
        }
        return GetDataBytes(type, 1, columns);
    }

    static bool IsDiskFloatStorageType(DataType type) {
        return type == DataType::FLOAT32 || type == DataType::FLOAT16 ||
               type == DataType::BFLOAT16;
    }

    static bool CanConvertDiskStorageType(DataType dst, DataType src) {
        return dst == src || (IsDiskFloatStorageType(dst) && IsDiskFloatStorageType(src));
    }

    static bool CanStreamDiskLinear(const Data &weight) {
        if (weight.dims.size() != 2 || weight.dims[0] <= 0 || weight.dims[1] <= 0 ||
            weight.diskWeightParts.empty()) {
            return false;
        }
        size_t rows = 0;
        for (const auto &part : weight.diskWeightParts) {
            if (part.isScalePart) {
                continue;
            }
            size_t partRows = DiskPartRows(part);
            if (partRows == 0 || part.dims.back() != weight.dims[1] ||
                !CanConvertDiskStorageType(weight.dataType, part.sourceDataType)) {
                return false;
            }
            size_t sourceRowBytes = DiskTypeRowBytes(part.sourceDataType, weight.dims[1], weight);
            if (sourceRowBytes == 0 || partRows > std::numeric_limits<size_t>::max() / sourceRowBytes ||
                part.bytes != partRows * sourceRowBytes) {
                return false;
            }
            rows += partRows;
        }
        if (rows != (size_t)weight.dims[0]) {
            return false;
        }
        if (weight.dataType == DataType::NVFP4 && weight.scales.empty()) {
            if (weight.blockK <= 0 || weight.blockM <= 0) {
                return false;
            }
            bool hasScalePart = false;
            for (const auto &part : weight.diskWeightParts) {
                hasScalePart |= part.isScalePart;
            }
            return hasScalePart;
        }
        return true;
    }

    template <class T>
    static void SliceRowMetadata(const std::vector<T> &src, std::vector<T> &dst,
                                 int totalRows, int rowStart, int rows) {
        if (src.empty()) {
            dst.clear();
            return;
        }
        if (totalRows > 0 && src.size() % (size_t)totalRows == 0) {
            size_t stride = src.size() / totalRows;
            size_t begin = (size_t)rowStart * stride;
            size_t end = (size_t)(rowStart + rows) * stride;
            AssertInFastLLM(end <= src.size(), "Disk Linear row metadata is out of bounds.\n");
            dst.assign(src.begin() + begin, src.begin() + end);
        } else {
            dst = src;
        }
    }

    static void SliceDiskLinearMetadata(const Data &src, Data &dst,
                                        int rowStart, int rows) {
        dst.weightType = WeightType::LINEAR;
        dst.tpLinearType = src.tpLinearType;
        dst.tpPackType = src.tpPackType;
        dst.perChannelAxis = src.perChannelAxis;
        dst.group = src.group;
        dst.groupCnt = src.groupCnt;
        dst.blockK = src.blockK;
        dst.blockM = src.blockM;
        dst.isGGUFData = src.isGGUFData;
        dst.ggmlType = src.ggmlType;
        dst.IsRepacked = src.IsRepacked;
        dst.disableGGUFRepack = src.disableGGUFRepack;
        dst.forceGGUFFp32Dequant = src.forceGGUFFp32Dequant;

        bool blockScales = (src.dataType == DataType::FP8_E4M3 ||
                            src.dataType == DataType::NVFP4) &&
                           !src.scales.empty() && src.blockK > 0 && src.blockM > 0;
        if (blockScales) {
            AssertInFastLLM(rowStart % src.blockK == 0,
                            "Disk Linear chunk must be aligned to blockK.\n");
            int ms = (src.dims[1] - 1) / src.blockM + 1;
            size_t begin = (size_t)(rowStart / src.blockK) * ms;
            size_t scaleRows = (rows - 1) / src.blockK + 1;
            size_t end = begin + scaleRows * ms;
            AssertInFastLLM(end <= src.scales.size(),
                            "Disk Linear block scale metadata is out of bounds.\n");
            dst.scales.assign(src.scales.begin() + begin, src.scales.begin() + end);
        } else {
            SliceRowMetadata(src.scales, dst.scales, src.dims[0], rowStart, rows);
        }
        SliceRowMetadata(src.mins, dst.mins, src.dims[0], rowStart, rows);
        SliceRowMetadata(src.zeros, dst.zeros, src.dims[0], rowStart, rows);
        SliceRowMetadata(src.halfScales, dst.halfScales, src.dims[0], rowStart, rows);
        SliceRowMetadata(src.perChannelsConfigs, dst.perChannelsConfigs,
                         src.dims[0], rowStart, rows);
        SliceRowMetadata(src.weightSum, dst.weightSum, src.dims[0], rowStart, rows);
    }

    static const DiskWeightPart *FindCompactScalePart(const Data &weight, size_t weightPartIndex) {
        for (size_t i = weightPartIndex + 1; i < weight.diskWeightParts.size(); i++) {
            if (!weight.diskWeightParts[i].isScalePart) {
                break;
            }
            return &weight.diskWeightParts[i];
        }
        return nullptr;
    }

    static void SetDiskGgmlChunkShape(ggml_tensor &tensor, int rows, int columns) {
        tensor.dims = {rows, columns};
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            tensor.ne[i] = 1;
        }
        tensor.ne[0] = columns;
        tensor.ne[1] = rows;
        tensor.nb[0] = ggml_type_size(tensor.type);
        tensor.nb[1] = tensor.nb[0] * (tensor.ne[0] / ggml_blck_size(tensor.type));
        for (int i = 2; i < GGML_MAX_DIMS; i++) {
            tensor.nb[i] = tensor.nb[i - 1] * tensor.ne[i - 1];
        }
    }

    static void RunDiskLinearChunk(Data &input, const Data &weight, const Data &bias,
                                   Data &output, uint8_t *weightData,
                                   uint64_t weightStorageBytes, int rowStart, int rows) {
        int columns = weight.dims[1];
        ggml_tensor ggmlChunk;
        Data chunkWeight(weight.dataType);
        chunkWeight.isFake = true;
        chunkWeight.name = weight.name;
        SliceDiskLinearMetadata(weight, chunkWeight, rowStart, rows);
        if (weight.dataType == DataType::DATA_GGUF_FORMAT) {
            AssertInFastLLM(weight.ggmlTensor != nullptr,
                            "Disk Linear GGUF metadata is missing.\n");
            ggmlChunk = *(ggml_tensor*)weight.ggmlTensor;
            SetDiskGgmlChunkShape(ggmlChunk, rows, columns);
            chunkWeight.ggmlTensor = &ggmlChunk;
            chunkWeight.disableGGUFRepack = true;
            chunkWeight.IsRepacked = false;
        }
        chunkWeight.Resize({rows, columns});
        chunkWeight.cpuData = weightData;
        chunkWeight.dataDevice = DataDevice::CPU;
        chunkWeight.expansionSize = chunkWeight.Count(0);
        chunkWeight.expansionBytes = weightStorageBytes;

        std::unique_ptr<Data> chunkBias;
        const Data *runBias = &bias;
        if (!bias.dims.empty()) {
            AssertInFastLLM(bias.dataType == DataType::FLOAT32 && bias.cpuData != nullptr &&
                            rowStart + rows <= bias.Count(0),
                            "Disk Linear bias metadata is invalid.\n");
            chunkBias.reset(new Data(DataType::FLOAT32, {rows}, DataDevice::CPU,
                                     bias.cpuData + (size_t)rowStart * sizeof(float)));
            runBias = chunkBias.get();
        }

        AssertInFastLLM(input.Count(0) == columns,
                        "Disk Linear chunked path expects one input row.\n");
        std::vector<int> chunkOutputDims = input.dims;
        chunkOutputDims.back() = rows;
        size_t outputOffset = GetDataBytes(output.dataType, 1, rowStart);
        AssertInFastLLM(GetDataBytes(output.dataType, 1, rows) > 0,
                        "Disk Linear output dtype is unsupported.\n");
        Data chunkOutput(output.dataType, chunkOutputDims, DataDevice::CPU,
                         output.cpuData + outputOffset);
        DoCpuLinear(input, chunkWeight, *runBias, chunkOutput);
    }

    void DiskLinearOp::Reshape(const std::string &opType, const DataDict &datas,
                               const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        AssertInFastLLM(weight.dims.size() == 2,
                        "Disk Linear weight shape should be 2D.\n");
        AssertInFastLLM(!input.dims.empty() && input.dims.back() == weight.dims[1],
                        "Disk Linear weight shape mismatch.\n");
        DoCpuLinearReshape(input, weight, output);
    }

    bool DiskLinearOp::CanRun(const std::string &opType, const DataDict &datas,
                              const FloatDict &floatParams, const IntDict &intParams) {
        if (intParams.find("exType") != intParams.end()) {
            return false;
        }
        auto it = datas.find("weight");
        return it != datas.end() && it->second != nullptr &&
               it->second->isDiskWeight && !it->second->diskWeightParts.empty();
    }

    void DiskLinearOp::Run(const std::string &opType, const DataDict &datas,
                           const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);
        AssertInFastLLM(weight.isDiskWeight && weight.cpuData == nullptr,
                        "Disk Linear expects a lazy disk weight.\n");
        AssertInFastLLM(bias.dataType == DataType::FLOAT32,
                        "Disk Linear bias should be float32.\n");

        int columns = weight.dims[1];
        int inputRows = input.Count(0) / columns;
        if (inputRows > 1) {
            // Prefill benefits from the regular full-matrix CPU kernel. The
            // weight is materialized only for this call and released as soon
            // as the Linear finishes. Decode continues through the bounded
            // chunked path below.
            std::unique_ptr<Data> loaded(LoadDiskWeight(&weight));
            DoCpuLinear(input, *loaded, bias, output);
            return;
        }

        if (!CanStreamDiskLinear(weight)) {
            std::unique_ptr<Data> loaded(LoadDiskWeight(&weight));
            DoCpuLinear(input, *loaded, bias, output);
            return;
        }

        output.Allocate();
        int rowStart = 0;
        const size_t chunkBudget = DiskLinearChunkBytes();
        for (size_t partIndex = 0; partIndex < weight.diskWeightParts.size(); partIndex++) {
            const DiskWeightPart &part = weight.diskWeightParts[partIndex];
            if (part.isScalePart) {
                continue;
            }
            size_t partRows = DiskPartRows(part);
            size_t sourceRowBytes = DiskTypeRowBytes(part.sourceDataType, columns, weight);
            size_t targetRowBytes = DiskTypeRowBytes(weight.dataType, columns, weight);
            size_t maxRowBytes = std::max(sourceRowBytes, targetRowBytes);
            size_t rowsPerChunk = std::max((size_t)1, chunkBudget / std::max((size_t)1, maxRowBytes));
            int rowAlignment = (weight.blockK > 0 &&
                                (weight.dataType == DataType::FP8_E4M3 ||
                                 weight.dataType == DataType::NVFP4)) ? weight.blockK : 1;
            if (rowsPerChunk >= (size_t)rowAlignment) {
                rowsPerChunk = rowsPerChunk / rowAlignment * rowAlignment;
            }
            rowsPerChunk = std::max(rowsPerChunk, (size_t)rowAlignment);

            const DiskWeightPart *compactScalePart = nullptr;
            bool compactNVFP4 = weight.dataType == DataType::NVFP4 && weight.scales.empty();
            if (compactNVFP4) {
                compactScalePart = FindCompactScalePart(weight, partIndex);
                AssertInFastLLM(compactScalePart != nullptr,
                                "Disk Linear compact NVFP4 scale part is missing.\n");
            }

            // The vocabulary projection is typically the largest decode
            // matrix and may require dtype conversion. Overlap reading its
            // next bounded chunk with conversion and matvec of the current
            // chunk. Each chunk is still discarded immediately after use.
            bool pipelineConvertedChunks = DiskDirectIoEnabled() &&
                                           !compactNVFP4 &&
                                           part.sourceDataType != weight.dataType &&
                                           partRows > rowsPerChunk;
            if (pipelineConvertedChunks) {
                struct ConvertedChunk {
                    std::vector<uint8_t> storage;
                    std::vector<uint8_t> sourceStorage;
                    std::unique_ptr<DiskDirectRange> directStorage;
                    int rows = 0;
                    size_t localRow = 0;
                    uint64_t storageBytes = 0;

                    uint8_t *Data() {
                        return directStorage == nullptr ? storage.data() : directStorage->Get();
                    }
                };
                auto loadChunk = [&](size_t localRow) {
                    std::unique_ptr<ConvertedChunk> chunk(new ConvertedChunk());
                    chunk->localRow = localRow;
                    chunk->rows = (int)std::min(rowsPerChunk, partRows - localRow);
                    uint64_t sourceOffset = localRow * sourceRowBytes;
                    uint64_t sourceBytes = (uint64_t)chunk->rows * sourceRowBytes;
                    uint64_t targetBytes = (uint64_t)chunk->rows * targetRowBytes;
                    chunk->storageBytes = targetBytes;
                    if (targetBytes <= sourceBytes) {
                        chunk->directStorage.reset(new DiskDirectRange(
                            part, sourceOffset, sourceBytes, false));
                        ConvertDiskPart(chunk->directStorage->Get(), weight.dataType,
                                        chunk->directStorage->Get(), part.sourceDataType,
                                        (size_t)chunk->rows * columns);
                    } else {
                        chunk->storage.resize((size_t)targetBytes);
                        chunk->sourceStorage.resize((size_t)sourceBytes);
                        ReadDiskPartRange(part, sourceOffset, sourceBytes,
                                          chunk->sourceStorage.data());
                        ConvertDiskPart(chunk->storage.data(), weight.dataType,
                                        chunk->sourceStorage.data(), part.sourceDataType,
                                        (size_t)chunk->rows * columns);
                    }
                    return chunk;
                };

                std::unique_ptr<ConvertedChunk> current = loadChunk(0);
                while (current != nullptr) {
                    size_t nextRow = current->localRow + current->rows;
                    std::future<std::unique_ptr<ConvertedChunk>> next;
                    if (nextRow < partRows) {
                        next = std::async(std::launch::async, loadChunk, nextRow);
                    }
                    RunDiskLinearChunk(input, weight, bias, output,
                                       current->Data(), current->storageBytes,
                                       rowStart + (int)current->localRow, current->rows);
                    current.reset();
                    if (!next.valid()) {
                        break;
                    }
                    current = next.get();
                }
                rowStart += (int)partRows;
                continue;
            }

            for (size_t localRow = 0; localRow < partRows; localRow += rowsPerChunk) {
                int chunkRows = (int)std::min(rowsPerChunk, partRows - localRow);
                uint64_t sourceOffset = localRow * sourceRowBytes;
                uint64_t sourceBytes = (uint64_t)chunkRows * sourceRowBytes;
                uint64_t targetBytes = (uint64_t)chunkRows * targetRowBytes;

                size_t nextRow = localRow + chunkRows;
                if (nextRow < partRows) {
                    size_t nextRows = std::min(rowsPerChunk, partRows - nextRow);
                    PrefetchDiskPartRange(part, nextRow * sourceRowBytes,
                                          nextRows * sourceRowBytes);
                }

                std::unique_ptr<DiskMappedRange> mapped;
                std::unique_ptr<DiskDirectRange> direct;
                std::vector<uint8_t> storage;
                std::vector<uint8_t> sourceStorage;
                uint8_t *chunkData = nullptr;
                uint64_t chunkStorageBytes = targetBytes;

                bool canMap = !compactNVFP4 && part.sourceDataType == weight.dataType;
                if (canMap) {
                    if (DiskDirectIoEnabled()) {
                        direct.reset(new DiskDirectRange(part, sourceOffset, sourceBytes));
                        chunkData = direct->Get();
                    } else {
                        mapped.reset(new DiskMappedRange(part, sourceOffset, sourceBytes));
                        chunkData = mapped->Get();
                        if (chunkData == nullptr) {
                            mapped.reset();
                        }
                    }
                }
                if (chunkData == nullptr && compactNVFP4) {
                    int ms = (columns - 1) / weight.blockM + 1;
                    AssertInFastLLM(localRow % weight.blockK == 0,
                                    "Disk Linear compact NVFP4 chunk alignment mismatch.\n");
                    size_t scaleRows = (chunkRows - 1) / weight.blockK + 1;
                    size_t scaleBytes = scaleRows * ms;
                    storage.resize((size_t)targetBytes + scaleBytes);
                    ReadDiskPartRange(part, sourceOffset, sourceBytes, storage.data());
                    ReadDiskPartRange(*compactScalePart,
                                      (localRow / weight.blockK) * ms,
                                      scaleBytes, storage.data() + targetBytes);
                    chunkData = storage.data();
                    chunkStorageBytes += scaleBytes;
                } else if (chunkData == nullptr && part.sourceDataType == weight.dataType) {
                    storage.resize((size_t)sourceBytes);
                    ReadDiskPartRange(part, sourceOffset, sourceBytes, storage.data());
                    chunkData = storage.data();
                } else if (chunkData == nullptr) {
                    bool canConvertInPlace = targetBytes <= sourceBytes;
                    if (DiskDirectIoEnabled() && canConvertInPlace) {
                        direct.reset(new DiskDirectRange(part, sourceOffset,
                                                         sourceBytes, false));
                        ConvertDiskPart(direct->Get(), weight.dataType, direct->Get(),
                                        part.sourceDataType, (size_t)chunkRows * columns);
                        chunkData = direct->Get();
                    } else if (canConvertInPlace) {
                        storage.resize((size_t)sourceBytes);
                        ReadDiskPartRange(part, sourceOffset, sourceBytes, storage.data());
                        ConvertDiskPart(storage.data(), weight.dataType, storage.data(),
                                        part.sourceDataType, (size_t)chunkRows * columns);
                    } else {
                        storage.resize((size_t)targetBytes);
                        sourceStorage.resize((size_t)sourceBytes);
                        ReadDiskPartRange(part, sourceOffset, sourceBytes, sourceStorage.data());
                        ConvertDiskPart(storage.data(), weight.dataType, sourceStorage.data(),
                                        part.sourceDataType, (size_t)chunkRows * columns);
                    }
                    if (chunkData == nullptr) {
                        chunkData = storage.data();
                    }
                }

                RunDiskLinearChunk(input, weight, bias, output, chunkData,
                                   chunkStorageBytes, rowStart + (int)localRow, chunkRows);
            }
            rowStart += (int)partRows;
        }
        AssertInFastLLM(rowStart == weight.dims[0],
                        "Disk Linear row count mismatch.\n");
    }

    static float ReadDiskFloatValue(const uint8_t *data, DataType type, size_t index) {
        if (type == DataType::FLOAT32) {
            return ((const float*)data)[index];
        }
        if (type == DataType::FLOAT16) {
            return half_to_float(((const uint16_t*)data)[index]);
        }
        if (type == DataType::BFLOAT16) {
            return BF16ToFloat(((const uint16_t*)data)[index]);
        }
        ErrorInFastLLM("Disk Embedding unsupported dtype.\n");
        return 0.0f;
    }

    static void WriteDiskFloatValue(uint8_t *data, DataType type, size_t index, float value) {
        if (type == DataType::FLOAT32) {
            ((float*)data)[index] = value;
        } else if (type == DataType::FLOAT16) {
            ((uint16_t*)data)[index] = float_to_half(value);
        } else if (type == DataType::BFLOAT16) {
            ((uint16_t*)data)[index] = FloatToBF16(value);
        } else {
            ErrorInFastLLM("Disk Embedding unsupported output dtype.\n");
        }
    }

    static const DiskWeightPart *FindDiskRowPart(const Data &weight, int row,
                                                  size_t &localRow) {
        size_t start = 0;
        for (const auto &part : weight.diskWeightParts) {
            if (part.isScalePart) {
                continue;
            }
            size_t rows = DiskPartRows(part);
            if ((size_t)row < start + rows) {
                localRow = row - start;
                return &part;
            }
            start += rows;
        }
        return nullptr;
    }

    void DiskEmbeddingOp::Reshape(const std::string &opType, const DataDict &datas,
                                  const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        AssertInFastLLM(weight.dims.size() == 2 && IsDiskFloatStorageType(weight.dataType),
                        "Disk Embedding expects a 2D floating-point weight.\n");
        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16,
                        "Disk Embedding input should be float32 or float16.\n");
        std::vector<int> dims = input.dims;
        dims.push_back(weight.dims[1]);
        output.dataType = direct ? weight.dataType : input.dataType;
        if (!direct && weight.dataType == DataType::FLOAT16) {
            output.dataType = DataType::FLOAT16;
        }
        output.Resize(dims);
        weight.weightType = WeightType::EMBEDDING;
    }

    bool DiskEmbeddingOp::CanRun(const std::string &opType, const DataDict &datas,
                                 const FloatDict &floatParams, const IntDict &intParams) {
        auto it = datas.find("weight");
        return it != datas.end() && it->second != nullptr &&
               it->second->isDiskWeight && !it->second->diskWeightParts.empty() &&
               IsDiskFloatStorageType(it->second->dataType);
    }

    void DiskEmbeddingOp::Run(const std::string &opType, const DataDict &datas,
                              const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        output.Allocate();
        int vocabSize = weight.dims[0];
        int columns = weight.dims[1];
        int inputLen = input.Count(0);
        size_t targetRowBytes = DiskTypeRowBytes(weight.dataType, columns, weight);
        size_t outputRowBytes = GetDataBytes(output.dataType, 1, columns);
        std::vector<uint8_t> targetRow(targetRowBytes);
        std::vector<uint8_t> sourceRow;
        std::unordered_map<int, int> previousRows;

        for (int i = 0; i < inputLen; i++) {
            float rawToken = input.dataType == DataType::FLOAT32 ?
                ((float*)input.cpuData)[i] : half_to_float(((uint16_t*)input.cpuData)[i]);
            AssertInFastLLM(std::isfinite(rawToken),
                            "Disk Embedding token is not finite.\n");
            int token = (int)(rawToken + 1e-9f);
            AssertInFastLLM(token >= 0 && token < vocabSize,
                            "Disk Embedding token is out of range: " + std::to_string(token) + "\n");
            auto previous = previousRows.find(token);
            if (previous != previousRows.end()) {
                memcpy(output.cpuData + (size_t)i * outputRowBytes,
                       output.cpuData + (size_t)previous->second * outputRowBytes,
                       outputRowBytes);
                continue;
            }

            size_t localRow = 0;
            const DiskWeightPart *part = FindDiskRowPart(weight, token, localRow);
            AssertInFastLLM(part != nullptr,
                            "Disk Embedding row metadata is incomplete.\n");
            size_t sourceRowBytes = DiskTypeRowBytes(part->sourceDataType, columns, weight);
            if (part->sourceDataType == weight.dataType) {
                ReadDiskPartRange(*part, localRow * sourceRowBytes,
                                  sourceRowBytes, targetRow.data());
            } else {
                sourceRow.resize(sourceRowBytes);
                ReadDiskPartRange(*part, localRow * sourceRowBytes,
                                  sourceRowBytes, sourceRow.data());
                ConvertDiskPart(targetRow.data(), weight.dataType, sourceRow.data(),
                                part->sourceDataType, columns);
            }

            uint8_t *dst = output.cpuData + (size_t)i * outputRowBytes;
            if (output.dataType == weight.dataType) {
                memcpy(dst, targetRow.data(), outputRowBytes);
            } else {
                for (int column = 0; column < columns; column++) {
                    WriteDiskFloatValue(dst, output.dataType, column,
                                        ReadDiskFloatValue(targetRow.data(), weight.dataType, column));
                }
            }
            previousRows[token] = i;
        }
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
        std::atomic<int> *nextIndex;

        LoadDiskWeightsOp(Data **weights, std::vector<Data*> *tempWeights,
                          const std::vector<int> *indices, std::atomic<int> *nextIndex) :
            weights(weights), tempWeights(tempWeights), indices(indices), nextIndex(nextIndex) {}

        void Run() {
            while (true) {
                int i = nextIndex->fetch_add(1, std::memory_order_relaxed);
                if (i >= (int)indices->size()) {
                    break;
                }
                int index = (*indices)[i];
                (*tempWeights)[index] = LoadDiskWeight(weights[index]);
            }
        }
    };

    static const DiskWeightPart *FirstDiskPayloadPart(const Data *weight) {
        if (weight == nullptr) {
            return nullptr;
        }
        for (const auto &part : weight->diskWeightParts) {
            if (!part.isScalePart) {
                return &part;
            }
        }
        return nullptr;
    }

    static void LoadDiskWeightsInParallel(
            Data **weights, std::vector<Data*> &loadedWeights,
            std::vector<int> &loadIndices) {
        if (loadIndices.empty()) {
            return;
        }
        AssertInFastLLM(weights != nullptr &&
                            loadedWeights.size() >
                                (size_t)*std::max_element(
                                    loadIndices.begin(), loadIndices.end()),
                        "Disk weight load index is out of bounds.\n");
        if (DiskDirectIoEnabled()) {
            std::stable_sort(
                loadIndices.begin(), loadIndices.end(),
                [&](int left, int right) {
                    const DiskWeightPart *a =
                        FirstDiskPayloadPart(weights[left]);
                    const DiskWeightPart *b =
                        FirstDiskPayloadPart(weights[right]);
                    if (a == nullptr || b == nullptr) {
                        return a != nullptr;
                    }
                    if (a->fileName != b->fileName) {
                        return a->fileName < b->fileName;
                    }
                    return a->fileOffset < b->fileOffset;
                });
        }

        AliveThreadPool *pool = GetAlivePool();
        int threadCount = std::min(
            {(int)loadIndices.size(), DiskMoeLoadThreads(),
             (int)pool->threads.size()});
        if (threadCount <= 1) {
            for (int index : loadIndices) {
                loadedWeights[index] = LoadDiskWeight(weights[index]);
            }
            return;
        }

        std::vector<LoadDiskWeightsOp*> tasks;
        tasks.reserve(threadCount);
        std::atomic<int> nextIndex(0);
        for (int thread = 0; thread < threadCount; thread++) {
            tasks.push_back(new LoadDiskWeightsOp(
                weights, &loadedWeights, &loadIndices, &nextIndex));
            pool->PushOp(thread, tasks.back());
        }
        for (int thread = 0; thread < threadCount; thread++) {
            pool->Wait(thread);
            delete tasks[thread];
        }
    }

    bool DiskKimiK3RoutedExpertsOp::CanRun(
            const std::string &, const DataDict &datas,
            const FloatDict &, const IntDict &intParams) {
        auto inputIt = datas.find("input");
        auto weightsIt = datas.find("w1s");
        auto countIt = intParams.find("experts___batch");
        if (inputIt == datas.end() || weightsIt == datas.end() ||
            countIt == intParams.end() || countIt->second <= 0 ||
            inputIt->second == nullptr ||
            inputIt->second->dataType != DataType::BFLOAT16) {
            return false;
        }
        Data **weights = (Data**)weightsIt->second;
        return weights != nullptr && weights[0] != nullptr &&
               weights[0]->isDiskWeight;
    }

    void DiskKimiK3RoutedExpertsOp::Run(
            const std::string &opType, const DataDict &datas,
            const FloatDict &floatParams, const IntDict &intParams) {
        Data &index = *datas.find("index")->second;
        Data **w1s = (Data**)datas.find("w1s")->second;
        Data **w2s = (Data**)datas.find("w2s")->second;
        Data **w3s = (Data**)datas.find("w3s")->second;
        int expertCount = intParams.find("experts___batch")->second;
        AssertInFastLLM(
            index.dataType == DataType::INT32 && index.dims.size() == 2 &&
                index.cpuData != nullptr && expertCount > 0 &&
                w1s != nullptr && w2s != nullptr && w3s != nullptr,
            "KimiK3 disk routed-expert metadata is invalid.\n");

        std::set<int> selectedExperts;
        const int32_t *indexData = (const int32_t*)index.cpuData;
        for (uint64_t i = 0; i < index.Count(0); i++) {
            int expert = indexData[i];
            AssertInFastLLM(
                expert >= 0 && expert < expertCount,
                "KimiK3 disk routed-expert index is out of range.\n");
            selectedExperts.insert(expert);
        }

        std::vector<Data*> sourceWeights;
        std::vector<int> sourceKinds;
        std::vector<int> sourceExperts;
        sourceWeights.reserve(selectedExperts.size() * 3);
        sourceKinds.reserve(selectedExperts.size() * 3);
        sourceExperts.reserve(selectedExperts.size() * 3);
        for (int expert : selectedExperts) {
            for (int kind = 0; kind < 3; kind++) {
                Data *weight = kind == 0 ? w1s[expert] :
                               (kind == 1 ? w2s[expert] : w3s[expert]);
                AssertInFastLLM(
                    weight != nullptr,
                    "KimiK3 disk routed-expert weight is missing.\n");
                if (weight->isDiskWeight) {
                    sourceWeights.push_back(weight);
                    sourceKinds.push_back(kind);
                    sourceExperts.push_back(expert);
                }
            }
        }

        std::vector<Data*> loadedWeights(sourceWeights.size(), nullptr);
        std::vector<int> loadIndices(sourceWeights.size());
        for (int i = 0; i < (int)loadIndices.size(); i++) {
            loadIndices[i] = i;
        }
        LoadDiskWeightsInParallel(
            sourceWeights.data(), loadedWeights, loadIndices);

        std::vector<std::unique_ptr<Data>> ownedWeights;
        ownedWeights.reserve(loadedWeights.size());
        std::vector<Data*> tempW1s(w1s, w1s + expertCount);
        std::vector<Data*> tempW2s(w2s, w2s + expertCount);
        std::vector<Data*> tempW3s(w3s, w3s + expertCount);
        for (int i = 0; i < (int)loadedWeights.size(); i++) {
            AssertInFastLLM(
                loadedWeights[i] != nullptr,
                "KimiK3 disk routed-expert weight load failed.\n");
            Data *loaded = loadedWeights[i];
            if (sourceKinds[i] == 0) {
                tempW1s[sourceExperts[i]] = loaded;
            } else if (sourceKinds[i] == 1) {
                tempW2s[sourceExperts[i]] = loaded;
            } else {
                tempW3s[sourceExperts[i]] = loaded;
            }
            ownedWeights.emplace_back(loaded);
        }

        DataDict diskDatas = datas;
        diskDatas["w1s"] = (Data*)tempW1s.data();
        diskDatas["w2s"] = (Data*)tempW2s.data();
        diskDatas["w3s"] = (Data*)tempW3s.data();
        CpuKimiK3RoutedExpertsOp::Run(
            opType, diskDatas, floatParams, intParams);
    }

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
        auto weightsBatchIt = intParams.find("weights___batch");
        if (weightsBatchIt == intParams.end() || weightsBatchIt->second <= 2) {
            return false;
        }
        Data **weights = (Data**)weightIt->second;
        if (weights == nullptr || weights[2] == nullptr) {
            return false;
        }
        auto biasIt = datas.find("biass");
        if (biasIt != datas.end()) {
            Data **biass = (Data**)biasIt->second;
            auto biassBatchIt = intParams.find("biass___batch");
            int biassBatch = biassBatchIt == intParams.end() ? 0 : biassBatchIt->second;
            if (biass != nullptr && biassBatch > 0 &&
                biass[0] != nullptr && biass[0]->dims.size() > 0) {
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
            LoadDiskWeightsInParallel(weights, tempWeights, loadIndices);
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
