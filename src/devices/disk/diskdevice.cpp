#include "devices/disk/diskdevice.h"
#include "utils.h"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <fcntl.h>
#include <mutex>
#include <set>
#include <unistd.h>
#include <unordered_map>

namespace fastllm {
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
        Data *loaded = new Data(weight->dataType, weight->dims);
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
        loaded->Allocate(false);

        uint64_t dstOffset = 0;
        std::vector<uint8_t> buffer;
        for (auto &part : weight->diskWeightParts) {
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
            ErrorInFastLLM("Disk MoE only supports FLOAT32/FLOAT16/BFLOAT16 input for BF16 weights.\n");
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
            ErrorInFastLLM("Disk MoE only supports FLOAT32/FLOAT16/BFLOAT16 output for BF16 weights.\n");
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
        Data &index = *(datas.find("index")->second);
        Data **weights = (Data**)datas.find("weights")->second;
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
        // CpuMergeMOE uses weights[2] as the representative dtype/shape.
        selectedExperts.insert(1);

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

        DataDict diskDatas = datas;
        diskDatas["weights"] = (Data*)tempWeights.data();
        Data promotedInput, promotedOutput;
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        DataType originalOutputType = output.dataType;
        bool promoteForBf16 = tempWeights[2] != nullptr &&
                              tempWeights[2]->dataType == DataType::BFLOAT16 &&
                              input.dataType != DataType::FLOAT32;
        if (promoteForBf16) {
            ConvertInputToFloat32(input, promotedInput);
            promotedOutput.dataType = DataType::FLOAT32;
            promotedOutput.Resize(input.dims);
            diskDatas["input"] = &promotedInput;
            diskDatas["output"] = &promotedOutput;
        }
        CpuMergeMOE::Run(opType, diskDatas, floatParams, intParams);
        if (promoteForBf16) {
            ConvertFloat32ToOutput(promotedOutput, output, originalOutputType);
        }

        for (auto *weight : ownedWeights) {
            delete weight;
        }
    }
}
