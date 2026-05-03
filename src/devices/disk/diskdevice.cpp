#include "devices/disk/diskdevice.h"
#include "utils.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>

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

    static void ReadDiskPart(const DiskWeightPart &part, std::vector<uint8_t> &buffer) {
        buffer.resize(part.bytes);
        FILE *fi = fopen(part.fileName.c_str(), "rb");
        if (fi == nullptr) {
            ErrorInFastLLM("Disk MoE can't open weight file: " + part.fileName + "\n");
        }
#if defined(_WIN32) || defined(_WIN64)
        _fseeki64(fi, part.fileOffset, 0);
#else
        fseek(fi, part.fileOffset, 0);
#endif
        size_t ret = fread(buffer.data(), 1, part.bytes, fi);
        fclose(fi);
        if (ret != part.bytes) {
            ErrorInFastLLM("Disk MoE read weight failed: " + part.fileName + "\n");
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
            Data partData(weight->dataType, part.dims);
            partData.Allocate(false);
            ReadDiskPart(part, buffer);
            ConvertDiskPart(partData.cpuData, weight->dataType, buffer.data(), part.sourceDataType, DiskPartCount(part));
            memcpy(loaded->cpuData + dstOffset, partData.cpuData, partData.GetBytes());
            dstOffset += partData.GetBytes();
        }
        return loaded;
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
        for (int expert : selectedExperts) {
            int gate = expert * 2;
            int down = gate + 1;
            if (gate >= weightsBatch || down >= weightsBatch || weights[gate] == nullptr || weights[down] == nullptr) {
                continue;
            }
            if (weights[gate]->isDiskWeight) {
                tempWeights[gate] = LoadDiskWeight(weights[gate]);
                ownedWeights.push_back(tempWeights[gate]);
            }
            if (weights[down]->isDiskWeight) {
                tempWeights[down] = LoadDiskWeight(weights[down]);
                ownedWeights.push_back(tempWeights[down]);
            }
        }

        DataDict diskDatas = datas;
        diskDatas["weights"] = (Data*)tempWeights.data();
        CpuMergeMOE::Run(opType, diskDatas, floatParams, intParams);

        for (auto *weight : ownedWeights) {
            delete weight;
        }
    }
}
