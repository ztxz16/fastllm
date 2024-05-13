//
// Created by huangyuyang on 4/11/23.
//

#include <fcntl.h>
#include <sys/mman.h>
#include <cstring>
#include <algorithm>

#include "utils.h"
#include "devices/tfacc/fastllm-tfacc.h"
#include "devices/cpu/alivethreadpool.h"
#include "json11.hpp"

namespace fastllm {
    extern AliveThreadPool *GetAlivePool();

    enum ComputeTaskType {
        None = 0,
        LinearInt4NoZero = 1,
        LinearInt8 = 2,
        LinearFloat16 = 3,
        LinearFloat32 = 4,
        LinearInt4Group = 5,

        AppendKVCache = 6,
        DoAttention = 7,

        GetComputeServerInfo = 10000,
        StartLongData = 10001,
        FinishLongData = 10002
    };

    const int PAGE = 16 * 1024;
    static int transLimit = 126 * 1024 * 1024;

    struct U8Buffer {
        std::vector <uint8_t> buffer;

        void Clear() {
            buffer.clear();
        }

        void WriteLongLong(long long v) {
            int oldLen = buffer.size();
            buffer.resize(oldLen + 8);
            ((long long*)(buffer.data() + oldLen))[0] = v;
        }

        void WriteInt(int v) {
            int oldLen = buffer.size();
            buffer.resize(oldLen + 4);
            ((int*)(buffer.data() + oldLen))[0] = v;
        }

        void WriteFloat(float v) {
            int oldLen = buffer.size();
            buffer.resize(oldLen + 4);
            ((float*)(buffer.data() + oldLen))[0] = v;
        }

        void WriteBytes(uint8_t *v, uint64_t len) {
            int oldLen = buffer.size();
            buffer.resize(oldLen + len);
            memcpy(buffer.data() + oldLen, v, len);
        }
    };

    TfaccClient::TfaccClient() {
        fd = open("/dev/thinkforce0", O_RDWR);
        buf = (volatile uint8_t *)mmap(NULL, 256 * 1024 * 1024, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 9997 * 0x1000);
        result = buf + 128 * 1024 * 1024;
        flag = (volatile int32_t*)(buf + 255 * 1024 * 1024);

        serverNumaCnt = 4;
        this->Launch(ComputeTaskType::GetComputeServerInfo);
        while (true) {
            int a = *flag;
            if (a == 0) {
                break;
            }
        }

        int len = ((int32_t*)this->result)[0];
        std::string infoString;
        for (int i = 0; i < len; i++) {
            infoString += this->result[4 + i];
        }
        std::string error;
        json11::Json info = json11::Json::parse(infoString, error);
        serverNumaCnt = info["numacnt"].int_value();
        this->Wait();
    }

    TfaccClient::~TfaccClient() {
        std::set <std::string> names = this->registerDataNames;
        for (auto &dataName : names) {
            this->UnregisterFastllmData(dataName);
        }
    }

    void TfaccClient::Launch(int opType) {
        asm volatile("dmb ish");
        volatile int *curFlag = flag;
        for (int i = 0; i < serverNumaCnt; i++) {
            *(curFlag) = opType;
            curFlag += PAGE;
            asm volatile("dmb ish");
        }
    }

    void TfaccClient::Wait() {
        while (true) {
            volatile int *curFlag = flag;
            int notFinish = 0;
            for (int i = 0; i < serverNumaCnt; i++) {
                notFinish |= (*curFlag);
                curFlag += PAGE;
            }
            if (!notFinish) {
                return;
            }
        }
    }

    void TfaccClient::SendLongMessage(uint8_t *buffer, uint64_t len) {
        for (uint64_t i = 0; i < len; i += transLimit) {
            int cur = (int)std::min((uint64_t)transLimit, len - i);
            ((int32_t*)this->buf)[0] = cur;
            memcpy((uint8_t*)this->buf + 4, buffer + i, cur);
            this->Launch(ComputeTaskType::StartLongData);
            this->Wait();
        }
        this->Launch(ComputeTaskType::FinishLongData);
        this->Wait();
    }

    void TfaccClient::RegisterFastllmData(fastllm::Data *data, const std::string &weightType) {
        if (data->name == "" || this->registerDataNames.find(data->name) != this->registerDataNames.end()) {
            return;
        }

        this->registerDataNames.insert(data->name);
        json11::Json config = json11::Json::object {
            {"op", "registerData"},
            {"dataName", data->name},
            {"weightType", weightType}
        };
        std::string configString = config.dump();

        U8Buffer buffer;
        buffer.WriteInt(configString.size());
        buffer.WriteBytes((uint8_t*)configString.data(), configString.size());

        buffer.WriteInt((int)data->dims.size());
        for (int i : data->dims) {
            buffer.WriteInt(i);
        }
        DataType dataType = data->dataType;
        if (dataType == DataType::FLOAT32 || dataType == DataType::BFLOAT16 || dataType == DataType::FLOAT16) {
            buffer.WriteInt((int) dataType);
            buffer.WriteBytes(data->cpuData, data->GetBytes());
        } else if (dataType == DataType::INT8 || dataType == DataType::INT4 || dataType == DataType::INT4_NOZERO) {
            buffer.WriteInt((int) dataType);
            buffer.WriteInt(data->perChannelAxis);
            int k = data->perChannelAxis == -1 ? 1 : data->dims[data->perChannelAxis];
            for (int i = 0; i < k; i++) {
                buffer.WriteFloat(data->perChannelsConfigs[i].min);
                buffer.WriteFloat(data->perChannelsConfigs[i].max);
            }
            buffer.WriteBytes(data->cpuData, data->GetBytes());
        } else if (dataType == DataType::INT4_GROUP) {
            buffer.WriteInt((int) dataType);
            buffer.WriteInt(data->perChannelAxis);
            buffer.WriteInt(data->group);
            buffer.WriteInt(data->groupCnt);
            int k = data->perChannelAxis == -1 ? 1 : data->dims[data->perChannelAxis];
            for (int i = 0; i < k * data->group; i++) {
                buffer.WriteFloat(data->perChannelsConfigs[i].min);
                buffer.WriteFloat(data->perChannelsConfigs[i].scale);
            }
            buffer.WriteBytes(data->cpuData, data->GetBytes());
        }

        SendLongMessage(buffer.buffer.data(), buffer.buffer.size());

        /// TODO: data->clear()
        delete[] data->cpuData;
        data->cpuData = new uint8_t[1];
    }

    void TfaccClient::UnregisterFastllmData(const std::string &dataName) {
        if (this->registerDataNames.find(dataName) == this->registerDataNames.end()) {
            return;
        }

        this->registerDataNames.erase(dataName);
        json11::Json config = json11::Json::object {
            {"op", "unregisterData"},
            {"dataName", dataName}
        };
        std::string configString = config.dump();

        U8Buffer buffer;
        buffer.WriteInt(configString.size());
        buffer.WriteBytes((uint8_t*)configString.data(), configString.size());
        SendLongMessage(buffer.buffer.data(), buffer.buffer.size());
    }

    void TfaccClient::RunTfaccLinearU(int n, int m, int k, int group, int groupCnt, 
                                     fastllm::Data *weight, fastllm::Data *bias,
                                     std::vector <LowBitConfig> *inputConfigs,
                                     uint8_t *uinput, float *output, 
                                     LinearExType exType) {
        std::string linearType = "linear";
        if (exType == LinearExType::ExSwiglu) {
            linearType = "linearSwiglu";
        }
        RegisterFastllmData(weight, linearType);
        RegisterFastllmData(bias, "bias");

        int opType = ComputeTaskType::LinearInt4NoZero;
        if (weight->dataType == DataType::INT8) {
            opType = ComputeTaskType::LinearInt8;
        }
        if (weight->dataType == DataType::INT4_GROUP) {
            opType = ComputeTaskType::LinearInt4Group;
        }

        float *biasData = bias->dims.size() > 0 ? (float *) bias->cpuData : nullptr;
        std::string biasName = biasData == nullptr ? "" : bias->name;

        int maxN = n;
        maxN = std::min(maxN, transLimit / m);
        maxN = std::min(maxN, (int)(transLimit / (k * sizeof(float))));

        // printf("maxN = %d\n", maxN);
        for (int baseN = 0; baseN < n; baseN += maxN) {
// auto st0 = std::chrono::system_clock::now();
            int curN = std::min(maxN, n - baseN);
            ((int32_t*)buf)[0] = curN;
            ((int32_t*)buf)[1] = m;
            ((int32_t*)buf)[2] = k;
            ((int32_t*)buf)[3] = group;
            ((int32_t*)buf)[4] = groupCnt;
            ((int32_t*)buf)[5] = weight->name.size();
            ((int32_t*)buf)[6] = biasName.size();
            ((int32_t*)buf)[7] = exType;
            
            volatile uint8_t *cur = (uint8_t*)buf + 10 * sizeof(int32_t);
            for (int i = 0; i < curN * group; i++) {
                ((float*)cur)[0] = (*inputConfigs)[baseN * group + i].min;
                ((float*)cur)[1] = (*inputConfigs)[baseN * group + i].max;
                cur += 2 * sizeof(float);
            }
            memcpy((uint8_t*)cur, weight->name.c_str(), weight->name.size());
            cur += weight->name.size();
            memcpy((uint8_t*)cur, biasName.c_str(), biasName.size());
            cur += biasName.size();
            RunMultiThreadMemcpy((uint8_t*)cur, uinput + baseN * m, curN * m, GetAlivePool());
// auto st1 = std::chrono::system_clock::now();
            this->Launch(opType);
            this->Wait();
// auto st2 = std::chrono::system_clock::now();
            int outK = k;
            if (exType == LinearExType::ExSwiglu) {
                outK /= 2;
            }

            auto pool = GetAlivePool();

            RunMultiThreadMemcpy(((uint8_t*) output) + baseN * outK * sizeof(int32_t),
                    (uint8_t*) result,
                    curN * outK * sizeof(int32_t), GetAlivePool());
// auto st3 = std::chrono::system_clock::now();
// if (n > 0) printf("n = %d, m = %d, k = %d, input = %f s, calc = %f s, output = %f. total = %f\n", n, m, k, GetSpan(st0, st1), GetSpan(st1, st2), GetSpan(st2, st3), GetSpan(st0, st3));
        }
    }

    void TfaccClient::RunTfaccLinearF(int n, int m, int k, fastllm::Data *weight, fastllm::Data *bias,
                                    float *input, float *output, LinearExType exType, DataType dataType) {
        std::string linearType = "linear";
        if (exType == LinearExType::ExSwiglu) {
            linearType = "linearSwiglu";
        }
        RegisterFastllmData(weight, linearType);
        RegisterFastllmData(bias, "bias");

        int opType = ComputeTaskType::LinearFloat32;
        if (weight->dataType == DataType::FLOAT16) {
            opType = ComputeTaskType::LinearFloat16;
        }
        float *biasData = bias->dims.size() > 0 ? (float *) bias->cpuData : nullptr;
        std::string biasName = biasData == nullptr ? "" : bias->name;

        int maxN = n;
        maxN = std::min(maxN, (int)(transLimit / (m * sizeof(float))));
        maxN = std::min(maxN, (int)(transLimit / (k * sizeof(float))));

        int unitSize = 4;
        if (dataType == DataType::FLOAT16) {
            unitSize = 2;
        } else if (dataType == DataType::FLOAT32) {
            unitSize = 4;
        }

        // printf("maxN = %d\n", maxN);
        for (int baseN = 0; baseN < n; baseN += maxN) {
            int curN = std::min(maxN, n - baseN);
            ((int32_t*)buf)[0] = curN;
            ((int32_t*)buf)[1] = m;
            ((int32_t*)buf)[2] = k;
            ((int32_t*)buf)[5] = weight->name.size();
            ((int32_t*)buf)[6] = biasName.size();
            ((int32_t*)buf)[7] = exType;
            ((int32_t*)buf)[8] = dataType;
            
            volatile uint8_t *cur = (uint8_t*)buf + 10 * sizeof(int32_t);
            memcpy((uint8_t*)cur, weight->name.c_str(), weight->name.size());
            cur += weight->name.size();
            memcpy((uint8_t*)cur, biasName.c_str(), biasName.size());
            cur += biasName.size();
            memcpy((uint8_t*)cur, (uint8_t*)input + baseN * m * unitSize, curN * m * unitSize);
            this->Launch(opType);
            this->Wait();

            int outK = k;
            if (exType == LinearExType::ExSwiglu) {
                outK /= 2;
            }
            memcpy(((uint8_t*) output) + baseN * outK * unitSize, (uint8_t*) result, curN * outK * unitSize);
        }
    }

    void TfaccClient::AppendKVCache(long long uid, Data *data) {
        int opType = ComputeTaskType::AppendKVCache;

        U8Buffer buffer;
        buffer.WriteLongLong(uid);
        buffer.WriteInt((int)data->dims.size());
        for (int i : data->dims) {
            buffer.WriteInt(i);
        }
        DataType dataType = data->dataType;
        if (dataType == DataType::FLOAT32 || dataType == DataType::BFLOAT16 || dataType == DataType::FLOAT16) {
            buffer.WriteInt((int) dataType);
            buffer.WriteBytes(data->cpuData, data->GetBytes());
        } else {
            ErrorInFastLLM("KVCache: Unsupport datatype.\n");
        }

        memcpy((uint8_t*)this->buf, buffer.buffer.data(), buffer.buffer.size());
        this->Launch(opType);
        this->Wait();
    }

    void TfaccClient::Attention(Data *q, Data *k, Data *v, int group, float scale, int maskType, Data *output) {
        int opType = ComputeTaskType::DoAttention;

        json11::Json config = json11::Json::object {
            {"op", "Attention"},
            {"kid", std::to_string(k->cacheUid)},
            {"vid", std::to_string(v->cacheUid)},
            {"qhead", q->dims[0]},
            {"qlen", q->dims[1]},
            {"qdim", q->dims[2]},
            {"qtype", q->dataType},
            {"group", group},
            {"scale", scale},
            {"maskType", maskType},
        };
        std::string configString = config.dump();

        U8Buffer buffer;
        buffer.WriteInt(configString.size());
        buffer.WriteBytes((uint8_t*)configString.data(), configString.size());
        //buffer.WriteBytes(q->cpuData, q->GetBytes());

        RunMultiThreadMemcpy((uint8_t*)this->buf, buffer.buffer.data(), buffer.buffer.size(), GetAlivePool());
        RunMultiThreadMemcpy((uint8_t*)this->buf + buffer.buffer.size(), q->cpuData, q->GetBytes(), GetAlivePool());
        this->Launch(opType);
        this->Wait();

        RunMultiThreadMemcpy(output->cpuData, (uint8_t*)result, output->GetBytes(), GetAlivePool());
    }
}
