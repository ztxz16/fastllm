//
// Created by huangyuyang on 24-4-11.
//

#include <fcntl.h>
#include <sys/mman.h>
#include <cstring>

#include "computeutils.h"
#include "computeserver.h"
#include "json11.hpp"
#include "cpudevice.h"
#include "gguf.h"

const int VERSION = 1;

namespace fastllm {
    extern FP16ToFP32Manager fp16tofp32;
    extern void Float16ToFloat32(uint16_t *float16, float *float32, int len);
    extern void Float32ToFloat16(float *float32, uint16_t *float16, int len);
    extern void Float32ToBFloat16(float *float32, uint16_t *bfloat16, int len);
    
    struct U8ReaderBuffer {
        uint8_t *cur;

        U8ReaderBuffer () {}

        U8ReaderBuffer (uint8_t *buffer) {
            this->cur = buffer;
        }

        long long ReadLongLong() {
            long long ret = ((long long*)this->cur)[0];
            this->cur += sizeof(long long);
            return ret;
        }

        int ReadInt() {
            int ret = ((int*)this->cur)[0];
            this->cur += sizeof(int);
            return ret;
        }

        float ReadFloat() {
            float ret = ((float*)this->cur)[0];
            this->cur += sizeof(float);
            return ret;
        }

        void ReadBytes(uint8_t *buffer, uint64_t bytes) {
            memcpy(buffer, cur, bytes);
            cur += bytes;
        }

        void Skip(uint64_t offset) {
            cur += offset;
        }
    };

    ComputeServer::ComputeServer(int partId, int partCnt, int threadNum) {
        this->partId = partId;
        this->partCnt = partCnt;
        this->threadNum = threadNum;

        SetThreads(threadNum);
        this->pool = GetAlivePool();

        // 获取共享内存段
        const char* shm_name = "/fastllm_shm";
        int shm_fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
        if (shm_fd == -1) {
            printf("err\n");
            exit(0);
        }

        if (ftruncate(shm_fd, DDRLEN) == -1) {
            printf("err\n");
            exit(0);
        }

        void* ptr = mmap(nullptr, DDRLEN, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (ptr == MAP_FAILED) {
            printf("err\n");
            exit(0);
        }
        char* data = static_cast<char*>(ptr);

        baseAddr = (volatile uint8_t*)data;
        baseOutputAddr = (volatile uint8_t*)baseAddr + OUTPUTOFFSET;
        flag = (volatile int*)(baseAddr + FLAGOFFSET + partId * PAGE * sizeof(int));

        this->inputBuffer.resize(DDRLEN);
        this->outputBuffer.resize(DDRLEN);
    }

    void ComputeServer::Start() {
        barrier();
        auto lastRunTime = std::chrono::system_clock::now();
        int parentId = getppid();
        while (true) {
            barrier();
            int taskType = *((volatile int32_t*)flag);
            if (taskType == 0) {
                auto duration = std::chrono::duration_cast<std::chrono::microseconds> (std::chrono::system_clock::now() - lastRunTime);
                double gap = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
                if (gap > 1) {
                    if (getppid() != parentId) {
                        parentId = getppid();
                        if (parentId == 1) {
                            printf("numa server %d exit.\n", this->partId);
                            exit(0);
                        }
                    }
                }

                if (gap > 3) {
                    sleep(0);
                }
                continue;
            }
            lastRunTime = std::chrono::system_clock::now();
            if (taskType == ComputeTaskType::LinearInt4NoZero ||
                taskType == ComputeTaskType::LinearInt4Group ||
                taskType == ComputeTaskType::LinearInt8) {
                RunLinearInt();
            } else if (taskType == ComputeTaskType::LinearFloat16 || taskType == ComputeTaskType::LinearFloat32 || taskType == ComputeTaskType::LinearFP8E4M3) {
                RunLinearFloat();
            } else if (taskType == ComputeTaskType::MOEInt4NoZero || taskType == ComputeTaskType::MOEInt4Group) {
                RunMOEInt();
            } else if (taskType == ComputeTaskType::MOEFP8E4M3) {
                RunMOEFloat();
            } else if (taskType == ComputeTaskType::AppendKVCache) {
                AppendKVCache();
            } else if (taskType == ComputeTaskType::DoAttention) {
                Attention();
            } else if (taskType == ComputeTaskType::GetComputeServerInfo) {
                SendComputeServerInfo();
            } else if (taskType == ComputeTaskType::FindData) {
                FindData();
            } else if (taskType == ComputeTaskType::StartLongData) {
                ReceiveLongData();
            } else if (taskType == ComputeTaskType::FinishLongData) {
                FinishLongData();
            }

            barrier();
            *((volatile int32_t*)flag) = 0;
            barrier();
        }
    }

    void ComputeServer::SendComputeServerInfo() {
        if (partId != 0) {
            return;
        }
        json11::Json ret = json11::Json::object {
            {"version", VERSION},
            {"numacnt", partCnt}
        };
        std::string retString = ret.dump();
        ((int32_t*)baseOutputAddr)[0] = (int)retString.size();
        memcpy((uint8_t*)baseOutputAddr + 4, retString.c_str(), retString.size());
    }

    void ComputeServer::ReceiveLongData() {
        int currentLen = ((int*)baseAddr)[0];
        uint64_t oldLen = this->longBuffer.size();
        this->longBuffer.resize(oldLen + currentLen);
        memcpy(this->longBuffer.data() + oldLen, (uint8_t*)baseAddr + 4, currentLen);
    }

    void ComputeServer::FinishLongData() {
        uint8_t *base = this->longBuffer.data();
        int configStringLen = ((int*)base)[0];
        std::string configString;
        for (int i = 0; i < configStringLen; i++) {
            configString += (char)base[4 + i];
        }
        // printf("config = %s\n", configString.c_str());

        json11::Json config;
        std::string error;
        config = json11::Json::parse(configString, error);
        const std::string &op = config["op"].string_value();
        if (op == "registerData") {
            RegisterData(&config, base + 4 + configStringLen);
        } else if (op == "unregisterData") {
            UnregisterData(&config);
        }

        this->longBuffer.clear();
    }

    // 矩阵膨胀
    // ori = [oriN, oriM]
    // ret = [oriN * factorN, oriM * factorM]
    template <typename T>
    void InflateMatrix (std::vector <T> &ori, std::vector <T> &ret, int oriN, int oriM, int factorN, int factorM) {
        int newN = oriN * factorN, newM = oriM * factorM;
        ret.resize(newN * newM);
        for (int i = 0; i < newN; i++) {
            for (int j = 0; j < newM; j++) {
                ret[i * newM + j] = ori[i / oriN * oriM + j / oriM];
            }
        }        
    }

    void ComputeServer::RegisterData(json11::Json *config, uint8_t *base) {
        std::string name = (*config)["dataName"].string_value();
        if (this->weights.weight.find(name) != this->weights.weight.end()) {
            return;
        }
        std::string weightType = (*config)["weightType"].string_value();

        auto &weight = this->weights.weight;
        U8ReaderBuffer buffer = U8ReaderBuffer(base);

        int dimsSize = buffer.ReadInt();
        //printf("size = %d\n", dimsSize);
        std::vector<int> dims;
        for (int j = 0; j < dimsSize; j++) {
            int x = buffer.ReadInt();
            dims.push_back(x);
            //printf("%d\n", x);
        }
        DataType dataType = (DataType) buffer.ReadInt();

        if (weightType == "linearColumn") {
            int k = dims[0], m = dims[1];
            int localM = m / partCnt;
            while (localM % 2 == 1) {
                localM++;
            }
            
            int ggmlType = -1;
            if (dataType == DataType::DATA_GGUF_FORMAT) {
                ggmlType = buffer.ReadInt();
                localM = localM / 32 * 32;
                if (localM % 32 != 0) {
                    printf("warning: localM %% 32 != 0.\n");
                }
            }

            int base = partId * localM;
            if (partId == partCnt - 1) {
                localM = m - partId * localM;
            }
            weight[name] = Data(dataType, ggmlType, {k, localM});
            weight[name].name = name;
            weight[name].Allocate();
            if (dataType == DataType::FLOAT16 || dataType == DataType::BFLOAT16) {
                for (int i = 0; i < k; i++) {
                    buffer.Skip(base * 2);
                    buffer.ReadBytes(weight[name].cpuData + i * localM * 2, localM * 2);
                    buffer.Skip((m - base - localM) * 2);
                }
            } else if (dataType == DataType::DATA_GGUF_FORMAT) {
                int skip0 = ggml_row_size((ggml_type)ggmlType, base);
                int len = ggml_row_size((ggml_type)ggmlType, localM);
                int skip1 = ggml_row_size((ggml_type)ggmlType, m - base - localM);
                for (int i = 0; i < k; i++) {
                    buffer.Skip(skip0);
                    buffer.ReadBytes(weight[name].cpuData + i * len, len);
                    buffer.Skip(skip1);
                }
            } else if (dataType == DataType::INT8) {
                int bit = 8;
                weight[name].perChannelAxis = buffer.ReadInt();
                int k = weight[name].perChannelAxis == -1 ? 1 : dims[weight[name].perChannelAxis];
                weight[name].perChannelsConfigs.resize(k);
                weight[name].zeros.resize(k);
                weight[name].scales.resize(k);
                for (int i = 0; i < k; i++) {
                    float minValue = buffer.ReadFloat();
                    float maxValue = buffer.ReadFloat();
                    weight[name].perChannelsConfigs[i] = LowBitConfig(minValue, maxValue, bit, 1);
                    weight[name].zeros[i] = weight[name].perChannelsConfigs[i].zeroPoint;
                    weight[name].scales[i] = weight[name].perChannelsConfigs[i].scale;
                }
                
                for (int i = 0; i < k; i++) {
                    buffer.Skip(base);
                    buffer.ReadBytes(weight[name].cpuData + i * localM, localM);
                    buffer.Skip(m - base - localM);
                }
            } else if (dataType == DataType::FP8_E4M3) {
                weight[name].blockK = buffer.ReadInt();
                weight[name].blockM = buffer.ReadInt();                
                std::vector <float> oriScales;
                oriScales.resize(buffer.ReadInt());
                buffer.ReadBytes((uint8_t*)oriScales.data(), (int)oriScales.size() * sizeof(float));
                
                int ks = (k - 1) / weight[name].blockK + 1;
                int ms = (localM - 1) / weight[name].blockM + 1;
                int oldMs = (m - 1) / weight[name].blockM + 1;
                weight[name].scales.resize(ks * ms);

                for (int i = 0; i < ks; i++) {
                    memcpy(weight[name].scales.data() + i * ms, 
                            oriScales.data() + i * oldMs + partId * ms, 
                            ms * sizeof(float));
                }

                for (int i = 0; i < k; i++) {
                    buffer.Skip(base);
                    buffer.ReadBytes(weight[name].cpuData + i * localM, localM);
                    buffer.Skip(m - base - localM);
                }
            } else if (dataType == DataType::INT4_NOZERO) {
                int bit = 4;
                weight[name].perChannelAxis = buffer.ReadInt();
                int k = weight[name].perChannelAxis == -1 ? 1 : dims[weight[name].perChannelAxis];
                weight[name].perChannelsConfigs.resize(k);
                weight[name].mins.resize(k);
                weight[name].scales.resize(k);
                for (int i = 0; i < k; i++) {
                    float minValue = buffer.ReadFloat();
                    float scale = buffer.ReadFloat();
                    weight[name].perChannelsConfigs[i] = LowBitConfig(minValue, minValue + 15 * scale, bit, 1);
                    weight[name].perChannelsConfigs[i].min = minValue;
                    weight[name].perChannelsConfigs[i].scale = scale;
                    weight[name].mins[i] = weight[name].perChannelsConfigs[i].min;
                    weight[name].scales[i] = weight[name].perChannelsConfigs[i].scale;
                }
                
                for (int i = 0; i < k; i++) {
                    buffer.Skip(base / 2);
                    buffer.ReadBytes(weight[name].cpuData + i * localM / 2, localM / 2);
                    buffer.Skip(m / 2 - base / 2 - localM / 2);
                }
            } else if (dataType == DataType::INT4_GROUP) {
                int bit = 4;
                weight[name].perChannelAxis = buffer.ReadInt();
                weight[name].group = buffer.ReadInt();
                weight[name].groupCnt = buffer.ReadInt();
                int k = weight[name].perChannelAxis == -1 ? 1 : dims[weight[name].perChannelAxis];
                int group = weight[name].group;
                // weight[name].perChannelsConfigs.resize(k);
                int curGroup = localM / weight[name].groupCnt;
                weight[name].group = curGroup;
                weight[name].mins.resize(k * curGroup);
                weight[name].scales.resize(k * curGroup);
                int groupBase = partId * localM / weight[name].groupCnt;
                for (int i = 0; i < k; i++) {
                    for (int g = 0; g < group; g++) {
                        float minValue = buffer.ReadFloat();
                        float scale = buffer.ReadFloat();
                        if (g - groupBase >= 0 && g - groupBase < curGroup) {
                            weight[name].mins[i * curGroup + g - groupBase] = minValue;
                            weight[name].scales[i * curGroup + g - groupBase] = scale;
                        }
                    }
                }
                
                for (int i = 0; i < k; i++) {
                    buffer.Skip(base / 2);
                    buffer.ReadBytes(weight[name].cpuData + i * localM / 2, localM / 2);
                    buffer.Skip(m / 2 - base / 2 - localM / 2);
                }
            } else {
                ErrorInFastLLM("Register LinearColumn Error: wrong data type");
            }
        } else {
            int ggmlType = -1;
            if (dataType == DataType::DATA_GGUF_FORMAT) {
                ggmlType = buffer.ReadInt();
            }
            auto curDims = dims;
            Data oriWeight = Data(dataType, ggmlType, curDims);
            int k = dims[0];
            int localK = k / partCnt;
            if (partId == partCnt - 1) {
                localK = k - partId * localK;
            }
            
            curDims[0] = localK;

            weight[name] = Data(dataType, ggmlType, curDims);
            weight[name].name = name;
            weight[name].Allocate();

            if (dataType == DataType::FLOAT32 || dataType == DataType::BFLOAT16 || dataType == DataType::FLOAT16) {
                int k = oriWeight.dims[0];
                int m = oriWeight.GetBytes() / k;
                int localK = k / partCnt;
                int base = partId * localK * m;
                if (partId == partCnt - 1) {
                    localK = k - partId * localK;
                }

                if (weightType == "linearSwiglu") {
                    buffer.Skip(partId * (localK / 2) * m);
                    buffer.ReadBytes(weight[name].cpuData, (localK / 2) * m);
                    buffer.Skip((k / 2 - localK / 2) * m);
                    buffer.ReadBytes(weight[name].cpuData+ (localK / 2) * m, (localK / 2) * m);
                    buffer.Skip(weight[name].GetBytes() - (partId * localK * m) - (k / 2 * m));
                } else {
                    buffer.Skip(base);
                    buffer.ReadBytes(weight[name].cpuData, m * localK);
                    buffer.Skip(weight[name].GetBytes() - (base + m * localK));
                }
            } else if (dataType == DataType::DATA_GGUF_FORMAT) {
                int k = oriWeight.dims[0];
                int m = ggml_row_size((ggml_type)ggmlType, oriWeight.dims[1]);

                int localK = k / partCnt;
                int base = partId * localK * m;
                if (partId == partCnt - 1) {
                    localK = k - partId * localK;
                }

                if (weightType == "linearSwiglu") {
                    buffer.Skip(partId * (localK / 2) * m);
                    buffer.ReadBytes(weight[name].cpuData, (localK / 2) * m);
                    buffer.Skip((k / 2 - localK / 2) * m);
                    buffer.ReadBytes(weight[name].cpuData+ (localK / 2) * m, (localK / 2) * m);
                    buffer.Skip(weight[name].GetBytes() - (partId * localK * m) - (k / 2 * m));
                } else {
                    buffer.Skip(base);
                    buffer.ReadBytes(weight[name].cpuData, m * localK);
                    buffer.Skip(weight[name].GetBytes() - (base + m * localK));
                }
            } else if (dataType == DataType::FP8_E4M3) {
                weight[name].blockK = buffer.ReadInt();
                weight[name].blockM = buffer.ReadInt();                
                std::vector <float> oriScales;
                oriScales.resize(buffer.ReadInt());
                buffer.ReadBytes((uint8_t*)oriScales.data(), (int)oriScales.size() * sizeof(float));
                
                int k = oriWeight.dims[0];
                int m = oriWeight.GetBytes() / k;
                int localK = k / partCnt;
                localK = ((localK - 1) / weight[name].blockK + 1) * weight[name].blockK;

                int base = partId * localK * m;
                if (partId == partCnt - 1) {
                    localK = k - partId * localK;
                }
                
                int ks = (localK - 1) / weight[name].blockK + 1;
                int ms = (m - 1) / weight[name].blockM + 1;
                weight[name].scales.resize(ks * ms);

                if (weightType == "linearSwiglu") {
                    buffer.Skip(partId * (localK / 2) * m);
                    buffer.ReadBytes(weight[name].cpuData, (localK / 2) * m);
                    buffer.Skip((k / 2 - localK / 2) * m);
                    buffer.ReadBytes(weight[name].cpuData+ (localK / 2) * m, (localK / 2) * m);
                    buffer.Skip(weight[name].GetBytes() - (partId * localK * m) - (k / 2 * m));

                    memcpy(weight[name].scales.data(), 
                        oriScales.data() + partId * (localK / 2) / weight[name].blockK * ms, 
                        (localK / 2 / weight[name].blockK) * ms * sizeof(float));
                    memcpy(weight[name].scales.data() + (localK / 2 / weight[name].blockK) * ms, 
                        oriScales.data() + (k / 2 + partId * (localK / 2)) / weight[name].blockK * ms, 
                        (localK / 2 / weight[name].blockK) * ms * sizeof(float));
                } else {
                    buffer.Skip(base);
                    buffer.ReadBytes(weight[name].cpuData, m * localK);
                    buffer.Skip(weight[name].GetBytes() - (base + m * localK));

                    memcpy(weight[name].scales.data(), oriScales.data() + base / m / weight[name].blockK * ms, ks * ms * sizeof(float));
                }
            } else if (dataType == DataType::INT8 || dataType == DataType::INT4) {
                int bit = (dataType == DataType::INT4 ? 4 : 8);
                weight[name].perChannelAxis = buffer.ReadInt();
                int k = weight[name].perChannelAxis == -1 ? 1 : dims[weight[name].perChannelAxis];
                weight[name].perChannelsConfigs.resize(k);
                weight[name].zeros.resize(k);
                weight[name].scales.resize(k);

                for (int i = 0; i < k; i++) {
                    float minValue = buffer.ReadFloat();
                    float maxValue = buffer.ReadFloat();
                    weight[name].perChannelsConfigs[i] = LowBitConfig(minValue, maxValue, bit, 0);
                    weight[name].zeros[i] = weight[name].perChannelsConfigs[i].zeroPoint;
                    weight[name].scales[i] = weight[name].perChannelsConfigs[i].scale;
                }

                int m = oriWeight.GetBytes() / k;
                int localK = k / partCnt;
                int base = partId * localK * m;
                if (partId == partCnt - 1) {
                    localK = k - partId * localK;
                }

                if (weightType == "linearSwiglu") {
                    std::vector <LowBitConfig*> configs;
                    configs.resize(localK);
                    for (int i = 0; i < localK / 2; i++) {
                        configs[i] = &weight[name].perChannelsConfigs[partId * (localK / 2) + i];
                        configs[i + localK / 2] = &weight[name].perChannelsConfigs[partId * (localK / 2) + k / 2 + i];
                    }
                    for (int i = 0; i < localK; i++) {
                        weight[name].perChannelsConfigs[i] = *configs[i];
                        weight[name].zeros[i] = weight[name].perChannelsConfigs[i].zeroPoint;
                        weight[name].scales[i] = weight[name].perChannelsConfigs[i].scale;
                    }

                    buffer.Skip(partId * (localK / 2) * m);
                    buffer.ReadBytes(weight[name].cpuData, (localK / 2) * m);
                    buffer.Skip((k / 2 - localK / 2) * m);
                    buffer.ReadBytes(weight[name].cpuData + (localK / 2) * m, (localK / 2) * m);
                    buffer.Skip(weight[name].GetBytes() - (partId * localK * m) - (k / 2 * m));
                } else {
                    for (int i = 0; i < localK; i++) {
                        weight[name].perChannelsConfigs[i] = weight[name].perChannelsConfigs[i + base / m];
                        weight[name].zeros[i] = weight[name].zeros[i + base / m];
                        weight[name].scales[i] = weight[name].scales[i + base / m];
                    }
                    
                    buffer.Skip(base);
                    buffer.ReadBytes(weight[name].cpuData, m * localK);
                    buffer.Skip(weight[name].GetBytes() - (base + m * localK));
                }
            } else if (dataType == DataType::INT4_NOZERO) {
                int bit = 4;
                weight[name].perChannelAxis = buffer.ReadInt();
                int k = weight[name].perChannelAxis == -1 ? 1 : dims[weight[name].perChannelAxis];
                weight[name].perChannelsConfigs.resize(k);
                weight[name].mins.resize(k);
                weight[name].scales.resize(k);
                for (int i = 0; i < k; i++) {
                    float minValue = buffer.ReadFloat();
                    float scale = buffer.ReadFloat();
                    weight[name].perChannelsConfigs[i] = LowBitConfig(minValue, minValue + 15 * scale, bit, 1);
                    weight[name].perChannelsConfigs[i].min = minValue;
                    weight[name].perChannelsConfigs[i].scale = scale;
                    weight[name].mins[i] = weight[name].perChannelsConfigs[i].min;
                    weight[name].scales[i] = weight[name].perChannelsConfigs[i].scale;
                }

                int m = oriWeight.GetBytes() / k;
                int localK = k / partCnt;
                int base = partId * localK * m;
                if (partId == partCnt - 1) {
                    localK = k - partId * localK;
                }

                if (weightType == "linearSwiglu") {
                    std::vector <LowBitConfig*> configs;
                    configs.resize(localK);
                    for (int i = 0; i < localK / 2; i++) {
                        configs[i] = &weight[name].perChannelsConfigs[partId * (localK / 2) + i];
                        configs[i + localK / 2] = &weight[name].perChannelsConfigs[partId * (localK / 2) + k / 2 + i];
                    }
                    for (int i = 0; i < localK; i++) {
                        weight[name].perChannelsConfigs[i] = *configs[i];
                        weight[name].mins[i] = weight[name].perChannelsConfigs[i].min;
                        weight[name].scales[i] = weight[name].perChannelsConfigs[i].scale;
                    }

                    buffer.Skip(partId * (localK / 2) * m);
                    buffer.ReadBytes(weight[name].cpuData, (localK / 2) * m);
                    buffer.Skip((k / 2 - localK / 2) * m);
                    buffer.ReadBytes(weight[name].cpuData + (localK / 2) * m, (localK / 2) * m);
                    buffer.Skip(weight[name].GetBytes() - (partId * localK * m) - (k / 2 * m));
                } else {
                    for (int i = 0; i < localK; i++) {
                        weight[name].perChannelsConfigs[i] = weight[name].perChannelsConfigs[i + base / m];
                        weight[name].mins[i] = weight[name].mins[i + base / m];
                        weight[name].scales[i] = weight[name].scales[i + base / m];
                    }

                    buffer.Skip(base);
                    buffer.ReadBytes(weight[name].cpuData, m * localK);
                    buffer.Skip(weight[name].GetBytes() - (base + m * localK));
                }
            } else if (dataType == DataType::INT4_GROUP) {
                auto &curWeight = weight[name];
                int bit = 4;
                curWeight.perChannelAxis = buffer.ReadInt();
                curWeight.group = buffer.ReadInt();
                curWeight.groupCnt = buffer.ReadInt();
                int k = curWeight.perChannelAxis == -1 ? 1 : dims[curWeight.perChannelAxis];
                k *= curWeight.group;
                curWeight.perChannelsConfigs.resize(k);
                curWeight.mins.resize(k);
                curWeight.scales.resize(k);
                for (int i = 0; i < k; i++) {
                    float minValue = buffer.ReadFloat();
                    float scale = buffer.ReadFloat();
                    auto config = LowBitConfig(minValue, minValue + scale * ((1 << bit) - 1), bit, 1);
                    config.min = minValue;
                    config.scale = scale;
                    curWeight.perChannelsConfigs[i] = config;
                    curWeight.mins[i] = config.min;
                    curWeight.scales[i] = config.scale;
                }

                int oldK = (k / curWeight.group);
                int m = oriWeight.GetBytes() / oldK;
                int localK = oldK / partCnt;
                int base = partId * localK * m;
                if (partId == partCnt - 1) {
                    localK = oldK - partId * localK;
                }

                if (weightType == "linearSwiglu") {
                    std::vector <LowBitConfig*> configs;
                    configs.resize(localK * curWeight.group);
                    for (int i = 0; i < localK / 2; i++) {
                        for (int j = 0; j < curWeight.group; j++) {
                            configs[i * curWeight.group + j] = &weight[name].perChannelsConfigs[(partId * (localK / 2) + i) * curWeight.group + j];
                            configs[(i + localK / 2) * curWeight.group + j] = &weight[name].perChannelsConfigs[(partId * (localK / 2) + oldK / 2 + i) * curWeight.group + j];
                        }
                    }
                    for (int i = 0; i < localK; i++) {
                        for (int j = 0; j < curWeight.group; j++) {
                            weight[name].perChannelsConfigs[(i) * curWeight.group + j] = *configs[i * curWeight.group + j];
                            weight[name].mins[(i) * curWeight.group + j] = configs[i * curWeight.group + j]->min;
                            weight[name].scales[(i) * curWeight.group + j] = configs[i * curWeight.group + j]->scale;
                        }
                    }

                    buffer.Skip(partId * (localK / 2) * m);
                    buffer.ReadBytes(weight[name].cpuData, (localK / 2) * m);
                    buffer.Skip((oldK / 2 - localK / 2) * m);
                    buffer.ReadBytes(weight[name].cpuData + (localK / 2) * m, (localK / 2) * m);
                    buffer.Skip(weight[name].GetBytes() - (partId * localK * m) - (oldK / 2 * m));
                } else {
                    for (int i = 0; i < localK; i++) {
                        for (int j = 0; j < curWeight.group; j++) {
                            weight[name].perChannelsConfigs[i * curWeight.group + j] = weight[name].perChannelsConfigs[(i + base / m) * curWeight.group + j];
                            weight[name].mins[i * curWeight.group + j] = weight[name].mins[(i + base / m) * curWeight.group + j];
                            weight[name].scales[i * curWeight.group + j] = weight[name].scales[(i + base / m) * curWeight.group + j];
                        }
                    }

                    buffer.Skip(base);
                    buffer.ReadBytes(weight[name].cpuData, m * localK);
                    buffer.Skip(weight[name].GetBytes() - (base + m * localK));
                }

                curWeight.perChannelsConfigs.resize(1);
                curWeight.perChannelsConfigs.shrink_to_fit();
            }
        }

        int weightId = this->weightsList.size();
        this->weightsList.push_back(&weight[name]);
        if (partId == 0) {
            // 传回weight id
            ((int32_t*)baseOutputAddr)[0] = weightId;
        }
    }

    void ComputeServer::UnregisterData(json11::Json *config) {
        std::string dataName = (*config)["dataName"].string_value();
        if (this->weights.weight.find(dataName) == this->weights.weight.end()) {
            return;
        }

        this->weights.weight.erase(dataName);
    }

    void ComputeServer::GetLinearIntBaseInfo(int &n, int &m, int &k, int &group, int &groupCnt,
                                          std::string &weightName, std::string &biasName,
                                          std::vector <LowBitConfig> &configs,
                                          LinearExType &exType, DataType &outputDataType,
                                          AliveThreadPool *pool) {
        n = ((volatile int32_t*)baseAddr)[0];
        m = ((volatile int32_t*)baseAddr)[1];
        k = ((volatile int32_t*)baseAddr)[2];
        group = ((volatile int32_t*)baseAddr)[3];
        groupCnt = ((volatile int32_t*)baseAddr)[4];

        int weightNameLen = ((volatile int32_t*)baseAddr)[5];
        int biasNameLen = ((volatile int32_t*)baseAddr)[6];

        exType = (LinearExType)((volatile int32_t*)baseAddr)[7];
        outputDataType = (DataType)((volatile int32_t*)baseAddr)[8];

        volatile char *buffer = (volatile char*)baseAddr + 10 * sizeof(int32_t);
        configs.clear();
        for (int i = 0; i < n * group; i++) {
            configs.push_back(fastllm::LowBitConfig(((float*)buffer)[0], ((float*)buffer)[1], 8, 0));
            buffer += 2 * sizeof(float);
        }

        weightName = biasName = "";
        for (int i = 0; i < weightNameLen; i++) {
            weightName += *buffer;
            buffer++;
        }
        for (int i = 0; i < biasNameLen; i++) {
            biasName += *buffer;
            buffer++;
        }

        RunMultiThreadMemcpy(this->inputBuffer.data(), (uint8_t*)buffer, n * m, pool);
         // memcpy(this->inputBuffer.data(), (uint8_t*)buffer, n * m);
    }

    void DoFloat32LinearExOp(LinearExType exType, float *localOutput, float *baseOutputAddr,
                             int n, int k, int localK, int base, AliveThreadPool *pool) {
        if (exType == LinearExType::ExSwiglu) {
            int mid = (localK / 2);
            SwigluMultiThread((float *) localOutput, mid, mid, ((float *) baseOutputAddr) + base / 2,
                              n, localK, k / 2, pool);
            return;
        } else if (exType == LinearExType::ExGelu) {
            GeluMultiThread((float*)localOutput, localK, ((float*)baseOutputAddr) + base, n, localK, k, pool);
            return;
        } else if (exType == LinearExType::ExSilu) {
            SiluMultiThread((float*)localOutput, localK, ((float*)baseOutputAddr) + base, n, localK, k, pool);
            return;
        }

        for (int i = 0; i < n; i++) {
            memcpy((uint8_t *) baseOutputAddr + (i * k + base) * sizeof(float),
                   (uint8_t *) localOutput + i * localK * sizeof(float),
                   localK * sizeof(float));
        }
    }

    // 输出到Float16
    void DoFloat16LinearExOp(LinearExType exType, float *localOutput, uint16_t *baseOutputAddr,
                             int n, int k, int localK, int base, AliveThreadPool *pool) {
        std::vector <float> tempOutput;
        tempOutput.resize(n * localK);
        if (exType == LinearExType::ExSwiglu) {
            int mid = (localK / 2);
            SwigluMultiThread((float *) localOutput, mid, mid, ((float *) tempOutput.data()),
                              n, localK, localK / 2, pool);
            for (int i = 0; i < n; i++) {
                float *floatOutput = (float*)(tempOutput.data()) + i * (localK / 2);
                uint16_t *halfOutput = baseOutputAddr + (i * k + base) / 2;
                Float32ToFloat16(floatOutput, halfOutput, localK / 2);
            }
            return;
        } else if (exType == LinearExType::ExGelu) {
            GeluMultiThread((float*)localOutput, localK, ((float*)tempOutput.data()), n, localK, localK, pool);
            return;
        } else if (exType == LinearExType::ExSilu) {
            SiluMultiThread((float*)localOutput, localK, ((float*)tempOutput.data()), n, localK, localK, pool);
            return;
        }

        for (int i = 0; i < n; i++) {
            float *floatOutput = localOutput + (i * localK);
            uint16_t *halfOutput = baseOutputAddr + (i * k + base);
            Float32ToFloat16(floatOutput, halfOutput, localK);            
        }
    }

    void GetInputSums(std::vector <float> &inputSums, uint8_t *input, int n, int m, int group, int groupCnt, DataType weightType) {
        if (weightType == DataType::INT8) {
            // for int8 * int8
            for (int i = 0; i < n; i++) {
                for (int g = 0; g < group; g++) {
                    int sum = 0;
                    for (int j = g * groupCnt; j < (g + 1) * groupCnt && j < m; j++) {
                        sum += input[i * m + j] ^ 128;
                    }
                    inputSums.push_back(sum);
                }
            }
        } else {
            // for int8 * int4
            for (int i = 0; i < n; i++) {
                for (int g = 0; g < group; g++) {
                    int sum = 0;
                    for (int j = g * groupCnt; j < (g + 1) * groupCnt && j < m; j++) {
                        sum += input[i * m + j];
                    }
                    inputSums.push_back(sum);
                }
            }
        }
    }

    void ComputeServer::RunLinearInt() {
        int n, m, k, group, groupCnt;
        std::string weightName, biasName;
        std::vector <fastllm::LowBitConfig> configs;
        LinearExType exType;
        DataType outputDataType;
        GetLinearIntBaseInfo(n, m, k, group, groupCnt, weightName, biasName, configs, exType, outputDataType, pool);

        uint8_t *localInput = this->inputBuffer.data();
        int32_t *localOutput = (int32_t*)this->outputBuffer.data();
        auto &w = weights[weightName];
        auto &bias = weights[biasName];
        fastllm::DataType wType = w.dataType;
        uint8_t *weight = w.cpuData;
        w.CalcWeightSum();

        int localK = k / partCnt;
        int base = partId * localK;
        if (partId == partCnt - 1) {
            localK = k - partId * localK;
        }

        float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
        bool finish = false;
#ifdef USE_TFACCX
        if (wType == DataType::INT8 && n > 32 && localK % 32 == 0) {
            finish = true;
            TfaccMultiplyInt8(localInput, weight, localOutput,
                              n, m, localK,
                              w.weightSum.data(), w.zeros.data(), w.scales.data(), biasData,
                              configs, &w, pool);
        }
        if (wType == DataType::INT4_NOZERO && n > 32 && localK % 32 == 0) {
            finish = true;
            TfaccMultiplyInt4NoZero(localInput, weight, localOutput,
                              n, m, localK,
                              w.weightSum.data(), w.mins.data(), w.scales.data(), biasData,
                              configs, &w, pool);
        }
        if (wType == DataType::INT4_GROUP && n > 32 && localK % 32 == 0) {
            finish = true;
            TfaccMultiplyInt4Group(localInput, weight , localOutput,
                    n, m, localK,
                    w.weightSum.data(), w.mins.data(),
                    w.scales.data(), biasData,
                    configs, group, groupCnt, &w, pool);
        }
#endif

        if (!finish) {
            if (group < 0 || groupCnt < 0) {
                group = 1;
                groupCnt = m;
            }
            std::vector <float> inputSums;
            GetInputSums(inputSums, localInput, n, m, group, groupCnt, wType);
            
            std::vector <float> iscales, izeros;
            for (int i = 0; i < configs.size(); i++) {
                iscales.push_back(configs[i].scale);
                izeros.push_back(configs[i].zeroPoint);
            }

            float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
            if (wType == fastllm::DataType::INT4_NOZERO) {
                fastllm::RunLinearInt8Int4Group(localInput, weight, (float*)localOutput, 
                        n, m, localK, 1, m, w.weightSum.data(), w.mins.data(), w.scales.data(), biasData, 
                        inputSums.data(), iscales.data(), izeros.data(), pool, 0, pool->threads.size());
            } else if (wType == fastllm::DataType::INT4_GROUP) {
                fastllm::RunLinearInt8Int4Group(localInput, weight, (float*)localOutput, 
                    n, m, localK, group, groupCnt, 
                    w.weightSum.data(), w.mins.data(), w.scales.data(), biasData, 
                    inputSums.data(), iscales.data(), izeros.data(), pool, 0, pool->threads.size());
            } else if (wType == fastllm::DataType::INT8) {
                fastllm::RunLinearInt8Int8(localInput, weight, (float*)localOutput, n, m, localK, 
                    w.weightSum.data(), w.zeros.data(), w.scales.data(), biasData,
                    inputSums.data(), iscales.data(), izeros.data(), pool, 0, pool->threads.size());
            }
        }

        if (outputDataType == DataType::FLOAT32) {
            DoFloat32LinearExOp(exType, (float*)localOutput, (float*)baseOutputAddr, n, k, localK, base, pool);
        } else {
            DoFloat16LinearExOp(exType, (float*)localOutput, (uint16_t*)baseOutputAddr, n, k, localK, base, pool);
        }
    }

    void ComputeServer::GetLinearFloatBaseInfo(int &n, int &m, int &k,
                                               std::string &weightName, std::string &biasName,
                                               DataType &dataType,
                                               LinearExType &exType) {
        n = ((volatile int32_t*)baseAddr)[0];
        m = ((volatile int32_t*)baseAddr)[1];
        k = ((volatile int32_t*)baseAddr)[2];

        int weightNameLen = ((volatile int32_t*)baseAddr)[5];
        int biasNameLen = ((volatile int32_t*)baseAddr)[6];

        exType = (LinearExType)((volatile int32_t*)baseAddr)[7];
        dataType = (DataType)((volatile int32_t*)baseAddr)[8];

        volatile char *buffer = (volatile char*)baseAddr + 10 * sizeof(int32_t);
        weightName = biasName = "";
        for (int i = 0; i < weightNameLen; i++) {
            weightName += *buffer;
            buffer++;
        }
        for (int i = 0; i < biasNameLen; i++) {
            biasName += *buffer;
            buffer++;
        }
        int unitSize = 4;
        if (dataType == DataType::FLOAT16) {
            unitSize = 2;
        } else if (dataType == DataType::FLOAT32) {
            unitSize = 4;
        }

        memcpy(this->inputBuffer.data(), (uint8_t*)buffer, n * m * unitSize);
    }

    void ComputeServer::RunLinearFloat() {
        int n, m, k;
        std::string weightName, biasName;
        LinearExType exType;
        DataType dataType;
        GetLinearFloatBaseInfo(n, m, k, weightName, biasName, dataType, exType);

        float *localInput = (float *) this->inputBuffer.data();
        float *localOutput = (float *) this->outputBuffer.data();
        auto &w = weights[weightName];
        auto &bias = weights[biasName];
        fastllm::DataType wType = w.dataType;
        uint8_t *weight = w.cpuData;

        int localK = k / partCnt;
        int base = partId * localK;
        if (partId == partCnt - 1) {
            localK = k - partId * localK;
        }

        float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
        if (dataType == fastllm::DataType::FLOAT32 && wType == fastllm::DataType::FLOAT16) {
            fastllm::RunLinearFloat32Float16(localInput, ((uint16_t *)weight), localOutput, biasData, n, m, localK, pool, 0, pool->threads.size());
        } else if (dataType == fastllm::DataType::FLOAT32 && wType == fastllm::DataType::FLOAT32) {
            fastllm::RunLinearFloat32Float32(localInput, ((float*)weight), localOutput, biasData, n, m, localK, pool, 0, pool->threads.size());
        } else if (dataType == fastllm::DataType::FLOAT32 && wType == fastllm::DataType::FP8_E4M3) {
            fastllm::RunLinearFloat32FP8E4M3(localInput, w, localOutput, biasData, n, m, localK, pool, 0, pool->threads.size());
        } else if (dataType == fastllm::DataType::FLOAT32 && wType == fastllm::DataType::DATA_GGUF_FORMAT) {
            fastllm::RunLinearFloat32GGUF(localInput, ((uint8_t *)weight), localOutput, biasData, &w, n, m, localK, pool, 0, pool->threads.size());
        } else {
            printf("RunLinearFloat: wrong data type: dataType = %d, wType = %d.", dataType, wType);
        }

        DoFloat32LinearExOp(exType, (float*)localOutput, (float*)baseOutputAddr, n, k, localK, base, pool);
    }

    void ComputeServer::GetMOEIntBaseInfo(int &n, int &m, int &k, int &group, int &groupCnt,
                                          std::string &weightName, std::string &biasName,
                                          std::vector <LowBitConfig> &configs,
                                          LinearExType &exType, DataType &outputDataType,
                                          AliveThreadPool *pool) {
    }

    struct MOEVarManager {
        Data oriInput, tempInput, w1, w2, w3;
        std::vector <float> tempResult, middleResult;
    };
    MOEVarManager moeVarManager;

    struct moeIntSingleVarManagerServer {
        std::vector<LowBitConfig> inputConfigs;
        std::vector<uint8_t> uinput;
        std::vector <float> inputSums;
        std::vector <float> iscales, izeros;
        std::vector <std::vector <float> > middles, results;
        std::vector <std::vector <LowBitConfig> > inputConfigsDown;
        std::vector <std::vector <uint8_t> > uinputsDown;
        std::vector <std::vector <float> > inputSumsDown;
        std::vector <std::vector <float> > iscalesDown, izerosDown;
        std::vector <int> localKs;
        std::vector <float*> tempResults;
    } moeIntSingleVarManagerServer;

    struct moeFloatSingleVarManagerServer {
        std::vector <std::vector <float> > middles, swigluResults, results;
        std::vector <int> localKs;
        std::vector <float*> tempResults;
        std::vector <uint16_t> bf16Input;
    } moeFloatSingleVarManagerServer;

    void ComputeServer::RunMOEInt() {
 // auto st = std::chrono::system_clock::now();
 // auto ttt = std::chrono::system_clock::now();
 // std::vector <std::pair <std::string, float> > record;
        if (((int*)this->baseAddr)[0] > 1) {
            int configStringLen = ((int*)this->baseAddr)[0];
            std::string configString;
            for (int i = 0; i < configStringLen; i++) {
                configString += (char)this->baseAddr[4 + i];
            }
            json11::Json config;
            std::string error;
            config = json11::Json::parse(configString, error);
            int n = config["n"].int_value(), m = config["m"].int_value(), k = config["k"].int_value();
            int group = config["group"].int_value(), groupCnt = config["groupCnt"].int_value();
            int outputType = config["outputType"].int_value();
            uint8_t *localInput = (uint8_t *) this->inputBuffer.data();
            float *localOutput = (float *) this->outputBuffer.data();

            std::vector <LowBitConfig> inputConfigs;
            std::vector <float> iscales, izeros;
            volatile char *buffer = (volatile char*)this->baseAddr + 4 + configStringLen;
            for (int i = 0; i < n * group; i++) {
                inputConfigs.push_back(fastllm::LowBitConfig(((float*)buffer)[0], ((float*)buffer)[1], 8, 0));
                iscales.push_back(inputConfigs.back().scale);
                izeros.push_back(inputConfigs.back().zeroPoint);
                buffer += 2 * sizeof(float);
            }
            RunMultiThreadMemcpy(this->inputBuffer.data(), (uint8_t*)buffer, n * m, pool);
    
// printf("server prepare spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
            std::vector <std::vector <fastllm::Data*> > weights;
            std::vector <std::vector <float> > v;
            weights.resize(n);
            v.resize(n);
            int idx = 0;
            for (auto &factor : config["factors"].array_items()) {
                for (auto &it : factor.array_items()) {
                    v[idx].push_back(it.number_value());
                }
                idx++;
            }
            idx = 0;
            for (auto &weight : config["weights"].array_items()) {
                for (auto &it : weight.array_items()) {
                    weights[idx].push_back(&this->weights[it.string_value()]);
                }
                idx++;
            }
// printf("server get meta spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
            int bs = n, dim = k;
            int inputDim = m;
            std::map <std::pair <Data*, Data*>, std::vector <std::pair <int, float> > > expertTasks; // expertTasks[x]代表专家x的task, expertTasks[x][j] = (第j个任务对应的行数， 权重)            
// printf("server get middle spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
            Data &oriInput = moeVarManager.oriInput;
            Data &tempInput = moeVarManager.tempInput;
            Data &w1 = moeVarManager.w1;
            Data &w2 = moeVarManager.w2;
            Data &w3 = moeVarManager.w3;

            std::vector <float> &tempResult = moeVarManager.tempResult;
            std::vector <float> &middleResult = moeVarManager.middleResult;

            tempResult.resize(bs * dim);
            std::fill(tempResult.begin(), tempResult.begin() + bs * dim, 0.0f);
            middleResult.resize(bs * dim);
            std::fill(middleResult.begin(), middleResult.begin() + bs * dim, 0.0f);

            oriInput.dataType = DataType::FLOAT32;
            oriInput.Resize({n, m});
            oriInput.Allocate();
// printf("server get oriInput spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
            float *floatInput = (float*)oriInput.cpuData;
// printf("server revert spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
            for (int i = 0; i < n; i++) {
                for (int g = 0; g < group; g++) {
                    auto &config = inputConfigs[i * group + g];
                    for (int j = g * groupCnt; j < (g + 1) * groupCnt && j < m; j++) {
                        floatInput[i * m + j] = config.invQuantization(localInput[i * m + j]);
                    }
                }
            }
// printf("server get input spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
            tempInput.dataType = DataType::FLOAT32;
            tempInput.Resize({n, m});
            tempInput.Allocate();

            for (int b = 0; b < bs; b++) {
                for (int j = 0; j < v[b].size(); j++) {
                    expertTasks[std::make_pair(weights[b][j * 2], weights[b][j * 2 + 1])].push_back(std::make_pair(b, v[b][j]));
                }
            }
// printf("server ready spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
            for (auto &it : expertTasks) {
                auto &task = it.second;
                if (task.size() == 0) {
                    continue;
                }

                tempInput.Resize({(int)task.size(), inputDim});
                std::vector <MultiThreadMemcpyMultiLinesTask> memcpyTasks;
                for (int i = 0; i < (int)task.size(); i++) {
                    memcpyTasks.push_back(MultiThreadMemcpyMultiLinesTask(tempInput.cpuData + i * inputDim * oriInput.unitSize, oriInput.cpuData + task[i].first * inputDim * oriInput.unitSize, inputDim * oriInput.unitSize));
                }
                RunMultiThreadMemcpyMultiLines(memcpyTasks, pool);
                
                DoCpuLinearReshape(tempInput, *it.first.first, w3);
                DoCpuLinear(tempInput, *it.first.first, Data(), w3);

                int mid = w3.dims[1] / 2;
                w1.Resize({w3.dims[0], mid});
                w1.dataType = w3.dataType;
                w1.Allocate();
                SwigluMultiThread((float *) w3.cpuData, mid, mid, ((float *) w1.cpuData),
                                    w3.dims[0], w3.dims[1], mid, GetAlivePool());
                
                DoCpuLinearReshape(w1, *it.first.second, w2);
                DoCpuLinear(w1, *it.first.second, Data(), w2);

                float *curOutput;
                if (w2.dataType == DataType::FLOAT32) {
                    curOutput = (float*)w2.cpuData;
                } else if (w2.dataType == DataType::FLOAT16) {
                    Float16ToFloat32((uint16_t*)w2.cpuData, middleResult.data(), w2.Count(0));
                    curOutput = middleResult.data();
                }

                for (int i = 0; i < (int)task.size(); i++) {
                    float value = task[i].second;
                    float *lastResult = tempResult.data() + task[i].first * dim;
                    float *curResult = curOutput + i * dim;
                    for (int j = 0; j < dim; j++) {
                        lastResult[j] += value * curResult[j];
                    }
                }
            }
// printf("server calc spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
            for (int i = 0; i < n; i++) {
                memcpy((uint8_t *) baseOutputAddr + partId * n * k * sizeof(float) + (i * k) * sizeof(float),
                    ((uint8_t *) tempResult.data()) + (i * k) * sizeof(float),
                    k * sizeof(float));
            }
//printf("server write spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
            // if (output.dataType == DataType::FLOAT32) {
                // memcpy(output.cpuData, tempResult.data(), output.GetBytes());
            // } else if (output.dataType == DataType::FLOAT16) {
               // Float32ToFloat16(tempResult.data(), (uint16_t*)output.cpuData, output.Count(0));
            // }
            return;    
        } else {
            volatile char *buffer = (volatile char*)this->baseAddr;
            int n = ((int*)buffer)[0];
            int m = ((int*)buffer)[1];
            int k = ((int*)buffer)[2];
            int group = ((int*)buffer)[3];
            int groupCnt = ((int*)buffer)[4];
            buffer += 5 * sizeof(int);
            std::vector <fastllm::Data*> weights;
            std::vector <float> v;
// record.push_back(std::make_pair("before get weight", GetSpan(ttt, std::chrono::system_clock::now())));
            int vSize = ((int*)buffer)[0];
            buffer += sizeof(int);
            for (int i = 0; i < vSize; i++) {
                v.push_back(((float*)buffer)[0]);
                buffer += sizeof(float);
            }
            int wSize = ((int*)buffer)[0];
            buffer += sizeof(int);
            for (int i = 0; i < wSize; i++) {
                weights.push_back(this->weightsList[((int*)buffer)[0]]);
                buffer += sizeof(int);
            }

            uint8_t *localInput = (uint8_t *) this->inputBuffer.data();
            float *localOutput = (float *) this->outputBuffer.data();

            std::vector <LowBitConfig> &inputConfigs = moeIntSingleVarManagerServer.inputConfigs;
            std::vector <float> &iscales = moeIntSingleVarManagerServer.iscales;
            std::vector <float> &izeros = moeIntSingleVarManagerServer.izeros;
            inputConfigs.resize(n * group);
            iscales.resize(n * group);
            izeros.resize(n * group);
            for (int i = 0; i < n * group; i++) {
                inputConfigs[i] = (fastllm::LowBitConfig(((float*)buffer)[0], ((float*)buffer)[1], 8, 0));
                iscales[i] = (inputConfigs[i].scale);
                izeros[i] = (inputConfigs[i].zeroPoint);
                buffer += 2 * sizeof(float);
            }
// record.push_back(std::make_pair("get config", GetSpan(ttt, std::chrono::system_clock::now())));
            RunMultiThreadMemcpy(this->inputBuffer.data(), (uint8_t*)buffer, n * m, pool);
/*
            for (auto &factor : config["factors"].array_items()) {
                v.push_back(factor.number_value());
            }
            for (auto &weight : config["weights"].array_items()) {
                // weights.push_back(&this->weights[weight.string_value()]);
                weights.push_back(this->weightsList[weight.int_value()]);
            }
*/
            std::vector <float> &inputSums = moeIntSingleVarManagerServer.inputSums;
            inputSums.clear();
            GetInputSums(inputSums, localInput, n, m, group, groupCnt, weights[0]->dataType);
            int permuteType = 1;
            if (weights[0]->dataType == DataType::INT8) {
                permuteType = 0;
            }

 // record.push_back(std::make_pair("get weight", GetSpan(ttt, std::chrono::system_clock::now())));
            std::vector <int> &localKs = moeIntSingleVarManagerServer.localKs;
            std::vector <std::vector <float> > &middles = moeIntSingleVarManagerServer.middles;
            std::vector <std::vector <float> > &results = moeIntSingleVarManagerServer.results;
            middles.resize(v.size());
            results.resize(v.size());
            localKs.resize(v.size());
            for (int j = 0; j < v.size(); j++) {
                int idx = j;
                weights[idx * 2]->CalcWeightSum();
                weights[idx * 2 + 1]->CalcWeightSum();

                int localK = weights[idx * 2]->dims[0];
                localKs[j] = (localK);
                middles[j].resize(localK);
                results[j].resize(weights[idx * 2 + 1]->dims[0]);
            }

            std::vector<fastllm::MultiThreadBaseOp*> ops;
            int threads = pool->threads.size();
            ops.resize(threads);

            std::vector <std::vector <LowBitConfig> > &inputConfigsDown = moeIntSingleVarManagerServer.inputConfigsDown;
            std::vector <std::vector <uint8_t> > &uinputsDown = moeIntSingleVarManagerServer.uinputsDown;
            std::vector <std::vector <float> > &inputSumsDown = moeIntSingleVarManagerServer.inputSumsDown;
            std::vector <std::vector <float> > &iscalesDown = moeIntSingleVarManagerServer.iscalesDown;
            std::vector <std::vector <float> > &izerosDown = moeIntSingleVarManagerServer.izerosDown;
            inputConfigsDown.resize(v.size());
            uinputsDown.resize(v.size());
            inputSumsDown.resize(v.size());
            iscalesDown.resize(v.size());
            izerosDown.resize(v.size());
 // record.push_back(std::make_pair("prepare datas", GetSpan(ttt, std::chrono::system_clock::now())));
            for (int st = 0; st < v.size(); st++) {
                int k = localKs[st];
                int end = st, selSum = 1; // 一共处理selSum * k个输出

                int curSum = 1;
                for (int l = st + 1; l < v.size(); l++) {
                    int curK = localKs[l];
                    if (curK % k != 0) {
                        break;
                    }
                    curSum += (curK / k);
                    if (threads % curSum == 0) {
                        end = l;
                        selSum = curSum;
                    }
                }
                int base = threads / selSum;
                int threadSt = 0;
                for (int l = st; l <= end; l++) {
                    int idx = l;
                    Data *weight = weights[idx * 2];
                    uint8_t *weightData = (uint8_t *) weight->cpuData;
                    float *outputData = middles[l].data();
                    float *biasData = nullptr;
                    int curK = localKs[l];
                    int curThread = (curK / k) * base;
                    if (weight->dataType == DataType::INT8) {
                        LaunchLinearInt8Int8(localInput, weightData, outputData, n, m, curK,
                                            weight->weightSum.data(), weight->zeros.data(), weight->scales.data(), biasData, 
                                            inputSums.data(), iscales.data(), izeros.data(), 
                                            ops, pool, threadSt, curThread);
                    } else {
                        MultiplyInt4GroupMultiThreadLaunch(localInput, weightData, outputData, n, m, curK,
                                                weight->weightSum.data(), weight->mins.data(), weight->scales.data(), 
                                                biasData, inputSums, iscales, izeros,
                                                inputConfigs, threadSt, curThread, group, groupCnt, ops, pool);
                    }
                    threadSt += curThread;
                }

                for (int j = 0; j < ops.size(); j++) {
                    pool->Wait(j);
                    delete ops[j];
                }
 // record.push_back(std::make_pair("mul 0", GetSpan(ttt, std::chrono::system_clock::now())));
                // swiglu
                for (int l = st; l <= end; l++) {
                    int idx = l;
                    int spatial = localKs[idx], mid = spatial / 2;
                    float *outputData = middles[l].data();
                    int curK = localKs[idx];

                    ops[l - st] = new fastllm::MultiThreadMultiOps();
                    ((fastllm::MultiThreadMultiOps*)ops[l - st])->ops.push_back(new fastllm::MultiThreadSwigluOp(outputData, mid, mid, outputData, 1, spatial, spatial));
                    Data *weightDown = weights[idx * 2 + 1];
                    int groupDown = weightDown->group, groupCntDown = weightDown->groupCnt;
                    if (weightDown->dataType != DataType::INT4_GROUP) {
                        groupDown = 1;
                        groupCntDown = mid;
                    }
                    auto &inputConfigs = inputConfigsDown[l];
                    auto &inputSums = inputSumsDown[l];
                    auto &iscales = iscalesDown[l];
                    auto &izeros = izerosDown[l];
                    auto &uinputDown = uinputsDown[l];
                    inputConfigs.resize(n * groupDown);
                    uinputDown.resize(n * mid);   
                    inputSums.resize(n * groupDown);
                    iscales.resize(n * groupDown);
                    izeros.resize(n * groupDown);
                    ((fastllm::MultiThreadMultiOps*)ops[l - st])->ops.push_back(new MultiThreadOnlineQuantizationOp(
                                    middles[l].data(), uinputDown.data(), inputConfigs.data(),
                                    n, mid, groupDown, groupCntDown,
                                    inputSums.data(), iscales.data(), izeros.data(), permuteType));
                    pool->PushOp(l - st, ops[l - st]);
                }
                for (int l = st; l <= end; l++) {
                    pool->Wait(l - st);
                    delete ops[l - st];
                }
 // record.push_back(std::make_pair("swiglu", GetSpan(ttt, std::chrono::system_clock::now())));
// record.push_back(std::make_pair("quant", GetSpan(ttt, std::chrono::system_clock::now())));
                threadSt = 0;
                for (int l = st; l <= end; l++) {
                    int idx = l;
                    int mid = localKs[idx] / 2;
                    int curK = localKs[idx];
                    Data *weightDown = weights[idx * 2 + 1];
                    int groupDown = weightDown->group, groupCntDown = weightDown->groupCnt;
                    auto &inputConfigs = inputConfigsDown[l];
                    auto &inputSums = inputSumsDown[l];
                    auto &iscales = iscalesDown[l];
                    auto &izeros = izerosDown[l];
                    auto &uinputDown = uinputsDown[l];
                    int curThread = (curK / k) * base;
                    if (weightDown->dataType != DataType::INT4_GROUP) {
                        groupDown = 1;
                        groupCntDown = mid;
                    }

                    if (weightDown->dataType == DataType::INT8) {
                        LaunchLinearInt8Int8(uinputDown.data(), (uint8_t*)weightDown->cpuData, results[l].data(), 1, mid, m,
                                                weightDown->weightSum.data(), weightDown->zeros.data(), weightDown->scales.data(), nullptr, 
                                                inputSums.data(), iscales.data(), izeros.data(),
                                                ops, pool, threadSt, curThread);
                    } else {
                        MultiplyInt4GroupMultiThreadLaunch(uinputDown.data(), (uint8_t*)weightDown->cpuData, results[l].data(), 1, mid, m,
                                                weightDown->weightSum.data(), weightDown->mins.data(), weightDown->scales.data(), nullptr, 
                                                inputSums, iscales, izeros,
                                                inputConfigs, threadSt, curThread, groupDown, groupCntDown, ops, pool);
                    }
                    threadSt += curThread;               
                }

                for (int j = 0; j < ops.size(); j++) {
                    pool->Wait(j);
                    delete ops[j];
                }
                st = end;
 // record.push_back(std::make_pair("mul 1", GetSpan(ttt, std::chrono::system_clock::now())));
            }
            memset(localOutput, 0, m * sizeof(float));

            std::vector <float*> &tempResults = moeIntSingleVarManagerServer.tempResults;
            tempResults.resize(results.size());
            for (int i = 0; i < results.size(); i++) {
                tempResults[i] = results[i].data();
            }
            RunMultiThreadReduce (
                (int)v.size(), tempResults.data(), v.data(), localOutput, 
                (float*)baseOutputAddr + partId * n * k, m, pool
            );
/*
            for (int j = 0; j < v.size(); j++) {
                float value = v[j];
                float *curOutput = (float*)results[j];

                int i = 0;
    #ifdef __AVX2__
                __m256 value_vec = _mm256_set1_ps(value);
                // 每次处理 8 个浮点数（AVX2 寄存器可以容纳 8 个 float）
                for (; i <= m - 8; i += 8) {
                    // 加载 curOutput 的 8 个浮点数
                    __m256 curOutput_vec = _mm256_loadu_ps(&curOutput[i]);

                    // 加载 localOutput 的 8 个浮点数
                    __m256 fLastOutput_vec = _mm256_loadu_ps(&localOutput[i]);

                    // 计算 curOutput * value
                    __m256 result_vec = _mm256_mul_ps(curOutput_vec, value_vec);

                    // 累加到 fLastOutput
                    fLastOutput_vec = _mm256_add_ps(fLastOutput_vec, result_vec);

                    // 将结果存回 fLastOutput
                    _mm256_storeu_ps(&localOutput[i], fLastOutput_vec);
                }
    #endif
                for (int k = i; k < m; k++) {
                    localOutput[k] += curOutput[k] * value;
                }

            }
// record.push_back(std::make_pair("get fp32 sum", GetSpan(ttt, std::chrono::system_clock::now())));
            for (int i = 0; i < n; i++) {
                RunMultiThreadMemcpy((uint8_t *) baseOutputAddr + partId * n * k * sizeof(float) + (i * k) * sizeof(float),
                    (uint8_t *) localOutput + (i * k) * sizeof(float),
                    k * sizeof(float), pool, true);
            }
*/

 // record.push_back(std::make_pair("copy output", GetSpan(ttt, std::chrono::system_clock::now())));
 // for (int i = 0; i < record.size(); i++) {
    //  printf("server %s spend %f s.\n", record[i].first.c_str(), record[i].second);
 // }
        }
    }

    void ComputeServer::RunMOEFloat() {
// auto ttt = std::chrono::system_clock::now();
// std::vector <std::pair <std::string, float> > record;
        if (((int*)this->baseAddr)[0] > 1) {
            int configStringLen = ((int*)this->baseAddr)[0];
            std::string configString;
            for (int i = 0; i < configStringLen; i++) {
                configString += (char)this->baseAddr[4 + i];
            }
            json11::Json config;
            std::string error;
            config = json11::Json::parse(configString, error);
            int n = config["n"].int_value(), m = config["m"].int_value(), k = config["k"].int_value();
            int outputType = config["outputType"].int_value();
            float *localInput = (float *) this->inputBuffer.data();
            float *localOutput = (float *) this->outputBuffer.data();
            volatile char *buffer = (volatile char*)this->baseAddr + 4 + configStringLen;
            RunMultiThreadMemcpy(this->inputBuffer.data(), (uint8_t*)buffer, n * m * sizeof(float), pool);

            std::vector <std::vector <fastllm::Data*> > weights;
            std::vector <std::vector <float> > v;
            weights.resize(n);
            v.resize(n);

            if (n == 1) {
                for (auto &factor : config["factors"].array_items()) {
                    v[0].push_back(factor.number_value());
                }
                for (auto &weight : config["weights"].array_items()) {
                    weights[0].push_back(&this->weights[weight.string_value()]);
                }
            } else {
                int idx = 0;
                for (auto &factor : config["factors"].array_items()) {
                    for (auto &it : factor.array_items()) {
                        v[idx].push_back(it.number_value());
                    }
                    idx++;
                }
                idx = 0;
                for (auto &weight : config["weights"].array_items()) {
                    for (auto &it : weight.array_items()) {
                        weights[idx].push_back(&this->weights[it.string_value()]);
                    }
                    idx++;
                }
            }

            int bs = n, dim = k;
            int inputDim = m;
            std::vector <float> tempResult, middleResult;
            tempResult.resize(bs * dim, 0.0f);
            middleResult.resize(bs * dim, 0.0f);
            std::map <std::pair <Data*, Data*>, std::vector <std::pair <int, float> > > expertTasks; // expertTasks[x]代表专家x的task, expertTasks[x][j] = (第j个任务对应的行数， 权重)            

            Data oriInput, tempInput, w1, w2, w3;

            oriInput.dataType = DataType::FLOAT32;
            oriInput.Resize({n, m});
            oriInput.Allocate();
// printf("server get meta spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
            float *floatInput = (float*)oriInput.cpuData;
            memcpy(floatInput, localInput, n * m * sizeof(float));
// printf("server revert spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
// printf("server get input spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
            tempInput.dataType = DataType::FLOAT32;
            tempInput.Resize({n, m});
            tempInput.Allocate();

            for (int b = 0; b < bs; b++) {
                for (int j = 0; j < v[b].size(); j++) {
                    expertTasks[std::make_pair(weights[b][j * 2], weights[b][j * 2 + 1])].push_back(std::make_pair(b, v[b][j]));
                }
            }
// printf("server ready spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
            for (auto &it : expertTasks) {
                auto &task = it.second;
                if (task.size() == 0) {
                    continue;
                }

                tempInput.Resize({(int)task.size(), inputDim});
                std::vector <MultiThreadMemcpyMultiLinesTask> memcpyTasks;
                for (int i = 0; i < (int)task.size(); i++) {
                    memcpyTasks.push_back(MultiThreadMemcpyMultiLinesTask(tempInput.cpuData + i * inputDim * oriInput.unitSize, oriInput.cpuData + task[i].first * inputDim * oriInput.unitSize, inputDim * oriInput.unitSize));
                }
                RunMultiThreadMemcpyMultiLines(memcpyTasks, pool);
                
                DoCpuLinearReshape(tempInput, *it.first.first, w3);
                DoCpuLinear(tempInput, *it.first.first, Data(), w3);

                int mid = w3.dims[1] / 2;
                w1.Resize({w3.dims[0], mid});
                w1.dataType = w3.dataType;
                w1.Allocate();
                SwigluMultiThread((float *) w3.cpuData, mid, mid, ((float *) w1.cpuData),
                                    w3.dims[0], w3.dims[1], mid, GetAlivePool());
                
                DoCpuLinearReshape(w1, *it.first.second, w2);
                DoCpuLinear(w1, *it.first.second, Data(), w2);

                float *curOutput;
                if (w2.dataType == DataType::FLOAT32) {
                    curOutput = (float*)w2.cpuData;
                } else if (w2.dataType == DataType::FLOAT16) {
                    Float16ToFloat32((uint16_t*)w2.cpuData, middleResult.data(), w2.Count(0));
                    curOutput = middleResult.data();
                }

                for (int i = 0; i < (int)task.size(); i++) {
                    float value = task[i].second;
                    float *lastResult = tempResult.data() + task[i].first * dim;
                    float *curResult = curOutput + i * dim;
                    for (int j = 0; j < dim; j++) {
                        lastResult[j] += value * curResult[j];
                    }
                }
            }
// printf("server calc spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
            for (int i = 0; i < n; i++) {
                memcpy((uint8_t *) baseOutputAddr + partId * n * k * sizeof(float) + (i * k) * sizeof(float),
                    ((uint8_t *) tempResult.data()) + (i * k) * sizeof(float),
                    k * sizeof(float));
            }
            return;    
        } else {
            volatile char *buffer = (volatile char*)this->baseAddr;
            int n = ((int*)buffer)[0];
            int m = ((int*)buffer)[1];
            int k = ((int*)buffer)[2];
            buffer += 3 * sizeof(int);
            std::vector <fastllm::Data*> weights;
            std::vector <float> v;
            int vSize = ((int*)buffer)[0];
            buffer += sizeof(int);
            for (int i = 0; i < vSize; i++) {
                v.push_back(((float*)buffer)[0]);
                buffer += sizeof(float);
            }
            int wSize = ((int*)buffer)[0];
            buffer += sizeof(int);
            for (int i = 0; i < wSize; i++) {
                weights.push_back(this->weightsList[((int*)buffer)[0]]);
                buffer += sizeof(int);
            }
            float *localInput = (float *) this->inputBuffer.data();
            float *localOutput = (float *) this->outputBuffer.data();
            RunMultiThreadMemcpy(this->inputBuffer.data(), (uint8_t*)buffer, n * m * sizeof(float), pool);

            
            auto &localKs = moeFloatSingleVarManagerServer.localKs;
            auto &middles = moeFloatSingleVarManagerServer.middles;
            auto &swigluResults = moeFloatSingleVarManagerServer.swigluResults;
            auto &results = moeFloatSingleVarManagerServer.results;

            localKs.resize(v.size());
            middles.resize(v.size());
            swigluResults.resize(v.size());
            results.resize(v.size());
            for (int j = 0; j < v.size(); j++) {
                int idx = j;
                int localK = weights[idx * 2]->dims[0];
                localKs[j] = (localK);
                middles[j].resize(localK);
                swigluResults[j].resize(localK);
                results[j].resize(weights[idx * 2 + 1]->dims[0]);
            }

            std::vector<fastllm::MultiThreadBaseOp*> ops;
            int threads = pool->threads.size();
            ops.resize(threads);

            auto &bf16Input = moeFloatSingleVarManagerServer.bf16Input;
            bf16Input.resize(m);
            if (weights[0]->dataType == DataType::DATA_GGUF_FORMAT) {
                iqk_quantize_row_q8_K (
                    (float*)localInput, bf16Input.data(), m, 
                    ggml_type_vec_dot_type((ggml_type)weights[0]->ggmlType),
                    (ggml_type)weights[0]->ggmlType
                );
            } else {
                Float32ToBFloat16((float*)localInput, bf16Input.data(), m);
            }

// record.push_back(std::make_pair("prepare datas", GetSpan(ttt, std::chrono::system_clock::now())));
            for (int st = 0; st < v.size(); st++) {
                int k = localKs[st];
                int end = st, selSum = 1; // 一共处理selSum * k个输出

                int curSum = 1;
                for (int l = st + 1; l < v.size(); l++) {
                    int curK = localKs[l];
                    if (curK % k != 0) {
                        break;
                    }
                    curSum += (curK / k);
                    if (threads % curSum == 0) {
                        end = l;
                        selSum = curSum;
                    }
                }
                int base = threads / selSum;
                int threadSt = 0;
                for (int l = st; l <= end; l++) {
                    int idx = l;
                    Data *weight = weights[idx * 2];
                    uint8_t *weightData = (uint8_t *) weight->cpuData;
                    float *outputData = middles[l].data();
                    float *biasData = nullptr;
                    int curK = localKs[l];
                    int curThread = (curK / k) * base;
                    if (weight->dataType == DataType::FP8_E4M3) {                            
                        LaunchLinearBFloat16FP8E4M3(bf16Input.data(), *weight, outputData, biasData, 1, m, curK, ops, pool, threadSt, curThread);
                    } else if (weight->dataType == DataType::BFLOAT16) {                            
                        LaunchLinearBFloat16BFloat16(bf16Input.data(), *weight, outputData, biasData, 1, m, curK, ops, pool, threadSt, curThread);
                    } else if (weight->dataType == DataType::FLOAT16) {
                        LaunchLinearFloat32Float16(localInput, *weight, outputData, biasData, 1, m, curK, ops, pool, threadSt, curThread);
                    } else if (weight->dataType == DataType::DATA_GGUF_FORMAT) {
                        LaunchLinearQ8KGGUF((uint8_t*)bf16Input.data(), weightData, outputData, biasData, weight, 1, m, curK, ops, pool, threadSt, curThread);
                    } else {
                        // TODO: other
                    }
                    threadSt += curThread;
                }

                for (int j = 0; j < ops.size(); j++) {
                    pool->Wait(j);
                    delete ops[j];
                }
// record.push_back(std::make_pair("mul 0", GetSpan(ttt, std::chrono::system_clock::now())));
                // swiglu
                for (int l = st; l <= end; l++) {
                    int idx = l;
                    int spatial = localKs[idx], mid = spatial / 2;
                    Data *weightDown = weights[idx * 2 + 1];
                    float *outputData = middles[l].data();
                    float *swigluData = swigluResults[l].data();
                    int curK = localKs[idx];

                    ops[l - st] = new fastllm::MultiThreadMultiOps();

                    // 如果不是原始精度，那么需要量化一次
                    if (weightDown->dataType == DataType::FLOAT16) {
                        ((fastllm::MultiThreadMultiOps*)ops[l - st])->ops.push_back(new fastllm::MultiThreadSwigluOp(outputData, mid, mid, outputData, 1, spatial, spatial));    
                    } else {
                        ((fastllm::MultiThreadMultiOps*)ops[l - st])->ops.push_back(new fastllm::MultiThreadSwigluOp(outputData, mid, mid, swigluData, 1, spatial, spatial));
                        
                        if (weightDown->dataType == DataType::FP8_E4M3 || weightDown->dataType == DataType::BFLOAT16) {
                            ((fastllm::MultiThreadMultiOps*)ops[l - st])->ops.push_back(new fastllm::MultiThreadFloat32ToBFloat16Op(swigluData, (uint16_t*)middles[l].data(), mid));
                        } else if (weightDown->dataType == DataType::DATA_GGUF_FORMAT) {
                            ((fastllm::MultiThreadMultiOps*)ops[l - st])->ops.push_back(new fastllm::MultiThreadFloat32ToQ8KOp(swigluData, (uint8_t*)middles[l].data(), mid, weightDown->ggmlType));
                        }
                    }

                    pool->PushOp(l - st, ops[l - st]);
                }
                for (int l = st; l <= end; l++) {
                    pool->Wait(l - st);
                    delete ops[l - st];
                }
// record.push_back(std::make_pair("swiglu", GetSpan(ttt, std::chrono::system_clock::now())));
// record.push_back(std::make_pair("quant", GetSpan(ttt, std::chrono::system_clock::now())));
                threadSt = 0;
                for (int l = st; l <= end; l++) {
                    int idx = l;
                    int mid = localKs[idx] / 2;
                    int curK = localKs[idx];
                    Data *weightDown = weights[idx * 2 + 1];
                    int curThread = (curK / k) * base;
                    if (weightDown->dataType == DataType::FP8_E4M3) {
                        LaunchLinearBFloat16FP8E4M3((uint16_t*)middles[l].data(), *weightDown, results[l].data(), nullptr, 1, mid, m, ops, pool, threadSt, curThread);
                    } else if (weightDown->dataType == DataType::BFLOAT16) {
                        LaunchLinearBFloat16BFloat16((uint16_t*)middles[l].data(), *weightDown, results[l].data(), nullptr, 1, mid, m, ops, pool, threadSt, curThread);
                    } else if (weightDown->dataType == DataType::FLOAT16) {
                        LaunchLinearFloat32Float16((float*)middles[l].data(), *weightDown, results[l].data(), nullptr, 1, mid, m, ops, pool, threadSt, curThread);
                    } else if (weightDown->dataType == DataType::DATA_GGUF_FORMAT) {
                        LaunchLinearQ8KGGUF((uint8_t*)middles[l].data(), (uint8_t*)weightDown->cpuData, results[l].data(), nullptr, weightDown, 1, mid, m, ops, pool, threadSt, curThread);
                    } else {
                        // TODO: other
                    }
                    threadSt += curThread;               
                }

                for (int j = 0; j < ops.size(); j++) {
                    pool->Wait(j);
                    delete ops[j];
                }
                st = end;
// record.push_back(std::make_pair("mul 1", GetSpan(ttt, std::chrono::system_clock::now())));
            }
            for (int k = 0; k < m; k++) {
                localOutput[k] = 0;
            }
            for (int j = 0; j < v.size(); j++) {
                float value = v[j];
                float *curOutput = (float*)results[j].data();

                int i = 0;
    #ifdef __AVX2__
                __m256 value_vec = _mm256_set1_ps(value);
                // 每次处理 8 个浮点数（AVX2 寄存器可以容纳 8 个 float）
                for (; i <= m - 8; i += 8) {
                    // 加载 curOutput 的 8 个浮点数
                    __m256 curOutput_vec = _mm256_loadu_ps(&curOutput[i]);

                    // 加载 localOutput 的 8 个浮点数
                    __m256 fLastOutput_vec = _mm256_loadu_ps(&localOutput[i]);

                    // 计算 curOutput * value
                    __m256 result_vec = _mm256_mul_ps(curOutput_vec, value_vec);

                    // 累加到 fLastOutput
                    fLastOutput_vec = _mm256_add_ps(fLastOutput_vec, result_vec);

                    // 将结果存回 fLastOutput
                    _mm256_storeu_ps(&localOutput[i], fLastOutput_vec);
                }
    #endif
                for (int k = i; k < m; k++) {
                    localOutput[k] += curOutput[k] * value;
                }
            }
// record.push_back(std::make_pair("get fp32 sum", GetSpan(ttt, std::chrono::system_clock::now())));
            for (int i = 0; i < n; i++) {
                RunMultiThreadMemcpy((uint8_t *) baseOutputAddr + partId * n * k * sizeof(float) + (i * k) * sizeof(float),
                    (uint8_t *) localOutput + (i * k) * sizeof(float),
                    k * sizeof(float), pool, true);
            }
// record.push_back(std::make_pair("copy output", GetSpan(ttt, std::chrono::system_clock::now())));
// for (int i = 0; i < record.size(); i++) {
     // printf("server %s spend %f s.\n", record[i].first.c_str(), record[i].second);
// }
        }
    }

    void ComputeServer::AppendKVCache() {
        U8ReaderBuffer buffer = U8ReaderBuffer((uint8_t*)this->baseAddr);

        long long uid = buffer.ReadLongLong();
        int dimsSize = buffer.ReadInt();

        std::vector <int> dims;
        for (int i = 0; i < dimsSize; i++) {
            dims.push_back(buffer.ReadInt());
        }
        DataType dataType = (DataType)buffer.ReadInt();
        if (dimsSize != 3) {
            ErrorInFastLLM("KVCache: dims's size should be 3.\n");
        }
        if (dataType == DataType::FLOAT32 || dataType == DataType::BFLOAT16 || dataType == DataType::FLOAT16) {
            int head = dims[0], len = dims[1], dim = dims[2];
            if (this->kvCacheManager.caches.find(uid) == this->kvCacheManager.caches.end()) {
                ClearSomeKVCache();
            }
            this->kvCacheManager.Get(uid, dataType, head, dim)->Append(len, buffer.cur);
        } else {
            ErrorInFastLLM("KVCache: Unsupport datatype.\n");
        }
    }

    // 清除一些长期不用的uid
    void ComputeServer::ClearSomeKVCache() {
        std::vector <long long> uids;
        auto now = std::chrono::system_clock::now();
        for (auto &it : kvCacheManager.caches) {
            auto duration = std::chrono::duration_cast<std::chrono::microseconds> (now - it.second->lastFlushTime);
            float gap = double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
            if (gap > 2 * 60) {
                uids.push_back(it.first);
            }
        }
        for (auto it : uids) {
            kvCacheManager.Delete(it);
        }
    }

    void ComputeServer::DeleteKVCache() {
        // TODO
    }

    void ComputeServer::Attention() {
        int configStringLen = ((int*)this->baseAddr)[0];
        std::string configString;
        for (int i = 0; i < configStringLen; i++) {
            configString += (char)this->baseAddr[4 + i];
        }
        json11::Json config;
        std::string error;
        config = json11::Json::parse(configString, error);

        long long kid = atoll((config)["kid"].string_value().c_str());
        long long vid = atoll((config)["vid"].string_value().c_str());
        int qhead = (config)["qhead"].int_value();
        int qlen = (config)["qlen"].int_value();
        int qdim = (config)["qdim"].int_value();
        DataType qtype = (DataType) (config)["qtype"].int_value();
        int group = (config)["group"].int_value();
        float scale = (config)["scale"].number_value();
        int maskType = (config)["maskType"].int_value();
        KVCache *k = kvCacheManager.Get(kid), *v = kvCacheManager.Get(vid);
        float *qd = (float *) (this->baseAddr + 4 + configStringLen), *kd = (float *) k->data, *vd = (float *) v->data;
        float *od = (float *) this->baseOutputAddr;

        int localHead = qhead / partCnt;
        int startHead = partId * localHead;
        int endHead = startHead + localHead;
        if (partId == partCnt - 1) {
            endHead = qhead;
        }

        std::vector <float> localInput;
        localInput.resize((endHead - startHead) * qlen * qdim, 0);
        RunMultiThreadMemcpy((uint8_t*)localInput.data(), (uint8_t*)(qd + startHead * qlen * qdim), localInput.size() * sizeof(float), pool);

        std::vector <float> localOutput;
        localOutput.resize((endHead - startHead) * qlen * v->dim, 0);

        std::vector<MultiThreadSingleAttentionCausalOp*> ops;
        for (int o = startHead; o < endHead; o++) {
            int block = 4;
            for (int i = 0; i < qlen; i += block) {
                int curBlock = std::min(block, qlen - i);
                ops.push_back(new MultiThreadSingleAttentionCausalOp(
                        localInput.data() + (o - startHead) * qlen * qdim + i * qdim,
                        kd + (o / group) * k->currentCap * k->dim,
                        vd + (o / group) * v->currentCap * v->dim,
                        localOutput.data() + (o - startHead) * qlen * v->dim + i * v->dim,
                        scale, curBlock, qdim, k->len - qlen + i, k->len, v->dim)
                );
            }
        }

        for (int st = 0; st < ops.size(); st += pool->threads.size()) {
            int end = std::min(ops.size(), st + pool->threads.size());
            for (int i = st; i < end; i++) {
                pool->PushOp(i - st, ops[i]);
            }
            for (int i = st; i < end; i++) {
                pool->Wait(i - st);
            }
        }

        RunMultiThreadMemcpy((uint8_t*)(((float*)this->baseOutputAddr) + startHead * qlen * v->dim),
                          (uint8_t*)localOutput.data(),
               (endHead - startHead) * qlen * v->dim * sizeof(float), pool);
    }

    void ComputeServer::FindData() {
        int nameLen = ((int*)this->baseAddr)[0];
        std::string name;
        for (int i = 0; i < nameLen; i++) {
            name += (char)this->baseAddr[4 + i];
        }
        ((int32_t*)baseOutputAddr)[0] = (int)(this->weights.weight.find(name) != this->weights.weight.end());
    }
}