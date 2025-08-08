//
// Created by huangyuyang on 4/11/23.
//

#include <sys/prctl.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <cstring>
#include <algorithm>
#include <numa.h>

#include "utils.h"
#include "devices/numa/fastllm-numa.h"
#include "devices/cpu/alivethreadpool.h"
#include "json11.hpp"
#include "computeserver.h"

namespace fastllm {
    extern AliveThreadPool *GetAlivePool();

    static int transLimit = 255 * 1024 * 1024;

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

    NumaClient::NumaClient() {
        try {
            std::string s = getenv("FASTLLM_ACTIVATE_NUMA");
            if (!(s != "" && s != "OFF")) {
                return;
            }
        } catch (...) {
            return;
        }
        // 获取所有可用NUMA节点
        struct bitmask* validNodes = numa_get_mems_allowed();
        std::vector<int> nodes;
        
        // 遍历所有可能的节点
        for (int node = 0; node <= numa_max_node(); ++node) {
            if (numa_bitmask_isbitset(validNodes, node)) {
                nodes.push_back(node);
            }
        }

        int numaThreads = 27;
        try {
            std::string s = getenv("FASTLLM_NUMA_THREADS");
            if (s != "") {
                int t = atoi(s.c_str());
                if (t > 0) {
                    numaThreads = t;
                }
            }
        } catch (...) {
        }
        try {
            std::string s = getenv("FASTLLM_NUMAS");
            if (s != "") {
                int t = atoi(s.c_str());
                if (t > 0 && t < nodes.size()) {
                    nodes.resize(t);
                }
            }
        } catch (...) {
        }

        for (int i = 0; i < nodes.size(); i++) {
            int forkId = fork();
            if (forkId == 0) {
                int partId = i;
                int partCnt = nodes.size();

                int numaId = nodes[i];
                // 绑定到指定NUMA节点
                if (numa_run_on_node(numaId) != 0) {
                    std::cerr << "Failed to bind process to node " << numaId << ": " << strerror(errno) << std::endl;
                    exit(EXIT_FAILURE);
                }

                // 设置内存分配策略
                struct bitmask *mask = numa_bitmask_alloc(numa_num_configured_nodes());
                numa_bitmask_clearall(mask);
                numa_bitmask_setbit(mask, numaId);
                numa_set_membind(mask);
                numa_bitmask_free(mask);
                
                printf("numa server running on node %d. (part %d / %d, %d threads)\n", numaId, partId, partCnt, numaThreads);
                fastllm::ComputeServer *computeServer = new fastllm::ComputeServer(partId, partCnt, numaThreads);
                computeServer->Start();
            }
        }

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

        buf = (volatile uint8_t*)data;

        // fd = open("/dev/thinkforce0", O_RDWR);
        // buf = (volatile uint8_t *)mmap(NULL, 256 * 1024 * 1024, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 9997 * 0x1000);

        result = buf + OUTPUTOFFSET;
        flag = (volatile int32_t*)(buf + FLAGOFFSET);

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

    NumaClient::~NumaClient() {
        auto names = this->registerDataNames;
        for (auto &dataName : names) {
            this->UnregisterFastllmData(dataName);
        }
    }

    void NumaClient::Launch(int opType) {
        barrier();
        volatile int *curFlag = flag;
        for (int i = 0; i < serverNumaCnt; i++) {
            *(curFlag) = opType;
            curFlag += PAGE;
            barrier();
        }
    }

    void NumaClient::Wait() {
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

    void NumaClient::SendLongMessage(uint8_t *buffer, uint64_t len) {
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

    void NumaClient::RegisterFastllmData(fastllm::Data *data, const std::string &weightType) {
        if (data->name == "" || this->registerDataNames.find(data->name) != this->registerDataNames.end()) {
            return;
        }
        this->registerDataNames.insert(data->name);
        data->isRegistered = true;
        ((int*)this->buf)[0] = data->name.size();
        memcpy((uint8_t*)this->buf + 4, data->name.data(), data->name.size());
        this->Launch(ComputeTaskType::FindData);
        this->Wait();
        if (((int*)this->result)[0]) {            
            return;
        }

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
        } else if (dataType == DataType::FP8_E4M3) {
            buffer.WriteInt((int) dataType);
            buffer.WriteInt(data->blockK);
            buffer.WriteInt(data->blockM);
            buffer.WriteInt((int)data->scales.size());
            buffer.WriteBytes((uint8_t*)data->scales.data(), (int)data->scales.size() * sizeof(float));
            buffer.WriteBytes(data->cpuData, data->GetBytes());
        } else if (dataType == DataType::INT8 || dataType == DataType::INT4 || dataType == DataType::INT4_NOZERO) {
            buffer.WriteInt((int) dataType);
            buffer.WriteInt(data->perChannelAxis);
            int k = data->perChannelAxis == -1 ? 1 : data->dims[data->perChannelAxis];
            for (int i = 0; i < k; i++) {
                buffer.WriteFloat(data->perChannelsConfigs[i].min);
                buffer.WriteFloat(dataType == DataType::INT4_NOZERO ? data->perChannelsConfigs[i].scale : data->perChannelsConfigs[i].max);
            }
            buffer.WriteBytes(data->cpuData, data->GetBytes());
        } else if (dataType == DataType::INT4_GROUP) {
            buffer.WriteInt((int) dataType);
            buffer.WriteInt(data->perChannelAxis);
            buffer.WriteInt(data->group);
            buffer.WriteInt(data->groupCnt);
            int k = data->perChannelAxis == -1 ? 1 : data->dims[data->perChannelAxis];
            for (int i = 0; i < k * data->group; i++) {
                buffer.WriteFloat(data->mins[i]);
                buffer.WriteFloat(data->scales[i]);
            }
            buffer.WriteBytes(data->cpuData, data->GetBytes());
        } else if (dataType == DataType::DATA_GGUF_FORMAT) {
            buffer.WriteInt((int) dataType);
            buffer.WriteInt((int) data->ggmlType);
            buffer.WriteBytes(data->cpuData, data->GetBytes());
        }

        SendLongMessage(buffer.buffer.data(), buffer.buffer.size());

        /// TODO: data->clear()
        data->weightId = ((int32_t*)this->result)[0];
        delete[] data->cpuData;
        data->cpuData = new uint8_t[1];
        // data->weightSum.clear();
        data->mins.clear();
        data->scales.clear();
        data->zeros.clear();
        data->perChannelsConfigs.clear();

        data->mins.shrink_to_fit();
        data->scales.shrink_to_fit();
        data->zeros.shrink_to_fit();
        data->perChannelsConfigs.shrink_to_fit();
    }

    void NumaClient::UnregisterFastllmData(const std::string &dataName) {
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

    void NumaClient::RunNumaLinearU(int n, int m, int k, int group, int groupCnt, 
                                     fastllm::Data *weight, fastllm::Data *bias,
                                     std::vector <LowBitConfig> *inputConfigs,
                                     uint8_t *uinput, float *output, 
                                     LinearExType exType, 
                                     DataType outputType) {
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
        int outputUnitSize = (outputType == DataType::FLOAT32 ? sizeof(float) : sizeof(uint16_t));
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
            ((int32_t*)buf)[8] = outputType;
            
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

            RunMultiThreadMemcpy(((uint8_t*) output) + baseN * outK * outputUnitSize,
                    (uint8_t*) result,
                    curN * outK * outputUnitSize, GetAlivePool());
// auto st3 = std::chrono::system_clock::now();
// if (n > 0) printf("n = %d, m = %d, k = %d, input = %f s, calc = %f s, output = %f. total = %f\n", n, m, k, GetSpan(st0, st1), GetSpan(st1, st2), GetSpan(st2, st3), GetSpan(st0, st3));
        }
    }

    void NumaClient::RunNumaLinearF(int n, int m, int k, fastllm::Data *weight, fastllm::Data *bias,
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
        if (weight->dataType == DataType::FP8_E4M3) {
            opType = ComputeTaskType::LinearFP8E4M3;
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

    void NumaClient::RunNumaMOEF(int n, int m, int k,
        std::vector <fastllm::Data*> weights, std::vector <float> factors,
        float *input, float *output, 
        DataType outputType) {
        // if (this->registerDataNames.find(weights[0]->name) == this->registerDataNames.end()) {
        if (!weights[0]->isRegistered) {
            for (int i = 0; i < weights.size(); i += 2) {
                RegisterFastllmData(weights[i], "linearSwiglu");
                RegisterFastllmData(weights[i + 1], "linearColumn");
            }
        }
        int opType = ComputeTaskType::MOEFP8E4M3;
        int maxN = 1;
        int outputUnitSize = (outputType == DataType::FLOAT32 ? sizeof(float) : sizeof(uint16_t));
        
        for (int baseN = 0; baseN < n; baseN += maxN) {
            int curN = std::min(maxN, n - baseN);
            U8Buffer buffer;
            buffer.WriteInt(n);
            buffer.WriteInt(m);
            buffer.WriteInt(k);
            buffer.WriteInt((int)factors.size());
            for (int i = 0; i < factors.size(); i++) {
                buffer.WriteFloat(factors[i]);
            }
            buffer.WriteInt((int)weights.size());
            for (int i = 0; i < weights.size(); i++) {
                buffer.WriteInt(weights[i]->weightId);
            }

            RunMultiThreadMemcpy((uint8_t*)this->buf, buffer.buffer.data(), buffer.buffer.size(), GetAlivePool());
            RunMultiThreadMemcpy((uint8_t*)this->buf + buffer.buffer.size(), (uint8_t*)(input + baseN * m), curN * m * sizeof(float), GetAlivePool());
            this->Launch(opType);
            this->Wait();

            auto pool = GetAlivePool();
            uint8_t *oriResult = new uint8_t[serverNumaCnt * curN * k * outputUnitSize];
            RunMultiThreadMemcpy(oriResult, (uint8_t*)result, serverNumaCnt * curN * k * outputUnitSize, GetAlivePool());
            float *floatResult = (float*)oriResult;
            for (int i = 1; i < serverNumaCnt; i++) {
                for (int j = 0; j < curN * k; j++) {
                    floatResult[j] += floatResult[i * curN * k + j];
                }
            }
            RunMultiThreadMemcpy(((uint8_t*) output) + baseN * k * outputUnitSize, (uint8_t*) oriResult,
                    curN * k * outputUnitSize, GetAlivePool());
            delete[] oriResult;
        }
    }

    void NumaClient::RunNumaMOEFMultiRow(int n, int m, int k,
                        std::vector <std::vector <fastllm::Data*> > &weights, std::vector <std::vector <float> > &factors,
                        float *input, float *output, 
                        DataType outputType) {
        if (this->registerDataNames.find(weights[0][0]->name) == this->registerDataNames.end()) {
            for (auto &w : weights) {
                for (int i = 0; i < w.size(); i += 2) {
                    RegisterFastllmData(w[i], "linearSwiglu");
                    RegisterFastllmData(w[i + 1], "linearColumn");
                }
            }
        }
        int opType = ComputeTaskType::MOEFP8E4M3;

        int maxN = n;
        maxN = std::min(maxN, transLimit / m);
        maxN = std::min(maxN, (int)(transLimit / (serverNumaCnt * k * sizeof(float))));

        int outputUnitSize = (outputType == DataType::FLOAT32 ? sizeof(float) : sizeof(uint16_t));
        for (int baseN = 0; baseN < n; baseN += maxN) {
            int curN = std::min(maxN, n - baseN);

            std::string factorString = "[", weightString = "[";
            for (int fid = 0; fid < factors.size(); fid++) {
                if (fid > 0) {
                    factorString += ",\n";
                }
                factorString += "[";
                for (int i = 0; i < factors[fid].size(); i++) {
                    if (i > 0) {
                        factorString += ",\n";
                    }
                    factorString += std::to_string(factors[fid][i]);
                }
                factorString += "]";
            }
            factorString += "]";

            for (int wid = 0; wid < weights.size(); wid++) {
                if (wid > 0) {
                    weightString += ",\n";
                }
                weightString += "[";
                for (int i = 0; i < weights[wid].size(); i++) {
                    if (i > 0) {
                        weightString += ",\n";
                    }
                    weightString += ('\"') + weights[wid][i]->name + ('\"');
                }
                weightString += "]";
            }
            weightString += "]";

            std::string configString = "{";
            configString += "\"factors\":" +  factorString + ",";
            configString += "\"n\":" +  std::to_string(n) + ",";
            configString += "\"m\":" +  std::to_string(m) + ",";
            configString += "\"k\":" +  std::to_string(k) + ",";
            configString += "\"op\":\"moe\",";
            configString += "\"outputType\":" +  std::to_string(0) + ",";
            configString += "\"weights\":" +  weightString + "}";
            U8Buffer buffer;
            buffer.WriteInt(configString.size());
            buffer.WriteBytes((uint8_t*)configString.data(), configString.size());
            RunMultiThreadMemcpy((uint8_t*)this->buf, buffer.buffer.data(), buffer.buffer.size(), GetAlivePool());
            RunMultiThreadMemcpy((uint8_t*)this->buf + buffer.buffer.size(), (uint8_t*)(input + baseN * m), curN * m * sizeof(float), GetAlivePool());
            this->Launch(opType);
            this->Wait();
            auto pool = GetAlivePool();

            uint8_t *oriResult = new uint8_t[serverNumaCnt * curN * k * outputUnitSize];
            RunMultiThreadMemcpy(oriResult, (uint8_t*)result, serverNumaCnt * curN * k * outputUnitSize, GetAlivePool());
            float *floatResult = (float*)oriResult;
            for (int i = 1; i < serverNumaCnt; i++) {
                for (int j = 0; j < curN * k; j++) {
                    floatResult[j] += floatResult[i * curN * k + j];
                }
            }
            RunMultiThreadMemcpy(((uint8_t*) output) + baseN * k * outputUnitSize, (uint8_t*) oriResult,
                    curN * k * outputUnitSize, GetAlivePool());
            delete[] oriResult;
        }   
    }

    void NumaClient::RunNumaMOEU(int n, int m, int k, int group, int groupCnt,
                        std::vector <fastllm::Data*> weights, std::vector <float> factors,
                        std::vector <LowBitConfig> *inputConfigs,
                        uint8_t *uinput, float *output, 
                        DataType outputType) {
 // auto ttt = std::chrono::system_clock::now();
 // std::vector <std::pair <std::string, float> > record;
        // if (this->registerDataNames.find(weights[0]->name) == this->registerDataNames.end()) {
        if (!weights[0]->isRegistered) {
            for (int i = 0; i < weights.size(); i += 2) {
                RegisterFastllmData(weights[i], "linearSwiglu");
                RegisterFastllmData(weights[i + 1], "linearColumn");
            }
        }
        int opType = ComputeTaskType::MOEInt4NoZero;
        /*if (weights[0]->dataType == DataType::INT8) {
            opType = ComputeTaskType::LinearInt8;
        }
        if (weight->dataType == DataType::INT4_GROUP) {
            opType = ComputeTaskType::LinearInt4Group;
        }*/


        int maxN = n;
        maxN = std::min(maxN, transLimit / m);
        maxN = std::min(maxN, (int)(transLimit / (serverNumaCnt * k * sizeof(float))));

        // printf("maxN = %d\n", maxN);
 // record.push_back(std::make_pair("before make strings", GetSpan(ttt, std::chrono::system_clock::now())));
 // record.push_back(std::make_pair("make strings", GetSpan(ttt, std::chrono::system_clock::now())));
        int outputUnitSize = (outputType == DataType::FLOAT32 ? sizeof(float) : sizeof(uint16_t));
        
        maxN = 1;
        for (int baseN = 0; baseN < n; baseN += maxN) {
// auto st0 = std::chrono::system_clock::now();
            int curN = std::min(maxN, n - baseN);
            U8Buffer buffer;
            buffer.WriteInt(n);
            buffer.WriteInt(m);
            buffer.WriteInt(k);
            buffer.WriteInt(group);
            buffer.WriteInt(groupCnt);

            buffer.WriteInt((int)factors.size());
            for (int i = 0; i < factors.size(); i++) {
                buffer.WriteFloat(factors[i]);
            }

            buffer.WriteInt((int)weights.size());
            for (int i = 0; i < weights.size(); i++) {
                buffer.WriteInt(weights[i]->weightId);
            }

            std::vector <float> minmaxs;
            for (int i = 0; i < curN * group; i++) {
                minmaxs.push_back((*inputConfigs)[baseN * group + i].min);
                minmaxs.push_back((*inputConfigs)[baseN * group + i].max);
            }
// record.push_back(std::make_pair("finish data", GetSpan(ttt, std::chrono::system_clock::now())));
            RunMultiThreadMemcpy((uint8_t*)this->buf, buffer.buffer.data(), buffer.buffer.size(), GetAlivePool());
            RunMultiThreadMemcpy((uint8_t*)this->buf + buffer.buffer.size(), (uint8_t*)minmaxs.data(), minmaxs.size() * sizeof(float), GetAlivePool());
            RunMultiThreadMemcpy((uint8_t*)this->buf + buffer.buffer.size() + minmaxs.size() * sizeof(float), (uint8_t*)uinput + baseN * m, curN * m, GetAlivePool());
// record.push_back(std::make_pair("copy data", GetSpan(ttt, std::chrono::system_clock::now())));
            this->Launch(opType);
            this->Wait();
// record.push_back(std::make_pair("calc", GetSpan(ttt, std::chrono::system_clock::now())));
            auto pool = GetAlivePool();

            uint8_t *oriResult = new uint8_t[serverNumaCnt * curN * k * outputUnitSize];
            RunMultiThreadMemcpy(oriResult, (uint8_t*)result, serverNumaCnt * curN * k * outputUnitSize, GetAlivePool());
            float *floatResult = (float*)oriResult;
            for (int i = 1; i < serverNumaCnt; i++) {
                for (int j = 0; j < curN * k; j++) {
                    floatResult[j] += floatResult[i * curN * k + j];
                }
            }
            RunMultiThreadMemcpy(((uint8_t*) output) + baseN * k * outputUnitSize, (uint8_t*) oriResult,
                    curN * k * outputUnitSize, GetAlivePool());
            delete[] oriResult;
// record.push_back(std::make_pair("get sum", GetSpan(ttt, std::chrono::system_clock::now())));
        }
// for (int i = 0; i < record.size(); i++) {
     // printf("inner %s spend %f s.\n", record[i].first.c_str(), record[i].second);
// }
    }

    void NumaClient::RunNumaMOEUMultiRow(int n, int m, int k, int group, int groupCnt,
                        std::vector <std::vector <fastllm::Data*> > &weights, std::vector <std::vector <float> > &factors,
                        std::vector <LowBitConfig> *inputConfigs,
                        uint8_t *uinput, float *output, 
                        DataType outputType) {
        if (this->registerDataNames.find(weights[0][0]->name) == this->registerDataNames.end()) {
            for (auto &w : weights) {
                for (int i = 0; i < w.size(); i += 2) {
                    RegisterFastllmData(w[i], "linearSwiglu");
                    RegisterFastllmData(w[i + 1], "linearColumn");
                }
            }
        }
        int opType = ComputeTaskType::MOEInt4NoZero;
        /*if (weights[0]->dataType == DataType::INT8) {
            opType = ComputeTaskType::LinearInt8;
        }
        if (weight->dataType == DataType::INT4_GROUP) {
            opType = ComputeTaskType::LinearInt4Group;
        }*/

        int maxN = n;
        maxN = std::min(maxN, transLimit / m);
        maxN = std::min(maxN, (int)(transLimit / (serverNumaCnt * k * sizeof(float))));

        int outputUnitSize = (outputType == DataType::FLOAT32 ? sizeof(float) : sizeof(uint16_t));
        for (int baseN = 0; baseN < n; baseN += maxN) {
            int curN = std::min(maxN, n - baseN);

            std::string factorString = "[", weightString = "[";
            for (int fid = 0; fid < factors.size(); fid++) {
                if (fid > 0) {
                    factorString += ",\n";
                }
                factorString += "[";
                for (int i = 0; i < factors[fid].size(); i++) {
                    if (i > 0) {
                        factorString += ",\n";
                    }
                    factorString += std::to_string(factors[fid][i]);
                }
                factorString += "]";
            }
            factorString += "]";

            for (int wid = 0; wid < weights.size(); wid++) {
                if (wid > 0) {
                    weightString += ",\n";
                }
                weightString += "[";
                for (int i = 0; i < weights[wid].size(); i++) {
                    if (i > 0) {
                        weightString += ",\n";
                    }
                    weightString += ('\"') + weights[wid][i]->name + ('\"');
                }
                weightString += "]";
            }
            weightString += "]";

            std::string configString = "{";
            configString += "\"factors\":" +  factorString + ",";
            configString += "\"group\":" +  std::to_string(group) + ",";
            configString += "\"groupCnt\":" +  std::to_string(groupCnt) + ",";
            configString += "\"n\":" +  std::to_string(n) + ",";
            configString += "\"m\":" +  std::to_string(m) + ",";
            configString += "\"k\":" +  std::to_string(k) + ",";
            configString += "\"op\":\"moe\",";
            configString += "\"outputType\":" +  std::to_string(0) + ",";
            configString += "\"weights\":" +  weightString + "}";
            U8Buffer buffer;
            buffer.WriteInt(configString.size());
            buffer.WriteBytes((uint8_t*)configString.data(), configString.size());
            std::vector <float> minmaxs;
            for (int i = 0; i < curN * group; i++) {
                minmaxs.push_back((*inputConfigs)[baseN * group + i].min);
                minmaxs.push_back((*inputConfigs)[baseN * group + i].max);
            }
            RunMultiThreadMemcpy((uint8_t*)this->buf, buffer.buffer.data(), buffer.buffer.size(), GetAlivePool());
            RunMultiThreadMemcpy((uint8_t*)this->buf + buffer.buffer.size(), (uint8_t*)minmaxs.data(), minmaxs.size() * sizeof(float), GetAlivePool());
            RunMultiThreadMemcpy((uint8_t*)this->buf + buffer.buffer.size() + minmaxs.size() * sizeof(float), (uint8_t*)uinput + baseN * m, curN * m, GetAlivePool());
            this->Launch(opType);
            this->Wait();
            auto pool = GetAlivePool();

            uint8_t *oriResult = new uint8_t[serverNumaCnt * curN * k * outputUnitSize];
            RunMultiThreadMemcpy(oriResult, (uint8_t*)result, serverNumaCnt * curN * k * outputUnitSize, GetAlivePool());
            float *floatResult = (float*)oriResult;
            for (int i = 1; i < serverNumaCnt; i++) {
                for (int j = 0; j < curN * k; j++) {
                    floatResult[j] += floatResult[i * curN * k + j];
                }
            }
            RunMultiThreadMemcpy(((uint8_t*) output) + baseN * k * outputUnitSize, (uint8_t*) oriResult,
                    curN * k * outputUnitSize, GetAlivePool());
            delete[] oriResult;
        }   
    }

    void NumaClient::AppendKVCache(long long uid, Data *data) {
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

    void NumaClient::Attention(Data *q, Data *k, Data *v, int group, float scale, int maskType, Data *output) {
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
