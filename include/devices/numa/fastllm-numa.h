//
// Created by huangyuyang on 4/11/24.
//

#ifndef FASTLLM_NUMA_COMPUTE_H
#define FASTLLM_NUMA_COMPUTE_H

#include "fastllm.h"

namespace fastllm {
    struct NumaClient {
        int fd;
        volatile uint8_t *buf;
        volatile uint8_t *result;
        volatile int32_t *flag;

        int serverVersion;
        int serverNumaCnt;

        std::set <std::string> registerDataNames; // 向服务器上注册过的DataName

        NumaClient ();

        ~NumaClient ();

        void Launch(int opType);

        void Wait();

        void SendLongMessage(uint8_t *buffer, uint64_t len);

        void RegisterFastllmData(fastllm::Data *data, const std::string &weightType);

        void UnregisterFastllmData(const std::string &dataName);

        void RunNumaLinearU(int n, int m, int k, int group, int groupCnt,
                             fastllm::Data *weight, fastllm::Data *bias,
                             std::vector <LowBitConfig> *inputConfigs,
                             uint8_t *uinput, float *output, 
                             LinearExType exType, 
                             DataType outputType);

        void RunNumaLinearF(int n, int m, int k, fastllm::Data *weight, fastllm::Data *bias,
                            float *input, float *output, LinearExType exType, DataType dataType);
        
        void RunNumaMOEU(int n, int m, int k, int group, int groupCnt,
                        std::vector <fastllm::Data*> weights, std::vector <float> factors,
                        std::vector <LowBitConfig> *inputConfigs,
                        uint8_t *uinput, float *output, 
                        DataType outputType);
        
        void RunNumaMOEUMultiRow(int n, int m, int k, int group, int groupCnt,
                        std::vector <std::vector <fastllm::Data*> > &weights, std::vector <std::vector <float> > &factors,
                        std::vector <LowBitConfig> *inputConfigs,
                        uint8_t *uinput, float *output, 
                        DataType outputType);
        
        void RunNumaMOEF(int n, int m, int k,
                        std::vector <fastllm::Data*> weights, std::vector <float> factors,
                        float *input, float *output, 
                        DataType outputType);
        
        void RunNumaMOEFMultiRow(int n, int m, int k,
                        std::vector <std::vector <fastllm::Data*> > &weights, std::vector <std::vector <float> > &factors,
                        float *input, float *output, 
                        DataType outputType);
        
        void AppendKVCache(long long uid, Data *content);

        void Attention(Data *q, Data *k, Data *v, int group, float scale, int maskType, Data *output);
    };

    void RegisterFastllmData(fastllm::Data *data, const std::string &weightType);
}

#endif