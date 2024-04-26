//
// Created by huangyuyang on 4/11/24.
//

#ifndef FASTLLM_TFACC_COMPUTE_H
#define FASTLLM_TFACC_COMPUTE_H

#include "fastllm.h"

namespace fastllm {
    struct TfaccClient {
        int fd;
        volatile uint8_t *buf;
        volatile uint8_t *result;
        volatile int32_t *flag;

        int serverVersion;
        int serverNumaCnt;

        std::set <std::string> registerDataNames; // 向服务器上注册过的DataName

        TfaccClient ();

        ~TfaccClient ();

        void Launch(int opType);

        void Wait();

        void SendLongMessage(uint8_t *buffer, uint64_t len);

        void RegisterFastllmData(fastllm::Data *data, const std::string &weightType);

        void UnregisterFastllmData(const std::string &dataName);

        void RunTfaccLinearU(int n, int m, int k, int group, int groupCnt,
                             fastllm::Data *weight, fastllm::Data *bias,
                             std::vector <LowBitConfig> *inputConfigs,
                             uint8_t *uinput, float *output, 
                             LinearExType exType);

        void RunTfaccLinearF(int n, int m, int k, fastllm::Data *weight, fastllm::Data *bias,
                            float *input, float *output, LinearExType exType, DataType dataType);
        
        void AppendKVCache(long long uid, Data *content);

        void Attention(Data *q, Data *k, Data *v, int group, float scale, int maskType, Data *output);
    };
}

#endif