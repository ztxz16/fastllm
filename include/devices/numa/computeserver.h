//
// Created by huangyuyang on 24-4-11.
//

#ifndef TEST_COMPUTESERVER_H
#define TEST_COMPUTESERVER_H

#include "utils.h"
#include "fastllm.h"
#include "alivethreadpool.h"
#include "json11.hpp"
#include "kvcache.h"

const int DDRLEN = 512 * 1024 * 1024;
const int OUTPUTOFFSET = 256 * 1024 * 1024;
const int FLAGOFFSET = 511 * 1024 * 1024;
const int PAGE = 16 * 1024;

namespace fastllm {
    enum ComputeTaskType {
        None = 0,
        LinearInt4NoZero = 1,
        LinearInt8 = 2,
        LinearFloat16 = 3,
        LinearFloat32 = 4,
        LinearInt4Group = 5,

        AppendKVCache = 6,
        DoAttention = 7,

        MOEInt4NoZero = 8,
        MOEInt4Group = 9,

        LinearFP8E4M3 = 10,
        MOEFP8E4M3 = 11,

        GetComputeServerInfo = 10000,
        StartLongData = 10001,
        FinishLongData = 10002,
        FindData = 10003
    };

    struct ComputeServer {
        std::vector <uint8_t> longBuffer;

        std::vector <uint8_t> inputBuffer;
        std::vector <uint8_t> outputBuffer;

        int partId, partCnt;
        int threadNum;

        WeightMap weights;
        std::vector <fastllm::Data*> weightsList;

        AliveThreadPool *pool;

        volatile uint8_t* baseAddr;
        volatile uint8_t* baseOutputAddr;
        volatile int* flag;

        KVCacheManager kvCacheManager;

        ComputeServer (int partId, int partCnt, int threadNum);

        void Start();

        void SendComputeServerInfo();

        void ReceiveLongData(); // 接收长数据，送入buffer

        void FinishLongData(); // 长数据接收完成，开始处理

        void RegisterData(json11::Json *config, uint8_t *buffer); // 注册data

        void UnregisterData(json11::Json *config); // 注销data

        void GetLinearIntBaseInfo(int &n, int &m, int &k, int &group, int &groupCnt,
                                  std::string &weightName, std::string &biasName,
                                  std::vector <LowBitConfig> &configs,
                                  LinearExType &exType, DataType &outputDataType, AliveThreadPool *pool);

        void RunLinearInt();

        void GetLinearFloatBaseInfo(int &n, int &m, int &k,
                                    std::string &weightName, std::string &biasName,
                                    DataType &dataType, LinearExType &exType);

        void RunLinearFloat();

        void GetMOEIntBaseInfo(int &n, int &m, int &k, int &group, int &groupCnt,
                                  std::string &weightName, std::string &biasName,
                                  std::vector <LowBitConfig> &configs,
                                  LinearExType &exType, DataType &outputDataType, AliveThreadPool *pool);

        void RunMOEInt();

        void RunMOEFloat();

        void AppendKVCache();

        void ClearSomeKVCache();

        void DeleteKVCache();

        void Attention(); // attention

        void FindData(); // 查询Data是否存在
    };
}

#endif //TEST_COMPUTESERVER_H
