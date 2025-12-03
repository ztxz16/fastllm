//
// Created by huangyuyang on 10/15/24.
//

#include <sys/mman.h>
#include <fcntl.h>

#include "devices/numas/numasdevice.h"
#include "devices/cpu/cpudevice.h"
#include "devices/cpu/alivethreadpool.h"

#include <cstring>
#include <thread>
#include <mutex>

#include <cfloat>
#include <cmath>

#ifdef __aarch64__
#include <arm_neon.h>
#include "armMath.h"
#endif

#include "utils.h"
#include "computeutils.h"
#include "numas.h"

namespace fastllm {
    extern CPUInstructInfo *GetCPUInstructInfo();

    static MachineNumaInfo machineNumaInfo;
    NumaConfig *fastllmNumaConfig = nullptr;
    std::mutex numaConfigLocker;

    NumaConfig *GetNumaConfig() {
        numaConfigLocker.lock();
        auto *pool = GetAlivePool();
        if (fastllmNumaConfig == nullptr) {
            fastllmNumaConfig = new NumaConfig(pool->threads.size(), pool, &machineNumaInfo);
        }
        numaConfigLocker.unlock();
        
        return fastllmNumaConfig;
    }

    NumasDevice::NumasDevice() {
        this->deviceType = "numa";
        // this->ops["Linear"] = (BaseOperator *) (new NumasLinearOp());
        this->ops["MergeMOE"] = (BaseOperator*)(new NumasMergeMOE());

        /*this->ops["CatDirect"] = (BaseOperator *) (new NumaCatDirectOp());
        this->ops["Attention"] = (BaseOperator *) (new NumaAttention());

        this->ops["AttentionBatch"] = (BaseOperator *) (new NumaAttentionBatchOp());
        this->ops["CatDirectBatch"] = (BaseOperator *) (new NumaCatDirectBatchOp());*/
    }

    bool NumasDevice::Malloc(void **ret, size_t size) {
        *ret = (void*)new uint8_t [size];
        return true;
    }

    bool NumasDevice::Free(void *ret) {
        delete[] (uint8_t *)ret;
        return true;
    }

    bool NumasDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
        return true;
    }
    
    bool NumasDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
        return true;
    }

    extern void Float16ToFloat32(uint16_t *float16, float *float32, int len);
    extern void Float32ToFloat16(float *float32, uint16_t *float16, int len);

    void Fp8ToFastllmFP8_E4M3_BLOCK128(int experts, int k, int m, uint8_t *fp8, float *scales, int blockK, int blockM, std::vector <uint8_t> &fp8Packed) {
        int ks = (k - 1) / blockK + 1;
        int ms = (m - 1) / blockM + 1;
        
        // 计算每行需要的总字节数
        // 每128个fp8需要1个float的scale，所以需要 (m + 127) / 128 个scale
        int numScalesPerRow = (m + 127) / 128;
        int rowSize = m + numScalesPerRow * sizeof(float);
        fp8Packed.resize((size_t)experts * k * rowSize);
        for (size_t i = 0; i < experts; i++) {
            for (size_t j = 0; j < k; j++) {
                size_t rowIdx = i * k + j;
                size_t packedOffset = rowIdx * rowSize;
                
                // 按照每128个fp8后接一个scale的格式打包
                size_t currentPos = packedOffset;
                
                for (int blockIdx = 0; blockIdx < numScalesPerRow; blockIdx++) {
                    size_t blockStart = blockIdx * 128;
                    size_t blockEnd = std::min(blockStart + 128, (size_t)m);
                    size_t blockSize = blockEnd - blockStart;
                    
                    // 复制当前block的fp8数据
                    for (size_t l = blockStart; l < blockEnd; l++) {
                        size_t srcIdx = i * k * m + j * m + l;
                        fp8Packed[currentPos++] = fp8[srcIdx];
                    }
                    
                    // 在这个block后面添加对应的scale
                    // scale的索引计算：需要根据当前block在整个矩阵中的位置
                    size_t scaleRow = j / blockK;  // 当前行属于哪个scale行块
                    size_t scaleCol = blockStart / blockM;  // 当前block属于哪个scale列块
                    size_t scaleIdx = i * ks * ms + scaleRow * ms + scaleCol;
                    
                    float* scalePtr = (float*)(&fp8Packed[currentPos]);
                    *scalePtr = scales[scaleIdx];
                    currentPos += sizeof(float);
                }
            }
        }
    }

    void Fp8PerchannelToFastllmFP8_E4M3_PERCHANNEL(int experts, int k, int m, uint8_t *fp8, float *scales, int blockK, int blockM, std::vector <uint8_t> &fp8Packed) {
        int ks = (k - 1) / blockK + 1;
        int ms = (m - 1) / blockM + 1;
        int rowSize = m + sizeof(float);
        fp8Packed.resize((size_t)experts * k * rowSize);
        for (size_t i = 0; i < experts; i++) {
            for (size_t j = 0; j < k; j++) {
                size_t rowIdx = i * k + j;
                size_t packedOffset = rowIdx * rowSize;
                size_t currentPos = packedOffset;
                    
                // 复制当前block的fp8数据
                for (size_t l = 0; l < m; l++) {
                    size_t srcIdx = i * k * m + j * m + l;
                    fp8Packed[currentPos++] = fp8[srcIdx];
                }
                    
                // 在这个block后面添加对应的scale
                // scale的索引计算：需要根据当前block在整个矩阵中的位置
                size_t scaleRow = j / blockK;  // 当前行属于哪个scale行块
                size_t scaleCol = 0;
                size_t scaleIdx = i * ks * ms + scaleRow * ms + scaleCol;
                    
                float* scalePtr = (float*)(&fp8Packed[currentPos]);
                *scalePtr = scales[scaleIdx];
                currentPos += sizeof(float);
            }
        }
    }

    void Int4ToFastllmInt4PerchannelRow(uint8_t *newWeight, uint8_t *oldWeight, int m) {
        if (GetCPUInstructInfo()->hasAVX512VNNI) {
            uint8_t *temp = new uint8_t[64];
            uint8_t *repack = new uint8_t[64];
            for (int i = 0; i < m; i += 64) {
                int len = std::min(m - i, 64);
                if (len == 64) {
                    for (int k = 0; k < 32; k++) {
                        temp[k * 2] = oldWeight[i / 2 + k] >> 4;
                        temp[k * 2 + 1] = oldWeight[i / 2 + k] & 0xF;
                    }
                            
                    for (int k = 0; k < 32; k++) {
                        repack[k * 2 + 1] = temp[k];
                        repack[k * 2] = temp[k + 32];
                    }

                    for (int k = 0; k < 32; k++) {
                        newWeight[i / 2 + k] = (repack[k * 2] << 4) + (repack[k * 2 + 1]);
                    }
                } else {
                    memcpy(newWeight + i / 2, oldWeight + i / 2, len / 2);
                }
            }
            delete[] temp;
            delete[] repack;
        } else if (GetCPUInstructInfo()->hasAVX2) {
            uint8_t *temp = new uint8_t[32];
            uint8_t *repack = new uint8_t[32];
            for (int i = 0; i < m; i += 32) {
                int len = std::min(m - i, 32);
                if (len == 32) {
                    for (int k = 0; k < 16; k++) {
                        temp[k * 2] = oldWeight[i / 2 + k] >> 4;
                        temp[k * 2 + 1] = oldWeight[i / 2 + k] & 0xF;
                    }
                            
                    for (int k = 0; k < 16; k++) {
                        repack[k * 2 + 1] = temp[k];
                        repack[k * 2] = temp[k + 16];
                    }

                    for (int k = 0; k < 16; k++) {
                        newWeight[i / 2 + k] = (repack[k * 2] << 4) + (repack[k * 2 + 1]);
                    }
                } else {
                    memcpy(newWeight + i / 2, oldWeight + i / 2, len / 2);
                }
            }
            delete[] temp;
            delete[] repack;
        } else {
            memcpy(newWeight, oldWeight, m / 2);
        }
    }

    void Int4ToFastllmInt4PerchannelPacked(int experts, int n, int m, uint8_t *qweight, float *mins, float *scales, std::vector <uint8_t> &int4Packed) {
        // 每行需要的字节数：m个uint4需要m/2个uint8，加上一个min和一个scale
        int int4BytesPerRow = m / 2;  // m是偶数，所以直接除以2
        int rowSize = int4BytesPerRow + sizeof(float) * 2;  // m/2个uint8 + 1个min + 1个scale
        
        // 调整输出vector大小
        int4Packed.resize((size_t)experts * n * rowSize);
        
        for (size_t i = 0; i < experts; i++) {
            for (size_t j = 0; j < n; j++) {
                size_t rowIdx = i * n + j;
                size_t packedOffset = rowIdx * rowSize;
                
                size_t currentPos = packedOffset;
                
                // 直接使用当前行的int4数据
                size_t srcOffset = i * n * int4BytesPerRow + j * int4BytesPerRow;
                Int4ToFastllmInt4PerchannelRow(&int4Packed[currentPos], &qweight[srcOffset], m);
                currentPos += int4BytesPerRow;
                
                // 添加当前行的min值
                float* minPtr = (float*)(&int4Packed[currentPos]);
                *minPtr = mins[rowIdx];
                currentPos += sizeof(float);
                
                // 添加当前行的scale值
                float* scalePtr = (float*)(&int4Packed[currentPos]);
                *scalePtr = scales[rowIdx];
                currentPos += sizeof(float);
            }
        }
    }

    void Int8ToFastllmInt8PerchannelPacked(int experts, int n, int m, uint8_t *qweight, int *zeros, float *scales, std::vector <uint8_t> &int8Packed) {
        int int8BytesPerRow = m;
        int rowSize = int8BytesPerRow + sizeof(float) * 2;  // m个uint8 + 1个min + 1个scale
        
        // 调整输出vector大小
        int8Packed.resize((size_t)experts * n * rowSize);
        
        for (size_t i = 0; i < experts; i++) {
            for (size_t j = 0; j < n; j++) {
                size_t rowIdx = i * n + j;
                size_t packedOffset = rowIdx * rowSize;
                
                size_t currentPos = packedOffset;
                
                // 直接使用当前行的int4数据
                size_t srcOffset = i * n * int8BytesPerRow + j * int8BytesPerRow;
                memcpy(&int8Packed[currentPos], &qweight[srcOffset], m);
                currentPos += int8BytesPerRow;
                
                // 添加当前行的min值
                float* minPtr = (float*)(&int8Packed[currentPos]);
                *minPtr = ((float)0 - zeros[rowIdx]) * scales[rowIdx];
                currentPos += sizeof(float);
                
                // 添加当前行的scale值
                float* scalePtr = (float*)(&int8Packed[currentPos]);
                *scalePtr = scales[rowIdx];
                currentPos += sizeof(float);
            }
        }
    }

    struct FastllmMoeDataManagerNumas {
            std::vector <float, alignedAllocator<float, 64> > gateUpOutput, swigluOutput, downOutput, reduceOutput;
            std::vector <uint8_t, alignedAllocator<uint8_t, 64> > realInput, expandInput, downInput;
    } fastllmMoeDataManagerNumas;

    void RegisterNumas(fastllm::Data *data) {
        data->Repack();
        auto *numaConfig = GetNumaConfig();
        if (data == nullptr) {
            return;
        }
        if (data->numasData.size() == 0) {
            data->numasData.resize(numaConfig->numaCnt);

            int k = data->dims[0], m = data->dims[1];
            if (k % numaConfig->numaCnt != 0) {
                ErrorInFastLLM("Linear weight's size %% numaCnt != 0.");
            }
            int kPerNuma = k / numaConfig->numaCnt;

            if (data->dataType == DataType::FLOAT32 || data->dataType == DataType::BFLOAT16 || data->dataType == DataType::FLOAT16) {
                size_t bytesPerRow = GetDataBytes(data->dataType, 1, m);
                for (int i = 0; i < numaConfig->numaCnt; i++) {
                    data->numasData[i] = (uint8_t*)allocate_aligned_numa(kPerNuma * bytesPerRow, i);
                    memcpy(data->numasData[i], data->cpuData + (size_t)i * kPerNuma * bytesPerRow, kPerNuma * bytesPerRow);
                }
            } else if (data->dataType == DataType::FP8_E4M3) {
                std::vector <uint8_t> fp8Packed;
                if (data->blockM == 128) {
                    data->dataType = DataType::FP8_E4M3_BLOCK_128;
                    Fp8ToFastllmFP8_E4M3_BLOCK128(1, k, m, (uint8_t*)data->cpuData, data->scales.data(), data->blockK, data->blockM, fp8Packed);
                } else if (data->blockM == m) {
                    data->dataType = DataType::FP8_E4M3_PERCHANNEL;
                    Fp8PerchannelToFastllmFP8_E4M3_PERCHANNEL(1, k, m, (uint8_t*)data->cpuData, data->scales.data(), data->blockK, data->blockM, fp8Packed);
                } else {
                    ErrorInFastLLM("RegisterNumas can't support fp8 with blockM = " + std::to_string(data->blockM));    
                }

                size_t bytesPerRow = GetDataBytes(data->dataType, 1, m);
                for (int i = 0; i < numaConfig->numaCnt; i++) {
                    data->numasData[i] = (uint8_t*)allocate_aligned_numa(kPerNuma * bytesPerRow, i);
                    memcpy(data->numasData[i], fp8Packed.data() + (size_t)i * kPerNuma * bytesPerRow, kPerNuma * bytesPerRow);
                }
            } else if (data->dataType == DataType::INT8) {
                std::vector <uint8_t> int8Packed;
                Int8ToFastllmInt8PerchannelPacked(1, k, m, (uint8_t*)data->cpuData, data->zeros.data(), data->scales.data(), int8Packed);
                data->dataType = DataType::INT8_PERCHANNEL;

                size_t bytesPerRow = GetDataBytes(data->dataType, 1, m);
                for (int i = 0; i < numaConfig->numaCnt; i++) {
                    data->numasData[i] = (uint8_t*)allocate_aligned_numa(kPerNuma * bytesPerRow, i);
                    memcpy(data->numasData[i], int8Packed.data() + (size_t)i * kPerNuma * bytesPerRow, kPerNuma * bytesPerRow);
                }
            } else if (data->dataType == DataType::INT4_NOZERO) {
                std::vector <uint8_t> int4Packed;
                Int4ToFastllmInt4PerchannelPacked(1, k, m, (uint8_t*)data->cpuData, data->mins.data(), data->scales.data(), int4Packed);
                data->dataType = DataType::INT4_PERCHANNEL;

                size_t bytesPerRow = GetDataBytes(data->dataType, 1, m);
                for (int i = 0; i < numaConfig->numaCnt; i++) {
                    data->numasData[i] = (uint8_t*)allocate_aligned_numa(kPerNuma * bytesPerRow, i);
                    memcpy(data->numasData[i], int4Packed.data() + (size_t)i * kPerNuma * bytesPerRow, kPerNuma * bytesPerRow);
                }
            } else if (data->dataType == DataType::INT4_GROUP) {
                if (m % data->groupCnt > 0) {
                    ErrorInFastLLM("RegisterNumas can't support data type int4g when m % groupCnt > 0.");
                }

                int groups = m / data->groupCnt;
                std::vector <uint8_t> int4Packed;
                Int4ToFastllmInt4PerchannelPacked(1, k * groups, data->groupCnt, (uint8_t*)data->cpuData, data->mins.data(), data->scales.data(), int4Packed);

                if (data->groupCnt == 128) {                    
                    data->dataType = DataType::INT4_GROUP128;
                } else {
                    ErrorInFastLLM("RegisterNumas can't support data type " + GetDataTypeName(data->dataType));
                }

                size_t bytesPerRow = GetDataBytes(data->dataType, 1, m);
                for (int i = 0; i < numaConfig->numaCnt; i++) {
                    data->numasData[i] = (uint8_t*)allocate_aligned_numa(kPerNuma * bytesPerRow, i);
                    memcpy(data->numasData[i], int4Packed.data() + (size_t)i * kPerNuma * bytesPerRow, kPerNuma * bytesPerRow);
                }
            } else if (data->dataType == DataType::DATA_GGUF_FORMAT) {
                size_t bytesPerRow = GetDataBytes((DataType)((int)data->dataType + data->ggmlType), 1, m);
                for (int i = 0; i < numaConfig->numaCnt; i++) {
                    data->numasData[i] = (uint8_t*)allocate_aligned_numa(kPerNuma * bytesPerRow, i);
                    memcpy(data->numasData[i], data->cpuData + (size_t)i * kPerNuma * bytesPerRow, kPerNuma * bytesPerRow);
                }
            } else {
                ErrorInFastLLM("RegisterNumas can't support data type " + GetDataTypeName(data->dataType));
            }
        }

        delete[] data->cpuData;
        data->cpuData = nullptr;
    }

    struct NumaWorkStealingOp : MultiThreadBaseOp {
        struct alignas(64) TaskState {
            std::atomic<int> curr;
            int end;
            std::vector<MultiThreadBaseOp*> tasks;
            std::atomic<bool> completed;
        };
        
        int threadId;
        int numaId;
        std::vector<TaskState*>* allStates;
        TaskState* myState;
        NumaConfig* numaConfig;
        
        NumaWorkStealingOp(int tid, int nid, std::vector<TaskState*>* states, 
                      TaskState* state, NumaConfig* config) 
            : threadId(tid), numaId(nid), allStates(states), 
              myState(state), numaConfig(config) {}
        
        void Run() override {
            // 首先执行自己的任务
            processOwnTasks();
            
            // 然后从同一NUMA节点的其他线程偷取任务
            stealFromSameNuma();
            
            // 标记完成
            myState->completed.store(true, std::memory_order_release);
        }
        
    private:
        void processOwnTasks() {
            while (true) {
                int taskId = myState->curr.fetch_add(1, std::memory_order_acq_rel);
                if (taskId >= myState->end) {
                    break;
                }
                if (taskId < myState->tasks.size()) {
                    myState->tasks[taskId]->Run();
                }
            }
        }
        
        void stealFromSameNuma() {
            // 获取同一NUMA节点的所有线程
            auto& numaThreads = numaConfig->numaToCpuDict[numaId];
            
            // 利用连续性计算位置：当前线程ID - NUMA节点第一个线程ID
            int numaStartThread = numaThreads[0].first;
            int myPos = threadId - numaStartThread;
            
            // 从当前线程开始，环形遍历其他线程
            for (int offset = 1; offset < numaThreads.size(); offset++) {
                int targetPos = (myPos + offset) % numaThreads.size();
                int tid = numaThreads[targetPos].first;
                
                TaskState* otherState = (*allStates)[tid];
                if (otherState == nullptr) continue;
                
                // 检查是否还有任务可偷
                while (true) {
                    int taskId = otherState->curr.fetch_add(1, std::memory_order_acq_rel);
                    if (taskId >= otherState->end) {
                        break;
                    }
                    if (taskId < otherState->tasks.size()) {
                        otherState->tasks[taskId]->Run();
                    }
                }
            }
        }
    };
    
    // 重构的动态任务调度函数，支持work-stealing
    void DynamicScheduleTasks(std::vector<std::vector<MultiThreadBaseOp*>>& ops) {
        auto *pool = GetAlivePool();
        auto *numaConfig = GetNumaConfig();
        
        // 创建任务状态数组
        using TaskState = typename NumaWorkStealingOp::TaskState;
        std::vector<TaskState*> taskStates(numaConfig->threads, nullptr);
        
        // 为每个线程分配任务状态
        for (int i = 0; i < numaConfig->threads; i++) {
            taskStates[i] = new (std::align_val_t{64}) TaskState();
            taskStates[i]->curr.store(0, std::memory_order_relaxed);
            taskStates[i]->end = 0;
            taskStates[i]->completed.store(false, std::memory_order_relaxed);
        }
        
        // 分配任务到各个线程
        int totalOps = 0;
        for (int nid = 0; nid < numaConfig->numaCnt; nid++) {
            totalOps += ops[nid].size();
            
            if (ops[nid].empty()) continue;
            
            int threadNum = numaConfig->numaToCpuDict[nid].size();
            if (threadNum == 0) continue;
            
            // 计算每个线程的任务数量
            int tasksPerThread = ops[nid].size() / threadNum;
            int remainingTasks = ops[nid].size() % threadNum;
            
            int taskIndex = 0;
            for (int i = 0; i < threadNum; i++) {
                int tid = numaConfig->numaToCpuDict[nid][i].first;
                int numTasks = tasksPerThread + (i < remainingTasks ? 1 : 0);
                
                if (numTasks > 0) {
                    // 分配任务到该线程
                    taskStates[tid]->tasks.clear();
                    taskStates[tid]->tasks.reserve(numTasks);
                    
                    for (int j = 0; j < numTasks && taskIndex < ops[nid].size(); j++) {
                        taskStates[tid]->tasks.push_back(ops[nid][taskIndex++]);
                    }
                    
                    taskStates[tid]->curr.store(0, std::memory_order_relaxed);
                    taskStates[tid]->end = taskStates[tid]->tasks.size();
                } else {
                    taskStates[tid]->end = 0;
                }
            }
        }
        
        // 创建work-stealing ops并提交到线程池
        std::vector<NumaWorkStealingOp*> wsOps(numaConfig->threads);
        for (int i = 0; i < numaConfig->threads; i++) {
            int numaId = numaConfig->threadIdToNumaDict[i];
            wsOps[i] = new NumaWorkStealingOp (
                i, numaId, &taskStates, taskStates[i], numaConfig
            );
            
            // 只有有任务的线程才启动
            if (taskStates[i] != nullptr && taskStates[i]->end > 0) {
                pool->PushOp(i, wsOps[i]);
            } else {
                // 没有任务的线程也要启动，以便参与work-stealing
                taskStates[i]->completed.store(true, std::memory_order_release);
                pool->PushOp(i, wsOps[i]);
            }
        }
        // 等待所有线程完成
        for (int i = 0; i < numaConfig->threads; i++) {
            pool->Wait(i);
        }
        
        // 清理资源
        for (int i = 0; i < numaConfig->threads; i++) {
            delete wsOps[i];
            if (taskStates[i] != nullptr) {
                taskStates[i]->~TaskState();
                #if __cpp_aligned_new >= 201606
                    operator delete(taskStates[i], std::align_val_t{64});
                #else
                    free_aligned(taskStates[i], sizeof(TaskState));
                #endif
            }
        }
        
        // 删除原始ops
        for (int nid = 0; nid < numaConfig->numaCnt; nid++) {
            for (auto* op : ops[nid]) {
                delete op;
            }
        }
    }

    void NumasMergeMOE::Run(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        fastllm::BaseOperator *op = (fastllm::BaseOperator*)(new CpuLinearOp());
 // auto ttt = std::chrono::system_clock::now();
 // std::vector <std::pair <std::string, float> > record;
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &gateBias = *(datas.find("gateBias")->second);
        Data &logits = *(datas.find("logits")->second);
        Data &w1 = *(datas.find("w1")->second);
        Data &w2 = *(datas.find("w2")->second);
        Data &w3 = *(datas.find("w3")->second);
        Data **weights = (Data**)(datas.find("weights")->second);
        Data **biass = (Data**)(datas.find("biass")->second);
        int topk = intParams.find("topk") != intParams.end() ? intParams.find("topk")->second : 1;
        int needNorm = intParams.find("needNorm") != intParams.end() ? intParams.find("needNorm")->second : 0;
        float sharedScale = floatParams.find("sharedScale") != floatParams.end() ? floatParams.find("sharedScale")->second : 1.0f;        
        float routeScale = floatParams.find("routeScale") != floatParams.end() ? floatParams.find("routeScale")->second : 1.0f;        
        output.Allocate();

        if (input.dims[0] < 32) {
auto st = std::chrono::system_clock::now();
            ToDataType(logits, DataType::FLOAT32);
            logits.ToDevice(DataDevice::CPU);
            float *cpuRouterLogits = (float*)logits.cpuData;
            int m = logits.dims.back();

            {
                auto *pool = GetAlivePool();

                int bs = input.dims[0];
                int inputDim = input.dims[1];
                int interDim = weights[2]->dims[0] / 2;
                int outputDim = output.dims[1];
                float *floatLogits = ((float*)logits.cpuData);

                for (int o = 0; o < bs; o++) {
                    std::vector <std::pair <float, int> > oriV;
                    oriV.resize(m);
                    for (int j = 0; j < m; j++) {
                        oriV[j].first = -floatLogits[o * m + j];
                        oriV[j].second = j;
                    }
                    if (gateBias.dims.size() > 0) {
                        if (gateBias.dataType != DataType::FLOAT32) {
                            ToDataType(gateBias, DataType::FLOAT32);
                        }
                        float *cpuBias = (float*)gateBias.cpuData;
                        for (int i = 0; i < m; i++) {
                            oriV[i].first -= cpuBias[i];
                        }
                    }
                    std::partial_sort(oriV.begin(), oriV.begin() + topk, oriV.end());
                    float sum = 1.0;
                    if (needNorm) {
                        sum = 0.0;
                        for (int j = 0; j < topk; j++) {
                            sum += floatLogits[o * m + oriV[j].second];
                        }
                    }
                    std::vector <std::pair <int, float> > v;
                    for (int j = 0; j < topk; j++) {
                        v.push_back(std::make_pair(oriV[j].second + 1, floatLogits[o * m + oriV[j].second] / sum * routeScale));
                    }
                    if (weights[0] != nullptr) {
                        v.push_back(std::make_pair(0, sharedScale));
                    }

                    DataType startDataType = weights[2]->GetLinearActDataType(1);
                    DataType downInputDataType = weights[3]->GetLinearActDataType(1);

                    // 从 fastllmMoeDataManagerNumas 获取缓存的 vector，并根据需要调整大小
                    auto& realInput = fastllmMoeDataManagerNumas.realInput;
                    auto& gateUpOutput = fastllmMoeDataManagerNumas.gateUpOutput;
                    auto& swigluOutput = fastllmMoeDataManagerNumas.swigluOutput;
                    auto& downInput = fastllmMoeDataManagerNumas.downInput;
                    auto& downOutput = fastllmMoeDataManagerNumas.downOutput;
                    auto& reduceOutput = fastllmMoeDataManagerNumas.reduceOutput;

                    // 计算所需大小
                    size_t realInputSize = GetDataBytes(startDataType, 1, inputDim);
                    size_t gateUpOutputSize = v.size() * interDim * 2;
                    size_t swigluOutputSize = v.size() * interDim;
                    size_t downInputSize = GetDataBytes(downInputDataType, v.size(), interDim);
                    size_t downOutputSize = v.size() * outputDim;
                    size_t reduceOutputSize = 1 * outputDim;

                    // 只在当前容量不足时才进行 resize
                    if (realInput.size() < realInputSize) {
                        realInput.resize(realInputSize);
                    }
                    if (gateUpOutput.size() < gateUpOutputSize) {
                        gateUpOutput.resize(gateUpOutputSize);
                    }
                    if (swigluOutput.size() < swigluOutputSize) {
                        swigluOutput.resize(swigluOutputSize);
                    }
                    if (downInput.size() < downInputSize) {
                        downInput.resize(downInputSize);
                    }
                    if (downOutput.size() < downOutputSize) {
                        downOutput.resize(downOutputSize);
                    }
                    if (reduceOutput.size() < reduceOutputSize) {
                        reduceOutput.resize(reduceOutputSize);
                    }

// printf("malloc spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                    // 0. input -> realInput
                    RunMultiThreadConvertFromFloat32(realInput.data(), startDataType, (float*)input.cpuData + o * inputDim, 1, inputDim, GetAlivePool());
// printf("RunMultiThreadConvertFromFloat32 spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));

                    // 1. gateUp
                    auto *numaConfig = GetNumaConfig();
                    std::vector<MultiThreadBaseOp*> ops;
                    ops.resize(numaConfig->threads);
                    for (int i = 0; i < ops.size(); i++) {
                        ops[i] = new MultiThreadMultiOps();
                    }

                    int totalExperts = v.size();
                    int k = interDim * 2;
                    int kPer = k / numaConfig->numaCnt;

                    for (int nid = 0; nid < numaConfig->numaCnt; nid++) {
                        int base = kPer * nid;
                        int threadNum = numaConfig->numaToCpuDict[nid].size();
                        
                        // 计算该NUMA节点上所有专家的总行数
                        int totalRows = kPer * totalExperts;
                        int unitRows = 4;
                        int rowsPerThread = (totalRows / unitRows) / threadNum;
                        int remainingRows = (totalRows / unitRows) % threadNum;
                        
                        int currentRow = 0;
                        
                        for (int tid = 0; tid < threadNum; tid++) {
                            int threadRows = (rowsPerThread + (tid < remainingRows ? 1 : 0)) * unitRows;
                            int endRow = currentRow + threadRows;
                            
                            // 处理当前线程负责的行范围
                            int rowStart = currentRow;
                            while (rowStart < endRow) {
                                // 确定当前行属于哪个专家
                                int expertIdx = rowStart / kPer;
                                if (expertIdx >= totalExperts) break;
                                
                                int e = v[expertIdx].first;
                                
                                // 计算在当前专家内的起始行和结束行
                                int expertStartRow = rowStart % kPer;
                                int expertEndRow = std::min(kPer, expertStartRow + (endRow - rowStart));
                                
                                // 计算输出偏移
                                size_t outputOffset = GetDataBytes(DataType::FLOAT32, expertIdx, k) + 
                                                    GetDataBytes(DataType::FLOAT32, 1, base);
                                
                                // 添加GEMM操作
                                ((MultiThreadMultiOps*)ops[numaConfig->numaToCpuDict[nid][tid].first])->ops.push_back(
                                    new MultiThreadGemmOp(
                                        (uint8_t*)realInput.data(), startDataType,
                                        weights[e * 2]->numasData[nid], weights[e * 2]->GetDataType(),
                                        (uint8_t*)gateUpOutput.data() + outputOffset, DataType::FLOAT32,
                                        1, inputDim, k, expertStartRow, expertEndRow
                                    )
                                );
                                
                                // 更新到下一个处理段
                                rowStart += (expertEndRow - expertStartRow);
                                if (expertEndRow == kPer) {
                                    // 如果当前专家处理完了，移动到下一个专家的起始位置
                                    rowStart = (expertIdx + 1) * kPer;
                                }
                            }
                            
                            currentRow = endRow;
                        }
                    }
// printf("gateup prepare spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                    for (int i = 0; i < ops.size(); i++) {
                        pool->PushOp(i, ops[i]);
                    }

                    for (int i = 0; i < ops.size(); i++) {
                        pool->Wait(i);
                        delete ops[i];
                    }


// printf("gateup spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                    // 3. swiglu
                    SwigluMultiThread((float *) gateUpOutput.data(), interDim, interDim, ((float *) swigluOutput.data()),
                                    v.size(), interDim * 2, interDim, GetAlivePool());
// printf("swiglu spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));

                    // 4. swigluOutput -> downInput
                    RunMultiThreadConvertFromFloat32(downInput.data(), downInputDataType, (float*)swigluOutput.data(), v.size(), interDim, GetAlivePool());

// printf("RunMultiThreadConvertFromFloat32 spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                    // 5. down
                    ops.resize(numaConfig->threads);
                    for (int i = 0; i < ops.size(); i++) {
                        ops[i] = new MultiThreadMultiOps();
                    }
                    totalExperts = v.size();
                    k = outputDim;
                    kPer = k / numaConfig->numaCnt;
                    for (int nid = 0; nid < numaConfig->numaCnt; nid++) {
                        int base = kPer * nid;
                        int threadNum = numaConfig->numaToCpuDict[nid].size();
                        
                        // 总共需要处理的行数：每个专家kPer行，共totalExperts个专家
                        int totalRows = kPer * totalExperts;
                        int unitRows = 4;
                        int rowsPerThread = (totalRows / unitRows) / threadNum;
                        int extraRows = (totalRows / unitRows) % threadNum;
                        
                        int currentRow = 0;
                        for (int tid = 0; tid < threadNum; tid++) {
                            int threadRows = (rowsPerThread + (tid < extraRows ? 1 : 0)) * unitRows;
                            int endRow = currentRow + threadRows;
                            
                            // 处理这个线程负责的行范围 [currentRow, endRow)
                            int startExpert = currentRow / kPer;
                            int startRowInExpert = currentRow % kPer;
                            
                            for (int row = currentRow; row < endRow; ) {
                                int expertIdx = row / kPer;
                                int rowInExpert = row % kPer;
                                
                                // 计算当前专家中要处理的行范围
                                int rowsToProcess = std::min(kPer - rowInExpert, endRow - row);
                                if (expertIdx < totalExperts) {
                                    int e = v[expertIdx].first;
                                    size_t inputOffset = expertIdx * GetDataBytes(downInputDataType, 1, interDim);
                                    size_t outputOffset = GetDataBytes(DataType::FLOAT32, expertIdx, k) + 
                                                        GetDataBytes(DataType::FLOAT32, 1, base);
                                    
                                    ((MultiThreadMultiOps*)ops[numaConfig->numaToCpuDict[nid][tid].first])->ops.push_back(
                                        new MultiThreadGemmOp(
                                            (uint8_t*)downInput.data() + inputOffset, downInputDataType,
                                            weights[e * 2 + 1]->numasData[nid], weights[e * 2 + 1]->GetDataType(),
                                            (uint8_t*)downOutput.data() + outputOffset, DataType::FLOAT32,
                                            1, interDim, k, rowInExpert, rowInExpert + rowsToProcess
                                        )
                                    );
                                }
                                
                                row += rowsToProcess;
                            }
                            
                            currentRow = endRow;
                        }
                    }

// printf("down prepare spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                    for (int i = 0; i < ops.size(); i++) {
                        pool->PushOp(i, ops[i]);
                    }
                    for (int i = 0; i < ops.size(); i++) {
                        pool->Wait(i);
                        delete ops[i];
                    }

// printf("down spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                    float *fLastOutput = reduceOutput.data();
                    if (output.dataType == DataType::FLOAT32) {
                        fLastOutput = ((float*)output.cpuData) + o * outputDim;
                    }

                    // 6. reduce
                    for (int i = 0; i < v.size(); i++) {
                        float value = v[i].second;
                        float *curOutput = ((float*)downOutput.data()) + i * outputDim;
                        if (i == 0) {
                            for (int j = 0; j < outputDim; j++) {
                                fLastOutput[j] = curOutput[j] * value;    
                            }
                        } else {
                            for (int j = 0; j < outputDim; j++) {
                                fLastOutput[j] += curOutput[j] * value;
                            }
                        }
                    }

// printf("reduce spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                    // 7. reduceOutput -> last Output
                    if (output.dataType != DataType::FLOAT32) {
                        if (output.dataType == DataType::FLOAT16) {
                            Float32ToFloat16(reduceOutput.data(), ((uint16_t*)output.cpuData) + o * outputDim, output.Count(0));
                        }
                    }
// printf("last spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                }
            }
        } else {
            auto st = std::chrono::system_clock::now();
            Data gate, attenPart, moePart;
            ToDataType(logits, DataType::FLOAT32);
            logits.ToDevice(DataDevice::CPU);
            float *cpuRouterLogits = (float*)logits.cpuData;
            int m = logits.dims.back();

            {
                auto *pool = GetAlivePool();

                int bs = input.dims[0], dim = output.dims[1];
                int inputDim = input.dims[1];
                int interDim = weights[2]->dims[0] / 2;
                int outputDim = output.dims[1];
                std::vector <std::pair <float, int> > v; // (value, idx)
                v.resize(m);

                std::vector <std::vector <std::pair <int, float> > > expertTasks; // expertTasks[i]代表专家i的task, expertTasks[i][j] = (第j个任务对应的行数， 权重)
                expertTasks.resize(m + 1);
                for (int b = 0; b < bs; b++) {
                    expertTasks[0].push_back(std::make_pair(b, sharedScale));
                    float *cur = cpuRouterLogits + b * m;
                    for (int i = 0; i < m; i++) {
                        v[i] = (std::make_pair(-cur[i], i));
                    }
                    if (gateBias.dims.size() > 0) {
                        ToDataType(gateBias, DataType::FLOAT32);
                        gateBias.ToDevice(DataDevice::CPU);
                        float *cpuBias = (float*)gateBias.cpuData;
                        for (int i = 0; i < m; i++) {
                            v[i].first -= cpuBias[i];
                        }
                    }
                    // sort(v.begin(), v.end());
                    partial_sort(v.begin(), v.begin() + topk, v.end());
                    float sum = 1.0;
                    if (needNorm) {
                        sum = 0.0;
                        for (int j = 0; j < topk; j++) {
                            sum += cur[v[j].second];
                        }
                    }
                    
                    for (int j = 0; j < topk; j++) {
                        int idx = v[j].second;
                        float value = cur[idx] / sum * routeScale;
                        expertTasks[idx + 1].push_back(std::make_pair(b, value));
                    }
                }

                int totalLines = 0;
                for (int e = 0; e < expertTasks.size(); e++) {
                    if (weights[e * 2] != nullptr) {
                        totalLines += expertTasks[e].size();
                    }
                }
// printf("prepare spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                DataType startDataType = weights[2]->GetLinearActDataType(bs);
                DataType downInputDataType = weights[3]->GetLinearActDataType(bs);

                // 从 fastllmMoeDataManagerNumas 获取缓存的 vector，并根据需要调整大小
                auto& realInput = fastllmMoeDataManagerNumas.realInput;
                auto& expandInput = fastllmMoeDataManagerNumas.expandInput;
                auto& gateUpOutput = fastllmMoeDataManagerNumas.gateUpOutput;
                auto& swigluOutput = fastllmMoeDataManagerNumas.swigluOutput;
                auto& downInput = fastllmMoeDataManagerNumas.downInput;
                auto& downOutput = fastllmMoeDataManagerNumas.downOutput;
                auto& reduceOutput = fastllmMoeDataManagerNumas.reduceOutput;

                int alignTotalLines = ((totalLines - 1) / 64 + 1) * 64;
                // 计算所需大小
                size_t realInputSize = GetDataBytes(startDataType, bs, inputDim);
                size_t expandInputSize = GetDataBytes(startDataType, alignTotalLines, inputDim);
                size_t gateUpOutputSize = alignTotalLines * interDim * 2;
                size_t swigluOutputSize = alignTotalLines * interDim;
                size_t downInputSize = GetDataBytes(downInputDataType, alignTotalLines, interDim);
                size_t downOutputSize = alignTotalLines * outputDim;
                size_t reduceOutputSize = bs * outputDim;

                // 只在当前容量不足时才进行 resize
                if (realInput.size() < realInputSize) {
                    realInput.resize(realInputSize);
                }
                if (expandInput.size() < expandInputSize) {
                    expandInput.resize(expandInputSize);
                }
                if (gateUpOutput.size() < gateUpOutputSize) {
                    gateUpOutput.resize(gateUpOutputSize);
                }
                if (swigluOutput.size() < swigluOutputSize) {
                    swigluOutput.resize(swigluOutputSize);
                }
                if (downInput.size() < downInputSize) {
                    downInput.resize(downInputSize);
                }
                if (downOutput.size() < downOutputSize) {
                    downOutput.resize(downOutputSize);
                }
                if (reduceOutput.size() < reduceOutputSize) {
                    reduceOutput.resize(reduceOutputSize);
                }

// printf("malloc spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                // 0. input -> realInput
                RunMultiThreadConvertFromFloat32(realInput.data(), startDataType, (float*)input.cpuData, bs, inputDim, GetAlivePool());
// printf("RunMultiThreadConvertFromFloat32 spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));

                // 1. realInput -> expandInput
                std::vector <MultiThreadMemcpyMultiLinesTask> memcpyTasks;
                memcpyTasks.resize(totalLines);
                {
                    int offset = 0;
                    uint8_t* realInputPtr = realInput.data();
                    uint8_t* expandInputPtr = expandInput.data();
                    int bytesPerLine = GetDataBytes(startDataType, 1, inputDim);
                    
                    for (int e = 0; e < expertTasks.size(); e++) {
                        if (weights[e * 2] != nullptr) {
                            for (auto& task : expertTasks[e]) {
                                int rowIdx = task.first;

                                memcpyTasks[offset] = MultiThreadMemcpyMultiLinesTask(
                                    expandInputPtr + offset * bytesPerLine, 
                                    realInputPtr + rowIdx * bytesPerLine, 
                                    bytesPerLine
                                );
                                offset++;
                            }
                        }
                    }
                }
                RunMultiThreadMemcpyMultiLines(memcpyTasks, GetAlivePool());
// printf("expand spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                // 2. gateUp
                
                auto *numaConfig = GetNumaConfig();

                int offset = 0;
                int stride = 64;

                std::vector<std::vector <fastllm::MultiThreadBaseOp*> > ops;
                ops.resize(numaConfig->numaCnt);

                for (int e = 0; e < expertTasks.size(); e++) {
                    if (weights[e * 2] != nullptr && expertTasks[e].size() > 0) {
                        int lines = expertTasks[e].size();

                        // Prepare input pointer for this expert's batch
                        uint16_t* expertInputPtr = (uint16_t*)(expandInput.data() + offset * GetDataBytes(startDataType, 1, inputDim));
                            
                        // Prepare output pointer for this expert's batch
                        float* expertGateUpOutputPtr = gateUpOutput.data() + offset * interDim * 2;

                        int k = interDim * 2;
                        int kPer = k / numaConfig->numaCnt;
                            
                        for (int nid = 0; nid < numaConfig->numaCnt; nid++) {
                            // Get weight data (assuming weights are stored as `startDataType`)
                            int base = kPer * nid;
                            size_t outputOffset = GetDataBytes(DataType::FLOAT32, 1, base);

                            for (int st = 0; st < kPer; st += stride) {
                                int end = std::min(st + stride, kPer);
                                ops[nid].push_back(new MultiThreadGemmOp(
                                    (uint8_t*)expertInputPtr, startDataType,
                                    weights[e * 2]->numasData[nid], weights[e * 2]->GetDataType(),
                                    (uint8_t*)expertGateUpOutputPtr + outputOffset, DataType::FLOAT32,
                                    lines, inputDim, k, st, end
                                ));
                            }
                        }
                        offset += lines;
                    }
                }

                DynamicScheduleTasks(ops);

// printf("gateup spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));

                // 3. swiglu
                SwigluMultiThread((float *) gateUpOutput.data(), interDim, interDim, ((float *) swigluOutput.data()),
                                    totalLines, interDim * 2, interDim, GetAlivePool());
// printf("swiglu spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));

                // 4. swigluOutput -> downInput
                RunMultiThreadConvertFromFloat32(downInput.data(), downInputDataType, (float*)swigluOutput.data(), totalLines, interDim, GetAlivePool());

// printf("RunMultiThreadConvertFromFloat32 spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                // 5. down
                offset = 0;
                stride = 64;
                ops.resize(numaConfig->numaCnt);
                for (int i = 0; i < ops.size(); i++) {
                    ops[i].clear();
                }

                for (int e = 0; e < expertTasks.size(); e++) {
                    if (weights[e * 2 + 1] != nullptr && expertTasks[e].size() > 0) {
                        int lines = expertTasks[e].size();

                        // Prepare input pointer for this expert's batch
                        uint16_t* expertDownInputPtr = (uint16_t*)(downInput.data() + offset * GetDataBytes(downInputDataType, 1, interDim));
                            
                        // Prepare output pointer for this expert's batch
                        float* expertDownOutputPtr = downOutput.data() + offset * dim;

                        int k = dim;
                        int kPer = k / numaConfig->numaCnt;
                            
                        for (int nid = 0; nid < numaConfig->numaCnt; nid++) {
                            // Get weight data (assuming weights are stored as `downInputDataType`)
                            int base = kPer * nid;
                            size_t outputOffset = GetDataBytes(DataType::FLOAT32, 1, base);

                            for (int st = 0; st < kPer; st += stride) {
                                int end = std::min(st + stride, kPer);
                                ops[nid].push_back(new MultiThreadGemmOp(
                                    (uint8_t*)expertDownInputPtr, downInputDataType,
                                    weights[e * 2 + 1]->numasData[nid], weights[e * 2 + 1]->GetDataType(),
                                    (uint8_t*)expertDownOutputPtr + outputOffset, DataType::FLOAT32,
                                    lines, interDim, k, st, end
                                ));
                            }
                        }
                        offset += lines;
                    }
                }

                DynamicScheduleTasks(ops);

// printf("down spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                // 6. reduce
                {
                    // 准备数据结构
                    int total_tasks = 0;
                    for (int e = 0; e < expertTasks.size(); e++) {
                        if (weights[e * 2] != nullptr) {
                            total_tasks += expertTasks[e].size();
                        }
                    }
                    // 假设每个样本最多选择k个专家
                    int k = 0; // 需要确定每个样本选择的专家数量
                    std::vector<int> samples_expert_count(bs, 0);
                    // 第一遍：统计每个样本的专家数量
                    for (int e = 0; e < expertTasks.size(); e++) {
                        if (weights[e * 2] != nullptr) {
                            for (auto& task : expertTasks[e]) {
                                int rowIdx = task.first;
                                samples_expert_count[rowIdx]++;
                                k = std::max(k, samples_expert_count[rowIdx]);
                            }
                        }
                    }
                    // 分配内存
                    std::vector<int> pos(bs * k, -1);  // 初始化为-1表示无效位置
                    std::vector<float> task_weights(total_tasks, 0.0f);
                    std::vector<int> sample_expert_idx(bs, 0);  // 记录每个样本当前填充到第几个专家
                    // 第二遍：填充pos和weights数组
                    int offset = 0;
                    for (int e = 0; e < expertTasks.size(); e++) {
                        if (weights[e * 2] != nullptr) {
                            for (auto& task : expertTasks[e]) {
                                int rowIdx = task.first;
                                float weight = task.second;
                                
                                // 在pos数组中记录这个任务的位置
                                int expert_idx = sample_expert_idx[rowIdx]++;
                                pos[rowIdx * k + expert_idx] = offset;
                                task_weights[offset] = weight;
                                
                                offset++;
                            }
                        }
                    }

                    // 调用多线程函数
                    MultiThreadReduceBatch(
                        (uint8_t*)downOutput.data(),  // downOutData
                        DataType::FLOAT32,             // downOutDataType (假设是float32)
                        task_weights.data(),           // weights
                        output.dataType == DataType::FLOAT32 ? (float*)output.cpuData : reduceOutput.data(),           // lastOutput
                        pos.data(),                    // pos
                        bs,                           // bsz
                        k,                            // k (每个样本的专家数)
                        dim                           // hidden_size
                    );
                    // 注意：如果某些样本的专家数少于k，需要特殊处理
                    // 可以在MultiThreadReduceBatchOp::Run()中添加检查：
                    // if (curPos == -1) continue; // 跳过无效位置
                }
// printf("reduce spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                // 7. reduceOutput -> last Output
                if (output.dataType != DataType::FLOAT32) {
                    if (output.dataType == DataType::FLOAT16) {
                        Float32ToFloat16(reduceOutput.data(), (uint16_t*)output.cpuData, output.Count(0));
                    }
                }
// printf("last spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
            }
        }
    }
}