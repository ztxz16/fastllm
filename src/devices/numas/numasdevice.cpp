//
// Created by huangyuyang on 10/15/24.
//

#include <sys/mman.h>
#include <fcntl.h>

#include "devices/numas/numasdevice.h"
#include "devices/cpu/cpudevice.h"
#include "devices/cpu/alivethreadpool.h"

#include <cstdlib>
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
    extern void Float32ToBFloat16(float *float32, uint16_t *bfloat16, int len);

    static void BFloat16ToFloat32(uint16_t *bfloat16, float *float32, int len) {
        for (int i = 0; i < len; i++) {
            uint32_t x = (uint32_t)bfloat16[i] << 16;
            float32[i] = *(float*)&x;
        }
    }

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
            std::vector <float, alignedAllocator<float, 64> > inputFloat32;  // 当 input 非 FLOAT32 时暂存转成 float32 的数据
            std::vector <uint8_t, alignedAllocator<uint8_t, 64> > realInput, expandInput, downInput;
    };
    // 每层一个 MoE 缓存，避免层间共享导致的数据竞争
    std::unordered_map<int, FastllmMoeDataManagerNumas> fastllmMoeDataManagerNumasPerLayer;

    // CrossSwiglu 重排：将 a[n, m] 重排为 b[n, m]
    // b[0] = a[0], b[1] = a[n/2], b[2] = a[1], b[3] = a[n/2+1], ...
    // 即前半部分和后半部分行交错排列
    void CrossSwigluReorder(uint8_t *src, int k, size_t bytesPerRow, std::vector<uint8_t> &dst) {
        dst.resize((size_t)k * bytesPerRow);
        int half = k / 2;
        for (int i = 0; i < half; i++) {
            memcpy(dst.data() + (size_t)(2 * i) * bytesPerRow, 
                   src + (size_t)i * bytesPerRow, bytesPerRow);
            memcpy(dst.data() + (size_t)(2 * i + 1) * bytesPerRow, 
                   src + (size_t)(half + i) * bytesPerRow, bytesPerRow);
        }
    }

    void RegisterNumas(fastllm::Data *data, std::string weightType) {
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

            bool isCrossSwiglu = (weightType == "linearSwiglu");

            if (data->dataType == DataType::DATA_GGUF_FORMAT) {
                // GGUF格式需要先交错存储，然后再repack
                // 因为repack会将多行打包在一起，打包后行之间的数据互相依赖，无法再做行交错
                size_t bytesPerRow = GetDataBytes((DataType)((int)data->dataType + data->ggmlType), 1, m);
                if (isCrossSwiglu) {
                    std::vector<uint8_t> reordered;
                    CrossSwigluReorder(data->cpuData, k, bytesPerRow, reordered);
                    memcpy(data->cpuData, reordered.data(), (size_t)k * bytesPerRow);
                }
                // 交错存储完成后再repack
                data->Repack();
                // repack后bytesPerRow可能变化，重新获取
                bytesPerRow = GetDataBytes((DataType)((int)data->dataType + data->ggmlType), 1, m);
                uint8_t *srcData = data->cpuData;
                for (int i = 0; i < numaConfig->numaCnt; i++) {
                    data->numasData[i] = (uint8_t*)allocate_aligned_numa(kPerNuma * bytesPerRow, i);
                    memcpy(data->numasData[i], srcData + (size_t)i * kPerNuma * bytesPerRow, kPerNuma * bytesPerRow);
                }
            } else {
                data->Repack();

                if (data->dataType == DataType::FLOAT32 || data->dataType == DataType::BFLOAT16 || data->dataType == DataType::FLOAT16) {
                    size_t bytesPerRow = GetDataBytes(data->dataType, 1, m);
                    uint8_t *srcData = data->cpuData;
                    std::vector<uint8_t> reordered;
                    if (isCrossSwiglu) {
                        CrossSwigluReorder(data->cpuData, k, bytesPerRow, reordered);
                        srcData = reordered.data();
                    }
                    for (int i = 0; i < numaConfig->numaCnt; i++) {
                        data->numasData[i] = (uint8_t*)allocate_aligned_numa(kPerNuma * bytesPerRow, i);
                        memcpy(data->numasData[i], srcData + (size_t)i * kPerNuma * bytesPerRow, kPerNuma * bytesPerRow);
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
                    if (isCrossSwiglu) {
                        std::vector<uint8_t> reordered;
                        CrossSwigluReorder(fp8Packed.data(), k, bytesPerRow, reordered);
                        fp8Packed.swap(reordered);
                    }
                    for (int i = 0; i < numaConfig->numaCnt; i++) {
                        data->numasData[i] = (uint8_t*)allocate_aligned_numa(kPerNuma * bytesPerRow, i);
                        memcpy(data->numasData[i], fp8Packed.data() + (size_t)i * kPerNuma * bytesPerRow, kPerNuma * bytesPerRow);
                    }
                } else if (data->dataType == DataType::INT8) {
                    std::vector <uint8_t> int8Packed;
                    Int8ToFastllmInt8PerchannelPacked(1, k, m, (uint8_t*)data->cpuData, data->zeros.data(), data->scales.data(), int8Packed);
                    data->dataType = DataType::INT8_PERCHANNEL;

                    size_t bytesPerRow = GetDataBytes(data->dataType, 1, m);
                    if (isCrossSwiglu) {
                        std::vector<uint8_t> reordered;
                        CrossSwigluReorder(int8Packed.data(), k, bytesPerRow, reordered);
                        int8Packed.swap(reordered);
                    }
                    for (int i = 0; i < numaConfig->numaCnt; i++) {
                        data->numasData[i] = (uint8_t*)allocate_aligned_numa(kPerNuma * bytesPerRow, i);
                        memcpy(data->numasData[i], int8Packed.data() + (size_t)i * kPerNuma * bytesPerRow, kPerNuma * bytesPerRow);
                    }
                } else if (data->dataType == DataType::INT4_NOZERO) {
                    std::vector <uint8_t> int4Packed;
                    Int4ToFastllmInt4PerchannelPacked(1, k, m, (uint8_t*)data->cpuData, data->mins.data(), data->scales.data(), int4Packed);
                    data->dataType = DataType::INT4_PERCHANNEL;

                    size_t bytesPerRow = GetDataBytes(data->dataType, 1, m);
                    if (isCrossSwiglu) {
                        std::vector<uint8_t> reordered;
                        CrossSwigluReorder(int4Packed.data(), k, bytesPerRow, reordered);
                        int4Packed.swap(reordered);
                    }
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
                    if (isCrossSwiglu) {
                        std::vector<uint8_t> reordered;
                        CrossSwigluReorder(int4Packed.data(), k, bytesPerRow, reordered);
                        int4Packed.swap(reordered);
                    }
                    for (int i = 0; i < numaConfig->numaCnt; i++) {
                        data->numasData[i] = (uint8_t*)allocate_aligned_numa(kPerNuma * bytesPerRow, i);
                        memcpy(data->numasData[i], int4Packed.data() + (size_t)i * kPerNuma * bytesPerRow, kPerNuma * bytesPerRow);
                    }
                } else {
                    ErrorInFastLLM("RegisterNumas can't support data type " + GetDataTypeName(data->dataType));
                }
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

    extern void DoCudaMergeMOEFromCPU (Data &input, Data &output, Data &index, Data &score, Data &w1, Data &w2, Data &w3, 
        Data **weights, Data **biass, float sharedScale, bool setZero, int expertLimit, bool isCrossSwiglu);
    extern void ReduceSumFromCPU(Data &output);

    void NumasMergeMOE::Run(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        fastllm::BaseOperator *op = (fastllm::BaseOperator*)(new CpuLinearOp());
 // auto ttt = std::chrono::system_clock::now();
 // std::vector <std::pair <std::string, float> > record;
 // auto st = std::chrono::system_clock::now();
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &index = *(datas.find("index")->second);
        Data &score = *(datas.find("score")->second);
        Data &w1 = *(datas.find("w1")->second);
        Data &w2 = *(datas.find("w2")->second);
        Data &w3 = *(datas.find("w3")->second);
        Data **weights = (Data**)(datas.find("weights")->second);
        Data **biass = (Data**)(datas.find("biass")->second);
        float sharedScale = floatParams.find("sharedScale") != floatParams.end() ? floatParams.find("sharedScale")->second : 1.0f;
        
        // index: [n, topk], score: [n, topk]
        int n = index.dims[0];
        int topk = index.dims[1];
        int weightsBatch = intParams.find("weights___batch") != intParams.end() ? intParams.find("weights___batch")->second : (topk + 1) * 2;
        int layer = intParams.find("layer") != intParams.end() ? intParams.find("layer")->second : 0;
        FastllmMoeDataManagerNumas &fastllmMoeDataManagerNumas = fastllmMoeDataManagerNumasPerLayer[layer % 2];
        output.Allocate();
// printf("allocate spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
        int32_t *indexData = (int32_t*)index.cpuData;
        float *scoreData = (float*)score.cpuData;

        if (input.dims[0] < 32) {
// auto st = std::chrono::system_clock::now();
            int32_t *indexData = (int32_t*)index.cpuData;
            float *scoreData = (float*)score.cpuData;

            {
                auto *pool = GetAlivePool();

                int bs = input.dims[0];
                int inputDim = input.dims[1];
                int interDim = weights[2]->dims[0] / 2;
                int outputDim = output.dims[1];

                for (int o = 0; o < bs; o++) {
                    std::vector <std::pair <int, float> > v;
                    for (int j = 0; j < topk; j++) {
                        // index 存储的是专家索引（从0开始），需要+1因为0表示shared expert
                        int expertIdx = indexData[o * topk + j];
                        float expertScore = scoreData[o * topk + j];
                        v.push_back(std::make_pair(expertIdx + 1, expertScore));
                    }
                    if (weights[0] != nullptr) {
                        v.push_back(std::make_pair(0, sharedScale));
                    }

                    DataType startDataType = weights[2]->GetLinearActDataType(1);
                    DataType downInputDataType = weights[3]->GetLinearActDataType(1);

                    // 从 fastllmMoeDataManagerNumas 获取缓存的 vector，并根据需要调整大小
                    auto& realInput = fastllmMoeDataManagerNumas.realInput;
                    auto& inputFloat32 = fastllmMoeDataManagerNumas.inputFloat32;
                    auto& gateUpOutput = fastllmMoeDataManagerNumas.gateUpOutput;
                    auto& swigluOutput = fastllmMoeDataManagerNumas.swigluOutput;
                    auto& downInput = fastllmMoeDataManagerNumas.downInput;
                    auto& downOutput = fastllmMoeDataManagerNumas.downOutput;
                    auto& reduceOutput = fastllmMoeDataManagerNumas.reduceOutput;

                    // 计算所需大小
                    size_t realInputSize = GetDataBytes(startDataType, 1, inputDim);
                    size_t inputFloat32Size = 1 * inputDim;
                    size_t gateUpOutputSize = v.size() * interDim * 2;
                    size_t swigluOutputSize = v.size() * interDim;
                    size_t downInputSize = GetDataBytes(downInputDataType, v.size(), interDim);
                    size_t downOutputSize = v.size() * outputDim;
                    size_t reduceOutputSize = 1 * outputDim;

                    // 只在当前容量不足时才进行 resize
                    if (realInput.size() < realInputSize) {
                        realInput.resize(realInputSize);
                    }
                    if (input.dataType != DataType::FLOAT32 && inputFloat32.size() < inputFloat32Size) {
                        inputFloat32.resize(inputFloat32Size);
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
                    // 0. input -> realInput（若 input 非 FLOAT32 则先转为 float32）
                    if (input.dataType == startDataType && input.dataType != DataType::FLOAT32) {
                        size_t bytes = GetDataBytes(startDataType, 1, inputDim);
                        memcpy(realInput.data(), (uint8_t*)input.cpuData + o * bytes, bytes);
                    } else {
                        const float *inputF32Ptr = nullptr;
                        if (input.dataType == DataType::FLOAT32) {
                            inputF32Ptr = (const float*)input.cpuData + o * inputDim;
                        } else {
                            if (input.dataType == DataType::FLOAT16) {
                                Float16ToFloat32((uint16_t*)input.cpuData + o * inputDim, inputFloat32.data(), inputDim);
                            } else if (input.dataType == DataType::BFLOAT16) {
                                BFloat16ToFloat32((uint16_t*)input.cpuData + o * inputDim, inputFloat32.data(), inputDim);
                            } else {
                                ErrorInFastLLM("NumasMergeMOE: unsupported input dataType.\n");
                            }
                            inputF32Ptr = inputFloat32.data();
                        }
                        RunMultiThreadConvertFromFloat32(realInput.data(), startDataType, inputF32Ptr, 1, inputDim, GetAlivePool());
                    }
// printf("RunMultiThreadConvertFromFloat32 spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));

                    // 1. gateUp + swiglu (融合算子)
                    auto *numaConfig = GetNumaConfig();

                    // 判断 downInputDataType 是否支持算子内直接转换
                    bool canFuseDstConvert = (downInputDataType == DataType::FLOAT32 ||
                                              downInputDataType == DataType::FLOAT16 ||
                                              downInputDataType == DataType::BFLOAT16);

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
                                
                                // 添加 GateUp GEMM + CrossSwiglu 融合操作
                                // 仅当 downInputDataType 支持直接转换时，才传 dstOutputData
                                uint8_t *dstPtr = canFuseDstConvert ?
                                    (uint8_t*)downInput.data() + expertIdx * GetDataBytes(downInputDataType, 1, interDim) : nullptr;
                                ((MultiThreadMultiOps*)ops[numaConfig->numaToCpuDict[nid][tid].first])->ops.push_back(
                                    new MultiThreadGemmAndCrossSwigluOp(
                                        (uint8_t*)realInput.data(), startDataType,
                                        weights[e * 2]->numasData[nid], weights[e * 2]->GetDataType(),
                                        (uint8_t*)gateUpOutput.data() + outputOffset, DataType::FLOAT32,
                                        swigluOutput.data() + expertIdx * interDim,
                                        1, inputDim, k, expertStartRow, expertEndRow, base,
                                        dstPtr, downInputDataType
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
// printf("gateup+swiglu prepare spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                    for (int i = 0; i < ops.size(); i++) {
                        pool->PushOp(i, ops[i]);
                    }

                    for (int i = 0; i < ops.size(); i++) {
                        pool->Wait(i);
                        delete ops[i];
                    }

// printf("gateup+swiglu spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));

                    // 4. swigluOutput -> downInput
                    //    如果 downInputDataType 支持算子内直接转换，则已在融合算子中完成；
                    //    否则需要在这里将 swigluOutput 转换为 downInput
                    if (!canFuseDstConvert) {
                        for (int expertIdx = 0; expertIdx < totalExperts; expertIdx++) {
                            RunMultiThreadConvertFromFloat32(
                                (uint8_t*)downInput.data() + expertIdx * GetDataBytes(downInputDataType, 1, interDim),
                                downInputDataType,
                                swigluOutput.data() + expertIdx * interDim,
                                1, interDim, GetAlivePool());
                        }
                    }

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
                            Float32ToFloat16(reduceOutput.data(), ((uint16_t*)output.cpuData) + o * outputDim, outputDim);
                        } else if (output.dataType == DataType::BFLOAT16) {
                            Float32ToBFloat16(reduceOutput.data(), (uint16_t*)output.cpuData + o * outputDim, outputDim);
                        }
                    }
// printf("last spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                }
            }
        } else {
            Data gate, attenPart, moePart;
            int bs = input.dims[0];
            int m = weightsBatch / 2 - 1; // num experts
            int expertLimit = 256;
            const char *expertLimitEnv = std::getenv("EXPERTLIMIT");
            if (expertLimitEnv != nullptr) {
                expertLimit = std::atoi(expertLimitEnv);
            }

            if (expertLimit == 0) {
                DoCudaMergeMOEFromCPU (
                    input, output, index, score, w1, w2, w3, weights, biass, sharedScale, true, 0, true
                );
                return;
            }

            std::thread gpuThread([&]() {
                DoCudaMergeMOEFromCPU (
                    input, output, index, score, w1, w2, w3, weights, biass, sharedScale, true, expertLimit, true
                );
            });
// printf("gpuThread create spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));

            {
                auto *pool = GetAlivePool();

                int dim = output.dims[1];
                int inputDim = input.dims[1];
                int interDim = weights[2]->dims[0] / 2;
                int outputDim = output.dims[1];

                std::vector <std::vector <std::pair <int, float> > > expertTasks; // expertTasks[i]代表专家i的task, expertTasks[i][j] = (第j个任务对应的行数， 权重)
                expertTasks.resize(m + 1);
                for (int b = 0; b < bs; b++) {
                    expertTasks[0].push_back(std::make_pair(b, sharedScale));
                    for (int j = 0; j < topk; j++) {
                        int expertIdx = indexData[b * topk + j];
                        float value = scoreData[b * topk + j];
                        expertTasks[expertIdx + 1].push_back(std::make_pair(b, value));
                    }
                }

                int totalLines = 0;
                for (int e = 0; e < expertTasks.size(); e++) {
                    if (weights[e * 2] != nullptr && (int)expertTasks[e].size() < expertLimit) {
                        totalLines += expertTasks[e].size();
                    }
                }
// printf("prepare spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                DataType startDataType = weights[2]->GetLinearActDataType(bs);
                DataType downInputDataType = weights[3]->GetLinearActDataType(bs);

                // 从 fastllmMoeDataManagerNumas 获取缓存的 vector，并根据需要调整大小
                auto& realInput = fastllmMoeDataManagerNumas.realInput;
                auto& inputFloat32 = fastllmMoeDataManagerNumas.inputFloat32;
                auto& expandInput = fastllmMoeDataManagerNumas.expandInput;
                auto& gateUpOutput = fastllmMoeDataManagerNumas.gateUpOutput;
                auto& swigluOutput = fastllmMoeDataManagerNumas.swigluOutput;
                auto& downInput = fastllmMoeDataManagerNumas.downInput;
                auto& downOutput = fastllmMoeDataManagerNumas.downOutput;
                auto& reduceOutput = fastllmMoeDataManagerNumas.reduceOutput;

                int alignTotalLines = ((totalLines - 1) / 64 + 1) * 64;
                // 计算所需大小
                size_t realInputSize = GetDataBytes(startDataType, bs, inputDim);
                size_t inputFloat32Size = (size_t)bs * inputDim;
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
                if (input.dataType != DataType::FLOAT32 && inputFloat32.size() < inputFloat32Size) {
                    inputFloat32.resize(inputFloat32Size);
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
                // downOutput 不需要 fill 0：所有有效行都会被 down 阶段完整写入；
                if (reduceOutput.size() < reduceOutputSize) {
                    reduceOutput.resize(reduceOutputSize);
                }

// printf("malloc spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                // 0. input -> realInput（若 input 非 FLOAT32 则先转为 float32）
                if (input.dataType == startDataType && input.dataType != DataType::FLOAT32) {
                    size_t bytes = GetDataBytes(startDataType, bs, inputDim);
                    RunMultiThreadMemcpy(realInput.data(), (uint8_t*)input.cpuData, bytes, GetAlivePool());
                } else {
                    const float *inputF32Ptr = nullptr;
                    if (input.dataType == DataType::FLOAT32) {
                        inputF32Ptr = (const float*)input.cpuData;
                    } else {
                        int inputCount = bs * inputDim;
                        if (input.dataType == DataType::FLOAT16) {
                            Float16ToFloat32((uint16_t*)input.cpuData, inputFloat32.data(), inputCount);
                        } else if (input.dataType == DataType::BFLOAT16) {
                            BFloat16ToFloat32((uint16_t*)input.cpuData, inputFloat32.data(), inputCount);
                        } else {
                            ErrorInFastLLM("NumasMergeMOE: unsupported input dataType.\n");
                        }
                        inputF32Ptr = inputFloat32.data();
                    }
                    RunMultiThreadConvertFromFloat32(realInput.data(), startDataType, inputF32Ptr, bs, inputDim, GetAlivePool());
                }
// printf("RunMultiThreadConvertFromFloat32 spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));

                // 1. realInput -> expandInput
                std::vector <MultiThreadMemcpyMultiLinesTask> memcpyTasks;
                memcpyTasks.resize(totalLines);
                {
                    uint8_t* realInputPtr = realInput.data();
                    uint8_t* expandInputPtr = expandInput.data();
                    int bytesPerLine = GetDataBytes(startDataType, 1, inputDim);

                    // 预计算每个 expert 在 expandInput 中的起始偏移（跳过 expertLimit 筛掉的专家）
                    std::vector<int> curPos(expertTasks.size(), -1);
                    {
                        int base = 0;
                        for (int e = 0; e < expertTasks.size(); e++) {
                            if (weights[e * 2] != nullptr && (int)expertTasks[e].size() < expertLimit) {
                                curPos[e] = base;
                                base += expertTasks[e].size();
                            }
                        }
                    }

                    // 按 token 顺序枚举：先 shared expert (expert 0)，再按 b * topk + j 顺序
                    // 跳过被 expertLimit 筛掉的专家
                    int idx = 0;
                    for (int b = 0; b < bs; b++) {
                        // shared expert (expert 0)
                        if (weights[0] != nullptr && curPos[0] >= 0) {
                            int pos = curPos[0]++;
                            memcpyTasks[idx++] = MultiThreadMemcpyMultiLinesTask(
                                expandInputPtr + pos * bytesPerLine,
                                realInputPtr + b * bytesPerLine,
                                bytesPerLine
                            );
                        }
                        // routed experts
                        for (int j = 0; j < topk; j++) {
                            int expertIdx = indexData[b * topk + j] + 1;
                            if (weights[expertIdx * 2] != nullptr && curPos[expertIdx] >= 0) {
                                int pos = curPos[expertIdx]++;
                                memcpyTasks[idx++] = MultiThreadMemcpyMultiLinesTask(
                                    expandInputPtr + pos * bytesPerLine,
                                    realInputPtr + b * bytesPerLine,
                                    bytesPerLine
                                );
                            }
                        }
                    }
                    memcpyTasks.resize(idx);
                }
// printf("prepare expand spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                RunMultiThreadMemcpyMultiLines(memcpyTasks, GetAlivePool());
// printf("expand spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                // 2. gateUp
                
                auto *numaConfig = GetNumaConfig();

                int offset = 0;
                int stride = 64;

                // 判断 downInputDataType 是否支持算子内直接转换
                bool canFuseDstConvert = (downInputDataType == DataType::FLOAT32 ||
                                          downInputDataType == DataType::FLOAT16 ||
                                          downInputDataType == DataType::BFLOAT16);

                std::vector<std::vector <fastllm::MultiThreadBaseOp*> > ops;
                ops.resize(numaConfig->numaCnt);

                for (int e = 0; e < expertTasks.size(); e++) {
                    if (weights[e * 2] != nullptr && expertTasks[e].size() > 0) {
                        int lines = expertTasks[e].size();
                        if (lines >= expertLimit) {
                            continue;
                        }
                        // Prepare input pointer for this expert's batch
                        uint16_t* expertInputPtr = (uint16_t*)(expandInput.data() + offset * GetDataBytes(startDataType, 1, inputDim));
                            
                        // Prepare output pointer for this expert's batch
                        float* expertGateUpOutputPtr = gateUpOutput.data() + offset * interDim * 2;
                        float* expertSwigluOutputPtr = swigluOutput.data() + offset * interDim;
                        uint8_t* expertDstOutputPtr = canFuseDstConvert ?
                            (uint8_t*)downInput.data() + offset * GetDataBytes(downInputDataType, 1, interDim) : nullptr;

                        int k = interDim * 2;
                        int kPer = k / numaConfig->numaCnt;
                            
                        for (int nid = 0; nid < numaConfig->numaCnt; nid++) {
                            // Get weight data (assuming weights are stored as `startDataType`)
                            int base = kPer * nid;
                            size_t outputOffset = GetDataBytes(DataType::FLOAT32, 1, base);

                            for (int st = 0; st < kPer; st += stride) {
                                int end = std::min(st + stride, kPer);
                                ops[nid].push_back(new MultiThreadGemmAndCrossSwigluOp(
                                    (uint8_t*)expertInputPtr, startDataType,
                                    weights[e * 2]->numasData[nid], weights[e * 2]->GetDataType(),
                                    (uint8_t*)expertGateUpOutputPtr + outputOffset, DataType::FLOAT32,
                                    expertSwigluOutputPtr,
                                    lines, inputDim, k, st, end, base,
                                    expertDstOutputPtr, downInputDataType
                                ));
                            }
                        }
                        offset += lines;
                    }
                }

                DynamicScheduleTasks(ops);

// printf("gateup+swiglu spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));

                // 4. swigluOutput -> downInput
                //    如果 downInputDataType 支持算子内直接转换，则已在融合算子中完成；
                //    否则需要在这里将 swigluOutput 转换为 downInput
                if (!canFuseDstConvert) {
                    offset = 0;
                    for (int e = 0; e < expertTasks.size(); e++) {
                        if (weights[e * 2] != nullptr && expertTasks[e].size() > 0) {
                            int lines = expertTasks[e].size();
                            if (lines >= expertLimit) {
                                continue;
                            }
                            float* expertSwigluOutputPtr = swigluOutput.data() + offset * interDim;
                            uint8_t* expertDstOutputPtr = (uint8_t*)downInput.data() + offset * GetDataBytes(downInputDataType, 1, interDim);
                            RunMultiThreadConvertFromFloat32(expertDstOutputPtr, downInputDataType,
                                                            expertSwigluOutputPtr, lines, interDim, GetAlivePool());
                            offset += lines;
                        }
                    }
                }

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
                        if (lines >= expertLimit) {
                            continue;
                        }
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
                    // 计算每个样本选择的专家数 k
                    int k = 0;
                    std::vector<int> samples_expert_count(bs, 0);
                    for (int e = 0; e < expertTasks.size(); e++) {
                        if (weights[e * 2] != nullptr && (int)expertTasks[e].size() < expertLimit) {
                            for (auto& task : expertTasks[e]) {
                                int rowIdx = task.first;
                                samples_expert_count[rowIdx]++;
                                k = std::max(k, samples_expert_count[rowIdx]);
                            }
                        }
                    }

                    // 分配内存: task_weights 按 totalLines 大小，以 downOutput 行号索引
                    std::vector<int> pos(bs * k, -1);
                    std::vector<float> task_weights(totalLines, 0.0f);
                    std::vector<int> sample_expert_idx(bs, 0);

                    int reduceOffset = 0;
                    for (int e = 0; e < expertTasks.size(); e++) {
                        if (weights[e * 2] != nullptr && (int)expertTasks[e].size() < expertLimit) {
                            for (auto& task : expertTasks[e]) {
                                int rowIdx = task.first;
                                float weight = task.second;
                                int idx = sample_expert_idx[rowIdx]++;
                                pos[rowIdx * k + idx] = reduceOffset;
                                task_weights[reduceOffset] = weight;
                                reduceOffset++;
                            }
                        }
                    }

                    // 有一些token不会被关联到任何专家，也需要先清零
                    float *lastOutput = output.dataType == DataType::FLOAT32 ? (float*)output.cpuData : reduceOutput.data();
                    memset(lastOutput, 0, bs * dim * sizeof(float));

                    // 调用多线程函数
                    MultiThreadReduceBatch(
                        (uint8_t*)downOutput.data(),  // downOutData
                        DataType::FLOAT32,             // downOutDataType
                        task_weights.data(),           // weights
                        lastOutput,                    // lastOutput
                        pos.data(),                    // pos
                        bs,                           // bsz
                        k,                            // k (每个样本的专家数)
                        dim                           // hidden_size
                    );
                }
 // printf("reduce spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                // 7. reduceOutput -> last Output
                if (output.dataType != DataType::FLOAT32) {
                    if (output.dataType == DataType::FLOAT16) {
                        // Float32ToFloat16(reduceOutput.data(), (uint16_t*)output.cpuData, bs * dim);
                        RunMultiThreadConvertFromFloat32((uint16_t*)output.cpuData, DataType::FLOAT16, 
                            reduceOutput.data(), bs, dim, GetAlivePool());
                    } else if (output.dataType == DataType::BFLOAT16) {
                        // Float32ToBFloat16(reduceOutput.data(), (uint16_t*)output.cpuData, bs * dim);
                        RunMultiThreadConvertFromFloat32((uint16_t*)output.cpuData, DataType::BFLOAT16, 
                            reduceOutput.data(), bs, dim, GetAlivePool());
                    }
                }
// printf("last spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
            }

            gpuThread.join();
// printf("gpuThread join spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
            ReduceSumFromCPU(output);
// printf("last sum spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
            return;
        }
    }
}