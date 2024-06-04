//
// Created by huangyuyang on 11/4/24.
//

#include <sys/mman.h>
#include <fcntl.h>

#include "devices/tfacc/tfaccdevice.h"
#include "devices/cpu/cpudevice.h"
#include "devices/tfacc/fastllm-tfacc.h"
#include "devices/cpu/alivethreadpool.h"

#include <cstring>
#include <thread>

#include <cfloat>
#include <cmath>

#ifdef __aarch64__
#include <arm_neon.h>
#include "armMath.h"
#endif

#include "utils.h"

namespace fastllm {
    static TfaccClient tfaccClient;

    TfaccDevice::TfaccDevice() {
        this->deviceType = "tfacc";
        this->ops["Linear"] = (BaseOperator *) (new TfaccLinearOp());
        this->ops["CatDirect"] = (BaseOperator *) (new TfaccCatDirectOp());
        this->ops["Attention"] = (BaseOperator *) (new TfaccAttention());

        this->ops["AttentionBatch"] = (BaseOperator *) (new TfaccAttentionBatchOp());
        this->ops["CatDirectBatch"] = (BaseOperator *) (new TfaccCatDirectBatchOp());
    }

    bool TfaccDevice::Malloc(void **ret, size_t size) {
        *ret = (void*)new uint8_t [size];
        return true;
    }

    bool TfaccDevice::Free(void *ret) {
        delete[] (uint8_t *)ret;
        return true;
    }

    bool TfaccDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
        return true;
    }
    
    bool TfaccDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
        return true;
    }

    bool TfaccLinearOp::CanRun(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        if (datas.find("weight") == datas.end()) {
            return true;
        }
        Data *input = (datas.find("input")->second);
        Data *weight = (datas.find("weight")->second);
        return weight == nullptr ||
                weight->dataType == DataType::INT4_NOZERO ||
                weight->dataType == DataType::INT8 ||
                weight->dataType == DataType::INT4_GROUP ||
                weight->dataType == DataType::FLOAT32 ||
                weight->dataType == DataType::FLOAT16;
    }

    void TfaccLinearOp::Reshape(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);

        AssertInFastLLM(weight.dims.size() == 2, "Linear's weight's shape's size should be 2.\n");
        AssertInFastLLM(input.dims.back() == weight.dims[1], "Linear's weight's shape error.\n");

        weight.weightType = WeightType::LINEAR;
        std::vector <int> dims = input.dims;
        dims.back() = weight.dims[0];

        if (intParams.find("exType") != intParams.end()) {
            LinearExType type = (LinearExType)intParams.find("exType")->second;
            if (type == LinearExType::ExSwiglu) {
                dims.back() /= 2;
            }
        }

        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void TfaccLinearOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
// float inner = 0.0;        
// auto st = std::chrono::system_clock::now();
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);

        output.Allocate();
        int n = input.Count(0) / input.dims.back();
        int m = input.dims.back();
        int k = output.dims.back();
        LinearExType exType = LinearExType::ExTypeNone;
        if (intParams.find("exType") != intParams.end()) {
            exType = (LinearExType)intParams.find("exType")->second;
            if (exType == LinearExType::ExSwiglu) {
                k *= 2;
            }
        }

        if (input.dataType == DataType::FLOAT32 && output.dataType == DataType::FLOAT32) {
            if (weight.dataType == DataType::FLOAT32 || weight.dataType == DataType::FLOAT16) {
                tfaccClient.RunTfaccLinearF(n, m, k, &weight, &bias, (float*)input.cpuData, (float*)output.cpuData, exType, input.dataType);
/*
                std::vector <float> tempOutputs;
                for (int i = 0; i < output.Count(0); i++) {
                    tempOutputs.push_back(((float*)output.cpuData)[i]);
                }
                CpuLinearOp::Run(opType, datas, floatParams, intParams);
                for (int i = 0; i < output.Count(0); i++) {
                    if (fabs(((float*)output.cpuData)[i] - tempOutputs[i]) > 1e-5) {
                        printf("wrong %d %f %f.\n", i, ((float*)output.cpuData)[i], tempOutputs[i]);
                        exit(0);
                    }
                }
*/
            } else if (weight.dataType == DataType::INT4 || 
                    weight.dataType == DataType::INT4_NOZERO ||
                    weight.dataType == DataType::INT4_GROUP ||
                    weight.dataType == DataType::INT8) {                
                float *inputData = (float *) input.cpuData;
                uint8_t *weightData = (uint8_t *) weight.cpuData;
                float *outputData = (float *) output.cpuData;
                float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
                weight.CalcWeightSum();

                int group = weight.group, groupCnt = weight.groupCnt;
                if (weight.dataType != DataType::INT4_GROUP) {
                    group = 1;
                    groupCnt = m;
                }

                std::vector<LowBitConfig> inputConfigs;
                inputConfigs.resize(n * group);
                std::vector<uint8_t> uinput;
                uinput.resize(n * m);

                if (n > 1) {
                    auto pool = GetAlivePool();
                    int threadNum = pool->threads.size();
                    int per = n / pool->threads.size();
                    int cur = 0;
                    std::vector<fastllm::MultiThreadOnlineQuantizationOp*> ops;
                    for (int i = 0; i < threadNum; i++) {
                        int end = (i == threadNum - 1 ? n : cur + per + (cur + per * (threadNum - i) < n));
                        ops.push_back(new MultiThreadOnlineQuantizationOp(
                                        inputData + cur * m, uinput.data() + cur * m, inputConfigs.data() + cur * group,
                                        end - cur, m, group, groupCnt, nullptr, nullptr, nullptr));
                        cur = end;
                    }
                    for (int i = 0; i < threadNum; i++) {
                        pool->PushOp(i, ops[i]);
                    }
                    for (int i = 0; i < threadNum; i++) {
                        pool->Wait(i);
                        delete ops[i];
                    }
                } else {
                    MultiThreadOnlineQuantizationOp(inputData, uinput.data(), inputConfigs.data(), n, m, group, groupCnt, nullptr, nullptr, nullptr).Run();
                }

                if (weight.dataType == DataType::INT4) {
                    ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
                } else if (weight.dataType == DataType::INT8 || weight.dataType == DataType::INT4_NOZERO || weight.dataType == DataType::INT4_GROUP) {
// auto st = std::chrono::system_clock::now();
                    tfaccClient.RunTfaccLinearU(n, m, k, group, groupCnt, &weight, &bias, &inputConfigs, uinput.data(), outputData, exType);
// float spend = GetSpan(st, std::chrono::system_clock::now());
// float gops = (float)n * m * k / spend / 1e9;
// inner = spend;
// if (n > 0) printf("n = %d, m = %d, k = %d, spend %f s, gops = %f (inner)\n", n, m, k, spend, gops);
                }
            }
        } else if (input.dataType == DataType::FLOAT16 && output.dataType == DataType::FLOAT16) {
            ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
        } else {
            ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
        }
// float spend = GetSpan(st, std::chrono::system_clock::now());
// float gops = (float)n * m * k / spend / 1e9;
// if (n > 0) printf("n = %d, m = %d, k = %d, spend %f s, gops = %f, outer = %f\n", n, m, k, spend, gops, spend - inner);
    }

    long long int TfaccLinearOp::Ops(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);

        int n = input.Count(0) / input.dims.back();
        int m = input.dims.back();
        int k = output.dims.back();

        return (long long int) n * m * k;
    }

    void TfaccCatDirectOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data *input0 = (datas.find("input0")->second);
        Data *input1 = (datas.find("input1")->second);

        if (input0->isKVCache) {
            // 如果是kvCache，那么要同步到server上
            tfaccClient.AppendKVCache(input0->cacheUid, input1);

            int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
            if (input0->dims.size() == 0) {
                input0->Resize(input1->dims);
                return;
            }

            int dimsLen = input0->dims.size();
            axis = (axis % dimsLen + dimsLen) % dimsLen;

            std::vector <int> dims = input0->dims;
            std::vector <int> oldDims = dims;
            dims[axis] += input1->dims[axis];
            input0->Resize(dims);
        } else {
            CpuCatDirectOp::Run(opType, datas, floatParams, intParams);
        }
    }

    void TfaccAttention::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &k = *(datas.find("k")->second);
        Data &v = *(datas.find("v")->second);
        int maskType = intParams.find("maskType") != intParams.end() ? intParams.find("maskType")->second : 0;

        if (!k.isKVCache || !v.isKVCache || maskType != 0) {
            CpuAttention::Run(opType, datas, floatParams, intParams);
            return;
        }

        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : 1;
        float scale = floatParams.find("scale") != floatParams.end() ? floatParams.find("scale")->second : 1.0;

        Data &output = *(datas.find("output")->second);
        output.Allocate();

        tfaccClient.Attention(&q, &k, &v, group, scale, maskType, &output);
/*
        std::vector <float> tempOutput;
        for (int i = 0; i < output.Count(0); i++) {
            tempOutput.push_back(((float*)output.cpuData)[i]);
        }

        CpuAttention::Run(opType, datas, floatParams, intParams);
        for (int i = 0; i < output.Count(0); i++) {
            float x = ((float*)output.cpuData)[i];
            if (fabs(x - tempOutput[i]) > 1e-5) {
                printf("wrong %d %f %f\n", i, x, tempOutput[i]);
                exit(0);
            }
        }
*/
    }

    void TfaccAttentionBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        fastllm::BaseOperator *op = (fastllm::BaseOperator*)(new TfaccAttention());
        int batch = intParams.find("q___batch")->second;
        DataDict tempDatas = datas;
        for (int i = 0; i < batch; i++) {
            tempDatas["q"] = ((Data**)datas.find("q")->second)[i];
            tempDatas["k"] = ((Data**)datas.find("k")->second)[i];
            tempDatas["v"] = ((Data**)datas.find("v")->second)[i];
            tempDatas["mask"] = ((Data**)datas.find("mask")->second)[i];
            tempDatas["output"] = ((Data**)datas.find("output")->second)[i];
            op->Run("Attention", tempDatas, floatParams, intParams);
        }
        delete op;
    }

    void TfaccCatDirectBatchOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                  const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        fastllm::BaseOperator *op = (fastllm::BaseOperator*)(new TfaccCatDirectOp());
        int batch = intParams.find("input0___batch")->second;
        DataDict tempDatas = datas;
        for (int i = 0; i < batch; i++) {
            tempDatas["input0"] = ((Data**)datas.find("input0")->second)[i];
            tempDatas["input1"] = ((Data**)datas.find("input1")->second)[i];
            op->Run("CatDirect", tempDatas, floatParams, intParams);
        }
        delete op;
    }
}