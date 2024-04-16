//
// Created by huangyuyang on 11/4/24.
//

#include <sys/mman.h>
#include <fcntl.h>

#include "devices/tfacc/tfaccdevice.h"
#include "devices/tfacc/fastllm-tfacc.h"

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
    TfaccDevice::TfaccDevice() {
        this->deviceType = "tfacc";
        this->ops["Linear"] = (BaseOperator *) (new TfaccLinearOp());
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
        Data &input = *(datas.find("input")->second);
        Data &weight = *(datas.find("weight")->second);
        return weight.dataType == DataType::INT4_NOZERO ||
                weight.dataType == DataType::INT8;
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

        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void TfaccLinearOp::Run(const std::string &opType, const DataDict &datas, const FloatDict &floatParams, const IntDict &intParams) {
        //auto st = std::chrono::system_clock::now();
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);

        output.Allocate(0.0f);
        int n = input.Count(0) / input.dims.back();
        int m = input.dims.back();
        int k = output.dims.back();

        if (input.dataType == DataType::FLOAT32 && output.dataType == DataType::FLOAT32) {
            if (weight.dataType == DataType::FLOAT32) {
                ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
            } else if (weight.dataType == DataType::INT4 || 
                    weight.dataType == DataType::INT4_NOZERO ||
                    weight.dataType == DataType::INT8) {
                float *inputData = (float *) input.cpuData;
                uint8_t *weightData = (uint8_t *) weight.cpuData;
                float *outputData = (float *) output.cpuData;
                float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
                weight.CalcWeightSum();

                std::vector<LowBitConfig> inputConfigs;
                for (int i = 0; i < n; i++) {
                    float minValue = 1e9, maxValue = -1e9;
                    int j = 0;
#ifdef __aarch64__
                    float32x4_t mins = vdupq_n_f32(1e100);
                    float32x4_t maxs = vdupq_n_f32(-1e100);
                    for (; j + 3 < m; j += 4) {
                        float32x4_t v = vld1q_f32(inputData + i * m + j);
                        mins = vminq_f32(mins, v);
                        maxs = vmaxq_f32(maxs, v);
                    }
                    for (int l = 0; l < 4; l++) {
                        minValue = std::min(minValue, mins[l]);
                        maxValue = std::max(maxValue, maxs[l]);
                    }
#endif
                    for (; j < m; j++) {
                        minValue = std::min(minValue, inputData[i * m + j]);
                        maxValue = std::max(maxValue, inputData[i * m + j]);
                    }
                    inputConfigs.push_back(LowBitConfig(minValue, maxValue, 8, 0));
                }
                std::vector<uint8_t> uinput;
                uinput.resize(n * m);

                for (int i = 0; i < n; i++) {
                    float scale = inputConfigs[i].scale;
                    float zeroPoint = inputConfigs[i].zeroPoint;
                    float *cur = inputData + i * m;
                    uint8_t *u = uinput.data() + i * m;
                    int j = 0;
#ifdef __aarch64__
                    float32x4_t scales = vdupq_n_f32(scale);
                    float32x4_t zeros = vdupq_n_f32(zeroPoint + 0.5);
                    int32x4_t maxds = vcombine_s32(vcreate_s32(0x000000ff000000ff), vcreate_s32(0x000000ff000000ff));
                    int32x4_t minds = vcombine_s32(vcreate_s32(0x0000000000000000), vcreate_s32(0x0000000000000000));
                    for (; j + 7 < m; j += 8) {
                        float32x4_t fin1 = vld1q_f32(cur + j);
                        float32x4_t fin2 = vld1q_f32(cur + j + 4);
                        fin1 = vaddq_f32(vdivq_f32(fin1, scales), zeros);
                        fin2 = vaddq_f32(vdivq_f32(fin2, scales), zeros);
                        int32x4_t out1 = vcvtq_s32_f32(fin1);
                        int32x4_t out2 = vcvtq_s32_f32(fin2);
                        out1 = vmaxq_s32(out1, minds);
                        out1 = vminq_s32(out1, maxds);
                        out2 = vmaxq_s32(out2, minds);
                        out2 = vminq_s32(out2, maxds);
                        uint16x8_t out3 = vpaddq_u16(vreinterpretq_u16_s32(out1), vreinterpretq_u16_s32(out2));
                        uint8x8_t out = vmovn_u16(out3);
                        vst1_u8(u + j, out);
                    }
#endif
                    for (; j < m; j++) {
                        u[j] = (uint8_t) (std::min(255., (double) std::max(cur[j] / scale + zeroPoint + 0.5, 0.0)));
                    }
                }
#ifdef __AVX__
                uint8_t *temp = new uint8_t[32];
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j + 31 < m; j += 32) {
                        memcpy(temp, uinput.data() + i * m + j, 32);
                        for (int k = 0; k < 16; k++) {
                            uinput[i * m + j + k] = temp[k * 2 + 1];
                            uinput[i * m + j + k + 16] = temp[k * 2];
                        }
                    }
                }
                delete[] temp;
#endif
                if (weight.dataType == DataType::INT4) {
                    ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
                } else if (weight.dataType == DataType::INT8 || weight.dataType == DataType::INT4_NOZERO) {
                    int opType = 1;
                    if (weight.dataType == DataType::INT8) {
                        opType = 2;
                    }

                    static int fd = open("/dev/thinkforce0", O_RDWR);
                    static volatile uint8_t *buf = (volatile uint8_t *)mmap(NULL, 64 * 1024 * 1024, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 9997 * 0x1000);
                    static volatile uint8_t *result = buf + 32 * 1024 * 1024;
                    static volatile int32_t *flag = (volatile int32_t*)(buf + 63 * 1024 * 1024);
                    static int PAGE = 16 * 1024;
                    static int maxPartCnt = 4;
                    static int transLimit = 28 * 1024 * 1024;

                    std::string biasName = biasData == nullptr ? "" : bias.name;
                    int maxN = n;
                    maxN = std::min(maxN, transLimit / m);
                    maxN = std::min(maxN, (int)(transLimit / (k * sizeof(float))));
                    
                    // printf("maxN = %d\n", maxN);
                    for (int baseN = 0; baseN < n; baseN += maxN) {
                        int curN = std::min(maxN, n - baseN);
                        ((int32_t*)buf)[0] = curN;
                        ((int32_t*)buf)[1] = m;
                        ((int32_t*)buf)[2] = k;
                        ((int32_t*)buf)[3] = 1; // group
                        ((int32_t*)buf)[4] = weight.name.size();
                        ((int32_t*)buf)[5] = biasName.size();

                        volatile uint8_t *cur = (uint8_t*)buf + 10 * sizeof(int32_t);
                        for (int i = 0; i < curN; i++) {
                            ((float*)cur)[0] = inputConfigs[baseN + i].min;
                            ((float*)cur)[1] = inputConfigs[baseN + i].max;
                            cur += 2 * sizeof(float);
                        }
                        memcpy((uint8_t*)cur, weight.name.c_str(), weight.name.size());
                        cur += weight.name.size();
                        memcpy((uint8_t*)cur, biasName.c_str(), biasName.size());
                        cur += biasName.size();
                        memcpy((uint8_t*)cur, uinput.data() + baseN * m, curN * m);
                        asm volatile("dmb ish");

                        volatile int *curFlag = flag;
                        for (int i = 0; i < maxPartCnt; i++) {
                            *(curFlag) = opType;
                            curFlag += PAGE;
                            asm volatile("dmb ish");
                        }

                        while (true) {
                            volatile int *curFlag = flag;
                            int notFinish = 0;
                            for (int i = 0; i < maxPartCnt; i++) {
                                notFinish |= (*curFlag);
                                curFlag += PAGE;
                            }

                            if (!notFinish) {
                                memcpy(((uint8_t*) outputData) + baseN * k * sizeof(int32_t), 
                                        (uint8_t*) result, 
                                        curN * k * sizeof(int32_t));
                                break;
                            }
                            //asm volatile("dmb ish");
                        }
                    }
                }
            } else if (weight.dataType == DataType::INT4_GROUP) {
                ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
            }
        } else if (input.dataType == DataType::FLOAT16 && output.dataType == DataType::FLOAT16) {
            ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
        } else {
            ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
        }
//float spend = GetSpan(st, std::chrono::system_clock::now());
//float gops = (float)n * m * k / spend / 1e9;
// printf("n = %d, m = %d, k = %d, spend %f s, gops = %f\n", n, m, k, spend, gops);
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
}