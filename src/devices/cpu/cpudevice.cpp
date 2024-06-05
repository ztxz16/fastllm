//
// Created by huangyuyang on 6/13/23.
//

#include "devices/cpu/cpudevice.h"

#include <cstring>
#include <thread>

#include <cfloat>
#include <cmath>

#ifdef __aarch64__
#include <arm_neon.h>
#include "armMath.h"
#endif

#include "utils.h"
#define M_PI       3.14159265358979323846   // pi

namespace fastllm {
    CpuDevice::CpuDevice() {
        this->deviceType = "cpu";
        this->ops["ToFloat16"] = (BaseOperator*)(new CpuToFloat16());
        this->ops["ToFloat32"] = (BaseOperator*)(new CpuToFloat32());
        this->ops["Attention"] = (BaseOperator*)(new CpuAttention());
        this->ops["MergeMOE"] = (BaseOperator*)(new CpuMergeMOE());
        this->ops["CopyKVCache"] = (BaseOperator*)(new CpuCopyKVCacheOp());
        this->ops["Embedding"] = (BaseOperator*)(new CpuEmbedding());
        this->ops["LayerNorm"] = (BaseOperator*)(new CpuLayerNormOp());
        this->ops["RMSNorm"] = (BaseOperator*)(new CpuRMSNormOp());
        this->ops["Linear"] = (BaseOperator*)(new CpuLinearOp());
        this->ops["Split"] = (BaseOperator*)(new CpuSplitOp());
        this->ops["Repeat"] = (BaseOperator*)(new CpuRepeatOp());
        this->ops["Cat"] = (BaseOperator*)(new CpuCatOp());
        this->ops["CatDirect"] = (BaseOperator*)(new CpuCatDirectOp());
        this->ops["MatMul"] = (BaseOperator*)(new CpuMatMulOp());
        this->ops["MatMulTransB"] = (BaseOperator*)(new CpuMatMulTransBOp());
        this->ops["SoftMax"] = (BaseOperator*)(new CpuSoftMaxOp());
        this->ops["Silu"] = (BaseOperator*)(new CpuSiluOp());
        this->ops["TanH"] = (BaseOperator*)(new CpuTanHOp());
        this->ops["Gelu"] = (BaseOperator*)(new CpuGeluOp());
        this->ops["GeluNew"] = (BaseOperator*)(new CpuGeluNewOp());
        this->ops["Swiglu"] = (BaseOperator*)(new CpuSwigluOp());
        this->ops["Mul"] = (BaseOperator*)(new CpuMulOp());
        this->ops["MulTo"] = (BaseOperator*)(new CpuMulToOp());
        this->ops["AddTo"] = (BaseOperator*)(new CpuAddToOp());
        this->ops["AttentionMask"] = (BaseOperator*)(new CpuAttentionMaskOp());
        this->ops["AttentionExtendedMask"] = (BaseOperator*)(new CpuAttentionExtendedMaskOp());
        this->ops["AlibiMask"] = (BaseOperator*)(new CpuAlibiMaskOp());
        this->ops["TopK"] = (BaseOperator*)(new CpuTopKOp());
        this->ops["Permute"] = (BaseOperator*)(new CpuPermuteOp());
        this->ops["PermuteSelf"] = (BaseOperator*)(new CpuPermuteSelfOp());
        this->ops["RotatePosition2D"] = (BaseOperator*)(new CpuRotatePosition2DOp());
        this->ops["NearlyRotatePosition2D"] = (BaseOperator*)(new CpuNearlyRotatePosition2DOp());
        this->ops["LlamaRotatePosition2D"] = (BaseOperator*)(new CpuLlamaRotatePosition2DOp());
        this->ops["RepeatPenalty"] = (BaseOperator*)(new CpuRepeatPenaltyOp());
        this->ops["ApplyLognAttn"] = (BaseOperator*)(new CpuApplyLognAttnOp());

        this->ops["SplitBatch"] = (BaseOperator*)(new CpuSplitBatchOp());
        this->ops["CatBatch"] = (BaseOperator*)(new CpuCatBatchOp());
        this->ops["MulBatch"] = (BaseOperator*)(new CpuMulBatchOp());
        this->ops["MatMulBatch"] = (BaseOperator*)(new CpuMatMulBatchOp());
        this->ops["MatMulTransBBatch"] = (BaseOperator*)(new CpuMatMulTransBBatchOp());
        this->ops["SoftMaxBatch"] = (BaseOperator*)(new CpuSoftmaxBatchOp());
        this->ops["CatDirectBatch"] = (BaseOperator*)(new CpuCatDirectBatchOp());
        this->ops["AppendKVCachebatch"] = (BaseOperator*)(new CpuAppendKVCacheBatchOp());
        this->ops["AttentionBatch"] = (BaseOperator*)(new CpuAttentionBatchOp());
    }

    bool CpuDevice::Malloc(void **ret, size_t size) {
        *ret = (void*)new uint8_t [size];
        return true;
    }

    bool CpuDevice::Free(void *ret) {
        delete[] (uint8_t*)ret;
        return true;
    }

    bool CpuDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
        return true;
    }

    bool CpuDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
        return true;
    }


#ifdef __AVX2__
    int DotU8U8(uint8_t *a, uint8_t *b, int n) {
        __m256i acc = _mm256_setzero_si256();
        int i = 0;
        int ans = 0;
        const __m256i lowMask = _mm256_set1_epi8(0xf);
        const __m256i ones = _mm256_set1_epi16(1);
        const __m256i ones8 = _mm256_set1_epi8(1);
        const __m256i xors = _mm256_set1_epi8(-128);
        for (; i + 31 < n; i += 32) {
            __m256i bx = _mm256_loadu_si256((const __m256i *) (a + i));
            __m256i by = _mm256_loadu_si256((const __m256i *) (b + i));

            by = _mm256_xor_si256(by, xors);
            by = _mm256_add_epi8(by, _mm256_and_si256(_mm256_cmpeq_epi8(by, xors), ones8));

            by = _mm256_sign_epi8(by, bx);
            bx = _mm256_sign_epi8(bx, bx);

            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(bx, by), ones));
        }
        for (; i < n; i++) {
            ans += ((int8_t*)a)[i] * ((int)b[i] - 128);
        }

        return ans + I32sum(acc);
    };
//#else
//    int DotU8U8(uint8_t *a, uint8_t *b, int n) {
//        __m256i acc = _mm256_setzero_si256();

//        int i = 0;
//        int ans = 0;
//        for (; i + 31 < n; i += 32) {
//            __m256i bx = _mm256_loadu_si256((const __m256i *) (a + i));
//            __m256i by = _mm256_loadu_si256((const __m256i *) (b + i));

//            __m256i mx0 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(bx, 0));
//            __m256i mx1 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(bx, 1));

//            __m256i my0 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(by, 0));
//            __m256i my1 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(by, 1));

//            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx0, my0));
//            //acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx1, my1));
//        }
//        for (; i < n; i++) {
//            ans += a[i] * b[i];
//        }

//        return ans + I32sum(acc);
//    };
    int DotU4U8(uint8_t *a, uint8_t *b, int n) {
        __m256i acc = _mm256_setzero_si256();

        int i = 0;
        int ans = 0;
        const __m256i lowMask = _mm256_set1_epi8(0xf);
        const __m256i ones = _mm256_set1_epi16(1);
        for (; i + 31 < n; i += 32) {
            __m128i orix = _mm_loadu_si128((const __m128i *) (a + i / 2));
            __m256i bytex = _mm256_set_m128i(_mm_srli_epi16(orix, 4), orix);
            __m256i bx = _mm256_and_si256(lowMask, bytex);
            __m256i by = _mm256_loadu_si256((const __m256i *) (b + i));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(by, bx), ones));
        }
        for (; i < n; i++) {
            ans += a[i] * b[i];
        }

        return ans + I32sum(acc);
    };
#endif

    struct FP16ToFP32Manager {
        float dict[65536];

        FP16ToFP32Manager() {
            for (uint16_t i = 0; i < 65535; i++) {
                dict[i] = half_to_float(i);
            }
        }
    } fp16tofp32;

    void Float16ToFloat32(uint16_t *float16, float *float32, int len) {
        for (int i = 0; i < len; i++) {
            float32[i] = fp16tofp32.dict[float16[i]];
        }
    }

    void Float32ToFloat16(float *float32, uint16_t *float16, int len) {
        for (int i = 0; i < len; i++) {
            float16[i] = float_to_half(float32[i]);
        }
    }

    void CpuToFloat16::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        if (data.dataType == DataType::FLOAT16) {
            return;
        }
        if (data.dims.size() == 0) {
            data.dataType = DataType::FLOAT16;
            data.UpdateUnitSize();
            return;
        }
        if (data.dataType == DataType::FLOAT32) {
            float *old = (float*)data.cpuData;
            data.dataType = DataType::FLOAT16;
            data.UpdateUnitSize();
            data.cpuData = new uint8_t[data.GetBytes()];
            uint16_t *cur = (uint16_t*)data.cpuData;
            int len = data.Count(0);
            for (int i = 0; i < len; i++) {
                cur[i] = float_to_half(old[i]);
            }
            delete[] old;
        } else {
            ErrorInFastLLM("ToFloat16: unsupport dataType.\n");
        }
    }

    void CpuToFloat32::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        if (data.dataType == DataType::FLOAT32) {
            return;
        }
        if (data.dims.size() == 0) {
            data.dataType = DataType::FLOAT32;
            data.UpdateUnitSize();
            return;
        }
        if (data.dataType == DataType::FLOAT16) {
            uint16_t *old = (uint16_t*)data.cpuData;
            data.dataType = DataType::FLOAT32;
            data.UpdateUnitSize();
            data.cpuData = new uint8_t[data.GetBytes()];
            float *cur = (float*)data.cpuData;
            int len = data.Count(0);
            for (int i = 0; i < len; i++) {
                cur[i] = fp16tofp32.dict[old[i]];
            }
            delete[] old;
        } else {
            ErrorInFastLLM("ToFloat32: unsupport dataType.\n");
        }
    }

    void CpuAttention::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &k = *(datas.find("k")->second);
        Data &v = *(datas.find("v")->second);
        Data &output = *(datas.find("output")->second);
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : 1;

        AssertInFastLLM(q.dims.size() == 3 && k.dims.size() == 3 && v.dims.size() == 3, "Attention: dims of q, k, v should be 3.\n");
        AssertInFastLLM(q.dims[2] == k.dims[2], "Attention: q.dims[2] should be equal to k.dims[2].\n");
        AssertInFastLLM(k.dims[1] == v.dims[1], "Attention: k.dims[1] should be equal to v.dims[1].\n");
        AssertInFastLLM(k.dims[0] == v.dims[0], "Attention: k.dims[0] should be equal to v.dims[0].\n");
        AssertInFastLLM(q.dims[0] == k.dims[0] * group, "Attention: q.dims[0] should be equal to k.dims[0] * group.\n");

        AssertInFastLLM(q.dataType == k.dataType && q.dataType == v.dataType,
                        "Attention: q, k, v's datatype should be same.\n");
        AssertInFastLLM(q.dataType == DataType::FLOAT32 ||
                        q.dataType == DataType::FLOAT16, 
                        "Attention's input's type should be float32.\n");

        std::vector <int> dims = {q.dims[0], q.dims[1], v.dims[2]};
        output.dataType = q.dataType;
        output.Resize(dims);
    }

    struct MultiThreadSingleAttentionOp : MultiThreadBaseOp {
        float *qd, *kd, *vd, *maskd, *od;
        float scale;
        int q1, q2, k1, v2;

        MultiThreadSingleAttentionOp(float *qd, float *kd, float *vd, float *maskd, float *od,
                         float scale, int q1, int q2, int k1, int v2) :
                         qd(qd), kd(kd), vd(vd), maskd(maskd), od(od), 
                         scale(scale), q1(q1), q2(q2), k1(k1), v2(v2) {}
        
        void Run() {
            float *qk = new float[k1];
            float *temp = new float[k1];
            for (int i = 0; i < q1; i++) {
                float maxValue = -10000, sum = 0.0;
                for (int j = 0; j < k1; j++) {
                    if (maskd && maskd[i * k1 + j] > 0.99) {
                        qk[j] = -10000;
                        continue;
                    }
                    float now = 0.0f;
                    int l = 0;
#ifdef __aarch64__
                    float32x4_t sum = {0, 0, 0, 0};
                    for (; l + 3 < q2; l += 4) {
                        sum = vaddq_f32(sum, vmulq_f32(vld1q_f32(qd + i * q2 + l),
                                                    vld1q_f32(kd + j * q2 + l)));
                    }
                    now += sum[0] + sum[1] + sum[2] + sum[3];
#elif defined(__AVX__)
                    __m256 vsum = _mm256_set1_ps(0.0f);
                    for (; l + 7 < q2; l += 8) {
                        __m256 vx = _mm256_loadu_ps((const float *) (qd + i * q2 + l));
                        __m256 vy = _mm256_loadu_ps((const float *) (kd + j * q2 + l));
                        vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vx, vy));
                    }
                    now += Floatsum(vsum);
#endif
                    for (; l < q2; l++) {
                        now += qd[i * q2 + l] * kd[j * q2 + l];
                    }
                    qk[j] = now * scale;
                    maxValue = std::max(maxValue, now * scale);
                }

                int j = 0;
#ifdef __aarch64__
                float32x4_t vmax = vdupq_n_f32(maxValue);
                for (; j + 3 < k1; j += 4) {
                    vst1q_f32(temp + j, exp_ps(vsubq_f32(vld1q_f32(qk + j), vmax)));
                }
#endif
                for (; j < k1; j++) {
                    temp[j] = expf(qk[j] - maxValue);
                }

                sum = 0.0f;
                for (int j = 0; j < k1; j++) {
                    sum += temp[j];
                }
                sum = std::max(sum, 0.1f);
                for (int j = 0; j < k1; j++) {
                    qk[j] = temp[j] / sum;
                }
                for (int j = 0; j < k1; j++) {
                    for (int l = 0; l < v2; l++) {
                        od[i * v2 + l] += qk[j] * vd[j * v2 + l];
                    }
                }
            }
            delete[] qk;
            delete[] temp;
        }
    };

    struct MultiThreadSingleAttentionFloat16Op : MultiThreadBaseOp {
        uint16_t *qd, *kd, *vd, *maskd, *od;
        float scale;
        int q1, q2, k1, v2;

        MultiThreadSingleAttentionFloat16Op(uint16_t *qd, uint16_t *kd, uint16_t *vd, uint16_t *maskd, uint16_t *od,
                         float scale, int q1, int q2, int k1, int v2) :
                         qd(qd), kd(kd), vd(vd), maskd(maskd), od(od), 
                         scale(scale), q1(q1), q2(q2), k1(k1), v2(v2) {}
        
        void Run() {
            std::vector <float> fqd, fkd, fvd, fmaskd, fod;
        
            fqd.resize(q1 * q2);
            fkd.resize(k1 * q2);
            fvd.resize(k1 * v2);
            fmaskd.resize(maskd ? q1 * k1 : 0);
            fod.resize(q1 * v2);

            Float16ToFloat32(qd, fqd.data(), (int)fqd.size());
            Float16ToFloat32(kd, fkd.data(), (int)fkd.size());
            Float16ToFloat32(vd, fvd.data(), (int)fvd.size());
            if (maskd) {
                Float16ToFloat32(maskd, fmaskd.data(), (int)fmaskd.size());
            }

            MultiThreadSingleAttentionOp(fqd.data(), fkd.data(), fvd.data(), maskd ? fmaskd.data() : nullptr, fod.data(), 
                            scale, q1, q2, k1, v2).Run();

            Float32ToFloat16(fod.data(), od, (int)fod.size());
        }
    };

    void CpuAttention::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &k = *(datas.find("k")->second);
        Data &v = *(datas.find("v")->second);
        Data &mask = *(datas.find("mask")->second);
        Data &output = *(datas.find("output")->second);
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : 1;
        float scale = floatParams.find("scale") != floatParams.end() ? floatParams.find("scale")->second : 1.0;
        output.Allocate();
        int q0 = q.dims[0], q1 = q.dims[1], q2 = q.dims[2], k0 = k.dims[0], k1 = k.dims[1], v2 = v.dims[2];

        if (q.dataType == DataType::FLOAT32) {
            float *qd = (float*)q.cpuData;
            float *kd = (float*)k.cpuData;
            float *vd = (float*)v.cpuData;
            float *maskd = (datas.find("mask")->second && mask.dims.size() > 0) ? (float*)mask.cpuData : nullptr;
            float *od = (float*)output.cpuData;
            int batch = (maskd != nullptr && mask.dims.size() == 3) ? mask.dims[0] : 1; 
            batch = intParams.find("mask___batch") != intParams.end() ? intParams.find("mask___batch")->second : batch;
            int maskStride = (maskd != nullptr) ? (mask.dims.size() == 3 ? mask.strides[0] : mask.Count(0)) : 0;
            std::fill(od, od + output.Count(0), 0.0f);

            auto *pool = GetAlivePool();
            int threads = pool->threads.size();
            std::vector<fastllm::MultiThreadSingleAttentionOp*> ops;
            for (int o = 0; o < q0; o++) {
                ops.push_back(new MultiThreadSingleAttentionOp(qd + o * q.strides[0], kd + (o / group) * k.strides[0], vd + (o / group) * v.strides[0],
                                maskd + (o / (q0 / batch)) * maskStride, od + o * output.strides[0], scale,
                                q1, q2, k1, v2));
            }
            for (int st = 0; st < ops.size(); st += threads) {
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->PushOp(i - st, ops[i]);
                }
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->Wait(i - st);
                }
            }
        } else if (q.dataType == DataType::FLOAT16) {
            uint16_t *qd = (uint16_t*)q.cpuData;
            uint16_t *kd = (uint16_t*)k.cpuData;
            uint16_t *vd = (uint16_t*)v.cpuData;
            uint16_t *maskd = (datas.find("mask")->second && mask.dims.size() > 0) ? (uint16_t*)mask.cpuData : nullptr;
            uint16_t *od = (uint16_t*)output.cpuData;
            int batch = (maskd != nullptr && mask.dims.size() == 3) ? mask.dims[0] : 1; 
            batch = intParams.find("mask___batch") != intParams.end() ? intParams.find("mask___batch")->second : batch;
            int maskStride = (maskd != nullptr) ? (mask.dims.size() == 3 ? mask.strides[0] : mask.Count(0)) : 0;
            std::fill(od, od + output.Count(0), float_to_half(0.0f));

            auto *pool = GetAlivePool();
            int threads = pool->threads.size();
            std::vector<fastllm::MultiThreadSingleAttentionFloat16Op*> ops;
            for (int o = 0; o < q0; o++) {
                ops.push_back(new MultiThreadSingleAttentionFloat16Op(qd + o * q.strides[0], kd + (o / group) * k.strides[0], vd + (o / group) * v.strides[0],
                                maskd + (o / (q0 / batch)) * maskStride, od + o * output.strides[0], scale,
                                q1, q2, k1, v2));
            }
            for (int st = 0; st < ops.size(); st += threads) {
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->PushOp(i - st, ops[i]);
                }
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->Wait(i - st);
                }
            }
        } else {
            ErrorInFastLLM("Attention error: unsupport dataType.\n");
        }
    }

    void OnlineQuantization(float *inputData, std::vector<uint8_t> &uinput, std::vector<LowBitConfig> &inputConfigs, 
                            int n, int m, int group, int groupCnt,
                            std::vector <float> &inputSums, std::vector <float> &iscales, std::vector <float> &izeros) {
        inputConfigs.resize(n * group);
        uinput.resize(n * m);
        inputSums.resize(n * group);
        iscales.resize(n * group);
        izeros.resize(n * group);

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
                                end - cur, m, group, groupCnt,
                                inputSums.data() + cur * group, iscales.data() + cur * group, izeros.data() + cur * group));
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
            MultiThreadOnlineQuantizationOp(inputData, uinput.data(), inputConfigs.data(), n, m, group, groupCnt,
                                            inputSums.data(), iscales.data(), izeros.data()).Run();
        }
    }

    struct MultiThreadSwigluOp : MultiThreadBaseOp {
        float *input, *output;
        int mid, len, n, inputStride, outputStride;

        MultiThreadSwigluOp (float *input, int mid, int len, float *output,
                             int n, int inputStride, int outputStride) :
            input(input), mid(mid), len(len), output(output),
            n(n), inputStride(inputStride), outputStride(outputStride) {}

        void Run() {
            for (int o = 0; o < n; o++) {
                float *cur = (float*)input + o * inputStride;
                float *out = (float*)output + o * outputStride;
                int i = 0;
    #ifdef __aarch64__
                float32x4_t c1 = vdupq_n_f32(1.0f);
                for (; i + 3 < len; i += 4) {
                    float32x4_t vx = vld1q_f32(cur + i);
                    float32x4_t vy = vld1q_f32(cur + i + mid);
                    vx = vdivq_f32(vx, vaddq_f32(c1, exp_ps(vnegq_f32(vx))));
                    vy = vmulq_f32(vx, vy);
                    vst1q_f32(out + i, vy);
                }
    #endif
                for (; i < len; i++) {
                    float x = cur[i], y = cur[i + mid];
                    out[i] = (x / (1.0 + expf(-x))) * y;
                }
            }
        }
    };

    struct MultiThreadLinearInt4NoZeroOp : MultiThreadBaseOp {
        uint8_t *a, *b;
        int32_t *c;
        int n, m, k, kstride;
        int *weightSums;
        float *weightMins, *scales, *bias;
        LowBitConfig *config;
        float *inputSums;

        MultiThreadLinearInt4NoZeroOp(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k, int kstride,
                      int *weightSums, float *weightMins, float *scales, float *bias, LowBitConfig *config,
                      float *inputSums) :
                      a(a), b(b), c(c), n(n), m(m), k(k), kstride(kstride), 
                      weightSums(weightSums), weightMins(weightMins), scales(scales), bias(bias), config(config), inputSums(inputSums) {}

#ifdef __ARM_FEATURE_DOTPROD
        inline static void RunSomeBlock(uint8_t *weightWalk, uint8_t *inputStart, int32_t *c, 
                            int curBlock, uint32x2_t *sum, uint8x8x2_t *vi, 
                            int block, int k, int m, int kstride) {
                uint8x8_t maskHigh = vdup_n_u8(0xF0);
                uint8x8_t maskLow = vdup_n_u8(0xF);
                for (int i = 0; i < k; i++) {
                    std::vector <int> values = std::vector <int> (curBlock, 0);
                    uint8_t *inputWalk = inputStart;
                    int j = 0;

                    for (int j = 0; j < curBlock; j++) {
                        sum[j][0] = sum[j][1] = 0;
                    }
                    for (; j + 15 < m; j += 16) {
                        for (int x = 0; x < curBlock; x++) {
                            vi[x] = vld2_u8(inputWalk + j + m * x);
                        }
                        uint8x8_t ori = vld1_u8(weightWalk + (i * m + j) / 2);
                        uint8x8_t va = vand_u8(ori, maskLow);
                        uint8x8_t vb = vshr_n_u8(vand_u8(ori, maskHigh), 4);
                        for (int x = 0; x < curBlock; x++) {
                            sum[x] = vdot_u32(sum[x], va, vi[x].val[1]);
                            sum[x] = vdot_u32(sum[x], vb, vi[x].val[0]);
                        }
                    }
                    for (int x = 0; x < curBlock; x++) {
                        values[x] += sum[x][0] + sum[x][1];
                    }

                    for (; j + 1 < m; j += 2) {
                        int id = (i * m + j) / 2;
                        for (int x = 0; x < curBlock; x++) {
                            values[x] += (weightWalk[id] >> 4) * inputWalk[j + x * m];
                            values[x] += (weightWalk[id] & 0xF) * inputWalk[j + 1 + x * m];
                        }
                    }
                    
                    for (int x = 0; x < curBlock; x++) {
                        c[(block + x) * kstride + i] = values[x];
                    }
                }
        }
#endif
        void Run() {
#ifdef __ARM_FEATURE_DOTPROD
#define RUNBLOCK(x) for (; block + (x - 1) < n; block += (x)) RunSomeBlock(b, a + block * m, c, (x), sum, vi, block, k, m, kstride);
            int block = 0;
            uint32x2_t sum[16];
            uint8x8x2_t vi[16];
            RUNBLOCK(16);
            RUNBLOCK(8);RUNBLOCK(7);RUNBLOCK(6);RUNBLOCK(5);
            RUNBLOCK(4);RUNBLOCK(3);RUNBLOCK(2);RUNBLOCK(1);
#undef RUNBLOCK
#else
            int block = 0;

            for (; block < n; block++) {
                uint8_t *weightWalk = b;
                uint8_t *inputStart = a + block * m;

                for (int i = 0; i < k; i++) {
                    int value = 0;
                    uint8_t *inputWalk = inputStart;
                    int j = 0;
#ifdef __ARM_FEATURE_DOTPROD
                    uint8x8_t maskHigh = vdup_n_u8(0xF0);
                    uint8x8_t maskLow = vdup_n_u8(0xF);
                    uint32x2_t sum0 = {0, 0};

                    for (; j + 15 < m; j += 16) {
                        uint8x8_t ori = vld1_u8(weightWalk + (i * m + j) / 2);
                        uint8x8x2_t in = vld2_u8(inputWalk + j);
                        uint8x8_t va = vand_u8(ori, maskLow);
                        uint8x8_t vb = vshr_n_u8(vand_u8(ori, maskHigh), 4);
                        sum0 = vdot_u32(sum0, va, in.val[1]);
                        sum0 = vdot_u32(sum0, vb, in.val[0]);
                    }
                    value += sum0[0] + sum0[1];
#elif defined(__aarch64__)
                    uint8x8_t maskHigh = vdup_n_u8(0xF0);
                    uint8x8_t maskLow = vdup_n_u8(0xF);
                    uint32x4_t sum0 = {0, 0, 0, 0};

                    for (; j + 15 < m; j += 16) {
                        uint8x8_t ori = vld1_u8(weightWalk + (i * m + j) / 2);
                        uint8x8x2_t in = vld2_u8(inputWalk + j);
                        uint8x8_t va = vand_u8(ori, maskLow);
                        uint8x8_t vb = vshr_n_u8(vand_u8(ori, maskHigh), 4);
                        sum0 = vpadalq_u16(sum0, vmull_u8(va, in.val[1]));
                        sum0 = vpadalq_u16(sum0, vmull_u8(vb, in.val[0]));
                    }
                    value += sum0[0] + sum0[1] + sum0[2] + sum0[3];
#elif defined(__AVX2__)
                    value += DotU4U8(weightWalk + i * m / 2, inputWalk, m);
                    j += m;
#endif

                    for (; j + 1 < m; j += 2) {
                        int id = (i * m + j) / 2;
                        value += (weightWalk[id] >> 4) * inputWalk[j];
                        value += (weightWalk[id] & 0xF) * inputWalk[j + 1];
                    }

                    c[block * kstride + i] = value;
                }
            }
#endif
            for (int block = 0; block < n; block++) {
                for (int i = 0; i < k; i++) {
                    int value = c[block * kstride + i];
                    value -= weightSums[i] * config[block].zeroPoint;
                    ((float*)c)[block * kstride + i] = scales[i] * config[block].scale * value +
                            weightMins[i] * ((float)inputSums[block] - (int)config[block].zeroPoint * m) * config[block].scale +
                            (bias == nullptr ? 0.0 : bias[i]);
                }
            }
        }
    };
    
    struct MultiThreadLinearInt4GroupOp : MultiThreadBaseOp {
        uint8_t *a, *b;
        int32_t *c;
        int n, m, k, kstride;
        int *weightSums;
        float *weightMins;
        float *scales;
        float *bias;
        float *iscales, *izeros;
        float *inputSums;
        int group, groupCnt;

        MultiThreadLinearInt4GroupOp(
                uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k, int kstride,
                int *weightSums, float *weightMins, float *scales, float *bias,
                float *iscales, float *izeros, float *inputSums, int group, int groupCnt
        ) :
                a(a), b(b), c(c), n(n), m(m), k(k), kstride(kstride),
                weightSums(weightSums), weightMins(weightMins), scales(scales), bias(bias),
                iscales(iscales), izeros(izeros), inputSums(inputSums), group(group), groupCnt(groupCnt) {}

        void Run() {
            std::vector <float> values;
            values.resize(group);

            int block = 0;
            for (; block < n; block++) {
                uint8_t *weightWalk = b;
                uint8_t *inputStart = a + block * m;

                for (int i = 0; i < k; i++) {
                    std::fill(values.begin(), values.end(), 0.0f);
                    uint8_t *inputWalk = inputStart;
                    float sum = 0.0;

                    for (int g = 0; g < group; g++) {
                        int st = g * groupCnt, end = std::min(m, (g + 1) * groupCnt);
                        float &value = values[g];
                        int j = st;
#ifdef __ARM_FEATURE_DOTPROD
                        uint8x8_t maskHigh = vdup_n_u8(0xF0);
                        uint8x8_t maskLow = vdup_n_u8(0xF);
                        uint32x2_t sum0 = {0, 0};

                        for (; j + 15 < end; j += 16) {
                            uint8x8_t ori = vld1_u8(weightWalk + (i * m + j) / 2);
                            uint8x8x2_t in = vld2_u8(inputWalk + j);
                            uint8x8_t va = vand_u8(ori, maskLow);
                            uint8x8_t vb = vshr_n_u8(vand_u8(ori, maskHigh), 4);
                            sum0 = vdot_u32(sum0, va, in.val[1]);
                            sum0 = vdot_u32(sum0, vb, in.val[0]);
                        }
                        value += sum0[0] + sum0[1];
#elif defined(__aarch64__)
                        uint8x8_t maskHigh = vdup_n_u8(0xF0);
                        uint8x8_t maskLow = vdup_n_u8(0xF);
                        uint32x4_t sum0 = {0, 0, 0, 0};

                        for (; j + 15 < end; j += 16) {
                            uint8x8_t ori = vld1_u8(weightWalk + (i * m + j) / 2);
                            uint8x8x2_t in = vld2_u8(inputWalk + j);
                            uint8x8_t va = vand_u8(ori, maskLow);
                            uint8x8_t vb = vshr_n_u8(vand_u8(ori, maskHigh), 4);
                            sum0 = vpadalq_u16(sum0, vmull_u8(va, in.val[1]));
                            sum0 = vpadalq_u16(sum0, vmull_u8(vb, in.val[0]));
                        }
                        value += sum0[0] + sum0[1] + sum0[2] + sum0[3];
#elif defined(__AVX2__)
                        value += DotU4U8(weightWalk + (i * m + st) / 2, inputWalk + st, end - st);
                        j += (end - st);
#endif
                        for (; j + 1 < end; j += 2) {
                            int id = (i * m + j) / 2;
                            value += (weightWalk[id] >> 4) * inputWalk[j];
                            value += (weightWalk[id] & 0xF) * inputWalk[j + 1];
                        }
                    }

                    int g = 0;
#ifdef __aarch64__
                    float32x4_t vSum = vdupq_n_f32(0.0f);
                    float32x4_t vGroupCnt = vdupq_n_f32(groupCnt);
                    for (; g + 3 < group; g += 4) {
                        int iid = block * group + g;
                        int gid = i * group + g;
                        float32x4_t vValue = vld1q_f32(values.data() + g);
                        float32x4_t vWeightSum = vcvtq_f32_s32(vld1q_s32(weightSums + gid));
                        float32x4_t vWeightMin = vld1q_f32(weightMins + gid);
                        float32x4_t vScale = vld1q_f32(scales + gid);
                        float32x4_t vIzero = vld1q_f32(izeros + iid);
                        float32x4_t vIscale = vld1q_f32(iscales + iid);
                        float32x4_t vInputSum = vld1q_f32(inputSums + iid);
                        float32x4_t vMiddle = vsubq_f32(vInputSum, vmulq_f32(vIzero, vGroupCnt));
                        vValue = vsubq_f32(vValue, vmulq_f32(vWeightSum, vIzero));
                        vSum = vaddq_f32(vSum, vmulq_f32(vScale, vmulq_f32(vIscale, vValue)));
                        vSum = vaddq_f32(vSum, vmulq_f32(vWeightMin, vmulq_f32(vMiddle, vIscale)));
                    }
                    sum += vSum[0] + vSum[1] + vSum[2] + vSum[3];
#endif
                    for (; g < group; g++) {
                        int iid = block * group + g;
                        int gid = i * group + g;
                        int value = values[g];
                        value -= weightSums[gid] * izeros[iid];
                        sum += scales[gid] * iscales[iid] * value +
                            weightMins[gid] * (inputSums[iid] - izeros[iid] * groupCnt) * iscales[iid];
                    }

                    if (group * groupCnt > m) {
                        int iid = block * group + group - 1;
                        int gid = i * group + group - 1;
                        sum += weightMins[gid] * izeros[iid] * (group * groupCnt - m) * iscales[iid];
                    }

                    ((float*)c)[block * kstride + i] = sum + (bias == nullptr ? 0.0 : bias[i]);
                }
            }
        }
    };

    void MultiplyInt4GroupMultiThread(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k,
                                 int *weightSums, float *weightMins, float *scales, float *bias,
                                 std::vector <LowBitConfig> &configs, int threadNum, int group, int groupCnt);
    void MultiplyInt4GroupMultiThreadLaunch(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k,
                                 int *weightSums, float *weightMins, float *scales, float *bias,
                                 std::vector <float> &inputSums, std::vector <float> &iscales, std::vector <float> &izeros,
                                 std::vector <LowBitConfig> &configs, int startTid, int threadNum, int group, int groupCnt,
                                 std::vector<fastllm::MultiThreadBaseOp*> &ops, AliveThreadPool *pool);

    void CpuMergeMOE::Run(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        fastllm::BaseOperator *op = (fastllm::BaseOperator*)(new CpuLinearOp());

        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &logits = *(datas.find("logits")->second);
        Data **weights = (Data**)(datas.find("weights")->second);
        Data **biass = (Data**)(datas.find("biass")->second);
        int topk = intParams.find("topk") != intParams.end() ? intParams.find("topk")->second : 1;
        float sharedScale = floatParams.find("sharedScale") != floatParams.end() ? floatParams.find("sharedScale")->second : 1.0f;        
        float routeScale = floatParams.find("routeScale") != floatParams.end() ? floatParams.find("routeScale")->second : 1.0f;        
        output.Allocate();

        if (input.dataType == DataType::FLOAT32 && 
            (weights[0]->dataType == DataType::INT4_GROUP || weights[0]->dataType == DataType::INT4_NOZERO) 
            && input.dims[0] == 1) {
            int dimsLen = logits.dims.size();
            int outer = logits.Count(0) / logits.Count(dimsLen - 1);
            int channels = logits.dims[dimsLen - 1];

            std::vector <std::pair <float, int> > oriV;
            for (int j = 0; j < channels; j++) {
                oriV.push_back(std::make_pair(-((float*)logits.cpuData)[j], j));
            }
            sort(oriV.begin(), oriV.end());
            
            std::vector <std::pair <int, float> > v;
            for (int j = 0; j < topk; j++) {
                v.push_back(std::make_pair(oriV[j].second + 1, -oriV[j].first * routeScale));
            }
            v.push_back(std::make_pair(0, sharedScale));
            float *inputData = (float *) input.cpuData;

            int n = input.dims[0], m = input.dims[1];
            int group = weights[0]->group, groupCnt = weights[0]->groupCnt;
            if (weights[0]->dataType != DataType::INT4_GROUP) {
                group = 1;
                groupCnt = m;
            }

            std::vector<LowBitConfig> inputConfigs;
            std::vector<uint8_t> uinput;
            std::vector <float> inputSums;
            std::vector <float> iscales, izeros;

            OnlineQuantization((float*)input.cpuData, uinput, inputConfigs, n, m, group, groupCnt, 
                                inputSums, iscales, izeros);
            std::vector <float*> middles;
            std::vector <float*> results;
            for (int j = 0; j < v.size(); j++) {
                int idx = v[j].first;
                weights[idx * 2]->CalcWeightSum();
                weights[idx * 2 + 1]->CalcWeightSum();
                middles.push_back(new float[weights[idx * 2]->dims[0]]);
                results.push_back(new float[weights[idx * 2 + 1]->dims[0]]);
            }

            output.Allocate(0.0f);
            std::vector<fastllm::MultiThreadBaseOp*> ops;
            auto *pool = GetAlivePool();
            int threads = pool->threads.size();
            ops.resize(threads);

            std::vector <std::vector <LowBitConfig> > inputConfigsDown;
            std::vector <std::vector <uint8_t> > uinputsDown;
            std::vector <std::vector <float> > inputSumsDown;
            std::vector <std::vector <float> > iscalesDown, izerosDown;
            inputConfigsDown.resize(v.size());
            uinputsDown.resize(v.size());
            inputSumsDown.resize(v.size());
            iscalesDown.resize(v.size());
            izerosDown.resize(v.size());

            for (int st = 0; st < v.size(); st++) {
                int k = weights[v[st].first * 2]->dims[0];
                int end = st, selSum = 1; // 一共处理selSum * k个输出

                int curSum = 1;
                for (int l = st + 1; l < v.size(); l++) {
                    int curK = weights[v[l].first * 2]->dims[0];
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
                    int idx = v[l].first;
                    Data *weight = weights[idx * 2];
                    uint8_t *weightData = (uint8_t *) weight->cpuData;
                    float *outputData = middles[l];
                    float *biasData = nullptr;
                    int curK = weight->dims[0];
                    int curThread = (curK / k) * base;
                    MultiplyInt4GroupMultiThreadLaunch(uinput.data(), weightData, (int32_t *) outputData, n, m, curK,
                                            weight->weightSum.data(), weight->mins.data(), weight->scales.data(), biasData, 
                                            inputSums, iscales, izeros,
                                            inputConfigs, threadSt, curThread, group, groupCnt, ops, pool);
                    threadSt += curThread;
                }

                for (int j = 0; j < ops.size(); j++) {
                    pool->Wait(j);
                    delete ops[j];
                }

                // swiglu
                threadSt = 0;
                for (int l = st; l <= end; l++) {
                    int idx = v[l].first;
                    int spatial = weights[idx * 2]->dims[0], mid = spatial / 2;
                    float *outputData = middles[l];
                    int curK = weights[idx * 2]->dims[0];
                    int curThread = (curK / k) * base;
                    int per = mid / curThread;
                    int cur = 0;
                    for (int i = 0; i < curThread; i++) {
                        int end = (i == curThread - 1 ? mid : cur + per + (cur + per * (curThread - i) < mid));
                        ops[threadSt + i] = (new fastllm::MultiThreadSwigluOp(outputData + cur, mid, end - cur, outputData + cur,
                                                                    n, spatial, spatial));
                        cur = end;
                    }
                    for (int i = 0; i < curThread; i++) {
                        pool->PushOp(threadSt + i, ops[threadSt + i]);
                    }
                    threadSt += curThread;
                }
                for (int j = 0; j < ops.size(); j++) {
                    pool->Wait(j);
                    delete ops[j];
                }

                for (int l = st; l <= end; l++) {
                    int idx = v[l].first;
                    int mid = weights[idx * 2]->dims[0] / 2;
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

                    ops[l - st] = new MultiThreadOnlineQuantizationOp(
                                middles[l], uinputDown.data(), inputConfigs.data(),
                                n, mid, groupDown, groupCntDown,
                                inputSums.data(), iscales.data(), izeros.data());
                    pool->PushOp(l - st, ops[l - st]);
                }

                for (int l = st; l <= end; l++) {
                    pool->Wait(l - st);
                    delete ops[l - st];
                }

                threadSt = 0;
                for (int l = st; l <= end; l++) {
                    int idx = v[l].first;
                    int mid = weights[idx * 2]->dims[0] / 2;
                    int curK = weights[idx * 2]->dims[0];
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
                    MultiplyInt4GroupMultiThreadLaunch(uinputDown.data(), (uint8_t*)weightDown->cpuData, (int32_t *) results[l], 1, mid, m,
                                                weightDown->weightSum.data(), weightDown->mins.data(), weightDown->scales.data(), nullptr, 
                                                inputSums, iscales, izeros,
                                                inputConfigs, threadSt, curThread, groupDown, groupCntDown, ops, pool);
                    threadSt += curThread;               
                }

                for (int j = 0; j < ops.size(); j++) {
                    pool->Wait(j);
                    delete ops[j];
                }

                st = end;
            }

            for (int j = 0; j < v.size(); j++) {
                float value = v[j].second;
                float *fLastOutput = (float*)output.cpuData;
                float *curOutput = (float*)results[j];
                for (int k = 0; k < m; k++) {
                    fLastOutput[k] += curOutput[k] * value;
                }
            }
        } else {
            // normal
            Data gate, attenPart, moePart, w1, w2, w3;
            TopK(logits, gate, topk);
            gate.ToDevice(DataDevice::CPU);
            float *gateData = (float*)gate.cpuData;

            if (input.dims[0] == 1) {
                output.Allocate(0.0f);
                for (int j = 0; j < topk; j++) {
                    int idx = (int)(gateData[j * 2] + 1e-1);
                    float value = gateData[j * 2 + 1] * routeScale;

                    Linear(input, *weights[(idx + 1) * 2], Data(), w3);
                    Swiglu(w3, w1);
                    Linear(w1, *weights[(idx + 1) * 2 + 1], Data(), w2);
                    AddTo(output, w2, value);
                }
                    
                Linear(input, *weights[0], Data(), w3);
                Swiglu(w3, w1);
                Linear(w1, *weights[1], Data(), w2);
                AddTo(output, w2, sharedScale);
            } else {
                Data moeFinal = Data();
                moeFinal.Resize({0, input.dims[1]});
                moeFinal.Expansion(input.dims);
                for (int b = 0; b < input.dims[0]; b++) {
                    Data *currentData = &input;
                    Split(input, 0, b, b + 1, attenPart);
                    currentData = &attenPart;
                    moePart.Resize(currentData->dims);
                    moePart.Allocate(0.0f);
                    
                    for (int j = 0; j < topk; j++) {
                        int idx = (int)(gateData[(b * topk + j) * 2] + 1e-1);
                        float value = gateData[(b * topk + j) * 2 + 1] * routeScale;

                        Linear(*currentData, *weights[(idx + 1) * 2], Data(), w3);
                        Swiglu(w3, w1);
                        Linear(w1, *weights[(idx + 1) * 2 + 1], Data(), w2);
                        AddTo(moePart, w2, value);
                    }
                    
                    Linear(*currentData, *weights[0], Data(), w3);
                    Swiglu(w3, w1);
                    Linear(w1, *weights[1], Data(), w2);
                    AddTo(moePart, w2, sharedScale);

                    CatDirect(moeFinal, moePart, 0);
                }
                memcpy(output.cpuData, moeFinal.cpuData, output.GetBytes());
            }
        }
    }

    void CpuCopyKVCacheOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                   const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        return;
    }

    void CpuCopyKVCacheOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &oldCache = *(datas.find("oldCache")->second);
        Data &newCache = *(datas.find("newCache")->second);

        int oldBsStart = intParams.find("oldBsStart") != intParams.end() ? intParams.find("oldBsStart")->second : -1;
        int newBsStart = intParams.find("newBsStart") != intParams.end() ? intParams.find("newBsStart")->second : -1;
        int bs = intParams.find("bs") != intParams.end() ? intParams.find("bs")->second : -1;
        int offset = intParams.find("offset") != intParams.end() ? intParams.find("offset")->second : -1;

        int unitSize = oldCache.unitSize;
        for (int o = 0; o < bs; o++) {
            uint8_t *cur = newCache.cpuData + (newBsStart + o) * newCache.strides[0] * unitSize;
            cur += offset * newCache.strides[1] * unitSize;
            uint8_t *old = oldCache.cpuData + (oldBsStart + o) * oldCache.strides[0] * unitSize;
            memcpy(cur, old, oldCache.dims[1] * oldCache.dims[2] * unitSize);
        }
    }

    void CpuEmbedding::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);

        AssertInFastLLM(weight.dims.size() == 2, "Embedding's weight's dim should be 2.\n");
        AssertInFastLLM(weight.dataType == DataType::FLOAT32 ||
                        weight.dataType == DataType::BFLOAT16, "Embedding's weight's type should be float32 or bfloat16.\n");
        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16, 
                        "Embedding's input's type should be float32 or float16.\n");

        weight.weightType = WeightType::EMBEDDING;
        int vocabSize = weight.dims[0], embSize = weight.dims[1];
        std::vector <int> dims = input.dims;
        dims.push_back(embSize);

        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void CpuEmbedding::Run(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);;

        output.Allocate();

        int vocabSize = weight.dims[0], embSize = weight.dims[1];
        uint64_t inputLen = input.Count(0);

        float *inputData = (float*)input.cpuData;
        float *dstOutputData = (float*)output.cpuData;

        std::vector <float> tempInputData, tempOutputData;
        if (input.dataType != DataType::FLOAT32) {
            tempInputData.resize(inputLen);
            tempOutputData.resize(inputLen * embSize);
            inputData = tempInputData.data();
            dstOutputData = tempOutputData.data();

            if (input.dataType == DataType::FLOAT16) {
                for (int i = 0; i < inputLen; i++) {
                    inputData[i] = half_to_float(((uint16_t*)input.cpuData)[i]);
                }
            } else {
                ErrorInFastLLM("Embedding error: unsupport dataType.\n");
            }
        }

        if (GetLowMemMode()) {
            FILE *fi = fopen(weight.fileName.c_str(), "rb");
            if (weight.dataType == DataType::FLOAT32) {
                float *outputData = (float *) dstOutputData;
                for (int i = 0; i < inputLen; i++) {
                    int token = (int) (inputData[i] + 1e-9);
#if defined(_WIN32) or defined(_WIN64)
                    _fseeki64(fi, (long long)token * embSize * sizeof(float) + weight.filePos, 0);
#else
                    fseek(fi, (long long)token * embSize * sizeof(float) + weight.filePos, 0);
#endif
                    int ret = fread(outputData + i * embSize, sizeof(float), embSize, fi);
                }
            } else {
                uint16_t *outputData = (uint16_t *) dstOutputData;
                uint16_t *weightData = new uint16_t[embSize];
                for (int i = 0; i < inputLen; i++) {
                    int token = (int) (inputData[i] + 1e-9);
#if defined(_WIN32) or defined(_WIN64)
                    _fseeki64(fi, (long long)token * embSize * sizeof(uint16_t) + weight.filePos, 0);
#else
                    fseek(fi, (long long)token * embSize * sizeof(uint16_t) + weight.filePos, 0);
#endif
                    int ret = fread(weightData, sizeof(uint16_t), embSize, fi);
                    for (int j = 0; j < embSize; j++) {
                        outputData[i * embSize * 2 + j * 2] = 0;
                        outputData[i * embSize * 2 + j * 2 + 1] = weightData[j];
                    }
                }
                delete[] weightData;
            }
            fclose(fi);
        } else {
            if (weight.dataType == DataType::FLOAT32) {
                float *outputData = (float *) dstOutputData;
                float *weightData = (float *) weight.cpuData;
                for (int i = 0; i < inputLen; i++) {
                    int token = (int) (inputData[i] + 1e-9);
                    memcpy(outputData + i * embSize, weightData + token * embSize, embSize * sizeof(float));
                }
            } else {
                uint16_t *outputData = (uint16_t *) dstOutputData;
                uint16_t *weightData = (uint16_t *) weight.cpuData;
                for (int i = 0; i < inputLen; i++) {
                    int token = (int) (inputData[i] + 1e-9);
                    for (int j = 0; j < embSize; j++) {
                        outputData[i * embSize * 2 + j * 2] = 0;
                        outputData[i * embSize * 2 + j * 2 + 1] = weightData[token * embSize + j];
                    }
                }
            }
        }

        if (output.dataType != DataType::FLOAT32) {
            if (output.dataType == DataType::FLOAT16) {
                for (int i = 0; i < inputLen * embSize; i++) {
                    ((uint16_t*)output.cpuData)[i] = float_to_half(dstOutputData[i]);
                }
            } else {
                ErrorInFastLLM("Embedding error: unsupport dataType.\n");
            }
        }
    }

    void CpuLayerNormOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                             const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &gamma = *(datas.find("gamma")->second);
        Data &beta = *(datas.find("beta")->second);

        output.Allocate();

        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        int outer = input.Count(0) / input.Count(axis);
        int channels = input.dims[axis];
        int inner = input.strides[axis];

        float *mean = new float[inner], *var = new float[inner];
        float *inputData = (float *) input.cpuData;
        float *outputData = (float *) output.cpuData;
        float *gammaData = (float *) gamma.cpuData;
        float *betaData = (float *) beta.cpuData;

        if (inner == 1) {
            for (int i = 0; i < outer; i++) {
                float mean = 0.f, s2 = 0.f, var = 0.f;
                int j = 0;
#ifdef __aarch64__
                float32x4_t sums = vdupq_n_f32(0.0);
                    float32x4_t sums2 = vdupq_n_f32(0.0);
                    for (; j + 3 < channels; j += 4) {
                        float32x4_t vi = vld1q_f32(inputData + j);
                        sums = vaddq_f32(sums, vi);
                        sums2 = vaddq_f32(sums2, vmulq_f32(vi, vi));
                    }
                    mean = sums[0] + sums[1] + sums[2] + sums[3];
                    s2 = sums2[0] + sums2[1] + sums2[2] + sums2[3];
#endif
#ifdef __AVX2__
                __m256 sum_vec = _mm256_setzero_ps();
                __m256 squared_sum_vec = _mm256_setzero_ps();

                for (; j < channels - 7; j += 8) {
                    __m256 data_vec = _mm256_loadu_ps(inputData + j);
                    sum_vec = _mm256_add_ps(sum_vec, data_vec);

                    __m256 squared_data_vec = _mm256_mul_ps(data_vec, data_vec);
                    squared_sum_vec = _mm256_add_ps(squared_sum_vec, squared_data_vec);
                }

                float sum_array[8];
                _mm256_storeu_ps(sum_array, sum_vec);
                mean = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                            sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];

                float squared_sum_array[8];
                _mm256_storeu_ps(squared_sum_array, squared_sum_vec);
                s2 = squared_sum_array[0] + squared_sum_array[1] +
                                    squared_sum_array[2] + squared_sum_array[3] +
                                    squared_sum_array[4] + squared_sum_array[5] +
                                    squared_sum_array[6] + squared_sum_array[7];
#endif
                for (; j < channels; j++) {
                    mean += inputData[j];
                    s2 += inputData[j] * inputData[j];
                }
                mean /= channels;
                var = sqrt(s2 / channels - mean*mean + 1e-10);
                j = 0;
#ifdef __aarch64__
                float32x4_t means = vdupq_n_f32(mean);
                    float32x4_t vars = vdupq_n_f32(1.0 / var);
                    for (; j + 3 < channels; j += 4) {
                        float32x4_t va = vld1q_f32(gammaData + j), vb = vld1q_f32(betaData + j);
                        float32x4_t vi = vld1q_f32(inputData + j);
                        float32x4_t vo = vaddq_f32(vmulq_f32(vmulq_f32(vsubq_f32(vi, means), vars), va), vb);
                        vst1q_f32(outputData + j, vo);
                    }
#endif
                for (; j < channels; j++) {
                    float a = gammaData[j], b = betaData[j];
                    outputData[j] = (inputData[j] - mean) / var * a + b;
                }

                inputData += channels;
                outputData += channels;
            }
            return;
        } else {
            for (int i = 0; i < outer; i++) {
                std::fill(mean, mean + inner, 0.f);
                std::fill(var, var + inner, 0.f);
                float *inputWalk = inputData;
                for (int j = 0; j < channels; j++) {
                    for (int k = 0; k < inner; k++) {
                        mean[k] += *inputWalk++;
                    }
                }
                for (int k = 0; k < inner; k++) {
                    mean[k] /= channels;
                }
                inputWalk = inputData;
                for (int j = 0; j < channels; j++) {
                    for (int k = 0; k < inner; k++) {
                        float x = (*inputWalk++) - mean[k];
                        var[k] += x * x;
                    }
                }
                for (int k = 0; k < inner; k++) {
                    var[k] = sqrt(var[k] / channels + 1e-5);
                }

                inputWalk = inputData;
                float *outputWalk = outputData;
                for (int j = 0; j < channels; j++) {
                    float a = gammaData[j], b = betaData[j];
                    for (int k = 0; k < inner; k++) {
                        *outputWalk++ = ((*inputWalk++) - mean[k]) / var[k] * a + b;
                    }
                }

                inputData += channels * inner;
                outputData += channels * inner;
            }
            delete[] mean;
            delete[] var;
        }
    }

    struct MultiThreadRMSNormFloatOp : MultiThreadBaseOp {
        float *input, *output, *weight;
        int outer, channels;
        float eps;        

        MultiThreadRMSNormFloatOp (float *output, float *input, float *weight, int outer, int channels, float eps) : 
            input(input), output(output), weight(weight), outer(outer), channels(channels), eps(eps) {}

        void Run() {
            for (int i = 0; i < outer; i++) {
                float mean = 0.f;
                int j = 0;
#ifdef __aarch64__
                float32x4_t sums = vdupq_n_f32(0.0);
                for (; j + 3 < channels; j += 4) {
                    float32x4_t vi = vld1q_f32(input + j);
                    sums = vaddq_f32(sums, vmulq_f32(vi, vi));
                }
                mean = sums[0] + sums[1] + sums[2] + sums[3];
#endif
                for (; j < channels; j++) {
                    mean += input[j] * input[j];
                }
                float scale = 1.0 / sqrt(mean / channels + eps);
                j = 0;
#ifdef __aarch64__
                float32x4_t vscale = vdupq_n_f32(scale);
                for (; j + 3 < channels; j += 4) {
                    float32x4_t vi = vld1q_f32(input + j);
                    float32x4_t vw = vld1q_f32(weight + j);
                    vst1q_f32(output + j, vmulq_f32(vmulq_f32(vi, vscale), vw));
                }
#endif
                for (; j < channels; j++) {
                    output[j] = input[j] * scale * weight[j];
                }

                input += channels;
                output += channels;
            }
        }
    };

    static void RunMultiThreadRMSNormFloat(float *output, float *input, float *weight, int outer, int channels, float eps, AliveThreadPool *pool) {
        if (outer == 1) {
            (MultiThreadRMSNormFloatOp(output, input, weight, outer, channels, eps)).Run();
            return;
        }
        int threadNum = pool->threads.size();
        int per = outer / pool->threads.size();
        int cur = 0;
        std::vector<fastllm::MultiThreadRMSNormFloatOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? outer : cur + per + (cur + per * (threadNum - i) < outer));
            ops.push_back(new MultiThreadRMSNormFloatOp(output + cur * channels, input + cur * channels, weight, end - cur, channels, eps));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void CpuRMSNormOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                      const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();

        float eps = floatParams.find("eps") != floatParams.end() ? floatParams.find("eps")->second : 1e-5;
        int dimsLen = input.dims.size();
        int axis = dimsLen - 1;
        int outer = input.Count(0) / input.Count(axis);
        int channels = input.dims[axis];

        if (input.dataType == DataType::FLOAT32) {
            float *inputData = (float *) input.cpuData;
            float *outputData = (float *) output.cpuData;
            float *weightData = (float *) weight.cpuData;
            RunMultiThreadRMSNormFloat(outputData, inputData, weightData, outer, channels, eps, GetAlivePool());
        } else if (input.dataType == DataType::FLOAT16) {
            uint16_t *inputData = (uint16_t *) input.cpuData;
            uint16_t *outputData = (uint16_t *) output.cpuData;
            float *weightData = (float *) weight.cpuData;

            for (int i = 0; i < outer; i++) {
                float mean = 0.f;
                int j = 0;
                for (; j < channels; j++) {
                    float x = fp16tofp32.dict[inputData[j]];
                    mean += x * x;
                }
                float scale = 1.0 / sqrt(mean / channels + eps);
                j = 0;
                for (; j < channels; j++) {
                    outputData[j] = float_to_half(fp16tofp32.dict[inputData[j]] * scale * weightData[j]);
                }

                inputData += channels;
                outputData += channels;
            }
        } else {
            ErrorInFastLLM("RMSNorm error: unsupport dataType.\n");
        }
    }

    void CpuLinearOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
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

    struct MultiThreadFloatLinearOp : MultiThreadBaseOp {
        float *inputData;
        float *weightData;
        float *biasData, *outputData;
        int n, m, k, st, end;

        MultiThreadFloatLinearOp(float *inputData, float *weightData, float *biasData, float *outputData,
                           int n, int m, int k, int st, int end) : 
            inputData(inputData), weightData(weightData), biasData(biasData), outputData(outputData),
            n(n), m(m), k(k), st(st), end(end) {}

        void Run() {
            for (int i = 0; i < n; i++) {
                for (int j = st; j < end; j++) {
                    float now = biasData ? biasData[j] : 0.0f;
                    int l = 0;
#ifdef __aarch64__
                    float32x4_t sum = {0, 0, 0, 0};
                    for (; l + 3 < m; l += 4) {
                        sum = vaddq_f32(sum, vmulq_f32(vld1q_f32(inputData + i * m + l), vld1q_f32(weightData + j * m + l)));
                    }
                    now += sum[0] + sum[1] + sum[2] + sum[3];
#else
#ifdef __AVX2__
                    __m256 vsum = _mm256_setzero_ps();
                    for (; l + 7 < m; l += 8) {
                        __m256 vi = _mm256_loadu_ps(inputData + i * m + l);
                        __m256 vw = _mm256_loadu_ps(weightData + j * m + l);
                        vsum = _mm256_fmadd_ps(vi, vw, vsum);
                    }
                    now += Floatsum(vsum);
#endif
#endif
                    for (; l < m; l++) {
                        now += inputData[i * m + l] * weightData[j * m + l];
                    }
                    outputData[i * k + j] = now;
                }
            }
        }
    };

    struct MultiThreadFloat16LinearOp : MultiThreadBaseOp {
        float *inputData;
        uint16_t *weightData;
        float *biasData, *outputData;
        int n, m, k, st, end;

        MultiThreadFloat16LinearOp(float *inputData, uint16_t *weightData, float *biasData, float *outputData,
                           int n, int m, int k, int st, int end) : 
            inputData(inputData), weightData(weightData), biasData(biasData), outputData(outputData),
            n(n), m(m), k(k), st(st), end(end) {}

        void Run() {
            for (int i = 0; i < n; i++) {
                for (int j = st; j < end; j++) {
                    float now = biasData ? biasData[j] : 0.0f;
                    int l = 0;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                    float16x8_t sum = {0, 0, 0, 0, 0, 0, 0, 0};
                    for (; l + 7 < m; l += 8) {
                        sum = vfmaq_f16(sum, vld1q_f16((float16_t*)inputData + i * m + l),
                                            vld1q_f16((float16_t*)weightData + j * m + l));
                    }
                    now += sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
#else
#ifdef __aarch64__
                    float32x4_t sum = {0, 0, 0, 0};
                    for (; l + 3 < m; l += 4) {
                        float32x4_t vcur = {fp16tofp32.dict[weightData[j * m + l]], fp16tofp32.dict[weightData[j * m + l + 1]],
                                            fp16tofp32.dict[weightData[j * m + l + 2]], fp16tofp32.dict[weightData[j * m + l + 3]]};
                        sum = vaddq_f32(sum, vmulq_f32(vld1q_f32(inputData + i * m + l), vcur));
                    }
                    now += sum[0] + sum[1] + sum[2] + sum[3];
#else
#ifdef __AVX2__
                    __m256 vsum = _mm256_setzero_ps();
                    for (; l + 7 < m; l += 8) {
                        __m256 vi = _mm256_loadu_ps(inputData + i * m + l);
                        __m256 vw = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *) (weightData + j * m + l)));
                        vsum = _mm256_fmadd_ps(vi, vw, vsum);
                    }
                    now += Floatsum(vsum);
#endif
#endif
#endif
                    for (; l < m; l++) {
                        now += inputData[i * m + l] * fp16tofp32.dict[weightData[j * m + l]];
                    }
                    outputData[i * k + j] = now;
                }
            }
        }
    };

    struct MultiThreadFloat16Float16LinearOp : MultiThreadBaseOp {
        uint16_t *inputData;
        uint16_t *weightData;
        float *biasData;
        uint16_t *outputData;
        int n, m, k, st, end;

        MultiThreadFloat16Float16LinearOp(uint16_t *inputData, uint16_t *weightData, float *biasData, uint16_t *outputData,
                           int n, int m, int k, int st, int end) : 
            inputData(inputData), weightData(weightData), biasData(biasData), outputData(outputData),
            n(n), m(m), k(k), st(st), end(end) {}

        void Run() {
            for (int i = 0; i < n; i++) {
                for (int j = st; j < end; j++) {
                    float now = biasData ? biasData[j] : 0.0f;
                    int l = 0;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                    float16x8_t sum = {0, 0, 0, 0, 0, 0, 0, 0};
                    for (; l + 7 < m; l += 8) {
                        sum = vfmaq_f16(sum, vld1q_f16((float16_t*)inputData + i * m + l),
                                            vld1q_f16((float16_t*)weightData + j * m + l));
                    }
                    now += sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
#endif
                    for (; l < m; l++) {
                        now += inputData[i * m + l] * fp16tofp32.dict[weightData[j * m + l]];
                    }
                    outputData[i * k + j] = float_to_half(now);
                }
            }
        }
    };

    // float的input, int8的weight, 直接计算得到float的output
    void Int8LinearPart(float *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        LowBitConfig *configs, int n, int m, int k, int st, int end) {
        for (int i = 0; i < n; i++) {
            for (int j = st; j < end; j++) {
                float now = biasData ? biasData[j] : 0.0f;
                int l = 0;
#ifdef __aarch64__
                float32x4_t scales = vdupq_n_f32(configs[j].scale);
                uint8x8_t zeros = vdup_n_u8(configs[j].zeroPoint);
                float32x4_t sum0 = {0, 0, 0, 0};
                float32x4_t sum1 = {0, 0, 0, 0};
                for (; l + 7 < m; l += 8) {
                    uint8x8_t a = vld1_u8(weightData + j * m + l);
                    uint16x8_t result = vsubl_u8(a, zeros);
                    int16x8_t sresult = vreinterpretq_s16_u16(result);
                    int16x4_t result1 = vget_low_s16(sresult);
                    int16x4_t result2 = vget_high_s16(sresult);
                    int32x4_t result3 = vmovl_s16(result1);
                    int32x4_t result4 = vmovl_s16(result2);
                    float32x4_t f1 = vmulq_f32(scales, vcvtq_f32_s32(result3));
                    float32x4_t f2 = vmulq_f32(scales, vcvtq_f32_s32(result4));

                    sum0 = vaddq_f32(sum0, vmulq_f32(vld1q_f32(inputData + i * m + l + 0), f1));
                    sum1 = vaddq_f32(sum1, vmulq_f32(vld1q_f32(inputData + i * m + l + 4), f2));
                }
                now += sum0[0] + sum0[1] + sum0[2] + sum0[3];
                now += sum1[0] + sum1[1] + sum1[2] + sum1[3];
#endif

                for (; l < m; l++) {
                    now += inputData[i * m + l] * configs[j].invQuantization(weightData[j * m + l]);
                }

                outputData[i * k + j] = now;
            }
        }
    }

    // float的input, int4g的weight, 直接计算得到float的output
    void Int4GroupLinearPart(float *inputData, uint8_t *weightData, float *biasData, float *outputData,
                            LowBitConfig *configs, int n, int m, int k, int st, int end, int group, int groupCnt) {
        for (int i = 0; i < n; i++) {
            for (int j = st; j < end; j++) {
                float now = biasData ? biasData[j] : 0.0f;
                
                for (int g = 0; g < group; g++) {
                    int gst = g * groupCnt;
                    int gend = std::min((g + 1) * groupCnt, m);
                    int l = gst;
#ifdef __aarch64__
                    float32x4_t scales = vdupq_n_f32(configs[j * group + g].scale);
                    uint8x8_t zeros = vdup_n_u8(configs[j * group + g].zeroPoint);
                    uint8x8_t maskHigh = vdup_n_u8(0xF0);
                    uint8x8_t maskLow = vdup_n_u8(0xF);
                    float32x4_t sum0 = {0, 0, 0, 0};
                    float32x4_t sum1 = {0, 0, 0, 0};

                    for (; l + 15 < gend; l += 16) {
                        uint8x8_t ori = vld1_u8(weightData + (j * m + l) / 2);
                        float32x4x2_t in0 = vld2q_f32(inputData + i * m + l + 0);
                        float32x4x2_t in1 = vld2q_f32(inputData + i * m + l + 8);
                        uint8x8_t a = vand_u8(ori, maskLow);
                        uint16x8_t result = vsubl_u8(a, zeros);
                        int16x8_t sresult = vreinterpretq_s16_u16(result);
                        int16x4_t result1 = vget_low_s16(sresult);
                        int16x4_t result2 = vget_high_s16(sresult);
                        int32x4_t result3 = vmovl_s16(result1);
                        int32x4_t result4 = vmovl_s16(result2);
                        float32x4_t f1 = vmulq_f32(scales, vcvtq_f32_s32(result3));
                        float32x4_t f2 = vmulq_f32(scales, vcvtq_f32_s32(result4));
                        sum0 = vaddq_f32(sum0, vmulq_f32(in0.val[1], f1));
                        sum1 = vaddq_f32(sum1, vmulq_f32(in1.val[1], f2));

                        a = vshr_n_u8(vand_u8(ori, maskHigh), 4);
                        result = vsubl_u8(a, zeros);
                        sresult = vreinterpretq_s16_u16(result);
                        result1 = vget_low_s16(sresult);
                        result2 = vget_high_s16(sresult);
                        result3 = vmovl_s16(result1);
                        result4 = vmovl_s16(result2);
                        f1 = vmulq_f32(scales, vcvtq_f32_s32(result3));
                        f2 = vmulq_f32(scales, vcvtq_f32_s32(result4));

                        sum0 = vaddq_f32(sum0, vmulq_f32(in0.val[0], f1));
                        sum1 = vaddq_f32(sum1, vmulq_f32(in1.val[0], f2));
                    }
                    now += sum0[0] + sum0[1] + sum0[2] + sum0[3];
                    now += sum1[0] + sum1[1] + sum1[2] + sum1[3];
#endif
                    for (; l < gend; l++) {
                        int id = (j * m + l) / 2;
                        float weight = 0.0f;
                        if ((j * m + l) % 2) {
                            weight = configs[j * group + g].invQuantization(weightData[id] & 0xF);
                        } else {
                            weight = configs[j * group + g].invQuantization(weightData[id] >> 4);
                        }
                        now += inputData[i * m + l] * weight;
                    }
                }

                outputData[i * k + j] = now;
            }
        }
    }

    // float的input, int4的weight, 直接计算得到float的output
    void Int4LinearPart(float *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        LowBitConfig *configs, int n, int m, int k, int st, int end) {
        for (int i = 0; i < n; i++) {
            for (int j = st; j < end; j++) {
                float now = biasData ? biasData[j] : 0.0f;
                int l = 0;
#ifdef __aarch64__X
                float32x4_t scales = vdupq_n_f32(configs[j].scale);
                uint8x8_t zeros = vdup_n_u8(configs[j].zeroPoint);
                uint8x8_t maskHigh = vdup_n_u8(0xF0);
                uint8x8_t maskLow = vdup_n_u8(0xF);
                float32x4_t sum0 = {0, 0, 0, 0};
                float32x4_t sum1 = {0, 0, 0, 0};

                for (; l + 15 < m; l += 16) {
                    uint8x8_t ori = vld1_u8(weightData + (j * m + l) / 2);
                    float32x4x2_t in0 = vld2q_f32(inputData + i * m + l + 0);
                    float32x4x2_t in1 = vld2q_f32(inputData + i * m + l + 8);
                    uint8x8_t a = vand_u8(ori, maskLow);
                    uint16x8_t result = vsubl_u8(a, zeros);
                    int16x8_t sresult = vreinterpretq_s16_u16(result);
                    int16x4_t result1 = vget_low_s16(sresult);
                    int16x4_t result2 = vget_high_s16(sresult);
                    int32x4_t result3 = vmovl_s16(result1);
                    int32x4_t result4 = vmovl_s16(result2);
                    float32x4_t f1 = vmulq_f32(scales, vcvtq_f32_s32(result3));
                    float32x4_t f2 = vmulq_f32(scales, vcvtq_f32_s32(result4));
                    sum0 = vaddq_f32(sum0, vmulq_f32(in0.val[1], f1));
                    sum1 = vaddq_f32(sum1, vmulq_f32(in1.val[1], f2));

                    a = vshr_n_u8(vand_u8(ori, maskHigh), 4);
                    result = vsubl_u8(a, zeros);
                    sresult = vreinterpretq_s16_u16(result);
                    result1 = vget_low_s16(sresult);
                    result2 = vget_high_s16(sresult);
                    result3 = vmovl_s16(result1);
                    result4 = vmovl_s16(result2);
                    f1 = vmulq_f32(scales, vcvtq_f32_s32(result3));
                    f2 = vmulq_f32(scales, vcvtq_f32_s32(result4));

                    sum0 = vaddq_f32(sum0, vmulq_f32(in0.val[0], f1));
                    sum1 = vaddq_f32(sum1, vmulq_f32(in1.val[0], f2));
                }
                now += sum0[0] + sum0[1] + sum0[2] + sum0[3];
                now += sum1[0] + sum1[1] + sum1[2] + sum1[3];
#endif

                for (; l < m; l++) {
                    int id = (j * m + l) / 2;
                    float weight = 0.0f;
                    if ((j * m + l) % 2) {
                        weight = configs[j].invQuantization(weightData[id] & 0xF);
                    } else {
                        weight = configs[j].invQuantization(weightData[id] >> 4);
                    }
                    now += inputData[i * m + l] * weight;
                }

                outputData[i * k + j] = now;
            }
        }
    }

    struct MultiThreadMultiplyOp : MultiThreadBaseOp {
        uint8_t *a;
        uint8_t *b;
        int32_t *c;
        int n, m, k, kstride;

        MultiThreadMultiplyOp(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k, int kstride) : 
            a(a), b(b), c(c), n(n), m(m), k(k), kstride(kstride) {}
#ifdef __ARM_FEATURE_DOTPROD
        inline static void RunSomeBlock(uint8_t *weightWalk, uint8_t *inputStart, int32_t *c, 
                            int curBlock, uint32x4_t *sum, uint8x16_t *vi, 
                            int block, int k, int m, int kstride) {
                for (int i = 0; i < k; i++) {
                    std::vector <int> values = std::vector <int> (curBlock, 0);
                    uint8_t *inputWalk = inputStart;
                    int j = 0;

                    for (int j = 0; j < curBlock; j++) {
                        sum[j][0] = sum[j][1] = sum[j][2] = sum[j][3] = 0;
                    }
                    for (; j + 15 < m; j += 16) {
                        for (int x = 0; x < curBlock; x++) {
                            vi[x] = vld1q_u8(inputWalk + m * x);
                        }
                        uint8x16_t vw = vld1q_u8(weightWalk);
                        for (int x = 0; x < curBlock; x++) {
                            sum[x] = vdotq_u32(sum[x], vi[x], vw);
                        }
                        inputWalk += 16;
                        weightWalk += 16;
                    }
                    for (int x = 0; x < curBlock; x++) {
                        values[x] += sum[x][0] + sum[x][1] + sum[x][2] + sum[x][3];
                    }

                    for (; j < m; j++) {
                        int curWeight = (int)(*(weightWalk++));
                        for (int x = 0; x < curBlock; x++) {
                            values[x] += curWeight * (*(inputWalk + x * m));
                        }
                        inputWalk++;
                    }

                    for (int x = 0; x < curBlock; x++) {
                        c[(block + x) * kstride + i] = values[x];
                    }
                }
        }
#endif

        void Run() {
#ifdef __ARM_FEATURE_DOTPROD
#define RUNBLOCK(x) for (; block + (x - 1) < n; block += (x)) RunSomeBlock(b, a + block * m, c, (x), sum, vi, block, k, m, kstride);
            int block = 0;
            uint32x4_t sum[16];
            uint8x16_t vi[16];
            RUNBLOCK(16);
            RUNBLOCK(8);RUNBLOCK(7);RUNBLOCK(6);RUNBLOCK(5);
            RUNBLOCK(4);RUNBLOCK(3);RUNBLOCK(2);RUNBLOCK(1);
#undef RUNBLOCK
#elif defined(__aarch64__)
            int block = 0;
            for (; block < n; block++) {
                uint8_t *weightWalk = b;
                uint8_t *inputStart = a + block * m;

                for (int i = 0; i < k; i++) {
                    int value = 0;
                    uint8_t *inputWalk = inputStart;

                    int per = 64;
                    int cnt = m / per;
                    int sur = m % per;

                    uint32x4_t sum = {0};
                    uint16x8_t temp = {0};
                    uint16x8_t temp1 = {0};
                    uint16x8_t temp2 = {0};
                    uint16x8_t temp3 = {0};
                    uint16x8_t temp4 = {0};
                    uint16x8_t temp5 = {0};
                    uint16x8_t temp6 = {0};
                    uint16x8_t temp7 = {0};

                    while (cnt--) {
                        temp = vmull_u8(vld1_u8(inputWalk), vld1_u8(weightWalk));
                        temp1 = vmull_u8(vld1_u8(inputWalk + 8), vld1_u8(weightWalk + 8));
                        temp2 = vmull_u8(vld1_u8(inputWalk + 16), vld1_u8(weightWalk + 16));
                        temp3 = vmull_u8(vld1_u8(inputWalk + 24), vld1_u8(weightWalk + 24));
                        temp4 = vmull_u8(vld1_u8(inputWalk + 32), vld1_u8(weightWalk + 32));
                        temp5 = vmull_u8(vld1_u8(inputWalk + 40), vld1_u8(weightWalk + 40));
                        temp6 = vmull_u8(vld1_u8(inputWalk + 48), vld1_u8(weightWalk + 48));
                        temp7 = vmull_u8(vld1_u8(inputWalk + 56), vld1_u8(weightWalk + 56));

                        sum = vpadalq_u16(sum, temp);
                        sum = vpadalq_u16(sum, temp1);
                        sum = vpadalq_u16(sum, temp2);
                        sum = vpadalq_u16(sum, temp3);
                        sum = vpadalq_u16(sum, temp4);
                        sum = vpadalq_u16(sum, temp5);
                        sum = vpadalq_u16(sum, temp6);
                        sum = vpadalq_u16(sum, temp7);

                        inputWalk += per;
                        weightWalk += per;
                    }

                    value += (sum[0] + sum[1] + sum[2] + sum[3]);
                    while (sur--) {
                        value += (int)(*(weightWalk++)) * (*(inputWalk++));
                    }

                    c[block * kstride + i] = value;
                }
            }
#elif defined(__AVX2__)
            int block = 0;
            for (; block < n; block++) {
                uint8_t *weightWalk = b;
                uint8_t *inputStart = a + block * m;

                for (int i = 0; i < k; i++) {
                    uint8_t *inputWalk = inputStart;

                    c[block * kstride + i] = DotU8U8(inputWalk, weightWalk, m);
                    weightWalk += m;
                }
            }
#else
            int block = 0;
            for (; block < n; block++) {
                uint8_t *weightWalk = b;
                uint8_t *inputStart = a + block * m;

                for (int i = 0; i < k; i++) {
                    int value = 0;
                    uint8_t *inputWalk = inputStart;
                    for (int j = 0; j < m; j++) {
                        value += (int)(*(weightWalk++)) * (*(inputWalk++));
                    }

                    c[block * kstride + i] = value;
                }
            }
#endif
        }
    };

    struct MultiThreadLinearInt4Op : MultiThreadBaseOp {
        uint8_t *a;
        uint8_t *b;
        int32_t *c;
        int n, m, k, kstride;
        int *weightSums, *weightZeros;
        float *scales, *bias;
        LowBitConfig *config;
        int *inputSums;

        MultiThreadLinearInt4Op(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k, int kstride,
                      int *weightSums, int *weightZeros, float *scales, float *bias, LowBitConfig *config,
                      int *inputSums) : a(a), b(b), c(c), n(n), m(m), k(k), kstride(kstride), 
                      weightSums(weightSums), weightZeros(weightZeros), scales(scales), bias(bias), config(config), inputSums(inputSums) {}
        
        void Run() {
            int block = 0;
            for (; block < n; block++) {
                uint32_t inputSum = inputSums[block];
                uint8_t *weightWalk = b;
                uint8_t *inputStart = a + block * m;

                for (int i = 0; i < k; i++) {
                    int value = 0;
                    uint8_t *inputWalk = inputStart;
                    int j = 0;
#ifdef __ARM_FEATURE_DOTPROD
                    uint8x8_t maskHigh = vdup_n_u8(0xF0);
                    uint8x8_t maskLow = vdup_n_u8(0xF);
                    uint32x2_t sum0 = {0, 0};

                    for (; j + 15 < m; j += 16) {
                        uint8x8_t ori = vld1_u8(weightWalk + (i * m + j) / 2);
                        uint8x8x2_t in = vld2_u8(inputWalk + j);
                        uint8x8_t va = vand_u8(ori, maskLow);
                        uint8x8_t vb = vshr_n_u8(vand_u8(ori, maskHigh), 4);
                        sum0 = vdot_u32(sum0, va, in.val[1]);
                        sum0 = vdot_u32(sum0, vb, in.val[0]);
                    }
                    value += sum0[0] + sum0[1];
#elif defined(__aarch64__)
                    uint8x8_t maskHigh = vdup_n_u8(0xF0);
                    uint8x8_t maskLow = vdup_n_u8(0xF);
                    uint32x4_t sum0 = {0, 0, 0, 0};

                    for (; j + 15 < m; j += 16) {
                        uint8x8_t ori = vld1_u8(weightWalk + (i * m + j) / 2);
                        uint8x8x2_t in = vld2_u8(inputWalk + j);
                        uint8x8_t va = vand_u8(ori, maskLow);
                        uint8x8_t vb = vshr_n_u8(vand_u8(ori, maskHigh), 4);
                        sum0 = vpadalq_u16(sum0, vmull_u8(va, in.val[1]));
                        sum0 = vpadalq_u16(sum0, vmull_u8(vb, in.val[0]));
                    }
                    value += sum0[0] + sum0[1] + sum0[2] + sum0[3];
#elif defined(__AVX2__)
                    value += DotU4U8(weightWalk + i * m / 2, inputWalk, m);
                    j += m;
#endif
                    for (; j + 1 < m; j += 2) {
                        int id = (i * m + j) / 2;
                        value += (weightWalk[id] >> 4) * inputWalk[j];
                        value += (weightWalk[id] & 0xF) * inputWalk[j + 1];
                    }

                    for (; j < m; j++) {
                        int id = (i * m + j) / 2;
                        if ((i * m + j) % 2) {
                            value += (weightWalk[id] & 0xF) * inputWalk[j];
                        } else {
                            value += (weightWalk[id] >> 4) * inputWalk[j];
                        }
                    }

                    value -= weightSums[i] * config[block].zeroPoint;
                    value -= inputSum * weightZeros[i];
                    value += (int)config[block].zeroPoint * weightZeros[i] * m;

                    ((float*)c)[block * kstride + i] = scales[i] * config[block].scale * value +
                                                    (bias == nullptr ? 0.0 : bias[i]);
                }
            }
        }
    };

    //a = [n, m], b = [k, m], c = aT(b') = [n, k]
    void MultiplyMultiThread(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k, int threadNum) {
        auto *pool = GetAlivePool();
        threadNum = pool->threads.size();
        int per = k / threadNum;
        int cur = 0;
        if (threadNum == 1) {
            MultiThreadMultiplyOp(a, b + cur * m, c + cur, n, m, k - cur, k).Run();
        } else {
            std::vector<fastllm::MultiThreadMultiplyOp*> ops;
            for (int i = 0; i < threadNum; i++) {
                int end = cur + per + (cur + per * (threadNum - i) < k);
                if (i == threadNum - 1) {
                    end = k;
                }
                ops.push_back(new MultiThreadMultiplyOp(a, b + cur * m, c + cur, n, m, end - cur, k));
                cur = end;
            }
            for (int i = 0; i < threadNum; i++) {
                pool->PushOp(i, ops[i]);
            }
            for (int i = 0; i < threadNum; i++) {
                pool->Wait(i);
                delete ops[i];
            }
        }
    }

    //a = [n, m], b = [k, m], c = aT(b') = [n, k]
    void MultiplyInt4MultiThread(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k,
                                 int *weightSums, int *weightZeros, float *scales, float *bias, std::vector <LowBitConfig> &configs, int threadNum) {
        std::vector <int> inputSums;
        for (int i = 0; i < n; i++) {
            int sum = 0;
            for (int j = 0; j < m; j++) {
                sum += a[i * m + j];
            }
            inputSums.push_back(sum);
        }
        auto *pool = GetAlivePool();
        threadNum = pool->threads.size();
        int per = k / threadNum;
        int cur = 0;
        if (threadNum == 1) {
            MultiThreadLinearInt4Op(a, b + cur * m / 2, c + cur, n, m, k - cur, k,
                         weightSums + cur, weightZeros + cur, scales + cur,
                         (bias == nullptr ? (float*)nullptr : bias + cur), configs.data(), inputSums.data()).Run();
        } else {
            std::vector<fastllm::MultiThreadLinearInt4Op*> ops;
            for (int i = 0; i < threadNum; i++) {
                int end = (i == threadNum - 1 ? k : cur + per + (cur + per * (threadNum - i) < k));
                ops.push_back(new MultiThreadLinearInt4Op(a, b + cur * m / 2, c + cur, n, m, end - cur, k,
                                               weightSums + cur, weightZeros + cur, scales + cur,
                                               (bias == nullptr ? (float *) nullptr : bias + cur), configs.data(),
                                               inputSums.data()));
                cur = end;
            }
            for (int i = 0; i < threadNum; i++) {
                pool->PushOp(i, ops[i]);
            }
            for (int i = 0; i < threadNum; i++) {
                pool->Wait(i);
                delete ops[i];
            }
        }
    }

    //a = [n, m], b = [k, m], c = aT(b') = [n, k]
    void MultiplyInt4NoZeroMultiThread(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k,
                                 int *weightSums, float *weightMins, float *scales, float *bias,
                                 std::vector <LowBitConfig> &configs, int threadNum) {
        std::vector <float> inputSums;
        for (int i = 0; i < n; i++) {
            int sum = 0;
            for (int j = 0; j < m; j++) {
                sum += a[i * m + j];
            }
            inputSums.push_back(sum);
        }

        auto *pool = GetAlivePool();
        threadNum = pool->threads.size();
        int per = k / threadNum;
        int cur = 0;
        if (threadNum == 1) {
            MultiThreadLinearInt4NoZeroOp(a, b + cur * m / 2, c + cur, n, m, k - cur, k,
                         weightSums + cur, weightMins + cur, scales + cur,
                         (bias == nullptr ? (float*)nullptr : bias + cur), configs.data(), inputSums.data()).Run();
        } else {
            std::vector<fastllm::MultiThreadLinearInt4NoZeroOp*> ops;
            for (int i = 0; i < threadNum; i++) {
                int end = (i == threadNum - 1 ? k : cur + per + (cur + per * (threadNum - i) < k));
                ops.push_back(new MultiThreadLinearInt4NoZeroOp(a, b + cur * m / 2, c + cur, n, m, end - cur, k,
                                               weightSums + cur, weightMins + cur, scales + cur,
                                               (bias == nullptr ? (float *) nullptr : bias + cur), configs.data(),
                                               inputSums.data()));
                cur = end;
            }
            for (int i = 0; i < threadNum; i++) {
                pool->PushOp(i, ops[i]);
            }
            for (int i = 0; i < threadNum; i++) {
                pool->Wait(i);
                delete ops[i];
            }
        }
    }

    //a = [n, m], b = [k, m], c = aT(b') = [n, k]
    void MultiplyInt4GroupMultiThreadLaunch(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k,
                                 int *weightSums, float *weightMins, float *scales, float *bias,
                                 std::vector <float> &inputSums, std::vector <float> &iscales, std::vector <float> &izeros,
                                 std::vector <LowBitConfig> &configs, int startTid, int threadNum, int group, int groupCnt,
                                 std::vector<fastllm::MultiThreadBaseOp*> &ops, 
                                 AliveThreadPool *pool) {
        int per = k / threadNum;
        int cur = 0;
        
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? k : cur + per + (cur + per * (threadNum - i) < k));
            if (group > 1) {
                ops[startTid + i] = new MultiThreadLinearInt4GroupOp(a, b + cur * m / 2, c + cur, n, m, end - cur, k,
                                           weightSums + cur * group, weightMins + cur * group, scales + cur * group,
                                           (bias == nullptr ? (float *) nullptr : bias + cur), iscales.data(), izeros.data(),
                                           inputSums.data(), group, groupCnt);
            } else {
                ops[startTid + i] = new MultiThreadLinearInt4NoZeroOp(a, b + cur * m / 2, c + cur, n, m, end - cur, k,
                                           weightSums + cur * group, weightMins + cur * group, scales + cur * group,
                                           (bias == nullptr ? (float *) nullptr : bias + cur), configs.data(), inputSums.data());
            }
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(startTid + i, ops[startTid + i]);
        }
    }

    //a = [n, m], b = [k, m], c = aT(b') = [n, k]
    void MultiplyInt4GroupMultiThread(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k,
                                 int *weightSums, float *weightMins, float *scales, float *bias,
                                 std::vector <LowBitConfig> &configs, int threadNum, int group, int groupCnt) {
        std::vector <float> inputSums;
        for (int i = 0; i < n; i++) {
            for (int g = 0; g < group; g++) {
                int sum = 0;
                for (int j = g * groupCnt; j < (g + 1) * groupCnt && j < m; j++) {
                    sum += a[i * m + j];
                }
                inputSums.push_back(sum);
            }
        }
        std::vector <float> iscales, izeros;
        for (int i = 0; i < configs.size(); i++) {
            iscales.push_back(configs[i].scale);
            izeros.push_back(configs[i].zeroPoint);
        }

        auto *pool = GetAlivePool();
        threadNum = pool->threads.size();
        int per = k / threadNum;
        int cur = 0;
        if (threadNum == 1) {
            MultiThreadLinearInt4GroupOp(a, b, c, n, m, k, k,
                         weightSums, weightMins, scales,
                         (bias == nullptr ? (float*)nullptr : bias), iscales.data(), izeros.data(), inputSums.data(),
                         group, groupCnt).Run();
        } else {
            std::vector<fastllm::MultiThreadLinearInt4GroupOp*> ops;
            for (int i = 0; i < threadNum; i++) {
                int end = (i == threadNum - 1 ? k : cur + per + (cur + per * (threadNum - i) < k));
                ops.push_back(new MultiThreadLinearInt4GroupOp(a, b + cur * m / 2, c + cur, n, m, end - cur, k,
                                               weightSums + cur * group, weightMins + cur * group, scales + cur * group,
                                               (bias == nullptr ? (float *) nullptr : bias + cur), iscales.data(), izeros.data(),
                                               inputSums.data(), group, groupCnt));
                cur = end;
            }
            for (int i = 0; i < threadNum; i++) {
                pool->PushOp(i, ops[i]);
            }
            for (int i = 0; i < threadNum; i++) {
                pool->Wait(i);
                delete ops[i];
            }
        }
    }

    void GetArrayMinMax(float *a, int len, float &minValue, float &maxValue) {
        int j = 0;
        minValue = 1e100;
        maxValue = -1e100;
#ifdef __aarch64__
        float32x4_t mins = vdupq_n_f32(1e100);
        float32x4_t maxs = vdupq_n_f32(-1e100);
        for (; j + 3 < len; j += 4) {
            float32x4_t v = vld1q_f32(a + j);
            mins = vminq_f32(mins, v);
            maxs = vmaxq_f32(maxs, v);
        }
        for (int l = 0; l < 4; l++) {
            minValue = std::min(minValue, mins[l]);
            maxValue = std::max(maxValue, maxs[l]);
        }
#endif
        for (; j < len; j++) {
            minValue = std::min(minValue, a[j]);
            maxValue = std::max(maxValue, a[j]);
        }
    }

    void QuantizationAll(float *fValue, uint8_t *uValue, int len, LowBitConfig *config) {
        float scale = config->scale;
        float zeroPoint = config->zeroPoint;
        int j = 0;
#ifdef __aarch64__
        float32x4_t scales = vdupq_n_f32(scale);
        float32x4_t zeros = vdupq_n_f32(zeroPoint + 0.5);
        int32x4_t maxds = vcombine_s32(vcreate_s32(0x000000ff000000ff), vcreate_s32(0x000000ff000000ff));
        int32x4_t minds = vcombine_s32(vcreate_s32(0x0000000000000000), vcreate_s32(0x0000000000000000));
        for (; j + 7 < len; j += 8) {
            float32x4_t fin1 = vld1q_f32(fValue + j);
            float32x4_t fin2 = vld1q_f32(fValue + j + 4);
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
            vst1_u8(uValue + j, out);
        }
#endif
        for (; j < len; j++) {
            uValue[j] = (uint8_t) (std::min(255., (double) std::max(fValue[j] / scale + zeroPoint + 0.5, 0.0)));
        }
    }

    void MultiThreadOnlineQuantizationOp::Run() {
        for (int i = 0; i < n; i++) {
            float *cur = input + i * m;
            uint8_t *u = output + i * m;
            for (int g = 0; g < group; g++) {
                int st = g * groupCnt;
                int end = std::min(m, (g + 1) * groupCnt);
                float minValue = 1e9, maxValue = -1e9;
                GetArrayMinMax(input + i * m + st, end - st, minValue, maxValue);
                configs[i * group + g] = (LowBitConfig(minValue, maxValue, 8, 0));
                QuantizationAll(cur + st, u + st, end - st, &configs[i * group + g]);
            }
        }
#ifdef __AVX__
        uint8_t *temp = new uint8_t[32];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j + 31 < m; j += 32) {
                memcpy(temp, output + i * m + j, 32);
                for (int k = 0; k < 16; k++) {
                    output[i * m + j + k] = temp[k * 2 + 1];
                    output[i * m + j + k + 16] = temp[k * 2];
                }
            }
        }
        delete[] temp;
#endif
        if (inputSums != nullptr) {
            for (int i = 0; i < n; i++) {
                for (int g = 0; g < group; g++) {
                    iscales[i * group + g] = configs[i * group + g].scale;
                    izeros[i * group + g] = configs[i * group + g].zeroPoint;
                    int sum = 0;
                    for (int j = g * groupCnt; j < (g + 1) * groupCnt && j < m; j++) {
                        sum += output[i * m + j];
                    }
                    inputSums[i * group + g] = sum;
                }
            }
        }
    }

    bool CpuLinearOp::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        if (intParams.find("exType") != intParams.end()) {
            return false;
        }
        return true;
    }

    void CpuLinearOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
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
                float *inputData = (float *) input.cpuData;
                float *weightData = (float *) weight.cpuData;
                float *outputData = (float *) output.cpuData;
                float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;

                auto pool = GetAlivePool();
                int threadNum = pool->threads.size();
                int per = k / threadNum;
                int cur = 0;
                std::vector<fastllm::MultiThreadFloatLinearOp*> ops;
                for (int i = 0; i < threadNum; i++) {
                    int end = cur + per + (cur + per * (threadNum - i) < k);
                    ops.push_back(new MultiThreadFloatLinearOp(inputData, weightData, biasData, outputData,
                                                   n, m, k, cur, end));
                    cur = end;
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->PushOp(i, ops[i]);
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->Wait(i);
                    delete ops[i];
                }
            } else if (weight.dataType == DataType::FLOAT16) {
                float *inputData = (float *) input.cpuData;
                uint16_t *weightData = (uint16_t *) weight.cpuData;
                float *outputData = (float *) output.cpuData;
                float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                uint16_t *temp = new uint16_t[n * m];
                for (int i = 0; i < n * m; i++) {
                    temp[i] = float_to_half(inputData[i]);
                }
                inputData = (float*)temp;
#endif
                auto pool = GetAlivePool();
                int threadNum = pool->threads.size();
                int per = k / threadNum;
                int cur = 0;
                std::vector<fastllm::MultiThreadFloat16LinearOp*> ops;
                for (int i = 0; i < threadNum; i++) {
                    int end = cur + per + (cur + per * (threadNum - i) < k);
                    if (i == threadNum - 1) {
                        end = k;
                    }
                    ops.push_back(new MultiThreadFloat16LinearOp(inputData, weightData, biasData, outputData,
                                                   n, m, k, cur, end));
                    cur = end;
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->PushOp(i, ops[i]);
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->Wait(i);
                    delete ops[i];
                }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
                delete[] temp;
#endif
            } else if (weight.dataType == DataType::INT8) {
                float *inputData = (float *) input.cpuData;
                uint8_t *weightData = (uint8_t *) weight.cpuData;
                float *outputData = (float *) output.cpuData;
                float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
                weight.CalcWeightSum();

                std::vector<LowBitConfig> inputConfigs;
                for (int i = 0; i < n; i++) {
                    float minValue = 1e9, maxValue = -1e9;
                    for (int j = 0; j < m; j++) {
                        minValue = std::min(minValue, inputData[i * m + j]);
                        maxValue = std::max(maxValue, inputData[i * m + j]);
                    }
                    inputConfigs.push_back(LowBitConfig(minValue, maxValue, 8, 0));
                }
                std::vector<uint8_t> uinput;
                uinput.resize(n * m);
                for (int i = 0; i < n * m; i++) {
#ifdef __AVX2__
                    uinput[i] = inputConfigs[i / m].quantization(inputData[i]);
                    uinput[i] = (uinput[i] + !uinput[i]) ^ 128;
#else
                    uinput[i] = inputConfigs[i / m].quantization(inputData[i]);
#endif
                }

                MultiplyMultiThread(uinput.data(), weightData, (int32_t *) outputData, n, m, k, GetThreads());
                for (int i = 0; i < n; i++) {
                    uint32_t inputSum = 0;
                    for (int j = 0; j < m; j++) {
#ifdef __AVX2__
                        inputSum += uinput[i * m + j] ^ 128;
#else
                        inputSum += uinput[i * m + j];
#endif
                    }

                    for (int j = 0; j < k; j++) {
                        int value = ((int32_t *) outputData)[i * k + j];
#ifdef __AVX2__
                        value += (128 * weight.weightSum[j]);
                        value += (128 * inputSum);
                        value -= m * 128 * 128;
#endif
                        value -= weight.weightSum[j] * inputConfigs[i].zeroPoint;
                        value -= inputSum * weight.perChannelsConfigs[j].zeroPoint;
                        value += (int) inputConfigs[i].zeroPoint * weight.perChannelsConfigs[j].zeroPoint * m;
                        outputData[i * k + j] = weight.perChannelsConfigs[j].scale * inputConfigs[i].scale * value +
                                                (biasData == nullptr ? 0.0 : biasData[j]);
                    }
                }

                /*
                这部分是float输入，float输出
                int threadNum = threads;
                int per = k / threadNum;
                int cur = 0;
                std::vector<std::thread *> threads;
                for (int i = 0; i < threadNum - 1; i++) {
                    int end = cur + per + (cur + per * (threadNum - i) < k);
                    threads.push_back(new std::thread(&Int8LinearPart, inputData, weightData, biasData, outputData,
                                                      weight.perChannelsConfigs.data(), n, m, k, cur, end));
                    cur = end;
                }
                Int8LinearPart(inputData, weightData, biasData, outputData, weight.perChannelsConfigs.data(), n, m, k, cur, k);
                for (int i = 0; i < threadNum - 1; i++) {
                    threads[i]->join();
                    delete threads[i];
                }
                */
            } else if (weight.dataType == DataType::INT4 || weight.dataType == DataType::INT4_NOZERO) {
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
                    MultiplyInt4MultiThread(uinput.data(), weightData, (int32_t *) outputData, n, m, k,
                                            weight.weightSum.data(), weight.zeros.data(), weight.scales.data(),
                                            biasData,
                                            inputConfigs, GetThreads());
                } else {
                    MultiplyInt4NoZeroMultiThread(uinput.data(), weightData, (int32_t *) outputData, n, m, k,
                                                  weight.weightSum.data(), weight.mins.data(), weight.scales.data(),
                                                  biasData,
                                                  inputConfigs, GetThreads());
                }

/*
            //这部分是float输入，float输出
            int threadNum = GetThreads();
            int per = k / threadNum;
            int cur = 0;
            std::vector<std::thread *> threads;
            for (int i = 0; i < threadNum - 1; i++) {
                int end = cur + per + (cur + per * (threadNum - i) < k);
                threads.push_back(new std::thread(&Int4LinearPart, inputData, weightData, biasData, outputData,
                                                  weight.perChannelsConfigs.data(), n, m, k, cur, end));
                cur = end;
            }
            Int4LinearPart(inputData, weightData, biasData, outputData, weight.perChannelsConfigs.data(), n, m, k, cur, k);
            for (int i = 0; i < threadNum - 1; i++) {
                threads[i]->join();
                delete threads[i];
            }
*/
            } else if (weight.dataType == DataType::INT4_GROUP) {
                float *inputData = (float *) input.cpuData;
                uint8_t *weightData = (uint8_t *) weight.cpuData;
                float *outputData = (float *) output.cpuData;
                float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
                int group = weight.group, groupCnt = weight.groupCnt;
                weight.CalcWeightSum();

                std::vector<LowBitConfig> inputConfigs;
                for (int i = 0; i < n; i++) {
                    for (int g = 0; g < group; g++) {
                        int st = g * groupCnt;
                        int end = std::min(m, (g + 1) * groupCnt);
                        float minValue = 1e9, maxValue = -1e9;
                        for (int j = st; j < end; j++) {
                            minValue = std::min(minValue, inputData[i * m + j]);
                            maxValue = std::max(maxValue, inputData[i * m + j]);
                        }
                        inputConfigs.push_back(LowBitConfig(minValue, maxValue, 8, 0));
                    }
                }
                std::vector<uint8_t> uinput;
                uinput.resize(n * m);
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < m; j++) {
                        uinput[i * m + j] = inputConfigs[i * group + j / groupCnt].quantization(inputData[i * m + j]);    
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

                MultiplyInt4GroupMultiThread(uinput.data(), weightData, (int32_t *) outputData, n, m, k,
                                            weight.weightSum.data(), weight.mins.data(), weight.scales.data(),
                                            biasData, inputConfigs, GetThreads(), group, groupCnt);
/*
                //这部分是float输入，float输出
                float *inputData = (float *) input.cpuData;
                uint8_t *weightData = (uint8_t *) weight.cpuData;
                float *outputData = (float *) output.cpuData;
                float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
                int group = weight.group, groupCnt = weight.groupCnt;

                int threadNum = GetThreads();
                int per = k / threadNum;
                int cur = 0;
                std::vector<std::thread *> threads;
                for (int i = 0; i < threadNum - 1; i++) {
                    int end = cur + per + (cur + per * (threadNum - i) < k);
                    threads.push_back(new std::thread(&Int4GroupLinearPart, inputData, weightData, biasData, outputData,
                                                    weight.perChannelsConfigs.data(), n, m, k, cur, end, group, groupCnt));
                    cur = end;
                }
                Int4GroupLinearPart(inputData, weightData, biasData, outputData, weight.perChannelsConfigs.data(), n, m, k, cur, k, group, groupCnt);
                for (int i = 0; i < threadNum - 1; i++) {
                    threads[i]->join();
                    delete threads[i];
                }
*/
            } else {
                ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
            }
        } else if (input.dataType == DataType::FLOAT16 && output.dataType == DataType::FLOAT16) {
            if (weight.dataType == DataType::FLOAT16) {
                uint16_t *inputData = (uint16_t *) input.cpuData;
                uint16_t *weightData = (uint16_t *) weight.cpuData;
                uint16_t *outputData = (uint16_t *) output.cpuData;
                float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;

                auto pool = GetAlivePool();
                int threadNum = pool->threads.size();
                int per = k / threadNum;
                int cur = 0;
                std::vector<fastllm::MultiThreadFloat16Float16LinearOp*> ops;
                for (int i = 0; i < threadNum; i++) {
                    int end = cur + per + (cur + per * (threadNum - i) < k);
                    ops.push_back(new MultiThreadFloat16Float16LinearOp(inputData, weightData, biasData, outputData,
                                                   n, m, k, cur, end));
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
                ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
            }
        } else {
            ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
        }
//float spend = GetSpan(st, std::chrono::system_clock::now());
//float gops = (float)n * m * k / spend / 1e9;
// printf("n = %d, m = %d, k = %d, spend %f s, gops = %f\n", n, m, k, spend, gops);
    }

    void CpuSplitOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                             const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int start = intParams.find("start") != intParams.end() ? intParams.find("start")->second : 0;
        int end = intParams.find("end") != intParams.end() ? intParams.find("end")->second : 0;

        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        start = std::max(0, std::min(input.dims[axis] - 1, start));
        end = std::max(0, std::min(input.dims[axis], end));
        std::vector <int> dims = input.dims;
        dims[axis] = end - start;

        output.dataType = input.dataType;
        output.Resize(dims);
    }

    struct MultiThreadSliceOp : MultiThreadBaseOp {
        uint8_t *input, *output;
        int outer, inputStride, outputStride, copyLen;

        MultiThreadSliceOp (uint8_t *output, uint8_t *input, int outer, int outputStride, int inputStride, int copyLen) : 
            output(output), input(input), outer(outer), inputStride(inputStride), outputStride(outputStride), copyLen(copyLen) {}

        void Run() {
            for (int o = 0; o < outer; o++) {
                memcpy(output + o * outputStride, input + o * inputStride, copyLen);
            }
        }
    };

    static void RunMultiThreadSlice(uint8_t *output, uint8_t *input, int outer, int inputStride, int outputStride, int copyLen, AliveThreadPool *pool) {
        if (outer == 1) {
            (MultiThreadSliceOp(output, input, outer, outputStride, inputStride, copyLen)).Run();
            return;
        }
        int threadNum = pool->threads.size();
        int per = outer / pool->threads.size();
        int cur = 0;
        std::vector<fastllm::MultiThreadSliceOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? outer : cur + per + (cur + per * (threadNum - i) < outer));
            ops.push_back(new MultiThreadSliceOp(output + cur * outputStride, input + cur * inputStride, end - cur, outputStride, inputStride, copyLen));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void CpuSplitOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                         const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);

        output.Allocate();

        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int start = intParams.find("start") != intParams.end() ? intParams.find("start")->second : 0;
        int end = intParams.find("end") != intParams.end() ? intParams.find("end")->second : 0;

        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        start = std::max(0, std::min(input.dims[axis] - 1, start));
        end = std::max(0, std::min(input.dims[axis], end));

        int outer = input.Count(0) / input.Count(axis);
        int inputStride = input.Count(axis);
        int outputStride = output.Count(axis);
        int channels = input.dims[axis];
        int inner = input.strides[axis];
        int unitSize = input.unitSize;
        
        RunMultiThreadSlice(output.cpuData, input.cpuData + start * inner * unitSize, outer, 
            inputStride * unitSize, outputStride * unitSize, (end - start) * inner * unitSize, GetAlivePool());
    }

    void CpuRepeatOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                            const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int repeatTimes = intParams.find("repeatTimes") != intParams.end() ? intParams.find("repeatTimes")->second : 1;

        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        std::vector <int> dims = input.dims;
        dims[axis] *= repeatTimes;

        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void CpuRepeatOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        int repeatTimes = intParams.find("repeatTimes") != intParams.end() ? intParams.find("repeatTimes")->second : 1;
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        output.Allocate();

        int outer = output.Count(0) / output.Count(axis);
        int inputStride = input.Count(axis);
        int outputStride = output.Count(axis);
        int channels = input.dims[axis];
        int inner = input.strides[axis];
        int unitSize = input.unitSize;

        for (int o = 0; o < outer; o++) {
            for (int t = 0; t < repeatTimes; t++) {
                memcpy(output.cpuData + o * outputStride * unitSize + t * channels * inner * unitSize,
                    input.cpuData + (o * inputStride) * unitSize,
                    channels * inner * unitSize);
            }
        }
    }

    void CpuCatOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);

        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;

        if (input0.dims.size() == 0 && input1.dims.size() > 0) {
            output.Resize(input1.dims);
            return;
        }
        if (input1.dims.size() == 0 && input0.dims.size() > 0) {
            output.Resize(input0.dims);
            return;
        }

        AssertInFastLLM((input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) ||
                        (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16),
                        "Cat's input's type should be float32 or float16.\n");
        AssertInFastLLM(input0.dims.size() == input1.dims.size(), "Cat Error: input's shape's size should be same.");

        int dimsLen = input0.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        for (int i = 0; i < dimsLen; i++) {
            if (i != axis) {                
                AssertInFastLLM(input0.dims[i] == input1.dims[i], "Cat Error: input's shape doesn't match.");
            }
        }

        std::vector <int> dims = input0.dims;
        dims[axis] += input1.dims[axis];

        output.dataType = input0.dataType;
        output.Resize(dims);
    }

    void CpuCatOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                       const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);

        output.Allocate();

        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        if (input0.dims.size() == 0 && input1.dims.size() > 0) {
            output.CopyFrom(input1);
            return;
        }
        if (input1.dims.size() == 0 && input0.dims.size() > 0) {
            output.CopyFrom(input0);
            return;
        }

        int dimsLen = input0.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        int outer = output.Count(0) / output.Count(axis);
        int input0Stride = input0.Count(axis);
        int input1Stride = input1.Count(axis);
        int outputStride = output.Count(axis);
        int inner = input0.strides[axis];
        int unitSize = input0.unitSize;

        for (int o = 0; o < outer; o++) {
            memcpy(output.cpuData + o * outputStride * unitSize,
                   input0.cpuData + (o * input0Stride) * unitSize,
                   input0.dims[axis] * inner * unitSize);
            memcpy(output.cpuData + o * outputStride * unitSize + input0.dims[axis] * inner * unitSize,
                   input1.cpuData + (o * input1Stride) * unitSize,
                   input1.dims[axis] * inner * unitSize);
        }
    }

    void CpuCatDirectOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                             const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);

        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;

        AssertInFastLLM((input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) ||
                        (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16),
                        "CatDirect's input's type should be float32 or float16.\n");
        AssertInFastLLM(input0.dataDevice == input1.dataDevice, "CatDirect error: inputs should use same device.\n");

        if (input0.dims.size() == 0) {
            input0.Resize(input1.dims);
            AssertInFastLLM(input0.expansionDims.size() == input1.dims.size() &&
                            input1.dims[axis] <= input0.expansionDims[axis],
                            "CatDirect Error: input0's expansion size is not enough.\n");
            int outer = input1.Count(0) / input1.Count(axis);
            int input0Stride = input0.Count(axis);
            int input1Stride = input1.Count(axis);
            int inner = input0.strides[axis];
            int unitSize = input0.unitSize;
            for (int o = 0; o < outer; o++) {
                memcpy(input0.cpuData + o * input0Stride * unitSize,
                       input1.cpuData + o * input1Stride * unitSize,
                       input1.dims[axis] * inner * unitSize);
            }

            return;
        }

        std::vector <int> dims = input0.dims;
        std::vector <int> oldDims = dims;
        dims[axis] += input1.dims[axis];
        input0.Resize(dims);
        int outer = input0.Count(0) / input0.Count(axis);
        int input0Stride = input0.Count(axis);
        int input1Stride = input1.Count(axis);

        int inner = input0.strides[axis];
        int unitSize = input0.unitSize;

        for (int o = 0; o < outer; o++) {
            memcpy(input0.cpuData + o * input0Stride * unitSize + oldDims[axis] * inner * unitSize,
                   input1.cpuData + (o * input1Stride) * unitSize,
                   input1.dims[axis] * inner * unitSize);
        }
    }

    struct MultiThreadMatMulSingleOp : MultiThreadBaseOp {
        float *input0Base, *input1Base, *outputBase;
        int input0Spatial, input1Spatial, outputSpatial;
        int input0Stride, input1Stride, n, m, k;
        float alpha;
        int st, end;

        MultiThreadMatMulSingleOp(float *input0Base, float *input1Base, float *outputBase,
                      int input0Spatial, int input1Spatial, int outputSpatial,
                      int input0Stride, int input1Stride,
                      int n, int m, int k, float alpha, int st, int end) :
                      input0Base(input0Base), input1Base(input1Base), outputBase(outputBase),
                      input0Spatial(input0Spatial), input1Spatial(input1Spatial), outputSpatial(outputSpatial),
                      input0Stride(input0Stride), input1Stride(input1Stride), 
                      n(n), m(m), k(k), alpha(alpha), st(st), end(end) {}
        
        void Run() {
            for (int b = st; b < end; b++) {
                float *input0Data = input0Base + b * input0Spatial;
                float *input1Data = input1Base + b * input1Spatial;
                float *outputData = outputBase + b * outputSpatial;
                std::fill(outputData, outputData + n * k, 0.0f);
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < m; j++) {
                        float now = input0Data[i * input0Stride + j] * alpha;
                        for (int l = 0; l < k; l++) {
                            outputData[i * k + l] += (now * input1Data[j * k + l]);
                        }
                    }
                }
            }
        }
    };

    struct MultiThreadMatMulFloat16SingleOp : MultiThreadBaseOp {
        uint16_t *input0Base, *input1Base, *outputBase;
        int input0Spatial, input1Spatial, outputSpatial;
        int input0Stride, input1Stride, n, m, k;
        float alpha;
        int st, end;

        MultiThreadMatMulFloat16SingleOp(uint16_t *input0Base, uint16_t *input1Base, uint16_t *outputBase,
                             int input0Spatial, int input1Spatial, int outputSpatial,
                             int input0Stride, int input1Stride,
                             int n, int m, int k, float alpha, int st, int end) :
                      input0Base(input0Base), input1Base(input1Base), outputBase(outputBase),
                      input0Spatial(input0Spatial), input1Spatial(input1Spatial), outputSpatial(outputSpatial),
                      input0Stride(input0Stride), input1Stride(input1Stride), 
                      n(n), m(m), k(k), alpha(alpha), st(st), end(end) {}

        void Run() {
            float *input0 = new float[n * m];
            float *input1 = new float[m * k];
            float *output = new float[n * k];

            for (int b = st; b < end; b++) {
                uint16_t *input0Data = input0Base + b * input0Spatial;
                uint16_t *input1Data = input1Base + b * input1Spatial;
                uint16_t *outputData = outputBase + b * outputSpatial;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < m; j++) {
                        input0[i * m + j] = fp16tofp32.dict[input0Data[i * input0Stride + j]];
                    }
                }
                for (int j = 0; j < m; j++) {
                    for (int l = 0; l < k; l++) {
                        input1[j * k + l] = fp16tofp32.dict[input1Data[j * k + l]];
                    }
                }
                std::fill(output, output + n * k, 0.0f);
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < m; j++) {
                        float now = input0[i * m + j] * alpha;
                        for (int l = 0; l < k; l++) {
                            output[i * k + l] += (now * input1[j * k + l]);
                        }
                    }
                }
                for (int i = 0; i < n * k; i++) {
                    outputData[i] = float_to_half(output[i]);
                }
            }

            delete[] input0;
            delete[] input1;
            delete[] output;
        }
    };

    struct MultiThreadMatMulTransBSingleOp : MultiThreadBaseOp {
        float *input0Base, *input1Base, *outputBase;
        int input0Spatial, input1Spatial, outputSpatial;
        int input0Stride, input1Stride, n, m, k;
        float alpha;
        int st, end;

        MultiThreadMatMulTransBSingleOp(float *input0Base, float *input1Base, float *outputBase,
                      int input0Spatial, int input1Spatial, int outputSpatial,
                      int input0Stride, int input1Stride,
                      int n, int m, int k, float alpha, int st, int end) :
                      input0Base(input0Base), input1Base(input1Base), outputBase(outputBase),
                      input0Spatial(input0Spatial), input1Spatial(input1Spatial), outputSpatial(outputSpatial),
                      input0Stride(input0Stride), input1Stride(input1Stride), 
                      n(n), m(m), k(k), alpha(alpha), st(st), end(end) {}
        
        void Run() {
            for (int b = st; b < end; b++) {
                float *input0Data = input0Base + b * input0Spatial;
                float *input1Data = input1Base + b * input1Spatial;
                float *outputData = outputBase + b * outputSpatial;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < k; j++) {
                        float now = 0.0f;
                        int l = 0;
#ifdef __aarch64__
                        float32x4_t sum = {0, 0, 0, 0};
                        for (; l + 3 < m; l += 4) {
                            sum = vaddq_f32(sum, vmulq_f32(vld1q_f32(input0Data + i * input0Stride + l),
                                                        vld1q_f32(input1Data + j * input1Stride + l)));
                        }
                        now += sum[0] + sum[1] + sum[2] + sum[3];
#elif defined(__AVX__)
                        __m256 vsum = _mm256_set1_ps(0.0f);
                        for (; l + 7 < m; l += 8) {
                            __m256 vx = _mm256_loadu_ps((const float *) (input0Data + i * input0Stride + l));
                            __m256 vy = _mm256_loadu_ps((const float *) (input1Data + j * input1Stride + l));
                            vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vx, vy));
                        }
                        now += Floatsum(vsum);
#endif
                        for (; l < m; l++) {
                            now += input0Data[i * input0Stride + l] * input1Data[j * input1Stride + l];
                        }
                        outputData[i * k + j] = now * alpha;
                    }
                }
            }
        }
    };

    struct MultiThreadMatMulTransBFloat16SingleOp : MultiThreadBaseOp {
        uint16_t *input0Base, *input1Base, *outputBase;
        int input0Spatial, input1Spatial, outputSpatial;
        int input0Stride, input1Stride, n, m, k;
        float alpha;
        int st, end;

        MultiThreadMatMulTransBFloat16SingleOp(uint16_t *input0Base, uint16_t *input1Base, uint16_t *outputBase,
                      int input0Spatial, int input1Spatial, int outputSpatial,
                      int input0Stride, int input1Stride,
                      int n, int m, int k, float alpha, int st, int end) :
                      input0Base(input0Base), input1Base(input1Base), outputBase(outputBase),
                      input0Spatial(input0Spatial), input1Spatial(input1Spatial), outputSpatial(outputSpatial),
                      input0Stride(input0Stride), input1Stride(input1Stride), 
                      n(n), m(m), k(k), alpha(alpha), st(st), end(end) {}
        void Run() {
            for (int b = st; b < end; b++) {
                uint16_t *input0Data = input0Base + b * input0Spatial;
                uint16_t *input1Data = input1Base + b * input1Spatial;
                uint16_t *outputData = outputBase + b * outputSpatial;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < k; j++) {
                        float now = 0.0f;
                        int l = 0;
                        for (; l < m; l++) {
                            now += fp16tofp32.dict[input0Data[i * input0Stride + l]] *
                                    fp16tofp32.dict[input1Data[j * input1Stride + l]];
                        }
                        outputData[i * k + j] = float_to_half(now * alpha);
                    }
                }
            }
        }
    };

    void CpuMatMulOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);

        AssertInFastLLM(input0.dataDevice == input1.dataDevice, "MatMul error: inputs should use same device.\n");
        AssertInFastLLM((input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) ||
                        (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16),
                        "MatMul's input's type should be float32 or float16.\n");
        AssertInFastLLM(input0.dims.size() >= 2 && input1.dims.size() >= 2,
                        "MatMul's input's shape's size should be >= 2.\n");
        AssertInFastLLM(input0.dims.back() == input1.dims[input1.dims.size() - 2],
                        "MatMul's shape error.\n");
        int input0Spatial = input0.Count(input0.dims.size() - 2);
        int input1Spatial = input1.Count(input1.dims.size() - 2);
        int batch0 = input0.Count(0) / input0Spatial;
        int batch1 = input1.Count(0) / input1Spatial;
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : 1;
        AssertInFastLLM(batch0 == batch1 * group, "MatMul: input0.dims[1] should be equal to input1.dims[0] * group.\n");
        // AssertInFastLLM(batch0 == batch1, "MatMul's shape error.\n");

        std::vector <int> dims = input0.dims;
        dims.back() = input1.dims[input1.dims.size() - 1];

        output.dataType = input0.dataType;
        output.Resize(dims);
    }

    void CpuMatMulOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);

        output.Allocate();

        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0f;
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : 1;
        int input0Spatial = input0.Count(input0.dims.size() - 2) * group;
        int input1Spatial = input1.Count(input1.dims.size() - 2);
        int input0Stride = input0.strides[input0.dims.size() - 2];
        int input1Stride = input1.strides[input1.dims.size() - 2];
        int n = input0.dims[input0.dims.size() - 2] * group;
        int m = input0.dims.back();
        int k = input1.dims[input1.dims.size() - 1];
        int batch0 = input0.Count(0) / input0Spatial;
        int batch1 = input1.Count(0) / input1Spatial;

        int outputSpatial = output.Count(output.dims.size() - 2) * group;
        int threadNum = GetThreads();
#ifdef _WIN64
        threadNum = 1;
#endif
        if (batch0 * n * m * k < 64 * 4096) {
            threadNum = 1;
        }
        threadNum = std::min(threadNum, 4);
        // TODO: 汇编优化
        int per = batch0 / threadNum;
        int cur = 0;
        if (input0.dataType == DataType::FLOAT32) {
            auto *pool = GetAlivePool();
            int threads = pool->threads.size();
            std::vector<fastllm::MultiThreadMatMulSingleOp*> ops;
            for (int o = 0; o < batch0; o++) {
                ops.push_back(new MultiThreadMatMulSingleOp(
                    (float *) input0.cpuData, (float *) input1.cpuData, (float *) output.cpuData,
                    input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                    n, m, k, alpha, o, o + 1
                ));
            }
            for (int st = 0; st < ops.size(); st += threads) {
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->PushOp(i - st, ops[i]);
                }
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->Wait(i - st);
                }
            }
        } else if (input0.dataType == DataType::FLOAT16) {
            auto *pool = GetAlivePool();
            int threads = pool->threads.size();
            std::vector<fastllm::MultiThreadMatMulFloat16SingleOp*> ops;
            for (int o = 0; o < batch0; o++) {
                ops.push_back(new MultiThreadMatMulFloat16SingleOp(
                    (uint16_t *) input0.cpuData, (uint16_t *) input1.cpuData, (uint16_t *) output.cpuData,
                    input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                    n, m, k, alpha, o, o + 1
                ));
            }
            for (int st = 0; st < ops.size(); st += threads) {
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->PushOp(i - st, ops[i]);
                }
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->Wait(i - st);
                }
            }
        }
    }

    void CpuMatMulTransBOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);

        AssertInFastLLM(input0.dataDevice == input1.dataDevice, "MatMulTransB error: inputs should use same device.\n");
        AssertInFastLLM((input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) ||
                        (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16),
                        "MatMulTransB's input's type should be float32 or float16.\n");
        AssertInFastLLM(input0.dims.size() >= 2 && input1.dims.size() >= 2,
                        "MatMulTransB's input's shape's size should be >= 2.\n");
        AssertInFastLLM(input0.dims.back() == input1.dims.back(),
                        "MatMulTransB's shape error.\n");
        int input0Spatial = input0.Count(input0.dims.size() - 2);
        int input1Spatial = input1.Count(input1.dims.size() - 2);
        int batch0 = input0.Count(0) / input0Spatial;
        int batch1 = input1.Count(0) / input1Spatial;
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : 1;
        AssertInFastLLM(batch0 == batch1 * group, "MatMulTransB: input0.dims[0] should be equal to input1.dims[0] * group.\n");
        // AssertInFastLLM(batch0 == batch1, "MatMulTransB's shape error.\n");

        std::vector <int> dims = input0.dims;
        dims.back() = input1.dims[input1.dims.size() - 2];
        output.dataType = input0.dataType;
        output.Resize(dims);
    }

    void CpuMatMulTransBOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        Data &output = *(datas.find("output")->second);

        output.Allocate();

        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0;
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : 1;
        int input0Spatial = input0.Count(input0.dims.size() - 2) * group;
        int input1Spatial = input1.Count(input1.dims.size() - 2);
        int input0Stride = input0.strides[input0.dims.size() - 2];
        int input1Stride = input1.strides[input1.dims.size() - 2];
        int n = input0.dims[input0.dims.size() - 2] * group;
        int m = input0.dims.back();
        int k = input1.dims[input1.dims.size() - 2];
        int batch0 = input0.Count(0) / input0Spatial;
        int batch1 = input1.Count(0) / input1Spatial;

        int outputSpatial = output.Count(output.dims.size() - 2) * group;
        int threadNum = GetThreads();
#ifdef _WIN64
        threadNum = 1;
#endif
        if (batch0 * n * m * k < 64 * 4096) {
            threadNum = 1;
        }
        threadNum = std::min(threadNum, 4);
        int per = batch0 / threadNum;
        int cur = 0;
        if (input0.dataType == DataType::FLOAT32) {
            auto *pool = GetAlivePool();
            int threads = pool->threads.size();
            std::vector<fastllm::MultiThreadMatMulTransBSingleOp*> ops;
            for (int o = 0; o < batch0; o++) {
                ops.push_back(new MultiThreadMatMulTransBSingleOp(
                    (float *) input0.cpuData, (float *) input1.cpuData, (float *) output.cpuData,
                    input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                    n, m, k, alpha, o, o + 1
                ));
            }
            for (int st = 0; st < ops.size(); st += threads) {
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->PushOp(i - st, ops[i]);
                }
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->Wait(i - st);
                }
            }
        } else {
            auto *pool = GetAlivePool();
            int threads = pool->threads.size();
            std::vector<fastllm::MultiThreadMatMulTransBFloat16SingleOp*> ops;
            for (int o = 0; o < batch0; o++) {
                ops.push_back(new MultiThreadMatMulTransBFloat16SingleOp(
                    (uint16_t *) input0.cpuData, (uint16_t *) input1.cpuData, (uint16_t *) output.cpuData,
                    input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                    n, m, k, alpha, o, o + 1
                ));
            }
            for (int st = 0; st < ops.size(); st += threads) {
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->PushOp(i - st, ops[i]);
                }
                for (int i = st; i < ops.size() && i < st + threads; i++) {
                    pool->Wait(i - st);
                }
            }
        }
    }

    void CpuSoftMaxOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;

        AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                        "Softmax error: Data's type should be float32.\n");

        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int outer = input.Count(0) / input.Count(axis);
        int channels = input.dims[axis];
        int inner = input.Count(axis + 1);

        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;

        if (input.dataType == DataType::FLOAT16) {
            int len = input.Count(0);
            inputData = new float[len];
            outputData = new float[len];
            for (int i = 0; i < len; i++) {
                inputData[i] = fp16tofp32.dict[((uint16_t *) input.cpuData)[i]];
            }
        }

        if (inner == 1) {
            for (int i = 0; i < outer; i++) {
                float maxValue = 0;
                int j = 0;
#ifdef __aarch64__
                float32x4_t vmax = vdupq_n_f32(-1e9);
                for (; j + 3 < channels; j += 4) {
                    vmax = vmaxq_f32(vmax, vld1q_f32(inputData + j));
                }
                for (int k = 0; k < 4; k++) {
                    maxValue = std::max(maxValue, vmax[k]);
                }
#endif
                for (; j < channels; j++) {
                    maxValue = std::max(maxValue, inputData[j]);
                }

                j = 0;
#ifdef __aarch64__
                vmax = vdupq_n_f32(maxValue);
                for (; j + 3 < channels; j += 4) {
                    vst1q_f32(outputData + j, exp_ps(vsubq_f32(vld1q_f32(inputData + j), vmax)));
                }
#endif
                for (; j < channels; j++) {
                    outputData[j] = exp(inputData[j] - maxValue);
                }
                float sum = 0.0;
                j = 0;
                for (; j < channels; j++) {
                    sum += outputData[j];
                }
                if (fabs(sum) < 1e-9) {
                    sum = 0.1;
                }
                j = 0;
#ifdef __aarch64__
                float32x4_t fsum = vdupq_n_f32(sum);
                for (j = 0; j + 3 < channels; j += 4) {
                    vst1q_f32(outputData + j, vdivq_f32(vld1q_f32(outputData + j), fsum));
                }
#endif
                for (; j < channels; j++) {
                    outputData[j] = outputData[j] / sum;
                }
                inputData += channels;
                outputData += channels;
            }
        } else {
            for (int i = 0; i < outer; i++) {
                std::vector<float> maxValue(inner, -FLT_MAX);
                for (int j = 0; j < channels; j++) {
                    for (int k = 0; k < inner; k++) {
                        maxValue[k] = std::max(maxValue[k], inputData[j * inner + k]);
                    }
                }
                std::vector<float> sum(inner, 0.0);
                for (int j = 0; j < channels; j++) {
                    for (int k = 0; k < inner; k++) {
                        outputData[j * inner + k] = std::exp(inputData[j * inner + k] - maxValue[k]);
                        sum[k] += outputData[j * inner + k];
                    }
                }

                for (int j = 0; j < channels; j++) {
                    for (int k = 0; k < inner; k++) {
                        outputData[j * inner + k] /= sum[k];
                    }
                }

                inputData += channels * inner;
                outputData += channels * inner;
            }
        }

        if (input.dataType == DataType::FLOAT16) {
            int len = input.Count(0);
            inputData -= len;
            outputData -= len;
            for (int i = 0; i < len; i++) {
                ((uint16_t *) output.cpuData)[i] = float_to_half(outputData[i]);
            }

            delete[] inputData;
            delete[] outputData;
        }
    }

    void CpuSiluOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32, "Silu error: Data's type should be float32.\n");
        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;
        int len = input.Count(0);
        int i = 0;
#ifdef __aarch64__
        float32x4_t c1 = vdupq_n_f32(1.0f);
        for (; i + 3 < len; i += 4) {
            float32x4_t vx = vld1q_f32(inputData + i);
            float32x4_t vdiv = vaddq_f32(c1, exp_ps(vnegq_f32(vx)));
            vx = vdivq_f32(vx, vdiv);
            vst1q_f32(outputData + i, vx);
        }
#endif
        for (; i < len; i++) {
            float x = inputData[i];
            outputData[i] = x / (1.0 + expf(-x));
        }
    }

    void CpuTanHOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);                    
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32, "GeluNew error: Data's type should be float32.\n");

        float temp = sqrt(2.0f / M_PI), factor = 0.044715;
        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;
        int len = input.Count(0);
        int i = 0;
        for (; i < len; i++) {
            outputData[i] = tanhf(inputData[i]);
        }
    }

    float erf(float a)
    {
        float r, s, t, u;

        t = fabsf(a);
        s = a * a;
        if (t > 0.927734375f)
        {   // 475/512
            // maximum error 0.99527 ulp
            r = fmaf(-1.72853470e-5f, t, 3.83197126e-4f); // -0x1.220000p-16,0x1.91cfb2p-12
            u = fmaf(-3.88396438e-3f, t, 2.42546219e-2f); // -0x1.fd1438p-9, 0x1.8d6342p-6
            r = fmaf(r, s, u);
            r = fmaf(r, t, -1.06777877e-1f); // -0x1.b55cb8p-4
            r = fmaf(r, t, -6.34846687e-1f); // -0x1.450aa0p-1
            r = fmaf(r, t, -1.28717512e-1f); // -0x1.079d0cp-3
            r = fmaf(r, t, -t);
            r = 1.0f - expf(r);
            r = copysignf(r, a);
        }
        else
        {
            // maximum error 0.98929 ulp
            r = -5.96761703e-4f;             // -0x1.38e000p-11
            r = fmaf(r, s, 4.99119423e-3f);  //  0x1.471a58p-8
            r = fmaf(r, s, -2.67681349e-2f); // -0x1.b691b2p-6
            r = fmaf(r, s, 1.12819925e-1f);  //  0x1.ce1c44p-4
            r = fmaf(r, s, -3.76125336e-1f); // -0x1.812700p-2
            r = fmaf(r, s, 1.28379166e-1f);  //  0x1.06eba8p-3
            r = fmaf(r, a, a);
        }
        return r;
    }

    void CpuGeluOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32, "GeluNew error: Data's type should be float32.\n");

        float temp = sqrt(2.0f / M_PI), factor = 0.044715;
        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;
        int len = input.Count(0);
        int i = 0;
        for (; i < len; i++) {
            float x = inputData[i];
            outputData[i] = x * 0.5f * (1.0f + erf(x / sqrt(2.0)));
        }
    }

    void CpuGeluNewOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32, "GeluNew error: Data's type should be float32.\n");

        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;
        int len = input.Count(0);
        int i = 0;
#ifdef __aarch64__
        float32x4_t c0 = vdupq_n_f32(0.044715f);
        float32x4_t c1 = vdupq_n_f32(1.0f);
        float32x4_t c2 = vdupq_n_f32(0.7978845608028654f);
        float32x4_t c3 = vdupq_n_f32(0.5f);

        for (; i + 3 < len; i += 4) {
            float32x4_t vx = vld1q_f32(inputData + i);
            float32x4_t v1 = vaddq_f32(c1, vmulq_f32(vmulq_f32(c0, vx), vx));
            float32x4_t v2 = vmulq_f32(vmulq_f32(c2, vx), v1);
            float32x4_t vex = exp_ps(v2);
            float32x4_t venegx = exp_ps(vnegq_f32(v2));
            float32x4_t vtan = vdivq_f32(vsubq_f32(vex, venegx), vaddq_f32(vex, venegx));
            float32x4_t vout = vmulq_f32(vmulq_f32(c3, vx), vaddq_f32(c1, vtan));
            vst1q_f32(outputData + i, vout);
        }
#endif
#ifdef __AVX2__
        auto var1 = _mm256_set1_ps(0.044715f);
        auto var2 = _mm256_set1_ps(0.7978845608028654f);
        auto var3 = _mm256_set1_ps(378.f);
        auto var4 = _mm256_set1_ps(17325.f);
        auto var5 = _mm256_set1_ps(135135.f);
        auto var6 = _mm256_set1_ps(28.f);
        auto var7 = _mm256_set1_ps(3150.f);
        auto var8 = _mm256_set1_ps(62370.f);
        auto var9 = _mm256_set1_ps(135135.f);
        auto var10 = _mm256_set1_ps(0.5);
        auto varOne = _mm256_set1_ps(1.f);
        auto varNegOne = _mm256_set1_ps(-1.f);

        for (; i < len - 7; i+=8) {
            auto x = _mm256_loadu_ps(inputData + i);  
            // sqrt(2 / PI) * (0.044715 * x^3 + x)
            auto y = _mm256_mul_ps(x, x);
            y = _mm256_mul_ps(y, x);
            y = _mm256_mul_ps(y, var1);
            y = _mm256_add_ps(y, x);
            y = _mm256_mul_ps(y, var2);

            // y = tanh(y)
            {
            auto y2 = _mm256_mul_ps(y, y);
            auto w = _mm256_add_ps(y2, var3);
            w = _mm256_mul_ps(w, y2);
            w = _mm256_add_ps(w, var4);
            w = _mm256_mul_ps(w, y2);
            w = _mm256_add_ps(w, var5);
            w = _mm256_mul_ps(w, y);
            auto z = _mm256_mul_ps(y2, var6);
            z = _mm256_add_ps(z, var7);
            z = _mm256_mul_ps(z, y2);
            z = _mm256_add_ps(z, var8);
            z = _mm256_mul_ps(z, y2);
            z = _mm256_add_ps(z, var9);
            z = _mm256_div_ps(w, z);
            z = _mm256_max_ps(z, varNegOne);
            y = _mm256_min_ps(z, varOne);
            }

            y = _mm256_add_ps(y, varOne);
            y = _mm256_mul_ps(y, x);
            y = _mm256_mul_ps(y, var10);
            _mm256_storeu_ps(outputData + i, y);
        }
#endif
        for (; i < len; i++) {
            float x = inputData[i];
            outputData[i] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
        }
    }

    void CpuSwigluOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);

        std::vector <int> dims = input.dims;
        dims[dims.size() - 1] /= 2;
        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void CpuSwigluOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                        "Swiglu error: Data's type should be float32 or float16.\n");

        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;

        if (input.dataType == DataType::FLOAT16) {
            int len = input.Count(0);
            inputData = new float[len];
            outputData = new float[output.Count(0)];
            for (int i = 0; i < len; i++) {
                inputData[i] = fp16tofp32.dict[((uint16_t *) input.cpuData)[i]];
            }
        }

        int spatial = input.Count(input.dims.size() - 1), mid = spatial / 2;
        int outer = input.Count(0) / spatial;
        for (int o = 0; o < outer; o++) {
            int i = 0;
#ifdef __aarch64__
            float32x4_t c1 = vdupq_n_f32(1.0f);
            for (; i + 3 < mid; i += 4) {
                float32x4_t vx = vld1q_f32(inputData + i);
                float32x4_t vy = vld1q_f32(inputData + i + mid);
                vx = vdivq_f32(vx, vaddq_f32(c1, exp_ps(vnegq_f32(vx))));
                vy = vmulq_f32(vx, vy);
                vst1q_f32(outputData + i, vy);
            }
#endif
            for (; i < mid; i++) {
                float x = inputData[i], y = inputData[i + mid];
                outputData[i] = (x / (1.0 + expf(-x))) * y;
            }
            inputData += spatial;
            outputData += spatial / 2;
        }

        if (input.dataType == DataType::FLOAT16) {
            inputData -= input.Count(0);
            outputData -= output.Count(0);
            int len = output.Count(0);
            for (int i = 0; i < len; i++) {
                ((uint16_t *) output.cpuData)[i] = float_to_half(outputData[i]);
            }

            delete[] inputData;
            delete[] outputData;
        }
    }

    void CpuMulOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                       const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();

        float v = floatParams.find("v") != floatParams.end() ? floatParams.find("v")->second : 1.0;
        AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                        "Mul error: Data's type should be float32 or float16.\n");

        int len = input.Count(0);

        if (input.dataType == DataType::FLOAT32) {
            float *inputData = (float *) input.cpuData;
            float *outputData = (float *) output.cpuData;
            for (int i = 0; i < len; i++) {
                outputData[i] = inputData[i] * v;
            }
        } else if (input.dataType == DataType::FLOAT16) {
            uint16_t *inputData = (uint16_t *) input.cpuData;
            uint16_t *outputData = (uint16_t *) output.cpuData;
            for (int i = 0; i < len; i++) {
                outputData[i] = float_to_half(fp16tofp32.dict[inputData[i]] * v);
            }
        }
    }

    void CpuMulToOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        AssertInFastLLM(input0.dims == input1.dims, "MulTo error: input's shape should be same.\n");

        float *input0Data = (float*)input0.cpuData;
        float *input1Data = (float*)input1.cpuData;

        int len = input0.Count(0);
        int inner = input1.Count(0);
        AssertInFastLLM(len % inner == 0, "MulTo error: Data`s shape can`t perform MulTo operation.\n");
        int round = (len / inner);
        for (int j = 0; j < round; j++) {
            for (int i = 0; i < len; i++) {
               input0Data[i] *= input1Data[i];
            }
            input0Data += inner;
        }
    }

    struct MultiThreadAddToFloatOp : MultiThreadBaseOp {
        float *input, *output;
        int len;
        float alpha;

        MultiThreadAddToFloatOp (float *output, float *input, float alpha, int len) : input(input), output(output), alpha(alpha), len(len) {}

        void Run() {
            for (int i = 0; i < len; i++) {
                output[i] += input[i] * alpha;
            }
        }
    };

    static void RunMultiThreadAddToFloat(float *output, float *input, float alpha, int len, AliveThreadPool *pool) {
        if (len < 256 * 1024) {
            (MultiThreadAddToFloatOp(output, input, alpha, len)).Run();
            return;
        }
        int threadNum = pool->threads.size();
        int per = len / pool->threads.size();
        int cur = 0;
        std::vector<fastllm::MultiThreadAddToFloatOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? len : cur + per + (cur + per * (threadNum - i) < len));
            ops.push_back(new MultiThreadAddToFloatOp(output + cur, input + cur, alpha, end - cur));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void CpuAddToOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                         const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        float alpha = floatParams.find("alpha") != floatParams.end() ? floatParams.find("alpha")->second : 1.0;

        AssertInFastLLM(input0.dataType == DataType::FLOAT32 || input1.dataType == DataType::FLOAT16,
                        "AddTo error: Data's type should be float32 or float16.\n");
        AssertInFastLLM(input0.dims == input1.dims, "AddTo error: input's shape should be same.\n");

        int len = input0.Count(0);

        if (input0.dataType == DataType::FLOAT32) {
            float *input0Data = (float *) input0.cpuData;
            float *input1Data = (float *) input1.cpuData;
            RunMultiThreadAddToFloat(input0Data, input1Data, alpha, len, GetAlivePool());
        } else if (input0.dataType == DataType::FLOAT16) {
            uint16_t *input0Data = (uint16_t *) input0.cpuData;
            uint16_t *input1Data = (uint16_t *) input1.cpuData;
            for (int i = 0; i < len; i++) {
                input0Data[i] = float_to_half(fp16tofp32.dict[input0Data[i]] + fp16tofp32.dict[input1Data[i]] * alpha);
            }
        }
    }

    void CpuAttentionMaskOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &mask = *(datas.find("mask")->second);
        float maskValue = floatParams.find("maskValue") != floatParams.end() ? floatParams.find("maskValue")->second : -10000.0;
        int spatial = input.Count(2), n = input.dims[0], m = input.dims[1];

        AssertInFastLLM(mask.dataType == DataType::FLOAT32, "AttentionMask: mask's datatype should be float32.");
        if (input.dataType == DataType::FLOAT32) {
            float *maskData = (float *) mask.cpuData;
            float *attnData = (float *) input.cpuData;
            for (int on = 0; on < n; on++) {
                for (int om = 0; om < m; om++) {
                    int o = on * m + om;
                    for (int i = 0; i < spatial; i++) {
                        if (maskData[on * spatial + i] > 0.99) {
                            attnData[o * spatial + i] = maskValue;
                        }
                    }
                }
            }
        } else if (input.dataType == DataType::FLOAT16) {
            float *maskData = (float *) mask.cpuData;
            uint16_t *attnData = (uint16_t *) input.cpuData;
            uint16_t hMaskValue = float_to_half(maskValue);
            for (int on = 0; on < n; on++) {
                for (int om = 0; om < m; om++) {
                    int o = on * m + om;
                    for (int i = 0; i < spatial; i++) {
                        if (maskData[on * spatial + i] > 0.99) {
                            attnData[o * spatial + i] = hMaskValue;
                        }
                    }
                }
            }
        } else {
            ErrorInFastLLM("AttentionMask error: unsupport input's dataType.\n");
        }
    }

    void CpuAttentionExtendedMaskOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &mask = *(datas.find("mask")->second);
        int spatial = input.dims[3], n = input.dims[0], m = input.dims[1] * input.dims[2];

        AssertInFastLLM(mask.dataType == DataType::FLOAT32, "AttentionExtendedMask: mask's datatype should be float32.");
        if (input.dataType == DataType::FLOAT32) {
            float *maskData = (float *) mask.cpuData;
            float *attnData = (float *) input.cpuData;
            for (int on = 0; on < n; on++) {
                for (int om = 0; om < m; om++) {
                    int o = on * m + om;
                    for (int i = 0; i < spatial; i++) {
                        attnData[o * spatial + i] += maskData[on * spatial + i];
                    }
                }
            }
        } else {
            ErrorInFastLLM("AttentionExtendedMask error: unsupport input's dataType.\n");
        }
    }

    void CpuAlibiMaskOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                             const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &mask = *(datas.find("mask")->second);
        float maskValue = floatParams.find("maskValue") != floatParams.end() ? floatParams.find("maskValue")->second : -10000.0;
        float *maskData = (float *) mask.cpuData;
        float *attnData = (float *) input.cpuData;
        int n = input.dims[0], m = input.dims[1];
        int spn = input.dims[2], spm = input.dims[3];
        int spatial = input.Count(2);
        for (int on = 0; on < n; on++) {
            for (int om = 0; om < m; om++) {
                float now = maskData[om];
                int o = on * m + om;
                float *inputNow = attnData + o * spatial;
                for (int i = 0; i < spn; i++) {
                    int mid = (spm - spn + i);
                    for (int j = 0; j <= mid; j++) {
                        inputNow[i * spm + j] += now * j;
                    }
                    for (int j = mid + 1; j < spm; j++) {
                        inputNow[i * spm + j] = maskValue;
                    }
                }
            }
        }
    }

    void CpuTopKOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                            const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        int topk = intParams.find("topk") != intParams.end() ? intParams.find("topk")->second : 1;

        AssertInFastLLM(input.dataType == DataType::FLOAT32, "TopK error: Data's type should be float32.\n");

        int dimsLen = input.dims.size();
        std::vector<int> dims = input.dims;
        dims[dimsLen - 1] = topk * 2;

        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void CpuTopKOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();

        int topk = intParams.find("topk") != intParams.end() ? intParams.find("topk")->second : -1;
        int dimsLen = input.dims.size();
        int outer = input.Count(0) / input.Count(dimsLen - 1);
        int channels = input.dims[dimsLen - 1];

        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;

        if (topk == 1) {
            for (int o = 0; o < outer; o++) {
                float maxValue = -1e100, idx = -1;
                for (int j = 0; j < channels; j++) {
                    if (inputData[j] > maxValue) {
                        maxValue = inputData[j];
                        idx = j;
                    }
                }
                outputData[0] = idx;
                outputData[1] = maxValue;
                inputData += channels;
                outputData += 2;
            }
        } else {
            for (int o = 0; o < outer; o++) {
                std::vector <std::pair <float, int> > v;
                for (int j = 0; j < channels; j++) {
                    v.push_back(std::make_pair(-inputData[j], j));
                }
                sort(v.begin(), v.end());
                for (int j = 0; j < topk; j++) {
                    outputData[j * 2] = v[j].second;
                    outputData[j * 2 + 1] = -v[j].first;
                }

                inputData += channels;
                outputData += 2 * topk;
            }
        }
    }

    void CpuPermuteOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &axisData = *(datas.find("axis")->second);
        std::vector <int> axis;
        for (int i = 0; i < axisData.Count(0); i++) {
            axis.push_back(((int32_t *) axisData.cpuData)[i]);
        }

        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16, "Permute error: datatype should be float32 or float16.");
        AssertInFastLLM(axis.size() == input.dims.size(), "Permute error: axis's size should be equal to data's shape's size.");
        std::vector<int> new_dims;
        for (int i = 0; i < axis.size(); i++) {
            new_dims.push_back(input.dims[axis[i]]);
        }

        output.dataType = input.dataType;
        output.Resize(new_dims);
    }

    void Transpose4x4(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m) {
        if (n < 4 || m < 4) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    pDst[j * dstStride + i] = pSrc[i * srcStride + j];
                }
            }

            return;
        }

#ifdef __aarch64__
        float32x4x2_t q01 = vtrnq_f32(vld1q_f32(pSrc), vld1q_f32(pSrc + srcStride));
        float32x4x2_t q23 = vtrnq_f32(vld1q_f32(pSrc + 2 * srcStride), vld1q_f32(pSrc + 3 * srcStride));

        float32x4_t qq0 = q01.val[0];
        float32x2_t d00 = vget_low_f32(qq0);
        float32x2_t d01 = vget_high_f32(qq0);

        float32x4_t qq1 = q01.val[1];
        float32x2_t d10 = vget_low_f32(qq1);
        float32x2_t d11 = vget_high_f32(qq1);

        float32x4_t qq2 = q23.val[0];
        float32x2_t d20 = vget_low_f32(qq2);
        float32x2_t d21 = vget_high_f32(qq2);

        float32x4_t qq3 = q23.val[1];
        float32x2_t d30 = vget_low_f32(qq3);
        float32x2_t d31 = vget_high_f32(qq3);

        vst1q_f32(pDst, vcombine_f32(d00, d20));
        vst1q_f32(pDst + 1 * dstStride, vcombine_f32(d10, d30));
        vst1q_f32(pDst + 2 * dstStride, vcombine_f32(d01, d21));
        vst1q_f32(pDst + 3 * dstStride, vcombine_f32(d11, d31));
#else
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                pDst[j * dstStride + i] = pSrc[i * srcStride + j];
            }
        }
#endif
    }

    void Transpose(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m) {
        int per = 4;
        for (int i = 0; i < n; i += per) {
            for (int j = 0; j < m; j += per) {
                Transpose4x4(pDst + j * dstStride + i,
                             pSrc + i * srcStride + j,
                             dstStride, srcStride,
                             std::min(per, n - i),
                             std::min(per, m - j));
            }
        }
    }

    struct MultiThreadTransposeOp : MultiThreadBaseOp {
        float *pDst, *pSrc;
        int dstStride, srcStride, n, m;

        MultiThreadTransposeOp(float *pDst, float *pSrc, int dstStride, int srcStride, int n, int m) :
            pDst(pDst), pSrc(pSrc), dstStride(dstStride), srcStride(srcStride), n(n), m(m) {}
        
        void Run() {
            Transpose(pDst, pSrc, dstStride, srcStride, n, m);
        }
    };

    void CpuPermuteOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &axisData = *(datas.find("axis")->second);
        std::vector <int> axis;
        for (int i = 0; i < axisData.Count(0); i++) {
            axis.push_back(((int32_t *) axisData.cpuData)[i]);
        }

        output.Allocate();
        uint8_t *tmpData = (uint8_t *) output.cpuData;
        uint8_t *curData = (uint8_t *) input.cpuData;

        if (axis == std::vector <int> {1, 2, 0} && input.dataType == DataType::FLOAT32) {
            int n = input.dims[0];
            int m = input.Count(1);

            int threadNum = 1;
            int per = m / threadNum;
            int cur = 0;
            Transpose(((float*)tmpData) + cur * n, ((float*)curData) + cur, n, m, n, m - cur);
        } else if (axis == std::vector <int> {1, 0, 2}) {
            int n = input.dims[0];
            int m = input.dims[1];
            int k = input.dims[2];
            int unitSize = input.unitSize;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    memcpy(tmpData + (j * n + i) * k * unitSize, curData + (i * m + j) * k * unitSize, k * unitSize);
                }
            }
        } else if (axis == std::vector <int> {2, 0, 1, 3}) {
            int n = input.dims[0] * input.dims[1];
            int m = input.dims[2];
            int k = input.dims[3];
            int unitSize = input.unitSize;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    memcpy(tmpData + (j * n + i) * k * unitSize, curData + (i * m + j) * k * unitSize, k * unitSize);
                }
            }
        } else if (axis == std::vector<int> {0, 2, 1, 3}) {
            int b = input.dims[0];
            int n = input.dims[1];
            int m = input.dims[2];
            int k = input.dims[3];
            int unitSize = input.unitSize;
            for (int o = 0; o < b; o++) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < m; j++) {
                        memcpy(tmpData + (j * n + i) * k * unitSize, curData + (i * m + j) * k * unitSize, k * unitSize);
                    }
                }
                tmpData += output.Count(1) * unitSize;
                curData += input.Count(1) * unitSize;
            }
        } else {
            std::vector<int> oldSteps;
            std::vector<int> newSteps;
            int count = input.Count(0);
            auto oldPos = new int[count];
            for (int i = 0; i < axis.size(); i++) {
                oldSteps.push_back(input.Count(i + 1));
                newSteps.push_back(output.Count(i + 1));
            }

            for (int i = 0; i < count; ++i) {
                int old = 0;
                int idx = i;
                for (int j = 0; j < axis.size(); ++j) {
                    int order = axis[j];
                    old += (idx / newSteps[j]) * oldSteps[order];
                    idx %= newSteps[j];
                }
                oldPos[i] = old;
            }

            if (input.unitSize == 4) {
                for (int i = 0; i < count; ++i) {
                    ((float*)tmpData)[i] = ((float*)curData)[oldPos[i]];
                }
            } else if (input.unitSize == 2) {
                for (int i = 0; i < count; ++i) {
                    ((uint16_t*)tmpData)[i] = ((uint16_t*)curData)[oldPos[i]];
                }
            } else if (input.unitSize == 1) {
                for (int i = 0; i < count; ++i) {
                    ((uint8_t*)tmpData)[i] = ((uint8_t*)curData)[oldPos[i]];
                }
            }

            delete[] oldPos;
        }
    }

    void CpuPermuteSelfOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &axisData = *(datas.find("axis")->second);
        std::vector <int> axis;
        for (int i = 0; i < axisData.Count(0); i++) {
            axis.push_back(((int32_t *) axisData.cpuData)[i]);
        }

        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16, "Permute error: datatype should be float32 or float16.");
        AssertInFastLLM(axis.size() == input.dims.size(), "Permute error: axis's size should be equal to data's shape's size.");

        std::vector<int> new_dims;
        for (int i = 0; i < axis.size(); i++) {
            new_dims.push_back(input.dims[axis[i]]);
        }

        bool same = false;
        same |= ((axis == std::vector <int>{1, 2, 0} || axis == std::vector <int>{1, 0, 2}) && (input.dims[0] == 1 || input.dims[1] == 1));
        same |= ((axis == std::vector <int>{2, 0, 1, 3}) && input.dims[2] == 1);
        same |= ((axis == std::vector <int>{0, 2, 1, 3}) && (input.dims[1] == 1 || input.dims[2] == 1));
        same |= ((axis == std::vector <int>{1, 0, 2, 3}) && (input.dims[0] == 1 || input.dims[1] == 1));
        if (same) {
            input.Resize(new_dims);
            return;
        }

        bool swapLastTwoDims = false;
        if (input.dims.size() >= 2 && input.dims.size() == new_dims.size()) {
            std::vector <int> dims = input.dims;
            std::swap(dims[dims.size() - 2], dims[dims.size() - 1]);
            swapLastTwoDims = (dims == new_dims);
        }

        if (swapLastTwoDims && input.dataType == DataType::FLOAT32) {
            int dl = input.dims.size();
            int outer = input.Count(0) / input.Count(dl - 2);
            int n = input.dims[dl - 2], m = input.dims[dl - 1];
            float *temp = new float[n * m];
            float *finput = (float*)input.cpuData;
            for (int i = 0; i < outer; i++) {
                memcpy(temp, finput + i * n * m, n * m * sizeof(float));
                Transpose(finput + i * n * m, temp, n, m, n, m);
            }
            delete[] temp;
            input.Resize(new_dims);
        } else if (axis == std::vector<int> {0, 2, 1, 3}) {
            std::vector <uint8_t> vold;
            vold.resize(input.GetBytes());
            RunMultiThreadMemcpy(vold.data(), input.cpuData, input.GetBytes(), GetAlivePool());
            uint8_t *oldData = vold.data();
            uint8_t *newData = (uint8_t *) input.cpuData;
            int b = input.dims[0];
            int n = input.dims[1];
            int m = input.dims[2];
            int k = input.dims[3];
            int unitSize = input.unitSize;
            for (int o = 0; o < b; o++) {
                RunMultiThreadTransposeByLine(newData, oldData, n, m, k * unitSize, GetAlivePool());
                oldData += input.Count(1) * unitSize;
                newData += input.Count(1) * unitSize;
            }
            input.Resize(new_dims);
        } else if (axis == std::vector <int> {1, 0, 2}) {
            std::vector <uint8_t> vold;
            vold.resize(input.GetBytes());
            RunMultiThreadMemcpy(vold.data(), input.cpuData, input.GetBytes(), GetAlivePool());
            uint8_t *oldData = vold.data();
            uint8_t *newData = (uint8_t *) input.cpuData;
            int n = input.dims[0];
            int m = input.dims[1];
            int k = input.dims[2];
            int unitSize = input.unitSize;
            RunMultiThreadTransposeByLine(newData, oldData, n, m, k * unitSize, GetAlivePool());
            input.Resize(new_dims);
        } else {
            auto tmp = new Data();
            fastllm::Permute(input, axis, *tmp);

            memcpy(input.cpuData, tmp->cpuData, input.unitSize * input.Count(0));
            input.Resize(tmp->dims);
            delete tmp;
        }
    }

    void CpuRotatePosition2DOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        Data &sinData = *(datas.find("sin")->second);
        Data &cosData = *(datas.find("cos")->second);
        int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 64;

        int len = data.dims[0], bs = data.dims[1];
        int spatial = data.Count(2);
        int n = data.dims[2], m = data.dims[3];
        int stride = (int)sinData.dims[1];
        for (int l = 0; l < len; l++) {
            for (int b = 0; b < bs; b++) {
                for (int part = 0; part < 2; part++) {
                    int index = (int) ((float *) positionIds.cpuData)[(b * 2 + part) * positionIds.dims.back() + l];
                    float *sin = ((float*)sinData.cpuData) + stride * index;
                    float *cos = ((float*)cosData.cpuData) + stride * index;
                    float *d = (float *) data.cpuData + (l * bs + b) * spatial + part * m / 2;
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < rotaryDim && j < m / 4; j++) {
                            float a = d[j], b = d[j + m / 4];
                            d[j] = a * cos[j] - b * sin[j];
                            d[j + m / 4] = a * sin[j] + b * cos[j];
                        }

                        d += m;
                    }
                }
            }
        }
    }

    void CpuNearlyRotatePosition2DOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        Data &sinData = *(datas.find("sin")->second);
        Data &cosData = *(datas.find("cos")->second);
        int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 64;

        int len = data.dims[0], bs = data.dims[1];
        int spatial = data.Count(2);
        int n = data.dims[2], m = data.dims[3];
        int stride = (int)sinData.dims[1];
        for (int l = 0; l < len; l++) {
            for (int b = 0; b < bs; b++) {
                if (data.dataType == DataType::FLOAT32) {
                    int index = (int) ((float *) positionIds.cpuData)[(b * 2) * positionIds.dims.back() + l];
                    float *sin = ((float*)sinData.cpuData) + stride * index;
                    float *cos = ((float*)cosData.cpuData) + stride * index;

                    float *d = (float *) data.cpuData + (l * bs + b) * spatial;
                    for (int i = 0; i < n; i++) {
                        int j = 0;
                        for (; j < rotaryDim; j += 2) {
                            float a = d[j], b = d[j + 1];
                            d[j] = a * cos[j / 2] - b * sin[j / 2];
                            d[j + 1] = a * sin[j / 2] + b * cos[j / 2];
                        }
                        d += m;
                    }
                } else if (data.dataType == DataType::FLOAT16) {
                    int index = (int) ((float *) positionIds.cpuData)[(b * 2) * positionIds.dims.back() + l];
                    float *sin = ((float*)sinData.cpuData) + stride * index;
                    float *cos = ((float*)cosData.cpuData) + stride * index;

                    uint16_t *d = (uint16_t *) data.cpuData + (l * bs + b) * spatial;
                    for (int i = 0; i < n; i++) {
                        int j = 0;
                        for (; j < rotaryDim; j += 2) {
                            float a = fp16tofp32.dict[d[j]], b = fp16tofp32.dict[d[j + 1]];
                            d[j] = float_to_half(a * cos[j / 2] - b * sin[j / 2]);
                            d[j + 1] = float_to_half(a * sin[j / 2] + b * cos[j / 2]);
                        }
                        d += m;
                    }
                }
            }
        }
    }

    struct MultiThreadLlamaRotatePosition2DFloatOp : MultiThreadBaseOp {
        DataType dataType;
        float *data, *positionIds, *sinData, *cosData;
        int bs, len, n, m, stride, spatial, posDim, rotaryDim;      
        int st, end;

        MultiThreadLlamaRotatePosition2DFloatOp 
            (DataType dataType, float *data, float *positionIds, float *sinData, float *cosData, 
            int bs, int len, int n, int m, int stride, int spatial, int posDim, int rotaryDim, 
            int st, int end) : 
            dataType(dataType), data(data), positionIds(positionIds), sinData(sinData), cosData(cosData), 
            bs(bs), len(len), n(n), m(m), stride(stride), spatial(spatial), posDim(posDim), rotaryDim(rotaryDim), 
            st(st), end(end) {}

        void Run() {
            if (dataType == DataType::FLOAT32) {
                for (int idx = st; idx < end; idx++) {
                    int b = idx / len;
                    int l = idx % len;
                    int index = (int) ((float *) positionIds)[b * posDim + l];
                    float *sin = ((float *) sinData) + stride * index;
                    float *cos = ((float *) cosData) + stride * index;
                    float *d = (float *) data + (b * len + l) * spatial;
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < rotaryDim && j < m / 2; j++) {
                            float a = d[j], b = d[j + m / 2];
                            d[j] = a * cos[j] - b * sin[j];
                            d[j + m / 2] = a * sin[j] + b * cos[j];
                        }
                        d += m;
                    }
                }
            } else {
                for (int idx = st; idx < end; idx++) {
                    int b = idx / len;
                    int l = idx % len;
                    int index = (int) ((float *) positionIds)[b * posDim + l];
                    float *sin = ((float *) sinData) + stride * index;
                    float *cos = ((float *) cosData) + stride * index;
                    uint16_t *d = (uint16_t *) data + (b * len + l) * spatial;
                    for (int i = 0; i < n; i++) {
                        for (int j = 0; j < rotaryDim && j < m / 2; j++) {
                            float a = fp16tofp32.dict[d[j]], b = fp16tofp32.dict[d[j + m / 2]];
                            d[j] = float_to_half(a * cos[j] - b * sin[j]);
                            d[j + m / 2] = float_to_half(a * sin[j] + b * cos[j]);
                        }
                        d += m;
                    }
                }
            }
        }
    };

    static void RunMultiThreadLlamaRotatePosition2DFloat(DataType dataType, float *data, float *positionIds, float *sinData, float *cosData, 
            int bs, int len, int n, int m, int stride, int spatial, int posDim, int rotaryDim, AliveThreadPool *pool) {
        if (bs * len == 1) {
            (MultiThreadLlamaRotatePosition2DFloatOp(dataType, data, positionIds, sinData, cosData, bs, len, n, m, stride, spatial, posDim, rotaryDim, 0, bs * len)).Run();
            return;
        }

        int threadNum = pool->threads.size();
        int per = (bs * len) / pool->threads.size();
        int cur = 0;
        std::vector<fastllm::MultiThreadLlamaRotatePosition2DFloatOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? (bs * len) : cur + per + (cur + per * (threadNum - i) < (bs * len)));
            ops.push_back(new MultiThreadLlamaRotatePosition2DFloatOp(
                dataType, data, positionIds, sinData, cosData, bs, len, n, m, stride, spatial, posDim, rotaryDim, cur, end));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void CpuLlamaRotatePosition2DOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        Data &sinData = *(datas.find("sin")->second);
        Data &cosData = *(datas.find("cos")->second);
        int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 128;

        int bs = data.dims[0], len = data.dims[1];
        int spatial = data.Count(2);
        int n = data.dims[2], m = data.dims[3];
        int stride = (int)sinData.dims[1];
        RunMultiThreadLlamaRotatePosition2DFloat(data.dataType, (float*)data.cpuData, (float*)positionIds.cpuData, 
            (float*)sinData.cpuData, (float*)cosData.cpuData, bs, len, n, m, stride, spatial, 
            positionIds.dims.back(), rotaryDim, GetAlivePool());
    }

    void CpuRepeatPenaltyOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                         const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &penalty = *(datas.find("penalty")->second);
        AssertInFastLLM(input.dataType == DataType::FLOAT32 && penalty.dataType == DataType::FLOAT32,
                        "Repeat Penalty error: Data's type should be float32.\n");
        float *inputData = (float*)input.cpuData;
        float *penaltyData = (float*)penalty.cpuData;

        int len = input.Count(0);
        for (int i = 0; i < len; i++) {
            inputData[i] = inputData[i] < 0 ? inputData[i] * penaltyData[i] : inputData[i] / penaltyData[i];
        }
    }

    void CpuApplyLognAttnOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &lognAttn = *(datas.find("lognAttn")->second);
        Data &positionIds = *(datas.find("positionIds")->second);

        float *inputData = (float *) input.cpuData;
        float *lognData = (float *) lognAttn.cpuData;

        int batch = input.dims[0];
        int seqLen = input.dims[1];
        int spatial = input.Count(2);
        int curPos = (int) ((float *) positionIds.cpuData) [0];
        for (int b = 0; b < batch; b++) {
            float *curInput = inputData + b * seqLen * spatial;
            for (int i = 0; i < seqLen; i++) {
                float logn = lognData[i + curPos];
                for (int s = 0; s < spatial; s++) {
                    curInput[s] *= logn;
                }
                curInput += spatial;
            }
        }
    }
}