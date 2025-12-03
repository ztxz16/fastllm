//
// Created by huangyuyang on 6/13/23.
//

#define _USE_MATH_DEFINES
#include "devices/cpu/cpudevice.h"
#include "devices/cpu/computeutils.h"

#include <cstring>
#include <thread>

#include <cfloat>
#include <cmath>
#include <atomic>

#ifdef __aarch64__
#include <arm_neon.h>
#include "armMath.h"
#endif

#ifdef __AVX2__
#include "avxMath.h"
#endif

#include "utils.h"
#include "gguf.h"

namespace fastllm {
    CpuDevice::CpuDevice() {
        this->deviceType = "cpu";
        this->ops["ToFloat16"] = (BaseOperator*)(new CpuToFloat16());
        this->ops["ToFloat32"] = (BaseOperator*)(new CpuToFloat32());
        this->ops["ConvertToFloat16"] = (BaseOperator*)(new CpuConvertToFloat16());
        this->ops["ConvertToFloat32"] = (BaseOperator*)(new CpuConvertToFloat32());

        this->ops["Attention"] = (BaseOperator*)(new CpuAttention());
        this->ops["MergeMOE"] = (BaseOperator*)(new CpuMergeMOE());
        this->ops["MergeMLA"] = (BaseOperator*)(new CpuMergeMLA());
        this->ops["CopyKVCache"] = (BaseOperator*)(new CpuCopyKVCacheOp());
        this->ops["Embedding"] = (BaseOperator*)(new CpuEmbedding());
        this->ops["LayerNorm"] = (BaseOperator*)(new CpuLayerNormOp());
        this->ops["RMSNorm"] = (BaseOperator*)(new CpuRMSNormOp());
        this->ops["Linear"] = (BaseOperator*)(new CpuLinearOp());
        this->ops["Conv1DPerChannel"] = (BaseOperator*)(new CpuConv1DPerChannel());
        this->ops["Conv2D"] = (BaseOperator*)(new CpuConv2DOp());
        this->ops["Split"] = (BaseOperator*)(new CpuSplitOp());
        this->ops["Repeat"] = (BaseOperator*)(new CpuRepeatOp());
        this->ops["Cat"] = (BaseOperator*)(new CpuCatOp());
        this->ops["CatDirect"] = (BaseOperator*)(new CpuCatDirectOp());
        this->ops["MatMul"] = (BaseOperator*)(new CpuMatMulOp());
        this->ops["MatMulTransB"] = (BaseOperator*)(new CpuMatMulTransBOp());
        this->ops["SoftMax"] = (BaseOperator*)(new CpuSoftMaxOp());
        this->ops["Normalize"] = (BaseOperator*)(new CpuNormalizeOp());
        this->ops["Silu"] = (BaseOperator*)(new CpuSiluOp());
        this->ops["TanH"] = (BaseOperator*)(new CpuTanHOp());
        this->ops["Relu"] = (BaseOperator*)(new CpuReluOp());
        this->ops["Exp"] = (BaseOperator*)(new CpuExpOp());
        this->ops["Sigmoid"] = (BaseOperator*)(new CpuSigmoidOp());
        this->ops["Gelu"] = (BaseOperator*)(new CpuGeluOp());
        this->ops["GeluNew"] = (BaseOperator*)(new CpuGeluNewOp());
        this->ops["Swiglu"] = (BaseOperator*)(new CpuSwigluOp());
        this->ops["SwigluGptOss"] = (BaseOperator*)(new CpuSwigluGptOssOp());
        this->ops["MambaSoftplus"] = (BaseOperator*)(new CpuMambaSoftplusOp());
        this->ops["Mul"] = (BaseOperator*)(new CpuMulOp());
        this->ops["MulTo"] = (BaseOperator*)(new CpuMulToOp());
        this->ops["Add"] = (BaseOperator*)(new CpuAddOp());
        this->ops["AddTo"] = (BaseOperator*)(new CpuAddToOp());
        this->ops["AttentionMask"] = (BaseOperator*)(new CpuAttentionMaskOp());
        this->ops["AttentionExtendedMask"] = (BaseOperator*)(new CpuAttentionExtendedMaskOp());
        this->ops["AlibiMask"] = (BaseOperator*)(new CpuAlibiMaskOp());
        this->ops["TransferAttn"] = (BaseOperator*)(new CpuTransferAttnOp());
        this->ops["RecurrentGatedDeltaRule"] = (BaseOperator*)(new CpuRecurrentGatedDeltaRuleOp());
        this->ops["CausalMask"] = (BaseOperator*)(new CpuCausalMaskOp());
        this->ops["TopK"] = (BaseOperator*)(new CpuTopKOp());
        this->ops["Permute"] = (BaseOperator*)(new CpuPermuteOp());
        this->ops["PermuteSelf"] = (BaseOperator*)(new CpuPermuteSelfOp());
        this->ops["RotatePosition2D"] = (BaseOperator*)(new CpuRotatePosition2DOp());
        this->ops["NearlyRotatePosition2D"] = (BaseOperator*)(new CpuNearlyRotatePosition2DOp());
        this->ops["LlamaRotatePosition2D"] = (BaseOperator*)(new CpuLlamaRotatePosition2DOp());
        this->ops["LlamaRotatePosition2DPart"] = (BaseOperator*)(new CpuLlamaRotatePosition2DPartOp());
        this->ops["RepeatPenalty"] = (BaseOperator*)(new CpuRepeatPenaltyOp());
        this->ops["ApplyLognAttn"] = (BaseOperator*)(new CpuApplyLognAttnOp());
        this->ops["CumSumLastDim"] = (BaseOperator*)(new CpuCumSumLastDimOp());
        this->ops["MakeDecayMask"] = (BaseOperator*)(new CpuMakeDecayMaskOp());

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

    CPUInstructInfo cpuInstructInfo;

#ifdef __AVX2__
    extern int DotU4U8_AVX512VNNI(uint8_t *a, uint8_t *b, int n);

    int DotU4U8(uint8_t *a, uint8_t *b, int n) {
         if (cpuInstructInfo.hasAVX512VNNI) {
            return DotU4U8_AVX512VNNI(a, b, n);
         }
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

    FP16ToFP32Manager fp16tofp32;
    BF16ToFP32Manager bf16tofp32;
    FP8E4M3ToFP32Manager fp8e4m3tofp32;

    void Float16ToFloat32(uint16_t *float16, float *float32, int len) {
        int i = 0;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        for (; i + 7 < len; i += 8) {
            float16x8_t input_vec = vld1q_f16((float16_t*)float16 + i);
            float32x4_t output_vec1 = vcvt_f32_f16(vget_low_f16(input_vec));
            float32x4_t output_vec2 = vcvt_f32_f16(vget_high_f16(input_vec));
            vst1q_f32(float32 + i, output_vec1);
            vst1q_f32(float32 + i + 4, output_vec2);
        }
#endif
        for (; i < len; i++) {
            float32[i] = fp16tofp32.dict[float16[i]];
        }
    }

    void Float32ToFloat16(float *float32, uint16_t *float16, int len) {
        int i = 0;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        for (; i + 3 < len; i += 4) {
            float32x4_t input_vec = vld1q_f32(float32 + i);
            float16x4_t output_vec = vcvt_f16_f32(input_vec);
            vst1_f16((float16_t*)float16 + i, output_vec);
        }
#endif
#ifdef __AVX__
        for (; i + 7 < len; i += 8) {
            __m256 input_vec = _mm256_loadu_ps(float32 + i);  // 加载 8 个 float32
            __m128i output_vec = _mm256_cvtps_ph(input_vec, _MM_FROUND_TO_NEAREST_INT);  // 转换为 8 个 float16
            _mm_storeu_si128((__m128i*)(float16 + i), output_vec);  // 存储 8 个 float16
        }
#endif
        for (; i < len; i++) {
            float16[i] = float_to_half(float32[i]);
        }
    }

    void Float32ToBFloat16(float *float32, uint16_t *bfloat16, int len) {
        int i = 0;
        
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        for (; i + 3 < len; i += 4) {
            // Load 4 float32 values
            float32x4_t f32x4 = vld1q_f32(&float32[i]);
            
            // Reinterpret as uint32 to access bits
            uint32x4_t u32x4 = vreinterpretq_u32_f32(f32x4);
            
            // Shift right by 16 bits to get bfloat16 bits
            uint32x4_t shifted = vshrq_n_u32(u32x4, 16);
            
            // Narrow to 16-bit (takes bottom 16 bits from each 32-bit element)
            uint16x4_t bf16x4 = vmovn_u32(shifted);
            
            // Store 4 bfloat16 values
            vst1_u16(&bfloat16[i], bf16x4);
        }
#endif
    
#ifdef __AVX__
        for (; i + 7 < len; i += 8) {
            __m256i float_vec = _mm256_loadu_si256((__m256i*)&float32[i]);
            __m256i shifted = _mm256_srli_epi32(float_vec, 16);
            __m128i lo = _mm256_castsi256_si128(shifted);
            __m128i hi = _mm256_extracti128_si256(shifted, 1);
            __m128i packed = _mm_packus_epi32(lo, hi);
            _mm_storeu_si128((__m128i*)&bfloat16[i], packed);
        }
#endif
        // 标量处理剩余元素
        for (; i < len; i++) {
            uint32_t val;
            memcpy(&val, &float32[i], sizeof(val));
            bfloat16[i] = (uint16_t)(val >> 16);
        }
    }

    void Float16ToBFloat16(uint16_t *float16, uint16_t *bfloat16, int len) {
        int i = 0;
#ifdef __AVX__
        for (; i + 7 < len; i += 8) {
            __m256 _float_vec = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *) (float16 + i)));
            __m256i float_vec = *((__m256i *)&_float_vec);
            __m256i shifted = _mm256_srli_epi32(float_vec, 16);
            __m128i lo = _mm256_castsi256_si128(shifted);
            __m128i hi = _mm256_extracti128_si256(shifted, 1);
            __m128i packed = _mm_packus_epi32(lo, hi);
            _mm_storeu_si128((__m128i*)&bfloat16[i], packed);
        }
#endif
        for (; i < len; i++) {
            uint32_t val;
            memcpy(&val, &fp16tofp32.dict[float16[i]], sizeof(val));
            bfloat16[i] = (uint16_t)(val >> 16);
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

    void CpuConvertToFloat16::Reshape(const std::string &opType, const fastllm::DataDict &datas,
        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data *input = (datas.find("input")->second);
        Data *output = (datas.find("output")->second);
        output->dataType = DataType::FLOAT16;
        output->Resize(input->dims);
        if (input->expansionDims.size() != 0)
            output->Expansion(input->expansionDims);
    }

    void CpuConvertToFloat16::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        if (input.dataType == DataType::FLOAT16) {
            memcpy(output.cpuData, input.cpuData, input.GetBytes());
            return;
        }
        if (input.dataType == DataType::FLOAT32) {
            Float32ToFloat16((float*)input.cpuData, (uint16_t*)output.cpuData, input.Count(0));
        } else {
            ErrorInFastLLM("ToFloat16: unsupport dataType.\n");
        }
    }

    void CpuConvertToFloat32::Reshape(const std::string &opType, const fastllm::DataDict &datas,
        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data *input = (datas.find("input")->second);
        Data *output = (datas.find("output")->second);
        output->dataType = DataType::FLOAT32;
        output->Resize(input->dims);
        if (input->expansionDims.size() != 0)
            output->Expansion(input->expansionDims);
    }

    void CpuConvertToFloat32::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        if (input.dataType == DataType::FLOAT32) {
            memcpy(output.cpuData, input.cpuData, input.GetBytes());
            return;
        }
        if (input.dataType == DataType::FLOAT16) {
            Float16ToFloat32((uint16_t*)input.cpuData, (float*)output.cpuData, input.Count(0));
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
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : q.dims[0] / k.dims[0];

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
            int base = k1 - q1;
            for (int i = 0; i < q1; i++) {
                float maxValue = -10000, sum = 0.0;
                for (int j = 0; j < k1; j++) {
                    if (maskd && maskd[i * k1 + j] > 0.99) {
                        qk[j] = -10000;
                        continue;
                    }
                    if (!maskd && (base + i) < j) {
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
        int group = intParams.find("group") != intParams.end() ? intParams.find("group")->second : q.dims[0] / k.dims[0];
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
                            std::vector <float> &inputSums, std::vector <float> &iscales, std::vector <float> &izeros, 
                            int permuteType) {
        if (uinput.size() < n * m) {
            uinput.resize(n * m);
        }
        inputConfigs.resize(n * group);
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
                                inputSums.data() + cur * group, iscales.data() + cur * group, izeros.data() + cur * group, permuteType));
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
                                            inputSums.data(), iscales.data(), izeros.data(), permuteType).Run();
        }
    }

    void MultiThreadSwigluGptOssOp::Run() {
        for (int o = 0; o < n; o++) {
                float *cur = (float*)input + o * inputStride;
                float *out = (float*)output + o * outputStride;
                int i = 0;
    #ifdef __aarch64__X
                float32x4_t c1 = vdupq_n_f32(1.0f);
                for (; i + 3 < len; i += 4) {
                    float32x4_t vx = vld1q_f32(cur + i);
                    float32x4_t vy = vld1q_f32(cur + i + mid);
                    vx = vdivq_f32(vx, vaddq_f32(c1, exp_ps(vnegq_f32(vx))));
                    vy = vmulq_f32(vx, vy);
                    vst1q_f32(out + i, vy);
                }
    #endif
    #ifdef __AVX2__X
                for (; i + 7 < len; i += 8) {  // Process 8 elements at a time
                    // Load x values (inputData[i..i+7]) and y values (inputData[i+mid..i+mid+7])
                    __m256 x = _mm256_loadu_ps(&cur[i]);
                    __m256 y = _mm256_loadu_ps(&cur[i + mid]);
                    
                    // Compute sigmoid: 1.0 / (1.0 + expf(-x))
                    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
                    __m256 exp_neg_x = exp256_ps(neg_x);  // See note below about exp_ps
                    __m256 denom = _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg_x);
                    __m256 sigmoid = _mm256_div_ps(x, denom);
                    
                    // Multiply by y and store result
                    __m256 result = _mm256_mul_ps(sigmoid, y);
                    _mm256_storeu_ps(&out[i], result);
                }
    #endif
                for (; i < len; i++) {
                    float x = cur[i * 2], y = cur[i * 2 + 1];
                    float gate = std::min(x, 7.0f);
                    float up = std::max(-7.0f, std::min(y, 7.0f));
                    float glu = gate * (1.0 / (1.0 + exp(-(gate * 1.702))));
                    out[i] = (up + 1) * glu;                    
                }
        }
    }

    void MultiThreadSwigluOp::Run() {
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
    #ifdef __AVX2__
                for (; i + 7 < len; i += 8) {  // Process 8 elements at a time
                    // Load x values (inputData[i..i+7]) and y values (inputData[i+mid..i+mid+7])
                    __m256 x = _mm256_loadu_ps(&cur[i]);
                    __m256 y = _mm256_loadu_ps(&cur[i + mid]);
                    
                    // Compute sigmoid: 1.0 / (1.0 + expf(-x))
                    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
                    __m256 exp_neg_x = exp256_ps(neg_x);  // See note below about exp_ps
                    __m256 denom = _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg_x);
                    __m256 sigmoid = _mm256_div_ps(x, denom);
                    
                    // Multiply by y and store result
                    __m256 result = _mm256_mul_ps(sigmoid, y);
                    _mm256_storeu_ps(&out[i], result);
                }
    #endif
                for (; i < len; i++) {
                    float x = cur[i], y = cur[i + mid];
                    out[i] = (x / (1.0 + expf(-x))) * y;
                }
        }
    }

    void MultiThreadSwigluFloat16Op::Run() {
        for (int o = 0; o < n; o++) {
            uint16_t *cur = (uint16_t*)input + o * inputStride;
            uint16_t *out = (uint16_t*)output + o * outputStride;
            int i = 0;
#ifdef __AVX2__
            for (; i + 7 < len; i += 8) {  // Process 8 elements at a time
                __m128i x_half = _mm_loadu_si128((const __m128i*)&cur[i]);
                __m256 x = _mm256_cvtph_ps(x_half);  // Convert float16 to float32
    
                // Load 8 float16 values from cur[i+mid..i+mid+7] and convert to float32
                __m128i y_half = _mm_loadu_si128((const __m128i*)&cur[i + mid]);
                __m256 y = _mm256_cvtph_ps(y_half);  // Convert float16 to float32
                    
                // Compute sigmoid: 1.0 / (1.0 + expf(-x))
                __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
                __m256 exp_neg_x = exp256_ps(neg_x);  // See note below about exp_ps
                __m256 denom = _mm256_add_ps(_mm256_set1_ps(1.0f), exp_neg_x);
                __m256 sigmoid = _mm256_div_ps(x, denom);
                    
                // Multiply by y and store result
                __m256 result = _mm256_mul_ps(sigmoid, y);
                
                // Convert result back to float16 and store
                __m128i result_half = _mm256_cvtps_ph(result, _MM_FROUND_TO_NEAREST_INT);
                _mm_storeu_si128((__m128i*)&out[i], result_half);
            }
#endif
            for (; i < len; i++) {
                float x = fp16tofp32.dict[cur[i]], y = fp16tofp32.dict[cur[i + mid]];
                out[i] = float_to_half((x / (1.0 + expf(-x))) * y);
            }
        }
    }

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

    struct MOEIntSingleVarManager {
        std::vector<LowBitConfig> inputConfigs;
        std::vector<uint8_t> uinput;
        std::vector <float> inputSums;
        std::vector <float> iscales, izeros;
        std::vector <std::vector <float> > middles, results;
        std::vector <std::vector <LowBitConfig> > inputConfigsDown;
        std::vector <std::vector <uint8_t> > uinputsDown;
        std::vector <std::vector <float> > inputSumsDown;
        std::vector <std::vector <float> > iscalesDown, izerosDown;
    } moeIntSingleVarManager;

    struct moeFloatSingleVarManager {
        std::vector <std::vector <float> > middles, results;
        std::vector <int> localKs;
        std::vector <float*> tempResults;
        std::vector <uint16_t> bf16Input;
    } moeFloatSingleVarManager;

    struct FastllmMoeDataManager {
            std::vector <float, alignedAllocator<float, 64> > gateUpOutput, swigluOutput, downOutput, reduceOutput;
            std::vector <uint8_t, alignedAllocator<uint8_t, 64> > realInput, expandInput, downInput;
    } fastllmMoeDataManager;

    void FastllmGemm (int n, int m, int k, 
        const void *A, long lda, // A [n * m], lda = bytes for 1 row in A
        const void *B, long ldb, // B [k * m], ldb = bytes for 1 row in B
        void *C, long ldc, // C[n * k], ldc = bytes for 1 row in C
        int st, int end, // calc C[0 : n, st : end]
        DataType AType, DataType BType, DataType CType
    ) {
        bool finish = false;
// printf("into fastllm gemm %s %s %s\n", GetDataTypeName(AType).c_str(), GetDataTypeName(BType).c_str(), GetDataTypeName(CType).c_str());
        if (AType >= DataType::DATA_GGUF_FORMAT && AType < DataType::DATA_GGUF_FORMAT_END) {
            if (CType == DataType::FLOAT32) {
                LinearQ8K_GGUF_Kernel((uint8_t*)A, (uint8_t*)B, nullptr, (float*)C, n, m, ldc / sizeof(float), st, end, AType, BType);
                finish = true;
            }
        } else if (AType == DataType::INF_INT8_PERCHANNEL) {
        	if (CType == DataType::FLOAT32) {
        		if (BType == DataType::INT4_PERCHANNEL) {
                    LinearINT8PERCHANNEL_INT4PERCHANNEL_Kernel((uint8_t*)A, (uint8_t*)B, nullptr, (float*)C, n, m, ldc / sizeof(float), st, end);
                    finish = true;
        		} else if (BType == DataType::INT8_PERCHANNEL) {
                    LinearINT8PERCHANNEL_INT8PERCHANNEL_Kernel((uint8_t*)A, (uint8_t*)B, nullptr, (float*)C, n, m, ldc / sizeof(float), st, end);
                    finish = true;
                }
        	}
        } else if (AType == DataType::INF_INT8_GROUP128) {
        	if (CType == DataType::FLOAT32) {
        		if (BType == DataType::INT4_GROUP128) {
                    LinearINT8GROUP128_INT4GROUP128_Kernel((uint8_t*)A, (uint8_t*)B, nullptr, (float*)C, n, m, ldc / sizeof(float), st, end);
                    finish = true;
        		}
        	}
        } else if (AType == DataType::FLOAT32) {
            if (CType == DataType::FLOAT32) {
                if (BType == DataType::FLOAT32) {
                    for (int i = 0; i < n; i++) {
                        float *floatA = (float*)((uint8_t*)A + i * lda);
                        float *floatC = (float*)((uint8_t*)C + i * ldc);
                        for (int j = st; j < end; j++) {
                            float *floatB = (float*)((uint8_t*)B + j * ldb);
                            float sum = 0.0f;
                            for (int l = 0; l < m; l++) {
                                sum += floatA[l] * floatB[l];
                            }
                            floatC[j] = sum;
                        }
                    }
                    finish = true;
                } else if (BType == DataType::BFLOAT16) {
                    for (int i = 0; i < n; i++) {
                        float *floatA = (float*)((uint8_t*)A + i * lda);
                        float *floatC = (float*)((uint8_t*)C + i * ldc);
                        for (int j = st; j < end; j++) {
                            uint16_t *floatB = (uint16_t*)((uint8_t*)B + j * ldb);
                            float sum = 0.0f;
                            for (int l = 0; l < m; l++) {
                                sum += floatA[l] * bf16tofp32.dict[floatB[l]];
                            }
                            floatC[j] = sum;
                        }
                    }
                    finish = true;
                } else if (BType == DataType::FLOAT16) {
                    MultiThreadLinearFloat32Float16Op (
                        (float*)A, (uint16_t*)B, nullptr, (float*)C, n, m, ldc / sizeof(float), st, end
                    ).Run();
                    finish = true;
                }
            }
        } else if (AType == DataType::BFLOAT16) {
            if (CType == DataType::FLOAT32) {
                if (BType == DataType::BFLOAT16) {                    
                    // LinearBFloat16BFloat16_Kernel((uint16_t*)A, (uint16_t*)B, nullptr, (float*)C, n, m, ldc / sizeof(float), st, end);
                    MultiThreadLinearBFloat16BFloat16Op (
                        (uint16_t*)A, (uint16_t*)B, nullptr, (float*)C, n, m, ldc / sizeof(float), st, end
                    ).Run();
                    finish = true;
                } else if (BType == FP8_E4M3_BLOCK_128) {
                    // A是BFLOAT16, B是FP8_E4M3_BLOCK_128格式（fp8数据+scale）, C是FLOAT32
                    // 为需要计算的行分配临时bf16缓冲区
                    if (n > 31) {
                        std::vector<uint16_t> bf16B_temp((end - st) * m);
                        // 转换fp8到bf16，仅转换需要计算的行[st:end]
                        int block_size = 128;
                        int num_blocks = (m + block_size - 1) / block_size;
                        int last_block_size = (m % block_size == 0) ? block_size : (m % block_size);
                        for (int j = st; j < end; j++) {
                            uint8_t *rowStart = (uint8_t*)B + j * ldb;  // ldb应该是每行的总字节数
                            uint16_t *bf16B_row = bf16B_temp.data() + (j - st) * m;
                            
                            // 按block进行处理
                            for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
                                // 计算当前block的大小（最后一个block可能不完整）
                                int current_block_size = (block_idx == num_blocks - 1) ? last_block_size : block_size;
                                
                                // 计算当前block的起始位置
                                // 每个block占用 128字节(fp8) + 4字节(float scale)
                                uint8_t *block_start = rowStart + block_idx * (block_size + sizeof(float));
                                uint8_t *fp8_ptr = block_start;
                                float *scale_ptr = (float*)(block_start + block_size);
                                
                                // 转换当前block中的每个fp8到bf16
                                int base_idx = block_idx * block_size;
                                for (int l = 0; l < current_block_size; l++) {
                                    // fp8转fp32并乘以scale
                                    float fp32_val = fp8e4m3tofp32.dict[fp8_ptr[l]] * (*scale_ptr);
                                    
                                    // fp32转bf16
                                    uint32_t val;
                                    memcpy(&val, &fp32_val, sizeof(val));
                                    bf16B_row[base_idx + l] = (uint16_t)(val >> 16);
                                }
                            }
                        }

                        MultiThreadLinearBFloat16BFloat16Op (
                            (uint16_t*)A, bf16B_temp.data(), nullptr, ((float*)C) + st, n, m, ldc / sizeof(float), 0, end - st
                        ).Run();
                        finish = true;
                        // LinearBFloat16BFloat16_Kernel((uint16_t*)A, bf16B_temp.data(), nullptr, ((float*)C) + st, n, m, ldc / sizeof(float), 0, end - st);
                    } else {
                        LinearBFloat16_FP8E4M3BLOCK128_Kernel((uint16_t*)A, (uint8_t*)B, nullptr, (float*)C, n, m, ldc / sizeof(float), st, end);
                        finish = true;
                    }
                } else if (BType == FP8_E4M3_PERCHANNEL) {
                    // A是BFLOAT16, B是FP8_E4M3_PERCHANNEL格式（fp8数据+scale）, C是FLOAT32
                    // 为需要计算的行分配临时bf16缓冲区
                    if (n > 31) {
                        std::vector<uint16_t> bf16B_temp((end - st) * m);
                        // 转换fp8到bf16，仅转换需要计算的行[st:end]
                        int block_size = m;
                        int num_blocks = (m + block_size - 1) / block_size;
                        int last_block_size = (m % block_size == 0) ? block_size : (m % block_size);
                        for (int j = st; j < end; j++) {
                            uint8_t *rowStart = (uint8_t*)B + j * ldb;  // ldb应该是每行的总字节数
                            uint16_t *bf16B_row = bf16B_temp.data() + (j - st) * m;
                            
                            // 按block进行处理
                            for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
                                // 计算当前block的大小（最后一个block可能不完整）
                                int current_block_size = (block_idx == num_blocks - 1) ? last_block_size : block_size;
                                
                                // 计算当前block的起始位置
                                // 每个block占用 128字节(fp8) + 4字节(float scale)
                                uint8_t *block_start = rowStart + block_idx * (block_size + sizeof(float));
                                uint8_t *fp8_ptr = block_start;
                                float *scale_ptr = (float*)(block_start + block_size);
                                
                                // 转换当前block中的每个fp8到bf16
                                int base_idx = block_idx * block_size;
                                for (int l = 0; l < current_block_size; l++) {
                                    // fp8转fp32并乘以scale
                                    float fp32_val = fp8e4m3tofp32.dict[fp8_ptr[l]] * (*scale_ptr);
                                    
                                    // fp32转bf16
                                    uint32_t val;
                                    memcpy(&val, &fp32_val, sizeof(val));
                                    bf16B_row[base_idx + l] = (uint16_t)(val >> 16);
                                }
                            }
                        }

                        MultiThreadLinearBFloat16BFloat16Op (
                            (uint16_t*)A, bf16B_temp.data(), nullptr, ((float*)C) + st, n, m, ldc / sizeof(float), 0, end - st
                        ).Run();
                        finish = true;
                    } else {
                        LinearBFloat16_FP8E4M3PERCHANNEL_Kernel((uint16_t*)A, (uint8_t*)B, nullptr, (float*)C, n, m, ldc / sizeof(float), st, end);
                        finish = true;
                    }
                } else if (BType == AWQ_4BIT_128) {
                    // A是BFLOAT16, B是AWQ_4BIT_128格式（uint4权重+zero+scale）, C是FLOAT32
                    // 为需要计算的行分配临时bf16缓冲区
                    /* if (n > 31) {
                        std::vector<uint16_t> bf16B_temp((end - st) * m);
                        bool success = AWQ4BIT128_TO_BFloat16_Kernel((uint8_t*)B, bf16B_temp.data(), m, st, end, ldb);
                        LinearBFloat16BFloat16_Kernel((uint16_t*)A, bf16B_temp.data(), nullptr, ((float*)C) + st, n, m, ldc / sizeof(float), 0, end - st);
                    } else {
                        LinearBFloat16_AWQ4BIT128_Kernel((uint16_t*)A, (uint8_t*)B, nullptr, (float*)C, n, m, ldc / sizeof(float), st, end);
                    } */ 
                } else if (BType >= DataType::DATA_GGUF_FORMAT && BType < DataType::DATA_GGUF_FORMAT_END) {
                    std::vector <float> fp32B_temp((end - st) * m);
                    std::vector <uint16_t> bf16B_temp((end - st) * m);
                    ggml_type weightType = (ggml_type)((int)BType - (int)DataType::DATA_GGUF_FORMAT);

                    auto toFloat = ggml_type_to_float(weightType);
                    AssertInFastLLM(toFloat != nullptr, "WeightImportGGUFTensor: weight (type " + std::string(ggml_type_name(weightType)) + ") can't convert to fp32.");
                    toFloat(((uint8_t*)B) + ldb * st, fp32B_temp.data(), (end - st) * m);
                    Float32ToBFloat16(fp32B_temp.data(), bf16B_temp.data(), (end - st) * m);
                    MultiThreadLinearBFloat16BFloat16Op (
                            (uint16_t*)A, bf16B_temp.data(), nullptr, ((float*)C) + st, n, m, ldc / sizeof(float), 0, end - st
                    ).Run();
                    finish = true;
                }
            }
        }
        
        if (!finish) {
            ErrorInFastLLM("FastllmGemm Error: \nAType = " + GetDataTypeName(AType) + "\nBType = " + GetDataTypeName(BType) + "\nCType = " + GetDataTypeName(CType));
        }
    }

    void MultiThreadGemmOp::Run() {
            FastllmGemm(
                n, m, k,
                inputData, GetDataBytes(inputDataType, 1, m),
                weightData, GetDataBytes(weightDataType, 1, m),
                outputData, GetDataBytes(outputDataType, 1, k),
                st, end,
                inputDataType, weightDataType, outputDataType
            );
    }
            
    void MultiThreadReduceBatchOp::Run() {
            for (int i = batch_st; i < batch_end; i++) {
                // 处理第一个专家（初始化输出）
                int curPos = pos[i * k];
                float weight = weights[curPos];
                for (int h = hidden_st; h < hidden_end; h++) {
                    lastOutput[i * hidden_size + h] = weight * ((float*)downOutData)[curPos * hidden_size + h];
                }
                
                // 累加其余专家的贡献
                for (int expert_idx = 1; expert_idx < k; expert_idx++) {
                    curPos = pos[i * k + expert_idx];
                    weight = weights[curPos];
                    for (int h = hidden_st; h < hidden_end; h++) {
                        lastOutput[i * hidden_size + h] += weight * ((float*)downOutData)[curPos * hidden_size + h];
                    }
                }
            }
    }

    void MultiThreadReduceBatch(uint8_t *downOutData, DataType downOutDataType,
                    float *weights, float *lastOutput,
                    int *pos, int bsz, int k,
                    int hidden_size) {
        auto *pool = GetAlivePool();
        int threadNum = pool->threads.size();
        
        // 决定如何划分：尝试创建一个接近正方形的网格
        int batch_blocks = 1, hidden_blocks = threadNum;
        
        // 简单的启发式：如果bsz足够大，尝试在两个维度上划分
        if (bsz >= 4 && threadNum >= 4) {
            // 找到最佳的2D网格划分
            for (int b = 2; b <= std::min(bsz, threadNum); b++) {
                if (threadNum % b == 0) {
                    int h = threadNum / b;
                    if (h <= hidden_size) {
                        batch_blocks = b;
                        hidden_blocks = h;
                    }
                }
            }
        }
        
        std::vector<fastllm::MultiThreadReduceBatchOp*> ops;
        ops.reserve(threadNum);
        
        int batch_per = bsz / batch_blocks;
        int hidden_per = hidden_size / hidden_blocks;
        
        int op_idx = 0;
        for (int b = 0; b < batch_blocks; b++) {
            int batch_st = b * batch_per;
            int batch_end = (b == batch_blocks - 1) ? bsz : (b + 1) * batch_per;
            
            for (int h = 0; h < hidden_blocks; h++) {
                int hidden_st = h * hidden_per;
                int hidden_end = (h == hidden_blocks - 1) ? hidden_size : (h + 1) * hidden_per;
                
                ops.push_back(new MultiThreadReduceBatchOp(
                    downOutData, downOutDataType,
                    weights, lastOutput,
                    pos, bsz, k,
                    hidden_size,
                    batch_st, batch_end,
                    hidden_st, hidden_end));
                
                pool->PushOp(op_idx++, ops.back());
            }
        }
        
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    extern void Float32ToInfInt8PerChannelAVX2(const float* srcData, uint8_t* dstData, size_t columns);
    void Float32ToInfInt8PerChannel(const float* srcData, uint8_t* dstData, size_t columns) {
        if (cpuInstructInfo.hasAVX2) {
            Float32ToInfInt8PerChannelAVX2(srcData, dstData, columns);
            return;
        }
        // 目标内存布局：
        // [int8 * columns] [float scale] [int sum]
        int8_t* quantizedData = (int8_t*)dstData;
        float* scalePtr = (float*)(dstData + columns);
        int* sumPtr = (int*)(dstData + columns + sizeof(float));
        
        // 1. 找到这一行的最大绝对值
        float maxAbs = 0.0f;
        for (size_t i = 0; i < columns; i++) {
            float absVal = std::abs(srcData[i]);
            if (absVal > maxAbs) {
                maxAbs = absVal;
            }
        }
        
        // 2. 计算scale（对称量化，范围是 -127 到 127）
        float scale;
        if (maxAbs > 0) {
            scale = maxAbs / 127.0f;
        } else {
            scale = 1.0f;  // 避免除零
        }
        
        // 3. 量化并计算sum
        int sum = 0;
        for (size_t i = 0; i < columns; i++) {
            // 量化: q = round(x / scale)
            int quantized = std::round(srcData[i] / scale);
            
            // 裁剪到int8范围 [-127, 127]（对称量化通常不使用-128）
            if (quantized > 127) quantized = 127;
            if (quantized < -127) quantized = -127;
            
            quantizedData[i] = (int8_t)quantized;
            sum += quantized;
        }
        
        // 4. 存储scale和sum
        *scalePtr = scale;
        *sumPtr = sum;
    }

    void ConvertFromFloat32(void *dstData, DataType dstDataType, const float *floatData, size_t rows, size_t columns) {
        if (dstDataType == DataType::FLOAT32) {
            memcpy(dstData, floatData, rows * columns * sizeof(float));
        } else if (dstDataType == DataType::FLOAT16) {
            Float32ToFloat16((float*)floatData, (uint16_t*)dstData, rows * columns);
        } else if (dstDataType == DataType::BFLOAT16) {
            Float32ToBFloat16((float*)floatData, (uint16_t*)dstData, rows * columns);
        } else if (dstDataType == DataType::INF_INT8_PERCHANNEL) {
            size_t rowCount = GetDataBytes(dstDataType, 1, columns);
            for (int i = 0; i < rows; i++) {
                Float32ToInfInt8PerChannel (
                    (float*)floatData + i * columns, 
                    (uint8_t*)dstData + i * rowCount, 
                    columns
                );
            }
        } else if (dstDataType == DataType::INF_INT8_GROUP128) {
            rows *= (columns / 128);
            columns = 128;
            size_t rowCount = GetDataBytes(INF_INT8_PERCHANNEL, 1, columns);
            for (int i = 0; i < rows; i++) {
                Float32ToInfInt8PerChannel (
                    (float*)floatData + i * columns, 
                    (uint8_t*)dstData + i * rowCount, 
                    columns
                );
            }
        } else if (dstDataType >= DataType::DATA_GGUF_FORMAT && dstDataType < DataType::DATA_GGUF_FORMAT_END) {
            auto ggmlType = (ggml_type)((int)dstDataType - (int)DataType::DATA_GGUF_FORMAT);
            size_t rowCount = ggml_row_size(ggmlType, columns);
            for (int i = 0; i < rows; i++) {
                iqk_quantize_row_q8_K (
                        (float*)floatData + i * columns, (uint8_t*)dstData + i * rowCount, columns, 
                        ggmlType, ggmlType
                );
            }
        } else {
            ErrorInFastLLM("ConvertFromFloat32 failed with type" + GetDataTypeName(dstDataType));
        }
    }

    void MultiThreadConvertFromFloat32Op::Run() {
            // 计算每行的字节大小
            size_t rowSize = GetDataBytes(dstDataType, 1, columns);
            
            // 调用原始函数处理指定行范围
            void *dstStart = (char*)dstData + startRow * rowSize;
            const float *srcStart = floatData + startRow * columns;
            size_t rowsToProcess = endRow - startRow;
            
            ConvertFromFloat32(dstStart, dstDataType, srcStart, rowsToProcess, columns);
    }

    // 对应的多线程运行函数
    void RunMultiThreadConvertFromFloat32(void *dstData, DataType dstDataType, 
                                                const float *floatData, size_t rows, 
                                                size_t columns, AliveThreadPool *pool) {
        // 如果数据量较小，直接单线程处理
        if (rows * columns < 10000) {
            ConvertFromFloat32(dstData, dstDataType, floatData, rows, columns);
            return;
        }
        
        int threadNum = pool->threads.size();
        threadNum = std::min(threadNum, (int)rows);  // 线程数不超过行数
        
        // 如果行数太少，减少线程数
        if (rows < threadNum) {
            ConvertFromFloat32(dstData, dstDataType, floatData, rows, columns);
            return;
        }
        
        size_t rowsPerThread = rows / threadNum;
        size_t curRow = 0;
        
        std::vector<MultiThreadConvertFromFloat32Op*> ops;
        for (int i = 0; i < threadNum; i++) {
            size_t endRow = (i == threadNum - 1) ? rows : curRow + rowsPerThread;
            ops.push_back(new MultiThreadConvertFromFloat32Op(
                dstData, dstDataType, floatData, columns, curRow, endRow));
            curRow = endRow;
        }
        
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }
        
    void WorkStealingOp::Run() {
            // 首先执行自己的任务
            processOwnTasks();
            
            // 然后从其他线程偷取任务
            stealFromOthers();
            
            // 标记完成
            myState->completed.store(true, std::memory_order_release);
    }
        
    void WorkStealingOp::processOwnTasks() {
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
        
    void WorkStealingOp::stealFromOthers() {
            // 从当前线程开始，环形遍历其他线程
            for (int offset = 1; offset < totalThreads; offset++) {
                int targetId = (threadId + offset) % totalThreads;
                
                TaskState* otherState = (*allStates)[targetId];
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

    // 重构的动态任务调度函数，支持work-stealing
    void DynamicScheduleTasks(std::vector<MultiThreadBaseOp*>& ops) {
        auto *pool = GetAlivePool();
        int numThreads = pool->threads.size(); // 假设线程池有获取线程数的方法
        
        // 创建任务状态数组
        using TaskState = typename WorkStealingOp::TaskState;
        std::vector<TaskState*> taskStates(numThreads, nullptr);
        
        // 为每个线程分配任务状态
        for (int i = 0; i < numThreads; i++) {
            taskStates[i] = new (std::align_val_t{64}) TaskState();
            taskStates[i]->curr.store(0, std::memory_order_relaxed);
            taskStates[i]->end = 0;
            taskStates[i]->completed.store(false, std::memory_order_relaxed);
        }
        
        // 分配任务到各个线程
        int totalOps = ops.size();
        if (totalOps > 0) {
            // 计算每个线程的任务数量
            int tasksPerThread = totalOps / numThreads;
            int remainingTasks = totalOps % numThreads;
            
            int taskIndex = 0;
            for (int i = 0; i < numThreads; i++) {
                int numTasks = tasksPerThread + (i < remainingTasks ? 1 : 0);
                
                if (numTasks > 0) {
                    // 分配任务到该线程
                    taskStates[i]->tasks.clear();
                    taskStates[i]->tasks.reserve(numTasks);
                    
                    for (int j = 0; j < numTasks && taskIndex < totalOps; j++) {
                        taskStates[i]->tasks.push_back(ops[taskIndex++]);
                    }
                    
                    taskStates[i]->curr.store(0, std::memory_order_relaxed);
                    taskStates[i]->end = taskStates[i]->tasks.size();
                } else {
                    taskStates[i]->end = 0;
                }
            }
        }
        
        // 创建work-stealing ops并提交到线程池
        std::vector<WorkStealingOp*> wsOps(numThreads);
        for (int i = 0; i < numThreads; i++) {
            wsOps[i] = new WorkStealingOp(
                i, &taskStates, taskStates[i], numThreads
            );
            
            pool->PushOp(i, wsOps[i]);
        }
        
        // 等待所有线程完成
        for (int i = 0; i < numThreads; i++) {
            pool->Wait(i);
        }
        
        // 清理资源
        for (int i = 0; i < numThreads; i++) {
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
        for (auto* op : ops) {
            delete op;
        }
    }

    void MultiThreadRepackWeightsOp::Run() {
        for (int i = st; i < end; i++) {
            if (weights[i] != nullptr) {
                weights[i]->Repack();
            }
        }
    }

    void CpuMergeMOE::Run(const std::string &opType, const fastllm::DataDict &datas,
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

        if (weights[2]->dataType == DataType::DATA_GGUF_FORMAT && 
            !weights[2]->IsRepacked) {
            int dimsLen = logits.dims.size();
            int outer = logits.Count(0) / logits.Count(dimsLen - 1);
            int channels = logits.dims[dimsLen - 1];
            int len = channels * 2;
            
            auto *pool = GetAlivePool();
            int threadNum = pool->threads.size();
            int per = len / threadNum;
            int cur = 0;            
            std::vector<fastllm::MultiThreadRepackWeightsOp*> ops;
            for (int i = 0; i < threadNum; i++) {
                int end = cur + per;
                if (i == threadNum - 1) {
                    end = len;
                }
                ops.push_back(new MultiThreadRepackWeightsOp(weights, cur, end));
                cur = end;
            }
            for (int i = 0; i < ops.size(); i++) {
                pool->PushOp(i, ops[i]);
            }
            for (int i = 0; i < ops.size(); i++) {
                pool->Wait(i);
                delete ops[i];
            }
        }

        if ((input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16) && 
                (weights[2]->dataType == DataType::INT4_GROUP 
                || weights[2]->dataType == DataType::INT4_NOZERO 
                || weights[2]->dataType == DataType::INT8) &&
            input.dims[0] < 32) {
            int permuteType = 1;
            if (weights[2]->dataType == DataType::INT8) {
                permuteType = 0;
            }
            
            int dimsLen = logits.dims.size();
            int outer = logits.Count(0) / logits.Count(dimsLen - 1);
            int channels = logits.dims[dimsLen - 1];

            std::vector <float> vLogits, vInputs;
            float *floatLogits = ((float*)logits.cpuData);
            float *floatInput = (float*)input.cpuData;
            output.Allocate(0.0f);

            if (input.dataType == DataType::FLOAT16) {
                int len = input.Count(0);
                vInputs.resize(len);
                for (int i = 0; i < len; i++) {
                    vInputs[i] = fp16tofp32.dict[((uint16_t*)input.cpuData)[i]];
                }
                floatInput = vInputs.data();
            }
            if (logits.dataType == DataType::FLOAT16) {
                int len = logits.Count(0);
                vLogits.resize(len);
                for (int i = 0; i < len; i++) {
                    vLogits[i] = fp16tofp32.dict[((uint16_t*)logits.cpuData)[i]];
                }
                floatLogits = vLogits.data();
            }
            for (int o = 0; o < outer; o++) {
                std::vector <std::pair <float, int> > oriV;
                oriV.resize(channels);
                for (int j = 0; j < channels; j++) {
                    oriV[j].first = -floatLogits[o * channels + j];
                    oriV[j].second = j;
                }
                if (gateBias.dims.size() > 0) {
                    if (gateBias.dataType != DataType::FLOAT32) {
                        ToDataType(gateBias, DataType::FLOAT32);
                    }
                    float *cpuBias = (float*)gateBias.cpuData;
                    for (int i = 0; i < channels; i++) {
                        oriV[i].first -= cpuBias[i];
                    }
                }
// record.push_back(std::make_pair("very first", GetSpan(ttt, std::chrono::system_clock::now())));
                // sort(oriV.begin(), oriV.end());
                std::partial_sort(oriV.begin(), oriV.begin() + topk, oriV.end());
                // std::nth_element(oriV.begin(), oriV.begin() + topk, oriV.end());
// record.push_back(std::make_pair("sort", GetSpan(ttt, std::chrono::system_clock::now())));
                float sum = 1.0;
                if (needNorm) {
                    sum = 0.0;
                    for (int j = 0; j < topk; j++) {
                        sum += floatLogits[o * channels + oriV[j].second];
                    }
                }

                std::vector <std::pair <int, float> > v;
                for (int j = 0; j < topk; j++) {
                    v.push_back(std::make_pair(oriV[j].second + 1, floatLogits[o * channels + oriV[j].second] / sum * routeScale));
                }
                if (weights[0] != nullptr) {
                    v.push_back(std::make_pair(0, sharedScale));
                }
                int n = input.dims[0], m = input.dims[1];
                int group = weights[2]->group, groupCnt = weights[2]->groupCnt;
                if (weights[2]->dataType != DataType::INT4_GROUP) {
                    group = 1;
                    groupCnt = m;
                }
                float *inputData = floatInput + o * m;

                std::vector<LowBitConfig> &inputConfigs = moeIntSingleVarManager.inputConfigs;
                std::vector<uint8_t> &uinput = moeIntSingleVarManager.uinput;
                std::vector <float> &inputSums = moeIntSingleVarManager.inputSums;
                std::vector <float> &iscales = moeIntSingleVarManager.iscales;
                std::vector <float> &izeros = moeIntSingleVarManager.izeros;
// record.push_back(std::make_pair("before OnlineQuantization", GetSpan(ttt, std::chrono::system_clock::now())));
                OnlineQuantization(inputData, uinput, inputConfigs, 1, m, group, groupCnt, 
                                    inputSums, iscales, izeros, permuteType);
// record.push_back(std::make_pair("OnlineQuantization", GetSpan(ttt, std::chrono::system_clock::now())));
                std::vector <std::vector <float> > &middles = moeIntSingleVarManager.middles;
                std::vector <std::vector <float> > &results = moeIntSingleVarManager.results;
                middles.resize(v.size());
                results.resize(v.size());
                for (int j = 0; j < v.size(); j++) {
                    int idx = v[j].first;
                    weights[idx * 2]->CalcWeightSum();
                    weights[idx * 2 + 1]->CalcWeightSum();
                }
                for (int j = 0; j < v.size(); j++) {
                    int idx = v[j].first;
                    middles[j].resize(weights[idx * 2]->dims[0]);
                    results[j].resize(weights[idx * 2 + 1]->dims[0]);
                }
                std::vector<fastllm::MultiThreadBaseOp*> ops;
                auto *pool = GetAlivePool();
                int threads = pool->threads.size();
                ops.resize(threads);

                std::vector <std::vector <LowBitConfig> > &inputConfigsDown = moeIntSingleVarManager.inputConfigsDown;
                std::vector <std::vector <uint8_t> > &uinputsDown = moeIntSingleVarManager.uinputsDown;
                std::vector <std::vector <float> > &inputSumsDown = moeIntSingleVarManager.inputSumsDown;
                std::vector <std::vector <float> > &iscalesDown = moeIntSingleVarManager.iscalesDown;
                std::vector <std::vector <float> > &izerosDown = moeIntSingleVarManager.izerosDown;
                inputConfigsDown.resize(v.size());
                uinputsDown.resize(v.size());
                inputSumsDown.resize(v.size());
                iscalesDown.resize(v.size());
                izerosDown.resize(v.size());
 // record.push_back(std::make_pair("prepare", GetSpan(ttt, std::chrono::system_clock::now())));
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
// float xxx = 0;
                    for (int l = st; l <= end; l++) {
                        int idx = v[l].first;
                        Data *weight = weights[idx * 2];
                        uint8_t *weightData = (uint8_t *) weight->cpuData;
                        float *outputData = middles[l].data();
                        float *biasData = nullptr;
                        int curK = weight->dims[0];
                        int curThread = (curK / k) * base;
// xxx += m * curK;
                        if (weight->dataType == DataType::INT8) {
                            LaunchLinearInt8Int8(uinput.data(), weightData, outputData, 1, m, curK,
                                                weight->weightSum.data(), weight->zeros.data(), weight->scales.data(), biasData, 
                                                inputSums.data(), iscales.data(), izeros.data(), 
                                                ops, pool, threadSt, curThread);
                        } else {
                            MultiplyInt4GroupMultiThreadLaunch(uinput.data(), weightData, outputData, 1, m, curK,
                                                weight->weightSum.data(), weight->mins.data(), weight->scales.data(), biasData, 
                                                inputSums, iscales, izeros,
                                                inputConfigs, threadSt, curThread, group, groupCnt, ops, pool);
                        }
                        threadSt += curThread;
                    }
                    for (int j = 0; j < ops.size(); j++) {
                        pool->Wait(j);
                        delete ops[j];
                    }
// record.push_back(std::make_pair("mul0", GetSpan(ttt, std::chrono::system_clock::now())));
// float spend = record.back().second - record[record.size() - 2].second;
//printf("speed = %f gops.\n", xxx / spend / 1e9);
                    // swiglu
                    threadSt = 0;
                    for (int l = st; l <= end; l++) {
                        int idx = v[l].first;
                        int spatial = weights[idx * 2]->dims[0], mid = spatial / 2;
                        float *outputData = middles[l].data();
                        int curK = weights[idx * 2]->dims[0];
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
                                    1, mid, groupDown, groupCntDown,
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
 // record.push_back(std::make_pair("mul1", GetSpan(ttt, std::chrono::system_clock::now())));
                    st = end;
                }
// record.push_back(std::make_pair("finish", GetSpan(ttt, std::chrono::system_clock::now())));
                float *fLastOutput = ((float*)output.cpuData) + o * m;
                std::vector <float> tempOutput;
                if (output.dataType == DataType::FLOAT16) {
                    tempOutput.resize(m, 0);
                    fLastOutput = tempOutput.data();
                }
/*
                std::vector <float> vv;
                vv.resize(v.size());
                for (int i = 0; i < v.size(); i++) {
                    vv[i] = v[i].second;
                }
                RunMultiThreadReduce (
                    (int)vv.size(), results.data(), vv.data(), fLastOutput, 
                    nullptr, m, pool
                );
*/

                for (int j = 0; j < v.size(); j++) {
                    float value = v[j].second;
                    float *curOutput = (float*)results[j].data();
                    int i = 0;
#ifdef __AVX2__
                    __m256 value_vec = _mm256_set1_ps(value);

                    // 每次处理 8 个浮点数（AVX2 寄存器可以容纳 8 个 float）
                    for (; i <= m - 8; i += 8) {
                        // 加载 curOutput 的 8 个浮点数
                        __m256 curOutput_vec = _mm256_loadu_ps(&curOutput[i]);

                        // 加载 fLastOutput 的 8 个浮点数
                        __m256 fLastOutput_vec = _mm256_loadu_ps(&fLastOutput[i]);

                        // 计算 curOutput * value
                        __m256 result_vec = _mm256_mul_ps(curOutput_vec, value_vec);

                        // 累加到 fLastOutput
                        fLastOutput_vec = _mm256_add_ps(fLastOutput_vec, result_vec);

                        // 将结果存回 fLastOutput
                        _mm256_storeu_ps(&fLastOutput[i], fLastOutput_vec);
                    }
#endif
                    // 处理剩余的不足 8 个的元素
                    for (; i < m; i++) {
                        fLastOutput[i] += curOutput[i] * value;
                    }
                }
 // record.push_back(std::make_pair("get f32 output", GetSpan(ttt, std::chrono::system_clock::now())));
                if (output.dataType == DataType::FLOAT16) {
                    Float32ToFloat16(tempOutput.data(), ((uint16_t*)output.cpuData) + o * m, m);
                }
// record.push_back(std::make_pair("finish output", GetSpan(ttt, std::chrono::system_clock::now())));
// for (int i = 0; i < record.size(); i++) {
    //printf("%s spend %f s.\n", record[i].first.c_str(), record[i].second);
// }
            }
        } else if ((input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16) && 
                (weights[2]->dataType == DataType::DATA_GGUF_FORMAT) &&
            input.dims[0] < 32) {
            int dimsLen = logits.dims.size();
            int outer = logits.Count(0) / logits.Count(dimsLen - 1);
            int channels = logits.dims[dimsLen - 1];

            std::vector <float> vLogits, vInputs;
            float *floatLogits = ((float*)logits.cpuData);
            float *floatInput = (float*)input.cpuData;
            output.Allocate(0.0f);

            if (input.dataType == DataType::FLOAT16) {
                int len = input.Count(0);
                vInputs.resize(len);
                for (int i = 0; i < len; i++) {
                    vInputs[i] = fp16tofp32.dict[((uint16_t*)input.cpuData)[i]];
                }
                floatInput = vInputs.data();
            }
            if (logits.dataType == DataType::FLOAT16) {
                int len = logits.Count(0);
                vLogits.resize(len);
                for (int i = 0; i < len; i++) {
                    vLogits[i] = fp16tofp32.dict[((uint16_t*)logits.cpuData)[i]];
                }
                floatLogits = vLogits.data();
            }
            for (int o = 0; o < outer; o++) {
                std::vector <std::pair <float, int> > oriV;
                oriV.resize(channels);
                for (int j = 0; j < channels; j++) {
                    oriV[j].first = -floatLogits[o * channels + j];
                    oriV[j].second = j;
                }
                if (gateBias.dims.size() > 0) {
                    if (gateBias.dataType != DataType::FLOAT32) {
                        ToDataType(gateBias, DataType::FLOAT32);
                    }
                    float *cpuBias = (float*)gateBias.cpuData;
                    for (int i = 0; i < channels; i++) {
                        oriV[i].first -= cpuBias[i];
                    }
                }
// record.push_back(std::make_pair("very first", GetSpan(ttt, std::chrono::system_clock::now())));
                // sort(oriV.begin(), oriV.end());
                std::partial_sort(oriV.begin(), oriV.begin() + topk, oriV.end());
                // std::nth_element(oriV.begin(), oriV.begin() + topk, oriV.end());
// record.push_back(std::make_pair("sort", GetSpan(ttt, std::chrono::system_clock::now())));
                float sum = 1.0;
                if (needNorm) {
                    sum = 0.0;
                    for (int j = 0; j < topk; j++) {
                        sum += floatLogits[o * channels + oriV[j].second];
                    }
                }

                std::vector <std::pair <int, float> > v;
                for (int j = 0; j < topk; j++) {
                    v.push_back(std::make_pair(oriV[j].second + 1, floatLogits[o * channels + oriV[j].second] / sum * routeScale));
                }
                if (weights[0] != nullptr) {
                    v.push_back(std::make_pair(0, sharedScale));
                }
                int n = input.dims[0], m = input.dims[1];
                float *inputData = floatInput + o * m;

                std::vector <uint8_t> &q8kInputs = moeIntSingleVarManager.uinput;
                int rowCount = m / QK_K; // 每行有多少个block
                q8kInputs.resize(ggml_row_size(ggml_type_vec_dot_type((ggml_type)weights[2]->ggmlType), m));
                iqk_quantize_row_q8_K (
                    inputData, q8kInputs.data(), m, 
                    ggml_type_vec_dot_type((ggml_type)weights[2]->ggmlType),
                    (ggml_type)weights[2]->ggmlType
                );

                std::vector <std::vector <float> > &middles = moeIntSingleVarManager.middles;
                std::vector <std::vector <float> > &results = moeIntSingleVarManager.results;
                middles.resize(v.size());
                results.resize(v.size());
                for (int j = 0; j < v.size(); j++) {
                    int idx = v[j].first;
                    middles[j].resize(weights[idx * 2]->dims[0]);
                    results[j].resize(weights[idx * 2 + 1]->dims[0]);
                }
                std::vector<fastllm::MultiThreadBaseOp*> ops;
                auto *pool = GetAlivePool();
                int threads = pool->threads.size();
                ops.resize(threads);

                std::vector <std::vector <uint8_t> > &q8kInputsDown = moeIntSingleVarManager.uinputsDown;
                q8kInputsDown.resize(v.size());

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
                        float *outputData = middles[l].data();
                        float *biasData = nullptr;
                        int curK = weight->dims[0];
                        int curThread = (curK / k) * base;
                        
                        LaunchLinearQ8KGGUF(q8kInputs.data(), weightData, outputData, biasData, weight, 
                            1, m, curK, ops, pool, threadSt, curThread);
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
                        float *outputData = middles[l].data();
                        int curK = weights[idx * 2]->dims[0];
                        auto &uinputDown = q8kInputsDown[l];
                        int rowCount = mid / QK_K; // 每行有多少个block
                        uinputDown.resize(ggml_row_size(ggml_type_vec_dot_type((ggml_type)weights[idx * 2 + 1]->ggmlType), m));

                        ops[l - st] = new fastllm::MultiThreadMultiOps();
                        ((fastllm::MultiThreadMultiOps*)ops[l - st])->ops.push_back(new fastllm::MultiThreadSwigluOp(outputData, mid, mid, outputData, 1, spatial, spatial)); 
                        ((fastllm::MultiThreadMultiOps*)ops[l - st])->ops.push_back(new fastllm::MultiThreadFloat32ToQ8KOp(middles[l].data(), (uint8_t*)uinputDown.data(), mid, (ggml_type)weights[idx * 2 + 1]->ggmlType));
                        /* ((fastllm::MultiThreadMultiOps*)ops[l - st])->ops.push_back(new MultiThreadOnlineQuantizationOp(
                                    middles[l].data(), uinputDown.data(), inputConfigs.data(),
                                    1, mid, groupDown, groupCntDown,
                                    inputSums.data(), iscales.data(), izeros.data(), permuteType)); */
                        pool->PushOp(l - st, ops[l - st]);
                    }
                    for (int l = st; l <= end; l++) {
                        pool->Wait(l - st);
                        delete ops[l - st];
                    }
/*
                    for (int l = st; l <= end; l++) {
                        int idx = v[l].first;
                        int spatial = weights[idx * 2]->dims[0], mid = spatial / 2;
                        
                        auto &uinputDown = q8kInputsDown[l];
                        int rowCount = mid / QK_K; // 每行有多少个block

                        uinputDown.resize(ggml_row_size(ggml_type_vec_dot_type((ggml_type)weights[idx * 2 + 1]->ggmlType), m));
                        iqk_quantize_row_q8_K (
                                middles[l].data(), uinputDown.data(), mid, 
                                ggml_type_vec_dot_type((ggml_type)weights[idx * 2 + 1]->ggmlType),
                                (ggml_type)weights[idx * 2 + 1]->ggmlType
                        );
                    }
*/
                    threadSt = 0;
                    for (int l = st; l <= end; l++) {
                        int idx = v[l].first;
                        int mid = weights[idx * 2]->dims[0] / 2;
                        int curK = weights[idx * 2]->dims[0];
                        Data *weightDown = weights[idx * 2 + 1];

                        auto &uinputDown = q8kInputsDown[l];
                        int curThread = (curK / k) * base;

                        LaunchLinearQ8KGGUF(uinputDown.data(), weightDown->cpuData, results[l].data(), nullptr, weightDown, 
                            1, mid, m, ops, pool, threadSt, curThread);
                        threadSt += curThread;               
                    }

                    for (int j = 0; j < ops.size(); j++) {
                        pool->Wait(j);
                        delete ops[j];
                    }
                    st = end;
                }

                float *fLastOutput = ((float*)output.cpuData) + o * m;
                std::vector <float> tempOutput;
                if (output.dataType == DataType::FLOAT16) {
                    tempOutput.resize(m, 0);
                    fLastOutput = tempOutput.data();
                }
                
                for (int j = 0; j < v.size(); j++) {
                    float value = v[j].second;
                    float *curOutput = (float*)results[j].data();
                    int i = 0;
#ifdef __AVX2__
                    __m256 value_vec = _mm256_set1_ps(value);

                    // 每次处理 8 个浮点数（AVX2 寄存器可以容纳 8 个 float）
                    for (; i <= m - 8; i += 8) {
                        // 加载 curOutput 的 8 个浮点数
                        __m256 curOutput_vec = _mm256_loadu_ps(&curOutput[i]);

                        // 加载 fLastOutput 的 8 个浮点数
                        __m256 fLastOutput_vec = _mm256_loadu_ps(&fLastOutput[i]);

                        // 计算 curOutput * value
                        __m256 result_vec = _mm256_mul_ps(curOutput_vec, value_vec);

                        // 累加到 fLastOutput
                        fLastOutput_vec = _mm256_add_ps(fLastOutput_vec, result_vec);

                        // 将结果存回 fLastOutput
                        _mm256_storeu_ps(&fLastOutput[i], fLastOutput_vec);
                    }
#endif
                    // 处理剩余的不足 8 个的元素
                    for (; i < m; i++) {
                        fLastOutput[i] += curOutput[i] * value;
                    }
                }
 // record.push_back(std::make_pair("get f32 output", GetSpan(ttt, std::chrono::system_clock::now())));
                if (output.dataType == DataType::FLOAT16) {
                    Float32ToFloat16(tempOutput.data(), ((uint16_t*)output.cpuData) + o * m, m);
                }
// record.push_back(std::make_pair("finish output", GetSpan(ttt, std::chrono::system_clock::now())));
// for (int i = 0; i < record.size(); i++) {
    //printf("%s spend %f s.\n", record[i].first.c_str(), record[i].second);
// }
            }
        } else if ((input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16) && 
                (weights[2]->dataType == DataType::FLOAT16) &&
                input.dims[0] < 32) {
            int dimsLen = logits.dims.size();
            int outer = logits.Count(0) / logits.Count(dimsLen - 1);
            int channels = logits.dims[dimsLen - 1];

            std::vector <float> vLogits, vInputs;
            float *floatLogits = ((float*)logits.cpuData);
            float *floatInput = (float*)input.cpuData;
            output.Allocate(0.0f);

            if (input.dataType == DataType::FLOAT16) {
                int len = input.Count(0);
                vInputs.resize(len);
                for (int i = 0; i < len; i++) {
                    vInputs[i] = fp16tofp32.dict[((uint16_t*)input.cpuData)[i]];
                }
                floatInput = vInputs.data();
            }
            if (logits.dataType == DataType::FLOAT16) {
                int len = logits.Count(0);
                vLogits.resize(len);
                for (int i = 0; i < len; i++) {
                    vLogits[i] = fp16tofp32.dict[((uint16_t*)logits.cpuData)[i]];
                }
                floatLogits = vLogits.data();
            }
            for (int o = 0; o < outer; o++) {
                std::vector <std::pair <float, int> > oriV;
                oriV.resize(channels);
                for (int j = 0; j < channels; j++) {
                    oriV[j].first = -floatLogits[o * channels + j];
                    oriV[j].second = j;
                }
                if (gateBias.dims.size() > 0) {
                    if (gateBias.dataType != DataType::FLOAT32) {
                        ToDataType(gateBias, DataType::FLOAT32);
                    }
                    float *cpuBias = (float*)gateBias.cpuData;
                    for (int i = 0; i < channels; i++) {
                        oriV[i].first -= cpuBias[i];
                    }
                }
                std::partial_sort(oriV.begin(), oriV.begin() + topk, oriV.end());
                float sum = 1.0;
                if (needNorm) {
                    sum = 0.0;
                    for (int j = 0; j < topk; j++) {
                        sum += floatLogits[o * channels + oriV[j].second];
                    }
                }

                std::vector <std::pair <int, float> > v;
                for (int j = 0; j < topk; j++) {
                    v.push_back(std::make_pair(oriV[j].second + 1, floatLogits[o * channels + oriV[j].second] / sum * routeScale));
                }
                if (weights[0] != nullptr) {
                    v.push_back(std::make_pair(0, sharedScale));
                }
                int n = input.dims[0], m = input.dims[1];
                float *inputData = floatInput + o * m;
            
                auto &middles = moeFloatSingleVarManager.middles;
                auto &results = moeFloatSingleVarManager.results;
                middles.resize(v.size());
                results.resize(v.size());
                for (int j = 0; j < v.size(); j++) {
                    int idx = v[j].first;
                    middles[j].resize(weights[idx * 2]->dims[0]);
                    results[j].resize(weights[idx * 2 + 1]->dims[0]);
                }
                std::vector<fastllm::MultiThreadBaseOp*> ops;
                auto *pool = GetAlivePool();
                int threads = pool->threads.size();
                ops.resize(threads);

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
                        float *outputData = middles[l].data();
                        float *biasData = nullptr;
                        int curK = weight->dims[0];
                        int curThread = (curK / k) * base;
                        LaunchLinearFloat32Float16(inputData, *weight, outputData, biasData, 1, m, curK, ops, pool, threadSt, curThread);
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
                        float *outputData = middles[l].data();
                        int curK = weights[idx * 2]->dims[0];
                        ops[l - st] = new fastllm::MultiThreadMultiOps();
                        ((fastllm::MultiThreadMultiOps*)ops[l - st])->ops.push_back(new fastllm::MultiThreadSwigluOp(outputData, mid, mid, outputData, 1, spatial, spatial));
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
                        int curThread = (curK / k) * base;
                        LaunchLinearFloat32Float16((float*)middles[l].data(), *weightDown, results[l].data(), nullptr, 1, mid, m, ops, pool, threadSt, curThread);
                        threadSt += curThread;               
                    }

                    for (int j = 0; j < ops.size(); j++) {
                        pool->Wait(j);
                        delete ops[j];
                    }
                    st = end;
                }
                float *fLastOutput = ((float*)output.cpuData) + o * m;
                std::vector <float> tempOutput;
                if (output.dataType == DataType::FLOAT16) {
                    tempOutput.resize(m, 0);
                    fLastOutput = tempOutput.data();
                }
                for (int j = 0; j < v.size(); j++) {
                    float value = v[j].second;
                    float *curOutput = (float*)results[j].data();
                    int i = 0;
#ifdef __AVX2__
                    __m256 value_vec = _mm256_set1_ps(value);

                    // 每次处理 8 个浮点数（AVX2 寄存器可以容纳 8 个 float）
                    for (; i <= m - 8; i += 8) {
                        // 加载 curOutput 的 8 个浮点数
                        __m256 curOutput_vec = _mm256_loadu_ps(&curOutput[i]);

                        // 加载 fLastOutput 的 8 个浮点数
                        __m256 fLastOutput_vec = _mm256_loadu_ps(&fLastOutput[i]);

                        // 计算 curOutput * value
                        __m256 result_vec = _mm256_mul_ps(curOutput_vec, value_vec);

                        // 累加到 fLastOutput
                        fLastOutput_vec = _mm256_add_ps(fLastOutput_vec, result_vec);

                        // 将结果存回 fLastOutput
                        _mm256_storeu_ps(&fLastOutput[i], fLastOutput_vec);
                    }
#endif
                    // 处理剩余的不足 8 个的元素
                    for (; i < m; i++) {
                        fLastOutput[i] += curOutput[i] * value;
                    }
                }
                if (output.dataType == DataType::FLOAT16) {
                    Float32ToFloat16(tempOutput.data(), ((uint16_t*)output.cpuData) + o * m, m);
                }
            }
        } else if ((input.dataType == DataType::FLOAT32) && 
                (weights[2]->dataType == DataType::FP8_E4M3 ||
                 weights[2]->dataType == DataType::BFLOAT16) &&
                input.dims[0] < 32) {
            int dimsLen = logits.dims.size();
            int outer = logits.Count(0) / logits.Count(dimsLen - 1);
            int channels = logits.dims[dimsLen - 1];

            std::vector <float> vLogits, vInputs;
            float *floatLogits = ((float*)logits.cpuData);
            float *floatInput = (float*)input.cpuData;
            output.Allocate(0.0f);

            if (input.dataType == DataType::FLOAT16) {
                int len = input.Count(0);
                vInputs.resize(len);
                for (int i = 0; i < len; i++) {
                    vInputs[i] = fp16tofp32.dict[((uint16_t*)input.cpuData)[i]];
                }
                floatInput = vInputs.data();
            }
            if (logits.dataType == DataType::FLOAT16) {
                int len = logits.Count(0);
                vLogits.resize(len);
                for (int i = 0; i < len; i++) {
                    vLogits[i] = fp16tofp32.dict[((uint16_t*)logits.cpuData)[i]];
                }
                floatLogits = vLogits.data();
            }
            for (int o = 0; o < outer; o++) {
                std::vector <std::pair <float, int> > oriV;
                oriV.resize(channels);
                for (int j = 0; j < channels; j++) {
                    oriV[j].first = -floatLogits[o * channels + j];
                    oriV[j].second = j;
                }
                if (gateBias.dims.size() > 0) {
                    if (gateBias.dataType != DataType::FLOAT32) {
                        ToDataType(gateBias, DataType::FLOAT32);
                    }
                    float *cpuBias = (float*)gateBias.cpuData;
                    for (int i = 0; i < channels; i++) {
                        oriV[i].first -= cpuBias[i];
                    }
                }
                std::partial_sort(oriV.begin(), oriV.begin() + topk, oriV.end());
                float sum = 1.0;
                if (needNorm) {
                    sum = 0.0;
                    for (int j = 0; j < topk; j++) {
                        sum += floatLogits[o * channels + oriV[j].second];
                    }
                }

                std::vector <std::pair <int, float> > v;
                for (int j = 0; j < topk; j++) {
                    v.push_back(std::make_pair(oriV[j].second + 1, floatLogits[o * channels + oriV[j].second] / sum * routeScale));
                }
                if (weights[0] != nullptr) {
                    v.push_back(std::make_pair(0, sharedScale));
                }
                int n = input.dims[0], m = input.dims[1];
                float *inputData = floatInput + o * m;
                auto &bf16Input = moeFloatSingleVarManager.bf16Input;
                bf16Input.resize(m);
                Float32ToBFloat16(inputData, bf16Input.data(), m);
                
                auto &middles = moeFloatSingleVarManager.middles;
                auto &results = moeFloatSingleVarManager.results;
                middles.resize(v.size());
                results.resize(v.size());
                for (int j = 0; j < v.size(); j++) {
                    int idx = v[j].first;
                    middles[j].resize(weights[idx * 2]->dims[0]);
                    results[j].resize(weights[idx * 2 + 1]->dims[0]);
                }
                std::vector<fastllm::MultiThreadBaseOp*> ops;
                auto *pool = GetAlivePool();
                int threads = pool->threads.size();
                ops.resize(threads);

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
                        float *outputData = middles[l].data();
                        float *biasData = nullptr;
                        int curK = weight->dims[0];
                        int curThread = (curK / k) * base;
                        if (weight->dataType == DataType::FP8_E4M3) {                            
                            LaunchLinearBFloat16FP8E4M3(bf16Input.data(), *weight, outputData, biasData, 1, m, curK, ops, pool, threadSt, curThread);
                        } if (weight->dataType == DataType::BFLOAT16) {
                            LaunchLinearBFloat16BFloat16(bf16Input.data(), *weight, outputData, biasData, 1, m, curK, ops, pool, threadSt, curThread);
                        } else {
                            // TODO: other
                        }
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
                        float *outputData = middles[l].data();
                        int curK = weights[idx * 2]->dims[0];
                        ops[l - st] = new fastllm::MultiThreadMultiOps();
                        ((fastllm::MultiThreadMultiOps*)ops[l - st])->ops.push_back(new fastllm::MultiThreadSwigluOp(outputData, mid, mid, outputData, 1, spatial, spatial));
                        ((fastllm::MultiThreadMultiOps*)ops[l - st])->ops.push_back(new fastllm::MultiThreadFloat32ToBFloat16Op(middles[l].data(), (uint16_t*)middles[l].data(), mid));
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
                        int curThread = (curK / k) * base;
                        if (weightDown->dataType == DataType::FP8_E4M3) {
                            LaunchLinearBFloat16FP8E4M3((uint16_t*)middles[l].data(), *weightDown, results[l].data(), nullptr, 1, mid, m, ops, pool, threadSt, curThread);
                        } else if (weightDown->dataType == DataType::BFLOAT16) {
                            LaunchLinearBFloat16BFloat16((uint16_t*)middles[l].data(), *weightDown, results[l].data(), nullptr, 1, mid, m, ops, pool, threadSt, curThread);
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
                }
                float *fLastOutput = ((float*)output.cpuData) + o * m;
                std::vector <float> tempOutput;
                if (output.dataType == DataType::FLOAT16) {
                    tempOutput.resize(m, 0);
                    fLastOutput = tempOutput.data();
                }
                for (int j = 0; j < v.size(); j++) {
                    float value = v[j].second;
                    float *curOutput = (float*)results[j].data();
                    int i = 0;
#ifdef __AVX2__
                    __m256 value_vec = _mm256_set1_ps(value);

                    // 每次处理 8 个浮点数（AVX2 寄存器可以容纳 8 个 float）
                    for (; i <= m - 8; i += 8) {
                        // 加载 curOutput 的 8 个浮点数
                        __m256 curOutput_vec = _mm256_loadu_ps(&curOutput[i]);

                        // 加载 fLastOutput 的 8 个浮点数
                        __m256 fLastOutput_vec = _mm256_loadu_ps(&fLastOutput[i]);

                        // 计算 curOutput * value
                        __m256 result_vec = _mm256_mul_ps(curOutput_vec, value_vec);

                        // 累加到 fLastOutput
                        fLastOutput_vec = _mm256_add_ps(fLastOutput_vec, result_vec);

                        // 将结果存回 fLastOutput
                        _mm256_storeu_ps(&fLastOutput[i], fLastOutput_vec);
                    }
#endif
                    // 处理剩余的不足 8 个的元素
                    for (; i < m; i++) {
                        fLastOutput[i] += curOutput[i] * value;
                    }
                }
                if (output.dataType == DataType::FLOAT16) {
                    Float32ToFloat16(tempOutput.data(), ((uint16_t*)output.cpuData) + o * m, m);
                }
            }
        } else if (input.dataType == DataType::FLOAT32 && output.dataType == DataType::FLOAT32
                && weights[2]->dataType == DataType::BFLOAT16) {
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
//printf("prepare spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                DataType startDataType = DataType::BFLOAT16;
                DataType downInputDataType = DataType::BFLOAT16;

                // 从 fastllmMoeDataManager 获取缓存的 vector，并根据需要调整大小
                auto& realInput = fastllmMoeDataManager.realInput;
                auto& expandInput = fastllmMoeDataManager.expandInput;
                auto& gateUpOutput = fastllmMoeDataManager.gateUpOutput;
                auto& swigluOutput = fastllmMoeDataManager.swigluOutput;
                auto& downInput = fastllmMoeDataManager.downInput;
                auto& downOutput = fastllmMoeDataManager.downOutput;
                auto& reduceOutput = fastllmMoeDataManager.reduceOutput;

                // 计算所需大小
                size_t realInputSize = GetDataBytes(startDataType, bs, inputDim);
                size_t expandInputSize = GetDataBytes(startDataType, totalLines, inputDim);
                size_t gateUpOutputSize = totalLines * interDim * 2;
                size_t swigluOutputSize = totalLines * interDim;
                size_t downInputSize = GetDataBytes(downInputDataType, totalLines, outputDim);
                size_t downOutputSize = totalLines * outputDim;
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

//printf("malloc spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                // 0. input -> realInput
                RunMultiThreadConvertFromFloat32(realInput.data(), DataType::BFLOAT16, (float*)input.cpuData, bs, inputDim, GetAlivePool());
//printf("Float32ToBFloat16 spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));

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
//printf("expand spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                // 2. gateUp
                {
long long ops = 0;
                    int offset = 0;
                    int stride = 64;
                    std::vector<MultiThreadBaseOp*> gemmOps;
                    for (int e = 0; e < expertTasks.size(); e++) {
                        if (weights[e * 2] != nullptr && expertTasks[e].size() > 0) {
                            int lines = expertTasks[e].size();

                            // Prepare input pointer for this expert's batch
                            uint16_t* expertInputPtr = (uint16_t*)(expandInput.data() + offset * GetDataBytes(startDataType, 1, inputDim));
                            
                            // Prepare output pointer for this expert's batch
                            float* expertGateUpOutputPtr = gateUpOutput.data() + offset * interDim * 2;
                            
                            // Get weight data (assuming weights are stored as BFloat16)
                            uint16_t* weightPtr = (uint16_t*)(weights[e * 2]->cpuData);

                            for (int st = 0; st < interDim * 2; st += stride) {
                                int end = std::min(st + stride, interDim * 2);
                                gemmOps.push_back(new MultiThreadGemmOp(
                                    (uint8_t*)expertInputPtr, DataType::BFLOAT16,
                                    (uint8_t*)weightPtr, DataType::BFLOAT16,
                                    (uint8_t*)expertGateUpOutputPtr, DataType::FLOAT32,
                                    lines, inputDim, interDim * 2, st, end
                                ));
                            }
ops += (long long)lines * inputDim * interDim * 2;
                            offset += lines;
                        }
                    }
                    DynamicScheduleTasks(gemmOps);
//printf("ops = %f g\n", (float)ops / 1e9);
                }
//printf("gateup spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));

                // 3. swiglu
                SwigluMultiThread((float *) gateUpOutput.data(), interDim, interDim, ((float *) swigluOutput.data()),
                                    totalLines, interDim * 2, interDim, GetAlivePool());
//printf("swiglu spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));

                // 4. swigluOutput -> downInput
                RunMultiThreadConvertFromFloat32(downInput.data(), DataType::BFLOAT16, (float*)swigluOutput.data(), totalLines, interDim, GetAlivePool());

//printf("Float32ToBFloat16 spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                // 5. down
                {
                    int offset = 0;
                    int stride = 64;
                    std::vector <MultiThreadBaseOp*> gemmOps;
                    for (int e = 0; e < expertTasks.size(); e++) {
                        if (weights[e * 2 + 1] != nullptr && expertTasks[e].size() > 0) {
                            int lines = expertTasks[e].size();
                            
                            // Prepare input pointer for this expert's batch
                            uint16_t* expertDownInputPtr = (uint16_t*)(downInput.data() + offset * GetDataBytes(downInputDataType, 1, interDim));
                            
                            // Prepare output pointer for this expert's batch
                            float* expertDownOutputPtr = downOutput.data() + offset * dim;
                            
                            // Get weight data (assuming weights are stored as BFloat16)
                            uint16_t* weightPtr = (uint16_t*)(weights[e * 2 + 1]->cpuData);

                            for (int st = 0; st < dim; st += stride) {
                                int end = std::min(st + stride, dim);
                                gemmOps.push_back(new MultiThreadGemmOp (
                                    (uint8_t*)expertDownInputPtr, DataType::BFLOAT16, 
                                    (uint8_t*)weightPtr, DataType::BFLOAT16, 
                                    (uint8_t*)expertDownOutputPtr, DataType::FLOAT32, 
                                    lines, interDim, dim, st, end
                                ));
                            }
                            offset += lines;
                        }
                    }
                    DynamicScheduleTasks(gemmOps);
                }

//printf("down spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
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
//printf("reduce spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
                // 7. reduceOutput -> last Output
                if (output.dataType != DataType::FLOAT32) {
                    if (output.dataType == DataType::FLOAT16) {
                        Float32ToFloat16(reduceOutput.data(), (uint16_t*)output.cpuData, output.Count(0));
                    }
                }
//printf("last spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
            }
        } else {
  // auto st = std::chrono::system_clock::now();
  // auto veryst = std::chrono::system_clock::now();
  // std::map <std::string, float> cnt;
            // normal
            Data gate, attenPart, moePart;
            ToDataType(logits, DataType::FLOAT32);
            logits.ToDevice(DataDevice::CPU);
            float *cpuRouterLogits = (float*)logits.cpuData;
            int m = logits.dims.back();

            if (input.dims[0] == 1) {
                std::vector <std::pair <float, int> > v; // (value, idx)
                for (int i = 0; i < m; i++) {
                    v.push_back(std::make_pair(-cpuRouterLogits[i], i));
                }
                if (gateBias.dims.size() > 0) {
                    ToDataType(gateBias, DataType::FLOAT32);
                    gateBias.ToDevice(DataDevice::CPU);
                    float *cpuBias = (float*)gateBias.cpuData;
                    for (int i = 0; i < m; i++) {
                        v[i].first -= cpuBias[i];
                    }
                }
                sort(v.begin(), v.end());
                float sum = 1.0;
                if (needNorm) {
                    sum = 0.0;
                    for (int j = 0; j < topk; j++) {
                        sum += cpuRouterLogits[v[j].second];
                    }
                }

                output.Allocate(0.0f);
                for (int j = 0; j < topk; j++) {
                    int idx = v[j].second;
                    float value = cpuRouterLogits[idx] / sum * routeScale;

                    Linear(input, *weights[(idx + 1) * 2], Data(), w3);
                    Swiglu(w3, w1);
                    Linear(w1, *weights[(idx + 1) * 2 + 1], Data(), w2);
                    AddTo(output, w2, value);
                }

                if (weights[0] != nullptr) {
                    Linear(input, *weights[0], Data(), w3);
                    Swiglu(w3, w1);
                    Linear(w1, *weights[1], Data(), w2);
                    AddTo(output, w2, sharedScale);
                }
            } else {
                int bs = input.dims[0], dim = output.dims[1];
                int inputDim = input.dims[1];
                std::vector <float> tempResult, middleResult;
                tempResult.resize(bs * dim, 0.0f);
                middleResult.resize(bs * dim, 0.0f);
                std::vector <std::vector <std::pair <int, float> > > expertTasks; // expertTasks[i]代表专家i的task, expertTasks[i][j] = (第j个任务对应的行数， 权重)
                expertTasks.resize(m + 1);
                Data &tempInput = w2;
                tempInput.ToDevice(input.dataDevice);
                tempInput.Resize(input.dims);
  // cnt["prepare 0"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
                tempInput.Allocate();
  // cnt["allocate"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
                std::vector <std::pair <float, int> > v; // (value, idx)
                v.resize(m);
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
  // cnt["prepare"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
                for (int e = 0; e < expertTasks.size(); e++) {
                    auto &task = expertTasks[e];
                    if (task.size() == 0) {
                        continue;
                    }
                    if (weights[e * 2] == nullptr) {
                        continue;
                    }

                    tempInput.Resize({(int)task.size(), inputDim});
                    tempInput.Allocate();

                    std::vector <MultiThreadMemcpyMultiLinesTask> memcpyTasks;
                    for (int i = 0; i < (int)task.size(); i++) {
                        memcpyTasks.push_back(MultiThreadMemcpyMultiLinesTask(tempInput.cpuData + i * inputDim * input.unitSize, input.cpuData + task[i].first * inputDim * input.unitSize, inputDim * input.unitSize));
                    }
                    RunMultiThreadMemcpyMultiLines(memcpyTasks, GetAlivePool());
                    DoCpuLinearReshape(tempInput, *weights[e * 2], w3);
 // cnt["linear 0 prepare"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
                    DoCpuLinear(tempInput, *weights[e * 2], Data(), w3);
 // cnt["linear 0"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();                    
                    int mid = w3.dims[1] / 2;
                    w1.Resize({w3.dims[0], mid});
                    w1.dataType = w3.dataType;
                    w1.Allocate();

                    if (w3.dataType == DataType::FLOAT32) {
                        SwigluMultiThread((float *) w3.cpuData, mid, mid, ((float *) w1.cpuData),
                                    w3.dims[0], w3.dims[1], mid, GetAlivePool());
                    } else {
                        SwigluMultiThreadFloat16((uint16_t *) w3.cpuData, mid, mid, ((uint16_t *) w1.cpuData),
                                    w3.dims[0], w3.dims[1], mid, GetAlivePool());
                    }
  // cnt["swiglu"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
                    DoCpuLinearReshape(w1, *weights[e * 2 + 1], w3);
  // cnt["linear 1 prepare"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
                    DoCpuLinear(w1, *weights[e * 2 + 1], Data(), w3);
  // cnt["linear 1"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();                    
                    float *curOutput;
                    if (w3.dataType == DataType::FLOAT32) {
                        curOutput = (float*)w3.cpuData;
                    } else if (w3.dataType == DataType::FLOAT16) {
                        Float16ToFloat32((uint16_t*)w3.cpuData, middleResult.data(), w3.Count(0));
                        curOutput = middleResult.data();
                    }

                    RunMultiThreadMoeReduce(&task, &tempResult, curOutput, dim, GetAlivePool());
  // cnt["reduce"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
                }
                if (output.dataType == DataType::FLOAT32) {
                    memcpy(output.cpuData, tempResult.data(), output.GetBytes());
                } else if (output.dataType == DataType::FLOAT16) {
                    Float32ToFloat16(tempResult.data(), (uint16_t*)output.cpuData, output.Count(0));
                }
  // cnt["output memcpy"] += GetSpan(st, std::chrono::system_clock::now()); st = std::chrono::system_clock::now();
            }
  //printf("moe spend %f s.\n", GetSpan(veryst, std::chrono::system_clock::now()));
  // for (auto &it : cnt) {
     //printf("%s spend %f s.\n", it.first.c_str(), it.second);
  // }
        }
    }

    void CpuMergeMLA::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &qNope = *(datas.find("qNope")->second);
        Data &output = *(datas.find("output")->second);
        // int b = qNope.dims[0], s = q_nope.dims[1], h = q_nope.dims[2], d = q_nope.dims[3], c = qNope.dims.back();
        output.dataType = qNope.dataType;
        output.Resize(qNope.dims);
    }

    void CpuMergeMLA::Run(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &qNope = *(datas.find("qNope")->second);
        Data &qPe = *(datas.find("qPe")->second);
        Data &kvCache = *(datas.find("kvCache")->second);
        Data &peCache = *(datas.find("peCache")->second);
        Data &mask = *(datas.find("mask")->second);
        Data &output = *(datas.find("output")->second);
        float softmaxScale = floatParams.find("softmaxScale") != floatParams.end() ? floatParams.find("softmaxScale")->second : 1.0f;        
        int b = qPe.dims[0], s = qPe.dims[1], h = qPe.dims[2], c = qNope.dims.back(), t = kvCache.dims[1], r = qPe.dims[3];
        output.Allocate();

        // qNope: {b * s, h, c}
        // qPe: {b, s, h, r}
        // kvCache : {1, t, r}
        // peCache : {1, t, c}
        // output : {b * s, h, c}

        Data score0, score1;
        if (b == 1 && s == 1 && false) {
            // FastllmCudaMLA(qNope, qPe, kvCache, peCache, score0, output, softmaxScale);
        } else {
            if ((double)b * s * h * t * 2 * 4 > 1e9) {
                int parth = 1;
                Data qNopePart, qPePart;
                std::vector <Data> outputParts;
                std::vector <Data*> outputPartPointers;
                outputParts.resize((h - 1) / parth + 1);
                for (int i = 0; i < outputParts.size(); i++) {
                    outputPartPointers.push_back(&outputParts[i]);
                }
                for (int sth = 0; sth < h; sth += parth) {
                    int idx = sth / parth;
                    int curh = std::min(parth, h - sth);
                    Split(qNope, 1, sth, sth + curh, qNopePart);
                    Split(qPe, 2, sth, sth + curh, qPePart);
                    qNopePart.Reshape({b, s * curh, c});
                    MatMulTransB(qNopePart, peCache, score0);
                    score0.Reshape({b, s, curh, t});
                    qPePart.Reshape({b, s * curh, r});
                    MatMulTransB(qPePart, kvCache, score1);
                    score1.Reshape({b, s, curh, t});
                    AddTo(score1, score0);
                    Mul(score1, softmaxScale, score0);
                    if (mask.dims.size() > 0) {
                        score0.Reshape({b * s, curh, t});
                        ToDataType(mask, qNope.dataType);
                        AttentionMask(score0, mask, -10000);
                    }

                    Softmax(score0, score0, -1);
                    score0.Reshape({b, s * curh, t});
                    MatMul(score0, peCache, outputParts[idx]);
                    outputParts[idx].Reshape({b, s, curh, c});
                }
                CatBatch(outputPartPointers, 2, output);
                output.Reshape({b * s, h, c});
            } else {
                qNope.Reshape({b, s * h, c});
                MatMulTransB(qNope, peCache, score0);
                score0.Reshape({b, s, h, t});

                qPe.Reshape({qPe.dims[0], -1, qPe.dims[3]});
                MatMulTransB(qPe, kvCache, score1);
                score1.Reshape({b, s, h, t});
                AddTo(score1, score0);
                Mul(score1, softmaxScale, score0);

                if (mask.dims.size() > 0) {
                    score0.Reshape({b * s, h, t});
                    ToDataType(mask, qNope.dataType);
                    AttentionMask(score0, mask, -10000);
                }

                Softmax(score0, score0, -1);
                score0.Reshape({b, s * h, t});
                MatMul(score0, peCache, output);
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
                        weight.dataType == DataType::FLOAT16 ||
                        weight.dataType == DataType::BFLOAT16, "Embedding's weight's type should be float32 or float16 or bfloat16.\n");
        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16, 
                        "Embedding's input's type should be float32 or float16.\n");

        weight.weightType = WeightType::EMBEDDING;
        int vocabSize = weight.dims[0], embSize = weight.dims[1];
        std::vector <int> dims = input.dims;
        dims.push_back(embSize);

        output.dataType = input.dataType;
        if (weight.dataType == DataType::FLOAT16) {
            output.dataType = DataType::FLOAT16;
        }
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

    bool CpuConv1DPerChannel::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        return true;
    }

    void CpuConv1DPerChannel::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);

        int inputChannels = intParams.find("inputChannels")->second;
        int outputChannels = intParams.find("outputChannels")->second;
        int kernel = intParams.find("kernel")->second;
        int pad = intParams.find("pad")->second;
        int stride = intParams.find("stride")->second;
        
        AssertInFastLLM(weight.dims.size() == 3, "Conv1D's weight's shape's size should be 3.\n");
        AssertInFastLLM(input.dims[1] == inputChannels, "Conv1D's input's shape error.\n");

        weight.weightType = WeightType::CONV1D;

        std::vector <int> dims = input.dims;
        int inputLen = dims[2];
        int outputLen = (inputLen + pad + pad - kernel) / stride + 1;
        dims[1] = outputChannels;
        dims[2] = outputLen;

        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void CpuConv1DPerChannel::Run(const std::string &opType, const fastllm::DataDict &datas,
                      const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);
        output.Allocate(0.0f);
        int inputChannels = intParams.find("inputChannels")->second;    
        int outputChannels = intParams.find("outputChannels")->second; 
        int kernelSize = intParams.find("kernel")->second;
        int padding = intParams.find("pad")->second; 
        int stride = intParams.find("stride")->second;
        int groups = inputChannels;  // 组数等于通道数，实现逐通道卷积
        
        // 如果有groups参数，使用它
        if (intParams.find("groups") != intParams.end()) {
            groups = intParams.find("groups")->second;
        }
        
        std::vector<int> dims = input.dims;
        int batchSize = dims[0];        
        int inputLength = dims[2];      
        int outputLength = (inputLength + 2 * padding - kernelSize) / stride + 1;
        float *floatInput = (float*)input.cpuData;
        float *floatOutput = (float*)output.cpuData;

        float *floatWeight = (float*)weight.cpuData;
        float *floatBias = bias.dims.size() > 0 ? nullptr : (float*)bias.cpuData;

        std::vector <float> floatInputVector, floatOutputVector;
        if (input.dataType == DataType::FLOAT16) {
            floatInputVector.resize(input.Count(0));
            floatOutputVector.resize(output.Count(0));
            floatInput = (float*)floatInputVector.data();
            floatOutput = (float*)floatOutputVector.data();
            Float16ToFloat32((uint16_t*)input.cpuData, floatInput, (int)floatInputVector.size());
        }
        
        int channelsPerGroup = inputChannels / groups;   // 对于逐通道卷积，这是1
        int outputChannelsPerGroup = outputChannels / groups;  // 对于逐通道卷积，这也是1
        
        // 遍历批次
        for (int b = 0; b < batchSize; b++) {
            float *batchInput = floatInput + b * (inputChannels * inputLength);
            float *batchOutput = floatOutput + b * (outputChannels * outputLength);
            
            // 对于逐通道卷积，每个通道独立处理
            for (int g = 0; g < groups; g++) {
                // 对于每个组内的输出通道
                for (int oc = 0; oc < outputChannelsPerGroup; oc++) {
                    int globalOc = g * outputChannelsPerGroup + oc;
                    float *curWeight = floatWeight + globalOc * (channelsPerGroup * kernelSize);
                    float *curOutput = batchOutput + globalOc * outputLength;
                    
                    // 遍历输出序列位置
                    for (int ol = 0; ol < outputLength; ol++) {
                        int il = ol * stride - padding;
                        float value = floatBias ? floatBias[globalOc] : 0.0f;
                        
                        // 对于逐通道卷积，只处理对应的一个输入通道
                        for (int ic = 0; ic < channelsPerGroup; ic++) {
                            int globalIc = g * channelsPerGroup + ic;
                            float *curInput = batchInput + globalIc * inputLength;
                            
                            // 遍历kernel
                            for (int k = 0; k < kernelSize; k++) {
                                float inputValue = 0;
                                int inputPos = il + k;
                                
                                // 检查边界
                                if (inputPos >= 0 && inputPos < inputLength) {
                                    inputValue = curInput[inputPos];
                                }
                                
                                value += inputValue * curWeight[ic * kernelSize + k];
                            }
                        }
                        
                        curOutput[ol] = value;
                    }
                }
            }
        }

        if (input.dataType == DataType::FLOAT16) {
            Float32ToFloat16(floatOutput, (uint16_t*)output.cpuData, (int)floatOutputVector.size());
        }
    }

    bool CpuConv2DOp::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        return true;
    }

    void CpuConv2DOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);

        output.Allocate(0.0f);

        int inputChannels = intParams.find("inputChannels")->second;
        int outputChannels = intParams.find("outputChannels")->second;
        int kernelH = intParams.find("kernelH")->second;
        int kernelW = intParams.find("kernelW")->second;
        int padH = intParams.find("padH")->second;
        int padW = intParams.find("padW")->second;
        int strideH = intParams.find("strideH")->second;
        int strideW = intParams.find("strideW")->second;
        
        std::vector <int> dims = input.dims;
        int inputHeight = dims[2], inputWidth = dims[3];
        int outputHeight = (inputHeight + padH + padH - kernelH) / strideH + 1;
        int outputWidth = (inputWidth + padW + padW - kernelW) / strideW + 1;

        float *floatInput = (float*)input.cpuData;
        float *floatWeight = (float*)weight.cpuData;
        float *floatBias = (float*)bias.cpuData;
        float *floatOutput = (float*)output.cpuData;
        for (int oc = 0; oc < outputChannels; oc++) {
            float *startWeight = (float*)floatWeight + oc * (inputChannels * kernelH * kernelW);
            for (int oh = 0; oh < outputHeight; oh++) {
                for (int ow = 0; ow < outputWidth; ow++) {
                    int ih = oh * strideH - padH;
                    int iw = ow * strideW - padW;
                    float value = floatBias[oc];
                    float *curWeight = startWeight;
                    for (int c = 0; c < inputChannels; c++) {
                        float *curInput = (float*)floatInput + c * inputHeight * inputWidth;
                        for (int h = 0; h < kernelH; h++) {
                            for (int w = 0; w < kernelW; w++) {
                                float inputValue = 0;
                                if (ih + h >= 0 && ih + h < inputHeight && iw + w >= 0 && iw + w < inputWidth) {
                                    inputValue = curInput[(ih + h) * inputWidth + (iw + w)];
                                }
                                value += inputValue * (*(curWeight++));
                            }
                        }
                    }

                    *(floatOutput++) = value;
                }
            }
        }
    }

    void CpuConv2DOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);

        int inputChannels = intParams.find("inputChannels")->second;
        int outputChannels = intParams.find("outputChannels")->second;
        int kernelH = intParams.find("kernelH")->second;
        int kernelW = intParams.find("kernelW")->second;
        int padH = intParams.find("padH")->second;
        int padW = intParams.find("padW")->second;
        int strideH = intParams.find("strideH")->second;
        int strideW = intParams.find("strideW")->second;
        
        AssertInFastLLM(weight.dims.size() == 4, "Conv2D's weight's shape's size should be 4.\n");
        AssertInFastLLM(input.dims[1] == inputChannels, "Conv2D's input's shape error.\n");

        weight.weightType = WeightType::CONV2D;

        std::vector <int> dims = input.dims;
        int inputHeight = dims[2], inputWidth = dims[3];
        int outputHeight = (inputHeight + padH + padH - kernelH) / strideH + 1;
        int outputWidth = (inputWidth + padW + padW - kernelW) / strideW + 1;
        dims[1] = outputChannels;
        dims[2] = outputHeight;
        dims[3] = outputWidth;

        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void DoCpuLinearReshape(Data &input, Data &weight, Data &output) {
        weight.weightType = WeightType::LINEAR;
        std::vector <int> dims = input.dims;
        dims.back() = weight.dims[0];

        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void CpuLinearOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);

        AssertInFastLLM(weight.dims.size() == 2, "Linear's weight's shape's size should be 2.\n");
        AssertInFastLLM(input.dims.back() == weight.dims[1], "Linear's weight's shape error.\n");

        DoCpuLinearReshape(input, weight, output);
    }

    void MultiThreadInt4GroupLinearOp::Run() {
        for (int i = 0; i < n; i++) {
            for (int j = st; j < end; j++) {
                float now = biasData ? biasData[j] : 0.0f;
                for (int g = 0; g < group; g++) {
                    int gst = g * groupCnt;
                    int gend = std::min((g + 1) * groupCnt, m);
                    int l = gst;
                    float curMin = fp16tofp32.dict[mins[j * group + g]];
                    float curScale = fp16tofp32.dict[scales[j * group + g]];
#ifdef __AVX2__
                    __m256 now_vec = _mm256_setzero_ps();
                    const __m256 scale_vec = _mm256_set1_ps(curScale);
                    const __m256 min_vec = _mm256_set1_ps(curMin);
                        
                    for (; l + 8 <= gend; l += 8) {
                        // 计算权重索引（每次处理4个字节）
                        size_t weight_offset = (j * m + l) / 2;
                        const uint8_t* weight_ptr = &weightData[weight_offset];
                            
                        // 加载4个权重字节
                        __m128i v = _mm_loadl_epi64((const __m128i*)weight_ptr);
                            
                        // 拆分高/低4位
                        __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), _mm_set1_epi8(0x0F));
                        __m128i lo = _mm_and_si128(v, _mm_set1_epi8(0x0F));
                            
                        // 交错排列成 [hi0, lo0, hi1, lo1, ...]
                        __m128i hi_lo = _mm_unpacklo_epi8(hi, lo);
                            
                        // 将8个字节扩展为8个int32
                        __m128i lo_nib = _mm_cvtepu8_epi32(hi_lo);
                        __m128i hi_nib = _mm_cvtepu8_epi32(_mm_srli_si128(hi_lo, 4));
                        __m256i nibbles = _mm256_set_m128i(hi_nib, lo_nib);
                        
                        // 转换为浮点数并计算权重
                        __m256 weights = _mm256_fmadd_ps(
                        _mm256_cvtepi32_ps(nibbles), scale_vec, min_vec);
                        
                        // 加载8个输入元素
                        const float* input_ptr = &inputData[i * m + l];
                        __m256 input = _mm256_loadu_ps(input_ptr);
                            
                        // 乘积累加
                        now_vec = _mm256_fmadd_ps(input, weights, now_vec);
                    }
                        
                    // 水平求和
                    __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(now_vec, 1),
                                                _mm256_castps256_ps128(now_vec));
                    sum128 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
                    sum128 = _mm_add_ss(sum128, _mm_shuffle_ps(sum128, sum128, 0x55));
                    now += _mm_cvtss_f32(sum128);
#endif
                    // 处理剩余不足8个元素的情况
                    for (; l + 1 < gend; l += 2) {
                        uint8_t v = weightData[(j * m + l) / 2];
                        now += inputData[i * m + l] * (curMin + (v >> 4) * curScale);
                        now += inputData[i * m + l + 1] * (curMin + (v & 0x0F) * curScale);
                    }

                    for (; l < gend; l++) {
                        int id = (j * m + l) / 2;
                        float weight = 0.0f;
                        if ((j * m + l) % 2) {
                            weight = curMin + (weightData[id] & 0xF) * curScale;
                        } else {
                            weight = curMin + (weightData[id] >> 4) * curScale;
                        }
                        now += inputData[i * m + l] * weight;
                    }
                }

                outputData[i * k + j] = now;
            }
        }
    }

    struct MultiThreadBase3GroupLinearOp : MultiThreadBaseOp {
        float *inputData;
        uint8_t *weightData;
        float *biasData, *outputData;
        int n, m, k, st, end, group, groupCnt;
        uint16_t *halfScales;

        MultiThreadBase3GroupLinearOp(float *inputData, uint8_t *weightData, float *biasData, float *outputData,
                           int n, int m, int k, int st, int end, int group, int groupCnt, uint16_t *halfScales) : 
            inputData(inputData), weightData(weightData), biasData(biasData), outputData(outputData),
            n(n), m(m), k(k), st(st), end(end), group(group), groupCnt(groupCnt), halfScales(halfScales) {}

        void Run() {
            std::vector <uint8_t> base = {1, 3, 9, 27, 81};
            int bytesPerGroup = ((groupCnt - 1) / 5) + 1;   
            for (int i = 0; i < n; i++) {
                for (int j = st; j < end; j++) {
                    float now = biasData ? biasData[j] : 0.0f;
                    for (int g = 0; g < group; g++) {
                        uint8_t *cur = weightData + j * group * bytesPerGroup + g * bytesPerGroup;
                        float sum = 0.0;
                        int l = 0;
                        for (; l < groupCnt && g * groupCnt + l < m; l++) {
                            sum += inputData[i * m + g * groupCnt + l] * (cur[l / 5] / base[l % 5] % 3 - 1);
                        }
                        now += sum * fp16tofp32.dict[halfScales[j * group + g]];
                    }
                    outputData[i * k + j] = now;
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
    void MultiplyInt4GroupMultiThreadLaunch(uint8_t *a, uint8_t *b, float *c, int n, int m, int k,
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
                ops[startTid + i] = new MultiThreadLinearInt8Int4GroupOp(a, b + cur * m / 2, c + cur, n, m, end - cur, k,
                                           weightSums + cur * group, weightMins + cur * group, scales + cur * group,
                                           (bias == nullptr ? (float *) nullptr : bias + cur), iscales.data(), izeros.data(),
                                           inputSums.data(), group, groupCnt);
            } else {
                ops[startTid + i] = new MultiThreadLinearInt4NoZeroOp(a, b + cur * m / 2, (int32_t*)c + cur, n, m, end - cur, k,
                                           weightSums + cur * group, weightMins + cur * group, scales + cur * group,
                                           (bias == nullptr ? (float *) nullptr : bias + cur), configs.data(), inputSums.data());
            }
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(startTid + i, ops[startTid + i]);
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
#ifdef __AVX2__
        __m256 mins = _mm256_set1_ps(1e100);
        __m256 maxs = _mm256_set1_ps(-1e100);
        for (; j + 7 < len; j += 8) {
            __m256 v = _mm256_loadu_ps(a + j);
            mins = _mm256_min_ps(mins, v);
            maxs = _mm256_max_ps(maxs, v);
        }
        // 将 AVX2 寄存器中的最小值、最大值提取到标量
        float temp_min[8], temp_max[8];
        _mm256_storeu_ps(temp_min, mins);
        _mm256_storeu_ps(temp_max, maxs);
        for (int l = 0; l < 8; l++) {
            minValue = std::min(minValue, temp_min[l]);
            maxValue = std::max(maxValue, temp_max[l]);
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
#ifdef __AVX2__
        __m256 vScale = _mm256_set1_ps(scale);
        __m256 vZeroPoint = _mm256_set1_ps(zeroPoint);
        __m256 vZero = _mm256_setzero_ps();
        __m256 vHalf = _mm256_set1_ps(0.5f);
        __m256 vMax = _mm256_set1_ps(255.0f);
        for (; j + 7 < len; j += 8) {
            // Load 8 floats
            __m256 vValue = _mm256_loadu_ps(&fValue[j]);
            
            // fValue[j] / scale + zeroPoint + 0.5
            __m256 vScaled = _mm256_div_ps(vValue, vScale);
            __m256 vWithZP = _mm256_add_ps(vScaled, vZeroPoint);
            __m256 vWithHalf = _mm256_add_ps(vWithZP, vHalf);
            
            // max(..., 0.0)
            __m256 vClampedLow = _mm256_max_ps(vWithHalf, vZero);
            
            // min(..., 255.0)
            __m256 vClampedHigh = _mm256_min_ps(vClampedLow, vMax);
            
            // Convert to int32 (truncate)
            __m256i vInt32 = _mm256_cvtps_epi32(vClampedHigh);
            
            // Pack into 16-bit integers
            __m128i vInt16 = _mm_packus_epi32(
                _mm256_extractf128_si256(vInt32, 0),
                _mm256_extractf128_si256(vInt32, 1));
            
            // Pack into 8-bit integers
            __m128i vInt8 = _mm_packus_epi16(vInt16, vInt16);
            
            // Store the lower 64 bits (8 bytes)
            _mm_storel_epi64((__m128i*)&uValue[j], vInt8);
        }
#endif
        for (; j < len; j++) {
            uValue[j] = (uint8_t) (std::min(255., (double) std::max(fValue[j] / scale + zeroPoint + 0.5, 0.0)));
        }
    }

#ifdef __AVX2__
    void Avx2InputPermute(uint8_t* output, int n, int m) {
         if (cpuInstructInfo.hasAVX512VNNI) {
            uint8_t *temp = new uint8_t[64];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j + 63 < m; j += 64) {
                    memcpy(temp, output + i * m + j, 64);
                    for (int k = 0; k < 32; k++) {
                        output[i * m + j + k] = temp[k * 2 + 1];
                        output[i * m + j + k + 32] = temp[k * 2];
                    }
                }
            }
            delete[] temp;
            return;
        } 
        
        /*uint8_t *temp = new uint8_t[32];
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
        return;*/

        const __m256i mask_even = _mm256_setr_epi8(
            0, 2, 4, 6, 8, 10, 12, 14, 
            16, 18, 20, 22, 24, 26, 28, 30,
            0, 2, 4, 6, 8, 10, 12, 14,
            16, 18, 20, 22, 24, 26, 28, 30
        );
        const __m256i mask_odd = _mm256_setr_epi8(
            1, 3, 5, 7, 9, 11, 13, 15,
            17, 19, 21, 23, 25, 27, 29, 31,
            1, 3, 5, 7, 9, 11, 13, 15,
            17, 19, 21, 23, 25, 27, 29, 31
        );
        for (int i = 0; i < n; i++) {
            for (int j = 0; j + 31 < m; j += 32) {
                // 加载32字节数据
                __m256i data = _mm256_loadu_si256((__m256i*)(output + i * m + j));
                __m256i evens = _mm256_shuffle_epi8(data, mask_even);
                __m256i odds = _mm256_shuffle_epi8(data, mask_odd);
                __m128i evenLow = _mm256_castsi256_si128(evens); // a[0]~a[15]
                __m128i evenHigh = _mm256_extracti128_si256(evens, 1); // a[16]~a[31]
                __m128i low = _mm_unpacklo_epi64(evenLow, evenHigh);

                __m128i oddLow = _mm256_castsi256_si128(odds); // a[0]~a[15]
                __m128i oddHigh = _mm256_extracti128_si256(odds, 1); // a[16]~a[31]
                __m128i high = _mm_unpacklo_epi64(oddLow, oddHigh);

                // 存储结果
                _mm256_storeu_si256((__m256i*)(output + i * m + j), _mm256_set_m128i(low, high));
            }
        }
    }
#endif

    void MultiThreadFloat32ToBFloat16Op::Run() {
        Float32ToBFloat16(input, output, len);
    }

    void MultiThreadFloat32ToQ8KOp::Run() {
        AssertInFastLLM((void*)input != (void*)output, "MultiThreadFloat32ToQ8KOp's input and output should be diff.\n");
        iqk_quantize_row_q8_K (
            input, output, len, 
            ggml_type_vec_dot_type((ggml_type)ggmlType), (ggml_type)ggmlType
        );
    }

    void MultiThreadOnlineQuantizationOp::Run() {
        int realGroup = (m - 1) / groupCnt + 1;
        for (int i = 0; i < n; i++) {
            float *cur = input + i * m;
            uint8_t *u = output + i * m;
            for (int g = 0; g < realGroup; g++) {
                int st = g * groupCnt;
                int end = std::min(m, (g + 1) * groupCnt);
                float minValue = 1e9, maxValue = -1e9;
                GetArrayMinMax(input + i * m + st, end - st, minValue, maxValue);
                configs[i * group + g] = (LowBitConfig(minValue, maxValue, 8, 0));
                QuantizationAll(cur + st, u + st, end - st, &configs[i * group + g]);
            }
        }

        if (permuteType == 0) {
            // for INT8 * INT8
#ifdef __AVX2__
            for (int i = 0; i < n * m; i++) {
                output[i] = (output[i] + !output[i]);
            }
#endif
        }

        if (permuteType == 1) {
            // for INT8 * INT4
#ifdef __AVX2__
            Avx2InputPermute(output, n, m);
#endif
        }

        if (inputSums != nullptr) {
            for (int i = 0; i < n; i++) {
                for (int g = 0; g < realGroup; g++) {
                    iscales[i * group + g] = configs[i * group + g].scale;
                    izeros[i * group + g] = configs[i * group + g].zeroPoint;
                    int sum = 0;
                    int j = g * groupCnt;
#ifdef __AVX2__
                    const __m256i ones8 = _mm256_set1_epi8(1);
                    const __m256i ones16 = _mm256_set1_epi16(1);
                    __m256i acc = _mm256_setzero_si256();
                    for (; j + 31 < (g + 1) * groupCnt && j + 31 < m; j += 32) {
                        __m256i data = _mm256_loadu_si256((__m256i*)(output + i * m + j));
                        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(data, ones8), ones16));
                    }
                    sum += I32sum(acc);
#endif
                    for (; j < (g + 1) * groupCnt && j < m; j++) {
                        sum += output[i * m + j];
                    }
                    inputSums[i * group + g] = sum;
                }
            }
        }

        if (permuteType == 0) {
            // for INT8 * INT8
#ifdef __AVX2__
            for (int i = 0; i < n * m; i++) {
                output[i] ^= 128;
            }
#endif
        }
    }

    bool CpuLinearOp::CanRun(const std::string &opType, const fastllm::DataDict &datas,
                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        if (intParams.find("exType") != intParams.end()) {
            return false;
        }
        return true;
    }

    void DoCpuLinear(Data &input, Data &weight, const Data &bias, Data &output) {
//auto st = std::chrono::system_clock::now();
        output.Allocate();
        int n = input.Count(0) / input.dims.back();
        int m = input.dims.back();
        int k = output.dims.back();
        int threadSt = GetAlivePool()->curActivateThreadInterval.first;
        int threadLen = GetAlivePool()->curActivateThreadInterval.second - GetAlivePool()->curActivateThreadInterval.first;

        if (input.dataType == DataType::FLOAT32 && output.dataType == DataType::FLOAT32) {
            if (weight.dataType == DataType::FLOAT32) {
                RunLinearFloat32Float32((float*)input.cpuData, (float*)weight.cpuData, (float*)output.cpuData, 
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::BFLOAT16) {
                RunLinearFloat32BFloat16((float*)input.cpuData, (uint16_t*)weight.cpuData, (float*)output.cpuData, 
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::FLOAT16) {
                RunLinearFloat32Float16((float*)input.cpuData, (uint16_t*)weight.cpuData, (float*)output.cpuData, 
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::INT8) {
                RunLinearFloat32Int8((float*)input.cpuData, weight, (float*)output.cpuData, 
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::INT4_GROUP || weight.dataType == DataType::INT4_NOZERO) {
                int group = weight.group, groupCnt = weight.groupCnt;
                if (weight.dataType == DataType::INT4_NOZERO) {
                    group = 1, groupCnt = m;
                }
                RunLinearFloat32Int4Group((float*)input.cpuData, weight, (float*)output.cpuData, 
                                        bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, group, groupCnt,
                                        GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::INT2_GROUP) {
                int group = weight.group, groupCnt = weight.groupCnt;
                RunLinearFloat32Int2Group((float*)input.cpuData, weight, (float*)output.cpuData, 
                                        bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, group, groupCnt,
                                        GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::BASE3_GROUP) {
                std::vector <uint8_t> base = {1, 3, 9, 27, 81};
                float *inputData = (float *) input.cpuData;
                uint8_t *weightData = (uint8_t *) weight.cpuData;
                float *outputData = (float *) output.cpuData;
                float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
                
                auto pool = GetAlivePool();
                int threadNum = pool->threads.size();
                int per = k / threadNum;
                int cur = 0;
                std::vector<fastllm::MultiThreadBase3GroupLinearOp*> ops;
                for (int i = 0; i < threadNum; i++) {
                    int end = cur + per + (cur + per * (threadNum - i) < k);
                    if (i == threadNum - 1) {
                        end = k;
                    }
                    ops.push_back(new MultiThreadBase3GroupLinearOp(inputData, weightData, biasData, outputData,
                                                   n, m, k, cur, end, weight.group, weight.groupCnt, weight.halfScales.data()));
                    cur = end;
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->PushOp(i, ops[i]);
                }
                for (int i = 0; i < threadNum; i++) {
                    pool->Wait(i);
                    delete ops[i];
                }
            } else if (weight.dataType == DataType::INT4) {
                // 目前已经不用这种数据类型了
                float *inputData = (float *) input.cpuData;
                uint8_t *weightData = (uint8_t *) weight.cpuData;
                float *outputData = (float *) output.cpuData;
                float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
                weight.CalcWeightSum();

                std::vector<LowBitConfig> inputConfigs;
                std::vector<uint8_t> uinput;
                std::vector <float> inputSums, iscales, izeros;
                OnlineQuantization(inputData, uinput, inputConfigs, n, m, 1, m, inputSums, iscales, izeros, 1);
                MultiplyInt4MultiThread(uinput.data(), weightData, (int32_t *) outputData, n, m, k,
                                            weight.weightSum.data(), weight.zeros.data(), weight.scales.data(),
                                            biasData,
                                            inputConfigs, GetThreads());
            } else if (weight.dataType == DataType::FP8_E4M3) {
                RunLinearFloat32FP8E4M3((float*)input.cpuData, weight, (float*)output.cpuData, 
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::DATA_GGUF_FORMAT) {
                RunLinearFloat32GGUF((float*)input.cpuData, (uint8_t*)weight.cpuData, (float*)output.cpuData, bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, 
                    &weight, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else {
                ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
            }
        } else if (input.dataType == DataType::FLOAT16 && output.dataType == DataType::FLOAT16) {
            if (weight.dataType == DataType::FLOAT32) {
                RunLinearFloat16Float32((uint16_t*)input.cpuData, (float*)weight.cpuData, (uint16_t*)output.cpuData, 
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::FLOAT16) {
                RunLinearFloat16Float16((uint16_t*)input.cpuData, (uint16_t*)weight.cpuData, (uint16_t*)output.cpuData, 
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::INT8) {
                RunLinearFloat16Int8((uint16_t*)input.cpuData, weight, (uint16_t*)output.cpuData, 
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::INT4_GROUP || weight.dataType == DataType::INT4_NOZERO) {
                int group = weight.group, groupCnt = weight.groupCnt;
                if (weight.dataType == DataType::INT4_NOZERO) {
                    group = 1, groupCnt = m;
                }
                RunLinearFloat16Int4Group((uint16_t*)input.cpuData, weight, (uint16_t*)output.cpuData, 
                                        bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, group, groupCnt,
                                        GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::FP8_E4M3) {
                RunLinearFloat16FP8E4M3((uint16_t*)input.cpuData, weight, (uint16_t*)output.cpuData, 
                    bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else if (weight.dataType == DataType::DATA_GGUF_FORMAT) {
                RunLinearFloat16GGUF((uint16_t*)input.cpuData, (uint8_t*)weight.cpuData, (uint16_t*)output.cpuData, bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr, 
                    &weight, n, m, k, GetAlivePool(), threadSt, threadLen);
            } else {
                ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
            }
        } else {
            ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
        }
//float spend = GetSpan(st, std::chrono::system_clock::now());
//float gops = (float)n * m * k / spend / 1e9;
//printf("n = %d, m = %d, k = %d, spend %f s, gops = %f\n", n, m, k, spend, gops);
    }

    void CpuLinearOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);
        AssertInFastLLM(bias.dataType == DataType::FLOAT32, "Linear's bias' type should be float32.\n");
        DoCpuLinear(input, weight, bias, output);
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

    void DoCpuCatDirect(Data &input0, Data &input1, int axis) {
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

    void CpuCatDirectOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                             const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);

        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        DoCpuCatDirect(input0, input1, axis);
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
#if defined(__AVX__)
                        __m256 vsum = _mm256_set1_ps(0.0f);
                        for (; l + 7 < m; l += 8) {
                            __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *) (input0Data + i * input0Stride + l)));
                            __m256 vy = _mm256_cvtph_ps(_mm_loadu_si128((__m128i *) (input1Data + j * input1Stride + l)));
                            vsum = _mm256_add_ps(vsum, _mm256_mul_ps(vx, vy));
                        }
                        now += Floatsum(vsum);
#endif
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
                        (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16) ||
                        (input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT16),
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
        if (input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) {
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
        } else if (input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT16) {
            std::vector <uint16_t> fp16InputData;
            fp16InputData.resize(input0.Count(0));
            Float32ToFloat16((float*)input0.cpuData, fp16InputData.data(), input0.Count(0));

            auto *pool = GetAlivePool();
            int threads = pool->threads.size();
            std::vector<fastllm::MultiThreadMatMulFloat16SingleOp*> ops;
            for (int o = 0; o < batch0; o++) {
                ops.push_back(new MultiThreadMatMulFloat16SingleOp(
                    (uint16_t *) fp16InputData.data(), (uint16_t *) input1.cpuData, (uint16_t *) output.cpuData,
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
            if (batch0 == 1) {
                int partn = std::max(1, n / threads);
                for (int o = 0; o < n; o += partn) {
                    int len = std::min(partn, n - o);
                    ops.push_back(new MultiThreadMatMulFloat16SingleOp(
                        ((uint16_t *) input0.cpuData) + o * m, 
                        (uint16_t *) input1.cpuData, 
                        ((uint16_t *) output.cpuData) + o * k,
                        input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                        len, m, k, alpha, 0, 1
                    ));
                }
            } else {
                for (int o = 0; o < batch0; o++) {
                    ops.push_back(new MultiThreadMatMulFloat16SingleOp(
                        (uint16_t *) input0.cpuData, (uint16_t *) input1.cpuData, (uint16_t *) output.cpuData,
                        input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                        n, m, k, alpha, o, o + 1
                    ));
                }
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
                        (input0.dataType == DataType::FLOAT16 && input1.dataType == DataType::FLOAT16) ||
                        (input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT16),
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
        if (input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32) {
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
        } else if (input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT16) {
            std::vector <uint16_t> fp16InputData;
            fp16InputData.resize(input0.Count(0));
            Float32ToFloat16((float*)input0.cpuData, fp16InputData.data(), input0.Count(0));

            auto *pool = GetAlivePool();
            int threads = pool->threads.size();
            std::vector<fastllm::MultiThreadMatMulTransBFloat16SingleOp*> ops;
            for (int o = 0; o < batch0; o++) {
                ops.push_back(new MultiThreadMatMulTransBFloat16SingleOp(
                    (uint16_t *) fp16InputData.data(), (uint16_t *) input1.cpuData, (uint16_t *) output.cpuData,
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
            if (batch0 == 1) {
                int partn = std::max(1, n / threads);
                for (int o = 0; o < n; o += partn) {
                    int len = std::min(partn, n - o);
                    ops.push_back(new MultiThreadMatMulTransBFloat16SingleOp(
                        ((uint16_t *) input0.cpuData) + o * m, 
                        (uint16_t *) input1.cpuData, 
                        ((uint16_t *) output.cpuData) + o * k,
                        input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                        len, m, k, alpha, 0, 1
                    ));
                }
            } else {
                for (int o = 0; o < batch0; o++) {
                    ops.push_back(new MultiThreadMatMulTransBFloat16SingleOp(
                        (uint16_t *) input0.cpuData, (uint16_t *) input1.cpuData, (uint16_t *) output.cpuData,
                        input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                        n, m, k, alpha, o, o + 1
                    ));
                }
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

    void CpuNormalizeOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;

        AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                        "Normalize error: Data's type should be float32 or float16.\n");

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
                float sum = 0;
                for (int j = 0; j < channels; j++) {
                    sum += inputData[j];
                }
                for (int j = 0; j < channels; j++) {
                    inputData[j] /= sum;
                }
                inputData += channels;
                outputData += channels;
            }
        } else {
            /*for (int i = 0; i < outer; i++) {
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
            }*/
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

    struct FP16SiluManager {
        uint16_t dict[65536];

        FP16SiluManager() {
            for (int i = 0; i < 65536; i++) {
                float x = half_to_float(i);
                float y = x / (1.0 + expf(-x));
                dict[i] = float_to_half(y);
            }
        }
    } fp16SiluManager;

    struct FP16SigmoidManager {
        uint16_t dict[65536];

        FP16SigmoidManager() {
            for (int i = 0; i < 65536; i++) {
                float x = half_to_float(i);
                float y = 1.0 / (1.0 + expf(-x));
                dict[i] = float_to_half(y);
            }
        }
    } fp16SigmoidManager;

    void CpuSiluOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16, 
                        "Silu error: Data's type should be float32 or float16.\n");
        int len = input.Count(0);

        if (input.dataType == DataType::FLOAT16) {
            uint16_t *inputData = (uint16_t*)input.cpuData;
            uint16_t *outputData = (uint16_t*)output.cpuData;
            for (int i = 0; i < len; i++) {
                outputData[i] = fp16SiluManager.dict[inputData[i]];
            }
        } else {
            float *inputData = (float*)input.cpuData;
            float *outputData = (float*)output.cpuData;
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

    void CpuReluOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32, "Relu error: Data's type should be float32.\n");

        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;
        int len = input.Count(0);
        int i = 0;
        for (; i < len; i++) {
            float x = inputData[i];
            outputData[i] = x > 0 ? x : 0;
        }
    }

    void CpuSigmoidOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16, 
                        "Sigmoid error: Data's type should be float32 or float16.\n");

        int len = input.Count(0);
        if (input.dataType == DataType::FLOAT16) {
            uint16_t *inputData = (uint16_t*)input.cpuData;
            uint16_t *outputData = (uint16_t*)output.cpuData;
            for (int i = 0; i < len; i++) {
                outputData[i] = fp16SigmoidManager.dict[inputData[i]];
            }
        } else {
            float *inputData = (float*)input.cpuData;
            float *outputData = (float*)output.cpuData;
            int i = 0;
            for (; i < len; i++) {
                float x = inputData[i];
                outputData[i] = 1.0 / (1.0 + exp(-x));
            }
        }
    }

    void CpuExpOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16, "Exp error: Data's type should be float32 or float16.\n");

        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;

        std::vector <float> floatInputVector, floatOutputVector;
        if (input.dataType == DataType::FLOAT16) {
            floatInputVector.resize(input.Count(0));
            floatOutputVector.resize(output.Count(0));
            inputData = (float*)floatInputVector.data();
            outputData = (float*)floatOutputVector.data();
            Float16ToFloat32((uint16_t*)input.cpuData, inputData, (int)floatInputVector.size());
        }

        int len = input.Count(0);
        int i = 0;
        for (; i < len; i++) {
            float x = inputData[i];
            outputData[i] = exp(x);
        }

        if (input.dataType == DataType::FLOAT16) {
            Float32ToFloat16(outputData, (uint16_t*)output.cpuData, (int)floatOutputVector.size());
        }
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

    void DoCpuSwigluReshape(Data &input, Data &output) {
        std::vector <int> dims = input.dims;
        dims[dims.size() - 1] /= 2;
        output.dataType = input.dataType;
        output.Resize(dims);
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

    void DoCpuSwiglu(Data &input, Data &output) {
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                        "Swiglu error: Data's type should be float32 or float16.\n");

        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;

        int spatial = input.Count(input.dims.size() - 1), mid = spatial / 2;
        int outer = input.Count(0) / spatial;

        if (input.dataType == DataType::FLOAT32) {
            (MultiThreadSwigluOp((float*)inputData, spatial / 2, spatial / 2, (float*)outputData, outer, spatial, spatial / 2)).Run();
        } else if (input.dataType == DataType::FLOAT16) {
            (MultiThreadSwigluFloat16Op((uint16_t*)inputData, spatial / 2, spatial / 2, (uint16_t*)outputData, outer, spatial, spatial / 2)).Run();
        } else {
            printf("Unsupport swiglu type.");
        }
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

        int spatial = input.Count(input.dims.size() - 1), mid = spatial / 2;
        int outer = input.Count(0) / spatial;

        if (input.dataType == DataType::FLOAT32) {
            (SwigluMultiThread((float*)inputData, spatial / 2, spatial / 2, (float*)outputData, outer, spatial, spatial / 2, GetAlivePool()));
        } else if (input.dataType == DataType::FLOAT16) {
            (SwigluMultiThreadFloat16((uint16_t*)inputData, spatial / 2, spatial / 2, (uint16_t*)outputData, outer, spatial, spatial / 2, GetAlivePool()));
        } else {
            printf("Unsupport swiglu type.");
        }
        return;
    }

    void CpuSwigluGptOssOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                              const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);

        std::vector <int> dims = input.dims;
        dims[dims.size() - 1] /= 2;
        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void CpuSwigluGptOssOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                           const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                        "Swiglu error: Data's type should be float32 or float16.\n");

        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;

        int spatial = input.Count(input.dims.size() - 1), mid = spatial / 2;
        int outer = input.Count(0) / spatial;

        if (input.dataType == DataType::FLOAT32) {
            (SwigluGptOssMultiThread((float*)inputData, spatial / 2, spatial / 2, (float*)outputData, outer, spatial, spatial / 2, GetAlivePool()));
        } else if (input.dataType == DataType::FLOAT16) {
            ErrorInFastLLM("Error: Gpt oss swiglu, data type should be f32.");
            // (SwigluMultiThreadFloat16((uint16_t*)inputData, spatial / 2, spatial / 2, (uint16_t*)outputData, outer, spatial, spatial / 2, GetAlivePool()));
        } else {
            printf("Unsupport swiglu type.");
        }
        return;
    }

    inline float softplus(float x) {
        return  x > 20.0f ? x : std::log1p(std::exp(x));
    }

    void CpuMambaSoftplusOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                       const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &aLogData = *(datas.find("aLog")->second);
        Data &dtBiasData = *(datas.find("dtBias")->second);
        output.Allocate();
        AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                        "CpuMambaSoftplusOp error: Data's type should be float32 or float16.\n");
        AssertInFastLLM(aLogData.dataType == DataType::FLOAT32 && dtBiasData.dataType == DataType::FLOAT32,
                        "CpuMambaSoftplusOp error: alog's type and dtbias's type should be float32.\n");

        int dimsLen = input.dims.size();
        int outer = input.Count(0) / input.Count(dimsLen - 1);
        int channels = input.dims[dimsLen - 1];

        float *aLog = (float*)aLogData.cpuData;
        float *dtBias = (float*)dtBiasData.cpuData;

        // g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if (input.dataType == DataType::FLOAT32) {
            float *inputData = (float *) input.cpuData;
            float *outputData = (float *) output.cpuData;
            for (int o = 0; o < outer; o++) {
                for (int i = 0; i < channels; i++) {
                    outputData[o * channels + i] = -exp(aLog[i]) * softplus(inputData[o * channels + i] + dtBias[i]);
                }
            }
        } else if (input.dataType == DataType::FLOAT16) {
            uint16_t *inputData = (uint16_t *) input.cpuData;
            uint16_t *outputData = (uint16_t *) output.cpuData;

            for (int o = 0; o < outer; o++) {
                for (int i = 0; i < channels; i++) {
                    outputData[o * channels + i] = float_to_half(-exp(aLog[i]) * softplus(
                        fp16tofp32.dict[inputData[o * channels + i]] + dtBias[i]));
                }
            }
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

    void CpuAddOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                       const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();

        float v = floatParams.find("v") != floatParams.end() ? floatParams.find("v")->second : 1.0;
        AssertInFastLLM(input.dataType == DataType::FLOAT32 || input.dataType == DataType::FLOAT16,
                        "Add error: Data's type should be float32 or float16.\n");

        int len = input.Count(0);

        if (input.dataType == DataType::FLOAT32) {
            float *inputData = (float *) input.cpuData;
            float *outputData = (float *) output.cpuData;
            for (int i = 0; i < len; i++) {
                outputData[i] = inputData[i] + v;
            }
        } else if (input.dataType == DataType::FLOAT16) {
            uint16_t *inputData = (uint16_t *) input.cpuData;
            uint16_t *outputData = (uint16_t *) output.cpuData;
            for (int i = 0; i < len; i++) {
                outputData[i] = float_to_half(fp16tofp32.dict[inputData[i]] + v);
            }
        }
    }

    void CpuMulToOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input0 = *(datas.find("input0")->second);
        Data &input1 = *(datas.find("input1")->second);
        int input0Len = input0.Count(0);
        int input1Len = input1.Count(0);
        AssertInFastLLM(input0.dims == input1.dims || input1Len == 1 || input0Len % input1Len == 0, "MulTo error: input's shape should be same.\n");

        int len = input0.Count(0);
        int inner = input1.Count(0);
        AssertInFastLLM(len % inner == 0, "MulTo error: Data`s shape can`t perform MulTo operation.\n");
        int round = (len / inner);

        if (input1Len == 1) {
            if (input0.dataType == DataType::FLOAT16) {
                uint16_t *input0Data = (uint16_t*)input0.cpuData;
                uint16_t *input1Data = (uint16_t*)input1.cpuData;
                for (int i = 0; i < len; i++) {
                    input0Data[i] = float_to_half(fp16tofp32.dict[input0Data[i]] * fp16tofp32.dict[input1Data[0]]);
                }
            } else {
                float *input0Data = (float*)input0.cpuData;
                float *input1Data = (float*)input1.cpuData;
                for (int i = 0; i < len; i++) {
                    input0Data[i] *= input1Data[0];
                }
            }
        } else if (input0Len == input1Len) {
            if (input0.dataType == DataType::FLOAT16) {
                uint16_t *input0Data = (uint16_t*)input0.cpuData;
                uint16_t *input1Data = (uint16_t*)input1.cpuData;
                for (int i = 0; i < len; i++) {
                    input0Data[i] = float_to_half(fp16tofp32.dict[input0Data[i]] * fp16tofp32.dict[input1Data[i]]);
                }
            } else {
                float *input0Data = (float*)input0.cpuData;
                float *input1Data = (float*)input1.cpuData;
                for (int i = 0; i < len; i++) {
                    input0Data[i] *= input1Data[i];
                }
            }
        } else {
            int channelLen = input0Len / input1Len;
            if (input0.dataType == DataType::FLOAT16) {
                uint16_t *input0Data = (uint16_t*)input0.cpuData;
                uint16_t *input1Data = (uint16_t*)input1.cpuData;
                for (int i = 0; i < len; i++) {
                    input0Data[i] = float_to_half(fp16tofp32.dict[input0Data[i]] * fp16tofp32.dict[input1Data[i / channelLen]]);
                }
            } else {
                float *input0Data = (float*)input0.cpuData;
                float *input1Data = (float*)input1.cpuData;
                for (int i = 0; i < len; i++) {
                    input0Data[i] *= input1Data[i / channelLen];
                }
            }
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

        AssertInFastLLM(input0.dataType == DataType::FLOAT32 || input0.dataType == DataType::FLOAT16,
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

    void CpuRecurrentGatedDeltaRuleOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &last_recurrent_state = *(datas.find("last_recurrent_state")->second);
        Data &core_attn_out = *(datas.find("core_attn_out")->second);
        
        std::vector <int> dims = last_recurrent_state.dims;
        core_attn_out.dataType = last_recurrent_state.dataType;
        core_attn_out.Resize({dims[0], dims[1], 1, dims[3]});
    }

    struct MultiThreadRecurrentGatedDeltaRuleOp : MultiThreadBaseOp {
        int n0, n1, n2, n3, group;
        float *flast, *fgt, *fkt, *fvt, *fbt, *fqt, *fatv;
        int st, end;

        MultiThreadRecurrentGatedDeltaRuleOp(
            int n0, int n1, int n2, int n3, int group, 
            float *flast, float *fgt, float *fkt, float *fvt, 
            float *fbt, float *fqt, float *fatv,
            int st, int end
        ) : n0(n0), n1(n1), n2(n2), n3(n3), group(group),
            flast(flast), fgt(fgt), fkt(fkt), fvt(fvt),
            fbt(fbt), fqt(fqt), fatv(fatv),
            st(st), end(end) {}
        
        void Run() {
            std::vector <float> fkv_mem, temp;
            fkv_mem.resize(n3);
            temp.resize(n3);
            for (int i = st; i < end; i++) {
                float v = exp(fgt[i]);
                for (int j = 0; j < n2 * n3; j++) {
                    flast[i * n2 * n3 + j] *= v;
                }

                std::fill(fkv_mem.begin(), fkv_mem.end(), 0.0f);
                for (int j = 0; j < n2; j++) {
                    float curfkt = fkt[i / group * n2 + j];
                    for (int k = 0; k < n3; k++) {
                        fkv_mem[k] += flast[i * n2 * n3 + j * n3 + k] * curfkt;
                    }
                }

                float curfbt = fbt[i];
                for (int k = 0; k < n3; k++) {
                    temp[k] = ((fvt[i * n3 + k] - fkv_mem[k]) * curfbt);
                }

                for (int j = 0; j < n2; j++) {
                    for (int k = 0; k < n3; k++) {
                        flast[i * n2 * n3 + j * n3 + k] += fkt[i / group * n2 + j] * temp[k];
                    }
                }

                for (int j = 0; j < n2; j++) {
                    float curfqt = fqt[i / group * n2 + j];
                    for (int k = 0; k < n3; k++) {
                        fatv[i * n3 + k] += flast[i * n2 * n3 + j * n3 + k] * curfqt;
                    }
                }
            }
        }
    };

    void CpuRecurrentGatedDeltaRuleOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &k = *(datas.find("k")->second);
        Data &v = *(datas.find("v")->second);
        Data &g = *(datas.find("g")->second);
        Data &b = *(datas.find("b")->second);
        Data &last_recurrent_state = *(datas.find("last_recurrent_state")->second);
        Data &core_attn_out = *(datas.find("core_attn_out")->second);
        core_attn_out.Allocate(0.0f);
        
        Data &q_t = q, &k_t = k, &v_t = v, &g_t = g, &b_t = b;

        // last_recurrent_state = last_recurrent_state * g_t
        // kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        // delta = (v_t - kv_mem) * beta_t
        // last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        // core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)
        int n0 = last_recurrent_state.dims[0], n1 = last_recurrent_state.dims[1], n2 = last_recurrent_state.dims[2], n3 = last_recurrent_state.dims[3];
        float *flast = (float*)last_recurrent_state.cpuData;
        float *fgt = (float*)g_t.cpuData;
        float *fkt = (float*)k_t.cpuData;
        float *fvt = (float*)v_t.cpuData;
        float *fbt = (float*)b_t.cpuData;
        float *fqt = (float*)q_t.cpuData;
        float *fatv = (float*)core_attn_out.cpuData;

        int group = v.dims[1] / q.dims[1];
        std::vector <float> lastVector, gtVector, ktVector, vtVector, btVector, qtVector, atvVector;
        if (q.dataType == DataType::FLOAT16) {
            lastVector.resize(last_recurrent_state.Count(0));
            gtVector.resize(g_t.Count(0));
            ktVector.resize(k_t.Count(0));
            vtVector.resize(v_t.Count(0));
            btVector.resize(b_t.Count(0));
            qtVector.resize(q_t.Count(0));
            atvVector.resize(core_attn_out.Count(0));

            flast = (float*)lastVector.data();
            fgt = (float*)gtVector.data();
            fkt = (float*)ktVector.data();
            fvt = (float*)vtVector.data();
            fbt = (float*)btVector.data();
            fqt = (float*)qtVector.data();
            fatv = (float*)atvVector.data();

            Float16ToFloat32((uint16_t*)last_recurrent_state.cpuData, flast, (int)lastVector.size());
            Float16ToFloat32((uint16_t*)g_t.cpuData, fgt, (int)gtVector.size());
            Float16ToFloat32((uint16_t*)k_t.cpuData, fkt, (int)ktVector.size());
            Float16ToFloat32((uint16_t*)v_t.cpuData, fvt, (int)vtVector.size());
            Float16ToFloat32((uint16_t*)b_t.cpuData, fbt, (int)btVector.size());
            Float16ToFloat32((uint16_t*)q_t.cpuData, fqt, (int)qtVector.size());
        }

        int n = n0 * n1;
        auto pool = GetAlivePool();
        int threadNum = pool->threads.size();
        int per = n / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadRecurrentGatedDeltaRuleOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? n : cur + per + (cur + per * (threadNum - i) < n));
            ops.push_back(new MultiThreadRecurrentGatedDeltaRuleOp(n0, n1, n2, n3, group, flast, fgt, fkt, fvt, fbt, fqt, fatv, cur, end));
            cur = end;
        }
        for (int i = 0; i < threadNum; i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < threadNum; i++) {
            pool->Wait(i);
            delete ops[i];
        }

        if (q.dataType == DataType::FLOAT16) {
            Float32ToFloat16(fatv, (uint16_t*)core_attn_out.cpuData, (int)atvVector.size());
        }
    }

    void CpuTransferAttnOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        float *inputData = (float*)input.cpuData;
        int dimsLen = input.dims.size();
        int n = input.dims[dimsLen - 2], m = input.dims[dimsLen - 1], outer = input.Count(0) / input.Count(dimsLen - 2);

        std::vector <float> floatInputVector;
        if (input.dataType == DataType::FLOAT16) {
            floatInputVector.resize(input.Count(0));
            inputData = (float*)floatInputVector.data();
            Float16ToFloat32((uint16_t*)input.cpuData, inputData, (int)floatInputVector.size());
        }

        // 预分配最大所需的临时空间
        std::vector<float> tempRow(m);
        std::vector<float> tempSub(m * m);
        for (int o = 0; o < outer; o++) {
            float *batchData = inputData + o * n * m;
            
            for (int i = 1; i < n; i++) {
                // 复制第 i 行的前 i 个元素到临时数组
                std::memcpy(tempRow.data(), batchData + i * m, i * sizeof(float));
                
                // 复制子矩阵到临时数组
                for (int k = 0; k < i; k++) {
                    std::memcpy(tempSub.data() + k * m, batchData + k * m, i * sizeof(float));
                }
                
                // 更新第 i 行的前 i 个元素
                for (int j = 0; j < i; j++) {
                    float sum = tempRow[j];
                    for (int k = 0; k < i; k++) {
                        sum += tempRow[k] * tempSub[k * m + j];
                    }
                    batchData[i * m + j] = sum;
                }
            }
        }

        // attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < n; i++) {
                inputData[o * n * n + i * m + i] += 1.0f;
            }
        }

        if (input.dataType == DataType::FLOAT16) {
            Float32ToFloat16(inputData, (uint16_t*)input.cpuData, (int)floatInputVector.size());
        }
    }

    void CpuCausalMaskOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        int base = intParams.find("base") != intParams.end() ? intParams.find("base")->second : 0;
        float maskValue = floatParams.find("maskValue") != floatParams.end() ? floatParams.find("maskValue")->second : -10000.0;

        float *inputData = (float*)input.cpuData;

        std::vector <float> floatInputVector;
        if (input.dataType == DataType::FLOAT16) {
            floatInputVector.resize(input.Count(0));
            inputData = (float*)floatInputVector.data();
            Float16ToFloat32((uint16_t*)input.cpuData, inputData, (int)floatInputVector.size());
        }

        int dimsLen = input.dims.size();
        int n = input.dims[dimsLen - 2], m = input.dims[dimsLen - 1], outer = input.Count(0) / input.Count(dimsLen - 2);
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < n; i++) {
                for (int j = i + base; j < m; j++) {
                    inputData[o * n * m + i * m + j] = maskValue;
                }
            }
        }

        if (input.dataType == DataType::FLOAT16) {
            Float32ToFloat16(inputData, (uint16_t*)input.cpuData, (int)floatInputVector.size());
        }
    }

    void CpuAttentionMaskOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &mask = *(datas.find("mask")->second);
        float maskValue = floatParams.find("maskValue") != floatParams.end() ? floatParams.find("maskValue")->second : -10000.0;
        int spatial = input.Count(2), n = input.dims[0], m = input.dims[1];

        AssertInFastLLM(mask.dataType == DataType::FLOAT32 || mask.dataType == input.dataType, "AttentionMask: mask's datatype should be float32.");
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
        } else if (input.dataType == DataType::FLOAT16 && mask.dataType == DataType::FLOAT32) {
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
        } else if (input.dataType == DataType::FLOAT16 && mask.dataType == DataType::FLOAT16) {
            std::vector <float> floatMaskData;
            floatMaskData.resize(mask.Count(0));
            Float16ToFloat32((uint16_t*)mask.cpuData, floatMaskData.data(), mask.Count(0));
            float *maskData = floatMaskData.data();
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
                std::set <std::pair <float, int> > ans;
                for (int j = 0; j < channels; j++) {
                    if (ans.size() == topk) {
                        if (ans.begin()->first < inputData[j]) {
                            ans.erase(ans.begin());
                            ans.insert(std::make_pair(inputData[j], j));
                        }
                    } else {
                        ans.insert(std::make_pair(inputData[j], j));
                    }
                }

                int j = topk - 1;
                for (auto &it : ans) {
                    outputData[j * 2] = it.second;
                    outputData[j * 2 + 1] = it.first;
                    j--;
                }

                inputData += channels;
                outputData += 2 * topk;
            }
            return;
/*
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
*/
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

    static std::vector <uint8_t> vold;

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
        same |= ((axis == std::vector <int>{2, 0, 1, 3}) && input.dims[0] == 1 && input.dims[1] == 1);
        same |= ((axis == std::vector <int>{0, 2, 1, 3}) && (input.dims[1] == 1 || input.dims[2] == 1));
        same |= ((axis == std::vector <int>{1, 0, 2, 3}) && (input.dims[0] == 1 || input.dims[1] == 1));
        same |= ((axis == std::vector <int>{1, 2, 0, 3}) && input.dims[1] == 1 && input.dims[2] == 1);
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
            if (vold.size() < input.GetBytes()) {
                vold.resize(input.GetBytes());
            }
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
            if (vold.size() < input.GetBytes()) {
                vold.resize(input.GetBytes());
            }
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
        int positionStride = intParams.find("positionStride") != intParams.end() ? intParams.find("positionStride")->second : 1;

        int len = data.dims[0], bs = data.dims[1];
        int spatial = data.Count(2);
        int n = data.dims[2], m = data.dims[3];
        int stride = (int)sinData.dims[1];
        positionStride *= positionIds.dims.back();
        for (int l = 0; l < len; l++) {
            for (int b = 0; b < bs; b++) {
                if (data.dataType == DataType::FLOAT32) {
                    int index = (int) ((float *) positionIds.cpuData)[b * positionStride + l];
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
                    int index = (int) ((float *) positionIds.cpuData)[b * positionStride + l];
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

    struct MultiThreadLlamaRotatePosition2DPartFloatOp : MultiThreadBaseOp {
        DataType dataType;
        float *data, *positionIds, *sinData, *cosData;
        int bs, len, n, m, stride, spatial, posDim, rotaryDim, part;      
        int st, end;

        MultiThreadLlamaRotatePosition2DPartFloatOp 
            (DataType dataType, float *data, float *positionIds, float *sinData, float *cosData, 
            int bs, int len, int n, int m, int stride, int spatial, int posDim, int rotaryDim, int part,
            int st, int end) : 
            dataType(dataType), data(data), positionIds(positionIds), sinData(sinData), cosData(cosData), 
            bs(bs), len(len), n(n), m(m), stride(stride), spatial(spatial), posDim(posDim), rotaryDim(rotaryDim), part(part),
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
                        for (int j = 0; j < rotaryDim && j < m / 2 && j < part / 2; j++) {
                            float a = d[j], b = d[j + part / 2];
                            d[j] = a * cos[j] - b * sin[j];
                            d[j + part / 2] = a * sin[j] + b * cos[j];
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
                        for (int j = 0; j < rotaryDim && j < m / 2 && j < part / 2; j++) {
                            float a = fp16tofp32.dict[d[j]], b = fp16tofp32.dict[d[j + part / 2]];
                            d[j] = float_to_half(a * cos[j] - b * sin[j]);
                            d[j + part / 2] = float_to_half(a * sin[j] + b * cos[j]);
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

    static void RunMultiThreadLlamaRotatePosition2DPartFloat(DataType dataType, float *data, float *positionIds, float *sinData, float *cosData, 
            int bs, int len, int n, int m, int stride, int spatial, int posDim, int rotaryDim, int part, AliveThreadPool *pool) {
        if (bs * len == 1) {
            (MultiThreadLlamaRotatePosition2DPartFloatOp(dataType, data, positionIds, sinData, cosData, bs, len, n, m, stride, spatial, posDim, rotaryDim, part, 0, bs * len)).Run();
            return;
        }

        int threadNum = pool->threads.size();
        int per = (bs * len) / pool->threads.size();
        int cur = 0;
        std::vector<fastllm::MultiThreadLlamaRotatePosition2DPartFloatOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? (bs * len) : cur + per + (cur + per * (threadNum - i) < (bs * len)));
            ops.push_back(new MultiThreadLlamaRotatePosition2DPartFloatOp(
                dataType, data, positionIds, sinData, cosData, bs, len, n, m, stride, spatial, posDim, rotaryDim, part, cur, end));
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

    void CpuLlamaRotatePosition2DPartOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        Data &sinData = *(datas.find("sin")->second);
        Data &cosData = *(datas.find("cos")->second);
        int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 128;
        int part = intParams.find("part") != intParams.end() ? intParams.find("part")->second : 128;

        int bs = data.dims[0], len = data.dims[1];
        int spatial = data.Count(2);
        int n = data.dims[2], m = data.dims[3];
        int stride = (int)sinData.dims[1];
        RunMultiThreadLlamaRotatePosition2DPartFloat(data.dataType, (float*)data.cpuData, (float*)positionIds.cpuData, 
            (float*)sinData.cpuData, (float*)cosData.cpuData, bs, len, n, m, stride, spatial, 
            positionIds.dims.back(), rotaryDim, part, GetAlivePool());
    }

    void CpuRepeatPenaltyOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                         const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &penalty = *(datas.find("penalty")->second);
        Data &penaltyScale = *(datas.find("penaltyScale")->second);
        AssertInFastLLM(input.dataType == DataType::FLOAT32 && penalty.dataType == DataType::FLOAT32 && penaltyScale.dataType == DataType::FLOAT32,
                        "Repeat Penalty error: Data's type should be float32.\n");
        float *inputData = (float*)input.cpuData;
        float *penaltyData = (float*)penalty.cpuData;
        float *penaltyScaleData = (float*)penaltyScale.cpuData;

        int batch = penalty.dims[0], tokens = penalty.dims[1];
        int vocabs = input.dims.back();
        for (int b = 0; b < batch; b++) {
            float scale = penaltyScaleData[b];
            for (int i = 0; i < tokens; i++) {
                int token = (int)(penaltyData[b * tokens + i] + 1e-6);
                if (token >= 0) {
                    int id = b * vocabs + token;
                    inputData[id] = inputData[id] < 0 ? inputData[id] * scale : inputData[id] / scale;
                }
            }
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

    void CpuCumSumLastDimOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);

        float *inputData = (float *) input.cpuData;

        std::vector <float> floatInputVector;
        if (input.dataType == DataType::FLOAT16) {
            floatInputVector.resize(input.Count(0));
            inputData = (float*)floatInputVector.data();
            Float16ToFloat32((uint16_t*)input.cpuData, inputData, (int)floatInputVector.size());
        }

        int dim = input.dims.back();
        int outer = input.Count(0) / dim;
        for (int o = 0; o < outer; o++) {
            for (int j = 1; j < dim; j++) {
                inputData[o * dim + j] += inputData[o * dim + j - 1];
            }
        }

        if (input.dataType == DataType::FLOAT16) {
            Float32ToFloat16(inputData, (uint16_t*)input.cpuData, (int)floatInputVector.size());
        }
    }

    void CpuMakeDecayMaskOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);

        AssertInFastLLM(input.dataType == DataType::FLOAT32 ||
                        input.dataType == DataType::FLOAT16, 
                        "CpuMakeDecayMaskOp's input's type should be float32 or float16.\n");

        std::vector <int> dims = input.dims;
        dims.push_back(dims.back());
        output.dataType = input.dataType;
        output.Resize(dims);
    }

    void CpuMakeDecayMaskOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        output.Allocate();

        int dim = input.dims.back();
        int outer = input.Count(0) / dim;

        float *inputData = (float *) input.cpuData;
        float *outputData = (float *) output.cpuData;

        std::vector <float> floatInputVector, floatOutputVector;
        if (input.dataType == DataType::FLOAT16) {
            floatInputVector.resize(input.Count(0));
            floatOutputVector.resize(output.Count(0));
            inputData = (float*)floatInputVector.data();
            outputData = (float*)floatOutputVector.data();
            Float16ToFloat32((uint16_t*)input.cpuData, inputData, (int)floatInputVector.size());
        }

        // decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
        for (int o = 0; o < outer; o++) {
            for (int i = 0; i < dim; i++) {
                for (int j = 0; j <= i && j < dim; j++) {
                    outputData[o * dim * dim + i * dim + j] = std::exp(inputData[o * dim + i] - inputData[o * dim + j]);
                }
                for (int j = i + 1; j < dim; j++) {
                    outputData[o * dim * dim + i * dim + j] = 0.0f;
                }
            }
        }

        if (input.dataType == DataType::FLOAT16) {
            Float32ToFloat16(outputData, (uint16_t*)output.cpuData, (int)floatOutputVector.size());
        }
    }

    void SiluMultiThread(float *input, int len, float *output,
                         int n, int inputStride, int outputStride, AliveThreadPool *pool) {
        int threadNum = pool->threads.size();
        int per = len / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadSiluOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? len : cur + per + (cur + per * (threadNum - i) < len));
            ops.push_back(new fastllm::MultiThreadSiluOp(input + cur, end - cur, output + cur,
                                                         n, inputStride, outputStride));
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

    void GeluMultiThread(float *input, int len, float *output,
                         int n, int inputStride, int outputStride, AliveThreadPool *pool) {
        int threadNum = pool->threads.size();
        int per = len / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadGeluOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? len : cur + per + (cur + per * (threadNum - i) < len));
            ops.push_back(new fastllm::MultiThreadGeluOp(input + cur, end - cur, output + cur,
                                                           n, inputStride, outputStride));
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

    void SwigluGptOssMultiThread(float *input, int mid, int len, float *output,
                           int n, int inputStride, int outputStride, AliveThreadPool *pool) {
        int threadNum = pool->threads.size();
        int per = len / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadSwigluGptOssOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? len : cur + per + (cur + per * (threadNum - i) < len));
            ops.push_back(new fastllm::MultiThreadSwigluGptOssOp(input + cur, mid, end - cur, output + cur,
                                                           n, inputStride, outputStride));
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

    void SwigluMultiThread(float *input, int mid, int len, float *output,
                           int n, int inputStride, int outputStride, AliveThreadPool *pool) {
        int threadNum = pool->threads.size();
        int per = len / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadSwigluOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? len : cur + per + (cur + per * (threadNum - i) < len));
            ops.push_back(new fastllm::MultiThreadSwigluOp(input + cur, mid, end - cur, output + cur,
                                                           n, inputStride, outputStride));
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

    void SwigluMultiThreadFloat16(uint16_t *input, int mid, int len, uint16_t *output,
                           int n, int inputStride, int outputStride, AliveThreadPool *pool) {
        int threadNum = pool->threads.size();
        int per = len / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadSwigluFloat16Op*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? len : cur + per + (cur + per * (threadNum - i) < len));
            ops.push_back(new fastllm::MultiThreadSwigluFloat16Op(input + cur, mid, end - cur, output + cur,
                                                           n, inputStride, outputStride));
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

    void SoftmaxMultiThread(float *input, int n, int m, int lastlen, AliveThreadPool *pool) {
        if (n == 1) {
            (MultiThreadSoftmaxOp(input, n, m, lastlen)).Run();
            return;
        }
        int threadNum = pool->threads.size();
        int per = n / threadNum;
        int cur = 0;
        std::vector<fastllm::MultiThreadSoftmaxOp*> ops;
        for (int i = 0; i < threadNum; i++) {
            int end = (i == threadNum - 1 ? n : cur + per + (cur + per * (threadNum - i) < n));
            ops.push_back(new fastllm::MultiThreadSoftmaxOp(input + cur * m, end - cur, m, lastlen + cur));
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

    void MultiThreadSoftmaxOp::Run() {
        for (int i = 0; i < n; i++) {
            float maxValue = -1e100;
            for (int j = 0; j < m; j++) {
                if (lastlen + i < j) {
                    value[i * m + j] = -10000;
                }
                maxValue = std::max(maxValue, value[i * m + j]);
            }
            float sum = 0.0;
            for (int j = 0; j < m; j++) {
                value[i * m + j] = expf(value[i * m + j] - maxValue);
                sum += value[i * m + j];
            }
            for (int j = 0; j < m; j++) {
                value[i * m + j] /= sum;
            }
        }
    }

    void MultiThreadSiluOp::Run() {
        for (int o = 0; o < n; o++) {
            float *cur = (float *) input + o * inputStride;
            float *out = (float *) output + o * outputStride;

            int i = 0;
#ifdef __aarch64__
            float32x4_t c1 = vdupq_n_f32(1.0f);
            for (; i + 3 < len; i += 4) {
                float32x4_t vx = vld1q_f32(cur + i);
                float32x4_t vdiv = vaddq_f32(c1, exp_ps(vnegq_f32(vx)));
                vx = vdivq_f32(vx, vdiv);
                vst1q_f32(out + i, vx);
            }
#endif
            for (; i < len; i++) {
                float x = cur[i];
                out[i] = x / (1.0 + expf(-x));
            }
        }
    }

    void MultiThreadGeluOp::Run() {
        for (int o = 0; o < n; o++) {
            float *cur = (float *) input + o * inputStride;
            float *out = (float *) output + o * outputStride;
            for (int i = 0; i < len; i++) {
                out[i] = gelu(cur[i]);
            }
        }
    }
    
    void MultiThreadSingleAttentionCausalOp::Run() {
        float *qk = new float[klen];
        float *temp = new float[klen];
        for (int i = 0; i < qlen; i++) {
            float maxValue = -10000, sum = 0.0;
            for (int j = 0; j < klen; j++) {
                if (lastlen + i < j) {
                    qk[j] = -10000;
                    continue;
                }

                float now = 0.0f;
                int l = 0;
#ifdef __aarch64__
                float32x4_t sum = {0, 0, 0, 0};
                for (; l + 3 < qdim; l += 4) {
                    sum = vaddq_f32(sum, vmulq_f32(vld1q_f32(qd + i * qdim + l),
                                                   vld1q_f32(kd + j * qdim + l)));
                }
                now += sum[0] + sum[1] + sum[2] + sum[3];
#endif
                for (; l < qdim; l++) {
                    now += qd[i * qdim + l] * kd[j * qdim + l];
                }
                qk[j] = now * scale;
                maxValue = std::max(maxValue, now * scale);
            }

            int j = 0;
#ifdef __aarch64__
            float32x4_t vmax = vdupq_n_f32(maxValue);
            for (; j + 3 < klen; j += 4) {
                vst1q_f32(temp + j, exp_ps(vsubq_f32(vld1q_f32(qk + j), vmax)));
            }
#endif
            for (; j < klen; j++) {
                temp[j] = expf(qk[j] - maxValue);
            }

            sum = 0.0f;
            for (j = 0; j < klen; j++) {
                sum += temp[j];
            }
            sum = std::max(sum, 0.1f);
            for (j = 0; j < klen; j++) {
                qk[j] = temp[j] / sum;
            }
            for (j = 0; j < klen; j++) {
                for (int l = 0; l < vdim; l++) {
                    od[i * vdim + l] += qk[j] * vd[j * vdim + l];
                }
            }
        }
        delete[] qk;
        delete[] temp;
    }
}