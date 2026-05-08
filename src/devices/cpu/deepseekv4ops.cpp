#include "devices/cpu/cpudevice.h"
#include "utils.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace fastllm {
    static float DeepSeekV4HcPreSigmoidFloat(float x) {
        if (x >= 0.0f) {
            float z = std::exp(-x);
            return 1.0f / (1.0f + z);
        }
        float z = std::exp(x);
        return z / (1.0f + z);
    }

    static bool DeepSeekV4HcPreEnvFlagEnabled(const char *name) {
        const char *v = std::getenv(name);
        return v != nullptr && v[0] != '\0' && strcmp(v, "0") != 0 &&
               strcmp(v, "false") != 0 && strcmp(v, "FALSE") != 0 &&
               strcmp(v, "off") != 0 && strcmp(v, "OFF") != 0;
    }

    static uint16_t DeepSeekV4HcPreFloatToBFloat16(float v) {
        uint32_t x;
        memcpy(&x, &v, sizeof(uint32_t));
        x += 0x7FFF + ((x >> 16) & 1);
        return (uint16_t)(x >> 16);
    }

    static float DeepSeekV4HcPreBFloat16ToFloat(uint16_t v) {
        uint32_t x = ((uint32_t)v) << 16;
        float ret;
        memcpy(&ret, &x, sizeof(float));
        return ret;
    }

    static std::vector<float> DeepSeekV4HcPreReadFloatData(Data &input) {
        uint64_t count = input.Count(0);
        std::vector<float> ret(count);
        if (count == 0) {
            return ret;
        }
        if (input.dataType == DataType::FLOAT32) {
            memcpy(ret.data(), input.cpuData, count * sizeof(float));
        } else if (input.dataType == DataType::FLOAT16) {
            uint16_t *src = (uint16_t*)input.cpuData;
            for (uint64_t i = 0; i < count; i++) {
                ret[i] = half_to_float(src[i]);
            }
        } else if (input.dataType == DataType::BFLOAT16) {
            uint16_t *src = (uint16_t*)input.cpuData;
            for (uint64_t i = 0; i < count; i++) {
                ret[i] = DeepSeekV4HcPreBFloat16ToFloat(src[i]);
            }
        } else {
            ErrorInFastLLM("DeepSeekV4HcPre error: unsupported input dtype.\n");
        }
        return ret;
    }

    static void DeepSeekV4HcPreWriteFloatData(const std::vector<float> &values, Data &output) {
        output.Allocate();
        if (output.dataType == DataType::FLOAT32) {
            memcpy(output.cpuData, values.data(), values.size() * sizeof(float));
        } else if (output.dataType == DataType::FLOAT16) {
            uint16_t *dst = (uint16_t*)output.cpuData;
            for (size_t i = 0; i < values.size(); i++) {
                dst[i] = float_to_half(values[i]);
            }
        } else if (output.dataType == DataType::BFLOAT16) {
            uint16_t *dst = (uint16_t*)output.cpuData;
            for (size_t i = 0; i < values.size(); i++) {
                dst[i] = DeepSeekV4HcPreFloatToBFloat16(values[i]);
            }
        } else {
            ErrorInFastLLM("DeepSeekV4HcPre error: unsupported output dtype.\n");
        }
    }

    static std::vector<float> ScaleQRatoryBuildInvFreq(int ropeDim, float base, int originalSeqLen,
                                                       float factor, int betaFast, int betaSlow) {
        std::vector<float> invFreq;
        for (int i = 0; i < ropeDim; i += 2) {
            invFreq.push_back(1.0f / std::pow(base, (float)i / ropeDim));
        }
        if (originalSeqLen > 0) {
            const float pi = 3.14159265358979323846f;
            float lowF = ropeDim * std::log((float)originalSeqLen / (betaFast * 2.0f * pi)) /
                         (2.0f * std::log(base));
            float highF = ropeDim * std::log((float)originalSeqLen / (betaSlow * 2.0f * pi)) /
                          (2.0f * std::log(base));
            int low = std::max((int)std::floor(lowF), 0);
            int high = std::min((int)std::ceil(highF), ropeDim - 1);
            if (low == high) {
                high++;
            }
            for (int idx = 0; idx < (int)invFreq.size(); idx++) {
                float ramp = std::min(1.0f, std::max(0.0f, ((float)idx - low) / (float)(high - low)));
                float smooth = 1.0f - ramp;
                invFreq[idx] = invFreq[idx] / factor * (1.0f - smooth) + invFreq[idx] * smooth;
            }
        }
        return invFreq;
    }

    void CpuScaleQRatoryOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        int ropeDim = intParams.find("ropeDim") != intParams.end() ? intParams.find("ropeDim")->second : 0;
        int startPos = intParams.find("startPos") != intParams.end() ? intParams.find("startPos")->second : 0;
        int originalSeqLen = intParams.find("originalSeqLen") != intParams.end() ? intParams.find("originalSeqLen")->second : 0;
        int betaFast = intParams.find("betaFast") != intParams.end() ? intParams.find("betaFast")->second : 32;
        int betaSlow = intParams.find("betaSlow") != intParams.end() ? intParams.find("betaSlow")->second : 1;
        float eps = floatParams.find("eps") != floatParams.end() ? floatParams.find("eps")->second : 1e-6f;
        float ropeBase = floatParams.find("ropeBase") != floatParams.end() ? floatParams.find("ropeBase")->second : 10000.0f;
        float ropeFactor = floatParams.find("ropeFactor") != floatParams.end() ? floatParams.find("ropeFactor")->second : 1.0f;

        AssertInFastLLM(q.dims.size() == 4, "ScaleQRatory error: q's shape's size should be 4.\n");
        int bsz = q.dims[0], seqlen = q.dims[1], heads = q.dims[2], dim = q.dims[3];
        AssertInFastLLM(ropeDim > 0 && ropeDim <= dim && ropeDim % 2 == 0,
                        "ScaleQRatory error: invalid ropeDim.\n");

        auto qv = DeepSeekV4HcPreReadFloatData(q);
        int rows = bsz * seqlen * heads;
        for (int r = 0; r < rows; r++) {
            float *row = qv.data() + (uint64_t)r * dim;
            double ss = 0.0;
            for (int d = 0; d < dim; d++) {
                ss += (double)row[d] * row[d];
            }
            float scale = 1.0f / std::sqrt((float)(ss / dim) + eps);
            for (int d = 0; d < dim; d++) {
                row[d] *= scale;
            }
        }

        int off = dim - ropeDim;
        auto invFreq = ScaleQRatoryBuildInvFreq(ropeDim, ropeBase, originalSeqLen, ropeFactor, betaFast, betaSlow);
        for (int b = 0; b < bsz; b++) {
            for (int s = 0; s < seqlen; s++) {
                int pos = startPos + s;
                for (int h = 0; h < heads; h++) {
                    float *row = qv.data() + (((uint64_t)b * seqlen + s) * heads + h) * dim + off;
                    for (int i = 0; i < ropeDim; i += 2) {
                        float ang = pos * invFreq[i / 2];
                        float c = std::cos(ang), sn = std::sin(ang);
                        float a = row[i], bb = row[i + 1];
                        row[i] = a * c - bb * sn;
                        row[i + 1] = a * sn + bb * c;
                    }
                }
            }
        }

        q.dataType = DataType::BFLOAT16;
        q.Resize({bsz, seqlen, heads, dim});
        DeepSeekV4HcPreWriteFloatData(qv, q);
    }

    struct DeepSeekV4HcPreDotsOp : MultiThreadBaseOp {
        const float *xrow;
        const float *fn;
        float *mixes;
        float rsqrt;
        int flatDim, mixSt, mixEnd;

        DeepSeekV4HcPreDotsOp(const float *xrow, const float *fn, float *mixes, float rsqrt,
                              int flatDim, int mixSt, int mixEnd)
            : xrow(xrow), fn(fn), mixes(mixes), rsqrt(rsqrt),
              flatDim(flatDim), mixSt(mixSt), mixEnd(mixEnd) {}

        void Run() override {
            for (int m = mixSt; m < mixEnd; m++) {
                double v = 0.0;
                const float *w = fn + (uint64_t)m * flatDim;
                for (int k = 0; k < flatDim; k++) {
                    v += (double)xrow[k] * w[k];
                }
                mixes[m] = (float)v * rsqrt;
            }
        }
    };

    static void DeepSeekV4HcPreComputeDotsCpu(const float *xrow, const float *fn, float *mixes,
                                              float rsqrt, int flatDim, int mixHc) {
        auto *pool = GetAlivePool();
        int threadNum = std::min((int)pool->threads.size(), mixHc);
        std::vector<DeepSeekV4HcPreDotsOp*> ops;
        int per = (mixHc + threadNum - 1) / threadNum;
        for (int i = 0; i < threadNum; i++) {
            int st = i * per;
            int end = std::min(mixHc, st + per);
            if (st >= end) {
                break;
            }
            ops.push_back(new DeepSeekV4HcPreDotsOp(xrow, fn, mixes, rsqrt, flatDim, st, end));
        }
        for (int i = 0; i < (int)ops.size(); i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < (int)ops.size(); i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    void CpuDeepSeekV4HcPreOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                       const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &hcFn = *(datas.find("hcFn")->second);
        Data &hcScale = *(datas.find("hcScale")->second);
        Data &hcBase = *(datas.find("hcBase")->second);
        Data &output = *(datas.find("output")->second);
        Data &post = *(datas.find("post")->second);
        Data &comb = *(datas.find("comb")->second);
        int hcMult = intParams.find("hcMult") != intParams.end() ? intParams.find("hcMult")->second : 1;

        AssertInFastLLM(input.dims.size() == 4, "DeepSeekV4HcPre error: input's shape's size should be 4.\n");
        int bsz = input.dims[0], seqlen = input.dims[1], dim = input.dims[3];
        int flatDim = hcMult * dim;
        int mixHc = (2 + hcMult) * hcMult;
        AssertInFastLLM(hcMult > 0 && input.dims[2] == hcMult,
                        "DeepSeekV4HcPre error: input's hc dimension mismatch.\n");
        AssertInFastLLM(hcFn.Count(0) == (uint64_t)mixHc * flatDim &&
                        hcScale.Count(0) >= 3 && hcBase.Count(0) >= (uint64_t)mixHc,
                        "DeepSeekV4HcPre error: weight shape mismatch.\n");
        AssertInFastLLM(hcScale.dataType == DataType::FLOAT32 && hcBase.dataType == DataType::FLOAT32,
                        "DeepSeekV4HcPre error: hcScale and hcBase should be float32.\n");

        output.dataType = input.dataType;
        output.Resize({bsz, seqlen, dim});
        post.dataType = DataType::FLOAT32;
        post.Resize({bsz, seqlen, hcMult});
        comb.dataType = DataType::FLOAT32;
        comb.Resize({bsz, seqlen, hcMult, hcMult});
    }

    void CpuDeepSeekV4HcPreOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                   const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &hcFn = *(datas.find("hcFn")->second);
        Data &hcScale = *(datas.find("hcScale")->second);
        Data &hcBase = *(datas.find("hcBase")->second);
        Data &output = *(datas.find("output")->second);
        Data &postData = *(datas.find("post")->second);
        Data &combData = *(datas.find("comb")->second);
        int hcMult = intParams.find("hcMult") != intParams.end() ? intParams.find("hcMult")->second : 1;
        int sinkhornIters = intParams.find("sinkhornIters") != intParams.end() ? intParams.find("sinkhornIters")->second : 1;
        float eps = floatParams.find("eps") != floatParams.end() ? floatParams.find("eps")->second : 1e-6f;
        float normEps = floatParams.find("normEps") != floatParams.end() ? floatParams.find("normEps")->second : 1e-6f;

        int bsz = input.dims[0], seqlen = input.dims[1], dim = input.dims[3];
        int flatDim = hcMult * dim;
        int mixHc = (2 + hcMult) * hcMult;
        int tokens = bsz * seqlen;
        auto xv = DeepSeekV4HcPreReadFloatData(input);
        auto fn = DeepSeekV4HcPreReadFloatData(hcFn);
        float *scale = (float*)hcScale.cpuData;
        float *base = (float*)hcBase.cpuData;
        std::vector<float> y((uint64_t)tokens * dim, 0.0f);
        std::vector<float> mixes(mixHc);
        std::vector<float> pre(hcMult);
        std::vector<float> combLocal(hcMult * hcMult);
        postData.Allocate();
        combData.Allocate();
        float *post = (float*)postData.cpuData;
        float *comb = (float*)combData.cpuData;

        for (int t = 0; t < tokens; t++) {
            const float *xrow = xv.data() + (uint64_t)t * flatDim;
            double ss = 0.0;
            for (int k = 0; k < flatDim; k++) {
                ss += (double)xrow[k] * xrow[k];
            }
            float rsqrt = 1.0f / std::sqrt((float)(ss / flatDim) + normEps);
            DeepSeekV4HcPreComputeDotsCpu(xrow, fn.data(), mixes.data(), rsqrt, flatDim, mixHc);
            for (int h = 0; h < hcMult; h++) {
                pre[h] = DeepSeekV4HcPreSigmoidFloat(mixes[h] * scale[0] + base[h]) + eps;
                post[(uint64_t)t * hcMult + h] =
                    2.0f * DeepSeekV4HcPreSigmoidFloat(mixes[h + hcMult] * scale[1] + base[h + hcMult]);
            }
            for (int r = 0; r < hcMult; r++) {
                float rowMax = -FLT_MAX;
                for (int c = 0; c < hcMult; c++) {
                    int idx = r * hcMult + c + 2 * hcMult;
                    combLocal[r * hcMult + c] = mixes[idx] * scale[2] + base[idx];
                    rowMax = std::max(rowMax, combLocal[r * hcMult + c]);
                }
                float rowSum = 0.0f;
                for (int c = 0; c < hcMult; c++) {
                    float v = std::exp(combLocal[r * hcMult + c] - rowMax);
                    combLocal[r * hcMult + c] = v;
                    rowSum += v;
                }
                for (int c = 0; c < hcMult; c++) {
                    combLocal[r * hcMult + c] = combLocal[r * hcMult + c] / rowSum + eps;
                }
            }
            for (int c = 0; c < hcMult; c++) {
                float colSum = 0.0f;
                for (int r = 0; r < hcMult; r++) {
                    colSum += combLocal[r * hcMult + c];
                }
                for (int r = 0; r < hcMult; r++) {
                    combLocal[r * hcMult + c] /= (colSum + eps);
                }
            }
            for (int it = 1; it < sinkhornIters; it++) {
                for (int r = 0; r < hcMult; r++) {
                    float rowSum = 0.0f;
                    for (int c = 0; c < hcMult; c++) {
                        rowSum += combLocal[r * hcMult + c];
                    }
                    for (int c = 0; c < hcMult; c++) {
                        combLocal[r * hcMult + c] /= (rowSum + eps);
                    }
                }
                for (int c = 0; c < hcMult; c++) {
                    float colSum = 0.0f;
                    for (int r = 0; r < hcMult; r++) {
                        colSum += combLocal[r * hcMult + c];
                    }
                    for (int r = 0; r < hcMult; r++) {
                        combLocal[r * hcMult + c] /= (colSum + eps);
                    }
                }
            }
            memcpy(comb + (uint64_t)t * hcMult * hcMult, combLocal.data(),
                   hcMult * hcMult * sizeof(float));
            for (int d = 0; d < dim; d++) {
                double v = 0.0;
                for (int h = 0; h < hcMult; h++) {
                    v += (double)pre[h] * xrow[(uint64_t)h * dim + d];
                }
                y[(uint64_t)t * dim + d] = (float)v;
            }
        }
        DeepSeekV4HcPreWriteFloatData(y, output);
    }
}
