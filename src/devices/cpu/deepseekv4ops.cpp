#include "devices/cpu/cpudevice.h"
#include "devices/cpu/computeutils.h"
#include "utils.h"

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
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

    namespace {
        struct DeepSeekV4ActivationQuantizationTask : MultiThreadBaseOp {
            const uint16_t *input;
            uint16_t *output;
            uint64_t blockBegin, blockEnd;

            DeepSeekV4ActivationQuantizationTask(
                    const uint16_t *input, uint16_t *output,
                    uint64_t blockBegin, uint64_t blockEnd)
                : input(input), output(output), blockBegin(blockBegin),
                  blockEnd(blockEnd) {}

            void Run() override {
                static const FP8E4M3ToFP32Manager fp8;
                for (uint64_t block = blockBegin; block < blockEnd; block++) {
                    const uint64_t start = block * 128;
                    float amax = 1e-4f;
                    for (uint64_t i = start; i < start + 128; i++) {
                        amax = std::max(
                            amax,
                            std::fabs(BFloat16BitsToFloat32(input[i])));
                    }
                    float normalized = amax / 448.0f;
                    uint32_t bits;
                    memcpy(&bits, &normalized, sizeof(bits));
                    int exponent = (int)((bits >> 23) & 0xFF) - 127 +
                                   ((bits & ((1u << 23) - 1)) != 0);
                    float scale = std::ldexp(1.0f, exponent);
                    const uint16_t *lookupRow =
                        GetFP8E4M3BFloat16QuantizeLookupRow(exponent);
                    if (lookupRow != nullptr) {
                        for (uint64_t i = start; i < start + 128; i++) {
                            output[i] = lookupRow[input[i]];
                        }
                    } else {
                        for (uint64_t i = start; i < start + 128; i++) {
                            float value = BFloat16BitsToFloat32(input[i]);
                            float q = std::max(
                                -448.0f,
                                std::min(448.0f, value / scale));
                            output[i] = Float32ToBFloat16RNEBits(
                                fp8.quantizeDequantize(q) * scale);
                        }
                    }
                }
            }
        };

        struct DeepSeekV4RMSNormBFloat16Task : MultiThreadBaseOp {
            const uint16_t *input;
            const float *weight;
            uint16_t *output;
            int rowBegin, rowEnd, channels;
            float eps;

            DeepSeekV4RMSNormBFloat16Task(
                    const uint16_t *input, const float *weight,
                    uint16_t *output, int rowBegin, int rowEnd,
                    int channels, float eps)
                : input(input), weight(weight), output(output),
                  rowBegin(rowBegin), rowEnd(rowEnd), channels(channels),
                  eps(eps) {}

            void Run() override {
                for (int row = rowBegin; row < rowEnd; row++) {
                    const uint16_t *source =
                        input + (uint64_t)row * channels;
                    uint16_t *destination =
                        output + (uint64_t)row * channels;
                    double sumSquares = 0.0;
                    for (int channel = 0; channel < channels; channel++) {
                        float value =
                            BFloat16BitsToFloat32(source[channel]);
                        sumSquares += (double)value * value;
                    }
                    float scale = 1.0f / std::sqrt(
                        (float)(sumSquares / channels) + eps);
                    for (int channel = 0; channel < channels; channel++) {
                        float value =
                            BFloat16BitsToFloat32(source[channel]);
                        destination[channel] = Float32ToBFloat16RNEBits(
                            value * scale * weight[channel]);
                    }
                }
            }
        };
    }

    void RunCpuDeepSeekV4ActivationQuantization(
            const uint16_t *input, uint16_t *output, uint64_t count) {
        AssertInFastLLM(
            input != nullptr && output != nullptr && count % 128 == 0,
            "DeepSeek-V4 CPU activation quantization received invalid storage.\n");
        const uint64_t blocks = count / 128;
        if (blocks == 0) {
            return;
        }
        AliveThreadPool *pool = GetAlivePool();
        const int firstThread = pool->curActivateThreadInterval.first;
        const int availableThreads = std::max(
            1, pool->curActivateThreadInterval.second - firstThread);
        // A worker gets at least 256 blocks (64 KiB of BF16 input+output), so
        // decode and short prefills remain inline while long prefills scale.
        const int threadCount = std::min<int>(
            availableThreads, std::max<uint64_t>(1, blocks / 256));
        if (threadCount <= 1) {
            DeepSeekV4ActivationQuantizationTask(
                input, output, 0, blocks).Run();
            return;
        }

        std::vector<DeepSeekV4ActivationQuantizationTask> tasks;
        tasks.reserve(threadCount);
        for (int thread = 0; thread < threadCount; thread++) {
            uint64_t begin = blocks * thread / threadCount;
            uint64_t end = blocks * (thread + 1) / threadCount;
            tasks.emplace_back(input, output, begin, end);
        }
        for (int thread = 0; thread < threadCount; thread++) {
            pool->PushOp(firstThread + thread, &tasks[thread]);
        }
        for (int thread = 0; thread < threadCount; thread++) {
            pool->Wait(firstThread + thread);
        }
    }

    void RunCpuDeepSeekV4RMSNormBFloat16(
            const uint16_t *input, const float *weight, uint16_t *output,
            int rows, int channels, float eps) {
        AssertInFastLLM(
            input != nullptr && weight != nullptr && output != nullptr &&
                rows > 0 && channels > 0,
            "DeepSeek-V4 CPU BF16 RMSNorm received invalid storage.\n");
        AliveThreadPool *pool = GetAlivePool();
        const int firstThread = pool->curActivateThreadInterval.first;
        const int availableThreads = std::max(
            1, pool->curActivateThreadInterval.second - firstThread);
        const uint64_t elements = (uint64_t)rows * channels;
        // Target at least 64K elements per worker.  A single-token norm stays
        // inline; the large hidden/Q/KV prefill norms use all useful workers.
        const int threadCount = std::min(
            std::min(availableThreads, rows),
            std::max<int>(1, elements / (64 * 1024)));
        if (threadCount <= 1) {
            DeepSeekV4RMSNormBFloat16Task(
                input, weight, output, 0, rows, channels, eps).Run();
            return;
        }

        std::vector<DeepSeekV4RMSNormBFloat16Task> tasks;
        tasks.reserve(threadCount);
        for (int thread = 0; thread < threadCount; thread++) {
            int begin = (int)((int64_t)rows * thread / threadCount);
            int end = (int)((int64_t)rows * (thread + 1) / threadCount);
            tasks.emplace_back(
                input, weight, output, begin, end, channels, eps);
        }
        for (int thread = 0; thread < threadCount; thread++) {
            pool->PushOp(firstThread + thread, &tasks[thread]);
        }
        for (int thread = 0; thread < threadCount; thread++) {
            pool->Wait(firstThread + thread);
        }
    }

    static void DeepSeekV4ConvertRawFloatData(
            const uint8_t *input, DataType dataType, float *output,
            uint64_t count) {
        if (dataType == DataType::FLOAT32) {
            memcpy(output, input, count * sizeof(float));
        } else if (dataType == DataType::FLOAT16) {
            const uint16_t *src = (const uint16_t*)input;
            for (uint64_t i = 0; i < count; i++) {
                output[i] = half_to_float(src[i]);
            }
        } else if (dataType == DataType::BFLOAT16) {
            const uint16_t *src = (const uint16_t*)input;
            uint64_t i = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
            for (; i + 16 <= count; i += 16) {
                __m256i packed = _mm256_loadu_si256(
                    (const __m256i*)(src + i));
                __m512i expanded = _mm512_cvtepu16_epi32(packed);
                expanded = _mm512_slli_epi32(expanded, 16);
                _mm512_storeu_si512(
                    (__m512i*)(output + i), expanded);
            }
#endif
            for (; i < count; i++) {
                output[i] = DeepSeekV4HcPreBFloat16ToFloat(src[i]);
            }
        } else {
            ErrorInFastLLM(
                "DeepSeek-V4 CPU op received an unsupported float dtype.\n");
        }
    }

    static void DeepSeekV4HcPreReadFloatDataInto(
            Data &input, std::vector<float> &ret) {
        uint64_t count = input.Count(0);
        ret.resize(count);
        if (count == 0) {
            return;
        }
        AssertInFastLLM(input.dataDevice == DataDevice::CPU && input.cpuData != nullptr,
                        "DeepSeek-V4 CPU op received a tensor without CPU storage.\n");
        DeepSeekV4ConvertRawFloatData(
            input.cpuData, input.dataType, ret.data(), count);
    }

    static std::vector<float> DeepSeekV4HcPreReadFloatData(Data &input) {
        std::vector<float> ret;
        DeepSeekV4HcPreReadFloatDataInto(input, ret);
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

    static void DeepSeekV4BuildRotaryCache(
            int seqlen, int ropeDim, float base, int startPos,
            int originalSeqLen, float factor, int betaFast, int betaSlow,
            int posStep, bool inverse, std::vector<float> &cosValues,
            std::vector<float> &sinValues) {
        auto invFreq = ScaleQRatoryBuildInvFreq(
            ropeDim, base, originalSeqLen, factor, betaFast, betaSlow);
        int pairs = ropeDim / 2;
        cosValues.resize((uint64_t)seqlen * pairs);
        sinValues.resize((uint64_t)seqlen * pairs);
        for (int s = 0; s < seqlen; s++) {
            int pos = startPos + s * posStep;
            for (int p = 0; p < pairs; p++) {
                float angle = pos * invFreq[p];
                cosValues[(uint64_t)s * pairs + p] = std::cos(angle);
                float sinValue = std::sin(angle);
                sinValues[(uint64_t)s * pairs + p] =
                    inverse ? -sinValue : sinValue;
            }
        }
    }

    static void DeepSeekV4ApplyRotaryReference(std::vector<float> &x, const std::vector<int> &dims,
                                               int ropeDim, float base, int startPos,
                                               int originalSeqLen, float factor,
                                               int betaFast, int betaSlow, int posStep = 1,
                                               bool inverse = false) {
        int bsz = dims[0], seqlen = dims[1];
        int heads = (dims.size() == 4) ? dims[2] : 1;
        int dim = (dims.size() == 4) ? dims[3] : dims[2];
        int off = dim - ropeDim;
        int pairs = ropeDim / 2;
        std::vector<float> cosValues, sinValues;
        DeepSeekV4BuildRotaryCache(
            seqlen, ropeDim, base, startPos, originalSeqLen, factor,
            betaFast, betaSlow, posStep, inverse, cosValues, sinValues);
        for (int b = 0; b < bsz; b++) {
            for (int s = 0; s < seqlen; s++) {
                for (int h = 0; h < heads; h++) {
                    uint64_t rowIndex = dims.size() == 4 ? (((uint64_t)b * seqlen + s) * heads + h)
                                                         : ((uint64_t)b * seqlen + s);
                    float *row = x.data() + rowIndex * dim + off;
                    for (int i = 0; i < ropeDim; i += 2) {
                        float c = cosValues[(uint64_t)s * pairs + i / 2];
                        float sn = sinValues[(uint64_t)s * pairs + i / 2];
                        float a = row[i], bb = row[i + 1];
                        row[i] = a * c - bb * sn;
                        row[i + 1] = a * sn + bb * c;
                    }
                }
            }
        }
    }

    static void DeepSeekV4ActQuantInplaceReference(std::vector<float> &x, const std::vector<int> &dims,
                                                   int quantDim, int blockSize) {
        int dim = dims.back();
        int rows = (int)(x.size() / dim);
        for (int r = 0; r < rows; r++) {
            float *row = x.data() + (uint64_t)r * dim;
            for (int start = 0; start < quantDim; start += blockSize) {
                int end = std::min(start + blockSize, quantDim);
                float amax = 1e-4f;
                for (int d = start; d < end; d++) {
                    amax = std::max(amax, std::fabs(row[d]));
                }
                float scale = std::pow(2.0f, std::ceil(std::log2(amax / 448.0f)));
                for (int d = start; d < end; d++) {
                    float q = std::max(-448.0f, std::min(448.0f, row[d] / scale));
                    row[d] = DeepSeekV4HcPreBFloat16ToFloat(DeepSeekV4HcPreFloatToBFloat16(q)) * scale;
                }
            }
        }
    }

    static inline float DeepSeekV4RoundBFloat16(float value) {
        return DeepSeekV4HcPreBFloat16ToFloat(
            DeepSeekV4HcPreFloatToBFloat16(value));
    }

    static inline void DeepSeekV4HadamardRowInplace(float *row, int dim,
                                                     float scale) {
        for (int width = 1; width < dim; width <<= 1) {
            for (int base = 0; base < dim; base += width << 1) {
                for (int i = 0; i < width; i++) {
                    float a = row[base + i];
                    float b = row[base + width + i];
                    row[base + i] = a + b;
                    row[base + width + i] = a - b;
                }
            }
        }
        for (int d = 0; d < dim; d++) {
            row[d] = DeepSeekV4RoundBFloat16(row[d] * scale);
        }
    }

    static inline float DeepSeekV4CeilPowerOfTwo(float value) {
        // The caller clamps value to the normal-float range.  For a normal
        // positive float, ceil(log2(value)) is its unbiased exponent plus one
        // exactly when the mantissa is non-zero.  Constructing the result's
        // exponent avoids libm log2/ceil/pow for every 32-value FP4 block.
        uint32_t bits;
        memcpy(&bits, &value, sizeof(bits));
        uint32_t powerBits = bits & 0x7F800000U;
        if ((bits & 0x007FFFFFU) != 0) {
            powerBits += 0x00800000U;
        }
        float result;
        memcpy(&result, &powerBits, sizeof(result));
        return result;
    }

    static inline float DeepSeekV4QuantizeDequantizeFp4(float value,
                                                         float scale) {
        static const float positiveValues[8] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
        };
        float normalized = std::min(6.0f, std::fabs(value / scale));
        int best = 0;
        float bestDistance = std::fabs(normalized - positiveValues[0]);
        for (int i = 1; i < 8; i++) {
            float distance = std::fabs(normalized - positiveValues[i]);
            if (distance < bestDistance ||
                (distance == bestDistance && (i & 1) == 0)) {
                best = i;
                bestDistance = distance;
            }
        }
        float quantized = std::signbit(value) ? -positiveValues[best]
                                               : positiveValues[best];
        return DeepSeekV4RoundBFloat16(quantized * scale);
    }

#if defined(__AVX512F__)
    static inline void DeepSeekV4Fp4ActQuantBlock32(float *row) {
        const __m512i absMask = _mm512_set1_epi32(0x7FFFFFFF);
        const __m512i signMask = _mm512_set1_epi32(0x80000000U);
        __m512 values0 = _mm512_loadu_ps(row);
        __m512 values1 = _mm512_loadu_ps(row + 16);
        __m512i bits0 = _mm512_castps_si512(values0);
        __m512i bits1 = _mm512_castps_si512(values1);
        __m512 abs0 = _mm512_castsi512_ps(_mm512_and_si512(bits0, absMask));
        __m512 abs1 = _mm512_castsi512_ps(_mm512_and_si512(bits1, absMask));
        float amax = std::max(
            6.0f * std::ldexp(1.0f, -126),
            _mm512_reduce_max_ps(_mm512_max_ps(abs0, abs1)));
        float scale = DeepSeekV4CeilPowerOfTwo(amax / 6.0f);
        __m512 inverseScale = _mm512_set1_ps(1.0f / scale);
        __m512 scaleVector = _mm512_set1_ps(scale);

        auto quantize = [&](const __m512 absolute, const __m512i bits) {
            __m512 normalized = _mm512_mul_ps(absolute, inverseScale);
            __m512 quantized = _mm512_set1_ps(6.0f);
            quantized = _mm512_mask_mov_ps(
                quantized,
                _mm512_cmp_ps_mask(normalized, _mm512_set1_ps(5.0f),
                                   _CMP_LE_OQ),
                _mm512_set1_ps(4.0f));
            quantized = _mm512_mask_mov_ps(
                quantized,
                _mm512_cmp_ps_mask(normalized, _mm512_set1_ps(3.5f),
                                   _CMP_LT_OQ),
                _mm512_set1_ps(3.0f));
            quantized = _mm512_mask_mov_ps(
                quantized,
                _mm512_cmp_ps_mask(normalized, _mm512_set1_ps(2.5f),
                                   _CMP_LE_OQ),
                _mm512_set1_ps(2.0f));
            quantized = _mm512_mask_mov_ps(
                quantized,
                _mm512_cmp_ps_mask(normalized, _mm512_set1_ps(1.75f),
                                   _CMP_LT_OQ),
                _mm512_set1_ps(1.5f));
            quantized = _mm512_mask_mov_ps(
                quantized,
                _mm512_cmp_ps_mask(normalized, _mm512_set1_ps(1.25f),
                                   _CMP_LE_OQ),
                _mm512_set1_ps(1.0f));
            quantized = _mm512_mask_mov_ps(
                quantized,
                _mm512_cmp_ps_mask(normalized, _mm512_set1_ps(0.75f),
                                   _CMP_LT_OQ),
                _mm512_set1_ps(0.5f));
            quantized = _mm512_mask_mov_ps(
                quantized,
                _mm512_cmp_ps_mask(normalized, _mm512_set1_ps(0.25f),
                                   _CMP_LE_OQ),
                _mm512_setzero_ps());
            __m512i quantizedBits = _mm512_castps_si512(quantized);
            quantizedBits = _mm512_xor_si512(
                quantizedBits, _mm512_and_si512(bits, signMask));
            return _mm512_mul_ps(
                _mm512_castsi512_ps(quantizedBits), scaleVector);
        };

        _mm512_storeu_ps(row, quantize(abs0, bits0));
        _mm512_storeu_ps(row + 16, quantize(abs1, bits1));
    }
#endif

    static inline void DeepSeekV4Fp4ActQuantRowInplace(float *row, int dim,
                                                       int blockSize,
                                                       bool useFp4Simd) {
        for (int start = 0; start < dim; start += blockSize) {
#if defined(__AVX512F__)
            if (useFp4Simd && blockSize == 32) {
                DeepSeekV4Fp4ActQuantBlock32(row + start);
                continue;
            }
#else
            (void)useFp4Simd;
#endif
            float amax = 6.0f * std::ldexp(1.0f, -126);
            for (int d = start; d < start + blockSize; d++) {
                amax = std::max(amax, std::fabs(row[d]));
            }
            float scale = std::pow(
                2.0f, std::ceil(std::log2(amax / 6.0f)));
            for (int d = start; d < start + blockSize; d++) {
                row[d] = DeepSeekV4QuantizeDequantizeFp4(row[d], scale);
            }
        }
    }

    static inline void DeepSeekV4PrepareIndexerRow(
            float *row, const float *cosRow, const float *sinRow,
            int dim, int ropeDim, float hadamardScale, bool useFp4Simd) {
        int ropeOffset = dim - ropeDim;
        int ropePairs = ropeDim / 2;
        for (int pair = 0; pair < ropePairs; pair++) {
            int d = ropeOffset + pair * 2;
            float a = row[d], b = row[d + 1];
            row[d] = a * cosRow[pair] - b * sinRow[pair];
            row[d + 1] = a * sinRow[pair] + b * cosRow[pair];
        }
        for (int d = 0; d < dim; d++) {
            row[d] = DeepSeekV4RoundBFloat16(row[d]);
        }
        DeepSeekV4HadamardRowInplace(row, dim, hadamardScale);
        DeepSeekV4Fp4ActQuantRowInplace(row, dim, 32, useFp4Simd);
    }

    struct DeepSeekV4PrepareIndexerRowsTask : MultiThreadBaseOp {
        float *values;
        const float *ropeCos;
        const float *ropeSin;
        int totalRows, seqlen, heads, dim, ropeDim;
        int taskId, taskCount;
        bool useFp4Simd;

        DeepSeekV4PrepareIndexerRowsTask(
            float *values, const float *ropeCos, const float *ropeSin,
            int totalRows, int seqlen, int heads, int dim, int ropeDim,
            int taskId, int taskCount, bool useFp4Simd
        ) : values(values), ropeCos(ropeCos), ropeSin(ropeSin),
            totalRows(totalRows), seqlen(seqlen), heads(heads), dim(dim),
            ropeDim(ropeDim), taskId(taskId), taskCount(taskCount),
            useFp4Simd(useFp4Simd) {}

        void Run() override {
            int ropePairs = ropeDim / 2;
            float hadamardScale = 1.0f / std::sqrt((float)dim);
            int rowBegin = (int)((int64_t)totalRows * taskId / taskCount);
            int rowEnd = (int)((int64_t)totalRows * (taskId + 1) /
                               taskCount);
            for (int rowIdx = rowBegin; rowIdx < rowEnd; rowIdx++) {
                int seq = (rowIdx / heads) % seqlen;
                float *row = values + (uint64_t)rowIdx * dim;
                const float *cosRow = ropeCos + (uint64_t)seq * ropePairs;
                const float *sinRow = ropeSin + (uint64_t)seq * ropePairs;
                DeepSeekV4PrepareIndexerRow(
                    row, cosRow, sinRow, dim, ropeDim, hadamardScale,
                    useFp4Simd);
            }
        }
    };

    static void DeepSeekV4PrepareIndexerRows(std::vector<float> &values,
                                             const std::vector<int> &dims,
                                             int ropeDim, float ropeBase,
                                             int startPos, int originalSeqLen,
                                             float ropeFactor, int betaFast,
                                             int betaSlow, int posStep) {
        int bsz = dims[0], seqlen = dims[1];
        int heads = dims.size() == 4 ? dims[2] : 1;
        int dim = dims.back();
        int totalRows = bsz * seqlen * heads;
        AssertInFastLLM(
            dim > 0 && (dim & (dim - 1)) == 0 && dim % 32 == 0,
            "DeepSeekV4 indexer preprocessing requires a power-of-two "
            "dimension divisible by 32.\n");
        bool useFp4Simd = !DeepSeekV4HcPreEnvFlagEnabled(
            "FASTLLM_DSV4_DISABLE_CPU_INDEXER_FP4_SIMD");
        std::vector<float> ropeCos, ropeSin;
        DeepSeekV4BuildRotaryCache(
            seqlen, ropeDim, ropeBase, startPos, originalSeqLen,
            ropeFactor, betaFast, betaSlow, posStep, false,
            ropeCos, ropeSin);
        AliveThreadPool *pool = GetAlivePool();
        int firstThread = pool->curActivateThreadInterval.first;
        int availableThreads = std::max(
            1, pool->curActivateThreadInterval.second - firstThread);
        int threadCount = std::min(availableThreads, totalRows);
        if (threadCount <= 1 || totalRows < 64) {
            DeepSeekV4PrepareIndexerRowsTask(
                values.data(), ropeCos.data(), ropeSin.data(), totalRows,
                seqlen, heads, dim, ropeDim, 0, 1, useFp4Simd).Run();
            return;
        }
        std::vector<DeepSeekV4PrepareIndexerRowsTask*> tasks;
        tasks.reserve(threadCount);
        for (int thread = 0; thread < threadCount; thread++) {
            tasks.push_back(new DeepSeekV4PrepareIndexerRowsTask(
                values.data(), ropeCos.data(), ropeSin.data(), totalRows,
                seqlen, heads, dim, ropeDim, thread, threadCount,
                useFp4Simd));
            pool->PushOp(firstThread + thread, tasks.back());
        }
        for (int thread = 0; thread < threadCount; thread++) {
            pool->Wait(firstThread + thread);
            delete tasks[thread];
        }
    }

    struct DeepSeekV4ScaleQRatoryBFloat16RowsTask : MultiThreadBaseOp {
        uint16_t *values;
        const float *ropeCos;
        const float *ropeSin;
        int totalRows, seqlen, heads, dim, ropeDim;
        int taskId, taskCount;
        float eps;

        DeepSeekV4ScaleQRatoryBFloat16RowsTask(
            uint16_t *values, const float *ropeCos, const float *ropeSin,
            int totalRows, int seqlen, int heads, int dim, int ropeDim,
            int taskId, int taskCount, float eps
        ) : values(values), ropeCos(ropeCos), ropeSin(ropeSin),
            totalRows(totalRows), seqlen(seqlen), heads(heads), dim(dim),
            ropeDim(ropeDim), taskId(taskId), taskCount(taskCount), eps(eps) {}

        void Run() override {
            int rowBegin = (int)((int64_t)totalRows * taskId / taskCount);
            int rowEnd = (int)((int64_t)totalRows * (taskId + 1) / taskCount);
            int ropeOffset = dim - ropeDim;
            int ropePairs = ropeDim / 2;
            for (int rowIdx = rowBegin; rowIdx < rowEnd; rowIdx++) {
                uint16_t *row = values + (uint64_t)rowIdx * dim;
                double squareSum = 0.0;
                for (int d = 0; d < dim; d++) {
                    float value = DeepSeekV4HcPreBFloat16ToFloat(row[d]);
                    squareSum += (double)value * value;
                }
                float scale = 1.0f /
                    std::sqrt((float)(squareSum / dim) + eps);

                for (int d = 0; d < ropeOffset; d++) {
                    float value =
                        DeepSeekV4HcPreBFloat16ToFloat(row[d]) * scale;
                    row[d] = DeepSeekV4HcPreFloatToBFloat16(value);
                }

                int sequence = (rowIdx / heads) % seqlen;
                const float *cosRow =
                    ropeCos + (uint64_t)sequence * ropePairs;
                const float *sinRow =
                    ropeSin + (uint64_t)sequence * ropePairs;
                for (int pair = 0; pair < ropePairs; pair++) {
                    int d = ropeOffset + pair * 2;
                    float a =
                        DeepSeekV4HcPreBFloat16ToFloat(row[d]) * scale;
                    float b =
                        DeepSeekV4HcPreBFloat16ToFloat(row[d + 1]) * scale;
                    float rotatedA =
                        a * cosRow[pair] - b * sinRow[pair];
                    float rotatedB =
                        a * sinRow[pair] + b * cosRow[pair];
                    row[d] = DeepSeekV4HcPreFloatToBFloat16(rotatedA);
                    row[d + 1] =
                        DeepSeekV4HcPreFloatToBFloat16(rotatedB);
                }
            }
        }
    };

    static void DeepSeekV4ScaleQRatoryBFloat16Rows(
            uint16_t *values, int totalRows, int seqlen, int heads, int dim,
            int ropeDim, float eps, const std::vector<float> &ropeCos,
            const std::vector<float> &ropeSin) {
        AliveThreadPool *pool = GetAlivePool();
        int firstThread = pool->curActivateThreadInterval.first;
        int availableThreads = std::max(
            1, pool->curActivateThreadInterval.second - firstThread);
        // Keep decode and very short prefills inline.  Each worker receives at
        // least 64 complete rows so pool dispatch never dominates the work.
        int threadCount = std::min(
            availableThreads, std::max(1, totalRows / 64));
        if (threadCount <= 1) {
            DeepSeekV4ScaleQRatoryBFloat16RowsTask(
                values, ropeCos.data(), ropeSin.data(), totalRows, seqlen,
                heads, dim, ropeDim, 0, 1, eps).Run();
            return;
        }

        std::vector<DeepSeekV4ScaleQRatoryBFloat16RowsTask*> tasks;
        tasks.reserve(threadCount);
        for (int thread = 0; thread < threadCount; thread++) {
            tasks.push_back(new DeepSeekV4ScaleQRatoryBFloat16RowsTask(
                values, ropeCos.data(), ropeSin.data(), totalRows, seqlen,
                heads, dim, ropeDim, thread, threadCount, eps));
            pool->PushOp(firstThread + thread, tasks.back());
        }
        for (int thread = 0; thread < threadCount; thread++) {
            pool->Wait(firstThread + thread);
            delete tasks[thread];
        }
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

        if (q.dataType == DataType::BFLOAT16) {
            AssertInFastLLM(
                q.dataDevice == DataDevice::CPU && q.cpuData != nullptr,
                "ScaleQRatory CPU BF16 input has no CPU storage.\n");
            std::vector<float> ropeCos, ropeSin;
            DeepSeekV4BuildRotaryCache(
                seqlen, ropeDim, ropeBase, startPos, originalSeqLen,
                ropeFactor, betaFast, betaSlow, 1, false,
                ropeCos, ropeSin);
            DeepSeekV4ScaleQRatoryBFloat16Rows(
                (uint16_t*)q.cpuData, bsz * seqlen * heads, seqlen, heads,
                dim, ropeDim, eps, ropeCos, ropeSin);
            return;
        }

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

        DeepSeekV4ApplyRotaryReference(qv, q.dims, ropeDim, ropeBase, startPos,
                                       originalSeqLen, ropeFactor, betaFast, betaSlow);

        q.dataType = DataType::BFLOAT16;
        q.Resize({bsz, seqlen, heads, dim});
        DeepSeekV4HcPreWriteFloatData(qv, q);
    }

    void CpuDeepSeekV4RotaryQuantOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                         const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        int ropeDim = intParams.find("ropeDim") != intParams.end() ? intParams.find("ropeDim")->second : 0;
        int startPos = intParams.find("startPos") != intParams.end() ? intParams.find("startPos")->second : 0;
        int originalSeqLen = intParams.find("originalSeqLen") != intParams.end() ? intParams.find("originalSeqLen")->second : 0;
        int betaFast = intParams.find("betaFast") != intParams.end() ? intParams.find("betaFast")->second : 32;
        int betaSlow = intParams.find("betaSlow") != intParams.end() ? intParams.find("betaSlow")->second : 1;
        int quantDim = intParams.find("quantDim") != intParams.end() ? intParams.find("quantDim")->second : 0;
        int blockSize = intParams.find("blockSize") != intParams.end() ? intParams.find("blockSize")->second : 64;
        int posStep = intParams.find("posStep") != intParams.end() ? intParams.find("posStep")->second : 1;
        float ropeBase = floatParams.find("ropeBase") != floatParams.end() ? floatParams.find("ropeBase")->second : 10000.0f;
        float ropeFactor = floatParams.find("ropeFactor") != floatParams.end() ? floatParams.find("ropeFactor")->second : 1.0f;

        AssertInFastLLM(input.dims.size() >= 3 && input.dims.size() <= 4,
                        "DeepSeekV4RotaryQuant error: input's shape's size should be 3 or 4.\n");
        int dim = input.dims.back();
        AssertInFastLLM(ropeDim > 0 && ropeDim <= dim && ropeDim % 2 == 0,
                        "DeepSeekV4RotaryQuant error: invalid ropeDim.\n");
        AssertInFastLLM(quantDim >= 0 && quantDim <= dim && blockSize > 0,
                        "DeepSeekV4RotaryQuant error: invalid quant params.\n");

        std::vector<int> dims = input.dims;
        auto values = DeepSeekV4HcPreReadFloatData(input);
        DeepSeekV4ApplyRotaryReference(values, dims, ropeDim, ropeBase, startPos,
                                       originalSeqLen, ropeFactor, betaFast, betaSlow, posStep);
        DeepSeekV4ActQuantInplaceReference(values, dims, quantDim, blockSize);

        input.dataType = DataType::BFLOAT16;
        input.Resize(dims);
        DeepSeekV4HcPreWriteFloatData(values, input);
    }

    static inline double DeepSeekV4SparseAttentionDot(
            const float *a, const float *b, int len) {
        static const bool disableSimd = DeepSeekV4HcPreEnvFlagEnabled(
            "FASTLLM_DSV4_DISABLE_CPU_SPARSE_PREFILL_SIMD");
        int i = 0;
        double sum = 0.0;
        if (!disableSimd) {
#if defined(__AVX512F__)
            __m512d sum0 = _mm512_setzero_pd();
            __m512d sum1 = _mm512_setzero_pd();
            __m512d sum2 = _mm512_setzero_pd();
            __m512d sum3 = _mm512_setzero_pd();
            for (; i + 31 < len; i += 32) {
                sum0 = _mm512_add_pd(
                    sum0,
                    _mm512_mul_pd(
                        _mm512_cvtps_pd(_mm256_loadu_ps(a + i)),
                        _mm512_cvtps_pd(_mm256_loadu_ps(b + i))));
                sum1 = _mm512_add_pd(
                    sum1,
                    _mm512_mul_pd(
                        _mm512_cvtps_pd(_mm256_loadu_ps(a + i + 8)),
                        _mm512_cvtps_pd(_mm256_loadu_ps(b + i + 8))));
                sum2 = _mm512_add_pd(
                    sum2,
                    _mm512_mul_pd(
                        _mm512_cvtps_pd(_mm256_loadu_ps(a + i + 16)),
                        _mm512_cvtps_pd(_mm256_loadu_ps(b + i + 16))));
                sum3 = _mm512_add_pd(
                    sum3,
                    _mm512_mul_pd(
                        _mm512_cvtps_pd(_mm256_loadu_ps(a + i + 24)),
                        _mm512_cvtps_pd(_mm256_loadu_ps(b + i + 24))));
            }
            sum += _mm512_reduce_add_pd(_mm512_add_pd(
                _mm512_add_pd(sum0, sum1),
                _mm512_add_pd(sum2, sum3)));
#elif defined(__AVX2__)
            __m256d sum0 = _mm256_setzero_pd();
            __m256d sum1 = _mm256_setzero_pd();
            __m256d sum2 = _mm256_setzero_pd();
            __m256d sum3 = _mm256_setzero_pd();
            for (; i + 15 < len; i += 16) {
                sum0 = _mm256_add_pd(
                    sum0,
                    _mm256_mul_pd(
                        _mm256_cvtps_pd(_mm_loadu_ps(a + i)),
                        _mm256_cvtps_pd(_mm_loadu_ps(b + i))));
                sum1 = _mm256_add_pd(
                    sum1,
                    _mm256_mul_pd(
                        _mm256_cvtps_pd(_mm_loadu_ps(a + i + 4)),
                        _mm256_cvtps_pd(_mm_loadu_ps(b + i + 4))));
                sum2 = _mm256_add_pd(
                    sum2,
                    _mm256_mul_pd(
                        _mm256_cvtps_pd(_mm_loadu_ps(a + i + 8)),
                        _mm256_cvtps_pd(_mm_loadu_ps(b + i + 8))));
                sum3 = _mm256_add_pd(
                    sum3,
                    _mm256_mul_pd(
                        _mm256_cvtps_pd(_mm_loadu_ps(a + i + 12)),
                        _mm256_cvtps_pd(_mm_loadu_ps(b + i + 12))));
            }
            __m256d total = _mm256_add_pd(
                _mm256_add_pd(sum0, sum1),
                _mm256_add_pd(sum2, sum3));
            __m128d halves = _mm_add_pd(
                _mm256_castpd256_pd128(total),
                _mm256_extractf128_pd(total, 1));
            sum += _mm_cvtsd_f64(_mm_hadd_pd(halves, halves));
#endif
        }
        for (; i < len; i++) {
            sum += (double)a[i] * b[i];
        }
        return sum;
    }

    struct DeepSeekV4SparseAttentionCpuParams {
        int bsz, seqlen, heads, dim, kvSeqLen, ropeDim;
        int windowSize, startPos, compressRatio;
        int realPrefixLen, compressedStart, compressedCount, prefixStartPos;
        const int32_t *compressedTopK;
        int compressedTopKWidth;
        float softmaxScale;
    };

    struct DeepSeekV4SparseAttentionTokenTask : MultiThreadBaseOp {
        const uint8_t *q;
        DataType qDataType;
        const float *kv;
        const float *sink;
        uint16_t *output;
        const float *ropeCos;
        const float *ropeSin;
        const DeepSeekV4SparseAttentionCpuParams *params;
        int taskId, taskCount;

        DeepSeekV4SparseAttentionTokenTask(
            const uint8_t *q, DataType qDataType, const float *kv,
            const float *sink, uint16_t *output, const float *ropeCos,
            const float *ropeSin,
            const DeepSeekV4SparseAttentionCpuParams *params,
            int taskId, int taskCount
        ) : q(q), qDataType(qDataType), kv(kv), sink(sink), output(output),
            ropeCos(ropeCos), ropeSin(ropeSin), params(params), taskId(taskId),
            taskCount(taskCount) {}

        void Run() override {
            const auto &p = *params;
            int totalTokens = p.bsz * p.seqlen;
            int maxWindow = std::min(
                p.windowSize, p.realPrefixLen + p.seqlen);
            std::vector<float> scores;
            std::vector<double> scoreExps;
            int maxCompressedCandidates = p.compressedTopK == nullptr ?
                p.compressedCount : p.compressedTopKWidth;
            scores.reserve(std::max(0, maxWindow + maxCompressedCandidates));
            scoreExps.reserve(std::max(0, maxWindow + maxCompressedCandidates));
            std::vector<int> selectedCompressed;
            selectedCompressed.reserve(std::max(0, maxCompressedCandidates));
            std::vector<float> qScratch(p.dim);
            std::vector<float> outputRow(p.dim);
            for (int token = taskId; token < totalTokens; token += taskCount) {
                int s = token % p.seqlen;
                int b = token / p.seqlen;
                int liveWindow = std::min(
                    p.windowSize, p.realPrefixLen + s + 1);
                int beginPos = p.startPos + s - liveWindow + 1;
                int availableCompressed = p.compressRatio > 0 ?
                    std::min(p.compressedCount,
                             std::max(0, (p.startPos + s + 1) /
                                         p.compressRatio)) :
                    0;
                selectedCompressed.clear();
                if (p.compressedTopK != nullptr) {
                    const int32_t *topKRow = p.compressedTopK +
                        (uint64_t)token * p.compressedTopKWidth;
                    for (int i = 0; i < p.compressedTopKWidth; i++) {
                        int idx = topKRow[i];
                        if (idx >= 0 && idx < availableCompressed) {
                            selectedCompressed.push_back(idx);
                        }
                    }
                } else {
                    selectedCompressed.resize(availableCompressed);
                    std::iota(selectedCompressed.begin(),
                              selectedCompressed.end(), 0);
                }
                int selectedCount = (int)selectedCompressed.size();
                int candidateCount = liveWindow + selectedCount;
                scores.resize(candidateCount);
                scoreExps.resize(candidateCount);

                for (int head = 0; head < p.heads; head++) {
                    int flatHead = token * p.heads + head;
                    uint64_t qOffset = (uint64_t)flatHead * p.dim;
                    const float *qrow = nullptr;
                    if (qDataType == DataType::FLOAT32) {
                        qrow = (const float*)q + qOffset;
                    } else if (qDataType == DataType::BFLOAT16) {
                        const uint16_t *src = (const uint16_t*)q + qOffset;
                        for (int d = 0; d < p.dim; d++) {
                            qScratch[d] =
                                DeepSeekV4HcPreBFloat16ToFloat(src[d]);
                        }
                        qrow = qScratch.data();
                    } else {
                        const uint16_t *src = (const uint16_t*)q + qOffset;
                        for (int d = 0; d < p.dim; d++) {
                            qScratch[d] = half_to_float(src[d]);
                        }
                        qrow = qScratch.data();
                    }
                    float maxScore = -std::numeric_limits<float>::infinity();
                    for (int k = 0; k < liveWindow; k++) {
                        int pos = beginPos + k;
                        int kvIndex = pos < p.startPos ?
                            pos - p.prefixStartPos :
                            p.realPrefixLen + pos - p.startPos;
                        const float *kvrow =
                            kv + ((uint64_t)b * p.kvSeqLen + kvIndex) * p.dim;
                        scores[k] = (float)DeepSeekV4SparseAttentionDot(
                            qrow, kvrow, p.dim) * p.softmaxScale;
                        maxScore = std::max(maxScore, scores[k]);
                    }
                    for (int k = 0; k < selectedCount; k++) {
                        int compressedIndex = selectedCompressed[k];
                        const float *kvrow =
                            kv + ((uint64_t)b * p.kvSeqLen +
                                  p.compressedStart + compressedIndex) * p.dim;
                        int scoreIndex = liveWindow + k;
                        scores[scoreIndex] =
                            (float)DeepSeekV4SparseAttentionDot(
                                qrow, kvrow, p.dim) * p.softmaxScale;
                        maxScore = std::max(maxScore, scores[scoreIndex]);
                    }

                    float safeMax =
                        std::isfinite(maxScore) ? maxScore : 0.0f;
                    double denominator =
                        std::exp((double)sink[head] - safeMax);
                    for (int k = 0; k < candidateCount; k++) {
                        if (!std::isfinite(scores[k])) {
                            scoreExps[k] = 0.0;
                            continue;
                        }
                        scoreExps[k] =
                            std::exp((double)scores[k] - safeMax);
                        denominator += scoreExps[k];
                    }
                    denominator = std::max(denominator, 1e-30);

                    std::fill(outputRow.begin(), outputRow.end(), 0.0f);
                    for (int k = 0; k < liveWindow; k++) {
                        if (!std::isfinite(scores[k])) {
                            continue;
                        }
                        int pos = beginPos + k;
                        int kvIndex = pos < p.startPos ?
                            pos - p.prefixStartPos :
                            p.realPrefixLen + pos - p.startPos;
                        const float *kvrow =
                            kv + ((uint64_t)b * p.kvSeqLen + kvIndex) * p.dim;
                        float weight = (float)(scoreExps[k] / denominator);
                        for (int d = 0; d < p.dim; d++) {
                            outputRow[d] += weight * kvrow[d];
                        }
                    }
                    for (int k = 0; k < selectedCount; k++) {
                        int scoreIndex = liveWindow + k;
                        if (!std::isfinite(scores[scoreIndex])) {
                            continue;
                        }
                        int compressedIndex = selectedCompressed[k];
                        const float *kvrow =
                            kv + ((uint64_t)b * p.kvSeqLen +
                                  p.compressedStart + compressedIndex) * p.dim;
                        float weight =
                            (float)(scoreExps[scoreIndex] / denominator);
                        for (int d = 0; d < p.dim; d++) {
                            outputRow[d] += weight * kvrow[d];
                        }
                    }

                    int ropeOffset = p.dim - p.ropeDim;
                    int ropePairs = p.ropeDim / 2;
                    const float *cosRow =
                        ropeCos + (uint64_t)s * ropePairs;
                    const float *sinRow =
                        ropeSin + (uint64_t)s * ropePairs;
                    for (int pair = 0; pair < ropePairs; pair++) {
                        int d = ropeOffset + pair * 2;
                        float c = cosRow[pair], sn = sinRow[pair];
                        float a = outputRow[d], bb = outputRow[d + 1];
                        outputRow[d] = a * c - bb * sn;
                        outputRow[d + 1] = a * sn + bb * c;
                    }

                    uint16_t *outputBits =
                        output + (uint64_t)flatHead * p.dim;
                    for (int d = 0; d < p.dim; d++) {
                        outputBits[d] =
                            DeepSeekV4HcPreFloatToBFloat16(outputRow[d]);
                    }
                }
            }
        }
    };

    void CpuDeepSeekV4SparseAttentionOp::Reshape(
            const std::string &opType, const fastllm::DataDict &datas,
            const fastllm::FloatDict &floatParams,
            const fastllm::IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &output = *(datas.find("output")->second);
        output.dataType = DataType::BFLOAT16;
        output.Resize(q.dims);
    }

    void CpuDeepSeekV4SparseAttentionOp::Run(
            const std::string &opType, const fastllm::DataDict &datas,
            const fastllm::FloatDict &floatParams,
            const fastllm::IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &kv = *(datas.find("kv")->second);
        Data &attnSink = *(datas.find("attnSink")->second);
        Data &output = *(datas.find("output")->second);
        auto compressedTopKIt = datas.find("compressedTopK");
        Data *compressedTopK = compressedTopKIt == datas.end() ?
            nullptr : compressedTopKIt->second;
        auto getInt = [&](const char *name, int fallback) {
            auto it = intParams.find(name);
            return it == intParams.end() ? fallback : it->second;
        };
        auto getFloat = [&](const char *name, float fallback) {
            auto it = floatParams.find(name);
            return it == floatParams.end() ? fallback : it->second;
        };

        int windowSize = getInt("windowSize", 0);
        int ropeDim = getInt("ropeDim", 0);
        int startPos = getInt("startPos", 0);
        int compressRatio = getInt("compressRatio", 0);
        int originalSeqLen = getInt("originalSeqLen", 0);
        int betaFast = getInt("betaFast", 32);
        int betaSlow = getInt("betaSlow", 1);
        int prefixLen = getInt("prefixLen", 0);
        float ropeBase = getFloat("ropeBase", 10000.0f);
        float softmaxScale = getFloat("softmaxScale", 1.0f);
        float ropeFactor = getFloat("ropeFactor", 1.0f);

        AssertInFastLLM(q.dims.size() == 4 && kv.dims.size() == 3,
                        "DeepSeekV4SparseAttention error: invalid q or kv rank.\n");
        int bsz = q.dims[0], seqlen = q.dims[1];
        int heads = q.dims[2], dim = q.dims[3];
        AssertInFastLLM(
            kv.dims[0] == bsz && kv.dims[1] >= seqlen &&
                kv.dims[2] == dim &&
                attnSink.Count(0) >= (uint64_t)heads,
            "DeepSeekV4SparseAttention error: q, kv or sink shape mismatch.\n");
        AssertInFastLLM(
            windowSize > 0 && ropeDim > 0 && ropeDim <= dim &&
                ropeDim % 2 == 0 && compressRatio >= 0,
            "DeepSeekV4SparseAttention error: invalid attention params.\n");
        AssertInFastLLM(
            q.dataDevice == DataDevice::CPU && q.cpuData != nullptr &&
                (q.dataType == DataType::FLOAT32 ||
                 q.dataType == DataType::FLOAT16 ||
                 q.dataType == DataType::BFLOAT16),
            "DeepSeekV4SparseAttention error: unsupported q storage.\n");

        auto kvValues = DeepSeekV4HcPreReadFloatData(kv);
        auto sinkValues = DeepSeekV4HcPreReadFloatData(attnSink);
        std::vector<float> ropeCos, ropeSin;
        DeepSeekV4BuildRotaryCache(
            seqlen, ropeDim, ropeBase, startPos, originalSeqLen, ropeFactor,
            betaFast, betaSlow, 1, true, ropeCos, ropeSin);
        int realPrefixLen = std::max(
            0, std::min(prefixLen, kv.dims[1] - seqlen));
        const int32_t *compressedTopKData = nullptr;
        int compressedTopKWidth = 0;
        if (compressedTopK != nullptr && compressedTopK->Count(0) > 0) {
            AssertInFastLLM(
                compressedTopK->dataDevice == DataDevice::CPU &&
                    compressedTopK->dataType == DataType::INT32 &&
                    compressedTopK->dims.size() == 3 &&
                    compressedTopK->dims[0] == bsz &&
                    compressedTopK->dims[1] == seqlen &&
                    compressedTopK->cpuData != nullptr,
                "DeepSeekV4SparseAttention error: invalid compressed top-k indices.\n");
            compressedTopKData = (const int32_t*)compressedTopK->cpuData;
            compressedTopKWidth = compressedTopK->dims[2];
        }
        DeepSeekV4SparseAttentionCpuParams params = {
            bsz, seqlen, heads, dim, kv.dims[1], ropeDim, windowSize,
            startPos, compressRatio, realPrefixLen, realPrefixLen + seqlen,
            std::max(0, kv.dims[1] - realPrefixLen - seqlen),
            startPos - realPrefixLen, compressedTopKData,
            compressedTopKWidth, softmaxScale
        };
        output.dataType = DataType::BFLOAT16;
        output.Resize(q.dims);
        output.Allocate(false);
        uint16_t *outputBits = (uint16_t*)output.cpuData;

        AliveThreadPool *pool = GetAlivePool();
        int firstThread = pool->curActivateThreadInterval.first;
        int availableThreads = std::max(
            1, pool->curActivateThreadInterval.second - firstThread);
        int totalTokens = bsz * seqlen;
        int threadCount = std::min(availableThreads, totalTokens);
        if (threadCount > 1 &&
            !DeepSeekV4HcPreEnvFlagEnabled(
                "FASTLLM_DSV4_DISABLE_CPU_SPARSE_PREFILL_PARALLEL")) {
            std::vector<DeepSeekV4SparseAttentionTokenTask*> tasks;
            tasks.reserve(threadCount);
            for (int thread = 0; thread < threadCount; thread++) {
                tasks.push_back(new DeepSeekV4SparseAttentionTokenTask(
                    q.cpuData, q.dataType, kvValues.data(), sinkValues.data(),
                    outputBits, ropeCos.data(), ropeSin.data(), &params,
                    thread, threadCount));
                pool->PushOp(firstThread + thread, tasks.back());
            }
            for (int thread = 0; thread < threadCount; thread++) {
                pool->Wait(firstThread + thread);
                delete tasks[thread];
            }
        } else {
            DeepSeekV4SparseAttentionTokenTask(
                q.cpuData, q.dataType, kvValues.data(), sinkValues.data(),
                outputBits, ropeCos.data(), ropeSin.data(), &params,
                0, 1).Run();
        }
    }

    struct DeepSeekV4SparseAttentionDecodeCachedHeadsTask
            : MultiThreadBaseOp {
        const float *q;
        const float *sink;
        const float *const *kvRows;
        float *output;
        int flatHeadStart, flatHeadEnd;
        int heads, dim, candidateCount;
        float softmaxScale;

        DeepSeekV4SparseAttentionDecodeCachedHeadsTask(
                const float *q, const float *sink,
                const float *const *kvRows, float *output,
                int flatHeadStart, int flatHeadEnd, int heads, int dim,
                int candidateCount, float softmaxScale)
            : q(q), sink(sink), kvRows(kvRows), output(output),
              flatHeadStart(flatHeadStart), flatHeadEnd(flatHeadEnd),
              heads(heads), dim(dim), candidateCount(candidateCount),
              softmaxScale(softmaxScale) {}

        void Run() override {
            std::vector<float> scores(candidateCount);
            std::vector<double> scoreExps(candidateCount);
            for (int flatHead = flatHeadStart;
                 flatHead < flatHeadEnd; flatHead++) {
                int batch = flatHead / heads;
                int head = flatHead % heads;
                const float *qrow = q + (uint64_t)flatHead * dim;
                const float *const *batchRows =
                    kvRows + (uint64_t)batch * candidateCount;
                float maxScore =
                    -std::numeric_limits<float>::infinity();
                for (int k = 0; k < candidateCount; k++) {
                    const float *kvrow = batchRows[k];
                    double dot = 0.0;
                    // Keep the reference decode reduction order.  Heads are
                    // independent, so parallelizing them does not alter this
                    // per-output floating-point path.
                    for (int d = 0; d < dim; d++) {
                        dot += (double)qrow[d] * kvrow[d];
                    }
                    scores[k] = (float)dot * softmaxScale;
                    maxScore = std::max(maxScore, scores[k]);
                }

                float safeMax =
                    std::isfinite(maxScore) ? maxScore : 0.0f;
                double denominator =
                    std::exp((double)sink[head] - safeMax);
                for (int k = 0; k < candidateCount; k++) {
                    scoreExps[k] =
                        std::exp((double)scores[k] - safeMax);
                    denominator += scoreExps[k];
                }
                double safeDenominator =
                    std::max(denominator, 1e-30);
                float *outputRow =
                    output + (uint64_t)flatHead * dim;
                for (int k = 0; k < candidateCount; k++) {
                    float weight =
                        (float)(scoreExps[k] / safeDenominator);
                    const float *kvrow = batchRows[k];
                    for (int d = 0; d < dim; d++) {
                        outputRow[d] += weight * kvrow[d];
                    }
                }
            }
        }
    };

#if defined(__AVX512F__) && defined(__AVX2__) && defined(__FMA__)
    struct DeepSeekV4SparseAttentionDecodeCachedScoreBlocksTask
            : MultiThreadBaseOp {
        static constexpr int blockHeads = 8;

        const float *qTransposed;
        const float *const *kvRows;
        float *scores;
        int unitBegin, unitEnd;
        int heads, dim, candidateCount;
        float softmaxScale;

        DeepSeekV4SparseAttentionDecodeCachedScoreBlocksTask(
                const float *qTransposed, const float *const *kvRows,
                float *scores, int unitBegin, int unitEnd,
                int heads, int dim, int candidateCount,
                float softmaxScale)
            : qTransposed(qTransposed), kvRows(kvRows), scores(scores),
              unitBegin(unitBegin), unitEnd(unitEnd), heads(heads),
              dim(dim), candidateCount(candidateCount),
              softmaxScale(softmaxScale) {}

        void Run() override {
            const int blocksPerBatch = heads / blockHeads;
            for (int unit = unitBegin; unit < unitEnd; unit++) {
                const int block = unit / candidateCount;
                const int candidate = unit % candidateCount;
                const int batch = block / blocksPerBatch;
                const int headStart =
                    (block % blocksPerBatch) * blockHeads;
                const float *qBlock = qTransposed +
                    (uint64_t)block * dim * blockHeads;
                const float *kvrow = kvRows[
                    (uint64_t)batch * candidateCount + candidate];
                __m512d dots = _mm512_setzero_pd();
                for (int d = 0; d < dim; d++) {
                    __m512d qHeads = _mm512_cvtps_pd(
                        _mm256_loadu_ps(
                            qBlock + (uint64_t)d * blockHeads));
                    __m512d kvValue = _mm512_set1_pd(
                        (double)kvrow[d]);
                    constexpr int rounding =
                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
                    __m512d products = _mm512_mul_round_pd(
                        qHeads, kvValue, rounding);
                    dots = _mm512_add_round_pd(
                        dots, products, rounding);
                }
                __m256 scoreValues = _mm256_mul_ps(
                    _mm512_cvtpd_ps(dots),
                    _mm256_set1_ps(softmaxScale));
                _mm256_storeu_ps(
                    scores +
                        ((uint64_t)batch * candidateCount + candidate) *
                            heads + headStart,
                    scoreValues);
            }
        }
    };

    struct DeepSeekV4SparseAttentionDecodeCachedHeadBlocksTask
            : MultiThreadBaseOp {
        static constexpr int blockHeads = 8;

        struct Workspace {
            std::vector<float> scores;
            std::vector<double> scoreExps;
            std::vector<float> weights;
            std::vector<float> blockOutput;
        };

        const float *qTransposed;
        const float *sink;
        const float *const *kvRows;
        const float *precomputedScores;
        float *output;
        int taskId, taskCount;
        int bsz, heads, dim, candidateCount;
        float softmaxScale;

        DeepSeekV4SparseAttentionDecodeCachedHeadBlocksTask(
                const float *qTransposed, const float *sink,
                const float *const *kvRows,
                const float *precomputedScores, float *output,
                int taskId, int taskCount, int bsz, int heads, int dim,
                int candidateCount, float softmaxScale)
            : qTransposed(qTransposed), sink(sink), kvRows(kvRows),
              precomputedScores(precomputedScores), output(output),
              taskId(taskId), taskCount(taskCount),
              bsz(bsz), heads(heads), dim(dim),
              candidateCount(candidateCount),
              softmaxScale(softmaxScale) {}

        void Run() override {
            int blocksPerBatch = heads / blockHeads;
            int totalBlocks = bsz * blocksPerBatch;
            static thread_local Workspace workspace;
            workspace.scores.resize(
                (uint64_t)candidateCount * blockHeads);
            workspace.scoreExps.resize(
                (uint64_t)candidateCount * blockHeads);
            workspace.weights.resize(
                (uint64_t)candidateCount * blockHeads);
            workspace.blockOutput.resize((uint64_t)dim * blockHeads);
            float *scores = workspace.scores.data();
            double *scoreExps = workspace.scoreExps.data();
            float *weights = workspace.weights.data();
            float *blockOutput = workspace.blockOutput.data();

            for (int block = taskId; block < totalBlocks;
                 block += taskCount) {
                int batch = block / blocksPerBatch;
                int headStart =
                    (block % blocksPerBatch) * blockHeads;
                const float *qBlock = qTransposed +
                    (uint64_t)block * dim * blockHeads;
                const float *const *batchRows = kvRows +
                    (uint64_t)batch * candidateCount;
                float maxScores[blockHeads];
                std::fill(
                    maxScores, maxScores + blockHeads,
                    -std::numeric_limits<float>::infinity());

                if (precomputedScores == nullptr) {
                    for (int k = 0; k < candidateCount; k++) {
                        const float *kvrow = batchRows[k];
                        __m512d dots = _mm512_setzero_pd();
                        // Each AVX-512 lane is one head.  The d loop remains
                        // in the exact scalar order, and multiplication/add
                        // are deliberately separate to match the reference.
                        for (int d = 0; d < dim; d++) {
                            __m512d qHeads = _mm512_cvtps_pd(
                                _mm256_loadu_ps(
                                    qBlock +
                                        (uint64_t)d * blockHeads));
                            __m512d kvValue = _mm512_set1_pd(
                                (double)kvrow[d]);
                            constexpr int rounding =
                                _MM_FROUND_TO_NEAREST_INT |
                                _MM_FROUND_NO_EXC;
                            __m512d products = _mm512_mul_round_pd(
                                qHeads, kvValue, rounding);
                            dots = _mm512_add_round_pd(
                                dots, products, rounding);
                        }
                        __m256 scoreValues = _mm256_mul_ps(
                            _mm512_cvtpd_ps(dots),
                            _mm256_set1_ps(softmaxScale));
                        float *scoreRow =
                            scores + (uint64_t)k * blockHeads;
                        _mm256_storeu_ps(scoreRow, scoreValues);
                        for (int lane = 0; lane < blockHeads; lane++) {
                            maxScores[lane] =
                                std::max(maxScores[lane], scoreRow[lane]);
                        }
                    }
                } else {
                    for (int k = 0; k < candidateCount; k++) {
                        const float *scoreRow = precomputedScores +
                            ((uint64_t)batch * candidateCount + k) *
                                heads + headStart;
                        for (int lane = 0; lane < blockHeads; lane++) {
                            maxScores[lane] = std::max(
                                maxScores[lane], scoreRow[lane]);
                        }
                    }
                }

                for (int lane = 0; lane < blockHeads; lane++) {
                    float safeMax = std::isfinite(maxScores[lane]) ?
                        maxScores[lane] : 0.0f;
                    double denominator = std::exp(
                        (double)sink[headStart + lane] - safeMax);
                    for (int k = 0; k < candidateCount; k++) {
                        uint64_t index =
                            (uint64_t)k * blockHeads + lane;
                        float score = precomputedScores == nullptr ?
                            scores[index] :
                            precomputedScores[
                                ((uint64_t)batch * candidateCount + k) *
                                    heads + headStart + lane];
                        scoreExps[index] = std::exp(
                            (double)score - safeMax);
                        denominator += scoreExps[index];
                    }
                    double safeDenominator =
                        std::max(denominator, 1e-30);
                    for (int k = 0; k < candidateCount; k++) {
                        uint64_t index =
                            (uint64_t)k * blockHeads + lane;
                        weights[index] = (float)(
                            scoreExps[index] / safeDenominator);
                    }
                }

                std::fill(
                    blockOutput, blockOutput + (uint64_t)dim * blockHeads,
                    0.0f);
                for (int k = 0; k < candidateCount; k++) {
                    __m256 weightValues = _mm256_loadu_ps(
                        weights + (uint64_t)k * blockHeads);
                    const float *kvrow = batchRows[k];
                    for (int d = 0; d < dim; d++) {
                        float *outputHeads = blockOutput +
                            (uint64_t)d * blockHeads;
                        _mm256_storeu_ps(
                            outputHeads,
                            _mm256_fmadd_ps(
                                weightValues,
                                _mm256_set1_ps(kvrow[d]),
                                _mm256_loadu_ps(outputHeads)));
                    }
                }

                for (int lane = 0; lane < blockHeads; lane++) {
                    float *outputRow = output +
                        ((uint64_t)batch * heads + headStart + lane) * dim;
                    for (int d = 0; d < dim; d++) {
                        outputRow[d] = blockOutput[
                            (uint64_t)d * blockHeads + lane];
                    }
                }
            }
        }
    };
#endif

    struct DeepSeekV4SparseAttentionDecodeCachedWorkspace {
        std::vector<float> qStorage;
        std::vector<float> sinkStorage;
        std::vector<float> convertedKVRows;
        std::vector<float> outputValues;
        std::vector<float> qTransposed;
        std::vector<float> precomputedScores;
        std::vector<int> indices;
        std::vector<const float*> kvRows;
    };

    void CpuDeepSeekV4SparseAttentionDecodeCachedOp::Reshape(
            const std::string &opType, const fastllm::DataDict &datas,
            const fastllm::FloatDict &floatParams,
            const fastllm::IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &output = *(datas.find("output")->second);
        output.dataType = DataType::BFLOAT16;
        output.Resize(q.dims);
    }

    void CpuDeepSeekV4SparseAttentionDecodeCachedOp::Run(
            const std::string &opType, const fastllm::DataDict &datas,
            const fastllm::FloatDict &floatParams,
            const fastllm::IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &windowKV = *(datas.find("windowKV")->second);
        Data &compressedKV = *(datas.find("compressedKV")->second);
        Data &attnSink = *(datas.find("attnSink")->second);
        Data &output = *(datas.find("output")->second);
        auto topKIt = datas.find("compressedTopK");
        Data *compressedTopK =
            topKIt == datas.end() ? nullptr : topKIt->second;
        auto getInt = [&](const char *name, int fallback) {
            auto it = intParams.find(name);
            return it == intParams.end() ? fallback : it->second;
        };
        auto getFloat = [&](const char *name, float fallback) {
            auto it = floatParams.find(name);
            return it == floatParams.end() ? fallback : it->second;
        };

        int windowSize = getInt("windowSize", 0);
        int startPos = getInt("startPos", 0);
        int compressedCount = getInt("compressedCount", 0);
        int ropeDim = getInt("ropeDim", 0);
        int originalSeqLen = getInt("originalSeqLen", 0);
        int betaFast = getInt("betaFast", 32);
        int betaSlow = getInt("betaSlow", 1);
        float ropeBase = getFloat("ropeBase", 10000.0f);
        float softmaxScale = getFloat("softmaxScale", 1.0f);
        float ropeFactor = getFloat("ropeFactor", 1.0f);

        AssertInFastLLM(
            q.dims.size() == 4 && q.dims[1] == 1 &&
                windowKV.dims.size() == 3,
            "DeepSeekV4SparseAttentionDecodeCached error: invalid q or "
            "window rank.\n");
        int bsz = q.dims[0], heads = q.dims[2], dim = q.dims[3];
        AssertInFastLLM(
            bsz > 0 && heads > 0 && dim > 0 && windowSize > 0 &&
                startPos >= 0 && compressedCount >= 0 &&
                windowKV.dims[0] == bsz &&
                windowKV.dims[1] >= windowSize &&
                windowKV.dims[2] == dim &&
                ropeDim > 0 && ropeDim <= dim && ropeDim % 2 == 0 &&
                attnSink.Count(0) >= (uint64_t)heads,
            "DeepSeekV4SparseAttentionDecodeCached error: shape or "
            "parameter mismatch.\n");
        AssertInFastLLM(
            q.dataDevice == DataDevice::CPU && q.cpuData != nullptr &&
                windowKV.dataDevice == DataDevice::CPU &&
                windowKV.cpuData != nullptr &&
                attnSink.dataDevice == DataDevice::CPU &&
                attnSink.cpuData != nullptr,
            "DeepSeekV4SparseAttentionDecodeCached error: tensors need CPU "
            "storage.\n");
        auto supportedFloatType = [](DataType type) {
            return type == DataType::FLOAT32 ||
                   type == DataType::FLOAT16 ||
                   type == DataType::BFLOAT16;
        };
        AssertInFastLLM(
            supportedFloatType(q.dataType) &&
                supportedFloatType(windowKV.dataType) &&
                supportedFloatType(attnSink.dataType),
            "DeepSeekV4SparseAttentionDecodeCached error: unsupported "
            "tensor dtype.\n");
        if (compressedCount > 0) {
            AssertInFastLLM(
                compressedKV.dims.size() == 3 &&
                    compressedKV.dims[0] == bsz &&
                    compressedKV.dims[1] >= compressedCount &&
                    compressedKV.dims[2] == dim &&
                    compressedKV.dataDevice == DataDevice::CPU &&
                    compressedKV.cpuData != nullptr &&
                    supportedFloatType(compressedKV.dataType),
                "DeepSeekV4SparseAttentionDecodeCached error: invalid "
                "compressed cache.\n");
        }

        static thread_local
            DeepSeekV4SparseAttentionDecodeCachedWorkspace workspace;
        const float *qValues;
        if (q.dataType == DataType::FLOAT32) {
            qValues = (const float*)q.cpuData;
        } else {
            DeepSeekV4HcPreReadFloatDataInto(q, workspace.qStorage);
            qValues = workspace.qStorage.data();
        }
        const float *sinkValues;
        if (attnSink.dataType == DataType::FLOAT32) {
            sinkValues = (const float*)attnSink.cpuData;
        } else {
            DeepSeekV4HcPreReadFloatDataInto(
                attnSink, workspace.sinkStorage);
            sinkValues = workspace.sinkStorage.data();
        }

        std::vector<int> &indices = workspace.indices;
        indices.clear();
        indices.reserve(windowSize + compressedCount);
        if (startPos >= windowSize - 1) {
            int ringPosition = startPos % windowSize;
            for (int i = ringPosition + 1; i < windowSize; i++) {
                indices.push_back(i);
            }
            for (int i = 0; i <= ringPosition; i++) {
                indices.push_back(i);
            }
        } else {
            for (int i = 0; i <= startPos; i++) {
                indices.push_back(i);
            }
        }
        if (compressedTopK != nullptr &&
            compressedTopK->Count(0) > 0) {
            AssertInFastLLM(
                compressedTopK->dataDevice == DataDevice::CPU &&
                    compressedTopK->dataType == DataType::INT32 &&
                    compressedTopK->cpuData != nullptr,
                "DeepSeekV4SparseAttentionDecodeCached error: invalid "
                "compressed top-k indices.\n");
            const int32_t *topKData =
                (const int32_t*)compressedTopK->cpuData;
            int topKCount = (int)compressedTopK->Count(0);
            for (int i = 0; i < topKCount; i++) {
                int index = topKData[i];
                if (index >= 0 && index < compressedCount) {
                    indices.push_back(windowSize + index);
                }
            }
        } else {
            for (int i = 0; i < compressedCount; i++) {
                indices.push_back(windowSize + i);
            }
        }

        int candidateCount = (int)indices.size();
        AssertInFastLLM(
            candidateCount > 0,
            "DeepSeekV4SparseAttentionDecodeCached error: no attention "
            "candidates.\n");

        size_t convertedRowCount = 0;
        for (int batch = 0; batch < bsz; batch++) {
            for (int k = 0; k < candidateCount; k++) {
                const Data &source = indices[k] < windowSize ?
                    windowKV : compressedKV;
                convertedRowCount +=
                    source.dataType == DataType::FLOAT32 ? 0 : 1;
            }
        }
        workspace.convertedKVRows.resize(
            convertedRowCount * (size_t)dim);
        std::vector<const float*> &kvRows = workspace.kvRows;
        kvRows.resize((uint64_t)bsz * candidateCount);
        size_t convertedOffset = 0;
        for (int batch = 0; batch < bsz; batch++) {
            for (int k = 0; k < candidateCount; k++) {
                int index = indices[k];
                const bool fromWindow = index < windowSize;
                const Data &source = fromWindow ? windowKV : compressedKV;
                const int sourceRows = fromWindow ?
                    windowSize : compressedCount;
                const int sourceRow = fromWindow ?
                    index : index - windowSize;
                const uint64_t sourceOffset =
                    ((uint64_t)batch * sourceRows + sourceRow) * dim;
                const float *row;
                if (source.dataType == DataType::FLOAT32) {
                    row = (const float*)source.cpuData + sourceOffset;
                } else {
                    float *converted =
                        workspace.convertedKVRows.data() + convertedOffset;
                    DeepSeekV4ConvertRawFloatData(
                        source.cpuData + sourceOffset * sizeof(uint16_t),
                        source.dataType, converted, dim);
                    row = converted;
                    convertedOffset += dim;
                }
                kvRows[(uint64_t)batch * candidateCount + k] = row;
            }
        }

        std::vector<float> &outputValues = workspace.outputValues;
        outputValues.resize((uint64_t)bsz * heads * dim);
        std::fill(outputValues.begin(), outputValues.end(), 0.0f);
        AliveThreadPool *pool = GetAlivePool();
        int firstThread = pool->curActivateThreadInterval.first;
        int availableThreads = std::max(
            1, pool->curActivateThreadInterval.second - firstThread);
        int totalHeads = bsz * heads;
        bool ranHeadBlocks = false;
#if defined(__AVX512F__) && defined(__AVX2__) && defined(__FMA__)
        if (heads %
                DeepSeekV4SparseAttentionDecodeCachedHeadBlocksTask::
                    blockHeads == 0) {
            constexpr int blockHeads =
                DeepSeekV4SparseAttentionDecodeCachedHeadBlocksTask::
                    blockHeads;
            int totalBlocks = bsz * heads / blockHeads;
            int threadCount = std::min(availableThreads, totalBlocks);
            std::vector<float> &qTransposed = workspace.qTransposed;
            qTransposed.resize((uint64_t)totalHeads * dim);
            for (int batch = 0; batch < bsz; batch++) {
                for (int headBlock = 0;
                     headBlock < heads / blockHeads; headBlock++) {
                    int block = batch * (heads / blockHeads) + headBlock;
                    for (int d = 0; d < dim; d++) {
                        for (int lane = 0; lane < blockHeads; lane++) {
                            int head = headBlock * blockHeads + lane;
                            qTransposed[
                                ((uint64_t)block * dim + d) * blockHeads +
                                lane] = qValues[
                                    ((uint64_t)batch * heads + head) * dim +
                                    d];
                        }
                    }
                }
            }
            const float *precomputedScores = nullptr;
            if (candidateCount >= 256 && availableThreads > totalBlocks &&
                !DeepSeekV4HcPreEnvFlagEnabled(
                    "FASTLLM_DSV4_DISABLE_CPU_SPARSE_SCORE_SPLIT")) {
                const int totalScoreUnits = totalBlocks * candidateCount;
                const int scoreThreads = std::min(
                    {availableThreads, 40, totalScoreUnits});
                workspace.precomputedScores.resize(
                    (uint64_t)bsz * candidateCount * heads);
                static thread_local std::vector<
                    DeepSeekV4SparseAttentionDecodeCachedScoreBlocksTask>
                    scoreTasks;
                scoreTasks.clear();
                scoreTasks.reserve(scoreThreads);
                for (int thread = 0; thread < scoreThreads; thread++) {
                    int begin = (int)(
                        (int64_t)totalScoreUnits * thread / scoreThreads);
                    int end = (int)(
                        (int64_t)totalScoreUnits * (thread + 1) /
                        scoreThreads);
                    scoreTasks.emplace_back(
                        qTransposed.data(), kvRows.data(),
                        workspace.precomputedScores.data(), begin, end,
                        heads, dim, candidateCount, softmaxScale);
                    pool->PushOp(
                        firstThread + thread, &scoreTasks.back());
                }
                for (int thread = 0; thread < scoreThreads; thread++) {
                    pool->Wait(firstThread + thread);
                }
                precomputedScores = workspace.precomputedScores.data();
            }
            if (threadCount > 1) {
                static thread_local std::vector<
                    DeepSeekV4SparseAttentionDecodeCachedHeadBlocksTask>
                    tasks;
                tasks.clear();
                tasks.reserve(threadCount);
                for (int thread = 0; thread < threadCount; thread++) {
                    tasks.emplace_back(
                        qTransposed.data(), sinkValues, kvRows.data(),
                        precomputedScores, outputValues.data(), thread,
                        threadCount, bsz, heads, dim, candidateCount,
                        softmaxScale);
                    pool->PushOp(firstThread + thread, &tasks.back());
                }
                for (int thread = 0; thread < threadCount; thread++) {
                    pool->Wait(firstThread + thread);
                }
            } else {
                DeepSeekV4SparseAttentionDecodeCachedHeadBlocksTask(
                    qTransposed.data(), sinkValues, kvRows.data(),
                    precomputedScores, outputValues.data(), 0, 1, bsz,
                    heads, dim, candidateCount, softmaxScale).Run();
            }
            ranHeadBlocks = true;
        }
#endif
        int threadCount = std::min(availableThreads, totalHeads);
        if (!ranHeadBlocks && threadCount > 1) {
            static thread_local std::vector<
                DeepSeekV4SparseAttentionDecodeCachedHeadsTask> tasks;
            tasks.clear();
            tasks.reserve(threadCount);
            for (int thread = 0; thread < threadCount; thread++) {
                int begin = (int)((int64_t)totalHeads * thread /
                                  threadCount);
                int end = (int)((int64_t)totalHeads * (thread + 1) /
                                threadCount);
                tasks.emplace_back(
                    qValues, sinkValues, kvRows.data(),
                    outputValues.data(), begin, end, heads, dim,
                    candidateCount, softmaxScale);
                pool->PushOp(firstThread + thread, &tasks.back());
            }
            for (int thread = 0; thread < threadCount; thread++) {
                pool->Wait(firstThread + thread);
            }
        } else if (!ranHeadBlocks) {
            DeepSeekV4SparseAttentionDecodeCachedHeadsTask(
                qValues, sinkValues, kvRows.data(), outputValues.data(),
                0, totalHeads, heads, dim, candidateCount,
                softmaxScale).Run();
        }

        DeepSeekV4ApplyRotaryReference(
            outputValues, {bsz, 1, heads, dim}, ropeDim, ropeBase,
            startPos, originalSeqLen, ropeFactor, betaFast, betaSlow, 1,
            true);
        output.dataType = DataType::BFLOAT16;
        output.Resize({bsz, 1, heads, dim});
        DeepSeekV4HcPreWriteFloatData(outputValues, output);
    }

    static inline float DeepSeekV4IndexerDot(const float *a, const float *b,
                                              int len) {
        int i = 0;
        float sum = 0.0f;
#if defined(__AVX512F__)
        __m512 sum0 = _mm512_setzero_ps();
        __m512 sum1 = _mm512_setzero_ps();
        for (; i + 31 < len; i += 32) {
            sum0 = _mm512_add_ps(
                sum0, _mm512_mul_ps(_mm512_loadu_ps(a + i),
                                    _mm512_loadu_ps(b + i)));
            sum1 = _mm512_add_ps(
                sum1, _mm512_mul_ps(_mm512_loadu_ps(a + i + 16),
                                    _mm512_loadu_ps(b + i + 16)));
        }
        sum += _mm512_reduce_add_ps(_mm512_add_ps(sum0, sum1));
#elif defined(__AVX2__)
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        for (; i + 15 < len; i += 16) {
            sum0 = _mm256_add_ps(
                sum0, _mm256_mul_ps(_mm256_loadu_ps(a + i),
                                    _mm256_loadu_ps(b + i)));
            sum1 = _mm256_add_ps(
                sum1, _mm256_mul_ps(_mm256_loadu_ps(a + i + 8),
                                    _mm256_loadu_ps(b + i + 8)));
        }
        __m256 total = _mm256_add_ps(sum0, sum1);
        __m128 low = _mm256_castps256_ps128(total);
        __m128 high = _mm256_extractf128_ps(total, 1);
        __m128 halves = _mm_add_ps(low, high);
        halves = _mm_hadd_ps(halves, halves);
        halves = _mm_hadd_ps(halves, halves);
        sum += _mm_cvtss_f32(halves);
#endif
        for (; i < len; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    static inline void DeepSeekV4LoadIndexerRow(
            const uint8_t *source, DataType dataType, uint64_t elementOffset,
            float *row, int dim) {
        int d = 0;
        if (dataType == DataType::FLOAT32) {
            const float *src = (const float*)source + elementOffset;
#if defined(__AVX512F__)
            for (; d + 15 < dim; d += 16) {
                _mm512_storeu_ps(row + d, _mm512_loadu_ps(src + d));
            }
#endif
            for (; d < dim; d++) {
                row[d] = src[d];
            }
            return;
        }
        const uint16_t *src = (const uint16_t*)source + elementOffset;
        if (dataType == DataType::BFLOAT16) {
#if defined(__AVX512F__)
            for (; d + 15 < dim; d += 16) {
                __m256i packed = _mm256_loadu_si256(
                    (const __m256i*)(src + d));
                __m512i expanded = _mm512_slli_epi32(
                    _mm512_cvtepu16_epi32(packed), 16);
                _mm512_storeu_ps(row + d, _mm512_castsi512_ps(expanded));
            }
#endif
            for (; d < dim; d++) {
                row[d] = DeepSeekV4HcPreBFloat16ToFloat(src[d]);
            }
            return;
        }
        if (dataType == DataType::FLOAT16) {
#if defined(__F16C__)
            for (; d + 7 < dim; d += 8) {
                __m128i packed = _mm_loadu_si128((const __m128i*)(src + d));
                _mm256_storeu_ps(row + d, _mm256_cvtph_ps(packed));
            }
#endif
            for (; d < dim; d++) {
                row[d] = half_to_float(src[d]);
            }
        }
    }

    struct DeepSeekV4IndexerQueryRotaryCache {
        bool valid = false;
        int seqlen = 0, ropeDim = 0, startPos = 0, originalSeqLen = 0;
        int betaFast = 0, betaSlow = 0;
        float ropeBase = 0.0f, ropeFactor = 0.0f;
        std::vector<float> cosValues, sinValues;

        bool Matches(int targetSeqlen, int targetRopeDim,
                     float targetRopeBase, int targetStartPos,
                     int targetOriginalSeqLen, float targetRopeFactor,
                     int targetBetaFast, int targetBetaSlow) const {
            return valid && seqlen == targetSeqlen &&
                ropeDim == targetRopeDim && ropeBase == targetRopeBase &&
                startPos == targetStartPos &&
                originalSeqLen == targetOriginalSeqLen &&
                ropeFactor == targetRopeFactor &&
                betaFast == targetBetaFast && betaSlow == targetBetaSlow;
        }
    };

    static DeepSeekV4IndexerQueryRotaryCache &
            DeepSeekV4GetIndexerQueryRotaryCache(
                int seqlen, int ropeDim, float ropeBase, int startPos,
                int originalSeqLen, float ropeFactor, int betaFast,
                int betaSlow) {
        static thread_local DeepSeekV4IndexerQueryRotaryCache cache;
        if (!cache.Matches(
                seqlen, ropeDim, ropeBase, startPos, originalSeqLen,
                ropeFactor, betaFast, betaSlow)) {
            DeepSeekV4BuildRotaryCache(
                seqlen, ropeDim, ropeBase, startPos, originalSeqLen,
                ropeFactor, betaFast, betaSlow, 1, false,
                cache.cosValues, cache.sinValues);
            cache.valid = true;
            cache.seqlen = seqlen;
            cache.ropeDim = ropeDim;
            cache.ropeBase = ropeBase;
            cache.startPos = startPos;
            cache.originalSeqLen = originalSeqLen;
            cache.ropeFactor = ropeFactor;
            cache.betaFast = betaFast;
            cache.betaSlow = betaSlow;
        }
        return cache;
    }

    struct DeepSeekV4IndexerTopKTask : MultiThreadBaseOp {
        const float *q;
        const uint8_t *rawQ;
        DataType rawQType;
        const float *weights;
        const float *kv;
        const float *ropeCos;
        const float *ropeSin;
        int32_t *output;
        int bsz, seqlen, heads, dim, blocks, topKWidth;
        int startPos, compressRatio, taskId, taskCount;
        int ropeDim;
        bool fusedPrepare, useHeadSimd, useFp4Simd;

        DeepSeekV4IndexerTopKTask(
            const float *q, const uint8_t *rawQ, DataType rawQType,
            const float *weights, const float *kv, const float *ropeCos,
            const float *ropeSin, int32_t *output, int bsz, int seqlen,
            int heads, int dim, int blocks, int topKWidth, int startPos,
            int compressRatio, int ropeDim, bool fusedPrepare,
            bool useHeadSimd, bool useFp4Simd, int taskId, int taskCount
        ) : q(q), rawQ(rawQ), rawQType(rawQType), weights(weights), kv(kv),
            ropeCos(ropeCos), ropeSin(ropeSin), output(output), bsz(bsz),
            seqlen(seqlen), heads(heads), dim(dim), blocks(blocks),
            topKWidth(topKWidth), startPos(startPos),
            compressRatio(compressRatio), taskId(taskId),
            taskCount(taskCount), ropeDim(ropeDim),
            fusedPrepare(fusedPrepare), useHeadSimd(useHeadSimd),
            useFp4Simd(useFp4Simd) {}

        static bool Better(const std::pair<float, int> &a,
                           const std::pair<float, int> &b) {
            if (a.first != b.first) {
                return a.first > b.first;
            }
            return a.second < b.second;
        }

#if defined(__AVX512F__)
        static inline float Score64Heads128Dim(
                const float *qByDim, const float *headWeights,
                const float *kvRow) {
            // qByDim is [128, 64].  Vector lanes represent index heads, so
            // each compressed KV scalar advances 64 independent dot products
            // without a horizontal reduction per head.
            __m512 dot0 = _mm512_setzero_ps();
            __m512 dot1 = _mm512_setzero_ps();
            __m512 dot2 = _mm512_setzero_ps();
            __m512 dot3 = _mm512_setzero_ps();
            for (int d = 0; d < 128; d++) {
                const float *qRow = qByDim + (uint64_t)d * 64;
                __m512 key = _mm512_set1_ps(kvRow[d]);
                dot0 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(qRow), key, dot0);
                dot1 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(qRow + 16), key, dot1);
                dot2 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(qRow + 32), key, dot2);
                dot3 = _mm512_fmadd_ps(
                    _mm512_loadu_ps(qRow + 48), key, dot3);
            }
            const __m512 zero = _mm512_setzero_ps();
            __m512 weighted = _mm512_mul_ps(
                _mm512_max_ps(dot0, zero),
                _mm512_loadu_ps(headWeights));
            weighted = _mm512_fmadd_ps(
                _mm512_max_ps(dot1, zero),
                _mm512_loadu_ps(headWeights + 16), weighted);
            weighted = _mm512_fmadd_ps(
                _mm512_max_ps(dot2, zero),
                _mm512_loadu_ps(headWeights + 32), weighted);
            weighted = _mm512_fmadd_ps(
                _mm512_max_ps(dot3, zero),
                _mm512_loadu_ps(headWeights + 48), weighted);
            return _mm512_reduce_add_ps(weighted);
        }

        static inline void Score64Heads128Dim4Blocks(
                const float *qByDim, const float *headWeights,
                const float *kvRows, float *blockScores) {
            __m512 dots[4][4];
#pragma GCC unroll 4
            for (int block = 0; block < 4; block++) {
#pragma GCC unroll 4
                for (int headGroup = 0; headGroup < 4; headGroup++) {
                    dots[block][headGroup] = _mm512_setzero_ps();
                }
            }
            for (int d = 0; d < 128; d++) {
                const float *qRow = qByDim + (uint64_t)d * 64;
                __m512 query[4] = {
                    _mm512_loadu_ps(qRow),
                    _mm512_loadu_ps(qRow + 16),
                    _mm512_loadu_ps(qRow + 32),
                    _mm512_loadu_ps(qRow + 48)
                };
#pragma GCC unroll 4
                for (int block = 0; block < 4; block++) {
                    __m512 key = _mm512_set1_ps(
                        kvRows[(uint64_t)block * 128 + d]);
#pragma GCC unroll 4
                    for (int headGroup = 0; headGroup < 4;
                         headGroup++) {
                        dots[block][headGroup] = _mm512_fmadd_ps(
                            query[headGroup], key,
                            dots[block][headGroup]);
                    }
                }
            }
            const __m512 zero = _mm512_setzero_ps();
            __m512 headWeightVectors[4] = {
                _mm512_loadu_ps(headWeights),
                _mm512_loadu_ps(headWeights + 16),
                _mm512_loadu_ps(headWeights + 32),
                _mm512_loadu_ps(headWeights + 48)
            };
#pragma GCC unroll 4
            for (int block = 0; block < 4; block++) {
                __m512 weighted = _mm512_mul_ps(
                    _mm512_max_ps(dots[block][0], zero),
                    headWeightVectors[0]);
#pragma GCC unroll 3
                for (int headGroup = 1; headGroup < 4;
                     headGroup++) {
                    weighted = _mm512_fmadd_ps(
                        _mm512_max_ps(dots[block][headGroup], zero),
                        headWeightVectors[headGroup], weighted);
                }
                blockScores[block] = _mm512_reduce_add_ps(weighted);
            }
        }

        static inline void Score64Heads128Dim8Blocks(
                const float *qByDim, const float *headWeights,
                const float *kvRows, float *blockScores) {
            // Eight blocks need 32 accumulators for all four head groups,
            // leaving no registers for queries on AVX-512.  Compute two head
            // groups at a time and spill only the first 16 dot vectors.  This
            // halves repeated Q loads versus two four-block calls while
            // preserving the original head-weight accumulation order.
            alignas(64) float firstDots[8][2][16];
            {
                __m512 dots[8][2];
#pragma GCC unroll 8
                for (int block = 0; block < 8; block++) {
                    dots[block][0] = _mm512_setzero_ps();
                    dots[block][1] = _mm512_setzero_ps();
                }
                for (int d = 0; d < 128; d++) {
                    const float *qRow = qByDim + (uint64_t)d * 64;
                    __m512 query0 = _mm512_loadu_ps(qRow);
                    __m512 query1 = _mm512_loadu_ps(qRow + 16);
#pragma GCC unroll 8
                    for (int block = 0; block < 8; block++) {
                        __m512 key = _mm512_set1_ps(
                            kvRows[(uint64_t)block * 128 + d]);
                        dots[block][0] = _mm512_fmadd_ps(
                            query0, key, dots[block][0]);
                        dots[block][1] = _mm512_fmadd_ps(
                            query1, key, dots[block][1]);
                    }
                }
#pragma GCC unroll 8
                for (int block = 0; block < 8; block++) {
                    _mm512_store_ps(firstDots[block][0], dots[block][0]);
                    _mm512_store_ps(firstDots[block][1], dots[block][1]);
                }
            }

            __m512 dots[8][2];
#pragma GCC unroll 8
            for (int block = 0; block < 8; block++) {
                dots[block][0] = _mm512_setzero_ps();
                dots[block][1] = _mm512_setzero_ps();
            }
            for (int d = 0; d < 128; d++) {
                const float *qRow = qByDim + (uint64_t)d * 64;
                __m512 query2 = _mm512_loadu_ps(qRow + 32);
                __m512 query3 = _mm512_loadu_ps(qRow + 48);
#pragma GCC unroll 8
                for (int block = 0; block < 8; block++) {
                    __m512 key = _mm512_set1_ps(
                        kvRows[(uint64_t)block * 128 + d]);
                    dots[block][0] = _mm512_fmadd_ps(
                        query2, key, dots[block][0]);
                    dots[block][1] = _mm512_fmadd_ps(
                        query3, key, dots[block][1]);
                }
            }
            const __m512 zero = _mm512_setzero_ps();
            __m512 headWeightVectors[4] = {
                _mm512_loadu_ps(headWeights),
                _mm512_loadu_ps(headWeights + 16),
                _mm512_loadu_ps(headWeights + 32),
                _mm512_loadu_ps(headWeights + 48)
            };
#pragma GCC unroll 8
            for (int block = 0; block < 8; block++) {
                __m512 weighted = _mm512_mul_ps(
                    _mm512_max_ps(_mm512_load_ps(firstDots[block][0]), zero),
                    headWeightVectors[0]);
                weighted = _mm512_fmadd_ps(
                    _mm512_max_ps(_mm512_load_ps(firstDots[block][1]), zero),
                    headWeightVectors[1], weighted);
                weighted = _mm512_fmadd_ps(
                    _mm512_max_ps(dots[block][0], zero),
                    headWeightVectors[2], weighted);
                weighted = _mm512_fmadd_ps(
                    _mm512_max_ps(dots[block][1], zero),
                    headWeightVectors[3], weighted);
                blockScores[block] = _mm512_reduce_add_ps(weighted);
            }
        }
#endif

        void Run() override {
            int totalTokens = bsz * seqlen;
            std::vector<float> scores;
            std::vector<std::pair<float, int> > ranked;
            // The 64x128 SIMD path consumes the transposed query directly.
            // Keep only one source row while preparing it and write it into
            // qByDim immediately; materializing the complete row-major query
            // adds another 32 KiB write/read pair for every token.
            std::vector<float> qRows(
                fusedPrepare ?
                    (useHeadSimd ? dim : (uint64_t)heads * dim) : 0);
#if defined(__AVX512F__)
            std::vector<float> qByDim(useHeadSimd ? heads * dim : 0);
            bool useEightBlocks =
                !DeepSeekV4HcPreEnvFlagEnabled(
                    "FASTLLM_DSV4_DISABLE_CPU_INDEXER_8BLOCK");
#endif
            scores.reserve(blocks);
            ranked.reserve(blocks);
            int ropePairs = ropeDim / 2;
            float hadamardScale = 1.0f / std::sqrt((float)dim);
            constexpr int tokenChunk = 4;
            for (int chunkBegin = taskId * tokenChunk;
                 chunkBegin < totalTokens;
                 chunkBegin += taskCount * tokenChunk) {
              int chunkEnd = std::min(totalTokens, chunkBegin + tokenChunk);
              for (int token = chunkBegin; token < chunkEnd; token++) {
                int b = token / seqlen;
                int s = token % seqlen;
                int available = std::min(
                    blocks, std::max(0, (startPos + s + 1) /
                                           compressRatio));
                int32_t *outRow = output + (uint64_t)token * topKWidth;
                std::fill(outRow, outRow + topKWidth, -1);
                if (available <= 0) {
                    continue;
                }
                const float *qTokenRows = nullptr;
                if (fusedPrepare) {
                    const float *cosRow = ropeCos + (uint64_t)s * ropePairs;
                    const float *sinRow = ropeSin + (uint64_t)s * ropePairs;
                    uint64_t tokenOffset = (uint64_t)token * heads * dim;
                    for (int head = 0; head < heads; head++) {
                        float *row = qRows.data() +
                            (useHeadSimd ? 0 : (uint64_t)head * dim);
                        DeepSeekV4LoadIndexerRow(
                            rawQ, rawQType,
                            tokenOffset + (uint64_t)head * dim, row, dim);
                        DeepSeekV4PrepareIndexerRow(
                            row, cosRow, sinRow, dim, ropeDim,
                            hadamardScale, useFp4Simd);
#if defined(__AVX512F__)
                        if (useHeadSimd) {
                            for (int d = 0; d < dim; d++) {
                                qByDim[(uint64_t)d * heads + head] = row[d];
                            }
                        }
#endif
                    }
                    qTokenRows = useHeadSimd ? nullptr : qRows.data();
                } else {
                    qTokenRows = q + (uint64_t)token * heads * dim;
                }
                scores.assign(available, 0.0f);
#if defined(__AVX512F__)
                if (useHeadSimd) {
                    if (!fusedPrepare) {
                        for (int dimBegin = 0; dimBegin < 128;
                             dimBegin += 16) {
                            for (int head = 0; head < 64; head++) {
                                const float *qHead = qTokenRows +
                                    (uint64_t)head * 128;
                                for (int d = dimBegin;
                                     d < dimBegin + 16; d++) {
                                    qByDim[(uint64_t)d * 64 + head] = qHead[d];
                                }
                            }
                        }
                    }
                    const float *headWeights = weights +
                        (uint64_t)token * heads;
                    int block = 0;
                    if (useEightBlocks) {
                        for (; block + 7 < available; block += 8) {
                            const float *kvRows = kv +
                                ((uint64_t)b * blocks + block) * dim;
                            Score64Heads128Dim8Blocks(
                                qByDim.data(), headWeights, kvRows,
                                scores.data() + block);
                        }
                    }
                    for (; block + 3 < available; block += 4) {
                        const float *kvRows = kv +
                            ((uint64_t)b * blocks + block) * dim;
                        Score64Heads128Dim4Blocks(
                            qByDim.data(), headWeights, kvRows,
                            scores.data() + block);
                    }
                    for (; block < available; block++) {
                        const float *kvRow = kv +
                            ((uint64_t)b * blocks + block) * dim;
                        scores[block] = Score64Heads128Dim(
                            qByDim.data(), headWeights, kvRow);
                    }
                } else
#endif
                {
                // Keep one compressed row hot while walking all query heads.
                // The opposite loop order rereads the complete compressed KV
                // cache once per head and becomes memory-bandwidth bound.
                for (int block = 0; block < available; block++) {
                    const float *kvRow = kv +
                        ((uint64_t)b * blocks + block) * dim;
                    float score = 0.0f;
                    for (int head = 0; head < heads; head++) {
                        float headWeight =
                            weights[(uint64_t)token * heads + head];
                        if (headWeight == 0.0f) {
                            continue;
                        }
                        const float *qRow = qTokenRows +
                            (uint64_t)head * dim;
                        float dot = DeepSeekV4IndexerDot(qRow, kvRow, dim);
                        score += headWeight * std::max(dot, 0.0f);
                    }
                    scores[block] = score;
                }
                }
                ranked.resize(available);
                for (int block = 0; block < available; block++) {
                    float score = std::isfinite(scores[block]) ?
                        scores[block] : -std::numeric_limits<float>::infinity();
                    ranked[block] = {score, block};
                }
                int keep = std::min(topKWidth, available);
                if (keep < available) {
                    std::nth_element(ranked.begin(), ranked.begin() + keep,
                                     ranked.end(), Better);
                }
                std::sort(ranked.begin(), ranked.begin() + keep, Better);
                for (int i = 0; i < keep; i++) {
                    outRow[i] = ranked[i].second;
                }
              }
            }
        }
    };

    void CpuDeepSeekV4IndexerTopKOp::Reshape(
            const std::string &opType, const fastllm::DataDict &datas,
            const fastllm::FloatDict &floatParams,
            const fastllm::IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &compressedKV = *(datas.find("compressedKV")->second);
        Data &output = *(datas.find("output")->second);
        int topK = intParams.find("topK") == intParams.end() ? 0 :
            intParams.find("topK")->second;
        int blocks = compressedKV.dims.size() == 3 ? compressedKV.dims[1] : 0;
        output.dataType = DataType::INT32;
        output.Resize({q.dims[0], q.dims[1], std::min(topK, blocks)});
    }

    void CpuDeepSeekV4IndexerTopKOp::Run(
            const std::string &opType, const fastllm::DataDict &datas,
            const fastllm::FloatDict &floatParams,
            const fastllm::IntDict &intParams) {
        Data &q = *(datas.find("q")->second);
        Data &weights = *(datas.find("weights")->second);
        Data &compressedKV = *(datas.find("compressedKV")->second);
        Data &output = *(datas.find("output")->second);
        auto getInt = [&](const char *name, int fallback) {
            auto it = intParams.find(name);
            return it == intParams.end() ? fallback : it->second;
        };
        auto getFloat = [&](const char *name, float fallback) {
            auto it = floatParams.find(name);
            return it == floatParams.end() ? fallback : it->second;
        };
        int topK = getInt("topK", 0);
        int compressRatio = getInt("compressRatio", 0);
        int ropeDim = getInt("ropeDim", 0);
        int startPos = getInt("startPos", 0);
        int originalSeqLen = getInt("originalSeqLen", 0);
        int betaFast = getInt("betaFast", 32);
        int betaSlow = getInt("betaSlow", 1);
        float ropeBase = getFloat("ropeBase", 10000.0f);
        float ropeFactor = getFloat("ropeFactor", 1.0f);
        AssertInFastLLM(
            q.dataDevice == DataDevice::CPU &&
                weights.dataDevice == DataDevice::CPU &&
                compressedKV.dataDevice == DataDevice::CPU &&
                q.dims.size() == 4 && weights.dims.size() == 3 &&
                compressedKV.dims.size() == 3,
            "DeepSeekV4IndexerTopK error: invalid CPU tensors.\n");
        int bsz = q.dims[0], seqlen = q.dims[1];
        int heads = q.dims[2], dim = q.dims[3];
        int blocks = compressedKV.dims[1];
        AssertInFastLLM(
            weights.dims[0] == bsz && weights.dims[1] == seqlen &&
                weights.dims[2] == heads &&
                compressedKV.dims[0] == bsz &&
                compressedKV.dims[2] == dim && topK > 0 &&
                compressRatio > 0 && ropeDim > 0 && ropeDim <= dim &&
                ropeDim % 2 == 0 && dim > 0 &&
                (dim & (dim - 1)) == 0 && dim % 32 == 0 &&
                (q.dataType == DataType::FLOAT32 ||
                 q.dataType == DataType::FLOAT16 ||
                 q.dataType == DataType::BFLOAT16),
            "DeepSeekV4IndexerTopK error: invalid shapes or params.\n");

        auto weightValues = DeepSeekV4HcPreReadFloatData(weights);
        auto kvValues = DeepSeekV4HcPreReadFloatData(compressedKV);
        bool fusedPrepare = !DeepSeekV4HcPreEnvFlagEnabled(
            "FASTLLM_DSV4_DISABLE_CPU_INDEXER_FUSED_PREP");
        bool useHeadSimd = false;
#if defined(__AVX512F__)
        useHeadSimd = heads == 64 && dim == 128 &&
            !DeepSeekV4HcPreEnvFlagEnabled(
                "FASTLLM_DSV4_DISABLE_CPU_INDEXER_HEAD_SIMD");
#endif
        bool useFp4Simd = !DeepSeekV4HcPreEnvFlagEnabled(
            "FASTLLM_DSV4_DISABLE_CPU_INDEXER_FP4_SIMD");
        std::vector<float> qValues;
        const float *preparedQ = nullptr;
        const float *ropeCos = nullptr;
        const float *ropeSin = nullptr;
        if (fusedPrepare) {
            auto &ropeCache = DeepSeekV4GetIndexerQueryRotaryCache(
                seqlen, ropeDim, ropeBase, startPos, originalSeqLen,
                ropeFactor, betaFast, betaSlow);
            ropeCos = ropeCache.cosValues.data();
            ropeSin = ropeCache.sinValues.data();
        } else {
            qValues = DeepSeekV4HcPreReadFloatData(q);
            DeepSeekV4PrepareIndexerRows(
                qValues, q.dims, ropeDim, ropeBase, startPos,
                originalSeqLen, ropeFactor, betaFast, betaSlow, 1);
            preparedQ = qValues.data();
        }
        int topKWidth = std::min(topK, blocks);
        output.dataType = DataType::INT32;
        output.Resize({bsz, seqlen, topKWidth});
        output.Allocate(false);
        int32_t *outputData = (int32_t*)output.cpuData;
        AliveThreadPool *pool = GetAlivePool();
        int firstThread = pool->curActivateThreadInterval.first;
        int availableThreads = std::max(
            1, pool->curActivateThreadInterval.second - firstThread);
        int totalTokens = bsz * seqlen;
        constexpr int tokenChunk = 4;
        int threadCount = std::min(
            availableThreads, (totalTokens + tokenChunk - 1) / tokenChunk);
        if (threadCount <= 1 || totalTokens < 16) {
            DeepSeekV4IndexerTopKTask task(
                preparedQ, q.cpuData, q.dataType, weightValues.data(),
                kvValues.data(), ropeCos, ropeSin, outputData, bsz, seqlen,
                heads, dim, blocks, topKWidth, startPos, compressRatio,
                ropeDim, fusedPrepare, useHeadSimd, useFp4Simd,
                0, 1);
            task.Run();
            return;
        }
        std::vector<DeepSeekV4IndexerTopKTask*> tasks;
        tasks.reserve(threadCount);
        for (int thread = 0; thread < threadCount; thread++) {
            tasks.push_back(new DeepSeekV4IndexerTopKTask(
                preparedQ, q.cpuData, q.dataType, weightValues.data(),
                kvValues.data(), ropeCos, ropeSin, outputData, bsz, seqlen,
                heads, dim, blocks, topKWidth, startPos, compressRatio,
                ropeDim, fusedPrepare, useHeadSimd, useFp4Simd,
                thread, threadCount));
            pool->PushOp(firstThread + thread, tasks.back());
        }
        for (int thread = 0; thread < threadCount; thread++) {
            pool->Wait(firstThread + thread);
            delete tasks[thread];
        }
    }

    struct DeepSeekV4WoAReferenceOp : MultiThreadBaseOp {
        const std::vector<float> *ov;
        const std::vector<float> *wv;
        float *y;
        int st, end;
        int bsz, seqlen, heads, headDim, groups, oRank, headsPerGroup, groupDim;

        DeepSeekV4WoAReferenceOp(const std::vector<float> *ov, const std::vector<float> *wv, float *y,
                                 int st, int end, int bsz, int seqlen, int heads, int headDim,
                                 int groups, int oRank)
            : ov(ov), wv(wv), y(y), st(st), end(end), bsz(bsz), seqlen(seqlen),
              heads(heads), headDim(headDim), groups(groups), oRank(oRank) {
            headsPerGroup = heads / groups;
            groupDim = headsPerGroup * headDim;
        }

        void Run() override {
            const float *ovData = ov->data();
            const float *wvData = wv->data();
            for (int idx = st; idx < end; idx++) {
                int r = idx % oRank;
                int tmp = idx / oRank;
                int g = tmp % groups;
                tmp /= groups;
                int s = tmp % seqlen;
                int b = tmp / seqlen;
                const float *w = wvData + ((uint64_t)g * oRank + r) * groupDim;
                double v = 0.0;
                int d = 0;
                for (int hh = 0; hh < headsPerGroup; hh++) {
                    const float *src = ovData + (((uint64_t)b * seqlen + s) * heads +
                                                 g * headsPerGroup + hh) * headDim;
                    for (int localD = 0; localD < headDim; localD++, d++) {
                        v += (double)src[localD] * w[d];
                    }
                }
                y[idx] = (float)v;
            }
        }
    };

    struct DeepSeekV4WoAPackInputTask : MultiThreadBaseOp {
        const void *input;
        DataType inputType;
        float *groupInput;
        int tokens, groups, inputStride, groupDim;
        int taskId, taskCount;

        DeepSeekV4WoAPackInputTask(
            const void *input, DataType inputType, float *groupInput,
            int tokens, int groups, int inputStride, int groupDim,
            int taskId, int taskCount
        ) : input(input), inputType(inputType), groupInput(groupInput),
            tokens(tokens), groups(groups), inputStride(inputStride),
            groupDim(groupDim), taskId(taskId), taskCount(taskCount) {}

        void Run() override {
            int totalChunks = tokens * groups;
            int chunkBegin = (int)((int64_t)totalChunks * taskId / taskCount);
            int chunkEnd =
                (int)((int64_t)totalChunks * (taskId + 1) / taskCount);
            const float *inputF32 = (const float*)input;
            const uint16_t *inputF16 = (const uint16_t*)input;
            for (int chunk = chunkBegin; chunk < chunkEnd; chunk++) {
                int group = chunk / tokens;
                int token = chunk - group * tokens;
                uint64_t sourceOffset =
                    (uint64_t)token * inputStride +
                    (uint64_t)group * groupDim;
                float *destination =
                    groupInput + (uint64_t)chunk * groupDim;
                if (inputType == DataType::FLOAT32) {
                    memcpy(destination, inputF32 + sourceOffset,
                           (uint64_t)groupDim * sizeof(float));
                } else if (inputType == DataType::FLOAT16) {
                    for (int d = 0; d < groupDim; d++) {
                        destination[d] = half_to_float(
                            inputF16[sourceOffset + d]);
                    }
                } else {
                    for (int d = 0; d < groupDim; d++) {
                        destination[d] =
                            DeepSeekV4HcPreBFloat16ToFloat(
                                inputF16[sourceOffset + d]);
                    }
                }
            }
        }
    };

    static void DeepSeekV4WoAPackInput(
            const Data &input, float *groupInput, int tokens, int groups,
            int inputStride, int groupDim, AliveThreadPool *pool) {
        int firstThread = pool->curActivateThreadInterval.first;
        int availableThreads = std::max(
            1, pool->curActivateThreadInterval.second - firstThread);
        int totalChunks = tokens * groups;
        int taskCount = std::min(availableThreads, totalChunks);
        if (taskCount <= 1) {
            DeepSeekV4WoAPackInputTask(
                input.cpuData, input.dataType, groupInput, tokens, groups,
                inputStride, groupDim, 0, 1).Run();
            return;
        }

        std::vector<DeepSeekV4WoAPackInputTask*> tasks;
        tasks.reserve(taskCount);
        for (int task = 0; task < taskCount; task++) {
            tasks.push_back(new DeepSeekV4WoAPackInputTask(
                input.cpuData, input.dataType, groupInput, tokens, groups,
                inputStride, groupDim, task, taskCount));
            pool->PushOp(firstThread + task, tasks.back());
        }
        for (int task = 0; task < taskCount; task++) {
            pool->Wait(firstThread + task);
            delete tasks[task];
        }
    }

    struct DeepSeekV4WoAWriteOutputTask : MultiThreadBaseOp {
        const float *groupOutput;
        uint16_t *output;
        int tokens, groups, oRank, outputStride;
        int taskId, taskCount;

        DeepSeekV4WoAWriteOutputTask(
            const float *groupOutput, uint16_t *output,
            int tokens, int groups, int oRank,
            int taskId, int taskCount
        ) : groupOutput(groupOutput), output(output), tokens(tokens),
            groups(groups), oRank(oRank), outputStride(groups * oRank),
            taskId(taskId), taskCount(taskCount) {}

        void Run() override {
            int totalChunks = tokens * groups;
            int chunkBegin = (int)((int64_t)totalChunks * taskId / taskCount);
            int chunkEnd =
                (int)((int64_t)totalChunks * (taskId + 1) / taskCount);
            for (int chunk = chunkBegin; chunk < chunkEnd; chunk++) {
                int group = chunk / tokens;
                int token = chunk - group * tokens;
                const float *source =
                    groupOutput + (uint64_t)chunk * oRank;
                uint16_t *destination =
                    output + (uint64_t)token * outputStride +
                    (uint64_t)group * oRank;
                for (int d = 0; d < oRank; d++) {
                    destination[d] =
                        DeepSeekV4HcPreFloatToBFloat16(source[d]);
                }
            }
        }
    };

    static void DeepSeekV4WoAWriteOutput(
            const float *groupOutput, uint16_t *output,
            int tokens, int groups, int oRank, AliveThreadPool *pool) {
        int firstThread = pool->curActivateThreadInterval.first;
        int availableThreads = std::max(
            1, pool->curActivateThreadInterval.second - firstThread);
        int totalChunks = tokens * groups;
        int taskCount = std::min(availableThreads, totalChunks);
        if (taskCount <= 1) {
            DeepSeekV4WoAWriteOutputTask(
                groupOutput, output, tokens, groups, oRank, 0, 1).Run();
            return;
        }

        std::vector<DeepSeekV4WoAWriteOutputTask*> tasks;
        tasks.reserve(taskCount);
        for (int task = 0; task < taskCount; task++) {
            tasks.push_back(new DeepSeekV4WoAWriteOutputTask(
                groupOutput, output, tokens, groups, oRank,
                task, taskCount));
            pool->PushOp(firstThread + task, tasks.back());
        }
        for (int task = 0; task < taskCount; task++) {
            pool->Wait(firstThread + task);
            delete tasks[task];
        }
    }

    void CpuDeepSeekV4WoAOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &output = *(datas.find("output")->second);
        int groups = intParams.find("groups") != intParams.end() ? intParams.find("groups")->second : 1;
        int oRank = intParams.find("oRank") != intParams.end() ? intParams.find("oRank")->second : 1;

        AssertInFastLLM(input.dims.size() == 4, "DeepSeekV4WoA error: input's shape's size should be 4.\n");
        AssertInFastLLM(groups > 0 && oRank > 0, "DeepSeekV4WoA error: invalid groups or oRank.\n");
        int bsz = input.dims[0], seqlen = input.dims[1], heads = input.dims[2], headDim = input.dims[3];
        AssertInFastLLM(heads % groups == 0, "DeepSeekV4WoA error: heads should be divisible by groups.\n");
        int headsPerGroup = heads / groups;
        int groupDim = headsPerGroup * headDim;
        AssertInFastLLM(weight.Count(0) == (uint64_t)groups * oRank * groupDim,
                        "DeepSeekV4WoA error: weight shape mismatch.\n");

        const bool profileWoALinear = GetFastllmEnv().printProfile;
        const int tokens = bsz * seqlen;
        auto *pool = GetAlivePool();
        int threadNum = (int)pool->threads.size();
        bool useFastFloat16Path = weight.dataType == DataType::FLOAT16 &&
                                  !DeepSeekV4HcPreEnvFlagEnabled(
                                      "FASTLLM_DSV4_DISABLE_CPU_WOA_FAST");

        if (useFastFloat16Path && tokens > 1) {
            AssertInFastLLM(
                input.dataDevice == DataDevice::CPU &&
                    input.cpuData != nullptr &&
                    weight.dataDevice == DataDevice::CPU &&
                    weight.cpuData != nullptr,
                "DeepSeekV4WoA CPU fast path requires CPU tensors.\n");
            AssertInFastLLM(
                input.dataType == DataType::FLOAT32 ||
                    input.dataType == DataType::FLOAT16 ||
                    input.dataType == DataType::BFLOAT16,
                "DeepSeekV4WoA CPU fast path received an unsupported input dtype.\n");

            auto operatorBegin = profileWoALinear ?
                std::chrono::steady_clock::now() :
                std::chrono::steady_clock::time_point();
            std::unique_ptr<float[]> groupInput(
                new float[(uint64_t)groups * tokens * groupDim]);
            std::unique_ptr<float[]> groupOutput(
                new float[(uint64_t)groups * tokens * oRank]);

            output.dataType = DataType::BFLOAT16;
            output.Resize({bsz, seqlen, groups * oRank});
            output.Allocate();
            DeepSeekV4WoAPackInput(
                input, groupInput.get(), tokens, groups,
                heads * headDim, groupDim, pool);
            auto prepareEnd = profileWoALinear ?
                std::chrono::steady_clock::now() :
                std::chrono::steady_clock::time_point();

            const int outputRowsPerTask = 64;
            std::vector<MultiThreadBaseOp*> ops;
            ops.reserve((uint64_t)groups *
                        ((oRank + outputRowsPerTask - 1) /
                         outputRowsPerTask));
            uint16_t *weightData = (uint16_t*)weight.cpuData;
            for (int group = 0; group < groups; group++) {
                float *currentInput =
                    groupInput.get() + (uint64_t)group * tokens * groupDim;
                uint16_t *currentWeight =
                    weightData + (uint64_t)group * oRank * groupDim;
                float *currentOutput =
                    groupOutput.get() + (uint64_t)group * tokens * oRank;
                for (int start = 0; start < oRank;
                     start += outputRowsPerTask) {
                    int end = std::min(start + outputRowsPerTask, oRank);
                    ops.push_back(new MultiThreadLinearFloat32Float16Op(
                        currentInput, currentWeight, nullptr, currentOutput,
                        tokens, groupDim, oRank, start, end));
                }
            }
            DynamicScheduleTasks(ops);
            auto linearEnd = profileWoALinear ?
                std::chrono::steady_clock::now() :
                std::chrono::steady_clock::time_point();

            DeepSeekV4WoAWriteOutput(
                groupOutput.get(), (uint16_t*)output.cpuData,
                tokens, groups, oRank, pool);
            if (profileWoALinear) {
                auto operatorEnd = std::chrono::steady_clock::now();
                double prepareSeconds = std::chrono::duration<double>(
                    prepareEnd - operatorBegin).count();
                double linearSeconds = std::chrono::duration<double>(
                    linearEnd - prepareEnd).count();
                double outputSeconds = std::chrono::duration<double>(
                    operatorEnd - linearEnd).count();
                double flops = 2.0 * (double)tokens * groups * oRank *
                               groupDim;
                printf("[fastllm-profile-dsv4-woa-cpu] "
                       "tokens=%d groups=%d m=%d k=%d prepare=%.6f "
                       "linear=%.6f output=%.6f total=%.6f "
                       "gflops=%.3f\n",
                       tokens, groups, groupDim, oRank,
                       prepareSeconds, linearSeconds, outputSeconds,
                       prepareSeconds + linearSeconds + outputSeconds,
                       flops / linearSeconds / 1.0e9);
            }
            return;
        }

        double woALinearSeconds = 0.0;
        auto ov = DeepSeekV4HcPreReadFloatData(input);
        std::vector<float> y((uint64_t)bsz * seqlen * groups * oRank, 0.0f);
        if (useFastFloat16Path) {
            uint16_t *weightData = (uint16_t*)weight.cpuData;
            if (tokens == 1) {
                const int outputRowsPerTask = 64;
                std::vector<MultiThreadBaseOp*> ops;
                ops.reserve((uint64_t)groups *
                            ((oRank + outputRowsPerTask - 1) / outputRowsPerTask));
                for (int g = 0; g < groups; g++) {
                    float *groupInput = ov.data() + (uint64_t)g * groupDim;
                    uint16_t *groupWeight =
                        weightData + (uint64_t)g * oRank * groupDim;
                    float *groupOutput = y.data() + (uint64_t)g * oRank;
                    for (int st = 0; st < oRank; st += outputRowsPerTask) {
                        int end = std::min(st + outputRowsPerTask, oRank);
                        ops.push_back(new MultiThreadLinearFloat32Float16Op(
                            groupInput, groupWeight, nullptr, groupOutput,
                            1, groupDim, oRank, st, end));
                    }
                }
                auto linearBegin = profileWoALinear ?
                    std::chrono::steady_clock::now() :
                    std::chrono::steady_clock::time_point();
                DynamicScheduleTasks(ops);
                if (profileWoALinear) {
                    woALinearSeconds += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - linearBegin).count();
                }
            }
        } else {
            auto wv = DeepSeekV4HcPreReadFloatData(weight);
            int total = bsz * seqlen * groups * oRank;
            int referenceThreadNum = std::min(threadNum, total);
            if (referenceThreadNum <= 1 || total < 1024) {
                DeepSeekV4WoAReferenceOp(&ov, &wv, y.data(), 0, total, bsz, seqlen,
                                         heads, headDim, groups, oRank).Run();
            } else {
                std::vector<DeepSeekV4WoAReferenceOp*> ops;
                int per = total / referenceThreadNum;
                int cur = 0;
                for (int i = 0; i < referenceThreadNum; i++) {
                    int end = (i == referenceThreadNum - 1) ? total : cur + per;
                    ops.push_back(new DeepSeekV4WoAReferenceOp(
                        &ov, &wv, y.data(), cur, end, bsz, seqlen,
                        heads, headDim, groups, oRank));
                    cur = end;
                }
                for (int i = 0; i < (int)ops.size(); i++) {
                    pool->PushOp(i, ops[i]);
                }
                for (int i = 0; i < (int)ops.size(); i++) {
                    pool->Wait(i);
                    delete ops[i];
                }
            }
        }

        output.dataType = DataType::BFLOAT16;
        output.Resize({bsz, seqlen, groups * oRank});
        DeepSeekV4HcPreWriteFloatData(y, output);
        if (profileWoALinear && useFastFloat16Path &&
            woALinearSeconds > 0.0) {
            double flops = 2.0 * (double)bsz * seqlen * groups * oRank *
                           groupDim;
            printf("[fastllm-profile-dsv4-woa-cpu] "
                   "tokens=%d groups=%d m=%d k=%d linear=%.6f "
                   "gflops=%.3f\n",
                   bsz * seqlen, groups, groupDim, oRank,
                   woALinearSeconds, flops / woALinearSeconds / 1.0e9);
        }
    }

    struct DeepSeekV4BuildCompressedKVFromRawReferenceOp : MultiThreadBaseOp {
        const float *kv;
        const float *score;
        const float *ape;
        float *compressed;
        uint64_t st, end;
        int bsz, rawTokenBase, rawLen, blockStart, blockCount, compressRatio, headDim, wideDim;
        bool overlap;

        DeepSeekV4BuildCompressedKVFromRawReferenceOp(const float *kv, const float *score,
                                                      const float *ape, float *compressed,
                                                      uint64_t st, uint64_t end,
                                                      int bsz, int rawTokenBase, int rawLen,
                                                      int blockStart, int blockCount,
                                                      int compressRatio, int headDim,
                                                      int wideDim, bool overlap)
            : kv(kv), score(score), ape(ape), compressed(compressed), st(st), end(end),
              bsz(bsz), rawTokenBase(rawTokenBase), rawLen(rawLen), blockStart(blockStart),
              blockCount(blockCount), compressRatio(compressRatio), headDim(headDim),
              wideDim(wideDim), overlap(overlap) {}

        uint64_t RawOffset(int b, int token, int dimOffset) const {
            int localToken = token - rawTokenBase;
            return ((uint64_t)b * rawLen + localToken) * wideDim + dimOffset;
        }

        void ScanTerms(int b, int block, int d, float &mx) const {
            if (overlap) {
                if (block > 0) {
                    for (int r = 0; r < compressRatio; r++) {
                        int tok = (block - 1) * compressRatio + r;
                        uint64_t off = RawOffset(b, tok, d);
                        mx = std::max(mx, score[off] + ape[(uint64_t)r * wideDim + d]);
                    }
                }
                for (int r = 0; r < compressRatio; r++) {
                    int tok = block * compressRatio + r;
                    uint64_t off = RawOffset(b, tok, headDim + d);
                    mx = std::max(mx, score[off] + ape[(uint64_t)r * wideDim + headDim + d]);
                }
            } else {
                for (int r = 0; r < compressRatio; r++) {
                    int tok = block * compressRatio + r;
                    uint64_t off = RawOffset(b, tok, d);
                    mx = std::max(mx, score[off] + ape[(uint64_t)r * wideDim + d]);
                }
            }
        }

        void AccumulateTerms(int b, int block, int d, float mx, double &sum, double &value) const {
            if (overlap) {
                if (block > 0) {
                    for (int r = 0; r < compressRatio; r++) {
                        int tok = (block - 1) * compressRatio + r;
                        uint64_t off = RawOffset(b, tok, d);
                        double e = std::exp((double)(score[off] + ape[(uint64_t)r * wideDim + d]) - mx);
                        sum += e;
                        value += e * kv[off];
                    }
                }
                for (int r = 0; r < compressRatio; r++) {
                    int tok = block * compressRatio + r;
                    uint64_t off = RawOffset(b, tok, headDim + d);
                    double e = std::exp((double)(score[off] + ape[(uint64_t)r * wideDim + headDim + d]) - mx);
                    sum += e;
                    value += e * kv[off];
                }
            } else {
                for (int r = 0; r < compressRatio; r++) {
                    int tok = block * compressRatio + r;
                    uint64_t off = RawOffset(b, tok, d);
                    double e = std::exp((double)(score[off] + ape[(uint64_t)r * wideDim + d]) - mx);
                    sum += e;
                    value += e * kv[off];
                }
            }
        }

        void Run() override {
            (void)bsz;
            for (uint64_t idx = st; idx < end; idx++) {
                int d = (int)(idx % headDim);
                uint64_t tmp = idx / headDim;
                int localBlock = (int)(tmp % blockCount);
                int b = (int)(tmp / blockCount);
                int block = blockStart + localBlock;

                float mx = -FLT_MAX;
                ScanTerms(b, block, d, mx);

                double sum = 0.0, value = 0.0;
                AccumulateTerms(b, block, d, mx, sum, value);
                compressed[((uint64_t)b * blockCount + localBlock) * headDim + d] =
                    (float)(value / std::max(sum, 1e-30));
            }
        }
    };

    static void DeepSeekV4ComputeCompressedKVFromRawCpu(const std::vector<float> &kv,
                                                        const std::vector<float> &score,
                                                        const std::vector<float> &ape,
                                                        int bsz, int rawTokenBase, int rawLen,
                                                        int blockStart, int blockCount,
                                                        int compressRatio, int headDim,
                                                        int wideDim, bool overlap,
                                                        std::vector<float> &compressed) {
        compressed.assign((uint64_t)bsz * blockCount * headDim, 0.0f);
        uint64_t total = (uint64_t)bsz * blockCount * headDim;
        if (total == 0) {
            return;
        }
        auto *pool = GetAlivePool();
        int threadNum = std::min((int)pool->threads.size(), (int)std::min<uint64_t>(total, 64));
        if (threadNum <= 1 || total < 4096 ||
            DeepSeekV4HcPreEnvFlagEnabled("FASTLLM_DSV4_DISABLE_CPU_COMPRESSKV_PARALLEL")) {
            DeepSeekV4BuildCompressedKVFromRawReferenceOp(
                kv.data(), score.data(), ape.data(), compressed.data(), 0, total,
                bsz, rawTokenBase, rawLen, blockStart, blockCount,
                compressRatio, headDim, wideDim, overlap).Run();
            return;
        }

        std::vector<DeepSeekV4BuildCompressedKVFromRawReferenceOp*> ops;
        uint64_t per = (total + threadNum - 1) / threadNum;
        for (int i = 0; i < threadNum; i++) {
            uint64_t st = (uint64_t)i * per;
            uint64_t end = std::min(total, st + per);
            if (st >= end) {
                break;
            }
            ops.push_back(new DeepSeekV4BuildCompressedKVFromRawReferenceOp(
                kv.data(), score.data(), ape.data(), compressed.data(), st, end,
                bsz, rawTokenBase, rawLen, blockStart, blockCount,
                compressRatio, headDim, wideDim, overlap));
        }
        for (int i = 0; i < (int)ops.size(); i++) {
            pool->PushOp(i, ops[i]);
        }
        for (int i = 0; i < (int)ops.size(); i++) {
            pool->Wait(i);
            delete ops[i];
        }
    }

    static void DeepSeekV4FinalizeCompressedKVRowsCpu(const std::vector<float> &compressed,
                                                      const std::vector<float> &normWeight,
                                                      int bsz, int blockCount, int blockStart,
                                                      int compressRatio, int headDim, int ropeDim,
                                                      float ropeBase, float ropeFactor,
                                                      int betaFast, int betaSlow,
                                                      int originalSeqLen, bool indexer,
                                                      std::vector<float> &rows) {
        rows.resize((uint64_t)bsz * blockCount * headDim);
        for (uint64_t i = 0; i < rows.size(); i++) {
            rows[i] = DeepSeekV4HcPreBFloat16ToFloat(DeepSeekV4HcPreFloatToBFloat16(compressed[i]));
        }

        int totalRows = bsz * blockCount;
        for (int r = 0; r < totalRows; r++) {
            float *row = rows.data() + (uint64_t)r * headDim;
            double ss = 0.0;
            for (int d = 0; d < headDim; d++) {
                ss += (double)row[d] * row[d];
            }
            float scale = 1.0f / std::sqrt((float)(ss / headDim) + 1e-6f);
            for (int d = 0; d < headDim; d++) {
                float w = d < (int)normWeight.size() ? normWeight[d] : 1.0f;
                row[d] = DeepSeekV4HcPreBFloat16ToFloat(DeepSeekV4HcPreFloatToBFloat16(row[d] * scale * w));
            }
        }

        if (indexer) {
            DeepSeekV4PrepareIndexerRows(
                rows, {bsz, blockCount, headDim}, ropeDim, ropeBase,
                blockStart * compressRatio, originalSeqLen, ropeFactor,
                betaFast, betaSlow, compressRatio);
        } else {
            DeepSeekV4ApplyRotaryReference(
                rows, {bsz, blockCount, headDim}, ropeDim, ropeBase,
                blockStart * compressRatio, originalSeqLen, ropeFactor,
                betaFast, betaSlow, compressRatio);
            DeepSeekV4ActQuantInplaceReference(
                rows, {bsz, blockCount, headDim}, headDim - ropeDim, 64);
        }
    }

    void CpuDeepSeekV4BuildCompressedKVFromRawOp::Run(const std::string &opType,
                                                      const fastllm::DataDict &datas,
                                                      const fastllm::FloatDict &floatParams,
                                                      const fastllm::IntDict &intParams) {
        Data &kv = *(datas.find("kv")->second);
        Data &score = *(datas.find("score")->second);
        Data &ape = *(datas.find("ape")->second);
        Data &normWeight = *(datas.find("normWeight")->second);
        Data &cache = *(datas.find("cache")->second);

        int rawTokenBase = intParams.find("rawTokenBase") != intParams.end() ? intParams.find("rawTokenBase")->second : 0;
        int rawLen = intParams.find("rawLen") != intParams.end() ? intParams.find("rawLen")->second : 0;
        int blockStart = intParams.find("blockStart") != intParams.end() ? intParams.find("blockStart")->second : 0;
        int blockCount = intParams.find("blockCount") != intParams.end() ? intParams.find("blockCount")->second : 0;
        int compressRatio = intParams.find("compressRatio") != intParams.end() ? intParams.find("compressRatio")->second : 0;
        int headDim = intParams.find("headDim") != intParams.end() ? intParams.find("headDim")->second : 0;
        int ropeDim = intParams.find("ropeDim") != intParams.end() ? intParams.find("ropeDim")->second : 0;
        int betaFast = intParams.find("betaFast") != intParams.end() ? intParams.find("betaFast")->second : 32;
        int betaSlow = intParams.find("betaSlow") != intParams.end() ? intParams.find("betaSlow")->second : 1;
        int originalSeqLen = intParams.find("originalSeqLen") != intParams.end() ? intParams.find("originalSeqLen")->second : 0;
        bool overlap = intParams.find("overlap") != intParams.end() && intParams.find("overlap")->second != 0;
        bool indexer = intParams.find("indexer") != intParams.end() && intParams.find("indexer")->second != 0;
        float ropeBase = floatParams.find("ropeBase") != floatParams.end() ? floatParams.find("ropeBase")->second : 10000.0f;
        float ropeFactor = floatParams.find("ropeFactor") != floatParams.end() ? floatParams.find("ropeFactor")->second : 1.0f;

        if (blockCount <= 0) {
            return;
        }
        int bsz = kv.dims.empty() ? 0 : kv.dims[0];
        int wideDim = (overlap ? 2 : 1) * headDim;
        AssertInFastLLM(kv.dims.size() == 3 && score.dims == kv.dims &&
                        kv.dims[0] == bsz && kv.dims[1] >= rawLen && kv.dims[2] == wideDim,
                        "DeepSeekV4BuildCompressedKVFromRaw error: invalid kv or score.\n");
        AssertInFastLLM(ape.Count(0) >= (uint64_t)compressRatio * wideDim &&
                        rawLen > 0 && compressRatio > 0 && headDim > 0 &&
                        ropeDim > 0 && ropeDim <= headDim,
                        "DeepSeekV4BuildCompressedKVFromRaw error: invalid params.\n");

        auto kvValues = DeepSeekV4HcPreReadFloatData(kv);
        auto scoreValues = DeepSeekV4HcPreReadFloatData(score);
        auto apeValues = DeepSeekV4HcPreReadFloatData(ape);
        auto normValues = DeepSeekV4HcPreReadFloatData(normWeight);

        std::vector<float> compressed;
        DeepSeekV4ComputeCompressedKVFromRawCpu(kvValues, scoreValues, apeValues,
                                                bsz, rawTokenBase, rawLen,
                                                blockStart, blockCount, compressRatio,
                                                headDim, wideDim, overlap, compressed);

        std::vector<float> rowValues;
        DeepSeekV4FinalizeCompressedKVRowsCpu(compressed, normValues, bsz, blockCount,
                                              blockStart, compressRatio, headDim, ropeDim,
                                              ropeBase, ropeFactor, betaFast, betaSlow,
                                              originalSeqLen, indexer, rowValues);

        int totalBlocks = blockStart + blockCount;
        Data newRows(DataType::BFLOAT16, {bsz, blockCount, headDim});
        DeepSeekV4HcPreWriteFloatData(rowValues, newRows);
        if (blockStart <= 0) {
            cache.CopyFrom(newRows);
            cache.SetKVCache();
            return;
        }

        AssertInFastLLM(cache.dataType == DataType::BFLOAT16 && cache.dims.size() == 3 &&
                        cache.dims[0] == bsz && cache.dims[1] >= blockStart &&
                        cache.dims[2] == headDim && cache.cpuData != nullptr,
                        "DeepSeekV4BuildCompressedKVFromRaw error: invalid old cache.\n");
        Data merged(DataType::BFLOAT16, {bsz, totalBlocks, headDim});
        merged.Allocate(false);
        uint16_t *dst = (uint16_t*)merged.cpuData;
        const uint16_t *oldData = (const uint16_t*)cache.cpuData;
        const uint16_t *newData = (const uint16_t*)newRows.cpuData;
        for (int b = 0; b < bsz; b++) {
            memcpy(dst + (uint64_t)b * totalBlocks * headDim,
                   oldData + (uint64_t)b * cache.dims[1] * headDim,
                   (uint64_t)blockStart * headDim * sizeof(uint16_t));
            memcpy(dst + ((uint64_t)b * totalBlocks + blockStart) * headDim,
                   newData + (uint64_t)b * blockCount * headDim,
                   (uint64_t)blockCount * headDim * sizeof(uint16_t));
        }
        cache.CopyFrom(merged);
        cache.SetKVCache();
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
        int firstThread = pool->curActivateThreadInterval.first;
        int availableThreads = std::max(
            1, pool->curActivateThreadInterval.second - firstThread);
        constexpr int decodeDotTasks = 12;
        int threadNum = std::min(
            availableThreads, std::min(mixHc, decodeDotTasks));
        std::vector<DeepSeekV4HcPreDotsOp*> ops;
        ops.reserve(threadNum);
        int per = (mixHc + threadNum - 1) / threadNum;
        for (int i = 0; i < threadNum; i++) {
            int st = i * per;
            int end = std::min(mixHc, st + per);
            if (st >= end) {
                break;
            }
            ops.push_back(new DeepSeekV4HcPreDotsOp(
                xrow, fn, mixes, rsqrt, flatDim, st, end));
        }
        for (int i = 0; i < (int)ops.size(); i++) {
            pool->PushOp(firstThread + i, ops[i]);
        }
        for (int i = 0; i < (int)ops.size(); i++) {
            pool->Wait(firstThread + i);
            delete ops[i];
        }
    }

    struct DeepSeekV4HcPreBFloat16TokenTask : MultiThreadBaseOp {
        const uint16_t *input;
        const float *fn;
        const float *scale;
        const float *base;
        uint16_t *output;
        float *post;
        float *comb;
        int tokenStart, tokenEnd;
        int dim, flatDim, hcMult, mixHc, sinkhornIters;
        float eps, normEps;

        DeepSeekV4HcPreBFloat16TokenTask(
                const uint16_t *input, const float *fn,
                const float *scale, const float *base,
                uint16_t *output, float *post, float *comb,
                int tokenStart, int tokenEnd, int dim, int hcMult,
                int sinkhornIters, float eps, float normEps)
            : input(input), fn(fn), scale(scale), base(base), output(output),
              post(post), comb(comb), tokenStart(tokenStart),
              tokenEnd(tokenEnd), dim(dim), flatDim(hcMult * dim),
              hcMult(hcMult), mixHc((2 + hcMult) * hcMult),
              sinkhornIters(sinkhornIters), eps(eps), normEps(normEps) {}

        void Run() override {
            // Keep all per-token temporaries local to the worker.  The old path
            // launches `mixHc` pool jobs and waits for them for every token;
            // prefill has enough independent tokens to parallelize at this
            // outer level and compute all small-N dot products locally.
            std::vector<float> xrow(flatDim);
            std::vector<float> mixes(mixHc);
            std::vector<float> pre(hcMult);
            std::vector<float> combLocal(hcMult * hcMult);
            for (int token = tokenStart; token < tokenEnd; token++) {
                const uint16_t *src = input + (uint64_t)token * flatDim;
                double ss = 0.0;
                for (int k = 0; k < flatDim; k++) {
                    float value = DeepSeekV4HcPreBFloat16ToFloat(src[k]);
                    xrow[k] = value;
                    ss += (double)value * value;
                }
                float rsqrt = 1.0f /
                    std::sqrt((float)(ss / flatDim) + normEps);

                for (int m = 0; m < mixHc; m++) {
                    const float *weight = fn + (uint64_t)m * flatDim;
                    double value = 0.0;
                    for (int k = 0; k < flatDim; k++) {
                        value += (double)xrow[k] * weight[k];
                    }
                    mixes[m] = (float)value * rsqrt;
                }

                for (int h = 0; h < hcMult; h++) {
                    pre[h] = DeepSeekV4HcPreSigmoidFloat(
                        mixes[h] * scale[0] + base[h]) + eps;
                    post[(uint64_t)token * hcMult + h] =
                        2.0f * DeepSeekV4HcPreSigmoidFloat(
                            mixes[h + hcMult] * scale[1] +
                            base[h + hcMult]);
                }
                for (int r = 0; r < hcMult; r++) {
                    float rowMax = -FLT_MAX;
                    for (int c = 0; c < hcMult; c++) {
                        int idx = r * hcMult + c + 2 * hcMult;
                        combLocal[r * hcMult + c] =
                            mixes[idx] * scale[2] + base[idx];
                        rowMax = std::max(
                            rowMax, combLocal[r * hcMult + c]);
                    }
                    float rowSum = 0.0f;
                    for (int c = 0; c < hcMult; c++) {
                        float value = std::exp(
                            combLocal[r * hcMult + c] - rowMax);
                        combLocal[r * hcMult + c] = value;
                        rowSum += value;
                    }
                    for (int c = 0; c < hcMult; c++) {
                        combLocal[r * hcMult + c] =
                            combLocal[r * hcMult + c] / rowSum + eps;
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
                for (int iteration = 1; iteration < sinkhornIters;
                     iteration++) {
                    for (int r = 0; r < hcMult; r++) {
                        float rowSum = 0.0f;
                        for (int c = 0; c < hcMult; c++) {
                            rowSum += combLocal[r * hcMult + c];
                        }
                        for (int c = 0; c < hcMult; c++) {
                            combLocal[r * hcMult + c] /=
                                (rowSum + eps);
                        }
                    }
                    for (int c = 0; c < hcMult; c++) {
                        float colSum = 0.0f;
                        for (int r = 0; r < hcMult; r++) {
                            colSum += combLocal[r * hcMult + c];
                        }
                        for (int r = 0; r < hcMult; r++) {
                            combLocal[r * hcMult + c] /=
                                (colSum + eps);
                        }
                    }
                }
                memcpy(comb + (uint64_t)token * hcMult * hcMult,
                       combLocal.data(),
                       hcMult * hcMult * sizeof(float));

                uint16_t *dst = output + (uint64_t)token * dim;
                for (int d = 0; d < dim; d++) {
                    double value = 0.0;
                    for (int h = 0; h < hcMult; h++) {
                        value += (double)pre[h] *
                            xrow[(uint64_t)h * dim + d];
                    }
                    dst[d] = DeepSeekV4HcPreFloatToBFloat16(
                        (float)value);
                }
            }
        }
    };

    static void DeepSeekV4RunHcPreBFloat16TokenParallel(
            const Data &input, const float *fn, const float *scale,
            const float *base, Data &output, Data &postData, Data &combData,
            int tokens, int dim, int hcMult, int sinkhornIters,
            float eps, float normEps) {
        output.Allocate(false);
        postData.Allocate();
        combData.Allocate();
        auto *pool = GetAlivePool();
        int firstThread = pool->curActivateThreadInterval.first;
        int availableThreads = std::max(
            1, pool->curActivateThreadInterval.second - firstThread);
        int threadCount = std::min(tokens, availableThreads);
        std::vector<DeepSeekV4HcPreBFloat16TokenTask*> tasks;
        tasks.reserve(threadCount);
        for (int thread = 0; thread < threadCount; thread++) {
            int tokenStart = (int)((int64_t)tokens * thread /
                                   threadCount);
            int tokenEnd = (int)((int64_t)tokens * (thread + 1) /
                                 threadCount);
            tasks.push_back(new DeepSeekV4HcPreBFloat16TokenTask(
                (const uint16_t*)input.cpuData, fn, scale, base,
                (uint16_t*)output.cpuData, (float*)postData.cpuData,
                (float*)combData.cpuData, tokenStart, tokenEnd, dim,
                hcMult, sinkhornIters, eps, normEps));
        }
        if (threadCount == 1) {
            tasks[0]->Run();
            delete tasks[0];
            return;
        }
        for (int thread = 0; thread < threadCount; thread++) {
            pool->PushOp(firstThread + thread, tasks[thread]);
        }
        for (int thread = 0; thread < threadCount; thread++) {
            pool->Wait(firstThread + thread);
            delete tasks[thread];
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
        std::vector<float> fnStorage;
        const float *fn = nullptr;
        if (hcFn.dataType == DataType::FLOAT32 &&
            hcFn.dataDevice == DataDevice::CPU &&
            hcFn.cpuData != nullptr) {
            fn = (const float*)hcFn.cpuData;
        } else {
            fnStorage = DeepSeekV4HcPreReadFloatData(hcFn);
            fn = fnStorage.data();
        }
        float *scale = (float*)hcScale.cpuData;
        float *base = (float*)hcBase.cpuData;
        bool useBFloat16TokenParallel =
            input.dataType == DataType::BFLOAT16 &&
            input.dataDevice == DataDevice::CPU &&
            input.cpuData != nullptr && !input.multiDeviceData &&
            output.dataType == DataType::BFLOAT16 &&
            &input != &output && tokens > 1;
        if (useBFloat16TokenParallel) {
            DeepSeekV4RunHcPreBFloat16TokenParallel(
                input, fn, scale, base, output, postData, combData,
                tokens, dim, hcMult, sinkhornIters, eps, normEps);
            return;
        }

        struct DecodeWorkspace {
            std::vector<float> input;
            std::vector<float> output;
            std::vector<float> mixes;
            std::vector<float> pre;
            std::vector<float> comb;
        };
        static thread_local DecodeWorkspace workspace;
        DeepSeekV4HcPreReadFloatDataInto(input, workspace.input);
        workspace.output.assign((uint64_t)tokens * dim, 0.0f);
        workspace.mixes.resize(mixHc);
        workspace.pre.resize(hcMult);
        workspace.comb.resize(hcMult * hcMult);
        std::vector<float> &xv = workspace.input;
        std::vector<float> &y = workspace.output;
        std::vector<float> &mixes = workspace.mixes;
        std::vector<float> &pre = workspace.pre;
        std::vector<float> &combLocal = workspace.comb;
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
            DeepSeekV4HcPreComputeDotsCpu(xrow, fn, mixes.data(), rsqrt, flatDim, mixHc);
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

    void CpuDeepSeekV4HcPostOp::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                                        const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &residual = *(datas.find("residual")->second);
        Data &post = *(datas.find("post")->second);
        Data &comb = *(datas.find("comb")->second);
        Data &output = *(datas.find("output")->second);

        AssertInFastLLM(residual.dims.size() == 4, "DeepSeekV4HcPost error: residual's shape's size should be 4.\n");
        int bsz = residual.dims[0], seqlen = residual.dims[1], hcMult = residual.dims[2], dim = residual.dims[3];
        AssertInFastLLM(input.Count(0) == (uint64_t)bsz * seqlen * dim,
                        "DeepSeekV4HcPost error: input shape mismatch.\n");
        AssertInFastLLM(post.Count(0) == (uint64_t)bsz * seqlen * hcMult &&
                        comb.Count(0) == (uint64_t)bsz * seqlen * hcMult * hcMult,
                        "DeepSeekV4HcPost error: mix shape mismatch.\n");
        AssertInFastLLM(post.dataType == DataType::FLOAT32 && comb.dataType == DataType::FLOAT32,
                        "DeepSeekV4HcPost error: post and comb should be float32.\n");

        if (&residual == &output) {
            return;
        }
        output.dataType = input.dataType;
        output.Resize({bsz, seqlen, hcMult, dim});
    }

    struct DeepSeekV4HcPostBFloat16TargetOp : MultiThreadBaseOp {
        const uint16_t *input;
        const uint16_t *residual;
        const float *post;
        const float *comb;
        uint16_t *output;
        int dim, hcMult, token, target;

        DeepSeekV4HcPostBFloat16TargetOp(
            const uint16_t *input, const uint16_t *residual,
            const float *post, const float *comb, uint16_t *output,
            int dim, int hcMult, int token, int target
        ) : input(input), residual(residual), post(post), comb(comb),
            output(output), dim(dim), hcMult(hcMult), token(token),
            target(target) {}

        void Run() override {
            const uint16_t *xrow = input + (uint64_t)token * dim;
            const uint16_t *rrow =
                residual + (uint64_t)token * hcMult * dim;
            const float *postRow = post + (uint64_t)token * hcMult;
            const float *combRow =
                comb + (uint64_t)token * hcMult * hcMult;
            uint16_t *dst =
                output + ((uint64_t)token * hcMult + target) * dim;
            for (int d = 0; d < dim; d++) {
                double v = (double)postRow[target] *
                    DeepSeekV4HcPreBFloat16ToFloat(xrow[d]);
                for (int src = 0; src < hcMult; src++) {
                    v += (double)combRow[src * hcMult + target] *
                        DeepSeekV4HcPreBFloat16ToFloat(
                            rrow[(uint64_t)src * dim + d]);
                }
                dst[d] = DeepSeekV4HcPreFloatToBFloat16((float)v);
            }
        }
    };

    void CpuDeepSeekV4HcPostOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &input = *(datas.find("input")->second);
        Data &residual = *(datas.find("residual")->second);
        Data &postData = *(datas.find("post")->second);
        Data &combData = *(datas.find("comb")->second);
        Data &output = *(datas.find("output")->second);

        int bsz = residual.dims[0], seqlen = residual.dims[1], hcMult = residual.dims[2], dim = residual.dims[3];
        int tokens = bsz * seqlen;
        if (input.dataType == DataType::BFLOAT16 &&
            residual.dataType == DataType::BFLOAT16 &&
            output.dataType == DataType::BFLOAT16 &&
            input.cpuData != nullptr && residual.cpuData != nullptr &&
            postData.cpuData != nullptr && combData.cpuData != nullptr &&
            &output != &residual &&
            !DeepSeekV4HcPreEnvFlagEnabled(
                "FASTLLM_DSV4_DISABLE_CPU_HCPOST_FAST")) {
            output.Allocate(false);
            std::vector<MultiThreadBaseOp*> ops;
            ops.reserve(tokens * hcMult);
            for (int token = 0; token < tokens; token++) {
                for (int target = 0; target < hcMult; target++) {
                    ops.push_back(new DeepSeekV4HcPostBFloat16TargetOp(
                        (const uint16_t*)input.cpuData,
                        (const uint16_t*)residual.cpuData,
                        (const float*)postData.cpuData,
                        (const float*)combData.cpuData,
                        (uint16_t*)output.cpuData,
                        dim, hcMult, token, target));
                }
            }
            auto *pool = GetAlivePool();
            if ((int)ops.size() <= (int)pool->threads.size()) {
                for (int i = 0; i < (int)ops.size(); i++) {
                    pool->PushOp(i, ops[i]);
                }
                for (int i = 0; i < (int)ops.size(); i++) {
                    pool->Wait(i);
                    delete ops[i];
                }
            } else {
                DynamicScheduleTasks(ops);
            }
            return;
        }
        auto xv = DeepSeekV4HcPreReadFloatData(input);
        auto rv = DeepSeekV4HcPreReadFloatData(residual);
        auto post = DeepSeekV4HcPreReadFloatData(postData);
        auto comb = DeepSeekV4HcPreReadFloatData(combData);
        std::vector<float> y((uint64_t)tokens * hcMult * dim, 0.0f);

        for (int t = 0; t < tokens; t++) {
            const float *xrow = xv.data() + (uint64_t)t * dim;
            const float *rrow = rv.data() + (uint64_t)t * hcMult * dim;
            const float *postRow = post.data() + (uint64_t)t * hcMult;
            const float *combRow = comb.data() + (uint64_t)t * hcMult * hcMult;
            for (int target = 0; target < hcMult; target++) {
                for (int d = 0; d < dim; d++) {
                    double v = (double)postRow[target] * xrow[d];
                    for (int src = 0; src < hcMult; src++) {
                        v += (double)combRow[src * hcMult + target] * rrow[(uint64_t)src * dim + d];
                    }
                    y[((uint64_t)t * hcMult + target) * dim + d] = (float)v;
                }
            }
        }
        DeepSeekV4HcPreWriteFloatData(y, output);
    }

    void CpuDeepSeekV4StoreWindowKVCacheOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                                const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &kv = *(datas.find("input")->second);
        Data &windowKV = *(datas.find("cache")->second);
        int startPos = intParams.find("startPos") != intParams.end() ? intParams.find("startPos")->second : 0;
        int windowSize = intParams.find("windowSize") != intParams.end() ? intParams.find("windowSize")->second : 0;

        AssertInFastLLM(kv.dims.size() == 3 && kv.dims[1] > 0 && startPos >= 0 && windowSize > 0,
                        "DeepSeekV4StoreWindowKVCache error: invalid input.\n");
        int bsz = kv.dims[0], seqlen = kv.dims[1], headDim = kv.dims[2];
        auto kvValues = DeepSeekV4HcPreReadFloatData(kv);
        std::vector<float> cached((uint64_t)bsz * windowSize * headDim, 0.0f);
        if (startPos == 0 && seqlen <= windowSize) {
            for (int b = 0; b < bsz; b++) {
                memcpy(cached.data() + (uint64_t)b * windowSize * headDim,
                       kvValues.data() + (uint64_t)b * seqlen * headDim,
                       (uint64_t)seqlen * headDim * sizeof(float));
            }
        } else if (startPos == 0) {
            int cutoff = seqlen % windowSize;
            int first = windowSize - cutoff;
            for (int b = 0; b < bsz; b++) {
                const float *src = kvValues.data() + ((uint64_t)b * seqlen + seqlen - windowSize) * headDim;
                memcpy(cached.data() + ((uint64_t)b * windowSize + cutoff) * headDim,
                       src, (uint64_t)first * headDim * sizeof(float));
                if (cutoff > 0) {
                    memcpy(cached.data() + (uint64_t)b * windowSize * headDim,
                           src + (uint64_t)first * headDim,
                           (uint64_t)cutoff * headDim * sizeof(float));
                }
            }
        } else {
            for (int b = 0; b < bsz; b++) {
                memcpy(cached.data() + ((uint64_t)b * windowSize + (startPos % windowSize)) * headDim,
                       kvValues.data() + (uint64_t)b * seqlen * headDim,
                       (uint64_t)headDim * sizeof(float));
            }
        }
        windowKV.dataType = DataType::FLOAT32;
        windowKV.Resize({bsz, windowSize, headDim});
        DeepSeekV4HcPreWriteFloatData(cached, windowKV);
        windowKV.SetKVCache();
    }

    void CpuDeepSeekV4UpdateWindowKVCacheOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                                 const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &kv = *(datas.find("input")->second);
        Data &windowKV = *(datas.find("cache")->second);
        int startPos = intParams.find("startPos") != intParams.end() ? intParams.find("startPos")->second : 0;
        int windowSize = intParams.find("windowSize") != intParams.end() ? intParams.find("windowSize")->second : 0;

        AssertInFastLLM(kv.dims.size() == 3 && kv.dims[1] > 0 && startPos >= 0 && windowSize > 0,
                        "DeepSeekV4UpdateWindowKVCache error: invalid input.\n");
        int bsz = kv.dims[0], seqlen = kv.dims[1], headDim = kv.dims[2];
        AssertInFastLLM(kv.dataDevice == DataDevice::CPU && kv.cpuData != nullptr,
                        "DeepSeekV4UpdateWindowKVCache error: input has no CPU storage.\n");

        // Decode updates only one logical row, while the cache contains the
        // complete sliding window.  Updating a valid FP32 cache in place avoids
        // copying and rewriting windowSize * headDim values on every layer.
        // The per-element conversions below are the same conversions used by
        // DeepSeekV4HcPreReadFloatData, so the cache bits stay unchanged.
        bool canUpdateInPlace =
            windowKV.dataType == DataType::FLOAT32 &&
            windowKV.dataDevice == DataDevice::CPU &&
            windowKV.dims.size() == 3 &&
            windowKV.dims[0] == bsz &&
            windowKV.dims[1] == windowSize &&
            windowKV.dims[2] == headDim &&
            windowKV.cpuData != nullptr;
        if (canUpdateInPlace) {
            float *cacheData = (float*)windowKV.cpuData;
            for (int b = 0; b < bsz; b++) {
                for (int s = 0; s < seqlen; s++) {
                    uint64_t srcOffset =
                        ((uint64_t)b * seqlen + s) * headDim;
                    float *dst = cacheData +
                        ((uint64_t)b * windowSize +
                         ((startPos + s) % windowSize)) * headDim;
                    if (kv.dataType == DataType::FLOAT32) {
                        memcpy(dst,
                               (const float*)kv.cpuData + srcOffset,
                               (uint64_t)headDim * sizeof(float));
                    } else if (kv.dataType == DataType::FLOAT16) {
                        const uint16_t *src =
                            (const uint16_t*)kv.cpuData + srcOffset;
                        for (int d = 0; d < headDim; d++) {
                            dst[d] = half_to_float(src[d]);
                        }
                    } else if (kv.dataType == DataType::BFLOAT16) {
                        const uint16_t *src =
                            (const uint16_t*)kv.cpuData + srcOffset;
                        for (int d = 0; d < headDim; d++) {
                            dst[d] = DeepSeekV4HcPreBFloat16ToFloat(src[d]);
                        }
                    } else {
                        ErrorInFastLLM(
                            "DeepSeekV4UpdateWindowKVCache error: unsupported input dtype.\n");
                    }
                }
            }
            windowKV.SetKVCache();
            return;
        }

        auto kvValues = DeepSeekV4HcPreReadFloatData(kv);
        std::vector<float> cached;
        if (windowKV.dataType == DataType::FLOAT32 && windowKV.dims.size() == 3 &&
            windowKV.dims[0] == bsz && windowKV.dims[1] == windowSize && windowKV.dims[2] == headDim &&
            windowKV.cpuData != nullptr) {
            cached = DeepSeekV4HcPreReadFloatData(windowKV);
        }
        if ((uint64_t)cached.size() != (uint64_t)bsz * windowSize * headDim) {
            cached.assign((uint64_t)bsz * windowSize * headDim, 0.0f);
        }
        for (int b = 0; b < bsz; b++) {
            for (int s = 0; s < seqlen; s++) {
                memcpy(cached.data() + ((uint64_t)b * windowSize + ((startPos + s) % windowSize)) * headDim,
                       kvValues.data() + ((uint64_t)b * seqlen + s) * headDim,
                       (uint64_t)headDim * sizeof(float));
            }
        }
        windowKV.dataType = DataType::FLOAT32;
        windowKV.Resize({bsz, windowSize, headDim});
        DeepSeekV4HcPreWriteFloatData(cached, windowKV);
        windowKV.SetKVCache();
    }
}
