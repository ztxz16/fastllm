//
// Created by huangyuyang on 5/11/23.
//

#include "fastllm.h"

#include <cstring>
#include <cmath>
#include <cfloat>
#include <thread>

#ifdef __aarch64__
#include <arm_neon.h>
#include "armMath.h"
#endif

#ifdef __AVX__
#include "immintrin.h"
#endif

#ifdef USE_CUDA
#include "fastllm-cuda.h"
#endif

namespace fastllm {
    double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds> (time2 - time1);
        return double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
    };

    void ErrorInFastLLM(const std::string &error) {
        printf("FastLLM Error: %s\n", error.c_str());
        throw error;
    }

    void AssertInFastLLM(bool condition, const std::string &error) {
        if (!condition) {
            ErrorInFastLLM(error);
        }
    }

    static int threads = 4;

    void SetThreads(int t) {
        threads = t;
    }

    struct FileBuffer {
        FILE *f;

        FileBuffer (const std::string &fileName) {
            f = fopen(fileName.c_str(), "rb");
        }

        int ReadInt() {
            int v;
            if (fread(&v, 1, 4, f) != 4) {
                ErrorInFastLLM("FileBuffer.ReadInt error.\n");
            };
            return v;
        }

        float ReadFloat() {
            float v;
            if (fread(&v, 1, 4, f) != 4) {
                ErrorInFastLLM("FileBuffer.ReadFloat error.\n");
            };
            return v;
        }

        std::string ReadString() {
            int len = ReadInt();
            std::string ret = "";
            char *v = new char[len + 5];
            v[len] = 0;
            if (fread(v, 1, len, f) != len) {
                ErrorInFastLLM("FileBuffer.ReadString error.\n");
            }
            return v;
        }

        void ReadBytes(uint8_t *buffer, uint64_t bytes) {
            if (fread(buffer, 1, bytes, f) != bytes) {
                ErrorInFastLLM("FileBuffer.ReadBytes error.\n");
            }
        }

        ~FileBuffer() {
            fclose(f);
        }
    };

    struct FileWriter {
        FILE *f;

        FileWriter (const std::string &fileName) {
            f = fopen(fileName.c_str(), "wb");
        }

        void WriteInt(int v) {
            if (fwrite(&v, 1, 4, f) != 4) {
                ErrorInFastLLM("FileWriter.WriteInt error.\n");
            };
        }

        void WriteFloat(float v) {
            if (fwrite(&v, 1, 4, f) != 4) {
                ErrorInFastLLM("FileWriter.WriteFloat error.\n");
            };
        }

        void WriteString(const std::string &s) {
            WriteInt((int)s.size());
            if (fwrite(s.c_str(), 1, (int)s.size(), f) != (int)s.size()) {
                ErrorInFastLLM("FileWriter.WriteString Error.\n");
            }
        }

        void WriteBytes(uint8_t *buffer, uint64_t bytes) {
            if (fwrite(buffer, 1, bytes, f) != bytes) {
                ErrorInFastLLM("FileWriter.WriteBytes error.\n");
            }
        }

        ~FileWriter() {
            fclose(f);
        }
    };

#ifdef __AVX__
	static inline float Floatsum(const __m256 a) {
		__m128 res = _mm256_extractf128_ps(a, 1);
		res = _mm_add_ps(res, _mm256_castps256_ps128(a));
		res = _mm_add_ps(res, _mm_movehl_ps(res, res));
		res = _mm_add_ss(res, _mm_movehdup_ps(res));
		return _mm_cvtss_f32(res);
	}

	static inline int I32sum(const __m256i a) {
		const __m128i sum128 = _mm_add_epi32(_mm256_extractf128_si256(a, 0), _mm256_extractf128_si256(a, 1));
		const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
		const __m128i sum64 = _mm_add_epi32(hi64, sum128);
		const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
		return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
	}

	int DotU8U8(uint8_t *a, uint8_t *b, int n) {
		__m256i acc = _mm256_setzero_si256();

		int i = 0;
		int ans = 0;
		for (; i + 31 < n; i += 32) {
			__m256i bx = _mm256_loadu_si256((const __m256i *) (a + i));
			__m256i by = _mm256_loadu_si256((const __m256i *) (b + i));

			__m256i mx0 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(bx, 0));
			__m256i mx1 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(bx, 1));

			__m256i my0 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(by, 0));
			__m256i my1 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(by, 1));

			acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx0, my0));
			acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx1, my1));
		}
		for (; i < n; i++) {
			ans += a[i] * b[i];
		}

		return ans + I32sum(acc);
	};

	int DotU4U8(uint8_t *a, uint8_t *b, int n) {
		int value = 0, j = 0;
		for (; j + 1 < n; j += 2) {
			value += (a[j / 2] >> 4) * b[j];
			value += (a[j / 2] & 0xF) * b[j + 1];
		}
		//return value;

		__m256i acc = _mm256_setzero_si256();

		int i = 0;
		int ans = 0;
		const __m256i lowMask = _mm256_set1_epi8(0xf);
		for (; i + 31 < n; i += 32) {
			__m128i orix = _mm_loadu_si128((const __m128i *) (a + i / 2));
			__m256i bytex = _mm256_set_m128i(_mm_srli_epi16(orix, 4), orix);
			__m256i bx = _mm256_and_si256(lowMask, bytex);
			__m256i by = _mm256_loadu_si256((const __m256i *) (b + i));
			__m256i mx0 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(bx, 0));
			__m256i mx1 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(bx, 1));

			__m256i my0 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(by, 0));
			__m256i my1 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(by, 1));

			acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx0, my0));
			acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx1, my1));
		}
		for (; i < n; i++) {
			ans += a[i] * b[i];
		}

		return ans + I32sum(acc);
	};
#endif

    Data::Data(fastllm::DataType type) {
        this->dataType = type;
        this->UpdateUnitSize();
    }

    Data::Data(fastllm::DataType type, const std::vector<int> &dims) {
        this->dataType = type;
        Resize(dims);
    }

    Data::Data(fastllm::DataType type, const std::vector<int> &dims, const std::vector<float> &data) : Data::Data(type, dims) {
        this->Allocate();
        if (type == DataType::FLOAT32) {
            std::memcpy(this->cpuData, data.data(), this->GetBytes());
        }
    }

    Data::Data(const Data &ori) {
        CopyFrom(ori);
    }

    void Data::CopyFrom(const Data &ori) {
        if (ori.dims != this->dims || this->cpuData == nullptr) {
            if (ori.dims.size() == 0) {
                delete[] this->cpuData;
                this->dataType = ori.dataType;
                this->UpdateUnitSize();
                this->dims.resize(0);
                this->cpuData = nullptr;
                return;
            }
            this->dataType = ori.dataType;
            this->Resize(ori.dims);
            this->Allocate();
        }
        std::memcpy(this->cpuData, ori.cpuData, this->GetBytes());
    }

    uint64_t Data::Count(int i) const {
        if (i >= this->dims.size()) {
            return 1;
        }
        if (i - 1 >= 0 && i - 1 < this->strides.size()) {
            return this->strides[i - 1];
        }
        return this->dims[i] * this->strides[i];
    }

    void Data::UpdateUnitSize() {
        if (this->dataType == DataType::FLOAT32) {
            this->unitSize = 4;
        } else if (this->dataType == DataType::BFLOAT16 || this->dataType == DataType::INT16) {
            this->unitSize = 2;
        } else if (this->dataType == DataType::INT8) {
            this->unitSize = 1;
        } else if (this->dataType == DataType::INT4) {
            this->unitSize = 1;
            this->unitSizeDiv = 2;
        } else if (this->dataType == DataType::INT2) {
            this->unitSize = 1;
            this->unitSizeDiv = 4;
        } else if (this->dataType == DataType::BIT) {
            this->unitSize = 1;
            this->unitSizeDiv = 8;
        }
    }

    void Data::Resize(const std::vector<int> &dims) {
        this->dims = dims;
        this->UpdateUnitSize();

        if (this->expansionDims.size() == 0) {
            this->strides.resize(dims.size(), 1);
            this->strides.back() = 1;
            for (int i = this->dims.size() - 2; i >= 0; i--) {
                this->strides[i] = this->dims[i + 1] * this->strides[i + 1];
            }
        }
    }

    void Data::Reshape(const std::vector<int> &dims) {
        std::vector <int> outputDims = dims;
        uint64_t old = 1;
        for (int i : this->dims) {
            old *= i;
        }
        int index = -1;
        uint64_t mul = 1;
        for (int i = 0; i < dims.size(); i++) {
            if (dims[i] < 0) {
                if (index == -1) {
                    index = i;
                } else {
                    ErrorInFastLLM("Reshape error.\n");
                }
            } else {
                mul *= dims[i];
            }
        }
        outputDims = dims;
        if (index == -1) {
            AssertInFastLLM(mul == old, "Reshape error.\n");
        } else {
            AssertInFastLLM(mul != 0, "Reshape error.\n");
            AssertInFastLLM(old % mul == 0, "Reshape error.\n");
            outputDims[index] = old / mul;
        }
        Resize(outputDims);
    }

    uint64_t Data::GetBytes() const {
        return (this->strides[0] * this->dims[0] * this->unitSize - 1) / this->unitSizeDiv + 1;
    }

    void Data::Allocate() {
        if (Count(0) > expansionSize) {
            delete[] this->cpuData;
            this->cpuData = new uint8_t[GetBytes()];
            expansionSize = Count(0);
        }
    }

    void Data::Allocate(float v) {
        AssertInFastLLM(this->dataType == DataType::FLOAT32, "Allocate error: Data's type should be float32.\n");
        this->Allocate();
        float *f = (float*)cpuData;
        std::fill(f, f + Count(0), v);
    }

    void Data::Expansion(uint64_t size) {
        AssertInFastLLM(Count(0) <= size, "Expansion error: real size should <= expansion size.\n");
        this->expansionSize = size;

        if (this->cpuData != nullptr) {
            uint8_t *old = this->cpuData;
            this->cpuData = new uint8_t[(size * unitSize - 1) / unitSizeDiv + 1];
            memcpy(this->cpuData, old, GetBytes());
            delete[] old;
        } else {
            this->cpuData = new uint8_t[(size * unitSize - 1) / unitSizeDiv + 1];
        }
    }

    void Data::Expansion(const std::vector<int> &dims) {
        if (this->dims.size() == 0) {
            this->strides.resize(dims.size(), 1);
            this->strides.back() = 1;
            for (int i = dims.size() - 2; i >= 0; i--) {
                this->strides[i] = dims[i + 1] * this->strides[i + 1];
            }
            this->expansionSize = this->strides[0] * dims[0];
            this->expansionDims = dims;
            this->cpuData = new uint8_t[(this->expansionSize * unitSize - 1) / unitSizeDiv + 1];
            return;
        }

        AssertInFastLLM(dims.size() == this->dims.size(), "Expansion error: real dims's size should equal to expansion dims's size.\n");
        for (int i = 0; i < dims.size(); i++) {
            AssertInFastLLM(dims[i] == -1 || dims[i] >= this->dims[i], "Expansion error: real size should <= expansion size.\n");
        }

        int axis = -1;
        for (int i = 0; i < this->dims.size(); i++) {
            if (this->dims[i] < dims[i]) {
                axis = i;
                break;
            }
        }

        uint64_t oldBytes = GetBytes();
        int input1Stride = this->Count(axis);

        this->strides.resize(dims.size(), 1);
        this->strides.back() = 1;
        for (int i = this->dims.size() - 2; i >= 0; i--) {
            this->strides[i] = std::max(this->dims[i + 1], dims[i + 1]) * this->strides[i + 1];
        }
        this->expansionSize = this->strides[0] * std::max(this->dims[0], dims[0]);
        this->expansionDims = dims;
        if (this->cpuData != nullptr) {
            uint8_t *old = this->cpuData;
            this->cpuData = new uint8_t[(this->expansionSize * unitSize - 1) / unitSizeDiv + 1];
            int outer = this->Count(0) / this->Count(axis);
            int input0Stride = this->Count(axis);
            int inner = this->strides[axis];
            int unitSize = this->unitSize;
            for (int o = 0; o < outer; o++) {
                memcpy(this->cpuData + o * input0Stride * unitSize,
                       old + o * input1Stride * unitSize,
                       this->dims[axis] * inner * unitSize);
            }

            delete[] old;
        } else {
            this->cpuData = new uint8_t[(this->expansionSize * unitSize - 1) / unitSizeDiv + 1];
        }
    }

    Data::~Data() {
        delete[] this->cpuData;
    }

    void Data::Print() const {
        printf("shape: ");
        for (int i : this->dims) {
            printf("%d ", i);
        }
        printf("\ndata: ");
        /*
        int len = Count(0);
        if (len < 20) {
            for (int i = 0; i < len; i++) {
                printf("%f ", ((float*)cpuData)[i]);
            }
        } else {
            for (int i = 0; i < 10; i++) {
                printf("%f ", ((float *) cpuData)[i]);
            }
            printf("... ");
            for (int i = 0; i < 10; i++) {
                printf("%f ", ((float *) cpuData)[len - 10 + i]);
            }
        }
        printf("\n");
         */
        int n = Count(0) / dims.back(), m = dims.back();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 10 && j < m; j++) {
                printf("%f ", ((float*)cpuData)[i * m + j]);
            }
            printf("... ");
            for (int j = 0; j < 10 && j < m; j++) {
                printf("%f ", ((float*)cpuData)[i * m + (m - 10 + j)]);
            }
            printf("\n");
        }
    }

    void Data::Permute(const std::vector<int> &axis) {
        AssertInFastLLM(this->dataType == DataType::FLOAT32, "Permute error: datatype should be float32.");
        AssertInFastLLM(axis.size() == this->dims.size(), "Permute error: axis's size should be equal to data's shape's size.");

        if ((axis == std::vector <int>{1, 2, 0} || axis == std::vector <int>{1, 0, 2}) && this->dims[0] == 1) {
            std::vector<int> new_dims;
            for (int i = 0; i < axis.size(); i++) {
                new_dims.push_back(this->dims[axis[i]]);
            }
            this->Resize(new_dims);
            return;
        }

        auto tmp = new Data();
        fastllm::Permute(*this, axis, *tmp);

        memcpy(this->cpuData, tmp->cpuData, unitSize * this->Count(0));
        this->Resize(tmp->dims);
        delete tmp;
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

    void Permute(const Data &input, const std::vector<int> &axis, Data &output) {
        AssertInFastLLM(input.dataType == DataType::FLOAT32, "Permute error: datatype should be float32.");
        AssertInFastLLM(axis.size() == input.dims.size(), "Permute error: axis's size should be equal to data's shape's size.");
        std::vector<int> new_dims;
        for (int i = 0; i < axis.size(); i++) {
            new_dims.push_back(input.dims[axis[i]]);
        }

        output.dataType = input.dataType;
        output.Resize(new_dims);
        output.Allocate();

        float *tmpData = (float *) output.cpuData;
        float *curData = (float *) input.cpuData;

        if (axis == std::vector <int> {1, 2, 0}) {
            int n = input.dims[0];
            int m = input.Count(1);

            int threadNum = 1;
            int per = m / threadNum;
            int cur = 0;
            std::vector <std::thread*> threads;
            for (int i = 0; i < threadNum - 1; i++) {
                int end = cur + per + (cur + per * (threadNum - i) < m);
                threads.push_back(new std::thread(&Transpose, tmpData + cur * n, curData + cur, n, m, n, end - cur));
                cur = end;
            }
            Transpose(tmpData + cur * n, curData + cur, n, m, n, m - cur);
            for (int i = 0; i < threadNum - 1; i++) {
                threads[i]->join();
                delete threads[i];
            }
        } else if (axis == std::vector <int> {1, 0, 2}) {
            int n = input.dims[0];
            int m = input.dims[1];
            int k = input.dims[2];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    memcpy(tmpData + (j * n + i) * k, curData + (i * m + j) * k, k * sizeof(float));
                }
            }
        } else if (axis == std::vector <int> {2, 0, 1, 3}) {
            int n = input.dims[0] * input.dims[1];
            int m = input.dims[2];
            int k = input.dims[3];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    memcpy(tmpData + (j * n + i) * k, curData + (i * m + j) * k, k * sizeof(float));
                }
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
            for (int i = 0; i < count; ++i) {
                tmpData[i] = curData[oldPos[i]];
            }

            delete[] oldPos;
        }
    }

    void Data::CalcWeightSum() {
        if (this->weightSum.size() > 0) {
            return;
        }
        int n = this->dims[0], m = this->dims[1];
        if (this->dataType == DataType::INT8) {
            weightSum.resize(n);
            for (int i = 0; i < n; i++) {
                int j = 0;
#ifdef __AVX__
                __m256i acc = _mm256_setzero_si256();
                const __m256i ones = _mm256_set1_epi16(1);
                for (; j + 31 < m; j += 32) {
                    __m256i ax = _mm256_loadu_si256((const __m256i *) (cpuData + i * m + j));
                    __m256i mx0 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(ax, 0));
                    __m256i mx1 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(ax, 1));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx0, ones));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx1, ones));
                }
                weightSum[i] += I32sum(acc);
#endif
#ifdef __aarch64__
                uint32x4_t sum0 = {0, 0, 0, 0};
                for (; j + 7 < m; j += 8) {
                    uint8x8_t ori = vld1_u8(cpuData + (i * m + j));
                    uint16x4_t sa = vpaddl_u8 (ori);
                    sum0 = vaddw_u16(sum0, sa);
                }
                weightSum[i] += sum0[0] + sum0[1] + sum0[2] + sum0[3];
#endif
                for (; j < m; j++) {
                    weightSum[i] += cpuData[i * m + j];
                }
            }
        } else if (this->dataType == DataType::INT4) {
            weightSum.resize(n);
            for (int i = 0; i < n; i++) {
                int j = 0;
#ifdef __aarch64__
                uint8x8_t maskHigh = vdup_n_u8(0xF0);
                uint8x8_t maskLow = vdup_n_u8(0xF);
                uint32x4_t sum0 = {0, 0, 0, 0};

                for (; j + 15 < m; j += 16) {
                    uint8x8_t ori = vld1_u8(cpuData + (i * m + j) / 2);
                    uint8x8_t va = vand_u8(ori, maskLow);
                    uint8x8_t vb = vshr_n_u8(vand_u8(ori, maskHigh), 4);

                    uint16x4_t sa = vpaddl_u8 (va);
                    uint16x4_t sb = vpaddl_u8 (vb);

                    sum0 = vaddw_u16(sum0, vadd_u16(sa, sb));
                }
                weightSum[i] += sum0[0] + sum0[1] + sum0[2] + sum0[3];
#endif
#ifdef __AVX__
	            __m256i acc = _mm256_setzero_si256();
	            const __m256i lowMask = _mm256_set1_epi8(0xf);
	            const __m256i ones = _mm256_set1_epi16(1);
	            for (; j + 31 < m; j += 32) {
		            __m128i orix = _mm_loadu_si128((const __m128i *) (cpuData + (i * m + j) / 2));
		            __m256i bytex = _mm256_set_m128i(_mm_srli_epi16(orix, 4), orix);
		            __m256i bx = _mm256_and_si256(lowMask, bytex);

		            __m256i mx0 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(bx, 0));
		            __m256i mx1 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(bx, 1));

		            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx0, ones));
		            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx1, ones));
	            }
	            weightSum[i] += I32sum(acc);
#endif
                for (; j + 1 < m; j += 2) {
	                int id = (i * m + j) / 2;
	                weightSum[i] += (cpuData[id] & 0xF) + (cpuData[id] >> 4);
                }
                for (; j < m; j++) {
                    int id = (i * m + j) / 2;
                    if ((i * m + j) % 2) {
                        weightSum[i] += (cpuData[id] & 0xF);
                    } else {
                        weightSum[i] += (cpuData[id] >> 4);
                    }
                }
            }
        }
    }

    Tokenizer::TrieNode::TrieNode() {
        this->tokenId = -999999;
    }

    Tokenizer::Tokenizer() {
        root = new TrieNode();
    }

    Tokenizer::~Tokenizer() {
        Clear();
        delete root;
    }

    void Tokenizer::Clear() {
        std::vector <TrieNode*> q;
        q.push_back(root);
        for (int i = 0; i < q.size(); i++) {
            TrieNode *now = q[i];
            for (auto it : now->next) {
                q.push_back(it.second);
            }
        }
        root = new TrieNode();
        tokenToStringDict.clear();
    }

    void Tokenizer::Insert(const std::string &s, int tokenId) {
        TrieNode *now = this->root;
        for (int i = 0; i < s.size(); i++) {
            if (now->next.find(s[i]) == now->next.end()) {
                now->next[s[i]] = new TrieNode();
            }
            now = now->next[s[i]];
        }
        now->tokenId = tokenId;
        tokenToStringDict[tokenId] = s;
    }

    Data Tokenizer::Encode(const std::string &s) {
        std::vector <float> v;
        for (int i = 0; i < s.size(); i++) {
            int tokenId = -999999, pos = i - 1;
            TrieNode *now = this->root;
            for (int j = i; j < s.size(); j++) {
                if (now->next.find(s[j]) != now->next.end()) {
                    now = now->next[s[j]];
                    if (now->tokenId != -999999) {
                        tokenId = now->tokenId;
                        pos = j;
                    }
                } else {
                    break;
                }
            }
            if (pos >= i) {
                i = pos;
                v.push_back(tokenId);
                //printf("%d ", tokenId);
            }
        }
        //printf("\n");

        return Data (DataType::FLOAT32, {1, (int)v.size()}, v);
    }

    std::string Tokenizer::Decode(const Data &data) {
        std::string ret = "";
        for (int i = 0; i < data.Count(0); i++) {
            std::string &s = tokenToStringDict[(int) ((float *) data.cpuData)[i]];
            if (s == "<n>") {
                ret += "\n";
            } else if (s == "<|tab|>") {
                s = "\t";
            } else {
                ret += s;
            }
        }

        std::string blank = "";
        blank += 226, blank += 150, blank += 129;
        while (true) {
            std::string::size_type pos(0);
            if ((pos = ret.find(blank)) != std::string::npos)
                ret.replace(pos, blank.length(), " ");
            else break;
        }
	    int pos = ret.find("<|blank_");
	    if (pos != -1) {
		    int space_num = atoi(ret.substr(8, ret.size() - 10).c_str());
		    return std::string(space_num, ' ');
	    }
        return ret;
    }

    void WeightMap::LoadFromFile(const std::string &fileName) {
        FileBuffer buffer(fileName);
        this->versionId = buffer.ReadInt();

        int vocabLen = buffer.ReadInt();
        for (int i = 0; i < vocabLen; i++) {
            int len = buffer.ReadInt();
            std::string x = "";
            for (int j = 0; j < len; j++) {
                x += buffer.ReadInt();
            }
            int id = buffer.ReadInt();
            tokenizer.Insert(x, id);
        }

        int len = buffer.ReadInt();
        for (int i = 0; i < len; i++) {
            std::string name = buffer.ReadString();
            //printf("%s\n", name.c_str());
            int dimsSize = buffer.ReadInt();
            //printf("size = %d\n", dimsSize);
            std::vector <int> dims;
            for (int j = 0; j < dimsSize; j++) {
                int x = buffer.ReadInt();
                dims.push_back(x);
                //printf("%d\n", x);
            }
            DataType dataType = (DataType)buffer.ReadInt();
            weight[name] = Data(dataType, dims);
            weight[name].Allocate();

            if (dataType == DataType::FLOAT32 || dataType == DataType::BFLOAT16) {
                buffer.ReadBytes(weight[name].cpuData, weight[name].GetBytes());
            } else if (dataType == DataType::INT8 || dataType == DataType::INT4) {
                int bit = (dataType == DataType::INT4 ? 4 : 8);
                weight[name].perChannelAxis = buffer.ReadInt();
                int k = weight[name].perChannelAxis == -1 ? 1 : dims[weight[name].perChannelAxis];
                weight[name].perChannelsConfigs.resize(k);
                weight[name].zeros.resize(k);
                weight[name].scales.resize(k);
                for (int i = 0; i < k; i++) {
                    float minValue = buffer.ReadFloat();
                    float maxValue = buffer.ReadFloat();
                    weight[name].perChannelsConfigs[i] = LowBitConfig(minValue, maxValue, bit);
                    weight[name].zeros[i] = weight[name].perChannelsConfigs[i].zeroPoint;
                    weight[name].scales[i] = weight[name].perChannelsConfigs[i].scale;
                }
                buffer.ReadBytes(weight[name].cpuData, weight[name].GetBytes());
            }

            printf("Load (%d / %d) \r", (i + 1), len);
            fflush(stdout);
        }
        printf("\n");
        fflush(stdout);
        return;
    }

    void WeightMap::SaveLowBitModel(const std::string &fileName, int bit) {
        AssertInFastLLM(fileName != "", "Error: output's name shouldn't be empty.\n");
        AssertInFastLLM(bit == 4 || bit == 8, "Error: only support 8 bit or 4 bit model.\n");
        FileWriter buffer(fileName);
        buffer.WriteInt(this->versionId);

        // 写入词表
        buffer.WriteInt((int)tokenizer.tokenToStringDict.size());
        for (auto &it : tokenizer.tokenToStringDict) {
            buffer.WriteInt((int)it.second.size());
            for (int i = 0; i < it.second.size(); i++) {
                buffer.WriteInt((int)it.second[i]);
            }
            buffer.WriteInt(it.first);
        }

        // 写入权重
        buffer.WriteInt((int)weight.size());
        for (auto &it : weight) {
            buffer.WriteString(it.first);
            Data &data = it.second;
            buffer.WriteInt((int)data.dims.size());
            for (int i : data.dims) {
                buffer.WriteInt(i);
            }

            if (data.weightType == WeightType::NONE) {
                // 普通权重，直接写入浮点数据
                buffer.WriteInt((int)DataType::FLOAT32);
                buffer.WriteBytes(data.cpuData, data.GetBytes());
            } else if (data.weightType == WeightType::EMBEDDING) {
                // Embedding权重，存储成BF16
                buffer.WriteInt((int)DataType::BFLOAT16);
                int len = data.Count(0);
                std::vector <uint16_t> uDatas;
                uDatas.resize(len);
                for (int i = 0; i < len; i++) {
                    uDatas[i] = ((uint16_t *)data.cpuData)[i * 2 + 1];
                }
                buffer.WriteBytes((uint8_t*)uDatas.data(), len * sizeof(uint16_t));
            } else if (data.weightType == WeightType::LINEAR) {
                // Linear层权重，分通道量化之
                int k = data.dims[0], m = data.dims[1];
                int threadNum = 8;
                int per = k / threadNum;
                int cur = 0;
                std::vector <std::thread*> threads;
                std::vector <LowBitConfig> configs;
                std::vector <uint8_t> uDatas;
                configs.resize(k);

                int bytes = k * m;
                if (bit == 4) {
                    bytes = (k * m + 1) / 2;
                }
                uDatas.resize(bytes);
                for (int i = 0; i < threadNum; i++) {
                    int end = cur + per;
                    if (i == threadNum - 1) {
                        end = k;
                    }
                    threads.push_back(new std::thread([&bit] (int st, int end, int m,
                            float *f, uint8_t *u8, LowBitConfig *configs) {
                        for (int i = st; i < end; i++) {
                            float minValue = 1e9, maxValue = -1e9;
                            for (int j = 0; j < m; j++) {
                                minValue = std::min(minValue, f[i * m + j]);
                                maxValue = std::max(maxValue, f[i * m + j]);
                            }
                            if (bit == 8) {
                                configs[i] = LowBitConfig(minValue, maxValue, 8);
                                for (int j = 0; j < m; j++) {
                                    u8[i * m + j] = configs[i].quantization(f[i * m + j]);
                                }
                            } else {
                                configs[i] = LowBitConfig(minValue, maxValue, 4);
                                for (int j = 0; j < m; j++) {
                                    int id = (i * m + j) / 2;
                                    uint8_t value = configs[i].quantization(f[i * m + j]);
                                    if ((i * m + j) % 2) {
                                        u8[id] = (u8[id] & 0xF0) | value;
                                    } else {
                                        u8[id] = (u8[id] & 0xF) | (value << 4);
                                    }
                                }
                            }
                        }
                    }, cur, end, m, (float*)data.cpuData, uDatas.data(), configs.data()));
                    cur = end;
                }
                for (int i = 0; i < threadNum; i++) {
                    threads[i]->join();
                    delete threads[i];
                }

                buffer.WriteInt(bit == 8 ? (int)DataType::INT8 : (int)DataType::INT4);
                buffer.WriteInt(0); // 按通道0分通道量化
                for (int i = 0; i < k; i++) {
                    buffer.WriteFloat(configs[i].min);
                    buffer.WriteFloat(configs[i].max);
                }
                buffer.WriteBytes(uDatas.data(), bytes);
            }
        }

        return;
    }

    Data &WeightMap::operator[](const std::string &key) {
        return weight[key];
    }

    //a = [n, m], b = [k, m], c = aT(b') = [n, k]
    void Multiply(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k, int kstride) {
#ifdef __ARM_FEATURE_DOTPROD
        int block = 0;
        for (; block < n; block++) {
            uint8_t *weightWalk = b;
            uint8_t *inputStart = a + block * m;

            for (int i = 0; i < k; i++) {
                int value = 0;
                uint8_t *inputWalk = inputStart;
                int j = 0;
                uint32x4_t sum0 = {0, 0, 0, 0};
                for (; j + 31 < m; j += 32) {
                    uint8x16_t vi = vld1q_u8(inputWalk);
                    uint8x16_t vi0 = vld1q_u8(inputWalk + 16);
                    uint8x16_t vw = vld1q_u8(weightWalk);
                    uint8x16_t vw0 = vld1q_u8(weightWalk + 16);
                    sum0 = vdotq_u32(sum0, vi, vw);
                    sum0 = vdotq_u32(sum0, vi0, vw0);
                    inputWalk += 32;
                    weightWalk += 32;
                }

                value += sum0[0] + sum0[1] + sum0[2] + sum0[3];
                for (; j < m; j++) {
				    value += (int)(*(weightWalk++)) * (*(inputWalk++));
			    }
                c[block * kstride + i] = value;
            }
        }
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
#elif defined(__AVX__)
        int block = 0;
	    for (; block < n; block++) {
		    uint8_t *weightWalk = b;
		    uint8_t *inputStart = a + block * m;

		    for (int i = 0; i < k; i++) {
			    int value = 0;
			    uint8_t *inputWalk = inputStart;

                c[block * kstride + i] = DotU8U8(inputWalk, weightWalk, m);
                inputWalk += m;
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

    //a = [n, m], b = [k, m], c = aT(b') = [n, k]
    void MultiplyInt4(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k, int kstride,
                      int *weightSums, int *weightZeros, float *scales, float *bias, LowBitConfig *config,
                      int *inputSums) {
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
#elif defined(__AVX__)
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

                value -= weightSums[i] * config->zeroPoint;
                value -= inputSum * weightZeros[i];
                value += (int)config->zeroPoint * weightZeros[i] * m;

                ((float*)c)[block * kstride + i] = scales[i] * config->scale * value +
                                               (bias == nullptr ? 0.0 : bias[i]);
            }
        }
    }

    //a = [n, m], b = [k, m], c = aT(b') = [n, k]
    void MultiplyMultiThread(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k, int threadNum) {
        int per = k / threadNum;
        int cur = 0;
        std::vector <std::thread*> threads;
        for (int i = 0; i < threadNum - 1; i++) {
            int end = cur + per + (cur + per * (threadNum - i) < k);
            threads.push_back(new std::thread(&Multiply, a, b + cur * m, c + cur, n, m, end - cur, k));
            cur = end;
        }
        Multiply(a, b + cur * m, c + cur, n, m, k - cur, k);
        for (int i = 0; i < threadNum - 1; i++) {
            threads[i]->join();
            delete threads[i];
        }
    }

    //a = [n, m], b = [k, m], c = aT(b') = [n, k]
    void MultiplyInt4MultiThread(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k,
                                 int *weightSums, int *weightZeros, float *scales, float *bias, LowBitConfig &config, int threadNum) {
        std::vector <int> inputSums;
        for (int i = 0; i < n; i++) {
            int sum = 0;
            for (int j = 0; j < m; j++) {
                sum += a[i * m + j];
            }
            inputSums.push_back(sum);
        }
        int per = k / threadNum;
        int cur = 0;
        std::vector <std::thread*> threads;
        for (int i = 0; i < threadNum - 1; i++) {
            int end = cur + per + (cur + per * (threadNum - i) < k);
            threads.push_back(new std::thread(&MultiplyInt4, a, b + cur * m / 2, c + cur, n, m, end - cur, k,
                                              weightSums + cur, weightZeros + cur, scales + cur,
                                              (bias == nullptr ? (float*)nullptr : bias + cur), &config, inputSums.data()));
            cur = end;
        }
        MultiplyInt4(a, b + cur * m / 2, c + cur, n, m, k - cur, k,
                     weightSums + cur, weightZeros + cur, scales + cur,
                     (bias == nullptr ? (float*)nullptr : bias + cur), &config, inputSums.data());
        for (int i = 0; i < threadNum - 1; i++) {
            threads[i]->join();
            delete threads[i];
        }
    }

    void Embedding(const Data &input, Data &weight, Data &output) {
        AssertInFastLLM(weight.dims.size() == 2, "Embedding's weight's dim should be 2.\n");
        AssertInFastLLM(weight.dataType == DataType::FLOAT32 ||
                        weight.dataType == DataType::BFLOAT16, "Embedding's weight's type should be float32 or bfloat16.\n");
        AssertInFastLLM(input.dataType == DataType::FLOAT32, "Embedding's input's type should be float32.\n");

        weight.weightType = WeightType::EMBEDDING;

        int vocabSize = weight.dims[0], embSize = weight.dims[1];
        std::vector <int> dims = input.dims;
        dims.push_back(embSize);
        output.dataType = DataType::FLOAT32;
        output.Resize(dims);
        output.Allocate();
        uint64_t inputLen = input.Count(0);
        float *inputData = (float*)input.cpuData;

        if (weight.dataType == DataType::FLOAT32) {
            float *outputData = (float *) output.cpuData;
            float *weightData = (float *) weight.cpuData;
            for (int i = 0; i < inputLen; i++) {
                int token = (int) (inputData[i] + 1e-9);
                memcpy(outputData + i * embSize, weightData + token * embSize, embSize * sizeof(float));
            }
        } else {
            uint16_t *outputData = (uint16_t *) output.cpuData;
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

    void LayerNorm(const Data &input, const Data &gamma, const Data &beta, int axis, Data &output) {
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        if (output.dims != input.dims || output.dataType != input.dataType || output.cpuData == nullptr) {
            output.dataType = input.dataType;
            output.Resize(input.dims);
            output.Allocate();
        }
        int outer = input.Count(0) / input.Count(axis);
        int channels = input.dims[axis];
        int inner = input.strides[axis];

        float *mean = new float[inner], *var = new float[inner];
        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;
        float *gammaData = (float*)gamma.cpuData;
        float *betaData = (float*)beta.cpuData;

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
                for (; j < channels; j++) {
                    mean += inputData[j];
                    s2 += inputData[j] * inputData[j];
                }
                mean /= channels;
                var = s2 + mean * mean * channels - 2 * mean * channels * mean;
                var = sqrt(var / channels + 1e-10);
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

    void FloatLinearPart(float *inputData, float *weightData, float *biasData, float *outputData,
                         int n, int m, int k, int st, int end) {
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
#endif
                for (; l < m; l++) {
                    now += inputData[i * m + l] * weightData[j * m + l];
                }
                outputData[i * k + j] = now;
            }
        }
    }

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

    // float的input, int4的weight, 直接计算得到float的output
    void Int4LinearPart(float *inputData, uint8_t *weightData, float *biasData, float *outputData,
                        LowBitConfig *configs, int n, int m, int k, int st, int end) {
        for (int i = 0; i < n; i++) {
            for (int j = st; j < end; j++) {
                float now = biasData ? biasData[j] : 0.0f;
                int l = 0;
#ifdef __aarch64__
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

    void Linear(const Data &input, Data &weight, const Data &bias, Data &output) {
        AssertInFastLLM(weight.dims.size() == 2, "Linear's weight's shape's size should be 2.\n");
        AssertInFastLLM(input.dims.back() == weight.dims[1], "Linear's weight's shape error.\n");
//auto st = std::chrono::system_clock::now();
        weight.weightType = WeightType::LINEAR;

        std::vector <int> dims = input.dims;
        dims.back() = weight.dims[0];
        output.Resize(dims);
        output.Allocate(0.0f);

        int n = input.Count(0) / input.dims.back();
        int m = input.dims.back();
        int k = output.dims.back();

        if (weight.dataType == DataType::FLOAT32) {
            float *inputData = (float *) input.cpuData;
            float *weightData = (float *) weight.cpuData;
            float *outputData = (float *) output.cpuData;
            float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;

            int threadNum = threads;
            int per = k / threadNum;
            int cur = 0;
            std::vector<std::thread *> threads;
            for (int i = 0; i < threadNum - 1; i++) {
                int end = cur + per + (cur + per * (threadNum - i) < k);
                threads.push_back(new std::thread(&FloatLinearPart, inputData, weightData, biasData, outputData,
                                                  n, m, k, cur, end));
                cur = end;
            }
            FloatLinearPart(inputData, weightData, biasData, outputData, n, m, k, cur, k);
            for (int i = 0; i < threadNum - 1; i++) {
                threads[i]->join();
                delete threads[i];
            }
        } else if (weight.dataType == DataType::INT8) {
            float *inputData = (float *) input.cpuData;
            uint8_t *weightData = (uint8_t *) weight.cpuData;
            float *outputData = (float *) output.cpuData;
            float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
            weight.CalcWeightSum();

#ifdef USE_CUDA
            FastllmMatMulFloatInt8(input, weight, bias, output, n, m, k);
            return;
#endif
            float minValue = 1e9, maxValue = -1e9;
            for (int i = 0; i < n * m; i++) {
                minValue = std::min(minValue, inputData[i]);
                maxValue = std::max(maxValue, inputData[i]);
            }
            std::vector <uint8_t> uinput;
            uinput.resize(n * m);
            LowBitConfig inputConfig = LowBitConfig(minValue, maxValue, 8);
            for (int i = 0; i < n * m; i++) {
                uinput[i] = inputConfig.quantization(inputData[i]);
            }
            MultiplyMultiThread(uinput.data(), weightData, (int32_t*)outputData, n, m, k, threads);
            for (int i = 0; i < n; i++) {
                uint32_t inputSum = 0;
                for (int j = 0; j < m; j++) {
                    inputSum += uinput[i * m + j];
                }

                for (int j = 0; j < k; j++) {
                    int value = ((int32_t*)outputData)[i * k + j];
                    value -= weight.weightSum[j] * inputConfig.zeroPoint;
                    value -= inputSum * weight.perChannelsConfigs[j].zeroPoint;
                    value += (int)inputConfig.zeroPoint * weight.perChannelsConfigs[j].zeroPoint * m;

                    outputData[i * k + j] = weight.perChannelsConfigs[j].scale * inputConfig.scale * value +
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
        } else if (weight.dataType == DataType::INT4) {
            float *inputData = (float *) input.cpuData;
            uint8_t *weightData = (uint8_t *) weight.cpuData;
            float *outputData = (float *) output.cpuData;
            float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
            weight.CalcWeightSum();

#ifdef USE_CUDA
	        FastllmMatMulFloatInt4(input, weight, bias, output, n, m, k);
	        return;
#endif

            float minValue = 1e9, maxValue = -1e9;
            for (int i = 0; i < n * m; i++) {
                minValue = std::min(minValue, inputData[i]);
                maxValue = std::max(maxValue, inputData[i]);
            }
            std::vector <uint8_t> uinput;
            uinput.resize(n * m);
            LowBitConfig inputConfig = LowBitConfig(minValue, maxValue, 8);
            for (int i = 0; i < n * m; i++) {
                uinput[i] = inputConfig.quantization(inputData[i]);
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
            MultiplyInt4MultiThread(uinput.data(), weightData, (int32_t*)outputData, n, m, k,
                                    weight.weightSum.data(), weight.zeros.data(), weight.scales.data(), biasData,
                                    inputConfig, threads);
            /*
            这部分是float输入，float输出
            int threadNum = threads;
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
        } else {
            ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
        }
//float spend = GetSpan(st, std::chrono::system_clock::now());
//float gops = (float)n * m * k / spend / 1e9;
//printf("n = %d, m = %d, k = %d, spend %f s, gops = %f\n", n, m, k, spend, gops);
    }

    void Split(const Data &input, int axis, int start, int end, Data &output) {
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        start = std::max(0, std::min(input.dims[axis] - 1, start));
        end = std::max(0, std::min(input.dims[axis], end));
        std::vector <int> dims = input.dims;
        dims[axis] = end - start;

        output.dataType = input.dataType;
        output.Resize(dims);
        output.Allocate();

        int outer = input.Count(0) / input.Count(axis);
        int inputStride = input.Count(axis);
        int outputStride = output.Count(axis);
        int channels = input.dims[axis];
        int inner = input.strides[axis];
        int unitSize = input.unitSize;

        for (int o = 0; o < outer; o++) {
            memcpy(output.cpuData + o * outputStride * unitSize,
                   input.cpuData + (o * inputStride + start * inner) * unitSize,
                   (end - start) * inner * unitSize);
        }
    }

    void Cat(const Data &input0, const Data &input1, int axis, Data &output) {
        if (input0.dims.size() == 0 && input1.dims.size() > 0) {
            output.CopyFrom(input1);
            return;
        }
        if (input1.dims.size() == 0 && input0.dims.size() > 0) {
            output.CopyFrom(input0);
            return;
        }

        AssertInFastLLM(input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32,
                        "Cat's input's type should be float32.\n");
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
        output.Allocate();

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

    void CatDirect(Data &input0, const Data &input1, int axis) {
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

        AssertInFastLLM(input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32,
                        "Cat's input's type should be float32.\n");
        AssertInFastLLM(input0.dims.size() == input1.dims.size(), "Cat Error: input's shape's size should be same.\n");
        int dimsLen = input0.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        for (int i = 0; i < dimsLen; i++) {
            if (i != axis) {
                AssertInFastLLM(input0.dims[i] == input1.dims[i], "Cat Error: input's shape doesn't match.");
            }
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

    void CatDirectAxis0(Data &input0, const Data &input1) {
        if (input0.dims.size() == 0) {
            input0.Resize(input1.dims);
            AssertInFastLLM(input0.Count(0) <= input0.expansionSize, "CatDirectAxis0 Error: input0's expansion size is not enough.\n");
            memcpy(input0.cpuData, input1.cpuData, input1.GetBytes());
            return;
        }

        AssertInFastLLM(input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32,
                        "Cat's input's type should be float32.\n");
        AssertInFastLLM(input0.dims.size() == input1.dims.size(), "Cat Error: input's shape's size should be same.\n");
        for (int i = 1; i < input0.dims.size(); i++) {
            AssertInFastLLM(input0.dims[i] == input1.dims[i],
                            "CatDirectAxis0 Error: shape error.\n");
        }

        uint64_t input0Bytes = input0.GetBytes();
        uint64_t input1Bytes = input1.GetBytes();
        std::vector <int> dims = input0.dims;
        dims[0] += input1.dims[0];
        input0.Resize(dims);
        AssertInFastLLM(input0.Count(0) <= input0.expansionSize, "CatDirectAxis0 Error: input0's expansion size is not enough.\n");
        memcpy(input0.cpuData + input0Bytes, input1.cpuData, input1Bytes);
    }

    void MatMulSingle(float *input0Base, float *input1Base, float *outputBase,
                      int input0Spatial, int input1Spatial, int outputSpatial,
                      int input0Stride, int input1Stride,
                      int n, int m, int k, float alpha, int st, int end) {
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

    void MatMulTransB(const Data &input0, const Data &input1, Data &output, float alpha) {
        AssertInFastLLM(input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32,
                        "MatMulTransB's input's type should be float32.\n");
        AssertInFastLLM(input0.dims.size() >= 2 && input1.dims.size() >= 2,
                        "MatMulTransB's input's shape's size should be >= 2.\n");
        AssertInFastLLM(input0.dims.back() == input1.dims.back(),
                        "MatMulTransB's shape error.\n");
        int input0Spatial = input0.Count(input0.dims.size() - 2);
        int input1Spatial = input1.Count(input1.dims.size() - 2);
        int input0Stride = input0.strides[input0.dims.size() - 2];
        int input1Stride = input1.strides[input1.dims.size() - 2];
        int n = input0.dims[input0.dims.size() - 2];
        int m = input0.dims.back();
        int k = input1.dims[input1.dims.size() - 2];
        int batch0 = input0.Count(0) / input0Spatial;
        int batch1 = input1.Count(0) / input1Spatial;
        AssertInFastLLM(batch0 == batch1, "MatMulTransB's shape error.\n");

        std::vector <int> dims = input0.dims;
        dims.back() = input1.dims[input1.dims.size() - 2];
        output.dataType = input0.dataType;
        output.Resize(dims);
        output.Allocate();

        int outputSpatial = output.Count(output.dims.size() - 2);

        int threadNum = threads;
#ifdef _WIN64
        threadNum = 1;
#endif
        if (batch0 * n * m * k < 64 * 4096) {
            threadNum = 1;
        }
        threadNum = std::min(threadNum, 4);

        int per = batch0 / threadNum;
        int cur = 0;
        std::vector<std::thread *> threads;
        for (int i = 0; i < threadNum - 1; i++) {
            int end = cur + per + (cur + per * (threadNum - i) < batch0);
            threads.push_back(new std::thread(&MatMulSingle,
                                              (float*)input0.cpuData, (float*)input1.cpuData, (float*)output.cpuData,
                                              input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                                              n, m, k, alpha, cur, end));
            cur = end;
        }
        MatMulSingle((float*)input0.cpuData, (float*)input1.cpuData, (float*)output.cpuData,
                     input0Spatial, input1Spatial, outputSpatial, input0Stride, input1Stride,
                     n, m, k, alpha, cur, batch0);
        for (int i = 0; i < threadNum - 1; i++) {
            threads[i]->join();
            delete threads[i];
        }
    }

    void Softmax(const Data &input, Data &output, int axis) {
        AssertInFastLLM(input.dataType == DataType::FLOAT32, "Softmax error: Data's type should be float32.\n");

        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;

        if (output.dims != input.dims || output.dataType != input.dataType || output.cpuData == nullptr) {
            output.dataType = input.dataType;
            output.Resize(input.dims);
            output.Allocate();
        }

        int outer = input.Count(0) / input.Count(axis);
        int channels = input.dims[axis];
        int inner = input.Count(axis + 1);

        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;

        if (inner == 1) {
            for (int i = 0; i < outer; i++) {
                float maxValue = 0;
                int j = 0;
#ifdef ARM
                float32x4_t vmax = vdupq_n_f32(-1e9);
                for (; j + 3 < channels; j += 4) {
                    vmax = vmaxq_f32(vmax, vld1q_f32(inputData + j));
                }
                for (int k = 0; k < 4; k++) {
                    maxValue = max(maxValue, vmax[k]);
                }
#endif
                for (; j < channels; j++) {
                    maxValue = std::max(maxValue, inputData[j]);
                }

                j = 0;
#ifdef ARM
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

                j = 0;
#ifdef ARM
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
            return;
        }

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

    void GeluNew(const fastllm::Data &input, fastllm::Data &output) {
        AssertInFastLLM(input.dataType == DataType::FLOAT32, "GeluNew error: Data's type should be float32.\n");

        if (output.dims != input.dims || output.dataType != input.dataType || output.cpuData == nullptr) {
            output.dataType = input.dataType;
            output.Resize(input.dims);
            output.Allocate();
        }

        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;
        int len = input.Count(0);
        int i = 0;
#ifdef USE_CUDA
        if (FastllmGelu(input, output)) {
            return;
        }
#endif

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
        for (; i < len; i++) {
            float x = inputData[i];
            outputData[i] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * x * (1.0f + 0.044715f * x * x)));
        }
    }

    void Mul(const fastllm::Data &input, float v, fastllm::Data &output) {
        AssertInFastLLM(input.dataType == DataType::FLOAT32, "Mul error: Data's type should be float32.\n");

        if (output.dims != input.dims || output.dataType != input.dataType || output.cpuData == nullptr) {
            output.dataType = input.dataType;
            output.Resize(input.dims);
            output.Allocate();
        }

        float *inputData = (float*)input.cpuData;
        float *outputData = (float*)output.cpuData;
        int len = input.Count(0);
        for (int i = 0; i < len; i++) {
            outputData[i] = inputData[i] * v;
        }
    }

    void AddTo(Data &input0, const Data &input1) {
        AssertInFastLLM(input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32,
                        "AddTo error: Data's type should be float32.\n");
        AssertInFastLLM(input0.dims == input1.dims, "AddTo error: input's shape should be same.\n");

        float *input0Data = (float*)input0.cpuData;
        float *input1Data = (float*)input1.cpuData;

        int len = input0.Count(0);
        for (int i = 0; i < len; i++) {
            input0Data[i] += input1Data[i];
        }
    }

    void AddTo(Data &input0, const Data &input1, float alpha) {
        AssertInFastLLM(input0.dataType == DataType::FLOAT32 && input1.dataType == DataType::FLOAT32,
                        "AddTo error: Data's type should be float32.\n");
        AssertInFastLLM(input0.dims == input1.dims, "AddTo error: input's shape should be same.\n");

        float *input0Data = (float*)input0.cpuData;
        float *input1Data = (float*)input1.cpuData;

        int len = input0.Count(0);
        for (int i = 0; i < len; i++) {
            input0Data[i] += input1Data[i] * alpha;
        }
    }
}