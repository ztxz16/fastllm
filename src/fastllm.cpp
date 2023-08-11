//
// Created by huangyuyang on 5/11/23.
//

#include "utils.h"

#include "fastllm.h"

#include "executor.h"

#include <cstring>
#include <cmath>
#include <cfloat>
#include <thread>

#ifdef USE_MMAP
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#endif

#ifdef __aarch64__
#include <arm_neon.h>
#include "armMath.h"
#endif

#ifdef __AVX__
#include "immintrin.h"
#endif

#ifdef USE_CUDA
#include "fastllm-cuda.cuh"
#endif

namespace fastllm {
    std::map <std::string, int> defaultDeviceMap;
    Executor defaultExecutor;
    Executor *curExecutor = &defaultExecutor;

    static std::mutex globalLocker;
    static int threads = 4;
    static ThreadPool *fastllmThreadPool = new ThreadPool(threads);
    static bool lowMemMode = false;
    static bool kvCacheInCPU = false;

    void PrintInstructionInfo() {
        std::string avx = "OFF", avx2 = "OFF", aarch64 = "OFF", neonFp16 = "OFF", neonDot = "OFF";
#ifdef __AVX__
        avx = "ON";
#endif
#ifdef __AVX2__
        avx2 = "ON";
#endif
#ifdef __aarch64__
        aarch64 = "ON";
#endif
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        neonFp16 = "ON";
#endif
#ifdef __ARM_FEATURE_DOTPROD
        neonDot = "ON";
#endif
        printf("AVX: %s\n", avx.c_str());
        printf("AVX2: %s\n", avx2.c_str());
        printf("AARCH64: %s\n", aarch64.c_str());
        printf("Neon FP16: %s\n", neonFp16.c_str());
        printf("Neon DOT: %s\n", neonDot.c_str());
    }

    void SetKVCacheInCPU(bool v) {
        kvCacheInCPU = v;
    }

    void SetThreads(int t) {
        globalLocker.lock();
        threads = t;
        if (fastllmThreadPool != nullptr) {
            fastllmThreadPool->Shutdown();
            delete fastllmThreadPool;
        }
        fastllmThreadPool = new ThreadPool(t);
        globalLocker.unlock();
    }

    void SetLowMemMode(bool m) {
    	lowMemMode = m;
    }

    bool GetKVCacheInCPU() {
        return kvCacheInCPU;
    }

    bool GetLowMemMode() {
        return lowMemMode;
    }

    int GetThreads() {
        return threads;
    }

    ThreadPool *GetPool() {
        return fastllmThreadPool;
    }
#ifdef USE_MMAP
    FileMmap::FileMmap(const std::string &path) {
        int fd = open(path.c_str(), O_RDONLY);
        AssertInFastLLM(fd > 0, "cannot open file ");

        struct stat sb;
        AssertInFastLLM(fstat(fd, &sb) == 0, "fstat error");
        size = sb.st_size;

        data = (char *)mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        AssertInFastLLM(data != MAP_FAILED, "mmap failed");

        AssertInFastLLM(close(fd) == 0, "close file error");
    }

    FileMmap::~FileMmap() { AssertInFastLLM(munmap(data, size) == 0, "munmap failed");}
#endif
    void ModelLoader::seek(int64_t offset, int whence) {
        if (whence == SEEK_SET) {
            ptr = data + offset;
        } else if (whence == SEEK_CUR) {
            ptr += offset;
        } else if (whence == SEEK_END) {
            ptr = data + size + offset;
        } else {
            printf("invalid seek mode: %d", whence);
        }
    }

    std::string ModelLoader::ReadString() {
        int length = ReadInt();
        std::string s(ptr, ptr + length);
        ptr += length;
        return s;
    }

    int ModelLoader::ReadInt(){
        return read_basic<int>();
    }

    float ModelLoader::ReadFloat(){
        return read_basic<float>();
    }

    uint8_t* ModelLoader::ReadBytes(uint64_t bytes){
        // memcpy(buffer, ptr, bytes);
        uint8_t* buffer = (uint8_t *) ptr;
        ptr += bytes;
        return buffer;
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
            this->unitSizeDiv = 1;
        } else if (this->dataType == DataType::BFLOAT16 ||
                this->dataType == DataType::INT16 ||
                this->dataType == DataType::FLOAT16) {
            this->unitSize = 2;
            this->unitSizeDiv = 1;
        } else if (this->dataType == DataType::INT8) {
            this->unitSize = 1;
            this->unitSizeDiv = 1;
        } else if (this->dataType == DataType::INT4 || this->dataType == DataType::INT4_NOZERO) {
            this->unitSize = 1;
            this->unitSizeDiv = 2;
        } else if (this->dataType == DataType::INT2) {
            this->unitSize = 1;
            this->unitSizeDiv = 4;
        } else if (this->dataType == DataType::BIT) {
            this->unitSize = 1;
            this->unitSizeDiv = 8;
        } else if (this->dataType == DataType::INT32PARAM) {
            this->unitSize = 4;
            this->unitSizeDiv = 1;
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

    void Data::MallocSpace(uint64_t size) {
        this->expansionSize = size;
        this->expansionBytes = (size * this->unitSize - 1) / this->unitSizeDiv + 1;
        if (this->dataDevice == DataDevice::CPU) {
            this->cpuData = new uint8_t[this->expansionBytes];
        } else if (this->dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
            this->cudaData = FastllmCudaMalloc(this->expansionBytes);
#else
            ErrorInFastLLM("Error: cuda is not supported.\n");
#endif
        }
    }

    void Data::FreeSpace() {
        this->expansionSize = 0;
        this->expansionBytes = 0;
        if (this->dataDevice == DataDevice::CPU) {
            delete[] this->cpuData;
        } else if (this->dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
            FastllmCudaFree(this->cudaData);
#else
            ErrorInFastLLM("Error: cuda is not supported.\n");
#endif
        }
    }

    void Data::Allocate() {
        if (Count(0) > expansionSize) {
            FreeSpace();
            MallocSpace(Count(0));
        }
    }

    void Data::Allocate(float v) {
        AssertInFastLLM(this->dataType == DataType::FLOAT32, "Allocate error: Data's type should be float32.\n");
        this->Allocate();
        float *f = (float*)cpuData;
        if (this->dataDevice == DataDevice::CPU) {
            std::fill(f, f + Count(0), v);
        } else {
            // TODO: 别的设备上的初始化
        }
    }

    void Data::Expansion(const std::vector<int> &dims) {
        if (this->dims.size() == 0) {
            this->strides.resize(dims.size(), 1);
            this->strides.back() = 1;
            for (int i = dims.size() - 2; i >= 0; i--) {
                this->strides[i] = dims[i + 1] * this->strides[i + 1];
            }
            this->expansionDims = dims;
            this->MallocSpace(this->strides[0] * dims[0]);
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
        this->expansionDims = dims;
        if (this->expansionBytes != 0) {
            if (this->dataDevice == DataDevice::CPU) {
                uint8_t *old = this->cpuData;
                MallocSpace(this->strides[0] * std::max(this->dims[0], dims[0]));
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
            } else if (this->dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
                uint8_t *old = (uint8_t*)this->cudaData;
                MallocSpace(this->strides[0] * std::max(this->dims[0], dims[0]));
                int outer = this->Count(0) / this->Count(axis);
                int input0Stride = this->Count(axis);
                int inner = this->strides[axis];
                int unitSize = this->unitSize;
                FastllmCudaMemcpy2DDeviceToDevice((uint8_t*)this->cudaData, input0Stride * unitSize,
                                            (uint8_t*)old, input1Stride * unitSize, this->dims[axis] * inner * unitSize, outer);
                FastllmCudaFree(old);
                FastllmCudaClearBigBuffer();
#else
                ErrorInFastLLM("Error: cuda is not supported.\n");
#endif
            }
        } else {
            MallocSpace(this->strides[0] * std::max(this->dims[0], dims[0]));
        }
    }

    Data::~Data() {
#ifndef USE_MMAP
        delete[] this->cpuData;
#endif
#ifdef USE_CUDA
        if (this->cudaData != nullptr) {
            FastllmCudaFree(this->cudaData);
        }
#endif
    }

    void Data::PrintShape() const {
        printf("shape: ");
        for (int i : this->dims) {
            printf("%d ", i);
        }
        printf("\n");
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
            if (m > 10) {
                printf("... ");
                for (int j = 0; j < 10 && j < m; j++) {
                    printf("%f ", ((float *) cpuData)[i * m + (m - 10 + j)]);
                }
            }
            printf("\n");
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
        } else if (this->dataType == DataType::INT4 || this->dataType == DataType::INT4_NOZERO) {
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

    void Data::ToDevice(void *device) {
        BaseDevice *dev = (BaseDevice*)device;
        if (dev->deviceType == "cuda") {
            this->ToDevice(DataDevice::CUDA, dev->deviceIds);
        } else {
            this->ToDevice(DataDevice::CPU, dev->deviceIds);
        }
    }

    void Data::ToDevice(fastllm::DataDevice device) {
        if (device == DataDevice::CUDA) {
            ToDevice(device, curExecutor->GetDeviceIds("cuda"));
        } else {
            ToDevice(device, {0});
        }
    }

    void Data::ToDevice(fastllm::DataDevice device, const std::vector <int> &deviceIds) {
        // TODO: 同一个Weight切分到不同 Device 上
        // NOTICE: 目前还不支持，暂时只切到deviceIds[0]上

        if (this->dataType == DataType::INT32PARAM) {
            return;
        }
#ifndef USE_CUDA
        // TODO: 这里先直接跳过了
        return;
#endif
        if (this->dataDevice == device &&
            (this->dataDevice == DataDevice::CPU || deviceIds.size() == 0 || this->dataDeviceIds == deviceIds)) {
            return;
        }

        if (this->expansionBytes != 0) {
#ifdef USE_CUDA
            if (this->dataDevice == DataDevice::CPU) {
                if (device == DataDevice::CUDA) {
                    FastllmCudaSetDevice(deviceIds.size() == 0 ? 0 : deviceIds[0]);
                    this->cudaData = FastllmCudaMalloc(expansionBytes);
                    FastllmCudaCopyFromHostToDevice(this->cudaData, this->cpuData, expansionBytes);
                    delete[] this->cpuData;
                    this->cpuData = nullptr;
                }
            } else if (this->dataDevice == DataDevice::CUDA) {
                if (device == DataDevice::CPU) {
                    this->cpuData = new uint8_t[expansionBytes];
                    FastllmCudaCopyFromDeviceToHost(this->cpuData, this->cudaData, expansionBytes);
                    FastllmCudaFree(this->cudaData);
                    this->cudaData = nullptr;
                } else if (device == DataDevice::CUDA) {
                    FastllmCudaSetDevice(this->dataDeviceIds.size() == 0 ? 0 : this->dataDeviceIds[0]);
                    uint8_t *cpuData = new uint8_t[expansionBytes];
                    FastllmCudaCopyFromDeviceToHost(cpuData, this->cudaData, expansionBytes);
                    FastllmCudaFree(this->cudaData);

                    FastllmCudaSetDevice(deviceIds.size() == 0 ? 0 : deviceIds[0]);
                    this->cudaData = FastllmCudaMalloc(expansionBytes);

                    FastllmCudaCopyFromHostToDevice(this->cudaData, cpuData, expansionBytes);
                    delete[] cpuData;
                }
            }
#endif
        }
        if (deviceIds.size() == 0) {
            this->dataDeviceIds = {0};
        } else {
            this->dataDeviceIds = deviceIds;
        };
        this->dataDevice = device;
    }

    std::string GetModelTypeFromFile(const std::string &fileName) {
        std::string ret = "unknown";
    #ifdef USE_MMAP
        std::unique_ptr<FileMmap> mapped_file = std::make_unique<FileMmap>(fileName);
        ModelLoader buffer((char *)mapped_file->data, mapped_file->size);
    #else
        FileBuffer buffer(fileName);
    #endif
        int versionId = buffer.ReadInt();
        std::map <std::string, std::string> dicts;
        if (versionId >= 1) {
            int keyValueLen = buffer.ReadInt();
            for (int i = 0; i < keyValueLen; i++) {
                std::string key = buffer.ReadString();
                std::string value = buffer.ReadString();
                dicts[key] = value;
            }
        }
        if (versionId <= 1) {
            // 老旧的模型，直接通过vocab_size判定模型类型
            int vocabLen = buffer.ReadInt();
            if (vocabLen == 106072) {
                ret = "moss";
            } else if (vocabLen == 64000) {
                ret = "baichuan";
            } else {
                ret = "chatglm";
            }
        } else {
            if (dicts.find("model_type") != dicts.end()) {
                ret = dicts["model_type"];
            }
        }
        return ret;
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

    void Tokenizer::Insert(const std::string &s, int tokenId, float score) {
        TrieNode *now = this->root;
        for (int i = 0; i < s.size(); i++) {
            if (now->next.find(s[i]) == now->next.end()) {
                now->next[s[i]] = new TrieNode();
            }
            now = now->next[s[i]];
        }
        now->tokenId = tokenId;
        now->score = score;
        tokenToStringDict[tokenId] = s;
        stringToTokenDict[s] = tokenId;
    }

    void Tokenizer::TryMergePairs(std::vector<Symbol> &symbols, int l, int r, std::priority_queue <SymbolPairs> &q) {
        if (l == -1 || r == -1 || symbols[l].len == 0 || symbols[r].len == 0) {
            return;
        }
        auto now = symbols[l].node;
        char *s = symbols[r].s;
        int pos = symbols[r].pos, len = symbols[r].len;
        for (int i = pos; i < pos + len; i++) {
            if (now->next.find(s[i]) != now->next.end()) {
                now = now->next[s[i]];
            } else {
                return;
            }
        }
        if (now->tokenId == -999999) {
            return;
        }
        q.push(SymbolPairs(now->score, l, r, symbols[l].len + symbols[r].len));
    }

    Data Tokenizer::Encode(const std::string &ori) {
        if (this->type == TokenizerType::BPE) {
            std::string blank = "";
            blank += 226, blank += 150, blank += 129;
            std::string s = blank;
            for (int i = 0; i < ori.size(); i++) {
                if (ori[i] == ' ') {
                    if (i != 0 && ori[i - 1] != ' ') {
                        s += blank;
                    }
                } else {
                    s += ori[i];
                }
            }

            std::vector<Symbol> symbols;
            for (int i = 0; i < s.size(); i++) {
                int tokenId = -999999, pos = i - 1;
                TrieNode *now = this->root;
                for (int j = i; j < s.size(); j++) {
                    if (now->next.find(s[j]) != now->next.end()) {
                        now = now->next[s[j]];
                        if (now->tokenId != -999999) {
                            tokenId = now->tokenId;
                            pos = j;
                            break;
                        }
                    } else {
                        break;
                    }
                }
                if (pos >= i) {
                    symbols.push_back(Symbol(now, (char *) s.data(), i, pos - i + 1, (int) symbols.size() - 1,
                                             (int) symbols.size() + 1));
                    i = pos;
                } else {
                    symbols.push_back(Symbol(nullptr, (char *) s.data(), i, 0, (int) symbols.size() - 1,
                                             (int) symbols.size() + 1));
                }
            }
            symbols.back().next = -1;

            std::priority_queue<SymbolPairs> workQueue;
            for (int i = 1; i < symbols.size(); i++) {
                TryMergePairs(symbols, i - 1, i, workQueue);
            }

            while (!workQueue.empty()) {
                auto top = workQueue.top();
                workQueue.pop();
                if (symbols[top.l].len == 0 || symbols[top.r].len == 0 ||
                    symbols[top.l].len + symbols[top.r].len != top.size) {
                    continue;
                }

                for (int i = symbols[top.r].pos; i < symbols[top.r].pos + symbols[top.r].len; i++) {
                    symbols[top.l].node = symbols[top.l].node->next[symbols[top.r].s[i]];
                }
                symbols[top.l].len += symbols[top.r].len;
                symbols[top.r].len = 0;
                symbols[top.l].next = symbols[top.r].next;
                if (symbols[top.r].next >= 0) {
                    symbols[symbols[top.r].next].prev = top.l;
                }

                TryMergePairs(symbols, symbols[top.l].prev, top.l, workQueue);
                TryMergePairs(symbols, top.l, symbols[top.l].next, workQueue);
            }

            std::vector<float> v;
            for (int i = 0; i < symbols.size(); i++) {
                if (symbols[i].len > 0) {
                    v.push_back(symbols[i].node->tokenId);
                } else if (symbols[i].node == nullptr) {
                    // 未识别的字符
                    uint8_t c = (uint8_t) (symbols[i].s[symbols[i].pos]);
                    std::string now = "<0x00>";
                    now[3] = (c / 16 > 9 ? ('A' + c / 16 - 10) : ('0' + c / 16));
                    now[4] = (c % 16 > 9 ? ('A' + c % 16 - 10) : ('0' + c % 16));
                    if (stringToTokenDict.find(now) != stringToTokenDict.end()) {
                        v.push_back(stringToTokenDict[now]);
                    }
                }
            }
            return Data (DataType::FLOAT32, {1, (int)v.size()}, v);
        } else {
            std::vector <float> v;
            for (int i = 0; i < ori.size(); i++) {
                int tokenId = -999999, pos = i - 1;
                TrieNode *now = this->root;
                for (int j = i; j < ori.size(); j++) {
                    if (now->next.find(ori[j]) != now->next.end()) {
                        now = now->next[ori[j]];
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
                }
            }

            return Data (DataType::FLOAT32, {1, (int)v.size()}, v);
        }
    }

    std::string Tokenizer::DecodeTokens(const std::vector<int> &tokens) {
        std::string ret = "";
        for (int i = 0; i < tokens.size(); i++) {
            std::string s = tokenToStringDict[tokens[i]];
            if (s.size() == 6 && s.substr(0, 3) == "<0x" && s.back() == '>') {
                int c = 0;
                for (int i = 3; i < 5; i++) {
                    c *= 16;
                    if (s[i] >= '0' && s[i] <= '9') {
                        c += (s[i] - '0');
                    } else {
                        c += (s[i] - 'A' + 10);
                    }
                }

                s = " ";
                s[0] = c;
            }
            if (s == "<n>") {
                ret += "\n";
            } else if (s == "<|tab|>") {
                ret += "\t";
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

    std::string Tokenizer::Decode(const Data &data) {
        std::vector <int> tokens;
        for (int i = 0; i < data.Count(0); i++) {
            tokens.push_back((int) ((float *) data.cpuData)[i]);
        }
        return DecodeTokens(tokens);
    }

    struct Random {
        Random () {
            srand(time(NULL));
        }

        float randP() {
            return (float)(rand() % 10001) * 0.0001;
        }
    };

    Random fastllmRandom;

    int LLMSampling(Data &logits, int outerOffset,
                    const GenerationConfig &config, const LastTokensUnit &tokens) {
        logits.ToDevice(DataDevice::CPU);
        int vocabSize = logits.dims.back();
        float *base = ((float*)logits.cpuData) + outerOffset * vocabSize;

        if (fabs(config.repeat_penalty - 1.0) > 1e-6) {
            for (int id : tokens.tokenSet) {
                base[id] = (base[id] < 0 ? base[id] * config.repeat_penalty : base[id] / config.repeat_penalty);
            }
        }
        float invTemp = 1.0f / config.temperature;
        std::vector <std::pair <float, int> > v;
        for (int i = 0; i < vocabSize; i++) {
            v.push_back(std::make_pair(-base[i] * invTemp, i));
        }
        int topk = std::min(vocabSize, config.top_k);
        std::partial_sort(v.begin(), v.begin() + topk, v.end());
        float psum = 0.0, maxValue = -v.begin()->first;
        std::vector <float> ps;
        for (int i = 0; i < topk; i++) {
            ps.push_back(expf(-v[i].first - maxValue));
            psum += ps.back();
        }
        float curSum = 0.0;
        for (int i = 0; i < topk; i++) {
            ps[i] /= psum;
            curSum += ps[i];
            if (curSum > config.top_p) {
                topk = i + 1;
                break;
            }
        }
        float rnd = fastllmRandom.randP();
        curSum = 0.0;
        for (int i = 0; i < topk; i++) {
            curSum += ps[i];
            if (curSum > rnd || i == topk - 1) {
                return v[i].second;
            }
        }
        return -1;
    }

    void WeightMap::LoadFromFile(const std::string &fileName) {
    #ifdef USE_MMAP
        std::shared_ptr<FileMmap> mapped_file = std::make_shared<FileMmap>(fileName);
        ModelLoader buffer((char *)mapped_file->data, mapped_file->size);
    #else
        FileBuffer buffer(fileName);
    #endif
        this->versionId = buffer.ReadInt();

        if (this->versionId >= 1) {
            // versionId >= 1, 前置了一个key-value表
            int keyValueLen = buffer.ReadInt();
            for (int i = 0; i < keyValueLen; i++) {
                std::string key = buffer.ReadString();
                std::string value = buffer.ReadString();
                //printf("%s %s\n", key.c_str(), value.c_str());
                this->dicts[key] = value;
            }
        }

        bool useScore = this->dicts["tokenizer_use_score"] == "1";
        int vocabLen = buffer.ReadInt();
        for (int i = 0; i < vocabLen; i++) {
            int len = buffer.ReadInt();
            std::string x = "";
            for (int j = 0; j < len; j++) {
                x += buffer.ReadInt();
            }
            int id = buffer.ReadInt();
            float score = useScore ? buffer.ReadFloat() : -i;
            tokenizer.Insert(x, id, score);
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

            if (lowMemMode && this->embeddingNames.find(name) != this->embeddingNames.end()) {
	            if (dataType == DataType::FLOAT32 || dataType == DataType::BFLOAT16 || dataType == DataType::FLOAT16) {
	            	weight[name].fileName = fileName;
#if defined(_WIN32) or defined(_WIN64)
	            	weight[name].filePos = _ftelli64(buffer.f);
#else
#ifdef USE_MMAP
                    weight[name].filePos =  buffer.tell();
#else
                    weight[name].filePos = ftell(buffer.f);
#endif
#endif
#ifdef USE_MMAP
                    buffer.seek(weight[name].GetBytes(), SEEK_CUR);
#else
	            	fseek(buffer.f, weight[name].GetBytes(), SEEK_CUR);
#endif
	            } else {
	            	ErrorInFastLLM("Error: embedding's type should be float32 or bfloat16.\n");
	            }
            } else {
#ifdef USE_MMAP
                weight[name].set_file(mapped_file);
#else
	            weight[name].Allocate();
#endif
	            if (dataType == DataType::FLOAT32 || dataType == DataType::BFLOAT16 || dataType == DataType::FLOAT16) {
#ifdef USE_MMAP
                    weight[name].cpuData = buffer.ReadBytes(weight[name].GetBytes());
#else
                    buffer.ReadBytes(weight[name].cpuData, weight[name].GetBytes());
#endif
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
			            weight[name].perChannelsConfigs[i] = LowBitConfig(minValue, maxValue, bit, 0);
			            weight[name].zeros[i] = weight[name].perChannelsConfigs[i].zeroPoint;
			            weight[name].scales[i] = weight[name].perChannelsConfigs[i].scale;
		            }
#ifdef USE_MMAP
                    weight[name].cpuData = buffer.ReadBytes(weight[name].GetBytes());
#else
                    buffer.ReadBytes(weight[name].cpuData, weight[name].GetBytes());
#endif
	            } else if (dataType == DataType::INT4_NOZERO) {
                    int bit = 4;
                    weight[name].perChannelAxis = buffer.ReadInt();
                    int k = weight[name].perChannelAxis == -1 ? 1 : dims[weight[name].perChannelAxis];
                    weight[name].perChannelsConfigs.resize(k);
                    weight[name].mins.resize(k);
                    weight[name].scales.resize(k);
                    for (int i = 0; i < k; i++) {
                        float minValue = buffer.ReadFloat();
                        float maxValue = buffer.ReadFloat();
                        weight[name].perChannelsConfigs[i] = LowBitConfig(minValue, maxValue, bit, 1);
                        weight[name].mins[i] = weight[name].perChannelsConfigs[i].min;
                        weight[name].scales[i] = weight[name].perChannelsConfigs[i].scale;
                    }
#ifdef USE_MMAP
                    weight[name].cpuData = buffer.ReadBytes(weight[name].GetBytes());
#else
                    buffer.ReadBytes(weight[name].cpuData, weight[name].GetBytes());
#endif
                }
            }

            printf("Load (%d / %d) \r", (i + 1), len);
            fflush(stdout);
        }
        printf("\n");
        fflush(stdout);
        return;
    }

    void PerChannelQuantizationMultiThread(int st, int end, int m,
                                           float *f, uint8_t *u8, LowBitConfig *configs, int bit) {
        int type = (bit == 4) ? 1 : 0;
        for (int i = st; i < end; i++) {
            float minValue = 1e9, maxValue = -1e9;
            for (int j = 0; j < m; j++) {
                minValue = std::min(minValue, f[i * m + j]);
                maxValue = std::max(maxValue, f[i * m + j]);
            }
            if (bit == 8) {
                configs[i] = LowBitConfig(minValue, maxValue, 8, type);
                for (int j = 0; j < m; j++) {
                    u8[i * m + j] = configs[i].quantization(f[i * m + j]);
                }
            } else {
                configs[i] = LowBitConfig(minValue, maxValue, 4, type);
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
    }

    void WeightMap::SaveLowBitModel(const std::string &fileName, int bit) {
        AssertInFastLLM(fileName != "", "Error: output's name shouldn't be empty.\n");
        AssertInFastLLM(bit == 0 || bit == 4 || bit == 8 || bit == 16, "Error: only support 16 bit or 8 bit or 4 bit model.\n");
        FileWriter buffer(fileName);
        buffer.WriteInt(this->versionId);
        if (this->versionId >= 1) {
            // versionId >= 1, 前置了一个key-value表
            buffer.WriteInt((int)dicts.size());
            for (auto &it : dicts) {
                buffer.WriteString(it.first);
                buffer.WriteString(it.second);
            }
        }

        // 写入词表
        bool useScore = this->dicts["tokenizer_use_score"] == "1";
        buffer.WriteInt((int)tokenizer.tokenToStringDict.size());
        for (auto &it : tokenizer.tokenToStringDict) {
            buffer.WriteInt((int)it.second.size());
            for (int i = 0; i < it.second.size(); i++) {
                buffer.WriteInt((int)it.second[i]);
            }
            buffer.WriteInt(it.first);
            if (useScore) {
                buffer.WriteFloat(tokenizer.tokenToScoreDict[it.first]);
            }
        }

        // 写入权重
        int need = 0;
        for (auto &it : weight) {
            need += (it.second.dims.size() > 0);
        }
        buffer.WriteInt(need);
        int tot = 0;
        for (auto &it : weight) {
            if (it.second.dims.size() == 0) {
                continue;
            }
            buffer.WriteString(it.first);
            Data &data = it.second;
            buffer.WriteInt((int)data.dims.size());
            for (int i : data.dims) {
                buffer.WriteInt(i);
            }
            data.ToDevice(DataDevice::CPU);

            if (bit == 0) {
                DataType dataType = data.dataType;
                if (dataType == DataType::FLOAT32 || dataType == DataType::BFLOAT16 || dataType == DataType::FLOAT16) {
                    buffer.WriteInt((int) dataType);
                    buffer.WriteBytes(data.cpuData, data.GetBytes());
                } else if (dataType == DataType::INT8 || dataType == DataType::INT4 || dataType == DataType::INT4_NOZERO) {
                    buffer.WriteInt((int) dataType);
                    buffer.WriteInt(data.perChannelAxis);
                    int k = data.perChannelAxis == -1 ? 1 : data.dims[data.perChannelAxis];
                    for (int i = 0; i < k; i++) {
                        buffer.WriteFloat(data.perChannelsConfigs[i].min);
                        buffer.WriteFloat(data.perChannelsConfigs[i].max);
                    }
                    buffer.WriteBytes(data.cpuData, data.GetBytes());
                } else {
                    ErrorInFastLLM("unknown datatype");
                }
            } else {
                if (data.weightType == WeightType::NONE) {
                    // 普通权重，直接写入浮点数据
                    buffer.WriteInt((int) DataType::FLOAT32);
                    buffer.WriteBytes(data.cpuData, data.GetBytes());
                } else if (data.weightType == WeightType::EMBEDDING) {
                    // Embedding权重，存储成BF16
                    buffer.WriteInt((int) DataType::BFLOAT16);
                    int len = data.Count(0);
                    std::vector<uint16_t> uDatas;
                    uDatas.resize(len);
                    for (int i = 0; i < len; i++) {
                        uDatas[i] = ((uint16_t *) data.cpuData)[i * 2 + 1];
                    }
                    buffer.WriteBytes((uint8_t *) uDatas.data(), len * sizeof(uint16_t));
                } else if (data.weightType == WeightType::LINEAR) {
                    if (bit == 16) {
                        // fp16, 直接转换
                        buffer.WriteInt((int) DataType::FLOAT16);
                        int len = data.Count(0);
                        std::vector<uint16_t> uDatas;
                        uDatas.resize(len);
                        for (int i = 0; i < len; i++) {
                            uDatas[i] = float_to_half(((float *) data.cpuData)[i]);
                        }
                        buffer.WriteBytes((uint8_t *) uDatas.data(), len * sizeof(uint16_t));
                    } else {
                        // Linear层权重，分通道量化之
                        int k = data.dims[0], m = data.dims[1];
                        int threadNum = 8;
                        int per = k / threadNum;
                        int cur = 0;
                        auto pool = GetPool();
                        std::vector <std::future <void> > futures;
                        std::vector<LowBitConfig> configs;
                        std::vector<uint8_t> uDatas;
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
                            futures.push_back(pool->Submit(PerChannelQuantizationMultiThread, cur, end, m,
                                                              (float *) data.cpuData, uDatas.data(), configs.data(),
                                                              bit));
                            cur = end;
                        }
                        for (int i = 0; i < threadNum; i++) {
                            futures[i].get();
                        }

                        buffer.WriteInt(bit == 8 ? (int) DataType::INT8 : (int) DataType::INT4_NOZERO);
                        buffer.WriteInt(0); // 按通道0分通道量化
                        for (int i = 0; i < k; i++) {
                            buffer.WriteFloat(configs[i].min);
                            buffer.WriteFloat(configs[i].max);
                        }
                        buffer.WriteBytes(uDatas.data(), bytes);
                    }
                }
            }
            printf("output (%d / %d)\r", ++tot, need);
            fflush(stdout);
        }
        printf("\n");
        return;
    }

    void WeightMap::AddTokenizerWord(const std::string &key, int value, float score) {
        this->tokenizer.Insert(key, value, score);
    }

    void WeightMap::AddDict(const std::string &key, const std::string &value) {
        this->dicts[key] = value;
    }

    void WeightMap::AddQLinearWeight(const std::string &key, const std::vector <int> &dims,
                          int bit, float *scales, uint8_t *oriData) {
        AssertInFastLLM(bit == 4 || bit == 8, "Error: only support 8 bit or 4 bit QLinear.\n");
        DataType dataType = (bit == 4 ? DataType::INT4_NOZERO : DataType::INT8);
        std::vector <int> realDims = dims;
        if (bit == 4) {
            realDims[1] *= 2;
        }
        this->weight[key] = Data(dataType, realDims);
        Data &data = this->weight[key];
        data.weightType = WeightType::LINEAR;
        data.UpdateUnitSize();
        data.Allocate();

        int k = data.dims[0], m = data.dims[1];
        int bytes = k * m / (bit == 4 ? 2 : 1);
        data.perChannelAxis = 0;
        data.perChannelsConfigs.resize(k);
        data.zeros.resize(k);
        data.scales.resize(k);
        data.mins.resize(k);

        if (bit == 4) {
            for (int i = 0; i < k; i++) {
                data.perChannelsConfigs[i] = LowBitConfig(-8.0 * scales[i], 7 * scales[i], bit, 1);
                data.mins[i] = data.perChannelsConfigs[i].min;
                data.zeros[i] = data.perChannelsConfigs[i].zeroPoint;
                data.scales[i] = data.perChannelsConfigs[i].scale;
            }
            int mask = (8 << 4) | 8;
            for (int i = 0; i < 20; i++) {
                uint8_t a = oriData[i] >> 4, b = oriData[i] & 15;
                int8_t ia = *(int8_t*)(&a), ib = *(int8_t*)(&b);
            }
            for (int i = 0; i < bytes; i++) {
                oriData[i] = oriData[i] ^ mask;
            }
            memcpy((uint8_t*)data.cpuData, oriData, bytes);
        } else {
            for (int i = 0; i < k; i++) {
                data.perChannelsConfigs[i] = LowBitConfig(-128.0 * scales[i], 127 * scales[i], bit, 0);
                data.mins[i] = data.perChannelsConfigs[i].min;
                data.zeros[i] = data.perChannelsConfigs[i].zeroPoint;
                data.scales[i] = data.perChannelsConfigs[i].scale;
            }
            for (int i = 0; i < bytes; i++) {
                oriData[i] = oriData[i] ^ 128;
            }
            memcpy((uint8_t*)data.cpuData, oriData, bytes);
        }
    }

    void WeightMap::AddWeight(const std::string &key, const std::vector<int> &dims, fastllm::DataType dataType,
                              fastllm::WeightType weightType, fastllm::DataType oriDataType, uint8_t *oriData) {
        this->weight[key] = Data(dataType, dims);
        Data &data = this->weight[key];
        data.weightType = weightType;
        data.UpdateUnitSize();
        data.Allocate();
        if (dataType == oriDataType) {
            memcpy(data.cpuData, oriData, data.GetBytes());
        } else if (oriDataType == DataType::FLOAT32 &&
                (dataType == DataType::INT8 || dataType == DataType::INT4_NOZERO)) {
            int bit = (dataType == DataType::INT4_NOZERO) ? 4 : 8;
            int type = (bit == 4) ? 1 : 0;
            int k = data.dims[0], m = data.dims[1];
            int threadNum = 8;
            int per = k / threadNum;
            int cur = 0;
            auto pool = GetPool();
            std::vector <std::future <void> > futures;
            std::vector<LowBitConfig> configs;
            std::vector<uint8_t> uDatas;
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
                futures.push_back(pool->Submit(PerChannelQuantizationMultiThread, cur, end, m,
                                                  (float *) oriData, uDatas.data(), configs.data(), bit));
                cur = end;
            }
            for (int i = 0; i < threadNum; i++) {
                futures[i].get();
            }

            data.perChannelAxis = 0;
            data.perChannelsConfigs.resize(k);
            data.zeros.resize(k);
            data.scales.resize(k);
            data.mins.resize(k);
            for (int i = 0; i < k; i++) {
                data.perChannelsConfigs[i] = LowBitConfig(configs[i].min, configs[i].max, bit, type);
                data.mins[i] = data.perChannelsConfigs[i].min;
                data.zeros[i] = data.perChannelsConfigs[i].zeroPoint;
                data.scales[i] = data.perChannelsConfigs[i].scale;
            }
            memcpy((uint8_t*)data.cpuData, (uint8_t*)uDatas.data(), bytes);
        } else {
            ErrorInFastLLM("wrong data type");
        }
    }

    Data &WeightMap::operator[](const std::string &key) {
        return weight[key];
    }

    void Embedding(const Data &input, Data &weight, Data &output) {
        curExecutor->Run("Embedding", {
                {"input", (Data*)&input}, {"weight", &weight}, {"output", &output}
        }, {}, {});
    }

    void RMSNorm(const Data &input, const Data &weight, float eps, Data &output) {
        curExecutor->Run("RMSNorm", {
                {"input", (Data*)&input}, {"weight", (Data*)&weight}, {"output", &output}
        }, {{"eps", eps}}, {});
    }

    void LayerNorm(Data &input, Data &gamma, Data &beta, int axis, Data &output) {
        curExecutor->Run("LayerNorm", {
            {"input", &input}, {"gamma", &gamma}, {"beta", &beta}, {"output", &output}
        }, {}, {{"axis", axis}});
    }

    void Linear(Data &input, Data &weight, const Data &bias, Data &output) {
        curExecutor->Run("Linear", {
                {"input", &input}, {"weight", &weight}, {"bias", (Data*)&bias}, {"output", &output}
        }, {}, {});
    }

    void Split(const Data &input, int axis, int start, int end, Data &output) {
        curExecutor->Run("Split", {
                {"input", (Data*)&input}, {"output", &output}
        }, {}, {{"axis", axis}, {"start", start}, {"end", end}});
    }

    void Cat(const Data &input0, const Data &input1, int axis, Data &output) {
        curExecutor->Run("Cat", {
                {"input0", (Data*)&input0}, {"input1", (Data*)&input1}, {"output", &output}
        }, {}, {{"axis", axis}});
    }

    void CatDirect(Data &input0, const Data &input1, int axis) {
        curExecutor->Run("CatDirect", {
                {"input0", (Data*)&input0}, {"input1", (Data*)&input1}
        }, {}, {{"axis", axis}});
    }

    void MatMul(const Data &input0, const Data &input1, Data &output, float alpha) {
        curExecutor->Run("MatMul", {
                {"input0", (Data*)&input0}, {"input1", (Data*)&input1}, {"output", &output}
        }, {{"alpha", alpha}}, {});
    }

    void MatMulTransB(const Data &input0, const Data &input1, Data &output, float alpha) {
        curExecutor->Run("MatMulTransB", {
                {"input0", (Data*)&input0}, {"input1", (Data*)&input1}, {"output", &output}
        }, {{"alpha", alpha}}, {});
    }

    void Softmax(const Data &input, Data &output, int axis) {
        curExecutor->Run("SoftMax", {
                {"input", (Data*)&input}, {"output", &output}
        }, {}, {{"axis", axis}});
    }

    void Silu(const fastllm::Data &input, fastllm::Data &output) {
        curExecutor->Run("Silu", {
                {"input", (Data*)&input}, {"output", &output}
        }, {}, {});
    }

    void GeluNew(const fastllm::Data &input, fastllm::Data &output) {
        curExecutor->Run("GeluNew", {
                {"input", (Data*)&input}, {"output", &output}
        }, {}, {});
    }

    void Swiglu(const fastllm::Data &input, fastllm::Data &output) {
        curExecutor->Run("Swiglu", {
                {"input", (Data*)&input}, {"output", &output}
        }, {}, {});
    }

    void Mul(const fastllm::Data &input, float v, fastllm::Data &output) {
        curExecutor->Run("Mul", {
                {"input", (Data*)&input}, {"output", &output}
        }, {{"v", v}}, {});
    }

    void MulTo(Data &input0, const Data &input1) {
        curExecutor->Run("MulTo", {
                {"input0", &input0}, {"input1", (Data*)&input1}
        }, {}, {});
    }

    void AddTo(Data &input0, const Data &input1, float alpha) {
        curExecutor->Run("AddTo", {
                {"input0", &input0}, {"input1", (Data*)&input1}
        }, {{"alpha", alpha}}, {});
    }

    void AttentionMask(Data &input, const Data &mask, float maskValue) {
        curExecutor->Run("AttentionMask", {
                {"input", &input}, {"mask", (Data*)&mask}
        }, {{"maskValue", maskValue}}, {});
    }

    void AlibiMask(Data &input, const Data &mask, float maskValue) {
        curExecutor->Run("AlibiMask", {
                {"input", &input}, {"mask", (Data*)&mask}
        }, {{"maskValue", maskValue}}, {});
    }

    void Permute(const Data &input, const std::vector<int> &axis, Data &output) {
        Data axisData = Data(DataType::INT32PARAM, {(int)axis.size()});
        axisData.Allocate();
        for (int i = 0; i < axisData.Count(0); i++) {
            ((int32_t*)axisData.cpuData)[i] = axis[i];
        }
        curExecutor->Run("Permute", {
                {"input", (Data*)&input}, {"axis", &axisData}, {"output", (Data*)&output}
        }, {}, {});
    }

    void PermuteSelf(const Data &input, const std::vector<int> &axis) {
        Data axisData = Data(DataType::INT32PARAM, {(int)axis.size()});
        axisData.Allocate();
        for (int i = 0; i < axisData.Count(0); i++) {
            ((int32_t*)axisData.cpuData)[i] = axis[i];
        }
        curExecutor->Run("PermuteSelf", {
                {"input", (Data*)&input}, {"axis", &axisData}
        }, {}, {});
    }

    void TopK(const Data &input, Data &output, int topk) {
        curExecutor->Run("TopK", {
                {"input", (Data*)&input}, {"output", &output}
        }, {}, {{"topk", topk}});
    };

    void RotatePosition2D(Data &input, const Data &positionIds, Data &sinData, Data &cosData, int rotaryDim) {
        curExecutor->Run("RotatePosition2D", {
                {"input", &input}, {"positionIds", (Data*)&positionIds}, {"sin", &sinData}, {"cos", &cosData}
        }, {}, {{"rotaryDim", rotaryDim}});
    }

    void NearlyRotatePosition2D(Data &input, const Data &positionIds, Data &sinData, Data &cosData, int rotaryDim) {
        curExecutor->Run("NearlyRotatePosition2D", {
                {"input", &input}, {"positionIds", (Data*)&positionIds}, {"sin", &sinData}, {"cos", &cosData}
        }, {}, {{"rotaryDim", rotaryDim}});
    }

    void LlamaRotatePosition2D(Data &input, const Data &positionIds, Data &sinData, Data &cosData, int rotaryDim) {
        curExecutor->Run("LlamaRotatePosition2D", {
                {"input", &input}, {"positionIds", (Data*)&positionIds}, {"sin", &sinData}, {"cos", &cosData}
        }, {}, {{"rotaryDim", rotaryDim}});
    }

    void RepeatPenalty(Data &input, const Data &penalty) {
        curExecutor->Run("RepeatPenalty", {
                {"input", &input}, {"penalty", (Data*)&penalty}
        }, {}, {});
    }

    void SplitBatch(const Data &input, int axis, int part, std::vector <Data*> &outputs) {
        curExecutor->Run("SplitBatch", {
                {"input", (Data*)&input}, {"output", (Data*)outputs.data()}
        }, {}, {{"axis", axis}, {"output___batch", part}});
    }

    void CatBatch(std::vector <Data*> &input, int axis, Data &outputs) {
        curExecutor->Run("CatBatch", {
                {"input", (Data*)input.data()}, {"output", (Data*)&outputs}
        }, {}, {{"axis", axis}, {"input___batch", (int)input.size()}});
    }

    void MulBatch(std::vector <Data*> &input, float v, std::vector <Data*> &output) {
        curExecutor->Run("MulBatch", {
                {"input", (Data*)input.data()}, {"output", (Data*)output.data()}
        }, {{"v", v}}, {{"input___batch", (int)input.size()}, {"output___batch", (int)output.size()}});
    }

    void MatMulBatch(std::vector <Data*> &input0, std::vector <Data*> &input1, std::vector <Data*> &output, float alpha) {
        curExecutor->Run("MatMulBatch", {
                        {"input0", (Data*)input0.data()}, {"input1", (Data*)input1.data()}, {"output", (Data*)output.data()}
                         }, {{"alpha", alpha}},
                         {{"input0___batch", (int)input0.size()},
                          {"input1___batch", (int)input1.size()},
                          {"output___batch", (int)output.size()}});
    }

    void MatMulTransBBatch(std::vector <Data*> &input0, std::vector <Data*> &input1, std::vector <Data*> &output, float alpha) {
        curExecutor->Run("MatMulTransBBatch", {
                {"input0", (Data*)input0.data()}, {"input1", (Data*)input1.data()}, {"output", (Data*)output.data()}
        }, {{"alpha", alpha}},
        {{"input0___batch", (int)input0.size()},
         {"input1___batch", (int)input1.size()},
         {"output___batch", (int)output.size()}});
    }

    void SoftmaxBatch(std::vector <Data*> &input, std::vector <Data*> &output, int axis) {
        curExecutor->Run("SoftMaxBatch", {
                {"input", (Data*)input.data()}, {"output", (Data*)output.data()}
        }, {}, {{"axis", axis}, {"input___batch", (int)input.size()}, {"output___batch", (int)output.size()}});
    }

    void CatDirectBatch(std::vector <Data*> &input0, std::vector <Data*> &input1, int axis) {
        curExecutor->Run("CatDirectBatch", {
                {"input0", (Data*)input0.data()}, {"input1", (Data*)input1.data()}
        }, {}, {{"axis", axis}, {"input0___batch", (int)input0.size()}, {"input1___batch", (int)input1.size()}});
    }

    void ClearProfiler() {
        curExecutor->ClearProfiler();
    }

    void PrintProfiler() {
        curExecutor->PrintProfiler();
    }

    void ApplyDeviceMap(const std::map <std::string, int> &deviceMap, int current, int total) {
        if (deviceMap.size() == 0) {
            return;
        }
        int sum = 0, cur = 0;
        for (auto &it : deviceMap) {
            sum += it.second;
        }
        std::string curDevice = deviceMap.begin()->first;
        for (auto &it : deviceMap) {
            cur += it.second;
            // current / total <= cur / sum
            if (current * sum <= cur * total) {
                curDevice = it.first;
                break;
            }
        }
        curExecutor->SetFirstDevice(curDevice);
    }

    void SetDeviceMap(const std::map <std::string, int> &deviceMap) {
        defaultDeviceMap = deviceMap;
    }

    std::map <std::string, int> GetDeviceMap() {
        return defaultDeviceMap;
    }
}