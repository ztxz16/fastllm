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
#include <algorithm>

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

#ifdef PY_API
#include <pybind11/embed.h>
namespace py = pybind11;
#endif

#include <mutex>

#include "gguf.h"

namespace fastllm {
    std::map <std::string, int> defaultDeviceMap, defaultMoeDeviceMap;
    Executor defaultExecutor;
    Executor *curExecutor = &defaultExecutor;

    static std::mutex globalLocker;
    static int threads = 4;
    static AliveThreadPool *fastllmAliveThreadPool = nullptr;
    static bool lowMemMode = false;
    static bool kvCacheInCPU = false;
    static bool historyCacheInCPU = false;
    static bool cudaEmbedding = false;
    static bool cudaSharedExpert = false;
    static bool enableAMX = false;

    static std::map <DataType, int> DataTypeBits = {
        {DataType::FLOAT32, 32}, {DataType::BFLOAT16, 16}, {DataType::INT16, 16}, 
        {DataType::INT8, 8}, {DataType::INT4, 4}, {DataType::INT2, 2}, {DataType::BIT, 1}, 
        {DataType::FLOAT16, 16}, {DataType::INT4_NOZERO, 4}, {DataType::INT4_GROUP, 4},
        {DataType::FP8_E4M3, 8}, {DataType::INT2_GROUP, 2}, {DataType::BASE3_GROUP, 2}
    };

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

    void SetCudaEmbedding(bool v) {
        cudaEmbedding = v;
    }

    bool GetCudaEmbedding() {
        return cudaEmbedding;
    }

    void SetCudaSharedExpert(bool v) {
        cudaSharedExpert = v;
    }

    bool GetCudaSharedExpert() {
        return cudaSharedExpert;
    }

    void SetKVCacheInCPU(bool v) {
        kvCacheInCPU = v;
    }

    void SetHistoryCacheInCPU(bool v) {
        historyCacheInCPU = v;
    }

    void SetAliveThreads(int t) {
#ifdef PY_API
        py::gil_scoped_release release;
#endif
        globalLocker.lock();
        if (fastllmAliveThreadPool != nullptr) {
            fastllmAliveThreadPool->ResizeThreads(t);
        } else {
            fastllmAliveThreadPool = new AliveThreadPool(t);
        }
        threads = fastllmAliveThreadPool->threads.size();
        globalLocker.unlock();
#ifdef PY_API
        py::gil_scoped_acquire acquire;
#endif
    }
    void SetThreads(int t) {
        SetAliveThreads(t);
    }

    void SetLowMemMode(bool m) {
    	lowMemMode = m;
    }

    bool GetKVCacheInCPU() {
        return kvCacheInCPU;
    }

    bool GetHistoryCacheInCPU() {
        return historyCacheInCPU;
    }

    bool GetLowMemMode() {
        return lowMemMode;
    }

    int GetThreads() {
        return threads;
    }

    extern void InitAMX();
    void EnableAMX(bool enable) {
        enableAMX = enable;
        if (enable) {
            InitAMX();
        }
    }

    bool GetEnableAMX() {
        return enableAMX;
    }

    AliveThreadPool *GetAlivePool() {
        if (fastllmAliveThreadPool == nullptr) {
            SetAliveThreads(threads);
        }
        return fastllmAliveThreadPool;
    }

    std::map <DataType, std::vector <std::string> > dataTypeNames = {
        {DataType::FLOAT32, {"float32", "fp32"}}, {DataType::BFLOAT16, {"bfloat16", "bf16"}}, {DataType::INT16, {"int16"}}, 
        {DataType::INT8, {"int8"}}, {DataType::INT4, {"int4o"}}, {DataType::INT2, {"int2"}}, {DataType::BIT, {"bit"}}, 
        {DataType::FLOAT16, {"float16", "fp16", "half"}}, {DataType::INT4_NOZERO, {"int4"}}, {DataType::INT4_GROUP, {"int4g"}},
        {DataType::FP8_E4M3, {"float8", "fp8", "fp8_e4m3"}}, {DataType::INT2_GROUP, {"int2g"}}, {DataType::BASE3_GROUP, {"base3g"}}
    };

    std::string GetDataTypeName(DataType type) {
        if (dataTypeNames.find(type) != dataTypeNames.end()) {
            return dataTypeNames[type][0];
        } else if (type >= DataType::DATA_GGUF_FORMAT && type < DataType::DATA_GGUF_FORMAT_END) {
            return "GGML Type " + std::string(ggml_type_name((ggml_type)((int)type - (int)DataType::DATA_GGUF_FORMAT)));
        } else {
            return "Type " + std::to_string((int)type);
        }
    }

    size_t GetDataBytes(DataType type, size_t rows, size_t columns) {
        if (type == DataType::FLOAT32) {
            return rows * columns * sizeof(float);
        } else if (type == DataType::BFLOAT16 || type == DataType::FLOAT16) {
            return rows * columns * sizeof(uint16_t);
        } else if (type == DataType::FP8_E4M3_BLOCK_128) {
            // columns * [fp8] + ((columns - 1) / 128 + 1) * [float]
            return rows * (columns + ((columns - 1) / 128 + 1) * sizeof(float));
        } else if (type == DataType::FP8_E4M3_PERCHANNEL) {
            return rows * (columns + sizeof(float));
        } else if (type == DataType::FP8_E4M3) {
            return rows * columns * sizeof(uint8_t);
        } else if (type == DataType::INT4_PERCHANNEL) {
            return rows * (columns / 2 + 2 * sizeof(float));
        } else if (type == DataType::INT8_PERCHANNEL) {
            return rows * (columns + 2 * sizeof(float));
        } else if (type == DataType::INT4_GROUP128) {
            rows *= (columns / 128);
            columns = 128;
            return rows * (columns / 2 + 2 * sizeof(float));
        } else if (type == DataType::AWQ_4BIT_128) {
            int groups = (columns - 1) / 128 + 1;
            size_t colBytes = columns / 2 + groups + groups * sizeof(float);
            return rows * colBytes;
        } else if (type == DataType::INF_INT8_PERCHANNEL) {
            size_t colBytes = columns + sizeof(float) + sizeof(int); // [int8 values] + scale + sum
            return rows * colBytes;
        } else if (type == DataType::INF_INT8_GROUP128) {
            size_t colBytes = (columns / 128) * (128 + sizeof(float) + sizeof(int)); // [int8 values] + scale + sum
            return rows * colBytes;
        } else if (type >= DataType::DATA_GGUF_FORMAT && type < DataType::DATA_GGUF_FORMAT_END) {
            size_t colBytes = ggml_row_size((ggml_type)(type - DataType::DATA_GGUF_FORMAT), columns);
            return rows * colBytes;
        } else {
            ErrorInFastLLM("GetDataBytes failed. " + GetDataTypeName(type) + "\n");
            return 0;
        }
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

    struct ByteReader {
        uint8_t *cur;

        ByteReader (uint8_t *data) {
            this->cur = data;
        }

        int ReadInt() {
            int ret = *((int*)cur);
            cur += sizeof(int);
            return ret;
        }

        float ReadFloat() {
            float ret = *((float*)cur);
            cur += sizeof(float);
            return ret;
        }

        std::string ReadString() {
            int len = ReadInt();
            std::string ret = "";
            char *v = new char[len + 5];
            v[len] = 0;
            memcpy(v, cur, len);
            cur += len;
            return v;
        }

        void ReadBytes(uint8_t *buffer, uint64_t bytes) {
            memcpy(buffer, cur, bytes);
            cur += bytes;
        }
    };

    struct ByteWriter {
        uint8_t *cur;

        ByteWriter (uint8_t *data) {
            this->cur = data;
        }

        void WriteInt(int v) {
            *((int*)cur) = v;
            cur += sizeof(int);
        }

        void WriteFloat(float v) {
            *((float*)cur) = v;
            cur += sizeof(float);
        }

        void WriteString(const std::string &s) {
            WriteInt((int)s.size());
            memcpy(cur, s.data(), s.size());
            cur += s.size();
        }

        void WriteBytes(uint8_t *buffer, uint64_t bytes) {
            memcpy(cur, buffer, bytes);
            cur += bytes;
        }
    };

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

    Data::Data(fastllm::DataType type, int ggmlType, const std::vector <int> &dims) {
        this->dataType = type;
        this->ggmlType = (ggml_type)ggmlType;
        Resize(dims);
    }

    Data::Data (DataType type, const std::vector <int> &dims, DataDevice device, void *ptr): Data::Data(type, dims) {
        this->isFake = true;
        this->expansionSize = this->Count(0);
        this->UpdateUnitSize();
        this->dataDevice = device;
        if (device == DataDevice::CPU) {
            this->cpuData = (uint8_t*)ptr;
        } else if (this->dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
            this->cudaData = ptr;
            this->dataDeviceIds = {0}; // todo 支持多卡
#else
            ErrorInFastLLM("Error: cuda is not supported.\n");
#endif
        }
    }

    Data::Data(fastllm::DataType type, const std::vector<int> &dims, const std::vector<float> &data) : Data::Data(type, dims) {
        // std::cout<<"调用数值构造"<<std::endl;
        this->Allocate();
        if (type == DataType::FLOAT32) {
            std::memcpy(this->cpuData, data.data(), this->GetBytes());
        }
    }

    Data::Data(const Data &ori) {
        CopyFrom(ori);
    }

    void Data::FakeFrom(const Data &ori, size_t offset) {
        this->dataType = ori.dataType;
        this->UpdateUnitSize();
        this->isFake = true;
        this->dataDevice = ori.dataDevice;
        if (this->dataDevice == DataDevice::CPU) {
            this->cpuData = ori.cpuData + offset;
        } else if (this->dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
            this->cudaData = (void*)((uint8_t*)ori.cudaData + offset);
#else
            ErrorInFastLLM("Error: cuda is not supported.\n");
#endif
        }
    }

    void Data::CopyFrom(const Data &ori) {
        this->ToDevice(ori.dataDevice);
        this->name = ori.name;
        this->isKVCache = ori.isKVCache;
        this->isLinearAttention = ori.isLinearAttention;
        this->cacheUid = ori.cacheUid;
        this->dataDevice = ori.dataDevice;
        
        // std::cout<<"调用拷贝构造"<<std::endl;
        if (ori.expansionDims != this->expansionDims || ori.dims != this->dims || this->cpuData == nullptr || ori.dataType != this->dataType) {
            if (ori.dims.size() == 0) {
                this->dataType = ori.dataType;
                this->UpdateUnitSize();
                this->dims.resize(0);

                if (this->dataDevice == DataDevice::CPU) {
                    delete[] this->cpuData;
                    this->cpuData = nullptr;
                } else if (this->dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
                    FastllmCudaFree(this->cudaData);
                    this->cudaData = nullptr;
#endif
                }
                return;
            }
            this->dataType = ori.dataType;
            this->UpdateUnitSize();
            if (ori.expansionDims.size() > 0 && ori.expansionDims != ori.dims) {
                this->Expansion(ori.expansionDims);
                this->Resize(ori.dims);
                this->Allocate();
            } else {
                this->expansionDims.clear();
                this->Resize(ori.dims);
                this->FreeSpace();
                this->MallocSpace(Count(0));
            }
        }

        if (this->dataDevice == DataDevice::CPU) {
            std::memcpy(this->cpuData, ori.cpuData, this->GetBytes());
        } else if (this->dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
            FastllmCudaCopyFromDeviceToDevice(this->cudaData, ori.cudaData, this->GetBytes());
#endif
        }
    }

    struct BF16ToFP16Manager {
        float dict[65536];

        BF16ToFP16Manager() {
            for (uint16_t i = 0; i < 65535; i++) {
                uint32_t x = (i << 16);
                dict[i] = float_to_half(*((float*)&x));
            }
        }
    } bf16tofp16;

    extern BF16ToFP32Manager bf16tofp32;

    struct MultiThreadGroupQuantizationBF16Op : MultiThreadBaseOp {
        int st, end, m;
        uint16_t *bf;
        uint8_t *u8;
        LowBitConfig *configs;
        int bit;
        int group, groupCnt;

        MultiThreadGroupQuantizationBF16Op (int st, int end, int m,
                                        uint16_t *bf, uint8_t *u8, LowBitConfig *configs, int bit, int group, int groupCnt) :
                                        st(st), end(end), m(m), bf(bf), u8(u8), configs(configs), bit(bit), group(group), groupCnt(groupCnt) {}
        
        void Run() {
            int type = (bit == 4 || bit == 2) ? 1 : 0;
            for (int i = st; i < end; i++) {
                for (int g = 0; g < group; g++) {
                    int cid = i * group + g;
                    int groupStart = g * groupCnt;
                    int groupEnd = std::min((g + 1) * groupCnt, m);

                    float minValue = 1e9, maxValue = -1e9;
                    for (int j = groupStart; j < groupEnd; j++) {
                        minValue = std::min(minValue, bf16tofp32.dict[bf[i * m + j]]);
                        maxValue = std::max(maxValue, bf16tofp32.dict[bf[i * m + j]]);
                    }
                    if (bit == 8) {
                        configs[cid] = LowBitConfig(minValue, maxValue, 8, type);
                        for (int j = groupStart; j < groupEnd; j++) {
                            u8[i * m + j] = configs[cid].quantization(bf16tofp32.dict[bf[i * m + j]]);
                        }
                    } else if (bit == 4) {
                        configs[cid] = LowBitConfig(minValue, maxValue, 4, type);
                        for (int j = groupStart; j < groupEnd; j++) {
                            int id = (i * m + j) / 2;
                            uint8_t value = configs[cid].quantization(bf16tofp32.dict[bf[i * m + j]]);
                            if ((i * m + j) % 2) {
                                u8[id] = (u8[id] & 0xF0) | value;
                            } else {
                                u8[id] = (u8[id] & 0xF) | (value << 4);
                            }
                        }
                    } else if (bit == 2) {
                        configs[cid] = LowBitConfig(minValue, maxValue, 2, type);
                        for (int j = groupStart; j + 3 < groupEnd; j += 4) {
                            int id = (i * m + j) / 4;
                            uint8_t value0 = configs[cid].quantization(bf16tofp32.dict[bf[i * m + j + 0]]);
                            uint8_t value1 = configs[cid].quantization(bf16tofp32.dict[bf[i * m + j + 1]]);
                            uint8_t value2 = configs[cid].quantization(bf16tofp32.dict[bf[i * m + j + 2]]);
                            uint8_t value3 = configs[cid].quantization(bf16tofp32.dict[bf[i * m + j + 3]]);
                            u8[id] = (value0 << 6) | (value1 << 4) | (value2 << 2) | (value3);
                        }
                    }
                }
            }
        }
    };

    struct MultiThreadGroupQuantizationOp : MultiThreadBaseOp {
        int st, end, m;
        float *f;
        uint8_t *u8;
        LowBitConfig *configs;
        int bit;
        int group, groupCnt;

        MultiThreadGroupQuantizationOp (int st, int end, int m,
                                        float *f, uint8_t *u8, LowBitConfig *configs, int bit, int group, int groupCnt) :
                                        st(st), end(end), m(m), f(f), u8(u8), configs(configs), bit(bit), group(group), groupCnt(groupCnt) {}
        
        void Run() {
            int type = (bit == 4 || bit == 2) ? 1 : 0;
            for (int i = st; i < end; i++) {
                for (int g = 0; g < group; g++) {
                    int cid = i * group + g;
                    int groupStart = g * groupCnt;
                    int groupEnd = std::min((g + 1) * groupCnt, m);

                    float minValue = 1e9, maxValue = -1e9;
                    for (int j = groupStart; j < groupEnd; j++) {
                        minValue = std::min(minValue, f[i * m + j]);
                        maxValue = std::max(maxValue, f[i * m + j]);
                    }
                    if (bit == 8) {
                        configs[cid] = LowBitConfig(minValue, maxValue, 8, type);
                        for (int j = groupStart; j < groupEnd; j++) {
                            u8[i * m + j] = configs[cid].quantization(f[i * m + j]);
                        }
                    } else if (bit == 4) {
                        configs[cid] = LowBitConfig(minValue, maxValue, 4, type);
                        for (int j = groupStart; j < groupEnd; j++) {
                            int id = (i * m + j) / 2;
                            uint8_t value = configs[cid].quantization(f[i * m + j]);
                            if ((i * m + j) % 2) {
                                u8[id] = (u8[id] & 0xF0) | value;
                            } else {
                                u8[id] = (u8[id] & 0xF) | (value << 4);
                            }
                        }
                    } else if (bit == 2) {
                        configs[cid] = LowBitConfig(minValue, maxValue, 2, type);
                        for (int j = groupStart; j + 3 < groupEnd; j += 4) {
                            int id = (i * m + j) / 4;
                            uint8_t value0 = configs[cid].quantization(f[i * m + j + 0]);
                            uint8_t value1 = configs[cid].quantization(f[i * m + j + 1]);
                            uint8_t value2 = configs[cid].quantization(f[i * m + j + 2]);
                            uint8_t value3 = configs[cid].quantization(f[i * m + j + 3]);
                            u8[id] = (value0 << 6) | (value1 << 4) | (value2 << 2) | (value3);
                        }
                    }
                }
            }
        }
    };

    struct MultiThreadBase3GroupQuantizationOp : MultiThreadBaseOp {
        int st, end, m;
        float *f32;
        uint8_t *u8;
        uint16_t *halfScales;
        int group, groupCnt;

        MultiThreadBase3GroupQuantizationOp (int st, int end, int m,
                                        float *f32, uint8_t *u8, uint16_t *halfScales, int group, int groupCnt) :
                                        st(st), end(end), m(m), f32(f32), u8(u8), halfScales(halfScales), group(group), groupCnt(groupCnt) {}
        
        void Run() {
            std::vector <uint8_t> base = {1, 3, 9, 27, 81};
            int bytesPerGroup = ((groupCnt - 1) / 5) + 1;
            for (int i = st; i < end; i++) {
                for (int g = 0; g < group; g++) {
                    uint8_t *cur = u8 + i * group * bytesPerGroup + g * bytesPerGroup;
                    int cid = i * group + g;
                    int groupStart = g * groupCnt;
                    int groupEnd = std::min((g + 1) * groupCnt, m);

                    float minValue = 1e9, maxValue = -1e9, mean = 0.0;
                    for (int j = groupStart; j < groupEnd; j++) {
                        minValue = std::min(minValue, f32[i * m + j]);
                        maxValue = std::max(maxValue, f32[i * m + j]);
                        mean += fabs(f32[i * m + j]);
                    }
                    mean = std::max(1e-5f, mean / (groupEnd - groupStart));
                    float scale = mean;
                    halfScales[i * group + g] = float_to_half(scale);

                    memcpy(cur, cur + bytesPerGroup, 0);
                    for (int j = groupStart; j < groupEnd; j++) {
                        float now = f32[i * m + j];
                        uint8_t curV = (now > -scale * 0.5) + (now > scale * 0.5);
                        cur[(j - groupStart) / 5] += curV * base[(j - groupStart) % 5];
                    }
                }
            }
        }
    };

    struct MultiThreadBase3GroupQuantizationBF16Op : MultiThreadBaseOp {
        int st, end, m;
        uint16_t *bf;
        uint8_t *u8;
        uint16_t *halfScales;
        int group, groupCnt;

        MultiThreadBase3GroupQuantizationBF16Op (int st, int end, int m,
                                        uint16_t *bf, uint8_t *u8, uint16_t *halfScales, int group, int groupCnt) :
                                        st(st), end(end), m(m), bf(bf), u8(u8), halfScales(halfScales), group(group), groupCnt(groupCnt) {}
        
        void Run() {
            std::vector <uint8_t> base = {1, 3, 9, 27, 81};
            int bytesPerGroup = ((groupCnt - 1) / 5) + 1;
            for (int i = st; i < end; i++) {
                for (int g = 0; g < group; g++) {
                    uint8_t *cur = u8 + i * group * bytesPerGroup + g * bytesPerGroup;
                    int cid = i * group + g;
                    int groupStart = g * groupCnt;
                    int groupEnd = std::min((g + 1) * groupCnt, m);

                    float minValue = 1e9, maxValue = -1e9, mean = 0.0;
                    for (int j = groupStart; j < groupEnd; j++) {
                        minValue = std::min(minValue, bf16tofp32.dict[bf[i * m + j]]);
                        maxValue = std::max(maxValue, bf16tofp32.dict[bf[i * m + j]]);
                        mean += fabs(bf16tofp32.dict[bf[i * m + j]]);
                    }
                    mean = std::max(1e-5f, mean / (groupEnd - groupStart));
                    float scale = mean;
                    halfScales[i * group + g] = float_to_half(scale);

                    memcpy(cur, cur + bytesPerGroup, 0);
                    for (int j = groupStart; j < groupEnd; j++) {
                        float now = bf16tofp32.dict[bf[i * m + j]];
                        uint8_t curV = (now > -scale * 0.5) + (now > scale * 0.5);
                        cur[(j - groupStart) / 5] += curV * base[(j - groupStart) % 5];
                    }
                }
            }
        }
    };

    struct MultiThreadPerChannelQuantizationBF16Op : MultiThreadBaseOp {
        int st, end, m;
        uint16_t *bf;
        uint8_t *u8;
        LowBitConfig *configs;
        int bit;

        MultiThreadPerChannelQuantizationBF16Op (int st, int end, int m,
                                           uint16_t *bf, uint8_t *u8, LowBitConfig *configs, int bit) :
                                           st(st), end(end), m(m), bf(bf), u8(u8), configs(configs), bit(bit) {}
    
        void Run() {
            int type = (bit == 4) ? 1 : 0;
            for (int i = st; i < end; i++) {
                float minValue = 1e9, maxValue = -1e9;
                for (int j = 0; j < m; j++) {
                    minValue = std::min(minValue, bf16tofp32.dict[bf[i * m + j]]);
                    maxValue = std::max(maxValue, bf16tofp32.dict[bf[i * m + j]]);
                }
                if (bit == 8) {
                    configs[i] = LowBitConfig(minValue, maxValue, 8, type);
                    for (int j = 0; j < m; j++) {
                        u8[i * m + j] = configs[i].quantization(bf16tofp32.dict[bf[i * m + j]]);
                    }
                } else {
                    configs[i] = LowBitConfig(minValue, maxValue, 4, type);
                    for (int j = 0; j < m; j++) {
                        int id = (i * m + j) / 2;
                        uint8_t value = configs[i].quantization(bf16tofp32.dict[bf[i * m + j]]);
                        if ((i * m + j) % 2) {
                            u8[id] = (u8[id] & 0xF0) | value;
                        } else {
                            u8[id] = (u8[id] & 0xF) | (value << 4);
                        }
                    }
                }
            }
        }
    };

    struct MultiThreadPerChannelQuantizationOp : MultiThreadBaseOp {
        int st, end, m;
        float *f;
        uint8_t *u8;
        LowBitConfig *configs;
        int bit;

        MultiThreadPerChannelQuantizationOp (int st, int end, int m,
                                           float *f, uint8_t *u8, LowBitConfig *configs, int bit) :
                                           st(st), end(end), m(m), f(f), u8(u8), configs(configs), bit(bit) {}
    
        void Run() {
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
    };

    void Data::CreateFromOriData(WeightType weightType, DataType oriDataType, uint8_t *oriData, float *oriMins, float *oriScales, 
            int groupCnt, int blockK, int blockM) {
        auto &data = *this;
        data.weightType = weightType;
        data.UpdateUnitSize();
        data.Allocate();
        if (dataType == oriDataType) {
            if (oriData != nullptr) {
                memcpy(data.cpuData, oriData, data.GetBytes());
            } 
            if (dataType == DataType::INT4_GROUP) {
                int k = this->dims[0], m = this->dims[1], group = (m - 1) / groupCnt + 1;
                this->group = group;
                this->groupCnt = groupCnt;
                data.mins.resize(k * group);
                data.scales.resize(k * group);
                memcpy(data.mins.data(), oriMins, k * group * sizeof(float));
                memcpy(data.scales.data(), oriScales, k * group * sizeof(float));
                data.perChannelAxis = 0;
                /* data.perChannelsConfigs.resize(k * group);
                for (int i = 0; i < k * group; i++) {
                    data.perChannelsConfigs[i] = LowBitConfig(data.mins[i], data.mins[i] + 15 * data.scales[i], 4, 1);
                    data.perChannelsConfigs[i].min = data.mins[i];
                    data.perChannelsConfigs[i].scale = data.scales[i];
                } */
            } else if (dataType == DataType::FP8_E4M3) {
                this->blockK = blockK;
                this->blockM = blockM;
                int ks = (this->dims[0] - 1) / this->blockK + 1;
                int ms = (this->dims[1] - 1) / this->blockM + 1;
                data.scales.resize(ks * ms);
                memcpy(data.scales.data(), oriScales, ks * ms * sizeof(float));
            }
        } else if (oriDataType == DataType::BFLOAT16
                && dataType == DataType::FLOAT16) {
            uint16_t *a = (uint16_t*)data.cpuData;
            uint16_t *b = (uint16_t*)oriData;
            int len = data.Count(0);
            for (int i = 0; i < len; i++) {
                a[i] = bf16tofp16.dict[b[i]];
            }
        } else if (oriDataType == DataType::FLOAT32 
                && dataType == DataType::FLOAT16) {
            uint16_t *a = (uint16_t*)data.cpuData;
            float *b = (float*)oriData;
            int len = data.Count(0);
            for (int i = 0; i < len; i++) {
                a[i] = float_to_half(b[i]);
            }
        } else if ((oriDataType == DataType::FLOAT32 || oriDataType == DataType::BFLOAT16)
                && dataType == DataType::INT4_GROUP) {
            int bit = (dataType == DataType::INT4_GROUP) ? 4 : 8;
            int type = (bit == 4) ? 1 : 0;
            int k = data.dims[0], m = data.dims[1];
            if (groupCnt == -1) {
                groupCnt = 128;
            }
            int group = (m - 1) / groupCnt + 1;
            std::vector<LowBitConfig> configs;
            std::vector<uint8_t> uDatas;
            configs.resize(k * group);

            int bytes = k * m;
            if (bit == 4) {
                bytes = (k * m + 1) / 2;
            }
            uDatas.resize(bytes);
            if (oriDataType == DataType::FLOAT32) {
                (MultiThreadGroupQuantizationOp(0, k, m, (float*)oriData, uDatas.data(), configs.data(), bit, group, groupCnt)).Run();
            } else if (oriDataType == DataType::BFLOAT16) {
                (MultiThreadGroupQuantizationBF16Op(0, k, m, (uint16_t*)oriData, uDatas.data(), configs.data(), bit, group, groupCnt)).Run();
            }
            data.perChannelAxis = 0;
            // data.perChannelsConfigs.resize(k * group);
            data.group = group;
            data.groupCnt = groupCnt;
            // data.zeros.resize(k * group);
            data.scales.resize(k * group);
            data.mins.resize(k * group);
            for (int i = 0; i < k * group; i++) {
                auto config = LowBitConfig(configs[i].min, configs[i].max, bit, type);
                data.mins[i] = config.min;
                // data.zeros[i] = config.zeroPoint;
                data.scales[i] = config.scale;
            }
            memcpy((uint8_t*)data.cpuData, (uint8_t*)uDatas.data(), bytes);
        } else if ((oriDataType == DataType::FLOAT32 || oriDataType == DataType::BFLOAT16) &&
                (dataType == DataType::INT8 || dataType == DataType::INT4_NOZERO)) {
            int bit = (dataType == DataType::INT4_NOZERO) ? 4 : 8;
            int type = (bit == 4) ? 1 : 0;
            int k = data.dims[0], m = data.dims[1];
            std::vector<LowBitConfig> configs;
            std::vector<uint8_t> uDatas;
            configs.resize(k);

            int bytes = k * m;
            if (bit == 4) {
                bytes = (k * m + 1) / 2;
            }
            uDatas.resize(bytes);
            if (oriDataType == DataType::FLOAT32) {
                (MultiThreadPerChannelQuantizationOp(0, k, m, (float *) oriData, uDatas.data(), configs.data(), bit)).Run();
            } else if (oriDataType == DataType::BFLOAT16) {
                (MultiThreadPerChannelQuantizationBF16Op(0, k, m, (uint16_t *) oriData, uDatas.data(), configs.data(), bit)).Run();
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
        } else if ((oriDataType == DataType::FLOAT32 || oriDataType == DataType::BFLOAT16) &&
                (dataType == DataType::BASE3_GROUP)) {
            int k = data.dims[0], m = data.dims[1];
            if (groupCnt == -1) {
                groupCnt = 128;
            }
            int group = (m - 1) / groupCnt + 1;
            int bytesPerGroup = ((groupCnt - 1) / 5) + 1;
            std::vector<uint16_t> scales;
            std::vector<uint8_t> uDatas;
            scales.resize(k * group);
            int bytes = k * group * bytesPerGroup;
            uDatas.resize(bytes);
            data.group = group;
            data.groupCnt = groupCnt;
            data.halfScales.resize(k * group);

            if (oriDataType == DataType::FLOAT32) {
               (MultiThreadBase3GroupQuantizationOp(0, k, m, (float*)oriData, uDatas.data(), data.halfScales.data(), group, groupCnt)).Run();
            } else if (oriDataType == DataType::BFLOAT16) {
               (MultiThreadBase3GroupQuantizationBF16Op(0, k, m, (uint16_t*)oriData, uDatas.data(), data.halfScales.data(), group, groupCnt)).Run();
            }
            memcpy((uint8_t*)data.cpuData, (uint8_t*)uDatas.data(), bytes);
        } else if ((oriDataType == DataType::FLOAT32 || oriDataType == DataType::BFLOAT16)
                && dataType == DataType::INT2_GROUP) {
            int bit = 2;
            int type = 1;
            int k = data.dims[0], m = data.dims[1];
            if (groupCnt == -1) {
                groupCnt = 32;
            }
            int group = (m - 1) / groupCnt + 1;
            std::vector<LowBitConfig> configs;
            std::vector<uint8_t> uDatas;
            configs.resize(k * group);

            int bytes = k * m / 4;
            uDatas.resize(bytes);
            if (oriDataType == DataType::FLOAT32) {
                (MultiThreadGroupQuantizationOp(0, k, m, (float*)oriData, uDatas.data(), configs.data(), bit, group, groupCnt)).Run();
            } else if (oriDataType == DataType::BFLOAT16) {
                (MultiThreadGroupQuantizationBF16Op(0, k, m, (uint16_t*)oriData, uDatas.data(), configs.data(), bit, group, groupCnt)).Run();
            }
            data.perChannelAxis = 0;
            data.group = group;
            data.groupCnt = groupCnt;
            data.scales.resize(k * group);
            data.mins.resize(k * group);
            for (int i = 0; i < k * group; i++) {
                auto config = LowBitConfig(configs[i].min, configs[i].max, bit, type);
                data.mins[i] = config.min;
                data.scales[i] = config.scale;
            }
            memcpy((uint8_t*)data.cpuData, (uint8_t*)uDatas.data(), bytes);
        } else if (oriDataType == DataType::FLOAT32 && dataType == DATA_GGUF_FORMAT) {
            uint8_t *a = (uint8_t*)data.cpuData;
            float *b = (float*)oriData;
            int len = data.dims[0] * data.dims[1];

            auto from_float = ggml_type_from_float_ref((ggml_type)data.ggmlType);
            if (from_float == nullptr) {
                printf("Failed to find convert function for %s\n", ggml_type_name((ggml_type)data.ggmlType));
                exit(0);
            }
            from_float (
                b, a, len
            );
        } else {
            ErrorInFastLLM("wrong data type " + dataTypeNames[oriDataType][0] + " -> " + dataTypeNames[dataType][0]);
        }
    }

    uint64_t Data::Count(int i) const {
        if (this->dataType == DataType::DATA_GGUF_FORMAT && i == 0) {
            return ggml_nbytes((ggml_tensor*)this->ggmlTensor);
        }
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
        } else if (this->dataType == DataType::INT8 || this->dataType == DataType::FP8_E4M3) {
            this->unitSize = 1;
            this->unitSizeDiv = 1;
        } else if (this->dataType == DataType::INT4 
                || this->dataType == DataType::INT4_NOZERO
                || this->dataType == DataType::INT4_GROUP) {
            this->unitSize = 1;
            this->unitSizeDiv = 2;
        } else if (this->dataType == DataType::INT2
                || this->dataType == DataType::INT2_GROUP) {
            this->unitSize = 1;
            this->unitSizeDiv = 4;
        } else if (this->dataType == DataType::BIT) {
            this->unitSize = 1;
            this->unitSizeDiv = 8;
        } else if (this->dataType == DataType::INT32PARAM) {
            this->unitSize = 4;
            this->unitSizeDiv = 1;
        } else if (this->dataType == DataType::DATA_GGUF_FORMAT) {
            // 用GGUF的函数来计算长度
            this->unitSize = 1;
            this->unitSizeDiv = 1;
        }

        this->expansionBytes = (this->expansionSize * this->unitSize - 1) / this->unitSizeDiv + 1;
    }

    void Data::Resize(const std::vector<int> &dims) {
        this->dims = dims;
        this->UpdateUnitSize();

        if (this->dataType == DATA_GGUF_FORMAT) {
            std::vector <int> cur = dims;
            std::reverse(cur.begin(), cur.end());

            if (this->ggmlTensor == nullptr) {
                this->ggmlTensor = new ggml_tensor();
            }
            ggml_tensor* tensor = (ggml_tensor*)this->ggmlTensor;
            tensor->type = (ggml_type)this->ggmlType;
            for (int j = 0; j < GGML_MAX_DIMS; j++) {
                tensor->ne[j] = 1;
                if (j < cur.size()) {
                    tensor->ne[j] = cur[j];
                }
            }

            {
                tensor->type = (ggml_type)this->ggmlType;
                const size_t  type_size = ggml_type_size(tensor->type);
                const int64_t blck_size = ggml_blck_size(tensor->type);
                tensor->nb[0] = type_size;
                tensor->nb[1] = tensor->nb[0] * (tensor->ne[0] / blck_size);
                for (int j = 2; j < GGML_MAX_DIMS; ++j) {
                    tensor->nb[j] = tensor->nb[j - 1] * tensor->ne[j - 1];
                }
            }
        }

        if (this->expansionDims.size() == 0) {
            this->strides.resize(dims.size(), 1);
            this->strides.back() = 1;
            for (int i = this->dims.size() - 2; i >= 0; i--) {
                this->strides[i] = this->dims[i + 1] * this->strides[i + 1];
            }
        }
    }

    void Data::Reshape(const std::vector<int> &dims) {
        if (this->dims == dims) {
            return;
        }
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
        if (this->dataType == DataType::DATA_GGUF_FORMAT) {
            return ggml_nbytes((ggml_tensor*)this->ggmlTensor);
        }
        return (this->strides[0] * this->dims[0] * this->unitSize - 1) / this->unitSizeDiv + 1;
    }

    void Data::MallocSpace(uint64_t size) {
        this->expansionSize = size;
        this->expansionBytes = (size * this->unitSize - 1) / this->unitSizeDiv + 1;
        if (this->dataDevice == DataDevice::CPU) {
            this->cpuData = new uint8_t[this->expansionBytes];
            memset(this->cpuData, 0, this->expansionBytes*sizeof(uint8_t));
        } else if (this->dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
            if (this->directMemory) {
                this->cudaData = FastllmCudaDirectMalloc(this->expansionBytes);
            } else {
                this->cudaData = FastllmCudaMalloc(this->expansionBytes);
            }
            FastllmCudaMemset0(this->cudaData, this->expansionBytes);
#else
            ErrorInFastLLM("Error: cuda is not supported.\n");
#endif
        }
    }

    void Data::FreeSpace() {
        if (isFake)
            return;
        this->expansionSize = 0;
        this->expansionBytes = 0;
        if (this->dataDevice == DataDevice::CPU) {
#ifdef USE_MMAP
            if (this->name.empty())
                delete[] this->cpuData;
#else
            delete[] this->cpuData;
#endif
            this->cpuData = nullptr;
        } else if (this->dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
            if (this->directMemory) {
                FastllmCudaDirectFree(this->cudaData);
            } else {
                FastllmCudaFree(this->cudaData);
            }
            this->cudaData = nullptr;
#else
            ErrorInFastLLM("Error: cuda is not supported.\n");
#endif
        }
    }

    void Data::Allocate() {
        if (!isFake && Count(0) > expansionSize) {
            FreeSpace();
            MallocSpace(Count(0));
        }
    }

    void Data::Allocate(float v) {
        AssertInFastLLM(this->dataType == DataType::FLOAT32
                        || this->dataType == DataType::FLOAT16, "Allocate error: Data's type should be float32 or float16.\n");
        this->Allocate();
        if (this->dataDevice == DataDevice::CPU) {
            if (this->dataType == DataType::FLOAT32) {
                float *f = (float*)cpuData;
                std::fill(f, f + Count(0), v);
            } else if (this->dataType == DataType::FLOAT16) {
                uint16_t *h = (uint16_t*)cpuData;
                std::fill(h, h + Count(0), float_to_half(v));
            }
        } if (this->dataDevice == DataDevice::CUDA) {
#ifdef USE_CUDA
            if (this->dataType == DataType::FLOAT32) {
                std::vector <float> f = std::vector <float> (Count(0), v);
                FastllmCudaCopyFromHostToDevice(cudaData, f.data(), Count(0) * sizeof(float));
            } else if (this->dataType == DataType::FLOAT16) {
                std::vector <uint16_t> f = std::vector <uint16_t> (Count(0), float_to_half(v));
                FastllmCudaCopyFromHostToDevice(cudaData, f.data(), Count(0) * sizeof(uint16_t));
            }
#endif
        } else {
            // TODO: 别的设备上的初始化
        }
    }

    void Data::Expansion(const std::vector<int> &dims) {
        if (this->dims.size() == 0) {
            this->directMemory = true;
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
        if (this->multiDeviceData) {
            for (auto it : this->multiDeviceDatas) {
                delete it.second;
            }
        }
        if (isFake) {
            return;
        }
        if (this->cpuData != nullptr)
#ifdef USE_MMAP
            if (this->name.empty())
                delete[] this->cpuData;
#else
           delete[] this->cpuData;
#endif
#ifdef USE_CUDA
        if (this->cudaData != nullptr) {
            FastllmCudaFree(this->cudaData);
            /*if (this->directMemory) {
                FastllmCudaDirectFree(this->cudaData);
            } else {
                FastllmCudaFree(this->cudaData);
            }*/
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

    std::vector<int> Data::Shape() const{
        return this->dims;
    }

    void Data::Print() const {
        ((Data*)this)->ToDevice(DataDevice::CPU);
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
        std::vector <float> floatData;
        floatData.resize(this->Count(0));
        if (this->dataType == DataType::FLOAT32) {
            memcpy(floatData.data(), cpuData, this->Count(0) * sizeof(float));
        } else if (this->dataType == DataType::FLOAT16) {
            for (int i = 0; i < floatData.size(); i++) {
                floatData[i] = half_to_float(((uint16_t*)cpuData)[i]);
            }
        }

        for (int i = 0; i < n; i++) {
            if (i == 10) {
                printf("...\n");
            }
            if (i >= 10 && i <= n - 10) {
                continue;
            }
            for (int j = 0; j < 3 && j < m; j++) {
                printf("%f ", floatData[i * m + j]);
            }
            if (m > 3) {
                printf("... ");
                for (int j = 0; j < 3 && j < m; j++) {
                    printf("%f ", floatData[i * m + (m - 3 + j)]);
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
#ifdef __AVX2__
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
#ifdef __AVX2__
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
        } else if (this->dataType == DataType::INT4_GROUP) {
            weightSum.resize(n * this->group);
            for (int i = 0; i < n; i++) {
                for (int g = 0; g < this->group; g++) {
                    int gid = i * this->group + g;
                    int st = g * this->groupCnt;
                    int end = std::min(m, (g + 1) * this->groupCnt);
                    int j = st;
#ifdef __aarch64__
                    uint8x8_t maskHigh = vdup_n_u8(0xF0);
                    uint8x8_t maskLow = vdup_n_u8(0xF);
                    uint32x4_t sum0 = {0, 0, 0, 0};

                    for (; j + 15 < end; j += 16) {
                        uint8x8_t ori = vld1_u8(cpuData + (i * m + j) / 2);
                        uint8x8_t va = vand_u8(ori, maskLow);
                        uint8x8_t vb = vshr_n_u8(vand_u8(ori, maskHigh), 4);

                        uint16x4_t sa = vpaddl_u8 (va);
                        uint16x4_t sb = vpaddl_u8 (vb);

                        sum0 = vaddw_u16(sum0, vadd_u16(sa, sb));
                    }
                    weightSum[gid] += sum0[0] + sum0[1] + sum0[2] + sum0[3];
#endif
#ifdef __AVX2__
                    __m256i acc = _mm256_setzero_si256();
                    const __m256i lowMask = _mm256_set1_epi8(0xf);
                    const __m256i ones = _mm256_set1_epi16(1);
                    for (; j + 31 < end; j += 32) {
                        __m128i orix = _mm_loadu_si128((const __m128i *) (cpuData + (i * m + j) / 2));
                        __m256i bytex = _mm256_set_m128i(_mm_srli_epi16(orix, 4), orix);
                        __m256i bx = _mm256_and_si256(lowMask, bytex);

                        __m256i mx0 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(bx, 0));
                        __m256i mx1 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(bx, 1));

                        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx0, ones));
                        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx1, ones));
                    }
                    weightSum[gid] += I32sum(acc);
#endif
                    for (; j + 1 < end; j += 2) {
                        int id = (i * m + j) / 2;
                        weightSum[gid] += (cpuData[id] & 0xF) + (cpuData[id] >> 4);
                    }
                    for (; j < end; j++) {
                        int id = (i * m + j) / 2;
                        if ((i * m + j) % 2) {
                            weightSum[gid] += (cpuData[id] & 0xF);
                        } else {
                            weightSum[gid] += (cpuData[id] >> 4);
                        }
                    }
                }
            }
        } 
    }

    void Data::ToDevice(void *device) {
        BaseDevice *dev = (BaseDevice*)device;
        if (dev->deviceType == "cuda" || dev->deviceType == "multicuda") {
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
                    uint8_t *cpuData = this->cpuData;
#ifdef USE_MMAP
                    cpuData = new uint8_t[expansionBytes];
                    memcpy(cpuData, this->cpuData, expansionBytes);
#endif
                    // FastllmCudaSetDevice(deviceIds.size() == 0 ? 0 : deviceIds[0]);
                    this->cudaData = FastllmCudaMalloc(expansionBytes);
                    FastllmCudaCopyFromHostToDevice(this->cudaData, cpuData, expansionBytes);
#ifdef USE_MMAP
                    delete[] cpuData;
#else
                    delete[] this->cpuData;
                    this->cpuData = nullptr;
#endif
                }
            } else if (this->dataDevice == DataDevice::CUDA) {
                if (device == DataDevice::CPU) {
                    this->cpuData = new uint8_t[expansionBytes];
                    FastllmCudaCopyFromDeviceToHost(this->cpuData, this->cudaData, expansionBytes);
                    FastllmCudaFree(this->cudaData);
                    this->cudaData = nullptr;
                } else if (device == DataDevice::CUDA) {
                    int sourceDevice = this->dataDeviceIds.size() == 0 ? 0 : this->dataDeviceIds[0];
                    int destDevice = deviceIds.size() == 0 ? 0 : deviceIds[0];
                    if (sourceDevice != destDevice) {
                                        FastllmCudaSetDevice(destDevice);
                                        void *newCudaData = FastllmCudaMalloc(expansionBytes);
                                        FastllmCudaMemcpyBetweenDevices(destDevice, newCudaData, sourceDevice, this->cudaData, expansionBytes);
                                        FastllmCudaSetDevice(sourceDevice);
                                        FastllmCudaFree(this->cudaData);
                                        this->cudaData = newCudaData;
                                        FastllmCudaSetDevice(destDevice);
                    }
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

    extern CPUInstructInfo cpuInstructInfo;

    void Data::Repack() {
        if (this->IsRepacked || this->dataType != DATA_GGUF_FORMAT) {
            return;
        }
        if (GetEnableAMX() && cpuInstructInfo.hasAMX) {
            return;
        }
        this->IsRepacked = true;
        ggml_tensor *tensor = (ggml_tensor*)this->ggmlTensor;
        auto repack = get_repack_info(tensor->type);
        if (repack != nullptr) {
// printf("repack %s (%s).\n", tensor->name.c_str(), ggml_type_name(tensor->type));
            int nrows = tensor->ne[1], n_per_row = tensor->ne[0];
            auto row_size = ggml_row_size(tensor->type, n_per_row);
            std::vector<uint8_t> qtmp(repack->num_rows * row_size);
            uint8_t *qcur = (uint8_t*)this->cpuData;
            for (int row = 0; row < nrows; row += repack->num_rows) {
                memcpy(qtmp.data(), qcur, repack->num_rows * row_size);
                repack->repack(repack->num_rows, n_per_row, (const char *)qtmp.data(), (char *)qcur, false);
                qcur += repack->num_rows * row_size;
            }

            ((ggml_tensor*)this->ggmlTensor)->type = repack->new_type;
            this->ggmlType = (int)repack->new_type;
        } else {
            // printf("name = %s, type = %s\n", tensor->name.c_str(), ggml_type_name(tensor->type));
            // weight->PrintShape();
        }
    }

    void Data::SetKVCache() {
        this->isKVCache = true;
        this->cacheUid = ((long long)this) * rand() * rand() * rand() * rand();
    }

    // 计算形成Fastllm格式需要多少Bytes
    uint64_t Data::GetFastllmFormateBytes() {
        if (this->dataType == FLOAT16 || this->dataType == FLOAT32 || this->dataType == BFLOAT16) {
            return this->GetBytes();
        } 
        uint64_t ret = 0;
        ret += sizeof(int) * 2;
        if (this->dataType == FP8_E4M3) {
            ret += sizeof(int) * 3;
            ret += this->scales.size() * sizeof(float);
            ret += this->GetBytes();
        } else if (this->dataType == INT4_NOZERO ||
                    this->dataType == INT4 ||
                    this->dataType == INT8) {
            ret += sizeof(int);
            int k = this->perChannelAxis == -1 ? 1 : this->dims[this->perChannelAxis];
            ret += k * 2 * sizeof(float);
            ret += this->GetBytes();
        } else if (this->dataType == INT4_GROUP) {
            ret += sizeof(int) * 3;
            int k = this->perChannelAxis == -1 ? 1 : this->dims[this->perChannelAxis];
            ret += k * this->group * 2 * sizeof(float);
            ret += this->GetBytes();
        } else if (this->dataType == DATA_GGUF_FORMAT) {
            ret += sizeof(int);
            ret += this->GetBytes();
        } else {
            ErrorInFastLLM("ExportFastllmFormat Error: data type error.");
        }
        return ret;
    }

    // 导出成Fastllm格式
    void Data::ExportFastllmFormat(uint8_t *bytes) {
        ByteWriter writer(bytes);
        if (this->dataType == FLOAT16 || this->dataType == FLOAT32 || this->dataType == BFLOAT16) {
            writer.WriteBytes(this->cpuData, GetBytes());
            return;
        } 
        writer.WriteInt(1); // 版本号
        writer.WriteInt((int)this->dataType);
        if (this->dataType == FP8_E4M3) {
            writer.WriteInt(this->blockK);
            writer.WriteInt(this->blockM);
            writer.WriteInt((int)this->scales.size());
            writer.WriteBytes((uint8_t*)this->scales.data(), (int)this->scales.size() * sizeof(float));
            writer.WriteBytes(this->cpuData, this->GetBytes());
        } else if (this->dataType == INT8 || this->dataType == INT4 || this->dataType == INT4_NOZERO) {
            writer.WriteInt(this->perChannelAxis);
            int k = this->perChannelAxis == -1 ? 1 : this->dims[this->perChannelAxis];
            for (int i = 0; i < k; i++) {
                writer.WriteFloat(this->perChannelsConfigs[i].min);
                if (this->dataType == INT4_NOZERO) {
                    writer.WriteFloat(this->perChannelsConfigs[i].scale);
                } else {
                    writer.WriteFloat(this->perChannelsConfigs[i].max);
                }
            }
            writer.WriteBytes(this->cpuData, this->GetBytes());
        } else if (this->dataType == INT4_GROUP) {
            writer.WriteInt(this->perChannelAxis);
            writer.WriteInt(this->group);
            writer.WriteInt(this->groupCnt);
            int k = this->perChannelAxis == -1 ? 1 : this->dims[this->perChannelAxis];
            for (int i = 0; i < k * this->group; i++) {
                writer.WriteFloat(this->mins[i]);
                writer.WriteFloat(this->scales[i]);
            }
            writer.WriteBytes(this->cpuData, this->GetBytes());
        } else if (this->dataType == DATA_GGUF_FORMAT) {
            writer.WriteInt(this->ggmlType);
            writer.WriteBytes(this->cpuData, this->GetBytes());
        } else {
            ErrorInFastLLM("ExportFastllmFormat Error: data type error.");
        }
    }

    // 从Fastllm格式中创建
    void Data::CreateFromFastllmFormat(uint8_t *datas, uint64_t len) {
        this->weightType = WeightType::AUTO;
        ByteReader reader(datas);
        int version = reader.ReadInt();
        if (version == 1) {
            this->dataType = (DataType)reader.ReadInt();
            if (this->dataType == DATA_GGUF_FORMAT) {
                this->ggmlType = reader.ReadInt();
            }
            this->Resize(this->dims);
            this->Allocate();
            if (this->dataType == FLOAT16 || this->dataType == FLOAT32 || this->dataType == BFLOAT16) {
                reader.ReadBytes(this->cpuData, len);
                return;
            } else if (this->dataType == FP8_E4M3) {
                this->blockK = reader.ReadInt();
                this->blockM = reader.ReadInt();
                this->scales.resize(reader.ReadInt());
                reader.ReadBytes((uint8_t*)this->scales.data(), (int)this->scales.size() * sizeof(float));
                reader.ReadBytes(this->cpuData, this->GetBytes());
            } else if (this->dataType == INT8 || this->dataType == INT4 || this->dataType == INT4_NOZERO) {
                this->perChannelAxis = reader.ReadInt();
                int k = this->perChannelAxis == -1 ? 1 : this->dims[this->perChannelAxis];
                this->perChannelsConfigs.resize(k);
                this->mins.resize(k);
                this->scales.resize(k);
                this->zeros.resize(k);
                for (int i = 0; i < k; i++) {
                    if (this->dataType == INT4_NOZERO) {
                        float minValue = reader.ReadFloat();
                        float scale = reader.ReadFloat();
                        this->perChannelsConfigs[i] = LowBitConfig(minValue, minValue + 15 * scale, 4, 1);
                        this->perChannelsConfigs[i].min = minValue;
                        this->perChannelsConfigs[i].scale = scale;
                    } else {
                        int bit = (dataType == DataType::INT4 ? 4 : 8);
                        float minValue = reader.ReadFloat();
                        float maxValue = reader.ReadFloat();
                        this->perChannelsConfigs[i] = LowBitConfig(minValue, maxValue, bit, 0);
                    }
                    this->mins[i] = this->perChannelsConfigs[i].min;
                    this->scales[i] = this->perChannelsConfigs[i].scale;
                    this->zeros[i] = this->perChannelsConfigs[i].zeroPoint;
                }
                reader.ReadBytes(this->cpuData, this->GetBytes());
            } else if (this->dataType == INT4_GROUP) {
                this->perChannelAxis = reader.ReadInt();
                this->group = reader.ReadInt();
                this->groupCnt = reader.ReadInt();
                int k = this->perChannelAxis == -1 ? 1 : this->dims[this->perChannelAxis];
                // this->perChannelsConfigs.resize(k * this->group);
                this->mins.resize(k * this->group);
                this->scales.resize(k * this->group);
                // this->zeros.resize(k * this->group);
                for (int i = 0; i < k * this->group; i++) {
                    float minValue = reader.ReadFloat();
                    float scale = reader.ReadFloat();
                    // this->perChannelsConfigs[i] = LowBitConfig(minValue, minValue + 15 * scale, 4, 1);
                    // this->perChannelsConfigs[i].min = minValue;
                    // this->perChannelsConfigs[i].scale = scale;
                    this->mins[i] = minValue;
                    this->scales[i] = scale;
                    // this->zeros[i] = this->perChannelsConfigs[i].zeroPoint;
                }
                reader.ReadBytes(this->cpuData, this->GetBytes());
            } else if (this->dataType == DATA_GGUF_FORMAT) {
                reader.ReadBytes(this->cpuData, this->GetBytes());
            } else {
                ErrorInFastLLM("CreateFromFastllmFormat Error: data type error.");
            }
        } else {
            ErrorInFastLLM("CreateFromFastllmFormat error: unsupport version " + std::to_string(version));
        }
    }

    DataType Data::GetDataType() {
        if (this->dataType == DataType::DATA_GGUF_FORMAT) {
            return (DataType)((int)DataType::DATA_GGUF_FORMAT + this->ggmlType);
        } else {
            return this->dataType;
        }
    }

    DataType Data::GetLinearActDataType(int batchSize) {
        if (this->dataType == DataType::DATA_GGUF_FORMAT) {
            if (batchSize > 31 && GetEnableAMX() && cpuInstructInfo.hasAMX) {
                return DataType::BFLOAT16;
            } else {
                return (DataType)((int)DataType::DATA_GGUF_FORMAT + ggml_type_vec_dot_type((ggml_type)this->ggmlType));
            }
        } else if (this->dataType == DataType::FLOAT16) {
            return DataType::FLOAT32;
        } else if (this->dataType == DataType::INT4_PERCHANNEL ||
                    this->dataType == DataType::INT8_PERCHANNEL) {
            return DataType::INF_INT8_PERCHANNEL;
        } else if (this->dataType == DataType::INT4_GROUP128) {
            return DataType::INF_INT8_GROUP128;
        } else if (this->dataType == DataType::BFLOAT16 || 
                    this->dataType == DataType::FP8_E4M3 ||
                    this->dataType == DataType::FP8_E4M3_BLOCK_128 ||
                    this->dataType == DataType::FP8_E4M3_PERCHANNEL) {
            return DataType::BFLOAT16;
        } else {
            ErrorInFastLLM("GetLinearActDataType failed with type " + GetDataTypeName(this->dataType));
            return DataType::FLOAT32;
        }
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
            std::multiset<int>::iterator begin = tokens.tokenSet.begin();
            std::multiset<int>::iterator end = tokens.tokenSet.end();
            std::set<int> unique(tokens.tokenSet.begin(), tokens.tokenSet.end());
            if (config.last_n <= 0) {
                begin = unique.begin();
                end = unique.end();
            }
            for (std::multiset<int>::iterator iter = begin; iter != end; ++iter) {
                int id = *iter;
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
        float rnd = fastllmRandom.randP() * curSum;
        curSum = 0.0;
        for (int i = 0; i < topk; i++) {
            curSum += ps[i];
            if (curSum > rnd || i == topk - 1) {
                return v[i].second;
            }
        }
        return -1;
    }

    // 已经做过repeat_penalty和topk，仅做采样
    int LLMSamplingOnly(Data &logits, int outerOffset, const GenerationConfig &config) {
        int maxTopKSize = logits.dims.back() / 2;
        float *base = ((float*)logits.cpuData) + outerOffset * maxTopKSize * 2;
        float invTemp = 1.0f / config.temperature;
        int topk = config.top_k;
        float psum = 0.0, maxValue = base[1];
        std::vector <float> ps;
        for (int i = 0; i < topk; i++) {
            ps.push_back(expf(base[i * 2 + 1] - maxValue));
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
        float rnd = fastllmRandom.randP() * curSum;
        curSum = 0.0;
        for (int i = 0; i < topk; i++) {
            curSum += ps[i];
            if (curSum > rnd || i == topk - 1) {
                return base[i * 2];
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

        if (this->dicts.find("peft_size") != this->dicts.end()) {
            int peftSize = atoi(this->dicts["peft_size"].c_str());
            for (int i = 0; i < peftSize; i++) {
                std::string adapter_name = buffer.ReadString();
                this->peftDict[adapter_name] = {};

                int adapter_size = buffer.ReadInt();
                for (int j = 0; j < adapter_size; j++) {
                    std::string key = buffer.ReadString();
                    std::string value = buffer.ReadString();
                    //printf("%s %s\n", key.c_str(), value.c_str());
                    this->peftDict[adapter_name][key] = value;
                }
            }
        }

        bool useScore = this->dicts.find("tokenizer_use_score") != this->dicts.end()
                && this->dicts["tokenizer_use_score"] == "1";
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
        bool hasSpecialTokens = this->dicts.find("tokenizer_has_special_tokens") != this->dicts.end()
                && this->dicts["tokenizer_has_special_tokens"] == "1";
        if (hasSpecialTokens) {
            std::map <std::string, int> specialTokens;
            int specialTokenLen = buffer.ReadInt();
            for (int i = 0; i < specialTokenLen; i++) {
                std::string token = buffer.ReadString();
                int id = tokenizer.stringToTokenDict[token];
                specialTokens[token] = id;
            }
            tokenizer.SetSpecialTokens(specialTokens);
        }
        if (this->dicts.find("chat_template") != this->dicts.end())
            tokenizer.chatTemplate = this->dicts["chat_template"];

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
            weight[name].name = name;

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
                weight[name].SetMapFile(mapped_file);
                weight[name].expansionBytes = (weight[name].Count(0) * weight[name].unitSize - 1) / weight[name].unitSizeDiv + 1;
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
                } else if (dataType == DataType::INT4_GROUP) {
                    auto &curWeight = weight[name];
                    int bit = 4;
                    curWeight.perChannelAxis = buffer.ReadInt();
                    curWeight.group = buffer.ReadInt();
                    curWeight.groupCnt = buffer.ReadInt();
                    int k = curWeight.perChannelAxis == -1 ? 1 : dims[curWeight.perChannelAxis];
                    k *= curWeight.group;
                    curWeight.perChannelsConfigs.resize(k);
                    curWeight.mins.resize(k);
                    curWeight.scales.resize(k);
                    for (int i = 0; i < k; i++) {
                        float minValue = buffer.ReadFloat();
                        float maxValue = buffer.ReadFloat();
                        auto config = LowBitConfig(minValue, maxValue, bit, 1);
                        curWeight.perChannelsConfigs[i] = config;
                        curWeight.mins[i] = config.min;
                        curWeight.scales[i] = config.scale;
                    }
#ifdef USE_MMAP
                    curWeight.cpuData = buffer.ReadBytes(curWeight.GetBytes());
#else
                    buffer.ReadBytes(curWeight.cpuData, curWeight.GetBytes());
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
        bool useScore = this->dicts.find("tokenizer_use_score") != this->dicts.end()
                && this->dicts["tokenizer_use_score"] == "1";
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
        bool hasSpecialTokens = this->dicts.find("tokenizer_has_special_tokens") != this->dicts.end()
                && this->dicts["tokenizer_has_special_tokens"] == "1";
        if (hasSpecialTokens) {
            int specialTokenLen = tokenizer.specialTokens.size();
            buffer.WriteInt(specialTokenLen);
            for (int i = 0; i < specialTokenLen; i++) {
                buffer.WriteString(tokenizer.specialTokens[i]);
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
                } else if (dataType == DataType::INT4_GROUP) {
                    buffer.WriteInt((int) dataType);
                    buffer.WriteInt(data.perChannelAxis);
                    buffer.WriteInt(data.group);
                    buffer.WriteInt(data.groupCnt);
                    int k = data.perChannelAxis == -1 ? 1 : data.dims[data.perChannelAxis];
                    for (int i = 0; i < k * data.group; i++) {
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
                        auto *pool = GetAlivePool();
                        int threadNum = pool->threads.size();
                        int per = k / threadNum;
                        int cur = 0;
                        std::vector<LowBitConfig> configs;
                        std::vector<uint8_t> uDatas;
                        configs.resize(k);

                        int bytes = k * m;
                        if (bit == 4) {
                            bytes = (k * m + 1) / 2;
                        }
                        uDatas.resize(bytes);
                        std::vector<fastllm::MultiThreadPerChannelQuantizationOp*> ops;
                
                        for (int i = 0; i < threadNum; i++) {
                            int end = cur + per;
                            if (i == threadNum - 1) {
                                end = k;
                            }
                            ops.push_back(new MultiThreadPerChannelQuantizationOp(cur, end, m,
                                                              (float *) data.cpuData, uDatas.data(), configs.data(), bit));
                            cur = end;
                        }
                        for (int i = 0; i < ops.size(); i++) {
                            pool->PushOp(i, ops[i]);
                        }
                        for (int i = 0; i < ops.size(); i++) {
                            pool->Wait(i);
                            delete ops[i];
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

    void WeightMap::AddAdapterDict(const std::string &name, const std::string &key, const std::string &value) {
        this->peftDict[name][key] = value;
    }

    WeightType WeightMap::GetWeightType(const std::string &key) {
        if (this->embeddingNames.find(key) != this->embeddingNames.end()) {
            return WeightType::EMBEDDING;
        }
        for (auto &linearName : this->linearNames) {
            int n = key.size(), m = linearName.size();
            std::vector <std::vector <bool> > f = std::vector <std::vector <bool> > (n + 1, std::vector <bool>(m + 1, 0));
            f[0][0] = 1;
            for (int i = 0; i <= n; i++) {
                for (int j = 0; j <= m; j++) {
                    if (f[i][j]) {
                        if (i + 1 <= n && key[i] == '*') {
                            for (int l = j; l <= m; l++) {
                                f[i + 1][l] = 1;
                            }
                        }
                        if (j + 1 <= m && linearName[j] == '*') {
                            for (int l = i; l <= n; l++) {
                                f[l][j + 1] = 1;
                            }
                        }
                        if (i + 1 <= n && j + 1 <= m && key[i] == linearName[j]) {
                            f[i + 1][j + 1] = 1;
                        }
                    }
                }
            }
            if (f[n][m]) {
                return WeightType::LINEAR;
            }
        }
        return WeightType::NONE;
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
        this->weight[key].name = std::string(key);
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

    void WeightMap::AddEmptyWeight(const std::string &key, const std::vector<int> &dims, fastllm::DataType dataType) {
        this->weight[key] = Data(dataType, dims);
        this->weight[key].name = std::string(key);
    }

    void WeightMap::AddEmptyGGMLWeight(const std::string &key, const std::vector<int> &dims, fastllm::DataType dataType, int ggmlType) {
        this->weight[key] = Data(dataType, ggmlType, dims);
        this->weight[key].name = std::string(key);
    }

    void WeightMap::AddWeight(const std::string &key, const std::vector<int> &dims, fastllm::DataType dataType,
                              fastllm::WeightType weightType, fastllm::DataType oriDataType, uint8_t *oriData, int groupCnt) {
        if (weightType == WeightType::AUTO) {
            weightType = GetWeightType(key);
            if (weightType == WeightType::EMBEDDING) {
                dataType = oriDataType;
            }
            if (weightType == WeightType::NONE) {
                dataType = oriDataType;
            }
        }

        this->weight[key] = Data(dataType, dims);
        this->weight[key].name = std::string(key);
        Data &data = this->weight[key];
        data.weightType = weightType;
        data.UpdateUnitSize();
        data.Allocate();
        if (dataType == oriDataType) {
            memcpy(data.cpuData, oriData, data.GetBytes());
        } else if (oriDataType == DataType::FLOAT32 
                && dataType == DataType::FLOAT16) {
            uint16_t *a = (uint16_t*)data.cpuData;
            float *b = (float*)oriData;
            int len = data.Count(0);
            for (int i = 0; i < len; i++) {
                a[i] = float_to_half(b[i]);
            }
        } else if (oriDataType == DataType::FLOAT32 
                && dataType == DataType::INT4_GROUP) {
            int bit = (dataType == DataType::INT4_GROUP) ? 4 : 8;
            int type = (bit == 4) ? 1 : 0;
            int k = data.dims[0], m = data.dims[1];
            auto pool = GetAlivePool();
            int threadNum = pool->threads.size();
            int per = k / threadNum;
            int cur = 0;
            if (groupCnt == -1) {
                groupCnt = 128;
            }
            int group = (m - 1) / groupCnt + 1;
            std::vector<LowBitConfig> configs;
            std::vector<uint8_t> uDatas;
            configs.resize(k * group);

            int bytes = k * m;
            if (bit == 4) {
                bytes = (k * m + 1) / 2;
            }
            uDatas.resize(bytes);
            std::vector<fastllm::MultiThreadGroupQuantizationOp*> ops;
            for (int i = 0; i < threadNum; i++) {
                int end = cur + per;
                if (i == threadNum - 1) {
                    end = k;
                }
                ops.push_back(new MultiThreadGroupQuantizationOp(cur, end, m,
                                                (float *) oriData, uDatas.data(), configs.data(), bit, group, groupCnt));
                cur = end;
            }
            for (int i = 0; i < ops.size(); i++) {
                pool->PushOp(i, ops[i]);
            }
            for (int i = 0; i < ops.size(); i++) {
                pool->Wait(i);
                delete ops[i];
            }

            data.perChannelAxis = 0;
            data.perChannelsConfigs.resize(k * group);
            data.group = group;
            data.groupCnt = groupCnt;
            data.zeros.resize(k * group);
            data.scales.resize(k * group);
            data.mins.resize(k * group);
            for (int i = 0; i < k * group; i++) {
                data.perChannelsConfigs[i] = LowBitConfig(configs[i].min, configs[i].max, bit, type);
                data.mins[i] = data.perChannelsConfigs[i].min;
                data.zeros[i] = data.perChannelsConfigs[i].zeroPoint;
                data.scales[i] = data.perChannelsConfigs[i].scale;
            }
            memcpy((uint8_t*)data.cpuData, (uint8_t*)uDatas.data(), bytes);
        } else if (oriDataType == DataType::FLOAT32 &&
                (dataType == DataType::INT8 || dataType == DataType::INT4_NOZERO)) {
            int bit = (dataType == DataType::INT4_NOZERO) ? 4 : 8;
            int type = (bit == 4) ? 1 : 0;
            int k = data.dims[0], m = data.dims[1];
            auto pool = GetAlivePool();
            int threadNum = pool->threads.size();
            int per = k / threadNum;
            int cur = 0;
            std::vector<LowBitConfig> configs;
            std::vector<uint8_t> uDatas;
            configs.resize(k);

            int bytes = k * m;
            if (bit == 4) {
                bytes = (k * m + 1) / 2;
            }
            uDatas.resize(bytes);
            std::vector<fastllm::MultiThreadPerChannelQuantizationOp*> ops;
            for (int i = 0; i < threadNum; i++) {
                int end = cur + per;
                if (i == threadNum - 1) {
                    end = k;
                }
                ops.push_back(new MultiThreadPerChannelQuantizationOp(cur, end, m,
                (float *) oriData, uDatas.data(), configs.data(), bit));
                cur = end;
            }
            for (int i = 0; i < ops.size(); i++) {
                pool->PushOp(i, ops[i]);
            }
            for (int i = 0; i < ops.size(); i++) {
                pool->Wait(i);
                delete ops[i];
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
            ErrorInFastLLM("wrong data type " + dataTypeNames[oriDataType][0] + " -> " + dataTypeNames[dataType][0]);
        }
    }

    void WeightMap::ReleaseWeight() {
        for (auto &w : this->weight) {
            w.second.FreeSpace();
        }
#ifdef USE_CUDA
        FastllmCudaClearBigBuffer();
#endif
    }

    Data &WeightMap::operator[](const std::string &key) {
        return weight[key];
    }

    void ToDataType(const Data &input, DataType dataType) {
        if (input.dataType == dataType) {
            return;
        }
        if (dataType == DataType::FLOAT32) {
            curExecutor->Run("ToFloat32", {
                    {"input", (Data*)&input}
            }, {}, {});
        } else if (dataType == DataType::FLOAT16) {
            curExecutor->Run("ToFloat16", {
                    {"input", (Data*)&input}
            }, {}, {});
        } else {
            ErrorInFastLLM("ToDataType: Unsupport data type.\n");
        }
    }

    void ToDataType(const Data &input, Data &output, DataType dataType) {
        if (dataType == DataType::FLOAT32) {
            curExecutor->Run("ConvertToFloat32", {
                    {"input", (Data*)&input}, {"output", (Data*)&output}
            }, {}, {});
        } else if (dataType == DataType::FLOAT16) {
            curExecutor->Run("ConvertToFloat16", {
                {"input", (Data*)&input}, {"output", (Data*)&output}
            }, {}, {});
        } else {
            ErrorInFastLLM("ToDataType: Unsupport data type.\n");
        }
    }

    void CopyKVCache(Data &oldCache, Data &newCache, int oldBsStart, int newBsStart, int bs, int offset) {
        curExecutor->Run("CopyKVCache", {
                {"oldCache", (Data*)&oldCache}, {"newCache", (Data*)&newCache}
        }, {}, {
            {"oldBsStart", oldBsStart}, {"newBsStart", newBsStart}, {"bs", bs}, {"offset", offset}
        });
    }

    bool CanRunMergeMOE(const Data &input, std::vector <Data*> &biass) {
        return curExecutor->CanRunOnFirstDevice("MergeMOE", {{"input", (Data*)&input}, {"biass", (Data*)biass.data()}}, {}, {});
    }

    void MergeMOE(const Data &input, const Data &logits, Data &gateBias, std::vector <Data*> &weights, std::vector <Data*> &biass, 
                Data &w1, Data &w2, Data &w3, Data &curInput, Data &curOutput,
                float routeScale, float sharedScale, int topk, bool needNorm, Data &output) {
        curExecutor->Run("MergeMOE", {
                {"input", (Data*)&input}, {"logits", (Data*)&logits}, {"gateBias", (Data*)&gateBias},
                {"weights", (Data*)weights.data()}, {"biass", (Data*)biass.data()},
                {"w1", (Data*)&w1}, {"w2", (Data*)&w2}, {"w3", (Data*)&w3},
                {"curInput", &curInput}, {"curOutput", &curOutput},
                {"output", (Data*)&output}
        }, {{"sharedScale", sharedScale}, {"routeScale", routeScale}}, {{"topk", topk}, {"needNorm", needNorm}, 
                                        {"weights___batch", (int)weights.size()}, {"biass___batch", (int)biass.size()}});
    }

    void MergeMLA(Data &qNope, Data &qPe, Data &kvCache, Data &peCache, const Data &mask, Data &output, float softmaxScale) {
        curExecutor->Run("MergeMLA", {
            {"qNope", (Data*)&qNope}, {"qPe", (Data*)&qPe}, {"kvCache", (Data*)&kvCache}, {"peCache", (Data*)&peCache},
            {"mask", (Data*)&mask}, {"output", (Data*)&output}
        }, {{"softmaxScale", softmaxScale}}, {});
    }

    // attentionType
    // 1: normal
    // 2: 不做mask

    void Attention(const Data &q, const Data &k, const Data &v, const Data &mask, Data &output,
                   int group, float scale, int attentionType) {
        int maskType = 0; // 0: 因果mask, 2: 不做mask
        if (attentionType == 2) {
            maskType = 2;
        }
        curExecutor->Run("Attention", {
                {"q", (Data*)&q}, {"k", (Data*)&k}, {"v", (Data*)&v},
                {"mask", (Data*)&mask}, {"output", (Data*)&output}
        }, {{"scale", scale}}, {{"group", group}, {"maskType", maskType}});
    }

    void Conv1DPerChannel(const Data &input, Data &weight, Data &bias, int inputChannels, int outputChannels, 
            int kernel, int stride, int pad, Data &output) {
        curExecutor->Run("Conv1DPerChannel", {
                {"input", (Data*)&input}, {"weight", &weight}, {"bias", (Data*)&bias}, {"output", &output}
        }, {}, {{"inputChannels", inputChannels}, {"outputChannels", outputChannels}, {"kernel", kernel}, 
                {"stride", stride}, {"pad", pad}});
    }

    void Conv2D(const Data &input, Data &weight, Data &bias, int inputChannels, int outputChannels, int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, Data &output) {
        curExecutor->Run("Conv2D", {
                {"input", (Data*)&input}, {"weight", &weight}, {"bias", (Data*)&bias}, {"output", &output}
        }, {}, {{"inputChannels", inputChannels}, {"outputChannels", outputChannels}, {"kernelH", kernelH}, {"kernelW", kernelW}, 
                {"strideH", strideH}, {"strideW", strideW}, {"padH", padH}, {"padW", padW}});
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

    bool CanRunLinearEx(LinearExType exType) {
        return curExecutor->CanRunOnFirstDevice("Linear", {}, {}, {{"exType", (int)exType}});
    }

    bool CanRunMLP() {
        return curExecutor->CanRunOnFirstDevice("MLP", {}, {}, {});
    }

    bool CanRunMergeAttention() {
        return curExecutor->CanRunOnFirstDevice("MergeAttention", {}, {}, {});
    }

    void LinearEx(Data &input, Data &weight, const Data &bias, Data &output, LinearExType exType) {
        curExecutor->Run("Linear", {
                {"input", &input}, {"weight", &weight}, {"bias", (Data*)&bias}, {"output", &output}
        }, {}, {{"exType", (int)exType}});
    }

    void MLP(Data &input, Data &weight0, const Data &bias0, Data &weight1, const Data &bias1, 
            Data &w1, Data &w2, Data &w3, Data &output) {
        curExecutor->Run("MLP", {
                {"input", &input}, 
                {"weight0", &weight0}, {"bias0", (Data*)&bias0}, 
                {"weight1", &weight1}, {"bias1", (Data*)&bias1}, 
                {"w1", &w1}, {"w2", &w2}, {"w3", &w3},
                {"output", &output}
        }, {}, {});
    }

    void MergeAttention(Data &input, Data &weight0, Data &bias0, Data &weight1, Data &bias1, 
        Data &qkv, Data &q, Data &k, Data &v, Data &curInput, Data &curOutput,
        int qNum, int kvNum, int headDim, int rotDim, float attentionScale,
        const Data &positionIds, Data &sinData, Data &cosData,
        std::vector <Data*> &keys, std::vector <Data*> &values, std::vector <Data*> &masks, 
        Data &output) {
        curExecutor->Run("MergeAttention", {
                {"input", &input}, 
                {"weight0", &weight0}, {"bias0", &bias0}, 
                {"weight1", &weight1}, {"bias1", &bias1}, 
                {"qkv", &qkv}, {"q", &q}, {"k", &k}, {"v", &v}, 
                {"curInput", &curInput}, {"curOutput", &curOutput},
                {"positionIds", (Data*)&positionIds},
                {"sinData", (Data*)&sinData},
                {"cosData", (Data*)&cosData},
                {"keys", (Data*)keys.data()}, {"values", (Data*)values.data()}, {"masks", (Data*)masks.data()},
                {"output", &output}
        }, {{"attentionScale", attentionScale}}, 
        {{"qNum", qNum}, {"kvNum",kvNum}, {"headDim", headDim}, {"rotDim", rotDim},
        {"keys___batch", (int)keys.size()}, {"values___batch", (int)values.size()}, {"masks___batch", (int)masks.size()}});
    }

    void Split(const Data &input, int axis, int start, int end, Data &output) {
        curExecutor->Run("Split", {
                {"input", (Data*)&input}, {"output", &output}
        }, {}, {{"axis", axis}, {"start", start}, {"end", end}});
    }

    void Repeat(const Data &input, int axis, int repeatTimes, Data &output) {
        curExecutor->Run("Repeat", {
                {"input", (Data*)&input}, {"output", &output}
        }, {}, {{"axis", axis}, {"repeatTimes", repeatTimes}});
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

    void MatMul(const Data &input0, const Data &input1, Data &output, float alpha, int group) {
        curExecutor->Run("MatMul", {
                {"input0", (Data*)&input0}, {"input1", (Data*)&input1}, {"output", &output}
        }, {{"alpha", alpha}}, {{"group", group}});
    }

    void MatMulTransB(const Data &input0, const Data &input1, Data &output, float alpha, int group) {
        curExecutor->Run("MatMulTransB", {
                {"input0", (Data*)&input0}, {"input1", (Data*)&input1}, {"output", &output}
        }, {{"alpha", alpha}}, {{"group", group}});
    }

    void Softmax(const Data &input, Data &output, int axis) {
        curExecutor->Run("SoftMax", {
                {"input", (Data*)&input}, {"output", &output}
        }, {}, {{"axis", axis}});
    }

    void Normalize(const Data &input, Data &output, int axis) {
        curExecutor->Run("Normalize", {
                {"input", (Data*)&input}, {"output", &output}
        }, {}, {{"axis", axis}});
    }

    void Silu(const fastllm::Data &input, fastllm::Data &output) {
        curExecutor->Run("Silu", {
                {"input", (Data*)&input}, {"output", &output}
        }, {}, {});
    }

    void TanH(const Data &input, Data &output) {
        curExecutor->Run("TanH", {
                {"input", (Data*)&input}, {"output", &output}
        }, {}, {});
    }

    void Relu(const fastllm::Data &input, fastllm::Data &output) {
        curExecutor->Run("Relu", {
                {"input", (Data*)&input}, {"output", &output}
        }, {}, {});
    }

    void Sigmoid(const fastllm::Data &input, fastllm::Data &output) {
        curExecutor->Run("Sigmoid", {
                {"input", (Data*)&input}, {"output", &output}
        }, {}, {});
    }

    void Exp(const fastllm::Data &input, fastllm::Data &output) {
        curExecutor->Run("Exp", {
                {"input", (Data*)&input}, {"output", &output}
        }, {}, {});
    }

    void Gelu(const fastllm::Data &input, fastllm::Data &output) {
        curExecutor->Run("Gelu", {
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

    void MambaSoftplus(const Data &input, Data &aLog, Data &dtBias, Data &output) {
        curExecutor->Run("MambaSoftplus", {
                {"input", (Data*)&input}, {"aLog", &aLog}, {"dtBias", &dtBias}, {"output", &output}
        }, {}, {});
    }

    void SwigluGptOss(const fastllm::Data &input, fastllm::Data &output) {
        curExecutor->Run("SwigluGptOss", {
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

    void CausalMask(Data &input, int base, float maskValue) {
        curExecutor->Run("CausalMask", {
                {"input", &input}
        }, {{"maskValue", maskValue}}, {{"base", base}});
    }

    void TransferAttn(Data &input) {
        curExecutor->Run("TransferAttn", {
                {"input", &input}
        }, {}, {});
    }

    void RecurrentGatedDeltaRule(Data &q, Data &k, Data &v, Data &g, Data &b, 
                                Data &last_recurrent_state, Data &core_attn_out) {
        curExecutor->Run("RecurrentGatedDeltaRule", {
            {"q", &q}, {"k", &k}, {"v", &v}, {"g", &g}, {"b", &b}, 
            {"last_recurrent_state", &last_recurrent_state}, {"core_attn_out", &core_attn_out}
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

    void AttentionExtendedMask(Data &input, const Data &mask) {
        curExecutor->Run("AttentionExtendedMask", {
                {"input", &input}, {"mask", (Data*)&mask}
        }, {}, {});
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

    void NearlyRotatePosition2D(Data &input, const Data &positionIds, Data &sinData, Data &cosData, int rotaryDim, int positionStride) {
        curExecutor->Run("NearlyRotatePosition2D", {
                {"input", &input}, {"positionIds", (Data*)&positionIds}, {"sin", &sinData}, {"cos", &cosData}
        }, {}, {{"rotaryDim", rotaryDim}, {"positionStride", positionStride}});
    }

    void LlamaRotatePosition2D(Data &input, const Data &positionIds, Data &sinData, Data &cosData, int rotaryDim) {
        curExecutor->Run("LlamaRotatePosition2D", {
                {"input", &input}, {"positionIds", (Data*)&positionIds}, {"sin", &sinData}, {"cos", &cosData}
        }, {}, {{"rotaryDim", rotaryDim}});
    }

    void LlamaRotatePosition2DPart(Data &input, const Data &positionIds, Data &sinData, Data &cosData, int rotaryDim, int part) {
        curExecutor->Run("LlamaRotatePosition2DPart", {
                {"input", &input}, {"positionIds", (Data*)&positionIds}, {"sin", &sinData}, {"cos", &cosData}
        }, {}, {{"rotaryDim", rotaryDim}, {"part", part}});
    }

    void RepeatPenalty(Data &input, const Data &penalty, const Data &penaltyScale) {
        curExecutor->Run("RepeatPenalty", {
                {"input", &input}, {"penalty", (Data*)&penalty}, {"penaltyScale", (Data*)&penaltyScale}
        }, {}, {});
    }

    void ApplyLognAttn(Data &input, const Data &lognAttn, const Data &positionIds) {
        curExecutor->Run("ApplyLognAttn", {
            {"input", &input}, {"lognAttn", (Data *) &lognAttn}, {"positionIds", (Data *) &positionIds}
        }, {}, {});
    }

    void CumSumLastDim(Data &input) {
        curExecutor->Run("CumSumLastDim", {
            {"input", &input}
        }, {}, {});
    }

    void MakeDecayMask(Data &input, Data &output) {
        curExecutor->Run("MakeDecayMask", {
            {"input", &input}, {"output", &output}
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

    void AppendKVCacheBatch(std::vector <Data*> &caches, const Data &input) {
        curExecutor->Run("AppendKVCachebatch", {
                {"caches", (Data*)caches.data()}, {"input", (Data*)&input}
        }, {}, {{"caches___batch", (int)caches.size()}});
    }

    void AttentionBatch(std::vector <Data*> &q, std::vector <Data*> &k, std::vector <Data*> &v,
                        std::vector <Data*> &mask, std::vector <Data*> &output,
                        int group, float scale, int attentionType) {
        curExecutor->Run("AttentionBatch", {
                {"q", (Data*)q.data()}, {"k", (Data*)k.data()}, {"v", (Data*)v.data()},
                {"mask", (Data*)mask.data()}, {"output", (Data*)output.data()}
        },
        {{"scale", scale}},
        {
            {"group", group},
            {"q___batch", (int)q.size()}, {"k___batch", (int)k.size()}, {"v___batch", (int)v.size()},
            {"mask___batch", (int)mask.size()}, {"output___batch", (int)output.size()}
        });
    }

    void LoraLayer(Data &input, Data &weight, Data &loraA, Data &loraB, const Data &bias, Data &output, 
                   std::map <std::string, std::string> loraConfig) {
        float r = std::atof(loraConfig["r"].c_str());
        float lora_alpha = std::atof(loraConfig["lora_alpha"].c_str());
        bool fan_in_fan_out = loraConfig["fan_in_fan_out"] == "true";
        if (r > 0) {
            float scaling = lora_alpha / r;
            if (fan_in_fan_out) {
                Data weightTrans;
                Data result, loraAOut, loraBOut;
                Permute(weight, {1, 0}, weightTrans);
                Linear(input, weightTrans, bias, result);
                Linear(input, loraA, Data(), loraAOut);
                Linear(loraAOut, loraB, Data(), loraBOut);
                Mul(loraBOut, scaling, output);
                AddTo(output, result);  
            } else {
                Data result, loraAOut, loraBOut;
                Linear(input, weight, bias, result);
                Linear(input, loraA, Data(), loraAOut);
                Linear(loraAOut, loraB, Data(), loraBOut);
                Mul(loraBOut, scaling, output);
                AddTo(output, result);  
            }
        } else {
            if (fan_in_fan_out) {
                Data weightTrans;
                Permute(weight, {1, 0}, weightTrans);
                Linear(input, weightTrans, bias, output);
            } else {
                Linear(input, weight, bias, output);
            }
        }
    }

    void IA3Layer(Data &input, Data &weight, Data &ia3_l, Data &bias, Data &output,
                  std::map <std::string, std::string> ia3Config) {
        bool is_feedforward = ia3Config["if_feedforward"] == "true";
        bool fan_in_fan_out = ia3Config["fan_in_fan_out"] == "true";
        if (is_feedforward) {
            // IA3_L shape: (1, in_features)
            // output = linear(input * ia3_l)
            if (fan_in_fan_out) {
                Data weightTrans;
                Permute(weight, {1, 0}, weightTrans);
                MulTo(input, ia3_l);
                Linear(input, weightTrans, bias, output);
            } else {
                MulTo(input, ia3_l);
                Linear(input, weight, bias, output);
            }
        } else {
            // IA3_L shape: (out_features, 1)
            // output = linear(input) * ia3_l
            if (fan_in_fan_out) {
                Data weightTrans;
                Permute(weight, {1, 0}, weightTrans);
                Linear(input, weightTrans, bias, output);
                MulTo(output, ia3_l);
            } else {
                Linear(input, weight, bias, output);
                MulTo(output, ia3_l);
            }
        }
    }

    void *GetExecutor() {
        return (void*)curExecutor;
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

    void SetMoeDeviceMap(const std::map <std::string, int> &deviceMap) {
        defaultMoeDeviceMap = deviceMap;
    }

    std::map <std::string, int> GetMoeDeviceMap() {
        return defaultMoeDeviceMap;
    }
}
