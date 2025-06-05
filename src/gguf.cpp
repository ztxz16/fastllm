#include "gguf.h"

namespace fastllm {
    GGUFBuffer::GGUFBuffer (const std::string &fileName) {
        this->f = fopen(fileName.c_str(), "rb");
    }

    GGUFBuffer::~GGUFBuffer () {
        fclose(this->f);
    }

    template <typename T>
    T GGUFBuffer::Read() {
        T v;
        if (fread(&v, 1, sizeof(T), f) != sizeof(T)) {
            ErrorInFastLLM("GGUFBuffer.Read error.\n");
        };
        return v;
    }

    bool GGUFBuffer::ReadBool() {
        int8_t v;
        int ret = fread(&v, 1, 1, f);
        return (v != 0);
    }

    std::string GGUFBuffer::ReadString() {
        uint64_t len = Read <uint64_t> ();
        std::vector <char> v;
        v.resize(len + 5);
        int ret = fread(v.data(), 1, len, f);
        std::string s;
        for (int i = 0; i < len; i++) {
            s += v[i];
        }
        return s;
    }

    void GGUFBuffer::ReadBytes(uint8_t *buffer, uint64_t bytes) {
        if (fread(buffer, 1, bytes, f) != bytes) {
            ErrorInFastLLM("GGUFBuffer.ReadBytes error.\n");
        }
    }

    template uint8_t GGUFBuffer::Read<uint8_t>();
    template uint16_t GGUFBuffer::Read<uint16_t>();
    template uint32_t GGUFBuffer::Read<uint32_t>();
    template uint64_t GGUFBuffer::Read<uint64_t>();
    template int8_t GGUFBuffer::Read<int8_t>();
    template int16_t GGUFBuffer::Read<int16_t>();
    template int32_t GGUFBuffer::Read<int32_t>();
    template int64_t GGUFBuffer::Read<int64_t>();
    template float GGUFBuffer::Read<float>();

    void ReadGGUF(const std::string &fileName) {
        // 仅做测试用
        int ggufAlignment = GGUF_DEFAULT_ALIGNMENT;
        GGUFBuffer ggufBuffer = GGUFBuffer(fileName);
        int magic = ggufBuffer.Read<int> ();
        int version = ggufBuffer.Read<int> ();
        uint64_t tensorCount = ggufBuffer.Read <uint64_t> ();
        uint64_t metaDataCount = ggufBuffer.Read <uint64_t> ();

        printf("magic = %d\n", magic);
        printf("version = %d\n", version);
        printf("tensorCount = %d\n", (int)tensorCount);
        printf("metaDataCount = %d\n", (int)metaDataCount);

        for (int i = 0; i < metaDataCount; i++) {
            std::string key = ggufBuffer.ReadString();
            printf("key = %s\n", key.c_str());
            int type = ggufBuffer.Read <int> ();            
            if (type == GGUF_TYPE_STRING) {
                std::string value = ggufBuffer.ReadString();
                printf("value = %s\n", value.c_str());
            } else if (type == GGUF_TYPE_UINT8) {
                int8_t value = ggufBuffer.Read <int8_t> ();
                printf("value = %d\n", value);
            } else if (type == GGUF_TYPE_UINT16) {
                uint16_t value = ggufBuffer.Read <uint16_t> ();
                printf("value = %d\n", value);
            } else if (type == GGUF_TYPE_UINT32) {
                uint32_t value = ggufBuffer.Read <uint32_t> ();
                printf("value = %u\n", value);
            } else if (type == GGUF_TYPE_FLOAT32) {
                float value = ggufBuffer.Read <float> ();
                printf("value = %f\n", value);
            } else if (type == GGUF_TYPE_INT32) {
                int value = ggufBuffer.Read <int> ();
                printf("value = %d\n", value);
            } else if (type == GGUF_TYPE_BOOL) {
                bool value = ggufBuffer.ReadBool();
                printf("value = %d\n", value);
            } else if (type == GGUF_TYPE_ARRAY) {
                int type = ggufBuffer.Read <int> ();
                uint64_t n = ggufBuffer.Read <uint64_t> ();
                printf("type = %d\n", type);
                for (int i = 0; i < n; i++) {
                    if (type == GGUF_TYPE_STRING) {
                        std::string value = ggufBuffer.ReadString();
                    } else if (type == GGUF_TYPE_INT32) {
                        int a = ggufBuffer.Read <int> ();
                    }
                }
            } else {
                printf("type = %d\n", type);
                exit(0);
            }
        }

        for (int i = 0; i < tensorCount; i++) {
            std::string tensorName = ggufBuffer.ReadString();
            uint32_t ndims = ggufBuffer.Read <uint32_t> ();
            
            std::vector <int64_t> dims;
            for (int i = 0; i < ndims; i++) {
                int64_t dim = ggufBuffer.Read <int64_t> ();
                dims.push_back(dim);
            }
            int type = ggufBuffer.Read <int> ();
            uint64_t offset = ggufBuffer.Read <uint64_t> ();

            if (i < 30) {
                printf("name = %s\n", tensorName.c_str());
                printf("ndims = %d\n", ndims);
                for (int i = 0; i < dims.size(); i++) {
                    printf("%d ", (int)dims[i]);
                }
                printf("\n");
                printf("type = %d\n", type);
                printf("offset = %llu\n", (long long unsigned int)offset);
            }
        }

        // we require the data section to be aligned, so take into account any padding
        if (fseek(ggufBuffer.f, GGML_PAD(ftell(ggufBuffer.f), ggufAlignment), SEEK_SET) != 0) {
            printf("alignment error\n");
            exit(0);
        }
        
        exit(0);
    }
}