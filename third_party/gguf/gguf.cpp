#include <map>
#include <regex>

#include <assert.h>
#include "gguf.h"

typedef uint16_t ggml_fp16_t;
GGML_API float       ggml_fp16_to_fp32(ggml_fp16_t);
GGML_API ggml_fp16_t ggml_fp32_to_fp16(float);
GGML_API void        ggml_fp16_to_fp32_row(const ggml_fp16_t *, float *, int64_t);
GGML_API void        ggml_fp32_to_fp16_row(const float *, ggml_fp16_t *, int64_t);

// google brain half-precision bfloat16
typedef struct { uint16_t bits; } ggml_bf16_t;
GGML_API ggml_bf16_t ggml_fp32_to_bf16(float);
GGML_API float       ggml_bf16_to_fp32(ggml_bf16_t);  // consider just doing << 16
GGML_API void        ggml_bf16_to_fp32_row(const ggml_bf16_t *, float *, int64_t);
GGML_API void        ggml_fp32_to_bf16_row_ref(const float *, ggml_bf16_t *, int64_t);
GGML_API void        ggml_fp32_to_bf16_row(const float *, ggml_bf16_t *, int64_t);

std::map <ggml_type, ggml_type_traits> type_traits = {
        {GGML_TYPE_I8, {
            .type_name                = "i8",
            .blck_size                = 1,
            .type_size                = sizeof(int8_t),
            .is_quantized             = false,
        }},
        {GGML_TYPE_I16, {
            .type_name                = "i16",
            .blck_size                = 1,
            .type_size                = sizeof(int16_t),
            .is_quantized             = false,
        }},
        {GGML_TYPE_I32, {
            .type_name                = "i32",
            .blck_size                = 1,
            .type_size                = sizeof(int32_t),
            .is_quantized             = false,
        }},
        {GGML_TYPE_I64, {
            .type_name                = "i64",
            .blck_size                = 1,
            .type_size                = sizeof(int64_t),
            .is_quantized             = false,
        }},
        {GGML_TYPE_F64, {
            .type_name                = "f64",
            .blck_size                = 1,
            .type_size                = sizeof(double),
            .is_quantized             = false,
        }},
        {GGML_TYPE_F32, {
            .type_name                = "f32",
            .blck_size                = 1,
            .type_size                = sizeof(float),
            .is_quantized             = false,
        }},
        {GGML_TYPE_F16, {
            .type_name                = "f16",
            .blck_size                = 1,
            .type_size                = sizeof(ggml_fp16_t),
            .is_quantized             = false,
            // .to_float                 = (ggml_to_float_t) ggml_fp16_to_fp32_row,
            // .from_float_ref           = (ggml_from_float_t) ggml_fp32_to_fp16_row,
        }},
        {GGML_TYPE_Q4_0, {
            .type_name                = "q4_0",
            .blck_size                = QK4_0,
            .type_size                = sizeof(block_q4_0),
            .is_quantized             = true,
            // .to_float                 = (ggml_to_float_t) dequantize_row_q4_0,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q4_0_ref,
        }},
        {GGML_TYPE_Q4_1, {
            .type_name                = "q4_1",
            .blck_size                = QK4_1,
            .type_size                = sizeof(block_q4_1),
            .is_quantized             = true,
            // .to_float                 = (ggml_to_float_t) dequantize_row_q4_1,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q4_1_ref,
        }},
        {GGML_TYPE_Q5_0, {
            .type_name                = "q5_0",
            .blck_size                = QK5_0,
            .type_size                = sizeof(block_q5_0),
            .is_quantized             = true,
            // .to_float                 = (ggml_to_float_t) dequantize_row_q5_0,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q5_0_ref,
        }},
        {GGML_TYPE_Q5_1, {
            .type_name                = "q5_1",
            .blck_size                = QK5_1,
            .type_size                = sizeof(block_q5_1),
            .is_quantized             = true,
            // .to_float                 = (ggml_to_float_t) dequantize_row_q5_1,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q5_1_ref,
        }},
        {GGML_TYPE_Q8_0, {
            .type_name                = "q8_0",
            .blck_size                = QK8_0,
            .type_size                = sizeof(block_q8_0),
            .is_quantized             = true,
            // .to_float                 = (ggml_to_float_t) dequantize_row_q8_0,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q8_0_ref,
        }},
        {GGML_TYPE_Q8_1, {
            .type_name                = "q8_1",
            .blck_size                = QK8_1,
            .type_size                = sizeof(block_q8_1),
            .is_quantized             = true,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q8_1_ref,
        }},
        {GGML_TYPE_Q2_K, {
            .type_name                = "q2_K",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_q2_K),
            .is_quantized             = true,
            .vec_dot                  = ggml_vec_dot_q2_K_q8_K
            // .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
        }},
        {GGML_TYPE_Q3_K, {
            .type_name                = "q3_K",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_q3_K),
            .is_quantized             = true,
            .vec_dot                  = ggml_vec_dot_q3_K_q8_K
            // .to_float                 = (ggml_to_float_t) dequantize_row_q3_K,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q3_K_ref,
        }},
        {GGML_TYPE_Q4_K, {
            .type_name                = "q4_K",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_q4_K),
            .is_quantized             = true,
            .vec_dot                  = ggml_vec_dot_q4_K_q8_K
            // .to_float                 = (ggml_to_float_t) dequantize_row_q4_K,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q4_K_ref,
        }},
        {GGML_TYPE_Q5_K, {
            .type_name                = "q5_K",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_q5_K),
            .is_quantized             = true,
            // .to_float                 = (ggml_to_float_t) dequantize_row_q5_K,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q5_K_ref,
        }},
        {GGML_TYPE_Q6_K, {
            .type_name                = "q6_K",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_q6_K),
            .is_quantized             = true,
            .vec_dot                  = ggml_vec_dot_q6_K_q8_K
            // .to_float                 = (ggml_to_float_t) dequantize_row_q6_K,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q6_K_ref,
        }},
        {GGML_TYPE_IQ2_XXS, {
            .type_name                = "iq2_xxs",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_iq2_xxs),
            .is_quantized             = true,
            // .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xxs,
            // .from_float_ref           = nullptr,
        }},
        {GGML_TYPE_IQ2_XS, {
            .type_name                = "iq2_xs",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_iq2_xs),
            .is_quantized             = true,
            // .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xs,
            // .from_float_ref           = nullptr,
        }},
        {GGML_TYPE_IQ3_XXS, {
            .type_name                = "iq3_xxs",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_iq3_xxs),
            .is_quantized             = true,
            // .to_float                 = (ggml_to_float_t) dequantize_row_iq3_xxs,
            // .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_xxs_ref,
        }},
        {GGML_TYPE_IQ3_S, {
            .type_name                = "iq3_s",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_iq3_s),
            .is_quantized             = true,
            // .to_float                 = (ggml_to_float_t) dequantize_row_iq3_s,
            // .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_s_ref,
        }},
        {GGML_TYPE_IQ2_S, {
            .type_name                = "iq2_s",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_iq2_s),
            .is_quantized             = true,
            // .to_float                 = (ggml_to_float_t) dequantize_row_iq2_s,
            // .from_float_ref           = (ggml_from_float_t)quantize_row_iq2_s_ref,
        }},
        {GGML_TYPE_IQ1_S, {
            .type_name                = "iq1_s",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_iq1_s),
            .is_quantized             = true,
            // .to_float                 = (ggml_to_float_t) dequantize_row_iq1_s,
            // .from_float_ref           = nullptr,
        }},
        {GGML_TYPE_IQ1_M, {
            .type_name                = "iq1_m",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_iq1_m),
            .is_quantized             = true,
            // .to_float                 = (ggml_to_float_t) dequantize_row_iq1_m,
            // .from_float_ref           = nullptr,
        }},
        {GGML_TYPE_IQ4_NL, {
            .type_name                = "iq4_nl",
            .blck_size                = QK4_NL,
            .type_size                = sizeof(block_iq4_nl),
            .is_quantized             = true,
            // .to_float                 = (ggml_to_float_t) dequantize_row_iq4_nl,
            // .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_nl_ref,
        }},
        {GGML_TYPE_IQ4_XS, {
            .type_name                = "iq4_xs",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_iq4_xs),
            .is_quantized             = true,
            // .to_float                 = (ggml_to_float_t) dequantize_row_iq4_xs,
            // .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_xs_ref,
        }},
        {GGML_TYPE_Q8_K, {
            .type_name                = "q8_K",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_q8_K),
            .is_quantized             = true,
        }},
        {GGML_TYPE_BF16, {
            .type_name                = "bf16",
            .blck_size                = 1,
            .type_size                = sizeof(ggml_bf16_t),
            .is_quantized             = false,
            // .to_float                 = (ggml_to_float_t) ggml_bf16_to_fp32_row,
            // .from_float_ref           = (ggml_from_float_t) ggml_fp32_to_bf16_row_ref,
        }},
        {GGML_TYPE_TQ1_0, {
            .type_name                = "tq1_0",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_tq1_0),
            .is_quantized             = true,
            // .to_float                 = (ggml_to_float_t) dequantize_row_tq1_0,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_tq1_0_ref,
        }},
        {GGML_TYPE_TQ2_0, {
            .type_name                = "tq2_0",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_tq2_0),
            .is_quantized             = true,
            // .to_float                 = (ggml_to_float_t) dequantize_row_tq2_0,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_tq2_0_ref,
        }},
};

int64_t ggml_blck_size(enum ggml_type type) {
    return type_traits[type].blck_size;
}

size_t ggml_type_size(enum ggml_type type) {
    return type_traits[type].type_size;
}

size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    assert(ne % ggml_blck_size(type) == 0);
    return ggml_type_size(type)*ne/ggml_blck_size(type);
}

double ggml_type_sizef(enum ggml_type type) {
    return ((double)(type_traits[type].type_size))/type_traits[type].blck_size;
}

const char * ggml_type_name(enum ggml_type type) {
    return type < GGML_TYPE_COUNT ? type_traits[type].type_name : "NONE";
}

ggml_vec_dot_t ggml_type_vec_dot(enum ggml_type type) {
    return type < GGML_TYPE_COUNT ? type_traits[type].vec_dot : nullptr;
}

bool ggml_is_quantized(enum ggml_type type) {
    return type_traits[type].is_quantized;
}

size_t ggml_nbytes(const struct ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] <= 0) {
            return 0;
        }
    }
    size_t nbytes;
    const size_t blck_size = ggml_blck_size(tensor->type);
    if (blck_size == 1) {
        nbytes = ggml_type_size(tensor->type);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }
    else {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_size;
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    return nbytes;
}

size_t ggml_nbytes_pad(const struct ggml_tensor * tensor) {
    return GGML_PAD(ggml_nbytes(tensor), GGML_MEM_ALIGN);
}

namespace fastllm {
    GGUFBuffer::GGUFBuffer (const std::string &fileName) {
        this->fileName = fileName;
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

    void WeightImportGGUFTensor(Data* weight, ggml_tensor *tensor, std::string &fileName, uint64_t offset) {

        printf("name = %s, type = %s\n", tensor->name.c_str(), ggml_type_name(tensor->type));
        printf("before: ");
        weight->PrintShape();

        if (tensor->type == ggml_type::GGML_TYPE_Q6_K) {
            // return;
        }

        if (tensor->type == ggml_type::GGML_TYPE_F32) {
            weight->dataType = DataType::FLOAT32;    
        } else {
            weight->dataType = DataType::DATA_GGUF_FORMAT;
        }

        weight->dims = tensor->dims;
        weight->ggmlTensor = (void*)(new ggml_tensor());
        weight->ggmlType = tensor->type;

        weight->expansionBytes = ggml_nbytes(tensor);
        weight->cpuData = new uint8_t[ggml_nbytes(tensor)];
        
        (*(ggml_tensor*)weight->ggmlTensor) = *tensor;

        FILE *fi = fopen(fileName.c_str(), "rb");
#if defined(_WIN32) || defined(_WIN64)
        _fseeki64(fi, offset, 0);
#else
        fseek(fi, offset, 0);
#endif
        int ret = fread(weight->cpuData, 1, ggml_nbytes(tensor), fi);
        fclose(fi);
    }

    struct GGUFWeightReplaceRule {
        enum GGUFWeightReplaceType {
            GGUFWeightReplaceDirect = 0, // 直接替换
            GGUFWeightReplacePacked = 1 // 拆包替换，例如[128, 2048, 2048]的矩阵替换为128个2048 * 2048，常见于moe
        };

        GGUFWeightReplaceType type;
        std::regex pattern;
        std::vector <std::string> names;

        GGUFWeightReplaceRule (std::regex pattern, const std::string &name, 
                                GGUFWeightReplaceType type = GGUFWeightReplaceDirect) {
            this->type = type;
            this->pattern = pattern;
            this->names = {name};
        }

        GGUFWeightReplaceRule (std::regex pattern, const std::vector <std::string> &names, 
                                GGUFWeightReplaceType type = GGUFWeightReplaceDirect) {
            this->type = type;
            this->pattern = pattern;
            this->names = names;
        }
    };

    void ReadGGUF(basellm *model, const std::string &fileName, std::vector <ReadGGUFTask> &tasks) {
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

        std::vector <std::pair <ggml_tensor, uint64_t> > tensors; // <tensors, offset>
        tensors.resize(tensorCount);

        for (int i = 0; i < tensorCount; i++) {
            std::string tensorName = ggufBuffer.ReadString();
            uint32_t ndims = ggufBuffer.Read <uint32_t> ();
        
            for (int j = 0; j < ndims; j++) {
                int64_t dim = ggufBuffer.Read <int64_t> ();
                tensors[i].first.dims.push_back(dim);
            }

            for (int j = 0; j < GGML_MAX_DIMS; j++) {
                tensors[i].first.ne[j] = 1;
                if (j < ndims) {
                    tensors[i].first.ne[j] = tensors[i].first.dims[j];
                }
            }

            std::reverse(tensors[i].first.dims.begin(), tensors[i].first.dims.end());

            int type = ggufBuffer.Read <int> ();
            uint64_t offset = ggufBuffer.Read <uint64_t> ();

            {
                tensors[i].first.type = (ggml_type)type;
                // check that tensor type is within defined range
                /* if (tensors[i].first.type < 0 || tensors[i].first.type >= GGML_TYPE_COUNT) {
                    GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d (%s)\n",
                        __func__, tensors[i].first.name, tensors[i].first.type, ggml_type_name(tensors[i].first.type));
                    ok = false;
                    break;
                }*/ 
                const size_t  type_size = ggml_type_size(tensors[i].first.type);
                const int64_t blck_size = ggml_blck_size(tensors[i].first.type);

                // check that row size is divisible by block size
                /*if (blck_size == 0 || tensors[i].first.ne[0] % blck_size != 0) {
                    GGML_LOG_ERROR("%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                        "not a multiple of block size (%" PRId64 ")\n",
                        __func__, tensors[i].first.name, (int) tensors[i].first.type, ggml_type_name(tensors[i].first.type), tensors[i].first.ne[0], blck_size);
                    ok = false;
                    break;
                }*/

                // calculate byte offsets given the tensor shape and type
                tensors[i].first.nb[0] = type_size;
                tensors[i].first.nb[1] = tensors[i].first.nb[0] * (tensors[i].first.ne[0] / blck_size);
                for (int j = 2; j < GGML_MAX_DIMS; ++j) {
                    tensors[i].first.nb[j] = tensors[i].first.nb[j - 1] * tensors[i].first.ne[j - 1];
                }
            }

            tensors[i].first.name = tensorName;
            tensors[i].second = offset;
        }

        // we require the data section to be aligned, so take into account any padding
        if (fseek(ggufBuffer.f, GGML_PAD(ftell(ggufBuffer.f), ggufAlignment), SEEK_SET) != 0) {
            printf("alignment error\n");
            exit(0);
        }

        uint64_t baseOffset = ftell(ggufBuffer.f);

        uint64_t curPos = baseOffset;

        std::vector <GGUFWeightReplaceRule> weightNameConverterRules = {
            GGUFWeightReplaceRule ( 
                std::regex(R"(blk\.(\d+)\.attn_(q|k|v)\.(weight|bias))"),
                "model.layers.$1.self_attn.$2_proj.$3"
            ), // qkv
            GGUFWeightReplaceRule (
                std::regex(R"(blk\.(\d+)\.attn_output\.(weight|bias))"),
                "model.layers.$1.self_attn.o_proj.$2"
            ), // o 
            GGUFWeightReplaceRule (
                std::regex(R"(blk\.(\d+)\.ffn_(gate|up|down)\.(weight|bias))"),
                "model.layers.$1.mlp.$2_proj.$3"
            ), // mlp 
            GGUFWeightReplaceRule (
                std::regex(R"(blk\.(\d+)\.attn_norm\.weight)"),
                "model.layers.$1.input_layernorm.weight"
            ),
            GGUFWeightReplaceRule (
                std::regex(R"(blk\.(\d+)\.ffn_norm\.weight)"),
                "model.layers.$1.post_attention_layernorm.weight"
            ),
            /*{
                std::regex(R"(token_embd.weight)"),
                "model.embed_tokens.weight"
            },*/
            GGUFWeightReplaceRule (
                std::regex(R"(output.weight)"),
                "lm_head.weight"
            ), 
            GGUFWeightReplaceRule (
                std::regex(R"(output_norm.weight)"),
                "model.norm.weight"
            ),

            GGUFWeightReplaceRule (
                std::regex(R"(blk.(\d+).ffn_(gate|up|down)_exps.weight)"),
                std::vector <std::string> ({"model.layers.$1.mlp.experts.", ".$2_proj.weight"}),
                GGUFWeightReplaceRule::GGUFWeightReplacePacked
            ), // experts
            GGUFWeightReplaceRule (
                std::regex(R"(blk.(\d+).ffn_(gate|up|down)_shexp.weight)"),
                "model.layers.$1.mlp.shared_experts.$2_proj.weight"
            ) // shared experts
        };

        for (int i = 0; i < tensorCount; i++) {
            if (curPos != baseOffset + tensors[i].second) {
                ErrorInFastLLM("read weight " + tensors[i].first.name + " error.\n");
                exit(0);
            } 

            std::string name = tensors[i].first.name;
            bool matched = false;

            for (auto &it : weightNameConverterRules) {
                if (std::regex_search(name, it.pattern)) {
                    matched = true;

                    if (it.type == GGUFWeightReplaceRule::GGUFWeightReplaceDirect) {
                        name = std::regex_replace(name, it.pattern, it.names[0]);
                        if (model->weight.weight.find(name) != model->weight.weight.end()) {
                            tasks.push_back (
                                ReadGGUFTask (
                                    name, &model->weight.weight[name], tensors[i].first, ggufBuffer.fileName, baseOffset + tensors[i].second
                                )
                            );
                            // printf("replace %s\n", name.c_str());
                        }
                    } else if (it.type == GGUFWeightReplaceRule::GGUFWeightReplacePacked) {
                        std::string prefix = std::regex_replace(name, it.pattern, it.names[0]);
                        std::string suffix = std::regex_replace(name, it.pattern, it.names[1]);

                        int packedBatch = tensors[i].first.ne[2];
                        ggml_tensor singleTensor = tensors[i].first;
                        singleTensor.dims.erase(singleTensor.dims.begin());
                        singleTensor.ne[2] = 1;
                        singleTensor.nb[2] = singleTensor.nb[3] = singleTensor.nb[1];

                        for (int idx = 0; idx < packedBatch; idx++) {
                            std::string modelName = prefix + std::to_string(idx) + suffix;
                            if (model->weight.weight.find(modelName) != model->weight.weight.end()) {
                                tasks.push_back (
                                    ReadGGUFTask (
                                        modelName, &model->weight.weight[modelName], singleTensor, 
                                        ggufBuffer.fileName, baseOffset + tensors[i].second + idx * ggml_nbytes(&singleTensor)
                                    )
                                );
                            }
                        }
/*
                        printf("name = %s\n", name.c_str());
                        printf("prefix = %s\n", prefix.c_str());
                        printf("suffix = %s\n", suffix.c_str());
                        printf("nbytes = %d\n", ggml_nbytes(&tensors[i].first));

                        for (int j = 0; j < GGML_MAX_DIMS; j++) {
                            printf("i = %d, ne = %d\n", j, tensors[i].first.ne[j]);
                        }
*/
                    }
                }
            } 

            if (!matched) {
                printf("unmatched weight %s (", name.c_str());
                for (auto it : tensors[i].first.dims) {
                    printf("%d ", it);
                }
                printf(") type = %s\n", ggml_type_name(tensors[i].first.type));
            }

            curPos += GGML_PAD(ggml_nbytes(&tensors[i].first), ggufAlignment);
        }
        
        // exit(0);
    }
}