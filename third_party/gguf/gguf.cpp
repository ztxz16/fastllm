#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <map>
#include <stdexcept>

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

GGML_API void        ggml_bf16_to_fp32_row(const ggml_bf16_t *bf16, float *fp32, int64_t len) {
    for (int i = 0; i < len; i++) {
        uint32_t x = ((int)bf16[i].bits << 16);
        fp32[i] = *((float*)&x);
    }
}

#if (defined(_MSC_VER) && _MSC_VER <= 1922) || (defined(__GNUC__) && __GNUC__ < 8 && !defined(__clang__))  // VS 2015/2017
std::map <ggml_type, ggml_type_traits> type_traits = {
        {GGML_TYPE_I8, {/* type_name */"i8", /* blck_size */1,
            /* type_size */ sizeof(int8_t),/* is_quantized */  false,
        }},
        {GGML_TYPE_I16, {/* type_name */"i16", /* blck_size */1,
            /* type_size */ sizeof(int16_t),/* is_quantized */  false,
        }},
        {GGML_TYPE_I32, {/* type_name */"i32", /* blck_size */1,
            /* type_size */ sizeof(int32_t),/* is_quantized */  false,
        }},
        {GGML_TYPE_I64, {/* type_name */"i64", /* blck_size */1,
            /* type_size */ sizeof(int64_t),/* is_quantized */  false,
        }},
        {GGML_TYPE_F64, {/* type_name */"f64", /* blck_size */1,
            /* type_size */ sizeof(double),/* is_quantized */  false,
        }},
        {GGML_TYPE_F32, {/* type_name */"f32", /* blck_size */1,
            /* type_size */ sizeof(float),/* is_quantized */  false,
        }},
        {GGML_TYPE_F16, ggml_type_traits{/* type_name */"f16", /* blck_size */1,
            /* type_size */ sizeof(ggml_fp16_t),/* is_quantized */  false,
            nullptr, GGML_TYPE_F32,
            /* to_float */ (ggml_to_float_t) ggml_fp16_to_fp32_row,
            // .from_float_ref           = (ggml_from_float_t) ggml_fp32_to_fp16_row,
        }},
        {GGML_TYPE_Q4_0, ggml_type_traits{/* type_name */"q4_0", /* blck_size */QK4_0,
            /* type_size */ sizeof(block_q4_0),/* is_quantized */  true,
            nullptr, GGML_TYPE_F32,
            /* to_float */ (ggml_to_float_t) dequantize_row_q4_0,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q4_0_ref,
        }},
        {GGML_TYPE_Q4_1, ggml_type_traits{/* type_name */"q4_1", /* blck_size */QK4_1,
            /* type_size */ sizeof(block_q4_1),/* is_quantized */  true,
            nullptr, GGML_TYPE_F32,
            /* to_float */ (ggml_to_float_t) dequantize_row_q4_1,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q4_1_ref,
        }},
        {GGML_TYPE_Q5_0, ggml_type_traits{/* type_name */"q5_0", /* blck_size */QK5_0,
            /* type_size */ sizeof(block_q5_0),/* is_quantized */  true,
            nullptr, GGML_TYPE_F32,
            /* to_float */ (ggml_to_float_t) dequantize_row_q5_0,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q5_0_ref,
        }},
        {GGML_TYPE_Q5_1, ggml_type_traits{/* type_name */"q5_1", /* blck_size */QK5_1,
            /* type_size */ sizeof(block_q5_1),/* is_quantized */  true,
            nullptr, GGML_TYPE_F32,
            /* to_float */ (ggml_to_float_t) dequantize_row_q5_1,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q5_1_ref,
        }},
        {GGML_TYPE_Q8_0, ggml_type_traits{/* type_name */"q8_0", /* blck_size */QK8_0,
            /* type_size */ sizeof(block_q8_0),/* is_quantized */  true,
            nullptr, GGML_TYPE_F32,
            /* to_float */ (ggml_to_float_t) dequantize_row_q8_0,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q8_0_ref,
        }},
        {GGML_TYPE_Q8_1, {/* type_name */"q8_1", /* blck_size */QK8_1,
            /* type_size */ sizeof(block_q8_1),/* is_quantized */  true,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q8_1_ref,
        }},
        {GGML_TYPE_MXFP4, ggml_type_traits{/* type_name */"mxfp4", /* blck_size */QK_MXFP4,
            /* type_size */ sizeof(block_mxfp4),/* is_quantized */  true,
            /* vec_dot */ nullptr,
            /* vec_dot_type */ GGML_TYPE_Q8_K,
            /* to_float */ (ggml_to_float_t) dequantize_row_mxfp4,
        }},
        {GGML_TYPE_Q2_K, ggml_type_traits{/* type_name */"q2_K", /* blck_size */QK_K,
            /* type_size */ sizeof(block_q2_K),/* is_quantized */  true,
            /* vec_dot */ ggml_vec_dot_q2_K_q8_K,
            /* vec_dot_type */ GGML_TYPE_Q8_K,
            /* to_float */ (ggml_to_float_t) dequantize_row_q2_K,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
        }},
        {GGML_TYPE_Q2_K_R4, {/* type_name */"q2_k_r4", /* blck_size */QK_K,
            /* type_size */ sizeof(block_q2_K),/* is_quantized */  true,
            /* vec_dot */ nullptr,
            /* vec_dot_type */ GGML_TYPE_Q8_K
            // .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
        }},
        {GGML_TYPE_Q3_K, ggml_type_traits{/* type_name */"q3_K", /* blck_size */QK_K,
            /* type_size */ sizeof(block_q3_K),/* is_quantized */  true,
            /* vec_dot */ ggml_vec_dot_q3_K_q8_K,
            /* vec_dot_type */ GGML_TYPE_Q8_K,
            /* to_float */ (ggml_to_float_t) dequantize_row_q3_K,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q3_K_ref,
        }},
        {GGML_TYPE_Q3_K_R4, {/* type_name */"q3_k_r4", /* blck_size */QK_K,
            /* type_size */ sizeof(block_q3_K),/* is_quantized */  true,
            /* vec_dot */ nullptr,
            /* vec_dot_type */ GGML_TYPE_Q8_K
            // .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
        }},
        {GGML_TYPE_Q4_K, ggml_type_traits{/* type_name */"q4_K", /* blck_size */QK_K,
            /* type_size */ sizeof(block_q4_K),/* is_quantized */  true,
            /* vec_dot */ ggml_vec_dot_q4_K_q8_K,
            /* vec_dot_type */ GGML_TYPE_Q8_K,
            /* to_float */ (ggml_to_float_t) dequantize_row_q4_K,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q4_K_ref,
        }},
        {GGML_TYPE_Q4_K_R4, {/* type_name */"q4_k_r4", /* blck_size */QK_K,
            /* type_size */ sizeof(block_q4_K),/* is_quantized */  true,
            /* vec_dot */ nullptr,
            /* vec_dot_type */ GGML_TYPE_Q8_K32
            // .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
        }},
        {GGML_TYPE_Q5_K, ggml_type_traits{/* type_name */"q5_K", /* blck_size */QK_K,
            /* type_size */ sizeof(block_q5_K),/* is_quantized */  true,
            /* vec_dot */ ggml_vec_dot_q5_K_q8_K,
            /* vec_dot_type */ GGML_TYPE_Q8_K,
            /* to_float */ (ggml_to_float_t) dequantize_row_q5_K,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q5_K_ref,
        }},
        {GGML_TYPE_Q5_K_R4, {/* type_name */"q5_k_r4", /* blck_size */QK_K,
            /* type_size */ sizeof(block_q5_K),/* is_quantized */  true,
            /* vec_dot */ nullptr,
            /* vec_dot_type */ GGML_TYPE_Q8_K32
            // .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
        }},
        {GGML_TYPE_Q6_K, ggml_type_traits{/* type_name */"q6_K", /* blck_size */QK_K,
            /* type_size */ sizeof(block_q6_K),/* is_quantized */  true,
            /* vec_dot */ ggml_vec_dot_q6_K_q8_K,
            /* vec_dot_type */ GGML_TYPE_Q8_K,
            /* to_float */ (ggml_to_float_t) dequantize_row_q6_K,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q6_K_ref,
        }},
        {GGML_TYPE_Q6_K_R4, {/* type_name */"q6_k_r4", /* blck_size */QK_K,
            /* type_size */ sizeof(block_q6_K),/* is_quantized */  true,
            /* vec_dot */ nullptr,
            /* vec_dot_type */ GGML_TYPE_Q8_K
            // .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
        }},
        {GGML_TYPE_IQ2_XXS, ggml_type_traits{/* type_name */"iq2_xxs", /* blck_size */QK_K,
            /* type_size */ sizeof(block_iq2_xxs),/* is_quantized */  true,
            /* vec_dot */ ggml_vec_dot_iq2_xxs_q8_K,
            /* vec_dot_type */ GGML_TYPE_Q8_K,
            /* to_float */ (ggml_to_float_t) dequantize_row_iq2_xxs,
            // .from_float_ref           = nullptr,
        }},
        {GGML_TYPE_IQ2_XXS_R4, {/* type_name */"iq2_xxs_r4", /* blck_size */QK_K,
            /* type_size */ sizeof(block_iq2_xxs),/* is_quantized */  true,
            /* vec_dot */ nullptr,
            /* vec_dot_type */ GGML_TYPE_Q8_K
            // .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
        }},
        {GGML_TYPE_IQ2_XS, ggml_type_traits{/* type_name */"iq2_xs", /* blck_size */QK_K,
            /* type_size */ sizeof(block_iq2_xs),/* is_quantized */  true,
            /* vec_dot */ nullptr,
            /* vec_dot_type */ GGML_TYPE_Q8_K,
            /* to_float */ (ggml_to_float_t) dequantize_row_iq2_xs,
            // .from_float_ref           = nullptr,
        }},
        {GGML_TYPE_IQ3_XXS, ggml_type_traits{/* type_name */"iq3_xxs", /* blck_size */QK_K,
            /* type_size */ sizeof(block_iq3_xxs),/* is_quantized */  true,
            /* vec_dot */ nullptr,
            /* vec_dot_type */ GGML_TYPE_Q8_K,
            /* to_float */ (ggml_to_float_t) dequantize_row_iq3_xxs,
            // .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_xxs_ref,
        }},
        {GGML_TYPE_IQ3_S, {/* type_name */"iq3_s", /* blck_size */QK_K,
            /* type_size */ sizeof(block_iq3_s),/* is_quantized */  true,
            // .to_float                 = (ggml_to_float_t) dequantize_row_iq3_s,
            // .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_s_ref,
        }},
        {GGML_TYPE_IQ2_S, {/* type_name */"iq2_s", /* blck_size */QK_K,
            /* type_size */ sizeof(block_iq2_s),/* is_quantized */  true,
            // .to_float                 = (ggml_to_float_t) dequantize_row_iq2_s,
            // .from_float_ref           = (ggml_from_float_t)quantize_row_iq2_s_ref,
        }},
        {GGML_TYPE_IQ1_S, {/* type_name */"iq1_s", /* blck_size */QK_K,
            /* type_size */ sizeof(block_iq1_s),/* is_quantized */  true,
            // .to_float                 = (ggml_to_float_t) dequantize_row_iq1_s,
            // .from_float_ref           = nullptr,
        }},
        {GGML_TYPE_IQ1_M, {/* type_name */"iq1_m", /* blck_size */QK_K,
            /* type_size */ sizeof(block_iq1_m),/* is_quantized */  true,
            // .to_float                 = (ggml_to_float_t) dequantize_row_iq1_m,
            // .from_float_ref           = nullptr,
        }},
        {GGML_TYPE_IQ4_NL, {/* type_name */"iq4_nl", /* blck_size */QK4_NL,
            /* type_size */ sizeof(block_iq4_nl),/* is_quantized */  true,
            // .to_float                 = (ggml_to_float_t) dequantize_row_iq4_nl,
            // .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_nl_ref,
        }},
        {GGML_TYPE_IQ4_XS, {/* type_name */"iq4_xs", /* blck_size */QK_K,
            /* type_size */ sizeof(block_iq4_xs),/* is_quantized */  true,
            /* to_float */ (ggml_to_float_t) dequantize_row_iq4_xs,
            // .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_xs_ref,
        }},
        {GGML_TYPE_Q8_K, {/* type_name */"q8_K", /* blck_size */QK_K,
            /* type_size */ sizeof(block_q8_K),/* is_quantized */  true,
        }},
        {GGML_TYPE_BF16, {/* type_name */"bf16", /* blck_size */1,
            /* type_size */ sizeof(ggml_bf16_t),/* is_quantized */  false,
            // .to_float                 = (ggml_to_float_t) ggml_bf16_to_fp32_row,
            // .from_float_ref           = (ggml_from_float_t) ggml_fp32_to_bf16_row_ref,
        }},
        {GGML_TYPE_TQ1_0, {/* type_name */"tq1_0", /* blck_size */QK_K,
            /* type_size */ sizeof(block_tq1_0),/* is_quantized */  true,
            // .to_float                 = (ggml_to_float_t) dequantize_row_tq1_0,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_tq1_0_ref,
        }},
        {GGML_TYPE_TQ2_0, {/* type_name */"tq2_0", /* blck_size */QK_K,
            /* type_size */ sizeof(block_tq2_0),/* is_quantized */  true,
            // .to_float                 = (ggml_to_float_t) dequantize_row_tq2_0,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_tq2_0_ref,
        }},
};
#else
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
        {GGML_TYPE_F16, ggml_type_traits{
            .type_name                = "f16",
            .blck_size                = 1,
            .type_size                = sizeof(ggml_fp16_t),
            .is_quantized             = false,
            .to_float                 = (ggml_to_float_t) ggml_fp16_to_fp32_row,
            // .from_float_ref           = (ggml_from_float_t) ggml_fp32_to_fp16_row,
        }},
        {GGML_TYPE_Q4_0, ggml_type_traits{
            .type_name                = "q4_0",
            .blck_size                = QK4_0,
            .type_size                = sizeof(block_q4_0),
            .is_quantized             = true,
            .to_float                 = (ggml_to_float_t) dequantize_row_q4_0,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q4_0_ref,
        }},
        {GGML_TYPE_Q4_1, ggml_type_traits{
            .type_name                = "q4_1",
            .blck_size                = QK4_1,
            .type_size                = sizeof(block_q4_1),
            .is_quantized             = true,
            .to_float                 = (ggml_to_float_t) dequantize_row_q4_1,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q4_1_ref,
        }},
        {GGML_TYPE_Q5_0, ggml_type_traits{
            .type_name                = "q5_0",
            .blck_size                = QK5_0,
            .type_size                = sizeof(block_q5_0),
            .is_quantized             = true,
            .vec_dot                  = ggml_vec_dot_q5_0_q8_0,
            .vec_dot_type             = GGML_TYPE_Q8_0,
            .to_float                 = (ggml_to_float_t) dequantize_row_q5_0,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q5_0_ref,
        }},
        {GGML_TYPE_Q5_1, ggml_type_traits{
            .type_name                = "q5_1",
            .blck_size                = QK5_1,
            .type_size                = sizeof(block_q5_1),
            .is_quantized             = true,
            .vec_dot                  = ggml_vec_dot_q5_1_q8_1,
            .vec_dot_type             = GGML_TYPE_Q8_1,
            .to_float                 = (ggml_to_float_t) dequantize_row_q5_1,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q5_1_ref,
        }},
        {GGML_TYPE_Q8_0, ggml_type_traits{
            .type_name                = "q8_0",
            .blck_size                = QK8_0,
            .type_size                = sizeof(block_q8_0),
            .is_quantized             = true,
            .vec_dot                  = ggml_vec_dot_q8_0_q8_0,
            .vec_dot_type             = GGML_TYPE_Q8_0,
            .to_float                 = (ggml_to_float_t) dequantize_row_q8_0,
            .from_float_ref           = (ggml_from_float_t) quantize_row_q8_0_ref,
        }},
        {GGML_TYPE_Q8_1, {
            .type_name                = "q8_1",
            .blck_size                = QK8_1,
            .type_size                = sizeof(block_q8_1),
            .is_quantized             = true,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q8_1_ref,
        }},
        {GGML_TYPE_MXFP4, {
            .type_name                = "mxfp4",
            .blck_size                = QK_MXFP4,
            .type_size                = sizeof(block_mxfp4),
            .is_quantized             = true,
            .to_float                 = (ggml_to_float_t) dequantize_row_mxfp4,
        }},
        {GGML_TYPE_Q2_K, ggml_type_traits{
            .type_name                = "q2_K",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_q2_K),
            .is_quantized             = true,
            .vec_dot                  = ggml_vec_dot_q2_K_q8_K,
            .vec_dot_type             = GGML_TYPE_Q8_K,
            .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
            .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
        }},
        {GGML_TYPE_Q2_K_R4, {
            .type_name                = "q2_k_r4",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_q2_K),
            .is_quantized             = true,
            .vec_dot                  = nullptr,
            .vec_dot_type             = GGML_TYPE_Q8_K
            // .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
        }},
        {GGML_TYPE_Q3_K, ggml_type_traits{
            .type_name                = "q3_K",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_q3_K),
            .is_quantized             = true,
            .vec_dot                  = ggml_vec_dot_q3_K_q8_K,
            .vec_dot_type             = GGML_TYPE_Q8_K,
            .to_float                 = (ggml_to_float_t) dequantize_row_q3_K,
            .from_float_ref           = (ggml_from_float_t) quantize_row_q3_K_ref,
        }},
        {GGML_TYPE_Q3_K_R4, {
            .type_name                = "q3_k_r4",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_q3_K),
            .is_quantized             = true,
            .vec_dot                  = nullptr,
            .vec_dot_type             = GGML_TYPE_Q8_K
            // .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
        }},
        {GGML_TYPE_Q4_K, ggml_type_traits{
            .type_name                = "q4_K",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_q4_K),
            .is_quantized             = true,
            .vec_dot                  = ggml_vec_dot_q4_K_q8_K,
            .vec_dot_type             = GGML_TYPE_Q8_K,
            .to_float                 = (ggml_to_float_t) dequantize_row_q4_K,
            .from_float_ref           = (ggml_from_float_t) quantize_row_q4_K_ref,
        }},
        {GGML_TYPE_Q4_K_R4, {
            .type_name                = "q4_k_r4",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_q4_K),
            .is_quantized             = true,
            .vec_dot                  = nullptr,
            .vec_dot_type             = GGML_TYPE_Q8_K32
            // .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
        }},
        {GGML_TYPE_Q5_K, ggml_type_traits{
            .type_name                = "q5_K",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_q5_K),
            .is_quantized             = true,
            .vec_dot                  = ggml_vec_dot_q5_K_q8_K,
            .vec_dot_type             = GGML_TYPE_Q8_K,
            .to_float                 = (ggml_to_float_t) dequantize_row_q5_K,
            .from_float_ref           = (ggml_from_float_t) quantize_row_q5_K_ref,
        }},
        {GGML_TYPE_Q5_K_R4, {
            .type_name                = "q5_k_r4",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_q5_K),
            .is_quantized             = true,
            .vec_dot                  = nullptr,
            .vec_dot_type             = GGML_TYPE_Q8_K32
            // .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
        }},
        {GGML_TYPE_Q6_K, ggml_type_traits{
            .type_name                = "q6_K",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_q6_K),
            .is_quantized             = true,
            .vec_dot                  = ggml_vec_dot_q6_K_q8_K,
            .vec_dot_type             = GGML_TYPE_Q8_K,
            .to_float                 = (ggml_to_float_t) dequantize_row_q6_K,
            .from_float_ref           = (ggml_from_float_t) quantize_row_q6_K_ref,
        }},
        {GGML_TYPE_Q6_K_R4, {
            .type_name                = "q6_k_r4",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_q6_K),
            .is_quantized             = true,
            .vec_dot                  = nullptr,
            .vec_dot_type             = GGML_TYPE_Q8_K
            // .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
        }},
        {GGML_TYPE_IQ2_XXS, ggml_type_traits{
            .type_name                = "iq2_xxs",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_iq2_xxs),
            .is_quantized             = true,
            .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
            .vec_dot_type             = GGML_TYPE_Q8_K,
            .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xxs,
            // .from_float_ref           = nullptr,
        }},
        {GGML_TYPE_IQ2_XXS_R4, {
            .type_name                = "iq2_xxs_r4",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_iq2_xxs),
            .is_quantized             = true,
            .vec_dot                  = nullptr,
            .vec_dot_type             = GGML_TYPE_Q8_K,
            // .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
        }},
        {GGML_TYPE_IQ2_XS, {
            .type_name                = "iq2_xs",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_iq2_xs),
            .is_quantized             = true,
            .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xs,
            // .from_float_ref           = nullptr,
        }},
        {GGML_TYPE_IQ2_XS_R4, {
            .type_name                = "iq2_xs_r4",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_iq2_xs),
            .is_quantized             = true,
            .vec_dot                  = nullptr,
            .vec_dot_type             = GGML_TYPE_Q8_K,
            // .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
            // .from_float_ref           = (ggml_from_float_t) quantize_row_q2_K_ref,
        }},
        {GGML_TYPE_IQ3_XXS, {
            .type_name                = "iq3_xxs",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_iq3_xxs),
            .is_quantized             = true,
            .to_float                 = (ggml_to_float_t) dequantize_row_iq3_xxs,
            // .from_float_ref           = (ggml_from_float_t)quantize_row_iq3_xxs_ref,
        }},
        {GGML_TYPE_IQ3_XXS_R4, {
            .type_name                = "iq3_xxs_r4",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_iq3_xxs),
            .is_quantized             = true,
            .vec_dot                  = nullptr,
            .vec_dot_type             = GGML_TYPE_Q8_K,
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
            .vec_dot                  = nullptr,
            .vec_dot_type             = GGML_TYPE_Q8_K,
            // .to_float                 = (ggml_to_float_t) dequantize_row_iq2_s,
            // .from_float_ref           = (ggml_from_float_t)quantize_row_iq2_s_ref,
        }},
        {GGML_TYPE_IQ2_S_R4, {
            .type_name                = "iq2_s_r4",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_iq2_s),
            .is_quantized             = true,
            .vec_dot                  = nullptr,
            .vec_dot_type             = GGML_TYPE_Q8_K,
            // .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xs,
            // .from_float_ref           = nullptr,
        }},
        {GGML_TYPE_IQ1_S, {
            .type_name                = "iq1_s",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_iq1_s),
            .is_quantized             = true,
            .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
            .vec_dot_type             = GGML_TYPE_Q8_K,
            // .to_float                 = (ggml_to_float_t) dequantize_row_iq1_s,
            // .from_float_ref           = nullptr,
        }},
        {GGML_TYPE_IQ1_M, {
            .type_name                = "iq1_m",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_iq1_m),
            .is_quantized             = true,
            .vec_dot                  = ggml_vec_dot_iq1_m_q8_K,
            .vec_dot_type             = GGML_TYPE_Q8_K,
            // .to_float                 = (ggml_to_float_t) dequantize_row_iq1_m,
            // .from_float_ref           = nullptr,
        }},
        {GGML_TYPE_IQ4_NL, {
            .type_name                = "iq4_nl",
            .blck_size                = QK4_NL,
            .type_size                = sizeof(block_iq4_nl),
            .is_quantized             = true,
            .vec_dot                  = ggml_vec_dot_iq4_nl_q8_0,
            .vec_dot_type             = GGML_TYPE_Q8_0,
            // .to_float                 = (ggml_to_float_t) dequantize_row_iq4_nl,
            // .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_nl_ref,
        }},
        {GGML_TYPE_IQ4_XS, {
            .type_name                = "iq4_xs",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_iq4_xs),
            .is_quantized             = true,
            .to_float                 = (ggml_to_float_t) dequantize_row_iq4_xs,
            // .from_float_ref           = (ggml_from_float_t)quantize_row_iq4_xs_ref,
        }},
        {GGML_TYPE_Q8_K, {
            .type_name                = "q8_K",
            .blck_size                = QK_K,
            .type_size                = sizeof(block_q8_K),
            .is_quantized             = true,
        }},
        {GGML_TYPE_Q8_K32, {
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
            .to_float                 = (ggml_to_float_t) ggml_bf16_to_fp32_row,
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
#endif

namespace {
    const ggml_type_traits *FindGGMLTypeTraits(enum ggml_type type) {
        const auto it = type_traits.find(type);
        if (it == type_traits.end() || it->second.blck_size <= 0 ||
            it->second.type_size == 0) {
            return nullptr;
        }
        return &it->second;
    }

    std::string SanitizeGGUFTensorName(const std::string &name) {
        std::string result;
        result.reserve(std::min<size_t>(name.size(), 160));
        for (unsigned char c : name) {
            if (result.size() == 160) {
                result += "...";
                break;
            }
            result += std::isalnum(c) || c == '.' || c == '_' || c == '-'
                ? (char)c
                : '?';
        }
        return result.empty() ? "<unnamed>" : result;
    }

    size_t CheckedSizeAdd(size_t left, size_t right,
                          const std::string &context) {
        if (right > std::numeric_limits<size_t>::max() - left) {
            throw std::runtime_error(context + " size addition overflow.");
        }
        return left + right;
    }

    size_t CheckedSizeMultiply(size_t left, size_t right,
                               const std::string &context) {
        if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
            throw std::runtime_error(context + " size multiplication overflow.");
        }
        return left * right;
    }

    size_t CheckedSizePad(size_t value, size_t alignment,
                          const std::string &context) {
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
            throw std::runtime_error(context + " has invalid alignment " +
                                     std::to_string(alignment) + ".");
        }
        return CheckedSizeAdd(value, alignment - 1, context) & ~(alignment - 1);
    }

    size_t CheckedUint64ToSize(uint64_t value,
                               const std::string &context) {
        if (value > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error(context + " exceeds host size_t.");
        }
        return (size_t)value;
    }
}

int64_t ggml_blck_size(enum ggml_type type) {
    const auto *traits = FindGGMLTypeTraits(type);
    return traits == nullptr ? 0 : traits->blck_size;
}

size_t ggml_type_size(enum ggml_type type) {
    const auto *traits = FindGGMLTypeTraits(type);
    return traits == nullptr ? 0 : traits->type_size;
}

size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    const int64_t blockSize = ggml_blck_size(type);
    const size_t typeSize = ggml_type_size(type);
    if (blockSize <= 0 || typeSize == 0) {
        throw std::runtime_error("Unsupported GGML type " +
                                 std::to_string((int)type) + ".");
    }
    if (ne < 0 || ne % blockSize != 0) {
        throw std::runtime_error("GGML row length is not divisible by block size.");
    }
    return typeSize * ne / blockSize;
}

double ggml_type_sizef(enum ggml_type type) {
    const auto *traits = FindGGMLTypeTraits(type);
    return traits == nullptr
        ? 0.0
        : (double)traits->type_size / traits->blck_size;
}

const char * ggml_type_name(enum ggml_type type) {
    const auto *traits = FindGGMLTypeTraits(type);
    return traits == nullptr ? "UNSUPPORTED" : traits->type_name;
}

ggml_from_float_t ggml_type_from_float_ref(enum ggml_type type) {
    const auto *traits = FindGGMLTypeTraits(type);
    return traits == nullptr ? nullptr : traits->from_float_ref;
}

ggml_to_float_t ggml_type_to_float(enum ggml_type type) {
    const auto *traits = FindGGMLTypeTraits(type);
    return traits == nullptr ? nullptr : traits->to_float;
}

ggml_vec_dot_t ggml_type_vec_dot(enum ggml_type type) {
    const auto *traits = FindGGMLTypeTraits(type);
    return traits == nullptr ? nullptr : traits->vec_dot;
}

ggml_type ggml_type_vec_dot_type(enum ggml_type type) {
    const auto *traits = FindGGMLTypeTraits(type);
    return traits == nullptr ? GGML_TYPE_Q8_K : traits->vec_dot_type;
}

bool ggml_is_quantized(enum ggml_type type) {
    const auto *traits = FindGGMLTypeTraits(type);
    return traits != nullptr && traits->is_quantized;
}

size_t ggml_nbytes(const struct ggml_tensor * tensor) {
    const std::string context =
        "GGUF tensor '" + SanitizeGGUFTensorName(tensor->name) + "'";
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] <= 0) {
            throw std::runtime_error(context + " has a non-positive dimension.");
        }
        if ((uint64_t)tensor->ne[i] > std::numeric_limits<size_t>::max()) {
            throw std::runtime_error(context + " dimension exceeds host size_t.");
        }
    }

    const size_t blck_size = ggml_blck_size(tensor->type);
    const size_t type_size = ggml_type_size(tensor->type);
    if (blck_size == 0 || type_size == 0) {
        throw std::runtime_error(
            context + " uses unsupported GGML type " +
            std::to_string((int)tensor->type) + ".");
    }
    if ((size_t)tensor->ne[0] % blck_size != 0) {
        throw std::runtime_error(context +
                                 " row length is not divisible by block size.");
    }

    size_t nbytes = CheckedSizeMultiply(
        (size_t)tensor->ne[0] / blck_size, type_size, context);
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        const size_t span = CheckedSizeMultiply(
            (size_t)(tensor->ne[i] - 1), tensor->nb[i], context);
        nbytes = CheckedSizeAdd(nbytes, span, context);
    }
    return nbytes;
}

size_t ggml_nbytes_pad(const struct ggml_tensor * tensor) {
    return CheckedSizePad(ggml_nbytes(tensor), GGML_MEM_ALIGN,
                          "GGUF padded tensor size");
}

namespace fastllm {
    GGUFBuffer::GGUFBuffer (const std::string &fileName) {
        this->fileName = fileName;
        this->f = fopen(fileName.c_str(), "rb");
        if (this->f == nullptr) {
            ErrorInFastLLM("Unable to open GGUF file.\n");
        }
    }

    GGUFBuffer::~GGUFBuffer () {
        if (this->f != nullptr) {
            fclose(this->f);
        }
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
        return Read<uint8_t>() != 0;
    }

    std::string GGUFBuffer::ReadString() {
        uint64_t len = Read<uint64_t>();
        if (len > std::numeric_limits<size_t>::max()) {
            ErrorInFastLLM("GGUFBuffer.ReadString length overflow.\n");
        }
        std::string s((size_t)len, '\0');
        if (len > 0) {
            ReadBytes(reinterpret_cast<uint8_t*>(&s[0]), len);
        }
        return s;
    }

    void GGUFBuffer::ReadBytes(uint8_t *buffer, uint64_t bytes) {
        if (fread(buffer, 1, bytes, f) != bytes) {
            ErrorInFastLLM("GGUFBuffer.ReadBytes error.\n");
        }
    }

    void GGUFBuffer::SkipBytes(uint64_t bytes) {
        std::array<uint8_t, 8192> scratch;
        while (bytes > 0) {
            const size_t chunk = (size_t)std::min<uint64_t>(bytes, scratch.size());
            if (fread(scratch.data(), 1, chunk, f) != chunk) {
                ErrorInFastLLM("GGUFBuffer.SkipBytes error.\n");
            }
            bytes -= chunk;
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
    template double GGUFBuffer::Read<double>();

    namespace {
        json11::Json ReadGGUFValue(GGUFBuffer &buffer, gguf_type type) {
            switch (type) {
                case GGUF_TYPE_UINT8:
                    return (int)buffer.Read<uint8_t>();
                case GGUF_TYPE_INT8:
                    return (int)buffer.Read<int8_t>();
                case GGUF_TYPE_UINT16:
                    return (int)buffer.Read<uint16_t>();
                case GGUF_TYPE_INT16:
                    return (int)buffer.Read<int16_t>();
                case GGUF_TYPE_UINT32:
                    return (double)buffer.Read<uint32_t>();
                case GGUF_TYPE_INT32:
                    return (int)buffer.Read<int32_t>();
                case GGUF_TYPE_FLOAT32:
                    return (double)buffer.Read<float>();
                case GGUF_TYPE_BOOL:
                    return buffer.ReadBool();
                case GGUF_TYPE_STRING:
                    return buffer.ReadString();
                case GGUF_TYPE_ARRAY: {
                    const uint32_t elementTypeValue = buffer.Read<uint32_t>();
                    if (elementTypeValue >= GGUF_TYPE_COUNT ||
                        elementTypeValue == GGUF_TYPE_ARRAY) {
                        ErrorInFastLLM("GGUF array has invalid or nested element type " +
                                       std::to_string(elementTypeValue) + ".\n");
                    }
                    const uint64_t count = buffer.Read<uint64_t>();
                    if (count > std::numeric_limits<size_t>::max()) {
                        ErrorInFastLLM("GGUF array length overflow.\n");
                    }
                    json11::Json::array values;
                    values.reserve((size_t)count);
                    for (uint64_t i = 0; i < count; ++i) {
                        values.push_back(ReadGGUFValue(
                            buffer, (gguf_type)elementTypeValue));
                    }
                    return values;
                }
                case GGUF_TYPE_UINT64:
                    return (double)buffer.Read<uint64_t>();
                case GGUF_TYPE_INT64:
                    return (double)buffer.Read<int64_t>();
                case GGUF_TYPE_FLOAT64:
                    return buffer.Read<double>();
                default:
                    ErrorInFastLLM("Unsupported GGUF metadata type " +
                                   std::to_string((int)type) + ".\n");
            }
            return json11::Json();
        }

        void SkipGGUFValue(GGUFBuffer &buffer, gguf_type type) {
            switch (type) {
                case GGUF_TYPE_UINT8:
                case GGUF_TYPE_INT8:
                case GGUF_TYPE_BOOL:
                    buffer.SkipBytes(1);
                    return;
                case GGUF_TYPE_UINT16:
                case GGUF_TYPE_INT16:
                    buffer.SkipBytes(2);
                    return;
                case GGUF_TYPE_UINT32:
                case GGUF_TYPE_INT32:
                case GGUF_TYPE_FLOAT32:
                    buffer.SkipBytes(4);
                    return;
                case GGUF_TYPE_UINT64:
                case GGUF_TYPE_INT64:
                case GGUF_TYPE_FLOAT64:
                    buffer.SkipBytes(8);
                    return;
                case GGUF_TYPE_STRING:
                    buffer.SkipBytes(buffer.Read<uint64_t>());
                    return;
                case GGUF_TYPE_ARRAY: {
                    const uint32_t elementTypeValue = buffer.Read<uint32_t>();
                    if (elementTypeValue >= GGUF_TYPE_COUNT ||
                        elementTypeValue == GGUF_TYPE_ARRAY) {
                        ErrorInFastLLM("GGUF array has invalid or nested element type " +
                                       std::to_string(elementTypeValue) + ".\n");
                    }
                    const uint64_t count = buffer.Read<uint64_t>();
                    uint64_t fixedSize = 0;
                    switch ((gguf_type)elementTypeValue) {
                        case GGUF_TYPE_UINT8:
                        case GGUF_TYPE_INT8:
                        case GGUF_TYPE_BOOL:
                            fixedSize = 1;
                            break;
                        case GGUF_TYPE_UINT16:
                        case GGUF_TYPE_INT16:
                            fixedSize = 2;
                            break;
                        case GGUF_TYPE_UINT32:
                        case GGUF_TYPE_INT32:
                        case GGUF_TYPE_FLOAT32:
                            fixedSize = 4;
                            break;
                        case GGUF_TYPE_UINT64:
                        case GGUF_TYPE_INT64:
                        case GGUF_TYPE_FLOAT64:
                            fixedSize = 8;
                            break;
                        default:
                            break;
                    }
                    if (fixedSize != 0) {
                        if (count > std::numeric_limits<uint64_t>::max() / fixedSize) {
                            ErrorInFastLLM("GGUF array byte size overflow.\n");
                        }
                        buffer.SkipBytes(count * fixedSize);
                    } else {
                        for (uint64_t i = 0; i < count; ++i) {
                            SkipGGUFValue(buffer, (gguf_type)elementTypeValue);
                        }
                    }
                    return;
                }
                default:
                    ErrorInFastLLM("Unsupported GGUF metadata type " +
                                   std::to_string((int)type) + ".\n");
            }
        }

        void ReadGGUFPayload(const std::string &fileName, uint64_t offset,
                             uint8_t *buffer, size_t bytes,
                             const std::string &tensorName) {
            const std::string safeName = SanitizeGGUFTensorName(tensorName);
            FILE *file = fopen(fileName.c_str(), "rb");
            if (file == nullptr) {
                ErrorInFastLLM("Unable to open payload for GGUF tensor '" +
                               safeName + "'.\n");
            }

        #if defined(_WIN32) || defined(_WIN64)
            const bool offsetFits =
                offset <= (uint64_t)std::numeric_limits<int64_t>::max();
            const int seekResult = offsetFits
                ? _fseeki64(file, (int64_t)offset, SEEK_SET)
                : -1;
        #else
            const bool offsetFits =
                offset <= (uint64_t)std::numeric_limits<long>::max();
            const int seekResult = offsetFits
                ? fseek(file, (long)offset, SEEK_SET)
                : -1;
        #endif
            if (seekResult != 0) {
                fclose(file);
                ErrorInFastLLM("Unable to seek payload for GGUF tensor '" +
                               safeName + "'.\n");
            }

            const size_t readBytes = fread(buffer, 1, bytes, file);
            const int closeResult = fclose(file);
            if (readBytes != bytes) {
                ErrorInFastLLM("Short payload read for GGUF tensor '" +
                               safeName + "': expected " +
                               std::to_string(bytes) + " bytes, got " +
                               std::to_string(readBytes) + ".\n");
            }
            if (closeResult != 0) {
                ErrorInFastLLM("Unable to close payload for GGUF tensor '" +
                               safeName + "'.\n");
            }
        }
    }

    extern void Float32ToFloat16(float *float32, uint16_t *float16, int len);

    void WeightImportGGUFTensor(Data* weight, ggml_tensor *tensor, std::string &fileName, uint64_t offset,
                                GGUFWeightReplaceRule::GGUFWeightReplaceType replaceType,
                                int untileNumKHeads, int untileNumVHeads,
                                int untileVRowStart, bool untileComposeNegLog) {
        if (tensor->type == ggml_type::GGML_TYPE_F32) {
            weight->dataType = DataType::FLOAT32;
        } else if (tensor->type == ggml_type::GGML_TYPE_F16) {
            weight->dataType = DataType::FLOAT16;
        } else if (tensor->type == ggml_type::GGML_TYPE_I32) {
            weight->dataType = DataType::INT32;
        } else {
            weight->dataType = DataType::DATA_GGUF_FORMAT;
        }

        if (replaceType == GGUFWeightReplaceRule::GGUFWeightReplaceDirect) {
            if (weight->dataType != DataType::DATA_GGUF_FORMAT) {
                weight->Resize(tensor->dims);
                weight->Allocate();
            } else {
                weight->dims = tensor->dims;
                weight->ggmlTensor = (void*)(new ggml_tensor());
                weight->ggmlType = tensor->type;

                weight->expansionBytes = ggml_nbytes(tensor);
                weight->cpuData = new uint8_t[ggml_nbytes(tensor)];
                (*(ggml_tensor*)weight->ggmlTensor) = *tensor;
            }

            const size_t payloadBytes = ggml_nbytes(tensor);
            ReadGGUFPayload(fileName, offset, (uint8_t*)weight->cpuData,
                            payloadBytes, tensor->name);
/*
            auto repack = get_repack_info(tensor->type);
            if (repack != nullptr && regex_search(tensor->name, std::regex(R"(blk.(\d+).ffn_(gate|up|down)_exps.weight)"))) {
                int nrows = tensor->ne[1], n_per_row = tensor->ne[0];
                auto row_size = ggml_row_size(tensor->type, n_per_row);
                std::vector<uint8_t> qtmp(repack->num_rows * row_size);
                uint8_t *qcur = (uint8_t*)weight->cpuData;
                for (int row = 0; row < nrows; row += repack->num_rows) {
                    memcpy(qtmp.data(), qcur, repack->num_rows * row_size);
                    repack->repack(repack->num_rows, n_per_row, (const char *)qtmp.data(), (char *)qcur, false);
                    qcur += repack->num_rows * row_size;
                }

                ((ggml_tensor*)weight->ggmlTensor)->type = repack->new_type;
                weight->ggmlType = (int)repack->new_type;
            } else {
                // printf("name = %s, type = %s\n", tensor->name.c_str(), ggml_type_name(tensor->type));
                // weight->PrintShape();
            }
*/
        } else if (replaceType == GGUFWeightReplaceRule::GGUFWeightReplaceForceFP32) {
            weight->dataType = DataType::FLOAT32;    
            weight->Resize(tensor->dims);
            weight->Allocate();

            auto len = ggml_nbytes(tensor);
            std::vector <uint8_t> oriData;
            oriData.resize(len);

            ReadGGUFPayload(fileName, offset, oriData.data(), len,
                            tensor->name);

            auto toFloat = ggml_type_to_float(tensor->type);
            AssertInFastLLM(toFloat != nullptr, "WeightImportGGUFTensor: weight " + tensor->name + "(type " + 
                ggml_type_name(tensor->type) + ") can't convert to fp32.");
            toFloat(oriData.data(), (float*)weight->cpuData, weight->Count(0));
        } else if (replaceType == GGUFWeightReplaceRule::GGUFWeightReplaceForceFP16) {
            weight->dataType = DataType::FLOAT16;    
            weight->Resize(tensor->dims);
            weight->Allocate();

            auto len = ggml_nbytes(tensor);
            std::vector <uint8_t> oriData;
            std::vector <float> floatData;
            oriData.resize(len);
            floatData.resize(weight->Count(0));

            ReadGGUFPayload(fileName, offset, oriData.data(), len,
                            tensor->name);

            auto toFloat = ggml_type_to_float(tensor->type);
            AssertInFastLLM(toFloat != nullptr, "WeightImportGGUFTensor: weight " + tensor->name + "(type " + 
                ggml_type_name(tensor->type) + ") can't convert to fp32.");
            toFloat(oriData.data(), floatData.data(), weight->Count(0));
            Float32ToFloat16(floatData.data(), (uint16_t*)weight->cpuData, weight->Count(0));
        } else if (replaceType == GGUFWeightReplaceRule::GGUFWeightReplaceNegLogFP32) {
            weight->dataType = DataType::FLOAT32;
            weight->Resize(tensor->dims);
            weight->Allocate();

            auto len = ggml_nbytes(tensor);
            std::vector <uint8_t> oriData;
            oriData.resize(len);

            ReadGGUFPayload(fileName, offset, oriData.data(), len,
                            tensor->name);

            if (tensor->type == ggml_type::GGML_TYPE_F32) {
                memcpy((float*)weight->cpuData, oriData.data(), ggml_nbytes(tensor));
            } else {
                auto toFloat = ggml_type_to_float(tensor->type);
                AssertInFastLLM(toFloat != nullptr, "WeightImportGGUFTensor: weight " + tensor->name + "(type " +
                    ggml_type_name(tensor->type) + ") can't convert to fp32.");
                toFloat(oriData.data(), (float*)weight->cpuData, weight->Count(0));
            }
            // GGUF stores ssm_a as -exp(A_log); FastLLM's GDN kernel applies -exp(A_log) again,
            // so invert to recover raw A_log = log(-ssm_a).
            float *data = (float*)weight->cpuData;
            for (size_t i = 0; i < (size_t)weight->Count(0); i++) {
                AssertInFastLLM(data[i] < 0.0f, "WeightImportGGUFTensor: ssm_a value >= 0, cannot take log(-x).");
                data[i] = logf(-data[i]);
            }
        } else if (replaceType == GGUFWeightReplaceRule::GGUFWeightReplaceUntileVHeads) {
            // Qwen3.5/3.6 GGUF converter tiles V-heads at export:
            //   T[((r*H+h)*W)+d] = G[((h*R+r)*W)+d]
            // Inverse (load-time, tiled→grouped) so all internal tensors use
            // grouped layout. Quantized rows are moved byte-exactly.
            auto requireLayout = [&](bool condition, const std::string &message) {
                if (!condition) {
                    throw std::runtime_error("WeightImportGGUFTensor: " + message +
                                             " for " + tensor->name + ".");
                }
            };
            requireLayout(untileNumKHeads > 0 &&
                          untileNumVHeads >= untileNumKHeads &&
                          untileNumVHeads % untileNumKHeads == 0,
                          "invalid V-head counts");
            requireLayout(tensor->dims.size() == 1 || tensor->dims.size() == 2,
                          "V-head untile expects a 1D or 2D tensor");
            const int H = untileNumKHeads;
            const int R = untileNumVHeads / H;
            const bool isMultiDim = tensor->dims.size() == 2;
            const int totalUnits = isMultiDim
                ? (int)tensor->ne[1]
                : (int)tensor->ne[0];
            requireLayout(untileVRowStart >= 0 && untileVRowStart < totalUnits,
                          "invalid V-row start");
            const int vRowCount = totalUnits - untileVRowStart;
            requireLayout(vRowCount > 0 &&
                          vRowCount % untileNumVHeads == 0,
                          "V rows do not divide into heads");
            const int D = vRowCount / untileNumVHeads;

            auto untileBytes = [&](uint8_t *storage, size_t storageBytes,
                                   size_t unitStride) {
                const size_t baseOffset =
                    (size_t)untileVRowStart * unitStride;
                const size_t groupStride = (size_t)D * unitStride;
                const size_t totalVBytes = (size_t)vRowCount * unitStride;
                requireLayout(unitStride > 0 &&
                              baseOffset + totalVBytes <= storageBytes,
                              "V-head byte range exceeds tensor storage");
                uint8_t *vSeg = storage + baseOffset;
                std::vector<uint8_t> temp(totalVBytes);
                for (int h = 0; h < H; h++) {
                    for (int r = 0; r < R; r++) {
                        memcpy(temp.data() + (size_t)(h * R + r) * groupStride,
                               vSeg + (size_t)(r * H + h) * groupStride,
                               groupStride);
                    }
                }
                memcpy(vSeg, temp.data(), totalVBytes);
            };

            if (untileComposeNegLog) {
                weight->dataType = DataType::FLOAT32;
                weight->Resize(tensor->dims);
                weight->Allocate();
                const size_t len = ggml_nbytes(tensor);
                std::vector<uint8_t> oriData(len);
                ReadGGUFPayload(fileName, offset, oriData.data(), len,
                                tensor->name);
                if (tensor->type == ggml_type::GGML_TYPE_F32) {
                    memcpy(weight->cpuData, oriData.data(), len);
                } else {
                    auto toFloat = ggml_type_to_float(tensor->type);
                    requireLayout(toFloat != nullptr,
                                  "type " + std::string(ggml_type_name(tensor->type)) +
                                  " cannot convert to fp32");
                    toFloat(oriData.data(), (float*)weight->cpuData,
                            weight->Count(0));
                }
                const size_t unitStride = isMultiDim
                    ? (size_t)tensor->ne[0] * sizeof(float)
                    : sizeof(float);
                untileBytes((uint8_t*)weight->cpuData, weight->GetBytes(),
                            unitStride);
                float *data = (float*)weight->cpuData;
                for (size_t i = 0; i < (size_t)weight->Count(0); i++) {
                    requireLayout(data[i] < 0.0f,
                                  "ssm_a value is outside the log(-x) domain");
                    data[i] = logf(-data[i]);
                }
            } else {
                if (weight->dataType != DataType::DATA_GGUF_FORMAT) {
                    weight->Resize(tensor->dims);
                    weight->Allocate();
                } else {
                    weight->dims = tensor->dims;
                    weight->ggmlTensor = (void*)(new ggml_tensor());
                    weight->ggmlType = tensor->type;
                    weight->expansionBytes = ggml_nbytes(tensor);
                    weight->cpuData = new uint8_t[weight->expansionBytes];
                    (*(ggml_tensor*)weight->ggmlTensor) = *tensor;
                }
                const size_t len = ggml_nbytes(tensor);
                ReadGGUFPayload(fileName, offset,
                                (uint8_t*)weight->cpuData, len,
                                tensor->name);
                const size_t unitStride = isMultiDim
                    ? (size_t)tensor->nb[1]
                    : (size_t)tensor->nb[0];
                untileBytes((uint8_t*)weight->cpuData, len, unitStride);
            }
        } else {
            ErrorInFastLLM("WeightImportGGUFTensor: Unsupport replace type.");
        }
    }

    void ReadGGUFMetaData(const std::string &fileName, json11::Json &config) {
        size_t ggufAlignment = GGUF_DEFAULT_ALIGNMENT;
        GGUFBuffer ggufBuffer = GGUFBuffer(fileName);
        int magic = ggufBuffer.Read<int> ();
        int version = ggufBuffer.Read<int> ();
        uint64_t tensorCount = ggufBuffer.Read <uint64_t> ();
        uint64_t metaDataCount = ggufBuffer.Read <uint64_t> ();

        json11::Json::object jsonConfig;
        jsonConfig["magic"] = magic;
        jsonConfig["version"] = version;
        jsonConfig["tensorCount"] = (double)tensorCount;
        jsonConfig["metaDataCount"] = (double)metaDataCount;

        json11::Json::object paramsConfig;

        for (uint64_t i = 0; i < metaDataCount; i++) {
            std::string key = ggufBuffer.ReadString();
            // printf("key = %s\n", key.c_str());
            const uint32_t type = ggufBuffer.Read<uint32_t>();
            if (type >= GGUF_TYPE_COUNT) {
                ErrorInFastLLM("GGUF metadata key " + key + " has invalid type " +
                               std::to_string(type) + ".\n");
            }
            paramsConfig[key] = ReadGGUFValue(ggufBuffer, (gguf_type)type);
        }

        jsonConfig["params"] = paramsConfig;
        config = json11::Json(jsonConfig);
    }

    void AppendGGUFTasks(std::string arch, const std::string &fileName, std::vector <ReadGGUFTask> &tasks) {
        size_t ggufAlignment = GGUF_DEFAULT_ALIGNMENT;
        GGUFBuffer ggufBuffer = GGUFBuffer(fileName);
        int magic = ggufBuffer.Read<int> ();
        int version = ggufBuffer.Read<int> ();
        uint64_t tensorCount = ggufBuffer.Read <uint64_t> ();
        uint64_t metaDataCount = ggufBuffer.Read <uint64_t> ();

        for (uint64_t i = 0; i < metaDataCount; i++) {
            const std::string key = ggufBuffer.ReadString();
            const uint32_t type = ggufBuffer.Read<uint32_t>();
            if (type >= GGUF_TYPE_COUNT) {
                ErrorInFastLLM("GGUF metadata key " + key + " has invalid type " +
                               std::to_string(type) + ".\n");
            }
            if (key == GGUF_KEY_GENERAL_ALIGNMENT) {
                if (type != GGUF_TYPE_UINT32) {
                    ErrorInFastLLM("GGUF general.alignment must be UINT32.\n");
                }
                const uint32_t alignment = ggufBuffer.Read<uint32_t>();
                if (alignment == 0 ||
                    (alignment & (alignment - 1)) != 0) {
                    ErrorInFastLLM("GGUF general.alignment must be a nonzero power of two.\n");
                }
                ggufAlignment = alignment;
            } else {
                SkipGGUFValue(ggufBuffer, (gguf_type)type);
            }
        }

        std::vector <std::pair <ggml_tensor, uint64_t> > tensors; // <tensors, offset>
        tensors.resize(CheckedUint64ToSize(tensorCount, "GGUF tensor count"));

        for (uint64_t i = 0; i < tensorCount; i++) {
            std::string tensorName = ggufBuffer.ReadString();
            uint32_t ndims = ggufBuffer.Read<uint32_t>();
            if (ndims == 0 || ndims > GGML_MAX_DIMS) {
                ErrorInFastLLM(
                    "GGUF tensor '" + SanitizeGGUFTensorName(tensorName) +
                    "' has invalid dimension count " +
                    std::to_string(ndims) + ".\n");
            }

            for (uint32_t j = 0; j < ndims; j++) {
                const int64_t dim = ggufBuffer.Read<int64_t>();
                if (dim <= 0 ||
                    dim > std::numeric_limits<int>::max()) {
                    ErrorInFastLLM(
                        "GGUF tensor '" + SanitizeGGUFTensorName(tensorName) +
                        "' has unsupported dimension " +
                        std::to_string(dim) + ".\n");
                }
                tensors[i].first.dims.push_back((int)dim);
            }

            for (uint32_t j = 0; j < GGML_MAX_DIMS; j++) {
                tensors[i].first.ne[j] = 1;
                if (j < ndims) {
                    tensors[i].first.ne[j] = tensors[i].first.dims[j];
                }
            }

            std::reverse(tensors[i].first.dims.begin(), tensors[i].first.dims.end());

            int type = ggufBuffer.Read <int> ();
            uint64_t offset = ggufBuffer.Read <uint64_t> ();

            {
                tensors[i].first.name = tensorName;
                tensors[i].first.type = (ggml_type)type;
                const auto *traits = FindGGMLTypeTraits(tensors[i].first.type);
                if (traits == nullptr) {
                    ErrorInFastLLM(
                        "GGUF tensor '" + SanitizeGGUFTensorName(tensorName) +
                        "' uses unsupported GGML type " + std::to_string(type) +
                        "; implement its block layout before loading.\n");
                }
                const size_t type_size = traits->type_size;
                const int64_t blck_size = traits->blck_size;
                if (tensors[i].first.ne[0] <= 0 ||
                    tensors[i].first.ne[0] % blck_size != 0) {
                    ErrorInFastLLM(
                        "GGUF tensor '" + SanitizeGGUFTensorName(tensorName) +
                        "' row length " + std::to_string(tensors[i].first.ne[0]) +
                        " is not divisible by block size " +
                        std::to_string(blck_size) + " for GGML type " +
                        std::to_string(type) + ".\n");
                }

                // calculate byte offsets given the tensor shape and type
                const std::string sizeContext =
                    "GGUF tensor '" + SanitizeGGUFTensorName(tensorName) + "'";
                tensors[i].first.nb[0] = type_size;
                tensors[i].first.nb[1] = CheckedSizeMultiply(
                    tensors[i].first.nb[0],
                    (size_t)(tensors[i].first.ne[0] / blck_size),
                    sizeContext);
                for (int j = 2; j < GGML_MAX_DIMS; ++j) {
                    tensors[i].first.nb[j] = CheckedSizeMultiply(
                        tensors[i].first.nb[j - 1],
                        (size_t)tensors[i].first.ne[j - 1],
                        sizeContext);
                }
            }

            tensors[i].second = offset;
        }

        // we require the data section to be aligned, so take into account any padding
        const long descriptorEnd = ftell(ggufBuffer.f);
        if (descriptorEnd < 0) {
            ErrorInFastLLM("Unable to determine GGUF descriptor position.\n");
        }
        const size_t alignedDescriptorEnd = CheckedSizePad(
            (size_t)descriptorEnd, ggufAlignment, "GGUF data section");
        if (alignedDescriptorEnd > (size_t)std::numeric_limits<long>::max() ||
            fseek(ggufBuffer.f, (long)alignedDescriptorEnd, SEEK_SET) != 0) {
            ErrorInFastLLM("Unable to seek to aligned GGUF data section.\n");
        }

        const long basePosition = ftell(ggufBuffer.f);
        if (basePosition < 0) {
            ErrorInFastLLM("Unable to determine GGUF data position.\n");
        }
        const size_t baseOffset = (size_t)basePosition;
        size_t curPos = baseOffset;

        std::vector <GGUFWeightReplaceRule> weightNameConverterRules = GetGGUFWeightReplaceRules(arch);

        for (uint64_t i = 0; i < tensorCount; i++) {
            const size_t tensorOffset = CheckedSizeAdd(
                baseOffset,
                CheckedUint64ToSize(tensors[i].second,
                                    "GGUF tensor relative offset"),
                "GGUF tensor absolute offset");
            if (curPos != tensorOffset) {
                ErrorInFastLLM("GGUF tensor '" +
                               SanitizeGGUFTensorName(tensors[i].first.name) +
                               "' has a non-contiguous or overflowed offset.\n");
            }

            std::string name = tensors[i].first.name;
            bool matched = false;
            int matchedCount = 0;

            for (auto &it : weightNameConverterRules) {
                if (std::regex_search(name, it.pattern)) {
                    matched = true;
                    matchedCount++;

                    if (it.type == GGUFWeightReplaceRule::GGUFWeightReplaceDirect) {
                        name = std::regex_replace(name, it.pattern, it.names[0]);
                        if (name == "ignore") {
                            break;
                        }
                        tasks.push_back (
                                ReadGGUFTask (
                                    name, nullptr, tensors[i].first,
                                    ggufBuffer.fileName, tensorOffset
                                )
                        );
                    } else if (it.type == GGUFWeightReplaceRule::GGUFWeightReplaceForceFP32 ||
                                it.type == GGUFWeightReplaceRule::GGUFWeightReplaceForceFP16 ||
                                it.type == GGUFWeightReplaceRule::GGUFWeightReplaceNegLogFP32 ||
                                it.type == GGUFWeightReplaceRule::GGUFWeightReplaceUntileVHeads) {
                        name = std::regex_replace(name, it.pattern, it.names[0]);
                        if (name == "ignore") {
                            break;
                        }
                        tasks.push_back (
                            ReadGGUFTask (
                                name, nullptr, tensors[i].first,
                                ggufBuffer.fileName, tensorOffset, it.type
                            )
                        );
                        tasks.back().untileComposeNegLog = it.untileComposeNegLog;
                    } else if (it.type == GGUFWeightReplaceRule::GGUFWeightReplacePacked) {
                        std::string prefix = std::regex_replace(name, it.pattern, it.names[0]);
                        std::string suffix = std::regex_replace(name, it.pattern, it.names[1]);

                        int packedBatch = tensors[i].first.ne[2];
                        ggml_tensor singleTensor = tensors[i].first;
                        singleTensor.dims.erase(singleTensor.dims.begin());
                        singleTensor.ne[2] = 1;
                        singleTensor.nb[2] = singleTensor.nb[3] = singleTensor.nb[1];

                        const size_t expertBytes = ggml_nbytes(&singleTensor);
                        for (int idx = 0; idx < packedBatch; idx++) {
                            std::string modelName = prefix + std::to_string(idx) + suffix;
                            const size_t expertOffset = CheckedSizeAdd(
                                tensorOffset,
                                CheckedSizeMultiply((size_t)idx, expertBytes,
                                                    "GGUF packed expert offset"),
                                "GGUF packed expert offset");
                            tasks.push_back (
                                ReadGGUFTask (
                                    modelName, nullptr, singleTensor,
                                    ggufBuffer.fileName, expertOffset
                                )
                            );
                        }
                    }
                }
            } 

            if ((arch == "deepseek4" || arch == "deepseek_v4") && matchedCount != 1) {
                ErrorInFastLLM("DeepSeek V4 GGUF tensor " + name + " matched " +
                               std::to_string(matchedCount) + " adapter rules; expected exactly one.\n");
            }
            if (!matched) {
                printf("unmatched weight %s (", name.c_str());
                for (auto it : tensors[i].first.dims) {
                    printf("%d ", it);
                }
                printf(") type = %s\n", ggml_type_name(tensors[i].first.type));
            }

            curPos = CheckedSizeAdd(
                (size_t)curPos,
                CheckedSizePad(ggml_nbytes(&tensors[i].first),
                               (size_t)ggufAlignment,
                               "GGUF tensor padding"),
                "GGUF tensor progression");
        }
    }

    void ReadGGUF(basellm *model, const std::string &fileName, std::vector <ReadGGUFTask> &tasks) {
        // 仅做测试用
        size_t ggufAlignment = GGUF_DEFAULT_ALIGNMENT;
        GGUFBuffer ggufBuffer = GGUFBuffer(fileName);
        int magic = ggufBuffer.Read<int> ();
        int version = ggufBuffer.Read<int> ();
        uint64_t tensorCount = ggufBuffer.Read <uint64_t> ();
        uint64_t metaDataCount = ggufBuffer.Read <uint64_t> ();

        printf("magic = %d\n", magic);
        printf("version = %d\n", version);
        printf("tensorCount = %d\n", (int)tensorCount);
        printf("metaDataCount = %d\n", (int)metaDataCount);

        for (uint64_t i = 0; i < metaDataCount; i++) {
            const std::string key = ggufBuffer.ReadString();
            const uint32_t type = ggufBuffer.Read<uint32_t>();
            if (type >= GGUF_TYPE_COUNT) {
                ErrorInFastLLM("GGUF metadata key " + key + " has invalid type " +
                               std::to_string(type) + ".\n");
            }
            const json11::Json value =
                ReadGGUFValue(ggufBuffer, (gguf_type)type);
            if (key == GGUF_KEY_GENERAL_ALIGNMENT) {
                if (type != GGUF_TYPE_UINT32) {
                    ErrorInFastLLM("GGUF general.alignment must be UINT32.\n");
                }
                const uint32_t alignment = (uint32_t)value.number_value();
                if (alignment == 0 ||
                    (alignment & (alignment - 1)) != 0) {
                    ErrorInFastLLM("GGUF general.alignment must be a nonzero power of two.\n");
                }
                ggufAlignment = alignment;
            }
            printf("key = %s\nvalue = %s\n", key.c_str(), value.dump().c_str());
        }

        std::vector <std::pair <ggml_tensor, uint64_t> > tensors; // <tensors, offset>
        tensors.resize(CheckedUint64ToSize(tensorCount, "GGUF tensor count"));

        for (uint64_t i = 0; i < tensorCount; i++) {
            std::string tensorName = ggufBuffer.ReadString();
            uint32_t ndims = ggufBuffer.Read<uint32_t>();
            if (ndims == 0 || ndims > GGML_MAX_DIMS) {
                ErrorInFastLLM(
                    "GGUF tensor '" + SanitizeGGUFTensorName(tensorName) +
                    "' has invalid dimension count " +
                    std::to_string(ndims) + ".\n");
            }

            for (uint32_t j = 0; j < ndims; j++) {
                const int64_t dim = ggufBuffer.Read<int64_t>();
                if (dim <= 0 ||
                    dim > std::numeric_limits<int>::max()) {
                    ErrorInFastLLM(
                        "GGUF tensor '" + SanitizeGGUFTensorName(tensorName) +
                        "' has unsupported dimension " +
                        std::to_string(dim) + ".\n");
                }
                tensors[i].first.dims.push_back((int)dim);
            }

            for (uint32_t j = 0; j < GGML_MAX_DIMS; j++) {
                tensors[i].first.ne[j] = 1;
                if (j < ndims) {
                    tensors[i].first.ne[j] = tensors[i].first.dims[j];
                }
            }

            std::reverse(tensors[i].first.dims.begin(), tensors[i].first.dims.end());

            int type = ggufBuffer.Read <int> ();
            uint64_t offset = ggufBuffer.Read <uint64_t> ();

            {
                tensors[i].first.name = tensorName;
                tensors[i].first.type = (ggml_type)type;
                const auto *traits = FindGGMLTypeTraits(tensors[i].first.type);
                if (traits == nullptr) {
                    ErrorInFastLLM(
                        "GGUF tensor '" + SanitizeGGUFTensorName(tensorName) +
                        "' uses unsupported GGML type " + std::to_string(type) +
                        "; implement its block layout before loading.\n");
                }
                const size_t type_size = traits->type_size;
                const int64_t blck_size = traits->blck_size;
                if (tensors[i].first.ne[0] <= 0 ||
                    tensors[i].first.ne[0] % blck_size != 0) {
                    ErrorInFastLLM(
                        "GGUF tensor '" + SanitizeGGUFTensorName(tensorName) +
                        "' row length " + std::to_string(tensors[i].first.ne[0]) +
                        " is not divisible by block size " +
                        std::to_string(blck_size) + " for GGML type " +
                        std::to_string(type) + ".\n");
                }

                // calculate byte offsets given the tensor shape and type
                const std::string sizeContext =
                    "GGUF tensor '" + SanitizeGGUFTensorName(tensorName) + "'";
                tensors[i].first.nb[0] = type_size;
                tensors[i].first.nb[1] = CheckedSizeMultiply(
                    tensors[i].first.nb[0],
                    (size_t)(tensors[i].first.ne[0] / blck_size),
                    sizeContext);
                for (int j = 2; j < GGML_MAX_DIMS; ++j) {
                    tensors[i].first.nb[j] = CheckedSizeMultiply(
                        tensors[i].first.nb[j - 1],
                        (size_t)tensors[i].first.ne[j - 1],
                        sizeContext);
                }
            }

            tensors[i].second = offset;
        }

        // we require the data section to be aligned, so take into account any padding
        const long descriptorEnd = ftell(ggufBuffer.f);
        if (descriptorEnd < 0) {
            ErrorInFastLLM("Unable to determine GGUF descriptor position.\n");
        }
        const size_t alignedDescriptorEnd = CheckedSizePad(
            (size_t)descriptorEnd, ggufAlignment, "GGUF data section");
        if (alignedDescriptorEnd > (size_t)std::numeric_limits<long>::max() ||
            fseek(ggufBuffer.f, (long)alignedDescriptorEnd, SEEK_SET) != 0) {
            ErrorInFastLLM("Unable to seek to aligned GGUF data section.\n");
        }

        const long basePosition = ftell(ggufBuffer.f);
        if (basePosition < 0) {
            ErrorInFastLLM("Unable to determine GGUF data position.\n");
        }
        const size_t baseOffset = (size_t)basePosition;
        size_t curPos = baseOffset;

        std::vector <GGUFWeightReplaceRule> weightNameConverterRules = GetGGUFWeightReplaceRules(model->model_type);
        for (uint64_t i = 0; i < tensorCount; i++) {
            const size_t tensorOffset = CheckedSizeAdd(
                baseOffset,
                CheckedUint64ToSize(tensors[i].second,
                                    "GGUF tensor relative offset"),
                "GGUF tensor absolute offset");
            if (curPos != tensorOffset) {
                ErrorInFastLLM("GGUF tensor '" +
                               SanitizeGGUFTensorName(tensors[i].first.name) +
                               "' has a non-contiguous or overflowed offset.\n");
            }

            std::string name = tensors[i].first.name;
            bool matched = false;
            int matchedCount = 0;

            for (auto &it : weightNameConverterRules) {
                if (std::regex_search(name, it.pattern)) {
                    matched = true;
                    matchedCount++;

                    if (it.type == GGUFWeightReplaceRule::GGUFWeightReplaceDirect) {
                        name = std::regex_replace(name, it.pattern, it.names[0]);
                        if (model->weight.weight.find(name) != model->weight.weight.end()) {
                            tasks.push_back (
                                ReadGGUFTask (
                                    name, &model->weight.weight[name],
                                    tensors[i].first, ggufBuffer.fileName,
                                    tensorOffset
                                )
                            );
                            // printf("replace %s\n", name.c_str());
                        }
                    } else if (it.type == GGUFWeightReplaceRule::GGUFWeightReplaceForceFP32 ||
                               it.type == GGUFWeightReplaceRule::GGUFWeightReplaceForceFP16 ||
                               it.type == GGUFWeightReplaceRule::GGUFWeightReplaceNegLogFP32 ||
                               it.type == GGUFWeightReplaceRule::GGUFWeightReplaceUntileVHeads) {
                        name = std::regex_replace(name, it.pattern, it.names[0]);
                        if (model->weight.weight.find(name) != model->weight.weight.end()) {
                            tasks.push_back (
                                ReadGGUFTask (
                                    name, &model->weight.weight[name],
                                    tensors[i].first, ggufBuffer.fileName,
                                    tensorOffset, it.type
                                )
                            );
                            tasks.back().untileComposeNegLog = it.untileComposeNegLog;
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

                        const size_t expertBytes = ggml_nbytes(&singleTensor);
                        for (int idx = 0; idx < packedBatch; idx++) {
                            std::string modelName = prefix + std::to_string(idx) + suffix;
                            if (model->weight.weight.find(modelName) != model->weight.weight.end()) {
                                const size_t expertOffset = CheckedSizeAdd(
                                    tensorOffset,
                                    CheckedSizeMultiply((size_t)idx, expertBytes,
                                                        "GGUF packed expert offset"),
                                    "GGUF packed expert offset");
                                tasks.push_back (
                                    ReadGGUFTask (
                                        modelName, &model->weight.weight[modelName], singleTensor,
                                        ggufBuffer.fileName, expertOffset
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

            if ((model->model_type == "deepseek4" ||
                 model->model_type == "deepseek_v4") &&
                matchedCount != 1) {
                ErrorInFastLLM("DeepSeek V4 GGUF tensor " + name + " matched " +
                               std::to_string(matchedCount) +
                               " adapter rules; expected exactly one.\n");
            }
            if (!matched) {
                printf("unmatched weight %s (", name.c_str());
                for (auto it : tensors[i].first.dims) {
                    printf("%d ", it);
                }
                printf(") type = %s\n", ggml_type_name(tensors[i].first.type));
            }

            curPos = CheckedSizeAdd(
                curPos,
                CheckedSizePad(ggml_nbytes(&tensors[i].first),
                               ggufAlignment,
                               "GGUF tensor padding"),
                "GGUF tensor progression");
        }
    }
}
