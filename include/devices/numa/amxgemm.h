//
// AMX (Advanced Matrix Extensions) Support for FastLLM
// Ported from lktransformers/llamafile
//

#ifndef FASTLLM_AMXGEMM_H
#define FASTLLM_AMXGEMM_H

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)

#include <cstdint>
#include <cstddef>
#include "gguf.h"

namespace fastllm {

// AMX tile configuration constants
#define TILE_M 16
#define TILE_N 16
#define TILE_K 32
#define VNNI_BLK 4
#define AMX_BLK_SIZE 32

// Tile register indices
#define TMM0 0
#define TMM1 1
#define TMM2 2
#define TMM3 3
#define TMM4 4
#define TMM5 5
#define TMM6 6
#define TMM7 7

// AMX tile configuration structure
struct AmxTileConfig {
    uint8_t palette_id = 0;
    uint8_t start_row = 0;
    uint8_t reserved_0[14] = {0};
    uint16_t colsb[16] = {0};
    uint8_t rows[16] = {0};
};

/**
 * AMX GEMM Engine
 * High-performance matrix multiplication using Intel AMX
 */
class AmxGemm {
public:
    // Initialize AMX tile configuration (thread-local)
    static void InitTileConfig();
    
    // Convert weight to AMX-optimized format
    static void ConvertWeightToAmxFormat(
        void* packed_data,
        const void* original_data,
        ggml_type type,
        int K,  // input dimension
        int N   // output dimension
    );
    
    // Get packed weight size for AMX format
    static size_t GetAmxPackedSize(ggml_type type, int K, int N);
    
    // Perform AMX-accelerated GEMM
    // weight: {N, K} in VNNI format
    // input: {M, K}
    // output: {M, N}
    static void Compute(
        ggml_type weight_type,
        const void* weight_data,
        const void* input_data,
        float* output_data,
        int M,      // batch size
        int N,      // output dimension
        int K,      // input dimension
        int ldc     // leading dimension of C (output)
    );
    
    // Check if AMX is available on current CPU
    static bool IsAvailable();
};

// GGML type dispatcher macro
#define FASTLLM_DISPATCH_GGML_TYPES(QT, ...)                                   \
    [&] {                                                                       \
        switch (QT) {                                                           \
            case GGML_TYPE_Q4_0: {                                              \
                using type = block_q4_0;                                        \
                using vec_dot_type = block_q8_0;                                \
                constexpr int blck_size = QK4_0;                                \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case GGML_TYPE_Q4_1: {                                              \
                using type = block_q4_1;                                        \
                using vec_dot_type = block_q8_1;                                \
                constexpr int blck_size = QK4_1;                                \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case GGML_TYPE_Q8_0: {                                              \
                using type = block_q8_0;                                        \
                using vec_dot_type = block_q8_0;                                \
                constexpr int blck_size = QK8_0;                                \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case GGML_TYPE_Q4_K: {                                              \
                using type = block_q4_K;                                        \
                using vec_dot_type = block_q8_K;                                \
                constexpr int blck_size = QK_K;                                 \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case GGML_TYPE_Q5_K: {                                              \
                using type = block_q5_K;                                        \
                using vec_dot_type = block_q8_K;                                \
                constexpr int blck_size = QK_K;                                 \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case GGML_TYPE_Q6_K: {                                              \
                using type = block_q6_K;                                        \
                using vec_dot_type = block_q8_K;                                \
                constexpr int blck_size = QK_K;                                 \
                return __VA_ARGS__();                                           \
            }                                                                   \
            case GGML_TYPE_IQ4_XS: {                                            \
                using type = block_iq4_xs;                                      \
                using vec_dot_type = block_q8_K;                                \
                constexpr int blck_size = QK_K;                                 \
                return __VA_ARGS__();                                           \
            }                                                                   \
            default:                                                            \
                ErrorInFastLLM("Unsupported GGML quantized type: " +            \
                             std::to_string(int(QT)));                          \
        }                                                                       \
    }()

}  // namespace fastllm

#endif  // defined(__AMX_INT8__) && defined(__AVX512VNNI__)

#endif  // FASTLLM_AMXGEMM_H

