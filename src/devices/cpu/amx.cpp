//
// Created by huangyuyang on 5/8/25.
//

#include <cstdint>

#ifdef __AVX2__
#include "immintrin.h"
#endif

#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include <cstdlib>

// AMX tile 配置
#define TILE_M 16       // tile 行数 (Batch Size / Sequence Length 方向)
#define TILE_N 16       // tile 列数 (Output Feature 方向)
#define TILE_K 32       // K维度 (Input Feature 方向, BF16是2字节，所以是32个元素)
// 一个 Tile B (权重) 占用的字节数: 16行(VNNI packing后) * 64字节(宽) = 1024字节
#define TILE_B_SIZE_BYTES 1024 

// BF16 类型定义
typedef uint16_t bf16_t;

// Tile 配置结构
typedef struct __tile_config {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} __tilecfg;

#include "fastllm.h"
// ARCH_REQ_XCOMP_PERM 系统调用
#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18
#include <sys/syscall.h>

namespace fastllm {
    extern void AddBiasAVX512(float *outputData, float *biasData, int n, int k, int st, int end);

    void InitAMX() {
#if defined(__AMX_TILE__)
        if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0) {
            printf("init amx failed.\n");
            exit(0);
        } else {
            printf("enable amx finish.\n");
        }
#else
        printf("enable amx failed.\n");
#endif
    }

    // -------------------------------------------------------------------------
    // 辅助函数: 上取整除法
    static inline int ceil_div(int a, int b) {
        return (a + b - 1) / b;
    }

    // -------------------------------------------------------------------------
    // 2. Compute 阶段
    // 使用预先打包好的 packed_B 进行计算
    // A: [N, M] (Row Major)
    // PackedB: VNNI Block Layout
    // C: [N, width_k] (Row Major, result offset applied internally per tile)
    void amx_bf16_matmul_packed(const uint16_t *A, const uint16_t *PackedB, float *C,
                               int N, int M, int GlobalK, 
                               int st, int end,
                               int lda, int ldc) {
#if defined(__AMX_TILE__)
        __tilecfg tile_cfg;
        memset(&tile_cfg, 0, sizeof(tile_cfg));
        tile_cfg.palette_id = 1;

        // 配置 Tile 寄存器
        // Tile 0 (A): Input, 16 rows, 64 bytes wide
        tile_cfg.rows[0] = TILE_M;
        tile_cfg.colsb[0] = TILE_K * sizeof(uint16_t);
        
        // Tile 2 (B): Weight (VNNI), 16 rows, 64 bytes wide
        // 注意：packed B 的 scanline stride 是 64 字节 (TILE_N * 2 * sizeof(bf16))
        tile_cfg.rows[2] = TILE_K / 2; 
        tile_cfg.colsb[2] = TILE_N * sizeof(float); // = 64 bytes
        
        // Tile 4 (C): Accumulator, 16 rows, 64 bytes wide
        tile_cfg.rows[4] = TILE_M;
        tile_cfg.colsb[4] = TILE_N * sizeof(float);
        
        _tile_loadconfig(&tile_cfg);

        // 预分配 A 和 C 的临时块
        uint16_t *A_block = (uint16_t *)aligned_alloc(64, TILE_M * TILE_K * sizeof(uint16_t));
        float *C_block = (float *)aligned_alloc(64, TILE_M * TILE_N * sizeof(float));
        
        int width_k = end - st;
        int num_m_blocks = ceil_div(M, TILE_K); // 用于计算 PackedB 的偏移

        // 1. 遍历 Batch (N)
        for (int n_outer = 0; n_outer < N; n_outer += TILE_M) {
            int tile_rows_a = (N - n_outer) < TILE_M ? (N - n_outer) : TILE_M;
            
            // 2. 遍历 Output Features (K)
            // block_idx_k 追踪 PackedB 在 K 维度的索引
            int block_idx_k = 0;
            
            for (int k_outer = 0; k_outer < width_k; k_outer += TILE_N) {
                int tile_cols_c = (width_k - k_outer) < TILE_N ? (width_k - k_outer) : TILE_N;
                
                // 清零累加器
                _tile_zero(4);
                
                // 3. 遍历 Reduction (M)
                for (int m_outer = 0; m_outer < M; m_outer += TILE_K) {
                    int m_block_idx = m_outer / TILE_K;
                    int tile_common_m = (M - m_outer) < TILE_K ? (M - m_outer) : TILE_K;

                    // --- 打包 A (Input) ---
                    // 这里仍然需要运行时打包，因为 A 的 Batch Size 是动态的，且通常不重用
                    _tile_loadd(0, &A[n_outer * lda + m_outer], lda * sizeof(uint16_t));

                    // --- 加载 B (From Packed Buffer) ---
                    // 计算 PackedB 中的偏移量
                    // Layout 是 [K_block][M_block][1024 bytes]
                    long long current_block_offset = (long long)block_idx_k * num_m_blocks + m_block_idx;
                    const uint16_t* b_ptr = PackedB + current_block_offset * (TILE_B_SIZE_BYTES / sizeof(uint16_t));
                    
                    // stride 固定为 64 (tile 宽度)
                    _tile_loadd(2, b_ptr, 64);

                    // 计算
                    _tile_dpbf16ps(4, 0, 2);
                }

                // 存回结果
                _tile_stored(4, C_block, TILE_N * sizeof(float));
                for (int i = 0; i < tile_rows_a; i++) {
                    for (int j = 0; j < tile_cols_c; j++) {
                        C[(n_outer + i) * ldc + (st + k_outer + j)] = C_block[i * TILE_N + j];
                    }
                }
                
                block_idx_k++;
            }
        }
        
        _tile_release();
        free(A_block);
        free(C_block);
#else
        printf("Unsupport AMX.\n");
        exit(0);
#endif
    }

#if defined(__AMX_TILE__)
    // This function copy from https://github.com/alibaba/MNN
    inline void transpose16x16F(
                    __m512& r0f, __m512& r1f, __m512& r2f, __m512& r3f,
                    __m512& r4f, __m512& r5f, __m512& r6f, __m512& r7f,
                    __m512& r8f, __m512& r9f, __m512& raf, __m512& rbf,
                    __m512& rcf, __m512& rdf, __m512& ref, __m512& rff) {
        auto r0 = _mm512_castps_si512(r0f);
        auto r1 = _mm512_castps_si512(r1f);
        auto r2 = _mm512_castps_si512(r2f);
        auto r3 = _mm512_castps_si512(r3f);
        auto r4 = _mm512_castps_si512(r4f);
        auto r5 = _mm512_castps_si512(r5f);
        auto r6 = _mm512_castps_si512(r6f);
        auto r7 = _mm512_castps_si512(r7f);
        auto r8 = _mm512_castps_si512(r8f);
        auto r9 = _mm512_castps_si512(r9f);
        auto ra = _mm512_castps_si512(raf);
        auto rb = _mm512_castps_si512(rbf);
        auto rc = _mm512_castps_si512(rcf);
        auto rd = _mm512_castps_si512(rdf);
        auto re = _mm512_castps_si512(ref);
        auto rf = _mm512_castps_si512(rff);
        //given __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;
        __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;

        t0 = _mm512_unpacklo_epi32(r0,r1); //   0  16   1  17   4  20   5  21   8  24   9  25  12  28  13  29
        t1 = _mm512_unpackhi_epi32(r0,r1); //   2  18   3  19   6  22   7  23  10  26  11  27  14  30  15  31
        t2 = _mm512_unpacklo_epi32(r2,r3); //  32  48  33  49 ...
        t3 = _mm512_unpackhi_epi32(r2,r3); //  34  50  35  51 ...
        t4 = _mm512_unpacklo_epi32(r4,r5); //  64  80  65  81 ...
        t5 = _mm512_unpackhi_epi32(r4,r5); //  66  82  67  83 ...
        t6 = _mm512_unpacklo_epi32(r6,r7); //  96 112  97 113 ...
        t7 = _mm512_unpackhi_epi32(r6,r7); //  98 114  99 115 ...
        t8 = _mm512_unpacklo_epi32(r8,r9); // 128 ...
        t9 = _mm512_unpackhi_epi32(r8,r9); // 130 ...
        ta = _mm512_unpacklo_epi32(ra,rb); // 160 ...
        tb = _mm512_unpackhi_epi32(ra,rb); // 162 ...
        tc = _mm512_unpacklo_epi32(rc,rd); // 196 ...
        td = _mm512_unpackhi_epi32(rc,rd); // 198 ...
        te = _mm512_unpacklo_epi32(re,rf); // 228 ...
        tf = _mm512_unpackhi_epi32(re,rf); // 230 ...

        r0 = _mm512_unpacklo_epi64(t0,t2); //   0  16  32  48 ...
        r1 = _mm512_unpackhi_epi64(t0,t2); //   1  17  33  49 ...
        r2 = _mm512_unpacklo_epi64(t1,t3); //   2  18  34  49 ...
        r3 = _mm512_unpackhi_epi64(t1,t3); //   3  19  35  51 ...
        r4 = _mm512_unpacklo_epi64(t4,t6); //  64  80  96 112 ...
        r5 = _mm512_unpackhi_epi64(t4,t6); //  65  81  97 114 ...
        r6 = _mm512_unpacklo_epi64(t5,t7); //  66  82  98 113 ...
        r7 = _mm512_unpackhi_epi64(t5,t7); //  67  83  99 115 ...
        r8 = _mm512_unpacklo_epi64(t8,ta); // 128 144 160 176 ...
        r9 = _mm512_unpackhi_epi64(t8,ta); // 129 145 161 178 ...
        ra = _mm512_unpacklo_epi64(t9,tb); // 130 146 162 177 ...
        rb = _mm512_unpackhi_epi64(t9,tb); // 131 147 163 179 ...
        rc = _mm512_unpacklo_epi64(tc,te); // 192 208 228 240 ...
        rd = _mm512_unpackhi_epi64(tc,te); // 193 209 229 241 ...
        re = _mm512_unpacklo_epi64(td,tf); // 194 210 230 242 ...
        rf = _mm512_unpackhi_epi64(td,tf); // 195 211 231 243 ...

        t0 = _mm512_shuffle_i32x4(r0, r4, 0x88); //   0  16  32  48   8  24  40  56  64  80  96  112 ...
        t1 = _mm512_shuffle_i32x4(r1, r5, 0x88); //   1  17  33  49 ...
        t2 = _mm512_shuffle_i32x4(r2, r6, 0x88); //   2  18  34  50 ...
        t3 = _mm512_shuffle_i32x4(r3, r7, 0x88); //   3  19  35  51 ...
        t4 = _mm512_shuffle_i32x4(r0, r4, 0xdd); //   4  20  36  52 ...
        t5 = _mm512_shuffle_i32x4(r1, r5, 0xdd); //   5  21  37  53 ...
        t6 = _mm512_shuffle_i32x4(r2, r6, 0xdd); //   6  22  38  54 ...
        t7 = _mm512_shuffle_i32x4(r3, r7, 0xdd); //   7  23  39  55 ...
        t8 = _mm512_shuffle_i32x4(r8, rc, 0x88); // 128 144 160 176 ...
        t9 = _mm512_shuffle_i32x4(r9, rd, 0x88); // 129 145 161 177 ...
        ta = _mm512_shuffle_i32x4(ra, re, 0x88); // 130 146 162 178 ...
        tb = _mm512_shuffle_i32x4(rb, rf, 0x88); // 131 147 163 179 ...
        tc = _mm512_shuffle_i32x4(r8, rc, 0xdd); // 132 148 164 180 ...
        td = _mm512_shuffle_i32x4(r9, rd, 0xdd); // 133 149 165 181 ...
        te = _mm512_shuffle_i32x4(ra, re, 0xdd); // 134 150 166 182 ...
        tf = _mm512_shuffle_i32x4(rb, rf, 0xdd); // 135 151 167 183 ...

        r0 = _mm512_shuffle_i32x4(t0, t8, 0x88); //   0  16  32  48  64  80  96 112 ... 240
        r1 = _mm512_shuffle_i32x4(t1, t9, 0x88); //   1  17  33  49  66  81  97 113 ... 241
        r2 = _mm512_shuffle_i32x4(t2, ta, 0x88); //   2  18  34  50  67  82  98 114 ... 242
        r3 = _mm512_shuffle_i32x4(t3, tb, 0x88); //   3  19  35  51  68  83  99 115 ... 243
        r4 = _mm512_shuffle_i32x4(t4, tc, 0x88); //   4 ...
        r5 = _mm512_shuffle_i32x4(t5, td, 0x88); //   5 ...
        r6 = _mm512_shuffle_i32x4(t6, te, 0x88); //   6 ...
        r7 = _mm512_shuffle_i32x4(t7, tf, 0x88); //   7 ...
        r8 = _mm512_shuffle_i32x4(t0, t8, 0xdd); //   8 ...
        r9 = _mm512_shuffle_i32x4(t1, t9, 0xdd); //   9 ...
        ra = _mm512_shuffle_i32x4(t2, ta, 0xdd); //  10 ...
        rb = _mm512_shuffle_i32x4(t3, tb, 0xdd); //  11 ...
        rc = _mm512_shuffle_i32x4(t4, tc, 0xdd); //  12 ...
        rd = _mm512_shuffle_i32x4(t5, td, 0xdd); //  13 ...
        re = _mm512_shuffle_i32x4(t6, te, 0xdd); //  14 ...
        rf = _mm512_shuffle_i32x4(t7, tf, 0xdd); //  15  31  47  63  79  96 111 127 ... 255

        r0f = _mm512_castsi512_ps(r0);
        r1f = _mm512_castsi512_ps(r1);
        r2f = _mm512_castsi512_ps(r2);
        r3f = _mm512_castsi512_ps(r3);
        r4f = _mm512_castsi512_ps(r4);
        r5f = _mm512_castsi512_ps(r5);
        r6f = _mm512_castsi512_ps(r6);
        r7f = _mm512_castsi512_ps(r7);
        r8f = _mm512_castsi512_ps(r8);
        r9f = _mm512_castsi512_ps(r9);
        raf = _mm512_castsi512_ps(ra);
        rbf = _mm512_castsi512_ps(rb);
        rcf = _mm512_castsi512_ps(rc);
        rdf = _mm512_castsi512_ps(rd);
        ref = _mm512_castsi512_ps(re);
        rff = _mm512_castsi512_ps(rf);
    }
#endif

    // 假设已经包含了 transpose16x16F 函数定义
    static inline void amx_transpose_pack_kernel_avx512(float *dst_tile, 
                                                            const float *src_base,
                                                            int ldb_float, 
                                                            int valid_cols, // k (width)
                                                            int valid_rows) // m (height) 
    {
#if defined(__AMX_TILE__)
        // 1. 准备 K 维度的 Mask
        // 如果 valid_cols = 16，mask 为 0xFFFF；如果 < 16，高位为 0，防止非法读取并自动填充 0
        __mmask16 load_mask = (1 << valid_cols) - 1;
        // 2. 准备全零寄存器 (用于填充 M 维度的 padding)
        __m512 vzero = _mm512_setzero_ps();
        // 3. 定义寄存器数组
        // 使用数组可以让编译器更好地展开循环，并映射到 ZMM0-ZMM15
        __m512 regs[16];
        // 4. 加载数据 (Load Phase)
        // 这里替代了 Gather。我们按行加载，而不是按列收集。
        // 编译器会自动展开这个循环 (Unroll)
        #pragma unroll(16)
        for (int i = 0; i < 16; ++i) {
            if (i < valid_rows) {
                // Case A: 有效行
                // 使用 maskz_loadu 读取 src_base + i行。
                // mask 负责处理 valid_cols 不足 16 的情况，越界部分自动置 0。
                regs[i] = _mm512_maskz_loadu_ps(load_mask, src_base + i * ldb_float);
            } else {
                // Case B: M 维度 Padding 行 (无效行)
                // 直接置 0，防止计算错误
                regs[i] = vzero;
            }
        }
        // 5. 核心转置 (Transpose Phase)
        // 利用提供的 Shuffle 函数在寄存器内完成 16x16 转置
        transpose16x16F(
            regs[0], regs[1], regs[2], regs[3],
            regs[4], regs[5], regs[6], regs[7],
            regs[8], regs[9], regs[10], regs[11],
            regs[12], regs[13], regs[14], regs[15]
        );
        // 6. 存储数据 (Store Phase)
        // 转置后，数据的顺序正好符合 dst_tile 的连续写入需求
        #pragma unroll(16)
        for (int i = 0; i < 16; ++i) {
            _mm512_storeu_ps(dst_tile + i * 16, regs[i]);
        }
#else
        printf("Unsupport AMX.\n");
        exit(0);
#endif
    }

    // -------------------------------------------------------------------------
    // 调用层修改
    // -------------------------------------------------------------------------
    void amx_pack_weight(const uint16_t *B, uint16_t *packed_buffer,
                        int M, int GlobalK, int st, int end, int ldb) {
        int width_k = end - st;
        int block_idx = 0;
        
        // 预计算 float 步长 (bf16 x 2 = float x 1)
        int ldb_float = ldb >> 1; 
        for (int k_outer = 0; k_outer < width_k; k_outer += 16) { // TILE_N = 16
            int tile_cols_c = (width_k - k_outer) < 16 ? (width_k - k_outer) : 16;
            
            for (int m_outer = 0; m_outer < M; m_outer += 32) { // TILE_K = 32
                int tile_common_m = (M - m_outer) < 32 ? (M - m_outer) : 32;
                int tile_m_padded = ((tile_common_m + 1) / 2) * 2; // 对齐到 2
                
                // 指针定位
                uint16_t *current_block_dst = packed_buffer + block_idx * (1024 / sizeof(uint16_t)); // 1KB per tile
                
                // 计算源地址起始位置 (对应 float*)
                // B 的起始: B + r * ldb + c
                // row = st + k_outer (K维度起始)
                // col = m_outer (M维度起始)
                // 指针偏移量 (short): (st + k_outer) * ldb + m_outer
                // 转为 float 指针时，偏移量除以 2 (假设 m_outer 是2的倍数，必然成立)
                size_t src_offset = (size_t)(st + k_outer) * ldb + m_outer;
                const float *src_float_ptr = (const float*)(B) + (src_offset >> 1);
                // 调用 AVX512 Kernel
                // 注意：valid_rows 传入的是 float 的行数 (pairs)，即 tile_m_padded / 2
                amx_transpose_pack_kernel_avx512(
                    (float*)current_block_dst,
                    src_float_ptr,
                    ldb_float,
                    tile_cols_c,
                    tile_m_padded / 2
                );
                block_idx++;
            }
        }
    }

    // -------------------------------------------------------------------------
    // 辅助计算函数: 处理 M 维度的边界 (当 M 不能被 32 整除时)
    static inline uint32_t get_A_pair_safe(const uint16_t* A, int current_m, int max_m) {
        uint32_t val = 0;
        // 第一个值
        if (current_m < max_m) {
            val |= A[current_m];
        }
        // 第二个值 (放入高16位)
        if (current_m + 1 < max_m) {
            val |= ((uint32_t)A[current_m + 1] << 16);
        }
        return val;
    }
    // -------------------------------------------------------------------------
    // 2.5 Compute 阶段 (AVX512 VNNI 版本)
    // 专门针对 Small Batch (N=1~4) 优化，避免 AMX setup 开销
    // 使用完全相同的 PackedB 数据结构
    void amx_bf16_matmul_packed_avx512(const uint16_t *A, const uint16_t *PackedB, float *C,
                                    int N, int M, int GlobalK, 
                                    int st, int end,
                                    int lda, int ldc) {
#if defined(__AMX_TILE__)
        int width_k = end - st;
        int num_m_blocks = ceil_div(M, TILE_K); // M / 32
        // 1. 遍历 Batch (N)
        // 由于是 AVX512 CPU 循环，这里 N 的循环在最外层比较简单
        for (int n_idx = 0; n_idx < N; ++n_idx) {
            
            // 2. 遍历 Output Features (K) - 步长 16 (对应 TILE_N / ZMM 宽度)
            for (int k_outer = 0; k_outer < width_k; k_outer += 16) {
                
                // 初始化累加器: 16 个 float 对应 16 个输出通道
                __m512 acc = _mm512_setzero_ps();
                
                // 计算当前 K Block 在 PackedB 中的逻辑索引
                int block_idx_k = k_outer / 16;
                
                // 3. 遍历 Reduction (M) - 步长 32 (对应 TILE_K)
                for (int m_outer = 0; m_outer < M; m_outer += 32) {
                    // 定位 PackedB 的块位置
                    // Layout: [K_blocks][M_blocks][1024 bytes]
                    long long current_block_offset = (long long)block_idx_k * num_m_blocks + (m_outer / 32);
                    
                    // 获取当前 Block 的 Weight 指针 (转换为 byte 指针方便步进 64 字节)
                    const uint8_t* w_ptr = (const uint8_t*)PackedB + current_block_offset * TILE_B_SIZE_BYTES;
                    
                    // 定位 Input A 的指针
                    const uint16_t* a_ptr_base = &A[n_idx * lda + m_outer];
                    
                    // 处理 32 个 M 元素 (packed block 内部有 16 行，每行处理 2 个 M)
                    // 检查 A 读取是否会越界 (针对 M 不能被 32 整除的情况)
                    if (M - m_outer >= 32) {
                        // --- Fast Path: M 剩余部分足够一个 Block ---
                        #pragma unroll(4)
                        for (int j = 0; j < 16; ++j) {
                            // 1. Load B: 加载 64 字节 (16 个通道 x 2 个 BF16)
                            //    这正好对应 AMX Tile 的一行
                            __m512i v_weight = _mm512_load_si512((const void*)(w_ptr + j * 64));
                            
                            // 2. Load & Broadcast A: 读取 A 中 m, m+1 的一对值，并广播到所有 lane
                            //    A[m] 对应低16位, A[m+1] 对应高16位
                            uint32_t a_pair = *(uint32_t*)(a_ptr_base + j * 2);
                            __m512i v_input = _mm512_set1_epi32(a_pair);
                            
                            // 3. Dot Product: BF16 乘加
                            //    acc[k] += input_low * weight_low[k] + input_high * weight_high[k]
                            acc = _mm512_dpbf16_ps(acc, (__m512bh)v_input, (__m512bh)v_weight);
                        }
                    } else {
                        // --- Slow Path: M 剩余部分不足 32 ---
                        // PackedB 是 padding 0 的，所以只需保证 A 读取不越界
                        for (int j = 0; j < 16; ++j) {
                            int current_m_offset = j * 2;
                            
                            // 这里的 Weight 读取是安全的，因为 PackedB 总是按 Block 分配的
                            __m512i v_weight = _mm512_load_si512((const void*)(w_ptr + j * 64));
                            
                            // 安全读取 A
                            uint32_t a_pair = get_A_pair_safe(a_ptr_base, current_m_offset, M - m_outer);
                            __m512i v_input = _mm512_set1_epi32(a_pair);
                            
                            acc = _mm512_dpbf16_ps(acc, (__m512bh)v_input, (__m512bh)v_weight);
                        }
                    }
                }
                
                // 4. 存回 C
                // 处理 K 维度边界 (如果 K 不能被 16 整除)
                int k_remain = (width_k - k_outer);
                if (k_remain >= 16) {
                    _mm512_storeu_ps(&C[n_idx * ldc + st + k_outer], acc);
                } else {
                    __mmask16 mask = (1 << k_remain) - 1;
                    _mm512_mask_storeu_ps(&C[n_idx * ldc + st + k_outer], mask, acc);
                }
            }
        }
#else
        printf("Unsupport AMX.\n");
        exit(0);
#endif
    }
    // -------------------------------------------------------------------------
    // 3. 原始接口封装
    // 在这里我们演示如何组合 Repack 和 Compute。
    // *注意*：为了性能最佳，应将 amx_pack_weight 移至模型加载阶段，
    // 而在此函数中直接复用已打包的内存。
    bool LinearBFloat16BFloat16_AMX_Kernel(uint16_t *inputData, uint16_t *weightData, float *biasData, float *outputData,
                        int n, int m, int k, int st, int end) {
#if defined(__AMX_TILE__)
        // 1. 计算需要的 buffer 大小
        // M 维度按 32 对齐，K 维度 (end-st) 按 16 对齐
        int width_k = end - st;
        int m_blocks = ceil_div(m, TILE_K);
        int k_blocks = ceil_div(width_k, TILE_N);
        size_t packed_size_bytes = (size_t)m_blocks * k_blocks * TILE_B_SIZE_BYTES;

        // 2. 分配对齐的内存 (用于Packed B)
        // 实际场景中，这里应该检查 input Weight 是否已经是 Packing 过的格式
        uint16_t *packedB = (uint16_t *)aligned_alloc(64, packed_size_bytes);
        if (!packedB) return false;

        // 3. 调用 Repack
        amx_pack_weight(weightData, packedB, m, k, st, end, m);

        // 4. 调用 Compute (不再需要在内层循环里反复处理 B)
        // amx_bf16_matmul_packed(inputData, packedB, outputData, n, m, k, st, end, m, k);

        if (n <= 8) {
            amx_bf16_matmul_packed_avx512(inputData, packedB, outputData, n, m, k, st, end, m, k);
        } else {
            amx_bf16_matmul_packed(inputData, packedB, outputData, n, m, k, st, end, m, k);
        }


        // 5. 添加偏置
        AddBiasAVX512(outputData, biasData, n, k, st, end);

        // 6. 释放临时 buffer
        free(packedB);
        
        return true;
#else
        printf("Unsupport AMX.\n");
        exit(0);
        return false;
#endif
    }
}
