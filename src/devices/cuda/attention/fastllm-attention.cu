#include "fastllm-attention-common.cuh"

template <int BN, int BM, int BK>
__global__ void HalfFC(
    half * __restrict__ a, half * __restrict__ b, half * __restrict__ c,
    const int N, const int M, const int K,
    half scale, const int base) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 // support tensor core
    int tid = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int wid = tid >> 5;

    int stN = bx * BN;
    int stK = by * BK;
    int wrap0 = wid >> 1;
    int wrap1 = wid & 1;

    if (base + stN + BN <= stK) {
        return;
    }

    __shared__ half cur[BN][BK];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[4][8];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> frag_b[4][8];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[4][4];

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            wmma::load_matrix_sync(frag_a[i][j], &a[(stN + wrap0 * 64 + i * 16) * M + j * 16], M);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            wmma::load_matrix_sync(frag_b[i][j], &b[(stK + wrap1 * 64 + i * 16) * M + j * 16], M);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            #pragma unroll
            for (int k = 0; k < 8; k++) {
                wmma::mma_sync(frag_c[i][j], frag_a[i][k], frag_b[j][k], frag_c[i][j]);
            }
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::store_matrix_sync(&cur[(wrap0 * 64 + i * 16)][(wrap1 * 64 + j * 16)], frag_c[i][j], BK, wmma::mem_row_major);
        }
    }
    __syncthreads();

    for (int i = 0; i < BN; i++) {
        if (base + stN + i < stK + tid) {
            cur[i][tid] = (half)0;
        }
    }

    for (int i = 0; i < BN; i++) {
        c[(stN + i) * K + stK + tid] = __hmul(cur[i][tid], scale);
    }
#endif
}

void GpuQK(half *q, half *k, half *qk, int qlen, int klen, int dim, float scale, int base) {    
    const int BQ = 128, BK = 128, DIM = 128;
    dim3 blockDim(128);
    int BX = (qlen + BQ - 1) / BQ;
    int BY = (klen + BK - 1) / BK;
    dim3 gridDim(BX, BY);
    HalfFC <BQ, DIM, BK> <<<gridDim, blockDim>>> (q, k, qk, qlen, dim, klen, (half)scale, base);
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmHalfMatMulTransBBatchKernel(uint8_t** pointer, float alpha) {
    int id = blockIdx.x;
    half *input0 = (half*)pointer[id * 8 + 0];
    half *input1 = (half*)pointer[id * 8 + 1];
    half *output = (half*)pointer[id * 8 + 2];
    int n = (int)((size_t)pointer[id * 8 + 3]);
    int m = (int)((size_t)pointer[id * 8 + 4]);
    int k = (int)((size_t)pointer[id * 8 + 5]);
    int input0Stride = (int)((size_t)pointer[id * 8 + 6]);
    int input1Stride = (int)((size_t)pointer[id * 8 + 7]);

    int tid = threadIdx.x;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 
    if (m == 128) {
        int wid = tid >> 5;
        int perN = 8, perK = 128;

        const int BN = 8, BK = 128;
        __shared__ float curC[BN][BK];
        half hscale = (half)alpha;

        for (int stN = 0; stN < n; stN += perN) {
            int endN = min(n, stN + perN);
            for (int stK = 0; stK < k; stK += perK) {
                int endK = min(k, stK + perK);
                wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a[8];
                wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::col_major> frag_b[8];
                wmma::fragment<wmma::accumulator, 8, 32, 16, float> frag_c;

                wmma::fill_fragment(frag_c, 0.0);
                __syncthreads();

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    wmma::load_matrix_sync(frag_a[j], &input0[(stN) * input0Stride + j * 16], input0Stride);
                }
                __syncthreads();

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    wmma::load_matrix_sync(frag_b[j], &input1[(stK + wid * 32) * input1Stride + j * 16], input1Stride);
                }
                __syncthreads();

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    wmma::mma_sync(frag_c, frag_a[j], frag_b[j], frag_c);
                }
                __syncthreads();

                wmma::store_matrix_sync(&curC[0][wid * 32], frag_c, BK, wmma::mem_row_major);
                __syncthreads();

                if (stK + tid < endK) {
                    for (int i = 0; stN + i < endN; i++) {
                        output[(stN + i) * k + stK + tid] = (half)(curC[i][tid] * alpha);
                    }
                }
                __syncthreads();
            }
        }
        return;
    }
#endif
    int pera = 4, perb = 4;
    half cura[4][4], curb[4][4];
    float curc[4][4];
    int cnta = (n - 1) / pera + 1, cntb = (k - 1) / perb + 1;
    for (int taskId = tid; taskId < cnta * cntb; taskId += THREAD_PER_BLOCK) {
        int taska = taskId / cntb, taskb = taskId % cntb;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                curc[i][j] = 0.0f;
            }
        }
        for (int l = 0; l < m; l += 4) {
            for (int a = taska * pera; a < (taska + 1) * pera && a < n; a++) {
                FETCH_FLOAT2(cura[a - taska * pera]) = FETCH_FLOAT2(input0[a * input0Stride + l]);
            }
            for (int b = taskb * perb; b < (taskb + 1) * perb && b < k; b++) {
                FETCH_FLOAT2(curb[b - taskb * perb]) = FETCH_FLOAT2(input1[b * input1Stride + l]);
            }

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        curc[i][j] += (float)cura[i][k] * (float)curb[j][k];
                    }
                }
            }
        }

        if ((taska + 1) * pera <= n && (taskb + 1) * perb <= k) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = (half)(curc[i][j] * alpha);
                }
            }
        } else {
            for (int i = 0; i < pera && taska * pera + i < n; i++) {
                for (int j = 0; j < perb && taskb * perb + j < k; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = (half)(curc[i][j] * alpha);
                }
            }
        }
    }
/*
    int tid = threadIdx.x;
    for (int i = 0; i < n; i++) {
        half *curInput0 = input0 + i * input0Stride;
        for (int j = tid; j < k; j += THREAD_PER_BLOCK) {
            half *curInput1 = input1 + j * input1Stride;
            float sum = 0.0;
            for (int l = 0; l < m; l++) {
                sum += (float)curInput0[l] * (float)curInput1[l];
            }
            output[i * k + j] = (half)(sum * alpha);
        }
    }
*/
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmMatMulTransBBatchKernel(uint8_t** pointer, float alpha) {
    int id = blockIdx.x;
    float *input0 = (float*)pointer[id * 8 + 0];
    float *input1 = (float*)pointer[id * 8 + 1];
    float *output = (float*)pointer[id * 8 + 2];
    int n = (int)((size_t)pointer[id * 8 + 3]);
    int m = (int)((size_t)pointer[id * 8 + 4]);
    int k = (int)((size_t)pointer[id * 8 + 5]);
    int input0Stride = (int)((size_t)pointer[id * 8 + 6]);
    int input1Stride = (int)((size_t)pointer[id * 8 + 7]);

    int tid = threadIdx.x;
    int pera = 4, perb = 4;
    float cura[4][4], curb[4][4], curc[4][4];
    int cnta = (n - 1) / pera + 1, cntb = (k - 1) / perb + 1;
    for (int taskId = tid; taskId < cnta * cntb; taskId += THREAD_PER_BLOCK) {
        int taska = taskId / cntb, taskb = taskId % cntb;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                cura[i][j] = 0;
                curb[i][j] = 0;
                curc[i][j] = 0;
            }
        }

        for (int l = 0; l < m; l += 4) {
            for (int a = taska * pera; a < (taska + 1) * pera && a < n; a++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    cura[a - taska * pera][x] = input0[a * input0Stride + l + x];
                }
            }
            for (int b = taskb * perb; b < (taskb + 1) * perb && b < k; b++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    curb[b - taskb * perb][x] = input1[b * input1Stride + l + x];
                }
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        curc[i][j] += cura[i][k] * curb[j][k];
                    }
                }
            }
        }

        if ((taska + 1) * pera <= n && (taskb + 1) * perb <= k) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = curc[i][j] * alpha;
                }
            }
        } else {
            for (int i = 0; i < pera && taska * pera + i < n; i++) {
                for (int j = 0; j < perb && taskb * perb + j < k; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = curc[i][j] * alpha;
                }
            }
        }
    }

/*
    int tid = threadIdx.x;
    for (int i = 0; i < n; i++) {
        float *curInput0 = input0 + i * input0Stride;
        for (int j = tid; j < k; j += THREAD_PER_BLOCK) {
            float *curInput1 = input1 + j * input1Stride;
            float sum = 0.0;
            for (int l = 0; l < m; l++) {
                sum += curInput0[l] * curInput1[l];
            }
            output[i * k + j] = sum * alpha;
        }
    }
*/
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmHalfMatMulKernel(uint8_t** pointer, float alpha) {
    int id = blockIdx.x;
    half *input0 = (half*)pointer[id * 8 + 0];
    half *input1 = (half*)pointer[id * 8 + 1];
    half *output = (half*)pointer[id * 8 + 2];
    int n = (int)((size_t)pointer[id * 8 + 3]);
    int m = (int)((size_t)pointer[id * 8 + 4]);
    int k = (int)((size_t)pointer[id * 8 + 5]);
    int input0Stride = (int)((size_t)pointer[id * 8 + 6]);
    int input1Stride = (int)((size_t)pointer[id * 8 + 7]);
    int tid = threadIdx.x;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 
    if (k == 128) {
        int wid = tid >> 5;
        int perN = 8, perM = 128;
        for (int i = 0; i < n; i++) {
            output[i * k + tid] = (half)0;
        }

        __shared__ half curA[8][128];
        __shared__ float curC[8][128];

        for (int stN = 0; stN < n; stN += perN) {
            int endN = min(stN + perN, n);
            wmma::fragment<wmma::accumulator, 8, 32, 16, float> frag_c;
            wmma::fill_fragment(frag_c, 0.0);

            for (int stM = 0; stM < m; stM += perM) {
                int endM = min(stM + perM, m);
                if (stM + tid < m) {
                    for (int i = 0; stN + i < endN; i++) {
                        curA[i][tid] = input0[(stN + i) * input0Stride + stM + tid];
                    }
                } else {
                    for (int i = 0; stN + i < endN; i++) {
                        curA[i][tid] = (half)0.0;
                    }
                }

                wmma::fragment<wmma::matrix_a, 8, 32, 16, half, wmma::row_major> frag_a[8];
                wmma::fragment<wmma::matrix_b, 8, 32, 16, half, wmma::row_major> frag_b[8];
                __syncthreads();

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    wmma::load_matrix_sync(frag_a[j], &curA[0][16 * j], 128);
                }
                __syncthreads();

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    wmma::load_matrix_sync(frag_b[j], &input1[(stM + 16 * j) * input1Stride + wid * 32], input1Stride);
                }
                __syncthreads();

                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    wmma::mma_sync(frag_c, frag_a[j], frag_b[j], frag_c);
                }
                __syncthreads();
            }
            wmma::store_matrix_sync(&curC[0][wid * 32], frag_c, 128, wmma::mem_row_major);
            __syncthreads();

            for (int i = 0; stN + i < endN; i++) {
                output[(stN + i) * k + tid] = (half)((float)output[(stN + i) * k + tid] + (float)curC[i][tid] * alpha);
            }
            __syncthreads();
        }
        return;
    }
#endif
    int pera = 4, perb = 4;
    float cura[4][4], curb[4][4], curc[4][4];
    int cnta = (n - 1) / pera + 1, cntb = (k - 1) / perb + 1;
    for (int taskId = tid; taskId < cnta * cntb; taskId += THREAD_PER_BLOCK) {
        int taska = taskId / cntb, taskb = taskId % cntb;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                cura[i][j] = 0;
                curb[i][j] = 0;
                curc[i][j] = 0;
            }
        }

        for (int l = 0; l < m; l += 4) {
            for (int a = taska * pera; a < (taska + 1) * pera && a < n; a++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    cura[a - taska * pera][x] = (l + x < m ? (float)input0[a * input0Stride + l + x] : 0.f);
                }
            }
            for (int b = taskb * perb; b < (taskb + 1) * perb && b < k; b++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    curb[b - taskb * perb][x] = (l + x < m ? (float)input1[(l + x) * input1Stride + b] : 0.f);
                }
            }

#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        curc[i][j] += cura[i][k] * curb[j][k];
                    }
                }
            }
        }

        if ((taska + 1) * pera <= n && (taskb + 1) * perb <= k) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = (half)(curc[i][j] * alpha);
                }
            }
        } else {
            for (int i = 0; i < pera && taska * pera + i < n; i++) {
                for (int j = 0; j < perb && taskb * perb + j < k; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = (half)(curc[i][j] * alpha);
                }
            }
        }
    }
/*
    for (int i = 0; i < n; i++) {
        half *curInput0 = input0 + i * input0Stride;
        for (int j = tid; j < k; j += THREAD_PER_BLOCK) {
            half *curInput1 = input1 + j;
            float sum = 0.0;
            for (int l = 0; l < m; l++) {
                sum += (float)curInput0[l] * (float)curInput1[l * input1Stride];
            }
            output[i * k + j] = (half)(sum * alpha);
        }
    }
*/
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmMatMulKernel(uint8_t** pointer, float alpha) {
    int id = blockIdx.x;
    float *input0 = (float*)pointer[id * 8 + 0];
    float *input1 = (float*)pointer[id * 8 + 1];
    float *output = (float*)pointer[id * 8 + 2];
    int n = (int)((size_t)pointer[id * 8 + 3]);
    int m = (int)((size_t)pointer[id * 8 + 4]);
    int k = (int)((size_t)pointer[id * 8 + 5]);
    int input0Stride = (int)((size_t)pointer[id * 8 + 6]);
    int input1Stride = (int)((size_t)pointer[id * 8 + 7]);

    int tid = threadIdx.x;
    int pera = 4, perb = 4;
    float cura[4][4], curb[4][4], curc[4][4];
    int cnta = (n - 1) / pera + 1, cntb = (k - 1) / perb + 1;
    for (int taskId = tid; taskId < cnta * cntb; taskId += THREAD_PER_BLOCK) {
        int taska = taskId / cntb, taskb = taskId % cntb;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                cura[i][j] = 0;
                curb[i][j] = 0;
                curc[i][j] = 0;
            }
        }

        for (int l = 0; l < m; l += 4) {
            for (int a = taska * pera; a < (taska + 1) * pera && a < n; a++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    cura[a - taska * pera][x] = l + x < m ? input0[a * input0Stride + l + x] : 0;
                }
            }
            for (int b = taskb * perb; b < (taskb + 1) * perb && b < k; b++) {
#pragma unroll
                for (int x = 0; x < 4; x++) {
                    curb[b - taskb * perb][x] = l + x < m ? input1[(l + x) * input1Stride + b] : 0;
                }
            }

#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        curc[i][j] += cura[i][k] * curb[j][k];
                    }
                }
            }
        }

        if ((taska + 1) * pera <= n && (taskb + 1) * perb <= k) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = curc[i][j] * alpha;
                }
            }
        } else {
            for (int i = 0; i < pera && taska * pera + i < n; i++) {
                for (int j = 0; j < perb && taskb * perb + j < k; j++) {
                    output[(taska * pera + i) * k + (taskb * perb + j)] = curc[i][j] * alpha;
                }
            }
        }
    }

/*
    //int tid = threadIdx.x;
    for (int i = 0; i < n; i++) {
        float *curInput0 = input0 + i * input0Stride;
        for (int j = tid; j < k; j += THREAD_PER_BLOCK) {
            float *curInput1 = input1 + j;
            float sum = 0.0;
            for (int l = 0; l < m; l++) {
                sum += curInput0[l] * curInput1[l * input1Stride];
            }
            output[i * k + j] = sum * alpha;
        }
    }
*/
}

template <int THREAD_PER_BLOCK>
__global__ void SimpleMask(float* a, float *b, float maskValue, int spatial) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < spatial) {
        if (b[i] > 0.99) {
            a[i] = maskValue;
        }
    }
}

template <int THREAD_PER_BLOCK>
__global__ void SimpleMask(half* a, half *b, half maskValue, int spatial) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < spatial) {
        if (__half2float(b[i]) > 0.99) {
            a[i] = maskValue;
        }
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmAttentionMaskKernel(float* a, float *b, float maskValue, int n, int m, int spatial) {
    int on = blockIdx.x / m;
    int om = blockIdx.x % m;
    int o = on * m + om;
    int idx = threadIdx.x;
    for (int i = idx; i < spatial; i += THREAD_PER_BLOCK) {
        if (b[on * spatial + i] > 0.99) {
            a[o * spatial + i] = maskValue;
        }
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmAttentionMaskKernel(half *a, half *b, half maskValue, int n, int m, int spatial) {
    int on = blockIdx.x / m;
    int om = blockIdx.x % m;
    int o = on * m + om;
    int idx = threadIdx.x;
    for (int i = idx; i < spatial; i += THREAD_PER_BLOCK) {
        if (__half2float(b[on * spatial + i]) > 0.99) {
            a[o * spatial + i] = maskValue;
        }
    }
}

template <int THREAD_PER_BLOCK, typename T>
__global__ void CausalMask(T* a, T maskValue, int q, int k, int base) {
    a += blockIdx.x * k;
    for (int i = base + blockIdx.x + threadIdx.x + 1; i < k; i += THREAD_PER_BLOCK) {
        a[i] = maskValue;
    }
}

__global__ void InitBlockAtten(float *sum0, float *max0, float *sum1, float *max1, int len) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < len) {
        sum0[i] = sum1[i] = 0.0f;
        max0[i] = max1[i] = -10000.0f;
    }
}

template <int THREAD_PER_BLOCK>
__global__ void AttnBlockUpdate(half *data, int n, int m, float *lastMax, float *lastSum, float *curMax, float *curSum) {
    __shared__ float scale;
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;

    if (tid == 0) {
        float diff = fminf(lastMax[bid] - curMax[bid], 0.f);
        float oldSum = lastSum[bid] * expf(diff);
        scale = (curSum[bid] > 1e-10f) ? (oldSum / curSum[bid]) : 0.0f;

        lastSum[bid] = curSum[bid];
        lastMax[bid] = curMax[bid];
    }
    __syncthreads();

    for (int i = tid; i < m; i += THREAD_PER_BLOCK) {
        data[bid * m + i] = (half)((float)data[bid * m + i] * scale);
    }
}

template <int THREAD_PER_BLOCK>
__device__ void FastllmSoftmaxKernelInner1Func(float *input, float *output, int channels, float *maxp, float *sump) {
    __shared__ float sdata[THREAD_PER_BLOCK];
    __shared__ float maxV;

    // 1. 每个线程计算一部分
    unsigned int tid = threadIdx.x;
    float maxValue = -1e100;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        maxValue = max(maxValue, input[i]);
    }
    sdata[tid] = maxValue;
    __syncthreads();

    // 2. 求max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // 3. 记录max
    if (tid == 0) {
        maxV = sdata[0];
        if (maxp != nullptr) {
            maxp[0] = sdata[0];
        }
    }
    __syncthreads();

    // 4. 求和
    float sum = 0;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] = exp(input[i] - maxV);
        sum += output[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        if (fabs(sdata[0]) < 1e-6) {
            sdata[0] = 0.0001;
        }
        if (sump != nullptr) {
            sump[0] = sdata[0];
        }
    }
    __syncthreads();

    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] /= sdata[0];
    }
}

__device__ half FastllmHalfMaxFunc(const __half a, const __half b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530
    return __half2float(a) >= __half2float(b) ? a : b;
#else
#if defined(CUDART_VERSION) && CUDART_VERSION > 11000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __hmax(a, b);
#else
    return __hge(a, b) ? a : b;
#endif
#endif
}

template <int THREAD_PER_BLOCK>
__device__ void FastllmSoftmaxKernelInner1Func(half *input, half *output, int channels, float *maxp, float *sump) {
    __shared__ float sdata[THREAD_PER_BLOCK];

    // 1. 每个线程计算一部分
    unsigned int tid = threadIdx.x;
    float maxValue = -1e10;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        maxValue = max(maxValue, (float)input[i]);
    }
    sdata[tid] = maxValue;
    __syncthreads();

    // 2. 求max
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // 3. 记录max
    if (tid == 0) {
        if (maxp != nullptr) {
            sdata[0] = max(maxp[0], sdata[0]);
        }
    }
    __syncthreads();
    float maxV = sdata[0];
    __syncthreads();

    // 4. 求和
    float sum = 0;
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        sum = sum + exp((float)input[i] - maxV);
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        if (fabs(sdata[0]) < 1e-6) {
            sdata[0] = 0.0001;
        }
        if (sump != nullptr) {
            sump[0] = sump[0] * exp(maxp[0] - maxV) + sdata[0];
            sdata[0] = sump[0];
            maxp[0] = maxV;
        }
    }
    __syncthreads();

    float scale = 1.0 / sdata[0];
    for (int i = tid; i < channels; i += THREAD_PER_BLOCK) {
        output[i] = (half)(exp((float)input[i] - maxV) * scale);
    }
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmSoftmaxKernelInner1(float* input, float *output, int outer, int channels) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> (input + o * channels, output + o * channels, channels, nullptr, nullptr);
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmSoftmaxKernelInner1(half* input, half *output, int outer, int channels) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> (input + o * channels, output + o * channels, channels, nullptr, nullptr);
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmMaskedSoftmaxHalfKernel(
        half *scores, const half *mask, int queries, int keys,
        int headsPerMask, uint64_t maskBatchStride, uint64_t maskRowStride) {
    const int row = blockIdx.x;
    const int head = row / queries;
    const half *maskRow = mask + (head / headsPerMask) * maskBatchStride +
                          (row % queries) * maskRowStride;
    half *scoreRow = scores + (uint64_t)row * keys;
    for (int column = threadIdx.x; column < keys; column += THREAD_PER_BLOCK) {
        if (__half2float(maskRow[column]) > 0.99f) {
            scoreRow[column] = __float2half_rn(-10000.0f);
        }
    }
    __syncthreads();
    // Keep the existing FP16 score/probability boundaries and reduction tree.
    FastllmSoftmaxKernelInner1Func<THREAD_PER_BLOCK>(
        scoreRow, scoreRow, keys, nullptr, nullptr);
}

__global__ void FastllmMaskedAttentionPointersKernel(
        half **pointers, half *q, half *k, half *v, half *output, half *scores,
        int heads, int group, uint64_t queryStride, uint64_t keyStride,
        uint64_t valueStride, uint64_t outputStride, uint64_t scoreStride) {
    const int head = blockIdx.x * blockDim.x + threadIdx.x;
    if (head < heads) {
        pointers[head] = k + (head / group) * keyStride;
        pointers[heads + head] = q + head * queryStride;
        pointers[2 * heads + head] = scores + head * scoreStride;
        pointers[3 * heads + head] = v + (head / group) * valueStride;
        pointers[4 * heads + head] = output + head * outputStride;
    }
}

static bool TryBatchedMaskedHalfAttention(
        const fastllm::Data &q, const fastllm::Data &k,
        const fastllm::Data &v, const fastllm::Data &mask,
        const fastllm::Data &output, int group, float scale) {
    if (q.dims.size() != 3 || k.dims.size() != 3 || v.dims.size() != 3 ||
        (mask.dims.size() != 2 && mask.dims.size() != 3) ||
        mask.dataType != fastllm::DataType::FLOAT16 || mask.cudaData == nullptr ||
        q.dims[0] <= 0 || q.dims[1] <= 1 || q.dims[2] <= 0 ||
        group <= 0 || q.dims[0] != k.dims[0] * group ||
        v.dims[0] != k.dims[0] || v.dims[1] != k.dims[1] ||
        v.dims[2] <= 0 || output.dims != std::vector<int>({q.dims[0], q.dims[1], v.dims[2]}) ||
        q.dims[2] != k.dims[2] || k.dims[1] <= 0 ||
        q.strides.size() != 3 || k.strides.size() != 3 || v.strides.size() != 3 ||
        output.strides.size() != 3 || mask.strides.size() != mask.dims.size() ||
        q.strides[2] != 1 || k.strides[2] != 1 || v.strides[2] != 1 ||
        output.strides[2] != 1 ||
        q.strides[1] < q.dims[2] || k.strides[1] < k.dims[2] ||
        v.strides[1] < v.dims[2] || output.strides[1] < v.dims[2] ||
        mask.strides.back() != 1 || mask.dims.back() != k.dims[1] ||
        mask.dims[mask.dims.size() - 2] != q.dims[1] ||
        FastllmCudaGraphIsCapturing()) {
        return false;
    }
    const int heads = q.dims[0], queries = q.dims[1], keys = k.dims[1];
    const int batches = mask.dims.size() == 3 ? mask.dims[0] : 1;
    if (batches <= 0 || heads % batches != 0) return false;
    // Batch heads only while the complete score workspace is small. Large
    // prefill/long-KV shapes retain the bounded, per-head implementation.
    constexpr size_t scratchLimit = 8ULL * 1024 * 1024;
    const uint64_t scoreRowBytes = (uint64_t)keys * sizeof(half);
    const uint64_t rows = (uint64_t)heads * queries;
    if (rows > scratchLimit / scoreRowBytes) return false;
    const size_t pointerOffset = (rows * scoreRowBytes + 255) / 256 * 256;
    const size_t scratchBytes = pointerOffset + 5 * (size_t)heads * sizeof(half *);
    void *scratch = nullptr;
    if (FastllmCudaTryMalloc(&scratch, scratchBytes) !=
            FASTLLM_CUDA_TRY_MALLOC_SUCCESS || scratch == nullptr) return false;
    half *scores = (half *)scratch;
    half **pointers = (half **)((uint8_t *)scratch + pointerOffset);
    const half zero = __float2half_rn(0), one = __float2half_rn(1);
    const half hscale = __float2half_rn(scale);
    auto handle = getFastllmCublasHandle();
    FastllmMaskedAttentionPointersKernel<<<(heads + 127) / 128, 128>>>(
        pointers, (half *)q.cudaData, (half *)k.cudaData, (half *)v.cudaData,
        (half *)output.cudaData, scores, heads, group, q.strides[0],
        k.strides[0], v.strides[0], output.strides[0], (uint64_t)queries * keys);
    // Keep each head's GEMM shape instead of concatenating the GQA query
    // group. Pointer batching also respects independent K/V head capacities.
    auto status = cublasHgemmBatched(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        keys, queries, q.dims[2], &hscale,
        (const half **)pointers, k.strides[1],
        (const half **)(pointers + heads), q.strides[1],
        &zero, pointers + 2 * heads, keys, heads);
    if (status == CUBLAS_STATUS_SUCCESS) {
        const uint64_t maskBatchStride = mask.dims.size() == 3 ? mask.strides[0] : 0;
        const uint64_t maskRowStride = mask.strides[mask.dims.size() - 2];
#define FASTLLM_MASKED_SOFTMAX(THREADS) \
        FastllmMaskedSoftmaxHalfKernel<THREADS><<<rows, THREADS>>>( \
            scores, (const half *)mask.cudaData, queries, keys, heads / batches, \
            maskBatchStride, maskRowStride)
        if (keys < 8) { FASTLLM_MASKED_SOFTMAX(1); }
        else if (keys < 64) { FASTLLM_MASKED_SOFTMAX(8); }
        else if (keys < 512) { FASTLLM_MASKED_SOFTMAX(64); }
        else { FASTLLM_MASKED_SOFTMAX(256); }
#undef FASTLLM_MASKED_SOFTMAX
        status = cublasHgemmBatched(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            v.dims[2], queries, keys, &one,
            (const half **)(pointers + 3 * heads), v.strides[1],
            (const half **)(pointers + 2 * heads), keys,
            &zero, pointers + 4 * heads, output.strides[1], heads);
    }
    // The allocator may hand the returned buffer to another PTDS. Complete
    // its consumers before releasing it; graph capture keeps its old path.
    FastllmCudaSyncCurrentThreadStream();
    FastllmCudaFree(scratch);
    return status == CUBLAS_STATUS_SUCCESS;
}

template <int THREAD_PER_BLOCK>
__global__ void FastllmSoftmaxKernelInner1(half* input, half *output, int outer, int channels, float *maxp, float *sump) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> (input + o * channels, output + o * channels, channels, maxp + o, sump + o);
}

template <int THREAD_PER_BLOCK, typename T>
__global__ void FastllmSoftmaxKernelInner1WithCausalMask(T* input, T *output, int outer, int channels, int base) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> (input + o * channels, output + o * channels, o + base + 1, nullptr, nullptr);
}

template <int THREAD_PER_BLOCK, typename T>
__global__ void FastllmSoftmaxKernelInner1WithCausalMask(T* input, T *output, int outer, int channels, int base, float *maxp, float *sump) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> (input + o * channels, output + o * channels, min(channels, o + base + 1), maxp + o, sump + o);
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmSoftmaxKernelBatchInner1(uint8_t** pointer) {
    int o = blockIdx.x;
    FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> ((T*)pointer[o * 3], (T*)pointer[o * 3 + 1],
                                                       (int)((size_t)pointer[o * 3 + 2]), nullptr, nullptr);
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmSoftmaxKernelBatchInner1(uint8_t** pointer, int outer) {
    int o = blockIdx.x;
    int channels = (int)((size_t)pointer[o / outer * 2 + 1]);
    FastllmSoftmaxKernelInner1Func <THREAD_PER_BLOCK> ((T*)pointer[o / outer * 2] + (o % outer) * channels, (T*)pointer[o / outer * 2] + (o % outer) * channels,
                                                       channels, nullptr, nullptr);
}

template <typename T, int THREAD_PER_BLOCK>
__global__ void FastllmSoftmaxKernelBatchInner1WithCausalMask(
        uint8_t **pointer, int queryHeads, int queryLength) {
    int o = blockIdx.x;
    int rowsPerRequest = queryHeads * queryLength;
    int request = o / rowsPerRequest;
    int requestRow = o - request * rowsPerRequest;
    int queryRow = requestRow % queryLength;
    int channels = (int)((size_t)pointer[request * 2 + 1]);
    int validChannels = channels - queryLength + queryRow + 1;
    validChannels = max(0, min(channels, validChannels));
    T *row = (T*)pointer[request * 2] + (size_t)requestRow * channels;
    FastllmSoftmaxKernelInner1Func<THREAD_PER_BLOCK>(
        row, row, validChannels, nullptr, nullptr);
    for (int i = validChannels + threadIdx.x;
         i < channels; i += THREAD_PER_BLOCK) {
        row[i] = (T)0;
    }
}

bool FastllmCudaSoftmax(const fastllm::Data &input, fastllm::Data &output, int axis) {
    float *cudaInput = (float *) FastllmCudaPrepareInput(input);
    float *cudaOutput = (float *) FastllmCudaPrepareInput(output);

    int dimsLen = input.dims.size();
    axis = (axis % dimsLen + dimsLen) % dimsLen;
    int outer = input.Count(0) / input.Count(axis);
    int channels = input.dims[axis];
    int inner = input.Count(axis + 1);
    if (inner == 1) {
        if (input.dataType == fastllm::DataType::FLOAT32) {
            if (channels < 8) {
                FastllmSoftmaxKernelInner1 <1> <<< outer, 1 >>> (cudaInput, cudaOutput, outer, channels);
            } else if (channels < 64) {
                FastllmSoftmaxKernelInner1 <8> <<< outer, 8 >>> (cudaInput, cudaOutput, outer, channels);
            } else if (channels < 512) {
                FastllmSoftmaxKernelInner1 <64> <<< outer, 64 >>> (cudaInput, cudaOutput, outer, channels);
            } else {
                FastllmSoftmaxKernelInner1 <256> <<< outer, 256 >>> (cudaInput, cudaOutput, outer, channels);
            }
        } else {
            if (channels < 8) {
                FastllmSoftmaxKernelInner1 <1> <<< outer, 1 >>> ((half*)cudaInput, (half*)cudaOutput, outer, channels);
            } else if (channels < 64) {
                FastllmSoftmaxKernelInner1 <8> <<< outer, 8 >>> ((half*)cudaInput, (half*)cudaOutput, outer, channels);
            } else if (channels < 512) {
                FastllmSoftmaxKernelInner1 <64> <<< outer, 64 >>> ((half*)cudaInput, (half*)cudaOutput, outer, channels);
            } else {
                FastllmSoftmaxKernelInner1 <256> <<< outer, 256 >>> ((half*)cudaInput, (half*)cudaOutput, outer, channels);
            }
        }
    } else {
        printf("softmax error.\n");
        exit(0);
    }

    FastllmCudaFinishInput(input, cudaInput);
    FastllmCudaFinishOutput(output, cudaOutput);
    return true;
}

bool FastllmCudaSoftmaxBatch(fastllm::Data **inputs, fastllm::Data **outputs, int axis, int batch) {
    int total = 0;
    for (int b = 0; b < batch; b++) {
        auto &input = *inputs[b];
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int outer = input.Count(0) / input.Count(axis);
        total += outer;
    }
    uint8_t ** pointers = (uint8_t**)FastllmCudaMalloc(sizeof(uint8_t*) * total * 3);
    uint8_t ** cpuPointers = new uint8_t*[total * 3];
    int cur = 0;

    for (int b = 0; b < batch; b++) {
        auto &input = *inputs[b];
        auto &output = *outputs[b];
        float *cudaInput = (float *) input.cudaData;
        float *cudaOutput = (float *) output.cudaData;

        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        int outer = input.Count(0) / input.Count(axis);
        int channels = input.dims[axis];
        int inner = input.Count(axis + 1);

        if (inner == 1) {
            for (int o = 0; o < outer; o++) {
                cpuPointers[cur * 3 + 0] = (uint8_t*)(cudaInput + o * channels);
                cpuPointers[cur * 3 + 1] = (uint8_t*)(cudaOutput + o * channels);
                cpuPointers[cur * 3 + 2] = (uint8_t*)((size_t)channels);
                cur++;
            }
        } else {
            printf("softmax error.\n");
            exit(0);
        }
    }

    cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * total * 3, cudaMemcpyHostToDevice);
    FastllmSoftmaxKernelBatchInner1 <float, 256> <<<total, 256>>> (pointers);

    FastllmCudaFree(pointers);
    delete[] cpuPointers;
    DeviceSync();
    return true;
}

extern bool FastllmCudaPermute(fastllm::Data &input, const std::vector<int> &axis);

bool FastllmCudaHalfAttention(const fastllm::Data &q, const fastllm::Data &k, const fastllm::Data &v,
                              const fastllm::Data &mask, const fastllm::Data &output, int group, float scale, int maskType) {
#ifdef FASTLLM_ENABLE_FLASHINFER
    using namespace flashinfer;
#endif
    
    int q0 = q.dims[0], q1 = q.dims[1], q2 = q.dims[2], k0 = k.dims[0], k1 = k.dims[1], v2 = v.dims[2];
    half *qd = (half*)q.cudaData;
    half *kd = (half*)k.cudaData;
    half *vd = (half*)v.cudaData;
    half *maskd = mask.dims.size() > 0 ? (half*)mask.cudaData : nullptr;
    half *od = (half*)output.cudaData;
    int batch = (mask.dims.size() == 3) ? mask.dims[0] : 1;
    int maskStride = (mask.dims.size() == 3 ? mask.strides[0] : mask.Count(0));

    // 使用 FlashInfer 实现 attention
    
    uint32_t num_kv_heads = k0;          // KV heads 数（所有 batch 共享）

    // 最可能的情况：group 就是 group_size（GQA 的 group size）
    // 所以 num_qo_heads（每个 batch）= group * num_kv_heads
    uint32_t num_qo_heads = group * num_kv_heads;  // 每个 batch 的 Q heads 数 = group_size * num_kv_heads
    uint32_t actual_batch = q0 / num_qo_heads;     // batch 数 = q0 / (每个 batch 的 Q heads 数)
    
    // 验证参数有效性，避免除零错误
    if (num_kv_heads == 0) {
        printf("Error: num_kv_heads is 0 (k0=%d)\n", k0);
        return false;
    }
    if (num_qo_heads == 0) {
        printf("Error: num_qo_heads is 0 (group=%d, num_kv_heads=%u)\n", group, num_kv_heads);
        return false;
    }
    if (num_qo_heads % num_kv_heads != 0) {
        printf("Error: num_qo_heads (%u) is not divisible by num_kv_heads (%u), group=%d\n", 
               num_qo_heads, num_kv_heads, group);
        return false;
    }
    if (actual_batch == 0) {
        printf("Error: actual_batch is 0 (q0=%d, num_qo_heads=%u)\n", q0, num_qo_heads);
        return false;
    }
    if (q0 % num_qo_heads != 0) {
        printf("Error: q0 (%d) is not divisible by num_qo_heads (%u)\n", q0, num_qo_heads);
        return false;
    }
    uint32_t qo_len = q1;                // query 序列长度
    uint32_t kv_len = k1;                // key/value 序列长度
    uint32_t head_dim_qk = q2;           // QK head dimension
    uint32_t head_dim_vo = v2;           // VO head dimension
    
    // 确定 mask mode - FlashInfer 的 custom mask 需要 bit-packed 格式，暂时不支持
    const bool use_custom_mask = (maskd != nullptr);
// printf("maskType = %d, use_custom_mask = %d, batch = %d\n", maskType, use_custom_mask, batch);
#ifdef FASTLLM_ENABLE_FLASHINFER
    MaskMode mask_mode = MaskMode::kNone;
    if (maskType == 0 && !use_custom_mask && batch == 1) {
        mask_mode = MaskMode::kCausal;
    }
#endif
    // FlashInfer's custom mask is bit-packed and is not compatible with
    // FastLLM's dense mask.  Keep use_custom_mask=true so the selection below
    // falls back to the original attention implementation, which consumes the
    // dense sliding-window mask.  Clearing this flag would silently run causal
    // full attention and makes restored sliding KV tails numerically wrong.
    
    // FlashInfer 支持 HND 布局，使用 HND 布局实现
    // fastllm 的数据布局是 HND: [num_heads, seq_len, head_dim]
    // 对于 HND 布局：
    // - stride_n (token 之间的 stride) = head_dim
    // - stride_h (head 之间的 stride) = seq_len * head_dim
#ifdef FASTLLM_ENABLE_FLASHINFER
    // A single HND query has the same output layout as FlashInfer's NHD
    // output. Split long KV across CTAs instead of materializing QK and P.
    // Keep other shapes and graph capture on their existing paths.
    if (head_dim_qk == 256 && head_dim_vo == 256 && qo_len == 1 &&
        kv_len > 4096 && actual_batch == 1 && !use_custom_mask && maskType == 0 &&
        FastllmCudaFlashInferSupported() && !FastllmCudaGraphIsCapturing()) {
        // Only SM75 CTA16 uses compact FP16 storage. Match the dispatcher's
        // architecture gate and preserve the original minimum on other GPUs.
        using MinSplitTraits = KernelTraits<MaskMode::kCausal, 16, 1, 1, 16, 16, 1, 4,
            PosEncodingMode::kNone, half, half, half, float, int,
            DefaultAttention<false, false, false, false>, true>;
        int maxSharedMemory = 0;
        cudaError_t state = cudaDeviceGetAttribute(&maxSharedMemory,
            cudaDevAttrMaxSharedMemoryPerBlockOptin, FastllmCudaGetDevice());
        if (state != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA split attention device query: ") +
                                     cudaGetErrorString(state));
        }
        // The single-prefill planner uses chunks of at least 256 tokens.
        // Each chunk stores one output vector and an FP32 LSE per Q head.
        const size_t maxChunks = ((size_t)kv_len + 255) / 256;
        const size_t bytesPerChunk = (size_t)num_qo_heads *
                                    (256 * sizeof(half) + sizeof(float));
        const auto capability = GetCudaComputeCapability();
        const size_t minimumSharedMemory =
            use_sm75_single_prefill_vo_split(capability.first, capability.second,
                                             num_qo_heads / num_kv_heads)
                ? sizeof(MinSplitTraits::SharedStorageSingle)
                : sizeof(MinSplitTraits::SharedStorage);
        if ((size_t)maxSharedMemory >= minimumSharedMemory &&
            maxChunks <= std::numeric_limits<size_t>::max() / bytesPerChunk) {
            void *scratch = nullptr;
            auto allocation = FastllmCudaTryMalloc(&scratch, maxChunks * bytesPerChunk);
            if (allocation == FASTLLM_CUDA_TRY_MALLOC_ERROR) {
                throw std::runtime_error("CUDA error allocating split attention workspace");
            }
            if (scratch != nullptr) {
                auto release = [](half *ptr) {
                    // The allocator can hand this buffer to another thread.
                    // Finish the merge before returning it to the pool.
                    FastllmCudaSyncCurrentThreadStream();
                    FastllmCudaFree(ptr);
                };
                std::unique_ptr<half, decltype(release)> tmp((half*)scratch, release);
                SinglePrefillParams<half, half, half> params(
                    qd, kd, vd, nullptr, od, nullptr, nullptr,
                    num_qo_heads, num_kv_heads, 1, kv_len,
                    q.strides[1], q.Count(1), k.strides[1], k.Count(1),
                    256, -1, 0.0f, scale, 1.0f, 10000.0f);
                // K and V can have different physical capacities after
                // expansion/rollback; do not derive either stride from kv_len.
                params.v_stride_n = v.strides[1];
                params.v_stride_h = v.Count(1);
                cudaError_t status = SinglePrefillWithKVCacheDispatched<
                    256, 256, PosEncodingMode::kNone, false, MaskMode::kCausal,
                    DefaultAttention<false, false, false, false>>(
                        params, tmp.get(), cudaStreamPerThread);
                if (status != cudaSuccess) {
                    throw std::runtime_error(std::string("FlashInfer split attention: ") +
                                             cudaGetErrorString(status));
                }
                return true;
            }
        }
    }
    bool use_flashinfer = (head_dim_qk == 128 && head_dim_vo == 128 && !use_custom_mask) &&
                          FastllmCudaFlashInferSupported();
#else
    bool use_flashinfer = false;
#endif
// use_flashinfer = false;
    // 调试信息：打印参数值
    if (use_flashinfer) {
        // printf("FlashInfer params: q0=%d, q1=%d, q2=%d, k0=%d, k1=%d, v2=%d, group=%d, batch=%d\n", q0, q1, q2, k0, k1, v2, group, batch);
        // printf("  num_kv_heads=%u, num_qo_heads=%u, actual_batch=%u, qo_len=%u, kv_len=%u\n", num_kv_heads, num_qo_heads, actual_batch, qo_len, kv_len);
    }
    
#ifdef FASTLLM_ENABLE_FLASHINFER
    if (use_flashinfer) {
        // 为每个 batch item 调用 FlashInfer
        // q0 = batch * num_qo_heads，所以需要按 batch 循环
        for (int batch_idx = 0; batch_idx < actual_batch; batch_idx++) {
            // 准备参数
            // q 的布局: [batch*num_qo_heads, seq_len, head_dim] (HND)
            // 对于单个 batch，需要 group 个 heads 的数据
            // cur_q 指向 batch_idx * group 个 heads 的起始位置
            half *cur_q = qd + batch_idx * group * q.Count(1);
            half *cur_k = kd + batch_idx * k.Count(1);
            half *cur_v = vd + batch_idx * v.Count(1);
            half *cur_o = od + batch_idx * group * output.Count(1);
            
            // 对于 HND 布局 [num_heads, seq_len, head_dim]:
            // - stride_n (token 之间的 stride) = head_dim
            // - stride_h (head 之间的 stride) = seq_len * head_dim
            uint32_t q_stride_n = q.strides[1];    // head_dim (token 之间的 stride)
            uint32_t q_stride_h = q.Count(1);      // seq_len * head_dim (head 之间的 stride)
            
            // k/v 也是 HND 布局: [num_kv_heads, kv_len, head_dim]
            uint32_t kv_stride_n = k.strides[1];   // head_dim (token 之间的 stride)
            uint32_t kv_stride_h = k.Count(1);     // kv_len * head_dim (head 之间的 stride)

            // 验证 stride 值
            if (q_stride_n == 0 || q_stride_h == 0 || kv_stride_n == 0 || kv_stride_h == 0) {
                printf("Error: Invalid stride values: q_stride_n=%u, q_stride_h=%u, kv_stride_n=%u, kv_stride_h=%u\n",
                       q_stride_n, q_stride_h, kv_stride_n, kv_stride_h);
                use_flashinfer = false;
                break;
            }
            
            // 验证序列长度
            if (qo_len == 0 || kv_len == 0) {
                printf("Error: Invalid sequence lengths: qo_len=%u, kv_len=%u\n", qo_len, kv_len);
                use_flashinfer = false;
                break;
            }
            
            // 再次验证，避免运行时除零
            uint32_t expected_group_size = num_qo_heads / num_kv_heads;
            if (expected_group_size == 0) {
                printf("Error: expected_group_size is 0 (num_qo_heads=%u, num_kv_heads=%u)\n",
                       num_qo_heads, num_kv_heads);
                use_flashinfer = false;
                break;
            }
            
            // 分配临时缓冲区（如果需要 partition-kv）
            half *tmp = nullptr;
            cudaError_t status = cudaSuccess;
            // Pad/Split, the FastLLM allocator, and the consumers below all
            // use the per-thread default stream. Launch FlashInfer on that
            // same stream so a completed host call also preserves producer /
            // consumer ordering when the temporary head buffers are reused.
            cudaStream_t stream = cudaStreamPerThread;
            
            {
                // Prefill 阶段：q 的形状是 [num_qo_heads, qo_len, head_dim]
                // 创建 SinglePrefillParams (使用 HND 布局)
                SinglePrefillParams<half, half, half> params(
                    cur_q, cur_k, cur_v, nullptr,  // q, k, v, custom_mask (暂时不支持)
                    cur_o, nullptr, nullptr,        // o, lse, alibi_slopes
                    num_qo_heads,                   // num_qo_heads (每个 batch 的 Q heads 数)
                    num_kv_heads,                   // num_kv_heads (KV heads 数)
                    qo_len,                         // qo_len
                    kv_len,                         // kv_len
                    q_stride_n,                     // q_stride_n (token stride for HND = head_dim)
                    q_stride_h,                     // q_stride_h (head stride for HND = seq_len * head_dim)
                    kv_stride_n,                    // k_stride_n (token stride for HND = head_dim)
                    kv_stride_h,                    // k_stride_h (head stride for HND = kv_len * head_dim)
                    head_dim_qk,                    // head_dim
                    -1,                             // window_left (-1 means no sliding window)
                    0.0f,                           // logits_soft_cap
                    scale,                          // sm_scale
                    1.0f,                           // rope_scale (不使用 RoPE)
                    10000.0f                        // rope_theta (不使用 RoPE)
                );
                
                // 调用 FlashInfer prefill 接口，根据 mask_mode 选择不同的 variant
                if (mask_mode == MaskMode::kCausal) {
                    status = SinglePrefillWithKVCacheDispatched<128, 128, PosEncodingMode::kNone, false, MaskMode::kCausal, DefaultAttention<false, false, false, false>>(
                        params, tmp, stream);
                } else {
                    status = SinglePrefillWithKVCacheDispatched<128, 128, PosEncodingMode::kNone, false, MaskMode::kNone, DefaultAttention<false, false, false, false>>(
                        params, tmp, stream);
                }
            }
            
            if (tmp != nullptr) {
                FastllmCudaFree(tmp);
            }
            
            if (status != cudaSuccess) {
                printf("FlashInfer error: %s\n", cudaGetErrorString(status));
                // Fallback 到原始实现
                use_flashinfer = false;
                break;
            }
((fastllm::Data*)&output)->Resize({output.dims[1], output.dims[0], output.dims[2]});
FastllmCudaPermute(*((fastllm::Data*)&output), {1, 0, 2});
        }
        
        if (use_flashinfer) {
            DeviceSync();
            return true;
        }
    }
#endif
    
    // Fallback 到原始实现
    half beta = __float2half_rn(0.0f), one = __float2half_rn(1.0f), hscale = __float2half_rn(scale);

    if (use_custom_mask && TryBatchedMaskedHalfAttention(q, k, v, mask, output, group, scale)) {
        return true;
    }

    // Vision self-attention is non-causal and can have tens of thousands of
    // queries.  The legacy fallback below processes one head at a time, but it
    // still materializes a full [q1, k1] score matrix (8 GiB at 65536 x 65536
    // in FP16).  Split only the query rows instead: every row still sees the
    // complete K/V sequence, so this is numerically the same softmax while the
    // temporary allocation stays bounded.  This path uses only cuBLAS and
    // ordinary CUDA kernels and therefore also works on SM70 where FlashInfer
    // is unavailable.
    if (!use_custom_mask && maskType == 2 && q1 >= 1024 && k1 >= 1024) {
        constexpr size_t scratchLimit = 32ULL * 1024ULL * 1024ULL;
        const size_t scoreRowBytes = (size_t)k1 * sizeof(half);
        int queryChunk = (int)std::max<size_t>(
            1, std::min<size_t>((size_t)q1, scratchLimit / scoreRowBytes));
        size_t scoreBytes = 0;
        half *qk = nullptr;
        // On a crowded SM70 card even the default 32 MiB may be unavailable.
        // Retry with progressively fewer query rows; one complete score row is
        // the minimum required to preserve exact softmax over all keys.
        while (queryChunk >= 1) {
            scoreBytes = (size_t)queryChunk * scoreRowBytes;
            void *scratch = nullptr;
            FastllmCudaTryMallocResult allocResult =
                FastllmCudaTryMalloc(&scratch, scoreBytes);
            qk = (half *)scratch;
            if (allocResult == FASTLLM_CUDA_TRY_MALLOC_SUCCESS &&
                qk != nullptr) {
                break;
            }
            if (qk != nullptr) {
                FastllmCudaFree(qk);
            }
            qk = nullptr;
            if (queryChunk == 1) {
                break;
            }
            queryChunk = std::max(1, queryChunk / 2);
        }
        if (qk != nullptr) {
            auto fastllmCublasHandle = getFastllmCublasHandle();
            cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
            for (int i = 0; i < q0; i++) {
                const half *curK =
                    kd + (size_t)(i / group) * k.Count(1);
                const half *curV =
                    vd + (size_t)(i / group) * v.Count(1);
                for (int queryStart = 0; queryStart < q1;
                     queryStart += queryChunk) {
                    const int queryRows =
                        std::min(queryChunk, q1 - queryStart);
                    const half *curQ = qd + (size_t)i * q.Count(1) +
                        (size_t)queryStart * q.strides[1];
                    half *curOutput = od + (size_t)i * output.Count(1) +
                        (size_t)queryStart * output.strides[1];

                    status = cublasHgemm(
                        fastllmCublasHandle,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        k1, queryRows, q2, &hscale,
                        curK, k.strides[1],
                        curQ, q.strides[1],
                        &beta, qk, k1);
                    if (status != CUBLAS_STATUS_SUCCESS) {
                        FastllmCudaSyncCurrentThreadStream();
                        FastllmCudaFree(qk);
                        throw std::runtime_error(
                            "cuBLAS failed during chunked non-causal QK");
                    }

                    FastllmSoftmaxKernelInner1<256>
                        <<<queryRows, 256>>>(
                            qk, qk, queryRows, k1);
                    status = cublasHgemm(
                        fastllmCublasHandle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        v2, queryRows, k1, &one,
                        curV, v.strides[1],
                        qk, k1,
                        &beta, curOutput, output.strides[1]);
                    if (status != CUBLAS_STATUS_SUCCESS) {
                        FastllmCudaSyncCurrentThreadStream();
                        FastllmCudaFree(qk);
                        throw std::runtime_error(
                            "cuBLAS failed during chunked non-causal PV");
                    }
                }
            }
            // qk is pooled and may be reused by another request thread. Make
            // the cuBLAS/kernels on this PTDS complete before returning it.
            FastllmCudaSyncCurrentThreadStream();
            FastllmCudaFree(qk);
            return true;
        }
        throw std::runtime_error(
            "CUDA non-causal attention could not allocate even one bounded "
            "score row; refusing the quadratic-memory fallback");
    }

    if (q1 >= 1024 || (q1 > 1 && q1 != k1 && k1 >= 1024)) {
        int alignQ1 = q1, alignK1 = k1;
        int part = alignK1;
        bool useFastAttn = getCudaInfos()->hasTensorCore && batch == 1 &&
                           (q2 == 128 && v2 == 128) && maskType == 0 &&
                           maskd == nullptr;
        useFastAttn &= (q1 % 1024 == 0 && k1 % 1024 == 0);

        if (useFastAttn) {
            alignQ1 = ((q1 - 1) / 128 + 1) * 128;
            alignK1 = ((k1 - 1) / 128 + 1) * 128;
            part = (alignK1 > 8192 ? 8192 : alignK1);
        }
        const size_t qkBytes =
            (size_t)alignQ1 * (size_t)part * sizeof(half);
        half *qk = (half *)FastllmCudaMalloc(qkBytes);

        cudaMemset(qk, 0, qkBytes);
        auto fastllmCublasHandle = getFastllmCublasHandle();
        cublasStatus_t status;
        for (int i = 0; i < q0; i++) {
//DeviceSync();
//auto st = std::chrono::system_clock::now();
            if (useFastAttn) { 
                if (alignK1 > 8192) {
                    float *lastSum = (float*)FastllmCudaMalloc(alignQ1 * sizeof(float));
                    float *lastMax = (float*)FastllmCudaMalloc(alignQ1 * sizeof(float));
                    float *currentSum = (float*)FastllmCudaMalloc(alignQ1 * sizeof(float));
                    float *currentMax = (float*)FastllmCudaMalloc(alignQ1 * sizeof(float));

                    int threadPerBlock = std::min(256, alignQ1);
                    InitBlockAtten <<< (alignQ1 - 1) / threadPerBlock + 1, threadPerBlock>>> (lastSum, lastMax, currentSum, currentMax, alignQ1);

                    int part = 8192;
                    for (int st = 0; st < alignK1; st += part) {
                        int len = std::min(part, alignK1 - st);
                        status = cublasHgemm(fastllmCublasHandle,
                                            CUBLAS_OP_T, CUBLAS_OP_N,
                                            len, alignQ1, q2, &hscale,
                                            kd + (i / group) * k.Count(1) + st * k.strides[1], k.strides[1],
                                            qd + i * q.Count(1), q.strides[1],
                                            &beta, 
                                            qk, len);
                        CausalMask<256, half> <<<q1, 256>>>(qk, __float2half_rn(0.0f), alignQ1, len, k1 - q1 - st);
                        FastllmSoftmaxKernelInner1WithCausalMask<256> <<< q1, 256 >>>(qk, qk, alignQ1, len, k1 - q1 - st, currentMax, currentSum);
                        if (st > 0) {
                            AttnBlockUpdate <128> <<< alignQ1, 128 >>> (od + i * v2 * q1, alignQ1, v2, lastMax, lastSum, currentMax, currentSum);
                        } else {
                            cudaMemcpy(lastMax, currentMax, alignQ1 * sizeof(float), cudaMemcpyDeviceToDevice);
                            cudaMemcpy(lastSum, currentSum, alignQ1 * sizeof(float), cudaMemcpyDeviceToDevice);
                        }
                        half currentScale = __float2half_rn(st > 0 ? 1.0f : 0.0f);
                        status = cublasHgemm(fastllmCublasHandle,
                                            CUBLAS_OP_N, CUBLAS_OP_N,
                                            v2, alignQ1, len, &one,
                                            vd + (i / group) * v.Count(1) + st * v.strides[1], v.strides[1],
                                            qk, len,
                                            &currentScale,
                                            od + i * v2 * q1, v2);
                    }

                    FastllmCudaFree(lastSum);
                    FastllmCudaFree(lastMax);
                    FastllmCudaFree(currentSum);
                    FastllmCudaFree(currentMax);
                } else {
                    GpuQK(qd + i * q.Count(1), kd + (i / group) * k.Count(1), qk, alignQ1, alignK1, q2, scale, k1 - q1);
                    FastllmSoftmaxKernelInner1WithCausalMask<128> <<< q1, 128 >>>(qk, qk, q1, alignK1, k1 - q1);
                    status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                                CUBLAS_OP_N, CUBLAS_OP_N,
                                                v2, q1, alignK1, &one,
                                                vd + (i / group) * v.Count(1), v.strides[1], v.Count(1),
                                                qk, alignK1, alignK1 * alignQ1,
                                                &beta,
                                                od + i * v2 * q1, v2, v2 * q1, 1);
                }
            } else {
                status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                                CUBLAS_OP_T, CUBLAS_OP_N,
                                                k1, q1, q2, &hscale,
                                                kd + (i / group) * k.Count(1), k.strides[1], k.Count(1),
                                                qd + i * q.Count(1), q.strides[1], q.Count(1),
                                                &beta,
                                                qk, k1, k1 * q1, 1);
                if (status != CUBLAS_STATUS_SUCCESS) {
                    printf("status = %d\n", (int) status);
                    printf("Error: cublas error during MatMulTransB in Attention operator.\n");
                    throw ("cublas error");
                    exit(0);
                }

                if (batch == 1 && maskd == nullptr && maskType == 0) {
                    CausalMask<256, half> <<<q1, 256>>>(qk, __float2half_rn(0), q1, k1, k1 - q1);
                    FastllmSoftmaxKernelInner1WithCausalMask<128> <<< q1, 128 >>>(qk, qk, q1, k1, k1 - q1);
                } else {
                    if (maskd != nullptr) {
                        SimpleMask<256> <<< (q1 * k1 / 256) + 1, 256>>>(qk, maskd + (i / (q0 / batch)) * maskStride, __float2half_rn(-10000), q1 * k1);
                    }

                    int outer = q1;
                    if (k1 < 8) {
                        FastllmSoftmaxKernelInner1<1> <<< outer, 1 >>>(qk, qk, outer, k1);
                    } else if (k1 < 64) {
                        FastllmSoftmaxKernelInner1<8> <<< outer, 8 >>>(qk, qk, outer, k1);
                    } else if (k1 < 512) {
                        FastllmSoftmaxKernelInner1<64> <<< outer, 64 >>>(qk, qk, outer, k1);
                    } else {
                        FastllmSoftmaxKernelInner1<256> <<< outer, 256 >>>(qk, qk, outer, k1);
                    }
                }

                status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                               CUBLAS_OP_N, CUBLAS_OP_N,
                                               v2, q1, k1, &one,
                                               vd + (i / group) * v.Count(1), v.strides[1], v.Count(1),
                                               qk, k1, k1 * q1,
                                               &beta,
                                               od + i * v2 * q1, v2, v2 * q1, 1);
            }

//DeviceSync(); printf("softmax spend %f s.\n", GetSpan(st, std::chrono::system_clock::now()));
/*DeviceSync();
int n = k1, m = q1, k = q2;
float spend = GetSpan(st, std::chrono::system_clock::now());
float gops = (float)n * m * k * 4 / spend / 1e9;
printf("n = %d, m = %d, k = %d, spend %f s, gops = %f\n", n, m, k, spend, gops);*/
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("status = %d\n", (int) status);
                printf("Error: cublas error during MatMul in Attention operator.\n");
                throw ("cublas error");
                exit(0);
            }
        }

        FastllmCudaFree(qk);
        DeviceSync();
        return true;
    }

    if (true) {
        half *qk = (half *) FastllmCudaMalloc(q0 * q1 * k1 * sizeof(half));
        half *temp = (half *) FastllmCudaMalloc(q0 * q1 * k1 * sizeof(half));
        if (qk == nullptr || temp == nullptr) {
            FastllmCudaFree(qk);
            FastllmCudaFree(temp);
            // Let all TP ranks reach the capture abort without passing null to cuBLAS.
            if (FastllmCudaGraphIsCapturingFast()) return false;
            throw std::runtime_error("CUDA half attention could not allocate score workspace");
        }
        auto fastllmCublasHandle = getFastllmCublasHandle();
        cublasStatus_t status;

        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_T, CUBLAS_OP_N,
                                           k1, q1 * group, q2, &hscale,
                                           kd, k.strides[1], k.Count(1),
                                           qd, q.strides[1], q.Count(1) * group,
                                           &beta,
                                           qk, k1, k1 * q1 * group, q0 / group);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("status = %d\n", (int) status);
            printf("Error: cublas error during MatMulTransB in Attention operator.\n");
            throw ("cublas error");
            exit(0);
        }

        if (maskd) {
            int spatial = q1 * k1, n = batch, m = q0 / batch;
            FastllmAttentionMaskKernel <256> <<< n * m, 256>>>(qk, maskd, __float2half_rn(-10000), n, m, spatial);
        } else if (batch == 1 && maskType == 0 && q1 > 1) {
            // 没有显式 mask 且为 prefill（q1>1）时按因果方式屏蔽未来 token。
            // qk 物理布局为 [q0, q1, k1]，对每个 head 单独应用因果 mask。
            // base=k1-q1 使 query 行 r 可见 key [0, k1-q1+r]（兼容带历史上下文的分块 prefill）。
            for (int h = 0; h < q0; h++) {
                CausalMask<256, half> <<< q1, 256 >>>(qk + (size_t)h * q1 * k1, __float2half_rn(-10000.0f), q1, k1, k1 - q1);
            }
        }

        int outer = q0 * q1;
        if (k1 < 8) {
            FastllmSoftmaxKernelInner1<1> <<< outer, 1 >>>(qk, temp, outer, k1);
        } else if (k1 < 64) {
            FastllmSoftmaxKernelInner1<8> <<< outer, 8 >>>(qk, temp, outer, k1);
        } else if (k1 < 512) {
            FastllmSoftmaxKernelInner1<64> <<< outer, 64 >>>(qk, temp, outer, k1);
        } else {
            FastllmSoftmaxKernelInner1<256> <<< outer, 256 >>>(qk, temp, outer, k1);
        }

        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_N, CUBLAS_OP_N,
                                           v2, q1 * group, k1, &one,
                                           vd, v.strides[1], v.Count(1),
                                           temp, k1, k1 * q1 * group,
                                           &beta,
                                           od, v2, v2 * q1 * group, q0 / group);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("status = %d\n", (int) status);
            printf("Error: cublas error during MatMul in Attention operator.\n");
            throw ("cublas error");
            exit(0);
        }
        FastllmCudaFree(qk);
        FastllmCudaFree(temp);
        DeviceSync();
        return true;
    }
    return true;
}

__global__ void FastllmDFlashImplicitAttentionMaskKernel(
        half *scores, int heads, int queries, int keys, int cachedTokens,
        int runtimeBlockSize, int slidingWindow) {
    int head = blockIdx.x;
    if (head >= heads) {
        return;
    }
    half *headScores = scores + (size_t)head * queries * keys;
    for (int query = 0; query < queries; query++) {
        // Cached key positions are [committed-cached, committed). The dense
        // reference masks distance >= window, so the masked prefix includes
        // cached + query - window itself.
        int prefix = max(0, cachedTokens + query - slidingWindow + 1);
        for (int key = threadIdx.x; key < prefix; key += blockDim.x) {
            headScores[(size_t)query * keys + key] =
                __float2half_rn(-10000.0f);
        }
        for (int key = cachedTokens + runtimeBlockSize + threadIdx.x;
             key < keys; key += blockDim.x) {
            headScores[(size_t)query * keys + key] =
                __float2half_rn(-10000.0f);
        }
    }
}

#ifdef FASTLLM_ENABLE_FLASHINFER
namespace fastllm_dflash_attention {
// Reuse FlashInfer's attention and split-KV merge kernels. DFlash is
// bidirectional within the draft block; inactive draft
// slots are masked, and each query has its own sliding-window boundary.
struct Params : flashinfer::SinglePrefillParams<half, half, half> {
    int runtimeBlockSize;
    int slidingWindow;
};

template <bool SyncOutput = false>
struct Attention : flashinfer::DefaultAttention<false, true, false, false> {
    // Opt into the shared-reduction/output-stage synchronization required by
    // the SM75 Q32 tile. Generic prefill attention policies stay unchanged.
    static constexpr bool dflash_sync_output = SyncOutput;
    template <class P>
    __host__ __device__ Attention(const P &p, unsigned batch, uint8_t *smem)
        : flashinfer::DefaultAttention<false, true, false, false>(p, batch, smem) {}
    REGISTER_LOGITS_MASK(p, batch, qi, ki, qh, kh, {
        int cached = int(p.kv_len) - int(p.qo_len);
        int distance = int(ki) - (cached + int(qi));
        return ki < unsigned(cached + p.runtimeBlockSize) && distance > -p.slidingWindow &&
               distance < p.slidingWindow;
    })
};

__global__ void ToHnd(const uint4 *input, uint4 *output, int heads, int queries) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < heads * queries * 16) {
        int d = i % 16, q = (i / 16) % queries, h = i / (16 * queries);
        output[i] = input[(q * heads + h) * 16 + d];
    }
}

template <int TileQ = 64>
inline cudaError_t Run(const half *q, const half *k, const half *v, half *out, half *scratch,
                       int heads, int kvHeads, int queries, int keys, int kStrideH, int vStrideH,
                       int runtimeBlock, int window, float scale, cudaStream_t stream) {
    // SM75 uses 32 query rows and two KV warps. B8 with GQA=4
    // exactly fills that tile; it avoids the 64-row tile's inactive Q work.
    // FlashInfer's half MMA uses m16n8k8 and synchronous shared loads on
    // SM75. Keep FP32 QK/softmax accumulation and the custom window mask.
    constexpr int MmaKV = 2, Chunk = 128;
    int trim = std::max(0, keys - queries - window + 1);
    k += trim * 128;
    v += trim * 128;
    keys -= trim;
    constexpr int WQ = TileQ / 16, WK = 4 / WQ;
    using Traits = flashinfer::KernelTraits<flashinfer::MaskMode::kCustom, TileQ, 1, MmaKV, 8, 8,
                                            WQ, WK, flashinfer::PosEncodingMode::kNone, half, half,
                                            half, float, int, Attention<TileQ == 32>>;
    // IsInvalid also prunes Q=32 for the *generic* prefill tile selector,
    // which never selects it for head_dim=128. This fixed DFlash tile is
    // instantiated directly: two Q warps, two KV warps, one MMA Q tile,
    // FP16 operands and a bounded register/shared-memory footprint.
    static_assert(TileQ == 32 ?
        (Traits::NUM_THREADS == 128 && Traits::NUM_MMA_Q == 1 &&
         Traits::HEAD_DIM_QK == 128 && Traits::HEAD_DIM_VO == 128 &&
         Traits::NUM_MMA_Q * (8 * Traits::NUM_MMA_D_VO_TILE +
                             2 * sizeof(float) * MmaKV) < 256) :
        !Traits::IsInvalid());
    static_assert(sizeof(typename Traits::SharedStorageSingle) <= 49152);
    int splits = (keys + Chunk - 1) / Chunk;
    half *nhd = scratch, *partition = scratch + heads * queries * 128;
    Params p;
    static_cast<flashinfer::SinglePrefillParams<half, half, half> &>(p) =
        flashinfer::SinglePrefillParams<half, half, half>(
            const_cast<half *>(q), const_cast<half *>(k), const_cast<half *>(v), nullptr,
            splits > 1 ? partition : nhd, nullptr, nullptr, heads, kvHeads, queries, keys, 128,
            queries * 128, 128, kStrideH, 128, window - 1, 0, scale, 1, 10000);
    p.v_stride_h = vStrideH;
    p.runtimeBlockSize = runtimeBlock;
    p.slidingWindow = window;
    p.partition_kv = splits > 1;
    p.lse = splits > 1 ? reinterpret_cast<float *>(partition + splits * heads * queries * 128)
                       : nullptr;
    flashinfer::SinglePrefillWithKVCacheKernel<Traits, Params>
        <<<dim3((queries * (heads / kvHeads) + TileQ - 1) / TileQ, splits, kvHeads),
           dim3(32, WQ, WK), sizeof(typename Traits::SharedStorageSingle), stream>>>(p);
    auto status = cudaGetLastError();
    if (status != cudaSuccess)
        return status;
    if (splits > 1) {
        status = flashinfer::MergeStates(partition, p.lse, nhd, nullptr, splits, queries, heads,
                                         128, stream);
        if (status != cudaSuccess)
            return status;
    }
    ToHnd<<<(heads * queries * 16 + 255) / 256, 256, 0, stream>>>(
        reinterpret_cast<uint4 *>(nhd), reinterpret_cast<uint4 *>(out), heads, queries);
    return cudaGetLastError();
}
} // namespace fastllm_dflash_attention
#endif

bool FastllmCudaDFlashAttention(
        const fastllm::Data &q, const fastllm::Data &k,
        const fastllm::Data &v, fastllm::Data &output,
        int group, float scale, int runtimeBlockSize, int slidingWindow) {
    if (q.dataType != fastllm::DataType::FLOAT16 ||
        k.dataType != fastllm::DataType::FLOAT16 ||
        v.dataType != fastllm::DataType::FLOAT16 ||
        output.dataType != fastllm::DataType::FLOAT16 ||
        q.dataDevice != fastllm::DataDevice::CUDA ||
        k.dataDevice != fastllm::DataDevice::CUDA ||
        v.dataDevice != fastllm::DataDevice::CUDA ||
        output.dataDevice != fastllm::DataDevice::CUDA ||
        q.cudaData == nullptr || k.cudaData == nullptr ||
        v.cudaData == nullptr || output.cudaData == nullptr ||
        q.dims.size() != 3 || k.dims.size() != 3 ||
        v.dims.size() != 3 || output.dims.size() != 3 ||
        group <= 0 || runtimeBlockSize <= 0 ||
        runtimeBlockSize > q.dims[1] || slidingWindow <= 0 ||
        q.dims[0] != k.dims[0] * group ||
        k.dims[0] != v.dims[0] || k.dims[1] != v.dims[1] ||
        q.dims[2] != 128 || k.dims[2] != 128 || v.dims[2] != 128 ||
        k.dims[1] < q.dims[1] ||
        output.dims !=
            std::vector<int>({q.dims[0], q.dims[1], v.dims[2]}) ||
        q.strides.size() != 3 || k.strides.size() != 3 ||
        v.strides.size() != 3 || output.strides.size() != 3 ||
        q.strides[2] != 1 || k.strides[2] != 1 ||
        v.strides[2] != 1 || output.strides[2] != 1 ||
        q.strides[1] != 128 || k.strides[1] != 128 ||
        v.strides[1] != 128 || output.strides[1] != 128 ||
        q.strides[0] != (uint64_t)q.dims[1] * 128 ||
        output.strides[0] != (uint64_t)output.dims[1] * 128) {
        return false;
    }
    int device = FastllmCudaGetDevice();
    auto onCurrentDevice = [device](const fastllm::Data &data) {
        return !data.dataDeviceIds.empty() &&
            data.dataDeviceIds[0] == device;
    };
    if (!onCurrentDevice(q) || !onCurrentDevice(k) ||
        !onCurrentDevice(v) || !onCurrentDevice(output)) {
        return false;
    }

    const int heads = q.dims[0];
    const int queries = q.dims[1];
    const int keys = k.dims[1];
    const int headDim = q.dims[2];
    const int valueDim = v.dims[2];
    const int cachedTokens = keys - queries;
#ifdef FASTLLM_ENABLE_FLASHINFER
    const auto capability = flashinfer::GetCudaComputeCapability();
    const bool isSm75 = capability.first == 7 && capability.second == 5;
    const char *attentionFlag = std::getenv("FASTLLM_DFLASH_ATTENTION");
    const bool useFusedAttention = attentionFlag ?
        std::strcmp(attentionFlag, "1") == 0 : (isSm75 || capability.first >= 8);
    const bool useSm75Tile = isSm75 && useFusedAttention;
    const bool useSm80Tile = capability.first >= 8 && useFusedAttention;
    // One switch controls both tiles: 0 restores cuBLAS, 1 opts in on
    // supported devices. The Q32 (SM75) and Q64 (SM80+) fused paths default
    // on. Unsupported shapes/architectures keep cuBLAS.
    if (queries > 0 && queries <= 16 && group <= 64 / queries &&
        heads > 0 && k.dims[0] <= 65535 &&
        slidingWindow >= queries && slidingWindow <= 4096 &&
        k.Count(1) <= std::numeric_limits<int>::max() &&
        v.Count(1) <= std::numeric_limits<int>::max() &&
        (useSm75Tile || useSm80Tile) &&
        FastllmCudaFlashInferSupported()) {
        // Crop keys invisible to every query, preserving the independent
        // physical K/V head strides after expansion or rollback.
        const size_t visibleKeys = std::min(cachedTokens, slidingWindow - 1) + queries;
        const size_t chunks = (visibleKeys + 127) / 128;
        const size_t rows = (size_t)heads * queries;
        const size_t workspaceBytes = rows * 128 * sizeof(half) * (chunks + 1) +
                                      rows * chunks * sizeof(float);
        size_t availableBytes = 0;
        bool own = false;
        half *workspace = (half*)FastllmBorrowCudaTempBuffer(workspaceBytes, &availableBytes, &own);
        if (workspace != nullptr && availableBytes >= workspaceBytes) {
            const cudaError_t state = useSm75Tile ?
                fastllm_dflash_attention::Run<32>(
                    (const half*)q.cudaData, (const half*)k.cudaData,
                    (const half*)v.cudaData, (half*)output.cudaData, workspace,
                    heads, k.dims[0], queries, keys, k.Count(1), v.Count(1),
                    runtimeBlockSize, slidingWindow, scale, cudaStreamPerThread) :
                fastllm_dflash_attention::Run<64>(
                    (const half*)q.cudaData, (const half*)k.cudaData,
                    (const half*)v.cudaData, (half*)output.cudaData, workspace,
                    heads, k.dims[0], queries, keys, k.Count(1), v.Count(1),
                    runtimeBlockSize, slidingWindow, scale, cudaStreamPerThread);
            FastllmReleaseCudaTempBuffer(workspace, own);
            if (state != cudaSuccess) {
                throw std::runtime_error(std::string("DFlash FlashInfer attention: ") +
                                         cudaGetErrorString(state));
            }
            static thread_local std::map<int, bool> loggedSm75;
            if (useSm75Tile && !loggedSm75[device]) {
                printf("[DFlash attention] SM75 FP16 tileQ=32 on GPU %d, queries=%d heads=%d\n",
                    device, queries, heads);
                loggedSm75[device] = true;
            }
            return true;
        }
        FastllmReleaseCudaTempBuffer(workspace, own);
    }
#endif
    const size_t scoreElements = (size_t)heads * queries * keys;
    const size_t scratchBytes = scoreElements * sizeof(half) * 2;
    size_t availableBytes = 0;
    bool scratchOwn = false;
    half *scratch = (half *)FastllmBorrowCudaTempBuffer(
        scratchBytes, &availableBytes, &scratchOwn);
    if (scratch == nullptr || availableBytes < scratchBytes) {
        FastllmReleaseCudaTempBuffer(scratch, scratchOwn);
        return false;
    }
    half *scores = scratch;
    half *probabilities = scratch + scoreElements;

    half zero = __float2half_rn(0.0f);
    half one = __float2half_rn(1.0f);
    half halfScale = __float2half_rn(scale);
    cublasHandle_t handle = getFastllmCublasHandle();
    cublasStatus_t cublasState = cublasHgemmStridedBatched(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        keys, queries * group, headDim, &halfScale,
        (const half *)k.cudaData, k.strides[1], k.Count(1),
        (const half *)q.cudaData, q.strides[1], q.Count(1) * group,
        &zero, scores, keys, (long long)keys * queries * group,
        heads / group);
    if (cublasState == CUBLAS_STATUS_SUCCESS) {
        FastllmDFlashImplicitAttentionMaskKernel<<<
            heads, 256, 0, cudaStreamPerThread>>>(
                scores, heads, queries, keys, cachedTokens,
                runtimeBlockSize, slidingWindow);
        if (keys < 8) {
            FastllmSoftmaxKernelInner1<1><<<
                heads * queries, 1, 0, cudaStreamPerThread>>>(
                    scores, probabilities, heads * queries, keys);
        } else if (keys < 64) {
            FastllmSoftmaxKernelInner1<8><<<
                heads * queries, 8, 0, cudaStreamPerThread>>>(
                    scores, probabilities, heads * queries, keys);
        } else if (keys < 512) {
            FastllmSoftmaxKernelInner1<64><<<
                heads * queries, 64, 0, cudaStreamPerThread>>>(
                    scores, probabilities, heads * queries, keys);
        } else {
            FastllmSoftmaxKernelInner1<256><<<
                heads * queries, 256, 0, cudaStreamPerThread>>>(
                    scores, probabilities, heads * queries, keys);
        }
        cublasState = cublasHgemmStridedBatched(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            valueDim, queries * group, keys, &one,
            (const half *)v.cudaData, v.strides[1], v.Count(1),
            probabilities, keys, (long long)keys * queries * group,
            &zero, (half *)output.cudaData, valueDim,
            (long long)valueDim * queries * group, heads / group);
    }
    cudaError_t cudaState = cudaPeekAtLastError();
    FastllmReleaseCudaTempBuffer(scratch, scratchOwn);
    if (cublasState != CUBLAS_STATUS_SUCCESS || cudaState != cudaSuccess) {
        if (cudaState != cudaSuccess) {
            cudaGetLastError();
        }
        return false;
    }
    return true;
}

bool FastllmCudaAttention(const fastllm::Data &q, const fastllm::Data &k, const fastllm::Data &v,
                          const fastllm::Data &mask, const fastllm::Data &output, int group, float scale, int maskType) {
    int q0 = q.dims[0], q1 = q.dims[1], q2 = q.dims[2], k0 = k.dims[0], k1 = k.dims[1], v2 = v.dims[2];
    float *qd = (float*)q.cudaData;
    float *kd = (float*)k.cudaData;
    float *vd = (float*)v.cudaData;
    float *maskd = mask.dims.size() > 0 ? (float*)mask.cudaData : nullptr;
    float *od = (float*)output.cudaData;
    int batch = (mask.dims.size() == 3) ? mask.dims[0] : 1;
    int maskStride = (mask.dims.size() == 3 ? mask.strides[0] : mask.Count(0));

    if (q1 >= 1024 || (q1 > 1 && q1 != k1 && k1 >= 1024)) {
        float *qk = (float *) FastllmCudaMalloc(q1 * k1 * sizeof(float));
        float beta = 0, one = 1;
        auto fastllmCublasHandle = getFastllmCublasHandle();
        cublasStatus_t status;


        for (int i = 0; i < q0; i++) {
            status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                               CUBLAS_OP_T, CUBLAS_OP_N,
                                               k1, q1, q2, &scale,
                                               kd + (i / group) * k.Count(1), k.strides[1], k.Count(1),
                                               qd + i * q.Count(1), q.strides[1], q.Count(1),
                                               &beta,
                                               qk, k1, k1 * q1, 1);
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("status = %d\n", (int) status);
                printf("Error: cublas error during MatMulTransB in Attention operator.\n");
                throw ("cublas error");
                exit(0);
            }

            if (batch == 1 && maskd == nullptr && maskType == 0) {
                CausalMask<256, float> <<<q1, 256>>>(qk, 0, q1, k1, k1 - q1);
                FastllmSoftmaxKernelInner1WithCausalMask<128> <<< q1, 128 >>>(qk, qk, q1, k1, k1 - q1);
            } else {
                if (maskd) {
                    SimpleMask<256> <<< (q1 * k1 / 256) + 1, 256>>>(qk, maskd + (i / (q0 / batch)) * maskStride, -10000, q1 * k1);
                }
                int outer = q1;
                if (k1 < 8) {
                    FastllmSoftmaxKernelInner1<1> <<< outer, 1 >>>(qk, qk, outer, k1);
                } else if (k1 < 64) {
                    FastllmSoftmaxKernelInner1<8> <<< outer, 8 >>>(qk, qk, outer, k1);
                } else if (k1 < 512) {
                    FastllmSoftmaxKernelInner1<64> <<< outer, 64 >>>(qk, qk, outer, k1);
                } else {
                    FastllmSoftmaxKernelInner1<256> <<< outer, 256 >>>(qk, qk, outer, k1);
                }
            }

            status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                               CUBLAS_OP_N, CUBLAS_OP_N,
                                               v2, q1, k1, &one,
                                               vd + (i / group) * v.Count(1), v.strides[1], v.Count(1),
                                               qk, k1, k1 * q1,
                                               &beta,
                                               od + i * v2 * q1, v2, v2 * q1, 1);
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("status = %d\n", (int) status);
                printf("Error: cublas error during MatMul in Attention operator.\n");
                throw ("cublas error");
                exit(0);
            }
        }

        FastllmCudaFree(qk);
        DeviceSync();
        return true;
    }

    if (true) {
        float *qk = (float *) FastllmCudaMalloc(q0 * q1 * k1 * sizeof(float));
        float *temp = (float *) FastllmCudaMalloc(q0 * q1 * k1 * sizeof(float));
        float beta = 0, one = 1;
        auto fastllmCublasHandle = getFastllmCublasHandle();
        cublasStatus_t status;

        status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_T, CUBLAS_OP_N,
                                           k1, q1 * group, q2, &scale,
                                           kd, k.strides[1], k.Count(1),
                                           qd, q.strides[1], q.Count(1) * group,
                                           &beta,
                                           qk, k1, k1 * q1 * group, q0 / group);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("status = %d\n", (int) status);
            printf("Error: cublas error during MatMulTransB in Attention operator.\n");
            throw ("cublas error");
            exit(0);
        }

        if (maskd) {
            int spatial = q1 * k1, n = batch, m = q0 / batch;
            FastllmAttentionMaskKernel <256> <<< n * m, 256>>>(qk, maskd, -10000, n, m, spatial);
        } else if (batch == 1 && maskType == 0 && q1 > 1) {
            // qk is laid out as [q0, q1, k1]. Apply the causal mask per head;
            // base=k1-q1 also covers chunked prefill with an existing KV cache.
            for (int h = 0; h < q0; h++) {
                CausalMask<256, float> <<< q1, 256 >>>(
                    qk + (size_t)h * q1 * k1, -10000.0f,
                    q1, k1, k1 - q1);
            }
        }

        int outer = q0 * q1;
        if (k1 < 8) {
            FastllmSoftmaxKernelInner1<1> <<< outer, 1 >>>(qk, temp, outer, k1);
        } else if (k1 < 64) {
            FastllmSoftmaxKernelInner1<8> <<< outer, 8 >>>(qk, temp, outer, k1);
        } else if (k1 < 512) {
            FastllmSoftmaxKernelInner1<64> <<< outer, 64 >>>(qk, temp, outer, k1);
        } else {
            FastllmSoftmaxKernelInner1<256> <<< outer, 256 >>>(qk, temp, outer, k1);
        }

        status = cublasSgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_N, CUBLAS_OP_N,
                                           v2, q1 * group, k1, &one,
                                           vd, v.strides[1], v.Count(1),
                                           temp, k1, k1 * q1 * group,
                                           &beta,
                                           od, v2, v2 * q1 * group, q0 / group);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("status = %d\n", (int) status);
            printf("Error: cublas error during MatMul in Attention operator.\n");
            throw ("cublas error");
            exit(0);
        }
        FastllmCudaFree(qk);
        FastllmCudaFree(temp);
        DeviceSync();
        return true;
    }
    return true;
}

namespace {
    constexpr int FASTLLM_LONG_KV_ATTENTION_MAX_BATCH = 32;
    constexpr int FASTLLM_LONG_KV_ATTENTION_MAX_QUERY = 8;

    struct FastllmLongKvAttentionBatchParams {
        const half *queries[FASTLLM_LONG_KV_ATTENTION_MAX_BATCH];
        const half *keys[FASTLLM_LONG_KV_ATTENTION_MAX_BATCH];
        const half *values[FASTLLM_LONG_KV_ATTENTION_MAX_BATCH];
        half *outputs[FASTLLM_LONG_KV_ATTENTION_MAX_BATCH];
        size_t queryHeadStrides[FASTLLM_LONG_KV_ATTENTION_MAX_BATCH];
        size_t keyHeadStrides[FASTLLM_LONG_KV_ATTENTION_MAX_BATCH];
        size_t valueHeadStrides[FASTLLM_LONG_KV_ATTENTION_MAX_BATCH];
        size_t outputHeadStrides[FASTLLM_LONG_KV_ATTENTION_MAX_BATCH];
        size_t scoreOffsets[FASTLLM_LONG_KV_ATTENTION_MAX_BATCH];
        int kvLengths[FASTLLM_LONG_KV_ATTENTION_MAX_BATCH];
        int requestOrder[FASTLLM_LONG_KV_ATTENTION_MAX_BATCH];
        int batch;
        int kvHeads;
        int group;
        int queryLength;
    };

    __global__ void FastllmLongKvAttentionSetupPointersKernel(
            FastllmLongKvAttentionBatchParams params, half *scores,
            const half **qkKeys, const half **qkQueries, half **qkScores,
            const half **pvValues, const half **pvScores, half **pvOutputs,
            uint8_t **softmaxPointers) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        int matrixCount = params.batch * params.kvHeads;
        if (index < matrixCount) {
            int orderedRequest = index / params.kvHeads;
            int kvHead = index - orderedRequest * params.kvHeads;
            int request = params.requestOrder[orderedRequest];
            int kvLength = params.kvLengths[request];
            half *requestScores = scores + params.scoreOffsets[request];

            qkKeys[index] = params.keys[request] +
                (size_t)kvHead * params.keyHeadStrides[request];
            qkQueries[index] = params.queries[request] +
                (size_t)kvHead * params.group *
                    params.queryHeadStrides[request];
            qkScores[index] = requestScores +
                (size_t)kvHead * params.group * params.queryLength * kvLength;
            pvValues[index] = params.values[request] +
                (size_t)kvHead * params.valueHeadStrides[request];
            pvScores[index] = qkScores[index];
            pvOutputs[index] = params.outputs[request] +
                (size_t)kvHead * params.group *
                    params.outputHeadStrides[request];
        }
        if (index < params.batch) {
            softmaxPointers[index * 2] = reinterpret_cast<uint8_t *>(
                scores + params.scoreOffsets[index]);
            softmaxPointers[index * 2 + 1] = reinterpret_cast<uint8_t *>(
                static_cast<uintptr_t>(params.kvLengths[index]));
        }
    }

    static bool FastllmLongKvAttentionBatchEnabled() {
        const char *env = std::getenv(
            "FASTLLM_QWEN35_MTP_LONG_KV_BATCH_ATTENTION");
        if (env == nullptr || env[0] == '\0') {
            return true;
        }
        return std::strcmp(env, "0") != 0 &&
               std::strcmp(env, "false") != 0 &&
               std::strcmp(env, "FALSE") != 0 &&
               std::strcmp(env, "off") != 0 &&
               std::strcmp(env, "OFF") != 0 &&
               std::strcmp(env, "disable") != 0 &&
               std::strcmp(env, "DISABLE") != 0;
    }

    static bool FastllmLongKvAttentionBatchExtendEnabled() {
        const char *env = std::getenv(
            "FASTLLM_QWEN35_MTP_LONG_KV_BATCH_EXTEND");
        if (env == nullptr || env[0] == '\0') {
            return true;
        }
        return std::strcmp(env, "0") != 0 &&
               std::strcmp(env, "false") != 0 &&
               std::strcmp(env, "FALSE") != 0 &&
               std::strcmp(env, "off") != 0 &&
               std::strcmp(env, "OFF") != 0 &&
               std::strcmp(env, "disable") != 0 &&
               std::strcmp(env, "DISABLE") != 0;
    }

    static bool TryFastllmCudaHalfLongKvAttentionBatch(
            fastllm::Data **q, fastllm::Data **k, fastllm::Data **v,
            fastllm::Data **mask, fastllm::Data **output,
            int group, float scale, int batch) {
        if (!FastllmLongKvAttentionBatchEnabled() || batch < 2 ||
            batch > FASTLLM_LONG_KV_ATTENTION_MAX_BATCH || group <= 0 ||
            q == nullptr || k == nullptr || v == nullptr ||
            output == nullptr) {
            return false;
        }

        FastllmLongKvAttentionBatchParams params{};
        params.batch = batch;
        params.group = group;
        int qHeads = -1;
        int kvHeads = -1;
        int headDim = -1;
        int valueDim = -1;
        int queryLength = -1;
        int minKvLength = -1;
        int maxKvLength = 0;
        size_t scoreElements = 0;
        std::map<int, std::vector<int> > requestsByKvLength;
        for (int b = 0; b < batch; b++) {
            if (q[b] == nullptr || k[b] == nullptr || v[b] == nullptr ||
                output[b] == nullptr ||
                (mask != nullptr && mask[b] != nullptr &&
                 !mask[b]->dims.empty()) ||
                q[b]->dataType != fastllm::DataType::FLOAT16 ||
                k[b]->dataType != fastllm::DataType::FLOAT16 ||
                v[b]->dataType != fastllm::DataType::FLOAT16 ||
                output[b]->dataType != fastllm::DataType::FLOAT16 ||
                q[b]->dataDevice != fastllm::DataDevice::CUDA ||
                k[b]->dataDevice != fastllm::DataDevice::CUDA ||
                v[b]->dataDevice != fastllm::DataDevice::CUDA ||
                output[b]->dataDevice != fastllm::DataDevice::CUDA ||
                q[b]->cudaData == nullptr || k[b]->cudaData == nullptr ||
                v[b]->cudaData == nullptr || output[b]->cudaData == nullptr ||
                q[b]->dims.size() != 3 || k[b]->dims.size() != 3 ||
                v[b]->dims.size() != 3 || output[b]->dims.size() != 3 ||
                q[b]->dims[1] <= 0 ||
                q[b]->dims[1] > FASTLLM_LONG_KV_ATTENTION_MAX_QUERY ||
                k[b]->dims[1] <= 4096 ||
                k[b]->dims[1] < q[b]->dims[1] ||
                k[b]->dims[1] != v[b]->dims[1] ||
                k[b]->dims[0] != v[b]->dims[0] ||
                q[b]->dims[0] != k[b]->dims[0] * group ||
                q[b]->dims[2] != k[b]->dims[2] ||
                output[b]->dims[0] != q[b]->dims[0] ||
                output[b]->dims[1] != q[b]->dims[1] ||
                output[b]->dims[2] != v[b]->dims[2] ||
                q[b]->strides.size() != 3 || k[b]->strides.size() != 3 ||
                v[b]->strides.size() != 3 || output[b]->strides.size() != 3 ||
                q[b]->strides[1] != (uint64_t)q[b]->dims[2] ||
                k[b]->strides[1] != (uint64_t)k[b]->dims[2] ||
                v[b]->strides[1] != (uint64_t)v[b]->dims[2] ||
                output[b]->strides[1] != (uint64_t)output[b]->dims[2] ||
                q[b]->strides[0] !=
                    (uint64_t)q[b]->dims[1] * q[b]->dims[2] ||
                output[b]->strides[0] !=
                    (uint64_t)output[b]->dims[1] * output[b]->dims[2]) {
                return false;
            }
            if (b == 0) {
                qHeads = q[b]->dims[0];
                kvHeads = k[b]->dims[0];
                headDim = q[b]->dims[2];
                valueDim = v[b]->dims[2];
                queryLength = q[b]->dims[1];
                minKvLength = k[b]->dims[1];
            } else if (q[b]->dims[0] != qHeads ||
                       k[b]->dims[0] != kvHeads ||
                       q[b]->dims[2] != headDim ||
                       v[b]->dims[2] != valueDim ||
                       q[b]->dims[1] != queryLength) {
                return false;
            }
            int kvLength = k[b]->dims[1];
            minKvLength = std::min(minKvLength, kvLength);
            maxKvLength = std::max(maxKvLength, kvLength);
            params.queries[b] = reinterpret_cast<const half *>(q[b]->cudaData);
            params.keys[b] = reinterpret_cast<const half *>(k[b]->cudaData);
            params.values[b] = reinterpret_cast<const half *>(v[b]->cudaData);
            params.outputs[b] = reinterpret_cast<half *>(output[b]->cudaData);
            params.queryHeadStrides[b] = q[b]->strides[0];
            params.keyHeadStrides[b] = k[b]->strides[0];
            params.valueHeadStrides[b] = v[b]->strides[0];
            params.outputHeadStrides[b] = output[b]->strides[0];
            params.scoreOffsets[b] = scoreElements;
            params.kvLengths[b] = kvLength;
            scoreElements +=
                (size_t)qHeads * queryLength * kvLength;
            requestsByKvLength[kvLength].push_back(b);
        }
        if (minKvLength <= 4096 || kvHeads <= 0 || qHeads <= 0 ||
            headDim <= 0 || valueDim <= 0 || qHeads != kvHeads * group) {
            return false;
        }
        params.kvHeads = kvHeads;
        params.queryLength = queryLength;
        if (queryLength > 1 &&
            !FastllmLongKvAttentionBatchExtendEnabled()) {
            return false;
        }

        struct KvLengthGroup {
            int kvLength;
            int matrixOffset;
            int matrixCount;
        };
        std::vector<KvLengthGroup> groups;
        int orderedRequest = 0;
        for (const auto &entry : requestsByKvLength) {
            int requestBegin = orderedRequest;
            for (int request : entry.second) {
                params.requestOrder[orderedRequest++] = request;
            }
            groups.push_back({entry.first, requestBegin * kvHeads,
                              (int)entry.second.size() * kvHeads});
        }

        const int matrixCount = batch * kvHeads;
        const size_t scoreBytes = scoreElements * sizeof(half);
        const size_t pointerOffset = (scoreBytes + 255) & ~(size_t)255;
        const size_t pointerCount = (size_t)matrixCount * 6 + batch * 2;
        const size_t scratchBytes = pointerOffset +
            pointerCount * sizeof(void *);
        size_t availableBytes = 0;
        bool scratchOwn = false;
        uint8_t *scratch = reinterpret_cast<uint8_t *>(
            FastllmBorrowCudaTempBuffer(
                scratchBytes, &availableBytes, &scratchOwn));
        if (scratch == nullptr || availableBytes < scratchBytes) {
            FastllmReleaseCudaTempBuffer(scratch, scratchOwn);
            return false;
        }
        half *scores = reinterpret_cast<half *>(scratch);
        half **devicePointers = reinterpret_cast<half **>(
            scratch + pointerOffset);
        const half **qkKeys = const_cast<const half **>(devicePointers);
        const half **qkQueries = qkKeys + matrixCount;
        half **qkScores = devicePointers + matrixCount * 2;
        const half **pvValues = const_cast<const half **>(
            devicePointers + matrixCount * 3);
        const half **pvScores = pvValues + matrixCount;
        half **pvOutputs = devicePointers + matrixCount * 5;
        uint8_t **softmaxPointers = reinterpret_cast<uint8_t **>(
            devicePointers + matrixCount * 6);

        int threads = 256;
        int setupCount = std::max(matrixCount, batch);
        FastllmLongKvAttentionSetupPointersKernel<<<
            (setupCount + threads - 1) / threads, threads, 0,
            cudaStreamPerThread>>>(
                params, scores, qkKeys, qkQueries, qkScores,
                pvValues, pvScores, pvOutputs, softmaxPointers);
        cudaError_t cudaState = cudaGetLastError();
        if (cudaState != cudaSuccess) {
            FastllmReleaseCudaTempBuffer(scratch, scratchOwn);
            return false;
        }

        half zero = __float2half_rn(0.0f);
        half one = __float2half_rn(1.0f);
        half halfScale = __float2half_rn(scale);
        cublasHandle_t handle = getFastllmCublasHandle();
        for (const KvLengthGroup &item : groups) {
            cublasStatus_t status = cublasHgemmBatched(
                handle, CUBLAS_OP_T, CUBLAS_OP_N,
                item.kvLength, queryLength * group, headDim, &halfScale,
                qkKeys + item.matrixOffset, headDim,
                qkQueries + item.matrixOffset, headDim,
                &zero, qkScores + item.matrixOffset, item.kvLength,
                item.matrixCount);
            if (status != CUBLAS_STATUS_SUCCESS) {
                FastllmReleaseCudaTempBuffer(scratch, scratchOwn);
                return false;
            }
        }
        if (queryLength == 1) {
            FastllmSoftmaxKernelBatchInner1<half, 256><<<
                batch * qHeads, 256, 0, cudaStreamPerThread>>>(
                    softmaxPointers, qHeads);
        } else {
            FastllmSoftmaxKernelBatchInner1WithCausalMask<half, 128><<<
                batch * qHeads * queryLength, 128, 0,
                cudaStreamPerThread>>>(
                    softmaxPointers, qHeads, queryLength);
        }
        cudaState = cudaGetLastError();
        if (cudaState != cudaSuccess) {
            FastllmReleaseCudaTempBuffer(scratch, scratchOwn);
            return false;
        }
        for (const KvLengthGroup &item : groups) {
            cublasStatus_t status = cublasHgemmBatched(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                valueDim, queryLength * group, item.kvLength, &one,
                pvValues + item.matrixOffset, valueDim,
                pvScores + item.matrixOffset, item.kvLength,
                &zero, pvOutputs + item.matrixOffset, valueDim,
                item.matrixCount);
            if (status != CUBLAS_STATUS_SUCCESS) {
                FastllmReleaseCudaTempBuffer(scratch, scratchOwn);
                return false;
            }
        }
        cudaState = cudaPeekAtLastError();
        FastllmReleaseCudaTempBuffer(scratch, scratchOwn);
        if (cudaState != cudaSuccess) {
            cudaGetLastError();
            return false;
        }

        static thread_local bool logged[2] = {false, false};
        int logKind = queryLength > 1 ? 1 : 0;
        if (!logged[logKind]) {
            int device = -1;
            cudaGetDevice(&device);
            printf("[Fastllm] grouped long-KV MTP attention enabled on GPU %d "
                   "(batch=%d, qHeads=%d, kvHeads=%d, headDim=%d, "
                   "query=%d, kv=%d..%d).\n",
                   device, batch, qHeads, kvHeads, headDim, queryLength,
                   minKvLength, maxKvLength);
            fflush(stdout);
            logged[logKind] = true;
        }
        return true;
    }
}

template <typename T>
bool DoFastllmCudaAttentionBatch(fastllm::Data **q, fastllm::Data **k, fastllm::Data **v,
                               fastllm::Data **mask, fastllm::Data **output, int group, float scale, int batch) {
    if (false) {
        half beta = __float2half_rn(0.0f), one = __float2half_rn(1.0f), hscale = __float2half_rn(scale);
        int q0 = q[0]->dims[0], q1 = q[0]->dims[1], q2 = q[0]->dims[2], k0 = k[0]->dims[0], k1 = k[0]->dims[1], v2 = v[0]->dims[2];
        for (int i = 0; i < batch; i++) {
            q1 = max(q1, q[i]->dims[1]);
            k1 = max(k1, k[i]->dims[1]);
        }

        half *allKeys = (half*) FastllmCudaMalloc(batch * k0 * k1 * q2 * sizeof(half));
        half *allValues = (half*) FastllmCudaMalloc(batch * k0 * k1 * v2 * sizeof(half));

        std::vector <void*> dsts, srcs;
        std::vector <size_t> dpitchs, spitchs, widths, heights;
        for (int i = 0; i < batch; i++) {
            dsts.push_back((uint8_t *) (allKeys + i * k0 * k1 * q2));
            dpitchs.push_back(k1 * q2 * sizeof(half));
            srcs.push_back(k[i]->cudaData);
            spitchs.push_back(k[i]->strides[0] * sizeof(half));
            widths.push_back(k[i]->dims[1] * q2 * sizeof(half));
            heights.push_back(k0);

            dsts.push_back((uint8_t *) (allValues + i * k0 * k1 * v2));
            dpitchs.push_back(k1 * v2 * sizeof(half));
            srcs.push_back(v[i]->cudaData);
            spitchs.push_back(v[i]->strides[0] * sizeof(half));
            widths.push_back(v[i]->dims[1] * v2 * sizeof(half));
            heights.push_back(k0);
        }
        FastllmCudaMemcpy2DDeviceToDeviceBatch(dsts.data(), dpitchs.data(), srcs.data(), spitchs.data(), widths.data(), heights.data(), dsts.size());
/*
        for (int i = 0; i < batch; i++) {
            cudaMemcpy2D(
                allKeys + i * k0 * k1 * q2, k1 * q2 * sizeof(half), 
                k[i]->cudaData, k[i]->strides[0] * sizeof(half), 
                k[i]->dims[1] * q2 * sizeof(half), k0, 
                cudaMemcpyDeviceToDevice
            );
            cudaMemcpy2D(
                allValues + i * k0 * k1 * v2, k1 * v2 * sizeof(half), 
                v[i]->cudaData, v[i]->strides[0] * sizeof(half), 
                v[i]->dims[1] * v2 * sizeof(half), k0, 
                cudaMemcpyDeviceToDevice
            );
        }
*/
        half *qd = (half*)q[0]->cudaData;
        half *od = (half*)output[0]->cudaData;
        half *qk = (half *) FastllmCudaMalloc(batch * q0 * q1 * k1 * sizeof(half));
        half *temp = (half *) FastllmCudaMalloc(batch * q0 * q1 * k1 * sizeof(half));
        auto fastllmCublasHandle = getFastllmCublasHandle();
        cublasStatus_t status;

        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_T, CUBLAS_OP_N,
                                           k1, q1 * group, q2, &hscale,
                                           allKeys, q2, k1 * q2,
                                           qd, q2, group * q1 * q2,
                                           &beta,
                                           qk, k1, k1 * q1 * group, batch * q0 / group);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("status = %d\n", (int) status);
            printf("Error: cublas error during MatMulTransB in Attention operator.\n");
            throw ("cublas error");
            exit(0);
        }

        int outer = batch * q0 * q1;
        if (k1 < 8) {
            FastllmSoftmaxKernelInner1<1> <<< outer, 1 >>>(qk, temp, outer, k1);
        } else if (k1 < 64) {
            FastllmSoftmaxKernelInner1<8> <<< outer, 8 >>>(qk, temp, outer, k1);
        } else if (k1 < 512) {
            FastllmSoftmaxKernelInner1<64> <<< outer, 64 >>>(qk, temp, outer, k1);
        } else {
            FastllmSoftmaxKernelInner1<256> <<< outer, 256 >>>(qk, temp, outer, k1);
        }

        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                                           CUBLAS_OP_N, CUBLAS_OP_N,
                                           v2, q1 * group, k1, &one,
                                           allValues, v2, k1 * v2,
                                           temp, k1, k1 * q1 * group,
                                           &beta,
                                           od, v2, v2 * q1 * group, batch * q0 / group);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("status = %d\n", (int) status);
            printf("Error: cublas error during MatMul in Attention operator.\n");
            throw ("cublas error");
            exit(0);
        }

        FastllmCudaFree(allKeys);
        FastllmCudaFree(allValues);
        FastllmCudaFree(qk);
        FastllmCudaFree(temp);
        DeviceSync();
        return true;
    }

    int k0 = k[0]->dims[0];
    size_t memSum = 0;
    for (int b = 0; b < batch; b++) {
        memSum += q[b]->dims[0] * q[b]->dims[1] * k[b]->dims[1];
    }
    T *mem = (T*) FastllmCudaMalloc(memSum * sizeof(T));
    T **qk = new T*[batch];
    memSum = 0;
    for (int b = 0; b < batch; b++) {
        int s = q[b]->dims[0] * q[b]->dims[1] * k[b]->dims[1];
        qk[b] = mem + memSum;
        memSum += s;
    }

    uint8_t ** pointers = (uint8_t**)FastllmCudaMalloc(sizeof(uint8_t*) * batch * k0 * 8);
    uint8_t ** cpuPointers = new uint8_t*[batch * k0 * 8];
    if (true) {
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < k0; i++) {
                cpuPointers[(b * k0 + i) * 8 + 0] = (uint8_t *) q[b]->cudaData + i * group * q[b]->dims[1] * q[b]->dims[2] * sizeof(T);
                cpuPointers[(b * k0 + i) * 8 + 1] = (uint8_t *) k[b]->cudaData + i * k[b]->strides[0] * sizeof(T);
                cpuPointers[(b * k0 + i) * 8 + 2] = (uint8_t *) qk[b] + i * group * q[b]->dims[1] * k[b]->dims[1] * sizeof(T);
                cpuPointers[(b * k0 + i) * 8 + 3] = (uint8_t *) (size_t) (group * q[b]->dims[1]);
                cpuPointers[(b * k0 + i) * 8 + 4] = (uint8_t *) (size_t) q[b]->dims[2];
                cpuPointers[(b * k0 + i) * 8 + 5] = (uint8_t *) (size_t) k[b]->dims[1];
                cpuPointers[(b * k0 + i) * 8 + 6] = (uint8_t *) (size_t) q[b]->strides[1];
                cpuPointers[(b * k0 + i) * 8 + 7] = (uint8_t *) (size_t) k[b]->strides[1];
            }
        }
        cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * batch * k0 * 8, cudaMemcpyHostToDevice);
        if (typeid(T) == typeid(half)) {
            FastllmHalfMatMulTransBBatchKernel <128> <<<batch * k0, 128>>> (pointers, scale);
        } else {
            FastllmMatMulTransBBatchKernel <128> <<<batch * k0, 128>>> (pointers, scale);
        }
    }

    if (true) {
        int outer = q[0]->dims[0] * q[0]->dims[1];
        int maxChannels = 0;
        bool useCausalBatch = q[0]->dims[1] > 1;
        int queryHeads = q[0]->dims[0];
        int queryLength = q[0]->dims[1];
        for (int b = 0; b < batch; b++) {
            int outer = q[b]->dims[0] * q[b]->dims[1];
            int channels = k[b]->dims[1];
            cpuPointers[b * 2 + 0] = (uint8_t*)(qk[b]);
            cpuPointers[b * 2 + 1] = (uint8_t*)((size_t)channels);
            maxChannels = max(maxChannels, channels);
            useCausalBatch = useCausalBatch &&
                q[b]->dims[0] == queryHeads &&
                q[b]->dims[1] == queryLength &&
                (mask == nullptr || mask[b] == nullptr ||
                 mask[b]->dims.empty());
        }
        cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * batch * 2, cudaMemcpyHostToDevice);
        if (useCausalBatch && maxChannels < 128) {
            FastllmSoftmaxKernelBatchInner1WithCausalMask<T, 32><<<
                batch * outer, 32>>>(
                    pointers, queryHeads, queryLength);
        } else if (useCausalBatch && maxChannels < 512) {
            FastllmSoftmaxKernelBatchInner1WithCausalMask<T, 64><<<
                batch * outer, 64>>>(
                    pointers, queryHeads, queryLength);
        } else if (useCausalBatch) {
            FastllmSoftmaxKernelBatchInner1WithCausalMask<T, 128><<<
                batch * outer, 128>>>(
                    pointers, queryHeads, queryLength);
        } else if (maxChannels < 128) {
            FastllmSoftmaxKernelBatchInner1 <T, 32> <<<batch * outer, 32>>> (pointers, outer);
        } else if (maxChannels < 512) {
            FastllmSoftmaxKernelBatchInner1 <T, 64> <<<batch * outer, 64>>> (pointers, outer);
        } else {
            FastllmSoftmaxKernelBatchInner1 <T, 128> <<<batch * outer, 128>>> (pointers, outer);
        }
    }

    if (true) {
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < k0; i++) {
                cpuPointers[(b * k0 + i) * 8 + 0] = (uint8_t *) qk[b] + i * group * q[b]->dims[1] * k[b]->dims[1] * sizeof(T);
                cpuPointers[(b * k0 + i) * 8 + 1] = (uint8_t *) v[b]->cudaData + i * v[b]->strides[0] * sizeof(T);
                cpuPointers[(b * k0 + i) * 8 + 2] = (uint8_t *) output[b]->cudaData + i * group * q[b]->dims[1] * v[b]->dims[2] * sizeof(T);
                cpuPointers[(b * k0 + i) * 8 + 3] = (uint8_t *) (size_t) (group * q[b]->dims[1]);
                cpuPointers[(b * k0 + i) * 8 + 4] = (uint8_t *) (size_t) k[b]->dims[1];
                cpuPointers[(b * k0 + i) * 8 + 5] = (uint8_t *) (size_t) v[b]->dims[2];
                cpuPointers[(b * k0 + i) * 8 + 6] = (uint8_t *) (size_t) k[b]->dims[1];
                cpuPointers[(b * k0 + i) * 8 + 7] = (uint8_t *) (size_t) v[b]->strides[1];
            }
        }
        cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * batch * k0 * 8, cudaMemcpyHostToDevice);
        
        if (typeid(T) == typeid(half)) {
            FastllmHalfMatMulKernel <128> <<<batch * k0, 128>>> (pointers, 1.0f);
        } else {
            FastllmMatMulKernel <128> <<<batch * k0, 128>>> (pointers, 1.0f);
        }
    }

    FastllmCudaFree(pointers);
    delete[] cpuPointers;

    FastllmCudaFree(mem);
    delete[] qk;
    
    DeviceSync();
    return true;
}

bool FastllmCudaAttentionBatch(fastllm::Data **q, fastllm::Data **k, fastllm::Data **v,
                               fastllm::Data **mask, fastllm::Data **output, int group, float scale, int batch) {
    if (q[0]->dataType == fastllm::DataType::FLOAT32) {
        return DoFastllmCudaAttentionBatch <float> (q, k, v, mask, output, group, scale, batch);
    } else if (q[0]->dataType == fastllm::DataType::FLOAT16) {
        if (TryFastllmCudaHalfLongKvAttentionBatch(
                q, k, v, mask, output, group, scale, batch)) {
            return true;
        }
        return DoFastllmCudaAttentionBatch <half> (q, k, v, mask, output, group, scale, batch);
    } else {
        printf("Error: attention datatype error.\n");
        throw ("Error: attention datatype error.");
        exit(0);
    }
}

bool FastllmCudaAttentionMask(fastllm::Data &input, const fastllm::Data &mask, float maskValue) {
    int spatial = input.Count(2), n = input.dims[0], m = input.dims[1];
    float *cudaData = (float *) FastllmCudaPrepareInput(input);
    float *maskData = (float *) FastllmCudaPrepareInput(mask);

    if (input.dataType == fastllm::DataType::FLOAT32) {
        FastllmAttentionMaskKernel <256> <<< n * m, 256>>>(cudaData, maskData, maskValue,
                                                       n, m, spatial);
    } else {
        FastllmAttentionMaskKernel <256> <<< n * m, 256>>>((half*)cudaData, (half*)maskData, __float2half(maskValue),
                                                        n, m, spatial);
    }
    FastllmCudaFinishInput(mask, maskData);
    FastllmCudaFinishOutput(input, cudaData);
    return true;
}

bool FastllmCudaMLA(const fastllm::Data &qNope, const fastllm::Data &qPe, const fastllm::Data &kvCache, const fastllm::Data &peCache, 
    fastllm::Data &ss, fastllm::Data &output, float softmaxScale) {
    int b = qPe.dims[0], s = qPe.dims[1], h = qPe.dims[2], c = qNope.dims.back(), t = kvCache.dims[1], r = qPe.dims[3];
    auto fastllmCublasHandle = getFastllmCublasHandle();
    cublasStatus_t status;

    if (qNope.dataType == fastllm::DataType::FLOAT32) {
        float *score = (float*)FastllmCudaMalloc(b * s * h * t * sizeof(float));
        float alpha = softmaxScale, beta0 = 0.0f, beta1 = 1.0f;
        status = cublasSgemmStridedBatched(fastllmCublasHandle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            t, h, c, &alpha,
            (float*)peCache.cudaData, c, t * c,
            (float*)qNope.cudaData, c, h * c,
            &beta0,
            score, t, t * h, 1);
        status = cublasSgemmStridedBatched(fastllmCublasHandle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            t, h, r, &alpha,
            (float*)kvCache.cudaData, r, t * r,
            (float*)qPe.cudaData, r, h * r,
            &beta1,
            score, t, t * h, 1);        
        int outer = b * s * h, channels = t;
        FastllmSoftmaxKernelInner1 <64> <<< outer, 64 >>> (score, score, outer, channels);
        status = cublasSgemmStridedBatched(fastllmCublasHandle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    c, b * s * h, t, &beta1,
                    (float*)peCache.cudaData, c, t * c,
                    score, t, b * s * h * t,
                    &beta0,
                    (float*)output.cudaData, c, c * b * s * h, 1);
        FastllmCudaFree(score);
    } else if (qNope.dataType == fastllm::DataType::FLOAT16) {
        half *score = (half*)FastllmCudaMalloc(b * s * h * t * sizeof(half));
        half alpha = __float2half_rn(softmaxScale), beta0 = __float2half_rn(0.0f), beta1 = __float2half_rn(1.0f);
        status = cublasHgemmStridedBatched(fastllmCublasHandle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            t, h, c, &alpha,
            (half*)peCache.cudaData, c, t * c,
            (half*)qNope.cudaData, c, h * c,
            &beta0,
            score, t, t * h, 1);
        status = cublasHgemmStridedBatched(fastllmCublasHandle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            t, h, r, &alpha,
            (half*)kvCache.cudaData, r, t * r,
            (half*)qPe.cudaData, r, h * r,
            &beta1,
            score, t, t * h, 1);        
        int outer = b * s * h, channels = t;
        FastllmSoftmaxKernelInner1 <64> <<< outer, 64 >>> (score, score, outer, channels);
        status = cublasHgemmStridedBatched(fastllmCublasHandle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    c, b * s * h, t, &beta1,
                    (half*)peCache.cudaData, c, t * c,
                    score, t, b * s * h * t,
                    &beta0,
                    (half*)output.cudaData, c, c * b * s * h, 1);
        FastllmCudaFree(score);
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("status = %d\n", (int) status);
        printf("Error: cublas error during MatMul in MLA operator.\n");
        throw("cublas error");
    }

    DeviceSync();
    return true;
}

bool FastllmCudaBatchMatMulTransBBatch(void **i0s, void **i1s, void **os,
                                       int *ns, int *ms, int *ks,
                                       int *i0Strides, int *i1Strides, float alpha, int batch) {
    uint8_t ** pointers = (uint8_t**)FastllmCudaMalloc(sizeof(uint8_t*) * batch * 8);
    uint8_t ** cpuPointers = new uint8_t*[batch * 8];
    for (int i = 0; i < batch; i++) {
        cpuPointers[i * 8 + 0] = (uint8_t *) i0s[i];
        cpuPointers[i * 8 + 1] = (uint8_t *) i1s[i];
        cpuPointers[i * 8 + 2] = (uint8_t *) os[i];
        cpuPointers[i * 8 + 3] = (uint8_t *) (size_t) ns[i];
        cpuPointers[i * 8 + 4] = (uint8_t *) (size_t) ms[i];
        cpuPointers[i * 8 + 5] = (uint8_t *) (size_t) ks[i];
        cpuPointers[i * 8 + 6] = (uint8_t *) (size_t) i0Strides[i];
        cpuPointers[i * 8 + 7] = (uint8_t *) (size_t) i1Strides[i];
    }
    cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * batch * 8, cudaMemcpyHostToDevice);
    FastllmMatMulTransBBatchKernel <128> <<<batch, 128>>> (pointers, alpha);
    FastllmCudaFree(pointers);
    delete[] cpuPointers;
    DeviceSync();
    return true;
}

bool FastllmCudaBatchMatMulBatch(void **i0s, void **i1s, void **os,
                                 int *ns, int *ms, int *ks,
                                 int *i0Strides, int *i1Strides, float alpha, int batch) {
    uint8_t ** pointers = (uint8_t**)FastllmCudaMalloc(sizeof(uint8_t*) * batch * 8);
    uint8_t ** cpuPointers = new uint8_t*[batch * 8];
    for (int i = 0; i < batch; i++) {
        cpuPointers[i * 8 + 0] = (uint8_t *) i0s[i];
        cpuPointers[i * 8 + 1] = (uint8_t *) i1s[i];
        cpuPointers[i * 8 + 2] = (uint8_t *) os[i];
        cpuPointers[i * 8 + 3] = (uint8_t *) (size_t) ns[i];
        cpuPointers[i * 8 + 4] = (uint8_t *) (size_t) ms[i];
        cpuPointers[i * 8 + 5] = (uint8_t *) (size_t) ks[i];
        cpuPointers[i * 8 + 6] = (uint8_t *) (size_t) i0Strides[i];
        cpuPointers[i * 8 + 7] = (uint8_t *) (size_t) i1Strides[i];
    }
    cudaMemcpy(pointers, cpuPointers, sizeof(uint8_t*) * batch * 8, cudaMemcpyHostToDevice);
    FastllmMatMulKernel <128> <<<batch, 128>>> (pointers, alpha);
    FastllmCudaFree(pointers);
    delete[] cpuPointers;
    DeviceSync();
    return true;
}

// Keep the supported source/storage combinations shared by all paged KV writers.
// The return value describes dtype support; launch errors belong to the caller.
