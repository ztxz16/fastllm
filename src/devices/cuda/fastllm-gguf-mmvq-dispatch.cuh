#pragma once
#include "fastllm-cuda.cuh"
#include "gguf.h"

template <ggml_type type, typename OType, int StoreMode = 0>
void mul_mat_vec_q_cuda(
    const void * vx, const void * vy, OType * dst, const char * ids_data,
    const int ncols_x, const int nrows_x, const int nrows_y, const int ncols_y, const int nrows_dst,
    const int ne2, const uint64_t nb02, const uint64_t nb12, const uint64_t nb2, const int64_t ids_nb0,
    cudaStream_t stream);
