//
// Created by huangyuyang on 8/14/24.
//

#ifndef FASTLLM_COMPUTE_LINEAR_H
#define FASTLLM_COMPUTE_LINEAR_H

#include <cstdint>

namespace fastllm {
    void MatMulFloat16Float16(uint16_t *inputData, uint16_t *weightData, float *biasData, uint16_t *outputData, 
                            int n, int m, int k, int st, int end);
}

#endif // FASTLLM_COMPUTE_LINEAR_H