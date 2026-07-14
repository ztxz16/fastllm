#include "fastllm-cuda.cuh"

bool TryCudaCutlassW4A8(const fastllm::Data &input, fastllm::Data &weight,
                        const fastllm::Data &bias, fastllm::Data &output,
                        int n, int m, int k) {
    (void)input;
    (void)weight;
    (void)bias;
    (void)output;
    (void)n;
    (void)m;
    (void)k;
    return false;
}
