#include "fastllm.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    const bool graphDefault = std::atoi(argv[1]) != 0;
    if (fastllm::GetCudaEmbedding() != graphDefault ||
        fastllm::GetCudaEmbeddingRequested()) return 1;
    for (bool enabled : {false, true, false}) {
        fastllm::SetCudaEmbedding(enabled);
        if (fastllm::GetCudaEmbedding() != enabled ||
            fastllm::GetCudaEmbeddingRequested() != enabled) {
            std::fprintf(stderr, "Explicit CUDA embedding policy was overridden.\n");
            return 1;
        }
    }
    std::puts("PASS: explicit CUDA embedding policy");
    return 0;
}
