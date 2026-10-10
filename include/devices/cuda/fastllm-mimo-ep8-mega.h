#pragma once
#include "fastllm.h"
#include <memory>

// Opt-in MiMo complete-expert EP8 backend. All handles and symmetric buffers
// are prepared before worker execution / CUDA Graph capture.
struct FastllmCudaMimoEp8Mega;
bool FastllmCudaMimoEp8MegaSupported(const std::vector<int> &devices);
std::shared_ptr<FastllmCudaMimoEp8Mega> FastllmCudaCreateMimoEp8Mega(
    const std::vector<int> &devices,
    const std::vector<std::vector<std::vector<fastllm::Data *>>> &weights);
void FastllmCudaRunMimoEp8Mega(FastllmCudaMimoEp8Mega &state, int rank, int layer,
    const fastllm::Data &input, const fastllm::Data &index, const fastllm::Data &score,
    fastllm::Data &output);
