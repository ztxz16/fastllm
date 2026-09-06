// Test-only adapter; PyTorch owns all device buffers.
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime_api.h>

extern "C" void *FlashInferGdnTestStream() { return cudaStreamPerThread; }

static int FailedLaunch(void **p, int, int, int, void *) {
    cudaMemsetAsync(p[6], 0, 48 * 128 * 128 * sizeof(float), cudaStreamPerThread);
    return 1;  // Simulate a failure after a kernel has written the scratch state.
}

extern "C" bool FlashInferGdnTestRun(
        void *qkv, void *w, void *g, void *b, void *s, void *o,
        int tokens, bool emptyState, int invalid) {
    using namespace fastllm;
    Data input(DataType::FLOAT16, {1, tokens, 10240});
    Data weight(DataType::FLOAT32, {128});
    Data decay(DataType::FLOAT16, {1, tokens, 48});
    Data beta(DataType::FLOAT16, {1, tokens, 48});
    Data state(DataType::FLOAT16);
    if (!emptyState) state.Resize({1, 48, 128, 128});
    Data output(DataType::FLOAT16, {1, tokens, 48, 128});
    for (Data *data : {&input, &weight, &decay, &beta, &state, &output}) {
        data->isFake = true;
        data->dataDevice = DataDevice::CUDA;
    }
    input.cudaData = qkv; weight.cudaData = w;
    decay.cudaData = g; beta.cudaData = b;
    state.cudaData = s; output.cudaData = o;
    switch (invalid) {
        case 1: input.dataType = DataType::BFLOAT16; break;
        case 2: input.strides[1]++; break;
        case 3: state.isLinearAttentionTransposed = true; break;
        case 4: decay.dataType = DataType::FLOAT32; break;
        case 5: input.dataDevice = DataDevice::CPU; break;
        case 6: input.cudaData = nullptr; break;
        case 8: input.cudaData = (char *)qkv + 2; break;
        case 9: state.dataType = DataType::FLOAT32; break;
        case 10: {
            if (!FastllmCudaGraphBeginCapture()) return true;
            bool used = FastllmCudaTryFlashInferGdnPrefill(
                input, weight, decay, beta, tokens, 16, 48, 1e-6f, state, output);
            void *graph = nullptr;
            bool ended = FastllmCudaGraphEndCapture(&graph);
            if (graph) FastllmCudaGraphDestroy(graph);
            return used || !ended;
        }
        case 7: return FastllmCudaFlashInferGdnPrefill(
            FailedLaunch, input, weight, decay, beta, tokens, 16, 48,
            1e-6f, state, output);
    }
    return FastllmCudaTryFlashInferGdnPrefill(
        input, weight, decay, beta, tokens, 16, 48, 1e-6f, state, output);
}
