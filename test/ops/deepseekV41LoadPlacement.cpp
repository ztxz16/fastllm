// Exercise the loader's virtual hook: host ownership, PP boundaries, merged
// weights, and byte-exact placement of both plain and canonical GGUF payloads.
#include "models/deepseekv41.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "gguf.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace fastllm;
static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

static void Place(basellm &model, const std::string &name, int device,
                  DataType dtype = FLOAT16, int ggml = 0) {
    Data weight(dtype, ggml, {16, 256});
    weight.isModelWeight = true;
    weight.name = name;
    weight.Allocate();
    std::vector<uint8_t> expected(weight.GetBytes());
    for (size_t i = 0; i < expected.size(); ++i) expected[i] = (i * 17 + i / 13) % 256;
    std::memcpy(weight.cpuData, expected.data(), expected.size());
    Check(model.MoveSpecialWeightToCudaIfNeeded(name, weight) == (device >= 0), "unexpected placement decision");
    Check(weight.dataType == dtype && weight.ggmlType == ggml, "placement changed dtype");
    if (device < 0) {
        Check(weight.dataDevice == CPU && weight.cudaData == nullptr && weight.cpuData != nullptr,
              "CPU weight moved early");
        Check(std::memcmp(weight.cpuData, expected.data(), expected.size()) == 0, "CPU payload changed");
    } else {
        Check(weight.dataDevice == CUDA && weight.dataDeviceIds == std::vector<int>{device}, "wrong PP stage");
        Check(weight.cpuData == nullptr && weight.cudaData != nullptr, "host payload retained");
        cudaPointerAttributes attributes{};
        Check(cudaPointerGetAttributes(&attributes, weight.cudaData) == cudaSuccess && attributes.device == device,
              "physical allocation on wrong GPU");
        FastllmCudaSetDevice(device);
        std::vector<uint8_t> actual(expected.size());
        Check(cudaMemcpy(actual.data(), weight.cudaData, actual.size(), cudaMemcpyDeviceToHost) == cudaSuccess,
              "readback failed");
        Check(actual == expected, "GPU payload changed");
        Check(!model.MoveSpecialWeightToCudaIfNeeded(name, weight), "ordinary weight moved twice");
    }
}

int main(int argc, char **argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) return 77;
        const bool reference = argc > 1 && std::strcmp(argv[1], "--reference") == 0;
        setenv("FASTLLM_DSV41_REFERENCE_MATH", reference ? "1" : "0", 1);
        DeepSeekV41Model model;
        basellm &loader = model;
        model.block_cnt = 40;
        model.deviceMap = {{"cuda:0", 1}, {"cuda:1", 1}};
        model.moeDeviceMap = {{"numa", 1}};
        SetCudaSharedExpert(true);
        if (reference) {
            Place(loader, "layers.0.attn.wo_a.weight", -1);
            Place(loader, "head.weight", -1);
            std::puts("PASS: V4.1 load placement reference");
            return 0;
        }
        for (int layer : {0, 19, 20, 39}) {
            const std::string prefix = "layers." + std::to_string(layer) + ".";
            const int gpu = layer < 20 ? 0 : 1;
            Place(loader, prefix + "attn.wo_a.weight", gpu);
            Place(loader, prefix + "attn_norm.weight", gpu, FLOAT32);
            Place(loader, prefix + "engram.wkv.weight", gpu, DATA_GGUF_FORMAT, GGML_TYPE_Q4_K);
            Place(loader, prefix + "ffn.shared_experts.gateup.weight", gpu, DATA_GGUF_FORMAT, GGML_TYPE_IQ1_S);
            Place(loader, prefix + "ffn.shared_experts.w1.weight", -1);
            Place(loader, prefix + "ffn.shared_experts.w3.weight", -1);
            Place(loader, prefix + "ffn.experts.0.gateup.weight", -1);
            Place(loader, prefix + "engram.embed.weight", -1);
        }
        Place(loader, "head.weight", 1, DATA_GGUF_FORMAT, GGML_TYPE_Q4_K);
        Place(loader, "norm.weight", 1, BFLOAT16);
        Place(loader, "embed.weight", -1);
        Place(loader, "mtp.0.head.weight", -1);
        Place(loader, "layers.40.attn.wo_a.weight", -1);
        Place(loader, "layers.-1.attn.wo_a.weight", -1);
        Place(loader, "layers.99999999999999999.attn.wo_a.weight", -1);
        SetCudaSharedExpert(false);
        Place(loader, "layers.0.ffn.shared_experts.gateup.weight", -1);
        SetCudaSharedExpert(true);
        // Unequal PP ratios and a final CPU stage must follow the execution map.
        model.deviceMap = {{"cuda:0", 1}, {"cuda:1", 3}};
        Place(loader, "layers.9.attn.wo_a.weight", 0);
        Place(loader, "layers.10.attn.wo_a.weight", 1);
        for (const char *device : {"cpu", "numa", "multicuda:0,1", "cuda:0,1"}) {
            model.deviceMap = {{device, 1}};
            Place(loader, "layers.0.attn.wo_a.weight", -1);
            Place(loader, "head.weight", -1);
        }
        std::puts("PASS: V4.1 load placement");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
