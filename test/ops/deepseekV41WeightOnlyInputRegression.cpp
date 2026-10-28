#include "models/deepseekv41.h"
#include "devices/cuda/fastllm-fp8-auxiliary.h"
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include <cuda_runtime.h>
#include "devices/multicuda/fastllm-multicuda.cuh"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace fastllm;
static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
static void Check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static void Fill(Data &out, const std::vector<int> &dims, DataType type, int seed, float amplitude) {
    size_t count = 1;
    for (int d : dims) count *= d;
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i)
        values[i] = amplitude * std::sin(float((i * 37 + seed * 19) % 1031));
    out.CopyFrom(Data(DataType::FLOAT32, dims, values));
    ToDataType(out, type);
    out.ToDevice(DataDevice::CUDA, std::vector<int>{0});
}
static std::vector<unsigned char> Bits(const Data &d) {
    std::vector<unsigned char> result(d.GetBytes());
    Check(cudaMemcpy(result.data(), d.cudaData, result.size(), cudaMemcpyDeviceToHost));
    return result;
}
static void Weight(Data &w, int inner, int rows, int bm, int bk, int seed) {
    w.dataType = DataType::FP8_E4M3;
    w.UpdateUnitSize();
    w.Resize({rows, inner});
    w.blockM = bm;
    w.blockK = bk;
    w.Allocate();
    for (size_t i = 0; i < w.GetBytes(); ++i)
        w.cpuData[i] = ((i + seed) % 2 ? 128 : 0) | ((i * 31 + i / inner * 7 + seed) % 105);
    w.scales.resize(size_t((rows - 1) / bk + 1) * ((inner - 1) / bm + 1));
    for (size_t i = 0; i < w.scales.size(); ++i)
        w.scales[i] = std::ldexp(1.0f, int((i * 13 + seed) % 19) - 16);
    w.ToDevice(DataDevice::CUDA, std::vector<int>{0});
}
static void Quant(const Data &in, Data &out) {
    Fill(out, in.dims, DataType::BFLOAT16, 0, 0);
    Require(FastllmCudaDeepSeekV41QuantizeActivation(in, out), "reference quant rejected");
}
struct Access : DeepSeekV41Model {
    using DeepSeekV41Model::LinearWithActivationQuant;
    using DeepSeekV41Model::quantizedLinearNames;
};
int main() { try {
    ApplyDeviceMap({{"cuda:0", 1}}, 0, 1); FastllmCudaSetDevice(0);
    int cases = 0;
    for (int block : {32, 64}) for (int rows = 1; rows <= 9; ++rows) {
        Access model;
        const std::string name = "layers.0.attn.wq_b.weight";
        auto &weight = model.weight[name];
        Weight(weight, 512, 512, block, block, 19);
        model.quantizedLinearNames.insert(name);
        Data input, quant, scratch, output, reference;
        Fill(input, {1, rows, 512}, DataType::BFLOAT16, rows, .8f);
        Quant(input, quant); Fill(scratch, input.dims, DataType::BFLOAT16, 0, 123.f);
        Fill(reference, {1, rows, 512}, DataType::BFLOAT16, 0, 0);
        const auto before = Bits(input);
        const bool direct = rows > 1 && rows <= 8 && block == 32;
        Require(FastllmCudaCanUseMarlinBFloat16FP8(weight, rows, 512, 512) == direct,
                "incorrect W8A16 eligibility");
        Require(FastllmCudaBFloat16MatMulFP8E4M3(direct ? input : quant,
                    weight, Data(), reference, rows, 512, 512), "reference GEMM rejected");
        model.LinearWithActivationQuant(input, name, output, false, &scratch);
        Check(cudaDeviceSynchronize());
        Require(Bits(input) == before, "input changed");
        Require(Bits(scratch) == Bits(quant), "shared quantized scratch was not populated");
        Require(Bits(output) == Bits(reference), "wrong activation precision selected");
        ++cases;
    }
    int devices = 0; Check(cudaGetDeviceCount(&devices));
    if (devices >= 2) {
        for (int block : {32, 64}) for (int rows : {1, 2, 6, 8, 9}) for (bool reuse : {false, true}) {
            ApplyDeviceMap({{"cuda:0", 1}}, 0, 1); FastllmCudaSetDevice(0);
            Access model; const std::string name = "layers.0.attn.wo_b.weight";
            Data &weight = model.weight[name]; Weight(weight, 1024, 512, block, block, 29);
            weight.ToDevice(DataDevice::CPU); weight.tpLinearType = TP_LINEAR_COLUMN;
            model.quantizedLinearNames.insert(name);
            Data input, output, scratch, bias;
            input.dataType = BFLOAT16; input.Resize({1, rows, 1024});
            std::vector<int> ranks{0, 1}; DivisionScheme scheme{{0, {{0, 512}}}, {1, {{512, 1024}}}};
            ApplyDeviceMap({{"multicuda:0,1", 1}}, 0, 1); FastllmMultiCudaSetDevice(ranks);
            PrepareMultiCudaShardedData(input, ranks, input.dims, 2, scheme);
            SplitMultiCudaWeight(weight, bias, ranks, scheme, 1);
            input.dataDevice = DataDevice::CPU; // The root has metadata; each rank owns CUDA storage.
            std::vector<std::vector<unsigned char>> before(2), quantBits(2), partial(2);
            for (int rank : ranks) {
                Data local, quant, result;
                Fill(local, {1, rows, 512}, BFLOAT16, rows + rank * 31, .8f);
                local.ToDevice(DataDevice::CUDA, std::vector<int>{rank});
                input.multiDeviceDatas.at(rank)->CopyFrom(local); FastllmCudaSetDevice(rank);
                quant.dataType = BFLOAT16; quant.Resize(local.dims);
                quant.ToDevice(DataDevice::CUDA, std::vector<int>{rank}, false); quant.Allocate();
                Require(FastllmCudaDeepSeekV41QuantizeActivation(local, quant), "TP reference quant rejected");
                result.dataType = BFLOAT16; result.Resize({1, rows, 512});
                result.ToDevice(DataDevice::CUDA, std::vector<int>{rank}, false); result.Allocate();
                const bool direct = block == 32 && rows >= 2 && rows <= 8;
                Require(FastllmCudaBFloat16MatMulFP8E4M3(direct ? local : quant,
                    *weight.multiDeviceDatas.at(rank), Data(), result, rows, 512, 512), "TP reference GEMM rejected");
                before[rank] = Bits(local); quantBits[rank] = Bits(quant); partial[rank] = Bits(result);
            }
            std::vector<unsigned char> expected(partial[0].size());
            for (size_t i = 0; i < expected.size(); i += 2) {
                float values[2];
                for (int rank : ranks) { uint16_t half; memcpy(&half, &partial[rank][i], 2);
                    uint32_t bits = uint32_t(half) << 16; memcpy(&values[rank], &bits, 4); }
                float sum = values[0] + values[1]; uint32_t bits; memcpy(&bits, &sum, 4);
                uint16_t rounded = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
                memcpy(&expected[i], &rounded, 2);
            }
            model.LinearWithActivationQuant(input, name, output, false, reuse ? &scratch : nullptr);
            for (int rank : ranks) {
                FastllmCudaSyncDevice(rank); FastllmCudaSetDevice(rank);
                Require(Bits(*output.multiDeviceDatas.at(rank)) == expected, "TP W8A16 result differs from local-GEMM oracle");
                Require(Bits(*input.multiDeviceDatas.at(rank)) == before[rank], "TP input changed");
                if (reuse) Require(Bits(*scratch.multiDeviceDatas.at(rank)) == quantBits[rank], "TP shared quantized scratch missing");
            }
            ++cases;
        }
    }
    printf("PASS %d W8A16/shared-scratch/fallback cases\n", cases);
} catch (const std::exception &e) { fprintf(stderr, "%s\n", e.what()); return 1; } }
