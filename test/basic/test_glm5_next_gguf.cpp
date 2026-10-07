#include "gguf.h"
#include "executor.h"
#include "models/glm5_next.h"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#endif
#include "../../src/models/glm5_next_gguf.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <unistd.h>
using namespace fastllm;
namespace fastllm {
struct Glm5NextGGUFTestAccess {
    static void InitTp(Glm5NextModel &model, bool resident = false, bool sharded = false) {
        model.block_cnt = 4;
        model.kdaHeads = model.num_attention_heads = 4;
        model.qkNopeHeadDim = model.valueHeadDim = 8;
        model.kvLoraRank = 16;
        model.denseMlpLayers.assign(4, false);
        model.deviceMap = {{"cuda:0", 1}};
        model.moeDeviceMap = resident ? std::map<std::string, int>{{"cuda:0",1},{"cuda:1",1},{"numa",2}} :
            std::map<std::string, int>{{"cpu",1}};
        if (sharded) {
            model.moeDeviceMap = {{"cuda:0,1",1}};
            model.layeredMoeDeviceMap = {{"numa",1}};
            model.moeDeviceLayers = 2;
        }
        model.weight.dicts["gguf_architecture"] = "glm5next";
        model.InitThreadTp();
    }
};
}
namespace {
void Check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
using Bytes = std::vector<uint8_t>;
template<class T> void Put(Bytes &out, T value) {
    const auto *p = reinterpret_cast<const uint8_t *>(&value);
    out.insert(out.end(), p, p + sizeof(T));
}
void String(Bytes &out, const std::string &s) {
    Put<uint64_t>(out, s.size()); out.insert(out.end(), s.begin(), s.end());
}
struct Tensor { std::string name; std::vector<int> dims; ggml_type type; Bytes bytes; };
Tensor FloatTensor(const std::string &name, const std::vector<int> &dims, ggml_type type,
                   const std::vector<float> &values) {
    Tensor t{name, dims, type, {}};
    if (type == GGML_TYPE_Q8_0) {
        Check(values.size() % 32 == 0, "Q8 fixture block alignment");
        for (size_t b = 0; b < values.size(); b += 32) {
            Put<uint16_t>(t.bytes, float_to_half(0.125f));
            for (int c = 0; c < 32; ++c) Put<int8_t>(t.bytes, int8_t(values[b + c] * 8));
        }
        return t;
    }
    Check(type == GGML_TYPE_F32, "unsupported float fixture type");
    for (float value : values) Put(t.bytes, value);
    return t;
}
void Write(const std::string &path, const std::map<std::string, std::string> &metadata,
           const std::vector<Tensor> &tensors) {
    Bytes header, payload;
    Put<uint32_t>(header, 0x46554747); Put<uint32_t>(header, 3);
    Put<uint64_t>(header, tensors.size()); Put<uint64_t>(header, metadata.size());
    for (const auto &item : metadata) {
        String(header, item.first); Put<uint32_t>(header, 8); String(header, item.second);
    }
    for (const auto &t : tensors) {
        String(header, t.name); Put<uint32_t>(header, t.dims.size());
        for (auto it = t.dims.rbegin(); it != t.dims.rend(); ++it) Put<uint64_t>(header, *it);
        Put<uint32_t>(header, t.type); Put<uint64_t>(header, payload.size());
        payload.insert(payload.end(), t.bytes.begin(), t.bytes.end());
        payload.resize((payload.size() + 31) / 32 * 32);
    }
    header.resize((header.size() + 31) / 32 * 32);
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(header.data()), header.size());
    file.write(reinterpret_cast<const char *>(payload.data()), payload.size());
    Check(file.good(), "fixture write failed");
}

void TestEmbedding(const std::string &directory) {
    const std::string path = directory + "/embedding.gguf";
    std::vector<float> values(7 * 256);
    for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(float(i) * .017f);
    Tensor tensor{"token_embd.weight", {7, 256}, GGML_TYPE_Q4_K,
                  Bytes(7 * ggml_row_size(GGML_TYPE_Q4_K, 256))};
    ggml_type_from_float_ref(GGML_TYPE_Q4_K)(values.data(), tensor.bytes.data(), values.size());
    Write(path, {}, {tensor});
    const bool previousCuda = GetCudaEmbedding(), previousLowMem = GetLowMemMode();
    for (bool lowMem : {false, true}) for (bool cuda : {false, true}) {
        SetLowMemMode(lowMem); SetCudaEmbedding(cuda);
        std::vector<ReadGGUFTask> tasks;
        AppendGGUFTasks("glm5_next", path, tasks);
        Check(tasks.size() == 1, "GLM embedding task count");
        auto &task = tasks[0];
        const bool packed = lowMem || !cuda;
        Check(task.replaceType == (packed ? GGUFWeightReplaceRule::GGUFWeightReplaceDirect :
              GGUFWeightReplaceRule::GGUFWeightReplaceForceFP32), "GLM embedding placement policy");
        Data actualWeight(FLOAT32, {1}), reference(FLOAT32, {1});
        WeightImportGGUFTensor(&actualWeight, &task.tensor, task.fileName, task.offset, task.replaceType);
        WeightImportGGUFTensor(&reference, &task.tensor, task.fileName, task.offset,
                              GGUFWeightReplaceRule::GGUFWeightReplaceForceFP32);
        auto *storage = actualWeight.cpuData;
        if (packed) Check(actualWeight.dataType == DATA_GGUF_FORMAT &&
                          actualWeight.GetBytes() == tensor.bytes.size(), "GLM embedding expanded");
        auto &executor = *static_cast<Executor *>(GetExecutor());
        for (DataType type : {FLOAT32, FLOAT16}) {
            Data ids(type, {1, 5}, {6, 0, 3, 6, 1}), actual, expected;
            DataDict a{{"input", &ids}, {"weight", &actualWeight}, {"output", &actual}};
            DataDict b{{"input", &ids}, {"weight", &reference}, {"output", &expected}};
            executor.RunOnDevice("cpu", "Embedding", a, {}, {});
            executor.RunOnDevice("cpu", "Embedding", b, {}, {});
            Check(actual.GetBytes() == expected.GetBytes() &&
                  std::memcmp(actual.cpuData, expected.cpuData, actual.GetBytes()) == 0,
                  "GLM packed embedding changed lookup values");
            ToDataTypeForceCPU(actual, BFLOAT16); ToDataTypeForceCPU(expected, BFLOAT16);
            Check(actual.dims == expected.dims && actual.GetBytes() == expected.GetBytes() &&
                  std::memcmp(actual.cpuData, expected.cpuData, actual.GetBytes()) == 0,
                  "GLM packed embedding changed BF16 activations");
        }
        Check(storage == actualWeight.cpuData, "GLM embedding lookup replaced packed storage");
    }
    SetLowMemMode(previousLowMem); SetCudaEmbedding(previousCuda);
    unlink(path.c_str());
}

int TestExactGgufLinear() {
#ifdef USE_CUDA
    if (FastllmCudaGetDeviceCount() == 0) return 77;
    constexpr int columns = 2048, outputs = 256;
    std::vector<float> values(columns * outputs);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = std::sin(float(i) * .037f) * std::cos(float(i) * .009f);
    const int previous = FastllmCudaGetLinearExactBatchThreshold();
    for (int device = 0; device < FastllmCudaGetDeviceCount(); ++device) {
        FastllmCudaSetDevice(device);
        ApplyDeviceMap({{"cuda:" + std::to_string(device), 1}}, 0, 1);
        for (auto quant : {GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_Q4_0}) {
            Data weight(DATA_GGUF_FORMAT, quant, {outputs, columns});
            weight.isGGUFData = weight.isModelWeight = true;
            weight.Allocate();
            if (quant == GGML_TYPE_Q4_0) {
                auto *blocks = reinterpret_cast<block_q4_0 *>(weight.cpuData);
                for (size_t i = 0; i < values.size() / QK4_0; ++i) {
                    blocks[i].d = float_to_half(.0127f * (1 + i % 7));
                    for (int j = 0; j < QK4_0 / 2; ++j) blocks[i].qs[j] = uint8_t(i * 37 + j * 13);
                }
            } else {
                auto quantize = ggml_type_from_float_ref(quant);
                Check(quantize != nullptr, "GGUF test quantizer unavailable");
                quantize(values.data(), weight.cpuData, values.size());
            }
            weight.ToDevice(CUDA, std::vector<int>{device});
            for (int rows : {2, 3, 4, 5, 7, 8}) {
                std::vector<float> inputValues(rows * columns);
                for (size_t i = 0; i < inputValues.size(); ++i)
                    inputValues[i] = std::sin(float(i) * .029f);
                for (auto type : {FLOAT32, BFLOAT16}) {
                    Data input(FLOAT32, {rows, columns}, inputValues), actual;
                    input.ToDevice(CUDA, std::vector<int>{device}); ToDataType(input, type);
                    FastllmCudaSetLinearExactBatchThreshold(rows + 1);
                    Linear(input, weight, Data(), actual);
                    for (int row = 0; row < rows; ++row) {
                        Data rowInput, expected, got;
                        Split(input, 0, row, row + 1, rowInput);
                        FastllmCudaSetLinearExactBatchThreshold(0);
                        Linear(rowInput, weight, Data(), expected);
                        Split(actual, 0, row, row + 1, got);
                        expected.ToDevice(CPU); got.ToDevice(CPU);
                        if (expected.GetBytes() != got.GetBytes() ||
                            std::memcmp(expected.cpuData, got.cpuData, got.GetBytes()) != 0) {
                            std::fprintf(stderr, "GGUF exact linear device=%d quant=%s dtype=%d rows=%d row=%d\n",
                                device, ggml_type_name(quant), int(type), rows, row);
                            throw std::runtime_error("GGUF verifier reduction differs from one-row decode");
                        }
                    }
                }
            }
        }
    }
    FastllmCudaSetLinearExactBatchThreshold(previous);
    std::cout << "PASS GGUF exact verifier linear rows 2/3/4/5/7/8 on both GPUs\n";
    return 0;
#else
    return 77;
#endif
}

int TestStreamingTp() {
#ifdef USE_CUDA
    if (FastllmCudaGetDeviceCount() < 2) return 77;
    setenv("FASTLLM_TP", "0,1", 1);
    SetCudaSharedExpert(true);
    Glm5NextModel model;
    Glm5NextGGUFTestAccess::InitTp(model);
    const std::string base = "model.language_model.layers.0.";
    std::map<std::string, std::vector<int>> shapes = {
        {"self_attn.q_proj.weight", {16, 32}}, {"self_attn.o_proj.weight", {32, 16}},
        {"self_attn.q_conv1d.weight", {16, 1, 4}}, {"self_attn.A_log", {4}},
        {"mlp.shared_experts.gateup_proj.weight", {32, 32}},
        {"mlp.shared_experts.down_proj.weight", {32, 16}},
        {"mlp.gate.weight", {8, 32}}, {"hc_attn_fn", {4, 32}}};
    std::map<std::string, std::vector<float>> expected;
    std::set<std::string> names;
    for (const auto &item : shapes) {
        const std::string name = base + item.first;
        Data &w = model.weight[name]; w = Data(FLOAT32, item.second); w.Allocate();
        auto &values = expected[name]; values.resize(w.Count(0));
        for (size_t i = 0; i < values.size(); ++i) values[i] = i * .125f;
        std::memcpy(w.cpuData, values.data(), w.GetBytes());
        w.name = name; w.isModelWeight = true; names.insert(name);
        for (const char *arch : {"glm5next", "glm5-next"}) {
            model.weight.dicts["gguf_architecture"] = arch;
            Check(model.ShouldLoadWeightSeriallyBeforeOthers(name, {}), "TP dense group missing");
        }
    }
    for (const std::string name : {base + "mlp.experts.0.down_proj.weight",
                                  std::string("model.language_model.embed_tokens.weight")}) {
        Check(!model.ShouldLoadWeightSeriallyBeforeOthers(name, {}), "TP streamed host weight");
    }
    FastllmCudaSetDevice(1);
    model.OnWeightLoadGroupStarted(names);
    model.OnWeightLoadGroupFinished();
    Check(FastllmCudaGetDevice() == 1, "streaming TP changed caller CUDA device");
    model.OnWeightLoadGroupFinished();
    for (const auto &item : shapes) {
        const std::string name = base + item.first;
        auto &source = model.weight[name];
        Check(!source.cpuData && source.multiDeviceData && source.multiDeviceDatas.size() == 2 &&
              source.dims == item.second, "streamed source retained CPU memory or changed shape");
        const bool row = item.first == "self_attn.q_proj.weight" ||
            item.first == "self_attn.q_conv1d.weight" || item.first == "self_attn.A_log";
        const bool column = item.first.find("o_proj") != std::string::npos ||
            item.first.find("down_proj") != std::string::npos;
        const bool gate = item.first.find("gateup_proj") != std::string::npos;
        const int rows = item.second[0], cols = expected[name].size() / rows;
        for (int rank : {0, 1}) {
            auto &shard = *source.multiDeviceDatas.at(rank);
            Check(shard.cudaData && !shard.isFake, "streamed shard has no owner");
            std::vector<float> actual(shard.Count(0)), reference;
            FastllmCudaSetDevice(rank);
            FastllmCudaCopyFromDeviceToHost(actual.data(), shard.cudaData, shard.GetBytes());
            for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c) {
                if (row && r / (rows / 2) != rank) continue;
                if (column && c / (cols / 2) != rank) continue;
                if (gate && (r % (rows / 2)) / (rows / 4) != rank) continue;
                reference.push_back(expected[name][r * cols + c]);
            }
            Check(actual == reference, "streamed TP weight differs from reference shard");
        }
    }
    {
        Glm5NextModel mixed;
        Glm5NextGGUFTestAccess::InitTp(mixed, true);
        for (int layer=0; layer<4; ++layer) {
            const std::string name = "model.language_model.layers." + std::to_string(layer) + ".mlp.experts.0.down_proj.weight";
            mixed.specialWeights[name] = "linearColumn";
            mixed.specialWeightLayerIds[name] = layer;
            Data &w = mixed.weight[name];
            w = Data(DATA_GGUF_FORMAT, GGML_TYPE_IQ3_XXS, {256,256});
            w.isModelWeight = true; w.Allocate();
            std::memset(w.cpuData, 0, w.GetBytes());
            Check(!mixed.ShouldDelaySpecialWeightCudaMove(name), "resident upload delayed until warmup");
            const bool moved = mixed.MoveSpecialWeightToCudaIfNeeded(name, w);
            Check(moved == (layer<2), "mixed layer placement");
            if (layer<2) Check(w.cudaData && !w.cpuData && w.numasData.empty() &&
                w.dataDeviceIds == std::vector<int>{layer}, "resident weight retained host source or wrong GPU");
            else Check(w.cpuData && w.dataDevice == CPU, "NUMA layer moved to CUDA");
        }
    }
    {
        Glm5NextModel sharded;
        Glm5NextGGUFTestAccess::InitTp(sharded, false, true);
        for (int layer=0; layer<4; ++layer) {
            std::set<std::string> group;
            std::map<std::string, Bytes> originals;
            for (int part=0; part<2; ++part) {
                const std::string name = "model.language_model.layers." + std::to_string(layer) +
                    ".mlp.experts.0." + (part ? "down_proj.weight" : "gateup_proj.weight");
                sharded.specialWeights[name] = part ? "linearColumn" : "linearSwiglu";
                sharded.specialWeightLayerIds[name] = layer;
                Data &w = sharded.weight[name];
                w = Data(DATA_GGUF_FORMAT, part ? GGML_TYPE_IQ3_XXS : GGML_TYPE_IQ2_XXS,
                    part ? std::vector<int>{256,512} : std::vector<int>{1024,256});
                w.name=name; w.isGGUFData=w.isModelWeight=true; w.Allocate();
                for (size_t i=0; i<w.GetBytes(); ++i) w.cpuData[i]=uint8_t(i*17+i/37);
                originals[name]=Bytes(w.cpuData,w.cpuData+w.GetBytes());
                Check(sharded.ShouldDelaySpecialWeightCudaMove(name)==(layer<2), "TP expert upload policy");
                Check(sharded.ShouldLoadWeightSeriallyBeforeOthers(name,{})==(layer<2), "TP expert streaming group");
                Check(!sharded.MoveSpecialWeightToCudaIfNeeded(name,w), "TP expert uploaded before split");
                group.insert(name);
            }
            sharded.OnWeightLoadGroupStarted(group);
            sharded.OnWeightLoadGroupFinished();
            for (const auto &name:group) {
                auto &w=sharded.weight[name];
                if (layer>=2) { Check(w.cpuData && !w.multiDeviceData,"NUMA expert was split"); continue; }
                Check(!w.cpuData && w.numasData.empty() && w.multiDeviceDatas.size()==2,
                    "TP expert retained host source");
                const bool down=name.find("down_proj")!=std::string::npos;
                const size_t rowBytes=ggml_row_size(ggml_type(w.ggmlType),w.dims[1]);
                for (int r=0;r<2;++r) {
                    const auto &local=*w.multiDeviceDatas.at(r);
                    Check(local.dataDeviceIds==std::vector<int>{r} && local.cudaData && !local.cpuData,
                        "TP expert shard ownership");
                    Bytes actual(local.GetBytes()), expected;
                    FastllmCudaSetDevice(r);
                    FastllmCudaCopyFromDeviceToHost(actual.data(),local.cudaData,actual.size());
                    const auto &bytes=originals.at(name);
                    for (int row=0;row<w.dims[0];++row) {
                        if (!down && (row%512)/256!=r) continue;
                        const size_t first=row*rowBytes+(down ? r*rowBytes/2 : 0);
                        const size_t count=down ? rowBytes/2 : rowBytes;
                        expected.insert(expected.end(),bytes.begin()+first,bytes.begin()+first+count);
                    }
                    Check(actual==expected,"TP expert shard changed packed quantization blocks");
                }
            }
        }
    }
    unsetenv("FASTLLM_TP");
    std::cout << "PASS: GLM-5.3 GGUF streaming TP weights and released CPU sources\n";
    return 0;
#else
    return 77;
#endif
}

void Run(const std::string &directory) {
    std::vector<Tensor> tensors;
    constexpr int heads=2, keyDim=3, valueDim=4, rank=5;
    std::vector<float> k(heads*keyDim*rank), v(heads*valueDim*rank);
    // Canonical KV-B must contain ascending row-major integers after import.
    for (int h=0; h<heads; ++h) {
        for (int d=0; d<keyDim; ++d) for (int r=0; r<rank; ++r)
            k[(h*rank+r)*keyDim+d]=1+(h*(keyDim+valueDim)+d)*rank+r;
        for (int d=0; d<valueDim; ++d) for (int r=0; r<rank; ++r)
            v[(h*valueDim+d)*rank+r]=1+(h*(keyDim+valueDim)+keyDim+d)*rank+r;
    }
    tensors.push_back(FloatTensor("blk.3.attn_k_b.weight", {heads,rank,keyDim}, GGML_TYPE_F32, k));
    tensors.push_back(FloatTensor("blk.3.attn_v_b.weight", {heads,valueDim,rank}, GGML_TYPE_F32, v));
    tensors.push_back(FloatTensor("blk.0.ssm_a", {heads}, GGML_TYPE_F32, {-std::exp(.25f),-std::exp(-.75f)}));
    for (const char *kind : {"q", "k", "v"}) {
        const std::vector<int> shape = std::string(kind) == "k" ?
            std::vector<int>{2, 1, 4} : std::vector<int>{1, 2, 1, 4};
        tensors.push_back(FloatTensor("blk.0.ssm_conv1d_" + std::string(kind) + ".weight",
            shape, GGML_TYPE_F32, {1, 2, 3, 4, 5, 6, 7, 8}));
    }
    tensors.push_back(FloatTensor("blk.0.hc_attn_fn.weight", {24,32}, GGML_TYPE_Q8_0, std::vector<float>(24*32,.5f)));
    tensors.push_back(FloatTensor("blk.3.indexer_compressor_ape.weight", {4,128}, GGML_TYPE_F32, std::vector<float>(512,.25f)));
    tensors.push_back(FloatTensor("blk.4.nextn.enorm.weight", {32}, GGML_TYPE_F32, std::vector<float>(32,1.f)));
    tensors.push_back(FloatTensor("blk.4.nextn.hnorm.weight", {32}, GGML_TYPE_F32, std::vector<float>(32,2.f)));
    tensors.push_back(FloatTensor("blk.4.nextn.shared_head_norm.weight", {32}, GGML_TYPE_F32, std::vector<float>(32,3.f)));
    tensors.push_back(FloatTensor("blk.4.nextn.eh_proj.weight", {32,64}, GGML_TYPE_Q8_0, std::vector<float>(2048,.5f)));
    for (const auto &kind : {"gate","up","down"}) {
        const bool down=std::string(kind)=="down";
        const auto type=down ? GGML_TYPE_IQ3_XXS : GGML_TYPE_IQ2_XXS;
        const std::vector<int> shape={2,8,256};
        const size_t bytes=8*ggml_row_size(type,256);
        Bytes data(2*bytes,0);
        std::fill(data.begin()+bytes,data.end(),uint8_t(0x5a));
        tensors.push_back({"blk.3.ffn_"+std::string(kind)+"_exps.weight",shape,type,data});
    }
    const std::string first=directory+"/tiny-00001-of-00002.gguf";
    const std::string second=directory+"/tiny-00002-of-00002.gguf";
    Write(first, {{"general.architecture","glm5next"}}, {});
    Write(second, {}, tensors);
    std::vector<ReadGGUFTask> tasks;
    AppendGGUFTasks("glm5_next",first,tasks);
    Check(tasks.empty(),"metadata-only first shard");
    AppendGGUFTasks("glm5_next",second,tasks);
    Check(tasks.size()==18,"split expert tasks and NextN weight mappings");
    WeightMap weights;
    for (auto &task : tasks) WeightImportGGUFTensor(&weights[task.name],&task.tensor,task.fileName,task.offset,task.replaceType);
    const std::string base="model.language_model.layers.";
    for (const char *name : {"enorm.weight", "hnorm.weight", "shared_head.norm.weight"})
        Check(weights[base+"4."+name].dataType==FLOAT32 && weights[base+"4."+name].dims==std::vector<int>{32}, "NextN norm mapping");
    Check(weights[base+"4.eh_proj.weight"].dataType==DATA_GGUF_FORMAT &&
        weights[base+"4.eh_proj.weight"].dims==std::vector<int>({32,64}), "NextN projection stays packed");
    Check(weights[base+"0.hc_attn_fn"].dataType==FLOAT32,"quantized HC import type");
    Check(reinterpret_cast<float*>(weights[base+"0.hc_attn_fn"].cpuData)[31]==.5f,"quantized HC dequantization");
    Check(weights[base+"3.self_attn.indexer.index_kpool_compress_ape"].dims==std::vector<int>({4,128}),"KPool APE mapping");
    for (const auto &kind : {"gate","up","down"}) for(int e=0;e<2;++e) {
        const std::string name=base+"3.mlp.experts."+std::to_string(e)+"."+kind+"_proj.weight";
        Data &weight=weights[name];
        Check(weight.dataType==DATA_GGUF_FORMAT && weight.dims==std::vector<int>({8,256}),"expert must stay quantized");
        Check(weight.ggmlType==(std::string(kind)=="down" ? GGML_TYPE_IQ3_XXS : GGML_TYPE_IQ2_XXS),"expert quantization changed");
        for (uint64_t i=0;i<weight.GetBytes();++i) Check(weight.cpuData[i]==(e ? 0x5a : 0),"packed expert split offset");
    }
    for (int repeat=0; repeat<2; ++repeat) {
        glm5_next_detail::RestoreGgufWeights(weights,4,heads,keyDim,valueDim,rank);
        for (const char *kind : {"q", "k", "v"}) {
            Data &conv = weights[base + "0.self_attn." + kind + "_conv1d.weight"];
            Check(conv.dims == std::vector<int>({2, 1, 4}), "GGUF convolution shape normalization");
            for (int i = 0; i < 8; ++i) {
                Check(reinterpret_cast<float *>(conv.cpuData)[i] == float(i + 1),
                    "GGUF convolution normalization changed weight values");
            }
        }
        Data &combined=weights[base+"3.self_attn.kv_b_proj.weight"];
        Check(combined.dataType==BFLOAT16 && combined.dims==std::vector<int>({heads*(keyDim+valueDim),rank}),"KV-B reconstructed shape/dtype");
        for(uint64_t i=0;i<combined.Count(0);++i)
            Check(BFloat16BitsToFloat32(reinterpret_cast<uint16_t*>(combined.cpuData)[i])==float(i+1),"KV-B head transpose/concat");
        float *a=reinterpret_cast<float*>(weights[base+"0.self_attn.A_log"].cpuData);
        Check(std::fabs(a[0]-.25f)<1e-6f && std::fabs(a[1]+.75f)<1e-6f,"KDA decay inverse/idempotence");
    }
    Check(weights.weight.count(base+"0.self_attn.gguf_decay")==0 && weights.weight.count(base+"3.self_attn.gguf_k_b.weight")==0,"converted source weights retained");
    unlink(first.c_str());unlink(second.c_str());
}
}
int main(int argc, char **argv) {
    char directory[]="/tmp/fastllm-glm53-gguf-XXXXXX";
    try {
#ifdef USE_CUDA
        if (argc == 2 && std::string(argv[1]) == "--prefill-formats") {
            for (auto type : {BFLOAT16, FLOAT16, FLOAT32}) {
                for (int quant : {GGML_TYPE_IQ2_XXS_R4, GGML_TYPE_IQ2_S_R4, GGML_TYPE_IQ3_XXS_R4,
                                  GGML_TYPE_IQ4_XS, GGML_TYPE_Q2_K_R4, GGML_TYPE_Q3_K, GGML_TYPE_Q3_K_R4})
                    Check(FastllmCudaGGUFPrefillSupported(type, quant), "supported GGUF CUDA prefill rejected");
            }
            std::cout<<"PASS GGUF NUMA prefill CUDA format dispatch\n";return 0;
        }
#endif
        if (argc == 2 && std::string(argv[1]) == "--exact-linear") return TestExactGgufLinear();
        if (argc == 2 && std::string(argv[1]) == "--tp-load") return TestStreamingTp();
        Check(mkdtemp(directory)!=nullptr,"temporary directory");
        Run(directory);TestEmbedding(directory);rmdir(directory);
        std::cout << "PASS: GLM-5.3 GGUF shard mapping, packed IQ experts, KDA decay and MLA layout\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << "\n"; return 1; }
}
