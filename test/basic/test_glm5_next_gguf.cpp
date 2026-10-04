#include "gguf.h"
#include "../../src/models/glm5_next_gguf.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <unistd.h>
using namespace fastllm;
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
    tensors.push_back(FloatTensor("blk.0.hc_attn_fn.weight", {24,32}, GGML_TYPE_Q8_0, std::vector<float>(24*32,.5f)));
    tensors.push_back(FloatTensor("blk.3.indexer_compressor_ape.weight", {4,128}, GGML_TYPE_F32, std::vector<float>(512,.25f)));
    tensors.push_back(FloatTensor("blk.4.nextn.enorm.weight", {32}, GGML_TYPE_F32, std::vector<float>(32,1.f)));
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
    Check(tasks.size()==11,"split expert task count / ignored NextN weights");
    WeightMap weights;
    for (auto &task : tasks) WeightImportGGUFTensor(&weights[task.name],&task.tensor,task.fileName,task.offset,task.replaceType);
    const std::string base="model.language_model.layers.";
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
int main() {
    char directory[]="/tmp/fastllm-glm53-gguf-XXXXXX";
    try {
        Check(mkdtemp(directory)!=nullptr,"temporary directory");
        Run(directory);rmdir(directory);
        std::cout << "PASS: GLM-5.3 GGUF shard mapping, packed IQ experts, KDA decay and MLA layout\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << "\n"; return 1; }
}
