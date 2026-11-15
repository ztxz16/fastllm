#include "quantized.h"
#include "../../src/models/minimax_h3_image/quantized_backends.h"
#include "json11.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>
using namespace fastllm::qwen_image;
using fastllm::nomas::DType;
using Json = json11::Json;
namespace fs = std::filesystem;
namespace backend = fastllm::minimax_h3_image::detail;
namespace {
void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F f, const std::string &reason = "") {
    bool rejected = false;
    try { f(); } catch (const std::exception &e) {
        rejected = reason.empty() || std::string(e.what()).find(reason) != std::string::npos;
    }
    require(rejected, "invalid checkpoint accepted");
}
float bf16(float x) {
    uint32_t bits; std::memcpy(&bits, &x, 4);
    bits = (bits + 32767 + ((bits >> 16) & 1)) & 0xffff0000;
    std::memcpy(&x, &bits, 4); return x;
}
const std::vector<std::string> qkv{
    "transformer_blocks.0.attn.to_q", "transformer_blocks.0.attn.to_k", "transformer_blocks.0.attn.to_v"};
void int8Fixture(const fs::path &path, const std::string &fault = "") {
    Json::object header; std::vector<char> data;
    auto add = [&](const std::string &name, const std::string &dtype, Json::array shape, const void *p, size_t n) {
        const size_t begin = data.size();
        data.insert(data.end(), static_cast<const char *>(p), static_cast<const char *>(p) + n);
        header[name] = Json::object{{"dtype", dtype}, {"shape", shape},
            {"data_offsets", Json::array{double(begin), double(data.size())}}};
    };
    auto names = qkv; names.push_back("transformer_blocks.0.img_mlp.out");
    for (int part = 0; part < 4; ++part) {
        const int cols = part == 3 ? 12288 : 4096, rows = 4096;
        std::vector<int8_t> weight(size_t(rows)*cols);
        std::vector<float> scales(rows);
        for (int r = 0; r < rows; ++r) {
            weight[size_t(r)*cols + (r+part*7)%cols] = r%2 ? -3 : 3;
            scales[r] = .125f * (1 << (r%3));
        }
        if (fault == "nonfinite") scales[0] = NAN;
        add(names[part]+".weight", "I8", {rows,cols}, weight.data(), weight.size());
        if (fault != "scale") add(names[part]+".weight_scale", "F32", {rows,1}, scales.data(), scales.size()*4);
        Json marker = Json::object{{"format","int8_tensorwise"},{"convrot",true},
            {"convrot_groupsize",fault == "rotation" ? 128 : 256}};
        const auto text = marker.dump();
        add(names[part]+".comfy_quant", "U8", {int(text.size())}, text.data(), text.size());
    }
    if (fault == "partial") header.erase(qkv[1]+".weight");
    if (fault == "mixed-fp8") {
        auto descriptor = header[qkv[0] + ".weight"].object_items();
        descriptor["dtype"] = "F8_E4M3";
        header["transformer_blocks.1.attn.to_q._weight_qdata"] = descriptor;
    }
    const auto text = Json(header).dump(); const uint64_t n = text.size();
    std::ofstream f(path,std::ios::binary); f.write(reinterpret_cast<const char *>(&n),8);f<<text;
    f.write(data.data(),data.size());require(bool(f),"fixture write");
}
// Tiny mixed-type GGUF fixture; no H3 name mapping or model validation involved.
void ggufFixture(const fs::path &path, int cols = 256, int outputs = 32) {
    std::vector<char> header, payload;
    auto number = [&](uint64_t v,int bytes) {for(int i=0;i<bytes;++i) header.push_back(char(v>>(8*i)));};
    auto string = [&](const std::string &s) {number(s.size(),8);header.insert(header.end(),s.begin(),s.end());};
    number(0x46554747,4);number(3,4);number(3,8);number(0,8);
    for (int part=0;part<3;++part) {
        const int type=part==1?8:2, blockBytes=type==8?34:18;
        string(qkv[part]+".weight");number(2,4);number(cols,8);number(outputs,8);number(type,4);number(payload.size(),8);
        for(int row=0;row<outputs;++row) for(int block=0;block<cols/32;++block) {
            const auto start=payload.size();payload.resize(start+blockBytes);
            payload[start]=0;payload[start+1]=0x3c; // FP16 scale = 1
            for(int i=2;i<blockBytes;++i) payload[start+i]=type==8?char(part+1):char((8+part+1)*17);
        }
        payload.resize((payload.size()+31)/32*32);
    }
    header.resize((header.size()+31)/32*32);
    std::ofstream f(path,std::ios::binary);f.write(header.data(),header.size());f.write(payload.data(),payload.size());
}
}
int main() {
    int count=0;cudaDeviceProp prop{};
    if(cudaGetDeviceCount(&count)!=cudaSuccess || !count || cudaGetDeviceProperties(&prop,0)!=cudaSuccess || prop.major<8) return 77;
    auto root=fs::temp_directory_path()/("qwen-quant-"+std::to_string(getpid()));fs::create_directories(root);
    try {
        Runtime rt(0);const auto file=root/"weights.safetensors";int8Fixture(file);
        auto weights=QuantizedProjections::load(rt,file.string());require(bool(weights),"INT8 detection");
        require(weights->loadedBytes()==0,"eager allocation");
        for(int cols:{4096,12288}) for(int rows:{1,17,137}) {
            std::vector<float> x(size_t(rows)*cols), scale(rows);
            // Constant groups make dense H256's independently computed result
            // exactly representable, including zero rows and BF16 boundaries.
            for(int r=0;r<rows;++r) {
                float largest=0;
                for(int c=0;c<cols;++c) {
                    x[size_t(r)*cols+c]=float((c/256+r)%7-3)/8;
                    largest=std::max(largest,std::abs(x[size_t(r)*cols+c]));
                }
                scale[r]=std::max(largest/127,1e-12f);
            }
            auto names=cols==4096?qkv:std::vector<std::string>{"transformer_blocks.0.img_mlp.out"};
            auto input=rt.upload(x.data(),rows,cols);
            Tensor storage(rows+3,int(names.size())*4096), out=storage.rowSlice(3,rows);
            auto actual=rt.download(weights->linear(input,names,&out));
            for(int r=0;r<rows;++r) for(size_t part=0;part<names.size();++part) for(int c=0;c<4096;++c) {
                const int index=(c+(cols==4096?int(part):3)*7)%cols;
                const float quant=std::nearbyint(x[size_t(r)*cols+index]/scale[r]);
                const float expected=bf16((quant*(c%2?-3:3)*scale[r])*(.125f*(1<<(c%3))));
                require(actual[size_t(r)*out.cols+part*4096+c]==expected,"INT8 packed order, rotation, scale or tail");
            }
            if(cols==4096) {
                auto single=rt.download(weights->linear(input,{qkv[1]}));
                for(int r=0;r<rows;++r) for(int c=0;c<4096;++c)
                    require(single[size_t(r)*4096+c]==actual[size_t(r)*out.cols+4096+c],"packed cache collision");
            }
            require(weights->loadedBytes()>0,"INT8 byte accounting");weights->clear();require(weights->loadedBytes()==0,"INT8 release");
        }
        Runtime fp32(0,true);rejects([&]{QuantizedProjections::load(fp32,file.string());});weights.reset();
        for(const auto *fault:{"scale","rotation","partial"}) {int8Fixture(file,fault);rejects([&]{QuantizedProjections::load(rt,file.string());});}
        int8Fixture(file,"nonfinite");weights=QuantizedProjections::load(rt,file.string());std::vector<float> zeros(4096);
        rejects([&]{weights->linear(rt.upload(zeros.data(),1,4096),qkv);});weights.reset();
        int8Fixture(file, "mixed-fp8");
        rejects([&] { QuantizedProjections::load(rt, file.string()); }, "mixed FP8 and INT8");
        const auto gguf=root/"mixed.gguf";ggufFixture(gguf);
        rejects([&]{QuantizedProjections::load(rt,gguf.string());}); // Valid GGUF, wrong Qwen architecture.
        auto mixed=backend::loadGguf(gguf.string(),{});
        for(int rows:{1,17,137,513}) {
            std::vector<float> x(size_t(rows)*256,1);auto input=rt.upload(x.data(),rows,256);
            Tensor storage(rows+2,96), out=storage.rowSlice(2,rows);auto actual=rt.download(mixed->linearPacked(input,qkv,&out));
            for(int r=0;r<rows;++r) for(int part=0;part<3;++part) for(int c=0;c<32;++c)
                require(actual[size_t(r)*96+part*32+c]==float(256*(part+1)),"mixed GGUF packed output order");
            require(mixed->loadedBytes()>0,"GGUF byte accounting");mixed->clear();require(mixed->loadedBytes()==0,"GGUF release");
        }
        // Real Qwen width triggers direct writes into three interleaved
        // column ranges. Distinct Q4_0/Q8_0 constants catch packing/stride
        // errors, and sentinels cover both output dtypes and caller views.
        const auto wide = root/"wide-qkv.gguf";
        ggufFixture(wide, 4096, 4096);
        auto projections = backend::loadGguf(wide.string(), {});
        for (int rows : {513, 1024, 1025}) for (auto dtype : {DType::BF16, DType::FP32}) {
            auto input = rt.upload(std::vector<float>(size_t(rows)*4096, 1).data(), rows, 4096);
            auto storage = rt.cast(rt.upload(std::vector<float>(size_t(rows+2)*12288, -123).data(),
                                            rows+2, 12288), dtype);
            auto out = storage.rowSlice(1, rows);
            for (bool prefill : {true, false}) {
                projections->setPrefill(prefill);
                projections->linearPacked(input, qkv, &out);
                auto actual = rt.download(storage);
                for (size_t i = 0; i < actual.size(); ++i) {
                    const bool interior = i >= 12288 && i < size_t(rows+1)*12288;
                    const float expected = interior ? float(4096*(1+(i%12288)/4096)) : -123.f;
                    require(actual[i] == expected, "GGUF direct QKV stride/type/output guard");
                }
            }
        }
        projections->clear(); require(projections->loadedBytes()==0, "GGUF direct QKV release");
        fs::remove_all(root);std::cout<<"PASS Qwen INT8/ mixed GGUF packing, widths, tails, destinations, release and malformed metadata\n";
    } catch(const std::exception &e) {fs::remove_all(root);std::cerr<<e.what()<<'\n';return 1;}
}
