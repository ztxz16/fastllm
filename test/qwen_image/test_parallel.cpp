#include "models/qwen_image/runtime.h"
#include "models/minimax_h3_image/quantized.h"
#include "quantized_partition.h"
#include "quantized_backends.h"
#include "models/minimax_h3_image/fp8.h"
#include "json11.hpp"
#include <filesystem>
#include <fstream>
#include <cstring>
#include <unistd.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace fastllm::qwen_image;
namespace backend = fastllm::minimax_h3_image;
static void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
static float bf16(float value) {
    uint32_t bits;std::memcpy(&bits,&value,4);
    bits=(bits+32767+((bits>>16)&1))&0xffff0000u;std::memcpy(&value,&bits,4);return value;
}
static void sharedScale(Runtime &rt, bool fp8) {
    // Each half row has a different range. A sparse matrix spanning both
    // halves detects local calibration accidentally replacing the global one.
    constexpr int cols=4096,outputs=32;
    const std::string name="transformer_blocks.0.attn.to_out.0";
    const std::string legacy="blocks.0.attn.out_proj";
    auto file=std::filesystem::temp_directory_path()/("qwen-tp-scale-"+std::to_string(getpid())+".safetensors");
    struct Remove {std::filesystem::path path;~Remove(){std::error_code e;std::filesystem::remove(path,e);}} cleanup{file};
    using Json=json11::Json;
    Json::object header;std::vector<char> data;
    auto add=[&](const std::string &key,const std::string &type,Json::array shape,const void *p,size_t bytes) {
        size_t begin=data.size();data.insert(data.end(),static_cast<const char *>(p),static_cast<const char *>(p)+bytes);
        header[key]=Json::object{{"dtype",type},{"shape",shape},{"data_offsets",Json::array{double(begin),double(data.size())}}};
    };
    std::vector<uint8_t> matrix(size_t(outputs)*cols);
    std::vector<float> weightScale(outputs,.25f);
    for(int row=0;row<outputs;++row)matrix[size_t(row)*cols+(row%2)*2048+row*17]=(fp8?0x38:1);
    add(name+(fp8?"._weight_qdata":".weight"),fp8?"F8_E4M3":"I8",{outputs,cols},matrix.data(),matrix.size());
    add(name+(fp8?"._weight_scale":".weight_scale"),"F32",{outputs,1},weightScale.data(),weightScale.size()*4);
    add(legacy+".weight",fp8?"F8_E4M3":"I8",{outputs,cols},matrix.data(),matrix.size());
    if(fp8)add(legacy+".weight_scale","F32",{},weightScale.data(),4);
    else add(legacy+".weight_scale","F32",{outputs,1},weightScale.data(),weightScale.size()*4);
    if(!fp8) {
        auto marker=Json(Json::object{{"format","int8_tensorwise"},{"convrot",true},{"convrot_groupsize",256}}).dump();
        add(name+".comfy_quant","U8",{int(marker.size())},marker.data(),marker.size());
        add(legacy+".comfy_quant","U8",{int(marker.size())},marker.data(),marker.size());
    }
    {std::ofstream f(file,std::ios::binary);auto h=Json(header).dump();uint64_t n=h.size();
     f.write(reinterpret_cast<const char *>(&n),8);f<<h;f.write(data.data(),data.size());require(bool(f),"scale fixture write");}
    auto load=[&](backend::ProjectionOptions options)->std::unique_ptr<backend::QuantizedWeights> {
        if(fp8)return std::make_unique<backend::Fp8Weights>(file.string(),options);
        return backend::detail::loadInt8(file.string(),options);
    };
    for(int rows : {1,17,137}) {
        std::vector<float> input(size_t(rows)*cols),scales((rows+15)/16*16);
        for(int r=0;r<rows;++r) {
            float largest=0;
            for(int c=0;c<cols;++c) {
                float value=r%3==0?0:float((c/256+r)%7-3)/8*(c<cols/2?1:32);
                input[size_t(r)*cols+c]=value;largest=std::max(largest,std::abs(value));
            }
            scales[r]=fp8?bf16(std::max(largest,bf16(1e-12f))/448.f):std::max(largest/127.f,1e-12f);
        }
        for(size_t r=rows;r<scales.size();++r)scales[r]=fp8?bf16(bf16(1e-12f)/448.f):1e-12f;
        auto x=rt.upload(input.data(),rows,cols);Tensor oracle(rows,outputs,true);
        auto single=load({});single->linearPacked(x,{name},&oracle);auto expected=rt.download(oracle);
        std::vector<float> sum(expected.size(),0);int calls=0;
        for(int rank=0;rank<2;++rank) {
            backend::ProjectionOptions options;options.qwenImage=true;options.rank=rank;options.size=2;
            options.activationMaximum=[&](Tensor &scale) {
                require(scale.size()==scales.size(),"calibration scale shape");++calls;
                require(cudaMemcpyAsync(scale.data,scales.data(),scales.size()*4,cudaMemcpyHostToDevice,
                                         cudaStreamPerThread)==cudaSuccess,"share scale upload");
            };
            auto shard=load(options);Tensor partial(rows,outputs,true);
            shard->linearPacked(rt.columns(x,rank*cols/2,cols/2),{name},&partial);
            auto actual=rt.download(partial);for(size_t i=0;i<sum.size();++i)sum[i]+=actual[i];
        }
        require(calls==2,"row projection did not share calibration");
        for(size_t i=0;i<sum.size();++i)require(sum[i]==expected[i],"shared-scale TP projection differs from full row");
        // H3 video supplies a calibration callback for NVFP4. FP8/INT8 must
        // retain their previous local calibration even when it is present.
        for(int rank=0;rank<2;++rank) {
            backend::ProjectionOptions options;options.comfyVideo=true;options.rank=rank;options.size=2;
            auto local=rt.columns(x,rank*cols/2,cols/2);
            auto original=load(options);
            auto reference=rt.download(original->linear(local,legacy));
            options.activationMaximum=[](Tensor &) {throw std::runtime_error("Qwen calibration leaked into H3");};
            auto guarded=load(options);
            require(rt.download(guarded->linear(local,legacy))==reference,"H3 calibration path changed");
        }
    }
}
#ifdef FASTLLM_QWEN_TEST_NCCL
static void bf16Reduction() {
    fastllm::nomas::ParallelGroup group({0,1});
    // Odd counts and differently aligned views must make the same collective
    // choice on both ranks. Include either side of the peer-reduction limit.
    for(int count : {1,7,8,137,4096*1024,4096*1025}) {
        const int repeats=count<1024?24:3;
        std::vector<float> expected(count);
        auto value=[](int rank,int index) {return bf16(std::sin(float(index)*.013f+rank)*3.7f);};
        for(int j=0;j<count;++j) {
            expected[j]=bf16(value(0,j)+value(1,j));
            for(int k=1;k<repeats;++k)expected[j]=bf16(expected[j]+expected[j]);
        }
        group.run([&](int rank) {
            Runtime rt(rank);
            fastllm::nomas::TensorPool pool(rank,1u<<20);
            fastllm::nomas::TensorPool::Scope scope(pool);
            std::vector<float> input(count+2,-19.f);
            for(int j=0;j<count;++j)input[j+rank]=value(rank,j);
            auto storage=rt.upload(input.data(),count+2,1);
            auto x=storage.rowSlice(rank,count);
            for(int k=0;k<repeats;++k) {
                x=group.sumActivation(std::move(x),rank);
            }
            require(rt.download(x)==expected,"BF16 peer reduction differs from rounded sum");
            auto whole=rt.download(storage);
            if(rank)require(whole.front()==-19.f,"BF16 peer reduction overwrote leading guard");
            require(whole.back()==-19.f,"BF16 peer reduction overwrote trailing guard");
        });
    }
    bool failed=false;
    try {
        group.run([&](int rank) {
            Runtime rt(rank);float value=rank;auto x=rt.upload(&value,1,1);
            x=group.sumActivation(std::move(x),rank);
            if(rank==1)throw std::runtime_error("peer-reduction-test");
            x=group.sumActivation(std::move(x),rank);
        });
    } catch(const std::exception &) {failed=true;}
    require(failed,"BF16 reduction did not propagate rank failure");
    group.run([&](int rank) {
        Runtime rt(rank);float value=rank+1;auto x=rt.upload(&value,1,1);
        x=group.sumActivation(std::move(x),rank);
        require(rt.download(x)==std::vector<float>{3.f},"BF16 reduction recovery failed");
    });
}
static void scaledReduction() {
    fastllm::nomas::ParallelGroup group({0,1});
    for (auto shape : {std::pair<int,int>{1,1},{1,7},{17,64},{63,4096},{64,4096},{257,2048},{1024,4096},{1025,4096}})
        for(auto dtype : {fastllm::nomas::DType::BF16,fastllm::nomas::DType::FP16,fastllm::nomas::DType::FP32}) {
            const auto [rows,cols]=shape;const size_t count=size_t(rows)*cols;
            std::vector<float> xs(rows),ws(cols),expected(count);
            for(int r=0;r<rows;++r)xs[r]=.017f+float(r%7)*.013f;
            for(int c=0;c<cols;++c)ws[c]=.37f+float(c%11)*.017f;
            auto value=[&](int rank,size_t i) {
                // Casting each partial to FP32 would lose integer bits here.
                return rank==0?int32_t((1<<24)+1+int(i%7)):int32_t(-(1<<24)+int(i%3));
            };
            for(size_t i=0;i<count;++i) {
                float v=(float(value(0,i)+value(1,i))*xs[i/cols])*ws[i%cols];
                expected[i]=dtype==fastllm::nomas::DType::BF16?bf16(v):
                    dtype==fastllm::nomas::DType::FP16?__half2float(__float2half_rn(v)):v;
            }
            group.run([&](int rank) {
                Runtime rt(rank);
                fastllm::nomas::TensorPool pool(rank,1u<<20);
                fastllm::nomas::TensorPool::Scope scope(pool);
                Tensor rawStorage(rows+2,cols,true);
                auto raw=rawStorage.rowSlice(rank,rows);
                std::vector<float> guard(size_t(rows+2)*cols,-19.f);
                auto outputStorage=rt.cast(rt.upload(guard.data(),rows+2,cols),dtype);
                auto out=outputStorage.rowSlice(rank,rows);
                Tensor x(1,rows,true),w(1,cols,true);
                std::vector<int32_t> bits(count);
                for(size_t i=0;i<count;++i)bits[i]=value(rank,i);
                require(cudaMemcpy(x.data,xs.data(),rows*4,cudaMemcpyHostToDevice)==cudaSuccess,"activation scale upload");
                require(cudaMemcpy(w.data,ws.data(),cols*4,cudaMemcpyHostToDevice)==cudaSuccess,"weight scale upload");
                for(int repeat=0;repeat<3;++repeat) {
                    require(cudaMemcpy(raw.data,bits.data(),count*4,cudaMemcpyHostToDevice)==cudaSuccess,"raw accumulator upload");
                    group.sumScaledInt32(raw,x,w,out,rank);
                    require(rt.download(out)==expected,"INT32 reduction/scaling differs from full accumulator");
                }
                auto guards=rt.download(outputStorage);
                if(rank)require(guards.front()==-19.f,"INT32 reduction overwrote leading output guard");
                require(guards.back()==-19.f,"INT32 reduction overwrote trailing output guard");
            });
        }
    bool failed=false;
    try {
        group.run([&](int rank) {
            Runtime rt(rank,true);float one=1;auto scale=rt.upload(&one,1,1);
            Tensor raw(1,1,true),out(1,1);int32_t value=rank+1;
            require(cudaMemcpy(raw.data,&value,4,cudaMemcpyHostToDevice)==cudaSuccess,"raw recovery upload");
            group.sumScaledInt32(raw,scale,scale,out,rank);
            if(rank==1)throw std::runtime_error("scaled-peer-reduction-test");
            group.sumScaledInt32(raw,scale,scale,out,rank);
        });
    } catch(const std::exception &) {failed=true;}
    require(failed,"INT32 reduction did not propagate rank failure");
    group.run([&](int rank) {
        Runtime rt(rank,true);float one=1;auto scale=rt.upload(&one,1,1);
        Tensor raw(1,1,true),out(1,1);int32_t value=rank+1;
        require(cudaMemcpy(raw.data,&value,4,cudaMemcpyHostToDevice)==cudaSuccess,"raw recovery upload");
        group.sumScaledInt32(raw,scale,scale,out,rank);
        require(rt.download(out)==std::vector<float>{3.f},"INT32 reduction recovery failed");
    });
}
#endif
int main() {
    try {
        // Check original byte/block offsets, grouped output rows and replicated auxiliary matrices.
        for (int size : {2, 4, 8}) for (int rank = 0; rank < size; ++rank) {
            backend::ProjectionOptions options;
            options.qwenImage = true; options.size = size; options.rank = rank;
            for (int groups : {1, 2, 3}) {
                backend::detail::ProjectionSlice s("transformer_blocks.0.attn.to_q", groups*4096, 4096,
                                                    options, groups);
                require(s.rows == groups*4096/size && s.cols == 4096, "column partition shape");
                for (int g = 0; g < groups; ++g)
                    require(s.sourceRow(g*4096/size) == g*4096+rank*4096/size, "packed group offset");
            }
            backend::detail::ProjectionSlice gate("transformer_blocks.0.img_mlp.gate_up",24576,4096,options);
            require(gate.sourceRow(gate.rows/2) == 12288+rank*12288/size, "fused gate/up partition");
            backend::detail::ProjectionSlice row("transformer_blocks.0.img_mlp.out",4096,12288,options);
            require(row.rows==4096 && row.cols==12288/size && row.firstColumn()==rank*row.cols,
                    "row partition offset");
            backend::detail::ProjectionSlice replicated("proj_out",64,4096,options);
            require(replicated.rows==64 && replicated.cols==4096 && replicated.sourceRow(0)==0,
                    "replicated output projection");
        }
        int count=0;
        if (cudaGetDeviceCount(&count)!=cudaSuccess || !count) return 77;
#ifdef FASTLLM_QWEN_TEST_NCCL
        if(count>=2){scaledReduction();bf16Reduction();}
#endif
        Runtime rt(0), full(0,true);
        cudaDeviceProp device{};cudaGetDeviceProperties(&device,0);
        if(device.major>=8)sharedScale(rt,false);
        if(device.major*10+device.minor>=89)sharedScale(rt,true);
        // Compare each head shard with the unchanged full-width result. Odd
        // token counts exercise the final half tile for TP8, including guards.
        for (int rows : {1, 17, 137}) {
            std::vector<float> input(size_t(rows)*12288), norm(128), cs(size_t(rows)*128), sn(cs.size());
            for (size_t i=0;i<input.size();++i) input[i]=std::sin(float(i)*.031f);
            for (int i=0;i<128;++i) norm[i]=1.f+float(i%7)/32;
            for (size_t i=0;i<cs.size();++i) { cs[i]=std::cos(float(i)*.003f);sn[i]=std::sin(float(i)*.003f); }
            auto x=rt.upload(input.data(),rows,12288), w=rt.upload(norm.data(),1,128);
            auto rope=rt.prepareRope(cs.data(),sn.data(),rows,128);
            Tensor q(rows,4096),k(rows,4096),v(rows,4096);
            rt.qkvNormRope(x,w,w,rope,q,k,v);
            for (int size : {2,4,8}) for (int rank=0;rank<size;++rank) {
                int cols=4096/size;
                std::vector<float> localValues(size_t(rows)*3*cols);
                for (int r=0;r<rows;++r) for (int g=0;g<3;++g) for (int c=0;c<cols;++c)
                    localValues[(size_t(r)*3+g)*cols+c]=input[size_t(r)*12288+g*4096+rank*cols+c];
                auto local=rt.upload(localValues.data(),rows,3*cols);
                Tensor a(rows,cols),b(rows,cols),c(rows,cols);
                rt.qkvNormRope(local,w,w,rope,a,b,c);
                require(rt.download(a)==rt.download(rt.columns(q,rank*cols,cols)),"TP Q RoPE");
                require(rt.download(b)==rt.download(rt.columns(k,rank*cols,cols)),"TP K RoPE");
                require(rt.download(c)==rt.download(rt.columns(v,rank*cols,cols)),"TP V layout");
            }
            auto fq=full.cast(rt.columns(x,0,4096),fastllm::nomas::DType::FP32);
            auto fk=full.cast(rt.columns(x,4096,4096),fastllm::nomas::DType::FP32);
            auto fw=full.cast(w,fastllm::nomas::DType::FP32);
            Tensor oq(rows,4096,true),ok(rows,4096,true);
            full.qkNormRope(fq,fk,fw,fw,rope,oq,ok);
            for (int size : {2,4,8}) for (int rank=0;rank<size;++rank) {
                int cols=4096/size;
                auto lq=full.columns(fq,rank*cols,cols),lk=full.columns(fk,rank*cols,cols);
                Tensor a(rows,cols,true),b(rows,cols,true);
                full.qkNormRope(lq,lk,fw,fw,rope,a,b);
                require(full.download(a)==full.download(full.columns(oq,rank*cols,cols)),"FP32 TP Q RoPE");
                require(full.download(b)==full.download(full.columns(ok,rank*cols,cols)),"FP32 TP K RoPE");
            }
        }
        std::cout << "Qwen TP partition, shared-scale reduction and local-head kernels passed\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
