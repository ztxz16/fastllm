#include "quantized.h"
#include "models/minimax_h3/model_source.h"
#include "models/minimax_h3_image/quantized.h"
#include "../minimax_h3_image/quantized_checkpoint.h"
#include "../minimax_h3_image/gguf_checkpoint.h"
#include <map>
#include <set>
#include <stdexcept>

namespace fastllm::minimax_h3 {
namespace {
using Format = minimax_h3_image::WeightFormat;
using Backend = minimax_h3_image::QuantizedWeights;
using Checkpoint = minimax_h3_image::detail::QuantizedCheckpoint;
void require(bool ok, const std::string &why) {
    if(!ok) throw std::invalid_argument("H3 video weights: "+why);
}
const std::map<std::string,std::pair<int,int>> &expected() {
    static const auto shapes=[] {
    std::map<std::string,std::pair<int,int>> result;
    for(int i=0;i<52;++i) {
        auto p=i<50 ? "blocks."+std::to_string(i) : "token_refiner.blocks."+std::to_string(i-50);
        result[p+".attn.qkv_proj"]={21504,5376};
        result[p+".attn.out_proj"]={5376,7168};
        result[p+".mlp.fc1"]={28672,5376};
        result[p+".mlp.fc2"]={5376,14336};
    }
    return result;
    }();
    return shapes;
}
std::string nativeName(std::string name) {
    const std::pair<const char *,const char *> replace[] = {
        {"transformer_blocks.","blocks."},{"token_refiner.refiner_blocks.","token_refiner.blocks."},
        {".attn.to_out.0",".attn.out_proj"},{".ff.net.0.proj",".mlp.fc1"},{".ff.net.2",".mlp.fc2"}};
    for(const auto &r:replace) {
        auto pos=name.find(r.first);
        if(pos!=std::string::npos) name.replace(pos,std::string(r.first).size(),r.second);
    }
    return name;
}
struct Description {
    VideoWeightInfo info;
    std::map<std::string,Format> projections;
};
Description inspect(const std::string &path) {
    std::ifstream file(path,std::ios::binary);
    char magic[4]{}; file.read(magic,4);
    require(bool(file),"cannot read checkpoint "+path);
    Description result;
    const auto &shapes=expected();
    if(std::string(magic,4)=="GGUF") {
        minimax_h3_image::detail::GgufCheckpoint checkpoint(path);
        checkpoint.validateVideoModel();
        for(const auto &entry:shapes) {
            result.projections[entry.first]=Format::GGUF;
            result.info.bytes+=checkpoint.tensor(entry.first+".weight").bytes;
        }
        result.info.precision="gguf";
        return result;
    }
    Checkpoint checkpoint(path);
    auto &header=checkpoint.header;
    // These distinguish video from image PDD and catch incompatible exports
    // before any projection or CUDA allocation is attempted.
    for(const auto &head:std::map<std::string,std::pair<int,int>>{
        {"video_patch_proj.weight",{5376,96}},{"audio_patch_proj.weight",{5376,32}},
        {"final_layer.video_out.weight",{96,5376}},{"final_layer.audio_out.weight",{32,5376}}}) {
        const auto dtype=header[head.first]["dtype"].string_value();
        require(dtype=="F32" || dtype=="BF16" || dtype=="F16","invalid video head "+head.first);
        const auto h=header[head.first];
        require(h["shape"]==json11::Json::array{head.second.first,head.second.second},"video head shape "+head.first);
    }
    std::string error;
    const auto metadataText=header["__metadata__"]["_quantization_metadata"].string_value();
    auto metadata=metadataText.empty()?json11::Json{}:json11::Json::parse(metadataText,error);
    require(error.empty(),"invalid quantization metadata");
    std::set<Format> formats;
    for(const auto &entry:shapes) {
        const auto &name=entry.first;
        const int n=entry.second.first,k=entry.second.second;
        const auto dtype=header[name+".weight"]["dtype"].string_value();
        require(dtype=="F32" || dtype=="BF16" || dtype=="F8_E4M3" || dtype=="I8" || dtype=="U8",
                "unsupported projection dtype: "+name);
        const auto d=checkpoint.descriptor(name+".weight",dtype);
        Format format=Format::BF16;
        require(d["shape"]==json11::Json::array{n,dtype=="U8"?k/2:k},"projection shape: "+name);
        require(header[name+".bias"].is_null(),"biased video projection is not supported: "+name);
        result.info.bytes+=Checkpoint::payloadBytes(d);
        if(dtype=="F8_E4M3" || dtype=="I8") {
            auto marker=header[name+".comfy_quant"].is_object() ? checkpoint.marker(name+".comfy_quant") : metadata["layers"][name];
            const bool fp8=dtype=="F8_E4M3";
            require(marker["format"].string_value()==(fp8?"float8_e4m3fn":"int8_tensorwise"),
                    "quantization marker: "+name);
            if(!fp8) require(marker["convrot"].bool_value() && marker["convrot_groupsize"].number_value()==256,
                              "INT8 requires H256 ConvRot: "+name);
            const auto scales=checkpoint.desc(name+".weight_scale","F32",(fp8?1:size_t(n))*4);
            result.info.bytes+=Checkpoint::payloadBytes(scales);
            format=fp8?Format::FP8:Format::INT8;
        } else if(dtype=="U8") {
            const auto marker=metadata["layers"][name];
            require(marker["format"].string_value()=="nvfp4" && marker["group_size"].number_value()==16,
                    "NVFP4 layout: "+name);
            const auto scales=checkpoint.desc(name+".weight_scale","F8_E4M3",size_t(n)*k/16);
            require(scales["shape"]==json11::Json::array{n,k/16},"NVFP4 scale shape: "+name);
            checkpoint.desc(name+".weight_scale_2","F32",4);
            result.info.bytes+=Checkpoint::payloadBytes(scales)+4;
            format=Format::NVFP4;
        }
        result.projections[name]=format;
        if(format!=Format::BF16) formats.insert(format);
    }
    require(!formats.empty(),"expected quantized video projections");
    result.info.precision=formats.size()>1?"mixed":minimax_h3_image::weightFormatName(*formats.begin());
    return result;
}
} // namespace
VideoWeightInfo inspectVideoWeights(const std::string &path) { return inspect(path).info; }
struct VideoProjections::Impl {
    Runtime &rt;
    Description description;
    nomas::ParallelContext parallel;
    bool calibration=false;
    std::unique_ptr<Weights> plain;
    std::map<Format,std::unique_ptr<Backend>> backends;
    Impl(Runtime &runtime,const std::string &path,nomas::ParallelContext context,bool sequenceParallel)
        :rt(runtime),description(inspect(path)),parallel(sequenceParallel?nomas::ParallelContext{}:context) {
        require(rt.dtype()==DType::BF16,"quantized video requires BF16 activations");
        calibration=context.size()>1 && std::any_of(description.projections.begin(),description.projections.end(),
            [](const auto &entry){return entry.second==Format::NVFP4;});
        int device=0;
        auto e=cudaGetDevice(&device);
        if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
        for(const auto &entry:description.projections) {
            auto slice=minimax_h3_image::detail::ProjectionSlice(entry.first,
                expected().at(entry.first).first,expected().at(entry.first).second,
                {true,parallel.rank,parallel.size()});
            if(entry.second==Format::BF16) {
                if(!plain) plain=std::make_unique<Weights>(rt,path);
                if(parallel.size()>1) plain->shard(entry.first+".weight",slice.rowParallel?1:0,
                                                   parallel.rank,parallel.size(),slice.groups);
            } else {
                if((entry.second==Format::INT8 || entry.second==Format::GGUF) && slice.cols%256)
                    require(false,"TP input shard is not H256/block aligned; use TP2/TP4 or SP");
                auto &backend=backends[entry.second];
                if(!backend) {
                    Backend::validateDevice(entry.second,device,context.size()>1);
                    minimax_h3_image::ProjectionOptions options{true,parallel.rank,parallel.size()};
                    if(context.size()>1)
                        options.activationMaximum=[context](Tensor &x){context.group->maximum(x,context.rank);};
                    backend=Backend::load(path,entry.second,options);
                }
            }
        }
    }
};
VideoProjections::VideoProjections(Runtime &rt,const std::string &path,nomas::ParallelContext context,bool sequenceParallel)
    :impl_(std::make_unique<Impl>(rt,path,context,sequenceParallel)) {}
VideoProjections::~VideoProjections()=default;
bool VideoProjections::collectiveCalibration() const {return impl_->calibration;}
bool VideoProjections::fusedProjections() const {return impl_->backends.count(Format::NVFP4)==0;}
Tensor VideoProjections::linear(const Tensor &x,const std::string &diffusersName,DType output) {
    auto &i=*impl_;
    const auto name=nativeName(diffusersName);
    auto found=i.description.projections.find(name);
    require(found!=i.description.projections.end(),"unknown projection: "+diffusersName);
    const auto shape=expected().at(name);
    const minimax_h3_image::detail::ProjectionSlice slice(name,shape.first,shape.second,
                                                         {true,i.parallel.rank,i.parallel.size()});
    Tensor out(x.rows,slice.rows,output);
    if(found->second==Format::BF16)
        i.rt.linearInto(x,i.plain->get(name+".weight"),out);
    else
        i.backends.at(found->second)->linear(x,name,&out);
    if(!fusedProjections() && name.find(".mlp.fc1")!=std::string::npos) {
        Tensor swapped(out.rows,out.cols,out.dtype);
        const size_t pitch=size_t(out.cols)*(out.isFloat32()?4:2),half=pitch/2;
        for(int part=0;part<2;++part) {
            auto e=cudaMemcpy2DAsync(reinterpret_cast<char *>(swapped.data)+part*half,pitch,
                                    reinterpret_cast<char *>(out.data)+(1-part)*half,pitch,
                                    half,out.rows,cudaMemcpyDeviceToDevice,cudaStreamPerThread);
            if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
        }
        return swapped;
    }
    // Fused projections retain [gate, value] through LoRA and SwiGLU. NVFP4
    // preserves the original layout/GEMM shapes and separate delta rounding.
    return out;
}
void VideoProjections::erasePrefix(const std::string &prefix) {
    const auto name=nativeName(prefix);
    if(impl_->plain) impl_->plain->erasePrefix(name);
    for(auto &entry:impl_->backends) entry.second->erasePrefix(name);
}
void VideoProjections::clear() {
    if(impl_->plain) impl_->plain->clear();
    for(auto &entry:impl_->backends) entry.second->clear();
}
void VideoProjections::setHostMemoryBudget(const std::shared_ptr<nomas::HostMemoryBudget> &budget) {
    if(impl_->plain) impl_->plain->setHostMemoryBudget(budget);
}
} // namespace fastllm::minimax_h3
