#include "models/lingbot_world/runtime.h"
#include <cmath>
#include <stdexcept>

namespace fastllm { namespace lingbot {
namespace {
constexpr int Dim=1536, Layers=30, TextLength=512;
void require(bool ok,const char *message) { if(!ok) throw std::invalid_argument(message); }
}
DiT::DiT(Runtime &rt,const std::string &directory)
    :rt_(rt),weights_(rt,directory),full_(rt.fp32,directory) {}
void DiT::clearCamera() {
    cameraInput_={}; cameraFeatures_={}; cameraModulation_.clear();
}
Tensor DiT::projectCamera(const Tensor &camera,int frames,int h,int w) {
    auto cam=linear(rt_.patch(rt_.cast(camera,false),frames,h,w),"patch_embedding_wancamctrl");
    return rt_.add(cam,linear(rt_.activation(linear(cam,"c2ws_hidden_states_layer1")),"c2ws_hidden_states_layer2"));
}
void DiT::prepareCamera(const Tensor &camera,int frames,int h,int w) {
    require(camera.rows==frames*h*w && camera.cols==384,"invalid LingBot camera tensor");
    clearCamera();
    cameraFeatures_=projectCamera(camera,frames,h,w);
    for(int layer=0;layer<Layers;++layer) {
        auto p="blocks."+std::to_string(layer)+".";
        auto cam=linear(rt_.activation(linear(cameraFeatures_,p+"cam_injector_layer1")),p+"cam_injector_layer2");
        cam=rt_.add(cam,cameraFeatures_);
        cameraModulation_.emplace_back(linear(cam,p+"cam_scale_layer"),linear(cam,p+"cam_shift_layer"));
    }
    cameraInput_=camera; cameraFrames_=frames; cameraHeight_=h; cameraWidth_=w;
}
Tensor DiT::linear(const Tensor &x,const std::string &prefix) {
    return weights_.linear(rt_.cast(x,false),prefix);
}
Tensor DiT::fullLinear(const Tensor &x,const std::string &prefix) {
    return rt_.wanLinearFP32(x,full_.get(prefix+".weight"),full_.get(prefix+".bias"));
}
Tensor DiT::rms(const Tensor &x,const std::string &prefix) {
    // Wan first rounds the normalized value to the projection dtype, then
    // multiplies by the original FP32 RMS weight before rotary embedding.
    return rt_.wanRms(x,full_.get(prefix+".weight"));
}
Tensor DiT::block(int layer,const Tensor &input,const Tensor &time,
                  const Tensor &context,const Tensor *camera,int frames,
                  int gh,int gw,Cache &cache,bool commit) {
    require(layer>=0 && layer<Layers && input.cols==Dim && input.rows==frames*gh*gw &&
        time.rows==1 && time.cols==6*Dim && time.fp32 && context.cols==Dim,
        "invalid LingBot block input");
    if(cache.keys.empty()) { cache.keys.resize(Layers); cache.values.resize(Layers); }
    if(cache.textKeys.empty()) { cache.textKeys.resize(Layers); cache.textValues.resize(Layers); }
    auto p="blocks."+std::to_string(layer)+".";
    auto mod=rt_.add(time,full_.get(p+"modulation"));
    auto part=[&](int i) { return rt_.columns(mod,i*Dim,Dim); };
    auto x=input;
    auto norm=rt_.affine(rt_.wanLayerNorm(x),part(1),part(0));
    auto q=linear(norm,p+"self_attn.q"),k=linear(norm,p+"self_attn.k");
    q=rms(q,p+"self_attn.norm_q"); k=rms(k,p+"self_attn.norm_k");
    q=rt_.rope3d(q,frames,gh,gw,cache.frames);
    k=rt_.rope3d(k,frames,gh,gw,cache.frames);
    auto v=linear(norm,p+"self_attn.v");
    auto append=[&](const Tensor &history,const Tensor &current) {
        if(!history.rows) return current;
        int keep=(cache.window-frames)*gh*gw;
        if(history.rows<=keep) return rt_.concat({history,current});
        int sink=cache.sink*gh*gw, tail=keep-sink;
        require(tail>=0,"LingBot cache window must fit sink and current chunk");
        std::vector<Tensor> pieces;
        if(sink) pieces.push_back(history.rowSlice(0,sink));
        if(tail) pieces.push_back(history.rowSlice(history.rows-tail,tail));
        pieces.push_back(current);
        return rt_.concat(pieces);
    };
    auto allK=append(cache.keys[layer],k),allV=append(cache.values[layer],v);
    auto self=linear(rt_.wanAttention(q,allK,allV),p+"self_attn.o");
    x=rt_.gated(x,self,part(2));
    if(commit) { cache.keys[layer]=allK; cache.values[layer]=allV; }

    if(camera) {
        if(cameraInput_.rows && camera->data==cameraFeatures_.data) {
            const auto &mod=cameraModulation_[layer];
            x=rt_.affine(x,mod.first,mod.second,true);
        } else {
            auto cam=linear(rt_.activation(linear(*camera,p+"cam_injector_layer1")),p+"cam_injector_layer2");
            cam=rt_.add(cam,*camera);
            x=rt_.affine(x,linear(cam,p+"cam_scale_layer"),linear(cam,p+"cam_shift_layer"),true);
        }
    }
    auto cross=rt_.wanLayerNorm(x,&full_.get(p+"norm3.weight"),&full_.get(p+"norm3.bias"));
    q=rt_.cast(rms(linear(cross,p+"cross_attn.q"),p+"cross_attn.norm_q"),false);
    if(!cache.textKeys[layer].rows) {
        cache.textKeys[layer]=rt_.cast(rms(linear(context,p+"cross_attn.k"),p+"cross_attn.norm_k"),false);
        cache.textValues[layer]=linear(context,p+"cross_attn.v");
    }
    cross=linear(rt_.wanAttention(q,cache.textKeys[layer],cache.textValues[layer]),p+"cross_attn.o");
    x=rt_.add(x,rt_.cast(cross,true));
    auto ff=rt_.affine(rt_.wanLayerNorm(x),part(4),part(3));
    ff=linear(ff,p+"ffn.0");
    ff=linear(rt_.activation(ff,true),p+"ffn.2");
    return rt_.gated(x,ff,part(5));
}
Tensor DiT::forward(const Tensor &latent,const Tensor &condition,const Tensor &text,
                    const Tensor *camera,float timestep,int frames,int h,int w,
                    Cache &cache,bool commit) {
    require(frames>0 && h>0 && w>0 && h%2==0 && w%2==0 && latent.rows==frames*h*w &&
        latent.cols==16 && condition.rows==latent.rows && condition.cols==20 &&
        text.rows>0 && text.rows<=TextLength && text.cols==4096 && std::isfinite(timestep) &&
        cache.window>=cache.sink+frames,"invalid LingBot DiT input");
    // PyTorch Conv3d rounds its convolution result before adding the bias.
    auto x=rt_.patchConv2d(rt_.join(rt_.cast(latent,false),rt_.cast(condition,false)),
        weights_.get("patch_embedding.weight"),weights_.get("patch_embedding.bias"),frames,h,w);
    std::vector<float> embedding(256);
    for(int i=0;i<128;++i) {
        double angle=double(timestep)*std::pow(10000.,-double(i)/128.);
        embedding[i]=float(std::cos(angle)); embedding[i+128]=float(std::sin(angle));
    }
    auto timeInput=rt_.repeatRows(rt_.fp32.upload(embedding.data(),1,256),x.rows);
    auto allTime=fullLinear(timeInput,"time_embedding.0");
    allTime=rt_.activation(allTime);
    allTime=fullLinear(allTime,"time_embedding.2");
    auto mod=fullLinear(rt_.activation(allTime),"time_projection.1").rowSlice(0,1);
    auto time=allTime.rowSlice(0,1);
    auto context=rt_.cast(text,false);
    if(context.rows<TextLength) {
        std::vector<float> zero(size_t(TextLength-context.rows)*4096,0.f);
        context=rt_.concat({context,rt_.upload(zero.data(),TextLength-context.rows,4096)});
    }
    context=linear(context,"text_embedding.0");
    context=rt_.activation(context,true);
    context=linear(context,"text_embedding.2");
    Tensor cam;
    if(camera) {
        require(camera->rows==latent.rows && camera->cols==384,"invalid LingBot camera tensor");
        cam=camera->data==cameraInput_.data && frames==cameraFrames_ && h==cameraHeight_ && w==cameraWidth_
            ?cameraFeatures_:projectCamera(*camera,frames,h,w);
    }
    for(int i=0;i<Layers;++i) {
        x=block(i,x,mod,context,camera?&cam:nullptr,frames,h/2,w/2,cache,commit);
    }
    auto head=full_.get("head.modulation");
    auto shift=rt_.add(time,rt_.columns(head,0,Dim));
    auto scale=rt_.add(time,rt_.columns(head,Dim,Dim));
    x=fullLinear(rt_.affine(rt_.wanLayerNorm(x),scale,shift),"head.head");
    if(commit) cache.frames+=frames;
    // The official unpatchify uses einsum inside BF16 autocast, even though
    // it only permutes dimensions. Preserve that rounding before returning
    // FP32 flow to the scheduler.
    return rt_.cast(rt_.patch(rt_.cast(x,false),frames,h,w,true),true);
}
}} // namespace fastllm::lingbot
