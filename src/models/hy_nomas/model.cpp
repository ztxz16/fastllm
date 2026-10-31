#include "models/hy_nomas/runtime.h"
#include <cmath>
#include <stdexcept>

namespace fastllm { namespace nomas {
namespace {
int tokenCount(int t,int h,int w) {
    // Bound public requests before allocating activations or constructing grids.
    if(t<1 || h<1 || w<1 || t>64 || h>128 || w>128 || int64_t(t)*h*w>16384)
        throw std::invalid_argument("HY-Nomas grid exceeds baseline limit (16384 tokens)");
    return t*h*w;
}
}

Tensor DiT::timeEmbedding(float timestep,const std::string &prefix) {
    if(!std::isfinite(timestep) || timestep<0 || timestep>1000)
        throw std::invalid_argument("timestep must be finite and in [0,1000]");
    float values[256];
    for(int i=0;i<128;++i) {
        float freq=std::exp(-std::log(10000.f)*float(i)/128.f);
        values[i]=std::cos(timestep*freq); values[i+128]=std::sin(timestep*freq);
    }
    return weights_.linear(rt_.activation(weights_.linear(rt_.upload(values,1,256),prefix+".mlp.0")),prefix+".mlp.2");
}

Tensor DiT::refineText(const Tensor &text,float timestep) {
    if(text.cols!=3584 || text.rows>1280) throw std::invalid_argument("expected up to 1280 valid Qwen features of width 3584");
    auto context=weights_.linear(rt_.activation(weights_.linear(rt_.mean(text),"txt_in.c_embedder.linear_1")),"txt_in.c_embedder.linear_2");
    auto c=rt_.activation(rt_.add(timeEmbedding(timestep,"txt_in.t_embedder"),context));
    auto x=weights_.linear(text,"txt_in.input_embedder");
    for(int layer=0;layer<2;++layer) {
        auto p="txt_in.individual_token_refiner.blocks."+std::to_string(layer)+".";
        auto mod=weights_.linear(c,p+"adaLN_modulation.1");
        auto nx=rt_.norm(x,false,&weights_.get(p+"norm1.weight"),&weights_.get(p+"norm1.bias"));
        auto qkv=weights_.linear(nx,p+"self_attn_qkv");
        auto a=rt_.attention(rt_.columns(qkv,0,2048),rt_.columns(qkv,2048,2048),rt_.columns(qkv,4096,2048),16);
        x=rt_.residual(x,weights_.linear(a,p+"self_attn_proj"),mod,0);
        nx=rt_.norm(x,false,&weights_.get(p+"norm2.weight"),&weights_.get(p+"norm2.bias"));
        auto mlp=weights_.linear(rt_.activation(weights_.linear(nx,p+"mlp.fc1")),p+"mlp.fc2");
        x=rt_.residual(x,mlp,mod,1);
    }
    return x;
}

Tensor DiT::forward(const Tensor &latent,const Tensor &features,float timestep,int t,int h,int w,
                    int offset,DiTCache *cache,bool commit,const Tensor *camera) {
    int tokens=tokenCount(t,h,w);
    if(latent.rows!=tokens || latent.cols!=32 || offset<0 || offset>1000000 || (commit && !cache))
        throw std::invalid_argument("invalid DiT forward input");
    bool hasCache=cache && !cache->key.empty();
    if(hasCache && (cache->key.size()!=54 || cache->value.size()!=54))
        throw std::invalid_argument("invalid DiT cache layer count");
    int history=hasCache?cache->key[0].rows:0;
    if(history+tokens+features.rows>32768) throw std::invalid_argument("DiT cache limit exceeded; start a new session");
    if(camera && (camera->rows!=t || camera->cols!=64)) throw std::invalid_argument("invalid camera embedding");
    std::vector<float> cos(size_t(tokens)*128),sin(cos.size());
    int dims[]={16,56,56};
    for(int token=0;token<tokens;++token) {
        int pos[]={token/(h*w)+offset,token/w%h,token%w},base=0;
        for(int axis=0;axis<3;++axis) {
            for(int d=0;d<dims[axis];++d) {
                float angle=float(pos[axis])*std::pow(256.f,-float(d/2*2)/dims[axis]);
                cos[size_t(token)*128+base+d]=std::cos(angle); sin[size_t(token)*128+base+d]=std::sin(angle);
            }
            base+=dims[axis];
        }
    }
    auto rope=rt_.prepareRope(cos.data(),sin.data(),tokens,128);
    auto x=weights_.linear(latent,"img_in.proj"),vec=timeEmbedding(timestep,"time_in"),txt=refineText(features,timestep);
    DiTCache next;
    for(int layer=0;layer<54;++layer) {
        auto out=block(layer,x,txt,vec,rope,hasCache?&cache->key[layer]:nullptr,hasCache?&cache->value[layer]:nullptr);
        x=out.image; txt=out.text;
        if(camera) x=cameraBlock(layer,x,*camera,h*w);
        if(commit) {
            next.key.push_back(hasCache?rt_.concat({cache->key[layer],out.key}):out.key);
            next.value.push_back(hasCache?rt_.concat({cache->value[layer],out.value}):out.value);
        }
    }
    auto mod=weights_.linear(rt_.activation(vec),"final_layer.adaLN_modulation.1");
    auto out=weights_.linear(rt_.modulate(rt_.norm(x,false),mod,0,1),"final_layer.linear");
    // Publish cache atomically after every layer and the output have succeeded.
    rt_.synchronize();
    if(commit) *cache=std::move(next);
    return out;
}

Tensor DiT::cameraEmbedding(const Tensor &extrinsics) {
    if(extrinsics.cols!=64) throw std::invalid_argument("camera expects four 4x4 matrices per latent frame");
    auto x=weights_.linear(extrinsics,"extrinsics_embedding_camctrl");
    return rt_.add(x,weights_.linear(rt_.activation(weights_.linear(x,"c2ws_hidden_states_layer1")),"c2ws_hidden_states_layer2"));
}
Tensor DiT::cameraBlock(int layer,const Tensor &image,const Tensor &camera,int spatial) {
    if(layer<0 || layer>=54 || spatial<1 || image.cols!=2048 || camera.cols!=64 || image.rows!=camera.rows*spatial)
        throw std::invalid_argument("invalid camera block shape");
    auto p="action_modules."+std::to_string(layer)+".";
    auto keyboard=weights_.linear(rt_.activation(weights_.linear(camera,p+"keyboard_embed.0")),p+"keyboard_embed.2");
    auto kv=weights_.linear(keyboard,p+"keyboard_attn_kv");
    // Released inference uses camera_chunk_size=1. Each image query attends
    // exactly one camera key: softmax is exactly one, so Q/K norm and RoPE
    // cannot affect the result. Preserve the V and output projection rounding.
    auto value=rt_.columns(kv,1024,1024);
    auto projected=weights_.linear(rt_.repeatRows(value,spatial),p+"proj_keyboard");
    return rt_.add(image,projected);
}

PixelDecoder::PixelDecoder(Runtime &rt,const std::string &path):rt_(rt),weights_(rt,path) {}
Tensor PixelDecoder::block(int layer,const Tensor &input,int t,int h,int w,int tile) {
    if(layer<0 || layer>=36 || input.rows!=tokenCount(t,h,w) || input.cols!=4096 || tile<0)
        throw std::invalid_argument("invalid PixelUMM block input");
    auto p="decoder.layers."+std::to_string(layer)+".";
    auto norm=rt_.norm(input,true,&weights_.get(p+"input_norm.weight"));
    auto q=weights_.linear(norm,p+"attn.q_proj"),k=weights_.linear(norm,p+"attn.k_proj"),v=weights_.linear(norm,p+"attn.v_proj");
    q=rt_.pixelQKRope(q,weights_.get(p+"attn.q_norm_t.weight"),weights_.get(p+"attn.q_norm_hw.weight"),t,h,w);
    k=rt_.pixelQKRope(k,weights_.get(p+"attn.k_norm_t.weight"),weights_.get(p+"attn.k_norm_hw.weight"),t,h,w);
    auto a=rt_.attention(q,k,v,32,nullptr,128,8,false,tile,h,w);
    auto x=rt_.add(input,weights_.linear(a,p+"attn.o_proj"));
    norm=rt_.norm(x,true,&weights_.get(p+"post_attn_norm.weight"));
    auto gate=weights_.linear(norm,p+"mlp.gate_proj");
    auto mlp=weights_.linear(rt_.siluMultiply(gate,weights_.linear(norm,p+"mlp.up_proj")),p+"mlp.down_proj");
    return rt_.add(x,mlp);
}
Tensor PixelDecoder::forward(const Tensor &latent,const Tensor &noise,int t,int h,int w) {
    int n=tokenCount(t,h,w); tokenCount(t,h*2,w*2);
    if(latent.rows!=n || latent.cols!=32 || noise.rows!=n*4 || noise.cols!=3072)
        throw std::invalid_argument("invalid PixelUMM latent/noise shape");
    // Official cond_proj receives a noncontiguous [B,T,H,W,C] view; its
    // matmul and bias addition have separate BF16 rounding boundaries.
    auto x=rt_.add(weights_.linear(noise,"decoder.embed"),rt_.pixelCondition(weights_.linear(latent,"decoder.cond_proj",true),t,h,w));
    for(int layer=0;layer<36;++layer) x=block(layer,x,t,h*2,w*2);
    x=rt_.norm(x,true,&weights_.get("decoder.final_norm.weight"));
    x=rt_.norm(x,true,&weights_.get("decoder.head.norm.weight"));
    return weights_.linear(x,"decoder.head.linear");
}

QwenTextEncoder::QwenTextEncoder(Runtime &rt,const std::string &directory):rt_(rt),weights_(rt,directory) {}
Tensor QwenTextEncoder::encode(const std::vector<int> &tokens) {
    if(tokens.empty() || tokens.size()>1536) throw std::invalid_argument("Qwen encoder accepts 1..1536 tokens including template");
    auto x=rt_.embedding(weights_.get("model.embed_tokens.weight"),tokens);
    // hidden_states[-3] for 28 layers is the output of layer 25. The official
    // encoder has apply_final_norm=False; do not apply model.norm or lm_head.
    for(int layer=0;layer<26;++layer) {
        auto p="model.layers."+std::to_string(layer)+".";
        auto nx=rt_.norm(x,true,&weights_.get(p+"input_layernorm.weight"));
        auto q=rt_.textRope(weights_.linear(nx,p+"self_attn.q_proj"));
        auto k=rt_.textRope(weights_.linear(nx,p+"self_attn.k_proj"));
        auto v=weights_.linear(nx,p+"self_attn.v_proj");
        x=rt_.add(x,weights_.linear(rt_.attention(q,k,v,28,nullptr,128,4,true),p+"self_attn.o_proj"));
        nx=rt_.norm(x,true,&weights_.get(p+"post_attention_layernorm.weight"));
        auto up=weights_.linear(nx,p+"mlp.up_proj"),gate=weights_.linear(nx,p+"mlp.gate_proj");
        x=rt_.add(x,weights_.linear(rt_.siluMultiply(gate,up),p+"mlp.down_proj"));
    }
    return x;
}

}} // namespace fastllm::nomas
