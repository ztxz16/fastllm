#include "models/lingbot_world/vae.h"
#include <algorithm>
#include <stdexcept>

namespace fastllm { namespace lingbot {
namespace {
void require(bool ok,const char *message) { if(!ok) throw std::invalid_argument(message); }
void validateState(VAEState &s,int h,int w) {
    require(!s.height || (s.height==h && s.width==w),"Wan VAE stream resolution cannot change");
    s.height=h; s.width=w;
}
}
VAE::Tensor VAE::pointwise(const Tensor &x,const std::string &p) {
    return rt_.conv2d(x,convWeights_.get(p+".weight"),convWeights_.optional(p+".bias"),1,x.rows,1);
}
VAE::Tensor VAE::causal(const Tensor &x,const std::string &p,int t,int h,int w,VAEState &s) {
    auto &weight=convWeights_.get(p+".weight");
    if(weight.cols==x.cols) return pointwise(x,p);
    require(weight.cols==x.cols*27,"unsupported Wan VAE causal kernel");
    auto found=s.features.find(p);
    auto prior=found==s.features.end()?Tensor{}:found->second;
    auto input=prior.rows?rt_.concat({prior,x}):x;
    int history=prior.rows/(h*w);
    // A rowSlice retains its complete backing allocation. Copy just the
    // causal history so a two-frame cache does not pin a six-frame chunk.
    s.features[p]=rt_.concat({input.rowSlice(std::max(0,input.rows-2*h*w),std::min(input.rows,2*h*w))});
    return rt_.conv3d(input,weight,convWeights_.optional(p+".bias"),t+history,h,w,3,3,1,2-history);
}
VAE::Tensor VAE::residual(const Tensor &x,const std::string &p,int t,int h,int w,VAEState &s) {
    auto skip=convWeights_.optional(p+".shortcut.weight")?pointwise(x,p+".shortcut"):x;
    auto y=rt_.activation(rt_.channelNorm(x,weights_.get(p+".residual.0.gamma")));
    y=causal(y,p+".residual.2",t,h,w,s);
    y=rt_.activation(rt_.channelNorm(y,weights_.get(p+".residual.3.gamma")));
    return rt_.add(causal(y,p+".residual.6",t,h,w,s),skip);
}
VAE::Tensor VAE::middle(const Tensor &input,const std::string &p,int t,int h,int w,VAEState &s) {
    auto x=residual(input,p+".0",t,h,w,s);
    auto y=rt_.channelNorm(x,weights_.get(p+".1.norm.gamma"));
    auto qkv=pointwise(y,p+".1.to_qkv");
    std::vector<Tensor> frames;
    for(int i=0;i<t;++i) {
        auto frame=qkv.rowSlice(i*h*w,h*w);
        frames.push_back(rt_.spatialAttention(rt_.columns(frame,0,x.cols),rt_.columns(frame,x.cols,x.cols),
                                              rt_.columns(frame,2*x.cols,x.cols)));
    }
    y=rt_.concat(frames);
    y=pointwise(y,p+".1.proj");
    return residual(rt_.add(x,y),p+".2",t,h,w,s);
}
VAE::Tensor VAE::resample(const Tensor &input,const std::string &p,int &t,int &h,int &w,
                         bool up,bool temporal,VAEState &s) {
    auto x=input;
    auto key=p+".time_conv";
    if(up && temporal) {
        auto found=s.features.find(key);
        if(found==s.features.end()) {
            // First frame passes through; the next chunk has zero temporal history.
            s.features.emplace(key,Tensor{});
        } else {
            auto prior=found->second;
            auto joined=prior.rows?rt_.concat({prior,x}):x;
            int history=prior.rows/(h*w);
            Tensor next=rt_.concat({joined.rowSlice(std::max(0,joined.rows-2*h*w),std::min(joined.rows,2*h*w))});
            if(next.rows<h*w*2) {
                std::vector<float> zeros(size_t(h)*w*x.cols,0.f);
                next=rt_.concat({rt_.cast(rt_.upload(zeros.data(),h*w,x.cols),x.fp32),next});
            }
            x=rt_.conv3d(joined,convWeights_.get(key+".weight"),convWeights_.optional(key+".bias"),
                         t+history,h,w,3,1,1,2-history);
            s.features[key]=next;
            x=rt_.temporalShuffle(x,t,h,w); t*=2;
        }
    }
    std::vector<Tensor> frames;
    auto &weight=convWeights_.get(p+".resample.1.weight");
    auto bias=convWeights_.optional(p+".resample.1.bias");
    for(int i=0;i<t;++i) {
        auto frame=x.rowSlice(i*h*w,h*w);
        if(up) frame=rt_.conv2d(rt_.resize2x(frame,h,w),weight,bias,h*2,w*2,3,1,1);
        else frame=rt_.conv2d(frame,weight,bias,h,w,3,2,0,1);
        frames.push_back(frame);
    }
    x=rt_.concat(frames); h=up?h*2:h/2; w=up?w*2:w/2;
    if(!up && temporal) {
        auto found=s.features.find(key);
        auto last=rt_.concat({x.rowSlice((t-1)*h*w,h*w)});
        if(found!=s.features.end()) {
            auto joined=rt_.concat({found->second,x});
            x=rt_.conv3d(joined,convWeights_.get(key+".weight"),convWeights_.optional(key+".bias"),t+1,h,w,3,1,2,0);
            t/=2;
        }
        s.features[key]=last;
    }
    return x;
}
VAE::Tensor VAE::encodeChunk(const Tensor &input,int t,int h,int w,VAEState &s) {
    // Temporal receptive fields after the four encoder stages: 11, 21, 41,
    // 73 input frames; middle + head extend this to 113. Spatial attention and
    // normalization do not mix time. Once an entire four-frame input chunk
    // lies beyond that prefix's receptive field, its output is stationary.
    constexpr int stableAfter[]={10,20,40,72};
    bool condition=s.conditionFrames>=0 && t==4;
    auto key=[](int stage){return "condition.prefix."+std::to_string(stage);};
    if(condition) {
        auto found=s.features.find(key(4));
        if(found!=s.features.end()) return found->second;
    }
    Tensor x;
    int firstStage=0;
    if(condition) for(int stage=3;stage>=0;--stage) {
        auto found=s.features.find(key(stage));
        if(found==s.features.end()) continue;
        x=found->second; firstStage=stage+1;
        h>>=std::min(firstStage,3); w>>=std::min(firstStage,3);
        t>>=std::min(stage,2);
        break;
    }
    if(!x.rows) x=causal(input,"encoder.conv1",t,h,w,s);
    int layer=firstStage*3;
    for(int stage=firstStage;stage<4;++stage) {
        for(int block=0;block<2;++block) {
            auto p="encoder.downsamples."+std::to_string(layer++);
            x=residual(x,p,t,h,w,s);
        }
        if(stage<3) {
            auto p="encoder.downsamples."+std::to_string(layer++);
            x=resample(x,p,t,h,w,false,stage>0,s);
        }
        if(condition && s.conditionFrames>=stableAfter[stage]) {
            s.features[key(stage)]=x;
            // A stable prefix permanently replaces its upstream causal state.
            // Do not keep high-resolution history alive or copy it each move.
            const std::string down="encoder.downsamples.";
            for(auto it=s.features.begin();it!=s.features.end();) {
                const auto &name=it->first;
                bool upstream=name=="encoder.conv1";
                if(name.compare(0,down.size(),down)==0)
                    upstream=std::stoi(name.substr(down.size()))<(stage+1)*3;
                for(int earlier=0;earlier<stage;++earlier) upstream|=name==key(earlier);
                if(upstream) it=s.features.erase(it); else ++it;
            }
        }
    }
    x=middle(x,"encoder.middle",t,h,w,s);
    x=rt_.activation(rt_.channelNorm(x,weights_.get("encoder.head.0.gamma")));
    x=causal(x,"encoder.head.2",t,h,w,s);
    x=pointwise(x,"conv1");
    auto result=rt_.latentScale(rt_.columns(x,0,16),false);
    if(condition && s.conditionFrames>=112) {
        // No later condition chunk needs the causal history or larger prefix
        // tensors. The remaining immutable latent also survives world rollback.
        s.features.clear(); s.features.emplace(key(4),result);
    }
    return result;
}
VAE::Tensor VAE::decodeChunk(const Tensor &input,int h,int w,VAEState &s) {
    int t=1;
    auto x=causal(input,"decoder.conv1",t,h,w,s);
    x=middle(x,"decoder.middle",t,h,w,s);
    int layer=0;
    for(int stage=0;stage<4;++stage) {
        for(int block=0;block<3;++block) {
            auto p="decoder.upsamples."+std::to_string(layer++);
            x=residual(x,p,t,h,w,s);
        }
        if(stage<3) {
            auto p="decoder.upsamples."+std::to_string(layer++);
            x=resample(x,p,t,h,w,true,stage<2,s);
        }
    }
    x=rt_.activation(rt_.channelNorm(x,weights_.get("decoder.head.0.gamma")));
    x=causal(x,"decoder.head.2",t,h,w,s);
    return rt_.clampPixels(x);
}
VAE::Tensor VAE::encode(const Tensor &pixels,int t,int h,int w,VAEState &s) {
    require(s.conditionFrames<0,"condition-only VAE state cannot encode arbitrary video");
    bool first=s.features.empty();
    require(t>0 && h>=8 && w>=8 && h%8==0 && w%8==0 && pixels.fp32 && pixels.cols==3 &&
            pixels.rows==t*h*w && (t-(first?1:0))%4==0,"invalid Wan VAE encoder input");
    validateState(s,h,w);
    std::vector<Tensor> chunks;
    for(int start=0;start<t;) {
        int frames=first && start==0?1:4;
        chunks.push_back(encodeChunk(pixels.rowSlice(start*h*w,frames*h*w),frames,h,w,s));
        start+=frames;
    }
    return rt_.concat(chunks);
}
VAE::Tensor VAE::encodeCondition(const Tensor &image,int count,int h,int w,VAEState &s) {
    bool first=s.features.empty();
    require(count>0 && h>=8 && w>=8 && h%8==0 && w%8==0,
            "invalid Wan image condition shape");
    require(first ? image.fp32 && image.cols==3 && image.rows==h*w : s.conditionFrames>=0,
            "invalid Wan image condition state");
    validateState(s,h,w);
    std::vector<Tensor> chunks;
    if(first) {
        s.conditionFrames=0;
        chunks.push_back(encodeChunk(image,1,h,w,s));
        s.conditionFrames=1;
    }
    Tensor zeros;
    while(int(chunks.size())<count) {
        if(!zeros.rows && !s.features.count("condition.prefix.4")) {
            std::vector<float> values(size_t(4)*h*w*3,0.f);
            zeros=rt_.upload(values.data(),4*h*w,3);
        }
        chunks.push_back(encodeChunk(zeros,4,h,w,s));
        s.conditionFrames=std::min(s.conditionFrames+4,113);
    }
    return rt_.concat(chunks);
}
VAE::Tensor VAE::decode(const Tensor &latent,int t,int h,int w,VAEState &s) {
    require(t>0 && h>0 && w>0 && latent.fp32 && latent.cols==16 && latent.rows==t*h*w,
            "invalid Wan VAE decoder input");
    validateState(s,h,w);
    auto x=pointwise(rt_.latentScale(latent,true),"conv2");
    std::vector<Tensor> chunks;
    for(int i=0;i<t;++i) chunks.push_back(decodeChunk(x.rowSlice(i*h*w,h*w),h,w,s));
    return rt_.concat(chunks);
}
}} // namespace fastllm::lingbot
