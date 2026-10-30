#include "models/lingbot_world/pipeline.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fastllm { namespace lingbot {
Pose identityPose() { return {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1}; }
Pose movePose(const Pose &pose,const std::string &action,float distance,float degrees) {
    if(!std::isfinite(distance) || !std::isfinite(degrees)) throw std::invalid_argument("invalid camera motion");
    auto delta=identityPose();
    double angle=degrees*3.14159265358979323846/180.;
    if(action=="forward") delta[11]=distance;
    else if(action=="backward") delta[11]=-distance;
    else if(action=="left") delta[3]=-distance;
    else if(action=="right") delta[3]=distance;
    else if(action=="up") delta[7]=-distance;
    else if(action=="down") delta[7]=distance;
    else if(action=="look_left" || action=="look_right") {
        if(action=="look_left") angle=-angle;
        delta[0]=delta[10]=float(std::cos(angle)); delta[2]=float(std::sin(angle)); delta[8]=-delta[2];
    } else if(action=="look_up" || action=="look_down") {
        if(action=="look_down") angle=-angle;
        delta[5]=delta[10]=float(std::cos(angle)); delta[6]=-float(std::sin(angle)); delta[9]=-delta[6];
    } else if(action!="stay") throw std::invalid_argument("unknown LingBot camera action: "+action);
    Pose out{};
    for(int i=0;i<4;++i) for(int j=0;j<4;++j)
        for(int k=0;k<4;++k) out[i*4+j]+=pose[i*4+k]*delta[k*4+j];
    return out;
}
std::vector<Pose> relativePoses(const std::vector<Pose> &poses,const Pose *previous) {
    if(poses.empty()) throw std::invalid_argument("empty camera trajectory");
    auto multiply=[](const Pose &a,const Pose &b) {
        Pose result{};
        for(int i=0;i<4;++i) for(int j=0;j<4;++j)
            for(int k=0;k<4;++k) result[i*4+j]+=a[i*4+k]*b[k*4+j];
        return result;
    };
    auto inverse=[](const Pose &pose) {
        auto result=identityPose();
        for(int i=0;i<3;++i) {
            float translation=0;
            for(int j=0;j<3;++j) {
                result[i*4+j]=pose[j*4+i];
                translation+=pose[j*4+i]*pose[j*4+3];
            }
            result[i*4+3]=-translation;
        }
        return result;
    };
    // Preserve the official two-stage transform: first relative to the
    // trajectory origin, then relative to the preceding frame. Simplifying
    // it algebraically changes FP32 rounding before camera features use BF16.
    std::vector<Pose> absolute;
    if(previous) absolute.push_back(*previous);
    absolute.insert(absolute.end(),poses.begin(),poses.end());
    auto origin=inverse(absolute.front());
    for(auto &pose:absolute) pose=multiply(origin,pose);
    absolute.front()=identityPose();
    std::vector<Pose> relative(absolute.size(),identityPose());
    for(size_t n=1;n<absolute.size();++n) relative[n]=multiply(inverse(absolute[n-1]),absolute[n]);
    float maxNorm=0;
    for(const auto &p:relative)
        maxNorm=std::max(maxNorm,std::sqrt(p[3]*p[3]+p[7]*p[7]+p[11]*p[11]));
    if(maxNorm>0) for(auto &p:relative) for(int i=0;i<3;++i) p[i*4+3]/=maxNorm;
    if(previous) relative.erase(relative.begin());
    return relative;
}
std::vector<SamplingStep> fastSchedule(float shift,const std::vector<int> &indices) {
    if(!std::isfinite(shift) || shift<=0 || indices.empty()) throw std::invalid_argument("invalid LingBot schedule");
    std::vector<int> timesteps(1000);
    std::vector<float> sigmas(1000);
    // Preserve the official float32 sigma_max, float64 linspace/shift and
    // integer timestep truncation, including duplicate-timestep lookup rules.
    double maximum=double(float(.999));
    for(int i=0;i<1000;++i) {
        double s=maximum*(1.-double(i)/1000.);
        s=double(shift)*s/(1.+(double(shift)-1.)*s);
        timesteps[i]=int(s*1000.); sigmas[i]=float(s);
    }
    std::vector<SamplingStep> out;
    int previous=-1;
    for(int index:indices) {
        if(index<0 || index>=1000 || index<=previous) throw std::invalid_argument("LingBot timestep indices must increase within 0..999");
        int t=timesteps[index],first=int(std::find(timesteps.begin(),timesteps.end(),t)-timesteps.begin());
        int noiseIndex=first+1<1000 && timesteps[first+1]==t?first+1:first;
        out.push_back({t,sigmas[first],sigmas[noiseIndex]}); previous=index;
    }
    return out;
}
}} // namespace fastllm::lingbot
