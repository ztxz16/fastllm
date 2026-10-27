#include "fastllm-cuda.cuh"
#include "devices/cuda/fastllm-cuda-gdn.h"
#include <cuda_runtime_api.h>
using namespace fastllm;
static void Bind(Data &d, void *p) {
    d.isFake=true; d.dataDevice=DataDevice::CUDA; d.cudaData=p;
}
extern "C" void *GdnTestStream() { return cudaStreamPerThread; }
extern "C" int GdnTestExact(int value) {
    int old=FastllmCudaGetLinearExactBatchThreshold();
    FastllmCudaSetLinearExactBatchThreshold(value); return old;
}
extern "C" bool GdnTestRun(int fused, int hk, int hv, int variant,
        void *qp, void *kp, void *vp, void *ap, void *bp, void *wp,
        void *lp, void *dp, void *sp, void *yp, void *scratch, float eps, float scale) {
    Data q(BFLOAT16,{1,1,hk,128}), k(BFLOAT16,{1,1,hk,128}), v(BFLOAT16,{1,1,hv,128});
    Data a(BFLOAT16,{1,1,hv}), b(BFLOAT16,{1,1,hv}), w(FLOAT32,{128});
    Data l(FLOAT32,{hv}), d(FLOAT32,{hv}), state(BFLOAT16,{1,hv,128,128}), y(BFLOAT16,{1,1,hv,128});
    Bind(q,qp);Bind(k,kp);Bind(v,vp);Bind(a,ap);Bind(b,bp);Bind(w,wp);
    Bind(l,lp);Bind(d,dp);Bind(state,sp);Bind(y,yp);
    if (fused) {
        if(variant==1)state.isLinearAttentionTransposed=true;
        if(variant==2)q.dataType=FLOAT16;
        if(variant==3)q.strides.back()=2;
        if(variant==4)state.dims[0]=2;
        if(variant==5)l.dataType=BFLOAT16;
        if(variant==6)q.dataDeviceIds={999};
        if(variant==7)state.dims[3]=64;
        if(variant==8)w.cudaData=nullptr;
        return FastllmRecurrentGatedDeltaRuleNormBaBFloat16(q,k,v,a,b,w,l,d,state,y,eps,scale);
    }
    auto *p=(uint16_t *)scratch;
    Data qn(BFLOAT16,{1,hk,1,128}), kn(BFLOAT16,{1,hk,1,128});
    Data bn(BFLOAT16,{1,hv,1}), g(BFLOAT16,{1,hv,1});
    Bind(qn,p);Bind(kn,p+hk*128);Bind(bn,p+2*hk*128);Bind(g,p+2*hk*128+hv);
    cudaMemcpyAsync(bn.cudaData,bp,hv*2,cudaMemcpyDeviceToDevice,cudaStreamPerThread);
    FastllmCudaRMSNorm(q,w,qn,eps);FastllmCudaRMSNorm(k,w,kn,eps);
    FastllmCudaSigmoidMambaSoftplus(bn,a,g,l,d);
    v.Reshape({1,hv,1,128});y.Reshape({1,hv,1,128});
    FastllmRecurrentGatedDeltaRule(qn,kn,v,g,bn,state,y,scale);
    return true;
}
extern "C" bool GdnTestGate(int fused,int dtype,int rows,int cols,int variant,
        void *xp,void *wp,void *zp,void *yp,void *scratch,float eps) {
    DataType type=dtype==1?FLOAT16:BFLOAT16;
    Data x(type,{rows,cols}), w(FLOAT32,{cols}), z(type,{rows,cols}), y(type,{rows,cols});
    Bind(x,xp);Bind(w,wp);Bind(z,zp);Bind(y,yp);
    if(fused==2)return FastllmCudaRMSNormSiluMulFloat16(x,w,z,y,eps);
    if(fused){
        if(variant==1)x.strides.back()=2;
        if(variant==2)w.dataType=BFLOAT16;
        if(variant==3)y.dataType=FLOAT16;
        if(variant==4)z.dataDeviceIds={999};
        if(variant==5)w.cudaData=nullptr;
        return FastllmCudaRMSNormSiluMulBFloat16(x,w,z,y,eps);
    }
    Data silu(type,{rows,cols});Bind(silu,scratch);
    return FastllmCudaRMSNorm(x,w,y,eps) && FastllmCudaSilu(z,silu) && FastllmCudaMulTo(y,silu,1.0f);
}

extern "C" bool GraphTestSlots(int batch, int hk, int hv, int capacity, int variant,
        void *cp, void *bp, void *np, void *lp, void *dp, void *sp, void *ip, void *yp) {
    Data c(BFLOAT16,{1,batch,(hk*2+hv)*128}), b(BFLOAT16,{1,batch,hv*2});
    Data n(FLOAT32,{128}), l(FLOAT32,{hv}), d(FLOAT32,{hv});
    Data s(BFLOAT16,{capacity,hv,128,128}), i(INT32,{batch}), y(BFLOAT16,{batch,hv,1,128});
    Bind(c,cp);Bind(b,bp);Bind(n,np);Bind(l,lp);Bind(d,dp);Bind(s,sp);Bind(i,ip);Bind(y,yp);
    if(variant==1)s.dataType=FLOAT16;
    if(variant==2)s.isLinearAttentionTransposed=true;
    if(variant==3)c.strides[1]++;
    if(variant==4)i.dataType=FLOAT32;
    if(variant==5)s.dims[2]=64;
    if(variant==6)b.dataDeviceIds={999};
    if(variant==7)i.dims[0]--;
    return FastllmRecurrentGatedDeltaRuleBFloat16Slots(c,b,n,l,d,s,i,y,batch,hk,hv,1e-6,1.0f/std::sqrt(128.0f));
}
extern "C" bool GraphTestConv(int batch,int channels,int capacity,int hasBias,
        void *sp,void *ip,void *xp,void *wp,void *bp,void *yp) {
    Data s(BFLOAT16,{capacity,1,channels,4}), i(INT32,{batch}), x(BFLOAT16,{batch,channels,1});
    Data w(FLOAT32,{channels,4}), b(FLOAT32);
    if(hasBias)b.Resize({channels});
    Data y(BFLOAT16,{batch,channels,1});
    Bind(s,sp);Bind(i,ip);Bind(x,xp);Bind(w,wp);Bind(b,bp);Bind(y,yp);
    return FastllmCudaBFloat16ConvSiluSlots(s,i,x,w,b,y,batch);
}
