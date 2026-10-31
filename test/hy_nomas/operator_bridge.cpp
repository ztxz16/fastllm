// Test-only C ABI. Link against fastllm_nomas; never included in production.
#include "models/hy_nomas/runtime.h"
#include <algorithm>
#include <stdexcept>
#include <string>

using namespace fastllm::nomas;
namespace { thread_local std::string error; }
extern "C" {
const char *nomas_test_error() { return error.c_str(); }
void *nomas_test_create_precision(int fp32) {
    try { return new Runtime(0,fp32!=0); }
    catch(const std::exception &e) { error=e.what(); return nullptr; }
}
void *nomas_test_create() { return nomas_test_create_precision(0); }
void nomas_test_destroy(void *ctx) { delete static_cast<Runtime *>(ctx); }
int nomas_test_pixel_noise(void *ctx,uint64_t seed,int firstFrame,int frames,
                           int height,int width,float *output) {
    try {
        auto &rt=*static_cast<Runtime *>(ctx);
        auto values=rt.download(rt.pixelNoise(seed,firstFrame,frames,height,width));
        std::copy(values.begin(),values.end(),output);
        return 0;
    } catch(const std::exception &e) { error=e.what(); return -1; }
}
int nomas_test_op(void *ctx, const char *name, const int *shape,
                  const float *x, const float *y, const float *z, float *output) {
    try {
        auto &rt=*static_cast<Runtime *>(ctx);
        const int rows=shape[0], cols=shape[1];
        auto a=rt.upload(x,rows,cols); Tensor out;
        const std::string op(name);
        if(op=="linear") {
            auto w=rt.upload(y,shape[2],cols); Tensor bias;
            if(z) bias=rt.upload(z,1,shape[2]);
            out=rt.linear(a,w,z?&bias:nullptr);
        } else if(op=="rms" || op=="layernorm") {
            auto weight=rt.upload(y,1,cols); Tensor bias;
            if(z) bias=rt.upload(z,1,cols);
            out=rt.norm(a,op=="rms",&weight,z?&bias:nullptr);
        } else if(op=="silu" || op=="gelu") {
            out=rt.activation(a,op=="gelu");
        } else if(op=="swiglu") {
            out=rt.siluMultiply(a,rt.upload(y,rows,cols));
        } else if(op=="modulate") {
            out=rt.modulate(a,rt.upload(y,1,2*cols),0,1);
        } else if(op=="residual") {
            out=rt.residual(a,rt.upload(y,rows,cols),rt.upload(z,1,cols),0);
        } else if(op=="rope") {
            out=rt.rope(a,y,z,128);
        } else if(op=="pixel_qk") {
            out=rt.pixelQKRope(a,rt.upload(y,1,64),rt.upload(z,1,64),shape[2],shape[3],shape[4]);
        } else if(op=="attention") {
            const int keys=shape[2], heads=shape[3], kvHeads=shape[4];
            out=rt.attention(a,rt.upload(y,keys,kvHeads*128),rt.upload(z,keys,kvHeads*128),
                             heads,nullptr,128,kvHeads,shape[5],shape[6],shape[7],shape[8]);
        } else throw std::invalid_argument("unknown test operator");
        auto values=rt.download(out); std::copy(values.begin(),values.end(),output);
        return 0;
    } catch(const std::exception &e) { error=e.what(); return -1; }
}
}
