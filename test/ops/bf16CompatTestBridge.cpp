#include "fastllm.h"
#include "device.h"
#include "devices/cpu/cpudevice.h"
#include "cudadevice.h"
#include "fastllm-cuda.cuh"
#include <cuda_runtime_api.h>

using namespace fastllm;
static DataType Dtype(int dtype) {
    return dtype == 0 ? DataType::FLOAT32 : dtype == 1 ? DataType::FLOAT16 : DataType::BFLOAT16;
}
static void Bind(Data &data, void *pointer) {
    data.isFake = true;
    data.dataDevice = DataDevice::CUDA;
    data.cudaData = pointer;
}
extern "C" void *Bf16CompatStream() { return cudaStreamPerThread; }

extern "C" bool Bf16CompatConv(int dtype, int batch, int channels, int length,
        int kernel, int stride, int pad, void *xp, void *wp, void *bp, void *yp) {
    Data x(Dtype(dtype), {batch,channels,length});
    Data w(DataType::FLOAT32, {channels,1,kernel}), b(DataType::FLOAT32);
    if (bp) b.Resize({channels});
    Data y(Dtype(dtype), {batch,channels,(length+2*pad-kernel)/stride+1});
    Bind(x,xp); Bind(w,wp); Bind(b,bp); Bind(y,yp);
    return FastllmCudaConv1DPerChannelFloat32(x,w,b,channels,channels,kernel,stride,pad,y);
}

extern "C" void Bf16CompatDecayMask(int dtype, int rows, int cols, void *xp, void *yp) {
    Data x(Dtype(dtype), {rows,cols}), y(Dtype(dtype), {rows,cols,cols});
    Bind(x,xp); Bind(y,yp);
    static CudaDevice device;
    BaseDevice *base = (BaseDevice*)&device;
    base->Reshape("MakeDecayMask", {{"input",&x},{"output",&y}}, {}, {});
    base->Run("MakeDecayMask", {{"input",&x},{"output",&y}}, {}, {});
}

extern "C" void Bf16CompatPointwise(int op, int dtype, int rows, int channels,
        void *xp, void *ap, void *yp, void *lp, void *dp, float scale) {
    Data x(Dtype(dtype), {rows,channels}), a(Dtype(dtype), {rows,channels});
    Data y(Dtype(dtype), {rows,channels}), log(DataType::FLOAT32, {channels}), dt(DataType::FLOAT32, {channels});
    Bind(x,xp); Bind(a,ap); Bind(y,yp); Bind(log,lp); Bind(dt,dp);
    static CudaDevice device;
    BaseDevice *base = (BaseDevice*)&device;
    if (op == 0) {
        base->Run("Exp", {{"input",&x},{"output",&y}}, {}, {});
    } else if (op == 1) {
        base->Run("MambaSoftplus", {{"input",&a},{"output",&y},{"aLog",&log},{"dtBias",&dt}},
                  {{"outputScale",scale}}, {});
    } else {
        base->Run("SigmoidMambaSoftplus", {{"sigmoidInputOutput",&x},{"softplusInput",&a},
                  {"softplusOutput",&y},{"aLog",&log},{"dtBias",&dt}}, {}, {});
    }
}

extern "C" bool Bf16CompatGdn(int dtype, int batch, int keyHeads, int valueHeads, int kd, int vd,
        void *qp, void *kp, void *vp, void *gp, void *bp, void *sp, void *yp, void *pointers, float scale) {
    Data q(Dtype(dtype), {batch,keyHeads,1,kd}), k(Dtype(dtype), {batch,keyHeads,1,kd});
    Data v(Dtype(dtype), {batch,valueHeads,1,vd}), g(Dtype(dtype), {batch,valueHeads,1});
    Data b(Dtype(dtype), {batch,valueHeads,1}), state(Dtype(dtype), {pointers ? 1 : batch,valueHeads,kd,vd});
    Data y(Dtype(dtype), {batch,valueHeads,1,vd});
    Bind(q,qp); Bind(k,kp); Bind(v,vp); Bind(g,gp); Bind(b,bp); Bind(state,sp); Bind(y,yp);
    if (pointers) return FastllmRecurrentGatedDeltaRuleBatchDevicePointers(
        q,k,v,g,b,state,pointers,batch,y,scale);
    FastllmRecurrentGatedDeltaRule(q,k,v,g,b,state,y,scale);
    return true;
}
