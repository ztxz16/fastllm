// Independent bit-at-a-time trellis and explicit Walsh-matrix oracle.
#include "exl3.h"
#include "utils.h"
#include <cmath>
#include <limits>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif
using namespace fastllm;
namespace {
void Require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
void Walsh(std::vector<double> &x) {
    for (size_t base = 0; base < x.size(); base += 128) {
        double result[128]{};
        for (int a=0;a<128;++a) for (int b=0;b<128;++b) {
            unsigned v=a&b, parity=0; while(v) {parity^=v&1;v>>=1;}
            result[a] += (parity ? -1 : 1) * x[base+b] / std::sqrt(128.0);
        }
        for(int a=0;a<128;++a)x[base+a]=result[a];
    }
}
std::vector<double> Oracle(const uint8_t *packed, int n, int k, int bits) {
    std::vector<double> matrix(size_t(n)*k);
    const auto *words=reinterpret_cast<const uint32_t *>(packed);
    const int ring=128*bits;
    for(int tileR=0;tileR<k/16;++tileR) for(int tileC=0;tileC<n/16;++tileC) {
        const auto *tile=words+(tileR*(n/16)+tileC)*(ring/32);
        for(int step=0;step<256;++step) {
            unsigned state=0;
            // Read the actual circular bit stream, independent of the optimized decoder's word windows.
            for(int j=0;j<16;++j) {
                int bit=((step+1)*bits/2+ring-16+j)%ring;
                state=(state<<1)|((tile[bit/32]>>(31-bit%32))&1);
            }
            uint32_t mixed=state*0x83dcd12du; unsigned sum=0;
            for(int j=0;j<4;++j)sum+=(mixed>>(j*8))&255;
            double value=(1024.0+sum)*half_to_float(0x1eee)+half_to_float(0xc931);
            int r=((step/8)%4)*2+(step%2)+8*((step/2)%2);
            int c=step/32+8*((step/4)%2);
            matrix[size_t(tileC*16+c)*k+tileR*16+r]=half_to_float(float_to_half(float(value)));
        }
    }
    return matrix;
}
void Case(int bits,int rows,int parts,bool gpu) {
    const int k=256,n=128;
    std::mt19937 rng(913+bits);
    Data weight(EXL3);weight.blockK=bits;weight.blockM=parts;weight.Resize({n*parts,k});weight.Allocate(false);
    size_t bytes=Exl3PartBytes(n,k,bits), trellis=size_t(n)*k*bits/16;
    for(int part=0;part<parts;++part) {
        auto *base=weight.cpuData+part*bytes;
        for(size_t i=0;i<trellis;i+=4){uint32_t word=rng();std::memcpy(base+i,&word,4);}
        auto *scale=reinterpret_cast<uint16_t *>(base+trellis);
        for(int i=0;i<k+n;++i)scale[i]=float_to_half((int(rng()%127)-63)/1024.0f);
    }
    Data copied;copied.CopyFrom(weight);
    Require(copied.blockK==bits && copied.blockM==parts && copied.GetBytes()==weight.GetBytes() &&
        std::memcmp(copied.cpuData,weight.cpuData,weight.GetBytes())==0,"EXL3 copy metadata/storage");
    Data input(FLOAT32,{rows,k}),bias(FLOAT32,{n*parts}),output(FLOAT32,{rows,n*parts});
    input.Allocate(false);bias.Allocate(false);
    for(int i=0;i<rows*k;++i)reinterpret_cast<float *>(input.cpuData)[i]=(int(rng()%257)-128)/64.0f;
    for(int i=0;i<n*parts;++i)reinterpret_cast<float *>(bias.cpuData)[i]=(int(rng()%257)-128)/4096.0f;
    std::vector<float> reference(size_t(rows)*n*parts);
    for(int part=0;part<parts;++part) {
        const auto *base=weight.cpuData+part*bytes;
        auto matrix=Oracle(base,n,k,bits);
        const auto *scales=reinterpret_cast<const uint16_t *>(base+trellis);
        for(int row=0;row<rows;++row) {
            std::vector<double> x(k),y(n,0.0);
            for(int i=0;i<k;++i)x[i]=reinterpret_cast<float *>(input.cpuData)[row*k+i]*half_to_float(scales[i]);
            Walsh(x);for(auto &v:x)v=half_to_float(float_to_half(float(v)));
            for(int c=0;c<n;++c)for(int i=0;i<k;++i)y[c]+=x[i]*matrix[size_t(c)*k+i];
            Walsh(y);
            for(int c=0;c<n;++c)reference[size_t(row)*n*parts+part*n+c]=float(y[c]*half_to_float(scales[k+c]))+
                reinterpret_cast<float *>(bias.cpuData)[part*n+c];
        }
    }
#ifdef USE_CUDA
    if(gpu) {
        for(auto *value:{&input,&weight,&bias})value->ToDevice(DataDevice::CUDA,std::vector<int>{0});
        output.ToDevice(DataDevice::CUDA,std::vector<int>{0},false);output.Allocate(false);
        Require(FastllmCudaExl3Linear(input,weight,bias,output),"CUDA dispatch");
        Require(cudaDeviceSynchronize()==cudaSuccess,"CUDA execution");output.ToDevice(DataDevice::CPU);
    } else
#endif
        Exl3LinearCPU(input,weight,bias,output);
    double error=0,norm=0;
    for(size_t i=0;i<reference.size();++i){double delta=reinterpret_cast<float *>(output.cpuData)[i]-reference[i];error+=delta*delta;norm+=double(reference[i])*reference[i];}
    Require(std::isfinite(error) && std::sqrt(error/std::max(norm,1e-30))<0.0005,"EXL3 numeric oracle");
}
// Compare the decode dot against the bit-at-a-time matrix oracle. Output views
// also exercise the canonical-storage mapping used before NUMA registration.
void DecodeDotCase(int bits) {
    const int n=1024,k=1024;
    std::mt19937 rng(417+bits);
    Data weight(EXL3);weight.blockK=bits;weight.blockM=2;
    weight.Resize({2*n,k});weight.Allocate(false);
    const size_t bytes=Exl3PartBytes(n,k,bits),trellis=size_t(n)*k*bits/16;
    for(int part=0;part<2;++part) {
        auto *packed=weight.cpuData+part*bytes;
        for(size_t i=0;i<trellis;i+=4){uint32_t value=rng();std::memcpy(packed+i,&value,4);}
        auto matrix=Oracle(packed,n,k,bits);
        for(int mode=0;mode<6;++mode) {
            std::vector<float> x(k);
            for(int i=0;i<k;++i) {
                float v=(int(rng()%1001)-500)/1024.f;
                if(mode==1)v=1;else if(mode==2)v=i%2?-1:1;
                else if(mode==3)v=0;else if(mode==4)v*=1e-37f;
                x[i]=v;
            }
            if(mode==5)x[17]=std::numeric_limits<float>::quiet_NaN();
            for(int view=0;view<3;++view) {
                const int nodes=view?2:1,node=view==2?1:0,first=view?16:0,count=view?96:n;
                std::vector<float> y(count+2,-12345.f);
                Exl3DotRows(weight,part,node,nodes,x.data(),y.data()+1,first,count);
                Require(y.front()==-12345.f&&y.back()==-12345.f,"Decode dot wrote past its output view");
                double error=0,norm=0;
                for(int c=0;c<count;++c) {
                    if(mode==5){Require(std::isnan(y[c+1]),"Decode dot lost NaN input");continue;}
                    double reference=0;
                    for(int i=0;i<k;++i)reference+=double(x[i])*matrix[size_t(node*(n/nodes)+first+c)*k+i];
                    Require(std::isfinite(y[c+1]),"Nonfinite decode dot result");
                    error+=double(y[c+1]-reference)*(y[c+1]-reference);norm+=reference*reference;
                }
                Require(std::sqrt(error/std::max(norm,1e-30))<.01,"Decode dot exceeds relative L2 limit");
            }
        }
    }
}

}
int main(int argc,char **argv) {
    const bool gpu=argc>1 && std::string(argv[1])=="--cuda";
#ifdef USE_CUDA
    if(gpu){int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;}
#else
    if(gpu)return 77;
#endif
    SetThreads(4);
    if(!gpu)for(int bits:{2,3,4,5,6,7,8,10,12,14,16})DecodeDotCase(bits);
    for(int bits:{2,3,4,5,6,7,8,10,12,14,16}) for(int rows:{1,2,3,4,5,7,8,9,16,17,64}) for(int parts:{1,2}) Case(bits,rows,parts,gpu);
    std::cout<<"EXL3 PASS: 242 independent numeric cases and CPU decode-dot precision/views"<<std::endl;
}
