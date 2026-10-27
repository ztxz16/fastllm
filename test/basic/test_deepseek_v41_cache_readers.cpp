#include "devices/cuda/fastllm-append-only-cuda-rows.h"
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
using namespace fastllm;
static void Check(bool ok, const char *s) { if (!ok) throw std::runtime_error(s); }
static void Upload(Data &source, int rows, int seed) {
    source.Resize({1,rows,source.dims[2]});
    const size_t bytes=(size_t)rows*source.dims[2]*source.unitSize;
    std::vector<unsigned char> data(bytes);
    for (size_t i=0;i<bytes;++i) data[i]=(i*37+seed*19+(i>>8))&255;
    FastllmCudaSetDevice(source.dataDeviceIds[0]);
    Check(cudaMemcpy(source.cudaData,data.data(),bytes,cudaMemcpyHostToDevice)==cudaSuccess,"upload");
}
static void Equal(Data &source, Data *copy) {
    Check(copy && copy->dims==source.dims && copy->dataType==source.dataType,"replica shape/type");
    Check(copy->strides.size()==3 && copy->strides[2]==1 &&
          copy->strides[1]==(uint64_t)source.dims[2] &&
          copy->strides[0]==copy->expansionSize,"replica capacity strides");
    Check(copy->Count(0)==copy->expansionSize &&
          copy->Count(1)==copy->expansionSize &&
          copy->Count(2)==(uint64_t)source.dims[2] &&
          copy->GetBytes()==copy->expansionBytes,"replica executor metadata");
    const size_t bytes=(size_t)source.dims[1]*source.dims[2]*source.unitSize;
    std::vector<unsigned char> expected(bytes),actual(bytes);
    FastllmCudaSetDevice(source.dataDeviceIds[0]);
    Check(cudaMemcpy(expected.data(),source.cudaData,bytes,cudaMemcpyDeviceToHost)==cudaSuccess,"read source");
    FastllmCudaSetDevice(copy->dataDeviceIds[0]);
    Check(cudaMemcpy(actual.data(),copy->cudaData,bytes,cudaMemcpyDeviceToHost)==cudaSuccess,"read replica");
    Check(expected==actual,"replica bytes differ");
}
int main() {
 try {
    int count=0; if(cudaGetDeviceCount(&count)!=cudaSuccess || count<2) return 77;
    setenv("FASTLLM_CUDA_FORCE_STAGED_COPY","1",1);
    const int consumers=std::min(count,3)-1;
    int cases=0;
    for(auto type:{INT8,BFLOAT16,FLOAT16,FLOAT32}) for(int width:{68,288,512}) {
        Data source(type,{1,0,width});source.ToDevice(CUDA,{0},false);
        source.Expansion({1,2048,width});
        CudaAppendOnlyRows replicas;
        uint64_t expectedCopied=0;
        int previous=0;
        void *readerPointer=nullptr;
        for(int rows:{131,132,132,200,256,257,513,1025}) {
            Upload(source,rows,1);
            Check(replicas.Get(source,0)==&source,"writer must remain canonical");
            for(int d=1;d<=consumers;++d) {
                const auto oldAlloc=replicas.allocations;
                Data *out=replicas.Get(source,d);
                Check(!FastllmCudaGetThreadError(),"CUDA replica error");Equal(source,out);
                const bool grew=replicas.allocations!=oldAlloc;
                expectedCopied+=(size_t)(grew?rows:rows-previous)*width*source.unitSize;
                if(d==1) {
                    if(rows==132 && previous==132)Check(out->cudaData==readerPointer,"unchanged prefix reallocated");
                    readerPointer=out->cudaData;
                }
                Check(source.dataDeviceIds==std::vector<int>{0},"writer migrated");
            }
            Check(replicas.copiedBytes==expectedCopied,"copied unused capacity or repeated prefix");
            previous=rows;
        }
        // Rollback, then overwrite an existing prefix with different bytes.
        replicas.Invalidate();Upload(source,137,2);
        for(int d=1;d<=consumers;++d)Equal(source,replicas.Get(source,d));
        // Same-size rewrite requires explicit invalidation; allocations survive.
        const auto allocations=replicas.allocations;
        replicas.Invalidate();Upload(source,137,3);
        for(int d=1;d<=consumers;++d)Equal(source,replicas.Get(source,d));
        Check(allocations==replicas.allocations,"invalidation discarded reusable buffers");
        // Shrink observed by the helper is also safe without an explicit reset.
        Upload(source,99,4);
        for(int d=1;d<=consumers;++d)Equal(source,replicas.Get(source,d));
        Check(source.dataDeviceIds==std::vector<int>{0},"source ownership changed");
        ++cases;
    }
    Data cpu(FLOAT32,{1,2,16}); CudaAppendOnlyRows unsupported;
    Check(unsupported.Get(cpu,1)==nullptr,"CPU fallback");
    std::printf("CACHE_READERS_PASS cases=%d consumers=%d staged-copy append/growth/reset/rollback bitwise\n",cases,consumers);
    return 0;
 }catch(const std::exception &e){std::fprintf(stderr,"ERROR: %s\n",e.what());return 1;}
}
