#undef NDEBUG
#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-moe-weight-owner.h"
#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdio>
#include <vector>
using namespace fastllm;
static size_t freeBytes() {
    size_t f=0,t=0; assert(cudaDeviceSynchronize()==cudaSuccess);
    assert(cudaMemGetInfo(&f,&t)==cudaSuccess); return f;
}
static void init(Data &x, int gpu, int layer, int expert) {
    x.name="model.layers."+std::to_string(layer)+".mlp.experts."+std::to_string(expert)+".gateup_proj.weight";
    x.isModelWeight=true; x.cudaRepackSource=true;
    x.ToDevice(DataDevice::CUDA,{gpu},false); x.Allocate(false);
    assert(FastllmCudaIsWeightSlabPointer(x.cudaData));
    assert(!x.directMemory);
}
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 8) return 77;
    constexpr int bytes=1024*1024;
    const size_t old=FastllmCudaGetWeightSlabBytes();
    FastllmCudaSetWeightSlabBytes(0); // Repack grouping works even with normal slabs disabled.
    for(int gpu=0;gpu<8;++gpu) {
        FastllmCudaSetDevice(gpu);
        // Initialize library/CUDA state before measuring allocations.
        Data probe(INT8,{bytes}); init(probe,gpu,7,0); ReleasePackedMoeSource(probe);
        const size_t before=freeBytes();
        Data a(INT8,{bytes}),b(INT8,{bytes}),c(INT8,{bytes});
        init(a,gpu,7,0);init(b,gpu,7,1);init(c,gpu,8,0);
        const size_t allocated=freeBytes();
        assert(before-allocated>=128ULL*1024*1024);
        std::vector<unsigned char> expected(bytes,0x5a),got(bytes);
        void* packed=nullptr; assert(cudaMalloc(&packed,bytes)==cudaSuccess);
        assert(cudaMemcpy(a.cudaData,expected.data(),bytes,cudaMemcpyHostToDevice)==cudaSuccess);
        assert(cudaMemcpy(packed,a.cudaData,bytes,cudaMemcpyDeviceToDevice)==cudaSuccess);
        const size_t active=freeBytes();
        ReleasePackedMoeSource(a);const size_t first=freeBytes();
        assert(first==active); // One live source still owns the layer's slab.
        ReleasePackedMoeSource(b);const size_t last=freeBytes();
        assert(last-first==64ULL*1024*1024);
        ReleasePackedMoeSource(c);assert(freeBytes()-last==64ULL*1024*1024);
        assert(!a.cudaData&&!b.cudaData&&!c.cudaData);
        assert(cudaMemcpy(got.data(),packed,bytes,cudaMemcpyDeviceToHost)==cudaSuccess);
        assert(got==expected);assert(cudaFree(packed)==cudaSuccess);
        assert(freeBytes()==before);
        // A live ordinary weight with the same layer key must not pin a
        // repack slab. Exercise both allocation orders and preserve its bytes.
        FastllmCudaSetWeightSlabBytes(128ULL * 1024 * 1024);
        for (bool packedFirst : {false, true}) {
            const size_t start = freeBytes();
            Data persistent(INT8, {bytes}), source(INT8, {bytes});
            persistent.name = source.name = "model.layers.7.ffn.experts.0.weight";
            persistent.isModelWeight = source.isModelWeight = true;
            source.cudaRepackSource = true;
            auto allocate = [&](Data &w) {
                w.ToDevice(DataDevice::CUDA, {gpu}, false);
                w.Allocate(false);
                assert(FastllmCudaIsWeightSlabPointer(w.cudaData));
            };
            if (packedFirst) { allocate(source); allocate(persistent); }
            else { allocate(persistent); allocate(source); }
            assert(cudaMemcpy(persistent.cudaData, expected.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess);
            const size_t both = freeBytes();
            ReleasePackedMoeSource(source);
            assert(freeBytes() - both == 64ULL * 1024 * 1024);
            assert(cudaMemcpy(got.data(), persistent.cudaData, bytes, cudaMemcpyDeviceToHost) == cudaSuccess);
            assert(got == expected);
            persistent.FreeSpace();
            assert(freeBytes() == start);
        }
        FastllmCudaSetWeightSlabBytes(0);
        std::printf("SOURCE_SLAB_PASS gpu=%d isolation_orders=2\n",gpu);
    }
    FastllmCudaSetWeightSlabBytes(old);
}
