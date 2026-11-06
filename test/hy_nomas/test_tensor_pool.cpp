#include "models/hy_nomas/runtime.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <thread>
using namespace fastllm::nomas;
int main() {
    int count=0;
    if(cudaGetDeviceCount(&count)!=cudaSuccess || !count) return 77;
    try {
    Runtime rt(0,DType::FP32);
    Tensor survivor;
    {
        TensorPool pool(0,1u<<20),nested(0,1u<<18);
        for(auto dtype:{DType::FP32,DType::BF16,DType::FP16}) {
            TensorPool::Scope scope(pool);
            std::vector<float> values(257*16,.5f);
            auto x=rt.cast(rt.upload(values.data(),257,16),dtype);
            auto snapshot=rt.add(x,x);
            Tensor output;
            // No synchronization in the loop: reuse must respect stream order,
            // and live snapshots must not be overwritten by later temporaries.
            for(int i=0;i<200;++i) {auto y=rt.add(x,x);output=rt.add(y,x);}
            if(rt.download(output)!=std::vector<float>(values.size(),1.5f) ||
               rt.download(snapshot)!=std::vector<float>(values.size(),1.f))
                throw std::runtime_error("reused storage corrupted live tensors");
            {TensorPool::Scope inner(nested);survivor=rt.add(snapshot,x);}
        }
        // Reuse a pipeline on a new caller thread, including a thread that
        // exits. Its CUDA stream identity must not be inferred from OS ids.
        for(int i=0;i<3;++i) {
            std::exception_ptr error;
            std::thread worker([&] {try {
                Runtime local(0,DType::FP32);TensorPool::Scope scope(pool);
                std::vector<float> values(257*16,2.f);
                auto x=local.upload(values.data(),257,16);
                for(int j=0;j<40;++j) survivor=local.add(x,x);
                local.synchronize();
            } catch(...) {error=std::current_exception();}});
            worker.join();if(error) std::rethrow_exception(error);
            if(rt.download(survivor)!=std::vector<float>(257*16,4.f))
                throw std::runtime_error("cross-thread pool reuse corrupted output");
        }
        rt.synchronize();pool.trim();nested.trim();
    }
    if(rt.download(survivor)!=std::vector<float>(257*16,4.f))
        throw std::runtime_error("tensor did not retain allocation pool");
#if CUDART_VERSION >= 12080
    {
        TensorPool limited(0,0,256u<<20,true);
        if(limited.reusableBytes() < (256u<<20)) throw std::runtime_error("GPU reservation is not physically backed");
        TensorPool::Scope scope(limited);
        Tensor first(12288,4096,DType::BF16),second(12288,4096,DType::BF16);
        bool rejected=false;
        try { Tensor third(12288,4096,DType::BF16); }
        catch(const std::runtime_error &e) {
            rejected=std::string(e.what()).find("GPU memory budget exhausted")!=std::string::npos;
        }
        if(!rejected) throw std::runtime_error("CUDA pool exceeded its physical memory limit");
        first={};second={};rt.synchronize();
        if(limited.reusableBytes() < (256u<<20)) throw std::runtime_error("idle reserved GPU pages were released");
        second=Tensor(12288,4096,DType::BF16);
        if(!second.data) throw std::runtime_error("bounded pool could not recover after release");
    }
    // The optional scope must not restrict unrelated allocations afterwards.
    { Tensor unrestricted(36864,4096,DType::BF16); rt.synchronize(); }
#endif
    std::cout<<"PASS bounded tensor reuse, stream order, snapshots and pool lifetime\n";
} catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}}
