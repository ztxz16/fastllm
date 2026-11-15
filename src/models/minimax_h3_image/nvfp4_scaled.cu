// Separate translation units avoid nvcc's dependent block-scaled Params stub bug.
#define FASTLLM_NVFP4_OUTPUT cutlass::bfloat16_t
#define FASTLLM_NVFP4_ENTRY nvfp4GemmBf16
#include "nvfp4_gemm.cuh"
#undef FASTLLM_NVFP4_OUTPUT
#undef FASTLLM_NVFP4_ENTRY
namespace fastllm::minimax_h3_image::detail {
void nvfp4GemmFloat(void *,const void *,const void *,const void *,const void *,const float *,int,int,int,cudaStream_t);
void nvfp4Gemm(void *out,const void *x,const void *w,const void *xs,const void *ws,const float *alpha,
               int m,int n,int k,cudaStream_t stream,bool fp32Output) {
    if(fp32Output) nvfp4GemmFloat(out,x,w,xs,ws,alpha,m,n,k,stream);
    else nvfp4GemmBf16(out,x,w,xs,ws,alpha,m,n,k,stream);
}
}
