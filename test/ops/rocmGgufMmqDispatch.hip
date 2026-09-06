// Build with the ROCm SDK: hipcc --offload-arch=gfx1100 -std=c++17 -O2
//   test/ops/rocmGgufMmqDispatch.hip -o rocmGgufMmqDispatch
// Run with no arguments for host policy + actual GPU configuration checks,
// or --host-only for host dispatch checks without a GPU.
#define GGML_USE_HIP
#include "../../src/devices/rocm/gguf_mmq/common.cuh"
#include <cstring>
#include <stdexcept>
#include <iostream>
namespace fastllm_rocm_mmq {
#include "../../src/devices/rocm/gguf_mmq/mmq.cuh"
#include "../../src/devices/rocm/gguf_mmq/policy.cuh"
}
using namespace fastllm_rocm_mmq;

static constexpr ggml_type types[] = {
    GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K,
    GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
    GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS
};
static void Check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
static void Hip(hipError_t error) { Check(error == hipSuccess, hipGetErrorString(error)); }
static void CheckSame(const ggml_cuda_mmq_config &a, const ggml_cuda_mmq_config &b) {
    Check(a.type == b.type && a.nthreads == b.nthreads && a.occupancy == b.occupancy &&
          a.I == b.I && a.J == b.J && a.sram_layout == b.sram_layout &&
          a.K_vram == b.K_vram && a.stream_k == b.stream_k && a.fallback == b.fallback,
          "host/device configuration mismatch");
}
static __global__ void Probe(ggml_type type, int J, bool fallback,
                             ggml_cuda_mmq_config *config, int *warp) {
    *config = ggml_cuda_mmq_get_config(type, J, fallback);
    *warp = ggml_cuda_get_physical_warp_size();
}

int main(int argc, char **argv) {
    try {
        const bool host_only = argc == 2 && std::string(argv[1]) == "--host-only";
        Check(argc == 1 || host_only, "expected no arguments or --host-only");
        const int isas[] = {0x803,0x900,0x906,0x908,0x90a,0x942,0x950,
            0x1010,0x1012,0x1030,0x1031,0x1032,0x1033,0x1034,0x1035,0x1036,
            0x1100,0x1101,0x1102,0x1103,0x1150,0x1151,0x1152,0x1153,0x1200,0x1201};
        int cases = 0;
        for (int isa : isas) {
            char arch[64]; snprintf(arch, sizeof(arch), "gfx%x", isa);
            const int warp = isa < 0x1000 ? 64 : 32;
            const int cc = mmq_device_cc(arch, warp);
            Check(cc == GGML_CUDA_CC_OFFSET_AMD + isa, "wrong ISA mapping");
            Check(mmq_device_cc(arch, warp == 32 ? 64 : 32) < 0, "mismatched wave size accepted");
            const std::string decorated = std::string(arch) + ":sramecc+:xnack-";
            Check(mmq_device_cc(decorated.c_str(), warp) == cc, "ISA feature suffix rejected");
            for (auto type : types) for (int n : {9,65,1024}) for (int k : {37,128}) {
                const auto config = mmq_select_config(type,n,k,cc,warp,65536,1024);
                Check(config.type == type, "supported architecture/type declined");
                Check(config.nthreads % warp == 0 && config.I % warp == 0, "invalid wave geometry");
                Check(mmq_get_nbytes_shared(config,cc) <= 65536, "shared memory overflow");
                const bool cdna = isa == 0x908 || isa == 0x90a || isa == 0x942 || isa == 0x950;
                Check(config.stream_k == cdna, "wrong stream-K family");
                Check(config.use_mma_data_layout(cc) == (cdna || isa >= 0x1100), "wrong MMA layout");
                if (isa < 0x1100) Check(config.J == 64, "unsupported pre-WMMA tile width");
                Check(mmq_select_config(type,n,k,cc,warp,1024,1024).type == GGML_TYPE_COUNT,
                      "insufficient shared memory accepted");
                Check(mmq_select_config(type,n,k,cc,warp,65536,64).type == GGML_TYPE_COUNT,
                      "insufficient thread limit accepted");
                ++cases;
            }
        }
        for (const char *arch : {"", "gfx", "gfx1100bad", "gfx1100oops", "gfx1300", "sm_80"})
            Check(mmq_device_cc(arch,32) < 0, "unknown architecture accepted");
        Check(mmq_device_cc(nullptr,32) < 0, "null architecture accepted");
        std::cout << "HOST PASS architectures=26 cases=" << cases << std::endl;
        if (host_only) return 0;
        int id = 0; Hip(hipGetDevice(&id));
        hipDeviceProp_t props; Hip(hipGetDeviceProperties(&props,id));
        const int cc = mmq_device_cc(props.gcnArchName,props.warpSize);
        Check(cc >= 0, "unsupported GPU for device configuration test");
        ggml_cuda_mmq_config *config;
        int *warp; Hip(hipMalloc(&config,sizeof(*config))); Hip(hipMalloc(&warp,sizeof(*warp)));
        for (auto type : types) for (int J : {64,128}) for (bool fallback : {false,true}) {
            Probe<<<1,1>>>(type,J,fallback,config,warp); Hip(hipGetLastError());
            auto got = ggml_cuda_mmq_get_config(type,J,fallback,cc);
            int got_warp; Hip(hipMemcpy(&got,config,sizeof(got),hipMemcpyDeviceToHost));
            Hip(hipMemcpy(&got_warp,warp,sizeof(got_warp),hipMemcpyDeviceToHost));
            CheckSame(got,ggml_cuda_mmq_get_config(type,J,fallback,cc));
            Check(got_warp == props.warpSize, "host/device wave size mismatch");
        }
        Hip(hipFree(config)); Hip(hipFree(warp));
        std::cout << "DEVICE PASS architecture=" << props.gcnArchName << " cases=48" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "FAIL " << e.what() << std::endl; return 1;
    }
}
