#include "fastllm-native-prefill-policy.cuh"
#include <cassert>
#include <initializer_list>
#include <cstdio>
using namespace fastllm_native_prefill;
int main() {
    for (int sm : {75, 80, 86, 89, 90, 100, 120, 121}) {
        bool expected = sm == 120;
        assert(AutomaticLinearPrefill(8, sm/10, sm%10, 1024, 16384, 5120) == expected);
        assert(AutomaticLinearPrefill(4, sm/10, sm%10, 775, 34816, 5120) == expected);
        assert(AutomaticGdnPrepare(sm/10, sm%10, 1, 48, 16, 64, 128) == expected);
        assert(!ResolvePrefillSwitch("0", expected));
        assert(ResolvePrefillSwitch("1", expected));
        assert(ResolvePrefillSwitch(nullptr, expected) == expected);
    }
    assert(!AutomaticLinearPrefill(8,12,0,1,16384,5120));
    assert(!AutomaticLinearPrefill(8,12,0,1024,8192,5120)); // TP shape
    assert(!AutomaticLinearPrefill(4,12,0,1024,17408,5120));
    assert(!AutomaticLinearPrefill(4,12,0,4097,34816,5120));
    assert(!AutomaticGdnPrepare(12,0,2,48,16,64,128));
    assert(!AutomaticGdnPrepare(12,0,1,24,16,64,128));
    assert(!AutomaticGdnPrepare(12,0,1,48,65,64,128));
    int device=0,major=0,minor=0;
    assert(cudaGetDevice(&device)==cudaSuccess);
    assert(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device)==cudaSuccess);
    assert(cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device)==cudaSuccess);
    for (int bits : {4,8}) {
        const char *name=bits==8?"FASTLLM_CUDA_NATIVE_FP8_PREFILL":"FASTLLM_CUDA_NATIVE_NVFP4_PREFILL";
        unsetenv(name);
        assert(LinearPrefillEnabled(bits,1024,34816,5120)==(major==12 && minor==0));
        assert(!LinearPrefillEnabled(bits,1024,256,256));
        setenv(name,"0",1); assert(!LinearPrefillEnabled(bits,1024,34816,5120));
        setenv(name,"1",1); assert(LinearPrefillEnabled(bits,1024,256,256));
        unsetenv(name);
    }
    puts("PASS automatic/disabled/manual policy, SM and shape boundaries");
}
