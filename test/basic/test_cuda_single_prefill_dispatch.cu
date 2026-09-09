#include "flashinfer/attention/prefill.cuh"

#include <cstdio>

namespace dispatch_test {
using namespace flashinfer;
using Attention = DefaultAttention<false, false, false, false>;

template <bool Sm75, typename KV = half, uint32_t TileQ = 16, uint32_t Dim = 256>
using Traits = KernelTraits<MaskMode::kCausal, TileQ, 1, 1, Dim / 16, Dim / 16, 1, 4,
                            PosEncodingMode::kNone, half, KV, half, float, int,
                            Attention, Sm75>;
using Original = Traits<false>;
using Sm75 = Traits<true>;

// Preserve the baseline layout unless the host explicitly selects SM75.
static_assert(!Original::USE_SINGLE_PREFILL_SOFTMAX_VO_SPLIT);
static_assert(Sm75::USE_SINGLE_PREFILL_SOFTMAX_VO_SPLIT);
static_assert(sizeof(Original::SharedStorageSingle) == 66048);
static_assert(sizeof(Sm75::SharedStorageSingle) == 43536);
static_assert(std::is_same_v<Original::SharedStorageSingle, Original::SharedStorage>);
static_assert(std::is_same_v<Original::SharedStoragePaged, Sm75::SharedStoragePaged>);
static_assert(!Traits<true, half, 64>::USE_SINGLE_PREFILL_SOFTMAX_VO_SPLIT);
static_assert(!Traits<true, half, 16, 128>::USE_SINGLE_PREFILL_SOFTMAX_VO_SPLIT);
static_assert(!Traits<true, half, 16, 512>::USE_SINGLE_PREFILL_SOFTMAX_VO_SPLIT);
static_assert(!Traits<true, __nv_bfloat16>::USE_SINGLE_PREFILL_SOFTMAX_VO_SPLIT);
static_assert(!use_single_prefill_vo_split<16, 256, 256, 4, PosEncodingMode::kRoPELlama,
                                         half, half, half, Attention, true>());

#if CUDA_VERSION >= 12080
// Architecture gating must not disable or change the existing FP4 path.
using Fp4Original = Traits<false, __nv_fp4x2_e2m1>;
using Fp4Sm75 = Traits<true, __nv_fp4x2_e2m1>;
static_assert(Fp4Original::USE_SINGLE_PREFILL_SOFTMAX_VO_SPLIT);
static_assert(Fp4Sm75::USE_SINGLE_PREFILL_SOFTMAX_VO_SPLIT);
static_assert(std::is_same_v<Fp4Original::SharedStorageSingle, Fp4Sm75::SharedStorageSingle>);
#endif

// Cover all architectures and packed-query boundaries, including future SMs.
constexpr bool CheckArchitectureGate() {
    for (int major = 0; major <= 15; ++major) {
        for (int minor = 0; minor <= 9; ++minor) {
            for (int packed : {-1, 0, 1, 6, 16, 17, 64, 128}) {
                const bool expected = major == 7 && minor == 5 && packed >= 1 && packed <= 16;
                if (use_sm75_single_prefill_vo_split(major, minor, packed) != expected) return false;
            }
        }
    }
    return true;
}
static_assert(CheckArchitectureGate());

// This probe can be built for several SMs in one fat binary: host and device
// must agree on both layouts regardless of the compilation architecture.
__global__ void StorageSizes(size_t *sizes) {
    sizes[0] = sizeof(Original::SharedStorageSingle);
    sizes[1] = sizeof(Sm75::SharedStorageSingle);
}
}  // namespace dispatch_test

int main() {
    using namespace dispatch_test;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    for (int device = 0; device < count; ++device) {
        cudaDeviceProp prop{};
        size_t *sizes = nullptr;
        size_t actual[2] = {};
        if (cudaSetDevice(device) != cudaSuccess ||
            cudaGetDeviceProperties(&prop, device) != cudaSuccess ||
            cudaMalloc(&sizes, sizeof(actual)) != cudaSuccess) return 1;
        StorageSizes<<<1, 1>>>(sizes);
        const auto launch = cudaGetLastError();
        const auto copy = cudaMemcpy(actual, sizes, sizeof(actual), cudaMemcpyDeviceToHost);
        const auto released = cudaFree(sizes);
        if (launch != cudaSuccess || copy != cudaSuccess || released != cudaSuccess ||
            actual[0] != sizeof(Original::SharedStorageSingle) ||
            actual[1] != sizeof(Sm75::SharedStorageSingle)) return 1;
        const bool compact = use_sm75_single_prefill_vo_split(prop.major, prop.minor, 6);
        std::printf("device=%d sm=%d%d compact_fp16=%d original_bytes=%zu compact_bytes=%zu PASS\n",
                    device, prop.major, prop.minor, int(compact), actual[0], actual[1]);
    }
    std::puts("Architecture gate: 1,280 cases PASS; host/device storage layouts PASS");
    return 0;
}
