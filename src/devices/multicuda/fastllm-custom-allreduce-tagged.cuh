#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <memory>
#include <vector>
#include "fastllm-custom-allreduce-multicast.cuh"

namespace {

// Small-message peer-write collectives. A single aligned 64-bit store publishes
// both payload and sequence tag, eliminating separate start/end peer barriers.
// The rank-ordered FP32 sum and the owner's BF16 gather bits are unchanged.
//
// All ranks submit the same ordered calls on one stream per device. A completed
// call has consumed every rank's publication, bounding producer lead to one
// call. Three rotating frames prevent reuse while a slower peer is reading.
constexpr int kTaggedPeerMaxElements = 40960;
constexpr int kTaggedPeerBlocks = 32;
constexpr int kTaggedPeerThreads = 256;
constexpr int kTaggedPeerBuffers = 3;
constexpr uint64_t kTaggedPeerEpochPeriod = 0xffffffffULL;

__device__ __forceinline__ uint64_t TaggedPeerLoad(const uint64_t *p) {
    uint64_t x = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];"
                 : "=l"(x) : "l"(p) : "memory");
#endif
    return x;
}
struct TaggedPeerPointers {
    char *data[8]{};
    char *multicast = nullptr;
};

template <int Ranks, bool Multicast = false>
__device__ __forceinline__ void TaggedPeerStore(
        size_t offset, uint64_t x, TaggedPeerPointers peers) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    if constexpr (Multicast) {
        asm volatile("st.relaxed.sys.global.u64 [%0], %1;"
                     :: "l"(peers.multicast + offset), "l"(x) : "memory");
    } else {
#pragma unroll
        for (int peer = 0; peer < Ranks; ++peer) {
            asm volatile("st.relaxed.sys.global.u64 [%0], %1;"
                         :: "l"(peers.data[peer] + offset), "l"(x) : "memory");
        }
    }
#endif
}
template <int Ranks, bool Float32 = false, bool Multicast = false>
__global__ void FastllmCustomTaggedAllReduceKernel(
        const void *input, void *output,
        char *local, uint64_t *epochs,
        int count, int rank, size_t slotBytes, size_t frameBytes, TaggedPeerPointers peers) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __shared__ uint64_t step;
    if (threadIdx.x == 0) {
        step = epochs[blockIdx.x];
        epochs[blockIdx.x] = step + 1 == kTaggedPeerEpochPeriod ? 0 : step + 1;
    }
    __syncthreads();
    const uint32_t tag = uint32_t(step) + 1;
    // Clear unused words on the LAST use of each frame before wrap. Clearing
    // after wrap would let a reader accept an old matching tag before its
    // producer arrives. The period is divisible by the three-frame ring size.
    const bool clearUnused = tag >= kTaggedPeerEpochPeriod - 2;
    const size_t base = (step % kTaggedPeerBuffers) * frameBytes;
    const int first = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int payloadWords = Float32 ? count : count / 2;
    const int words = clearUnused ? kTaggedPeerMaxElements * 3 / 2 : payloadWords;
    for (int i = first; i < words; i += stride) {
        uint64_t word = i < payloadWords
            ? (uint64_t(reinterpret_cast<const uint32_t *>(input)[i]) << 32) | tag : 0;
        TaggedPeerStore<Ranks, Multicast>(base + rank * slotBytes + i * 8, word, peers);
    }
    for (int i = first; i < payloadWords; i += stride) {
        uint64_t values[Ranks];
        bool ready;
        do {
            ready = true;
#pragma unroll
            for (int peer = 0; peer < Ranks; ++peer) {
                values[peer] = TaggedPeerLoad(reinterpret_cast<const uint64_t *>(
                    local + base + peer * slotBytes) + i);
                ready = ready && uint32_t(values[peer]) == tag;
            }
        } while (!ready);
        if constexpr (Float32) {
            float sum = __uint_as_float(uint32_t(values[0] >> 32));
#pragma unroll
            for (int peer = 1; peer < Ranks; ++peer)
                sum += __uint_as_float(uint32_t(values[peer] >> 32));
            static_cast<float *>(output)[i] = sum;
        } else {
            float a, b;
#pragma unroll
            for (int peer = 0; peer < Ranks; ++peer) {
                uint32_t bits = uint32_t(values[peer] >> 32);
                float x = __bfloat162float(__ushort_as_bfloat16(bits & 65535u));
                float y = __bfloat162float(__ushort_as_bfloat16(bits >> 16));
                if (peer == 0) {
                    a = x;
                    b = y;
                } else {
                    a += x;
                    b += y;
                }
            }
            reinterpret_cast<uint32_t *>(output)[i] = uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(a))) |
                (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(b))) << 16);
        }
    }
#endif
}

template <int Ranks>
__global__ void FastllmCustomTaggedReduceGatherKernel(
        const float *input, float *output, __nv_bfloat16 *shared,
        char *local, uint64_t *epochs,
        int count, int rank, size_t slotBytes, size_t frameBytes, TaggedPeerPointers peers) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __shared__ uint64_t step;
    if (threadIdx.x == 0) {
        step = epochs[blockIdx.x];
        epochs[blockIdx.x] = step + 1 == kTaggedPeerEpochPeriod ? 0 : step + 1;
    }
    __syncthreads();
    const uint32_t tag = uint32_t(step) + 1;
    // Clear unused words on the LAST use of each frame before wrap. Clearing
    // after wrap would let a reader accept an old matching tag before its
    // producer arrives. The period is divisible by the three-frame ring size.
    const bool clearUnused = tag >= kTaggedPeerEpochPeriod - 2;
    const size_t base = (step % kTaggedPeerBuffers) * frameBytes;
    const int first = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    size_t dst = base + rank * slotBytes;
    const int payloadCount = count;
    const int clearCount = clearUnused ? kTaggedPeerMaxElements : payloadCount;
    for (int i = first; i < clearCount; i += stride) {
        uint64_t word = 0;
        if (i < payloadCount) {
            uint32_t bits = __float_as_uint(input[i]);
            word = (uint64_t(bits) << 32) | tag;
        }
        TaggedPeerStore<Ranks>(dst + i * 8, word, peers);
    }
    const int sharedWords = clearUnused ? kTaggedPeerMaxElements / 2 : count / 2;
    for (int i = first; i < sharedWords; i += stride) {
        bool owned = i < count / 2 && i / (count / (2 * Ranks)) == rank;
        if (owned || clearUnused) {
            uint64_t word = owned
                ? (uint64_t(reinterpret_cast<const uint32_t *>(shared)[i]) << 32) | tag
                : 0;
            TaggedPeerStore<Ranks>(dst + (kTaggedPeerMaxElements + i) * 8, word, peers);
        }
    }
    for (int i = first; i < payloadCount; i += stride) {
        uint64_t values[Ranks];
        bool ready;
        do {
            ready = true;
#pragma unroll
            for (int peer = 0; peer < Ranks; ++peer) {
                values[peer] = TaggedPeerLoad(reinterpret_cast<const uint64_t *>(
                    local + base + peer * slotBytes) + i);
                ready = ready && uint32_t(values[peer]) == tag;
            }
        } while (!ready);
        float sum = __uint_as_float(uint32_t(values[0] >> 32));
#pragma unroll
        for (int peer = 1; peer < Ranks; ++peer) {
            sum += __uint_as_float(uint32_t(values[peer] >> 32));
        }
        output[i] = sum;
    }
    for (int i = first; i < count / 2; i += stride) {
        int owner = i / (count / (2 * Ranks));
        uint64_t word;
        do {
            word = TaggedPeerLoad(reinterpret_cast<const uint64_t *>(
                local + base + owner * slotBytes) + kTaggedPeerMaxElements + i);
        } while (uint32_t(word) != tag);
        reinterpret_cast<uint32_t *>(shared)[i] = uint32_t(word >> 32);
    }
#endif
}

// Larger FP32 messages scatter each input only to its reducing owner. The
// owner adds ranks in the original FP32 order and publishes tagged results to
// every peer. Inline tags replace both global barriers; no precision is lost.
// Input publication precedes consumption of the result for the same element,
// so this is also safe when input and output alias.
template <int Ranks, bool Multicast = false>
__global__ void FastllmCustomTaggedFloatAllReduceKernel(
        const float *input, float *output, char *local, uint64_t *epochs,
        int count, int rank, size_t slotBytes, size_t frameBytes, TaggedPeerPointers peers) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __shared__ uint64_t step;
    if (threadIdx.x == 0) {
        step = epochs[blockIdx.x];
        epochs[blockIdx.x] = step + 1 == kTaggedPeerEpochPeriod ? 0 : step + 1;
    }
    __syncthreads();
    const uint32_t tag = uint32_t(step) + 1;
    const bool clearUnused = tag >= kTaggedPeerEpochPeriod - 2;
    const size_t base = (step % kTaggedPeerBuffers) * frameBytes;
    const int first = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int part = count / Ranks;
    const size_t ownSlot = base + rank * slotBytes;
    for (int i = first; i < (clearUnused ? kTaggedPeerMaxElements : count); i += stride) {
        if (i < count) {
            const int owner = i / part;
            const uint64_t word = (uint64_t(__float_as_uint(input[i])) << 32) | tag;
            asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::
                "l"(peers.data[owner] + ownSlot + i * 8), "l"(word) : "memory");
            if (clearUnused) {
#pragma unroll
                for (int peer = 0; peer < Ranks; ++peer) if (peer != owner)
                    asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::
                        "l"(peers.data[peer] + ownSlot + i * 8), "l"(uint64_t(0)) : "memory");
            }
        } else {
            TaggedPeerStore<Ranks>(ownSlot + i * 8, 0, peers);
        }
    }
    // Clear all unused input destinations and result words on the last use
    // of each frame before wrap. Message size and reducing owner may change
    // on the next call; no old matching tag may survive in an unused slot.
    if (clearUnused) {
        for (int i = first; i < kTaggedPeerMaxElements / 2; i += stride)
            if (i >= part) TaggedPeerStore<Ranks, Multicast>(ownSlot + (kTaggedPeerMaxElements + i) * 8, 0, peers);
    }
    for (int i = first; i < part; i += stride) {
        const int index = rank * part + i;
        uint64_t values[Ranks];
        bool ready;
        do {
            ready = true;
#pragma unroll
            for (int peer = 0; peer < Ranks; ++peer) {
                values[peer] = TaggedPeerLoad(reinterpret_cast<const uint64_t *>(
                    local + base + peer * slotBytes) + index);
                ready = ready && uint32_t(values[peer]) == tag;
            }
        } while (!ready);
        float sum = __uint_as_float(uint32_t(values[0] >> 32));
#pragma unroll
        for (int peer = 1; peer < Ranks; ++peer)
            sum += __uint_as_float(uint32_t(values[peer] >> 32));
        const uint64_t result = (uint64_t(__float_as_uint(sum)) << 32) | tag;
        // Every rank consumes the result word. Multicast memory turns the
        // eight peer writes into one system-scope store on Hopper.
        TaggedPeerStore<Ranks, Multicast>(ownSlot + (kTaggedPeerMaxElements + i) * 8, result, peers);
    }
    for (int i = first; i < count; i += stride) {
        const int owner = i / part, offset = i % part;
        uint64_t word;
        do {
            word = TaggedPeerLoad(reinterpret_cast<const uint64_t *>(
                local + base + owner * slotBytes) + kTaggedPeerMaxElements + offset);
        } while (uint32_t(word) != tag);
        output[i] = __uint_as_float(uint32_t(word >> 32));
    }
#endif
}

class CustomArTaggedPeer {
public:
    CustomArTaggedPeer() = default;
    CustomArTaggedPeer(const CustomArTaggedPeer &) = delete;
    CustomArTaggedPeer &operator=(const CustomArTaggedPeer &) = delete;
    ~CustomArTaggedPeer() { Release(true); }

    static std::unique_ptr<CustomArTaggedPeer> Create(const std::vector<int> &devices) {
        auto result = std::make_unique<CustomArTaggedPeer>();
        if (!result->Initialize(devices)) {
            cudaGetLastError();
            return nullptr;
        }
        return result;
    }

    bool CanUse(int count) const {
        return ready && count > 0 && count <= kTaggedPeerMaxElements && count % 64 == 0;
    }

    bool Launch(const void *input, void *output, int count, int rank, bool fp32 = false) const {
        if (!CanUse(count) || rank < 0 || rank >= 8) return false;
        if (!fp32 && multicast) {
            TaggedPeerPointers mapped = peers;
            mapped.multicast = multicast->Broadcast(rank);
            FastllmCustomTaggedAllReduceKernel<8, false, true>
                <<<kTaggedPeerBlocks, kTaggedPeerThreads, 0, cudaStreamPerThread>>>(
                    input, output, peers.data[rank], epochs[rank], count, rank, slotBytes, frameBytes, mapped);
            return true;
        }
        if (fp32 && count > 8192) {
            if (multicast) {
                TaggedPeerPointers mapped = peers;
                mapped.multicast = multicast->Broadcast(rank);
                FastllmCustomTaggedFloatAllReduceKernel<8, true>
                    <<<kTaggedPeerBlocks, kTaggedPeerThreads, 0, cudaStreamPerThread>>>(
                        static_cast<const float *>(input), static_cast<float *>(output), peers.data[rank], epochs[rank],
                        count, rank, slotBytes, frameBytes, mapped);
            } else {
                FastllmCustomTaggedFloatAllReduceKernel<8, false>
                    <<<kTaggedPeerBlocks, kTaggedPeerThreads, 0, cudaStreamPerThread>>>(
                        static_cast<const float *>(input), static_cast<float *>(output), peers.data[rank], epochs[rank],
                        count, rank, slotBytes, frameBytes, peers);
            }
            return true;
        }
        if (fp32) {
            FastllmCustomTaggedAllReduceKernel<8, true>
                <<<kTaggedPeerBlocks, kTaggedPeerThreads, 0, cudaStreamPerThread>>>(
                    input, output, peers.data[rank], epochs[rank],
                    count, rank, slotBytes, frameBytes, peers);
            return true;
        }
        FastllmCustomTaggedAllReduceKernel<8>
            <<<kTaggedPeerBlocks, kTaggedPeerThreads, 0, cudaStreamPerThread>>>(
                static_cast<const __nv_bfloat16 *>(input),
                static_cast<__nv_bfloat16 *>(output), peers.data[rank], epochs[rank],
                count, rank, slotBytes, frameBytes, peers);
        return true;
    }

    bool CanUseJoint(int count) const {
        return CanUse(count) && count <= 8192;
    }

    bool LaunchJoint(const void *input, void *output, void *shared,
                     int count, int rank) const {
        if (!CanUseJoint(count) || rank < 0 || rank >= 8) return false;
        FastllmCustomTaggedReduceGatherKernel<8>
            <<<kTaggedPeerBlocks, kTaggedPeerThreads, 0, cudaStreamPerThread>>>(
                static_cast<const float *>(input), static_cast<float *>(output),
                static_cast<__nv_bfloat16 *>(shared), peers.data[rank], epochs[rank],
                count, rank, slotBytes, frameBytes, peers);
        return true;
    }

    void ReleaseAfterDrain() { Release(false); }

private:
    bool ready = false;
    TaggedPeerPointers peers{};
    std::unique_ptr<TaggedMulticastMemory> multicast;
    std::array<uint64_t *, 8> epochs{};
    std::vector<int> devices;
    static constexpr size_t slotBytes = kTaggedPeerMaxElements * 12 + 128;
    static constexpr size_t frameBytes = slotBytes * 8;

    bool Initialize(const std::vector<int> &requested) {
        // This geometry is validated on eight Hopper GPUs with fast peer links.
        // Other topologies/architectures retain the existing custom/NCCL paths.
        if (requested.size() != 8) return false;
        for (int device : requested) {
            int major = 0, multiprocessors = 0;
            if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                       device) != cudaSuccess || major != 9) return false;
            // FP32 scatter/reduce/gather has dependencies between CTAs. Keep
            // the bounded grid resident even on devices with reduced SM counts.
            if (cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount,
                                       device) != cudaSuccess || multiprocessors < kTaggedPeerBlocks)
                return false;
            for (int peer : requested) {
                if (peer == device) continue;
                int nativeAtomics = 0;
                if (cudaDeviceGetP2PAttribute(&nativeAtomics,
                        cudaDevP2PAttrNativeAtomicSupported, device, peer) != cudaSuccess ||
                    !nativeAtomics) return false;
            }
        }
        int original = 0;
        if (cudaGetDevice(&original) != cudaSuccess) return false;
        struct RestoreDevice {
            int device;
            ~RestoreDevice() { cudaSetDevice(device); }
        } restore{original};
        devices = requested;
        multicast = TaggedMulticastMemory::Create(devices, frameBytes * kTaggedPeerBuffers);
        for (size_t rank = 0; rank < devices.size(); ++rank) {
            if (cudaSetDevice(devices[rank]) != cudaSuccess) return false;
            // A build targeting pre-Volta PTX can run on Hopper via JIT, but
            // cannot contain the system-scope load/store instructions.
            // Runtime GPU capability alone would incorrectly enable it.
            cudaFuncAttributes attributes{};
            if (cudaFuncGetAttributes(&attributes,
                    FastllmCustomTaggedAllReduceKernel<8>) != cudaSuccess ||
                attributes.ptxVersion < 70) return false;
            if (multicast) peers.data[rank] = multicast->Local(rank);
            else if (cudaMalloc(&peers.data[rank], frameBytes * kTaggedPeerBuffers) != cudaSuccess) return false;
            if (cudaMemset(peers.data[rank], 0, frameBytes * kTaggedPeerBuffers) != cudaSuccess ||
                cudaMalloc(&epochs[rank], kTaggedPeerBlocks * sizeof(uint64_t)) != cudaSuccess ||
                cudaMemset(epochs[rank], 0, kTaggedPeerBlocks * sizeof(uint64_t)) != cudaSuccess) {
                return false;
            }
        }
        // The coordinator initializes storage; persistent device workers use
        // different PTDS streams. Publish only after every zero-fill completes.
        for (int device : devices) {
            if (cudaSetDevice(device) != cudaSuccess ||
                cudaDeviceSynchronize() != cudaSuccess) return false;
        }
        ready = true;
        return true;
    }

    void Release(bool synchronize) {
        if (devices.empty()) return;
        int original = 0;
        cudaGetDevice(&original);
        if (synchronize) {
            for (int device : devices) {
                cudaSetDevice(device);
                cudaDeviceSynchronize();
            }
        }
        for (size_t rank = 0; rank < devices.size(); ++rank) {
            cudaSetDevice(devices[rank]);
            if (!multicast) cudaFree(peers.data[rank]);
            cudaFree(epochs[rank]);
            peers.data[rank] = nullptr;
            epochs[rank] = nullptr;
        }
        multicast.reset();
        devices.clear();
        ready = false;
        cudaSetDevice(original);
    }
};
}  // namespace
