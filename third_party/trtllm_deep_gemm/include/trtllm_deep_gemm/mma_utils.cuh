/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 DeepSeek
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License.
 * You may obtain a copy of the License at
 *
 * https://opensource.org/licenses/MIT
 *
 *
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda.h>

#include "utils.cuh"

namespace fastllm_trtllm_deep_gemm
{

struct SM90_64x16x32_F32E4M3E4M3_SS
{
    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float& d00, float& d01, float& d02,
        float& d03, float& d04, float& d05, float& d06, float& d07, bool scale_d)
    {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, %10, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3"
            "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7},"
            " %8,"
            " %9,"
            " p   , 1,    1;\n"
            "}\n"
            : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07)
            : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_d)));
    }

    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d)
    {
        wgmma(desc_a, desc_b, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], scale_d);
    }

    static constexpr int M = 64;
    static constexpr int N = 16;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
};

struct SM90_64x24x32_F32E4M3E4M3_SS
{
    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float& d00, float& d01, float& d02,
        float& d03, float& d04, float& d05, float& d06, float& d07, float& d08, float& d09, float& d10, float& d11,
        bool scale_d)
    {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, %14, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n24k32.f32.e4m3.e4m3"
            "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7, "
            " %8,   %9,   %10,  %11},"
            " %12,"
            " %13,"
            " p   , 1,    1;\n"
            "}\n"
            : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07), "+f"(d08),
            "+f"(d09), "+f"(d10), "+f"(d11)
            : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_d)));
    }

    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d)
    {
        wgmma(desc_a, desc_b, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], scale_d);
    }

    static constexpr int M = 64;
    static constexpr int N = 24;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
};

struct SM90_64x32x32_F32E4M3E4M3_SS
{
    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float& d00, float& d01, float& d02,
        float& d03, float& d04, float& d05, float& d06, float& d07, float& d08, float& d09, float& d10, float& d11,
        float& d12, float& d13, float& d14, float& d15, bool scale_d)
    {
        asm volatile(
            "{\n"
            ".reg .pred p;\n"
            "setp.ne.b32 p, %18, 0;\n"
            "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3"
            "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7, "
            " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15},"
            " %16,"
            " %17,"
            " p   , 1,    1;\n"
            "}\n"
            : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07), "+f"(d08),
            "+f"(d09), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15)
            : "l"(desc_a), "l"(desc_b), "r"(int32_t(scale_d)));
    }

    __device__ static void wgmma(uint64_t const& desc_a, uint64_t const& desc_b, float* d, bool scale_d)
    {
        wgmma(desc_a, desc_b, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], d[12], d[13],
            d[14], d[15], scale_d);
    }

    static constexpr int M = 64;
    static constexpr int N = 32;
    static constexpr int K = 32;
    static constexpr int kNumAccum = M * N / 128;
};

template <typename dtype_t>
struct SM90_U32x2_STSM_N
{
    __device__ __forceinline__ static void copy(dtype_t src_0, dtype_t src_1, void* smem_dst)
    {
        uint32_t const src[2] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1)};
        asm volatile(
            "stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n" ::"l"(smem_dst), "r"(src[0]), "r"(src[1]));
    }
};

template <typename dtype_t>
struct SM90_U32x4_STSM_N
{
    __device__ __forceinline__ static void copy(
        dtype_t src_0, dtype_t src_1, dtype_t src_2, dtype_t src_3, void* smem_dst)
    {
        uint32_t const src[4] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1),
            *reinterpret_cast<uint32_t*>(&src_2), *reinterpret_cast<uint32_t*>(&src_3)};
        asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"l"(smem_dst), "r"(src[0]),
            "r"(src[1]), "r"(src[2]), "r"(src[3]));
    }
};

template <typename dtype_t>
struct SM90_U32x2_STSM_T
{
    __device__ __forceinline__ static void copy(dtype_t src_0, dtype_t src_1, void* smem_dst)
    {
        const uint32_t src[2] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1)};
        asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16.trans [%0], {%1, %2};\n" ::"l"(smem_dst), "r"(src[0]),
            "r"(src[1]));
    }
};

template <typename dtype_t>
struct SM90_U32x4_STSM_T
{
    __device__ __forceinline__ static void copy(
        dtype_t src_0, dtype_t src_1, dtype_t src_2, dtype_t src_3, void* smem_dst)
    {
        const uint32_t src[4] = {*reinterpret_cast<uint32_t*>(&src_0), *reinterpret_cast<uint32_t*>(&src_1),
            *reinterpret_cast<uint32_t*>(&src_2), *reinterpret_cast<uint32_t*>(&src_3)};
        asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16.trans [%0], {%1, %2, %3, %4};\n" ::"l"(smem_dst),
            "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]));
    }
};

__device__ void warpgroup_arrive()
{
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch()
{
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_fence_operand(float& reg)
{
    asm volatile("" : "+f"(reg)::"memory");
}

__forceinline__ __device__ uint32_t get_lane_id()
{
    uint32_t lane_id;
    asm("mov.u32 %0, %laneid;" : "=r"(lane_id));
    return lane_id;
}

__device__ __forceinline__ uint32_t ld_shared(uint32_t const* __restrict__ ptr)
{
    uint32_t ret;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ int4 ld_shared(int4 const* __restrict__ ptr)
{
    int4 ret;
    asm volatile("ld.shared.v4.s32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
                 : "l"(ptr));
    return ret;
}

__device__ __forceinline__ float ld_shared(float const* __restrict__ ptr)
{
    float ret;
    asm volatile("ld.shared.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ float2 ld_shared(float2 const* __restrict__ ptr)
{
    float2 ret;
    asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(ret.x), "=f"(ret.y) : "l"(ptr));
    return ret;
}

__device__ __forceinline__ void st_shared(float const* ptr, float val)
{
    asm volatile("st.shared.f32 [%0], %1;" ::"l"(ptr), "f"(val));
}

__device__ __forceinline__ void st_shared(uint32_t const* ptr, uint32_t val)
{
    asm volatile("st.shared.u32 [%0], %1;" ::"l"(ptr), "r"(val));
}

template <int N>
__device__ void warpgroup_wait()
{
    DG_STATIC_ASSERT(N >= 0 and N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

union GmmaDescriptor
{
    __host__ __device__ constexpr GmmaDescriptor() noexcept
        : desc_(0)
    {
    }

    __host__ __device__ constexpr GmmaDescriptor(uint64_t desc) noexcept
        : desc_(desc)
    {
    }

    __host__ __device__ constexpr GmmaDescriptor(GmmaDescriptor const& t) noexcept
        : desc_(t.desc_)
    {
    }

    __host__ __device__ constexpr GmmaDescriptor(GmmaDescriptor&& t) noexcept
        : desc_(t.desc_)
    {
    }

    __host__ __device__ constexpr GmmaDescriptor& operator=(GmmaDescriptor const& t) noexcept
    {
        desc_ = t.desc_;
        return *this;
    }

    __host__ __device__ constexpr GmmaDescriptor& operator=(GmmaDescriptor&& t) noexcept
    {
        desc_ = t.desc_;
        return *this;
    }

    uint64_t desc_;
    uint32_t reg32_[2];
    uint16_t reg16_[4];

    struct
    {
        uint16_t start_address_ : 14, : 2;
        uint16_t leading_byte_offset_ : 14, : 2;
        uint16_t stride_byte_offset_ : 14, : 2;
        uint8_t : 1, base_offset_ : 3, : 4;
        uint8_t : 6, layout_type_ : 2;
    } bitfield;

    // Decay to an `uint64_t`
    __host__ __device__ constexpr operator uint64_t() const noexcept
    {
        return desc_;
    }
};

template <class PointerType>
__device__ GmmaDescriptor make_smem_desc(
    PointerType smem_ptr, int layout_type, int leading_byte_offset = 0, int stride_byte_offset = 1024)
{
    GmmaDescriptor desc;
    auto uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    desc.bitfield.start_address_ = uint_ptr >> 4;
    desc.bitfield.layout_type_ = layout_type;
    desc.bitfield.leading_byte_offset_ = leading_byte_offset >> 4;
    desc.bitfield.stride_byte_offset_ = stride_byte_offset >> 4;
    desc.bitfield.base_offset_ = 0;
    return desc;
}

template <int N>
struct FP8MMASelector
{
    static constexpr auto select_type()
    {
        if constexpr (N == 16)
            return SM90_64x16x32_F32E4M3E4M3_SS();
        if constexpr (N == 24)
            return SM90_64x24x32_F32E4M3E4M3_SS();
        if constexpr (N == 32)
            return SM90_64x32x32_F32E4M3E4M3_SS();
    }

    using type = decltype(select_type());
};

} // namespace fastllm_trtllm_deep_gemm
