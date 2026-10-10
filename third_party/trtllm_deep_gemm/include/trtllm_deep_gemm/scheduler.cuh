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
#include <cstdint>

#include "utils.cuh"

namespace fastllm_trtllm_deep_gemm
{

enum class GemmType
{
    Normal,
    GroupedContiguous,
    GroupedMasked,
    GroupedWithOffset,
    StridedBatched
};

template <uint32_t kNumTMAMulticast, uint32_t kNumNBlocks, uint32_t kNumNBlocksPerGroup>
__device__ __forceinline__ void get_swizzled_block_idx(
    const uint32_t num_m_blocks, int block_idx, uint32_t& m_block_idx, uint32_t& n_block_idx)
{
    DG_STATIC_ASSERT(kNumNBlocksPerGroup % kNumTMAMulticast == 0, "Invalid group size");

    // Swizzle for better L2 usages
    auto num_blocks_per_group = num_m_blocks * kNumNBlocksPerGroup;
    auto group_idx = block_idx / num_blocks_per_group;
    auto first_n_block_idx = group_idx * kNumNBlocksPerGroup;
    auto num_n_blocks_in_group = min(kNumNBlocksPerGroup, kNumNBlocks - first_n_block_idx);
    auto in_group_idx = block_idx % num_blocks_per_group;
    m_block_idx = in_group_idx / num_n_blocks_in_group;
    n_block_idx = first_n_block_idx + in_group_idx % num_n_blocks_in_group;
}

struct NormalSchedulerInputSwapAB
{
    uint32_t shape_n;
    int* grouped_layout; // no use
};

template <uint32_t SHAPE_M, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kNumGroups, uint32_t kNumTMAMulticast,
    uint32_t kNumMBlocks = ceil_div(SHAPE_M, BLOCK_M), uint32_t kNumMBlocksPerGroup = 16>
struct NormalSchedulerSwapAB
{
    static constexpr GemmType gemm_type = GemmType::Normal;

    int current_iter = -1;
    uint32_t num_aligned_n_blocks;
    uint32_t num_blocks;

    using Input = NormalSchedulerInputSwapAB;
    Input input;

    NormalSchedulerSwapAB() {}

    __device__ __forceinline__ NormalSchedulerSwapAB(Input& input)
    {
        num_aligned_n_blocks = ceil_div(input.shape_n, BLOCK_N);
        num_blocks = num_aligned_n_blocks * kNumMBlocks;
    }

    // weight
    __device__ __forceinline__ uint32_t get_global_m_idx(
        const uint32_t shape_dim, const uint32_t block_size, uint32_t const& block_idx, uint32_t const& n_block_idx = 0)
    {
        return block_idx * block_size;
    }

    // act
    __device__ __forceinline__ uint32_t get_global_n_idx(uint32_t const& block_idx)
    {
        return block_idx * BLOCK_N;
    }

    // act scales
    __device__ __forceinline__ uint32_t get_global_scales_b_idx(uint32_t const& block_idx)
    {
        return block_idx;
    }

    // weight scales
    __device__ __forceinline__ uint32_t get_global_scales_a_idx(
        const uint32_t shape_dim, const uint32_t block_size, uint32_t const& block_idx, uint32_t const& n_block_idx = 0)
    {
        return block_idx * block_size;
    }

    __device__ __forceinline__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx)
    {
        ++current_iter;
        auto const next_block_idx = current_iter * gridDim.x + blockIdx.x;
        if (next_block_idx >= num_blocks)
        {
            return false;
        }

        get_swizzled_block_idx<kNumTMAMulticast, kNumMBlocks, kNumMBlocksPerGroup>(
            num_aligned_n_blocks, next_block_idx, n_block_idx, m_block_idx);
        return true;
    }
};

} // namespace fastllm_trtllm_deep_gemm
