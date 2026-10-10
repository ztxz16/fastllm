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

#include <cute/arch/copy_sm90_tma.hpp>

namespace fastllm_trtllm_deep_gemm
{

template <uint32_t kNumTMAMulticast = 1>
__device__ __forceinline__ void tma_copy(
    void const* desc_ptr, uint64_t* barrier_ptr, void* smem_ptr, int32_t const& crd_0, int32_t const& crd_1)
{
    constexpr auto cache_hint = static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL);
    if constexpr (kNumTMAMulticast == 1)
    {
        cute::SM90_TMA_LOAD_2D::copy(desc_ptr, barrier_ptr, cache_hint, smem_ptr, crd_0, crd_1);
    }
    else if (cute::block_rank_in_cluster() == 0)
    {
        cute::SM90_TMA_LOAD_MULTICAST_2D::copy(
            desc_ptr, barrier_ptr, (1 << kNumTMAMulticast) - 1, cache_hint, smem_ptr, crd_0, crd_1);
    }
}

} // namespace fastllm_trtllm_deep_gemm
