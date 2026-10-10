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

#ifndef DG_DEVICE_ASSERT
#define DG_DEVICE_ASSERT(cond)                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        if (not(cond))                                                                                                 \
        {                                                                                                              \
            printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond);                             \
            asm("trap;");                                                                                              \
        }                                                                                                              \
    } while (0)
#endif

#ifndef DG_STATIC_ASSERT
#define DG_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

namespace fastllm_trtllm_deep_gemm
{
template <typename T>
__device__ __host__ constexpr T ceil_div(T a, T b)
{
    return (a + b - 1) / b;
}

} // namespace fastllm_trtllm_deep_gemm
