#pragma once

#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define FASTLLM_MATH_HOST_DEVICE __host__ __device__
#else
#define FASTLLM_MATH_HOST_DEVICE
#endif

namespace fastllm {

/**
 * 向上取到最近的2的幂，保留零值。
 * @param num 非负整数，不超过2^31，保证结果可由uint32_t表示。
 * @return 不小于num的最小2的幂；num为0时返回0。
 */
FASTLLM_MATH_HOST_DEVICE inline constexpr std::uint32_t next_pow_2(std::uint32_t num) {
    if (num <= 1) {
        return num;
    }
    --num;
    num |= num >> 1;
    num |= num >> 2;
    num |= num >> 4;
    num |= num >> 8;
    num |= num >> 16;
    return num + 1;
}

/**
 * 计算整数除法的向上取整值，调用方保证中间运算不溢出。
 * @param a 非负整数。
 * @param b 正整数。
 * @return a除以b的向上取整值。
 */
template <typename A, typename B>
FASTLLM_MATH_HOST_DEVICE inline constexpr auto div_ceil(A a, B b) {
    static_assert(std::is_integral<A>::value && std::is_integral<B>::value,
                  "div_ceil requires integer arguments");
    return (a + b - 1) / b;
}

/**
 * 将非负整数向下对齐到指定粒度。
 * @param a 非负整数。
 * @param b 正整数对齐粒度。
 * @return 不大于a的最大b的倍数。
 */
template <typename T>
FASTLLM_MATH_HOST_DEVICE inline constexpr T round_to_previous_multiple_of(T a, T b) {
    static_assert(std::is_integral<T>::value, "alignment requires integer arguments");
    return a % b == 0 ? a : (a / b) * b;
}

/**
 * 将非负整数向上对齐到指定粒度，调用方保证结果和中间运算不溢出。
 * @param a 非负整数。
 * @param b 正整数对齐粒度。
 * @return 不小于a的最小b的倍数。
 */
template <typename T>
FASTLLM_MATH_HOST_DEVICE inline constexpr T round_to_next_multiple_of(T a, T b) {
    static_assert(std::is_integral<T>::value, "alignment requires integer arguments");
    return a % b == 0 ? a : ((a / b) + 1) * b;
}

}  // namespace fastllm

#undef FASTLLM_MATH_HOST_DEVICE
