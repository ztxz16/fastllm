//
// Created by huangyuyang on 6/2/23.
//
#pragma once

#ifndef FASTLLM_UTILS_H
#define FASTLLM_UTILS_H

#include <map>
#include <chrono>
#include <string>
#include <cstdio>
#include <cstdint>

namespace fastllm {
    static uint as_uint(const float x) {
        return *(uint32_t*)&x;
    }
    static float as_float(const uint32_t x) {
        return *(float*)&x;
    }

    static float half_to_float(const uint16_t x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
        const uint e = (x & 0x7C00) >> 10; // exponent
        const uint m = (x & 0x03FF) << 13; // mantissa
        const uint v = as_uint((float) m) >> 23; // evil log2 bit hack to count leading zeros in denormalized format
        return as_float((x & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m) | ((e == 0) & (m != 0)) * ((v - 37) << 23 |
                                                                                                         ((m << (150 - v)) &
                                                                                                          0x007FE000))); // sign : normalized : denormalized
    }
    static uint16_t float_to_half(const float x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
        const uint32_t b = as_uint(x) + 0x00001000; // round-to-nearest-even: add last bit after truncated mantissa
        const uint32_t e = (b & 0x7F800000) >> 23; // exponent
        const uint32_t m = b &
                       0x007FFFFF; // mantissa; in line below: 0x007FF000 = 0x00800000-0x00001000 = decimal indicator flag - initial rounding
        return (b & 0x80000000) >> 16 | (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13) |
               ((e < 113) & (e > 101)) * ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) |
               (e > 143) * 0x7FFF; // sign : normalized : denormalized : saturate
    }

    static double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds> (time2 - time1);
        return double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
    };

    struct TimeRecord {
        std::map<std::string, float> v;
        std::chrono::system_clock::time_point t;

        void Clear() {
            v.clear();
        }

        void Record() {
            t = std::chrono::system_clock::now();
        }

        void Record(const std::string &key) {
            auto now = std::chrono::system_clock::now();
            v[key] += GetSpan(t, now);
            t = now;
        }

        void Print() {
            float s = 0;
            for (auto &it: v) {
                printf("%s: %f s.\n", it.first.c_str(), it.second);
                s += it.second;
            }
            printf("Total: %f s.\n", s);
        }
    };
}

#endif //FASTLLM_UTILS_H
