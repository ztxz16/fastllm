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
#include <vector>

#if defined(_WIN32) or defined(_WIN64)
#include <Windows.h>
#else
#include <unistd.h>
#endif

#ifdef __AVX__
#include "immintrin.h"
#ifdef __GNUC__
#if __GNUC__ < 8
#define _mm256_set_m128i(/* __m128i */ hi, /* __m128i */ lo) \
    _mm256_insertf128_si256(_mm256_castsi128_si256(lo), (hi), 0x1)
#endif
#endif
#endif

namespace fastllm {
    static void MySleep(int t) {
#if defined(_WIN32) or defined(_WIN64)
        Sleep(t);
#else
        sleep(t);
#endif
    }

    static void ErrorInFastLLM(const std::string &error) {
        printf("FastLLM Error: %s\n", error.c_str());
        throw error;
    }

    static void AssertInFastLLM(bool condition, const std::string &error) {
        if (!condition) {
            ErrorInFastLLM(error);
        }
    }

    static uint32_t as_uint(const float x) {
        return *(uint32_t*)&x;
    }
    static float as_float(const uint32_t x) {
        return *(float*)&x;
    }

    static float half_to_float(const uint16_t x) { // IEEE-754 16-bit floating-point format (without infinity): 1-5-10, exp-15, +-131008.0, +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
        const uint32_t e = (x & 0x7C00) >> 10; // exponent
        const uint32_t m = (x & 0x03FF) << 13; // mantissa
        const uint32_t v = as_uint((float) m) >> 23; // evil log2 bit hack to count leading zeros in denormalized format
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

    static bool StartWith(const std::string &a, const std::string &b) {
        return a.size() >= b.size() && a.substr(0, b.size()) == b;
    }

    template <typename T>
    static std::vector <T> AppendVector(const std::vector <T> &a, const std::vector <T> &b) {
        std::vector <T> ret = a;
        for (int i = 0; i < b.size(); i++) {
            ret.push_back(b[i]);
        }
        return ret;
    }

    static std::vector <int> ParseDeviceIds(const std::string &s, const std::string &type) {
        int i = type.size();
        std::vector <int> ret;
        std::string cur = "";
        if (s.size() > i && s[i] == ':') {
            i++;
            while (i < s.size()) {
                if (s[i] >= '0' && s[i] <= '9') {
                    cur += s[i];
                } else {
                    if (cur != "") {
                        ret.push_back(atoi(cur.c_str()));
                        cur = "";
                    }
                }
                i++;
            }
        }
        if (cur != "") {
            ret.push_back(atoi(cur.c_str()));
        }
        return ret;
    }

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

#ifdef __AVX__
    static inline float Floatsum(const __m256 a) {
        __m128 res = _mm256_extractf128_ps(a, 1);
        res = _mm_add_ps(res, _mm256_castps256_ps128(a));
        res = _mm_add_ps(res, _mm_movehl_ps(res, res));
        res = _mm_add_ss(res, _mm_movehdup_ps(res));
        return _mm_cvtss_f32(res);
    }

    static inline int I32sum(const __m256i a) {
        const __m128i sum128 = _mm_add_epi32(_mm256_extractf128_si256(a, 0), _mm256_extractf128_si256(a, 1));
        const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
        const __m128i sum64 = _mm_add_epi32(hi64, sum128);
        const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
        return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
    }

    static inline int I16Sum(const __m256i a) {
        int sum = I32sum(_mm256_madd_epi16(a, _mm256_set1_epi16(1)));
        return sum;
    }
#endif
}

#endif //FASTLLM_UTILS_H
