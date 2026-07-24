//
// Created by huangyuyang on 6/2/23.
//
#pragma once

#ifndef FASTLLM_UTILS_H
#define FASTLLM_UTILS_H

#include <cmath>
#include <algorithm>
#include <map>
#include <chrono>
#include <string>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <thread>
#include <mutex>
#include <vector>
#include <deque>
#include <array>
#ifndef __CUDACC__
#if defined(__GNUC__) && __GNUC__ < 8 && !defined(__clang__)
#include <experimental/filesystem>
#else
#include <filesystem>
#endif
#endif

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
#define _mm256_set_m128(/* __m128 */ hi, /* __m128 */ lo) \
    (__m256) _mm256_insertf128_ps(_mm256_castps128_ps256(lo), (hi), 0x1)
#endif
#endif
#endif

#ifndef __CUDACC__
#if (defined(_MSC_VER) && _MSC_VER <= 1900) || (defined(__GNUC__) && __GNUC__ < 8 && !defined(__clang__))  // VS 2015) 
    namespace fs = std::experimental::filesystem;
#else
    namespace fs = std::filesystem;
#endif
#endif

#ifndef __aarch64__
// Intrinsics for CPUID
#if defined(_MSC_VER)
    #include <intrin.h> // For __cpuid, __cpuidex, _xgetbv
#elif defined(__GNUC__) || defined(__clang__)
    #include <cpuid.h> // For __get_cpuid, __get_cpuid_count
    #include <x86intrin.h> // For _xgetbv (usually included by cpuid.h or available)
    // GCC/Clang might not have _xgetbv as an intrinsic like MSVC,
    // or it might be in a different header.
    // If _xgetbv is not found, you might need to implement it with inline assembly.
    #ifndef _XCR_XFEATURE_ENABLED_MASK // Often defined with _xgetbv
    #define _XCR_XFEATURE_ENABLED_MASK 0
    #if __GNUC__ < 8 and !defined(USE_ROCM)
    static uint64_t _xgetbv(uint32_t xcr_index) {
        uint32_t eax, edx;
        __asm__ __volatile__ (
            "xgetbv"
            : "=a" (eax), "=d" (edx)  // Output operands: eax, edx
            : "c" (xcr_index)         // Input operand: ecx (xcr_index)
            :                         // Clobbered registers (none explicitly clobbered by xgetbv beyond outputs)
        );
        return ((uint64_t)edx << 32) | eax;
    }
    #endif
    #endif
#else
    #warning "CPUID detection not implemented for this compiler."
#endif
#endif // ifndef __aarch64__

namespace fastllm {
    static bool StringEndWith(const std::string &s, const std::string &end) {
        return s.size() >= end.size() && s.substr(s.size() - end.size()) == end;
    }

    static bool StringStartWith(const std::string &s, const std::string &end) {
        return s.size() >= end.size() && s.substr(0, end.size()) == end;
    }

    static void MySleep(int t) {
        std::this_thread::sleep_for(std::chrono::seconds(t));
    }

    static void ErrorInFastLLM(const std::string &error) {
        printf("FastLLM Error: %s\n", error.c_str());
        printf("Press any key to exit...\n");
        getchar();
        exit(0);
    }

    static void AssertInFastLLM(bool condition, const std::string &error) {
        if (!condition) {
            ErrorInFastLLM(error);
        }
    }

    static float gelu(float x) {
        return x * 0.5f * (1.0f + erf(x / sqrt(2.0)));
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

    struct CPUInstructInfo {
        bool hasAVX2 = false;
        bool hasAVX512F = false;
        bool hasAVX512BF16 = false;
        bool hasAVX512VNNI = false;
        bool hasAMX = false; // New flag for AMX

        CPUInstructInfo() {
#ifndef __aarch64__
#if defined(_MSC_VER) || defined(__GNUC__) || defined(__clang__)
            std::array<int, 4> regs; // For EAX, EBX, ECX, EDX
            
            // Step 1: Check OSXSAVE bit (CPUID EAX=1, ECX bit 27)
            bool os_supports_xsave = false;
            #if defined(_MSC_VER)
            __cpuid(regs.data(), 1);
            #else 
            __get_cpuid(1, (unsigned int*)&regs[0], (unsigned int*)&regs[1], (unsigned int*)&regs[2], (unsigned int*)&regs[3]);
            #endif
            if (regs[2] & (1 << 27)) {
                os_supports_xsave = true;
            }
            
            bool os_avx_enabled = false;
            bool os_avx512_enabled = false;
            bool os_amx_enabled = false; // OS state support for AMX

            if (os_supports_xsave) {
                uint64_t xcr0 = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
                
                // Check for AVX support (bits 1 and 2)
                if ((xcr0 & 0x6) == 0x6) {
                    os_avx_enabled = true;
                    
                    // Check for AVX512 support (bits 1,2,5,6,7)
                    // OPMASK(5), ZMM_Hi256(6), Hi16_ZMM(7)
                    if ((xcr0 & 0xE6) == 0xE6) {
                        os_avx512_enabled = true;
                    }

                    // Check for AMX support (bits 17 and 18)
                    // Bit 17: XTILECFG state
                    // Bit 18: XTILEDATA state
                    // Both must be 1 for AMX to be usable
                    if ((xcr0 & ((1 << 17) | (1 << 18))) == ((1 << 17) | (1 << 18))) {
                        os_amx_enabled = true;
                    }
                }
            }
            
            if (os_avx_enabled) {
                // CPUID with EAX=7, ECX=0 for extended features
                #if defined(_MSC_VER)
                __cpuidex(regs.data(), 7, 0);
                #else 
                __get_cpuid_count(7, 0, (unsigned int*)&regs[0], (unsigned int*)&regs[1], (unsigned int*)&regs[2], (unsigned int*)&regs[3]);
                #endif
                
                // AVX2: EAX=7, ECX=0, EBX bit 5
                hasAVX2 = (regs[1] & (1 << 5)) != 0;
                
                // Only check AVX512 features if OS supports AVX512 states
                if (os_avx512_enabled) {
                    // AVX512F: EAX=7, ECX=0, EBX bit 16
                    hasAVX512F = (regs[1] & (1 << 16)) != 0;
                    
                    // AVX512VNNI: EAX=7, ECX=0, ECX bit 11
                    hasAVX512VNNI = (regs[2] & (1 << 11)) != 0;
                    
                    // AMX-TILE is the base requirement (EAX=7, ECX=0, EDX bit 24)
                    if(os_amx_enabled) {
                        bool hasAMX_TILE = (regs[3] & (1 << 24)) != 0;
                        // Current AMX kernels require AMX-BF16 (EDX bit 22).
                        bool hasAMX_BF16 = (regs[3] & (1 << 22)) != 0;

                        hasAMX = hasAMX_TILE && hasAMX_BF16;
                    }

                    // AVX512_BF16: EAX=7, ECX=1, EAX bit 5
                    #if defined(_MSC_VER)
                    __cpuidex(regs.data(), 7, 1);
                    #else 
                    __get_cpuid_count(7, 1, (unsigned int*)&regs[0], (unsigned int*)&regs[1], (unsigned int*)&regs[2], (unsigned int*)&regs[3]);
                    #endif
                    hasAVX512BF16 = (regs[0] & (1 << 5)) != 0;
                    
                    // Dependencies
                    hasAVX512BF16 = hasAVX512BF16 && hasAVX512F;
                    hasAVX512VNNI = hasAVX512VNNI && hasAVX512F;
                    // Current AMX kernels use BF16 tile compute and live in an AVX512BF16 target file.
                    hasAMX = hasAMX && hasAVX512BF16;
                }
            }
#endif             
            std::string x[2] = {"OFF", "ON"};
            printf("CPU Instruction Info: ");
            printf("[AVX2: %s] ", x[hasAVX2].c_str());
            printf("[AVX512F: %s] ", x[hasAVX512F].c_str());
            printf("[AVX512_VNNI: %s] ", x[hasAVX512VNNI].c_str());
            printf("[AVX512_BF16: %s] ", x[hasAVX512BF16].c_str());
            printf("[AMX: %s] ", x[hasAMX].c_str()); // Print AMX status
            printf("\n");
#endif 
        }
    };

    // static CPUInstructInfo cpuInstructInfo;

    struct FP16ToFP32Manager {
        float dict[65536];

        FP16ToFP32Manager() {
            for (int i = 0; i < 65536; i++) {
                dict[i] = half_to_float(i);
            }
        }
    };

    struct BF16ToFP32Manager {
        float dict[65536];

        BF16ToFP32Manager() {
            for (uint16_t i = 0; i < 65535; i++) {
                uint32_t x = (i << 16);
                dict[i] = *((float*)&x);
            }
        }
    };

    struct BF16ToFP16Manager {
        uint16_t dict[65536];

        BF16ToFP16Manager() {
            for (uint16_t i = 0; i < 65535; i++) {
                uint32_t x = (i << 16);
                dict[i] = float_to_half(*((float*)&x));
            }
        }
    };

    struct FP16ToBF16Manager {
        uint16_t dict[65536];

        FP16ToBF16Manager() {
            for (int i = 0; i < 65536; i++) {
                float f = half_to_float(i);
                dict[i] = (uint16_t)(*((uint32_t*)&f) >> 16);
            }
        }
    };

    struct FP8E4M3ToFP32Manager {
        float dict[256] = {
            0.0, 0.001953125, 0.00390625, 0.005859375, 0.0078125, 0.009765625, 0.01171875, 0.013671875, 0.015625, 0.017578125, 0.01953125, 0.021484375, 0.0234375, 0.025390625, 0.02734375, 0.029296875, 0.03125, 0.03515625, 0.0390625, 0.04296875, 0.046875, 0.05078125, 0.0546875, 0.05859375, 0.0625, 0.0703125, 0.078125, 0.0859375, 0.09375, 0.1015625, 0.109375, 0.1171875, 0.125, 0.140625, 0.15625, 0.171875, 0.1875, 0.203125, 0.21875, 0.234375, 0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.40625, 0.4375, 0.46875, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0, 64.0, 72.0, 80.0, 88.0, 96.0, 104.0, 112.0, 120.0, 128.0, 144.0, 160.0, 176.0, 192.0, 208.0, 224.0, 240.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0, 480, -0.0, -0.001953125, -0.00390625, -0.005859375, -0.0078125, -0.009765625, -0.01171875, -0.013671875, -0.015625, -0.017578125, -0.01953125, -0.021484375, -0.0234375, -0.025390625, -0.02734375, -0.029296875, -0.03125, -0.03515625, -0.0390625, -0.04296875, -0.046875, -0.05078125, -0.0546875, -0.05859375, -0.0625, -0.0703125, -0.078125, -0.0859375, -0.09375, -0.1015625, -0.109375, -0.1171875, -0.125, -0.140625, -0.15625, -0.171875, -0.1875, -0.203125, -0.21875, -0.234375, -0.25, -0.28125, -0.3125, -0.34375, -0.375, -0.40625, -0.4375, -0.46875, -0.5, -0.5625, -0.625, -0.6875, -0.75, -0.8125, -0.875, -0.9375, -1.0, -1.125, -1.25, -1.375, -1.5, -1.625, -1.75, -1.875, -2.0, -2.25, -2.5, -2.75, -3.0, -3.25, -3.5, -3.75, -4.0, -4.5, -5.0, -5.5, -6.0, -6.5, -7.0, -7.5, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0, -16.0, -18.0, -20.0, -22.0, -24.0, -26.0, -28.0, -30.0, -32.0, -36.0, -40.0, -44.0, -48.0, -52.0, -56.0, -60.0, -64.0, -72.0, -80.0, -88.0, -96.0, -104.0, -112.0, -120.0, -128.0, -144.0, -160.0, -176.0, -192.0, -208.0, -224.0, -240.0, -256.0, -288.0, -320.0, -352.0, -384.0, -416.0, -448.0, -480
        };

        // IEEE-like E4M3FN round-to-nearest-even encoder.  Codes 0..126 are
        // monotonic finite non-negative values; 127 is NaN in the format used
        // by PyTorch/CUDA.  Decode the FP32 exponent directly instead of doing
        // a binary search over the 127 positive values.
        uint8_t quantization(float value) const {
            if (std::isnan(value)) {
                return std::signbit(value) ? 0xFF : 0x7F;
            }
            bool negative = std::signbit(value);
            float magnitude = std::min(std::fabs(value), 448.0f);
            int code;
            if (magnitude < 0.015625f) {
                // E4M3 subnormals have a fixed 2^-9 spacing.
                float scaled = magnitude * 512.0f;
                int lower = (int)scaled;
                float fraction = scaled - lower;
                code = lower + (fraction > 0.5f ||
                                (fraction == 0.5f && (lower & 1)));
            } else {
                uint32_t bits;
                memcpy(&bits, &magnitude, sizeof(bits));
                int exponent = (int)((bits >> 23) & 0xFF) - 127;
                float scaled = std::ldexp(magnitude, 3 - exponent);
                int lower = (int)scaled;
                float fraction = scaled - lower;
                int significand = lower +
                    (fraction > 0.5f ||
                     (fraction == 0.5f && (lower & 1)));
                if (significand == 16) {
                    exponent++;
                    significand = 8;
                }
                code = ((exponent + 7) << 3) + significand - 8;
                code = std::min(code, 126);
            }
            return (uint8_t)(code | (negative ? 0x80 : 0));
        }

        float quantizeDequantize(float value) const {
            return dict[quantization(value)];
        }
    };

    inline uint16_t Float32ToBFloat16RNEBits(float value) {
        uint32_t bits;
        memcpy(&bits, &value, sizeof(bits));
        bits += 0x7FFFu + ((bits >> 16) & 1u);
        return (uint16_t)(bits >> 16);
    }

    inline float BFloat16BitsToFloat32(uint16_t value) {
        uint32_t bits = (uint32_t)value << 16;
        float result;
        memcpy(&result, &bits, sizeof(result));
        return result;
    }

    inline float RoundFloat32ToBFloat16RNE(float value) {
        return BFloat16BitsToFloat32(Float32ToBFloat16RNEBits(value));
    }

    class FP8E4M3BFloat16QuantizeLookup {
    public:
        static constexpr int minExponent = -32;
        static constexpr int maxExponent = 31;
        static constexpr int exponentCount =
            maxExponent - minExponent + 1;

        const uint16_t *GetRow(int exponent) {
            if (exponent < minExponent || exponent > maxExponent) {
                return nullptr;
            }
            int index = exponent - minExponent;
            std::call_once(initFlags[index], [&, index, exponent]() {
                static const FP8E4M3ToFP32Manager fp8;
                float scale = std::ldexp(1.0f, exponent);
                rows[index].resize(65536);
                for (uint32_t bits = 0; bits < 65536; bits++) {
                    float value = BFloat16BitsToFloat32((uint16_t)bits);
                    float q = std::max(
                        -448.0f, std::min(448.0f, value / scale));
                    rows[index][bits] = Float32ToBFloat16RNEBits(
                        fp8.quantizeDequantize(q) * scale);
                }
            });
            return rows[index].data();
        }

    private:
        std::array<std::once_flag, exponentCount> initFlags;
        std::array<std::vector<uint16_t>, exponentCount> rows;
    };

    inline const uint16_t *GetFP8E4M3BFloat16QuantizeLookupRow(
        int exponent
    ) {
        static FP8E4M3BFloat16QuantizeLookup lookup;
        return lookup.GetRow(exponent);
    }

    // Simulate the official DeepSeek FP8 activation path while retaining a
    // BF16 buffer for CPU GEMM kernels: block-128 E4M3FN, UE8M0 power-of-two
    // scale, saturating RNE conversion, then dequantize and round to BF16.
    inline void QuantizeDequantizeFP8E4M3Block128(float *values, int len) {
        static const FP8E4M3ToFP32Manager fp8;
        for (int start = 0; start < len; start += 128) {
            int end = std::min(start + 128, len);
            float amax = 1e-4f;
            bool allBFloat16 = true;
            for (int i = start; i < end; i++) {
                amax = std::max(amax, std::fabs(values[i]));
                uint16_t bits = Float32ToBFloat16RNEBits(values[i]);
                allBFloat16 = allBFloat16 &&
                    BFloat16BitsToFloat32(bits) == values[i];
            }
            float normalized = amax / 448.0f;
            uint32_t bits;
            memcpy(&bits, &normalized, sizeof(bits));
            int exponent = (int)((bits >> 23) & 0xFF) - 127 +
                           ((bits & ((1u << 23) - 1)) != 0);
            float scale = std::ldexp(1.0f, exponent);
            const uint16_t *lookupRow = allBFloat16 ?
                GetFP8E4M3BFloat16QuantizeLookupRow(exponent) : nullptr;
            if (lookupRow != nullptr) {
                for (int i = start; i < end; i++) {
                    uint16_t inputBits =
                        Float32ToBFloat16RNEBits(values[i]);
                    values[i] =
                        BFloat16BitsToFloat32(lookupRow[inputBits]);
                }
            } else {
                for (int i = start; i < end; i++) {
                    float q = std::max(
                        -448.0f, std::min(448.0f, values[i] / scale));
                    values[i] = RoundFloat32ToBFloat16RNE(
                        fp8.quantizeDequantize(q) * scale);
                }
            }
        }
    }

    static double GetSpan(std::chrono::system_clock::time_point time1, std::chrono::system_clock::time_point time2) {
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (time2 - time1);
        return double(duration.count()) * std::chrono::nanoseconds::period::num / std::chrono::nanoseconds::period::den;
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

    static std::vector <int> ParseDeviceIds(const std::string &s, const std::string &type, std::map <int, int> &ratios) {
        int i = type.size();
        std::vector <int> ret;
        std::string cur[2] = {"", ""};
        int idx = 0;
        if (s.size() > i && s[i] == ':') {
            i++;
            while (i < s.size()) {
                if (s[i] == 'c' && i + 2 < s.size() && s[i + 1] == 'p' && s[i + 2] == 'u') {
                    cur[0] = "99999";
                    i += 2;
                } else if (s[i] >= '0' && s[i] <= '9') {
                    cur[idx] += s[i];
                } else if (s[i] == ':' || s[i] == '-') {
                    idx = 1;
                } else {
                    if (cur[0] != "") {
                        ret.push_back(atoi(cur[0].c_str()));
                        if (cur[1] != "") {
                            ratios[atoi(cur[0].c_str())] = atoi(cur[1].c_str());
                        }
                        cur[0] = "";
                        cur[1] = "";
                        idx = 0;
                    }
                }
                i++;
            }
        }
        if (cur[0] != "") {
            ret.push_back(atoi(cur[0].c_str()));
            if (cur[1] != "") {
                ratios[atoi(cur[0].c_str())] = atoi(cur[1].c_str());
            }
        }
        return ret;
    }

#ifndef __CUDACC__
    static bool FileExists(std::string filePath) {
#if defined(__GNUC__) && __GNUC__ < 9
        return access(filePath.c_str(), R_OK) == 0;
#else
        fs::path path(filePath);
        return fs::exists(path);
#endif
    }
#endif

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
