#pragma once
#include "fastllm.h"
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace fastllm {
    // Fixed eight-lane, four-way cascade reduction, following PyTorch's CPU
    // SumKernel.cpp (BSD; see third_party/pytorch/LICENSE). Keeping the FP32
    // grouping explicit avoids a long serial sum changing BF16 rounding in
    // Engram and RMSNorm. The CPU reference's SumKernel dispatch uses AVX2
    // even on AVX512 hosts. This scalar implementation has no ISA requirement.
    template <typename Load> float V41ReferenceSum(size_t count, const Load &load) {
        if (count < 8) {
            float partial[4] = {};
            size_t i = 0;
            for (; i + 4 <= count; i += 4)
                for (int j = 0; j < 4; ++j)
                    partial[j] += load(i + j);
            for (; i < count; ++i)
                partial[0] += load(i);
            for (int j = 1; j < 4; ++j)
                partial[0] += partial[j];
            return partial[0];
        }
        const size_t groups = count / 32;
        size_t bits = 0;
        for (size_t value = groups ? groups - 1 : 0; value; value >>= 1)
            ++bits;
        const size_t power = std::max<size_t>(4, bits / 4), step = size_t(1) << power;
        float sums[4][4][8] = {};
        size_t group = 0;
        while (group + step <= groups) {
            for (size_t end = group + step; group < end; ++group)
                for (int stream = 0; stream < 4; ++stream)
                    for (int lane = 0; lane < 8; ++lane)
                        sums[0][stream][lane] += load(group * 32 + stream * 8 + lane);
            for (int level = 1; level < 4; ++level) {
                for (int stream = 0; stream < 4; ++stream)
                    for (int lane = 0; lane < 8; ++lane) {
                        sums[level][stream][lane] += sums[level - 1][stream][lane];
                        sums[level - 1][stream][lane] = 0;
                    }
                if (group & ((step - 1) << (level * power)))
                    break;
            }
        }
        for (; group < groups; ++group)
            for (int stream = 0; stream < 4; ++stream)
                for (int lane = 0; lane < 8; ++lane)
                    sums[0][stream][lane] += load(group * 32 + stream * 8 + lane);
        for (int level = 1; level < 4; ++level)
            for (int stream = 0; stream < 4; ++stream)
                for (int lane = 0; lane < 8; ++lane)
                    sums[0][stream][lane] += sums[level][stream][lane];
        for (size_t i = groups * 32; i + 8 <= count; i += 8)
            for (int lane = 0; lane < 8; ++lane)
                sums[0][0][lane] += load(i + lane);
        for (int stream = 1; stream < 4; ++stream)
            for (int lane = 0; lane < 8; ++lane)
                sums[0][0][lane] += sums[0][stream][lane];
        float total = 0;
        for (size_t i = count / 8 * 8; i < count; ++i)
            total += load(i);
        for (int lane = 0; lane < 8; ++lane)
            total += sums[0][0][lane];
        return total;
    }
    // Optional provider uses the same SLEEF/VML entry points as the CPU reference.
    // Keep it loaded: its worker pools may outlive an individual model instance.
    struct V41ReferenceCPUMath {
#if defined(__AVX512F__) && !defined(_WIN32)
        using Vector = float __attribute__((vector_size(64)));
        using Unary = Vector (*)(Vector);
        Unary exp = nullptr, log1p = nullptr;
#elif defined(__AVX2__) && !defined(_WIN32)
        using Vector = float __attribute__((vector_size(32)));
        using Unary = Vector (*)(Vector);
        Unary exp = nullptr, log1p = nullptr;
#endif
        void (*sqrt)(int, const float *, float *, int64_t) = nullptr;
        V41ReferenceCPUMath() {
#ifndef _WIN32
            const char *path = std::getenv("FASTLLM_DEEPSEEK_V41_BLAS_LIBRARY");
            if (!path || !*path)
                return;
            void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
            if (!handle)
                throw std::runtime_error(std::string("Cannot load V4.1 reference math provider: ") + dlerror());
#if defined(__AVX512F__)
            exp = reinterpret_cast<Unary>(dlsym(handle, "Sleef_expf16_u10avx512f"));
            log1p = reinterpret_cast<Unary>(dlsym(handle, "Sleef_log1pf16_u10avx512f"));
#elif defined(__AVX2__)
            exp = reinterpret_cast<Unary>(dlsym(handle, "Sleef_expf8_u10avx2"));
            log1p = reinterpret_cast<Unary>(dlsym(handle, "Sleef_log1pf8_u10avx2"));
#endif
            sqrt = reinterpret_cast<decltype(sqrt)>(dlsym(handle, "vmsSqrt"));
#endif
        }
        float Exp(float value) const {
#if (defined(__AVX512F__) || defined(__AVX2__)) && !defined(_WIN32)
            if (exp) {
                Vector v;
                for (size_t i = 0; i < sizeof(v) / sizeof(float); ++i)
                    v[i] = value;
                return exp(v)[0];
            }
#endif
            return std::exp(value);
        }
        float Log1p(float value) const {
#if (defined(__AVX512F__) || defined(__AVX2__)) && !defined(_WIN32)
            if (log1p) {
                Vector v;
                for (size_t i = 0; i < sizeof(v) / sizeof(float); ++i)
                    v[i] = value;
                return log1p(v)[0];
            }
#endif
            return std::log1p(value);
        }
        float Sqrt(float value) const {
            if (sqrt) {
                float result;
                sqrt(1, &value, &result, 0x00280102);
                return result;
            }
            return std::sqrt(value);
        }
    };
    inline bool V41ReferenceMathEnabled() {
        static const bool enabled = [] {
            const char *v = std::getenv("FASTLLM_DSV41_REFERENCE_MATH");
            return v && *v && std::strcmp(v, "0") != 0;
        }();
        return enabled;
    }
    inline const V41ReferenceCPUMath &V41CPUMath() {
        static const V41ReferenceCPUMath math;
        return math;
    }
} // namespace fastllm
