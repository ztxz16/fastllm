// Compile for the local GPU. A second build with -D__gfx942__ exercises the
// gfx942 OCP software-conversion branch on the same hardware.
#include <hip/hip_runtime.h>
#include "fastllm-hip.h"
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

void CheckHip(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}

float Decode(unsigned int code) {
    unsigned int magnitude = code & 127, exponent = magnitude >> 3, mantissa = magnitude & 7;
    float value = magnitude == 127 ? NAN : exponent == 0 ? std::ldexp(float(mantissa), -9)
                                                           : std::ldexp(float(8 + mantissa), int(exponent) - 10);
    return std::copysign(value, code & 128 ? -1.0f : 1.0f);
}

unsigned int Encode(float value) {
    unsigned int sign = std::signbit(value) ? 128 : 0;
    if (std::isnan(value)) return 127 | sign;
    float magnitude = std::fabs(value);
    if (magnitude >= 448) return 126 | sign;
    unsigned int best = 0;
    float distance = magnitude;
    for (unsigned int i = 1; i <= 126; ++i) {
        float difference = std::fabs(magnitude - Decode(i));
        if (difference < distance || (difference == distance && (i & 1) == 0)) {
            distance = difference;
            best = i;
        }
    }
    return best | sign;
}

__global__ void Convert(const float *input, unsigned char *encoded, int count, float *decoded) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        fastllm_hip::Fp8E4M3 value(input[i]);
        encoded[i] = value.__x;
    }
    if (i < 65536) {
        fastllm_hip::Fp8x2E4M3 pair;
        pair.__x = i;
        float2 values = static_cast<float2>(pair);
        decoded[2 * i] = values.x;
        decoded[2 * i + 1] = values.y;
    }
}

int main() {
    try {
        std::vector<float> input{0.0f, -0.0f, INFINITY, -INFINITY, NAN, -NAN,
                                 std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};
        for (int i = 0; i < 256; ++i) input.push_back(Decode(i));
        for (int i = 0; i < 126; ++i) {
            float midpoint = (Decode(i) + Decode(i + 1)) * 0.5f;
            for (float v : {midpoint, std::nextafter(midpoint, INFINITY), std::nextafter(midpoint, -INFINITY)}) {
                input.push_back(v); input.push_back(-v);
            }
        }
        for (int i = -16000; i <= 16000; ++i) input.push_back(i / 32.0f);
        float *gpuInput, *gpuDecoded;
        unsigned char *gpuEncoded;
        CheckHip(hipMalloc(&gpuInput, input.size() * sizeof(float)));
        CheckHip(hipMalloc(&gpuEncoded, input.size()));
        CheckHip(hipMalloc(&gpuDecoded, 131072 * sizeof(float)));
        CheckHip(hipMemcpy(gpuInput, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice));
        Convert<<<256, 256>>>(gpuInput, gpuEncoded, input.size(), gpuDecoded);
        CheckHip(hipGetLastError());
        std::vector<unsigned char> encoded(input.size());
        std::vector<float> decoded(131072);
        CheckHip(hipMemcpy(encoded.data(), gpuEncoded, encoded.size(), hipMemcpyDeviceToHost));
        CheckHip(hipMemcpy(decoded.data(), gpuDecoded, decoded.size() * sizeof(float), hipMemcpyDeviceToHost));
        for (size_t i = 0; i < input.size(); ++i) {
            unsigned int expected = Encode(input[i]);
            if (std::isnan(input[i]) ? (encoded[i] & 127) != 127 : encoded[i] != expected)
                throw std::runtime_error("FP8 rounding/saturation mismatch at " + std::to_string(i));
        }
        for (int i = 0; i < 65536; ++i) {
            for (int lane = 0; lane < 2; ++lane) {
                float expected = Decode((i >> (lane * 8)) & 255), actual = decoded[2 * i + lane];
                if (std::isnan(expected) ? !std::isnan(actual) :
                    (expected != actual || std::signbit(expected) != std::signbit(actual)))
                    throw std::runtime_error("FP8 byte-pair decode mismatch");
            }
        }
        CheckHip(hipFree(gpuInput)); CheckHip(hipFree(gpuEncoded)); CheckHip(hipFree(gpuDecoded));
        std::puts("PASS: all 65536 FP8 byte pairs, rounding ties, subnormals, signed zeros, NaN and saturation");
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
