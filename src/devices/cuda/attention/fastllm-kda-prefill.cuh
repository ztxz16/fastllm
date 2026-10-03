#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {
    constexpr int KIMI_K3_KDA_DIMENSION = 128;
    constexpr int KIMI_K3_KDA_VALUE_TILE = 16;

    // Each warp prepares one token/head. Preserve the recurrent kernel's FP32
    // reduction order and its rounded sigmoid, retention and normalization.
    __global__ void KimiK3KdaPrepareKernel(
            const __nv_bfloat16 *q, const __nv_bfloat16 *k,
            const __nv_bfloat16 *rawGate, const float *rawBeta,
            const float *aLog, const float *dtBias,
            float *query, float *key, float *retention, float *beta,
            int rows, int sequence, int heads, float lowerBound) {
        constexpr int dimension = KIMI_K3_KDA_DIMENSION;
        int lane = threadIdx.x % 32;
        int warp = threadIdx.x / 32;
        int row = blockIdx.x * 4 + warp;
        if (row >= rows) {
            return;
        }
        int head = row % heads;
        int token = (row / heads) % sequence;
        int batch = row / heads / sequence;
        size_t source = (size_t)row * dimension;
        size_t targetRow = ((size_t)batch * heads + head) * sequence + token;
        size_t target = targetRow * dimension;
        __shared__ float rawQuery[4][dimension], rawKey[4][dimension];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int channel = lane + i * 32;
            rawQuery[warp][channel] = __bfloat162float(q[source + channel]);
            rawKey[warp][channel] = __bfloat162float(k[source + channel]);
        }
        __syncwarp();
        float queryScale = 0.0f, keyScale = 0.0f;
        if (lane == 0) {
            for (int channel = 0; channel < dimension; ++channel) {
                float qv = rawQuery[warp][channel];
                float kv = rawKey[warp][channel];
                queryScale = __fmaf_rn(qv, qv, queryScale);
                keyScale = __fmaf_rn(kv, kv, keyScale);
            }
            queryScale = rsqrtf(queryScale + 1e-6f);
            keyScale = rsqrtf(keyScale + 1e-6f);
            float activated = 1.0f / (1.0f + expf(-rawBeta[row]));
            beta[targetRow] = __bfloat162float(__float2bfloat16_rn(activated));
        }
        queryScale = __shfl_sync(0xffffffff, queryScale, 0);
        keyScale = __shfl_sync(0xffffffff, keyScale, 0);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int channel = lane + i * 32;
            query[target + channel] = rawQuery[warp][channel] * queryScale;
            key[target + channel] = rawKey[warp][channel] * keyScale;
            float raw = __bfloat162float(rawGate[source + channel]);
            float value = expf(aLog[head]) *
                (raw + dtBias[(size_t)head * dimension + channel]);
            // Multiplication after the rounded reciprocal is intentional:
            // lowerBound / denominator changes the original FP32 semantics.
            float gate = __fmul_rn(lowerBound, 1.0f / (1.0f + expf(-value)));
            retention[target + channel] = expf(gate);
        }
    }

    // One warp owns 16 value columns. The complete FP32 state column lives in
    // registers, eliminating per-token global state traffic and CTA barriers.
    // Unlike a triangular chunk solve, this keeps every accumulation in the
    // same order as KimiK3RecurrentKDAKernel, including the decay rounding.
    __global__ void KimiK3KdaRegisterScanKernel(
            const float *query, const float *key, const float *retention,
            const float *beta, const __nv_bfloat16 *v,
            float *state, __nv_bfloat16 *output,
            int sequence, int heads, int runtimeDimension) {
        constexpr int dimension = KIMI_K3_KDA_DIMENSION;
        constexpr int valueTile = KIMI_K3_KDA_VALUE_TILE;
        constexpr int tiles = dimension / valueTile;
        int item = blockIdx.x / tiles;
        int head = item % heads;
        int batch = item / heads;
        int lane = threadIdx.x;
        int column = (blockIdx.x % tiles) * valueTile + lane;
        __shared__ float q[dimension], r[dimension];
        // Reload keys for the output pass instead of retaining another 128
        // values per thread alongside the state and spilling registers.
        __shared__ volatile float k[dimension];
        float columnState[dimension];
        #pragma unroll
        for (int channel = 0; channel < dimension; ++channel) {
            columnState[channel] = lane < valueTile ?
                state[((size_t)item * dimension + channel) * dimension + column] : 0.0f;
        }
        float outputScale = rsqrtf((float)runtimeDimension);
        for (int token = 0; token < sequence; ++token) {
            size_t preparedRow = (size_t)item * sequence + token;
            size_t prepared = preparedRow * dimension;
            // A fixed, fully unrolled stride regresses the scan on SM120.
            #pragma unroll
            for (int channel = lane; channel < dimension; channel += blockDim.x) {
                q[channel] = query[prepared + channel];
                k[channel] = key[prepared + channel];
                r[channel] = retention[prepared + channel];
            }
            __syncwarp();
            if (lane < valueTile) {
                float prediction = 0.0f;
                #pragma unroll
                for (int channel = 0; channel < dimension; ++channel) {
                    columnState[channel] = __fmul_rn(columnState[channel], r[channel]);
                    prediction = __fmaf_rn(k[channel], columnState[channel], prediction);
                }
                size_t source = (((size_t)batch * sequence + token) * heads + head) *
                    dimension + column;
                float delta = (__bfloat162float(v[source]) - prediction) * beta[preparedRow];
                float result = 0.0f;
                #pragma unroll
                for (int channel = 0; channel < dimension; ++channel) {
                    columnState[channel] = __fmaf_rn(k[channel], delta, columnState[channel]);
                    result = __fmaf_rn(q[channel], columnState[channel], result);
                }
                output[source] = __float2bfloat16_rn(result * outputScale);
            }
            __syncwarp();
        }
        if (lane < valueTile) {
            #pragma unroll
            for (int channel = 0; channel < dimension; ++channel) {
                state[((size_t)item * dimension + channel) * dimension + column] =
                    columnState[channel];
            }
        }
    }

    void KimiK3LaunchKdaPrefill(
            const void *q, const void *k, const void *v, const void *gate,
            const float *beta, const float *aLog, const float *dtBias,
            float *state, void *output, float *scratch,
            int batch, int sequence, int heads, int dimension, float lowerBound) {
        int rows = batch * sequence * heads;
        size_t elements = (size_t)rows * dimension;
        float *query = scratch;
        float *key = query + elements;
        float *retention = key + elements;
        float *activatedBeta = retention + elements;
        KimiK3KdaPrepareKernel<<<(rows + 3) / 4, 128, 0, cudaStreamPerThread>>>(
            (const __nv_bfloat16*)q, (const __nv_bfloat16*)k,
            (const __nv_bfloat16*)gate, beta, aLog, dtBias,
            query, key, retention, activatedBeta, rows, sequence, heads, lowerBound);
        KimiK3KdaRegisterScanKernel
            <<<batch * heads * (dimension / KIMI_K3_KDA_VALUE_TILE), 32, 0, cudaStreamPerThread>>>(
                query, key, retention, activatedBeta, (const __nv_bfloat16*)v,
                state, (__nv_bfloat16*)output, sequence, heads, dimension);
    }
}
