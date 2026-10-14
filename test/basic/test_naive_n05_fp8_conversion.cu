#include "devices/cuda/naive-n05-kv.cuh"
#include <cuda_runtime.h>
#include <cstdio>

__global__ void CheckConversion(unsigned short *output) {
    __shared__ __align__(16) unsigned char encoded[256];
    const int i = threadIdx.x;
    encoded[i] = i;
    __syncthreads();
    FastllmNaiveKVRow row(encoded, true);
    __nv_fp8_e4m3 value;
    value.__x = i;
    output[i] = __bfloat16_as_ushort(__float2bfloat16_rn((float)value));
    output[256 + i] = __bfloat16_as_ushort(row[i]);
    if (i < 32) ((uint4 *)(output + 512))[i] = (row + i * 8).Load8();
    if (i < 64) ((uint2 *)(output + 768))[i] = (row + i * 4).Load4();
    ((unsigned *)(output + 1024))[i] = __float_as_uint(row.LoadFloat(i));
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    unsigned short *device = nullptr, result[1536];
    if (cudaMalloc(&device, sizeof(result)) != cudaSuccess) return 1;
    CheckConversion<<<1, 256>>>(device);
    auto status = cudaMemcpy(result, device, sizeof(result), cudaMemcpyDeviceToHost);
    cudaFree(device);
    if (status != cudaSuccess) return 1;
    for (int part = 1; part < 4; ++part) for (int i = 0; i < 256; ++i) {
        if (result[part * 256 + i] != result[i]) {
            std::fprintf(stderr, "E4M3 conversion mismatch: path=%d bits=%02x\n", part, i);
            return 1;
        }
    }
    for (int i = 0; i < 256; ++i) {
        unsigned actual = ((unsigned *)(result + 1024))[i], expected = (unsigned)result[i] << 16;
        bool nan = (expected & 0x7f800000u) == 0x7f800000u && (expected & 0x7fffffu);
        if (nan ? !((actual & 0x7f800000u) == 0x7f800000u && (actual & 0x7fffffu)) : actual != expected) {
            std::fprintf(stderr, "FP8 float load mismatch: bits=%02x\n", i);
            return 1;
        }
    }
    std::puts("FP8 CONVERSION PASS: all 256 encodings, scalar and vector loads");
    return 0;
}
