#include "fastllm.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/naive-n05-cuda.cuh"
#include "devices/multicuda/fastllm-multicuda.cuh"
#include "utils/persistent_worker_group.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
using namespace fastllm;
static void Check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}

static void PackedSplit(const std::vector<int> &ids, bool rows, bool cpuSource) {
    int ranks = ids.size(), n = ranks * 6, m = ranks * 48;
    Data weight(NVFP4_BLOCK_16_E4M3_PACKED, {n, m});
    weight.name = "test.packed";
    weight.blockK = 1;
    weight.blockM = 16;
    weight.Allocate();
    size_t stride = GetDataBytes(weight.dataType, 1, m);
    std::vector<uint8_t> original(weight.GetBytes());
    for (size_t i = 0; i < original.size(); ++i)
        original[i] = (i * 73 + i / stride * 19) & 255;
    std::memcpy(weight.cpuData, original.data(), original.size());
    if (!cpuSource)
        weight.ToDevice(DataDevice::CUDA, std::vector<int>{ids[0]});
    DivisionScheme scheme;
    for (int r = 0; r < ranks; ++r) {
        if (rows)
            scheme[ids[r]] = {{r * 3, (r + 1) * 3}, {n / 2 + r * 3, n / 2 + (r + 1) * 3}};
        else
            scheme[ids[r]] = {{r * 48, (r + 1) * 48}};
    }
    Data bias;
    auto devices = ids;
    Check(SplitMultiCudaWeight(weight, bias, devices, scheme, rows ? 0 : 1, true), "packed split failed");
    for (int r = 0; r < ranks; ++r) {
        Data local(*weight.multiDeviceDatas.at(ids[r]));
        local.ToDevice(DataDevice::CPU);
        size_t localStride = GetDataBytes(weight.dataType, 1, local.dims[1]);
        for (int row = 0; row < local.dims[0]; ++row) {
            int sourceRow = rows ? (row < 3 ? r * 3 + row : n / 2 + r * 3 + row - 3) : row;
            const uint8_t *source = original.data() + sourceRow * stride;
            const uint8_t *dest = local.cpuData + row * localStride;
            Check(std::memcmp(dest, source, 4) == 0, "packed global scale changed");
            size_t offset = rows ? 0 : r * 3 * 9;
            size_t payload = ((local.dims[1] + 15) / 16) * 9;
            Check(std::memcmp(dest + 4, source + 4 + offset, payload) == 0, "packed FP4 blocks changed");
        }
    }
}

static void ReplicatedKV(const std::vector<int> &ids, int kvHeads, int queries, int past) {
    const int heads = 64, dim = 192, vd = 128, ranks = ids.size();
    int extra = kvHeads == 4 ? 128 : 0;
    std::vector<float> qv(queries * heads * dim), kv(past * (kvHeads * dim + extra)), vv(past * kvHeads * vd);
    for (int i = 0; i < (int)qv.size(); ++i)
        qv[i] = ((i * 7 % 31) - 15) * .007f;
    for (int i = 0; i < (int)kv.size(); ++i)
        kv[i] = ((i * 13 % 29) - 14) * .008f;
    for (int i = 0; i < (int)vv.size(); ++i)
        vv[i] = ((i * 11 % 37) - 18) * .02f;
    std::vector<float> sv(heads);
    for (int i = 0; i < heads; ++i)
        sv[i] = i * .01f;
    FastllmCudaSetDevice(ids[0]);
    Data q(BFLOAT16, {1, queries, heads * dim}, qv), k(BFLOAT16, {1, past, kvHeads * dim + extra}, kv);
    Data v(BFLOAT16, {1, past, kvHeads * vd}, vv), sink(FLOAT32, {heads}, sv), none, full;
    q.ToDevice(DataDevice::CUDA, std::vector<int>{ids[0]});
    k.ToDevice(DataDevice::CUDA, std::vector<int>{ids[0]});
    v.ToDevice(DataDevice::CUDA, std::vector<int>{ids[0]});
    sink.ToDevice(DataDevice::CUDA, std::vector<int>{ids[0]});
    FastllmCudaNaiveAttention(q, k, v, none, sink, heads, kvHeads, dim, vd, past - queries,
                              kvHeads == 8 ? 128 : 0, full);
    full.ToDevice(DataDevice::CPU);
    for (int r = 0; r < ranks; ++r) {
        int localHeads = heads / ranks, localKv = std::max(1, kvHeads / ranks), begin = r * kvHeads / ranks;
        auto cut = [](const std::vector<float> &src, int rows, int stride, int start, int width) {
            std::vector<float> out;
            for (int row = 0; row < rows; ++row)
                out.insert(out.end(), src.begin() + row * stride + start,
                           src.begin() + row * stride + start + width);
            return out;
        };
        auto keys = cut(kv, past, kvHeads * dim + extra, begin * dim, localKv * dim);
        if (extra) {
            std::vector<float> packed;
            for (int t = 0; t < past; ++t) {
                packed.insert(packed.end(), keys.begin() + t * localKv * dim,
                              keys.begin() + (t + 1) * localKv * dim);
                packed.insert(packed.end(), kv.begin() + t * (kvHeads * dim + extra) + kvHeads * dim,
                              kv.begin() + (t + 1) * (kvHeads * dim + extra));
            }
            keys.swap(packed);
        }
        FastllmCudaSetDevice(ids[r]);
        Data lq(BFLOAT16, {1, queries, localHeads * dim},
                cut(qv, queries, heads * dim, r * localHeads * dim, localHeads * dim));
        Data lk(BFLOAT16, {1, past, localKv * dim + extra}, keys);
        Data lv(BFLOAT16, {1, past, localKv * vd}, cut(vv, past, kvHeads * vd, begin * vd, localKv * vd));
        Data ls(FLOAT32, {localHeads}, cut(sv, 1, heads, r * localHeads, localHeads)), out;
        for (Data *x : {&lq, &lk, &lv, &ls})
            x->ToDevice(DataDevice::CUDA, std::vector<int>{ids[r]});
        FastllmCudaNaiveAttention(lq, lk, lv, none, ls, localHeads, localKv, dim, vd, past - queries,
                                  kvHeads == 8 ? 128 : 0, out);
        out.ToDevice(DataDevice::CPU);
        for (int t = 0; t < queries; ++t)
            Check(std::memcmp(out.cpuData + t * localHeads * vd * 2,
                              full.cpuData + (t * heads + r * localHeads) * vd * 2, localHeads * vd * 2) == 0,
                  "TP KV replication changed attention");
    }
}
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 8)
        return 77;
    try {
        std::vector<int> devices(8);
        std::iota(devices.begin(), devices.end(), 0);
        for (bool rows : {false, true})
            for (bool cpu : {false, true})
                PackedSplit(devices, rows, cpu);
        for (int kv : {4, 8})
            for (int queries : {1, 3})
                ReplicatedKV(devices, kv, queries, 143);
        Check(FastllmInitNccl(devices), "NCCL initialization failed");
        PersistentWorkerGroup workers;
        std::vector<std::exception_ptr> errors(8);
        for (int iteration = 0; iteration < 3; ++iteration) {
            workers.Run(
                devices,
                [&](int rank) {
                    FastllmCudaSetDevice(devices[rank]);
                    Data data(BFLOAT16, {1, 4096}, std::vector<float>(4096, rank + 1));
                    data.ToDevice(DataDevice::CUDA, std::vector<int>{devices[rank]});
                    FastllmNcclAllReduce(data.cudaData, data.cudaData, 4096, BFLOAT16, devices[rank]);
                    data.ToDevice(DataDevice::CPU);
                    for (int i = 0; i < 4096; ++i)
                        Check(((uint16_t *)data.cpuData)[i] == 0x4210, "allreduce sum incorrect");
                },
                errors);
            for (auto e : errors)
                if (e)
                    std::rethrow_exception(e);
        }
        std::cout << "PASS: packed CPU/GPU row/column shards; global/SWA KV heads; 3 persistent-worker "
                     "allreduces\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
