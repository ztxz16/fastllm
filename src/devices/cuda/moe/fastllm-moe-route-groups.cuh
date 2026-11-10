#pragma once

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <algorithm>
#include <cstdint>

// Device-only CSR grouping shared by compressed expert backends. The caller
// supplies one record identity and one original route ID per route. Zero keys
// are inactive. Stable sorting preserves route order within each expert.
// All storage is caller-owned and prepared before CUDA graph capture.
namespace fastllm::moe_routes {

static __global__ void MarkGroups(const uint64_t *keys, int *heads, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) heads[i] = keys[i] && (!i || keys[i] != keys[i - 1]);
}
static __global__ void GroupRanges(const uint64_t *keys, const int *ids,
        uint64_t *records, int *starts, int *groups, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    if (keys[i] && (!i || keys[i] != keys[i - 1])) {
        records[ids[i] - 1] = keys[i];
        starts[ids[i] - 1] = i;
    }
    if (i == count - 1) { *groups = ids[i]; starts[ids[i]] = count; }
}

// A batch fitting one physical CTA needs neither a device-wide scan nor
// intermediate launches. Larger batches still use one global CSR grouping.
static constexpr int GroupThreads = 256;
static __global__ void SmallPlan(const uint64_t *keys, const int *routes,
        uint64_t *sorted, int *entries, uint64_t *records, int *starts, int *groups, int count) {
    using Sort = cub::BlockRadixSort<uint64_t, GroupThreads, 1, int>;
    using Scan = cub::BlockScan<int, GroupThreads>;
    __shared__ union { typename Sort::TempStorage sort; typename Scan::TempStorage scan; } temporary;
    const int i = threadIdx.x;
    uint64_t key[1] = {i < count ? keys[i] : UINT64_MAX};
    int entry[1] = {i < count ? routes[i] : -1};
    Sort(temporary.sort).Sort(key,entry);
    if (i < count) { sorted[i]=key[0]; entries[i]=entry[0]; }
    __syncthreads();
    const int head = i < count && key[0] && (!i || key[0] != sorted[i-1]);
    int id; Scan(temporary.scan).InclusiveSum(head,id);
    if (head) {records[id-1]=key[0];starts[id-1]=i;}
    if (i == count-1) {*groups=id;starts[id]=count;}
}

struct Plan {
    uint64_t *keys, *sortedKeys, *records;
    int *routes, *entries, *heads, *ids, *starts, *groups;
    void *temporary;
    size_t bytes, temporaryBytes;
    int count;

    static size_t TemporaryBytes(int count) {
        if (count > 0 && count <= GroupThreads) return 1;
        size_t sort = 0, scan = 0;
        if (count < 1 || cub::DeviceRadixSort::SortPairs(nullptr, sort,
                static_cast<const uint64_t *>(nullptr), static_cast<uint64_t *>(nullptr),
                static_cast<const int *>(nullptr), static_cast<int *>(nullptr), count,
                0, 64, cudaStreamPerThread) != cudaSuccess ||
            cub::DeviceScan::InclusiveSum(nullptr, scan, static_cast<const int *>(nullptr),
                static_cast<int *>(nullptr), count, cudaStreamPerThread) != cudaSuccess) return 0;
        return std::max(sort, scan);
    }
    Plan(void *buffer, int routeCount, size_t scratchBytes) : count(routeCount) {
        size_t offset = 0;
        const auto take = [&](size_t size) {
            offset = (offset + 255) & ~size_t(255);
            const uintptr_t address = reinterpret_cast<uintptr_t>(buffer) + offset;
            offset += size;
            return address;
        };
        keys = reinterpret_cast<uint64_t *>(take(size_t(count) * 8));
        sortedKeys = reinterpret_cast<uint64_t *>(take(size_t(count) * 8));
        records = reinterpret_cast<uint64_t *>(take(size_t(count) * 8));
        routes = reinterpret_cast<int *>(take(size_t(count) * 4));
        entries = reinterpret_cast<int *>(take(size_t(count) * 4));
        heads = reinterpret_cast<int *>(take(size_t(count) * 4));
        ids = reinterpret_cast<int *>(take(size_t(count) * 4));
        starts = reinterpret_cast<int *>(take((size_t(count) + 1) * 4));
        groups = reinterpret_cast<int *>(take(4));
        temporary = reinterpret_cast<void *>(take(scratchBytes));
        temporaryBytes = scratchBytes; bytes = offset;
    }
    static size_t WorkspaceBytes(int count) {
        const size_t scratch = TemporaryBytes(count);
        return scratch ? Plan(nullptr, count, scratch).bytes : 0;
    }
    bool Build(cudaStream_t stream) {
        if (count <= GroupThreads) {
            SmallPlan<<<1,GroupThreads,0,stream>>>(keys,routes,sortedKeys,entries,records,starts,groups,count);
            return cudaGetLastError() == cudaSuccess;
        }
        size_t scratch = temporaryBytes;
        if (cub::DeviceRadixSort::SortPairs(temporary, scratch, keys, sortedKeys,
                routes, entries, count, 0, 64, stream) != cudaSuccess) return false;
        MarkGroups<<<(count + 255) / 256, 256, 0, stream>>>(sortedKeys, heads, count);
        scratch = temporaryBytes;
        if (cub::DeviceScan::InclusiveSum(temporary, scratch, heads, ids, count, stream) != cudaSuccess)
            return false;
        GroupRanges<<<(count + 255) / 256, 256, 0, stream>>>(sortedKeys, ids, records, starts, groups, count);
        return cudaGetLastError() == cudaSuccess;
    }
};
} // namespace fastllm::moe_routes
