#pragma once
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

// Captured request boundaries. Live lengths, positions and KV bindings remain replay inputs.
// Equal-width graphs need no offset table, irrespective of the number of owners.
struct FastllmNaiveVerifyGroups {
    static constexpr int Tile = 8, MaxRaggedOwners = 128;
    int count = 0, width = 0, maxWidth = 0;
    unsigned short offsets[MaxRaggedOwners + 1]{};

    static FastllmNaiveVerifyGroups Plan(int rows, int width, const std::vector<int> *lengths = nullptr) {
        FastllmNaiveVerifyGroups g{};
        if (rows <= 0 || width < 1 || width > Tile ||
            (lengths && (lengths->empty() || lengths->front() != width))) return g;
        if (!lengths || std::all_of(lengths->begin(), lengths->end(), [width](int n) { return n == width; })) {
            if (rows % width || (lengths && lengths->size() != size_t(rows / width))) return g;
            g.count = rows / width;
            g.width = g.maxWidth = width;
            return g;
        }
        if (lengths->size() > MaxRaggedOwners) return g;
        int first = 0;
        for (int n : *lengths) {
            if (n < 1 || n > Tile) return {};
            first += n;
            g.offsets[++g.count] = first;
            g.maxWidth = std::max(g.maxWidth, n);
        }
        return first == rows ? g : FastllmNaiveVerifyGroups{};
    }
    __host__ __device__ int First(int owner) const { return width ? owner * width : offsets[owner]; }
    __host__ __device__ int Size(int owner) const { return width ? width : offsets[owner + 1] - offsets[owner]; }
    __device__ int Owner(int row) const {
        if (width) return row / width;
        int lo = 0, hi = count;
        while (hi - lo > 1) {
            const int mid = (lo + hi) / 2;
            if (row < offsets[mid]) hi = mid;
            else lo = mid;
        }
        return lo;
    }
};
