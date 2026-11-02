#pragma once

#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/disk/diskdevice.h"
#include <stdexcept>

// A failed callback may already have enqueued a graph. Both failure forms must
// drain GPU readers before releasing the cache lease, and permit another replay.
inline void CheckDiskGraphCancellation(fastllm::DiskMoeV41Graph &graph, void *exec) {
    struct Cancelled {};
    for (bool submitted : {false, true}) {
        auto enqueue = [&] {
            if (submitted && !FastllmCudaGraphLaunch(exec))
                throw std::runtime_error("cancellation test could not launch CUDA graph");
        };
        if (graph.Launch([&] {
                enqueue();
                return false;
            }))
            throw std::runtime_error("cancelled disk graph replay succeeded");
        bool caught = false;
        try {
            graph.Launch([&]() -> bool {
                enqueue();
                throw Cancelled{};
            });
        } catch (const Cancelled &) {
            caught = true;
        }
        if (!caught)
            throw std::runtime_error("disk graph replay lost callback exception");
    }
}
