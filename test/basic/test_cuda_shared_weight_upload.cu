#include "devices/cuda/fastllm-cuda-shared-weight.cuh"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void Check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

static void Require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

static void Run(unsigned hidden, unsigned inter, unsigned shards, bool planar,
                unsigned sourceSkew = 0, bool auxiliary = false, bool packed = false, bool registered = false,
                unsigned records = 2, fastllm::cuda::RecordCopyLaunch launch = {32, 256}, bool mappedControl = false) {
    using namespace fastllm::cuda;
    SharedExpertLayout layout;
    layout.shards = shards;
    layout.weights[0] = SharedWeightView::Rows(2 * inter, hidden / 32 * 17, shards, 2);
    layout.weights[1] = SharedWeightView::Rows(hidden, inter / 32 * 17, shards);
    layout.weights[1].destinationOffset = 2 * inter * layout.weights[0].rowBytes;
    layout.recordBytes = layout.weights[1].destinationOffset + hidden * layout.weights[1].rowBytes;
    if (packed) {
        for (unsigned part = 0; part < 2; ++part) {
            auto &view = layout.weights[part];
            const unsigned columns = part ? inter : hidden;
            view.rowBytes = columns / 2;
            view.rowStride = view.tileStride = (4 + (columns / 16) * 9 + 3) & ~3u;
            view.sourceOffset = 4; view.blockBytes = 8; view.blockStride = 9;
        }
        layout.weights[1].destinationOffset = 2 * inter * layout.weights[0].rowBytes;
        layout.recordBytes = layout.weights[1].destinationOffset + hidden * layout.weights[1].rowBytes;
    }
    if (auxiliary) {
        layout.auxiliaryBytes = 64;
        layout.auxiliary[0] = {0, layout.recordBytes, 16};
        layout.auxiliary[1] = {16, layout.recordBytes + 16, 32};
        layout.auxiliary[2] = {48, layout.recordBytes + 48, 16};
        layout.recordBytes += 64;
    }
    for (auto &view : layout.weights) view.block32Planar = planar;
    Require(layout.Prepare(), "layout rejected");
    if (packed) for (const auto &view : layout.weights)
        Require(view.shardBytes == view.rowsPerShard * view.rowStride, "row prefix extended NUMA shard");
    const size_t bytes = size_t(layout.sourceBytes[0]) + layout.sourceBytes[1] + layout.auxiliaryBytes;
    constexpr unsigned guard = 32;
    const size_t stride = size_t(layout.recordBytes) + 2 * guard;
    std::vector<void *> host(records * 2 * shards), mapped(host.size()), owners(host.size());
    uint8_t *hostAuxiliary = nullptr, *mappedAuxiliary = nullptr;
    MappedHostBuffer auxiliaryOwner;
    if (auxiliary) {
        if (registered) {
            Check(auxiliaryOwner.Allocate((size_t(32) << 20) + records * layout.auxiliaryBytes));
            Require(auxiliaryOwner.Registrations() >= 3, "auxiliary split registration not exercised");
            const size_t offset = (size_t(16) << 20) - layout.auxiliaryBytes / 2;
            hostAuxiliary = static_cast<uint8_t *>(auxiliaryOwner.Host()) + offset;
            mappedAuxiliary = static_cast<uint8_t *>(auxiliaryOwner.Device()) + offset;
        } else {
            Check(cudaHostAlloc(&hostAuxiliary, records * layout.auxiliaryBytes, cudaHostAllocMapped));
            Check(cudaHostGetDevicePointer(&mappedAuxiliary, hostAuxiliary, 0));
        }
    }
    for (unsigned i = 0; i < host.size(); ++i) {
        Check(cudaHostAlloc(&owners[i], layout.weights[(i / shards) % 2].shardBytes + sourceSkew, cudaHostAllocMapped));
        Check(cudaHostGetDevicePointer(&mapped[i], owners[i], 0));
        host[i] = static_cast<uint8_t *>(owners[i]) + sourceSkew;
        mapped[i] = static_cast<uint8_t *>(mapped[i]) + sourceSkew;
    }
    void **devicePointers;
    uint8_t *scratch, *actual, *legacy;
    Check(cudaMalloc(&devicePointers, mapped.size() * sizeof(void *)));
    Check(cudaMemcpy(devicePointers, mapped.data(), mapped.size() * sizeof(void *), cudaMemcpyHostToDevice));
    Check(cudaMalloc(&scratch, bytes));
    Check(cudaMalloc(&actual, records * stride));
    Check(cudaMalloc(&legacy, records * layout.recordBytes));
    cudaStream_t stream;
    Check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    Require(!UploadSharedExpertRecord(layout, host.data(), scratch, bytes - 1, actual + guard, stream, hostAuxiliary),
        "short staging accepted");
    Require(!UploadSharedExpertRecord(layout, host.data(), scratch, bytes, scratch, stream, hostAuxiliary),
        "aliased staging accepted");
    if (auxiliary) Require(!UploadSharedExpertRecord(layout, host.data(), scratch, bytes, actual + guard, stream),
        "missing auxiliary metadata accepted");
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    // Different source records share one staging buffer. Both outputs must
    // survive, detecting an overwrite before the preceding gather finishes.
    for (unsigned record = 0; record < records; ++record)
        Require(UploadSharedExpertRecord(layout, host.data() + record * 2 * shards,
            scratch, bytes, actual + record * stride + guard, stream,
            auxiliary ? hostAuxiliary + record * layout.auxiliaryBytes : nullptr,
            registered ? &auxiliaryOwner : nullptr), "upload rejected");
    Check(cudaStreamEndCapture(stream, &graph));
    Check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (unsigned pass = 0; pass < 3; ++pass) {
        std::vector<uint8_t> expected(records * layout.recordBytes, 0), output(records * stride), old(expected.size());
        for (unsigned i = 0; i < host.size(); ++i)
            for (size_t j = 0; j < layout.weights[(i / shards) % 2].shardBytes; ++j)
                static_cast<uint8_t *>(host[i])[j] = (j * 23 + (j >> 7) * 11 + i * 41 + pass * 73) & 255;
        // Independent CPU oracle maps canonical gate/up rows to interleaved
        // NUMA rows; all packed codes and scale bytes must remain identical.
        for (unsigned record = 0; record < records; ++record) {
            if (auxiliary) {
                for (unsigned byte = 0; byte < layout.auxiliaryBytes; ++byte)
                    hostAuxiliary[record * layout.auxiliaryBytes + byte] = (byte * 7 + record * 41 + pass * 19) & 255;
                std::memcpy(expected.data() + (record + 1) * layout.recordBytes - layout.auxiliaryBytes,
                    hostAuxiliary + record * layout.auxiliaryBytes, layout.auxiliaryBytes);
            }
            for (unsigned part = 0; part < 2; ++part) {
                const auto &v = layout.weights[part];
                for (unsigned row = 0; row < v.rows; ++row) {
                    const unsigned physical = part == 0 ? 2 * (row % inter) + row / inter : row;
                    const unsigned node = physical / v.rowsPerShard, local = physical % v.rowsPerShard;
                    const auto *source = static_cast<uint8_t *>(host[(record * 2 + part) * shards + node]) + size_t(local) * v.rowStride + v.sourceOffset;
                    auto *destination = expected.data() + record * layout.recordBytes + v.destinationOffset;
                    for (unsigned byte = 0; byte < v.rowBytes; ++byte) {
                        const size_t canonical = size_t(row) * v.rowBytes + byte;
                        const size_t block = canonical / 17, within = canonical % 17;
                        const size_t offset = planar ? (within < 16 ? block * 16 + within :
                            size_t(v.rows) * (v.rowBytes / 17) * 16 + block) : canonical;
                        destination[offset] = source[(byte / v.blockBytes) * v.blockStride + byte % v.blockBytes];
                    }
                }
            }
            for (int stage : {0, 1})
                Require(CopySharedExpertRecord(layout, devicePointers + record * 2 * shards,
                    nullptr, nullptr, auxiliary ? mappedAuxiliary + record * layout.auxiliaryBytes : nullptr,
                    legacy + record * layout.recordBytes, stream, stage), "staged copy rejected");
        }
        Check(cudaMemsetAsync(actual, 0xa5, output.size(), stream));
        Check(cudaGraphLaunch(exec, stream));
        Check(cudaStreamSynchronize(stream));
        Check(cudaMemcpy(output.data(), actual, output.size(), cudaMemcpyDeviceToHost));
        Check(cudaMemcpy(old.data(), legacy, old.size(), cudaMemcpyDeviceToHost));
        Require(old == expected, "legacy/CPU oracle mismatch");
        for (unsigned record = 0; record < records; ++record) {
            const auto *result = output.data() + record * stride;
            Require(!std::memcmp(result + guard, expected.data() + record * layout.recordBytes,
                layout.recordBytes), "DMA/CPU oracle mismatch");
            for (unsigned j = 0; j < guard; ++j)
                Require(result[j] == 0xa5 && result[layout.recordBytes + guard + j] == 0xa5, "guard overwritten");
        }
        // Batched graph replay reads its count and route IDs on the device.
        // Change both between replays, including zero work and a repeated source.
        int32_t *ids, *hostIds = nullptr;
        if (mappedControl) {
            Check(cudaHostAlloc(&hostIds, (2 * records + 1) * sizeof(int32_t), cudaHostAllocMapped));
            Check(cudaHostGetDevicePointer(&ids, hostIds, 0));
        } else Check(cudaMalloc(&ids, (2 * records + 1) * sizeof(int32_t)));
        cudaGraph_t batchGraph;
        cudaGraphExec_t batchExec;
        Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        Require(CopySharedExpertRecords(layout, devicePointers, mappedAuxiliary, legacy,
            ids, ids + records, ids + 2 * records, records, launch, stream), "batch copy rejected");
        Check(cudaStreamEndCapture(stream, &batchGraph));
        Check(cudaGraphInstantiate(&batchExec, batchGraph, nullptr, nullptr, 0));
        for (unsigned count : {0u, 1u, (records + 1) / 2, records}) {
            std::vector<int32_t> mapping(2 * records + 1);
            for (unsigned i = 0; i < records; ++i) {
                mapping[i] = pass == 1 ? records - 1 : records - 1 - i;
                mapping[records + i] = records - 1 - i;
            }
            mapping[2 * records] = count;
            Check(cudaMemsetAsync(legacy, 0xa5, old.size(), stream));
            if (mappedControl) std::memcpy(hostIds, mapping.data(), mapping.size() * sizeof(int32_t));
            else Check(cudaMemcpyAsync(ids, mapping.data(), mapping.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            Check(cudaGraphLaunch(batchExec, stream));
            Check(cudaStreamSynchronize(stream));
            Check(cudaMemcpy(old.data(), legacy, old.size(), cudaMemcpyDeviceToHost));
            std::vector<uint8_t> batchExpected(old.size(), 0xa5);
            for (int i = 0; i < count; ++i)
                std::memcpy(batchExpected.data() + mapping[records + i] * layout.recordBytes,
                    expected.data() + mapping[i] * layout.recordBytes, layout.recordBytes);
            Require(old == batchExpected, "batch dynamic-count/CPU oracle mismatch");
        }
        cudaGraphExecDestroy(batchExec); cudaGraphDestroy(batchGraph);
        if (mappedControl) cudaFreeHost(hostIds); else cudaFree(ids);
    }
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
    cudaFree(devicePointers); cudaFree(scratch); cudaFree(actual); cudaFree(legacy);
    for (auto p : owners) cudaFreeHost(p);
    if (hostAuxiliary && !registered) cudaFreeHost(hostAuxiliary);
    std::printf("PASS shared upload hidden=%u inter=%u shards=%u skew=%u auxiliary=%u bytes=%u records=%u grid=%d threads=%d mapped_control=%d staged and dynamic batch graphs\n",
        hidden, inter, shards, sourceSkew, layout.auxiliaryBytes, layout.recordBytes, records, launch.blocks, launch.threads, int(mappedControl));
}

int main(int argc, char **argv) {
    const bool planar = argc > 1 && std::string(argv[1]) == "--planar";
    const bool registered = argc > 1 && std::string(argv[1]) == "--registered-auxiliary";
    const bool auxiliary = registered || (argc > 1 && std::string(argv[1]) == "--auxiliary");
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try {
        if (registered) {
            for (unsigned shards : {1u, 2u, 4u, 8u}) Run(2560, 640, shards, false, 0, true, true, true);
            std::puts("ALL_PASS shared auxiliary DMA across registrations");
            return 0;
        }
        for (unsigned shards : {1u, 2u, 4u, 8u}) {
            Run(96, 96, shards, planar, 0, auxiliary);
            Run(128, 128, shards, planar, 0, auxiliary);
            Run(512, 256, shards, planar, 0, auxiliary);
            Run(5120, 2304, shards, planar, 0, auxiliary);
        }
        if (auxiliary) for (unsigned shards : {1u, 2u, 4u, 8u})
            Run(2560, 640, shards, false, 0, true, true);
        if (auxiliary) {
            // Odd packed strides, skewed borrowed bases, shard ends and
            // mapped routing IDs must survive dynamic graph replay exactly.
            for (unsigned skew : {0u, 1u, 2u, 3u})
                for (unsigned records : {3u, 7u, 16u, 33u})
                    Run(96, 160, 2, false, skew, true, true, false,
                        records, {7, 256}, true);
            Run(16, 16, 4, false, 3, true, true, false, 3, {1, 128}, true);
            Run(2560, 640, 2, false, 0, true, true, false, 3, {48, 1024}, true);
        }
        if (planar) for (unsigned skew : {1u, 2u, 3u}) Run(128, 96, 2, true, skew);
        if (planar) {
            // Uneven tiles, sparse outputs, repeated sources and multiple
            // records share a bounded grid, including a single CTA.
            for (auto launch : {fastllm::cuda::RecordCopyLaunch{1, 128}, {7, 256}, {48, 1024}})
                for (unsigned records : {3u, 7u, 16u, 33u})
                    Run(96, 160, 2, true, records % 4, false, false, false, records, launch, true);
            Run(5120, 2304, 2, true, 0, false, false, false, 8, {48, 1024}, true);
        }
        std::puts("ALL_PASS shared expert DMA upload");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
