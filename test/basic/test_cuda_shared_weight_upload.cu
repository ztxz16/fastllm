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

static void Run(unsigned hidden, unsigned inter, unsigned shards, bool planar) {
    using namespace fastllm::cuda;
    SharedExpertLayout layout;
    layout.shards = shards;
    layout.weights[0] = SharedWeightView::Rows(2 * inter, hidden / 32 * 17, shards, 2);
    layout.weights[1] = SharedWeightView::Rows(hidden, inter / 32 * 17, shards);
    layout.weights[1].destinationOffset = 2 * inter * layout.weights[0].rowBytes;
    layout.recordBytes = layout.weights[1].destinationOffset + hidden * layout.weights[1].rowBytes;
    for (auto &view : layout.weights) view.block32Planar = planar;
    Require(layout.Prepare(), "layout rejected");
    const size_t bytes = size_t(layout.sourceBytes[0]) + layout.sourceBytes[1];
    constexpr unsigned records = 2, guard = 32;
    const size_t stride = size_t(layout.recordBytes) + 2 * guard;
    std::vector<void *> host(records * 2 * shards), mapped(host.size());
    for (unsigned i = 0; i < host.size(); ++i) {
        Check(cudaHostAlloc(&host[i], layout.weights[(i / shards) % 2].shardBytes, cudaHostAllocMapped));
        Check(cudaHostGetDevicePointer(&mapped[i], host[i], 0));
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
    Require(!UploadSharedExpertRecord(layout, host.data(), scratch, bytes - 1, actual + guard, stream),
        "short staging accepted");
    Require(!UploadSharedExpertRecord(layout, host.data(), scratch, bytes, scratch, stream),
        "aliased staging accepted");
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    // Different source records share one staging buffer. Both outputs must
    // survive, detecting an overwrite before the preceding gather finishes.
    for (unsigned record = 0; record < records; ++record)
        Require(UploadSharedExpertRecord(layout, host.data() + record * 2 * shards,
            scratch, bytes, actual + record * stride + guard, stream), "upload rejected");
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
            for (unsigned part = 0; part < 2; ++part) {
                const auto &v = layout.weights[part];
                for (unsigned row = 0; row < v.rows; ++row) {
                    const unsigned physical = part == 0 ? 2 * (row % inter) + row / inter : row;
                    const unsigned node = physical / v.rowsPerShard, local = physical % v.rowsPerShard;
                    const auto *source = static_cast<uint8_t *>(host[(record * 2 + part) * shards + node]) + size_t(local) * v.rowBytes;
                    auto *destination = expected.data() + record * layout.recordBytes + v.destinationOffset;
                    for (unsigned byte = 0; byte < v.rowBytes; ++byte) {
                        const size_t canonical = size_t(row) * v.rowBytes + byte;
                        const size_t block = canonical / 17, within = canonical % 17;
                        const size_t offset = planar ? (within < 16 ? block * 16 + within :
                            size_t(v.rows) * (v.rowBytes / 17) * 16 + block) : canonical;
                        destination[offset] = source[byte];
                    }
                }
            }
            Require(CopySharedExpertRecord(layout, devicePointers + record * 2 * shards,
                nullptr, nullptr, nullptr, legacy + record * layout.recordBytes, stream), "legacy copy rejected");
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
    }
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); cudaStreamDestroy(stream);
    cudaFree(devicePointers); cudaFree(scratch); cudaFree(actual); cudaFree(legacy);
    for (auto p : host) cudaFreeHost(p);
    std::printf("PASS shared upload hidden=%u inter=%u shards=%u bytes=%u two records, three graph replays\n",
        hidden, inter, shards, layout.recordBytes);
}

int main(int argc, char **argv) {
    const bool planar = argc > 1 && std::string(argv[1]) == "--planar";
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    try {
        for (unsigned shards : {1u, 2u, 4u}) {
            Run(128, 128, shards, planar);
            Run(512, 256, shards, planar);
            Run(5120, 2304, shards, planar);
        }
        std::puts("ALL_PASS shared expert DMA upload");
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
