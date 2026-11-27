// Exercise the production admission and cleanup paths without pretending that
// a different GPU is available. Only CUDA host capabilities/resources are mocked.
#include "devices/cuda/fastllm-bf16-ordered-dot.cuh"
#include "../../src/devices/multicuda/fastllm-attention-output-rows.cuh"
#include "../../src/devices/multicuda/fastllm-swa-verify-layer.cuh"
#include "../../src/devices/multicuda/fastllm-swa-layer.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace {
int currentDevice = 3, badDevice = -1, badArch = 103;
int cooperative = 1, multiprocessors = 148, occupancy = 4, ptx = 120, atomicPeer = 1;
int mallocCalls = 0, allocations = 0, frees = 0, memsetCalls = 0;
int failMalloc = -1, failMemset = -1;
bool failAttributes = false, failLayerShared = false, failRowsAttributes = false;
int layerOccupancy = 1, outputRouterOccupancy = 1, rowsOccupancy = 1;
int verifyFailure[8]{}, mmaFailure = 0, preparedFailure = 0, routedFailure = 0, vectorFailure = 0, directFailure = 0, rowsDirectFailure = 0, widePreparedFailure = 0;
size_t bytesAllocated = 0;
const void *launched = nullptr;
void Require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
void Env(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}
void Reset() {
    currentDevice = 3; badDevice = -1; badArch = 103;
    for (int &failure : verifyFailure) failure = 0;
    mmaFailure = preparedFailure = routedFailure = vectorFailure = directFailure = rowsDirectFailure = widePreparedFailure = 0;
    cooperative = 1; multiprocessors = 148; occupancy = 4; ptx = 120; atomicPeer = 1;
    mallocCalls = allocations = frees = memsetCalls = 0;
    failMalloc = failMemset = -1; failAttributes = false; bytesAllocated = 0; launched = nullptr;
    failLayerShared = failRowsAttributes = false; layerOccupancy = outputRouterOccupancy = rowsOccupancy = 1;
    Env("FASTLLM_NAIVE_SWA_QKV_FUSION", nullptr);
    Env("FASTLLM_NAIVE_SWA_CORE", nullptr);
    Env("FASTLLM_NAIVE_SWA_FUSION", nullptr);
}
cudaError_t GetDevice(int *id) { *id = currentDevice; return cudaSuccess; }
cudaError_t SetDevice(int id) { currentDevice = id; return cudaSuccess; }
cudaError_t Properties(cudaDeviceProp *prop, int id) {
    *prop = {};
    int arch = id == badDevice ? badArch : 103;
    prop->major = arch / 10; prop->minor = arch % 10;
    prop->cooperativeLaunch = cooperative; prop->multiProcessorCount = multiprocessors;
    return cudaSuccess;
}
bool IsRowsKernel(const void *kernel) {
    for (int rows : {2, 4, 7, 8})
        if (kernel == fastllm_swa_output::OutputRowsKernel(rows)) return true;
    return false;
}
int VerifyRows(int variant) {
    if (variant == 6) return 16;
    return variant == 1 || variant == 2 || variant >= 5 ? 64 : 8;
}
bool VerifyFp8(int variant) { return variant == 4 || variant == 5; }
int VerifyVariant(const void *kernel) {
    for (int variant = 0; variant < 8; ++variant)
        if (kernel == fastllm_swa_verify::Kernel(VerifyRows(variant), variant == 2, variant >= 3, VerifyFp8(variant))) return variant;
    return -1;
}
cudaError_t Attributes(cudaFuncAttributes *a, const void *kernel) {
    if (failAttributes || (failRowsAttributes && IsRowsKernel(kernel))) return cudaErrorInvalidDeviceFunction;
    const int variant = VerifyVariant(kernel);
    const int failure = currentDevice == 5 && variant >= 0 ? verifyFailure[variant] : 0;
    if (failure == 1 || (currentDevice == 5 && mmaFailure == 1 &&
        kernel == fastllm_swa_output::OutputRowsKernel(8, true))) return cudaErrorInvalidDeviceFunction;
    if (currentDevice == 5) {
        struct Probe { const void *kernel; int failure; };
        const Probe probes[] = {
            {(const void *)fastllm_swa_output::NormRowsRouterExperts<4, true>, preparedFailure},
            {(const void *)fastllm_swa_output::NormRowsRouterRoutes, routedFailure},
            {(const void *)fastllm_swa_output::NormRowsRouterPreparedVector<4>, vectorFailure},
            {(const void *)fastllm_swa_output::NormRowsRouterDirect<8>, directFailure},
            {(const void *)fastllm_swa_output::NormRowsRouterDirect<8, 4, false>, rowsDirectFailure},
            {(const void *)fastllm_swa_output::NormRowsRouterDirect<8, 4, true, 16>, widePreparedFailure}
        };
        for (const auto &probe : probes) if (kernel == probe.kernel) {
            if (probe.failure == 1) return cudaErrorInvalidDeviceFunction;
            *a = {}; a->ptxVersion = probe.failure == 2 ? 70 : ptx;
            a->maxThreadsPerBlock = probe.failure == 3 ? 128 : 1024; return cudaSuccess;
        }
    }
    *a = {}; a->ptxVersion = failure == 4 ? 70 : ptx;
    a->maxThreadsPerBlock = failure == 5 ? 128 : 1024; return cudaSuccess;
}
cudaError_t Occupancy(int *active, const void *kernel, int threads, size_t shared) {
    const int variant = VerifyVariant(kernel);
    if (kernel == (const void *)fastllm_swa_output::NormRowsRouterRoutes) {
        *active = currentDevice == 5 && routedFailure == 4 ? 1 : 4;
        return currentDevice == 5 && routedFailure == 5 ? cudaErrorInvalidValue : cudaSuccess;
    }
    if (variant >= 0) {
        Require(shared == fastllm_swa_verify::SharedBytes(VerifyRows(variant), variant == 2, variant >= 3, VerifyFp8(variant)), "unexpected verify occupancy size");
        *active = currentDevice == 5 && verifyFailure[variant] == 3 ? 0 : 1;
    } else if (kernel == fastllm_swa_output::OutputRowsKernel(8, true)) {
        *active = currentDevice == 5 && mmaFailure == 2 ? 0 : 1;
    } else if (IsRowsKernel(kernel)) *active = rowsOccupancy;
    else *active = shared == 131072 ? layerOccupancy : threads == 512 ? outputRouterOccupancy : occupancy;
    return cudaSuccess;
}
cudaError_t SetAttribute(const void *kernel, cudaFuncAttribute, int bytes) {
    const int variant = VerifyVariant(kernel);
    if (variant >= 0) {
        Require(bytes == fastllm_swa_verify::SharedBytes(VerifyRows(variant), variant == 2, variant >= 3, VerifyFp8(variant)), "unexpected verify shared memory");
        return currentDevice == 5 && verifyFailure[variant] == 2 ? cudaErrorInvalidValue : cudaSuccess;
    }
    const int expected = kernel == (const void *)fastllm_swa_layer::MxOutput ? 0 :
        kernel == (const void *)fastllm_swa_layer::Layer<true> ? fastllm_swa_layer::MxStagingBytes :
        fastllm_swa_layer::StagingBytes;
    Require(bytes == expected, "unexpected layer shared memory");
    return failLayerShared ? cudaErrorInvalidValue : cudaSuccess;
}
cudaError_t Peer(int *value, cudaDeviceP2PAttr, int, int) { *value = atomicPeer; return cudaSuccess; }
template<class T> cudaError_t Malloc(T **ptr, size_t bytes) {
    if (mallocCalls++ == failMalloc) return cudaErrorMemoryAllocation;
    *ptr = static_cast<T *>(std::malloc(bytes));
    if (!*ptr) return cudaErrorMemoryAllocation;
    ++allocations; bytesAllocated += bytes; return cudaSuccess;
}
cudaError_t Memset(void *ptr, int value, size_t bytes) {
    if (memsetCalls++ == failMemset) return cudaErrorInvalidValue;
    std::memset(ptr, value, bytes); return cudaSuccess;
}
cudaError_t Free(void *ptr) { if (ptr) { ++frees; std::free(ptr); } return cudaSuccess; }
cudaError_t Success() { return cudaSuccess; }
cudaError_t MockSwaLaunch(const void *kernel, dim3 grid, dim3 block, void **, size_t, cudaStream_t) {
    Require(grid.x == 128, "unexpected launch grid");
    launched = kernel;
    const int variant = VerifyVariant(kernel);
    if (variant >= 0) {
        Require(block.x == (variant >= 3 ? fastllm_swa_verify::PackedThreads(VerifyRows(variant), VerifyFp8(variant)) : fastllm_swa_verify::Threads(VerifyRows(variant))), "unexpected verify threads");
        // Stop before the real post-norm launch; only the host dispatch is mocked.
        return cudaErrorNotSupported;
    }
    Require(block.x == 256, "unexpected output threads");
    return cudaSuccess;
}
}
#undef cudaMemset
#undef cudaGetDeviceProperties
#undef cudaLaunchCooperativeKernel
#define cudaGetDevice GetDevice
#define cudaSetDevice SetDevice
#define cudaGetDeviceProperties Properties
#define cudaFuncGetAttributes Attributes
#define cudaFuncSetAttribute SetAttribute
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor Occupancy
#define cudaDeviceGetP2PAttribute Peer
#define cudaMalloc Malloc
#define cudaMemset Memset
#define cudaFree Free
#define cudaDeviceSynchronize Success
#define cudaGetLastError Success
#define cudaLaunchCooperativeKernel MockSwaLaunch
#include "../../src/devices/multicuda/fastllm-swa-output-state.cuh"
#undef cudaGetDevice
#undef cudaSetDevice
#undef cudaGetDeviceProperties
#undef cudaFuncGetAttributes
#undef cudaFuncSetAttribute
#undef cudaOccupancyMaxActiveBlocksPerMultiprocessor
#undef cudaDeviceGetP2PAttribute
#undef cudaMalloc
#undef cudaMemset
#undef cudaGetDeviceProperties
#undef cudaFree
#undef cudaDeviceSynchronize
#undef cudaGetLastError
#undef cudaLaunchCooperativeKernel

int main() {
    try {
        const std::vector<int> devices{0, 1, 2, 3, 4, 5, 6, 7};
        int checks = 0;
        auto reject = [&](bool noAllocation = true) {
            Require(!SwaOutputState::Create(devices), "unsupported group was accepted");
            Require(currentDevice == 3, "original device was not restored");
            Require(allocations == frees, "failed initialization leaked memory");
            Require(!noAllocation || mallocCalls == 0, "unsupported group allocated memory");
            ++checks;
        };
        for (int arch : {70, 75, 80, 86, 89, 90, 100, 120}) {
            Reset(); badDevice = 5; badArch = arch; reject();
        }
        Reset(); cooperative = 0; reject();
        Reset(); multiprocessors = 31; reject();
        Reset(); occupancy = 0; reject();
        Reset(); ptx = 70; reject();
        Reset(); atomicPeer = 0; reject();
        Reset(); failAttributes = true; reject();
        for (const char *value : {"0", "invalid"}) {
            Reset(); Env("FASTLLM_NAIVE_SWA_FUSION", value); reject();
        }
        for (int count : {1, 2, 4, 6}) {
            Reset();
            Require(!SwaOutputState::Create(std::vector<int>(devices.begin(), devices.begin() + count)),
                    "unsupported rank count was accepted");
            Require(mallocCalls == 0, "unsupported rank count allocated"); ++checks;
        }
        for (int i = 0; i < 16; ++i) {
            Reset(); failMalloc = i; reject(false);
            Reset(); failMemset = i; reject(false);
        }
        {
            Reset();
            auto state = SwaOutputState::Create(devices);
            Require(bool(state), "supported group was rejected");
            Require(currentDevice == 3 && allocations == 16, "initialization lifecycle failed");
            Require(bytesAllocated == 8 * 3080 * 1024, "workspace size changed");
            // The disable switch applies to new groups, not an existing graph.
            Env("FASTLLM_NAIVE_SWA_FUSION", "0");
            Require(state->Launch(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1e-6f, 0) == cudaSuccess,
                    "launch failed");
            Require(launched == (const void *)fastllm_swa_output::OutputNorm<32>,
                    "immutable backend selection failed");
            state->ReleaseAfterDrain(); state.reset();
            Require(allocations == frees && currentDevice == 3, "release leaked or changed device"); ++checks;
        }
        for (int kind = 0; kind < 7; ++kind) {
            Reset();
            if (kind == 1) failLayerShared = true;
            if (kind == 2) layerOccupancy = 0;
            if (kind == 3) multiprocessors = 32;
            if (kind == 4) Env("FASTLLM_NAIVE_SWA_QKV_FUSION", "0");
            if (kind == 5) Env("FASTLLM_NAIVE_SWA_CORE", "0");
            if (kind == 6) Env("FASTLLM_NAIVE_SWA_CORE", "invalid");
            auto state = SwaOutputState::Create(devices);
            Require(bool(state), "optional layer failure disabled the output backend");
            Require(state->LayerSupported() == (kind == 0), "layer capability dispatch mismatch");
            if (kind == 1 || kind == 2)
                Require(state->VerifySupported(8) && state->VerifySupported(16),
                        "decode-layer capability failure disabled verification");
            if (kind >= 4)
                Require(!state->VerifySupported(8) && !state->VerifySupported(16), "disabled verification accepted");
            Require(bytesAllocated == 8 * 3080 * 1024, "layer allocated an extra mailbox");
            state->ReleaseAfterDrain(); state.reset();
            Require(allocations == frees && currentDevice == 3, "layer cleanup failed"); ++checks;
        }
        for (bool supported : {false, true}) {
            Reset(); outputRouterOccupancy = supported ? 1 : 0;
            auto state = SwaOutputState::Create(devices);
            Require(bool(state), "optional norm-router failure disabled original output");
            Require(state->OutputRouterSupported() == supported, "norm-router capability mismatch");
            Require(bytesAllocated == 8 * 3080 * 1024, "norm-router allocated extra mailbox");
            state->ReleaseAfterDrain(); state.reset();
            Require(allocations == frees && currentDevice == 3, "norm-router cleanup failed"); ++checks;
        }
        for (int kind = 0; kind < 3; ++kind) {
            Reset();
            if (kind == 1) rowsOccupancy = 0;
            if (kind == 2) failRowsAttributes = true;
            auto state = SwaOutputState::Create(devices);
            Require(bool(state) && state->LayerSupported() && state->OutputRouterSupported(),
                    "optional rows failure disabled established backends");
            Require(state->RowsSupported() == (kind == 0), "multi-row capability mismatch");
            Require(bytesAllocated == 8 * 3080 * 1024, "multi-row workspace size mismatch");
            state->ReleaseAfterDrain(); state.reset();
            Require(allocations == frees && currentDevice == 3, "multi-row cleanup failed"); ++checks;
        }
        for (int variant = 0; variant < 8; ++variant) for (int failure = 0; failure <= 5; ++failure) {
            Reset(); verifyFailure[variant] = failure;
            auto state = SwaOutputState::Create(devices);
            Require(state && state->LayerSupported() && state->RowsSupported() && state->RowsMmaSupported(),
                    "optional verification failure disabled established backends");
            for (int rows : {1, 8, 9, 16, 36, 64})
                Require(state->VerifySupported(rows) == (rows > 8 || failure == 0 || variant != 0),
                        "verification variant capability mismatch");
            for (bool fp8 : {false, true}) for (int rows : {1, 2, 8, 9, 16, 31, 32, 33, 64, 65}) {
                const int path = fp8 ? (rows <= 8 ? 4 : 5) : (rows <= 8 ? 3 : rows < 32 ? 6 : 7);
                Require(state->VerifySupported(rows, true, fp8) ==
                        (rows >= 2 && rows <= 64 && (variant != path || failure == 0)),
                        "MXFP8 verification capability mismatch");
            }
            Require(!state->VerifySupported(0) && !state->VerifySupported(65), "unsupported row count accepted");
            fastllm_swa_verify::Args args{}; args.rows = 16;
            Require(state->LaunchVerify(args, 0, nullptr, nullptr, nullptr) == cudaErrorNotSupported,
                    "mock verification launch failed");
            Require(launched == fastllm_swa_verify::Kernel(16, variant != 2 || failure == 0),
                    "wide verification did not fall back group-wide");
            Require(bytesAllocated == 8 * 3080 * 1024, "verification allocated extra mailbox");
            state->ReleaseAfterDrain(); state.reset();
            Require(allocations == frees && currentDevice == 3, "verification cleanup failed"); ++checks;
        }
        for (int failure = 1; failure <= 5; ++failure) {
            Reset(); verifyFailure[1] = verifyFailure[2] = failure;
            auto state = SwaOutputState::Create(devices);
            Require(state && state->VerifySupported(8) && !state->VerifySupported(16),
                    "both large-row schedules unavailable but admitted");
            Require(bytesAllocated == 8 * 3080 * 1024, "wide verification enlarged workspace");
            state->ReleaseAfterDrain(); state.reset();
            Require(allocations == frees, "wide verification fallback leaked"); ++checks;
        }
        for (int failure = 0; failure <= 2; ++failure) {
            Reset(); mmaFailure = failure;
            auto state = SwaOutputState::Create(devices);
            Require(state && state->LayerSupported() && state->RowsSupported() && state->VerifySupported(8),
                    "optional MMA projection failure disabled other backends");
            Require(state->RowsMmaSupported() == (failure == 0), "MMA projection capability mismatch");
            Require(bytesAllocated == 8 * 3080 * 1024, "MMA projection allocated extra mailbox");
            state->ReleaseAfterDrain(); state.reset();
            Require(allocations == frees && currentDevice == 3, "MMA projection cleanup failed"); ++checks;
        }
        for (int failure = 0; failure <= 3; ++failure) {
            Reset(); preparedFailure = failure;
            auto state = SwaOutputState::Create(devices);
            Require(state && state->RowsSupported() && state->RowsMmaSupported() && state->VerifySupported(8),
                    "optional prepared router failure disabled existing backends");
            Require(state->PreparedRouterSupported() == (failure == 0), "prepared router group capability mismatch");
            Require(bytesAllocated == 8 * 3080 * 1024, "prepared router allocated extra workspace");
            state->ReleaseAfterDrain(); state.reset();
            Require(allocations == frees && currentDevice == 3, "prepared router cleanup failed"); ++checks;
        }
        for (int direct : {0, 1}) for (int failure = 0; failure <= 3; ++failure) {
            Reset();
            (direct ? directFailure : vectorFailure) = failure;
            auto state = SwaOutputState::Create(devices);
            Require(state && state->PreparedRouterSupported(true) && state->VerifySupported(8),
                    "vector router failure disabled established backends");
            Require(state->PreparedVectorRouterSupported() == (failure == 0), "vector group capability mismatch");
            Require(bytesAllocated == 8 * 3080 * 1024, "vector router allocated extra workspace");
            state->ReleaseAfterDrain(); state.reset();
            Require(allocations == frees && currentDevice == 3, "vector router cleanup failed"); ++checks;
        }
        for (int failure = 0; failure <= 3; ++failure) {
            Reset(); rowsDirectFailure = failure;
            auto state = SwaOutputState::Create(devices);
            Require(state && state->PreparedVectorRouterSupported() && state->PreparedRouterSupported(true),
                    "four-row router failure disabled prepared paths");
            alignas(16) char buffers[4][32]{};
            for (int rows : {0, 1, 8, 9, 13, 16, 17, 32, 64})
                Require(state->UseRowsDirectRouter(buffers[0], buffers[1], buffers[2], buffers[3], rows)
                        == (failure == 0 && rows >= 9 && rows <= 16), "four-row router shape/capability mismatch");
            for (int field = 0; field < 4; ++field) {
                const void *ptr[4] = {buffers[0], buffers[1], buffers[2], buffers[3]};
                ptr[field] = buffers[field] + 2;
                Require(!state->UseRowsDirectRouter(ptr[0], ptr[1], ptr[2], ptr[3], 16),
                        "four-row router accepted misaligned vectors");
            }
            Require(bytesAllocated == 8 * 3080 * 1024, "four-row router allocated extra workspace");
            state->ReleaseAfterDrain(); state.reset();
            Require(allocations == frees && currentDevice == 3, "four-row router cleanup failed"); ++checks;
        }
        for (int failure = 0; failure <= 3; ++failure) {
            Reset(); widePreparedFailure = failure;
            auto state = SwaOutputState::Create(devices);
            Require(state && state->PreparedRouterSupported(true) && state->PreparedVectorRouterSupported(),
                    "wide preparation failure disabled narrow paths");
            for (int rows : {9, 13, 16}) {
                Require(state->PreparedRouterSupported(false, rows) == (failure == 0), "wide prepared capability mismatch");
                Require(!state->PreparedRouterSupported(true, rows), "wide preselected layout admitted");
            }
            Require(!state->PreparedRouterSupported(false,17), "oversized prepared tile admitted");
            Require(bytesAllocated == 8 * 3080 * 1024, "wide preparation grew workspace");
            state->ReleaseAfterDrain();state.reset();
            Require(allocations == frees && currentDevice == 3, "wide preparation cleanup failed");++checks;
        }
        for (int failure = 0; failure <= 5; ++failure) {
            Reset(); routedFailure = failure;
            auto state = SwaOutputState::Create(devices);
            Require(state && state->PreparedRouterSupported() && state->VerifySupported(8),
                    "cooperative routing failure disabled established backends");
            Require(state->PreparedRouterSupported(true) == (failure == 0),
                    "cooperative routing capability mismatch");
            Require(bytesAllocated == 8 * 3080 * 1024, "routing allocated extra workspace");
            state->ReleaseAfterDrain(); state.reset();
            Require(allocations == frees && currentDevice == 3, "routing cleanup failed"); ++checks;
        }
        {
            alignas(16) unsigned char storage[8 * 4096 + 8 * 128 + 256 * sizeof(int) + sizeof(uint64_t)]{};
            FastllmMoePreparedInput input{storage, storage + 8 * 4096,
                (int *)(storage + 8 * 4096 + 8 * 128),
                (uint64_t *)(storage + sizeof(storage) - sizeof(uint64_t)), 8};
            Require(input.Valid(8) && !input.Valid(7) && !input.Valid(9), "prepared shape admission failed");
            Require(!input.Disjoint(storage + 100, 16), "prepared source alias accepted");
            auto bad = input; bad.scales = bad.values + 16;
            Require(!bad.Valid(8), "prepared buffers overlap");
            bad = input; bad.values += 1;
            Require(!bad.Valid(8), "prepared misalignment accepted");
            bad = input; bad.epoch = (uint64_t *)(UINTPTR_MAX - 7);
            Require(!bad.Valid(8), "prepared address overflow accepted"); ++checks;
            alignas(16) int indices[64], jobs[64], count, unique, experts[64], members[512], rowCounts[64];
            alignas(16) float scores[64], bias[256];
            input.routed = true; input.indices = indices; input.scores = scores;
            input.jobs = jobs; input.count = &count; input.unique = &unique;
            input.experts = experts; input.members = members; input.rowCounts = rowCounts; input.bias = bias;
            Require(input.Valid(8), "valid routing descriptor rejected"); ++checks;
            for (int field = 0; field < 9; ++field) {
                bad = input;
                switch (field) {
                case 0: bad.indices = nullptr; break;
                case 1: bad.scores = (float *)indices; break;
                case 2: bad.jobs = (int *)input.values; break;
                case 3: bad.count = (int *)(uintptr_t(&count) + 1); break;
                case 4: bad.unique = (int *)(UINTPTR_MAX - 3); break;
                case 5: bad.experts = jobs; break;
                case 6: bad.members = nullptr; break;
                case 7: bad.rowCounts = nullptr; break;
                case 8: bad.bias = scores; break;
                }
                Require(!bad.Valid(8), "invalid routing descriptor accepted"); ++checks;
            }
            bad = input; bad.routeScale = INFINITY;
            Require(!bad.Valid(8), "infinite routing scale accepted"); ++checks;
            bad = input; bad.bias = nullptr;
            Require(bad.Valid(8), "optional routing bias rejected"); ++checks;
            alignas(16) uint8_t coarse[FastllmMoePreparedInput::Block128Bytes];
            input.block128=coarse;
            Require(input.Valid(8), "valid block128 rejected"); ++checks;
            Require(!input.Disjoint(coarse+100,16), "block128 source alias accepted"); ++checks;
            for (int rows : {2, 3, 4, 5, 6, 7, 8}) {
                bad = input; bad.rows = rows;
                Require(bad.Valid(rows) == (rows >= 4), "block128 padded row admission"); ++checks;
            }
            bad=input;bad.block128+=1;Require(!bad.Valid(8), "block128 alignment accepted"); ++checks;
            bad=input;bad.block128=(uint8_t*)(UINTPTR_MAX-15);Require(!bad.Valid(8), "block128 overflow accepted"); ++checks;
            FastllmMoePreparedInput::Range base[4];input.Ranges(base);
            for(auto range:base){bad=input;bad.block128=(uint8_t*)(uintptr_t(range.data)&~uintptr_t(15));
                Require(!bad.Valid(8), "block128 workspace overlap accepted"); ++checks;}
            FastllmMoePreparedInput::Range routes[9];input.RoutingRanges(routes);
            for(auto range:routes){bad=input;bad.block128=(uint8_t*)(uintptr_t(range.data)&~uintptr_t(15));
                Require(!bad.Valid(8), "block128 route overlap accepted"); ++checks;}
            alignas(16) uint8_t wideImage[2*FastllmMoePreparedInput::Block128Bytes];
            FastllmMoePreparedInput wide{nullptr,nullptr,input.counters,input.epoch,16};wide.block128=wideImage;
            Require(wide.Valid(16), "wide descriptor rejected");++checks;
            Require(!wide.Disjoint(wideImage+sizeof(wideImage)-16,16), "wide suffix alias admitted");++checks;
            bad=wide;bad.values=storage;Require(!bad.Valid(16), "wide descriptor permits old xq buffer");++checks;
            bad=wide;bad.scales=storage;Require(!bad.Valid(16), "wide descriptor permits old scale buffer");++checks;
            bad=wide;bad.block128=nullptr;Require(!bad.Valid(16), "wide image missing");++checks;
            bad=wide;bad.block128=(uint8_t*)input.counters;Require(!bad.Valid(16), "wide counter alias admitted");++checks;
            bad=wide;bad.routed=true;Require(!bad.Valid(16), "wide routing metadata admitted");++checks;
            bad=wide;bad.rows=17;Require(!bad.Valid(17), "wide overflow shape admitted");++checks;
            FastllmMoePreparedInput::Range routing[9]; input.RoutingRanges(routing);
            for (auto range : routing) {
                Require(!input.Disjoint(range.data, range.bytes), "routing alias accepted"); ++checks;
            }

        }
        std::printf("SWA admission PASS: %d capability/rollback/lifecycle cases\n", checks);
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "SWA admission FAIL: %s\n", e.what()); return 1;
    }
}
