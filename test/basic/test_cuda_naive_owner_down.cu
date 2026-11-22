// Standalone SM103 numerical regression. Include the private implementation so
// the original and optional dispatch share exactly the production kernels.
#include "../../src/devices/cuda/moe/fastllm-shared-flex-tp.cu"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>

namespace {
using namespace fastllm_shared_flex_tp;
using BF = __nv_bfloat16;
constexpr int Experts = 64;
constexpr size_t BoxWords = 2 * MAX_M * R * (H / 2);
constexpr int Guard = 64;
struct Arena {
    std::vector<void *> buffers;
    template<class T> T *get(size_t count) {
        auto *p = static_cast<T *>(alloc(count * sizeof(T)));
        buffers.push_back(p); check(cudaMemset(p, 0, count * sizeof(T))); return p;
    }
    ~Arena() { for (void *p : buffers) cudaFree(p); }
};
void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
__device__ unsigned randomBits(unsigned x) {
    x = x * 747796405u + 2891336453u;
    return ((x >> ((x >> 28) + 4)) ^ x) * 277803737u;
}
__global__ void weights(uint8_t *w, uint8_t *scales) {
    const size_t t = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = t; i < size_t(Experts) * H * I; i += gridDim.x * blockDim.x)
        w[i] = __nv_fp8_e4m3((int(randomBits(i) % 201) - 100) * .01f).__x;
    for (size_t i = t; i < size_t(Experts) * H * (I / 32); i += gridDim.x * blockDim.x)
        scales[i] = 123 + randomBits(i + 17) % 7;
}
__global__ void inputs(Rank a, const unsigned *seed) {
    const unsigned iteration = *seed;
    const size_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t == 0) *a.epoch = iteration;
    if (t < 256) a.gcount[t] = 7;
    for (size_t i = t; i < MAX_M * K * I; i += gridDim.x * blockDim.x)
        a.mid[i] = __nv_fp8_e4m3((int(randomBits(i + iteration * 31) % 201) - 100) * .01f).__x;
    for (size_t i = t; i < MAX_M * K * (I / 32); i += gridDim.x * blockDim.x)
        a.ms[i] = 121 + randomBits(i + iteration) % 9;
    // Emulate already published remote ranks with independent, finite values.
    // Real peer ordering and simultaneous sends are covered by the TP8 test.
    const size_t bank = size_t(iteration & 1) * MAX_M * R * (H / 2);
    for (size_t i = t; i < size_t(a.rows) * R * (H / 2); i += gridDim.x * blockDim.x) {
        const float value = (int(randomBits(i + iteration * 17) % 51) - 25) * .03125f;
        a.box[bank + i] = (uint64_t(iteration) << 32) | pack_bf16(value, -value * .5f);
    }
}
__global__ void referenceLocal(Rank a) {
    const int t = blockIdx.x * 128 + threadIdx.x, row = blockIdx.y;
    float value = 0.f;
    for (int slot = 0; slot < K; ++slot)
        value = __fadd_rn(value, a.dpartial[(row * K + slot) * H + t]);
    a.wlocal[row * H + t] = value;
}
template<class T> std::vector<T> read(const T *p, size_t n) {
    std::vector<T> result(n);
    check(cudaMemcpy(result.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return result;
}
void equal(const void *a, const void *b, size_t bytes, const char *message) {
    require(std::memcmp(a, b, bytes) == 0, message);
}
}
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess || prop.major != 10 || prop.minor != 3) return 77;
    try {
        check(cudaSetDevice(0));
        check(cudaFuncSetAttribute(owner_down, cudaFuncAttributeMaxDynamicSharedMemorySize, OwnerDownBytes));
        check(cudaFuncSetAttribute(multi_down<>, cudaFuncAttributeMaxDynamicSharedMemorySize, 36864));
        Arena arena;
        auto *w = arena.get<uint8_t>(size_t(Experts) * H * I);
        auto *scales = arena.get<uint8_t>(size_t(Experts) * H * (I / 32));
        weights<<<256, 256>>>(w, scales);
        CUtensorMap map;
        cuuint64_t dims[] = {I, H, Experts}, strides[] = {I, I * H};
        cuuint32_t box[] = {128, 128, 1}, steps[] = {1, 1, 1};
        require(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, w, dims, strides, box, steps,
            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS, "down tensor map");
        Rank ranks[2]{}; BF *out[2]; unsigned *seed = arena.get<unsigned>(1);
        int *ids = arena.get<int>(MAX_M * K);
        for (int mode = 0; mode < 2; ++mode) {
            auto &a = ranks[mode]; a.rank = 3; a.down = w; a.ds = scales;
            a.dmap = arena.get<CUtensorMap>(1);
            check(cudaMemcpy(a.dmap, &map, sizeof(map), cudaMemcpyHostToDevice));
            a.mid = arena.get<uint8_t>(MAX_M * K * I); a.ms = arena.get<uint8_t>(MAX_M * K * (I / 32));
            a.dpartial = arena.get<float>(MAX_M * K * H); a.wlocal = arena.get<float>(MAX_M * H);
            a.gcount = arena.get<int>(256); a.epoch = arena.get<uint64_t>(1);
            a.count = arena.get<int>(1); a.jobs = arena.get<int>(64); a.nuniq = arena.get<int>(1);
            a.uniq = arena.get<int>(64); a.rowcounts = arena.get<int>(64); a.members = arena.get<int>(64 * 8);
            for (int peer = 0; peer < R; ++peer) a.peers[peer] = arena.get<uint64_t>(BoxWords + 2 * Guard) + Guard;
            a.box = a.peers[a.rank]; out[mode] = arena.get<BF>(MAX_M * H + 2 * Guard) + Guard;
        }
        // Capture all row shapes before any changing-input replay.
        cudaGraphExec_t exec[5][2]{};
        for (int rows = 4; rows <= 8; ++rows) for (int mode = 0; mode < 2; ++mode) {
            auto a = ranks[mode]; a.rows = rows; cudaGraph_t graph;
            check(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            inputs<<<128, 256, 0, cudaStreamPerThread>>>(a, seed);
            launch_prepared_down(a, ids, nullptr, out[mode], cudaStreamPerThread, mode != 0);
            if (!mode) referenceLocal<<<dim3(H / 128, rows), 128, 0, cudaStreamPerThread>>>(a);
            check(cudaStreamEndCapture(cudaStreamPerThread, &graph));
            check(cudaGraphInstantiate(&exec[rows - 4][mode], graph, 0)); check(cudaGraphDestroy(graph));
        }
        unsigned iteration = 0;
        for (int rows : {8, 4, 7, 5, 6}) for (int pool : {8, rows * 4, rows * 4 + 1, rows * 8}) {
            std::vector<int> chosen(rows * K), jobs(rows * K), experts, members(64 * 8), totals(64);
            std::map<int, std::vector<int>> groups;
            for (int j = 0; j < rows * K; ++j) { chosen[j] = (j + iteration) % pool; jobs[j] = j; groups[chosen[j]].push_back(j); }
            for (auto &group : groups) {
                int n = experts.size(); experts.push_back(group.first); totals[n] = group.second.size();
                std::copy(group.second.begin(), group.second.end(), members.begin() + n * 8);
            }
            int unique = experts.size(), jobsCount = jobs.size();
            check(cudaMemcpy(ids, chosen.data(), chosen.size() * sizeof(int), cudaMemcpyHostToDevice));
            for (auto a : ranks) {
                check(cudaMemcpy(a.jobs, jobs.data(), jobs.size() * sizeof(int), cudaMemcpyHostToDevice));
                check(cudaMemcpy(a.count, &jobsCount, sizeof(int), cudaMemcpyHostToDevice));
                check(cudaMemcpy(a.nuniq, &unique, sizeof(int), cudaMemcpyHostToDevice));
                check(cudaMemcpy(a.uniq, experts.data(), experts.size() * sizeof(int), cudaMemcpyHostToDevice));
                check(cudaMemcpy(a.members, members.data(), members.size() * sizeof(int), cudaMemcpyHostToDevice));
                check(cudaMemcpy(a.rowcounts, totals.data(), totals.size() * sizeof(int), cudaMemcpyHostToDevice));
            }
            for (int repeat = 0; repeat < 4; ++repeat) {
                ++iteration; check(cudaMemcpy(seed, &iteration, sizeof(iteration), cudaMemcpyHostToDevice));
                for (int mode = 0; mode < 2; ++mode) check(cudaGraphLaunch(exec[rows - 4][mode], cudaStreamPerThread));
                check(cudaDeviceSynchronize());
                auto expected = read(out[0] - Guard, MAX_M * H + 2 * Guard);
                auto actual = read(out[1] - Guard, expected.size());
                equal(expected.data(), actual.data(), expected.size() * sizeof(BF), "output/guard mismatch");
                for (int i = 0; i < Guard; ++i) require(__bfloat162float(actual[i]) == 0.f && __bfloat162float(actual[actual.size() - 1 - i]) == 0.f, "output guard overwritten");
                for (int i = Guard; i < Guard + rows * H; ++i) require(std::isfinite(__bfloat162float(actual[i])), "nonfinite output");
                if (unique > rows * 4) {
                    auto ref = read(ranks[0].wlocal, rows * H), got = read(ranks[1].wlocal, rows * H);
                    equal(ref.data(), got.data(), ref.size() * sizeof(float), "FP32 expert sum mismatch");
                }
                for (int peer = 0; peer < R; ++peer) {
                    auto ref = read(ranks[0].peers[peer] - Guard, BoxWords + 2 * Guard);
                    auto got = read(ranks[1].peers[peer] - Guard, ref.size());
                    equal(ref.data(), got.data(), ref.size() * sizeof(uint64_t), "mailbox/guard mismatch");
                    for (int i = 0; i < Guard; ++i) require(got[i] == 0 && got[got.size() - 1 - i] == 0, "mailbox guard overwritten");
                }
                auto counters = read(ranks[1].gcount, 256);
                for (int n : counters) require(n == 0, "gate counter not reset");
            }
            printf("OWNER_DOWN rows=%d unique=%d exact=1 replays=4\n", rows, unique);
        }
        for (auto &row : exec) for (auto e : row) check(cudaGraphExecDestroy(e));
        puts("OWNER_DOWN_PASS cases=80 graph_captures=10 runtime_captures=0"); return 0;
    } catch (const std::exception &e) { fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
