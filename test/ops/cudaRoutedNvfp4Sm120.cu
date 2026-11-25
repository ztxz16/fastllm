// Compare the routed SM120 block-scaled MMA with an independent FP64 dot.
// Include shuffled routes, signed block scales and independently scaled experts.
#include "../../src/models/inkling2/runtime.cuh"
#include "../../src/devices/cuda/moe/fastllm-nvfp4-routed-sm120.cuh"
#ifdef FASTLLM_INKLING_GROUPED_SM120
#include "../../src/devices/cuda/moe/fastllm-grouped-sm120.cuh"
#endif
#include <cstring>
#include <cstdio>
#include <random>

namespace {
namespace op = fastllm::cuda::routed_fp4;
using fastllm::inkling2::Buffer;
using fastllm::inkling2::check;

template <class T> Buffer<T> copy(const std::vector<T> &v) {
    Buffer<T> d(v.size());
    check(cudaMemcpy(d.p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
double fp4(unsigned packed, int pos) {
    const double levels[] = {0, .5, 1, 1.5, 2, 3, 4, 6};
    unsigned v = (packed >> (4 * pos)) & 15;
    return (v & 8 ? -1 : 1) * levels[v & 7];
}
double scale(unsigned byte) {
    int a = byte & 127, exp = a / 8;
    double v = exp ? std::ldexp(1. + (a % 8) / 8., exp - 7) : a / 512.;
    return byte & 128 ? -v : v;
}
template <int TileRows> void test(int tokens, bool up, bool skew, bool verify = false, bool grouped = false) {
    const int experts = verify ? 97 : 6, top = verify ? 8 : 3, n = grouped ? 256 : 48;
    const int k = grouped ? 320 : 192, routes = tokens * top, div = up ? top : 1;
    const int capacity = (routes + (TileRows - 1) * std::min(experts, routes)) / TileRows;
    std::mt19937 rng(107);
    std::vector<uint32_t> w(size_t(experts) * n * k / 8);
    std::vector<unsigned char> sf(size_t(experts) * n * k / 16);
    std::vector<float> global(experts);
    std::vector<op::Block16> x(size_t(routes / div) * k / 16);
    std::vector<int> ids(routes);
    for (auto &i : w)
        i = rng();
    for (auto &i : sf)
        i = (rng() % 82) | ((rng() % 2) * 128);
    for (int i = 0; i < experts; ++i)
        global[i] = float(i) / 7.f;
    for (size_t i = 0; i < x.size(); ++i) {
        auto &b = x[i];
        b.code[0] = rng();
        b.code[1] = rng();
        b.scale = rng() % 82;
        b.outer = float((i / (k / 16) + 1) % 9) / 7.f;
    }
    for (int i = 0; i < routes; ++i)
        ids[i] = skew ? (verify ? i % top : 4) : (i / top * 5 + i % top * 2) % experts;
    auto wi = copy(w);
    auto si = copy(sf);
    auto gi = copy(global);
    auto xi = copy(x);
    auto di = copy(ids);
    Buffer<uint4> codes(w.size() / 4);
    Buffer<uint32_t> scales(sf.size() / 4);
    Buffer<op::RouteGroup<TileRows>> plan(capacity);
    Buffer<op::PlanSize> used(1);
    Buffer<op::InputTile> packed(size_t(capacity) * (TileRows / 8) * (k / 64));
    Buffer<__nv_bfloat16> y(size_t(routes) * n);
    const size_t words = size_t(n) * k / 8;
    for (int e = 0; e < experts; ++e) {
        op::pack<<<(words / 4 + 255) / 256, 256>>>(wi.p + e * words,
                                                   reinterpret_cast<uint32_t *>(si.p) + e * words / 8,
                                                   codes.p + e * words / 4, scales.p + e * words / 8, n, k, up);
    }
    op::Weights view{codes.p, scales.p, gi.p};
    for (int pass = 0; pass < 2; ++pass) {
        std::reverse(ids.begin(), ids.end());
        check(cudaMemcpy(di.p, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice));
        check(cudaMemset(y.p, 0xff, size_t(routes) * n * sizeof(__nv_bfloat16)));
        std::vector<int> outputRows(routes);
        std::vector<__nv_bfloat16> got(size_t(routes) * n);
        for (int i = 0; i < routes; ++i) outputRows[i] = i;
        if (verify) {
            Buffer<__nv_bfloat16> ref(size_t(routes) * n);
            op::decodeProject<4><<<dim3(n / 16, routes), 128>>>(view, xi.p, di.p, ref.p, n, k, div);
#ifdef FASTLLM_INKLING_GROUPED_SM120
            if (grouped) {
                int device, smCount;
                check(cudaGetDevice(&device));
                check(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
                op::GroupedWeightMaps maps;
                if (!op::makeGroupedWeightMaps(maps, view, experts, n, k)) throw std::runtime_error("Invalid weight maps");
                auto scratch = op::groupedScratch(experts, smCount);
                Buffer<int> offsets(experts + 1), ranks(routes);
                Buffer<uint2> values(size_t(routes) * k / 16);
                Buffer<uint8_t> scales(fastllm::cuda::grouped_sm120::scaleRows(routes, experts) * k / 16);
                Buffer<float> sorted(size_t(routes) * n);
                Buffer<uint8_t> metadata(scratch.metadata), gemm(scratch.gemm);
                op::GroupedBuffers buffers{values.p, scales.p, sorted.p, metadata.p, gemm.p};
                check(op::planGroupedRows(di.p, offsets.p, ranks.p, experts, routes, cudaStreamPerThread));
                check(op::packGroupedInput(xi.p, di.p, offsets.p, ranks.p, buffers, routes, k, div, cudaStreamPerThread));
                check(op::projectGroupedRows(view, maps, offsets.p, buffers, n, k, experts, smCount, cudaStreamPerThread));

                std::vector<float> outer(routes), dots(got.size()), sums(size_t(tokens) * n);
                std::vector<int> starts(experts + 1), rank(routes);
                check(cudaMemcpy(dots.data(), sorted.p, dots.size() * sizeof(float), cudaMemcpyDeviceToHost));
                check(cudaMemcpy(starts.data(), offsets.p, starts.size() * sizeof(int), cudaMemcpyDeviceToHost));
                check(cudaMemcpy(rank.data(), ranks.p, rank.size() * sizeof(int), cudaMemcpyDeviceToHost));
                for (int i = 0; i < routes; ++i) outer[i] = x[size_t(i / div) * (k / 16)].outer;
                auto rowScales = copy(outer);
                Buffer<float> result(sums.size());
                check(op::sumGroupedRows(sorted.p, result.p, di.p, offsets.p, ranks.p, gi.p, rowScales.p,
                                         tokens, n, top, cudaStreamPerThread));
                check(cudaMemcpy(sums.data(), result.p, sums.size() * sizeof(float), cudaMemcpyDeviceToHost));
                Buffer<__nv_bfloat16> reduced(sums.size());
                check(op::sumGroupedRows(sorted.p, reduced.p, di.p, offsets.p, ranks.p, gi.p, rowScales.p,
                                         tokens, n, top, cudaStreamPerThread));
                std::vector<__nv_bfloat16> rounded(sums.size());
                check(cudaMemcpy(rounded.data(), reduced.p, sums.size() * sizeof(rounded[0]), cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < sums.size(); ++i)
                    if (float(rounded[i]) != float(__float2bfloat16_rn(sums[i])))
                        throw std::runtime_error("Incorrect grouped BF16 top-k sum");
                for (int i = 0; i < routes; ++i) {
                    int expert = ids[i], row = starts[expert] + rank[i];
                    for (int j = 0; j < n; ++j)
                        got[i * n + j] = __float2bfloat16_rn(dots[row * n + j] * global[expert] * outer[i]);
                }
                for (int i = 0; i < tokens; ++i)
                    for (int j = 0; j < n; ++j) {
                        float sum = 0;
                        for (int t = 0; t < top; ++t) sum += float(got[(i * top + t) * n + j]);
                        if (sum != sums[i * n + j]) throw std::runtime_error("Incorrect grouped top-k sum");
                    }
            } else
#endif
            {
                const int tile = op::decodeRouteTileRows(tokens, top);
                if (!tile) throw std::runtime_error("Expected a multi-row decode tile");
                op::decodeProject<4, true><<<dim3(n / 16, tile * top, (tokens + tile - 1) / tile), 128>>>(
                    view, xi.p, di.p, y.p, n, k, div, routes, tile * top);
            }
            std::vector<__nv_bfloat16> expected(got.size());
            const size_t bytes = expected.size() * sizeof(__nv_bfloat16);
            check(cudaMemcpy(expected.data(), ref.p, bytes, cudaMemcpyDeviceToHost));
            if (!grouped) check(cudaMemcpy(got.data(), y.p, bytes, cudaMemcpyDeviceToHost));
            if (std::memcmp(got.data(), expected.data(), bytes))
                throw std::runtime_error("Grouped decode differs from the single-route projection");
        } else if (tokens == 1) {
            op::decodeProject<4><<<dim3(n / 16, routes), 128>>>(view, xi.p, di.p, y.p, n, k, div);
        } else if constexpr (TileRows == 16) {
            // Padded groups from a different gate backend. The down input and its
            // scale stay in original route order, even for empty or partial tiles.
            std::vector<int> counts(experts), offsets(experts + 1), tileExperts, sorted;
            for (int e = 0; e < experts; ++e) {
                for (int r = 0; r < routes; ++r) if (ids[r] == e) sorted.push_back(r);
                counts[e] = int(sorted.size()) - offsets[e];
                const int tiles = (counts[e] + TileRows - 1) / TileRows;
                offsets[e + 1] = offsets[e] + tiles * TileRows;
                sorted.resize(offsets[e + 1], -1);
                tileExperts.insert(tileExperts.end(), tiles, e);
            }
            auto count = copy(counts), offset = copy(offsets), expert = copy(tileExperts), route = copy(sorted);
            const op::PaddedRouteTiles<TileRows> padded{count.p, offset.p, expert.p, route.p, experts};
            op::packInput<<<dim3((k / 64 + 7) / 8, capacity, TileRows / 8), 256>>>(
                padded, xi.p, packed.p, k, 1);
            op::groupedProject<4, 2, false><<<dim3((n + 127) / 128, capacity), 128>>>(
                view, xi.p, nullptr, packed.p, padded, y.p, n, k, 1);
            Buffer<__nv_bfloat16> ref(size_t(routes) * n);
            op::decodeProject<4><<<dim3(n / 16, routes), 128>>>(view, xi.p, di.p, ref.p, n, k, 1);
            std::vector<__nv_bfloat16> expected(got.size());
            const size_t bytes = got.size() * sizeof(got[0]);
            check(cudaMemcpy(expected.data(), ref.p, bytes, cudaMemcpyDeviceToHost));
            check(cudaMemcpy(got.data(), y.p, bytes, cudaMemcpyDeviceToHost));
            if (std::memcmp(expected.data(), got.data(), bytes))
                throw std::runtime_error("Padded expert down differs from the single-route projection");
        } else {
            check(cudaMemsetAsync(used.p, 0, sizeof(op::PlanSize), cudaStreamPerThread));
            op::groupRoutes<<<experts, 256>>>(di.p, routes, plan.p, used.p);
            const op::CompactRouteTiles<TileRows> compact{plan.p, used.p};
            op::packInput<<<dim3((k / 64 + 7) / 8, capacity, TileRows / 8), 256>>>(compact, xi.p,
                                                                                        packed.p, k, div);
            op::PlanSize size;
            check(cudaMemcpy(&size, used.p, sizeof(size), cudaMemcpyDeviceToHost));
            if (size.rows != routes || size.tiles > capacity) throw std::runtime_error("Invalid expert plan");
            std::vector<op::RouteGroup<TileRows>> tasks(size.tiles);
            check(cudaMemcpy(tasks.data(), plan.p, tasks.size() * sizeof(tasks[0]), cudaMemcpyDeviceToHost));
            std::vector<float> rowScales(routes);
            for (const auto &task : tasks)
                for (int j = 0; j < task.count; ++j) {
                    const int id = task.routes[j], row = task.first + j;
                    if (up) outputRows[id] = row;
                    rowScales[row] = x[size_t(id / div) * (k / 16)].outer;
                }
            auto scalesByRow = copy(rowScales);
            constexpr int columns = TileRows == 32 ? 1 : 2, warps = 8 / columns;
            constexpr int threads = warps * 32 * ((TileRows + op::WarpTileRows - 1) / op::WarpTileRows);
            dim3 grid((n + 127) / 128, capacity);
            if (up)
                op::groupedProject<warps, columns, true><<<grid, threads>>>(
                    view, xi.p, nullptr, packed.p, compact, y.p, n, k, div);
            else
                op::groupedProject<warps, columns, false><<<grid, threads>>>(
                    view, nullptr, scalesByRow.p, packed.p, compact, y.p, n, k, div);
        }
        if (!verify) check(cudaMemcpy(got.data(), y.p, got.size() * sizeof(got[0]), cudaMemcpyDeviceToHost));
        double error = 0, norm = 0;
        for (int i = 0; i < routes; ++i)
            for (int j = 0; j < n; ++j) {
                const int expertRow = ids[i] * n + (up ? 2 * (j % (n / 2)) + j / (n / 2) : j);
                double ref = 0;
                for (int c = 0; c < k; ++c) {
                    const auto &b = x[(i / div) * (k / 16) + c / 16];
                    ref += fp4(b.code[(c % 16) / 8], c % 8) * scale(b.scale) *
                           fp4(w[expertRow * (k / 8) + c / 8], c % 8) * scale(sf[expertRow * (k / 16) + c / 16]);
                }
                ref *= global[ids[i]] * double(x[(i / div) * (k / 16)].outer);
                const double diff = float(got[outputRows[i] * n + j]) - ref;
                error += diff * diff;
                norm += ref * ref;
                // The wider grouped cases also check every BF16 bit against the
                // decode kernel above. Use a norm for their independent FP64
                // check because near-cancelling large terms can leave tiny dots.
                if (!std::isfinite(diff) || (!grouped && std::abs(diff) > std::abs(ref) * .004 + 1e-4))
                    throw std::runtime_error("Incorrect routed NVFP4 output");
            }
        if (grouped && error > .0025 * .0025 * norm)
            throw std::runtime_error("Grouped NVFP4 error exceeds the FP64 reference tolerance");
        printf("tokens=%d gate=%d skew=%d pass=%d relL2=%.6g\n", tokens, up, skew, pass,
               std::sqrt(error / std::max(norm, 1e-30)));
    }
}

// Exactly representable FP4 activations exercise both row mappings, tail
// padding and the no-shared-cache path. Every block has magnitude 6 or 3.
struct KnownActivation {
    __host__ __device__ static unsigned code(int row, int route, int col) {
        return ((col * 5 + route + row) & 7) | ((col + route) & 8);
    }
    __host__ __device__ static float gain(int route) { return float(route % 4); }
    __host__ __device__ float get(int row, int route, int col, int) const {
        const float levels[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
        const unsigned q = code(row, route, col);
        return (q & 8 ? -1.f : 1.f) * levels[q & 7] * gain(route) * (col & 16 ? .5f : 1.f);
    }
    __device__ void loadBlock(int row, int route, int col, int width, float (&v)[16]) const {
        for (int i = 0; i < 16; ++i) v[i] = get(row, route, col + i, width);
    }
};
void checkBand(const op::InputTile &tile, int laneRow, int row, int route, int col) {
    const int block = col % 64 / 16, half = col % 16 / 8;
    const uint2 pair = tile.words[laneRow * 4 + (block % 2) * 2 + half];
    const unsigned word = block < 2 ? pair.x : pair.y;
    const unsigned q = (word >> ((col % 8) * 4)) & 15;
    const unsigned scale = (tile.scales[laneRow] >> (block * 8)) & 255;
    const bool nonzero = KnownActivation::gain(route) != 0;
    if (q != (nonzero ? KnownActivation::code(row, route, col) : 0) ||
        scale != (nonzero ? (block & 1 ? 0x70 : 0x78) : 0))
        throw std::runtime_error("Incorrect packed activation codes or scales");
}

template <int T, bool Local = false> void testScatter(int width) {
    // Include empty experts, several tiles in the busiest expert, and padding.
    const int tokens = 97, experts = 43, top = 3, routes = tokens * top;
    const int capacity = (routes + (T - 1) * experts) / T;
    std::vector<int> ids(routes), ranks(routes), counts(experts);
    std::vector<float> data(size_t(tokens) * width);
    for (int r = routes - 1; r >= 0; --r) {
        ids[r] = (r % top) == 0 ? 42 : (r * 13) % 37;
        if (Local && (r / top % 5 == 0 || r % 4 == 0)) {
            ids[r] = -1;
            ranks[r] = -1;
        } else {
            ranks[r] = counts[ids[r]]++;
        }
    }
    for (int row = 0; row < tokens; ++row)
        for (int col = 0; col < width; ++col)
            data[size_t(row) * width + col] = KnownActivation{}.get(row, row, col, width);
    auto x = copy(data);
    auto index = copy(ids), rank = copy(ranks), count = copy(counts);
    Buffer<int2> starts(experts);
    Buffer<op::PlanSize> used(1);
    Buffer<op::RouteGroup<T>> tasks(capacity);
    std::vector<op::InputTile> tiles(size_t(capacity) * (T / 8) * (width / 64));
    Buffer<op::InputTile> packed(tiles.size());
    auto scales = copy(std::vector<float>(tokens, -1.f));
    op::expertOffsets<T><<<1, 32>>>(count.p, starts.p, experts, used.p);
    op::scatterInput<T, Local><<<tokens, op::QuantThreads>>>(x.p, width, index.p, rank.p, count.p, starts.p, top,
                                                  tasks.p, packed.p, scales.p);
    op::PlanSize size;
    check(cudaMemcpy(&size, used.p, sizeof(size), cudaMemcpyDeviceToHost));
    int expected = 0;
    for (int n : counts) expected += (n + T - 1) / T;
    const int valid = routes - std::count(ids.begin(), ids.end(), -1);
    if (size.tiles != expected || size.rows != valid) throw std::runtime_error("Incorrect expert prefixes");
    std::vector<float> outer(tokens);
    std::vector<op::RouteGroup<T>> plan(size.tiles);
    check(cudaMemcpy(plan.data(), tasks.p, plan.size() * sizeof(plan[0]), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(outer.data(), scales.p, tokens * sizeof(float), cudaMemcpyDeviceToHost));
    if (Local)
        for (int row = 0; row < tokens; row += 5)
            if (outer[row] != -1.f) throw std::runtime_error("Quantized a token with no local routes");
    check(cudaMemcpy(tiles.data(), packed.p, tiles.size() * sizeof(tiles[0]), cudaMemcpyDeviceToHost));
    int tile = 0, first = 0;
    for (int e = 0; e < experts; ++e) {
        for (int start = 0; start < counts[e]; start += T, ++tile) {
            const auto &task = plan[tile];
            if (task.expert != e || task.first != first + start || task.count != std::min(T, counts[e] - start))
                throw std::runtime_error("Incorrect expert tile header");
            for (int local = 0; local < (task.count + 7) / 8 * 8; ++local) {
                const int i = std::min(task.count - 1, local), r = task.routes[i];
                if (r < 0 || r >= routes || ids[r] != e || ranks[r] != start + i)
                    throw std::runtime_error("Incorrect route slot");
                const int row = r / top;
                if (outer[row] != KnownActivation::gain(row) / 256.f) throw std::runtime_error("Incorrect token scale");
                for (int col = 0; col < width; ++col) {
                    const size_t offset = (size_t(tile) * (width / 64) + col / 64) * (T / 8) + local / 8;
                    checkBand(tiles[offset], local % 8, row, row, col);
                }
            }
        }
        first += counts[e];
    }
}
template <int T> void testPackedActivation(int width, bool useRegisters = false) {
    constexpr int count = T - 3;
    op::RouteGroup<T> plan{};
    plan.count = count;
    for (int i = 0; i < count; ++i) plan.routes[i] = count - i - 1;
    auto tasks = copy(std::vector<op::RouteGroup<T>>{plan});
    auto used = copy(std::vector<op::PlanSize>{{1, count}});
    const size_t tiles = size_t(T / 8) * (width / 64);
    Buffer<op::InputTile> packed(tiles);
    Buffer<float> scales(count);
    if (useRegisters) {
        op::activateBlocks<T><<<T, (width / 16 + 31) / 32 * 32>>>(KnownActivation{}, tasks.p, used.p,
                                                                   width, packed.p, scales.p);
    } else {
        op::launchPackedActivation(KnownActivation{}, tasks.p, used.p, 1, width, packed.p, scales.p);
    }
    std::vector<op::InputTile> out(tiles);
    std::vector<float> outer(count);
    check(cudaMemcpy(out.data(), packed.p, tiles * sizeof(out[0]), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(outer.data(), scales.p, count * sizeof(float), cudaMemcpyDeviceToHost));
    for (int row = 0; row < count; ++row) {
        const int route = plan.routes[row];
        const float gain = KnownActivation::gain(route);
        if (outer[row] != gain / 256.f) throw std::runtime_error("Incorrect packed row scale");
        for (int col = 0; col < width; ++col)
            checkBand(out[(col / 64) * (T / 8) + row / 8], row % 8, row, route, col);
    }
}
#ifdef FASTLLM_INKLING_GROUPED_SM120
// Remote routes must occupy no rows, even when every route in a tile is remote.
void testLocalRoutePlan(int routes, bool allRemote) {
    constexpr int experts = 64;
    std::vector<int> ids(routes, -1), expected(experts + 1), offsets(experts + 1), ranks(routes);
    for (int i = 0; !allRemote && i < routes; i += 3) {
        ids[i] = i % experts;
        ++expected[ids[i] + 1];
    }
    for (int e = 0; e < experts; ++e) expected[e + 1] += expected[e];
    auto deviceIds = copy(ids);
    Buffer<int> starts(experts + 1), localRanks(routes);
    check(op::planGroupedRows(deviceIds.p, starts.p, localRanks.p, experts, routes, cudaStreamPerThread));
    check(cudaMemcpy(offsets.data(), starts.p, offsets.size() * sizeof(int), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(ranks.data(), localRanks.p, ranks.size() * sizeof(int), cudaMemcpyDeviceToHost));
    if (offsets != expected) throw std::runtime_error("Incorrect local expert offsets");
    std::vector<bool> used(expected.back());
    for (int i = 0; i < routes; ++i) if (ids[i] >= 0) {
        const int row = offsets[ids[i]] + ranks[i];
        if (row < offsets[ids[i]] || row >= offsets[ids[i] + 1] || used[row])
            throw std::runtime_error("Overlapping or missing local expert rows");
        used[row] = true;
    }
}
#endif
}
int main() {
    int count = 0, image = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    check(cudaMemcpyFromSymbol(&image, op::nativeImage, sizeof(image)));
    if (!image) return 77;
    for (bool skew : {false, true}) test<16>(73, false, skew);
    for (bool up : {false, true})
        for (int rows : {1, 3, 31, 257})
            for (bool skew : {false, true}) {
                test<32>(rows, up, skew);
                test<64>(rows, up, skew);
            }
    for (bool up : {false, true})
        for (int rows : {5, 7, 8, 9, 12, 17})
            for (bool skew : {false, true})
                test<32>(rows, up, skew, true);
#ifdef FASTLLM_INKLING_GROUPED_SM120
    for (int routes : {64, 2048, 8192})
        for (bool allRemote : {false, true}) testLocalRoutePlan(routes, allRemote);
    op::GroupedWeightMaps invalidMaps{};
    if (op::makeGroupedWeightMaps(invalidMaps, {}, 1, 48, 320) ||
        op::makeGroupedWeightMaps(invalidMaps, {}, 1, 256, 96))
        throw std::runtime_error("Expected the unaligned weight-map fallback");
    if (op::useGroupedVerify(31, 8, 128, 2304, 1152) || !op::useGroupedVerify(33, 8, 128, 2304, 1152))
        throw std::runtime_error("Incorrect grouped verification crossover");
    for (bool up : {false, true})
        for (bool skew : {false, true}) test<32>(73, up, skew, true, true);
#endif
    if (op::decodeRouteTileRows(4, 8) || op::decodeRouteTileRows(1, 40))
        throw std::runtime_error("Expected the small-route decode fallback");
    testScatter<32>(192);
    testScatter<64>(2304);
    testScatter<32, true>(192);
    testScatter<64, true>(2304);
    testPackedActivation<32>(192, true);
    testPackedActivation<64>(1152, true);
    testPackedActivation<64>(2048, true);
    testPackedActivation<32>(192);
    testPackedActivation<64>(1152);
    testPackedActivation<64>(12288);
}
