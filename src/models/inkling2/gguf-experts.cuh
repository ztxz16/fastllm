// GGUF records are contiguous by expert. The raw and Tensor Core paths share
// the range checks and stream at most one expert's file data at a time.
struct GGUFExpertReader {
    std::string name;
    std::ifstream input;
    size_t rowBytes;
    std::vector<uint8_t> record;
    std::vector<float> scales;

    GGUFExpertReader(const std::map<std::string, TensorRange> &files, const std::string &name, int experts,
                     int rows, int cols) : name(name) {
        const auto &w = files.at(name);
        if (w.shape != std::vector<int>{experts, rows, cols} || w.ggmlType < 0 ||
            cols % ggml_blck_size(ggml_type(w.ggmlType)))
            throw std::runtime_error("Invalid GGUF expert matrix: " + name);
        rowBytes = ggml_row_size(ggml_type(w.ggmlType), cols);
        if (w.end < w.start || w.end - w.start != size_t(experts) * rows * rowBytes)
            throw std::runtime_error("Invalid GGUF expert byte range: " + name);
        input.open(w.file, std::ios::binary);
        if (!input.seekg(w.start)) throw std::runtime_error("Cannot open GGUF expert matrix: " + name);
        record.resize(rows * rowBytes);
        const auto it = files.find(name + "_scale");
        if (it != files.end()) {
            if (it->second.ggmlType != GGML_TYPE_F32) throw std::runtime_error("Invalid GGUF expert scale: " + name);
            scales = readRange<float>(it->second, {experts}, "GGUF");
            for (float v : scales)
                if (!std::isfinite(v)) throw std::runtime_error("Invalid GGUF expert scale: " + name);
        }
    }

    void read() {
        if (!input.read(reinterpret_cast<char *>(record.data()), record.size()))
            throw std::runtime_error("Truncated GGUF expert matrix: " + name);
    }
};

void Projection::loadNativeGGUF(const std::map<std::string, TensorRange> &files, const std::string &name, int experts) {
    GGUFExpertReader source(files, name, experts, rows, cols);
    const size_t blocks = size_t(rows) * cols / 64, bytes = blocks * 32;
    if (source.scales.empty()) source.scales.assign(experts, 1.f);
    std::vector<uint8_t> planar(source.record.size());
    // GGUF gate and up rows are already ordered. Repack only the codes within each block.
    packNative(source.scales, false, [&](int, unsigned *staging) {
        source.read();
        for (size_t b = 0; b < blocks; ++b) {
            const uint8_t *block = source.record.data() + 36 * b;
            for (int g = 0; g < 4; ++g) {
                if (block[g] & 0x80) throw std::runtime_error("Invalid GGUF NVFP4 block scale: " + name);
                // GGML decodes its reserved UE4M3 scale as zero.
                planar[bytes + 4 * b + g] = block[g] == 0x7f ? 0 : block[g];
                const uint8_t *q = block + 4 + 8 * g;
                uint8_t *out = planar.data() + 32 * b + 8 * g;
                // A GGUF byte pairs columns j and j+8. Native bytes pair adjacent columns.
                for (int j = 0; j < 4; ++j) {
                    out[j] = (q[2 * j] & 15) | (q[2 * j + 1] << 4);
                    out[j + 4] = (q[2 * j] >> 4) | (q[2 * j + 1] & 0xf0);
                }
            }
        }
        check(cudaMemcpy(staging, planar.data(), planar.size(), cudaMemcpyHostToDevice));
    });
}

struct ResidentGGUFProjection {
    int type, rows, cols;
    FastllmGGUFExpertProjection gemv;
    size_t rowBytes;
    Buffer<uint8_t> blocks;
    Buffer<float> scale;

    ResidentGGUFProjection(const std::map<std::string, TensorRange> &files, const std::string &name, int experts,
                           int rows, int cols)
        : type(files.at(name).ggmlType), rows(rows), cols(cols), gemv(FastllmCudaGetGGUFExpertProjection(type)) {
        GGUFExpertReader source(files, name, experts, rows, cols);
        rowBytes = source.rowBytes;
        const size_t bytes = source.record.size();
        blocks = Buffer<uint8_t>(experts * bytes);
        for (int e = 0; e < experts; ++e) {
            source.read();
            check(cudaMemcpy(blocks.p + e * bytes, source.record.data(), bytes, cudaMemcpyHostToDevice));
        }
        if (!source.scales.empty()) {
            scale = Buffer<float>(experts);
            check(cudaMemcpy(scale.p, source.scales.data(), experts * sizeof(float), cudaMemcpyHostToDevice));
        }
    }
};

__global__ void selectGGUFExperts(RouteSelection route) {
    if (route.experts == 128 && route.top == 8)
        selectRow<128>(route);
    else
        selectRow<0>(route);
    route.restrictToRank(blockIdx.x);
}
template <class Activation>
__global__ void ggufRoutedActivation(Activation activation, float *out, int routes, int width) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size_t(routes) * width) out[i] = activation.get(i / width, i % width, width);
}
class GpuGGUFExperts final : public ExpertBackend {
    const Data &router;
    const int firstExpert, residentExperts;
    ResidentGGUFProjection gate;
    std::unique_ptr<ResidentGGUFProjection> ggufDown;
    std::unique_ptr<Projection> nativeDown;
    Buffer<const uint8_t *> expertPointers;
    int mmqMaxRows, smCount;

    int prefillChunk(const Config &c, int rows) const {
        // Amortize the last partially occupied 64-row MMQ tile of each expert.
        // Use whole router groups so the router GEMM and top-k keep their original shapes.
        const int target = (4 * FastllmCudaGGUFExpertGroups::wideTileRows * c.experts + c.topK - 1) / c.topK;
        const int limit = std::min(rows, FastllmCudaGGUFExpertGroups::gridRowLimit(residentExperts, c.topK));
        int hi = std::min(limit / mmqMaxRows, (target + mmqMaxRows - 1) / mmqMaxRows);
        if (hi < 2) return mmqMaxRows;
        size_t freeBytes, totalBytes;
        check(cudaMemGetInfo(&freeBytes, &totalBytes));
        size_t staging = 0;
        if (nativeDown)
            staging = size_t(c.intermediate / 16) * sizeof(native::Block16) + size_t(c.hidden) * sizeof(__nv_bfloat16);
        else if (ggufDown->type == GGML_TYPE_BF16)
            staging = size_t(c.intermediate) * sizeof(float);
        int lo = 1;
        while (lo < hi) {
            const int groups = (lo + hi + 1) / 2, n = groups * mmqMaxRows;
            const size_t bytes = FastllmCudaGGUFExpertMMQBytes(n, c.hidden, c.intermediate, residentExperts, c.topK)
                + size_t(n) * c.topK * (staging + sizeof(int) + sizeof(float))
                + size_t(mmqMaxRows) * c.experts * sizeof(float);
            if (bytes <= freeBytes / 8) lo = groups;
            else hi = groups - 1;
        }
        return lo * mmqMaxRows;
    }

public:
    GpuGGUFExperts(const Config &c, const std::map<std::string, TensorRange> &files, const std::string &prefix,
                   const Data &router, ExpertLoadContext &topology, bool useNativeDown)
        : router(router), firstExpert(topology.firstExpert),
          residentExperts(topology.localExperts ? topology.localExperts : c.experts),
          gate(files, prefix + "w13_weight", residentExperts, 2 * c.intermediate, c.hidden), smCount(topology.smCount) {
        if (c.topK < 1 || c.topK > 50) throw std::runtime_error("Unsupported Inkling expert top-k");
        const auto name = prefix + "w2_weight";
        if (useNativeDown)
            nativeDown = std::make_unique<Projection>(files, name, residentExperts, c.hidden, c.intermediate,
                                                      ExpertDevice::Cuda, topology, false);
        else
            ggufDown = std::make_unique<ResidentGGUFProjection>(files, name, residentExperts, c.hidden, c.intermediate);
        mmqMaxRows = FastllmCudaGGUFExpertMMQMaxRows(gate.type, ggufDown ? ggufDown->type : -1, c.hidden,
                                                     c.intermediate, residentExperts, c.topK);
        if (mmqMaxRows > 1) {
            std::vector<const uint8_t *> pointers(2 * residentExperts);
            for (int e = 0; e < residentExperts; ++e) {
                pointers[2 * e] = gate.blocks.p + size_t(e) * gate.rows * gate.rowBytes;
                pointers[2 * e + 1] =
                    ggufDown ? ggufDown->blocks.p + size_t(e) * ggufDown->rows * ggufDown->rowBytes : pointers[2 * e];
            }
            expertPointers = Buffer<const uint8_t *>(pointers.size());
            check(cudaMemcpy(expertPointers.p, pointers.data(), pointers.size() * sizeof(pointers[0]),
                             cudaMemcpyHostToDevice));
        }
    }

    Matrix run(const Config &c, const Matrix &x, const float *bias, ExpertPhase phase,
               Rows<__nv_bfloat16> *, Rows<__nv_bfloat16> *unreduced, int) const override {
        // Bound both MMQ scratch and the one-route-per-grid-row fallback.
        const int routeRows = mmqMaxRows > 1 ? mmqMaxRows : 65535 / c.topK;
        if (!nativeDown || x.rows > routeRows ||
            !(phase == ExpertPhase::Verify || (phase == ExpertPhase::Decode && x.rows == 1)))
            unreduced = nullptr;
        Matrix out(unreduced ? 0 : x.rows, c.hidden);
        auto runChunk = [&](int first, int rows) {
            const int routes = rows * c.topK;
            const bool sharded = residentExperts != c.experts;
            auto input = x.slice(first, rows);
            Matrix scores(rows, c.topK);
            Buffer<int> ids(routes);
            for (int start = 0; start < rows; start += routeRows) {
                const int count = std::min(routeRows, rows - start);
                Matrix logits = routeLogits(input.slice(start, count), router);
                const size_t offset = size_t(start) * c.topK;
                selectGGUFExperts<<<count, 64, 0, cudaStreamPerThread>>>(
                    {logits.p, bias, ids.p + offset, scores.p + offset, c.experts, c.topK, c.routeScale,
                     firstExpert, sharded ? residentExperts : 0, -1});
            }
            // A few routes do not fill even two 16-route MMQ tiles.
            const bool grouped = expertPointers.p && rows > 1 && routes >= 32;
            Buffer<uint8_t> workspace(
                grouped ? FastllmCudaGGUFExpertMMQBytes(rows, c.hidden, c.intermediate, residentExperts, c.topK) : 0);
            const bool groupedPrefill = grouped && phase == ExpertPhase::Prefill;
            const auto groups = groupedPrefill ? FastllmCudaGGUFExpertGroupsView(
                workspace.p, rows, c.hidden, c.intermediate, residentExperts, c.topK) : FastllmCudaGGUFExpertGroups{};
            Matrix pair(groupedPrefill ? 0 : routes, 2 * c.intermediate);
            auto project = [&](const ResidentGGUFProjection &w, bool gateUp, const Matrix &in, Matrix &dst) {
                if (grouped) {
                    FastllmCudaGGUFExpertMMQ(gateUp, w.type, expertPointers.p, w.scale.p, ids.p, in.p, dst.p,
                                             workspace.p, rows, c.hidden, c.intermediate, residentExperts, c.topK);
                    return;
                }
                if (gateUp && rows == 1 && nativeDown) {
                    Buffer<uint8_t> q(FastllmCudaQ8ProjectionScratchBytes(1, w.cols));
                    if (FastllmCudaGGUFSingleExpertProjection(w.type, w.blocks.p, w.scale.p, ids.p, in.p, dst.p,
                                                              q.p, c.topK, w.cols, w.rows, w.rowBytes)) return;
                }
                w.gemv(w.blocks.p, w.scale.p, ids.p, in.p, dst.p, routes, gateUp ? c.topK : 1, w.cols, w.rows, w.rowBytes);
            };
            project(gate, true, input, pair);
            RoutedSiTU<float> activation{pair.p, scores.p, c.gateCap, c.upCap};
            cuda::GroupedRoutedSiTU sorted{{groups.products, scores.p, c.gateCap, c.upCap},
                                          groups.routeGroups, ids.p, gate.scale.p};
            if (nativeDown) {
                Buffer<native::Block16> mid(size_t(routes) * c.intermediate / 16);
                Rows<__nv_bfloat16> projected(routes, c.hidden);
                if (groupedPrefill) {
                    quantizeFp4Into(sorted, mid.p, routes, c.intermediate, smCount);
                    // The MMQ gate has finished reading its Q8 scratch. FP4 input tiles fit there,
                    // and use exactly the same expert groups without a second sort.
                    constexpr int tileRows = FastllmCudaGGUFExpertGroups::tileRows;
                    const native::PaddedRouteTiles<tileRows> plan{groups.counts, groups.offsets, groups.tileExperts, groups.routes, residentExperts};
                    const int tiles = groups.capacity / tileRows;
                    auto *packed = static_cast<native::InputTile *>(groups.activationScratch);
                    static_assert(sizeof(native::InputTile) <= 8 * 64); // MMQ reserves at least one byte per value
                    native::packInput<<<dim3((c.intermediate / 64 + 7) / 8, tiles, tileRows / 8), 256, 0, cudaStreamPerThread>>>(
                        plan, mid.p, packed, c.intermediate, 1);
                    native::groupedProject<4, 2, false><<<dim3((c.hidden + 127) / 128, tiles), 128, 0, cudaStreamPerThread>>>(
                        nativeDown->nativeView(), mid.p, nullptr, packed, plan, projected.p, c.hidden, c.intermediate, 1);
                } else {
                    quantizeFp4Into(activation, mid.p, routes, c.intermediate, smCount, true);
                    projectNativeRoutes<false>(*nativeDown, mid.p, ids.p, rows, c.topK, projected.p);
                }
                finishNativeRoutes(c, projected, unreduced ? nullptr : out.p + size_t(first) * c.hidden,
                                   unreduced, groupedPrefill && sharded ? ids.p : nullptr);
                return;
            }
            float *dst = out.p + size_t(first) * c.hidden;
            if (groupedPrefill && ggufDown->type != GGML_TYPE_BF16) {
                FastllmCudaGGUFExpertSiTUDown(ggufDown->type, expertPointers.p, ids.p, scores.p, gate.scale.p,
                    c.gateCap, c.upCap, workspace.p, rows, c.hidden, c.intermediate, residentExperts, c.topK);
            } else {
                Matrix mid(routes, c.intermediate), result(groupedPrefill ? 0 : routes, c.hidden);
                const int blocks = (size_t(routes) * c.intermediate + 255) / 256;
                if (groupedPrefill)
                    ggufRoutedActivation<<<blocks, 256, 0, cudaStreamPerThread>>>(sorted, mid.p, routes, c.intermediate);
                else ggufRoutedActivation<<<blocks, 256, 0, cudaStreamPerThread>>>(activation, mid.p, routes, c.intermediate);
                project(*ggufDown, false, mid, result);
                if (!groupedPrefill) {
                    sumNativeRoutes<false, false, float, float><<<(size_t(rows) * c.hidden + 255) / 256, 256, 0, cudaStreamPerThread>>>(
                        result.p, dst, rows, c.hidden, c.topK, c.hidden, c.hidden);
                    return;
                }
            }
            FastllmCudaGGUFSumExpertGroups(groups, ids.p, ggufDown->scale.p, dst, rows, c.hidden, c.topK);
        };
        int chunk = routeRows;
        if (phase == ExpertPhase::Prefill && expertPointers.p && !DecodeGraphScratch::active)
            chunk = prefillChunk(c, x.rows);
        const int tail = x.rows % routeRows;
        // A short final group keeps its original GEMV or narrow MMQ kernel.
        const int joinableRows = tail < FastllmCudaGGUFExpertGroups::wideBatchRows ? x.rows - tail : x.rows;
        for (int first = 0; first < x.rows;) {
            const int rows = std::min(chunk, (first < joinableRows ? joinableRows : x.rows) - first);
            try {
                runChunk(first, rows);
                first += rows;
            } catch (const CudaWorkspaceExhausted &) {
                if (rows <= routeRows) throw;
                const auto status = cudaGetLastError();
                if (status != cudaErrorMemoryAllocation) check(status);
                // runChunk only reads routing inputs. A retry on this stream can
                // safely replace the same output rows if another allocator won the race.
                chunk = routeRows;
            }
        }
        return out;
    }
};
