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
}
__global__ void ggufRoutedActivation(RoutedSiTU<float> activation, float *out, int routes, int width) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size_t(routes) * width) out[i] = activation.get(i / width, i % width, width);
}
class GpuGGUFExperts final : public ExpertBackend {
    const Data &router;
    ResidentGGUFProjection gate;
    std::unique_ptr<ResidentGGUFProjection> ggufDown;
    std::unique_ptr<Projection> nativeDown;
    Buffer<const uint8_t *> expertPointers;
    int mmqMaxRows, smCount;

public:
    GpuGGUFExperts(const Config &c, const std::map<std::string, TensorRange> &files, const std::string &prefix,
                   const Data &router, ExpertLoadContext *nativeContext)
        : router(router), gate(files, prefix + "w13_weight", c.experts, 2 * c.intermediate, c.hidden),
          smCount(nativeContext ? nativeContext->smCount : 0) {
        if (c.topK < 1 || c.topK > 50) throw std::runtime_error("Unsupported Inkling expert top-k");
        const auto name = prefix + "w2_weight";
        if (nativeContext)
            nativeDown = std::make_unique<Projection>(files, name, c.experts, c.hidden, c.intermediate,
                                                      ExpertDevice::Cuda, *nativeContext, false);
        else
            ggufDown = std::make_unique<ResidentGGUFProjection>(files, name, c.experts, c.hidden, c.intermediate);
        mmqMaxRows = FastllmCudaGGUFExpertMMQMaxRows(gate.type, ggufDown ? ggufDown->type : -1, c.hidden,
                                                     c.intermediate, c.experts, c.topK);
        if (mmqMaxRows > 1) {
            std::vector<const uint8_t *> pointers(2 * c.experts);
            for (int e = 0; e < c.experts; ++e) {
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
               Rows<__nv_bfloat16> *unreduced) const override {
        // Bound both MMQ scratch and the one-route-per-grid-row fallback.
        const int chunk = mmqMaxRows > 1 ? mmqMaxRows : 65535 / c.topK;
        if (!nativeDown || x.rows > chunk ||
            !(phase == ExpertPhase::Verify || (phase == ExpertPhase::Decode && x.rows == 1)))
            unreduced = nullptr;
        Matrix out(unreduced ? 0 : x.rows, c.hidden);
        for (int first = 0; first < x.rows; first += chunk) {
            const int rows = std::min(chunk, x.rows - first), routes = rows * c.topK;
            auto input = x.slice(first, rows);
            Matrix logits = routeLogits(input, router), scores(rows, c.topK);
            Buffer<int> ids(routes);
            selectGGUFExperts<<<rows, 64, 0, cudaStreamPerThread>>>(
                {logits.p, bias, ids.p, scores.p, c.experts, c.topK, c.routeScale});
            Matrix pair(routes, 2 * c.intermediate);
            // A few routes do not fill even two 16-route MMQ tiles.
            const bool grouped = expertPointers.p && rows > 1 && routes >= 32;
            Buffer<uint8_t> workspace(
                grouped ? FastllmCudaGGUFExpertMMQBytes(rows, c.hidden, c.intermediate, c.experts, c.topK) : 0);
            auto project = [&](const ResidentGGUFProjection &w, bool gateUp, const Matrix &in, Matrix &dst) {
                if (grouped) {
                    FastllmCudaGGUFExpertMMQ(gateUp, w.type, expertPointers.p, w.scale.p, ids.p, in.p, dst.p,
                                             workspace.p, rows, c.hidden, c.intermediate, c.experts, c.topK);
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
            if (nativeDown) {
                Buffer<native::Block16> mid(size_t(routes) * c.intermediate / 16);
                Rows<__nv_bfloat16> projected(routes, c.hidden);
                quantizeFp4Into(activation, mid.p, routes, c.intermediate, smCount, true);
                projectNativeRoutes<false>(*nativeDown, mid.p, ids.p, rows, c.topK, projected.p);
                finishNativeRoutes(c, projected, unreduced ? nullptr : out.p + size_t(first) * c.hidden, unreduced);
                continue;
            }
            Matrix mid(routes, c.intermediate), result(routes, c.hidden);
            ggufRoutedActivation<<<(size_t(routes) * c.intermediate + 255) / 256, 256, 0, cudaStreamPerThread>>>(
                activation, mid.p, routes, c.intermediate);
            project(*ggufDown, false, mid, result);
            sumNativeRoutes<false, float><<<(size_t(rows) * c.hidden + 255) / 256, 256, 0, cudaStreamPerThread>>>(
                result.p, out.p + size_t(first) * c.hidden, rows, c.hidden, c.topK);
        }
        return out;
    }
};
