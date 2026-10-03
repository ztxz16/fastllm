#include "../../src/models/glm5_next_dsa.h"
#include "../../src/devices/cuda/models/glm5-next-dsa.cuh"
#include "executor.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
using namespace fastllm;
namespace {
void Check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
void Cuda(cudaError_t e) { Check(e == cudaSuccess, cudaGetErrorString(e)); }
uint16_t Bf(float x) {
    uint32_t u; std::memcpy(&u, &x, 4);
    return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
float Float(uint16_t x) {
    uint32_t u = uint32_t(x) << 16; float f; std::memcpy(&f, &u, 4); return f;
}
void Allocate(Data &x, DataType type, const std::vector<int> &dims) {
    x.dataType = type; x.UpdateUnitSize(); x.Resize(dims);
    x.ToDevice(DataDevice::CUDA, {0}, false); x.Allocate(false);
}
template<class T> void Upload(Data &x, DataType type, const std::vector<int> &dims,
                               const std::vector<T> &values) {
    Allocate(x, type, dims);
    Cuda(cudaMemcpy(x.cudaData, values.data(), values.size()*sizeof(T), cudaMemcpyHostToDevice));
}
std::vector<uint16_t> Download(Data &x) {
    size_t count = 1; for (int d : x.dims) count *= d;
    std::vector<uint16_t> y(count);
    Cuda(cudaMemcpy(y.data(), x.cudaData, count*2, cudaMemcpyDeviceToHost)); return y;
}
struct BorrowedCache : Data { ~BorrowedCache() { isPagedKVCache = false; } };

void Case(int tokens, int rows, bool available) {
    constexpr int heads = 64, dim = 512, width = 2051, paddedWidth = 2112;
    std::mt19937 rng(17); std::normal_distribution<float> normal(0, .25f);
    std::vector<uint16_t> q(size_t(rows)*heads*dim), kv(size_t(tokens)*dim);
    for (auto &v : q) v = Bf(normal(rng));
    for (auto &v : kv) v = Bf(normal(rng));
    std::vector<int32_t> ids(rows*width, -1), padded(rows*paddedWidth, -1);
    for (int r = 1; r < rows; ++r) {
        int pos = tokens-rows+r, groups = (pos+1)/4, step = 37;
        while (std::gcd(step, groups) != 1) ++step;
        if (r == 1) ids[r*width] = pos;
        else {
            for (int j = 0; j < 512; ++j) for (int k = 0; k < 4; ++k)
                ids[r*width+j*4+k] = ((j*step+r*7)%groups)*4+k;
            for (int t = groups*4; t <= pos; ++t) ids[r*width+2048+t-groups*4] = t;
        }
        std::copy_n(ids.data()+r*width, width, padded.data()+r*paddedWidth);
    }
    Data query, latent, indices, output;
    Upload(query, BFLOAT16, {1, rows, heads, dim}, q);
    Upload(latent, BFLOAT16, {1, tokens, dim}, kv);
    Upload(indices, INT32, {1, rows, width}, ids);
    const bool used = FastllmCudaGlm5NextDsaPrefill(query, latent, indices, 1.f/16, output);
    Check(used == available, "backend dispatch");
    if (!available) {
        Check(output.cudaData == nullptr && output.dims.empty(), "fallback changed output");
        std::puts("PASS unsupported backend returns without changing output"); return;
    }
#ifdef FASTLLM_ENABLE_GLM53_FLASHINFER_SM120
    auto got = Download(output);
    for (size_t i = 0; i < got.size(); ++i) {
        Check(std::isfinite(Float(got[i])), "nonfinite attention output");
        if (i < heads*dim) Check(Float(got[i]) == 0, "all-masked output");
    }
    // Independent CPU E4M3 rounding and packing, including arbitrary FP32 scales.
    float table[127]{};
    for (int i = 1; i < 127; ++i)
        table[i] = i/8 ? std::ldexp(1.f+(i%8)/8.f, i/8-7) : std::ldexp(float(i), -9);
    auto encode = [&](float x) {
        float a = std::abs(x); int hi = int(std::lower_bound(table, table+127, a)-table);
        int code = std::min(hi, 126);
        if (hi > 0 && hi < 127) {
            float lowDistance = a-table[hi-1], highDistance = table[hi]-a;
            if (lowDistance < highDistance || (lowDistance == highDistance && !((hi-1)&1))) code = hi-1;
        }
        return uint8_t(code | (std::signbit(x) ? 128 : 0));
    };
    std::vector<uint8_t> cache(size_t(tokens)*528);
    std::vector<float> decoded(kv.size());
    for (int t = 0; t < tokens; ++t) for (int g = 0; g < 4; ++g) {
        float peak = 1e-4f;
        for (int d = 0; d < 128; ++d) peak = std::max(peak, std::abs(Float(kv[t*512+g*128+d])));
        float scale = peak/448;
        std::memcpy(cache.data()+t*528+512+g*4, &scale, 4);
        for (int d = 0; d < 128; ++d) {
            int at = t*512+g*128+d; uint8_t code = encode(Float(kv[at])/scale);
            cache[t*528+g*128+d] = code;
            decoded[at] = table[code&127]*scale*(code&128 ? -1 : 1);
        }
    }
    Data packed, paddedIndices, reference, lse;
    Upload(packed, INT8, {tokens, 528}, cache);
    Upload(paddedIndices, INT32, {1, rows, paddedWidth}, padded);
    Allocate(reference, BFLOAT16, query.dims); Allocate(lse, FLOAT32, {rows, heads});
    Cuda(FastllmCudaGlm5NextDsaPrefillSm120Raw(query.cudaData, packed.cudaData,
        static_cast<const int32_t *>(paddedIndices.cudaData), reference.cudaData,
        static_cast<float *>(lse.cudaData), rows, paddedWidth, 1.f/16, cudaStreamPerThread));
    Check(got == Download(reference), "GPU quantization/packing/padding differs from CPU reference");
    std::vector<float> hostLse(rows*heads);
    Cuda(cudaMemcpy(hostLse.data(), lse.cudaData, hostLse.size()*4, cudaMemcpyDeviceToHost));
    for (int h = 0; h < heads; ++h) Check(hostLse[h] == -INFINITY, "all-masked LSE");
    // Independent double softmax against the quantized KV, without using a GPU reference kernel.
    double worst = 0;
    for (int r : {1, rows-1}) for (int h : {0, 17, 63}) {
        std::vector<double> p(width); double den = 0;
        for (int j = 0; j < width; ++j) {
            int t = ids[r*width+j]; if (t < 0) continue;
            double dot = 0;
            for (int d = 0; d < dim; ++d) dot += Float(q[(r*heads+h)*dim+d])*double(decoded[t*dim+d]);
            den += (p[j] = std::exp(dot/16));
        }
        for (int d : {0, 63, 127, 255, 511}) {
            double sum = 0;
            for (int j = 0; j < width; ++j) if (ids[r*width+j] >= 0)
                sum += p[j]*decoded[ids[r*width+j]*dim+d];
            worst = std::max(worst, std::abs(Float(got[(r*heads+h)*dim+d])-sum/den));
        }
    }
    Check(worst < .003, "FP8 attention vs double reference");
    // Exercise the real paged-cache gather and query/output permutations.
    if (tokens == 4099) {
        const int pageLen = 32, pages = (tokens+pageLen-1)/pageLen;
        std::vector<int> pageIds(pages);
        std::vector<uint16_t> physical(size_t(pages)*2*pageLen*dim), transposed(q.size());
        for (int p = 0; p < pages; ++p) pageIds[p] = (pages-1-p)*2;
        for (int t = 0; t < tokens; ++t)
            std::copy_n(kv.data()+t*dim, dim, physical.data()+(pageIds[t/pageLen]*pageLen+t%pageLen)*dim);
        for (int r = 0; r < rows; ++r) for (int h = 0; h < heads; ++h)
            std::copy_n(q.data()+(r*heads+h)*dim, dim, transposed.data()+(h*rows+r)*dim);
        PagedCacheManager pool; Upload(pool, BFLOAT16, {pages*2, pageLen, 1, dim}, physical);
        BorrowedCache state; state.dataType = BFLOAT16; state.Resize({1, tokens, dim});
        state.isPagedKVCache = true; state.pageLen = pageLen; state.lastPageLen = (tokens-1)%pageLen+1;
        state.pageIndex = pageIds; state.pagedKVCacheData = &pool;
        Data headQuery, selected, result;
        Upload(headQuery, BFLOAT16, {heads, rows, dim}, transposed);
        Upload(selected, INT32, {rows, width}, ids);
        glm5_next_detail::SparseLatentAttention(headQuery, state, selected, 1.f/16, result);
        auto y = Download(result);
        for (int r = 0; r < rows; ++r) for (int h = 0; h < heads; ++h)
            Check(std::equal(y.data()+(h*rows+r)*dim, y.data()+(h*rows+r+1)*dim,
                             got.data()+(r*heads+h)*dim), "fragmented paged attention mismatch");
        // Force BF16 through the same paged helper and compare with its direct
        // kernel. This case is large enough that auto would select FlashInfer.
        Data sink(FLOAT32), bf16Output;
        sink.Resize({heads}); sink.ToDevice(DataDevice::CUDA, {0}, false);
        sink.Allocate(-std::numeric_limits<float>::infinity());
        Check(FastllmCudaDeepSeekV41SparseAttention(query, latent, nullptr,
            &latent, &indices, sink, 0, 0, 1.f/16, bf16Output), "BF16 reference launch");
        const auto bf16 = Download(bf16Output);
        Check(bf16 != got, "fixture must distinguish BF16 and FlashInfer dispatch");
        Upload(headQuery, BFLOAT16, {heads, rows, dim}, transposed);
        glm5_next_detail::SparseLatentAttention(headQuery, state, selected, 1.f/16, result, false);
        y = Download(result);
        for (int r = 0; r < rows; ++r) for (int h = 0; h < heads; ++h)
            Check(std::equal(y.data()+(h*rows+r)*dim, y.data()+(h*rows+r+1)*dim,
                             bf16.data()+(r*heads+h)*dim), "forced BF16 paged attention mismatch");
        std::puts("PASS forced BF16 paged attention matches direct kernel exactly");
    }
    // Unsupported layouts and short queries must not mutate an existing output.
    const auto saved = Download(output);
    query.dims[1] = 63;
    Check(!FastllmCudaGlm5NextDsaPrefill(query, latent, indices, 1.f/16, output), "short query fallback");
    query.dims[1] = rows; ++latent.strides[1];
    Check(!FastllmCudaGlm5NextDsaPrefill(query, latent, indices, 1.f/16, output), "strided latent fallback");
    --latent.strides[1]; query.dataType = FLOAT32;
    Check(!FastllmCudaGlm5NextDsaPrefill(query, latent, indices, 1.f/16, output), "dtype fallback");
    query.dataType = BFLOAT16; query.dims[2] = 32;
    Check(!FastllmCudaGlm5NextDsaPrefill(query, latent, indices, 1.f/16, output), "head count fallback");
    query.dims[2] = heads;
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    bool captured = FastllmCudaGlm5NextDsaPrefill(query, latent, indices, 1.f/16, output);
    cudaGraph_t graph = nullptr; Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Cuda(cudaGraphDestroy(graph));
    Check(!captured, "graph capture fallback");
    Check(saved == Download(output), "unsupported input changed output");
    std::printf("PASS FlashInfer tokens=%d rows=%d CPU packing exact, double max_error=%.8g\n", tokens, rows, worst);
#endif
}
} // namespace
int main() {
    int count = 0; if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    FastllmCudaSetDevice(0); static_cast<Executor *>(GetExecutor())->SetFirstDevice("cuda:0");
    cudaDeviceProp prop{}; Cuda(cudaGetDeviceProperties(&prop, 0));
    bool available = false;
#ifdef FASTLLM_ENABLE_GLM53_FLASHINFER_SM120
    available = prop.major == 12 && prop.minor == 0;
#endif
    try {
        Case(4099, 64, available);
        if (available) { Case(16384, 64, true); Case(32768, 65, true); }
        Cuda(cudaDeviceSynchronize()); std::puts("PASS GLM53 FlashInfer integration"); return 0;
    } catch (const std::exception &e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
