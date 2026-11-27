#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#include "devices/cuda/naive-n05-dp-dsa-verify.cuh"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/multicuda/fastllm-naive-verify.h"
#include "utils/utils.h"
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace fastllm;
using B = __nv_bfloat16;
#include "naive_n05_dsa_cache_fixture.cuh"
namespace {
void VerifyDsaRequests(int owners, bool paged, DataType storage = BFLOAT16, std::vector<int> lengths = {}, bool pair = false) {
    if (lengths.empty()) lengths.assign(owners, 8);
    std::vector<int> offsets{0};
    for (int n : lengths) offsets.push_back(offsets.back() + n);
    const int rows = offsets.back(), pitch = (rows + 3) / 4 * 4;
    constexpr int cols = 8192;
    const int capacity = paged ? 16896 : 8192;
    FastllmNaiveVerifyAttentionInput x{};
    FastllmNaiveDecodeScratch work, reference;
    Data hidden, qkv, wq, wk, ww, gamma, bias, sink, positions, live, query, ids, output, pack;
    auto make = [&](Data &d, DataType type, std::vector<int> shape, unsigned seed) {
        Init(d, type, shape);
        Fill(d, seed, .3f);
    };
    make(hidden, BFLOAT16, {1, rows, 4096}, 33);
    make(qkv, BFLOAT16, {1, rows, 13568}, 12);
    make(wk, BFLOAT16, {128, 4096}, 11);
    make(wq, BFLOAT16, {2048, 4096}, 21);
    make(ww, BFLOAT16, {16, 4096}, 31);
    Require(FastllmCudaPackIndexerWeights(wq, wk, ww), "pack index weights");
    make(gamma, FLOAT32, {128}, 13);
    make(bias, FLOAT32, {128}, 17);
    make(sink, FLOAT32, {64}, 19);
    Init(positions, FLOAT32, {1, rows});
    Init(live, INT32, {rows});
    Init(pack, INT32, {pitch * (cols + cols / 32) / 4});
    auto *base = (uint8_t *)pack.cudaData;
    FastllmMxfp8Activation packed{base, base + pitch * cols, rows, cols, pitch};
    x.qkv = &qkv;
    x.query = &query;
    x.normed = &hidden;
    x.sink = &sink;
    x.indexNorm = &gamma;
    x.indexBias = &bias;
    x.indexKeyWeight = &wk;
    x.indexQueryWeight = &wq;
    x.indexWeight = &ww;
    x.live = &live;
    x.positions = &positions;
    x.indices = &ids;
    x.attention = &output;
    x.scratch = &work;
    x.capacity = capacity;
    x.theta = 10000.f;
    x.valueScale = .707f;
    x.requests = lengths;
    x.config.heads = 64;
    x.config.kvHeads = 4;
    x.config.headDim = 192;
    x.config.valueDim = 128;
    x.config.rotaryDim = 64;
    x.indexDim = 128;
    x.indexHeads = 16;
    x.topK = 2048;
    std::vector<std::unique_ptr<Cache>> caches, refs;
    for (int j = 0; j < owners; ++j) {
        caches.push_back(std::make_unique<Cache>(capacity, paged, false, 41 + j, 4, storage));
        refs.push_back(std::make_unique<Cache>(capacity, paged, false, 41 + j, 4, storage));
        for (int i = 0; i < lengths[j]; ++i)
            x.caches.push_back(
                {caches[j]->K(), caches[j]->V(), nullptr, (const void *const *)caches[j]->table.cudaData});
    }
    if (pair) {
        Require(owners == 2 && storage == BFLOAT16, "native pair test geometry");
        auto runPair = [&] {
            return FastllmCudaTryDpDSAVerify(hidden,qkv,wk,wq,ww,gamma,bias,sink,positions,live,
                *caches[0]->K(),*caches[0]->V(),x.caches[0].cache,capacity,x.theta,x.valueScale,
                packed,work,output,false,&x.caches[lengths[0]],lengths[0]);
        };
        Cuda(cudaMemset(live.cudaData,0,live.GetBytes()));
        Require(runPair(),"native pair warmup");
        cudaGraph_t graph; cudaGraphExec_t exec;
        Cuda(cudaStreamBeginCapture(cudaStreamPerThread,cudaStreamCaptureModeThreadLocal));
        Require(runPair(),"native pair capture");
        Cuda(cudaStreamEndCapture(cudaStreamPerThread,&graph));
        size_t nodes = 0;
        Cuda(cudaGraphGetNodes(graph,nullptr,&nodes));
        Require(nodes == 1,"native DSA pair must stay fused");
        Cuda(cudaGraphInstantiate(&exec,graph,0));
        for (int past : {17, 4096}) {
            std::vector<int> keys(rows); std::vector<float> pos(rows);
            for (int j=0; j<owners; ++j) for (int t=0; t<lengths[j]; ++t) {
                pos[offsets[j]+t] = past+j*7+t;
                keys[offsets[j]+t] = int(pos[offsets[j]+t])+1;
            }
            Cuda(cudaMemcpy(positions.cudaData,pos.data(),positions.GetBytes(),cudaMemcpyHostToDevice));
            Cuda(cudaMemcpy(live.cudaData,keys.data(),live.GetBytes(),cudaMemcpyHostToDevice));
            Cuda(cudaGraphLaunch(exec,cudaStreamPerThread));
            const auto result=Read(output);
            for (int j=0; j<owners; ++j) {
                const int n=lengths[j], first=offsets[j], pitch=(n+3)/4*4;
                auto slice=[&](Data &dst, const Data &src, int width) {
                    auto shape=src.dims; shape[src.dims.size()==3 ? 1 : shape.size()-1] = n;
                    Init(dst,src.dataType,shape);
                    Cuda(cudaMemcpy(dst.cudaData,static_cast<char*>(src.cudaData)+size_t(first)*width*src.unitSize,
                                    dst.GetBytes(),cudaMemcpyDeviceToDevice));
                };
                Data h,q,pos,liveSlice,arena,single;
                slice(h,hidden,4096); slice(q,qkv,13568); slice(pos,positions,1); slice(liveSlice,live,1);
                Init(arena,INT32,{pitch*(cols+cols/32)/4});
                auto *ptr=static_cast<uint8_t*>(arena.cudaData);
                FastllmMxfp8Activation target{ptr,ptr+pitch*cols,n,cols,pitch};
                Require(FastllmCudaTryDpDSAVerify(h,q,wk,wq,ww,gamma,bias,sink,pos,liveSlice,
                    *refs[j]->K(),*refs[j]->V(),(const void *const*)refs[j]->table.cudaData,capacity,
                    x.theta,x.valueScale,target,reference,single),"native singleton reference");
                Require(Read(caches[j]->key)==Read(refs[j]->key) && Read(caches[j]->value)==Read(refs[j]->value),
                        "native pair cache mismatch");
                const auto expected=Read(single);
                const auto *actual=reinterpret_cast<const B*>(result.data())+size_t(first)*cols;
                const auto *baseline=reinterpret_cast<const B*>(expected.data());
                for(int r=0;r<n;++r) {
                    double norm=0,err=0;
                    for(int c=0;c<cols;++c) {
                        const int i=r*cols+c;
                        const double v=float(baseline[i]), delta=float(actual[i])-v;
                        norm+=v*v; err+=delta*delta;
                    }
                    Require(err <= 0.005*0.005*std::max(norm,1.e-30),"native pair relative L2");
                }
            }
        }
        Cuda(cudaGraphExecDestroy(exec)); Cuda(cudaGraphDestroy(graph));
        printf("Native BF16 DSA pair %d+%d PASS\n",lengths[0],lengths[1]);
        return;
    }
    Cuda(cudaMemset(live.cudaData, 0, live.GetBytes()));
    auto missingBinding = x.caches.back().cache;
    x.caches.back().cache = nullptr;
    Require(!FastllmCudaSm90VerifyDsaBatch(x, packed), "missing binding accepted");
    x.caches.back().cache = missingBinding;
    Require(FastllmCudaSm90VerifyDsaBatch(x, packed), "DSA warmup declined");
    Cuda(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
    Require(FastllmCudaSm90VerifyDsaBatch(x, packed), "DSA capture declined");
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
    Cuda(cudaGraphInstantiate(&exec, graph, 0));
    Data iq, ik, iw, merged, q, k, v, expected;
    Init(q, BFLOAT16, {1, rows, 12288});
    Init(k, BFLOAT16, {1, rows, 768});
    Init(v, BFLOAT16, {1, rows, 512});
    std::vector<Data> views(rows), selected(owners);
    const int lastPast = capacity - std::max(16, owners + 7);
    for (int past : {0, 2045, 4230, capacity - 32, lastPast, -1}) {
        // The projection is tied for the final long replay. Compare the entire ordered list.
        if (past == lastPast) Cuda(cudaMemset(ww.cudaData, 0, ww.GetBytes()));
        const int shift = past == 4230 ? 1 : 0;
        std::vector<float> pos(rows);
        std::vector<int> lengths(rows);
        std::vector<FastllmNaiveRopeCacheRow> binds;
        std::vector<FastllmNaiveVerifyIndexerItem> items;
        std::vector<FastllmNaiveDSARow> desc(rows);
        if (rows > 64) {
            // Validate the wide tier against the three independent BF16 projections.
            auto project = [&](Data &weight, Data &out) {
                int width = weight.dims[0];
                if (out.dims.empty()) Init(out, BFLOAT16, {1, rows, width});
                Require(FastllmCudaBFloat16MatMulBFloat16(hidden, weight, Data(), out, rows, 4096, width),
                        "reference BF16 projection");
            };
            project(wq, iq);
            project(wk, ik);
            project(ww, iw);
            if (past == 0 && rows == 128) {
                const auto packed = Read(work.dsaLayer);
                size_t offset = 0;
                for (const Data *part : {&iq, &ik, &iw}) {
                    const auto bytes = Read(*part);
                    const size_t width = bytes.size() / rows;
                    for (int r = 0; r < rows; ++r)
                        Require(!memcmp(packed.data() + r * (packed.size() / rows) + offset,
                                        bytes.data() + r * width, width), "packed Indexer BF16 mismatch");
                    offset += width;
                }
            }
            Require(FastllmCudaMul(iw, .25f, iw), "reference index weights");
        } else {
            Require(FastllmCudaIndexerProjection(hidden, wq, wk, ww, iq, ik, iw, merged),
                    "reference index projection");
        }
        Require(FastllmCudaNaiveIndexerKeyNorm(ik, gamma, bias), "reference index norm");
        for (int j = 0; j < owners; ++j) {
            const int owner = (j + shift) % owners;
            caches[j]->Bind(*caches[owner]);
            refs[j]->Bind(*refs[owner]);
            auto &r = *refs[j];
            auto binding = (const void *const *)r.table.cudaData;
            items.push_back({r.K(), &selected[j], offsets[j], x.requests[j], 0, capacity, binding});
            for (int t = 0; t < x.requests[j]; ++t) {
                int i = offsets[j] + t;
                pos[i] = past + t + j;
                lengths[i] = past < 0 || (past == 4230 && (j == 2 || t == 3)) ? 0 : (int)pos[i] + 1;
                View(views[i], ik, {1, 1, 128}, size_t(i) * 128 * 2);
                binds.push_back({r.K(), r.V(), &views[i], binding});
                auto &d = desc[i];
                d.cache = binding;
                d.liveKeys = (int *)live.cudaData + i;
                d.capacity = capacity;
                d.key = r.key.cudaData;
                d.value = r.value.cudaData;
                d.keyPages = (const int *)r.kp.cudaData;
                d.valuePages = (const int *)r.vp.cudaData;
                d.fp8 = storage == FP8_E4M3;
                d.keyStride = 896;
                d.pageLen = paged ? 128 : 0;
                d.selectedCount = 2048;
            }
        }
        Cuda(cudaMemcpy(live.cudaData, lengths.data(), live.GetBytes(), cudaMemcpyHostToDevice));
        Cuda(cudaMemcpy(positions.cudaData, pos.data(), positions.GetBytes(), cudaMemcpyHostToDevice));
        Require(FastllmCudaNaiveBatchRopeAppendCache(q, k, v, positions, binds, live, 64, 4, 192, 128, 64, x.theta,
                                                     x.valueScale, 0, &qkv),
                "reference append");
        FastllmCudaNaiveRope(iq, positions, 16, 128, 64, x.theta);
        Require(FastllmCudaNaiveBatchVerifyIndexer(iq, iw, items, true, reference, live), "reference top-k");
        for (int j = 0; j < owners; ++j)
            for (int t = 0; t < x.requests[j]; ++t)
                desc[offsets[j] + t].indices = (int *)selected[j].cudaData + t * 2048;
        FastllmCudaNaiveBatchDSADecode(q, sink, desc, 64, 4, 192, 128, reference, expected);
        Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
        Cuda(cudaDeviceSynchronize());
        const auto actual = Read(ids);
        for (int j = 0; j < owners; ++j) {
            Require(Read(caches[j]->key) == Read(refs[j]->key) && Read(caches[j]->value) == Read(refs[j]->value),
                    "DSA KV mismatch");
            auto ref = Read(selected[j]);
            Require(!memcmp(actual.data() + size_t(offsets[j]) * 2048 * sizeof(int), ref.data(), ref.size()),
                    "DSA ordered shortlist mismatch");
        }
        Require(Read(query) == Read(q), "DSA rotated Q mismatch");
        // The unfused dispatcher uses split-PV below its BF16 and FP8 parallelism thresholds.
        // Compare that entry row by row and use its wide gather as a bitwise oracle.
        if (storage == BFLOAT16 ? rows < 32 : rows * x.config.heads < 128) {
            const auto result = Read(output), small = Read(expected);
            const auto *act = reinterpret_cast<const B *>(result.data());
            const auto *ref = reinterpret_cast<const B *>(small.data());
            for (int r = 0; r < rows; ++r) {
                double error = 0, norm = 0;
                for (int c = 0; c < cols; ++c) {
                    const double v = float(ref[r * cols + c]), diff = float(act[r * cols + c]) - v;
                    norm += v * v;
                    error += diff * diff;
                }
                Require(error <= 0.005 * 0.005 * std::max(norm, 1.e-30), "DSA small-row relative L2 exceeded");
            }
            Data paddedQ, paddedOutput;
            Init(paddedQ, BFLOAT16, {1, 32, 12288});
            Cuda(cudaMemset(paddedQ.cudaData, 0, paddedQ.GetBytes()));
            Cuda(cudaMemcpy(paddedQ.cudaData, q.cudaData, q.GetBytes(), cudaMemcpyDeviceToDevice));
            auto paddedDesc = desc;
            paddedDesc.resize(32, desc.back());
            FastllmCudaNaiveBatchDSADecode(paddedQ, sink, paddedDesc, 64, 4, 192, 128, reference, paddedOutput);
            Require(!memcmp(result.data(), Read(paddedOutput).data(), result.size()), "DSA wide gather mismatch");
        } else Require(Read(output) == Read(expected), "DSA attention mismatch");
        auto encoded = Read(pack), values = Read(output);
        auto *raw = (const B *)values.data();
        for (int r = 0; r < rows; ++r)
            for (int col = 0; col < cols; col += 32) {
                float mx = rows > 16 ? 1.e-4f : 0.f;
                for (int j = 0; j < 32; ++j)
                    mx = std::max(mx, std::abs(float(raw[r * cols + col + j])));
                int exponent = mx > 0 ? std::clamp(int(std::ceil(std::log2(mx / 448.f))) + 127, 1, 254) : 1;
                Require(encoded[pitch * cols + (col / 128 * pitch + r) * 4 + col / 32 % 4] == exponent,
                        "DSA O exponent mismatch");
                for (int j = 0; j < 32; ++j)
                    if (encoded[r * cols + col + j] != __nv_fp8_e4m3(std::ldexp(float(raw[r * cols + col + j]), 127 - exponent)).__x) {
                        fprintf(stderr, "O owner=%d past=%d paged=%d row=%d col=%d j=%d value=%g mx=%g exp=%d ref=%02x got=%02x\n", owners,past,paged,r,col,j,float(raw[r*cols+col+j]),mx,exponent,__nv_fp8_e4m3(std::ldexp(float(raw[r*cols+col+j]),127-exponent)).__x,encoded[r*cols+col+j]);
                        Require(false,"DSA O value mismatch");
                    }
            }
        printf("DP%d paged=%d past=%d: KV, ordered top-k, BF16 and FP8 exact\n", rows, paged, past);
    }
    Cuda(cudaGraphExecDestroy(exec));
    Cuda(cudaGraphDestroy(graph));
}
} // namespace

int main() {
    try {
        FastllmCudaSetDevice(0);
        if (FastllmCudaRuntimeArch() != 90) return 77;
        for (int n : {1,2,3,7}) VerifyDsaRequests(1, false, FP8_E4M3, {n});
        for (auto lengths : {std::vector<int>{1,8}, {3,5}, {7,2}, {8,8}})
            VerifyDsaRequests(2, false, BFLOAT16, lengths, true);
        for (int owners : {1, 2, 4, 8, 16}) {
            VerifyDsaRequests(owners, false, FP8_E4M3);
            VerifyDsaRequests(owners, true, FP8_E4M3);
        }
        for (int owners : {4, 8, 16}) {
            VerifyDsaRequests(owners, false);
            VerifyDsaRequests(owners, true);
        }
        for (auto lengths : {std::vector<int>{3, 5, 1}, {7, 1, 8, 3}, {8, 8, 8}, {8, 7, 6, 5, 4, 3}}) {
            VerifyDsaRequests(lengths.size(), false, BFLOAT16, lengths);
            VerifyDsaRequests(lengths.size(), true, FP8_E4M3, lengths);
        }
        for (int rows : {65, 97, 127}) {
            std::vector<int> lengths((rows + 7) / 8, 8);
            lengths.back() = rows - 8 * (int(lengths.size()) - 1);
            VerifyDsaRequests(lengths.size(), false, FP8_E4M3, lengths);
        }
        puts("DSA verification PASS");
        return 0;
    } catch (const std::exception &e) {
        fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
