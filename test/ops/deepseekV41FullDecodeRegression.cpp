// GPU regression for a graph captured once and replayed with changing request
// lengths, masks and KV pointers. Requires one SM80+ GPU; returns 77 otherwise.
// The eager attention path is the independent reference. Pass --stress for
// larger activations. Both modes require bitwise equality with eager decode.
#include "fastllm.h"
#include "utils.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "devices/cuda/fastllm-v41-decode.cuh"
#include "devices/cuda/fastllm-v41-full-decode.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace fastllm;
static void Check(bool ok, const char *what) {
    if (!ok)
        throw std::runtime_error(what);
}
static void Cuda(cudaError_t e) {
    Check(e == cudaSuccess, cudaGetErrorString(e));
}
static float amplitude = .3f;
static bool stress = false;
static void Fill(Data &d, const std::vector<int> &dims, int seed,
                 DataType type = DataType::BFLOAT16) {
    size_t n = 1;
    for (int v : dims)
        n *= v;
    std::vector<float> values(n);
    for (size_t i = 0; i < n; ++i)
        values[i] = std::sin(float((i * 17 + seed * 13) % 1009)) * amplitude;
    if (type == DataType::INT32 || type == DataType::INT8) {
        d.dataType = type;
        d.Resize(dims);
        d.Allocate();
        std::memset(d.cpuData, 0, d.GetBytes());
        d.ToDevice(DataDevice::CUDA, std::vector<int>{0});
        return;
    }
    Data cpu(DataType::FLOAT32, dims, values);
    ToDataType(cpu, type);
    d.CopyFrom(cpu);
    d.ToDevice(DataDevice::CUDA, std::vector<int>{0});
}
static void View(Data &v, const Data &d, size_t offset, const std::vector<int> &dims) {
    v.FakeFrom(d, offset);
    v.expansionDims.clear();
    v.Resize(dims);
    v.expansionDims = dims;
    v.expansionSize = v.Count(0);
    v.expansionBytes = v.GetBytes();
}
static void CopyDevice(Data &dst, const Data &src) {
    Cuda(cudaMemcpyAsync(dst.cudaData, src.cudaData, src.GetBytes(), cudaMemcpyDeviceToDevice,
                         cudaStreamPerThread));
}
static void Equal(const Data &a, const Data &b, const char *name) {
    Check(a.GetBytes() == b.GetBytes(), "shape changed");
    std::vector<unsigned char> x(a.GetBytes()), y(b.GetBytes());
    Cuda(cudaMemcpy(x.data(), a.cudaData, x.size(), cudaMemcpyDeviceToHost));
    Cuda(cudaMemcpy(y.data(), b.cudaData, y.size(), cudaMemcpyDeviceToHost));
    if (x != y)
        throw std::runtime_error(std::string("bits changed: ") + name);
}
int main(int argc, char **argv) {
    int devices = 0, major = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 1 ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) != cudaSuccess ||
        major < 8)
        return 77;
    if (argc > 1 && std::string(argv[1]) == "--stress") {
        stress = true;
        amplitude = 1.f;
    }
    try {
        ApplyDeviceMap({{"cpu", 1}}, 0, 1);
        FastllmCudaSetDevice(0);
        int checked = 0;
        for (int batch : {1, 8, 128})
            for (int ratio : {0, 1, 2})
                for (int maskMode : {0, 1, 2}) {
                    if ((!ratio && maskMode) || (stress && batch != 1))
                        continue;
                    const int maxKeys = batch == 128 ? 2048 : 32768;
                    FastllmV41FullDecodeConfig c;
                    c.batch = batch;
                    c.ratio = ratio;
                    c.maxKeys = maxKeys;
                    c.indexSource = ratio != 0;
                    c.writeCandidates = maskMode == 1;
                    c.useCandidates = maskMode == 2;
                    c.candidateTopK = 16;
                    c.softmaxScale = 1 / std::sqrt(512.f);
                    const int maxBlocks = (maxKeys + c.blockSize - 1) / c.blockSize;
                    Data q, kv, out, iq, iw, sink, top, cand, q0, kv0, iq0, qr, kr, orr, iqr, tr,
                        cr;
                    Fill(q, {1, batch, 8, 512}, 1);
                    Fill(kv, {1, batch, 512}, 2);
                    Fill(out, {1, batch, 8, 512}, 3);
                    Fill(iq, {1, batch, 32, 128}, 4);
                    Fill(iw, {1, batch, 32}, 5, DataType::FLOAT32);
                    Fill(sink, {8}, 6, DataType::FLOAT32);
                    Fill(top, {1, batch, 512}, 7, DataType::INT32);
                    Fill(cand, {1, batch, maxBlocks}, 8, DataType::INT8);
                    for (auto pair : {std::pair<Data *, Data *>{&q0, &q},
                                      {&kv0, &kv},
                                      {&iq0, &iq},
                                      {&qr, &q},
                                      {&kr, &kv},
                                      {&orr, &out},
                                      {&iqr, &iq},
                                      {&tr, &top},
                                      {&cr, &cand}}) {
                        pair.first->CopyFrom(*pair.second);
                        pair.first->ToDevice(DataDevice::CUDA, std::vector<int>{0});
                    }
                    std::vector<Data> rings(batch), relocatedRings(batch), cmps(batch), keys(batch),
                        rrings(batch);
                    std::vector<FastllmV41FullDecodeRequest> requests(batch);
                    for (int i = 0; i < batch; ++i) {
                        Fill(rings[i], {1, 128, 512}, 10 + i);
                        Fill(relocatedRings[i], {1, 128, 512}, 100 + i);
                        rrings[i].CopyFrom(rings[i]);
                        rrings[i].ToDevice(DataDevice::CUDA, std::vector<int>{0});
                        if (ratio) {
                            Fill(cmps[i], {1, maxKeys, 512}, 20 + i);
                            Fill(keys[i], {1, maxKeys, 128}, 30 + i);
                        }
                        requests[i] = {rings[i].cudaData,
                                       ratio ? cmps[i].cudaData : nullptr,
                                       ratio ? keys[i].cudaData : nullptr,
                                       nullptr,
                                       nullptr,
                                       1};
                    }
                    FastllmV41FullDecodeBuffers b;
                    b.q = q.cudaData;
                    b.kv = kv.cudaData;
                    b.output = out.cudaData;
                    b.indexQ = iq.cudaData;
                    b.indexWeights = iw.cudaData;
                    b.topk = top.cudaData;
                    b.candidates = cand.cudaData;
                    b.sink = sink.cudaData;
                    auto unsupported = c;
                    unsupported.ratio = 3;
                    Check(!FastllmCudaV41CreateFullDecode(unsupported, b),
                          "unsupported ratio accepted");
                    void *plan = FastllmCudaV41CreateFullDecode(c, b);
                    Check(plan, "plan create");
                    Check(!FastllmCudaV41UploadFullDecode(plan, requests.data(), batch + 1),
                          "wrong request count accepted");
                    auto invalid = requests;
                    invalid[0].position = -1;
                    Check(!FastllmCudaV41UploadFullDecode(plan, invalid.data(), batch),
                          "negative position accepted");
                    Check(FastllmCudaV41UploadFullDecode(plan, requests.data(), batch),
                          "initial upload");
                    Cuda(cudaMemsetAsync(cand.cudaData, 1, cand.GetBytes(), cudaStreamPerThread));
                    Check(FastllmCudaV41PrepareFullDecode(plan) &&
                              FastllmCudaV41FinishFullDecode(plan),
                          "warmup");
                    Cuda(cudaDeviceSynchronize());
                    cudaGraph_t graph;
                    cudaGraphExec_t exec;
                    Cuda(cudaStreamBeginCapture(cudaStreamPerThread,
                                                cudaStreamCaptureModeThreadLocal));
                    Check(FastllmCudaV41PrepareFullDecode(plan) &&
                              FastllmCudaV41FinishFullDecode(plan),
                          "capture");
                    Cuda(cudaStreamEndCapture(cudaStreamPerThread, &graph));
                    Cuda(cudaGraphInstantiate(&exec, graph, 0));
                    for (int position :
                         {1, 126, 127, 510, 511, 1022, 1023, 2046, 8192, 16384, 32766}) {
                        if ((position + batch + 1) / (ratio ? ratio : 1) > maxKeys)
                            continue;
                        CopyDevice(q, q0);
                        CopyDevice(kv, kv0);
                        CopyDevice(iq, iq0);
                        CopyDevice(qr, q0);
                        CopyDevice(kr, kv0);
                        CopyDevice(iqr, iq0);
                        Cuda(cudaMemsetAsync(cand.cudaData, 1, cand.GetBytes(),
                                             cudaStreamPerThread));
                        CopyDevice(cr, cand);
                        std::vector<Data> vq(batch), vk(batch), vo(batch), viq(batch), viw(batch),
                            vkeys(batch), vtop(batch), vcand(batch), vcmp(batch);
                        std::vector<FastllmCudaV41DecodeTransform> trans, finish;
                        std::vector<FastllmCudaV41DecodeIndexTask> idx;
                        std::vector<FastllmCudaV41AttentionRequest> att;
                        for (int i = 0; i < batch; ++i) {
                            const int pos = position + i, count = ratio ? (pos + 1) / ratio : 0,
                                      blocks = (count + c.blockSize - 1) / c.blockSize;
                            requests[i].position = pos;
                            Data &ring = position >= 510 ? relocatedRings[i] : rings[i];
                            Check(rings[i].cudaData != relocatedRings[i].cudaData,
                                  "ring relocation reused address");
                            requests[i].windowKV = ring.cudaData;
                            CopyDevice(rrings[i], ring);
                            View(vq[i], qr, (size_t)i * 8 * 512 * 2, {1, 1, 8, 512});
                            View(vk[i], kr, (size_t)i * 512 * 2, {1, 1, 512});
                            View(vo[i], orr, (size_t)i * 8 * 512 * 2, {1, 1, 8, 512});
                            trans.push_back({&vq[i], nullptr, pos, 0, false});
                            trans.push_back({&vk[i], nullptr, pos, 1, false});
                            if (ratio) {
                                View(viq[i], iqr, (size_t)i * 32 * 128 * 2, {1, 1, 32, 128});
                                View(viw[i], iw, (size_t)i * 32 * 4, {1, 1, 32});
                                View(vkeys[i], keys[i], 0, {1, count, 128});
                                View(vcmp[i], cmps[i], 0, {1, count, 512});
                                View(vtop[i], tr, (size_t)i * 512 * 4,
                                     {1, 1, std::min(512, count)});
                                View(vcand[i], cr, (size_t)i * maxBlocks, {1, 1, blocks});
                                idx.push_back({&viq[i], &viw[i], &vkeys[i], &vcand[i], &vtop[i],
                                               pos, c.writeCandidates, c.useCandidates});
                            }
                            att.push_back({&vq[i], &vk[i], &rrings[i], ratio ? &vcmp[i] : nullptr,
                                           ratio ? &vtop[i] : nullptr, &vo[i], pos});
                            finish.push_back({&vo[i], nullptr, pos, 1, true});
                            finish.push_back({&vk[i], &rrings[i], pos, 0, false});
                        }
                        Check(FastllmCudaV41DecodeTransformsBatch(trans, c.ropeDim, c.ropeBase,
                                                                  c.originalSeqLen, c.ropeFactor,
                                                                  c.betaFast, c.betaSlow, c.window),
                              "reference transform");
                        if (ratio)
                            Check(FastllmCudaV41DecodeIndexerBatch(
                                      idx, ratio, c.topK, c.blockSize, c.candidateTopK, c.ropeDim,
                                      c.ropeBase, c.originalSeqLen, c.ropeFactor, c.betaFast,
                                      c.betaSlow, true),
                                  "reference indexer");
                        Check(FastllmCudaDeepSeekV41SparseAttentionBatch(att, sink, c.window,
                                                                         c.softmaxScale),
                              "reference attention");
                        Check(FastllmCudaV41DecodeTransformsBatch(finish, c.ropeDim, c.ropeBase,
                                                                  c.originalSeqLen, c.ropeFactor,
                                                                  c.betaFast, c.betaSlow, c.window),
                              "reference finish");
                        Check(FastllmCudaV41UploadFullDecode(plan, requests.data(), batch),
                              "runtime upload");
                        Cuda(cudaGraphLaunch(exec, cudaStreamPerThread));
                        Cuda(cudaDeviceSynchronize());
                        Equal(q, qr, "q");
                        Equal(kv, kr, "kv");
                        Equal(out, orr, "attention output");
                        if (ratio) {
                            Equal(iq, iqr, "index query");
                            for (int i = 0; i < batch; ++i) {
                                Data actual;
                                View(actual, top, (size_t)i * 512 * 4, vtop[i].dims);
                                Equal(actual, vtop[i], "top-k");
                            }
                        }
                        for (int i = 0; i < batch; ++i)
                            Equal(position >= 510 ? relocatedRings[i] : rings[i], rrings[i],
                                  "ring");
                        checked += batch;
                    }
                    Cuda(cudaGraphExecDestroy(exec));
                    Cuda(cudaGraphDestroy(graph));
                    FastllmCudaV41DestroyFullDecode(plan);
                    std::cout << "batch=" << batch << " ratio=" << ratio << " mask=" << maskMode
                              << " cumulative=" << checked << std::endl;
                }
        std::cout << "PASS " << checked << " changing-context graph replays; stress=" << stress
                  << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
