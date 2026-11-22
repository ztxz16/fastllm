#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#endif
#include "utils/cuda_decode_sampling.h"
#include "models/mimo_tool_grammar.h"
#include <cuda_runtime_api.h>
#include <iostream>
#include <set>
#include <cmath>
#include <thread>
#include <stdexcept>
#include <limits>
#ifdef FASTLLM_TEST_DECODE_SAMPLING_HOOKS
#include <dlfcn.h>
#include <atomic>
enum class CaptureFailure { None, Begin, SecondSlot };
static CaptureFailure captureFailure = CaptureFailure::None;
static std::atomic<int> captureCalls{0}, instantiateCalls{0};
static void LeaveCudaError() {
    cudaLaunchKernel(nullptr, dim3(1), dim3(1), nullptr, 0, cudaStreamPerThread);
}
extern "C" bool FastllmCudaGraphBeginCapture() {
    ++captureCalls;
    if (captureFailure == CaptureFailure::Begin) { LeaveCudaError(); return false; }
    static auto fn = (bool (*)())dlsym(RTLD_NEXT, "FastllmCudaGraphBeginCapture");
    return fn();
}
extern "C" bool FastllmCudaGraphInstantiate(void *graph, void **exec) {
    ++instantiateCalls;
    if (captureFailure == CaptureFailure::SecondSlot && instantiateCalls == 2) {
        LeaveCudaError(); *exec = nullptr; return false;
    }
    static auto fn = (bool (*)(void *, void **))dlsym(RTLD_NEXT, "FastllmCudaGraphInstantiate");
    return fn(graph, exec);
}
#endif
using namespace fastllm;
static void Check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }

static void SoftmaxWorkspaceChecks() {
    Check(FastllmCudaSamplingSoftmaxWorkspaceSize(0, 152576) == 0 &&
          FastllmCudaSamplingSoftmaxWorkspaceSize(65, 152576) == 0 &&
          FastllmCudaSamplingSoftmaxWorkspaceSize(1, 4096) == 0, "softmax fallback admission");
    for (int rows : {1,3,8,65}) for (int vocab : {4096,65536,152577}) {
        struct Buffers {
            std::vector<void *> pointers;
            void *Make(size_t bytes) { void *p=nullptr; Check(cudaMalloc(&p,bytes)==cudaSuccess,"test allocation"); pointers.push_back(p); return p; }
            ~Buffers() { for(auto p:pointers) cudaFree(p); }
        } buffers;
        size_t count=(size_t)rows*vocab, bytes=count*sizeof(float);
        std::vector<float> logits(count), temperatures(rows), topPs(rows,.95f), reference(count), result(count);
        std::vector<int> topKs(rows,vocab), tokens(rows);
        std::vector<uint64_t> seeds(rows,123), offsets(rows,0);
        for(int r=0;r<rows;++r) {
            temperatures[r]=r%3==0?.4f:r%3==1?1.f:2.f;
            for(int v=0;v<vocab;++v) logits[(size_t)r*vocab+v]=v%97==0?-INFINITY:3.f*std::sin(v*.071f+r);
        }
        auto upload=[&](const void *host,size_t n) {void *p=buffers.Make(n);Check(cudaMemcpy(p,host,n,cudaMemcpyHostToDevice)==cudaSuccess,"test upload");return p;};
        auto *x=(float*)upload(logits.data(),bytes), *y=(float*)buffers.Make(bytes);
        auto *ts=(float*)upload(temperatures.data(),rows*4), *ps=(float*)upload(topPs.data(),rows*4);
        auto *ks=(int*)upload(topKs.data(),rows*4), *out=(int*)buffers.Make(rows*4);
        auto *fo=(float*)buffers.Make(rows*4);
        auto *rs=(uint64_t*)upload(seeds.data(),rows*8), *ro=(uint64_t*)upload(offsets.data(),rows*8);
        const size_t scratchBytes=FastllmCudaSamplingSoftmaxWorkspaceSize(rows,vocab);
        void *scratch=buffers.Make(std::max<size_t>(scratchBytes+8,16));
        for(int mode=0;mode<(rows==1 && vocab==65536?5:2);++mode) {
            Check(cudaMemcpy(ro,offsets.data(),rows*8,cudaMemcpyHostToDevice)==cudaSuccess,"offset reset");
            bool ok=mode==0 ? FastllmCudaTopKTopPSamplingToDeviceWithRng(x,y,ts,ks,ps,nullptr,nullptr,0,out,fo,rows,vocab,rs,ro) :
                FastllmCudaTopKTopPSamplingWithWorkspace(x,y,ts,ks,ps,nullptr,nullptr,0,out,fo,rows,vocab,rs,ro,
                    mode==4?nullptr:mode==3?(char*)scratch+1:scratch, mode==2?1:scratchBytes);
            Check(ok && cudaStreamSynchronize(cudaStreamPerThread)==cudaSuccess,"workspace sampling launch");
            Check(cudaMemcpy(result.data(),y,bytes,cudaMemcpyDeviceToHost)==cudaSuccess,"probability readback");
            Check(cudaMemcpy(tokens.data(),out,rows*4,cudaMemcpyDeviceToHost)==cudaSuccess,"token readback");
            if(mode==0) reference=result;
            for(int r=0;r<rows;++r) {
                double err=0,scale=0,sum=0;
                for(int v=0;v<vocab;++v) {
                    size_t i=(size_t)r*vocab+v;double a=result[i],b=reference[i];
                    Check(std::isfinite(a)&&a>=0,"invalid probability");
                    err+=(a-b)*(a-b);scale+=b*b;sum+=a;
                }
                Check(std::sqrt(err/scale)<5e-6 && std::abs(sum-1)<5e-6,"tiled softmax probability error");
                Check(tokens[r]>=0 && tokens[r]<vocab,"workspace sampler token range");
            }
            if(mode>=2 || !scratchBytes) Check(result==reference,"workspace fallback changed arithmetic");
        }
    }
}
static void MixedSoftmaxBuckets() {
    // The largest bucket uses the legacy softmax while the smaller one uses
    // tiled softmax. Both must be ready before any request is submitted.
    constexpr int rows = 65, vocab = 65536, expectedToken = 17;
    std::vector<float> values(rows * vocab, -100.f);
    for (int row = 0; row < rows; ++row) values[row * vocab + expectedToken] = 100.f;
    Data logits(FLOAT32, {rows, vocab}, values);
    logits.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
    CudaDecodeSampling sampler;
    Check(sampler.Prepare(logits, {1, rows}, 12345), "mixed softmax bucket prepare");
#ifdef FASTLLM_TEST_DECODE_SAMPLING_HOOKS
    const int captures = captureCalls.load(), instantiates = instantiateCalls.load();
#endif
    int step = 0;
    for (int batch : {1, rows, 1}) {
        std::vector<CudaDecodeSampling::Request> requests;
        for (int row = 0; row < batch; ++row)
            requests.push_back({uint64_t(row + 1), uint64_t(step), 1.f, .95f, vocab});
        const int slot = step++ % 2;
        Check(sampler.Submit(requests, slot), "mixed softmax bucket submit");
        for (int token : sampler.Collect(slot)) Check(token == expectedToken, "mixed softmax bucket result");
    }
#ifdef FASTLLM_TEST_DECODE_SAMPLING_HOOKS
    Check(captureCalls == captures && instantiateCalls == instantiates, "runtime softmax graph recapture");
#endif
}

static void ConcurrentSamplers(int vocab) {
    std::exception_ptr failures[2];
    auto run=[&](int worker) {try {
        FastllmCudaSetDevice(0);
        std::vector<float> values(3*vocab,-100.f);
        for(int r=0;r<3;++r) values[r*vocab+17+worker]=100.f;
        Data logits(FLOAT32,{3,vocab},values);logits.ToDevice(DataDevice::CUDA,std::vector<int>{0},true);
        CudaDecodeSampling sampler;
        Check(sampler.Prepare(logits,{1,2,3},12345+worker),"concurrent sampler prepare");
        std::vector<CudaDecodeSampling::Request> requests;
        for(int r=0;r<3;++r) requests.push_back({uint64_t(100+worker*10+r),0,.4f,.95f,vocab});
        for(int step=0;step<16;++step) {
            Check(sampler.Submit(requests,step%2),"concurrent submit");
            for(int token:sampler.Collect(step%2)) Check(token==17+worker,"sampler scratch crossed instances");
            for(auto &r:requests) ++r.step;
        }
    }catch(...) {failures[worker]=std::current_exception();}};
    std::thread a(run,0),b(run,1);a.join();b.join();
    for(auto e:failures) if(e) std::rethrow_exception(e);
}
static void ConstraintSamplingChecks(int vocab, bool sequentialTopK) {
    constexpr int capacity = 4;
    auto words = std::make_shared<std::vector<std::string>>(513, "z");
    // Empty padding produces wholly masked tiles in the large-vocabulary case.
    words->resize(vocab);
    for (int i = 0; i < 256; ++i) (*words)[i] = std::string(1, char(i));
    const std::vector<std::string> extra = {"<tool_call>", "<function=A>", "<parameter=x>",
        "</parameter></function></tool_call>", "<eos>", "<think>", "</think>", "<function=B>", ""};
    std::copy(extra.begin(), extra.end(), words->begin() + 256);
    auto vocabulary = std::make_shared<MimoToolVocabulary>(words);
    MimoToolGrammar a({{"A", {"x"}}}, false, true), b({{"B", {"x"}}}, false, true);
    a.Bind(vocabulary, {260}); b.Bind(vocabulary, {260});
    std::vector<MimoToolGrammar *> grammars{&a, &b, nullptr};
    std::vector<int> states(3, 0);
    std::vector<uint64_t> steps(3, 0);
    Data actual(FLOAT32, {capacity, vocab}, std::vector<float>(capacity * vocab, 0)), reference(actual);
    actual.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
    reference.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
    CudaDecodeSampling gpu, cpu;
    Check(gpu.Prepare(actual, {1, 2, capacity}, 853, false, sequentialTopK) &&
          cpu.Prepare(reference, {1, 2, capacity}, 853, false, sequentialTopK), "grammar sampler prepare");
#ifdef FASTLLM_TEST_DECODE_SAMPLING_HOOKS
    const int captures = captureCalls;
#endif
    // Each pair is submitted before publishing either result. The reference
    // applies CPU grammar masks and advances only its own committed states.
    for (int pair = 0; pair < 24; ++pair) {
        std::vector<int> order = pair < 8 ? std::vector<int>{0, 1, 2} :
            pair < 16 ? std::vector<int>{2, 1, 0} : std::vector<int>{1};
        if (pair == 20) { states[1] = b.Advance(0, "<tool_call><function="); steps[1] = 0; }
        std::vector<CudaDecodeSampling::Request> requests;
        for (int id : order) {
            auto p = grammars[id] ? grammars[id]->Compile(states[id]) :
                std::make_pair(std::shared_ptr<const TokenConstraintProgram>(), 0);
            requests.push_back({uint64_t(100 + id), steps[id], .7f, .95f, vocab, p.first, p.second});
        }
        std::vector<int> expected[2];
        for (int slot = 0; slot < 2; ++slot) {
            std::vector<float> values(capacity * vocab), masked;
            for (int i = 0; i < capacity * vocab; ++i) values[i] = std::sin(i * .071f + pair);
            for (size_t row = 0; row < order.size(); ++row) {
                const int id = order[row], path[] = {256, id == 0 ? 257 : 263, 258, int('x'), 259, 260, 261, 262};
                values[row * vocab + path[(pair * 2 + slot) % 8]] = 10;
                values[row * vocab + 264] = 20; // Empty token: illegal only in constrained rows.
            }
            masked = values;
            for (size_t row = 0; row < order.size(); ++row) {
                const int id = order[row];
                if (!grammars[id]) continue;
                const auto mask = grammars[id]->Mask(states[id]);
                for (int token = 0; token < vocab; ++token)
                    if (!((*mask)[token / 32] & (uint32_t(1) << (token % 32))))
                        masked[row * vocab + token] = -std::numeric_limits<float>::infinity();
            }
            FastllmCudaCopyFromHostToDevice(actual.cudaData, values.data(), values.size() * sizeof(float));
            FastllmCudaCopyFromHostToDevice(reference.cudaData, masked.data(), masked.size() * sizeof(float));
            // Change sampling parameters with the first GPU ticket still outstanding.
            // The constrained survivors must keep their device-advanced cursors.
            if (slot == 1) for (auto &request : requests) request.temperature = .4f;
            auto oracle = requests;
            for (auto &r : oracle) r.constraint.reset();
            Check(gpu.Submit(requests, slot) && cpu.Submit(oracle, slot), "grammar lookahead submit");
            expected[slot] = cpu.Collect(slot);
            for (size_t row = 0; row < order.size(); ++row) {
                const int id = order[row];
                if (grammars[id]) {
                    states[id] = grammars[id]->AdvanceToken(states[id], expected[slot][row]);
                    Check(states[id] >= 0, "CPU oracle selected illegal grammar token");
                }
                ++steps[id]; ++requests[row].step;
                // Deliberately leave the GPU request's CPU cursor behind.
            }
        }
        for (int slot = 0; slot < 2; ++slot)
            Check(gpu.Collect(slot) == expected[slot], "GPU grammar changed sampled token/RNG/state");
    }
#ifdef FASTLLM_TEST_DECODE_SAMPLING_HOOKS
    Check(captureCalls == captures, "grammar sampler captured at runtime");
#endif
}

static void InPlaceSamplingChecks() {
    constexpr int capacity = 65, vocab = 65536;
    std::vector<float> values((size_t)capacity * vocab);
    for (size_t i = 0; i < values.size(); ++i) values[i] = 3.f * std::sin(i * .071f);
    Data input(FLOAT32, {capacity, vocab}, values), reference(FLOAT32, {capacity, vocab}, values);
    input.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
    reference.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
    CudaDecodeSampling consumed, preserved;
    Check(consumed.Prepare(input, {1, 3, capacity}, 4321, true, true) &&
          preserved.Prepare(reference, {1, 3, capacity}, 4321, false, true), "in-place sampler prepare");
#ifdef FASTLLM_TEST_DECODE_SAMPLING_HOOKS
    const int captures = captureCalls;
#endif
    int step = 0;
    for (int rows : {1, 3, capacity, 3, 1, capacity}) {
        FastllmCudaCopyFromHostToDevice(input.cudaData, values.data(), values.size() * sizeof(float));
        std::vector<CudaDecodeSampling::Request> requests;
        for (int row = 0; row < rows; ++row)
            requests.push_back({uint64_t(1000 + rows - row), uint64_t(step), .4f, .95f, row % 3 == 0 ? 1 : row % 3 == 1 ? 50 : vocab});
        const int slot = step++ % 2;
        Check(consumed.Submit(requests, slot) && preserved.Submit(requests, slot), "in-place sampler submit");
        Check(consumed.Collect(slot) == preserved.Collect(slot), "in-place sampling changed token/RNG");
    }
#ifdef FASTLLM_TEST_DECODE_SAMPLING_HOOKS
    Check(captureCalls == captures, "in-place sampler recaptured on batch change");
#endif
}

static void ShardedSamplingChecks(bool exceptional = false) {
    constexpr int cap = FastllmCudaLogitsTopKCapacity, batch = 7, vocab = 4099, shards = 3;
    const std::vector<int> split{0, 511, 2537, vocab};
    std::vector<float> score(batch * vocab);
    for (int row = 0; row < batch; ++row) for (int id = 0; id < vocab; ++id) {
        score[row * vocab + id] = row == 0 ? 3.f : float((id * 77 + row * 13) % 193) * .017f;
        if (row == 1 && id < split[1]) score[row * vocab + id] = -INFINITY;
    }
    if (exceptional) {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        std::fill(score.begin(), score.end(), -1.f);
        // Invalid compact column zero must map to global zero. A single NaN
        // can also leave a sampleable distribution when k exceeds its count.
        score[3] = nan;
        score[39] = 60.f;
        score[45] = 100.f;
        score[vocab + split[2] + 39] = 20.f;
        score[vocab + split[2] + 45] = 100.f; // overflows only after scaling
        std::fill_n(score.begin() + 2 * vocab, vocab, -INFINITY);
        score[3 * vocab + split[2]] = INFINITY;
        // Preserve enough tied numeric candidates even when NaNs also occupy
        // the radix top-k. Exercise both local and cross-shard NaN counts.
        std::fill_n(score.begin() + 4 * vocab + split[1], cap - 1, nan);
        std::fill_n(score.begin() + 4 * vocab + split[1] + cap, 2 * cap, 4.f);
        score[5 * vocab + 5] = nan;
        score[5 * vocab + split[1] + 10] = -nan;
        score[5 * vocab + split[2] + 20] = 50.f;
        std::fill_n(score.begin() + 6 * vocab, vocab, nan);
    }
    Data dense(FLOAT32, {batch, vocab}, score), expanded(dense);
    constexpr int shardWords = sizeof(FastllmCudaTopKShard) / sizeof(int);
    Data packed(INT32, {batch, shards, shardWords}), selected(INT32, {batch, shardWords});
    Data scratch(INT8, {int(FastllmCudaLogitsTopKWorkspaceBytes(batch, 2026))});
    for (Data *d : {&dense, &expanded, &packed, &selected, &scratch}) {
        d->ToDevice(CUDA, {0}, true); d->Allocate();
    }
    FastllmCudaMemset0(packed.cudaData, packed.GetBytes());
    auto *inbox = (FastllmCudaTopKShard *)packed.cudaData;
    CudaDecodeSampling full, compact;
    Check(full.Prepare(dense, {3, batch}, 971, false, true) &&
          compact.Prepare(expanded, {3, batch}, 971, true, true, inbox, shards), "sharded sampler prepare");
#ifdef FASTLLM_TEST_DECODE_SAMPLING_HOOKS
    const int captures = captureCalls;
#endif
    for (int step = 0; step < 20; ++step) {
        constexpr int ks[] = {1, 3, 17, 50, cap};
        const int rows = step % 3 ? batch : 2;
        std::vector<CudaDecodeSampling::Request> requests;
        for (int row = 0; row < rows; ++row)
            requests.push_back({uint64_t(row), uint64_t(step), .5f + .127f * row,
                                row % 2 ? 1.f : .83f, ks[(row + step) % 5]});
        if (exceptional) {
            requests[1].temperature = 1e-37f;
            requests[1].topK = 2;
            if (rows == batch) requests[4].topK = cap;
        }
        for (int rank = 0; rank < shards; ++rank) {
            std::vector<FastllmCudaTopKRow> inputs;
            std::vector<FastllmCudaTopKShard *> destinations;
            for (int row = 0; row < rows; ++row) {
                inputs.push_back({(float *)dense.cudaData + row * vocab + split[rank], 1.f / requests[row].temperature});
                destinations.push_back(inbox + row * shards + rank);
            }
            Check(FastllmCudaDispatchLogitsTopK(inputs.data(), destinations.data(), rows, split[rank+1]-split[rank],
                split[rank], (FastllmCudaTopKShard *)selected.cudaData, scratch.cudaData, scratch.GetBytes()),
                "shard top-k dispatch");
        }
        const bool useShards = step % 5 != 4;
        if (!useShards) {
            requests[0].topK = cap + 1;
            Check(!compact.Submit(requests, step % 2, true), "unsupported sharded top-k advanced sampling");
            FastllmCudaCopyFromHostToDevice(expanded.cudaData, score.data(), score.size() * sizeof(float));
        }
        Check(full.Submit(requests, step % 2) && compact.Submit(requests, step % 2, useShards), "sharded sampler submit");
        const auto reference = full.Collect(step % 2);
        Check(reference == compact.Collect(step % 2), "sharded sampler changed seeded tokens");
        if (exceptional && useShards) {
            Check(reference[1] == 0, "expected no sampleable mass after temperature overflow");
            if (requests[0].topK == 1) Check(reference[0] == 0, "expected NaN radix cutoff");
            if (requests[0].topK == 3) Check(reference[0] == 45, "valid candidate after NaN cutoff");
        }
    }
#ifdef FASTLLM_TEST_DECODE_SAMPLING_HOOKS
    Check(captureCalls == captures, "sharded sampler recaptured on input change");
#endif
}
static void SequentialSamplingChecks(int vocab) {
    const int rows = 8;
    std::vector<float> raw(size_t(rows) * vocab), ts(rows), ps(rows);
    std::vector<int> ks{1, 4, 50, 129, vocab, 1, 50, vocab};
    for (int row=0; row<rows; ++row) {
        ts[row]=.7f + row*.1f; ps[row]=row>=5 ? 1.f : .83f;
        for (int id=0; id<vocab; ++id)
            raw[size_t(row)*vocab+id]=row>=5 ? 0.f : -.017f * ((int64_t(id)*73)%vocab);
    }
    Data input(FLOAT32,{rows,vocab},raw), probs(FLOAT32,{rows,vocab});
    Data temps(FLOAT32,{rows},ts), topPs(FLOAT32,{rows},ps), topKs(INT32,{rows});
    topKs.Allocate();std::memcpy(topKs.cpuData,ks.data(),rows*sizeof(int));
    Data output(INT32,{rows}), floats(FLOAT32,{rows});
    Data scratch(INT8,{int(FastllmCudaTopKFirstSamplingWorkspaceBytes(rows))});
    for(auto *d:{&input,&probs,&temps,&topPs,&topKs,&output,&floats,&scratch}) {
        d->ToDevice(CUDA,std::vector<int>{0},true);d->Allocate();
    }
    Check(FastllmCudaTopKFirstSamplingToDeviceWithRng((float*)input.cudaData,(float*)probs.cudaData,
        (float*)temps.cudaData,(int*)topKs.cudaData,(float*)topPs.cudaData,nullptr,nullptr,0,
        (int*)output.cudaData,(float*)floats.cudaData,rows,vocab,nullptr,nullptr,
        scratch.cudaData,scratch.GetBytes()),"sequential device sampler");
    std::vector<float> actual(size_t(rows)*vocab), preserved(actual.size());
    FastllmCudaCopyFromDeviceToHost(actual.data(),probs.cudaData,actual.size()*sizeof(float));
    FastllmCudaCopyFromDeviceToHost(preserved.data(),input.cudaData,preserved.size()*sizeof(float));
    Check(preserved==raw,"out-of-place sampler mutated logits");
    for(int row=0;row<rows;++row) {
        std::vector<std::pair<float,int>> ranked;
        for(int id=0;id<vocab;++id)ranked.push_back({raw[size_t(row)*vocab+id]*(1.f/ts[row]),id});
        std::sort(ranked.begin(),ranked.end(),[](auto a,auto b){return a.first>b.first || (a.first==b.first&&a.second<b.second);});
        std::vector<float> expected(vocab,0);double sum=0;
        for(int j=0;j<ks[row];++j)sum+=std::exp(ranked[j].first-ranked[0].first);
        for(int j=0;j<ks[row];++j)expected[ranked[j].second]=std::exp(ranked[j].first-ranked[0].first)/sum;
        for(int id=0;id<vocab;++id)Check(std::abs(actual[size_t(row)*vocab+id]-expected[id])<3e-6,"top-k normalized probabilities/tie ordering differ");
    }
    // Empirical top-k-then-top-p support, using a CPU sorting oracle and many
    // independent rows. No changing sampler parameters to hide finite-k cases.
    constexpr int repeats=4096, width=257, groups=5;
    std::vector<float> batch(size_t(repeats)*groups*width), temperatures(repeats*groups), topP(repeats*groups);
    std::vector<int> topK(repeats*groups), tokens(repeats*groups);
    std::vector<std::vector<float>> expected(groups,std::vector<float>(width));
    for(int group=0;group<groups;++group) {
        const int k=group==0?1:group==1?4:group==2?50:group==3?129:width;
        const float temp=.5f+.2f*group,p=group%2?.91f:1.f;
        std::vector<std::pair<float,int>> ranked;
        for(int id=0;id<width;++id)ranked.push_back({-.035f*((id*73)%width)*(1.f/temp),id});
        std::sort(ranked.begin(),ranked.end(),[](auto a,auto b){return a.first>b.first;});
        double total=0;for(int i=0;i<k;++i)total+=std::exp(ranked[i].first-ranked[0].first);
        double mass=0;int keep=0;
        do {mass+=std::exp(ranked[keep].first-ranked[0].first)/total;++keep;} while(keep<k&&mass<=p);
        for(int i=0;i<keep;++i)expected[group][ranked[i].second]=std::exp(ranked[i].first-ranked[0].first)/total/mass;
        for(int i=0;i<repeats;++i) {
            const int row=group*repeats+i;temperatures[row]=temp;topP[row]=p;topK[row]=k;
            for(int id=0;id<width;++id)batch[size_t(row)*width+id]=-.035f*((id*73)%width);
        }
    }
    Data all(FLOAT32,{groups*repeats,width},batch);all.ToDevice(CUDA,std::vector<int>{0},true);
    Check(FastllmCudaTopKFirstSampling((float*)all.cudaData,temperatures.data(),topK.data(),topP.data(),tokens.data(),groups*repeats,width),"batched sequential host sampler");
    for(int group=0;group<groups;++group) {
        std::vector<int> count(width);
        for(int i=0;i<repeats;++i) {
            int token=tokens[group*repeats+i];Check(token>=0&&token<width&&expected[group][token]>0,"draw outside sequential nucleus");++count[token];
        }
        for(int id=0;id<width;++id)Check(std::abs(double(count[id])/repeats-expected[group][id])<.035,"sequential sampling distribution mismatch");
    }
    std::cout<<"TOP-K-FIRST SAMPLING PASS vocab="<<vocab<<"\n";
}

int main(int argc, char **argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) { std::cout << "FASTLLM_TEST_SKIP_NO_DEVICE\n"; return 0; }
    FastllmCudaSetDevice(0);
    const int rows = 8, vocab = argc>1?std::stoi(argv[1]):4096;
    if (vocab > 4096) {
        SoftmaxWorkspaceChecks();
        MixedSoftmaxBuckets();
    }
    Data logits(FLOAT32, {rows,vocab}, std::vector<float>(rows*vocab,0));
    logits.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
    if(vocab>4096) ConcurrentSamplers(vocab);
    InPlaceSamplingChecks();
    const int constraintVocab = vocab > 4096 ? 65536 : 513;
    ConstraintSamplingChecks(constraintVocab, false);
    ConstraintSamplingChecks(constraintVocab, true);
    SequentialSamplingChecks(vocab);
    ShardedSamplingChecks();
    ShardedSamplingChecks(true);
    CudaDecodeSampling a,b,invalid;
    Check(!invalid.Prepare(logits,{0,8}) && !invalid.Prepare(logits,{4,2}) &&
          !invalid.Prepare(logits,{1,1}) && !invalid.Prepare(logits,{1,16}), "invalid startup buckets");
    Check(a.Prepare(logits,{1,2,4,8},12345) && b.Prepare(logits,{1,2,4,8},12345),"startup sampler capture");
    Check(!a.Prepare(logits,{1,2,4,8},12345),"sampler recaptured after sealing");
    Check(!a.Supports(0) && !a.Supports(9),"unsupported batch admission");
#ifdef FASTLLM_TEST_DECODE_SAMPLING_HOOKS
    for (auto failure : {CaptureFailure::Begin, CaptureFailure::SecondSlot}) {
        captureFailure = failure; instantiateCalls = 0;
        CudaDecodeSampling failed;
        Check(!failed.Prepare(logits,{1},12345),"incomplete sampler slots admitted");
        Check(cudaGetLastError()==cudaSuccess,"startup capture left a CUDA error");
        const int captures = captureCalls;
        Check(!failed.Supports(1) && !failed.Submit({{1,0,1,1,vocab}},0),"failed sampler did not fall back");
        Check(!failed.Prepare(logits,{1},12345) && captureCalls==captures,"failed sampler recaptured at runtime");
    }
    captureFailure = CaptureFailure::None;
    const int runtimeCaptures = captureCalls;
#endif
    std::vector<CudaDecodeSampling::Request> requests;
    for(int row=0;row<3;++row) requests.push_back({uint64_t(100+row),0,1,1,vocab});
    std::set<int> values;
    for(int step=0;step<64;++step) {
        auto shuffled=requests;
        std::reverse(shuffled.begin(),shuffled.end());
        Check(a.Submit(requests,step%2) && b.Submit(shuffled,step%2),"sampling submit");
        auto x=a.Collect(step%2),y=b.Collect(step%2);
        std::reverse(y.begin(),y.end());
        Check(x==y,"request RNG changed after batch reorder");
        for(int token:x) Check(token>=0 && token<vocab,"invalid token");
        values.insert(x[0]);
        for(auto &r:requests) ++r.step;
    }
    Check(values.size()>48,"graph replay repeats RNG");
    // A row removed from a padded batch keeps the same sequence when compacted.
    auto one=std::vector<CudaDecodeSampling::Request>{requests[2]};
    Check(a.Submit(requests,0) && b.Submit(one,0),"compaction submit");
    auto x=a.Collect(0),y=b.Collect(0);
    Check(x[2]==y[0],"RNG depends on physical batch row");
    one[0].identity=10000;one[0].step=0;one[0].topK=1;
    Check(b.Submit(one,1),"parameter and identity rebind");b.Collect(1);
    ++one[0].step;
    Check(b.Submit(one,0),"queued token before discard");
    b.Discard();
    ++one[0].step;
    Check(b.Submit(one,0),"reusing slot after discard");b.Collect(0);
    // The input update uses zero live length as its padding contract.
    Data state(FLOAT32,{3,8},std::vector<float>(24,0));
    auto *p=(float*)state.cpuData;auto *n=(int*)(p+16);
    p[8]=10;p[9]=100;n[0]=11;n[1]=101;
    state.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
    Data next(FLOAT32,{4},std::vector<float>{17,18,19,20});
    next.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
    Check(FastllmCudaAdvanceDecodeInput(state.cudaData,8,4,(const float *)next.cudaData),"input advance");
    state.ToDevice(DataDevice::CPU);
    p=(float*)state.cpuData;n=(int*)(p+16);
    Check(p[0]==17 && p[1]==18 && p[2]==0 && p[8]==11 && p[9]==101 && n[0]==12 && n[1]==102 && n[2]==0 && p[10]==0,"input advance/padding semantics");
    state.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
    const float *mapped[4]{(float *)next.cudaData + 3, (float *)next.cudaData, nullptr, nullptr};
    Data mapping(INT32, {int(sizeof(mapped) / sizeof(int))});
    mapping.ToDevice(DataDevice::CUDA, std::vector<int>{0}, false); mapping.Allocate();
    FastllmCudaCopyFromHostToDevice(mapping.cudaData, mapped, sizeof(mapped));
    Data localPos(FLOAT32, {2}, std::vector<float>{11, -1});
    Data localLens(INT32, {2}); localLens.Allocate();
    ((int *)localLens.cpuData)[0] = 12; ((int *)localLens.cpuData)[1] = 0;
    localPos.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
    localLens.ToDevice(DataDevice::CUDA, std::vector<int>{0}, true);
    Check(FastllmCudaAdvanceMappedDecodeInput(state.cudaData, 8, 4, (const float *const *)mapping.cudaData,
        (float *)localPos.cudaData, (int *)localLens.cudaData, 2), "mapped input advance");
    state.ToDevice(DataDevice::CPU); localPos.ToDevice(DataDevice::CPU); localLens.ToDevice(DataDevice::CPU);
    p=(float *)state.cpuData; n=(int *)(p+16);
    Check(p[0]==20 && p[1]==17 && p[2]==0 && p[8]==12 && n[0]==13 && n[2]==0 &&
        ((float *)localPos.cpuData)[0]==12 && ((float *)localPos.cpuData)[1]==-1 &&
        ((int *)localLens.cpuData)[0]==13 && ((int *)localLens.cpuData)[1]==0,
        "mapped token permutation/local positions/padding semantics");
#ifdef FASTLLM_TEST_DECODE_SAMPLING_HOOKS
    Check(captureCalls==runtimeCaptures,"sampling replay recaptured a graph");
#endif
    std::cout << "CUDA DECODE SAMPLING PASS\n";
}
