#include "json11.hpp"
#include "models/basellm.h"
#include "pd.h"
#ifdef USE_CUDA
#include "devices/cuda/fastllm-cuda.cuh"
#endif
#include <chrono>
#include <algorithm>
#include <future>
#include <iostream>
#include <thread>

using namespace fastllm;
static void Check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}

class StateModel : public basellm {
  public:
    std::atomic<int> forwards{0};
    int historyRecords = 0;
    std::string observedSession;
    void OnResponseContextCreated(ResponseContext *context) override { observedSession = context->requestMetadata.sessionKey; }
    bool residentHistory = false;
    bool HasPDResidentPrefixCache() const override { return residentHistory; }
    void TryRecordResponseContext(ResponseContext *) override { ++historyRecords; }
    StateModel() {
        block_cnt = 1;
        max_positions = 4096;
        use_new_engine = true;
        maxBatch = 2;
    }
    ~StateModel() override { ShutdownRuntime(); }
    std::string MakeInput(const std::string &, int, const std::string &s) override { return s; }
    std::string MakeHistory(const std::string &, int, const std::string &, const std::string &s) override {
        return s;
    }
    std::string PDLayout() const override { return "test-recurrent-state-v1"; }
    int Forward(const Data &ids, const Data &, const Data &positions,
                std::vector<std::pair<Data, Data>> &kv, const GenerationConfig &, const LastTokensManager &,
                std::vector<float> *) override {
        ++forwards;
        float state = kv[0].first.dims.empty() ? 0 : ((float *)kv[0].first.cpuData)[0];
        for (size_t i = 0; i < ids.Count(0); ++i)
            state += ((float *)ids.cpuData)[i] * (1 + ((float *)positions.cpuData)[i]);
        kv[0].first.CopyFrom(Data(FLOAT32, {1}, std::vector<float>{state}));
        return 0;
    }
    fastkv::Message ExportPDState(ResponseContext &context) override {
        PDStateWriter writer;
        writer.metadata.U32(context.preTokens);
        writer.Tensor(context.pastKeyValues[0].first);
        return writer.Finish();
    }
    void ImportPDState(ResponseContext &context, const fastkv::Message &message) override {
        PDStateReader reader(message);
        Check(reader.metadata.U32() == context.allTokens.size() - 1, "prefix length mismatch");
        reader.Tensor(context.pastKeyValues[0].first);
        reader.End();
    }
};

// Exercise the actual chunked serving loop after transfer, rather than only
// calling ImportPDState and supplying correct positions directly to Forward.
static void TestChunkedDecodeHandoff() {
    class ScheduledModel final : public StateModel {
      public:
        std::atomic<int> positionErrors{0};
        ScheduledModel() {
            use_new_engine = false; canDoBatchForward = true; eos_token_id = -100;
            dataType = kvCacheDataType = FLOAT32;
        }
        ~ScheduledModel() override { ShutdownRuntime(); }
        void Stop() { ShutdownRuntime(); }
        bool SupportsPrefillBatching() const override { return true; }
        bool SupportsChunkedPrefillScheduling() const override { return true; }
        bool NeedAttentionMask(int, int) override { return false; }
        bool UseGenericHistoryCache() const override { return false; }
        bool RetainCudaWorkspace() const override { return true; }
        void ImportPDState(ResponseContext &context, const fastkv::Message &state) override {
            StateModel::ImportPDState(context, state);
            context.pastKeyValues[0].second.CopyFrom(context.pastKeyValues[0].first);
            for (Data *cache : {&context.pastKeyValues[0].first, &context.pastKeyValues[0].second})
                cache->expansionDims = cache->dims;
        }
        int Forward(const Data &ids, const Data &, const Data &positions,
                    std::vector<std::pair<Data,Data>> &kv, const GenerationConfig &,
                    const LastTokensManager &, std::vector<float> *) override {
            const int past = kv[0].first.dims.empty() ? 0 : kv[0].first.dims[1];
            for (size_t i = 0; i < ids.Count(0); ++i)
                if (((float *)positions.cpuData)[i] != past + i) ++positionErrors;
            const int length = past + ids.Count(0);
            for (Data *cache : {&kv[0].first,&kv[0].second}) {
                cache->CopyFrom(Data(FLOAT32,{1,length,1},std::vector<float>(length,1)));
                cache->expansionDims = cache->dims;
            }
            return 42;
        }
    } prefill, decode;
    fastkv::Service port; port.Start("127.0.0.1:0", [](const fastkv::Message &m) { return m; });
    auto endpoint = port.Address(); port.Stop();
    SetMaxTokens(4096);
    Check(decode.SetPrefillBatching(16,0) && decode.SetPrefillScheduling(false,0,true,0,0),
          "chunked decode policy rejected");
    prefill.ConfigurePD(json11::Json::object{{"role","prefill"},{"model_id","positions"},
        {"transport","tcp"},{"listen",endpoint},{"chunk_tokens",2}});
    decode.ConfigurePD(json11::Json::object{{"role","decode"},{"model_id","positions"},
        {"transport","tcp"},{"prefill",json11::Json::array{endpoint}}});
    GenerationConfig config; config.output_token_limit = 3; config.top_k = 1; config.add_special_tokens = false;
    int handle = decode.LaunchResponseTokens({3,7,11,19,23,29},config);
    for (int step = 0; step < 3; ++step) {
        for (int i = 0; i < 5000 && !decode.CanFetchResponse(handle); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        Check(decode.CanFetchResponse(handle), "chunked PD scheduler stalled");
        int token;
        Check(decode.FetchResponseTokensBatch(handle,&token,1) == 1 && token == 42,
              "chunked PD continuation failed");
    }
    Check(prefill.positionErrors == 0 && decode.positionErrors == 0,
          "PD handoff reset prompt/decode positions");
    decode.Stop(); prefill.StopPD();
    std::cout << "PASS PD chunked serving positions after handoff\n";
}

static void TestGroupAdmission() {
    class GroupModel : public StateModel {
        int next = 0;
      public:
        int PDGroupCount() const override { return 2; }
        int PDRequestGroup(const ResponseContext &c) const override {
            auto it = c.intParams.find("test_pd_group"); return it == c.intParams.end() ? 0 : it->second;
        }
        void OnResponseContextCreated(ResponseContext *c) override {
            StateModel::OnResponseContextCreated(c); c->intParams["test_pd_group"] = next++ % 2;
        }
    } prefill, decode;
    fastkv::Service port;
    port.Start("127.0.0.1:0", [](const fastkv::Message &m) { return m; });
    const auto endpoint = port.Address(); port.Stop();
    SetPageLen(128); SetMaxTokens(128);
    prefill.maxBatch = decode.maxBatch = 4;
    prefill.ConfigurePD(json11::Json::object{{"role", "prefill"}, {"model_id", "groups"},
        {"transport", "tcp"}, {"listen", endpoint}, {"chunk_tokens", 2}});
    decode.ConfigurePD(json11::Json::object{{"role", "decode"}, {"model_id", "groups"},
        {"transport", "tcp"}, {"prefill", json11::Json::array{endpoint}}});
    auto begin = [&](int group) {
        auto c = std::make_unique<ResponseContext>(); c->Init(1, FLOAT32, FLOAT32);
        c->allTokens = c->currentTokens = {3, 7, 11}; c->inputTokens = 3;
        c->generationConfig.output_token_limit = 1;
        decode.OnResponseContextCreated(c.get()); c->intParams["test_pd_group"] = group;
        decode.BeginPD(c.get()); return c;
    };
    auto a = begin(0), b = begin(0), c = begin(1);
    for (int i = 0; i < 3000 && (!a->preTokens || !c->preTokens); ++i) {
        decode.PollPD(a.get()); decode.PollPD(b.get()); decode.PollPD(c.get());
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    Check(a->preTokens == 2 && c->preTokens == 2 && !a->isEnding && !c->isEnding,
          "independent PD groups did not progress");
    Check(!b->preTokens && !b->isEnding, "PD overcommitted one group's pool");
    a.reset();
    for (int i = 0; i < 3000 && !b->preTokens; ++i) {
        decode.PollPD(b.get()); std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    Check(b->preTokens == 2 && !b->isEnding, "PD group credit was not returned");
    decode.StopPD(); prefill.StopPD(); SetMaxTokens(65536);
}

static void TestAdmissionLifetime() {
    class DecodeModel final : public StateModel {
      public:
        bool failImport = false;
        bool stateReleasedBeforeAdmission = true;
        std::weak_ptr<void> stateOwner;
        void ImportPDState(ResponseContext &context, const fastkv::Message &message) override {
            StateModel::ImportPDState(context, message);
            std::weak_ptr<void> admission = context.pdAdmission;
            auto owner = std::shared_ptr<int>(new int(1), [this, admission](int *value) {
                if (admission.expired()) stateReleasedBeforeAdmission = false;
                delete value;
            });
            stateOwner = owner;
            context.pastKeyValues[0].first.kvStorageOwner = std::move(owner);
            if (failImport) throw std::runtime_error("injected partial import failure");
        }
    } decode;
    StateModel prefill;
    fastkv::Service port;
    port.Start("127.0.0.1:0", [](const fastkv::Message &m) { return m; });
    const auto endpoint = port.Address(); port.Stop();
    SetPageLen(128); SetMaxTokens(65536);
    prefill.ConfigurePD(json11::Json::object{{"role", "prefill"}, {"model_id", "lifetime"},
        {"transport", "tcp"}, {"listen", endpoint}, {"chunk_tokens", 2}});
    decode.maxBatch = 1;
    decode.ConfigurePD(json11::Json::object{{"role", "decode"}, {"model_id", "lifetime"},
        {"transport", "tcp"}, {"prefill", json11::Json::array{endpoint}}});
    auto begin = [&](ResponseContext &context) {
        context.Init(1, FLOAT32, FLOAT32);
        context.currentTokens = context.allTokens = {3, 7, 11};
        context.inputTokens = 3;
        context.generationConfig.output_token_limit = 1;
        decode.BeginPD(&context);
    };
    auto imported = [&](ResponseContext &context) {
        for (int i = 0; i < 3000 && !context.isEnding && !decode.PollPD(&context); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    };
    // The scheduler retires PD transport state before a slow consumer drains
    // output. Admission must still cover retained KV, including failed imports.
    for (int mode = 0; mode < 3; ++mode) {
        decode.failImport = mode == 1;
        const int handle = decode.responseContextDict.CreateHandle();
        auto *context = decode.responseContextDict.GetHandle(handle);
        begin(*context); imported(*context);
        const auto owner = decode.stateOwner;
        Check(!owner.expired(), "lifetime fixture did not import state");
        Check(context->error == (mode == 1 ? ResponseContextErrorRuntime : ResponseContextErrorNone),
              "unexpected admission fixture import failure");
        context->isEnding = true;
        context->resultTokenQueue.push(42);
        context->pdPending.reset();
        decode.OnResponseContextFinished(context);
        ResponseContext queued; begin(queued);
        const int forwards = prefill.forwards;
        for (int i = 0; i < 20; ++i) {
            Check(!decode.PollPD(&queued) && queued.preTokens == 0 && !queued.isEnding,
                  "PD admission released while completed request still owns KV");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        Check(prefill.forwards == forwards && !owner.expired(),
              "queued request started P work before old KV was released");
        if (mode == 2) {
            decode.AbortResponse(handle);
            decode.RemoveResponseContext(handle);
        } else {
            int token = -1;
            Check(decode.FetchResponseTokensBatch(handle, &token, 1) == 1 && token == 42,
                  "retired request lost buffered output");
            Check(!owner.expired(), "buffered output unexpectedly destroyed the context");
            Check(decode.FetchResponseTokensBatch(handle, &token, 1) == (mode == 1 ? -3 : -1),
                  "terminal fetch lost the completion/error status");
        }
        Check(owner.expired(), "completed/cancelled request leaked KV");
        Check(decode.stateReleasedBeforeAdmission, "admission was destroyed before retained KV");
        decode.failImport = false;
        imported(queued);
        Check(!queued.isEnding && queued.preTokens == 2 &&
                  ((float *)queued.pastKeyValues[0].first.cpuData)[0] == 17,
              "released admission did not resume queued request correctly");
    }
    decode.StopPD(); prefill.StopPD();
    std::cout << "PASS PD admission lifetime: buffered output, partial import failure and cancellation\n";
}

class BatchStateModel : public StateModel {
  public:
    std::atomic<int> batches{0}, largest{0}, live{0}, peakTokens{0}, liveTokens{0};
    std::atomic<bool> fail{false}, entered{false}, hold{false};
    bool fullChunks = false;
    BatchStateModel() { maxBatch = 4; use_new_engine = false; canDoBatchForward = true; }
    ~BatchStateModel() override { hold = false; ShutdownRuntime(); }
    bool SupportsPrefillBatching() const override { return true; }
    void OnResponseContextCreated(ResponseContext *context) override {
        if (!context->requestMetadata.sessionKey.empty())
            Check(context->requestMetadata.sessionKey == "queued-" + std::to_string(context->allTokens.front()),
                  "concurrent PD requests mixed session metadata");
        ++live;
        const int tokens = (liveTokens += context->inputTokens);
        peakTokens = std::max(peakTokens.load(), tokens);
    }
    void OnResponseContextRemoved(ResponseContext *context) override {
        --live; liveTokens -= context->inputTokens;
    }
    std::vector<int> ForwardBatch(int batch, const Data &ids, const std::vector<Data *> &,
            const std::vector<Data *> &positions, const std::vector<int> &lengths,
            std::vector<std::pair<Data *, Data *>> &kv, const std::vector<GenerationConfig> &,
            const LastTokensManager &, std::vector<std::vector<float> *> *) override {
        ++batches; largest = std::max(largest.load(), batch);
        Check(isIntermediateChunkedPrefill && ids.Count(0) <= 8, "PD batch token budget/sampling flag");
        entered = true;
        while (hold) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (fail) throw std::runtime_error("injected batch failure");
        size_t offset = 0;
        for (int i = 0; i < batch; ++i) {
            Check(lengths[i] > 0 && lengths[i] <= 2, "per-request chunk bound");
            if (fullChunks) Check(lengths[i] == 2, "token budget fragmented full chunks");
            auto &state = *kv[i].first;
            float value = state.dims.empty() ? 0 : ((float *)state.cpuData)[0];
            for (int j = 0; j < lengths[i]; ++j)
                value += ((float *)ids.cpuData)[offset++] * (1 + ((float *)positions[i]->cpuData)[j]);
            state.CopyFrom(Data(FLOAT32, {1}, std::vector<float>{value}));
        }
        Check(offset == ids.Count(0), "packed batch lost input tokens");
        return std::vector<int>(batch, 0);
    }
};

static void TestPreparedPrefill() {
    class PreparedModel final : public BatchStateModel {
      public:
        std::atomic<int> enteredSteps{0}, permits{0};
        std::mutex createdMutex;
        std::vector<int> created;
        void OnResponseContextCreated(ResponseContext *context) override {
            BatchStateModel::OnResponseContextCreated(context);
            std::lock_guard<std::mutex> lock(createdMutex);
            created.push_back(context->allTokens.front());
        }
        std::vector<int> ForwardBatch(int batch, const Data &ids, const std::vector<Data *> &masks,
                const std::vector<Data *> &positions, const std::vector<int> &lengths,
                std::vector<std::pair<Data *, Data *>> &kv, const std::vector<GenerationConfig> &configs,
                const LastTokensManager &last, std::vector<std::vector<float> *> *logits) override {
            const int step = ++enteredSteps;
            while (permits < step) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            return BatchStateModel::ForwardBatch(batch, ids, masks, positions, lengths, kv, configs, last, logits);
        }
    } model;
    // Four-token page credits, including partial final pages. Active: 2*8;
    // prepared: floor((48/2)/8)=3; max_batch allows more but cannot overbook.
    SetPageLen(4); SetMaxTokens(48);
    model.maxBatch = 8; model.residentHistory = true;
    Check(model.SetPrefillBatching(4, 100, 2, 2), "configure prepared queue");
    fastkv::Service port;
    port.Start("127.0.0.1:0", [](const fastkv::Message &m) { return m; });
    const auto endpoint = port.Address(); port.Stop();
    model.ConfigurePD(json11::Json::object{{"role","prefill"},{"model_id","prepared"},
        {"transport","tcp"},{"listen",endpoint},{"chunk_tokens",2}});
    fastkv::Options options; options.workers = 8;
    fastkv::Service client(options), control;
    std::vector<std::future<fastkv::Message>> results;
    std::vector<std::shared_ptr<fastkv::Cancellation>> cancellations;
    auto submit = [&](int i) {
        fastkv::Writer w; w.U32(1); w.String("prepared:gpu-state-v2:test-recurrent-state-v1");
        w.Ints(std::vector<int>(7, 100+i));
        cancellations.push_back(std::make_shared<fastkv::Cancellation>());
        std::future<fastkv::Message> result;
        fastkv::Message request{std::move(w.bytes),{}};
        Check(client.TrySubmit(endpoint, request, result, cancellations.back()),
              "prepared fixture exhausted client admission");
        results.push_back(std::move(result));
    };
    auto wait = [&](auto predicate, const char *message) {
        for (int i = 0; i < 3000 && !predicate(); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        Check(predicate(), message);
    };
    try {
        submit(0); submit(1);
        wait([&] { return model.enteredSteps == 1; }, "prepared fixture did not enter forward");
        for (int i = 2; i < 8; ++i) submit(i);
        fastkv::Writer query; query.U32(2); query.String("prepared:gpu-state-v2:test-recurrent-state-v1");
        query.Ints({1});
        wait([&] {
            auto reply = control.Call(endpoint, {query.bytes,{}});
            fastkv::Reader r(reply.metadata); r.U32(); r.U64();
            return r.U64() == 56;
        }, "unprotected queued work was undercounted");
        model.permits = 1;
        wait([&] { return model.enteredSteps == 2; }, "prepared queue stalled forward");
        Check(model.live == 5 && model.liveTokens == 35,
              "prepared requests did not obey half-pool/page-rounded reservation");
        int cancelled;
        { std::lock_guard<std::mutex> lock(model.createdMutex); cancelled = model.created.back() - 100; }
        cancellations[cancelled]->Cancel();
        model.permits = 10000;
        for (size_t i = 0; i < results.size(); ++i) {
            Check(results[i].wait_for(std::chrono::seconds(5)) == std::future_status::ready,
                  "prepared cancellation stranded a request");
            bool failed = false;
            try {
                auto reply = results[i].get();
                PDStateReader reader(reply); Check(reader.metadata.U32() == 7, "prepared state length");
                Data state; reader.Tensor(state); reader.End();
                Check(((float *)state.cpuData)[0] == (100+i)*28,
                      "prepared request changed recurrent state");
            } catch (const std::exception &) { failed = true; if (int(i) != cancelled) throw; }
            Check((int(i) == cancelled) == failed, "cancelled prepared request succeeded");
        }
        Check(model.live == 0 && model.liveTokens == 0 && model.peakTokens <= 35,
              "prepared cancellation leaked state or capacity");

        // Stop while active, prepared and raw work coexist. Completed forward
        // must retire all three states without leaving a future or lease alive.
        model.enteredSteps = 0; model.permits = 0;
        results.clear(); cancellations.clear();
        submit(0); submit(1);
        wait([&] { return model.enteredSteps == 1; }, "shutdown fixture did not enter forward");
        for (int i = 2; i < 8; ++i) submit(i);
        wait([&] {
            auto reply = control.Call(endpoint, {query.bytes,{}});
            fastkv::Reader r(reply.metadata); r.U32(); r.U64();
            return r.U64() == 56;
        }, "shutdown fixture lost queued work");
        model.permits = 1;
        wait([&] { return model.enteredSteps == 2; }, "shutdown fixture did not prepare requests");
        Check(model.live == 5, "shutdown fixture did not own prepared contexts");
        auto stopped = std::async(std::launch::async, [&] { model.StopPD(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        model.permits = 10000;
        Check(stopped.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
              "prepared shutdown deadlocked");
        stopped.get();
        for (auto &result : results) {
            Check(result.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
                  "prepared shutdown stranded a waiter");
            bool failed = false;
            try { result.get(); } catch (const std::exception &) { failed = true; }
            Check(failed, "shutdown request succeeded unexpectedly");
        }
        Check(model.live == 0 && model.liveTokens == 0, "prepared shutdown leaked state");
    } catch (...) { model.permits = 10000; model.StopPD(); throw; }
    SetPageLen(128);
    std::cout << "PASS PD bounded prepared queue, page credits, cancellation and exact state\n";
}

static void TestExportBackpressure() {
    class BusyModel : public StateModel {
      public:
        std::atomic<int> blocked{0}, live{0};
        std::atomic<bool> freeze{false};
        void OnResponseContextCreated(ResponseContext *) override { ++live; }
        void OnResponseContextRemoved(ResponseContext *) override { --live; }
        fastkv::Message ExportPDState(ResponseContext &context) override {
            if (freeze || forwards < 2) {
                ++blocked;
                throw fastkv::CapacityBusy("test export lease held by another transfer");
            }
            return StateModel::ExportPDState(context);
        }
    } model;
    SetMaxTokens(4096);
    fastkv::Service port;
    port.Start("127.0.0.1:0", [](const fastkv::Message &m) { return m; });
    const auto endpoint = port.Address(); port.Stop();
    model.ConfigurePD(json11::Json::object{{"role","prefill"},{"model_id","pressure"},
        {"transport","tcp"},{"listen",endpoint},{"chunk_tokens",64},{"transfer_workers",2}});
    fastkv::Options options; options.transport = fastkv::Transport::Tcp; options.workers = 2;
    fastkv::Service client(options);
    auto submit = [&](int token, std::shared_ptr<fastkv::Cancellation> cancellation = {}) {
        fastkv::Writer writer;
        writer.U32(1); writer.String("pressure:gpu-state-v2:test-recurrent-state-v1");
        writer.Ints({token}); fastkv::Message request{std::move(writer.bytes),{}};
        std::future<fastkv::Message> result;
        Check(client.TrySubmit(endpoint,request,result,std::move(cancellation)),"export fixture admission failed");
        return result;
    };
    auto first = submit(3);
    for (int i = 0; i < 3000 && !model.blocked; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    Check(model.blocked > 0, "export backpressure fixture did not block");
    auto second = submit(7);
    for (auto *result : {&first, &second})
        Check(result->wait_for(std::chrono::seconds(5)) == std::future_status::ready,
              "busy export blocked the next prefill");
    int token = 3;
    for (auto *result : {&first, &second}) {
        PDStateReader reader(result->get()); Check(reader.metadata.U32() == 1,"busy export length");
        Data state; reader.Tensor(state); reader.End();
        Check(((float *)state.cpuData)[0] == token,"busy export changed state"); token = 7;
    }
    Check(model.forwards == 2,"busy export recomputed a completed prefix");
    model.freeze = true;
    for (bool shutdown : {false,true}) {
        const int blocked = model.blocked;
        auto cancellation = std::make_shared<fastkv::Cancellation>();
        auto pending = submit(shutdown ? 19 : 11,cancellation);
        for (int i = 0; i < 3000 && model.blocked == blocked; ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        Check(model.blocked > blocked && model.live == 1,"missing queued export context");
        if (shutdown) model.StopPD(); else cancellation->Cancel();
        Check(pending.wait_for(std::chrono::seconds(5)) == std::future_status::ready,"queued export cancellation stalled");
        bool rejected = false; try { pending.get(); } catch (const std::exception &) { rejected = true; }
        Check(rejected,"cancelled export succeeded");
        for (int i = 0; i < 3000 && model.live; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        Check(model.live == 0,"queued export leaked its context");
    }
    std::cout << "PASS PD export pressure: forward progress, exact state, cancellation and shutdown\n";
}

static void TestPrefillBatch() {
    auto endpoint = [] {
        fastkv::Service port;
        port.Start("127.0.0.1:0", [](const fastkv::Message &m) { return m; });
        const auto address = port.Address(); port.Stop(); return address;
    }();
    SetMaxTokens(24);
    BatchStateModel model;
    Check(model.SetPrefillBatching(4, 50, 4), "configure PD batch policy");
    model.ConfigurePD(json11::Json::object{{"role", "prefill"}, {"model_id", "test"},
        {"transport", "tcp"}, {"listen", endpoint}, {"cache_bytes", 65536}, {"chunk_tokens", 2}, {"transfer_workers", 1}});
    fastkv::Options options; options.transport = fastkv::Transport::Tcp; options.workers = 8;
    fastkv::Service client(options);
    auto call = [&](const std::vector<int> &tokens) {
        fastkv::Writer writer;
        writer.U32(1); writer.String("test:gpu-state-v2:test-recurrent-state-v1"); writer.Ints(tokens);
        return client.Call(endpoint, {std::move(writer.bytes), {}});
    };
    auto verify = [&](const std::vector<int> &tokens, const fastkv::Message &reply) {
        PDStateReader reader(reply);
        Check(reader.metadata.U32() == tokens.size(), "batch handoff length");
        Data state; reader.Tensor(state); reader.End();
        float expected = 0;
        for (size_t i = 0; i < tokens.size(); ++i) expected += tokens[i] * (i + 1);
        Check(((float *)state.cpuData)[0] == expected, "cross-request KV/position contamination");
    };
    auto run = [&](const std::vector<std::vector<int>> &inputs) {
        std::promise<void> go;
        auto ready = go.get_future().share();
        std::vector<std::future<fastkv::Message>> calls;
        for (const auto &input : inputs) calls.push_back(std::async(std::launch::async,
            [&, input, ready] { ready.wait(); return call(input); }));
        go.set_value();
        for (size_t i = 0; i < inputs.size(); ++i) verify(inputs[i], calls[i].get());
    };
    const std::vector<std::vector<int>> inputs{{3,4,5}, {7,8,9,10}, {11,12,13,14,15},
        {17,18,19,20,21,22,23}};
    model.fullChunks = true;
    run({{31,32,33,34}, {41,42,43,44}, {51,52,53,54}, {61,62,63,64}});
    model.fullChunks = false;
    run(inputs);
    Check(model.largest >= 2 && model.batches > 0, "RPC concurrency did not become model batching");
    Check(model.live == 0 && model.liveTokens == 0 && model.peakTokens <= 24, "prefill state/credit leak");
    const int batches = model.batches, forwards = model.forwards;
    run(inputs);
    Check(model.batches == batches && model.forwards == forwards, "exact prefix hits recomputed");
    auto extended = inputs;
    for (auto &input : extended) { input.push_back(29); input.push_back(31); }
    run(extended);
    Check(model.peakTokens <= 24 && model.live == 0, "aggregate P token capacity was overbooked");

    // Stop with both active and transport-queued work. Every waiter must be
    // released and every request-owned state retired before the model dies.
    model.entered = false; model.hold = true;
    std::vector<std::future<fastkv::Message>> calls;
    for (int i = 0; i < 8; ++i) calls.push_back(std::async(std::launch::async,
        [&, i] { return call(std::vector<int>(7, 101 + i)); }));
    for (int i = 0; i < 2000 && !model.entered; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    Check(model.entered, "batch worker did not enter held forward");
    auto stopped = std::async(std::launch::async, [&] { model.StopPD(); });
    std::this_thread::sleep_for(std::chrono::milliseconds(5)); model.hold = false;
    Check(stopped.wait_for(std::chrono::seconds(5)) == std::future_status::ready, "shutdown deadlocked");
    stopped.get();
    for (auto &result : calls) {
        Check(result.wait_for(std::chrono::seconds(5)) == std::future_status::ready, "stranded prefill waiter");
        bool failed = false;
        try { result.get(); } catch (const std::exception &) { failed = true; }
        Check(failed, "cancelled prefill succeeded");
    }
    Check(model.live == 0 && model.liveTokens == 0, "shutdown leaked P contexts");

    BatchStateModel failing;
    Check(failing.SetPrefillBatching(8, 50, 4), "configure failing batch policy");
    failing.fail = true;
    failing.ConfigurePD(json11::Json::object{{"role", "prefill"}, {"model_id", "test"},
        {"transport", "tcp"}, {"listen", endpoint}, {"chunk_tokens", 2}});
    calls.clear();
    for (int i = 0; i < 4; ++i) calls.push_back(std::async(std::launch::async,
        [&, i] { return call(std::vector<int>(3, 211 + i)); }));
    for (auto &result : calls) {
        Check(result.wait_for(std::chrono::seconds(5)) == std::future_status::ready, "batch error stranded a waiter");
        bool failed = false;
        try { result.get(); } catch (const std::exception &) { failed = true; }
        Check(failed, "batch failure was hidden");
    }
    failing.StopPD();
    Check(failing.live == 0, "batch error leaked P contexts");
    std::cout << "PASS PD real prefill batching, ragged chunks, prefix reuse, token credits and shutdown\n";
}

static void TestPrefillDPBatch() {
    class GroupedModel final : public BatchStateModel {
        int created = 0;
        std::map<const Data *, int> owners;
      public:
        int balancedBatches = 0;
        int GetPrefillBatchGroups() const override { return 2; }
        int GetPrefillBatchGroup(const ResponseContext *context) const override {
            return context->intParams.at("owner");
        }
        void OnResponseContextCreated(ResponseContext *context) override {
            BatchStateModel::OnResponseContextCreated(context);
            // Deliberately skew arrival order: 0,0,1,1. Each request retains
            // that resident owner through every subsequent prefill chunk.
            const int owner = AllocatePrefillGroup((created++ / 2) % 2);
            context->intParams["owner"] = owner;
            owners[&context->pastKeyValues[0].first] = owner;
        }
        void OnResponseContextRemoved(ResponseContext *context) override {
            owners.erase(&context->pastKeyValues[0].first);
            BatchStateModel::OnResponseContextRemoved(context);
        }
        std::vector<int> ForwardBatch(int batch, const Data &ids, const std::vector<Data *> &masks,
                const std::vector<Data *> &positions, const std::vector<int> &lengths,
                std::vector<std::pair<Data *, Data *>> &kv, const std::vector<GenerationConfig> &configs,
                const LastTokensManager &last, std::vector<std::vector<float> *> *logits) override {
            int tokens[2] = {};
            for (int i = 0; i < batch; ++i) tokens[owners.at(kv[i].first)] += lengths[i];
            Check(tokens[0] <= 2 && tokens[1] <= 2, "PD ignored per-DP token budget");
            if (tokens[0] == 2 && tokens[1] == 2) ++balancedBatches;
            return BatchStateModel::ForwardBatch(batch, ids, masks, positions, lengths, kv, configs, last, logits);
        }
    } model;
    fastkv::Service port;
    port.Start("127.0.0.1:0", [](const fastkv::Message &m) { return m; });
    const auto endpoint = port.Address(); port.Stop();
    SetMaxTokens(64);
    model.SetChunkedPrefillSize(2);
    Check(model.SetPrefillBatching(4, 5000, 4, 4) && model.SetPrefillScheduling(true, 0, false, 0, 0),
          "configure DP prefill policy");
    model.ConfigurePD(json11::Json::object{{"role", "prefill"}, {"model_id", "dp-batch"},
        {"transport", "tcp"}, {"listen", endpoint}, {"chunk_tokens", 2}, {"device_cache_bytes", 0}});
    fastkv::Options options; options.transport = fastkv::Transport::Tcp; options.workers = 8;
    fastkv::Service client(options);
    for (bool fail : {false, true}) {
        model.fail = fail;
        std::vector<std::future<fastkv::Message>> calls;
        // Hold model execution until all four RPCs are queued. Otherwise a
        // fast CPU can finish both owner-0 requests before owner-1 arrives,
        // making this a network timing test instead of a DP selection test.
        std::unique_lock<std::mutex> forwardGuard(model.forwardLocker);
        for (int i = 0; i < 4; ++i) calls.push_back(std::async(std::launch::async, [&, i] {
            fastkv::Writer writer;
            writer.U32(1); writer.String("dp-batch:gpu-state-v2:test-recurrent-state-v1");
            writer.Ints(std::vector<int>(3 + i, 11 + i));
            return client.Call(endpoint, {std::move(writer.bytes), {}});
        }));
        fastkv::Writer query;
        query.U32(2); query.String("dp-batch:gpu-state-v2:test-recurrent-state-v1"); query.Ints({1});
        bool queued = false;
        for (int i = 0; i < 3000 && !queued; ++i) {
            const auto reply = client.Call(endpoint,{query.bytes,{}});
            fastkv::Reader reader(reply.metadata); reader.U32(); reader.U64(); queued = reader.U64() == 18;
            if (!queued) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        forwardGuard.unlock();
        Check(queued,"DP fixture did not queue all owners");
        for (int i = 0; i < 4; ++i) {
            Check(calls[i].wait_for(std::chrono::seconds(10)) == std::future_status::ready,
                  "DP selection stranded a request");
            fastkv::Message reply;
            try { reply = calls[i].get(); }
            catch (const std::exception &) { if (!fail) throw; continue; }
            Check(!fail, "DP forward error was hidden");
            PDStateReader reader(reply);
            const int length = 3 + i;
            Check(reader.metadata.U32() == length, "DP reordered handoff length");
            Data state; reader.Tensor(state); reader.End();
            Check(((float *)state.cpuData)[0] == (11 + i) * length * (length + 1) / 2,
                  "DP selection mixed request state or chunk positions");
        }
        Check(model.live == 0 && model.liveTokens == 0, "DP selection leaked request/credit");
    }
    Check(model.balancedBatches >= 2, "PD did not select across nonadjacent owners");
    model.StopPD();
    std::cout << "PASS PD balanced owner selection, ragged chunks, exact state and failure cleanup\n";
}

static void TestQueuedTransfers() {
    fastkv::Service port;
    port.Start("127.0.0.1:0", [](const fastkv::Message &m) { return m; });
    const auto endpoint = port.Address();
    port.Stop();
    SetMaxTokens(65536);
    BatchStateModel prefill;
    Check(prefill.SetPrefillBatching(8, 50, 4), "configure queued PD batch");
    prefill.hold = true;
    prefill.ConfigurePD(json11::Json::object{{"role", "prefill"}, {"model_id", "queued"},
        {"transport", "tcp"}, {"listen", endpoint}, {"chunk_tokens", 2}, {"queue_depth", 128}});
    StateModel decode;
    decode.maxBatch = 128;
    decode.ConfigurePD(json11::Json::object{{"role", "decode"}, {"model_id", "queued"},
        {"transport", "tcp"}, {"prefill", json11::Json::array{endpoint}}});
    std::vector<std::unique_ptr<ResponseContext>> requests;
    bool admitted = true;
    for (int i = 0; i < 128; ++i) {
        requests.emplace_back(new ResponseContext());
        auto &context = *requests.back();
        context.Init(1, FLOAT32, FLOAT32);
        context.generationConfig.output_token_limit = 1;
        context.currentTokens = context.allTokens = {i + 1, i + 2, i + 3};
        context.requestMetadata.sessionKey = "queued-" + std::to_string(context.allTokens.front());
        context.inputTokens = 3;
        decode.BeginPD(&context);
        decode.PollPD(&context);
        admitted &= !context.isEnding;
    }
    // A timeout wave must cancel both P work and the caller's transfer. New
    // model requests may wait for draining leases but must never become 500s.
    for (int i = 0; i < 2000 && !prefill.entered; ++i) {
        for (auto &request : requests) decode.PollPD(request.get());
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    Check(prefill.entered, "queued cancellation did not reach held P work");
    requests.clear();
    for (int i = 0; i < 128; ++i) {
        requests.emplace_back(new ResponseContext());
        auto &context = *requests.back();
        context.Init(1, FLOAT32, FLOAT32);
        context.generationConfig.output_token_limit = 1;
        context.currentTokens = context.allTokens = {i + 1001, i + 1002, i + 1003};
        context.requestMetadata.sessionKey = "queued-" + std::to_string(context.allTokens.front());
        context.inputTokens = 3;
        decode.BeginPD(&context);
        decode.PollPD(&context);
        admitted &= !context.isEnding;
    }
    prefill.hold = false;
    std::vector<bool> completed(requests.size(), false);
    for (int i = 0; i < 5000 && std::find(completed.begin(), completed.end(), false) != completed.end(); ++i) {
        for (size_t j = 0; j < requests.size(); ++j)
            if (!completed[j]) completed[j] = requests[j]->isEnding || decode.PollPD(requests[j].get());
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    for (auto &request : requests) {
        admitted &= !request->isEnding && request->preTokens == 2;
        if (!request->isEnding && request->preTokens == 2) {
            const float expected = request->allTokens[0] + 2 * request->allTokens[1];
            Check(((float *)request->pastKeyValues[0].first.cpuData)[0] == expected,
                  "queued transfer mixed request states");
        }
        request->pdPending.reset();
    }
    decode.StopPD();
    prefill.StopPD();
    Check(admitted, "admitted requests must fit the transfer queue");
    Check(prefill.live == 0 && prefill.liveTokens == 0, "cancel/replacement wave leaked P states");
    std::cout << "PASS PD 128-request cancel/replacement wave, admission and state correctness\n";
}

static void TestLoadAwareRouting() {
    auto endpoint = [] {
        fastkv::Service temporary;
        temporary.Start("127.0.0.1:0", [](const fastkv::Message &m) { return m; });
        return temporary.Address();
    };
    const auto a = endpoint(), b = endpoint();
    SetMaxTokens(65536);
    BatchStateModel p0, p1;
    p0.maxBatch = p1.maxBatch = 2;
    Check(p0.SetPrefillBatching(4, 50, 2) && p1.SetPrefillBatching(4, 50, 2), "routing P batching");
    for (auto pair : {std::make_pair(&p0, a), std::make_pair(&p1, b)})
        pair.first->ConfigurePD(json11::Json::object{{"role","prefill"},{"model_id","routing"},
            {"transport","tcp"},{"listen",pair.second},{"cache_bytes",65536},{"chunk_tokens",2}});
    fastkv::Service client;
    auto submit = [&](const std::string &peer, const std::vector<int> &tokens) {
        fastkv::Writer w; w.U32(1); w.String("routing:gpu-state-v2:test-recurrent-state-v1"); w.Ints(tokens);
        return client.Submit(peer, {std::move(w.bytes), {}});
    };
    (void)submit(a, {3,4}).get(); // P0 owns the only reusable prefix.
    StateModel decode;
    decode.ConfigurePD(json11::Json::object{{"role","decode"},{"model_id","routing"},
        {"transport","tcp"},{"prefill",json11::Json::array{a,b}}});
    auto run = [&](const std::string &session = std::string()) {
        ResponseContext context; context.Init(1,FLOAT32,FLOAT32);
        context.requestMetadata.sessionKey = session;
        context.currentTokens = context.allTokens = {3,4,5}; context.inputTokens = 3;
        context.generationConfig.output_token_limit = 1;
        decode.BeginPD(&context);
        for (int i = 0; i < 4000 && !context.isEnding && !decode.PollPD(&context); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        Check(!context.isEnding && context.preTokens == 2, "routing stranded a request");
        Check(((float *)context.pastKeyValues[0].first.cpuData)[0] == 11, "routing changed KV state");
    };
    run("queued-3");
    Check(p1.forwards == 0 && p1.batches == 0, "routing ignored P0 resident cache");
    p0.hold = true;
    auto first = submit(a, std::vector<int>(2048, 11)), second = submit(a, std::vector<int>(2048, 21));
    for (int i = 0; i < 3000 && !p0.entered; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    Check(p0.entered, "routing fixture did not saturate P0");
    try { run(); } catch (...) { p0.hold = false; throw; }
    Check(p1.forwards + p1.batches > 0, "warm affinity blocked behind saturated P0");
    p0.hold = false; (void)first.get(); (void)second.get();
    p0.StopPD();
    // The old owner is unreachable. The live peer may satisfy this entirely
    // from its prefix cache, so require correct completion, not extra forward.
    run("queued-3");
    decode.StopPD(); p0.StopPD(); p1.StopPD();
    Check(p0.live == 0 && p1.live == 0, "routing leaked producer contexts");
    std::cout << "PASS PD live-prefix routing, saturated/dead producer bypass and cross-peer state correctness\n";
}

static void TestGPUState() {
#ifdef USE_CUDA
    uint64_t freeBytes = 0, totalBytes = 0;
    if (!FastllmCudaGetMemoryInfo(0, freeBytes, totalBytes)) return;
    std::vector<int> devices{0};
    if (FastllmCudaGetMemoryInfo(1, freeBytes, totalBytes)) devices.push_back(1);
    fastkv::BufferPool pool(256 << 20);
    PDStateWriter writer(&pool);
    std::vector<std::vector<uint8_t>> reference;
    for (int device : devices) {
        FastllmCudaSetDevice(device);
        Data tensor(FP4_E2M1, {1, 17, 192});
        tensor.dataDevice = CUDA; tensor.dataDeviceIds = {device};
        writer.AllocateTensor(tensor);
        reference.emplace_back(tensor.GetBytes());
        for (size_t i = 0; i < tensor.GetBytes(); ++i) reference.back()[i] = (i * 17 + device * 31) & 255;
        FastllmCudaCopyFromHostToDevice(tensor.cudaData, reference.back().data(), tensor.GetBytes());
        FastllmCudaSyncCurrentThreadStream();
    }
    auto message = writer.Finish(devices);
    auto destination = devices;
    std::reverse(destination.begin(), destination.end());
    fastkv::Options options;
    options.receivePlacement = [destination](const auto &metadata, const auto &sizes) {
        return PDReceivePlacement(metadata, sizes.size(), destination);
    };
    fastkv::Service server, client(options);
    server.Start("127.0.0.1:0", [&](const fastkv::Message &) { return message; });
    auto received = client.Call(server.Address(), {});
    PDStateReader reader(received);
    for (size_t i = 0; i < devices.size(); ++i) {
        Check(received.buffers[i].Kind() == fastkv::MemoryKind::Cuda &&
              received.buffers[i].Device() == destination[i], "receiver rank placement");
        Data tensor;
        reader.Tensor(tensor, true);
        Check(tensor.cudaData == received.buffers[i].Data() && tensor.cudaDataBorrowed,
              "GPU reader copied a borrowed state");
        std::vector<uint8_t> actual(tensor.GetBytes());
        FastllmCudaSetDevice(destination[i]);
        FastllmCudaCopyFromDeviceToHost(actual.data(), tensor.cudaData, actual.size());
        Check(actual == reference[i], "GPU FP4 payload or scales changed");
    }
    reader.End();
    PDStateReader ownedReader(received);
    for (size_t i = 0; i < devices.size(); ++i) {
        const int callerDevice = devices[i];
        FastllmCudaSetDevice(callerDevice);
        Data tensor;
        ownedReader.Tensor(tensor);
        Check(FastllmCudaGetDevice() == callerDevice, "GPU reader changed caller device");
        Check(!tensor.cudaDataBorrowed && tensor.cudaData != received.buffers[i].Data() &&
              FastllmCudaValidatePointerRange(tensor.cudaData, tensor.GetBytes(), destination[i]),
              "owned GPU reader allocated on wrong device");
        std::vector<uint8_t> actual(tensor.GetBytes());
        FastllmCudaSetDevice(destination[i]);
        FastllmCudaCopyFromDeviceToHost(actual.data(), tensor.cudaData, actual.size());
        Check(actual == reference[i], "owned GPU reader payload changed");
    }
    ownedReader.End();
    {
        fastkv::BufferPool smallPool(64 << 20);
        PDStateWriter first(&smallPool), second = first.Fork();
        Data firstTensor(FP4_E2M1, {1, 17, 192}), secondTensor(FP4_E2M1, {1, 17, 192});
        for (Data *tensor : {&firstTensor, &secondTensor}) {
            tensor->dataDevice = CUDA; tensor->dataDeviceIds = {devices[0]};
        }
        first.AllocateTensor(firstTensor);
        bool rejected = false;
        try { second.AllocateTensor(secondTensor); }
        catch (const std::runtime_error &) { rejected = true; }
        Check(rejected, "rank writers must reject an oversized message instead of deadlocking");
    }
    {
        // A planned export must fit by actual tensor bytes, including across
        // forked writers. The legacy 64 MiB fallback above remains unchanged.
        fastkv::BufferPool compactPool(2 << 20);
        PDStateWriter first(&compactPool), second = first.Fork();
        Data a(FP4_E2M1, {1, 17, 192}), b(BFLOAT16, {1, 19, 256});
        for (Data *tensor : {&a, &b}) {
            tensor->dataDevice = CUDA; tensor->dataDeviceIds = {devices[0]};
        }
        first.PlanTensor(a); second.PlanTensor(b); first.Reserve();
        first.AllocateTensor(a); second.AllocateTensor(b);
        Check(first.buffers[0].Id() == second.buffers[0].Id() &&
              second.buffers[0].Offset() >= first.buffers[0].Size(), "planned rank slices overlap");
        std::vector<uint8_t> source(a.GetBytes(), 0x63), actual(source.size());
        first.buffers[0].Write(source.data(), 0, source.size());
        first.buffers[0].Read(actual.data(), 0, actual.size());
        Check(actual == source, "compact export changed FP4 bytes");
        source.assign(b.GetBytes(), 0x29); actual.resize(source.size());
        second.buffers[0].Write(source.data(), 0, source.size());
        second.buffers[0].Read(actual.data(), 0, actual.size());
        Check(actual == source, "compact export changed BF16 bytes");
    }
    fastkv::PrefixCache cache(256 << 20, 0, -2);
    cache.Publish("gpu", {1,2,3}, received);
    auto hit = cache.Find("gpu", {1,2,3,4}, 4);
    Check(hit.tokens == 3, "GPU checkpoint prefix match");
    for (size_t i = 0; i < devices.size(); ++i) {
        Check(hit.state->buffers[i].Device() == destination[i], "GPU cache lost owner device");
        std::vector<uint8_t> actual(reference[i].size());
        hit.state->buffers[i].Read(actual.data(), 0, actual.size());
        Check(actual == reference[i], "GPU cache payload changed");
    }
    server.Stop(); client.Stop();
    std::cout << "PASS GPU PD placement, pooled payload, borrowed import and device checkpoint\n";
#endif
}

static void CheckLegacySessionPeer(bool sessionProtocol = false) {
    fastkv::Service legacy;
    int executions = 0;
    legacy.Start("127.0.0.1:0", [&](const fastkv::Message &request) {
        fastkv::Reader reader(request.metadata);
        const auto kind = reader.U32();
        Check(reader.String() == "legacy:gpu-state-v2:test-recurrent-state-v1", "legacy identity");
        const auto tokens = reader.Ints();
        if (kind == 3 && sessionProtocol) Check(reader.String(512) == "legacy-session", "legacy session lost");
        reader.End();
        if (kind == 2 || (kind == 4 && sessionProtocol)) {
            fastkv::Writer writer; writer.U32(kind); writer.U64(0); writer.U64(0); writer.U64(10000);
            return fastkv::Message{std::move(writer.bytes), {}};
        }
        Check(kind == 1 || (kind == 3 && sessionProtocol), "legacy peer rejects session protocol");
        ++executions;
        float value = 0; for (size_t i = 0; i < tokens.size(); ++i) value += tokens[i] * (i + 1);
        Data state(FLOAT32, {1}, std::vector<float>{value});
        PDStateWriter writer; writer.metadata.U32(tokens.size()); writer.Tensor(state);
        return writer.Finish();
    });
    StateModel decode;
    decode.ConfigurePD(json11::Json::object{{"role", "decode"}, {"model_id", "legacy"},
        {"transport", "tcp"}, {"prefill", json11::Json::array{legacy.Address()}}});
    ResponseContext context; context.Init(1, FLOAT32, FLOAT32);
    context.currentTokens = context.allTokens = {3, 7, 11}; context.inputTokens = 3;
    context.generationConfig.output_token_limit = 1;
    context.requestMetadata.sessionKey = "legacy-session";
    decode.BeginPD(&context);
    for (int i = 0; i < 5000 && !decode.PollPD(&context); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    Check(!context.isEnding && context.preTokens == 2 && executions == 1,
          "session negotiation retried model work or broke a legacy peer");
    Check(((float *)context.pastKeyValues[0].first.cpuData)[0] == 17, "legacy fallback changed state");
    decode.StopPD(); legacy.Stop();
}

int main() {
    try {
        TestAdmissionLifetime();
        CheckLegacySessionPeer();
        CheckLegacySessionPeer(true);
        // Packed FP4 rows include E4M3 scales. Transfer all stored bytes
        // without expanding or re-quantizing their values.
        Data fp4(FP4_E2M1, {1, 3, 192});
        fp4.Allocate();
        for (size_t i = 0; i < fp4.GetBytes(); ++i) fp4.cpuData[i] = i * 17;
        PDStateWriter packedWriter;
        packedWriter.Tensor(fp4);
        auto packed = packedWriter.Finish();
        Check(packed.buffers.size() == 1 && packed.buffers[0].Size() == 3 * 12 * 9,
              "FP4 wire byte count");
        PDStateReader packedReader(packed);
        Data unpacked;
        packedReader.Tensor(unpacked); packedReader.End();
        Check(unpacked.dataType == FP4_E2M1 && unpacked.dims == fp4.dims &&
                  std::equal(fp4.cpuData, fp4.cpuData + fp4.GetBytes(), unpacked.cpuData),
              "FP4 payload/scales round trip");
        Data source(FLOAT32, {2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
        source.Expansion({2, 8});
        PDStateWriter writer;
        writer.Tensor(source);
        auto message = writer.Finish();
        Check(message.buffers.size() == 1 && message.buffers[0].Size() == 24,
              "padding leaked into payload");
        Data restored;
        PDStateReader reader(message);
        reader.Tensor(restored);
        reader.End();
        Check(restored.dims == std::vector<int>({2, 3}), "shape round trip");
        for (int i = 0; i < 6; ++i)
            Check(((float *)restored.cpuData)[i] == i + 1, "strided tensor round trip");
        bool rejected = false;
        message.buffers.push_back(fastkv::Buffer::Allocate(1));
        try {
            PDStateReader invalid(message);
            invalid.Tensor(restored);
            invalid.End();
        } catch (const std::exception &) {
            rejected = true;
        }
        Check(rejected, "trailing payload must be rejected");

        fastkv::Service port;
        port.Start("127.0.0.1:0", [](const fastkv::Message &m) { return m; });
        const auto endpoint = port.Address();
        port.Stop();
        StateModel prefill, decode;
        SetPageLen(128);
        SetMaxTokens(16);
        prefill.ConfigurePD(json11::Json::object{{"role", "prefill"},
                                                 {"model_id", "test"},
                                                 {"transport", "tcp"},
                                                 {"listen", endpoint},
                                                 {"cache_bytes", 16384},
                                                 {"device_cache_bytes", 0},
                                                 {"chunk_tokens", 2}});
        SetMaxTokens(4096);
        decode.tokensLimit = 130;
        decode.ConfigurePD(json11::Json::object{{"role", "decode"},
                                                {"model_id", "test"},
                                                {"transport", "tcp"},
                                                {"prefill", json11::Json::array{endpoint}}});
        auto contextOwner = std::make_unique<ResponseContext>();
        auto &context = *contextOwner;
        context.Init(1, FLOAT32, FLOAT32);
        context.generationConfig.output_token_limit = 1;
        context.currentTokens = context.allTokens = {3, 7, 11, 19, 23, 29};
        context.inputTokens = 6;
        context.requestMetadata.sessionKey = "session-transport-test";
        decode.BeginPD(&context);
        for (int i = 0; i < 3000 && !decode.PollPD(&context); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        Check(!context.isEnding && context.preTokens == 5, "PD completion");
        Check(context.currentTokens == std::vector<int>{29} && context.cacheLen == 0,
              "decode handoff boundary");
        Check(((float *)context.pastKeyValues[0].first.cpuData)[0] == 3 + 14 + 33 + 76 + 115,
              "chunk position/state correctness");
        Check(prefill.forwards == 3, "three prefill chunks");
        Check(prefill.observedSession == "session-transport-test", "session metadata did not reach P before restore");
        {
            fastkv::Service control;
            fastkv::Writer query; query.U32(6); query.String("test:gpu-state-v2:test-recurrent-state-v1");
            query.Ints({3,7,11,19,23}); query.String("session-transport-test");
            auto placement = [&] {
                auto reply = control.Submit(endpoint, {query.bytes,{}}).get();
                fastkv::Reader reader(reply.metadata);
                Check(reader.U32() == 6, "placement capability reply");
                reader.U64(); reader.U64(); reader.U64();
                fastkv::SessionDirectory::Record record{reader.U64(), reader.String()};
                reader.End();
                return record;
            };
            for (int i = 0; i < 3000 && !placement().version; ++i) {
                decode.PollPD(&context);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            Check(placement().version == 1 && placement().owner == endpoint,
                  "another decoder cannot discover committed session ownership");
            auto announce = [&](uint64_t version, bool owned) {
                fastkv::Writer writer; writer.U32(7); writer.String("test:gpu-state-v2:test-recurrent-state-v1");
                writer.String("session-transport-test"); writer.U64(version); writer.String(endpoint); writer.U32(owned);
                control.Submit(endpoint, {writer.bytes,{}}).get();
            };
            announce(2, false); announce(1, true);
            Check(placement().version == 2,
                  "delayed placement announcement resurrected a retired owner");
            announce(3, true);
            Check(placement().version == 3, "new placement was not committed");
        }
        context.TryRecord(&decode);
        Check(decode.historyRecords == 0, "PD decode retained an unused host snapshot");
        decode.residentHistory = true;
        context.TryRecord(&decode);
        Check(decode.historyRecords == 1, "PD decode skipped resident history");
        decode.residentHistory = false;
        context.TryRecord(&prefill);
        Check(prefill.historyRecords == 1, "non-decode history policy changed");
        auto queuedOwner = std::make_unique<ResponseContext>();
        auto &queued = *queuedOwner;
        queued.Init(1, FLOAT32, FLOAT32);
        queued.allTokens = {1, 2, 3};
        queued.generationConfig.output_token_limit = 1;
        decode.BeginPD(&queued);
        for (int i = 0; i < 20; ++i) {
            Check(!decode.PollPD(&queued) && queued.preTokens == 0,
                  "two partial pages must not overbook the 130-token pool");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        Check(prefill.forwards == 3, "blocked admission must not start P work");
        contextOwner.reset();
        for (int i = 0; i < 3000 && !decode.PollPD(&queued); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        Check(!queued.isEnding && queued.preTokens == 2, "released page credit admits queued request");
        queuedOwner.reset();
        const int afterAdmission = prefill.forwards;

        for (int extended = 0; extended < 2; ++extended) {
            ResponseContext reused;
            reused.Init(1, FLOAT32, FLOAT32);
            reused.generationConfig.output_token_limit = 1;
            reused.allTokens = {3, 7, 11, 19, 23, 29};
            if (extended)
                reused.allTokens.insert(reused.allTokens.end(), {31, 37});
            reused.currentTokens = reused.allTokens;
            reused.inputTokens = reused.allTokens.size();
            decode.BeginPD(&reused);
            for (int i = 0; i < 3000 && !decode.PollPD(&reused); ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            Check(!reused.isEnding && reused.preTokens == (extended ? 7 : 5), "cached handoff");
            Check(((float *)reused.pastKeyValues[0].first.cpuData)[0] == (extended ? 632 : 241),
                  "restored prefix continuation positions");
            Check(prefill.forwards == afterAdmission + extended, "cached prefix must not recompute");
        }
        StateModel mismatch;
        mismatch.ConfigurePD(json11::Json::object{{"role", "decode"},
                                                  {"model_id", "wrong-checkpoint"},
                                                  {"transport", "tcp"},
                                                  {"prefill", json11::Json::array{endpoint}}});
        ResponseContext wrong;
        wrong.Init(1, FLOAT32, FLOAT32);
        wrong.generationConfig.output_token_limit = 1;
        wrong.allTokens = {1, 2, 3};
        wrong.inputTokens = 3;
        mismatch.BeginPD(&wrong);
        for (int i = 0; i < 3000 && wrong.pdPending; ++i) {
            mismatch.PollPD(&wrong);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        Check(wrong.isEnding && wrong.error == ResponseContextErrorRuntime && wrong.preTokens == 0,
              "incompatible state must not enter decode");
        // Abort the consumer while the transfer is queued; the service owns
        // the independent staging lifetime until it has completed or drained.
        {
            ResponseContext cancelled;
            cancelled.Init(1, FLOAT32, FLOAT32);
            cancelled.generationConfig.output_token_limit = 1;
            cancelled.allTokens = {1, 2, 3};
            decode.BeginPD(&cancelled);
            decode.PollPD(&cancelled);
        }
        ResponseContext oversized;
        oversized.Init(1, FLOAT32, FLOAT32);
        oversized.allTokens = std::vector<int>(20, 1);
        oversized.generationConfig.output_token_limit = 1;
        decode.BeginPD(&oversized);
        for (int i = 0; i < 3000 && oversized.pdPending; ++i) {
            decode.PollPD(&oversized);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        Check(oversized.isEnding && oversized.error == ResponseContextErrorRuntime,
              "heterogeneous P capacity must reject before allocating model pages");
        decode.StopPD();
        mismatch.StopPD();
        prefill.StopPD();
        TestGPUState();
        TestExportBackpressure();
        TestPrefillBatch();
        TestPrefillDPBatch();
        TestPreparedPrefill();
        TestQueuedTransfers();
        TestLoadAwareRouting();
        TestGroupAdmission();
        TestChunkedDecodeHandoff();
        std::cout << "PASS PD tensor packing, chunk positions, handoff, rejection "
                     "and cancellation\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
