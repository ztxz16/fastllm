#include "json11.hpp"
#include "models/basellm.h"
#include "pd.h"
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

class BatchStateModel final : public StateModel {
  public:
    std::atomic<int> batches{0}, largest{0}, live{0}, peakTokens{0}, liveTokens{0};
    std::atomic<bool> fail{false}, entered{false}, hold{false};
    BatchStateModel() { maxBatch = 4; use_new_engine = false; canDoBatchForward = true; }
    ~BatchStateModel() override { hold = false; ShutdownRuntime(); }
    bool SupportsPrefillBatching() const override { return true; }
    void OnResponseContextCreated(ResponseContext *context) override {
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

static void TestPrefillBatch() {
    auto endpoint = [] {
        fastkv::Service port;
        port.Start("127.0.0.1:0", [](const fastkv::Message &m) { return m; });
        const auto address = port.Address(); port.Stop(); return address;
    }();
    SetMaxTokens(24);
    BatchStateModel model;
    Check(model.SetPrefillBatching(8, 50, 4), "configure PD batch policy");
    model.ConfigurePD(json11::Json::object{{"role", "prefill"}, {"model_id", "test"},
        {"transport", "tcp"}, {"listen", endpoint}, {"cache_bytes", 65536}, {"chunk_tokens", 2}});
    fastkv::Options options; options.transport = fastkv::Transport::Tcp; options.workers = 8;
    fastkv::Service client(options);
    auto call = [&](const std::vector<int> &tokens) {
        fastkv::Writer writer;
        writer.U32(1); writer.String("test:test-recurrent-state-v1"); writer.Ints(tokens);
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

int main() {
    try {
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
                                                 {"chunk_tokens", 2}});
        SetMaxTokens(130);
        decode.ConfigurePD(json11::Json::object{{"role", "decode"},
                                                {"model_id", "test"},
                                                {"transport", "tcp"},
                                                {"prefill", json11::Json::array{endpoint}}});
        ResponseContext context;
        context.Init(1, FLOAT32, FLOAT32);
        context.generationConfig.output_token_limit = 1;
        context.currentTokens = context.allTokens = {3, 7, 11, 19, 23, 29};
        context.inputTokens = 6;
        decode.BeginPD(&context);
        for (int i = 0; i < 3000 && !decode.PollPD(&context); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        Check(!context.isEnding && context.preTokens == 5, "PD completion");
        Check(context.currentTokens == std::vector<int>{29} && context.cacheLen == 0,
              "decode handoff boundary");
        Check(((float *)context.pastKeyValues[0].first.cpuData)[0] == 3 + 14 + 33 + 76 + 115,
              "chunk position/state correctness");
        Check(prefill.forwards == 3, "three prefill chunks");
        ResponseContext queued;
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
        context.pdPending.reset();
        for (int i = 0; i < 3000 && !decode.PollPD(&queued); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        Check(!queued.isEnding && queued.preTokens == 2, "released page credit admits queued request");
        queued.pdPending.reset();
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
        TestPrefillBatch();
        std::cout << "PASS PD tensor packing, chunk positions, handoff, rejection "
                     "and cancellation\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
