#include "json11.hpp"
#include "models/basellm.h"
#include "pd.h"
#include <chrono>
#include <iostream>
#include <thread>

using namespace fastllm;
static void Check(bool ok, const char *message) {
    if (!ok)
        throw std::runtime_error(message);
}

class StateModel final : public basellm {
  public:
    std::atomic<int> forwards{0};
    StateModel() {
        block_cnt = 1;
        max_positions = 4096;
        use_new_engine = true;
        maxBatch = 2;
    }
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
        std::cout << "PASS PD tensor packing, chunk positions, handoff, rejection "
                     "and cancellation\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
