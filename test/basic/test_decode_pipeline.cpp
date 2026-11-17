#include "models/basellm.h"
#include <iostream>
#include <stdexcept>
using namespace fastllm;
static void Check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
struct Model : basellm {
    int submits = 0, collects = 0, outstanding = 0, failAt = -1, eosAt = -1, cancelAt = -1;
    int invalidResultAt = -1;
    bool supported = true, historySupported = false;
    long long generated = 0;
    std::vector<std::string> events;
    Model(int rows, int limit) {
        saveHistoryChat = false; max_positions = 100; eos_token_id = 99;
        for (int row = 0; row < rows; ++row) {
            auto *c = new ResponseContext();
            c->Init(0, FLOAT32, FLOAT32); c->curTokens = 1; c->preTokens = 4;
            c->intParams["index"] = 0; c->intParams["promptLen"] = 4;
            c->currentTokens = {7}; c->allTokens = {1,2,3,4,7};
            c->generationConfig.repeat_penalty = 1;
            c->generationConfig.output_token_limit = limit;
            responseContextDict.dicts[row] = c;
        }
    }
    std::string MakeInput(const std::string &, int, const std::string &) override { return {}; }
    std::string MakeHistory(const std::string &, int, const std::string &, const std::string &) override { return {}; }
    void TryRecordResponseContext(ResponseContext *) override {
        Check(outstanding == 0, "terminal request published before pipeline drained");
    }
    struct Pipe : DecodePipeline {
        Model &m; std::vector<ResponseContext *> contexts; int tickets[2]{-1,-1};
        Pipe(Model &m, const std::vector<ResponseContext *> &c) : m(m), contexts(c) {}
        bool Submit(int slot, int input = -1) override {
            if (m.submits == m.failAt) return false;
            Check(tickets[slot] < 0, "result slot overwritten");
            if (input >= 0) Check(tickets[input] >= 0, "GPU token released too early");
            tickets[slot] = ++m.submits; ++m.outstanding;
            m.events.push_back("submit"); return true;
        }
        std::vector<int> Collect(int slot) override {
            Check(tickets[slot] > 0, "invalid result slot");
            const int step = tickets[slot]; tickets[slot] = -1; --m.outstanding; ++m.collects;
            if (step == m.cancelAt) contexts[0]->isAbort = true;
            std::vector<int> result(contexts.size(), 10 + step);
            if (step == m.eosAt) result[0] = 99;
            if (step == m.invalidResultAt) result.clear();
            m.events.push_back("collect"); return result;
        }
        void Drain() override {
            for (int &ticket : tickets) if (ticket >= 0) { ticket = -1; --m.outstanding; }
        }
    };
    bool SupportsDecodePipeline() const override { return supported; }
    bool SupportsDecodePipelineHistory() const override { return historySupported; }
    std::unique_ptr<DecodePipeline> CreateDecodePipeline(const std::vector<ResponseContext *> &c) override {
        if (!supported) return nullptr;
        return std::unique_ptr<DecodePipeline>(new Pipe(*this,c));
    }
    bool Run() {
        std::unique_lock<std::mutex> lock(dictLocker);
        const bool ran = RunDecodePipeline(lock, 64, 100, &generated);
        size_t published = 0;
        for (auto &entry : responseContextDict.dicts) published += entry.second->resultTokenQueue.size();
        Check(generated == (long long)published, "decode speed counter includes missing/discarded tokens");
        return ran;
    }
    uint64_t OutputVersion() {
        uint64_t version = 0;
        Check(FetchResponses(nullptr, nullptr, 0, 1, nullptr, 0, nullptr, 0, 0, version) == 0,
              "output notification probe failed");
        return version;
    }
};
int main() {
    for (bool history : {false, true}) for (int batch : {1,2,3,5,8,17}) for (int limit : {2,3,7}) {
        Model m(batch,limit); m.saveHistoryChat = history; m.historySupported = true;
        Check(m.Run(), "pipeline not selected");
        Check(m.OutputVersion() > 0, "pipeline tokens did not wake batched response readers");
        Check(m.submits == limit-1 && m.collects == m.submits, "wrong forward count");
        for (auto &entry : m.responseContextDict.dicts) {
            auto *c=entry.second;
            Check(c->curTokens==limit && c->resultTokenQueue.size()==size_t(limit-1), "missing/extra output");
            Check(c->intParams["index"]==limit-1 && c->preTokens==limit+3, "wrong scheduler position");
        }
        if(limit>2) Check(m.events[0]=="submit" && m.events[1]=="submit" && m.events[2]=="collect", "host wait before next submit");
    }
    for (bool history : {false, true}) for (int stop : {1,2,3}) {
        Model m(3,8); m.saveHistoryChat = history; m.historySupported = true;
        m.eosAt=stop; Check(m.Run(), "EOS pipeline");
        Check(m.outstanding==0, "EOS did not drain");
        Check(m.responseContextDict.dicts[0]->isEnding, "EOS ignored");
        Check(m.responseContextDict.dicts[0]->resultTokenQueue.size()==size_t(stop-1), "post-EOS token exposed");
        Check(m.responseContextDict.dicts[1]->curTokens==stop+2, "survivor lookahead lost");
    }
    { Model m(3,8); m.cancelAt=1; Check(m.Run(),"cancel"); Check(m.responseContextDict.dicts[0]->resultTokenQueue.empty(),"post-cancel output"); }
    { Model m(2,8); m.failAt=0; Check(!m.Run() && m.submits==0,"initial admission mutated state"); }
    { Model m(2,8); m.failAt=1; Check(m.Run() && m.collects==1,"mid-chain admission lost current token"); }
    {
        Model m(2,8); m.invalidResultAt=1;
        Check(m.Run() && m.outstanding==0,"malformed result did not drain lookahead");
        Check(m.OutputVersion() > 0, "pipeline error did not wake batched response readers");
        for (auto &entry : m.responseContextDict.dicts)
            Check(entry.second->isEnding && entry.second->error==ResponseContextErrorRuntime &&
                  entry.second->resultTokenQueue.empty(),"malformed result published output or lost error");
    }
    { Model m(2,8); m.supported=false; Check(!m.Run(),"unsupported adapter fallback"); }
    { Model m(2,8); m.saveHistoryChat=true; Check(!m.Run(),"history fallback"); }
    { Model m(2,8); m.responseContextDict.dicts[0]->generationConfig.repeat_penalty=1.2f; Check(!m.Run(),"history-dependent sampling fallback"); }
    std::cout << "DECODE PIPELINE PASS\n";
}
