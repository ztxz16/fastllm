#include "models/basellm.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

using namespace fastllm;
namespace {
void Require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
struct Fixture : basellm {
    bool enabled;
    explicit Fixture(bool enabled) : enabled(enabled) {}
    bool UseGpuToolCallConstraints() const override { return enabled; }
    bool HasCache() const { return bool(toolCallConstraintCache); }
    std::string MakeInput(const std::string &, int, const std::string &) override { return {}; }
    std::string MakeHistory(const std::string &, int, const std::string &, const std::string &) override { return {}; }
};

void CheckCache() {
    Fixture cached(true), uncached(false);
    Require(cached.UseGpuToolCallConstraints() && !uncached.UseGpuToolCallConstraints(), "execution mode gate");
    const std::vector<std::string> pieces = {"read", "write", "path", "mode", "data", "re", "ad", "\"", "\">body", "INVALID"};
    for (auto *m : {&cached, &uncached}) for (int i = 0; i < int(pieces.size()); ++i) m->weight.tokenizer.Insert(pieces[i], i);
    GenerationConfig c;
    c.tool_call_name_constraint_enabled = c.tool_call_parameter_name_constraint_enabled = true;
    c.tool_call_invoke_name_prefixes = {"<invoke name=\""};
    c.tool_call_parameter_name_prefixes = {"<parameter name=\""};
    c.tool_call_allowed_names = {"read", "write"};
    c.tool_call_allowed_parameter_names = {{"read", {"path", "mode"}}, {"write", {"data"}}};
    for (int pass = 0; pass < 3; ++pass) {
        for (const std::string text : {"ordinary text", "<invoke name=\"", "<invoke name=\"re", "<invoke name=\"read",
                "<invoke name=\"read\"><parameter name=\"", "<invoke name=\"write\"><parameter name=\"",
                "<invoke name=\"read\"><parameter name=\"path\">value", "<invoke name=\"read\"></invoke>"}) {
            c.tool_call_generated_text = text;
            auto a = c, b = c;
            cached.PrepareToolCallConstraint(a); uncached.PrepareToolCallConstraint(b);
            Require(a.tool_call_allowed_token_ids == b.tool_call_allowed_token_ids, "cache changed matcher");
        }
        // Identical text with a different request schema must get a different mask.
        c.tool_call_generated_text = "<invoke name=\"";
        c.tool_call_allowed_names = {pass % 2 ? "read" : "write"};
        auto a = c, b = c;
        cached.PrepareToolCallConstraint(a); uncached.PrepareToolCallConstraint(b);
        Require(a.tool_call_allowed_token_ids == b.tool_call_allowed_token_ids, "schema leaked across requests");
    }
    Require(cached.HasCache() && !uncached.HasCache(), "uncached created GPU constraint cache");
    for (int i = 0; i < 520; ++i) {
        c.tool_call_allowed_names = {"read", "request_" + std::to_string(i)};
        auto a = c, b = c;
        cached.PrepareToolCallConstraint(a); uncached.PrepareToolCallConstraint(b);
        Require(a.tool_call_allowed_token_ids == b.tool_call_allowed_token_ids, "cache eviction changed constraints");
    }
    cached.weight.tokenizer.Insert("request_519", pieces.size());
    uncached.weight.tokenizer.Insert("request_519", pieces.size());
    auto a = c, b = c;
    cached.PrepareToolCallConstraint(a); uncached.PrepareToolCallConstraint(b);
    Require(a.tool_call_allowed_token_ids == b.tool_call_allowed_token_ids, "tokenizer growth kept a stale cache");

    auto compare = [&] {
        auto actual = c, reference = c;
        cached.PrepareToolCallConstraint(actual); uncached.PrepareToolCallConstraint(reference);
        Require(actual.tool_call_allowed_token_ids == reference.tool_call_allowed_token_ids,
                "tokenizer mutation kept a stale cache");
        return actual.tool_call_allowed_token_ids;
    };
    c.tool_call_allowed_names = {"read", "a", " read"};
    compare();
    // Replacing a word or special token can leave vocabulary size unchanged.
    for (auto *m : {&cached, &uncached}) m->weight.tokenizer.Insert("write", 0);
    auto replaced = compare();
    Require(std::find(replaced.begin(), replaced.end(), 0) == replaced.end(), "stale replaced-token permission");
    for (auto *m : {&cached, &uncached}) m->weight.tokenizer.SetSpecialTokens({{"read", 0}});
    replaced = compare();
    Require(std::find(replaced.begin(), replaced.end(), 0) != replaced.end(), "special-token update lost");
    for (auto *m : {&cached, &uncached}) {
        m->weight.tokenizer.Clear();
        for (int i = 0; i < int(pieces.size()); ++i) m->weight.tokenizer.Insert(pieces[i], i);
        m->weight.tokenizer.Insert("<0x61>", pieces.size());
    }
    compare();
    // Decoder options also change decoded pieces without changing the vocabulary.
    for (auto *m : {&cached, &uncached}) m->weight.tokenizer.preTokenizer = "joyai-llm";
    compare();
    for (auto *m : {&cached, &uncached}) {
        m->weight.tokenizer.preTokenizer.clear();
        m->weight.tokenizer.Insert("Ġread", 0);
    }
    compare();
    for (auto *m : {&cached, &uncached}) m->weight.tokenizer.byteAsChar = true;
    compare();
    // Model reinitialization invalidates even dictionaries filled directly by loaders.
    for (auto *m : {&cached, &uncached}) {
        m->weight.tokenizer.tokenToStringDict[0] = "write";
        m->InitParams();
    }
    compare();
}
}
int main() {
    try {
        CheckCache();
        std::cout << "tool constraint cache PASS\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
