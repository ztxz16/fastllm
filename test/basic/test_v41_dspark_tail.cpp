#include "models/deepseekv41.h"
#include <algorithm>
#include <climits>
#include <iostream>
#include <stdexcept>

using namespace fastllm;
struct Access : DeepSeekV41Model {
    using DeepSeekV41Model::DsparkAcceptedDraftCount;
};
static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
int main() try {
    Access model;
    model.eos_token_id = 900;
    model.eos_token_ids = {901};
    size_t checks = 0;
    // Compare retained full-window verification against the old shortened
    // prefix for every rejection and output boundary, with sampled and greedy
    // acceptance. The suffix may contain stop tokens but cannot affect commit.
    for (int drafts : {0, 1, 2, 5, 7, 16, 33, 65}) {
        for (int accepted = 0; accepted <= drafts; ++accepted) {
            for (int room = 0; room <= drafts + 1; ++room) {
                for (bool sampled : {false, true}) {
                    GenerationConfig config;
                    config.input_token_length = 4096;
                    config.output_token_limit = 20 + room + 1;
                    config.stop_token_ids = {902};
                    DeepSeekV41SpecScratch full, truncated;
                    for (int i = 0; i < drafts; ++i) full.draftTokens.push_back(10 + i);
                    full.tokens = full.draftTokens; full.tokens.push_back(500);
                    if (accepted < drafts) full.tokens[accepted] = 500;
                    full.acceptedDrafts = sampled ? accepted : -1;
                    truncated.draftTokens.assign(full.draftTokens.begin(), full.draftTokens.begin() + std::min(room, drafts));
                    truncated.acceptedDrafts = sampled ? std::min(accepted, room) : -1;
                    for (int stop : {-1, 0, std::min(drafts, room), drafts}) {
                        auto tokens = full.tokens;
                        if (stop >= 0) full.tokens[stop] = 902;
                        truncated.tokens.assign(full.tokens.begin(), full.tokens.begin() + truncated.draftTokens.size() + 1);
                        const int got = model.DsparkAcceptedDraftCount(full, config, 4115);
                        const int expected = model.DsparkAcceptedDraftCount(truncated, config);
                        Check(got == expected, "tail changed accepted output prefix");
                        Check(got <= room && got <= accepted, "tail advanced KV beyond publishable output");
                        full.tokens = std::move(tokens);
                        ++checks;
                    }
                }
            }
        }
    }
    DeepSeekV41SpecScratch scratch;
    scratch.draftTokens = {10, 11, 12}; scratch.tokens = {10, 11, 12, 500}; scratch.acceptedDrafts = 3;
    GenerationConfig config;
    config.input_token_length = INT_MAX - 16; config.output_token_limit = 16;
    Check(model.DsparkAcceptedDraftCount(scratch, config, INT_MAX - 2) == 0, "large position overflow");
    config.output_token_limit = 0;
    Check(model.DsparkAcceptedDraftCount(scratch, config, INT_MAX - 2) == 3, "unlimited output was capped");
    config.output_token_limit = 1; config.input_token_length = 0;
    Check(model.DsparkAcceptedDraftCount(scratch, config, 20) == 3, "unknown input length was capped");
    config.input_token_length = 4096;
    config.output_token_limit = 3;
    config.stop_token_ids = {902};
    // Both model EOS forms and caller stop tokens must stop a publishable
    // prefix, while a stop beyond the output budget cannot extend it.
    for (int stop : {900, 901, 902}) {
        scratch.tokens = {stop, 11, 12, 500};
        Check(model.DsparkAcceptedDraftCount(scratch, config, 4096) == 0, "EOS inside output budget was ignored");
        scratch.tokens = {10, 11, stop, 500};
        Check(model.DsparkAcceptedDraftCount(scratch, config, 4096) == 1, "EOS beyond output budget extended commit");
        checks += 2;
    }
    std::cout << "PASS: " << checks << " DSpark tail prefixes\n";
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n'; return 1;
}
