#ifndef FASTLLM_MIMO_TOOL_GRAMMAR_H
#define FASTLLM_MIMO_TOOL_GRAMMAR_H

#include "token_constraint.h"
#include <algorithm>
#include <deque>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace fastllm {

// Shared by requests using the same tokenizer. Only tokens containing '<'
// can open a marker while reading unrestricted text or argument values.
class MimoToolVocabulary {
    friend class MimoToolGrammar;
    std::shared_ptr<const std::vector<std::string>> tokens;
    std::array<std::vector<int>, 256> plainByFirstByte;
    std::vector<int> withMarker;
    std::vector<uint32_t> plainMask;
public:
    explicit MimoToolVocabulary(std::shared_ptr<const std::vector<std::string>> tokens)
        : tokens(std::move(tokens)) {
        if (!this->tokens) throw std::invalid_argument("MiMo grammar needs a vocabulary.");
        plainMask.resize((this->tokens->size() + 31) / 32);
        for (int id = 0; id < (int)this->tokens->size(); ++id) {
            const auto &text = (*this->tokens)[id];
            if (text.empty()) continue;
            if (text.find('<') != std::string::npos) withMarker.push_back(id);
            else {
                plainByFirstByte[(unsigned char)text.front()].push_back(id);
                plainMask[id / 32] |= uint32_t(1) << (id % 32);
            }
        }
    }
};

// MiMo's closed XML tool dialect. Argument values remain raw text; JSON/schema
// validation belongs to the existing parser. State IDs are cheap branch copies.
class MimoToolGrammar {
    enum Phase { Text, Reasoning, Function, Name, Body, Value, Close };
    using StateKey = std::tuple<Phase, int, std::string, bool, bool>;
    struct State {
        Phase phase = Text;
        int tool = -1;
        std::string prefix;
        bool gap = false, called = false;
        StateKey Key() const { return std::make_tuple(phase, tool, prefix, gap, called); }
    };
    struct Node {
        State state;
        std::array<int, 256> next;
        std::shared_ptr<const std::vector<uint32_t>> mask;
        explicit Node(State s) : state(std::move(s)) { next.fill(-2); }
    };
    std::vector<std::string> names;
    std::vector<std::vector<std::string>> bodies;
    std::vector<Node> nodes;
    std::map<StateKey, int> states;
    std::shared_ptr<const MimoToolVocabulary> vocabulary;
    std::set<int> stops;
    bool required, single;
    std::mutex mutex;

    static bool White(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }
    static bool Starts(std::string_view s, std::string_view p) {
        return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
    }
    static void CheckName(const std::string &name) {
        if (name.empty() || White(name.front()) || White(name.back()) ||
            name.find_first_of("<>") != std::string::npos)
            throw std::invalid_argument("Unrepresentable MiMo tool/parameter name.");
    }
    int Intern(const State &s) {
        auto found = states.find(s.Key());
        if (found != states.end()) return found->second;
        int id = nodes.size();
        states.emplace(s.Key(), id);
        nodes.emplace_back(s);
        return id;
    }
    static void Suffix(State &s, std::initializer_list<std::string_view> markers) {
        while (!s.prefix.empty()) {
            for (const auto &m : markers) if (Starts(m, s.prefix)) return;
            s.prefix.erase(0, 1);
        }
    }
    int Step(int id, unsigned char c) {
        if (id < 0) return -1;
        int cached = nodes.at(id).next[c];
        if (cached != -2) return cached;
        State s = nodes[id].state;
        bool valid = true;
        if (s.phase == Text || s.phase == Reasoning || s.phase == Value) {
            s.prefix += c;
            if (s.phase == Reasoning) {
                if (s.prefix == "</think>") { s.phase = Text; s.prefix.clear(); }
                else Suffix(s, {"</think>"});
            } else if (s.phase == Text) {
                if (s.prefix == "<think>") { s.phase = Reasoning; s.prefix.clear(); }
                else if (s.prefix == "<tool_call>") {
                    if (single && s.called) valid = false;
                    else { s.phase = Function; s.prefix.clear(); s.gap = false; }
                } else Suffix(s, {"<tool_call>", "<think>"});
            } else {
                if (s.prefix == "</parameter>") {
                    s.phase = Body; s.prefix.clear(); s.gap = false;
                } else if (s.prefix == "</function>" || s.prefix == "<tool_call>" || s.prefix == "</tool_call>") {
                    valid = false;
                } else Suffix(s, {"</parameter>", "</function>", "<tool_call>", "</tool_call>"});
            }
        } else if (s.prefix.empty() && !s.gap && White(c)) {
            s.gap = true;
        } else {
            s.prefix += c;
            if (s.phase == Name) {
                valid = false;
                for (int i = 0; i < (int)names.size(); ++i) {
                    const auto &name = names[i];
                    if (Starts(name, s.prefix) || (s.prefix.size() == name.size() + 1 &&
                            Starts(s.prefix, name) && White(s.prefix.back()))) valid = true;
                    if (s.prefix == name + ">" || (s.prefix.size() == name.size() + 2 &&
                            Starts(s.prefix, name) && White(s.prefix[name.size()]) && c == '>')) {
                        s.phase = Body; s.tool = i; s.prefix.clear(); s.gap = false;
                        valid = true; break;
                    }
                }
            } else {
                static const std::vector<std::string> function = {"<function="}, close = {"</tool_call>"};
                const auto &choices = s.phase == Body ? bodies.at(s.tool) :
                    s.phase == Function ? function : close;
                valid = false;
                for (const auto &choice : choices) {
                    if (!Starts(choice, s.prefix)) continue;
                    valid = true;
                    if (choice == s.prefix) {
                        if (s.phase == Function) s.phase = Name;
                        else if (s.phase == Body) s.phase = choice == "</function>" ? Close : Value;
                        else { s.phase = Text; s.tool = -1; s.called = true; }
                        s.prefix.clear(); s.gap = false;
                        break;
                    }
                }
            }
        }
        int result = valid ? Intern(s) : -1;
        nodes[id].next[c] = result; // Intern may reallocate nodes.
        return result;
    }
    int AdvanceBytes(int state, const std::string &text) {
        for (unsigned char c : text) { state = Step(state, c); if (state < 0) break; }
        return state;
    }
    bool PlainText(int state) const {
        if (state < 0) return false;
        const auto &s = nodes.at(state).state;
        return s.prefix.empty() && (s.phase == Text || s.phase == Reasoning || s.phase == Value);
    }
    bool BoundTo(const std::shared_ptr<const std::vector<std::string>> &tokens,
                 const std::set<int> &endTokens) const {
        if (!vocabulary) return false;
        if (vocabulary->tokens != tokens || stops != endTokens)
            throw std::logic_error("MiMo grammar cannot change tokenizer or stop tokens.");
        return true;
    }
    bool EndAllowed(int state) const {
        return nodes.at(state).state.phase == Text && (!required || nodes[state].state.called);
    }
    struct Compiled {
        std::shared_ptr<const TokenConstraintProgram> program;
        std::map<StateKey, int> states;
        // Keep tokenizer identity alive while cached.
        std::shared_ptr<const MimoToolVocabulary> vocabulary;
    };
    std::shared_ptr<Compiled> compiledProgram;
public:
    // CPU request states remain untouched. Canonical compiled IDs are private
    // speculative cursors, so differently explored CPU branches can share tables.
    std::pair<std::shared_ptr<const TokenConstraintProgram>, int> Compile(int state) {
        std::lock_guard<std::mutex> guard(mutex);
        if (!vocabulary || state < 0) throw std::logic_error("Unbound MiMo grammar.");
        if (compiledProgram) return {compiledProgram->program,
            compiledProgram->states.at(nodes.at(state).state.Key())};
        using Key = std::tuple<const MimoToolVocabulary *, std::vector<std::string>,
            std::vector<std::vector<std::string>>, std::set<int>, bool, bool>;
        struct Entry { Key key; std::shared_ptr<Compiled> value; };
        static std::mutex cacheMutex;
        static std::deque<Entry> cache;
        std::lock_guard<std::mutex> cacheGuard(cacheMutex);
        Key key{vocabulary.get(), names, bodies, stops, required, single};
        std::shared_ptr<Compiled> compiled;
        for (auto it = cache.begin(); it != cache.end(); ++it) if (it->key == key) {
            compiled = it->value; cache.erase(it); break;
        }
        if (!compiled) {
            std::map<std::string, std::vector<std::string>> tools;
            for (size_t i = 0; i < names.size(); ++i) {
                auto &parameters = tools[names[i]];
                for (size_t j = 1; j < bodies[i].size(); ++j)
                    parameters.push_back(bodies[i][j].substr(11, bodies[i][j].size() - 12));
            }
            MimoToolGrammar canonical(tools, required, single);
            canonical.Bind(vocabulary, stops);
            auto program = std::make_shared<TokenConstraintProgram>();
            program->vocab = vocabulary->tokens->size();
            // Close the byte DFA before building masks: Mask must not change IDs.
            for (size_t i = 0; i < canonical.nodes.size(); ++i)
                for (int c = 0; c < 256; ++c) canonical.Step(i, (unsigned char)c);
            std::map<std::vector<uint32_t>, int> masks;
            for (size_t i = 0; i < canonical.nodes.size(); ++i) {
                const auto bits = canonical.Mask(i, false);
                auto found = masks.emplace(*bits, masks.size());
                if (found.second) program->masks.insert(program->masks.end(), bits->begin(), bits->end());
                program->maskIds.push_back(found.first->second);
                const auto &next = canonical.nodes[i].next;
                program->transitions.insert(program->transitions.end(), next.begin(), next.end());
            }
            program->tokenOffsets.push_back(0);
            program->stopTokens.resize(program->vocab, 0);
            for (int token = 0; token < program->vocab; ++token) {
                const auto &bytes = (*vocabulary->tokens)[token];
                program->tokenBytes.insert(program->tokenBytes.end(), bytes.begin(), bytes.end());
                program->tokenOffsets.push_back(program->tokenBytes.size());
                program->stopTokens[token] = stops.count(token) != 0;
            }
            compiled = std::make_shared<Compiled>();
            compiled->program = program; compiled->states = std::move(canonical.states);
            compiled->vocabulary = vocabulary;
        }
        cache.push_back({std::move(key), compiled});
        size_t bytes = 0;
        for (const auto &entry : cache) bytes += entry.value->program->Bytes();
        while (cache.size() > 1 && (bytes > 128 * 1024 * 1024 || cache.size() > 32)) {
            bytes -= cache.front().value->program->Bytes(); cache.pop_front();
        }
        compiledProgram = compiled;
        return {compiled->program, compiled->states.at(nodes.at(state).state.Key())};
    }
    explicit MimoToolGrammar(const std::map<std::string, std::vector<std::string>> &tools,
                             bool requireCall = false, bool singleCall = false)
        : required(requireCall), single(singleCall) {
        if (tools.empty()) throw std::invalid_argument("MiMo grammar needs tools.");
        for (const auto &entry : tools) {
            CheckName(entry.first); names.push_back(entry.first);
            bodies.push_back({"</function>"});
            for (const auto &key : entry.second) {
                CheckName(key); bodies.back().push_back("<parameter=" + key + ">");
            }
        }
        Intern(State());
    }
    void Bind(std::shared_ptr<const MimoToolVocabulary> tokens, const std::set<int> &endTokens) {
        std::lock_guard<std::mutex> guard(mutex);
        if (!tokens) throw std::invalid_argument("MiMo grammar needs a vocabulary.");
        if (BoundTo(tokens->tokens, endTokens)) return;
        vocabulary = std::move(tokens); stops = endTokens;
    }
    void Bind(std::shared_ptr<const std::vector<std::string>> tokens, const std::set<int> &endTokens) {
        std::lock_guard<std::mutex> guard(mutex);
        if (BoundTo(tokens, endTokens)) return;
        vocabulary = std::make_shared<MimoToolVocabulary>(std::move(tokens)); stops = endTokens;
    }
    int Advance(int state, const std::string &text) {
        std::lock_guard<std::mutex> guard(mutex);
        return AdvanceBytes(state, text);
    }
    bool CanEnd(int state) {
        std::lock_guard<std::mutex> guard(mutex);
        return EndAllowed(state);
    }
    int AdvanceToken(int state, int token) {
        std::lock_guard<std::mutex> guard(mutex);
        if (!vocabulary || token < 0 || token >= (int)vocabulary->tokens->size()) return -1;
        if (stops.count(token)) return EndAllowed(state) ? state : -1;
        if ((*vocabulary->tokens)[token].empty()) return -1;
        return AdvanceBytes(state, (*vocabulary->tokens)[token]);
    }
    std::shared_ptr<const std::vector<uint32_t>> Mask(int state, bool requireToken = true) {
        std::lock_guard<std::mutex> guard(mutex);
        if (!vocabulary || state < 0) throw std::logic_error("Invalid MiMo grammar state.");
        if (nodes.at(state).mask) return nodes[state].mask;
        const auto &tokens = *vocabulary->tokens;
        const auto phase = nodes[state].state.phase;
        const bool freePhase = phase == Text || phase == Reasoning || phase == Value;
        auto mask = std::make_shared<std::vector<uint32_t>>(vocabulary->plainMask.size(), 0);
        if (freePhase) *mask = vocabulary->plainMask;
        auto allow = [&](int token, bool valid) {
            const auto bit = uint32_t(1) << (token % 32);
            if (valid) (*mask)[token / 32] |= bit;
            else (*mask)[token / 32] &= ~bit;
        };
        if (!PlainText(state)) {
            for (int c = 0; c < 256; ++c) {
                const auto &ids = vocabulary->plainByFirstByte[c];
                if (ids.empty()) continue;
                int next = Step(state, (unsigned char)c);
                if ((freePhase && PlainText(next)) || (!freePhase && next < 0)) continue;
                for (int token : ids) {
                    int current = next;
                    // These tokens contain no '<'. Once an unfinished marker
                    // returns to unrestricted text, the remaining bytes are legal.
                    for (size_t i = 1; i < tokens[token].size() && current >= 0 && !PlainText(current); ++i)
                        current = Step(current, (unsigned char)tokens[token][i]);
                    allow(token, current >= 0);
                }
            }
        }
        // Marker-bearing tokens keep the generic byte-by-byte state machine.
        for (int token : vocabulary->withMarker)
            allow(token, AdvanceBytes(state, tokens[token]) >= 0);
        // Stop tokens override their textual spelling, including empty entries.
        for (int token : stops) if (token >= 0 && token < (int)tokens.size())
            allow(token, EndAllowed(state));
        if (requireToken && std::none_of(mask->begin(), mask->end(), [](uint32_t word) { return word != 0; }))
            throw std::runtime_error("MiMo grammar has no legal token; refusing unconstrained fallback.");
        nodes[state].mask = mask;
        return mask;
    }
};

} // namespace fastllm
#endif
