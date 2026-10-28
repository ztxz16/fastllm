#ifndef FASTLLM_MIMO_TOOL_GRAMMAR_H
#define FASTLLM_MIMO_TOOL_GRAMMAR_H

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
    std::shared_ptr<const std::vector<std::string>> vocabulary;
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
    bool EndAllowed(int state) const {
        return nodes.at(state).state.phase == Text && (!required || nodes[state].state.called);
    }
public:
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
    void Bind(std::shared_ptr<const std::vector<std::string>> tokens, const std::set<int> &endTokens) {
        std::lock_guard<std::mutex> guard(mutex);
        if (vocabulary) {
            if (vocabulary != tokens || stops != endTokens)
                throw std::logic_error("MiMo grammar cannot change tokenizer or stop tokens.");
            return;
        }
        vocabulary = std::move(tokens); stops = endTokens;
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
        if (!vocabulary || token < 0 || token >= (int)vocabulary->size()) return -1;
        if (stops.count(token)) return EndAllowed(state) ? state : -1;
        if ((*vocabulary)[token].empty()) return -1;
        return AdvanceBytes(state, (*vocabulary)[token]);
    }
    std::shared_ptr<const std::vector<uint32_t>> Mask(int state) {
        std::lock_guard<std::mutex> guard(mutex);
        if (!vocabulary || state < 0) throw std::logic_error("Invalid MiMo grammar state.");
        if (nodes.at(state).mask) return nodes[state].mask;
        auto mask = std::make_shared<std::vector<uint32_t>>((vocabulary->size() + 31) / 32, 0);
        int count = 0;
        for (int token = 0; token < (int)vocabulary->size(); ++token) {
            bool allowed = stops.count(token) ? EndAllowed(state) :
                !(*vocabulary)[token].empty() && AdvanceBytes(state, (*vocabulary)[token]) >= 0;
            if (allowed) { (*mask)[token / 32] |= uint32_t(1) << (token % 32); ++count; }
        }
        if (!count) throw std::runtime_error("MiMo grammar has no legal token; refusing unconstrained fallback.");
        nodes[state].mask = mask;
        return mask;
    }
};

} // namespace fastllm
#endif
