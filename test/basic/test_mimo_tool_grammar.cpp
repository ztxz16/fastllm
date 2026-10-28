#include "models/mimo_tool_grammar.h"
#include <iostream>
#include <stdexcept>

using fastllm::MimoToolGrammar;
static void Check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
int main() {
    try {
        const std::map<std::string, std::vector<std::string>> tools = {
            {"Bash", {"command", "description"}}, {"Read", {"path"}}, {"Ping", {}}};
        MimoToolGrammar grammar(tools);
        const std::string call = "<tool_call>\n<function=Bash>\n<parameter=command>echo '你好'; x < 3\n"
            "</parameter>\n</function>\n</tool_call>";
        int state = 0;
        for (char c : call) { state = grammar.Advance(state, std::string(1, c)); Check(state >= 0, "valid byte prefix"); }
        Check(grammar.CanEnd(state), "closed call can end");
        Check(grammar.Advance(0, call) == state, "token boundaries cannot change state");
        Check(grammar.Advance(0, "<think>example: <tool_call>function is invalid</think>" + call) >= 0,
              "reasoning text is outside tool grammar");
        Check(grammar.Advance(0, "thinking\n" + call + call) >= 0, "text and multiple calls");
        Check(grammar.Advance(0, "<tool_call><function=Ping></function></tool_call>") >= 0, "zero parameters");
        Check(grammar.Advance(0, "<tool_call> <function= Bash ></function> </tool_call>") >= 0, "optional gaps");
        for (const auto &bad : {
                "<tool_call>\nfunction_calls>", "<tool_call>\n<function=Unknown>",
                "<tool_call><function=Read><parameter=command>",
                "<tool_call><function=Bash><parameter=command>x</function>",
                "<tool_call><function=Bash><parameter=command>x<tool_call>",
                "<tool_call><function=Bash></function>junk", "<tool_call>\n\n"})
            Check(grammar.Advance(0, bad) < 0, "malformed call must be rejected");
        const std::string value = "<tool_call><function=Bash><parameter=command>";
        int inValue = grammar.Advance(0, value);
        Check(!grammar.CanEnd(inValue), "no EOS inside call");
        int longValue = grammar.Advance(inValue, std::string(20000, 'x'));
        Check(grammar.Advance(longValue, "</parameter></function></tool_call>") >= 0, "long argument retains grammar");
        int branch = grammar.Advance(inValue, "</parameter><parameter=description>");
        Check(branch >= 0 && grammar.Advance(inValue, "still command") >= 0, "private speculative branches");
        Check(grammar.Advance(inValue, "</parameX") >= 0, "non-marker raw argument text");

        auto vocab = std::make_shared<std::vector<std::string>>(std::initializer_list<std::string>{
            "<tool_call>", "\n", "function", "<", "function=Bash>",
            "</function></tool_call>", "<eos>", "<parameter=command>x</parameter>", "", "oops"});
        vocab->resize(128); // Padding in lm_head has no tokenizer entry.
        grammar.Bind(vocab, {6});
        int header = grammar.AdvanceToken(grammar.AdvanceToken(0, 0), 1);
        auto mask = grammar.Mask(header);
        auto allows = [&](int id) { return ((*mask)[id / 32] >> (id % 32)) & 1; };
        Check(!allows(2) && allows(3) && !allows(6), "actual failing header mask");
        int afterName = grammar.AdvanceToken(grammar.AdvanceToken(header, 3), 4);
        Check(afterName >= 0 && grammar.AdvanceToken(afterName, 7) >= 0, "multi-transition token");
        Check(grammar.AdvanceToken(afterName, 6) < 0, "EOS is not a raw value");
        int closed = grammar.AdvanceToken(afterName, 5);
        Check(closed >= 0 && grammar.AdvanceToken(closed, 6) >= 0, "EOS after full closure");
        Check(grammar.Mask(header) == mask, "immutable cached state mask");
        Check(mask->size() == 4 && mask->back() == 0, "padded output vocabulary stays masked");

        MimoToolGrammar required(tools, true, true);
        Check(!required.CanEnd(0), "required tool blocks early EOS");
        int done = required.Advance(0, call);
        Check(required.CanEnd(done), "required call completed");
        Check(required.Advance(done, "<tool_call>") < 0, "parallel=false blocks a second call");
        bool invalid = false;
        try { MimoToolGrammar bad({{"bad>name", {}}}); } catch (const std::invalid_argument &) { invalid = true; }
        Check(invalid, "unrepresentable names fail explicitly");
        std::cout << "ALL_PASS MiMo tool grammar\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
