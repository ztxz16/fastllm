#include <cstdio>
#include <cstring>

#include <set>
#include <string>
#include <vector>
#ifndef _WIN32
#include <sys/socket.h>
#include <unistd.h>
#endif

#include "../../example/apiserver/socket_writer.h"
#include "../../example/apiserver/http_request_reader.h"
#include "../../example/apiserver/openai_output_parser.h"
#include "../../example/apiserver/stop_parser.h"
#include "../../include/utils/stop_token_matcher.h"
#include "../../include/utils/stop_string_matcher.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "check failed at line %d: %s\n", \
                     __LINE__, #condition); \
        return 1; \
    } \
} while (false)

int main() {
    CHECK(FormatSseData("{\"ok\":true}") ==
          "data: {\"ok\":true}\r\n\r\n");
    CHECK(FormatSseData("[DONE]") == "data: [DONE]\r\n\r\n");
    CHECK(FormatSseData("{\n\t\"ok\": true\n}") ==
          "data: {\r\ndata: \t\"ok\": true\r\ndata: }\r\n\r\n");

    CHECK(ResolveOpenAIFinishReason(false, false, 42, 256) == "stop");
    CHECK(ResolveOpenAIFinishReason(false, false, 256, 256) == "length");
    CHECK(ResolveOpenAIFinishReason(false, true, 256, 256) == "stop");
    CHECK(ResolveOpenAIFinishReason(true, false, 256, 256) == "tool_calls");

    const std::string getRequest =
        "GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n";
    CHECK(IsHttpRequestComplete(getRequest.data(), getRequest.size()));
    const std::string emptyPost =
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Host: localhost\r\nContent-Length: 0\r\n\r\n";
    CHECK(IsHttpRequestComplete(emptyPost.data(), emptyPost.size()));
    const std::string partialPost =
        "POST /v1/chat/completions HTTP/1.1\r\n"
        "Content-Length: 5\r\n\r\nabc";
    CHECK(!IsHttpRequestComplete(partialPost.data(), partialPost.size()));
    const std::string completePost = partialPost + "de";
    CHECK(IsHttpRequestComplete(completePost.data(), completePost.size()));
    CHECK(!IsHttpRequestComplete("GET /health HTTP/1.1\r\n", 22));

    auto encodeStop = [](const std::string &stop) {
        OpenAIStopEncoding encoding;
        if (stop == "single-a") {
            encoding.tokenSequence = {101};
        } else if (stop == "single-b") {
            encoding.tokenSequence = {202};
        } else if (stop == "multiple") {
            encoding.tokenSequence = {1, 2};
        }
        return encoding;
    };

    auto exactTokenLookup = [](const std::string &stop, int &tokenId) {
        if (stop == "vocab-entry") {
            tokenId = 303;
            return true;
        }
        return false;
    };
    auto fallbackEncode = [](const std::string &stop) {
        if (stop == "fallback-single") {
            return std::vector<int>{404};
        }
        return std::vector<int>{1, 2};
    };
    OpenAIStopEncoding vocabEncoding = EncodeOpenAIStop(
        "vocab-entry", exactTokenLookup, fallbackEncode);
    CHECK(vocabEncoding.exactTokenId == 303);
    CHECK(vocabEncoding.tokenSequence == std::vector<int>({1, 2}));
    OpenAIStopEncoding fallbackEncoding = EncodeOpenAIStop(
        "fallback-single", exactTokenLookup, fallbackEncode);
    CHECK(fallbackEncoding.exactTokenId == -1);
    CHECK(fallbackEncoding.tokenSequence == std::vector<int>{404});

    std::multiset<int> stopTokenIds;
    std::vector<std::vector<int>> parsedStopSequences;
    std::vector<std::string> parsedStopStrings;
    std::string stopError;
    CHECK(ParseOpenAIStop(json11::Json(), encodeStop, stopTokenIds,
                          parsedStopSequences, parsedStopStrings, stopError));
    CHECK(stopTokenIds.empty());
    CHECK(parsedStopSequences.empty());
    CHECK(parsedStopStrings.empty());
    CHECK(stopError.empty());

    CHECK(ParseOpenAIStop(json11::Json("single-a"), encodeStop,
                          stopTokenIds, parsedStopSequences,
                          parsedStopStrings, stopError));
    CHECK(stopTokenIds.size() == 1);
    CHECK(stopTokenIds.count(101) == 1);
    CHECK(parsedStopSequences.empty());
    CHECK(parsedStopStrings == std::vector<std::string>{"single-a"});

    parsedStopStrings.clear();
    stopTokenIds.clear();
    CHECK(ParseOpenAIStop(json11::Json::array{"single-a", "single-b"},
                          encodeStop, stopTokenIds, parsedStopSequences,
                          parsedStopStrings, stopError));
    CHECK(stopTokenIds.size() == 2);
    CHECK(stopTokenIds.count(101) == 1);
    CHECK(stopTokenIds.count(202) == 1);
    CHECK(parsedStopSequences.empty());
    parsedStopStrings.clear();
    stopTokenIds.clear();

    CHECK(ParseOpenAIStop(json11::Json("multiple"), encodeStop,
                          stopTokenIds, parsedStopSequences,
                          parsedStopStrings, stopError));
    CHECK(stopTokenIds.empty());
    CHECK(parsedStopSequences ==
          std::vector<std::vector<int>>({{1, 2}}));
    parsedStopStrings.clear();

    stopTokenIds.insert(7);
    CHECK(!ParseOpenAIStop(json11::Json::array{"single-a", 42},
                           encodeStop, stopTokenIds, parsedStopSequences,
                           parsedStopStrings, stopError));
    CHECK(stopTokenIds.size() == 1);
    CHECK(stopTokenIds.count(7) == 1);
    CHECK(parsedStopSequences ==
          std::vector<std::vector<int>>({{1, 2}}));
    CHECK(stopError.find("string or an array of strings") !=
          std::string::npos);

    CHECK(!ParseOpenAIStop(json11::Json(true), encodeStop,
                           stopTokenIds, parsedStopSequences,
                           parsedStopStrings, stopError));
    CHECK(stopTokenIds.size() == 1);
    CHECK(stopTokenIds.count(7) == 1);
    CHECK(stopError.find("string or an array of strings") !=
          std::string::npos);

    CHECK(!ParseOpenAIStop(json11::Json(""), encodeStop,
                           stopTokenIds, parsedStopSequences,
                           parsedStopStrings, stopError));
    CHECK(stopTokenIds.size() == 1);
    CHECK(stopTokenIds.count(7) == 1);
    CHECK(stopError.find("must not be empty") != std::string::npos);
    CHECK(parsedStopStrings.empty());

    std::vector<std::vector<int>> stopSequences{{11, 12, 13}};
    std::vector<int> pendingStopTokens;
    std::vector<int> readyTokens;
    CHECK(PushStopToken(stopSequences, pendingStopTokens, 11,
                        readyTokens) == 0);
    CHECK(readyTokens.empty());
    CHECK(pendingStopTokens == std::vector<int>{11});
    CHECK(PushStopToken(stopSequences, pendingStopTokens, 12,
                        readyTokens) == 0);
    CHECK(readyTokens.empty());
    CHECK(PushStopToken(stopSequences, pendingStopTokens, 13,
                        readyTokens) == 3);
    CHECK(readyTokens.empty());
    CHECK(pendingStopTokens.empty());

    CHECK(PushStopToken(stopSequences, pendingStopTokens, 11,
                        readyTokens) == 0);
    CHECK(PushStopToken(stopSequences, pendingStopTokens, 99,
                        readyTokens) == 0);
    CHECK(readyTokens == std::vector<int>({11, 99}));
    CHECK(pendingStopTokens.empty());

    CHECK(PushStopToken(stopSequences, pendingStopTokens, 11,
                        readyTokens) == 0);
    CHECK(PushStopToken(stopSequences, pendingStopTokens, 11,
                        readyTokens) == 0);
    CHECK(readyTokens == std::vector<int>{11});
    CHECK(pendingStopTokens == std::vector<int>{11});
    CHECK(PushStopToken(stopSequences, pendingStopTokens, 12,
                        readyTokens) == 0);
    FlushPendingStopTokens(pendingStopTokens, readyTokens);
    CHECK(readyTokens == std::vector<int>({11, 12}));
    CHECK(pendingStopTokens.empty());

    stopSequences = {{21, 22}, {22, 23}};
    CHECK(PushStopToken(stopSequences, pendingStopTokens, 21,
                        readyTokens) == 0);
    CHECK(PushStopToken(stopSequences, pendingStopTokens, 22,
                        readyTokens) == 2);
    CHECK(readyTokens.empty());
    CHECK(pendingStopTokens.empty());

    pendingStopTokens = {31, 32};
    std::vector<int> emittedTokens;
    FlushPendingStopTokensTo(pendingStopTokens, [&](int token) {
        emittedTokens.push_back(token);
    });
    CHECK(emittedTokens == std::vector<int>({31, 32}));
    CHECK(pendingStopTokens.empty());

    std::vector<std::string> stopStrings{"<|end|>"};
    std::string pendingStopText;
    std::string readyText;
    CHECK(!PushStopText(stopStrings, pendingStopText, "answer?<",
                        readyText));
    CHECK(readyText == "answer?");
    CHECK(pendingStopText == "<");
    CHECK(!PushStopText(stopStrings, pendingStopText, "|end",
                        readyText));
    CHECK(readyText.empty());
    CHECK(pendingStopText == "<|end");
    CHECK(PushStopText(stopStrings, pendingStopText, "|>ignored",
                       readyText));
    CHECK(readyText.empty());
    CHECK(pendingStopText.empty());

    CHECK(!PushStopText(stopStrings, pendingStopText, "plain text",
                        readyText));
    CHECK(readyText == "plain text");
    CHECK(pendingStopText.empty());

    CHECK(OpenAIReasoningParser::PromptEndsInReasoning(
        "<|im_start|>assistant\n<think>\n"));
    CHECK(!OpenAIReasoningParser::PromptEndsInReasoning(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    CHECK(OpenAIReasoningParser::PromptEndsInReasoning(
        "<think>old</think><think>new"));

    OpenAIReasoningParser reasoningParser(true);
    std::string reasoningText;
    std::string answerText;
    for (const std::string &fragment :
         std::vector<std::string>{"work", "\n", "</", "think", ">",
                                  "\n", "OK"}) {
        OpenAIReasoningDelta delta = reasoningParser.Push(fragment);
        reasoningText += delta.reasoningContent;
        answerText += delta.content;
    }
    CHECK(reasoningText == "work\n");
    CHECK(answerText == "\nOK");
    CHECK(reasoningParser.Flush().Empty());

    OpenAIReasoningParser longMarkerParser(true);
    OpenAIReasoningDelta longMarkerDelta =
        longMarkerParser.Push("analysis</thinking>answer");
    CHECK(longMarkerDelta.reasoningContent == "analysis");
    CHECK(longMarkerDelta.content == "answer");

    OpenAIReasoningParser disabledReasoningParser(false);
    OpenAIReasoningDelta disabledDelta =
        disabledReasoningParser.Push("plain answer");
    CHECK(disabledDelta.reasoningContent.empty());
    CHECK(disabledDelta.content == "plain answer");

    OpenAIReasoningParser unclosedReasoningParser(true);
    OpenAIReasoningDelta unclosedDelta =
        unclosedReasoningParser.Push("unfinished </thi");
    OpenAIReasoningDelta unclosedTrailing = unclosedReasoningParser.Flush();
    CHECK(unclosedDelta.reasoningContent +
          unclosedTrailing.reasoningContent == "unfinished </thi");
    CHECK(unclosedDelta.content.empty());
    CHECK(unclosedTrailing.content.empty());

    OpenAIToolCallParser toolParser(true);
    std::string toolText;
    std::vector<OpenAIParsedToolCall> parsedToolCalls;
    for (const std::string &fragment : std::vector<std::string>{
             "\n<tool", "_call>\n<function=builtin_", "web_search>\n",
             "</function>\n</tool_call>"}) {
        OpenAIToolCallDelta delta = toolParser.Push(fragment);
        toolText += delta.content;
        parsedToolCalls.insert(parsedToolCalls.end(), delta.toolCalls.begin(),
                               delta.toolCalls.end());
    }
    CHECK(toolText == "\n");
    CHECK(parsedToolCalls.size() == 1);
    CHECK(parsedToolCalls[0].name == "builtin_web_search");
    CHECK(parsedToolCalls[0].arguments == "{}");
    CHECK(toolParser.Flush().Empty());

    OpenAIToolCallParser typoToolParser(true);
    OpenAIToolCallDelta typoToolDelta = typoToolParser.Push(
        "<tool_call>\n<fuction=builtin_web_search>\n"
        "<parameter=query>\n原神 最新消息 2024\n</parameter>\n"
        "</function>\n</tool_call>");
    CHECK(typoToolDelta.content.empty());
    CHECK(typoToolDelta.toolCalls.size() == 1);
    CHECK(typoToolDelta.toolCalls[0].name == "builtin_web_search");
    CHECK(typoToolDelta.toolCalls[0].arguments ==
          "{\"query\":\"原神 最新消息 2024\"}");

    OpenAIToolCallParser parameterToolParser(true);
    OpenAIToolCallDelta parameterDelta = parameterToolParser.Push(
        "<tool_call>\n<function=search>\n"
        "<parameter=query>\n原神 latest\n</parameter>\n"
        "<parameter=limit>\n3\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=lookup>\n"
        "<parameter=filters>\n{\"lang\":\"zh\"}\n</parameter>\n"
        "</function>\n</tool_call>");
    CHECK(parameterDelta.toolCalls.size() == 2);
    CHECK(parameterDelta.toolCalls[0].name == "search");
    CHECK(parameterDelta.toolCalls[0].arguments ==
          "{\"limit\":3,\"query\":\"原神 latest\"}");
    CHECK(parameterDelta.toolCalls[1].name == "lookup");
    CHECK(parameterDelta.toolCalls[1].arguments ==
          "{\"filters\":{\"lang\":\"zh\"}}");

    OpenAIToolCallParser disabledToolParser(false);
    OpenAIToolCallDelta disabledToolDelta = disabledToolParser.Push(
        "literal <tool_call><function=demo></function></tool_call>");
    CHECK(disabledToolDelta.content ==
          "literal <tool_call><function=demo></function></tool_call>");
    CHECK(disabledToolDelta.toolCalls.empty());

    OpenAIToolCallParser unfinishedToolParser(true);
    OpenAIToolCallDelta unfinishedToolDelta = unfinishedToolParser.Push(
        "before<tool_call><function=demo>");
    OpenAIToolCallDelta unfinishedToolTrailing = unfinishedToolParser.Flush();
    CHECK(unfinishedToolDelta.content + unfinishedToolTrailing.content ==
          "before<tool_call><function=demo>");
    CHECK(unfinishedToolDelta.toolCalls.empty());
    CHECK(unfinishedToolTrailing.toolCalls.empty());

#ifndef _WIN32
    int sockets[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);

    const char payload[] = "ready";
    CHECK(WriteAllToSocket(sockets[0], payload, sizeof(payload) - 1));
    char received[sizeof(payload)] = {};
    CHECK(read(sockets[1], received, sizeof(payload) - 1) ==
          static_cast<ssize_t>(sizeof(payload) - 1));
    CHECK(std::memcmp(payload, received, sizeof(payload) - 1) == 0);

    close(sockets[1]);
    CHECK(!WriteAllToSocket(sockets[0], payload, sizeof(payload) - 1));
    close(sockets[0]);
#endif
    return 0;
}
