#pragma once

#include <algorithm>
#include <string>
#include <utility>
#include <map>

#include "json11.hpp"
#include <vector>

inline std::string ResolveOpenAIFinishReason(bool hasToolCalls,
                                             bool matchedStop,
                                             int outputTokens,
                                             int outputTokenLimit) {
    if (hasToolCalls) {
        return "tool_calls";
    }
    if (matchedStop) {
        return "stop";
    }
    if (outputTokenLimit > 0 && outputTokens >= outputTokenLimit) {
        return "length";
    }
    return "stop";
}

struct OpenAIReasoningDelta {
    std::string reasoningContent;
    std::string content;

    bool Empty() const {
        return reasoningContent.empty() && content.empty();
    }
};

class OpenAIReasoningParser {
public:
    explicit OpenAIReasoningParser(bool inReasoning)
        : inReasoning(inReasoning) {}

    static bool PromptEndsInReasoning(const std::string &prompt) {
        const size_t start = prompt.rfind("<think>");
        if (start == std::string::npos) {
            return false;
        }
        const size_t shortEnd = prompt.rfind("</think>");
        const size_t longEnd = prompt.rfind("</thinking>");
        const size_t end = shortEnd == std::string::npos ? longEnd :
            (longEnd == std::string::npos ? shortEnd :
             std::max(shortEnd, longEnd));
        return end == std::string::npos || start > end;
    }

    OpenAIReasoningDelta Push(const std::string &fragment) {
        OpenAIReasoningDelta delta;
        if (fragment.empty()) {
            return delta;
        }
        if (!inReasoning) {
            delta.content = fragment;
            return delta;
        }

        pending += fragment;
        size_t markerPosition = std::string::npos;
        size_t markerLength = 0;
        for (const std::string &marker : EndMarkers()) {
            const size_t position = pending.find(marker);
            if (position != std::string::npos &&
                (markerPosition == std::string::npos ||
                 position < markerPosition)) {
                markerPosition = position;
                markerLength = marker.size();
            }
        }
        if (markerPosition != std::string::npos) {
            delta.reasoningContent = pending.substr(0, markerPosition);
            delta.content = pending.substr(markerPosition + markerLength);
            pending.clear();
            inReasoning = false;
            return delta;
        }

        size_t held = 0;
        for (const std::string &marker : EndMarkers()) {
            const size_t maxPrefix = std::min(marker.size() - 1,
                                              pending.size());
            for (size_t length = maxPrefix; length > 0; --length) {
                if (pending.compare(pending.size() - length, length,
                                    marker, 0, length) == 0) {
                    held = std::max(held, length);
                    break;
                }
            }
        }
        delta.reasoningContent = pending.substr(0, pending.size() - held);
        pending.erase(0, pending.size() - held);
        return delta;
    }

    OpenAIReasoningDelta Flush() {
        OpenAIReasoningDelta delta;
        if (inReasoning) {
            delta.reasoningContent.swap(pending);
        } else {
            delta.content.swap(pending);
        }
        return delta;
    }

    bool InReasoning() const {
        return inReasoning;
    }

private:
    static const std::vector<std::string> &EndMarkers() {
        static const std::vector<std::string> markers = {
            "</think>", "</thinking>"
        };
        return markers;
    }

    bool inReasoning = false;
    std::string pending;
};

struct OpenAIParsedToolCall {
    std::string name;
    std::string arguments;
};

struct OpenAIToolCallDelta {
    std::string content;
    std::vector<OpenAIParsedToolCall> toolCalls;

    bool Empty() const {
        return content.empty() && toolCalls.empty();
    }
};

class OpenAIToolCallParser {
public:
    explicit OpenAIToolCallParser(bool enabled) : enabled(enabled) {}

    OpenAIToolCallDelta Push(const std::string &fragment) {
        OpenAIToolCallDelta delta;
        if (fragment.empty()) {
            return delta;
        }
        if (!enabled) {
            delta.content = fragment;
            return delta;
        }

        pending += fragment;
        while (true) {
            if (!inToolCall) {
                const size_t open = pending.find(OpenMarker());
                if (open == std::string::npos) {
                    const size_t held = HeldOpenMarkerPrefix(pending);
                    delta.content += pending.substr(0, pending.size() - held);
                    pending.erase(0, pending.size() - held);
                    return delta;
                }
                delta.content += pending.substr(0, open);
                toolBuffer = pending.substr(open);
                pending.clear();
                inToolCall = true;
            } else if (!pending.empty()) {
                toolBuffer += pending;
                pending.clear();
            }

            const size_t close = toolBuffer.find(CloseMarker());
            if (close == std::string::npos) {
                return delta;
            }
            const size_t blockEnd = close + CloseMarker().size();
            const std::string block = toolBuffer.substr(0, blockEnd);
            OpenAIParsedToolCall parsed;
            if (ParseBlock(block, parsed)) {
                delta.toolCalls.push_back(std::move(parsed));
            } else {
                delta.content += block;
            }
            pending = toolBuffer.substr(blockEnd);
            toolBuffer.clear();
            inToolCall = false;
        }
    }

    OpenAIToolCallDelta Flush() {
        OpenAIToolCallDelta delta;
        if (inToolCall) {
            delta.content = toolBuffer + pending;
        } else {
            delta.content = pending;
        }
        pending.clear();
        toolBuffer.clear();
        inToolCall = false;
        return delta;
    }

private:
    static const std::string &OpenMarker() {
        static const std::string marker = "<tool_call>";
        return marker;
    }

    static const std::string &CloseMarker() {
        static const std::string marker = "</tool_call>";
        return marker;
    }

    static std::string Trim(const std::string &value) {
        const size_t first = value.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            return "";
        }
        const size_t last = value.find_last_not_of(" \t\r\n");
        return value.substr(first, last - first + 1);
    }

    static size_t HeldOpenMarkerPrefix(const std::string &value) {
        const size_t maxPrefix = std::min(OpenMarker().size() - 1,
                                          value.size());
        for (size_t length = maxPrefix; length > 0; --length) {
            if (value.compare(value.size() - length, length,
                              OpenMarker(), 0, length) == 0) {
                return length;
            }
        }
        return 0;
    }

    static json11::Json ParseParameterValue(const std::string &raw) {
        const std::string value = Trim(raw);
        std::string error;
        json11::Json parsed = json11::Json::parse(value, error);
        if (error.empty()) {
            return parsed;
        }
        return json11::Json(value);
    }

    static std::string CompactJson(const json11::Json &value) {
        if (value.is_object()) {
            std::string result = "{";
            bool first = true;
            for (const auto &item : value.object_items()) {
                if (!first) {
                    result += ",";
                }
                result += json11::Json(item.first).dump();
                result += ":";
                result += CompactJson(item.second);
                first = false;
            }
            result += "}";
            return result;
        }
        if (value.is_array()) {
            std::string result = "[";
            bool first = true;
            for (const auto &item : value.array_items()) {
                if (!first) {
                    result += ",";
                }
                result += CompactJson(item);
                first = false;
            }
            result += "]";
            return result;
        }
        return value.dump();
    }

    static bool ParseBlock(const std::string &block,
                           OpenAIParsedToolCall &parsed) {
        static const std::vector<std::string> functionOpenMarkers = {
            "<function=", "<fuction="
        };
        const std::string functionClose = "</function>";
        size_t function = std::string::npos;
        const std::string *functionOpen = nullptr;
        for (const auto &marker : functionOpenMarkers) {
            const size_t position = block.find(marker);
            if (position != std::string::npos &&
                (function == std::string::npos || position < function)) {
                function = position;
                functionOpen = &marker;
            }
        }
        if (function == std::string::npos || functionOpen == nullptr) {
            return false;
        }
        const size_t nameStart = function + functionOpen->size();
        const size_t nameEnd = block.find('>', nameStart);
        const size_t bodyEnd = block.find(functionClose, nameEnd);
        if (nameEnd == std::string::npos || bodyEnd == std::string::npos) {
            return false;
        }
        parsed.name = Trim(block.substr(nameStart, nameEnd - nameStart));
        if (parsed.name.empty()) {
            return false;
        }

        const std::string parameterOpen = "<parameter=";
        const std::string parameterClose = "</parameter>";
        std::map<std::string, json11::Json> arguments;
        size_t cursor = nameEnd + 1;
        while (true) {
            const size_t parameter = block.find(parameterOpen, cursor);
            if (parameter == std::string::npos || parameter >= bodyEnd) {
                break;
            }
            const size_t parameterNameStart = parameter + parameterOpen.size();
            const size_t parameterNameEnd = block.find('>', parameterNameStart);
            if (parameterNameEnd == std::string::npos ||
                parameterNameEnd >= bodyEnd) {
                return false;
            }
            const size_t parameterValueEnd =
                block.find(parameterClose, parameterNameEnd + 1);
            if (parameterValueEnd == std::string::npos ||
                parameterValueEnd > bodyEnd) {
                return false;
            }
            const std::string name = Trim(block.substr(
                parameterNameStart, parameterNameEnd - parameterNameStart));
            if (name.empty()) {
                return false;
            }
            arguments[name] = ParseParameterValue(block.substr(
                parameterNameEnd + 1,
                parameterValueEnd - parameterNameEnd - 1));
            cursor = parameterValueEnd + parameterClose.size();
        }
        parsed.arguments = CompactJson(json11::Json(arguments));
        return true;
    }

    bool enabled = false;
    bool inToolCall = false;
    std::string pending;
    std::string toolBuffer;
};

struct OpenAIOutputDelta {
    std::string reasoningContent;
    std::string content;
    std::vector<OpenAIParsedToolCall> toolCalls;

    bool Empty() const {
        return reasoningContent.empty() && content.empty() && toolCalls.empty();
    }
};

class OpenAIOutputParser {
public:
    OpenAIOutputParser(bool inReasoning, bool toolsEnabled)
        : reasoningParser(inReasoning), toolCallParser(toolsEnabled) {}

    OpenAIOutputDelta Push(const std::string &fragment) {
        OpenAIOutputDelta output;
        OpenAIReasoningDelta reasoning = reasoningParser.Push(fragment);
        output.reasoningContent = std::move(reasoning.reasoningContent);
        OpenAIToolCallDelta tools = toolCallParser.Push(reasoning.content);
        output.content = std::move(tools.content);
        output.toolCalls = std::move(tools.toolCalls);
        return output;
    }

    OpenAIOutputDelta Flush() {
        OpenAIOutputDelta output;
        OpenAIReasoningDelta reasoning = reasoningParser.Flush();
        output.reasoningContent = std::move(reasoning.reasoningContent);
        OpenAIToolCallDelta tools = toolCallParser.Push(reasoning.content);
        output.content = std::move(tools.content);
        output.toolCalls = std::move(tools.toolCalls);
        OpenAIToolCallDelta trailing = toolCallParser.Flush();
        output.content += trailing.content;
        output.toolCalls.insert(output.toolCalls.end(),
                                trailing.toolCalls.begin(),
                                trailing.toolCalls.end());
        return output;
    }

private:
    OpenAIReasoningParser reasoningParser;
    OpenAIToolCallParser toolCallParser;
};
