#pragma once

#include <string>
#include <utility>
#include <vector>

#include "json11.hpp"

struct OpenAIParsedChatInput {
    std::vector<std::pair<std::string, std::string>> messages;
    std::vector<std::string> imageUrls;
};

inline bool ParseOpenAIChatInput(const json11::Json &messages,
                                 const std::string &imagePlaceholder,
                                 OpenAIParsedChatInput &parsed,
                                 std::string &error) {
    OpenAIParsedChatInput candidate;
    error.clear();
    if (!messages.is_array() || messages.array_items().empty()) {
        error = "messages must be a non-empty array";
        return false;
    }

    for (const auto &message : messages.array_items()) {
        if (!message.is_object() || !message["role"].is_string()) {
            error = "each message must be an object with a string role";
            return false;
        }
        const std::string role = message["role"].string_value();
        const auto &content = message["content"];
        std::string rendered;
        if (content.is_string()) {
            rendered = content.string_value();
        } else if (content.is_array()) {
            for (const auto &part : content.array_items()) {
                if (!part.is_object() || !part["type"].is_string()) {
                    error = "each message content part must have a string type";
                    return false;
                }
                const std::string type = part["type"].string_value();
                if (type == "text") {
                    if (!part["text"].is_string()) {
                        error = "text content parts require a string text field";
                        return false;
                    }
                    rendered += part["text"].string_value();
                } else if (type == "image_url") {
                    if (role != "user") {
                        error = "image_url content is only supported in user messages";
                        return false;
                    }
                    if (imagePlaceholder.empty()) {
                        error = "the selected model does not support image input";
                        return false;
                    }
                    const auto &imageUrl = part["image_url"];
                    if (!imageUrl.is_object() ||
                        !imageUrl["url"].is_string() ||
                        imageUrl["url"].string_value().empty()) {
                        error = "image_url content parts require a non-empty image_url.url string";
                        return false;
                    }
                    rendered += imagePlaceholder;
                    candidate.imageUrls.push_back(imageUrl["url"].string_value());
                } else {
                    error = "unsupported message content part type: " + type;
                    return false;
                }
            }
        } else {
            error = "message content must be a string or an array of content parts";
            return false;
        }
        candidate.messages.push_back({role, std::move(rendered)});
    }

    parsed = std::move(candidate);
    return true;
}
