#pragma once

#include <string>
#include <utility>
#include <vector>

#include "json11.hpp"
#include "socket_writer.h"

inline const char *HttpReasonPhrase(int status) {
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        default: return "Unknown";
    }
}

inline std::string BuildFixedHttpResponse(
        int status,
        const std::string &body,
        const std::string &contentType = "application/json; charset=utf-8",
        const std::vector<std::pair<std::string, std::string>> &headers = {}) {
    std::string response = "HTTP/1.1 " + std::to_string(status) + " " +
                           HttpReasonPhrase(status) + "\r\n";
    response += "Content-Type: " + contentType + "\r\n";
    response += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    response += "Connection: close\r\n";
    response += "Server: fastllm api server\r\n";
    for (const auto &header : headers) {
        response += header.first + ": " + header.second + "\r\n";
    }
    response += "\r\n";
    response += body;
    return response;
}

inline json11::Json OpenAIHttpError(const std::string &message,
                                    const std::string &type,
                                    const std::string &code) {
    return json11::Json::object {
        {"error", json11::Json::object {
            {"message", message},
            {"type", type},
            {"param", nullptr},
            {"code", code}
        }}
    };
}

inline bool WriteFixedJsonResponse(
        int socket,
        int status,
        const json11::Json &body,
        const std::vector<std::pair<std::string, std::string>> &headers = {}) {
    return WriteAllToSocket(
        socket, BuildFixedHttpResponse(status, body.dump(),
                                       "application/json; charset=utf-8",
                                       headers));
}
