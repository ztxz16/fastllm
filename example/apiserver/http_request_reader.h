#pragma once

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <string>

inline bool IsHttpRequestComplete(const char *buffer, size_t size) {
    if (buffer == nullptr || size == 0) {
        return false;
    }

    const std::string request(buffer, size);
    const size_t headerEnd = request.find("\r\n\r\n");
    if (headerEnd == std::string::npos) {
        return false;
    }

    size_t contentLength = 0;
    bool hasContentLength = false;
    size_t lineStart = request.find("\r\n");
    if (lineStart == std::string::npos || lineStart >= headerEnd) {
        return false;
    }
    lineStart += 2;

    while (lineStart < headerEnd) {
        const size_t lineEnd = request.find("\r\n", lineStart);
        if (lineEnd == std::string::npos || lineEnd > headerEnd) {
            return false;
        }
        const size_t colon = request.find(':', lineStart);
        if (colon != std::string::npos && colon < lineEnd) {
            std::string name = request.substr(lineStart, colon - lineStart);
            for (char &c : name) {
                c = static_cast<char>(std::tolower(
                    static_cast<unsigned char>(c)));
            }
            if (name == "content-length") {
                size_t valueStart = colon + 1;
                while (valueStart < lineEnd &&
                       std::isspace(static_cast<unsigned char>(request[valueStart]))) {
                    valueStart++;
                }
                if (valueStart == lineEnd) {
                    return false;
                }
                errno = 0;
                char *valueEnd = nullptr;
                const char *value = request.c_str() + valueStart;
                unsigned long long parsed = std::strtoull(value, &valueEnd, 10);
                if (errno != 0 || valueEnd != request.c_str() + lineEnd ||
                    parsed > std::numeric_limits<size_t>::max()) {
                    return false;
                }
                contentLength = static_cast<size_t>(parsed);
                hasContentLength = true;
            }
        }
        lineStart = lineEnd + 2;
    }

    const size_t bodyStart = headerEnd + 4;
    return !hasContentLength ||
           (contentLength <= size - bodyStart);
}
