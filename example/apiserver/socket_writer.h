#pragma once

#include <cerrno>
#include <cstdio>
#include <string>

#ifdef _WIN32
#include <winsock2.h>
#else
#include <sys/socket.h>
#endif

inline bool WriteAllToSocket(int socket, const char *data, size_t size) {
    size_t written = 0;
    while (written < size) {
#ifdef _WIN32
        int ret = send(socket, data + written,
                       static_cast<int>(size - written), 0);
        if (ret == SOCKET_ERROR) {
            return false;
        }
#else
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags |= MSG_NOSIGNAL;
#endif
        ssize_t ret = send(socket, data + written, size - written, flags);
        if (ret < 0 && errno == EINTR) {
            continue;
        }
        if (ret < 0) {
            return false;
        }
#endif
        if (ret == 0) {
            return false;
        }
        written += static_cast<size_t>(ret);
    }
    return true;
}

inline bool WriteAllToSocket(int socket, const std::string &data) {
    return WriteAllToSocket(socket, data.data(), data.size());
}


inline std::string FormatSseData(const std::string &payload) {
    std::string event;
    size_t lineStart = 0;
    while (true) {
        size_t lineEnd = payload.find('\n', lineStart);
        event += "data: ";
        event.append(payload, lineStart,
                     lineEnd == std::string::npos
                         ? std::string::npos
                         : lineEnd - lineStart);
        event += "\r\n";
        if (lineEnd == std::string::npos) {
            break;
        }
        lineStart = lineEnd + 1;
    }
    event += "\r\n";
    return event;
}

inline bool WriteHttpChunk(int socket, const std::string &payload) {
    char header[32];
    int headerLength = std::snprintf(header, sizeof(header), "%zx\r\n",
                                     payload.size());
    return headerLength > 0
        && WriteAllToSocket(socket, header, static_cast<size_t>(headerLength))
        && WriteAllToSocket(socket, payload)
        && WriteAllToSocket(socket, "\r\n", 2);
}
