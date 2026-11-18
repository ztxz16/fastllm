#pragma once
#include "httplib.h"
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>

// The provider waits for the producer to commit one complete packet. The same
// HTTP response carries every block, so playback needs no per-block round trip.
template <class Next> void serveAvcStream(httplib::Response &r, Next next) {
    r.set_header("Cache-Control", "no-store");
    r.set_chunked_content_provider(
        "application/octet-stream", [next = std::move(next)](size_t, httplib::DataSink &sink) mutable {
            auto path = next();
            if (!path) {
                sink.done();
                return true;
            }
            std::ifstream in(*path, std::ios::binary);
            if (!in)
                return false;
            std::string bytes(std::istreambuf_iterator<char>(in), {});
            return !bytes.empty() && !in.bad() && sink.write(bytes.data(), bytes.size());
        });
}
