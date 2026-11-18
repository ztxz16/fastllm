#include "../../../tools/duo/io.h"
#include "httplib.h"
#include <iostream>
#include <random>
#include <sstream>
using namespace fastllm::duo;
using Json = json11::Json;
namespace {
// Wire format: uint32 little-endian JSON length, UTF-8 JSON, then packed
// little-endian float32 tensors. No client-selected filesystem paths.
struct Packet {
    Json meta;
    const std::string &body;
    size_t offset;
    explicit Packet(const std::string &b) : body(b), offset(4) {
        if (b.size() < 4)
            throw std::invalid_argument("missing packet header");
        uint32_t n = 0;
        for (int i = 0; i < 4; ++i)
            n |= uint32_t(uint8_t(b[i])) << (8 * i);
        if (n > 65536 || n > b.size() - 4)
            throw std::invalid_argument("invalid packet header length");
        std::string error;
        meta = Json::parse(b.substr(4, n), error);
        offset += n;
        if (!error.empty() || !meta.is_object())
            throw std::invalid_argument("invalid packet JSON");
    }
    std::vector<float> take(size_t n) {
        if (n > (body.size() - offset) / 4)
            throw std::invalid_argument("truncated tensor");
        std::vector<float> values(n);
        std::memcpy(values.data(), body.data() + offset, n * 4);
        offset += n * 4;
        return values;
    }
    void end() {
        if (offset != body.size())
            throw std::invalid_argument("unexpected tensor payload");
    }
};
void reply(httplib::Response &res, int code, const Json &data) {
    res.status = code;
    res.set_content(data.dump(), "application/json");
}
std::string pack(const Json &meta, const ChunkResult &r) {
    auto json = meta.dump();
    std::string out(4, '\0');
    for (int i = 0; i < 4; ++i)
        out[i] = char((json.size() >> (8 * i)) & 255);
    out += json;
    out.append(reinterpret_cast<const char *>(r.video.data()), r.video.size() * 4);
    out.append(reinterpret_cast<const char *>(r.audio.data()), r.audio.size() * 4);
    return out;
}
std::string id() {
    std::random_device r;
    std::ostringstream o;
    o << std::hex << r() << r() << r() << r();
    return o.str();
}
} // namespace
int main(int argc, char **argv) {
    try {
        if (argc < 5 || argc > 7) {
            std::cerr << "Usage: duo_server MODEL LATENT_HEIGHT LATENT_WIDTH PORT [ADAPTER|-] [DEVICE]\n";
            return 2;
        }
        int h = std::stoi(argv[2]), w = std::stoi(argv[3]), port = std::stoi(argv[4]);
        if (port < 1 || port > 65535)
            throw std::invalid_argument("invalid port");
        ModelSource source(argv[1], argc > 5 && std::string(argv[5]) != "-" ? argv[5] : "");
        Session session(argc > 6 ? std::stoi(argv[6]) : 0, source, h, w);
        const int text = source.entry["config"]["text_dim"].int_value();
        std::mutex state;
        bool busy = false, ready = false;
        std::string active;
        httplib::Server server;
        server.set_payload_max_length(32u * 1024 * 1024);
        server.new_task_queue = [] { return new httplib::ThreadPool(4); };
        server.Get("/health", [&](const auto &, auto &res) {
            std::lock_guard<std::mutex> lock(state);
            reply(res, 200,
                  Json::object{{"backend", "duo"},
                               {"protocol", "fastllm-duo-v1"},
                               {"busy", busy},
                               {"ready", ready},
                               {"input", "intent_features_and_reference_latents"},
                               {"output", "latents"},
                               {"latent_height", h},
                               {"latent_width", w},
                               {"text_dim", text},
                               {"stage", source.stage}});
        });
        server.Post("/v1/session", [&](const auto &req, auto &res) {
            // Parse before taking ownership so malformed requests cannot reset a live session.
            try {
                Packet p(req.body);
                int rows = integer(p.meta, "shared_rows", 1, 32768);
                auto reference = p.take(size_t(h) * w / 4 * 96), shared = p.take(size_t(rows) * text);
                p.end();
                std::string current;
                {
                    std::lock_guard<std::mutex> lock(state);
                    if (busy) {
                        reply(res, 409,
                              Json::object{{"error", "generation busy; cancel and wait before reset"}});
                        return;
                    }
                    busy = true;
                    ready = false;
                    active = current = id();
                }
                try {
                    session.reset(reference, shared, rows);
                    {
                        std::lock_guard<std::mutex> lock(state);
                        busy = false;
                        ready = true;
                    }
                    reply(res, 201, Json::object{{"session", current}, {"next_chunk", 0}});
                } catch (...) {
                    std::lock_guard<std::mutex> lock(state);
                    busy = false;
                    throw;
                }
            } catch (const std::exception &e) {
                reply(res, 400, Json::object{{"error", e.what()}});
            }
        });
        server.Post("/v1/chunk", [&](const auto &req, auto &res) {
            try {
                Packet p(req.body);
                int rows = integer(p.meta, "rows", 1, 32768), chunk = integer(p.meta, "chunk", 0, 10000000);
                auto features = p.take(size_t(rows) * text);
                p.end();
                uint64_t seed = p.meta["seed"].is_null() ? 0 : integer(p.meta, "seed", 0, 2147483647);
                std::string current = p.meta["session"].string_value();
                {
                    std::lock_guard<std::mutex> lock(state);
                    if (current.empty() || current != active || !ready || busy) {
                        reply(res, 409, Json::object{{"error", "stale, unready, or busy session"}});
                        return;
                    }
                    busy = true;
                }
                try {
                    auto r = session.generate(chunk, features, rows, seed);
                    auto meta = resultMetadata(r, h, w).object_items();
                    meta["session"] = current;
                    auto body = pack(meta, r);
                    {
                        std::lock_guard<std::mutex> lock(state);
                        busy = false;
                    }
                    res.set_content(std::move(body), "application/x-duo-tensors");
                } catch (...) {
                    std::lock_guard<std::mutex> lock(state);
                    busy = false;
                    ready = false;
                    throw;
                }
            } catch (const std::exception &e) {
                reply(res, 400, Json::object{{"error", e.what()}});
            }
        });
        server.Post("/v1/cancel", [&](const auto &req, auto &res) {
            std::string error;
            auto j = Json::parse(req.body, error);
            std::lock_guard<std::mutex> lock(state);
            if (!error.empty() || active.empty() || j["session"].string_value() != active) {
                reply(res, 409, Json::object{{"error", "stale session"}});
                return;
            }
            session.cancel();
            ready = false;
            reply(res, 202, Json::object{{"session", active}, {"cancelled", true}});
        });
        std::cerr << "Duo latent service listening on 127.0.0.1:" << port << "\n";
        if (!server.listen("127.0.0.1", port))
            throw std::runtime_error("cannot listen");
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
