#include "io.h"
#include <iostream>
using namespace fastllm::duo;
namespace fs = std::filesystem;
int main(int argc, char **argv) {
    try {
        if (argc < 4 || argc > 6) {
            std::cerr
                << "Usage: duo_generate MODEL REQUEST.json OUTPUT_DIRECTORY [ADAPTER_DIRECTORY|-] [DEVICE]\n";
            return 2;
        }
        ModelSource source(argv[1], argc > 4 && std::string(argv[4]) != "-" ? argv[4] : "");
        auto j = readJson(argv[2]);
        fs::path base = fs::absolute(argv[2]).parent_path(), out = argv[3];
        const int h = integer(j, "latent_height", 2, 256), w = integer(j, "latent_width", 2, 256),
                  rows = integer(j, "shared_rows", 1, 32768),
                  history = j["history_chunks"].is_null() ? 2 : integer(j, "history_chunks", 1, 128);
        Session session(argc > 5 ? std::stoi(argv[5]) : 0, source, h, w, history,
                        j["low_memory"].bool_value());
        auto load = [&](const json11::Json &f, size_t n) { return readFloats(base / f.string_value(), n); };
        int text = source.entry["config"]["text_dim"].int_value();
        session.reset(load(j["reference"], size_t(h) * w / 4 * 96), load(j["shared"], size_t(rows) * text),
                      rows);
        if (!j["chunks"].is_array() || j["chunks"].array_items().empty())
            throw std::invalid_argument("request needs chunks");
        fs::create_directories(out);
        json11::Json::array results;
        int index = 0;
        for (const auto &chunk : j["chunks"].array_items()) {
            int count = integer(chunk, "rows", 1, 32768);
            std::vector<std::vector<float>> noise;
            for (const auto &file : chunk["noise"].array_items())
                noise.push_back(load(file, noise.size() % 2 ? 58 * 32 : size_t(5) * h * w / 4 * 96));
            auto r = session.generate(
                index, load(chunk["intent"], size_t(count) * text), count,
                chunk["seed"].is_null() ? 0 : uint64_t(integer(chunk, "seed", 0, 2147483647)), noise);
            std::string name = "chunk-" + std::to_string(index++);
            writeFloats(out / (name + ".video.f32"), r.video);
            writeFloats(out / (name + ".audio.f32"), r.audio);
            auto meta = resultMetadata(r, h, w).object_items();
            meta["video"] = name + ".video.f32";
            meta["audio"] = name + ".audio.f32";
            results.push_back(meta);
            std::cout << json11::Json(meta).dump() << std::endl;
        }
        std::ofstream f(out / "result.json");
        f << json11::Json(results).dump() << "\n";
        if (!f)
            throw std::runtime_error("cannot write result manifest");
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
