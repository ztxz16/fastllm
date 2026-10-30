#include "models/minimax_h3/runtime.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace fastllm::minimax_h3;
int main(int argc, char **argv) {
    try {
        GenerateOptions options;
        std::string model, output = "minimax-h3.mp4", ffmpeg = "ffmpeg", traceDirectory, initialVideo,
                           initialAudio, vaePrecision;
        int device = 0;
        bool fp32 = false;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (i + 1 >= argc)
                throw std::invalid_argument("missing value for " + arg);
            std::string v = argv[++i];
            if (arg == "--model")
                model = v;
            else if (arg == "--prompt")
                options.prompt = v;
            else if (arg == "--output")
                output = v;
            else if (arg == "--ffmpeg")
                ffmpeg = v;
            else if (arg == "--width")
                options.width = std::stoi(v);
            else if (arg == "--height")
                options.height = std::stoi(v);
            else if (arg == "--frames")
                options.frames = std::stoi(v);
            else if (arg == "--steps")
                options.steps = std::stoi(v);
            else if (arg == "--seed")
                options.seed = std::stoull(v);
            else if (arg == "--device")
                device = std::stoi(v);
            else if (arg == "--trace")
                traceDirectory = v;
            else if (arg == "--precision") {
                if (v != "fp32" && v != "bf16")
                    throw std::invalid_argument("precision must be fp32 or bf16");
                fp32 = v == "fp32";
            } else if (arg == "--vae-precision")
                vaePrecision = v;
            else if (arg == "--initial-video")
                initialVideo = v;
            else if (arg == "--initial-audio")
                initialAudio = v;
            else
                throw std::invalid_argument("unknown argument " + arg);
        }
        if (model.empty())
            throw std::invalid_argument("--model is required");
        auto videoType = parseVideoPrecision(vaePrecision, fp32);
        options.validate();
        if (!traceDirectory.empty())
            std::filesystem::create_directories(traceDirectory);
        Runtime reader(device, true);
        auto trace = [&](const std::string &name, const Tensor &x) {
            if (traceDirectory.empty())
                return;
            auto data = reader.download(x);
            std::ofstream f(traceDirectory + "/" + name + ".f32", std::ios::binary);
            f.write(reinterpret_cast<const char *>(data.data()), data.size() * 4);
            if (!f)
                throw std::runtime_error("cannot write diagnostic tensor");
        };
        auto read = [](const std::string &path) {
            std::vector<float> out;
            if (path.empty())
                return out;
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (!f)
                throw std::runtime_error("cannot open " + path);
            auto n = f.tellg();
            if (n < 0 || size_t(n) % 4)
                throw std::runtime_error("invalid FP32 fixture");
            out.resize(size_t(n) / 4);
            f.seekg(0);
            f.read(reinterpret_cast<char *>(out.data()), n);
            if (!f)
                throw std::runtime_error("truncated fixture");
            return out;
        };
        auto video = read(initialVideo), audio = read(initialAudio);
        Pipeline pipeline(model, device, fp32, videoType);
        std::string last;
        int current = -1;
        std::cerr << "Precision: " << (fp32 ? "fp32" : "bf16") << ", video VAE: " << precisionName(videoType)
                  << "\n";
        auto start = std::chrono::steady_clock::now();
        auto result = pipeline.generate(
            options,
            [&](const std::string &stage, int i, int n) {
                if (stage != last || i != current)
                    std::cerr << stage << " " << i << "/" << n << "\n";
                last = stage;
                current = i;
            },
            trace, initialVideo.empty() ? nullptr : &video, initialAudio.empty() ? nullptr : &audio);
        if (!traceDirectory.empty()) {
            std::ofstream rgb(traceDirectory + "/output.rgb", std::ios::binary),
                audioOut(traceDirectory + "/output.audio.f32", std::ios::binary);
            rgb.write(reinterpret_cast<const char *>(result.rgb.data()), result.rgb.size());
            audioOut.write(reinterpret_cast<const char *>(result.audio.data()), result.audio.size() * 4);
            if (!rgb || !audioOut)
                throw std::runtime_error("cannot write media diagnostics");
        }
        writeMP4(result, output, ffmpeg);
        std::cerr << "Saved " << output << " in "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                  << " seconds\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
