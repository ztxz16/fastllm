#include "models/minimax_h3/runtime.h"
#include "models/minimax_h3/model_source.h"
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
                           initialAudio, vaePrecision, pika, baseModel;
        std::vector<int> devices{0};
        bool fp32 = false, stepsSpecified = false;
        bool widthSpecified = false, heightSpecified = false;
        bool referenceImageSpecified = false, referenceVideoSpecified = false;
        std::string firstFrame, lastFrame;
        std::vector<std::pair<Reference::Kind, std::string>> references;
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (i + 1 >= argc)
                throw std::invalid_argument("missing value for " + arg);
            std::string v = argv[++i];
            if (arg == "--model")
                model = v;
            else if (arg == "--pika")
                pika = v;
            else if (arg == "--base-model")
                baseModel = v;
            else if (arg == "--prompt")
                options.prompt = v;
            else if (arg == "--output")
                output = v;
            else if (arg == "--ffmpeg")
                ffmpeg = v;
            else if (arg == "--width") {
                options.width = std::stoi(v);
                widthSpecified = true;
            } else if (arg == "--height") {
                options.height = std::stoi(v);
                heightSpecified = true;
            } else if (arg == "--frames")
                options.frames = std::stoi(v);
            else if (arg == "--steps") {
                options.steps = std::stoi(v);
                stepsSpecified = true;
            }
            else if (arg == "--seed")
                options.seed = std::stoull(v);
            else if (arg == "--device" || arg == "--devices")
                devices = fastllm::nomas::parseParallelDevices(v);
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
            else if (arg == "--first-frame")
                firstFrame = v;
            else if (arg == "--last-frame")
                lastFrame = v;
            else if (arg == "--reference-image-short-edge") {
                options.referenceImageShortEdge = std::stoi(v);
                referenceImageSpecified = true;
            } else if (arg == "--reference-video-short-edge") {
                options.referenceVideoShortEdge = std::stoi(v);
                referenceVideoSpecified = true;
            } else if (arg == "--reference-image")
                references.push_back({Reference::Kind::Image, v});
            else if (arg == "--reference-video")
                references.push_back({Reference::Kind::Video, v});
            else if (arg == "--reference-audio")
                references.push_back({Reference::Kind::Audio, v});
            else
                throw std::invalid_argument("unknown argument " + arg);
        }
        auto source = resolveModelSource(model, baseModel, pika);
        model = source.baseDirectory;
        pika = source.pikaDirectory;
        if (!pika.empty()) {
            if (stepsSpecified && options.steps != 6)
                throw std::invalid_argument("Pika H3 Flex requires --steps 6");
            options.steps = 6;
            if (!widthSpecified) options.width = 832;
            if (!heightSpecified) options.height = 480;
            int edge = std::min(options.width, options.height);
            if (!referenceImageSpecified) options.referenceImageShortEdge = edge <= 544 ? 1024 : 2048;
            if (!referenceVideoSpecified) options.referenceVideoShortEdge = edge <= 480 ? 480 : edge <= 544 ? 544 : 768;
        }
        auto videoType = parseVideoPrecision(vaePrecision, fp32);
        if (!firstFrame.empty())
            options.firstFrame =
                readReference(firstFrame, Reference::Kind::Image, options.frames, ffmpeg).frames[0];
        if (!lastFrame.empty())
            options.lastFrame =
                readReference(lastFrame, Reference::Kind::Image, options.frames, ffmpeg).frames[0];
        for (const auto &ref : references)
            options.references.push_back(readReference(ref.second, ref.first, options.frames, ffmpeg,
                                                       "ffprobe", options.referenceVideoShortEdge));
        options.validate();
        if (!traceDirectory.empty())
            std::filesystem::create_directories(traceDirectory);
        Runtime reader(devices[0], true);
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
        Pipeline pipeline(model, devices, fp32, videoType);
        if (!pika.empty()) pipeline.loadPika(pika);
        std::string last;
        int current = -1;
        std::cerr << "Precision: " << (fp32 ? "fp32" : "bf16") << ", video VAE: " << precisionName(videoType)
                  << ", TP: " << devices.size() << "\n";
        auto start = std::chrono::steady_clock::now();
        auto result = pipeline.generateMP4(
            options, output, ffmpeg,
            [&](const std::string &stage, int i, int n) {
                if (stage != last || i != current)
                    std::cerr << stage << " " << i << "/" << n << "\n";
                last = stage;
                current = i;
            },
            trace, initialVideo.empty() ? nullptr : &video, initialAudio.empty() ? nullptr : &audio,
            !traceDirectory.empty());
        if (!traceDirectory.empty()) {
            std::ofstream rgb(traceDirectory + "/output.rgb", std::ios::binary),
                audioOut(traceDirectory + "/output.audio.f32", std::ios::binary);
            rgb.write(reinterpret_cast<const char *>(result.rgb.data()), result.rgb.size());
            audioOut.write(reinterpret_cast<const char *>(result.audio.data()), result.audio.size() * 4);
            if (!rgb || !audioOut)
                throw std::runtime_error("cannot write media diagnostics");
        }
        std::cerr << "Saved " << output << " in "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
                  << " seconds\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
