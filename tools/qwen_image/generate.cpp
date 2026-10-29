#include "models/qwen_image/runtime.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace fastllm::qwen_image;
int main(int argc, char **argv) {
    try {
        GenerateOptions options;
        std::string model, output = "output.png", traceDirectory;
        int device = 0;
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (i + 1 >= argc)
                throw std::invalid_argument("missing argument");
            std::string value = argv[++i];
            if (key == "--model")
                model = value;
            else if (key == "--prompt")
                options.prompt = value;
            else if (key == "--output")
                output = value;
            else if (key == "--width")
                options.width = std::stoi(value);
            else if (key == "--height")
                options.height = std::stoi(value);
            else if (key == "--steps")
                options.steps = std::stoi(value);
            else if (key == "--image") {
                std::ifstream f(value, std::ios::binary);
                if (!f)
                    throw std::invalid_argument("cannot read image");
                options.images.push_back(decodePNG(std::string((std::istreambuf_iterator<char>(f)), {})));
            } else if (key == "--trace")
                traceDirectory = value;
            else if (key == "--reference-resolution")
                options.referenceResolution = std::stoi(value);
            else if (key == "--seed")
                options.seed = std::stoull(value);
            else if (key == "--device")
                device = std::stoi(value);
            else
                throw std::invalid_argument("unknown argument: " + key);
        }
        if (model.empty())
            throw std::invalid_argument("--model is required");
        Trace trace;
        if (!traceDirectory.empty()) {
            std::filesystem::create_directories(traceDirectory);
            trace = [&](const std::string &name, const std::vector<float> &values) {
                std::ofstream file(std::filesystem::path(traceDirectory) / (name + ".f32"), std::ios::binary);
                file.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
                file.close();
                if (!file)
                    throw std::runtime_error("cannot write trace: " + name);
            };
        }
        Pipeline pipeline(model, device);
        auto png = encodePNG(pipeline.generate(
            options,
            [](const std::string &stage, int step, int total) {
                std::cerr << stage << " " << step << "/" << total << "\n";
            },
            trace));
        std::ofstream f(output, std::ios::binary);
        f.write(reinterpret_cast<const char *>(png.data()), png.size());
        f.close();
        if (!f)
            throw std::runtime_error("cannot write output");
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
