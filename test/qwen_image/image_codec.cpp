// CPU-only utility for differential PNG/Lanczos tests, no model required.
#include "models/qwen_image/runtime.h"
#include <fstream>
#include <iostream>
int main(int argc, char **argv) {
    try {
        if (argc != 5)
            return 2;
        std::ifstream f(argv[1], std::ios::binary);
        auto image = fastllm::qwen_image::decodePNG(std::string((std::istreambuf_iterator<char>(f)), {}));
        image = fastllm::qwen_image::resizeImage(image, std::stoi(argv[3]), std::stoi(argv[4]));
        std::ofstream out(argv[2], std::ios::binary);
        out.write((const char *)image.rgba.data(), image.rgba.size());
        return !out;
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
