#include "solarx.h"
#include <fstream>
#include <chrono>
#include <iostream>

int main(int argc, char **argv) {
    try {
        if (argc < 7) {
            std::cerr << "Usage: duo_solarx VIDEO_WEIGHTS DEVICE LATENT_H LATENT_W OUTPUT.rgb "
                         "CHUNK.video.f32 ...\n";
            return 2;
        }
        int h = std::stoi(argv[3]), w = std::stoi(argv[4]);
        if (h < 2 || w < 2 || h % 2 || w % 2 || h > 256 || w > 256)
            throw std::invalid_argument("invalid latent geometry");
        fastllm::duo::live::SolarX codec(argv[1], std::stoi(argv[2]));
        std::ofstream out(argv[5], std::ios::binary);
        if (!out)
            throw std::runtime_error("cannot open RGB output");
        for (int i = 6; i < argc; ++i) {
            std::ifstream in(argv[i], std::ios::binary | std::ios::ate);
            auto packed = at::empty({5, h / 2, w / 2, 24, 2, 2}, at::kFloat);
            if (!in || in.tellg() != int64_t(packed.nbytes()))
                throw std::runtime_error("wrong input size");
            in.seekg(0);
            in.read(static_cast<char *>(packed.data_ptr()), packed.nbytes());
            if (!in)
                throw std::runtime_error("incomplete latent read");
            auto start = std::chrono::steady_clock::now();
            auto rgb = codec.decode(packed.permute({3, 0, 1, 4, 2, 5}).reshape({24, 5, h, w})).cpu();
            double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            out.write(static_cast<const char *>(rgb.data_ptr()), rgb.nbytes());
            if (!out)
                throw std::runtime_error("RGB write failed");
            std::cout << "block " << i - 6 << " frames " << rgb.size(0) << " seconds " << seconds
                      << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
