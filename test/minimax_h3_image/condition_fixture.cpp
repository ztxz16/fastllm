// Numerical fixture: mixed text/vision tags, two asymmetric references and six PDD steps.
#include "models/minimax_h3_image/model.h"
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace fastllm::minimax_h3_image;
int main(int argc, char **argv) {
    try {
        if (argc != 3)
            throw std::runtime_error("usage: condition_fixture MODEL FIXTURE_DIR");
        const std::string dir = argv[2];
        Runtime rt(0), full(0, true);
        auto read = [&](const std::string &name, int rows, int cols) {
            std::vector<float> data(size_t(rows) * cols);
            std::ifstream f(dir + "/" + name + ".f32", std::ios::binary);
            f.read(reinterpret_cast<char *>(data.data()), data.size() * 4);
            if (!f)
                throw std::runtime_error("invalid fixture: " + name);
            return full.upload(data.data(), rows, cols);
        };
        std::filesystem::create_directories(dir + "/native");
        auto save = [&](const std::string &name, const Tensor &tensor) {
            auto data = full.download(tensor);
            std::ofstream f(dir + "/native/" + name + ".f32", std::ios::binary);
            f.write(reinterpret_cast<char *>(data.data()), data.size() * 4);
            if (!f)
                throw std::runtime_error("cannot save " + name);
        };
        auto text = read("text", 11, 5120), refs = read("references", 12, 96), latent = read("noise", 24, 96);
        Conditioning conditioning{{1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1}, {{4, 6}, {6, 4}}};
        Transformer model(rt, full, argv[1]);
        model.prepare();
        for (int step = 0; step < 6; ++step) {
            auto velocity = model.forward(
                full.concat({refs, latent}), text, 128, 192, step,
                [&](const std::string &name, const Tensor &value) {
                    if (name == "block.0" || name == "block.24" || name == "block.49")
                        save("step." + std::to_string(step) + "." + name, value);
                },
                &conditioning);
            latent = full.add(latent, full.affine(velocity, model.sigmas()[step] - model.sigmas()[step + 1]));
            save("latent." + std::to_string(step), latent);
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
