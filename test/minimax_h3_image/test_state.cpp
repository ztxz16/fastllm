#include "models/minimax_h3_image/model.h"
#include <iostream>
#include <stdexcept>
using namespace fastllm::minimax_h3_image;

int main(int argc, char **argv) {
    try {
        if (argc != 3)
            throw std::invalid_argument("usage: minimax_h3_image_state MODEL REFERENCE_IMAGE");
        Pipeline pipeline(argv[1]);
        Options options;
        options.prompt = "A red panda beside a mountain stream, autumn forest";
        options.width = 64;
        options.height = 96;
        auto expected = pipeline.generate(options);
        auto same = [&] {
            auto actual = pipeline.generate(options);
            if (actual.rgba != expected.rgba)
                throw std::runtime_error("request state leaked across requests");
            std::cout << "repeat byte-identical" << std::endl;
        };
        Options edit = options;
        edit.prompt = "Change Picture 1 to a snowy winter forest";
        edit.images.push_back(readImage(argv[2]));
        edit.width = 128;
        edit.height = 64;
        edit.seed = 17;
        pipeline.generate(edit);
        same();
        for (const char *stage : {"reference", "text", "prepare", "denoise", "decode"}) {
            // A prepare cancellation needs an uncached schedule. The preceding
            // text cancellation has already dropped the persistent components.
            bool cancelled = false;
            try {
                pipeline.generate(std::string(stage) == "reference" ? edit : options,
                                  [&](const std::string &name, int done, int) {
                    if (name == stage && done >= (name == "decode" ? 1 : 2)) {
                        cancelled = true;
                        throw std::runtime_error("test cancellation");
                    }
                });
            } catch (const std::runtime_error &error) {
                if (std::string(error.what()) != "test cancellation")
                    throw;
            }
            if (!cancelled || pipeline.memoryPolicy().host->used() != 0)
                throw std::runtime_error("cancelled request retained model state");
            std::cout << stage << " cancellation released host cache" << std::endl;
            if (std::string(stage) != "text")
                same();
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
