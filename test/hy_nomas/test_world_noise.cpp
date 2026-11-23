#include "world_noise.h"
#include <array>
#include <stdexcept>

using namespace fastllm::nomas::world_noise;

int main() {
    // Official CPU BF16 worker, seed 42, 3x3 latent grid. Five frames span
    // 45 spatial positions per channel, crossing normal-generation blocks.
    std::mt19937 rng(42);
    auto bank = draw(rng, horizon * 9);
    auto check = [](const std::vector<uint16_t> &src, size_t offset, size_t tokens,
                    const std::array<uint16_t, 9> &expected) {
        std::vector<float> out(tokens * channels);
        read(src, offset, out);
        const size_t indices[] = {0,1,8,17,31,32,45,198,out.size()-1};
        for (size_t i = 0; i < expected.size(); ++i)
            if (pack(out[indices[i]]) != expected[i])
                throw std::runtime_error("seeded world noise differs from reference");
    };
    check(bank, 30*9, 5*9, {49149,16229,16277,16149,16387,15507,16421,48533,48946});
    check(draw(rng, 30*9), 0, 30*9, {16150,48971,49165,16376,48946,49045,48725,49060,48952});
    check(draw(rng, 5*9), 0, 5*9, {16135,48945,48990,16188,49140,16145,48983,16179,16317});
}
