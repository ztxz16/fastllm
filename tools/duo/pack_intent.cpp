#include "intent.h"
#include <iostream>
int main(int argc, char **argv) {
    try {
        if (argc != 3) {
            std::cerr << "Usage: duo_pack_intent MODEL NEW_CACHE_DIRECTORY\n";
            return 2;
        }
        fastllm::duo::live::Intent(argv[1], 0).writeCache(argv[2]);
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
