// HOSTTRIM unit check (header-only). Build: g++ -std=c++17 -Iinclude test/basic/t0_host_malloc_trim.cpp -lpthread
// Run without the flag (prints nothing from the trimmer), with FASTLLM_HOST_MALLOC_TRIM_SECONDS=1 (prints two
// "[HostMem] malloc-trim:" lines, the first with released_mib > 0) and with an invalid value such as 1x (nothing).
#include "utils/host_malloc_trim.h"

#include <cstdlib>
#include <vector>

int main() {
    std::vector<char *> blocks;
    for (int i = 0; i < 20000; i++) blocks.push_back((char *)std::malloc(100000));
    for (size_t i = 0; i < blocks.size(); i++) if (i % 50) std::free(blocks[i]);  // keep every 50th: fragmented arena
    fastllm::HostMallocTrimStartOnce();
    std::this_thread::sleep_for(std::chrono::milliseconds(2500));
    return 0;
}
