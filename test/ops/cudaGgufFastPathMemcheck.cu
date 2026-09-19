#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdio>

int main(int argc, char **argv) {
    // Initialize the sanitizer before loading FastLLM's CUDA-using globals.
    if (argc < 2 || cudaFree(nullptr) != cudaSuccess) return 2;
    void *handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!handle) { std::fprintf(stderr, "%s\n", dlerror()); return 3; }
    using Entry = int (*)(int, char **);
    Entry entry = reinterpret_cast<Entry>(dlsym(handle, "main"));
    if (!entry) { std::fprintf(stderr, "%s\n", dlerror()); return 4; }
    return entry(argc - 1, argv + 1);
}
