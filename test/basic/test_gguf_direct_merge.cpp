#include "ggufdirectmerge.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unistd.h>
using namespace fastllm;
static void Check(bool ok, const char *why) { if (!ok) throw std::runtime_error(why); }
struct Files {
    std::string dir, a, b;
    Files() {
        char path[] = "/tmp/fastllm-direct-merge-XXXXXX";
        Check(mkdtemp(path) != nullptr, "mkdtemp failed");
        dir = path; a = dir + "/a.bin"; b = dir + "/b.bin";
    }
    ~Files() { unlink(a.c_str()); unlink(b.c_str()); rmdir(dir.c_str()); }
};
int main() {
    try {
        Files files;
        const ggml_type types[] = {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16,
            GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_Q2_K,
            GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS,
            GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ4_NL};
        int cases = 0;
        for (auto type : types) for (int rank : {1, 2}) {
            std::vector<std::unique_ptr<Data>> metadata;
            std::vector<ReadGGUFTask> tasks;
            std::vector<uint8_t> expected;
            std::ofstream a(files.a, std::ios::binary), b(files.b, std::ios::binary);
            a << "nonzero-offset"; b << "other-shard";
            for (int i = 0; i < 3; ++i) {
                std::vector<int> dims = rank == 1 ? std::vector<int>{256 * (i + 1)} :
                    std::vector<int>{4 * (i + 1), 256};
                metadata.emplace_back(new Data(DATA_GGUF_FORMAT, type, dims));
                auto tensor = *static_cast<ggml_tensor*>(metadata.back()->ggmlTensor);
                tensor.dims = dims;
                // Packed expert tasks use arbitrary strides on singleton axes.
                for (int d = rank; d < GGML_MAX_DIMS; ++d) tensor.nb[d] = tensor.nb[0];
                auto &file = i == 1 ? b : a;
                const uint64_t offset = file.tellp();
                std::vector<uint8_t> bytes(ggml_nbytes(&tensor));
                for (size_t j = 0; j < bytes.size(); ++j) bytes[j] = (j * 19 + i * 61) & 255;
                expected.insert(expected.end(), bytes.begin(), bytes.end());
                file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
                file << "gap";
                std::string name = "source" + std::to_string(i);
                tasks.emplace_back(name, nullptr, tensor, i == 1 ? files.b : files.a, offset);
            }
            a.close(); b.close();
            std::vector<const ReadGGUFTask*> inputs;
            for (const auto &task : tasks) inputs.push_back(&task);
            std::vector<int> dims; uint64_t bytes = 0;
            Check(GetGGUFDirectMergeLayout(inputs, dims, bytes), "eligible source layout rejected");
            Check(dims == (rank == 1 ? std::vector<int>{1536} : std::vector<int>{24, 256}), "wrong merged dimensions");
            Check(bytes == expected.size(), "wrong merged byte count");
            std::vector<uint8_t> actual(bytes); size_t offset = 0;
            for (const auto *input : inputs) {
                const size_t count = ggml_nbytes(&input->tensor);
                ReadGGUFTensorBytes(input->fileName, input->offset, actual.data() + offset, count);
                offset += count;
            }
            Check(actual == expected, "direct read changed bytes or concatenation order");
            tasks[1].replaceType = GGUFWeightReplaceRule::GGUFWeightReplaceForceFP16;
            Check(!GetGGUFDirectMergeLayout(inputs, dims, bytes) && dims.empty() && bytes == 0, "conversion must fall back");
            tasks[1].replaceType = GGUFWeightReplaceRule::GGUFWeightReplaceDirect;
            tasks[1].tensor.nb[0]++;
            Check(!GetGGUFDirectMergeLayout(inputs, dims, bytes), "strided source must fall back");
            tasks[1].tensor.nb[0]--;
            tasks[1].tensor.type = type == GGML_TYPE_F32 ? GGML_TYPE_F16 : GGML_TYPE_F32;
            Check(!GetGGUFDirectMergeLayout(inputs, dims, bytes), "mixed storage types must fall back");
            tasks[1].tensor.type = type;
            Check(!GetGGUFDirectMergeLayout({inputs[0]}, dims, bytes), "single input must fall back");
            tasks[0].tensor.type = GGML_TYPE_IQ3_XXS_R4;
            Check(!GetGGUFDirectMergeLayout(inputs, dims, bytes), "row-interleaved format must fall back");
            tasks[0].tensor.type = type;
            tasks[1].tensor.dims.push_back(1);
            Check(!GetGGUFDirectMergeLayout(inputs, dims, bytes), "different ranks must fall back");
            ++cases;
        }
        std::vector<int> dims; uint64_t bytes;
        Check(!GetGGUFDirectMergeLayout({}, dims, bytes), "empty merge must fall back");
        uint8_t buffer[32];
        std::ifstream input(files.a, std::ios::binary | std::ios::ate);
        const uint64_t end = input.tellg();
        bool failed = false;
        try { ReadGGUFTensorBytes(files.a, end - 1, buffer, sizeof(buffer)); }
        catch (const std::runtime_error &) { failed = true; }
        Check(failed, "truncated input was not reported");
        failed = false;
        try { ReadGGUFTensorBytes(files.a, UINT64_MAX, buffer, 1); }
        catch (const std::runtime_error &) { failed = true; }
        Check(failed, "overflowing offset was not reported");
        std::cout << "PASS: generic GGUF direct merge " << cases << " layouts, byte order and fallback/read errors\n";
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
