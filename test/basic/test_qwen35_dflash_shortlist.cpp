#include "models/qwen3_5.h"
#include "devices/cuda/fastllm-cuda.cuh"
#include "executor.h"
#include "gguf.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <unistd.h>
using namespace fastllm;
struct Probe : Qwen3_5Model {
    using Qwen3_5Model::PrepareDFlashDraftShortlist;
    using Qwen3_5Model::dflashSelectorTopK;
    using Qwen3_5Model::dflashDraftTokenIds;
    using Qwen3_5Model::dflashDraftLmHeads;
    using Qwen3_5Model::threadTpLmHeadScheme;
    using Qwen3_5Model::RunDFlashLmHead;
    using Qwen3_5Model::MapDFlashShortlistCandidates;
};
static void Check(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

static void TestGgufShortlist() {
    constexpr int vocab = 512, width = 256;
    char path[] = "/tmp/fastllm-shortlist-XXXXXX";
    const int fd = mkstemp(path);
    Check(fd >= 0, "temporary shortlist file");
    close(fd);
    const char *previous = std::getenv("FASTLLM_DFLASH_DRAFT_TOKEN_IDS");
    struct Cleanup {
        const char *path;
        bool hadPrevious;
        std::string previous;
        ~Cleanup() {
            if (hadPrevious) setenv("FASTLLM_DFLASH_DRAFT_TOKEN_IDS", previous.c_str(), 1);
            else unsetenv("FASTLLM_DFLASH_DRAFT_TOKEN_IDS");
            unlink(path);
        }
    } cleanup{path, previous != nullptr, previous ? previous : ""};
    Check(setenv("FASTLLM_DFLASH_DRAFT_TOKEN_IDS", path, 1) == 0, "set shortlist path");
    Probe model;
    model.dflashSelectorTopK = 16;
    Data &source = model.weight["lm_head.weight"];
    source.dataType = DATA_GGUF_FORMAT;
    source.ggmlType = GGML_TYPE_Q4_K;
    source.isGGUFData = true;
    source.disableGGUFRepack = true;
    source.forceGGUFFp32Dequant = true;
    source.Resize({vocab, width});
    source.Allocate();
    source.strides = {1}; // Match the GGUF loader metadata.
    std::vector<float> values(vocab * width);
    for (int row = 0; row < vocab; ++row)
        for (int k = 0; k < width; ++k)
            values[row * width + k] = row == 510 ? 2.0f :
                0.005f * (row % 13 + 1) + 0.0001f * (k % 19);
    quantize_row_q4_K_ref(values.data(), (block_q4_K*)source.cpuData, values.size());
    const size_t rowBytes = source.GetBytes() / vocab;
    std::vector<uint8_t> original(source.cpuData, source.cpuData + source.GetBytes());
    source.ToDevice(CUDA, std::vector<int>{0});
    auto writeIds = [&](const std::vector<int> &ids) {
        std::ofstream file(path);
        for (int id : ids) file << id << '\n';
        file.close();
        Check(!file.fail(), "write shortlist IDs");
    };
    for (int count : {129, 256}) {
        std::vector<int> ids;
        for (int i = 0; i < count - 1; ++i) ids.push_back(2*i);
        ids.push_back(510);
        writeIds(ids);
        Check(model.PrepareDFlashDraftShortlist({0}), "GGUF shortlist preparation");
        Data &head = model.dflashDraftLmHeads.at(0);
        Check(head.dataType == source.dataType && head.ggmlType == source.ggmlType,
              "GGUF quantization format changed");
        Check(head.disableGGUFRepack && head.forceGGUFFp32Dequant,
              "GGUF execution flags changed");
        Check(head.dims == std::vector<int>({256, width}), "GGUF padded shape");
        std::vector<uint8_t> packed(head.GetBytes());
        FastllmCudaCopyFromDeviceToHost(packed.data(), head.cudaData, packed.size());
        for (int row = 0; row < 256; ++row)
            Check(std::memcmp(packed.data() + row*rowBytes,
                  original.data() + ids[std::min(row,count-1)]*rowBytes, rowBytes) == 0,
                  "GGUF row encoding or padding changed");
        for (DataType type : {FLOAT16, BFLOAT16}) for (int rows = 1; rows <= 8; ++rows) {
            Data input(FLOAT32, {1,rows,width}, std::vector<float>(rows*width, 0.01f));
            input.ToDevice(CUDA, std::vector<int>{0});
            ToDataType(input, type);
            Data output, full, empty;
            model.RunDFlashLmHead(0, input, source, empty, output, true);
            model.RunDFlashLmHead(0, input, source, empty, full, false);
            Check(output.dims == std::vector<int>({1,rows,count}), "GGUF logical shape");
            Check(full.dims == std::vector<int>({1,rows,vocab}), "GGUF full-head fallback");
            ToDataType(output, FLOAT32);
            ToDataType(full, FLOAT32);
            Data top;
            TopK(output, top, 16);
            top.ToDevice(CPU);
            model.MapDFlashShortlistCandidates(top);
            const float *pairs = (const float*)top.cpuData;
            for (int row = 0; row < rows; ++row) {
                Check(int(pairs[row*32]) == 510, "GGUF highest candidate changed");
                std::set<int> unique;
                for (int k = 0; k < 16; ++k) {
                    const int id = int(pairs[row*32+2*k]);
                    Check(std::binary_search(ids.begin(), ids.end(), id), "GGUF mapped token missing");
                    unique.insert(id);
                }
                Check(unique.size() == 16, "GGUF duplicate candidates");
            }
            output.ToDevice(CPU);
            full.ToDevice(CPU);
            const float *a = (const float*)output.cpuData, *b = (const float*)full.cpuData;
            for (int row = 0; row < rows; ++row) for (int k = 0; k < count; ++k)
                Check(a[row*count+k] == b[row*vocab+ids[k]], "GGUF selected logits differ");
        }
        auto *tensor = static_cast<ggml_tensor*>(source.ggmlTensor);
        const size_t savedStride = tensor->nb[1];
        tensor->nb[1] += 16;
        Check(!model.PrepareDFlashDraftShortlist({0}) && model.dflashDraftTokenIds.empty(),
              "non-dense GGUF rows should fall back");
        tensor->nb[1] = savedStride;
        source.IsRepacked = true;
        Check(!model.PrepareDFlashDraftShortlist({0}) && model.dflashDraftTokenIds.empty(),
              "repacked GGUF should fall back");
        source.IsRepacked = false;
    }
    auto *tensor = static_cast<ggml_tensor*>(source.ggmlTensor);
    for (int type : {-1, 4, int(GGML_TYPE_COUNT)}) {
        // Type 4 is a removed GGML format with a zero-sized block.
        source.ggmlType = type;
        tensor->type = type == 4 ? static_cast<ggml_type>(type) : GGML_TYPE_Q4_K;
        Check(!model.PrepareDFlashDraftShortlist({0}) && model.dflashDraftTokenIds.empty(),
              "invalid GGUF type should fall back before querying its byte size");
        source.ggmlType = GGML_TYPE_Q4_K;
        tensor->type = GGML_TYPE_Q4_K;
    }
    for (const std::vector<int> &invalid : {std::vector<int>{0,0}, std::vector<int>{0,vocab}}) {
        writeIds(invalid);
        Check(!model.PrepareDFlashDraftShortlist({0}) && model.dflashDraftTokenIds.empty(),
              "invalid GGUF IDs should fall back");
    }
    std::vector<uint8_t> after(original.size());
    FastllmCudaCopyFromDeviceToHost(after.data(), source.cudaData, after.size());
    Check(original == after, "full GGUF head was modified");
    std::cout << "PASS: GGUF packed rows unchanged; FP16/BF16 1-8 rows, padding, mapping, fallback\n";
}
int main() {
    if (FastllmCudaGetDeviceCount() == 0) return 77;
    try {
        FastllmCudaSetDevice(0);
        TestGgufShortlist();
        Probe model;
        model.weight["lm_head.weight"].multiDeviceData = true;
        model.threadTpLmHeadScheme[0] = {{0,512}};
        model.threadTpLmHeadScheme[1] = {{512,1024}};
        std::vector<int> rows;
        for (int i = 0; i < 129; ++i) rows.push_back(2*i);
        model.dflashDraftTokenIds[0] = rows;
        rows.resize(256, rows.back());
        std::vector<float> values(512*128);
        for (int row=0; row<512; ++row)
            for (int k=0; k<128; ++k)
                values[row*128+k] = row == 256 ? 2.0f : 0.005f*(row%13+1);
        Data original(FLOAT32, {512,128}, values), empty;
        original.ToDevice(CUDA, std::vector<int>{0});
        ToDataType(original, FLOAT16);
        Data &head = model.dflashDraftLmHeads[0];
        Check(FastllmCudaQuantizeLinearWeightNVFP4Block16Rows(original, head, rows), "quantize rows");
        head.weightType = WeightType::LINEAR;
        head.isModelWeight = true;
        for (DataType type : {FLOAT16, BFLOAT16}) {
            Data input(FLOAT32, {1,8,128}, std::vector<float>(8*128, 0.01f)), output;
            input.ToDevice(CUDA, std::vector<int>{0});
            ToDataType(input, type);
            model.RunDFlashLmHead(0, input, original, empty, output, true);
            ForceDeviceSync();
            Check(output.dims == std::vector<int>({1,8,129}), "padding was not removed");
            ToDataType(output, FLOAT32);
            Data top;
            TopK(output, top, 16);
            top.ToDevice(CPU);
            model.MapDFlashShortlistCandidates(top);
            const float *pairs = reinterpret_cast<const float*>(top.cpuData);
            for (int row=0; row<8; ++row) {
                Check(int(pairs[row*32]) == 256, "highest candidate changed");
                std::set<int> ids;
                for (int i=0; i<16; ++i) {
                    int id = int(pairs[row*32+2*i]);
                    Check(id >= 0 && id <= 256 && id%2 == 0, "invalid global token ID");
                    ids.insert(id);
                }
                Check(ids.size() == 16, "padding created duplicate candidates");
            }
            Data full;
            model.RunDFlashLmHead(0, input, original, empty, full, false);
            Check(full.dims.back() == 512, "sampling fallback used compact head");
        }
        model.dflashDraftTokenIds[1] = {700,900};
        Data candidates(FLOAT32, {2,4}, {0,1,128,3,512,5,513,6});
        model.MapDFlashShortlistCandidates(candidates);
        const float *p = reinterpret_cast<const float*>(candidates.cpuData);
        const float expected[] = {0,1,256,3,700,5,900,6};
        for (int i=0; i<8; ++i) Check(p[i] == expected[i], "cross-shard mapping or score changed");
        std::cout << "PASS: padded rows excluded, unique top-k, global IDs, full-head fallback\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
