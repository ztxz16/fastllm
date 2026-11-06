#include "distributed/qwen3_shard_loader.h"
#include "json11.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace fastllm::distributed;
using Json = json11::Json;
namespace fs = std::filesystem;

namespace {
/** 读取无符号十进制参数，拒绝负数、尾随字符和溢出。 */
uint64_t Number(const std::string &text) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) {
        throw std::runtime_error("invalid nonnegative integer: " + text);
    }
    return std::stoull(text);
}

/** 转换有界整数用于 json11 输出，防止有符号截断。 */
Json Count(uint64_t n) {
    if (n > 9007199254740991ULL) {
        throw std::runtime_error("report integer exceeds exact JSON range");
    }
    return Json(static_cast<long long>(n));
}
} // namespace

/**
 * 通过正式读取器导出单个 rank 的计划和可选原始权重，供独立参考实现验证。
 * @param argc 命令行参数数量。
 * @param argv --model、--rank、--world-size 必填；--output 触发实际读取并保存逐张量二进制。
 *             无 --output 时仅检查 metadata；--max-bytes 限制每个目标张量的内存分配；
 *             --change-cwd 在构造读取器后切换目录，用于验证相对模型路径的生命周期。
 * @return 成功为 0，任一参数/读取/写出错误为 1；stdout 始终输出含错误原因的 JSON。
 */
int main(int argc, char **argv) {
    Json::object report{{"status", "FAIL"}, {"backend", "qwen3_rank_local_safetensors"}};
    Json::array tensors;
    try {
        std::string directory, output, changeCwd;
        int rank = -1, world = 0;
        uint64_t maxBytes = 512ULL * 1024 * 1024;
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (++i == argc) {
                throw std::runtime_error("missing value for " + key);
            }
            std::string value = argv[i];
            if (key == "--model") {
                directory = value;
            } else if (key == "--output") {
                output = value;
            } else if (key == "--change-cwd") {
                changeCwd = value;
            } else if (key == "--rank" || key == "--world-size") {
                uint64_t n = Number(value);
                if (n > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
                    throw std::runtime_error("rank/world size overflow");
                }
                (key == "--rank" ? rank : world) = static_cast<int>(n);
            } else if (key == "--max-bytes") {
                maxBytes = Number(value);
            } else {
                throw std::runtime_error("unknown argument: " + key);
            }
        }
        if (directory.empty() || rank < 0 || world <= 0) {
            throw std::runtime_error("required: --model DIR --rank N --world-size N [--output NEW_DIR]");
        }
        if (!output.empty()) {
            output = fs::absolute(output).string();
        }
        Qwen3ShardLoader loader(directory, rank, world);
        if (!changeCwd.empty()) {
            fs::current_path(changeCwd);
        }
        const auto &plan = loader.Plan();
        report["rank"] = rank;
        report["world_size"] = world;
        report["query_head_begin"] = Count(plan.queryHeadBegin);
        report["query_head_count"] = Count(plan.queryHeadCount);
        report["kv_head_begin"] = Count(plan.kvHeadBegin);
        report["kv_head_count"] = Count(plan.kvHeadCount);
        report["intermediate_begin"] = Count(plan.intermediateBegin);
        report["intermediate_count"] = Count(plan.intermediateCount);
        report["vocab_begin"] = Count(plan.vocabBegin);
        report["vocab_count"] = Count(plan.vocabCount);
        report["tensor_bytes"] = Count(plan.tensorBytes);
        report["plan_only"] = output.empty();
        if (!output.empty() && !fs::create_directory(output)) {
            throw std::runtime_error("output directory must not exist: " + output);
        }

        uint64_t totalRead = 0, maxBuffer = 0;
        for (const auto &tensor : plan.tensors) {
            Json::array shape, slices;
            for (uint64_t dim : tensor.shape) {
                shape.push_back(Count(dim));
            }
            for (const auto &slice : tensor.slices) {
                slices.push_back(Json::object{{"source", slice.source}, {"row_begin", Count(slice.rowBegin)},
                    {"row_end", Count(slice.rowEnd)}, {"column_begin", Count(slice.columnBegin)},
                    {"column_end", Count(slice.columnEnd)}});
            }
            Json::object entry{{"name", tensor.name}, {"dtype", tensor.dtype}, {"shape", shape},
                {"bytes", Count(tensor.bytes)}, {"zero_fill", tensor.zeroFill}, {"slices", slices}};
            if (!output.empty()) {
                LoadedShardTensor loaded = loader.LoadTensor(tensor.name, maxBytes);
                std::string filename = std::to_string(tensors.size()) + ".bin";
                std::ofstream file(fs::path(output) / filename, std::ios::binary);
                file.write(reinterpret_cast<const char*>(loaded.data.data()), loaded.data.size());
                file.close();
                if (!file) {
                    throw std::runtime_error("failed to write tensor: " + tensor.name);
                }
                entry["file"] = filename;
                entry["bytes_read"] = Count(loaded.bytesRead);
                totalRead += loaded.bytesRead;
                maxBuffer = std::max<uint64_t>(maxBuffer, loaded.data.size());
            }
            tensors.push_back(entry);
        }
        report["payload_bytes_read"] = Count(totalRead);
        report["max_tensor_buffer_bytes"] = Count(maxBuffer);
        report["status"] = "PASS";
    } catch (const std::exception &error) {
        report["reason"] = error.what();
    }
    report["tensors"] = tensors;
    std::cout << Json(report).dump() << std::endl;
    return report["status"].string_value() == "PASS" ? 0 : 1;
}
