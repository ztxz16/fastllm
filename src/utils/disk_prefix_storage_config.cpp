#include "utils/disk_prefix_storage_config.h"

#include "json11.hpp"

#include <filesystem>
#include <mutex>

namespace fastllm {
namespace {
std::mutex g_configMutex;
PersistentPrefixStorageConfig g_config;
bool g_configSet = false;

bool Fail(std::string &error, const char *message) {
    error = message;
    return false;
}
} // namespace

bool ParsePersistentPrefixStorageConfig(const std::string &text,
                                        PersistentPrefixStorageConfig &out,
                                        std::string &error) {
    std::string parseError;
    const auto parsed = json11::Json::parse(text, parseError);
    if (!parseError.empty() || !parsed.is_object())
        return Fail(error, "storage_config_invalid_json");
    bool sawFormat = false;
    int writerFormat = 0;
    for (const auto &entry : parsed.object_items()) {
        if (entry.first != "writer_format")
            return Fail(error, "storage_config_unknown_field");
        if (!entry.second.is_number())
            return Fail(error, "storage_config_writer_format_type");
        const double value = entry.second.number_value();
        if (value != double(int(value)))
            return Fail(error, "storage_config_writer_format_range");
        writerFormat = int(value);
        sawFormat = true;
    }
    if (!sawFormat) return Fail(error, "storage_config_missing_writer_format");
    if (writerFormat != kStorageFormatV4)
        return Fail(error, "storage_config_only_v4_supported");
    out.writerFormat = kStorageFormatV4;
    return true;
}

bool SetPersistentPrefixStorageConfig(const PersistentPrefixStorageConfig &config,
                                      std::string &error) {
    if (config.writerFormat != kStorageFormatV4)
        return Fail(error, "storage_config_only_v4_supported");
    if ((PersistentPrefixStorageCapabilities() & STORAGE_WRITE_V4) == 0)
        return Fail(error, "storage_config_writer_format_unavailable");
    std::lock_guard<std::mutex> lock(g_configMutex);
    g_config = config;
    g_configSet = true;
    return true;
}

PersistentPrefixStorageConfig GetPersistentPrefixStorageConfig() {
    std::lock_guard<std::mutex> lock(g_configMutex);
    return g_config;
}

bool PersistentPrefixStorageConfigSet() {
    std::lock_guard<std::mutex> lock(g_configMutex);
    return g_configSet;
}

uint32_t PersistentPrefixStorageCapabilities() {
    return STORAGE_READ_V4 | STORAGE_WRITE_V4 | STORAGE_CONFIG;
}

bool PlanPersistentPrefixStorage(const std::string &prefixCacheDir,
                                 PersistentPrefixStoragePlan &out,
                                 std::string &error) {
    error.clear();
    if (prefixCacheDir.empty()) return Fail(error, "storage_plan_empty_prefix_cache_dir");
    const std::filesystem::path directory(prefixCacheDir);
    if (!directory.is_absolute()) return Fail(error, "storage_plan_prefix_cache_dir_not_absolute");
    PersistentPrefixStoragePlan plan;
    plan.writerFormat = kStorageFormatV4;
    plan.v4Writer = true;
    if ((PersistentPrefixStorageCapabilities() & STORAGE_WRITE_V4) == 0)
        return Fail(error, "storage_config_writer_format_unavailable");
    plan.writerRoot = (directory.lexically_normal() / "v4").string();
    out = std::move(plan);
    return true;
}

} // namespace fastllm
