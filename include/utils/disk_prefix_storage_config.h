#ifndef FASTLLM_DISK_PREFIX_STORAGE_CONFIG_H
#define FASTLLM_DISK_PREFIX_STORAGE_CONFIG_H

#include <cstdint>
#include <string>

namespace fastllm {

// Storage capability bitmask reported to Python by
// get_persistent_prefix_cache_storage_capabilities().
//
enum PersistentPrefixStorageCapability : uint32_t {
    STORAGE_READ_V4  = 1u << 2,
    STORAGE_WRITE_V4 = 1u << 3,
    STORAGE_CONFIG   = 1u << 4,
};

// v4 is the only supported format and is written under <prefix_cache_dir>/v4.
constexpr int kStorageFormatV4 = 4;
// v4 manifest schema version. This is a third, independent version axis:
// storage format != manifest schema != model state ABI.
constexpr int kStorageManifestSchemaV4 = 2;
// Model state semantics. Must not change merely because the storage format
// changed; a storage upgrade does not change the model's math identity.
constexpr int kStateAbiVersion = 4;

struct PersistentPrefixStorageConfig {
    int writerFormat = kStorageFormatV4;
};

// Strict JSON parse of {"writer_format":4}. Never touches the filesystem.
bool ParsePersistentPrefixStorageConfig(const std::string &text,
                                        PersistentPrefixStorageConfig &out,
                                        std::string &error);

// Process-wide storage configuration, set once by Python before the model
// opens the cache. Storage config is runtime-only: it must never enter
// identity_keys(), family_identity or the model math hash.
// Fails closed for any writer format other than v4.
bool SetPersistentPrefixStorageConfig(const PersistentPrefixStorageConfig &config,
                                      std::string &error);
bool PersistentPrefixStorageConfigSet();
PersistentPrefixStorageConfig GetPersistentPrefixStorageConfig();

// Capability bits actually implemented by this build. This must report the
// truth: advertising WRITE_V4 while no v4 writer exists would make Python
// silently believe a v4 request succeeded.
uint32_t PersistentPrefixStorageCapabilities();

struct PersistentPrefixStoragePlan {
    int writerFormat = kStorageFormatV4;
    bool v4Writer = true;
    std::string writerRoot;   // <prefix_cache_dir>/v4
};
bool PlanPersistentPrefixStorage(const std::string &prefixCacheDir,
                                 PersistentPrefixStoragePlan &out,
                                 std::string &error);

} // namespace fastllm
#endif
