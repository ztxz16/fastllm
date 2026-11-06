#include "devices/cuda/attention/fastllm-attention-backend.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <mutex>
#include <set>
#include <tuple>
#include <vector>

namespace fastllm {
    namespace {
        struct CachedBackend {
            CudaAttentionBackendRegistration registration;
            std::string supportReason;
        };

        struct SelectionSignature {
            int deviceId;
            CudaAttentionCacheLayout cacheLayout;
            CudaAttentionPhase phase;
            DataType dataType;
            int headDim;
            int group;
            int attentionType;
            int windowLeft;

            bool operator<(const SelectionSignature &other) const {
                return std::tie(deviceId, cacheLayout, phase, dataType, headDim,
                                group, attentionType, windowLeft) <
                       std::tie(other.deviceId, other.cacheLayout, other.phase,
                                other.dataType, other.headDim, other.group,
                                other.attentionType, other.windowLeft);
            }
        };

        struct RuntimeAttentionConfig {
            std::string requested;
            bool strict;
        };

        std::mutex registryMutex;
        std::map<std::string, CudaAttentionBackendRegistration> registry;
        std::map<SelectionSignature, CachedBackend> selectionCache;
        std::set<SelectionSignature> tracedSelectionKeys;
        std::once_flag builtinRegistrationFlag;

        /** 返回用于Trace的KV Cache布局名称。 */
        std::string CacheLayoutName(CudaAttentionCacheLayout layout) {
            return layout == CudaAttentionCacheLayout::PAGED ? "paged" : "contiguous";
        }

        /** 返回用于Trace的Attention阶段名称。 */
        std::string PhaseName(CudaAttentionPhase phase) {
            return phase == CudaAttentionPhase::PREFILL ? "prefill" : "decode";
        }

        /** 构造不包含动态Batch大小的静态能力选择缓存键。 */
        SelectionSignature MakeSelectionSignature(
                const CudaAttentionInvocation &invocation) {
            return {
                invocation.deviceId, invocation.cacheLayout, invocation.phase,
                invocation.dataType, invocation.headDim, invocation.group,
                invocation.attentionType, invocation.windowLeft,
            };
        }

        /** 首次执行时冻结进程配置，避免正式推理逐层读取配置锁。 */
        const RuntimeAttentionConfig &GetRuntimeAttentionConfig() {
            static const RuntimeAttentionConfig config = {
                GetAttentionBackend(), GetAttentionBackendStrict(),
            };
            return config;
        }

        /** 输出一次可由测试解析的Attention Backend选择记录。 */
        void TraceSelection(const CudaAttentionInvocation &invocation,
                            const std::string &requested,
                            const std::string &actual,
                            const std::string &reason) {
            if (!GetAttentionBackendTrace()) {
                return;
            }
            printf("[fastllm][attention] requested=%s actual=%s cache=%s phase=%s "
                   "dtype=%s batch=%d head_dim=%d group=%d reason=%s\n",
                   requested.c_str(), actual.c_str(),
                   CacheLayoutName(invocation.cacheLayout).c_str(),
                   PhaseName(invocation.phase).c_str(),
                   GetDataTypeName(invocation.dataType).c_str(), invocation.batch,
                   invocation.headDim, invocation.group, reason.c_str());
            fflush(stdout);
        }
    }

    bool RegisterCudaAttentionBackend(
        const CudaAttentionBackendRegistration &registration) {
        if (registration.name.empty() || registration.supports == nullptr ||
            registration.run == nullptr) {
            return false;
        }

        std::lock_guard<std::mutex> guard(registryMutex);
        if (registry.find(registration.name) != registry.end()) {
            return false;
        }
        registry.emplace(registration.name, registration);
        selectionCache.clear();
        tracedSelectionKeys.clear();
        return true;
    }

    /** 注册当前CUDA构建内置的全部Attention Backend。 */
    void RegisterBuiltinCudaAttentionBackends() {
        RegisterCudaContiguousAttentionBackends();
        RegisterCudaPagedAttentionBackends();
    }

    bool RunCudaAttentionBackend(CudaAttentionInvocation &invocation,
                                 std::string &actualBackend,
                                 std::string &lastError) {
        std::call_once(builtinRegistrationFlag, RegisterBuiltinCudaAttentionBackends);
        const RuntimeAttentionConfig &config = GetRuntimeAttentionConfig();
        const std::string &requested = config.requested;
        const bool strict = config.strict;
        const SelectionSignature key = MakeSelectionSignature(invocation);
        CudaAttentionBackendRegistration selected;
        std::string selectionReason;
        static thread_local std::map<SelectionSignature, CachedBackend>
            threadSelectionCache;
        bool firstThreadSelection = false;

        auto local = threadSelectionCache.find(key);
        if (local != threadSelectionCache.end()) {
            selected = local->second.registration;
            selectionReason = local->second.supportReason;
        }

        if (selected.name.empty()) {
            firstThreadSelection = true;
            std::lock_guard<std::mutex> guard(registryMutex);
            auto cached = selectionCache.find(key);
            if (cached != selectionCache.end()) {
                selected = cached->second.registration;
                selectionReason = cached->second.supportReason;
            } else {
                std::string explicitRejection;
                if (requested != "auto") {
                    auto candidate = registry.find(requested);
                    if (candidate == registry.end()) {
                        lastError = "unknown backend '" + requested + "'";
                        TraceSelection(invocation, requested, "none", lastError);
                        return false;
                    }
                    CudaAttentionSupportResult support = candidate->second.supports(invocation);
                    if (support.supported) {
                        selected = candidate->second;
                        selectionReason = "explicit";
                    } else {
                        explicitRejection = "backend '" + requested +
                            "' is unsupported: " + support.reason;
                        if (strict) {
                            lastError = explicitRejection;
                            TraceSelection(invocation, requested, "none", lastError);
                            return false;
                        }
                    }
                }

                std::vector<CudaAttentionBackendRegistration> candidates;
                if (selected.name.empty()) {
                    for (const auto &item : registry) {
                        candidates.push_back(item.second);
                    }
                    std::sort(candidates.begin(), candidates.end(),
                              [](const auto &left, const auto &right) {
                                  if (left.priority != right.priority) {
                                      return left.priority > right.priority;
                                  }
                                  return left.name < right.name;
                              });
                    std::string rejected = explicitRejection;
                    for (const auto &candidate : candidates) {
                        CudaAttentionSupportResult support = candidate.supports(invocation);
                        if (support.supported) {
                            selected = candidate;
                            selectionReason = explicitRejection.empty()
                                ? "auto" : "fallback";
                            break;
                        }
                        if (!rejected.empty()) {
                            rejected += "; ";
                        }
                        rejected += candidate.name + ": " + support.reason;
                    }
                    if (selected.name.empty()) {
                        lastError = "no compatible backend: " + rejected;
                        TraceSelection(invocation, requested, "none", lastError);
                        return false;
                    }
                }
                selectionCache.emplace(key, CachedBackend{selected, selectionReason});
            }
            threadSelectionCache.emplace(
                key, CachedBackend{selected, selectionReason});
        }

        actualBackend = selected.name;
        const bool ok = selected.run(invocation);
        if (!ok) {
            lastError = "backend '" + selected.name + "' execution failed";
            TraceSelection(invocation, requested, selected.name, "run_failed");
            return false;
        }
        bool shouldTrace = false;
        if (firstThreadSelection) {
            std::lock_guard<std::mutex> guard(registryMutex);
            shouldTrace = tracedSelectionKeys.insert(key).second;
        }
        if (shouldTrace) {
            TraceSelection(invocation, requested, selected.name, selectionReason);
        }
        return true;
    }
}
