#include "fastllm.h"
#include "executor.h"
#include "json11.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>

using namespace fastllm;
using json11::Json;

/** 检查真实页管理器输出，不编写另一份前缀索引或淘汰实现。 */
static void Require(bool value, const char *reason) {
    if (!value) throw std::runtime_error(reason);
}

/** 在CPU页池写入可独立核对的[page,token,head,dim]浮点值。 */
static std::unique_ptr<PagedCacheManager> Pool() {
    Data desc(DataType::FLOAT32, {1, 0, 2});
    auto pool = std::unique_ptr<PagedCacheManager>(CreatePagedCacheManager(
        PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, desc, 4, 3));
    for (int i = 0; i < 24; ++i) reinterpret_cast<float*>(pool->cpuData)[i] = float(i + 1);
    return pool;
}

/** 准备实际页面引用；末页可以不满，用于验证未计算槽位不会被登记。 */
static void Fill(Data &cache, PagedCacheManager &pool, int tokens) {
    RestorePagedCachePages(cache, pool, {}, 0);
    for (int i = 0; i < (tokens + 3) / 4; ++i) cache.pageIndex.push_back(pool.GetUnusedPageIndex(true));
    cache.lastPageLen = (tokens - 1) % 4 + 1;
    cache.Resize({1, tokens, 2});
}

/** 记录实际输出与独立数学参照的逐元素误差。 */
struct ErrorMetrics {
    double maximum = 0, sum = 0;
    int count = 0, failed = 0;
};

/**
 * 回归FT原公开分页追加、恢复和Attention入口，包括共享页后的活动清理。
 *
 * 输入使用HND布局，H=2、D=8、P=4，Q/K为零，因此Attention输出是历史V的均值。
 * 原请求与恢复后的请求各自追加不同尾部；CPU均值同时检查共享完整页未被覆盖。
 * @param type 原FT分页路径支持的FP16或BF16类型。
 * @param metrics 返回32个实际输出元素的误差；页面或数值不符时抛异常。
 */
static void CheckPublicCudaPaged(DataType type, ErrorMetrics &metrics) {
    Require(HasDeviceType("cuda"), "CUDA FT backend unavailable");
    static_cast<Executor*>(GetExecutor())->SetFirstDevice("cuda:0");
    SetAttentionBackend("native_paged");
    SetAttentionBackendStrict(true);
    SetAttentionBackendTrace(true);
    struct Pools {
        ~Pools() { ClearAllPagedCacheManagers(); }
    } pools;

    Data desc(type, {2, 0, 8});
    desc.dataDevice = DataDevice::CUDA; desc.dataDeviceIds = {0};
    auto *km = AllocatePagedCacheManager(990001, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, desc, 4, 4);
    auto *vm = AllocatePagedCacheManager(990002, PagedCacheManager::PAGED_CACHE_MANAGER_TYPE_KV_CACHE, desc, 4, 4);
    Data k(type), v(type), restoredK(type), restoredV(type);
    auto append = [&](Data &key, Data &value, int count, int start) {
        std::vector<float> values(2 * count * 8), zero(values.size(), 0);
        for (int h = 0; h < 2; ++h) for (int t = 0; t < count; ++t) for (int d = 0; d < 8; ++d)
            values[(h * count + t) * 8 + d] = h * 2 + d * 0.125f + start + t;
        Data keys(type, {2, count, 8}, zero), vals(type, {2, count, 8}, values);
        keys.ToDevice(DataDevice::CUDA, {0}); vals.ToDevice(DataDevice::CUDA, {0});
        AppendPagedCache(*km, key, keys); AppendPagedCache(*vm, value, vals);
    };
    append(k, v, 5, 0);
    RecordPagedCachePrefix(k, {1,2,3,4,5}); RecordPagedCachePrefix(v, {1,2,3,4,5});
    std::vector<int> kp, vp;
    km->Query({1,2,3,4,6}, kp); vm->Query({1,2,3,4,6}, vp);
    RestorePagedCachePages(restoredK, *km, kp, 1); RestorePagedCachePages(restoredV, *vm, vp, 1);
    Require(km->pageRefCount[kp.front()] == 2, "GPU prefix was not shared");
    append(restoredK, restoredV, 1, 8);
    km->ClearPrefix(); vm->ClearPrefix();
    append(restoredK, restoredV, 1, 12);

    auto check = [&](Data &key, Data &value, double mean) {
        Data q(type, {2, 1, 8}, std::vector<float>(16, 0)), output(type), qs, ps, pi, last;
        q.ToDevice(DataDevice::CUDA, {0});
        GeneratePagedBatchParams(q, {&key}, 1, qs, ps, pi, last, {1}, true);
        AttentionPagedBatch(q, key, value, qs, ps, pi, last, output, 1, 1.0f, 1, false, true);
        output.ToDevice(DataDevice::CPU);
        ToDataTypeForceCPU(output, DataType::FLOAT32);
        Require(output.Count(0) == 16, "FT public paged Attention output shape");
        const auto *actual = reinterpret_cast<const float*>(output.cpuData);
        for (int h = 0; h < 2; ++h) for (int d = 0; d < 8; ++d) {
            double expected = h * 2 + d * 0.125 + mean;
            Require(std::isfinite(actual[h * 8 + d]), "nonfinite FT public paged output");
            double error = std::abs(actual[h * 8 + d] - expected);
            metrics.maximum = std::max(metrics.maximum, error); metrics.sum += error; ++metrics.count;
            metrics.failed += error > 0.008 + 0.008 * std::abs(expected);
        }
    };
    check(k, v, 2.0);
    check(restoredK, restoredV, 26.0 / 6);
    Require(metrics.failed == 0, "FT public paged Attention differs from CPU mean");
}

int main(int argc, char **argv) {
    const bool cuda = argc == 2 && std::string(argv[1]) == "--cuda";
    if (argc != 1 && !cuda) return 2;

    Json::array rows;
    bool passed = true;
    auto run = [&](const char *name, const std::function<void()> &test) {
        Json::object row{{"case", name}, {"backend", "FT/PagedCacheManager/CPU"},
                         {"status", "PASS"}, {"max_abs_error", 0.0}, {"mean_abs_error", 0.0}, {"failed_elements", 0}};
        try { test(); }
        catch (const std::exception &error) {
            row["status"] = "FAIL"; row["reason"] = error.what(); row["failed_elements"] = 1; passed = false;
        }
        rows.emplace_back(row);
    };
    run("paged_complete_pages_and_values", [] {
        auto pool = Pool(); Data cache(DataType::FLOAT32), next(DataType::FLOAT32);
        Fill(cache, *pool, 7);
        std::vector<int> ids{1,2,3,4,5,6,7,8};
        RecordPagedCachePrefix(cache, ids);
        std::vector<int> pages; pool->Query(ids, pages);
        Require(pages.size() == 1, "partial/uncalculated page entered prefix index");
        int physical = cache.pageIndex.front();
        RestorePagedCachePages(next, *pool, pages, 1);
        Require(next.pageIndex.front() == physical && pool->pageRefCount[physical] == 2, "prefix copied instead of referenced");
        for (int i = 0; i < 8; ++i)
            Require(reinterpret_cast<float*>(next.pagedKVCacheData->cpuData)[physical * 8 + i] == float(physical * 8 + i + 1), "restored KV value mismatch");
        ReleasePagedCacheReferences(cache, true); ReleasePagedCacheReferences(next, true);
        Require(pool->FreePageCount() == 3, "page references leaked");
    });
    run("paged_position_isolation", [] {
        auto pool = Pool(); Data cache(DataType::FLOAT32);
        Fill(cache, *pool, 4); std::vector<int> ids{1,2,3,4,5};
        RecordPagedCachePrefix(cache, ids, 23);
        Require(QueryPagedCachePrefix({&cache}, ids, 0) == 0, "different RoPE origin hit");
        Require(QueryPagedCachePrefix({&cache}, ids, 23) == 4, "matching RoPE origin missed");
        Require(QueryPagedCachePrefix({&cache}, {1,2,3,4}, 23) == 0, "full hit did not leave output work");
        ReleasePagedCacheReferences(cache, true);
    });
    run("paged_multi_manager_common_prefix", [] {
        auto a = Pool(), b = Pool(); Data ca(DataType::FLOAT32), cb(DataType::FLOAT32);
        Fill(ca, *a, 8); Fill(cb, *b, 4); std::vector<int> ids{1,2,3,4,5,6,7,8,9};
        RecordPagedCachePrefix(ca, ids); RecordPagedCachePrefix(cb, ids);
        Require(QueryPagedCachePrefix({&ca, &cb}, ids) == 4, "did not choose common complete prefix");
        b->ClearPrefix();
        Require(QueryPagedCachePrefix({&ca, &cb}, ids) == 0, "one missing manager was ignored");
        ReleasePagedCacheReferences(ca, true); ReleasePagedCacheReferences(cb, true);
    });
    run("paged_clear_active_and_recycle", [] {
        auto pool = Pool(); Data cache(DataType::FLOAT32), shared(DataType::FLOAT32);
        Fill(cache, *pool, 4); RecordPagedCachePrefix(cache, {1,2,3,4});
        std::vector<int> pages; pool->Query({1,2,3,4,5}, pages);
        RestorePagedCachePages(shared, *pool, pages, 1);
        int page = pages.front(); float value = reinterpret_cast<float*>(pool->cpuData)[page * 8];
        pool->ClearPrefix();
        Require(pool->pageRefCount[page] == 2 && pool->PrefixDirectory().empty(), "clear destroyed active references");
        int other = pool->GetUnusedPageIndex(true);
        Require(other != page, "active page reused");
        reinterpret_cast<float*>(pool->cpuData)[other * 8] = -100;
        Require(reinterpret_cast<float*>(pool->cpuData)[page * 8] == value, "active KV overwritten");
        pool->ReleasePageIndex(other);
        ReleasePagedCacheReferences(cache, true); ReleasePagedCacheReferences(shared, true);
        Require(pool->FreePageCount() == 3, "clear/release leaked pages");
    });
    run("paged_pool_eviction", [] {
        auto pool = Pool(); Data cache(DataType::FLOAT32);
        for (int token = 1; token <= 6; ++token) {
            Fill(cache, *pool, 4); RecordPagedCachePrefix(cache, std::vector<int>(4, token));
            ReleasePagedCacheReferences(cache, true);
        }
        Require(pool->evictedPages >= 3, "full page pool did not evict");
        std::vector<int> pages; pool->Query(std::vector<int>(5, 6), pages);
        Require(pages.size() == 1 && pool->FreePageCount() == 3, "latest prefix or free-page accounting incorrect");
    });
    if (cuda) for (auto type : {DataType::FLOAT16, DataType::BFLOAT16}) {
        ErrorMetrics metrics;
        run(type == DataType::FLOAT16 ? "ft_public_paged_fp16" : "ft_public_paged_bf16",
            [&] { CheckPublicCudaPaged(type, metrics); });
        auto row = rows.back().object_items();
        row["backend"] = "FT/public/strict-native_paged";
        row["input_shape"] = "H=2,D=8,P=4; original=5 tokens, restored=6 tokens";
        row["max_abs_error"] = metrics.maximum;
        row["mean_abs_error"] = metrics.count ? metrics.sum / metrics.count : 0.0;
        row["failed_elements"] = metrics.failed;
        row["compared_elements"] = metrics.count;
        rows.back() = row;
    }
    std::cout << Json(Json::object{{"status", passed ? "PASS" : "FAIL"}, {"results", rows}}).dump() << '\n';
    return passed ? 0 : 1;
}
