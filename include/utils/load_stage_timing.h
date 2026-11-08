#pragma once

// 加载阶段的默认关闭分段计时（用于定位耗时大头，不改变任何行为）。
//
// 开关：FASTLLM_LOAD_TIMING=1 才启用；未设置 / "0" → 全部为无副作用的空操作。
// 计时只在加载结束时打印一行摘要，便于与既有基线对比。

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace fastllm {

struct LoadStageTiming {
    // 各阶段内部的"数据面"耗时（累加、跨线程原子）
    std::atomic<uint64_t> bufferTotalUs{0};   // 5 个 CreateBuffer* 方法的总耗时
    std::atomic<uint64_t> fileIoUs{0};        // 其中 fopen/fseek/fread/fclose 的耗时
    std::atomic<uint64_t> bufferConvertUs{0}; // 其中数值格式转换（ConvertDataType）的耗时
    std::atomic<uint64_t> bufferPackUs{0};    // 其中 NVFP4 逐元素打包循环的耗时
    std::atomic<uint64_t> uploadUs{0};        // 权重上传到显卡（PrepareThreadTpWeights）总耗时
    std::atomic<uint64_t> uploadCpuUs{0};     // 其中 CPU 侧切分/复制（CopyFrom）的耗时
    std::atomic<uint64_t> uploadH2dUs{0};     // 其中真正往显卡搬（ToDevice CUDA）的耗时
    std::atomic<uint64_t> uploadSplitUs{0};   // 其中 TP 切分（SplitMultiCudaWeight）的耗时
    std::atomic<uint64_t> uploadRepackUs{0};  // 其中 NVFP4 专家重排（PrepareNVFP4E4M3Moe）的耗时
    std::atomic<uint64_t> bufferAllocUs{0};   // CreateBuffer* 里"申请/释放内存"的耗时
    std::atomic<uint64_t> splitCalls{0};      // SplitMultiCudaWeight 被调用次数
    std::atomic<uint64_t> splitSegments{0};   // 实际提交的权重/内嵌尺度拷贝调用数，不含 bias
    std::atomic<uint64_t> splitBytes{0};      // 上述调用的请求字节数（2D 为 width * height）
    std::atomic<uint64_t> splitAllocUs{0};    // 其中"给每个 rank 分配显存"的耗时
    std::atomic<uint64_t> splitPeerUs{0};     // 其中"跨卡互访使能"的耗时
    std::atomic<uint64_t> splitCpuSourceCalls{0}, splitCudaSourceCalls{0};
    // compact E4M3 分支按 rank 累计：行/列切分次数与拷贝循环耗时。
    std::atomic<uint64_t> splitCompactRowCalls{0}, splitCompactRowCopyUs{0};
    std::atomic<uint64_t> splitCompactColCalls{0}, splitCompactColCopyUs{0};
    std::atomic<uint64_t> bufferCalls{0};
    // 五个 CreateBuffer* 方法各自被调用次数与耗时（用于定位 91 秒到底在哪个方法里）
    std::atomic<uint64_t> callsCreateBuffer{0},   usCreateBuffer{0};
    std::atomic<uint64_t> callsWithScale{0},      usWithScale{0};
    std::atomic<uint64_t> callsAwq{0},            usAwq{0};
    std::atomic<uint64_t> callsInt4Group{0},      usInt4Group{0};
    std::atomic<uint64_t> callsAffineInt4{0},     usAffineInt4{0};
    std::atomic<uint64_t> uploadCalls{0};
    // CreateBufferWithScale 的 compact E4M3 分支整段（fopen+seek+fread+fclose）；用于把 WS 拆成读盘与其它
    std::atomic<uint64_t> compactReadUs{0};
    // NVFP4 专家重排（PrepareNVFP4E4M3Moe）内部细分：判定瓶颈在"主机中转"、"CPU 启动开销"还是"GPU 执行"
    std::atomic<uint64_t> repackBuilds{0};       // 重排被调用次数（校验用：期望 48 层 × 5 rank = 240）
    std::atomic<uint64_t> repackExperts{0};      // 累计处理的专家组数（期望 240 × 512 = 122880）
    std::atomic<uint64_t> repackStageBuilds{0};  // 其中走"显存紧张→主机中转"分支的次数
    std::atomic<uint64_t> repackStageBytes{0};   // 主机中转搬动的字节数
    std::atomic<uint64_t> repackStageInUs{0};    // 主机中转：逐个专家 D2H 落地耗时
    std::atomic<uint64_t> repackStageH2dUs{0};   // 主机中转：逐个专家 H2D 回显卡耗时（含在 loop_cpu 内）
    std::atomic<uint64_t> repackLoopCpuUs{0};    // 专家循环 CPU 侧总耗时（驱动调用+内核启动+中转拷贝）
    std::atomic<uint64_t> repackSyncUs{0};       // 专家循环后的 cudaStreamSynchronize 等待（GPU 实际执行）
    std::atomic<uint64_t> repackPrologueUs{0};   // 前置校验+cudaMemGetInfo+尺寸计算
    std::atomic<uint64_t> repackAllocUs{0};      // 每层 cache 的 10 次大块显存申请 + 转置 scratch 申请
    std::atomic<uint64_t> repackReleaseUs{0};    // scratch 释放 + 1024 个专家源权重释放
    // 主加载循环的墙钟拆分（用于定位 load 阶段里没被 buffer/upload 覆盖的那部分时间）
    std::atomic<uint64_t> workerWallUs{0};       // 各加载线程"处理自己那批张量"的墙钟之和
    std::atomic<uint64_t> serialGroupsUs{0};     // 按优先级串行分组的墙钟（组内可能是多线程）
    std::atomic<uint64_t> parallelPartsUs{0};    // 最后一批并行分片的墙钟
    std::atomic<uint64_t> budgetWaitUs{0};       // StreamingLoadBudget::Acquire 的等待耗时
    // HF 加载线程的累计耗时，不是启动墙钟；mergeProcess 包含规则扫描、实际合并及嵌套锁等待。
    std::atomic<uint64_t> dtypeParseUs{0}, mergeProcessUs{0}, lockerWaitUs{0};
    // 显存分配器埋点：定位"每专家一次分配/释放"的真实开销落点（项目簿记 vs 裸 cudaMalloc/cudaFree）
    std::atomic<uint64_t> allocForDataCalls{0}, allocForDataUs{0};
    std::atomic<uint64_t> directMallocCalls{0}, directMallocUs{0}, directMallocInnerUs{0};
    std::atomic<uint64_t> directFreeCalls{0}, directFreeUs{0}, directFreeInnerUs{0}, directFreeWsUs{0};

    // 顶层阶段时间戳（从 CreateLLMModelFromHF 进入算起，毫秒）
    std::atomic<uint64_t> tHeadersMs{0};      // 206 个文件头解析完毕 + SafeTensors 构造完成
    std::atomic<uint64_t> tSortedNamesMs{0};  // GetSortedItemNames 完成
    std::atomic<uint64_t> tTensorMapMs{0};    // GetTensorMap 完成
    std::atomic<uint64_t> tFilterMs{0};       // tensorMap 过滤完成
    std::atomic<uint64_t> tPrepareMs{0};      // 类型解析（weights_prepare 循环）完成
    std::atomic<uint64_t> tLoadMs{0};         // 主加载循环（线程 join）完成
    std::atomic<uint64_t> tStartMs{0};        // 函数入口
};

// 单一全局实例：定义在 src/model.cpp。放在头文件里用 inline+static 会导致
// "同一份代码被编进两个构建目标时各有一份实例"（本项目 fastllm 与 fastllm_tools
// 都编了 fastllm-multicuda.cu），计数会写进另一份实例，读数变 0。
LoadStageTiming &GetLoadStageTiming();

inline bool LoadStageTimingEnabled() {
    static const bool enabled = [] {
        const char *env = std::getenv("FASTLLM_LOAD_TIMING");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return enabled;
}

inline uint64_t LoadNowMs() {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count());
}

inline void MarkLoadStage(std::atomic<uint64_t> &slot) {
    if (!LoadStageTimingEnabled()) return;
    slot.store(LoadNowMs(), std::memory_order_relaxed);
}

// 把一个作用域的耗时累加到指定计数器（仅在启用时计时）。
class ScopedLoadTimer {
public:
    explicit ScopedLoadTimer(std::atomic<uint64_t> &sinkUs, bool selected = true)
        : sink_(sinkUs), active_(selected && LoadStageTimingEnabled()) {
        if (active_) {
            start_ = std::chrono::steady_clock::now();
        }
    }
    ~ScopedLoadTimer() { Stop(); }
    void Stop() {
        if (!active_) return;
        const auto end = std::chrono::steady_clock::now();
        sink_.fetch_add(
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count()),
            std::memory_order_relaxed);
        active_ = false;
    }
    ScopedLoadTimer(const ScopedLoadTimer &) = delete;
    ScopedLoadTimer &operator=(const ScopedLoadTimer &) = delete;

private:
    std::atomic<uint64_t> &sink_;
    bool active_;
    std::chrono::steady_clock::time_point start_{};
};

inline void PrintLoadStageTiming() {
    auto &t = GetLoadStageTiming();
    const uint64_t start = t.tStartMs.load(std::memory_order_relaxed);
    auto rel = [&](std::atomic<uint64_t> &slot) -> long long {
        const uint64_t v = slot.load(std::memory_order_relaxed);
        return v == 0 ? -1 : static_cast<long long>(v - start);
    };
    std::printf(
        "[Fastllm] load stage timing (enabled=%d): headers=%lldms sorted=%lldms tensorMap=%lldms "
        "filter=%lldms prepare=%lldms load=%lldms | "
        "buffer_calls=%llu buffer_total=%llu us file_io=%llu us convert=%llu us "
        "pack=%llu us alloc_other=%lld us | "
        "upload_calls=%llu upload=%llu us upload_cpu_split=%llu us upload_h2d=%llu us "
        "upload_split=%llu us upload_repack=%llu us buffer_alloc=%llu us | "
        "split_calls=%llu split_segments=%llu split_bytes=%llu avg_seg_bytes=%llu "
        "split_alloc=%llu us split_peer=%llu us "
        "split_copy_other=%lld us | "
        "CB=%llu/%llu us WS=%llu/%llu us AWQ=%llu/%llu us I4=%llu/%llu us AI4=%llu/%llu us | "
        "compact_read=%llu us | "
        "repack: builds=%llu experts=%llu stage_builds=%llu stage_MiB=%llu "
        "stage_in=%llu us stage_h2d=%llu us loop_cpu=%llu us sync=%llu us "
        "prologue=%llu us alloc=%llu us release=%llu us other=%lld us | "
        "load_other=%lld us | "
        "load3: worker_sum=%llu us serial_groups=%llu us parallel_parts=%llu us "
        "budget_wait=%llu us | "
        "alloc: for_data=%llu/%llu us d_malloc=%llu/%llu us inner=%llu us "
        "d_free=%llu/%llu us inner=%llu us ws=%llu us.\n",
        LoadStageTimingEnabled() ? 1 : 0,
        rel(t.tHeadersMs), rel(t.tSortedNamesMs), rel(t.tTensorMapMs), rel(t.tFilterMs),
        rel(t.tPrepareMs), rel(t.tLoadMs),
        (unsigned long long)t.bufferCalls.load(std::memory_order_relaxed),
        (unsigned long long)t.bufferTotalUs.load(std::memory_order_relaxed),
        (unsigned long long)t.fileIoUs.load(std::memory_order_relaxed),
        (unsigned long long)t.bufferConvertUs.load(std::memory_order_relaxed),
        (unsigned long long)t.bufferPackUs.load(std::memory_order_relaxed),
        (long long)(t.bufferTotalUs.load(std::memory_order_relaxed) -
                    t.fileIoUs.load(std::memory_order_relaxed) -
                    t.bufferConvertUs.load(std::memory_order_relaxed) -
                    t.bufferPackUs.load(std::memory_order_relaxed)),
        (unsigned long long)t.uploadCalls.load(std::memory_order_relaxed),
        (unsigned long long)t.uploadUs.load(std::memory_order_relaxed),
        (unsigned long long)t.uploadCpuUs.load(std::memory_order_relaxed),
        (unsigned long long)t.uploadH2dUs.load(std::memory_order_relaxed),
        (unsigned long long)t.uploadSplitUs.load(std::memory_order_relaxed),
        (unsigned long long)t.uploadRepackUs.load(std::memory_order_relaxed),
        (unsigned long long)t.bufferAllocUs.load(std::memory_order_relaxed),
        (unsigned long long)t.splitCalls.load(std::memory_order_relaxed),
        (unsigned long long)t.splitSegments.load(std::memory_order_relaxed),
        (unsigned long long)t.splitBytes.load(std::memory_order_relaxed),
        (unsigned long long)(t.splitSegments.load(std::memory_order_relaxed) == 0 ? 0 :
            t.splitBytes.load(std::memory_order_relaxed) / t.splitSegments.load(std::memory_order_relaxed)),
        (unsigned long long)t.splitAllocUs.load(std::memory_order_relaxed),
        (unsigned long long)t.splitPeerUs.load(std::memory_order_relaxed),
        (long long)(t.uploadSplitUs.load(std::memory_order_relaxed) -
                    t.splitAllocUs.load(std::memory_order_relaxed) -
                    t.splitPeerUs.load(std::memory_order_relaxed)),
        (unsigned long long)t.callsCreateBuffer.load(std::memory_order_relaxed),
        (unsigned long long)t.usCreateBuffer.load(std::memory_order_relaxed),
        (unsigned long long)t.callsWithScale.load(std::memory_order_relaxed),
        (unsigned long long)t.usWithScale.load(std::memory_order_relaxed),
        (unsigned long long)t.callsAwq.load(std::memory_order_relaxed),
        (unsigned long long)t.usAwq.load(std::memory_order_relaxed),
        (unsigned long long)t.callsInt4Group.load(std::memory_order_relaxed),
        (unsigned long long)t.usInt4Group.load(std::memory_order_relaxed),
        (unsigned long long)t.callsAffineInt4.load(std::memory_order_relaxed),
        (unsigned long long)t.usAffineInt4.load(std::memory_order_relaxed),
        (unsigned long long)t.compactReadUs.load(std::memory_order_relaxed),
        (unsigned long long)t.repackBuilds.load(std::memory_order_relaxed),
        (unsigned long long)t.repackExperts.load(std::memory_order_relaxed),
        (unsigned long long)t.repackStageBuilds.load(std::memory_order_relaxed),
        (unsigned long long)(t.repackStageBytes.load(std::memory_order_relaxed) >> 20),
        (unsigned long long)t.repackStageInUs.load(std::memory_order_relaxed),
        (unsigned long long)t.repackStageH2dUs.load(std::memory_order_relaxed),
        (unsigned long long)t.repackLoopCpuUs.load(std::memory_order_relaxed),
        (unsigned long long)t.repackSyncUs.load(std::memory_order_relaxed),
        (unsigned long long)t.repackPrologueUs.load(std::memory_order_relaxed),
        (unsigned long long)t.repackAllocUs.load(std::memory_order_relaxed),
        (unsigned long long)t.repackReleaseUs.load(std::memory_order_relaxed),
        (long long)(t.uploadRepackUs.load(std::memory_order_relaxed) -
                    t.repackStageInUs.load(std::memory_order_relaxed) -
                    t.repackLoopCpuUs.load(std::memory_order_relaxed) -
                    t.repackSyncUs.load(std::memory_order_relaxed) -
                    t.repackPrologueUs.load(std::memory_order_relaxed) -
                    t.repackAllocUs.load(std::memory_order_relaxed) -
                    t.repackReleaseUs.load(std::memory_order_relaxed)),
        (long long)((t.tLoadMs.load(std::memory_order_relaxed) == 0 ||
                     t.tPrepareMs.load(std::memory_order_relaxed) == 0) ? -1 :
                    (long long)((t.tLoadMs.load(std::memory_order_relaxed) -
                                 t.tPrepareMs.load(std::memory_order_relaxed)) * 1000) -
                    (long long)t.bufferTotalUs.load(std::memory_order_relaxed) -
                    (long long)t.uploadUs.load(std::memory_order_relaxed)),
        (unsigned long long)t.workerWallUs.load(std::memory_order_relaxed),
        (unsigned long long)t.serialGroupsUs.load(std::memory_order_relaxed),
        (unsigned long long)t.parallelPartsUs.load(std::memory_order_relaxed),
        (unsigned long long)t.budgetWaitUs.load(std::memory_order_relaxed),
        (unsigned long long)t.allocForDataCalls.load(std::memory_order_relaxed),
        (unsigned long long)t.allocForDataUs.load(std::memory_order_relaxed),
        (unsigned long long)t.directMallocCalls.load(std::memory_order_relaxed),
        (unsigned long long)t.directMallocUs.load(std::memory_order_relaxed),
        (unsigned long long)t.directMallocInnerUs.load(std::memory_order_relaxed),
        (unsigned long long)t.directFreeCalls.load(std::memory_order_relaxed),
        (unsigned long long)t.directFreeUs.load(std::memory_order_relaxed),
        (unsigned long long)t.directFreeInnerUs.load(std::memory_order_relaxed),
        (unsigned long long)t.directFreeWsUs.load(std::memory_order_relaxed));
    if (LoadStageTimingEnabled()) {
        std::printf(
            "[Fastllm] load stage detail (enabled=1, scope=thread_sum): "
            "split_cpu_source=%llu split_cuda_source=%llu "
            "split_compact_row=%llu/%llu us split_compact_col=%llu/%llu us | "
            "dtype_parse=%llu us merge_process=%llu us locker_wait=%llu us.\n",
            (unsigned long long)t.splitCpuSourceCalls.load(std::memory_order_relaxed),
            (unsigned long long)t.splitCudaSourceCalls.load(std::memory_order_relaxed),
            (unsigned long long)t.splitCompactRowCalls.load(std::memory_order_relaxed),
            (unsigned long long)t.splitCompactRowCopyUs.load(std::memory_order_relaxed),
            (unsigned long long)t.splitCompactColCalls.load(std::memory_order_relaxed),
            (unsigned long long)t.splitCompactColCopyUs.load(std::memory_order_relaxed),
            (unsigned long long)t.dtypeParseUs.load(std::memory_order_relaxed),
            (unsigned long long)t.mergeProcessUs.load(std::memory_order_relaxed),
            (unsigned long long)t.lockerWaitUs.load(std::memory_order_relaxed));
    }
    std::fflush(stdout);
}

}  // namespace fastllm
