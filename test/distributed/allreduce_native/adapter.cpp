#include "distributed/process_allreduce.h"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string lastError;

struct Handle {
    cudaStream_t stream = nullptr;
    ncclComm_t comm = nullptr;
    std::unique_ptr<fastllm::distributed::ProcessAllReduce> ft;
};

/** 将 CUDA 错误转换为由 Python 协调器处理的异常。 */
void CheckCuda(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

/** 将 NCCL 错误转换为异常；失败后由外层终止整个测试组。 */
void CheckNccl(ncclResult_t result) {
    if (result != ncclSuccess) throw std::runtime_error(ncclGetErrorString(result));
}

/** 捕获 C++ 异常，防止其跨越 ctypes 的 C ABI 边界。 */
template<class Function> int Guard(Function function) {
    try {
        lastError.clear();
        function();
        return 0;
    } catch (const std::exception &error) {
        lastError = error.what();
    } catch (...) {
        lastError = "unknown native AllReduce error";
    }
    return -1;
}
}

extern "C" {
/** 返回最近一次当前线程调用的错误描述；下一次调用之前有效。 */
const char *ar_error() { return lastError.c_str(); }

/** 返回实际编译架构列表，供报告记录，不代替设备执行检查。 */
const char *ar_build_arch() { return AR_BUILD_ARCH; }

/** 返回编译时生产 CUDA 源码的 SHA256，识别未重新构建的旧产物。 */
const char *ar_source_sha() { return AR_SOURCE_SHA; }

/** 返回编译时生产接口头文件的 SHA256。 */
const char *ar_header_sha() { return AR_HEADER_SHA; }

/** 返回配置构建时的 Git 提交；源码指纹用于校验包含未提交修改的构建。 */
const char *ar_build_git() { return AR_BUILD_GIT; }

/** 返回当前 NCCL ABI 所需的 unique ID 缓冲字节数。 */
size_t ar_id_size() { return sizeof(ncclUniqueId); }

/** 创建组内共享的 NCCL unique ID；由 rank 0 经 CPU 控制组广播。 */
int ar_unique_id(void *output) {
    return Guard([&] {
        if (!output) throw std::invalid_argument("null unique ID output");
        ncclUniqueId id;
        CheckNccl(ncclGetUniqueId(&id));
        std::memcpy(output, &id, sizeof(id));
    });
}

/**
 * 在调用方提供的生产 CUDA stream 上创建通信资源。
 *
 * 不创建 tensor、模型或额外 stream；FT 路径直接使用原实现的能力检查与生命周期。
 * 部分初始化失败后不在单 rank 发起集体清理，由协调器终止整组并回收进程。
 * @param device CUDA_VISIBLE_DEVICES 下的逻辑设备编号。
 * @param world 通信组成员数，FT 自定义路径要求为 2。
 * @param rank 本进程在组内的编号，范围为 [0, world)。
 * @param mode 0 为原生 NCCL 基线，1 为 FT 自定义开启，2 为 FT 自定义关闭。
 * @param id 组内一致的 NCCL unique ID 字节。
 * @param stream 调用方当前设备的 CUDA stream，必须保持到销毁完成。
 * @param output 成功时取得句柄；失败时保持空指针。
 * @return 0 成功；-1 失败，具体原因通过 ar_error 取得。
 */
int ar_create(int device, int world, int rank, int mode, const void *id,
              void *stream, void **output) {
    return Guard([&] {
        if (!output || !id || world < 1 || rank < 0 || rank >= world || mode < 0 || mode > 2)
            throw std::invalid_argument("invalid AllReduce initialization arguments");
        *output = nullptr;
        CheckCuda(cudaSetDevice(device));
        auto handle = std::make_unique<Handle>();
        handle->stream = reinterpret_cast<cudaStream_t>(stream);
        ncclUniqueId unique;
        std::memcpy(&unique, id, sizeof(unique));
        CheckNccl(ncclCommInitRank(&handle->comm, world, unique, rank));
        if (mode != 0) {
            handle->ft = std::make_unique<fastllm::distributed::ProcessAllReduce>(
                handle->stream, handle->comm, mode == 1);
        }
        *output = handle.release();
    });
}

/**
 * 提交原地 FP32[count] SUM；不增加复制、同步或绕过正式算子的门槛。
 * @param opaque ar_create 返回的句柄。
 * @param data 当前设备连续 FP32 缓冲，大小至少 count 个元素。
 * @param count 正整数；零长度明确拒绝。
 * @return 0 表示提交成功；异步失败由后续同步发现，调用错误通过 ar_error 取得。
 */
int ar_sum(void *opaque, void *data, size_t count) {
    return Guard([&] {
        if (!opaque || !data || count == 0) throw std::invalid_argument("invalid AllReduce input");
        auto &handle = *static_cast<Handle*>(opaque);
        if (handle.ft) {
            handle.ft->Sum(static_cast<float*>(data), count);
        } else {
            CheckNccl(ncclAllReduce(data, data, count, ncclFloat, ncclSum, handle.comm, handle.stream));
        }
    });
}

/** 读取 FT 的 GPU 执行计数；必须位于计时和 Graph capture 之外。 */
int ar_info(void *opaque, unsigned long long *custom, unsigned long long *nccl) {
    return Guard([&] {
        if (!opaque || !custom || !nccl) throw std::invalid_argument("invalid info arguments");
        auto &handle = *static_cast<Handle*>(opaque);
        if (!handle.ft) throw std::invalid_argument("raw NCCL has no FT GPU counters");
        auto info = handle.ft->Info();
        *custom = info.customCalls;
        *nccl = info.ncclCalls;
    });
}

/** 正常情况下所有 rank 共同关闭；故障路径不调用此集体清理接口。 */
int ar_destroy(void *opaque) {
    return Guard([&] {
        if (!opaque) return;
        auto &handle = *static_cast<Handle*>(opaque);
        CheckCuda(cudaStreamSynchronize(handle.stream));
        if (handle.ft) handle.ft->Close();
        CheckNccl(ncclCommDestroy(handle.comm));
        delete &handle;
    });
}

/** 读取实际加载的 CUDA runtime 与 NCCL 版本，核对三方环境一致性。 */
int ar_versions(int *runtime, int *nccl) {
    return Guard([&] {
        if (!runtime || !nccl) throw std::invalid_argument("null version output");
        CheckCuda(cudaRuntimeGetVersion(runtime));
        CheckNccl(ncclGetVersion(nccl));
    });
}
}
