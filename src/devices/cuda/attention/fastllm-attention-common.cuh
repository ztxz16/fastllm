#pragma once
//
// Created by huangyuyang on 1/21/26.
//

// FlashInfer 依赖 cuda/std/barrier 等同步原语（要求 sm_70+），其头文件在 sm_70 以下的
// device 编译 pass（如 sm_60）中会直接 #error。为兼容多 CUDA_ARCH 一起编译，这里仅在
// host pass 与 sm_70+ 的 device pass 中启用 FlashInfer；sm_70 以下的 device pass 完全
// 排除 FlashInfer 代码，运行期由原生分页注意力兜底。
#if !defined(FASTLLM_CUDA_LEGACY_ONLY) && \
    (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700))
#define FASTLLM_ENABLE_FLASHINFER
#endif

// FlashInfer includes
#ifdef FASTLLM_ENABLE_FLASHINFER
#ifndef FLASHINFER_ENABLE_FP4_E2M1
#define FLASHINFER_ENABLE_FP4_E2M1
#endif
#include "attention_impl.cuh"
#include "attention/default_prefill_params.cuh"
#include "attention/variants.cuh"
#include "attention/mask.cuh"
#include "attention/scheduler.cuh"
#include "attention/mla.cuh"
#include "fastdiv.cuh"
#include "pos_enc.cuh"
#include "utils.cuh"
#endif

#include "fastllm-cuda.cuh"
#include "fastllm.h"
#include "utils/utils.h"
#include "attention/fastllm-attention-dtype.cuh"
#include "attention/fastllm-fp4-kv.cuh"
#include "attention/fastllm-paged-attention-native.cuh"

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cuda_fp8.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
