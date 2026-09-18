#pragma once
#include "fastllm.h"
namespace fastllm {
// Optional activation-quantized prefill, with complete Linear fallback.
bool FastllmCudaNativeFp8FusedCanRun(const Data &, const Data &, const Data &, const Data &, bool);
bool FastllmCudaNativeFp8Fused(Data &, Data &, Data &, bool);

bool FastllmCudaNativeNvfp4FusedCanRun(const Data &, const Data &, const Data &, const Data &, bool);
bool FastllmCudaNativeNvfp4Fused(Data &, Data &, Data &, bool);

bool FastllmCudaTryNativeNvfp4Linear(const Data &, Data &, const Data &, Data &, int, int, int);
bool FastllmCudaNativeNvfp4LayoutFusedCanRun(const Data &, const Data &, const Data &, const Data &, bool);
bool FastllmCudaNativeNvfp4LayoutFused(Data &, Data &, Data &, bool);
void FastllmCudaRestoreNativeNvfp4(Data &);
} // namespace fastllm
