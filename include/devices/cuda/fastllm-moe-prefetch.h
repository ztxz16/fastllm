#pragma once
namespace fastllm { class Data; }
// Per-device bounded lookahead of previously streamed, registered GGUF experts.
bool FastllmCudaPrefetchMoeGGUFHost(fastllm::Data **weights, int expertCount);
// Order a host-input readback after outstanding expert uploads.
void FastllmCudaWaitMoeGGUFHostPrefetch(void *stream);
void FastllmCudaClearMoeGGUFHostPrefetch();
