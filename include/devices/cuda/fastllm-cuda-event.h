#ifndef FASTLLM_CUDA_EVENT_H
#define FASTLLM_CUDA_EVENT_H

// Publish current-thread work through an explicit source-device stream before
// other devices wait on event. All three objects belong to the source device;
// callers serialize submissions and retain them until consumers finish.
void FastllmCudaEventRecordForPeer(void *event, void *sourceEvent, void *relayStream);

#endif
