// Architecture dispatch preserves the public EP ABI and binds every handle once.
#include "devices/cuda/fastllm-mimo-mega-backends.h"
#include <cuda_runtime.h>
#include <map>
extern "C" int fastllm_sm100_mega_moe_weight_view(void*,const void**,bool*);
#include <memory>
#include <string>

namespace {
struct Api {
    const char *name;
#define FIELD(n) decltype(&fastllm_ep8_mega_moe_##n) n;
    FIELD(error) FIELD(max_rows) FIELD(workspace_bytes) FIELD(init_workspace)
    FIELD(supported) FIELD(create) FIELD(run) FIELD(run_masked)
    FIELD(decode_capacity) FIELD(run_batched) FIELD(prefill_workspace_bytes)
    FIELD(prepare_prefill) FIELD(run_prefill) FIELD(destroy)
#undef FIELD
};
#define API(sm, label) {label, fastllm_##sm##_mega_moe_error, \
    fastllm_##sm##_mega_moe_max_rows, fastllm_##sm##_mega_moe_workspace_bytes, \
    fastllm_##sm##_mega_moe_init_workspace, fastllm_##sm##_mega_moe_supported, \
    fastllm_##sm##_mega_moe_create, fastllm_##sm##_mega_moe_run, \
    fastllm_##sm##_mega_moe_run_masked, fastllm_##sm##_mega_moe_decode_capacity, \
    fastllm_##sm##_mega_moe_run_batched, fastllm_##sm##_mega_moe_prefill_workspace_bytes, \
    fastllm_##sm##_mega_moe_prepare_prefill, fastllm_##sm##_mega_moe_run_prefill, \
    fastllm_##sm##_mega_moe_destroy}
#ifdef FASTLLM_MIMO_MEGA_SM90
const Api hopper = API(sm90, "sm90-native-fp8-w8a8");
#endif
#ifdef FASTLLM_MIMO_MEGA_SM100
const Api blackwell = API(sm100, "sm10x-fp8-mxfp4-ue8m0");
#endif
#undef API
thread_local const Api *lastApi = nullptr;
thread_local std::string error;
struct Handle { const Api *api; void *inner; };
const Api *select() {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        lastApi = nullptr; error = "Cannot select MegaMoE CUDA device"; return nullptr;
    }
    static thread_local std::map<int, const Api *> cache;
    auto it = cache.find(device);
    if (it != cache.end()) return lastApi = it->second;
    cudaDeviceProp prop{};
    const Api *api = nullptr;
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
#ifdef FASTLLM_MIMO_MEGA_SM90
        if (prop.major == 9 && prop.minor == 0) api = &hopper;
#endif
#ifdef FASTLLM_MIMO_MEGA_SM100
        if (prop.major == 10 && (prop.minor == 0 || prop.minor == 3)) api = &blackwell;
#endif
    }
    cache.emplace(device, api);
    if (!api) error = "No compiled MegaMoE backend supports this architecture";
    return lastApi = api;
}
Handle *handle(void *ptr) {
    if (!ptr) { lastApi = nullptr; error = "Missing MegaMoE handle"; return nullptr; }
    auto *h = static_cast<Handle *>(ptr); lastApi = h->api; return h;
}
}
extern "C" const char *fastllm_ep8_mega_moe_backend_name() {
    const auto *a = select(); return a ? a->name : "unavailable";
}
extern "C" const char *fastllm_ep8_mega_moe_error() { return lastApi ? lastApi->error() : error.c_str(); }
extern "C" int fastllm_ep8_mega_moe_supported() { const auto *a = select(); return a && a->supported(); }
extern "C" int fastllm_ep8_mega_moe_max_rows() { const auto *a = select(); return a ? a->max_rows() : 0; }
extern "C" size_t fastllm_ep8_mega_moe_workspace_bytes(int ranks) {
    const auto *a = select(); return a ? a->workspace_bytes(ranks) : 0;
}
extern "C" int fastllm_ep8_mega_moe_init_workspace(void *w, size_t b, const int *d, int r, int n) {
    const auto *a = select(); return a ? a->init_workspace(w,b,d,r,n) : 0;
}
extern "C" void *fastllm_ep8_mega_moe_create(const void *const *g, const void *const *d,
    const void *const *gs, const void *const *ds, void *w, size_t bytes,
    const void *const *p, int r, int slot, int n) {
    const auto *a = select(); if (!a) return nullptr;
    auto h = std::make_unique<Handle>();
    h->api = a; h->inner = a->create(g,d,gs,ds,w,bytes,p,r,slot,n);
    return h->inner ? h.release() : nullptr;
}
extern "C" void *fastllm_ep8_mega_moe_create_mxfp4(const void *const *g, const void *const *d,
    void *w, size_t bytes, const void *const *p, int r, int slot, int n) {
#ifdef FASTLLM_MIMO_MEGA_SM100
    if (select() == &blackwell) {
        auto h = std::make_unique<Handle>();
        h->api = &blackwell;
        h->inner = fastllm_sm100_mega_moe_create_mxfp4(g,d,w,bytes,p,r,slot,n);
        return h->inner ? h.release() : nullptr;
    }
#endif
    lastApi = nullptr; error = "Native MXFP4 MegaMoE requires Blackwell"; return nullptr;
}
extern "C" int fastllm_ep8_mega_moe_decode_capacity(int rows, int ranks) {
    const auto *a = select(); return a ? a->decode_capacity(rows,ranks) : 0;
}
extern "C" int fastllm_ep8_mega_moe_run(void *p, const void *x, void *y, const int32_t *i, const float *s, int rows) {
    auto *h = handle(p); return h ? h->api->run(h->inner,x,y,i,s,rows) : 0;
}
extern "C" int fastllm_ep8_mega_moe_run_masked(void *p, const void *x, void *y, const int32_t *i, const float *s, int rows, const int *live) {
    auto *h = handle(p); return h ? h->api->run_masked(h->inner,x,y,i,s,rows,live) : 0;
}
extern "C" int fastllm_ep8_mega_moe_run_batched(void *p, const void *x, void *y, const int32_t *i, const float *s, int rows, int capacity, const int *live) {
    auto *h = handle(p); return h ? h->api->run_batched(h->inner,x,y,i,s,rows,capacity,live) : 0;
}
extern "C" int fastllm_ep8_mega_moe_run_verify(void *p, const void *x, void *y,
        const int32_t *i, const float *s, int rows, int capacity, const int *live) {
    auto *h = handle(p);
    if (!h) return 0;
#ifdef FASTLLM_MIMO_MEGA_SM100
    if (h->api == &blackwell)
        return fastllm_sm100_mega_moe_run_verify(h->inner,x,y,i,s,rows,capacity,live);
#endif
    return h->api->run_batched(h->inner,x,y,i,s,rows,capacity,live);
}
extern "C" size_t fastllm_ep8_mega_moe_prefill_workspace_bytes(int rows, int ranks) {
    const auto *a = select(); return a ? a->prefill_workspace_bytes(rows,ranks) : 0;
}
extern "C" int fastllm_ep8_mega_moe_prepare_prefill(void *p, void *w, size_t b, int rows) {
    auto *h = handle(p); return h ? h->api->prepare_prefill(h->inner,w,b,rows) : -1;
}
extern "C" int fastllm_ep8_mega_moe_run_prefill(void *p, const void *const *peers, const void *x, void *y, const int32_t *i, const float *s, int rows) {
    auto *h = handle(p); return h ? h->api->run_prefill(h->inner,peers,x,y,i,s,rows) : 0;
}
extern "C" void fastllm_ep8_mega_moe_destroy(void *p) {
    if (!p) return;
    std::unique_ptr<Handle> h(handle(p)); h->api->destroy(h->inner);
}

extern "C" int fastllm_ep8_mega_moe_weight_view(void*p,const void**out,bool *mxfp4){
#ifdef FASTLLM_MIMO_MEGA_SM100
 auto*h=handle(p);if(h&&h->api==&blackwell)return fastllm_sm100_mega_moe_weight_view(h->inner,out,mxfp4);
#endif
 return 0;
}

extern "C" int fastllm_ep8_mega_moe_weights(void *p, const void **out) {
    return fastllm_ep8_mega_moe_weight_view(p,out,nullptr);
}
extern "C" int fastllm_ep8_mega_moe_owns_packed_weights() {
#ifdef FASTLLM_MIMO_MEGA_SM100
    return select()==&blackwell;
#else
    return 0;
#endif
}
extern "C" int fastllm_ep8_mega_moe_group_supported(const int*devices,int ranks) {
    if(!devices||(ranks!=4&&ranks!=8))return 0;
    for(int r=0;r<ranks;++r)for(int q=0;q<ranks;++q)if(q!=r){
        int access=0;
        if(devices[r]==devices[q]||cudaDeviceCanAccessPeer(&access,devices[r],devices[q])!=cudaSuccess||!access){
            cudaGetLastError();return 0;
        }
    }
    return 1;
}

// The SM90 MXFP4 adapter is compiled only when that architecture is requested.
// Keep the shared built-in dispatch linkable in Blackwell-only builds.
#ifndef FASTLLM_MIMO_MEGA_SM90
#include "devices/cuda/fastllm-mxfp4-ep.h"
extern "C" const char *fastllm_mxfp4_ep_error() { return "SM90 MXFP4 EP was not built"; }
extern "C" int fastllm_mxfp4_ep_capacity() { return 0; }
extern "C" int fastllm_mxfp4_ep_prefill_capacity() { return 0; }
extern "C" size_t fastllm_mxfp4_ep_workspace_bytes(int) { return 0; }
extern "C" int fastllm_mxfp4_ep_init(void *, size_t, const int *, int, int) { return 0; }
extern "C" size_t fastllm_mxfp4_ep_draft_workspace(int, int *prefillCapacity) {
    if (prefillCapacity) *prefillCapacity = 0;
    return 0;
}
extern "C" void *fastllm_mxfp4_ep_create(const void *const *, const void *const *, void *, size_t,
    const void *const *, int, int) { return nullptr; }
extern "C" void *fastllm_mxfp4_ep_create_shape(const void *const *, const void *const *, void *, size_t,
    const void *const *, int, int, int, int) { return nullptr; }
extern "C" int fastllm_mxfp4_ep_set_map(void *, const uint8_t *) { return 0; }
extern "C" int fastllm_mxfp4_ep_run(void *, const void *, void *, const int32_t *, const float *, int, int, const int *) { return 0; }
extern "C" void fastllm_mxfp4_ep_destroy(void *) {}
#endif
