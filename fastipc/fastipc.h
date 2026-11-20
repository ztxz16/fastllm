#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FastIpcRouter FastIpcRouter;
FastIpcRouter *fastipc_router_create(const int *capacities, int count, int queue, int sessions,
                                     double idle_ttl_seconds);
void fastipc_router_destroy(FastIpcRouter *router);
// Reserve an active or queued request without blocking. Release once with the same key.
int fastipc_reserve(FastIpcRouter *router, const char *session);
void fastipc_release(FastIpcRouter *router, int rank, const char *session);
int fastipc_forget(FastIpcRouter *router, const char *session);
void fastipc_rank_offline(FastIpcRouter *router, int rank);
void fastipc_rank_state(FastIpcRouter *router, int rank, int values[2]);

// A size-sealed memfd. The caller owns the returned descriptor.
int fastipc_create_buffer(size_t bytes);
#ifdef __cplusplus
}
#endif
