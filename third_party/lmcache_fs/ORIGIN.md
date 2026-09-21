# LMCache native filesystem connector

Source: https://github.com/LMCache/LMCache
Commit: `1a4b40b1d79b0e76244f127f96ee0982f8bd270f`
License: Apache-2.0 (see LICENSE).

These upstream files are copied without modifications. FastLLM's
`src/utils/disk_prefix_cache.cpp` adapter uses their batch I/O and completion APIs
without Torch, Python, CUDA or an LMCache server. The adapter supplies checksums,
durable checkpoint publication and quota accounting around this raw-byte transport.

SHA-256 of upstream files:

- `connector_base.h`: `0ede07813b0b93f52d4aa02425493afa7e9f3ec24215ac2da276bd6ed91cdb2b`
- `connector_interface.h`: `87485a59406a90ae8db24e44b2bdbde9ac4f381b6074a4917aeeb30e45a506c7`
- `connector_types.h`: `181d1f5f279b025662b1e34520a5ea55f3ac98307bb8158c7234489e1a9243b2`
- `event_notifier.h`: `2849cabae7e1d2576ae38180484ddd1971917f7d16af020b4b9b7b2d858ec142`
- `keys.h`: `651600048497e50fbf585731c298809267aa8b3e96b38e868f974445302e8b4b`
- `fs/connector.h`: `944b1d9b1d1cbf6700954ccd3e2ae6f96308b2a6713e41936b06acaa760767dc`
- `fs/connector.cpp`: `601ec1eef857e624b140187db5162eb44d129e8d44d3faf6be5024dcb47031d4`
