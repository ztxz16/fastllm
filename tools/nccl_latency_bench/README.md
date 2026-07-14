# Standalone NCCL latency benchmark

This benchmark calls NCCL directly from C++ in one process controlling two GPUs.
It reports three different latency definitions:

- `host enqueue`: CPU time spent submitting one grouped two-rank all-reduce.
- `CUDA event complete`: completion time observed on the CUDA streams.
- `host enqueue+wait`: host-visible latency when every collective is immediately
  synchronized. This includes CUDA/NCCL API and host scheduling overhead.
- `batched / operation`: many collectives are queued before one synchronization,
  which amortizes per-call host synchronization and is closest to the timing style
  commonly used by throughput-oriented collective benchmarks.

Build and run:

```bash
cd tools/nccl_latency_bench
make
NCCL_DEBUG=INFO ./nccl_latency_bench
```

Useful focused run for a 2048-element FP16 hidden state:

```bash
./nccl_latency_bench --bytes 4K --warmup 100 --iters 1000 --batch-iters 2000
```

Compare NCCL's automatic protocol selection with the low-latency protocol:

```bash
./nccl_latency_bench --bytes 4K
NCCL_PROTO=LL ./nccl_latency_bench --bytes 4K
```

To inspect the selected algorithm and protocol for a minimal run:

```bash
NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=COLL,TUNING \
  ./nccl_latency_bench --bytes 4K --warmup 1 --iters 1 --batch-iters 1
```

The benchmark prints CUDA peer-access capability. If it is disabled, NCCL may
select a shared-memory or network transport instead of direct GPU P2P, so results
should not be compared directly with an NVLink/P2P-enabled machine.

See [RESULTS.md](RESULTS.md) for measurements from the local dual-RTX-5090
machine that motivated this benchmark.
