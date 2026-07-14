# Local results: dual RTX 5090

Environment:

- CUDA runtime/driver API: 13.2
- NCCL: 2.30.4
- Payload: 4096 bytes (2048 FP16 elements)
- CUDA peer access: disabled in both directions
- NCCL transport: `SHM/direct/direct`
- Warmup: 100 operations
- Individually synchronized samples: 1000
- Batched operations: 3000

## Direct C++ NCCL

| Mode | Host enqueue mean | CUDA event, one op | Host enqueue + wait | Batched CUDA event/op |
|---|---:|---:|---:|---:|
| Auto (`RING + SIMPLE`) | 12.56 us | 28.24 us | 32.35 us | 16.45 us |
| `NCCL_PROTO=LL` | 12.78 us | 20.67 us | 24.75 us | 11.54 us |
| SHM disabled (`NET/Socket`) | 23.15 us | 83.45 us | 87.83 us | 57.49 us |

The automatic-protocol numbers are the mean of three complete runs. Their
batched CUDA-event results were 16.43, 16.50, and 16.42 us/op.

With `NCCL_DEBUG=TRACE`, automatic selection for 4096 bytes was reported as:

```text
AllReduce: 4096 Bytes -> Algo RING proto SIMPLE channel{Lo..Hi}={0..0}
```

## Why the earlier PyTorch result was about 129 us

The earlier test launched two Python processes and timed every operation as:

```python
begin = time.perf_counter_ns()
dist.all_reduce(tensor)
torch.cuda.synchronize()
```

That measured about 129 us/op. It includes Python and PyTorch dispatch, two
processes reaching each collective at slightly different times, CUDA runtime
submission, and a host synchronization after every operation. It is therefore
not a measurement of NCCL device-side latency alone.

For this machine, the directly measured steady-state NCCL value is about 16.5 us
with automatic protocol selection, or 11.5 us when forcing the LL protocol.
