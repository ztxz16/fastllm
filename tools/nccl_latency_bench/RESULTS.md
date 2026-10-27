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
