# Kernel Launch Latency Microbenchmark

Measures host-side kernel dispatch overhead — from argument packing through `cuLaunchKernel()` return — comparing numba-cuda vs numba-cuda-mlir.

## Kernels

| Kernel | Args | Purpose |
|--------|------|---------|
| `empty` | 0 | No-op; minimum dispatch overhead |
| `1_array_arg` | 1 array | Single `float32[::1]` arg; measures array dispatch overhead |
| `16_scalar_args` | 16 scalars | Isolates scalar parameter packing cost |
| `16_array_args` | 16 arrays | Isolates array parameter packing cost |
| `256_scalar_args` | 256 scalars | Stress-tests parameter packing |

All kernels use `grid=1, block=1` so GPU execution time is negligible. Kernels are pre-compiled and warmed up before the timed loop.

## Usage

```bash
python tests/benchmarks/launch_latency_ubench/launch_latency_ubench.py
```

## Output

Machine: AMD Ryzen 9 9950X | NVIDIA RTX PRO 6000 Blackwell | CUDA driver 580.95 | Python 3.12 | Ubuntu 24.04 | CUDA Toolkit 13.0

```
--------------------------------------------------------------------------------------
Benchmark                |  numba_cuda (ns) |  numba_cuda_mlir (ns) |    Speedup
--------------------------------------------------------------------------------------
launch_empty             |           4089.4 |                3862.4 |      1.06x
launch_1_array_arg       |           5865.4 |                2983.8 |      1.97x
launch_16_scalar_args    |          15489.9 |                4355.2 |      3.56x
launch_16_array_args     |          30487.5 |               12508.1 |      2.44x
launch_256_scalar_args   |         179437.1 |               10162.5 |     17.66x
--------------------------------------------------------------------------------------
```
