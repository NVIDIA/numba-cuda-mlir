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

```
--------------------------------------------------------------------------------------
Benchmark                |  numba_cuda (ns) |  numba_cuda_mlir (ns) |    Speedup
--------------------------------------------------------------------------------------
launch_empty             |           4101.9 |                3871.2 |      1.06x
launch_1_array_arg       |           5839.0 |                2962.3 |      1.97x
launch_16_scalar_args    |          15204.4 |                4306.9 |      3.53x
launch_16_array_args     |          29780.4 |               12370.7 |      2.41x
launch_256_scalar_args   |         177007.1 |               10108.6 |     17.51x
--------------------------------------------------------------------------------------
```
