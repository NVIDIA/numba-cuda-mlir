# numba-cuda-mlir Benchmarks

Benchmarks to compare JIT compile-time and kernel performance between Numba CUDA and numba-cuda-mlir implementations. Uses NVIDIA Nsight Compute (NCU) to profile kernel execution times.

## Quick Start

```bash
# Run correctness tests
pytest tests/benchmarks/

# Run performance benchmarks (requires NCU)
pytest tests/benchmarks/ --benchmark -s
```

## Available Benchmarks

| Benchmark              | Description              | Key Features                            |
|------------------------|--------------------------|------------------------------------------|
| `vector_add/`          | Vector addition          | Scalar & vectorized versions             |
| `softmax/`             | Softmax normalization    | Numerically stable 3-phase reduction     |
| `cholesky/`            | Cholesky factorization   | Blocked & unblocked algorithms           |
| `attention/`           | Self-attention           | Dynamic shared memory                    |
| `blackscholes/`        | Option pricing           | Transcendental functions                 |
| `fft/`                 | Fast Fourier Transform   | Radix-2, bit-reversal                    |
| `test_matmul_smem.py`  | Matrix multiplication    | Shared memory tiling                     |

## Usage

### Three ways to run benchmarks:

1. **Correctness only**: `pytest tests/benchmarks/vector_add/`
2. **With profiling scripts**: `pytest tests/benchmarks/vector_add/ --benchmark -s`
3. **Direct execution**: `python tests/benchmarks/vector_add/test_vector_addition.py scalar --compile-mode warm`

### Compile modes

Standalone benchmark scripts accept `--compile-mode {cold,warm}`:

- `cold` measures compilation in a fresh subprocess without benchmark-side warmup.
- `warm` first compiles a trivial kernel through both backends, then times the benchmark kernel compilation. This removes one-time initialization costs from the measured compile time.

The pytest benchmark runner invokes each script once per compile mode and reports both results in the consolidated table.

## Output

Running benchmarks using pytest with `--benchmark -s` produces:

Machine: AMD Ryzen 9 9950X | NVIDIA RTX PRO 6000 Blackwell | CUDA driver 580.95 | Python 3.12 | Ubuntu 24.04 | CUDA Toolkit 13.0

```
====================================================================================================
BENCHMARK RESULTS SUMMARY
====================================================================================================
Benchmark                    | Numba-CUDA Cold Compile (ms) | numba-cuda-mlir Cold Compile (ms) | Cold Compile Speedup | Numba-CUDA Warm Compile (ms) | numba-cuda-mlir Warm Compile (ms) | Warm Compile Speedup | Numba-CUDA Kernel (ms) | numba-cuda-mlir Kernel (ms) | Kernel Speedup
-----------------------------+------------------------------+-----------------------------------+----------------------+------------------------------+-----------------------------------+----------------------+------------------------+-----------------------------+---------------
Attention                    | 354.84                       | 333.05                            | 1.07x                | 89.53                        | 48.23                             | 1.86x                | 34.5719                | 35.7985                     | 0.97x
Blackscholes                 | 248.04                       | 356.83                            | 0.70x                | 55.85                        | 55.80                             | 1.00x                | 0.5028                 | 0.5019                      | 1.00x
Cholesky                     | 260.55                       | 362.27                            | 0.72x                | 93.78                        | 78.73                             | 1.19x                | 40.5486                | 40.8926                     | 0.99x
Cholesky Blocked             | 488.24                       | 419.72                            | 1.16x                | 178.33                       | 130.23                            | 1.37x                | 8.9310                 | 9.3268                      | 0.96x
Fft                          | 262.19                       | 64.29                             | 4.08x                | 120.23                       | 62.33                             | 1.93x                | 0.0788                 | 0.0817                      | 0.96x
Matmul Smem                  | 238.35                       | 355.13                            | 0.67x                | 67.18                        | 49.12                             | 1.37x                | 0.9163                 | 0.9332                      | 0.98x
Softmax                      | 255.60                       | 349.96                            | 0.73x                | 74.67                        | 49.29                             | 1.51x                | 0.0082                 | 0.0068                      | 1.21x
Softmax Large                | 362.57                       | 368.74                            | 0.98x                | 74.83                        | 47.70                             | 1.57x                | 4.7444                 | 6.1678                      | 0.77x
Vector Addition (scalar)     | 195.31                       | 317.99                            | 0.61x                | 21.47                        | 16.61                             | 1.29x                | 0.1614                 | 0.1597                      | 1.01x
Vector Addition (vectorized) | 212.06                       | 339.67                            | 0.62x                | 44.74                        | 34.34                             | 1.30x                | 0.1724                 | 0.1763                      | 0.98x
GEOMEAN                      | 277.27                       | 299.21                            | 0.93x                | 71.77                        | 50.75                             | 1.41x                | 0.9318                 | 0.9526                      | 0.98x
====================================================================================================
```
