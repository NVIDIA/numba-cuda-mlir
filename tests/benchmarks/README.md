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
Attention                    | 602.50                       | 687.16                            | 0.88x                | 146.19                       | 70.32                             | 2.08x                | 10.2776                | 11.5557                     | 0.89x
Blackscholes                 | 539.22                       | 786.59                            | 0.69x                | 111.90                       | 69.40                             | 1.61x                | 0.0223                 | 0.0237                      | 0.94x
Cholesky                     | 681.36                       | 740.55                            | 0.92x                | 146.72                       | 86.61                             | 1.69x                | 33.9785                | 32.9067                     | 1.03x
Cholesky Blocked             | 754.24                       | 831.22                            | 0.91x                | 369.29                       | 164.75                            | 2.24x                | 4.4017                 | 4.9077                      | 0.90x
Fft                          | 553.71                       | 94.42                             | 5.86x                | 288.48                       | 82.65                             | 3.49x                | 0.0637                 | 0.0581                      | 1.10x
Matmul Smem                  | 530.24                       | 739.77                            | 0.72x                | 104.89                       | 69.57                             | 1.51x                | 0.1548                 | 0.1545                      | 1.00x
Softmax                      | 546.75                       | 709.99                            | 0.77x                | 117.58                       | 71.78                             | 1.64x                | 0.0058                 | 0.0046                      | 1.27x
Softmax Large                | 520.67                       | 742.32                            | 0.70x                | 118.17                       | 67.83                             | 1.74x                | 2.6443                 | 4.4679                      | 0.59x
Vector Addition (scalar)     | 431.75                       | 625.48                            | 0.69x                | 27.02                        | 23.00                             | 1.17x                | 0.0734                 | 0.0812                      | 0.90x
Vector Addition (vectorized) | 484.02                       | 683.08                            | 0.71x                | 69.34                        | 41.94                             | 1.65x                | 0.0687                 | 0.0694                      | 0.99x
GEOMEAN                      | 557.80                       | 591.39                            | 0.94x                | 121.45                       | 67.21                             | 1.81x                | 0.3481                 | 0.3682                      | 0.95x
====================================================================================================
```
